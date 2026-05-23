//! 32-byte AVX2 scanner, same shape as find_pattern_simd128 with bigger registers

use std::arch::x86_64::*;

use crate::scalar;
use crate::SimdPatternScanData;

const AVX_REGISTER_LENGTH: usize = 32;

#[target_feature(enable = "avx2")]
unsafe fn vector256_set1(v: u8) -> __m256i {
    _mm256_set1_epi8(v as i8)
}

#[target_feature(enable = "avx2")]
unsafe fn vector256_load(p: *const u8) -> __m256i {
    _mm256_loadu_si256(p as *const __m256i)
}

#[target_feature(enable = "avx2")]
unsafe fn vector256_cmpeq(a: __m256i, b: __m256i) -> __m256i {
    _mm256_cmpeq_epi8(a, b)
}

#[target_feature(enable = "avx2")]
unsafe fn vector256_movemask(v: __m256i) -> u32 {
    _mm256_movemask_epi8(v) as u32
}

#[target_feature(enable = "avx2")]
unsafe fn any_byte_set_4(a: __m256i, b: __m256i, c: __m256i, d: __m256i) -> bool {
    let ab = _mm256_or_si256(a, b);
    let cd = _mm256_or_si256(c, d);
    _mm256_movemask_epi8(_mm256_or_si256(ab, cd)) != 0
}

fn pattern_to_vec256(pattern: &SimdPatternScanData) -> Vec<[u8; AVX_REGISTER_LENGTH]> {
    let mut len = pattern.mask.len();
    if len <= 1 {
        return Vec::new();
    }
    let pattern_bytes = &pattern.bytes[1..];
    len -= 1;
    let vector_count = len.div_ceil(AVX_REGISTER_LENGTH);

    let mut vecs = Vec::with_capacity(vector_count);
    for i in 0..vector_count {
        let base = i * AVX_REGISTER_LENGTH;
        let mut buf = [0u8; AVX_REGISTER_LENGTH];
        let take = std::cmp::min(AVX_REGISTER_LENGTH, len - base);
        buf[..take].copy_from_slice(&pattern_bytes[base..base + take]);
        vecs.push(buf);
    }
    vecs
}

pub unsafe fn find_pattern_avx2(data: &[u8], pattern: &SimdPatternScanData) -> Option<usize> {
    let data_len = data.len();
    if pattern.bytes.is_empty() || data_len < pattern.bytes.len() {
        return None;
    }
    let vector_count = pattern.mask.len().saturating_sub(1).div_ceil(AVX_REGISTER_LENGTH).max(1);
    if data_len < (vector_count + 1) * AVX_REGISTER_LENGTH + 1 {
        return scalar::find_pattern_in(data, pattern);
    }

    let mut match_table = crate::build_match_indexes(pattern);
    // trim trailing zero entries, see find_pattern_simd128 for why
    let valid_match_count = pattern.mask.iter().skip(1).filter(|&&m| m != 0).count();
    match_table.truncate(valid_match_count);
    let pattern_vecs_bytes = pattern_to_vec256(pattern);
    let pattern_vecs: Vec<__m256i> = pattern_vecs_bytes
        .iter()
        .map(|b| vector256_load(b.as_ptr()))
        .collect();
    let match_table_len = match_table.len();

    let first_byte_vec = vector256_set1(pattern.bytes[pattern.leading_ignore_count]);
    let leading_ignore_count = pattern.leading_ignore_count;
    let data_base = data.as_ptr() as usize;
    // same safety bound as the 128-bit path, scaled to 32-byte register
    let safe_len = data_len - (vector_count + 1) * AVX_REGISTER_LENGTH;
    let data_ptr_max = data_base + safe_len;
    let mut data_ptr = data_base;
    let data_end = data_base + data_len;

    'data: while data_ptr < data_ptr_max {
        // 4x skim, 128 bytes per iter, see find_pattern_simd128 for the rationale
        while data_ptr + 4 * AVX_REGISTER_LENGTH <= data_end {
            let v0 = vector256_load(data_ptr as *const u8);
            let v1 = vector256_load((data_ptr + AVX_REGISTER_LENGTH) as *const u8);
            let v2 = vector256_load((data_ptr + 2 * AVX_REGISTER_LENGTH) as *const u8);
            let v3 = vector256_load((data_ptr + 3 * AVX_REGISTER_LENGTH) as *const u8);
            let e0 = vector256_cmpeq(first_byte_vec, v0);
            let e1 = vector256_cmpeq(first_byte_vec, v1);
            let e2 = vector256_cmpeq(first_byte_vec, v2);
            let e3 = vector256_cmpeq(first_byte_vec, v3);
            if any_byte_set_4(e0, e1, e2, e3) {
                break;
            }
            data_ptr += 4 * AVX_REGISTER_LENGTH;
        }
        if data_ptr >= data_ptr_max {
            break;
        }

        let rhs = vector256_load(data_ptr as *const u8);
        let equal = vector256_cmpeq(first_byte_vec, rhs);
        let mut find_first_byte = vector256_movemask(equal);

        // mask out hits that'd anchor before data_ptr, see the 128-bit path
        if leading_ignore_count > 0 {
            find_first_byte &= !((1u32 << leading_ignore_count) - 1);
        }

        if find_first_byte == 0 {
            data_ptr += AVX_REGISTER_LENGTH;
            continue;
        }

        let trailing = find_first_byte.trailing_zeros() as usize;
        data_ptr = data_ptr + trailing - leading_ignore_count;
        if data_ptr > data_ptr_max {
            break;
        }

        let mut match_table_index = 0;
        for (i, cur_pattern_vec) in pattern_vecs.iter().enumerate() {
            let register_byte_offs = i * AVX_REGISTER_LENGTH;
            let next_byte = data_ptr + register_byte_offs + 1;
            let rhs_2 = vector256_load(next_byte as *const u8);
            let compare_result = vector256_movemask(vector256_cmpeq(*cur_pattern_vec, rhs_2));

            while match_table_index < match_table_len {
                let match_index = std::num::Wrapping(match_table[match_table_index] as usize)
                    - std::num::Wrapping(register_byte_offs);
                if match_index.0 < AVX_REGISTER_LENGTH {
                    if ((compare_result >> match_index.0) & 1) != 1 {
                        data_ptr += 1;
                        continue 'data;
                    } else {
                        match_table_index += 1;
                        continue;
                    }
                }
                break;
            }
        }

        return Some(data_ptr - data_base);
    }

    scalar::find_pattern_from(data, pattern, safe_len)
}
