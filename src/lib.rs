mod scalar;

#[cfg(target_arch = "aarch64")]
#[path = "imp/aarch64.rs"]
mod imp;

#[cfg(target_arch = "x86_64")]
#[path = "imp/x86.rs"]
mod imp;

#[cfg(target_arch = "x86_64")]
#[path = "imp/x86_avx2.rs"]
mod imp_avx2;

mod dispatch;

pub use scalar::find_pattern_scalar;

const NEON_REGISTER_LENGTH: usize = 16;

pub fn get_offset_neon(data: &[u8], pattern: &str) -> Option<usize> {
    find_pattern_neon(data.as_ptr(), data.len(), pattern)
}

pub fn find_pattern_neon<S: AsRef<str>>(data: *const u8, data_len: usize, pattern: S) -> Option<usize> {
    let pattern = SimdPatternScanData::new(&pattern);
    if pattern.bytes.is_empty() || data_len < pattern.bytes.len() {
        return None;
    }
    let data_slice = unsafe { std::slice::from_raw_parts(data, data_len) };
    find_pattern_in(data_slice, &pattern)
}

pub fn find_pattern(data: &[u8], pattern: &str) -> Option<usize> {
    dispatch::find(data, pattern)
}

pub fn get_offset(data: &[u8], pattern: &str) -> Option<usize> {
    find_pattern(data, pattern)
}

pub(crate) fn find_pattern_in(data: &[u8], pattern: &SimdPatternScanData) -> Option<usize> {
    if pattern.bytes.is_empty() || data.len() < pattern.bytes.len() {
        return None;
    }
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        let vec_count = pattern_vec_count(pattern);
        // need enough headroom so inner load never reads past the end, see find_pattern_simd128
        if data.len() >= (vec_count + 1) * NEON_REGISTER_LENGTH + 1 {
            return find_pattern_simd128(data, pattern);
        }
    }
    scalar::find_pattern_in(data, pattern)
}

#[inline]
fn pattern_vec_count(pattern: &SimdPatternScanData) -> usize {
    let mask_len = pattern.mask.len();
    if mask_len <= 1 { 1 } else { (mask_len - 1).div_ceil(NEON_REGISTER_LENGTH) }
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn find_pattern_simd128(data: &[u8], pattern: &SimdPatternScanData) -> Option<usize> {
    let mut match_table = build_match_indexes(pattern);
    // build_match_indexes returns mask.len() slots but only fills the literal ones,
    // trailing zeros cause false rejects when bytes[1] is a wildcard so trim them off
    let valid_match_count = pattern.mask.iter().skip(1).filter(|&&m| m != 0).count();
    match_table.truncate(valid_match_count);
    let pattern_vecs = pattern_to_vec(pattern);
    let match_table_len = match_table.len();
    let vector_count = pattern_vecs.len();

    let first_byte_vec = imp::vector128_create(pattern.bytes[pattern.leading_ignore_count]);
    let leading_ignore_count = pattern.leading_ignore_count;
    let data_len = data.len();
    let data_base = data.as_ptr() as usize;

    // safe bound: outer load reads 16, trailing_zeros bumps up to +15, last inner load
    // reads up to data_ptr + vector_count*16, so we need
    // data_ptr + 15 + vector_count*16 < data_len
    let safe_len = data_len - (vector_count + 1) * NEON_REGISTER_LENGTH;
    let data_ptr_max = data_base + safe_len;
    let mut data_ptr = data_base;

    let data_end = data_base + data_len;
    'data: while data_ptr < data_ptr_max {
        // skim: while we have room for 4 windows, check all 4 via OR-reduce instead of
        // 4 movemasks, skips 64 bytes per iter on no-hit and never advances past a hit
        // so the slow path below handles the actual match
        while data_ptr + 4 * NEON_REGISTER_LENGTH <= data_end {
            let v0 = imp::load_vector128(data_ptr as *const u8);
            let v1 = imp::load_vector128((data_ptr + NEON_REGISTER_LENGTH) as *const u8);
            let v2 = imp::load_vector128((data_ptr + 2 * NEON_REGISTER_LENGTH) as *const u8);
            let v3 = imp::load_vector128((data_ptr + 3 * NEON_REGISTER_LENGTH) as *const u8);
            let e0 = imp::compare_equal(first_byte_vec, v0);
            let e1 = imp::compare_equal(first_byte_vec, v1);
            let e2 = imp::compare_equal(first_byte_vec, v2);
            let e3 = imp::compare_equal(first_byte_vec, v3);
            if imp::any_byte_set_4(e0, e1, e2, e3) {
                break;
            }
            data_ptr += 4 * NEON_REGISTER_LENGTH;
        }
        if data_ptr >= data_ptr_max {
            break;
        }

        let rhs = imp::load_vector128(data_ptr as *const u8);
        let equal = imp::compare_equal(first_byte_vec, rhs);
        let mut find_first_byte = imp::movemask(equal);

        // mask out hits that'd anchor before data_ptr, those are positions we've already
        // covered (or would underflow on first iter), without this a hit can drag
        // data_ptr backwards and infinite-loop
        if leading_ignore_count > 0 {
            find_first_byte &= !((1u32 << leading_ignore_count) - 1);
        }

        if find_first_byte == 0 {
            data_ptr += NEON_REGISTER_LENGTH;
            continue
        }

        let trailing = find_first_byte.trailing_zeros() as usize;
        data_ptr = data_ptr + trailing - leading_ignore_count;
        if data_ptr > data_ptr_max {
            break;
        }

        let mut match_table_index = 0;
        for (i, cur_pattern_vec) in pattern_vecs.iter().enumerate() {
            let register_byte_offs = i * NEON_REGISTER_LENGTH;
            let next_byte = data_ptr + register_byte_offs + 1;
            let rhs_2 = imp::load_vector128(next_byte as _);
            let compare_result = imp::movemask(imp::compare_equal(*cur_pattern_vec, rhs_2));

            while match_table_index < match_table_len {
                let match_index = std::num::Wrapping(match_table[match_table_index] as usize)
                    - std::num::Wrapping(register_byte_offs);
                if match_index.0 < NEON_REGISTER_LENGTH {
                    if ((compare_result >> match_index.0) & 1) != 1 {
                        data_ptr += 1;
                        continue 'data
                    } else {
                        match_table_index += 1;
                        continue
                    }
                }
                break
            }
        }

        return Some(data_ptr - data_base)
    }

    // scalar handles whatever the SIMD bound left behind
    scalar::find_pattern_from(data, pattern, safe_len)
}

pub fn pattern_to_vec(cb_pattern: &SimdPatternScanData) -> Vec<imp::Vector128> {
    let mut pattern_len = cb_pattern.mask.len();
    let vector_count = pattern_vec_count(cb_pattern);
    let mut pattern_vecs: Vec<imp::Vector128> = Vec::with_capacity(vector_count);

    let pattern = unsafe { cb_pattern.bytes.as_slice().get_unchecked(1) } as *const u8;

    pattern_len -= 1;

    for i in 0..vector_count {
        if i < vector_count - 1 {
            unsafe { pattern_vecs.push(imp::load_vector128(pattern.add(i * NEON_REGISTER_LENGTH))) }
        } else {
            let o = i * NEON_REGISTER_LENGTH;
            let mut neon: [u8; NEON_REGISTER_LENGTH] = [0; NEON_REGISTER_LENGTH];

            unsafe {
                for (j, slot) in neon.iter_mut().enumerate() {
                    if o + j < pattern_len {
                        *slot = *pattern.add(o + j);
                    }
                }
            }

            pattern_vecs.push(imp::load_vector128(neon.as_ptr()));
        }
    }

    pattern_vecs
}

pub fn build_match_indexes(scan_pattern: &SimdPatternScanData) -> Vec<u16> {
    let mask_length = scan_pattern.mask.len();
    let mut full_match_table: Vec<u16> = vec![0; mask_length];

    let mut match_count = 0;

    for i in 1..mask_length {
        if scan_pattern.mask[i] != 1 {
            continue
        }
        full_match_table[match_count] = i as u16 - 1;
        match_count += 1;
    }

    full_match_table
}

pub struct SimdPatternScanData {
    pub bytes: Vec<u8>,
    pub mask: Vec<u8>,
    pub leading_ignore_count: usize,
}

impl SimdPatternScanData {
    pub fn new<S: AsRef<str>>(pattern: S) -> Self {
        let pattern = pattern.as_ref();
        let mut leading_ignore_count = 0;

        let mut bytes = vec![];
        let mut mask = vec![];
        let mut found_non_ignore = false;

        let iter = pattern.split(' ').map(|value| value.trim_start_matches("0x"));

        for curr in iter {
            if curr == "??" {
                mask.push(0);
                bytes.push(0);

                if !found_non_ignore {
                    leading_ignore_count += 1;
                }
            } else {
                bytes.push(u8::from_str_radix(curr, 16).unwrap());
                mask.push(1);
                found_non_ignore = true;
            }
        }

        Self {
            bytes,
            mask,
            leading_ignore_count,
        }
    }
}
