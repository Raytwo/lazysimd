#![feature(int_roundings)]

use std::arch::aarch64::{uint8x16_t, vaddv_u8, vandq_u8, vceqq_u8, vdupq_n_u8, vget_high_u8, vget_low_u8, vld1q_s8, vld1q_u8, vshlq_u8};

pub use lazysimd_macro::*;
pub mod scan;

const NEON_REGISTER_LENGTH: usize = 16;

pub fn get_offset_neon(data: &[u8], pattern: &str) -> Option<usize> {
    find_pattern_neon(data.as_ptr(), data.len(), pattern)
}

pub fn find_pattern_neon<S: AsRef<str>>(data: *const u8, data_len: usize, pattern: S) -> Option<usize> {
    let pattern = SimdPatternScanData::new(&pattern);

    let match_table = build_match_indexes(&pattern);
    let pattern_vecs = pattern_to_vec(&pattern);
    let match_table_len = match_table.len();

    // Fills a register with the first byte of the pattern
    let first_byte_vec = unsafe { vdupq_n_u8(pattern.bytes[pattern.leading_ignore_count]) };
    // Compute the size of the array minus what's the biggest size between the pattern or a Simd register
    let search_length = data_len - std::cmp::max(pattern.bytes.len(), NEON_REGISTER_LENGTH);

    let leading_ignore_count = pattern.leading_ignore_count;

    let mut data_ptr = data as usize;
    let data_ptr_max = data_ptr + search_length;

    'data: while data_ptr < data_ptr_max {
        // Fills a register with bytes
        let rhs = unsafe { vld1q_u8(data_ptr as _) };

        // Compare the register filled with the first byte with the 16 next bytes and return a vector where matching bytes are represented by 0xFF and the rest by 0x0
        let equal = unsafe { vceqq_u8(first_byte_vec, rhs) };

        // Converts vceqq's output to a u32 bitfield equivalent where matching bytes are represented by a bit being set
        let find_first_byte = _mm_movemask_aarch64(equal);

        // If the value is 0, it means no bit was set, and therefore the first byte of the signature is missing.
        // Abort early and move on to the next 16 bytes
        if find_first_byte == 0 {
            data_ptr += NEON_REGISTER_LENGTH - 1;
            continue
        }

        // Advance the pointer by the amount of non-matching bytes in the current window
        let test = (find_first_byte.trailing_zeros() as i32).wrapping_sub(leading_ignore_count as i32);
        data_ptr = data_ptr.wrapping_add_signed(test as isize);

        let mut match_table_index = 0;

        // For each array of pattern
        for (i, cur_pattern_vec) in pattern_vecs.iter().enumerate() {
            let register_byte_offs = i * NEON_REGISTER_LENGTH;

            let next_byte = data_ptr + register_byte_offs + 1;

            let rhs_2 = unsafe { vld1q_u8(next_byte as _) };

            let compare_result = unsafe { _mm_movemask_aarch64(vceqq_u8(*cur_pattern_vec, rhs_2)) };

            while match_table_index < match_table_len {
                let match_index = std::num::Wrapping(match_table[match_table_index] as usize) - std::num::Wrapping(register_byte_offs as usize);

                if match_index.0 < NEON_REGISTER_LENGTH {
                    if ((compare_result >> match_index.0) & 1) != 1 {
                        // TODO: Improve this. Moves by one
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

        return Some(data_ptr - data as usize)
    }

    None

    // // We are past the point where we can still look for the signature without risking an overflow, so tread carefully
    // let position = data_ptr - data as usize;

    // // TODO: Do a simpler search in the remaining bytes here
    // data_ptr - data as usize
}

pub fn pattern_to_vec(cb_pattern: &SimdPatternScanData) -> Vec<uint8x16_t> {
    let mut pattern_len = cb_pattern.mask.len();
    let vector_count = (pattern_len - 1).div_ceil(NEON_REGISTER_LENGTH);
    let mut pattern_vecs: Vec<uint8x16_t> = Vec::with_capacity(vector_count);

    let pattern = unsafe { cb_pattern.bytes.as_slice().get_unchecked(1) } as *const u8;

    pattern_len -= 1;

    for i in 0..vector_count {
        if i < vector_count - 1 {
            pattern_vecs.push(unsafe { vld1q_u8(pattern.add(i * NEON_REGISTER_LENGTH)) })
        } else {
            let o = i * NEON_REGISTER_LENGTH;
            let neon: &mut [u8; NEON_REGISTER_LENGTH] = &mut [0; NEON_REGISTER_LENGTH];

            unsafe {
                neon[0] = *pattern.add(o);
                neon[1] = if o + 1 < pattern_len { *pattern.add(o + 1) } else { 0 };
                neon[2] = if o + 2 < pattern_len { *pattern.add(o + 2) } else { 0 };
                neon[3] = if o + 3 < pattern_len { *pattern.add(o + 3) } else { 0 };
                neon[4] = if o + 4 < pattern_len { *pattern.add(o + 4) } else { 0 };
                neon[5] = if o + 5 < pattern_len { *pattern.add(o + 5) } else { 0 };
                neon[6] = if o + 6 < pattern_len { *pattern.add(o + 6) } else { 0 };
                neon[7] = if o + 7 < pattern_len { *pattern.add(o + 7) } else { 0 };
                neon[8] = if o + 8 < pattern_len { *pattern.add(o + 8) } else { 0 };
                neon[9] = if o + 9 < pattern_len { *pattern.add(o + 9) } else { 0 };
                neon[10] = if o + 10 < pattern_len { *pattern.add(o + 10) } else { 0 };
                neon[11] = if o + 11 < pattern_len { *pattern.add(o + 11) } else { 0 };
                neon[12] = if o + 12 < pattern_len { *pattern.add(o + 12) } else { 0 };
                neon[13] = if o + 13 < pattern_len { *pattern.add(o + 13) } else { 0 };
                neon[14] = if o + 14 < pattern_len { *pattern.add(o + 14) } else { 0 };
                neon[15] = if o + 15 < pattern_len { *pattern.add(o + 15) } else { 0 };
            }

            unsafe { pattern_vecs.push(vld1q_u8(neon.as_ptr())) };
        }
    }

    pattern_vecs
}

pub fn build_match_indexes(scan_pattern: &SimdPatternScanData) -> Vec<u16> {
    let mask_length = scan_pattern.mask.len();
    let mut full_match_table: Vec<u16> = vec![0; mask_length];

    let mut match_count = 0;

    for i in 1..mask_length {
        // If this byte is masked, we continue
        if scan_pattern.mask[i] != 1 {
            continue
        }
        // Add the index of the byte that wasn't in the vector
        full_match_table[match_count] = i as u16 - 1;
        match_count += 1;
    }

    full_match_table
}

/// Equivalent to MoveMask in x86
#[inline]
fn _mm_movemask_aarch64(input: uint8x16_t) -> u32 {
    const UC_SHIFT: [i8; 16] = [-7, -6, -5, -4, -3, -2, -1, 0, -7, -6, -5, -4, -3, -2, -1, 0];
    // Fills a vector with UC_SHIFT
    let vshift = unsafe { vld1q_s8(UC_SHIFT.as_ptr()) };
    // Fills a vector with 0x80 and performs AND on the input vector
    let vmask = unsafe { vandq_u8(input, vdupq_n_u8(0x80)) };
    // Shift-left vmask using UC_SHIFT
    let vmask = unsafe { vshlq_u8(vmask, vshift) };

    // Takes the lower 64 bits of vmask and add all bytes together
    let mut out: u32 = unsafe { vaddv_u8(vget_low_u8(vmask)) }.into();
    // Takes the higher 64 bits of vmask, add all bytes together then shift left by 8 and add the result to out
    out += unsafe { (vaddv_u8(vget_high_u8(vmask)) as u32) << 8 };

    out
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
