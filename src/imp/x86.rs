use std::arch::x86_64::{
    __m128i, _mm_cmpeq_epi8, _mm_loadu_si128, _mm_movemask_epi8, _mm_or_si128, _mm_set1_epi8,
};

pub type Vector128 = __m128i;
pub fn vector128_create(data: u8) -> Vector128 {
    unsafe { _mm_set1_epi8(data as i8) }
}

pub fn load_vector128(data: *const u8) -> Vector128 {
    unsafe { _mm_loadu_si128(data as *const Vector128) }
}

pub fn compare_equal(left: Vector128, right: Vector128) -> Vector128 {
    unsafe { _mm_cmpeq_epi8(left, right) }
}

pub fn movemask(data: Vector128) -> u32 {
    unsafe { _mm_movemask_epi8(data) as u32 }
}

/// true if any byte across the 4 vectors is non-zero, mirrors the aarch64 helper
pub fn any_byte_set_4(a: Vector128, b: Vector128, c: Vector128, d: Vector128) -> bool {
    unsafe {
        let ab = _mm_or_si128(a, b);
        let cd = _mm_or_si128(c, d);
        _mm_movemask_epi8(_mm_or_si128(ab, cd)) != 0
    }
}