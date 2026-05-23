use std::arch::aarch64::{
    uint8x16_t, vaddv_u8, vandq_u8, vceqq_u8, vdupq_n_u8, vget_high_u8, vget_low_u8, vld1q_s8,
    vld1q_u8, vmaxvq_u8, vorrq_u8, vshlq_u8,
};

pub type Vector128 = uint8x16_t;

pub fn vector128_create(data: u8) -> Vector128 {
    unsafe { vdupq_n_u8(data) }
}

pub fn load_vector128(data: *const u8) -> Vector128 {
    unsafe { vld1q_u8(data as _) }
}

pub fn compare_equal(left: Vector128, right: Vector128) -> Vector128 {
    unsafe { vceqq_u8(left, right) }
}

/// true if any byte across the 4 vectors is non-zero, used by the 64-byte skim loop
/// so we don't pay for 4 movemasks (each is a shift + 2 horizontal adds on NEON)
pub fn any_byte_set_4(a: Vector128, b: Vector128, c: Vector128, d: Vector128) -> bool {
    unsafe {
        let ab = vorrq_u8(a, b);
        let cd = vorrq_u8(c, d);
        vmaxvq_u8(vorrq_u8(ab, cd)) != 0
    }
}

/// AArch64 has no direct _mm_movemask_epi8, so we synthesize it: AND with 0x80,
/// shift each lane to a unique bit via UC_SHIFT, horizontal-add the low and high
/// halves separately and combine
pub fn movemask(data: Vector128) -> u32 {
    const UC_SHIFT: [i8; 16] = [-7, -6, -5, -4, -3, -2, -1, 0, -7, -6, -5, -4, -3, -2, -1, 0];
    let vshift = unsafe { vld1q_s8(UC_SHIFT.as_ptr()) };
    let vmask = unsafe { vandq_u8(data, vdupq_n_u8(0x80)) };
    let vmask = unsafe { vshlq_u8(vmask, vshift) };

    let mut out: u32 = unsafe { vaddv_u8(vget_low_u8(vmask)) }.into();
    out += unsafe { (vaddv_u8(vget_high_u8(vmask)) as u32) << 8 };

    out
}