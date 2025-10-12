#[cfg(target_arch = "aarch64")]
use skyline::hooks::{getRegionAddress, Region};

#[cfg(target_arch = "aarch64")]
pub fn get_text() -> &'static [u8] {
    unsafe {
        let ptr = getRegionAddress(Region::Text) as *const u8;
        let size = (getRegionAddress(Region::Rodata) as usize) - (ptr as usize);
        std::slice::from_raw_parts(ptr, size)
    }
}

#[cfg(target_arch = "x86_64")]
pub fn get_text() -> &'static [u8] {
    unimplemented!()
}