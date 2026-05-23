use crate::SimdPatternScanData;

pub fn find(data: &[u8], pattern: &str) -> Option<usize> {
    let pat = SimdPatternScanData::new(pattern);
    find_with(data, &pat)
}

pub fn find_with(data: &[u8], pattern: &SimdPatternScanData) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    {
        if avx2_available() {
            return unsafe { crate::imp_avx2::find_pattern_avx2(data, pattern) };
        }
    }
    crate::find_pattern_in(data, pattern)
}

#[cfg(target_arch = "x86_64")]
fn avx2_available() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::is_x86_feature_detected!("avx2"))
}
