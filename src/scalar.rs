use crate::SimdPatternScanData;

pub fn find_pattern_in(data: &[u8], pattern: &SimdPatternScanData) -> Option<usize> {
    find_pattern_from(data, pattern, 0)
}

pub fn find_pattern_from(data: &[u8], pattern: &SimdPatternScanData, start: usize) -> Option<usize> {
    let pat_len = pattern.bytes.len();
    if pat_len == 0 || data.len() < pat_len || start > data.len() - pat_len {
        return None;
    }
    let last = data.len() - pat_len;
    'outer: for i in start..=last {
        for (j, (&b, &m)) in pattern.bytes.iter().zip(pattern.mask.iter()).enumerate() {
            if m != 0 && data[i + j] != b {
                continue 'outer;
            }
        }
        return Some(i);
    }
    None
}

pub fn find_pattern_scalar(data: &[u8], pattern: &str) -> Option<usize> {
    let pat = SimdPatternScanData::new(pattern);
    find_pattern_in(data, &pat)
}
