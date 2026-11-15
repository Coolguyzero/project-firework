//! FP-Native: 비트 배치를 고를 수 있는 작은 부동소수점.
//!
//! `Format { e, m, fn_mode }` 하나로 FP12(E5M6), FP16(E5M10), FP8(E4M3, E5M2) 등을 다 다룬다.
//! - IEEE 방식(`fn_mode = false`): 지수가 전부 1이면 Inf/NaN.
//! - FN 방식(`fn_mode = true`, OCP FP8 E4M3FN): Inf 없음, 지수·가수 전부 1만 NaN. 그만큼 범위가 넓다.
//!
//! 반올림은 가장 가까운 값, 딱 중간이면 짝수 쪽 (round-to-nearest-even). 서브노멀 지원.
//! 저장은 하위 (1 + e + m) 비트를 쓰는 u16.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Format {
    pub e: u32,
    pub m: u32,
    pub fn_mode: bool,
}

impl Format {
    pub const FP12: Format = Format { e: 5, m: 6, fn_mode: false };
    pub const FP16: Format = Format { e: 5, m: 10, fn_mode: false };
    pub const BF16: Format = Format { e: 8, m: 7, fn_mode: false };
    pub const FP8_E4M3: Format = Format { e: 4, m: 3, fn_mode: true };
    pub const FP8_E5M2: Format = Format { e: 5, m: 2, fn_mode: false };

    pub const fn new(e: u32, m: u32, fn_mode: bool) -> Self {
        assert!(e >= 2 && e <= 8 && m <= 23 && 1 + e + m <= 16);
        Format { e, m, fn_mode }
    }

    pub const fn bits(self) -> u32 {
        1 + self.e + self.m
    }
    const fn bias(self) -> i32 {
        (1 << (self.e - 1)) - 1
    }
    const fn exp_max(self) -> u32 {
        (1 << self.e) - 1
    }
    const fn man_mask(self) -> u32 {
        (1 << self.m) - 1
    }
    const fn sign_bit(self) -> u32 {
        1 << (self.e + self.m)
    }
    /// 가장 큰 유한값의 코드 (부호 제외).
    const fn max_code(self) -> u32 {
        if self.fn_mode {
            (self.exp_max() << self.m) | (self.man_mask() - 1)
        } else {
            ((self.exp_max() - 1) << self.m) | self.man_mask()
        }
    }
    fn nan_code(self) -> u16 {
        if self.fn_mode {
            ((self.exp_max() << self.m) | self.man_mask()) as u16
        } else {
            ((self.exp_max() << self.m) | (1 << self.m.saturating_sub(1)).max(1)) as u16
        }
    }

    pub fn max_value(self) -> f32 {
        self.decode(self.max_code() as u16)
    }

    /// f32 -> 코드. `saturate`면 범위를 넘는 값은 Inf 대신 최대값으로 자른다.
    pub fn encode(self, x: f32, saturate: bool) -> u16 {
        let b = x.to_bits();
        let sign = if b >> 31 == 1 { self.sign_bit() } else { 0 };
        let exp = ((b >> 23) & 0xff) as i32;
        let man = b & 0x7f_ffff;

        if exp == 0xff {
            if man != 0 {
                return self.nan_code();
            }
            if saturate || self.fn_mode {
                return (sign | self.max_code()) as u16;
            }
            return (sign | (self.exp_max() << self.m)) as u16;
        }

        // f32 서브노멀은 숨은 1이 없고 지수가 -126으로 고정이다.
        let e = exp.max(1) - 127 + self.bias();
        let full = if exp == 0 { man } else { man | 0x80_0000 };
        let shift = (23 - self.m) + (1 - e).max(0) as u32;
        let bits = if shift > 25 {
            0
        } else {
            let mut q = full >> shift;
            if shift > 0 {
                let rem = full & ((1u32 << shift) - 1);
                let half = 1u32 << (shift - 1);
                if rem > half || (rem == half && (q & 1) == 1) {
                    q += 1;
                }
            }
            // q는 숨은 1 포함(정규수일 때). 가수가 넘치면 지수로 자연스럽게 올라간다.
            (((e.max(1) - 1) as u32) << self.m) + q
        };
        if bits > self.max_code() {
            if saturate || self.fn_mode {
                return (sign | self.max_code()) as u16;
            }
            return (sign | (self.exp_max() << self.m)) as u16;
        }
        (sign | bits) as u16
    }

    pub fn decode(self, code: u16) -> f32 {
        let v = code as u32;
        let neg = v & self.sign_bit() != 0;
        let e = ((v >> self.m) & self.exp_max()) as i32;
        let m = v & self.man_mask();
        let mag = if self.fn_mode {
            if e == self.exp_max() as i32 && m == self.man_mask() {
                return f32::NAN;
            }
            self.finite(e, m)
        } else if e == self.exp_max() as i32 {
            if m != 0 {
                return f32::NAN;
            }
            f32::INFINITY
        } else {
            self.finite(e, m)
        };
        if neg { -mag } else { mag }
    }

    fn finite(self, e: i32, m: u32) -> f32 {
        let frac = m as f64 / (1u64 << self.m) as f64;
        let v = if e == 0 {
            frac * 2f64.powi(1 - self.bias())
        } else {
            (1.0 + frac) * 2f64.powi(e - self.bias())
        };
        v as f32
    }

    /// f32 -> 이 포맷 -> f32 (양자화 오차 확인용).
    pub fn round_trip(self, x: f32, saturate: bool) -> f32 {
        self.decode(self.encode(x, saturate))
    }
}

/// 자주 쓰는 FP12 래퍼.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Fp12(pub u16);

impl Fp12 {
    pub const MAX: f32 = 65024.0;
    pub fn from_f32(x: f32) -> Self {
        Fp12(Format::FP12.encode(x, false))
    }
    pub fn to_f32(self) -> f32 {
        Format::FP12.decode(self.0)
    }
}

/// FP12 두 개를 3바이트로 묶기.
pub fn pack(vals: &[Fp12]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len().div_ceil(2) * 3);
    for pair in vals.chunks(2) {
        let a = pair[0].0 & 0xfff;
        let b = pair.get(1).map_or(0, |v| v.0 & 0xfff);
        out.push(a as u8);
        out.push(((a >> 8) as u8) | (((b & 0xf) as u8) << 4));
        out.push((b >> 4) as u8);
    }
    out
}

pub fn unpack(bytes: &[u8], n: usize) -> Vec<Fp12> {
    let mut out = Vec::with_capacity(n);
    for c in bytes.chunks(3) {
        let a = c[0] as u16 | ((c[1] as u16 & 0xf) << 8);
        let b = (c[1] as u16 >> 4) | ((c[2] as u16) << 4);
        out.push(Fp12(a));
        out.push(Fp12(b));
    }
    out.truncate(n);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const FORMATS: [Format; 7] = [
        Format::FP12,
        Format::FP16,
        Format::BF16,
        Format::FP8_E4M3,
        Format::FP8_E5M2,
        Format { e: 4, m: 7, fn_mode: false },
        Format { e: 6, m: 5, fn_mode: false },
    ];

    #[test]
    fn all_codes_roundtrip() {
        for f in FORMATS {
            for c in 0u32..(1 << f.bits()) {
                let x = f.decode(c as u16);
                if x.is_nan() {
                    continue;
                }
                assert_eq!(f.encode(x, false) as u32, c, "{f:?} code {c:#x} = {x}");
            }
        }
    }

    #[test]
    fn nearest_rounding() {
        for f in FORMATS {
            for c in 0..f.max_code() {
                let lo = f.decode(c as u16) as f64;
                let hi = f.decode(c as u16 + 1) as f64;
                let mid = (lo + hi) / 2.0;
                let d = (hi - lo) * 0.01;
                let (below, above) = ((mid - d) as f32, (mid + d) as f32);
                if below as f64 <= lo || above as f64 >= hi {
                    continue; // f32로 표현이 안 될 만큼 좁은 칸
                }
                assert_eq!(f.encode(below, false) as u32, c, "{f:?} below {c}");
                assert_eq!(f.encode(above, false) as u32, c + 1, "{f:?} above {c}");
            }
        }
    }

    #[test]
    fn known_max_values() {
        assert_eq!(Format::FP12.max_value(), 65024.0);
        assert_eq!(Format::FP16.max_value(), 65504.0);
        assert_eq!(Format::FP8_E4M3.max_value(), 448.0);
        assert_eq!(Format::FP8_E5M2.max_value(), 57344.0);
    }

    #[test]
    fn fp16_matches_ieee_half_bits() {
        assert_eq!(Format::FP16.encode(1.0, false), 0x3c00);
        assert_eq!(Format::FP16.encode(-2.0, false), 0xc000);
        assert_eq!(Format::FP16.encode(65504.0, false), 0x7bff);
        assert_eq!(Format::FP16.encode(2f32.powi(-24), false), 0x0001);
    }

    #[test]
    fn bf16_is_rounded_top_half_of_f32() {
        for x in [1.0f32, -3.5, 0.0123, 1e-30, 3e38, 1.0 + 1.0 / 256.0] {
            let b = x.to_bits();
            let want = (b + 0x7fff + ((b >> 16) & 1)) >> 16;
            assert_eq!(Format::BF16.encode(x, false) as u32, want, "{x}");
        }
    }

    #[test]
    fn specials() {
        let f = Format::FP12;
        assert!(f.decode(f.encode(1e6, false)).is_infinite());
        assert_eq!(f.decode(f.encode(1e6, true)), 65024.0);
        assert!(f.decode(f.encode(f32::NAN, false)).is_nan());
        assert_eq!(f.decode(f.encode(1e-12, false)), 0.0);
        assert_eq!(f.encode(-0.0, false), 0x800);
        assert_eq!(Format::FP8_E4M3.round_trip(1e6, false), 448.0);
        assert!(Format::FP8_E4M3.decode(0x7f).is_nan());
    }

    #[test]
    fn fp8_fits_exactly_in_fp12() {
        for c in 0u16..256 {
            let x = Format::FP8_E4M3.decode(c);
            if x.is_nan() {
                continue;
            }
            assert_eq!(Format::FP12.round_trip(x, false), x, "E4M3 code {c}");
        }
    }

    #[test]
    fn pack_roundtrip() {
        let v: Vec<Fp12> = (0..7).map(|i| Fp12((i * 577) & 0xfff)).collect();
        let p = pack(&v);
        assert_eq!(p.len(), 12);
        assert_eq!(unpack(&p, 7), v);
    }
}
