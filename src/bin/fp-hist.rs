//! 가중치 값 분포를 본다. "E4M7이 왜 거의 무손실인가"를 추정이 아니라 숫자로 확인하는 용도.
//!
//! 사용법: fp-hist <입력.safetensors> <출력_접두어> [--all]
//!   --all  1차원 텐서(norm, bias)도 포함. 기본은 fp-quant와 같게 2차원 이상만.
//!
//! 결과 파일 2개:
//!   <접두어>_hist.csv   |값|을 2의 지수로 나눈 히스토그램 (2^k <= |x| < 2^(k+1))
//!                       exp2, 개수, 비율, 에너지 비율(x² 합 기준 = 그 칸이 출력에 주는 영향의 대략)
//!   <접두어>_range.csv  포맷마다 범위 밖으로 나가는 값의 비율
//!                       overflow  = 최대값보다 큼 (잘림)
//!                       subnormal = 최소 정규수보다 작음 (가수 비트를 일부 잃음)
//!                       to_zero   = 0으로 사라짐
//!                       scale 열: none = 배율 없음 (1단계 e4m7/e5m6/e6m5 방식),
//!                                 tensor = fp-quant --scale 과 같은 텐서별 배율

use fp12::Format;
use std::{env, fs, io::Write, process};

const FORMATS: [(&str, Format); 5] = [
    ("e3m8", Format { e: 3, m: 8, fn_mode: false }),
    ("e4m7", Format { e: 4, m: 7, fn_mode: false }),
    ("e5m6", Format { e: 5, m: 6, fn_mode: false }),
    ("e6m5", Format { e: 6, m: 5, fn_mode: false }),
    ("e4m3fn", Format::FP8_E4M3),
];
const LO: i32 = -40; // 이보다 작은 지수는 이 칸에 모음
const HI: i32 = 20;

fn die(msg: &str) -> ! {
    eprintln!("오류: {msg}");
    process::exit(1);
}

/// 포맷의 (최대값, 최소 정규수, 최소 서브노멀).
fn limits(f: Format) -> (f32, f32, f32) {
    let bias = (1i32 << (f.e - 1)) - 1;
    let min_norm = 2f32.powi(1 - bias);
    (f.max_value(), min_norm, min_norm * 2f32.powi(-(f.m as i32)))
}

#[derive(Default, Clone, Copy)]
struct Tally {
    over: u64,
    sub: u64,
    zero: u64,
    sub_e: f64,
    zero_e: f64,
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let all = args.iter().any(|a| a == "--all");
    let pos: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    if pos.len() != 2 {
        die("사용법: fp-hist <입력.safetensors> <출력_접두어> [--all]");
    }

    let buf = fs::read(pos[0]).unwrap_or_else(|e| die(&format!("읽기 실패: {e}")));
    let n = u64::from_le_bytes(buf[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value =
        serde_json::from_slice(&buf[8..8 + n]).unwrap_or_else(|e| die(&format!("헤더 파싱 실패: {e}")));
    let base = 8 + n;
    let obj = header.as_object().unwrap_or_else(|| die("헤더가 객체가 아님"));

    let bins = (HI - LO + 1) as usize;
    let mut count = vec![0u64; bins];
    let mut energy = vec![0f64; bins];
    let (mut zeros, mut total, mut tot_e) = (0u64, 0u64, 0f64);
    // [포맷][0=배율없음, 1=텐서배율]
    let mut tally = [[Tally::default(); 2]; FORMATS.len()];
    let lims: Vec<_> = FORMATS.iter().map(|(_, f)| limits(*f)).collect();

    let names: Vec<&String> = obj.keys().filter(|k| *k != "__metadata__").collect();
    for (ti, name) in names.iter().enumerate() {
        let info = &obj[name.as_str()];
        let dtype = info["dtype"].as_str().unwrap_or("");
        let shape = info["shape"].as_array().map_or(0, |s| s.len());
        if shape < 2 && !all {
            continue;
        }
        let width = match dtype {
            "BF16" | "F16" => 2,
            "F32" => 4,
            _ => continue,
        };
        let off = &info["data_offsets"];
        let (a, b) = (off[0].as_u64().unwrap() as usize + base, off[1].as_u64().unwrap() as usize + base);
        let vals: Vec<f32> = buf[a..b]
            .chunks(width)
            .map(|c| match dtype {
                "BF16" => f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16),
                "F16" => Format::FP16.decode(u16::from_le_bytes([c[0], c[1]])),
                _ => f32::from_le_bytes([c[0], c[1], c[2], c[3]]),
            })
            .filter(|v| v.is_finite())
            .collect();
        let amax = vals.iter().fold(0f32, |m, v| m.max(v.abs()));
        let scales: Vec<f32> = FORMATS
            .iter()
            .map(|(_, f)| if amax > 0.0 { amax / f.max_value() } else { 1.0 })
            .collect();

        for &x in &vals {
            let ax = x.abs();
            total += 1;
            if ax == 0.0 {
                zeros += 1;
                continue;
            }
            let e2 = (ax as f64).powi(2);
            tot_e += e2;
            let k = (((ax.to_bits() >> 23) as i32) - 127).clamp(LO, HI);
            count[(k - LO) as usize] += 1;
            energy[(k - LO) as usize] += e2;
            for (fi, &(max, min_norm, min_sub)) in lims.iter().enumerate() {
                for (si, s) in [1.0, scales[fi]].into_iter().enumerate() {
                    let v = ax / s;
                    let t = &mut tally[fi][si];
                    if v > max {
                        t.over += 1;
                    } else if v <= min_sub * 0.5 {
                        t.zero += 1;
                        t.zero_e += e2;
                    } else if v < min_norm {
                        t.sub += 1;
                        t.sub_e += e2;
                    }
                }
            }
        }
        println!("{:.1}% {name}", 100.0 * (ti + 1) as f64 / names.len() as f64);
    }

    let pre = pos[1];
    let mut h = fs::File::create(format!("{pre}_hist.csv")).unwrap_or_else(|e| die(&format!("쓰기 실패: {e}")));
    writeln!(h, "exp2,count,frac,energy_frac").unwrap();
    writeln!(h, "zero,{zeros},{:.6e},0", zeros as f64 / total.max(1) as f64).unwrap();
    for (i, (&c, &en)) in count.iter().zip(&energy).enumerate() {
        if c > 0 {
            let fr = c as f64 / total.max(1) as f64;
            writeln!(h, "{},{c},{fr:.6e},{:.6e}", i as i32 + LO, en / tot_e.max(1e-30)).unwrap();
        }
    }

    let mut r = fs::File::create(format!("{pre}_range.csv")).unwrap_or_else(|e| die(&format!("쓰기 실패: {e}")));
    let head = "format,scale,max,min_normal,min_sub,overflow_frac,subnormal_frac,to_zero_frac,subnormal_energy,to_zero_energy";
    writeln!(r, "{head}").unwrap();
    println!("\n값 {total}개 (정확히 0: {zeros}개)\n{head}");
    for (fi, (fname, _)) in FORMATS.iter().enumerate() {
        let (max, mn, ms) = lims[fi];
        for (si, sname) in ["none", "tensor"].iter().enumerate() {
            let t = tally[fi][si];
            let tf = total.max(1) as f64;
            let line = format!(
                "{fname},{sname},{max},{mn:e},{ms:e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}",
                t.over as f64 / tf,
                t.sub as f64 / tf,
                t.zero as f64 / tf,
                t.sub_e / tot_e.max(1e-30),
                t.zero_e / tot_e.max(1e-30)
            );
            writeln!(r, "{line}").unwrap();
            println!("{line}");
        }
    }
    println!("\n저장: {pre}_hist.csv, {pre}_range.csv");
}
