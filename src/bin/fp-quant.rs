//! safetensors 가중치를 작은 float 포맷으로 "왕복" 시킨다 (양자화 시뮬레이션).
//! 원래 dtype(BF16/F16/F32)과 파일 구조는 그대로 두고, 값만 포맷이 표현할 수 있는 값으로 바꾼다.
//! 그래서 결과 파일을 llama.cpp 변환기에 그대로 넣어 perplexity를 잴 수 있다.
//!
//! 사용법: fp-quant <입력.safetensors> <출력.safetensors> [--format fp12] [--scale] [--all]
//!   --format  fp12(=e5m6) | fp16 | bf16 | fp8(=e4m3fn) | e5m2 | e4m7 | e<E>m<M>[fn]
//!   --scale   텐서마다 배율을 둬서 최대값을 포맷 최대값에 맞춤 (실제 FP8 체크포인트 방식)
//!   --all     1차원 텐서(norm, bias)도 바꿈. 기본은 2차원 이상 가중치만.

use fp12::Format;
use std::{env, fs, process};

fn parse_format(s: &str) -> Option<Format> {
    let s = s.to_ascii_lowercase();
    match s.as_str() {
        "fp12" => return Some(Format::FP12),
        "fp16" => return Some(Format::FP16),
        "bf16" => return Some(Format::BF16),
        "fp8" | "e4m3" | "e4m3fn" => return Some(Format::FP8_E4M3),
        _ => {}
    }
    let fn_mode = s.ends_with("fn");
    let core = s.strip_suffix("fn").unwrap_or(&s);
    let rest = core.strip_prefix('e')?;
    let (e, m) = rest.split_once('m')?;
    let (e, m) = (e.parse().ok()?, m.parse().ok()?);
    if !(2..=8).contains(&e) || m > 23 || 1 + e + m > 16 {
        return None;
    }
    Some(Format { e, m, fn_mode })
}

fn die(msg: &str) -> ! {
    eprintln!("오류: {msg}");
    process::exit(1);
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut pos = vec![];
    let mut fmt = Format::FP12;
    let (mut scale, mut all) = (false, false);
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--format" => {
                i += 1;
                let s = args.get(i).unwrap_or_else(|| die("--format 값이 없어"));
                fmt = parse_format(s).unwrap_or_else(|| die(&format!("모르는 포맷: {s}")));
            }
            "--scale" => scale = true,
            "--all" => all = true,
            a => pos.push(a.to_string()),
        }
        i += 1;
    }
    if pos.len() != 2 {
        die("사용법: fp-quant <입력.safetensors> <출력.safetensors> [--format fp12] [--scale] [--all]");
    }

    let mut buf = fs::read(&pos[0]).unwrap_or_else(|e| die(&format!("읽기 실패: {e}")));
    let n = u64::from_le_bytes(buf[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value =
        serde_json::from_slice(&buf[8..8 + n]).unwrap_or_else(|e| die(&format!("헤더 파싱 실패: {e}")));
    let base = 8 + n;

    println!(
        "포맷 E{}M{}{} ({}비트, 최대 {}), 배율 {}, 1차원 포함 {}",
        fmt.e, fmt.m, if fmt.fn_mode { "FN" } else { "" }, fmt.bits(), fmt.max_value(),
        if scale { "켬" } else { "끔" }, if all { "예" } else { "아니오" }
    );

    let (mut tot_err, mut tot_sig, mut tot_n, mut tot_sat) = (0f64, 0f64, 0usize, 0usize);
    let mut worst: Vec<(f64, String)> = vec![];
    let obj = header.as_object().unwrap_or_else(|| die("헤더가 객체가 아님"));
    for (name, info) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype = info["dtype"].as_str().unwrap_or("");
        let shape = info["shape"].as_array().map_or(0, |s| s.len());
        let off = &info["data_offsets"];
        let (a, b) = (off[0].as_u64().unwrap() as usize + base, off[1].as_u64().unwrap() as usize + base);
        if shape < 2 && !all {
            continue;
        }
        let width = match dtype {
            "BF16" | "F16" => 2,
            "F32" => 4,
            _ => continue, // 정수 텐서 등은 건드리지 않음
        };
        let data = &mut buf[a..b];
        let read = |c: &[u8]| -> f32 {
            match dtype {
                "BF16" => f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16),
                "F16" => Format::FP16.decode(u16::from_le_bytes([c[0], c[1]])),
                _ => f32::from_le_bytes([c[0], c[1], c[2], c[3]]),
            }
        };
        let amax = data.chunks(width).map(|c| read(c).abs()).filter(|v| v.is_finite()).fold(0f32, f32::max);
        let s = if scale && amax > 0.0 { amax / fmt.max_value() } else { 1.0 };
        let (mut err, mut sig, mut sat) = (0f64, 0f64, 0usize);
        for c in data.chunks_mut(width) {
            let x = read(c);
            let y = fmt.round_trip(x / s, true) * s;
            if (x / s).abs() > fmt.max_value() {
                sat += 1;
            }
            err += ((y - x) as f64).powi(2);
            sig += (x as f64).powi(2);
            match dtype {
                "BF16" => c.copy_from_slice(&Format::BF16.encode(y, true).to_le_bytes()),
                "F16" => c.copy_from_slice(&Format::FP16.encode(y, true).to_le_bytes()),
                _ => c.copy_from_slice(&y.to_le_bytes()),
            }
        }
        tot_err += err;
        tot_sig += sig;
        tot_n += data.len() / width;
        tot_sat += sat;
        if sig > 0.0 {
            worst.push(((err / sig).sqrt(), name.clone()));
        }
    }

    fs::write(&pos[1], &buf).unwrap_or_else(|e| die(&format!("쓰기 실패: {e}")));
    worst.sort_by(|x, y| y.0.total_cmp(&x.0));
    println!("바꾼 값: {tot_n}개, 범위 넘어 잘린 값: {tot_sat}개");
    println!("전체 상대 RMS 오차: {:.4}%", 100.0 * (tot_err / tot_sig.max(1e-30)).sqrt());
    println!("오차 큰 텐서 5개:");
    for (r, name) in worst.iter().take(5) {
        println!("  {:.4}%  {name}", 100.0 * r);
    }
}
