use fp12::Fp12;
fn main() {
    for x in [0.1f32, -0.0123, 2.71, 1000.0] {
        let f = Fp12::from_f32(x);
        println!("{x:>9} -> {:012b} -> {}", f.0, f.to_f32());
    }
}
