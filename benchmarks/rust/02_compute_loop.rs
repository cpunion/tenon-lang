fn compute_mix(iters: i64) -> u64 {
    let mut x: u64 = 123456789;
    let mut acc: u64 = 0;
    const MASK: u64 = 9223372036854775807;
    for _ in 0..iters {
        x = x.wrapping_add(1140071481932319848) & MASK;
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(490278349) & MASK;
        z = (z ^ (z >> 27)).wrapping_mul(16777619) & MASK;
        z = z ^ (z >> 31);
        acc = (acc ^ z) & MASK;
    }
    acc
}

fn main() {
    println!("{}", compute_mix(100_000_000));
}
