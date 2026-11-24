fn main() {
    const N: usize = 1024;
    const REPS: usize = 50_000;
    let mut a = vec![[0.0f64; 4]; N];
    let mut b = vec![[0.0f64; 4]; N];

    for i in 0..N {
        a[i] = [
            ((i + 1) & 15) as f64 * 0.25,
            ((i + 3) & 15) as f64 * 0.25,
            ((i + 5) & 15) as f64 * 0.25,
            ((i + 7) & 15) as f64 * 0.25,
        ];
        b[i] = [
            ((i * 3 + 2) & 15) as f64 * 0.25,
            ((i * 3 + 4) & 15) as f64 * 0.25,
            ((i * 3 + 6) & 15) as f64 * 0.25,
            ((i * 3 + 8) & 15) as f64 * 0.25,
        ];
    }

    let mut total = 0.0f64;
    for r in 0..REPS {
        for j in 0..N {
            let x = a[j];
            let y = b[(j + r) & 1023];
            total += x[0] * y[0] + x[1] * y[1] + x[2] * y[2] + x[3] * y[3];
        }
    }

    println!("{}", (total * 16.0) as i64);
}
