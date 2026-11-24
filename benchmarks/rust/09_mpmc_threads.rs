use std::sync::mpsc;
use std::thread;

fn worker_mix(mut x: u64, iters: i64) -> u64 {
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
    let seeds = [100000007u64, 200000033, 300000007, 400000009];
    let (tx, rx) = mpsc::channel();
    let mut handles = Vec::new();
    for s in seeds {
        let tx_c = tx.clone();
        handles.push(thread::spawn(move || {
            tx_c.send(worker_mix(s, 25_000_000)).unwrap();
        }));
    }
    drop(tx);
    const MASK: u64 = 9223372036854775807;
    let mut total = 0u64;
    for _ in 0..4 {
        total = (total ^ rx.recv().unwrap()) & MASK;
    }
    for h in handles {
        h.join().unwrap();
    }
    println!("{}", total);
}
