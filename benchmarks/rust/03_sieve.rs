fn sieve_count(limit: usize) -> i64 {
    let mut flags = vec![1i64; limit + 1];
    flags[0] = 0;
    flags[1] = 0;
    let mut p = 2usize;
    while p * p <= limit {
        if flags[p] == 1 {
            let mut m = p * p;
            while m <= limit {
                flags[m] = 0;
                m += p;
            }
        }
        p += 1;
    }
    let mut count: i64 = 0;
    for i in 2..=limit {
        count += flags[i];
    }
    count
}

fn main() {
    let mut total: i64 = 0;
    for _ in 0..5 {
        total += sieve_count(1_000_000);
    }
    println!("{}", total);
}
