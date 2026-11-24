use std::collections::HashMap;

fn main() {
    let n = 50_000usize;
    let mut m: HashMap<String, String> = HashMap::new();
    for i in 0..n {
        m.insert(format!("key_{}", i), format!("val_{}", i * 3));
    }
    for u in 0..10_000usize {
        m.insert(format!("key_{}", u), format!("upd_{}", u));
    }
    let mut checksum: i64 = 0;
    for q in 0..n {
        if let Some(v) = m.get(&format!("key_{}", q)) {
            checksum += v.len() as i64;
        }
    }
    let mut d = 0usize;
    while d < n {
        m.remove(&format!("key_{}", d));
        d += 2;
    }
    for q in 0..n {
        if let Some(v) = m.get(&format!("key_{}", q)) {
            checksum += v.len() as i64;
        }
    }
    checksum += m.len() as i64;
    println!("{}", checksum);
}
