const MASK: u64 = 9223372036854775807;

trait Op {
    fn apply(&self, x: i64) -> i64;
}

struct AddOp { k: i64 }
struct XorOp { k: i64 }
struct MulOp { k: i64 }
struct ShiftOp { k: i64 }

impl Op for AddOp {
    fn apply(&self, x: i64) -> i64 {
        ((x as u64).wrapping_add(self.k as u64) & MASK) as i64
    }
}
impl Op for XorOp {
    fn apply(&self, x: i64) -> i64 {
        (((x as u64) ^ (self.k as u64)) & MASK) as i64
    }
}
impl Op for MulOp {
    fn apply(&self, x: i64) -> i64 {
        ((x as u64).wrapping_mul(self.k as u64) & MASK) as i64
    }
}
impl Op for ShiftOp {
    fn apply(&self, x: i64) -> i64 {
        let u = x as u64;
        ((u ^ (u >> (self.k as u64))) & MASK) as i64
    }
}

fn main() {
    let ops: Vec<Box<dyn Op>> = vec![
        Box::new(AddOp { k: 17 }),
        Box::new(XorOp { k: 2654435761 }),
        Box::new(MulOp { k: 16777619 }),
        Box::new(ShiftOp { k: 13 }),
    ];
    let mut acc: i64 = 1234567;
    for i in 0..5_000_000usize {
        acc = ops[i & 3].apply(acc);
    }
    println!("{}", acc);
}
