#[repr(C, packed)]
struct FrameHdr {
    ctrl: u16,
    seq: u32,
    payload_len: u16,
}

fn run_codec(iters: i64) -> u64 {
    let mut s: u64 = 987654321;
    let mut acc: u64 = 0;
    const MASK: u64 = 9223372036854775807;
    for _ in 0..iters {
        s = s.wrapping_add(1140071481932319848) & MASK;
        let ver = (s & 15) as u16;
        let kind = ((s >> 4) & 15) as u16;
        let flags = ((s >> 8) & 255) as u16;
        let raw = ver | (kind << 4) | (flags << 8);
        let hdr = FrameHdr {
            ctrl: raw,
            seq: ((s >> 16) & 4294967295) as u32,
            payload_len: ((s >> 48) & 65535) as u16,
        };
        let ctrl = hdr.ctrl;
        let seq = hdr.seq;
        let plen = hdr.payload_len;
        let d_ver = (ctrl & 15) as u64;
        let d_kind = ((ctrl >> 4) & 15) as u64;
        let d_flags = ((ctrl >> 8) & 255) as u64;
        let mixed = (d_ver << 24) ^ (d_kind << 16) ^ (d_flags << 8) ^ (seq as u64) ^ (plen as u64);
        acc = (acc ^ mixed).wrapping_mul(16777619) & MASK;
    }
    acc
}

fn main() {
    println!("{}", run_codec(20_000_000));
}
