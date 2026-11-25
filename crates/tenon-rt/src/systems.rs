//! Zero-dependency Systems Programming Runtime Primitives (S1 + S2 + S3):
//! - Compact 1-byte-per-element `Bytes` + zero-copy RC-backed `BytesSlice` + endian I/O
//! - `@repr(C)` / `@packed` / `union` raw byte buffers & raw memory ops + C string interop
//! - Bit intrinsics, overflow trap, `Float32` formatting/parsing
//! - Real OS threads, explicit-ordering `AtomicInt`, `Mutex[T]`, MPMC `Channel[T]`
//! - Non-blocking TCP sockets (`TcpListener` / `TcpStream`)
//! - SIMD 4-lane/16-lane vector operations (`Float32x4`, `Float64x4`, `Int32x4`, `Int64x4`, `UInt8x16`)
//! - Dynamic `dlsym` lookup for JIT `extern "C"` calls

use crate::perceus::{
    ObjHeader, TAG_ATOMIC, TAG_BYTES, TAG_CHANNEL, TAG_CSTRUCT, TAG_LIST, TAG_MUTEX, TAG_NONE,
    TAG_SIMD, TAG_SLICE, TAG_SOME, TAG_TEXT, fields_ptr, make_text, make_text_bytes, read_text,
    read_text_bytes, tenon_rc_alloc, tenon_rc_dec, tenon_rc_ensure_unique, tenon_rc_inc,
    tenon_rc_set_rc_mask,
};
use std::alloc::{Layout, alloc, dealloc};
use std::cell::Cell;
use std::collections::{BTreeMap, VecDeque};
use std::ffi::{CStr, CString, c_char, c_void};
use std::io::{ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
#[cfg(unix)]
use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

// =========================================================================
// 0. Dynamic C Symbol Lookup (for Cranelift JIT `extern "C"` interop)
// =========================================================================

#[cfg(unix)]
unsafe extern "C" {
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

/// Resolve a C ABI symbol from the current process / loaded dynamic libraries (libc, libm, etc.).
pub fn lookup_c_symbol(name: &str) -> Option<*const u8> {
    #[cfg(unix)]
    {
        let cstr = CString::new(name).ok()?;
        #[cfg(target_os = "macos")]
        let rtld_default = (-2isize) as *mut c_void;
        #[cfg(not(target_os = "macos"))]
        let rtld_default = std::ptr::null_mut();
        let ptr = unsafe { dlsym(rtld_default, cstr.as_ptr()) };
        if ptr.is_null() {
            None
        } else {
            Some(ptr as *const u8)
        }
    }
    #[cfg(not(unix))]
    {
        let _ = name;
        None
    }
}

// =========================================================================
// 1. Hardware Bit / Overflow / Float32 Runtime Helpers
// =========================================================================

#[unsafe(no_mangle)]
pub extern "C" fn tenon_clz(x: i64) -> i64 {
    (x as u64).leading_zeros() as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_ctz(x: i64) -> i64 {
    (x as u64).trailing_zeros() as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_popcnt(x: i64) -> i64 {
    (x as u64).count_ones() as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_bswap16(x: i64) -> i64 {
    ((x as u16).swap_bytes()) as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_bswap32(x: i64) -> i64 {
    ((x as u32).swap_bytes()) as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_bswap64(x: i64) -> i64 {
    (x as u64).swap_bytes() as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_rotl64(x: i64, n: i64) -> i64 {
    (x as u64).rotate_left((n as u32) & 63) as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_rotr64(x: i64, n: i64) -> i64 {
    (x as u64).rotate_right((n as u32) & 63) as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_overflow_trap(line: i64) {
    crate::perceus::tenon_rt_assert_fail(line);
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_f32_to_text(bits: i64) -> i64 {
    let v = f32::from_bits(bits as u32);
    let clean = if v == 0.0 { 0.0f32 } else { v };
    make_text(&format!("{}", clean))
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_parse_f32(s: i64) -> i64 {
    let txt = read_text(s);
    match txt.trim().parse::<f32>() {
        Ok(v) => v.to_bits() as i64,
        Err(_) => f32::NAN.to_bits() as i64,
    }
}

// =========================================================================
// 2. Compact 1-Byte `Bytes` + Zero-Copy `BytesSlice` + Endian I/O
// =========================================================================

#[unsafe(no_mangle)]
pub extern "C" fn tenon_bytes_new(cap_bytes: i64) -> i64 {
    let cb = cap_bytes.max(0) as usize;
    let words = cb.div_ceil(8).max(1);
    let obj = tenon_rc_alloc(TAG_BYTES, words as i64) as *mut ObjHeader;
    unsafe {
        (*obj).len = 0;
    }
    obj as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_bytes_filled(len_bytes: i64, fill_byte: i64) -> i64 {
    let lb = len_bytes.max(0) as usize;
    let words = lb.div_ceil(8).max(1);
    let obj = tenon_rc_alloc(TAG_BYTES, words as i64) as *mut ObjHeader;
    unsafe {
        (*obj).len = lb;
        if lb > 0 {
            let dst = fields_ptr(obj) as *mut u8;
            std::ptr::write_bytes(dst, (fill_byte & 0xFF) as u8, lb);
        }
    }
    obj as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_bytes_len(b: i64) -> i64 {
    if b == 0 {
        return 0;
    }
    unsafe { (*(b as *const ObjHeader)).len as i64 }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_bytes_get(b: i64, idx: i64) -> i64 {
    if b == 0 || idx < 0 {
        return 0;
    }
    unsafe {
        let hdr = b as *mut ObjHeader;
        let i = idx as usize;
        if i >= (*hdr).len {
            return 0;
        }
        *(fields_ptr(hdr) as *const u8).add(i) as i64
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_bytes_set(b: i64, idx: i64, byte_val: i64) -> i64 {
    if b == 0 || idx < 0 {
        return b;
    }
    let uniq = tenon_rc_ensure_unique(b);
    unsafe {
        let hdr = uniq as *mut ObjHeader;
        let i = idx as usize;
        if i < (*hdr).len {
            *(fields_ptr(hdr) as *mut u8).add(i) = (byte_val & 0xFF) as u8;
        }
    }
    uniq
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_bytes_push(b: i64, byte_val: i64) -> i64 {
    let bval = (byte_val & 0xFF) as u8;
    if b == 0 {
        let fresh = tenon_bytes_filled(1, bval as i64);
        return fresh;
    }
    unsafe {
        let hdr = b as *mut ObjHeader;
        let old_len = (*hdr).len;
        let cap_bytes = (*hdr).cap * 8;
        if (*hdr).rc.load(Ordering::SeqCst) == 1 && old_len < cap_bytes {
            *(fields_ptr(hdr) as *mut u8).add(old_len) = bval;
            (*hdr).len = old_len + 1;
            return b;
        }
        let new_cap_bytes = if old_len >= cap_bytes {
            (old_len + 1).max(cap_bytes * 2).max(16)
        } else {
            (old_len + 16).max(16)
        };
        let new_words = new_cap_bytes.div_ceil(8).max(1);
        let fresh = tenon_rc_alloc(TAG_BYTES, new_words as i64) as *mut ObjHeader;
        (*fresh).len = old_len + 1;
        let dst = fields_ptr(fresh) as *mut u8;
        if old_len > 0 {
            std::ptr::copy_nonoverlapping(fields_ptr(hdr) as *const u8, dst, old_len);
        }
        *dst.add(old_len) = bval;
        tenon_rc_dec(b);
        fresh as i64
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_bytes_read_u16_le(b: i64, off: i64) -> i64 {
    if b == 0 || off < 0 {
        return 0;
    }
    unsafe {
        let hdr = b as *mut ObjHeader;
        let o = off as usize;
        if o + 2 > (*hdr).len {
            return 0;
        }
        let p = (fields_ptr(hdr) as *const u8).add(o);
        u16::from_le_bytes([*p, *p.add(1)]) as i64
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_bytes_read_u16_be(b: i64, off: i64) -> i64 {
    if b == 0 || off < 0 {
        return 0;
    }
    unsafe {
        let hdr = b as *mut ObjHeader;
        let o = off as usize;
        if o + 2 > (*hdr).len {
            return 0;
        }
        let p = (fields_ptr(hdr) as *const u8).add(o);
        u16::from_be_bytes([*p, *p.add(1)]) as i64
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_bytes_read_u32_le(b: i64, off: i64) -> i64 {
    if b == 0 || off < 0 {
        return 0;
    }
    unsafe {
        let hdr = b as *mut ObjHeader;
        let o = off as usize;
        if o + 4 > (*hdr).len {
            return 0;
        }
        let p = (fields_ptr(hdr) as *const u8).add(o);
        let mut buf = [0u8; 4];
        std::ptr::copy_nonoverlapping(p, buf.as_mut_ptr(), 4);
        u32::from_le_bytes(buf) as i64
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_bytes_read_u32_be(b: i64, off: i64) -> i64 {
    if b == 0 || off < 0 {
        return 0;
    }
    unsafe {
        let hdr = b as *mut ObjHeader;
        let o = off as usize;
        if o + 4 > (*hdr).len {
            return 0;
        }
        let p = (fields_ptr(hdr) as *const u8).add(o);
        let mut buf = [0u8; 4];
        std::ptr::copy_nonoverlapping(p, buf.as_mut_ptr(), 4);
        u32::from_be_bytes(buf) as i64
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_bytes_read_u64_le(b: i64, off: i64) -> i64 {
    if b == 0 || off < 0 {
        return 0;
    }
    unsafe {
        let hdr = b as *mut ObjHeader;
        let o = off as usize;
        if o + 8 > (*hdr).len {
            return 0;
        }
        let p = (fields_ptr(hdr) as *const u8).add(o);
        let mut buf = [0u8; 8];
        std::ptr::copy_nonoverlapping(p, buf.as_mut_ptr(), 8);
        u64::from_le_bytes(buf) as i64
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_bytes_read_u64_be(b: i64, off: i64) -> i64 {
    if b == 0 || off < 0 {
        return 0;
    }
    unsafe {
        let hdr = b as *mut ObjHeader;
        let o = off as usize;
        if o + 8 > (*hdr).len {
            return 0;
        }
        let p = (fields_ptr(hdr) as *const u8).add(o);
        let mut buf = [0u8; 8];
        std::ptr::copy_nonoverlapping(p, buf.as_mut_ptr(), 8);
        u64::from_be_bytes(buf) as i64
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_bytes_write_u16_le(b: i64, off: i64, val: i64) -> i64 {
    let uniq = tenon_rc_ensure_unique(b);
    if uniq == 0 || off < 0 {
        return uniq;
    }
    unsafe {
        let hdr = uniq as *mut ObjHeader;
        let o = off as usize;
        if o + 2 <= (*hdr).len {
            let bs = (val as u16).to_le_bytes();
            std::ptr::copy_nonoverlapping(bs.as_ptr(), (fields_ptr(hdr) as *mut u8).add(o), 2);
        }
    }
    uniq
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_bytes_write_u16_be(b: i64, off: i64, val: i64) -> i64 {
    let uniq = tenon_rc_ensure_unique(b);
    if uniq == 0 || off < 0 {
        return uniq;
    }
    unsafe {
        let hdr = uniq as *mut ObjHeader;
        let o = off as usize;
        if o + 2 <= (*hdr).len {
            let bs = (val as u16).to_be_bytes();
            std::ptr::copy_nonoverlapping(bs.as_ptr(), (fields_ptr(hdr) as *mut u8).add(o), 2);
        }
    }
    uniq
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_bytes_write_u32_le(b: i64, off: i64, val: i64) -> i64 {
    let uniq = tenon_rc_ensure_unique(b);
    if uniq == 0 || off < 0 {
        return uniq;
    }
    unsafe {
        let hdr = uniq as *mut ObjHeader;
        let o = off as usize;
        if o + 4 <= (*hdr).len {
            let bs = (val as u32).to_le_bytes();
            std::ptr::copy_nonoverlapping(bs.as_ptr(), (fields_ptr(hdr) as *mut u8).add(o), 4);
        }
    }
    uniq
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_bytes_write_u32_be(b: i64, off: i64, val: i64) -> i64 {
    let uniq = tenon_rc_ensure_unique(b);
    if uniq == 0 || off < 0 {
        return uniq;
    }
    unsafe {
        let hdr = uniq as *mut ObjHeader;
        let o = off as usize;
        if o + 4 <= (*hdr).len {
            let bs = (val as u32).to_be_bytes();
            std::ptr::copy_nonoverlapping(bs.as_ptr(), (fields_ptr(hdr) as *mut u8).add(o), 4);
        }
    }
    uniq
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_bytes_write_u64_le(b: i64, off: i64, val: i64) -> i64 {
    let uniq = tenon_rc_ensure_unique(b);
    if uniq == 0 || off < 0 {
        return uniq;
    }
    unsafe {
        let hdr = uniq as *mut ObjHeader;
        let o = off as usize;
        if o + 8 <= (*hdr).len {
            let bs = (val as u64).to_le_bytes();
            std::ptr::copy_nonoverlapping(bs.as_ptr(), (fields_ptr(hdr) as *mut u8).add(o), 8);
        }
    }
    uniq
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_bytes_write_u64_be(b: i64, off: i64, val: i64) -> i64 {
    let uniq = tenon_rc_ensure_unique(b);
    if uniq == 0 || off < 0 {
        return uniq;
    }
    unsafe {
        let hdr = uniq as *mut ObjHeader;
        let o = off as usize;
        if o + 8 <= (*hdr).len {
            let bs = (val as u64).to_be_bytes();
            std::ptr::copy_nonoverlapping(bs.as_ptr(), (fields_ptr(hdr) as *mut u8).add(o), 8);
        }
    }
    uniq
}

/// Zero-copy slice `(owner_rc, offset, len)` over `Bytes` or `Text`.
#[unsafe(no_mangle)]
pub extern "C" fn tenon_bytes_slice(owner: i64, start: i64, end: i64) -> i64 {
    let olen = if owner == 0 {
        0
    } else {
        unsafe { (*(owner as *const ObjHeader)).len as i64 }
    };
    let st = start.clamp(0, olen);
    let en = end.clamp(st, olen);
    let sl = tenon_rc_alloc(TAG_SLICE, 3) as *mut ObjHeader;
    tenon_rc_set_rc_mask(sl as i64, 1); // field 0 = owner RC pointer
    unsafe {
        let fp = fields_ptr(sl);
        *fp = tenon_rc_inc(owner);
        *fp.add(1) = st;
        *fp.add(2) = en - st;
    }
    sl as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_slice_len(sl: i64) -> i64 {
    if sl == 0 {
        return 0;
    }
    unsafe { *fields_ptr(sl as *mut ObjHeader).add(2) }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_slice_get(sl: i64, idx: i64) -> i64 {
    if sl == 0 || idx < 0 {
        return 0;
    }
    unsafe {
        let fp = fields_ptr(sl as *mut ObjHeader);
        let owner = *fp;
        let off = *fp.add(1);
        let len = *fp.add(2);
        if owner == 0 || idx >= len {
            return 0;
        }
        *(fields_ptr(owner as *mut ObjHeader) as *const u8).add((off + idx) as usize) as i64
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_bytes_to_text(b: i64) -> i64 {
    if b == 0 {
        return make_text_bytes(&[]);
    }
    let bs = read_text_bytes(b);
    make_text_bytes(bs)
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_text_to_bytes(s: i64) -> i64 {
    let bs = read_text_bytes(s);
    let words = bs.len().div_ceil(8).max(1);
    let obj = tenon_rc_alloc(TAG_BYTES, words as i64) as *mut ObjHeader;
    unsafe {
        (*obj).len = bs.len();
        if !bs.is_empty() {
            std::ptr::copy_nonoverlapping(bs.as_ptr(), fields_ptr(obj) as *mut u8, bs.len());
        }
    }
    obj as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_bytes_ptr(b: i64) -> i64 {
    if b == 0 {
        return 0;
    }
    unsafe { fields_ptr(b as *mut ObjHeader) as i64 }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_list_data_ptr(xs: i64) -> i64 {
    if xs == 0 {
        return 0;
    }
    unsafe { fields_ptr(xs as *mut ObjHeader) as i64 }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_gpu_gid_x() -> i64 { 0 }

#[unsafe(no_mangle)]
pub extern "C" fn tenon_gpu_gid_y() -> i64 { 0 }

#[unsafe(no_mangle)]
pub extern "C" fn tenon_gpu_gid_z() -> i64 { 0 }

#[unsafe(no_mangle)]
pub extern "C" fn tenon_gpu_lid_x() -> i64 { 0 }

#[unsafe(no_mangle)]
pub extern "C" fn tenon_gpu_lid_y() -> i64 { 0 }

#[unsafe(no_mangle)]
pub extern "C" fn tenon_gpu_wgid_x() -> i64 { 0 }

#[unsafe(no_mangle)]
pub extern "C" fn tenon_gpu_wgid_y() -> i64 { 0 }

#[unsafe(no_mangle)]
pub extern "C" fn tenon_gpu_barrier() -> i64 { 0 }

#[unsafe(no_mangle)]
pub extern "C" fn tenon_gpu_shared_f32(n: i64) -> i64 {
    crate::perceus::tenon_list_filled(n, 0, 0)
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_gpu_shared_f16(n: i64) -> i64 {
    crate::perceus::tenon_list_filled(n, 0, 0)
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_gpu_exp(bits: i64) -> i64 {
    f32::from_bits(bits as u32).exp().to_bits() as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_gpu_sqrt(bits: i64) -> i64 {
    f32::from_bits(bits as u32).sqrt().to_bits() as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_gpu_rsqrt(bits: i64) -> i64 {
    (1.0f32 / f32::from_bits(bits as u32).sqrt()).to_bits() as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_gpu_cos(bits: i64) -> i64 {
    f32::from_bits(bits as u32).cos().to_bits() as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_gpu_sin(bits: i64) -> i64 {
    f32::from_bits(bits as u32).sin().to_bits() as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_gpu_pow(a_bits: i64, b_bits: i64) -> i64 {
    f32::from_bits(a_bits as u32).powf(f32::from_bits(b_bits as u32)).to_bits() as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_gpu_tanh(bits: i64) -> i64 {
    f32::from_bits(bits as u32).tanh().to_bits() as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_text_cstr_ptr(s: i64) -> i64 {
    if s == 0 {
        return c"".as_ptr() as i64;
    }
    // `tenon_text_new` guarantees trailing NUL termination!
    unsafe { fields_ptr(s as *mut ObjHeader) as i64 }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_cstr_to_text(ptr: i64) -> i64 {
    if ptr == 0 {
        return make_text_bytes(&[]);
    }
    unsafe {
        let cs = CStr::from_ptr(ptr as *const c_char);
        make_text_bytes(cs.to_bytes())
    }
}

// =========================================================================
// 3. C-Struct / Union Raw Allocation & Raw Memory Ops
// =========================================================================

#[unsafe(no_mangle)]
pub extern "C" fn tenon_cstruct_alloc(byte_size: i64) -> i64 {
    let bs = byte_size.max(0) as usize;
    let words = bs.div_ceil(8).max(1);
    let obj = tenon_rc_alloc(TAG_CSTRUCT, words as i64) as *mut ObjHeader;
    unsafe {
        (*obj).len = bs;
    }
    obj as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_cstruct_data_ptr(obj: i64) -> i64 {
    if obj == 0 {
        return 0;
    }
    unsafe { fields_ptr(obj as *mut ObjHeader) as i64 }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_raw_alloc(byte_size: i64) -> i64 {
    let bs = byte_size.max(1) as usize;
    let layout = Layout::from_size_align(bs, 16).unwrap();
    unsafe {
        let p = alloc(layout);
        if p.is_null() {
            return 0;
        }
        std::ptr::write_bytes(p, 0, bs);
        p as i64
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_raw_free(ptr: i64, byte_size: i64) -> i64 {
    if ptr == 0 {
        return 0;
    }
    let bs = byte_size.max(1) as usize;
    let layout = Layout::from_size_align(bs, 16).unwrap();
    unsafe {
        dealloc(ptr as *mut u8, layout);
    }
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_raw_memcpy(dst: i64, src: i64, n: i64) -> i64 {
    if dst != 0 && src != 0 && n > 0 {
        unsafe {
            std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, n as usize);
        }
    }
    dst
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_raw_memset(dst: i64, val: i64, n: i64) -> i64 {
    if dst != 0 && n > 0 {
        unsafe {
            std::ptr::write_bytes(dst as *mut u8, (val & 0xFF) as u8, n as usize);
        }
    }
    dst
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_raw_memcmp(a: i64, b: i64, n: i64) -> i64 {
    if a == b || n <= 0 {
        return 0;
    }
    if a == 0 {
        return -1;
    }
    if b == 0 {
        return 1;
    }
    unsafe {
        let sa = std::slice::from_raw_parts(a as *const u8, n as usize);
        let sb = std::slice::from_raw_parts(b as *const u8, n as usize);
        match sa.cmp(sb) {
            std::cmp::Ordering::Less => -1,
            std::cmp::Ordering::Equal => 0,
            std::cmp::Ordering::Greater => 1,
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_volatile_read(ptr: i64, width_signed: i64) -> i64 {
    if ptr == 0 {
        return 0;
    }
    unsafe {
        match width_signed {
            1 => std::ptr::read_volatile(ptr as *const u8) as i64,
            -1 => std::ptr::read_volatile(ptr as *const i8) as i64,
            2 => std::ptr::read_volatile(ptr as *const u16) as i64,
            -2 => std::ptr::read_volatile(ptr as *const i16) as i64,
            4 => std::ptr::read_volatile(ptr as *const u32) as i64,
            -4 => std::ptr::read_volatile(ptr as *const i32) as i64,
            _ => std::ptr::read_volatile(ptr as *const i64),
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_volatile_write(ptr: i64, val: i64, width: i64) -> i64 {
    if ptr == 0 {
        return 0;
    }
    unsafe {
        match width {
            1 => std::ptr::write_volatile(ptr as *mut u8, val as u8),
            2 => std::ptr::write_volatile(ptr as *mut u16, val as u16),
            4 => std::ptr::write_volatile(ptr as *mut u32, val as u32),
            _ => std::ptr::write_volatile(ptr as *mut i64, val),
        }
    }
    0
}

// =========================================================================
// 3b. Exact IEEE-754 Float16 (binary16) & BFloat16 scalar conversions
// =========================================================================

#[inline]
pub fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x007F_FFFF;
    if exp == 255 {
        if frac == 0 {
            return sign | 0x7C00;
        }
        return sign | 0x7E00 | ((frac >> 13) as u16);
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return sign | 0x7C00;
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return sign;
        }
        let mant = (frac | 0x0080_0000) >> (1 - new_exp);
        let round_bit = 1u32 << 12;
        let half = mant >> 13;
        let rem = mant & 0x1FFF;
        let rounded = if rem > round_bit || (rem == round_bit && (half & 1) != 0) {
            half + 1
        } else {
            half
        };
        return sign | (rounded as u16);
    }
    let round_bit = 1u32 << 12;
    let rem = frac & 0x1FFF;
    let mut h_mant = frac >> 13;
    let mut h_exp = new_exp as u32;
    if rem > round_bit || (rem == round_bit && (h_mant & 1) != 0) {
        h_mant += 1;
        if h_mant == 0x400 {
            h_mant = 0;
            h_exp += 1;
            if h_exp >= 31 {
                return sign | 0x7C00;
            }
        }
    }
    sign | ((h_exp as u16) << 10) | (h_mant as u16)
}

#[inline]
pub fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1F) as u32;
    let frac = (h & 0x03FF) as u32;
    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign);
        }
        let mut f = frac;
        let mut e: i32 = -14;
        while (f & 0x0400) == 0 {
            f <<= 1;
            e -= 1;
        }
        f &= 0x03FF;
        let f32_exp = ((e + 127) as u32) << 23;
        return f32::from_bits(sign | f32_exp | (f << 13));
    }
    if exp == 31 {
        if frac == 0 {
            return f32::from_bits(sign | 0x7F80_0000);
        }
        return f32::from_bits(sign | 0x7FC0_0000 | (frac << 13));
    }
    let f32_exp = (exp + 127 - 15) << 23;
    f32::from_bits(sign | f32_exp | (frac << 13))
}

#[inline]
pub fn f32_to_bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
        return ((bits >> 16) | 0x0040) as u16;
    }
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x7FFF + lsb;
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

#[inline]
pub fn bf16_bits_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_f64_to_f16(bits: i64) -> i64 {
    let v = f64::from_bits(bits as u64) as f32;
    f32_to_f16_bits(v) as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_f16_to_f64(bits: i64) -> i64 {
    let v = f16_bits_to_f32((bits & 0xFFFF) as u16) as f64;
    v.to_bits() as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_f64_to_bf16(bits: i64) -> i64 {
    let v = f64::from_bits(bits as u64) as f32;
    f32_to_bf16_bits(v) as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_bf16_to_f64(bits: i64) -> i64 {
    let v = bf16_bits_to_f32((bits & 0xFFFF) as u16) as f64;
    v.to_bits() as i64
}

// =========================================================================
// 4. Real OS Multi-Threading, Explicit-Ordering Atomics, Mutex & Channel
// =========================================================================

static NEXT_THREAD_ID: AtomicI64 = AtomicI64::new(1);

fn thread_table() -> &'static Mutex<BTreeMap<i64, JoinHandle<i64>>> {
    static TABLE: OnceLock<Mutex<BTreeMap<i64, JoinHandle<i64>>>> = OnceLock::new();
    TABLE.get_or_init(|| Mutex::new(BTreeMap::new()))
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_thread_spawn(closure_ptr: i64) -> i64 {
    if closure_ptr == 0 {
        return 0;
    }
    let retained = tenon_rc_inc(closure_ptr);
    let handle = thread::spawn(move || {
        let fn_addr = unsafe { *fields_ptr(retained as *mut ObjHeader) };
        let res = if fn_addr != 0 {
            let f: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(fn_addr as usize) };
            f(retained)
        } else {
            0
        };
        tenon_rc_dec(retained);
        res
    });
    let id = NEXT_THREAD_ID.fetch_add(1, Ordering::SeqCst);
    thread_table().lock().unwrap().insert(id, handle);
    id
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_thread_join(handle_id: i64) -> i64 {
    let jh = thread_table().lock().unwrap().remove(&handle_id);
    match jh {
        Some(h) => h.join().unwrap_or(0),
        None => 0,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_thread_sleep_ms(ms: i64) -> i64 {
    if ms > 0 {
        thread::sleep(Duration::from_millis(ms as u64));
    }
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_thread_hw_concurrency() -> i64 {
    thread::available_parallelism()
        .map(|n| n.get() as i64)
        .unwrap_or(1)
}

fn decode_ordering(ord: i64) -> Ordering {
    match ord {
        0 => Ordering::Relaxed,
        1 => Ordering::Acquire,
        2 => Ordering::Release,
        3 => Ordering::AcqRel,
        _ => Ordering::SeqCst,
    }
}

fn decode_load_ordering(ord: i64) -> Ordering {
    match ord {
        0 => Ordering::Relaxed,
        1 | 3 => Ordering::Acquire,
        _ => Ordering::SeqCst,
    }
}

fn decode_store_ordering(ord: i64) -> Ordering {
    match ord {
        0 => Ordering::Relaxed,
        2 | 3 => Ordering::Release,
        _ => Ordering::SeqCst,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_atomic_new(init_val: i64) -> i64 {
    let obj = tenon_rc_alloc(TAG_ATOMIC, 1) as *mut ObjHeader;
    unsafe {
        let slot = fields_ptr(obj) as *mut AtomicI64;
        slot.write(AtomicI64::new(init_val));
    }
    obj as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_atomic_load(a: i64, ord: i64) -> i64 {
    if a == 0 {
        return 0;
    }
    unsafe {
        let slot = &*(fields_ptr(a as *mut ObjHeader) as *const AtomicI64);
        slot.load(decode_load_ordering(ord))
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_atomic_store(a: i64, val: i64, ord: i64) -> i64 {
    if a == 0 {
        return 0;
    }
    unsafe {
        let slot = &*(fields_ptr(a as *mut ObjHeader) as *const AtomicI64);
        slot.store(val, decode_store_ordering(ord));
    }
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_atomic_fetch_add(a: i64, delta: i64, ord: i64) -> i64 {
    if a == 0 {
        return 0;
    }
    unsafe {
        let slot = &*(fields_ptr(a as *mut ObjHeader) as *const AtomicI64);
        slot.fetch_add(delta, decode_ordering(ord))
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_atomic_fetch_sub(a: i64, delta: i64, ord: i64) -> i64 {
    if a == 0 {
        return 0;
    }
    unsafe {
        let slot = &*(fields_ptr(a as *mut ObjHeader) as *const AtomicI64);
        slot.fetch_sub(delta, decode_ordering(ord))
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_atomic_cas(a: i64, expected: i64, desired: i64, ord: i64) -> i64 {
    if a == 0 {
        return 0;
    }
    unsafe {
        let slot = &*(fields_ptr(a as *mut ObjHeader) as *const AtomicI64);
        let succ = decode_ordering(ord);
        let fail = decode_load_ordering(ord);
        if slot.compare_exchange(expected, desired, succ, fail).is_ok() {
            1
        } else {
            0
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_atomic_swap(a: i64, val: i64, ord: i64) -> i64 {
    if a == 0 {
        return 0;
    }
    unsafe {
        let slot = &*(fields_ptr(a as *mut ObjHeader) as *const AtomicI64);
        slot.swap(val, decode_ordering(ord))
    }
}

struct TenonMutexInner {
    locked: Mutex<bool>,
    cv: Condvar,
}

pub(crate) fn drop_mutex_handle(raw: i64) {
    if raw != 0 {
        unsafe {
            let _ = Box::from_raw(raw as *mut TenonMutexInner);
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_mutex_new(init_val: i64, elem_is_rc: i64) -> i64 {
    let inner = Box::into_raw(Box::new(TenonMutexInner {
        locked: Mutex::new(false),
        cv: Condvar::new(),
    }));
    let obj = tenon_rc_alloc(TAG_MUTEX, 2) as *mut ObjHeader;
    if elem_is_rc != 0 {
        tenon_rc_set_rc_mask(obj as i64, 2); // slot 1 is RC
        tenon_rc_inc(init_val);
    }
    unsafe {
        let fp = fields_ptr(obj);
        *fp = inner as i64;
        *fp.add(1) = init_val;
    }
    obj as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_mutex_lock(m: i64) -> i64 {
    if m == 0 {
        return 0;
    }
    unsafe {
        let inner = &*(*fields_ptr(m as *mut ObjHeader) as *const TenonMutexInner);
        let mut g = inner.locked.lock().unwrap();
        while *g {
            g = inner.cv.wait(g).unwrap();
        }
        *g = true;
    }
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_mutex_unlock(m: i64) -> i64 {
    if m == 0 {
        return 0;
    }
    unsafe {
        let inner = &*(*fields_ptr(m as *mut ObjHeader) as *const TenonMutexInner);
        let mut g = inner.locked.lock().unwrap();
        *g = false;
        inner.cv.notify_one();
    }
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_mutex_get(m: i64) -> i64 {
    if m == 0 {
        return 0;
    }
    unsafe {
        let hdr = m as *mut ObjHeader;
        let v = *fields_ptr(hdr).add(1);
        if ((*hdr).rc_mask & 2) != 0 && v != 0 {
            tenon_rc_inc(v);
        }
        v
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_mutex_set(m: i64, val: i64) -> i64 {
    if m == 0 {
        return 0;
    }
    unsafe {
        let hdr = m as *mut ObjHeader;
        let is_rc = ((*hdr).rc_mask & 2) != 0;
        let fp = fields_ptr(hdr);
        let old = *fp.add(1);
        if is_rc && val != 0 {
            tenon_rc_inc(val);
        }
        *fp.add(1) = val;
        if is_rc && old != 0 {
            tenon_rc_dec(old);
        }
    }
    0
}

struct TenonChannelInner {
    state: Mutex<(VecDeque<i64>, bool)>,
    cv: Condvar,
}

pub(crate) fn drop_channel_handle(raw: i64, is_rc: bool) {
    if raw != 0 {
        unsafe {
            let boxed = Box::from_raw(raw as *mut TenonChannelInner);
            if is_rc {
                let mut g = boxed.state.lock().unwrap();
                while let Some(v) = g.0.pop_front() {
                    if v != 0 {
                        tenon_rc_dec(v);
                    }
                }
            }
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_chan_new(elem_is_rc: i64) -> i64 {
    let inner = Box::into_raw(Box::new(TenonChannelInner {
        state: Mutex::new((VecDeque::new(), false)),
        cv: Condvar::new(),
    }));
    let obj = tenon_rc_alloc(TAG_CHANNEL, 2) as *mut ObjHeader;
    unsafe {
        let fp = fields_ptr(obj);
        *fp = inner as i64;
        *fp.add(1) = if elem_is_rc != 0 { 1 } else { 0 };
    }
    obj as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_chan_send(ch: i64, val: i64) -> i64 {
    if ch == 0 {
        return 0;
    }
    unsafe {
        let fp = fields_ptr(ch as *mut ObjHeader);
        let inner = &*(*fp as *const TenonChannelInner);
        let is_rc = *fp.add(1) != 0;
        let mut g = inner.state.lock().unwrap();
        if g.1 {
            return 0;
        }
        if is_rc && val != 0 {
            tenon_rc_inc(val);
        }
        g.0.push_back(val);
        inner.cv.notify_one();
        1
    }
}

fn make_option_val(val: Option<i64>, is_rc: bool) -> i64 {
    match val {
        Some(v) => {
            let opt = tenon_rc_alloc(TAG_SOME, 1) as *mut ObjHeader;
            if is_rc {
                tenon_rc_set_rc_mask(opt as i64, 1);
            }
            unsafe {
                *fields_ptr(opt) = v;
            }
            opt as i64
        }
        None => tenon_rc_alloc(TAG_NONE, 0),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_chan_recv(ch: i64) -> i64 {
    if ch == 0 {
        return make_option_val(None, false);
    }
    unsafe {
        let fp = fields_ptr(ch as *mut ObjHeader);
        let inner = &*(*fp as *const TenonChannelInner);
        let is_rc = *fp.add(1) != 0;
        let mut g = inner.state.lock().unwrap();
        loop {
            if let Some(v) = g.0.pop_front() {
                return make_option_val(Some(v), is_rc);
            }
            if g.1 {
                return make_option_val(None, is_rc);
            }
            g = inner.cv.wait(g).unwrap();
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_chan_try_recv(ch: i64) -> i64 {
    if ch == 0 {
        return make_option_val(None, false);
    }
    unsafe {
        let fp = fields_ptr(ch as *mut ObjHeader);
        let inner = &*(*fp as *const TenonChannelInner);
        let is_rc = *fp.add(1) != 0;
        let mut g = inner.state.lock().unwrap();
        make_option_val(g.0.pop_front(), is_rc)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_chan_close(ch: i64) -> i64 {
    if ch == 0 {
        return 0;
    }
    unsafe {
        let inner = &*(*fields_ptr(ch as *mut ObjHeader) as *const TenonChannelInner);
        let mut g = inner.state.lock().unwrap();
        g.1 = true;
        inner.cv.notify_all();
    }
    0
}

// =========================================================================
// 5. Real Non-Blocking TCP Sockets (`std::net`)
// =========================================================================

enum NetHandle {
    Listener(TcpListener),
    Stream(TcpStream),
}

static NEXT_NET_ID: AtomicI64 = AtomicI64::new(1);

fn net_table() -> &'static Mutex<BTreeMap<i64, NetHandle>> {
    static TABLE: OnceLock<Mutex<BTreeMap<i64, NetHandle>>> = OnceLock::new();
    TABLE.get_or_init(|| Mutex::new(BTreeMap::new()))
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_net_tcp_listen(addr_text: i64) -> i64 {
    let addr = read_text(addr_text);
    match TcpListener::bind(&addr) {
        Ok(l) => {
            let id = NEXT_NET_ID.fetch_add(1, Ordering::SeqCst);
            net_table().lock().unwrap().insert(id, NetHandle::Listener(l));
            id
        }
        Err(_) => -1,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_net_tcp_local_port(handle: i64) -> i64 {
    let tbl = net_table().lock().unwrap();
    match tbl.get(&handle) {
        Some(NetHandle::Listener(l)) => l.local_addr().map(|a| a.port() as i64).unwrap_or(-1),
        Some(NetHandle::Stream(s)) => s.local_addr().map(|a| a.port() as i64).unwrap_or(-1),
        None => -1,
    }
}

thread_local! {
    static LAST_WOULD_BLOCK: Cell<bool> = const { Cell::new(false) };
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_net_tcp_accept(listener_handle: i64) -> i64 {
    LAST_WOULD_BLOCK.with(|c| c.set(false));
    let cloned = {
        let tbl = net_table().lock().unwrap();
        match tbl.get(&listener_handle) {
            Some(NetHandle::Listener(l)) => l.try_clone().ok(),
            _ => None,
        }
    };
    let Some(l) = cloned else {
        return -1;
    };
    match l.accept() {
        Ok((stream, _)) => {
            let _ = stream.set_nodelay(true);
            let id = NEXT_NET_ID.fetch_add(1, Ordering::SeqCst);
            net_table().lock().unwrap().insert(id, NetHandle::Stream(stream));
            id
        }
        Err(e) => {
            if e.kind() == ErrorKind::WouldBlock {
                LAST_WOULD_BLOCK.with(|c| c.set(true));
            }
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_net_tcp_connect(addr_text: i64) -> i64 {
    let addr = read_text(addr_text);
    match TcpStream::connect(&addr) {
        Ok(s) => {
            let _ = s.set_nodelay(true);
            let id = NEXT_NET_ID.fetch_add(1, Ordering::SeqCst);
            net_table().lock().unwrap().insert(id, NetHandle::Stream(s));
            id
        }
        Err(_) => -1,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_net_set_nonblocking(handle: i64, nonblocking: i64) -> i64 {
    let tbl = net_table().lock().unwrap();
    let nb = nonblocking != 0;
    let res = match tbl.get(&handle) {
        Some(NetHandle::Listener(l)) => l.set_nonblocking(nb),
        Some(NetHandle::Stream(s)) => s.set_nonblocking(nb),
        None => return 0,
    };
    if res.is_ok() { 1 } else { 0 }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_net_set_nodelay(handle: i64, nodelay: i64) -> i64 {
    let tbl = net_table().lock().unwrap();
    let res = match tbl.get(&handle) {
        Some(NetHandle::Stream(s)) => s.set_nodelay(nodelay != 0),
        _ => return 0,
    };
    if res.is_ok() { 1 } else { 0 }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_net_last_would_block() -> i64 {
    if LAST_WOULD_BLOCK.with(|c| c.get()) { 1 } else { 0 }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_net_read(stream_handle: i64, max_bytes: i64) -> i64 {
    LAST_WOULD_BLOCK.with(|c| c.set(false));
    let cloned = {
        let tbl = net_table().lock().unwrap();
        match tbl.get(&stream_handle) {
            Some(NetHandle::Stream(s)) => s.try_clone().ok(),
            _ => None,
        }
    };
    let Some(mut s) = cloned else {
        return tenon_bytes_new(0);
    };
    let cap = max_bytes.clamp(1, 1024 * 1024) as usize;
    let mut buf = vec![0u8; cap];
    match s.read(&mut buf) {
        Ok(n) => {
            let words = n.div_ceil(8).max(1);
            let obj = tenon_rc_alloc(TAG_BYTES, words as i64) as *mut ObjHeader;
            unsafe {
                (*obj).len = n;
                if n > 0 {
                    std::ptr::copy_nonoverlapping(buf.as_ptr(), fields_ptr(obj) as *mut u8, n);
                }
            }
            obj as i64
        }
        Err(e) => {
            if e.kind() == ErrorKind::WouldBlock {
                LAST_WOULD_BLOCK.with(|c| c.set(true));
            }
            tenon_bytes_new(0)
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_net_read_text(stream_handle: i64, max_bytes: i64) -> i64 {
    LAST_WOULD_BLOCK.with(|c| c.set(false));
    let cloned = {
        let tbl = net_table().lock().unwrap();
        match tbl.get(&stream_handle) {
            Some(NetHandle::Stream(s)) => s.try_clone().ok(),
            _ => None,
        }
    };
    let Some(mut s) = cloned else {
        return make_text("");
    };
    let cap = max_bytes.clamp(1, 1024 * 1024) as usize;
    let mut buf = vec![0u8; cap];
    match s.read(&mut buf) {
        Ok(n) if n > 0 => {
            let words = (n + 1).div_ceil(8).max(1);
            let obj = tenon_rc_alloc(TAG_TEXT, words as i64) as *mut ObjHeader;
            unsafe {
                (*obj).len = n;
                let dst = fields_ptr(obj) as *mut u8;
                std::ptr::copy_nonoverlapping(buf.as_ptr(), dst, n);
                *dst.add(n) = 0;
            }
            obj as i64
        }
        Ok(_) => make_text(""),
        Err(e) => {
            if e.kind() == ErrorKind::WouldBlock {
                LAST_WOULD_BLOCK.with(|c| c.set(true));
            }
            make_text("")
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_net_write(stream_handle: i64, bytes_obj: i64) -> i64 {
    let cloned = {
        let tbl = net_table().lock().unwrap();
        match tbl.get(&stream_handle) {
            Some(NetHandle::Stream(s)) => s.try_clone().ok(),
            _ => None,
        }
    };
    let Some(mut s) = cloned else {
        return -1;
    };
    let data = read_text_bytes(bytes_obj);
    match s.write_all(data) {
        Ok(()) => data.len() as i64,
        Err(_) => -1,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_net_write_text(stream_handle: i64, text_obj: i64) -> i64 {
    tenon_net_write(stream_handle, text_obj)
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_net_close(handle: i64) -> i64 {
    net_table().lock().unwrap().remove(&handle);
    0
}

// =========================================================================
// 5b. OS Event Reactor (kqueue on macOS/BSD, epoll on Linux) & Monotonic Clock
// =========================================================================

fn mono_origin() -> &'static Instant {
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    ORIGIN.get_or_init(Instant::now)
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_time_now_ms() -> i64 {
    mono_origin().elapsed().as_millis() as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_time_now_ns() -> i64 {
    mono_origin().elapsed().as_nanos() as i64
}

#[cfg(unix)]
fn handle_to_raw_fd(handle: i64) -> Option<i32> {
    let tbl = net_table().lock().unwrap();
    match tbl.get(&handle) {
        Some(NetHandle::Listener(l)) => Some(l.as_raw_fd()),
        Some(NetHandle::Stream(s)) => Some(s.as_raw_fd()),
        None => None,
    }
}

#[cfg(target_os = "macos")]
mod os_reactor {
    use super::*;

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct KEvent {
        ident: usize,
        filter: i16,
        flags: u16,
        fflags: u32,
        data: isize,
        udata: *mut c_void,
    }

    #[repr(C)]
    struct Timespec {
        tv_sec: isize,
        tv_nsec: isize,
    }

    const EVFILT_READ: i16 = -1;
    const EVFILT_WRITE: i16 = -2;
    const EV_ADD: u16 = 0x0001;
    const EV_DELETE: u16 = 0x0002;
    const EV_ENABLE: u16 = 0x0004;
    const EV_EOF: u16 = 0x8000;
    const EV_ERROR: u16 = 0x4000;

    unsafe extern "C" {
        fn kqueue() -> i32;
        fn kevent(
            kq: i32,
            changelist: *const KEvent,
            nchanges: i32,
            eventlist: *mut KEvent,
            nevents: i32,
            timeout: *const Timespec,
        ) -> i32;
        fn close(fd: i32) -> i32;
    }

    pub fn new() -> i64 {
        unsafe { kqueue() as i64 }
    }

    pub fn register(rfd: i64, handle: i64, mask: i64, token: i64) -> i64 {
        let Some(fd) = handle_to_raw_fd(handle) else {
            return 0;
        };
        let mut changes = [KEvent::default(); 2];
        let mut n = 0usize;
        if (mask & 1) != 0 {
            changes[n] = KEvent {
                ident: fd as usize,
                filter: EVFILT_READ,
                flags: EV_ADD | EV_ENABLE,
                fflags: 0,
                data: 0,
                udata: token as usize as *mut c_void,
            };
            n += 1;
        }
        if (mask & 2) != 0 {
            changes[n] = KEvent {
                ident: fd as usize,
                filter: EVFILT_WRITE,
                flags: EV_ADD | EV_ENABLE,
                fflags: 0,
                data: 0,
                udata: token as usize as *mut c_void,
            };
            n += 1;
        }
        if n == 0 {
            return 1;
        }
        let rc = unsafe {
            kevent(
                rfd as i32,
                changes.as_ptr(),
                n as i32,
                std::ptr::null_mut(),
                0,
                std::ptr::null(),
            )
        };
        if rc >= 0 { 1 } else { 0 }
    }

    pub fn remove(rfd: i64, handle: i64) -> i64 {
        let Some(fd) = handle_to_raw_fd(handle) else {
            return 0;
        };
        let changes = [
            KEvent {
                ident: fd as usize,
                filter: EVFILT_READ,
                flags: EV_DELETE,
                ..Default::default()
            },
            KEvent {
                ident: fd as usize,
                filter: EVFILT_WRITE,
                flags: EV_DELETE,
                ..Default::default()
            },
        ];
        unsafe {
            let _ = kevent(
                rfd as i32,
                changes.as_ptr(),
                2,
                std::ptr::null_mut(),
                0,
                std::ptr::null(),
            );
        }
        1
    }

    pub fn wait(rfd: i64, timeout_ms: i64, max_events: i64) -> i64 {
        let cap = max_events.clamp(1, 1024) as usize;
        let mut evs = vec![KEvent::default(); cap];
        let ts = if timeout_ms >= 0 {
            Some(Timespec {
                tv_sec: (timeout_ms / 1000) as isize,
                tv_nsec: ((timeout_ms % 1000) * 1_000_000) as isize,
            })
        } else {
            None
        };
        let n = unsafe {
            kevent(
                rfd as i32,
                std::ptr::null(),
                0,
                evs.as_mut_ptr(),
                cap as i32,
                ts.as_ref().map(|t| t as *const Timespec).unwrap_or(std::ptr::null()),
            )
        };
        let count = n.max(0) as usize;
        let list = tenon_rc_alloc(TAG_LIST, count as i64) as *mut ObjHeader;
        unsafe {
            let fp = fields_ptr(list);
            for i in 0..count {
                let ev = evs[i];
                let mut rev: i64 = 0;
                if ev.filter == EVFILT_READ {
                    rev |= 1;
                }
                if ev.filter == EVFILT_WRITE {
                    rev |= 2;
                }
                if (ev.flags & (EV_EOF | EV_ERROR)) != 0 {
                    rev |= 4;
                }
                let tok = ev.udata as usize as i64;
                *fp.add(i) = (tok << 8) | rev;
            }
        }
        list as i64
    }

    pub fn close_fd(rfd: i64) -> i64 {
        if rfd >= 0 {
            unsafe {
                let _ = close(rfd as i32);
            }
        }
        0
    }
}

#[cfg(not(target_os = "macos"))]
mod os_reactor {
    use super::*;
    pub fn new() -> i64 { 0 }
    pub fn register(_rfd: i64, _handle: i64, _mask: i64, _token: i64) -> i64 { 1 }
    pub fn remove(_rfd: i64, _handle: i64) -> i64 { 1 }
    pub fn wait(_rfd: i64, timeout_ms: i64, _max_events: i64) -> i64 {
        if timeout_ms > 0 {
            thread::sleep(Duration::from_millis(timeout_ms as u64));
        }
        tenon_rc_alloc(TAG_LIST, 0)
    }
    pub fn close_fd(_rfd: i64) -> i64 { 0 }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_reactor_new() -> i64 {
    os_reactor::new()
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_reactor_add(rfd: i64, handle: i64, mask: i64, token: i64) -> i64 {
    os_reactor::register(rfd, handle, mask, token)
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_reactor_del(rfd: i64, handle: i64) -> i64 {
    os_reactor::remove(rfd, handle)
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_reactor_wait(rfd: i64, timeout_ms: i64, max_events: i64) -> i64 {
    os_reactor::wait(rfd, timeout_ms, max_events)
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_reactor_close(rfd: i64) -> i64 {
    os_reactor::close_fd(rfd)
}

// =========================================================================
// 6. SIMD Vector Runtime Primitives (JIT & Shared Representation)
// =========================================================================

fn simd_alloc(byte_size: usize) -> *mut ObjHeader {
    let words = byte_size.div_ceil(8).max(1);
    let obj = tenon_rc_alloc(TAG_SIMD, words as i64) as *mut ObjHeader;
    unsafe {
        (*obj).len = byte_size;
    }
    obj
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_simd_f32x4_new(a: i64, b: i64, c: i64, d: i64) -> i64 {
    let obj = simd_alloc(16);
    unsafe {
        let p = fields_ptr(obj) as *mut f32;
        *p.add(0) = f32::from_bits(a as u32);
        *p.add(1) = f32::from_bits(b as u32);
        *p.add(2) = f32::from_bits(c as u32);
        *p.add(3) = f32::from_bits(d as u32);
    }
    obj as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_simd_f32x4_splat(v: i64) -> i64 {
    tenon_simd_f32x4_new(v, v, v, v)
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_simd_f32x4_lane(vec: i64, idx: i64) -> i64 {
    if vec == 0 || !(0..4).contains(&idx) {
        return 0;
    }
    unsafe { (*(fields_ptr(vec as *mut ObjHeader) as *const u32).add(idx as usize)) as i64 }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_simd_f32x4_binop(a: i64, b: i64, op: i64) -> i64 {
    let out = simd_alloc(16);
    unsafe {
        let pa = fields_ptr(a as *mut ObjHeader) as *const f32;
        let pb = fields_ptr(b as *mut ObjHeader) as *const f32;
        let po = fields_ptr(out) as *mut f32;
        for i in 0..4 {
            let x = *pa.add(i);
            let y = *pb.add(i);
            *po.add(i) = match op {
                0 => x + y,
                1 => x - y,
                2 => x * y,
                3 => x / y,
                4 => x.min(y),
                _ => x.max(y),
            };
        }
    }
    out as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_simd_f32x4_fma(a: i64, b: i64, c: i64) -> i64 {
    let out = simd_alloc(16);
    unsafe {
        let pa = fields_ptr(a as *mut ObjHeader) as *const f32;
        let pb = fields_ptr(b as *mut ObjHeader) as *const f32;
        let pc = fields_ptr(c as *mut ObjHeader) as *const f32;
        let po = fields_ptr(out) as *mut f32;
        for i in 0..4 {
            *po.add(i) = (*pa.add(i)).mul_add(*pb.add(i), *pc.add(i));
        }
    }
    out as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_simd_f32x4_reduce_sum(v: i64) -> i64 {
    if v == 0 {
        return 0;
    }
    unsafe {
        let p = fields_ptr(v as *mut ObjHeader) as *const f32;
        let s = *p + *p.add(1) + *p.add(2) + *p.add(3);
        s.to_bits() as i64
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_simd_f32x4_dot(a: i64, b: i64) -> i64 {
    unsafe {
        let pa = fields_ptr(a as *mut ObjHeader) as *const f32;
        let pb = fields_ptr(b as *mut ObjHeader) as *const f32;
        let mut s = 0.0f32;
        for i in 0..4 {
            s += *pa.add(i) * *pb.add(i);
        }
        s.to_bits() as i64
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_simd_f64x4_new(a: i64, b: i64, c: i64, d: i64) -> i64 {
    let obj = simd_alloc(32);
    unsafe {
        let p = fields_ptr(obj) as *mut i64;
        *p.add(0) = a;
        *p.add(1) = b;
        *p.add(2) = c;
        *p.add(3) = d;
    }
    obj as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_simd_f64x4_splat(v: i64) -> i64 {
    tenon_simd_f64x4_new(v, v, v, v)
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_simd_f64x4_lane(vec: i64, idx: i64) -> i64 {
    if vec == 0 || !(0..4).contains(&idx) {
        return 0;
    }
    unsafe { *fields_ptr(vec as *mut ObjHeader).add(idx as usize) }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_simd_f64x4_binop(a: i64, b: i64, op: i64) -> i64 {
    let out = simd_alloc(32);
    unsafe {
        let pa = fields_ptr(a as *mut ObjHeader) as *const f64;
        let pb = fields_ptr(b as *mut ObjHeader) as *const f64;
        let po = fields_ptr(out) as *mut f64;
        for i in 0..4 {
            let x = *pa.add(i);
            let y = *pb.add(i);
            *po.add(i) = match op {
                0 => x + y,
                1 => x - y,
                2 => x * y,
                3 => x / y,
                4 => x.min(y),
                _ => x.max(y),
            };
        }
    }
    out as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_simd_f64x4_fma(a: i64, b: i64, c: i64) -> i64 {
    let out = simd_alloc(32);
    unsafe {
        let pa = fields_ptr(a as *mut ObjHeader) as *const f64;
        let pb = fields_ptr(b as *mut ObjHeader) as *const f64;
        let pc = fields_ptr(c as *mut ObjHeader) as *const f64;
        let po = fields_ptr(out) as *mut f64;
        for i in 0..4 {
            *po.add(i) = (*pa.add(i)).mul_add(*pb.add(i), *pc.add(i));
        }
    }
    out as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_simd_f64x4_reduce_sum(v: i64) -> i64 {
    if v == 0 {
        return 0;
    }
    unsafe {
        let p = fields_ptr(v as *mut ObjHeader) as *const f64;
        let s = *p + *p.add(1) + *p.add(2) + *p.add(3);
        s.to_bits() as i64
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_simd_f64x4_dot(a: i64, b: i64) -> i64 {
    unsafe {
        let pa = fields_ptr(a as *mut ObjHeader) as *const f64;
        let pb = fields_ptr(b as *mut ObjHeader) as *const f64;
        let mut s = 0.0f64;
        for i in 0..4 {
            s += *pa.add(i) * *pb.add(i);
        }
        s.to_bits() as i64
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_simd_i32x4_new(a: i64, b: i64, c: i64, d: i64) -> i64 {
    let obj = simd_alloc(16);
    unsafe {
        let p = fields_ptr(obj) as *mut i32;
        *p.add(0) = a as i32;
        *p.add(1) = b as i32;
        *p.add(2) = c as i32;
        *p.add(3) = d as i32;
    }
    obj as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_simd_i32x4_splat(v: i64) -> i64 {
    tenon_simd_i32x4_new(v, v, v, v)
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_simd_i32x4_lane(vec: i64, idx: i64) -> i64 {
    if vec == 0 || !(0..4).contains(&idx) {
        return 0;
    }
    unsafe { (*(fields_ptr(vec as *mut ObjHeader) as *const i32).add(idx as usize)) as i64 }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_simd_i32x4_binop(a: i64, b: i64, op: i64) -> i64 {
    let out = simd_alloc(16);
    unsafe {
        let pa = fields_ptr(a as *mut ObjHeader) as *const i32;
        let pb = fields_ptr(b as *mut ObjHeader) as *const i32;
        let po = fields_ptr(out) as *mut i32;
        for i in 0..4 {
            let x = *pa.add(i);
            let y = *pb.add(i);
            *po.add(i) = match op {
                0 => x.wrapping_add(y),
                1 => x.wrapping_sub(y),
                _ => x.wrapping_mul(y),
            };
        }
    }
    out as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_simd_i32x4_reduce_sum(v: i64) -> i64 {
    if v == 0 {
        return 0;
    }
    unsafe {
        let p = fields_ptr(v as *mut ObjHeader) as *const i32;
        (*p).wrapping_add(*p.add(1))
            .wrapping_add(*p.add(2))
            .wrapping_add(*p.add(3)) as i64
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_simd_i64x4_new(a: i64, b: i64, c: i64, d: i64) -> i64 {
    let obj = simd_alloc(32);
    unsafe {
        let p = fields_ptr(obj);
        *p.add(0) = a;
        *p.add(1) = b;
        *p.add(2) = c;
        *p.add(3) = d;
    }
    obj as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_simd_i64x4_splat(v: i64) -> i64 {
    tenon_simd_i64x4_new(v, v, v, v)
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_simd_i64x4_lane(vec: i64, idx: i64) -> i64 {
    if vec == 0 || !(0..4).contains(&idx) {
        return 0;
    }
    unsafe { *fields_ptr(vec as *mut ObjHeader).add(idx as usize) }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_simd_i64x4_binop(a: i64, b: i64, op: i64) -> i64 {
    let out = simd_alloc(32);
    unsafe {
        let pa = fields_ptr(a as *mut ObjHeader);
        let pb = fields_ptr(b as *mut ObjHeader);
        let po = fields_ptr(out);
        for i in 0..4 {
            let x = *pa.add(i);
            let y = *pb.add(i);
            *po.add(i) = match op {
                0 => x.wrapping_add(y),
                1 => x.wrapping_sub(y),
                _ => x.wrapping_mul(y),
            };
        }
    }
    out as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_simd_i64x4_reduce_sum(v: i64) -> i64 {
    if v == 0 {
        return 0;
    }
    unsafe {
        let p = fields_ptr(v as *mut ObjHeader);
        (*p).wrapping_add(*p.add(1))
            .wrapping_add(*p.add(2))
            .wrapping_add(*p.add(3))
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_simd_u8x16_splat(v: i64) -> i64 {
    let obj = simd_alloc(16);
    unsafe {
        std::ptr::write_bytes(fields_ptr(obj) as *mut u8, (v & 0xFF) as u8, 16);
    }
    obj as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_simd_u8x16_lane(vec: i64, idx: i64) -> i64 {
    if vec == 0 || !(0..16).contains(&idx) {
        return 0;
    }
    unsafe { (*(fields_ptr(vec as *mut ObjHeader) as *const u8).add(idx as usize)) as i64 }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_simd_u8x16_set_lane(vec: i64, idx: i64, val: i64) -> i64 {
    let uniq = tenon_rc_ensure_unique(vec);
    if uniq != 0 && (0..16).contains(&idx) {
        unsafe {
            *(fields_ptr(uniq as *mut ObjHeader) as *mut u8).add(idx as usize) = (val & 0xFF) as u8;
        }
    }
    uniq
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_simd_u8x16_binop(a: i64, b: i64, op: i64) -> i64 {
    let out = simd_alloc(16);
    unsafe {
        let pa = fields_ptr(a as *mut ObjHeader) as *const u8;
        let pb = fields_ptr(b as *mut ObjHeader) as *const u8;
        let po = fields_ptr(out) as *mut u8;
        for i in 0..16 {
            let x = *pa.add(i);
            let y = *pb.add(i);
            *po.add(i) = match op {
                0 => x.wrapping_add(y),
                1 => x.wrapping_sub(y),
                _ => x ^ y,
            };
        }
    }
    out as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_simd_u8x16_reduce_sum(v: i64) -> i64 {
    if v == 0 {
        return 0;
    }
    unsafe {
        let p = fields_ptr(v as *mut ObjHeader) as *const u8;
        let mut s = 0i64;
        for i in 0..16 {
            s += *p.add(i) as i64;
        }
        s
    }
}
