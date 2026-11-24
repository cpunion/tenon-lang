use std::alloc::{Layout, alloc, dealloc};
use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};

pub const TAG_ERR: i64 = 0;
pub const TAG_OK: i64 = 1;
pub const TAG_NONE: i64 = 2;
pub const TAG_SOME: i64 = 3;
pub const TAG_LIST: i64 = 4;
pub const TAG_TEXT: i64 = 5;
pub const TAG_ASYNC_FRAME: i64 = 6;
pub const TAG_DYN: i64 = 7;
pub const TAG_CLOSURE: i64 = 8;
pub const TAG_TUPLE: i64 = 9;
pub const TAG_BYTES: i64 = 10;
pub const TAG_SLICE: i64 = 11;
pub const TAG_CSTRUCT: i64 = 12;
pub const TAG_ATOMIC: i64 = 13;
pub const TAG_MUTEX: i64 = 14;
pub const TAG_CHANNEL: i64 = 15;
pub const TAG_SIMD: i64 = 16;
pub const TAG_ARRAY: i64 = 17;
pub const TAG_ENUM_BASE: i64 = 100;
pub const TAG_RECORD_BASE: i64 = 10_000;
/// Sentinel `rc_mask` meaning EVERY element in `0..len` is an RC pointer (used for `List[T]` of RC `T`).
pub const RC_MASK_ALL: u64 = u64::MAX;

static LIVE_OBJECTS: AtomicUsize = AtomicUsize::new(0);
static TOTAL_ALLOCS: AtomicUsize = AtomicUsize::new(0);
static REUSE_HITS: AtomicUsize = AtomicUsize::new(0);
static COW_CLONES: AtomicUsize = AtomicUsize::new(0);
static ASYNC_POLLS: AtomicUsize = AtomicUsize::new(0);
static ASYNC_SUSPENDS: AtomicUsize = AtomicUsize::new(0);
static LAST_ASSERT_FAILED: AtomicI64 = AtomicI64::new(0);
static LAST_PRINTED: AtomicI64 = AtomicI64::new(0);

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PerceusStats {
    pub live_objects: usize,
    pub total_allocs: usize,
    pub reuse_hits: usize,
    pub cow_clones: usize,
    pub async_polls: usize,
    pub async_suspends: usize,
}

pub fn reset_stats() {
    LIVE_OBJECTS.store(0, Ordering::SeqCst);
    TOTAL_ALLOCS.store(0, Ordering::SeqCst);
    REUSE_HITS.store(0, Ordering::SeqCst);
    COW_CLONES.store(0, Ordering::SeqCst);
    ASYNC_POLLS.store(0, Ordering::SeqCst);
    ASYNC_SUSPENDS.store(0, Ordering::SeqCst);
}

pub fn snapshot_stats() -> PerceusStats {
    PerceusStats {
        live_objects: LIVE_OBJECTS.load(Ordering::SeqCst),
        total_allocs: TOTAL_ALLOCS.load(Ordering::SeqCst),
        reuse_hits: REUSE_HITS.load(Ordering::SeqCst),
        cow_clones: COW_CLONES.load(Ordering::SeqCst),
        async_polls: ASYNC_POLLS.load(Ordering::SeqCst),
        async_suspends: ASYNC_SUSPENDS.load(Ordering::SeqCst),
    }
}

pub fn clear_assert_failed() {
    LAST_ASSERT_FAILED.store(0, Ordering::SeqCst);
}

pub fn last_assert_failed() -> i64 {
    LAST_ASSERT_FAILED.load(Ordering::SeqCst)
}

pub fn last_printed() -> i64 {
    LAST_PRINTED.load(Ordering::SeqCst)
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_rt_assert_fail(line: i64) {
    LAST_ASSERT_FAILED.store(line, Ordering::SeqCst);
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_rt_print_int(val: i64) -> i64 {
    LAST_PRINTED.store(val, Ordering::SeqCst);
    println!("{}", val);
    val
}

#[repr(C)]
pub(crate) struct ObjHeader {
    pub(crate) rc: AtomicUsize,
    pub(crate) tag: i64,
    pub(crate) len: usize,
    pub(crate) cap: usize,
    pub(crate) rc_mask: u64,
}

pub(crate) fn obj_layout(cap: usize) -> Layout {
    let header_size = std::mem::size_of::<ObjHeader>();
    let fields_size = cap * std::mem::size_of::<i64>();
    Layout::from_size_align(header_size + fields_size, std::mem::align_of::<ObjHeader>()).unwrap()
}

pub(crate) unsafe fn fields_ptr(hdr: *mut ObjHeader) -> *mut i64 {
    unsafe { (hdr as *mut u8).add(std::mem::size_of::<ObjHeader>()) as *mut i64 }
}

pub(crate) fn is_byte_payload_tag(tag: i64) -> bool {
    matches!(tag, TAG_TEXT | TAG_BYTES | TAG_CSTRUCT | TAG_SIMD)
}

pub fn read_text_bytes<'a>(ptr: i64) -> &'a [u8] {
    if ptr == 0 {
        return &[];
    }
    unsafe {
        let hdr = ptr as *mut ObjHeader;
        let blen = (*hdr).len;
        std::slice::from_raw_parts(fields_ptr(hdr) as *const u8, blen)
    }
}

pub fn read_text(ptr: i64) -> String {
    String::from_utf8_lossy(read_text_bytes(ptr)).to_string()
}

pub fn make_text_bytes(bytes: &[u8]) -> i64 {
    tenon_text_new(bytes.as_ptr() as usize as i64, bytes.len() as i64)
}

pub fn make_text(s: &str) -> i64 {
    make_text_bytes(s.as_bytes())
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_rc_alloc(tag: i64, len: i64) -> i64 {
    let len_u = len.max(0) as usize;
    let cap = len_u.max(1);
    let layout = obj_layout(cap);
    unsafe {
        let raw = alloc(layout) as *mut ObjHeader;
        if raw.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        raw.write(ObjHeader {
            rc: AtomicUsize::new(1),
            tag,
            len: len_u,
            cap,
            rc_mask: 0,
        });
        let fp = fields_ptr(raw);
        for i in 0..cap {
            *fp.add(i) = 0;
        }
        LIVE_OBJECTS.fetch_add(1, Ordering::SeqCst);
        TOTAL_ALLOCS.fetch_add(1, Ordering::SeqCst);
        raw as i64
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_rc_set_rc_mask(ptr: i64, mask: i64) {
    if ptr == 0 {
        return;
    }
    unsafe {
        (*(ptr as *mut ObjHeader)).rc_mask = mask as u64;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_rc_inc(ptr: i64) -> i64 {
    if ptr == 0 {
        return 0;
    }
    unsafe {
        let hdr = ptr as *const ObjHeader;
        (*hdr).rc.fetch_add(1, Ordering::SeqCst);
    }
    ptr
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_rc_dec(ptr: i64) {
    if ptr == 0 {
        return;
    }
    unsafe {
        let hdr = ptr as *mut ObjHeader;
        let prev = (*hdr).rc.fetch_sub(1, Ordering::SeqCst);
        if prev == 1 {
            let rc_mask = (*hdr).rc_mask;
            if rc_mask == RC_MASK_ALL {
                let len = (*hdr).len;
                let fp = fields_ptr(hdr);
                for i in 0..len {
                    let child = *fp.add(i);
                    if child != 0 {
                        tenon_rc_dec(child);
                    }
                }
            } else if rc_mask != 0 {
                let len = (*hdr).len.min(64);
                let fp = fields_ptr(hdr);
                for i in 0..len {
                    if (rc_mask & (1u64 << i)) != 0 {
                        let child = *fp.add(i);
                        if child != 0 {
                            tenon_rc_dec(child);
                        }
                    }
                }
            }
            let tag = (*hdr).tag;
            if tag == TAG_MUTEX {
                let raw_mu = *fields_ptr(hdr);
                crate::systems::drop_mutex_handle(raw_mu);
            } else if tag == TAG_CHANNEL {
                let raw_ch = *fields_ptr(hdr);
                let is_rc = *fields_ptr(hdr).add(1) != 0;
                crate::systems::drop_channel_handle(raw_ch, is_rc);
            }
            let cap = (*hdr).cap;
            LIVE_OBJECTS.fetch_sub(1, Ordering::SeqCst);
            dealloc(hdr as *mut u8, obj_layout(cap));
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_rc_ensure_unique(ptr: i64) -> i64 {
    if ptr == 0 {
        return 0;
    }
    unsafe {
        let hdr = ptr as *mut ObjHeader;
        if (*hdr).rc.load(Ordering::SeqCst) == 1 {
            REUSE_HITS.fetch_add(1, Ordering::SeqCst);
            return ptr;
        }
        COW_CLONES.fetch_add(1, Ordering::SeqCst);
        let tag = (*hdr).tag;
        let len = (*hdr).len;
        let rc_mask = (*hdr).rc_mask;
        if is_byte_payload_tag(tag) {
            let extra = if tag == TAG_TEXT { 1 } else { 0 };
            let words = (len + extra).div_ceil(8).max(1);
            let fresh = tenon_rc_alloc(tag, words as i64) as *mut ObjHeader;
            (*fresh).len = len;
            (*fresh).rc_mask = 0;
            let src_b = fields_ptr(hdr) as *const u8;
            let dst_b = fields_ptr(fresh) as *mut u8;
            if len > 0 {
                std::ptr::copy_nonoverlapping(src_b, dst_b, len);
            }
            if tag == TAG_TEXT {
                *dst_b.add(len) = 0;
            }
            tenon_rc_dec(ptr);
            return fresh as i64;
        }
        let fresh = tenon_rc_alloc(tag, len as i64) as *mut ObjHeader;
        (*fresh).rc_mask = rc_mask;
        let src_f = fields_ptr(hdr);
        let dst_f = fields_ptr(fresh);
        for i in 0..len {
            let v = *src_f.add(i);
            *dst_f.add(i) = v;
            let is_rc = rc_mask == RC_MASK_ALL || (i < 64 && (rc_mask & (1u64 << i)) != 0);
            if is_rc && v != 0 {
                tenon_rc_inc(v);
            }
        }
        tenon_rc_dec(ptr);
        fresh as i64
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_rc_get_field(ptr: i64, idx: i64) -> i64 {
    if ptr == 0 || idx < 0 {
        return 0;
    }
    unsafe {
        let hdr = ptr as *mut ObjHeader;
        let i = idx as usize;
        if i >= (*hdr).len {
            return 0;
        }
        *fields_ptr(hdr).add(i)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_rc_set_field(ptr: i64, idx: i64, val: i64) {
    if ptr == 0 || idx < 0 {
        return;
    }
    unsafe {
        let hdr = ptr as *mut ObjHeader;
        let i = idx as usize;
        if i < (*hdr).len {
            let fp = fields_ptr(hdr);
            let old = *fp.add(i);
            *fp.add(i) = val;
            let mask = (*hdr).rc_mask;
            let is_rc = mask == RC_MASK_ALL || (i < 64 && (mask & (1u64 << i)) != 0);
            if is_rc && old != 0 {
                tenon_rc_dec(old);
            }
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_rc_set_field_raw(ptr: i64, idx: i64, val: i64) {
    if ptr == 0 || idx < 0 {
        return;
    }
    unsafe {
        let hdr = ptr as *mut ObjHeader;
        let i = idx as usize;
        if i < (*hdr).len {
            *fields_ptr(hdr).add(i) = val;
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_rc_get_tag(ptr: i64) -> i64 {
    if ptr == 0 {
        return -1;
    }
    unsafe { (*(ptr as *const ObjHeader)).tag }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_rc_get_len(ptr: i64) -> i64 {
    if ptr == 0 {
        return 0;
    }
    unsafe { (*(ptr as *const ObjHeader)).len as i64 }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_text_new(data_ptr: i64, byte_len: i64) -> i64 {
    let blen = byte_len.max(0) as usize;
    let words = (blen + 1).div_ceil(8).max(1);
    let obj = tenon_rc_alloc(TAG_TEXT, words as i64) as *mut ObjHeader;
    unsafe {
        (*obj).len = blen;
        let dst = fields_ptr(obj) as *mut u8;
        if blen > 0 && data_ptr != 0 {
            let src = data_ptr as *const u8;
            std::ptr::copy_nonoverlapping(src, dst, blen);
        }
        *dst.add(blen) = 0;
    }
    obj as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_text_eq(a: i64, b: i64) -> i64 {
    if a == b {
        return 1;
    }
    if a == 0 || b == 0 {
        return 0;
    }
    unsafe {
        let ha = a as *mut ObjHeader;
        let hb = b as *mut ObjHeader;
        if (*ha).len != (*hb).len {
            return 0;
        }
        let sa = std::slice::from_raw_parts(fields_ptr(ha) as *const u8, (*ha).len);
        let sb = std::slice::from_raw_parts(fields_ptr(hb) as *const u8, (*hb).len);
        if sa == sb { 1 } else { 0 }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_list_push(ptr: i64, val: i64) -> i64 {
    if ptr == 0 {
        let fresh = tenon_rc_alloc(TAG_LIST, 1);
        tenon_rc_set_field(fresh, 0, val);
        return fresh;
    }
    unsafe {
        let hdr = ptr as *mut ObjHeader;
        let old_len = (*hdr).len;
        let rc_mask = (*hdr).rc_mask;
        let shared = (*hdr).rc.load(Ordering::SeqCst) > 1;
        if !shared && old_len < (*hdr).cap {
            REUSE_HITS.fetch_add(1, Ordering::SeqCst);
            *fields_ptr(hdr).add(old_len) = val;
            (*hdr).len = old_len + 1;
            return ptr;
        }
        let new_cap = (old_len + 1).max((*hdr).cap * 2);
        let fresh = tenon_rc_alloc(TAG_LIST, new_cap as i64) as *mut ObjHeader;
        (*fresh).len = old_len + 1;
        (*fresh).rc_mask = rc_mask;
        let src_f = fields_ptr(hdr);
        let dst_f = fields_ptr(fresh);
        for i in 0..old_len {
            let elem = *src_f.add(i);
            *dst_f.add(i) = elem;
            if shared && rc_mask == RC_MASK_ALL && elem != 0 {
                tenon_rc_inc(elem);
            }
        }
        *dst_f.add(old_len) = val;
        if shared {
            COW_CLONES.fetch_add(1, Ordering::SeqCst);
        } else {
            REUSE_HITS.fetch_add(1, Ordering::SeqCst);
            // Elements were moved into `fresh`; prevent `tenon_rc_dec(ptr)` from freeing them!
            (*hdr).rc_mask = 0;
        }
        tenon_rc_dec(ptr);
        fresh as i64
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_list_filled(len: i64, init_val: i64, elem_is_rc: i64) -> i64 {
    let n = len.max(0) as usize;
    let fresh = tenon_rc_alloc(TAG_LIST, n as i64) as *mut ObjHeader;
    unsafe {
        (*fresh).len = n;
        if elem_is_rc != 0 {
            (*fresh).rc_mask = RC_MASK_ALL;
        }
        let fp = fields_ptr(fresh);
        for i in 0..n {
            *fp.add(i) = init_val;
            if elem_is_rc != 0 && init_val != 0 {
                tenon_rc_inc(init_val);
            }
        }
    }
    fresh as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_list_pop_shrink(ptr: i64) -> i64 {
    if ptr == 0 {
        return 0;
    }
    let uniq = tenon_rc_ensure_unique(ptr);
    unsafe {
        let hdr = uniq as *mut ObjHeader;
        if (*hdr).len > 0 {
            let last_idx = (*hdr).len - 1;
            let fp = fields_ptr(hdr);
            let old = *fp.add(last_idx);
            *fp.add(last_idx) = 0;
            (*hdr).len = last_idx;
            if (*hdr).rc_mask == RC_MASK_ALL && old != 0 {
                tenon_rc_dec(old);
            }
        }
    }
    uniq
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_async_frame_alloc(total_slots: i64, poll_fn_ptr: i64) -> i64 {
    let slots = total_slots.max(4);
    let frame = tenon_rc_alloc(TAG_ASYNC_FRAME, slots);
    tenon_rc_set_field(frame, 0, 0);
    tenon_rc_set_field(frame, 1, poll_fn_ptr);
    tenon_rc_set_field(frame, 2, 0);
    tenon_rc_set_field(frame, 3, 0);
    frame
}

extern "C" fn tenon_async_yield_poll(frame: i64) -> i64 {
    let st = tenon_rc_get_field(frame, 0);
    if st == 0 {
        tenon_rc_set_field(frame, 0, 1);
        ASYNC_SUSPENDS.fetch_add(1, Ordering::SeqCst);
        0
    } else {
        let val = tenon_rc_get_field(frame, 4);
        tenon_rc_set_field(frame, 2, val);
        tenon_rc_set_field(frame, 0, -1);
        1
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_async_yield_create(val: i64) -> i64 {
    let frame = tenon_async_frame_alloc(5, tenon_async_yield_poll as *const u8 as usize as i64);
    tenon_rc_set_field(frame, 4, val);
    frame
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_async_poll_frame(frame: i64) -> i64 {
    if frame == 0 {
        return 1;
    }
    ASYNC_POLLS.fetch_add(1, Ordering::SeqCst);
    let poll_ptr = tenon_rc_get_field(frame, 1);
    if poll_ptr == 0 {
        return 1;
    }
    let poll_fn: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(poll_ptr as usize) };
    poll_fn(frame)
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_async_block_on(frame: i64) -> i64 {
    loop {
        if tenon_async_poll_frame(frame) != 0 {
            let res = tenon_rc_get_field(frame, 2);
            tenon_rc_dec(frame);
            return res;
        }
    }
}
