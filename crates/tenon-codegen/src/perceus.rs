use std::alloc::{Layout, alloc, dealloc};
use std::sync::atomic::{AtomicUsize, Ordering};

pub const TAG_ERR: i64 = 0;
pub const TAG_OK: i64 = 1;
pub const TAG_NONE: i64 = 2;
pub const TAG_SOME: i64 = 3;
pub const TAG_LIST: i64 = 4;
pub const TAG_TEXT: i64 = 5;
pub const TAG_ASYNC_FRAME: i64 = 6;
pub const TAG_RECORD_BASE: i64 = 100;

static LIVE_OBJECTS: AtomicUsize = AtomicUsize::new(0);
static TOTAL_ALLOCS: AtomicUsize = AtomicUsize::new(0);
static REUSE_HITS: AtomicUsize = AtomicUsize::new(0);
static COW_CLONES: AtomicUsize = AtomicUsize::new(0);
static ASYNC_POLLS: AtomicUsize = AtomicUsize::new(0);
static ASYNC_SUSPENDS: AtomicUsize = AtomicUsize::new(0);

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

#[repr(C)]
struct ObjHeader {
    rc: AtomicUsize,
    tag: i64,
    len: usize,
    cap: usize,
}

fn obj_layout(cap: usize) -> Layout {
    let header_size = std::mem::size_of::<ObjHeader>();
    let fields_size = cap * std::mem::size_of::<i64>();
    Layout::from_size_align(header_size + fields_size, std::mem::align_of::<ObjHeader>()).unwrap()
}

unsafe fn fields_ptr(hdr: *mut ObjHeader) -> *mut i64 {
    unsafe { (hdr as *mut u8).add(std::mem::size_of::<ObjHeader>()) as *mut i64 }
}

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

pub extern "C" fn tenon_rc_dec(ptr: i64) {
    if ptr == 0 {
        return;
    }
    unsafe {
        let hdr = ptr as *mut ObjHeader;
        let prev = (*hdr).rc.fetch_sub(1, Ordering::SeqCst);
        if prev == 1 {
            let cap = (*hdr).cap;
            LIVE_OBJECTS.fetch_sub(1, Ordering::SeqCst);
            dealloc(hdr as *mut u8, obj_layout(cap));
        }
    }
}

/// Perceus FBIP / MVS Copy-on-Write primitive:
/// - If `rc == 1`: mutates in-place with ZERO allocation (increments `REUSE_HITS`)!
/// - If `rc > 1`: decrements old rc and allocates an independent clone with `rc == 1` (increments `COW_CLONES`)!
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
        let fresh = tenon_rc_alloc(tag, len as i64) as *mut ObjHeader;
        let src_f = fields_ptr(hdr);
        let dst_f = fields_ptr(fresh);
        for i in 0..len {
            *dst_f.add(i) = *src_f.add(i);
        }
        tenon_rc_dec(ptr);
        fresh as i64
    }
}

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

pub extern "C" fn tenon_rc_set_field(ptr: i64, idx: i64, val: i64) {
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

pub extern "C" fn tenon_rc_get_tag(ptr: i64) -> i64 {
    if ptr == 0 {
        return -1;
    }
    unsafe { (*(ptr as *const ObjHeader)).tag }
}

pub extern "C" fn tenon_rc_get_len(ptr: i64) -> i64 {
    if ptr == 0 {
        return 0;
    }
    unsafe { (*(ptr as *const ObjHeader)).len as i64 }
}

pub extern "C" fn tenon_text_new(data_ptr: i64, byte_len: i64) -> i64 {
    let blen = byte_len.max(0) as usize;
    let words = blen.div_ceil(8).max(1);
    let obj = tenon_rc_alloc(TAG_TEXT, words as i64) as *mut ObjHeader;
    unsafe {
        (*obj).len = blen;
        if blen > 0 && data_ptr != 0 {
            let src = data_ptr as *const u8;
            let dst = fields_ptr(obj) as *mut u8;
            std::ptr::copy_nonoverlapping(src, dst, blen);
        }
    }
    obj as i64
}

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

pub extern "C" fn tenon_list_push(ptr: i64, val: i64) -> i64 {
    if ptr == 0 {
        let fresh = tenon_rc_alloc(TAG_LIST, 1);
        tenon_rc_set_field(fresh, 0, val);
        return fresh;
    }
    unsafe {
        let hdr = ptr as *mut ObjHeader;
        let old_len = (*hdr).len;
        if (*hdr).rc.load(Ordering::SeqCst) == 1 && old_len < (*hdr).cap {
            REUSE_HITS.fetch_add(1, Ordering::SeqCst);
            *fields_ptr(hdr).add(old_len) = val;
            (*hdr).len = old_len + 1;
            return ptr;
        }
        let new_cap = (old_len + 1).max((*hdr).cap * 2);
        let fresh = tenon_rc_alloc(TAG_LIST, new_cap as i64) as *mut ObjHeader;
        (*fresh).len = old_len + 1;
        let src_f = fields_ptr(hdr);
        let dst_f = fields_ptr(fresh);
        for i in 0..old_len {
            *dst_f.add(i) = *src_f.add(i);
        }
        *dst_f.add(old_len) = val;
        if (*hdr).rc.load(Ordering::SeqCst) > 1 {
            COW_CLONES.fetch_add(1, Ordering::SeqCst);
        } else {
            REUSE_HITS.fetch_add(1, Ordering::SeqCst);
        }
        tenon_rc_dec(ptr);
        fresh as i64
    }
}

// ---------------------------------------------------------------------------
// Stackless Async Coroutine State-Machine Runtime (ADR-0001 D04)
// Frame layout:
//   slot 0: state (0 = init, ..., -1 = done)
//   slot 1: poll_fn_ptr (`extern "C" fn(frame_ptr: i64) -> i64`, 0=Pending, 1=Ready)
//   slot 2: result value (valid when Ready)
//   slot 3: active child future frame_ptr (if suspended awaiting child)
//   slot 4+: captured params & locals surviving across `await` points
// ---------------------------------------------------------------------------

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
        0 // Pending: genuine stackless suspension!
    } else {
        let val = tenon_rc_get_field(frame, 4);
        tenon_rc_set_field(frame, 2, val);
        tenon_rc_set_field(frame, 0, -1);
        1 // Ready!
    }
}

pub extern "C" fn tenon_async_yield_create(val: i64) -> i64 {
    let frame = tenon_async_frame_alloc(5, tenon_async_yield_poll as *const u8 as usize as i64);
    tenon_rc_set_field(frame, 4, val);
    frame
}

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

pub extern "C" fn tenon_async_block_on(frame: i64) -> i64 {
    loop {
        if tenon_async_poll_frame(frame) != 0 {
            let res = tenon_rc_get_field(frame, 2);
            tenon_rc_dec(frame);
            return res;
        }
    }
}
