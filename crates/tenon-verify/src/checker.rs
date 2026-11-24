use crate::l0::{AbsInt, Interval, L0Env, L0Result};
use crate::smt::{SmtOutcome, SmtQuery};
use crate::sync_solver::{DistributedAppManifest, solve_distributed_manifest};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use tenon_syntax::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProofTier {
    L0Fast,
    L1Smt,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofRecord {
    pub obligation: String,
    pub tier: ProofTier,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostic {
    pub code: String,
    pub message: String,
    pub span: Span,
    pub notes: Vec<String>,
    pub counterexample: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckReport {
    pub ok: bool,
    pub proofs: Vec<ProofRecord>,
    pub diagnostics: Vec<Diagnostic>,
    pub sql_checks: BTreeMap<String, String>,
    pub manifest: Option<DistributedAppManifest>,
}

struct BuiltinSig {
    name: &'static str,
    arity: usize,
    is_pure: bool,
    allocates: bool,
    param_types: &'static [&'static str],
    ret_type: &'static str,
}

const BUILTIN_SIGS: &[BuiltinSig] = &[
    // Core builtins
    BuiltinSig { name: "print_int", arity: 1, is_pure: false, allocates: false, param_types: &["Int"], ret_type: "Int" },
    BuiltinSig { name: "len", arity: 1, is_pure: true, allocates: false, param_types: &["Any"], ret_type: "Int" },
    BuiltinSig { name: "push", arity: 2, is_pure: true, allocates: true, param_types: &["Any", "Any"], ret_type: "List[Any]" },
    BuiltinSig { name: "list_filled", arity: 2, is_pure: true, allocates: true, param_types: &["Int", "Any"], ret_type: "List[Any]" },
    BuiltinSig { name: "list_pop_shrink", arity: 1, is_pure: true, allocates: true, param_types: &["Any"], ret_type: "List[Any]" },
    BuiltinSig { name: "yield_val", arity: 1, is_pure: false, allocates: true, param_types: &["Any"], ret_type: "Any" },
    BuiltinSig { name: "block_on", arity: 1, is_pure: false, allocates: true, param_types: &["Any"], ret_type: "Any" },
    // Polymorphic Option / Result constructors
    BuiltinSig { name: "Ok", arity: 1, is_pure: true, allocates: true, param_types: &["Any"], ret_type: "Result[Any, Any]" },
    BuiltinSig { name: "Err", arity: 1, is_pure: true, allocates: true, param_types: &["Any"], ret_type: "Result[Any, Any]" },
    BuiltinSig { name: "Some", arity: 1, is_pure: true, allocates: true, param_types: &["Any"], ret_type: "Option[Any]" },
    // 4 Minimal core byte / text primitives (pure)
    BuiltinSig { name: "byte_at", arity: 2, is_pure: true, allocates: false, param_types: &["Text", "Int"], ret_type: "Int" },
    BuiltinSig { name: "str_slice", arity: 3, is_pure: true, allocates: true, param_types: &["Text", "Int", "Int"], ret_type: "Text" },
    BuiltinSig { name: "byte_to_text", arity: 1, is_pure: true, allocates: true, param_types: &["Int"], ret_type: "Text" },
    BuiltinSig { name: "str_concat", arity: 2, is_pure: true, allocates: true, param_types: &["Text", "Text"], ret_type: "Text" },
    // 6 Minimal IEEE-754 Float64 primitives (pure)
    BuiltinSig { name: "int_to_float", arity: 1, is_pure: true, allocates: false, param_types: &["Int"], ret_type: "Float" },
    BuiltinSig { name: "float_to_int", arity: 1, is_pure: true, allocates: false, param_types: &["Float"], ret_type: "Int" },
    BuiltinSig { name: "float_to_text", arity: 1, is_pure: true, allocates: true, param_types: &["Float"], ret_type: "Text" },
    BuiltinSig { name: "parse_float", arity: 1, is_pure: true, allocates: false, param_types: &["Text"], ret_type: "Float" },
    BuiltinSig { name: "float_is_nan", arity: 1, is_pure: true, allocates: false, param_types: &["Float"], ret_type: "Bool" },
    BuiltinSig { name: "float_bits", arity: 1, is_pure: true, allocates: false, param_types: &["Float"], ret_type: "Int" },
    // 4 O(n) Byte buffer primitives (pure)
    BuiltinSig { name: "buf_new", arity: 0, is_pure: true, allocates: true, param_types: &[], ret_type: "Int" },
    BuiltinSig { name: "buf_push_str", arity: 2, is_pure: true, allocates: true, param_types: &["Int", "Text"], ret_type: "Int" },
    BuiltinSig { name: "buf_push_byte", arity: 2, is_pure: true, allocates: true, param_types: &["Int", "Int"], ret_type: "Int" },
    BuiltinSig { name: "buf_finish", arity: 1, is_pure: true, allocates: true, param_types: &["Int"], ret_type: "Text" },
    // 10 Minimal OS / I/O / FS / Process syscalls (impure)
    BuiltinSig { name: "sys_write_stdout", arity: 1, is_pure: false, allocates: false, param_types: &["Text"], ret_type: "Int" },
    BuiltinSig { name: "sys_read_line", arity: 0, is_pure: false, allocates: true, param_types: &[], ret_type: "Text" },
    BuiltinSig { name: "sys_is_eof", arity: 0, is_pure: false, allocates: false, param_types: &[], ret_type: "Bool" },
    BuiltinSig { name: "sys_env", arity: 1, is_pure: false, allocates: true, param_types: &["Text"], ret_type: "Text" },
    BuiltinSig { name: "sys_cwd", arity: 0, is_pure: false, allocates: true, param_types: &[], ret_type: "Text" },
    BuiltinSig { name: "sys_read_file", arity: 1, is_pure: false, allocates: true, param_types: &["Text"], ret_type: "Text" },
    BuiltinSig { name: "sys_file_ok", arity: 0, is_pure: false, allocates: false, param_types: &[], ret_type: "Bool" },
    BuiltinSig { name: "sys_write_file", arity: 2, is_pure: false, allocates: false, param_types: &["Text", "Text"], ret_type: "Bool" },
    BuiltinSig { name: "sys_exec", arity: 1, is_pure: false, allocates: true, param_types: &["Text"], ret_type: "Text" },
    BuiltinSig { name: "sys_exec_exit_code", arity: 0, is_pure: false, allocates: false, param_types: &[], ret_type: "Int" },
    // 6 Minimal Raw HTTPS Line-Stream transport syscalls (impure)
    BuiltinSig { name: "sys_http_open", arity: 4, is_pure: false, allocates: false, param_types: &["Text", "Text", "Text", "Text"], ret_type: "Int" },
    BuiltinSig { name: "sys_http_sse_open", arity: 3, is_pure: false, allocates: false, param_types: &["Text", "Text", "Text"], ret_type: "Int" },
    BuiltinSig { name: "sys_http_sse_next_line", arity: 1, is_pure: false, allocates: true, param_types: &["Int"], ret_type: "Text" },
    BuiltinSig { name: "sys_http_sse_eof", arity: 1, is_pure: false, allocates: false, param_types: &["Int"], ret_type: "Bool" },
    BuiltinSig { name: "sys_http_sse_error", arity: 1, is_pure: false, allocates: true, param_types: &["Int"], ret_type: "Text" },
    BuiltinSig { name: "sys_http_sse_close", arity: 1, is_pure: false, allocates: false, param_types: &["Int"], ret_type: "Int" },
    // S1: Hardware bit intrinsics, wrapping/saturating arithmetic, Float32 text ops
    BuiltinSig { name: "clz", arity: 1, is_pure: true, allocates: false, param_types: &["AnyInt"], ret_type: "Int" },
    BuiltinSig { name: "ctz", arity: 1, is_pure: true, allocates: false, param_types: &["AnyInt"], ret_type: "Int" },
    BuiltinSig { name: "popcnt", arity: 1, is_pure: true, allocates: false, param_types: &["AnyInt"], ret_type: "Int" },
    BuiltinSig { name: "bswap16", arity: 1, is_pure: true, allocates: false, param_types: &["AnyInt"], ret_type: "UInt16" },
    BuiltinSig { name: "bswap32", arity: 1, is_pure: true, allocates: false, param_types: &["AnyInt"], ret_type: "UInt32" },
    BuiltinSig { name: "bswap64", arity: 1, is_pure: true, allocates: false, param_types: &["AnyInt"], ret_type: "UInt64" },
    BuiltinSig { name: "rotl64", arity: 2, is_pure: true, allocates: false, param_types: &["AnyInt", "AnyInt"], ret_type: "UInt64" },
    BuiltinSig { name: "rotr64", arity: 2, is_pure: true, allocates: false, param_types: &["AnyInt", "AnyInt"], ret_type: "UInt64" },
    BuiltinSig { name: "wrapping_add", arity: 2, is_pure: true, allocates: false, param_types: &["AnyInt", "AnyInt"], ret_type: "Int" },
    BuiltinSig { name: "wrapping_sub", arity: 2, is_pure: true, allocates: false, param_types: &["AnyInt", "AnyInt"], ret_type: "Int" },
    BuiltinSig { name: "wrapping_mul", arity: 2, is_pure: true, allocates: false, param_types: &["AnyInt", "AnyInt"], ret_type: "Int" },
    BuiltinSig { name: "saturating_add", arity: 2, is_pure: true, allocates: false, param_types: &["AnyInt", "AnyInt"], ret_type: "Int" },
    BuiltinSig { name: "saturating_sub", arity: 2, is_pure: true, allocates: false, param_types: &["AnyInt", "AnyInt"], ret_type: "Int" },
    BuiltinSig { name: "f32_to_text", arity: 1, is_pure: true, allocates: true, param_types: &["Float32"], ret_type: "Text" },
    BuiltinSig { name: "parse_f32", arity: 1, is_pure: true, allocates: false, param_types: &["Text"], ret_type: "Float32" },
    // S1: Compact 1-byte `Bytes` + zero-copy `BytesSlice` + endian I/O
    BuiltinSig { name: "bytes_new", arity: 1, is_pure: true, allocates: true, param_types: &["AnyInt"], ret_type: "Bytes" },
    BuiltinSig { name: "bytes_filled", arity: 2, is_pure: true, allocates: true, param_types: &["AnyInt", "AnyInt"], ret_type: "Bytes" },
    BuiltinSig { name: "bytes_len", arity: 1, is_pure: true, allocates: false, param_types: &["Bytes"], ret_type: "Int" },
    BuiltinSig { name: "bytes_get", arity: 2, is_pure: true, allocates: false, param_types: &["Bytes", "AnyInt"], ret_type: "UInt8" },
    BuiltinSig { name: "bytes_set", arity: 3, is_pure: true, allocates: true, param_types: &["Bytes", "AnyInt", "AnyInt"], ret_type: "Bytes" },
    BuiltinSig { name: "bytes_push", arity: 2, is_pure: true, allocates: true, param_types: &["Bytes", "AnyInt"], ret_type: "Bytes" },
    BuiltinSig { name: "bytes_read_u16_le", arity: 2, is_pure: true, allocates: false, param_types: &["Bytes", "AnyInt"], ret_type: "UInt16" },
    BuiltinSig { name: "bytes_read_u16_be", arity: 2, is_pure: true, allocates: false, param_types: &["Bytes", "AnyInt"], ret_type: "UInt16" },
    BuiltinSig { name: "bytes_read_u32_le", arity: 2, is_pure: true, allocates: false, param_types: &["Bytes", "AnyInt"], ret_type: "UInt32" },
    BuiltinSig { name: "bytes_read_u32_be", arity: 2, is_pure: true, allocates: false, param_types: &["Bytes", "AnyInt"], ret_type: "UInt32" },
    BuiltinSig { name: "bytes_read_u64_le", arity: 2, is_pure: true, allocates: false, param_types: &["Bytes", "AnyInt"], ret_type: "UInt64" },
    BuiltinSig { name: "bytes_read_u64_be", arity: 2, is_pure: true, allocates: false, param_types: &["Bytes", "AnyInt"], ret_type: "UInt64" },
    BuiltinSig { name: "bytes_write_u16_le", arity: 3, is_pure: true, allocates: true, param_types: &["Bytes", "AnyInt", "AnyInt"], ret_type: "Bytes" },
    BuiltinSig { name: "bytes_write_u16_be", arity: 3, is_pure: true, allocates: true, param_types: &["Bytes", "AnyInt", "AnyInt"], ret_type: "Bytes" },
    BuiltinSig { name: "bytes_write_u32_le", arity: 3, is_pure: true, allocates: true, param_types: &["Bytes", "AnyInt", "AnyInt"], ret_type: "Bytes" },
    BuiltinSig { name: "bytes_write_u32_be", arity: 3, is_pure: true, allocates: true, param_types: &["Bytes", "AnyInt", "AnyInt"], ret_type: "Bytes" },
    BuiltinSig { name: "bytes_write_u64_le", arity: 3, is_pure: true, allocates: true, param_types: &["Bytes", "AnyInt", "AnyInt"], ret_type: "Bytes" },
    BuiltinSig { name: "bytes_write_u64_be", arity: 3, is_pure: true, allocates: true, param_types: &["Bytes", "AnyInt", "AnyInt"], ret_type: "Bytes" },
    BuiltinSig { name: "bytes_slice", arity: 3, is_pure: true, allocates: true, param_types: &["Any", "AnyInt", "AnyInt"], ret_type: "BytesSlice" },
    BuiltinSig { name: "slice_len", arity: 1, is_pure: true, allocates: false, param_types: &["BytesSlice"], ret_type: "Int" },
    BuiltinSig { name: "slice_get", arity: 2, is_pure: true, allocates: false, param_types: &["BytesSlice", "AnyInt"], ret_type: "UInt8" },
    BuiltinSig { name: "bytes_to_text", arity: 1, is_pure: true, allocates: true, param_types: &["Bytes"], ret_type: "Text" },
    BuiltinSig { name: "text_to_bytes", arity: 1, is_pure: true, allocates: true, param_types: &["Text"], ret_type: "Bytes" },
    BuiltinSig { name: "bytes_ptr", arity: 1, is_pure: true, allocates: false, param_types: &["Bytes"], ret_type: "Ptr[UInt8]" },
    BuiltinSig { name: "text_cstr_ptr", arity: 1, is_pure: true, allocates: false, param_types: &["Text"], ret_type: "Ptr[UInt8]" },
    BuiltinSig { name: "cstr_to_text", arity: 1, is_pure: false, allocates: true, param_types: &["Any"], ret_type: "Text" },
    // S2: Layout intrinsics, raw pointers & unsafe memory ops
    BuiltinSig { name: "size_of", arity: 0, is_pure: true, allocates: false, param_types: &[], ret_type: "USize" },
    BuiltinSig { name: "align_of", arity: 0, is_pure: true, allocates: false, param_types: &[], ret_type: "USize" },
    BuiltinSig { name: "offset_of", arity: 2, is_pure: true, allocates: false, param_types: &["Any", "Text"], ret_type: "USize" },
    BuiltinSig { name: "null_ptr", arity: 0, is_pure: true, allocates: false, param_types: &[], ret_type: "MutPtr[Any]" },
    BuiltinSig { name: "ptr_is_null", arity: 1, is_pure: true, allocates: false, param_types: &["Any"], ret_type: "Bool" },
    BuiltinSig { name: "ptr_of", arity: 1, is_pure: true, allocates: false, param_types: &["Any"], ret_type: "Ptr[Any]" },
    BuiltinSig { name: "mut_ptr_of", arity: 1, is_pure: true, allocates: false, param_types: &["Any"], ret_type: "MutPtr[Any]" },
    BuiltinSig { name: "ptr_add", arity: 2, is_pure: false, allocates: false, param_types: &["Any", "AnyInt"], ret_type: "Any" },
    BuiltinSig { name: "ptr_read", arity: 1, is_pure: false, allocates: false, param_types: &["Any"], ret_type: "Any" },
    BuiltinSig { name: "ptr_write", arity: 2, is_pure: false, allocates: false, param_types: &["Any", "Any"], ret_type: "Int" },
    BuiltinSig { name: "volatile_read", arity: 1, is_pure: false, allocates: false, param_types: &["Any"], ret_type: "Any" },
    BuiltinSig { name: "volatile_write", arity: 2, is_pure: false, allocates: false, param_types: &["Any", "Any"], ret_type: "Int" },
    BuiltinSig { name: "raw_alloc", arity: 1, is_pure: false, allocates: true, param_types: &["AnyInt"], ret_type: "MutPtr[UInt8]" },
    BuiltinSig { name: "raw_free", arity: 2, is_pure: false, allocates: false, param_types: &["Any", "AnyInt"], ret_type: "Int" },
    BuiltinSig { name: "raw_memcpy", arity: 3, is_pure: false, allocates: false, param_types: &["Any", "Any", "AnyInt"], ret_type: "MutPtr[UInt8]" },
    BuiltinSig { name: "raw_memset", arity: 3, is_pure: false, allocates: false, param_types: &["Any", "AnyInt", "AnyInt"], ret_type: "MutPtr[UInt8]" },
    BuiltinSig { name: "raw_memcmp", arity: 3, is_pure: false, allocates: false, param_types: &["Any", "Any", "AnyInt"], ret_type: "Int" },
    // S3: OS Threads, Atomics, Mutex, Channel
    BuiltinSig { name: "thread_spawn", arity: 1, is_pure: false, allocates: true, param_types: &["Any"], ret_type: "Int" },
    BuiltinSig { name: "thread_join", arity: 1, is_pure: false, allocates: false, param_types: &["AnyInt"], ret_type: "Int" },
    BuiltinSig { name: "thread_sleep_ms", arity: 1, is_pure: false, allocates: false, param_types: &["AnyInt"], ret_type: "Int" },
    BuiltinSig { name: "thread_hw_concurrency", arity: 0, is_pure: false, allocates: false, param_types: &[], ret_type: "Int" },
    BuiltinSig { name: "atomic_new", arity: 1, is_pure: false, allocates: true, param_types: &["AnyInt"], ret_type: "AtomicInt" },
    BuiltinSig { name: "atomic_load", arity: 2, is_pure: false, allocates: false, param_types: &["AtomicInt", "AnyInt"], ret_type: "Int" },
    BuiltinSig { name: "atomic_store", arity: 3, is_pure: false, allocates: false, param_types: &["AtomicInt", "AnyInt", "AnyInt"], ret_type: "Int" },
    BuiltinSig { name: "atomic_fetch_add", arity: 3, is_pure: false, allocates: false, param_types: &["AtomicInt", "AnyInt", "AnyInt"], ret_type: "Int" },
    BuiltinSig { name: "atomic_fetch_sub", arity: 3, is_pure: false, allocates: false, param_types: &["AtomicInt", "AnyInt", "AnyInt"], ret_type: "Int" },
    BuiltinSig { name: "atomic_cas", arity: 4, is_pure: false, allocates: false, param_types: &["AtomicInt", "AnyInt", "AnyInt", "AnyInt"], ret_type: "Bool" },
    BuiltinSig { name: "atomic_swap", arity: 3, is_pure: false, allocates: false, param_types: &["AtomicInt", "AnyInt", "AnyInt"], ret_type: "Int" },
    BuiltinSig { name: "mutex_new", arity: 1, is_pure: false, allocates: true, param_types: &["Any"], ret_type: "Mutex[Any]" },
    BuiltinSig { name: "mutex_lock", arity: 1, is_pure: false, allocates: false, param_types: &["Any"], ret_type: "Int" },
    BuiltinSig { name: "mutex_unlock", arity: 1, is_pure: false, allocates: false, param_types: &["Any"], ret_type: "Int" },
    BuiltinSig { name: "mutex_get", arity: 1, is_pure: false, allocates: false, param_types: &["Any"], ret_type: "Any" },
    BuiltinSig { name: "mutex_set", arity: 2, is_pure: false, allocates: false, param_types: &["Any", "Any"], ret_type: "Int" },
    BuiltinSig { name: "chan_new", arity: 0, is_pure: false, allocates: true, param_types: &[], ret_type: "Channel[Any]" },
    BuiltinSig { name: "chan_send", arity: 2, is_pure: false, allocates: false, param_types: &["Any", "Any"], ret_type: "Bool" },
    BuiltinSig { name: "chan_recv", arity: 1, is_pure: false, allocates: true, param_types: &["Any"], ret_type: "Option[Any]" },
    BuiltinSig { name: "chan_try_recv", arity: 1, is_pure: false, allocates: true, param_types: &["Any"], ret_type: "Option[Any]" },
    BuiltinSig { name: "chan_close", arity: 1, is_pure: false, allocates: false, param_types: &["Any"], ret_type: "Int" },
    // S3: Non-blocking TCP networking
    BuiltinSig { name: "net_tcp_listen", arity: 1, is_pure: false, allocates: false, param_types: &["Text"], ret_type: "Int" },
    BuiltinSig { name: "net_tcp_local_port", arity: 1, is_pure: false, allocates: false, param_types: &["AnyInt"], ret_type: "Int" },
    BuiltinSig { name: "net_tcp_accept", arity: 1, is_pure: false, allocates: false, param_types: &["AnyInt"], ret_type: "Int" },
    BuiltinSig { name: "net_tcp_connect", arity: 1, is_pure: false, allocates: false, param_types: &["Text"], ret_type: "Int" },
    BuiltinSig { name: "net_set_nonblocking", arity: 2, is_pure: false, allocates: false, param_types: &["AnyInt", "Bool"], ret_type: "Bool" },
    BuiltinSig { name: "net_read", arity: 2, is_pure: false, allocates: true, param_types: &["AnyInt", "AnyInt"], ret_type: "Bytes" },
    BuiltinSig { name: "net_write", arity: 2, is_pure: false, allocates: false, param_types: &["AnyInt", "Any"], ret_type: "Int" },
    BuiltinSig { name: "net_close", arity: 1, is_pure: false, allocates: false, param_types: &["AnyInt"], ret_type: "Int" },
    // S3: SIMD vector primitives
    BuiltinSig { name: "simd_f32x4_new", arity: 4, is_pure: true, allocates: false, param_types: &["Float32", "Float32", "Float32", "Float32"], ret_type: "Float32x4" },
    BuiltinSig { name: "simd_f32x4_splat", arity: 1, is_pure: true, allocates: false, param_types: &["Float32"], ret_type: "Float32x4" },
    BuiltinSig { name: "simd_f32x4_lane", arity: 2, is_pure: true, allocates: false, param_types: &["Float32x4", "AnyInt"], ret_type: "Float32" },
    BuiltinSig { name: "simd_f32x4_binop", arity: 3, is_pure: true, allocates: false, param_types: &["Float32x4", "Float32x4", "AnyInt"], ret_type: "Float32x4" },
    BuiltinSig { name: "simd_f32x4_fma", arity: 3, is_pure: true, allocates: false, param_types: &["Float32x4", "Float32x4", "Float32x4"], ret_type: "Float32x4" },
    BuiltinSig { name: "simd_f32x4_reduce_sum", arity: 1, is_pure: true, allocates: false, param_types: &["Float32x4"], ret_type: "Float32" },
    BuiltinSig { name: "simd_f32x4_dot", arity: 2, is_pure: true, allocates: false, param_types: &["Float32x4", "Float32x4"], ret_type: "Float32" },
    BuiltinSig { name: "simd_f64x4_new", arity: 4, is_pure: true, allocates: false, param_types: &["Float", "Float", "Float", "Float"], ret_type: "Float64x4" },
    BuiltinSig { name: "simd_f64x4_splat", arity: 1, is_pure: true, allocates: false, param_types: &["Float"], ret_type: "Float64x4" },
    BuiltinSig { name: "simd_f64x4_lane", arity: 2, is_pure: true, allocates: false, param_types: &["Float64x4", "AnyInt"], ret_type: "Float" },
    BuiltinSig { name: "simd_f64x4_binop", arity: 3, is_pure: true, allocates: false, param_types: &["Float64x4", "Float64x4", "AnyInt"], ret_type: "Float64x4" },
    BuiltinSig { name: "simd_f64x4_fma", arity: 3, is_pure: true, allocates: false, param_types: &["Float64x4", "Float64x4", "Float64x4"], ret_type: "Float64x4" },
    BuiltinSig { name: "simd_f64x4_reduce_sum", arity: 1, is_pure: true, allocates: false, param_types: &["Float64x4"], ret_type: "Float" },
    BuiltinSig { name: "simd_f64x4_dot", arity: 2, is_pure: true, allocates: false, param_types: &["Float64x4", "Float64x4"], ret_type: "Float" },
    BuiltinSig { name: "simd_i32x4_new", arity: 4, is_pure: true, allocates: false, param_types: &["Int32", "Int32", "Int32", "Int32"], ret_type: "Int32x4" },
    BuiltinSig { name: "simd_i32x4_splat", arity: 1, is_pure: true, allocates: false, param_types: &["Int32"], ret_type: "Int32x4" },
    BuiltinSig { name: "simd_i32x4_lane", arity: 2, is_pure: true, allocates: false, param_types: &["Int32x4", "AnyInt"], ret_type: "Int32" },
    BuiltinSig { name: "simd_i32x4_binop", arity: 3, is_pure: true, allocates: false, param_types: &["Int32x4", "Int32x4", "AnyInt"], ret_type: "Int32x4" },
    BuiltinSig { name: "simd_i32x4_reduce_sum", arity: 1, is_pure: true, allocates: false, param_types: &["Int32x4"], ret_type: "Int32" },
    BuiltinSig { name: "simd_i64x4_new", arity: 4, is_pure: true, allocates: false, param_types: &["Int", "Int", "Int", "Int"], ret_type: "Int64x4" },
    BuiltinSig { name: "simd_i64x4_splat", arity: 1, is_pure: true, allocates: false, param_types: &["Int"], ret_type: "Int64x4" },
    BuiltinSig { name: "simd_i64x4_lane", arity: 2, is_pure: true, allocates: false, param_types: &["Int64x4", "AnyInt"], ret_type: "Int" },
    BuiltinSig { name: "simd_i64x4_binop", arity: 3, is_pure: true, allocates: false, param_types: &["Int64x4", "Int64x4", "AnyInt"], ret_type: "Int64x4" },
    BuiltinSig { name: "simd_i64x4_reduce_sum", arity: 1, is_pure: true, allocates: false, param_types: &["Int64x4"], ret_type: "Int" },
    BuiltinSig { name: "simd_u8x16_splat", arity: 1, is_pure: true, allocates: false, param_types: &["UInt8"], ret_type: "UInt8x16" },
    BuiltinSig { name: "simd_u8x16_lane", arity: 2, is_pure: true, allocates: false, param_types: &["UInt8x16", "AnyInt"], ret_type: "UInt8" },
    BuiltinSig { name: "simd_u8x16_set_lane", arity: 3, is_pure: true, allocates: false, param_types: &["UInt8x16", "AnyInt", "UInt8"], ret_type: "UInt8x16" },
    BuiltinSig { name: "simd_u8x16_binop", arity: 3, is_pure: true, allocates: false, param_types: &["UInt8x16", "UInt8x16", "AnyInt"], ret_type: "UInt8x16" },
    BuiltinSig { name: "simd_u8x16_reduce_sum", arity: 1, is_pure: true, allocates: false, param_types: &["UInt8x16"], ret_type: "Int" },
];

fn is_unsafe_builtin(name: &str) -> bool {
    matches!(
        name,
        "ptr_read"
            | "ptr_write"
            | "ptr_add"
            | "volatile_read"
            | "volatile_write"
            | "raw_alloc"
            | "raw_free"
            | "raw_memcpy"
            | "raw_memset"
            | "raw_memcmp"
            | "cstr_to_text"
    )
}

fn lookup_builtin(name: &str) -> Option<&'static BuiltinSig> {
    BUILTIN_SIGS.iter().find(|b| b.name == name)
}

fn named_ty(name: &str) -> TypeExpr {
    TypeExpr::Named {
        name: name.to_string(),
        version: None,
        args: Vec::new(),
        span: Span::default(),
    }
}

fn generic_ty(name: &str, args: Vec<TypeExpr>) -> TypeExpr {
    TypeExpr::Named {
        name: name.to_string(),
        version: None,
        args,
        span: Span::default(),
    }
}

pub fn check_program(program: &Program) -> CheckReport {
    let mut ctx = CheckerCtx::new(program);
    ctx.check_all(program);

    // Run Protocol Constraint Solver if any distributed views/streams/subscribers exist
    let manifest = match solve_distributed_manifest(program) {
        Ok(m) => m,
        Err(errs) => {
            for e in errs {
                ctx.diagnostics.push(e);
            }
            None
        }
    };

    // Derive SQL CHECK constraints for visible refined records
    let sql_checks = derive_sqlite_checks(program);

    CheckReport {
        ok: ctx.diagnostics.is_empty(),
        proofs: ctx.proofs,
        diagnostics: ctx.diagnostics,
        sql_checks,
        manifest,
    }
}

#[derive(Debug, Clone)]
struct VariantInfo {
    enum_name: String,
    payload_tys: Vec<TypeExpr>,
}

struct CheckerCtx {
    type_aliases: BTreeMap<String, TypeAliasDecl>,
    records: BTreeMap<String, RecordDecl>,
    enums: BTreeMap<String, EnumDecl>,
    enum_variants: BTreeMap<String, VariantInfo>,
    bitfields: BTreeMap<String, BitfieldDecl>,
    unions: BTreeMap<String, UnionDecl>,
    extern_fns: BTreeMap<String, ExternFnDecl>,
    consts: BTreeMap<String, (ConstDecl, Option<i64>, Option<f64>)>,
    layout: LayoutEngine,
    interfaces: BTreeMap<String, InterfaceDecl>,
    inherent_methods: BTreeMap<(String, String), FnDecl>,
    interface_impls: BTreeMap<(String, String), ImplDecl>,
    impl_methods: BTreeMap<(String, String), Vec<(String, FnDecl)>>,
    pure_fns: BTreeMap<String, FnDecl>,
    smt_pure_fns: BTreeMap<String, FnDecl>,
    no_heap_fns: BTreeSet<String>,
    all_fns: BTreeMap<String, FnDecl>,
    async_fns: BTreeSet<String>,
    cur_type_params: BTreeSet<String>,
    cur_type_param_bounds: BTreeMap<String, Vec<String>>,
    in_unsafe: bool,
    cur_file_id: u32,
    proofs: Vec<ProofRecord>,
    diagnostics: Vec<Diagnostic>,
}

fn type_head_name(ty: &TypeExpr) -> String {
    match ty {
        TypeExpr::Named { name, .. } => name.clone(),
        TypeExpr::Dyn { interface, .. } => format!("dyn {}", interface),
        TypeExpr::Tuple(_, _) => "Tuple".into(),
        TypeExpr::Fn { .. } => "Fn".into(),
        TypeExpr::Array { .. } => "Array".into(),
    }
}

fn subst_type_expr(ty: &TypeExpr, subst: &BTreeMap<String, TypeExpr>) -> TypeExpr {
    match ty {
        TypeExpr::Named {
            name,
            version,
            args,
            span,
        } => {
            if args.is_empty() {
                if let Some(rep) = subst.get(name) {
                    return rep.clone();
                }
            }
            TypeExpr::Named {
                name: name.clone(),
                version: version.clone(),
                args: args.iter().map(|a| subst_type_expr(a, subst)).collect(),
                span: *span,
            }
        }
        TypeExpr::Dyn {
            interface,
            args,
            span,
        } => TypeExpr::Dyn {
            interface: interface.clone(),
            args: args.iter().map(|a| subst_type_expr(a, subst)).collect(),
            span: *span,
        },
        TypeExpr::Tuple(es, span) => TypeExpr::Tuple(
            es.iter().map(|e| subst_type_expr(e, subst)).collect(),
            *span,
        ),
        TypeExpr::Fn { params, ret, span } => TypeExpr::Fn {
            params: params.iter().map(|p| subst_type_expr(p, subst)).collect(),
            ret: ret.as_ref().map(|r| Box::new(subst_type_expr(r, subst))),
            span: *span,
        },
        TypeExpr::Array { elem, len, span } => TypeExpr::Array {
            elem: Box::new(subst_type_expr(elem, subst)),
            len: *len,
            span: *span,
        },
    }
}

fn subst_fn_self(f: &FnDecl, self_ty: &TypeExpr) -> FnDecl {
    let mut subst = BTreeMap::new();
    subst.insert("Self".into(), self_ty.clone());
    let mut out = f.clone();
    for p in &mut out.params {
        p.ty = subst_type_expr(&p.ty, &subst);
    }
    if let Some(rt) = &out.ret_ty {
        out.ret_ty = Some(subst_type_expr(rt, &subst));
    }
    out
}

fn iface_method_to_fn_decl(m: &InterfaceMethodDecl, self_ty: &TypeExpr) -> FnDecl {
    let f = FnDecl {
        attrs: m.attrs.clone(),
        is_pub: true,
        is_pure: m.is_pure,
        is_async: m.is_async,
        is_test: false,
        is_unsafe: false,
        name: m.name.clone(),
        type_params: m.type_params.clone(),
        type_param_bounds: m.type_param_bounds.clone(),
        params: m.params.clone(),
        ret_ty: m.ret_ty.clone(),
        requires: m.requires.clone(),
        ensures: m.ensures.clone(),
        body: m.default_body.clone().unwrap_or(Block {
            stmts: vec![],
            tail: None,
            span: m.span,
        }),
        span: m.span,
    };
    subst_fn_self(&f, self_ty)
}

impl CheckerCtx {
    fn new(program: &Program) -> Self {
        let mut type_aliases = BTreeMap::new();
        let mut records = BTreeMap::new();
        let mut enums = BTreeMap::new();
        let mut enum_variants = BTreeMap::new();
        let mut bitfields = BTreeMap::new();
        let mut unions = BTreeMap::new();
        let mut extern_fns = BTreeMap::new();
        let mut raw_consts = Vec::new();
        let layout = LayoutEngine::from_program(program);
        let mut interfaces = BTreeMap::new();
        let mut inherent_methods = BTreeMap::new();
        let mut interface_impls = BTreeMap::new();
        let mut impl_methods: BTreeMap<(String, String), Vec<(String, FnDecl)>> = BTreeMap::new();
        let mut pure_fns = BTreeMap::new();
        let mut no_heap_fns = BTreeSet::new();
        let mut all_fns = BTreeMap::new();
        let mut async_fns = BTreeSet::new();
        async_fns.insert("yield_val".into());

        for item in &program.items {
            match item {
                Item::TypeAlias(t) => {
                    type_aliases.insert(t.name.clone(), t.clone());
                }
                Item::Record(r) => {
                    records.insert(r.name.clone(), r.clone());
                    for m in &r.methods {
                        let sub = subst_fn_self(m, &named_ty(&r.name));
                        inherent_methods.insert((r.name.clone(), m.name.clone()), sub.clone());
                        all_fns.insert(format!("{}.{}", r.name, m.name), sub);
                    }
                }
                Item::Enum(e) => {
                    enums.insert(e.name.clone(), e.clone());
                    for v in &e.variants {
                        let payload_tys: Vec<TypeExpr> =
                            v.fields.iter().map(|f| f.ty.clone()).collect();
                        let info = VariantInfo {
                            enum_name: e.name.clone(),
                            payload_tys,
                        };
                        enum_variants.insert(v.name.clone(), info.clone());
                        enum_variants.insert(format!("{}.{}", e.name, v.name), info);
                    }
                }
                Item::Bitfield(bf) => {
                    bitfields.insert(bf.name.clone(), bf.clone());
                }
                Item::Union(u) => {
                    unions.insert(u.name.clone(), u.clone());
                }
                Item::ExternFn(ef) => {
                    extern_fns.insert(ef.name.clone(), ef.clone());
                }
                Item::Const(c) => {
                    raw_consts.push(c.clone());
                }
                Item::Interface(iface) => {
                    interfaces.insert(iface.name.clone(), iface.clone());
                }
                Item::Impl(_) => {}
                Item::Fn(f) => {
                    if f.is_pure {
                        pure_fns.insert(f.name.clone(), f.clone());
                    }
                    if f.attrs.iter().any(|a| a.name == "no_heap_alloc") {
                        no_heap_fns.insert(f.name.clone());
                    }
                    if f.is_async {
                        async_fns.insert(f.name.clone());
                    }
                    all_fns.insert(f.name.clone(), f.clone());
                }
                _ => {}
            }
        }

        // Second pass: index all `impl` blocks (inherent and interface impls, including default methods)
        for item in &program.items {
            if let Item::Impl(imp) = item {
                let tname = type_head_name(&imp.target_ty);
                if let Some(iface_ty) = &imp.interface {
                    let iname = type_head_name(iface_ty);
                    interface_impls.insert((iname.clone(), tname.clone()), imp.clone());
                    let mut provided = BTreeSet::new();
                    for m in &imp.methods {
                        provided.insert(m.name.clone());
                        let sub = subst_fn_self(m, &imp.target_ty);
                        impl_methods
                            .entry((tname.clone(), m.name.clone()))
                            .or_default()
                            .push((iname.clone(), sub));
                    }
                    // Inherit any default methods from the interface not overridden by this impl
                    if let Some(idecl) = interfaces.get(&iname) {
                        for im in &idecl.methods {
                            if !provided.contains(&im.name) && im.default_body.is_some() {
                                let sub = iface_method_to_fn_decl(im, &imp.target_ty);
                                impl_methods
                                    .entry((tname.clone(), im.name.clone()))
                                    .or_default()
                                    .push((iname.clone(), sub));
                            }
                        }
                    }
                } else {
                    for m in &imp.methods {
                        let mut sub = subst_fn_self(m, &imp.target_ty);
                        // Prepend impl-level type params (e.g. `T` on `impl[T] List[T]`) before method-level type params (e.g. `U` on `fn map[U]`)
                        let mut combined_tps = Vec::new();
                        for tp in &imp.type_params {
                            if !combined_tps.contains(tp) {
                                combined_tps.push(tp.clone());
                            }
                        }
                        for tp in &sub.type_params {
                            if !combined_tps.contains(tp) {
                                combined_tps.push(tp.clone());
                            }
                        }
                        sub.type_params = combined_tps;
                        for (tp, bs) in &imp.type_param_bounds {
                            sub.type_param_bounds
                                .entry(tp.clone())
                                .or_default()
                                .extend(bs.iter().cloned());
                        }
                        inherent_methods.insert((tname.clone(), m.name.clone()), sub);
                    }
                }
            }
        }

        // Compute fixpoint subset of `pure_fns` that are strictly scalar (Int/Bool) and SMT-encodable
        let is_scalar_smt_ty = |ty: &TypeExpr| -> bool {
            let mut cur = ty;
            for _ in 0..8 {
                if let TypeExpr::Named { name, args, .. } = cur {
                    if !args.is_empty() {
                        return false;
                    }
                    if ScalarKind::from_name(name).is_some_and(|k| k.is_int()) || name == "Bool" {
                        return true;
                    }
                    if let Some(alias) = type_aliases.get(name) {
                        cur = &alias.base;
                        continue;
                    }
                    return false;
                }
                return false;
            }
            false
        };
        let mut smt_pure_fns: BTreeMap<String, FnDecl> = BTreeMap::new();
        loop {
            let mut added = false;
            for (name, f) in &pure_fns {
                if smt_pure_fns.contains_key(name) {
                    continue;
                }
                let ret_ok = f.ret_ty.as_ref().is_some_and(&is_scalar_smt_ty);
                let params_ok = f.params.iter().all(|p| is_scalar_smt_ty(&p.ty));
                if !ret_ok || !params_ok || !crate::smt::can_encode_pure_fn(f) {
                    continue;
                }
                let callees = crate::smt::pure_fn_callees(f);
                if callees.iter().all(|c| c != name && smt_pure_fns.contains_key(c)) {
                    smt_pure_fns.insert(name.clone(), f.clone());
                    added = true;
                }
            }
            if !added {
                break;
            }
        }

        let mut consts: BTreeMap<String, (ConstDecl, Option<i64>, Option<f64>)> = BTreeMap::new();
        for c in raw_consts {
            let iv = eval_const_int(&c.value, &consts, &layout);
            let fv = eval_const_float(&c.value, &consts);
            consts.insert(c.name.clone(), (c, iv, fv));
        }

        Self {
            type_aliases,
            records,
            enums,
            enum_variants,
            bitfields,
            unions,
            extern_fns,
            consts,
            layout,
            interfaces,
            inherent_methods,
            interface_impls,
            impl_methods,
            pure_fns,
            smt_pure_fns,
            no_heap_fns,
            all_fns,
            async_fns,
            cur_type_params: BTreeSet::new(),
            cur_type_param_bounds: BTreeMap::new(),
            in_unsafe: false,
            cur_file_id: 0,
            proofs: Vec::new(),
            diagnostics: Vec::new(),
        }
    }

    fn check_all(&mut self, program: &Program) {
        for item in &program.items {
            match item {
                Item::Bitfield(bf) => self.check_bitfield_decl(bf),
                Item::Const(c) => self.check_const_decl(c),
                Item::Fn(f) => self.check_fn(f),
                Item::Impl(imp) => self.check_impl(imp),
                _ => {}
            }
        }
    }

    fn check_bitfield_decl(&mut self, bf: &BitfieldDecl) {
        let backing = self.resolve_base_type(&bf.backing_ty);
        let sk = match &backing {
            TypeExpr::Named { name, .. } => ScalarKind::from_name(name),
            _ => None,
        };
        let Some(sk) = sk.filter(|k| k.is_unsigned()) else {
            self.diagnostics.push(Diagnostic {
                code: "E0610".into(),
                message: format!(
                    "bitfield `{}` backing type `{}` must be an unsigned integer (UInt8/UInt16/UInt32/UInt64)",
                    bf.name,
                    bf.backing_ty.display_name()
                ),
                span: bf.span,
                notes: vec![],
                counterexample: BTreeMap::new(),
            });
            return;
        };
        let total_bits: u32 = bf.fields.iter().map(|f| f.bits).sum();
        if total_bits > sk.bit_width() {
            self.diagnostics.push(Diagnostic {
                code: "E0610".into(),
                message: format!(
                    "bitfield `{}` uses {} bits, exceeding backing `{}` ({} bits)",
                    bf.name,
                    total_bits,
                    bf.backing_ty.display_name(),
                    sk.bit_width()
                ),
                span: bf.span,
                notes: vec![],
                counterexample: BTreeMap::new(),
            });
        } else {
            self.proofs.push(ProofRecord {
                obligation: format!(
                    "bitfield `{}` ({} / {} bits) fits in `{}`",
                    bf.name,
                    total_bits,
                    sk.bit_width(),
                    bf.backing_ty.display_name()
                ),
                tier: ProofTier::L0Fast,
                span: bf.span,
            });
        }
    }

    fn check_const_decl(&mut self, c: &ConstDecl) {
        let vt = BTreeMap::new();
        let actual = self.infer_expr_type_expected(&c.value, Some(&c.ty), &vt);
        if !self.types_compatible(&c.ty, &actual) {
            self.diagnostics.push(Diagnostic {
                code: "E0205".into(),
                message: format!(
                    "const `{}` expects `{}`, found `{}`",
                    c.name,
                    c.ty.display_name(),
                    actual.display_name()
                ),
                span: c.span,
                notes: vec![],
                counterexample: BTreeMap::new(),
            });
        }
    }

    fn type_implements(&self, ty: &TypeExpr, iface_name: &str) -> bool {
        let base = self.resolve_base_type(ty);
        match &base {
            TypeExpr::Dyn { interface, .. } => interface == iface_name,
            TypeExpr::Named { name, .. } => {
                if name == "Any" {
                    return true;
                }
                if self.interface_impls.contains_key(&(iface_name.to_string(), name.clone())) {
                    return true;
                }
                if let TypeExpr::Named { name: orig, .. } = ty {
                    if self.interface_impls.contains_key(&(iface_name.to_string(), orig.clone())) {
                        return true;
                    }
                }
                if let Some(bounds) = self.cur_type_param_bounds.get(name) {
                    if bounds.iter().any(|b| b == iface_name) {
                        return true;
                    }
                }
                false
            }
            _ => false,
        }
    }

    fn is_object_safe(&self, iface_name: &str) -> Result<(), String> {
        let Some(iface) = self.interfaces.get(iface_name) else {
            return Err(format!("unknown interface `{}`", iface_name));
        };
        for m in &iface.methods {
            if !m.params.first().is_some_and(|p| p.name == "self") {
                return Err(format!(
                    "interface `{}` is not object-safe: method `{}` has no `self` receiver",
                    iface_name, m.name
                ));
            }
            if !m.type_params.is_empty() {
                return Err(format!(
                    "interface `{}` is not object-safe: method `{}` has generic type parameters",
                    iface_name, m.name
                ));
            }
            for p in m.params.iter().skip(1) {
                if p.ty.display_name().contains("Self") {
                    return Err(format!(
                        "interface `{}` is not object-safe: parameter `{}` of `{}` mentions `Self`",
                        iface_name, p.name, m.name
                    ));
                }
            }
            if let Some(rt) = &m.ret_ty {
                if rt.display_name().contains("Self") {
                    return Err(format!(
                        "interface `{}` is not object-safe: return type of `{}` mentions `Self`",
                        iface_name, m.name
                    ));
                }
            }
        }
        Ok(())
    }

    fn check_dyn_type_valid(&mut self, ty: &TypeExpr, span: Span) {
        match ty {
            TypeExpr::Dyn { interface, args, .. } => {
                if !self.interfaces.contains_key(interface) {
                    self.diagnostics.push(Diagnostic {
                        code: "E0500".into(),
                        message: format!("unknown interface `{}` in `dyn {}`", interface, interface),
                        span,
                        notes: vec![],
                        counterexample: BTreeMap::new(),
                    });
                } else if let Err(reason) = self.is_object_safe(interface) {
                    self.diagnostics.push(Diagnostic {
                        code: "E0506".into(),
                        message: reason,
                        span,
                        notes: vec!["use static generic constraint `[T Interface]` instead of `dyn Interface`".into()],
                        counterexample: BTreeMap::new(),
                    });
                }
                for a in args {
                    self.check_dyn_type_valid(a, span);
                }
            }
            TypeExpr::Named { args, .. } => {
                for a in args {
                    self.check_dyn_type_valid(a, span);
                }
            }
            TypeExpr::Tuple(es, _) => {
                for e in es {
                    self.check_dyn_type_valid(e, span);
                }
            }
            TypeExpr::Fn { params, ret, .. } => {
                for p in params {
                    self.check_dyn_type_valid(p, span);
                }
                if let Some(r) = ret {
                    self.check_dyn_type_valid(r, span);
                }
            }
            TypeExpr::Array { elem, .. } => {
                self.check_dyn_type_valid(elem, span);
            }
        }
    }

    fn resolve_method_sig(&self, recv_ty: &TypeExpr, method_name: &str) -> Option<FnDecl> {
        let base = self.resolve_base_type(recv_ty);
        match &base {
            TypeExpr::Dyn { interface, .. } => {
                let idecl = self.interfaces.get(interface)?;
                let m = idecl.methods.iter().find(|m| m.name == method_name)?;
                Some(iface_method_to_fn_decl(m, recv_ty))
            }
            TypeExpr::Named { name, args, .. } => {
                // 1. Bounded generic type parameter `T` -> check bound interfaces
                if let Some(bounds) = self.cur_type_param_bounds.get(name) {
                    for b in bounds {
                        if let Some(idecl) = self.interfaces.get(b) {
                            if let Some(m) = idecl.methods.iter().find(|m| m.name == method_name) {
                                return Some(iface_method_to_fn_decl(m, recv_ty));
                            }
                        }
                    }
                }
                // 2. Inherent method on exact or base type name
                let lookup_names: Vec<String> = match recv_ty {
                    TypeExpr::Named { name: orig, .. } if orig != name => {
                        vec![orig.clone(), name.clone()]
                    }
                    _ => vec![name.clone()],
                };
                for tn in &lookup_names {
                    if let Some(m) = self.inherent_methods.get(&(tn.clone(), method_name.to_string())) {
                        return Some(self.instantiate_container_method(m, tn, args));
                    }
                }
                // 3. Interface impl method on concrete type
                for tn in &lookup_names {
                    if let Some(list) = self.impl_methods.get(&(tn.clone(), method_name.to_string())) {
                        if let Some((_, m)) = list.first() {
                            return Some(self.instantiate_container_method(m, tn, args));
                        }
                    }
                }
                None
            }
            _ => None,
        }
    }

    fn instantiate_container_method(&self, m: &FnDecl, _tn: &str, type_args: &[TypeExpr]) -> FnDecl {
        if type_args.is_empty() || m.type_params.is_empty() {
            return m.clone();
        }
        let mut subst = BTreeMap::new();
        for (tp, ta) in m.type_params.iter().zip(type_args.iter()) {
            subst.insert(tp.clone(), ta.clone());
        }
        let mut out = m.clone();
        for p in &mut out.params {
            p.ty = subst_type_expr(&p.ty, &subst);
        }
        if let Some(rt) = &out.ret_ty {
            out.ret_ty = Some(subst_type_expr(rt, &subst));
        }
        // Keep only remaining unbound method-level type parameters (e.g. `U` in `List[T].map[U]`)
        out.type_params = m.type_params.iter().skip(type_args.len()).cloned().collect();
        out
    }

    fn resolve_assoc_method_sig(&self, type_name: &str, method_name: &str) -> Option<FnDecl> {
        // Check bounded generic type param `T.from_json(...)`
        if let Some(bounds) = self.cur_type_param_bounds.get(type_name) {
            for b in bounds {
                if let Some(idecl) = self.interfaces.get(b) {
                    if let Some(m) = idecl.methods.iter().find(|m| m.name == method_name) {
                        return Some(iface_method_to_fn_decl(m, &named_ty(type_name)));
                    }
                }
            }
        }
        if let Some(m) = self.inherent_methods.get(&(type_name.to_string(), method_name.to_string())) {
            return Some(m.clone());
        }
        if let Some(list) = self.impl_methods.get(&(type_name.to_string(), method_name.to_string())) {
            if let Some((_, m)) = list.first() {
                return Some(m.clone());
            }
        }
        None
    }

    fn unify_type_pattern(
        pat: &TypeExpr,
        actual: &TypeExpr,
        tps: &[String],
        subst: &mut BTreeMap<String, TypeExpr>,
    ) {
        match (pat, actual) {
            (TypeExpr::Named { name, args, .. }, _)
                if args.is_empty() && tps.contains(name) =>
            {
                if actual.display_name() != "Any" {
                    subst.entry(name.clone()).or_insert_with(|| actual.clone());
                }
            }
            (
                TypeExpr::Named { name: pn, args: pa, .. },
                TypeExpr::Named { name: an, args: aa, .. },
            ) if pn == an && pa.len() == aa.len() => {
                for (x, y) in pa.iter().zip(aa.iter()) {
                    Self::unify_type_pattern(x, y, tps, subst);
                }
            }
            (TypeExpr::Tuple(pe, _), TypeExpr::Tuple(ae, _)) if pe.len() == ae.len() => {
                for (x, y) in pe.iter().zip(ae.iter()) {
                    Self::unify_type_pattern(x, y, tps, subst);
                }
            }
            (
                TypeExpr::Fn { params: pp, ret: pr, .. },
                TypeExpr::Fn { params: ap, ret: ar, .. },
            ) if pp.len() == ap.len() => {
                for (x, y) in pp.iter().zip(ap.iter()) {
                    Self::unify_type_pattern(x, y, tps, subst);
                }
                if let (Some(pr_ty), Some(ar_ty)) = (pr, ar) {
                    Self::unify_type_pattern(pr_ty, ar_ty, tps, subst);
                }
            }
            _ => {}
        }
    }

    fn instantiate_method_from_args(
        &self,
        m: &FnDecl,
        has_self: bool,
        args: &[CallArg],
        var_types: &BTreeMap<String, TypeExpr>,
    ) -> FnDecl {
        if m.type_params.is_empty() {
            return m.clone();
        }
        let explicit_params = if has_self && m.params.first().is_some_and(|p| p.name == "self") {
            &m.params[1..]
        } else {
            &m.params[..]
        };
        let mut subst = BTreeMap::new();
        // Pass 1: unify non-lambda arguments first (e.g. `init U` in `fold(init, |acc, x| ...)`!)
        for (p, a) in explicit_params.iter().zip(args.iter()) {
            if !matches!(a.value.kind, ExprKind::Lambda { .. }) {
                let aty = self.infer_expr_type(&a.value, var_types);
                Self::unify_type_pattern(&p.ty, &aty, &m.type_params, &mut subst);
            }
        }
        // Pass 2: unify lambda arguments using expected parameter types after substituting Pass 1!
        for (p, a) in explicit_params.iter().zip(args.iter()) {
            if matches!(a.value.kind, ExprKind::Lambda { .. }) {
                let expected_pty = subst_type_expr(&p.ty, &subst);
                let aty = self.infer_expr_type_expected(&a.value, Some(&expected_pty), var_types);
                Self::unify_type_pattern(&p.ty, &aty, &m.type_params, &mut subst);
            }
        }
        for tp in &m.type_params {
            subst.entry(tp.clone()).or_insert_with(|| named_ty("Any"));
        }
        let mut out = m.clone();
        for p in &mut out.params {
            p.ty = subst_type_expr(&p.ty, &subst);
        }
        if let Some(rt) = &out.ret_ty {
            out.ret_ty = Some(subst_type_expr(rt, &subst));
        }
        out.type_params.clear();
        out
    }

    fn instantiate_assoc_method_from_args(
        &self,
        m: &FnDecl,
        args: &[CallArg],
        var_types: &BTreeMap<String, TypeExpr>,
    ) -> FnDecl {
        self.instantiate_method_from_args(m, false, args, var_types)
    }

    fn check_type_bounds(&mut self, ty: &TypeExpr, span: Span) {
        match ty {
            TypeExpr::Named { name, args, .. } => {
                if let Some(rec) = self.records.get(name).cloned() {
                    for (tp, ta) in rec.type_params.iter().zip(args.iter()) {
                        if ta.display_name() != "Any" && !self.cur_type_params.contains(&ta.display_name()) {
                            if let Some(bounds) = rec.type_param_bounds.get(tp) {
                                for b in bounds {
                                    if !self.type_implements(ta, b) {
                                        self.diagnostics.push(Diagnostic {
                                            code: "E0505".into(),
                                            message: format!(
                                                "type `{}` does not satisfy interface bound `{}` for type parameter `{}` of `{}`",
                                                ta.display_name(),
                                                b,
                                                tp,
                                                name
                                            ),
                                            span,
                                            notes: vec![format!("add `impl {} for {} {{ ... }}`", b, ta.display_name())],
                                            counterexample: BTreeMap::new(),
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
                if let Some(en) = self.enums.get(name).cloned() {
                    for (tp, ta) in en.type_params.iter().zip(args.iter()) {
                        if ta.display_name() != "Any" && !self.cur_type_params.contains(&ta.display_name()) {
                            if let Some(bounds) = en.type_param_bounds.get(tp) {
                                for b in bounds {
                                    if !self.type_implements(ta, b) {
                                        self.diagnostics.push(Diagnostic {
                                            code: "E0505".into(),
                                            message: format!(
                                                "type `{}` does not satisfy interface bound `{}` for type parameter `{}` of `{}`",
                                                ta.display_name(),
                                                b,
                                                tp,
                                                name
                                            ),
                                            span,
                                            notes: vec![format!("add `impl {} for {} {{ ... }}`", b, ta.display_name())],
                                            counterexample: BTreeMap::new(),
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
                for a in args {
                    self.check_type_bounds(a, span);
                }
            }
            TypeExpr::Tuple(es, _) => {
                for e in es {
                    self.check_type_bounds(e, span);
                }
            }
            TypeExpr::Fn { params, ret, .. } => {
                for p in params {
                    self.check_type_bounds(p, span);
                }
                if let Some(r) = ret {
                    self.check_type_bounds(r, span);
                }
            }
            TypeExpr::Dyn { args, .. } => {
                for a in args {
                    self.check_type_bounds(a, span);
                }
            }
            TypeExpr::Array { elem, .. } => {
                self.check_type_bounds(elem, span);
            }
        }
    }

    fn is_known_type_name(&self, name: &str) -> bool {
        self.records.contains_key(name)
            || self.enums.contains_key(name)
            || self.bitfields.contains_key(name)
            || self.unions.contains_key(name)
            || self.type_aliases.contains_key(name)
            || self.cur_type_params.contains(name)
            || ScalarKind::from_name(name).is_some()
            || SimdKind::from_name(name).is_some()
            || matches!(
                name,
                "Bool"
                    | "Text"
                    | "Bytes"
                    | "BytesSlice"
                    | "Option"
                    | "Result"
                    | "List"
                    | "Range"
                    | "Array"
                    | "Ptr"
                    | "MutPtr"
                    | "CStr"
                    | "CVoid"
                    | "AtomicInt"
                    | "Mutex"
                    | "Channel"
                    | "Simd"
            )
    }

    fn check_impl(&mut self, imp: &ImplDecl) {
        let tname = type_head_name(&imp.target_ty);
        let prev_tp = self.cur_type_params.clone();
        let prev_bounds = self.cur_type_param_bounds.clone();
        for tp in &imp.type_params {
            self.cur_type_params.insert(tp.clone());
        }
        for (k, v) in &imp.type_param_bounds {
            self.cur_type_param_bounds.insert(k.clone(), v.clone());
        }

        if let Some(iface_ty) = &imp.interface {
            let iname = type_head_name(iface_ty);
            let Some(idecl) = self.interfaces.get(&iname).cloned() else {
                self.diagnostics.push(Diagnostic {
                    code: "E0500".into(),
                    message: format!("unknown interface `{}` in `impl {} for {}`", iname, iname, tname),
                    span: imp.span,
                    notes: vec![],
                    counterexample: BTreeMap::new(),
                });
                self.cur_type_params = prev_tp;
                self.cur_type_param_bounds = prev_bounds;
                return;
            };

            let provided_map: BTreeMap<String, FnDecl> = imp
                .methods
                .iter()
                .map(|m| (m.name.clone(), m.clone()))
                .collect();

            // 1. Verify all required interface methods are implemented & conform to signature, effects, and Liskov contracts!
            for im in &idecl.methods {
                let expected = iface_method_to_fn_decl(im, &imp.target_ty);
                let Some(impl_m) = provided_map.get(&im.name) else {
                    if im.default_body.is_none() {
                        self.diagnostics.push(Diagnostic {
                            code: "E0501".into(),
                            message: format!(
                                "missing method `{}` in `impl {} for {}`",
                                im.name, iname, tname
                            ),
                            span: imp.span,
                            notes: vec![format!("interface `{}` requires method `{}`", iname, im.name)],
                            counterexample: BTreeMap::new(),
                        });
                    }
                    continue;
                };
                let mut concrete_m = subst_fn_self(impl_m, &imp.target_ty);

                // Signature compatibility: arity, `inout`, parameter types, return type
                if concrete_m.params.len() != expected.params.len() {
                    self.diagnostics.push(Diagnostic {
                        code: "E0502".into(),
                        message: format!(
                            "method `{}` in `impl {} for {}` expects {} parameter(s), found {}",
                            im.name,
                            iname,
                            tname,
                            expected.params.len(),
                            concrete_m.params.len()
                        ),
                        span: impl_m.span,
                        notes: vec![],
                        counterexample: BTreeMap::new(),
                    });
                } else {
                    for (ep, ap) in expected.params.iter().zip(concrete_m.params.iter()) {
                        if ep.is_inout != ap.is_inout {
                            self.diagnostics.push(Diagnostic {
                                code: "E0502".into(),
                                message: format!(
                                    "method `{}` in `impl {} for {}`: parameter `{}` `inout` mismatch with interface `{}`",
                                    im.name, iname, tname, ap.name, iname
                                ),
                                span: ap.span,
                                notes: vec![],
                                counterexample: BTreeMap::new(),
                            });
                        }
                        if !self.types_compatible(&ep.ty, &ap.ty) {
                            self.diagnostics.push(Diagnostic {
                                code: "E0502".into(),
                                message: format!(
                                    "method `{}` in `impl {} for {}`: parameter `{}` expects `{}`, found `{}`",
                                    im.name,
                                    iname,
                                    tname,
                                    ap.name,
                                    ep.ty.display_name(),
                                    ap.ty.display_name()
                                ),
                                span: ap.span,
                                notes: vec![],
                                counterexample: BTreeMap::new(),
                            });
                        }
                    }
                }
                match (&expected.ret_ty, &concrete_m.ret_ty) {
                    (Some(ert), Some(art)) if !self.types_compatible(ert, art) => {
                        self.diagnostics.push(Diagnostic {
                            code: "E0502".into(),
                            message: format!(
                                "method `{}` in `impl {} for {}`: return type expects `{}`, found `{}`",
                                im.name,
                                iname,
                                tname,
                                ert.display_name(),
                                art.display_name()
                            ),
                            span: impl_m.span,
                            notes: vec![],
                            counterexample: BTreeMap::new(),
                        });
                    }
                    (Some(ert), None) => {
                        self.diagnostics.push(Diagnostic {
                            code: "E0502".into(),
                            message: format!(
                                "method `{}` in `impl {} for {}` must return `{}`",
                                im.name,
                                iname,
                                tname,
                                ert.display_name()
                            ),
                            span: impl_m.span,
                            notes: vec![],
                            counterexample: BTreeMap::new(),
                        });
                    }
                    _ => {}
                }

                // Effect discipline: interface `pure` / `@no_heap_alloc` / `async` MUST be honored!
                if expected.is_pure && !concrete_m.is_pure {
                    self.diagnostics.push(Diagnostic {
                        code: "E0503".into(),
                        message: format!(
                            "effect mismatch in `impl {} for {}`: interface method `{}` requires `pure`",
                            iname, tname, im.name
                        ),
                        span: impl_m.span,
                        notes: vec!["mark the implementation `pure fn`".into()],
                        counterexample: BTreeMap::new(),
                    });
                }
                let iface_no_heap = expected.attrs.iter().any(|a| a.name == "no_heap_alloc");
                let impl_no_heap = concrete_m.attrs.iter().any(|a| a.name == "no_heap_alloc");
                if iface_no_heap && !impl_no_heap {
                    self.diagnostics.push(Diagnostic {
                        code: "E0503".into(),
                        message: format!(
                            "effect mismatch in `impl {} for {}`: interface method `{}` requires `@no_heap_alloc`",
                            iname, tname, im.name
                        ),
                        span: impl_m.span,
                        notes: vec!["annotate the implementation with `@no_heap_alloc`".into()],
                        counterexample: BTreeMap::new(),
                    });
                }
                if expected.is_async != concrete_m.is_async {
                    self.diagnostics.push(Diagnostic {
                        code: "E0503".into(),
                        message: format!(
                            "effect mismatch in `impl {} for {}`: `async` modifier of `{}` does not match interface `{}`",
                            iname, tname, im.name, iname
                        ),
                        span: impl_m.span,
                        notes: vec![],
                        counterexample: BTreeMap::new(),
                    });
                }

                // Liskov behavioural subtyping: merge interface `requires` (available to impl)
                // and interface `ensures` (MUST be proved by impl body via L0/L1 Z3!)
                for req in &expected.requires {
                    concrete_m.requires.push(req.clone());
                }
                for ens in &expected.ensures {
                    concrete_m.ensures.push(ens.clone());
                }
                concrete_m.name = format!("{}::{}::{}", tname, iname, im.name);
                let diag_before = self.diagnostics.len();
                self.check_fn(&concrete_m);
                if self.diagnostics.len() == diag_before {
                    self.proofs.push(ProofRecord {
                        obligation: format!("impl `{}` for `{}` satisfies `{}` contract", iname, tname, im.name),
                        tier: ProofTier::L0Fast,
                        span: impl_m.span,
                    });
                }
            }
        } else {
            // Inherent `impl Target { ... }`
            for m in &imp.methods {
                let mut concrete_m = subst_fn_self(m, &imp.target_ty);
                concrete_m.name = format!("{}::{}", tname, m.name);
                self.check_fn(&concrete_m);
            }
        }

        self.cur_type_params = prev_tp;
        self.cur_type_param_bounds = prev_bounds;
    }

    fn resolve_base_type(&self, ty: &TypeExpr) -> TypeExpr {
        if let TypeExpr::Named { name, .. } = ty {
            if let Some(alias) = self.type_aliases.get(name) {
                return self.resolve_base_type(&alias.base);
            }
        }
        ty.clone()
    }

    fn scalar_kind_of_name(&self, name: &str) -> Option<ScalarKind> {
        if let Some(sk) = ScalarKind::from_name(name) {
            return Some(sk);
        }
        if let Some(alias) = self.type_aliases.get(name) {
            if let TypeExpr::Named { name: base, .. } = self.resolve_base_type(&alias.base) {
                return ScalarKind::from_name(&base);
            }
        }
        None
    }

    fn scalar_kind_of_ty(&self, ty: &TypeExpr) -> Option<ScalarKind> {
        match self.resolve_base_type(ty) {
            TypeExpr::Named { ref name, .. } => ScalarKind::from_name(name),
            _ => None,
        }
    }

    fn is_int_type_name(&self, name: &str) -> bool {
        self.scalar_kind_of_name(name).is_some_and(|k| k.is_int())
    }

    fn is_float_type_name(&self, name: &str) -> bool {
        self.scalar_kind_of_name(name).is_some_and(|k| k.is_float())
    }

    fn is_float_ty(&self, ty: &TypeExpr) -> bool {
        matches!(
            self.resolve_base_type(ty),
            TypeExpr::Named { ref name, .. } if self.is_float_type_name(name)
        )
    }

    fn is_int_ty(&self, ty: &TypeExpr) -> bool {
        matches!(
            self.resolve_base_type(ty),
            TypeExpr::Named { ref name, .. } if self.is_int_type_name(name)
        )
    }

    fn expr_smt_encodable(&self, expr: &Expr) -> bool {
        match &expr.kind {
            ExprKind::Int(_) | ExprKind::Bool(_) | ExprKind::Ident(_) => true,
            ExprKind::Field { target, .. } => matches!(&target.kind, ExprKind::Ident(_)),
            ExprKind::Unary { expr: inner, .. } => self.expr_smt_encodable(inner),
            ExprKind::Binary { op, left, right } => {
                !matches!(
                    op,
                    BinOp::Concat | BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr
                ) && self.expr_smt_encodable(left)
                    && self.expr_smt_encodable(right)
            }
            ExprKind::Call { callee, args } => {
                let ExprKind::Ident(fname) = &callee.kind else {
                    return false;
                };
                self.smt_pure_fns.contains_key(fname)
                    && args.iter().all(|a| self.expr_smt_encodable(&a.value))
            }
            _ => false,
        }
    }

    fn types_compatible(&self, expected: &TypeExpr, actual: &TypeExpr) -> bool {
        let exp = self.resolve_base_type(expected);
        let act = self.resolve_base_type(actual);
        match (&exp, &act) {
            (TypeExpr::Named { name: en, .. }, _) if en == "Any" => true,
            (_, TypeExpr::Named { name: an, .. }) if an == "Any" => true,
            (TypeExpr::Named { name: en, .. }, _) if en == "AnyInt" => self.is_int_ty(&act),
            (TypeExpr::Dyn { interface: ei, .. }, TypeExpr::Dyn { interface: ai, .. }) => ei == ai,
            (TypeExpr::Dyn { interface: ei, .. }, _) => self.type_implements(actual, ei),
            (
                TypeExpr::Array { elem: ee, len: el, .. },
                TypeExpr::Array { elem: ae, len: al, .. },
            ) => el == al && self.types_compatible(ee, ae),
            (
                TypeExpr::Named { name: en, args: ea, .. },
                TypeExpr::Named { name: an, args: aa, .. },
            ) => {
                if self.cur_type_params.contains(en) || self.cur_type_params.contains(an) {
                    return true;
                }
                if let (Some(ek), Some(ak)) = (self.scalar_kind_of_name(en), self.scalar_kind_of_name(an)) {
                    return ek == ak;
                }
                if let (Some(es), Some(as_)) = (SimdKind::from_type_expr(&exp), SimdKind::from_type_expr(&act)) {
                    return es == as_;
                }
                // C pointer interop: `MutPtr[T]` coerces to `Ptr[T]`; `CStr` <-> `Ptr[UInt8]`; `CVoid` ptr wildcard
                if en == "CStr" && (an == "Ptr" || an == "MutPtr") {
                    return true;
                }
                if (en == "Ptr" || en == "MutPtr") && an == "CStr" {
                    return true;
                }
                if en == "Ptr" && an == "MutPtr" {
                    if ea.is_empty() || aa.is_empty() {
                        return true;
                    }
                    if ea[0].display_name() == "CVoid" || aa[0].display_name() == "CVoid" {
                        return true;
                    }
                    return self.types_compatible(&ea[0], &aa[0]);
                }
                if (en == "Ptr" || en == "MutPtr") && en == an {
                    if ea.is_empty() || aa.is_empty() {
                        return true;
                    }
                    if ea[0].display_name() == "CVoid" || aa[0].display_name() == "CVoid" {
                        return true;
                    }
                    return self.types_compatible(&ea[0], &aa[0]);
                }
                if en != an {
                    return false;
                }
                if ea.is_empty() || aa.is_empty() {
                    return true;
                }
                if ea.len() != aa.len() {
                    return false;
                }
                ea.iter()
                    .zip(aa.iter())
                    .all(|(x, y)| self.types_compatible(x, y))
            }
            (TypeExpr::Tuple(es, _), TypeExpr::Tuple(as_, _)) => {
                es.len() == as_.len()
                    && es
                        .iter()
                        .zip(as_.iter())
                        .all(|(x, y)| self.types_compatible(x, y))
            }
            (
                TypeExpr::Fn { params: ep, ret: er, .. },
                TypeExpr::Fn { params: ap, ret: ar, .. },
            ) => {
                ep.len() == ap.len()
                    && ep
                        .iter()
                        .zip(ap.iter())
                        .all(|(x, y)| self.types_compatible(x, y))
                    && match (er, ar) {
                        (None, _) => true,
                        (Some(e), Some(a)) => self.types_compatible(e, a),
                        (Some(e), None) => e.display_name() == "Any",
                    }
            }
            _ => false,
        }
    }

    fn infer_expr_type(
        &self,
        expr: &Expr,
        var_types: &BTreeMap<String, TypeExpr>,
    ) -> TypeExpr {
        self.infer_expr_type_expected(expr, None, var_types)
    }

    fn infer_expr_type_expected(
        &self,
        expr: &Expr,
        expected: Option<&TypeExpr>,
        var_types: &BTreeMap<String, TypeExpr>,
    ) -> TypeExpr {
        match &expr.kind {
            ExprKind::Int(_) => {
                if let Some(exp) = expected {
                    if self.scalar_kind_of_ty(exp).is_some_and(|k| k.is_int()) {
                        return exp.clone();
                    }
                }
                named_ty("Int")
            }
            ExprKind::Float(_) => {
                if let Some(exp) = expected {
                    if self.scalar_kind_of_ty(exp).is_some_and(|k| k.is_float()) {
                        return exp.clone();
                    }
                }
                named_ty("Float")
            }
            ExprKind::Bool(_) => named_ty("Bool"),
            ExprKind::Str(_) => named_ty("Text"),
            ExprKind::Ident(s) => {
                if s == "None" {
                    return generic_ty("Option", vec![named_ty("Any")]);
                }
                if let Some(ty) = var_types.get(s) {
                    return ty.clone();
                }
                if let Some((c, _, _)) = self.consts.get(s) {
                    return c.ty.clone();
                }
                if let Some(vi) = self.enum_variants.get(s) {
                    if let Some(ed) = self.enums.get(&vi.enum_name) {
                        if !ed.type_params.is_empty() {
                            return generic_ty(
                                &vi.enum_name,
                                vec![named_ty("Any"); ed.type_params.len()],
                            );
                        }
                    }
                    return named_ty(&vi.enum_name);
                }
                named_ty("Any")
            }
            ExprKind::RecordLit { name, fields, .. } => {
                if let Some(rec) = self.records.get(name) {
                    if !rec.type_params.is_empty() {
                        let mut subst = BTreeMap::new();
                        for (fname, fval) in fields {
                            if let Some(fd) = rec.fields.iter().find(|f| f.name == *fname) {
                                let aty = self.infer_expr_type(fval, var_types);
                                Self::unify_type_pattern(&fd.ty, &aty, &rec.type_params, &mut subst);
                            }
                        }
                        let targs: Vec<TypeExpr> = rec
                            .type_params
                            .iter()
                            .map(|tp| subst.get(tp).cloned().unwrap_or_else(|| named_ty("Any")))
                            .collect();
                        return generic_ty(name, targs);
                    }
                }
                named_ty(name)
            }
            ExprKind::List(elems) => {
                if let Some(TypeExpr::Array { elem, len, .. }) =
                    expected.map(|t| self.resolve_base_type(t))
                {
                    if elems.len() == len {
                        return TypeExpr::Array {
                            elem,
                            len,
                            span: expr.span,
                        };
                    }
                }
                if let Some(first) = elems.first() {
                    generic_ty("List", vec![self.infer_expr_type(first, var_types)])
                } else {
                    generic_ty("List", vec![named_ty("Any")])
                }
            }
            ExprKind::ArrayRepeat { value, len } => {
                let exp_elem = match expected.map(|t| self.resolve_base_type(t)) {
                    Some(TypeExpr::Array { elem, .. }) => Some(*elem),
                    _ => None,
                };
                let ety = self.infer_expr_type_expected(value, exp_elem.as_ref(), var_types);
                TypeExpr::Array {
                    elem: Box::new(ety),
                    len: *len,
                    span: expr.span,
                }
            }
            ExprKind::Cast { ty, .. } => ty.clone(),
            ExprKind::Unsafe(b) => {
                if let Some(t) = &b.tail {
                    self.infer_expr_type_expected(t, expected, var_types)
                } else {
                    named_ty("Any")
                }
            }
            ExprKind::Tuple(elems) => TypeExpr::Tuple(
                elems.iter().map(|e| self.infer_expr_type(e, var_types)).collect(),
                expr.span,
            ),
            ExprKind::TupleIndex { target, index } => {
                let t = self.resolve_base_type(&self.infer_expr_type(target, var_types));
                if let TypeExpr::Tuple(elems, _) = t {
                    if let Some(et) = elems.get(*index) {
                        return et.clone();
                    }
                }
                named_ty("Any")
            }
            ExprKind::Range { .. } => generic_ty("Range", vec![named_ty("Int")]),
            ExprKind::Try(inner) => {
                let inner_ty = self.resolve_base_type(&self.infer_expr_type(inner, var_types));
                if let TypeExpr::Named { name, args, .. } = inner_ty {
                    if (name == "Result" || name == "Option") && !args.is_empty() {
                        return args[0].clone();
                    }
                }
                named_ty("Any")
            }
            ExprKind::Lambda { params, ret_ty, body } => {
                let (exp_params, exp_ret): (Vec<TypeExpr>, Option<TypeExpr>) = match expected
                    .map(|t| self.resolve_base_type(t))
                {
                    Some(TypeExpr::Fn { params: ep, ret: er, .. }) => {
                        (ep, er.map(|r| *r))
                    }
                    _ => (Vec::new(), None),
                };
                let mut lam_vars = var_types.clone();
                let mut resolved_params = Vec::with_capacity(params.len());
                for (i, p) in params.iter().enumerate() {
                    let pty = p
                        .ty
                        .clone()
                        .or_else(|| exp_params.get(i).cloned())
                        .unwrap_or_else(|| named_ty("Any"));
                    if p.name != "_" {
                        lam_vars.insert(p.name.clone(), pty.clone());
                    }
                    resolved_params.push(pty);
                }
                let inferred_ret = if let Some(rt) = ret_ty {
                    Some(Box::new(rt.clone()))
                } else if matches!(&body.kind, ExprKind::Block(b) if b.tail.is_none()) {
                    None
                } else {
                    Some(Box::new(self.infer_expr_type_expected(
                        body,
                        exp_ret.as_ref(),
                        &lam_vars,
                    )))
                };
                TypeExpr::Fn {
                    params: resolved_params,
                    ret: inferred_ret,
                    span: expr.span,
                }
            }
            ExprKind::Unary { op, expr: inner } => match op {
                UnOp::Neg | UnOp::BitNot => {
                    let it = self.infer_expr_type_expected(inner, expected, var_types);
                    if self.scalar_kind_of_ty(&it).is_some() {
                        it
                    } else if matches!(op, UnOp::Neg) && self.is_float_ty(&it) {
                        named_ty("Float")
                    } else {
                        named_ty("Int")
                    }
                }
                UnOp::Not => named_ty("Bool"),
            },
            ExprKind::Binary { op, left, right } => match op {
                BinOp::Concat => named_ty("Text"),
                BinOp::Add
                | BinOp::Sub
                | BinOp::Mul
                | BinOp::Div
                | BinOp::Mod
                | BinOp::BitAnd
                | BinOp::BitOr
                | BinOp::BitXor
                | BinOp::Shl
                | BinOp::Shr => {
                    let lt = self.infer_expr_type_expected(left, expected, var_types);
                    let rt = self.infer_expr_type_expected(right, Some(&lt), var_types);
                    if matches!(left.kind, ExprKind::Int(_) | ExprKind::Float(_))
                        && self.scalar_kind_of_ty(&rt).is_some()
                    {
                        rt
                    } else if self.scalar_kind_of_ty(&lt).is_some() {
                        lt
                    } else if self.scalar_kind_of_ty(&rt).is_some() {
                        rt
                    } else {
                        named_ty("Int")
                    }
                }
                BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::And | BinOp::Or | BinOp::Implies => named_ty("Bool"),
            },
            ExprKind::Field { target, field } => {
                if let ExprKind::Ident(ename) = &target.kind {
                    if let Some(vi) = self.enum_variants.get(&format!("{}.{}", ename, field)) {
                        if let Some(ed) = self.enums.get(&vi.enum_name) {
                            if !ed.type_params.is_empty() {
                                return generic_ty(
                                    &vi.enum_name,
                                    vec![named_ty("Any"); ed.type_params.len()],
                                );
                            }
                        }
                        return named_ty(&vi.enum_name);
                    }
                }
                let t = self.resolve_base_type(&self.infer_expr_type(target, var_types));
                if let TypeExpr::Named { name: rname, args: rargs, .. } = t {
                    if let Some(rec) = self.records.get(&rname) {
                        if let Some(f) = rec.fields.iter().find(|f| f.name == *field) {
                            if !rec.type_params.is_empty() && !rargs.is_empty() {
                                let mut fsubst = BTreeMap::new();
                                for (tp, ra) in rec.type_params.iter().zip(rargs.iter()) {
                                    fsubst.insert(tp.clone(), ra.clone());
                                }
                                return subst_type_expr(&f.ty, &fsubst);
                            }
                            return f.ty.clone();
                        }
                    }
                    if let Some(bf) = self.bitfields.get(&rname) {
                        if bf.fields.iter().any(|f| f.name == *field) {
                            return bf.backing_ty.clone();
                        }
                    }
                    if let Some(u) = self.unions.get(&rname) {
                        if let Some(f) = u.fields.iter().find(|f| f.name == *field) {
                            return f.ty.clone();
                        }
                    }
                }
                named_ty("Any")
            }
            ExprKind::Index { target, .. } => {
                let t = self.resolve_base_type(&self.infer_expr_type(target, var_types));
                match t {
                    TypeExpr::Array { elem, .. } => *elem,
                    TypeExpr::Named { name, args, .. } if name == "List" && !args.is_empty() => {
                        args[0].clone()
                    }
                    _ => named_ty("Any"),
                }
            }
            ExprKind::Call { callee, args } => match &callee.kind {
                ExprKind::Field { target, field }
                    if field == "parse"
                        && matches!(&target.kind, ExprKind::Ident(rname) if self.records.contains_key(rname) && self.resolve_assoc_method_sig(rname, "parse").is_none()) =>
                {
                    if let ExprKind::Ident(rname) = &target.kind {
                        return generic_ty("Result", vec![named_ty(rname), named_ty("Int")]);
                    }
                    named_ty("Any")
                }
                ExprKind::Field { target, field } => {
                    if let ExprKind::GenericInst { expr: inner, type_args } = &target.kind {
                        if let ExprKind::Ident(tname) = &inner.kind {
                            if !var_types.contains_key(tname) && self.is_known_type_name(tname) {
                                if let Some(m) = self.resolve_assoc_method_sig(tname, field) {
                                    let inst0 = self.instantiate_container_method(&m, tname, type_args);
                                    let inst = self.instantiate_assoc_method_from_args(&inst0, args, var_types);
                                    return inst.ret_ty.unwrap_or_else(|| named_ty("Any"));
                                }
                            }
                        }
                    }
                    if let ExprKind::Ident(ename) = &target.kind {
                        if let Some(vi) = self.enum_variants.get(&format!("{}.{}", ename, field)) {
                            return self.infer_variant_ctor_type(vi, args, var_types);
                        }
                        if !var_types.contains_key(ename) && self.is_known_type_name(ename) {
                            if field == "from_raw" && self.bitfields.contains_key(ename) {
                                return named_ty(ename);
                            }
                            if let Some(m) = self.resolve_assoc_method_sig(ename, field) {
                                let inst = self.instantiate_assoc_method_from_args(&m, args, var_types);
                                return inst.ret_ty.unwrap_or_else(|| named_ty("Any"));
                            }
                        }
                    }
                    let recv_ty = self.infer_expr_type(target, var_types);
                    if field == "raw" {
                        if let TypeExpr::Named { name: bn, .. } = self.resolve_base_type(&recv_ty) {
                            if let Some(bf) = self.bitfields.get(&bn) {
                                return bf.backing_ty.clone();
                            }
                        }
                    }
                    if let Some(m) = self.resolve_method_sig(&recv_ty, field) {
                        let inst = self.instantiate_method_from_args(&m, true, args, var_types);
                        return inst.ret_ty.unwrap_or_else(|| named_ty("Any"));
                    }
                    // Callable field on a record (e.g. `obj.callback(x)`)
                    let fty = self.resolve_base_type(&self.infer_expr_type(callee, var_types));
                    if let TypeExpr::Fn { ret, .. } = fty {
                        return ret.map(|r| *r).unwrap_or_else(|| named_ty("Any"));
                    }
                    named_ty("Any")
                }
                ExprKind::GenericInst { expr: inner, type_args } => {
                    if let ExprKind::Ident(fname) = &inner.kind {
                        if fname == "size_of" || fname == "align_of" {
                            return named_ty("USize");
                        }
                        if fname == "null_ptr" {
                            return generic_ty(
                                "MutPtr",
                                vec![type_args.first().cloned().unwrap_or_else(|| named_ty("CVoid"))],
                            );
                        }
                        if fname == "chan_new" {
                            return generic_ty(
                                "Channel",
                                vec![type_args.first().cloned().unwrap_or_else(|| named_ty("Any"))],
                            );
                        }
                        if let Some(fdecl) = self.all_fns.get(fname) {
                            if let Some(rt) = &fdecl.ret_ty {
                                let mut subst = BTreeMap::new();
                                for (tp, ta) in fdecl.type_params.iter().zip(type_args.iter()) {
                                    subst.insert(tp.clone(), ta.clone());
                                }
                                return subst_type_expr(rt, &subst);
                            }
                        }
                    }
                    named_ty("Any")
                }
                ExprKind::Ident(fname) => {
                    if let Some(vty) = var_types.get(fname) {
                        if let TypeExpr::Fn { ret, .. } = self.resolve_base_type(vty) {
                            return ret.map(|r| *r).unwrap_or_else(|| named_ty("Any"));
                        }
                    }
                    if (fname == "ptr_of" || fname == "mut_ptr_of") && args.len() == 1 {
                        let raw_inner = self.infer_expr_type(&args[0].value, var_types);
                        let inner = match self.resolve_base_type(&raw_inner) {
                            TypeExpr::Array { elem, .. } => *elem,
                            _ => raw_inner,
                        };
                        let pname = if fname == "mut_ptr_of" { "MutPtr" } else { "Ptr" };
                        return generic_ty(pname, vec![inner]);
                    }
                    if matches!(
                        fname.as_str(),
                        "wrapping_add"
                            | "wrapping_sub"
                            | "wrapping_mul"
                            | "saturating_add"
                            | "saturating_sub"
                    ) && !args.is_empty()
                    {
                        return self.infer_expr_type(&args[0].value, var_types);
                    }
                    if fname == "ptr_add" && args.len() == 2 {
                        return self.infer_expr_type(&args[0].value, var_types);
                    }
                    if (fname == "ptr_read" || fname == "volatile_read") && args.len() == 1 {
                        let pt = self.resolve_base_type(&self.infer_expr_type(&args[0].value, var_types));
                        if let TypeExpr::Named { name, args: pa, .. } = pt {
                            if (name == "Ptr" || name == "MutPtr") && !pa.is_empty() {
                                return pa[0].clone();
                            }
                        }
                        return named_ty("Any");
                    }
                    if fname == "mutex_new" && args.len() == 1 {
                        return generic_ty("Mutex", vec![self.infer_expr_type(&args[0].value, var_types)]);
                    }
                    if fname == "mutex_get" && args.len() == 1 {
                        let mt = self.resolve_base_type(&self.infer_expr_type(&args[0].value, var_types));
                        if let TypeExpr::Named { name, args: ma, .. } = mt {
                            if name == "Mutex" && !ma.is_empty() {
                                return ma[0].clone();
                            }
                        }
                        return named_ty("Any");
                    }
                    if (fname == "chan_recv" || fname == "chan_try_recv") && args.len() == 1 {
                        let ct = self.resolve_base_type(&self.infer_expr_type(&args[0].value, var_types));
                        if let TypeExpr::Named { name, args: ca, .. } = ct {
                            if name == "Channel" && !ca.is_empty() {
                                return generic_ty("Option", vec![ca[0].clone()]);
                            }
                        }
                        return generic_ty("Option", vec![named_ty("Any")]);
                    }
                    if let Some(ef) = self.extern_fns.get(fname) {
                        return ef.ret_ty.clone().unwrap_or_else(|| named_ty("Any"));
                    }
                    if fname == "Ok" && args.len() == 1 {
                        return generic_ty(
                            "Result",
                            vec![self.infer_expr_type(&args[0].value, var_types), named_ty("Any")],
                        );
                    }
                    if fname == "Err" && args.len() == 1 {
                        return generic_ty(
                            "Result",
                            vec![named_ty("Any"), self.infer_expr_type(&args[0].value, var_types)],
                        );
                    }
                    if fname == "Some" && args.len() == 1 {
                        return generic_ty(
                            "Option",
                            vec![self.infer_expr_type(&args[0].value, var_types)],
                        );
                    }
                    if fname == "push" && args.len() == 2 {
                        let lt = self.infer_expr_type(&args[0].value, var_types);
                        if matches!(&lt, TypeExpr::Named { name, args: la, .. } if name == "List" && la.first().is_some_and(|x| x.display_name() != "Any")) {
                            return lt;
                        }
                        return generic_ty("List", vec![self.infer_expr_type(&args[1].value, var_types)]);
                    }
                    if fname == "list_filled" && args.len() == 2 {
                        return generic_ty("List", vec![self.infer_expr_type(&args[1].value, var_types)]);
                    }
                    if fname == "list_pop_shrink" && args.len() == 1 {
                        return self.infer_expr_type(&args[0].value, var_types);
                    }
                    if fname == "block_on" && args.len() == 1 {
                        return self.infer_expr_type(&args[0].value, var_types);
                    }
                    if let Some(fdecl) = self.all_fns.get(fname) {
                        let inst = self.instantiate_method_from_args(fdecl, false, args, var_types);
                        return inst.ret_ty.unwrap_or_else(|| named_ty("Any"));
                    }
                    if let Some(vi) = self.enum_variants.get(fname) {
                        return self.infer_variant_ctor_type(vi, args, var_types);
                    }
                    if let Some(b) = lookup_builtin(fname) {
                        return if b.ret_type == "List[Any]" {
                            generic_ty("List", vec![named_ty("Any")])
                        } else if b.ret_type == "Option[Any]" {
                            generic_ty("Option", vec![named_ty("Any")])
                        } else if b.ret_type == "Result[Any, Any]" {
                            generic_ty("Result", vec![named_ty("Any"), named_ty("Any")])
                        } else if b.ret_type == "Ptr[UInt8]" {
                            generic_ty("Ptr", vec![named_ty("UInt8")])
                        } else if b.ret_type == "MutPtr[UInt8]" {
                            generic_ty("MutPtr", vec![named_ty("UInt8")])
                        } else if b.ret_type == "Ptr[Any]" {
                            generic_ty("Ptr", vec![named_ty("Any")])
                        } else if b.ret_type == "MutPtr[Any]" {
                            generic_ty("MutPtr", vec![named_ty("Any")])
                        } else if b.ret_type == "Mutex[Any]" {
                            generic_ty("Mutex", vec![named_ty("Any")])
                        } else if b.ret_type == "Channel[Any]" {
                            generic_ty("Channel", vec![named_ty("Any")])
                        } else {
                            named_ty(b.ret_type)
                        };
                    }
                    named_ty("Any")
                }
                _ => {
                    let cty = self.resolve_base_type(&self.infer_expr_type(callee, var_types));
                    if let TypeExpr::Fn { ret, .. } = cty {
                        ret.map(|r| *r).unwrap_or_else(|| named_ty("Any"))
                    } else {
                        named_ty("Any")
                    }
                }
            },
            ExprKind::If { then_block, else_branch, .. } => {
                if let Some(t) = &then_block.tail {
                    let ty = self.infer_expr_type_expected(t, expected, var_types);
                    if ty.display_name() != "Any" {
                        return ty;
                    }
                }
                if let Some(eb) = else_branch {
                    return self.infer_expr_type_expected(eb, expected, var_types);
                }
                named_ty("Any")
            }
            ExprKind::Match { scrutinee, arms } => {
                let st = self.infer_expr_type(scrutinee, var_types);
                for arm in arms {
                    let mut arm_vars = var_types.clone();
                    self.bind_match_arm_types(&st, arm, &mut arm_vars);
                    let aty = self.infer_expr_type_expected(&arm.body, expected, &arm_vars);
                    if aty.display_name() != "Any" {
                        return aty;
                    }
                }
                named_ty("Any")
            }
            ExprKind::Block(b) => {
                if let Some(t) = &b.tail {
                    self.infer_expr_type_expected(t, expected, var_types)
                } else {
                    named_ty("Any")
                }
            }
            ExprKind::Await(inner) => self.infer_expr_type(inner, var_types),
            _ => named_ty("Any"),
        }
    }

    fn infer_variant_ctor_type(
        &self,
        vi: &VariantInfo,
        args: &[CallArg],
        var_types: &BTreeMap<String, TypeExpr>,
    ) -> TypeExpr {
        if let Some(ed) = self.enums.get(&vi.enum_name) {
            if !ed.type_params.is_empty() {
                let mut subst = BTreeMap::new();
                for (pty, a) in vi.payload_tys.iter().zip(args.iter()) {
                    let aty = self.infer_expr_type(&a.value, var_types);
                    Self::unify_type_pattern(pty, &aty, &ed.type_params, &mut subst);
                }
                let targs: Vec<TypeExpr> = ed
                    .type_params
                    .iter()
                    .map(|tp| subst.get(tp).cloned().unwrap_or_else(|| named_ty("Any")))
                    .collect();
                return generic_ty(&vi.enum_name, targs);
            }
        }
        named_ty(&vi.enum_name)
    }

    fn resolve_variant_payload_tys(&self, scrutinee_base: &TypeExpr, variant_name: &str) -> Vec<TypeExpr> {
        let short = variant_name.rsplit('.').next().unwrap_or(variant_name);
        if short == "Some" {
            if let TypeExpr::Named { name: tn, args, .. } = scrutinee_base {
                if tn == "Option" && !args.is_empty() {
                    return vec![args[0].clone()];
                }
            }
            return vec![named_ty("Any")];
        }
        if short == "Ok" {
            if let TypeExpr::Named { name: tn, args, .. } = scrutinee_base {
                if tn == "Result" && !args.is_empty() {
                    return vec![args[0].clone()];
                }
            }
            return vec![named_ty("Any")];
        }
        if short == "Err" {
            if let TypeExpr::Named { name: tn, args, .. } = scrutinee_base {
                if tn == "Result" && args.len() >= 2 {
                    return vec![args[1].clone()];
                }
            }
            return vec![named_ty("Any")];
        }
        if let Some(vi) = self.enum_variants.get(variant_name).or_else(|| self.enum_variants.get(short)) {
            if let Some(ed) = self.enums.get(&vi.enum_name) {
                if !ed.type_params.is_empty() {
                    if let TypeExpr::Named { args, .. } = scrutinee_base {
                        if !args.is_empty() {
                            let mut subst = BTreeMap::new();
                            for (tp, ta) in ed.type_params.iter().zip(args.iter()) {
                                subst.insert(tp.clone(), ta.clone());
                            }
                            return vi
                                .payload_tys
                                .iter()
                                .map(|pty| subst_type_expr(pty, &subst))
                                .collect();
                        }
                    }
                }
            }
            return vi.payload_tys.clone();
        }
        Vec::new()
    }

    fn bind_pattern_types(
        &self,
        scrutinee_ty: &TypeExpr,
        pat: &Pattern,
        arm_vars: &mut BTreeMap<String, TypeExpr>,
    ) {
        let base = self.resolve_base_type(scrutinee_ty);
        match pat {
            Pattern::Ident(name, _) => {
                if name != "_" {
                    arm_vars.insert(name.clone(), scrutinee_ty.clone());
                }
            }
            Pattern::Tuple(pats, _) => {
                if let TypeExpr::Tuple(elems, _) = &base {
                    for (i, sp) in pats.iter().enumerate() {
                        let ety = elems.get(i).cloned().unwrap_or_else(|| named_ty("Any"));
                        self.bind_pattern_types(&ety, sp, arm_vars);
                    }
                } else {
                    for sp in pats {
                        self.bind_pattern_types(&named_ty("Any"), sp, arm_vars);
                    }
                }
            }
            Pattern::Variant { name, binding: Some(bname), .. } => {
                if bname != "_" {
                    let ptys = self.resolve_variant_payload_tys(&base, name);
                    let ty = ptys.into_iter().next().unwrap_or_else(|| named_ty("Any"));
                    arm_vars.insert(bname.clone(), ty);
                }
            }
            Pattern::VariantMulti { name, patterns, .. } => {
                let ptys = self.resolve_variant_payload_tys(&base, name);
                for (i, sp) in patterns.iter().enumerate() {
                    let pty = ptys.get(i).cloned().unwrap_or_else(|| named_ty("Any"));
                    self.bind_pattern_types(&pty, sp, arm_vars);
                }
            }
            _ => {}
        }
    }

    fn bind_match_arm_types(
        &self,
        scrutinee_ty: &TypeExpr,
        arm: &MatchArm,
        arm_vars: &mut BTreeMap<String, TypeExpr>,
    ) {
        for p in &arm.patterns {
            self.bind_pattern_types(scrutinee_ty, p, arm_vars);
        }
    }

    fn check_fn(&mut self, f: &FnDecl) {
        let prev_tp = self.cur_type_params.clone();
        let prev_bounds = self.cur_type_param_bounds.clone();
        let prev_unsafe = self.in_unsafe;
        let prev_file_id = self.cur_file_id;
        self.in_unsafe = f.is_unsafe;
        if f.span.file_id != 0 {
            self.cur_file_id = f.span.file_id;
        }
        for tp in &f.type_params {
            self.cur_type_params.insert(tp.clone());
        }
        for (k, v) in &f.type_param_bounds {
            self.cur_type_param_bounds.insert(k.clone(), v.clone());
        }

        let mut env = L0Env {
            pure_fns: self.pure_fns.clone(),
            type_aliases: self.type_aliases.clone(),
            ..Default::default()
        };
        let mut smt_premises: Vec<(Expr, BTreeMap<String, String>)> = Vec::new();
        let mut smt_int_vars: BTreeSet<String> = BTreeSet::new();
        let mut var_types: BTreeMap<String, TypeExpr> = BTreeMap::new();
        let mut mutable_vars: BTreeSet<String> = BTreeSet::new();
        for (cname, (cdecl, iv, _)) in &self.consts {
            var_types.insert(cname.clone(), cdecl.ty.clone());
            if let Some(n) = iv {
                env.set_int(cname, AbsInt::exact(*n as i128));
                smt_int_vars.insert(cname.clone());
            }
        }

        // 1. Register parameters, mutability, dyn object-safety, refinements & record invariants
        for p in &f.params {
            self.check_dyn_type_valid(&p.ty, p.span);
            var_types.insert(p.name.clone(), p.ty.clone());
            if p.is_inout {
                mutable_vars.insert(p.name.clone());
            }
            self.assume_type_facts(
                &p.name,
                &p.ty,
                &mut env,
                &mut smt_int_vars,
                &mut smt_premises,
            );
            if let Some(pred) = &p.predicate {
                env.assume_expr(pred, Some(&p.name));
                let mut subst = BTreeMap::new();
                subst.insert("self".into(), p.name.clone());
                smt_premises.push((pred.clone(), subst));
            }
        }
        if let Some(rt) = &f.ret_ty {
            self.check_dyn_type_valid(rt, f.span);
        }

        // 2. Assume function `requires`
        for req in &f.requires {
            env.assume_expr(req, None);
            smt_premises.push((req.clone(), BTreeMap::new()));
        }

        // 3. Static Effect, Purity & Memory Contract Checks (`async`, `pure`, `@no_heap_alloc`)
        let no_heap = f.attrs.iter().any(|a| a.name == "no_heap_alloc");
        self.check_block_effects(&f.body, f.is_async, f.is_pure, no_heap, &f.name, false);

        // 4. Walk statements & tail expression checking types, exhaustiveness, requires, div-by-zero, mutate & early returns
        let mut active_mutate: Option<String> = None;
        let mut saw_return = false;
        let tail_expr = self.check_block(
            &f.body,
            f,
            &mut env,
            &mut smt_int_vars,
            &mut smt_premises,
            &mut var_types,
            &mut mutable_vars,
            &mut active_mutate,
            &mut saw_return,
        );

        // 5. Verify return type & refinement on final tail expression (early `Stmt::Return` paths were already checked!)
        if let Some(ret_expr) = &tail_expr {
            if let Some(ret_ty) = &f.ret_ty {
                let actual_ty = self.infer_expr_type_expected(ret_expr, Some(ret_ty), &var_types);
                if !self.types_compatible(ret_ty, &actual_ty) {
                    self.diagnostics.push(Diagnostic {
                        code: "E0205".into(),
                        message: format!(
                            "return type mismatch in `{}`: expected `{}`, found `{}`",
                            f.name,
                            ret_ty.display_name(),
                            actual_ty.display_name()
                        ),
                        span: ret_expr.span,
                        notes: vec![],
                        counterexample: BTreeMap::new(),
                    });
                }
                self.check_expr_satisfies_type(
                    ret_expr,
                    ret_ty,
                    &env,
                    &smt_int_vars,
                    &smt_premises,
                    &format!("return value of `{}`", f.name),
                    ret_expr.span,
                );
            }
            for ens in &f.ensures {
                self.check_ensures(f, ens, Some(ret_expr), &env, &smt_int_vars, &smt_premises);
            }
        } else if !saw_return {
            for ens in &f.ensures {
                self.check_ensures(f, ens, None, &env, &smt_int_vars, &smt_premises);
            }
        }
        self.cur_type_params = prev_tp;
        self.cur_type_param_bounds = prev_bounds;
        self.in_unsafe = prev_unsafe;
        self.cur_file_id = prev_file_id;
    }

    fn check_block_effects(
        &mut self,
        blk: &Block,
        in_async: bool,
        in_pure: bool,
        no_heap: bool,
        fn_name: &str,
        inside_block_on: bool,
    ) {
        for s in &blk.stmts {
            match s {
                Stmt::Let { value, .. } | Stmt::LetTuple { value, .. } => {
                    self.check_expr_effects(value, in_async, in_pure, no_heap, fn_name, inside_block_on);
                }
                Stmt::Assign { value, .. } => {
                    self.check_expr_effects(value, in_async, in_pure, no_heap, fn_name, inside_block_on);
                }
                Stmt::Return { value: Some(e), .. } | Stmt::Expr { expr: e, .. } => {
                    self.check_expr_effects(e, in_async, in_pure, no_heap, fn_name, inside_block_on);
                }
                Stmt::Assert { cond, .. } => {
                    self.check_expr_effects(cond, in_async, in_pure, no_heap, fn_name, inside_block_on);
                }
                Stmt::Mutate { body, .. } | Stmt::Defer { body, .. } => {
                    self.check_block_effects(body, in_async, in_pure, no_heap, fn_name, inside_block_on);
                }
                Stmt::While { cond, body, .. } => {
                    self.check_expr_effects(cond, in_async, in_pure, no_heap, fn_name, inside_block_on);
                    self.check_block_effects(body, in_async, in_pure, no_heap, fn_name, inside_block_on);
                }
                Stmt::For { iter, body, .. } => {
                    self.check_expr_effects(iter, in_async, in_pure, no_heap, fn_name, inside_block_on);
                    self.check_block_effects(body, in_async, in_pure, no_heap, fn_name, inside_block_on);
                }
                _ => {}
            }
        }
        if let Some(t) = &blk.tail {
            self.check_expr_effects(t, in_async, in_pure, no_heap, fn_name, inside_block_on);
        }
    }

    fn check_expr_effects(
        &mut self,
        expr: &Expr,
        in_async: bool,
        in_pure: bool,
        no_heap: bool,
        fn_name: &str,
        inside_block_on: bool,
    ) {
        match &expr.kind {
            ExprKind::Await(inner) => {
                if in_pure {
                    self.diagnostics.push(Diagnostic {
                        code: "E0602".into(),
                        message: format!("pure function `{}` cannot use `await`", fn_name),
                        span: expr.span,
                        notes: vec!["remove `pure` or move suspension to an `async fn`".into()],
                        counterexample: BTreeMap::new(),
                    });
                } else if !in_async {
                    self.diagnostics.push(Diagnostic {
                        code: "E0601".into(),
                        message: format!(
                            "async effect violation: `await` used inside synchronous function `{}`",
                            fn_name
                        ),
                        span: expr.span,
                        notes: vec![
                            "mark the enclosing function `async fn` or drive futures via `block_on(...)`".into(),
                        ],
                        counterexample: BTreeMap::new(),
                    });
                }
                self.check_expr_effects(inner, in_async, in_pure, no_heap, fn_name, true);
            }
            ExprKind::RecordLit { name, fields, .. } => {
                if no_heap {
                    self.diagnostics.push(Diagnostic {
                        code: "E0701".into(),
                        message: format!(
                            "`@no_heap_alloc` contract violated in `{}`: record allocation `{}`",
                            fn_name, name
                        ),
                        span: expr.span,
                        notes: vec!["functions annotated `@no_heap_alloc` must stay strictly stack/register-only".into()],
                        counterexample: BTreeMap::new(),
                    });
                }
                for (_, v) in fields {
                    self.check_expr_effects(v, in_async, in_pure, no_heap, fn_name, inside_block_on);
                }
            }
            ExprKind::List(elems) => {
                if no_heap {
                    self.diagnostics.push(Diagnostic {
                        code: "E0701".into(),
                        message: format!(
                            "`@no_heap_alloc` contract violated in `{}`: heap list allocation",
                            fn_name
                        ),
                        span: expr.span,
                        notes: vec!["avoid dynamic `List` literals inside `@no_heap_alloc` functions".into()],
                        counterexample: BTreeMap::new(),
                    });
                }
                for e in elems {
                    self.check_expr_effects(e, in_async, in_pure, no_heap, fn_name, inside_block_on);
                }
            }
            ExprKind::Str(_) => {
                if no_heap {
                    self.diagnostics.push(Diagnostic {
                        code: "E0701".into(),
                        message: format!(
                            "`@no_heap_alloc` contract violated in `{}`: heap Text allocation",
                            fn_name
                        ),
                        span: expr.span,
                        notes: vec!["avoid dynamic `Text` allocations inside `@no_heap_alloc` functions".into()],
                        counterexample: BTreeMap::new(),
                    });
                }
            }
            ExprKind::Call { callee, args } => {
                let is_block_on = matches!(&callee.kind, ExprKind::Ident(s) if s == "block_on");
                if let ExprKind::Ident(fname) = &callee.kind {
                    if self.async_fns.contains(fname) && !in_async && !inside_block_on {
                        self.diagnostics.push(Diagnostic {
                            code: "E0601".into(),
                            message: format!(
                                "async effect violation: `{}` is `async` and cannot be called directly from sync fn `{}`",
                                fname, fn_name
                            ),
                            span: expr.span,
                            notes: vec![format!("use `await {}(...)` inside an `async fn` or `block_on({}(...))`", fname, fname)],
                            counterexample: BTreeMap::new(),
                        });
                    }
                    // Purity enforcement: `pure fn` cannot call impure user functions or impure OS syscalls!
                    if in_pure {
                        if let Some(b) = lookup_builtin(fname) {
                            if !b.is_pure {
                                self.diagnostics.push(Diagnostic {
                                    code: "E0610".into(),
                                    message: format!(
                                        "purity violation: `pure fn {}` cannot call impure builtin/syscall `{}`",
                                        fn_name, fname
                                    ),
                                    span: expr.span,
                                    notes: vec!["remove `pure` or move I/O effects out of pure functions".into()],
                                    counterexample: BTreeMap::new(),
                                });
                            }
                        } else if let Some(target_fn) = self.all_fns.get(fname) {
                            if !target_fn.is_pure {
                                self.diagnostics.push(Diagnostic {
                                    code: "E0610".into(),
                                    message: format!(
                                        "purity violation: `pure fn {}` cannot call non-pure function `{}`",
                                        fn_name, fname
                                    ),
                                    span: expr.span,
                                    notes: vec![format!("mark `{}` as `pure fn` or remove `pure` from `{}`", fname, fn_name)],
                                    counterexample: BTreeMap::new(),
                                });
                            }
                        }
                    }
                    // `@no_heap_alloc` enforcement on calls!
                    if no_heap {
                        if let Some(b) = lookup_builtin(fname) {
                            if b.allocates {
                                self.diagnostics.push(Diagnostic {
                                    code: "E0701".into(),
                                    message: format!(
                                        "`@no_heap_alloc` contract violated in `{}`: `{}(...)` allocates on the heap",
                                        fn_name, fname
                                    ),
                                    span: expr.span,
                                    notes: vec![],
                                    counterexample: BTreeMap::new(),
                                });
                            }
                        } else if self.enum_variants.contains_key(fname) && !self.all_fns.contains_key(fname) {
                            self.diagnostics.push(Diagnostic {
                                code: "E0701".into(),
                                message: format!(
                                    "`@no_heap_alloc` contract violated in `{}`: enum allocation `{}(...)`",
                                    fn_name, fname
                                ),
                                span: expr.span,
                                notes: vec![],
                                counterexample: BTreeMap::new(),
                            });
                        } else if self.all_fns.contains_key(fname) && !self.no_heap_fns.contains(fname) {
                            self.diagnostics.push(Diagnostic {
                                code: "E0701".into(),
                                message: format!(
                                    "`@no_heap_alloc` contract violated in `{}`: callee `{}` is not annotated `@no_heap_alloc`",
                                    fn_name, fname
                                ),
                                span: expr.span,
                                notes: vec![format!("annotate `{}` with `@no_heap_alloc` to call it from a no-alloc hot path", fname)],
                                counterexample: BTreeMap::new(),
                            });
                        }
                    }
                }
                if let ExprKind::Field { field, .. } = &callee.kind {
                    if no_heap && field == "parse" {
                        self.diagnostics.push(Diagnostic {
                            code: "E0701".into(),
                            message: format!(
                                "`@no_heap_alloc` contract violated in `{}`: `.parse(...)` allocates a Result/Record",
                                fn_name
                            ),
                            span: expr.span,
                            notes: vec![],
                            counterexample: BTreeMap::new(),
                        });
                    }
                }
                for a in args {
                    self.check_expr_effects(
                        &a.value,
                        in_async,
                        in_pure,
                        no_heap,
                        fn_name,
                        inside_block_on || is_block_on,
                    );
                }
            }
            ExprKind::Binary { op, left, right } => {
                if no_heap && matches!(op, BinOp::Concat) {
                    self.diagnostics.push(Diagnostic {
                        code: "E0701".into(),
                        message: format!(
                            "`@no_heap_alloc` contract violated in `{}`: string concatenation `++` allocates on the heap",
                            fn_name
                        ),
                        span: expr.span,
                        notes: vec![],
                        counterexample: BTreeMap::new(),
                    });
                }
                self.check_expr_effects(left, in_async, in_pure, no_heap, fn_name, inside_block_on);
                self.check_expr_effects(right, in_async, in_pure, no_heap, fn_name, inside_block_on);
            }
            ExprKind::Unary { expr: inner, .. } => {
                self.check_expr_effects(inner, in_async, in_pure, no_heap, fn_name, inside_block_on);
            }
            ExprKind::If {
                cond,
                then_block,
                else_branch,
            } => {
                self.check_expr_effects(cond, in_async, in_pure, no_heap, fn_name, inside_block_on);
                self.check_block_effects(then_block, in_async, in_pure, no_heap, fn_name, inside_block_on);
                if let Some(eb) = else_branch {
                    self.check_expr_effects(eb, in_async, in_pure, no_heap, fn_name, inside_block_on);
                }
            }
            ExprKind::Match { scrutinee, arms } => {
                self.check_expr_effects(scrutinee, in_async, in_pure, no_heap, fn_name, inside_block_on);
                for a in arms {
                    self.check_expr_effects(&a.body, in_async, in_pure, no_heap, fn_name, inside_block_on);
                }
            }
            ExprKind::Block(b) => {
                self.check_block_effects(b, in_async, in_pure, no_heap, fn_name, inside_block_on);
            }
            ExprKind::Tuple(elems) => {
                if no_heap {
                    self.diagnostics.push(Diagnostic {
                        code: "E0701".into(),
                        message: format!(
                            "`@no_heap_alloc` contract violated in `{}`: heap tuple allocation",
                            fn_name
                        ),
                        span: expr.span,
                        notes: vec!["avoid tuple allocations inside `@no_heap_alloc` functions".into()],
                        counterexample: BTreeMap::new(),
                    });
                }
                for e in elems {
                    self.check_expr_effects(e, in_async, in_pure, no_heap, fn_name, inside_block_on);
                }
            }
            ExprKind::TupleIndex { target, .. } | ExprKind::Try(target) => {
                self.check_expr_effects(target, in_async, in_pure, no_heap, fn_name, inside_block_on);
            }
            ExprKind::Range { start, end } => {
                self.check_expr_effects(start, in_async, in_pure, no_heap, fn_name, inside_block_on);
                self.check_expr_effects(end, in_async, in_pure, no_heap, fn_name, inside_block_on);
            }
            ExprKind::Lambda { body, .. } => {
                if no_heap {
                    self.diagnostics.push(Diagnostic {
                        code: "E0701".into(),
                        message: format!(
                            "`@no_heap_alloc` contract violated in `{}`: closure allocation",
                            fn_name
                        ),
                        span: expr.span,
                        notes: vec!["avoid closures inside `@no_heap_alloc` functions".into()],
                        counterexample: BTreeMap::new(),
                    });
                }
                self.check_expr_effects(body, false, in_pure, no_heap, fn_name, false);
            }
            _ => {}
        }
    }

    fn assume_type_facts(
        &self,
        var_name: &str,
        ty: &TypeExpr,
        env: &mut L0Env,
        smt_int_vars: &mut BTreeSet<String>,
        smt_premises: &mut Vec<(Expr, BTreeMap<String, String>)>,
    ) {
        if let TypeExpr::Named { name, .. } = ty {
            if let Some(sk) = ScalarKind::from_name(name) {
                if sk.is_int() {
                    smt_int_vars.insert(var_name.to_string());
                    if let (Some(lo), Some(hi)) = (sk.min_i128(), sk.max_i128()) {
                        if sk != ScalarKind::I64 {
                            env.set_int(
                                var_name,
                                AbsInt::top().refine_interval(Interval { min: lo, max: hi }),
                            );
                        }
                    }
                }
            } else if let Some(alias) = self.type_aliases.get(name).cloned() {
                self.assume_type_facts(var_name, &alias.base, env, smt_int_vars, smt_premises);
                if let Some(pred) = &alias.predicate {
                    env.assume_expr(pred, Some(var_name));
                    let mut subst = BTreeMap::new();
                    subst.insert("self".into(), var_name.to_string());
                    smt_premises.push((pred.clone(), subst));
                }
            } else if let Some(rec) = self.records.get(name).cloned() {
                let mut field_subst = BTreeMap::new();
                field_subst.insert("self".into(), var_name.to_string());
                for field in &rec.fields {
                    let fvar = format!("{}.{}", var_name, field.name);
                    field_subst.insert(field.name.clone(), fvar.clone());
                    self.assume_type_facts(&fvar, &field.ty, env, smt_int_vars, smt_premises);
                    if let Some(fpred) = &field.predicate {
                        let rewritten = substitute_field_idents(fpred, &rec.fields, var_name);
                        env.assume_expr(&rewritten, Some(&fvar));
                        let mut s = field_subst.clone();
                        s.insert("self".into(), fvar.clone());
                        smt_premises.push((fpred.clone(), s));
                    }
                }
                for wc in &rec.where_clauses {
                    let rewritten = substitute_field_idents(wc, &rec.fields, var_name);
                    env.assume_expr(&rewritten, Some(var_name));
                    smt_premises.push((wc.clone(), field_subst.clone()));
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn check_block(
        &mut self,
        blk: &Block,
        cur_fn: &FnDecl,
        env: &mut L0Env,
        smt_int_vars: &mut BTreeSet<String>,
        smt_premises: &mut Vec<(Expr, BTreeMap<String, String>)>,
        var_types: &mut BTreeMap<String, TypeExpr>,
        mutable_vars: &mut BTreeSet<String>,
        active_mutate: &mut Option<String>,
        saw_return: &mut bool,
    ) -> Option<Expr> {
        for stmt in &blk.stmts {
            match stmt {
                Stmt::Let {
                    name,
                    ty,
                    value,
                    is_mut,
                    span,
                } => {
                    if let ExprKind::Lambda { params, ret_ty, body } = &value.kind {
                        self.check_lambda(
                            params,
                            ret_ty.as_ref(),
                            body,
                            ty.as_ref(),
                            var_types,
                            value.span,
                        );
                    } else {
                        self.check_expr(
                            value,
                            cur_fn,
                            env,
                            smt_int_vars,
                            smt_premises,
                            var_types,
                            mutable_vars,
                            active_mutate,
                            saw_return,
                        );
                    }
                    let inferred = self.infer_expr_type_expected(value, ty.as_ref(), var_types);
                    let effective_ty = if let Some(t) = ty {
                        self.check_dyn_type_valid(t, *span);
                        self.check_type_bounds(t, *span);
                        if let TypeExpr::Dyn { interface, .. } = t {
                            if !self.type_implements(&inferred, interface) {
                                self.diagnostics.push(Diagnostic {
                                    code: "E0505".into(),
                                    message: format!(
                                        "type `{}` does not implement interface `{}` required by `{}`",
                                        inferred.display_name(),
                                        interface,
                                        t.display_name()
                                    ),
                                    span: *span,
                                    notes: vec![format!("add `impl {} for {} {{ ... }}`", interface, inferred.display_name())],
                                    counterexample: BTreeMap::new(),
                                });
                            } else if !matches!(inferred, TypeExpr::Dyn { .. })
                                && cur_fn.attrs.iter().any(|a| a.name == "no_heap_alloc")
                            {
                                self.diagnostics.push(Diagnostic {
                                    code: "E0701".into(),
                                    message: format!(
                                        "`@no_heap_alloc` contract violated in `{}`: boxing `{}` into `dyn {}` allocates",
                                        cur_fn.name,
                                        inferred.display_name(),
                                        interface
                                    ),
                                    span: *span,
                                    notes: vec!["use static generic parameter `[T Interface]` on `@no_heap_alloc` paths".into()],
                                    counterexample: BTreeMap::new(),
                                });
                            }
                        } else if !self.types_compatible(t, &inferred) {
                            self.diagnostics.push(Diagnostic {
                                code: "E0205".into(),
                                message: format!(
                                    "type mismatch in binding `{}`: expected `{}`, found `{}`",
                                    name,
                                    t.display_name(),
                                    inferred.display_name()
                                ),
                                span: *span,
                                notes: vec![],
                                counterexample: BTreeMap::new(),
                            });
                        }
                        self.check_expr_satisfies_type(
                            value,
                            t,
                            env,
                            smt_int_vars,
                            smt_premises,
                            &format!("binding `{}`", name),
                            *span,
                        );
                        t.clone()
                    } else {
                        inferred
                    };

                    var_types.insert(name.clone(), effective_ty.clone());
                    if *is_mut {
                        mutable_vars.insert(name.clone());
                    } else {
                        mutable_vars.remove(name);
                    }
                    self.assume_type_facts(name, &effective_ty, env, smt_int_vars, smt_premises);

                    // Propagate concrete RecordLit fields or copied Ident record fields into `env` & `smt_premises`!
                    match &value.kind {
                        ExprKind::RecordLit { fields, .. } => {
                            for (fname, fexpr) in fields {
                                let fkey = format!("{}.{}", name, fname);
                                let fabs = env.eval_abs_int(fexpr, None);
                                env.set_int(fkey.clone(), fabs);
                                smt_int_vars.insert(fkey.clone());
                                let eq = Expr {
                                    kind: ExprKind::Binary {
                                        op: BinOp::Eq,
                                        left: Box::new(Expr {
                                            kind: ExprKind::Ident(fkey),
                                            span: fexpr.span,
                                        }),
                                        right: Box::new(fexpr.clone()),
                                    },
                                    span: fexpr.span,
                                };
                                smt_premises.push((eq, BTreeMap::new()));
                            }
                        }
                        ExprKind::Ident(src) => {
                            if let TypeExpr::Named { name: rname, .. } = self.resolve_base_type(&effective_ty) {
                                if let Some(rec) = self.records.get(&rname).cloned() {
                                    for f in &rec.fields {
                                        let src_k = format!("{}.{}", src, f.name);
                                        let dst_k = format!("{}.{}", name, f.name);
                                        let fabs = env.get_int(&src_k);
                                        env.set_int(dst_k.clone(), fabs);
                                        smt_int_vars.insert(dst_k.clone());
                                        let eq = Expr {
                                            kind: ExprKind::Binary {
                                                op: BinOp::Eq,
                                                left: Box::new(Expr {
                                                    kind: ExprKind::Ident(dst_k),
                                                    span: *span,
                                                }),
                                                right: Box::new(Expr {
                                                    kind: ExprKind::Ident(src_k),
                                                    span: *span,
                                                }),
                                            },
                                            span: *span,
                                        };
                                        smt_premises.push((eq, BTreeMap::new()));
                                    }
                                }
                            }
                        }
                        _ => {}
                    }

                    let abs = env.eval_abs_int(value, None);
                    // Preserve refined type facts when `eval_abs_int` returns Top!
                    let existing = env.get_int(name);
                    let refined = AbsInt {
                        interval: existing.interval.intersect(abs.interval),
                        cong: existing.cong.meet(abs.cong),
                        non_zero: existing.non_zero || abs.non_zero,
                    };
                    env.set_int(name.clone(), refined);
                    if self.is_int_type_name(&effective_ty.display_name()) {
                        smt_int_vars.insert(name.clone());
                        if self.expr_smt_encodable(value) {
                            let eq = Expr {
                                kind: ExprKind::Binary {
                                    op: BinOp::Eq,
                                    left: Box::new(Expr {
                                        kind: ExprKind::Ident(name.clone()),
                                        span: *span,
                                    }),
                                    right: Box::new(value.clone()),
                                },
                                span: *span,
                            };
                            smt_premises.push((eq, BTreeMap::new()));
                        } else {
                            // Bridge sound L0 domain bounds (e.g. `len(x) >= 0`, `byte_at in [-1, 255]`,
                            // callee `ensures` bounds) into L1 SMT premises.
                            if refined.interval.min > i64::MIN as i128
                                && refined.interval.min <= i64::MAX as i128
                            {
                                let ge = Expr {
                                    kind: ExprKind::Binary {
                                        op: BinOp::Ge,
                                        left: Box::new(Expr {
                                            kind: ExprKind::Ident(name.clone()),
                                            span: *span,
                                        }),
                                        right: Box::new(Expr {
                                            kind: ExprKind::Int(refined.interval.min as i64),
                                            span: *span,
                                        }),
                                    },
                                    span: *span,
                                };
                                smt_premises.push((ge, BTreeMap::new()));
                            }
                            if refined.interval.max < i64::MAX as i128
                                && refined.interval.max >= i64::MIN as i128
                            {
                                let le = Expr {
                                    kind: ExprKind::Binary {
                                        op: BinOp::Le,
                                        left: Box::new(Expr {
                                            kind: ExprKind::Ident(name.clone()),
                                            span: *span,
                                        }),
                                        right: Box::new(Expr {
                                            kind: ExprKind::Int(refined.interval.max as i64),
                                            span: *span,
                                        }),
                                    },
                                    span: *span,
                                };
                                smt_premises.push((le, BTreeMap::new()));
                            }
                        }
                    }
                }
                Stmt::Assert { cond, .. } => {
                    self.check_expr(
                        cond,
                        cur_fn,
                        env,
                        smt_int_vars,
                        smt_premises,
                        var_types,
                        mutable_vars,
                        active_mutate,
                        saw_return,
                    );
                    env.assume_expr(cond, None);
                    smt_premises.push((cond.clone(), BTreeMap::new()));
                }
                Stmt::Assign {
                    target,
                    op,
                    value,
                    span,
                } => {
                    self.check_expr(
                        value,
                        cur_fn,
                        env,
                        smt_int_vars,
                        smt_premises,
                        var_types,
                        mutable_vars,
                        active_mutate,
                        saw_return,
                    );
                    match &target.kind {
                        ExprKind::Ident(vname) => {
                            if !mutable_vars.contains(vname) && var_types.contains_key(vname) {
                                self.diagnostics.push(Diagnostic {
                                    code: "E0203".into(),
                                    message: format!(
                                        "cannot assign to immutable binding `{}`",
                                        vname
                                    ),
                                    span: *span,
                                    notes: vec![format!("declare `{}` with `var` (or `inout` parameter) to mutate it", vname)],
                                    counterexample: BTreeMap::new(),
                                });
                            }
                            if let Some(expected_ty) = var_types.get(vname).cloned() {
                                let rhs_ty = self.infer_expr_type_expected(
                                    value,
                                    Some(&expected_ty),
                                    var_types,
                                );
                                if !self.types_compatible(&expected_ty, &rhs_ty) {
                                    self.diagnostics.push(Diagnostic {
                                        code: "E0205".into(),
                                        message: format!(
                                            "type mismatch assigning to `{}`: expected `{}`, found `{}`",
                                            vname,
                                            expected_ty.display_name(),
                                            rhs_ty.display_name()
                                        ),
                                        span: *span,
                                        notes: vec![],
                                        counterexample: BTreeMap::new(),
                                    });
                                }
                            }
                            let new_expr = match assign_op_to_binop(*op) {
                                None => value.clone(),
                                Some(bop) => {
                                    if matches!(bop, BinOp::Div) {
                                        let lhs_is_float = var_types
                                            .get(vname)
                                            .is_some_and(|t| self.is_float_ty(t));
                                        if !lhs_is_float {
                                            let denom = env.eval_abs_int(value, None);
                                            if !denom.non_zero {
                                                self.diagnostics.push(Diagnostic {
                                                    code: "E0430".into(),
                                                    message: "potential division by zero in `/=`: denominator is not proved NonZero".into(),
                                                    span: *span,
                                                    notes: vec!["guard denominator with `if denom != 0`".into()],
                                                    counterexample: BTreeMap::new(),
                                                });
                                            }
                                        }
                                    }
                                    Expr {
                                        kind: ExprKind::Binary {
                                            op: bop,
                                            left: Box::new(target.clone()),
                                            right: Box::new(value.clone()),
                                        },
                                        span: *span,
                                    }
                                }
                            };
                            let new_abs = env.eval_abs_int(&new_expr, None);
                            env.set_int(vname.clone(), new_abs);
                            if let Some(expected_ty) = var_types.get(vname).cloned() {
                                self.check_expr_satisfies_type(
                                    &new_expr,
                                    &expected_ty,
                                    env,
                                    smt_int_vars,
                                    smt_premises,
                                    &format!("assignment to `{}`", vname),
                                    *span,
                                );
                            }
                            // Remove stale equality premises referencing `vname`
                            smt_premises.retain(|(p, _)| !expr_mentions_ident(p, vname));
                        }
                        ExprKind::Field {
                            target: obj,
                            field,
                        } => {
                            if let ExprKind::Ident(var) = &obj.kind {
                                if !mutable_vars.contains(var) && var_types.contains_key(var) {
                                    self.diagnostics.push(Diagnostic {
                                        code: "E0203".into(),
                                        message: format!(
                                            "cannot mutate field `{}.{}` on immutable binding `{}`",
                                            var, field, var
                                        ),
                                        span: *span,
                                        notes: vec![format!("declare `{}` with `var` or `inout`", var)],
                                        counterexample: BTreeMap::new(),
                                    });
                                }
                                let is_inside_mutate = active_mutate.as_deref() == Some(var.as_str());
                                if let Some(ty) = var_types.get(var).cloned() {
                                    let rec_name = type_head_name(&self.resolve_base_type(&ty));
                                    if let Some(rec) = self.records.get(&rec_name).cloned() {
                                        let has_cross_inv = !rec.where_clauses.is_empty()
                                            || rec.fields.iter().any(|f| {
                                                f.predicate.is_some()
                                                    || self
                                                        .type_aliases
                                                        .get(&f.ty.display_name())
                                                        .is_some_and(|a| a.predicate.is_some())
                                            });
                                        if has_cross_inv && !is_inside_mutate {
                                            self.diagnostics.push(Diagnostic {
                                                code: "E0420".into(),
                                                message: format!(
                                                    "direct field mutation `{}.{}` on invariant record `{}` must be wrapped in `mutate {} {{ ... }}`",
                                                    var, field, rec_name, var
                                                ),
                                                span: *span,
                                                notes: vec![
                                                    "mutating one field directly could expose an intermediate invalid invariant state".into(),
                                                ],
                                                counterexample: BTreeMap::new(),
                                            });
                                        }
                                        let fkey = format!("{}.{}", var, field);
                                        let new_val_expr = match assign_op_to_binop(*op) {
                                            None => value.clone(),
                                            Some(bop) => Expr {
                                                kind: ExprKind::Binary {
                                                    op: bop,
                                                    left: Box::new(target.clone()),
                                                    right: Box::new(value.clone()),
                                                },
                                                span: *span,
                                            },
                                        };
                                        let new_abs = env.eval_abs_int(&new_val_expr, None);
                                        env.set_int(fkey, new_abs);
                                    }
                                }
                            }
                        }
                        ExprKind::Index {
                            target: list_expr,
                            index,
                        } => {
                            self.check_expr(
                                index,
                                cur_fn,
                                env,
                                smt_int_vars,
                                smt_premises,
                                var_types,
                                mutable_vars,
                                active_mutate,
                                saw_return,
                            );
                            let root_var = match &list_expr.kind {
                                ExprKind::Ident(v) => Some(v.as_str()),
                                ExprKind::Field { target: obj, .. } => match &obj.kind {
                                    ExprKind::Ident(v) => Some(v.as_str()),
                                    _ => None,
                                },
                                _ => None,
                            };
                            if let Some(v) = root_var {
                                if !mutable_vars.contains(v) && var_types.contains_key(v) {
                                    self.diagnostics.push(Diagnostic {
                                        code: "E0203".into(),
                                        message: format!(
                                            "cannot assign to index of immutable binding `{}`",
                                            v
                                        ),
                                        span: *span,
                                        notes: vec![format!("declare `{}` with `var` or `inout`", v)],
                                        counterexample: BTreeMap::new(),
                                    });
                                }
                            }
                            let list_ty = self.resolve_base_type(&self.infer_expr_type(list_expr, var_types));
                            if let TypeExpr::Named { name: ln, args: la, .. } = list_ty {
                                if ln == "List" && la.len() == 1 {
                                    let rhs_ty = self.infer_expr_type(value, var_types);
                                    if let TypeExpr::Dyn { interface, .. } = &la[0] {
                                        if !self.type_implements(&rhs_ty, interface) {
                                            self.diagnostics.push(Diagnostic {
                                                code: "E0505".into(),
                                                message: format!(
                                                    "cannot assign `{}` into `List[dyn {}]`: `{}` does not implement `{}`",
                                                    rhs_ty.display_name(),
                                                    interface,
                                                    rhs_ty.display_name(),
                                                    interface
                                                ),
                                                span: *span,
                                                notes: vec![],
                                                counterexample: BTreeMap::new(),
                                            });
                                        }
                                    } else if !self.types_compatible(&la[0], &rhs_ty) {
                                        self.diagnostics.push(Diagnostic {
                                            code: "E0205".into(),
                                            message: format!(
                                                "type mismatch assigning to list element: expected `{}`, found `{}`",
                                                la[0].display_name(),
                                                rhs_ty.display_name()
                                            ),
                                            span: *span,
                                            notes: vec![],
                                            counterexample: BTreeMap::new(),
                                        });
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
                Stmt::Mutate { target, body, span } => {
                    let prev = active_mutate.replace(target.clone());
                    let _ = self.check_block(
                        body,
                        cur_fn,
                        env,
                        smt_int_vars,
                        smt_premises,
                        var_types,
                        mutable_vars,
                        active_mutate,
                        saw_return,
                    );
                    *active_mutate = prev;

                    if let Some(ty) = var_types.get(target).cloned() {
                        let rec_name = type_head_name(&self.resolve_base_type(&ty));
                        if let Some(rec) = self.records.get(&rec_name).cloned() {
                            self.verify_record_invariants_after_mutate(
                                target,
                                &rec,
                                body,
                                env,
                                smt_int_vars,
                                smt_premises,
                                *span,
                            );
                        }
                    }
                }
                Stmt::While {
                    cond,
                    invariants,
                    body,
                    ..
                } => {
                    // Soundly widen variables mutated inside the loop before checking cond & body!
                    let mut mutated = BTreeSet::new();
                    collect_mutated_vars_in_block(body, &mut mutated);
                    for mv in &mutated {
                        env.ints.remove(mv);
                        env.bools.remove(mv);
                        env.diffs.retain(|(a, b), _| a != mv && b != mv);
                        smt_premises.retain(|(p, _)| !expr_mentions_ident(p, mv));
                        if let Some(mty) = var_types.get(mv).cloned() {
                            self.assume_type_facts(mv, &mty, env, smt_int_vars, smt_premises);
                        }
                    }
                    self.check_expr(
                        cond,
                        cur_fn,
                        env,
                        smt_int_vars,
                        smt_premises,
                        var_types,
                        mutable_vars,
                        active_mutate,
                        saw_return,
                    );
                    let mut body_env = env.clone();
                    let mut body_vars = smt_int_vars.clone();
                    let mut body_premises = smt_premises.clone();
                    let mut body_types = var_types.clone();
                    let mut body_mut = mutable_vars.clone();
                    body_env.assume_expr(cond, None);
                    body_premises.push((cond.clone(), BTreeMap::new()));
                    for inv in invariants {
                        body_env.assume_expr(inv, None);
                        body_premises.push((inv.clone(), BTreeMap::new()));
                    }
                    let _ = self.check_block(
                        body,
                        cur_fn,
                        &mut body_env,
                        &mut body_vars,
                        &mut body_premises,
                        &mut body_types,
                        &mut body_mut,
                        active_mutate,
                        saw_return,
                    );
                }
                Stmt::Expr { expr, .. } => {
                    self.check_expr(
                        expr,
                        cur_fn,
                        env,
                        smt_int_vars,
                        smt_premises,
                        var_types,
                        mutable_vars,
                        active_mutate,
                        saw_return,
                    );
                    // Flow-sensitive continuation narrowing: if `if cond { ... return/break/continue }`
                    // has no `else`, then execution reaching the next statement guarantees `!cond`!
                    if let ExprKind::If {
                        cond,
                        then_block,
                        else_branch: None,
                    } = &expr.kind
                    {
                        if block_always_diverges(then_block) {
                            let neg = Expr {
                                kind: ExprKind::Unary {
                                    op: UnOp::Not,
                                    expr: cond.clone(),
                                },
                                span: cond.span,
                            };
                            env.assume_expr(&neg, None);
                            smt_premises.push((neg, BTreeMap::new()));
                        }
                    }
                }
                Stmt::Return { value, span } => {
                    *saw_return = true;
                    if let Some(e) = value {
                        if let ExprKind::Lambda { params, ret_ty, body } = &e.kind {
                            self.check_lambda(
                                params,
                                ret_ty.as_ref(),
                                body,
                                cur_fn.ret_ty.as_ref(),
                                var_types,
                                e.span,
                            );
                        } else {
                            self.check_expr(
                                e,
                                cur_fn,
                                env,
                                smt_int_vars,
                                smt_premises,
                                var_types,
                                mutable_vars,
                                active_mutate,
                                saw_return,
                            );
                        }
                        if let Some(ret_ty) = &cur_fn.ret_ty {
                            let actual_ty = self.infer_expr_type_expected(e, Some(ret_ty), var_types);
                            if !self.types_compatible(ret_ty, &actual_ty) {
                                self.diagnostics.push(Diagnostic {
                                    code: "E0205".into(),
                                    message: format!(
                                        "return type mismatch in `{}`: expected `{}`, found `{}`",
                                        cur_fn.name,
                                        ret_ty.display_name(),
                                        actual_ty.display_name()
                                    ),
                                    span: *span,
                                    notes: vec![],
                                    counterexample: BTreeMap::new(),
                                });
                            }
                            self.check_expr_satisfies_type(
                                e,
                                ret_ty,
                                env,
                                smt_int_vars,
                                smt_premises,
                                &format!("return value of `{}`", cur_fn.name),
                                e.span,
                            );
                        }
                        for ens in &cur_fn.ensures {
                            self.check_ensures(cur_fn, ens, Some(e), env, smt_int_vars, smt_premises);
                        }
                    }
                }
                Stmt::LetTuple {
                    names,
                    ty,
                    value,
                    is_mut,
                    span,
                } => {
                    self.check_expr(
                        value,
                        cur_fn,
                        env,
                        smt_int_vars,
                        smt_premises,
                        var_types,
                        mutable_vars,
                        active_mutate,
                        saw_return,
                    );
                    let inferred = self.infer_expr_type(value, var_types);
                    let effective_ty = if let Some(t) = ty {
                        self.check_dyn_type_valid(t, *span);
                        self.check_type_bounds(t, *span);
                        if !self.types_compatible(t, &inferred) {
                            self.diagnostics.push(Diagnostic {
                                code: "E0205".into(),
                                message: format!(
                                    "type mismatch in tuple binding: expected `{}`, found `{}`",
                                    t.display_name(),
                                    inferred.display_name()
                                ),
                                span: *span,
                                notes: vec![],
                                counterexample: BTreeMap::new(),
                            });
                        }
                        t.clone()
                    } else {
                        inferred
                    };
                    let base = self.resolve_base_type(&effective_ty);
                    match base {
                        TypeExpr::Tuple(elems, _) => {
                            if elems.len() != names.len() {
                                self.diagnostics.push(Diagnostic {
                                    code: "E0205".into(),
                                    message: format!(
                                        "tuple destructuring arity mismatch: expected {} element(s), found {}",
                                        elems.len(),
                                        names.len()
                                    ),
                                    span: *span,
                                    notes: vec![],
                                    counterexample: BTreeMap::new(),
                                });
                            }
                            for (i, n) in names.iter().enumerate() {
                                if n == "_" {
                                    continue;
                                }
                                let ety = elems.get(i).cloned().unwrap_or_else(|| named_ty("Any"));
                                var_types.insert(n.clone(), ety.clone());
                                if *is_mut {
                                    mutable_vars.insert(n.clone());
                                } else {
                                    mutable_vars.remove(n);
                                }
                                self.assume_type_facts(n, &ety, env, smt_int_vars, smt_premises);
                            }
                        }
                        TypeExpr::Named { ref name, .. } if name == "Any" => {
                            for n in names {
                                if n != "_" {
                                    var_types.insert(n.clone(), named_ty("Any"));
                                    if *is_mut {
                                        mutable_vars.insert(n.clone());
                                    } else {
                                        mutable_vars.remove(n);
                                    }
                                }
                            }
                        }
                        other => {
                            self.diagnostics.push(Diagnostic {
                                code: "E0205".into(),
                                message: format!(
                                    "cannot destructure non-tuple type `{}`",
                                    other.display_name()
                                ),
                                span: *span,
                                notes: vec![],
                                counterexample: BTreeMap::new(),
                            });
                        }
                    }
                }
                Stmt::For {
                    binding,
                    iter,
                    body,
                    span,
                } => {
                    // Soundly widen variables mutated inside the `for` loop body!
                    let mut mutated = BTreeSet::new();
                    collect_mutated_vars_in_block(body, &mut mutated);
                    for mv in &mutated {
                        env.ints.remove(mv);
                        env.bools.remove(mv);
                        env.diffs.retain(|(a, b), _| a != mv && b != mv);
                        smt_premises.retain(|(p, _)| !expr_mentions_ident(p, mv));
                        if let Some(mty) = var_types.get(mv).cloned() {
                            self.assume_type_facts(mv, &mty, env, smt_int_vars, smt_premises);
                        }
                    }
                    self.check_expr(
                        iter,
                        cur_fn,
                        env,
                        smt_int_vars,
                        smt_premises,
                        var_types,
                        mutable_vars,
                        active_mutate,
                        saw_return,
                    );
                    let iter_ty = self.resolve_base_type(&self.infer_expr_type(iter, var_types));
                    let mut body_env = env.clone();
                    let mut body_vars = smt_int_vars.clone();
                    let mut body_premises = smt_premises.clone();
                    let mut body_types = var_types.clone();
                    let mut body_mut = mutable_vars.clone();

                    match &iter_ty {
                        TypeExpr::Named { name, args, .. } if name == "Range" => match binding {
                            ForBinding::Ident(v) => {
                                if v != "_" {
                                    body_types.insert(v.clone(), named_ty("Int"));
                                    body_mut.remove(v);
                                    body_vars.insert(v.clone());
                                    if let ExprKind::Range { start, end } = &iter.kind {
                                        let ge = Expr {
                                            kind: ExprKind::Binary {
                                                op: BinOp::Ge,
                                                left: Box::new(Expr {
                                                    kind: ExprKind::Ident(v.clone()),
                                                    span: *span,
                                                }),
                                                right: start.clone(),
                                            },
                                            span: *span,
                                        };
                                        let lt = Expr {
                                            kind: ExprKind::Binary {
                                                op: BinOp::Lt,
                                                left: Box::new(Expr {
                                                    kind: ExprKind::Ident(v.clone()),
                                                    span: *span,
                                                }),
                                                right: end.clone(),
                                            },
                                            span: *span,
                                        };
                                        body_env.assume_expr(&ge, None);
                                        body_env.assume_expr(&lt, None);
                                        body_premises.push((ge, BTreeMap::new()));
                                        body_premises.push((lt, BTreeMap::new()));
                                    }
                                }
                            }
                            ForBinding::Tuple(_) => {
                                self.diagnostics.push(Diagnostic {
                                    code: "E0205".into(),
                                    message: "cannot destructure `Int` range item as tuple in `for` loop".into(),
                                    span: *span,
                                    notes: vec![],
                                    counterexample: BTreeMap::new(),
                                });
                            }
                        },
                        TypeExpr::Array { elem, .. } => match binding {
                            ForBinding::Ident(v) => {
                                if v != "_" {
                                    let ety: TypeExpr = (**elem).clone();
                                    body_types.insert(v.clone(), ety.clone());
                                    body_mut.remove(v);
                                    self.assume_type_facts(
                                        v,
                                        &ety,
                                        &mut body_env,
                                        &mut body_vars,
                                        &mut body_premises,
                                    );
                                }
                            }
                            ForBinding::Tuple(_) => {
                                self.diagnostics.push(Diagnostic {
                                    code: "E0205".into(),
                                    message: "tuple destructuring in `for` over fixed array requires tuple elements".into(),
                                    span: *span,
                                    notes: vec![],
                                    counterexample: BTreeMap::new(),
                                });
                            }
                        },
                        TypeExpr::Named { name, args, .. } if name == "List" => {
                            let elem_ty = args.first().cloned().unwrap_or_else(|| named_ty("Any"));
                            match binding {
                                ForBinding::Ident(v) => {
                                    if v != "_" {
                                        body_types.insert(v.clone(), elem_ty.clone());
                                        body_mut.remove(v);
                                        self.assume_type_facts(
                                            v,
                                            &elem_ty,
                                            &mut body_env,
                                            &mut body_vars,
                                            &mut body_premises,
                                        );
                                    }
                                }
                                ForBinding::Tuple(names) => {
                                    let eb = self.resolve_base_type(&elem_ty);
                                    match eb {
                                        TypeExpr::Tuple(elems, _) => {
                                            if elems.len() != names.len() {
                                                self.diagnostics.push(Diagnostic {
                                                    code: "E0205".into(),
                                                    message: format!(
                                                        "for-loop tuple pattern expects {} element(s), found {}",
                                                        elems.len(),
                                                        names.len()
                                                    ),
                                                    span: *span,
                                                    notes: vec![],
                                                    counterexample: BTreeMap::new(),
                                                });
                                            }
                                            for (i, n) in names.iter().enumerate() {
                                                if n != "_" {
                                                    let ety = elems.get(i).cloned().unwrap_or_else(|| named_ty("Any"));
                                                    body_types.insert(n.clone(), ety.clone());
                                                    body_mut.remove(n);
                                                    self.assume_type_facts(
                                                        n,
                                                        &ety,
                                                        &mut body_env,
                                                        &mut body_vars,
                                                        &mut body_premises,
                                                    );
                                                }
                                            }
                                        }
                                        TypeExpr::Named { ref name, .. } if name == "Any" => {
                                            for n in names {
                                                if n != "_" {
                                                    body_types.insert(n.clone(), named_ty("Any"));
                                                    body_mut.remove(n);
                                                }
                                            }
                                        }
                                        other => {
                                            self.diagnostics.push(Diagnostic {
                                                code: "E0205".into(),
                                                message: format!(
                                                    "cannot destructure list element `{}` as tuple in `for` loop",
                                                    other.display_name()
                                                ),
                                                span: *span,
                                                notes: vec![],
                                                counterexample: BTreeMap::new(),
                                            });
                                        }
                                    }
                                }
                            }
                        }
                        TypeExpr::Named { name, args, .. } if name == "Map" => {
                            let k_ty = args.first().cloned().unwrap_or_else(|| named_ty("Any"));
                            let v_ty = args.get(1).cloned().unwrap_or_else(|| named_ty("Any"));
                            match binding {
                                ForBinding::Tuple(names) if names.len() == 2 => {
                                    if names[0] != "_" {
                                        body_types.insert(names[0].clone(), k_ty.clone());
                                        body_mut.remove(&names[0]);
                                        self.assume_type_facts(
                                            &names[0],
                                            &k_ty,
                                            &mut body_env,
                                            &mut body_vars,
                                            &mut body_premises,
                                        );
                                    }
                                    if names[1] != "_" {
                                        body_types.insert(names[1].clone(), v_ty.clone());
                                        body_mut.remove(&names[1]);
                                        self.assume_type_facts(
                                            &names[1],
                                            &v_ty,
                                            &mut body_env,
                                            &mut body_vars,
                                            &mut body_premises,
                                        );
                                    }
                                }
                                ForBinding::Ident(k) => {
                                    if k != "_" {
                                        body_types.insert(k.clone(), k_ty.clone());
                                        body_mut.remove(k);
                                        self.assume_type_facts(
                                            k,
                                            &k_ty,
                                            &mut body_env,
                                            &mut body_vars,
                                            &mut body_premises,
                                        );
                                    }
                                }
                                _ => {
                                    self.diagnostics.push(Diagnostic {
                                        code: "E0205".into(),
                                        message: "`for` over `Map` expects `for (k, v) in m` or `for k in m`".into(),
                                        span: *span,
                                        notes: vec![],
                                        counterexample: BTreeMap::new(),
                                    });
                                }
                            }
                        }
                        TypeExpr::Named { name, .. } if name == "Any" => match binding {
                            ForBinding::Ident(v) => {
                                if v != "_" {
                                    body_types.insert(v.clone(), named_ty("Any"));
                                    body_mut.remove(v);
                                }
                            }
                            ForBinding::Tuple(names) => {
                                for n in names {
                                    if n != "_" {
                                        body_types.insert(n.clone(), named_ty("Any"));
                                        body_mut.remove(n);
                                    }
                                }
                            }
                        },
                        other => {
                            self.diagnostics.push(Diagnostic {
                                code: "E0205".into(),
                                message: format!(
                                    "cannot iterate over `{}` in `for` loop (expected Range, List, or Map)",
                                    other.display_name()
                                ),
                                span: *span,
                                notes: vec![],
                                counterexample: BTreeMap::new(),
                            });
                        }
                    }
                    let _ = self.check_block(
                        body,
                        cur_fn,
                        &mut body_env,
                        &mut body_vars,
                        &mut body_premises,
                        &mut body_types,
                        &mut body_mut,
                        active_mutate,
                        saw_return,
                    );
                }
                Stmt::Defer { body, .. } => {
                    let mut d_env = env.clone();
                    let mut d_vars = smt_int_vars.clone();
                    let mut d_prem = smt_premises.clone();
                    let mut d_types = var_types.clone();
                    let mut d_mut = mutable_vars.clone();
                    let mut d_sr = false;
                    let _ = self.check_block(
                        body,
                        cur_fn,
                        &mut d_env,
                        &mut d_vars,
                        &mut d_prem,
                        &mut d_types,
                        &mut d_mut,
                        active_mutate,
                        &mut d_sr,
                    );
                }
                Stmt::Break { .. } | Stmt::Continue { .. } => {}
            }
        }
        if let Some(tail) = &blk.tail {
            if let ExprKind::Lambda { params, ret_ty, body } = &tail.kind {
                let exp = match cur_fn.ret_ty.as_ref().map(|t| self.resolve_base_type(t)) {
                    Some(t @ TypeExpr::Fn { .. }) => Some(t),
                    _ => None,
                };
                self.check_lambda(params, ret_ty.as_ref(), body, exp.as_ref(), var_types, tail.span);
            } else {
                self.check_expr(
                    tail,
                    cur_fn,
                    env,
                    smt_int_vars,
                    smt_premises,
                    var_types,
                    mutable_vars,
                    active_mutate,
                    saw_return,
                );
            }
            return Some((**tail).clone());
        }
        None
    }

    #[allow(clippy::too_many_arguments)]
    fn verify_record_invariants_after_mutate(
        &mut self,
        var: &str,
        rec: &RecordDecl,
        body: &Block,
        env: &L0Env,
        smt_int_vars: &BTreeSet<String>,
        smt_premises: &[(Expr, BTreeMap<String, String>)],
        span: Span,
    ) {
        let mut sym_fields: BTreeMap<String, String> = BTreeMap::new();
        for f in &rec.fields {
            sym_fields.insert(
                f.name.clone(),
                crate::smt::smt_ident(&format!("{}.{}", var, f.name)),
            );
        }
        let mut mutated_any = false;
        apply_mutate_block_symbolic(var, body, &mut sym_fields, &mut mutated_any);

        let mut subst = BTreeMap::new();
        subst.insert("self".into(), var.to_string());
        for f in &rec.fields {
            let post_expr = sym_fields
                .get(&f.name)
                .cloned()
                .unwrap_or_else(|| crate::smt::smt_ident(&format!("{}.{}", var, f.name)));
            subst.insert(f.name.clone(), post_expr.clone());
            subst.insert(format!("{}.{}", var, f.name), post_expr);
        }

        for f in &rec.fields {
            let mut field_preds: Vec<Expr> = Vec::new();
            if let Some(alias) = self.type_aliases.get(&f.ty.display_name()) {
                if let Some(ap) = &alias.predicate {
                    field_preds.push(ap.clone());
                }
            }
            if let Some(pred) = &f.predicate {
                field_preds.push(pred.clone());
            }
            for pred in &field_preds {
                let fkey = format!("{}.{}", var, f.name);
                let rewritten = substitute_field_idents(pred, &rec.fields, var);
                let l0_res = if mutated_any {
                    L0Result::Unknown
                } else {
                    env.check_predicate(&rewritten, Some(&fkey))
                };
                match l0_res {
                    L0Result::Proved => {
                        self.proofs.push(ProofRecord {
                            obligation: format!("mutate({}) invariant `{}.{}`", var, rec.name, f.name),
                            tier: ProofTier::L0Fast,
                            span,
                        });
                    }
                    L0Result::Disproved(reason) => {
                        self.diagnostics.push(Diagnostic {
                            code: "E0421".into(),
                            message: format!(
                                "mutate `{}` violates field invariant `{}.{}`",
                                var, rec.name, f.name
                            ),
                            span,
                            notes: vec![reason],
                            counterexample: BTreeMap::new(),
                        });
                    }
                    L0Result::Unknown => {
                        let mut q = SmtQuery::new(self.smt_pure_fns.clone());
                        for v in smt_int_vars {
                            q.declare_int(v.clone());
                        }
                        for (p, s) in smt_premises {
                            if var == "self" && !s.contains_key("self") {
                                let mut s2 = s.clone();
                                s2.insert("self".into(), "self".into());
                                let _ = q.add_premise_expr(p, &s2);
                            } else {
                                let _ = q.add_premise_expr(p, s);
                            }
                        }
                        let mut fs = subst.clone();
                        fs.insert(
                            "self".into(),
                            sym_fields
                                .get(&f.name)
                                .cloned()
                                .unwrap_or_else(|| crate::smt::smt_ident(&fkey)),
                        );
                        if q.set_goal_expr(pred, &fs).is_ok() {
                            match q.solve() {
                                SmtOutcome::Proved => {
                                    self.proofs.push(ProofRecord {
                                        obligation: format!(
                                            "mutate({}) invariant `{}.{}`",
                                            var, rec.name, f.name
                                        ),
                                        tier: ProofTier::L1Smt,
                                        span,
                                    });
                                }
                                SmtOutcome::Counterexample(ce) => {
                                    self.diagnostics.push(Diagnostic {
                                        code: "E0421".into(),
                                        message: format!(
                                            "mutate `{}` may violate field invariant `{}.{}`",
                                            var, rec.name, f.name
                                        ),
                                        span,
                                        notes: vec!["counterexample found by L1 SMT (Z3)".into()],
                                        counterexample: ce,
                                    });
                                }
                                SmtOutcome::Unknown(msg) => {
                                    self.diagnostics.push(Diagnostic {
                                        code: "E0422".into(),
                                        message: format!(
                                            "unproved mutate `{}` invariant `{}.{}` (Fail-Closed)",
                                            var, rec.name, f.name
                                        ),
                                        span,
                                        notes: vec![msg],
                                        counterexample: BTreeMap::new(),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        for wc in &rec.where_clauses {
            let mut q = SmtQuery::new(self.smt_pure_fns.clone());
            for v in smt_int_vars {
                q.declare_int(v.clone());
            }
            for (p, s) in smt_premises {
                if var == "self" && !s.contains_key("self") {
                    let mut s2 = s.clone();
                    s2.insert("self".into(), "self".into());
                    let _ = q.add_premise_expr(p, &s2);
                } else {
                    let _ = q.add_premise_expr(p, s);
                }
            }
            if q.set_goal_expr(wc, &subst).is_ok() {
                match q.solve() {
                    SmtOutcome::Proved => {
                        self.proofs.push(ProofRecord {
                            obligation: format!("mutate({}) record invariant `{}`", var, rec.name),
                            tier: ProofTier::L1Smt,
                            span,
                        });
                    }
                    SmtOutcome::Counterexample(ce) => {
                        self.diagnostics.push(Diagnostic {
                            code: "E0421".into(),
                            message: format!(
                                "mutate `{}` may violate record invariant on `{}`",
                                var, rec.name
                            ),
                            span,
                            notes: vec!["counterexample found by L1 SMT (Z3)".into()],
                            counterexample: ce,
                        });
                    }
                    SmtOutcome::Unknown(msg) => {
                        self.diagnostics.push(Diagnostic {
                            code: "E0422".into(),
                            message: format!(
                                "unproved mutate `{}` record invariant on `{}` (Fail-Closed)",
                                var, rec.name
                            ),
                            span,
                            notes: vec![msg],
                            counterexample: BTreeMap::new(),
                        });
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn check_expr(
        &mut self,
        expr: &Expr,
        cur_fn: &FnDecl,
        env: &mut L0Env,
        smt_int_vars: &mut BTreeSet<String>,
        smt_premises: &mut Vec<(Expr, BTreeMap<String, String>)>,
        var_types: &BTreeMap<String, TypeExpr>,
        mutable_vars: &BTreeSet<String>,
        active_mutate: &Option<String>,
        saw_return: &mut bool,
    ) {
        match &expr.kind {
            ExprKind::Binary { op, left, right } => {
                self.check_expr(
                    left,
                    cur_fn,
                    env,
                    smt_int_vars,
                    smt_premises,
                    var_types,
                    mutable_vars,
                    active_mutate,
                    saw_return,
                );
                self.check_expr(
                    right,
                    cur_fn,
                    env,
                    smt_int_vars,
                    smt_premises,
                    var_types,
                    mutable_vars,
                    active_mutate,
                    saw_return,
                );
                let mut lt = self.resolve_base_type(&self.infer_expr_type(left, var_types));
                let mut rt = self.resolve_base_type(&self.infer_expr_type(right, var_types));
                if is_untyped_int_expr(left) && self.is_int_ty(&rt) {
                    lt = rt.clone();
                } else if is_untyped_int_expr(right) && self.is_int_ty(&lt) {
                    rt = lt.clone();
                } else if is_untyped_float_expr(left) && self.is_float_ty(&rt) {
                    lt = rt.clone();
                } else if is_untyped_float_expr(right) && self.is_float_ty(&lt) {
                    rt = lt.clone();
                }
                let l_float = self.is_float_ty(&lt);
                let r_float = self.is_float_ty(&rt);
                let l_int = self.is_int_ty(&lt);
                let r_int = self.is_int_ty(&rt);
                let l_sk = self.scalar_kind_of_ty(&lt);
                let r_sk = self.scalar_kind_of_ty(&rt);
                match op {
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
                        if (l_float && r_int)
                            || (l_int && r_float)
                            || matches!((l_sk, r_sk), (Some(a), Some(b)) if a != b)
                        {
                            self.diagnostics.push(Diagnostic {
                                code: "E0205".into(),
                                message: format!(
                                    "cannot mix `{}` and `{}` in arithmetic expression",
                                    lt.display_name(),
                                    rt.display_name()
                                ),
                                span: expr.span,
                                notes: vec!["convert explicitly with `as` cast (e.g. `x as Int64`)".into()],
                                counterexample: BTreeMap::new(),
                            });
                        }
                    }
                    BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr => {
                        if l_float || r_float {
                            self.diagnostics.push(Diagnostic {
                                code: "E0205".into(),
                                message: "bitwise operators are not supported on `Float`".into(),
                                span: expr.span,
                                notes: vec![],
                                counterexample: BTreeMap::new(),
                            });
                        } else if matches!(op, BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor)
                            && matches!((l_sk, r_sk), (Some(a), Some(b)) if a != b)
                        {
                            self.diagnostics.push(Diagnostic {
                                code: "E0205".into(),
                                message: format!(
                                    "cannot mix `{}` and `{}` in bitwise expression",
                                    lt.display_name(),
                                    rt.display_name()
                                ),
                                span: expr.span,
                                notes: vec!["convert explicitly with `as` cast".into()],
                                counterexample: BTreeMap::new(),
                            });
                        }
                    }
                    BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                        if (l_float && r_int)
                            || (l_int && r_float)
                            || matches!((l_sk, r_sk), (Some(a), Some(b)) if a != b)
                        {
                            self.diagnostics.push(Diagnostic {
                                code: "E0205".into(),
                                message: format!(
                                    "cannot compare `{}` and `{}`",
                                    lt.display_name(),
                                    rt.display_name()
                                ),
                                span: expr.span,
                                notes: vec!["convert explicitly with `as` cast".into()],
                                counterexample: BTreeMap::new(),
                            });
                        }
                    }
                    BinOp::Eq | BinOp::Ne => {
                        if !self.types_compatible(&lt, &rt) {
                            self.diagnostics.push(Diagnostic {
                                code: "E0205".into(),
                                message: format!(
                                    "cannot compare `{}` and `{}` for equality",
                                    lt.display_name(),
                                    rt.display_name()
                                ),
                                span: expr.span,
                                notes: vec![],
                                counterexample: BTreeMap::new(),
                            });
                        } else if let TypeExpr::Named { name: ln, .. } = &lt {
                            let is_prim = ln == "Any"
                                || ln == "Bool"
                                || ln == "Text"
                                || self.is_int_type_name(ln)
                                || self.is_float_type_name(ln)
                                || self.enums.contains_key(ln);
                            if !is_prim && !self.type_implements(&lt, "Eq") {
                                self.diagnostics.push(Diagnostic {
                                    code: "E0505".into(),
                                    message: format!(
                                        "cannot compare `{}` with `==`/`!=`: type does not implement `Eq`",
                                        lt.display_name()
                                    ),
                                    span: expr.span,
                                    notes: vec![format!("add `@derive(Eq)` or `impl Eq for {} {{ ... }}`", lt.display_name())],
                                    counterexample: BTreeMap::new(),
                                });
                            }
                        }
                    }
                    _ => {}
                }
                if matches!(op, BinOp::Div | BinOp::Mod) && !l_float && !r_float {
                    let denom = env.eval_abs_int(right, None);
                    if !denom.non_zero {
                        let ne_zero = Expr {
                            kind: ExprKind::Binary {
                                op: BinOp::Ne,
                                left: right.clone(),
                                right: Box::new(Expr {
                                    kind: ExprKind::Int(0),
                                    span: right.span,
                                }),
                            },
                            span: right.span,
                        };
                        if matches!(env.check_predicate(&ne_zero, None), L0Result::Proved) {
                            self.proofs.push(ProofRecord {
                                obligation: "division denominator != 0".into(),
                                tier: ProofTier::L0Fast,
                                span: expr.span,
                            });
                        } else {
                            self.diagnostics.push(Diagnostic {
                                code: "E0430".into(),
                                message: "potential division/modulo by zero: denominator is not proved NonZero".into(),
                                span: expr.span,
                                notes: vec!["refine denominator with `where self != 0` or guard with `if denom != 0`".into()],
                                counterexample: BTreeMap::new(),
                            });
                        }
                    } else {
                        self.proofs.push(ProofRecord {
                            obligation: "division denominator != 0 (elided)".into(),
                            tier: ProofTier::L0Fast,
                            span: expr.span,
                        });
                    }
                }
            }
            ExprKind::Unary { expr: inner, .. } | ExprKind::Await(inner) | ExprKind::Old(inner) => {
                self.check_expr(
                    inner,
                    cur_fn,
                    env,
                    smt_int_vars,
                    smt_premises,
                    var_types,
                    mutable_vars,
                    active_mutate,
                    saw_return,
                );
            }
            ExprKind::Try(inner) => {
                self.check_expr(
                    inner,
                    cur_fn,
                    env,
                    smt_int_vars,
                    smt_premises,
                    var_types,
                    mutable_vars,
                    active_mutate,
                    saw_return,
                );
                let inner_ty = self.resolve_base_type(&self.infer_expr_type(inner, var_types));
                let fn_ret = cur_fn.ret_ty.as_ref().map(|t| self.resolve_base_type(t));
                match &inner_ty {
                    TypeExpr::Named { name, args, .. } if name == "Result" => match &fn_ret {
                        Some(TypeExpr::Named { name: rn, args: rargs, .. }) if rn == "Result" => {
                            if args.len() >= 2 && rargs.len() >= 2 && !self.types_compatible(&rargs[1], &args[1]) {
                                self.diagnostics.push(Diagnostic {
                                    code: "E0205".into(),
                                    message: format!(
                                        "`?` error type mismatch in `{}`: function returns error `{}`, expression has error `{}`",
                                        cur_fn.name,
                                        rargs[1].display_name(),
                                        args[1].display_name()
                                    ),
                                    span: expr.span,
                                    notes: vec![],
                                    counterexample: BTreeMap::new(),
                                });
                            }
                        }
                        Some(TypeExpr::Named { name: rn, .. }) if rn == "Any" => {}
                        _ => {
                            self.diagnostics.push(Diagnostic {
                                code: "E0208".into(),
                                message: format!(
                                    "operator `?` on `Result` can only be used in a function returning `Result`, but `{}` returns `{}`",
                                    cur_fn.name,
                                    cur_fn
                                        .ret_ty
                                        .as_ref()
                                        .map(|t| t.display_name())
                                        .unwrap_or_else(|| "()".into())
                                ),
                                span: expr.span,
                                notes: vec![],
                                counterexample: BTreeMap::new(),
                            });
                        }
                    },
                    TypeExpr::Named { name, .. } if name == "Option" => match &fn_ret {
                        Some(TypeExpr::Named { name: rn, .. }) if rn == "Option" || rn == "Any" => {}
                        _ => {
                            self.diagnostics.push(Diagnostic {
                                code: "E0208".into(),
                                message: format!(
                                    "operator `?` on `Option` can only be used in a function returning `Option`, but `{}` returns `{}`",
                                    cur_fn.name,
                                    cur_fn
                                        .ret_ty
                                        .as_ref()
                                        .map(|t| t.display_name())
                                        .unwrap_or_else(|| "()".into())
                                ),
                                span: expr.span,
                                notes: vec![],
                                counterexample: BTreeMap::new(),
                            });
                        }
                    },
                    TypeExpr::Named { name, .. } if name == "Any" => {}
                    other => {
                        self.diagnostics.push(Diagnostic {
                            code: "E0208".into(),
                            message: format!(
                                "operator `?` requires `Result` or `Option`, found `{}`",
                                other.display_name()
                            ),
                            span: expr.span,
                            notes: vec![],
                            counterexample: BTreeMap::new(),
                        });
                    }
                }
            }
            ExprKind::TupleIndex { target, index } => {
                self.check_expr(
                    target,
                    cur_fn,
                    env,
                    smt_int_vars,
                    smt_premises,
                    var_types,
                    mutable_vars,
                    active_mutate,
                    saw_return,
                );
                let tty = self.resolve_base_type(&self.infer_expr_type(target, var_types));
                match tty {
                    TypeExpr::Tuple(elems, _) => {
                        if *index >= elems.len() {
                            self.diagnostics.push(Diagnostic {
                                code: "E0206".into(),
                                message: format!(
                                    "tuple index `.{}` out of bounds for {}-tuple",
                                    index,
                                    elems.len()
                                ),
                                span: expr.span,
                                notes: vec![],
                                counterexample: BTreeMap::new(),
                            });
                        }
                    }
                    TypeExpr::Named { ref name, .. } if name == "Any" => {}
                    other => {
                        self.diagnostics.push(Diagnostic {
                            code: "E0206".into(),
                            message: format!(
                                "cannot use tuple index `.{}` on non-tuple type `{}`",
                                index,
                                other.display_name()
                            ),
                            span: expr.span,
                            notes: vec![],
                            counterexample: BTreeMap::new(),
                        });
                    }
                }
            }
            ExprKind::Range { start, end } => {
                self.check_expr(
                    start,
                    cur_fn,
                    env,
                    smt_int_vars,
                    smt_premises,
                    var_types,
                    mutable_vars,
                    active_mutate,
                    saw_return,
                );
                self.check_expr(
                    end,
                    cur_fn,
                    env,
                    smt_int_vars,
                    smt_premises,
                    var_types,
                    mutable_vars,
                    active_mutate,
                    saw_return,
                );
                let st = self.infer_expr_type(start, var_types);
                let et = self.infer_expr_type(end, var_types);
                if !self.types_compatible(&named_ty("Int"), &st) {
                    self.diagnostics.push(Diagnostic {
                        code: "E0205".into(),
                        message: format!(
                            "range start expects `Int`, found `{}`",
                            st.display_name()
                        ),
                        span: start.span,
                        notes: vec![],
                        counterexample: BTreeMap::new(),
                    });
                }
                if !self.types_compatible(&named_ty("Int"), &et) {
                    self.diagnostics.push(Diagnostic {
                        code: "E0205".into(),
                        message: format!(
                            "range end expects `Int`, found `{}`",
                            et.display_name()
                        ),
                        span: end.span,
                        notes: vec![],
                        counterexample: BTreeMap::new(),
                    });
                }
            }
            ExprKind::Lambda { params, ret_ty, body } => {
                self.check_lambda(params, ret_ty.as_ref(), body, None, var_types, expr.span);
            }
            ExprKind::List(elems) | ExprKind::Tuple(elems) => {
                for e in elems {
                    self.check_expr(
                        e,
                        cur_fn,
                        env,
                        smt_int_vars,
                        smt_premises,
                        var_types,
                        mutable_vars,
                        active_mutate,
                        saw_return,
                    );
                }
            }
            ExprKind::ArrayRepeat { value, .. } => {
                self.check_expr(
                    value,
                    cur_fn,
                    env,
                    smt_int_vars,
                    smt_premises,
                    var_types,
                    mutable_vars,
                    active_mutate,
                    saw_return,
                );
            }
            ExprKind::Cast { expr: inner, ty } => {
                self.check_expr(
                    inner,
                    cur_fn,
                    env,
                    smt_int_vars,
                    smt_premises,
                    var_types,
                    mutable_vars,
                    active_mutate,
                    saw_return,
                );
                let src_ty = self.resolve_base_type(&self.infer_expr_type(inner, var_types));
                let dst_ty = self.resolve_base_type(ty);
                let src_ok = self.scalar_kind_of_ty(&src_ty).is_some()
                    || matches!(&src_ty, TypeExpr::Named { name, .. } if matches!(name.as_str(), "Bool" | "Ptr" | "MutPtr" | "CStr" | "Any") || self.bitfields.contains_key(name));
                let dst_ok = self.scalar_kind_of_ty(&dst_ty).is_some()
                    || matches!(&dst_ty, TypeExpr::Named { name, .. } if matches!(name.as_str(), "Bool" | "Ptr" | "MutPtr" | "CStr" | "Any") || self.bitfields.contains_key(name));
                if !src_ok || !dst_ok {
                    self.diagnostics.push(Diagnostic {
                        code: "E0205".into(),
                        message: format!(
                            "invalid cast from `{}` to `{}`",
                            src_ty.display_name(),
                            dst_ty.display_name()
                        ),
                        span: expr.span,
                        notes: vec!["`as` supports scalar, pointer, bool, and bitfield casts".into()],
                        counterexample: BTreeMap::new(),
                    });
                }
            }
            ExprKind::Unsafe(blk) => {
                let prev = self.in_unsafe;
                self.in_unsafe = true;
                let mut u_env = env.clone();
                let mut u_vars = var_types.clone();
                let mut u_mut = mutable_vars.clone();
                let mut m = active_mutate.clone();
                let _ = self.check_block(
                    blk,
                    cur_fn,
                    &mut u_env,
                    &mut smt_int_vars.clone(),
                    &mut smt_premises.clone(),
                    &mut u_vars,
                    &mut u_mut,
                    &mut m,
                    saw_return,
                );
                self.in_unsafe = prev;
            }
            ExprKind::Index { target, index } => {
                self.check_expr(
                    target,
                    cur_fn,
                    env,
                    smt_int_vars,
                    smt_premises,
                    var_types,
                    mutable_vars,
                    active_mutate,
                    saw_return,
                );
                self.check_expr(
                    index,
                    cur_fn,
                    env,
                    smt_int_vars,
                    smt_premises,
                    var_types,
                    mutable_vars,
                    active_mutate,
                    saw_return,
                );
                let tty = self.resolve_base_type(&self.infer_expr_type(target, var_types));
                if let TypeExpr::Array { len, .. } = tty {
                    let idx_abs = env.eval_abs_int(index, None);
                    if idx_abs.interval.max < 0 || idx_abs.interval.min >= len as i128 {
                        self.diagnostics.push(Diagnostic {
                            code: "E0410".into(),
                            message: format!(
                                "fixed array index out of bounds for length {}",
                                len
                            ),
                            span: index.span,
                            notes: vec![format!("index interval [{}, {}] outside 0..{}", idx_abs.interval.min, idx_abs.interval.max, len)],
                            counterexample: BTreeMap::new(),
                        });
                    } else if idx_abs.interval.min >= 0 && idx_abs.interval.max < len as i128 {
                        self.proofs.push(ProofRecord {
                            obligation: format!("array index in bounds 0..{} (elided)", len),
                            tier: ProofTier::L0Fast,
                            span: index.span,
                        });
                    }
                }
            }
            ExprKind::Field { target, field } => {
                if let ExprKind::Ident(ename) = &target.kind {
                    if self.enum_variants.contains_key(&format!("{}.{}", ename, field)) {
                        return;
                    }
                }
                self.check_expr(
                    target,
                    cur_fn,
                    env,
                    smt_int_vars,
                    smt_premises,
                    var_types,
                    mutable_vars,
                    active_mutate,
                    saw_return,
                );
                let tty = self.resolve_base_type(&self.infer_expr_type(target, var_types));
                if let TypeExpr::Named { name: rname, .. } = tty {
                    if let Some(rec) = self.records.get(&rname) {
                        if !rec.fields.iter().any(|f| f.name == *field) {
                            self.diagnostics.push(Diagnostic {
                                code: "E0206".into(),
                                message: format!("unknown field `{}` on record `{}`", field, rname),
                                span: expr.span,
                                notes: vec![],
                                counterexample: BTreeMap::new(),
                            });
                        }
                    } else if let Some(bf) = self.bitfields.get(&rname) {
                        if !bf.fields.iter().any(|f| f.name == *field) {
                            self.diagnostics.push(Diagnostic {
                                code: "E0206".into(),
                                message: format!("unknown field `{}` on bitfield `{}`", field, rname),
                                span: expr.span,
                                notes: vec![],
                                counterexample: BTreeMap::new(),
                            });
                        }
                    } else if let Some(u) = self.unions.get(&rname) {
                        if !u.fields.iter().any(|f| f.name == *field) {
                            self.diagnostics.push(Diagnostic {
                                code: "E0206".into(),
                                message: format!("unknown field `{}` on union `{}`", field, rname),
                                span: expr.span,
                                notes: vec![],
                                counterexample: BTreeMap::new(),
                            });
                        } else if !self.in_unsafe {
                            self.diagnostics.push(Diagnostic {
                                code: "E0800".into(),
                                message: format!(
                                    "reading field `{}` of untagged union `{}` requires an `unsafe` block",
                                    field, rname
                                ),
                                span: expr.span,
                                notes: vec!["wrap union field access in `unsafe { ... }`".into()],
                                counterexample: BTreeMap::new(),
                            });
                        }
                    }
                }
            }
            ExprKind::RecordLit { name, fields, .. } => {
                for (_, val) in fields {
                    self.check_expr(
                        val,
                        cur_fn,
                        env,
                        smt_int_vars,
                        smt_premises,
                        var_types,
                        mutable_vars,
                        active_mutate,
                        saw_return,
                    );
                }
                if let Some(rec) = self.records.get(name).cloned() {
                    let mut rsubst = BTreeMap::new();
                    if !rec.type_params.is_empty() {
                        fn unify_lit(
                            pat: &TypeExpr,
                            actual: &TypeExpr,
                            tps: &[String],
                            subst: &mut BTreeMap<String, TypeExpr>,
                        ) {
                            match (pat, actual) {
                                (TypeExpr::Named { name, args, .. }, _)
                                    if args.is_empty() && tps.contains(name) =>
                                {
                                    if actual.display_name() != "Any" {
                                        subst.entry(name.clone()).or_insert_with(|| actual.clone());
                                    }
                                }
                                (
                                    TypeExpr::Named { name: pn, args: pa, .. },
                                    TypeExpr::Named { name: an, args: aa, .. },
                                ) if pn == an && pa.len() == aa.len() => {
                                    for (x, y) in pa.iter().zip(aa.iter()) {
                                        unify_lit(x, y, tps, subst);
                                    }
                                }
                                _ => {}
                            }
                        }
                        for (fname, fval) in fields {
                            if let Some(fdecl) = rec.fields.iter().find(|f| f.name == *fname) {
                                let aty = self.infer_expr_type(fval, var_types);
                                unify_lit(&fdecl.ty, &aty, &rec.type_params, &mut rsubst);
                            }
                        }
                    }
                    // Check unknown & missing required fields + field type compatibility!
                    let provided_names: BTreeSet<&str> =
                        fields.iter().map(|(n, _)| n.as_str()).collect();
                    for (fname, fval) in fields {
                        if let Some(fdecl) = rec.fields.iter().find(|f| f.name == *fname) {
                            let expected_fty = if rsubst.is_empty() {
                                fdecl.ty.clone()
                            } else {
                                subst_type_expr(&fdecl.ty, &rsubst)
                            };
                            let actual_ty =
                                self.infer_expr_type_expected(fval, Some(&expected_fty), var_types);
                            if !self.types_compatible(&expected_fty, &actual_ty) {
                                self.diagnostics.push(Diagnostic {
                                    code: "E0205".into(),
                                    message: format!(
                                        "field `{}` of `{}` expects `{}`, found `{}`",
                                        fname,
                                        rec.name,
                                        expected_fty.display_name(),
                                        actual_ty.display_name()
                                    ),
                                    span: fval.span,
                                    notes: vec![],
                                    counterexample: BTreeMap::new(),
                                });
                            }
                        } else {
                            self.diagnostics.push(Diagnostic {
                                code: "E0204".into(),
                                message: format!(
                                    "unknown field `{}` in `{}` literal",
                                    fname, rec.name
                                ),
                                span: fval.span,
                                notes: vec![],
                                counterexample: BTreeMap::new(),
                            });
                        }
                    }
                    for fdecl in &rec.fields {
                        if fdecl.default.is_none() && !provided_names.contains(fdecl.name.as_str()) {
                            self.diagnostics.push(Diagnostic {
                                code: "E0204".into(),
                                message: format!(
                                    "missing required field `{}` in `{}` literal",
                                    fdecl.name, rec.name
                                ),
                                span: expr.span,
                                notes: vec![],
                                counterexample: BTreeMap::new(),
                            });
                        }
                    }
                    self.check_record_construction(
                        &rec,
                        fields,
                        env,
                        smt_int_vars,
                        smt_premises,
                        expr.span,
                    );
                } else if let Some(bf) = self.bitfields.get(name).cloned() {
                    for (fname, fval) in fields {
                        if let Some(bfd) = bf.fields.iter().find(|f| f.name == *fname) {
                            let max_v = if bfd.bits >= 64 {
                                u64::MAX as i128
                            } else {
                                ((1u64 << bfd.bits) - 1) as i128
                            };
                            let abs = env.eval_abs_int(fval, None);
                            if abs.interval.max < 0 || abs.interval.min > max_v {
                                self.diagnostics.push(Diagnostic {
                                    code: "E0410".into(),
                                    message: format!(
                                        "bitfield `{}.{}` ({}-bit) value out of range 0..={}",
                                        bf.name, fname, bfd.bits, max_v
                                    ),
                                    span: fval.span,
                                    notes: vec![],
                                    counterexample: BTreeMap::new(),
                                });
                            }
                        } else {
                            self.diagnostics.push(Diagnostic {
                                code: "E0204".into(),
                                message: format!(
                                    "unknown field `{}` in bitfield `{}` literal",
                                    fname, bf.name
                                ),
                                span: fval.span,
                                notes: vec![],
                                counterexample: BTreeMap::new(),
                            });
                        }
                    }
                } else if let Some(u) = self.unions.get(name).cloned() {
                    if fields.len() != 1 {
                        self.diagnostics.push(Diagnostic {
                            code: "E0204".into(),
                            message: format!(
                                "union `{}` literal must initialize exactly 1 field, found {}",
                                u.name,
                                fields.len()
                            ),
                            span: expr.span,
                            notes: vec![],
                            counterexample: BTreeMap::new(),
                        });
                    }
                    for (fname, fval) in fields {
                        if let Some(uf) = u.fields.iter().find(|f| f.name == *fname) {
                            let actual =
                                self.infer_expr_type_expected(fval, Some(&uf.ty), var_types);
                            if !self.types_compatible(&uf.ty, &actual) {
                                self.diagnostics.push(Diagnostic {
                                    code: "E0205".into(),
                                    message: format!(
                                        "union field `{}.{}` expects `{}`, found `{}`",
                                        u.name,
                                        fname,
                                        uf.ty.display_name(),
                                        actual.display_name()
                                    ),
                                    span: fval.span,
                                    notes: vec![],
                                    counterexample: BTreeMap::new(),
                                });
                            }
                        } else {
                            self.diagnostics.push(Diagnostic {
                                code: "E0204".into(),
                                message: format!(
                                    "unknown field `{}` in union `{}` literal",
                                    fname, u.name
                                ),
                                span: fval.span,
                                notes: vec![],
                                counterexample: BTreeMap::new(),
                            });
                        }
                    }
                }
            }
            ExprKind::Call { callee, args } => {
                // Check non-lambda arguments upfront; lambda arguments are checked with bidirectional expected fn types!
                for a in args {
                    if !matches!(a.value.kind, ExprKind::Lambda { .. }) {
                        self.check_expr(
                            &a.value,
                            cur_fn,
                            env,
                            smt_int_vars,
                            smt_premises,
                            var_types,
                            mutable_vars,
                            active_mutate,
                            saw_return,
                        );
                    }
                }
                let mut lambdas_checked = false;
                match &callee.kind {
                    ExprKind::Field { target, field }
                        if field == "parse"
                            && matches!(&target.kind, ExprKind::Ident(rname) if self.records.contains_key(rname) && self.resolve_assoc_method_sig(rname, "parse").is_none()) =>
                    {
                        if let ExprKind::Ident(rname) = &target.kind {
                            if let Some(rec) = self.records.get(rname) {
                                if args.len() != rec.fields.len() {
                                    self.diagnostics.push(Diagnostic {
                                        code: "E0202".into(),
                                        message: format!(
                                            "`{}.parse` expects {} argument(s), found {}",
                                            rname,
                                            rec.fields.len(),
                                            args.len()
                                        ),
                                        span: expr.span,
                                        notes: vec![],
                                        counterexample: BTreeMap::new(),
                                    });
                                }
                            }
                        }
                    }
                    ExprKind::Field { target, field } => {
                        let is_enum_ctor = matches!(&target.kind, ExprKind::Ident(ename)
                            if self.enum_variants.contains_key(&format!("{}.{}", ename, field)));
                        if is_enum_ctor {
                            if let ExprKind::Ident(ename) = &target.kind {
                                if let Some(vi) =
                                    self.enum_variants.get(&format!("{}.{}", ename, field)).cloned()
                                {
                                    self.check_variant_ctor_args(&format!("{}.{}", ename, field), &vi, args, var_types, expr.span);
                                }
                            }
                        } else if let ExprKind::GenericInst { expr: inner, type_args } = &target.kind {
                            if let ExprKind::Ident(tname) = &inner.kind {
                                if !var_types.contains_key(tname) && self.is_known_type_name(tname) {
                                    let inst_ty = generic_ty(tname, type_args.clone());
                                    self.check_type_bounds(&inst_ty, expr.span);
                                    if let Some(mdecl) = self.resolve_assoc_method_sig(tname, field) {
                                        for (tp, ta) in mdecl.type_params.iter().zip(type_args.iter()) {
                                            if ta.display_name() != "Any" && !self.cur_type_params.contains(&ta.display_name()) {
                                                if let Some(bounds) = mdecl.type_param_bounds.get(tp) {
                                                    for b in bounds {
                                                        if !self.type_implements(ta, b) {
                                                            self.diagnostics.push(Diagnostic {
                                                                code: "E0505".into(),
                                                                message: format!(
                                                                    "type `{}` does not satisfy interface bound `{}` for type parameter `{}` of `{}.{}`",
                                                                    ta.display_name(),
                                                                    b,
                                                                    tp,
                                                                    tname,
                                                                    field
                                                                ),
                                                                span: expr.span,
                                                                notes: vec![format!("add `impl {} for {} {{ ... }}`", b, ta.display_name())],
                                                                counterexample: BTreeMap::new(),
                                                            });
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        let inst0 = self.instantiate_container_method(&mdecl, tname, type_args);
                                        let inst = self.instantiate_assoc_method_from_args(&inst0, args, var_types);
                                        self.check_method_call_contract(
                                            &format!("{}.{}", inst_ty.display_name(), field),
                                            &inst,
                                            None,
                                            args,
                                            cur_fn,
                                            env,
                                            smt_int_vars,
                                            smt_premises,
                                            var_types,
                                            mutable_vars,
                                            active_mutate,
                                            expr.span,
                                        );
                                        lambdas_checked = true;
                                    } else {
                                        self.diagnostics.push(Diagnostic {
                                            code: "E0207".into(),
                                            message: format!(
                                                "unknown associated method `{}` on type `{}`",
                                                field, tname
                                            ),
                                            span: expr.span,
                                            notes: vec![],
                                            counterexample: BTreeMap::new(),
                                        });
                                    }
                                }
                            }
                        } else if let ExprKind::Ident(tname) = &target.kind {
                            if !var_types.contains_key(tname) && self.is_known_type_name(tname) {
                                // Bitfield `from_raw(val)` constructor
                                if field == "from_raw" && self.bitfields.contains_key(tname) {
                                    if args.len() != 1 {
                                        self.diagnostics.push(Diagnostic {
                                            code: "E0202".into(),
                                            message: format!(
                                                "`{}.from_raw` expects 1 argument, found {}",
                                                tname,
                                                args.len()
                                            ),
                                            span: expr.span,
                                            notes: vec![],
                                            counterexample: BTreeMap::new(),
                                        });
                                    }
                                    lambdas_checked = true;
                                } else if let Some(mdecl) = self.resolve_assoc_method_sig(tname, field) {
                                    let inst = self.instantiate_assoc_method_from_args(&mdecl, args, var_types);
                                    self.check_method_call_contract(
                                        &format!("{}.{}", tname, field),
                                        &inst,
                                        None,
                                        args,
                                        cur_fn,
                                        env,
                                        smt_int_vars,
                                        smt_premises,
                                        var_types,
                                        mutable_vars,
                                        active_mutate,
                                        expr.span,
                                    );
                                    lambdas_checked = true;
                                } else {
                                    self.diagnostics.push(Diagnostic {
                                        code: "E0207".into(),
                                        message: format!(
                                            "unknown associated method `{}` on type `{}`",
                                            field, tname
                                        ),
                                        span: expr.span,
                                        notes: vec![],
                                        counterexample: BTreeMap::new(),
                                    });
                                }
                            } else {
                                // Receiver method call on variable `target.field(args)` (or callable record field!)
                                self.check_expr(
                                    target,
                                    cur_fn,
                                    env,
                                    smt_int_vars,
                                    smt_premises,
                                    var_types,
                                    mutable_vars,
                                    active_mutate,
                                    saw_return,
                                );
                                let recv_ty = self.infer_expr_type(target, var_types);
                                if field == "raw"
                                    && matches!(self.resolve_base_type(&recv_ty), TypeExpr::Named { ref name, .. } if self.bitfields.contains_key(name))
                                {
                                    if !args.is_empty() {
                                        self.diagnostics.push(Diagnostic {
                                            code: "E0202".into(),
                                            message: "bitfield `.raw()` expects 0 arguments".into(),
                                            span: expr.span,
                                            notes: vec![],
                                            counterexample: BTreeMap::new(),
                                        });
                                    }
                                    lambdas_checked = true;
                                } else if let Some(mdecl) = self.resolve_method_sig(&recv_ty, field) {
                                    let inst = self.instantiate_method_from_args(&mdecl, true, args, var_types);
                                    self.check_method_call_contract(
                                        &format!("{}.{}", recv_ty.display_name(), field),
                                        &inst,
                                        Some(target),
                                        args,
                                        cur_fn,
                                        env,
                                        smt_int_vars,
                                        smt_premises,
                                        var_types,
                                        mutable_vars,
                                        active_mutate,
                                        expr.span,
                                    );
                                    lambdas_checked = true;
                                } else if let TypeExpr::Fn { params, .. } =
                                    self.resolve_base_type(&self.infer_expr_type(callee, var_types))
                                {
                                    self.check_indirect_fn_call(
                                        &format!("{}.{}", recv_ty.display_name(), field),
                                        &params,
                                        args,
                                        var_types,
                                        expr.span,
                                    );
                                    lambdas_checked = true;
                                } else if recv_ty.display_name() != "Any" {
                                    self.diagnostics.push(Diagnostic {
                                        code: "E0207".into(),
                                        message: format!(
                                            "unknown method `{}` on type `{}`",
                                            field,
                                            recv_ty.display_name()
                                        ),
                                        span: expr.span,
                                        notes: vec![],
                                        counterexample: BTreeMap::new(),
                                    });
                                }
                            }
                        } else {
                            // Receiver method call on arbitrary expression `(expr).field(args)`
                            self.check_expr(
                                target,
                                cur_fn,
                                env,
                                smt_int_vars,
                                smt_premises,
                                var_types,
                                mutable_vars,
                                active_mutate,
                                saw_return,
                            );
                            let recv_ty = self.infer_expr_type(target, var_types);
                            if let Some(mdecl) = self.resolve_method_sig(&recv_ty, field) {
                                let inst = self.instantiate_method_from_args(&mdecl, true, args, var_types);
                                self.check_method_call_contract(
                                    &format!("{}.{}", recv_ty.display_name(), field),
                                    &inst,
                                    Some(target),
                                    args,
                                    cur_fn,
                                    env,
                                    smt_int_vars,
                                    smt_premises,
                                    var_types,
                                    mutable_vars,
                                    active_mutate,
                                    expr.span,
                                );
                                lambdas_checked = true;
                            } else if let TypeExpr::Fn { params, .. } =
                                self.resolve_base_type(&self.infer_expr_type(callee, var_types))
                            {
                                self.check_indirect_fn_call(
                                    &format!("{}.{}", recv_ty.display_name(), field),
                                    &params,
                                    args,
                                    var_types,
                                    expr.span,
                                );
                                lambdas_checked = true;
                            } else if recv_ty.display_name() != "Any" {
                                self.diagnostics.push(Diagnostic {
                                    code: "E0207".into(),
                                    message: format!(
                                        "unknown method `{}` on type `{}`",
                                        field,
                                        recv_ty.display_name()
                                    ),
                                    span: expr.span,
                                    notes: vec![],
                                    counterexample: BTreeMap::new(),
                                });
                            }
                        }
                    }
                    ExprKind::GenericInst { expr: inner, type_args } => {
                        if let ExprKind::Ident(fname) = &inner.kind {
                            if let Some(fdecl) = self.all_fns.get(fname).cloned() {
                                let mut tsubst = BTreeMap::new();
                                for (tp, ta) in fdecl.type_params.iter().zip(type_args.iter()) {
                                    tsubst.insert(tp.clone(), ta.clone());
                                    if let Some(bounds) = fdecl.type_param_bounds.get(tp) {
                                        for b in bounds {
                                            if !self.type_implements(ta, b) {
                                                self.diagnostics.push(Diagnostic {
                                                    code: "E0505".into(),
                                                    message: format!(
                                                        "type `{}` does not satisfy interface bound `{}` for type parameter `{}` of `{}`",
                                                        ta.display_name(),
                                                        b,
                                                        tp,
                                                        fname
                                                    ),
                                                    span: expr.span,
                                                    notes: vec![format!("add `impl {} for {} {{ ... }}`", b, ta.display_name())],
                                                    counterexample: BTreeMap::new(),
                                                });
                                            }
                                        }
                                    }
                                }
                                let mut inst_fn = fdecl.clone();
                                for p in &mut inst_fn.params {
                                    p.ty = subst_type_expr(&p.ty, &tsubst);
                                }
                                if let Some(rt) = &inst_fn.ret_ty {
                                    inst_fn.ret_ty = Some(subst_type_expr(rt, &tsubst));
                                }
                                self.check_method_call_contract(
                                    fname,
                                    &inst_fn,
                                    None,
                                    args,
                                    cur_fn,
                                    env,
                                    smt_int_vars,
                                    smt_premises,
                                    var_types,
                                    mutable_vars,
                                    active_mutate,
                                    expr.span,
                                );
                                lambdas_checked = true;
                            }
                        }
                    }
                    ExprKind::Ident(fname) => {
                        if let Some(vty) = var_types.get(fname).cloned() {
                            if let TypeExpr::Fn { params, .. } = self.resolve_base_type(&vty) {
                                self.check_indirect_fn_call(fname, &params, args, var_types, expr.span);
                                lambdas_checked = true;
                            } else if let Some(fdecl) = self.all_fns.get(fname).cloned() {
                                self.check_direct_fn_call(
                                    fname,
                                    &fdecl,
                                    args,
                                    cur_fn,
                                    env,
                                    smt_int_vars,
                                    smt_premises,
                                    var_types,
                                    mutable_vars,
                                    active_mutate,
                                    expr.span,
                                );
                                lambdas_checked = true;
                            } else {
                                self.diagnostics.push(Diagnostic {
                                    code: "E0201".into(),
                                    message: format!(
                                        "`{}` of type `{}` is not callable",
                                        fname,
                                        vty.display_name()
                                    ),
                                    span: expr.span,
                                    notes: vec![],
                                    counterexample: BTreeMap::new(),
                                });
                            }
                        } else if let Some(fdecl) = self.all_fns.get(fname).cloned() {
                            if fdecl.is_unsafe && !self.in_unsafe {
                                self.diagnostics.push(Diagnostic {
                                    code: "E0800".into(),
                                    message: format!(
                                        "calling `unsafe fn {}` requires an `unsafe` block or `unsafe fn`",
                                        fname
                                    ),
                                    span: expr.span,
                                    notes: vec!["wrap the call in `unsafe { ... }`".into()],
                                    counterexample: BTreeMap::new(),
                                });
                            }
                            if self.cur_file_id != 0
                                && fdecl.span.file_id != 0
                                && self.cur_file_id != fdecl.span.file_id
                                && !fdecl.is_pub
                                && !fdecl.name.starts_with("__")
                            {
                                self.diagnostics.push(Diagnostic {
                                    code: "E0105".into(),
                                    message: format!(
                                        "private function `{}` cannot be accessed outside its defining module",
                                        fname
                                    ),
                                    span: expr.span,
                                    notes: vec![format!("mark `pub fn {}` to export it", fname)],
                                    counterexample: BTreeMap::new(),
                                });
                            }
                            self.check_direct_fn_call(
                                fname,
                                &fdecl,
                                args,
                                cur_fn,
                                env,
                                smt_int_vars,
                                smt_premises,
                                var_types,
                                mutable_vars,
                                active_mutate,
                                expr.span,
                            );
                            lambdas_checked = true;
                        } else if let Some(ef) = self.extern_fns.get(fname).cloned() {
                            if !self.in_unsafe {
                                self.diagnostics.push(Diagnostic {
                                    code: "E0800".into(),
                                    message: format!(
                                        "calling `extern \"{}\"` function `{}` requires an `unsafe` block",
                                        ef.abi, fname
                                    ),
                                    span: expr.span,
                                    notes: vec!["wrap C FFI calls in `unsafe { ... }`".into()],
                                    counterexample: BTreeMap::new(),
                                });
                            }
                            let arity_ok = if ef.is_variadic {
                                args.len() >= ef.params.len()
                            } else {
                                args.len() == ef.params.len()
                            };
                            if !arity_ok {
                                self.diagnostics.push(Diagnostic {
                                    code: "E0202".into(),
                                    message: format!(
                                        "extern function `{}` expects {} argument(s), found {}",
                                        fname,
                                        ef.params.len(),
                                        args.len()
                                    ),
                                    span: expr.span,
                                    notes: vec![],
                                    counterexample: BTreeMap::new(),
                                });
                            }
                            for (p, a) in ef.params.iter().zip(args.iter()) {
                                let actual = self.infer_expr_type_expected(&a.value, Some(&p.ty), var_types);
                                if !self.types_compatible(&p.ty, &actual) {
                                    self.diagnostics.push(Diagnostic {
                                        code: "E0205".into(),
                                        message: format!(
                                            "extern `{}` argument `{}` expects `{}`, found `{}`",
                                            fname,
                                            p.name,
                                            p.ty.display_name(),
                                            actual.display_name()
                                        ),
                                        span: a.value.span,
                                        notes: vec![],
                                        counterexample: BTreeMap::new(),
                                    });
                                }
                            }
                            lambdas_checked = true;
                        } else if let Some(vi) = self.enum_variants.get(fname).cloned() {
                            self.check_variant_ctor_args(fname, &vi, args, var_types, expr.span);
                        } else if let Some(b) = lookup_builtin(fname) {
                            if is_unsafe_builtin(fname) && !self.in_unsafe {
                                self.diagnostics.push(Diagnostic {
                                    code: "E0800".into(),
                                    message: format!(
                                        "calling unsafe memory/pointer builtin `{}` requires an `unsafe` block",
                                        fname
                                    ),
                                    span: expr.span,
                                    notes: vec!["wrap raw pointer/memory calls in `unsafe { ... }`".into()],
                                    counterexample: BTreeMap::new(),
                                });
                            }
                            if fname == "offset_of" && args.len() == 2 {
                                match (&args[0].value.kind, &args[1].value.kind) {
                                    (ExprKind::Ident(tname), ExprKind::Str(field_name)) => {
                                        if self.layout.offset_of(tname, field_name).is_none() {
                                            self.diagnostics.push(Diagnostic {
                                                code: "E0206".into(),
                                                message: format!(
                                                    "unknown field `{}` on type `{}` in `offset_of`",
                                                    field_name, tname
                                                ),
                                                span: expr.span,
                                                notes: vec![],
                                                counterexample: BTreeMap::new(),
                                            });
                                        }
                                    }
                                    _ => {
                                        self.diagnostics.push(Diagnostic {
                                            code: "E0205".into(),
                                            message: "`offset_of` expects `(TypeName, \"field_name\")`".into(),
                                            span: expr.span,
                                            notes: vec![],
                                            counterexample: BTreeMap::new(),
                                        });
                                    }
                                }
                            }
                            if fname == "push" && args.len() == 2 {
                                let list_ty = self.resolve_base_type(&self.infer_expr_type(&args[0].value, var_types));
                                if let TypeExpr::Named { name: ln, args: la, .. } = list_ty {
                                    if ln == "List" && la.len() == 1 {
                                        let item_ty = self.infer_expr_type(&args[1].value, var_types);
                                        if let TypeExpr::Dyn { interface, .. } = &la[0] {
                                            if !self.type_implements(&item_ty, interface) {
                                                self.diagnostics.push(Diagnostic {
                                                    code: "E0505".into(),
                                                    message: format!(
                                                        "cannot push `{}` into `List[dyn {}]`: `{}` does not implement `{}`",
                                                        item_ty.display_name(),
                                                        interface,
                                                        item_ty.display_name(),
                                                        interface
                                                    ),
                                                    span: args[1].value.span,
                                                    notes: vec![format!("add `impl {} for {} {{ ... }}`", interface, item_ty.display_name())],
                                                    counterexample: BTreeMap::new(),
                                                });
                                            }
                                        }
                                    }
                                }
                            }
                            if args.len() != b.arity {
                                self.diagnostics.push(Diagnostic {
                                    code: "E0202".into(),
                                    message: format!(
                                        "builtin `{}` expects {} argument(s), found {}",
                                        fname,
                                        b.arity,
                                        args.len()
                                    ),
                                    span: expr.span,
                                    notes: vec![],
                                    counterexample: BTreeMap::new(),
                                });
                            }
                            for (pt, a) in b.param_types.iter().zip(args.iter()) {
                                let expected = named_ty(pt);
                                let actual =
                                    self.infer_expr_type_expected(&a.value, Some(&expected), var_types);
                                if !self.types_compatible(&expected, &actual) {
                                    self.diagnostics.push(Diagnostic {
                                        code: "E0205".into(),
                                        message: format!(
                                            "builtin `{}` argument expects `{}`, found `{}`",
                                            fname,
                                            pt,
                                            actual.display_name()
                                        ),
                                        span: a.value.span,
                                        notes: vec![],
                                        counterexample: BTreeMap::new(),
                                    });
                                }
                            }
                        } else {
                            self.diagnostics.push(Diagnostic {
                                code: "E0201".into(),
                                message: format!("call to undefined function `{}`", fname),
                                span: expr.span,
                                notes: vec![],
                                counterexample: BTreeMap::new(),
                            });
                        }
                    }
                    _ => {
                        self.check_expr(
                            callee,
                            cur_fn,
                            env,
                            smt_int_vars,
                            smt_premises,
                            var_types,
                            mutable_vars,
                            active_mutate,
                            saw_return,
                        );
                        if let TypeExpr::Fn { params, .. } =
                            self.resolve_base_type(&self.infer_expr_type(callee, var_types))
                        {
                            self.check_indirect_fn_call("<fn>", &params, args, var_types, expr.span);
                            lambdas_checked = true;
                        }
                    }
                }
                if !lambdas_checked {
                    for a in args {
                        if let ExprKind::Lambda { params, ret_ty, body } = &a.value.kind {
                            self.check_lambda(
                                params,
                                ret_ty.as_ref(),
                                body,
                                None,
                                var_types,
                                a.value.span,
                            );
                        }
                    }
                }
            }
            ExprKind::If {
                cond,
                then_block,
                else_branch,
            } => {
                self.check_expr(
                    cond,
                    cur_fn,
                    env,
                    smt_int_vars,
                    smt_premises,
                    var_types,
                    mutable_vars,
                    active_mutate,
                    saw_return,
                );
                let mut then_env = env.clone();
                then_env.assume_expr(cond, None);
                let mut then_premises = smt_premises.clone();
                then_premises.push(((**cond).clone(), BTreeMap::new()));
                let mut then_vars = var_types.clone();
                let mut then_mut = mutable_vars.clone();
                let mut m = active_mutate.clone();
                self.check_block(
                    then_block,
                    cur_fn,
                    &mut then_env,
                    &mut smt_int_vars.clone(),
                    &mut then_premises,
                    &mut then_vars,
                    &mut then_mut,
                    &mut m,
                    saw_return,
                );
                if let Some(eb) = else_branch {
                    let neg_cond = Expr {
                        kind: ExprKind::Unary {
                            op: UnOp::Not,
                            expr: cond.clone(),
                        },
                        span: cond.span,
                    };
                    let mut else_env = env.clone();
                    else_env.assume_expr(&neg_cond, None);
                    let mut else_premises = smt_premises.clone();
                    else_premises.push((neg_cond, BTreeMap::new()));
                    self.check_expr(
                        eb,
                        cur_fn,
                        &mut else_env,
                        &mut smt_int_vars.clone(),
                        &mut else_premises,
                        var_types,
                        mutable_vars,
                        active_mutate,
                        saw_return,
                    );
                }
            }
            ExprKind::Match { scrutinee, arms } => {
                self.check_expr(
                    scrutinee,
                    cur_fn,
                    env,
                    smt_int_vars,
                    smt_premises,
                    var_types,
                    mutable_vars,
                    active_mutate,
                    saw_return,
                );
                let st = self.infer_expr_type(scrutinee, var_types);
                self.check_match_exhaustiveness(&st, arms, expr.span);
                for arm in arms {
                    let mut arm_vars = var_types.clone();
                    let mut arm_env = env.clone();
                    let mut arm_smt_vars = smt_int_vars.clone();
                    let mut arm_premises = smt_premises.clone();
                    self.bind_match_arm_types(&st, arm, &mut arm_vars);
                    // Assume type facts for all bound pattern variables (including nested & tuple patterns)
                    let mut bound_names = BTreeSet::new();
                    for p in &arm.patterns {
                        collect_pattern_bindings(p, &mut bound_names);
                    }
                    for bname in &bound_names {
                        if let Some(bty) = arm_vars.get(bname).cloned() {
                            self.assume_type_facts(
                                bname,
                                &bty,
                                &mut arm_env,
                                &mut arm_smt_vars,
                                &mut arm_premises,
                            );
                        }
                    }
                    if let Some(g) = &arm.guard {
                        self.check_expr(
                            g,
                            cur_fn,
                            &mut arm_env,
                            &mut arm_smt_vars,
                            &mut arm_premises,
                            &arm_vars,
                            mutable_vars,
                            active_mutate,
                            saw_return,
                        );
                        arm_env.assume_expr(g, None);
                        arm_premises.push((g.clone(), BTreeMap::new()));
                    }
                    self.check_expr(
                        &arm.body,
                        cur_fn,
                        &mut arm_env,
                        &mut arm_smt_vars,
                        &mut arm_premises,
                        &arm_vars,
                        mutable_vars,
                        active_mutate,
                        saw_return,
                    );
                }
            }
            ExprKind::Block(blk) => {
                let mut b_env = env.clone();
                let mut b_vars = var_types.clone();
                let mut b_mut = mutable_vars.clone();
                let mut m = active_mutate.clone();
                self.check_block(
                    blk,
                    cur_fn,
                    &mut b_env,
                    &mut smt_int_vars.clone(),
                    &mut smt_premises.clone(),
                    &mut b_vars,
                    &mut b_mut,
                    &mut m,
                    saw_return,
                );
            }
            _ => {}
        }
    }

    fn patterns_exhaust_type(&self, ty: &TypeExpr, pats: &[&Pattern]) -> bool {
        if pats.is_empty() {
            return false;
        }
        if pats
            .iter()
            .any(|p| matches!(p, Pattern::Wildcard(_) | Pattern::Ident(_, _)))
        {
            return true;
        }
        let base = self.resolve_base_type(ty);
        match &base {
            TypeExpr::Tuple(elem_tys, _) => {
                let tuple_rows: Vec<&[Pattern]> = pats
                    .iter()
                    .filter_map(|p| match p {
                        Pattern::Tuple(ps, _) if ps.len() == elem_tys.len() => Some(ps.as_slice()),
                        _ => None,
                    })
                    .collect();
                if tuple_rows.is_empty() {
                    return false;
                }
                if tuple_rows
                    .iter()
                    .any(|row| row.iter().all(pattern_is_irrefutable))
                {
                    return true;
                }
                for col_idx in 0..elem_tys.len() {
                    let col: Vec<&Pattern> = tuple_rows
                        .iter()
                        .filter(|r| {
                            r.iter()
                                .enumerate()
                                .all(|(j, sp)| j == col_idx || pattern_is_irrefutable(sp))
                        })
                        .map(|r| &r[col_idx])
                        .collect();
                    if self.patterns_exhaust_type(&elem_tys[col_idx], &col) {
                        return true;
                    }
                }
                false
            }
            TypeExpr::Named { name: tname, .. } => {
                let required: Option<Vec<String>> = if tname == "Option" {
                    Some(vec!["Some".into(), "None".into()])
                } else if tname == "Result" {
                    Some(vec!["Ok".into(), "Err".into()])
                } else if tname == "Bool" {
                    Some(vec!["true".into(), "false".into()])
                } else if let Some(ed) = self.enums.get(tname) {
                    Some(ed.variants.iter().map(|v| v.name.clone()).collect())
                } else {
                    None
                };
                let Some(req_variants) = required else {
                    return false;
                };
                req_variants
                    .iter()
                    .all(|vname| self.variant_covered_by(&base, tname, vname, pats))
            }
            _ => false,
        }
    }

    fn variant_covered_by(
        &self,
        scrutinee_base: &TypeExpr,
        tname: &str,
        vname: &str,
        pats: &[&Pattern],
    ) -> bool {
        if tname == "Bool" {
            let target = vname == "true";
            return pats
                .iter()
                .any(|p| matches!(p, Pattern::Bool(b, _) if *b == target));
        }
        let mut multi_rows: Vec<&[Pattern]> = Vec::new();
        for p in pats {
            match p {
                Pattern::Variant { name, .. } => {
                    let short = name.rsplit('.').next().unwrap_or(name);
                    if short == vname {
                        return true;
                    }
                }
                Pattern::VariantMulti { name, patterns, .. } => {
                    let short = name.rsplit('.').next().unwrap_or(name);
                    if short == vname {
                        if patterns.iter().all(pattern_is_irrefutable) {
                            return true;
                        }
                        multi_rows.push(patterns.as_slice());
                    }
                }
                _ => {}
            }
        }
        if multi_rows.is_empty() {
            return false;
        }
        let ptys = self.resolve_variant_payload_tys(scrutinee_base, vname);
        if ptys.is_empty() {
            return true;
        }
        if ptys.len() == 1 {
            let sub: Vec<&Pattern> = multi_rows.iter().filter_map(|r| r.first()).collect();
            return self.patterns_exhaust_type(&ptys[0], &sub);
        }
        for col_idx in 0..ptys.len() {
            let col: Vec<&Pattern> = multi_rows
                .iter()
                .filter(|r| {
                    r.len() == ptys.len()
                        && r.iter()
                            .enumerate()
                            .all(|(j, sp)| j == col_idx || pattern_is_irrefutable(sp))
                })
                .map(|r| &r[col_idx])
                .collect();
            if self.patterns_exhaust_type(&ptys[col_idx], &col) {
                return true;
            }
        }
        false
    }

    fn check_match_exhaustiveness(&mut self, scrutinee_ty: &TypeExpr, arms: &[MatchArm], span: Span) {
        let base = self.resolve_base_type(scrutinee_ty);
        let TypeExpr::Named { name: tname, .. } = &base else {
            return;
        };

        let required: Option<Vec<String>> = if tname == "Option" {
            Some(vec!["Some".into(), "None".into()])
        } else if tname == "Result" {
            Some(vec!["Ok".into(), "Err".into()])
        } else if tname == "Bool" {
            Some(vec!["true".into(), "false".into()])
        } else if let Some(ed) = self.enums.get(tname) {
            Some(ed.variants.iter().map(|v| v.name.clone()).collect())
        } else {
            None
        };

        let Some(req_variants) = required else {
            return;
        };

        let unguarded: Vec<&Pattern> = arms
            .iter()
            .filter(|a| a.guard.is_none())
            .flat_map(|a| a.patterns.iter())
            .collect();

        let has_wildcard = unguarded.iter().any(|p| match p {
            Pattern::Wildcard(_) | Pattern::Ident(_, _) => true,
            Pattern::Tuple(ps, _) => ps.iter().all(pattern_is_irrefutable),
            _ => false,
        });

        if has_wildcard {
            self.proofs.push(ProofRecord {
                obligation: format!("exhaustive match on `{}` (wildcard)", tname),
                tier: ProofTier::L0Fast,
                span,
            });
            return;
        }

        let missing: Vec<String> = req_variants
            .into_iter()
            .filter(|v| !self.variant_covered_by(&base, tname, v, &unguarded))
            .collect();
        if missing.is_empty() {
            self.proofs.push(ProofRecord {
                obligation: format!("exhaustive match on `{}`", tname),
                tier: ProofTier::L0Fast,
                span,
            });
        } else {
            self.diagnostics.push(Diagnostic {
                code: "E0301".into(),
                message: format!(
                    "non-exhaustive `match` on `{}`: missing variant(s) {}",
                    tname,
                    missing.join(", ")
                ),
                span,
                notes: vec!["handle all variants explicitly or add a wildcard `_ => ...` arm".into()],
                counterexample: BTreeMap::new(),
            });
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn check_predicate_obligation(
        &mut self,
        pred: &Expr,
        self_val_expr: Option<&Expr>,
        env: &L0Env,
        smt_int_vars: &BTreeSet<String>,
        smt_premises: &[(Expr, BTreeMap<String, String>)],
        context: &str,
        fail_code: &str,
        unknown_code: &str,
        span: Span,
    ) {
        let mut check_env = env.clone();
        let self_var_name = if let Some(se) = self_val_expr {
            let abs = env.eval_abs_int(se, None);
            check_env.set_int("__val", abs);
            Some("__val")
        } else {
            None
        };

        match check_env.check_predicate(pred, self_var_name) {
            L0Result::Proved => {
                self.proofs.push(ProofRecord {
                    obligation: context.to_string(),
                    tier: ProofTier::L0Fast,
                    span,
                });
            }
            L0Result::Disproved(reason) => {
                self.diagnostics.push(Diagnostic {
                    code: fail_code.into(),
                    message: format!("{} violated", context),
                    span,
                    notes: vec![reason],
                    counterexample: BTreeMap::new(),
                });
            }
            L0Result::Unknown => {
                let mut q = SmtQuery::new(self.smt_pure_fns.clone());
                for v in smt_int_vars {
                    q.declare_int(v.clone());
                }
                let mut subst = BTreeMap::new();
                if let Some(se) = self_val_expr {
                    q.declare_int("__val");
                    subst.insert("self".into(), "__val".into());
                    let eq = Expr {
                        kind: ExprKind::Binary {
                            op: BinOp::Eq,
                            left: Box::new(Expr {
                                kind: ExprKind::Ident("__val".into()),
                                span,
                            }),
                            right: Box::new(se.clone()),
                        },
                        span,
                    };
                    if q.add_premise_expr(&eq, &BTreeMap::new()).is_err() {
                        self.diagnostics.push(Diagnostic {
                            code: unknown_code.into(),
                            message: format!("unproved {} (Fail-Closed)", context),
                            span,
                            notes: vec!["expression outside supported SMT fragment".into()],
                            counterexample: BTreeMap::new(),
                        });
                        return;
                    }
                }
                for (p, s) in smt_premises {
                    let _ = q.add_premise_expr(p, s);
                }
                if let Err(e) = q.set_goal_expr(pred, &subst) {
                    self.diagnostics.push(Diagnostic {
                        code: unknown_code.into(),
                        message: format!("unproved {} (Fail-Closed)", context),
                        span,
                        notes: vec![e],
                        counterexample: BTreeMap::new(),
                    });
                    return;
                }
                match q.solve() {
                    SmtOutcome::Proved => self.proofs.push(ProofRecord {
                        obligation: context.to_string(),
                        tier: ProofTier::L1Smt,
                        span,
                    }),
                    SmtOutcome::Counterexample(ce) => self.diagnostics.push(Diagnostic {
                        code: fail_code.into(),
                        message: format!("{} may be violated", context),
                        span,
                        notes: vec!["counterexample found by L1 SMT (Z3)".into()],
                        counterexample: ce,
                    }),
                    SmtOutcome::Unknown(msg) => self.diagnostics.push(Diagnostic {
                        code: unknown_code.into(),
                        message: format!("unproved {} (Fail-Closed)", context),
                        span,
                        notes: vec![msg],
                        counterexample: BTreeMap::new(),
                    }),
                }
            }
        }
    }

    fn check_record_construction(
        &mut self,
        rec: &RecordDecl,
        provided: &[(String, Expr)],
        env: &L0Env,
        smt_int_vars: &BTreeSet<String>,
        smt_premises: &[(Expr, BTreeMap<String, String>)],
        span: Span,
    ) {
        let mut local_env = env.clone();
        let mut local_vars = smt_int_vars.clone();
        let mut local_premises = smt_premises.to_vec();
        let mut field_exprs: BTreeMap<String, Expr> = BTreeMap::new();

        for (fname, fexpr) in provided {
            field_exprs.insert(fname.clone(), fexpr.clone());
        }

        for f in &rec.fields {
            let Some(val_expr) = field_exprs.get(&f.name) else {
                continue;
            };
            let abs = local_env.eval_abs_int(val_expr, None);
            local_env.set_int(f.name.clone(), abs);
            if let ExprKind::Ident(vname) = &val_expr.kind {
                let extra: Vec<String> = local_env
                    .proved_calls
                    .iter()
                    .filter_map(|c| {
                        let pat = format!("({})", vname);
                        if c.ends_with(&pat) {
                            Some(format!("{}({})", &c[..c.len() - pat.len()], f.name))
                        } else {
                            None
                        }
                    })
                    .collect();
                for e in extra {
                    local_env.proved_calls.insert(e);
                }
            }
            local_vars.insert(f.name.clone());
            let eq = Expr {
                kind: ExprKind::Binary {
                    op: BinOp::Eq,
                    left: Box::new(Expr {
                        kind: ExprKind::Ident(f.name.clone()),
                        span: val_expr.span,
                    }),
                    right: Box::new(val_expr.clone()),
                },
                span: val_expr.span,
            };
            local_premises.push((eq, BTreeMap::new()));

            self.check_expr_satisfies_type(
                val_expr,
                &f.ty,
                &local_env,
                &local_vars,
                &local_premises,
                &format!("field `{}` of `{}`", f.name, rec.name),
                val_expr.span,
            );

            if let Some(pred) = &f.predicate {
                match local_env.check_predicate(pred, Some(&f.name)) {
                    L0Result::Proved => {
                        self.proofs.push(ProofRecord {
                            obligation: format!("record `{}` field `{}` refinement", rec.name, f.name),
                            tier: ProofTier::L0Fast,
                            span: val_expr.span,
                        });
                    }
                    L0Result::Disproved(reason) => {
                        self.diagnostics.push(Diagnostic {
                            code: "E0412".into(),
                            message: format!(
                                "refinement constraint not satisfied for field `{}` of `{}`",
                                f.name, rec.name
                            ),
                            span: val_expr.span,
                            notes: vec![
                                reason,
                                format!("hint: use `{}.parse(...)` for unchecked runtime inputs", rec.name),
                            ],
                            counterexample: BTreeMap::new(),
                        });
                    }
                    L0Result::Unknown => {
                        let mut q = SmtQuery::new(self.smt_pure_fns.clone());
                        for v in &local_vars {
                            q.declare_int(v.clone());
                        }
                        for (p, s) in &local_premises {
                            let _ = q.add_premise_expr(p, s);
                        }
                        let mut subst = BTreeMap::new();
                        subst.insert("self".into(), f.name.clone());
                        if let Err(e) = q.set_goal_expr(pred, &subst) {
                            self.diagnostics.push(Diagnostic {
                                code: "E0413".into(),
                                message: format!(
                                    "cannot statically prove refinement for `{}.{}` (Fail-Closed)",
                                    rec.name, f.name
                                ),
                                span: val_expr.span,
                                notes: vec![e],
                                counterexample: BTreeMap::new(),
                            });
                            continue;
                        }
                        match q.solve() {
                            SmtOutcome::Proved => {
                                self.proofs.push(ProofRecord {
                                    obligation: format!(
                                        "record `{}` field `{}` refinement",
                                        rec.name, f.name
                                    ),
                                    tier: ProofTier::L1Smt,
                                    span: val_expr.span,
                                });
                            }
                            SmtOutcome::Counterexample(ce) => {
                                self.diagnostics.push(Diagnostic {
                                    code: "E0412".into(),
                                    message: format!(
                                        "refinement constraint not satisfied for field `{}` of `{}`",
                                        f.name, rec.name
                                    ),
                                    span: val_expr.span,
                                    notes: vec![
                                        "counterexample found by L1 SMT (Z3)".into(),
                                        format!("hint: use `{}.parse(...)` for unchecked runtime inputs", rec.name),
                                    ],
                                    counterexample: ce,
                                });
                            }
                            SmtOutcome::Unknown(msg) => {
                                self.diagnostics.push(Diagnostic {
                                    code: "E0413".into(),
                                    message: format!(
                                        "unproved refinement for `{}.{}` (Fail-Closed)",
                                        rec.name, f.name
                                    ),
                                    span: val_expr.span,
                                    notes: vec![msg],
                                    counterexample: BTreeMap::new(),
                                });
                            }
                        }
                    }
                }
            }
        }

        for wc in &rec.where_clauses {
            let rewritten = substitute_self_fields(wc);
            match local_env.check_predicate(&rewritten, None) {
                L0Result::Proved => {
                    self.proofs.push(ProofRecord {
                        obligation: format!("record `{}` invariant", rec.name),
                        tier: ProofTier::L0Fast,
                        span,
                    });
                }
                L0Result::Disproved(reason) => {
                    self.diagnostics.push(Diagnostic {
                        code: "E0414".into(),
                        message: format!("record invariant not satisfied for `{}`", rec.name),
                        span,
                        notes: vec![reason],
                        counterexample: BTreeMap::new(),
                    });
                }
                L0Result::Unknown => {
                    let mut q = SmtQuery::new(self.smt_pure_fns.clone());
                    for v in &local_vars {
                        q.declare_int(v.clone());
                    }
                    for (p, s) in &local_premises {
                        let _ = q.add_premise_expr(p, s);
                    }
                    if q.set_goal_expr(&rewritten, &BTreeMap::new()).is_ok() {
                        match q.solve() {
                            SmtOutcome::Proved => self.proofs.push(ProofRecord {
                                obligation: format!("record `{}` invariant", rec.name),
                                tier: ProofTier::L1Smt,
                                span,
                            }),
                            SmtOutcome::Counterexample(ce) => self.diagnostics.push(Diagnostic {
                                code: "E0414".into(),
                                message: format!("record invariant not satisfied for `{}`", rec.name),
                                span,
                                notes: vec!["counterexample found by L1 SMT (Z3)".into()],
                                counterexample: ce,
                            }),
                            SmtOutcome::Unknown(msg) => self.diagnostics.push(Diagnostic {
                                code: "E0415".into(),
                                message: format!("unproved record invariant for `{}` (Fail-Closed)", rec.name),
                                span,
                                notes: vec![msg],
                                counterexample: BTreeMap::new(),
                            }),
                        }
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn check_method_call_contract(
        &mut self,
        label: &str,
        mdecl: &FnDecl,
        recv: Option<&Expr>,
        args: &[CallArg],
        cur_fn: &FnDecl,
        env: &mut L0Env,
        smt_int_vars: &mut BTreeSet<String>,
        smt_premises: &mut Vec<(Expr, BTreeMap<String, String>)>,
        var_types: &BTreeMap<String, TypeExpr>,
        mutable_vars: &BTreeSet<String>,
        active_mutate: &Option<String>,
        span: Span,
    ) {
        if cur_fn.is_pure && !mdecl.is_pure {
            self.diagnostics.push(Diagnostic {
                code: "E0610".into(),
                message: format!(
                    "purity violation: `pure fn {}` cannot call non-pure method `{}`",
                    cur_fn.name, label
                ),
                span,
                notes: vec![],
                counterexample: BTreeMap::new(),
            });
        }
        if cur_fn.attrs.iter().any(|a| a.name == "no_heap_alloc")
            && !mdecl.attrs.iter().any(|a| a.name == "no_heap_alloc")
        {
            self.diagnostics.push(Diagnostic {
                code: "E0701".into(),
                message: format!(
                    "`@no_heap_alloc` contract violated in `{}`: method `{}` is not `@no_heap_alloc`",
                    cur_fn.name, label
                ),
                span,
                notes: vec![],
                counterexample: BTreeMap::new(),
            });
        }

        let has_self = recv.is_some() && mdecl.params.first().is_some_and(|p| p.name == "self");
        let explicit_params = if has_self {
            &mdecl.params[1..]
        } else {
            &mdecl.params[..]
        };

        if has_self {
            if let Some(r_expr) = recv {
                if mdecl.params[0].is_inout {
                    let root_var = match &r_expr.kind {
                        ExprKind::Ident(v) => Some(v.as_str()),
                        ExprKind::Field { target, .. } => match &target.kind {
                            ExprKind::Ident(v) => Some(v.as_str()),
                            _ => None,
                        },
                        _ => None,
                    };
                    if let Some(v) = root_var {
                        if !mutable_vars.contains(v) && var_types.contains_key(v) {
                            self.diagnostics.push(Diagnostic {
                                code: "E0203".into(),
                                message: format!(
                                    "cannot call `inout self` method `{}` on immutable binding `{}`",
                                    label, v
                                ),
                                span: r_expr.span,
                                notes: vec![format!("declare `{}` with `var` or `inout`", v)],
                                counterexample: BTreeMap::new(),
                            });
                        }
                    }
                }
            }
        }

        if args.len() != explicit_params.len() {
            self.diagnostics.push(Diagnostic {
                code: "E0202".into(),
                message: format!(
                    "method `{}` expects {} argument(s), found {}",
                    label,
                    explicit_params.len(),
                    args.len()
                ),
                span,
                notes: vec![],
                counterexample: BTreeMap::new(),
            });
        }

        let mut param_arg_map: BTreeMap<String, Expr> = BTreeMap::new();
        if let Some(r_expr) = recv {
            param_arg_map.insert("self".into(), r_expr.clone());
        }
        for (p, a) in explicit_params.iter().zip(args.iter()) {
            param_arg_map.insert(p.name.clone(), a.value.clone());
            if let ExprKind::Lambda { params, ret_ty, body } = &a.value.kind {
                self.check_lambda(params, ret_ty.as_ref(), body, Some(&p.ty), var_types, a.value.span);
            }
            let actual_ty = self.infer_expr_type_expected(&a.value, Some(&p.ty), var_types);
            if let TypeExpr::Dyn { interface, .. } = &p.ty {
                if !self.type_implements(&actual_ty, interface) {
                    self.diagnostics.push(Diagnostic {
                        code: "E0505".into(),
                        message: format!(
                            "argument `{}` of `{}`: type `{}` does not implement interface `{}`",
                            p.name,
                            label,
                            actual_ty.display_name(),
                            interface
                        ),
                        span: a.value.span,
                        notes: vec![],
                        counterexample: BTreeMap::new(),
                    });
                }
            } else if !self.types_compatible(&p.ty, &actual_ty) {
                self.diagnostics.push(Diagnostic {
                    code: "E0205".into(),
                    message: format!(
                        "type mismatch for argument `{}` of `{}`: expected `{}`, found `{}`",
                        p.name,
                        label,
                        p.ty.display_name(),
                        actual_ty.display_name()
                    ),
                    span: a.value.span,
                    notes: vec![],
                    counterexample: BTreeMap::new(),
                });
            }
            if p.is_inout {
                if let ExprKind::Ident(v) = &a.value.kind {
                    if active_mutate.as_deref() == Some(v.as_str()) {
                        self.diagnostics.push(Diagnostic {
                            code: "E0502".into(),
                            message: format!(
                                "cannot pass `{}` (currently unsealed inside `mutate {}`) to `{}`",
                                v, v, label
                            ),
                            span: a.value.span,
                            notes: vec![],
                            counterexample: BTreeMap::new(),
                        });
                    }
                    if !mutable_vars.contains(v) && var_types.contains_key(v) {
                        self.diagnostics.push(Diagnostic {
                            code: "E0203".into(),
                            message: format!(
                                "cannot pass immutable binding `{}` to `inout` parameter `{}` of `{}`",
                                v, p.name, label
                            ),
                            span: a.value.span,
                            notes: vec![format!("declare `{}` with `var`", v)],
                            counterexample: BTreeMap::new(),
                        });
                    }
                }
            }
            self.check_expr_satisfies_type(
                &a.value,
                &p.ty,
                env,
                smt_int_vars,
                smt_premises,
                &format!("argument `{}` of `{}`", p.name, label),
                a.value.span,
            );
            if let Some(pred) = &p.predicate {
                self.check_predicate_obligation(
                    pred,
                    Some(&a.value),
                    env,
                    smt_int_vars,
                    smt_premises,
                    &format!("parameter `{}` predicate of `{}`", p.name, label),
                    "E0440",
                    "E0441",
                    a.value.span,
                );
            }
        }
        for req in &mdecl.requires {
            let instantiated = substitute_expr_idents(req, &param_arg_map);
            self.check_predicate_obligation(
                &instantiated,
                None,
                env,
                smt_int_vars,
                smt_premises,
                &format!("precondition `requires` of `{}`", label),
                "E0440",
                "E0441",
                span,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn check_direct_fn_call(
        &mut self,
        fname: &str,
        fdecl: &FnDecl,
        args: &[CallArg],
        cur_fn: &FnDecl,
        env: &mut L0Env,
        smt_int_vars: &mut BTreeSet<String>,
        smt_premises: &mut Vec<(Expr, BTreeMap<String, String>)>,
        var_types: &BTreeMap<String, TypeExpr>,
        mutable_vars: &BTreeSet<String>,
        active_mutate: &Option<String>,
        span: Span,
    ) {
        // Infer generic type parameter bindings (two-pass: non-lambda then lambda) & verify interface bounds!
        let mut inferred_tp: BTreeMap<String, TypeExpr> = BTreeMap::new();
        if !fdecl.type_params.is_empty() {
            for (p, a) in fdecl.params.iter().zip(args.iter()) {
                if !matches!(a.value.kind, ExprKind::Lambda { .. }) {
                    let aty = self.infer_expr_type(&a.value, var_types);
                    Self::unify_type_pattern(&p.ty, &aty, &fdecl.type_params, &mut inferred_tp);
                }
            }
            for (p, a) in fdecl.params.iter().zip(args.iter()) {
                if matches!(a.value.kind, ExprKind::Lambda { .. }) {
                    let expected_pty = subst_type_expr(&p.ty, &inferred_tp);
                    let aty = self.infer_expr_type_expected(&a.value, Some(&expected_pty), var_types);
                    Self::unify_type_pattern(&p.ty, &aty, &fdecl.type_params, &mut inferred_tp);
                }
            }
            for (tp, concrete_ty) in &inferred_tp {
                if let Some(bounds) = fdecl.type_param_bounds.get(tp) {
                    for b in bounds {
                        if !self.type_implements(concrete_ty, b) {
                            self.diagnostics.push(Diagnostic {
                                code: "E0505".into(),
                                message: format!(
                                    "type `{}` does not implement interface `{}` required by `{}`",
                                    concrete_ty.display_name(),
                                    b,
                                    fname
                                ),
                                span,
                                notes: vec![format!("add `impl {} for {} {{ ... }}`", b, concrete_ty.display_name())],
                                counterexample: BTreeMap::new(),
                            });
                        }
                    }
                }
            }
        }
        if args.len() != fdecl.params.len() {
            self.diagnostics.push(Diagnostic {
                code: "E0202".into(),
                message: format!(
                    "function `{}` expects {} argument(s), found {}",
                    fname,
                    fdecl.params.len(),
                    args.len()
                ),
                span,
                notes: vec![],
                counterexample: BTreeMap::new(),
            });
        }
        let mut inout_vars_seen = BTreeSet::new();
        let mut param_arg_map: BTreeMap<String, Expr> = BTreeMap::new();
        let mut mutated_inout_targets: Vec<(String, TypeExpr)> = Vec::new();
        for (p, a) in fdecl.params.iter().zip(args.iter()) {
            param_arg_map.insert(p.name.clone(), a.value.clone());
            let expected_pty = if inferred_tp.is_empty() {
                p.ty.clone()
            } else {
                subst_type_expr(&p.ty, &inferred_tp)
            };
            if let ExprKind::Lambda { params, ret_ty, body } = &a.value.kind {
                self.check_lambda(params, ret_ty.as_ref(), body, Some(&expected_pty), var_types, a.value.span);
            }
            let actual_ty = self.infer_expr_type_expected(&a.value, Some(&expected_pty), var_types);
            if let TypeExpr::Dyn { interface, .. } = &expected_pty {
                if !self.type_implements(&actual_ty, interface) {
                    self.diagnostics.push(Diagnostic {
                        code: "E0505".into(),
                        message: format!(
                            "argument `{}` of `{}`: type `{}` does not implement interface `{}`",
                            p.name,
                            fname,
                            actual_ty.display_name(),
                            interface
                        ),
                        span: a.value.span,
                        notes: vec![format!("add `impl {} for {} {{ ... }}`", interface, actual_ty.display_name())],
                        counterexample: BTreeMap::new(),
                    });
                } else if !matches!(actual_ty, TypeExpr::Dyn { .. })
                    && cur_fn.attrs.iter().any(|at| at.name == "no_heap_alloc")
                {
                    self.diagnostics.push(Diagnostic {
                        code: "E0701".into(),
                        message: format!(
                            "`@no_heap_alloc` contract violated in `{}`: boxing `{}` into `dyn {}` allocates",
                            cur_fn.name,
                            actual_ty.display_name(),
                            interface
                        ),
                        span: a.value.span,
                        notes: vec![],
                        counterexample: BTreeMap::new(),
                    });
                }
            } else if !self.types_compatible(&expected_pty, &actual_ty) {
                self.diagnostics.push(Diagnostic {
                    code: "E0205".into(),
                    message: format!(
                        "type mismatch for argument `{}` of `{}`: expected `{}`, found `{}`",
                        p.name,
                        fname,
                        expected_pty.display_name(),
                        actual_ty.display_name()
                    ),
                    span: a.value.span,
                    notes: vec![],
                    counterexample: BTreeMap::new(),
                });
            }
            if p.is_inout {
                if let ExprKind::Ident(v) = &a.value.kind {
                    if !inout_vars_seen.insert(v.clone()) {
                        self.diagnostics.push(Diagnostic {
                            code: "E0501".into(),
                            message: format!(
                                "MVS exclusivity violation: variable `{}` passed to multiple `inout` parameters of `{}`",
                                v, fname
                            ),
                            span: a.value.span,
                            notes: vec!["Tenon forbids overlapping mutable aliases".into()],
                            counterexample: BTreeMap::new(),
                        });
                    }
                    if active_mutate.as_deref() == Some(v.as_str()) {
                        self.diagnostics.push(Diagnostic {
                            code: "E0502".into(),
                            message: format!(
                                "cannot pass `{}` (currently unsealed inside `mutate {}`) to `{}`",
                                v, v, fname
                            ),
                            span: a.value.span,
                            notes: vec!["record invariant is temporarily suspended inside `mutate`".into()],
                            counterexample: BTreeMap::new(),
                        });
                    }
                    if !mutable_vars.contains(v) && var_types.contains_key(v) {
                        self.diagnostics.push(Diagnostic {
                            code: "E0203".into(),
                            message: format!(
                                "cannot pass immutable binding `{}` to `inout` parameter `{}` of `{}`",
                                v, p.name, fname
                            ),
                            span: a.value.span,
                            notes: vec![format!("declare `{}` with `var`", v)],
                            counterexample: BTreeMap::new(),
                        });
                    }
                    mutated_inout_targets.push((v.clone(), p.ty.clone()));
                }
            }
            self.check_expr_satisfies_type(
                &a.value,
                &p.ty,
                env,
                smt_int_vars,
                smt_premises,
                &format!("argument `{}` of `{}`", p.name, fname),
                a.value.span,
            );
            if let Some(pred) = &p.predicate {
                self.check_predicate_obligation(
                    pred,
                    Some(&a.value),
                    env,
                    smt_int_vars,
                    smt_premises,
                    &format!("parameter `{}` predicate of `{}`", p.name, fname),
                    "E0440",
                    "E0441",
                    a.value.span,
                );
            }
        }
        for req in &fdecl.requires {
            let instantiated = substitute_expr_idents(req, &param_arg_map);
            self.check_predicate_obligation(
                &instantiated,
                None,
                env,
                smt_int_vars,
                smt_premises,
                &format!("precondition `requires` of `{}`", fname),
                "E0440",
                "E0441",
                span,
            );
        }
        for (v, pty) in mutated_inout_targets {
            let prefix = format!("{}.", v);
            env.ints.retain(|k, _| k != &v && !k.starts_with(&prefix));
            env.bools.retain(|k, _| k != &v && !k.starts_with(&prefix));
            smt_premises.retain(|(p, _)| !expr_mentions_ident(p, &v));
            self.assume_type_facts(&v, &pty, env, smt_int_vars, smt_premises);
        }
    }

    fn check_indirect_fn_call(
        &mut self,
        label: &str,
        expected_params: &[TypeExpr],
        args: &[CallArg],
        var_types: &BTreeMap<String, TypeExpr>,
        span: Span,
    ) {
        if args.len() != expected_params.len() {
            self.diagnostics.push(Diagnostic {
                code: "E0202".into(),
                message: format!(
                    "function value `{}` expects {} argument(s), found {}",
                    label,
                    expected_params.len(),
                    args.len()
                ),
                span,
                notes: vec![],
                counterexample: BTreeMap::new(),
            });
        }
        for (ep, a) in expected_params.iter().zip(args.iter()) {
            if let ExprKind::Lambda { params, ret_ty, body } = &a.value.kind {
                self.check_lambda(params, ret_ty.as_ref(), body, Some(ep), var_types, a.value.span);
            }
            let actual = self.infer_expr_type_expected(&a.value, Some(ep), var_types);
            if !self.types_compatible(ep, &actual) {
                self.diagnostics.push(Diagnostic {
                    code: "E0205".into(),
                    message: format!(
                        "type mismatch calling `{}`: expected `{}`, found `{}`",
                        label,
                        ep.display_name(),
                        actual.display_name()
                    ),
                    span: a.value.span,
                    notes: vec![],
                    counterexample: BTreeMap::new(),
                });
            }
        }
    }

    fn check_variant_ctor_args(
        &mut self,
        label: &str,
        vi: &VariantInfo,
        args: &[CallArg],
        var_types: &BTreeMap<String, TypeExpr>,
        span: Span,
    ) {
        if args.len() != vi.payload_tys.len() {
            self.diagnostics.push(Diagnostic {
                code: "E0202".into(),
                message: format!(
                    "variant `{}` expects {} argument(s), found {}",
                    label,
                    vi.payload_tys.len(),
                    args.len()
                ),
                span,
                notes: vec![],
                counterexample: BTreeMap::new(),
            });
        }
        let mut esubst = BTreeMap::new();
        if let Some(ed) = self.enums.get(&vi.enum_name).cloned() {
            if !ed.type_params.is_empty() {
                for (pty, a) in vi.payload_tys.iter().zip(args.iter()) {
                    let aty = self.infer_expr_type(&a.value, var_types);
                    Self::unify_type_pattern(pty, &aty, &ed.type_params, &mut esubst);
                }
                for (tp, ta) in &esubst {
                    if ta.display_name() != "Any" && !self.cur_type_params.contains(&ta.display_name()) {
                        if let Some(bounds) = ed.type_param_bounds.get(tp) {
                            for b in bounds {
                                if !self.type_implements(ta, b) {
                                    self.diagnostics.push(Diagnostic {
                                        code: "E0505".into(),
                                        message: format!(
                                            "type `{}` does not satisfy interface bound `{}` for type parameter `{}` of `{}`",
                                            ta.display_name(),
                                            b,
                                            tp,
                                            ed.name
                                        ),
                                        span,
                                        notes: vec![format!("add `impl {} for {} {{ ... }}`", b, ta.display_name())],
                                        counterexample: BTreeMap::new(),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
        for (pty, a) in vi.payload_tys.iter().zip(args.iter()) {
            let expected = if esubst.is_empty() {
                pty.clone()
            } else {
                subst_type_expr(pty, &esubst)
            };
            let actual_ty = self.infer_expr_type(&a.value, var_types);
            if !self.types_compatible(&expected, &actual_ty) {
                self.diagnostics.push(Diagnostic {
                    code: "E0205".into(),
                    message: format!(
                        "variant `{}` payload expects `{}`, found `{}`",
                        label,
                        expected.display_name(),
                        actual_ty.display_name()
                    ),
                    span: a.value.span,
                    notes: vec![],
                    counterexample: BTreeMap::new(),
                });
            }
        }
    }

    fn check_lambda(
        &mut self,
        params: &[LambdaParam],
        ret_ty: Option<&TypeExpr>,
        body: &Expr,
        expected: Option<&TypeExpr>,
        outer_var_types: &BTreeMap<String, TypeExpr>,
        span: Span,
    ) {
        let (exp_params, exp_ret): (Vec<TypeExpr>, Option<TypeExpr>) =
            match expected.map(|t| self.resolve_base_type(t)) {
                Some(TypeExpr::Fn { params: ep, ret: er, .. }) => (ep, er.map(|r| *r)),
                _ => (Vec::new(), None),
            };
        let mut lam_vars = outer_var_types.clone();
        let mut lam_env = L0Env {
            pure_fns: self.pure_fns.clone(),
            type_aliases: self.type_aliases.clone(),
            ..Default::default()
        };
        let mut lam_smt_vars = BTreeSet::new();
        let mut lam_premises = Vec::new();
        for (i, p) in params.iter().enumerate() {
            let pty = p
                .ty
                .clone()
                .or_else(|| exp_params.get(i).cloned())
                .unwrap_or_else(|| named_ty("Any"));
            if p.name != "_" {
                lam_vars.insert(p.name.clone(), pty.clone());
                self.assume_type_facts(&p.name, &pty, &mut lam_env, &mut lam_smt_vars, &mut lam_premises);
            }
        }
        let effective_ret = ret_ty.cloned().or(exp_ret);
        let synth_fn = FnDecl {
            attrs: vec![],
            is_pub: false,
            is_pure: false,
            is_async: false,
            is_test: false,
            is_unsafe: false,
            name: "<lambda>".into(),
            type_params: vec![],
            type_param_bounds: BTreeMap::new(),
            params: vec![],
            ret_ty: effective_ret.clone(),
            requires: vec![],
            ensures: vec![],
            body: Block { stmts: vec![], tail: None, span },
            span,
        };
        let mut lam_mut = BTreeSet::new();
        let mut lam_saw_return = false;
        self.check_expr(
            body,
            &synth_fn,
            &mut lam_env,
            &mut lam_smt_vars,
            &mut lam_premises,
            &lam_vars,
            &lam_mut,
            &None,
            &mut lam_saw_return,
        );
        if let Some(rt) = ret_ty {
            let actual = self.infer_expr_type_expected(body, Some(rt), &lam_vars);
            if !self.types_compatible(rt, &actual) {
                self.diagnostics.push(Diagnostic {
                    code: "E0205".into(),
                    message: format!(
                        "lambda return type mismatch: expected `{}`, found `{}`",
                        rt.display_name(),
                        actual.display_name()
                    ),
                    span: body.span,
                    notes: vec![],
                    counterexample: BTreeMap::new(),
                });
            }
        }
        let _ = &mut lam_mut;
    }

    #[allow(clippy::too_many_arguments)]
    fn check_expr_satisfies_type(
        &mut self,
        expr: &Expr,
        ty: &TypeExpr,
        env: &L0Env,
        smt_int_vars: &BTreeSet<String>,
        smt_premises: &[(Expr, BTreeMap<String, String>)],
        context: &str,
        span: Span,
    ) {
        let TypeExpr::Named { name, .. } = ty else {
            return;
        };
        // Check exact-width scalar range bounds (e.g. `let x UInt8 = 256` rejected at L0!)
        if let Some(sk) = ScalarKind::from_name(name) {
            if sk.is_int() && sk != ScalarKind::I64 {
                if let (Some(lo), Some(hi)) = (sk.min_i128(), sk.max_i128()) {
                    let abs = if sk == ScalarKind::U64 {
                        if let ExprKind::Int(n) = &expr.kind {
                            AbsInt::exact((*n as u64) as i128)
                        } else {
                            env.eval_abs_int(expr, None)
                        }
                    } else {
                        env.eval_abs_int(expr, None)
                    };
                    if abs.interval.max < lo || abs.interval.min > hi {
                        self.diagnostics.push(Diagnostic {
                            code: "E0410".into(),
                            message: format!(
                                "{} violates range of `{}` ({}..={})",
                                context, name, lo, hi
                            ),
                            span,
                            notes: vec![format!(
                                "value interval [{}, {}] is outside {}..={}",
                                abs.interval.min, abs.interval.max, lo, hi
                            )],
                            counterexample: BTreeMap::new(),
                        });
                    } else if abs.interval.min >= lo && abs.interval.max <= hi {
                        self.proofs.push(ProofRecord {
                            obligation: format!("{} fits in `{}` ({}..={})", context, name, lo, hi),
                            tier: ProofTier::L0Fast,
                            span,
                        });
                    }
                }
            }
            return;
        }
        let Some(alias) = self.type_aliases.get(name).cloned() else {
            return;
        };
        let Some(pred) = &alias.predicate else {
            return;
        };

        // Float refinement verification (L0 float constant / bound checker for e.g. `UnitFloat = Float where 0.0 <= self && self <= 1.0`)
        if self.is_float_ty(&alias.base) {
            if let Some(fval) = eval_const_float(expr, &self.consts) {
                match check_float_refinement_l0(pred, fval) {
                    Some(true) => {
                        self.proofs.push(ProofRecord {
                            obligation: format!("{} satisfies float refinement `{}`", context, name),
                            tier: ProofTier::L0Fast,
                            span,
                        });
                        return;
                    }
                    Some(false) => {
                        self.diagnostics.push(Diagnostic {
                            code: "E0410".into(),
                            message: format!(
                                "{} ({}) violates float refinement `{}`",
                                context, fval, name
                            ),
                            span,
                            notes: vec![],
                            counterexample: BTreeMap::new(),
                        });
                        return;
                    }
                    None => {}
                }
            }
            // If dynamic variable already has exact alias type, accept
            if let ExprKind::Ident(_) = &expr.kind {
                return;
            }
            return;
        }

        let mut check_env = env.clone();
        let abs = env.eval_abs_int(expr, None);
        check_env.set_int("__val", abs);
        if let ExprKind::Ident(vname) = &expr.kind {
            let extra: Vec<String> = env
                .proved_calls
                .iter()
                .filter_map(|c| {
                    let pat = format!("({})", vname);
                    if c.ends_with(&pat) {
                        Some(format!("{}(__val)", &c[..c.len() - pat.len()]))
                    } else {
                        None
                    }
                })
                .collect();
            for e in extra {
                check_env.proved_calls.insert(e);
            }
        }

        match check_env.check_predicate(pred, Some("__val")) {
            L0Result::Proved => {
                self.proofs.push(ProofRecord {
                    obligation: format!("{} satisfies `{}`", context, name),
                    tier: ProofTier::L0Fast,
                    span,
                });
            }
            L0Result::Disproved(reason) => {
                self.diagnostics.push(Diagnostic {
                    code: "E0410".into(),
                    message: format!("{} violates refinement type `{}`", context, name),
                    span,
                    notes: vec![reason],
                    counterexample: BTreeMap::new(),
                });
            }
            L0Result::Unknown => {
                let mut q = SmtQuery::new(self.smt_pure_fns.clone());
                for v in smt_int_vars {
                    q.declare_int(v.clone());
                }
                q.declare_int("__val");
                for (p, s) in smt_premises {
                    let _ = q.add_premise_expr(p, s);
                }
                let eq = Expr {
                    kind: ExprKind::Binary {
                        op: BinOp::Eq,
                        left: Box::new(Expr {
                            kind: ExprKind::Ident("__val".into()),
                            span,
                        }),
                        right: Box::new(expr.clone()),
                    },
                    span,
                };
                if q.add_premise_expr(&eq, &BTreeMap::new()).is_err() {
                    self.diagnostics.push(Diagnostic {
                        code: "E0411".into(),
                        message: format!("unproved refinement `{}` for {} (Fail-Closed)", name, context),
                        span,
                        notes: vec!["expression outside supported SMT fragment".into()],
                        counterexample: BTreeMap::new(),
                    });
                    return;
                }
                let mut subst = BTreeMap::new();
                subst.insert("self".into(), "__val".into());
                if let Err(e) = q.set_goal_expr(pred, &subst) {
                    self.diagnostics.push(Diagnostic {
                        code: "E0411".into(),
                        message: format!("unproved refinement `{}` for {} (Fail-Closed)", name, context),
                        span,
                        notes: vec![e],
                        counterexample: BTreeMap::new(),
                    });
                    return;
                }
                match q.solve() {
                    SmtOutcome::Proved => self.proofs.push(ProofRecord {
                        obligation: format!("{} satisfies `{}`", context, name),
                        tier: ProofTier::L1Smt,
                        span,
                    }),
                    SmtOutcome::Counterexample(ce) => self.diagnostics.push(Diagnostic {
                        code: "E0410".into(),
                        message: format!("{} may violate refinement type `{}`", context, name),
                        span,
                        notes: vec!["counterexample found by L1 SMT (Z3)".into()],
                        counterexample: ce,
                    }),
                    SmtOutcome::Unknown(msg) => self.diagnostics.push(Diagnostic {
                        code: "E0411".into(),
                        message: format!("unproved refinement `{}` for {} (Fail-Closed)", name, context),
                        span,
                        notes: vec![msg],
                        counterexample: BTreeMap::new(),
                    }),
                }
            }
        }
    }

    fn check_ensures(
        &mut self,
        f: &FnDecl,
        ens: &Expr,
        tail_expr: Option<&Expr>,
        env: &L0Env,
        smt_int_vars: &BTreeSet<String>,
        smt_premises: &[(Expr, BTreeMap<String, String>)],
    ) {
        let mut check_env = env.clone();
        if let Some(te) = tail_expr {
            let abs = env.eval_abs_int(te, None);
            check_env.set_int("result", abs);
        }

        match check_env.check_predicate(ens, Some("result")) {
            L0Result::Proved => {
                self.proofs.push(ProofRecord {
                    obligation: format!("ensures on `{}`", f.name),
                    tier: ProofTier::L0Fast,
                    span: ens.span,
                });
                return;
            }
            L0Result::Disproved(reason) => {
                self.diagnostics.push(Diagnostic {
                    code: "V1004".into(),
                    message: format!("postcondition `ensures` violated on `{}`", f.name),
                    span: ens.span,
                    notes: vec![reason],
                    counterexample: BTreeMap::new(),
                });
                return;
            }
            L0Result::Unknown => {}
        }

        let mut q = SmtQuery::new(self.smt_pure_fns.clone());
        for v in smt_int_vars {
            q.declare_int(v.clone());
        }
        q.declare_int("result");
        for (p, s) in smt_premises {
            let _ = q.add_premise_expr(p, s);
        }
        if let Some(te) = tail_expr {
            if let Ok(te_smt) = crate::smt::encode_expr(te, &BTreeMap::new()) {
                q.premises.push(format!("(= result {})", te_smt));
            } else if let Ok(body_smt) = crate::smt::encode_block(&f.body, &BTreeMap::new()) {
                q.premises.push(format!("(= result {})", body_smt));
            }
        } else if let Ok(body_smt) = crate::smt::encode_block(&f.body, &BTreeMap::new()) {
            q.premises.push(format!("(= result {})", body_smt));
        }
        if let Err(e) = q.set_goal_expr(ens, &BTreeMap::new()) {
            self.diagnostics.push(Diagnostic {
                code: "V1005".into(),
                message: format!("unproved `ensures` on `{}` (Fail-Closed)", f.name),
                span: ens.span,
                notes: vec![e],
                counterexample: BTreeMap::new(),
            });
            return;
        }
        match q.solve() {
            SmtOutcome::Proved => self.proofs.push(ProofRecord {
                obligation: format!("ensures on `{}`", f.name),
                tier: ProofTier::L1Smt,
                span: ens.span,
            }),
            SmtOutcome::Counterexample(ce) => self.diagnostics.push(Diagnostic {
                code: "V1004".into(),
                message: format!("postcondition `ensures` may be violated on `{}`", f.name),
                span: ens.span,
                notes: vec!["counterexample found by L1 SMT (Z3)".into()],
                counterexample: ce,
            }),
            SmtOutcome::Unknown(msg) => self.diagnostics.push(Diagnostic {
                code: "V1005".into(),
                message: format!("unproved `ensures` on `{}` (Fail-Closed)", f.name),
                span: ens.span,
                notes: vec![msg],
                counterexample: BTreeMap::new(),
            }),
        }
    }
}

fn assign_op_to_binop(op: AssignOp) -> Option<BinOp> {
    match op {
        AssignOp::Set => None,
        AssignOp::AddSet => Some(BinOp::Add),
        AssignOp::SubSet => Some(BinOp::Sub),
        AssignOp::MulSet => Some(BinOp::Mul),
        AssignOp::DivSet => Some(BinOp::Div),
        AssignOp::BitAndSet => Some(BinOp::BitAnd),
        AssignOp::BitOrSet => Some(BinOp::BitOr),
        AssignOp::BitXorSet => Some(BinOp::BitXor),
        AssignOp::ShlSet => Some(BinOp::Shl),
        AssignOp::ShrSet => Some(BinOp::Shr),
    }
}

fn pattern_is_irrefutable(pat: &Pattern) -> bool {
    match pat {
        Pattern::Wildcard(_) | Pattern::Ident(_, _) => true,
        Pattern::Tuple(pats, _) => pats.iter().all(pattern_is_irrefutable),
        _ => false,
    }
}

fn collect_pattern_bindings(pat: &Pattern, out: &mut BTreeSet<String>) {
    match pat {
        Pattern::Ident(s, _) if s != "_" => {
            out.insert(s.clone());
        }
        Pattern::Variant { binding: Some(b), .. } if b != "_" => {
            out.insert(b.clone());
        }
        Pattern::VariantMulti { patterns, .. } | Pattern::Tuple(patterns, _) => {
            for p in patterns {
                collect_pattern_bindings(p, out);
            }
        }
        _ => {}
    }
}

fn block_always_diverges(blk: &Block) -> bool {
    if blk.stmts.iter().any(|s| {
        matches!(
            s,
            Stmt::Return { .. } | Stmt::Break { .. } | Stmt::Continue { .. }
        )
    }) {
        return true;
    }
    if let Some(tail) = &blk.tail {
        if let ExprKind::If {
            then_block,
            else_branch: Some(eb),
            ..
        } = &tail.kind
        {
            return block_always_diverges(then_block) && expr_always_diverges(eb);
        }
    }
    false
}

fn expr_always_diverges(expr: &Expr) -> bool {
    match &expr.kind {
        ExprKind::Block(b) => block_always_diverges(b),
        ExprKind::If {
            then_block,
            else_branch: Some(eb),
            ..
        } => block_always_diverges(then_block) && expr_always_diverges(eb),
        _ => false,
    }
}

fn collect_mutated_vars_in_block(blk: &Block, out: &mut BTreeSet<String>) {
    for stmt in &blk.stmts {
        match stmt {
            Stmt::Assign { target, .. } => match &target.kind {
                ExprKind::Ident(s) => {
                    out.insert(s.clone());
                }
                ExprKind::Field { target: obj, .. } => {
                    if let ExprKind::Ident(s) = &obj.kind {
                        out.insert(s.clone());
                    }
                }
                _ => {}
            },
            Stmt::Mutate { target, body, .. } => {
                out.insert(target.clone());
                collect_mutated_vars_in_block(body, out);
            }
            Stmt::While { body, .. } | Stmt::For { body, .. } => {
                collect_mutated_vars_in_block(body, out);
            }
            Stmt::Expr { expr, .. } => collect_mutated_vars_in_expr(expr, out),
            _ => {}
        }
    }
    if let Some(tail) = &blk.tail {
        collect_mutated_vars_in_expr(tail, out);
    }
}

fn collect_mutated_vars_in_expr(expr: &Expr, out: &mut BTreeSet<String>) {
    match &expr.kind {
        ExprKind::If {
            then_block,
            else_branch,
            ..
        } => {
            collect_mutated_vars_in_block(then_block, out);
            if let Some(eb) = else_branch {
                collect_mutated_vars_in_expr(eb, out);
            }
        }
        ExprKind::Match { arms, .. } => {
            for a in arms {
                collect_mutated_vars_in_expr(&a.body, out);
            }
        }
        ExprKind::Block(b) => collect_mutated_vars_in_block(b, out),
        _ => {}
    }
}

fn expr_mentions_ident(expr: &Expr, ident: &str) -> bool {
    let prefix = format!("{}.", ident);
    match &expr.kind {
        ExprKind::Ident(s) => s == ident || s.starts_with(&prefix),
        ExprKind::Field { target, .. } => expr_mentions_ident(target, ident),
        ExprKind::Binary { left, right, .. } => {
            expr_mentions_ident(left, ident) || expr_mentions_ident(right, ident)
        }
        ExprKind::Unary { expr: inner, .. } => expr_mentions_ident(inner, ident),
        ExprKind::Call { args, .. } => args.iter().any(|a| expr_mentions_ident(&a.value, ident)),
        _ => false,
    }
}

fn substitute_expr_idents(expr: &Expr, map: &BTreeMap<String, Expr>) -> Expr {
    let kind = match &expr.kind {
        ExprKind::Ident(s) => {
            if let Some(repl) = map.get(s) {
                return repl.clone();
            }
            expr.kind.clone()
        }
        ExprKind::Field { target, field } => ExprKind::Field {
            target: Box::new(substitute_expr_idents(target, map)),
            field: field.clone(),
        },
        ExprKind::Binary { op, left, right } => ExprKind::Binary {
            op: *op,
            left: Box::new(substitute_expr_idents(left, map)),
            right: Box::new(substitute_expr_idents(right, map)),
        },
        ExprKind::Unary { op, expr: inner } => ExprKind::Unary {
            op: *op,
            expr: Box::new(substitute_expr_idents(inner, map)),
        },
        ExprKind::Call { callee, args } => ExprKind::Call {
            callee: callee.clone(),
            args: args
                .iter()
                .map(|a| CallArg {
                    name: a.name.clone(),
                    value: substitute_expr_idents(&a.value, map),
                })
                .collect(),
        },
        other => other.clone(),
    };
    Expr {
        kind,
        span: expr.span,
    }
}

fn substitute_field_idents(expr: &Expr, fields: &[FieldDecl], prefix: &str) -> Expr {
    let field_set: BTreeSet<&str> = fields.iter().map(|f| f.name.as_str()).collect();
    fn rewrite(e: &Expr, fields: &BTreeSet<&str>, prefix: &str) -> Expr {
        let kind = match &e.kind {
            ExprKind::Ident(s) if fields.contains(s.as_str()) => ExprKind::Field {
                target: Box::new(Expr {
                    kind: ExprKind::Ident(prefix.to_string()),
                    span: e.span,
                }),
                field: s.clone(),
            },
            ExprKind::Binary { op, left, right } => ExprKind::Binary {
                op: *op,
                left: Box::new(rewrite(left, fields, prefix)),
                right: Box::new(rewrite(right, fields, prefix)),
            },
            ExprKind::Unary { op, expr } => ExprKind::Unary {
                op: *op,
                expr: Box::new(rewrite(expr, fields, prefix)),
            },
            ExprKind::Call { callee, args } => ExprKind::Call {
                callee: callee.clone(),
                args: args
                    .iter()
                    .map(|a| CallArg {
                        name: a.name.clone(),
                        value: rewrite(&a.value, fields, prefix),
                    })
                    .collect(),
            },
            other => other.clone(),
        };
        Expr { kind, span: e.span }
    }
    rewrite(expr, &field_set, prefix)
}

fn substitute_self_fields(expr: &Expr) -> Expr {
    let kind = match &expr.kind {
        ExprKind::Field { target, field } => {
            if let ExprKind::Ident(s) = &target.kind {
                if s == "self" {
                    return Expr {
                        kind: ExprKind::Ident(field.clone()),
                        span: expr.span,
                    };
                }
            }
            expr.kind.clone()
        }
        ExprKind::Binary { op, left, right } => ExprKind::Binary {
            op: *op,
            left: Box::new(substitute_self_fields(left)),
            right: Box::new(substitute_self_fields(right)),
        },
        ExprKind::Unary { op, expr: inner } => ExprKind::Unary {
            op: *op,
            expr: Box::new(substitute_self_fields(inner)),
        },
        other => other.clone(),
    };
    Expr {
        kind,
        span: expr.span,
    }
}

/// Automatically derive SQLite CHECK clauses from direct scalar predicates on records.
pub fn derive_sqlite_checks(program: &Program) -> BTreeMap<String, String> {
    let mut aliases: BTreeMap<String, Expr> = BTreeMap::new();
    for item in &program.items {
        if let Item::TypeAlias(ta) = item {
            if let Some(p) = &ta.predicate {
                aliases.insert(ta.name.clone(), p.clone());
            }
        }
    }
    let mut checks = BTreeMap::new();
    for item in &program.items {
        if let Item::Record(r) = item {
            let mut clauses = Vec::new();
            for f in &r.fields {
                if let Some(ap) = aliases.get(&f.ty.display_name()) {
                    if let Some(sql) = expr_to_sql(ap, Some(&f.name)) {
                        clauses.push(format!("({})", sql));
                    }
                }
                if let Some(pred) = &f.predicate {
                    if let Some(sql) = expr_to_sql(pred, Some(&f.name)) {
                        clauses.push(format!("({})", sql));
                    }
                }
            }
            for wc in &r.where_clauses {
                if let Some(sql) = expr_to_sql(&substitute_self_fields(wc), None) {
                    clauses.push(format!("({})", sql));
                }
            }
            if !clauses.is_empty() {
                checks.insert(r.name.clone(), format!("CHECK (({}) IS 1)", clauses.join(" AND ")));
            }
        }
    }
    checks
}

fn expr_to_sql(expr: &Expr, self_col: Option<&str>) -> Option<String> {
    match &expr.kind {
        ExprKind::Int(n) => Some(n.to_string()),
        ExprKind::Bool(b) => Some(if *b { "1".into() } else { "0".into() }),
        ExprKind::Ident(s) => {
            let col = if s == "self" {
                self_col?
            } else {
                s.as_str()
            };
            Some(format!("\"{}\"", col))
        }
        ExprKind::Unary { op, expr } => {
            let inner = expr_to_sql(expr, self_col)?;
            match op {
                UnOp::Neg => Some(format!("(-{})", inner)),
                UnOp::Not => Some(format!("(NOT {})", inner)),
                UnOp::BitNot => None,
            }
        }
        ExprKind::Binary { op, left, right } => {
            let l = expr_to_sql(left, self_col)?;
            let r = expr_to_sql(right, self_col)?;
            let sym = match op {
                BinOp::And => "AND",
                BinOp::Or => "OR",
                BinOp::Eq => "=",
                BinOp::Ne => "!=",
                BinOp::Lt => "<",
                BinOp::Le => "<=",
                BinOp::Gt => ">",
                BinOp::Ge => ">=",
                BinOp::Add => "+",
                BinOp::Sub => "-",
                BinOp::Mul => "*",
                BinOp::Mod => "%",
                _ => return None,
            };
            Some(format!("({} {} {})", l, sym, r))
        }
        _ => None,
    }
}

fn apply_mutate_block_symbolic(
    var: &str,
    blk: &Block,
    sym_fields: &mut BTreeMap<String, String>,
    mutated_any: &mut bool,
) {
    for stmt in &blk.stmts {
        match stmt {
            Stmt::Assign {
                target, op, value, ..
            } => {
                if let ExprKind::Field { target: obj, field } = &target.kind {
                    if matches!(&obj.kind, ExprKind::Ident(s) if s == var) {
                        let mut subst = BTreeMap::new();
                        for (k, v) in sym_fields.iter() {
                            subst.insert(format!("{}.{}", var, k), v.clone());
                            subst.insert(k.clone(), v.clone());
                        }
                        if let Ok(rhs) = crate::smt::encode_expr(value, &subst) {
                            let cur = sym_fields
                                .get(field)
                                .cloned()
                                .unwrap_or_else(|| crate::smt::smt_ident(&format!("{}.{}", var, field)));
                            let updated = match op {
                                AssignOp::Set => rhs,
                                AssignOp::AddSet => format!("(+ {} {})", cur, rhs),
                                AssignOp::SubSet => format!("(- {} {})", cur, rhs),
                                AssignOp::MulSet => format!("(* {} {})", cur, rhs),
                                AssignOp::DivSet => format!("(div {} {})", cur, rhs),
                                AssignOp::BitAndSet
                                | AssignOp::BitOrSet
                                | AssignOp::BitXorSet
                                | AssignOp::ShlSet
                                | AssignOp::ShrSet => cur,
                            };
                            sym_fields.insert(field.clone(), updated);
                            *mutated_any = true;
                        }
                    }
                }
            }
            Stmt::Expr { expr, .. } => {
                apply_mutate_expr_symbolic(var, expr, sym_fields, mutated_any);
            }
            _ => {}
        }
    }
    if let Some(tail) = &blk.tail {
        apply_mutate_expr_symbolic(var, tail, sym_fields, mutated_any);
    }
}

fn apply_mutate_expr_symbolic(
    var: &str,
    expr: &Expr,
    sym_fields: &mut BTreeMap<String, String>,
    mutated_any: &mut bool,
) {
    match &expr.kind {
        ExprKind::If {
            cond,
            then_block,
            else_branch,
        } => {
            let mut subst = BTreeMap::new();
            for (k, v) in sym_fields.iter() {
                subst.insert(format!("{}.{}", var, k), v.clone());
                subst.insert(k.clone(), v.clone());
            }
            let Ok(cond_smt) = crate::smt::encode_expr(cond, &subst) else {
                return;
            };
            let mut then_fields = sym_fields.clone();
            let mut else_fields = sym_fields.clone();
            let mut m1 = false;
            let mut m2 = false;
            apply_mutate_block_symbolic(var, then_block, &mut then_fields, &mut m1);
            if let Some(eb) = else_branch {
                apply_mutate_expr_symbolic(var, eb, &mut else_fields, &mut m2);
            }
            if m1 || m2 {
                *mutated_any = true;
                let keys: Vec<String> = sym_fields.keys().cloned().collect();
                for k in keys {
                    let t = &then_fields[&k];
                    let e = &else_fields[&k];
                    if t != e {
                        sym_fields.insert(k, format!("(ite {} {} {})", cond_smt, t, e));
                    }
                }
            }
        }
        ExprKind::Block(b) => apply_mutate_block_symbolic(var, b, sym_fields, mutated_any),
        _ => {}
    }
}

fn eval_const_int(
    expr: &Expr,
    consts: &BTreeMap<String, (ConstDecl, Option<i64>, Option<f64>)>,
    layout: &LayoutEngine,
) -> Option<i64> {
    match &expr.kind {
        ExprKind::Int(n) => Some(*n),
        ExprKind::Ident(s) => consts.get(s).and_then(|(_, iv, _)| *iv),
        ExprKind::Cast { expr: inner, ty } => {
            let v = eval_const_int(inner, consts, layout)?;
            if let TypeExpr::Named { name, .. } = layout.resolve_alias(ty) {
                if let Some(sk) = ScalarKind::from_name(&name) {
                    return Some(match sk {
                        ScalarKind::I8 => (v as i8) as i64,
                        ScalarKind::I16 => (v as i16) as i64,
                        ScalarKind::I32 => (v as i32) as i64,
                        ScalarKind::U8 => (v as u8) as i64,
                        ScalarKind::U16 => (v as u16) as i64,
                        ScalarKind::U32 => (v as u32) as i64,
                        _ => v,
                    });
                }
            }
            Some(v)
        }
        ExprKind::Unary { op, expr: inner } => {
            let v = eval_const_int(inner, consts, layout)?;
            match op {
                UnOp::Neg => Some(v.wrapping_neg()),
                UnOp::BitNot => Some(!v),
                UnOp::Not => None,
            }
        }
        ExprKind::Binary { op, left, right } => {
            let l = eval_const_int(left, consts, layout)?;
            let r = eval_const_int(right, consts, layout)?;
            match op {
                BinOp::Add => Some(l.wrapping_add(r)),
                BinOp::Sub => Some(l.wrapping_sub(r)),
                BinOp::Mul => Some(l.wrapping_mul(r)),
                BinOp::Div if r != 0 => Some(l.wrapping_div(r)),
                BinOp::Mod if r != 0 => Some(l.wrapping_rem(r)),
                BinOp::BitAnd => Some(l & r),
                BinOp::BitOr => Some(l | r),
                BinOp::BitXor => Some(l ^ r),
                BinOp::Shl => Some(l.wrapping_shl(r as u32)),
                BinOp::Shr => Some(l.wrapping_shr(r as u32)),
                _ => None,
            }
        }
        ExprKind::Call { callee, args } => match &callee.kind {
            ExprKind::GenericInst { expr: inner, type_args } if args.is_empty() => {
                if let ExprKind::Ident(fn_name) = &inner.kind {
                    if let Some(ta) = type_args.first() {
                        if fn_name == "size_of" {
                            return Some(layout.size_and_align(ta).0 as i64);
                        }
                        if fn_name == "align_of" {
                            return Some(layout.size_and_align(ta).1 as i64);
                        }
                    }
                }
                None
            }
            ExprKind::Ident(fn_name) if fn_name == "offset_of" && args.len() == 2 => {
                if let (ExprKind::Ident(tname), ExprKind::Str(fname)) =
                    (&args[0].value.kind, &args[1].value.kind)
                {
                    return layout.offset_of(tname, fname).map(|o| o as i64);
                }
                None
            }
            _ => None,
        },
        _ => None,
    }
}

fn eval_const_float(
    expr: &Expr,
    consts: &BTreeMap<String, (ConstDecl, Option<i64>, Option<f64>)>,
) -> Option<f64> {
    match &expr.kind {
        ExprKind::Float(f) => Some(*f),
        ExprKind::Ident(s) => consts.get(s).and_then(|(_, _, fv)| *fv),
        ExprKind::Unary { op: UnOp::Neg, expr: inner } => eval_const_float(inner, consts).map(|v| -v),
        ExprKind::Binary { op, left, right } => {
            let l = eval_const_float(left, consts)?;
            let r = eval_const_float(right, consts)?;
            match op {
                BinOp::Add => Some(l + r),
                BinOp::Sub => Some(l - r),
                BinOp::Mul => Some(l * r),
                BinOp::Div => Some(l / r),
                _ => None,
            }
        }
        _ => None,
    }
}

fn eval_float_term(expr: &Expr, self_val: f64) -> Option<f64> {
    match &expr.kind {
        ExprKind::Float(f) => Some(*f),
        ExprKind::Int(n) => Some(*n as f64),
        ExprKind::Ident(s) if s == "self" => Some(self_val),
        ExprKind::Unary { op: UnOp::Neg, expr: inner } => eval_float_term(inner, self_val).map(|v| -v),
        ExprKind::Binary { op, left, right } => {
            let l = eval_float_term(left, self_val)?;
            let r = eval_float_term(right, self_val)?;
            match op {
                BinOp::Add => Some(l + r),
                BinOp::Sub => Some(l - r),
                BinOp::Mul => Some(l * r),
                BinOp::Div => Some(l / r),
                _ => None,
            }
        }
        _ => None,
    }
}

fn check_float_refinement_l0(pred: &Expr, self_val: f64) -> Option<bool> {
    match &pred.kind {
        ExprKind::Binary { op, left, right } => match op {
            BinOp::And => Some(
                check_float_refinement_l0(left, self_val)?
                    && check_float_refinement_l0(right, self_val)?,
            ),
            BinOp::Or => Some(
                check_float_refinement_l0(left, self_val)?
                    || check_float_refinement_l0(right, self_val)?,
            ),
            BinOp::Lt => Some(eval_float_term(left, self_val)? < eval_float_term(right, self_val)?),
            BinOp::Le => Some(eval_float_term(left, self_val)? <= eval_float_term(right, self_val)?),
            BinOp::Gt => Some(eval_float_term(left, self_val)? > eval_float_term(right, self_val)?),
            BinOp::Ge => Some(eval_float_term(left, self_val)? >= eval_float_term(right, self_val)?),
            BinOp::Eq => Some(eval_float_term(left, self_val)? == eval_float_term(right, self_val)?),
            BinOp::Ne => Some(eval_float_term(left, self_val)? != eval_float_term(right, self_val)?),
            _ => None,
        },
        ExprKind::Unary { op: UnOp::Not, expr: inner } => {
            check_float_refinement_l0(inner, self_val).map(|b| !b)
        }
        _ => None,
    }
}

fn is_untyped_int_expr(expr: &Expr) -> bool {
    match &expr.kind {
        ExprKind::Int(_) => true,
        ExprKind::Unary {
            op: UnOp::Neg | UnOp::BitNot,
            expr: inner,
        } => is_untyped_int_expr(inner),
        ExprKind::Binary { op, left, right }
            if matches!(
                op,
                BinOp::Add
                    | BinOp::Sub
                    | BinOp::Mul
                    | BinOp::Div
                    | BinOp::Mod
                    | BinOp::BitAnd
                    | BinOp::BitOr
                    | BinOp::BitXor
                    | BinOp::Shl
                    | BinOp::Shr
            ) =>
        {
            is_untyped_int_expr(left) && is_untyped_int_expr(right)
        }
        _ => false,
    }
}

fn is_untyped_float_expr(expr: &Expr) -> bool {
    match &expr.kind {
        ExprKind::Float(_) => true,
        ExprKind::Unary {
            op: UnOp::Neg,
            expr: inner,
        } => is_untyped_float_expr(inner),
        _ => false,
    }
}
