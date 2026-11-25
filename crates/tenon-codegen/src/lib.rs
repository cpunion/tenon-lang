pub mod bindings;
pub mod gpu_shader;
pub mod llvm_aot;
pub(crate) mod llvm_aot_rt;
pub mod llvm_runtime;
pub mod lower_interfaces;
pub use tenon_rt::perceus;
pub mod wasm_client;
pub mod wasm_general;
pub mod wasm_llvm;

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{
    AbiParam, Block as ClifBlock, InstBuilder, MemFlags, StackSlotData, StackSlotKind, TrapCode,
    Type, Value, types,
};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataDescription, DataId, FuncId, Linkage, Module};
use lower_interfaces::{DynInterfaceLayout, prepare_program_for_codegen};
use perceus::*;
use std::collections::{BTreeMap, BTreeSet};
use tenon_rt::{syscalls, systems};
use tenon_syntax::*;

/// Minimal Tenon Core Byte, OS/Network & Systems Syscall Table.
/// Shared between Cranelift JIT (`lib.rs`) and LLVM 22 AOT (`llvm_aot.rs`).
pub(crate) struct CoreSyscallSpec {
    pub tenon_name: &'static str,
    pub c_symbol: &'static str,
    pub arity: usize,
    pub returns_rc: bool,
    #[allow(dead_code)]
    pub returns_text: bool,
    #[allow(dead_code)]
    pub returns_float: bool,
    pub ret_ty: &'static str,
}

pub(crate) fn is_consuming_cow_builtin(name: &str) -> bool {
    matches!(
        name,
        "bytes_set"
            | "bytes_push"
            | "bytes_write_u16_le"
            | "bytes_write_u16_be"
            | "bytes_write_u32_le"
            | "bytes_write_u32_be"
            | "bytes_write_u64_le"
            | "bytes_write_u64_be"
            | "simd_u8x16_set_lane"
    )
}

pub(crate) const CORE_SYSCALLS: &[CoreSyscallSpec] = &[
    // 1. Core byte / text primitives (4)
    CoreSyscallSpec { tenon_name: "byte_at", c_symbol: "tenon_byte_at", arity: 2, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    CoreSyscallSpec { tenon_name: "str_slice", c_symbol: "tenon_str_slice", arity: 3, returns_rc: true, returns_text: true, returns_float: false, ret_ty: "Text" },
    CoreSyscallSpec { tenon_name: "byte_to_text", c_symbol: "tenon_byte_to_text", arity: 1, returns_rc: true, returns_text: true, returns_float: false, ret_ty: "Text" },
    CoreSyscallSpec { tenon_name: "str_concat", c_symbol: "tenon_str_concat", arity: 2, returns_rc: true, returns_text: true, returns_float: false, ret_ty: "Text" },
    // 1b. O(n) Byte buffer primitives (4)
    CoreSyscallSpec { tenon_name: "buf_new", c_symbol: "tenon_buf_new", arity: 0, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    CoreSyscallSpec { tenon_name: "buf_push_str", c_symbol: "tenon_buf_push_str", arity: 2, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    CoreSyscallSpec { tenon_name: "buf_push_byte", c_symbol: "tenon_buf_push_byte", arity: 2, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    CoreSyscallSpec { tenon_name: "buf_finish", c_symbol: "tenon_buf_finish", arity: 1, returns_rc: true, returns_text: true, returns_float: false, ret_ty: "Text" },
    // 1c. Minimal IEEE-754 Float64 primitives (6)
    CoreSyscallSpec { tenon_name: "int_to_float", c_symbol: "tenon_int_to_float", arity: 1, returns_rc: false, returns_text: false, returns_float: true, ret_ty: "Float" },
    CoreSyscallSpec { tenon_name: "float_to_int", c_symbol: "tenon_float_to_int", arity: 1, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    CoreSyscallSpec { tenon_name: "float_to_text", c_symbol: "tenon_float_to_text", arity: 1, returns_rc: true, returns_text: true, returns_float: false, ret_ty: "Text" },
    CoreSyscallSpec { tenon_name: "parse_float", c_symbol: "tenon_parse_float", arity: 1, returns_rc: false, returns_text: false, returns_float: true, ret_ty: "Float" },
    CoreSyscallSpec { tenon_name: "float_is_nan", c_symbol: "tenon_float_is_nan", arity: 1, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Bool" },
    CoreSyscallSpec { tenon_name: "is_finite", c_symbol: "tenon_float_is_finite", arity: 1, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Bool" },
    CoreSyscallSpec { tenon_name: "float_bits", c_symbol: "tenon_float_bits", arity: 1, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    // 2. Minimal OS / I/O / FS / Process syscalls (10)
    CoreSyscallSpec { tenon_name: "sys_write_stdout", c_symbol: "tenon_sys_write_stdout", arity: 1, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    CoreSyscallSpec { tenon_name: "sys_read_line", c_symbol: "tenon_sys_read_line", arity: 0, returns_rc: true, returns_text: true, returns_float: false, ret_ty: "Text" },
    CoreSyscallSpec { tenon_name: "sys_is_eof", c_symbol: "tenon_sys_is_eof", arity: 0, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Bool" },
    CoreSyscallSpec { tenon_name: "sys_env", c_symbol: "tenon_sys_env", arity: 1, returns_rc: true, returns_text: true, returns_float: false, ret_ty: "Text" },
    CoreSyscallSpec { tenon_name: "sys_cwd", c_symbol: "tenon_sys_cwd", arity: 0, returns_rc: true, returns_text: true, returns_float: false, ret_ty: "Text" },
    CoreSyscallSpec { tenon_name: "sys_read_file", c_symbol: "tenon_sys_read_file", arity: 1, returns_rc: true, returns_text: true, returns_float: false, ret_ty: "Text" },
    CoreSyscallSpec { tenon_name: "sys_file_ok", c_symbol: "tenon_sys_file_ok", arity: 0, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Bool" },
    CoreSyscallSpec { tenon_name: "sys_write_file", c_symbol: "tenon_sys_write_file", arity: 2, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Bool" },
    CoreSyscallSpec { tenon_name: "sys_exec", c_symbol: "tenon_sys_exec", arity: 1, returns_rc: true, returns_text: true, returns_float: false, ret_ty: "Text" },
    CoreSyscallSpec { tenon_name: "sys_exec_exit_code", c_symbol: "tenon_sys_exec_exit_code", arity: 0, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    // 3. Minimal Raw HTTPS line-stream transport syscalls (6)
    CoreSyscallSpec { tenon_name: "sys_http_open", c_symbol: "tenon_sys_http_open", arity: 4, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    CoreSyscallSpec { tenon_name: "sys_http_sse_open", c_symbol: "tenon_sys_http_sse_open", arity: 3, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    CoreSyscallSpec { tenon_name: "sys_http_sse_next_line", c_symbol: "tenon_sys_http_sse_next_line", arity: 1, returns_rc: true, returns_text: true, returns_float: false, ret_ty: "Text" },
    CoreSyscallSpec { tenon_name: "sys_http_sse_eof", c_symbol: "tenon_sys_http_sse_eof", arity: 1, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Bool" },
    CoreSyscallSpec { tenon_name: "sys_http_sse_error", c_symbol: "tenon_sys_http_sse_error", arity: 1, returns_rc: true, returns_text: true, returns_float: false, ret_ty: "Text" },
    CoreSyscallSpec { tenon_name: "sys_http_sse_close", c_symbol: "tenon_sys_http_sse_close", arity: 1, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    // 4. S1 Bit intrinsics & Float32 text ops
    CoreSyscallSpec { tenon_name: "clz", c_symbol: "tenon_clz", arity: 1, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    CoreSyscallSpec { tenon_name: "ctz", c_symbol: "tenon_ctz", arity: 1, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    CoreSyscallSpec { tenon_name: "popcnt", c_symbol: "tenon_popcnt", arity: 1, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    CoreSyscallSpec { tenon_name: "bswap16", c_symbol: "tenon_bswap16", arity: 1, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "UInt16" },
    CoreSyscallSpec { tenon_name: "bswap32", c_symbol: "tenon_bswap32", arity: 1, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "UInt32" },
    CoreSyscallSpec { tenon_name: "bswap64", c_symbol: "tenon_bswap64", arity: 1, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "UInt64" },
    CoreSyscallSpec { tenon_name: "rotl64", c_symbol: "tenon_rotl64", arity: 2, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "UInt64" },
    CoreSyscallSpec { tenon_name: "rotr64", c_symbol: "tenon_rotr64", arity: 2, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "UInt64" },
    CoreSyscallSpec { tenon_name: "f32_to_text", c_symbol: "tenon_f32_to_text", arity: 1, returns_rc: true, returns_text: true, returns_float: false, ret_ty: "Text" },
    CoreSyscallSpec { tenon_name: "parse_f32", c_symbol: "tenon_parse_f32", arity: 1, returns_rc: false, returns_text: false, returns_float: true, ret_ty: "Float32" },
    // 5. S1 Compact Bytes & zero-copy BytesSlice
    CoreSyscallSpec { tenon_name: "bytes_new", c_symbol: "tenon_bytes_new", arity: 1, returns_rc: true, returns_text: false, returns_float: false, ret_ty: "Bytes" },
    CoreSyscallSpec { tenon_name: "bytes_filled", c_symbol: "tenon_bytes_filled", arity: 2, returns_rc: true, returns_text: false, returns_float: false, ret_ty: "Bytes" },
    CoreSyscallSpec { tenon_name: "bytes_len", c_symbol: "tenon_bytes_len", arity: 1, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    CoreSyscallSpec { tenon_name: "bytes_get", c_symbol: "tenon_bytes_get", arity: 2, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "UInt8" },
    CoreSyscallSpec { tenon_name: "bytes_set", c_symbol: "tenon_bytes_set", arity: 3, returns_rc: true, returns_text: false, returns_float: false, ret_ty: "Bytes" },
    CoreSyscallSpec { tenon_name: "bytes_push", c_symbol: "tenon_bytes_push", arity: 2, returns_rc: true, returns_text: false, returns_float: false, ret_ty: "Bytes" },
    CoreSyscallSpec { tenon_name: "bytes_read_u16_le", c_symbol: "tenon_bytes_read_u16_le", arity: 2, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "UInt16" },
    CoreSyscallSpec { tenon_name: "bytes_read_u16_be", c_symbol: "tenon_bytes_read_u16_be", arity: 2, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "UInt16" },
    CoreSyscallSpec { tenon_name: "bytes_read_u32_le", c_symbol: "tenon_bytes_read_u32_le", arity: 2, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "UInt32" },
    CoreSyscallSpec { tenon_name: "bytes_read_u32_be", c_symbol: "tenon_bytes_read_u32_be", arity: 2, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "UInt32" },
    CoreSyscallSpec { tenon_name: "bytes_read_u64_le", c_symbol: "tenon_bytes_read_u64_le", arity: 2, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "UInt64" },
    CoreSyscallSpec { tenon_name: "bytes_read_u64_be", c_symbol: "tenon_bytes_read_u64_be", arity: 2, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "UInt64" },
    CoreSyscallSpec { tenon_name: "bytes_write_u16_le", c_symbol: "tenon_bytes_write_u16_le", arity: 3, returns_rc: true, returns_text: false, returns_float: false, ret_ty: "Bytes" },
    CoreSyscallSpec { tenon_name: "bytes_write_u16_be", c_symbol: "tenon_bytes_write_u16_be", arity: 3, returns_rc: true, returns_text: false, returns_float: false, ret_ty: "Bytes" },
    CoreSyscallSpec { tenon_name: "bytes_write_u32_le", c_symbol: "tenon_bytes_write_u32_le", arity: 3, returns_rc: true, returns_text: false, returns_float: false, ret_ty: "Bytes" },
    CoreSyscallSpec { tenon_name: "bytes_write_u32_be", c_symbol: "tenon_bytes_write_u32_be", arity: 3, returns_rc: true, returns_text: false, returns_float: false, ret_ty: "Bytes" },
    CoreSyscallSpec { tenon_name: "bytes_write_u64_le", c_symbol: "tenon_bytes_write_u64_le", arity: 3, returns_rc: true, returns_text: false, returns_float: false, ret_ty: "Bytes" },
    CoreSyscallSpec { tenon_name: "bytes_write_u64_be", c_symbol: "tenon_bytes_write_u64_be", arity: 3, returns_rc: true, returns_text: false, returns_float: false, ret_ty: "Bytes" },
    CoreSyscallSpec { tenon_name: "bytes_slice", c_symbol: "tenon_bytes_slice", arity: 3, returns_rc: true, returns_text: false, returns_float: false, ret_ty: "BytesSlice" },
    CoreSyscallSpec { tenon_name: "slice_len", c_symbol: "tenon_slice_len", arity: 1, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    CoreSyscallSpec { tenon_name: "slice_get", c_symbol: "tenon_slice_get", arity: 2, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "UInt8" },
    CoreSyscallSpec { tenon_name: "bytes_to_text", c_symbol: "tenon_bytes_to_text", arity: 1, returns_rc: true, returns_text: true, returns_float: false, ret_ty: "Text" },
    CoreSyscallSpec { tenon_name: "text_to_bytes", c_symbol: "tenon_text_to_bytes", arity: 1, returns_rc: true, returns_text: false, returns_float: false, ret_ty: "Bytes" },
    CoreSyscallSpec { tenon_name: "bytes_ptr", c_symbol: "tenon_bytes_ptr", arity: 1, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Ptr[UInt8]" },
    CoreSyscallSpec { tenon_name: "list_data_ptr", c_symbol: "tenon_list_data_ptr", arity: 1, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Ptr[UInt8]" },
    CoreSyscallSpec { tenon_name: "gpu_global_id_x", c_symbol: "tenon_gpu_gid_x", arity: 0, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    CoreSyscallSpec { tenon_name: "gpu_global_id_y", c_symbol: "tenon_gpu_gid_y", arity: 0, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    CoreSyscallSpec { tenon_name: "gpu_global_id_z", c_symbol: "tenon_gpu_gid_z", arity: 0, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    CoreSyscallSpec { tenon_name: "gpu_local_id_x", c_symbol: "tenon_gpu_lid_x", arity: 0, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    CoreSyscallSpec { tenon_name: "gpu_local_id_y", c_symbol: "tenon_gpu_lid_y", arity: 0, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    CoreSyscallSpec { tenon_name: "gpu_group_id_x", c_symbol: "tenon_gpu_wgid_x", arity: 0, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    CoreSyscallSpec { tenon_name: "gpu_group_id_y", c_symbol: "tenon_gpu_wgid_y", arity: 0, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    CoreSyscallSpec { tenon_name: "gpu_barrier", c_symbol: "tenon_gpu_barrier", arity: 0, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    CoreSyscallSpec { tenon_name: "gpu_shared_f32", c_symbol: "tenon_gpu_shared_f32", arity: 1, returns_rc: true, returns_text: false, returns_float: false, ret_ty: "List[Float32]" },
    CoreSyscallSpec { tenon_name: "text_cstr_ptr", c_symbol: "tenon_text_cstr_ptr", arity: 1, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "CStr" },
    CoreSyscallSpec { tenon_name: "cstr_to_text", c_symbol: "tenon_cstr_to_text", arity: 1, returns_rc: true, returns_text: true, returns_float: false, ret_ty: "Text" },
    // 6. S2 Raw memory ops
    CoreSyscallSpec { tenon_name: "raw_alloc", c_symbol: "tenon_raw_alloc", arity: 1, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "MutPtr[UInt8]" },
    CoreSyscallSpec { tenon_name: "raw_free", c_symbol: "tenon_raw_free", arity: 2, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    CoreSyscallSpec { tenon_name: "raw_memcpy", c_symbol: "tenon_raw_memcpy", arity: 3, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "MutPtr[UInt8]" },
    CoreSyscallSpec { tenon_name: "raw_memset", c_symbol: "tenon_raw_memset", arity: 3, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "MutPtr[UInt8]" },
    CoreSyscallSpec { tenon_name: "raw_memcmp", c_symbol: "tenon_raw_memcmp", arity: 3, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    // 7. S3 Threads, Atomics, Mutex, Channel
    CoreSyscallSpec { tenon_name: "thread_spawn", c_symbol: "tenon_thread_spawn", arity: 1, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    CoreSyscallSpec { tenon_name: "thread_join", c_symbol: "tenon_thread_join", arity: 1, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    CoreSyscallSpec { tenon_name: "thread_sleep_ms", c_symbol: "tenon_thread_sleep_ms", arity: 1, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    CoreSyscallSpec { tenon_name: "thread_hw_concurrency", c_symbol: "tenon_thread_hw_concurrency", arity: 0, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    CoreSyscallSpec { tenon_name: "atomic_new", c_symbol: "tenon_atomic_new", arity: 1, returns_rc: true, returns_text: false, returns_float: false, ret_ty: "AtomicInt" },
    CoreSyscallSpec { tenon_name: "atomic_load", c_symbol: "tenon_atomic_load", arity: 2, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    CoreSyscallSpec { tenon_name: "atomic_store", c_symbol: "tenon_atomic_store", arity: 3, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    CoreSyscallSpec { tenon_name: "atomic_fetch_add", c_symbol: "tenon_atomic_fetch_add", arity: 3, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    CoreSyscallSpec { tenon_name: "atomic_fetch_sub", c_symbol: "tenon_atomic_fetch_sub", arity: 3, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    CoreSyscallSpec { tenon_name: "atomic_cas", c_symbol: "tenon_atomic_cas", arity: 4, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Bool" },
    CoreSyscallSpec { tenon_name: "atomic_swap", c_symbol: "tenon_atomic_swap", arity: 3, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    CoreSyscallSpec { tenon_name: "mutex_new", c_symbol: "tenon_mutex_new", arity: 2, returns_rc: true, returns_text: false, returns_float: false, ret_ty: "Mutex" },
    CoreSyscallSpec { tenon_name: "mutex_lock", c_symbol: "tenon_mutex_lock", arity: 1, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    CoreSyscallSpec { tenon_name: "mutex_unlock", c_symbol: "tenon_mutex_unlock", arity: 1, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    CoreSyscallSpec { tenon_name: "mutex_get", c_symbol: "tenon_mutex_get", arity: 1, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Any" },
    CoreSyscallSpec { tenon_name: "mutex_set", c_symbol: "tenon_mutex_set", arity: 2, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    CoreSyscallSpec { tenon_name: "chan_new", c_symbol: "tenon_chan_new", arity: 1, returns_rc: true, returns_text: false, returns_float: false, ret_ty: "Channel" },
    CoreSyscallSpec { tenon_name: "chan_send", c_symbol: "tenon_chan_send", arity: 2, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Bool" },
    CoreSyscallSpec { tenon_name: "chan_recv", c_symbol: "tenon_chan_recv", arity: 1, returns_rc: true, returns_text: false, returns_float: false, ret_ty: "Option" },
    CoreSyscallSpec { tenon_name: "chan_try_recv", c_symbol: "tenon_chan_try_recv", arity: 1, returns_rc: true, returns_text: false, returns_float: false, ret_ty: "Option" },
    CoreSyscallSpec { tenon_name: "chan_close", c_symbol: "tenon_chan_close", arity: 1, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    // 8. S3 Non-blocking TCP sockets
    CoreSyscallSpec { tenon_name: "net_tcp_listen", c_symbol: "tenon_net_tcp_listen", arity: 1, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    CoreSyscallSpec { tenon_name: "net_tcp_local_port", c_symbol: "tenon_net_tcp_local_port", arity: 1, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    CoreSyscallSpec { tenon_name: "net_tcp_accept", c_symbol: "tenon_net_tcp_accept", arity: 1, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    CoreSyscallSpec { tenon_name: "net_tcp_connect", c_symbol: "tenon_net_tcp_connect", arity: 1, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    CoreSyscallSpec { tenon_name: "net_set_nonblocking", c_symbol: "tenon_net_set_nonblocking", arity: 2, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Bool" },
    CoreSyscallSpec { tenon_name: "net_set_nodelay", c_symbol: "tenon_net_set_nodelay", arity: 2, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Bool" },
    CoreSyscallSpec { tenon_name: "net_last_would_block", c_symbol: "tenon_net_last_would_block", arity: 0, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Bool" },
    CoreSyscallSpec { tenon_name: "net_read", c_symbol: "tenon_net_read", arity: 2, returns_rc: true, returns_text: false, returns_float: false, ret_ty: "Bytes" },
    CoreSyscallSpec { tenon_name: "net_read_text", c_symbol: "tenon_net_read_text", arity: 2, returns_rc: true, returns_text: true, returns_float: false, ret_ty: "Text" },
    CoreSyscallSpec { tenon_name: "net_write", c_symbol: "tenon_net_write", arity: 2, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    CoreSyscallSpec { tenon_name: "net_write_text", c_symbol: "tenon_net_write_text", arity: 2, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    CoreSyscallSpec { tenon_name: "net_close", c_symbol: "tenon_net_close", arity: 1, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    // 8b. OS Event Reactor (kqueue/epoll) & Monotonic Clock
    CoreSyscallSpec { tenon_name: "reactor_new", c_symbol: "tenon_reactor_new", arity: 0, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    CoreSyscallSpec { tenon_name: "reactor_add", c_symbol: "tenon_reactor_add", arity: 4, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Bool" },
    CoreSyscallSpec { tenon_name: "reactor_del", c_symbol: "tenon_reactor_del", arity: 2, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Bool" },
    CoreSyscallSpec { tenon_name: "reactor_wait", c_symbol: "tenon_reactor_wait", arity: 3, returns_rc: true, returns_text: false, returns_float: false, ret_ty: "List[Int]" },
    CoreSyscallSpec { tenon_name: "reactor_close", c_symbol: "tenon_reactor_close", arity: 1, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    CoreSyscallSpec { tenon_name: "time_now_ms", c_symbol: "tenon_time_now_ms", arity: 0, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    CoreSyscallSpec { tenon_name: "time_now_ns", c_symbol: "tenon_time_now_ns", arity: 0, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    // 9. S3 SIMD vector operations
    CoreSyscallSpec { tenon_name: "simd_f32x4_new", c_symbol: "tenon_simd_f32x4_new", arity: 4, returns_rc: true, returns_text: false, returns_float: false, ret_ty: "Float32x4" },
    CoreSyscallSpec { tenon_name: "simd_f32x4_splat", c_symbol: "tenon_simd_f32x4_splat", arity: 1, returns_rc: true, returns_text: false, returns_float: false, ret_ty: "Float32x4" },
    CoreSyscallSpec { tenon_name: "simd_f32x4_lane", c_symbol: "tenon_simd_f32x4_lane", arity: 2, returns_rc: false, returns_text: false, returns_float: true, ret_ty: "Float32" },
    CoreSyscallSpec { tenon_name: "simd_f32x4_binop", c_symbol: "tenon_simd_f32x4_binop", arity: 3, returns_rc: true, returns_text: false, returns_float: false, ret_ty: "Float32x4" },
    CoreSyscallSpec { tenon_name: "simd_f32x4_fma", c_symbol: "tenon_simd_f32x4_fma", arity: 3, returns_rc: true, returns_text: false, returns_float: false, ret_ty: "Float32x4" },
    CoreSyscallSpec { tenon_name: "simd_f32x4_reduce_sum", c_symbol: "tenon_simd_f32x4_reduce_sum", arity: 1, returns_rc: false, returns_text: false, returns_float: true, ret_ty: "Float32" },
    CoreSyscallSpec { tenon_name: "simd_f32x4_dot", c_symbol: "tenon_simd_f32x4_dot", arity: 2, returns_rc: false, returns_text: false, returns_float: true, ret_ty: "Float32" },
    CoreSyscallSpec { tenon_name: "simd_f64x4_new", c_symbol: "tenon_simd_f64x4_new", arity: 4, returns_rc: true, returns_text: false, returns_float: false, ret_ty: "Float64x4" },
    CoreSyscallSpec { tenon_name: "simd_f64x4_splat", c_symbol: "tenon_simd_f64x4_splat", arity: 1, returns_rc: true, returns_text: false, returns_float: false, ret_ty: "Float64x4" },
    CoreSyscallSpec { tenon_name: "simd_f64x4_lane", c_symbol: "tenon_simd_f64x4_lane", arity: 2, returns_rc: false, returns_text: false, returns_float: true, ret_ty: "Float" },
    CoreSyscallSpec { tenon_name: "simd_f64x4_binop", c_symbol: "tenon_simd_f64x4_binop", arity: 3, returns_rc: true, returns_text: false, returns_float: false, ret_ty: "Float64x4" },
    CoreSyscallSpec { tenon_name: "simd_f64x4_fma", c_symbol: "tenon_simd_f64x4_fma", arity: 3, returns_rc: true, returns_text: false, returns_float: false, ret_ty: "Float64x4" },
    CoreSyscallSpec { tenon_name: "simd_f64x4_reduce_sum", c_symbol: "tenon_simd_f64x4_reduce_sum", arity: 1, returns_rc: false, returns_text: false, returns_float: true, ret_ty: "Float" },
    CoreSyscallSpec { tenon_name: "simd_f64x4_dot", c_symbol: "tenon_simd_f64x4_dot", arity: 2, returns_rc: false, returns_text: false, returns_float: true, ret_ty: "Float" },
    CoreSyscallSpec { tenon_name: "simd_i32x4_new", c_symbol: "tenon_simd_i32x4_new", arity: 4, returns_rc: true, returns_text: false, returns_float: false, ret_ty: "Int32x4" },
    CoreSyscallSpec { tenon_name: "simd_i32x4_splat", c_symbol: "tenon_simd_i32x4_splat", arity: 1, returns_rc: true, returns_text: false, returns_float: false, ret_ty: "Int32x4" },
    CoreSyscallSpec { tenon_name: "simd_i32x4_lane", c_symbol: "tenon_simd_i32x4_lane", arity: 2, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int32" },
    CoreSyscallSpec { tenon_name: "simd_i32x4_binop", c_symbol: "tenon_simd_i32x4_binop", arity: 3, returns_rc: true, returns_text: false, returns_float: false, ret_ty: "Int32x4" },
    CoreSyscallSpec { tenon_name: "simd_i32x4_reduce_sum", c_symbol: "tenon_simd_i32x4_reduce_sum", arity: 1, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int32" },
    CoreSyscallSpec { tenon_name: "simd_i64x4_new", c_symbol: "tenon_simd_i64x4_new", arity: 4, returns_rc: true, returns_text: false, returns_float: false, ret_ty: "Int64x4" },
    CoreSyscallSpec { tenon_name: "simd_i64x4_splat", c_symbol: "tenon_simd_i64x4_splat", arity: 1, returns_rc: true, returns_text: false, returns_float: false, ret_ty: "Int64x4" },
    CoreSyscallSpec { tenon_name: "simd_i64x4_lane", c_symbol: "tenon_simd_i64x4_lane", arity: 2, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    CoreSyscallSpec { tenon_name: "simd_i64x4_binop", c_symbol: "tenon_simd_i64x4_binop", arity: 3, returns_rc: true, returns_text: false, returns_float: false, ret_ty: "Int64x4" },
    CoreSyscallSpec { tenon_name: "simd_i64x4_reduce_sum", c_symbol: "tenon_simd_i64x4_reduce_sum", arity: 1, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
    CoreSyscallSpec { tenon_name: "simd_u8x16_splat", c_symbol: "tenon_simd_u8x16_splat", arity: 1, returns_rc: true, returns_text: false, returns_float: false, ret_ty: "UInt8x16" },
    CoreSyscallSpec { tenon_name: "simd_u8x16_lane", c_symbol: "tenon_simd_u8x16_lane", arity: 2, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "UInt8" },
    CoreSyscallSpec { tenon_name: "simd_u8x16_set_lane", c_symbol: "tenon_simd_u8x16_set_lane", arity: 3, returns_rc: true, returns_text: false, returns_float: false, ret_ty: "UInt8x16" },
    CoreSyscallSpec { tenon_name: "simd_u8x16_binop", c_symbol: "tenon_simd_u8x16_binop", arity: 3, returns_rc: true, returns_text: false, returns_float: false, ret_ty: "UInt8x16" },
    CoreSyscallSpec { tenon_name: "simd_u8x16_reduce_sum", c_symbol: "tenon_simd_u8x16_reduce_sum", arity: 1, returns_rc: false, returns_text: false, returns_float: false, ret_ty: "Int" },
];

pub(crate) fn named_ty(name: &str) -> TypeExpr {
    TypeExpr::Named {
        name: name.to_string(),
        version: None,
        args: Vec::new(),
        span: Span::default(),
    }
}

pub(crate) fn generic_ty(name: &str, args: Vec<TypeExpr>) -> TypeExpr {
    TypeExpr::Named {
        name: name.to_string(),
        version: None,
        args,
        span: Span::default(),
    }
}

pub(crate) fn resolve_alias_ty(ty: &TypeExpr, aliases: &BTreeMap<String, TypeExpr>) -> TypeExpr {
    if let TypeExpr::Named { name, .. } = ty {
        if let Some(base) = aliases.get(name) {
            return resolve_alias_ty(base, aliases);
        }
    }
    ty.clone()
}

pub(crate) fn is_rc_type_expr(
    ty: &TypeExpr,
    aliases: &BTreeMap<String, TypeExpr>,
    record_names: &BTreeSet<String>,
    enum_names: &BTreeSet<String>,
) -> bool {
    let resolved = resolve_alias_ty(ty, aliases);
    match resolved {
        TypeExpr::Dyn { .. }
        | TypeExpr::Tuple(_, _)
        | TypeExpr::Fn { .. }
        | TypeExpr::Array { .. } => true,
        TypeExpr::Named { name, .. } => {
            record_names.contains(&name)
                || enum_names.contains(&name)
                || SimdKind::from_name(&name).is_some()
                || matches!(
                    name.as_str(),
                    "Text"
                        | "List"
                        | "Option"
                        | "Result"
                        | "Range"
                        | "Bytes"
                        | "BytesSlice"
                        | "AtomicInt"
                        | "Mutex"
                        | "Channel"
                )
        }
    }
}

pub(crate) fn is_text_type_expr(ty: &TypeExpr, aliases: &BTreeMap<String, TypeExpr>) -> bool {
    matches!(resolve_alias_ty(ty, aliases), TypeExpr::Named { name, .. } if name == "Text")
}

pub(crate) fn is_float_type_expr(ty: &TypeExpr, aliases: &BTreeMap<String, TypeExpr>) -> bool {
    matches!(
        resolve_alias_ty(ty, aliases),
        TypeExpr::Named { name, .. } if matches!(name.as_str(), "Float" | "Float64" | "Float32" | "Float16" | "BFloat16")
    )
}

pub(crate) fn is_f32_type_expr(ty: &TypeExpr, aliases: &BTreeMap<String, TypeExpr>) -> bool {
    matches!(resolve_alias_ty(ty, aliases), TypeExpr::Named { name, .. } if name == "Float32")
}

pub(crate) fn scalar_kind_of_ty(
    ty: &TypeExpr,
    aliases: &BTreeMap<String, TypeExpr>,
) -> Option<ScalarKind> {
    match resolve_alias_ty(ty, aliases) {
        TypeExpr::Named { name, .. } => ScalarKind::from_name(&name),
        _ => None,
    }
}

#[derive(Debug, Clone)]
struct FieldLayout {
    name: String,
    ty: TypeExpr,
    preds: Vec<Expr>,
    is_rc: bool,
}

#[derive(Debug, Clone)]
struct RecordLayout {
    tag: i64,
    fields: Vec<FieldLayout>,
    rc_mask: i64,
    where_clauses: Vec<Expr>,
    c_layout: Option<StructByteLayout>,
}

#[derive(Debug, Clone)]
struct EnumVariantLayout {
    enum_name: String,
    payload_tys: Vec<TypeExpr>,
}

#[derive(Debug, Clone)]
struct FnMeta {
    params_inout: Vec<bool>,
    param_tys: Vec<TypeExpr>,
    ret_ty: Option<TypeExpr>,
    returns_rc: bool,
}

#[derive(Clone, Copy)]
struct RtFuncs {
    assert_fail: FuncId,
    rc_alloc: FuncId,
    rc_set_rc_mask: FuncId,
    rc_inc: FuncId,
    rc_dec: FuncId,
    rc_ensure_unique: FuncId,
    rc_get_field: FuncId,
    rc_set_field: FuncId,
    rc_set_field_raw: FuncId,
    rc_get_tag: FuncId,
    rc_get_len: FuncId,
    text_new: FuncId,
    text_eq: FuncId,
    str_concat: FuncId,
    list_push: FuncId,
    list_filled: FuncId,
    list_pop_shrink: FuncId,
    async_frame_alloc: FuncId,
    async_yield_create: FuncId,
    async_poll_frame: FuncId,
    async_block_on: FuncId,
    cstruct_alloc: FuncId,
    cstruct_data_ptr: FuncId,
    raw_memcpy: FuncId,
    volatile_read: FuncId,
    volatile_write: FuncId,
    f64_to_f16: FuncId,
    f16_to_f64: FuncId,
    f64_to_bf16: FuncId,
    bf16_to_f64: FuncId,
}

pub struct JitProgram {
    module: JITModule,
    func_ids: BTreeMap<String, FuncId>,
    pub compiled_fns: Vec<String>,
    pub derived_parsers: Vec<String>,
    _interned_strings: Vec<Box<[u8]>>,
}

impl JitProgram {
    pub fn compile(program: &Program) -> Result<Self, String> {
        let prepared = prepare_program_for_codegen(program);
        let program = &prepared.program;
        let dyn_interfaces = &prepared.dyn_interfaces;
        let vtable_ifaces: BTreeMap<String, String> = prepared
            .vtables
            .iter()
            .map(|v| (v.sym_name.clone(), v.interface.clone()))
            .collect();

        let mut flag_builder = settings::builder();
        flag_builder
            .set("use_colocated_libcalls", "false")
            .map_err(|e| e.to_string())?;
        flag_builder
            .set("is_pic", "false")
            .map_err(|e| e.to_string())?;
        let isa_builder = cranelift_native::builder().map_err(|e| e.to_string())?;
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| e.to_string())?;

        let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        jit_builder.symbol("tenon_rt_assert_fail", tenon_rt_assert_fail as *const u8);
        jit_builder.symbol("tenon_rt_print_int", tenon_rt_print_int as *const u8);
        jit_builder.symbol("tenon_rc_alloc", tenon_rc_alloc as *const u8);
        jit_builder.symbol("tenon_rc_set_rc_mask", tenon_rc_set_rc_mask as *const u8);
        jit_builder.symbol("tenon_rc_inc", tenon_rc_inc as *const u8);
        jit_builder.symbol("tenon_rc_dec", tenon_rc_dec as *const u8);
        jit_builder.symbol("tenon_rc_ensure_unique", tenon_rc_ensure_unique as *const u8);
        jit_builder.symbol("tenon_rc_get_field", tenon_rc_get_field as *const u8);
        jit_builder.symbol("tenon_rc_set_field", tenon_rc_set_field as *const u8);
        jit_builder.symbol("tenon_rc_set_field_raw", tenon_rc_set_field_raw as *const u8);
        jit_builder.symbol("tenon_rc_get_tag", tenon_rc_get_tag as *const u8);
        jit_builder.symbol("tenon_rc_get_len", tenon_rc_get_len as *const u8);
        jit_builder.symbol("tenon_text_new", tenon_text_new as *const u8);
        jit_builder.symbol("tenon_text_eq", tenon_text_eq as *const u8);
        jit_builder.symbol("tenon_list_push", tenon_list_push as *const u8);
        jit_builder.symbol("tenon_list_filled", tenon_list_filled as *const u8);
        jit_builder.symbol("tenon_list_pop_shrink", tenon_list_pop_shrink as *const u8);
        jit_builder.symbol("tenon_async_frame_alloc", tenon_async_frame_alloc as *const u8);
        jit_builder.symbol("tenon_async_yield_create", tenon_async_yield_create as *const u8);
        jit_builder.symbol("tenon_async_poll_frame", tenon_async_poll_frame as *const u8);
        jit_builder.symbol("tenon_async_block_on", tenon_async_block_on as *const u8);

        jit_builder.symbol("tenon_byte_at", syscalls::tenon_byte_at as *const u8);
        jit_builder.symbol("tenon_str_slice", syscalls::tenon_str_slice as *const u8);
        jit_builder.symbol("tenon_byte_to_text", syscalls::tenon_byte_to_text as *const u8);
        jit_builder.symbol("tenon_str_concat", syscalls::tenon_str_concat as *const u8);
        jit_builder.symbol("tenon_buf_new", syscalls::tenon_buf_new as *const u8);
        jit_builder.symbol("tenon_buf_push_str", syscalls::tenon_buf_push_str as *const u8);
        jit_builder.symbol("tenon_buf_push_byte", syscalls::tenon_buf_push_byte as *const u8);
        jit_builder.symbol("tenon_buf_finish", syscalls::tenon_buf_finish as *const u8);
        jit_builder.symbol("tenon_int_to_float", syscalls::tenon_int_to_float as *const u8);
        jit_builder.symbol("tenon_float_to_int", syscalls::tenon_float_to_int as *const u8);
        jit_builder.symbol("tenon_float_to_text", syscalls::tenon_float_to_text as *const u8);
        jit_builder.symbol("tenon_parse_float", syscalls::tenon_parse_float as *const u8);
        jit_builder.symbol("tenon_float_is_nan", syscalls::tenon_float_is_nan as *const u8);
        jit_builder.symbol("tenon_float_is_finite", syscalls::tenon_float_is_finite as *const u8);
        jit_builder.symbol("tenon_float_bits", syscalls::tenon_float_bits as *const u8);
        jit_builder.symbol("tenon_sys_write_stdout", syscalls::tenon_sys_write_stdout as *const u8);
        jit_builder.symbol("tenon_sys_read_line", syscalls::tenon_sys_read_line as *const u8);
        jit_builder.symbol("tenon_sys_is_eof", syscalls::tenon_sys_is_eof as *const u8);
        jit_builder.symbol("tenon_sys_env", syscalls::tenon_sys_env as *const u8);
        jit_builder.symbol("tenon_sys_cwd", syscalls::tenon_sys_cwd as *const u8);
        jit_builder.symbol("tenon_sys_read_file", syscalls::tenon_sys_read_file as *const u8);
        jit_builder.symbol("tenon_sys_file_ok", syscalls::tenon_sys_file_ok as *const u8);
        jit_builder.symbol("tenon_sys_write_file", syscalls::tenon_sys_write_file as *const u8);
        jit_builder.symbol("tenon_sys_exec", syscalls::tenon_sys_exec as *const u8);
        jit_builder.symbol("tenon_sys_exec_exit_code", syscalls::tenon_sys_exec_exit_code as *const u8);
        jit_builder.symbol("tenon_sys_http_open", syscalls::tenon_sys_http_open as *const u8);
        jit_builder.symbol("tenon_sys_http_sse_open", syscalls::tenon_sys_http_sse_open as *const u8);
        jit_builder.symbol("tenon_sys_http_sse_next_line", syscalls::tenon_sys_http_sse_next_line as *const u8);
        jit_builder.symbol("tenon_sys_http_sse_eof", syscalls::tenon_sys_http_sse_eof as *const u8);
        jit_builder.symbol("tenon_sys_http_sse_error", syscalls::tenon_sys_http_sse_error as *const u8);
        jit_builder.symbol("tenon_sys_http_sse_close", syscalls::tenon_sys_http_sse_close as *const u8);

        // Systems programming runtime symbols (S1 + S2 + S3)
        jit_builder.symbol("tenon_clz", systems::tenon_clz as *const u8);
        jit_builder.symbol("tenon_ctz", systems::tenon_ctz as *const u8);
        jit_builder.symbol("tenon_popcnt", systems::tenon_popcnt as *const u8);
        jit_builder.symbol("tenon_bswap16", systems::tenon_bswap16 as *const u8);
        jit_builder.symbol("tenon_bswap32", systems::tenon_bswap32 as *const u8);
        jit_builder.symbol("tenon_bswap64", systems::tenon_bswap64 as *const u8);
        jit_builder.symbol("tenon_rotl64", systems::tenon_rotl64 as *const u8);
        jit_builder.symbol("tenon_rotr64", systems::tenon_rotr64 as *const u8);
        jit_builder.symbol("tenon_f32_to_text", systems::tenon_f32_to_text as *const u8);
        jit_builder.symbol("tenon_parse_f32", systems::tenon_parse_f32 as *const u8);
        jit_builder.symbol("tenon_bytes_new", systems::tenon_bytes_new as *const u8);
        jit_builder.symbol("tenon_bytes_filled", systems::tenon_bytes_filled as *const u8);
        jit_builder.symbol("tenon_bytes_len", systems::tenon_bytes_len as *const u8);
        jit_builder.symbol("tenon_bytes_get", systems::tenon_bytes_get as *const u8);
        jit_builder.symbol("tenon_bytes_set", systems::tenon_bytes_set as *const u8);
        jit_builder.symbol("tenon_bytes_push", systems::tenon_bytes_push as *const u8);
        jit_builder.symbol("tenon_bytes_read_u16_le", systems::tenon_bytes_read_u16_le as *const u8);
        jit_builder.symbol("tenon_bytes_read_u16_be", systems::tenon_bytes_read_u16_be as *const u8);
        jit_builder.symbol("tenon_bytes_read_u32_le", systems::tenon_bytes_read_u32_le as *const u8);
        jit_builder.symbol("tenon_bytes_read_u32_be", systems::tenon_bytes_read_u32_be as *const u8);
        jit_builder.symbol("tenon_bytes_read_u64_le", systems::tenon_bytes_read_u64_le as *const u8);
        jit_builder.symbol("tenon_bytes_read_u64_be", systems::tenon_bytes_read_u64_be as *const u8);
        jit_builder.symbol("tenon_bytes_write_u16_le", systems::tenon_bytes_write_u16_le as *const u8);
        jit_builder.symbol("tenon_bytes_write_u16_be", systems::tenon_bytes_write_u16_be as *const u8);
        jit_builder.symbol("tenon_bytes_write_u32_le", systems::tenon_bytes_write_u32_le as *const u8);
        jit_builder.symbol("tenon_bytes_write_u32_be", systems::tenon_bytes_write_u32_be as *const u8);
        jit_builder.symbol("tenon_bytes_write_u64_le", systems::tenon_bytes_write_u64_le as *const u8);
        jit_builder.symbol("tenon_bytes_write_u64_be", systems::tenon_bytes_write_u64_be as *const u8);
        jit_builder.symbol("tenon_bytes_slice", systems::tenon_bytes_slice as *const u8);
        jit_builder.symbol("tenon_slice_len", systems::tenon_slice_len as *const u8);
        jit_builder.symbol("tenon_slice_get", systems::tenon_slice_get as *const u8);
        jit_builder.symbol("tenon_bytes_to_text", systems::tenon_bytes_to_text as *const u8);
        jit_builder.symbol("tenon_text_to_bytes", systems::tenon_text_to_bytes as *const u8);
        jit_builder.symbol("tenon_bytes_ptr", systems::tenon_bytes_ptr as *const u8);
        jit_builder.symbol("tenon_list_data_ptr", systems::tenon_list_data_ptr as *const u8);
        jit_builder.symbol("tenon_gpu_gid_x", systems::tenon_gpu_gid_x as *const u8);
        jit_builder.symbol("tenon_gpu_gid_y", systems::tenon_gpu_gid_y as *const u8);
        jit_builder.symbol("tenon_gpu_gid_z", systems::tenon_gpu_gid_z as *const u8);
        jit_builder.symbol("tenon_gpu_lid_x", systems::tenon_gpu_lid_x as *const u8);
        jit_builder.symbol("tenon_gpu_lid_y", systems::tenon_gpu_lid_y as *const u8);
        jit_builder.symbol("tenon_gpu_wgid_x", systems::tenon_gpu_wgid_x as *const u8);
        jit_builder.symbol("tenon_gpu_wgid_y", systems::tenon_gpu_wgid_y as *const u8);
        jit_builder.symbol("tenon_gpu_barrier", systems::tenon_gpu_barrier as *const u8);
        jit_builder.symbol("tenon_gpu_shared_f32", systems::tenon_gpu_shared_f32 as *const u8);
        jit_builder.symbol("tenon_text_cstr_ptr", systems::tenon_text_cstr_ptr as *const u8);
        jit_builder.symbol("tenon_cstr_to_text", systems::tenon_cstr_to_text as *const u8);
        jit_builder.symbol("tenon_cstruct_alloc", systems::tenon_cstruct_alloc as *const u8);
        jit_builder.symbol("tenon_cstruct_data_ptr", systems::tenon_cstruct_data_ptr as *const u8);
        jit_builder.symbol("tenon_raw_alloc", systems::tenon_raw_alloc as *const u8);
        jit_builder.symbol("tenon_raw_free", systems::tenon_raw_free as *const u8);
        jit_builder.symbol("tenon_raw_memcpy", systems::tenon_raw_memcpy as *const u8);
        jit_builder.symbol("tenon_raw_memset", systems::tenon_raw_memset as *const u8);
        jit_builder.symbol("tenon_raw_memcmp", systems::tenon_raw_memcmp as *const u8);
        jit_builder.symbol("tenon_volatile_read", systems::tenon_volatile_read as *const u8);
        jit_builder.symbol("tenon_volatile_write", systems::tenon_volatile_write as *const u8);
        jit_builder.symbol("tenon_f64_to_f16", systems::tenon_f64_to_f16 as *const u8);
        jit_builder.symbol("tenon_f16_to_f64", systems::tenon_f16_to_f64 as *const u8);
        jit_builder.symbol("tenon_f64_to_bf16", systems::tenon_f64_to_bf16 as *const u8);
        jit_builder.symbol("tenon_bf16_to_f64", systems::tenon_bf16_to_f64 as *const u8);
        jit_builder.symbol("tenon_thread_spawn", systems::tenon_thread_spawn as *const u8);
        jit_builder.symbol("tenon_thread_join", systems::tenon_thread_join as *const u8);
        jit_builder.symbol("tenon_thread_sleep_ms", systems::tenon_thread_sleep_ms as *const u8);
        jit_builder.symbol("tenon_thread_hw_concurrency", systems::tenon_thread_hw_concurrency as *const u8);
        jit_builder.symbol("tenon_atomic_new", systems::tenon_atomic_new as *const u8);
        jit_builder.symbol("tenon_atomic_load", systems::tenon_atomic_load as *const u8);
        jit_builder.symbol("tenon_atomic_store", systems::tenon_atomic_store as *const u8);
        jit_builder.symbol("tenon_atomic_fetch_add", systems::tenon_atomic_fetch_add as *const u8);
        jit_builder.symbol("tenon_atomic_fetch_sub", systems::tenon_atomic_fetch_sub as *const u8);
        jit_builder.symbol("tenon_atomic_cas", systems::tenon_atomic_cas as *const u8);
        jit_builder.symbol("tenon_atomic_swap", systems::tenon_atomic_swap as *const u8);
        jit_builder.symbol("tenon_mutex_new", systems::tenon_mutex_new as *const u8);
        jit_builder.symbol("tenon_mutex_lock", systems::tenon_mutex_lock as *const u8);
        jit_builder.symbol("tenon_mutex_unlock", systems::tenon_mutex_unlock as *const u8);
        jit_builder.symbol("tenon_mutex_get", systems::tenon_mutex_get as *const u8);
        jit_builder.symbol("tenon_mutex_set", systems::tenon_mutex_set as *const u8);
        jit_builder.symbol("tenon_chan_new", systems::tenon_chan_new as *const u8);
        jit_builder.symbol("tenon_chan_send", systems::tenon_chan_send as *const u8);
        jit_builder.symbol("tenon_chan_recv", systems::tenon_chan_recv as *const u8);
        jit_builder.symbol("tenon_chan_try_recv", systems::tenon_chan_try_recv as *const u8);
        jit_builder.symbol("tenon_chan_close", systems::tenon_chan_close as *const u8);
        jit_builder.symbol("tenon_net_tcp_listen", systems::tenon_net_tcp_listen as *const u8);
        jit_builder.symbol("tenon_net_tcp_local_port", systems::tenon_net_tcp_local_port as *const u8);
        jit_builder.symbol("tenon_net_tcp_accept", systems::tenon_net_tcp_accept as *const u8);
        jit_builder.symbol("tenon_net_tcp_connect", systems::tenon_net_tcp_connect as *const u8);
        jit_builder.symbol("tenon_net_set_nonblocking", systems::tenon_net_set_nonblocking as *const u8);
        jit_builder.symbol("tenon_net_set_nodelay", systems::tenon_net_set_nodelay as *const u8);
        jit_builder.symbol("tenon_net_last_would_block", systems::tenon_net_last_would_block as *const u8);
        jit_builder.symbol("tenon_net_read", systems::tenon_net_read as *const u8);
        jit_builder.symbol("tenon_net_read_text", systems::tenon_net_read_text as *const u8);
        jit_builder.symbol("tenon_net_write", systems::tenon_net_write as *const u8);
        jit_builder.symbol("tenon_net_write_text", systems::tenon_net_write_text as *const u8);
        jit_builder.symbol("tenon_net_close", systems::tenon_net_close as *const u8);
        jit_builder.symbol("tenon_reactor_new", systems::tenon_reactor_new as *const u8);
        jit_builder.symbol("tenon_reactor_add", systems::tenon_reactor_add as *const u8);
        jit_builder.symbol("tenon_reactor_del", systems::tenon_reactor_del as *const u8);
        jit_builder.symbol("tenon_reactor_wait", systems::tenon_reactor_wait as *const u8);
        jit_builder.symbol("tenon_reactor_close", systems::tenon_reactor_close as *const u8);
        jit_builder.symbol("tenon_time_now_ms", systems::tenon_time_now_ms as *const u8);
        jit_builder.symbol("tenon_time_now_ns", systems::tenon_time_now_ns as *const u8);
        jit_builder.symbol("tenon_simd_f32x4_new", systems::tenon_simd_f32x4_new as *const u8);
        jit_builder.symbol("tenon_simd_f32x4_splat", systems::tenon_simd_f32x4_splat as *const u8);
        jit_builder.symbol("tenon_simd_f32x4_lane", systems::tenon_simd_f32x4_lane as *const u8);
        jit_builder.symbol("tenon_simd_f32x4_binop", systems::tenon_simd_f32x4_binop as *const u8);
        jit_builder.symbol("tenon_simd_f32x4_fma", systems::tenon_simd_f32x4_fma as *const u8);
        jit_builder.symbol("tenon_simd_f32x4_reduce_sum", systems::tenon_simd_f32x4_reduce_sum as *const u8);
        jit_builder.symbol("tenon_simd_f32x4_dot", systems::tenon_simd_f32x4_dot as *const u8);
        jit_builder.symbol("tenon_simd_f64x4_new", systems::tenon_simd_f64x4_new as *const u8);
        jit_builder.symbol("tenon_simd_f64x4_splat", systems::tenon_simd_f64x4_splat as *const u8);
        jit_builder.symbol("tenon_simd_f64x4_lane", systems::tenon_simd_f64x4_lane as *const u8);
        jit_builder.symbol("tenon_simd_f64x4_binop", systems::tenon_simd_f64x4_binop as *const u8);
        jit_builder.symbol("tenon_simd_f64x4_fma", systems::tenon_simd_f64x4_fma as *const u8);
        jit_builder.symbol("tenon_simd_f64x4_reduce_sum", systems::tenon_simd_f64x4_reduce_sum as *const u8);
        jit_builder.symbol("tenon_simd_f64x4_dot", systems::tenon_simd_f64x4_dot as *const u8);
        jit_builder.symbol("tenon_simd_i32x4_new", systems::tenon_simd_i32x4_new as *const u8);
        jit_builder.symbol("tenon_simd_i32x4_splat", systems::tenon_simd_i32x4_splat as *const u8);
        jit_builder.symbol("tenon_simd_i32x4_lane", systems::tenon_simd_i32x4_lane as *const u8);
        jit_builder.symbol("tenon_simd_i32x4_binop", systems::tenon_simd_i32x4_binop as *const u8);
        jit_builder.symbol("tenon_simd_i32x4_reduce_sum", systems::tenon_simd_i32x4_reduce_sum as *const u8);
        jit_builder.symbol("tenon_simd_i64x4_new", systems::tenon_simd_i64x4_new as *const u8);
        jit_builder.symbol("tenon_simd_i64x4_splat", systems::tenon_simd_i64x4_splat as *const u8);
        jit_builder.symbol("tenon_simd_i64x4_lane", systems::tenon_simd_i64x4_lane as *const u8);
        jit_builder.symbol("tenon_simd_i64x4_binop", systems::tenon_simd_i64x4_binop as *const u8);
        jit_builder.symbol("tenon_simd_i64x4_reduce_sum", systems::tenon_simd_i64x4_reduce_sum as *const u8);
        jit_builder.symbol("tenon_simd_u8x16_splat", systems::tenon_simd_u8x16_splat as *const u8);
        jit_builder.symbol("tenon_simd_u8x16_lane", systems::tenon_simd_u8x16_lane as *const u8);
        jit_builder.symbol("tenon_simd_u8x16_set_lane", systems::tenon_simd_u8x16_set_lane as *const u8);
        jit_builder.symbol("tenon_simd_u8x16_binop", systems::tenon_simd_u8x16_binop as *const u8);
        jit_builder.symbol("tenon_simd_u8x16_reduce_sum", systems::tenon_simd_u8x16_reduce_sum as *const u8);

        // Register dynamic C `extern "C"` symbols via dlsym!
        let mut extern_fns: BTreeMap<String, ExternFnDecl> = BTreeMap::new();
        let mut resolved_externs: BTreeSet<String> = BTreeSet::new();
        for item in &program.items {
            if let Item::ExternFn(ef) = item {
                extern_fns.insert(ef.name.clone(), ef.clone());
                if let Some(addr) = systems::lookup_c_symbol(&ef.name) {
                    jit_builder.symbol(&ef.name, addr);
                    resolved_externs.insert(ef.name.clone());
                }
            }
        }

        let mut module = JITModule::new(jit_builder);

        let rt = declare_runtime_funcs(&mut module)?;
        let print_int_id = declare_sig(&mut module, "tenon_rt_print_int", 1, true)?;

        // Collect type aliases (bases & predicates)
        let mut alias_preds: BTreeMap<String, Expr> = BTreeMap::new();
        let mut alias_bases: BTreeMap<String, TypeExpr> = BTreeMap::new();
        for item in &program.items {
            if let Item::TypeAlias(ta) = item {
                alias_bases.insert(ta.name.clone(), ta.base.clone());
                if let Some(p) = &ta.predicate {
                    alias_preds.insert(ta.name.clone(), p.clone());
                }
            }
        }

        let layout_engine = LayoutEngine::from_program(program);
        let record_names: BTreeSet<String> = program
            .items
            .iter()
            .filter_map(|i| match i {
                Item::Record(r) if r.type_params.is_empty() => Some(r.name.clone()),
                Item::Union(u) => Some(u.name.clone()),
                _ => None,
            })
            .collect();
        let enum_names: BTreeSet<String> = program
            .items
            .iter()
            .filter_map(|i| match i {
                Item::Enum(e) => Some(e.name.clone()),
                _ => None,
            })
            .collect();

        // Collect Records, Unions & Enums (with full C byte layouts + variant payload types!)
        let mut records: BTreeMap<String, RecordLayout> = BTreeMap::new();
        let mut field_to_idx: BTreeMap<String, usize> = BTreeMap::new();
        let mut enum_tags: BTreeMap<String, i64> = BTreeMap::new();
        let mut enum_variants: BTreeMap<String, EnumVariantLayout> = BTreeMap::new();
        enum_tags.insert("Err".into(), TAG_ERR);
        enum_tags.insert("Ok".into(), TAG_OK);
        enum_tags.insert("None".into(), TAG_NONE);
        enum_tags.insert("Some".into(), TAG_SOME);

        let mut next_rec_tag = TAG_RECORD_BASE;
        let mut next_enum_tag = TAG_ENUM_BASE;
        for item in &program.items {
            match item {
                Item::Record(r) if r.type_params.is_empty() => {
                    let c_layout = if is_c_layout_attrs(&r.attrs) {
                        layout_engine.struct_layout(&r.name)
                    } else {
                        None
                    };
                    let mut fields = Vec::new();
                    let mut rc_mask = 0u64;
                    for (idx, f) in r.fields.iter().enumerate() {
                        let tname = f.ty.display_name();
                        let mut preds = Vec::new();
                        if let Some(ap) = alias_preds.get(&tname) {
                            preds.push(ap.clone());
                        }
                        if let Some(fp) = &f.predicate {
                            preds.push(fp.clone());
                        }
                        let is_rc = c_layout.is_none()
                            && is_rc_type_expr(&f.ty, &alias_bases, &record_names, &enum_names);
                        if is_rc && idx < 64 {
                            rc_mask |= 1u64 << idx;
                        }
                        fields.push(FieldLayout {
                            name: f.name.clone(),
                            ty: f.ty.clone(),
                            preds,
                            is_rc,
                        });
                        field_to_idx.entry(f.name.clone()).or_insert(idx);
                    }
                    records.insert(
                        r.name.clone(),
                        RecordLayout {
                            tag: next_rec_tag,
                            fields,
                            rc_mask: rc_mask as i64,
                            where_clauses: r.where_clauses.clone(),
                            c_layout,
                        },
                    );
                    next_rec_tag += 1;
                }
                Item::Union(u) => {
                    let c_layout = layout_engine.union_layout(&u.name);
                    let mut fields = Vec::new();
                    for (idx, f) in u.fields.iter().enumerate() {
                        fields.push(FieldLayout {
                            name: f.name.clone(),
                            ty: f.ty.clone(),
                            preds: Vec::new(),
                            is_rc: false,
                        });
                        field_to_idx.entry(f.name.clone()).or_insert(idx);
                    }
                    records.insert(
                        u.name.clone(),
                        RecordLayout {
                            tag: TAG_CSTRUCT,
                            fields,
                            rc_mask: 0,
                            where_clauses: Vec::new(),
                            c_layout,
                        },
                    );
                }
                Item::Enum(e) => {
                    for v in &e.variants {
                        let payload_tys: Vec<TypeExpr> =
                            v.fields.iter().map(|f| f.ty.clone()).collect();
                        let vl = EnumVariantLayout {
                            enum_name: e.name.clone(),
                            payload_tys,
                        };
                        enum_tags.insert(v.name.clone(), next_enum_tag);
                        enum_tags.insert(format!("{}.{}", e.name, v.name), next_enum_tag);
                        enum_variants.insert(v.name.clone(), vl.clone());
                        enum_variants.insert(format!("{}.{}", e.name, v.name), vl);
                        next_enum_tag += 1;
                    }
                }
                _ => {}
            }
        }

        let fns: Vec<&FnDecl> = program
            .items
            .iter()
            .filter_map(|i| match i {
                Item::Fn(f) if !f.is_async && f.type_params.is_empty() => Some(f),
                _ => None,
            })
            .collect();
        let async_fns: Vec<&FnDecl> = program
            .items
            .iter()
            .filter_map(|i| match i {
                Item::Fn(f) if f.is_async && f.type_params.is_empty() => Some(f),
                _ => None,
            })
            .collect();

        let mut func_ids = BTreeMap::new();
        let mut fn_metas: BTreeMap<String, FnMeta> = BTreeMap::new();
        func_ids.insert("print_int".into(), print_int_id);
        declare_core_syscalls(&mut module, rt.str_concat, rt.raw_memcpy, &mut func_ids, &mut fn_metas)?;

        // Declare resolved `extern "C"` functions with exact C ABI Cranelift signatures!
        let mut extern_func_ids: BTreeMap<String, FuncId> = BTreeMap::new();
        for (ename, ef) in &extern_fns {
            if resolved_externs.contains(ename) {
                let mut sig = module.make_signature();
                for p in &ef.params {
                    sig.params.push(AbiParam::new(c_abi_clif_type(&p.ty, &alias_bases)));
                }
                if let Some(rt_ty) = &ef.ret_ty {
                    let rb = resolve_alias_ty(rt_ty, &alias_bases);
                    if !matches!(&rb, TypeExpr::Named { name, .. } if name == "CVoid") {
                        sig.returns.push(AbiParam::new(c_abi_clif_type(&rb, &alias_bases)));
                    }
                }
                let fid = module
                    .declare_function(ename, Linkage::Import, &sig)
                    .map_err(|e| e.to_string())?;
                extern_func_ids.insert(ename.clone(), fid);
            }
            fn_metas.insert(
                ename.clone(),
                FnMeta {
                    params_inout: vec![false; ef.params.len()],
                    param_tys: ef.params.iter().map(|p| p.ty.clone()).collect(),
                    ret_ty: ef.ret_ty.clone(),
                    returns_rc: false,
                },
            );
        }

        for f in &async_fns {
            let mut create_sig = module.make_signature();
            for _ in &f.params {
                create_sig.params.push(AbiParam::new(types::I64));
            }
            create_sig.returns.push(AbiParam::new(types::I64));
            let create_id = module
                .declare_function(&f.name, Linkage::Export, &create_sig)
                .map_err(|e| e.to_string())?;
            func_ids.insert(f.name.clone(), create_id);
            fn_metas.insert(
                f.name.clone(),
                FnMeta {
                    params_inout: vec![false; f.params.len()],
                    param_tys: f.params.iter().map(|p| p.ty.clone()).collect(),
                    ret_ty: f.ret_ty.clone(),
                    returns_rc: true,
                },
            );

            let poll_name = format!("{}__poll", f.name);
            let mut poll_sig = module.make_signature();
            poll_sig.params.push(AbiParam::new(types::I64));
            poll_sig.returns.push(AbiParam::new(types::I64));
            let poll_id = module
                .declare_function(&poll_name, Linkage::Export, &poll_sig)
                .map_err(|e| e.to_string())?;
            func_ids.insert(poll_name, poll_id);
        }

        for f in &fns {
            let mut sig = module.make_signature();
            let mut params_inout = Vec::new();
            for p in &f.params {
                sig.params.push(AbiParam::new(types::I64));
                params_inout.push(p.is_inout);
            }
            sig.returns.push(AbiParam::new(types::I64));
            let id = module
                .declare_function(&f.name, Linkage::Export, &sig)
                .map_err(|e| e.to_string())?;
            func_ids.insert(f.name.clone(), id);
            fn_metas.insert(
                f.name.clone(),
                FnMeta {
                    params_inout,
                    param_tys: f.params.iter().map(|p| p.ty.clone()).collect(),
                    ret_ty: f.ret_ty.clone(),
                    returns_rc: f.ret_ty.as_ref().is_some_and(|t| {
                        is_rc_type_expr(t, &alias_bases, &record_names, &enum_names)
                    }),
                },
            );
        }

        let mut derived_parsers = Vec::new();
        for (rname, layout) in &records {
            if layout.fields.is_empty() || layout.c_layout.is_some() {
                continue;
            }
            let pname = format!("{}__parse", rname);
            let mut sig = module.make_signature();
            for _ in &layout.fields {
                sig.params.push(AbiParam::new(types::I64));
            }
            sig.returns.push(AbiParam::new(types::I64));
            let id = module
                .declare_function(&pname, Linkage::Export, &sig)
                .map_err(|e| e.to_string())?;
            func_ids.insert(pname.clone(), id);
            fn_metas.insert(
                pname.clone(),
                FnMeta {
                    params_inout: vec![false; layout.fields.len()],
                    param_tys: layout.fields.iter().map(|f| f.ty.clone()).collect(),
                    ret_ty: Some(generic_ty("Result", vec![named_ty(rname), named_ty("Int")])),
                    returns_rc: true,
                },
            );
            derived_parsers.push(pname);
        }

        // Build static read-only Cranelift data objects for every `dyn` vtable!
        let mut vtable_data_ids: BTreeMap<String, DataId> = BTreeMap::new();
        for vt in &prepared.vtables {
            let did = module
                .declare_data(&vt.sym_name, Linkage::Local, false, false)
                .map_err(|e| e.to_string())?;
            let mut dd = DataDescription::new();
            dd.define_zeroinit(vt.fn_names.len() * 8);
            for (slot, fn_name) in vt.fn_names.iter().enumerate() {
                let fid = *func_ids
                    .get(fn_name)
                    .ok_or_else(|| format!("vtable `{}` missing function `{}`", vt.sym_name, fn_name))?;
                let fref = module.declare_func_in_data(fid, &mut dd);
                dd.write_function_addr((slot * 8) as u32, fref);
            }
            module.define_data(did, &dd).map_err(|e| e.to_string())?;
            vtable_data_ids.insert(vt.sym_name.clone(), did);
        }

        let mut interned_strings: Vec<Box<[u8]>> = Vec::new();
        let mut ctx = module.make_context();
        let mut fb_ctx = FunctionBuilderContext::new();
        let mut compiled_fns = Vec::new();

        for f in &fns {
            let id = func_ids[&f.name];
            ctx.func.signature = module
                .declarations()
                .get_function_decl(id)
                .signature
                .clone();

            {
                let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);
                let entry_block = builder.create_block();
                builder.append_block_params_for_function_params(entry_block);
                builder.switch_to_block(entry_block);
                builder.seal_block(entry_block);

                let mut vars = BTreeMap::new();
                let mut var_types = BTreeMap::new();
                let mut inout_vars = BTreeSet::new();
                let mut borrowed_rc_vars = BTreeSet::new();

                for (idx, p) in f.params.iter().enumerate() {
                    let var = builder.declare_var(types::I64);
                    let val = builder.block_params(entry_block)[idx];
                    builder.def_var(var, val);
                    vars.insert(p.name.clone(), var);
                    var_types.insert(p.name.clone(), p.ty.clone());
                    if is_rc_type_expr(&p.ty, &alias_bases, &record_names, &enum_names) {
                        borrowed_rc_vars.insert(p.name.clone());
                    }
                    if p.is_inout {
                        inout_vars.insert(p.name.clone());
                    }
                }

                let fn_returns_rc = fn_metas.get(&f.name).is_some_and(|m| m.returns_rc);

                let mut lower = FnLower {
                    module: &mut module,
                    builder: &mut builder,
                    rt,
                    func_ids: &func_ids,
                    fn_metas: &fn_metas,
                    alias_bases: &alias_bases,
                    records: &records,
                    record_names: &record_names,
                    enum_names: &enum_names,
                    field_to_idx: &field_to_idx,
                    enum_tags: &enum_tags,
                    enum_variants: &enum_variants,
                    dyn_interfaces,
                    vtable_data_ids: &vtable_data_ids,
                    vtable_ifaces: &vtable_ifaces,
                    layout_engine: &layout_engine,
                    extern_fns: &extern_fns,
                    extern_func_ids: &extern_func_ids,
                    interned_strings: &mut interned_strings,
                    vars,
                    var_types,
                    inout_vars,
                    owned_rc_vars: BTreeSet::new(),
                    borrowed_rc_vars,
                    rc_scopes: Vec::new(),
                    temp_rc_scopes: Vec::new(),
                    loop_stack: Vec::new(),
                    fn_returns_rc,
                    fn_ret_ty: f.ret_ty.clone(),
                    self_var: None,
                    terminated: false,
                };
                let ret_val = lower.lower_block(&f.body, fn_returns_rc, f.ret_ty.as_ref())?;
                if !lower.terminated {
                    lower.builder.ins().return_(&[ret_val]);
                }
                builder.finalize();
            }

            module
                .define_function(id, &mut ctx)
                .map_err(|e| format!("Cranelift codegen failed for `{}`: {}", f.name, e))?;
            module.clear_context(&mut ctx);
            compiled_fns.push(f.name.clone());
        }

        // Auto-derived `<Record>__parse`
        for (rname, layout) in &records {
            if layout.fields.is_empty() || layout.c_layout.is_some() {
                continue;
            }
            let pname = format!("{}__parse", rname);
            let id = func_ids[&pname];
            ctx.func.signature = module
                .declarations()
                .get_function_decl(id)
                .signature
                .clone();

            {
                let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);
                let entry_block = builder.create_block();
                builder.append_block_params_for_function_params(entry_block);
                builder.switch_to_block(entry_block);
                builder.seal_block(entry_block);

                let mut vars = BTreeMap::new();
                let mut var_types = BTreeMap::new();
                let mut borrowed_rc_vars = BTreeSet::new();
                for (idx, fl) in layout.fields.iter().enumerate() {
                    let var = builder.declare_var(types::I64);
                    let val = builder.block_params(entry_block)[idx];
                    builder.def_var(var, val);
                    vars.insert(fl.name.clone(), var);
                    var_types.insert(fl.name.clone(), fl.ty.clone());
                    if fl.is_rc {
                        borrowed_rc_vars.insert(fl.name.clone());
                    }
                }

                let mut lower = FnLower {
                    module: &mut module,
                    builder: &mut builder,
                    rt,
                    func_ids: &func_ids,
                    fn_metas: &fn_metas,
                    alias_bases: &alias_bases,
                    records: &records,
                    record_names: &record_names,
                    enum_names: &enum_names,
                    field_to_idx: &field_to_idx,
                    enum_tags: &enum_tags,
                    enum_variants: &enum_variants,
                    dyn_interfaces,
                    vtable_data_ids: &vtable_data_ids,
                    vtable_ifaces: &vtable_ifaces,
                    layout_engine: &layout_engine,
                    extern_fns: &extern_fns,
                    extern_func_ids: &extern_func_ids,
                    interned_strings: &mut interned_strings,
                    vars,
                    var_types,
                    inout_vars: BTreeSet::new(),
                    owned_rc_vars: BTreeSet::new(),
                    borrowed_rc_vars,
                    rc_scopes: Vec::new(),
                    temp_rc_scopes: Vec::new(),
                    loop_stack: Vec::new(),
                    fn_returns_rc: true,
                    fn_ret_ty: None,
                    self_var: None,
                    terminated: false,
                };

                for (idx, fl) in layout.fields.iter().enumerate() {
                    for pred in &fl.preds {
                        lower.self_var = Some(fl.name.clone());
                        let mut temps = Vec::new();
                        let cond_val = lower.lower_expr_borrowed(pred, &mut temps)?;
                        lower.flush_temps(&temps);
                        lower.self_var = None;
                        let zero = lower.builder.ins().iconst(types::I64, 0);
                        let ok = lower.builder.ins().icmp(IntCC::NotEqual, cond_val, zero);
                        let fail_b = lower.builder.create_block();
                        let next_b = lower.builder.create_block();
                        lower.builder.ins().brif(ok, next_b, &[], fail_b, &[]);

                        lower.builder.switch_to_block(fail_b);
                        lower.builder.seal_block(fail_b);
                        let err_obj = lower.call_rt_2(rt.rc_alloc, TAG_ERR, 1);
                        let code_val = lower.builder.ins().iconst(types::I64, (idx + 1) as i64);
                        lower.call_rt_3_void(rt.rc_set_field, err_obj, 0, code_val);
                        lower.builder.ins().return_(&[err_obj]);

                        lower.builder.switch_to_block(next_b);
                        lower.builder.seal_block(next_b);
                    }
                }

                for wc in &layout.where_clauses {
                    let mut temps = Vec::new();
                    let cond_val = lower.lower_expr_borrowed(wc, &mut temps)?;
                    lower.flush_temps(&temps);
                    let zero = lower.builder.ins().iconst(types::I64, 0);
                    let ok = lower.builder.ins().icmp(IntCC::NotEqual, cond_val, zero);
                    let fail_b = lower.builder.create_block();
                    let next_b = lower.builder.create_block();
                    lower.builder.ins().brif(ok, next_b, &[], fail_b, &[]);

                    lower.builder.switch_to_block(fail_b);
                    lower.builder.seal_block(fail_b);
                    let err_obj = lower.call_rt_2(rt.rc_alloc, TAG_ERR, 1);
                    let code_val = lower.builder.ins().iconst(types::I64, 99);
                    lower.call_rt_3_void(rt.rc_set_field, err_obj, 0, code_val);
                    lower.builder.ins().return_(&[err_obj]);

                    lower.builder.switch_to_block(next_b);
                    lower.builder.seal_block(next_b);
                }

                let rec_obj = lower.call_rt_2(rt.rc_alloc, layout.tag, layout.fields.len() as i64);
                if layout.rc_mask != 0 {
                    let mv = lower.builder.ins().iconst(types::I64, layout.rc_mask);
                    lower.call_rt_2_void(rt.rc_set_rc_mask, rec_obj, mv);
                }
                for (idx, fl) in layout.fields.iter().enumerate() {
                    let mut v = lower.builder.use_var(lower.vars[&fl.name]);
                    if fl.is_rc {
                        v = lower.call_rt_1(rt.rc_inc, v);
                    }
                    lower.call_rt_3_void(rt.rc_set_field, rec_obj, idx as i64, v);
                }
                let ok_obj = lower.call_rt_2(rt.rc_alloc, TAG_OK, 1);
                let one = lower.builder.ins().iconst(types::I64, 1);
                lower.call_rt_2_void(rt.rc_set_rc_mask, ok_obj, one);
                lower.call_rt_3_void(rt.rc_set_field, ok_obj, 0, rec_obj);
                lower.builder.ins().return_(&[ok_obj]);
                builder.finalize();
            }

            module
                .define_function(id, &mut ctx)
                .map_err(|e| format!("Cranelift codegen failed for derived `{}`: {}", pname, e))?;
            module.clear_context(&mut ctx);
            compiled_fns.push(pname);
        }

        // Stackless `async fn` state machines
        for f in &async_fns {
            let poll_name = format!("{}__poll", f.name);
            let create_id = func_ids[&f.name];
            let poll_id = func_ids[&poll_name];

            let mut frame_slots: BTreeMap<String, i64> = BTreeMap::new();
            let mut next_slot = 4i64;
            for p in &f.params {
                frame_slots.insert(p.name.clone(), next_slot);
                next_slot += 1;
            }
            for s in &f.body.stmts {
                if let Stmt::Let { name, .. } = s {
                    if !frame_slots.contains_key(name) {
                        frame_slots.insert(name.clone(), next_slot);
                        next_slot += 1;
                    }
                }
            }
            let segments = split_async_segments(&f.body, &mut frame_slots, &mut next_slot);
            let total_slots = next_slot;

            ctx.func.signature = module
                .declarations()
                .get_function_decl(create_id)
                .signature
                .clone();
            {
                let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);
                let entry = builder.create_block();
                builder.append_block_params_for_function_params(entry);
                builder.switch_to_block(entry);
                builder.seal_block(entry);

                let poll_local = module.declare_func_in_func(poll_id, builder.func);
                let poll_ptr = builder.ins().func_addr(types::I64, poll_local);
                let slots_v = builder.ins().iconst(types::I64, total_slots);
                let alloc_callee = module.declare_func_in_func(rt.async_frame_alloc, builder.func);
                let call = builder.ins().call(alloc_callee, &[slots_v, poll_ptr]);
                let frame = builder.inst_results(call)[0];

                let set_callee = module.declare_func_in_func(rt.rc_set_field, builder.func);
                for (idx, p) in f.params.iter().enumerate() {
                    let pv = builder.block_params(entry)[idx];
                    let slot = builder.ins().iconst(types::I64, frame_slots[&p.name]);
                    builder.ins().call(set_callee, &[frame, slot, pv]);
                }
                builder.ins().return_(&[frame]);
                builder.finalize();
            }
            module
                .define_function(create_id, &mut ctx)
                .map_err(|e| format!("Cranelift async ctor failed for `{}`: {}", f.name, e))?;
            module.clear_context(&mut ctx);

            ctx.func.signature = module
                .declarations()
                .get_function_decl(poll_id)
                .signature
                .clone();
            {
                let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);
                let entry = builder.create_block();
                builder.append_block_params_for_function_params(entry);
                builder.switch_to_block(entry);
                let frame = builder.block_params(entry)[0];

                let mut vars = BTreeMap::new();
                let mut lower = FnLower {
                    module: &mut module,
                    builder: &mut builder,
                    rt,
                    func_ids: &func_ids,
                    fn_metas: &fn_metas,
                    alias_bases: &alias_bases,
                    records: &records,
                    record_names: &record_names,
                    enum_names: &enum_names,
                    field_to_idx: &field_to_idx,
                    enum_tags: &enum_tags,
                    enum_variants: &enum_variants,
                    dyn_interfaces,
                    vtable_data_ids: &vtable_data_ids,
                    vtable_ifaces: &vtable_ifaces,
                    layout_engine: &layout_engine,
                    extern_fns: &extern_fns,
                    extern_func_ids: &extern_func_ids,
                    interned_strings: &mut interned_strings,
                    vars: BTreeMap::new(),
                    var_types: BTreeMap::new(),
                    inout_vars: BTreeSet::new(),
                    owned_rc_vars: BTreeSet::new(),
                    borrowed_rc_vars: BTreeSet::new(),
                    rc_scopes: Vec::new(),
                    temp_rc_scopes: Vec::new(),
                    loop_stack: Vec::new(),
                    fn_returns_rc: false,
                    fn_ret_ty: None,
                    self_var: None,
                    terminated: false,
                };

                let z0 = lower.builder.ins().iconst(types::I64, 0);
                let cur_state = lower.call_rt_2_vals(rt.rc_get_field, frame, z0);
                for (vname, &slot) in &frame_slots {
                    let sv = lower.builder.ins().iconst(types::I64, slot);
                    let val = lower.call_rt_2_vals(rt.rc_get_field, frame, sv);
                    let var = lower.builder.declare_var(types::I64);
                    lower.builder.def_var(var, val);
                    lower.vars.insert(vname.clone(), var);
                    vars.insert(vname.clone(), var);
                }

                let n_awaits = segments.await_steps.len();
                let mut start_blocks = Vec::new();
                let mut poll_blocks = Vec::new();
                for _ in 0..n_awaits {
                    start_blocks.push(lower.builder.create_block());
                    poll_blocks.push(lower.builder.create_block());
                }
                let final_block = lower.builder.create_block();

                for s in 0..n_awaits {
                    let st_start = lower.builder.ins().iconst(types::I64, (2 * s) as i64);
                    let cmp_s = lower.builder.ins().icmp(IntCC::Equal, cur_state, st_start);
                    let next_chk1 = lower.builder.create_block();
                    lower
                        .builder
                        .ins()
                        .brif(cmp_s, start_blocks[s], &[], next_chk1, &[]);
                    lower.builder.switch_to_block(next_chk1);

                    let st_poll = lower.builder.ins().iconst(types::I64, (2 * s + 1) as i64);
                    let cmp_p = lower.builder.ins().icmp(IntCC::Equal, cur_state, st_poll);
                    let next_chk2 = lower.builder.create_block();
                    lower
                        .builder
                        .ins()
                        .brif(cmp_p, poll_blocks[s], &[], next_chk2, &[]);
                    lower.builder.switch_to_block(next_chk2);
                }
                lower.builder.ins().jump(final_block, &[]);

                for (s, step) in segments.await_steps.iter().enumerate() {
                    lower.builder.switch_to_block(start_blocks[s]);
                    lower.terminated = false;
                    let pre_blk = Block {
                        stmts: step.pre_stmts.clone(),
                        tail: None,
                        span: f.span,
                    };
                    let _ = lower.lower_block(&pre_blk, false, None)?;
                    for (vname, &slot) in &frame_slots {
                        let cur_v = lower.builder.use_var(vars[vname]);
                        lower.call_rt_3_void(rt.rc_set_field, frame, slot, cur_v);
                    }
                    let child_frame = lower.lower_expr_owned(&step.child_expr, None)?;
                    lower.call_rt_3_void(rt.rc_set_field, frame, 3, child_frame);
                    let poll_state = lower.builder.ins().iconst(types::I64, (2 * s + 1) as i64);
                    lower.call_rt_3_void(rt.rc_set_field, frame, 0, poll_state);
                    lower.builder.ins().jump(poll_blocks[s], &[]);

                    lower.builder.switch_to_block(poll_blocks[s]);
                    lower.terminated = false;
                    let slot3 = lower.builder.ins().iconst(types::I64, 3);
                    let cframe = lower.call_rt_2_vals(rt.rc_get_field, frame, slot3);
                    let ready = lower.call_rt_1(rt.async_poll_frame, cframe);
                    let zero = lower.builder.ins().iconst(types::I64, 0);
                    let is_ready = lower.builder.ins().icmp(IntCC::NotEqual, ready, zero);
                    let resume_b = lower.builder.create_block();
                    let pending_b = lower.builder.create_block();
                    lower
                        .builder
                        .ins()
                        .brif(is_ready, resume_b, &[], pending_b, &[]);

                    lower.builder.switch_to_block(pending_b);
                    let pend_ret = lower.builder.ins().iconst(types::I64, 0);
                    lower.builder.ins().return_(&[pend_ret]);

                    lower.builder.switch_to_block(resume_b);
                    let slot2 = lower.builder.ins().iconst(types::I64, 2);
                    let child_res = lower.call_rt_2_vals(rt.rc_get_field, cframe, slot2);
                    lower.call_rt_1_void(rt.rc_dec, cframe);
                    let zclear = lower.builder.ins().iconst(types::I64, 0);
                    lower.call_rt_3_void(rt.rc_set_field, frame, 3, zclear);
                    if let Some(bname) = &step.bind_var {
                        let slot = frame_slots[bname];
                        lower.call_rt_3_void(rt.rc_set_field, frame, slot, child_res);
                        lower.builder.def_var(vars[bname], child_res);
                    }
                    let next_state = lower.builder.ins().iconst(types::I64, (2 * (s + 1)) as i64);
                    lower.call_rt_3_void(rt.rc_set_field, frame, 0, next_state);
                    let target_b = if s + 1 < n_awaits {
                        start_blocks[s + 1]
                    } else {
                        final_block
                    };
                    lower.builder.ins().jump(target_b, &[]);
                }

                lower.builder.switch_to_block(final_block);
                lower.terminated = false;
                let final_blk = Block {
                    stmts: segments.final_stmts.clone(),
                    tail: segments.final_tail.clone().map(Box::new),
                    span: f.span,
                };
                let final_val = lower.lower_block(&final_blk, true, None)?;
                lower.call_rt_3_void(rt.rc_set_field, frame, 2, final_val);
                let done_st = lower.builder.ins().iconst(types::I64, -1);
                lower.call_rt_3_void(rt.rc_set_field, frame, 0, done_st);
                let ready_ret = lower.builder.ins().iconst(types::I64, 1);
                lower.builder.ins().return_(&[ready_ret]);

                lower.builder.seal_all_blocks();
                builder.finalize();
            }
            module
                .define_function(poll_id, &mut ctx)
                .map_err(|e| format!("Cranelift async poll failed for `{}`: {}", f.name, e))?;
            module.clear_context(&mut ctx);
            compiled_fns.push(f.name.clone());
            compiled_fns.push(poll_name);
        }

        module.finalize_definitions().map_err(|e| e.to_string())?;
        Ok(Self {
            module,
            func_ids,
            compiled_fns,
            derived_parsers,
            _interned_strings: interned_strings,
        })
    }

    pub fn call_0(&self, name: &str) -> Result<i64, String> {
        perceus::clear_assert_failed();
        let id = *self
            .func_ids
            .get(name)
            .ok_or_else(|| format!("function `{}` not compiled", name))?;
        let ptr = self.module.get_finalized_function(id);
        let f: extern "C" fn() -> i64 = unsafe { std::mem::transmute(ptr) };
        let res = f();
        let failed = perceus::last_assert_failed();
        if failed != 0 {
            return Err(format!("assertion failed at line {}", failed));
        }
        Ok(res)
    }

    pub fn call_2(&self, name: &str, a: i64, b: i64) -> Result<i64, String> {
        perceus::clear_assert_failed();
        let id = *self
            .func_ids
            .get(name)
            .ok_or_else(|| format!("function `{}` not compiled", name))?;
        let ptr = self.module.get_finalized_function(id);
        let f: extern "C" fn(i64, i64) -> i64 = unsafe { std::mem::transmute(ptr) };
        let res = f(a, b);
        let failed = perceus::last_assert_failed();
        if failed != 0 {
            return Err(format!("assertion failed at line {}", failed));
        }
        Ok(res)
    }

    pub fn last_printed() -> i64 {
        perceus::last_printed()
    }
}

fn declare_sig(
    module: &mut JITModule,
    name: &str,
    n_params: usize,
    has_ret: bool,
) -> Result<FuncId, String> {
    let mut sig = module.make_signature();
    for _ in 0..n_params {
        sig.params.push(AbiParam::new(types::I64));
    }
    if has_ret {
        sig.returns.push(AbiParam::new(types::I64));
    }
    module
        .declare_function(name, Linkage::Import, &sig)
        .map_err(|e| e.to_string())
}

fn declare_runtime_funcs(module: &mut JITModule) -> Result<RtFuncs, String> {
    Ok(RtFuncs {
        assert_fail: declare_sig(module, "tenon_rt_assert_fail", 1, false)?,
        rc_alloc: declare_sig(module, "tenon_rc_alloc", 2, true)?,
        rc_set_rc_mask: declare_sig(module, "tenon_rc_set_rc_mask", 2, false)?,
        rc_inc: declare_sig(module, "tenon_rc_inc", 1, true)?,
        rc_dec: declare_sig(module, "tenon_rc_dec", 1, false)?,
        rc_ensure_unique: declare_sig(module, "tenon_rc_ensure_unique", 1, true)?,
        rc_get_field: declare_sig(module, "tenon_rc_get_field", 2, true)?,
        rc_set_field: declare_sig(module, "tenon_rc_set_field", 3, false)?,
        rc_set_field_raw: declare_sig(module, "tenon_rc_set_field_raw", 3, false)?,
        rc_get_tag: declare_sig(module, "tenon_rc_get_tag", 1, true)?,
        rc_get_len: declare_sig(module, "tenon_rc_get_len", 1, true)?,
        text_new: declare_sig(module, "tenon_text_new", 2, true)?,
        text_eq: declare_sig(module, "tenon_text_eq", 2, true)?,
        str_concat: declare_sig(module, "tenon_str_concat", 2, true)?,
        list_push: declare_sig(module, "tenon_list_push", 2, true)?,
        list_filled: declare_sig(module, "tenon_list_filled", 3, true)?,
        list_pop_shrink: declare_sig(module, "tenon_list_pop_shrink", 1, true)?,
        async_frame_alloc: declare_sig(module, "tenon_async_frame_alloc", 2, true)?,
        async_yield_create: declare_sig(module, "tenon_async_yield_create", 1, true)?,
        async_poll_frame: declare_sig(module, "tenon_async_poll_frame", 1, true)?,
        async_block_on: declare_sig(module, "tenon_async_block_on", 1, true)?,
        cstruct_alloc: declare_sig(module, "tenon_cstruct_alloc", 1, true)?,
        cstruct_data_ptr: declare_sig(module, "tenon_cstruct_data_ptr", 1, true)?,
        raw_memcpy: declare_sig(module, "tenon_raw_memcpy", 3, true)?,
        volatile_read: declare_sig(module, "tenon_volatile_read", 2, true)?,
        volatile_write: declare_sig(module, "tenon_volatile_write", 3, true)?,
        f64_to_f16: declare_sig(module, "tenon_f64_to_f16", 1, true)?,
        f16_to_f64: declare_sig(module, "tenon_f16_to_f64", 1, true)?,
        f64_to_bf16: declare_sig(module, "tenon_f64_to_bf16", 1, true)?,
        bf16_to_f64: declare_sig(module, "tenon_bf16_to_f64", 1, true)?,
    })
}

fn c_abi_clif_type(ty: &TypeExpr, aliases: &BTreeMap<String, TypeExpr>) -> Type {
    match resolve_alias_ty(ty, aliases) {
        TypeExpr::Named { name, .. } => match ScalarKind::from_name(&name) {
            Some(ScalarKind::I8 | ScalarKind::U8) => types::I8,
            Some(ScalarKind::I16 | ScalarKind::U16 | ScalarKind::F16 | ScalarKind::BF16) => types::I16,
            Some(ScalarKind::I32 | ScalarKind::U32) => types::I32,
            Some(ScalarKind::F32) => types::F32,
            Some(ScalarKind::F64) => types::F64,
            _ if name == "Bool" => types::I8,
            _ => types::I64,
        },
        _ => types::I64,
    }
}

pub(crate) fn parse_syscall_ret_ty(spec: &str) -> TypeExpr {
    if let Some((head, rest)) = spec.split_once('[') {
        if let Some(inner) = rest.strip_suffix(']') {
            let args = inner
                .split(',')
                .map(|a| parse_syscall_ret_ty(a.trim()))
                .collect();
            return generic_ty(head.trim(), args);
        }
    }
    named_ty(spec)
}

fn declare_core_syscalls(
    module: &mut JITModule,
    str_concat_id: FuncId,
    raw_memcpy_id: FuncId,
    func_ids: &mut BTreeMap<String, FuncId>,
    fn_metas: &mut BTreeMap<String, FnMeta>,
) -> Result<(), String> {
    for spec in CORE_SYSCALLS {
        let fid = if spec.c_symbol == "tenon_str_concat" {
            str_concat_id
        } else if spec.c_symbol == "tenon_raw_memcpy" {
            raw_memcpy_id
        } else {
            declare_sig(module, spec.c_symbol, spec.arity, true)?
        };
        func_ids.insert(spec.tenon_name.to_string(), fid);
        fn_metas.insert(
            spec.tenon_name.to_string(),
            FnMeta {
                params_inout: vec![false; spec.arity],
                param_tys: vec![named_ty("Any"); spec.arity],
                ret_ty: Some(parse_syscall_ret_ty(spec.ret_ty)),
                returns_rc: spec.returns_rc,
            },
        );
    }
    Ok(())
}

pub(crate) struct AsyncAwaitStep {
    pub pre_stmts: Vec<Stmt>,
    pub bind_var: Option<String>,
    pub child_expr: Expr,
}

pub(crate) struct AsyncSegments {
    pub await_steps: Vec<AsyncAwaitStep>,
    pub final_stmts: Vec<Stmt>,
    pub final_tail: Option<Expr>,
}

pub(crate) fn split_async_segments(
    body: &Block,
    frame_slots: &mut BTreeMap<String, i64>,
    next_slot: &mut i64,
) -> AsyncSegments {
    let mut await_steps = Vec::new();
    let mut cur_stmts = Vec::new();

    for stmt in &body.stmts {
        match stmt {
            Stmt::Let {
                name,
                value:
                    Expr {
                        kind: ExprKind::Await(inner),
                        ..
                    },
                ..
            } => {
                if !frame_slots.contains_key(name) {
                    frame_slots.insert(name.clone(), *next_slot);
                    *next_slot += 1;
                }
                await_steps.push(AsyncAwaitStep {
                    pre_stmts: std::mem::take(&mut cur_stmts),
                    bind_var: Some(name.clone()),
                    child_expr: (**inner).clone(),
                });
            }
            Stmt::Expr {
                expr:
                    Expr {
                        kind: ExprKind::Await(inner),
                        ..
                    },
                ..
            } => {
                await_steps.push(AsyncAwaitStep {
                    pre_stmts: std::mem::take(&mut cur_stmts),
                    bind_var: None,
                    child_expr: (**inner).clone(),
                });
            }
            other => cur_stmts.push(other.clone()),
        }
    }

    let final_tail = match body.tail.as_deref() {
        Some(Expr {
            kind: ExprKind::Await(inner),
            span,
        }) => {
            let synth = "__await_tail".to_string();
            if !frame_slots.contains_key(&synth) {
                frame_slots.insert(synth.clone(), *next_slot);
                *next_slot += 1;
            }
            await_steps.push(AsyncAwaitStep {
                pre_stmts: std::mem::take(&mut cur_stmts),
                bind_var: Some(synth.clone()),
                child_expr: (**inner).clone(),
            });
            Some(Expr {
                kind: ExprKind::Ident(synth),
                span: *span,
            })
        }
        Some(e) => Some(e.clone()),
        None => None,
    };

    AsyncSegments {
        await_steps,
        final_stmts: cur_stmts,
        final_tail,
    }
}

struct LoopCtx {
    hdr_blk: ClifBlock,
    exit_blk: ClifBlock,
    rc_scope_depth: usize,
    temp_scope_depth: usize,
}

struct SavedLexical {
    vars: BTreeMap<String, Variable>,
    var_types: BTreeMap<String, TypeExpr>,
    owned_rc_vars: BTreeSet<String>,
    borrowed_rc_vars: BTreeSet<String>,
}

struct FnLower<'a, 'b> {
    module: &'a mut JITModule,
    builder: &'a mut FunctionBuilder<'b>,
    rt: RtFuncs,
    func_ids: &'a BTreeMap<String, FuncId>,
    fn_metas: &'a BTreeMap<String, FnMeta>,
    alias_bases: &'a BTreeMap<String, TypeExpr>,
    records: &'a BTreeMap<String, RecordLayout>,
    record_names: &'a BTreeSet<String>,
    enum_names: &'a BTreeSet<String>,
    field_to_idx: &'a BTreeMap<String, usize>,
    enum_tags: &'a BTreeMap<String, i64>,
    enum_variants: &'a BTreeMap<String, EnumVariantLayout>,
    dyn_interfaces: &'a BTreeMap<String, DynInterfaceLayout>,
    vtable_data_ids: &'a BTreeMap<String, DataId>,
    vtable_ifaces: &'a BTreeMap<String, String>,
    layout_engine: &'a LayoutEngine,
    extern_fns: &'a BTreeMap<String, ExternFnDecl>,
    extern_func_ids: &'a BTreeMap<String, FuncId>,
    interned_strings: &'a mut Vec<Box<[u8]>>,
    vars: BTreeMap<String, Variable>,
    var_types: BTreeMap<String, TypeExpr>,
    inout_vars: BTreeSet<String>,
    owned_rc_vars: BTreeSet<String>,
    borrowed_rc_vars: BTreeSet<String>,
    rc_scopes: Vec<Vec<(String, Variable)>>,
    temp_rc_scopes: Vec<Vec<Value>>,
    loop_stack: Vec<LoopCtx>,
    fn_returns_rc: bool,
    fn_ret_ty: Option<TypeExpr>,
    self_var: Option<String>,
    terminated: bool,
}

impl<'a, 'b> FnLower<'a, 'b> {
    fn save_lexical(&self) -> SavedLexical {
        SavedLexical {
            vars: self.vars.clone(),
            var_types: self.var_types.clone(),
            owned_rc_vars: self.owned_rc_vars.clone(),
            borrowed_rc_vars: self.borrowed_rc_vars.clone(),
        }
    }

    fn restore_lexical(&mut self, saved: SavedLexical) {
        self.vars = saved.vars;
        self.var_types = saved.var_types;
        self.owned_rc_vars = saved.owned_rc_vars;
        self.borrowed_rc_vars = saved.borrowed_rc_vars;
    }

    fn declare_fresh_var(&mut self, name: &str, ty: Type) -> Variable {
        let v = self.builder.declare_var(ty);
        self.vars.insert(name.to_string(), v);
        v
    }

    fn call_rt_1(&mut self, id: FuncId, a: Value) -> Value {
        let callee = self.module.declare_func_in_func(id, self.builder.func);
        let call = self.builder.ins().call(callee, &[a]);
        self.builder.inst_results(call)[0]
    }

    fn call_rt_1_void(&mut self, id: FuncId, a: Value) {
        let callee = self.module.declare_func_in_func(id, self.builder.func);
        self.builder.ins().call(callee, &[a]);
    }

    fn call_rt_2(&mut self, id: FuncId, a: i64, b: i64) -> Value {
        let va = self.builder.ins().iconst(types::I64, a);
        let vb = self.builder.ins().iconst(types::I64, b);
        self.call_rt_2_vals(id, va, vb)
    }

    fn call_rt_2_vals(&mut self, id: FuncId, va: Value, vb: Value) -> Value {
        let callee = self.module.declare_func_in_func(id, self.builder.func);
        let call = self.builder.ins().call(callee, &[va, vb]);
        self.builder.inst_results(call)[0]
    }

    fn call_rt_2_void(&mut self, id: FuncId, va: Value, vb: Value) {
        let callee = self.module.declare_func_in_func(id, self.builder.func);
        self.builder.ins().call(callee, &[va, vb]);
    }

    fn call_rt_3_void(&mut self, id: FuncId, obj: Value, idx: i64, val: Value) {
        let vidx = self.builder.ins().iconst(types::I64, idx);
        let callee = self.module.declare_func_in_func(id, self.builder.func);
        self.builder.ins().call(callee, &[obj, vidx, val]);
    }

    fn flush_temps(&mut self, temps: &[Value]) {
        for &v in temps.iter().rev() {
            self.call_rt_1_void(self.rt.rc_dec, v);
        }
    }

    fn emit_cleanup_scopes_from(&mut self, from_depth: usize, skip_var: Option<Variable>) {
        let scopes: Vec<Vec<(String, Variable)>> = self.rc_scopes[from_depth..].to_vec();
        for scope in scopes.iter().rev() {
            for (_, var) in scope.iter().rev() {
                if skip_var == Some(*var) {
                    continue;
                }
                let handle = self.builder.use_var(*var);
                self.call_rt_1_void(self.rt.rc_dec, handle);
            }
        }
    }

    fn emit_cleanup_temp_scopes_from(&mut self, from_depth: usize) {
        let scopes: Vec<Vec<Value>> = self.temp_rc_scopes[from_depth..].to_vec();
        for scope in scopes.iter().rev() {
            for &val in scope.iter().rev() {
                self.call_rt_1_void(self.rt.rc_dec, val);
            }
        }
    }

    fn find_owned_var_in_scopes(&self, var: Variable, from_depth: usize) -> bool {
        self.rc_scopes[from_depth..]
            .iter()
            .any(|sc| sc.iter().any(|(_, v)| *v == var))
    }

    fn type_is_rc(&self, ty: &TypeExpr) -> bool {
        is_rc_type_expr(ty, self.alias_bases, self.record_names, self.enum_names)
    }

    fn list_elem_ty(&self, ty: &TypeExpr) -> Option<TypeExpr> {
        let resolved = resolve_alias_ty(ty, self.alias_bases);
        match resolved {
            TypeExpr::Array { elem, .. } => Some(*elem),
            TypeExpr::Named { name, args, .. } if name == "List" && !args.is_empty() => {
                Some(args[0].clone())
            }
            _ => None,
        }
    }

    /// If `ty` is a fixed-size array `[T; N]`, returns `(elem_ty, len, elem_byte_stride, elem_is_rc)`.
    fn array_layout(&self, ty: &TypeExpr) -> Option<(TypeExpr, usize, usize, bool)> {
        if let TypeExpr::Array { elem, len, .. } = resolve_alias_ty(ty, self.alias_bases) {
            let is_rc = self.type_is_rc(&elem) && !self.is_c_value_type(&elem);
            let stride = if is_rc {
                8
            } else {
                self.layout_engine.size_and_align(&elem).0.max(1)
            };
            return Some((*elem, len, stride, is_rc));
        }
        None
    }

    /// True if `ty` is a C-layout struct, union, or non-RC fixed array stored inline as raw bytes.
    fn is_c_value_type(&self, ty: &TypeExpr) -> bool {
        match resolve_alias_ty(ty, self.alias_bases) {
            TypeExpr::Named { name, .. } => self
                .records
                .get(&name)
                .is_some_and(|r| r.c_layout.is_some()),
            TypeExpr::Array { elem, .. } => !self.type_is_rc(&elem) || self.is_c_value_type(&elem),
            _ => false,
        }
    }

    fn ptr_pointee_ty(&self, ty: &TypeExpr) -> Option<TypeExpr> {
        if let TypeExpr::Named { name, args, .. } = resolve_alias_ty(ty, self.alias_bases) {
            if (name == "Ptr" || name == "MutPtr") && !args.is_empty() {
                return Some(args[0].clone());
            }
            if name == "CStr" {
                return Some(named_ty("UInt8"));
            }
        }
        None
    }

    fn scalar_kind_of_expr(&self, expr: &Expr) -> Option<ScalarKind> {
        let ty = self.infer_expr_type(expr);
        if let Some(sk) = scalar_kind_of_ty(&ty, self.alias_bases) {
            return Some(sk);
        }
        match &expr.kind {
            ExprKind::Float(_) => Some(ScalarKind::F64),
            ExprKind::Cast { ty, .. } => scalar_kind_of_ty(ty, self.alias_bases),
            ExprKind::TryCast { .. } => None,
            ExprKind::Unary { op: UnOp::Neg | UnOp::BitNot, expr: inner } => {
                self.scalar_kind_of_expr(inner)
            }
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
                let lk = self.scalar_kind_of_expr(left);
                let rk = self.scalar_kind_of_expr(right);
                match (lk, rk) {
                    (Some(ScalarKind::I64), Some(k)) if k != ScalarKind::I64 && matches!(left.kind, ExprKind::Int(_)) => Some(k),
                    (Some(k), _) => Some(k),
                    (None, k) => k,
                }
            }
            _ => None,
        }
    }

    fn normalize_scalar(&mut self, val: Value, kind: ScalarKind) -> Value {
        match kind {
            ScalarKind::I8 => {
                let t = self.builder.ins().ireduce(types::I8, val);
                self.builder.ins().sextend(types::I64, t)
            }
            ScalarKind::U8 => {
                let t = self.builder.ins().ireduce(types::I8, val);
                self.builder.ins().uextend(types::I64, t)
            }
            ScalarKind::I16 => {
                let t = self.builder.ins().ireduce(types::I16, val);
                self.builder.ins().sextend(types::I64, t)
            }
            ScalarKind::U16 => {
                let t = self.builder.ins().ireduce(types::I16, val);
                self.builder.ins().uextend(types::I64, t)
            }
            ScalarKind::I32 => {
                let t = self.builder.ins().ireduce(types::I32, val);
                self.builder.ins().sextend(types::I64, t)
            }
            ScalarKind::U32 => {
                let t = self.builder.ins().ireduce(types::I32, val);
                self.builder.ins().uextend(types::I64, t)
            }
            ScalarKind::I64 | ScalarKind::U64 | ScalarKind::F16 | ScalarKind::BF16 | ScalarKind::F32 | ScalarKind::F64 => val,
        }
    }

    fn emit_mem_load(
        &mut self,
        base_ptr: Value,
        byte_off: i32,
        ty: &TypeExpr,
        volatile: bool,
    ) -> Result<Value, String> {
        let resolved = resolve_alias_ty(ty, self.alias_bases);
        if self.is_c_value_type(&resolved) {
            let (sz, _) = self.layout_engine.size_and_align(&resolved);
            let sz_v = self.builder.ins().iconst(types::I64, sz as i64);
            let obj = self.call_rt_1(self.rt.cstruct_alloc, sz_v);
            let dst = self.call_rt_1(self.rt.cstruct_data_ptr, obj);
            let src = if byte_off == 0 {
                base_ptr
            } else {
                let ov = self.builder.ins().iconst(types::I64, byte_off as i64);
                self.builder.ins().iadd(base_ptr, ov)
            };
            let fref = self.module.declare_func_in_func(self.rt.raw_memcpy, self.builder.func);
            self.builder.ins().call(fref, &[dst, src, sz_v]);
            return Ok(obj);
        }
        let sk = scalar_kind_of_ty(&resolved, self.alias_bases);
        if volatile {
            let ptr = if byte_off == 0 {
                base_ptr
            } else {
                let ov = self.builder.ins().iconst(types::I64, byte_off as i64);
                self.builder.ins().iadd(base_ptr, ov)
            };
            let ws: i64 = match sk {
                Some(ScalarKind::I8) => -1,
                Some(ScalarKind::U8) => 1,
                Some(ScalarKind::I16) => -2,
                Some(ScalarKind::U16 | ScalarKind::F16 | ScalarKind::BF16) => 2,
                Some(ScalarKind::I32) => -4,
                Some(ScalarKind::U32 | ScalarKind::F32) => 4,
                None if matches!(&resolved, TypeExpr::Named { name, .. } if name == "Bool") => 1,
                _ => 8,
            };
            let wv = self.builder.ins().iconst(types::I64, ws);
            return Ok(self.call_rt_2_vals(self.rt.volatile_read, ptr, wv));
        }
        let flags = MemFlags::new();
        let val = match sk {
            Some(ScalarKind::I8) => {
                let v = self.builder.ins().load(types::I8, flags, base_ptr, byte_off);
                self.builder.ins().sextend(types::I64, v)
            }
            Some(ScalarKind::U8) => {
                let v = self.builder.ins().load(types::I8, flags, base_ptr, byte_off);
                self.builder.ins().uextend(types::I64, v)
            }
            Some(ScalarKind::I16) => {
                let v = self.builder.ins().load(types::I16, flags, base_ptr, byte_off);
                self.builder.ins().sextend(types::I64, v)
            }
            Some(ScalarKind::U16 | ScalarKind::F16 | ScalarKind::BF16) => {
                let v = self.builder.ins().load(types::I16, flags, base_ptr, byte_off);
                self.builder.ins().uextend(types::I64, v)
            }
            Some(ScalarKind::I32) => {
                let v = self.builder.ins().load(types::I32, flags, base_ptr, byte_off);
                self.builder.ins().sextend(types::I64, v)
            }
            Some(ScalarKind::U32 | ScalarKind::F32) => {
                let v = self.builder.ins().load(types::I32, flags, base_ptr, byte_off);
                self.builder.ins().uextend(types::I64, v)
            }
            None if matches!(&resolved, TypeExpr::Named { name, .. } if name == "Bool") => {
                let v = self.builder.ins().load(types::I8, flags, base_ptr, byte_off);
                self.builder.ins().uextend(types::I64, v)
            }
            _ => self.builder.ins().load(types::I64, flags, base_ptr, byte_off),
        };
        Ok(val)
    }

    fn emit_mem_store(
        &mut self,
        base_ptr: Value,
        byte_off: i32,
        val: Value,
        ty: &TypeExpr,
        volatile: bool,
    ) -> Result<(), String> {
        let resolved = resolve_alias_ty(ty, self.alias_bases);
        if self.is_c_value_type(&resolved) {
            let (sz, _) = self.layout_engine.size_and_align(&resolved);
            let sz_v = self.builder.ins().iconst(types::I64, sz as i64);
            let src = self.call_rt_1(self.rt.cstruct_data_ptr, val);
            let dst = if byte_off == 0 {
                base_ptr
            } else {
                let ov = self.builder.ins().iconst(types::I64, byte_off as i64);
                self.builder.ins().iadd(base_ptr, ov)
            };
            let fref = self.module.declare_func_in_func(self.rt.raw_memcpy, self.builder.func);
            self.builder.ins().call(fref, &[dst, src, sz_v]);
            return Ok(());
        }
        let sk = scalar_kind_of_ty(&resolved, self.alias_bases);
        if volatile {
            let ptr = if byte_off == 0 {
                base_ptr
            } else {
                let ov = self.builder.ins().iconst(types::I64, byte_off as i64);
                self.builder.ins().iadd(base_ptr, ov)
            };
            let w: i64 = match sk {
                Some(ScalarKind::I8 | ScalarKind::U8) => 1,
                Some(ScalarKind::I16 | ScalarKind::U16 | ScalarKind::F16 | ScalarKind::BF16) => 2,
                Some(ScalarKind::I32 | ScalarKind::U32 | ScalarKind::F32) => 4,
                None if matches!(&resolved, TypeExpr::Named { name, .. } if name == "Bool") => 1,
                _ => 8,
            };
            let wv = self.builder.ins().iconst(types::I64, w);
            let fref = self.module.declare_func_in_func(self.rt.volatile_write, self.builder.func);
            self.builder.ins().call(fref, &[ptr, val, wv]);
            return Ok(());
        }
        let flags = MemFlags::new();
        match sk {
            Some(ScalarKind::I8 | ScalarKind::U8) => {
                let t = self.builder.ins().ireduce(types::I8, val);
                self.builder.ins().store(flags, t, base_ptr, byte_off);
            }
            Some(ScalarKind::I16 | ScalarKind::U16 | ScalarKind::F16 | ScalarKind::BF16) => {
                let t = self.builder.ins().ireduce(types::I16, val);
                self.builder.ins().store(flags, t, base_ptr, byte_off);
            }
            Some(ScalarKind::I32 | ScalarKind::U32 | ScalarKind::F32) => {
                let t = self.builder.ins().ireduce(types::I32, val);
                self.builder.ins().store(flags, t, base_ptr, byte_off);
            }
            None if matches!(&resolved, TypeExpr::Named { name, .. } if name == "Bool") => {
                let t = self.builder.ins().ireduce(types::I8, val);
                self.builder.ins().store(flags, t, base_ptr, byte_off);
            }
            _ => {
                self.builder.ins().store(flags, val, base_ptr, byte_off);
            }
        }
        Ok(())
    }

    fn emit_cast(&mut self, val: Value, src_ty: &TypeExpr, dst_ty: &TypeExpr) -> Value {
        let src_base = resolve_alias_ty(src_ty, self.alias_bases);
        let dst_base = resolve_alias_ty(dst_ty, self.alias_bases);
        let src_sk = scalar_kind_of_ty(&src_base, self.alias_bases);
        let dst_sk = scalar_kind_of_ty(&dst_base, self.alias_bases);

        if matches!(&dst_base, TypeExpr::Named { name, .. } if name == "Bool") {
            let zero = self.builder.ins().iconst(types::I64, 0);
            let ne = self.builder.ins().icmp(IntCC::NotEqual, val, zero);
            return self.builder.ins().uextend(types::I64, ne);
        }

        match (src_sk, dst_sk) {
            (Some(ScalarKind::F16), Some(ScalarKind::F16))
            | (Some(ScalarKind::BF16), Some(ScalarKind::BF16)) => val,
            (Some(ScalarKind::F16), Some(ScalarKind::F64)) => {
                self.call_rt_1(self.rt.f16_to_f64, val)
            }
            (Some(ScalarKind::BF16), Some(ScalarKind::F64)) => {
                self.call_rt_1(self.rt.bf16_to_f64, val)
            }
            (Some(ScalarKind::F64), Some(ScalarKind::F16)) => {
                self.call_rt_1(self.rt.f64_to_f16, val)
            }
            (Some(ScalarKind::F64), Some(ScalarKind::BF16)) => {
                self.call_rt_1(self.rt.f64_to_bf16, val)
            }
            (Some(ScalarKind::F16), Some(ScalarKind::F32)) => {
                let d64 = self.call_rt_1(self.rt.f16_to_f64, val);
                let f64v = self.builder.ins().bitcast(types::F64, MemFlags::new(), d64);
                let f32v = self.builder.ins().fdemote(types::F32, f64v);
                let i32v = self.builder.ins().bitcast(types::I32, MemFlags::new(), f32v);
                self.builder.ins().uextend(types::I64, i32v)
            }
            (Some(ScalarKind::BF16), Some(ScalarKind::F32)) => {
                let d64 = self.call_rt_1(self.rt.bf16_to_f64, val);
                let f64v = self.builder.ins().bitcast(types::F64, MemFlags::new(), d64);
                let f32v = self.builder.ins().fdemote(types::F32, f64v);
                let i32v = self.builder.ins().bitcast(types::I32, MemFlags::new(), f32v);
                self.builder.ins().uextend(types::I64, i32v)
            }
            (Some(ScalarKind::F32), Some(ScalarKind::F16)) => {
                let i32v = self.builder.ins().ireduce(types::I32, val);
                let f32v = self.builder.ins().bitcast(types::F32, MemFlags::new(), i32v);
                let f64v = self.builder.ins().fpromote(types::F64, f32v);
                let d64 = self.builder.ins().bitcast(types::I64, MemFlags::new(), f64v);
                self.call_rt_1(self.rt.f64_to_f16, d64)
            }
            (Some(ScalarKind::F32), Some(ScalarKind::BF16)) => {
                let i32v = self.builder.ins().ireduce(types::I32, val);
                let f32v = self.builder.ins().bitcast(types::F32, MemFlags::new(), i32v);
                let f64v = self.builder.ins().fpromote(types::F64, f32v);
                let d64 = self.builder.ins().bitcast(types::I64, MemFlags::new(), f64v);
                self.call_rt_1(self.rt.f64_to_bf16, d64)
            }
            (Some(ScalarKind::F16), Some(ScalarKind::BF16)) => {
                let d64 = self.call_rt_1(self.rt.f16_to_f64, val);
                self.call_rt_1(self.rt.f64_to_bf16, d64)
            }
            (Some(ScalarKind::BF16), Some(ScalarKind::F16)) => {
                let d64 = self.call_rt_1(self.rt.bf16_to_f64, val);
                self.call_rt_1(self.rt.f64_to_f16, d64)
            }
            (Some(ScalarKind::F16), Some(dk)) if dk.is_int() => {
                let d64 = self.call_rt_1(self.rt.f16_to_f64, val);
                let f64v = self.builder.ins().bitcast(types::F64, MemFlags::new(), d64);
                let iv = if dk.is_unsigned() {
                    self.builder.ins().fcvt_to_uint_sat(types::I64, f64v)
                } else {
                    self.builder.ins().fcvt_to_sint_sat(types::I64, f64v)
                };
                self.normalize_scalar(iv, dk)
            }
            (Some(ScalarKind::BF16), Some(dk)) if dk.is_int() => {
                let d64 = self.call_rt_1(self.rt.bf16_to_f64, val);
                let f64v = self.builder.ins().bitcast(types::F64, MemFlags::new(), d64);
                let iv = if dk.is_unsigned() {
                    self.builder.ins().fcvt_to_uint_sat(types::I64, f64v)
                } else {
                    self.builder.ins().fcvt_to_sint_sat(types::I64, f64v)
                };
                self.normalize_scalar(iv, dk)
            }
            (sk, Some(ScalarKind::F16)) if sk != Some(ScalarKind::F16) => {
                let f64v = if sk.is_some_and(|k| k.is_unsigned()) {
                    self.builder.ins().fcvt_from_uint(types::F64, val)
                } else {
                    self.builder.ins().fcvt_from_sint(types::F64, val)
                };
                let d64 = self.builder.ins().bitcast(types::I64, MemFlags::new(), f64v);
                self.call_rt_1(self.rt.f64_to_f16, d64)
            }
            (sk, Some(ScalarKind::BF16)) if sk != Some(ScalarKind::BF16) => {
                let f64v = if sk.is_some_and(|k| k.is_unsigned()) {
                    self.builder.ins().fcvt_from_uint(types::F64, val)
                } else {
                    self.builder.ins().fcvt_from_sint(types::F64, val)
                };
                let d64 = self.builder.ins().bitcast(types::I64, MemFlags::new(), f64v);
                self.call_rt_1(self.rt.f64_to_bf16, d64)
            }
            (Some(ScalarKind::F64), Some(ScalarKind::F32)) => {
                let f64v = self.builder.ins().bitcast(types::F64, MemFlags::new(), val);
                let f32v = self.builder.ins().fdemote(types::F32, f64v);
                let i32v = self.builder.ins().bitcast(types::I32, MemFlags::new(), f32v);
                self.builder.ins().uextend(types::I64, i32v)
            }
            (Some(ScalarKind::F32), Some(ScalarKind::F64)) => {
                let i32v = self.builder.ins().ireduce(types::I32, val);
                let f32v = self.builder.ins().bitcast(types::F32, MemFlags::new(), i32v);
                let f64v = self.builder.ins().fpromote(types::F64, f32v);
                self.builder.ins().bitcast(types::I64, MemFlags::new(), f64v)
            }
            (Some(ScalarKind::F64), Some(dk)) if dk.is_int() => {
                let f64v = self.builder.ins().bitcast(types::F64, MemFlags::new(), val);
                let iv = if dk.is_unsigned() {
                    self.builder.ins().fcvt_to_uint_sat(types::I64, f64v)
                } else {
                    self.builder.ins().fcvt_to_sint_sat(types::I64, f64v)
                };
                self.normalize_scalar(iv, dk)
            }
            (Some(ScalarKind::F32), Some(dk)) if dk.is_int() => {
                let i32v = self.builder.ins().ireduce(types::I32, val);
                let f32v = self.builder.ins().bitcast(types::F32, MemFlags::new(), i32v);
                let iv = if dk.is_unsigned() {
                    self.builder.ins().fcvt_to_uint_sat(types::I64, f32v)
                } else {
                    self.builder.ins().fcvt_to_sint_sat(types::I64, f32v)
                };
                self.normalize_scalar(iv, dk)
            }
            (sk, Some(ScalarKind::F64)) if sk != Some(ScalarKind::F64) => {
                let f64v = if sk.is_some_and(|k| k.is_unsigned()) {
                    self.builder.ins().fcvt_from_uint(types::F64, val)
                } else {
                    self.builder.ins().fcvt_from_sint(types::F64, val)
                };
                self.builder.ins().bitcast(types::I64, MemFlags::new(), f64v)
            }
            (sk, Some(ScalarKind::F32)) if sk != Some(ScalarKind::F32) => {
                let f32v = if sk.is_some_and(|k| k.is_unsigned()) {
                    self.builder.ins().fcvt_from_uint(types::F32, val)
                } else {
                    self.builder.ins().fcvt_from_sint(types::F32, val)
                };
                let i32v = self.builder.ins().bitcast(types::I32, MemFlags::new(), f32v);
                self.builder.ins().uextend(types::I64, i32v)
            }
            (_, Some(dk)) => self.normalize_scalar(val, dk),
            _ => val,
        }
    }

    fn emit_extern_c_call(
        &mut self,
        ef: &ExternFnDecl,
        fid: FuncId,
        args: &[CallArg],
    ) -> Result<Value, String> {
        let mut temps = Vec::new();
        let mut c_args = Vec::with_capacity(args.len());
        for (i, a) in args.iter().enumerate() {
            let raw = self.lower_expr_borrowed(&a.value, &mut temps)?;
            let pty = ef.params.get(i).map(|p| &p.ty);
            let cval = if let Some(pt) = pty {
                let pb = resolve_alias_ty(pt, self.alias_bases);
                match scalar_kind_of_ty(&pb, self.alias_bases) {
                    Some(ScalarKind::I8 | ScalarKind::U8) => self.builder.ins().ireduce(types::I8, raw),
                    Some(ScalarKind::I16 | ScalarKind::U16 | ScalarKind::F16 | ScalarKind::BF16) => self.builder.ins().ireduce(types::I16, raw),
                    Some(ScalarKind::I32 | ScalarKind::U32) => self.builder.ins().ireduce(types::I32, raw),
                    Some(ScalarKind::F32) => {
                        let i32v = self.builder.ins().ireduce(types::I32, raw);
                        self.builder.ins().bitcast(types::F32, MemFlags::new(), i32v)
                    }
                    Some(ScalarKind::F64) => self.builder.ins().bitcast(types::F64, MemFlags::new(), raw),
                    _ if matches!(&pb, TypeExpr::Named { name, .. } if name == "Bool") => {
                        self.builder.ins().ireduce(types::I8, raw)
                    }
                    _ => raw,
                }
            } else {
                raw
            };
            c_args.push(cval);
        }
        let callee = self.module.declare_func_in_func(fid, self.builder.func);
        let call = self.builder.ins().call(callee, &c_args);
        let results = self.builder.inst_results(call).to_vec();
        self.flush_temps(&temps);
        if let (Some(&rval), Some(rt)) = (results.first(), &ef.ret_ty) {
            let rb = resolve_alias_ty(rt, self.alias_bases);
            let out = match scalar_kind_of_ty(&rb, self.alias_bases) {
                Some(ScalarKind::I8 | ScalarKind::I16 | ScalarKind::I32) => {
                    self.builder.ins().sextend(types::I64, rval)
                }
                Some(ScalarKind::U8 | ScalarKind::U16 | ScalarKind::U32 | ScalarKind::F16 | ScalarKind::BF16) => {
                    self.builder.ins().uextend(types::I64, rval)
                }
                Some(ScalarKind::F32) => {
                    let i32v = self.builder.ins().bitcast(types::I32, MemFlags::new(), rval);
                    self.builder.ins().uextend(types::I64, i32v)
                }
                Some(ScalarKind::F64) => self.builder.ins().bitcast(types::I64, MemFlags::new(), rval),
                _ if matches!(&rb, TypeExpr::Named { name, .. } if name == "Bool") => {
                    self.builder.ins().uextend(types::I64, rval)
                }
                _ => rval,
            };
            Ok(out)
        } else {
            Ok(self.builder.ins().iconst(types::I64, 0))
        }
    }

    fn infer_expr_type(&self, expr: &Expr) -> TypeExpr {
        self.infer_expr_type_with(expr, &BTreeMap::new())
    }

    fn infer_expr_type_with(&self, expr: &Expr, extra_vars: &BTreeMap<String, TypeExpr>) -> TypeExpr {
        match &expr.kind {
            ExprKind::Int(_) => named_ty("Int"),
            ExprKind::Float(_) => named_ty("Float"),
            ExprKind::Bool(_) => named_ty("Bool"),
            ExprKind::Str(_) => named_ty("Text"),
            ExprKind::Ident(s) => {
                if s == "None" {
                    return generic_ty("Option", vec![named_ty("Any")]);
                }
                if let Some(ty) = extra_vars.get(s) {
                    return ty.clone();
                }
                if let Some(ty) = self.var_types.get(s) {
                    return ty.clone();
                }
                if let Some(vl) = self.enum_variants.get(s) {
                    return named_ty(&vl.enum_name);
                }
                named_ty("Any")
            }
            ExprKind::RecordLit { name, .. } => named_ty(name),
            ExprKind::List(elems) => {
                if let Some(first) = elems.first() {
                    generic_ty("List", vec![self.infer_expr_type_with(first, extra_vars)])
                } else {
                    generic_ty("List", vec![named_ty("Any")])
                }
            }
            ExprKind::Tuple(elems) => TypeExpr::Tuple(
                elems.iter().map(|e| self.infer_expr_type_with(e, extra_vars)).collect(),
                expr.span,
            ),
            ExprKind::TupleIndex { target, index } => {
                let tt = resolve_alias_ty(&self.infer_expr_type_with(target, extra_vars), self.alias_bases);
                if let TypeExpr::Tuple(es, _) = tt {
                    if let Some(et) = es.get(*index) {
                        return et.clone();
                    }
                }
                named_ty("Any")
            }
            ExprKind::Cast { ty, .. } => ty.clone(),
            ExprKind::TryCast { ty, .. } => generic_ty("Result", vec![ty.clone(), named_ty("Text")]),
            ExprKind::ArrayRepeat { value, len } => TypeExpr::Array {
                elem: Box::new(self.infer_expr_type_with(value, extra_vars)),
                len: *len,
                span: expr.span,
            },
            ExprKind::Unsafe(b) => self.infer_block_type_with(b, extra_vars),
            ExprKind::Binary { op: BinOp::Concat, .. } => named_ty("Text"),
            ExprKind::Binary {
                op: BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::And | BinOp::Or,
                ..
            } => named_ty("Bool"),
            ExprKind::Binary { op, left, right } => {
                let lt = self.infer_expr_type_with(left, extra_vars);
                let rt = self.infer_expr_type_with(right, extra_vars);
                if matches!(left.kind, ExprKind::Int(_) | ExprKind::Float(_))
                    && scalar_kind_of_ty(&rt, self.alias_bases).is_some()
                {
                    rt
                } else if scalar_kind_of_ty(&lt, self.alias_bases).is_some() {
                    lt
                } else if scalar_kind_of_ty(&rt, self.alias_bases).is_some() {
                    rt
                } else if matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod)
                    && (is_float_type_expr(&lt, self.alias_bases)
                        || is_float_type_expr(&rt, self.alias_bases))
                {
                    named_ty("Float")
                } else {
                    named_ty("Int")
                }
            }
            ExprKind::Unary { op: UnOp::Not, .. } => named_ty("Bool"),
            ExprKind::Unary { op: op @ (UnOp::Neg | UnOp::BitNot), expr: inner } => {
                let it = self.infer_expr_type_with(inner, extra_vars);
                if scalar_kind_of_ty(&it, self.alias_bases).is_some() {
                    it
                } else if matches!(op, UnOp::Neg) && is_float_type_expr(&it, self.alias_bases) {
                    named_ty("Float")
                } else {
                    named_ty("Int")
                }
            }
            ExprKind::Field { target, field } => {
                if let ExprKind::Ident(ename) = &target.kind {
                    if let Some(vl) = self.enum_variants.get(&format!("{}.{}", ename, field)) {
                        return named_ty(&vl.enum_name);
                    }
                }
                let obj_ty = resolve_alias_ty(&self.infer_expr_type_with(target, extra_vars), self.alias_bases);
                if let TypeExpr::Named { name: rname, .. } = obj_ty {
                    if let Some(layout) = self.records.get(&rname) {
                        if let Some(fl) = layout.fields.iter().find(|f| f.name == *field) {
                            return fl.ty.clone();
                        }
                    }
                }
                named_ty("Any")
            }
            ExprKind::Index { target, .. } => {
                let tt = self.infer_expr_type_with(target, extra_vars);
                self.list_elem_ty(&tt).unwrap_or_else(|| named_ty("Any"))
            }
            ExprKind::Call { callee, args } => match &callee.kind {
                ExprKind::Field { target, field } if field == "parse" => {
                    if let ExprKind::Ident(rname) = &target.kind {
                        return generic_ty("Result", vec![named_ty(rname), named_ty("Int")]);
                    }
                    named_ty("Any")
                }
                ExprKind::Field { target, field } => {
                    if let ExprKind::Ident(ename) = &target.kind {
                        if let Some(vl) = self.enum_variants.get(&format!("{}.{}", ename, field)) {
                            return named_ty(&vl.enum_name);
                        }
                    }
                    let recv_ty = resolve_alias_ty(
                        &self.infer_expr_type_with(target, extra_vars),
                        self.alias_bases,
                    );
                    if let TypeExpr::Dyn { interface, .. } = recv_ty {
                        if let Some(layout) = self.dyn_interfaces.get(&interface) {
                            if let Some(dm) = layout.methods.iter().find(|m| m.name == *field) {
                                return dm.ret_ty.clone().unwrap_or_else(|| named_ty("Any"));
                            }
                        }
                    }
                    named_ty("Any")
                }
                ExprKind::Ident(f) => {
                    if f == "__make_closure" {
                        if let Some(a0) = args.first() {
                            if let ExprKind::Str(csym) = &a0.value.kind {
                                if let Some(meta) = self.fn_metas.get(csym) {
                                    return TypeExpr::Fn {
                                        params: meta.param_tys.iter().skip(1).cloned().collect(),
                                        ret: meta.ret_ty.clone().map(Box::new),
                                        span: expr.span,
                                    };
                                }
                            }
                        }
                    }
                    if f == "__call_closure" {
                        if let Some(a0) = args.first() {
                            let cty = resolve_alias_ty(
                                &self.infer_expr_type_with(&a0.value, extra_vars),
                                self.alias_bases,
                            );
                            if let TypeExpr::Fn { ret, .. } = cty {
                                return ret.map(|r| *r).unwrap_or_else(|| named_ty("Any"));
                            }
                        }
                    }
                    if f == "__box_dyn" {
                        if let Some(a1) = args.get(1) {
                            if let ExprKind::Str(vsym) = &a1.value.kind {
                                if let Some(iface) = self.vtable_ifaces.get(vsym) {
                                    return TypeExpr::Dyn {
                                        interface: iface.clone(),
                                        args: vec![],
                                        span: expr.span,
                                    };
                                }
                            }
                        }
                    }
                    if f == "Ok" && args.len() == 1 {
                        return generic_ty("Result", vec![self.infer_expr_type_with(&args[0].value, extra_vars), named_ty("Any")]);
                    }
                    if f == "Err" && args.len() == 1 {
                        return generic_ty("Result", vec![named_ty("Any"), self.infer_expr_type_with(&args[0].value, extra_vars)]);
                    }
                    if f == "Some" && args.len() == 1 {
                        return generic_ty("Option", vec![self.infer_expr_type_with(&args[0].value, extra_vars)]);
                    }
                    if f == "push" && args.len() == 2 {
                        let lt = self.infer_expr_type_with(&args[0].value, extra_vars);
                        if self.list_elem_ty(&lt).is_some_and(|e| e.display_name() != "Any") {
                            return lt;
                        }
                        return generic_ty("List", vec![self.infer_expr_type_with(&args[1].value, extra_vars)]);
                    }
                    if f == "list_filled" && args.len() == 2 {
                        return generic_ty("List", vec![self.infer_expr_type_with(&args[1].value, extra_vars)]);
                    }
                    if f == "list_pop_shrink" && args.len() == 1 {
                        return self.infer_expr_type_with(&args[0].value, extra_vars);
                    }
                    if f == "mutex_get" && args.len() == 1 {
                        if let TypeExpr::Named { name, args: ma, .. } =
                            resolve_alias_ty(&self.infer_expr_type_with(&args[0].value, extra_vars), self.alias_bases)
                        {
                            if name == "Mutex" && !ma.is_empty() {
                                return ma[0].clone();
                            }
                        }
                    }
                    if (f == "chan_recv" || f == "chan_try_recv") && args.len() == 1 {
                        if let TypeExpr::Named { name, args: ca, .. } =
                            resolve_alias_ty(&self.infer_expr_type_with(&args[0].value, extra_vars), self.alias_bases)
                        {
                            if name == "Channel" && !ca.is_empty() {
                                return generic_ty("Option", vec![ca[0].clone()]);
                            }
                        }
                        return generic_ty("Option", vec![named_ty("Any")]);
                    }
                    if (f == "ptr_read" || f == "volatile_read") && args.len() == 1 {
                        let pty = self.infer_expr_type_with(&args[0].value, extra_vars);
                        if let Some(elem) = self.ptr_pointee_ty(&pty) {
                            return elem;
                        }
                    }
                    if (f == "__ptr_add" || f == "ptr_add") && !args.is_empty() {
                        return self.infer_expr_type_with(&args[0].value, extra_vars);
                    }
                    if (f == "ptr_of" || f == "mut_ptr_of") && args.len() == 1 {
                        let raw_inner = self.infer_expr_type_with(&args[0].value, extra_vars);
                        let inner = match resolve_alias_ty(&raw_inner, self.alias_bases) {
                            TypeExpr::Array { elem, .. } => *elem,
                            _ => raw_inner,
                        };
                        let pname = if f == "mut_ptr_of" { "MutPtr" } else { "Ptr" };
                        return generic_ty(pname, vec![inner]);
                    }
                    if matches!(
                        f.as_str(),
                        "wrapping_add" | "wrapping_sub" | "wrapping_mul" | "saturating_add" | "saturating_sub"
                    ) && !args.is_empty()
                    {
                        return self.infer_expr_type_with(&args[0].value, extra_vars);
                    }
                    if let Some(meta) = self.fn_metas.get(f) {
                        if let Some(rt) = &meta.ret_ty {
                            return rt.clone();
                        }
                    }
                    if let Some(vl) = self.enum_variants.get(f) {
                        return named_ty(&vl.enum_name);
                    }
                    named_ty("Any")
                }
                _ => named_ty("Any"),
            },
            ExprKind::If { then_block, else_branch, .. } => {
                let then_ty = self.infer_block_type_with(then_block, extra_vars);
                if then_ty.display_name() != "Any" {
                    return then_ty;
                }
                if let Some(eb) = else_branch {
                    return self.infer_expr_type_with(eb, extra_vars);
                }
                named_ty("Any")
            }
            ExprKind::Match { scrutinee, arms } => {
                let st = self.infer_expr_type_with(scrutinee, extra_vars);
                for arm in arms {
                    let mut arm_vars = extra_vars.clone();
                    for p in &arm.patterns {
                        self.bind_pattern_types_for_infer(&st, p, &mut arm_vars);
                    }
                    let t = self.infer_expr_type_with(&arm.body, &arm_vars);
                    if t.display_name() != "Any" {
                        return t;
                    }
                }
                named_ty("Any")
            }
            ExprKind::Block(blk) => self.infer_block_type_with(blk, extra_vars),
            _ => named_ty("Any"),
        }
    }

    fn bind_pattern_types_for_infer(&self, sty: &TypeExpr, pat: &Pattern, out: &mut BTreeMap<String, TypeExpr>) {
        let base = resolve_alias_ty(sty, self.alias_bases);
        match pat {
            Pattern::Ident(s, _) if s != "_" => {
                out.insert(s.clone(), sty.clone());
            }
            Pattern::Tuple(ps, _) => {
                let elems = match &base {
                    TypeExpr::Tuple(es, _) => es.clone(),
                    _ => Vec::new(),
                };
                for (i, sp) in ps.iter().enumerate() {
                    let ety = elems.get(i).cloned().unwrap_or_else(|| named_ty("Any"));
                    self.bind_pattern_types_for_infer(&ety, sp, out);
                }
            }
            Pattern::Variant { name, binding: Some(b), .. } if b != "_" => {
                let ptys = self.resolve_variant_payload_tys(&base, name);
                out.insert(b.clone(), ptys.into_iter().next().unwrap_or_else(|| named_ty("Any")));
            }
            Pattern::VariantMulti { name, patterns, .. } => {
                let ptys = self.resolve_variant_payload_tys(&base, name);
                for (i, sp) in patterns.iter().enumerate() {
                    let pty = ptys.get(i).cloned().unwrap_or_else(|| named_ty("Any"));
                    self.bind_pattern_types_for_infer(&pty, sp, out);
                }
            }
            _ => {}
        }
    }

    fn infer_block_type_with(&self, blk: &Block, extra_vars: &BTreeMap<String, TypeExpr>) -> TypeExpr {
        let mut local = extra_vars.clone();
        for s in &blk.stmts {
            if let Stmt::Let { name, ty, value, .. } = s {
                let lt = ty.clone().unwrap_or_else(|| self.infer_expr_type_with(value, &local));
                local.insert(name.clone(), lt);
            }
        }
        blk.tail
            .as_deref()
            .map(|t| self.infer_expr_type_with(t, &local))
            .unwrap_or_else(|| named_ty("Any"))
    }

    fn resolve_field_layout(&self, obj_expr: &Expr, field: &str) -> Result<(usize, Option<FieldLayout>), String> {
        let obj_ty = resolve_alias_ty(&self.infer_expr_type(obj_expr), self.alias_bases);
        if let TypeExpr::Named { name: rname, .. } = obj_ty {
            if let Some(layout) = self.records.get(&rname) {
                if let Some((pos, fl)) = layout
                    .fields
                    .iter()
                    .enumerate()
                    .find(|(_, f)| f.name == field)
                {
                    return Ok((pos, Some(fl.clone())));
                }
            }
        }
        let idx = self
            .field_to_idx
            .get(field)
            .copied()
            .ok_or_else(|| format!("unknown record field `{}` in codegen", field))?;
        Ok((idx, None))
    }

    fn is_rc_ident(&self, s: &str) -> bool {
        self.owned_rc_vars.contains(s)
            || self.borrowed_rc_vars.contains(s)
            || self.var_types.get(s).is_some_and(|t| self.type_is_rc(t))
    }

    fn expr_produces_rc(&self, expr: &Expr) -> bool {
        if self.type_is_rc(&self.infer_expr_type(expr)) {
            return true;
        }
        match &expr.kind {
            ExprKind::RecordLit { .. }
            | ExprKind::List(_)
            | ExprKind::ArrayRepeat { .. }
            | ExprKind::Tuple(_)
            | ExprKind::Str(_) => true,
            ExprKind::Binary { op: BinOp::Concat, .. } => true,
            ExprKind::Ident(s) => {
                s == "None"
                    || self.is_rc_ident(s)
                    || (self.enum_tags.contains_key(s) && !self.vars.contains_key(s))
            }
            ExprKind::TupleIndex { target, index } => {
                let tt = resolve_alias_ty(&self.infer_expr_type(target), self.alias_bases);
                if let TypeExpr::Tuple(es, _) = tt {
                    es.get(*index).is_some_and(|et| self.type_is_rc(et))
                } else {
                    false
                }
            }
            ExprKind::Field { target, field } => {
                if let ExprKind::Ident(ename) = &target.kind {
                    if self.enum_tags.contains_key(&format!("{}.{}", ename, field)) {
                        return true;
                    }
                }
                if let TypeExpr::Named { name: rname, .. } =
                    resolve_alias_ty(&self.infer_expr_type(target), self.alias_bases)
                {
                    if let Some(cfl) = self
                        .records
                        .get(&rname)
                        .and_then(|rl| rl.c_layout.as_ref())
                        .and_then(|cl| cl.fields.iter().find(|f| f.name == *field))
                    {
                        return self.is_c_value_type(&cfl.ty);
                    }
                }
                if let Ok((_, Some(fl))) = self.resolve_field_layout(target, field) {
                    return fl.is_rc;
                }
                false
            }
            ExprKind::Index { target, .. } => {
                let tt = self.infer_expr_type(target);
                self.list_elem_ty(&tt).is_some_and(|et| self.type_is_rc(&et))
            }
            ExprKind::Call { callee, args } => match &callee.kind {
                ExprKind::Ident(f) => {
                    if f == "mutex_get" {
                        if let Some(a0) = args.first() {
                            if let TypeExpr::Named { name, args: ma, .. } =
                                resolve_alias_ty(&self.infer_expr_type(&a0.value), self.alias_bases)
                            {
                                if name == "Mutex" && !ma.is_empty() {
                                    return self.type_is_rc(&ma[0]);
                                }
                            }
                        }
                        return false;
                    }
                    if f == "ptr_read" || f == "volatile_read" {
                        if let Some(a0) = args.first() {
                            let pty = self.infer_expr_type(&a0.value);
                            if let Some(pt) = self.ptr_pointee_ty(&pty) {
                                return self.is_c_value_type(&pt);
                            }
                        }
                        return false;
                    }
                    matches!(
                        f.as_str(),
                        "Ok" | "Err" | "Some" | "push" | "list_filled" | "list_pop_shrink" | "yield_val" | "__box_dyn" | "__make_closure"
                    )
                        || (self.enum_tags.contains_key(f) && !self.func_ids.contains_key(f))
                        || self.fn_metas.get(f).is_some_and(|m| m.returns_rc)
                }
                ExprKind::Field { target, field } => {
                    if field == "parse"
                        || matches!(&target.kind, ExprKind::Ident(e) if self.enum_tags.contains_key(&format!("{}.{}", e, field)))
                    {
                        return true;
                    }
                    let recv_ty = resolve_alias_ty(&self.infer_expr_type(target), self.alias_bases);
                    if let TypeExpr::Dyn { interface, .. } = recv_ty {
                        if let Some(layout) = self.dyn_interfaces.get(&interface) {
                            if let Some(dm) = layout.methods.iter().find(|m| m.name == *field) {
                                return dm.returns_rc;
                            }
                        }
                    }
                    false
                }
                _ => false,
            },
            ExprKind::If { then_block, else_branch, .. } => {
                else_branch.is_some()
                    && (then_block.tail.as_deref().is_some_and(|t| self.expr_produces_rc(t))
                        || else_branch.as_deref().is_some_and(|e| self.expr_produces_rc(e)))
            }
            ExprKind::Match { arms, .. } => arms.iter().any(|a| self.expr_produces_rc(&a.body)),
            ExprKind::Block(blk) => blk.tail.as_deref().is_some_and(|t| self.expr_produces_rc(t)),
            _ => false,
        }
    }

    fn is_text_expr(&self, expr: &Expr) -> bool {
        let ty = self.infer_expr_type(expr);
        if is_text_type_expr(&ty, self.alias_bases) {
            return true;
        }
        match &expr.kind {
            ExprKind::Str(_) | ExprKind::Binary { op: BinOp::Concat, .. } => true,
            ExprKind::If { then_block, else_branch, .. } => {
                then_block.tail.as_deref().is_some_and(|t| self.is_text_expr(t))
                    || else_branch.as_deref().is_some_and(|e| self.is_text_expr(e))
            }
            ExprKind::Match { arms, .. } => arms.iter().any(|a| self.is_text_expr(&a.body)),
            ExprKind::Block(blk) => blk.tail.as_deref().is_some_and(|t| self.is_text_expr(t)),
            _ => false,
        }
    }

    fn is_float_expr(&self, expr: &Expr) -> bool {
        let ty = self.infer_expr_type(expr);
        if is_float_type_expr(&ty, self.alias_bases) {
            return true;
        }
        match &expr.kind {
            ExprKind::Float(_) => true,
            ExprKind::Unary { op: UnOp::Neg, expr: inner } => self.is_float_expr(inner),
            ExprKind::Binary { op, left, right }
                if matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) =>
            {
                self.is_float_expr(left) || self.is_float_expr(right)
            }
            ExprKind::If { then_block, else_branch, .. } => {
                then_block.tail.as_deref().is_some_and(|t| self.is_float_expr(t))
                    || else_branch.as_deref().is_some_and(|e| self.is_float_expr(e))
            }
            ExprKind::Match { arms, .. } => arms.iter().any(|a| self.is_float_expr(&a.body)),
            ExprKind::Block(blk) => blk.tail.as_deref().is_some_and(|t| self.is_float_expr(t)),
            _ => false,
        }
    }

    fn resolve_variant_payload_tys(&self, scrutinee_base: &TypeExpr, variant_name: &str) -> Vec<TypeExpr> {
        if variant_name == "Some" {
            if let TypeExpr::Named { name, args, .. } = scrutinee_base {
                if name == "Option" && !args.is_empty() {
                    return vec![args[0].clone()];
                }
            }
            return vec![named_ty("Any")];
        }
        if variant_name == "Ok" {
            if let TypeExpr::Named { name, args, .. } = scrutinee_base {
                if name == "Result" && !args.is_empty() {
                    return vec![args[0].clone()];
                }
            }
            return vec![named_ty("Any")];
        }
        if variant_name == "Err" {
            if let TypeExpr::Named { name, args, .. } = scrutinee_base {
                if name == "Result" && args.len() >= 2 {
                    return vec![args[1].clone()];
                }
            }
            return vec![named_ty("Any")];
        }
        if let Some(vl) = self.enum_variants.get(variant_name) {
            return vl.payload_tys.clone();
        }
        Vec::new()
    }

    fn emit_compound_op(
        &mut self,
        cur: Value,
        rhs: Value,
        op: AssignOp,
        is_float: bool,
        sk: Option<ScalarKind>,
    ) -> Value {
        if matches!(sk, Some(ScalarKind::F16 | ScalarKind::BF16)) {
            let (to_f64, from_f64) = if sk == Some(ScalarKind::F16) {
                (self.rt.f16_to_f64, self.rt.f64_to_f16)
            } else {
                (self.rt.bf16_to_f64, self.rt.f64_to_bf16)
            };
            let c64 = self.call_rt_1(to_f64, cur);
            let r64 = self.call_rt_1(to_f64, rhs);
            let cf = self.builder.ins().bitcast(types::F64, MemFlags::new(), c64);
            let rf = self.builder.ins().bitcast(types::F64, MemFlags::new(), r64);
            let res = match op {
                AssignOp::AddSet => self.builder.ins().fadd(cf, rf),
                AssignOp::SubSet => self.builder.ins().fsub(cf, rf),
                AssignOp::MulSet => self.builder.ins().fmul(cf, rf),
                AssignOp::DivSet => self.builder.ins().fdiv(cf, rf),
                _ => rf,
            };
            let o64 = self.builder.ins().bitcast(types::I64, MemFlags::new(), res);
            return self.call_rt_1(from_f64, o64);
        }
        if sk == Some(ScalarKind::F32) {
            let c32 = self.builder.ins().ireduce(types::I32, cur);
            let r32 = self.builder.ins().ireduce(types::I32, rhs);
            let cf = self.builder.ins().bitcast(types::F32, MemFlags::new(), c32);
            let rf = self.builder.ins().bitcast(types::F32, MemFlags::new(), r32);
            let res = match op {
                AssignOp::AddSet => self.builder.ins().fadd(cf, rf),
                AssignOp::SubSet => self.builder.ins().fsub(cf, rf),
                AssignOp::MulSet => self.builder.ins().fmul(cf, rf),
                AssignOp::DivSet => self.builder.ins().fdiv(cf, rf),
                _ => rf,
            };
            let o32 = self.builder.ins().bitcast(types::I32, MemFlags::new(), res);
            return self.builder.ins().uextend(types::I64, o32);
        }
        if is_float {
            let cf = self.builder.ins().bitcast(types::F64, MemFlags::new(), cur);
            let rf = self.builder.ins().bitcast(types::F64, MemFlags::new(), rhs);
            let res = match op {
                AssignOp::AddSet => self.builder.ins().fadd(cf, rf),
                AssignOp::SubSet => self.builder.ins().fsub(cf, rf),
                AssignOp::MulSet => self.builder.ins().fmul(cf, rf),
                AssignOp::DivSet => self.builder.ins().fdiv(cf, rf),
                _ => rf,
            };
            return self.builder.ins().bitcast(types::I64, MemFlags::new(), res);
        }
        let is_unsigned = sk.is_some_and(|k| k.is_unsigned());
        let is_signed_narrow = matches!(sk, Some(ScalarKind::I8 | ScalarKind::I16 | ScalarKind::I32));
        let out = match op {
            AssignOp::Set => rhs,
            AssignOp::AddSet => self.builder.ins().iadd(cur, rhs),
            AssignOp::SubSet => self.builder.ins().isub(cur, rhs),
            AssignOp::MulSet => self.builder.ins().imul(cur, rhs),
            AssignOp::DivSet => {
                if is_unsigned {
                    self.builder.ins().udiv(cur, rhs)
                } else {
                    self.builder.ins().sdiv(cur, rhs)
                }
            }
            AssignOp::BitAndSet => self.builder.ins().band(cur, rhs),
            AssignOp::BitOrSet => self.builder.ins().bor(cur, rhs),
            AssignOp::BitXorSet => self.builder.ins().bxor(cur, rhs),
            AssignOp::ShlSet => self.builder.ins().ishl(cur, rhs),
            AssignOp::ShrSet => {
                if is_signed_narrow {
                    self.builder.ins().sshr(cur, rhs)
                } else {
                    self.builder.ins().ushr(cur, rhs)
                }
            }
        };
        if let Some(k) = sk {
            self.normalize_scalar(out, k)
        } else {
            out
        }
    }

    fn infer_match_binding_type(&self, scrutinee_ty: &TypeExpr, variant_name: &str) -> TypeExpr {
        let base = resolve_alias_ty(scrutinee_ty, self.alias_bases);
        if variant_name == "Some" {
            if let TypeExpr::Named { name, args, .. } = &base {
                if name == "Option" && !args.is_empty() {
                    return args[0].clone();
                }
            }
        }
        if variant_name == "Ok" {
            if let TypeExpr::Named { name, args, .. } = &base {
                if name == "Result" && !args.is_empty() {
                    return args[0].clone();
                }
            }
        }
        if variant_name == "Err" {
            if let TypeExpr::Named { name, args, .. } = &base {
                if name == "Result" && args.len() >= 2 {
                    return args[1].clone();
                }
            }
        }
        if let Some(vl) = self.enum_variants.get(variant_name) {
            if let Some(pty) = vl.payload_tys.first() {
                return pty.clone();
            }
        }
        named_ty("Any")
    }

    fn lower_block(
        &mut self,
        blk: &Block,
        want_owned_tail: bool,
        expected_tail_ty: Option<&TypeExpr>,
    ) -> Result<Value, String> {
        let saved = self.save_lexical();
        let scope_idx = self.rc_scopes.len();
        self.rc_scopes.push(Vec::new());

        let mut last = self.builder.ins().iconst(types::I64, 0);
        for stmt in &blk.stmts {
            if self.terminated {
                break;
            }
            match stmt {
                Stmt::Let {
                    name, ty, value, ..
                } => {
                    let effective_ty = ty.clone().unwrap_or_else(|| self.infer_expr_type(value));
                    let produces_rc = self.type_is_rc(&effective_ty) || self.expr_produces_rc(value);
                    let mut v = if produces_rc {
                        self.lower_expr_owned(value, ty.as_ref())?
                    } else {
                        let mut temps = Vec::new();
                        let val = self.lower_expr_borrowed(value, &mut temps)?;
                        self.flush_temps(&temps);
                        val
                    };
                    if let Some(sk) = scalar_kind_of_ty(&effective_ty, self.alias_bases) {
                        if sk.is_int() {
                            v = self.normalize_scalar(v, sk);
                        }
                    }
                    let var = self.declare_fresh_var(name, types::I64);
                    self.builder.def_var(var, v);

                    self.var_types.insert(name.clone(), effective_ty);
                    self.owned_rc_vars.remove(name);
                    self.borrowed_rc_vars.remove(name);
                    if produces_rc {
                        self.owned_rc_vars.insert(name.clone());
                        self.rc_scopes[scope_idx].push((name.clone(), var));
                    }
                }
                Stmt::Assign {
                    target, op, value, ..
                } => match &target.kind {
                    ExprKind::Ident(name) => {
                        let consuming_push = matches!(
                            &value.kind,
                            ExprKind::Call { callee, args }
                            if matches!(&callee.kind, ExprKind::Ident(f) if f == "push")
                                && args.len() == 2
                                && args.first().is_some_and(|a| matches!(&a.value.kind, ExprKind::Ident(s) if s == name))
                        );
                        let consuming_pop = matches!(
                            &value.kind,
                            ExprKind::Call { callee, args }
                            if matches!(&callee.kind, ExprKind::Ident(f) if f == "list_pop_shrink")
                                && args.len() == 1
                                && args.first().is_some_and(|a| matches!(&a.value.kind, ExprKind::Ident(s) if s == name))
                        );
                        let consuming_cow = matches!(
                            &value.kind,
                            ExprKind::Call { callee, args }
                            if matches!(&callee.kind, ExprKind::Ident(f) if is_consuming_cow_builtin(f))
                                && !args.is_empty()
                                && args.first().is_some_and(|a| matches!(&a.value.kind, ExprKind::Ident(s) if s == name))
                        );
                        let consuming_in_place = consuming_push || consuming_pop || consuming_cow;
                        let target_ty = self.var_types.get(name).cloned();
                        let is_inout = self.inout_vars.contains(name);
                        let target_is_rc = self.owned_rc_vars.contains(name);
                        let inout_is_rc = is_inout
                            && target_ty.as_ref().is_some_and(|t| self.type_is_rc(t));
                        let rhs = if consuming_push {
                            let ExprKind::Call { args, .. } = &value.kind else {
                                unreachable!()
                            };
                            let elem_is_rc = target_ty
                                .as_ref()
                                .and_then(|t| self.list_elem_ty(t))
                                .is_some_and(|et| self.type_is_rc(&et))
                                || self.expr_produces_rc(&args[1].value);
                            let xs_val = self.load_ident_val(name)?;
                            let val_v = if elem_is_rc {
                                let mv = self.builder.ins().iconst(types::I64, RC_MASK_ALL as i64);
                                self.call_rt_2_void(self.rt.rc_set_rc_mask, xs_val, mv);
                                self.lower_expr_owned(&args[1].value, None)?
                            } else {
                                let mut temps = Vec::new();
                                let v = self.lower_expr_borrowed(&args[1].value, &mut temps)?;
                                self.flush_temps(&temps);
                                v
                            };
                            self.call_rt_2_vals(self.rt.list_push, xs_val, val_v)
                        } else if consuming_pop {
                            let xs_val = self.load_ident_val(name)?;
                            self.call_rt_1(self.rt.list_pop_shrink, xs_val)
                        } else if consuming_cow {
                            let ExprKind::Call { callee, args } = &value.kind else {
                                unreachable!()
                            };
                            let ExprKind::Ident(fname) = &callee.kind else {
                                unreachable!()
                            };
                            let fid = *self
                                .func_ids
                                .get(fname)
                                .ok_or_else(|| format!("unknown cow builtin `{}`", fname))?;
                            let mut temps = Vec::new();
                            let mut call_args = Vec::with_capacity(args.len());
                            call_args.push(self.load_ident_val(name)?);
                            for a in &args[1..] {
                                call_args.push(self.lower_expr_borrowed(&a.value, &mut temps)?);
                            }
                            let fref = self.module.declare_func_in_func(fid, self.builder.func);
                            let call = self.builder.ins().call(fref, &call_args);
                            let ret = self.builder.inst_results(call)[0];
                            self.flush_temps(&temps);
                            ret
                        } else if (target_is_rc || inout_is_rc) && *op == AssignOp::Set {
                            self.lower_expr_owned(value, target_ty.as_ref())?
                        } else {
                            let mut temps = Vec::new();
                            let v = self.lower_expr_borrowed(value, &mut temps)?;
                            self.flush_temps(&temps);
                            v
                        };

                        let var = *self
                            .vars
                            .get(name)
                            .ok_or_else(|| format!("undefined variable `{}` in assignment", name))?;
                        let sk = target_ty
                            .as_ref()
                            .and_then(|t| scalar_kind_of_ty(t, self.alias_bases))
                            .or_else(|| self.scalar_kind_of_expr(value));
                        if target_is_rc && !consuming_in_place && !is_inout {
                            let old = self.builder.use_var(var);
                            self.builder.def_var(var, rhs);
                            self.call_rt_1_void(self.rt.rc_dec, old);
                        } else if is_inout {
                            let slot_ptr = self.builder.use_var(var);
                            if inout_is_rc && !consuming_in_place && *op == AssignOp::Set {
                                let old = self.builder.ins().load(types::I64, MemFlags::new(), slot_ptr, 0);
                                self.builder.ins().store(MemFlags::new(), rhs, slot_ptr, 0);
                                self.call_rt_1_void(self.rt.rc_dec, old);
                            } else {
                                let is_flt = target_ty
                                    .as_ref()
                                    .is_some_and(|t| is_float_type_expr(t, self.alias_bases))
                                    || self.is_float_expr(value);
                                let new_v = if *op == AssignOp::Set {
                                    if let Some(k) = sk {
                                        if k.is_int() { self.normalize_scalar(rhs, k) } else { rhs }
                                    } else {
                                        rhs
                                    }
                                } else {
                                    let cur = self.builder.ins().load(types::I64, MemFlags::new(), slot_ptr, 0);
                                    self.emit_compound_op(cur, rhs, *op, is_flt, sk)
                                };
                                self.builder.ins().store(MemFlags::new(), new_v, slot_ptr, 0);
                            }
                        } else {
                            let is_flt = target_ty
                                .as_ref()
                                .is_some_and(|t| is_float_type_expr(t, self.alias_bases))
                                || self.is_float_expr(value);
                            let new_v = if *op == AssignOp::Set {
                                if let Some(k) = sk {
                                    if k.is_int() { self.normalize_scalar(rhs, k) } else { rhs }
                                } else {
                                    rhs
                                }
                            } else {
                                let cur = self.builder.use_var(var);
                                self.emit_compound_op(cur, rhs, *op, is_flt, sk)
                            };
                            self.builder.def_var(var, new_v);
                        }
                    }
                    ExprKind::Field { target: obj, field } => {
                        if let ExprKind::Ident(vname) = &obj.kind {
                            let c_field_info = match resolve_alias_ty(&self.infer_expr_type(obj), self.alias_bases) {
                                TypeExpr::Named { name: rname, .. } => self
                                    .records
                                    .get(&rname)
                                    .and_then(|rl| rl.c_layout.as_ref())
                                    .and_then(|cl| cl.fields.iter().find(|f| f.name == *field))
                                    .cloned(),
                                _ => None,
                            };
                            if let Some(cfl) = c_field_info {
                                let mut temps = Vec::new();
                                let rhs = self.lower_expr_borrowed(value, &mut temps)?;
                                let is_inout = self.inout_vars.contains(vname);
                                let var = *self
                                    .vars
                                    .get(vname)
                                    .ok_or_else(|| format!("undefined variable `{}`", vname))?;
                                let cur_handle = if is_inout {
                                    let slot_ptr = self.builder.use_var(var);
                                    self.builder.ins().load(types::I64, MemFlags::new(), slot_ptr, 0)
                                } else {
                                    self.builder.use_var(var)
                                };
                                let uniq_handle = self.call_rt_1(self.rt.rc_ensure_unique, cur_handle);
                                if is_inout {
                                    let slot_ptr = self.builder.use_var(var);
                                    self.builder.ins().store(MemFlags::new(), uniq_handle, slot_ptr, 0);
                                } else {
                                    self.builder.def_var(var, uniq_handle);
                                }
                                let base = self.call_rt_1(self.rt.cstruct_data_ptr, uniq_handle);
                                let sk = scalar_kind_of_ty(&cfl.ty, self.alias_bases);
                                let is_flt = is_float_type_expr(&cfl.ty, self.alias_bases) || self.is_float_expr(value);
                                let new_val = if *op == AssignOp::Set {
                                    rhs
                                } else {
                                    let cur_f = self.emit_mem_load(base, cfl.byte_offset as i32, &cfl.ty, false)?;
                                    self.emit_compound_op(cur_f, rhs, *op, is_flt, sk)
                                };
                                self.emit_mem_store(base, cfl.byte_offset as i32, new_val, &cfl.ty, false)?;
                                self.flush_temps(&temps);
                            } else {
                                let (fidx, fl) = self.resolve_field_layout(obj, field)?;
                                let field_is_rc = fl.as_ref().is_some_and(|f| f.is_rc);
                                let field_ty = fl.as_ref().map(|f| &f.ty);
                                let rhs = if field_is_rc && *op == AssignOp::Set {
                                    self.lower_expr_owned(value, field_ty)?
                                } else {
                                    let mut temps = Vec::new();
                                    let v = self.lower_expr_borrowed(value, &mut temps)?;
                                    self.flush_temps(&temps);
                                    v
                                };
                                let is_inout = self.inout_vars.contains(vname);
                                let var = *self
                                    .vars
                                    .get(vname)
                                    .ok_or_else(|| format!("undefined variable `{}`", vname))?;
                                let cur_handle = if is_inout {
                                    let slot_ptr = self.builder.use_var(var);
                                    self.builder
                                        .ins()
                                        .load(types::I64, MemFlags::new(), slot_ptr, 0)
                                } else {
                                    self.builder.use_var(var)
                                };
                                let uniq_handle = self.call_rt_1(self.rt.rc_ensure_unique, cur_handle);
                                if is_inout {
                                    let slot_ptr = self.builder.use_var(var);
                                    self.builder
                                        .ins()
                                        .store(MemFlags::new(), uniq_handle, slot_ptr, 0);
                                } else {
                                    self.builder.def_var(var, uniq_handle);
                                }
                                let sk = field_ty.and_then(|t| scalar_kind_of_ty(t, self.alias_bases));
                                let is_flt = field_ty
                                    .is_some_and(|t| is_float_type_expr(t, self.alias_bases))
                                    || self.is_float_expr(value);
                                let new_field_val = if *op == AssignOp::Set {
                                    rhs
                                } else {
                                    let idx_v = self.builder.ins().iconst(types::I64, fidx as i64);
                                    let cur_f =
                                        self.call_rt_2_vals(self.rt.rc_get_field, uniq_handle, idx_v);
                                    self.emit_compound_op(cur_f, rhs, *op, is_flt, sk)
                                };
                                self.call_rt_3_void(
                                    self.rt.rc_set_field,
                                    uniq_handle,
                                    fidx as i64,
                                    new_field_val,
                                );
                            }
                        }
                    }
                    ExprKind::Index {
                        target: list_expr,
                        index,
                    } => {
                        let mut idx_temps = Vec::new();
                        let idx_v = self.lower_expr_borrowed(index, &mut idx_temps)?;
                        self.flush_temps(&idx_temps);
                        match &list_expr.kind {
                            ExprKind::Ident(vname) => {
                                let arr_info = self
                                    .var_types
                                    .get(vname)
                                    .cloned()
                                    .and_then(|t| self.array_layout(&t));
                                if let Some((elem_ty, _len, stride, false)) = arr_info {
                                    let mut temps = Vec::new();
                                    let rhs = self.lower_expr_borrowed(value, &mut temps)?;
                                    let cur_arr = self.load_ident_val(vname)?;
                                    let uniq_arr = self.call_rt_1(self.rt.rc_ensure_unique, cur_arr);
                                    let var = self.vars[vname];
                                    if self.inout_vars.contains(vname) {
                                        let slot_ptr = self.builder.use_var(var);
                                        self.builder.ins().store(MemFlags::new(), uniq_arr, slot_ptr, 0);
                                    } else {
                                        self.builder.def_var(var, uniq_arr);
                                    }
                                    let base = self.call_rt_1(self.rt.cstruct_data_ptr, uniq_arr);
                                    let sv = self.builder.ins().iconst(types::I64, stride as i64);
                                    let boff = self.builder.ins().imul(idx_v, sv);
                                    let eptr = self.builder.ins().iadd(base, boff);
                                    let sk = scalar_kind_of_ty(&elem_ty, self.alias_bases);
                                    let is_flt = is_float_type_expr(&elem_ty, self.alias_bases) || self.is_float_expr(value);
                                    let new_val = if *op == AssignOp::Set {
                                        rhs
                                    } else {
                                        let cur_e = self.emit_mem_load(eptr, 0, &elem_ty, false)?;
                                        self.emit_compound_op(cur_e, rhs, *op, is_flt, sk)
                                    };
                                    self.emit_mem_store(eptr, 0, new_val, &elem_ty, false)?;
                                    self.flush_temps(&temps);
                                } else {
                                    let elem_ty = self
                                        .var_types
                                        .get(vname)
                                        .and_then(|t| self.list_elem_ty(t));
                                    let elem_is_rc = elem_ty
                                        .as_ref()
                                        .is_some_and(|et| self.type_is_rc(et))
                                        || self.expr_produces_rc(value);
                                    let rhs = if elem_is_rc && *op == AssignOp::Set {
                                        self.lower_expr_owned(value, elem_ty.as_ref())?
                                    } else {
                                        let mut temps = Vec::new();
                                        let v = self.lower_expr_borrowed(value, &mut temps)?;
                                        self.flush_temps(&temps);
                                        v
                                    };
                                    let cur_list = self.load_ident_val(vname)?;
                                    let uniq_list = self.call_rt_1(self.rt.rc_ensure_unique, cur_list);
                                    let var = self.vars[vname];
                                    if self.inout_vars.contains(vname) {
                                        let slot_ptr = self.builder.use_var(var);
                                        self.builder
                                            .ins()
                                            .store(MemFlags::new(), uniq_list, slot_ptr, 0);
                                    } else {
                                        self.builder.def_var(var, uniq_list);
                                    }
                                    if elem_is_rc {
                                        let mv = self.builder.ins().iconst(types::I64, RC_MASK_ALL as i64);
                                        self.call_rt_2_void(self.rt.rc_set_rc_mask, uniq_list, mv);
                                    }
                                    let sk = elem_ty.as_ref().and_then(|t| scalar_kind_of_ty(t, self.alias_bases));
                                    let is_flt = elem_ty
                                        .as_ref()
                                        .is_some_and(|t| is_float_type_expr(t, self.alias_bases))
                                        || self.is_float_expr(value);
                                    let new_val = if *op == AssignOp::Set {
                                        rhs
                                    } else {
                                        let cur_e = self.call_rt_2_vals(self.rt.rc_get_field, uniq_list, idx_v);
                                        self.emit_compound_op(cur_e, rhs, *op, is_flt, sk)
                                    };
                                    let fref = self
                                        .module
                                        .declare_func_in_func(self.rt.rc_set_field, self.builder.func);
                                    self.builder.ins().call(fref, &[uniq_list, idx_v, new_val]);
                                }
                            }
                            ExprKind::Field { target: obj, field }
                                if matches!(&obj.kind, ExprKind::Ident(_)) =>
                            {
                                let ExprKind::Ident(vname) = &obj.kind else {
                                    unreachable!()
                                };
                                let (fidx, fl) = self.resolve_field_layout(obj, field)?;
                                let elem_ty = fl.as_ref().and_then(|f| self.list_elem_ty(&f.ty));
                                let elem_is_rc = elem_ty
                                    .as_ref()
                                    .is_some_and(|et| self.type_is_rc(et))
                                    || self.expr_produces_rc(value);
                                let rhs = if elem_is_rc && *op == AssignOp::Set {
                                    self.lower_expr_owned(value, elem_ty.as_ref())?
                                } else {
                                    let mut temps = Vec::new();
                                    let v = self.lower_expr_borrowed(value, &mut temps)?;
                                    self.flush_temps(&temps);
                                    v
                                };
                                let cur_obj = self.load_ident_val(vname)?;
                                let uniq_obj = self.call_rt_1(self.rt.rc_ensure_unique, cur_obj);
                                let var = self.vars[vname];
                                if self.inout_vars.contains(vname) {
                                    let slot_ptr = self.builder.use_var(var);
                                    self.builder
                                        .ins()
                                        .store(MemFlags::new(), uniq_obj, slot_ptr, 0);
                                } else {
                                    self.builder.def_var(var, uniq_obj);
                                }
                                let fidx_v = self.builder.ins().iconst(types::I64, fidx as i64);
                                let cur_list = self.call_rt_2_vals(self.rt.rc_get_field, uniq_obj, fidx_v);
                                let uniq_list = self.call_rt_1(self.rt.rc_ensure_unique, cur_list);
                                self.call_rt_3_void(
                                    self.rt.rc_set_field_raw,
                                    uniq_obj,
                                    fidx as i64,
                                    uniq_list,
                                );
                                if elem_is_rc {
                                    let mv = self.builder.ins().iconst(types::I64, RC_MASK_ALL as i64);
                                    self.call_rt_2_void(self.rt.rc_set_rc_mask, uniq_list, mv);
                                }
                                let sk = elem_ty.as_ref().and_then(|t| scalar_kind_of_ty(t, self.alias_bases));
                                let is_flt = elem_ty
                                    .as_ref()
                                    .is_some_and(|t| is_float_type_expr(t, self.alias_bases))
                                    || self.is_float_expr(value);
                                let new_val = if *op == AssignOp::Set {
                                    rhs
                                } else {
                                    let cur_e = self.call_rt_2_vals(self.rt.rc_get_field, uniq_list, idx_v);
                                    self.emit_compound_op(cur_e, rhs, *op, is_flt, sk)
                                };
                                let fref = self
                                    .module
                                    .declare_func_in_func(self.rt.rc_set_field, self.builder.func);
                                self.builder.ins().call(fref, &[uniq_list, idx_v, new_val]);
                            }
                            _ => {}
                        }
                    }
                    _ => {}
                },
                Stmt::Mutate { body, .. } => {
                    let _ = self.lower_block(body, false, None)?;
                }
                Stmt::Return { value, .. } => {
                    let mut move_skip_var: Option<Variable> = None;
                    let ret_ty_opt = self.fn_ret_ty.clone();
                    let v = match value {
                        Some(e) => {
                            if self.fn_returns_rc {
                                if let ExprKind::Ident(rname) = &e.kind {
                                    if !self.inout_vars.contains(rname) {
                                        if let Some(&rv) = self.vars.get(rname) {
                                            if self.find_owned_var_in_scopes(rv, 0) {
                                                move_skip_var = Some(rv);
                                                self.builder.use_var(rv)
                                            } else {
                                                self.lower_expr_owned(e, ret_ty_opt.as_ref())?
                                            }
                                        } else {
                                            self.lower_expr_owned(e, ret_ty_opt.as_ref())?
                                        }
                                    } else {
                                        self.lower_expr_owned(e, ret_ty_opt.as_ref())?
                                    }
                                } else {
                                    self.lower_expr_owned(e, ret_ty_opt.as_ref())?
                                }
                            } else {
                                let mut temps = Vec::new();
                                let val = self.lower_expr_borrowed(e, &mut temps)?;
                                self.flush_temps(&temps);
                                val
                            }
                        }
                        None => self.builder.ins().iconst(types::I64, 0),
                    };
                    self.emit_cleanup_temp_scopes_from(0);
                    self.emit_cleanup_scopes_from(0, move_skip_var);
                    self.builder.ins().return_(&[v]);
                    self.terminated = true;
                    last = v;
                    break;
                }
                Stmt::Break { .. } => {
                    let (exit_blk, rc_depth, temp_depth) = self
                        .loop_stack
                        .last()
                        .map(|l| (l.exit_blk, l.rc_scope_depth, l.temp_scope_depth))
                        .ok_or_else(|| "`break` used outside of `while` loop".to_string())?;
                    self.emit_cleanup_temp_scopes_from(temp_depth);
                    self.emit_cleanup_scopes_from(rc_depth, None);
                    self.builder.ins().jump(exit_blk, &[]);
                    self.terminated = true;
                    break;
                }
                Stmt::Continue { .. } => {
                    let (hdr_blk, rc_depth, temp_depth) = self
                        .loop_stack
                        .last()
                        .map(|l| (l.hdr_blk, l.rc_scope_depth, l.temp_scope_depth))
                        .ok_or_else(|| "`continue` used outside of `while` loop".to_string())?;
                    self.emit_cleanup_temp_scopes_from(temp_depth);
                    self.emit_cleanup_scopes_from(rc_depth, None);
                    self.builder.ins().jump(hdr_blk, &[]);
                    self.terminated = true;
                    break;
                }
                Stmt::Assert { cond, span } => {
                    let mut temps = Vec::new();
                    let c = self.lower_expr_borrowed(cond, &mut temps)?;
                    self.flush_temps(&temps);
                    let zero = self.builder.ins().iconst(types::I64, 0);
                    let is_ok = self.builder.ins().icmp(IntCC::NotEqual, c, zero);
                    let fail_blk = self.builder.create_block();
                    let cont_blk = self.builder.create_block();
                    self.builder
                        .ins()
                        .brif(is_ok, cont_blk, &[], fail_blk, &[]);
                    self.builder.switch_to_block(fail_blk);
                    self.builder.seal_block(fail_blk);
                    let line_val = self.builder.ins().iconst(types::I64, span.line as i64);
                    self.call_rt_1_void(self.rt.assert_fail, line_val);
                    self.builder.ins().jump(cont_blk, &[]);
                    self.builder.switch_to_block(cont_blk);
                    self.builder.seal_block(cont_blk);
                }
                Stmt::While { cond, body, .. } => {
                    let hdr = self.builder.create_block();
                    let loop_blk = self.builder.create_block();
                    let exit_blk = self.builder.create_block();
                    self.builder.ins().jump(hdr, &[]);
                    self.builder.switch_to_block(hdr);
                    let mut cond_temps = Vec::new();
                    let c = self.lower_expr_borrowed(cond, &mut cond_temps)?;
                    self.flush_temps(&cond_temps);
                    let zero = self.builder.ins().iconst(types::I64, 0);
                    let cmp = self.builder.ins().icmp(IntCC::NotEqual, c, zero);
                    self.builder
                        .ins()
                        .brif(cmp, loop_blk, &[], exit_blk, &[]);
                    self.builder.switch_to_block(loop_blk);
                    self.builder.seal_block(loop_blk);

                    self.loop_stack.push(LoopCtx {
                        hdr_blk: hdr,
                        exit_blk,
                        rc_scope_depth: self.rc_scopes.len(),
                        temp_scope_depth: self.temp_rc_scopes.len(),
                    });
                    let _ = self.lower_block(body, false, None)?;
                    self.loop_stack.pop();

                    if !self.terminated {
                        self.builder.ins().jump(hdr, &[]);
                    }
                    self.terminated = false;
                    self.builder.seal_block(hdr);
                    self.builder.switch_to_block(exit_blk);
                    self.builder.seal_block(exit_blk);
                }
                Stmt::Expr { expr, .. } => {
                    let mut temps = Vec::new();
                    last = self.lower_expr_borrowed(expr, &mut temps)?;
                    if !self.terminated {
                        self.flush_temps(&temps);
                    }
                }
                Stmt::LetTuple { .. } | Stmt::For { .. } | Stmt::Defer { .. } | Stmt::Ghost { .. } => {}
            }
        }

        let mut move_skip_var: Option<Variable> = None;
        if !self.terminated {
            if let Some(tail) = &blk.tail {
                if want_owned_tail {
                    if let ExprKind::Ident(tname) = &tail.kind {
                        if !self.inout_vars.contains(tname) {
                            if let Some(&tv) = self.vars.get(tname) {
                                if self.find_owned_var_in_scopes(tv, scope_idx) {
                                    move_skip_var = Some(tv);
                                    last = self.builder.use_var(tv);
                                } else {
                                    last = self.lower_expr_owned(tail, expected_tail_ty)?;
                                }
                            } else {
                                last = self.lower_expr_owned(tail, expected_tail_ty)?;
                            }
                        } else {
                            last = self.lower_expr_owned(tail, expected_tail_ty)?;
                        }
                    } else {
                        last = self.lower_expr_owned(tail, expected_tail_ty)?;
                    }
                } else {
                    let mut temps = Vec::new();
                    last = self.lower_expr_borrowed(tail, &mut temps)?;
                    if !self.terminated {
                        self.flush_temps(&temps);
                    }
                }
            }
            if !self.terminated {
                self.emit_cleanup_scopes_from(scope_idx, move_skip_var);
            }
        }

        self.rc_scopes.pop();
        self.restore_lexical(saved);
        Ok(last)
    }

    fn lower_expr_owned(
        &mut self,
        expr: &Expr,
        expected_ty: Option<&TypeExpr>,
    ) -> Result<Value, String> {
        match &expr.kind {
            ExprKind::Ident(s) => {
                if s == "self" {
                    if let Some(sv) = self.self_var.clone() {
                        let var = self.vars[&sv];
                        return Ok(self.builder.use_var(var));
                    }
                }
                if s == "None" {
                    return Ok(self.call_rt_2(self.rt.rc_alloc, TAG_NONE, 0));
                }
                if let Some(&etag) = self.enum_tags.get(s) {
                    if !self.vars.contains_key(s) {
                        return Ok(self.call_rt_2(self.rt.rc_alloc, etag, 0));
                    }
                }
                let val = self.load_ident_val(s)?;
                if self.is_rc_ident(s) {
                    Ok(self.call_rt_1(self.rt.rc_inc, val))
                } else {
                    Ok(val)
                }
            }
            ExprKind::Field { target, field } => {
                if let ExprKind::Ident(ename) = &target.kind {
                    let key = format!("{}.{}", ename, field);
                    if let Some(&tag) = self.enum_tags.get(&key) {
                        return Ok(self.call_rt_2(self.rt.rc_alloc, tag, 0));
                    }
                }
                let c_field_info = match resolve_alias_ty(&self.infer_expr_type(target), self.alias_bases) {
                    TypeExpr::Named { name: rname, .. } => self
                        .records
                        .get(&rname)
                        .and_then(|rl| rl.c_layout.as_ref())
                        .and_then(|cl| cl.fields.iter().find(|f| f.name == *field))
                        .cloned(),
                    _ => None,
                };
                if let Some(cfl) = c_field_info {
                    let mut temps = Vec::new();
                    let obj = self.lower_expr_borrowed(target, &mut temps)?;
                    let base = self.call_rt_1(self.rt.cstruct_data_ptr, obj);
                    let val = self.emit_mem_load(base, cfl.byte_offset as i32, &cfl.ty, false)?;
                    self.flush_temps(&temps);
                    return Ok(val);
                }
                let (idx, fl) = self.resolve_field_layout(target, field)?;
                let mut temps = Vec::new();
                let obj = self.lower_expr_borrowed(target, &mut temps)?;
                let idx_v = self.builder.ins().iconst(types::I64, idx as i64);
                let mut val = self.call_rt_2_vals(self.rt.rc_get_field, obj, idx_v);
                if fl.as_ref().is_some_and(|f| f.is_rc) {
                    val = self.call_rt_1(self.rt.rc_inc, val);
                }
                self.flush_temps(&temps);
                Ok(val)
            }
            ExprKind::Index { target, index } => {
                let target_ty = self.infer_expr_type(target);
                if let Some((elem_ty, _len, stride, false)) = self.array_layout(&target_ty) {
                    let mut temps = Vec::new();
                    let t = self.lower_expr_borrowed(target, &mut temps)?;
                    let i = self.lower_expr_borrowed(index, &mut temps)?;
                    let base = self.call_rt_1(self.rt.cstruct_data_ptr, t);
                    let sv = self.builder.ins().iconst(types::I64, stride as i64);
                    let boff = self.builder.ins().imul(i, sv);
                    let eptr = self.builder.ins().iadd(base, boff);
                    let res = self.emit_mem_load(eptr, 0, &elem_ty, false)?;
                    self.flush_temps(&temps);
                    return Ok(res);
                }
                let elem_is_rc = self
                    .list_elem_ty(&target_ty)
                    .is_some_and(|et| self.type_is_rc(&et));
                let mut temps = Vec::new();
                let t = self.lower_expr_borrowed(target, &mut temps)?;
                let i = self.lower_expr_borrowed(index, &mut temps)?;
                let mut res = self.call_rt_2_vals(self.rt.rc_get_field, t, i);
                if elem_is_rc {
                    res = self.call_rt_1(self.rt.rc_inc, res);
                }
                self.flush_temps(&temps);
                Ok(res)
            }
            ExprKind::TupleIndex { target, index } => {
                let elem_is_rc = match resolve_alias_ty(&self.infer_expr_type(target), self.alias_bases) {
                    TypeExpr::Tuple(es, _) => es.get(*index).is_some_and(|et| self.type_is_rc(et)),
                    _ => expected_ty.is_some_and(|t| self.type_is_rc(t)),
                };
                let mut temps = Vec::new();
                let t = self.lower_expr_borrowed(target, &mut temps)?;
                let idx_v = self.builder.ins().iconst(types::I64, *index as i64);
                let mut res = self.call_rt_2_vals(self.rt.rc_get_field, t, idx_v);
                if elem_is_rc {
                    res = self.call_rt_1(self.rt.rc_inc, res);
                }
                self.flush_temps(&temps);
                Ok(res)
            }
            ExprKind::Call { callee, args }
                if matches!(&callee.kind, ExprKind::Ident(f) if f == "__closure_get") && args.len() == 2 =>
            {
                let mut temps = Vec::new();
                let env = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                let idx = self.lower_expr_borrowed(&args[1].value, &mut temps)?;
                let mut res = self.call_rt_2_vals(self.rt.rc_get_field, env, idx);
                if expected_ty.is_some_and(|t| self.type_is_rc(t)) {
                    res = self.call_rt_1(self.rt.rc_inc, res);
                }
                self.flush_temps(&temps);
                Ok(res)
            }
            ExprKind::If {
                cond,
                then_block,
                else_branch,
            } => self.lower_if(cond, then_block, else_branch.as_deref(), true, expected_ty),
            ExprKind::Match { scrutinee, arms } => {
                self.lower_match(scrutinee, arms, true, expected_ty)
            }
            ExprKind::Block(blk) | ExprKind::Unsafe(blk) => self.lower_block(blk, true, expected_ty),
            _ => self.lower_expr_fresh(expr, expected_ty),
        }
    }

    fn lower_expr_borrowed(&mut self, expr: &Expr, temps: &mut Vec<Value>) -> Result<Value, String> {
        match &expr.kind {
            ExprKind::Int(n) => Ok(self.builder.ins().iconst(types::I64, *n)),
            ExprKind::Float(x) => Ok(self.builder.ins().iconst(types::I64, x.to_bits() as i64)),
            ExprKind::Bool(b) => Ok(self
                .builder
                .ins()
                .iconst(types::I64, if *b { 1 } else { 0 })),
            ExprKind::Ident(s) => {
                if s == "self" {
                    if let Some(sv) = self.self_var.clone() {
                        let var = self.vars[&sv];
                        return Ok(self.builder.use_var(var));
                    }
                }
                if s == "None" {
                    let v = self.call_rt_2(self.rt.rc_alloc, TAG_NONE, 0);
                    temps.push(v);
                    return Ok(v);
                }
                if let Some(&etag) = self.enum_tags.get(s) {
                    if !self.vars.contains_key(s) {
                        let v = self.call_rt_2(self.rt.rc_alloc, etag, 0);
                        temps.push(v);
                        return Ok(v);
                    }
                }
                self.load_ident_val(s)
            }
            ExprKind::Field { target, field } => {
                if let ExprKind::Ident(ename) = &target.kind {
                    let key = format!("{}.{}", ename, field);
                    if let Some(&tag) = self.enum_tags.get(&key) {
                        let v = self.call_rt_2(self.rt.rc_alloc, tag, 0);
                        temps.push(v);
                        return Ok(v);
                    }
                }
                let c_field_info = match resolve_alias_ty(&self.infer_expr_type(target), self.alias_bases) {
                    TypeExpr::Named { name: rname, .. } => self
                        .records
                        .get(&rname)
                        .and_then(|rl| rl.c_layout.as_ref())
                        .and_then(|cl| cl.fields.iter().find(|f| f.name == *field))
                        .cloned(),
                    _ => None,
                };
                if let Some(cfl) = c_field_info {
                    let obj = self.lower_expr_borrowed(target, temps)?;
                    let base = self.call_rt_1(self.rt.cstruct_data_ptr, obj);
                    let val = self.emit_mem_load(base, cfl.byte_offset as i32, &cfl.ty, false)?;
                    if self.is_c_value_type(&cfl.ty) {
                        temps.push(val);
                    }
                    return Ok(val);
                }
                let (idx, _) = self.resolve_field_layout(target, field)?;
                let obj = self.lower_expr_borrowed(target, temps)?;
                let idx_v = self.builder.ins().iconst(types::I64, idx as i64);
                Ok(self.call_rt_2_vals(self.rt.rc_get_field, obj, idx_v))
            }
            ExprKind::Index { target, index } => {
                let target_ty = self.infer_expr_type(target);
                if let Some((elem_ty, _len, stride, false)) = self.array_layout(&target_ty) {
                    let t = self.lower_expr_borrowed(target, temps)?;
                    let i = self.lower_expr_borrowed(index, temps)?;
                    let base = self.call_rt_1(self.rt.cstruct_data_ptr, t);
                    let sv = self.builder.ins().iconst(types::I64, stride as i64);
                    let boff = self.builder.ins().imul(i, sv);
                    let eptr = self.builder.ins().iadd(base, boff);
                    let val = self.emit_mem_load(eptr, 0, &elem_ty, false)?;
                    if self.is_c_value_type(&elem_ty) {
                        temps.push(val);
                    }
                    return Ok(val);
                }
                let t = self.lower_expr_borrowed(target, temps)?;
                let i = self.lower_expr_borrowed(index, temps)?;
                Ok(self.call_rt_2_vals(self.rt.rc_get_field, t, i))
            }
            ExprKind::TupleIndex { target, index } => {
                let t = self.lower_expr_borrowed(target, temps)?;
                let idx_v = self.builder.ins().iconst(types::I64, *index as i64);
                Ok(self.call_rt_2_vals(self.rt.rc_get_field, t, idx_v))
            }
            ExprKind::If {
                cond,
                then_block,
                else_branch,
            } => {
                let want_owned = self.expr_produces_rc(expr);
                let v = self.lower_if(cond, then_block, else_branch.as_deref(), want_owned, None)?;
                if want_owned && !self.terminated {
                    temps.push(v);
                }
                Ok(v)
            }
            ExprKind::Match { scrutinee, arms } => {
                let want_owned = self.expr_produces_rc(expr);
                let v = self.lower_match(scrutinee, arms, want_owned, None)?;
                if want_owned && !self.terminated {
                    temps.push(v);
                }
                Ok(v)
            }
            ExprKind::Block(blk) | ExprKind::Unsafe(blk) => {
                let want_owned = self.expr_produces_rc(expr);
                let v = self.lower_block(blk, want_owned, None)?;
                if want_owned && !self.terminated {
                    temps.push(v);
                }
                Ok(v)
            }
            ExprKind::Call { callee, args }
                if args.len() == 1
                    && matches!(
                        &callee.kind,
                        ExprKind::Ident(f) if f == "text_cstr_ptr" || f == "bytes_ptr"
                    ) =>
            {
                let ExprKind::Ident(fname) = &callee.kind else {
                    unreachable!()
                };
                // Keep the backing Text/Bytes temporary alive in caller's `temps`
                // until the enclosing call (e.g. `strlen(text_cstr_ptr(...))`) finishes!
                let arg_v = self.lower_expr_borrowed(&args[0].value, temps)?;
                let fid = *self
                    .func_ids
                    .get(fname)
                    .ok_or_else(|| format!("missing core syscall `{}`", fname))?;
                Ok(self.call_rt_1(fid, arg_v))
            }
            _ => {
                let v = self.lower_expr_fresh(expr, None)?;
                if self.expr_produces_rc(expr) {
                    temps.push(v);
                }
                Ok(v)
            }
        }
    }

    fn load_ident_val(&mut self, s: &str) -> Result<Value, String> {
        let var = *self
            .vars
            .get(s)
            .ok_or_else(|| format!("undefined variable `{}` in codegen", s))?;
        if self.inout_vars.contains(s) {
            let slot_ptr = self.builder.use_var(var);
            Ok(self
                .builder
                .ins()
                .load(types::I64, MemFlags::new(), slot_ptr, 0))
        } else {
            Ok(self.builder.use_var(var))
        }
    }

    fn lower_if(
        &mut self,
        cond: &Expr,
        then_block: &Block,
        else_branch: Option<&Expr>,
        want_owned: bool,
        expected_ty: Option<&TypeExpr>,
    ) -> Result<Value, String> {
        let mut ctemps = Vec::new();
        let c = self.lower_expr_borrowed(cond, &mut ctemps)?;
        self.flush_temps(&ctemps);

        let z = self.builder.ins().iconst(types::I64, 0);
        let cmp = self.builder.ins().icmp(IntCC::NotEqual, c, z);
        let then_b = self.builder.create_block();
        let else_b = self.builder.create_block();
        let merge_b = self.builder.create_block();
        self.builder.append_block_param(merge_b, types::I64);
        self.builder
            .ins()
            .brif(cmp, then_b, &[], else_b, &[]);

        self.builder.switch_to_block(then_b);
        self.builder.seal_block(then_b);
        let tv = self.lower_block(then_block, want_owned, expected_ty)?;
        let then_term = self.terminated;
        if !then_term {
            self.builder.ins().jump(merge_b, &[tv.into()]);
        }
        self.terminated = false;

        self.builder.switch_to_block(else_b);
        self.builder.seal_block(else_b);
        let ev = match else_branch {
            Some(eb) => {
                if want_owned {
                    self.lower_expr_owned(eb, expected_ty)?
                } else {
                    let mut etemps = Vec::new();
                    let v = self.lower_expr_borrowed(eb, &mut etemps)?;
                    if !self.terminated {
                        self.flush_temps(&etemps);
                    }
                    v
                }
            }
            None => self.builder.ins().iconst(types::I64, 0),
        };
        let else_term = self.terminated;
        if !else_term {
            self.builder.ins().jump(merge_b, &[ev.into()]);
        }

        self.builder.switch_to_block(merge_b);
        self.builder.seal_block(merge_b);
        if then_term && else_term {
            self.builder.ins().trap(TrapCode::unwrap_user(1));
            self.terminated = true;
            return Ok(self.builder.block_params(merge_b)[0]);
        }
        self.terminated = false;
        Ok(self.builder.block_params(merge_b)[0])
    }

    fn emit_pattern_check(
        &mut self,
        val: Value,
        ty: &TypeExpr,
        pat: &Pattern,
        fail_b: ClifBlock,
        bindings: &mut Vec<(String, Value, TypeExpr)>,
    ) -> Result<(), String> {
        let base = resolve_alias_ty(ty, self.alias_bases);
        match pat {
            Pattern::Wildcard(_) => Ok(()),
            Pattern::Ident(name, _) => {
                if name != "_" {
                    bindings.push((name.clone(), val, ty.clone()));
                }
                Ok(())
            }
            Pattern::Int(n, _) => {
                let pv = self.builder.ins().iconst(types::I64, *n);
                let eq = self.builder.ins().icmp(IntCC::Equal, val, pv);
                let ok_b = self.builder.create_block();
                self.builder.ins().brif(eq, ok_b, &[], fail_b, &[]);
                self.builder.switch_to_block(ok_b);
                self.builder.seal_block(ok_b);
                Ok(())
            }
            Pattern::Bool(b, _) => {
                let pv = self.builder.ins().iconst(types::I64, if *b { 1 } else { 0 });
                let eq = self.builder.ins().icmp(IntCC::Equal, val, pv);
                let ok_b = self.builder.create_block();
                self.builder.ins().brif(eq, ok_b, &[], fail_b, &[]);
                self.builder.switch_to_block(ok_b);
                self.builder.seal_block(ok_b);
                Ok(())
            }
            Pattern::Str(s, _) => {
                let boxed: Box<[u8]> = s.as_bytes().to_vec().into_boxed_slice();
                let ptr = boxed.as_ptr() as i64;
                let len = boxed.len() as i64;
                self.interned_strings.push(boxed);
                let ptxt = self.call_rt_2(self.rt.text_new, ptr, len);
                let teq = self.call_rt_2_vals(self.rt.text_eq, val, ptxt);
                self.call_rt_1_void(self.rt.rc_dec, ptxt);
                let zero = self.builder.ins().iconst(types::I64, 0);
                let eq = self.builder.ins().icmp(IntCC::NotEqual, teq, zero);
                let ok_b = self.builder.create_block();
                self.builder.ins().brif(eq, ok_b, &[], fail_b, &[]);
                self.builder.switch_to_block(ok_b);
                self.builder.seal_block(ok_b);
                Ok(())
            }
            Pattern::Tuple(subpats, _) => {
                let elems = match &base {
                    TypeExpr::Tuple(es, _) => es.clone(),
                    _ => Vec::new(),
                };
                for (idx, sp) in subpats.iter().enumerate() {
                    if matches!(sp, Pattern::Wildcard(_)) {
                        continue;
                    }
                    let idx_v = self.builder.ins().iconst(types::I64, idx as i64);
                    let ev = self.call_rt_2_vals(self.rt.rc_get_field, val, idx_v);
                    let ety = elems.get(idx).cloned().unwrap_or_else(|| named_ty("Any"));
                    self.emit_pattern_check(ev, &ety, sp, fail_b, bindings)?;
                }
                Ok(())
            }
            Pattern::Variant { name, binding, .. } => {
                let tag = *self
                    .enum_tags
                    .get(name)
                    .ok_or_else(|| format!("unknown enum variant `{}` in match", name))?;
                let actual_tag = self.call_rt_1(self.rt.rc_get_tag, val);
                let pv = self.builder.ins().iconst(types::I64, tag);
                let eq = self.builder.ins().icmp(IntCC::Equal, actual_tag, pv);
                let ok_b = self.builder.create_block();
                self.builder.ins().brif(eq, ok_b, &[], fail_b, &[]);
                self.builder.switch_to_block(ok_b);
                self.builder.seal_block(ok_b);
                if let Some(bname) = binding {
                    if bname != "_" {
                        let zero = self.builder.ins().iconst(types::I64, 0);
                        let payload = self.call_rt_2_vals(self.rt.rc_get_field, val, zero);
                        let bty = self.infer_match_binding_type(ty, name);
                        bindings.push((bname.clone(), payload, bty));
                    }
                }
                Ok(())
            }
            Pattern::VariantMulti { name, patterns, .. } => {
                let tag = *self
                    .enum_tags
                    .get(name)
                    .ok_or_else(|| format!("unknown enum variant `{}` in match", name))?;
                let actual_tag = self.call_rt_1(self.rt.rc_get_tag, val);
                let pv = self.builder.ins().iconst(types::I64, tag);
                let eq = self.builder.ins().icmp(IntCC::Equal, actual_tag, pv);
                let ok_b = self.builder.create_block();
                self.builder.ins().brif(eq, ok_b, &[], fail_b, &[]);
                self.builder.switch_to_block(ok_b);
                self.builder.seal_block(ok_b);
                let ptys = self.resolve_variant_payload_tys(&base, name);
                for (idx, sp) in patterns.iter().enumerate() {
                    if matches!(sp, Pattern::Wildcard(_)) {
                        continue;
                    }
                    let idx_v = self.builder.ins().iconst(types::I64, idx as i64);
                    let fv = self.call_rt_2_vals(self.rt.rc_get_field, val, idx_v);
                    let pty = ptys.get(idx).cloned().unwrap_or_else(|| named_ty("Any"));
                    self.emit_pattern_check(fv, &pty, sp, fail_b, bindings)?;
                }
                Ok(())
            }
        }
    }

    fn lower_match(
        &mut self,
        scrutinee: &Expr,
        arms: &[MatchArm],
        want_owned: bool,
        expected_ty: Option<&TypeExpr>,
    ) -> Result<Value, String> {
        let scrutinee_ty = self.infer_expr_type(scrutinee);

        let mut match_temps = Vec::new();
        let sv = self.lower_expr_borrowed(scrutinee, &mut match_temps)?;
        let temp_scope_idx = self.temp_rc_scopes.len();
        self.temp_rc_scopes.push(match_temps);

        let merge_b = self.builder.create_block();
        self.builder.append_block_param(merge_b, types::I64);
        let mut any_reachable_merge = false;

        for arm in arms {
            let is_unconditional_wild = arm.guard.is_none()
                && arm.patterns.len() == 1
                && matches!(arm.patterns[0], Pattern::Wildcard(_));
            if is_unconditional_wild {
                let v = if want_owned {
                    self.lower_expr_owned(&arm.body, expected_ty)?
                } else {
                    let mut atemps = Vec::new();
                    let av = self.lower_expr_borrowed(&arm.body, &mut atemps)?;
                    if !self.terminated {
                        self.flush_temps(&atemps);
                    }
                    av
                };
                if !self.terminated {
                    self.builder.ins().jump(merge_b, &[v.into()]);
                    any_reachable_merge = true;
                }
                self.terminated = false;
                self.builder.switch_to_block(merge_b);
                self.builder.seal_block(merge_b);
                let mt = self.temp_rc_scopes.pop().unwrap();
                if !any_reachable_merge {
                    self.builder.ins().trap(TrapCode::unwrap_user(2));
                    self.terminated = true;
                    return Ok(self.builder.block_params(merge_b)[0]);
                }
                self.flush_temps(&mt);
                return Ok(self.builder.block_params(merge_b)[0]);
            }

            let next_b = self.builder.create_block();
            let mut bindings: Vec<(String, Value, TypeExpr)> = Vec::new();

            if arm.patterns.len() == 1 {
                self.emit_pattern_check(sv, &scrutinee_ty, &arm.patterns[0], next_b, &mut bindings)?;
            } else {
                let matched_b = self.builder.create_block();
                for (pi, p) in arm.patterns.iter().enumerate() {
                    let fail_target = if pi + 1 < arm.patterns.len() {
                        self.builder.create_block()
                    } else {
                        next_b
                    };
                    let mut pbinds = Vec::new();
                    self.emit_pattern_check(sv, &scrutinee_ty, p, fail_target, &mut pbinds)?;
                    self.builder.ins().jump(matched_b, &[]);
                    if pi + 1 < arm.patterns.len() {
                        self.builder.switch_to_block(fail_target);
                        self.builder.seal_block(fail_target);
                    }
                }
                self.builder.switch_to_block(matched_b);
                self.builder.seal_block(matched_b);
            }

            let saved_arm = self.save_lexical();
            for (bname, bval, bty) in bindings {
                let bvar = self.declare_fresh_var(&bname, types::I64);
                self.builder.def_var(bvar, bval);
                self.owned_rc_vars.remove(&bname);
                if self.type_is_rc(&bty) {
                    self.borrowed_rc_vars.insert(bname.clone());
                } else {
                    self.borrowed_rc_vars.remove(&bname);
                }
                self.var_types.insert(bname, bty);
            }
            if let Some(guard_expr) = &arm.guard {
                let mut gtemps = Vec::new();
                let gv = self.lower_expr_borrowed(guard_expr, &mut gtemps)?;
                self.flush_temps(&gtemps);
                let gz = self.builder.ins().iconst(types::I64, 0);
                let gc = self.builder.ins().icmp(IntCC::NotEqual, gv, gz);
                let body_b = self.builder.create_block();
                self.builder.ins().brif(gc, body_b, &[], next_b, &[]);
                self.builder.switch_to_block(body_b);
                self.builder.seal_block(body_b);
            }
            let v = if want_owned {
                self.lower_expr_owned(&arm.body, expected_ty)?
            } else {
                let mut atemps = Vec::new();
                let av = self.lower_expr_borrowed(&arm.body, &mut atemps)?;
                if !self.terminated {
                    self.flush_temps(&atemps);
                }
                av
            };
            self.restore_lexical(saved_arm);
            if !self.terminated {
                self.builder.ins().jump(merge_b, &[v.into()]);
                any_reachable_merge = true;
            }
            self.terminated = false;

            self.builder.switch_to_block(next_b);
            self.builder.seal_block(next_b);
        }

        let zero = self.builder.ins().iconst(types::I64, 0);
        self.builder.ins().jump(merge_b, &[zero.into()]);
        self.builder.switch_to_block(merge_b);
        self.builder.seal_block(merge_b);
        let mt = self.temp_rc_scopes.remove(temp_scope_idx);
        self.flush_temps(&mt);
        Ok(self.builder.block_params(merge_b)[0])
    }

    fn lower_expr_fresh(
        &mut self,
        expr: &Expr,
        expected_ty: Option<&TypeExpr>,
    ) -> Result<Value, String> {
        match &expr.kind {
            ExprKind::Int(n) => Ok(self.builder.ins().iconst(types::I64, *n)),
            ExprKind::Float(x) => Ok(self.builder.ins().iconst(types::I64, x.to_bits() as i64)),
            ExprKind::Bool(b) => Ok(self
                .builder
                .ins()
                .iconst(types::I64, if *b { 1 } else { 0 })),
            ExprKind::Str(s) => {
                let boxed: Box<[u8]> = s.as_bytes().to_vec().into_boxed_slice();
                let ptr = boxed.as_ptr() as i64;
                let len = boxed.len() as i64;
                self.interned_strings.push(boxed);
                Ok(self.call_rt_2(self.rt.text_new, ptr, len))
            }
            ExprKind::RecordLit { name, fields, .. } => {
                let layout = self
                    .records
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("unknown record `{}` in codegen", name))?;
                if let Some(c_lay) = &layout.c_layout {
                    let sz_v = self.builder.ins().iconst(types::I64, c_lay.size as i64);
                    let obj = self.call_rt_1(self.rt.cstruct_alloc, sz_v);
                    let base = self.call_rt_1(self.rt.cstruct_data_ptr, obj);
                    for (fname, fexpr) in fields {
                        if let Some(cfl) = c_lay.fields.iter().find(|f| f.name == *fname) {
                            let mut temps = Vec::new();
                            let v = self.lower_expr_borrowed(fexpr, &mut temps)?;
                            self.emit_mem_store(base, cfl.byte_offset as i32, v, &cfl.ty, false)?;
                            self.flush_temps(&temps);
                        }
                    }
                    return Ok(obj);
                }
                let obj = self.call_rt_2(self.rt.rc_alloc, layout.tag, layout.fields.len() as i64);
                if layout.rc_mask != 0 {
                    let mv = self.builder.ins().iconst(types::I64, layout.rc_mask);
                    self.call_rt_2_void(self.rt.rc_set_rc_mask, obj, mv);
                }
                for (fname, fexpr) in fields {
                    if let Some((idx, fl)) = layout
                        .fields
                        .iter()
                        .enumerate()
                        .find(|(_, f)| f.name == *fname)
                    {
                        let val = if fl.is_rc {
                            self.lower_expr_owned(fexpr, Some(&fl.ty))?
                        } else {
                            let mut temps = Vec::new();
                            let v = self.lower_expr_borrowed(fexpr, &mut temps)?;
                            self.flush_temps(&temps);
                            v
                        };
                        self.call_rt_3_void(self.rt.rc_set_field, obj, idx as i64, val);
                    }
                }
                Ok(obj)
            }
            ExprKind::ArrayRepeat { value, len } => {
                let arr_ty = expected_ty
                    .cloned()
                    .filter(|t| matches!(resolve_alias_ty(t, self.alias_bases), TypeExpr::Array { .. }))
                    .unwrap_or_else(|| TypeExpr::Array {
                        elem: Box::new(self.infer_expr_type(value)),
                        len: *len,
                        span: expr.span,
                    });
                let (elem_ty, alen, stride, elem_is_rc) = self.array_layout(&arr_ty).unwrap();
                if !elem_is_rc {
                    let total_bytes = (alen * stride) as i64;
                    let sz_v = self.builder.ins().iconst(types::I64, total_bytes);
                    let obj = self.call_rt_1(self.rt.cstruct_alloc, sz_v);
                    let base = self.call_rt_1(self.rt.cstruct_data_ptr, obj);
                    let mut temps = Vec::new();
                    let val = self.lower_expr_borrowed(value, &mut temps)?;
                    if alen <= 16 {
                        for idx in 0..alen {
                            self.emit_mem_store(base, (idx * stride) as i32, val, &elem_ty, false)?;
                        }
                    } else {
                        let ivar = self.builder.declare_var(types::I64);
                        let z0 = self.builder.ins().iconst(types::I64, 0);
                        self.builder.def_var(ivar, z0);
                        let hdr = self.builder.create_block();
                        let body = self.builder.create_block();
                        let done = self.builder.create_block();
                        self.builder.ins().jump(hdr, &[]);
                        self.builder.switch_to_block(hdr);
                        let cur = self.builder.use_var(ivar);
                        let lim = self.builder.ins().iconst(types::I64, alen as i64);
                        let cond = self.builder.ins().icmp(IntCC::SignedLessThan, cur, lim);
                        self.builder.ins().brif(cond, body, &[], done, &[]);
                        self.builder.switch_to_block(body);
                        self.builder.seal_block(body);
                        let sv = self.builder.ins().iconst(types::I64, stride as i64);
                        let boff = self.builder.ins().imul(cur, sv);
                        let eptr = self.builder.ins().iadd(base, boff);
                        self.emit_mem_store(eptr, 0, val, &elem_ty, false)?;
                        let one = self.builder.ins().iconst(types::I64, 1);
                        let nxt = self.builder.ins().iadd(cur, one);
                        self.builder.def_var(ivar, nxt);
                        self.builder.ins().jump(hdr, &[]);
                        self.builder.seal_block(hdr);
                        self.builder.switch_to_block(done);
                        self.builder.seal_block(done);
                    }
                    self.flush_temps(&temps);
                    return Ok(obj);
                }
                let mut temps = Vec::new();
                let init_v = self.lower_expr_borrowed(value, &mut temps)?;
                let cnt_v = self.builder.ins().iconst(types::I64, alen as i64);
                let rc_flag = self.builder.ins().iconst(types::I64, 1);
                let fref = self.module.declare_func_in_func(self.rt.list_filled, self.builder.func);
                let call = self.builder.ins().call(fref, &[cnt_v, init_v, rc_flag]);
                let res = self.builder.inst_results(call)[0];
                self.flush_temps(&temps);
                Ok(res)
            }
            ExprKind::Cast { expr: inner, ty } | ExprKind::TryCast { expr: inner, ty } => {
                if let ExprKind::Float(x) = &inner.kind {
                    if is_f32_type_expr(ty, self.alias_bases) {
                        return Ok(self.builder.ins().iconst(types::I64, (*x as f32).to_bits() as i64));
                    }
                }
                let src_ty = self.infer_expr_type(inner);
                let mut temps = Vec::new();
                let v = self.lower_expr_borrowed(inner, &mut temps)?;
                self.flush_temps(&temps);
                Ok(self.emit_cast(v, &src_ty, ty))
            }
            ExprKind::List(elems) => {
                if let Some(exp) = expected_ty {
                    if let Some((elem_ty, alen, stride, false)) = self.array_layout(exp) {
                        let total_bytes = (alen * stride) as i64;
                        let sz_v = self.builder.ins().iconst(types::I64, total_bytes);
                        let obj = self.call_rt_1(self.rt.cstruct_alloc, sz_v);
                        let base = self.call_rt_1(self.rt.cstruct_data_ptr, obj);
                        for (idx, e) in elems.iter().enumerate() {
                            let mut temps = Vec::new();
                            let v = self.lower_expr_borrowed(e, &mut temps)?;
                            self.emit_mem_store(base, (idx * stride) as i32, v, &elem_ty, false)?;
                            self.flush_temps(&temps);
                        }
                        return Ok(obj);
                    }
                }
                let exp_elem_ty = expected_ty.and_then(|t| self.list_elem_ty(t));
                let elem_is_rc = exp_elem_ty
                    .as_ref()
                    .is_some_and(|et| self.type_is_rc(et))
                    || elems.iter().any(|e| self.expr_produces_rc(e));
                let obj = self.call_rt_2(self.rt.rc_alloc, TAG_LIST, elems.len() as i64);
                if elem_is_rc {
                    let mv = self.builder.ins().iconst(types::I64, RC_MASK_ALL as i64);
                    self.call_rt_2_void(self.rt.rc_set_rc_mask, obj, mv);
                }
                for (idx, e) in elems.iter().enumerate() {
                    let val = if elem_is_rc {
                        self.lower_expr_owned(e, exp_elem_ty.as_ref())?
                    } else {
                        let mut temps = Vec::new();
                        let v = self.lower_expr_borrowed(e, &mut temps)?;
                        self.flush_temps(&temps);
                        v
                    };
                    self.call_rt_3_void(self.rt.rc_set_field, obj, idx as i64, val);
                }
                Ok(obj)
            }
            ExprKind::Tuple(elems) => {
                let exp_elems = expected_ty.and_then(|t| match resolve_alias_ty(t, self.alias_bases) {
                    TypeExpr::Tuple(es, _) => Some(es),
                    _ => None,
                });
                let obj = self.call_rt_2(self.rt.rc_alloc, TAG_TUPLE, elems.len() as i64);
                let mut rc_mask = 0u64;
                for (idx, e) in elems.iter().enumerate() {
                    let ety = exp_elems.as_ref().and_then(|es| es.get(idx));
                    let is_rc = ety.is_some_and(|t| self.type_is_rc(t)) || self.expr_produces_rc(e);
                    let val = if is_rc {
                        if idx < 64 {
                            rc_mask |= 1u64 << idx;
                        }
                        self.lower_expr_owned(e, ety)?
                    } else {
                        let mut temps = Vec::new();
                        let v = self.lower_expr_borrowed(e, &mut temps)?;
                        self.flush_temps(&temps);
                        v
                    };
                    self.call_rt_3_void(self.rt.rc_set_field, obj, idx as i64, val);
                }
                if rc_mask != 0 {
                    let mv = self.builder.ins().iconst(types::I64, rc_mask as i64);
                    self.call_rt_2_void(self.rt.rc_set_rc_mask, obj, mv);
                }
                Ok(obj)
            }
            ExprKind::Unary { op, expr } => {
                let sk = self.scalar_kind_of_expr(expr);
                let is_flt = self.is_float_expr(expr);
                let mut temps = Vec::new();
                let v = self.lower_expr_borrowed(expr, &mut temps)?;
                self.flush_temps(&temps);
                match op {
                    UnOp::Neg => {
                        if matches!(sk, Some(ScalarKind::F16 | ScalarKind::BF16)) {
                            let mask = self.builder.ins().iconst(types::I64, 0x8000);
                            Ok(self.builder.ins().bxor(v, mask))
                        } else if sk == Some(ScalarKind::F32) {
                            let i32v = self.builder.ins().ireduce(types::I32, v);
                            let fv = self.builder.ins().bitcast(types::F32, MemFlags::new(), i32v);
                            let neg = self.builder.ins().fneg(fv);
                            let o32 = self.builder.ins().bitcast(types::I32, MemFlags::new(), neg);
                            Ok(self.builder.ins().uextend(types::I64, o32))
                        } else if is_flt {
                            let fv = self.builder.ins().bitcast(types::F64, MemFlags::new(), v);
                            let neg = self.builder.ins().fneg(fv);
                            Ok(self.builder.ins().bitcast(types::I64, MemFlags::new(), neg))
                        } else {
                            let neg = self.builder.ins().ineg(v);
                            Ok(if let Some(k) = sk { self.normalize_scalar(neg, k) } else { neg })
                        }
                    }
                    UnOp::BitNot => {
                        let inv = self.builder.ins().bnot(v);
                        Ok(if let Some(k) = sk { self.normalize_scalar(inv, k) } else { inv })
                    }
                    UnOp::Not => {
                        let zero = self.builder.ins().iconst(types::I64, 0);
                        let eq = self.builder.ins().icmp(IntCC::Equal, v, zero);
                        Ok(self.builder.ins().uextend(types::I64, eq))
                    }
                }
            }
            ExprKind::Binary {
                op: BinOp::And,
                left,
                right,
            } => {
                let mut ltemps = Vec::new();
                let l = self.lower_expr_borrowed(left, &mut ltemps)?;
                self.flush_temps(&ltemps);
                let z = self.builder.ins().iconst(types::I64, 0);
                let lb = self.builder.ins().icmp(IntCC::NotEqual, l, z);
                let rhs_b = self.builder.create_block();
                let short_b = self.builder.create_block();
                let merge_b = self.builder.create_block();
                self.builder.append_block_param(merge_b, types::I64);
                self.builder.ins().brif(lb, rhs_b, &[], short_b, &[]);

                self.builder.switch_to_block(short_b);
                self.builder.seal_block(short_b);
                let fval = self.builder.ins().iconst(types::I64, 0);
                self.builder.ins().jump(merge_b, &[fval.into()]);

                self.builder.switch_to_block(rhs_b);
                self.builder.seal_block(rhs_b);
                let mut rtemps = Vec::new();
                let r = self.lower_expr_borrowed(right, &mut rtemps)?;
                self.flush_temps(&rtemps);
                let rz = self.builder.ins().iconst(types::I64, 0);
                let rb = self.builder.ins().icmp(IntCC::NotEqual, r, rz);
                let rval = self.builder.ins().uextend(types::I64, rb);
                self.builder.ins().jump(merge_b, &[rval.into()]);

                self.builder.switch_to_block(merge_b);
                self.builder.seal_block(merge_b);
                Ok(self.builder.block_params(merge_b)[0])
            }
            ExprKind::Binary {
                op: BinOp::Or,
                left,
                right,
            } => {
                let mut ltemps = Vec::new();
                let l = self.lower_expr_borrowed(left, &mut ltemps)?;
                self.flush_temps(&ltemps);
                let z = self.builder.ins().iconst(types::I64, 0);
                let lb = self.builder.ins().icmp(IntCC::NotEqual, l, z);
                let short_b = self.builder.create_block();
                let rhs_b = self.builder.create_block();
                let merge_b = self.builder.create_block();
                self.builder.append_block_param(merge_b, types::I64);
                self.builder.ins().brif(lb, short_b, &[], rhs_b, &[]);

                self.builder.switch_to_block(short_b);
                self.builder.seal_block(short_b);
                let tval = self.builder.ins().iconst(types::I64, 1);
                self.builder.ins().jump(merge_b, &[tval.into()]);

                self.builder.switch_to_block(rhs_b);
                self.builder.seal_block(rhs_b);
                let mut rtemps = Vec::new();
                let r = self.lower_expr_borrowed(right, &mut rtemps)?;
                self.flush_temps(&rtemps);
                let rz = self.builder.ins().iconst(types::I64, 0);
                let rb = self.builder.ins().icmp(IntCC::NotEqual, r, rz);
                let rval = self.builder.ins().uextend(types::I64, rb);
                self.builder.ins().jump(merge_b, &[rval.into()]);

                self.builder.switch_to_block(merge_b);
                self.builder.seal_block(merge_b);
                Ok(self.builder.block_params(merge_b)[0])
            }
            ExprKind::Binary { op, left, right } => {
                let text_cmp = self.is_text_expr(left) || self.is_text_expr(right);
                let lk = self.scalar_kind_of_expr(left);
                let rk = self.scalar_kind_of_expr(right);
                let eff_sk = match (lk, rk) {
                    (Some(ScalarKind::I64), Some(k)) if k != ScalarKind::I64 && matches!(left.kind, ExprKind::Int(_)) => Some(k),
                    (Some(ScalarKind::F32), _) | (_, Some(ScalarKind::F32)) => Some(ScalarKind::F32),
                    (Some(ScalarKind::F16), _) | (_, Some(ScalarKind::F16)) => Some(ScalarKind::F16),
                    (Some(ScalarKind::BF16), _) | (_, Some(ScalarKind::BF16)) => Some(ScalarKind::BF16),
                    (Some(k), _) => Some(k),
                    (None, k) => k,
                };
                let float_op = self.is_float_expr(left) || self.is_float_expr(right);
                let mut temps = Vec::new();
                let l = self.lower_expr_borrowed(left, &mut temps)?;
                let r = self.lower_expr_borrowed(right, &mut temps)?;
                if *op == BinOp::Concat {
                    let res = self.call_rt_2_vals(self.rt.str_concat, l, r);
                    self.flush_temps(&temps);
                    return Ok(res);
                }
                if text_cmp && matches!(op, BinOp::Eq | BinOp::Ne) {
                    let eq = self.call_rt_2_vals(self.rt.text_eq, l, r);
                    self.flush_temps(&temps);
                    if *op == BinOp::Eq {
                        return Ok(eq);
                    }
                    let zero = self.builder.ins().iconst(types::I64, 0);
                    let ne = self.builder.ins().icmp(IntCC::Equal, eq, zero);
                    return Ok(self.builder.ins().uextend(types::I64, ne));
                }
                self.flush_temps(&temps);
                if matches!(eff_sk, Some(ScalarKind::F16 | ScalarKind::BF16)) {
                    let (to_f64, from_f64) = if eff_sk == Some(ScalarKind::F16) {
                        (self.rt.f16_to_f64, self.rt.f64_to_f16)
                    } else {
                        (self.rt.bf16_to_f64, self.rt.f64_to_bf16)
                    };
                    let l64 = self.call_rt_1(to_f64, l);
                    let r64 = self.call_rt_1(to_f64, r);
                    let lf = self.builder.ins().bitcast(types::F64, MemFlags::new(), l64);
                    let rf = self.builder.ins().bitcast(types::F64, MemFlags::new(), r64);
                    let v = match op {
                        BinOp::Add => {
                            let fv = self.builder.ins().fadd(lf, rf);
                            let b64 = self.builder.ins().bitcast(types::I64, MemFlags::new(), fv);
                            self.call_rt_1(from_f64, b64)
                        }
                        BinOp::Sub => {
                            let fv = self.builder.ins().fsub(lf, rf);
                            let b64 = self.builder.ins().bitcast(types::I64, MemFlags::new(), fv);
                            self.call_rt_1(from_f64, b64)
                        }
                        BinOp::Mul => {
                            let fv = self.builder.ins().fmul(lf, rf);
                            let b64 = self.builder.ins().bitcast(types::I64, MemFlags::new(), fv);
                            self.call_rt_1(from_f64, b64)
                        }
                        BinOp::Div => {
                            let fv = self.builder.ins().fdiv(lf, rf);
                            let b64 = self.builder.ins().bitcast(types::I64, MemFlags::new(), fv);
                            self.call_rt_1(from_f64, b64)
                        }
                        BinOp::Eq => {
                            let c = self.builder.ins().fcmp(FloatCC::Equal, lf, rf);
                            self.builder.ins().uextend(types::I64, c)
                        }
                        BinOp::Ne => {
                            let c = self.builder.ins().fcmp(FloatCC::NotEqual, lf, rf);
                            self.builder.ins().uextend(types::I64, c)
                        }
                        BinOp::Lt => {
                            let c = self.builder.ins().fcmp(FloatCC::LessThan, lf, rf);
                            self.builder.ins().uextend(types::I64, c)
                        }
                        BinOp::Le => {
                            let c = self.builder.ins().fcmp(FloatCC::LessThanOrEqual, lf, rf);
                            self.builder.ins().uextend(types::I64, c)
                        }
                        BinOp::Gt => {
                            let c = self.builder.ins().fcmp(FloatCC::GreaterThan, lf, rf);
                            self.builder.ins().uextend(types::I64, c)
                        }
                        BinOp::Ge => {
                            let c = self.builder.ins().fcmp(FloatCC::GreaterThanOrEqual, lf, rf);
                            self.builder.ins().uextend(types::I64, c)
                        }
                        other => return Err(format!("unsupported float16/bfloat16 op {:?} in codegen", other)),
                    };
                    return Ok(v);
                }
                if eff_sk == Some(ScalarKind::F32) {
                    let l32 = self.builder.ins().ireduce(types::I32, l);
                    let r32 = self.builder.ins().ireduce(types::I32, r);
                    let lf = self.builder.ins().bitcast(types::F32, MemFlags::new(), l32);
                    let rf = self.builder.ins().bitcast(types::F32, MemFlags::new(), r32);
                    let v = match op {
                        BinOp::Add => {
                            let fv = self.builder.ins().fadd(lf, rf);
                            let i32v = self.builder.ins().bitcast(types::I32, MemFlags::new(), fv);
                            self.builder.ins().uextend(types::I64, i32v)
                        }
                        BinOp::Sub => {
                            let fv = self.builder.ins().fsub(lf, rf);
                            let i32v = self.builder.ins().bitcast(types::I32, MemFlags::new(), fv);
                            self.builder.ins().uextend(types::I64, i32v)
                        }
                        BinOp::Mul => {
                            let fv = self.builder.ins().fmul(lf, rf);
                            let i32v = self.builder.ins().bitcast(types::I32, MemFlags::new(), fv);
                            self.builder.ins().uextend(types::I64, i32v)
                        }
                        BinOp::Div => {
                            let fv = self.builder.ins().fdiv(lf, rf);
                            let i32v = self.builder.ins().bitcast(types::I32, MemFlags::new(), fv);
                            self.builder.ins().uextend(types::I64, i32v)
                        }
                        BinOp::Eq => {
                            let c = self.builder.ins().fcmp(FloatCC::Equal, lf, rf);
                            self.builder.ins().uextend(types::I64, c)
                        }
                        BinOp::Ne => {
                            let c = self.builder.ins().fcmp(FloatCC::NotEqual, lf, rf);
                            self.builder.ins().uextend(types::I64, c)
                        }
                        BinOp::Lt => {
                            let c = self.builder.ins().fcmp(FloatCC::LessThan, lf, rf);
                            self.builder.ins().uextend(types::I64, c)
                        }
                        BinOp::Le => {
                            let c = self.builder.ins().fcmp(FloatCC::LessThanOrEqual, lf, rf);
                            self.builder.ins().uextend(types::I64, c)
                        }
                        BinOp::Gt => {
                            let c = self.builder.ins().fcmp(FloatCC::GreaterThan, lf, rf);
                            self.builder.ins().uextend(types::I64, c)
                        }
                        BinOp::Ge => {
                            let c = self.builder.ins().fcmp(FloatCC::GreaterThanOrEqual, lf, rf);
                            self.builder.ins().uextend(types::I64, c)
                        }
                        other => return Err(format!("unsupported float32 op {:?} in codegen", other)),
                    };
                    return Ok(v);
                }
                if float_op {
                    let lf = self.builder.ins().bitcast(types::F64, MemFlags::new(), l);
                    let rf = self.builder.ins().bitcast(types::F64, MemFlags::new(), r);
                    let v = match op {
                        BinOp::Add => {
                            let fv = self.builder.ins().fadd(lf, rf);
                            self.builder.ins().bitcast(types::I64, MemFlags::new(), fv)
                        }
                        BinOp::Sub => {
                            let fv = self.builder.ins().fsub(lf, rf);
                            self.builder.ins().bitcast(types::I64, MemFlags::new(), fv)
                        }
                        BinOp::Mul => {
                            let fv = self.builder.ins().fmul(lf, rf);
                            self.builder.ins().bitcast(types::I64, MemFlags::new(), fv)
                        }
                        BinOp::Div => {
                            let fv = self.builder.ins().fdiv(lf, rf);
                            self.builder.ins().bitcast(types::I64, MemFlags::new(), fv)
                        }
                        BinOp::Eq => {
                            let c = self.builder.ins().fcmp(FloatCC::Equal, lf, rf);
                            self.builder.ins().uextend(types::I64, c)
                        }
                        BinOp::Ne => {
                            let c = self.builder.ins().fcmp(FloatCC::NotEqual, lf, rf);
                            self.builder.ins().uextend(types::I64, c)
                        }
                        BinOp::Lt => {
                            let c = self.builder.ins().fcmp(FloatCC::LessThan, lf, rf);
                            self.builder.ins().uextend(types::I64, c)
                        }
                        BinOp::Le => {
                            let c = self.builder.ins().fcmp(FloatCC::LessThanOrEqual, lf, rf);
                            self.builder.ins().uextend(types::I64, c)
                        }
                        BinOp::Gt => {
                            let c = self.builder.ins().fcmp(FloatCC::GreaterThan, lf, rf);
                            self.builder.ins().uextend(types::I64, c)
                        }
                        BinOp::Ge => {
                            let c = self.builder.ins().fcmp(FloatCC::GreaterThanOrEqual, lf, rf);
                            self.builder.ins().uextend(types::I64, c)
                        }
                        other => return Err(format!("unsupported float op {:?} in codegen", other)),
                    };
                    return Ok(v);
                }
                let is_unsigned = eff_sk.is_some_and(|k| k.is_unsigned());
                let is_signed_narrow = matches!(eff_sk, Some(ScalarKind::I8 | ScalarKind::I16 | ScalarKind::I32));
                let is_cmp = matches!(
                    op,
                    BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
                );
                let v = match op {
                    BinOp::Add => self.builder.ins().iadd(l, r),
                    BinOp::Sub => self.builder.ins().isub(l, r),
                    BinOp::Mul => self.builder.ins().imul(l, r),
                    BinOp::Div => {
                        if is_unsigned {
                            self.builder.ins().udiv(l, r)
                        } else {
                            self.builder.ins().sdiv(l, r)
                        }
                    }
                    BinOp::Mod => {
                        if is_unsigned {
                            self.builder.ins().urem(l, r)
                        } else {
                            self.builder.ins().srem(l, r)
                        }
                    }
                    BinOp::BitAnd => self.builder.ins().band(l, r),
                    BinOp::BitOr => self.builder.ins().bor(l, r),
                    BinOp::BitXor => self.builder.ins().bxor(l, r),
                    BinOp::Shl => self.builder.ins().ishl(l, r),
                    BinOp::Shr => {
                        if is_signed_narrow {
                            self.builder.ins().sshr(l, r)
                        } else {
                            self.builder.ins().ushr(l, r)
                        }
                    }
                    BinOp::Eq => {
                        let c = self.builder.ins().icmp(IntCC::Equal, l, r);
                        self.builder.ins().uextend(types::I64, c)
                    }
                    BinOp::Ne => {
                        let c = self.builder.ins().icmp(IntCC::NotEqual, l, r);
                        self.builder.ins().uextend(types::I64, c)
                    }
                    BinOp::Lt => {
                        let cc = if is_unsigned { IntCC::UnsignedLessThan } else { IntCC::SignedLessThan };
                        let c = self.builder.ins().icmp(cc, l, r);
                        self.builder.ins().uextend(types::I64, c)
                    }
                    BinOp::Le => {
                        let cc = if is_unsigned { IntCC::UnsignedLessThanOrEqual } else { IntCC::SignedLessThanOrEqual };
                        let c = self.builder.ins().icmp(cc, l, r);
                        self.builder.ins().uextend(types::I64, c)
                    }
                    BinOp::Gt => {
                        let cc = if is_unsigned { IntCC::UnsignedGreaterThan } else { IntCC::SignedGreaterThan };
                        let c = self.builder.ins().icmp(cc, l, r);
                        self.builder.ins().uextend(types::I64, c)
                    }
                    BinOp::Ge => {
                        let cc = if is_unsigned { IntCC::UnsignedGreaterThanOrEqual } else { IntCC::SignedGreaterThanOrEqual };
                        let c = self.builder.ins().icmp(cc, l, r);
                        self.builder.ins().uextend(types::I64, c)
                    }
                    other => return Err(format!("unsupported binary op {:?} in codegen", other)),
                };
                if !is_cmp {
                    if let Some(k) = eff_sk {
                        return Ok(self.normalize_scalar(v, k));
                    }
                }
                Ok(v)
            }
            ExprKind::Call { callee, args } => match &callee.kind {
                ExprKind::Field { target, field } if field == "parse" => {
                    if let ExprKind::Ident(rname) = &target.kind {
                        let pname = format!("{}__parse", rname);
                        let fid = *self.func_ids.get(&pname).ok_or_else(|| {
                            format!("no derived parser `{}.parse` found", rname)
                        })?;
                        let mut temps = Vec::new();
                        let mut arg_vals = Vec::new();
                        for a in args {
                            arg_vals.push(self.lower_expr_borrowed(&a.value, &mut temps)?);
                        }
                        let local_callee = self.module.declare_func_in_func(fid, self.builder.func);
                        let call = self.builder.ins().call(local_callee, &arg_vals);
                        let ret = self.builder.inst_results(call)[0];
                        self.flush_temps(&temps);
                        return Ok(ret);
                    }
                    Err("unsupported `.parse()` target".into())
                }
                ExprKind::Field { target, field } => {
                    if let ExprKind::Ident(ename) = &target.kind {
                        let key = format!("{}.{}", ename, field);
                        if let Some(&tag) = self.enum_tags.get(&key) {
                            let ptys = self
                                .enum_variants
                                .get(&key)
                                .map(|vl| vl.payload_tys.clone())
                                .unwrap_or_default();
                            return self.lower_enum_ctor(tag, &ptys, args);
                        }
                    }
                    // Dynamic interface method dispatch: `d.method(args)` where `d: dyn Interface`!
                    let recv_ty = resolve_alias_ty(&self.infer_expr_type(target), self.alias_bases);
                    if let TypeExpr::Dyn { interface, .. } = recv_ty {
                        let dm = self
                            .dyn_interfaces
                            .get(&interface)
                            .and_then(|l| l.methods.iter().find(|m| m.name == *field))
                            .cloned()
                            .ok_or_else(|| {
                                format!("unknown method `{}` on `dyn {}` in Cranelift codegen", field, interface)
                            })?;
                        let mut temps = Vec::new();
                        let mut d_val = self.lower_expr_borrowed(target, &mut temps)?;
                        let mut call_args = Vec::new();
                        let mut inout_dyn_slot: Option<(Value, cranelift_codegen::ir::StackSlot)> = None;

                        if dm.is_inout_self {
                            let d_uniq = self.call_rt_1(self.rt.rc_ensure_unique, d_val);
                            if let ExprKind::Ident(vname) = &target.kind {
                                if let Some(&var) = self.vars.get(vname) {
                                    if self.inout_vars.contains(vname) {
                                        let slot_ptr = self.builder.use_var(var);
                                        self.builder.ins().store(MemFlags::new(), d_uniq, slot_ptr, 0);
                                    } else {
                                        self.builder.def_var(var, d_uniq);
                                    }
                                }
                            }
                            d_val = d_uniq;
                            let z0 = self.builder.ins().iconst(types::I64, 0);
                            let data_val = self.call_rt_2_vals(self.rt.rc_get_field, d_val, z0);
                            let ss = self.builder.create_sized_stack_slot(StackSlotData::new(
                                StackSlotKind::ExplicitSlot,
                                8,
                                3,
                            ));
                            self.builder.ins().stack_store(data_val, ss, 0);
                            let addr = self.builder.ins().stack_addr(types::I64, ss, 0);
                            call_args.push(addr);
                            inout_dyn_slot = Some((d_uniq, ss));
                        } else {
                            let z0 = self.builder.ins().iconst(types::I64, 0);
                            let data_val = self.call_rt_2_vals(self.rt.rc_get_field, d_val, z0);
                            call_args.push(data_val);
                        }

                        for a in args {
                            call_args.push(self.lower_expr_borrowed(&a.value, &mut temps)?);
                        }

                        let one = self.builder.ins().iconst(types::I64, 1);
                        let vt_ptr = self.call_rt_2_vals(self.rt.rc_get_field, d_val, one);
                        let fn_ptr = self.builder.ins().load(
                            types::I64,
                            MemFlags::new(),
                            vt_ptr,
                            (dm.slot_idx * 8) as i32,
                        );
                        let mut sig = self.module.make_signature();
                        for _ in 0..call_args.len() {
                            sig.params.push(AbiParam::new(types::I64));
                        }
                        sig.returns.push(AbiParam::new(types::I64));
                        let sig_ref = self.builder.import_signature(sig);
                        let call = self.builder.ins().call_indirect(sig_ref, fn_ptr, &call_args);
                        let ret = self.builder.inst_results(call)[0];

                        if let Some((d_uniq, ss)) = inout_dyn_slot {
                            let updated = self.builder.ins().stack_load(types::I64, ss, 0);
                            self.call_rt_3_void(self.rt.rc_set_field_raw, d_uniq, 0, updated);
                        }
                        self.flush_temps(&temps);
                        return Ok(ret);
                    }
                    Err(format!(
                        "unsupported method call `{}` on `{}` in codegen",
                        field,
                        recv_ty.display_name()
                    ))
                }
                ExprKind::Ident(fname) => {
                    if fname == "__make_closure" && args.len() >= 2 {
                        let csym = match &args[0].value.kind {
                            ExprKind::Str(s) => s.clone(),
                            _ => return Err("__make_closure expects closure fn symbol".into()),
                        };
                        let rc_mask = match &args[1].value.kind {
                            ExprKind::Int(m) => *m,
                            _ => 0,
                        };
                        let captures = &args[2..];
                        let obj = self.call_rt_2(self.rt.rc_alloc, TAG_CLOSURE, (1 + captures.len()) as i64);
                        if rc_mask != 0 {
                            let mv = self.builder.ins().iconst(types::I64, rc_mask);
                            self.call_rt_2_void(self.rt.rc_set_rc_mask, obj, mv);
                        }
                        let fid = *self
                            .func_ids
                            .get(&csym)
                            .ok_or_else(|| format!("missing closure fn `{}`", csym))?;
                        let fref = self.module.declare_func_in_func(fid, self.builder.func);
                        let fptr = self.builder.ins().func_addr(types::I64, fref);
                        self.call_rt_3_void(self.rt.rc_set_field, obj, 0, fptr);
                        for (k, cap) in captures.iter().enumerate() {
                            let bit = 1i64.checked_shl((k + 1) as u32).unwrap_or(0);
                            let is_rc = (rc_mask & bit) != 0;
                            let cv = if is_rc {
                                self.lower_expr_owned(&cap.value, None)?
                            } else {
                                let mut temps = Vec::new();
                                let v = self.lower_expr_borrowed(&cap.value, &mut temps)?;
                                self.flush_temps(&temps);
                                v
                            };
                            self.call_rt_3_void(self.rt.rc_set_field, obj, (k + 1) as i64, cv);
                        }
                        return Ok(obj);
                    }
                    if fname == "__closure_get" && args.len() == 2 {
                        let mut temps = Vec::new();
                        let env = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                        let idx = self.lower_expr_borrowed(&args[1].value, &mut temps)?;
                        let res = self.call_rt_2_vals(self.rt.rc_get_field, env, idx);
                        self.flush_temps(&temps);
                        return Ok(res);
                    }
                    if fname == "__call_closure" && !args.is_empty() {
                        let mut temps = Vec::new();
                        let cl_val = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                        let mut call_args = Vec::with_capacity(args.len());
                        call_args.push(cl_val);
                        for a in &args[1..] {
                            call_args.push(self.lower_expr_borrowed(&a.value, &mut temps)?);
                        }
                        let z0 = self.builder.ins().iconst(types::I64, 0);
                        let fn_ptr = self.call_rt_2_vals(self.rt.rc_get_field, cl_val, z0);
                        let mut sig = self.module.make_signature();
                        for _ in 0..call_args.len() {
                            sig.params.push(AbiParam::new(types::I64));
                        }
                        sig.returns.push(AbiParam::new(types::I64));
                        let sig_ref = self.builder.import_signature(sig);
                        let call = self.builder.ins().call_indirect(sig_ref, fn_ptr, &call_args);
                        let ret = self.builder.inst_results(call)[0];
                        self.flush_temps(&temps);
                        return Ok(ret);
                    }
                    if fname == "__box_dyn" && args.len() == 3 {
                        let vsym = match &args[1].value.kind {
                            ExprKind::Str(s) => s.clone(),
                            _ => return Err("__box_dyn expects vtable symbol string".into()),
                        };
                        let data_is_rc = matches!(&args[2].value.kind, ExprKind::Int(n) if *n != 0);
                        let data_val = if data_is_rc {
                            self.lower_expr_owned(&args[0].value, None)?
                        } else {
                            let mut temps = Vec::new();
                            let v = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                            self.flush_temps(&temps);
                            v
                        };
                        let obj = self.call_rt_2(self.rt.rc_alloc, TAG_DYN, 2);
                        if data_is_rc {
                            let mv = self.builder.ins().iconst(types::I64, 1);
                            self.call_rt_2_void(self.rt.rc_set_rc_mask, obj, mv);
                        }
                        self.call_rt_3_void(self.rt.rc_set_field, obj, 0, data_val);
                        let did = *self
                            .vtable_data_ids
                            .get(&vsym)
                            .ok_or_else(|| format!("missing vtable data `{}`", vsym))?;
                        let gv = self.module.declare_data_in_func(did, self.builder.func);
                        let vt_ptr = self.builder.ins().global_value(types::I64, gv);
                        self.call_rt_3_void(self.rt.rc_set_field, obj, 1, vt_ptr);
                        return Ok(obj);
                    }
                    if fname == "len" && args.len() == 1 {
                        let arg_ty = self.infer_expr_type(&args[0].value);
                        if let Some((_, alen, _, _)) = self.array_layout(&arg_ty) {
                            let mut temps = Vec::new();
                            let _ = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                            self.flush_temps(&temps);
                            return Ok(self.builder.ins().iconst(types::I64, alen as i64));
                        }
                        if matches!(resolve_alias_ty(&arg_ty, self.alias_bases), TypeExpr::Named { ref name, .. } if name == "BytesSlice") {
                            let fid = self.func_ids["slice_len"];
                            let mut temps = Vec::new();
                            let v = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                            let res = self.call_rt_1(fid, v);
                            self.flush_temps(&temps);
                            return Ok(res);
                        }
                        let mut temps = Vec::new();
                        let v = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                        let res = self.call_rt_1(self.rt.rc_get_len, v);
                        self.flush_temps(&temps);
                        return Ok(res);
                    }
                    if is_consuming_cow_builtin(fname) && !args.is_empty() {
                        let fid = *self
                            .func_ids
                            .get(fname)
                            .ok_or_else(|| format!("unknown cow builtin `{}`", fname))?;
                        let mut temps = Vec::new();
                        let mut call_args = Vec::with_capacity(args.len());
                        call_args.push(self.lower_expr_owned(&args[0].value, expected_ty)?);
                        for a in &args[1..] {
                            call_args.push(self.lower_expr_borrowed(&a.value, &mut temps)?);
                        }
                        let fref = self.module.declare_func_in_func(fid, self.builder.func);
                        let call = self.builder.ins().call(fref, &call_args);
                        let ret = self.builder.inst_results(call)[0];
                        self.flush_temps(&temps);
                        return Ok(ret);
                    }
                    if matches!(
                        fname.as_str(),
                        "wrapping_add" | "wrapping_sub" | "wrapping_mul" | "saturating_add" | "saturating_sub"
                    ) && args.len() == 2
                    {
                        let sk = self
                            .scalar_kind_of_expr(&args[0].value)
                            .or_else(|| self.scalar_kind_of_expr(&args[1].value))
                            .unwrap_or(ScalarKind::I64);
                        let mut temps = Vec::new();
                        let a = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                        let b = self.lower_expr_borrowed(&args[1].value, &mut temps)?;
                        self.flush_temps(&temps);
                        let out = match fname.as_str() {
                            "wrapping_add" => {
                                let r = self.builder.ins().iadd(a, b);
                                self.normalize_scalar(r, sk)
                            }
                            "wrapping_sub" => {
                                let r = self.builder.ins().isub(a, b);
                                self.normalize_scalar(r, sk)
                            }
                            "wrapping_mul" => {
                                let r = self.builder.ins().imul(a, b);
                                self.normalize_scalar(r, sk)
                            }
                            "saturating_add" => {
                                if sk.is_unsigned() {
                                    let max_u = self.builder.ins().iconst(types::I64, sk.max_i128().unwrap_or(u64::MAX as i128) as u64 as i64);
                                    let sum = self.builder.ins().iadd(a, b);
                                    if matches!(sk, ScalarKind::U64) {
                                        let ovf = self.builder.ins().icmp(IntCC::UnsignedLessThan, sum, a);
                                        self.builder.ins().select(ovf, max_u, sum)
                                    } else {
                                        let ovf = self.builder.ins().icmp(IntCC::UnsignedGreaterThan, sum, max_u);
                                        self.builder.ins().select(ovf, max_u, sum)
                                    }
                                } else {
                                    let lo = sk.min_i128().unwrap_or(i64::MIN as i128) as i64;
                                    let hi = sk.max_i128().unwrap_or(i64::MAX as i128) as i64;
                                    let min_v = self.builder.ins().iconst(types::I64, lo);
                                    let max_v = self.builder.ins().iconst(types::I64, hi);
                                    let sum = self.builder.ins().iadd(a, b);
                                    if matches!(sk, ScalarKind::I64) {
                                        let z = self.builder.ins().iconst(types::I64, 0);
                                        let b_pos = self.builder.ins().icmp(IntCC::SignedGreaterThan, b, z);
                                        let b_neg = self.builder.ins().icmp(IntCC::SignedLessThan, b, z);
                                        let pos_ovf = self.builder.ins().icmp(IntCC::SignedLessThan, sum, a);
                                        let neg_ovf = self.builder.ins().icmp(IntCC::SignedGreaterThan, sum, a);
                                        let is_hi = self.builder.ins().band(b_pos, pos_ovf);
                                        let is_lo = self.builder.ins().band(b_neg, neg_ovf);
                                        let s1 = self.builder.ins().select(is_hi, max_v, sum);
                                        self.builder.ins().select(is_lo, min_v, s1)
                                    } else {
                                        let too_hi = self.builder.ins().icmp(IntCC::SignedGreaterThan, sum, max_v);
                                        let too_lo = self.builder.ins().icmp(IntCC::SignedLessThan, sum, min_v);
                                        let s1 = self.builder.ins().select(too_hi, max_v, sum);
                                        self.builder.ins().select(too_lo, min_v, s1)
                                    }
                                }
                            }
                            _ => {
                                // saturating_sub
                                if sk.is_unsigned() {
                                    let z = self.builder.ins().iconst(types::I64, 0);
                                    let under = self.builder.ins().icmp(IntCC::UnsignedLessThan, a, b);
                                    let diff = self.builder.ins().isub(a, b);
                                    self.builder.ins().select(under, z, diff)
                                } else {
                                    let lo = sk.min_i128().unwrap_or(i64::MIN as i128) as i64;
                                    let hi = sk.max_i128().unwrap_or(i64::MAX as i128) as i64;
                                    let min_v = self.builder.ins().iconst(types::I64, lo);
                                    let max_v = self.builder.ins().iconst(types::I64, hi);
                                    let diff = self.builder.ins().isub(a, b);
                                    if matches!(sk, ScalarKind::I64) {
                                        let z = self.builder.ins().iconst(types::I64, 0);
                                        let b_neg = self.builder.ins().icmp(IntCC::SignedLessThan, b, z);
                                        let b_pos = self.builder.ins().icmp(IntCC::SignedGreaterThan, b, z);
                                        let pos_ovf = self.builder.ins().icmp(IntCC::SignedLessThan, diff, a);
                                        let neg_ovf = self.builder.ins().icmp(IntCC::SignedGreaterThan, diff, a);
                                        let is_hi = self.builder.ins().band(b_neg, pos_ovf);
                                        let is_lo = self.builder.ins().band(b_pos, neg_ovf);
                                        let s1 = self.builder.ins().select(is_hi, max_v, diff);
                                        self.builder.ins().select(is_lo, min_v, s1)
                                    } else {
                                        let too_hi = self.builder.ins().icmp(IntCC::SignedGreaterThan, diff, max_v);
                                        let too_lo = self.builder.ins().icmp(IntCC::SignedLessThan, diff, min_v);
                                        let s1 = self.builder.ins().select(too_hi, max_v, diff);
                                        self.builder.ins().select(too_lo, min_v, s1)
                                    }
                                }
                            }
                        };
                        return Ok(out);
                    }
                    if fname == "ptr_is_null" && args.len() == 1 {
                        let mut temps = Vec::new();
                        let p = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                        self.flush_temps(&temps);
                        let z = self.builder.ins().iconst(types::I64, 0);
                        let eq = self.builder.ins().icmp(IntCC::Equal, p, z);
                        return Ok(self.builder.ins().uextend(types::I64, eq));
                    }
                    if (fname == "ptr_of" || fname == "mut_ptr_of") && args.len() == 1 {
                        let arg = &args[0].value;
                        let aty = self.infer_expr_type(arg);
                        let abase = resolve_alias_ty(&aty, self.alias_bases);
                        let is_buf = self.is_c_value_type(&abase)
                            || matches!(&abase, TypeExpr::Named { name, .. } if name == "Bytes" || name == "Text");
                        if is_buf {
                            if fname == "mut_ptr_of" {
                                if let ExprKind::Ident(vname) = &arg.kind {
                                    let cur = self.load_ident_val(vname)?;
                                    let uniq = self.call_rt_1(self.rt.rc_ensure_unique, cur);
                                    let var = self.vars[vname];
                                    if self.inout_vars.contains(vname) {
                                        let slot_ptr = self.builder.use_var(var);
                                        self.builder.ins().store(MemFlags::new(), uniq, slot_ptr, 0);
                                    } else {
                                        self.builder.def_var(var, uniq);
                                    }
                                    return Ok(self.call_rt_1(self.rt.cstruct_data_ptr, uniq));
                                }
                            }
                            let mut temps = Vec::new();
                            let v = self.lower_expr_borrowed(arg, &mut temps)?;
                            let p = self.call_rt_1(self.rt.cstruct_data_ptr, v);
                            self.flush_temps(&temps);
                            return Ok(p);
                        }
                        if let ExprKind::Ident(vname) = &arg.kind {
                            if self.inout_vars.contains(vname) {
                                let var = self.vars[vname];
                                return Ok(self.builder.use_var(var));
                            }
                        }
                        let mut temps = Vec::new();
                        let v = self.lower_expr_borrowed(arg, &mut temps)?;
                        let ss = self.builder.create_sized_stack_slot(StackSlotData::new(
                            StackSlotKind::ExplicitSlot,
                            8,
                            3,
                        ));
                        let addr = self.builder.ins().stack_addr(types::I64, ss, 0);
                        self.emit_mem_store(addr, 0, v, &abase, false)?;
                        self.flush_temps(&temps);
                        return Ok(addr);
                    }
                    if fname == "__ptr_add" && args.len() == 3 {
                        let mut temps = Vec::new();
                        let p = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                        let n = self.lower_expr_borrowed(&args[1].value, &mut temps)?;
                        let sz = self.lower_expr_borrowed(&args[2].value, &mut temps)?;
                        self.flush_temps(&temps);
                        let off = self.builder.ins().imul(n, sz);
                        return Ok(self.builder.ins().iadd(p, off));
                    }
                    if (fname == "ptr_read" || fname == "volatile_read") && args.len() == 1 {
                        let pty = self.infer_expr_type(&args[0].value);
                        let elem_ty = self
                            .ptr_pointee_ty(&pty)
                            .unwrap_or_else(|| named_ty("Int64"));
                        let mut temps = Vec::new();
                        let p = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                        let v = self.emit_mem_load(p, 0, &elem_ty, fname == "volatile_read")?;
                        self.flush_temps(&temps);
                        return Ok(v);
                    }
                    if (fname == "ptr_write" || fname == "volatile_write") && args.len() == 2 {
                        let pty = self.infer_expr_type(&args[0].value);
                        let elem_ty = self
                            .ptr_pointee_ty(&pty)
                            .unwrap_or_else(|| self.infer_expr_type(&args[1].value));
                        let mut temps = Vec::new();
                        let p = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                        let v = self.lower_expr_borrowed(&args[1].value, &mut temps)?;
                        self.emit_mem_store(p, 0, v, &elem_ty, fname == "volatile_write")?;
                        self.flush_temps(&temps);
                        return Ok(self.builder.ins().iconst(types::I64, 0));
                    }
                    if let Some(ef) = self.extern_fns.get(fname).cloned() {
                        let fid = *self.extern_func_ids.get(fname).ok_or_else(|| {
                            format!(
                                "unresolved extern \"C\" symbol `{}` in JIT (use `tenon build --c-src ...` or `-l ...` to link external C code)",
                                fname
                            )
                        })?;
                        return self.emit_extern_c_call(&ef, fid, args);
                    }
                    if fname == "push" && args.len() == 2 {
                        let xs_ty = self.infer_expr_type(&args[0].value);
                        let elem_is_rc = self
                            .list_elem_ty(&xs_ty)
                            .is_some_and(|et| self.type_is_rc(&et))
                            || self.expr_produces_rc(&args[1].value);
                        // Non-consuming `push`: own `xs` (inc if ident so `tenon_list_push` COW-clones!)
                        let xs = self.lower_expr_owned(&args[0].value, expected_ty)?;
                        let val = if elem_is_rc {
                            let mv = self.builder.ins().iconst(types::I64, RC_MASK_ALL as i64);
                            self.call_rt_2_void(self.rt.rc_set_rc_mask, xs, mv);
                            self.lower_expr_owned(&args[1].value, None)?
                        } else {
                            let mut temps = Vec::new();
                            let v = self.lower_expr_borrowed(&args[1].value, &mut temps)?;
                            self.flush_temps(&temps);
                            v
                        };
                        return Ok(self.call_rt_2_vals(self.rt.list_push, xs, val));
                    }
                    if fname == "list_filled" && args.len() == 2 {
                        let elem_is_rc = expected_ty
                            .and_then(|t| self.list_elem_ty(t))
                            .is_some_and(|et| self.type_is_rc(&et))
                            || self.type_is_rc(&self.infer_expr_type(&args[1].value))
                            || self.expr_produces_rc(&args[1].value);
                        let mut temps = Vec::new();
                        let count_v = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                        let init_v = self.lower_expr_borrowed(&args[1].value, &mut temps)?;
                        let rc_flag = self
                            .builder
                            .ins()
                            .iconst(types::I64, if elem_is_rc { 1 } else { 0 });
                        let fref = self
                            .module
                            .declare_func_in_func(self.rt.list_filled, self.builder.func);
                        let call = self.builder.ins().call(fref, &[count_v, init_v, rc_flag]);
                        let res = self.builder.inst_results(call)[0];
                        self.flush_temps(&temps);
                        return Ok(res);
                    }
                    if fname == "list_pop_shrink" && args.len() == 1 {
                        let xs = self.lower_expr_owned(&args[0].value, expected_ty)?;
                        return Ok(self.call_rt_1(self.rt.list_pop_shrink, xs));
                    }
                    if fname == "yield_val" && args.len() == 1 {
                        let mut temps = Vec::new();
                        let v = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                        self.flush_temps(&temps);
                        return Ok(self.call_rt_1(self.rt.async_yield_create, v));
                    }
                    if fname == "block_on" && args.len() == 1 {
                        let fut = self.lower_expr_owned(&args[0].value, None)?;
                        return Ok(self.call_rt_1(self.rt.async_block_on, fut));
                    }
                    if let Some(&tag) = self.enum_tags.get(fname) {
                        if !self.func_ids.contains_key(fname) {
                            let mut ptys = self
                                .enum_variants
                                .get(fname)
                                .map(|vl| vl.payload_tys.clone())
                                .unwrap_or_default();
                            if ptys.is_empty() {
                                if let Some(exp) = expected_ty {
                                    let resolved = resolve_alias_ty(exp, self.alias_bases);
                                    if let TypeExpr::Named { name: tn, args: targs, .. } = resolved {
                                        if fname == "Some" && tn == "Option" && !targs.is_empty() {
                                            ptys.push(targs[0].clone());
                                        } else if fname == "Ok" && tn == "Result" && !targs.is_empty() {
                                            ptys.push(targs[0].clone());
                                        } else if fname == "Err" && tn == "Result" && targs.len() >= 2 {
                                            ptys.push(targs[1].clone());
                                        }
                                    }
                                }
                            }
                            return self.lower_enum_ctor(tag, &ptys, args);
                        }
                    }

                    let fid = *self
                        .func_ids
                        .get(fname)
                        .ok_or_else(|| format!("unknown function `{}` in codegen", fname))?;
                    let meta = self.fn_metas.get(fname).cloned();
                    let mut temps = Vec::new();
                    let mut arg_vals = Vec::new();
                    let mut inout_writebacks: Vec<(String, cranelift_codegen::ir::StackSlot)> =
                        Vec::new();
                    let mut inout_field_writebacks: Vec<(Value, usize, cranelift_codegen::ir::StackSlot)> =
                        Vec::new();

                    for (idx, a) in args.iter().enumerate() {
                        let is_inout = meta
                            .as_ref()
                            .and_then(|m| m.params_inout.get(idx).copied())
                            .unwrap_or(false);
                        let pty = meta.as_ref().and_then(|m| m.param_tys.get(idx));
                        if is_inout {
                            match &a.value.kind {
                                ExprKind::Ident(vname) => {
                                    let cur_val = self.load_ident_val(vname)?;
                                    let ss = self.builder.create_sized_stack_slot(StackSlotData::new(
                                        StackSlotKind::ExplicitSlot,
                                        8,
                                        3,
                                    ));
                                    self.builder.ins().stack_store(cur_val, ss, 0);
                                    let addr = self.builder.ins().stack_addr(types::I64, ss, 0);
                                    arg_vals.push(addr);
                                    inout_writebacks.push((vname.clone(), ss));
                                }
                                ExprKind::Field { target: obj, field }
                                    if matches!(&obj.kind, ExprKind::Ident(_)) =>
                                {
                                    let ExprKind::Ident(vname) = &obj.kind else {
                                        unreachable!()
                                    };
                                    let (fidx, _) = self.resolve_field_layout(obj, field)?;
                                    let cur_obj = self.load_ident_val(vname)?;
                                    let uniq_obj = self.call_rt_1(self.rt.rc_ensure_unique, cur_obj);
                                    let var = self.vars[vname];
                                    if self.inout_vars.contains(vname) {
                                        let slot_ptr = self.builder.use_var(var);
                                        self.builder.ins().store(MemFlags::new(), uniq_obj, slot_ptr, 0);
                                    } else {
                                        self.builder.def_var(var, uniq_obj);
                                    }
                                    let idx_v = self.builder.ins().iconst(types::I64, fidx as i64);
                                    let cur_fv = self.call_rt_2_vals(self.rt.rc_get_field, uniq_obj, idx_v);
                                    let ss = self.builder.create_sized_stack_slot(StackSlotData::new(
                                        StackSlotKind::ExplicitSlot,
                                        8,
                                        3,
                                    ));
                                    self.builder.ins().stack_store(cur_fv, ss, 0);
                                    let addr = self.builder.ins().stack_addr(types::I64, ss, 0);
                                    arg_vals.push(addr);
                                    inout_field_writebacks.push((uniq_obj, fidx, ss));
                                }
                                _ => {
                                    return Err(format!(
                                        "`inout` argument {} of `{}` must be a variable or record field",
                                        idx, fname
                                    ));
                                }
                            }
                        } else if matches!(&a.value.kind, ExprKind::List(es) if es.is_empty())
                            && pty.is_some()
                        {
                            let lv = self.lower_expr_owned(&a.value, pty)?;
                            temps.push(lv);
                            arg_vals.push(lv);
                        } else {
                            arg_vals.push(self.lower_expr_borrowed(&a.value, &mut temps)?);
                        }
                    }
                    let local_callee = self.module.declare_func_in_func(fid, self.builder.func);
                    let call = self.builder.ins().call(local_callee, &arg_vals);
                    let ret = self.builder.inst_results(call)[0];
                    self.flush_temps(&temps);

                    for (vname, ss) in inout_writebacks {
                        let updated = self.builder.ins().stack_load(types::I64, ss, 0);
                        let var = self.vars[&vname];
                        if self.inout_vars.contains(&vname) {
                            let slot_ptr = self.builder.use_var(var);
                            self.builder
                                .ins()
                                .store(MemFlags::new(), updated, slot_ptr, 0);
                        } else {
                            self.builder.def_var(var, updated);
                        }
                    }
                    for (uniq_obj, fidx, ss) in inout_field_writebacks {
                        let updated = self.builder.ins().stack_load(types::I64, ss, 0);
                        self.call_rt_3_void(self.rt.rc_set_field_raw, uniq_obj, fidx as i64, updated);
                    }
                    Ok(ret)
                }
                _ => Err("unsupported indirect call in codegen".into()),
            },
            _ => Ok(self.builder.ins().iconst(types::I64, 0)),
        }
    }

    fn lower_enum_ctor(
        &mut self,
        tag: i64,
        payload_tys: &[TypeExpr],
        args: &[CallArg],
    ) -> Result<Value, String> {
        let obj = self.call_rt_2(self.rt.rc_alloc, tag, args.len() as i64);
        let mut rc_mask = 0u64;
        for (idx, a) in args.iter().enumerate() {
            let pty = payload_tys.get(idx);
            let is_rc = pty.is_some_and(|t| self.type_is_rc(t)) || self.expr_produces_rc(&a.value);
            let av = if is_rc {
                if idx < 64 {
                    rc_mask |= 1u64 << idx;
                }
                self.lower_expr_owned(&a.value, pty)?
            } else {
                let mut temps = Vec::new();
                let v = self.lower_expr_borrowed(&a.value, &mut temps)?;
                self.flush_temps(&temps);
                v
            };
            self.call_rt_3_void(self.rt.rc_set_field, obj, idx as i64, av);
        }
        if rc_mask != 0 {
            let mv = self.builder.ins().iconst(types::I64, rc_mask as i64);
            self.call_rt_2_void(self.rt.rc_set_rc_mask, obj, mv);
        }
        Ok(obj)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenon_syntax::parse;

    #[test]
    fn cranelift_jits_and_runs_days_in_month_and_add_odds() {
        let src = r#"
pub type Odd = Int64 where self % 2 != 0
pub type Even = Int64 where self % 2 == 0

pub pure fn add_odds(a Odd, b Odd) Even {
    a + b
}

pub pure fn is_leap_year(y Int) Bool {
    (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0)
}

pub pure fn days_in_month(y Int, m Int) Int {
    match m {
        2 => if is_leap_year(y) { 29 } else { 28 }
        4 | 6 | 9 | 11 => 30
        _ => 31
    }
}

pub fn main() Int {
    assert days_in_month(2024, 2) == 29
    assert days_in_month(2023, 2) == 28
    assert days_in_month(1900, 2) == 28
    assert days_in_month(2000, 2) == 29
    assert add_odds(3, 5) == 8
    days_in_month(2024, 2) + add_odds(3, 5)
}
"#;
        let prog = parse(src).unwrap();
        let jit = JitProgram::compile(&prog).unwrap();
        assert_eq!(jit.call_2("days_in_month", 2024, 2).unwrap(), 29);
        assert_eq!(jit.call_2("days_in_month", 1900, 2).unwrap(), 28);
        assert_eq!(jit.call_2("add_odds", 7, 9).unwrap(), 16);
        assert_eq!(jit.call_0("main").unwrap(), 37);
    }

    static PERCEUS_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn perceus_records_parse_mutate_cow_and_zero_leaks() {
        let src = r#"
pub pure fn is_leap_year(y Int) Bool {
    (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0)
}

pub pure fn days_in_month(y Int, m Int where 1 <= self && self <= 12) Int {
    match m {
        2 => if is_leap_year(y) { 29 } else { 28 }
        4 | 6 | 9 | 11 => 30
        _ => 31
    }
}

pub record Date {
    year  Int where 1 <= self && self <= 9999
    month Int where 1 <= self && self <= 12
    day   Int where 1 <= self && self <= days_in_month(year, month)
}

pub fn shift_to_prev_year_safe(inout d Date)
    requires d.year > 1
{
    mutate d {
        d.year -= 1
        if d.month == 2 && d.day == 29 && !is_leap_year(d.year) {
            d.day = 28
        }
    }
}

pub fn main() Int {
    var d Date = Date { year = 2024, month = 2, day = 29 }
    shift_to_prev_year_safe(d)
    assert d.year == 2023
    assert d.day == 28

    let a Date = Date { year = 2024, month = 2, day = 29 }
    var b Date = a
    shift_to_prev_year_safe(b)
    assert a.year == 2024
    assert a.day == 29
    assert b.year == 2023
    assert b.day == 28

    let ok_day = match Date.parse(2024, 2, 29) {
        Ok(dt) => dt.day
        Err(code) => -code
    }
    let bad_code = match Date.parse(2023, 2, 29) {
        Ok(dt) => dt.day
        Err(code) => -code
    }
    assert ok_day == 29
    assert bad_code == -3

    let s = "hello"
    assert len(s) == 5
    var xs = [1, 2, 3]
    xs = push(xs, 4)
    assert len(xs) == 4
    assert xs[3] == 4

    ok_day + b.day + xs[3]
}
"#;
        let _perceus_guard = PERCEUS_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        perceus::reset_stats();
        let prog = parse(src).unwrap();
        let jit = JitProgram::compile(&prog).unwrap();
        assert!(jit.derived_parsers.contains(&"Date__parse".to_string()));
        let res = jit.call_0("main").unwrap();
        assert_eq!(res, 29 + 28 + 4);
        let stats = perceus::snapshot_stats();
        assert!(
            stats.reuse_hits >= 2,
            "expected Perceus unique in-place reuse hits, got {:?}",
            stats
        );
        assert!(
            stats.cow_clones >= 1,
            "expected MVS COW clone on shared Date, got {:?}",
            stats
        );
        assert_eq!(
            stats.live_objects, 0,
            "Perceus RC must leave 0 leaked heap objects, got {:?}",
            stats
        );
    }

    #[test]
    fn stackless_async_state_machine_suspends_resumes_and_frees_zero_leaks() {
        let src = r#"
@no_heap_alloc
pub pure fn combine(a Int, b Int) Int { a * 2 + b }

pub async fn step(seed Int) Int {
    let first = await yield_val(seed + 10)
    let second = await yield_val(first + 5)
    second
}

pub async fn pipeline(base Int) Int {
    let x = await step(base)
    let y = await yield_val(base * 3)
    combine(x, y)
}

pub fn main() Int {
    block_on(pipeline(7))
}
"#;
        let _perceus_guard = PERCEUS_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        perceus::reset_stats();
        let prog = parse(src).unwrap();
        let jit = JitProgram::compile(&prog).unwrap();
        assert!(jit.compiled_fns.contains(&"pipeline__poll".to_string()));
        assert_eq!(jit.call_0("main").unwrap(), 65);
        let st = perceus::snapshot_stats();
        assert!(
            st.async_suspends >= 3,
            "expected >=3 stackless suspensions, got {:?}",
            st
        );
        assert!(
            st.async_polls >= 6,
            "expected >=6 cooperative polls, got {:?}",
            st
        );
        assert_eq!(
            st.live_objects, 0,
            "stackless async frames must leave 0 leaks, got {:?}",
            st
        );
    }

    #[test]
    fn core_byte_ops_concat_break_continue_text_records_zero_leaks() {
        let src = r#"
pub record Pair {
    key Text
    val Text
}

pub fn make_pair(k Text, v Text) Pair {
    Pair { key = k ++ "_k", val = v }
}

pub fn reverse_ascii(s Text) Text {
    var out = ""
    var i = len(s) - 1
    while i >= 0 {
        let b = byte_at(s, i)
        if b == 32 {
            i -= 1
            continue
        }
        out = out ++ byte_to_text(b)
        if len(out) == 5 {
            break
        }
        i -= 1
    }
    out
}

pub fn main() Int {
    let p = make_pair("foo", "bar")
    assert p.key == "foo_k"
    assert p.val == "bar"
    let r = reverse_ascii("h e l l o !")
    assert r == "!olle"
    let sl = str_slice("tenon-lang", 0, 5)
    assert sl == "tenon"
    let m = match sl {
        "rust" => 1
        "tenon" => 42
        _ => 0
    }
    assert m == 42
    m
}
"#;
        let _perceus_guard = PERCEUS_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        perceus::reset_stats();
        let prog = parse(src).unwrap();
        let jit = JitProgram::compile(&prog).unwrap();
        assert_eq!(jit.call_0("main").unwrap(), 42);
        let st = perceus::snapshot_stats();
        assert_eq!(
            st.live_objects, 0,
            "core text/record/loop ops must leave 0 leaks, got {:?}",
            st
        );
    }

    #[test]
    fn recursive_payload_enum_option_result_and_rc_lists_zero_leaks() {
        let src = r#"
pub record Entry {
    key Text
    val Val
}

pub enum Val {
    VNull
    VInt(Int)
    VStr(Text)
    VArr(List[Val])
    VObj(List[Entry])
}

pub fn get_field(v Val, k Text) Option[Val] {
    match v {
        VObj(entries) => {
            var i = 0
            while i < len(entries) {
                let e = entries[i]
                if e.key == k {
                    return Some(e.val)
                }
                i += 1
            }
            None
        }
        _ => None
    }
}

pub fn summarize(v Val) Result[Text, Text] {
    match v {
        VNull => Ok("null")
        VInt(n) => Ok("int")
        VStr(s) => Ok("str:" ++ s)
        VArr(xs) => Ok("arr")
        VObj(es) => Ok("obj")
    }
}

pub fn main() Int {
    var xs List[Val] = []
    xs = push(xs, VInt(10))
    xs = push(xs, VStr("hello"))
    var entries List[Entry] = []
    entries = push(entries, Entry { key = "items", val = VArr(xs) })
    entries = push(entries, Entry { key = "name", val = VStr("tenon") })
    let obj = VObj(entries)

    let name_ok = match get_field(obj, "name") {
        Some(v) => match summarize(v) {
            Ok(txt) => txt == "str:tenon"
            Err(_) => false
        }
        None => false
    }
    assert name_ok

    let missing_ok = match get_field(obj, "nope") {
        Some(_) => false
        None => true
    }
    assert missing_ok
    99
}
"#;
        let _perceus_guard = PERCEUS_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        perceus::reset_stats();
        let prog = parse(src).unwrap();
        let jit = JitProgram::compile(&prog).unwrap();
        assert_eq!(jit.call_0("main").unwrap(), 99);
        let st = perceus::snapshot_stats();
        assert_eq!(
            st.live_objects, 0,
            "recursive payload enums + Option/Result + RC Lists must leave 0 leaks, got {:?}",
            st
        );
    }

    #[test]
    fn interfaces_methods_static_monomorph_and_dyn_vtable_zero_leaks() {
        let src = r#"
pub interface Greeter {
    pure fn greet(self, who Text) Text
    pure fn shout(self, who Text) Text {
        self.greet(who) ++ "!"
    }
}

pub interface Accumulator {
    fn add(inout self, delta Int) Int
    pure fn current(self) Int
}

pub record EnGreeter { prefix Text }
pub record ZhGreeter { prefix Text }

impl Greeter for EnGreeter {
    pure fn greet(self, who Text) Text { self.prefix ++ " " ++ who }
}

impl Greeter for ZhGreeter {
    pure fn greet(self, who Text) Text { self.prefix ++ ", " ++ who }
}

pub record Counter { val Int }

impl Counter {
    pure fn doubled(self) Int { self.val * 2 }
}

impl Accumulator for Counter {
    fn add(inout self, delta Int) Int {
        self.val += delta
        self.val
    }
    pure fn current(self) Int { self.val }
}

pub pure fn run_static[T Greeter](g T, name Text) Text {
    g.shout(name)
}

pub pure fn run_dyn(g dyn Greeter, name Text) Text {
    g.shout(name)
}

pub fn main() Int {
    let en = EnGreeter { prefix = "Hello" }
    let zh = ZhGreeter { prefix = "NiHao" }
    assert run_static(en, "Tenon") == "Hello Tenon!"
    assert run_static(zh, "Tenon") == "NiHao, Tenon!"

    var list List[dyn Greeter] = []
    list = push(list, en)
    list = push(list, zh)
    let b = buf_new()
    let _ = buf_push_str(b, run_dyn(list[0], "World"))
    let _ = buf_push_byte(b, 124)
    let _ = buf_push_str(b, list[1].greet("World"))
    let joined = buf_finish(b)
    assert joined == "Hello World!|NiHao, World"

    let c0 = Counter { val = 7 }
    assert c0.doubled() == 14

    var c dyn Accumulator = Counter { val = 10 }
    let c2 dyn Accumulator = c
    let v1 = c.add(5)
    assert v1 == 15
    assert c.current() == 15
    assert c2.current() == 10
    v1
}
"#;
        let _perceus_guard = PERCEUS_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        perceus::reset_stats();
        let prog = parse(src).unwrap();
        let jit = JitProgram::compile(&prog).unwrap();
        assert_eq!(jit.call_0("main").unwrap(), 15);
        let st = perceus::snapshot_stats();
        assert_eq!(
            st.live_objects, 0,
            "interfaces + default methods + static monomorph + dyn vtable + COW must leave 0 leaks, got {:?}",
            st
        );
    }

    #[test]
    fn oop_list_push_clear_and_inout_field_projection_zero_leaks() {
        let src = r#"
pub interface Plugin {
    pure fn label(self) Text
}

pub record Alpha { tag Text }
pub record Beta  { tag Text }

impl Plugin for Alpha {
    pure fn label(self) Text { "A:" ++ self.tag }
}

impl Plugin for Beta {
    pure fn label(self) Text { "B:" ++ self.tag }
}

impl[T] List[T] {
    pub pure fn len(self) Int { len(self) }
    pub pure fn push(inout self, item T) { self = push(self, item) }
    pub pure fn clear(inout self) {
        let empty List[T] = []
        self = empty
    }
}

pub record Registry {
    plugins List[dyn Plugin]
}

impl Registry {
    pub pure fn new() Registry {
        let empty List[dyn Plugin] = []
        Registry { plugins = empty }
    }

    pub pure fn register(inout self, p dyn Plugin) {
        self.plugins.push(p)
    }

    pub pure fn reset(inout self) {
        self.plugins.clear()
    }
}

pub fn main() Int {
    var reg = Registry.new()
    reg.register(Alpha { tag = "one" })
    let snapshot = reg
    reg.register(Beta { tag = "two" })

    assert snapshot.plugins.len() == 1
    assert reg.plugins.len() == 2
    assert reg.plugins[0].label() == "A:one"
    assert reg.plugins[1].label() == "B:two"

    reg.reset()
    assert reg.plugins.len() == 0
    assert snapshot.plugins.len() == 1
    42
}
"#;
        let _perceus_guard = PERCEUS_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        perceus::reset_stats();
        let prog = parse(src).unwrap();
        let jit = JitProgram::compile(&prog).unwrap();
        assert_eq!(jit.call_0("main").unwrap(), 42);
        let st = perceus::snapshot_stats();
        assert_eq!(
            st.live_objects, 0,
            "OOP List.push/clear + COW inout field projection must leave 0 leaks, got {:?}",
            st
        );
    }

    #[test]
    fn list_index_assign_filled_pop_bitwise_and_cow_zero_leaks() {
        let src = r#"
pub record Table {
    slots List[Int]
    names List[Text]
}

impl Table {
    pub pure fn set_slot(inout self, idx Int, v Int, name Text) {
        self.slots[idx] = v
        self.names[idx] = name
    }
}

pub fn main() Int {
    var xs = list_filled(4, -1)
    assert len(xs) == 4
    let snap = xs
    xs[0] = (15 ^ 6) << 2
    xs[1] = 128 >> 3
    xs[2] = 7
    xs[3] = 9
    assert snap[0] == -1
    assert xs[0] == 36
    assert xs[1] == 16
    xs = list_pop_shrink(xs)
    assert len(xs) == 3

    var tbl = Table {
        slots = list_filled(2, 0),
        names = list_filled(2, "init")
    }
    let tbl_snap = tbl
    tbl.set_slot(1, 42, "updated")
    assert tbl_snap.slots[1] == 0
    assert tbl_snap.names[1] == "init"
    assert tbl.slots[1] == 42
    assert tbl.names[1] == "updated"
    tbl.slots[1]
}
"#;
        let _perceus_guard = PERCEUS_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        perceus::reset_stats();
        let prog = parse(src).unwrap();
        let jit = JitProgram::compile(&prog).unwrap();
        assert_eq!(jit.call_0("main").unwrap(), 42);
        let st = perceus::snapshot_stats();
        assert_eq!(
            st.live_objects, 0,
            "list_filled + list_pop_shrink + index assign COW + bitwise ops must leave 0 leaks, got {:?}",
            st
        );
    }

    #[test]
    fn generic_record_monomorph_custom_key_map_zero_leaks() {
        let src = r#"
pub interface Eq {
    pure fn eq(self, other Self) Bool
}

pub interface Hash {
    pure fn hash(self) Int
}

impl Eq for Int {
    pub pure fn eq(self, other Int) Bool { self == other }
}

impl Hash for Int {
    pub pure fn hash(self) Int { self & 9223372036854775807 }
}

pub record PointKey {
    x Int
    y Int
}

impl Eq for PointKey {
    pub pure fn eq(self, other PointKey) Bool { self.x == other.x && self.y == other.y }
}

impl Hash for PointKey {
    pub pure fn hash(self) Int { ((self.x * 31) ^ self.y) & 9223372036854775807 }
}

pub record MiniMap[K Hash + Eq, V] {
    keys List[K]
    vals List[V]
}

impl[K Hash + Eq, V] MiniMap[K, V] {
    pub pure fn new() MiniMap[K, V] {
        let ek List[K] = []
        let ev List[V] = []
        MiniMap { keys = ek, vals = ev }
    }

    pub pure fn put(inout self, k K, v V) {
        var i = 0
        while i < len(self.keys) {
            if self.keys[i].eq(k) {
                self.vals[i] = v
                return
            }
            i += 1
        }
        self.keys = push(self.keys, k)
        self.vals = push(self.vals, v)
    }

    pub pure fn get_or(self, k K, fallback V) V {
        let _ = k.hash()
        var i = 0
        while i < len(self.keys) {
            if self.keys[i].eq(k) {
                return self.vals[i]
            }
            i += 1
        }
        fallback
    }
}

pub fn main() Int {
    var m1 = MiniMap[PointKey, Text].new()
    let p1 = PointKey { x = 1, y = 2 }
    let p2 = PointKey { x = 3, y = 4 }
    m1.put(p1, "alpha")
    m1.put(p2, "beta")
    m1.put(PointKey { x = 1, y = 2 }, "gamma")
    assert m1.get_or(PointKey { x = 1, y = 2 }, "") == "gamma"
    assert m1.get_or(PointKey { x = 3, y = 4 }, "") == "beta"

    var m2 MiniMap[Int, Int] = MiniMap.new()
    m2.put(10, 100)
    m2.put(20, 200)
    assert m2.get_or(10, 0) + m2.get_or(20, 0) == 300
    len(m1.keys) * 100 + m2.get_or(20, 0)
}
"#;
        let _perceus_guard = PERCEUS_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        perceus::reset_stats();
        let prog = parse(src).unwrap();
        let jit = JitProgram::compile(&prog).unwrap();
        assert_eq!(jit.call_0("main").unwrap(), 400);
        let st = perceus::snapshot_stats();
        assert_eq!(
            st.live_objects, 0,
            "generic record monomorphization across RC and primitive keys/vals must leave 0 leaks, got {:?}",
            st
        );
    }

    #[test]
    fn p0_expressiveness_jit_zero_leaks() {
        let src = r#"
pub type NonNegative = Int where self >= 0

pub interface Display {
    pure fn display(self) Text
}
pub interface Eq {
    pure fn eq(self, other Self) Bool
}
pub interface Hash {
    pure fn hash(self) NonNegative
}

impl Display for Text {
    pub pure fn display(self) Text { self }
}
impl Display for Float {
    pub pure fn display(self) Text { float_to_text(self) }
}
impl Eq for Int {
    pub pure fn eq(self, other Int) Bool { self == other }
}
impl Hash for Int {
    pub pure fn hash(self) NonNegative { self & 9223372036854775807 }
}
impl Eq for Text {
    pub pure fn eq(self, other Text) Bool { self == other }
}
impl Hash for Text {
    pub pure fn hash(self) NonNegative { len(self) }
}

@derive(Eq, Hash)
pub record TagPt {
    x   Int
    tag Text
}

pub enum Tree[T] {
    Empty
    Leaf(T)
    Node(Tree[T], T, Tree[T])
}

pub pure fn sum_tree(t Tree[Int]) Int {
    match t {
        Empty => 0
        Leaf(v) => v
        Node(Leaf(a), v, Leaf(b)) => a + v + b
        Node(l, v, r) => sum_tree(l) + v + sum_tree(r)
    }
}

pub pure fn join_tree(t Tree[Text]) Text {
    match t {
        Empty => ""
        Leaf(s) => s
        Node(l, s, r) => join_tree(l) ++ s ++ join_tree(r)
    }
}

pub pure fn make_greeter(prefix Text, bump Int) fn(Text) Text {
    |who| prefix ++ who ++ if bump > 0 { "!" } else { "." }
}

pub pure fn try_chain_res(ok_flag Bool) Result[Int, Text] {
    let alive = "keep_me_rc"
    let first Result[Int, Text] = if ok_flag { Ok(10) } else { Err("fail:" ++ alive) }
    let a = first?
    let second Result[Int, Text] = Ok(a + len(alive))
    let b = second?
    Ok(b)
}

pub pure fn try_chain_opt(present Bool) Option[Int] {
    let alive = "opt_rc"
    let o Option[Int] = if present { Some(20) } else { None }
    let v = o?
    Some(v + len(alive))
}

pub fn main() Int {
    // Float ops + parse/format
    var f = 2.0
    f += 1.5
    f *= 2.0
    f /= 3.5
    assert f == 2.0
    assert float_to_int(3.9) == 3
    assert int_to_float(5) == 5.0
    assert parse_float("1.25") == 1.25
    assert !float_is_nan(f)

    // Escaping closure capturing RC Text + primitive Int
    let g = make_greeter("hi ", 1)
    assert g("tenon") == "hi tenon!"

    // `for` over Range (with continue/break), standalone Range, and List
    var acc = 0
    for i in 0..8 {
        if i == 2 { continue }
        if i == 6 { break }
        acc += i
    }
    assert acc == 13

    let r = 1..5
    var rsum = 0
    for k in r { rsum += k }
    assert rsum == 10

    var lsum = 0
    let ws = ["a", "bb", "ccc"]
    for w in ws { lsum += len(w) }
    assert lsum == 6

    // Compound bitwise/mul/div assignments
    var bits = 4
    bits <<= 2
    bits >>= 1
    bits |= 3
    bits &= 10
    bits ^= 6
    bits *= 3
    bits /= 4
    assert bits == 9

    // `?` happy + early-return paths with RC locals alive
    assert match try_chain_res(true) { Ok(v) => v Err(_) => -1 } == 20
    assert match try_chain_res(false) { Ok(_) => "" Err(e) => e } == "fail:keep_me_rc"
    assert match try_chain_opt(true) { Some(v) => v None => -1 } == 26
    assert match try_chain_opt(false) { Some(_) => 1 None => 0 } == 0

    // Tuples + destructuring + nested match
    let tup = (7, "abc", 1.5)
    let (tn, ts, tf) = tup
    assert tn == 7 && ts == "abc" && tf == 1.5 && tup.0 == 7
    let nested Option[Result[(Int, Text), Text]] = Some(Ok((4, "xy")))
    let nv = match nested {
        Some(Ok((n, s))) => n + len(s)
        Some(Err(_)) => -1
        None => -2
    }
    assert nv == 6

    // Generic recursive enum at both primitive and RC types
    let it Tree[Int] = Node(Node(Leaf(1), 2, Leaf(3)), 4, Leaf(5))
    assert sum_tree(it) == 15
    let tt Tree[Text] = Node(Leaf("te"), "n", Leaf("on"))
    assert join_tree(tt) == "tenon"

    // f-string interpolation
    let fs = f"{ts}={tf}"
    assert fs == "abc=1.5"

    // @derive(Eq, Hash) + `==`/`!=`
    let p1 = TagPt { x = 1, tag = "k" }
    let p2 = TagPt { x = 1, tag = "k" }
    let p3 = TagPt { x = 2, tag = "k" }
    assert p1 == p2 && p1 != p3
    42
}
"#;
        let _perceus_guard = PERCEUS_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        perceus::reset_stats();
        let prog = parse(src).unwrap();
        let jit = JitProgram::compile(&prog).unwrap();
        assert_eq!(jit.call_0("main").unwrap(), 42);
        let st = perceus::snapshot_stats();
        assert_eq!(
            st.live_objects, 0,
            "P0 expressiveness features in Cranelift JIT must leave 0 leaks, got {:?}",
            st
        );
    }

    #[test]
    fn systems_suite_jit_zero_leaks() {
        let _perceus_guard = PERCEUS_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        let entry = repo_root.join("examples/systems_test.tn");
        let prog = load_program_with_imports(&entry).unwrap();

        perceus::reset_stats();
        let jit = JitProgram::compile(&prog).unwrap();
        assert_eq!(jit.call_0("main").unwrap(), 1110);
        let st = perceus::snapshot_stats();
        assert_eq!(
            st.live_objects, 0,
            "full systems programming suite (S1+S2+S3) in Cranelift JIT must leave 0 leaks, got {:?}",
            st
        );
    }

    #[test]
    fn llvm_aot_tier2_stack_promotion_slab_arena_and_flat_simd_list_zero_leaks() {
        let src = r#"
pub record Vec2 {
    x Int
    y Int
}

@arena
pub fn compute_in_arena(n Int where self >= 1 && self <= 64) Int {
    var xs = list_filled(n, 2)
    var total = 0
    var i = 0
    while i < n invariant i >= 0 && i <= n {
        let v = Vec2 { x = xs[i], y = i }
        total = total + v.x + v.y
        i += 1
    }
    total
}

pub fn simd_flat_dot() Int {
    var a = list_filled(8, simd_f64x4_splat(1.0))
    var b = list_filled(8, simd_f64x4_splat(2.0))
    a[0] = simd_f64x4_new(1.0, 2.0, 3.0, 4.0)
    b[0] = simd_f64x4_new(2.0, 3.0, 4.0, 5.0)
    var s = 0.0
    var i = 0
    while i < 8 {
        s = s + simd_f64x4_dot(a[i], b[i])
        i += 1
    }
    s as Int
}

pub fn main() Int {
    let a = compute_in_arena(10)
    let b = simd_flat_dot()
    a + b
}
"#;
        let prog = parse(src).unwrap();
        let rep = tenon_verify::check_program(&prog);
        assert!(rep.ok, "verify failed: {:?}", rep.diagnostics);
        assert!(rep.mir.arena_verified_fns.contains("compute_in_arena"));
        assert!(
            rep.mir
                .stack_promotable_vars
                .get("compute_in_arena")
                .is_some_and(|s| s.contains("v"))
        );

        let ll_path = std::env::temp_dir().join(format!("tenon_tier2_{}.ll", std::process::id()));
        let bin_path = std::env::temp_dir().join(format!("tenon_tier2_{}_bin", std::process::id()));
        let mut opts = llvm_aot::BuildOptions::default();
        opts.apply_verified_mir(&rep.mir);
        opts.emit_ll_path = Some(ll_path.clone());
        let (_art, out) = llvm_aot::compile_and_run_for_test(&prog, &bin_path, &opts).unwrap();

        let ir = std::fs::read_to_string(&ll_path).unwrap();
        assert!(ir.contains("@tenon_arena_enter"), "expected @arena enter in LLVM IR");
        assert!(ir.contains("@tenon_arena_exit"), "expected @arena exit in LLVM IR");
        assert!(ir.contains("%stkrec_"), "expected stack-promoted record alloca in LLVM IR");
        assert!(ir.contains("@tenon_flat_list_filled"), "expected flat unboxed SIMD list in LLVM IR");

        let stdout = String::from_utf8_lossy(&out.stdout);
        // compute_in_arena(10) = 10*2 + (0..9)=45 = 65
        // simd_flat_dot() = (1*2+2*3+3*4+4*5)=40 + 7*(4*2)=56 = 96
        // total = 161
        assert!(
            out.status.success() && stdout.contains("main() => 161"),
            "unexpected run output: status={:?} stdout={} stderr={}",
            out.status,
            stdout,
            String::from_utf8_lossy(&out.stderr)
        );
        if stdout.contains("live_objects=") {
            assert!(stdout.contains("live_objects=0"), "leaked objects: {}", stdout);
        }
        let _ = std::fs::remove_file(&ll_path);
    }

    #[test]
    fn differential_fuzz_jit_vs_llvm_aot_vs_wasm32_and_parser_roundtrip() {
        let has_wasm = wasm_llvm::has_llvm_wasm32_toolchain()
            && std::process::Command::new("node")
                .arg("--version")
                .output()
                .is_ok_and(|o| o.status.success());

        // Deterministic SplitMix64 differential fuzzer: generates diverse programs exercising
        // arithmetic, bitwise, branching, while loops, records, lists, and refinement casts,
        // and asserts IDENTICAL outputs + `live_objects == 0` across Cranelift JIT, LLVM AOT, and Wasm32 V8!
        let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next_u64 = || -> u64 {
            seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };

        // Combine all 8 fuzzed cases into ONE multi-function program so all 8 run identically across
        // Cranelift JIT, LLVM AOT (1 native binary), and Full LLVM Wasm32 V8 (1 wasm module) without
        // paying repeated OS Mach-O Gatekeeper/syspolicyd first-launch overhead per case.
        let pid = std::process::id();
        let mut combined = String::from(
            "pub type SmallPos = Int where self >= 1 && self <= 100\n\npub record Pair {\n    x Int\n    y Int\n}\n\n",
        );
        let mut call_sum = Vec::new();
        for case_idx in 0..8 {
            let a = ((next_u64() % 40) as i64) + 5;
            let b = ((next_u64() % 30) as i64) + 3;
            let n = ((next_u64() % 12) as i64) + 4;
            let mask = [7, 15, 31, 63][(next_u64() % 4) as usize];
            combined.push_str(&format!(
                r#"
pub pure fn step_{idx}(v Int, k Int) Int {{
    if (v & 1) == 0 {{
        (v + k) ^ (k & {mask})
    }} else {{
        (v * 3) + k - (v & {mask})
    }}
}}

pub fn fuzz_case_{idx}() Int {{
    let p = Pair {{ x = {a}, y = {b} }}
    var xs = list_filled({n}, p.x)
    var acc = p.y
    var i = 0
    while i < {n} invariant 0 <= i && i <= {n} {{
        let v = step_{idx}(xs[i] + i, p.y + i)
        xs[i] = v
        acc = acc + (v & 255)
        i += 1
    }}
    let sp SmallPos = {a}
    let widened Int = refine_cast[Int](sp)
    acc + widened + xs[0]
}}
"#,
                idx = case_idx,
                a = a,
                b = b,
                n = n,
                mask = mask
            ));
            call_sum.push(format!("fuzz_case_{}()", case_idx));
        }
        combined.push_str(&format!(
            "\npub fn main() Int {{\n    {}\n}}\n",
            call_sum.join(" + ")
        ));

        let prog = parse(&combined).expect("combined fuzzed program must parse");
        let rep = tenon_verify::check_program(&prog);
        assert!(rep.ok, "combined fuzz failed verify: {:?}", rep.diagnostics);

        // Backend 1: Cranelift JIT + Perceus live_objects == 0 (hold lock ONLY during JIT execution!)
        let jit_res = {
            let _perceus_guard = PERCEUS_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            perceus::reset_stats();
            let jit = JitProgram::compile(&prog).unwrap();
            let res = jit.call_0("main").unwrap();
            let st = perceus::snapshot_stats();
            assert_eq!(st.live_objects, 0, "JIT leak in combined differential fuzz");
            res
        };

        // Backend 2: LLVM AOT + Perceus live_objects == 0 (all 8 cases in 1 binary)
        let bin = std::env::temp_dir().join(format!("tenon_fuzz_all_{}", pid));
        let mut opts = llvm_aot::BuildOptions::default();
        opts.opt_level = "O0".into();
        opts.apply_verified_mir(&rep.mir);
        let (_art, aot_out) = llvm_aot::compile_and_run_for_test(&prog, &bin, &opts).unwrap();
        let aot_stdout = String::from_utf8_lossy(&aot_out.stdout);
        assert!(
            aot_out.status.success()
                && aot_stdout.contains(&format!("main() => {}", jit_res))
                && aot_stdout.contains("live_objects=0"),
            "LLVM AOT mismatch in combined fuzz: expected {}, got {}",
            jit_res,
            aot_stdout
        );

        // Backend 3: Full LLVM Wasm32 in Node.js V8 + Perceus live_objects == 0 (all 8 cases in 1 wasm module)
        if has_wasm {
            let wasm_p = std::env::temp_dir().join(format!("tenon_fuzz_all_{}.wasm", pid));
            wasm_llvm::compile_program_to_wasm32_llvm(&prog, &wasm_p, &opts).unwrap();
            let js = format!(
                "const fs=require('fs');(async()=>{{const {{instance}}=await WebAssembly.instantiate(fs.readFileSync({:?}),{{}});const r=instance.exports.main();const l=instance.exports.tenon_wasm_live_objects();console.log(Number(r)+','+Number(l));}})();",
                wasm_p.to_string_lossy()
            );
            let wout = std::process::Command::new("node").args(["-e", &js]).output().unwrap();
            let _ = std::fs::remove_file(&wasm_p);
            let ws = String::from_utf8_lossy(&wout.stdout);
            assert_eq!(
                ws.trim(),
                format!("{},0", jit_res),
                "Wasm32 mismatch in combined fuzz: expected {},0 got {}",
                jit_res,
                ws
            );
        }
    }
}
