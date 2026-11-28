// std/gpu/common/tenon_io_hal.h — Tenon v19 Zero-Dependency Multi-Platform Async Storage-to-GPU IO Engine
//
// Exploits each OS + hardware vendor's fastest storage-to-compute path with ZERO external library dependencies:
//   - Linux Tier 0: NVIDIA GPUDirect Storage (`dlopen("libcufile.so.0")`, persistent cuFile handles + `cuFileBufRegister`
//                   VRAM slabs + batched NVMe -> GPU VRAM zero-CPU-copy DMA)
//   - Linux Tier 1: Raw Linux `io_uring` syscalls (`SYS_io_uring_setup/enter/register`, ZERO liburing!):
//                   supports `IORING_SETUP_SQPOLL` (opt-in), `IORING_REGISTER_BUFFERS` (`IORING_OP_READ_FIXED` zero-pin),
//                   and `IORING_REGISTER_FILES` (`IOSQE_FIXED_FILE`)
//   - Linux Tier 2: `pread` + `POSIX_FADV_DONTNEED` (container seccomp fallback)
//   - macOS       : `F_NOCACHE` + `fcntl(F_RDADVISE)` NVMe lookahead + Apple Unified Memory zero-copy `MTLBuffer`
//   - Windows 11  : `FILE_FLAG_OVERLAPPED | FILE_FLAG_NO_BUFFERING` + persistent `CreateIoCompletionPort` (`IOCP`)
//                   with `GetQueuedCompletionStatusEx` batched reaping + `PrefetchVirtualMemory` lookahead

#ifndef TENON_IO_HAL_H
#define TENON_IO_HAL_H

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#  include <io.h>
#  define TN_IO_ATOMIC_ADD64(p, v) InterlockedExchangeAdd64((volatile LONG64*)(p), (LONG64)(v))
#elif defined(__APPLE__)
#  include <unistd.h>
#  include <fcntl.h>
#  include <errno.h>
#  include <sys/stat.h>
#  include <sys/mman.h>
#  define TN_IO_ATOMIC_ADD64(p, v) __sync_fetch_and_add((p), (v))
#elif defined(__linux__)
#  include <unistd.h>
#  include <fcntl.h>
#  include <errno.h>
#  include <dlfcn.h>
#  include <sys/stat.h>
#  include <sys/mman.h>
#  include <sys/uio.h>
#  include <sys/syscall.h>
#  if defined(__has_include)
#    if __has_include(<linux/io_uring.h>)
#      include <linux/io_uring.h>
#      define TENON_HAVE_IO_URING_HDR 1
#    endif
#  endif
#  define TN_IO_ATOMIC_ADD64(p, v) __sync_fetch_and_add((p), (v))
#else
#  define TN_IO_ATOMIC_ADD64(p, v) (*(p) += (v))
#endif

// IO Backend IDs (returned via `tn_*_info(h, 64)`)
#define TN_IO_BACKEND_UNKNOWN        0
#define TN_IO_BACKEND_MACOS_RDADVISE 1
#define TN_IO_BACKEND_LINUX_IOURING  2
#define TN_IO_BACKEND_LINUX_GDS      3
#define TN_IO_BACKEND_LINUX_PREAD    4
#define TN_IO_BACKEND_WIN_IOCP       5

#define TN_IO_MAX_BATCH      64
#define TN_IO_MAX_REG_BUFS   32
#define TN_IO_MAX_REG_FILES  64
#define TN_IO_MAX_GDS_FILES  64
#define TN_IO_MAX_GDS_VBUFS  64
#define TN_IO_MAX_IOCP_FILES 64

typedef struct {
    int      fd;          // POSIX fd (or slot index on _WIN32)
#if defined(_WIN32)
    HANDLE   hfile;
#endif
    void*    dst;         // Host staging buffer (pinned for DMA, registered io_uring slab, or Apple Unified MTLBuffer)
    size_t   len;
    uint64_t off;
} TnIoReq;

typedef struct {
    int      fd;
    void*    gpu_dptr;    // Target GPU VRAM device pointer (e.g., CUdeviceptr cast to void*)
    size_t   len;
    uint64_t file_off;
    uint64_t vram_off;    // Offset within registered VRAM buffer (usually 0)
} TnIoGdsReq;

#if defined(__linux__)
// Minimal GPUDirect Storage (cuFile) dynamic ABI — zero compile-time dependency on libcufile!
typedef struct { int err; int cu_err; } TnCUfileError;
typedef enum { TN_CU_FILE_HANDLE_TYPE_OPAQUE_FD = 1 } TnCUfileFileHandleType;
typedef struct { TnCUfileFileHandleType type; union { int fd; void* handle; } handle; } TnCUfileDescr;
typedef void* TnCUfileHandle;
typedef TnCUfileError (*pfn_cuFileDriverOpen)(void);
typedef TnCUfileError (*pfn_cuFileDriverClose)(void);
typedef TnCUfileError (*pfn_cuFileHandleRegister)(TnCUfileHandle*, TnCUfileDescr*);
typedef void          (*pfn_cuFileHandleDeregister)(TnCUfileHandle);
typedef TnCUfileError (*pfn_cuFileBufRegister)(const void*, size_t, int);
typedef TnCUfileError (*pfn_cuFileBufDeregister)(const void*);
typedef ssize_t       (*pfn_cuFileRead)(TnCUfileHandle, void*, size_t, off_t, off_t);
#endif

typedef struct {
    int      backend_id;
    int      use_nocache;
    int      sqpoll_active;
    int      reg_buf_count;
    int      reg_file_count;
    int      gds_handle_count;
    int      gds_vbuf_count;
    int64_t  total_bytes;
    int64_t  total_batches;
    int64_t  uring_sqes;
    int64_t  fixed_buf_reads;
    int64_t  gds_bytes;
    int64_t  advise_calls;
    int64_t  iocp_batches;

    // Registered pinned host buffer table (used by Linux IORING_OP_READ_FIXED & telemetry on all OSes)
    void*    reg_buf_ptrs[TN_IO_MAX_REG_BUFS];
    size_t   reg_buf_lens[TN_IO_MAX_REG_BUFS];
    int      reg_files[TN_IO_MAX_REG_FILES];

#if defined(__linux__) && defined(TENON_HAVE_IO_URING_HDR)
    int      uring_fd;
    int      uring_ready;
    volatile int uring_lock;
    void*    sq_ring;
    size_t   sq_ring_sz;
    void*    cq_ring;
    size_t   cq_ring_sz;
    struct io_uring_sqe* sqes;
    size_t   sqes_sz;
    uint32_t* sq_head;
    uint32_t* sq_tail;
    uint32_t* sq_mask;
    uint32_t* sq_flags;
    uint32_t* sq_array;
    uint32_t* cq_head;
    uint32_t* cq_tail;
    uint32_t* cq_mask;
    struct io_uring_cqe* cqes;
#endif
#if defined(__linux__)
    void*    cufile_lib;
    int      gds_ready;
    volatile int gds_lock;
    int      gds_fds[TN_IO_MAX_GDS_FILES];
    TnCUfileHandle gds_handles[TN_IO_MAX_GDS_FILES];
    void*    gds_vbufs[TN_IO_MAX_GDS_VBUFS];
    size_t   gds_vlens[TN_IO_MAX_GDS_VBUFS];
    pfn_cuFileDriverOpen        cuFileDriverOpen;
    pfn_cuFileDriverClose       cuFileDriverClose;
    pfn_cuFileHandleRegister    cuFileHandleRegister;
    pfn_cuFileHandleDeregister  cuFileHandleDeregister;
    pfn_cuFileBufRegister       cuFileBufRegister;
    pfn_cuFileBufDeregister     cuFileBufDeregister;
    pfn_cuFileRead              cuFileRead;
#endif
#if defined(_WIN32)
    HANDLE             iocp;
    CRITICAL_SECTION   iocp_cs;
    CONDITION_VARIABLE iocp_cv;
    int                iocp_cs_ready;
    int                iocp_leader_active;
    HANDLE             iocp_files[TN_IO_MAX_IOCP_FILES];
    int                iocp_file_count;
#endif
} TnIoEngine;

static inline const char* tn_io_backend_name(int id) {
    switch (id) {
        case TN_IO_BACKEND_MACOS_RDADVISE: return "macos-f_rdadvise+nocache-unified";
        case TN_IO_BACKEND_LINUX_IOURING:  return "linux-io_uring-raw-syscall+fixedbuf";
        case TN_IO_BACKEND_LINUX_GDS:      return "linux-gpudirect-storage-cufile+fixed";
        case TN_IO_BACKEND_LINUX_PREAD:    return "linux-pread+fadvise";
        case TN_IO_BACKEND_WIN_IOCP:       return "windows-iocp-overlapped-ex";
        default:                           return "generic-posix";
    }
}

static inline int64_t tn_io_info(const TnIoEngine* io, int field_id) {
    if (!io) return 0;
    switch (field_id) {
        case 0: return io->backend_id;
        case 1: return io->total_bytes;
        case 2: return io->total_batches;
        case 3: return io->uring_sqes;
        case 4: return io->gds_bytes;
        case 5: return io->advise_calls;
        case 6: return io->reg_buf_count;
        case 7: return io->fixed_buf_reads;
        case 8: return io->sqpoll_active;
        case 9: return io->gds_vbuf_count;
        case 10: return io->iocp_batches;
        default: return 0;
    }
}

static inline int tn_io_init(TnIoEngine* io, int use_nocache) {
    if (!io) return 0;
    memset(io, 0, sizeof(*io));
    io->use_nocache = use_nocache;
    for (int i = 0; i < TN_IO_MAX_REG_FILES; i++) io->reg_files[i] = -1;
#if defined(__linux__)
    for (int i = 0; i < TN_IO_MAX_GDS_FILES; i++) io->gds_fds[i] = -1;
#endif
#if defined(__APPLE__)
    io->backend_id = TN_IO_BACKEND_MACOS_RDADVISE;
    return 1;
#elif defined(_WIN32)
    InitializeCriticalSection(&io->iocp_cs);
    InitializeConditionVariable(&io->iocp_cv);
    io->iocp_cs_ready = 1;
    io->iocp_leader_active = 0;
    io->iocp = CreateIoCompletionPort(INVALID_HANDLE_VALUE, NULL, 0, 0);
    io->backend_id = TN_IO_BACKEND_WIN_IOCP;
    return 1;
#elif defined(__linux__)
    io->backend_id = TN_IO_BACKEND_LINUX_PREAD;
    // Tier 0: Probe NVIDIA GPUDirect Storage (`libcufile.so.0`) dynamically (zero link dependency)
    const char* gds_env = getenv("TENON_ENABLE_GDS");
    if (!gds_env || atoi(gds_env) != 0) {
        io->cufile_lib = dlopen("libcufile.so.0", RTLD_LAZY);
        if (!io->cufile_lib) io->cufile_lib = dlopen("libcufile.so", RTLD_LAZY);
        if (io->cufile_lib) {
            io->cuFileDriverOpen       = (pfn_cuFileDriverOpen)dlsym(io->cufile_lib, "cuFileDriverOpen");
            io->cuFileDriverClose      = (pfn_cuFileDriverClose)dlsym(io->cufile_lib, "cuFileDriverClose");
            io->cuFileHandleRegister   = (pfn_cuFileHandleRegister)dlsym(io->cufile_lib, "cuFileHandleRegister");
            io->cuFileHandleDeregister = (pfn_cuFileHandleDeregister)dlsym(io->cufile_lib, "cuFileHandleDeregister");
            io->cuFileBufRegister      = (pfn_cuFileBufRegister)dlsym(io->cufile_lib, "cuFileBufRegister");
            io->cuFileBufDeregister    = (pfn_cuFileBufDeregister)dlsym(io->cufile_lib, "cuFileBufDeregister");
            io->cuFileRead             = (pfn_cuFileRead)dlsym(io->cufile_lib, "cuFileRead");
            if (io->cuFileDriverOpen && io->cuFileHandleRegister && io->cuFileRead) {
                TnCUfileError e = io->cuFileDriverOpen();
                if (e.err == 0) {
                    io->gds_ready = 1;
                    io->backend_id = TN_IO_BACKEND_LINUX_GDS;
                }
            }
        }
    }
    // Tier 1: Initialize raw Linux `io_uring` ring (ZERO external liburing!) for batched NVMe->PinnedHost async reads
    // Supports optional `IORING_SETUP_SQPOLL` (`TENON_IO_URING_SQPOLL=1`) with seamless fallback to standard io_uring.
#  if defined(TENON_HAVE_IO_URING_HDR) && defined(__NR_io_uring_setup) && defined(__NR_io_uring_enter)
    const char* ur_env = getenv("TENON_DISABLE_IO_URING");
    if (!ur_env || atoi(ur_env) == 0) {
        struct io_uring_params p;
        memset(&p, 0, sizeof(p));
        int ufd = -1;
#    ifdef IORING_SETUP_SQPOLL
        const char* sqp_env = getenv("TENON_IO_URING_SQPOLL");
        if (sqp_env && atoi(sqp_env) > 0) {
            p.flags = IORING_SETUP_SQPOLL;
            p.sq_thread_idle = 2000u; // 2ms kernel poller idle timeout
            ufd = (int)syscall(__NR_io_uring_setup, 128u, &p);
            if (ufd >= 0) io->sqpoll_active = 1;
        }
#    endif
        if (ufd < 0) {
            memset(&p, 0, sizeof(p));
            ufd = (int)syscall(__NR_io_uring_setup, 128u, &p);
        }
        if (ufd >= 0) {
            size_t ssz = p.sq_off.array + (size_t)p.sq_entries * sizeof(uint32_t);
            size_t csz = p.cq_off.cqes  + (size_t)p.cq_entries * sizeof(struct io_uring_cqe);
            size_t esz = (size_t)p.sq_entries * sizeof(struct io_uring_sqe);
            void* sqr = mmap(NULL, ssz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ufd, IORING_OFF_SQ_RING);
            void* cqr = mmap(NULL, csz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ufd, IORING_OFF_CQ_RING);
            void* sqe = mmap(NULL, esz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ufd, IORING_OFF_SQES);
            if (sqr != MAP_FAILED && cqr != MAP_FAILED && sqe != MAP_FAILED) {
                io->uring_fd    = ufd;
                io->uring_ready = 1;
                io->sq_ring     = sqr; io->sq_ring_sz = ssz;
                io->cq_ring     = cqr; io->cq_ring_sz = csz;
                io->sqes        = (struct io_uring_sqe*)sqe; io->sqes_sz = esz;
                io->sq_head     = (uint32_t*)((uint8_t*)sqr + p.sq_off.head);
                io->sq_tail     = (uint32_t*)((uint8_t*)sqr + p.sq_off.tail);
                io->sq_mask     = (uint32_t*)((uint8_t*)sqr + p.sq_off.ring_mask);
                io->sq_flags    = (uint32_t*)((uint8_t*)sqr + p.sq_off.flags);
                io->sq_array    = (uint32_t*)((uint8_t*)sqr + p.sq_off.array);
                io->cq_head     = (uint32_t*)((uint8_t*)cqr + p.cq_off.head);
                io->cq_tail     = (uint32_t*)((uint8_t*)cqr + p.cq_off.tail);
                io->cq_mask     = (uint32_t*)((uint8_t*)cqr + p.cq_off.ring_mask);
                io->cqes        = (struct io_uring_cqe*)((uint8_t*)cqr + p.cq_off.cqes);
                if (io->backend_id != TN_IO_BACKEND_LINUX_GDS) {
                    io->backend_id = TN_IO_BACKEND_LINUX_IOURING;
                }
            } else {
                if (sqr != MAP_FAILED) munmap(sqr, ssz);
                if (cqr != MAP_FAILED) munmap(cqr, csz);
                if (sqe != MAP_FAILED) munmap(sqe, esz);
                close(ufd);
                io->sqpoll_active = 0;
            }
        }
    }
#  endif
    return 1;
#else
    io->backend_id = TN_IO_BACKEND_UNKNOWN;
    return 1;
#endif
}

// Register pinned host staging slabs for zero-pin `IORING_OP_READ_FIXED` (Linux io_uring) and tracking across all OSes.
static inline int tn_io_register_buffers(TnIoEngine* io, void* const* ptrs, const size_t* lens, int count) {
    if (!io || !ptrs || !lens || count <= 0) return 0;
    if (count > TN_IO_MAX_REG_BUFS) count = TN_IO_MAX_REG_BUFS;
    // Note: IORING_REGISTER_BUFFERS / IORING_UNREGISTER_BUFFERS are C enums (not #define macros) in <linux/io_uring.h> since Linux 5.1.
#if defined(__linux__) && defined(TENON_HAVE_IO_URING_HDR) && defined(__NR_io_uring_register)
    if (io->uring_ready) {
        while (__sync_lock_test_and_set(&io->uring_lock, 1)) {}
        if (io->reg_buf_count > 0) {
            (void)syscall(__NR_io_uring_register, io->uring_fd, IORING_UNREGISTER_BUFFERS, NULL, 0u);
            io->reg_buf_count = 0;
        }
        struct iovec iovs[TN_IO_MAX_REG_BUFS];
        int valid = 0;
        for (int i = 0; i < count; i++) {
            if (ptrs[i] && lens[i] > 0) {
                iovs[valid].iov_base = ptrs[i];
                iovs[valid].iov_len  = lens[i];
                io->reg_buf_ptrs[valid] = ptrs[i];
                io->reg_buf_lens[valid] = lens[i];
                valid++;
            }
        }
        int rc = (valid > 0) ? (int)syscall(__NR_io_uring_register, io->uring_fd, IORING_REGISTER_BUFFERS, iovs, (unsigned)valid) : -1;
        if (rc == 0) io->reg_buf_count = valid;
        __sync_lock_release(&io->uring_lock);
        return (rc == 0) ? valid : 0;
    }
#endif
    int valid = 0;
    for (int i = 0; i < count; i++) {
        if (ptrs[i] && lens[i] > 0) {
            io->reg_buf_ptrs[valid] = ptrs[i];
            io->reg_buf_lens[valid] = lens[i];
            valid++;
        }
    }
    io->reg_buf_count = valid;
    return valid;
}

// Register shard file descriptors for `IOSQE_FIXED_FILE` (eliminates per-IO fdget/fdput atomic refcounting on Linux).
static inline int tn_io_register_files(TnIoEngine* io, const int* fds, int count) {
    if (!io || !fds || count <= 0) return 0;
    if (count > TN_IO_MAX_REG_FILES) count = TN_IO_MAX_REG_FILES;
    // Note: IORING_REGISTER_FILES / IORING_UNREGISTER_FILES are C enums (not #define macros) in <linux/io_uring.h> since Linux 5.1.
#if defined(__linux__) && defined(TENON_HAVE_IO_URING_HDR) && defined(__NR_io_uring_register)
    if (io->uring_ready) {
        while (__sync_lock_test_and_set(&io->uring_lock, 1)) {}
        if (io->reg_file_count > 0) {
            (void)syscall(__NR_io_uring_register, io->uring_fd, IORING_UNREGISTER_FILES, NULL, 0u);
            io->reg_file_count = 0;
        }
        int copy_fds[TN_IO_MAX_REG_FILES];
        for (int i = 0; i < count; i++) copy_fds[i] = fds[i];
        int rc = (int)syscall(__NR_io_uring_register, io->uring_fd, IORING_REGISTER_FILES, copy_fds, (unsigned)count);
        if (rc == 0) {
            for (int i = 0; i < count; i++) io->reg_files[i] = copy_fds[i];
            io->reg_file_count = count;
        }
        __sync_lock_release(&io->uring_lock);
        return (rc == 0) ? count : 0;
    }
#endif
    (void)fds;
    return 0;
}

// Register a persistent GPU VRAM slab with NVIDIA GPUDirect Storage (`cuFileBufRegister`) for zero-overhead DMA.
static inline int tn_io_gds_register_vram(TnIoEngine* io, void* gpu_dptr, size_t nbytes) {
#if defined(__linux__)
    if (!io || !io->gds_ready || !io->cuFileBufRegister || !gpu_dptr || nbytes == 0) return 0;
    while (__sync_lock_test_and_set(&io->gds_lock, 1)) {}
    for (int i = 0; i < io->gds_vbuf_count; i++) {
        if (io->gds_vbufs[i] == gpu_dptr) { __sync_lock_release(&io->gds_lock); return 1; }
    }
    int ok = 0;
    if (io->gds_vbuf_count < TN_IO_MAX_GDS_VBUFS) {
        if (io->cuFileBufRegister(gpu_dptr, nbytes, 0).err == 0) {
            io->gds_vbufs[io->gds_vbuf_count] = gpu_dptr;
            io->gds_vlens[io->gds_vbuf_count] = nbytes;
            io->gds_vbuf_count++;
            ok = 1;
        }
    }
    __sync_lock_release(&io->gds_lock);
    return ok;
#else
    (void)io; (void)gpu_dptr; (void)nbytes;
    return 0;
#endif
}

static inline void tn_io_gds_deregister_vram(TnIoEngine* io, void* gpu_dptr) {
#if defined(__linux__)
    if (!io || !io->gds_ready || !io->cuFileBufDeregister || !gpu_dptr) return;
    while (__sync_lock_test_and_set(&io->gds_lock, 1)) {}
    for (int i = 0; i < io->gds_vbuf_count; i++) {
        if (io->gds_vbufs[i] == gpu_dptr) {
            (void)io->cuFileBufDeregister(gpu_dptr);
            io->gds_vbufs[i] = io->gds_vbufs[io->gds_vbuf_count - 1];
            io->gds_vlens[i] = io->gds_vlens[io->gds_vbuf_count - 1];
            io->gds_vbuf_count--;
            break;
        }
    }
    __sync_lock_release(&io->gds_lock);
#else
    (void)io; (void)gpu_dptr;
#endif
}

static inline void tn_io_destroy(TnIoEngine* io) {
    if (!io) return;
#if defined(__linux__) && defined(TENON_HAVE_IO_URING_HDR)
    if (io->uring_ready) {
#  if defined(__NR_io_uring_register)
        if (io->reg_buf_count > 0) (void)syscall(__NR_io_uring_register, io->uring_fd, IORING_UNREGISTER_BUFFERS, NULL, 0u);
        if (io->reg_file_count > 0) (void)syscall(__NR_io_uring_register, io->uring_fd, IORING_UNREGISTER_FILES, NULL, 0u);
#  endif
        if (io->sq_ring) munmap(io->sq_ring, io->sq_ring_sz);
        if (io->cq_ring) munmap(io->cq_ring, io->cq_ring_sz);
        if (io->sqes)    munmap(io->sqes, io->sqes_sz);
        if (io->uring_fd >= 0) close(io->uring_fd);
        io->uring_ready = 0;
        io->sqpoll_active = 0;
        io->reg_buf_count = 0;
        io->reg_file_count = 0;
    }
#endif
#if defined(__linux__)
    if (io->gds_ready) {
        if (io->cuFileBufDeregister) {
            for (int i = 0; i < io->gds_vbuf_count; i++) {
                if (io->gds_vbufs[i]) (void)io->cuFileBufDeregister(io->gds_vbufs[i]);
            }
        }
        if (io->cuFileHandleDeregister) {
            for (int i = 0; i < io->gds_handle_count; i++) {
                if (io->gds_handles[i]) io->cuFileHandleDeregister(io->gds_handles[i]);
            }
        }
        io->gds_vbuf_count = 0;
        io->gds_handle_count = 0;
        if (io->cuFileDriverClose) io->cuFileDriverClose();
        io->gds_ready = 0;
    }
    if (io->cufile_lib) {
        dlclose(io->cufile_lib);
        io->cufile_lib = NULL;
    }
#endif
#if defined(_WIN32)
    if (io->iocp) { CloseHandle(io->iocp); io->iocp = NULL; }
    if (io->iocp_cs_ready) { DeleteCriticalSection(&io->iocp_cs); io->iocp_cs_ready = 0; }
    io->iocp_leader_active = 0;
    io->iocp_file_count = 0;
#endif
}

// Non-blocking OS NVMe lookahead hint: issues async kernel prefetch for layer L+1 WHILE GPU computes layer L!
static inline void tn_io_advise_willneed(TnIoEngine* io, int fd, uint64_t off, size_t len) {
    if (len == 0) return;
    if (io) TN_IO_ATOMIC_ADD64(&io->advise_calls, 1);
#if defined(__APPLE__)
    if (fd >= 0) {
        struct radvisory ra;
        ra.ra_offset = (off_t)off;
        ra.ra_count  = (len > 0x7FFFFFFFu) ? 0x7FFFFFFF : (int)len;
        (void)fcntl(fd, F_RDADVISE, &ra);
    }
#elif defined(__linux__)
    if (fd >= 0) {
        (void)posix_fadvise(fd, (off_t)off, (off_t)len, POSIX_FADV_WILLNEED);
    }
#else
    (void)fd; (void)off;
#endif
}

static inline void tn_io_advise_dontneed(int fd, uint64_t off, size_t len) {
#if defined(__linux__)
    if (fd >= 0 && len > 0) {
        (void)posix_fadvise(fd, (off_t)off, (off_t)len, POSIX_FADV_DONTNEED);
    }
#else
    (void)fd; (void)off; (void)len;
#endif
}

// Cross-OS memory-mapped region prefetch (`madvise(MADV_WILLNEED)` on POSIX, `PrefetchVirtualMemory` on Windows 8+/11)
static inline void tn_io_prefetch_vmem(TnIoEngine* io, void* addr, size_t len) {
    if (!addr || len == 0) return;
    if (io) TN_IO_ATOMIC_ADD64(&io->advise_calls, 1);
#if defined(_WIN32)
    typedef struct { PVOID VirtualAddress; SIZE_T NumberOfBytes; } TnWinMemRange;
    typedef BOOL (WINAPI *pfn_PrefetchVirtualMemory)(HANDLE, ULONG_PTR, TnWinMemRange*, ULONG);
    static pfn_PrefetchVirtualMemory s_pvm = NULL;
    static int s_pvm_checked = 0;
    if (!s_pvm_checked) {
        HMODULE k32 = GetModuleHandleA("kernel32.dll");
        if (k32) s_pvm = (pfn_PrefetchVirtualMemory)GetProcAddress(k32, "PrefetchVirtualMemory");
        s_pvm_checked = 1;
    }
    if (s_pvm) {
        TnWinMemRange rng = { addr, (SIZE_T)len };
        (void)s_pvm(GetCurrentProcess(), 1, &rng, 0);
    }
#elif defined(__linux__) || defined(__APPLE__)
    (void)madvise(addr, len, MADV_WILLNEED);
#endif
}

#if defined(__linux__) && defined(TENON_HAVE_IO_URING_HDR)
static inline int tn_io_find_reg_buf(const TnIoEngine* io, const void* dst, size_t len) {
    uintptr_t p = (uintptr_t)dst;
    for (int i = 0; i < io->reg_buf_count; i++) {
        uintptr_t base = (uintptr_t)io->reg_buf_ptrs[i];
        if (p >= base && (p - base) + len <= io->reg_buf_lens[i]) return i;
    }
    return -1;
}
static inline int tn_io_find_reg_file(const TnIoEngine* io, int fd) {
    for (int i = 0; i < io->reg_file_count; i++) {
        if (io->reg_files[i] == fd) return i;
    }
    return -1;
}
#endif

#if defined(_WIN32)
typedef struct {
    OVERLAPPED       ov;          // MUST be first field so (TnWinIoOp*)entry.lpOverlapped is exact!
    volatile LONG*   rem;         // Owning batch's remaining-inflight refcount
    volatile LONG64* batch_bytes; // Owning batch's transferred bytes accumulator
} TnWinIoOp;

static inline HANDLE tn_io_win_open_overlapped(const char* path, int use_nocache) {
    if (!path) return INVALID_HANDLE_VALUE;
    DWORD flags = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED | FILE_FLAG_SEQUENTIAL_SCAN;
    if (use_nocache) flags |= FILE_FLAG_NO_BUFFERING;
    return CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, flags, NULL);
}
static inline HANDLE tn_io_win_req_handle(const TnIoReq* r) {
    if (!r) return INVALID_HANDLE_VALUE;
    if (r->hfile && r->hfile != INVALID_HANDLE_VALUE) return r->hfile;
    if (r->fd >= 0) {
        intptr_t osfh = _get_osfhandle(r->fd);
        if (osfh != -1) return (HANDLE)osfh;
    }
    return INVALID_HANDLE_VALUE;
}
static inline int tn_io_win_ensure_iocp_file(TnIoEngine* io, HANDLE hfile) {
    if (!io || !io->iocp || !io->iocp_cs_ready || hfile == INVALID_HANDLE_VALUE || !hfile) return 0;
    EnterCriticalSection(&io->iocp_cs);
    for (int i = 0; i < io->iocp_file_count; i++) {
        if (io->iocp_files[i] == hfile) { LeaveCriticalSection(&io->iocp_cs); return 1; }
    }
    int ok = 0;
    if (io->iocp_file_count < TN_IO_MAX_IOCP_FILES) {
        if (CreateIoCompletionPort(hfile, io->iocp, (ULONG_PTR)hfile, 0) != NULL) {
            io->iocp_files[io->iocp_file_count++] = hfile;
            ok = 1;
        }
    }
    LeaveCriticalSection(&io->iocp_cs);
    return ok;
}
#endif

// Batched multi-chunk asynchronous read:
//   - Linux `io_uring`: 1 syscall (0 syscalls in SQPOLL!) + auto `IORING_OP_READ_FIXED` & `IOSQE_FIXED_FILE`
//   - Windows 11 `IOCP`: zero-event-alloc `ReadFile` + single-call `GetQueuedCompletionStatusEx` batch reaping
//   - macOS / fallback: `pread` + optional `POSIX_FADV_DONTNEED`
static inline size_t tn_io_read_batch(TnIoEngine* io, const TnIoReq* reqs, int n_reqs) {
    if (!reqs || n_reqs <= 0) return 0;
    size_t total = 0;
    if (io) TN_IO_ATOMIC_ADD64(&io->total_batches, 1);

#if defined(__linux__) && defined(TENON_HAVE_IO_URING_HDR) && defined(__NR_io_uring_enter)
    if (io && io->uring_ready && n_reqs <= TN_IO_MAX_BATCH) {
        while (__sync_lock_test_and_set(&io->uring_lock, 1)) {}
        uint32_t mask = *io->sq_mask;
        uint32_t tail = *io->sq_tail;
        int submitted = 0;
        int fixed_hits = 0;
        for (int i = 0; i < n_reqs; i++) {
            if (reqs[i].len == 0 || reqs[i].fd < 0 || !reqs[i].dst) continue;
            uint32_t idx = tail & mask;
            struct io_uring_sqe* sqe = &io->sqes[idx];
            memset(sqe, 0, sizeof(*sqe));
            int bidx = (io->reg_buf_count > 0) ? tn_io_find_reg_buf(io, reqs[i].dst, reqs[i].len) : -1;
            int fidx = (io->reg_file_count > 0) ? tn_io_find_reg_file(io, reqs[i].fd) : -1;
            // Note: IORING_OP_READ_FIXED is a C enum in <linux/io_uring.h> (not a #define macro).
            if (bidx >= 0) {
                sqe->opcode    = IORING_OP_READ_FIXED;
                sqe->buf_index = (uint16_t)bidx;
                fixed_hits++;
            } else {
                sqe->opcode    = IORING_OP_READ;
            }
#  ifdef IOSQE_FIXED_FILE
            if (fidx >= 0) {
                sqe->flags |= IOSQE_FIXED_FILE;
                sqe->fd     = fidx;
            } else
#  endif
            {
                sqe->fd     = reqs[i].fd;
            }
            sqe->off       = reqs[i].off;
            sqe->addr      = (uint64_t)(uintptr_t)reqs[i].dst;
            sqe->len       = (uint32_t)(reqs[i].len > 0x7FFFFFFFu ? 0x7FFFFFFFu : reqs[i].len);
            sqe->user_data = (uint64_t)i;
            io->sq_array[idx] = idx;
            tail++;
            submitted++;
        }
        int uring_ok = 0;
        if (submitted > 0) {
            __atomic_store_n(io->sq_tail, tail, __ATOMIC_RELEASE);
            unsigned enter_flags = IORING_ENTER_GETEVENTS;
            unsigned to_submit = (unsigned)submitted;
#  if defined(IORING_SQ_NEED_WAKEUP) && defined(IORING_ENTER_SQ_WAKEUP)
            if (io->sqpoll_active) {
                to_submit = 0u;
                if (__atomic_load_n(io->sq_flags, __ATOMIC_ACQUIRE) & IORING_SQ_NEED_WAKEUP) {
                    enter_flags |= IORING_ENTER_SQ_WAKEUP;
                }
            }
#  endif
            int ret = (int)syscall(__NR_io_uring_enter, io->uring_fd, to_submit, (uint32_t)submitted,
                                   enter_flags, NULL, (size_t)0);
            if (ret >= 0) {
                uint32_t cmask = *io->cq_mask;
                uint32_t chead = *io->cq_head;
                int reaped = 0;
                while (reaped < submitted) {
                    uint32_t ctail = __atomic_load_n(io->cq_tail, __ATOMIC_ACQUIRE);
                    if (chead == ctail) {
                        (void)syscall(__NR_io_uring_enter, io->uring_fd, 0u, (uint32_t)(submitted - reaped),
                                      IORING_ENTER_GETEVENTS, NULL, (size_t)0);
                        continue;
                    }
                    struct io_uring_cqe* cqe = &io->cqes[chead & cmask];
                    int ridx = (int)cqe->user_data;
                    int32_t res = cqe->res;
                    chead++;
                    reaped++;
                    if (res > 0) {
                        total += (size_t)res;
                        if (ridx >= 0 && ridx < n_reqs && (size_t)res < reqs[ridx].len) {
                            uint8_t* p = (uint8_t*)reqs[ridx].dst + (size_t)res;
                            size_t rem = reqs[ridx].len - (size_t)res;
                            uint64_t roff = reqs[ridx].off + (uint64_t)res;
                            while (rem > 0) {
                                ssize_t r2 = pread(reqs[ridx].fd, p, rem, (off_t)roff);
                                if (r2 <= 0) { if (r2 < 0 && errno == EINTR) continue; break; }
                                p += (size_t)r2; roff += (uint64_t)r2; rem -= (size_t)r2; total += (size_t)r2;
                            }
                        }
                    } else if (ridx >= 0 && ridx < n_reqs) {
                        uint8_t* p = (uint8_t*)reqs[ridx].dst;
                        size_t rem = reqs[ridx].len;
                        uint64_t roff = reqs[ridx].off;
                        while (rem > 0) {
                            ssize_t r2 = pread(reqs[ridx].fd, p, rem, (off_t)roff);
                            if (r2 <= 0) { if (r2 < 0 && errno == EINTR) continue; break; }
                            p += (size_t)r2; roff += (uint64_t)r2; rem -= (size_t)r2; total += (size_t)r2;
                        }
                    }
                }
                __atomic_store_n(io->cq_head, chead, __ATOMIC_RELEASE);
                __sync_fetch_and_add(&io->uring_sqes, (int64_t)submitted);
                if (fixed_hits > 0) __sync_fetch_and_add(&io->fixed_buf_reads, (int64_t)fixed_hits);
                __sync_fetch_and_add(&io->total_bytes, (int64_t)total);
                uring_ok = 1;
            }
        }
        __sync_lock_release(&io->uring_lock);
        if (uring_ok) {
            if (io->use_nocache) {
                for (int i = 0; i < n_reqs; i++) tn_io_advise_dontneed(reqs[i].fd, reqs[i].off, reqs[i].len);
            }
            return total;
        }
    }
#endif

#if defined(_WIN32)
    int lim = n_reqs < TN_IO_MAX_BATCH ? n_reqs : TN_IO_MAX_BATCH;
    HANDLE handles[TN_IO_MAX_BATCH];
    int use_iocp[TN_IO_MAX_BATCH];
    int n_iocp_reqs = 0;
    int n_ev_reqs = 0;
    int can_iocp = (io && io->iocp != NULL && io->iocp_cs_ready) ? 1 : 0;
    for (int i = 0; i < lim; i++) {
        handles[i] = tn_io_win_req_handle(&reqs[i]);
        use_iocp[i] = 0;
        if (reqs[i].len == 0 || !reqs[i].dst || handles[i] == INVALID_HANDLE_VALUE) continue;
        if (can_iocp && tn_io_win_ensure_iocp_file(io, handles[i])) {
            use_iocp[i] = 1;
            n_iocp_reqs++;
        } else {
            n_ev_reqs++;
        }
    }

    // Path A (IOCP-associated handles): ALWAYS submit `TnWinIoOp` matched via `entries[k].lpOverlapped`.
    // Submitter guard `rem = 1` keeps `ops[]` alive on this stack until every accepted ReadFile is reaped.
    TnWinIoOp ops[TN_IO_MAX_BATCH];
    volatile LONG rem = 1;
    volatile LONG64 iocp_bytes = 0;
    if (n_iocp_reqs > 0) {
        for (int i = 0; i < lim; i++) {
            if (!use_iocp[i]) continue;
            memset(&ops[i], 0, sizeof(TnWinIoOp));
            ops[i].ov.Offset     = (DWORD)(reqs[i].off & 0xFFFFFFFFull);
            ops[i].ov.OffsetHigh = (DWORD)(reqs[i].off >> 32);
            ops[i].rem           = &rem;
            ops[i].batch_bytes   = &iocp_bytes;
            InterlockedIncrement(&rem);
            DWORD got = 0;
            BOOL ok = ReadFile(handles[i], reqs[i].dst, (DWORD)reqs[i].len, &got, &ops[i].ov);
            if (!ok && GetLastError() != ERROR_IO_PENDING) {
                InterlockedDecrement(&rem);
            }
        }
    }
    InterlockedDecrement(&rem);

    // Path B (Non-IOCP handles): submit OVERLAPPED event reads with bit 0 of `hEvent` SET (`| 1u`),
    // which per Microsoft Win32 docs strictly suppresses IOCP completion posting even if a handle was
    // externally associated with an IOCP — guaranteeing ZERO non-`TnWinIoOp` pointers ever enter `io->iocp`.
    OVERLAPPED ev_ovs[TN_IO_MAX_BATCH];
    HANDLE raw_evs[TN_IO_MAX_BATCH];
    int ev_inflight = 0;
    if (n_ev_reqs > 0) {
        memset(ev_ovs, 0, sizeof(ev_ovs));
        memset(raw_evs, 0, sizeof(raw_evs));
        for (int i = 0; i < lim; i++) {
            if (use_iocp[i] || reqs[i].len == 0 || !reqs[i].dst || handles[i] == INVALID_HANDLE_VALUE) continue;
            HANDLE ev = CreateEventA(NULL, TRUE, FALSE, NULL);
            if (!ev) continue;
            ev_ovs[i].Offset     = (DWORD)(reqs[i].off & 0xFFFFFFFFull);
            ev_ovs[i].OffsetHigh = (DWORD)(reqs[i].off >> 32);
            ev_ovs[i].hEvent     = (HANDLE)((uintptr_t)ev | (uintptr_t)1u);
            DWORD got = 0;
            if (ReadFile(handles[i], reqs[i].dst, (DWORD)reqs[i].len, &got, &ev_ovs[i])) {
                total += (size_t)got;
                CloseHandle(ev);
            } else if (GetLastError() == ERROR_IO_PENDING) {
                raw_evs[i] = ev;
                ev_inflight++;
            } else {
                CloseHandle(ev);
            }
        }
    }

    // Reap all Path A IOCP completions for this batch (Leader-Follower, lpOverlapped-demuxed)
    if (n_iocp_reqs > 0) {
        OVERLAPPED_ENTRY entries[TN_IO_MAX_BATCH];
        EnterCriticalSection(&io->iocp_cs);
        while (InterlockedCompareExchange(&rem, 0, 0) > 0) {
            if (!io->iocp_leader_active) {
                io->iocp_leader_active = 1;
                LeaveCriticalSection(&io->iocp_cs);
                ULONG got_n = 0;
                BOOL ok_g = GetQueuedCompletionStatusEx(io->iocp, entries, TN_IO_MAX_BATCH, &got_n, INFINITE, FALSE);
                EnterCriticalSection(&io->iocp_cs);
                if (ok_g) {
                    for (ULONG k = 0; k < got_n; k++) {
                        TnWinIoOp* op = (TnWinIoOp*)entries[k].lpOverlapped;
                        if (!op) continue;
                        if (op->batch_bytes) {
                            InterlockedExchangeAdd64(op->batch_bytes, (LONG64)entries[k].dwNumberOfBytesTransferred);
                        }
                        if (op->rem) InterlockedDecrement(op->rem);
                    }
                }
                io->iocp_leader_active = 0;
                WakeAllConditionVariable(&io->iocp_cv);
                if (!ok_g) break;
            } else {
                SleepConditionVariableCS(&io->iocp_cv, &io->iocp_cs, INFINITE);
            }
        }
        LeaveCriticalSection(&io->iocp_cs);
        total += (size_t)InterlockedCompareExchange64(&iocp_bytes, 0, 0);
        TN_IO_ATOMIC_ADD64(&io->iocp_batches, 1);
    }

    // Reap any pending Path B event reads (wait on raw unmasked HANDLE, then GetOverlappedResult + CloseHandle)
    for (int i = 0; i < lim && ev_inflight > 0; i++) {
        if (raw_evs[i]) {
            WaitForSingleObject(raw_evs[i], INFINITE);
            DWORD got = 0;
            if (GetOverlappedResult(handles[i], &ev_ovs[i], &got, FALSE)) total += (size_t)got;
            CloseHandle(raw_evs[i]);
            raw_evs[i] = NULL;
            ev_inflight--;
        }
    }
    if (io) TN_IO_ATOMIC_ADD64(&io->total_bytes, (int64_t)total);
    return total;
#else
    for (int i = 0; i < n_reqs; i++) {
        uint8_t* p = (uint8_t*)reqs[i].dst;
        size_t rem = reqs[i].len;
        uint64_t roff = reqs[i].off;
        while (rem > 0) {
            ssize_t r = pread(reqs[i].fd, p, rem, (off_t)roff);
            if (r <= 0) { if (r < 0 && errno == EINTR) continue; break; }
            p += (size_t)r; roff += (uint64_t)r; rem -= (size_t)r; total += (size_t)r;
        }
        if (io && io->use_nocache) tn_io_advise_dontneed(reqs[i].fd, reqs[i].off, reqs[i].len);
    }
    if (io) TN_IO_ATOMIC_ADD64(&io->total_bytes, (int64_t)total);
    return total;
#endif
}

#if defined(__linux__)
static inline TnCUfileHandle tn_io_gds_get_handle(TnIoEngine* io, int fd) {
    if (!io || !io->gds_ready || !io->cuFileHandleRegister || fd < 0) return NULL;
    while (__sync_lock_test_and_set(&io->gds_lock, 1)) {}
    for (int i = 0; i < io->gds_handle_count; i++) {
        if (io->gds_fds[i] == fd) {
            TnCUfileHandle h = io->gds_handles[i];
            __sync_lock_release(&io->gds_lock);
            return h;
        }
    }
    TnCUfileHandle cfh = NULL;
    if (io->gds_handle_count < TN_IO_MAX_GDS_FILES) {
        TnCUfileDescr desc;
        memset(&desc, 0, sizeof(desc));
        desc.type = TN_CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        desc.handle.fd = fd;
        if (io->cuFileHandleRegister(&cfh, &desc).err == 0 && cfh) {
            io->gds_fds[io->gds_handle_count] = fd;
            io->gds_handles[io->gds_handle_count] = cfh;
            io->gds_handle_count++;
        } else {
            cfh = NULL;
        }
    }
    __sync_lock_release(&io->gds_lock);
    return cfh;
}
#endif

// Persistent-handle Linux GPUDirect Storage direct-to-VRAM read (returns bytes read > 0 on GDS hit, 0 to fall back to io_uring+DMA)
static inline size_t tn_io_gds_read_vram(TnIoEngine* io, int fd, void* gpu_dptr, size_t nbytes, uint64_t off) {
#if defined(__linux__)
    if (!io || !io->gds_ready || !io->cuFileRead || fd < 0 || !gpu_dptr || nbytes == 0) return 0;
    TnCUfileHandle cfh = tn_io_gds_get_handle(io, fd);
    if (!cfh) return 0;
    ssize_t r = io->cuFileRead(cfh, gpu_dptr, nbytes, (off_t)off, 0);
    if (r > 0) {
        __sync_fetch_and_add(&io->gds_bytes, (int64_t)r);
        __sync_fetch_and_add(&io->total_bytes, (int64_t)r);
        return (size_t)r;
    }
    return 0;
#else
    (void)io; (void)fd; (void)gpu_dptr; (void)nbytes; (void)off;
    return 0;
#endif
}

// Batched multi-slice NVMe -> GPU VRAM GPUDirect Storage read (for zero-CPU-copy Gate+Up+Down offloaded MoE experts!)
static inline size_t tn_io_gds_read_vram_batch(TnIoEngine* io, const TnIoGdsReq* reqs, int n_reqs) {
#if defined(__linux__)
    if (!io || !io->gds_ready || !io->cuFileRead || !reqs || n_reqs <= 0) return 0;
    size_t tot = 0;
    for (int i = 0; i < n_reqs; i++) {
        if (reqs[i].len == 0 || !reqs[i].gpu_dptr || reqs[i].fd < 0) continue;
        TnCUfileHandle cfh = tn_io_gds_get_handle(io, reqs[i].fd);
        if (!cfh) return 0;
        ssize_t r = io->cuFileRead(cfh, reqs[i].gpu_dptr, reqs[i].len, (off_t)reqs[i].file_off, (off_t)reqs[i].vram_off);
        if (r != (ssize_t)reqs[i].len) return 0;
        tot += (size_t)r;
    }
    if (tot > 0) {
        __sync_fetch_and_add(&io->gds_bytes, (int64_t)tot);
        __sync_fetch_and_add(&io->total_bytes, (int64_t)tot);
    }
    return tot;
#else
    (void)io; (void)reqs; (void)n_reqs;
    return 0;
#endif
}

// Portable byte-exact IO HAL self-test (tests init, buffer registration, advise, multi-chunk read_batch, destroy)
static inline int tn_io_self_test(void) {
    TnIoEngine io;
    if (!tn_io_init(&io, 0)) return 0;
    const size_t N = 65536u;
    uint8_t* ref = (uint8_t*)malloc(N);
    uint8_t* dst = (uint8_t*)calloc(1, N);
    if (!ref || !dst) { free(ref); free(dst); tn_io_destroy(&io); return 0; }
    for (size_t i = 0; i < N; i++) ref[i] = (uint8_t)((i * 131u + 17u) & 0xFFu);
    void* ptrs[1] = { dst };
    size_t lens[1] = { N };
    (void)tn_io_register_buffers(&io, ptrs, lens, 1);
    int ok = 0;
#if defined(_WIN32)
    char tmp_path[MAX_PATH];
    if (GetTempPathA(MAX_PATH, tmp_path) > 0) {
        char fname[MAX_PATH];
        snprintf(fname, sizeof(fname), "%stenon_io_hal_test.bin", tmp_path);
        HANDLE hw = CreateFileA(fname, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hw != INVALID_HANDLE_VALUE) {
            DWORD wr = 0;
            WriteFile(hw, ref, (DWORD)N, &wr, NULL);
            CloseHandle(hw);
            HANDLE hr = tn_io_win_open_overlapped(fname, 0);
            if (hr != INVALID_HANDLE_VALUE) {
                TnIoReq reqs[4];
                memset(reqs, 0, sizeof(reqs));
                for (int i = 0; i < 4; i++) {
                    reqs[i].hfile = hr;
                    reqs[i].dst   = dst + (size_t)i * 16384u;
                    reqs[i].len   = 16384u;
                    reqs[i].off   = (uint64_t)i * 16384u;
                }
                size_t got = tn_io_read_batch(&io, reqs, 4);
                CloseHandle(hr);
                if (got == N && memcmp(ref, dst, N) == 0 && io.iocp_batches > 0) ok = 1;
            }
            DeleteFileA(fname);
        }
    }
#else
    char fname[128];
    snprintf(fname, sizeof(fname), "/tmp/tenon_io_hal_test_%d.bin", (int)getpid());
    int wfd = open(fname, O_CREAT | O_TRUNC | O_WRONLY, 0600);
    if (wfd >= 0) {
        ssize_t wr = write(wfd, ref, N);
        close(wfd);
        if (wr == (ssize_t)N) {
            int rfd = open(fname, O_RDONLY);
            if (rfd >= 0) {
                (void)tn_io_register_files(&io, &rfd, 1);
                tn_io_advise_willneed(&io, rfd, 0, N);
                TnIoReq reqs[4];
                memset(reqs, 0, sizeof(reqs));
                for (int i = 0; i < 4; i++) {
                    reqs[i].fd  = rfd;
                    reqs[i].dst = dst + (size_t)i * 16384u;
                    reqs[i].len = 16384u;
                    reqs[i].off = (uint64_t)i * 16384u;
                }
                size_t got = tn_io_read_batch(&io, reqs, 4);
                close(rfd);
                if (got == N && memcmp(ref, dst, N) == 0) ok = 1;
            }
        }
        unlink(fname);
    }
#endif
    free(ref);
    free(dst);
    tn_io_destroy(&io);
    return ok;
}

#endif // TENON_IO_HAL_H
