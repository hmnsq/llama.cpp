#define _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#define _USE_MATH_DEFINES // For M_PI on MSVC

#അന്തര്ഭാവ "ggml-backend-impl.h"
#അന്തര്ഭാവ "ggml-backend.h"
#അന്തര്ഭാവ "traits.h"
#അന്തര്ഭാവ "ggml-cpu-impl.h"
#അന്തര്ഭാവ "ggml-impl.h"
#അന്തര്ഭാവ "quants.h"
#അന്തര്ഭാവ "ggml-threading.h"
#അന്തര്ഭാവ "unary-ops.h"
#അന്തര്ഭാവ "binary-ops.h"
#അന്തര്ഭാവ "vec.h"
#അന്തര്ഭാവ "ops.h"
#അന്തര്ഭാവ "ggml.h"
#അന്തര്ഭാവ "common.h"
#അന്തര്ഭാവ "tiled/tiled.h"

#എങ്കിൽ defined(_MSC_VER) || defined(__MINGW32__)
#അന്തര്ഭാവ <malloc.h> // using malloc.h with MSC/MINGW
#അല്ലെങ്കിലെങ്കിൽ !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#അന്തര്ഭാവ <alloca.h>
#endif

#അന്തര്ഭാവ <assert.h>
#അന്തര്ഭാവ <errno.h>
#അന്തര്ഭാവ <time.h>
#അന്തര്ഭാവ <math.h>
#അന്തര്ഭാവ <stdlib.h>
#അന്തര്ഭാവ <string.h>
#അന്തര്ഭാവ <stdint.h>
#അന്തര്ഭാവ <inttypes.h>
#അന്തര്ഭാവ <stdio.h>
#അന്തര്ഭാവ <float.h>
#അന്തര്ഭാവ <limits.h>
#അന്തര്ഭാവ <stdarg.h>
#അന്തര്ഭാവ <signal.h>
#എങ്കിൽ defined(__gnu_linux__)
#അന്തര്ഭാവ <syscall.h>
#endif

#ifdef GGML_USE_OPENMP
#അന്തര്ഭാവ <omp.h>
#endif

#എങ്കിൽ defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_MATMUL_INT8)
#undef GGML_USE_LLAMAFILE
#endif

#ifdef GGML_USE_LLAMAFILE
#അന്തര്ഭാവ "llamafile/sgemm.h"
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
#    include "spacemit/ime.h"
#endif

// Note: once we move threading into a separate C++ file
// will use std::hardware_destructive_interference_size instead of hardcoding it here
// and we'll use C++ attribute syntax.
#define GGML_CACHE_വരി  64

#എങ്കിൽ defined(__clang__) || defined(__GNUC__)
#define GGML_CACHE_ALIGN __attribute__((aligned(GGML_CACHE_വരി)))
#endif

#എങ്കിൽ defined(__has_feature)
#എങ്കിൽ __has_feature(thread_sanitizer)
#define GGML_TSAN_ENABLED 1
#endif
#അല്ലെങ്കിൽ  // __has_feature
#എങ്കിൽ defined(__SANITIZE_THREAD__)
#define GGML_TSAN_ENABLED 1
#endif
#endif // __has_feature

#define UNUSED GGML_UNUSED
#define SWAP(x, y, T) ചെയ്യുക { T SWAP = x; (x) = y; (y) = SWAP; } ആകുമ്പോൾ (0)

// precomputed f32 table for f16 (256 KB) (simd-mappings.h)
ദശാംശം ggml_table_f32_f16[1 << 16];

// precomputed f32 table for e8m0 half (1 KB) (simd-mappings.h)
ദശാംശം ggml_table_f32_e8m0_half[1 << 8];

// precomputed f32 table for ue4m3 (1 KB) (simd-mappings.h)
ദശാംശം ggml_table_f32_ue4m3[1 << 8];

#എങ്കിൽ defined(__ARM_ARCH)
ഘടന ggml_arm_arch_features_type {
    പൂർണ്ണസംഖ്യ sve_cnt;
} ggml_arm_arch_features = { 0 };
#endif

#എങ്കിൽ defined(__riscv)
ഘടന ggml_riscv_arch_features_type {
    പൂർണ്ണസംഖ്യ rvv_vlen;
} ggml_riscv_arch_features = { 0 };
#endif

#എങ്കിൽ defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#അന്തര്ഭാവ <windows.h>

#എങ്കിൽ defined(_MSC_VER) && !defined(__clang__)
#define GGML_CACHE_ALIGN __declspec(align(GGML_CACHE_വരി))

തരനിർവചനം അസ്ഥിര LONG atomic_int;
തരനിർവചനം atomic_int atomic_bool;
തരനിർവചനം atomic_int atomic_flag;

#define ATOMIC_FLAG_INIT 0

തരനിർവചനം എണ്ണൽ {
    memory_order_relaxed,
    memory_order_consume,
    memory_order_acquire,
    memory_order_release,
    memory_order_acq_rel,
    memory_order_seq_cst
} memory_order;

സ്ഥിതം ശൂന്യം atomic_store(atomic_int * ptr, LONG val) {
    InterlockedExchange(ptr, val);
}
സ്ഥിതം ശൂന്യം atomic_store_explicit(atomic_int * ptr, LONG val, memory_order mo) {
    // TODO: add support for explicit memory order
    InterlockedExchange(ptr, val);
}
സ്ഥിതം LONG atomic_load(atomic_int * ptr) {
    തിരികെ InterlockedCompareExchange(ptr, 0, 0);
}
സ്ഥിതം LONG atomic_load_explicit(atomic_int * ptr, memory_order mo) {
    // TODO: add support for explicit memory order
    തിരികെ InterlockedCompareExchange(ptr, 0, 0);
}
സ്ഥിതം LONG atomic_fetch_add(atomic_int * ptr, LONG inc) {
    തിരികെ InterlockedExchangeAdd(ptr, inc);
}
സ്ഥിതം LONG atomic_fetch_add_explicit(atomic_int * ptr, LONG inc, memory_order mo) {
    // TODO: add support for explicit memory order
    തിരികെ InterlockedExchangeAdd(ptr, inc);
}
സ്ഥിതം atomic_bool atomic_flag_test_and_set(atomic_flag * ptr) {
    തിരികെ InterlockedExchange(ptr, 1);
}
സ്ഥിതം ശൂന്യം atomic_flag_clear(atomic_flag * ptr) {
    InterlockedExchange(ptr, 0);
}
സ്ഥിതം ശൂന്യം atomic_thread_fence(memory_order mo) {
    MemoryBarrier();
}
#അല്ലെങ്കിൽ // clang
#അന്തര്ഭാവ <stdatomic.h>
#endif

തരനിർവചനം HANDLE pthread_t;

തരനിർവചനം DWORD thread_ret_t;
സ്ഥിതം പൂർണ്ണസംഖ്യ pthread_create(pthread_t * out, ശൂന്യം * unused, thread_ret_t(*func)(ശൂന്യം *), ശൂന്യം * arg) {
    (ശൂന്യം) unused;
    HANDLE handle = CreateThread(ശൂന്യ, 0, (LPTHREAD_START_ROUTINE) func, arg, 0, ശൂന്യ);
    എങ്കിൽ (handle == ശൂന്യ)
    {
        തിരികെ EAGAIN;
    }

    *out = handle;
    തിരികെ 0;
}

സ്ഥിതം പൂർണ്ണസംഖ്യ pthread_join(pthread_t thread, ശൂന്യം * unused) {
    (ശൂന്യം) unused;
    പൂർണ്ണസംഖ്യ ret = (പൂർണ്ണസംഖ്യ) WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    തിരികെ ret;
}

സ്ഥിതം പൂർണ്ണസംഖ്യ sched_yield (ശൂന്യം) {
    Sleep (0);
    തിരികെ 0;
}
#അല്ലെങ്കിൽ

#അന്തര്ഭാവ <pthread.h>
#അന്തര്ഭാവ <stdatomic.h>
#അന്തര്ഭാവ <sched.h>
#എങ്കിൽ defined(__FreeBSD__)
#അന്തര്ഭാവ <pthread_np.h>
#endif

തരനിർവചനം ശൂന്യം * thread_ret_t;

#അന്തര്ഭാവ <sys/types.h>
#അന്തര്ഭാവ <sys/stat.h>
#അന്തര്ഭാവ <unistd.h>

#endif

തരനിർവചനം pthread_t ggml_thread_t;

#define GGML_THREADPOOL_N_THREADS_MASK (0xffffU)
#define GGML_THREADPOOL_N_THREADS_BITS (16)

#എങ്കിൽ defined(__APPLE__)
#അന്തര്ഭാവ <unistd.h>
#അന്തര്ഭാവ <mach/mach.h>
#അന്തര്ഭാവ <TargetConditionals.h>
#endif

സ്ഥിതം സ്ഥിരം ഘടന ggml_type_traits_cpu type_traits_cpu[GGML_TYPE_എണ്ണം] = {
    [GGML_TYPE_F32] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_fp32,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_f32,
        .vec_dot_type             = GGML_TYPE_F32,
        .nrows                    = 1,
    },
    [GGML_TYPE_F16] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_fp16,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_f16,
        .vec_dot_type             = GGML_TYPE_F16,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q1_0] = {
        .from_float               = quantize_row_q1_0,
        .vec_dot                  = ggml_vec_dot_q1_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q2_0] = {
        .from_float               = quantize_row_q2_0,
        .vec_dot                  = ggml_vec_dot_q2_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q4_0] = {
        .from_float               = quantize_row_q4_0,
        .vec_dot                  = ggml_vec_dot_q4_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
#എങ്കിൽ defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#അല്ലെങ്കിൽ
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q4_1] = {
        .from_float               = quantize_row_q4_1,
        .vec_dot                  = ggml_vec_dot_q4_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
#എങ്കിൽ defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#അല്ലെങ്കിൽ
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q5_0] = {
        .from_float               = quantize_row_q5_0,
        .vec_dot                  = ggml_vec_dot_q5_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q5_1] = {
        .from_float               = quantize_row_q5_1,
        .vec_dot                  = ggml_vec_dot_q5_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q8_0] = {
        .from_float               = quantize_row_q8_0,
        .vec_dot                  = ggml_vec_dot_q8_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
#എങ്കിൽ defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#അല്ലെങ്കിൽ
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q8_1] = {
        .from_float               = quantize_row_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
        .nrows                    = 1,
    },
    [GGML_TYPE_MXFP4] = {
        .from_float               = quantize_row_mxfp4,
        .vec_dot                  = ggml_vec_dot_mxfp4_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_NVFP4] = {
        .from_float               = quantize_row_nvfp4,
        .vec_dot                  = ggml_vec_dot_nvfp4_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q2_K] = {
        .from_float               = quantize_row_q2_K,
        .vec_dot                  = ggml_vec_dot_q2_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q3_K] = {
        .from_float               = quantize_row_q3_K,
        .vec_dot                  = ggml_vec_dot_q3_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q4_K] = {
        .from_float               = quantize_row_q4_K,
        .vec_dot                  = ggml_vec_dot_q4_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
#എങ്കിൽ defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#അല്ലെങ്കിൽ
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q5_K] = {
        .from_float               = quantize_row_q5_K,
        .vec_dot                  = ggml_vec_dot_q5_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q6_K] = {
        .from_float               = quantize_row_q6_K,
        .vec_dot                  = ggml_vec_dot_q6_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
#എങ്കിൽ defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#അല്ലെങ്കിൽ
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_IQ2_XXS] = {
        .from_float               = ശൂന്യ,
        .vec_dot                  = ggml_vec_dot_iq2_xxs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_XS] = {
        .from_float               = ശൂന്യ,
        .vec_dot                  = ggml_vec_dot_iq2_xs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ3_XXS] = {
        // NOTE: from_float for iq3 and iq2_s was removed because these quants require initialization in ggml_quantize_init
        //.from_float               = quantize_row_iq3_xxs,
        .vec_dot                  = ggml_vec_dot_iq3_xxs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ3_S] = {
        //.from_float               = quantize_row_iq3_s,
        .vec_dot                  = ggml_vec_dot_iq3_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_S] = {
        //.from_float               = quantize_row_iq2_s,
        .vec_dot                  = ggml_vec_dot_iq2_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_S] = {
        .from_float               = ശൂന്യ,
        .vec_dot                  = ggml_vec_dot_iq1_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_M] = {
        .from_float               = ശൂന്യ,
        .vec_dot                  = ggml_vec_dot_iq1_m_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_NL] = {
        .from_float               = quantize_row_iq4_nl,
        .vec_dot                  = ggml_vec_dot_iq4_nl_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_XS] = {
        .from_float               = quantize_row_iq4_xs,
        .vec_dot                  = ggml_vec_dot_iq4_xs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q8_K] = {
        .from_float               = quantize_row_q8_K,
    },
    [GGML_TYPE_BF16] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_bf16,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_bf16,
        .vec_dot_type             = GGML_TYPE_BF16,
        .nrows                    = 1,
    },
    [GGML_TYPE_TQ1_0] = {
        .from_float               = quantize_row_tq1_0,
        .vec_dot                  = ggml_vec_dot_tq1_0_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_TQ2_0] = {
        .from_float               = quantize_row_tq2_0,
        .vec_dot                  = ggml_vec_dot_tq2_0_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_I32] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_i32,
    },
};

സ്ഥിരം ഘടന ggml_type_traits_cpu * ggml_get_type_traits_cpu(എണ്ണൽ ggml_type type) {
    തിരികെ &type_traits_cpu[type];
}

//
// Threading defs
//

തരനിർവചനം pthread_t          ggml_thread_t;

#എങ്കിൽ defined(_WIN32)

തരനിർവചനം CONDITION_VARIABLE ggml_cond_t;
തരനിർവചനം SRWLOCK            ggml_mutex_t;

#define ggml_mutex_init(m)   InitializeSRWLock(m)
#define ggml_mutex_destroy(m)
#define ggml_mutex_lock(m)   AcquireSRWLockExclusive(m)
#define ggml_mutex_unlock(m) ReleaseSRWLockExclusive(m)
#define ggml_mutex_lock_shared(m)   AcquireSRWLockShared(m)
#define ggml_mutex_unlock_shared(m) ReleaseSRWLockShared(m)

#define ggml_cond_init(c)    InitializeConditionVariable(c)
#define ggml_cond_destroy(c)
#define ggml_cond_wait(c, m) SleepConditionVariableSRW(c, m, INFINITE, CONDITION_VARIABLE_LOCKMODE_SHARED)
#define ggml_cond_broadcast(c) WakeAllConditionVariable(c)

#define ggml_thread_create pthread_create
#define ggml_thread_join   pthread_join

#അല്ലെങ്കിൽ

തരനിർവചനം pthread_cond_t     ggml_cond_t;
തരനിർവചനം pthread_mutex_t    ggml_mutex_t;

#define ggml_mutex_init(m)          pthread_mutex_init(m, ശൂന്യ)
#define ggml_mutex_destroy(m)       pthread_mutex_destroy(m)
#define ggml_mutex_lock(m)          pthread_mutex_lock(m)
#define ggml_mutex_unlock(m)        pthread_mutex_unlock(m)
#define ggml_mutex_lock_shared(m)   pthread_mutex_lock(m)
#define ggml_mutex_unlock_shared(m) pthread_mutex_unlock(m)

#define ggml_lock_init(x)    UNUSED(x)
#define ggml_lock_destroy(x) UNUSED(x)
#എങ്കിൽ defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64) && !defined(_M_ARM64EC))
#define ggml_lock_lock(x)    _mm_pause()
#അല്ലെങ്കിൽ
#define ggml_lock_lock(x)    UNUSED(x)
#endif
#define ggml_lock_unlock(x)  UNUSED(x)

#define GGML_LOCK_INITIALIZER 0
#define ggml_cond_init(c)      pthread_cond_init(c, ശൂന്യ)
#define ggml_cond_destroy(c)   pthread_cond_destroy(c)
#define ggml_cond_wait(c, m)   pthread_cond_wait(c, m)
#define ggml_cond_broadcast(c) pthread_cond_broadcast(c)

#define ggml_thread_create pthread_create
#define ggml_thread_join   pthread_join

#endif

// Threadpool def
ഘടന ggml_threadpool {
    ggml_mutex_t mutex;       // mutex for cond.var
    ggml_cond_t  cond;        // cond.var for waiting for new work

    ഘടന ggml_cgraph * cgraph;
    ഘടന ggml_cplan  * cplan;

    // synchronization primitives
    atomic_int n_graph;       // updated when there is work to be done (i.e each graph) holds graph and active thread counts.
    atomic_int GGML_CACHE_ALIGN n_barrier;
    atomic_int GGML_CACHE_ALIGN n_barrier_passed;
    atomic_int GGML_CACHE_ALIGN current_chunk; // currently processing chunk during Mat_Mul, shared between all the threads.

    // these are atomic as an annotation for thread-sanitizer
    atomic_bool stop;         // Used for stopping the threadpool altogether
    atomic_bool pause;        // Used for pausing the threadpool or individual threads
    atomic_int  abort;        // Used for aborting processing of a graph

    ഘടന ggml_compute_state * workers;   // per thread state
    പൂർണ്ണസംഖ്യ          n_threads;   // Number of threads in the pool
    int32_t      prio;        // Scheduling priority
    uint32_t     poll;        // Polling level (0 - no polling)

    എണ്ണൽ ggml_status ec;
};

// Per-thread state
ഘടന ggml_compute_state {
#ifndef GGML_USE_OPENMP
    ggml_thread_t thrd;
    പൂർണ്ണസംഖ്യ  last_graph;
    bool pending;
#endif
    bool cpumask[GGML_MAX_N_THREADS];
    ഘടന ggml_threadpool * threadpool;
    പൂർണ്ണസംഖ്യ ith;
};

// Helpers for polling loops
#എങ്കിൽ defined(__aarch64__) && ( defined(__clang__) || defined(__GNUC__) )
സ്ഥിതം വരിയിൽ ശൂന്യം ggml_thread_cpu_relax(ശൂന്യം) {
    __asm__ അസ്ഥിര("yield" ::: "memory");
}
#അല്ലെങ്കിലെങ്കിൽ defined(__x86_64__)
സ്ഥിതം വരിയിൽ ശൂന്യം ggml_thread_cpu_relax(ശൂന്യം) {
    _mm_pause();
}
#അല്ലെങ്കിലെങ്കിൽ defined(__riscv)
സ്ഥിതം വരിയിൽ ശൂന്യം ggml_thread_cpu_relax(ശൂന്യം) {
    #ifdef __riscv_zihintpause
        __asm__ __volatile__ ("pause");
    #അല്ലെങ്കിൽ
        /* Encoding of the pause instruction */
        __asm__ __volatile__ (".4byte 0x100000F");
    #endif
}
#അല്ലെങ്കിൽ
സ്ഥിതം വരിയിൽ ശൂന്യം ggml_thread_cpu_relax(ശൂന്യം) {;}
#endif

//
// NUMA support
//

#define GGML_NUMA_MAX_NODES 8
#define GGML_NUMA_MAX_CPUS 512

ഘടന ggml_numa_കണ്ണി {
    uint32_t cpus[GGML_NUMA_MAX_CPUS]; // hardware threads on this node
    uint32_t n_cpus;
};

ഘടന ggml_numa_nodes {
    എണ്ണൽ ggml_numa_strategy numa_strategy;
    ഘടന ggml_numa_കണ്ണി nodes[GGML_NUMA_MAX_NODES];
    uint32_t n_nodes;
    uint32_t total_cpus; // hardware threads on system
    uint32_t current_കണ്ണി; // node on which main process is execting
#എങ്കിൽ defined(__gnu_linux__)
    cpu_set_t cpuset; // cpuset from numactl
#അല്ലെങ്കിൽ
    uint32_t cpuset; // no NUMA support outside of Linux at this time. Use a portable datatype
#endif
};

//
// ggml state
//

ഘടന ggml_state {
    ഘടന ggml_numa_nodes numa;
};

സ്ഥിതം ഘടന ggml_state g_state = {0};

ശൂന്യം ggml_barrier(ഘടന ggml_threadpool * tp) {
    പൂർണ്ണസംഖ്യ n_threads = atomic_load_explicit(&tp->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK;
    എങ്കിൽ (n_threads == 1) {
        തിരികെ;
    }

#ifdef GGML_USE_OPENMP
    #pragma omp barrier
#അല്ലെങ്കിൽ
    പൂർണ്ണസംഖ്യ n_passed = atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed);

    // enter barrier (full seq-cst fence)
    പൂർണ്ണസംഖ്യ n_barrier = atomic_fetch_add_explicit(&tp->n_barrier, 1, memory_order_seq_cst);

    എങ്കിൽ (n_barrier == (n_threads - 1)) {
        // last thread
        atomic_store_explicit(&tp->n_barrier, 0, memory_order_relaxed);

        // exit barrier (full seq-cst fence)
        atomic_fetch_add_explicit(&tp->n_barrier_passed, 1, memory_order_seq_cst);
        തിരികെ;
    }

    // wait for other threads
    ആകുമ്പോൾ (atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed) == n_passed) {
        ggml_thread_cpu_relax();
    }

    // exit barrier (full seq-cst fence)
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&tp->n_barrier_passed, 0, memory_order_seq_cst);
    #അല്ലെങ്കിൽ
    atomic_thread_fence(memory_order_seq_cst);
    #endif
#endif
}

ശൂന്യം ggml_threadpool_chunk_set(ഘടന ggml_threadpool * tp, പൂർണ്ണസംഖ്യ മൂല്യം) {
    atomic_store_explicit(&tp->current_chunk, മൂല്യം, memory_order_relaxed);
}

പൂർണ്ണസംഖ്യ ggml_threadpool_chunk_add(ഘടന ggml_threadpool * tp, പൂർണ്ണസംഖ്യ മൂല്യം) {
    തിരികെ atomic_fetch_add_explicit(&tp->current_chunk, മൂല്യം, memory_order_relaxed);
}

#എങ്കിൽ defined(__gnu_linux__)
സ്ഥിതം cpu_set_t ggml_get_numa_affinity(ശൂന്യം) {
    cpu_set_t cpuset;
    pthread_t thread;
    thread = pthread_self();
    CPU_ZERO(&cpuset);
    pthread_getaffinity_np(thread, വലുപ്പം(cpu_set_t), &cpuset);
    തിരികെ cpuset;
}
#അല്ലെങ്കിൽ
സ്ഥിതം uint32_t ggml_get_numa_affinity(ശൂന്യം) {
    തിരികെ 0; // no NUMA support
}
#endif

ശൂന്യം ggml_numa_init(എണ്ണൽ ggml_numa_strategy numa_flag) {
    എങ്കിൽ (g_state.numa.n_nodes > 0) {
        fprintf(stderr, "ggml_numa_init: NUMA already initialized\n");

        തിരികെ;
    }

#എങ്കിൽ defined(__gnu_linux__)
    ഘടന stat st;
    അക്ഷരം path[256];
    പൂർണ്ണസംഖ്യ rv;

    // set numa scheme
    g_state.numa.numa_strategy = numa_flag;

    GGML_PRINT_DEBUG("numa strategy %u\n",g_state.numa.numa_strategy);

    g_state.numa.cpuset = ggml_get_numa_affinity();

    // enumerate nodes
    ആകുമ്പോൾ (g_state.numa.n_nodes < GGML_NUMA_MAX_NODES) {
        rv = snprintf(path, വലുപ്പം(path), "/sys/devices/system/node/node%u", g_state.numa.n_nodes);
        GGML_ASSERT(rv > 0 && (unsigned)rv < വലുപ്പം(path));
        എങ്കിൽ (stat(path, &st) != 0) { നിർത്തുക; }
        ++g_state.numa.n_nodes;
    }

    // enumerate CPUs
    ആകുമ്പോൾ (g_state.numa.total_cpus < GGML_NUMA_MAX_CPUS) {
        rv = snprintf(path, വലുപ്പം(path), "/sys/devices/system/cpu/cpu%u", g_state.numa.total_cpus);
        GGML_ASSERT(rv > 0 && (unsigned)rv < വലുപ്പം(path));
        എങ്കിൽ (stat(path, &st) != 0) { നിർത്തുക; }
        ++g_state.numa.total_cpus;
    }

    GGML_PRINT_DEBUG("found %u numa nodes, %u CPUs\n", g_state.numa.n_nodes, g_state.numa.total_cpus);

    // figure out which node we're on
    uint current_cpu;
    പൂർണ്ണസംഖ്യ getcpu_ret = 0;
#എങ്കിൽ __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ > 33) || defined(__COSMOPOLITAN__)
    getcpu_ret = getcpu(&current_cpu, &g_state.numa.current_കണ്ണി);
#അല്ലെങ്കിൽ
    // old glibc doesn't have a wrapper for this call. Fall back on direct syscall
#   എങ്കിൽ !defined(SYS_getcpu) && defined(SYS_get_cpu)
#       define SYS_getcpu SYS_get_cpu // some older glibc versions use this name
#   endif
    getcpu_ret = syscall(SYS_getcpu, &current_cpu, &g_state.numa.current_കണ്ണി);
#endif

    എങ്കിൽ (g_state.numa.n_nodes < 1 || g_state.numa.total_cpus < 1 || getcpu_ret != 0) {
        g_state.numa.n_nodes = 0;
        തിരികെ;
    }

    GGML_PRINT_DEBUG("found our process on numa node %u, CPU %u\n", g_state.numa.current_കണ്ണി, current_cpu);

    വേണ്ടി (uint32_t n = 0; n < g_state.numa.n_nodes; ++n) {
        ഘടന ggml_numa_കണ്ണി * കണ്ണി = &g_state.numa.nodes[n];
        GGML_PRINT_DEBUG("CPUs on node %u:", n);
        കണ്ണി->n_cpus = 0;
        വേണ്ടി (uint32_t c = 0; c < g_state.numa.total_cpus; ++c) {
            rv = snprintf(path, വലുപ്പം(path), "/sys/devices/system/node/node%u/cpu%u", n, c);
            GGML_ASSERT(rv > 0 && (unsigned)rv < വലുപ്പം(path));
            എങ്കിൽ (stat(path, &st) == 0) {
                കണ്ണി->cpus[കണ്ണി->n_cpus++] = c;
                GGML_PRINT_DEBUG(" %u", c);
            }
        }
        GGML_PRINT_DEBUG("\n");
    }

    എങ്കിൽ (ggml_is_numa()) {
        FILE *fptr = fopen("/proc/sys/kernel/numa_balancing", "r");
        എങ്കിൽ (fptr != ശൂന്യ) {
            അക്ഷരം buf[42];
            എങ്കിൽ (fgets(buf, വലുപ്പം(buf), fptr) && strncmp(buf, "0\n", വലുപ്പം(buf)) != 0) {
                GGML_LOG_WARN("/proc/sys/kernel/numa_balancing is enabled, this has been observed to impair performance\n");
            }
            fclose(fptr);
        }
    }
#അല്ലെങ്കിൽ
    UNUSED(numa_flag);
    // TODO
#endif
}

bool ggml_is_numa(ശൂന്യം) {
    തിരികെ g_state.numa.n_nodes > 1;
}

#എങ്കിൽ defined(__ARM_ARCH)
#എങ്കിൽ defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
#അന്തര്ഭാവ <arm_sve.h>
സ്ഥിതം ശൂന്യം ggml_init_arm_arch_features(ശൂന്യം) {
    ggml_arm_arch_features.sve_cnt = svcntb();
}
#അല്ലെങ്കിൽ
സ്ഥിതം ശൂന്യം ggml_init_arm_arch_features(ശൂന്യം) {}
#endif
#endif // __ARM_ARCH

#എങ്കിൽ defined(__riscv) && defined(__riscv_v_intrinsic)
#അന്തര്ഭാവ <riscv_vector.h>
സ്ഥിതം ശൂന്യം ggml_init_riscv_arch_features(ശൂന്യം) {
    ggml_riscv_arch_features.rvv_vlen = __riscv_vlenb();
}
#അല്ലെങ്കിൽ
സ്ഥിതം ശൂന്യം ggml_init_riscv_arch_features(ശൂന്യം) {}
#endif

ഘടന ggml_tensor * ggml_new_i32(ഘടന ggml_context * ctx, int32_t മൂല്യം) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    ഘടന ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

    ggml_set_i32(result, മൂല്യം);

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_new_f32(ഘടന ggml_context * ctx, ദശാംശം മൂല്യം) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    ഘടന ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    ggml_set_f32(result, മൂല്യം);

    തിരികെ result;
}

ഘടന ggml_tensor * ggml_set_i32 (ഘടന ggml_tensor * tensor, int32_t മൂല്യം) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ n     = ggml_nrows(tensor);
    സ്ഥിരം പൂർണ്ണസംഖ്യ nc    = tensor->ne[0];
    സ്ഥിരം size_t n1 = tensor->nb[1];

    അക്ഷരം * സ്ഥിരം data = tensor->data;

    തിരഞ്ഞെടുപ്പ് (tensor->type) {
        അവസ്ഥ GGML_TYPE_I8:
            {
                പ്രതിജ്ഞാ(tensor->nb[0] == വലുപ്പം(int8_t));
                വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), മൂല്യം);
                }
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_I16:
            {
                പ്രതിജ്ഞാ(tensor->nb[0] == വലുപ്പം(int16_t));
                വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), മൂല്യം);
                }
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_I32:
            {
                പ്രതിജ്ഞാ(tensor->nb[0] == വലുപ്പം(int32_t));
                വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), മൂല്യം);
                }
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F16:
            {
                പ്രതിജ്ഞാ(tensor->nb[0] == വലുപ്പം(ggml_fp16_t));
                വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), GGML_CPU_FP32_TO_FP16(മൂല്യം));
                }
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_BF16:
            {
                പ്രതിജ്ഞാ(tensor->nb[0] == വലുപ്പം(ggml_fp16_t));
                വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n; i++) {
                    ggml_vec_set_bf16(nc, (ggml_bf16_t *)(data + i*n1), GGML_FP32_TO_BF16(മൂല്യം));
                }
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F32:
            {
                പ്രതിജ്ഞാ(tensor->nb[0] == വലുപ്പം(ദശാംശം));
                വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (ദശാംശം *)(data + i*n1), മൂല്യം);
                }
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }

    തിരികെ tensor;
}

ഘടന ggml_tensor * ggml_set_f32(ഘടന ggml_tensor * tensor, ദശാംശം മൂല്യം) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ n     = ggml_nrows(tensor);
    സ്ഥിരം പൂർണ്ണസംഖ്യ nc    = tensor->ne[0];
    സ്ഥിരം size_t n1 = tensor->nb[1];

    അക്ഷരം * സ്ഥിരം data = tensor->data;

    തിരഞ്ഞെടുപ്പ് (tensor->type) {
        അവസ്ഥ GGML_TYPE_I8:
            {
                പ്രതിജ്ഞാ(tensor->nb[0] == വലുപ്പം(int8_t));
                വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), മൂല്യം);
                }
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_I16:
            {
                പ്രതിജ്ഞാ(tensor->nb[0] == വലുപ്പം(int16_t));
                വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), മൂല്യം);
                }
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_I32:
            {
                പ്രതിജ്ഞാ(tensor->nb[0] == വലുപ്പം(int32_t));
                വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), മൂല്യം);
                }
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F16:
            {
                പ്രതിജ്ഞാ(tensor->nb[0] == വലുപ്പം(ggml_fp16_t));
                വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), GGML_CPU_FP32_TO_FP16(മൂല്യം));
                }
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_BF16:
            {
                പ്രതിജ്ഞാ(tensor->nb[0] == വലുപ്പം(ggml_bf16_t));
                വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n; i++) {
                    ggml_vec_set_bf16(nc, (ggml_bf16_t *)(data + i*n1), GGML_FP32_TO_BF16(മൂല്യം));
                }
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F32:
            {
                പ്രതിജ്ഞാ(tensor->nb[0] == വലുപ്പം(ദശാംശം));
                വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (ദശാംശം *)(data + i*n1), മൂല്യം);
                }
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }

    തിരികെ tensor;
}

int32_t ggml_get_i32_1d(സ്ഥിരം ഘടന ggml_tensor * tensor, പൂർണ്ണസംഖ്യ i) {
    എങ്കിൽ (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        തിരികെ ggml_get_i32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    തിരഞ്ഞെടുപ്പ് (tensor->type) {
        അവസ്ഥ GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == വലുപ്പം(int8_t));
                തിരികെ ((int8_t *)(tensor->data))[i];
            }
        അവസ്ഥ GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == വലുപ്പം(int16_t));
                തിരികെ ((int16_t *)(tensor->data))[i];
            }
        അവസ്ഥ GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == വലുപ്പം(int32_t));
                തിരികെ ((int32_t *)(tensor->data))[i];
            }
        അവസ്ഥ GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == വലുപ്പം(ggml_fp16_t));
                തിരികെ GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *)(tensor->data))[i]);
            }
        അവസ്ഥ GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == വലുപ്പം(ggml_bf16_t));
                തിരികെ GGML_BF16_TO_FP32(((ggml_bf16_t *)(tensor->data))[i]);
            }
        അവസ്ഥ GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == വലുപ്പം(ദശാംശം));
                തിരികെ ((ദശാംശം *)(tensor->data))[i];
            }
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ശൂന്യം ggml_set_i32_1d(സ്ഥിരം ഘടന ggml_tensor * tensor, പൂർണ്ണസംഖ്യ i, int32_t മൂല്യം) {
    എങ്കിൽ (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_set_i32_nd(tensor, id[0], id[1], id[2], id[3], മൂല്യം);
        തിരികെ;
    }
    തിരഞ്ഞെടുപ്പ് (tensor->type) {
        അവസ്ഥ GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == വലുപ്പം(int8_t));
                ((int8_t *)(tensor->data))[i] = മൂല്യം;
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == വലുപ്പം(int16_t));
                ((int16_t *)(tensor->data))[i] = മൂല്യം;
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == വലുപ്പം(int32_t));
                ((int32_t *)(tensor->data))[i] = മൂല്യം;
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == വലുപ്പം(ggml_fp16_t));
                ((ggml_fp16_t *)(tensor->data))[i] = GGML_CPU_FP32_TO_FP16(മൂല്യം);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == വലുപ്പം(ggml_bf16_t));
                ((ggml_bf16_t *)(tensor->data))[i] = GGML_FP32_TO_BF16(മൂല്യം);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == വലുപ്പം(ദശാംശം));
                ((ദശാംശം *)(tensor->data))[i] = മൂല്യം;
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

int32_t ggml_get_i32_nd(സ്ഥിരം ഘടന ggml_tensor * tensor, പൂർണ്ണസംഖ്യ i0, പൂർണ്ണസംഖ്യ i1, പൂർണ്ണസംഖ്യ i2, പൂർണ്ണസംഖ്യ i3) {
    ശൂന്യം * data   = (അക്ഷരം *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    തിരഞ്ഞെടുപ്പ് (tensor->type) {
        അവസ്ഥ GGML_TYPE_I8:
            തിരികെ ((int8_t *) data)[0];
        അവസ്ഥ GGML_TYPE_I16:
            തിരികെ ((int16_t *) data)[0];
        അവസ്ഥ GGML_TYPE_I32:
            തിരികെ ((int32_t *) data)[0];
        അവസ്ഥ GGML_TYPE_F16:
            തിരികെ GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *) data)[0]);
        അവസ്ഥ GGML_TYPE_BF16:
            തിരികെ GGML_BF16_TO_FP32(((ggml_bf16_t *) data)[0]);
        അവസ്ഥ GGML_TYPE_F32:
            തിരികെ ((ദശാംശം *) data)[0];
        സ്ഥിരസ്ഥിതി:
            GGML_ABORT("fatal error");
    }
}

ശൂന്യം ggml_set_i32_nd(സ്ഥിരം ഘടന ggml_tensor * tensor, പൂർണ്ണസംഖ്യ i0, പൂർണ്ണസംഖ്യ i1, പൂർണ്ണസംഖ്യ i2, പൂർണ്ണസംഖ്യ i3, int32_t മൂല്യം) {
    ശൂന്യം * data   = (അക്ഷരം *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    തിരഞ്ഞെടുപ്പ് (tensor->type) {
        അവസ്ഥ GGML_TYPE_I8:
            {
                ((int8_t *)(data))[0] = മൂല്യം;
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_I16:
            {
                ((int16_t *)(data))[0] = മൂല്യം;
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_I32:
            {
                ((int32_t *)(data))[0] = മൂല്യം;
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(data))[0] = GGML_CPU_FP32_TO_FP16(മൂല്യം);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(data))[0] = GGML_FP32_TO_BF16(മൂല്യം);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F32:
            {
                ((ദശാംശം *)(data))[0] = മൂല്യം;
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ദശാംശം ggml_get_f32_1d(സ്ഥിരം ഘടന ggml_tensor * tensor, പൂർണ്ണസംഖ്യ i) {
    എങ്കിൽ (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        തിരികെ ggml_get_f32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    തിരഞ്ഞെടുപ്പ് (tensor->type) {
        അവസ്ഥ GGML_TYPE_I8:
            {
                തിരികെ ((int8_t *)(tensor->data))[i];
            }
        അവസ്ഥ GGML_TYPE_I16:
            {
                തിരികെ ((int16_t *)(tensor->data))[i];
            }
        അവസ്ഥ GGML_TYPE_I32:
            {
                തിരികെ ((int32_t *)(tensor->data))[i];
            }
        അവസ്ഥ GGML_TYPE_F16:
            {
                തിരികെ GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *)(tensor->data))[i]);
            }
        അവസ്ഥ GGML_TYPE_BF16:
            {
                തിരികെ GGML_BF16_TO_FP32(((ggml_bf16_t *)(tensor->data))[i]);
            }
        അവസ്ഥ GGML_TYPE_F32:
            {
                തിരികെ ((ദശാംശം *)(tensor->data))[i];
            }
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ശൂന്യം ggml_set_f32_1d(സ്ഥിരം ഘടന ggml_tensor * tensor, പൂർണ്ണസംഖ്യ i, ദശാംശം മൂല്യം) {
    എങ്കിൽ (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_set_f32_nd(tensor, id[0], id[1], id[2], id[3], മൂല്യം);
        തിരികെ;
    }
    തിരഞ്ഞെടുപ്പ് (tensor->type) {
        അവസ്ഥ GGML_TYPE_I8:
            {
                ((int8_t *)(tensor->data))[i] = മൂല്യം;
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_I16:
            {
                ((int16_t *)(tensor->data))[i] = മൂല്യം;
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_I32:
            {
                ((int32_t *)(tensor->data))[i] = മൂല്യം;
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(tensor->data))[i] = GGML_CPU_FP32_TO_FP16(മൂല്യം);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(tensor->data))[i] = GGML_FP32_TO_BF16(മൂല്യം);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F32:
            {
                ((ദശാംശം *)(tensor->data))[i] = മൂല്യം;
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ദശാംശം ggml_get_f32_nd(സ്ഥിരം ഘടന ggml_tensor * tensor, പൂർണ്ണസംഖ്യ i0, പൂർണ്ണസംഖ്യ i1, പൂർണ്ണസംഖ്യ i2, പൂർണ്ണസംഖ്യ i3) {
    ശൂന്യം * data   = (അക്ഷരം *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    തിരഞ്ഞെടുപ്പ് (tensor->type) {
        അവസ്ഥ GGML_TYPE_I8:
            തിരികെ ((int8_t *) data)[0];
        അവസ്ഥ GGML_TYPE_I16:
            തിരികെ ((int16_t *) data)[0];
        അവസ്ഥ GGML_TYPE_I32:
            തിരികെ ((int32_t *) data)[0];
        അവസ്ഥ GGML_TYPE_F16:
            തിരികെ GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *) data)[0]);
        അവസ്ഥ GGML_TYPE_BF16:
            തിരികെ GGML_BF16_TO_FP32(((ggml_bf16_t *) data)[0]);
        അവസ്ഥ GGML_TYPE_F32:
            തിരികെ ((ദശാംശം *) data)[0];
        സ്ഥിരസ്ഥിതി:
            GGML_ABORT("fatal error");
    }
}

ശൂന്യം ggml_set_f32_nd(സ്ഥിരം ഘടന ggml_tensor * tensor, പൂർണ്ണസംഖ്യ i0, പൂർണ്ണസംഖ്യ i1, പൂർണ്ണസംഖ്യ i2, പൂർണ്ണസംഖ്യ i3, ദശാംശം മൂല്യം) {
    ശൂന്യം * data   = (അക്ഷരം *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    തിരഞ്ഞെടുപ്പ് (tensor->type) {
        അവസ്ഥ GGML_TYPE_I8:
            {
                ((int8_t *)(data))[0] = മൂല്യം;
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_I16:
            {
                ((int16_t *)(data))[0] = മൂല്യം;
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_I32:
            {
                ((int32_t *)(data))[0] = മൂല്യം;
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(data))[0] = GGML_CPU_FP32_TO_FP16(മൂല്യം);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(data))[0] = GGML_FP32_TO_BF16(മൂല്യം);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F32:
            {
                ((ദശാംശം *)(data))[0] = മൂല്യം;
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

////////////////////////////////////////////////////////////////////////////////

// ggml_compute_forward_mul_mat

സ്ഥിതം ശൂന്യം ggml_compute_forward_mul_mat_one_chunk(
    സ്ഥിരം ഘടന ggml_compute_params * params,
    ഘടന ggml_tensor * dst,
    സ്ഥിരം എണ്ണൽ ggml_type type,
    സ്ഥിരം int64_t അക്കം_rows_per_vec_dot,
    സ്ഥിരം int64_t ir0_start,
    സ്ഥിരം int64_t ir0_end,
    സ്ഥിരം int64_t ir1_start,
    സ്ഥിരം int64_t ir1_end) {

    സ്ഥിരം ഘടന ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ഘടന ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ദ്വിമാന_OP_LOCALS

    സ്ഥിരം bool src1_cont = ggml_is_contiguous(src1);

    ggml_vec_dot_t സ്ഥിരം vec_dot      = type_traits_cpu[type].vec_dot;
    എണ്ണൽ ggml_type സ്ഥിരം vec_dot_type = type_traits_cpu[type].vec_dot_type;

    // broadcast factors
    സ്ഥിരം int64_t r2 = ne12 / ne02;
    സ്ഥിരം int64_t r3 = ne13 / ne03;

    //printf("ir0_start = %6lld, ir0_end = %6lld, ir1_start = %6lld, ir1_end = %6lld\n", ir0_start, ir0_end, ir1_start, ir1_end);

    // threads with no work simply yield (not sure if it helps)
    എങ്കിൽ (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        തിരികെ;
    }

    സ്ഥിരം ശൂന്യം * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
    സ്ഥിരം size_t row_size = ggml_row_size(vec_dot_type, ne10);

    പ്രതിജ്ഞാ(ne12 % ne02 == 0);
    പ്രതിജ്ഞാ(ne13 % ne03 == 0);

    // block-tiling attempt
    സ്ഥിരം int64_t blck_0 = 16;
    സ്ഥിരം int64_t blck_1 = 16;

    സ്ഥിരം size_t src1_col_stride = src1_cont || src1->type != vec_dot_type ? row_size : nb11;

    // attempt to reduce false-sharing (does not seem to make a difference)
    // 16 * 2, accounting for mmla kernels
    ദശാംശം tmp[32];

    വേണ്ടി (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        വേണ്ടി (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            വേണ്ടി (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += അക്കം_rows_per_vec_dot) {
                സ്ഥിരം int64_t i13 = (ir1 / (ne12 * ne1));
                സ്ഥിരം int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                സ്ഥിരം int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                // broadcast src0 into src1
                സ്ഥിരം int64_t i03 = i13 / r3;
                സ്ഥിരം int64_t i02 = i12 / r2;

                സ്ഥിരം int64_t i1 = i11;
                സ്ഥിരം int64_t i2 = i12;
                സ്ഥിരം int64_t i3 = i13;

                സ്ഥിരം അക്ഷരം * src0_row = (സ്ഥിരം അക്ഷരം*)src0->data + (0 + i02 * nb02 + i03 * nb03);

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                സ്ഥിരം അക്ഷരം * src1_col = (സ്ഥിരം അക്ഷരം*)wdata +
                    (src1_cont || src1->type != vec_dot_type
                        ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                        : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                ദശാംശം * dst_col = (ദശാംശം*)((അക്ഷരം*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

                //for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                //    vec_dot(ne00, &dst_col[ir0], src0_row + ir0*nb01, src1_col);
                //}

                വേണ്ടി (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += അക്കം_rows_per_vec_dot) {
                    vec_dot(ne00, &tmp[ir0 - iir0], (അക്കം_rows_per_vec_dot > 1 ? 16 : 0), src0_row + ir0 * nb01, (അക്കം_rows_per_vec_dot > 1 ? nb01 : 0), src1_col, (അക്കം_rows_per_vec_dot > 1 ? src1_col_stride : 0), അക്കം_rows_per_vec_dot);
                }

                വേണ്ടി (പൂർണ്ണസംഖ്യ cn = 0; cn < അക്കം_rows_per_vec_dot; ++cn) {
                    memcpy(&dst_col[iir0 + cn * nb1 / nb0], tmp + (cn * 16), (MIN(iir0 + blck_0, ir0_end) - iir0) * വലുപ്പം(ദശാംശം));
                }
            }
        }
    }
}

ശൂന്യം ggml_compute_forward_mul_mat(
        സ്ഥിരം ഘടന ggml_compute_params * params,
              ഘടന ggml_tensor * dst) {

    സ്ഥിരം ഘടന ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ഘടന ggml_tensor * src1 = dst->src[1];

    സ്ഥിരം int32_t hint = ggml_get_op_params_i32(dst, 1);
    എങ്കിൽ (hint == GGML_HINT_SRC0_IS_HADAMARD && !params->use_ref) {
        ggml_compute_forward_fwht(params, dst);
        തിരികെ;
    }

    // If tiled is supported, it will execute the full op here and we return
    എങ്കിൽ (ggml_compute_forward_mul_mat_tiled(params, dst)) {
        തിരികെ;
    }

    GGML_TENSOR_ദ്വിമാന_OP_LOCALS

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    എണ്ണൽ ggml_type           സ്ഥിരം vec_dot_type         = type_traits_cpu[src0->type].vec_dot_type;
    ggml_from_float_t        സ്ഥിരം from_float           = type_traits_cpu[vec_dot_type].from_float;
    int64_t                  സ്ഥിരം vec_dot_അക്കം_rows     = type_traits_cpu[src0->type].nrows;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == വലുപ്പം(ദശാംശം));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    // TODO: extract to "extra_op"
#എങ്കിൽ GGML_USE_LLAMAFILE
    // broadcast factors
    സ്ഥിരം int64_t r2 = ne12 / ne02;
    സ്ഥിരം int64_t r3 = ne13 / ne03;

    സ്ഥിരം bool src1_cont = ggml_is_contiguous(src1);

    എങ്കിൽ (!params->use_ref && src1_cont) {
        വേണ്ടി (int64_t i13 = 0; i13 < ne13; i13++)
            വേണ്ടി (int64_t i12 = 0; i12 < ne12; i12++)
                എങ്കിൽ (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_size(src0->type),
                                     (സ്ഥിരം അക്ഷരം *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_size(src0->type),
                                     (സ്ഥിരം അക്ഷരം *)src1->data + i12*nb12 + i13*nb13,
                                     nb11/ggml_type_size(src1->type),
                                     (അക്ഷരം *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_size(dst->type),
                                     src0->type,
                                     src1->type,
                                     dst->type))
                    പോകുക UseGgmlGemm1;
        തിരികെ;
    }
UseGgmlGemm1:;
#endif

    എങ്കിൽ (src1->type != vec_dot_type) {
        അക്ഷരം * wdata = params->wdata;

        സ്ഥിരം size_t nbw0 = ggml_type_size(vec_dot_type);
        സ്ഥിരം size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        സ്ഥിരം size_t nbw2 = nbw1*ne11;
        സ്ഥിരം size_t nbw3 = nbw2*ne12;

        പ്രതിജ്ഞാ(params->wsize >= ne13*nbw3);
        // src1 is either packed from F32 into vec_dot_type, or widened from F16 or BF16 into the F32 work buffer
        സ്ഥിരം bool widen = src1->type != GGML_TYPE_F32;

        GGML_ASSERT(!widen || vec_dot_type == GGML_TYPE_F32);
        GGML_ASSERT(!widen || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_BF16);

    #എങ്കിൽ 0
        വേണ്ടി (int64_t i13 = 0; i13 < ne13; ++i13) {
            വേണ്ടി (int64_t i12 = 0; i12 < ne12; ++i12) {
                വേണ്ടി (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                    from_float((ദശാംശം *)((അക്ഷരം *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (ശൂന്യം *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                                ne10);
                }
            }
        }
    #അല്ലെങ്കിൽ
        സ്ഥിരം int64_t bs = ggml_blck_size(vec_dot_type);

        വേണ്ടി (int64_t i13 = 0; i13 < ne13; ++i13) {
            വേണ്ടി (int64_t i12 = 0; i12 < ne12; ++i12) {
                വേണ്ടി (int64_t i11 = 0; i11 < ne11; ++i11) {
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    സ്ഥിരം ശൂന്യം * src1_row  = (സ്ഥിരം അക്ഷരം *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10;
                    ശൂന്യം       * wdata_row =                    wdata  + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0;

                    സ്ഥിരം int64_t ne10_block_size = (ne10_block_end - ne10_block_start) * bs;

                    എങ്കിൽ (src1->type == GGML_TYPE_F16) {
                        ggml_cpu_fp16_to_fp32((സ്ഥിരം ggml_fp16_t *) src1_row, (ദശാംശം *) wdata_row, ne10_block_size);
                    } അല്ലെങ്കിൽ എങ്കിൽ (src1->type == GGML_TYPE_BF16) {
                        ggml_cpu_bf16_to_fp32((സ്ഥിരം ggml_bf16_t *) src1_row, (ദശാംശം *) wdata_row, ne10_block_size);
                    } അല്ലെങ്കിൽ {
                        from_float((സ്ഥിരം ദശാംശം *) src1_row, wdata_row, ne10_block_size);
                    }
                }
            }
        }
    #endif
    }

    എങ്കിൽ (ith == 0) {
        // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
        atomic_store_explicit(&params->threadpool->current_chunk, nth, memory_order_relaxed);
    }

    ggml_barrier(params->threadpool);

#എങ്കിൽ GGML_USE_LLAMAFILE
    എങ്കിൽ (!params->use_ref && src1->type != vec_dot_type) {
        സ്ഥിരം ശൂന്യം* wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        സ്ഥിരം size_t row_size = ggml_row_size(vec_dot_type, ne10);

        വേണ്ടി (int64_t i13 = 0; i13 < ne13; i13++)
            വേണ്ടി (int64_t i12 = 0; i12 < ne12; i12++)
                എങ്കിൽ (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_size(src0->type),
                                     (സ്ഥിരം അക്ഷരം *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_size(src0->type),
                                     (സ്ഥിരം അക്ഷരം *)wdata + (i12*ne11 + i13*ne12*ne11)*row_size,
                                     row_size/ggml_type_size(vec_dot_type),
                                     (അക്ഷരം *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_size(dst->type),
                                     src0->type,
                                     vec_dot_type,
                                     dst->type))
                    പോകുക UseGgmlGemm2;
        തിരികെ;
    }
UseGgmlGemm2:;
#endif

    // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
    സ്ഥിരം int64_t nr0 = ne0;

    // This is the size of the rest of the dimensions of the result
    സ്ഥിരം int64_t nr1 = ne1 * ne2 * ne3;

    // Now select a reasonable chunk size.
    പൂർണ്ണസംഖ്യ chunk_size = 16;

    // We need to step up the size if it's small
    എങ്കിൽ (nr0 == 1 || nr1 == 1) {
        chunk_size = 64;
    }

    // distribute the work across the inner or outer loop based on which one is larger
    // The number of chunks in the 0/1 dim.
    // CEIL(nr0/chunk_size)
    int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
    int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

    // If the chunking is poor for the number of threads on this setup, scrap the whole plan.  Re-chunk it by thread.
    //   Also, chunking by thread was measured to have perform better on NUMA systems.  See https://github.com/ggml-org/llama.cpp/pull/6915
    //   In theory, chunking should be just as useful on NUMA and non NUMA systems, but testing disagreed with that.
    എങ്കിൽ (nchunk0 * nchunk1 < nth * 4 || ggml_is_numa()) {
        // distribute the thread work across the inner or outer loop based on which one is larger
        nchunk0 = nr0 > nr1 ? nth : 1; // parallelize by src0 rows
        nchunk1 = nr0 > nr1 ? 1 : nth; // parallelize by src1 rows
    }

    // The number of elements in each chunk
    സ്ഥിരം int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    സ്ഥിരം int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    // The first chunk comes from our thread_id, the rest will get auto-assigned.
    പൂർണ്ണസംഖ്യ current_chunk = ith;

    ആകുമ്പോൾ (current_chunk < nchunk0 * nchunk1) {
        സ്ഥിരം int64_t ith0 = current_chunk % nchunk0;
        സ്ഥിരം int64_t ith1 = current_chunk / nchunk0;

        സ്ഥിരം int64_t ir0_start = dr0 * ith0;
        സ്ഥിരം int64_t ir0_end = MIN(ir0_start + dr0, nr0);

        സ്ഥിരം int64_t ir1_start = dr1 * ith1;
        സ്ഥിരം int64_t ir1_end = MIN(ir1_start + dr1, nr1);

        // dot kernels can handle 1 row and col at a time, but mmla kernels can process 2 rows and cols
        int64_t അക്കം_rows_per_vec_dot = vec_dot_അക്കം_rows;

        // these checks are needed to avoid crossing dim1 boundaries
        // can be optimized, but the logic would become more complicated, so keeping it like this for simplicity
        എങ്കിൽ ((nr0 % 2 != 0) || (ne11 % 2 != 0) || ((ir0_end - ir0_start) % 2 != 0) || ((ir1_end - ir1_start) % 2 != 0)) {
            അക്കം_rows_per_vec_dot = 1;
        }
        ggml_compute_forward_mul_mat_one_chunk(params, dst, src0->type, അക്കം_rows_per_vec_dot, ir0_start, ir0_end, ir1_start, ir1_end);

        എങ്കിൽ (nth >= nchunk0 * nchunk1) {
            നിർത്തുക;
        }

        current_chunk = atomic_fetch_add_explicit(&params->threadpool->current_chunk, 1, memory_order_relaxed);
    }
}

// ggml_compute_forward_mul_mat_id

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id)*ids->ne[0]*ids->ne[1] + (i1)]

ഘടന mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

സ്ഥിതം ശൂന്യം ggml_compute_forward_mul_mat_id_one_chunk(
    ഘടന ggml_tensor * dst,
    സ്ഥിരം ഘടന ggml_tensor * src0,
    സ്ഥിരം ഘടന ggml_tensor * src1,
    സ്ഥിരം ഘടന ggml_tensor * ids,
    സ്ഥിരം int64_t cur_a,
    സ്ഥിരം int64_t ir0_start,
    സ്ഥിരം int64_t ir0_end,
    സ്ഥിരം int64_t ir1_start,
    സ്ഥിരം int64_t ir1_end,
    സ്ഥിരം അക്ഷരം * src0_cur,
    സ്ഥിരം ഘടന mmid_row_mapping * matrix_rows,
    സ്ഥിരം size_t row_size,
    സ്ഥിരം bool src1_cont,
    സ്ഥിരം ശൂന്യം * wdata) {

    GGML_TENSOR_ദ്വിമാന_OP_LOCALS

    സ്ഥിരം എണ്ണൽ ggml_type type = src0->type;

    ggml_vec_dot_t    സ്ഥിരം vec_dot      = type_traits_cpu[type].vec_dot;
    എണ്ണൽ ggml_type    സ്ഥിരം vec_dot_type = type_traits_cpu[type].vec_dot_type;

    സ്ഥിരം int64_t blck_0 = 16;
    സ്ഥിരം int64_t blck_1 = 16;

    ദശാംശം tmp[16];

    വേണ്ടി (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        വേണ്ടി (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            വേണ്ടി (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ++ir1) {
                സ്ഥിരം int64_t _i12 = ir1; // logical row index for this expert

                ഘടന mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, _i12);
                സ്ഥിരം പൂർണ്ണസംഖ്യ id       = row_mapping.i1; // selected expert index

                സ്ഥിരം int64_t  i11 = id % ne11;
                സ്ഥിരം int64_t  i12 = row_mapping.i2; // row index in src1

                സ്ഥിരം int64_t  i1 = id;  // selected expert index
                സ്ഥിരം int64_t  i2 = i12; // row

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                സ്ഥിരം അക്ഷരം * src1_col = (സ്ഥിരം അക്ഷരം *) wdata +
                    (src1_cont || src1->type != vec_dot_type
                    ? (i11      + i12*ne11)*row_size
                    : (i11*nb11 + i12*nb12));

                ദശാംശം * dst_col = (ദശാംശം *) ((അക്ഷരം *) dst->data + (i1*nb1 + i2*nb2));

                വേണ്ടി (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                    vec_dot(ne00, &tmp[ir0 - iir0], 0, src0_cur + ir0*nb01, 0, src1_col, 0, 1);
                }

                memcpy(&dst_col[iir0], tmp, (MIN(iir0 + blck_0, ir0_end) - iir0)*വലുപ്പം(ദശാംശം));
            }
        }
    }
}

സ്ഥിതം ശൂന്യം * incr_ptr_aligned(ശൂന്യം ** p, size_t size, size_t align) {

    ശൂന്യം * ptr = *p;
    ptr = (ശൂന്യം *) GGML_PAD((uintptr_t) ptr, align);
    *p = (ശൂന്യം *) ((അക്ഷരം *) ptr + size);
    തിരികെ ptr;
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_mul_mat_id(
        സ്ഥിരം ഘടന ggml_compute_params * params,
              ഘടന ggml_tensor * dst) {

    സ്ഥിരം ഘടന ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ഘടന ggml_tensor * src1 = dst->src[1];
    സ്ഥിരം ഘടന ggml_tensor * ids = dst->src[2];

    GGML_TENSOR_ദ്വിമാന_OP_LOCALS

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം എണ്ണൽ ggml_type type = src0->type;

    സ്ഥിരം bool src1_cont = ggml_is_contiguous(src1);

    എണ്ണൽ ggml_type    സ്ഥിരം vec_dot_type    = type_traits_cpu[type].vec_dot_type;
    ggml_from_float_t സ്ഥിരം from_float      = type_traits_cpu[vec_dot_type].from_float;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == വലുപ്പം(ദശാംശം));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // row groups
    സ്ഥിരം പൂർണ്ണസംഖ്യ n_ids = ids->ne[0]; // n_expert_used
    സ്ഥിരം പൂർണ്ണസംഖ്യ n_as  = ne02;       // n_expert

    ശൂന്യം * wdata_cur = params->wdata;

    എങ്കിൽ (src1->type != vec_dot_type) {
        incr_ptr_aligned(&wdata_cur, ggml_row_size(vec_dot_type, ggml_nelements(src1)), വലുപ്പം(int64_t));
    }

    int64_t * matrix_row_counts = // [n_as]
        incr_ptr_aligned(&wdata_cur, n_as*വലുപ്പം(int64_t), വലുപ്പം(int64_t));

    ഘടന mmid_row_mapping * matrix_rows = // [n_as][ids->ne[0]*ids->ne[1]]
        incr_ptr_aligned(&wdata_cur, n_as*ids->ne[0]*ids->ne[1]*വലുപ്പം(ഘടന mmid_row_mapping), വലുപ്പം(int64_t));

    അക്ഷരം (*atomic_current_chunk)[CACHE_വരി_SIZE] = // [n_as]
        incr_ptr_aligned(&wdata_cur, CACHE_വരി_SIZE * n_as, CACHE_വരി_SIZE);

    // Tiled matmul (see tiled.h); per-thread work buffers, 0 bytes when disabled. The
    // reservation is unconditional, the per expert eligibility is decided at dispatch time
    അക്ഷരം * tiled_scratch = incr_ptr_aligned(&wdata_cur, ggml_tiled_wdata_size(nth, dst), 64);

    GGML_ASSERT(params->wsize >= (size_t)((അക്ഷരം *) wdata_cur - (അക്ഷരം *) params->wdata));

    എങ്കിൽ (src1->type != vec_dot_type) {
        അക്ഷരം * wdata = params->wdata;

        സ്ഥിരം size_t nbw0 = ggml_type_size(vec_dot_type);
        സ്ഥിരം size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        സ്ഥിരം size_t nbw2 = nbw1*ne11;
        സ്ഥിരം size_t nbw3 = nbw2*ne12;

        പ്രതിജ്ഞാ(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

#എങ്കിൽ 0
        വേണ്ടി (int64_t i13 = 0; i13 < ne13; ++i13) {
            വേണ്ടി (int64_t i12 = ith; i12 < ne12; i12 += nth) {
                വേണ്ടി (int64_t i11 = 0; i11 < ne11; ++i11) {
                    from_float((ദശാംശം *)((അക്ഷരം *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (ശൂന്യം *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                               ne10);
                }
            }
        }
#അല്ലെങ്കിൽ
        വേണ്ടി (int64_t i13 = 0; i13 < ne13; ++i13) {
            വേണ്ടി (int64_t i12 = 0; i12 < ne12; ++i12) {
                വേണ്ടി (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = ggml_blck_size(vec_dot_type);
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    from_float((ദശാംശം *)((അക്ഷരം *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10),
                               (ശൂന്യം *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0),
                               (ne10_block_end - ne10_block_start) * bs);
                }
            }
        }
#endif
    }

    എങ്കിൽ (ith == 0) {
        // initialize matrix_row_counts
        memset(matrix_row_counts, 0, n_as*വലുപ്പം(int64_t));

        // group rows by src0 matrix
        വേണ്ടി (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ id = 0; id < n_ids; ++id) {
                സ്ഥിരം int32_t i02 = *(സ്ഥിരം int32_t *) ((സ്ഥിരം അക്ഷരം *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);

                പ്രതിജ്ഞാ(i02 >= 0 && i02 < n_as);

                MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = (ഘടന mmid_row_mapping) {id, iid1};
                matrix_row_counts[i02] += 1;
            }
        }
    }

    // reset current_chunk
    വേണ്ടി (പൂർണ്ണസംഖ്യ cur_a = ith; cur_a < n_as; cur_a += nth) {
        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);
        *current_chunk_ctr = nth;
    }

    ggml_barrier(params->threadpool);

    വേണ്ടി (പൂർണ്ണസംഖ്യ cur_a = 0; cur_a < n_as; ++cur_a) {
        സ്ഥിരം int64_t cne1 = matrix_row_counts[cur_a];

        എങ്കിൽ (cne1 == 0) {
            തുടരുക;
        }

        // tiled takes over if profitable for this expert (see tiled.h)
        എങ്കിൽ (ggml_compute_forward_mul_mat_id_tiled(params, dst, cur_a, cne1, (സ്ഥിരം int32_t *) &MMID_MATRIX_ROW(cur_a, 0), tiled_scratch)) {
            തുടരുക;
        }

        സ്ഥിരം അക്ഷരം * src0_cur = (സ്ഥിരം അക്ഷരം *) src0->data + cur_a * nb02;
        സ്ഥിരം ശൂന്യം * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        സ്ഥിരം size_t row_size = ggml_row_size(vec_dot_type, ne10);

        സ്ഥിരം int64_t nr0 = ne01;
        സ്ഥിരം int64_t nr1 = cne1;

        പൂർണ്ണസംഖ്യ chunk_size = 16;
        എങ്കിൽ (nr0 == 1 || nr1 == 1) {
            chunk_size = 64;
        }

        // disable for NUMA
        സ്ഥിരം bool disable_chunking = ggml_is_numa();

        int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
        int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

        എങ്കിൽ (nchunk0 * nchunk1 < nth * 4 || disable_chunking) {
            nchunk0 = nr0 > nr1 ? nth : 1;
            nchunk1 = nr0 > nr1 ? 1 : nth;
        }

        സ്ഥിരം int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        സ്ഥിരം int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

        പൂർണ്ണസംഖ്യ current_chunk = ith;

        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);

        ആകുമ്പോൾ (current_chunk < nchunk0 * nchunk1) {
            സ്ഥിരം int64_t ith0 = current_chunk % nchunk0;
            സ്ഥിരം int64_t ith1 = current_chunk / nchunk0;

            സ്ഥിരം int64_t ir0_start = dr0 * ith0;
            സ്ഥിരം int64_t ir0_end = MIN(ir0_start + dr0, nr0);

            സ്ഥിരം int64_t ir1_start = dr1 * ith1;
            സ്ഥിരം int64_t ir1_end = MIN(ir1_start + dr1, nr1);

            ggml_compute_forward_mul_mat_id_one_chunk(
                dst, src0, src1, ids, cur_a,
                ir0_start, ir0_end, ir1_start, ir1_end,
                src0_cur, matrix_rows, row_size, src1_cont, wdata
            );

            എങ്കിൽ (nth >= nchunk0 * nchunk1) {
                നിർത്തുക;
            }

            current_chunk = atomic_fetch_add_explicit(current_chunk_ctr, 1, memory_order_relaxed);
        }
    }
}

/////////////////////////////////

സ്ഥിതം ശൂന്യം ggml_compute_forward(ഘടന ggml_compute_params * params, ഘടന ggml_tensor * tensor) {
    GGML_ASSERT(params);

    എങ്കിൽ (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {
        തിരികെ;
    }

    // extra_buffer op?
    എങ്കിൽ (ggml_cpu_extra_compute_forward(params, tensor)) {
        തിരികെ;
    }

    തിരഞ്ഞെടുപ്പ് (tensor->op) {
        അവസ്ഥ GGML_OP_DUP:
            {
                ggml_compute_forward_dup(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_ADD:
            {
                ggml_compute_forward_add(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_ADD_ID:
            {
                ggml_compute_forward_add_id(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_ADD1:
            {
                ggml_compute_forward_add1(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_ACC:
            {
                ggml_compute_forward_acc(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_SUB:
            {
                ggml_compute_forward_sub(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_MUL:
            {
                ggml_compute_forward_mul(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_DIV:
            {
                ggml_compute_forward_div(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_SQR:
            {
                ggml_compute_forward_sqr(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_SQRT:
            {
                ggml_compute_forward_sqrt(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_LOG:
            {
                ggml_compute_forward_log(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_SIN:
            {
                ggml_compute_forward_sin(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_COS:
            {
                ggml_compute_forward_cos(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_SUM:
            {
                ggml_compute_forward_sum(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_SUM_ROWS:
            {
                ggml_compute_forward_sum_rows(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_CUMSUM:
            {
                ggml_compute_forward_cumsum(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_MEAN:
            {
                ggml_compute_forward_mean(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_ARGMAX:
            {
                ggml_compute_forward_argmax(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_എണ്ണം_EQUAL:
            {
                ggml_compute_forward_എണ്ണം_equal(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_REPEAT:
            {
                ggml_compute_forward_repeat(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_REPEAT_BACK:
            {
                ggml_compute_forward_repeat_back(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_CONCAT:
            {
                ggml_compute_forward_concat(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_SILU_BACK:
            {
                ggml_compute_forward_silu_back(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_NORM:
            {
                ggml_compute_forward_norm(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_RMS_NORM:
            {
                ggml_compute_forward_rms_norm(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_RMS_NORM_BACK:
            {
                ggml_compute_forward_rms_norm_back(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_GROUP_NORM:
            {
                ggml_compute_forward_group_norm(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_L2_NORM:
            {
                ggml_compute_forward_l2_norm(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_MUL_MAT:
            {
                ggml_compute_forward_mul_mat(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_MUL_MAT_ID:
            {
                ggml_compute_forward_mul_mat_id(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_OUT_PROD:
            {
                ggml_compute_forward_out_prod(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_SCALE:
            {
                ggml_compute_forward_scale(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_SET:
            {
                ggml_compute_forward_set(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_CPY:
            {
                ggml_compute_forward_cpy(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_CONT:
            {
                ggml_compute_forward_cont(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_GET_ROWS:
            {
                ggml_compute_forward_get_rows(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_GET_ROWS_BACK:
            {
                ggml_compute_forward_get_rows_back(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_SET_ROWS:
            {
                ggml_compute_forward_set_rows(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_DIAG:
            {
                ggml_compute_forward_diag(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_DIAG_MASK_INF:
            {
                ggml_compute_forward_diag_mask_inf(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_DIAG_MASK_ZERO:
            {
                ggml_compute_forward_diag_mask_zero(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_SOFT_MAX:
            {
                ggml_compute_forward_soft_max(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_SOFT_MAX_BACK:
            {
                ggml_compute_forward_soft_max_ext_back(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_ROPE:
            {
                ggml_compute_forward_rope(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_ROPE_BACK:
            {
                ggml_compute_forward_rope_back(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_CLAMP:
            {
                ggml_compute_forward_clamp(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_compute_forward_conv_transpose_1d(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_IM2COL:
            {
                ggml_compute_forward_im2col(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_IM2COL_BACK:
            {
                ggml_compute_forward_im2col_back_f32(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_IM2COL_3D:
            {
                ggml_compute_forward_im2col_3d(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_COL2IM_1D:
            {
                ggml_compute_forward_col2im_1d(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_CONV_2D:
            {
                ggml_compute_forward_conv_2d(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_CONV_3D:
            {
                ggml_compute_forward_conv_3d(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_CONV_2D_DW:
            {
                ggml_compute_forward_conv_2d_dw(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_CONV_TRANSPOSE_2D:
            {
                ggml_compute_forward_conv_transpose_2d(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_POOL_1D:
            {
                ggml_compute_forward_pool_1d(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_POOL_2D:
            {
                ggml_compute_forward_pool_2d(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_POOL_2D_BACK:
            {
                ggml_compute_forward_pool_2d_back(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_UPSCALE:
            {
                ggml_compute_forward_upscale(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_PAD:
            {
                ggml_compute_forward_pad(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_PAD_REFLECT_1D:
            {
                ggml_compute_forward_pad_reflect_1d(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_ROLL:
            {
                ggml_compute_forward_roll(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_ARANGE:
            {
                ggml_compute_forward_arange(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_TIMESTEP_EMBEDDING:
            {
                ggml_compute_forward_timestep_embedding(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_ARGSORT:
            {
                ggml_compute_forward_argsort(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_മുകൾ_K:
            {
                ggml_compute_forward_മുകൾ_k(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_LEAKY_RELU:
            {
                ggml_compute_forward_leaky_relu(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_TRI:
            {
                ggml_compute_forward_tri(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_FILL:
            {
                ggml_compute_forward_fill(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_FLASH_ATTN_EXT:
            {
                ggml_compute_forward_flash_attn_ext(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_FLASH_ATTN_BACK:
            {
                int32_t t = ggml_get_op_params_i32(tensor, 0);
                GGML_ASSERT(t == 0 || t == 1);
                bool masked = t != 0;
                ggml_compute_forward_flash_attn_back(params, masked, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_SSM_CONV:
            {
                ggml_compute_forward_ssm_conv(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_SSM_SCAN:
            {
                ggml_compute_forward_ssm_scan(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_WIN_PART:
            {
                ggml_compute_forward_win_part(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_WIN_UNPART:
            {
                ggml_compute_forward_win_unpart(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_UNARY:
            {
                ggml_compute_forward_unary(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_GLU:
            {
                ggml_compute_forward_glu(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_GET_REL_POS:
            {
                ggml_compute_forward_get_rel_pos(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_ADD_REL_POS:
            {
                ggml_compute_forward_add_rel_pos(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_RWKV_WKV6:
            {
                ggml_compute_forward_rwkv_wkv6(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_GATED_LINEAR_ATTN:
            {
                ggml_compute_forward_gla(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_RWKV_WKV7:
            {
                ggml_compute_forward_rwkv_wkv7(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_SOLVE_TRI:
            {
                ggml_compute_forward_solve_tri(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_GATED_DELTA_NET:
            {
                ggml_compute_forward_gated_delta_net(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_LIGHTNING_INDEXER:
            {
                ggml_compute_forward_lightning_indexer(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_DSV4_HC_COMB:
            {
                ggml_compute_forward_dsv4_hc_comb(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_DSV4_HC_PRE:
            {
                ggml_compute_forward_dsv4_hc_pre(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_DSV4_HC_POST:
            {
                ggml_compute_forward_dsv4_hc_post(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_മുകൾ_K_MASK:
            {
                ggml_compute_forward_മുകൾ_k_mask(params, tensor);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_MAP_CUSTOM1:
            {
                ggml_compute_forward_map_custom1(params, tensor);
            }
            നിർത്തുക;
        അവസ്ഥ GGML_OP_MAP_CUSTOM2:
            {
                ggml_compute_forward_map_custom2(params, tensor);
            }
            നിർത്തുക;
        അവസ്ഥ GGML_OP_MAP_CUSTOM3:
            {
                ggml_compute_forward_map_custom3(params, tensor);
            }
            നിർത്തുക;
        അവസ്ഥ GGML_OP_CUSTOM:
            {
                ggml_compute_forward_custom(params, tensor);
            }
            നിർത്തുക;
        അവസ്ഥ GGML_OP_CROSS_ENTROPY_LOSS:
            {
                ggml_compute_forward_cross_entropy_loss(params, tensor);
            }
            നിർത്തുക;
        അവസ്ഥ GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            {
                ggml_compute_forward_cross_entropy_loss_back(params, tensor);
            }
            നിർത്തുക;
        അവസ്ഥ GGML_OP_OPT_STEP_ADAMW:
            {
                ggml_compute_forward_opt_step_adamw(params, tensor);
            }
            നിർത്തുക;
        അവസ്ഥ GGML_OP_OPT_STEP_SGD:
            {
                ggml_compute_forward_opt_step_sgd(params, tensor);
            }
            നിർത്തുക;
        അവസ്ഥ GGML_OP_NONE:
            {
                // nop
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_RESHAPE:
            {
                // nop
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_PERMUTE:
            {
                // nop
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_VIEW:
            {
                // nop
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_TRANSPOSE:
            {
                // nop
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_എണ്ണം:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// Android's libc implementation "bionic" does not support setting affinity
#എങ്കിൽ defined(__gnu_linux__)
സ്ഥിതം ശൂന്യം set_numa_thread_affinity(പൂർണ്ണസംഖ്യ thread_n) {
    എങ്കിൽ (!ggml_is_numa()) {
        തിരികെ;
    }

    പൂർണ്ണസംഖ്യ കണ്ണി_അക്കം;
    പൂർണ്ണസംഖ്യ rv;
    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    തിരഞ്ഞെടുപ്പ്(g_state.numa.numa_strategy) {
        അവസ്ഥ GGML_NUMA_STRATEGY_DISTRIBUTE:
            // run thread on node_num thread_n / (threads per node)
            കണ്ണി_അക്കം = thread_n % g_state.numa.n_nodes;
            നിർത്തുക;
        അവസ്ഥ GGML_NUMA_STRATEGY_ISOLATE:
            // run thread on current_node
            കണ്ണി_അക്കം = g_state.numa.current_കണ്ണി;
            നിർത്തുക;
        അവസ്ഥ GGML_NUMA_STRATEGY_NUMACTL:
            // use the cpuset that numactl gave us
            rv = pthread_setaffinity_np(pthread_self(), setsize, &g_state.numa.cpuset);
            എങ്കിൽ (rv) {
                fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n",strerror(rv));
            }
            തിരികെ;
        സ്ഥിരസ്ഥിതി:
            തിരികെ;
    }

    ഘടന ggml_numa_കണ്ണി * കണ്ണി = &g_state.numa.nodes[കണ്ണി_അക്കം];

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    വേണ്ടി (size_t i = 0; i < കണ്ണി->n_cpus; ++i) {
        CPU_SET_S(കണ്ണി->cpus[i], setsize, cpus);
    }

    rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    എങ്കിൽ (rv) {
            fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}

സ്ഥിതം ശൂന്യം clear_numa_thread_affinity(ശൂന്യം) {
    എങ്കിൽ (!ggml_is_numa()) {
        തിരികെ;
    }

    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    വേണ്ടി (unsigned i = 0; i < g_state.numa.total_cpus; ++i) {
        CPU_SET_S(i, setsize, cpus);
    }

    പൂർണ്ണസംഖ്യ rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    എങ്കിൽ (rv) {
        fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}
#അല്ലെങ്കിൽ
// TODO: Windows etc.
// (the linux implementation may also work on BSD, someone should test)
സ്ഥിതം ശൂന്യം set_numa_thread_affinity(പൂർണ്ണസംഖ്യ thread_n) { UNUSED(thread_n);  }
സ്ഥിതം ശൂന്യം clear_numa_thread_affinity(ശൂന്യം) {}
#endif

സ്ഥിതം പൂർണ്ണസംഖ്യ ggml_get_n_tasks(ഘടന ggml_tensor * കണ്ണി, പൂർണ്ണസംഖ്യ n_threads) {
    പൂർണ്ണസംഖ്യ n_tasks = 0;

    എങ്കിൽ (ggml_is_empty(കണ്ണി)) {
        // no need to multi-thread a no-op
        n_tasks = 1;
        തിരികെ n_tasks;
    }

    തിരഞ്ഞെടുപ്പ് (കണ്ണി->op) {
        അവസ്ഥ GGML_OP_CPY:
        അവസ്ഥ GGML_OP_DUP:
        അവസ്ഥ GGML_OP_CONT:
        അവസ്ഥ GGML_OP_ADD:
        അവസ്ഥ GGML_OP_ADD_ID:
        അവസ്ഥ GGML_OP_ADD1:
        അവസ്ഥ GGML_OP_ACC:
        അവസ്ഥ GGML_OP_CUMSUM:
        അവസ്ഥ GGML_OP_TRI:
        അവസ്ഥ GGML_OP_FILL:
            {
                n_tasks = n_threads;
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_SUB:
        അവസ്ഥ GGML_OP_SQR:
        അവസ്ഥ GGML_OP_SQRT:
        അവസ്ഥ GGML_OP_LOG:
        അവസ്ഥ GGML_OP_SIN:
        അവസ്ഥ GGML_OP_COS:
        അവസ്ഥ GGML_OP_SUM:
        അവസ്ഥ GGML_OP_SUM_ROWS:
        അവസ്ഥ GGML_OP_MEAN:
        അവസ്ഥ GGML_OP_ARGMAX:
            {
                n_tasks = 1;
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_എണ്ണം_EQUAL:
        അവസ്ഥ GGML_OP_SOLVE_TRI:
        അവസ്ഥ GGML_OP_GATED_DELTA_NET:
        അവസ്ഥ GGML_OP_DSV4_HC_COMB:
        അവസ്ഥ GGML_OP_DSV4_HC_PRE:
        അവസ്ഥ GGML_OP_DSV4_HC_POST:
        അവസ്ഥ GGML_OP_മുകൾ_K_MASK:
            {
                n_tasks = n_threads;
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_REPEAT:
        അവസ്ഥ GGML_OP_REPEAT_BACK:
        അവസ്ഥ GGML_OP_LEAKY_RELU:
            {
                n_tasks = 1;
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_UNARY:
            തിരഞ്ഞെടുപ്പ് (ggml_get_unary_op(കണ്ണി)) {
                അവസ്ഥ GGML_UNARY_OP_ABS:
                അവസ്ഥ GGML_UNARY_OP_SGN:
                അവസ്ഥ GGML_UNARY_OP_NEG:
                അവസ്ഥ GGML_UNARY_OP_STEP:
                അവസ്ഥ GGML_UNARY_OP_TANH:
                അവസ്ഥ GGML_UNARY_OP_ELU:
                അവസ്ഥ GGML_UNARY_OP_RELU:
                അവസ്ഥ GGML_UNARY_OP_SIGMOID:
                അവസ്ഥ GGML_UNARY_OP_HARDSWISH:
                അവസ്ഥ GGML_UNARY_OP_HARDSIGMOID:
                അവസ്ഥ GGML_UNARY_OP_EXP:
                അവസ്ഥ GGML_UNARY_OP_SOFTPLUS:
                അവസ്ഥ GGML_UNARY_OP_EXPM1:
                അവസ്ഥ GGML_UNARY_OP_FLOOR:
                അവസ്ഥ GGML_UNARY_OP_CEIL:
                അവസ്ഥ GGML_UNARY_OP_ROUND:
                അവസ്ഥ GGML_UNARY_OP_TRUNC:
                    {
                        n_tasks = 1;
                    } നിർത്തുക;

                അവസ്ഥ GGML_UNARY_OP_GELU:
                അവസ്ഥ GGML_UNARY_OP_GELU_ERF:
                അവസ്ഥ GGML_UNARY_OP_GELU_QUICK:
                അവസ്ഥ GGML_UNARY_OP_SILU:
                അവസ്ഥ GGML_UNARY_OP_XIELU:
                    {
                        n_tasks = n_threads;
                    } നിർത്തുക;
                സ്ഥിരസ്ഥിതി:
                    GGML_ABORT("fatal error");
            }
            നിർത്തുക;
        അവസ്ഥ GGML_OP_GLU:
            തിരഞ്ഞെടുപ്പ് (ggml_get_glu_op(കണ്ണി)) {
                അവസ്ഥ GGML_GLU_OP_REGLU:
                അവസ്ഥ GGML_GLU_OP_GEGLU:
                അവസ്ഥ GGML_GLU_OP_SWIGLU:
                അവസ്ഥ GGML_GLU_OP_SWIGLU_OAI:
                അവസ്ഥ GGML_GLU_OP_GEGLU_ERF:
                അവസ്ഥ GGML_GLU_OP_GEGLU_QUICK:
                അവസ്ഥ GGML_GLU_OP_SWIGLU_CLAMP:
                    {
                        n_tasks = n_threads;
                    } നിർത്തുക;
                സ്ഥിരസ്ഥിതി:
                    GGML_ABORT("fatal error");
            }
            നിർത്തുക;
        അവസ്ഥ GGML_OP_SILU_BACK:
        അവസ്ഥ GGML_OP_MUL:
        അവസ്ഥ GGML_OP_DIV:
        അവസ്ഥ GGML_OP_NORM:
        അവസ്ഥ GGML_OP_RMS_NORM:
        അവസ്ഥ GGML_OP_RMS_NORM_BACK:
        അവസ്ഥ GGML_OP_L2_NORM:
        അവസ്ഥ GGML_OP_GROUP_NORM:
        അവസ്ഥ GGML_OP_CONCAT:
        അവസ്ഥ GGML_OP_MUL_MAT:
        അവസ്ഥ GGML_OP_MUL_MAT_ID:
        അവസ്ഥ GGML_OP_OUT_PROD:
        അവസ്ഥ GGML_OP_SET_ROWS:
            {
                n_tasks = n_threads;
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_GET_ROWS:
            {
                // FIXME: get_rows can use additional threads, but the cost of launching additional threads
                // decreases performance with GPU offloading
                //n_tasks = n_threads;
                n_tasks = 1;
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_SCALE:
        അവസ്ഥ GGML_OP_SET:
        അവസ്ഥ GGML_OP_RESHAPE:
        അവസ്ഥ GGML_OP_VIEW:
        അവസ്ഥ GGML_OP_PERMUTE:
        അവസ്ഥ GGML_OP_TRANSPOSE:
        അവസ്ഥ GGML_OP_GET_ROWS_BACK:
        അവസ്ഥ GGML_OP_DIAG:
            {
                n_tasks = 1;
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_DIAG_MASK_ZERO:
        അവസ്ഥ GGML_OP_DIAG_MASK_INF:
        അവസ്ഥ GGML_OP_SOFT_MAX_BACK:
        അവസ്ഥ GGML_OP_ROPE:
        അവസ്ഥ GGML_OP_ROPE_BACK:
        അവസ്ഥ GGML_OP_ADD_REL_POS:
            {
                n_tasks = n_threads;
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_CLAMP:
            {
                n_tasks = 1; //TODO
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_SOFT_MAX:
            {
                n_tasks = MIN(n_threads, ggml_nrows(കണ്ണി->src[0]));
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_IM2COL:
        അവസ്ഥ GGML_OP_IM2COL_BACK:
        അവസ്ഥ GGML_OP_IM2COL_3D:
        അവസ്ഥ GGML_OP_CONV_2D:
        അവസ്ഥ GGML_OP_CONV_3D:
        അവസ്ഥ GGML_OP_CONV_2D_DW:
        അവസ്ഥ GGML_OP_COL2IM_1D:
        അവസ്ഥ GGML_OP_CONV_TRANSPOSE_1D:
        അവസ്ഥ GGML_OP_CONV_TRANSPOSE_2D:
            {
                n_tasks = n_threads;
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_POOL_1D:
        അവസ്ഥ GGML_OP_POOL_2D:
        അവസ്ഥ GGML_OP_POOL_2D_BACK:
            {
                n_tasks = 1;
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_UPSCALE:
        അവസ്ഥ GGML_OP_PAD:
        അവസ്ഥ GGML_OP_PAD_REFLECT_1D:
        അവസ്ഥ GGML_OP_ROLL:
        അവസ്ഥ GGML_OP_ARANGE:
        അവസ്ഥ GGML_OP_TIMESTEP_EMBEDDING:
        അവസ്ഥ GGML_OP_ARGSORT:
        അവസ്ഥ GGML_OP_മുകൾ_K:
        അവസ്ഥ GGML_OP_FLASH_ATTN_EXT:
        അവസ്ഥ GGML_OP_FLASH_ATTN_BACK:
        അവസ്ഥ GGML_OP_SSM_CONV:
        അവസ്ഥ GGML_OP_SSM_SCAN:
        അവസ്ഥ GGML_OP_LIGHTNING_INDEXER:
            {
                n_tasks = n_threads;
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_RWKV_WKV6:
        അവസ്ഥ GGML_OP_GATED_LINEAR_ATTN:
        അവസ്ഥ GGML_OP_RWKV_WKV7:
            {
                സ്ഥിരം int64_t n_heads = കണ്ണി->src[1]->ne[1];
                n_tasks = MIN(n_threads, n_heads);
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_WIN_PART:
        അവസ്ഥ GGML_OP_WIN_UNPART:
        അവസ്ഥ GGML_OP_GET_REL_POS:
            {
                n_tasks = 1;
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_MAP_CUSTOM1:
            {
                ഘടന ggml_map_custom1_op_params p;
                memcpy(&p, കണ്ണി->op_params, വലുപ്പം(p));
                എങ്കിൽ (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } അല്ലെങ്കിൽ {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_MAP_CUSTOM2:
            {
                ഘടന ggml_map_custom2_op_params p;
                memcpy(&p, കണ്ണി->op_params, വലുപ്പം(p));
                എങ്കിൽ (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } അല്ലെങ്കിൽ {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_MAP_CUSTOM3:
            {
                ഘടന ggml_map_custom3_op_params p;
                memcpy(&p, കണ്ണി->op_params, വലുപ്പം(p));
                എങ്കിൽ (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } അല്ലെങ്കിൽ {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_CUSTOM:
            {
                ഘടന ggml_custom_op_params p;
                memcpy(&p, കണ്ണി->op_params, വലുപ്പം(p));
                എങ്കിൽ (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } അല്ലെങ്കിൽ {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_CROSS_ENTROPY_LOSS:
        അവസ്ഥ GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        അവസ്ഥ GGML_OP_OPT_STEP_ADAMW:
        അവസ്ഥ GGML_OP_OPT_STEP_SGD:
            {
                n_tasks = n_threads;
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_NONE:
            {
                n_tasks = 1;
            } നിർത്തുക;
        അവസ്ഥ GGML_OP_എണ്ണം:
            {
                GGML_ABORT("fatal error");
            }
        സ്ഥിരസ്ഥിതി:
            {
                fprintf(stderr, "%s: op not implemented: ", __func__);
                എങ്കിൽ (കണ്ണി->op < GGML_OP_എണ്ണം) {
                    fprintf(stderr, "%s\n", ggml_op_name(കണ്ണി->op));
                } അല്ലെങ്കിൽ {
                    fprintf(stderr, "%d\n", കണ്ണി->op);
                }
                GGML_ABORT("fatal error");
            }
    }

    പ്രതിജ്ഞാ(n_tasks > 0);

    തിരികെ n_tasks;
}

സ്ഥിതം thread_ret_t ggml_graph_compute_secondary_thread(ശൂന്യം* data);

#എങ്കിൽ defined(_WIN32)
#അന്തര്ഭാവ "windows.h"

// TODO: support > 64 CPUs
സ്ഥിതം bool ggml_thread_apply_affinity(bool * mask) {
    HANDLE    h = GetCurrentThread();
    uint64_t  bitmask = 0ULL;

    പ്രതിജ്ഞാ(GGML_MAX_N_THREADS >= 64);

    വേണ്ടി (int32_t i = 0; i < 8; i++) {
        int32_t idx = i * 8;
        uint8_t val = 0;
        val |= mask[idx + 0] << 0;
        val |= mask[idx + 1] << 1;
        val |= mask[idx + 2] << 2;
        val |= mask[idx + 3] << 3;
        val |= mask[idx + 4] << 4;
        val |= mask[idx + 5] << 5;
        val |= mask[idx + 6] << 6;
        val |= mask[idx + 7] << 7;
        bitmask |= (uint64_t)val << idx;
    }

    വേണ്ടി (int32_t i = 64; i < GGML_MAX_N_THREADS; i++) {
        എങ്കിൽ (mask[i]) {
            fprintf(stderr, "warn: setting thread-affinity for > 64 CPUs isn't supported on windows!\n");
            നിർത്തുക;
        }
    }

    DWORD_PTR m = (DWORD_PTR)bitmask;

    m = SetThreadAffinityMask(h, m);

    തിരികെ m != 0;
}

സ്ഥിതം bool ggml_thread_apply_priority(int32_t prio) {
    // Note that on Windows the Process Priority Class must be updated in order to set Thread priority.
    // This is up to the applications.
    DWORD p = THREAD_PRIORITY_NORMAL;
    തിരഞ്ഞെടുപ്പ് (prio) {
        അവസ്ഥ GGML_SCHED_PRIO_താഴെയുള്ള:      p = THREAD_PRIORITY_BELOW_NORMAL;  നിർത്തുക;
        അവസ്ഥ GGML_SCHED_PRIO_NORMAL:   p = THREAD_PRIORITY_NORMAL;        നിർത്തുക;
        അവസ്ഥ GGML_SCHED_PRIO_MEDIUM:   p = THREAD_PRIORITY_ABOVE_NORMAL;  നിർത്തുക;
        അവസ്ഥ GGML_SCHED_PRIO_മുകളിലെ:     p = THREAD_PRIORITY_HIGHEST;       നിർത്തുക;
        അവസ്ഥ GGML_SCHED_PRIO_REALTIME: p = THREAD_PRIORITY_TIME_CRITICAL; നിർത്തുക;
    }

    എങ്കിൽ (prio != GGML_SCHED_PRIO_താഴെയുള്ള) {
        // Tell Windows that this thread should not be throttled (needs its own CPU core).
        // Newer Windows 11 versions aggressively park (offline) CPU cores and often place
        // all our threads onto the first 4 cores which results in terrible performance with
        // n_threads > 4
        #എങ്കിൽ _WIN32_WINNT >= 0x0602
        THREAD_POWER_THROTTLING_STATE t;
        ZeroMemory(&t, വലുപ്പം(t));
        t.Version     = THREAD_POWER_THROTTLING_CURRENT_VERSION;
        t.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
        t.StateMask   = 0;

        എങ്കിൽ (!SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &t, വലുപ്പം(t))) {
            GGML_LOG_DEBUG("failed to disable thread power throttling %d : (%d)\n", prio, (പൂർണ്ണസംഖ്യ) GetLastError());
            തിരികെ false;
        }
        #endif
    }

    എങ്കിൽ (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        തിരികെ true;
    }

    എങ്കിൽ (!SetThreadPriority(GetCurrentThread(), p)) {
        fprintf(stderr, "warn: failed to set thread priority %d : (%d)\n", prio, (പൂർണ്ണസംഖ്യ) GetLastError());
        തിരികെ false;
    }

    തിരികെ true;
}

#അല്ലെങ്കിലെങ്കിൽ defined(__APPLE__)
#അന്തര്ഭാവ <sys/types.h>
#അന്തര്ഭാവ <sys/resource.h>

സ്ഥിതം bool ggml_thread_apply_affinity(സ്ഥിരം bool * mask) {
    // Not supported on Apple platforms
    UNUSED(mask);
    തിരികെ true;
}

സ്ഥിതം bool ggml_thread_apply_priority(int32_t prio) {
    ഘടന sched_param p;
    int32_t policy = SCHED_OTHER;
    തിരഞ്ഞെടുപ്പ് (prio) {
        // TODO: there seems to be no way to set lower prio on Apple platforms
        അവസ്ഥ GGML_SCHED_PRIO_താഴെയുള്ള:      policy = SCHED_OTHER; p.sched_priority = 0;  നിർത്തുക;
        അവസ്ഥ GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  നിർത്തുക;
        അവസ്ഥ GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; നിർത്തുക;
        അവസ്ഥ GGML_SCHED_PRIO_മുകളിലെ:     policy = SCHED_FIFO;  p.sched_priority = 80; നിർത്തുക;
        അവസ്ഥ GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; നിർത്തുക;
    }

    എങ്കിൽ (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        തിരികെ true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    എങ്കിൽ (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        തിരികെ false;
    }

    തിരികെ true;
}

#അല്ലെങ്കിലെങ്കിൽ defined(__linux__)
// TODO: this may not work on BSD, to be verified

സ്ഥിതം bool ggml_thread_apply_affinity(സ്ഥിരം bool * mask) {
    cpu_set_t cpuset;
    പൂർണ്ണസംഖ്യ err;

    CPU_ZERO(&cpuset);

    വേണ്ടി (uint32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
        എങ്കിൽ (mask[i]) {
            GGML_PRINT_DEBUG("Thread %lx: adding %d to cpuset\n", pthread_self(), i);
            CPU_SET(i, &cpuset);
        }
    }

#ifdef __ANDROID__
    err = sched_setaffinity(0, വലുപ്പം(cpuset), &cpuset);
    എങ്കിൽ (err < 0) {
        err = errno;
    }
#അല്ലെങ്കിൽ
    err = pthread_setaffinity_np(pthread_self(), വലുപ്പം(cpuset), &cpuset);
#endif
    എങ്കിൽ (err != 0) {
        fprintf(stderr, "warn: failed to set affinity mask 0x%llx : %s (%d)\n", (unsigned long long)mask, strerror(err), err);
        തിരികെ false;
    }

    തിരികെ true;
}

സ്ഥിതം bool ggml_thread_apply_priority(int32_t prio) {
    ഘടന sched_param p;
    int32_t policy = SCHED_OTHER;
    തിരഞ്ഞെടുപ്പ് (prio) {
        അവസ്ഥ GGML_SCHED_PRIO_താഴെയുള്ള:      policy = SCHED_BATCH; p.sched_priority = 0;  നിർത്തുക;
        അവസ്ഥ GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  നിർത്തുക;
        അവസ്ഥ GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; നിർത്തുക;
        അവസ്ഥ GGML_SCHED_PRIO_മുകളിലെ:     policy = SCHED_FIFO;  p.sched_priority = 80; നിർത്തുക;
        അവസ്ഥ GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; നിർത്തുക;
    }

    എങ്കിൽ (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        തിരികെ true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    എങ്കിൽ (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        തിരികെ false;
    }

    തിരികെ true;
}

#അല്ലെങ്കിൽ // unsupported platforms

സ്ഥിതം bool ggml_thread_apply_affinity(സ്ഥിരം bool * mask) {
    UNUSED(mask);
    തിരികെ true;
}

സ്ഥിതം bool ggml_thread_apply_priority(int32_t prio) {
    UNUSED(prio);
    തിരികെ true;
}

#endif

സ്ഥിതം bool ggml_thread_cpumask_is_valid(സ്ഥിരം bool * mask) {
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_N_THREADS; i++) {
        എങ്കിൽ (mask[i]) { തിരികെ true; }
    }
    തിരികെ false;
}

സ്ഥിതം ശൂന്യം ggml_thread_cpumask_next(സ്ഥിരം bool * global_mask, bool * local_mask, bool strict, int32_t* iter) {
    എങ്കിൽ (!strict) {
        memcpy(local_mask, global_mask, GGML_MAX_N_THREADS);
        തിരികെ;
    } അല്ലെങ്കിൽ {
        memset(local_mask, 0, GGML_MAX_N_THREADS);
        int32_t base_idx = *iter;
        വേണ്ടി (int32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
            int32_t idx = base_idx + i;
            എങ്കിൽ (idx >= GGML_MAX_N_THREADS) {
                // Just a cheaper modulo
                idx -= GGML_MAX_N_THREADS;
            }
            എങ്കിൽ (global_mask[idx]) {
                local_mask[idx] = 1;
                *iter = idx + 1;
                തിരികെ;
            }
        }
    }
}

ശൂന്യം ggml_threadpool_free(ഘടന ggml_threadpool* threadpool) {
    എങ്കിൽ (!threadpool) തിരികെ;

    സ്ഥിരം പൂർണ്ണസംഖ്യ n_threads = threadpool->n_threads;

#ifndef GGML_USE_OPENMP
    ഘടന ggml_compute_state* workers = threadpool->workers;

    ggml_mutex_lock(&threadpool->mutex);

    threadpool->stop = true;
    threadpool->pause = false;

    ggml_cond_broadcast(&threadpool->cond);
    ggml_mutex_unlock(&threadpool->mutex);

    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 1; j < n_threads; j++) {
        int32_t rc = ggml_thread_join(workers[j].thrd, ശൂന്യ);
        GGML_ASSERT(rc == GGML_EXIT_SUCCESS || rc == GGML_EXIT_ABORTED);
        UNUSED(rc);
    }

    ggml_mutex_destroy(&threadpool->mutex);
    ggml_cond_destroy(&threadpool->cond);
#endif // GGML_USE_OPENMP

    സ്ഥിരം size_t workers_size = വലുപ്പം(ഘടന ggml_compute_state) * n_threads;
    ggml_aligned_free(threadpool->workers, workers_size);
    ggml_aligned_free(threadpool, വലുപ്പം(ഘടന ggml_threadpool));
}

#ifndef GGML_USE_OPENMP
// pause/resume must be called under mutex
സ്ഥിതം ശൂന്യം ggml_threadpool_pause_locked(ഘടന ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Pausing threadpool\n");
    threadpool->pause = true;
    ggml_cond_broadcast(&threadpool->cond);
}

സ്ഥിതം ശൂന്യം ggml_threadpool_resume_locked(ഘടന ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Resuming threadpool\n");
    threadpool->pause = false;
    ggml_cond_broadcast(&threadpool->cond);
}
#endif

ശൂന്യം ggml_threadpool_pause(ഘടന ggml_threadpool * threadpool) {
#ifndef GGML_USE_OPENMP
    ggml_mutex_lock(&threadpool->mutex);
    എങ്കിൽ (!threadpool->pause) {
       ggml_threadpool_pause_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#അല്ലെങ്കിൽ
    UNUSED(threadpool);
#endif
}

ശൂന്യം ggml_threadpool_resume(ഘടന ggml_threadpool * threadpool) {
#ifndef GGML_USE_OPENMP
    ggml_mutex_lock(&threadpool->mutex);
    എങ്കിൽ (threadpool->pause) {
       ggml_threadpool_resume_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#അല്ലെങ്കിൽ
    UNUSED(threadpool);
#endif
}

ഘടന ggml_cplan ggml_graph_plan(
          സ്ഥിരം ഘടന ggml_cgraph * cgraph,
                               പൂർണ്ണസംഖ്യ   n_threads,
            ഘടന ggml_threadpool * threadpool) {

    എങ്കിൽ (threadpool == ശൂന്യ) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
    }
    എങ്കിൽ (n_threads <= 0) {
        n_threads = threadpool ? threadpool->n_threads : GGML_DEFAULT_N_THREADS;
    }

#എങ്കിൽ defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
    // Emscripten without pthreads support can only use a single thread
    n_threads = 1;
#endif

#എങ്കിൽ defined(__wasi__)
    // WASI doesn't support parallelism yet
    n_threads = 1;
#endif

    size_t work_size = 0;

    ഘടന ggml_cplan cplan;
    memset(&cplan, 0, വലുപ്പം(ഘടന ggml_cplan));

    പൂർണ്ണസംഖ്യ max_tasks = 1;

    // thread scheduling for the different operations + work buffer size estimation
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < cgraph->n_nodes; i++) {
        ഘടന ggml_tensor * കണ്ണി = cgraph->nodes[i];

        സ്ഥിരം പൂർണ്ണസംഖ്യ n_tasks = ggml_get_n_tasks(കണ്ണി, n_threads);

        max_tasks = MAX(max_tasks, n_tasks);

        size_t cur = 0;

        എങ്കിൽ (!ggml_cpu_extra_work_size(n_threads, കണ്ണി, &cur)) {
            തിരഞ്ഞെടുപ്പ് (കണ്ണി->op) {
                അവസ്ഥ GGML_OP_CPY:
                അവസ്ഥ GGML_OP_DUP:
                    {
                        എങ്കിൽ (ggml_is_quantized(കണ്ണി->type) ||
                            // F16 -> BF16 and BF16 -> F16 copies go through intermediate F32
                            (കണ്ണി->src[0]->type == GGML_TYPE_F16  && കണ്ണി->src[1] && കണ്ണി->src[1]->type == GGML_TYPE_BF16) ||
                            (കണ്ണി->src[0]->type == GGML_TYPE_BF16 && കണ്ണി->src[1] && കണ്ണി->src[1]->type == GGML_TYPE_F16) ||
                            // conversion between F32 and I32
                            (കണ്ണി->src[0]->type == GGML_TYPE_F32 && കണ്ണി->src[1] && കണ്ണി->src[1]->type == GGML_TYPE_I32) ||
                            (കണ്ണി->src[0]->type == GGML_TYPE_I32 && കണ്ണി->src[1] && കണ്ണി->src[1]->type == GGML_TYPE_F32)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * കണ്ണി->ne[0] * n_tasks;
                        }
                    } നിർത്തുക;
                അവസ്ഥ GGML_OP_ADD:
                അവസ്ഥ GGML_OP_ADD_ID:
                അവസ്ഥ GGML_OP_ADD1:
                    {
                        എങ്കിൽ (ggml_is_quantized(കണ്ണി->src[0]->type)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * കണ്ണി->src[0]->ne[0] * n_tasks;
                        }
                    } നിർത്തുക;
                അവസ്ഥ GGML_OP_ACC:
                    {
                        എങ്കിൽ (ggml_is_quantized(കണ്ണി->src[0]->type)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * കണ്ണി->src[1]->ne[0] * n_tasks;
                        }
                    } നിർത്തുക;
                അവസ്ഥ GGML_OP_എണ്ണം_EQUAL:
                    {
                        cur = ggml_type_size(കണ്ണി->type)*n_tasks;
                    } നിർത്തുക;
                അവസ്ഥ GGML_OP_MUL_MAT:
                    {
                        സ്ഥിരം എണ്ണൽ ggml_type vec_dot_type = type_traits_cpu[കണ്ണി->src[0]->type].vec_dot_type;
                        എങ്കിൽ (കണ്ണി->src[1]->type != vec_dot_type) {
                            cur = ggml_row_size(vec_dot_type, ggml_nelements(കണ്ണി->src[1]));
                        }
                        // Workspace for tiled (see tiled.h)
                        cur = GGML_PAD(cur, 64);
                        cur += ggml_tiled_wdata_size(n_tasks, കണ്ണി);
                    } നിർത്തുക;
                അവസ്ഥ GGML_OP_MUL_MAT_ID:
                    {
                        cur = 0;
                        സ്ഥിരം ഘടന ggml_tensor * src0 = കണ്ണി->src[0];
                        സ്ഥിരം ഘടന ggml_tensor * src1 = കണ്ണി->src[1];
                        സ്ഥിരം ഘടന ggml_tensor * ids = കണ്ണി->src[2];
                        സ്ഥിരം എണ്ണൽ ggml_type vec_dot_type = type_traits_cpu[src0->type].vec_dot_type;
                        സ്ഥിരം പൂർണ്ണസംഖ്യ n_as = src0->ne[2];
                        // src1
                        എങ്കിൽ (src1->type != vec_dot_type) {
                            cur += ggml_row_size(vec_dot_type, ggml_nelements(src1)) + വലുപ്പം(int64_t);
                        }
                        // matrix_row_counts
                        cur += n_as * വലുപ്പം(int64_t) + വലുപ്പം(int64_t);
                        // matrix_rows
                        cur += n_as*ids->ne[0]*ids->ne[1]*വലുപ്പം(ഘടന mmid_row_mapping) + വലുപ്പം(int64_t);
                        // atomic_current_chunk
                        cur += CACHE_വരി_SIZE*n_as + CACHE_വരി_SIZE;
                        // Workspace for tiled (see tiled.h)
                        cur = GGML_PAD(cur, 64);
                        cur += ggml_tiled_wdata_size(n_tasks, കണ്ണി);
                    } നിർത്തുക;
                അവസ്ഥ GGML_OP_OUT_PROD:
                    {
                        എങ്കിൽ (ggml_is_quantized(കണ്ണി->src[0]->type) ||
                            കണ്ണി->src[0]->type == GGML_TYPE_F16) {
                            cur = ggml_type_size(GGML_TYPE_F32) * കണ്ണി->src[0]->ne[0] * n_tasks;
                        }
                    } നിർത്തുക;
                അവസ്ഥ GGML_OP_SET_ROWS:
                    {
                        എങ്കിൽ (കണ്ണി->src[0]->type == GGML_TYPE_F16 && കണ്ണി->type != GGML_TYPE_F16) {
                            cur = ggml_type_size(GGML_TYPE_F32) * കണ്ണി->src[0]->ne[0] * n_tasks;
                        }
                    } നിർത്തുക;
                അവസ്ഥ GGML_OP_SOFT_MAX:
                അവസ്ഥ GGML_OP_ROPE:
                അവസ്ഥ GGML_OP_ROPE_BACK:
                    {
                        cur = ggml_type_size(GGML_TYPE_F32) * കണ്ണി->ne[0] * n_tasks;
                    } നിർത്തുക;
                അവസ്ഥ GGML_OP_CONV_TRANSPOSE_1D:
                    {
                        GGML_ASSERT(കണ്ണി->src[0]->ne[3] == 1);
                        GGML_ASSERT(കണ്ണി->src[1]->ne[2] == 1);
                        GGML_ASSERT(കണ്ണി->src[1]->ne[3] == 1);

                        സ്ഥിരം int64_t ne00 = കണ്ണി->src[0]->ne[0];  // K
                        സ്ഥിരം int64_t ne01 = കണ്ണി->src[0]->ne[1];  // Cout
                        സ്ഥിരം int64_t ne02 = കണ്ണി->src[0]->ne[2];  // Cin
                        സ്ഥിരം int64_t ne10 = കണ്ണി->src[1]->ne[0];  // L
                        സ്ഥിരം int64_t ne11 = കണ്ണി->src[1]->ne[1];  // Cin

                        എങ്കിൽ ((കണ്ണി->src[0]->type == GGML_TYPE_F16 ||
                             കണ്ണി->src[0]->type == GGML_TYPE_BF16) &&
                            കണ്ണി->src[1]->type == GGML_TYPE_F32) {
                            cur += വലുപ്പം(ggml_fp16_t)*ne00*ne01*ne02;
                            cur += വലുപ്പം(ggml_fp16_t)*ne10*ne11;
                        } അല്ലെങ്കിൽ എങ്കിൽ (കണ്ണി->src[0]->type == GGML_TYPE_F32 &&
                                   കണ്ണി->src[1]->type == GGML_TYPE_F32) {
                            cur += വലുപ്പം(ദശാംശം)*ne00*ne01*ne02;
                            cur += വലുപ്പം(ദശാംശം)*ne10*ne11;
                        } അല്ലെങ്കിൽ {
                            GGML_ABORT("fatal error");
                        }
                    } നിർത്തുക;
                അവസ്ഥ GGML_OP_CONV_2D:
                അവസ്ഥ GGML_OP_CONV_3D:
                    {
                        cur = GGML_IM2COL_WORK_SIZE;
                    } നിർത്തുക;
                അവസ്ഥ GGML_OP_CONV_TRANSPOSE_2D:
                    {
                        സ്ഥിരം int64_t ne00 = കണ്ണി->src[0]->ne[0]; // W
                        സ്ഥിരം int64_t ne01 = കണ്ണി->src[0]->ne[1]; // H
                        സ്ഥിരം int64_t ne02 = കണ്ണി->src[0]->ne[2]; // Channels Out
                        സ്ഥിരം int64_t ne03 = കണ്ണി->src[0]->ne[3]; // Channels In

                        സ്ഥിരം int64_t ne10 = കണ്ണി->src[1]->ne[0]; // W
                        സ്ഥിരം int64_t ne11 = കണ്ണി->src[1]->ne[1]; // H
                        സ്ഥിരം int64_t ne12 = കണ്ണി->src[1]->ne[2]; // Channels In
                        സ്ഥിരം int64_t ne13 = കണ്ണി->src[1]->ne[3]; // Batch

                        GGML_ASSERT(കണ്ണി->src[0]->type == GGML_TYPE_F16 || കണ്ണി->src[0]->type == GGML_TYPE_F32);
                        GGML_ASSERT(കണ്ണി->src[1]->type == GGML_TYPE_F32);

                        cur += ggml_type_size(കണ്ണി->src[0]->type) * ne00 * ne01 * ne02 * ne03;
                        cur += ggml_type_size(കണ്ണി->src[0]->type) * ne10 * ne11 * ne12 * ne13;

                    } നിർത്തുക;
                അവസ്ഥ GGML_OP_മുകൾ_K:
                    {
                        cur += വലുപ്പം(int32_t)*കണ്ണി->src[0]->ne[0]*n_tasks;
                    } നിർത്തുക;
                അവസ്ഥ GGML_OP_FLASH_ATTN_EXT:
                    {
                        സ്ഥിരം int64_t neq2 = കണ്ണി->src[0]->ne[2]; // number of query heads
                        സ്ഥിരം int64_t DK = കണ്ണി->src[1]->ne[0];
                        സ്ഥിരം int64_t DV = കണ്ണി->src[2]->ne[0];

                        // Tiled flash attention scratch (tile sizes defined in common.h)
                        // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
                        size_t prefill  = വലുപ്പം(ദശാംശം)*(GGML_FA_TILE_Q*DK + 2*GGML_FA_TILE_Q*GGML_FA_TILE_KV + GGML_FA_TILE_Q*DV + GGML_FA_TILE_KV*DV + GGML_FA_TILE_KV*DK)*n_tasks;

                        // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
                        // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
                        size_t n_chunks = n_tasks;
                        size_t decode   = വലുപ്പം(ദശാംശം)*(neq2*n_chunks*(2+DV) + n_tasks*(DK + 2*DV));

                        cur += MAX(prefill, decode);
                    } നിർത്തുക;
                അവസ്ഥ GGML_OP_FLASH_ATTN_BACK:
                    {
                        സ്ഥിരം int64_t    D = കണ്ണി->src[0]->ne[0];
                        സ്ഥിരം int64_t ne11 = ggml_up(കണ്ണി->src[1]->ne[1], GGML_SOFT_MAX_UNROLL);
                        സ്ഥിരം int64_t mxDn = MAX(D, ne11) * 2; // *2 because of S and SM in ggml_compute_forward_flash_attn_back
                        എങ്കിൽ (കണ്ണി->src[1]->type == GGML_TYPE_F32) {
                            cur  = വലുപ്പം(ദശാംശം)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += വലുപ്പം(ദശാംശം)*mxDn*n_tasks; // this is overestimated by x2
                        } അല്ലെങ്കിൽ എങ്കിൽ (കണ്ണി->src[1]->type == GGML_TYPE_F16) {
                            cur  = വലുപ്പം(ദശാംശം)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += വലുപ്പം(ദശാംശം)*mxDn*n_tasks; // this is overestimated by x2
                        } അല്ലെങ്കിൽ എങ്കിൽ (കണ്ണി->src[1]->type == GGML_TYPE_BF16) {
                            cur  = വലുപ്പം(ദശാംശം)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += വലുപ്പം(ദശാംശം)*mxDn*n_tasks; // this is overestimated by x2
                        }
                    } നിർത്തുക;

                അവസ്ഥ GGML_OP_CROSS_ENTROPY_LOSS:
                    {
                        cur = ggml_type_size(കണ്ണി->type)*(n_tasks + കണ്ണി->src[0]->ne[0]*n_tasks);
                    } നിർത്തുക;
                അവസ്ഥ GGML_OP_GATED_DELTA_NET:
                    {
                        സ്ഥിരം int64_t S_v = കണ്ണി->src[2]->ne[0];
                        സ്ഥിരം int64_t K   = ggml_get_op_params_i32(കണ്ണി, 0);
                        സ്ഥിരം int64_t per_thread = S_v + (K > 1 ? S_v * S_v : 0);
                        cur = per_thread * വലുപ്പം(ദശാംശം) * n_tasks;
                    } നിർത്തുക;
                അവസ്ഥ GGML_OP_എണ്ണം:
                    {
                        GGML_ABORT("fatal error");
                    }
                അവസ്ഥ GGML_OP_LIGHTNING_INDEXER:
                    {
                        // temp buffer for dequantizing lightning indexer keys
                        സ്ഥിരം int64_t ne10 = കണ്ണി->src[1]->ne[0];
                        cur += വലുപ്പം(ദശാംശം)*ne10*n_tasks;
                    } നിർത്തുക;
                സ്ഥിരസ്ഥിതി:
                    നിർത്തുക;
            }
        }

        work_size = MAX(work_size, cur);
    }

    എങ്കിൽ (work_size > 0) {
        work_size += CACHE_വരി_SIZE*(n_threads);
    }

    cplan.threadpool = threadpool;
    cplan.n_threads  = MIN(max_tasks, n_threads);
    cplan.work_size  = work_size;
    cplan.work_data  = ശൂന്യ;

    തിരികെ cplan;
}


// Try to fuse the current node with subsequent nodes for better performance.
// Returns the number of nodes skipped by fusion (>=1), or 0 if no fusion was applied.
സ്ഥിതം bool ggml_cpu_disable_fusion = false;  // initialized once in ggml_cpu_init(), read-only afterwards

സ്ഥിതം പൂർണ്ണസംഖ്യ ggml_cpu_try_fuse_ops(
        സ്ഥിരം ഘടന ggml_cgraph * cgraph,
        സ്ഥിരം പൂർണ്ണസംഖ്യ കണ്ണി_n,
        സ്ഥിരം ഘടന ggml_compute_params * params,
        സ്ഥിരം ഘടന ggml_cplan * cplan) {

    എങ്കിൽ (ggml_cpu_disable_fusion || cplan->use_ref) {
        തിരികെ 0;
    }

    ഘടന ggml_tensor * കണ്ണി = cgraph->nodes[കണ്ണി_n];

    എങ്കിൽ (കണ്ണി->op == GGML_OP_RMS_NORM) {
        // RMS_NORM + MUL fusion
        സ്ഥിരം എണ്ണൽ ggml_op fuse_ops[] = { GGML_OP_RMS_NORM, GGML_OP_MUL };
        എങ്കിൽ (ggml_can_fuse(cgraph, കണ്ണി_n, fuse_ops, 2)) {
            ഘടന ggml_tensor * mul_കണ്ണി = cgraph->nodes[കണ്ണി_n + 1];
            സ്ഥിരം ഘടന ggml_tensor * mul_w = (mul_കണ്ണി->src[0] == കണ്ണി)
                ? mul_കണ്ണി->src[1] : mul_കണ്ണി->src[0];
            എങ്കിൽ (കണ്ണി->src[0]->type  == GGML_TYPE_F32 &&
                mul_കണ്ണി->type      == GGML_TYPE_F32 &&
                mul_w->type         == GGML_TYPE_F32 &&
                mul_w->ne[0]        == കണ്ണി->ne[0]   &&
                mul_w->nb[0]        == വലുപ്പം(ദശാംശം)) {

                ggml_compute_forward_rms_norm_mul_fused(params, കണ്ണി, mul_കണ്ണി);
                തിരികെ 1;
            }
        }
    }

    തിരികെ 0;
}

സ്ഥിതം thread_ret_t ggml_graph_compute_thread(ശൂന്യം * data) {
    ഘടന ggml_compute_state * state = (ഘടന ggml_compute_state *) data;
    ഘടന ggml_threadpool    * tp    = state->threadpool;

    സ്ഥിരം ഘടന ggml_cgraph * cgraph = tp->cgraph;
    സ്ഥിരം ഘടന ggml_cplan  * cplan  = tp->cplan;

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(state->ith);
#അല്ലെങ്കിൽ
    set_numa_thread_affinity(state->ith);
#endif

    ഘടന ggml_compute_params params = {
        /*.ith        =*/ state->ith,
        /*.nth        =*/ atomic_load_explicit(&tp->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK,
        /*.wsize      =*/ cplan->work_size,
        /*.wdata      =*/ cplan->work_data,
        /*.threadpool =*/ tp,
        /*.use_ref    =*/ cplan->use_ref,
    };

#ifdef GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p\n", state->ith, (സ്ഥിരം ശൂന്യം *)cplan);
#അല്ലെങ്കിൽ
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p last-graph %d\n", state->ith, (സ്ഥിരം ശൂന്യം *)cplan, state->last_graph);
#endif

    വേണ്ടി (പൂർണ്ണസംഖ്യ കണ്ണി_n = 0; കണ്ണി_n < cgraph->n_nodes && atomic_load_explicit(&tp->abort, memory_order_relaxed) != കണ്ണി_n; കണ്ണി_n++) {
        ഘടന ggml_tensor * കണ്ണി = cgraph->nodes[കണ്ണി_n];

        എങ്കിൽ (ggml_op_is_empty(കണ്ണി->op)) {
            // skip NOPs
            തുടരുക;
        }

        എങ്കിൽ ((കണ്ണി->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            തുടരുക;
        }

        // TODO: move fused-op detection into ggml_graph_plan so fusion decisions are made once at planning time
        // Try fused ops, fall back to normal compute
        സ്ഥിരം പൂർണ്ണസംഖ്യ n_fused = ggml_cpu_try_fuse_ops(cgraph, കണ്ണി_n, &params, cplan);
        എങ്കിൽ (n_fused > 0) {
            കണ്ണി_n += n_fused;
        } അല്ലെങ്കിൽ {
            ggml_compute_forward(&params, കണ്ണി);
        }

        എങ്കിൽ (state->ith == 0 && cplan->abort_callback &&
                cplan->abort_callback(cplan->abort_callback_data)) {
            atomic_store_explicit(&tp->abort, കണ്ണി_n + 1, memory_order_relaxed);
            tp->ec    = GGML_STATUS_ABORTED;
        }

        എങ്കിൽ (കണ്ണി_n + 1 < cgraph->n_nodes) {
            ggml_barrier(state->threadpool);
        }
    }

#ifdef GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p\n", state->ith, (സ്ഥിരം ശൂന്യം *)cplan);
#അല്ലെങ്കിൽ
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p last-graph %d\n", state->ith, (സ്ഥിരം ശൂന്യം *)cplan, state->last_graph);
#endif

    ggml_barrier(state->threadpool);

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(state->ith);
#endif

    തിരികെ 0;
}

#ifndef GGML_USE_OPENMP

// check if thread is ready to proceed (exit from polling or sleeping)
// returns true if loops should exit, sets state->pending to indicate new work
സ്ഥിതം വരിയിൽ bool ggml_graph_compute_thread_ready(ഘടന ggml_compute_state * state) {
    ഘടന ggml_threadpool * threadpool = state->threadpool;

    എങ്കിൽ (state->pending || threadpool->stop || threadpool->pause) { തിരികെ true; }

    // check for new graph/work
    പൂർണ്ണസംഖ്യ n_graph   = atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed);
    പൂർണ്ണസംഖ്യ n_threads = n_graph & GGML_THREADPOOL_N_THREADS_MASK;
    എങ്കിൽ (n_graph != state->last_graph) {
        state->pending    = (state->ith < n_threads);
        state->last_graph = n_graph;
        തിരികെ true;
    }

    തിരികെ false;
}

// sync thread state after polling
സ്ഥിതം വരിയിൽ ശൂന്യം ggml_graph_compute_thread_sync(ഘടന ggml_compute_state * state) {
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&state->threadpool->n_graph, 0, memory_order_seq_cst);
    #അല്ലെങ്കിൽ
    atomic_thread_fence(memory_order_seq_cst);
    #endif
    UNUSED(state);
}

സ്ഥിതം വരിയിൽ bool ggml_graph_compute_poll_for_work(ഘടന ggml_compute_state * state) {
    ഘടന ggml_threadpool * threadpool = state->threadpool;

    // This seems to make 0 ... 100 a decent range for polling level across modern processors.
    // Perhaps, we can adjust it dynamically based on load and things.
    സ്ഥിരം uint64_t n_rounds = 1024UL * 128 * threadpool->poll;

    വേണ്ടി (uint64_t i=0; !ggml_graph_compute_thread_ready(state) && i < n_rounds; i++) {
        // No new work. Keep polling.
        ggml_thread_cpu_relax();
    }

    തിരികെ state->pending;
}

സ്ഥിതം വരിയിൽ bool ggml_graph_compute_check_for_work(ഘടന ggml_compute_state * state) {
    ഘടന ggml_threadpool * threadpool = state->threadpool;

    എങ്കിൽ (ggml_graph_compute_poll_for_work(state)) {
        ggml_graph_compute_thread_sync(state);
        തിരികെ state->pending;
    }

    ggml_mutex_lock_shared(&threadpool->mutex);
    ആകുമ്പോൾ (!ggml_graph_compute_thread_ready(state)) {
        // No new work. Wait for the signal.
        GGML_PRINT_DEBUG("thread #%d waiting for work (sleeping)\n", state->ith);
        ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
    }
    ggml_mutex_unlock_shared(&threadpool->mutex);

    തിരികെ state->pending;
}

സ്ഥിതം thread_ret_t ggml_graph_compute_secondary_thread(ശൂന്യം* data) {
    ഘടന ggml_compute_state * state = (ഘടന ggml_compute_state *) data;
    ഘടന ggml_threadpool * threadpool = state->threadpool;

    ggml_thread_apply_priority(threadpool->prio);
    എങ്കിൽ (ggml_thread_cpumask_is_valid(state->cpumask)) {
        ggml_thread_apply_affinity(state->cpumask);
    }

    ആകുമ്പോൾ (true) {
        // Check if we need to sleep
        ആകുമ്പോൾ (threadpool->pause) {
            GGML_PRINT_DEBUG("thread #%d inside pause loop\n", state->ith);
            ggml_mutex_lock_shared(&threadpool->mutex);
            എങ്കിൽ (threadpool->pause) {
                ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
            }
            GGML_PRINT_DEBUG("thread #%d resuming after wait\n", state->ith);
            ggml_mutex_unlock_shared(&threadpool->mutex);
        }

        // This needs to be checked for after the cond_wait
        എങ്കിൽ (threadpool->stop) നിർത്തുക;

        // Check if there is new work
        // The main thread is the only one that can dispatch new work

        ggml_graph_compute_check_for_work(state);
        എങ്കിൽ (state->pending) {
            state->pending = false;
            ggml_graph_compute_thread(state);
        }
    }

    തിരികെ (thread_ret_t) 0;
}

// Start processing new graph
സ്ഥിതം ശൂന്യം ggml_graph_compute_kickoff(ഘടന ggml_threadpool * threadpool, പൂർണ്ണസംഖ്യ n_threads)
{
    // Always take the mutex here because the worker threads are doing hybrid poll/wait

    ggml_mutex_lock(&threadpool->mutex);

    // Update the number of active threads and the graph count
    പൂർണ്ണസംഖ്യ n_graph = atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed) >> GGML_THREADPOOL_N_THREADS_BITS;
    n_graph = ((n_graph + 1) << GGML_THREADPOOL_N_THREADS_BITS) | (n_threads & GGML_THREADPOOL_N_THREADS_MASK);

    GGML_PRINT_DEBUG("compute-kickoff: n_threads %d n_graph %d\n", n_threads, n_graph);

    // Indicate the graph is ready to be processed
    // We need the full seq-cst fence here because of the polling threads (used in thread_sync)
    atomic_store_explicit(&threadpool->n_graph, n_graph, memory_order_seq_cst);

    എങ്കിൽ (threadpool->pause) {
       // Update main thread prio and affinity to match the threadpool settings
       ggml_thread_apply_priority(threadpool->prio);
       എങ്കിൽ (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
           ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
       }

       // resume does cond broadcast
       ggml_threadpool_resume_locked(threadpool);
    } അല്ലെങ്കിൽ {
       ggml_cond_broadcast(&threadpool->cond);
    }

    ggml_mutex_unlock(&threadpool->mutex);
}

#endif // GGML_USE_OPENMP

സ്ഥിതം ഘടന ggml_threadpool * ggml_threadpool_new_impl(
    ഘടന ggml_threadpool_params * tpp,
               ഘടന ggml_cgraph * cgraph,
                ഘടന ggml_cplan * cplan) {

    ഘടന ggml_threadpool * threadpool =
        ggml_aligned_malloc(വലുപ്പം(ഘടന ggml_threadpool));
    {
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->n_graph          = 0;
        threadpool->n_barrier        = 0;
        threadpool->n_barrier_passed = 0;
        threadpool->current_chunk    = 0;
        threadpool->stop             = false;
        threadpool->pause            = tpp->paused;
        threadpool->abort            = -1;
        threadpool->workers          = ശൂന്യ;
        threadpool->n_threads        = tpp->n_threads;
        threadpool->poll             = tpp->poll;
        threadpool->prio             = tpp->prio;
        threadpool->ec               = GGML_STATUS_SUCCESS;
    }

    // Allocate and init workers state
    സ്ഥിരം size_t workers_size = വലുപ്പം(ഘടന ggml_compute_state) * tpp->n_threads;
    ഘടന ggml_compute_state * workers = ggml_aligned_malloc(workers_size);

    memset(workers, 0, workers_size);
    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < tpp->n_threads; j++) {
        workers[j].threadpool = threadpool;
        workers[j].ith        = j;
    }

    threadpool->workers = workers;

#ifdef GGML_USE_OPENMP
    int32_t cpumask_iter = 0;

    // Compute CPU masks for each thread
    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < tpp->n_threads; j++) {
        ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);
    }
#അല്ലെങ്കിൽ // GGML_USE_OPENMP
    ggml_mutex_init(&threadpool->mutex);
    ggml_cond_init(&threadpool->cond);

    // Spin the threads for all workers, and update CPU placements.
    // Place the main thread last (towards the higher numbered CPU cores).

    int32_t cpumask_iter = 0;

    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 1; j < tpp->n_threads; j++) {
        ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);

        int32_t rc = ggml_thread_create(&workers[j].thrd, ശൂന്യ, ggml_graph_compute_secondary_thread, &workers[j]);
        GGML_ASSERT(rc == 0);
    }

    ggml_thread_cpumask_next(tpp->cpumask, workers[0].cpumask, tpp->strict_cpu, &cpumask_iter);

    എങ്കിൽ (!threadpool->pause) {
        // Update main thread prio and affinity at the start, otherwise we'll do it in resume
        ggml_thread_apply_priority(threadpool->prio);
        എങ്കിൽ (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
            ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
        }
    }
#endif // GGML_USE_OPENMP

    തിരികെ threadpool;
}

ഘടന ggml_threadpool * ggml_threadpool_new(ഘടന ggml_threadpool_params * tpp) {
    തിരികെ ggml_threadpool_new_impl(tpp, ശൂന്യ, ശൂന്യ);
}

എണ്ണൽ ggml_status ggml_graph_compute(ഘടന ggml_cgraph * cgraph, ഘടന ggml_cplan * cplan) {
    ggml_cpu_init();

    GGML_ASSERT(cplan);
    GGML_ASSERT(cplan->n_threads > 0);
    GGML_ASSERT(cplan->work_size == 0 || cplan->work_data != ശൂന്യ);

    പൂർണ്ണസംഖ്യ n_threads                               = cplan->n_threads;
    ഘടന ggml_threadpool * threadpool = cplan->threadpool;

    bool disposable_threadpool = false;

    എങ്കിൽ (threadpool == ശൂന്യ) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
        disposable_threadpool = true;

        ഘടന ggml_threadpool_params ttp = ggml_threadpool_params_default(n_threads);
        threadpool = ggml_threadpool_new_impl(&ttp, cgraph, cplan);
    } അല്ലെങ്കിൽ {
        // Reset some of the parameters that need resetting
        // No worker threads should be accessing the parameters below at this stage
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->current_chunk    = 0;
        threadpool->abort            = -1;
        threadpool->ec               = GGML_STATUS_SUCCESS;
    }

#ifdef GGML_USE_OPENMP
    എങ്കിൽ (n_threads > 1) {
        #pragma omp parallel അക്കം_threads(n_threads)
        {
            #pragma omp single
            {
                // update the number of threads from the actual number of threads that we got from OpenMP
                n_threads = omp_get_അക്കം_threads();
                atomic_store_explicit(&threadpool->n_graph, n_threads, memory_order_relaxed);
            }

            // Apply thread CPU mask and priority
            പൂർണ്ണസംഖ്യ ith = omp_get_thread_അക്കം();

            ggml_thread_apply_priority(threadpool->prio);
            എങ്കിൽ (ggml_thread_cpumask_is_valid(threadpool->workers[ith].cpumask)) {
                ggml_thread_apply_affinity(threadpool->workers[ith].cpumask);
            }
            ggml_graph_compute_thread(&threadpool->workers[ith]);
        }
    } അല്ലെങ്കിൽ {
        atomic_store_explicit(&threadpool->n_graph, 1, memory_order_relaxed);
        ggml_graph_compute_thread(&threadpool->workers[0]);
    }
#അല്ലെങ്കിൽ
    എങ്കിൽ (n_threads > threadpool->n_threads) {
        GGML_LOG_WARN("cplan requested more threads (%d) than available (%d)\n", n_threads, threadpool->n_threads);
        n_threads = threadpool->n_threads;
    }

    // Kick all threads to start the new graph
    ggml_graph_compute_kickoff(threadpool, n_threads);

    // This is a work thread too
    ggml_graph_compute_thread(&threadpool->workers[0]);
#endif

    // don't leave affinity set on the main thread
    clear_numa_thread_affinity();

    എണ്ണൽ ggml_status ret = threadpool->ec;

    എങ്കിൽ (disposable_threadpool) {
        ggml_threadpool_free(threadpool);
    }

    തിരികെ ret;
}

എണ്ണൽ ggml_status ggml_graph_compute_with_ctx(ഘടന ggml_context * ctx, ഘടന ggml_cgraph * cgraph, പൂർണ്ണസംഖ്യ n_threads) {
    ഘടന ggml_cplan cplan = ggml_graph_plan(cgraph, n_threads, ശൂന്യ);

    cplan.work_data = (uint8_t *)ggml_new_buffer(ctx, cplan.work_size);

    തിരികെ ggml_graph_compute(cgraph, &cplan);
}

ശൂന്യം ggml_cpu_fp32_to_fp32(സ്ഥിരം ദശാംശം * x, ദശാംശം * y, int64_t n) {
    memcpy(y, x, n * വലുപ്പം(ദശാംശം));
}

ശൂന്യം ggml_cpu_fp32_to_fp16(സ്ഥിരം ദശാംശം * x, ggml_fp16_t * y, int64_t n) {
    int64_t i = 0;
#എങ്കിൽ defined(__F16C__)
#എങ്കിൽ defined(__AVX512F__)
    വേണ്ടി (; i + 15 < n; i += 16) {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        __m256i y_vec = _mm512_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256((__m256i *)(y + i), y_vec);
    }
#endif
    വേണ്ടി (; i + 7 < n; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m128i y_vec = _mm256_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i *)(y + i), y_vec);
    }
    വേണ്ടി (; i + 3 < n; i += 4) {
        __m128 x_vec = _mm_loadu_ps(x + i);
        __m128i y_vec = _mm_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storel_epi64((__m128i *)(y + i), y_vec);
    }
#അല്ലെങ്കിലെങ്കിൽ defined(__riscv_zvfh)
    വേണ്ടി (പൂർണ്ണസംഖ്യ vl; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(&x[i], vl);
        vfloat16m1_t vy = __riscv_vfncvt_f_f_w_f16m1(vx, vl);
        __riscv_vse16_v_f16m1((_Float16 *)&y[i], vy, vl);
    }
#അല്ലെങ്കിലെങ്കിൽ defined(__VXE__) || defined(__VXE2__)
    വേണ്ടി (; i + 7 < n; i += 8) {
        സ്ഥിരം uint32x4_t v_yl = __lzs_f32cx4_to_f16(vec_xl(0, x + i + 0));
        സ്ഥിരം uint32x4_t v_yh = __lzs_f32cx4_to_f16(vec_xl(0, x + i + 4));
        vec_xst(vec_pack(v_yl, v_yh), 0, (uint16_t *)(y + i));
    }
#endif
    വേണ്ടി (; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(x[i]);
    }
}

ശൂന്യം ggml_cpu_fp16_to_fp32(സ്ഥിരം ggml_fp16_t * x, ദശാംശം * y, int64_t n) {
    int64_t i = 0;
#എങ്കിൽ defined(__F16C__)
#എങ്കിൽ defined(__AVX512F__)
    വേണ്ടി (; i + 15 < n; i += 16) {
        __m256i x_vec = _mm256_loadu_si256((സ്ഥിരം __m256i *)(x + i));
        __m512 y_vec = _mm512_cvtph_ps(x_vec);
        _mm512_storeu_ps(y + i, y_vec);
    }
#endif
    വേണ്ടി (; i + 7 < n; i += 8) {
        __m128i x_vec = _mm_loadu_si128((സ്ഥിരം __m128i *)(x + i));
        __m256 y_vec = _mm256_cvtph_ps(x_vec);
        _mm256_storeu_ps(y + i, y_vec);
    }
    വേണ്ടി (; i + 3 < n; i += 4) {
        __m128i x_vec = _mm_loadl_epi64((സ്ഥിരം __m128i *)(x + i));
        __m128 y_vec = _mm_cvtph_ps(x_vec);
        _mm_storeu_ps(y + i, y_vec);
    }

#അല്ലെങ്കിലെങ്കിൽ defined(__riscv_v_intrinsic) && defined(__riscv_zvfhmin)
    // calculate step size
    സ്ഥിരം പൂർണ്ണസംഖ്യ epr = __riscv_vsetvlmax_e16m2();
    സ്ഥിരം പൂർണ്ണസംഖ്യ step = epr * 2;
    സ്ഥിരം പൂർണ്ണസംഖ്യ np = (n & ~(step - 1));

    // unroll by 2
    വേണ്ടി (; i < np; i += step) {
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((സ്ഥിരം _Float16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vfloat16m2_t ax1 = __riscv_vle16_v_f16m2((സ്ഥിരം _Float16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvt_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    പൂർണ്ണസംഖ്യ vl;
    വേണ്ടി (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((സ്ഥിരം _Float16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }

#endif

    വേണ്ടി (; i < n; ++i) {
        y[i] = GGML_CPU_FP16_TO_FP32(x[i]);
    }
}

ശൂന്യം ggml_cpu_fp32_to_bf16(സ്ഥിരം ദശാംശം * x, ggml_bf16_t * y, int64_t n) {
    int64_t i = 0;
    വേണ്ടി (; i < n; ++i) {
        y[i] = GGML_FP32_TO_BF16(x[i]);
    }
}

ശൂന്യം ggml_cpu_fp32_to_i32(സ്ഥിരം ദശാംശം * x, int32_t * y, int64_t n) {
    int64_t i = 0;
    വേണ്ടി (; i < n; ++i) {
        y[i] = x[i];
    }
}

ശൂന്യം ggml_cpu_bf16_to_fp32(സ്ഥിരം ggml_bf16_t * x, ദശാംശം * y, int64_t n) {
    int64_t i = 0;
#എങ്കിൽ defined(__AVX2__)
#എങ്കിൽ defined(__AVX512F__)
    വേണ്ടി (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i,
                        _mm512_castsi512_ps(
                            _mm512_slli_epi32(
                                _mm512_cvtepu16_epi32(
                                    _mm256_loadu_si256(
                                        (സ്ഥിരം __m256i *)(x + i))),
                                16)));
    }
#endif
    വേണ്ടി (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i,
                        _mm256_castsi256_ps(
                            _mm256_slli_epi32(
                                _mm256_cvtepu16_epi32(
                                    _mm_loadu_si128(
                                        (സ്ഥിരം __m128i *)(x + i))),
                                16)));
    }
#അല്ലെങ്കിലെങ്കിൽ defined(__riscv_v_intrinsic) && defined(__riscv_zvfbfmin)
    // calculate step size
    സ്ഥിരം പൂർണ്ണസംഖ്യ epr = __riscv_vsetvlmax_e16m2();
    സ്ഥിരം പൂർണ്ണസംഖ്യ step = epr * 2;
    സ്ഥിരം പൂർണ്ണസംഖ്യ np = (n & ~(step - 1));

    // unroll by 2
    വേണ്ടി (; i < np; i += step) {
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((സ്ഥിരം __bf16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vbfloat16m2_t ax1 = __riscv_vle16_v_bf16m2((സ്ഥിരം __bf16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    പൂർണ്ണസംഖ്യ vl;
    വേണ്ടി (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((സ്ഥിരം __bf16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }
#endif
    വേണ്ടി (; i < n; i++) {
        y[i] = GGML_BF16_TO_FP32(x[i]);
    }
}

പൂർണ്ണസംഖ്യ ggml_cpu_has_avx(ശൂന്യം) {
#എങ്കിൽ defined(__AVX__)
    തിരികെ 1;
#അല്ലെങ്കിൽ
    തിരികെ 0;
#endif
}

പൂർണ്ണസംഖ്യ ggml_cpu_has_avx_vnni(ശൂന്യം) {
#എങ്കിൽ defined(__AVXVNNI__)
    തിരികെ 1;
#അല്ലെങ്കിൽ
    തിരികെ 0;
#endif
}

പൂർണ്ണസംഖ്യ ggml_cpu_has_avx2(ശൂന്യം) {
#എങ്കിൽ defined(__AVX2__)
    തിരികെ 1;
#അല്ലെങ്കിൽ
    തിരികെ 0;
#endif
}

പൂർണ്ണസംഖ്യ ggml_cpu_has_avx512(ശൂന്യം) {
#എങ്കിൽ defined(__AVX512F__)
    തിരികെ 1;
#അല്ലെങ്കിൽ
    തിരികെ 0;
#endif
}

പൂർണ്ണസംഖ്യ ggml_cpu_has_avx512_vbmi(ശൂന്യം) {
#എങ്കിൽ defined(__AVX512VBMI__)
    തിരികെ 1;
#അല്ലെങ്കിൽ
    തിരികെ 0;
#endif
}

പൂർണ്ണസംഖ്യ ggml_cpu_has_avx512_vnni(ശൂന്യം) {
#എങ്കിൽ defined(__AVX512VNNI__)
    തിരികെ 1;
#അല്ലെങ്കിൽ
    തിരികെ 0;
#endif
}

പൂർണ്ണസംഖ്യ ggml_cpu_has_avx512_bf16(ശൂന്യം) {
#എങ്കിൽ defined(__AVX512BF16__)
    തിരികെ 1;
#അല്ലെങ്കിൽ
    തിരികെ 0;
#endif
}

പൂർണ്ണസംഖ്യ ggml_cpu_has_amx_int8(ശൂന്യം) {
#എങ്കിൽ defined(__AMX_INT8__)
    തിരികെ 1;
#അല്ലെങ്കിൽ
    തിരികെ 0;
#endif
}

പൂർണ്ണസംഖ്യ ggml_cpu_has_bmi2(ശൂന്യം) {
#എങ്കിൽ defined(__BMI2__)
    തിരികെ 1;
#അല്ലെങ്കിൽ
    തിരികെ 0;
#endif
}

പൂർണ്ണസംഖ്യ ggml_cpu_has_fma(ശൂന്യം) {
#എങ്കിൽ defined(__FMA__)
    തിരികെ 1;
#അല്ലെങ്കിൽ
    തിരികെ 0;
#endif
}

പൂർണ്ണസംഖ്യ ggml_cpu_has_arm_fma(ശൂന്യം) {
#എങ്കിൽ defined(__ARM_FEATURE_FMA)
    തിരികെ 1;
#അല്ലെങ്കിൽ
    തിരികെ 0;
#endif
}

പൂർണ്ണസംഖ്യ ggml_cpu_has_riscv_v(ശൂന്യം) {
#എങ്കിൽ defined(__riscv_v_intrinsic)
    തിരികെ 1;
#അല്ലെങ്കിൽ
    തിരികെ 0;
#endif
}

പൂർണ്ണസംഖ്യ ggml_cpu_get_rvv_vlen(ശൂന്യം) {
#എങ്കിൽ defined(__riscv) && defined(__riscv_v_intrinsic)
    തിരികെ ggml_riscv_arch_features.rvv_vlen;
#അല്ലെങ്കിൽ
    തിരികെ 0;
#endif
}

പൂർണ്ണസംഖ്യ ggml_cpu_has_f16c(ശൂന്യം) {
#എങ്കിൽ defined(__F16C__)
    തിരികെ 1;
#അല്ലെങ്കിൽ
    തിരികെ 0;
#endif
}

പൂർണ്ണസംഖ്യ ggml_cpu_has_fp16_va(ശൂന്യം) {
#എങ്കിൽ defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    തിരികെ 1;
#അല്ലെങ്കിൽ
    തിരികെ 0;
#endif
}

പൂർണ്ണസംഖ്യ ggml_cpu_has_wasm_simd(ശൂന്യം) {
#എങ്കിൽ defined(__wasm_simd128__)
    തിരികെ 1;
#അല്ലെങ്കിൽ
    തിരികെ 0;
#endif
}

പൂർണ്ണസംഖ്യ ggml_cpu_has_llamafile(ശൂന്യം) {
#എങ്കിൽ defined(GGML_USE_LLAMAFILE)
    തിരികെ 1;
#അല്ലെങ്കിൽ
    തിരികെ 0;
#endif
}

പൂർണ്ണസംഖ്യ ggml_cpu_has_sse3(ശൂന്യം) {
#എങ്കിൽ defined(__SSE3__)
    തിരികെ 1;
#അല്ലെങ്കിൽ
    തിരികെ 0;
#endif
}

പൂർണ്ണസംഖ്യ ggml_cpu_has_ssse3(ശൂന്യം) {
#എങ്കിൽ defined(__SSSE3__)
    തിരികെ 1;
#അല്ലെങ്കിൽ
    തിരികെ 0;
#endif
}

പൂർണ്ണസംഖ്യ ggml_cpu_has_vsx(ശൂന്യം) {
#എങ്കിൽ defined(__POWER9_VECTOR__)
    തിരികെ 1;
#അല്ലെങ്കിൽ
    തിരികെ 0;
#endif
}

പൂർണ്ണസംഖ്യ ggml_cpu_has_vxe(ശൂന്യം) {
#എങ്കിൽ defined(__VXE__) || defined(__VXE2__)
    തിരികെ 1;
#അല്ലെങ്കിൽ
    തിരികെ 0;
#endif
}

പൂർണ്ണസംഖ്യ ggml_cpu_has_neon(ശൂന്യം) {
#എങ്കിൽ defined(__ARM_ARCH) && defined(__ARM_NEON)
    തിരികെ 1;
#അല്ലെങ്കിൽ
    തിരികെ 0;
#endif
}

പൂർണ്ണസംഖ്യ ggml_cpu_has_dotprod(ശൂന്യം) {
#എങ്കിൽ defined(__ARM_ARCH) && defined(__ARM_FEATURE_DOTPROD)
    തിരികെ 1;
#അല്ലെങ്കിൽ
    തിരികെ 0;
#endif
}

പൂർണ്ണസംഖ്യ ggml_cpu_has_sve(ശൂന്യം) {
#എങ്കിൽ defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    തിരികെ 1;
#അല്ലെങ്കിൽ
    തിരികെ 0;
#endif
}

പൂർണ്ണസംഖ്യ ggml_cpu_has_matmul_int8(ശൂന്യം) {
#എങ്കിൽ defined(__ARM_ARCH) && defined(__ARM_FEATURE_MATMUL_INT8)
    തിരികെ 1;
#അല്ലെങ്കിൽ
    തിരികെ 0;
#endif
}

പൂർണ്ണസംഖ്യ ggml_cpu_get_sve_cnt(ശൂന്യം) {
#എങ്കിൽ defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    തിരികെ ggml_arm_arch_features.sve_cnt;
#അല്ലെങ്കിൽ
    തിരികെ 0;
#endif
}

പൂർണ്ണസംഖ്യ ggml_cpu_has_sme(ശൂന്യം) {
#എങ്കിൽ defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME)
    തിരികെ 1;
#അല്ലെങ്കിൽ
    തിരികെ 0;
#endif
}

പൂർണ്ണസംഖ്യ ggml_cpu_has_sme2(ശൂന്യം) {
#എങ്കിൽ defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME2)
    തിരികെ 1;
#അല്ലെങ്കിൽ
    തിരികെ 0;
#endif
}

ശൂന്യം ggml_cpu_init(ശൂന്യം) {
    // needed to initialize ggml_time
    {
        ഘടന ggml_init_params params = { 0, ശൂന്യ, false };
        ഘടന ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    ggml_critical_section_start();

    സ്ഥിതം bool is_first_call = true;

    എങ്കിൽ (is_first_call) {
        // initialize GELU, Quick GELU, SILU and EXP F32 tables
        {
            സ്ഥിരം uint64_t t_start = ggml_time_us(); UNUSED(t_start);

            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < (1 << 16); ++i) {
                സംഘം {
                    uint16_t u16;
                    ggml_fp16_t fp16;
                } u = {i};
                ദശാംശം f = GGML_COMPUTE_FP16_TO_FP32(u.fp16);
                ggml_table_f32_f16[i] = f;
                ggml_table_gelu_f16[i] = GGML_CPU_FP32_TO_FP16(ggml_gelu_f32(f));
                ggml_table_gelu_quick_f16[i] = GGML_CPU_FP32_TO_FP16(ggml_gelu_quick_f32(f));
            }

            // initialize E8M0 half table (256 entries)
            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < (1 << 8); ++i) {
                ggml_table_f32_e8m0_half[i] = GGML_E8M0_TO_FP32_HALF(i);
            }

            // initialize UE4M3 table (256 entries)
            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < (1 << 8); ++i) {
                ggml_table_f32_ue4m3[i] = ggml_ue4m3_to_fp32(i);
            }

            സ്ഥിരം uint64_t t_end = ggml_time_us(); UNUSED(t_end);

            GGML_PRINT_DEBUG("%s: GELU, Quick GELU, SILU and EXP tables initialized in %f ms\n", __func__, (t_end - t_start)/1000.0);

#ifdef GGML_USE_OPENMP
            //if (!getenv("OMP_WAIT_POLICY")) {
            //    // set the wait policy to active, so that OpenMP threads don't sleep
            //    setenv("OMP_WAIT_POLICY", "active", 0)
            //}

            എങ്കിൽ (!getenv("KMP_BLOCKTIME")) {
                // set the time to wait before sleeping a thread
                // this is less aggressive than setting the wait policy to active, but should achieve similar results in most cases
#ifdef _WIN32
                _putenv_s("KMP_BLOCKTIME", "200"); // 200ms
#അല്ലെങ്കിൽ
                setenv("KMP_BLOCKTIME", "200", 0); // 200ms
#endif
            }
#endif
        }

#എങ്കിൽ defined(__ARM_ARCH)
        ggml_init_arm_arch_features();
#endif

#എങ്കിൽ defined(__riscv)
        ggml_init_riscv_arch_features();
#endif

        {
            സ്ഥിരം അക്ഷരം * env = getenv("GGML_CPU_DISABLE_FUSION");
            ggml_cpu_disable_fusion = (env != ശൂന്യ && atoi(env) == 1);
        }

        is_first_call = false;
    }

    ggml_critical_section_end();
}
