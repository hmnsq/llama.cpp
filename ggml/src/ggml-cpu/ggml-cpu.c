#define _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#define _USE_MATH_DEFINES // For M_PI on MSVC

#அந்தர்பாவ "ggml-backend-impl.h"
#அந்தர்பாவ "ggml-backend.h"
#அந்தர்பாவ "traits.h"
#அந்தர்பாவ "ggml-cpu-impl.h"
#அந்தர்பாவ "ggml-impl.h"
#அந்தர்பாவ "quants.h"
#அந்தர்பாவ "ggml-threading.h"
#அந்தர்பாவ "unary-ops.h"
#அந்தர்பாவ "binary-ops.h"
#அந்தர்பாவ "vec.h"
#அந்தர்பாவ "ops.h"
#அந்தர்பாவ "ggml.h"
#அந்தர்பாவ "common.h"
#அந்தர்பாவ "tiled/tiled.h"

#எனில் defined(_MSC_VER) || defined(__MINGW32__)
#அந்தர்பாவ <malloc.h> // using malloc.h with MSC/MINGW
#அல்லதெனில் !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#அந்தர்பாவ <alloca.h>
#endif

#அந்தர்பாவ <assert.h>
#அந்தர்பாவ <errno.h>
#அந்தர்பாவ <time.h>
#அந்தர்பாவ <math.h>
#அந்தர்பாவ <stdlib.h>
#அந்தர்பாவ <string.h>
#அந்தர்பாவ <stdint.h>
#அந்தர்பாவ <inttypes.h>
#அந்தர்பாவ <stdio.h>
#அந்தர்பாவ <float.h>
#அந்தர்பாவ <limits.h>
#அந்தர்பாவ <stdarg.h>
#அந்தர்பாவ <signal.h>
#எனில் defined(__gnu_linux__)
#அந்தர்பாவ <syscall.h>
#endif

#ifdef GGML_USE_OPENMP
#அந்தர்பாவ <omp.h>
#endif

#எனில் defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_MATMUL_INT8)
#undef GGML_USE_LLAMAFILE
#endif

#ifdef GGML_USE_LLAMAFILE
#அந்தர்பாவ "llamafile/sgemm.h"
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
#    include "spacemit/ime.h"
#endif

// Note: once we move threading into a separate C++ file
// will use std::hardware_destructive_interference_size instead of hardcoding it here
// and we'll use C++ attribute syntax.
#define GGML_CACHE_வரி  64

#எனில் defined(__clang__) || defined(__GNUC__)
#define GGML_CACHE_ALIGN __attribute__((aligned(GGML_CACHE_வரி)))
#endif

#எனில் defined(__has_feature)
#எனில் __has_feature(thread_sanitizer)
#define GGML_TSAN_ENABLED 1
#endif
#இல்லையெனில்  // __has_feature
#எனில் defined(__SANITIZE_THREAD__)
#define GGML_TSAN_ENABLED 1
#endif
#endif // __has_feature

#define UNUSED GGML_UNUSED
#define SWAP(x, y, T) செய் { T SWAP = x; (x) = y; (y) = SWAP; } போது (0)

// precomputed f32 table for f16 (256 KB) (simd-mappings.h)
தசம ggml_table_f32_f16[1 << 16];

// precomputed f32 table for e8m0 half (1 KB) (simd-mappings.h)
தசம ggml_table_f32_e8m0_half[1 << 8];

// precomputed f32 table for ue4m3 (1 KB) (simd-mappings.h)
தசம ggml_table_f32_ue4m3[1 << 8];

#எனில் defined(__ARM_ARCH)
கட்டமைப்பு ggml_arm_arch_features_type {
    முழு sve_cnt;
} ggml_arm_arch_features = { 0 };
#endif

#எனில் defined(__riscv)
கட்டமைப்பு ggml_riscv_arch_features_type {
    முழு rvv_vlen;
} ggml_riscv_arch_features = { 0 };
#endif

#எனில் defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#அந்தர்பாவ <windows.h>

#எனில் defined(_MSC_VER) && !defined(__clang__)
#define GGML_CACHE_ALIGN __declspec(align(GGML_CACHE_வரி))

வகைவரை நிலையற்ற LONG atomic_int;
வகைவரை atomic_int atomic_bool;
வகைவரை atomic_int atomic_flag;

#define ATOMIC_FLAG_INIT 0

வகைவரை எண்ணிக்கை {
    memory_order_relaxed,
    memory_order_consume,
    memory_order_acquire,
    memory_order_release,
    memory_order_acq_rel,
    memory_order_seq_cst
} memory_order;

நிலையான வெறுமை atomic_store(atomic_int * ptr, LONG val) {
    InterlockedExchange(ptr, val);
}
நிலையான வெறுமை atomic_store_explicit(atomic_int * ptr, LONG val, memory_order mo) {
    // TODO: add support for explicit memory order
    InterlockedExchange(ptr, val);
}
நிலையான LONG atomic_load(atomic_int * ptr) {
    திரும்பு InterlockedCompareExchange(ptr, 0, 0);
}
நிலையான LONG atomic_load_explicit(atomic_int * ptr, memory_order mo) {
    // TODO: add support for explicit memory order
    திரும்பு InterlockedCompareExchange(ptr, 0, 0);
}
நிலையான LONG atomic_fetch_add(atomic_int * ptr, LONG inc) {
    திரும்பு InterlockedExchangeAdd(ptr, inc);
}
நிலையான LONG atomic_fetch_add_explicit(atomic_int * ptr, LONG inc, memory_order mo) {
    // TODO: add support for explicit memory order
    திரும்பு InterlockedExchangeAdd(ptr, inc);
}
நிலையான atomic_bool atomic_flag_test_and_set(atomic_flag * ptr) {
    திரும்பு InterlockedExchange(ptr, 1);
}
நிலையான வெறுமை atomic_flag_clear(atomic_flag * ptr) {
    InterlockedExchange(ptr, 0);
}
நிலையான வெறுமை atomic_thread_fence(memory_order mo) {
    MemoryBarrier();
}
#இல்லையெனில் // clang
#அந்தர்பாவ <stdatomic.h>
#endif

வகைவரை HANDLE pthread_t;

வகைவரை DWORD thread_ret_t;
நிலையான முழு pthread_create(pthread_t * out, வெறுமை * unused, thread_ret_t(*func)(வெறுமை *), வெறுமை * arg) {
    (வெறுமை) unused;
    HANDLE handle = CreateThread(வெற்று, 0, (LPTHREAD_START_ROUTINE) func, arg, 0, வெற்று);
    எனில் (handle == வெற்று)
    {
        திரும்பு EAGAIN;
    }

    *out = handle;
    திரும்பு 0;
}

நிலையான முழு pthread_join(pthread_t thread, வெறுமை * unused) {
    (வெறுமை) unused;
    முழு ret = (முழு) WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    திரும்பு ret;
}

நிலையான முழு sched_yield (வெறுமை) {
    Sleep (0);
    திரும்பு 0;
}
#இல்லையெனில்

#அந்தர்பாவ <pthread.h>
#அந்தர்பாவ <stdatomic.h>
#அந்தர்பாவ <sched.h>
#எனில் defined(__FreeBSD__)
#அந்தர்பாவ <pthread_np.h>
#endif

வகைவரை வெறுமை * thread_ret_t;

#அந்தர்பாவ <sys/types.h>
#அந்தர்பாவ <sys/stat.h>
#அந்தர்பாவ <unistd.h>

#endif

வகைவரை pthread_t ggml_thread_t;

#define GGML_THREADPOOL_N_THREADS_MASK (0xffffU)
#define GGML_THREADPOOL_N_THREADS_BITS (16)

#எனில் defined(__APPLE__)
#அந்தர்பாவ <unistd.h>
#அந்தர்பாவ <mach/mach.h>
#அந்தர்பாவ <TargetConditionals.h>
#endif

நிலையான மாறிலி கட்டமைப்பு ggml_type_traits_cpu type_traits_cpu[GGML_TYPE_மொத்தம்] = {
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
#எனில் defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#இல்லையெனில்
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q4_1] = {
        .from_float               = quantize_row_q4_1,
        .vec_dot                  = ggml_vec_dot_q4_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
#எனில் defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#இல்லையெனில்
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
#எனில் defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#இல்லையெனில்
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
#எனில் defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#இல்லையெனில்
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
#எனில் defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#இல்லையெனில்
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_IQ2_XXS] = {
        .from_float               = வெற்று,
        .vec_dot                  = ggml_vec_dot_iq2_xxs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_XS] = {
        .from_float               = வெற்று,
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
        .from_float               = வெற்று,
        .vec_dot                  = ggml_vec_dot_iq1_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_M] = {
        .from_float               = வெற்று,
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

மாறிலி கட்டமைப்பு ggml_type_traits_cpu * ggml_get_type_traits_cpu(எண்ணிக்கை ggml_type type) {
    திரும்பு &type_traits_cpu[type];
}

//
// Threading defs
//

வகைவரை pthread_t          ggml_thread_t;

#எனில் defined(_WIN32)

வகைவரை CONDITION_VARIABLE ggml_cond_t;
வகைவரை SRWLOCK            ggml_mutex_t;

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

#இல்லையெனில்

வகைவரை pthread_cond_t     ggml_cond_t;
வகைவரை pthread_mutex_t    ggml_mutex_t;

#define ggml_mutex_init(m)          pthread_mutex_init(m, வெற்று)
#define ggml_mutex_destroy(m)       pthread_mutex_destroy(m)
#define ggml_mutex_lock(m)          pthread_mutex_lock(m)
#define ggml_mutex_unlock(m)        pthread_mutex_unlock(m)
#define ggml_mutex_lock_shared(m)   pthread_mutex_lock(m)
#define ggml_mutex_unlock_shared(m) pthread_mutex_unlock(m)

#define ggml_lock_init(x)    UNUSED(x)
#define ggml_lock_destroy(x) UNUSED(x)
#எனில் defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64) && !defined(_M_ARM64EC))
#define ggml_lock_lock(x)    _mm_pause()
#இல்லையெனில்
#define ggml_lock_lock(x)    UNUSED(x)
#endif
#define ggml_lock_unlock(x)  UNUSED(x)

#define GGML_LOCK_INITIALIZER 0
#define ggml_cond_init(c)      pthread_cond_init(c, வெற்று)
#define ggml_cond_destroy(c)   pthread_cond_destroy(c)
#define ggml_cond_wait(c, m)   pthread_cond_wait(c, m)
#define ggml_cond_broadcast(c) pthread_cond_broadcast(c)

#define ggml_thread_create pthread_create
#define ggml_thread_join   pthread_join

#endif

// Threadpool def
கட்டமைப்பு ggml_threadpool {
    ggml_mutex_t mutex;       // mutex for cond.var
    ggml_cond_t  cond;        // cond.var for waiting for new work

    கட்டமைப்பு ggml_cgraph * cgraph;
    கட்டமைப்பு ggml_cplan  * cplan;

    // synchronization primitives
    atomic_int n_graph;       // updated when there is work to be done (i.e each graph) holds graph and active thread counts.
    atomic_int GGML_CACHE_ALIGN n_barrier;
    atomic_int GGML_CACHE_ALIGN n_barrier_passed;
    atomic_int GGML_CACHE_ALIGN current_chunk; // currently processing chunk during Mat_Mul, shared between all the threads.

    // these are atomic as an annotation for thread-sanitizer
    atomic_bool stop;         // Used for stopping the threadpool altogether
    atomic_bool pause;        // Used for pausing the threadpool or individual threads
    atomic_int  abort;        // Used for aborting processing of a graph

    கட்டமைப்பு ggml_compute_state * workers;   // per thread state
    முழு          n_threads;   // Number of threads in the pool
    int32_t      prio;        // Scheduling priority
    uint32_t     poll;        // Polling level (0 - no polling)

    எண்ணிக்கை ggml_status ec;
};

// Per-thread state
கட்டமைப்பு ggml_compute_state {
#ifndef GGML_USE_OPENMP
    ggml_thread_t thrd;
    முழு  last_graph;
    bool pending;
#endif
    bool cpumask[GGML_MAX_N_THREADS];
    கட்டமைப்பு ggml_threadpool * threadpool;
    முழு ith;
};

// Helpers for polling loops
#எனில் defined(__aarch64__) && ( defined(__clang__) || defined(__GNUC__) )
நிலையான உள்வரி வெறுமை ggml_thread_cpu_relax(வெறுமை) {
    __asm__ நிலையற்ற("yield" ::: "memory");
}
#அல்லதெனில் defined(__x86_64__)
நிலையான உள்வரி வெறுமை ggml_thread_cpu_relax(வெறுமை) {
    _mm_pause();
}
#அல்லதெனில் defined(__riscv)
நிலையான உள்வரி வெறுமை ggml_thread_cpu_relax(வெறுமை) {
    #ifdef __riscv_zihintpause
        __asm__ __volatile__ ("pause");
    #இல்லையெனில்
        /* Encoding of the pause instruction */
        __asm__ __volatile__ (".4byte 0x100000F");
    #endif
}
#இல்லையெனில்
நிலையான உள்வரி வெறுமை ggml_thread_cpu_relax(வெறுமை) {;}
#endif

//
// NUMA support
//

#define GGML_NUMA_MAX_NODES 8
#define GGML_NUMA_MAX_CPUS 512

கட்டமைப்பு ggml_numa_முனை {
    uint32_t cpus[GGML_NUMA_MAX_CPUS]; // hardware threads on this node
    uint32_t n_cpus;
};

கட்டமைப்பு ggml_numa_nodes {
    எண்ணிக்கை ggml_numa_strategy numa_strategy;
    கட்டமைப்பு ggml_numa_முனை nodes[GGML_NUMA_MAX_NODES];
    uint32_t n_nodes;
    uint32_t total_cpus; // hardware threads on system
    uint32_t current_முனை; // node on which main process is execting
#எனில் defined(__gnu_linux__)
    cpu_set_t cpuset; // cpuset from numactl
#இல்லையெனில்
    uint32_t cpuset; // no NUMA support outside of Linux at this time. Use a portable datatype
#endif
};

//
// ggml state
//

கட்டமைப்பு ggml_state {
    கட்டமைப்பு ggml_numa_nodes numa;
};

நிலையான கட்டமைப்பு ggml_state g_state = {0};

வெறுமை ggml_barrier(கட்டமைப்பு ggml_threadpool * tp) {
    முழு n_threads = atomic_load_explicit(&tp->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK;
    எனில் (n_threads == 1) {
        திரும்பு;
    }

#ifdef GGML_USE_OPENMP
    #pragma omp barrier
#இல்லையெனில்
    முழு n_passed = atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed);

    // enter barrier (full seq-cst fence)
    முழு n_barrier = atomic_fetch_add_explicit(&tp->n_barrier, 1, memory_order_seq_cst);

    எனில் (n_barrier == (n_threads - 1)) {
        // last thread
        atomic_store_explicit(&tp->n_barrier, 0, memory_order_relaxed);

        // exit barrier (full seq-cst fence)
        atomic_fetch_add_explicit(&tp->n_barrier_passed, 1, memory_order_seq_cst);
        திரும்பு;
    }

    // wait for other threads
    போது (atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed) == n_passed) {
        ggml_thread_cpu_relax();
    }

    // exit barrier (full seq-cst fence)
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&tp->n_barrier_passed, 0, memory_order_seq_cst);
    #இல்லையெனில்
    atomic_thread_fence(memory_order_seq_cst);
    #endif
#endif
}

வெறுமை ggml_threadpool_chunk_set(கட்டமைப்பு ggml_threadpool * tp, முழு மதிப்பு) {
    atomic_store_explicit(&tp->current_chunk, மதிப்பு, memory_order_relaxed);
}

முழு ggml_threadpool_chunk_add(கட்டமைப்பு ggml_threadpool * tp, முழு மதிப்பு) {
    திரும்பு atomic_fetch_add_explicit(&tp->current_chunk, மதிப்பு, memory_order_relaxed);
}

#எனில் defined(__gnu_linux__)
நிலையான cpu_set_t ggml_get_numa_affinity(வெறுமை) {
    cpu_set_t cpuset;
    pthread_t thread;
    thread = pthread_self();
    CPU_ZERO(&cpuset);
    pthread_getaffinity_np(thread, அளவு(cpu_set_t), &cpuset);
    திரும்பு cpuset;
}
#இல்லையெனில்
நிலையான uint32_t ggml_get_numa_affinity(வெறுமை) {
    திரும்பு 0; // no NUMA support
}
#endif

வெறுமை ggml_numa_init(எண்ணிக்கை ggml_numa_strategy numa_flag) {
    எனில் (g_state.numa.n_nodes > 0) {
        fprintf(stderr, "ggml_numa_init: NUMA already initialized\n");

        திரும்பு;
    }

#எனில் defined(__gnu_linux__)
    கட்டமைப்பு stat st;
    எழுத்து path[256];
    முழு rv;

    // set numa scheme
    g_state.numa.numa_strategy = numa_flag;

    GGML_PRINT_DEBUG("numa strategy %u\n",g_state.numa.numa_strategy);

    g_state.numa.cpuset = ggml_get_numa_affinity();

    // enumerate nodes
    போது (g_state.numa.n_nodes < GGML_NUMA_MAX_NODES) {
        rv = snprintf(path, அளவு(path), "/sys/devices/system/node/node%u", g_state.numa.n_nodes);
        GGML_ASSERT(rv > 0 && (unsigned)rv < அளவு(path));
        எனில் (stat(path, &st) != 0) { நிறுத்து; }
        ++g_state.numa.n_nodes;
    }

    // enumerate CPUs
    போது (g_state.numa.total_cpus < GGML_NUMA_MAX_CPUS) {
        rv = snprintf(path, அளவு(path), "/sys/devices/system/cpu/cpu%u", g_state.numa.total_cpus);
        GGML_ASSERT(rv > 0 && (unsigned)rv < அளவு(path));
        எனில் (stat(path, &st) != 0) { நிறுத்து; }
        ++g_state.numa.total_cpus;
    }

    GGML_PRINT_DEBUG("found %u numa nodes, %u CPUs\n", g_state.numa.n_nodes, g_state.numa.total_cpus);

    // figure out which node we're on
    uint current_cpu;
    முழு getcpu_ret = 0;
#எனில் __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ > 33) || defined(__COSMOPOLITAN__)
    getcpu_ret = getcpu(&current_cpu, &g_state.numa.current_முனை);
#இல்லையெனில்
    // old glibc doesn't have a wrapper for this call. Fall back on direct syscall
#   எனில் !defined(SYS_getcpu) && defined(SYS_get_cpu)
#       define SYS_getcpu SYS_get_cpu // some older glibc versions use this name
#   endif
    getcpu_ret = syscall(SYS_getcpu, &current_cpu, &g_state.numa.current_முனை);
#endif

    எனில் (g_state.numa.n_nodes < 1 || g_state.numa.total_cpus < 1 || getcpu_ret != 0) {
        g_state.numa.n_nodes = 0;
        திரும்பு;
    }

    GGML_PRINT_DEBUG("found our process on numa node %u, CPU %u\n", g_state.numa.current_முனை, current_cpu);

    க்கு (uint32_t n = 0; n < g_state.numa.n_nodes; ++n) {
        கட்டமைப்பு ggml_numa_முனை * முனை = &g_state.numa.nodes[n];
        GGML_PRINT_DEBUG("CPUs on node %u:", n);
        முனை->n_cpus = 0;
        க்கு (uint32_t c = 0; c < g_state.numa.total_cpus; ++c) {
            rv = snprintf(path, அளவு(path), "/sys/devices/system/node/node%u/cpu%u", n, c);
            GGML_ASSERT(rv > 0 && (unsigned)rv < அளவு(path));
            எனில் (stat(path, &st) == 0) {
                முனை->cpus[முனை->n_cpus++] = c;
                GGML_PRINT_DEBUG(" %u", c);
            }
        }
        GGML_PRINT_DEBUG("\n");
    }

    எனில் (ggml_is_numa()) {
        FILE *fptr = fopen("/proc/sys/kernel/numa_balancing", "r");
        எனில் (fptr != வெற்று) {
            எழுத்து buf[42];
            எனில் (fgets(buf, அளவு(buf), fptr) && strncmp(buf, "0\n", அளவு(buf)) != 0) {
                GGML_LOG_WARN("/proc/sys/kernel/numa_balancing is enabled, this has been observed to impair performance\n");
            }
            fclose(fptr);
        }
    }
#இல்லையெனில்
    UNUSED(numa_flag);
    // TODO
#endif
}

bool ggml_is_numa(வெறுமை) {
    திரும்பு g_state.numa.n_nodes > 1;
}

#எனில் defined(__ARM_ARCH)
#எனில் defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
#அந்தர்பாவ <arm_sve.h>
நிலையான வெறுமை ggml_init_arm_arch_features(வெறுமை) {
    ggml_arm_arch_features.sve_cnt = svcntb();
}
#இல்லையெனில்
நிலையான வெறுமை ggml_init_arm_arch_features(வெறுமை) {}
#endif
#endif // __ARM_ARCH

#எனில் defined(__riscv) && defined(__riscv_v_intrinsic)
#அந்தர்பாவ <riscv_vector.h>
நிலையான வெறுமை ggml_init_riscv_arch_features(வெறுமை) {
    ggml_riscv_arch_features.rvv_vlen = __riscv_vlenb();
}
#இல்லையெனில்
நிலையான வெறுமை ggml_init_riscv_arch_features(வெறுமை) {}
#endif

கட்டமைப்பு ggml_tensor * ggml_new_i32(கட்டமைப்பு ggml_context * ctx, int32_t மதிப்பு) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

    ggml_set_i32(result, மதிப்பு);

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_new_f32(கட்டமைப்பு ggml_context * ctx, தசம மதிப்பு) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    கட்டமைப்பு ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    ggml_set_f32(result, மதிப்பு);

    திரும்பு result;
}

கட்டமைப்பு ggml_tensor * ggml_set_i32 (கட்டமைப்பு ggml_tensor * tensor, int32_t மதிப்பு) {
    மாறிலி முழு n     = ggml_nrows(tensor);
    மாறிலி முழு nc    = tensor->ne[0];
    மாறிலி size_t n1 = tensor->nb[1];

    எழுத்து * மாறிலி data = tensor->data;

    தேர்வு (tensor->type) {
        நிலை GGML_TYPE_I8:
            {
                ப்ரதிஜ்ஞா(tensor->nb[0] == அளவு(int8_t));
                க்கு (முழு i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), மதிப்பு);
                }
            } நிறுத்து;
        நிலை GGML_TYPE_I16:
            {
                ப்ரதிஜ்ஞா(tensor->nb[0] == அளவு(int16_t));
                க்கு (முழு i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), மதிப்பு);
                }
            } நிறுத்து;
        நிலை GGML_TYPE_I32:
            {
                ப்ரதிஜ்ஞா(tensor->nb[0] == அளவு(int32_t));
                க்கு (முழு i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), மதிப்பு);
                }
            } நிறுத்து;
        நிலை GGML_TYPE_F16:
            {
                ப்ரதிஜ்ஞா(tensor->nb[0] == அளவு(ggml_fp16_t));
                க்கு (முழு i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), GGML_CPU_FP32_TO_FP16(மதிப்பு));
                }
            } நிறுத்து;
        நிலை GGML_TYPE_BF16:
            {
                ப்ரதிஜ்ஞா(tensor->nb[0] == அளவு(ggml_fp16_t));
                க்கு (முழு i = 0; i < n; i++) {
                    ggml_vec_set_bf16(nc, (ggml_bf16_t *)(data + i*n1), GGML_FP32_TO_BF16(மதிப்பு));
                }
            } நிறுத்து;
        நிலை GGML_TYPE_F32:
            {
                ப்ரதிஜ்ஞா(tensor->nb[0] == அளவு(தசம));
                க்கு (முழு i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (தசம *)(data + i*n1), மதிப்பு);
                }
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }

    திரும்பு tensor;
}

கட்டமைப்பு ggml_tensor * ggml_set_f32(கட்டமைப்பு ggml_tensor * tensor, தசம மதிப்பு) {
    மாறிலி முழு n     = ggml_nrows(tensor);
    மாறிலி முழு nc    = tensor->ne[0];
    மாறிலி size_t n1 = tensor->nb[1];

    எழுத்து * மாறிலி data = tensor->data;

    தேர்வு (tensor->type) {
        நிலை GGML_TYPE_I8:
            {
                ப்ரதிஜ்ஞா(tensor->nb[0] == அளவு(int8_t));
                க்கு (முழு i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), மதிப்பு);
                }
            } நிறுத்து;
        நிலை GGML_TYPE_I16:
            {
                ப்ரதிஜ்ஞா(tensor->nb[0] == அளவு(int16_t));
                க்கு (முழு i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), மதிப்பு);
                }
            } நிறுத்து;
        நிலை GGML_TYPE_I32:
            {
                ப்ரதிஜ்ஞா(tensor->nb[0] == அளவு(int32_t));
                க்கு (முழு i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), மதிப்பு);
                }
            } நிறுத்து;
        நிலை GGML_TYPE_F16:
            {
                ப்ரதிஜ்ஞா(tensor->nb[0] == அளவு(ggml_fp16_t));
                க்கு (முழு i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), GGML_CPU_FP32_TO_FP16(மதிப்பு));
                }
            } நிறுத்து;
        நிலை GGML_TYPE_BF16:
            {
                ப்ரதிஜ்ஞா(tensor->nb[0] == அளவு(ggml_bf16_t));
                க்கு (முழு i = 0; i < n; i++) {
                    ggml_vec_set_bf16(nc, (ggml_bf16_t *)(data + i*n1), GGML_FP32_TO_BF16(மதிப்பு));
                }
            } நிறுத்து;
        நிலை GGML_TYPE_F32:
            {
                ப்ரதிஜ்ஞா(tensor->nb[0] == அளவு(தசம));
                க்கு (முழு i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (தசம *)(data + i*n1), மதிப்பு);
                }
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }

    திரும்பு tensor;
}

int32_t ggml_get_i32_1d(மாறிலி கட்டமைப்பு ggml_tensor * tensor, முழு i) {
    எனில் (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        திரும்பு ggml_get_i32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    தேர்வு (tensor->type) {
        நிலை GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == அளவு(int8_t));
                திரும்பு ((int8_t *)(tensor->data))[i];
            }
        நிலை GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == அளவு(int16_t));
                திரும்பு ((int16_t *)(tensor->data))[i];
            }
        நிலை GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == அளவு(int32_t));
                திரும்பு ((int32_t *)(tensor->data))[i];
            }
        நிலை GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == அளவு(ggml_fp16_t));
                திரும்பு GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *)(tensor->data))[i]);
            }
        நிலை GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == அளவு(ggml_bf16_t));
                திரும்பு GGML_BF16_TO_FP32(((ggml_bf16_t *)(tensor->data))[i]);
            }
        நிலை GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == அளவு(தசம));
                திரும்பு ((தசம *)(tensor->data))[i];
            }
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

வெறுமை ggml_set_i32_1d(மாறிலி கட்டமைப்பு ggml_tensor * tensor, முழு i, int32_t மதிப்பு) {
    எனில் (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_set_i32_nd(tensor, id[0], id[1], id[2], id[3], மதிப்பு);
        திரும்பு;
    }
    தேர்வு (tensor->type) {
        நிலை GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == அளவு(int8_t));
                ((int8_t *)(tensor->data))[i] = மதிப்பு;
            } நிறுத்து;
        நிலை GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == அளவு(int16_t));
                ((int16_t *)(tensor->data))[i] = மதிப்பு;
            } நிறுத்து;
        நிலை GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == அளவு(int32_t));
                ((int32_t *)(tensor->data))[i] = மதிப்பு;
            } நிறுத்து;
        நிலை GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == அளவு(ggml_fp16_t));
                ((ggml_fp16_t *)(tensor->data))[i] = GGML_CPU_FP32_TO_FP16(மதிப்பு);
            } நிறுத்து;
        நிலை GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == அளவு(ggml_bf16_t));
                ((ggml_bf16_t *)(tensor->data))[i] = GGML_FP32_TO_BF16(மதிப்பு);
            } நிறுத்து;
        நிலை GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == அளவு(தசம));
                ((தசம *)(tensor->data))[i] = மதிப்பு;
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

int32_t ggml_get_i32_nd(மாறிலி கட்டமைப்பு ggml_tensor * tensor, முழு i0, முழு i1, முழு i2, முழு i3) {
    வெறுமை * data   = (எழுத்து *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    தேர்வு (tensor->type) {
        நிலை GGML_TYPE_I8:
            திரும்பு ((int8_t *) data)[0];
        நிலை GGML_TYPE_I16:
            திரும்பு ((int16_t *) data)[0];
        நிலை GGML_TYPE_I32:
            திரும்பு ((int32_t *) data)[0];
        நிலை GGML_TYPE_F16:
            திரும்பு GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *) data)[0]);
        நிலை GGML_TYPE_BF16:
            திரும்பு GGML_BF16_TO_FP32(((ggml_bf16_t *) data)[0]);
        நிலை GGML_TYPE_F32:
            திரும்பு ((தசம *) data)[0];
        இயல்பு:
            GGML_ABORT("fatal error");
    }
}

வெறுமை ggml_set_i32_nd(மாறிலி கட்டமைப்பு ggml_tensor * tensor, முழு i0, முழு i1, முழு i2, முழு i3, int32_t மதிப்பு) {
    வெறுமை * data   = (எழுத்து *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    தேர்வு (tensor->type) {
        நிலை GGML_TYPE_I8:
            {
                ((int8_t *)(data))[0] = மதிப்பு;
            } நிறுத்து;
        நிலை GGML_TYPE_I16:
            {
                ((int16_t *)(data))[0] = மதிப்பு;
            } நிறுத்து;
        நிலை GGML_TYPE_I32:
            {
                ((int32_t *)(data))[0] = மதிப்பு;
            } நிறுத்து;
        நிலை GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(data))[0] = GGML_CPU_FP32_TO_FP16(மதிப்பு);
            } நிறுத்து;
        நிலை GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(data))[0] = GGML_FP32_TO_BF16(மதிப்பு);
            } நிறுத்து;
        நிலை GGML_TYPE_F32:
            {
                ((தசம *)(data))[0] = மதிப்பு;
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

தசம ggml_get_f32_1d(மாறிலி கட்டமைப்பு ggml_tensor * tensor, முழு i) {
    எனில் (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        திரும்பு ggml_get_f32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    தேர்வு (tensor->type) {
        நிலை GGML_TYPE_I8:
            {
                திரும்பு ((int8_t *)(tensor->data))[i];
            }
        நிலை GGML_TYPE_I16:
            {
                திரும்பு ((int16_t *)(tensor->data))[i];
            }
        நிலை GGML_TYPE_I32:
            {
                திரும்பு ((int32_t *)(tensor->data))[i];
            }
        நிலை GGML_TYPE_F16:
            {
                திரும்பு GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *)(tensor->data))[i]);
            }
        நிலை GGML_TYPE_BF16:
            {
                திரும்பு GGML_BF16_TO_FP32(((ggml_bf16_t *)(tensor->data))[i]);
            }
        நிலை GGML_TYPE_F32:
            {
                திரும்பு ((தசம *)(tensor->data))[i];
            }
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

வெறுமை ggml_set_f32_1d(மாறிலி கட்டமைப்பு ggml_tensor * tensor, முழு i, தசம மதிப்பு) {
    எனில் (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_set_f32_nd(tensor, id[0], id[1], id[2], id[3], மதிப்பு);
        திரும்பு;
    }
    தேர்வு (tensor->type) {
        நிலை GGML_TYPE_I8:
            {
                ((int8_t *)(tensor->data))[i] = மதிப்பு;
            } நிறுத்து;
        நிலை GGML_TYPE_I16:
            {
                ((int16_t *)(tensor->data))[i] = மதிப்பு;
            } நிறுத்து;
        நிலை GGML_TYPE_I32:
            {
                ((int32_t *)(tensor->data))[i] = மதிப்பு;
            } நிறுத்து;
        நிலை GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(tensor->data))[i] = GGML_CPU_FP32_TO_FP16(மதிப்பு);
            } நிறுத்து;
        நிலை GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(tensor->data))[i] = GGML_FP32_TO_BF16(மதிப்பு);
            } நிறுத்து;
        நிலை GGML_TYPE_F32:
            {
                ((தசம *)(tensor->data))[i] = மதிப்பு;
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

தசம ggml_get_f32_nd(மாறிலி கட்டமைப்பு ggml_tensor * tensor, முழு i0, முழு i1, முழு i2, முழு i3) {
    வெறுமை * data   = (எழுத்து *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    தேர்வு (tensor->type) {
        நிலை GGML_TYPE_I8:
            திரும்பு ((int8_t *) data)[0];
        நிலை GGML_TYPE_I16:
            திரும்பு ((int16_t *) data)[0];
        நிலை GGML_TYPE_I32:
            திரும்பு ((int32_t *) data)[0];
        நிலை GGML_TYPE_F16:
            திரும்பு GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *) data)[0]);
        நிலை GGML_TYPE_BF16:
            திரும்பு GGML_BF16_TO_FP32(((ggml_bf16_t *) data)[0]);
        நிலை GGML_TYPE_F32:
            திரும்பு ((தசம *) data)[0];
        இயல்பு:
            GGML_ABORT("fatal error");
    }
}

வெறுமை ggml_set_f32_nd(மாறிலி கட்டமைப்பு ggml_tensor * tensor, முழு i0, முழு i1, முழு i2, முழு i3, தசம மதிப்பு) {
    வெறுமை * data   = (எழுத்து *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    தேர்வு (tensor->type) {
        நிலை GGML_TYPE_I8:
            {
                ((int8_t *)(data))[0] = மதிப்பு;
            } நிறுத்து;
        நிலை GGML_TYPE_I16:
            {
                ((int16_t *)(data))[0] = மதிப்பு;
            } நிறுத்து;
        நிலை GGML_TYPE_I32:
            {
                ((int32_t *)(data))[0] = மதிப்பு;
            } நிறுத்து;
        நிலை GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(data))[0] = GGML_CPU_FP32_TO_FP16(மதிப்பு);
            } நிறுத்து;
        நிலை GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(data))[0] = GGML_FP32_TO_BF16(மதிப்பு);
            } நிறுத்து;
        நிலை GGML_TYPE_F32:
            {
                ((தசம *)(data))[0] = மதிப்பு;
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

////////////////////////////////////////////////////////////////////////////////

// ggml_compute_forward_mul_mat

நிலையான வெறுமை ggml_compute_forward_mul_mat_one_chunk(
    மாறிலி கட்டமைப்பு ggml_compute_params * params,
    கட்டமைப்பு ggml_tensor * dst,
    மாறிலி எண்ணிக்கை ggml_type type,
    மாறிலி int64_t இலக்கம்_rows_per_vec_dot,
    மாறிலி int64_t ir0_start,
    மாறிலி int64_t ir0_end,
    மாறிலி int64_t ir1_start,
    மாறிலி int64_t ir1_end) {

    மாறிலி கட்டமைப்பு ggml_tensor * src0 = dst->src[0];
    மாறிலி கட்டமைப்பு ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_இருமம்_OP_LOCALS

    மாறிலி bool src1_cont = ggml_is_contiguous(src1);

    ggml_vec_dot_t மாறிலி vec_dot      = type_traits_cpu[type].vec_dot;
    எண்ணிக்கை ggml_type மாறிலி vec_dot_type = type_traits_cpu[type].vec_dot_type;

    // broadcast factors
    மாறிலி int64_t r2 = ne12 / ne02;
    மாறிலி int64_t r3 = ne13 / ne03;

    //printf("ir0_start = %6lld, ir0_end = %6lld, ir1_start = %6lld, ir1_end = %6lld\n", ir0_start, ir0_end, ir1_start, ir1_end);

    // threads with no work simply yield (not sure if it helps)
    எனில் (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        திரும்பு;
    }

    மாறிலி வெறுமை * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
    மாறிலி size_t row_size = ggml_row_size(vec_dot_type, ne10);

    ப்ரதிஜ்ஞா(ne12 % ne02 == 0);
    ப்ரதிஜ்ஞா(ne13 % ne03 == 0);

    // block-tiling attempt
    மாறிலி int64_t blck_0 = 16;
    மாறிலி int64_t blck_1 = 16;

    மாறிலி size_t src1_col_stride = src1_cont || src1->type != vec_dot_type ? row_size : nb11;

    // attempt to reduce false-sharing (does not seem to make a difference)
    // 16 * 2, accounting for mmla kernels
    தசம tmp[32];

    க்கு (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        க்கு (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            க்கு (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += இலக்கம்_rows_per_vec_dot) {
                மாறிலி int64_t i13 = (ir1 / (ne12 * ne1));
                மாறிலி int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                மாறிலி int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                // broadcast src0 into src1
                மாறிலி int64_t i03 = i13 / r3;
                மாறிலி int64_t i02 = i12 / r2;

                மாறிலி int64_t i1 = i11;
                மாறிலி int64_t i2 = i12;
                மாறிலி int64_t i3 = i13;

                மாறிலி எழுத்து * src0_row = (மாறிலி எழுத்து*)src0->data + (0 + i02 * nb02 + i03 * nb03);

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                மாறிலி எழுத்து * src1_col = (மாறிலி எழுத்து*)wdata +
                    (src1_cont || src1->type != vec_dot_type
                        ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                        : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                தசம * dst_col = (தசம*)((எழுத்து*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

                //for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                //    vec_dot(ne00, &dst_col[ir0], src0_row + ir0*nb01, src1_col);
                //}

                க்கு (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += இலக்கம்_rows_per_vec_dot) {
                    vec_dot(ne00, &tmp[ir0 - iir0], (இலக்கம்_rows_per_vec_dot > 1 ? 16 : 0), src0_row + ir0 * nb01, (இலக்கம்_rows_per_vec_dot > 1 ? nb01 : 0), src1_col, (இலக்கம்_rows_per_vec_dot > 1 ? src1_col_stride : 0), இலக்கம்_rows_per_vec_dot);
                }

                க்கு (முழு cn = 0; cn < இலக்கம்_rows_per_vec_dot; ++cn) {
                    memcpy(&dst_col[iir0 + cn * nb1 / nb0], tmp + (cn * 16), (MIN(iir0 + blck_0, ir0_end) - iir0) * அளவு(தசம));
                }
            }
        }
    }
}

வெறுமை ggml_compute_forward_mul_mat(
        மாறிலி கட்டமைப்பு ggml_compute_params * params,
              கட்டமைப்பு ggml_tensor * dst) {

    மாறிலி கட்டமைப்பு ggml_tensor * src0 = dst->src[0];
    மாறிலி கட்டமைப்பு ggml_tensor * src1 = dst->src[1];

    மாறிலி int32_t hint = ggml_get_op_params_i32(dst, 1);
    எனில் (hint == GGML_HINT_SRC0_IS_HADAMARD && !params->use_ref) {
        ggml_compute_forward_fwht(params, dst);
        திரும்பு;
    }

    // If tiled is supported, it will execute the full op here and we return
    எனில் (ggml_compute_forward_mul_mat_tiled(params, dst)) {
        திரும்பு;
    }

    GGML_TENSOR_இருமம்_OP_LOCALS

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    எண்ணிக்கை ggml_type           மாறிலி vec_dot_type         = type_traits_cpu[src0->type].vec_dot_type;
    ggml_from_float_t        மாறிலி from_float           = type_traits_cpu[vec_dot_type].from_float;
    int64_t                  மாறிலி vec_dot_இலக்கம்_rows     = type_traits_cpu[src0->type].nrows;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == அளவு(தசம));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    // TODO: extract to "extra_op"
#எனில் GGML_USE_LLAMAFILE
    // broadcast factors
    மாறிலி int64_t r2 = ne12 / ne02;
    மாறிலி int64_t r3 = ne13 / ne03;

    மாறிலி bool src1_cont = ggml_is_contiguous(src1);

    எனில் (!params->use_ref && src1_cont) {
        க்கு (int64_t i13 = 0; i13 < ne13; i13++)
            க்கு (int64_t i12 = 0; i12 < ne12; i12++)
                எனில் (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_size(src0->type),
                                     (மாறிலி எழுத்து *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_size(src0->type),
                                     (மாறிலி எழுத்து *)src1->data + i12*nb12 + i13*nb13,
                                     nb11/ggml_type_size(src1->type),
                                     (எழுத்து *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_size(dst->type),
                                     src0->type,
                                     src1->type,
                                     dst->type))
                    செல் UseGgmlGemm1;
        திரும்பு;
    }
UseGgmlGemm1:;
#endif

    எனில் (src1->type != vec_dot_type) {
        எழுத்து * wdata = params->wdata;

        மாறிலி size_t nbw0 = ggml_type_size(vec_dot_type);
        மாறிலி size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        மாறிலி size_t nbw2 = nbw1*ne11;
        மாறிலி size_t nbw3 = nbw2*ne12;

        ப்ரதிஜ்ஞா(params->wsize >= ne13*nbw3);
        // src1 is either packed from F32 into vec_dot_type, or widened from F16 or BF16 into the F32 work buffer
        மாறிலி bool widen = src1->type != GGML_TYPE_F32;

        GGML_ASSERT(!widen || vec_dot_type == GGML_TYPE_F32);
        GGML_ASSERT(!widen || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_BF16);

    #எனில் 0
        க்கு (int64_t i13 = 0; i13 < ne13; ++i13) {
            க்கு (int64_t i12 = 0; i12 < ne12; ++i12) {
                க்கு (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                    from_float((தசம *)((எழுத்து *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (வெறுமை *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                                ne10);
                }
            }
        }
    #இல்லையெனில்
        மாறிலி int64_t bs = ggml_blck_size(vec_dot_type);

        க்கு (int64_t i13 = 0; i13 < ne13; ++i13) {
            க்கு (int64_t i12 = 0; i12 < ne12; ++i12) {
                க்கு (int64_t i11 = 0; i11 < ne11; ++i11) {
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    மாறிலி வெறுமை * src1_row  = (மாறிலி எழுத்து *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10;
                    வெறுமை       * wdata_row =                    wdata  + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0;

                    மாறிலி int64_t ne10_block_size = (ne10_block_end - ne10_block_start) * bs;

                    எனில் (src1->type == GGML_TYPE_F16) {
                        ggml_cpu_fp16_to_fp32((மாறிலி ggml_fp16_t *) src1_row, (தசம *) wdata_row, ne10_block_size);
                    } இல்லையெனில் எனில் (src1->type == GGML_TYPE_BF16) {
                        ggml_cpu_bf16_to_fp32((மாறிலி ggml_bf16_t *) src1_row, (தசம *) wdata_row, ne10_block_size);
                    } இல்லையெனில் {
                        from_float((மாறிலி தசம *) src1_row, wdata_row, ne10_block_size);
                    }
                }
            }
        }
    #endif
    }

    எனில் (ith == 0) {
        // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
        atomic_store_explicit(&params->threadpool->current_chunk, nth, memory_order_relaxed);
    }

    ggml_barrier(params->threadpool);

#எனில் GGML_USE_LLAMAFILE
    எனில் (!params->use_ref && src1->type != vec_dot_type) {
        மாறிலி வெறுமை* wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        மாறிலி size_t row_size = ggml_row_size(vec_dot_type, ne10);

        க்கு (int64_t i13 = 0; i13 < ne13; i13++)
            க்கு (int64_t i12 = 0; i12 < ne12; i12++)
                எனில் (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_size(src0->type),
                                     (மாறிலி எழுத்து *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_size(src0->type),
                                     (மாறிலி எழுத்து *)wdata + (i12*ne11 + i13*ne12*ne11)*row_size,
                                     row_size/ggml_type_size(vec_dot_type),
                                     (எழுத்து *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_size(dst->type),
                                     src0->type,
                                     vec_dot_type,
                                     dst->type))
                    செல் UseGgmlGemm2;
        திரும்பு;
    }
UseGgmlGemm2:;
#endif

    // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
    மாறிலி int64_t nr0 = ne0;

    // This is the size of the rest of the dimensions of the result
    மாறிலி int64_t nr1 = ne1 * ne2 * ne3;

    // Now select a reasonable chunk size.
    முழு chunk_size = 16;

    // We need to step up the size if it's small
    எனில் (nr0 == 1 || nr1 == 1) {
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
    எனில் (nchunk0 * nchunk1 < nth * 4 || ggml_is_numa()) {
        // distribute the thread work across the inner or outer loop based on which one is larger
        nchunk0 = nr0 > nr1 ? nth : 1; // parallelize by src0 rows
        nchunk1 = nr0 > nr1 ? 1 : nth; // parallelize by src1 rows
    }

    // The number of elements in each chunk
    மாறிலி int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    மாறிலி int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    // The first chunk comes from our thread_id, the rest will get auto-assigned.
    முழு current_chunk = ith;

    போது (current_chunk < nchunk0 * nchunk1) {
        மாறிலி int64_t ith0 = current_chunk % nchunk0;
        மாறிலி int64_t ith1 = current_chunk / nchunk0;

        மாறிலி int64_t ir0_start = dr0 * ith0;
        மாறிலி int64_t ir0_end = MIN(ir0_start + dr0, nr0);

        மாறிலி int64_t ir1_start = dr1 * ith1;
        மாறிலி int64_t ir1_end = MIN(ir1_start + dr1, nr1);

        // dot kernels can handle 1 row and col at a time, but mmla kernels can process 2 rows and cols
        int64_t இலக்கம்_rows_per_vec_dot = vec_dot_இலக்கம்_rows;

        // these checks are needed to avoid crossing dim1 boundaries
        // can be optimized, but the logic would become more complicated, so keeping it like this for simplicity
        எனில் ((nr0 % 2 != 0) || (ne11 % 2 != 0) || ((ir0_end - ir0_start) % 2 != 0) || ((ir1_end - ir1_start) % 2 != 0)) {
            இலக்கம்_rows_per_vec_dot = 1;
        }
        ggml_compute_forward_mul_mat_one_chunk(params, dst, src0->type, இலக்கம்_rows_per_vec_dot, ir0_start, ir0_end, ir1_start, ir1_end);

        எனில் (nth >= nchunk0 * nchunk1) {
            நிறுத்து;
        }

        current_chunk = atomic_fetch_add_explicit(&params->threadpool->current_chunk, 1, memory_order_relaxed);
    }
}

// ggml_compute_forward_mul_mat_id

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id)*ids->ne[0]*ids->ne[1] + (i1)]

கட்டமைப்பு mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

நிலையான வெறுமை ggml_compute_forward_mul_mat_id_one_chunk(
    கட்டமைப்பு ggml_tensor * dst,
    மாறிலி கட்டமைப்பு ggml_tensor * src0,
    மாறிலி கட்டமைப்பு ggml_tensor * src1,
    மாறிலி கட்டமைப்பு ggml_tensor * ids,
    மாறிலி int64_t cur_a,
    மாறிலி int64_t ir0_start,
    மாறிலி int64_t ir0_end,
    மாறிலி int64_t ir1_start,
    மாறிலி int64_t ir1_end,
    மாறிலி எழுத்து * src0_cur,
    மாறிலி கட்டமைப்பு mmid_row_mapping * matrix_rows,
    மாறிலி size_t row_size,
    மாறிலி bool src1_cont,
    மாறிலி வெறுமை * wdata) {

    GGML_TENSOR_இருமம்_OP_LOCALS

    மாறிலி எண்ணிக்கை ggml_type type = src0->type;

    ggml_vec_dot_t    மாறிலி vec_dot      = type_traits_cpu[type].vec_dot;
    எண்ணிக்கை ggml_type    மாறிலி vec_dot_type = type_traits_cpu[type].vec_dot_type;

    மாறிலி int64_t blck_0 = 16;
    மாறிலி int64_t blck_1 = 16;

    தசம tmp[16];

    க்கு (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        க்கு (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            க்கு (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ++ir1) {
                மாறிலி int64_t _i12 = ir1; // logical row index for this expert

                கட்டமைப்பு mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, _i12);
                மாறிலி முழு id       = row_mapping.i1; // selected expert index

                மாறிலி int64_t  i11 = id % ne11;
                மாறிலி int64_t  i12 = row_mapping.i2; // row index in src1

                மாறிலி int64_t  i1 = id;  // selected expert index
                மாறிலி int64_t  i2 = i12; // row

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                மாறிலி எழுத்து * src1_col = (மாறிலி எழுத்து *) wdata +
                    (src1_cont || src1->type != vec_dot_type
                    ? (i11      + i12*ne11)*row_size
                    : (i11*nb11 + i12*nb12));

                தசம * dst_col = (தசம *) ((எழுத்து *) dst->data + (i1*nb1 + i2*nb2));

                க்கு (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                    vec_dot(ne00, &tmp[ir0 - iir0], 0, src0_cur + ir0*nb01, 0, src1_col, 0, 1);
                }

                memcpy(&dst_col[iir0], tmp, (MIN(iir0 + blck_0, ir0_end) - iir0)*அளவு(தசம));
            }
        }
    }
}

நிலையான வெறுமை * incr_ptr_aligned(வெறுமை ** p, size_t size, size_t align) {

    வெறுமை * ptr = *p;
    ptr = (வெறுமை *) GGML_PAD((uintptr_t) ptr, align);
    *p = (வெறுமை *) ((எழுத்து *) ptr + size);
    திரும்பு ptr;
}

நிலையான வெறுமை ggml_compute_forward_mul_mat_id(
        மாறிலி கட்டமைப்பு ggml_compute_params * params,
              கட்டமைப்பு ggml_tensor * dst) {

    மாறிலி கட்டமைப்பு ggml_tensor * src0 = dst->src[0];
    மாறிலி கட்டமைப்பு ggml_tensor * src1 = dst->src[1];
    மாறிலி கட்டமைப்பு ggml_tensor * ids = dst->src[2];

    GGML_TENSOR_இருமம்_OP_LOCALS

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி எண்ணிக்கை ggml_type type = src0->type;

    மாறிலி bool src1_cont = ggml_is_contiguous(src1);

    எண்ணிக்கை ggml_type    மாறிலி vec_dot_type    = type_traits_cpu[type].vec_dot_type;
    ggml_from_float_t மாறிலி from_float      = type_traits_cpu[vec_dot_type].from_float;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == அளவு(தசம));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // row groups
    மாறிலி முழு n_ids = ids->ne[0]; // n_expert_used
    மாறிலி முழு n_as  = ne02;       // n_expert

    வெறுமை * wdata_cur = params->wdata;

    எனில் (src1->type != vec_dot_type) {
        incr_ptr_aligned(&wdata_cur, ggml_row_size(vec_dot_type, ggml_nelements(src1)), அளவு(int64_t));
    }

    int64_t * matrix_row_counts = // [n_as]
        incr_ptr_aligned(&wdata_cur, n_as*அளவு(int64_t), அளவு(int64_t));

    கட்டமைப்பு mmid_row_mapping * matrix_rows = // [n_as][ids->ne[0]*ids->ne[1]]
        incr_ptr_aligned(&wdata_cur, n_as*ids->ne[0]*ids->ne[1]*அளவு(கட்டமைப்பு mmid_row_mapping), அளவு(int64_t));

    எழுத்து (*atomic_current_chunk)[CACHE_வரி_SIZE] = // [n_as]
        incr_ptr_aligned(&wdata_cur, CACHE_வரி_SIZE * n_as, CACHE_வரி_SIZE);

    // Tiled matmul (see tiled.h); per-thread work buffers, 0 bytes when disabled. The
    // reservation is unconditional, the per expert eligibility is decided at dispatch time
    எழுத்து * tiled_scratch = incr_ptr_aligned(&wdata_cur, ggml_tiled_wdata_size(nth, dst), 64);

    GGML_ASSERT(params->wsize >= (size_t)((எழுத்து *) wdata_cur - (எழுத்து *) params->wdata));

    எனில் (src1->type != vec_dot_type) {
        எழுத்து * wdata = params->wdata;

        மாறிலி size_t nbw0 = ggml_type_size(vec_dot_type);
        மாறிலி size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        மாறிலி size_t nbw2 = nbw1*ne11;
        மாறிலி size_t nbw3 = nbw2*ne12;

        ப்ரதிஜ்ஞா(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

#எனில் 0
        க்கு (int64_t i13 = 0; i13 < ne13; ++i13) {
            க்கு (int64_t i12 = ith; i12 < ne12; i12 += nth) {
                க்கு (int64_t i11 = 0; i11 < ne11; ++i11) {
                    from_float((தசம *)((எழுத்து *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (வெறுமை *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                               ne10);
                }
            }
        }
#இல்லையெனில்
        க்கு (int64_t i13 = 0; i13 < ne13; ++i13) {
            க்கு (int64_t i12 = 0; i12 < ne12; ++i12) {
                க்கு (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = ggml_blck_size(vec_dot_type);
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    from_float((தசம *)((எழுத்து *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10),
                               (வெறுமை *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0),
                               (ne10_block_end - ne10_block_start) * bs);
                }
            }
        }
#endif
    }

    எனில் (ith == 0) {
        // initialize matrix_row_counts
        memset(matrix_row_counts, 0, n_as*அளவு(int64_t));

        // group rows by src0 matrix
        க்கு (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            க்கு (முழு id = 0; id < n_ids; ++id) {
                மாறிலி int32_t i02 = *(மாறிலி int32_t *) ((மாறிலி எழுத்து *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);

                ப்ரதிஜ்ஞா(i02 >= 0 && i02 < n_as);

                MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = (கட்டமைப்பு mmid_row_mapping) {id, iid1};
                matrix_row_counts[i02] += 1;
            }
        }
    }

    // reset current_chunk
    க்கு (முழு cur_a = ith; cur_a < n_as; cur_a += nth) {
        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);
        *current_chunk_ctr = nth;
    }

    ggml_barrier(params->threadpool);

    க்கு (முழு cur_a = 0; cur_a < n_as; ++cur_a) {
        மாறிலி int64_t cne1 = matrix_row_counts[cur_a];

        எனில் (cne1 == 0) {
            தொடர்;
        }

        // tiled takes over if profitable for this expert (see tiled.h)
        எனில் (ggml_compute_forward_mul_mat_id_tiled(params, dst, cur_a, cne1, (மாறிலி int32_t *) &MMID_MATRIX_ROW(cur_a, 0), tiled_scratch)) {
            தொடர்;
        }

        மாறிலி எழுத்து * src0_cur = (மாறிலி எழுத்து *) src0->data + cur_a * nb02;
        மாறிலி வெறுமை * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        மாறிலி size_t row_size = ggml_row_size(vec_dot_type, ne10);

        மாறிலி int64_t nr0 = ne01;
        மாறிலி int64_t nr1 = cne1;

        முழு chunk_size = 16;
        எனில் (nr0 == 1 || nr1 == 1) {
            chunk_size = 64;
        }

        // disable for NUMA
        மாறிலி bool disable_chunking = ggml_is_numa();

        int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
        int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

        எனில் (nchunk0 * nchunk1 < nth * 4 || disable_chunking) {
            nchunk0 = nr0 > nr1 ? nth : 1;
            nchunk1 = nr0 > nr1 ? 1 : nth;
        }

        மாறிலி int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        மாறிலி int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

        முழு current_chunk = ith;

        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);

        போது (current_chunk < nchunk0 * nchunk1) {
            மாறிலி int64_t ith0 = current_chunk % nchunk0;
            மாறிலி int64_t ith1 = current_chunk / nchunk0;

            மாறிலி int64_t ir0_start = dr0 * ith0;
            மாறிலி int64_t ir0_end = MIN(ir0_start + dr0, nr0);

            மாறிலி int64_t ir1_start = dr1 * ith1;
            மாறிலி int64_t ir1_end = MIN(ir1_start + dr1, nr1);

            ggml_compute_forward_mul_mat_id_one_chunk(
                dst, src0, src1, ids, cur_a,
                ir0_start, ir0_end, ir1_start, ir1_end,
                src0_cur, matrix_rows, row_size, src1_cont, wdata
            );

            எனில் (nth >= nchunk0 * nchunk1) {
                நிறுத்து;
            }

            current_chunk = atomic_fetch_add_explicit(current_chunk_ctr, 1, memory_order_relaxed);
        }
    }
}

/////////////////////////////////

நிலையான வெறுமை ggml_compute_forward(கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * tensor) {
    GGML_ASSERT(params);

    எனில் (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {
        திரும்பு;
    }

    // extra_buffer op?
    எனில் (ggml_cpu_extra_compute_forward(params, tensor)) {
        திரும்பு;
    }

    தேர்வு (tensor->op) {
        நிலை GGML_OP_DUP:
            {
                ggml_compute_forward_dup(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_ADD:
            {
                ggml_compute_forward_add(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_ADD_ID:
            {
                ggml_compute_forward_add_id(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_ADD1:
            {
                ggml_compute_forward_add1(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_ACC:
            {
                ggml_compute_forward_acc(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_SUB:
            {
                ggml_compute_forward_sub(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_MUL:
            {
                ggml_compute_forward_mul(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_DIV:
            {
                ggml_compute_forward_div(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_SQR:
            {
                ggml_compute_forward_sqr(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_SQRT:
            {
                ggml_compute_forward_sqrt(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_LOG:
            {
                ggml_compute_forward_log(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_SIN:
            {
                ggml_compute_forward_sin(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_COS:
            {
                ggml_compute_forward_cos(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_SUM:
            {
                ggml_compute_forward_sum(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_SUM_ROWS:
            {
                ggml_compute_forward_sum_rows(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_CUMSUM:
            {
                ggml_compute_forward_cumsum(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_MEAN:
            {
                ggml_compute_forward_mean(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_ARGMAX:
            {
                ggml_compute_forward_argmax(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_மொத்தம்_EQUAL:
            {
                ggml_compute_forward_மொத்தம்_equal(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_REPEAT:
            {
                ggml_compute_forward_repeat(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_REPEAT_BACK:
            {
                ggml_compute_forward_repeat_back(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_CONCAT:
            {
                ggml_compute_forward_concat(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_SILU_BACK:
            {
                ggml_compute_forward_silu_back(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_NORM:
            {
                ggml_compute_forward_norm(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_RMS_NORM:
            {
                ggml_compute_forward_rms_norm(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_RMS_NORM_BACK:
            {
                ggml_compute_forward_rms_norm_back(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_GROUP_NORM:
            {
                ggml_compute_forward_group_norm(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_L2_NORM:
            {
                ggml_compute_forward_l2_norm(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_MUL_MAT:
            {
                ggml_compute_forward_mul_mat(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_MUL_MAT_ID:
            {
                ggml_compute_forward_mul_mat_id(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_OUT_PROD:
            {
                ggml_compute_forward_out_prod(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_SCALE:
            {
                ggml_compute_forward_scale(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_SET:
            {
                ggml_compute_forward_set(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_CPY:
            {
                ggml_compute_forward_cpy(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_CONT:
            {
                ggml_compute_forward_cont(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_GET_ROWS:
            {
                ggml_compute_forward_get_rows(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_GET_ROWS_BACK:
            {
                ggml_compute_forward_get_rows_back(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_SET_ROWS:
            {
                ggml_compute_forward_set_rows(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_DIAG:
            {
                ggml_compute_forward_diag(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_DIAG_MASK_INF:
            {
                ggml_compute_forward_diag_mask_inf(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_DIAG_MASK_ZERO:
            {
                ggml_compute_forward_diag_mask_zero(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_SOFT_MAX:
            {
                ggml_compute_forward_soft_max(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_SOFT_MAX_BACK:
            {
                ggml_compute_forward_soft_max_ext_back(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_ROPE:
            {
                ggml_compute_forward_rope(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_ROPE_BACK:
            {
                ggml_compute_forward_rope_back(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_CLAMP:
            {
                ggml_compute_forward_clamp(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_compute_forward_conv_transpose_1d(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_IM2COL:
            {
                ggml_compute_forward_im2col(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_IM2COL_BACK:
            {
                ggml_compute_forward_im2col_back_f32(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_IM2COL_3D:
            {
                ggml_compute_forward_im2col_3d(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_COL2IM_1D:
            {
                ggml_compute_forward_col2im_1d(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_CONV_2D:
            {
                ggml_compute_forward_conv_2d(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_CONV_3D:
            {
                ggml_compute_forward_conv_3d(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_CONV_2D_DW:
            {
                ggml_compute_forward_conv_2d_dw(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_CONV_TRANSPOSE_2D:
            {
                ggml_compute_forward_conv_transpose_2d(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_POOL_1D:
            {
                ggml_compute_forward_pool_1d(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_POOL_2D:
            {
                ggml_compute_forward_pool_2d(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_POOL_2D_BACK:
            {
                ggml_compute_forward_pool_2d_back(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_UPSCALE:
            {
                ggml_compute_forward_upscale(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_PAD:
            {
                ggml_compute_forward_pad(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_PAD_REFLECT_1D:
            {
                ggml_compute_forward_pad_reflect_1d(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_ROLL:
            {
                ggml_compute_forward_roll(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_ARANGE:
            {
                ggml_compute_forward_arange(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_TIMESTEP_EMBEDDING:
            {
                ggml_compute_forward_timestep_embedding(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_ARGSORT:
            {
                ggml_compute_forward_argsort(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_உச்சி_K:
            {
                ggml_compute_forward_உச்சி_k(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_LEAKY_RELU:
            {
                ggml_compute_forward_leaky_relu(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_TRI:
            {
                ggml_compute_forward_tri(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_FILL:
            {
                ggml_compute_forward_fill(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_FLASH_ATTN_EXT:
            {
                ggml_compute_forward_flash_attn_ext(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_FLASH_ATTN_BACK:
            {
                int32_t t = ggml_get_op_params_i32(tensor, 0);
                GGML_ASSERT(t == 0 || t == 1);
                bool masked = t != 0;
                ggml_compute_forward_flash_attn_back(params, masked, tensor);
            } நிறுத்து;
        நிலை GGML_OP_SSM_CONV:
            {
                ggml_compute_forward_ssm_conv(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_SSM_SCAN:
            {
                ggml_compute_forward_ssm_scan(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_WIN_PART:
            {
                ggml_compute_forward_win_part(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_WIN_UNPART:
            {
                ggml_compute_forward_win_unpart(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_UNARY:
            {
                ggml_compute_forward_unary(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_GLU:
            {
                ggml_compute_forward_glu(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_GET_REL_POS:
            {
                ggml_compute_forward_get_rel_pos(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_ADD_REL_POS:
            {
                ggml_compute_forward_add_rel_pos(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_RWKV_WKV6:
            {
                ggml_compute_forward_rwkv_wkv6(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_GATED_LINEAR_ATTN:
            {
                ggml_compute_forward_gla(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_RWKV_WKV7:
            {
                ggml_compute_forward_rwkv_wkv7(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_SOLVE_TRI:
            {
                ggml_compute_forward_solve_tri(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_GATED_DELTA_NET:
            {
                ggml_compute_forward_gated_delta_net(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_LIGHTNING_INDEXER:
            {
                ggml_compute_forward_lightning_indexer(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_DSV4_HC_COMB:
            {
                ggml_compute_forward_dsv4_hc_comb(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_DSV4_HC_PRE:
            {
                ggml_compute_forward_dsv4_hc_pre(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_DSV4_HC_POST:
            {
                ggml_compute_forward_dsv4_hc_post(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_உச்சி_K_MASK:
            {
                ggml_compute_forward_உச்சி_k_mask(params, tensor);
            } நிறுத்து;
        நிலை GGML_OP_MAP_CUSTOM1:
            {
                ggml_compute_forward_map_custom1(params, tensor);
            }
            நிறுத்து;
        நிலை GGML_OP_MAP_CUSTOM2:
            {
                ggml_compute_forward_map_custom2(params, tensor);
            }
            நிறுத்து;
        நிலை GGML_OP_MAP_CUSTOM3:
            {
                ggml_compute_forward_map_custom3(params, tensor);
            }
            நிறுத்து;
        நிலை GGML_OP_CUSTOM:
            {
                ggml_compute_forward_custom(params, tensor);
            }
            நிறுத்து;
        நிலை GGML_OP_CROSS_ENTROPY_LOSS:
            {
                ggml_compute_forward_cross_entropy_loss(params, tensor);
            }
            நிறுத்து;
        நிலை GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            {
                ggml_compute_forward_cross_entropy_loss_back(params, tensor);
            }
            நிறுத்து;
        நிலை GGML_OP_OPT_STEP_ADAMW:
            {
                ggml_compute_forward_opt_step_adamw(params, tensor);
            }
            நிறுத்து;
        நிலை GGML_OP_OPT_STEP_SGD:
            {
                ggml_compute_forward_opt_step_sgd(params, tensor);
            }
            நிறுத்து;
        நிலை GGML_OP_NONE:
            {
                // nop
            } நிறுத்து;
        நிலை GGML_OP_RESHAPE:
            {
                // nop
            } நிறுத்து;
        நிலை GGML_OP_PERMUTE:
            {
                // nop
            } நிறுத்து;
        நிலை GGML_OP_VIEW:
            {
                // nop
            } நிறுத்து;
        நிலை GGML_OP_TRANSPOSE:
            {
                // nop
            } நிறுத்து;
        நிலை GGML_OP_மொத்தம்:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// Android's libc implementation "bionic" does not support setting affinity
#எனில் defined(__gnu_linux__)
நிலையான வெறுமை set_numa_thread_affinity(முழு thread_n) {
    எனில் (!ggml_is_numa()) {
        திரும்பு;
    }

    முழு முனை_இலக்கம்;
    முழு rv;
    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    தேர்வு(g_state.numa.numa_strategy) {
        நிலை GGML_NUMA_STRATEGY_DISTRIBUTE:
            // run thread on node_num thread_n / (threads per node)
            முனை_இலக்கம் = thread_n % g_state.numa.n_nodes;
            நிறுத்து;
        நிலை GGML_NUMA_STRATEGY_ISOLATE:
            // run thread on current_node
            முனை_இலக்கம் = g_state.numa.current_முனை;
            நிறுத்து;
        நிலை GGML_NUMA_STRATEGY_NUMACTL:
            // use the cpuset that numactl gave us
            rv = pthread_setaffinity_np(pthread_self(), setsize, &g_state.numa.cpuset);
            எனில் (rv) {
                fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n",strerror(rv));
            }
            திரும்பு;
        இயல்பு:
            திரும்பு;
    }

    கட்டமைப்பு ggml_numa_முனை * முனை = &g_state.numa.nodes[முனை_இலக்கம்];

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    க்கு (size_t i = 0; i < முனை->n_cpus; ++i) {
        CPU_SET_S(முனை->cpus[i], setsize, cpus);
    }

    rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    எனில் (rv) {
            fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}

நிலையான வெறுமை clear_numa_thread_affinity(வெறுமை) {
    எனில் (!ggml_is_numa()) {
        திரும்பு;
    }

    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    க்கு (unsigned i = 0; i < g_state.numa.total_cpus; ++i) {
        CPU_SET_S(i, setsize, cpus);
    }

    முழு rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    எனில் (rv) {
        fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}
#இல்லையெனில்
// TODO: Windows etc.
// (the linux implementation may also work on BSD, someone should test)
நிலையான வெறுமை set_numa_thread_affinity(முழு thread_n) { UNUSED(thread_n);  }
நிலையான வெறுமை clear_numa_thread_affinity(வெறுமை) {}
#endif

நிலையான முழு ggml_get_n_tasks(கட்டமைப்பு ggml_tensor * முனை, முழு n_threads) {
    முழு n_tasks = 0;

    எனில் (ggml_is_empty(முனை)) {
        // no need to multi-thread a no-op
        n_tasks = 1;
        திரும்பு n_tasks;
    }

    தேர்வு (முனை->op) {
        நிலை GGML_OP_CPY:
        நிலை GGML_OP_DUP:
        நிலை GGML_OP_CONT:
        நிலை GGML_OP_ADD:
        நிலை GGML_OP_ADD_ID:
        நிலை GGML_OP_ADD1:
        நிலை GGML_OP_ACC:
        நிலை GGML_OP_CUMSUM:
        நிலை GGML_OP_TRI:
        நிலை GGML_OP_FILL:
            {
                n_tasks = n_threads;
            } நிறுத்து;
        நிலை GGML_OP_SUB:
        நிலை GGML_OP_SQR:
        நிலை GGML_OP_SQRT:
        நிலை GGML_OP_LOG:
        நிலை GGML_OP_SIN:
        நிலை GGML_OP_COS:
        நிலை GGML_OP_SUM:
        நிலை GGML_OP_SUM_ROWS:
        நிலை GGML_OP_MEAN:
        நிலை GGML_OP_ARGMAX:
            {
                n_tasks = 1;
            } நிறுத்து;
        நிலை GGML_OP_மொத்தம்_EQUAL:
        நிலை GGML_OP_SOLVE_TRI:
        நிலை GGML_OP_GATED_DELTA_NET:
        நிலை GGML_OP_DSV4_HC_COMB:
        நிலை GGML_OP_DSV4_HC_PRE:
        நிலை GGML_OP_DSV4_HC_POST:
        நிலை GGML_OP_உச்சி_K_MASK:
            {
                n_tasks = n_threads;
            } நிறுத்து;
        நிலை GGML_OP_REPEAT:
        நிலை GGML_OP_REPEAT_BACK:
        நிலை GGML_OP_LEAKY_RELU:
            {
                n_tasks = 1;
            } நிறுத்து;
        நிலை GGML_OP_UNARY:
            தேர்வு (ggml_get_unary_op(முனை)) {
                நிலை GGML_UNARY_OP_ABS:
                நிலை GGML_UNARY_OP_SGN:
                நிலை GGML_UNARY_OP_NEG:
                நிலை GGML_UNARY_OP_STEP:
                நிலை GGML_UNARY_OP_TANH:
                நிலை GGML_UNARY_OP_ELU:
                நிலை GGML_UNARY_OP_RELU:
                நிலை GGML_UNARY_OP_SIGMOID:
                நிலை GGML_UNARY_OP_HARDSWISH:
                நிலை GGML_UNARY_OP_HARDSIGMOID:
                நிலை GGML_UNARY_OP_EXP:
                நிலை GGML_UNARY_OP_SOFTPLUS:
                நிலை GGML_UNARY_OP_EXPM1:
                நிலை GGML_UNARY_OP_FLOOR:
                நிலை GGML_UNARY_OP_CEIL:
                நிலை GGML_UNARY_OP_ROUND:
                நிலை GGML_UNARY_OP_TRUNC:
                    {
                        n_tasks = 1;
                    } நிறுத்து;

                நிலை GGML_UNARY_OP_GELU:
                நிலை GGML_UNARY_OP_GELU_ERF:
                நிலை GGML_UNARY_OP_GELU_QUICK:
                நிலை GGML_UNARY_OP_SILU:
                நிலை GGML_UNARY_OP_XIELU:
                    {
                        n_tasks = n_threads;
                    } நிறுத்து;
                இயல்பு:
                    GGML_ABORT("fatal error");
            }
            நிறுத்து;
        நிலை GGML_OP_GLU:
            தேர்வு (ggml_get_glu_op(முனை)) {
                நிலை GGML_GLU_OP_REGLU:
                நிலை GGML_GLU_OP_GEGLU:
                நிலை GGML_GLU_OP_SWIGLU:
                நிலை GGML_GLU_OP_SWIGLU_OAI:
                நிலை GGML_GLU_OP_GEGLU_ERF:
                நிலை GGML_GLU_OP_GEGLU_QUICK:
                நிலை GGML_GLU_OP_SWIGLU_CLAMP:
                    {
                        n_tasks = n_threads;
                    } நிறுத்து;
                இயல்பு:
                    GGML_ABORT("fatal error");
            }
            நிறுத்து;
        நிலை GGML_OP_SILU_BACK:
        நிலை GGML_OP_MUL:
        நிலை GGML_OP_DIV:
        நிலை GGML_OP_NORM:
        நிலை GGML_OP_RMS_NORM:
        நிலை GGML_OP_RMS_NORM_BACK:
        நிலை GGML_OP_L2_NORM:
        நிலை GGML_OP_GROUP_NORM:
        நிலை GGML_OP_CONCAT:
        நிலை GGML_OP_MUL_MAT:
        நிலை GGML_OP_MUL_MAT_ID:
        நிலை GGML_OP_OUT_PROD:
        நிலை GGML_OP_SET_ROWS:
            {
                n_tasks = n_threads;
            } நிறுத்து;
        நிலை GGML_OP_GET_ROWS:
            {
                // FIXME: get_rows can use additional threads, but the cost of launching additional threads
                // decreases performance with GPU offloading
                //n_tasks = n_threads;
                n_tasks = 1;
            } நிறுத்து;
        நிலை GGML_OP_SCALE:
        நிலை GGML_OP_SET:
        நிலை GGML_OP_RESHAPE:
        நிலை GGML_OP_VIEW:
        நிலை GGML_OP_PERMUTE:
        நிலை GGML_OP_TRANSPOSE:
        நிலை GGML_OP_GET_ROWS_BACK:
        நிலை GGML_OP_DIAG:
            {
                n_tasks = 1;
            } நிறுத்து;
        நிலை GGML_OP_DIAG_MASK_ZERO:
        நிலை GGML_OP_DIAG_MASK_INF:
        நிலை GGML_OP_SOFT_MAX_BACK:
        நிலை GGML_OP_ROPE:
        நிலை GGML_OP_ROPE_BACK:
        நிலை GGML_OP_ADD_REL_POS:
            {
                n_tasks = n_threads;
            } நிறுத்து;
        நிலை GGML_OP_CLAMP:
            {
                n_tasks = 1; //TODO
            } நிறுத்து;
        நிலை GGML_OP_SOFT_MAX:
            {
                n_tasks = MIN(n_threads, ggml_nrows(முனை->src[0]));
            } நிறுத்து;
        நிலை GGML_OP_IM2COL:
        நிலை GGML_OP_IM2COL_BACK:
        நிலை GGML_OP_IM2COL_3D:
        நிலை GGML_OP_CONV_2D:
        நிலை GGML_OP_CONV_3D:
        நிலை GGML_OP_CONV_2D_DW:
        நிலை GGML_OP_COL2IM_1D:
        நிலை GGML_OP_CONV_TRANSPOSE_1D:
        நிலை GGML_OP_CONV_TRANSPOSE_2D:
            {
                n_tasks = n_threads;
            } நிறுத்து;
        நிலை GGML_OP_POOL_1D:
        நிலை GGML_OP_POOL_2D:
        நிலை GGML_OP_POOL_2D_BACK:
            {
                n_tasks = 1;
            } நிறுத்து;
        நிலை GGML_OP_UPSCALE:
        நிலை GGML_OP_PAD:
        நிலை GGML_OP_PAD_REFLECT_1D:
        நிலை GGML_OP_ROLL:
        நிலை GGML_OP_ARANGE:
        நிலை GGML_OP_TIMESTEP_EMBEDDING:
        நிலை GGML_OP_ARGSORT:
        நிலை GGML_OP_உச்சி_K:
        நிலை GGML_OP_FLASH_ATTN_EXT:
        நிலை GGML_OP_FLASH_ATTN_BACK:
        நிலை GGML_OP_SSM_CONV:
        நிலை GGML_OP_SSM_SCAN:
        நிலை GGML_OP_LIGHTNING_INDEXER:
            {
                n_tasks = n_threads;
            } நிறுத்து;
        நிலை GGML_OP_RWKV_WKV6:
        நிலை GGML_OP_GATED_LINEAR_ATTN:
        நிலை GGML_OP_RWKV_WKV7:
            {
                மாறிலி int64_t n_heads = முனை->src[1]->ne[1];
                n_tasks = MIN(n_threads, n_heads);
            } நிறுத்து;
        நிலை GGML_OP_WIN_PART:
        நிலை GGML_OP_WIN_UNPART:
        நிலை GGML_OP_GET_REL_POS:
            {
                n_tasks = 1;
            } நிறுத்து;
        நிலை GGML_OP_MAP_CUSTOM1:
            {
                கட்டமைப்பு ggml_map_custom1_op_params p;
                memcpy(&p, முனை->op_params, அளவு(p));
                எனில் (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } இல்லையெனில் {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } நிறுத்து;
        நிலை GGML_OP_MAP_CUSTOM2:
            {
                கட்டமைப்பு ggml_map_custom2_op_params p;
                memcpy(&p, முனை->op_params, அளவு(p));
                எனில் (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } இல்லையெனில் {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } நிறுத்து;
        நிலை GGML_OP_MAP_CUSTOM3:
            {
                கட்டமைப்பு ggml_map_custom3_op_params p;
                memcpy(&p, முனை->op_params, அளவு(p));
                எனில் (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } இல்லையெனில் {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } நிறுத்து;
        நிலை GGML_OP_CUSTOM:
            {
                கட்டமைப்பு ggml_custom_op_params p;
                memcpy(&p, முனை->op_params, அளவு(p));
                எனில் (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } இல்லையெனில் {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } நிறுத்து;
        நிலை GGML_OP_CROSS_ENTROPY_LOSS:
        நிலை GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        நிலை GGML_OP_OPT_STEP_ADAMW:
        நிலை GGML_OP_OPT_STEP_SGD:
            {
                n_tasks = n_threads;
            } நிறுத்து;
        நிலை GGML_OP_NONE:
            {
                n_tasks = 1;
            } நிறுத்து;
        நிலை GGML_OP_மொத்தம்:
            {
                GGML_ABORT("fatal error");
            }
        இயல்பு:
            {
                fprintf(stderr, "%s: op not implemented: ", __func__);
                எனில் (முனை->op < GGML_OP_மொத்தம்) {
                    fprintf(stderr, "%s\n", ggml_op_name(முனை->op));
                } இல்லையெனில் {
                    fprintf(stderr, "%d\n", முனை->op);
                }
                GGML_ABORT("fatal error");
            }
    }

    ப்ரதிஜ்ஞா(n_tasks > 0);

    திரும்பு n_tasks;
}

நிலையான thread_ret_t ggml_graph_compute_secondary_thread(வெறுமை* data);

#எனில் defined(_WIN32)
#அந்தர்பாவ "windows.h"

// TODO: support > 64 CPUs
நிலையான bool ggml_thread_apply_affinity(bool * mask) {
    HANDLE    h = GetCurrentThread();
    uint64_t  bitmask = 0ULL;

    ப்ரதிஜ்ஞா(GGML_MAX_N_THREADS >= 64);

    க்கு (int32_t i = 0; i < 8; i++) {
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

    க்கு (int32_t i = 64; i < GGML_MAX_N_THREADS; i++) {
        எனில் (mask[i]) {
            fprintf(stderr, "warn: setting thread-affinity for > 64 CPUs isn't supported on windows!\n");
            நிறுத்து;
        }
    }

    DWORD_PTR m = (DWORD_PTR)bitmask;

    m = SetThreadAffinityMask(h, m);

    திரும்பு m != 0;
}

நிலையான bool ggml_thread_apply_priority(int32_t prio) {
    // Note that on Windows the Process Priority Class must be updated in order to set Thread priority.
    // This is up to the applications.
    DWORD p = THREAD_PRIORITY_NORMAL;
    தேர்வு (prio) {
        நிலை GGML_SCHED_PRIO_கீழ்:      p = THREAD_PRIORITY_BELOW_NORMAL;  நிறுத்து;
        நிலை GGML_SCHED_PRIO_NORMAL:   p = THREAD_PRIORITY_NORMAL;        நிறுத்து;
        நிலை GGML_SCHED_PRIO_MEDIUM:   p = THREAD_PRIORITY_ABOVE_NORMAL;  நிறுத்து;
        நிலை GGML_SCHED_PRIO_உயர்வு:     p = THREAD_PRIORITY_HIGHEST;       நிறுத்து;
        நிலை GGML_SCHED_PRIO_REALTIME: p = THREAD_PRIORITY_TIME_CRITICAL; நிறுத்து;
    }

    எனில் (prio != GGML_SCHED_PRIO_கீழ்) {
        // Tell Windows that this thread should not be throttled (needs its own CPU core).
        // Newer Windows 11 versions aggressively park (offline) CPU cores and often place
        // all our threads onto the first 4 cores which results in terrible performance with
        // n_threads > 4
        #எனில் _WIN32_WINNT >= 0x0602
        THREAD_POWER_THROTTLING_STATE t;
        ZeroMemory(&t, அளவு(t));
        t.Version     = THREAD_POWER_THROTTLING_CURRENT_VERSION;
        t.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
        t.StateMask   = 0;

        எனில் (!SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &t, அளவு(t))) {
            GGML_LOG_DEBUG("failed to disable thread power throttling %d : (%d)\n", prio, (முழு) GetLastError());
            திரும்பு false;
        }
        #endif
    }

    எனில் (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        திரும்பு true;
    }

    எனில் (!SetThreadPriority(GetCurrentThread(), p)) {
        fprintf(stderr, "warn: failed to set thread priority %d : (%d)\n", prio, (முழு) GetLastError());
        திரும்பு false;
    }

    திரும்பு true;
}

#அல்லதெனில் defined(__APPLE__)
#அந்தர்பாவ <sys/types.h>
#அந்தர்பாவ <sys/resource.h>

நிலையான bool ggml_thread_apply_affinity(மாறிலி bool * mask) {
    // Not supported on Apple platforms
    UNUSED(mask);
    திரும்பு true;
}

நிலையான bool ggml_thread_apply_priority(int32_t prio) {
    கட்டமைப்பு sched_param p;
    int32_t policy = SCHED_OTHER;
    தேர்வு (prio) {
        // TODO: there seems to be no way to set lower prio on Apple platforms
        நிலை GGML_SCHED_PRIO_கீழ்:      policy = SCHED_OTHER; p.sched_priority = 0;  நிறுத்து;
        நிலை GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  நிறுத்து;
        நிலை GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; நிறுத்து;
        நிலை GGML_SCHED_PRIO_உயர்வு:     policy = SCHED_FIFO;  p.sched_priority = 80; நிறுத்து;
        நிலை GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; நிறுத்து;
    }

    எனில் (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        திரும்பு true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    எனில் (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        திரும்பு false;
    }

    திரும்பு true;
}

#அல்லதெனில் defined(__linux__)
// TODO: this may not work on BSD, to be verified

நிலையான bool ggml_thread_apply_affinity(மாறிலி bool * mask) {
    cpu_set_t cpuset;
    முழு err;

    CPU_ZERO(&cpuset);

    க்கு (uint32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
        எனில் (mask[i]) {
            GGML_PRINT_DEBUG("Thread %lx: adding %d to cpuset\n", pthread_self(), i);
            CPU_SET(i, &cpuset);
        }
    }

#ifdef __ANDROID__
    err = sched_setaffinity(0, அளவு(cpuset), &cpuset);
    எனில் (err < 0) {
        err = errno;
    }
#இல்லையெனில்
    err = pthread_setaffinity_np(pthread_self(), அளவு(cpuset), &cpuset);
#endif
    எனில் (err != 0) {
        fprintf(stderr, "warn: failed to set affinity mask 0x%llx : %s (%d)\n", (unsigned long long)mask, strerror(err), err);
        திரும்பு false;
    }

    திரும்பு true;
}

நிலையான bool ggml_thread_apply_priority(int32_t prio) {
    கட்டமைப்பு sched_param p;
    int32_t policy = SCHED_OTHER;
    தேர்வு (prio) {
        நிலை GGML_SCHED_PRIO_கீழ்:      policy = SCHED_BATCH; p.sched_priority = 0;  நிறுத்து;
        நிலை GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  நிறுத்து;
        நிலை GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; நிறுத்து;
        நிலை GGML_SCHED_PRIO_உயர்வு:     policy = SCHED_FIFO;  p.sched_priority = 80; நிறுத்து;
        நிலை GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; நிறுத்து;
    }

    எனில் (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        திரும்பு true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    எனில் (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        திரும்பு false;
    }

    திரும்பு true;
}

#இல்லையெனில் // unsupported platforms

நிலையான bool ggml_thread_apply_affinity(மாறிலி bool * mask) {
    UNUSED(mask);
    திரும்பு true;
}

நிலையான bool ggml_thread_apply_priority(int32_t prio) {
    UNUSED(prio);
    திரும்பு true;
}

#endif

நிலையான bool ggml_thread_cpumask_is_valid(மாறிலி bool * mask) {
    க்கு (முழு i = 0; i < GGML_MAX_N_THREADS; i++) {
        எனில் (mask[i]) { திரும்பு true; }
    }
    திரும்பு false;
}

நிலையான வெறுமை ggml_thread_cpumask_next(மாறிலி bool * global_mask, bool * local_mask, bool strict, int32_t* iter) {
    எனில் (!strict) {
        memcpy(local_mask, global_mask, GGML_MAX_N_THREADS);
        திரும்பு;
    } இல்லையெனில் {
        memset(local_mask, 0, GGML_MAX_N_THREADS);
        int32_t base_idx = *iter;
        க்கு (int32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
            int32_t idx = base_idx + i;
            எனில் (idx >= GGML_MAX_N_THREADS) {
                // Just a cheaper modulo
                idx -= GGML_MAX_N_THREADS;
            }
            எனில் (global_mask[idx]) {
                local_mask[idx] = 1;
                *iter = idx + 1;
                திரும்பு;
            }
        }
    }
}

வெறுமை ggml_threadpool_free(கட்டமைப்பு ggml_threadpool* threadpool) {
    எனில் (!threadpool) திரும்பு;

    மாறிலி முழு n_threads = threadpool->n_threads;

#ifndef GGML_USE_OPENMP
    கட்டமைப்பு ggml_compute_state* workers = threadpool->workers;

    ggml_mutex_lock(&threadpool->mutex);

    threadpool->stop = true;
    threadpool->pause = false;

    ggml_cond_broadcast(&threadpool->cond);
    ggml_mutex_unlock(&threadpool->mutex);

    க்கு (முழு j = 1; j < n_threads; j++) {
        int32_t rc = ggml_thread_join(workers[j].thrd, வெற்று);
        GGML_ASSERT(rc == GGML_EXIT_SUCCESS || rc == GGML_EXIT_ABORTED);
        UNUSED(rc);
    }

    ggml_mutex_destroy(&threadpool->mutex);
    ggml_cond_destroy(&threadpool->cond);
#endif // GGML_USE_OPENMP

    மாறிலி size_t workers_size = அளவு(கட்டமைப்பு ggml_compute_state) * n_threads;
    ggml_aligned_free(threadpool->workers, workers_size);
    ggml_aligned_free(threadpool, அளவு(கட்டமைப்பு ggml_threadpool));
}

#ifndef GGML_USE_OPENMP
// pause/resume must be called under mutex
நிலையான வெறுமை ggml_threadpool_pause_locked(கட்டமைப்பு ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Pausing threadpool\n");
    threadpool->pause = true;
    ggml_cond_broadcast(&threadpool->cond);
}

நிலையான வெறுமை ggml_threadpool_resume_locked(கட்டமைப்பு ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Resuming threadpool\n");
    threadpool->pause = false;
    ggml_cond_broadcast(&threadpool->cond);
}
#endif

வெறுமை ggml_threadpool_pause(கட்டமைப்பு ggml_threadpool * threadpool) {
#ifndef GGML_USE_OPENMP
    ggml_mutex_lock(&threadpool->mutex);
    எனில் (!threadpool->pause) {
       ggml_threadpool_pause_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#இல்லையெனில்
    UNUSED(threadpool);
#endif
}

வெறுமை ggml_threadpool_resume(கட்டமைப்பு ggml_threadpool * threadpool) {
#ifndef GGML_USE_OPENMP
    ggml_mutex_lock(&threadpool->mutex);
    எனில் (threadpool->pause) {
       ggml_threadpool_resume_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#இல்லையெனில்
    UNUSED(threadpool);
#endif
}

கட்டமைப்பு ggml_cplan ggml_graph_plan(
          மாறிலி கட்டமைப்பு ggml_cgraph * cgraph,
                               முழு   n_threads,
            கட்டமைப்பு ggml_threadpool * threadpool) {

    எனில் (threadpool == வெற்று) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
    }
    எனில் (n_threads <= 0) {
        n_threads = threadpool ? threadpool->n_threads : GGML_DEFAULT_N_THREADS;
    }

#எனில் defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
    // Emscripten without pthreads support can only use a single thread
    n_threads = 1;
#endif

#எனில் defined(__wasi__)
    // WASI doesn't support parallelism yet
    n_threads = 1;
#endif

    size_t work_size = 0;

    கட்டமைப்பு ggml_cplan cplan;
    memset(&cplan, 0, அளவு(கட்டமைப்பு ggml_cplan));

    முழு max_tasks = 1;

    // thread scheduling for the different operations + work buffer size estimation
    க்கு (முழு i = 0; i < cgraph->n_nodes; i++) {
        கட்டமைப்பு ggml_tensor * முனை = cgraph->nodes[i];

        மாறிலி முழு n_tasks = ggml_get_n_tasks(முனை, n_threads);

        max_tasks = MAX(max_tasks, n_tasks);

        size_t cur = 0;

        எனில் (!ggml_cpu_extra_work_size(n_threads, முனை, &cur)) {
            தேர்வு (முனை->op) {
                நிலை GGML_OP_CPY:
                நிலை GGML_OP_DUP:
                    {
                        எனில் (ggml_is_quantized(முனை->type) ||
                            // F16 -> BF16 and BF16 -> F16 copies go through intermediate F32
                            (முனை->src[0]->type == GGML_TYPE_F16  && முனை->src[1] && முனை->src[1]->type == GGML_TYPE_BF16) ||
                            (முனை->src[0]->type == GGML_TYPE_BF16 && முனை->src[1] && முனை->src[1]->type == GGML_TYPE_F16) ||
                            // conversion between F32 and I32
                            (முனை->src[0]->type == GGML_TYPE_F32 && முனை->src[1] && முனை->src[1]->type == GGML_TYPE_I32) ||
                            (முனை->src[0]->type == GGML_TYPE_I32 && முனை->src[1] && முனை->src[1]->type == GGML_TYPE_F32)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * முனை->ne[0] * n_tasks;
                        }
                    } நிறுத்து;
                நிலை GGML_OP_ADD:
                நிலை GGML_OP_ADD_ID:
                நிலை GGML_OP_ADD1:
                    {
                        எனில் (ggml_is_quantized(முனை->src[0]->type)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * முனை->src[0]->ne[0] * n_tasks;
                        }
                    } நிறுத்து;
                நிலை GGML_OP_ACC:
                    {
                        எனில் (ggml_is_quantized(முனை->src[0]->type)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * முனை->src[1]->ne[0] * n_tasks;
                        }
                    } நிறுத்து;
                நிலை GGML_OP_மொத்தம்_EQUAL:
                    {
                        cur = ggml_type_size(முனை->type)*n_tasks;
                    } நிறுத்து;
                நிலை GGML_OP_MUL_MAT:
                    {
                        மாறிலி எண்ணிக்கை ggml_type vec_dot_type = type_traits_cpu[முனை->src[0]->type].vec_dot_type;
                        எனில் (முனை->src[1]->type != vec_dot_type) {
                            cur = ggml_row_size(vec_dot_type, ggml_nelements(முனை->src[1]));
                        }
                        // Workspace for tiled (see tiled.h)
                        cur = GGML_PAD(cur, 64);
                        cur += ggml_tiled_wdata_size(n_tasks, முனை);
                    } நிறுத்து;
                நிலை GGML_OP_MUL_MAT_ID:
                    {
                        cur = 0;
                        மாறிலி கட்டமைப்பு ggml_tensor * src0 = முனை->src[0];
                        மாறிலி கட்டமைப்பு ggml_tensor * src1 = முனை->src[1];
                        மாறிலி கட்டமைப்பு ggml_tensor * ids = முனை->src[2];
                        மாறிலி எண்ணிக்கை ggml_type vec_dot_type = type_traits_cpu[src0->type].vec_dot_type;
                        மாறிலி முழு n_as = src0->ne[2];
                        // src1
                        எனில் (src1->type != vec_dot_type) {
                            cur += ggml_row_size(vec_dot_type, ggml_nelements(src1)) + அளவு(int64_t);
                        }
                        // matrix_row_counts
                        cur += n_as * அளவு(int64_t) + அளவு(int64_t);
                        // matrix_rows
                        cur += n_as*ids->ne[0]*ids->ne[1]*அளவு(கட்டமைப்பு mmid_row_mapping) + அளவு(int64_t);
                        // atomic_current_chunk
                        cur += CACHE_வரி_SIZE*n_as + CACHE_வரி_SIZE;
                        // Workspace for tiled (see tiled.h)
                        cur = GGML_PAD(cur, 64);
                        cur += ggml_tiled_wdata_size(n_tasks, முனை);
                    } நிறுத்து;
                நிலை GGML_OP_OUT_PROD:
                    {
                        எனில் (ggml_is_quantized(முனை->src[0]->type) ||
                            முனை->src[0]->type == GGML_TYPE_F16) {
                            cur = ggml_type_size(GGML_TYPE_F32) * முனை->src[0]->ne[0] * n_tasks;
                        }
                    } நிறுத்து;
                நிலை GGML_OP_SET_ROWS:
                    {
                        எனில் (முனை->src[0]->type == GGML_TYPE_F16 && முனை->type != GGML_TYPE_F16) {
                            cur = ggml_type_size(GGML_TYPE_F32) * முனை->src[0]->ne[0] * n_tasks;
                        }
                    } நிறுத்து;
                நிலை GGML_OP_SOFT_MAX:
                நிலை GGML_OP_ROPE:
                நிலை GGML_OP_ROPE_BACK:
                    {
                        cur = ggml_type_size(GGML_TYPE_F32) * முனை->ne[0] * n_tasks;
                    } நிறுத்து;
                நிலை GGML_OP_CONV_TRANSPOSE_1D:
                    {
                        GGML_ASSERT(முனை->src[0]->ne[3] == 1);
                        GGML_ASSERT(முனை->src[1]->ne[2] == 1);
                        GGML_ASSERT(முனை->src[1]->ne[3] == 1);

                        மாறிலி int64_t ne00 = முனை->src[0]->ne[0];  // K
                        மாறிலி int64_t ne01 = முனை->src[0]->ne[1];  // Cout
                        மாறிலி int64_t ne02 = முனை->src[0]->ne[2];  // Cin
                        மாறிலி int64_t ne10 = முனை->src[1]->ne[0];  // L
                        மாறிலி int64_t ne11 = முனை->src[1]->ne[1];  // Cin

                        எனில் ((முனை->src[0]->type == GGML_TYPE_F16 ||
                             முனை->src[0]->type == GGML_TYPE_BF16) &&
                            முனை->src[1]->type == GGML_TYPE_F32) {
                            cur += அளவு(ggml_fp16_t)*ne00*ne01*ne02;
                            cur += அளவு(ggml_fp16_t)*ne10*ne11;
                        } இல்லையெனில் எனில் (முனை->src[0]->type == GGML_TYPE_F32 &&
                                   முனை->src[1]->type == GGML_TYPE_F32) {
                            cur += அளவு(தசம)*ne00*ne01*ne02;
                            cur += அளவு(தசம)*ne10*ne11;
                        } இல்லையெனில் {
                            GGML_ABORT("fatal error");
                        }
                    } நிறுத்து;
                நிலை GGML_OP_CONV_2D:
                நிலை GGML_OP_CONV_3D:
                    {
                        cur = GGML_IM2COL_WORK_SIZE;
                    } நிறுத்து;
                நிலை GGML_OP_CONV_TRANSPOSE_2D:
                    {
                        மாறிலி int64_t ne00 = முனை->src[0]->ne[0]; // W
                        மாறிலி int64_t ne01 = முனை->src[0]->ne[1]; // H
                        மாறிலி int64_t ne02 = முனை->src[0]->ne[2]; // Channels Out
                        மாறிலி int64_t ne03 = முனை->src[0]->ne[3]; // Channels In

                        மாறிலி int64_t ne10 = முனை->src[1]->ne[0]; // W
                        மாறிலி int64_t ne11 = முனை->src[1]->ne[1]; // H
                        மாறிலி int64_t ne12 = முனை->src[1]->ne[2]; // Channels In
                        மாறிலி int64_t ne13 = முனை->src[1]->ne[3]; // Batch

                        GGML_ASSERT(முனை->src[0]->type == GGML_TYPE_F16 || முனை->src[0]->type == GGML_TYPE_F32);
                        GGML_ASSERT(முனை->src[1]->type == GGML_TYPE_F32);

                        cur += ggml_type_size(முனை->src[0]->type) * ne00 * ne01 * ne02 * ne03;
                        cur += ggml_type_size(முனை->src[0]->type) * ne10 * ne11 * ne12 * ne13;

                    } நிறுத்து;
                நிலை GGML_OP_உச்சி_K:
                    {
                        cur += அளவு(int32_t)*முனை->src[0]->ne[0]*n_tasks;
                    } நிறுத்து;
                நிலை GGML_OP_FLASH_ATTN_EXT:
                    {
                        மாறிலி int64_t neq2 = முனை->src[0]->ne[2]; // number of query heads
                        மாறிலி int64_t DK = முனை->src[1]->ne[0];
                        மாறிலி int64_t DV = முனை->src[2]->ne[0];

                        // Tiled flash attention scratch (tile sizes defined in common.h)
                        // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
                        size_t prefill  = அளவு(தசம)*(GGML_FA_TILE_Q*DK + 2*GGML_FA_TILE_Q*GGML_FA_TILE_KV + GGML_FA_TILE_Q*DV + GGML_FA_TILE_KV*DV + GGML_FA_TILE_KV*DK)*n_tasks;

                        // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
                        // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
                        size_t n_chunks = n_tasks;
                        size_t decode   = அளவு(தசம)*(neq2*n_chunks*(2+DV) + n_tasks*(DK + 2*DV));

                        cur += MAX(prefill, decode);
                    } நிறுத்து;
                நிலை GGML_OP_FLASH_ATTN_BACK:
                    {
                        மாறிலி int64_t    D = முனை->src[0]->ne[0];
                        மாறிலி int64_t ne11 = ggml_up(முனை->src[1]->ne[1], GGML_SOFT_MAX_UNROLL);
                        மாறிலி int64_t mxDn = MAX(D, ne11) * 2; // *2 because of S and SM in ggml_compute_forward_flash_attn_back
                        எனில் (முனை->src[1]->type == GGML_TYPE_F32) {
                            cur  = அளவு(தசம)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += அளவு(தசம)*mxDn*n_tasks; // this is overestimated by x2
                        } இல்லையெனில் எனில் (முனை->src[1]->type == GGML_TYPE_F16) {
                            cur  = அளவு(தசம)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += அளவு(தசம)*mxDn*n_tasks; // this is overestimated by x2
                        } இல்லையெனில் எனில் (முனை->src[1]->type == GGML_TYPE_BF16) {
                            cur  = அளவு(தசம)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += அளவு(தசம)*mxDn*n_tasks; // this is overestimated by x2
                        }
                    } நிறுத்து;

                நிலை GGML_OP_CROSS_ENTROPY_LOSS:
                    {
                        cur = ggml_type_size(முனை->type)*(n_tasks + முனை->src[0]->ne[0]*n_tasks);
                    } நிறுத்து;
                நிலை GGML_OP_GATED_DELTA_NET:
                    {
                        மாறிலி int64_t S_v = முனை->src[2]->ne[0];
                        மாறிலி int64_t K   = ggml_get_op_params_i32(முனை, 0);
                        மாறிலி int64_t per_thread = S_v + (K > 1 ? S_v * S_v : 0);
                        cur = per_thread * அளவு(தசம) * n_tasks;
                    } நிறுத்து;
                நிலை GGML_OP_மொத்தம்:
                    {
                        GGML_ABORT("fatal error");
                    }
                நிலை GGML_OP_LIGHTNING_INDEXER:
                    {
                        // temp buffer for dequantizing lightning indexer keys
                        மாறிலி int64_t ne10 = முனை->src[1]->ne[0];
                        cur += அளவு(தசம)*ne10*n_tasks;
                    } நிறுத்து;
                இயல்பு:
                    நிறுத்து;
            }
        }

        work_size = MAX(work_size, cur);
    }

    எனில் (work_size > 0) {
        work_size += CACHE_வரி_SIZE*(n_threads);
    }

    cplan.threadpool = threadpool;
    cplan.n_threads  = MIN(max_tasks, n_threads);
    cplan.work_size  = work_size;
    cplan.work_data  = வெற்று;

    திரும்பு cplan;
}


// Try to fuse the current node with subsequent nodes for better performance.
// Returns the number of nodes skipped by fusion (>=1), or 0 if no fusion was applied.
நிலையான bool ggml_cpu_disable_fusion = false;  // initialized once in ggml_cpu_init(), read-only afterwards

நிலையான முழு ggml_cpu_try_fuse_ops(
        மாறிலி கட்டமைப்பு ggml_cgraph * cgraph,
        மாறிலி முழு முனை_n,
        மாறிலி கட்டமைப்பு ggml_compute_params * params,
        மாறிலி கட்டமைப்பு ggml_cplan * cplan) {

    எனில் (ggml_cpu_disable_fusion || cplan->use_ref) {
        திரும்பு 0;
    }

    கட்டமைப்பு ggml_tensor * முனை = cgraph->nodes[முனை_n];

    எனில் (முனை->op == GGML_OP_RMS_NORM) {
        // RMS_NORM + MUL fusion
        மாறிலி எண்ணிக்கை ggml_op fuse_ops[] = { GGML_OP_RMS_NORM, GGML_OP_MUL };
        எனில் (ggml_can_fuse(cgraph, முனை_n, fuse_ops, 2)) {
            கட்டமைப்பு ggml_tensor * mul_முனை = cgraph->nodes[முனை_n + 1];
            மாறிலி கட்டமைப்பு ggml_tensor * mul_w = (mul_முனை->src[0] == முனை)
                ? mul_முனை->src[1] : mul_முனை->src[0];
            எனில் (முனை->src[0]->type  == GGML_TYPE_F32 &&
                mul_முனை->type      == GGML_TYPE_F32 &&
                mul_w->type         == GGML_TYPE_F32 &&
                mul_w->ne[0]        == முனை->ne[0]   &&
                mul_w->nb[0]        == அளவு(தசம)) {

                ggml_compute_forward_rms_norm_mul_fused(params, முனை, mul_முனை);
                திரும்பு 1;
            }
        }
    }

    திரும்பு 0;
}

நிலையான thread_ret_t ggml_graph_compute_thread(வெறுமை * data) {
    கட்டமைப்பு ggml_compute_state * state = (கட்டமைப்பு ggml_compute_state *) data;
    கட்டமைப்பு ggml_threadpool    * tp    = state->threadpool;

    மாறிலி கட்டமைப்பு ggml_cgraph * cgraph = tp->cgraph;
    மாறிலி கட்டமைப்பு ggml_cplan  * cplan  = tp->cplan;

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(state->ith);
#இல்லையெனில்
    set_numa_thread_affinity(state->ith);
#endif

    கட்டமைப்பு ggml_compute_params params = {
        /*.ith        =*/ state->ith,
        /*.nth        =*/ atomic_load_explicit(&tp->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK,
        /*.wsize      =*/ cplan->work_size,
        /*.wdata      =*/ cplan->work_data,
        /*.threadpool =*/ tp,
        /*.use_ref    =*/ cplan->use_ref,
    };

#ifdef GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p\n", state->ith, (மாறிலி வெறுமை *)cplan);
#இல்லையெனில்
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p last-graph %d\n", state->ith, (மாறிலி வெறுமை *)cplan, state->last_graph);
#endif

    க்கு (முழு முனை_n = 0; முனை_n < cgraph->n_nodes && atomic_load_explicit(&tp->abort, memory_order_relaxed) != முனை_n; முனை_n++) {
        கட்டமைப்பு ggml_tensor * முனை = cgraph->nodes[முனை_n];

        எனில் (ggml_op_is_empty(முனை->op)) {
            // skip NOPs
            தொடர்;
        }

        எனில் ((முனை->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            தொடர்;
        }

        // TODO: move fused-op detection into ggml_graph_plan so fusion decisions are made once at planning time
        // Try fused ops, fall back to normal compute
        மாறிலி முழு n_fused = ggml_cpu_try_fuse_ops(cgraph, முனை_n, &params, cplan);
        எனில் (n_fused > 0) {
            முனை_n += n_fused;
        } இல்லையெனில் {
            ggml_compute_forward(&params, முனை);
        }

        எனில் (state->ith == 0 && cplan->abort_callback &&
                cplan->abort_callback(cplan->abort_callback_data)) {
            atomic_store_explicit(&tp->abort, முனை_n + 1, memory_order_relaxed);
            tp->ec    = GGML_STATUS_ABORTED;
        }

        எனில் (முனை_n + 1 < cgraph->n_nodes) {
            ggml_barrier(state->threadpool);
        }
    }

#ifdef GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p\n", state->ith, (மாறிலி வெறுமை *)cplan);
#இல்லையெனில்
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p last-graph %d\n", state->ith, (மாறிலி வெறுமை *)cplan, state->last_graph);
#endif

    ggml_barrier(state->threadpool);

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(state->ith);
#endif

    திரும்பு 0;
}

#ifndef GGML_USE_OPENMP

// check if thread is ready to proceed (exit from polling or sleeping)
// returns true if loops should exit, sets state->pending to indicate new work
நிலையான உள்வரி bool ggml_graph_compute_thread_ready(கட்டமைப்பு ggml_compute_state * state) {
    கட்டமைப்பு ggml_threadpool * threadpool = state->threadpool;

    எனில் (state->pending || threadpool->stop || threadpool->pause) { திரும்பு true; }

    // check for new graph/work
    முழு n_graph   = atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed);
    முழு n_threads = n_graph & GGML_THREADPOOL_N_THREADS_MASK;
    எனில் (n_graph != state->last_graph) {
        state->pending    = (state->ith < n_threads);
        state->last_graph = n_graph;
        திரும்பு true;
    }

    திரும்பு false;
}

// sync thread state after polling
நிலையான உள்வரி வெறுமை ggml_graph_compute_thread_sync(கட்டமைப்பு ggml_compute_state * state) {
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&state->threadpool->n_graph, 0, memory_order_seq_cst);
    #இல்லையெனில்
    atomic_thread_fence(memory_order_seq_cst);
    #endif
    UNUSED(state);
}

நிலையான உள்வரி bool ggml_graph_compute_poll_for_work(கட்டமைப்பு ggml_compute_state * state) {
    கட்டமைப்பு ggml_threadpool * threadpool = state->threadpool;

    // This seems to make 0 ... 100 a decent range for polling level across modern processors.
    // Perhaps, we can adjust it dynamically based on load and things.
    மாறிலி uint64_t n_rounds = 1024UL * 128 * threadpool->poll;

    க்கு (uint64_t i=0; !ggml_graph_compute_thread_ready(state) && i < n_rounds; i++) {
        // No new work. Keep polling.
        ggml_thread_cpu_relax();
    }

    திரும்பு state->pending;
}

நிலையான உள்வரி bool ggml_graph_compute_check_for_work(கட்டமைப்பு ggml_compute_state * state) {
    கட்டமைப்பு ggml_threadpool * threadpool = state->threadpool;

    எனில் (ggml_graph_compute_poll_for_work(state)) {
        ggml_graph_compute_thread_sync(state);
        திரும்பு state->pending;
    }

    ggml_mutex_lock_shared(&threadpool->mutex);
    போது (!ggml_graph_compute_thread_ready(state)) {
        // No new work. Wait for the signal.
        GGML_PRINT_DEBUG("thread #%d waiting for work (sleeping)\n", state->ith);
        ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
    }
    ggml_mutex_unlock_shared(&threadpool->mutex);

    திரும்பு state->pending;
}

நிலையான thread_ret_t ggml_graph_compute_secondary_thread(வெறுமை* data) {
    கட்டமைப்பு ggml_compute_state * state = (கட்டமைப்பு ggml_compute_state *) data;
    கட்டமைப்பு ggml_threadpool * threadpool = state->threadpool;

    ggml_thread_apply_priority(threadpool->prio);
    எனில் (ggml_thread_cpumask_is_valid(state->cpumask)) {
        ggml_thread_apply_affinity(state->cpumask);
    }

    போது (true) {
        // Check if we need to sleep
        போது (threadpool->pause) {
            GGML_PRINT_DEBUG("thread #%d inside pause loop\n", state->ith);
            ggml_mutex_lock_shared(&threadpool->mutex);
            எனில் (threadpool->pause) {
                ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
            }
            GGML_PRINT_DEBUG("thread #%d resuming after wait\n", state->ith);
            ggml_mutex_unlock_shared(&threadpool->mutex);
        }

        // This needs to be checked for after the cond_wait
        எனில் (threadpool->stop) நிறுத்து;

        // Check if there is new work
        // The main thread is the only one that can dispatch new work

        ggml_graph_compute_check_for_work(state);
        எனில் (state->pending) {
            state->pending = false;
            ggml_graph_compute_thread(state);
        }
    }

    திரும்பு (thread_ret_t) 0;
}

// Start processing new graph
நிலையான வெறுமை ggml_graph_compute_kickoff(கட்டமைப்பு ggml_threadpool * threadpool, முழு n_threads)
{
    // Always take the mutex here because the worker threads are doing hybrid poll/wait

    ggml_mutex_lock(&threadpool->mutex);

    // Update the number of active threads and the graph count
    முழு n_graph = atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed) >> GGML_THREADPOOL_N_THREADS_BITS;
    n_graph = ((n_graph + 1) << GGML_THREADPOOL_N_THREADS_BITS) | (n_threads & GGML_THREADPOOL_N_THREADS_MASK);

    GGML_PRINT_DEBUG("compute-kickoff: n_threads %d n_graph %d\n", n_threads, n_graph);

    // Indicate the graph is ready to be processed
    // We need the full seq-cst fence here because of the polling threads (used in thread_sync)
    atomic_store_explicit(&threadpool->n_graph, n_graph, memory_order_seq_cst);

    எனில் (threadpool->pause) {
       // Update main thread prio and affinity to match the threadpool settings
       ggml_thread_apply_priority(threadpool->prio);
       எனில் (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
           ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
       }

       // resume does cond broadcast
       ggml_threadpool_resume_locked(threadpool);
    } இல்லையெனில் {
       ggml_cond_broadcast(&threadpool->cond);
    }

    ggml_mutex_unlock(&threadpool->mutex);
}

#endif // GGML_USE_OPENMP

நிலையான கட்டமைப்பு ggml_threadpool * ggml_threadpool_new_impl(
    கட்டமைப்பு ggml_threadpool_params * tpp,
               கட்டமைப்பு ggml_cgraph * cgraph,
                கட்டமைப்பு ggml_cplan * cplan) {

    கட்டமைப்பு ggml_threadpool * threadpool =
        ggml_aligned_malloc(அளவு(கட்டமைப்பு ggml_threadpool));
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
        threadpool->workers          = வெற்று;
        threadpool->n_threads        = tpp->n_threads;
        threadpool->poll             = tpp->poll;
        threadpool->prio             = tpp->prio;
        threadpool->ec               = GGML_STATUS_SUCCESS;
    }

    // Allocate and init workers state
    மாறிலி size_t workers_size = அளவு(கட்டமைப்பு ggml_compute_state) * tpp->n_threads;
    கட்டமைப்பு ggml_compute_state * workers = ggml_aligned_malloc(workers_size);

    memset(workers, 0, workers_size);
    க்கு (முழு j = 0; j < tpp->n_threads; j++) {
        workers[j].threadpool = threadpool;
        workers[j].ith        = j;
    }

    threadpool->workers = workers;

#ifdef GGML_USE_OPENMP
    int32_t cpumask_iter = 0;

    // Compute CPU masks for each thread
    க்கு (முழு j = 0; j < tpp->n_threads; j++) {
        ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);
    }
#இல்லையெனில் // GGML_USE_OPENMP
    ggml_mutex_init(&threadpool->mutex);
    ggml_cond_init(&threadpool->cond);

    // Spin the threads for all workers, and update CPU placements.
    // Place the main thread last (towards the higher numbered CPU cores).

    int32_t cpumask_iter = 0;

    க்கு (முழு j = 1; j < tpp->n_threads; j++) {
        ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);

        int32_t rc = ggml_thread_create(&workers[j].thrd, வெற்று, ggml_graph_compute_secondary_thread, &workers[j]);
        GGML_ASSERT(rc == 0);
    }

    ggml_thread_cpumask_next(tpp->cpumask, workers[0].cpumask, tpp->strict_cpu, &cpumask_iter);

    எனில் (!threadpool->pause) {
        // Update main thread prio and affinity at the start, otherwise we'll do it in resume
        ggml_thread_apply_priority(threadpool->prio);
        எனில் (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
            ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
        }
    }
#endif // GGML_USE_OPENMP

    திரும்பு threadpool;
}

கட்டமைப்பு ggml_threadpool * ggml_threadpool_new(கட்டமைப்பு ggml_threadpool_params * tpp) {
    திரும்பு ggml_threadpool_new_impl(tpp, வெற்று, வெற்று);
}

எண்ணிக்கை ggml_status ggml_graph_compute(கட்டமைப்பு ggml_cgraph * cgraph, கட்டமைப்பு ggml_cplan * cplan) {
    ggml_cpu_init();

    GGML_ASSERT(cplan);
    GGML_ASSERT(cplan->n_threads > 0);
    GGML_ASSERT(cplan->work_size == 0 || cplan->work_data != வெற்று);

    முழு n_threads                               = cplan->n_threads;
    கட்டமைப்பு ggml_threadpool * threadpool = cplan->threadpool;

    bool disposable_threadpool = false;

    எனில் (threadpool == வெற்று) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
        disposable_threadpool = true;

        கட்டமைப்பு ggml_threadpool_params ttp = ggml_threadpool_params_default(n_threads);
        threadpool = ggml_threadpool_new_impl(&ttp, cgraph, cplan);
    } இல்லையெனில் {
        // Reset some of the parameters that need resetting
        // No worker threads should be accessing the parameters below at this stage
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->current_chunk    = 0;
        threadpool->abort            = -1;
        threadpool->ec               = GGML_STATUS_SUCCESS;
    }

#ifdef GGML_USE_OPENMP
    எனில் (n_threads > 1) {
        #pragma omp parallel இலக்கம்_threads(n_threads)
        {
            #pragma omp single
            {
                // update the number of threads from the actual number of threads that we got from OpenMP
                n_threads = omp_get_இலக்கம்_threads();
                atomic_store_explicit(&threadpool->n_graph, n_threads, memory_order_relaxed);
            }

            // Apply thread CPU mask and priority
            முழு ith = omp_get_thread_இலக்கம்();

            ggml_thread_apply_priority(threadpool->prio);
            எனில் (ggml_thread_cpumask_is_valid(threadpool->workers[ith].cpumask)) {
                ggml_thread_apply_affinity(threadpool->workers[ith].cpumask);
            }
            ggml_graph_compute_thread(&threadpool->workers[ith]);
        }
    } இல்லையெனில் {
        atomic_store_explicit(&threadpool->n_graph, 1, memory_order_relaxed);
        ggml_graph_compute_thread(&threadpool->workers[0]);
    }
#இல்லையெனில்
    எனில் (n_threads > threadpool->n_threads) {
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

    எண்ணிக்கை ggml_status ret = threadpool->ec;

    எனில் (disposable_threadpool) {
        ggml_threadpool_free(threadpool);
    }

    திரும்பு ret;
}

எண்ணிக்கை ggml_status ggml_graph_compute_with_ctx(கட்டமைப்பு ggml_context * ctx, கட்டமைப்பு ggml_cgraph * cgraph, முழு n_threads) {
    கட்டமைப்பு ggml_cplan cplan = ggml_graph_plan(cgraph, n_threads, வெற்று);

    cplan.work_data = (uint8_t *)ggml_new_buffer(ctx, cplan.work_size);

    திரும்பு ggml_graph_compute(cgraph, &cplan);
}

வெறுமை ggml_cpu_fp32_to_fp32(மாறிலி தசம * x, தசம * y, int64_t n) {
    memcpy(y, x, n * அளவு(தசம));
}

வெறுமை ggml_cpu_fp32_to_fp16(மாறிலி தசம * x, ggml_fp16_t * y, int64_t n) {
    int64_t i = 0;
#எனில் defined(__F16C__)
#எனில் defined(__AVX512F__)
    க்கு (; i + 15 < n; i += 16) {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        __m256i y_vec = _mm512_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256((__m256i *)(y + i), y_vec);
    }
#endif
    க்கு (; i + 7 < n; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m128i y_vec = _mm256_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i *)(y + i), y_vec);
    }
    க்கு (; i + 3 < n; i += 4) {
        __m128 x_vec = _mm_loadu_ps(x + i);
        __m128i y_vec = _mm_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storel_epi64((__m128i *)(y + i), y_vec);
    }
#அல்லதெனில் defined(__riscv_zvfh)
    க்கு (முழு vl; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(&x[i], vl);
        vfloat16m1_t vy = __riscv_vfncvt_f_f_w_f16m1(vx, vl);
        __riscv_vse16_v_f16m1((_Float16 *)&y[i], vy, vl);
    }
#அல்லதெனில் defined(__VXE__) || defined(__VXE2__)
    க்கு (; i + 7 < n; i += 8) {
        மாறிலி uint32x4_t v_yl = __lzs_f32cx4_to_f16(vec_xl(0, x + i + 0));
        மாறிலி uint32x4_t v_yh = __lzs_f32cx4_to_f16(vec_xl(0, x + i + 4));
        vec_xst(vec_pack(v_yl, v_yh), 0, (uint16_t *)(y + i));
    }
#endif
    க்கு (; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(x[i]);
    }
}

வெறுமை ggml_cpu_fp16_to_fp32(மாறிலி ggml_fp16_t * x, தசம * y, int64_t n) {
    int64_t i = 0;
#எனில் defined(__F16C__)
#எனில் defined(__AVX512F__)
    க்கு (; i + 15 < n; i += 16) {
        __m256i x_vec = _mm256_loadu_si256((மாறிலி __m256i *)(x + i));
        __m512 y_vec = _mm512_cvtph_ps(x_vec);
        _mm512_storeu_ps(y + i, y_vec);
    }
#endif
    க்கு (; i + 7 < n; i += 8) {
        __m128i x_vec = _mm_loadu_si128((மாறிலி __m128i *)(x + i));
        __m256 y_vec = _mm256_cvtph_ps(x_vec);
        _mm256_storeu_ps(y + i, y_vec);
    }
    க்கு (; i + 3 < n; i += 4) {
        __m128i x_vec = _mm_loadl_epi64((மாறிலி __m128i *)(x + i));
        __m128 y_vec = _mm_cvtph_ps(x_vec);
        _mm_storeu_ps(y + i, y_vec);
    }

#அல்லதெனில் defined(__riscv_v_intrinsic) && defined(__riscv_zvfhmin)
    // calculate step size
    மாறிலி முழு epr = __riscv_vsetvlmax_e16m2();
    மாறிலி முழு step = epr * 2;
    மாறிலி முழு np = (n & ~(step - 1));

    // unroll by 2
    க்கு (; i < np; i += step) {
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((மாறிலி _Float16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vfloat16m2_t ax1 = __riscv_vle16_v_f16m2((மாறிலி _Float16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvt_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    முழு vl;
    க்கு (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((மாறிலி _Float16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }

#endif

    க்கு (; i < n; ++i) {
        y[i] = GGML_CPU_FP16_TO_FP32(x[i]);
    }
}

வெறுமை ggml_cpu_fp32_to_bf16(மாறிலி தசம * x, ggml_bf16_t * y, int64_t n) {
    int64_t i = 0;
    க்கு (; i < n; ++i) {
        y[i] = GGML_FP32_TO_BF16(x[i]);
    }
}

வெறுமை ggml_cpu_fp32_to_i32(மாறிலி தசம * x, int32_t * y, int64_t n) {
    int64_t i = 0;
    க்கு (; i < n; ++i) {
        y[i] = x[i];
    }
}

வெறுமை ggml_cpu_bf16_to_fp32(மாறிலி ggml_bf16_t * x, தசம * y, int64_t n) {
    int64_t i = 0;
#எனில் defined(__AVX2__)
#எனில் defined(__AVX512F__)
    க்கு (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i,
                        _mm512_castsi512_ps(
                            _mm512_slli_epi32(
                                _mm512_cvtepu16_epi32(
                                    _mm256_loadu_si256(
                                        (மாறிலி __m256i *)(x + i))),
                                16)));
    }
#endif
    க்கு (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i,
                        _mm256_castsi256_ps(
                            _mm256_slli_epi32(
                                _mm256_cvtepu16_epi32(
                                    _mm_loadu_si128(
                                        (மாறிலி __m128i *)(x + i))),
                                16)));
    }
#அல்லதெனில் defined(__riscv_v_intrinsic) && defined(__riscv_zvfbfmin)
    // calculate step size
    மாறிலி முழு epr = __riscv_vsetvlmax_e16m2();
    மாறிலி முழு step = epr * 2;
    மாறிலி முழு np = (n & ~(step - 1));

    // unroll by 2
    க்கு (; i < np; i += step) {
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((மாறிலி __bf16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vbfloat16m2_t ax1 = __riscv_vle16_v_bf16m2((மாறிலி __bf16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    முழு vl;
    க்கு (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((மாறிலி __bf16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }
#endif
    க்கு (; i < n; i++) {
        y[i] = GGML_BF16_TO_FP32(x[i]);
    }
}

முழு ggml_cpu_has_avx(வெறுமை) {
#எனில் defined(__AVX__)
    திரும்பு 1;
#இல்லையெனில்
    திரும்பு 0;
#endif
}

முழு ggml_cpu_has_avx_vnni(வெறுமை) {
#எனில் defined(__AVXVNNI__)
    திரும்பு 1;
#இல்லையெனில்
    திரும்பு 0;
#endif
}

முழு ggml_cpu_has_avx2(வெறுமை) {
#எனில் defined(__AVX2__)
    திரும்பு 1;
#இல்லையெனில்
    திரும்பு 0;
#endif
}

முழு ggml_cpu_has_avx512(வெறுமை) {
#எனில் defined(__AVX512F__)
    திரும்பு 1;
#இல்லையெனில்
    திரும்பு 0;
#endif
}

முழு ggml_cpu_has_avx512_vbmi(வெறுமை) {
#எனில் defined(__AVX512VBMI__)
    திரும்பு 1;
#இல்லையெனில்
    திரும்பு 0;
#endif
}

முழு ggml_cpu_has_avx512_vnni(வெறுமை) {
#எனில் defined(__AVX512VNNI__)
    திரும்பு 1;
#இல்லையெனில்
    திரும்பு 0;
#endif
}

முழு ggml_cpu_has_avx512_bf16(வெறுமை) {
#எனில் defined(__AVX512BF16__)
    திரும்பு 1;
#இல்லையெனில்
    திரும்பு 0;
#endif
}

முழு ggml_cpu_has_amx_int8(வெறுமை) {
#எனில் defined(__AMX_INT8__)
    திரும்பு 1;
#இல்லையெனில்
    திரும்பு 0;
#endif
}

முழு ggml_cpu_has_bmi2(வெறுமை) {
#எனில் defined(__BMI2__)
    திரும்பு 1;
#இல்லையெனில்
    திரும்பு 0;
#endif
}

முழு ggml_cpu_has_fma(வெறுமை) {
#எனில் defined(__FMA__)
    திரும்பு 1;
#இல்லையெனில்
    திரும்பு 0;
#endif
}

முழு ggml_cpu_has_arm_fma(வெறுமை) {
#எனில் defined(__ARM_FEATURE_FMA)
    திரும்பு 1;
#இல்லையெனில்
    திரும்பு 0;
#endif
}

முழு ggml_cpu_has_riscv_v(வெறுமை) {
#எனில் defined(__riscv_v_intrinsic)
    திரும்பு 1;
#இல்லையெனில்
    திரும்பு 0;
#endif
}

முழு ggml_cpu_get_rvv_vlen(வெறுமை) {
#எனில் defined(__riscv) && defined(__riscv_v_intrinsic)
    திரும்பு ggml_riscv_arch_features.rvv_vlen;
#இல்லையெனில்
    திரும்பு 0;
#endif
}

முழு ggml_cpu_has_f16c(வெறுமை) {
#எனில் defined(__F16C__)
    திரும்பு 1;
#இல்லையெனில்
    திரும்பு 0;
#endif
}

முழு ggml_cpu_has_fp16_va(வெறுமை) {
#எனில் defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    திரும்பு 1;
#இல்லையெனில்
    திரும்பு 0;
#endif
}

முழு ggml_cpu_has_wasm_simd(வெறுமை) {
#எனில் defined(__wasm_simd128__)
    திரும்பு 1;
#இல்லையெனில்
    திரும்பு 0;
#endif
}

முழு ggml_cpu_has_llamafile(வெறுமை) {
#எனில் defined(GGML_USE_LLAMAFILE)
    திரும்பு 1;
#இல்லையெனில்
    திரும்பு 0;
#endif
}

முழு ggml_cpu_has_sse3(வெறுமை) {
#எனில் defined(__SSE3__)
    திரும்பு 1;
#இல்லையெனில்
    திரும்பு 0;
#endif
}

முழு ggml_cpu_has_ssse3(வெறுமை) {
#எனில் defined(__SSSE3__)
    திரும்பு 1;
#இல்லையெனில்
    திரும்பு 0;
#endif
}

முழு ggml_cpu_has_vsx(வெறுமை) {
#எனில் defined(__POWER9_VECTOR__)
    திரும்பு 1;
#இல்லையெனில்
    திரும்பு 0;
#endif
}

முழு ggml_cpu_has_vxe(வெறுமை) {
#எனில் defined(__VXE__) || defined(__VXE2__)
    திரும்பு 1;
#இல்லையெனில்
    திரும்பு 0;
#endif
}

முழு ggml_cpu_has_neon(வெறுமை) {
#எனில் defined(__ARM_ARCH) && defined(__ARM_NEON)
    திரும்பு 1;
#இல்லையெனில்
    திரும்பு 0;
#endif
}

முழு ggml_cpu_has_dotprod(வெறுமை) {
#எனில் defined(__ARM_ARCH) && defined(__ARM_FEATURE_DOTPROD)
    திரும்பு 1;
#இல்லையெனில்
    திரும்பு 0;
#endif
}

முழு ggml_cpu_has_sve(வெறுமை) {
#எனில் defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    திரும்பு 1;
#இல்லையெனில்
    திரும்பு 0;
#endif
}

முழு ggml_cpu_has_matmul_int8(வெறுமை) {
#எனில் defined(__ARM_ARCH) && defined(__ARM_FEATURE_MATMUL_INT8)
    திரும்பு 1;
#இல்லையெனில்
    திரும்பு 0;
#endif
}

முழு ggml_cpu_get_sve_cnt(வெறுமை) {
#எனில் defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    திரும்பு ggml_arm_arch_features.sve_cnt;
#இல்லையெனில்
    திரும்பு 0;
#endif
}

முழு ggml_cpu_has_sme(வெறுமை) {
#எனில் defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME)
    திரும்பு 1;
#இல்லையெனில்
    திரும்பு 0;
#endif
}

முழு ggml_cpu_has_sme2(வெறுமை) {
#எனில் defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME2)
    திரும்பு 1;
#இல்லையெனில்
    திரும்பு 0;
#endif
}

வெறுமை ggml_cpu_init(வெறுமை) {
    // needed to initialize ggml_time
    {
        கட்டமைப்பு ggml_init_params params = { 0, வெற்று, false };
        கட்டமைப்பு ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    ggml_critical_section_start();

    நிலையான bool is_first_call = true;

    எனில் (is_first_call) {
        // initialize GELU, Quick GELU, SILU and EXP F32 tables
        {
            மாறிலி uint64_t t_start = ggml_time_us(); UNUSED(t_start);

            க்கு (முழு i = 0; i < (1 << 16); ++i) {
                ஒன்றிப்பு {
                    uint16_t u16;
                    ggml_fp16_t fp16;
                } u = {i};
                தசம f = GGML_COMPUTE_FP16_TO_FP32(u.fp16);
                ggml_table_f32_f16[i] = f;
                ggml_table_gelu_f16[i] = GGML_CPU_FP32_TO_FP16(ggml_gelu_f32(f));
                ggml_table_gelu_quick_f16[i] = GGML_CPU_FP32_TO_FP16(ggml_gelu_quick_f32(f));
            }

            // initialize E8M0 half table (256 entries)
            க்கு (முழு i = 0; i < (1 << 8); ++i) {
                ggml_table_f32_e8m0_half[i] = GGML_E8M0_TO_FP32_HALF(i);
            }

            // initialize UE4M3 table (256 entries)
            க்கு (முழு i = 0; i < (1 << 8); ++i) {
                ggml_table_f32_ue4m3[i] = ggml_ue4m3_to_fp32(i);
            }

            மாறிலி uint64_t t_end = ggml_time_us(); UNUSED(t_end);

            GGML_PRINT_DEBUG("%s: GELU, Quick GELU, SILU and EXP tables initialized in %f ms\n", __func__, (t_end - t_start)/1000.0);

#ifdef GGML_USE_OPENMP
            //if (!getenv("OMP_WAIT_POLICY")) {
            //    // set the wait policy to active, so that OpenMP threads don't sleep
            //    setenv("OMP_WAIT_POLICY", "active", 0)
            //}

            எனில் (!getenv("KMP_BLOCKTIME")) {
                // set the time to wait before sleeping a thread
                // this is less aggressive than setting the wait policy to active, but should achieve similar results in most cases
#ifdef _WIN32
                _putenv_s("KMP_BLOCKTIME", "200"); // 200ms
#இல்லையெனில்
                setenv("KMP_BLOCKTIME", "200", 0); // 200ms
#endif
            }
#endif
        }

#எனில் defined(__ARM_ARCH)
        ggml_init_arm_arch_features();
#endif

#எனில் defined(__riscv)
        ggml_init_riscv_arch_features();
#endif

        {
            மாறிலி எழுத்து * env = getenv("GGML_CPU_DISABLE_FUSION");
            ggml_cpu_disable_fusion = (env != வெற்று && atoi(env) == 1);
        }

        is_first_call = false;
    }

    ggml_critical_section_end();
}
