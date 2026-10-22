#define _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#define _USE_MATH_DEFINES // For M_PI on MSVC

#অন্তর্ভাৱ "ggml-backend-impl.h"
#অন্তর্ভাৱ "ggml-backend.h"
#অন্তর্ভাৱ "traits.h"
#অন্তর্ভাৱ "ggml-cpu-impl.h"
#অন্তর্ভাৱ "ggml-impl.h"
#অন্তর্ভাৱ "quants.h"
#অন্তর্ভাৱ "ggml-threading.h"
#অন্তর্ভাৱ "unary-ops.h"
#অন্তর্ভাৱ "binary-ops.h"
#অন্তর্ভাৱ "vec.h"
#অন্তর্ভাৱ "ops.h"
#অন্তর্ভাৱ "ggml.h"
#অন্তর্ভাৱ "common.h"
#অন্তর্ভাৱ "tiled/tiled.h"

#যদি defined(_MSC_VER) || defined(__MINGW32__)
#অন্তর্ভাৱ <malloc.h> // using malloc.h with MSC/MINGW
#নোচেদ্যদি !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#অন্তর্ভাৱ <alloca.h>
#endif

#অন্তর্ভাৱ <assert.h>
#অন্তর্ভাৱ <errno.h>
#অন্তর্ভাৱ <time.h>
#অন্তর্ভাৱ <math.h>
#অন্তর্ভাৱ <stdlib.h>
#অন্তর্ভাৱ <string.h>
#অন্তর্ভাৱ <stdint.h>
#অন্তর্ভাৱ <inttypes.h>
#অন্তর্ভাৱ <stdio.h>
#অন্তর্ভাৱ <float.h>
#অন্তর্ভাৱ <limits.h>
#অন্তর্ভাৱ <stdarg.h>
#অন্তর্ভাৱ <signal.h>
#যদি defined(__gnu_linux__)
#অন্তর্ভাৱ <syscall.h>
#endif

#ifdef GGML_USE_OPENMP
#অন্তর্ভাৱ <omp.h>
#endif

#যদি defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_MATMUL_INT8)
#undef GGML_USE_LLAMAFILE
#endif

#ifdef GGML_USE_LLAMAFILE
#অন্তর্ভাৱ "llamafile/sgemm.h"
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
#    include "spacemit/ime.h"
#endif

// Note: once we move threading into a separate C++ file
// will use std::hardware_destructive_interference_size instead of hardcoding it here
// and we'll use C++ attribute syntax.
#define GGML_CACHE_সাৰি  64

#যদি defined(__clang__) || defined(__GNUC__)
#define GGML_CACHE_ALIGN __attribute__((aligned(GGML_CACHE_সাৰি)))
#endif

#যদি defined(__has_feature)
#যদি __has_feature(thread_sanitizer)
#define GGML_TSAN_ENABLED 1
#endif
#অন্যথা  // __has_feature
#যদি defined(__SANITIZE_THREAD__)
#define GGML_TSAN_ENABLED 1
#endif
#endif // __has_feature

#define UNUSED GGML_UNUSED
#define SWAP(x, y, T) কৰা { T SWAP = x; (x) = y; (y) = SWAP; } যেতিয়ালৈ (0)

// precomputed f32 table for f16 (256 KB) (simd-mappings.h)
ভিন্ন ggml_table_f32_f16[1 << 16];

// precomputed f32 table for e8m0 half (1 KB) (simd-mappings.h)
ভিন্ন ggml_table_f32_e8m0_half[1 << 8];

// precomputed f32 table for ue4m3 (1 KB) (simd-mappings.h)
ভিন্ন ggml_table_f32_ue4m3[1 << 8];

#যদি defined(__ARM_ARCH)
সংরচনা ggml_arm_arch_features_type {
    পূর্ণাঙ্ক sve_cnt;
} ggml_arm_arch_features = { 0 };
#endif

#যদি defined(__riscv)
সংরচনা ggml_riscv_arch_features_type {
    পূর্ণাঙ্ক rvv_vlen;
} ggml_riscv_arch_features = { 0 };
#endif

#যদি defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#অন্তর্ভাৱ <windows.h>

#যদি defined(_MSC_VER) && !defined(__clang__)
#define GGML_CACHE_ALIGN __declspec(align(GGML_CACHE_সাৰি))

প্রকারনির্দেশ চঞ্চল LONG atomic_int;
প্রকারনির্দেশ atomic_int atomic_bool;
প্রকারনির্দেশ atomic_int atomic_flag;

#define ATOMIC_FLAG_INIT 0

প্রকারনির্দেশ গণনা {
    memory_order_relaxed,
    memory_order_consume,
    memory_order_acquire,
    memory_order_release,
    memory_order_acq_rel,
    memory_order_seq_cst
} memory_order;

স্থির শূন্য atomic_store(atomic_int * ptr, LONG val) {
    InterlockedExchange(ptr, val);
}
স্থির শূন্য atomic_store_explicit(atomic_int * ptr, LONG val, memory_order mo) {
    // TODO: add support for explicit memory order
    InterlockedExchange(ptr, val);
}
স্থির LONG atomic_load(atomic_int * ptr) {
    ঘূৰাই InterlockedCompareExchange(ptr, 0, 0);
}
স্থির LONG atomic_load_explicit(atomic_int * ptr, memory_order mo) {
    // TODO: add support for explicit memory order
    ঘূৰাই InterlockedCompareExchange(ptr, 0, 0);
}
স্থির LONG atomic_fetch_add(atomic_int * ptr, LONG inc) {
    ঘূৰাই InterlockedExchangeAdd(ptr, inc);
}
স্থির LONG atomic_fetch_add_explicit(atomic_int * ptr, LONG inc, memory_order mo) {
    // TODO: add support for explicit memory order
    ঘূৰাই InterlockedExchangeAdd(ptr, inc);
}
স্থির atomic_bool atomic_flag_test_and_set(atomic_flag * ptr) {
    ঘূৰাই InterlockedExchange(ptr, 1);
}
স্থির শূন্য atomic_flag_clear(atomic_flag * ptr) {
    InterlockedExchange(ptr, 0);
}
স্থির শূন্য atomic_thread_fence(memory_order mo) {
    MemoryBarrier();
}
#অন্যথা // clang
#অন্তর্ভাৱ <stdatomic.h>
#endif

প্রকারনির্দেশ HANDLE pthread_t;

প্রকারনির্দেশ DWORD thread_ret_t;
স্থির পূর্ণাঙ্ক pthread_create(pthread_t * out, শূন্য * unused, thread_ret_t(*func)(শূন্য *), শূন্য * arg) {
    (শূন্য) unused;
    HANDLE handle = CreateThread(রিক্ত, 0, (LPTHREAD_START_ROUTINE) func, arg, 0, রিক্ত);
    যদি (handle == রিক্ত)
    {
        ঘূৰাই EAGAIN;
    }

    *out = handle;
    ঘূৰাই 0;
}

স্থির পূর্ণাঙ্ক pthread_join(pthread_t thread, শূন্য * unused) {
    (শূন্য) unused;
    পূর্ণাঙ্ক ret = (পূর্ণাঙ্ক) WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    ঘূৰাই ret;
}

স্থির পূর্ণাঙ্ক sched_yield (শূন্য) {
    Sleep (0);
    ঘূৰাই 0;
}
#অন্যথা

#অন্তর্ভাৱ <pthread.h>
#অন্তর্ভাৱ <stdatomic.h>
#অন্তর্ভাৱ <sched.h>
#যদি defined(__FreeBSD__)
#অন্তর্ভাৱ <pthread_np.h>
#endif

প্রকারনির্দেশ শূন্য * thread_ret_t;

#অন্তর্ভাৱ <sys/types.h>
#অন্তর্ভাৱ <sys/stat.h>
#অন্তর্ভাৱ <unistd.h>

#endif

প্রকারনির্দেশ pthread_t ggml_thread_t;

#define GGML_THREADPOOL_N_THREADS_MASK (0xffffU)
#define GGML_THREADPOOL_N_THREADS_BITS (16)

#যদি defined(__APPLE__)
#অন্তর্ভাৱ <unistd.h>
#অন্তর্ভাৱ <mach/mach.h>
#অন্তর্ভাৱ <TargetConditionals.h>
#endif

স্থির ধ্রুৱ সংরচনা ggml_type_traits_cpu type_traits_cpu[GGML_TYPE_গুনতি] = {
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
#যদি defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#অন্যথা
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q4_1] = {
        .from_float               = quantize_row_q4_1,
        .vec_dot                  = ggml_vec_dot_q4_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
#যদি defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#অন্যথা
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
#যদি defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#অন্যথা
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
#যদি defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#অন্যথা
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
#যদি defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#অন্যথা
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_IQ2_XXS] = {
        .from_float               = রিক্ত,
        .vec_dot                  = ggml_vec_dot_iq2_xxs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_XS] = {
        .from_float               = রিক্ত,
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
        .from_float               = রিক্ত,
        .vec_dot                  = ggml_vec_dot_iq1_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_M] = {
        .from_float               = রিক্ত,
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

ধ্রুৱ সংরচনা ggml_type_traits_cpu * ggml_get_type_traits_cpu(গণনা ggml_type type) {
    ঘূৰাই &type_traits_cpu[type];
}

//
// Threading defs
//

প্রকারনির্দেশ pthread_t          ggml_thread_t;

#যদি defined(_WIN32)

প্রকারনির্দেশ CONDITION_VARIABLE ggml_cond_t;
প্রকারনির্দেশ SRWLOCK            ggml_mutex_t;

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

#অন্যথা

প্রকারনির্দেশ pthread_cond_t     ggml_cond_t;
প্রকারনির্দেশ pthread_mutex_t    ggml_mutex_t;

#define ggml_mutex_init(m)          pthread_mutex_init(m, রিক্ত)
#define ggml_mutex_destroy(m)       pthread_mutex_destroy(m)
#define ggml_mutex_lock(m)          pthread_mutex_lock(m)
#define ggml_mutex_unlock(m)        pthread_mutex_unlock(m)
#define ggml_mutex_lock_shared(m)   pthread_mutex_lock(m)
#define ggml_mutex_unlock_shared(m) pthread_mutex_unlock(m)

#define ggml_lock_init(x)    UNUSED(x)
#define ggml_lock_destroy(x) UNUSED(x)
#যদি defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64) && !defined(_M_ARM64EC))
#define ggml_lock_lock(x)    _mm_pause()
#অন্যথা
#define ggml_lock_lock(x)    UNUSED(x)
#endif
#define ggml_lock_unlock(x)  UNUSED(x)

#define GGML_LOCK_INITIALIZER 0
#define ggml_cond_init(c)      pthread_cond_init(c, রিক্ত)
#define ggml_cond_destroy(c)   pthread_cond_destroy(c)
#define ggml_cond_wait(c, m)   pthread_cond_wait(c, m)
#define ggml_cond_broadcast(c) pthread_cond_broadcast(c)

#define ggml_thread_create pthread_create
#define ggml_thread_join   pthread_join

#endif

// Threadpool def
সংরচনা ggml_threadpool {
    ggml_mutex_t mutex;       // mutex for cond.var
    ggml_cond_t  cond;        // cond.var for waiting for new work

    সংরচনা ggml_cgraph * cgraph;
    সংরচনা ggml_cplan  * cplan;

    // synchronization primitives
    atomic_int n_graph;       // updated when there is work to be done (i.e each graph) holds graph and active thread counts.
    atomic_int GGML_CACHE_ALIGN n_barrier;
    atomic_int GGML_CACHE_ALIGN n_barrier_passed;
    atomic_int GGML_CACHE_ALIGN current_chunk; // currently processing chunk during Mat_Mul, shared between all the threads.

    // these are atomic as an annotation for thread-sanitizer
    atomic_bool stop;         // Used for stopping the threadpool altogether
    atomic_bool pause;        // Used for pausing the threadpool or individual threads
    atomic_int  abort;        // Used for aborting processing of a graph

    সংরচনা ggml_compute_state * workers;   // per thread state
    পূর্ণাঙ্ক          n_threads;   // Number of threads in the pool
    int32_t      prio;        // Scheduling priority
    uint32_t     poll;        // Polling level (0 - no polling)

    গণনা ggml_status ec;
};

// Per-thread state
সংরচনা ggml_compute_state {
#ifndef GGML_USE_OPENMP
    ggml_thread_t thrd;
    পূর্ণাঙ্ক  last_graph;
    bool pending;
#endif
    bool cpumask[GGML_MAX_N_THREADS];
    সংরচনা ggml_threadpool * threadpool;
    পূর্ণাঙ্ক ith;
};

// Helpers for polling loops
#যদি defined(__aarch64__) && ( defined(__clang__) || defined(__GNUC__) )
স্থির অন্তঃস্থ শূন্য ggml_thread_cpu_relax(শূন্য) {
    __asm__ চঞ্চল("yield" ::: "memory");
}
#নোচেদ্যদি defined(__x86_64__)
স্থির অন্তঃস্থ শূন্য ggml_thread_cpu_relax(শূন্য) {
    _mm_pause();
}
#নোচেদ্যদি defined(__riscv)
স্থির অন্তঃস্থ শূন্য ggml_thread_cpu_relax(শূন্য) {
    #ifdef __riscv_zihintpause
        __asm__ __volatile__ ("pause");
    #অন্যথা
        /* Encoding of the pause instruction */
        __asm__ __volatile__ (".4byte 0x100000F");
    #endif
}
#অন্যথা
স্থির অন্তঃস্থ শূন্য ggml_thread_cpu_relax(শূন্য) {;}
#endif

//
// NUMA support
//

#define GGML_NUMA_MAX_NODES 8
#define GGML_NUMA_MAX_CPUS 512

সংরচনা ggml_numa_গ্ৰন্থি {
    uint32_t cpus[GGML_NUMA_MAX_CPUS]; // hardware threads on this node
    uint32_t n_cpus;
};

সংরচনা ggml_numa_nodes {
    গণনা ggml_numa_strategy numa_strategy;
    সংরচনা ggml_numa_গ্ৰন্থি nodes[GGML_NUMA_MAX_NODES];
    uint32_t n_nodes;
    uint32_t total_cpus; // hardware threads on system
    uint32_t current_গ্ৰন্থি; // node on which main process is execting
#যদি defined(__gnu_linux__)
    cpu_set_t cpuset; // cpuset from numactl
#অন্যথা
    uint32_t cpuset; // no NUMA support outside of Linux at this time. Use a portable datatype
#endif
};

//
// ggml state
//

সংরচনা ggml_state {
    সংরচনা ggml_numa_nodes numa;
};

স্থির সংরচনা ggml_state g_state = {0};

শূন্য ggml_barrier(সংরচনা ggml_threadpool * tp) {
    পূর্ণাঙ্ক n_threads = atomic_load_explicit(&tp->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK;
    যদি (n_threads == 1) {
        ঘূৰাই;
    }

#ifdef GGML_USE_OPENMP
    #pragma omp barrier
#অন্যথা
    পূর্ণাঙ্ক n_passed = atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed);

    // enter barrier (full seq-cst fence)
    পূর্ণাঙ্ক n_barrier = atomic_fetch_add_explicit(&tp->n_barrier, 1, memory_order_seq_cst);

    যদি (n_barrier == (n_threads - 1)) {
        // last thread
        atomic_store_explicit(&tp->n_barrier, 0, memory_order_relaxed);

        // exit barrier (full seq-cst fence)
        atomic_fetch_add_explicit(&tp->n_barrier_passed, 1, memory_order_seq_cst);
        ঘূৰাই;
    }

    // wait for other threads
    যেতিয়ালৈ (atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed) == n_passed) {
        ggml_thread_cpu_relax();
    }

    // exit barrier (full seq-cst fence)
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&tp->n_barrier_passed, 0, memory_order_seq_cst);
    #অন্যথা
    atomic_thread_fence(memory_order_seq_cst);
    #endif
#endif
}

শূন্য ggml_threadpool_chunk_set(সংরচনা ggml_threadpool * tp, পূর্ণাঙ্ক মান) {
    atomic_store_explicit(&tp->current_chunk, মান, memory_order_relaxed);
}

পূর্ণাঙ্ক ggml_threadpool_chunk_add(সংরচনা ggml_threadpool * tp, পূর্ণাঙ্ক মান) {
    ঘূৰাই atomic_fetch_add_explicit(&tp->current_chunk, মান, memory_order_relaxed);
}

#যদি defined(__gnu_linux__)
স্থির cpu_set_t ggml_get_numa_affinity(শূন্য) {
    cpu_set_t cpuset;
    pthread_t thread;
    thread = pthread_self();
    CPU_ZERO(&cpuset);
    pthread_getaffinity_np(thread, পরিমাণ(cpu_set_t), &cpuset);
    ঘূৰাই cpuset;
}
#অন্যথা
স্থির uint32_t ggml_get_numa_affinity(শূন্য) {
    ঘূৰাই 0; // no NUMA support
}
#endif

শূন্য ggml_numa_init(গণনা ggml_numa_strategy numa_flag) {
    যদি (g_state.numa.n_nodes > 0) {
        fprintf(stderr, "ggml_numa_init: NUMA already initialized\n");

        ঘূৰাই;
    }

#যদি defined(__gnu_linux__)
    সংরচনা stat st;
    ৱর্ণ path[256];
    পূর্ণাঙ্ক rv;

    // set numa scheme
    g_state.numa.numa_strategy = numa_flag;

    GGML_PRINT_DEBUG("numa strategy %u\n",g_state.numa.numa_strategy);

    g_state.numa.cpuset = ggml_get_numa_affinity();

    // enumerate nodes
    যেতিয়ালৈ (g_state.numa.n_nodes < GGML_NUMA_MAX_NODES) {
        rv = snprintf(path, পরিমাণ(path), "/sys/devices/system/node/node%u", g_state.numa.n_nodes);
        GGML_ASSERT(rv > 0 && (unsigned)rv < পরিমাণ(path));
        যদি (stat(path, &st) != 0) { ভঙ্গ; }
        ++g_state.numa.n_nodes;
    }

    // enumerate CPUs
    যেতিয়ালৈ (g_state.numa.total_cpus < GGML_NUMA_MAX_CPUS) {
        rv = snprintf(path, পরিমাণ(path), "/sys/devices/system/cpu/cpu%u", g_state.numa.total_cpus);
        GGML_ASSERT(rv > 0 && (unsigned)rv < পরিমাণ(path));
        যদি (stat(path, &st) != 0) { ভঙ্গ; }
        ++g_state.numa.total_cpus;
    }

    GGML_PRINT_DEBUG("found %u numa nodes, %u CPUs\n", g_state.numa.n_nodes, g_state.numa.total_cpus);

    // figure out which node we're on
    uint current_cpu;
    পূর্ণাঙ্ক getcpu_ret = 0;
#যদি __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ > 33) || defined(__COSMOPOLITAN__)
    getcpu_ret = getcpu(&current_cpu, &g_state.numa.current_গ্ৰন্থি);
#অন্যথা
    // old glibc doesn't have a wrapper for this call. Fall back on direct syscall
#   যদি !defined(SYS_getcpu) && defined(SYS_get_cpu)
#       define SYS_getcpu SYS_get_cpu // some older glibc versions use this name
#   endif
    getcpu_ret = syscall(SYS_getcpu, &current_cpu, &g_state.numa.current_গ্ৰন্থি);
#endif

    যদি (g_state.numa.n_nodes < 1 || g_state.numa.total_cpus < 1 || getcpu_ret != 0) {
        g_state.numa.n_nodes = 0;
        ঘূৰাই;
    }

    GGML_PRINT_DEBUG("found our process on numa node %u, CPU %u\n", g_state.numa.current_গ্ৰন্থি, current_cpu);

    বাবে (uint32_t n = 0; n < g_state.numa.n_nodes; ++n) {
        সংরচনা ggml_numa_গ্ৰন্থি * গ্ৰন্থি = &g_state.numa.nodes[n];
        GGML_PRINT_DEBUG("CPUs on node %u:", n);
        গ্ৰন্থি->n_cpus = 0;
        বাবে (uint32_t c = 0; c < g_state.numa.total_cpus; ++c) {
            rv = snprintf(path, পরিমাণ(path), "/sys/devices/system/node/node%u/cpu%u", n, c);
            GGML_ASSERT(rv > 0 && (unsigned)rv < পরিমাণ(path));
            যদি (stat(path, &st) == 0) {
                গ্ৰন্থি->cpus[গ্ৰন্থি->n_cpus++] = c;
                GGML_PRINT_DEBUG(" %u", c);
            }
        }
        GGML_PRINT_DEBUG("\n");
    }

    যদি (ggml_is_numa()) {
        FILE *fptr = fopen("/proc/sys/kernel/numa_balancing", "r");
        যদি (fptr != রিক্ত) {
            ৱর্ণ buf[42];
            যদি (fgets(buf, পরিমাণ(buf), fptr) && strncmp(buf, "0\n", পরিমাণ(buf)) != 0) {
                GGML_LOG_WARN("/proc/sys/kernel/numa_balancing is enabled, this has been observed to impair performance\n");
            }
            fclose(fptr);
        }
    }
#অন্যথা
    UNUSED(numa_flag);
    // TODO
#endif
}

bool ggml_is_numa(শূন্য) {
    ঘূৰাই g_state.numa.n_nodes > 1;
}

#যদি defined(__ARM_ARCH)
#যদি defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
#অন্তর্ভাৱ <arm_sve.h>
স্থির শূন্য ggml_init_arm_arch_features(শূন্য) {
    ggml_arm_arch_features.sve_cnt = svcntb();
}
#অন্যথা
স্থির শূন্য ggml_init_arm_arch_features(শূন্য) {}
#endif
#endif // __ARM_ARCH

#যদি defined(__riscv) && defined(__riscv_v_intrinsic)
#অন্তর্ভাৱ <riscv_vector.h>
স্থির শূন্য ggml_init_riscv_arch_features(শূন্য) {
    ggml_riscv_arch_features.rvv_vlen = __riscv_vlenb();
}
#অন্যথা
স্থির শূন্য ggml_init_riscv_arch_features(শূন্য) {}
#endif

সংরচনা ggml_tensor * ggml_new_i32(সংরচনা ggml_context * ctx, int32_t মান) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    সংরচনা ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

    ggml_set_i32(result, মান);

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_new_f32(সংরচনা ggml_context * ctx, ভিন্ন মান) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    সংরচনা ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    ggml_set_f32(result, মান);

    ঘূৰাই result;
}

সংরচনা ggml_tensor * ggml_set_i32 (সংরচনা ggml_tensor * tensor, int32_t মান) {
    ধ্রুৱ পূর্ণাঙ্ক n     = ggml_nrows(tensor);
    ধ্রুৱ পূর্ণাঙ্ক nc    = tensor->ne[0];
    ধ্রুৱ size_t n1 = tensor->nb[1];

    ৱর্ণ * ধ্রুৱ data = tensor->data;

    ৱিকল্প (tensor->type) {
        পক্ষ GGML_TYPE_I8:
            {
                প্রতিজ্ঞা(tensor->nb[0] == পরিমাণ(int8_t));
                বাবে (পূর্ণাঙ্ক i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), মান);
                }
            } ভঙ্গ;
        পক্ষ GGML_TYPE_I16:
            {
                প্রতিজ্ঞা(tensor->nb[0] == পরিমাণ(int16_t));
                বাবে (পূর্ণাঙ্ক i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), মান);
                }
            } ভঙ্গ;
        পক্ষ GGML_TYPE_I32:
            {
                প্রতিজ্ঞা(tensor->nb[0] == পরিমাণ(int32_t));
                বাবে (পূর্ণাঙ্ক i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), মান);
                }
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F16:
            {
                প্রতিজ্ঞা(tensor->nb[0] == পরিমাণ(ggml_fp16_t));
                বাবে (পূর্ণাঙ্ক i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), GGML_CPU_FP32_TO_FP16(মান));
                }
            } ভঙ্গ;
        পক্ষ GGML_TYPE_BF16:
            {
                প্রতিজ্ঞা(tensor->nb[0] == পরিমাণ(ggml_fp16_t));
                বাবে (পূর্ণাঙ্ক i = 0; i < n; i++) {
                    ggml_vec_set_bf16(nc, (ggml_bf16_t *)(data + i*n1), GGML_FP32_TO_BF16(মান));
                }
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F32:
            {
                প্রতিজ্ঞা(tensor->nb[0] == পরিমাণ(ভিন্ন));
                বাবে (পূর্ণাঙ্ক i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (ভিন্ন *)(data + i*n1), মান);
                }
            } ভঙ্গ;
        স্ৱভাৱ:
            {
                GGML_ABORT("fatal error");
            }
    }

    ঘূৰাই tensor;
}

সংরচনা ggml_tensor * ggml_set_f32(সংরচনা ggml_tensor * tensor, ভিন্ন মান) {
    ধ্রুৱ পূর্ণাঙ্ক n     = ggml_nrows(tensor);
    ধ্রুৱ পূর্ণাঙ্ক nc    = tensor->ne[0];
    ধ্রুৱ size_t n1 = tensor->nb[1];

    ৱর্ণ * ধ্রুৱ data = tensor->data;

    ৱিকল্প (tensor->type) {
        পক্ষ GGML_TYPE_I8:
            {
                প্রতিজ্ঞা(tensor->nb[0] == পরিমাণ(int8_t));
                বাবে (পূর্ণাঙ্ক i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), মান);
                }
            } ভঙ্গ;
        পক্ষ GGML_TYPE_I16:
            {
                প্রতিজ্ঞা(tensor->nb[0] == পরিমাণ(int16_t));
                বাবে (পূর্ণাঙ্ক i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), মান);
                }
            } ভঙ্গ;
        পক্ষ GGML_TYPE_I32:
            {
                প্রতিজ্ঞা(tensor->nb[0] == পরিমাণ(int32_t));
                বাবে (পূর্ণাঙ্ক i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), মান);
                }
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F16:
            {
                প্রতিজ্ঞা(tensor->nb[0] == পরিমাণ(ggml_fp16_t));
                বাবে (পূর্ণাঙ্ক i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), GGML_CPU_FP32_TO_FP16(মান));
                }
            } ভঙ্গ;
        পক্ষ GGML_TYPE_BF16:
            {
                প্রতিজ্ঞা(tensor->nb[0] == পরিমাণ(ggml_bf16_t));
                বাবে (পূর্ণাঙ্ক i = 0; i < n; i++) {
                    ggml_vec_set_bf16(nc, (ggml_bf16_t *)(data + i*n1), GGML_FP32_TO_BF16(মান));
                }
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F32:
            {
                প্রতিজ্ঞা(tensor->nb[0] == পরিমাণ(ভিন্ন));
                বাবে (পূর্ণাঙ্ক i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (ভিন্ন *)(data + i*n1), মান);
                }
            } ভঙ্গ;
        স্ৱভাৱ:
            {
                GGML_ABORT("fatal error");
            }
    }

    ঘূৰাই tensor;
}

int32_t ggml_get_i32_1d(ধ্রুৱ সংরচনা ggml_tensor * tensor, পূর্ণাঙ্ক i) {
    যদি (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ঘূৰাই ggml_get_i32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    ৱিকল্প (tensor->type) {
        পক্ষ GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == পরিমাণ(int8_t));
                ঘূৰাই ((int8_t *)(tensor->data))[i];
            }
        পক্ষ GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == পরিমাণ(int16_t));
                ঘূৰাই ((int16_t *)(tensor->data))[i];
            }
        পক্ষ GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == পরিমাণ(int32_t));
                ঘূৰাই ((int32_t *)(tensor->data))[i];
            }
        পক্ষ GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == পরিমাণ(ggml_fp16_t));
                ঘূৰাই GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *)(tensor->data))[i]);
            }
        পক্ষ GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == পরিমাণ(ggml_bf16_t));
                ঘূৰাই GGML_BF16_TO_FP32(((ggml_bf16_t *)(tensor->data))[i]);
            }
        পক্ষ GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == পরিমাণ(ভিন্ন));
                ঘূৰাই ((ভিন্ন *)(tensor->data))[i];
            }
        স্ৱভাৱ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

শূন্য ggml_set_i32_1d(ধ্রুৱ সংরচনা ggml_tensor * tensor, পূর্ণাঙ্ক i, int32_t মান) {
    যদি (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_set_i32_nd(tensor, id[0], id[1], id[2], id[3], মান);
        ঘূৰাই;
    }
    ৱিকল্প (tensor->type) {
        পক্ষ GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == পরিমাণ(int8_t));
                ((int8_t *)(tensor->data))[i] = মান;
            } ভঙ্গ;
        পক্ষ GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == পরিমাণ(int16_t));
                ((int16_t *)(tensor->data))[i] = মান;
            } ভঙ্গ;
        পক্ষ GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == পরিমাণ(int32_t));
                ((int32_t *)(tensor->data))[i] = মান;
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == পরিমাণ(ggml_fp16_t));
                ((ggml_fp16_t *)(tensor->data))[i] = GGML_CPU_FP32_TO_FP16(মান);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == পরিমাণ(ggml_bf16_t));
                ((ggml_bf16_t *)(tensor->data))[i] = GGML_FP32_TO_BF16(মান);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == পরিমাণ(ভিন্ন));
                ((ভিন্ন *)(tensor->data))[i] = মান;
            } ভঙ্গ;
        স্ৱভাৱ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

int32_t ggml_get_i32_nd(ধ্রুৱ সংরচনা ggml_tensor * tensor, পূর্ণাঙ্ক i0, পূর্ণাঙ্ক i1, পূর্ণাঙ্ক i2, পূর্ণাঙ্ক i3) {
    শূন্য * data   = (ৱর্ণ *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    ৱিকল্প (tensor->type) {
        পক্ষ GGML_TYPE_I8:
            ঘূৰাই ((int8_t *) data)[0];
        পক্ষ GGML_TYPE_I16:
            ঘূৰাই ((int16_t *) data)[0];
        পক্ষ GGML_TYPE_I32:
            ঘূৰাই ((int32_t *) data)[0];
        পক্ষ GGML_TYPE_F16:
            ঘূৰাই GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *) data)[0]);
        পক্ষ GGML_TYPE_BF16:
            ঘূৰাই GGML_BF16_TO_FP32(((ggml_bf16_t *) data)[0]);
        পক্ষ GGML_TYPE_F32:
            ঘূৰাই ((ভিন্ন *) data)[0];
        স্ৱভাৱ:
            GGML_ABORT("fatal error");
    }
}

শূন্য ggml_set_i32_nd(ধ্রুৱ সংরচনা ggml_tensor * tensor, পূর্ণাঙ্ক i0, পূর্ণাঙ্ক i1, পূর্ণাঙ্ক i2, পূর্ণাঙ্ক i3, int32_t মান) {
    শূন্য * data   = (ৱর্ণ *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    ৱিকল্প (tensor->type) {
        পক্ষ GGML_TYPE_I8:
            {
                ((int8_t *)(data))[0] = মান;
            } ভঙ্গ;
        পক্ষ GGML_TYPE_I16:
            {
                ((int16_t *)(data))[0] = মান;
            } ভঙ্গ;
        পক্ষ GGML_TYPE_I32:
            {
                ((int32_t *)(data))[0] = মান;
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(data))[0] = GGML_CPU_FP32_TO_FP16(মান);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(data))[0] = GGML_FP32_TO_BF16(মান);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F32:
            {
                ((ভিন্ন *)(data))[0] = মান;
            } ভঙ্গ;
        স্ৱভাৱ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ভিন্ন ggml_get_f32_1d(ধ্রুৱ সংরচনা ggml_tensor * tensor, পূর্ণাঙ্ক i) {
    যদি (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ঘূৰাই ggml_get_f32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    ৱিকল্প (tensor->type) {
        পক্ষ GGML_TYPE_I8:
            {
                ঘূৰাই ((int8_t *)(tensor->data))[i];
            }
        পক্ষ GGML_TYPE_I16:
            {
                ঘূৰাই ((int16_t *)(tensor->data))[i];
            }
        পক্ষ GGML_TYPE_I32:
            {
                ঘূৰাই ((int32_t *)(tensor->data))[i];
            }
        পক্ষ GGML_TYPE_F16:
            {
                ঘূৰাই GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *)(tensor->data))[i]);
            }
        পক্ষ GGML_TYPE_BF16:
            {
                ঘূৰাই GGML_BF16_TO_FP32(((ggml_bf16_t *)(tensor->data))[i]);
            }
        পক্ষ GGML_TYPE_F32:
            {
                ঘূৰাই ((ভিন্ন *)(tensor->data))[i];
            }
        স্ৱভাৱ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

শূন্য ggml_set_f32_1d(ধ্রুৱ সংরচনা ggml_tensor * tensor, পূর্ণাঙ্ক i, ভিন্ন মান) {
    যদি (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_set_f32_nd(tensor, id[0], id[1], id[2], id[3], মান);
        ঘূৰাই;
    }
    ৱিকল্প (tensor->type) {
        পক্ষ GGML_TYPE_I8:
            {
                ((int8_t *)(tensor->data))[i] = মান;
            } ভঙ্গ;
        পক্ষ GGML_TYPE_I16:
            {
                ((int16_t *)(tensor->data))[i] = মান;
            } ভঙ্গ;
        পক্ষ GGML_TYPE_I32:
            {
                ((int32_t *)(tensor->data))[i] = মান;
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(tensor->data))[i] = GGML_CPU_FP32_TO_FP16(মান);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(tensor->data))[i] = GGML_FP32_TO_BF16(মান);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F32:
            {
                ((ভিন্ন *)(tensor->data))[i] = মান;
            } ভঙ্গ;
        স্ৱভাৱ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ভিন্ন ggml_get_f32_nd(ধ্রুৱ সংরচনা ggml_tensor * tensor, পূর্ণাঙ্ক i0, পূর্ণাঙ্ক i1, পূর্ণাঙ্ক i2, পূর্ণাঙ্ক i3) {
    শূন্য * data   = (ৱর্ণ *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    ৱিকল্প (tensor->type) {
        পক্ষ GGML_TYPE_I8:
            ঘূৰাই ((int8_t *) data)[0];
        পক্ষ GGML_TYPE_I16:
            ঘূৰাই ((int16_t *) data)[0];
        পক্ষ GGML_TYPE_I32:
            ঘূৰাই ((int32_t *) data)[0];
        পক্ষ GGML_TYPE_F16:
            ঘূৰাই GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *) data)[0]);
        পক্ষ GGML_TYPE_BF16:
            ঘূৰাই GGML_BF16_TO_FP32(((ggml_bf16_t *) data)[0]);
        পক্ষ GGML_TYPE_F32:
            ঘূৰাই ((ভিন্ন *) data)[0];
        স্ৱভাৱ:
            GGML_ABORT("fatal error");
    }
}

শূন্য ggml_set_f32_nd(ধ্রুৱ সংরচনা ggml_tensor * tensor, পূর্ণাঙ্ক i0, পূর্ণাঙ্ক i1, পূর্ণাঙ্ক i2, পূর্ণাঙ্ক i3, ভিন্ন মান) {
    শূন্য * data   = (ৱর্ণ *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    ৱিকল্প (tensor->type) {
        পক্ষ GGML_TYPE_I8:
            {
                ((int8_t *)(data))[0] = মান;
            } ভঙ্গ;
        পক্ষ GGML_TYPE_I16:
            {
                ((int16_t *)(data))[0] = মান;
            } ভঙ্গ;
        পক্ষ GGML_TYPE_I32:
            {
                ((int32_t *)(data))[0] = মান;
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(data))[0] = GGML_CPU_FP32_TO_FP16(মান);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(data))[0] = GGML_FP32_TO_BF16(মান);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F32:
            {
                ((ভিন্ন *)(data))[0] = মান;
            } ভঙ্গ;
        স্ৱভাৱ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

////////////////////////////////////////////////////////////////////////////////

// ggml_compute_forward_mul_mat

স্থির শূন্য ggml_compute_forward_mul_mat_one_chunk(
    ধ্রুৱ সংরচনা ggml_compute_params * params,
    সংরচনা ggml_tensor * dst,
    ধ্রুৱ গণনা ggml_type type,
    ধ্রুৱ int64_t অঙ্ক_rows_per_vec_dot,
    ধ্রুৱ int64_t ir0_start,
    ধ্রুৱ int64_t ir0_end,
    ধ্রুৱ int64_t ir1_start,
    ধ্রুৱ int64_t ir1_end) {

    ধ্রুৱ সংরচনা ggml_tensor * src0 = dst->src[0];
    ধ্রুৱ সংরচনা ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_দ্বিমিক_OP_LOCALS

    ধ্রুৱ bool src1_cont = ggml_is_contiguous(src1);

    ggml_vec_dot_t ধ্রুৱ vec_dot      = type_traits_cpu[type].vec_dot;
    গণনা ggml_type ধ্রুৱ vec_dot_type = type_traits_cpu[type].vec_dot_type;

    // broadcast factors
    ধ্রুৱ int64_t r2 = ne12 / ne02;
    ধ্রুৱ int64_t r3 = ne13 / ne03;

    //printf("ir0_start = %6lld, ir0_end = %6lld, ir1_start = %6lld, ir1_end = %6lld\n", ir0_start, ir0_end, ir1_start, ir1_end);

    // threads with no work simply yield (not sure if it helps)
    যদি (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        ঘূৰাই;
    }

    ধ্রুৱ শূন্য * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
    ধ্রুৱ size_t row_size = ggml_row_size(vec_dot_type, ne10);

    প্রতিজ্ঞা(ne12 % ne02 == 0);
    প্রতিজ্ঞা(ne13 % ne03 == 0);

    // block-tiling attempt
    ধ্রুৱ int64_t blck_0 = 16;
    ধ্রুৱ int64_t blck_1 = 16;

    ধ্রুৱ size_t src1_col_stride = src1_cont || src1->type != vec_dot_type ? row_size : nb11;

    // attempt to reduce false-sharing (does not seem to make a difference)
    // 16 * 2, accounting for mmla kernels
    ভিন্ন tmp[32];

    বাবে (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        বাবে (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            বাবে (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += অঙ্ক_rows_per_vec_dot) {
                ধ্রুৱ int64_t i13 = (ir1 / (ne12 * ne1));
                ধ্রুৱ int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                ধ্রুৱ int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                // broadcast src0 into src1
                ধ্রুৱ int64_t i03 = i13 / r3;
                ধ্রুৱ int64_t i02 = i12 / r2;

                ধ্রুৱ int64_t i1 = i11;
                ধ্রুৱ int64_t i2 = i12;
                ধ্রুৱ int64_t i3 = i13;

                ধ্রুৱ ৱর্ণ * src0_row = (ধ্রুৱ ৱর্ণ*)src0->data + (0 + i02 * nb02 + i03 * nb03);

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                ধ্রুৱ ৱর্ণ * src1_col = (ধ্রুৱ ৱর্ণ*)wdata +
                    (src1_cont || src1->type != vec_dot_type
                        ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                        : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                ভিন্ন * dst_col = (ভিন্ন*)((ৱর্ণ*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

                //for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                //    vec_dot(ne00, &dst_col[ir0], src0_row + ir0*nb01, src1_col);
                //}

                বাবে (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += অঙ্ক_rows_per_vec_dot) {
                    vec_dot(ne00, &tmp[ir0 - iir0], (অঙ্ক_rows_per_vec_dot > 1 ? 16 : 0), src0_row + ir0 * nb01, (অঙ্ক_rows_per_vec_dot > 1 ? nb01 : 0), src1_col, (অঙ্ক_rows_per_vec_dot > 1 ? src1_col_stride : 0), অঙ্ক_rows_per_vec_dot);
                }

                বাবে (পূর্ণাঙ্ক cn = 0; cn < অঙ্ক_rows_per_vec_dot; ++cn) {
                    memcpy(&dst_col[iir0 + cn * nb1 / nb0], tmp + (cn * 16), (MIN(iir0 + blck_0, ir0_end) - iir0) * পরিমাণ(ভিন্ন));
                }
            }
        }
    }
}

শূন্য ggml_compute_forward_mul_mat(
        ধ্রুৱ সংরচনা ggml_compute_params * params,
              সংরচনা ggml_tensor * dst) {

    ধ্রুৱ সংরচনা ggml_tensor * src0 = dst->src[0];
    ধ্রুৱ সংরচনা ggml_tensor * src1 = dst->src[1];

    ধ্রুৱ int32_t hint = ggml_get_op_params_i32(dst, 1);
    যদি (hint == GGML_HINT_SRC0_IS_HADAMARD && !params->use_ref) {
        ggml_compute_forward_fwht(params, dst);
        ঘূৰাই;
    }

    // If tiled is supported, it will execute the full op here and we return
    যদি (ggml_compute_forward_mul_mat_tiled(params, dst)) {
        ঘূৰাই;
    }

    GGML_TENSOR_দ্বিমিক_OP_LOCALS

    ধ্রুৱ পূর্ণাঙ্ক ith = params->ith;
    ধ্রুৱ পূর্ণাঙ্ক nth = params->nth;

    গণনা ggml_type           ধ্রুৱ vec_dot_type         = type_traits_cpu[src0->type].vec_dot_type;
    ggml_from_float_t        ধ্রুৱ from_float           = type_traits_cpu[vec_dot_type].from_float;
    int64_t                  ধ্রুৱ vec_dot_অঙ্ক_rows     = type_traits_cpu[src0->type].nrows;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == পরিমাণ(ভিন্ন));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    // TODO: extract to "extra_op"
#যদি GGML_USE_LLAMAFILE
    // broadcast factors
    ধ্রুৱ int64_t r2 = ne12 / ne02;
    ধ্রুৱ int64_t r3 = ne13 / ne03;

    ধ্রুৱ bool src1_cont = ggml_is_contiguous(src1);

    যদি (!params->use_ref && src1_cont) {
        বাবে (int64_t i13 = 0; i13 < ne13; i13++)
            বাবে (int64_t i12 = 0; i12 < ne12; i12++)
                যদি (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_size(src0->type),
                                     (ধ্রুৱ ৱর্ণ *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_size(src0->type),
                                     (ধ্রুৱ ৱর্ণ *)src1->data + i12*nb12 + i13*nb13,
                                     nb11/ggml_type_size(src1->type),
                                     (ৱর্ণ *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_size(dst->type),
                                     src0->type,
                                     src1->type,
                                     dst->type))
                    গচ্ছ UseGgmlGemm1;
        ঘূৰাই;
    }
UseGgmlGemm1:;
#endif

    যদি (src1->type != vec_dot_type) {
        ৱর্ণ * wdata = params->wdata;

        ধ্রুৱ size_t nbw0 = ggml_type_size(vec_dot_type);
        ধ্রুৱ size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        ধ্রুৱ size_t nbw2 = nbw1*ne11;
        ধ্রুৱ size_t nbw3 = nbw2*ne12;

        প্রতিজ্ঞা(params->wsize >= ne13*nbw3);
        // src1 is either packed from F32 into vec_dot_type, or widened from F16 or BF16 into the F32 work buffer
        ধ্রুৱ bool widen = src1->type != GGML_TYPE_F32;

        GGML_ASSERT(!widen || vec_dot_type == GGML_TYPE_F32);
        GGML_ASSERT(!widen || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_BF16);

    #যদি 0
        বাবে (int64_t i13 = 0; i13 < ne13; ++i13) {
            বাবে (int64_t i12 = 0; i12 < ne12; ++i12) {
                বাবে (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                    from_float((ভিন্ন *)((ৱর্ণ *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (শূন্য *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                                ne10);
                }
            }
        }
    #অন্যথা
        ধ্রুৱ int64_t bs = ggml_blck_size(vec_dot_type);

        বাবে (int64_t i13 = 0; i13 < ne13; ++i13) {
            বাবে (int64_t i12 = 0; i12 < ne12; ++i12) {
                বাবে (int64_t i11 = 0; i11 < ne11; ++i11) {
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    ধ্রুৱ শূন্য * src1_row  = (ধ্রুৱ ৱর্ণ *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10;
                    শূন্য       * wdata_row =                    wdata  + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0;

                    ধ্রুৱ int64_t ne10_block_size = (ne10_block_end - ne10_block_start) * bs;

                    যদি (src1->type == GGML_TYPE_F16) {
                        ggml_cpu_fp16_to_fp32((ধ্রুৱ ggml_fp16_t *) src1_row, (ভিন্ন *) wdata_row, ne10_block_size);
                    } অন্যথা যদি (src1->type == GGML_TYPE_BF16) {
                        ggml_cpu_bf16_to_fp32((ধ্রুৱ ggml_bf16_t *) src1_row, (ভিন্ন *) wdata_row, ne10_block_size);
                    } অন্যথা {
                        from_float((ধ্রুৱ ভিন্ন *) src1_row, wdata_row, ne10_block_size);
                    }
                }
            }
        }
    #endif
    }

    যদি (ith == 0) {
        // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
        atomic_store_explicit(&params->threadpool->current_chunk, nth, memory_order_relaxed);
    }

    ggml_barrier(params->threadpool);

#যদি GGML_USE_LLAMAFILE
    যদি (!params->use_ref && src1->type != vec_dot_type) {
        ধ্রুৱ শূন্য* wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        ধ্রুৱ size_t row_size = ggml_row_size(vec_dot_type, ne10);

        বাবে (int64_t i13 = 0; i13 < ne13; i13++)
            বাবে (int64_t i12 = 0; i12 < ne12; i12++)
                যদি (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_size(src0->type),
                                     (ধ্রুৱ ৱর্ণ *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_size(src0->type),
                                     (ধ্রুৱ ৱর্ণ *)wdata + (i12*ne11 + i13*ne12*ne11)*row_size,
                                     row_size/ggml_type_size(vec_dot_type),
                                     (ৱর্ণ *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_size(dst->type),
                                     src0->type,
                                     vec_dot_type,
                                     dst->type))
                    গচ্ছ UseGgmlGemm2;
        ঘূৰাই;
    }
UseGgmlGemm2:;
#endif

    // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
    ধ্রুৱ int64_t nr0 = ne0;

    // This is the size of the rest of the dimensions of the result
    ধ্রুৱ int64_t nr1 = ne1 * ne2 * ne3;

    // Now select a reasonable chunk size.
    পূর্ণাঙ্ক chunk_size = 16;

    // We need to step up the size if it's small
    যদি (nr0 == 1 || nr1 == 1) {
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
    যদি (nchunk0 * nchunk1 < nth * 4 || ggml_is_numa()) {
        // distribute the thread work across the inner or outer loop based on which one is larger
        nchunk0 = nr0 > nr1 ? nth : 1; // parallelize by src0 rows
        nchunk1 = nr0 > nr1 ? 1 : nth; // parallelize by src1 rows
    }

    // The number of elements in each chunk
    ধ্রুৱ int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    ধ্রুৱ int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    // The first chunk comes from our thread_id, the rest will get auto-assigned.
    পূর্ণাঙ্ক current_chunk = ith;

    যেতিয়ালৈ (current_chunk < nchunk0 * nchunk1) {
        ধ্রুৱ int64_t ith0 = current_chunk % nchunk0;
        ধ্রুৱ int64_t ith1 = current_chunk / nchunk0;

        ধ্রুৱ int64_t ir0_start = dr0 * ith0;
        ধ্রুৱ int64_t ir0_end = MIN(ir0_start + dr0, nr0);

        ধ্রুৱ int64_t ir1_start = dr1 * ith1;
        ধ্রুৱ int64_t ir1_end = MIN(ir1_start + dr1, nr1);

        // dot kernels can handle 1 row and col at a time, but mmla kernels can process 2 rows and cols
        int64_t অঙ্ক_rows_per_vec_dot = vec_dot_অঙ্ক_rows;

        // these checks are needed to avoid crossing dim1 boundaries
        // can be optimized, but the logic would become more complicated, so keeping it like this for simplicity
        যদি ((nr0 % 2 != 0) || (ne11 % 2 != 0) || ((ir0_end - ir0_start) % 2 != 0) || ((ir1_end - ir1_start) % 2 != 0)) {
            অঙ্ক_rows_per_vec_dot = 1;
        }
        ggml_compute_forward_mul_mat_one_chunk(params, dst, src0->type, অঙ্ক_rows_per_vec_dot, ir0_start, ir0_end, ir1_start, ir1_end);

        যদি (nth >= nchunk0 * nchunk1) {
            ভঙ্গ;
        }

        current_chunk = atomic_fetch_add_explicit(&params->threadpool->current_chunk, 1, memory_order_relaxed);
    }
}

// ggml_compute_forward_mul_mat_id

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id)*ids->ne[0]*ids->ne[1] + (i1)]

সংরচনা mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

স্থির শূন্য ggml_compute_forward_mul_mat_id_one_chunk(
    সংরচনা ggml_tensor * dst,
    ধ্রুৱ সংরচনা ggml_tensor * src0,
    ধ্রুৱ সংরচনা ggml_tensor * src1,
    ধ্রুৱ সংরচনা ggml_tensor * ids,
    ধ্রুৱ int64_t cur_a,
    ধ্রুৱ int64_t ir0_start,
    ধ্রুৱ int64_t ir0_end,
    ধ্রুৱ int64_t ir1_start,
    ধ্রুৱ int64_t ir1_end,
    ধ্রুৱ ৱর্ণ * src0_cur,
    ধ্রুৱ সংরচনা mmid_row_mapping * matrix_rows,
    ধ্রুৱ size_t row_size,
    ধ্রুৱ bool src1_cont,
    ধ্রুৱ শূন্য * wdata) {

    GGML_TENSOR_দ্বিমিক_OP_LOCALS

    ধ্রুৱ গণনা ggml_type type = src0->type;

    ggml_vec_dot_t    ধ্রুৱ vec_dot      = type_traits_cpu[type].vec_dot;
    গণনা ggml_type    ধ্রুৱ vec_dot_type = type_traits_cpu[type].vec_dot_type;

    ধ্রুৱ int64_t blck_0 = 16;
    ধ্রুৱ int64_t blck_1 = 16;

    ভিন্ন tmp[16];

    বাবে (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        বাবে (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            বাবে (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ++ir1) {
                ধ্রুৱ int64_t _i12 = ir1; // logical row index for this expert

                সংরচনা mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, _i12);
                ধ্রুৱ পূর্ণাঙ্ক id       = row_mapping.i1; // selected expert index

                ধ্রুৱ int64_t  i11 = id % ne11;
                ধ্রুৱ int64_t  i12 = row_mapping.i2; // row index in src1

                ধ্রুৱ int64_t  i1 = id;  // selected expert index
                ধ্রুৱ int64_t  i2 = i12; // row

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                ধ্রুৱ ৱর্ণ * src1_col = (ধ্রুৱ ৱর্ণ *) wdata +
                    (src1_cont || src1->type != vec_dot_type
                    ? (i11      + i12*ne11)*row_size
                    : (i11*nb11 + i12*nb12));

                ভিন্ন * dst_col = (ভিন্ন *) ((ৱর্ণ *) dst->data + (i1*nb1 + i2*nb2));

                বাবে (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                    vec_dot(ne00, &tmp[ir0 - iir0], 0, src0_cur + ir0*nb01, 0, src1_col, 0, 1);
                }

                memcpy(&dst_col[iir0], tmp, (MIN(iir0 + blck_0, ir0_end) - iir0)*পরিমাণ(ভিন্ন));
            }
        }
    }
}

স্থির শূন্য * incr_ptr_aligned(শূন্য ** p, size_t size, size_t align) {

    শূন্য * ptr = *p;
    ptr = (শূন্য *) GGML_PAD((uintptr_t) ptr, align);
    *p = (শূন্য *) ((ৱর্ণ *) ptr + size);
    ঘূৰাই ptr;
}

স্থির শূন্য ggml_compute_forward_mul_mat_id(
        ধ্রুৱ সংরচনা ggml_compute_params * params,
              সংরচনা ggml_tensor * dst) {

    ধ্রুৱ সংরচনা ggml_tensor * src0 = dst->src[0];
    ধ্রুৱ সংরচনা ggml_tensor * src1 = dst->src[1];
    ধ্রুৱ সংরচনা ggml_tensor * ids = dst->src[2];

    GGML_TENSOR_দ্বিমিক_OP_LOCALS

    ধ্রুৱ পূর্ণাঙ্ক ith = params->ith;
    ধ্রুৱ পূর্ণাঙ্ক nth = params->nth;

    ধ্রুৱ গণনা ggml_type type = src0->type;

    ধ্রুৱ bool src1_cont = ggml_is_contiguous(src1);

    গণনা ggml_type    ধ্রুৱ vec_dot_type    = type_traits_cpu[type].vec_dot_type;
    ggml_from_float_t ধ্রুৱ from_float      = type_traits_cpu[vec_dot_type].from_float;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == পরিমাণ(ভিন্ন));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // row groups
    ধ্রুৱ পূর্ণাঙ্ক n_ids = ids->ne[0]; // n_expert_used
    ধ্রুৱ পূর্ণাঙ্ক n_as  = ne02;       // n_expert

    শূন্য * wdata_cur = params->wdata;

    যদি (src1->type != vec_dot_type) {
        incr_ptr_aligned(&wdata_cur, ggml_row_size(vec_dot_type, ggml_nelements(src1)), পরিমাণ(int64_t));
    }

    int64_t * matrix_row_counts = // [n_as]
        incr_ptr_aligned(&wdata_cur, n_as*পরিমাণ(int64_t), পরিমাণ(int64_t));

    সংরচনা mmid_row_mapping * matrix_rows = // [n_as][ids->ne[0]*ids->ne[1]]
        incr_ptr_aligned(&wdata_cur, n_as*ids->ne[0]*ids->ne[1]*পরিমাণ(সংরচনা mmid_row_mapping), পরিমাণ(int64_t));

    ৱর্ণ (*atomic_current_chunk)[CACHE_সাৰি_SIZE] = // [n_as]
        incr_ptr_aligned(&wdata_cur, CACHE_সাৰি_SIZE * n_as, CACHE_সাৰি_SIZE);

    // Tiled matmul (see tiled.h); per-thread work buffers, 0 bytes when disabled. The
    // reservation is unconditional, the per expert eligibility is decided at dispatch time
    ৱর্ণ * tiled_scratch = incr_ptr_aligned(&wdata_cur, ggml_tiled_wdata_size(nth, dst), 64);

    GGML_ASSERT(params->wsize >= (size_t)((ৱর্ণ *) wdata_cur - (ৱর্ণ *) params->wdata));

    যদি (src1->type != vec_dot_type) {
        ৱর্ণ * wdata = params->wdata;

        ধ্রুৱ size_t nbw0 = ggml_type_size(vec_dot_type);
        ধ্রুৱ size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        ধ্রুৱ size_t nbw2 = nbw1*ne11;
        ধ্রুৱ size_t nbw3 = nbw2*ne12;

        প্রতিজ্ঞা(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

#যদি 0
        বাবে (int64_t i13 = 0; i13 < ne13; ++i13) {
            বাবে (int64_t i12 = ith; i12 < ne12; i12 += nth) {
                বাবে (int64_t i11 = 0; i11 < ne11; ++i11) {
                    from_float((ভিন্ন *)((ৱর্ণ *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (শূন্য *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                               ne10);
                }
            }
        }
#অন্যথা
        বাবে (int64_t i13 = 0; i13 < ne13; ++i13) {
            বাবে (int64_t i12 = 0; i12 < ne12; ++i12) {
                বাবে (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = ggml_blck_size(vec_dot_type);
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    from_float((ভিন্ন *)((ৱর্ণ *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10),
                               (শূন্য *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0),
                               (ne10_block_end - ne10_block_start) * bs);
                }
            }
        }
#endif
    }

    যদি (ith == 0) {
        // initialize matrix_row_counts
        memset(matrix_row_counts, 0, n_as*পরিমাণ(int64_t));

        // group rows by src0 matrix
        বাবে (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            বাবে (পূর্ণাঙ্ক id = 0; id < n_ids; ++id) {
                ধ্রুৱ int32_t i02 = *(ধ্রুৱ int32_t *) ((ধ্রুৱ ৱর্ণ *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);

                প্রতিজ্ঞা(i02 >= 0 && i02 < n_as);

                MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = (সংরচনা mmid_row_mapping) {id, iid1};
                matrix_row_counts[i02] += 1;
            }
        }
    }

    // reset current_chunk
    বাবে (পূর্ণাঙ্ক cur_a = ith; cur_a < n_as; cur_a += nth) {
        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);
        *current_chunk_ctr = nth;
    }

    ggml_barrier(params->threadpool);

    বাবে (পূর্ণাঙ্ক cur_a = 0; cur_a < n_as; ++cur_a) {
        ধ্রুৱ int64_t cne1 = matrix_row_counts[cur_a];

        যদি (cne1 == 0) {
            অনুৱর্ত;
        }

        // tiled takes over if profitable for this expert (see tiled.h)
        যদি (ggml_compute_forward_mul_mat_id_tiled(params, dst, cur_a, cne1, (ধ্রুৱ int32_t *) &MMID_MATRIX_ROW(cur_a, 0), tiled_scratch)) {
            অনুৱর্ত;
        }

        ধ্রুৱ ৱর্ণ * src0_cur = (ধ্রুৱ ৱর্ণ *) src0->data + cur_a * nb02;
        ধ্রুৱ শূন্য * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        ধ্রুৱ size_t row_size = ggml_row_size(vec_dot_type, ne10);

        ধ্রুৱ int64_t nr0 = ne01;
        ধ্রুৱ int64_t nr1 = cne1;

        পূর্ণাঙ্ক chunk_size = 16;
        যদি (nr0 == 1 || nr1 == 1) {
            chunk_size = 64;
        }

        // disable for NUMA
        ধ্রুৱ bool disable_chunking = ggml_is_numa();

        int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
        int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

        যদি (nchunk0 * nchunk1 < nth * 4 || disable_chunking) {
            nchunk0 = nr0 > nr1 ? nth : 1;
            nchunk1 = nr0 > nr1 ? 1 : nth;
        }

        ধ্রুৱ int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        ধ্রুৱ int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

        পূর্ণাঙ্ক current_chunk = ith;

        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);

        যেতিয়ালৈ (current_chunk < nchunk0 * nchunk1) {
            ধ্রুৱ int64_t ith0 = current_chunk % nchunk0;
            ধ্রুৱ int64_t ith1 = current_chunk / nchunk0;

            ধ্রুৱ int64_t ir0_start = dr0 * ith0;
            ধ্রুৱ int64_t ir0_end = MIN(ir0_start + dr0, nr0);

            ধ্রুৱ int64_t ir1_start = dr1 * ith1;
            ধ্রুৱ int64_t ir1_end = MIN(ir1_start + dr1, nr1);

            ggml_compute_forward_mul_mat_id_one_chunk(
                dst, src0, src1, ids, cur_a,
                ir0_start, ir0_end, ir1_start, ir1_end,
                src0_cur, matrix_rows, row_size, src1_cont, wdata
            );

            যদি (nth >= nchunk0 * nchunk1) {
                ভঙ্গ;
            }

            current_chunk = atomic_fetch_add_explicit(current_chunk_ctr, 1, memory_order_relaxed);
        }
    }
}

/////////////////////////////////

স্থির শূন্য ggml_compute_forward(সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * tensor) {
    GGML_ASSERT(params);

    যদি (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {
        ঘূৰাই;
    }

    // extra_buffer op?
    যদি (ggml_cpu_extra_compute_forward(params, tensor)) {
        ঘূৰাই;
    }

    ৱিকল্প (tensor->op) {
        পক্ষ GGML_OP_DUP:
            {
                ggml_compute_forward_dup(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_ADD:
            {
                ggml_compute_forward_add(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_ADD_ID:
            {
                ggml_compute_forward_add_id(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_ADD1:
            {
                ggml_compute_forward_add1(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_ACC:
            {
                ggml_compute_forward_acc(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_SUB:
            {
                ggml_compute_forward_sub(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_MUL:
            {
                ggml_compute_forward_mul(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_DIV:
            {
                ggml_compute_forward_div(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_SQR:
            {
                ggml_compute_forward_sqr(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_SQRT:
            {
                ggml_compute_forward_sqrt(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_LOG:
            {
                ggml_compute_forward_log(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_SIN:
            {
                ggml_compute_forward_sin(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_COS:
            {
                ggml_compute_forward_cos(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_SUM:
            {
                ggml_compute_forward_sum(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_SUM_ROWS:
            {
                ggml_compute_forward_sum_rows(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_CUMSUM:
            {
                ggml_compute_forward_cumsum(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_MEAN:
            {
                ggml_compute_forward_mean(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_ARGMAX:
            {
                ggml_compute_forward_argmax(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_গুনতি_EQUAL:
            {
                ggml_compute_forward_গুনতি_equal(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_REPEAT:
            {
                ggml_compute_forward_repeat(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_REPEAT_BACK:
            {
                ggml_compute_forward_repeat_back(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_CONCAT:
            {
                ggml_compute_forward_concat(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_SILU_BACK:
            {
                ggml_compute_forward_silu_back(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_NORM:
            {
                ggml_compute_forward_norm(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_RMS_NORM:
            {
                ggml_compute_forward_rms_norm(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_RMS_NORM_BACK:
            {
                ggml_compute_forward_rms_norm_back(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_GROUP_NORM:
            {
                ggml_compute_forward_group_norm(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_L2_NORM:
            {
                ggml_compute_forward_l2_norm(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_MUL_MAT:
            {
                ggml_compute_forward_mul_mat(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_MUL_MAT_ID:
            {
                ggml_compute_forward_mul_mat_id(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_OUT_PROD:
            {
                ggml_compute_forward_out_prod(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_SCALE:
            {
                ggml_compute_forward_scale(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_SET:
            {
                ggml_compute_forward_set(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_CPY:
            {
                ggml_compute_forward_cpy(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_CONT:
            {
                ggml_compute_forward_cont(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_GET_ROWS:
            {
                ggml_compute_forward_get_rows(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_GET_ROWS_BACK:
            {
                ggml_compute_forward_get_rows_back(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_SET_ROWS:
            {
                ggml_compute_forward_set_rows(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_DIAG:
            {
                ggml_compute_forward_diag(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_DIAG_MASK_INF:
            {
                ggml_compute_forward_diag_mask_inf(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_DIAG_MASK_ZERO:
            {
                ggml_compute_forward_diag_mask_zero(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_SOFT_MAX:
            {
                ggml_compute_forward_soft_max(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_SOFT_MAX_BACK:
            {
                ggml_compute_forward_soft_max_ext_back(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_ROPE:
            {
                ggml_compute_forward_rope(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_ROPE_BACK:
            {
                ggml_compute_forward_rope_back(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_CLAMP:
            {
                ggml_compute_forward_clamp(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_compute_forward_conv_transpose_1d(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_IM2COL:
            {
                ggml_compute_forward_im2col(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_IM2COL_BACK:
            {
                ggml_compute_forward_im2col_back_f32(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_IM2COL_3D:
            {
                ggml_compute_forward_im2col_3d(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_COL2IM_1D:
            {
                ggml_compute_forward_col2im_1d(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_CONV_2D:
            {
                ggml_compute_forward_conv_2d(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_CONV_3D:
            {
                ggml_compute_forward_conv_3d(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_CONV_2D_DW:
            {
                ggml_compute_forward_conv_2d_dw(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_CONV_TRANSPOSE_2D:
            {
                ggml_compute_forward_conv_transpose_2d(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_POOL_1D:
            {
                ggml_compute_forward_pool_1d(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_POOL_2D:
            {
                ggml_compute_forward_pool_2d(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_POOL_2D_BACK:
            {
                ggml_compute_forward_pool_2d_back(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_UPSCALE:
            {
                ggml_compute_forward_upscale(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_PAD:
            {
                ggml_compute_forward_pad(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_PAD_REFLECT_1D:
            {
                ggml_compute_forward_pad_reflect_1d(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_ROLL:
            {
                ggml_compute_forward_roll(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_ARANGE:
            {
                ggml_compute_forward_arange(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_TIMESTEP_EMBEDDING:
            {
                ggml_compute_forward_timestep_embedding(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_ARGSORT:
            {
                ggml_compute_forward_argsort(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_শীৰ্ষ_K:
            {
                ggml_compute_forward_শীৰ্ষ_k(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_LEAKY_RELU:
            {
                ggml_compute_forward_leaky_relu(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_TRI:
            {
                ggml_compute_forward_tri(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_FILL:
            {
                ggml_compute_forward_fill(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_FLASH_ATTN_EXT:
            {
                ggml_compute_forward_flash_attn_ext(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_FLASH_ATTN_BACK:
            {
                int32_t t = ggml_get_op_params_i32(tensor, 0);
                GGML_ASSERT(t == 0 || t == 1);
                bool masked = t != 0;
                ggml_compute_forward_flash_attn_back(params, masked, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_SSM_CONV:
            {
                ggml_compute_forward_ssm_conv(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_SSM_SCAN:
            {
                ggml_compute_forward_ssm_scan(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_WIN_PART:
            {
                ggml_compute_forward_win_part(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_WIN_UNPART:
            {
                ggml_compute_forward_win_unpart(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_UNARY:
            {
                ggml_compute_forward_unary(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_GLU:
            {
                ggml_compute_forward_glu(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_GET_REL_POS:
            {
                ggml_compute_forward_get_rel_pos(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_ADD_REL_POS:
            {
                ggml_compute_forward_add_rel_pos(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_RWKV_WKV6:
            {
                ggml_compute_forward_rwkv_wkv6(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_GATED_LINEAR_ATTN:
            {
                ggml_compute_forward_gla(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_RWKV_WKV7:
            {
                ggml_compute_forward_rwkv_wkv7(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_SOLVE_TRI:
            {
                ggml_compute_forward_solve_tri(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_GATED_DELTA_NET:
            {
                ggml_compute_forward_gated_delta_net(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_LIGHTNING_INDEXER:
            {
                ggml_compute_forward_lightning_indexer(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_DSV4_HC_COMB:
            {
                ggml_compute_forward_dsv4_hc_comb(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_DSV4_HC_PRE:
            {
                ggml_compute_forward_dsv4_hc_pre(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_DSV4_HC_POST:
            {
                ggml_compute_forward_dsv4_hc_post(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_শীৰ্ষ_K_MASK:
            {
                ggml_compute_forward_শীৰ্ষ_k_mask(params, tensor);
            } ভঙ্গ;
        পক্ষ GGML_OP_MAP_CUSTOM1:
            {
                ggml_compute_forward_map_custom1(params, tensor);
            }
            ভঙ্গ;
        পক্ষ GGML_OP_MAP_CUSTOM2:
            {
                ggml_compute_forward_map_custom2(params, tensor);
            }
            ভঙ্গ;
        পক্ষ GGML_OP_MAP_CUSTOM3:
            {
                ggml_compute_forward_map_custom3(params, tensor);
            }
            ভঙ্গ;
        পক্ষ GGML_OP_CUSTOM:
            {
                ggml_compute_forward_custom(params, tensor);
            }
            ভঙ্গ;
        পক্ষ GGML_OP_CROSS_ENTROPY_LOSS:
            {
                ggml_compute_forward_cross_entropy_loss(params, tensor);
            }
            ভঙ্গ;
        পক্ষ GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            {
                ggml_compute_forward_cross_entropy_loss_back(params, tensor);
            }
            ভঙ্গ;
        পক্ষ GGML_OP_OPT_STEP_ADAMW:
            {
                ggml_compute_forward_opt_step_adamw(params, tensor);
            }
            ভঙ্গ;
        পক্ষ GGML_OP_OPT_STEP_SGD:
            {
                ggml_compute_forward_opt_step_sgd(params, tensor);
            }
            ভঙ্গ;
        পক্ষ GGML_OP_NONE:
            {
                // nop
            } ভঙ্গ;
        পক্ষ GGML_OP_RESHAPE:
            {
                // nop
            } ভঙ্গ;
        পক্ষ GGML_OP_PERMUTE:
            {
                // nop
            } ভঙ্গ;
        পক্ষ GGML_OP_VIEW:
            {
                // nop
            } ভঙ্গ;
        পক্ষ GGML_OP_TRANSPOSE:
            {
                // nop
            } ভঙ্গ;
        পক্ষ GGML_OP_গুনতি:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// Android's libc implementation "bionic" does not support setting affinity
#যদি defined(__gnu_linux__)
স্থির শূন্য set_numa_thread_affinity(পূর্ণাঙ্ক thread_n) {
    যদি (!ggml_is_numa()) {
        ঘূৰাই;
    }

    পূর্ণাঙ্ক গ্ৰন্থি_অঙ্ক;
    পূর্ণাঙ্ক rv;
    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    ৱিকল্প(g_state.numa.numa_strategy) {
        পক্ষ GGML_NUMA_STRATEGY_DISTRIBUTE:
            // run thread on node_num thread_n / (threads per node)
            গ্ৰন্থি_অঙ্ক = thread_n % g_state.numa.n_nodes;
            ভঙ্গ;
        পক্ষ GGML_NUMA_STRATEGY_ISOLATE:
            // run thread on current_node
            গ্ৰন্থি_অঙ্ক = g_state.numa.current_গ্ৰন্থি;
            ভঙ্গ;
        পক্ষ GGML_NUMA_STRATEGY_NUMACTL:
            // use the cpuset that numactl gave us
            rv = pthread_setaffinity_np(pthread_self(), setsize, &g_state.numa.cpuset);
            যদি (rv) {
                fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n",strerror(rv));
            }
            ঘূৰাই;
        স্ৱভাৱ:
            ঘূৰাই;
    }

    সংরচনা ggml_numa_গ্ৰন্থি * গ্ৰন্থি = &g_state.numa.nodes[গ্ৰন্থি_অঙ্ক];

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    বাবে (size_t i = 0; i < গ্ৰন্থি->n_cpus; ++i) {
        CPU_SET_S(গ্ৰন্থি->cpus[i], setsize, cpus);
    }

    rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    যদি (rv) {
            fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}

স্থির শূন্য clear_numa_thread_affinity(শূন্য) {
    যদি (!ggml_is_numa()) {
        ঘূৰাই;
    }

    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    বাবে (unsigned i = 0; i < g_state.numa.total_cpus; ++i) {
        CPU_SET_S(i, setsize, cpus);
    }

    পূর্ণাঙ্ক rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    যদি (rv) {
        fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}
#অন্যথা
// TODO: Windows etc.
// (the linux implementation may also work on BSD, someone should test)
স্থির শূন্য set_numa_thread_affinity(পূর্ণাঙ্ক thread_n) { UNUSED(thread_n);  }
স্থির শূন্য clear_numa_thread_affinity(শূন্য) {}
#endif

স্থির পূর্ণাঙ্ক ggml_get_n_tasks(সংরচনা ggml_tensor * গ্ৰন্থি, পূর্ণাঙ্ক n_threads) {
    পূর্ণাঙ্ক n_tasks = 0;

    যদি (ggml_is_empty(গ্ৰন্থি)) {
        // no need to multi-thread a no-op
        n_tasks = 1;
        ঘূৰাই n_tasks;
    }

    ৱিকল্প (গ্ৰন্থি->op) {
        পক্ষ GGML_OP_CPY:
        পক্ষ GGML_OP_DUP:
        পক্ষ GGML_OP_CONT:
        পক্ষ GGML_OP_ADD:
        পক্ষ GGML_OP_ADD_ID:
        পক্ষ GGML_OP_ADD1:
        পক্ষ GGML_OP_ACC:
        পক্ষ GGML_OP_CUMSUM:
        পক্ষ GGML_OP_TRI:
        পক্ষ GGML_OP_FILL:
            {
                n_tasks = n_threads;
            } ভঙ্গ;
        পক্ষ GGML_OP_SUB:
        পক্ষ GGML_OP_SQR:
        পক্ষ GGML_OP_SQRT:
        পক্ষ GGML_OP_LOG:
        পক্ষ GGML_OP_SIN:
        পক্ষ GGML_OP_COS:
        পক্ষ GGML_OP_SUM:
        পক্ষ GGML_OP_SUM_ROWS:
        পক্ষ GGML_OP_MEAN:
        পক্ষ GGML_OP_ARGMAX:
            {
                n_tasks = 1;
            } ভঙ্গ;
        পক্ষ GGML_OP_গুনতি_EQUAL:
        পক্ষ GGML_OP_SOLVE_TRI:
        পক্ষ GGML_OP_GATED_DELTA_NET:
        পক্ষ GGML_OP_DSV4_HC_COMB:
        পক্ষ GGML_OP_DSV4_HC_PRE:
        পক্ষ GGML_OP_DSV4_HC_POST:
        পক্ষ GGML_OP_শীৰ্ষ_K_MASK:
            {
                n_tasks = n_threads;
            } ভঙ্গ;
        পক্ষ GGML_OP_REPEAT:
        পক্ষ GGML_OP_REPEAT_BACK:
        পক্ষ GGML_OP_LEAKY_RELU:
            {
                n_tasks = 1;
            } ভঙ্গ;
        পক্ষ GGML_OP_UNARY:
            ৱিকল্প (ggml_get_unary_op(গ্ৰন্থি)) {
                পক্ষ GGML_UNARY_OP_ABS:
                পক্ষ GGML_UNARY_OP_SGN:
                পক্ষ GGML_UNARY_OP_NEG:
                পক্ষ GGML_UNARY_OP_STEP:
                পক্ষ GGML_UNARY_OP_TANH:
                পক্ষ GGML_UNARY_OP_ELU:
                পক্ষ GGML_UNARY_OP_RELU:
                পক্ষ GGML_UNARY_OP_SIGMOID:
                পক্ষ GGML_UNARY_OP_HARDSWISH:
                পক্ষ GGML_UNARY_OP_HARDSIGMOID:
                পক্ষ GGML_UNARY_OP_EXP:
                পক্ষ GGML_UNARY_OP_SOFTPLUS:
                পক্ষ GGML_UNARY_OP_EXPM1:
                পক্ষ GGML_UNARY_OP_FLOOR:
                পক্ষ GGML_UNARY_OP_CEIL:
                পক্ষ GGML_UNARY_OP_ROUND:
                পক্ষ GGML_UNARY_OP_TRUNC:
                    {
                        n_tasks = 1;
                    } ভঙ্গ;

                পক্ষ GGML_UNARY_OP_GELU:
                পক্ষ GGML_UNARY_OP_GELU_ERF:
                পক্ষ GGML_UNARY_OP_GELU_QUICK:
                পক্ষ GGML_UNARY_OP_SILU:
                পক্ষ GGML_UNARY_OP_XIELU:
                    {
                        n_tasks = n_threads;
                    } ভঙ্গ;
                স্ৱভাৱ:
                    GGML_ABORT("fatal error");
            }
            ভঙ্গ;
        পক্ষ GGML_OP_GLU:
            ৱিকল্প (ggml_get_glu_op(গ্ৰন্থি)) {
                পক্ষ GGML_GLU_OP_REGLU:
                পক্ষ GGML_GLU_OP_GEGLU:
                পক্ষ GGML_GLU_OP_SWIGLU:
                পক্ষ GGML_GLU_OP_SWIGLU_OAI:
                পক্ষ GGML_GLU_OP_GEGLU_ERF:
                পক্ষ GGML_GLU_OP_GEGLU_QUICK:
                পক্ষ GGML_GLU_OP_SWIGLU_CLAMP:
                    {
                        n_tasks = n_threads;
                    } ভঙ্গ;
                স্ৱভাৱ:
                    GGML_ABORT("fatal error");
            }
            ভঙ্গ;
        পক্ষ GGML_OP_SILU_BACK:
        পক্ষ GGML_OP_MUL:
        পক্ষ GGML_OP_DIV:
        পক্ষ GGML_OP_NORM:
        পক্ষ GGML_OP_RMS_NORM:
        পক্ষ GGML_OP_RMS_NORM_BACK:
        পক্ষ GGML_OP_L2_NORM:
        পক্ষ GGML_OP_GROUP_NORM:
        পক্ষ GGML_OP_CONCAT:
        পক্ষ GGML_OP_MUL_MAT:
        পক্ষ GGML_OP_MUL_MAT_ID:
        পক্ষ GGML_OP_OUT_PROD:
        পক্ষ GGML_OP_SET_ROWS:
            {
                n_tasks = n_threads;
            } ভঙ্গ;
        পক্ষ GGML_OP_GET_ROWS:
            {
                // FIXME: get_rows can use additional threads, but the cost of launching additional threads
                // decreases performance with GPU offloading
                //n_tasks = n_threads;
                n_tasks = 1;
            } ভঙ্গ;
        পক্ষ GGML_OP_SCALE:
        পক্ষ GGML_OP_SET:
        পক্ষ GGML_OP_RESHAPE:
        পক্ষ GGML_OP_VIEW:
        পক্ষ GGML_OP_PERMUTE:
        পক্ষ GGML_OP_TRANSPOSE:
        পক্ষ GGML_OP_GET_ROWS_BACK:
        পক্ষ GGML_OP_DIAG:
            {
                n_tasks = 1;
            } ভঙ্গ;
        পক্ষ GGML_OP_DIAG_MASK_ZERO:
        পক্ষ GGML_OP_DIAG_MASK_INF:
        পক্ষ GGML_OP_SOFT_MAX_BACK:
        পক্ষ GGML_OP_ROPE:
        পক্ষ GGML_OP_ROPE_BACK:
        পক্ষ GGML_OP_ADD_REL_POS:
            {
                n_tasks = n_threads;
            } ভঙ্গ;
        পক্ষ GGML_OP_CLAMP:
            {
                n_tasks = 1; //TODO
            } ভঙ্গ;
        পক্ষ GGML_OP_SOFT_MAX:
            {
                n_tasks = MIN(n_threads, ggml_nrows(গ্ৰন্থি->src[0]));
            } ভঙ্গ;
        পক্ষ GGML_OP_IM2COL:
        পক্ষ GGML_OP_IM2COL_BACK:
        পক্ষ GGML_OP_IM2COL_3D:
        পক্ষ GGML_OP_CONV_2D:
        পক্ষ GGML_OP_CONV_3D:
        পক্ষ GGML_OP_CONV_2D_DW:
        পক্ষ GGML_OP_COL2IM_1D:
        পক্ষ GGML_OP_CONV_TRANSPOSE_1D:
        পক্ষ GGML_OP_CONV_TRANSPOSE_2D:
            {
                n_tasks = n_threads;
            } ভঙ্গ;
        পক্ষ GGML_OP_POOL_1D:
        পক্ষ GGML_OP_POOL_2D:
        পক্ষ GGML_OP_POOL_2D_BACK:
            {
                n_tasks = 1;
            } ভঙ্গ;
        পক্ষ GGML_OP_UPSCALE:
        পক্ষ GGML_OP_PAD:
        পক্ষ GGML_OP_PAD_REFLECT_1D:
        পক্ষ GGML_OP_ROLL:
        পক্ষ GGML_OP_ARANGE:
        পক্ষ GGML_OP_TIMESTEP_EMBEDDING:
        পক্ষ GGML_OP_ARGSORT:
        পক্ষ GGML_OP_শীৰ্ষ_K:
        পক্ষ GGML_OP_FLASH_ATTN_EXT:
        পক্ষ GGML_OP_FLASH_ATTN_BACK:
        পক্ষ GGML_OP_SSM_CONV:
        পক্ষ GGML_OP_SSM_SCAN:
        পক্ষ GGML_OP_LIGHTNING_INDEXER:
            {
                n_tasks = n_threads;
            } ভঙ্গ;
        পক্ষ GGML_OP_RWKV_WKV6:
        পক্ষ GGML_OP_GATED_LINEAR_ATTN:
        পক্ষ GGML_OP_RWKV_WKV7:
            {
                ধ্রুৱ int64_t n_heads = গ্ৰন্থি->src[1]->ne[1];
                n_tasks = MIN(n_threads, n_heads);
            } ভঙ্গ;
        পক্ষ GGML_OP_WIN_PART:
        পক্ষ GGML_OP_WIN_UNPART:
        পক্ষ GGML_OP_GET_REL_POS:
            {
                n_tasks = 1;
            } ভঙ্গ;
        পক্ষ GGML_OP_MAP_CUSTOM1:
            {
                সংরচনা ggml_map_custom1_op_params p;
                memcpy(&p, গ্ৰন্থি->op_params, পরিমাণ(p));
                যদি (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } অন্যথা {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } ভঙ্গ;
        পক্ষ GGML_OP_MAP_CUSTOM2:
            {
                সংরচনা ggml_map_custom2_op_params p;
                memcpy(&p, গ্ৰন্থি->op_params, পরিমাণ(p));
                যদি (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } অন্যথা {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } ভঙ্গ;
        পক্ষ GGML_OP_MAP_CUSTOM3:
            {
                সংরচনা ggml_map_custom3_op_params p;
                memcpy(&p, গ্ৰন্থি->op_params, পরিমাণ(p));
                যদি (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } অন্যথা {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } ভঙ্গ;
        পক্ষ GGML_OP_CUSTOM:
            {
                সংরচনা ggml_custom_op_params p;
                memcpy(&p, গ্ৰন্থি->op_params, পরিমাণ(p));
                যদি (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } অন্যথা {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } ভঙ্গ;
        পক্ষ GGML_OP_CROSS_ENTROPY_LOSS:
        পক্ষ GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        পক্ষ GGML_OP_OPT_STEP_ADAMW:
        পক্ষ GGML_OP_OPT_STEP_SGD:
            {
                n_tasks = n_threads;
            } ভঙ্গ;
        পক্ষ GGML_OP_NONE:
            {
                n_tasks = 1;
            } ভঙ্গ;
        পক্ষ GGML_OP_গুনতি:
            {
                GGML_ABORT("fatal error");
            }
        স্ৱভাৱ:
            {
                fprintf(stderr, "%s: op not implemented: ", __func__);
                যদি (গ্ৰন্থি->op < GGML_OP_গুনতি) {
                    fprintf(stderr, "%s\n", ggml_op_name(গ্ৰন্থি->op));
                } অন্যথা {
                    fprintf(stderr, "%d\n", গ্ৰন্থি->op);
                }
                GGML_ABORT("fatal error");
            }
    }

    প্রতিজ্ঞা(n_tasks > 0);

    ঘূৰাই n_tasks;
}

স্থির thread_ret_t ggml_graph_compute_secondary_thread(শূন্য* data);

#যদি defined(_WIN32)
#অন্তর্ভাৱ "windows.h"

// TODO: support > 64 CPUs
স্থির bool ggml_thread_apply_affinity(bool * mask) {
    HANDLE    h = GetCurrentThread();
    uint64_t  bitmask = 0ULL;

    প্রতিজ্ঞা(GGML_MAX_N_THREADS >= 64);

    বাবে (int32_t i = 0; i < 8; i++) {
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

    বাবে (int32_t i = 64; i < GGML_MAX_N_THREADS; i++) {
        যদি (mask[i]) {
            fprintf(stderr, "warn: setting thread-affinity for > 64 CPUs isn't supported on windows!\n");
            ভঙ্গ;
        }
    }

    DWORD_PTR m = (DWORD_PTR)bitmask;

    m = SetThreadAffinityMask(h, m);

    ঘূৰাই m != 0;
}

স্থির bool ggml_thread_apply_priority(int32_t prio) {
    // Note that on Windows the Process Priority Class must be updated in order to set Thread priority.
    // This is up to the applications.
    DWORD p = THREAD_PRIORITY_NORMAL;
    ৱিকল্প (prio) {
        পক্ষ GGML_SCHED_PRIO_নিচেৰ:      p = THREAD_PRIORITY_BELOW_NORMAL;  ভঙ্গ;
        পক্ষ GGML_SCHED_PRIO_NORMAL:   p = THREAD_PRIORITY_NORMAL;        ভঙ্গ;
        পক্ষ GGML_SCHED_PRIO_MEDIUM:   p = THREAD_PRIORITY_ABOVE_NORMAL;  ভঙ্গ;
        পক্ষ GGML_SCHED_PRIO_উপৰেৰ:     p = THREAD_PRIORITY_HIGHEST;       ভঙ্গ;
        পক্ষ GGML_SCHED_PRIO_REALTIME: p = THREAD_PRIORITY_TIME_CRITICAL; ভঙ্গ;
    }

    যদি (prio != GGML_SCHED_PRIO_নিচেৰ) {
        // Tell Windows that this thread should not be throttled (needs its own CPU core).
        // Newer Windows 11 versions aggressively park (offline) CPU cores and often place
        // all our threads onto the first 4 cores which results in terrible performance with
        // n_threads > 4
        #যদি _WIN32_WINNT >= 0x0602
        THREAD_POWER_THROTTLING_STATE t;
        ZeroMemory(&t, পরিমাণ(t));
        t.Version     = THREAD_POWER_THROTTLING_CURRENT_VERSION;
        t.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
        t.StateMask   = 0;

        যদি (!SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &t, পরিমাণ(t))) {
            GGML_LOG_DEBUG("failed to disable thread power throttling %d : (%d)\n", prio, (পূর্ণাঙ্ক) GetLastError());
            ঘূৰাই false;
        }
        #endif
    }

    যদি (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        ঘূৰাই true;
    }

    যদি (!SetThreadPriority(GetCurrentThread(), p)) {
        fprintf(stderr, "warn: failed to set thread priority %d : (%d)\n", prio, (পূর্ণাঙ্ক) GetLastError());
        ঘূৰাই false;
    }

    ঘূৰাই true;
}

#নোচেদ্যদি defined(__APPLE__)
#অন্তর্ভাৱ <sys/types.h>
#অন্তর্ভাৱ <sys/resource.h>

স্থির bool ggml_thread_apply_affinity(ধ্রুৱ bool * mask) {
    // Not supported on Apple platforms
    UNUSED(mask);
    ঘূৰাই true;
}

স্থির bool ggml_thread_apply_priority(int32_t prio) {
    সংরচনা sched_param p;
    int32_t policy = SCHED_OTHER;
    ৱিকল্প (prio) {
        // TODO: there seems to be no way to set lower prio on Apple platforms
        পক্ষ GGML_SCHED_PRIO_নিচেৰ:      policy = SCHED_OTHER; p.sched_priority = 0;  ভঙ্গ;
        পক্ষ GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  ভঙ্গ;
        পক্ষ GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; ভঙ্গ;
        পক্ষ GGML_SCHED_PRIO_উপৰেৰ:     policy = SCHED_FIFO;  p.sched_priority = 80; ভঙ্গ;
        পক্ষ GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; ভঙ্গ;
    }

    যদি (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        ঘূৰাই true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    যদি (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        ঘূৰাই false;
    }

    ঘূৰাই true;
}

#নোচেদ্যদি defined(__linux__)
// TODO: this may not work on BSD, to be verified

স্থির bool ggml_thread_apply_affinity(ধ্রুৱ bool * mask) {
    cpu_set_t cpuset;
    পূর্ণাঙ্ক err;

    CPU_ZERO(&cpuset);

    বাবে (uint32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
        যদি (mask[i]) {
            GGML_PRINT_DEBUG("Thread %lx: adding %d to cpuset\n", pthread_self(), i);
            CPU_SET(i, &cpuset);
        }
    }

#ifdef __ANDROID__
    err = sched_setaffinity(0, পরিমাণ(cpuset), &cpuset);
    যদি (err < 0) {
        err = errno;
    }
#অন্যথা
    err = pthread_setaffinity_np(pthread_self(), পরিমাণ(cpuset), &cpuset);
#endif
    যদি (err != 0) {
        fprintf(stderr, "warn: failed to set affinity mask 0x%llx : %s (%d)\n", (unsigned long long)mask, strerror(err), err);
        ঘূৰাই false;
    }

    ঘূৰাই true;
}

স্থির bool ggml_thread_apply_priority(int32_t prio) {
    সংরচনা sched_param p;
    int32_t policy = SCHED_OTHER;
    ৱিকল্প (prio) {
        পক্ষ GGML_SCHED_PRIO_নিচেৰ:      policy = SCHED_BATCH; p.sched_priority = 0;  ভঙ্গ;
        পক্ষ GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  ভঙ্গ;
        পক্ষ GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; ভঙ্গ;
        পক্ষ GGML_SCHED_PRIO_উপৰেৰ:     policy = SCHED_FIFO;  p.sched_priority = 80; ভঙ্গ;
        পক্ষ GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; ভঙ্গ;
    }

    যদি (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        ঘূৰাই true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    যদি (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        ঘূৰাই false;
    }

    ঘূৰাই true;
}

#অন্যথা // unsupported platforms

স্থির bool ggml_thread_apply_affinity(ধ্রুৱ bool * mask) {
    UNUSED(mask);
    ঘূৰাই true;
}

স্থির bool ggml_thread_apply_priority(int32_t prio) {
    UNUSED(prio);
    ঘূৰাই true;
}

#endif

স্থির bool ggml_thread_cpumask_is_valid(ধ্রুৱ bool * mask) {
    বাবে (পূর্ণাঙ্ক i = 0; i < GGML_MAX_N_THREADS; i++) {
        যদি (mask[i]) { ঘূৰাই true; }
    }
    ঘূৰাই false;
}

স্থির শূন্য ggml_thread_cpumask_next(ধ্রুৱ bool * global_mask, bool * local_mask, bool strict, int32_t* iter) {
    যদি (!strict) {
        memcpy(local_mask, global_mask, GGML_MAX_N_THREADS);
        ঘূৰাই;
    } অন্যথা {
        memset(local_mask, 0, GGML_MAX_N_THREADS);
        int32_t base_idx = *iter;
        বাবে (int32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
            int32_t idx = base_idx + i;
            যদি (idx >= GGML_MAX_N_THREADS) {
                // Just a cheaper modulo
                idx -= GGML_MAX_N_THREADS;
            }
            যদি (global_mask[idx]) {
                local_mask[idx] = 1;
                *iter = idx + 1;
                ঘূৰাই;
            }
        }
    }
}

শূন্য ggml_threadpool_free(সংরচনা ggml_threadpool* threadpool) {
    যদি (!threadpool) ঘূৰাই;

    ধ্রুৱ পূর্ণাঙ্ক n_threads = threadpool->n_threads;

#ifndef GGML_USE_OPENMP
    সংরচনা ggml_compute_state* workers = threadpool->workers;

    ggml_mutex_lock(&threadpool->mutex);

    threadpool->stop = true;
    threadpool->pause = false;

    ggml_cond_broadcast(&threadpool->cond);
    ggml_mutex_unlock(&threadpool->mutex);

    বাবে (পূর্ণাঙ্ক j = 1; j < n_threads; j++) {
        int32_t rc = ggml_thread_join(workers[j].thrd, রিক্ত);
        GGML_ASSERT(rc == GGML_EXIT_SUCCESS || rc == GGML_EXIT_ABORTED);
        UNUSED(rc);
    }

    ggml_mutex_destroy(&threadpool->mutex);
    ggml_cond_destroy(&threadpool->cond);
#endif // GGML_USE_OPENMP

    ধ্রুৱ size_t workers_size = পরিমাণ(সংরচনা ggml_compute_state) * n_threads;
    ggml_aligned_free(threadpool->workers, workers_size);
    ggml_aligned_free(threadpool, পরিমাণ(সংরচনা ggml_threadpool));
}

#ifndef GGML_USE_OPENMP
// pause/resume must be called under mutex
স্থির শূন্য ggml_threadpool_pause_locked(সংরচনা ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Pausing threadpool\n");
    threadpool->pause = true;
    ggml_cond_broadcast(&threadpool->cond);
}

স্থির শূন্য ggml_threadpool_resume_locked(সংরচনা ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Resuming threadpool\n");
    threadpool->pause = false;
    ggml_cond_broadcast(&threadpool->cond);
}
#endif

শূন্য ggml_threadpool_pause(সংরচনা ggml_threadpool * threadpool) {
#ifndef GGML_USE_OPENMP
    ggml_mutex_lock(&threadpool->mutex);
    যদি (!threadpool->pause) {
       ggml_threadpool_pause_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#অন্যথা
    UNUSED(threadpool);
#endif
}

শূন্য ggml_threadpool_resume(সংরচনা ggml_threadpool * threadpool) {
#ifndef GGML_USE_OPENMP
    ggml_mutex_lock(&threadpool->mutex);
    যদি (threadpool->pause) {
       ggml_threadpool_resume_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#অন্যথা
    UNUSED(threadpool);
#endif
}

সংরচনা ggml_cplan ggml_graph_plan(
          ধ্রুৱ সংরচনা ggml_cgraph * cgraph,
                               পূর্ণাঙ্ক   n_threads,
            সংরচনা ggml_threadpool * threadpool) {

    যদি (threadpool == রিক্ত) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
    }
    যদি (n_threads <= 0) {
        n_threads = threadpool ? threadpool->n_threads : GGML_DEFAULT_N_THREADS;
    }

#যদি defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
    // Emscripten without pthreads support can only use a single thread
    n_threads = 1;
#endif

#যদি defined(__wasi__)
    // WASI doesn't support parallelism yet
    n_threads = 1;
#endif

    size_t work_size = 0;

    সংরচনা ggml_cplan cplan;
    memset(&cplan, 0, পরিমাণ(সংরচনা ggml_cplan));

    পূর্ণাঙ্ক max_tasks = 1;

    // thread scheduling for the different operations + work buffer size estimation
    বাবে (পূর্ণাঙ্ক i = 0; i < cgraph->n_nodes; i++) {
        সংরচনা ggml_tensor * গ্ৰন্থি = cgraph->nodes[i];

        ধ্রুৱ পূর্ণাঙ্ক n_tasks = ggml_get_n_tasks(গ্ৰন্থি, n_threads);

        max_tasks = MAX(max_tasks, n_tasks);

        size_t cur = 0;

        যদি (!ggml_cpu_extra_work_size(n_threads, গ্ৰন্থি, &cur)) {
            ৱিকল্প (গ্ৰন্থি->op) {
                পক্ষ GGML_OP_CPY:
                পক্ষ GGML_OP_DUP:
                    {
                        যদি (ggml_is_quantized(গ্ৰন্থি->type) ||
                            // F16 -> BF16 and BF16 -> F16 copies go through intermediate F32
                            (গ্ৰন্থি->src[0]->type == GGML_TYPE_F16  && গ্ৰন্থি->src[1] && গ্ৰন্থি->src[1]->type == GGML_TYPE_BF16) ||
                            (গ্ৰন্থি->src[0]->type == GGML_TYPE_BF16 && গ্ৰন্থি->src[1] && গ্ৰন্থি->src[1]->type == GGML_TYPE_F16) ||
                            // conversion between F32 and I32
                            (গ্ৰন্থি->src[0]->type == GGML_TYPE_F32 && গ্ৰন্থি->src[1] && গ্ৰন্থি->src[1]->type == GGML_TYPE_I32) ||
                            (গ্ৰন্থি->src[0]->type == GGML_TYPE_I32 && গ্ৰন্থি->src[1] && গ্ৰন্থি->src[1]->type == GGML_TYPE_F32)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * গ্ৰন্থি->ne[0] * n_tasks;
                        }
                    } ভঙ্গ;
                পক্ষ GGML_OP_ADD:
                পক্ষ GGML_OP_ADD_ID:
                পক্ষ GGML_OP_ADD1:
                    {
                        যদি (ggml_is_quantized(গ্ৰন্থি->src[0]->type)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * গ্ৰন্থি->src[0]->ne[0] * n_tasks;
                        }
                    } ভঙ্গ;
                পক্ষ GGML_OP_ACC:
                    {
                        যদি (ggml_is_quantized(গ্ৰন্থি->src[0]->type)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * গ্ৰন্থি->src[1]->ne[0] * n_tasks;
                        }
                    } ভঙ্গ;
                পক্ষ GGML_OP_গুনতি_EQUAL:
                    {
                        cur = ggml_type_size(গ্ৰন্থি->type)*n_tasks;
                    } ভঙ্গ;
                পক্ষ GGML_OP_MUL_MAT:
                    {
                        ধ্রুৱ গণনা ggml_type vec_dot_type = type_traits_cpu[গ্ৰন্থি->src[0]->type].vec_dot_type;
                        যদি (গ্ৰন্থি->src[1]->type != vec_dot_type) {
                            cur = ggml_row_size(vec_dot_type, ggml_nelements(গ্ৰন্থি->src[1]));
                        }
                        // Workspace for tiled (see tiled.h)
                        cur = GGML_PAD(cur, 64);
                        cur += ggml_tiled_wdata_size(n_tasks, গ্ৰন্থি);
                    } ভঙ্গ;
                পক্ষ GGML_OP_MUL_MAT_ID:
                    {
                        cur = 0;
                        ধ্রুৱ সংরচনা ggml_tensor * src0 = গ্ৰন্থি->src[0];
                        ধ্রুৱ সংরচনা ggml_tensor * src1 = গ্ৰন্থি->src[1];
                        ধ্রুৱ সংরচনা ggml_tensor * ids = গ্ৰন্থি->src[2];
                        ধ্রুৱ গণনা ggml_type vec_dot_type = type_traits_cpu[src0->type].vec_dot_type;
                        ধ্রুৱ পূর্ণাঙ্ক n_as = src0->ne[2];
                        // src1
                        যদি (src1->type != vec_dot_type) {
                            cur += ggml_row_size(vec_dot_type, ggml_nelements(src1)) + পরিমাণ(int64_t);
                        }
                        // matrix_row_counts
                        cur += n_as * পরিমাণ(int64_t) + পরিমাণ(int64_t);
                        // matrix_rows
                        cur += n_as*ids->ne[0]*ids->ne[1]*পরিমাণ(সংরচনা mmid_row_mapping) + পরিমাণ(int64_t);
                        // atomic_current_chunk
                        cur += CACHE_সাৰি_SIZE*n_as + CACHE_সাৰি_SIZE;
                        // Workspace for tiled (see tiled.h)
                        cur = GGML_PAD(cur, 64);
                        cur += ggml_tiled_wdata_size(n_tasks, গ্ৰন্থি);
                    } ভঙ্গ;
                পক্ষ GGML_OP_OUT_PROD:
                    {
                        যদি (ggml_is_quantized(গ্ৰন্থি->src[0]->type) ||
                            গ্ৰন্থি->src[0]->type == GGML_TYPE_F16) {
                            cur = ggml_type_size(GGML_TYPE_F32) * গ্ৰন্থি->src[0]->ne[0] * n_tasks;
                        }
                    } ভঙ্গ;
                পক্ষ GGML_OP_SET_ROWS:
                    {
                        যদি (গ্ৰন্থি->src[0]->type == GGML_TYPE_F16 && গ্ৰন্থি->type != GGML_TYPE_F16) {
                            cur = ggml_type_size(GGML_TYPE_F32) * গ্ৰন্থি->src[0]->ne[0] * n_tasks;
                        }
                    } ভঙ্গ;
                পক্ষ GGML_OP_SOFT_MAX:
                পক্ষ GGML_OP_ROPE:
                পক্ষ GGML_OP_ROPE_BACK:
                    {
                        cur = ggml_type_size(GGML_TYPE_F32) * গ্ৰন্থি->ne[0] * n_tasks;
                    } ভঙ্গ;
                পক্ষ GGML_OP_CONV_TRANSPOSE_1D:
                    {
                        GGML_ASSERT(গ্ৰন্থি->src[0]->ne[3] == 1);
                        GGML_ASSERT(গ্ৰন্থি->src[1]->ne[2] == 1);
                        GGML_ASSERT(গ্ৰন্থি->src[1]->ne[3] == 1);

                        ধ্রুৱ int64_t ne00 = গ্ৰন্থি->src[0]->ne[0];  // K
                        ধ্রুৱ int64_t ne01 = গ্ৰন্থি->src[0]->ne[1];  // Cout
                        ধ্রুৱ int64_t ne02 = গ্ৰন্থি->src[0]->ne[2];  // Cin
                        ধ্রুৱ int64_t ne10 = গ্ৰন্থি->src[1]->ne[0];  // L
                        ধ্রুৱ int64_t ne11 = গ্ৰন্থি->src[1]->ne[1];  // Cin

                        যদি ((গ্ৰন্থি->src[0]->type == GGML_TYPE_F16 ||
                             গ্ৰন্থি->src[0]->type == GGML_TYPE_BF16) &&
                            গ্ৰন্থি->src[1]->type == GGML_TYPE_F32) {
                            cur += পরিমাণ(ggml_fp16_t)*ne00*ne01*ne02;
                            cur += পরিমাণ(ggml_fp16_t)*ne10*ne11;
                        } অন্যথা যদি (গ্ৰন্থি->src[0]->type == GGML_TYPE_F32 &&
                                   গ্ৰন্থি->src[1]->type == GGML_TYPE_F32) {
                            cur += পরিমাণ(ভিন্ন)*ne00*ne01*ne02;
                            cur += পরিমাণ(ভিন্ন)*ne10*ne11;
                        } অন্যথা {
                            GGML_ABORT("fatal error");
                        }
                    } ভঙ্গ;
                পক্ষ GGML_OP_CONV_2D:
                পক্ষ GGML_OP_CONV_3D:
                    {
                        cur = GGML_IM2COL_WORK_SIZE;
                    } ভঙ্গ;
                পক্ষ GGML_OP_CONV_TRANSPOSE_2D:
                    {
                        ধ্রুৱ int64_t ne00 = গ্ৰন্থি->src[0]->ne[0]; // W
                        ধ্রুৱ int64_t ne01 = গ্ৰন্থি->src[0]->ne[1]; // H
                        ধ্রুৱ int64_t ne02 = গ্ৰন্থি->src[0]->ne[2]; // Channels Out
                        ধ্রুৱ int64_t ne03 = গ্ৰন্থি->src[0]->ne[3]; // Channels In

                        ধ্রুৱ int64_t ne10 = গ্ৰন্থি->src[1]->ne[0]; // W
                        ধ্রুৱ int64_t ne11 = গ্ৰন্থি->src[1]->ne[1]; // H
                        ধ্রুৱ int64_t ne12 = গ্ৰন্থি->src[1]->ne[2]; // Channels In
                        ধ্রুৱ int64_t ne13 = গ্ৰন্থি->src[1]->ne[3]; // Batch

                        GGML_ASSERT(গ্ৰন্থি->src[0]->type == GGML_TYPE_F16 || গ্ৰন্থি->src[0]->type == GGML_TYPE_F32);
                        GGML_ASSERT(গ্ৰন্থি->src[1]->type == GGML_TYPE_F32);

                        cur += ggml_type_size(গ্ৰন্থি->src[0]->type) * ne00 * ne01 * ne02 * ne03;
                        cur += ggml_type_size(গ্ৰন্থি->src[0]->type) * ne10 * ne11 * ne12 * ne13;

                    } ভঙ্গ;
                পক্ষ GGML_OP_শীৰ্ষ_K:
                    {
                        cur += পরিমাণ(int32_t)*গ্ৰন্থি->src[0]->ne[0]*n_tasks;
                    } ভঙ্গ;
                পক্ষ GGML_OP_FLASH_ATTN_EXT:
                    {
                        ধ্রুৱ int64_t neq2 = গ্ৰন্থি->src[0]->ne[2]; // number of query heads
                        ধ্রুৱ int64_t DK = গ্ৰন্থি->src[1]->ne[0];
                        ধ্রুৱ int64_t DV = গ্ৰন্থি->src[2]->ne[0];

                        // Tiled flash attention scratch (tile sizes defined in common.h)
                        // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
                        size_t prefill  = পরিমাণ(ভিন্ন)*(GGML_FA_TILE_Q*DK + 2*GGML_FA_TILE_Q*GGML_FA_TILE_KV + GGML_FA_TILE_Q*DV + GGML_FA_TILE_KV*DV + GGML_FA_TILE_KV*DK)*n_tasks;

                        // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
                        // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
                        size_t n_chunks = n_tasks;
                        size_t decode   = পরিমাণ(ভিন্ন)*(neq2*n_chunks*(2+DV) + n_tasks*(DK + 2*DV));

                        cur += MAX(prefill, decode);
                    } ভঙ্গ;
                পক্ষ GGML_OP_FLASH_ATTN_BACK:
                    {
                        ধ্রুৱ int64_t    D = গ্ৰন্থি->src[0]->ne[0];
                        ধ্রুৱ int64_t ne11 = ggml_up(গ্ৰন্থি->src[1]->ne[1], GGML_SOFT_MAX_UNROLL);
                        ধ্রুৱ int64_t mxDn = MAX(D, ne11) * 2; // *2 because of S and SM in ggml_compute_forward_flash_attn_back
                        যদি (গ্ৰন্থি->src[1]->type == GGML_TYPE_F32) {
                            cur  = পরিমাণ(ভিন্ন)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += পরিমাণ(ভিন্ন)*mxDn*n_tasks; // this is overestimated by x2
                        } অন্যথা যদি (গ্ৰন্থি->src[1]->type == GGML_TYPE_F16) {
                            cur  = পরিমাণ(ভিন্ন)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += পরিমাণ(ভিন্ন)*mxDn*n_tasks; // this is overestimated by x2
                        } অন্যথা যদি (গ্ৰন্থি->src[1]->type == GGML_TYPE_BF16) {
                            cur  = পরিমাণ(ভিন্ন)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += পরিমাণ(ভিন্ন)*mxDn*n_tasks; // this is overestimated by x2
                        }
                    } ভঙ্গ;

                পক্ষ GGML_OP_CROSS_ENTROPY_LOSS:
                    {
                        cur = ggml_type_size(গ্ৰন্থি->type)*(n_tasks + গ্ৰন্থি->src[0]->ne[0]*n_tasks);
                    } ভঙ্গ;
                পক্ষ GGML_OP_GATED_DELTA_NET:
                    {
                        ধ্রুৱ int64_t S_v = গ্ৰন্থি->src[2]->ne[0];
                        ধ্রুৱ int64_t K   = ggml_get_op_params_i32(গ্ৰন্থি, 0);
                        ধ্রুৱ int64_t per_thread = S_v + (K > 1 ? S_v * S_v : 0);
                        cur = per_thread * পরিমাণ(ভিন্ন) * n_tasks;
                    } ভঙ্গ;
                পক্ষ GGML_OP_গুনতি:
                    {
                        GGML_ABORT("fatal error");
                    }
                পক্ষ GGML_OP_LIGHTNING_INDEXER:
                    {
                        // temp buffer for dequantizing lightning indexer keys
                        ধ্রুৱ int64_t ne10 = গ্ৰন্থি->src[1]->ne[0];
                        cur += পরিমাণ(ভিন্ন)*ne10*n_tasks;
                    } ভঙ্গ;
                স্ৱভাৱ:
                    ভঙ্গ;
            }
        }

        work_size = MAX(work_size, cur);
    }

    যদি (work_size > 0) {
        work_size += CACHE_সাৰি_SIZE*(n_threads);
    }

    cplan.threadpool = threadpool;
    cplan.n_threads  = MIN(max_tasks, n_threads);
    cplan.work_size  = work_size;
    cplan.work_data  = রিক্ত;

    ঘূৰাই cplan;
}


// Try to fuse the current node with subsequent nodes for better performance.
// Returns the number of nodes skipped by fusion (>=1), or 0 if no fusion was applied.
স্থির bool ggml_cpu_disable_fusion = false;  // initialized once in ggml_cpu_init(), read-only afterwards

স্থির পূর্ণাঙ্ক ggml_cpu_try_fuse_ops(
        ধ্রুৱ সংরচনা ggml_cgraph * cgraph,
        ধ্রুৱ পূর্ণাঙ্ক গ্ৰন্থি_n,
        ধ্রুৱ সংরচনা ggml_compute_params * params,
        ধ্রুৱ সংরচনা ggml_cplan * cplan) {

    যদি (ggml_cpu_disable_fusion || cplan->use_ref) {
        ঘূৰাই 0;
    }

    সংরচনা ggml_tensor * গ্ৰন্থি = cgraph->nodes[গ্ৰন্থি_n];

    যদি (গ্ৰন্থি->op == GGML_OP_RMS_NORM) {
        // RMS_NORM + MUL fusion
        ধ্রুৱ গণনা ggml_op fuse_ops[] = { GGML_OP_RMS_NORM, GGML_OP_MUL };
        যদি (ggml_can_fuse(cgraph, গ্ৰন্থি_n, fuse_ops, 2)) {
            সংরচনা ggml_tensor * mul_গ্ৰন্থি = cgraph->nodes[গ্ৰন্থি_n + 1];
            ধ্রুৱ সংরচনা ggml_tensor * mul_w = (mul_গ্ৰন্থি->src[0] == গ্ৰন্থি)
                ? mul_গ্ৰন্থি->src[1] : mul_গ্ৰন্থি->src[0];
            যদি (গ্ৰন্থি->src[0]->type  == GGML_TYPE_F32 &&
                mul_গ্ৰন্থি->type      == GGML_TYPE_F32 &&
                mul_w->type         == GGML_TYPE_F32 &&
                mul_w->ne[0]        == গ্ৰন্থি->ne[0]   &&
                mul_w->nb[0]        == পরিমাণ(ভিন্ন)) {

                ggml_compute_forward_rms_norm_mul_fused(params, গ্ৰন্থি, mul_গ্ৰন্থি);
                ঘূৰাই 1;
            }
        }
    }

    ঘূৰাই 0;
}

স্থির thread_ret_t ggml_graph_compute_thread(শূন্য * data) {
    সংরচনা ggml_compute_state * state = (সংরচনা ggml_compute_state *) data;
    সংরচনা ggml_threadpool    * tp    = state->threadpool;

    ধ্রুৱ সংরচনা ggml_cgraph * cgraph = tp->cgraph;
    ধ্রুৱ সংরচনা ggml_cplan  * cplan  = tp->cplan;

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(state->ith);
#অন্যথা
    set_numa_thread_affinity(state->ith);
#endif

    সংরচনা ggml_compute_params params = {
        /*.ith        =*/ state->ith,
        /*.nth        =*/ atomic_load_explicit(&tp->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK,
        /*.wsize      =*/ cplan->work_size,
        /*.wdata      =*/ cplan->work_data,
        /*.threadpool =*/ tp,
        /*.use_ref    =*/ cplan->use_ref,
    };

#ifdef GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p\n", state->ith, (ধ্রুৱ শূন্য *)cplan);
#অন্যথা
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p last-graph %d\n", state->ith, (ধ্রুৱ শূন্য *)cplan, state->last_graph);
#endif

    বাবে (পূর্ণাঙ্ক গ্ৰন্থি_n = 0; গ্ৰন্থি_n < cgraph->n_nodes && atomic_load_explicit(&tp->abort, memory_order_relaxed) != গ্ৰন্থি_n; গ্ৰন্থি_n++) {
        সংরচনা ggml_tensor * গ্ৰন্থি = cgraph->nodes[গ্ৰন্থি_n];

        যদি (ggml_op_is_empty(গ্ৰন্থি->op)) {
            // skip NOPs
            অনুৱর্ত;
        }

        যদি ((গ্ৰন্থি->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            অনুৱর্ত;
        }

        // TODO: move fused-op detection into ggml_graph_plan so fusion decisions are made once at planning time
        // Try fused ops, fall back to normal compute
        ধ্রুৱ পূর্ণাঙ্ক n_fused = ggml_cpu_try_fuse_ops(cgraph, গ্ৰন্থি_n, &params, cplan);
        যদি (n_fused > 0) {
            গ্ৰন্থি_n += n_fused;
        } অন্যথা {
            ggml_compute_forward(&params, গ্ৰন্থি);
        }

        যদি (state->ith == 0 && cplan->abort_callback &&
                cplan->abort_callback(cplan->abort_callback_data)) {
            atomic_store_explicit(&tp->abort, গ্ৰন্থি_n + 1, memory_order_relaxed);
            tp->ec    = GGML_STATUS_ABORTED;
        }

        যদি (গ্ৰন্থি_n + 1 < cgraph->n_nodes) {
            ggml_barrier(state->threadpool);
        }
    }

#ifdef GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p\n", state->ith, (ধ্রুৱ শূন্য *)cplan);
#অন্যথা
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p last-graph %d\n", state->ith, (ধ্রুৱ শূন্য *)cplan, state->last_graph);
#endif

    ggml_barrier(state->threadpool);

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(state->ith);
#endif

    ঘূৰাই 0;
}

#ifndef GGML_USE_OPENMP

// check if thread is ready to proceed (exit from polling or sleeping)
// returns true if loops should exit, sets state->pending to indicate new work
স্থির অন্তঃস্থ bool ggml_graph_compute_thread_ready(সংরচনা ggml_compute_state * state) {
    সংরচনা ggml_threadpool * threadpool = state->threadpool;

    যদি (state->pending || threadpool->stop || threadpool->pause) { ঘূৰাই true; }

    // check for new graph/work
    পূর্ণাঙ্ক n_graph   = atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed);
    পূর্ণাঙ্ক n_threads = n_graph & GGML_THREADPOOL_N_THREADS_MASK;
    যদি (n_graph != state->last_graph) {
        state->pending    = (state->ith < n_threads);
        state->last_graph = n_graph;
        ঘূৰাই true;
    }

    ঘূৰাই false;
}

// sync thread state after polling
স্থির অন্তঃস্থ শূন্য ggml_graph_compute_thread_sync(সংরচনা ggml_compute_state * state) {
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&state->threadpool->n_graph, 0, memory_order_seq_cst);
    #অন্যথা
    atomic_thread_fence(memory_order_seq_cst);
    #endif
    UNUSED(state);
}

স্থির অন্তঃস্থ bool ggml_graph_compute_poll_for_work(সংরচনা ggml_compute_state * state) {
    সংরচনা ggml_threadpool * threadpool = state->threadpool;

    // This seems to make 0 ... 100 a decent range for polling level across modern processors.
    // Perhaps, we can adjust it dynamically based on load and things.
    ধ্রুৱ uint64_t n_rounds = 1024UL * 128 * threadpool->poll;

    বাবে (uint64_t i=0; !ggml_graph_compute_thread_ready(state) && i < n_rounds; i++) {
        // No new work. Keep polling.
        ggml_thread_cpu_relax();
    }

    ঘূৰাই state->pending;
}

স্থির অন্তঃস্থ bool ggml_graph_compute_check_for_work(সংরচনা ggml_compute_state * state) {
    সংরচনা ggml_threadpool * threadpool = state->threadpool;

    যদি (ggml_graph_compute_poll_for_work(state)) {
        ggml_graph_compute_thread_sync(state);
        ঘূৰাই state->pending;
    }

    ggml_mutex_lock_shared(&threadpool->mutex);
    যেতিয়ালৈ (!ggml_graph_compute_thread_ready(state)) {
        // No new work. Wait for the signal.
        GGML_PRINT_DEBUG("thread #%d waiting for work (sleeping)\n", state->ith);
        ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
    }
    ggml_mutex_unlock_shared(&threadpool->mutex);

    ঘূৰাই state->pending;
}

স্থির thread_ret_t ggml_graph_compute_secondary_thread(শূন্য* data) {
    সংরচনা ggml_compute_state * state = (সংরচনা ggml_compute_state *) data;
    সংরচনা ggml_threadpool * threadpool = state->threadpool;

    ggml_thread_apply_priority(threadpool->prio);
    যদি (ggml_thread_cpumask_is_valid(state->cpumask)) {
        ggml_thread_apply_affinity(state->cpumask);
    }

    যেতিয়ালৈ (true) {
        // Check if we need to sleep
        যেতিয়ালৈ (threadpool->pause) {
            GGML_PRINT_DEBUG("thread #%d inside pause loop\n", state->ith);
            ggml_mutex_lock_shared(&threadpool->mutex);
            যদি (threadpool->pause) {
                ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
            }
            GGML_PRINT_DEBUG("thread #%d resuming after wait\n", state->ith);
            ggml_mutex_unlock_shared(&threadpool->mutex);
        }

        // This needs to be checked for after the cond_wait
        যদি (threadpool->stop) ভঙ্গ;

        // Check if there is new work
        // The main thread is the only one that can dispatch new work

        ggml_graph_compute_check_for_work(state);
        যদি (state->pending) {
            state->pending = false;
            ggml_graph_compute_thread(state);
        }
    }

    ঘূৰাই (thread_ret_t) 0;
}

// Start processing new graph
স্থির শূন্য ggml_graph_compute_kickoff(সংরচনা ggml_threadpool * threadpool, পূর্ণাঙ্ক n_threads)
{
    // Always take the mutex here because the worker threads are doing hybrid poll/wait

    ggml_mutex_lock(&threadpool->mutex);

    // Update the number of active threads and the graph count
    পূর্ণাঙ্ক n_graph = atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed) >> GGML_THREADPOOL_N_THREADS_BITS;
    n_graph = ((n_graph + 1) << GGML_THREADPOOL_N_THREADS_BITS) | (n_threads & GGML_THREADPOOL_N_THREADS_MASK);

    GGML_PRINT_DEBUG("compute-kickoff: n_threads %d n_graph %d\n", n_threads, n_graph);

    // Indicate the graph is ready to be processed
    // We need the full seq-cst fence here because of the polling threads (used in thread_sync)
    atomic_store_explicit(&threadpool->n_graph, n_graph, memory_order_seq_cst);

    যদি (threadpool->pause) {
       // Update main thread prio and affinity to match the threadpool settings
       ggml_thread_apply_priority(threadpool->prio);
       যদি (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
           ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
       }

       // resume does cond broadcast
       ggml_threadpool_resume_locked(threadpool);
    } অন্যথা {
       ggml_cond_broadcast(&threadpool->cond);
    }

    ggml_mutex_unlock(&threadpool->mutex);
}

#endif // GGML_USE_OPENMP

স্থির সংরচনা ggml_threadpool * ggml_threadpool_new_impl(
    সংরচনা ggml_threadpool_params * tpp,
               সংরচনা ggml_cgraph * cgraph,
                সংরচনা ggml_cplan * cplan) {

    সংরচনা ggml_threadpool * threadpool =
        ggml_aligned_malloc(পরিমাণ(সংরচনা ggml_threadpool));
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
        threadpool->workers          = রিক্ত;
        threadpool->n_threads        = tpp->n_threads;
        threadpool->poll             = tpp->poll;
        threadpool->prio             = tpp->prio;
        threadpool->ec               = GGML_STATUS_SUCCESS;
    }

    // Allocate and init workers state
    ধ্রুৱ size_t workers_size = পরিমাণ(সংরচনা ggml_compute_state) * tpp->n_threads;
    সংরচনা ggml_compute_state * workers = ggml_aligned_malloc(workers_size);

    memset(workers, 0, workers_size);
    বাবে (পূর্ণাঙ্ক j = 0; j < tpp->n_threads; j++) {
        workers[j].threadpool = threadpool;
        workers[j].ith        = j;
    }

    threadpool->workers = workers;

#ifdef GGML_USE_OPENMP
    int32_t cpumask_iter = 0;

    // Compute CPU masks for each thread
    বাবে (পূর্ণাঙ্ক j = 0; j < tpp->n_threads; j++) {
        ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);
    }
#অন্যথা // GGML_USE_OPENMP
    ggml_mutex_init(&threadpool->mutex);
    ggml_cond_init(&threadpool->cond);

    // Spin the threads for all workers, and update CPU placements.
    // Place the main thread last (towards the higher numbered CPU cores).

    int32_t cpumask_iter = 0;

    বাবে (পূর্ণাঙ্ক j = 1; j < tpp->n_threads; j++) {
        ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);

        int32_t rc = ggml_thread_create(&workers[j].thrd, রিক্ত, ggml_graph_compute_secondary_thread, &workers[j]);
        GGML_ASSERT(rc == 0);
    }

    ggml_thread_cpumask_next(tpp->cpumask, workers[0].cpumask, tpp->strict_cpu, &cpumask_iter);

    যদি (!threadpool->pause) {
        // Update main thread prio and affinity at the start, otherwise we'll do it in resume
        ggml_thread_apply_priority(threadpool->prio);
        যদি (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
            ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
        }
    }
#endif // GGML_USE_OPENMP

    ঘূৰাই threadpool;
}

সংরচনা ggml_threadpool * ggml_threadpool_new(সংরচনা ggml_threadpool_params * tpp) {
    ঘূৰাই ggml_threadpool_new_impl(tpp, রিক্ত, রিক্ত);
}

গণনা ggml_status ggml_graph_compute(সংরচনা ggml_cgraph * cgraph, সংরচনা ggml_cplan * cplan) {
    ggml_cpu_init();

    GGML_ASSERT(cplan);
    GGML_ASSERT(cplan->n_threads > 0);
    GGML_ASSERT(cplan->work_size == 0 || cplan->work_data != রিক্ত);

    পূর্ণাঙ্ক n_threads                               = cplan->n_threads;
    সংরচনা ggml_threadpool * threadpool = cplan->threadpool;

    bool disposable_threadpool = false;

    যদি (threadpool == রিক্ত) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
        disposable_threadpool = true;

        সংরচনা ggml_threadpool_params ttp = ggml_threadpool_params_default(n_threads);
        threadpool = ggml_threadpool_new_impl(&ttp, cgraph, cplan);
    } অন্যথা {
        // Reset some of the parameters that need resetting
        // No worker threads should be accessing the parameters below at this stage
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->current_chunk    = 0;
        threadpool->abort            = -1;
        threadpool->ec               = GGML_STATUS_SUCCESS;
    }

#ifdef GGML_USE_OPENMP
    যদি (n_threads > 1) {
        #pragma omp parallel অঙ্ক_threads(n_threads)
        {
            #pragma omp single
            {
                // update the number of threads from the actual number of threads that we got from OpenMP
                n_threads = omp_get_অঙ্ক_threads();
                atomic_store_explicit(&threadpool->n_graph, n_threads, memory_order_relaxed);
            }

            // Apply thread CPU mask and priority
            পূর্ণাঙ্ক ith = omp_get_thread_অঙ্ক();

            ggml_thread_apply_priority(threadpool->prio);
            যদি (ggml_thread_cpumask_is_valid(threadpool->workers[ith].cpumask)) {
                ggml_thread_apply_affinity(threadpool->workers[ith].cpumask);
            }
            ggml_graph_compute_thread(&threadpool->workers[ith]);
        }
    } অন্যথা {
        atomic_store_explicit(&threadpool->n_graph, 1, memory_order_relaxed);
        ggml_graph_compute_thread(&threadpool->workers[0]);
    }
#অন্যথা
    যদি (n_threads > threadpool->n_threads) {
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

    গণনা ggml_status ret = threadpool->ec;

    যদি (disposable_threadpool) {
        ggml_threadpool_free(threadpool);
    }

    ঘূৰাই ret;
}

গণনা ggml_status ggml_graph_compute_with_ctx(সংরচনা ggml_context * ctx, সংরচনা ggml_cgraph * cgraph, পূর্ণাঙ্ক n_threads) {
    সংরচনা ggml_cplan cplan = ggml_graph_plan(cgraph, n_threads, রিক্ত);

    cplan.work_data = (uint8_t *)ggml_new_buffer(ctx, cplan.work_size);

    ঘূৰাই ggml_graph_compute(cgraph, &cplan);
}

শূন্য ggml_cpu_fp32_to_fp32(ধ্রুৱ ভিন্ন * x, ভিন্ন * y, int64_t n) {
    memcpy(y, x, n * পরিমাণ(ভিন্ন));
}

শূন্য ggml_cpu_fp32_to_fp16(ধ্রুৱ ভিন্ন * x, ggml_fp16_t * y, int64_t n) {
    int64_t i = 0;
#যদি defined(__F16C__)
#যদি defined(__AVX512F__)
    বাবে (; i + 15 < n; i += 16) {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        __m256i y_vec = _mm512_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256((__m256i *)(y + i), y_vec);
    }
#endif
    বাবে (; i + 7 < n; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m128i y_vec = _mm256_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i *)(y + i), y_vec);
    }
    বাবে (; i + 3 < n; i += 4) {
        __m128 x_vec = _mm_loadu_ps(x + i);
        __m128i y_vec = _mm_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storel_epi64((__m128i *)(y + i), y_vec);
    }
#নোচেদ্যদি defined(__riscv_zvfh)
    বাবে (পূর্ণাঙ্ক vl; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(&x[i], vl);
        vfloat16m1_t vy = __riscv_vfncvt_f_f_w_f16m1(vx, vl);
        __riscv_vse16_v_f16m1((_Float16 *)&y[i], vy, vl);
    }
#নোচেদ্যদি defined(__VXE__) || defined(__VXE2__)
    বাবে (; i + 7 < n; i += 8) {
        ধ্রুৱ uint32x4_t v_yl = __lzs_f32cx4_to_f16(vec_xl(0, x + i + 0));
        ধ্রুৱ uint32x4_t v_yh = __lzs_f32cx4_to_f16(vec_xl(0, x + i + 4));
        vec_xst(vec_pack(v_yl, v_yh), 0, (uint16_t *)(y + i));
    }
#endif
    বাবে (; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(x[i]);
    }
}

শূন্য ggml_cpu_fp16_to_fp32(ধ্রুৱ ggml_fp16_t * x, ভিন্ন * y, int64_t n) {
    int64_t i = 0;
#যদি defined(__F16C__)
#যদি defined(__AVX512F__)
    বাবে (; i + 15 < n; i += 16) {
        __m256i x_vec = _mm256_loadu_si256((ধ্রুৱ __m256i *)(x + i));
        __m512 y_vec = _mm512_cvtph_ps(x_vec);
        _mm512_storeu_ps(y + i, y_vec);
    }
#endif
    বাবে (; i + 7 < n; i += 8) {
        __m128i x_vec = _mm_loadu_si128((ধ্রুৱ __m128i *)(x + i));
        __m256 y_vec = _mm256_cvtph_ps(x_vec);
        _mm256_storeu_ps(y + i, y_vec);
    }
    বাবে (; i + 3 < n; i += 4) {
        __m128i x_vec = _mm_loadl_epi64((ধ্রুৱ __m128i *)(x + i));
        __m128 y_vec = _mm_cvtph_ps(x_vec);
        _mm_storeu_ps(y + i, y_vec);
    }

#নোচেদ্যদি defined(__riscv_v_intrinsic) && defined(__riscv_zvfhmin)
    // calculate step size
    ধ্রুৱ পূর্ণাঙ্ক epr = __riscv_vsetvlmax_e16m2();
    ধ্রুৱ পূর্ণাঙ্ক step = epr * 2;
    ধ্রুৱ পূর্ণাঙ্ক np = (n & ~(step - 1));

    // unroll by 2
    বাবে (; i < np; i += step) {
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((ধ্রুৱ _Float16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vfloat16m2_t ax1 = __riscv_vle16_v_f16m2((ধ্রুৱ _Float16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvt_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    পূর্ণাঙ্ক vl;
    বাবে (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((ধ্রুৱ _Float16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }

#endif

    বাবে (; i < n; ++i) {
        y[i] = GGML_CPU_FP16_TO_FP32(x[i]);
    }
}

শূন্য ggml_cpu_fp32_to_bf16(ধ্রুৱ ভিন্ন * x, ggml_bf16_t * y, int64_t n) {
    int64_t i = 0;
    বাবে (; i < n; ++i) {
        y[i] = GGML_FP32_TO_BF16(x[i]);
    }
}

শূন্য ggml_cpu_fp32_to_i32(ধ্রুৱ ভিন্ন * x, int32_t * y, int64_t n) {
    int64_t i = 0;
    বাবে (; i < n; ++i) {
        y[i] = x[i];
    }
}

শূন্য ggml_cpu_bf16_to_fp32(ধ্রুৱ ggml_bf16_t * x, ভিন্ন * y, int64_t n) {
    int64_t i = 0;
#যদি defined(__AVX2__)
#যদি defined(__AVX512F__)
    বাবে (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i,
                        _mm512_castsi512_ps(
                            _mm512_slli_epi32(
                                _mm512_cvtepu16_epi32(
                                    _mm256_loadu_si256(
                                        (ধ্রুৱ __m256i *)(x + i))),
                                16)));
    }
#endif
    বাবে (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i,
                        _mm256_castsi256_ps(
                            _mm256_slli_epi32(
                                _mm256_cvtepu16_epi32(
                                    _mm_loadu_si128(
                                        (ধ্রুৱ __m128i *)(x + i))),
                                16)));
    }
#নোচেদ্যদি defined(__riscv_v_intrinsic) && defined(__riscv_zvfbfmin)
    // calculate step size
    ধ্রুৱ পূর্ণাঙ্ক epr = __riscv_vsetvlmax_e16m2();
    ধ্রুৱ পূর্ণাঙ্ক step = epr * 2;
    ধ্রুৱ পূর্ণাঙ্ক np = (n & ~(step - 1));

    // unroll by 2
    বাবে (; i < np; i += step) {
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((ধ্রুৱ __bf16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vbfloat16m2_t ax1 = __riscv_vle16_v_bf16m2((ধ্রুৱ __bf16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    পূর্ণাঙ্ক vl;
    বাবে (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((ধ্রুৱ __bf16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }
#endif
    বাবে (; i < n; i++) {
        y[i] = GGML_BF16_TO_FP32(x[i]);
    }
}

পূর্ণাঙ্ক ggml_cpu_has_avx(শূন্য) {
#যদি defined(__AVX__)
    ঘূৰাই 1;
#অন্যথা
    ঘূৰাই 0;
#endif
}

পূর্ণাঙ্ক ggml_cpu_has_avx_vnni(শূন্য) {
#যদি defined(__AVXVNNI__)
    ঘূৰাই 1;
#অন্যথা
    ঘূৰাই 0;
#endif
}

পূর্ণাঙ্ক ggml_cpu_has_avx2(শূন্য) {
#যদি defined(__AVX2__)
    ঘূৰাই 1;
#অন্যথা
    ঘূৰাই 0;
#endif
}

পূর্ণাঙ্ক ggml_cpu_has_avx512(শূন্য) {
#যদি defined(__AVX512F__)
    ঘূৰাই 1;
#অন্যথা
    ঘূৰাই 0;
#endif
}

পূর্ণাঙ্ক ggml_cpu_has_avx512_vbmi(শূন্য) {
#যদি defined(__AVX512VBMI__)
    ঘূৰাই 1;
#অন্যথা
    ঘূৰাই 0;
#endif
}

পূর্ণাঙ্ক ggml_cpu_has_avx512_vnni(শূন্য) {
#যদি defined(__AVX512VNNI__)
    ঘূৰাই 1;
#অন্যথা
    ঘূৰাই 0;
#endif
}

পূর্ণাঙ্ক ggml_cpu_has_avx512_bf16(শূন্য) {
#যদি defined(__AVX512BF16__)
    ঘূৰাই 1;
#অন্যথা
    ঘূৰাই 0;
#endif
}

পূর্ণাঙ্ক ggml_cpu_has_amx_int8(শূন্য) {
#যদি defined(__AMX_INT8__)
    ঘূৰাই 1;
#অন্যথা
    ঘূৰাই 0;
#endif
}

পূর্ণাঙ্ক ggml_cpu_has_bmi2(শূন্য) {
#যদি defined(__BMI2__)
    ঘূৰাই 1;
#অন্যথা
    ঘূৰাই 0;
#endif
}

পূর্ণাঙ্ক ggml_cpu_has_fma(শূন্য) {
#যদি defined(__FMA__)
    ঘূৰাই 1;
#অন্যথা
    ঘূৰাই 0;
#endif
}

পূর্ণাঙ্ক ggml_cpu_has_arm_fma(শূন্য) {
#যদি defined(__ARM_FEATURE_FMA)
    ঘূৰাই 1;
#অন্যথা
    ঘূৰাই 0;
#endif
}

পূর্ণাঙ্ক ggml_cpu_has_riscv_v(শূন্য) {
#যদি defined(__riscv_v_intrinsic)
    ঘূৰাই 1;
#অন্যথা
    ঘূৰাই 0;
#endif
}

পূর্ণাঙ্ক ggml_cpu_get_rvv_vlen(শূন্য) {
#যদি defined(__riscv) && defined(__riscv_v_intrinsic)
    ঘূৰাই ggml_riscv_arch_features.rvv_vlen;
#অন্যথা
    ঘূৰাই 0;
#endif
}

পূর্ণাঙ্ক ggml_cpu_has_f16c(শূন্য) {
#যদি defined(__F16C__)
    ঘূৰাই 1;
#অন্যথা
    ঘূৰাই 0;
#endif
}

পূর্ণাঙ্ক ggml_cpu_has_fp16_va(শূন্য) {
#যদি defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    ঘূৰাই 1;
#অন্যথা
    ঘূৰাই 0;
#endif
}

পূর্ণাঙ্ক ggml_cpu_has_wasm_simd(শূন্য) {
#যদি defined(__wasm_simd128__)
    ঘূৰাই 1;
#অন্যথা
    ঘূৰাই 0;
#endif
}

পূর্ণাঙ্ক ggml_cpu_has_llamafile(শূন্য) {
#যদি defined(GGML_USE_LLAMAFILE)
    ঘূৰাই 1;
#অন্যথা
    ঘূৰাই 0;
#endif
}

পূর্ণাঙ্ক ggml_cpu_has_sse3(শূন্য) {
#যদি defined(__SSE3__)
    ঘূৰাই 1;
#অন্যথা
    ঘূৰাই 0;
#endif
}

পূর্ণাঙ্ক ggml_cpu_has_ssse3(শূন্য) {
#যদি defined(__SSSE3__)
    ঘূৰাই 1;
#অন্যথা
    ঘূৰাই 0;
#endif
}

পূর্ণাঙ্ক ggml_cpu_has_vsx(শূন্য) {
#যদি defined(__POWER9_VECTOR__)
    ঘূৰাই 1;
#অন্যথা
    ঘূৰাই 0;
#endif
}

পূর্ণাঙ্ক ggml_cpu_has_vxe(শূন্য) {
#যদি defined(__VXE__) || defined(__VXE2__)
    ঘূৰাই 1;
#অন্যথা
    ঘূৰাই 0;
#endif
}

পূর্ণাঙ্ক ggml_cpu_has_neon(শূন্য) {
#যদি defined(__ARM_ARCH) && defined(__ARM_NEON)
    ঘূৰাই 1;
#অন্যথা
    ঘূৰাই 0;
#endif
}

পূর্ণাঙ্ক ggml_cpu_has_dotprod(শূন্য) {
#যদি defined(__ARM_ARCH) && defined(__ARM_FEATURE_DOTPROD)
    ঘূৰাই 1;
#অন্যথা
    ঘূৰাই 0;
#endif
}

পূর্ণাঙ্ক ggml_cpu_has_sve(শূন্য) {
#যদি defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    ঘূৰাই 1;
#অন্যথা
    ঘূৰাই 0;
#endif
}

পূর্ণাঙ্ক ggml_cpu_has_matmul_int8(শূন্য) {
#যদি defined(__ARM_ARCH) && defined(__ARM_FEATURE_MATMUL_INT8)
    ঘূৰাই 1;
#অন্যথা
    ঘূৰাই 0;
#endif
}

পূর্ণাঙ্ক ggml_cpu_get_sve_cnt(শূন্য) {
#যদি defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    ঘূৰাই ggml_arm_arch_features.sve_cnt;
#অন্যথা
    ঘূৰাই 0;
#endif
}

পূর্ণাঙ্ক ggml_cpu_has_sme(শূন্য) {
#যদি defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME)
    ঘূৰাই 1;
#অন্যথা
    ঘূৰাই 0;
#endif
}

পূর্ণাঙ্ক ggml_cpu_has_sme2(শূন্য) {
#যদি defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME2)
    ঘূৰাই 1;
#অন্যথা
    ঘূৰাই 0;
#endif
}

শূন্য ggml_cpu_init(শূন্য) {
    // needed to initialize ggml_time
    {
        সংরচনা ggml_init_params params = { 0, রিক্ত, false };
        সংরচনা ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    ggml_critical_section_start();

    স্থির bool is_first_call = true;

    যদি (is_first_call) {
        // initialize GELU, Quick GELU, SILU and EXP F32 tables
        {
            ধ্রুৱ uint64_t t_start = ggml_time_us(); UNUSED(t_start);

            বাবে (পূর্ণাঙ্ক i = 0; i < (1 << 16); ++i) {
                সঙ্ঘ {
                    uint16_t u16;
                    ggml_fp16_t fp16;
                } u = {i};
                ভিন্ন f = GGML_COMPUTE_FP16_TO_FP32(u.fp16);
                ggml_table_f32_f16[i] = f;
                ggml_table_gelu_f16[i] = GGML_CPU_FP32_TO_FP16(ggml_gelu_f32(f));
                ggml_table_gelu_quick_f16[i] = GGML_CPU_FP32_TO_FP16(ggml_gelu_quick_f32(f));
            }

            // initialize E8M0 half table (256 entries)
            বাবে (পূর্ণাঙ্ক i = 0; i < (1 << 8); ++i) {
                ggml_table_f32_e8m0_half[i] = GGML_E8M0_TO_FP32_HALF(i);
            }

            // initialize UE4M3 table (256 entries)
            বাবে (পূর্ণাঙ্ক i = 0; i < (1 << 8); ++i) {
                ggml_table_f32_ue4m3[i] = ggml_ue4m3_to_fp32(i);
            }

            ধ্রুৱ uint64_t t_end = ggml_time_us(); UNUSED(t_end);

            GGML_PRINT_DEBUG("%s: GELU, Quick GELU, SILU and EXP tables initialized in %f ms\n", __func__, (t_end - t_start)/1000.0);

#ifdef GGML_USE_OPENMP
            //if (!getenv("OMP_WAIT_POLICY")) {
            //    // set the wait policy to active, so that OpenMP threads don't sleep
            //    setenv("OMP_WAIT_POLICY", "active", 0)
            //}

            যদি (!getenv("KMP_BLOCKTIME")) {
                // set the time to wait before sleeping a thread
                // this is less aggressive than setting the wait policy to active, but should achieve similar results in most cases
#ifdef _WIN32
                _putenv_s("KMP_BLOCKTIME", "200"); // 200ms
#অন্যথা
                setenv("KMP_BLOCKTIME", "200", 0); // 200ms
#endif
            }
#endif
        }

#যদি defined(__ARM_ARCH)
        ggml_init_arm_arch_features();
#endif

#যদি defined(__riscv)
        ggml_init_riscv_arch_features();
#endif

        {
            ধ্রুৱ ৱর্ণ * env = getenv("GGML_CPU_DISABLE_FUSION");
            ggml_cpu_disable_fusion = (env != রিক্ত && atoi(env) == 1);
        }

        is_first_call = false;
    }

    ggml_critical_section_end();
}
