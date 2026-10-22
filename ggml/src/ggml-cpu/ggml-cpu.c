#define _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#define _USE_MATH_DEFINES // For M_PI on MSVC

#شامل "ggml-backend-impl.h"
#شامل "ggml-backend.h"
#شامل "traits.h"
#شامل "ggml-cpu-impl.h"
#شامل "ggml-impl.h"
#شامل "quants.h"
#شامل "ggml-threading.h"
#شامل "unary-ops.h"
#شامل "binary-ops.h"
#شامل "vec.h"
#شامل "ops.h"
#شامل "ggml.h"
#شامل "common.h"
#شامل "tiled/tiled.h"

#اگر defined(_MSC_VER) || defined(__MINGW32__)
#شامل <malloc.h> // using malloc.h with MSC/MINGW
#وگرنهاگر !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#شامل <alloca.h>
#endif

#شامل <assert.h>
#شامل <errno.h>
#شامل <time.h>
#شامل <math.h>
#شامل <stdlib.h>
#شامل <string.h>
#شامل <stdint.h>
#شامل <inttypes.h>
#شامل <stdio.h>
#شامل <float.h>
#شامل <limits.h>
#شامل <stdarg.h>
#شامل <signal.h>
#اگر defined(__gnu_linux__)
#شامل <syscall.h>
#endif

#ifdef GGML_USE_OPENMP
#شامل <omp.h>
#endif

#اگر defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_MATMUL_INT8)
#undef GGML_USE_LLAMAFILE
#endif

#ifdef GGML_USE_LLAMAFILE
#شامل "llamafile/sgemm.h"
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
#    include "spacemit/ime.h"
#endif

// Note: once we move threading into a separate C++ file
// will use std::hardware_destructive_interference_size instead of hardcoding it here
// and we'll use C++ attribute syntax.
#define GGML_CACHE_سطر  64

#اگر defined(__clang__) || defined(__GNUC__)
#define GGML_CACHE_ALIGN __attribute__((aligned(GGML_CACHE_سطر)))
#endif

#اگر defined(__has_feature)
#اگر __has_feature(thread_sanitizer)
#define GGML_TSAN_ENABLED 1
#endif
#وگرنه  // __has_feature
#اگر defined(__SANITIZE_THREAD__)
#define GGML_TSAN_ENABLED 1
#endif
#endif // __has_feature

#define UNUSED GGML_UNUSED
#define SWAP(x, y, T) انجامبده { T SWAP = x; (x) = y; (y) = SWAP; } وقتی (0)

// precomputed f32 table for f16 (256 KB) (simd-mappings.h)
اعشاری ggml_table_f32_f16[1 << 16];

// precomputed f32 table for e8m0 half (1 KB) (simd-mappings.h)
اعشاری ggml_table_f32_e8m0_half[1 << 8];

// precomputed f32 table for ue4m3 (1 KB) (simd-mappings.h)
اعشاری ggml_table_f32_ue4m3[1 << 8];

#اگر defined(__ARM_ARCH)
ساختار ggml_arm_arch_features_type {
    صحیح sve_cnt;
} ggml_arm_arch_features = { 0 };
#endif

#اگر defined(__riscv)
ساختار ggml_riscv_arch_features_type {
    صحیح rvv_vlen;
} ggml_riscv_arch_features = { 0 };
#endif

#اگر defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#شامل <windows.h>

#اگر defined(_MSC_VER) && !defined(__clang__)
#define GGML_CACHE_ALIGN __declspec(align(GGML_CACHE_سطر))

تعریفنوع فرار LONG atomic_int;
تعریفنوع atomic_int atomic_bool;
تعریفنوع atomic_int atomic_flag;

#define ATOMIC_FLAG_INIT 0

تعریفنوع شمارش {
    memory_order_relaxed,
    memory_order_consume,
    memory_order_acquire,
    memory_order_release,
    memory_order_acq_rel,
    memory_order_seq_cst
} memory_order;

ایستا تهی atomic_store(atomic_int * ptr, LONG val) {
    InterlockedExchange(ptr, val);
}
ایستا تهی atomic_store_explicit(atomic_int * ptr, LONG val, memory_order mo) {
    // TODO: add support for explicit memory order
    InterlockedExchange(ptr, val);
}
ایستا LONG atomic_load(atomic_int * ptr) {
    بازگشت InterlockedCompareExchange(ptr, 0, 0);
}
ایستا LONG atomic_load_explicit(atomic_int * ptr, memory_order mo) {
    // TODO: add support for explicit memory order
    بازگشت InterlockedCompareExchange(ptr, 0, 0);
}
ایستا LONG atomic_fetch_add(atomic_int * ptr, LONG inc) {
    بازگشت InterlockedExchangeAdd(ptr, inc);
}
ایستا LONG atomic_fetch_add_explicit(atomic_int * ptr, LONG inc, memory_order mo) {
    // TODO: add support for explicit memory order
    بازگشت InterlockedExchangeAdd(ptr, inc);
}
ایستا atomic_bool atomic_flag_test_and_set(atomic_flag * ptr) {
    بازگشت InterlockedExchange(ptr, 1);
}
ایستا تهی atomic_flag_clear(atomic_flag * ptr) {
    InterlockedExchange(ptr, 0);
}
ایستا تهی atomic_thread_fence(memory_order mo) {
    MemoryBarrier();
}
#وگرنه // clang
#شامل <stdatomic.h>
#endif

تعریفنوع HANDLE pthread_t;

تعریفنوع DWORD thread_ret_t;
ایستا صحیح pthread_create(pthread_t * out, تهی * unused, thread_ret_t(*func)(تهی *), تهی * arg) {
    (تهی) unused;
    HANDLE handle = CreateThread(پوچ, 0, (LPTHREAD_START_ROUTINE) func, arg, 0, پوچ);
    اگر (handle == پوچ)
    {
        بازگشت EAGAIN;
    }

    *out = handle;
    بازگشت 0;
}

ایستا صحیح pthread_join(pthread_t thread, تهی * unused) {
    (تهی) unused;
    صحیح ret = (صحیح) WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    بازگشت ret;
}

ایستا صحیح sched_yield (تهی) {
    Sleep (0);
    بازگشت 0;
}
#وگرنه

#شامل <pthread.h>
#شامل <stdatomic.h>
#شامل <sched.h>
#اگر defined(__FreeBSD__)
#شامل <pthread_np.h>
#endif

تعریفنوع تهی * thread_ret_t;

#شامل <sys/types.h>
#شامل <sys/stat.h>
#شامل <unistd.h>

#endif

تعریفنوع pthread_t ggml_thread_t;

#define GGML_THREADPOOL_N_THREADS_MASK (0xffffU)
#define GGML_THREADPOOL_N_THREADS_BITS (16)

#اگر defined(__APPLE__)
#شامل <unistd.h>
#شامل <mach/mach.h>
#شامل <TargetConditionals.h>
#endif

ایستا ثابت ساختار ggml_type_traits_cpu type_traits_cpu[GGML_TYPE_شمار] = {
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
#اگر defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#وگرنه
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q4_1] = {
        .from_float               = quantize_row_q4_1,
        .vec_dot                  = ggml_vec_dot_q4_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
#اگر defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#وگرنه
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
#اگر defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#وگرنه
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
#اگر defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#وگرنه
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
#اگر defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#وگرنه
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_IQ2_XXS] = {
        .from_float               = پوچ,
        .vec_dot                  = ggml_vec_dot_iq2_xxs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_XS] = {
        .from_float               = پوچ,
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
        .from_float               = پوچ,
        .vec_dot                  = ggml_vec_dot_iq1_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_M] = {
        .from_float               = پوچ,
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

ثابت ساختار ggml_type_traits_cpu * ggml_get_type_traits_cpu(شمارش ggml_type type) {
    بازگشت &type_traits_cpu[type];
}

//
// Threading defs
//

تعریفنوع pthread_t          ggml_thread_t;

#اگر defined(_WIN32)

تعریفنوع CONDITION_VARIABLE ggml_cond_t;
تعریفنوع SRWLOCK            ggml_mutex_t;

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

#وگرنه

تعریفنوع pthread_cond_t     ggml_cond_t;
تعریفنوع pthread_mutex_t    ggml_mutex_t;

#define ggml_mutex_init(m)          pthread_mutex_init(m, پوچ)
#define ggml_mutex_destroy(m)       pthread_mutex_destroy(m)
#define ggml_mutex_lock(m)          pthread_mutex_lock(m)
#define ggml_mutex_unlock(m)        pthread_mutex_unlock(m)
#define ggml_mutex_lock_shared(m)   pthread_mutex_lock(m)
#define ggml_mutex_unlock_shared(m) pthread_mutex_unlock(m)

#define ggml_lock_init(x)    UNUSED(x)
#define ggml_lock_destroy(x) UNUSED(x)
#اگر defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64) && !defined(_M_ARM64EC))
#define ggml_lock_lock(x)    _mm_pause()
#وگرنه
#define ggml_lock_lock(x)    UNUSED(x)
#endif
#define ggml_lock_unlock(x)  UNUSED(x)

#define GGML_LOCK_INITIALIZER 0
#define ggml_cond_init(c)      pthread_cond_init(c, پوچ)
#define ggml_cond_destroy(c)   pthread_cond_destroy(c)
#define ggml_cond_wait(c, m)   pthread_cond_wait(c, m)
#define ggml_cond_broadcast(c) pthread_cond_broadcast(c)

#define ggml_thread_create pthread_create
#define ggml_thread_join   pthread_join

#endif

// Threadpool def
ساختار ggml_threadpool {
    ggml_mutex_t mutex;       // mutex for cond.var
    ggml_cond_t  cond;        // cond.var for waiting for new work

    ساختار ggml_cgraph * cgraph;
    ساختار ggml_cplan  * cplan;

    // synchronization primitives
    atomic_int n_graph;       // updated when there is work to be done (i.e each graph) holds graph and active thread counts.
    atomic_int GGML_CACHE_ALIGN n_barrier;
    atomic_int GGML_CACHE_ALIGN n_barrier_passed;
    atomic_int GGML_CACHE_ALIGN current_chunk; // currently processing chunk during Mat_Mul, shared between all the threads.

    // these are atomic as an annotation for thread-sanitizer
    atomic_bool stop;         // Used for stopping the threadpool altogether
    atomic_bool pause;        // Used for pausing the threadpool or individual threads
    atomic_int  abort;        // Used for aborting processing of a graph

    ساختار ggml_compute_state * workers;   // per thread state
    صحیح          n_threads;   // Number of threads in the pool
    int32_t      prio;        // Scheduling priority
    uint32_t     poll;        // Polling level (0 - no polling)

    شمارش ggml_status ec;
};

// Per-thread state
ساختار ggml_compute_state {
#ifndef GGML_USE_OPENMP
    ggml_thread_t thrd;
    صحیح  last_graph;
    bool pending;
#endif
    bool cpumask[GGML_MAX_N_THREADS];
    ساختار ggml_threadpool * threadpool;
    صحیح ith;
};

// Helpers for polling loops
#اگر defined(__aarch64__) && ( defined(__clang__) || defined(__GNUC__) )
ایستا درونخطی تهی ggml_thread_cpu_relax(تهی) {
    __asm__ فرار("yield" ::: "memory");
}
#وگرنهاگر defined(__x86_64__)
ایستا درونخطی تهی ggml_thread_cpu_relax(تهی) {
    _mm_pause();
}
#وگرنهاگر defined(__riscv)
ایستا درونخطی تهی ggml_thread_cpu_relax(تهی) {
    #ifdef __riscv_zihintpause
        __asm__ __volatile__ ("pause");
    #وگرنه
        /* Encoding of the pause instruction */
        __asm__ __volatile__ (".4byte 0x100000F");
    #endif
}
#وگرنه
ایستا درونخطی تهی ggml_thread_cpu_relax(تهی) {;}
#endif

//
// NUMA support
//

#define GGML_NUMA_MAX_NODES 8
#define GGML_NUMA_MAX_CPUS 512

ساختار ggml_numa_گره {
    uint32_t cpus[GGML_NUMA_MAX_CPUS]; // hardware threads on this node
    uint32_t n_cpus;
};

ساختار ggml_numa_nodes {
    شمارش ggml_numa_strategy numa_strategy;
    ساختار ggml_numa_گره nodes[GGML_NUMA_MAX_NODES];
    uint32_t n_nodes;
    uint32_t total_cpus; // hardware threads on system
    uint32_t current_گره; // node on which main process is execting
#اگر defined(__gnu_linux__)
    cpu_set_t cpuset; // cpuset from numactl
#وگرنه
    uint32_t cpuset; // no NUMA support outside of Linux at this time. Use a portable datatype
#endif
};

//
// ggml state
//

ساختار ggml_state {
    ساختار ggml_numa_nodes numa;
};

ایستا ساختار ggml_state g_state = {0};

تهی ggml_barrier(ساختار ggml_threadpool * tp) {
    صحیح n_threads = atomic_load_explicit(&tp->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK;
    اگر (n_threads == 1) {
        بازگشت;
    }

#ifdef GGML_USE_OPENMP
    #pragma omp barrier
#وگرنه
    صحیح n_passed = atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed);

    // enter barrier (full seq-cst fence)
    صحیح n_barrier = atomic_fetch_add_explicit(&tp->n_barrier, 1, memory_order_seq_cst);

    اگر (n_barrier == (n_threads - 1)) {
        // last thread
        atomic_store_explicit(&tp->n_barrier, 0, memory_order_relaxed);

        // exit barrier (full seq-cst fence)
        atomic_fetch_add_explicit(&tp->n_barrier_passed, 1, memory_order_seq_cst);
        بازگشت;
    }

    // wait for other threads
    وقتی (atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed) == n_passed) {
        ggml_thread_cpu_relax();
    }

    // exit barrier (full seq-cst fence)
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&tp->n_barrier_passed, 0, memory_order_seq_cst);
    #وگرنه
    atomic_thread_fence(memory_order_seq_cst);
    #endif
#endif
}

تهی ggml_threadpool_chunk_set(ساختار ggml_threadpool * tp, صحیح مقدار) {
    atomic_store_explicit(&tp->current_chunk, مقدار, memory_order_relaxed);
}

صحیح ggml_threadpool_chunk_add(ساختار ggml_threadpool * tp, صحیح مقدار) {
    بازگشت atomic_fetch_add_explicit(&tp->current_chunk, مقدار, memory_order_relaxed);
}

#اگر defined(__gnu_linux__)
ایستا cpu_set_t ggml_get_numa_affinity(تهی) {
    cpu_set_t cpuset;
    pthread_t thread;
    thread = pthread_self();
    CPU_ZERO(&cpuset);
    pthread_getaffinity_np(thread, اندازه(cpu_set_t), &cpuset);
    بازگشت cpuset;
}
#وگرنه
ایستا uint32_t ggml_get_numa_affinity(تهی) {
    بازگشت 0; // no NUMA support
}
#endif

تهی ggml_numa_init(شمارش ggml_numa_strategy numa_flag) {
    اگر (g_state.numa.n_nodes > 0) {
        fprintf(stderr, "ggml_numa_init: NUMA already initialized\n");

        بازگشت;
    }

#اگر defined(__gnu_linux__)
    ساختار stat st;
    نویسه path[256];
    صحیح rv;

    // set numa scheme
    g_state.numa.numa_strategy = numa_flag;

    GGML_PRINT_DEBUG("numa strategy %u\n",g_state.numa.numa_strategy);

    g_state.numa.cpuset = ggml_get_numa_affinity();

    // enumerate nodes
    وقتی (g_state.numa.n_nodes < GGML_NUMA_MAX_NODES) {
        rv = snprintf(path, اندازه(path), "/sys/devices/system/node/node%u", g_state.numa.n_nodes);
        GGML_ASSERT(rv > 0 && (unsigned)rv < اندازه(path));
        اگر (stat(path, &st) != 0) { بشکن; }
        ++g_state.numa.n_nodes;
    }

    // enumerate CPUs
    وقتی (g_state.numa.total_cpus < GGML_NUMA_MAX_CPUS) {
        rv = snprintf(path, اندازه(path), "/sys/devices/system/cpu/cpu%u", g_state.numa.total_cpus);
        GGML_ASSERT(rv > 0 && (unsigned)rv < اندازه(path));
        اگر (stat(path, &st) != 0) { بشکن; }
        ++g_state.numa.total_cpus;
    }

    GGML_PRINT_DEBUG("found %u numa nodes, %u CPUs\n", g_state.numa.n_nodes, g_state.numa.total_cpus);

    // figure out which node we're on
    uint current_cpu;
    صحیح getcpu_ret = 0;
#اگر __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ > 33) || defined(__COSMOPOLITAN__)
    getcpu_ret = getcpu(&current_cpu, &g_state.numa.current_گره);
#وگرنه
    // old glibc doesn't have a wrapper for this call. Fall back on direct syscall
#   اگر !defined(SYS_getcpu) && defined(SYS_get_cpu)
#       define SYS_getcpu SYS_get_cpu // some older glibc versions use this name
#   endif
    getcpu_ret = syscall(SYS_getcpu, &current_cpu, &g_state.numa.current_گره);
#endif

    اگر (g_state.numa.n_nodes < 1 || g_state.numa.total_cpus < 1 || getcpu_ret != 0) {
        g_state.numa.n_nodes = 0;
        بازگشت;
    }

    GGML_PRINT_DEBUG("found our process on numa node %u, CPU %u\n", g_state.numa.current_گره, current_cpu);

    برای (uint32_t n = 0; n < g_state.numa.n_nodes; ++n) {
        ساختار ggml_numa_گره * گره = &g_state.numa.nodes[n];
        GGML_PRINT_DEBUG("CPUs on node %u:", n);
        گره->n_cpus = 0;
        برای (uint32_t c = 0; c < g_state.numa.total_cpus; ++c) {
            rv = snprintf(path, اندازه(path), "/sys/devices/system/node/node%u/cpu%u", n, c);
            GGML_ASSERT(rv > 0 && (unsigned)rv < اندازه(path));
            اگر (stat(path, &st) == 0) {
                گره->cpus[گره->n_cpus++] = c;
                GGML_PRINT_DEBUG(" %u", c);
            }
        }
        GGML_PRINT_DEBUG("\n");
    }

    اگر (ggml_is_numa()) {
        FILE *fptr = fopen("/proc/sys/kernel/numa_balancing", "r");
        اگر (fptr != پوچ) {
            نویسه buf[42];
            اگر (fgets(buf, اندازه(buf), fptr) && strncmp(buf, "0\n", اندازه(buf)) != 0) {
                GGML_LOG_WARN("/proc/sys/kernel/numa_balancing is enabled, this has been observed to impair performance\n");
            }
            fclose(fptr);
        }
    }
#وگرنه
    UNUSED(numa_flag);
    // TODO
#endif
}

bool ggml_is_numa(تهی) {
    بازگشت g_state.numa.n_nodes > 1;
}

#اگر defined(__ARM_ARCH)
#اگر defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
#شامل <arm_sve.h>
ایستا تهی ggml_init_arm_arch_features(تهی) {
    ggml_arm_arch_features.sve_cnt = svcntb();
}
#وگرنه
ایستا تهی ggml_init_arm_arch_features(تهی) {}
#endif
#endif // __ARM_ARCH

#اگر defined(__riscv) && defined(__riscv_v_intrinsic)
#شامل <riscv_vector.h>
ایستا تهی ggml_init_riscv_arch_features(تهی) {
    ggml_riscv_arch_features.rvv_vlen = __riscv_vlenb();
}
#وگرنه
ایستا تهی ggml_init_riscv_arch_features(تهی) {}
#endif

ساختار ggml_tensor * ggml_new_i32(ساختار ggml_context * ctx, int32_t مقدار) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    ساختار ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

    ggml_set_i32(result, مقدار);

    بازگشت result;
}

ساختار ggml_tensor * ggml_new_f32(ساختار ggml_context * ctx, اعشاری مقدار) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    ساختار ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    ggml_set_f32(result, مقدار);

    بازگشت result;
}

ساختار ggml_tensor * ggml_set_i32 (ساختار ggml_tensor * tensor, int32_t مقدار) {
    ثابت صحیح n     = ggml_nrows(tensor);
    ثابت صحیح nc    = tensor->ne[0];
    ثابت size_t n1 = tensor->nb[1];

    نویسه * ثابت data = tensor->data;

    انتخاب (tensor->type) {
        حالت GGML_TYPE_I8:
            {
                أكد(tensor->nb[0] == اندازه(int8_t));
                برای (صحیح i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), مقدار);
                }
            } بشکن;
        حالت GGML_TYPE_I16:
            {
                أكد(tensor->nb[0] == اندازه(int16_t));
                برای (صحیح i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), مقدار);
                }
            } بشکن;
        حالت GGML_TYPE_I32:
            {
                أكد(tensor->nb[0] == اندازه(int32_t));
                برای (صحیح i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), مقدار);
                }
            } بشکن;
        حالت GGML_TYPE_F16:
            {
                أكد(tensor->nb[0] == اندازه(ggml_fp16_t));
                برای (صحیح i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), GGML_CPU_FP32_TO_FP16(مقدار));
                }
            } بشکن;
        حالت GGML_TYPE_BF16:
            {
                أكد(tensor->nb[0] == اندازه(ggml_fp16_t));
                برای (صحیح i = 0; i < n; i++) {
                    ggml_vec_set_bf16(nc, (ggml_bf16_t *)(data + i*n1), GGML_FP32_TO_BF16(مقدار));
                }
            } بشکن;
        حالت GGML_TYPE_F32:
            {
                أكد(tensor->nb[0] == اندازه(اعشاری));
                برای (صحیح i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (اعشاری *)(data + i*n1), مقدار);
                }
            } بشکن;
        پیشفرض:
            {
                GGML_ABORT("fatal error");
            }
    }

    بازگشت tensor;
}

ساختار ggml_tensor * ggml_set_f32(ساختار ggml_tensor * tensor, اعشاری مقدار) {
    ثابت صحیح n     = ggml_nrows(tensor);
    ثابت صحیح nc    = tensor->ne[0];
    ثابت size_t n1 = tensor->nb[1];

    نویسه * ثابت data = tensor->data;

    انتخاب (tensor->type) {
        حالت GGML_TYPE_I8:
            {
                أكد(tensor->nb[0] == اندازه(int8_t));
                برای (صحیح i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), مقدار);
                }
            } بشکن;
        حالت GGML_TYPE_I16:
            {
                أكد(tensor->nb[0] == اندازه(int16_t));
                برای (صحیح i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), مقدار);
                }
            } بشکن;
        حالت GGML_TYPE_I32:
            {
                أكد(tensor->nb[0] == اندازه(int32_t));
                برای (صحیح i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), مقدار);
                }
            } بشکن;
        حالت GGML_TYPE_F16:
            {
                أكد(tensor->nb[0] == اندازه(ggml_fp16_t));
                برای (صحیح i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), GGML_CPU_FP32_TO_FP16(مقدار));
                }
            } بشکن;
        حالت GGML_TYPE_BF16:
            {
                أكد(tensor->nb[0] == اندازه(ggml_bf16_t));
                برای (صحیح i = 0; i < n; i++) {
                    ggml_vec_set_bf16(nc, (ggml_bf16_t *)(data + i*n1), GGML_FP32_TO_BF16(مقدار));
                }
            } بشکن;
        حالت GGML_TYPE_F32:
            {
                أكد(tensor->nb[0] == اندازه(اعشاری));
                برای (صحیح i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (اعشاری *)(data + i*n1), مقدار);
                }
            } بشکن;
        پیشفرض:
            {
                GGML_ABORT("fatal error");
            }
    }

    بازگشت tensor;
}

int32_t ggml_get_i32_1d(ثابت ساختار ggml_tensor * tensor, صحیح i) {
    اگر (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        بازگشت ggml_get_i32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    انتخاب (tensor->type) {
        حالت GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == اندازه(int8_t));
                بازگشت ((int8_t *)(tensor->data))[i];
            }
        حالت GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == اندازه(int16_t));
                بازگشت ((int16_t *)(tensor->data))[i];
            }
        حالت GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == اندازه(int32_t));
                بازگشت ((int32_t *)(tensor->data))[i];
            }
        حالت GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == اندازه(ggml_fp16_t));
                بازگشت GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *)(tensor->data))[i]);
            }
        حالت GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == اندازه(ggml_bf16_t));
                بازگشت GGML_BF16_TO_FP32(((ggml_bf16_t *)(tensor->data))[i]);
            }
        حالت GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == اندازه(اعشاری));
                بازگشت ((اعشاری *)(tensor->data))[i];
            }
        پیشفرض:
            {
                GGML_ABORT("fatal error");
            }
    }
}

تهی ggml_set_i32_1d(ثابت ساختار ggml_tensor * tensor, صحیح i, int32_t مقدار) {
    اگر (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_set_i32_nd(tensor, id[0], id[1], id[2], id[3], مقدار);
        بازگشت;
    }
    انتخاب (tensor->type) {
        حالت GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == اندازه(int8_t));
                ((int8_t *)(tensor->data))[i] = مقدار;
            } بشکن;
        حالت GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == اندازه(int16_t));
                ((int16_t *)(tensor->data))[i] = مقدار;
            } بشکن;
        حالت GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == اندازه(int32_t));
                ((int32_t *)(tensor->data))[i] = مقدار;
            } بشکن;
        حالت GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == اندازه(ggml_fp16_t));
                ((ggml_fp16_t *)(tensor->data))[i] = GGML_CPU_FP32_TO_FP16(مقدار);
            } بشکن;
        حالت GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == اندازه(ggml_bf16_t));
                ((ggml_bf16_t *)(tensor->data))[i] = GGML_FP32_TO_BF16(مقدار);
            } بشکن;
        حالت GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == اندازه(اعشاری));
                ((اعشاری *)(tensor->data))[i] = مقدار;
            } بشکن;
        پیشفرض:
            {
                GGML_ABORT("fatal error");
            }
    }
}

int32_t ggml_get_i32_nd(ثابت ساختار ggml_tensor * tensor, صحیح i0, صحیح i1, صحیح i2, صحیح i3) {
    تهی * data   = (نویسه *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    انتخاب (tensor->type) {
        حالت GGML_TYPE_I8:
            بازگشت ((int8_t *) data)[0];
        حالت GGML_TYPE_I16:
            بازگشت ((int16_t *) data)[0];
        حالت GGML_TYPE_I32:
            بازگشت ((int32_t *) data)[0];
        حالت GGML_TYPE_F16:
            بازگشت GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *) data)[0]);
        حالت GGML_TYPE_BF16:
            بازگشت GGML_BF16_TO_FP32(((ggml_bf16_t *) data)[0]);
        حالت GGML_TYPE_F32:
            بازگشت ((اعشاری *) data)[0];
        پیشفرض:
            GGML_ABORT("fatal error");
    }
}

تهی ggml_set_i32_nd(ثابت ساختار ggml_tensor * tensor, صحیح i0, صحیح i1, صحیح i2, صحیح i3, int32_t مقدار) {
    تهی * data   = (نویسه *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    انتخاب (tensor->type) {
        حالت GGML_TYPE_I8:
            {
                ((int8_t *)(data))[0] = مقدار;
            } بشکن;
        حالت GGML_TYPE_I16:
            {
                ((int16_t *)(data))[0] = مقدار;
            } بشکن;
        حالت GGML_TYPE_I32:
            {
                ((int32_t *)(data))[0] = مقدار;
            } بشکن;
        حالت GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(data))[0] = GGML_CPU_FP32_TO_FP16(مقدار);
            } بشکن;
        حالت GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(data))[0] = GGML_FP32_TO_BF16(مقدار);
            } بشکن;
        حالت GGML_TYPE_F32:
            {
                ((اعشاری *)(data))[0] = مقدار;
            } بشکن;
        پیشفرض:
            {
                GGML_ABORT("fatal error");
            }
    }
}

اعشاری ggml_get_f32_1d(ثابت ساختار ggml_tensor * tensor, صحیح i) {
    اگر (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        بازگشت ggml_get_f32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    انتخاب (tensor->type) {
        حالت GGML_TYPE_I8:
            {
                بازگشت ((int8_t *)(tensor->data))[i];
            }
        حالت GGML_TYPE_I16:
            {
                بازگشت ((int16_t *)(tensor->data))[i];
            }
        حالت GGML_TYPE_I32:
            {
                بازگشت ((int32_t *)(tensor->data))[i];
            }
        حالت GGML_TYPE_F16:
            {
                بازگشت GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *)(tensor->data))[i]);
            }
        حالت GGML_TYPE_BF16:
            {
                بازگشت GGML_BF16_TO_FP32(((ggml_bf16_t *)(tensor->data))[i]);
            }
        حالت GGML_TYPE_F32:
            {
                بازگشت ((اعشاری *)(tensor->data))[i];
            }
        پیشفرض:
            {
                GGML_ABORT("fatal error");
            }
    }
}

تهی ggml_set_f32_1d(ثابت ساختار ggml_tensor * tensor, صحیح i, اعشاری مقدار) {
    اگر (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_set_f32_nd(tensor, id[0], id[1], id[2], id[3], مقدار);
        بازگشت;
    }
    انتخاب (tensor->type) {
        حالت GGML_TYPE_I8:
            {
                ((int8_t *)(tensor->data))[i] = مقدار;
            } بشکن;
        حالت GGML_TYPE_I16:
            {
                ((int16_t *)(tensor->data))[i] = مقدار;
            } بشکن;
        حالت GGML_TYPE_I32:
            {
                ((int32_t *)(tensor->data))[i] = مقدار;
            } بشکن;
        حالت GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(tensor->data))[i] = GGML_CPU_FP32_TO_FP16(مقدار);
            } بشکن;
        حالت GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(tensor->data))[i] = GGML_FP32_TO_BF16(مقدار);
            } بشکن;
        حالت GGML_TYPE_F32:
            {
                ((اعشاری *)(tensor->data))[i] = مقدار;
            } بشکن;
        پیشفرض:
            {
                GGML_ABORT("fatal error");
            }
    }
}

اعشاری ggml_get_f32_nd(ثابت ساختار ggml_tensor * tensor, صحیح i0, صحیح i1, صحیح i2, صحیح i3) {
    تهی * data   = (نویسه *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    انتخاب (tensor->type) {
        حالت GGML_TYPE_I8:
            بازگشت ((int8_t *) data)[0];
        حالت GGML_TYPE_I16:
            بازگشت ((int16_t *) data)[0];
        حالت GGML_TYPE_I32:
            بازگشت ((int32_t *) data)[0];
        حالت GGML_TYPE_F16:
            بازگشت GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *) data)[0]);
        حالت GGML_TYPE_BF16:
            بازگشت GGML_BF16_TO_FP32(((ggml_bf16_t *) data)[0]);
        حالت GGML_TYPE_F32:
            بازگشت ((اعشاری *) data)[0];
        پیشفرض:
            GGML_ABORT("fatal error");
    }
}

تهی ggml_set_f32_nd(ثابت ساختار ggml_tensor * tensor, صحیح i0, صحیح i1, صحیح i2, صحیح i3, اعشاری مقدار) {
    تهی * data   = (نویسه *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    انتخاب (tensor->type) {
        حالت GGML_TYPE_I8:
            {
                ((int8_t *)(data))[0] = مقدار;
            } بشکن;
        حالت GGML_TYPE_I16:
            {
                ((int16_t *)(data))[0] = مقدار;
            } بشکن;
        حالت GGML_TYPE_I32:
            {
                ((int32_t *)(data))[0] = مقدار;
            } بشکن;
        حالت GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(data))[0] = GGML_CPU_FP32_TO_FP16(مقدار);
            } بشکن;
        حالت GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(data))[0] = GGML_FP32_TO_BF16(مقدار);
            } بشکن;
        حالت GGML_TYPE_F32:
            {
                ((اعشاری *)(data))[0] = مقدار;
            } بشکن;
        پیشفرض:
            {
                GGML_ABORT("fatal error");
            }
    }
}

////////////////////////////////////////////////////////////////////////////////

// ggml_compute_forward_mul_mat

ایستا تهی ggml_compute_forward_mul_mat_one_chunk(
    ثابت ساختار ggml_compute_params * params,
    ساختار ggml_tensor * dst,
    ثابت شمارش ggml_type type,
    ثابت int64_t رقم_rows_per_vec_dot,
    ثابت int64_t ir0_start,
    ثابت int64_t ir0_end,
    ثابت int64_t ir1_start,
    ثابت int64_t ir1_end) {

    ثابت ساختار ggml_tensor * src0 = dst->src[0];
    ثابت ساختار ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_دودویی_OP_LOCALS

    ثابت bool src1_cont = ggml_is_contiguous(src1);

    ggml_vec_dot_t ثابت vec_dot      = type_traits_cpu[type].vec_dot;
    شمارش ggml_type ثابت vec_dot_type = type_traits_cpu[type].vec_dot_type;

    // broadcast factors
    ثابت int64_t r2 = ne12 / ne02;
    ثابت int64_t r3 = ne13 / ne03;

    //printf("ir0_start = %6lld, ir0_end = %6lld, ir1_start = %6lld, ir1_end = %6lld\n", ir0_start, ir0_end, ir1_start, ir1_end);

    // threads with no work simply yield (not sure if it helps)
    اگر (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        بازگشت;
    }

    ثابت تهی * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
    ثابت size_t row_size = ggml_row_size(vec_dot_type, ne10);

    أكد(ne12 % ne02 == 0);
    أكد(ne13 % ne03 == 0);

    // block-tiling attempt
    ثابت int64_t blck_0 = 16;
    ثابت int64_t blck_1 = 16;

    ثابت size_t src1_col_stride = src1_cont || src1->type != vec_dot_type ? row_size : nb11;

    // attempt to reduce false-sharing (does not seem to make a difference)
    // 16 * 2, accounting for mmla kernels
    اعشاری tmp[32];

    برای (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        برای (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            برای (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += رقم_rows_per_vec_dot) {
                ثابت int64_t i13 = (ir1 / (ne12 * ne1));
                ثابت int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                ثابت int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                // broadcast src0 into src1
                ثابت int64_t i03 = i13 / r3;
                ثابت int64_t i02 = i12 / r2;

                ثابت int64_t i1 = i11;
                ثابت int64_t i2 = i12;
                ثابت int64_t i3 = i13;

                ثابت نویسه * src0_row = (ثابت نویسه*)src0->data + (0 + i02 * nb02 + i03 * nb03);

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                ثابت نویسه * src1_col = (ثابت نویسه*)wdata +
                    (src1_cont || src1->type != vec_dot_type
                        ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                        : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                اعشاری * dst_col = (اعشاری*)((نویسه*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

                //for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                //    vec_dot(ne00, &dst_col[ir0], src0_row + ir0*nb01, src1_col);
                //}

                برای (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += رقم_rows_per_vec_dot) {
                    vec_dot(ne00, &tmp[ir0 - iir0], (رقم_rows_per_vec_dot > 1 ? 16 : 0), src0_row + ir0 * nb01, (رقم_rows_per_vec_dot > 1 ? nb01 : 0), src1_col, (رقم_rows_per_vec_dot > 1 ? src1_col_stride : 0), رقم_rows_per_vec_dot);
                }

                برای (صحیح cn = 0; cn < رقم_rows_per_vec_dot; ++cn) {
                    memcpy(&dst_col[iir0 + cn * nb1 / nb0], tmp + (cn * 16), (MIN(iir0 + blck_0, ir0_end) - iir0) * اندازه(اعشاری));
                }
            }
        }
    }
}

تهی ggml_compute_forward_mul_mat(
        ثابت ساختار ggml_compute_params * params,
              ساختار ggml_tensor * dst) {

    ثابت ساختار ggml_tensor * src0 = dst->src[0];
    ثابت ساختار ggml_tensor * src1 = dst->src[1];

    ثابت int32_t hint = ggml_get_op_params_i32(dst, 1);
    اگر (hint == GGML_HINT_SRC0_IS_HADAMARD && !params->use_ref) {
        ggml_compute_forward_fwht(params, dst);
        بازگشت;
    }

    // If tiled is supported, it will execute the full op here and we return
    اگر (ggml_compute_forward_mul_mat_tiled(params, dst)) {
        بازگشت;
    }

    GGML_TENSOR_دودویی_OP_LOCALS

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    شمارش ggml_type           ثابت vec_dot_type         = type_traits_cpu[src0->type].vec_dot_type;
    ggml_from_float_t        ثابت from_float           = type_traits_cpu[vec_dot_type].from_float;
    int64_t                  ثابت vec_dot_رقم_rows     = type_traits_cpu[src0->type].nrows;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == اندازه(اعشاری));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    // TODO: extract to "extra_op"
#اگر GGML_USE_LLAMAFILE
    // broadcast factors
    ثابت int64_t r2 = ne12 / ne02;
    ثابت int64_t r3 = ne13 / ne03;

    ثابت bool src1_cont = ggml_is_contiguous(src1);

    اگر (!params->use_ref && src1_cont) {
        برای (int64_t i13 = 0; i13 < ne13; i13++)
            برای (int64_t i12 = 0; i12 < ne12; i12++)
                اگر (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_size(src0->type),
                                     (ثابت نویسه *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_size(src0->type),
                                     (ثابت نویسه *)src1->data + i12*nb12 + i13*nb13,
                                     nb11/ggml_type_size(src1->type),
                                     (نویسه *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_size(dst->type),
                                     src0->type,
                                     src1->type,
                                     dst->type))
                    برو UseGgmlGemm1;
        بازگشت;
    }
UseGgmlGemm1:;
#endif

    اگر (src1->type != vec_dot_type) {
        نویسه * wdata = params->wdata;

        ثابت size_t nbw0 = ggml_type_size(vec_dot_type);
        ثابت size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        ثابت size_t nbw2 = nbw1*ne11;
        ثابت size_t nbw3 = nbw2*ne12;

        أكد(params->wsize >= ne13*nbw3);
        // src1 is either packed from F32 into vec_dot_type, or widened from F16 or BF16 into the F32 work buffer
        ثابت bool widen = src1->type != GGML_TYPE_F32;

        GGML_ASSERT(!widen || vec_dot_type == GGML_TYPE_F32);
        GGML_ASSERT(!widen || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_BF16);

    #اگر 0
        برای (int64_t i13 = 0; i13 < ne13; ++i13) {
            برای (int64_t i12 = 0; i12 < ne12; ++i12) {
                برای (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                    from_float((اعشاری *)((نویسه *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (تهی *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                                ne10);
                }
            }
        }
    #وگرنه
        ثابت int64_t bs = ggml_blck_size(vec_dot_type);

        برای (int64_t i13 = 0; i13 < ne13; ++i13) {
            برای (int64_t i12 = 0; i12 < ne12; ++i12) {
                برای (int64_t i11 = 0; i11 < ne11; ++i11) {
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    ثابت تهی * src1_row  = (ثابت نویسه *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10;
                    تهی       * wdata_row =                    wdata  + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0;

                    ثابت int64_t ne10_block_size = (ne10_block_end - ne10_block_start) * bs;

                    اگر (src1->type == GGML_TYPE_F16) {
                        ggml_cpu_fp16_to_fp32((ثابت ggml_fp16_t *) src1_row, (اعشاری *) wdata_row, ne10_block_size);
                    } وگرنه اگر (src1->type == GGML_TYPE_BF16) {
                        ggml_cpu_bf16_to_fp32((ثابت ggml_bf16_t *) src1_row, (اعشاری *) wdata_row, ne10_block_size);
                    } وگرنه {
                        from_float((ثابت اعشاری *) src1_row, wdata_row, ne10_block_size);
                    }
                }
            }
        }
    #endif
    }

    اگر (ith == 0) {
        // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
        atomic_store_explicit(&params->threadpool->current_chunk, nth, memory_order_relaxed);
    }

    ggml_barrier(params->threadpool);

#اگر GGML_USE_LLAMAFILE
    اگر (!params->use_ref && src1->type != vec_dot_type) {
        ثابت تهی* wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        ثابت size_t row_size = ggml_row_size(vec_dot_type, ne10);

        برای (int64_t i13 = 0; i13 < ne13; i13++)
            برای (int64_t i12 = 0; i12 < ne12; i12++)
                اگر (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_size(src0->type),
                                     (ثابت نویسه *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_size(src0->type),
                                     (ثابت نویسه *)wdata + (i12*ne11 + i13*ne12*ne11)*row_size,
                                     row_size/ggml_type_size(vec_dot_type),
                                     (نویسه *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_size(dst->type),
                                     src0->type,
                                     vec_dot_type,
                                     dst->type))
                    برو UseGgmlGemm2;
        بازگشت;
    }
UseGgmlGemm2:;
#endif

    // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
    ثابت int64_t nr0 = ne0;

    // This is the size of the rest of the dimensions of the result
    ثابت int64_t nr1 = ne1 * ne2 * ne3;

    // Now select a reasonable chunk size.
    صحیح chunk_size = 16;

    // We need to step up the size if it's small
    اگر (nr0 == 1 || nr1 == 1) {
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
    اگر (nchunk0 * nchunk1 < nth * 4 || ggml_is_numa()) {
        // distribute the thread work across the inner or outer loop based on which one is larger
        nchunk0 = nr0 > nr1 ? nth : 1; // parallelize by src0 rows
        nchunk1 = nr0 > nr1 ? 1 : nth; // parallelize by src1 rows
    }

    // The number of elements in each chunk
    ثابت int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    ثابت int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    // The first chunk comes from our thread_id, the rest will get auto-assigned.
    صحیح current_chunk = ith;

    وقتی (current_chunk < nchunk0 * nchunk1) {
        ثابت int64_t ith0 = current_chunk % nchunk0;
        ثابت int64_t ith1 = current_chunk / nchunk0;

        ثابت int64_t ir0_start = dr0 * ith0;
        ثابت int64_t ir0_end = MIN(ir0_start + dr0, nr0);

        ثابت int64_t ir1_start = dr1 * ith1;
        ثابت int64_t ir1_end = MIN(ir1_start + dr1, nr1);

        // dot kernels can handle 1 row and col at a time, but mmla kernels can process 2 rows and cols
        int64_t رقم_rows_per_vec_dot = vec_dot_رقم_rows;

        // these checks are needed to avoid crossing dim1 boundaries
        // can be optimized, but the logic would become more complicated, so keeping it like this for simplicity
        اگر ((nr0 % 2 != 0) || (ne11 % 2 != 0) || ((ir0_end - ir0_start) % 2 != 0) || ((ir1_end - ir1_start) % 2 != 0)) {
            رقم_rows_per_vec_dot = 1;
        }
        ggml_compute_forward_mul_mat_one_chunk(params, dst, src0->type, رقم_rows_per_vec_dot, ir0_start, ir0_end, ir1_start, ir1_end);

        اگر (nth >= nchunk0 * nchunk1) {
            بشکن;
        }

        current_chunk = atomic_fetch_add_explicit(&params->threadpool->current_chunk, 1, memory_order_relaxed);
    }
}

// ggml_compute_forward_mul_mat_id

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id)*ids->ne[0]*ids->ne[1] + (i1)]

ساختار mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

ایستا تهی ggml_compute_forward_mul_mat_id_one_chunk(
    ساختار ggml_tensor * dst,
    ثابت ساختار ggml_tensor * src0,
    ثابت ساختار ggml_tensor * src1,
    ثابت ساختار ggml_tensor * ids,
    ثابت int64_t cur_a,
    ثابت int64_t ir0_start,
    ثابت int64_t ir0_end,
    ثابت int64_t ir1_start,
    ثابت int64_t ir1_end,
    ثابت نویسه * src0_cur,
    ثابت ساختار mmid_row_mapping * matrix_rows,
    ثابت size_t row_size,
    ثابت bool src1_cont,
    ثابت تهی * wdata) {

    GGML_TENSOR_دودویی_OP_LOCALS

    ثابت شمارش ggml_type type = src0->type;

    ggml_vec_dot_t    ثابت vec_dot      = type_traits_cpu[type].vec_dot;
    شمارش ggml_type    ثابت vec_dot_type = type_traits_cpu[type].vec_dot_type;

    ثابت int64_t blck_0 = 16;
    ثابت int64_t blck_1 = 16;

    اعشاری tmp[16];

    برای (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        برای (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            برای (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ++ir1) {
                ثابت int64_t _i12 = ir1; // logical row index for this expert

                ساختار mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, _i12);
                ثابت صحیح id       = row_mapping.i1; // selected expert index

                ثابت int64_t  i11 = id % ne11;
                ثابت int64_t  i12 = row_mapping.i2; // row index in src1

                ثابت int64_t  i1 = id;  // selected expert index
                ثابت int64_t  i2 = i12; // row

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                ثابت نویسه * src1_col = (ثابت نویسه *) wdata +
                    (src1_cont || src1->type != vec_dot_type
                    ? (i11      + i12*ne11)*row_size
                    : (i11*nb11 + i12*nb12));

                اعشاری * dst_col = (اعشاری *) ((نویسه *) dst->data + (i1*nb1 + i2*nb2));

                برای (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                    vec_dot(ne00, &tmp[ir0 - iir0], 0, src0_cur + ir0*nb01, 0, src1_col, 0, 1);
                }

                memcpy(&dst_col[iir0], tmp, (MIN(iir0 + blck_0, ir0_end) - iir0)*اندازه(اعشاری));
            }
        }
    }
}

ایستا تهی * incr_ptr_aligned(تهی ** p, size_t size, size_t align) {

    تهی * ptr = *p;
    ptr = (تهی *) GGML_PAD((uintptr_t) ptr, align);
    *p = (تهی *) ((نویسه *) ptr + size);
    بازگشت ptr;
}

ایستا تهی ggml_compute_forward_mul_mat_id(
        ثابت ساختار ggml_compute_params * params,
              ساختار ggml_tensor * dst) {

    ثابت ساختار ggml_tensor * src0 = dst->src[0];
    ثابت ساختار ggml_tensor * src1 = dst->src[1];
    ثابت ساختار ggml_tensor * ids = dst->src[2];

    GGML_TENSOR_دودویی_OP_LOCALS

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت شمارش ggml_type type = src0->type;

    ثابت bool src1_cont = ggml_is_contiguous(src1);

    شمارش ggml_type    ثابت vec_dot_type    = type_traits_cpu[type].vec_dot_type;
    ggml_from_float_t ثابت from_float      = type_traits_cpu[vec_dot_type].from_float;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == اندازه(اعشاری));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // row groups
    ثابت صحیح n_ids = ids->ne[0]; // n_expert_used
    ثابت صحیح n_as  = ne02;       // n_expert

    تهی * wdata_cur = params->wdata;

    اگر (src1->type != vec_dot_type) {
        incr_ptr_aligned(&wdata_cur, ggml_row_size(vec_dot_type, ggml_nelements(src1)), اندازه(int64_t));
    }

    int64_t * matrix_row_counts = // [n_as]
        incr_ptr_aligned(&wdata_cur, n_as*اندازه(int64_t), اندازه(int64_t));

    ساختار mmid_row_mapping * matrix_rows = // [n_as][ids->ne[0]*ids->ne[1]]
        incr_ptr_aligned(&wdata_cur, n_as*ids->ne[0]*ids->ne[1]*اندازه(ساختار mmid_row_mapping), اندازه(int64_t));

    نویسه (*atomic_current_chunk)[CACHE_سطر_SIZE] = // [n_as]
        incr_ptr_aligned(&wdata_cur, CACHE_سطر_SIZE * n_as, CACHE_سطر_SIZE);

    // Tiled matmul (see tiled.h); per-thread work buffers, 0 bytes when disabled. The
    // reservation is unconditional, the per expert eligibility is decided at dispatch time
    نویسه * tiled_scratch = incr_ptr_aligned(&wdata_cur, ggml_tiled_wdata_size(nth, dst), 64);

    GGML_ASSERT(params->wsize >= (size_t)((نویسه *) wdata_cur - (نویسه *) params->wdata));

    اگر (src1->type != vec_dot_type) {
        نویسه * wdata = params->wdata;

        ثابت size_t nbw0 = ggml_type_size(vec_dot_type);
        ثابت size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        ثابت size_t nbw2 = nbw1*ne11;
        ثابت size_t nbw3 = nbw2*ne12;

        أكد(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

#اگر 0
        برای (int64_t i13 = 0; i13 < ne13; ++i13) {
            برای (int64_t i12 = ith; i12 < ne12; i12 += nth) {
                برای (int64_t i11 = 0; i11 < ne11; ++i11) {
                    from_float((اعشاری *)((نویسه *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (تهی *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                               ne10);
                }
            }
        }
#وگرنه
        برای (int64_t i13 = 0; i13 < ne13; ++i13) {
            برای (int64_t i12 = 0; i12 < ne12; ++i12) {
                برای (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = ggml_blck_size(vec_dot_type);
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    from_float((اعشاری *)((نویسه *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10),
                               (تهی *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0),
                               (ne10_block_end - ne10_block_start) * bs);
                }
            }
        }
#endif
    }

    اگر (ith == 0) {
        // initialize matrix_row_counts
        memset(matrix_row_counts, 0, n_as*اندازه(int64_t));

        // group rows by src0 matrix
        برای (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            برای (صحیح id = 0; id < n_ids; ++id) {
                ثابت int32_t i02 = *(ثابت int32_t *) ((ثابت نویسه *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);

                أكد(i02 >= 0 && i02 < n_as);

                MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = (ساختار mmid_row_mapping) {id, iid1};
                matrix_row_counts[i02] += 1;
            }
        }
    }

    // reset current_chunk
    برای (صحیح cur_a = ith; cur_a < n_as; cur_a += nth) {
        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);
        *current_chunk_ctr = nth;
    }

    ggml_barrier(params->threadpool);

    برای (صحیح cur_a = 0; cur_a < n_as; ++cur_a) {
        ثابت int64_t cne1 = matrix_row_counts[cur_a];

        اگر (cne1 == 0) {
            ادامه;
        }

        // tiled takes over if profitable for this expert (see tiled.h)
        اگر (ggml_compute_forward_mul_mat_id_tiled(params, dst, cur_a, cne1, (ثابت int32_t *) &MMID_MATRIX_ROW(cur_a, 0), tiled_scratch)) {
            ادامه;
        }

        ثابت نویسه * src0_cur = (ثابت نویسه *) src0->data + cur_a * nb02;
        ثابت تهی * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        ثابت size_t row_size = ggml_row_size(vec_dot_type, ne10);

        ثابت int64_t nr0 = ne01;
        ثابت int64_t nr1 = cne1;

        صحیح chunk_size = 16;
        اگر (nr0 == 1 || nr1 == 1) {
            chunk_size = 64;
        }

        // disable for NUMA
        ثابت bool disable_chunking = ggml_is_numa();

        int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
        int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

        اگر (nchunk0 * nchunk1 < nth * 4 || disable_chunking) {
            nchunk0 = nr0 > nr1 ? nth : 1;
            nchunk1 = nr0 > nr1 ? 1 : nth;
        }

        ثابت int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        ثابت int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

        صحیح current_chunk = ith;

        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);

        وقتی (current_chunk < nchunk0 * nchunk1) {
            ثابت int64_t ith0 = current_chunk % nchunk0;
            ثابت int64_t ith1 = current_chunk / nchunk0;

            ثابت int64_t ir0_start = dr0 * ith0;
            ثابت int64_t ir0_end = MIN(ir0_start + dr0, nr0);

            ثابت int64_t ir1_start = dr1 * ith1;
            ثابت int64_t ir1_end = MIN(ir1_start + dr1, nr1);

            ggml_compute_forward_mul_mat_id_one_chunk(
                dst, src0, src1, ids, cur_a,
                ir0_start, ir0_end, ir1_start, ir1_end,
                src0_cur, matrix_rows, row_size, src1_cont, wdata
            );

            اگر (nth >= nchunk0 * nchunk1) {
                بشکن;
            }

            current_chunk = atomic_fetch_add_explicit(current_chunk_ctr, 1, memory_order_relaxed);
        }
    }
}

/////////////////////////////////

ایستا تهی ggml_compute_forward(ساختار ggml_compute_params * params, ساختار ggml_tensor * tensor) {
    GGML_ASSERT(params);

    اگر (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {
        بازگشت;
    }

    // extra_buffer op?
    اگر (ggml_cpu_extra_compute_forward(params, tensor)) {
        بازگشت;
    }

    انتخاب (tensor->op) {
        حالت GGML_OP_DUP:
            {
                ggml_compute_forward_dup(params, tensor);
            } بشکن;
        حالت GGML_OP_ADD:
            {
                ggml_compute_forward_add(params, tensor);
            } بشکن;
        حالت GGML_OP_ADD_ID:
            {
                ggml_compute_forward_add_id(params, tensor);
            } بشکن;
        حالت GGML_OP_ADD1:
            {
                ggml_compute_forward_add1(params, tensor);
            } بشکن;
        حالت GGML_OP_ACC:
            {
                ggml_compute_forward_acc(params, tensor);
            } بشکن;
        حالت GGML_OP_SUB:
            {
                ggml_compute_forward_sub(params, tensor);
            } بشکن;
        حالت GGML_OP_MUL:
            {
                ggml_compute_forward_mul(params, tensor);
            } بشکن;
        حالت GGML_OP_DIV:
            {
                ggml_compute_forward_div(params, tensor);
            } بشکن;
        حالت GGML_OP_SQR:
            {
                ggml_compute_forward_sqr(params, tensor);
            } بشکن;
        حالت GGML_OP_SQRT:
            {
                ggml_compute_forward_sqrt(params, tensor);
            } بشکن;
        حالت GGML_OP_LOG:
            {
                ggml_compute_forward_log(params, tensor);
            } بشکن;
        حالت GGML_OP_SIN:
            {
                ggml_compute_forward_sin(params, tensor);
            } بشکن;
        حالت GGML_OP_COS:
            {
                ggml_compute_forward_cos(params, tensor);
            } بشکن;
        حالت GGML_OP_SUM:
            {
                ggml_compute_forward_sum(params, tensor);
            } بشکن;
        حالت GGML_OP_SUM_ROWS:
            {
                ggml_compute_forward_sum_rows(params, tensor);
            } بشکن;
        حالت GGML_OP_CUMSUM:
            {
                ggml_compute_forward_cumsum(params, tensor);
            } بشکن;
        حالت GGML_OP_MEAN:
            {
                ggml_compute_forward_mean(params, tensor);
            } بشکن;
        حالت GGML_OP_ARGMAX:
            {
                ggml_compute_forward_argmax(params, tensor);
            } بشکن;
        حالت GGML_OP_شمار_EQUAL:
            {
                ggml_compute_forward_شمار_equal(params, tensor);
            } بشکن;
        حالت GGML_OP_REPEAT:
            {
                ggml_compute_forward_repeat(params, tensor);
            } بشکن;
        حالت GGML_OP_REPEAT_BACK:
            {
                ggml_compute_forward_repeat_back(params, tensor);
            } بشکن;
        حالت GGML_OP_CONCAT:
            {
                ggml_compute_forward_concat(params, tensor);
            } بشکن;
        حالت GGML_OP_SILU_BACK:
            {
                ggml_compute_forward_silu_back(params, tensor);
            } بشکن;
        حالت GGML_OP_NORM:
            {
                ggml_compute_forward_norm(params, tensor);
            } بشکن;
        حالت GGML_OP_RMS_NORM:
            {
                ggml_compute_forward_rms_norm(params, tensor);
            } بشکن;
        حالت GGML_OP_RMS_NORM_BACK:
            {
                ggml_compute_forward_rms_norm_back(params, tensor);
            } بشکن;
        حالت GGML_OP_GROUP_NORM:
            {
                ggml_compute_forward_group_norm(params, tensor);
            } بشکن;
        حالت GGML_OP_L2_NORM:
            {
                ggml_compute_forward_l2_norm(params, tensor);
            } بشکن;
        حالت GGML_OP_MUL_MAT:
            {
                ggml_compute_forward_mul_mat(params, tensor);
            } بشکن;
        حالت GGML_OP_MUL_MAT_ID:
            {
                ggml_compute_forward_mul_mat_id(params, tensor);
            } بشکن;
        حالت GGML_OP_OUT_PROD:
            {
                ggml_compute_forward_out_prod(params, tensor);
            } بشکن;
        حالت GGML_OP_SCALE:
            {
                ggml_compute_forward_scale(params, tensor);
            } بشکن;
        حالت GGML_OP_SET:
            {
                ggml_compute_forward_set(params, tensor);
            } بشکن;
        حالت GGML_OP_CPY:
            {
                ggml_compute_forward_cpy(params, tensor);
            } بشکن;
        حالت GGML_OP_CONT:
            {
                ggml_compute_forward_cont(params, tensor);
            } بشکن;
        حالت GGML_OP_GET_ROWS:
            {
                ggml_compute_forward_get_rows(params, tensor);
            } بشکن;
        حالت GGML_OP_GET_ROWS_BACK:
            {
                ggml_compute_forward_get_rows_back(params, tensor);
            } بشکن;
        حالت GGML_OP_SET_ROWS:
            {
                ggml_compute_forward_set_rows(params, tensor);
            } بشکن;
        حالت GGML_OP_DIAG:
            {
                ggml_compute_forward_diag(params, tensor);
            } بشکن;
        حالت GGML_OP_DIAG_MASK_INF:
            {
                ggml_compute_forward_diag_mask_inf(params, tensor);
            } بشکن;
        حالت GGML_OP_DIAG_MASK_ZERO:
            {
                ggml_compute_forward_diag_mask_zero(params, tensor);
            } بشکن;
        حالت GGML_OP_SOFT_MAX:
            {
                ggml_compute_forward_soft_max(params, tensor);
            } بشکن;
        حالت GGML_OP_SOFT_MAX_BACK:
            {
                ggml_compute_forward_soft_max_ext_back(params, tensor);
            } بشکن;
        حالت GGML_OP_ROPE:
            {
                ggml_compute_forward_rope(params, tensor);
            } بشکن;
        حالت GGML_OP_ROPE_BACK:
            {
                ggml_compute_forward_rope_back(params, tensor);
            } بشکن;
        حالت GGML_OP_CLAMP:
            {
                ggml_compute_forward_clamp(params, tensor);
            } بشکن;
        حالت GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_compute_forward_conv_transpose_1d(params, tensor);
            } بشکن;
        حالت GGML_OP_IM2COL:
            {
                ggml_compute_forward_im2col(params, tensor);
            } بشکن;
        حالت GGML_OP_IM2COL_BACK:
            {
                ggml_compute_forward_im2col_back_f32(params, tensor);
            } بشکن;
        حالت GGML_OP_IM2COL_3D:
            {
                ggml_compute_forward_im2col_3d(params, tensor);
            } بشکن;
        حالت GGML_OP_COL2IM_1D:
            {
                ggml_compute_forward_col2im_1d(params, tensor);
            } بشکن;
        حالت GGML_OP_CONV_2D:
            {
                ggml_compute_forward_conv_2d(params, tensor);
            } بشکن;
        حالت GGML_OP_CONV_3D:
            {
                ggml_compute_forward_conv_3d(params, tensor);
            } بشکن;
        حالت GGML_OP_CONV_2D_DW:
            {
                ggml_compute_forward_conv_2d_dw(params, tensor);
            } بشکن;
        حالت GGML_OP_CONV_TRANSPOSE_2D:
            {
                ggml_compute_forward_conv_transpose_2d(params, tensor);
            } بشکن;
        حالت GGML_OP_POOL_1D:
            {
                ggml_compute_forward_pool_1d(params, tensor);
            } بشکن;
        حالت GGML_OP_POOL_2D:
            {
                ggml_compute_forward_pool_2d(params, tensor);
            } بشکن;
        حالت GGML_OP_POOL_2D_BACK:
            {
                ggml_compute_forward_pool_2d_back(params, tensor);
            } بشکن;
        حالت GGML_OP_UPSCALE:
            {
                ggml_compute_forward_upscale(params, tensor);
            } بشکن;
        حالت GGML_OP_PAD:
            {
                ggml_compute_forward_pad(params, tensor);
            } بشکن;
        حالت GGML_OP_PAD_REFLECT_1D:
            {
                ggml_compute_forward_pad_reflect_1d(params, tensor);
            } بشکن;
        حالت GGML_OP_ROLL:
            {
                ggml_compute_forward_roll(params, tensor);
            } بشکن;
        حالت GGML_OP_ARANGE:
            {
                ggml_compute_forward_arange(params, tensor);
            } بشکن;
        حالت GGML_OP_TIMESTEP_EMBEDDING:
            {
                ggml_compute_forward_timestep_embedding(params, tensor);
            } بشکن;
        حالت GGML_OP_ARGSORT:
            {
                ggml_compute_forward_argsort(params, tensor);
            } بشکن;
        حالت GGML_OP_بالاترین_K:
            {
                ggml_compute_forward_بالاترین_k(params, tensor);
            } بشکن;
        حالت GGML_OP_LEAKY_RELU:
            {
                ggml_compute_forward_leaky_relu(params, tensor);
            } بشکن;
        حالت GGML_OP_TRI:
            {
                ggml_compute_forward_tri(params, tensor);
            } بشکن;
        حالت GGML_OP_FILL:
            {
                ggml_compute_forward_fill(params, tensor);
            } بشکن;
        حالت GGML_OP_FLASH_ATTN_EXT:
            {
                ggml_compute_forward_flash_attn_ext(params, tensor);
            } بشکن;
        حالت GGML_OP_FLASH_ATTN_BACK:
            {
                int32_t t = ggml_get_op_params_i32(tensor, 0);
                GGML_ASSERT(t == 0 || t == 1);
                bool masked = t != 0;
                ggml_compute_forward_flash_attn_back(params, masked, tensor);
            } بشکن;
        حالت GGML_OP_SSM_CONV:
            {
                ggml_compute_forward_ssm_conv(params, tensor);
            } بشکن;
        حالت GGML_OP_SSM_SCAN:
            {
                ggml_compute_forward_ssm_scan(params, tensor);
            } بشکن;
        حالت GGML_OP_WIN_PART:
            {
                ggml_compute_forward_win_part(params, tensor);
            } بشکن;
        حالت GGML_OP_WIN_UNPART:
            {
                ggml_compute_forward_win_unpart(params, tensor);
            } بشکن;
        حالت GGML_OP_UNARY:
            {
                ggml_compute_forward_unary(params, tensor);
            } بشکن;
        حالت GGML_OP_GLU:
            {
                ggml_compute_forward_glu(params, tensor);
            } بشکن;
        حالت GGML_OP_GET_REL_POS:
            {
                ggml_compute_forward_get_rel_pos(params, tensor);
            } بشکن;
        حالت GGML_OP_ADD_REL_POS:
            {
                ggml_compute_forward_add_rel_pos(params, tensor);
            } بشکن;
        حالت GGML_OP_RWKV_WKV6:
            {
                ggml_compute_forward_rwkv_wkv6(params, tensor);
            } بشکن;
        حالت GGML_OP_GATED_LINEAR_ATTN:
            {
                ggml_compute_forward_gla(params, tensor);
            } بشکن;
        حالت GGML_OP_RWKV_WKV7:
            {
                ggml_compute_forward_rwkv_wkv7(params, tensor);
            } بشکن;
        حالت GGML_OP_SOLVE_TRI:
            {
                ggml_compute_forward_solve_tri(params, tensor);
            } بشکن;
        حالت GGML_OP_GATED_DELTA_NET:
            {
                ggml_compute_forward_gated_delta_net(params, tensor);
            } بشکن;
        حالت GGML_OP_LIGHTNING_INDEXER:
            {
                ggml_compute_forward_lightning_indexer(params, tensor);
            } بشکن;
        حالت GGML_OP_DSV4_HC_COMB:
            {
                ggml_compute_forward_dsv4_hc_comb(params, tensor);
            } بشکن;
        حالت GGML_OP_DSV4_HC_PRE:
            {
                ggml_compute_forward_dsv4_hc_pre(params, tensor);
            } بشکن;
        حالت GGML_OP_DSV4_HC_POST:
            {
                ggml_compute_forward_dsv4_hc_post(params, tensor);
            } بشکن;
        حالت GGML_OP_بالاترین_K_MASK:
            {
                ggml_compute_forward_بالاترین_k_mask(params, tensor);
            } بشکن;
        حالت GGML_OP_MAP_CUSTOM1:
            {
                ggml_compute_forward_map_custom1(params, tensor);
            }
            بشکن;
        حالت GGML_OP_MAP_CUSTOM2:
            {
                ggml_compute_forward_map_custom2(params, tensor);
            }
            بشکن;
        حالت GGML_OP_MAP_CUSTOM3:
            {
                ggml_compute_forward_map_custom3(params, tensor);
            }
            بشکن;
        حالت GGML_OP_CUSTOM:
            {
                ggml_compute_forward_custom(params, tensor);
            }
            بشکن;
        حالت GGML_OP_CROSS_ENTROPY_LOSS:
            {
                ggml_compute_forward_cross_entropy_loss(params, tensor);
            }
            بشکن;
        حالت GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            {
                ggml_compute_forward_cross_entropy_loss_back(params, tensor);
            }
            بشکن;
        حالت GGML_OP_OPT_STEP_ADAMW:
            {
                ggml_compute_forward_opt_step_adamw(params, tensor);
            }
            بشکن;
        حالت GGML_OP_OPT_STEP_SGD:
            {
                ggml_compute_forward_opt_step_sgd(params, tensor);
            }
            بشکن;
        حالت GGML_OP_NONE:
            {
                // nop
            } بشکن;
        حالت GGML_OP_RESHAPE:
            {
                // nop
            } بشکن;
        حالت GGML_OP_PERMUTE:
            {
                // nop
            } بشکن;
        حالت GGML_OP_VIEW:
            {
                // nop
            } بشکن;
        حالت GGML_OP_TRANSPOSE:
            {
                // nop
            } بشکن;
        حالت GGML_OP_شمار:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// Android's libc implementation "bionic" does not support setting affinity
#اگر defined(__gnu_linux__)
ایستا تهی set_numa_thread_affinity(صحیح thread_n) {
    اگر (!ggml_is_numa()) {
        بازگشت;
    }

    صحیح گره_رقم;
    صحیح rv;
    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    انتخاب(g_state.numa.numa_strategy) {
        حالت GGML_NUMA_STRATEGY_DISTRIBUTE:
            // run thread on node_num thread_n / (threads per node)
            گره_رقم = thread_n % g_state.numa.n_nodes;
            بشکن;
        حالت GGML_NUMA_STRATEGY_ISOLATE:
            // run thread on current_node
            گره_رقم = g_state.numa.current_گره;
            بشکن;
        حالت GGML_NUMA_STRATEGY_NUMACTL:
            // use the cpuset that numactl gave us
            rv = pthread_setaffinity_np(pthread_self(), setsize, &g_state.numa.cpuset);
            اگر (rv) {
                fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n",strerror(rv));
            }
            بازگشت;
        پیشفرض:
            بازگشت;
    }

    ساختار ggml_numa_گره * گره = &g_state.numa.nodes[گره_رقم];

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    برای (size_t i = 0; i < گره->n_cpus; ++i) {
        CPU_SET_S(گره->cpus[i], setsize, cpus);
    }

    rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    اگر (rv) {
            fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}

ایستا تهی clear_numa_thread_affinity(تهی) {
    اگر (!ggml_is_numa()) {
        بازگشت;
    }

    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    برای (unsigned i = 0; i < g_state.numa.total_cpus; ++i) {
        CPU_SET_S(i, setsize, cpus);
    }

    صحیح rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    اگر (rv) {
        fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}
#وگرنه
// TODO: Windows etc.
// (the linux implementation may also work on BSD, someone should test)
ایستا تهی set_numa_thread_affinity(صحیح thread_n) { UNUSED(thread_n);  }
ایستا تهی clear_numa_thread_affinity(تهی) {}
#endif

ایستا صحیح ggml_get_n_tasks(ساختار ggml_tensor * گره, صحیح n_threads) {
    صحیح n_tasks = 0;

    اگر (ggml_is_empty(گره)) {
        // no need to multi-thread a no-op
        n_tasks = 1;
        بازگشت n_tasks;
    }

    انتخاب (گره->op) {
        حالت GGML_OP_CPY:
        حالت GGML_OP_DUP:
        حالت GGML_OP_CONT:
        حالت GGML_OP_ADD:
        حالت GGML_OP_ADD_ID:
        حالت GGML_OP_ADD1:
        حالت GGML_OP_ACC:
        حالت GGML_OP_CUMSUM:
        حالت GGML_OP_TRI:
        حالت GGML_OP_FILL:
            {
                n_tasks = n_threads;
            } بشکن;
        حالت GGML_OP_SUB:
        حالت GGML_OP_SQR:
        حالت GGML_OP_SQRT:
        حالت GGML_OP_LOG:
        حالت GGML_OP_SIN:
        حالت GGML_OP_COS:
        حالت GGML_OP_SUM:
        حالت GGML_OP_SUM_ROWS:
        حالت GGML_OP_MEAN:
        حالت GGML_OP_ARGMAX:
            {
                n_tasks = 1;
            } بشکن;
        حالت GGML_OP_شمار_EQUAL:
        حالت GGML_OP_SOLVE_TRI:
        حالت GGML_OP_GATED_DELTA_NET:
        حالت GGML_OP_DSV4_HC_COMB:
        حالت GGML_OP_DSV4_HC_PRE:
        حالت GGML_OP_DSV4_HC_POST:
        حالت GGML_OP_بالاترین_K_MASK:
            {
                n_tasks = n_threads;
            } بشکن;
        حالت GGML_OP_REPEAT:
        حالت GGML_OP_REPEAT_BACK:
        حالت GGML_OP_LEAKY_RELU:
            {
                n_tasks = 1;
            } بشکن;
        حالت GGML_OP_UNARY:
            انتخاب (ggml_get_unary_op(گره)) {
                حالت GGML_UNARY_OP_ABS:
                حالت GGML_UNARY_OP_SGN:
                حالت GGML_UNARY_OP_NEG:
                حالت GGML_UNARY_OP_STEP:
                حالت GGML_UNARY_OP_TANH:
                حالت GGML_UNARY_OP_ELU:
                حالت GGML_UNARY_OP_RELU:
                حالت GGML_UNARY_OP_SIGMOID:
                حالت GGML_UNARY_OP_HARDSWISH:
                حالت GGML_UNARY_OP_HARDSIGMOID:
                حالت GGML_UNARY_OP_EXP:
                حالت GGML_UNARY_OP_SOFTPLUS:
                حالت GGML_UNARY_OP_EXPM1:
                حالت GGML_UNARY_OP_FLOOR:
                حالت GGML_UNARY_OP_CEIL:
                حالت GGML_UNARY_OP_ROUND:
                حالت GGML_UNARY_OP_TRUNC:
                    {
                        n_tasks = 1;
                    } بشکن;

                حالت GGML_UNARY_OP_GELU:
                حالت GGML_UNARY_OP_GELU_ERF:
                حالت GGML_UNARY_OP_GELU_QUICK:
                حالت GGML_UNARY_OP_SILU:
                حالت GGML_UNARY_OP_XIELU:
                    {
                        n_tasks = n_threads;
                    } بشکن;
                پیشفرض:
                    GGML_ABORT("fatal error");
            }
            بشکن;
        حالت GGML_OP_GLU:
            انتخاب (ggml_get_glu_op(گره)) {
                حالت GGML_GLU_OP_REGLU:
                حالت GGML_GLU_OP_GEGLU:
                حالت GGML_GLU_OP_SWIGLU:
                حالت GGML_GLU_OP_SWIGLU_OAI:
                حالت GGML_GLU_OP_GEGLU_ERF:
                حالت GGML_GLU_OP_GEGLU_QUICK:
                حالت GGML_GLU_OP_SWIGLU_CLAMP:
                    {
                        n_tasks = n_threads;
                    } بشکن;
                پیشفرض:
                    GGML_ABORT("fatal error");
            }
            بشکن;
        حالت GGML_OP_SILU_BACK:
        حالت GGML_OP_MUL:
        حالت GGML_OP_DIV:
        حالت GGML_OP_NORM:
        حالت GGML_OP_RMS_NORM:
        حالت GGML_OP_RMS_NORM_BACK:
        حالت GGML_OP_L2_NORM:
        حالت GGML_OP_GROUP_NORM:
        حالت GGML_OP_CONCAT:
        حالت GGML_OP_MUL_MAT:
        حالت GGML_OP_MUL_MAT_ID:
        حالت GGML_OP_OUT_PROD:
        حالت GGML_OP_SET_ROWS:
            {
                n_tasks = n_threads;
            } بشکن;
        حالت GGML_OP_GET_ROWS:
            {
                // FIXME: get_rows can use additional threads, but the cost of launching additional threads
                // decreases performance with GPU offloading
                //n_tasks = n_threads;
                n_tasks = 1;
            } بشکن;
        حالت GGML_OP_SCALE:
        حالت GGML_OP_SET:
        حالت GGML_OP_RESHAPE:
        حالت GGML_OP_VIEW:
        حالت GGML_OP_PERMUTE:
        حالت GGML_OP_TRANSPOSE:
        حالت GGML_OP_GET_ROWS_BACK:
        حالت GGML_OP_DIAG:
            {
                n_tasks = 1;
            } بشکن;
        حالت GGML_OP_DIAG_MASK_ZERO:
        حالت GGML_OP_DIAG_MASK_INF:
        حالت GGML_OP_SOFT_MAX_BACK:
        حالت GGML_OP_ROPE:
        حالت GGML_OP_ROPE_BACK:
        حالت GGML_OP_ADD_REL_POS:
            {
                n_tasks = n_threads;
            } بشکن;
        حالت GGML_OP_CLAMP:
            {
                n_tasks = 1; //TODO
            } بشکن;
        حالت GGML_OP_SOFT_MAX:
            {
                n_tasks = MIN(n_threads, ggml_nrows(گره->src[0]));
            } بشکن;
        حالت GGML_OP_IM2COL:
        حالت GGML_OP_IM2COL_BACK:
        حالت GGML_OP_IM2COL_3D:
        حالت GGML_OP_CONV_2D:
        حالت GGML_OP_CONV_3D:
        حالت GGML_OP_CONV_2D_DW:
        حالت GGML_OP_COL2IM_1D:
        حالت GGML_OP_CONV_TRANSPOSE_1D:
        حالت GGML_OP_CONV_TRANSPOSE_2D:
            {
                n_tasks = n_threads;
            } بشکن;
        حالت GGML_OP_POOL_1D:
        حالت GGML_OP_POOL_2D:
        حالت GGML_OP_POOL_2D_BACK:
            {
                n_tasks = 1;
            } بشکن;
        حالت GGML_OP_UPSCALE:
        حالت GGML_OP_PAD:
        حالت GGML_OP_PAD_REFLECT_1D:
        حالت GGML_OP_ROLL:
        حالت GGML_OP_ARANGE:
        حالت GGML_OP_TIMESTEP_EMBEDDING:
        حالت GGML_OP_ARGSORT:
        حالت GGML_OP_بالاترین_K:
        حالت GGML_OP_FLASH_ATTN_EXT:
        حالت GGML_OP_FLASH_ATTN_BACK:
        حالت GGML_OP_SSM_CONV:
        حالت GGML_OP_SSM_SCAN:
        حالت GGML_OP_LIGHTNING_INDEXER:
            {
                n_tasks = n_threads;
            } بشکن;
        حالت GGML_OP_RWKV_WKV6:
        حالت GGML_OP_GATED_LINEAR_ATTN:
        حالت GGML_OP_RWKV_WKV7:
            {
                ثابت int64_t n_heads = گره->src[1]->ne[1];
                n_tasks = MIN(n_threads, n_heads);
            } بشکن;
        حالت GGML_OP_WIN_PART:
        حالت GGML_OP_WIN_UNPART:
        حالت GGML_OP_GET_REL_POS:
            {
                n_tasks = 1;
            } بشکن;
        حالت GGML_OP_MAP_CUSTOM1:
            {
                ساختار ggml_map_custom1_op_params p;
                memcpy(&p, گره->op_params, اندازه(p));
                اگر (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } وگرنه {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } بشکن;
        حالت GGML_OP_MAP_CUSTOM2:
            {
                ساختار ggml_map_custom2_op_params p;
                memcpy(&p, گره->op_params, اندازه(p));
                اگر (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } وگرنه {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } بشکن;
        حالت GGML_OP_MAP_CUSTOM3:
            {
                ساختار ggml_map_custom3_op_params p;
                memcpy(&p, گره->op_params, اندازه(p));
                اگر (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } وگرنه {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } بشکن;
        حالت GGML_OP_CUSTOM:
            {
                ساختار ggml_custom_op_params p;
                memcpy(&p, گره->op_params, اندازه(p));
                اگر (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } وگرنه {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } بشکن;
        حالت GGML_OP_CROSS_ENTROPY_LOSS:
        حالت GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        حالت GGML_OP_OPT_STEP_ADAMW:
        حالت GGML_OP_OPT_STEP_SGD:
            {
                n_tasks = n_threads;
            } بشکن;
        حالت GGML_OP_NONE:
            {
                n_tasks = 1;
            } بشکن;
        حالت GGML_OP_شمار:
            {
                GGML_ABORT("fatal error");
            }
        پیشفرض:
            {
                fprintf(stderr, "%s: op not implemented: ", __func__);
                اگر (گره->op < GGML_OP_شمار) {
                    fprintf(stderr, "%s\n", ggml_op_name(گره->op));
                } وگرنه {
                    fprintf(stderr, "%d\n", گره->op);
                }
                GGML_ABORT("fatal error");
            }
    }

    أكد(n_tasks > 0);

    بازگشت n_tasks;
}

ایستا thread_ret_t ggml_graph_compute_secondary_thread(تهی* data);

#اگر defined(_WIN32)
#شامل "windows.h"

// TODO: support > 64 CPUs
ایستا bool ggml_thread_apply_affinity(bool * mask) {
    HANDLE    h = GetCurrentThread();
    uint64_t  bitmask = 0ULL;

    أكد(GGML_MAX_N_THREADS >= 64);

    برای (int32_t i = 0; i < 8; i++) {
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

    برای (int32_t i = 64; i < GGML_MAX_N_THREADS; i++) {
        اگر (mask[i]) {
            fprintf(stderr, "warn: setting thread-affinity for > 64 CPUs isn't supported on windows!\n");
            بشکن;
        }
    }

    DWORD_PTR m = (DWORD_PTR)bitmask;

    m = SetThreadAffinityMask(h, m);

    بازگشت m != 0;
}

ایستا bool ggml_thread_apply_priority(int32_t prio) {
    // Note that on Windows the Process Priority Class must be updated in order to set Thread priority.
    // This is up to the applications.
    DWORD p = THREAD_PRIORITY_NORMAL;
    انتخاب (prio) {
        حالت GGML_SCHED_PRIO_کم:      p = THREAD_PRIORITY_BELOW_NORMAL;  بشکن;
        حالت GGML_SCHED_PRIO_NORMAL:   p = THREAD_PRIORITY_NORMAL;        بشکن;
        حالت GGML_SCHED_PRIO_MEDIUM:   p = THREAD_PRIORITY_ABOVE_NORMAL;  بشکن;
        حالت GGML_SCHED_PRIO_بالا:     p = THREAD_PRIORITY_HIGHEST;       بشکن;
        حالت GGML_SCHED_PRIO_REALTIME: p = THREAD_PRIORITY_TIME_CRITICAL; بشکن;
    }

    اگر (prio != GGML_SCHED_PRIO_کم) {
        // Tell Windows that this thread should not be throttled (needs its own CPU core).
        // Newer Windows 11 versions aggressively park (offline) CPU cores and often place
        // all our threads onto the first 4 cores which results in terrible performance with
        // n_threads > 4
        #اگر _WIN32_WINNT >= 0x0602
        THREAD_POWER_THROTTLING_STATE t;
        ZeroMemory(&t, اندازه(t));
        t.Version     = THREAD_POWER_THROTTLING_CURRENT_VERSION;
        t.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
        t.StateMask   = 0;

        اگر (!SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &t, اندازه(t))) {
            GGML_LOG_DEBUG("failed to disable thread power throttling %d : (%d)\n", prio, (صحیح) GetLastError());
            بازگشت false;
        }
        #endif
    }

    اگر (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        بازگشت true;
    }

    اگر (!SetThreadPriority(GetCurrentThread(), p)) {
        fprintf(stderr, "warn: failed to set thread priority %d : (%d)\n", prio, (صحیح) GetLastError());
        بازگشت false;
    }

    بازگشت true;
}

#وگرنهاگر defined(__APPLE__)
#شامل <sys/types.h>
#شامل <sys/resource.h>

ایستا bool ggml_thread_apply_affinity(ثابت bool * mask) {
    // Not supported on Apple platforms
    UNUSED(mask);
    بازگشت true;
}

ایستا bool ggml_thread_apply_priority(int32_t prio) {
    ساختار sched_param p;
    int32_t policy = SCHED_OTHER;
    انتخاب (prio) {
        // TODO: there seems to be no way to set lower prio on Apple platforms
        حالت GGML_SCHED_PRIO_کم:      policy = SCHED_OTHER; p.sched_priority = 0;  بشکن;
        حالت GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  بشکن;
        حالت GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; بشکن;
        حالت GGML_SCHED_PRIO_بالا:     policy = SCHED_FIFO;  p.sched_priority = 80; بشکن;
        حالت GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; بشکن;
    }

    اگر (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        بازگشت true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    اگر (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        بازگشت false;
    }

    بازگشت true;
}

#وگرنهاگر defined(__linux__)
// TODO: this may not work on BSD, to be verified

ایستا bool ggml_thread_apply_affinity(ثابت bool * mask) {
    cpu_set_t cpuset;
    صحیح err;

    CPU_ZERO(&cpuset);

    برای (uint32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
        اگر (mask[i]) {
            GGML_PRINT_DEBUG("Thread %lx: adding %d to cpuset\n", pthread_self(), i);
            CPU_SET(i, &cpuset);
        }
    }

#ifdef __ANDROID__
    err = sched_setaffinity(0, اندازه(cpuset), &cpuset);
    اگر (err < 0) {
        err = errno;
    }
#وگرنه
    err = pthread_setaffinity_np(pthread_self(), اندازه(cpuset), &cpuset);
#endif
    اگر (err != 0) {
        fprintf(stderr, "warn: failed to set affinity mask 0x%llx : %s (%d)\n", (unsigned long long)mask, strerror(err), err);
        بازگشت false;
    }

    بازگشت true;
}

ایستا bool ggml_thread_apply_priority(int32_t prio) {
    ساختار sched_param p;
    int32_t policy = SCHED_OTHER;
    انتخاب (prio) {
        حالت GGML_SCHED_PRIO_کم:      policy = SCHED_BATCH; p.sched_priority = 0;  بشکن;
        حالت GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  بشکن;
        حالت GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; بشکن;
        حالت GGML_SCHED_PRIO_بالا:     policy = SCHED_FIFO;  p.sched_priority = 80; بشکن;
        حالت GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; بشکن;
    }

    اگر (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        بازگشت true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    اگر (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        بازگشت false;
    }

    بازگشت true;
}

#وگرنه // unsupported platforms

ایستا bool ggml_thread_apply_affinity(ثابت bool * mask) {
    UNUSED(mask);
    بازگشت true;
}

ایستا bool ggml_thread_apply_priority(int32_t prio) {
    UNUSED(prio);
    بازگشت true;
}

#endif

ایستا bool ggml_thread_cpumask_is_valid(ثابت bool * mask) {
    برای (صحیح i = 0; i < GGML_MAX_N_THREADS; i++) {
        اگر (mask[i]) { بازگشت true; }
    }
    بازگشت false;
}

ایستا تهی ggml_thread_cpumask_next(ثابت bool * global_mask, bool * local_mask, bool strict, int32_t* iter) {
    اگر (!strict) {
        memcpy(local_mask, global_mask, GGML_MAX_N_THREADS);
        بازگشت;
    } وگرنه {
        memset(local_mask, 0, GGML_MAX_N_THREADS);
        int32_t base_idx = *iter;
        برای (int32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
            int32_t idx = base_idx + i;
            اگر (idx >= GGML_MAX_N_THREADS) {
                // Just a cheaper modulo
                idx -= GGML_MAX_N_THREADS;
            }
            اگر (global_mask[idx]) {
                local_mask[idx] = 1;
                *iter = idx + 1;
                بازگشت;
            }
        }
    }
}

تهی ggml_threadpool_free(ساختار ggml_threadpool* threadpool) {
    اگر (!threadpool) بازگشت;

    ثابت صحیح n_threads = threadpool->n_threads;

#ifndef GGML_USE_OPENMP
    ساختار ggml_compute_state* workers = threadpool->workers;

    ggml_mutex_lock(&threadpool->mutex);

    threadpool->stop = true;
    threadpool->pause = false;

    ggml_cond_broadcast(&threadpool->cond);
    ggml_mutex_unlock(&threadpool->mutex);

    برای (صحیح j = 1; j < n_threads; j++) {
        int32_t rc = ggml_thread_join(workers[j].thrd, پوچ);
        GGML_ASSERT(rc == GGML_EXIT_SUCCESS || rc == GGML_EXIT_ABORTED);
        UNUSED(rc);
    }

    ggml_mutex_destroy(&threadpool->mutex);
    ggml_cond_destroy(&threadpool->cond);
#endif // GGML_USE_OPENMP

    ثابت size_t workers_size = اندازه(ساختار ggml_compute_state) * n_threads;
    ggml_aligned_free(threadpool->workers, workers_size);
    ggml_aligned_free(threadpool, اندازه(ساختار ggml_threadpool));
}

#ifndef GGML_USE_OPENMP
// pause/resume must be called under mutex
ایستا تهی ggml_threadpool_pause_locked(ساختار ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Pausing threadpool\n");
    threadpool->pause = true;
    ggml_cond_broadcast(&threadpool->cond);
}

ایستا تهی ggml_threadpool_resume_locked(ساختار ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Resuming threadpool\n");
    threadpool->pause = false;
    ggml_cond_broadcast(&threadpool->cond);
}
#endif

تهی ggml_threadpool_pause(ساختار ggml_threadpool * threadpool) {
#ifndef GGML_USE_OPENMP
    ggml_mutex_lock(&threadpool->mutex);
    اگر (!threadpool->pause) {
       ggml_threadpool_pause_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#وگرنه
    UNUSED(threadpool);
#endif
}

تهی ggml_threadpool_resume(ساختار ggml_threadpool * threadpool) {
#ifndef GGML_USE_OPENMP
    ggml_mutex_lock(&threadpool->mutex);
    اگر (threadpool->pause) {
       ggml_threadpool_resume_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#وگرنه
    UNUSED(threadpool);
#endif
}

ساختار ggml_cplan ggml_graph_plan(
          ثابت ساختار ggml_cgraph * cgraph,
                               صحیح   n_threads,
            ساختار ggml_threadpool * threadpool) {

    اگر (threadpool == پوچ) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
    }
    اگر (n_threads <= 0) {
        n_threads = threadpool ? threadpool->n_threads : GGML_DEFAULT_N_THREADS;
    }

#اگر defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
    // Emscripten without pthreads support can only use a single thread
    n_threads = 1;
#endif

#اگر defined(__wasi__)
    // WASI doesn't support parallelism yet
    n_threads = 1;
#endif

    size_t work_size = 0;

    ساختار ggml_cplan cplan;
    memset(&cplan, 0, اندازه(ساختار ggml_cplan));

    صحیح max_tasks = 1;

    // thread scheduling for the different operations + work buffer size estimation
    برای (صحیح i = 0; i < cgraph->n_nodes; i++) {
        ساختار ggml_tensor * گره = cgraph->nodes[i];

        ثابت صحیح n_tasks = ggml_get_n_tasks(گره, n_threads);

        max_tasks = MAX(max_tasks, n_tasks);

        size_t cur = 0;

        اگر (!ggml_cpu_extra_work_size(n_threads, گره, &cur)) {
            انتخاب (گره->op) {
                حالت GGML_OP_CPY:
                حالت GGML_OP_DUP:
                    {
                        اگر (ggml_is_quantized(گره->type) ||
                            // F16 -> BF16 and BF16 -> F16 copies go through intermediate F32
                            (گره->src[0]->type == GGML_TYPE_F16  && گره->src[1] && گره->src[1]->type == GGML_TYPE_BF16) ||
                            (گره->src[0]->type == GGML_TYPE_BF16 && گره->src[1] && گره->src[1]->type == GGML_TYPE_F16) ||
                            // conversion between F32 and I32
                            (گره->src[0]->type == GGML_TYPE_F32 && گره->src[1] && گره->src[1]->type == GGML_TYPE_I32) ||
                            (گره->src[0]->type == GGML_TYPE_I32 && گره->src[1] && گره->src[1]->type == GGML_TYPE_F32)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * گره->ne[0] * n_tasks;
                        }
                    } بشکن;
                حالت GGML_OP_ADD:
                حالت GGML_OP_ADD_ID:
                حالت GGML_OP_ADD1:
                    {
                        اگر (ggml_is_quantized(گره->src[0]->type)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * گره->src[0]->ne[0] * n_tasks;
                        }
                    } بشکن;
                حالت GGML_OP_ACC:
                    {
                        اگر (ggml_is_quantized(گره->src[0]->type)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * گره->src[1]->ne[0] * n_tasks;
                        }
                    } بشکن;
                حالت GGML_OP_شمار_EQUAL:
                    {
                        cur = ggml_type_size(گره->type)*n_tasks;
                    } بشکن;
                حالت GGML_OP_MUL_MAT:
                    {
                        ثابت شمارش ggml_type vec_dot_type = type_traits_cpu[گره->src[0]->type].vec_dot_type;
                        اگر (گره->src[1]->type != vec_dot_type) {
                            cur = ggml_row_size(vec_dot_type, ggml_nelements(گره->src[1]));
                        }
                        // Workspace for tiled (see tiled.h)
                        cur = GGML_PAD(cur, 64);
                        cur += ggml_tiled_wdata_size(n_tasks, گره);
                    } بشکن;
                حالت GGML_OP_MUL_MAT_ID:
                    {
                        cur = 0;
                        ثابت ساختار ggml_tensor * src0 = گره->src[0];
                        ثابت ساختار ggml_tensor * src1 = گره->src[1];
                        ثابت ساختار ggml_tensor * ids = گره->src[2];
                        ثابت شمارش ggml_type vec_dot_type = type_traits_cpu[src0->type].vec_dot_type;
                        ثابت صحیح n_as = src0->ne[2];
                        // src1
                        اگر (src1->type != vec_dot_type) {
                            cur += ggml_row_size(vec_dot_type, ggml_nelements(src1)) + اندازه(int64_t);
                        }
                        // matrix_row_counts
                        cur += n_as * اندازه(int64_t) + اندازه(int64_t);
                        // matrix_rows
                        cur += n_as*ids->ne[0]*ids->ne[1]*اندازه(ساختار mmid_row_mapping) + اندازه(int64_t);
                        // atomic_current_chunk
                        cur += CACHE_سطر_SIZE*n_as + CACHE_سطر_SIZE;
                        // Workspace for tiled (see tiled.h)
                        cur = GGML_PAD(cur, 64);
                        cur += ggml_tiled_wdata_size(n_tasks, گره);
                    } بشکن;
                حالت GGML_OP_OUT_PROD:
                    {
                        اگر (ggml_is_quantized(گره->src[0]->type) ||
                            گره->src[0]->type == GGML_TYPE_F16) {
                            cur = ggml_type_size(GGML_TYPE_F32) * گره->src[0]->ne[0] * n_tasks;
                        }
                    } بشکن;
                حالت GGML_OP_SET_ROWS:
                    {
                        اگر (گره->src[0]->type == GGML_TYPE_F16 && گره->type != GGML_TYPE_F16) {
                            cur = ggml_type_size(GGML_TYPE_F32) * گره->src[0]->ne[0] * n_tasks;
                        }
                    } بشکن;
                حالت GGML_OP_SOFT_MAX:
                حالت GGML_OP_ROPE:
                حالت GGML_OP_ROPE_BACK:
                    {
                        cur = ggml_type_size(GGML_TYPE_F32) * گره->ne[0] * n_tasks;
                    } بشکن;
                حالت GGML_OP_CONV_TRANSPOSE_1D:
                    {
                        GGML_ASSERT(گره->src[0]->ne[3] == 1);
                        GGML_ASSERT(گره->src[1]->ne[2] == 1);
                        GGML_ASSERT(گره->src[1]->ne[3] == 1);

                        ثابت int64_t ne00 = گره->src[0]->ne[0];  // K
                        ثابت int64_t ne01 = گره->src[0]->ne[1];  // Cout
                        ثابت int64_t ne02 = گره->src[0]->ne[2];  // Cin
                        ثابت int64_t ne10 = گره->src[1]->ne[0];  // L
                        ثابت int64_t ne11 = گره->src[1]->ne[1];  // Cin

                        اگر ((گره->src[0]->type == GGML_TYPE_F16 ||
                             گره->src[0]->type == GGML_TYPE_BF16) &&
                            گره->src[1]->type == GGML_TYPE_F32) {
                            cur += اندازه(ggml_fp16_t)*ne00*ne01*ne02;
                            cur += اندازه(ggml_fp16_t)*ne10*ne11;
                        } وگرنه اگر (گره->src[0]->type == GGML_TYPE_F32 &&
                                   گره->src[1]->type == GGML_TYPE_F32) {
                            cur += اندازه(اعشاری)*ne00*ne01*ne02;
                            cur += اندازه(اعشاری)*ne10*ne11;
                        } وگرنه {
                            GGML_ABORT("fatal error");
                        }
                    } بشکن;
                حالت GGML_OP_CONV_2D:
                حالت GGML_OP_CONV_3D:
                    {
                        cur = GGML_IM2COL_WORK_SIZE;
                    } بشکن;
                حالت GGML_OP_CONV_TRANSPOSE_2D:
                    {
                        ثابت int64_t ne00 = گره->src[0]->ne[0]; // W
                        ثابت int64_t ne01 = گره->src[0]->ne[1]; // H
                        ثابت int64_t ne02 = گره->src[0]->ne[2]; // Channels Out
                        ثابت int64_t ne03 = گره->src[0]->ne[3]; // Channels In

                        ثابت int64_t ne10 = گره->src[1]->ne[0]; // W
                        ثابت int64_t ne11 = گره->src[1]->ne[1]; // H
                        ثابت int64_t ne12 = گره->src[1]->ne[2]; // Channels In
                        ثابت int64_t ne13 = گره->src[1]->ne[3]; // Batch

                        GGML_ASSERT(گره->src[0]->type == GGML_TYPE_F16 || گره->src[0]->type == GGML_TYPE_F32);
                        GGML_ASSERT(گره->src[1]->type == GGML_TYPE_F32);

                        cur += ggml_type_size(گره->src[0]->type) * ne00 * ne01 * ne02 * ne03;
                        cur += ggml_type_size(گره->src[0]->type) * ne10 * ne11 * ne12 * ne13;

                    } بشکن;
                حالت GGML_OP_بالاترین_K:
                    {
                        cur += اندازه(int32_t)*گره->src[0]->ne[0]*n_tasks;
                    } بشکن;
                حالت GGML_OP_FLASH_ATTN_EXT:
                    {
                        ثابت int64_t neq2 = گره->src[0]->ne[2]; // number of query heads
                        ثابت int64_t DK = گره->src[1]->ne[0];
                        ثابت int64_t DV = گره->src[2]->ne[0];

                        // Tiled flash attention scratch (tile sizes defined in common.h)
                        // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
                        size_t prefill  = اندازه(اعشاری)*(GGML_FA_TILE_Q*DK + 2*GGML_FA_TILE_Q*GGML_FA_TILE_KV + GGML_FA_TILE_Q*DV + GGML_FA_TILE_KV*DV + GGML_FA_TILE_KV*DK)*n_tasks;

                        // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
                        // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
                        size_t n_chunks = n_tasks;
                        size_t decode   = اندازه(اعشاری)*(neq2*n_chunks*(2+DV) + n_tasks*(DK + 2*DV));

                        cur += MAX(prefill, decode);
                    } بشکن;
                حالت GGML_OP_FLASH_ATTN_BACK:
                    {
                        ثابت int64_t    D = گره->src[0]->ne[0];
                        ثابت int64_t ne11 = ggml_up(گره->src[1]->ne[1], GGML_SOFT_MAX_UNROLL);
                        ثابت int64_t mxDn = MAX(D, ne11) * 2; // *2 because of S and SM in ggml_compute_forward_flash_attn_back
                        اگر (گره->src[1]->type == GGML_TYPE_F32) {
                            cur  = اندازه(اعشاری)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += اندازه(اعشاری)*mxDn*n_tasks; // this is overestimated by x2
                        } وگرنه اگر (گره->src[1]->type == GGML_TYPE_F16) {
                            cur  = اندازه(اعشاری)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += اندازه(اعشاری)*mxDn*n_tasks; // this is overestimated by x2
                        } وگرنه اگر (گره->src[1]->type == GGML_TYPE_BF16) {
                            cur  = اندازه(اعشاری)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += اندازه(اعشاری)*mxDn*n_tasks; // this is overestimated by x2
                        }
                    } بشکن;

                حالت GGML_OP_CROSS_ENTROPY_LOSS:
                    {
                        cur = ggml_type_size(گره->type)*(n_tasks + گره->src[0]->ne[0]*n_tasks);
                    } بشکن;
                حالت GGML_OP_GATED_DELTA_NET:
                    {
                        ثابت int64_t S_v = گره->src[2]->ne[0];
                        ثابت int64_t K   = ggml_get_op_params_i32(گره, 0);
                        ثابت int64_t per_thread = S_v + (K > 1 ? S_v * S_v : 0);
                        cur = per_thread * اندازه(اعشاری) * n_tasks;
                    } بشکن;
                حالت GGML_OP_شمار:
                    {
                        GGML_ABORT("fatal error");
                    }
                حالت GGML_OP_LIGHTNING_INDEXER:
                    {
                        // temp buffer for dequantizing lightning indexer keys
                        ثابت int64_t ne10 = گره->src[1]->ne[0];
                        cur += اندازه(اعشاری)*ne10*n_tasks;
                    } بشکن;
                پیشفرض:
                    بشکن;
            }
        }

        work_size = MAX(work_size, cur);
    }

    اگر (work_size > 0) {
        work_size += CACHE_سطر_SIZE*(n_threads);
    }

    cplan.threadpool = threadpool;
    cplan.n_threads  = MIN(max_tasks, n_threads);
    cplan.work_size  = work_size;
    cplan.work_data  = پوچ;

    بازگشت cplan;
}


// Try to fuse the current node with subsequent nodes for better performance.
// Returns the number of nodes skipped by fusion (>=1), or 0 if no fusion was applied.
ایستا bool ggml_cpu_disable_fusion = false;  // initialized once in ggml_cpu_init(), read-only afterwards

ایستا صحیح ggml_cpu_try_fuse_ops(
        ثابت ساختار ggml_cgraph * cgraph,
        ثابت صحیح گره_n,
        ثابت ساختار ggml_compute_params * params,
        ثابت ساختار ggml_cplan * cplan) {

    اگر (ggml_cpu_disable_fusion || cplan->use_ref) {
        بازگشت 0;
    }

    ساختار ggml_tensor * گره = cgraph->nodes[گره_n];

    اگر (گره->op == GGML_OP_RMS_NORM) {
        // RMS_NORM + MUL fusion
        ثابت شمارش ggml_op fuse_ops[] = { GGML_OP_RMS_NORM, GGML_OP_MUL };
        اگر (ggml_can_fuse(cgraph, گره_n, fuse_ops, 2)) {
            ساختار ggml_tensor * mul_گره = cgraph->nodes[گره_n + 1];
            ثابت ساختار ggml_tensor * mul_w = (mul_گره->src[0] == گره)
                ? mul_گره->src[1] : mul_گره->src[0];
            اگر (گره->src[0]->type  == GGML_TYPE_F32 &&
                mul_گره->type      == GGML_TYPE_F32 &&
                mul_w->type         == GGML_TYPE_F32 &&
                mul_w->ne[0]        == گره->ne[0]   &&
                mul_w->nb[0]        == اندازه(اعشاری)) {

                ggml_compute_forward_rms_norm_mul_fused(params, گره, mul_گره);
                بازگشت 1;
            }
        }
    }

    بازگشت 0;
}

ایستا thread_ret_t ggml_graph_compute_thread(تهی * data) {
    ساختار ggml_compute_state * state = (ساختار ggml_compute_state *) data;
    ساختار ggml_threadpool    * tp    = state->threadpool;

    ثابت ساختار ggml_cgraph * cgraph = tp->cgraph;
    ثابت ساختار ggml_cplan  * cplan  = tp->cplan;

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(state->ith);
#وگرنه
    set_numa_thread_affinity(state->ith);
#endif

    ساختار ggml_compute_params params = {
        /*.ith        =*/ state->ith,
        /*.nth        =*/ atomic_load_explicit(&tp->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK,
        /*.wsize      =*/ cplan->work_size,
        /*.wdata      =*/ cplan->work_data,
        /*.threadpool =*/ tp,
        /*.use_ref    =*/ cplan->use_ref,
    };

#ifdef GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p\n", state->ith, (ثابت تهی *)cplan);
#وگرنه
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p last-graph %d\n", state->ith, (ثابت تهی *)cplan, state->last_graph);
#endif

    برای (صحیح گره_n = 0; گره_n < cgraph->n_nodes && atomic_load_explicit(&tp->abort, memory_order_relaxed) != گره_n; گره_n++) {
        ساختار ggml_tensor * گره = cgraph->nodes[گره_n];

        اگر (ggml_op_is_empty(گره->op)) {
            // skip NOPs
            ادامه;
        }

        اگر ((گره->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ادامه;
        }

        // TODO: move fused-op detection into ggml_graph_plan so fusion decisions are made once at planning time
        // Try fused ops, fall back to normal compute
        ثابت صحیح n_fused = ggml_cpu_try_fuse_ops(cgraph, گره_n, &params, cplan);
        اگر (n_fused > 0) {
            گره_n += n_fused;
        } وگرنه {
            ggml_compute_forward(&params, گره);
        }

        اگر (state->ith == 0 && cplan->abort_callback &&
                cplan->abort_callback(cplan->abort_callback_data)) {
            atomic_store_explicit(&tp->abort, گره_n + 1, memory_order_relaxed);
            tp->ec    = GGML_STATUS_ABORTED;
        }

        اگر (گره_n + 1 < cgraph->n_nodes) {
            ggml_barrier(state->threadpool);
        }
    }

#ifdef GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p\n", state->ith, (ثابت تهی *)cplan);
#وگرنه
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p last-graph %d\n", state->ith, (ثابت تهی *)cplan, state->last_graph);
#endif

    ggml_barrier(state->threadpool);

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(state->ith);
#endif

    بازگشت 0;
}

#ifndef GGML_USE_OPENMP

// check if thread is ready to proceed (exit from polling or sleeping)
// returns true if loops should exit, sets state->pending to indicate new work
ایستا درونخطی bool ggml_graph_compute_thread_ready(ساختار ggml_compute_state * state) {
    ساختار ggml_threadpool * threadpool = state->threadpool;

    اگر (state->pending || threadpool->stop || threadpool->pause) { بازگشت true; }

    // check for new graph/work
    صحیح n_graph   = atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed);
    صحیح n_threads = n_graph & GGML_THREADPOOL_N_THREADS_MASK;
    اگر (n_graph != state->last_graph) {
        state->pending    = (state->ith < n_threads);
        state->last_graph = n_graph;
        بازگشت true;
    }

    بازگشت false;
}

// sync thread state after polling
ایستا درونخطی تهی ggml_graph_compute_thread_sync(ساختار ggml_compute_state * state) {
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&state->threadpool->n_graph, 0, memory_order_seq_cst);
    #وگرنه
    atomic_thread_fence(memory_order_seq_cst);
    #endif
    UNUSED(state);
}

ایستا درونخطی bool ggml_graph_compute_poll_for_work(ساختار ggml_compute_state * state) {
    ساختار ggml_threadpool * threadpool = state->threadpool;

    // This seems to make 0 ... 100 a decent range for polling level across modern processors.
    // Perhaps, we can adjust it dynamically based on load and things.
    ثابت uint64_t n_rounds = 1024UL * 128 * threadpool->poll;

    برای (uint64_t i=0; !ggml_graph_compute_thread_ready(state) && i < n_rounds; i++) {
        // No new work. Keep polling.
        ggml_thread_cpu_relax();
    }

    بازگشت state->pending;
}

ایستا درونخطی bool ggml_graph_compute_check_for_work(ساختار ggml_compute_state * state) {
    ساختار ggml_threadpool * threadpool = state->threadpool;

    اگر (ggml_graph_compute_poll_for_work(state)) {
        ggml_graph_compute_thread_sync(state);
        بازگشت state->pending;
    }

    ggml_mutex_lock_shared(&threadpool->mutex);
    وقتی (!ggml_graph_compute_thread_ready(state)) {
        // No new work. Wait for the signal.
        GGML_PRINT_DEBUG("thread #%d waiting for work (sleeping)\n", state->ith);
        ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
    }
    ggml_mutex_unlock_shared(&threadpool->mutex);

    بازگشت state->pending;
}

ایستا thread_ret_t ggml_graph_compute_secondary_thread(تهی* data) {
    ساختار ggml_compute_state * state = (ساختار ggml_compute_state *) data;
    ساختار ggml_threadpool * threadpool = state->threadpool;

    ggml_thread_apply_priority(threadpool->prio);
    اگر (ggml_thread_cpumask_is_valid(state->cpumask)) {
        ggml_thread_apply_affinity(state->cpumask);
    }

    وقتی (true) {
        // Check if we need to sleep
        وقتی (threadpool->pause) {
            GGML_PRINT_DEBUG("thread #%d inside pause loop\n", state->ith);
            ggml_mutex_lock_shared(&threadpool->mutex);
            اگر (threadpool->pause) {
                ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
            }
            GGML_PRINT_DEBUG("thread #%d resuming after wait\n", state->ith);
            ggml_mutex_unlock_shared(&threadpool->mutex);
        }

        // This needs to be checked for after the cond_wait
        اگر (threadpool->stop) بشکن;

        // Check if there is new work
        // The main thread is the only one that can dispatch new work

        ggml_graph_compute_check_for_work(state);
        اگر (state->pending) {
            state->pending = false;
            ggml_graph_compute_thread(state);
        }
    }

    بازگشت (thread_ret_t) 0;
}

// Start processing new graph
ایستا تهی ggml_graph_compute_kickoff(ساختار ggml_threadpool * threadpool, صحیح n_threads)
{
    // Always take the mutex here because the worker threads are doing hybrid poll/wait

    ggml_mutex_lock(&threadpool->mutex);

    // Update the number of active threads and the graph count
    صحیح n_graph = atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed) >> GGML_THREADPOOL_N_THREADS_BITS;
    n_graph = ((n_graph + 1) << GGML_THREADPOOL_N_THREADS_BITS) | (n_threads & GGML_THREADPOOL_N_THREADS_MASK);

    GGML_PRINT_DEBUG("compute-kickoff: n_threads %d n_graph %d\n", n_threads, n_graph);

    // Indicate the graph is ready to be processed
    // We need the full seq-cst fence here because of the polling threads (used in thread_sync)
    atomic_store_explicit(&threadpool->n_graph, n_graph, memory_order_seq_cst);

    اگر (threadpool->pause) {
       // Update main thread prio and affinity to match the threadpool settings
       ggml_thread_apply_priority(threadpool->prio);
       اگر (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
           ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
       }

       // resume does cond broadcast
       ggml_threadpool_resume_locked(threadpool);
    } وگرنه {
       ggml_cond_broadcast(&threadpool->cond);
    }

    ggml_mutex_unlock(&threadpool->mutex);
}

#endif // GGML_USE_OPENMP

ایستا ساختار ggml_threadpool * ggml_threadpool_new_impl(
    ساختار ggml_threadpool_params * tpp,
               ساختار ggml_cgraph * cgraph,
                ساختار ggml_cplan * cplan) {

    ساختار ggml_threadpool * threadpool =
        ggml_aligned_malloc(اندازه(ساختار ggml_threadpool));
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
        threadpool->workers          = پوچ;
        threadpool->n_threads        = tpp->n_threads;
        threadpool->poll             = tpp->poll;
        threadpool->prio             = tpp->prio;
        threadpool->ec               = GGML_STATUS_SUCCESS;
    }

    // Allocate and init workers state
    ثابت size_t workers_size = اندازه(ساختار ggml_compute_state) * tpp->n_threads;
    ساختار ggml_compute_state * workers = ggml_aligned_malloc(workers_size);

    memset(workers, 0, workers_size);
    برای (صحیح j = 0; j < tpp->n_threads; j++) {
        workers[j].threadpool = threadpool;
        workers[j].ith        = j;
    }

    threadpool->workers = workers;

#ifdef GGML_USE_OPENMP
    int32_t cpumask_iter = 0;

    // Compute CPU masks for each thread
    برای (صحیح j = 0; j < tpp->n_threads; j++) {
        ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);
    }
#وگرنه // GGML_USE_OPENMP
    ggml_mutex_init(&threadpool->mutex);
    ggml_cond_init(&threadpool->cond);

    // Spin the threads for all workers, and update CPU placements.
    // Place the main thread last (towards the higher numbered CPU cores).

    int32_t cpumask_iter = 0;

    برای (صحیح j = 1; j < tpp->n_threads; j++) {
        ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);

        int32_t rc = ggml_thread_create(&workers[j].thrd, پوچ, ggml_graph_compute_secondary_thread, &workers[j]);
        GGML_ASSERT(rc == 0);
    }

    ggml_thread_cpumask_next(tpp->cpumask, workers[0].cpumask, tpp->strict_cpu, &cpumask_iter);

    اگر (!threadpool->pause) {
        // Update main thread prio and affinity at the start, otherwise we'll do it in resume
        ggml_thread_apply_priority(threadpool->prio);
        اگر (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
            ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
        }
    }
#endif // GGML_USE_OPENMP

    بازگشت threadpool;
}

ساختار ggml_threadpool * ggml_threadpool_new(ساختار ggml_threadpool_params * tpp) {
    بازگشت ggml_threadpool_new_impl(tpp, پوچ, پوچ);
}

شمارش ggml_status ggml_graph_compute(ساختار ggml_cgraph * cgraph, ساختار ggml_cplan * cplan) {
    ggml_cpu_init();

    GGML_ASSERT(cplan);
    GGML_ASSERT(cplan->n_threads > 0);
    GGML_ASSERT(cplan->work_size == 0 || cplan->work_data != پوچ);

    صحیح n_threads                               = cplan->n_threads;
    ساختار ggml_threadpool * threadpool = cplan->threadpool;

    bool disposable_threadpool = false;

    اگر (threadpool == پوچ) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
        disposable_threadpool = true;

        ساختار ggml_threadpool_params ttp = ggml_threadpool_params_default(n_threads);
        threadpool = ggml_threadpool_new_impl(&ttp, cgraph, cplan);
    } وگرنه {
        // Reset some of the parameters that need resetting
        // No worker threads should be accessing the parameters below at this stage
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->current_chunk    = 0;
        threadpool->abort            = -1;
        threadpool->ec               = GGML_STATUS_SUCCESS;
    }

#ifdef GGML_USE_OPENMP
    اگر (n_threads > 1) {
        #pragma omp parallel رقم_threads(n_threads)
        {
            #pragma omp single
            {
                // update the number of threads from the actual number of threads that we got from OpenMP
                n_threads = omp_get_رقم_threads();
                atomic_store_explicit(&threadpool->n_graph, n_threads, memory_order_relaxed);
            }

            // Apply thread CPU mask and priority
            صحیح ith = omp_get_thread_رقم();

            ggml_thread_apply_priority(threadpool->prio);
            اگر (ggml_thread_cpumask_is_valid(threadpool->workers[ith].cpumask)) {
                ggml_thread_apply_affinity(threadpool->workers[ith].cpumask);
            }
            ggml_graph_compute_thread(&threadpool->workers[ith]);
        }
    } وگرنه {
        atomic_store_explicit(&threadpool->n_graph, 1, memory_order_relaxed);
        ggml_graph_compute_thread(&threadpool->workers[0]);
    }
#وگرنه
    اگر (n_threads > threadpool->n_threads) {
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

    شمارش ggml_status ret = threadpool->ec;

    اگر (disposable_threadpool) {
        ggml_threadpool_free(threadpool);
    }

    بازگشت ret;
}

شمارش ggml_status ggml_graph_compute_with_ctx(ساختار ggml_context * ctx, ساختار ggml_cgraph * cgraph, صحیح n_threads) {
    ساختار ggml_cplan cplan = ggml_graph_plan(cgraph, n_threads, پوچ);

    cplan.work_data = (uint8_t *)ggml_new_buffer(ctx, cplan.work_size);

    بازگشت ggml_graph_compute(cgraph, &cplan);
}

تهی ggml_cpu_fp32_to_fp32(ثابت اعشاری * x, اعشاری * y, int64_t n) {
    memcpy(y, x, n * اندازه(اعشاری));
}

تهی ggml_cpu_fp32_to_fp16(ثابت اعشاری * x, ggml_fp16_t * y, int64_t n) {
    int64_t i = 0;
#اگر defined(__F16C__)
#اگر defined(__AVX512F__)
    برای (; i + 15 < n; i += 16) {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        __m256i y_vec = _mm512_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256((__m256i *)(y + i), y_vec);
    }
#endif
    برای (; i + 7 < n; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m128i y_vec = _mm256_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i *)(y + i), y_vec);
    }
    برای (; i + 3 < n; i += 4) {
        __m128 x_vec = _mm_loadu_ps(x + i);
        __m128i y_vec = _mm_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storel_epi64((__m128i *)(y + i), y_vec);
    }
#وگرنهاگر defined(__riscv_zvfh)
    برای (صحیح vl; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(&x[i], vl);
        vfloat16m1_t vy = __riscv_vfncvt_f_f_w_f16m1(vx, vl);
        __riscv_vse16_v_f16m1((_Float16 *)&y[i], vy, vl);
    }
#وگرنهاگر defined(__VXE__) || defined(__VXE2__)
    برای (; i + 7 < n; i += 8) {
        ثابت uint32x4_t v_yl = __lzs_f32cx4_to_f16(vec_xl(0, x + i + 0));
        ثابت uint32x4_t v_yh = __lzs_f32cx4_to_f16(vec_xl(0, x + i + 4));
        vec_xst(vec_pack(v_yl, v_yh), 0, (uint16_t *)(y + i));
    }
#endif
    برای (; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(x[i]);
    }
}

تهی ggml_cpu_fp16_to_fp32(ثابت ggml_fp16_t * x, اعشاری * y, int64_t n) {
    int64_t i = 0;
#اگر defined(__F16C__)
#اگر defined(__AVX512F__)
    برای (; i + 15 < n; i += 16) {
        __m256i x_vec = _mm256_loadu_si256((ثابت __m256i *)(x + i));
        __m512 y_vec = _mm512_cvtph_ps(x_vec);
        _mm512_storeu_ps(y + i, y_vec);
    }
#endif
    برای (; i + 7 < n; i += 8) {
        __m128i x_vec = _mm_loadu_si128((ثابت __m128i *)(x + i));
        __m256 y_vec = _mm256_cvtph_ps(x_vec);
        _mm256_storeu_ps(y + i, y_vec);
    }
    برای (; i + 3 < n; i += 4) {
        __m128i x_vec = _mm_loadl_epi64((ثابت __m128i *)(x + i));
        __m128 y_vec = _mm_cvtph_ps(x_vec);
        _mm_storeu_ps(y + i, y_vec);
    }

#وگرنهاگر defined(__riscv_v_intrinsic) && defined(__riscv_zvfhmin)
    // calculate step size
    ثابت صحیح epr = __riscv_vsetvlmax_e16m2();
    ثابت صحیح step = epr * 2;
    ثابت صحیح np = (n & ~(step - 1));

    // unroll by 2
    برای (; i < np; i += step) {
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((ثابت _Float16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vfloat16m2_t ax1 = __riscv_vle16_v_f16m2((ثابت _Float16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvt_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    صحیح vl;
    برای (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((ثابت _Float16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }

#endif

    برای (; i < n; ++i) {
        y[i] = GGML_CPU_FP16_TO_FP32(x[i]);
    }
}

تهی ggml_cpu_fp32_to_bf16(ثابت اعشاری * x, ggml_bf16_t * y, int64_t n) {
    int64_t i = 0;
    برای (; i < n; ++i) {
        y[i] = GGML_FP32_TO_BF16(x[i]);
    }
}

تهی ggml_cpu_fp32_to_i32(ثابت اعشاری * x, int32_t * y, int64_t n) {
    int64_t i = 0;
    برای (; i < n; ++i) {
        y[i] = x[i];
    }
}

تهی ggml_cpu_bf16_to_fp32(ثابت ggml_bf16_t * x, اعشاری * y, int64_t n) {
    int64_t i = 0;
#اگر defined(__AVX2__)
#اگر defined(__AVX512F__)
    برای (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i,
                        _mm512_castsi512_ps(
                            _mm512_slli_epi32(
                                _mm512_cvtepu16_epi32(
                                    _mm256_loadu_si256(
                                        (ثابت __m256i *)(x + i))),
                                16)));
    }
#endif
    برای (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i,
                        _mm256_castsi256_ps(
                            _mm256_slli_epi32(
                                _mm256_cvtepu16_epi32(
                                    _mm_loadu_si128(
                                        (ثابت __m128i *)(x + i))),
                                16)));
    }
#وگرنهاگر defined(__riscv_v_intrinsic) && defined(__riscv_zvfbfmin)
    // calculate step size
    ثابت صحیح epr = __riscv_vsetvlmax_e16m2();
    ثابت صحیح step = epr * 2;
    ثابت صحیح np = (n & ~(step - 1));

    // unroll by 2
    برای (; i < np; i += step) {
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((ثابت __bf16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vbfloat16m2_t ax1 = __riscv_vle16_v_bf16m2((ثابت __bf16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    صحیح vl;
    برای (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((ثابت __bf16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }
#endif
    برای (; i < n; i++) {
        y[i] = GGML_BF16_TO_FP32(x[i]);
    }
}

صحیح ggml_cpu_has_avx(تهی) {
#اگر defined(__AVX__)
    بازگشت 1;
#وگرنه
    بازگشت 0;
#endif
}

صحیح ggml_cpu_has_avx_vnni(تهی) {
#اگر defined(__AVXVNNI__)
    بازگشت 1;
#وگرنه
    بازگشت 0;
#endif
}

صحیح ggml_cpu_has_avx2(تهی) {
#اگر defined(__AVX2__)
    بازگشت 1;
#وگرنه
    بازگشت 0;
#endif
}

صحیح ggml_cpu_has_avx512(تهی) {
#اگر defined(__AVX512F__)
    بازگشت 1;
#وگرنه
    بازگشت 0;
#endif
}

صحیح ggml_cpu_has_avx512_vbmi(تهی) {
#اگر defined(__AVX512VBMI__)
    بازگشت 1;
#وگرنه
    بازگشت 0;
#endif
}

صحیح ggml_cpu_has_avx512_vnni(تهی) {
#اگر defined(__AVX512VNNI__)
    بازگشت 1;
#وگرنه
    بازگشت 0;
#endif
}

صحیح ggml_cpu_has_avx512_bf16(تهی) {
#اگر defined(__AVX512BF16__)
    بازگشت 1;
#وگرنه
    بازگشت 0;
#endif
}

صحیح ggml_cpu_has_amx_int8(تهی) {
#اگر defined(__AMX_INT8__)
    بازگشت 1;
#وگرنه
    بازگشت 0;
#endif
}

صحیح ggml_cpu_has_bmi2(تهی) {
#اگر defined(__BMI2__)
    بازگشت 1;
#وگرنه
    بازگشت 0;
#endif
}

صحیح ggml_cpu_has_fma(تهی) {
#اگر defined(__FMA__)
    بازگشت 1;
#وگرنه
    بازگشت 0;
#endif
}

صحیح ggml_cpu_has_arm_fma(تهی) {
#اگر defined(__ARM_FEATURE_FMA)
    بازگشت 1;
#وگرنه
    بازگشت 0;
#endif
}

صحیح ggml_cpu_has_riscv_v(تهی) {
#اگر defined(__riscv_v_intrinsic)
    بازگشت 1;
#وگرنه
    بازگشت 0;
#endif
}

صحیح ggml_cpu_get_rvv_vlen(تهی) {
#اگر defined(__riscv) && defined(__riscv_v_intrinsic)
    بازگشت ggml_riscv_arch_features.rvv_vlen;
#وگرنه
    بازگشت 0;
#endif
}

صحیح ggml_cpu_has_f16c(تهی) {
#اگر defined(__F16C__)
    بازگشت 1;
#وگرنه
    بازگشت 0;
#endif
}

صحیح ggml_cpu_has_fp16_va(تهی) {
#اگر defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    بازگشت 1;
#وگرنه
    بازگشت 0;
#endif
}

صحیح ggml_cpu_has_wasm_simd(تهی) {
#اگر defined(__wasm_simd128__)
    بازگشت 1;
#وگرنه
    بازگشت 0;
#endif
}

صحیح ggml_cpu_has_llamafile(تهی) {
#اگر defined(GGML_USE_LLAMAFILE)
    بازگشت 1;
#وگرنه
    بازگشت 0;
#endif
}

صحیح ggml_cpu_has_sse3(تهی) {
#اگر defined(__SSE3__)
    بازگشت 1;
#وگرنه
    بازگشت 0;
#endif
}

صحیح ggml_cpu_has_ssse3(تهی) {
#اگر defined(__SSSE3__)
    بازگشت 1;
#وگرنه
    بازگشت 0;
#endif
}

صحیح ggml_cpu_has_vsx(تهی) {
#اگر defined(__POWER9_VECTOR__)
    بازگشت 1;
#وگرنه
    بازگشت 0;
#endif
}

صحیح ggml_cpu_has_vxe(تهی) {
#اگر defined(__VXE__) || defined(__VXE2__)
    بازگشت 1;
#وگرنه
    بازگشت 0;
#endif
}

صحیح ggml_cpu_has_neon(تهی) {
#اگر defined(__ARM_ARCH) && defined(__ARM_NEON)
    بازگشت 1;
#وگرنه
    بازگشت 0;
#endif
}

صحیح ggml_cpu_has_dotprod(تهی) {
#اگر defined(__ARM_ARCH) && defined(__ARM_FEATURE_DOTPROD)
    بازگشت 1;
#وگرنه
    بازگشت 0;
#endif
}

صحیح ggml_cpu_has_sve(تهی) {
#اگر defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    بازگشت 1;
#وگرنه
    بازگشت 0;
#endif
}

صحیح ggml_cpu_has_matmul_int8(تهی) {
#اگر defined(__ARM_ARCH) && defined(__ARM_FEATURE_MATMUL_INT8)
    بازگشت 1;
#وگرنه
    بازگشت 0;
#endif
}

صحیح ggml_cpu_get_sve_cnt(تهی) {
#اگر defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    بازگشت ggml_arm_arch_features.sve_cnt;
#وگرنه
    بازگشت 0;
#endif
}

صحیح ggml_cpu_has_sme(تهی) {
#اگر defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME)
    بازگشت 1;
#وگرنه
    بازگشت 0;
#endif
}

صحیح ggml_cpu_has_sme2(تهی) {
#اگر defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME2)
    بازگشت 1;
#وگرنه
    بازگشت 0;
#endif
}

تهی ggml_cpu_init(تهی) {
    // needed to initialize ggml_time
    {
        ساختار ggml_init_params params = { 0, پوچ, false };
        ساختار ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    ggml_critical_section_start();

    ایستا bool is_first_call = true;

    اگر (is_first_call) {
        // initialize GELU, Quick GELU, SILU and EXP F32 tables
        {
            ثابت uint64_t t_start = ggml_time_us(); UNUSED(t_start);

            برای (صحیح i = 0; i < (1 << 16); ++i) {
                اجتماع {
                    uint16_t u16;
                    ggml_fp16_t fp16;
                } u = {i};
                اعشاری f = GGML_COMPUTE_FP16_TO_FP32(u.fp16);
                ggml_table_f32_f16[i] = f;
                ggml_table_gelu_f16[i] = GGML_CPU_FP32_TO_FP16(ggml_gelu_f32(f));
                ggml_table_gelu_quick_f16[i] = GGML_CPU_FP32_TO_FP16(ggml_gelu_quick_f32(f));
            }

            // initialize E8M0 half table (256 entries)
            برای (صحیح i = 0; i < (1 << 8); ++i) {
                ggml_table_f32_e8m0_half[i] = GGML_E8M0_TO_FP32_HALF(i);
            }

            // initialize UE4M3 table (256 entries)
            برای (صحیح i = 0; i < (1 << 8); ++i) {
                ggml_table_f32_ue4m3[i] = ggml_ue4m3_to_fp32(i);
            }

            ثابت uint64_t t_end = ggml_time_us(); UNUSED(t_end);

            GGML_PRINT_DEBUG("%s: GELU, Quick GELU, SILU and EXP tables initialized in %f ms\n", __func__, (t_end - t_start)/1000.0);

#ifdef GGML_USE_OPENMP
            //if (!getenv("OMP_WAIT_POLICY")) {
            //    // set the wait policy to active, so that OpenMP threads don't sleep
            //    setenv("OMP_WAIT_POLICY", "active", 0)
            //}

            اگر (!getenv("KMP_BLOCKTIME")) {
                // set the time to wait before sleeping a thread
                // this is less aggressive than setting the wait policy to active, but should achieve similar results in most cases
#ifdef _WIN32
                _putenv_s("KMP_BLOCKTIME", "200"); // 200ms
#وگرنه
                setenv("KMP_BLOCKTIME", "200", 0); // 200ms
#endif
            }
#endif
        }

#اگر defined(__ARM_ARCH)
        ggml_init_arm_arch_features();
#endif

#اگر defined(__riscv)
        ggml_init_riscv_arch_features();
#endif

        {
            ثابت نویسه * env = getenv("GGML_CPU_DISABLE_FUSION");
            ggml_cpu_disable_fusion = (env != پوچ && atoi(env) == 1);
        }

        is_first_call = false;
    }

    ggml_critical_section_end();
}
