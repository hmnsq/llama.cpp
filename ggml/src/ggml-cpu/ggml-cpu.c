#define _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#define _USE_MATH_DEFINES // For M_PI on MSVC

#ضمن "ggml-backend-impl.h"
#ضمن "ggml-backend.h"
#ضمن "traits.h"
#ضمن "ggml-cpu-impl.h"
#ضمن "ggml-impl.h"
#ضمن "quants.h"
#ضمن "ggml-threading.h"
#ضمن "unary-ops.h"
#ضمن "binary-ops.h"
#ضمن "vec.h"
#ضمن "ops.h"
#ضمن "ggml.h"
#ضمن "common.h"
#ضمن "tiled/tiled.h"

#𐤀𐤌 defined(_MSC_VER) || defined(__MINGW32__)
#ضمن <malloc.h> // using malloc.h with MSC/MINGW
#وإلاإذا !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#ضمن <alloca.h>
#endif

#ضمن <assert.h>
#ضمن <errno.h>
#ضمن <time.h>
#ضمن <math.h>
#ضمن <stdlib.h>
#ضمن <string.h>
#ضمن <stdint.h>
#ضمن <inttypes.h>
#ضمن <stdio.h>
#ضمن <float.h>
#ضمن <limits.h>
#ضمن <stdarg.h>
#ضمن <signal.h>
#𐤀𐤌 defined(__gnu_linux__)
#ضمن <syscall.h>
#endif

#ifdef GGML_USE_OPENMP
#ضمن <omp.h>
#endif

#𐤀𐤌 defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_MATMUL_INT8)
#undef GGML_USE_LLAMAFILE
#endif

#ifdef GGML_USE_LLAMAFILE
#ضمن "llamafile/sgemm.h"
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
#    include "spacemit/ime.h"
#endif

// Note: once we move threading into a separate C++ file
// will use std::hardware_destructive_interference_size instead of hardcoding it here
// and we'll use C++ attribute syntax.
#define GGML_CACHE_LINE  64

#𐤀𐤌 defined(__clang__) || defined(__GNUC__)
#define GGML_CACHE_ALIGN __attribute__((aligned(GGML_CACHE_LINE)))
#endif

#𐤀𐤌 defined(__has_feature)
#𐤀𐤌 __has_feature(thread_sanitizer)
#define GGML_TSAN_ENABLED 1
#endif
#وإلا  // __has_feature
#𐤀𐤌 defined(__SANITIZE_THREAD__)
#define GGML_TSAN_ENABLED 1
#endif
#endif // __has_feature

#define UNUSED GGML_UNUSED
#define SWAP(x, y, T) 𐤐𐤏𐤋 { T SWAP = x; (x) = y; (y) = SWAP; } طالما (0)

// precomputed f32 table for f16 (256 KB) (simd-mappings.h)
عائم ggml_table_f32_f16[1 << 16];

// precomputed f32 table for e8m0 half (1 KB) (simd-mappings.h)
عائم ggml_table_f32_e8m0_half[1 << 8];

// precomputed f32 table for ue4m3 (1 KB) (simd-mappings.h)
عائم ggml_table_f32_ue4m3[1 << 8];

#𐤀𐤌 defined(__ARM_ARCH)
بنية ggml_arm_arch_features_type {
    صحيح sve_cnt;
} ggml_arm_arch_features = { 0 };
#endif

#𐤀𐤌 defined(__riscv)
بنية ggml_riscv_arch_features_type {
    صحيح rvv_vlen;
} ggml_riscv_arch_features = { 0 };
#endif

#𐤀𐤌 defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#ضمن <windows.h>

#𐤀𐤌 defined(_MSC_VER) && !defined(__clang__)
#define GGML_CACHE_ALIGN __declspec(align(GGML_CACHE_LINE))

تعريفنوع متطاير LONG atomic_int;
تعريفنوع atomic_int atomic_bool;
تعريفنوع atomic_int atomic_flag;

#define ATOMIC_FLAG_INIT 0

تعريفنوع تعداد {
    memory_order_relaxed,
    memory_order_consume,
    memory_order_acquire,
    memory_order_release,
    memory_order_acq_rel,
    memory_order_seq_cst
} memory_order;

ساكن فراغ atomic_store(atomic_int * ptr, LONG val) {
    InterlockedExchange(ptr, val);
}
ساكن فراغ atomic_store_explicit(atomic_int * ptr, LONG val, memory_order mo) {
    // TODO: add support for explicit memory order
    InterlockedExchange(ptr, val);
}
ساكن LONG atomic_load(atomic_int * ptr) {
    أرجع InterlockedCompareExchange(ptr, 0, 0);
}
ساكن LONG atomic_load_explicit(atomic_int * ptr, memory_order mo) {
    // TODO: add support for explicit memory order
    أرجع InterlockedCompareExchange(ptr, 0, 0);
}
ساكن LONG atomic_fetch_add(atomic_int * ptr, LONG inc) {
    أرجع InterlockedExchangeAdd(ptr, inc);
}
ساكن LONG atomic_fetch_add_explicit(atomic_int * ptr, LONG inc, memory_order mo) {
    // TODO: add support for explicit memory order
    أرجع InterlockedExchangeAdd(ptr, inc);
}
ساكن atomic_bool atomic_flag_test_and_set(atomic_flag * ptr) {
    أرجع InterlockedExchange(ptr, 1);
}
ساكن فراغ atomic_flag_clear(atomic_flag * ptr) {
    InterlockedExchange(ptr, 0);
}
ساكن فراغ atomic_thread_fence(memory_order mo) {
    MemoryBarrier();
}
#وإلا // clang
#ضمن <stdatomic.h>
#endif

تعريفنوع HANDLE pthread_t;

تعريفنوع DWORD thread_ret_t;
ساكن صحيح pthread_create(pthread_t * out, فراغ * unused, thread_ret_t(*func)(فراغ *), فراغ * arg) {
    (فراغ) unused;
    HANDLE handle = CreateThread(عدم, 0, (LPTHREAD_START_ROUTINE) func, arg, 0, عدم);
    𐤀𐤌 (handle == عدم)
    {
        أرجع EAGAIN;
    }

    *out = handle;
    أرجع 0;
}

ساكن صحيح pthread_join(pthread_t thread, فراغ * unused) {
    (فراغ) unused;
    صحيح ret = (صحيح) WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    أرجع ret;
}

ساكن صحيح sched_yield (فراغ) {
    Sleep (0);
    أرجع 0;
}
#وإلا

#ضمن <pthread.h>
#ضمن <stdatomic.h>
#ضمن <sched.h>
#𐤀𐤌 defined(__FreeBSD__)
#ضمن <pthread_np.h>
#endif

تعريفنوع فراغ * thread_ret_t;

#ضمن <sys/types.h>
#ضمن <sys/stat.h>
#ضمن <unistd.h>

#endif

تعريفنوع pthread_t ggml_thread_t;

#define GGML_THREADPOOL_N_THREADS_MASK (0xffffU)
#define GGML_THREADPOOL_N_THREADS_BITS (16)

#𐤀𐤌 defined(__APPLE__)
#ضمن <unistd.h>
#ضمن <mach/mach.h>
#ضمن <TargetConditionals.h>
#endif

ساكن ثابت بنية ggml_type_traits_cpu type_traits_cpu[GGML_TYPE_COUNT] = {
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
#𐤀𐤌 defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#وإلا
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q4_1] = {
        .from_float               = quantize_row_q4_1,
        .vec_dot                  = ggml_vec_dot_q4_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
#𐤀𐤌 defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#وإلا
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
#𐤀𐤌 defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#وإلا
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
#𐤀𐤌 defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#وإلا
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
#𐤀𐤌 defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#وإلا
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_IQ2_XXS] = {
        .from_float               = عدم,
        .vec_dot                  = ggml_vec_dot_iq2_xxs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_XS] = {
        .from_float               = عدم,
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
        .from_float               = عدم,
        .vec_dot                  = ggml_vec_dot_iq1_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_M] = {
        .from_float               = عدم,
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

ثابت بنية ggml_type_traits_cpu * ggml_get_type_traits_cpu(تعداد ggml_type type) {
    أرجع &type_traits_cpu[type];
}

//
// Threading defs
//

تعريفنوع pthread_t          ggml_thread_t;

#𐤀𐤌 defined(_WIN32)

تعريفنوع CONDITION_VARIABLE ggml_cond_t;
تعريفنوع SRWLOCK            ggml_mutex_t;

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

#وإلا

تعريفنوع pthread_cond_t     ggml_cond_t;
تعريفنوع pthread_mutex_t    ggml_mutex_t;

#define ggml_mutex_init(m)          pthread_mutex_init(m, عدم)
#define ggml_mutex_destroy(m)       pthread_mutex_destroy(m)
#define ggml_mutex_lock(m)          pthread_mutex_lock(m)
#define ggml_mutex_unlock(m)        pthread_mutex_unlock(m)
#define ggml_mutex_lock_shared(m)   pthread_mutex_lock(m)
#define ggml_mutex_unlock_shared(m) pthread_mutex_unlock(m)

#define ggml_lock_init(x)    UNUSED(x)
#define ggml_lock_destroy(x) UNUSED(x)
#𐤀𐤌 defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64) && !defined(_M_ARM64EC))
#define ggml_lock_lock(x)    _mm_pause()
#وإلا
#define ggml_lock_lock(x)    UNUSED(x)
#endif
#define ggml_lock_unlock(x)  UNUSED(x)

#define GGML_LOCK_INITIALIZER 0
#define ggml_cond_init(c)      pthread_cond_init(c, عدم)
#define ggml_cond_destroy(c)   pthread_cond_destroy(c)
#define ggml_cond_wait(c, m)   pthread_cond_wait(c, m)
#define ggml_cond_broadcast(c) pthread_cond_broadcast(c)

#define ggml_thread_create pthread_create
#define ggml_thread_join   pthread_join

#endif

// Threadpool def
بنية ggml_threadpool {
    ggml_mutex_t mutex;       // mutex for cond.var
    ggml_cond_t  cond;        // cond.var for waiting for new work

    بنية ggml_cgraph * cgraph;
    بنية ggml_cplan  * cplan;

    // synchronization primitives
    atomic_int n_graph;       // updated when there is work to be done (i.e each graph) holds graph and active thread counts.
    atomic_int GGML_CACHE_ALIGN n_barrier;
    atomic_int GGML_CACHE_ALIGN n_barrier_passed;
    atomic_int GGML_CACHE_ALIGN current_chunk; // currently processing chunk during Mat_Mul, shared between all the threads.

    // these are atomic as an annotation for thread-sanitizer
    atomic_bool stop;         // Used for stopping the threadpool altogether
    atomic_bool pause;        // Used for pausing the threadpool or individual threads
    atomic_int  abort;        // Used for aborting processing of a graph

    بنية ggml_compute_state * workers;   // per thread state
    صحيح          n_threads;   // Number of threads in the pool
    int32_t      prio;        // Scheduling priority
    uint32_t     poll;        // Polling level (0 - no polling)

    تعداد ggml_status ec;
};

// Per-thread state
بنية ggml_compute_state {
#ifndef GGML_USE_OPENMP
    ggml_thread_t thrd;
    صحيح  last_graph;
    bool pending;
#endif
    bool cpumask[GGML_MAX_N_THREADS];
    بنية ggml_threadpool * threadpool;
    صحيح ith;
};

// Helpers for polling loops
#𐤀𐤌 defined(__aarch64__) && ( defined(__clang__) || defined(__GNUC__) )
ساكن مضمن فراغ ggml_thread_cpu_relax(فراغ) {
    __asm__ متطاير("yield" ::: "memory");
}
#وإلاإذا defined(__x86_64__)
ساكن مضمن فراغ ggml_thread_cpu_relax(فراغ) {
    _mm_pause();
}
#وإلاإذا defined(__riscv)
ساكن مضمن فراغ ggml_thread_cpu_relax(فراغ) {
    #ifdef __riscv_zihintpause
        __asm__ __volatile__ ("pause");
    #وإلا
        /* Encoding of the pause instruction */
        __asm__ __volatile__ (".4byte 0x100000F");
    #endif
}
#وإلا
ساكن مضمن فراغ ggml_thread_cpu_relax(فراغ) {;}
#endif

//
// NUMA support
//

#define GGML_NUMA_MAX_NODES 8
#define GGML_NUMA_MAX_CPUS 512

بنية ggml_numa_node {
    uint32_t cpus[GGML_NUMA_MAX_CPUS]; // hardware threads on this node
    uint32_t n_cpus;
};

بنية ggml_numa_nodes {
    تعداد ggml_numa_strategy numa_strategy;
    بنية ggml_numa_node nodes[GGML_NUMA_MAX_NODES];
    uint32_t n_nodes;
    uint32_t total_cpus; // hardware threads on system
    uint32_t current_node; // node on which main process is execting
#𐤀𐤌 defined(__gnu_linux__)
    cpu_set_t cpuset; // cpuset from numactl
#وإلا
    uint32_t cpuset; // no NUMA support outside of Linux at this time. Use a portable datatype
#endif
};

//
// ggml state
//

بنية ggml_state {
    بنية ggml_numa_nodes numa;
};

ساكن بنية ggml_state g_state = {0};

فراغ ggml_barrier(بنية ggml_threadpool * tp) {
    صحيح n_threads = atomic_load_explicit(&tp->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK;
    𐤀𐤌 (n_threads == 1) {
        أرجع;
    }

#ifdef GGML_USE_OPENMP
    #pragma omp barrier
#وإلا
    صحيح n_passed = atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed);

    // enter barrier (full seq-cst fence)
    صحيح n_barrier = atomic_fetch_add_explicit(&tp->n_barrier, 1, memory_order_seq_cst);

    𐤀𐤌 (n_barrier == (n_threads - 1)) {
        // last thread
        atomic_store_explicit(&tp->n_barrier, 0, memory_order_relaxed);

        // exit barrier (full seq-cst fence)
        atomic_fetch_add_explicit(&tp->n_barrier_passed, 1, memory_order_seq_cst);
        أرجع;
    }

    // wait for other threads
    طالما (atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed) == n_passed) {
        ggml_thread_cpu_relax();
    }

    // exit barrier (full seq-cst fence)
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&tp->n_barrier_passed, 0, memory_order_seq_cst);
    #وإلا
    atomic_thread_fence(memory_order_seq_cst);
    #endif
#endif
}

فراغ ggml_threadpool_chunk_set(بنية ggml_threadpool * tp, صحيح value) {
    atomic_store_explicit(&tp->current_chunk, value, memory_order_relaxed);
}

صحيح ggml_threadpool_chunk_add(بنية ggml_threadpool * tp, صحيح value) {
    أرجع atomic_fetch_add_explicit(&tp->current_chunk, value, memory_order_relaxed);
}

#𐤀𐤌 defined(__gnu_linux__)
ساكن cpu_set_t ggml_get_numa_affinity(فراغ) {
    cpu_set_t cpuset;
    pthread_t thread;
    thread = pthread_self();
    CPU_ZERO(&cpuset);
    pthread_getaffinity_np(thread, حجم(cpu_set_t), &cpuset);
    أرجع cpuset;
}
#وإلا
ساكن uint32_t ggml_get_numa_affinity(فراغ) {
    أرجع 0; // no NUMA support
}
#endif

فراغ ggml_numa_init(تعداد ggml_numa_strategy numa_flag) {
    𐤀𐤌 (g_state.numa.n_nodes > 0) {
        fprintf(stderr, "ggml_numa_init: NUMA already initialized\n");

        أرجع;
    }

#𐤀𐤌 defined(__gnu_linux__)
    بنية stat st;
    حرف path[256];
    صحيح rv;

    // set numa scheme
    g_state.numa.numa_strategy = numa_flag;

    GGML_PRINT_DEBUG("numa strategy %u\n",g_state.numa.numa_strategy);

    g_state.numa.cpuset = ggml_get_numa_affinity();

    // enumerate nodes
    طالما (g_state.numa.n_nodes < GGML_NUMA_MAX_NODES) {
        rv = snprintf(path, حجم(path), "/sys/devices/system/node/node%u", g_state.numa.n_nodes);
        GGML_ASSERT(rv > 0 && (unsigned)rv < حجم(path));
        𐤀𐤌 (stat(path, &st) != 0) { اكسر; }
        ++g_state.numa.n_nodes;
    }

    // enumerate CPUs
    طالما (g_state.numa.total_cpus < GGML_NUMA_MAX_CPUS) {
        rv = snprintf(path, حجم(path), "/sys/devices/system/cpu/cpu%u", g_state.numa.total_cpus);
        GGML_ASSERT(rv > 0 && (unsigned)rv < حجم(path));
        𐤀𐤌 (stat(path, &st) != 0) { اكسر; }
        ++g_state.numa.total_cpus;
    }

    GGML_PRINT_DEBUG("found %u numa nodes, %u CPUs\n", g_state.numa.n_nodes, g_state.numa.total_cpus);

    // figure out which node we're on
    uint current_cpu;
    صحيح getcpu_ret = 0;
#𐤀𐤌 __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ > 33) || defined(__COSMOPOLITAN__)
    getcpu_ret = getcpu(&current_cpu, &g_state.numa.current_node);
#وإلا
    // old glibc doesn't have a wrapper for this call. Fall back on direct syscall
#   𐤀𐤌 !defined(SYS_getcpu) && defined(SYS_get_cpu)
#       define SYS_getcpu SYS_get_cpu // some older glibc versions use this name
#   endif
    getcpu_ret = syscall(SYS_getcpu, &current_cpu, &g_state.numa.current_node);
#endif

    𐤀𐤌 (g_state.numa.n_nodes < 1 || g_state.numa.total_cpus < 1 || getcpu_ret != 0) {
        g_state.numa.n_nodes = 0;
        أرجع;
    }

    GGML_PRINT_DEBUG("found our process on numa node %u, CPU %u\n", g_state.numa.current_node, current_cpu);

    لكل (uint32_t n = 0; n < g_state.numa.n_nodes; ++n) {
        بنية ggml_numa_node * node = &g_state.numa.nodes[n];
        GGML_PRINT_DEBUG("CPUs on node %u:", n);
        node->n_cpus = 0;
        لكل (uint32_t c = 0; c < g_state.numa.total_cpus; ++c) {
            rv = snprintf(path, حجم(path), "/sys/devices/system/node/node%u/cpu%u", n, c);
            GGML_ASSERT(rv > 0 && (unsigned)rv < حجم(path));
            𐤀𐤌 (stat(path, &st) == 0) {
                node->cpus[node->n_cpus++] = c;
                GGML_PRINT_DEBUG(" %u", c);
            }
        }
        GGML_PRINT_DEBUG("\n");
    }

    𐤀𐤌 (ggml_is_numa()) {
        FILE *fptr = fopen("/proc/sys/kernel/numa_balancing", "r");
        𐤀𐤌 (fptr != عدم) {
            حرف buf[42];
            𐤀𐤌 (fgets(buf, حجم(buf), fptr) && strncmp(buf, "0\n", حجم(buf)) != 0) {
                GGML_LOG_WARN("/proc/sys/kernel/numa_balancing is enabled, this has been observed to impair performance\n");
            }
            fclose(fptr);
        }
    }
#وإلا
    UNUSED(numa_flag);
    // TODO
#endif
}

bool ggml_is_numa(فراغ) {
    أرجع g_state.numa.n_nodes > 1;
}

#𐤀𐤌 defined(__ARM_ARCH)
#𐤀𐤌 defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
#ضمن <arm_sve.h>
ساكن فراغ ggml_init_arm_arch_features(فراغ) {
    ggml_arm_arch_features.sve_cnt = svcntb();
}
#وإلا
ساكن فراغ ggml_init_arm_arch_features(فراغ) {}
#endif
#endif // __ARM_ARCH

#𐤀𐤌 defined(__riscv) && defined(__riscv_v_intrinsic)
#ضمن <riscv_vector.h>
ساكن فراغ ggml_init_riscv_arch_features(فراغ) {
    ggml_riscv_arch_features.rvv_vlen = __riscv_vlenb();
}
#وإلا
ساكن فراغ ggml_init_riscv_arch_features(فراغ) {}
#endif

بنية ggml_tensor * ggml_new_i32(بنية ggml_context * ctx, int32_t value) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    بنية ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

    ggml_set_i32(result, value);

    أرجع result;
}

بنية ggml_tensor * ggml_new_f32(بنية ggml_context * ctx, عائم value) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    بنية ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    ggml_set_f32(result, value);

    أرجع result;
}

بنية ggml_tensor * ggml_set_i32 (بنية ggml_tensor * tensor, int32_t value) {
    ثابت صحيح n     = ggml_nrows(tensor);
    ثابت صحيح nc    = tensor->ne[0];
    ثابت size_t n1 = tensor->nb[1];

    حرف * ثابت data = tensor->data;

    بدل (tensor->type) {
        حالة GGML_TYPE_I8:
            {
                أكد(tensor->nb[0] == حجم(int8_t));
                لكل (صحيح i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), value);
                }
            } اكسر;
        حالة GGML_TYPE_I16:
            {
                أكد(tensor->nb[0] == حجم(int16_t));
                لكل (صحيح i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), value);
                }
            } اكسر;
        حالة GGML_TYPE_I32:
            {
                أكد(tensor->nb[0] == حجم(int32_t));
                لكل (صحيح i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), value);
                }
            } اكسر;
        حالة GGML_TYPE_F16:
            {
                أكد(tensor->nb[0] == حجم(ggml_fp16_t));
                لكل (صحيح i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), GGML_CPU_FP32_TO_FP16(value));
                }
            } اكسر;
        حالة GGML_TYPE_BF16:
            {
                أكد(tensor->nb[0] == حجم(ggml_fp16_t));
                لكل (صحيح i = 0; i < n; i++) {
                    ggml_vec_set_bf16(nc, (ggml_bf16_t *)(data + i*n1), GGML_FP32_TO_BF16(value));
                }
            } اكسر;
        حالة GGML_TYPE_F32:
            {
                أكد(tensor->nb[0] == حجم(عائم));
                لكل (صحيح i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (عائم *)(data + i*n1), value);
                }
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }

    أرجع tensor;
}

بنية ggml_tensor * ggml_set_f32(بنية ggml_tensor * tensor, عائم value) {
    ثابت صحيح n     = ggml_nrows(tensor);
    ثابت صحيح nc    = tensor->ne[0];
    ثابت size_t n1 = tensor->nb[1];

    حرف * ثابت data = tensor->data;

    بدل (tensor->type) {
        حالة GGML_TYPE_I8:
            {
                أكد(tensor->nb[0] == حجم(int8_t));
                لكل (صحيح i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), value);
                }
            } اكسر;
        حالة GGML_TYPE_I16:
            {
                أكد(tensor->nb[0] == حجم(int16_t));
                لكل (صحيح i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), value);
                }
            } اكسر;
        حالة GGML_TYPE_I32:
            {
                أكد(tensor->nb[0] == حجم(int32_t));
                لكل (صحيح i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), value);
                }
            } اكسر;
        حالة GGML_TYPE_F16:
            {
                أكد(tensor->nb[0] == حجم(ggml_fp16_t));
                لكل (صحيح i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), GGML_CPU_FP32_TO_FP16(value));
                }
            } اكسر;
        حالة GGML_TYPE_BF16:
            {
                أكد(tensor->nb[0] == حجم(ggml_bf16_t));
                لكل (صحيح i = 0; i < n; i++) {
                    ggml_vec_set_bf16(nc, (ggml_bf16_t *)(data + i*n1), GGML_FP32_TO_BF16(value));
                }
            } اكسر;
        حالة GGML_TYPE_F32:
            {
                أكد(tensor->nb[0] == حجم(عائم));
                لكل (صحيح i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (عائم *)(data + i*n1), value);
                }
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }

    أرجع tensor;
}

int32_t ggml_get_i32_1d(ثابت بنية ggml_tensor * tensor, صحيح i) {
    𐤀𐤌 (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        أرجع ggml_get_i32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    بدل (tensor->type) {
        حالة GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == حجم(int8_t));
                أرجع ((int8_t *)(tensor->data))[i];
            }
        حالة GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == حجم(int16_t));
                أرجع ((int16_t *)(tensor->data))[i];
            }
        حالة GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == حجم(int32_t));
                أرجع ((int32_t *)(tensor->data))[i];
            }
        حالة GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == حجم(ggml_fp16_t));
                أرجع GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *)(tensor->data))[i]);
            }
        حالة GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == حجم(ggml_bf16_t));
                أرجع GGML_BF16_TO_FP32(((ggml_bf16_t *)(tensor->data))[i]);
            }
        حالة GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == حجم(عائم));
                أرجع ((عائم *)(tensor->data))[i];
            }
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

فراغ ggml_set_i32_1d(ثابت بنية ggml_tensor * tensor, صحيح i, int32_t value) {
    𐤀𐤌 (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_set_i32_nd(tensor, id[0], id[1], id[2], id[3], value);
        أرجع;
    }
    بدل (tensor->type) {
        حالة GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == حجم(int8_t));
                ((int8_t *)(tensor->data))[i] = value;
            } اكسر;
        حالة GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == حجم(int16_t));
                ((int16_t *)(tensor->data))[i] = value;
            } اكسر;
        حالة GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == حجم(int32_t));
                ((int32_t *)(tensor->data))[i] = value;
            } اكسر;
        حالة GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == حجم(ggml_fp16_t));
                ((ggml_fp16_t *)(tensor->data))[i] = GGML_CPU_FP32_TO_FP16(value);
            } اكسر;
        حالة GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == حجم(ggml_bf16_t));
                ((ggml_bf16_t *)(tensor->data))[i] = GGML_FP32_TO_BF16(value);
            } اكسر;
        حالة GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == حجم(عائم));
                ((عائم *)(tensor->data))[i] = value;
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

int32_t ggml_get_i32_nd(ثابت بنية ggml_tensor * tensor, صحيح i0, صحيح i1, صحيح i2, صحيح i3) {
    فراغ * data   = (حرف *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    بدل (tensor->type) {
        حالة GGML_TYPE_I8:
            أرجع ((int8_t *) data)[0];
        حالة GGML_TYPE_I16:
            أرجع ((int16_t *) data)[0];
        حالة GGML_TYPE_I32:
            أرجع ((int32_t *) data)[0];
        حالة GGML_TYPE_F16:
            أرجع GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *) data)[0]);
        حالة GGML_TYPE_BF16:
            أرجع GGML_BF16_TO_FP32(((ggml_bf16_t *) data)[0]);
        حالة GGML_TYPE_F32:
            أرجع ((عائم *) data)[0];
        افتراضي:
            GGML_ABORT("fatal error");
    }
}

فراغ ggml_set_i32_nd(ثابت بنية ggml_tensor * tensor, صحيح i0, صحيح i1, صحيح i2, صحيح i3, int32_t value) {
    فراغ * data   = (حرف *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    بدل (tensor->type) {
        حالة GGML_TYPE_I8:
            {
                ((int8_t *)(data))[0] = value;
            } اكسر;
        حالة GGML_TYPE_I16:
            {
                ((int16_t *)(data))[0] = value;
            } اكسر;
        حالة GGML_TYPE_I32:
            {
                ((int32_t *)(data))[0] = value;
            } اكسر;
        حالة GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(data))[0] = GGML_CPU_FP32_TO_FP16(value);
            } اكسر;
        حالة GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(data))[0] = GGML_FP32_TO_BF16(value);
            } اكسر;
        حالة GGML_TYPE_F32:
            {
                ((عائم *)(data))[0] = value;
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

عائم ggml_get_f32_1d(ثابت بنية ggml_tensor * tensor, صحيح i) {
    𐤀𐤌 (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        أرجع ggml_get_f32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    بدل (tensor->type) {
        حالة GGML_TYPE_I8:
            {
                أرجع ((int8_t *)(tensor->data))[i];
            }
        حالة GGML_TYPE_I16:
            {
                أرجع ((int16_t *)(tensor->data))[i];
            }
        حالة GGML_TYPE_I32:
            {
                أرجع ((int32_t *)(tensor->data))[i];
            }
        حالة GGML_TYPE_F16:
            {
                أرجع GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *)(tensor->data))[i]);
            }
        حالة GGML_TYPE_BF16:
            {
                أرجع GGML_BF16_TO_FP32(((ggml_bf16_t *)(tensor->data))[i]);
            }
        حالة GGML_TYPE_F32:
            {
                أرجع ((عائم *)(tensor->data))[i];
            }
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

فراغ ggml_set_f32_1d(ثابت بنية ggml_tensor * tensor, صحيح i, عائم value) {
    𐤀𐤌 (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_set_f32_nd(tensor, id[0], id[1], id[2], id[3], value);
        أرجع;
    }
    بدل (tensor->type) {
        حالة GGML_TYPE_I8:
            {
                ((int8_t *)(tensor->data))[i] = value;
            } اكسر;
        حالة GGML_TYPE_I16:
            {
                ((int16_t *)(tensor->data))[i] = value;
            } اكسر;
        حالة GGML_TYPE_I32:
            {
                ((int32_t *)(tensor->data))[i] = value;
            } اكسر;
        حالة GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(tensor->data))[i] = GGML_CPU_FP32_TO_FP16(value);
            } اكسر;
        حالة GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(tensor->data))[i] = GGML_FP32_TO_BF16(value);
            } اكسر;
        حالة GGML_TYPE_F32:
            {
                ((عائم *)(tensor->data))[i] = value;
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

عائم ggml_get_f32_nd(ثابت بنية ggml_tensor * tensor, صحيح i0, صحيح i1, صحيح i2, صحيح i3) {
    فراغ * data   = (حرف *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    بدل (tensor->type) {
        حالة GGML_TYPE_I8:
            أرجع ((int8_t *) data)[0];
        حالة GGML_TYPE_I16:
            أرجع ((int16_t *) data)[0];
        حالة GGML_TYPE_I32:
            أرجع ((int32_t *) data)[0];
        حالة GGML_TYPE_F16:
            أرجع GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *) data)[0]);
        حالة GGML_TYPE_BF16:
            أرجع GGML_BF16_TO_FP32(((ggml_bf16_t *) data)[0]);
        حالة GGML_TYPE_F32:
            أرجع ((عائم *) data)[0];
        افتراضي:
            GGML_ABORT("fatal error");
    }
}

فراغ ggml_set_f32_nd(ثابت بنية ggml_tensor * tensor, صحيح i0, صحيح i1, صحيح i2, صحيح i3, عائم value) {
    فراغ * data   = (حرف *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    بدل (tensor->type) {
        حالة GGML_TYPE_I8:
            {
                ((int8_t *)(data))[0] = value;
            } اكسر;
        حالة GGML_TYPE_I16:
            {
                ((int16_t *)(data))[0] = value;
            } اكسر;
        حالة GGML_TYPE_I32:
            {
                ((int32_t *)(data))[0] = value;
            } اكسر;
        حالة GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(data))[0] = GGML_CPU_FP32_TO_FP16(value);
            } اكسر;
        حالة GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(data))[0] = GGML_FP32_TO_BF16(value);
            } اكسر;
        حالة GGML_TYPE_F32:
            {
                ((عائم *)(data))[0] = value;
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

////////////////////////////////////////////////////////////////////////////////

// ggml_compute_forward_mul_mat

ساكن فراغ ggml_compute_forward_mul_mat_one_chunk(
    ثابت بنية ggml_compute_params * params,
    بنية ggml_tensor * dst,
    ثابت تعداد ggml_type type,
    ثابت int64_t num_rows_per_vec_dot,
    ثابت int64_t ir0_start,
    ثابت int64_t ir0_end,
    ثابت int64_t ir1_start,
    ثابت int64_t ir1_end) {

    ثابت بنية ggml_tensor * src0 = dst->src[0];
    ثابت بنية ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    ثابت bool src1_cont = ggml_is_contiguous(src1);

    ggml_vec_dot_t ثابت vec_dot      = type_traits_cpu[type].vec_dot;
    تعداد ggml_type ثابت vec_dot_type = type_traits_cpu[type].vec_dot_type;

    // broadcast factors
    ثابت int64_t r2 = ne12 / ne02;
    ثابت int64_t r3 = ne13 / ne03;

    //printf("ir0_start = %6lld, ir0_end = %6lld, ir1_start = %6lld, ir1_end = %6lld\n", ir0_start, ir0_end, ir1_start, ir1_end);

    // threads with no work simply yield (not sure if it helps)
    𐤀𐤌 (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        أرجع;
    }

    ثابت فراغ * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
    ثابت size_t row_size = ggml_row_size(vec_dot_type, ne10);

    أكد(ne12 % ne02 == 0);
    أكد(ne13 % ne03 == 0);

    // block-tiling attempt
    ثابت int64_t blck_0 = 16;
    ثابت int64_t blck_1 = 16;

    ثابت size_t src1_col_stride = src1_cont || src1->type != vec_dot_type ? row_size : nb11;

    // attempt to reduce false-sharing (does not seem to make a difference)
    // 16 * 2, accounting for mmla kernels
    عائم tmp[32];

    لكل (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        لكل (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            لكل (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += num_rows_per_vec_dot) {
                ثابت int64_t i13 = (ir1 / (ne12 * ne1));
                ثابت int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                ثابت int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                // broadcast src0 into src1
                ثابت int64_t i03 = i13 / r3;
                ثابت int64_t i02 = i12 / r2;

                ثابت int64_t i1 = i11;
                ثابت int64_t i2 = i12;
                ثابت int64_t i3 = i13;

                ثابت حرف * src0_row = (ثابت حرف*)src0->data + (0 + i02 * nb02 + i03 * nb03);

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                ثابت حرف * src1_col = (ثابت حرف*)wdata +
                    (src1_cont || src1->type != vec_dot_type
                        ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                        : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                عائم * dst_col = (عائم*)((حرف*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

                //for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                //    vec_dot(ne00, &dst_col[ir0], src0_row + ir0*nb01, src1_col);
                //}

                لكل (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += num_rows_per_vec_dot) {
                    vec_dot(ne00, &tmp[ir0 - iir0], (num_rows_per_vec_dot > 1 ? 16 : 0), src0_row + ir0 * nb01, (num_rows_per_vec_dot > 1 ? nb01 : 0), src1_col, (num_rows_per_vec_dot > 1 ? src1_col_stride : 0), num_rows_per_vec_dot);
                }

                لكل (صحيح cn = 0; cn < num_rows_per_vec_dot; ++cn) {
                    memcpy(&dst_col[iir0 + cn * nb1 / nb0], tmp + (cn * 16), (MIN(iir0 + blck_0, ir0_end) - iir0) * حجم(عائم));
                }
            }
        }
    }
}

فراغ ggml_compute_forward_mul_mat(
        ثابت بنية ggml_compute_params * params,
              بنية ggml_tensor * dst) {

    ثابت بنية ggml_tensor * src0 = dst->src[0];
    ثابت بنية ggml_tensor * src1 = dst->src[1];

    ثابت int32_t hint = ggml_get_op_params_i32(dst, 1);
    𐤀𐤌 (hint == GGML_HINT_SRC0_IS_HADAMARD && !params->use_ref) {
        ggml_compute_forward_fwht(params, dst);
        أرجع;
    }

    // If tiled is supported, it will execute the full op here and we return
    𐤀𐤌 (ggml_compute_forward_mul_mat_tiled(params, dst)) {
        أرجع;
    }

    GGML_TENSOR_BINARY_OP_LOCALS

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    تعداد ggml_type           ثابت vec_dot_type         = type_traits_cpu[src0->type].vec_dot_type;
    ggml_from_float_t        ثابت from_float           = type_traits_cpu[vec_dot_type].from_float;
    int64_t                  ثابت vec_dot_num_rows     = type_traits_cpu[src0->type].nrows;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == حجم(عائم));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    // TODO: extract to "extra_op"
#𐤀𐤌 GGML_USE_LLAMAFILE
    // broadcast factors
    ثابت int64_t r2 = ne12 / ne02;
    ثابت int64_t r3 = ne13 / ne03;

    ثابت bool src1_cont = ggml_is_contiguous(src1);

    𐤀𐤌 (!params->use_ref && src1_cont) {
        لكل (int64_t i13 = 0; i13 < ne13; i13++)
            لكل (int64_t i12 = 0; i12 < ne12; i12++)
                𐤀𐤌 (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_size(src0->type),
                                     (ثابت حرف *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_size(src0->type),
                                     (ثابت حرف *)src1->data + i12*nb12 + i13*nb13,
                                     nb11/ggml_type_size(src1->type),
                                     (حرف *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_size(dst->type),
                                     src0->type,
                                     src1->type,
                                     dst->type))
                    اذهب UseGgmlGemm1;
        أرجع;
    }
UseGgmlGemm1:;
#endif

    𐤀𐤌 (src1->type != vec_dot_type) {
        حرف * wdata = params->wdata;

        ثابت size_t nbw0 = ggml_type_size(vec_dot_type);
        ثابت size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        ثابت size_t nbw2 = nbw1*ne11;
        ثابت size_t nbw3 = nbw2*ne12;

        أكد(params->wsize >= ne13*nbw3);
        // src1 is either packed from F32 into vec_dot_type, or widened from F16 or BF16 into the F32 work buffer
        ثابت bool widen = src1->type != GGML_TYPE_F32;

        GGML_ASSERT(!widen || vec_dot_type == GGML_TYPE_F32);
        GGML_ASSERT(!widen || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_BF16);

    #𐤀𐤌 0
        لكل (int64_t i13 = 0; i13 < ne13; ++i13) {
            لكل (int64_t i12 = 0; i12 < ne12; ++i12) {
                لكل (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                    from_float((عائم *)((حرف *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (فراغ *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                                ne10);
                }
            }
        }
    #وإلا
        ثابت int64_t bs = ggml_blck_size(vec_dot_type);

        لكل (int64_t i13 = 0; i13 < ne13; ++i13) {
            لكل (int64_t i12 = 0; i12 < ne12; ++i12) {
                لكل (int64_t i11 = 0; i11 < ne11; ++i11) {
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    ثابت فراغ * src1_row  = (ثابت حرف *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10;
                    فراغ       * wdata_row =                    wdata  + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0;

                    ثابت int64_t ne10_block_size = (ne10_block_end - ne10_block_start) * bs;

                    𐤀𐤌 (src1->type == GGML_TYPE_F16) {
                        ggml_cpu_fp16_to_fp32((ثابت ggml_fp16_t *) src1_row, (عائم *) wdata_row, ne10_block_size);
                    } وإلا 𐤀𐤌 (src1->type == GGML_TYPE_BF16) {
                        ggml_cpu_bf16_to_fp32((ثابت ggml_bf16_t *) src1_row, (عائم *) wdata_row, ne10_block_size);
                    } وإلا {
                        from_float((ثابت عائم *) src1_row, wdata_row, ne10_block_size);
                    }
                }
            }
        }
    #endif
    }

    𐤀𐤌 (ith == 0) {
        // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
        atomic_store_explicit(&params->threadpool->current_chunk, nth, memory_order_relaxed);
    }

    ggml_barrier(params->threadpool);

#𐤀𐤌 GGML_USE_LLAMAFILE
    𐤀𐤌 (!params->use_ref && src1->type != vec_dot_type) {
        ثابت فراغ* wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        ثابت size_t row_size = ggml_row_size(vec_dot_type, ne10);

        لكل (int64_t i13 = 0; i13 < ne13; i13++)
            لكل (int64_t i12 = 0; i12 < ne12; i12++)
                𐤀𐤌 (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_size(src0->type),
                                     (ثابت حرف *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_size(src0->type),
                                     (ثابت حرف *)wdata + (i12*ne11 + i13*ne12*ne11)*row_size,
                                     row_size/ggml_type_size(vec_dot_type),
                                     (حرف *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_size(dst->type),
                                     src0->type,
                                     vec_dot_type,
                                     dst->type))
                    اذهب UseGgmlGemm2;
        أرجع;
    }
UseGgmlGemm2:;
#endif

    // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
    ثابت int64_t nr0 = ne0;

    // This is the size of the rest of the dimensions of the result
    ثابت int64_t nr1 = ne1 * ne2 * ne3;

    // Now select a reasonable chunk size.
    صحيح chunk_size = 16;

    // We need to step up the size if it's small
    𐤀𐤌 (nr0 == 1 || nr1 == 1) {
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
    𐤀𐤌 (nchunk0 * nchunk1 < nth * 4 || ggml_is_numa()) {
        // distribute the thread work across the inner or outer loop based on which one is larger
        nchunk0 = nr0 > nr1 ? nth : 1; // parallelize by src0 rows
        nchunk1 = nr0 > nr1 ? 1 : nth; // parallelize by src1 rows
    }

    // The number of elements in each chunk
    ثابت int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    ثابت int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    // The first chunk comes from our thread_id, the rest will get auto-assigned.
    صحيح current_chunk = ith;

    طالما (current_chunk < nchunk0 * nchunk1) {
        ثابت int64_t ith0 = current_chunk % nchunk0;
        ثابت int64_t ith1 = current_chunk / nchunk0;

        ثابت int64_t ir0_start = dr0 * ith0;
        ثابت int64_t ir0_end = MIN(ir0_start + dr0, nr0);

        ثابت int64_t ir1_start = dr1 * ith1;
        ثابت int64_t ir1_end = MIN(ir1_start + dr1, nr1);

        // dot kernels can handle 1 row and col at a time, but mmla kernels can process 2 rows and cols
        int64_t num_rows_per_vec_dot = vec_dot_num_rows;

        // these checks are needed to avoid crossing dim1 boundaries
        // can be optimized, but the logic would become more complicated, so keeping it like this for simplicity
        𐤀𐤌 ((nr0 % 2 != 0) || (ne11 % 2 != 0) || ((ir0_end - ir0_start) % 2 != 0) || ((ir1_end - ir1_start) % 2 != 0)) {
            num_rows_per_vec_dot = 1;
        }
        ggml_compute_forward_mul_mat_one_chunk(params, dst, src0->type, num_rows_per_vec_dot, ir0_start, ir0_end, ir1_start, ir1_end);

        𐤀𐤌 (nth >= nchunk0 * nchunk1) {
            اكسر;
        }

        current_chunk = atomic_fetch_add_explicit(&params->threadpool->current_chunk, 1, memory_order_relaxed);
    }
}

// ggml_compute_forward_mul_mat_id

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id)*ids->ne[0]*ids->ne[1] + (i1)]

بنية mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

ساكن فراغ ggml_compute_forward_mul_mat_id_one_chunk(
    بنية ggml_tensor * dst,
    ثابت بنية ggml_tensor * src0,
    ثابت بنية ggml_tensor * src1,
    ثابت بنية ggml_tensor * ids,
    ثابت int64_t cur_a,
    ثابت int64_t ir0_start,
    ثابت int64_t ir0_end,
    ثابت int64_t ir1_start,
    ثابت int64_t ir1_end,
    ثابت حرف * src0_cur,
    ثابت بنية mmid_row_mapping * matrix_rows,
    ثابت size_t row_size,
    ثابت bool src1_cont,
    ثابت فراغ * wdata) {

    GGML_TENSOR_BINARY_OP_LOCALS

    ثابت تعداد ggml_type type = src0->type;

    ggml_vec_dot_t    ثابت vec_dot      = type_traits_cpu[type].vec_dot;
    تعداد ggml_type    ثابت vec_dot_type = type_traits_cpu[type].vec_dot_type;

    ثابت int64_t blck_0 = 16;
    ثابت int64_t blck_1 = 16;

    عائم tmp[16];

    لكل (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        لكل (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            لكل (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ++ir1) {
                ثابت int64_t _i12 = ir1; // logical row index for this expert

                بنية mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, _i12);
                ثابت صحيح id       = row_mapping.i1; // selected expert index

                ثابت int64_t  i11 = id % ne11;
                ثابت int64_t  i12 = row_mapping.i2; // row index in src1

                ثابت int64_t  i1 = id;  // selected expert index
                ثابت int64_t  i2 = i12; // row

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                ثابت حرف * src1_col = (ثابت حرف *) wdata +
                    (src1_cont || src1->type != vec_dot_type
                    ? (i11      + i12*ne11)*row_size
                    : (i11*nb11 + i12*nb12));

                عائم * dst_col = (عائم *) ((حرف *) dst->data + (i1*nb1 + i2*nb2));

                لكل (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                    vec_dot(ne00, &tmp[ir0 - iir0], 0, src0_cur + ir0*nb01, 0, src1_col, 0, 1);
                }

                memcpy(&dst_col[iir0], tmp, (MIN(iir0 + blck_0, ir0_end) - iir0)*حجم(عائم));
            }
        }
    }
}

ساكن فراغ * incr_ptr_aligned(فراغ ** p, size_t size, size_t align) {

    فراغ * ptr = *p;
    ptr = (فراغ *) GGML_PAD((uintptr_t) ptr, align);
    *p = (فراغ *) ((حرف *) ptr + size);
    أرجع ptr;
}

ساكن فراغ ggml_compute_forward_mul_mat_id(
        ثابت بنية ggml_compute_params * params,
              بنية ggml_tensor * dst) {

    ثابت بنية ggml_tensor * src0 = dst->src[0];
    ثابت بنية ggml_tensor * src1 = dst->src[1];
    ثابت بنية ggml_tensor * ids = dst->src[2];

    GGML_TENSOR_BINARY_OP_LOCALS

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت تعداد ggml_type type = src0->type;

    ثابت bool src1_cont = ggml_is_contiguous(src1);

    تعداد ggml_type    ثابت vec_dot_type    = type_traits_cpu[type].vec_dot_type;
    ggml_from_float_t ثابت from_float      = type_traits_cpu[vec_dot_type].from_float;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == حجم(عائم));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // row groups
    ثابت صحيح n_ids = ids->ne[0]; // n_expert_used
    ثابت صحيح n_as  = ne02;       // n_expert

    فراغ * wdata_cur = params->wdata;

    𐤀𐤌 (src1->type != vec_dot_type) {
        incr_ptr_aligned(&wdata_cur, ggml_row_size(vec_dot_type, ggml_nelements(src1)), حجم(int64_t));
    }

    int64_t * matrix_row_counts = // [n_as]
        incr_ptr_aligned(&wdata_cur, n_as*حجم(int64_t), حجم(int64_t));

    بنية mmid_row_mapping * matrix_rows = // [n_as][ids->ne[0]*ids->ne[1]]
        incr_ptr_aligned(&wdata_cur, n_as*ids->ne[0]*ids->ne[1]*حجم(بنية mmid_row_mapping), حجم(int64_t));

    حرف (*atomic_current_chunk)[CACHE_LINE_SIZE] = // [n_as]
        incr_ptr_aligned(&wdata_cur, CACHE_LINE_SIZE * n_as, CACHE_LINE_SIZE);

    // Tiled matmul (see tiled.h); per-thread work buffers, 0 bytes when disabled. The
    // reservation is unconditional, the per expert eligibility is decided at dispatch time
    حرف * tiled_scratch = incr_ptr_aligned(&wdata_cur, ggml_tiled_wdata_size(nth, dst), 64);

    GGML_ASSERT(params->wsize >= (size_t)((حرف *) wdata_cur - (حرف *) params->wdata));

    𐤀𐤌 (src1->type != vec_dot_type) {
        حرف * wdata = params->wdata;

        ثابت size_t nbw0 = ggml_type_size(vec_dot_type);
        ثابت size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        ثابت size_t nbw2 = nbw1*ne11;
        ثابت size_t nbw3 = nbw2*ne12;

        أكد(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

#𐤀𐤌 0
        لكل (int64_t i13 = 0; i13 < ne13; ++i13) {
            لكل (int64_t i12 = ith; i12 < ne12; i12 += nth) {
                لكل (int64_t i11 = 0; i11 < ne11; ++i11) {
                    from_float((عائم *)((حرف *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (فراغ *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                               ne10);
                }
            }
        }
#وإلا
        لكل (int64_t i13 = 0; i13 < ne13; ++i13) {
            لكل (int64_t i12 = 0; i12 < ne12; ++i12) {
                لكل (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = ggml_blck_size(vec_dot_type);
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    from_float((عائم *)((حرف *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10),
                               (فراغ *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0),
                               (ne10_block_end - ne10_block_start) * bs);
                }
            }
        }
#endif
    }

    𐤀𐤌 (ith == 0) {
        // initialize matrix_row_counts
        memset(matrix_row_counts, 0, n_as*حجم(int64_t));

        // group rows by src0 matrix
        لكل (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            لكل (صحيح id = 0; id < n_ids; ++id) {
                ثابت int32_t i02 = *(ثابت int32_t *) ((ثابت حرف *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);

                أكد(i02 >= 0 && i02 < n_as);

                MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = (بنية mmid_row_mapping) {id, iid1};
                matrix_row_counts[i02] += 1;
            }
        }
    }

    // reset current_chunk
    لكل (صحيح cur_a = ith; cur_a < n_as; cur_a += nth) {
        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);
        *current_chunk_ctr = nth;
    }

    ggml_barrier(params->threadpool);

    لكل (صحيح cur_a = 0; cur_a < n_as; ++cur_a) {
        ثابت int64_t cne1 = matrix_row_counts[cur_a];

        𐤀𐤌 (cne1 == 0) {
            استمر;
        }

        // tiled takes over if profitable for this expert (see tiled.h)
        𐤀𐤌 (ggml_compute_forward_mul_mat_id_tiled(params, dst, cur_a, cne1, (ثابت int32_t *) &MMID_MATRIX_ROW(cur_a, 0), tiled_scratch)) {
            استمر;
        }

        ثابت حرف * src0_cur = (ثابت حرف *) src0->data + cur_a * nb02;
        ثابت فراغ * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        ثابت size_t row_size = ggml_row_size(vec_dot_type, ne10);

        ثابت int64_t nr0 = ne01;
        ثابت int64_t nr1 = cne1;

        صحيح chunk_size = 16;
        𐤀𐤌 (nr0 == 1 || nr1 == 1) {
            chunk_size = 64;
        }

        // disable for NUMA
        ثابت bool disable_chunking = ggml_is_numa();

        int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
        int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

        𐤀𐤌 (nchunk0 * nchunk1 < nth * 4 || disable_chunking) {
            nchunk0 = nr0 > nr1 ? nth : 1;
            nchunk1 = nr0 > nr1 ? 1 : nth;
        }

        ثابت int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        ثابت int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

        صحيح current_chunk = ith;

        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);

        طالما (current_chunk < nchunk0 * nchunk1) {
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

            𐤀𐤌 (nth >= nchunk0 * nchunk1) {
                اكسر;
            }

            current_chunk = atomic_fetch_add_explicit(current_chunk_ctr, 1, memory_order_relaxed);
        }
    }
}

/////////////////////////////////

ساكن فراغ ggml_compute_forward(بنية ggml_compute_params * params, بنية ggml_tensor * tensor) {
    GGML_ASSERT(params);

    𐤀𐤌 (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {
        أرجع;
    }

    // extra_buffer op?
    𐤀𐤌 (ggml_cpu_extra_compute_forward(params, tensor)) {
        أرجع;
    }

    بدل (tensor->op) {
        حالة GGML_OP_DUP:
            {
                ggml_compute_forward_dup(params, tensor);
            } اكسر;
        حالة GGML_OP_ADD:
            {
                ggml_compute_forward_add(params, tensor);
            } اكسر;
        حالة GGML_OP_ADD_ID:
            {
                ggml_compute_forward_add_id(params, tensor);
            } اكسر;
        حالة GGML_OP_ADD1:
            {
                ggml_compute_forward_add1(params, tensor);
            } اكسر;
        حالة GGML_OP_ACC:
            {
                ggml_compute_forward_acc(params, tensor);
            } اكسر;
        حالة GGML_OP_SUB:
            {
                ggml_compute_forward_sub(params, tensor);
            } اكسر;
        حالة GGML_OP_MUL:
            {
                ggml_compute_forward_mul(params, tensor);
            } اكسر;
        حالة GGML_OP_DIV:
            {
                ggml_compute_forward_div(params, tensor);
            } اكسر;
        حالة GGML_OP_SQR:
            {
                ggml_compute_forward_sqr(params, tensor);
            } اكسر;
        حالة GGML_OP_SQRT:
            {
                ggml_compute_forward_sqrt(params, tensor);
            } اكسر;
        حالة GGML_OP_LOG:
            {
                ggml_compute_forward_log(params, tensor);
            } اكسر;
        حالة GGML_OP_SIN:
            {
                ggml_compute_forward_sin(params, tensor);
            } اكسر;
        حالة GGML_OP_COS:
            {
                ggml_compute_forward_cos(params, tensor);
            } اكسر;
        حالة GGML_OP_SUM:
            {
                ggml_compute_forward_sum(params, tensor);
            } اكسر;
        حالة GGML_OP_SUM_ROWS:
            {
                ggml_compute_forward_sum_rows(params, tensor);
            } اكسر;
        حالة GGML_OP_CUMSUM:
            {
                ggml_compute_forward_cumsum(params, tensor);
            } اكسر;
        حالة GGML_OP_MEAN:
            {
                ggml_compute_forward_mean(params, tensor);
            } اكسر;
        حالة GGML_OP_ARGMAX:
            {
                ggml_compute_forward_argmax(params, tensor);
            } اكسر;
        حالة GGML_OP_COUNT_EQUAL:
            {
                ggml_compute_forward_count_equal(params, tensor);
            } اكسر;
        حالة GGML_OP_REPEAT:
            {
                ggml_compute_forward_repeat(params, tensor);
            } اكسر;
        حالة GGML_OP_REPEAT_BACK:
            {
                ggml_compute_forward_repeat_back(params, tensor);
            } اكسر;
        حالة GGML_OP_CONCAT:
            {
                ggml_compute_forward_concat(params, tensor);
            } اكسر;
        حالة GGML_OP_SILU_BACK:
            {
                ggml_compute_forward_silu_back(params, tensor);
            } اكسر;
        حالة GGML_OP_NORM:
            {
                ggml_compute_forward_norm(params, tensor);
            } اكسر;
        حالة GGML_OP_RMS_NORM:
            {
                ggml_compute_forward_rms_norm(params, tensor);
            } اكسر;
        حالة GGML_OP_RMS_NORM_BACK:
            {
                ggml_compute_forward_rms_norm_back(params, tensor);
            } اكسر;
        حالة GGML_OP_GROUP_NORM:
            {
                ggml_compute_forward_group_norm(params, tensor);
            } اكسر;
        حالة GGML_OP_L2_NORM:
            {
                ggml_compute_forward_l2_norm(params, tensor);
            } اكسر;
        حالة GGML_OP_MUL_MAT:
            {
                ggml_compute_forward_mul_mat(params, tensor);
            } اكسر;
        حالة GGML_OP_MUL_MAT_ID:
            {
                ggml_compute_forward_mul_mat_id(params, tensor);
            } اكسر;
        حالة GGML_OP_OUT_PROD:
            {
                ggml_compute_forward_out_prod(params, tensor);
            } اكسر;
        حالة GGML_OP_SCALE:
            {
                ggml_compute_forward_scale(params, tensor);
            } اكسر;
        حالة GGML_OP_SET:
            {
                ggml_compute_forward_set(params, tensor);
            } اكسر;
        حالة GGML_OP_CPY:
            {
                ggml_compute_forward_cpy(params, tensor);
            } اكسر;
        حالة GGML_OP_CONT:
            {
                ggml_compute_forward_cont(params, tensor);
            } اكسر;
        حالة GGML_OP_GET_ROWS:
            {
                ggml_compute_forward_get_rows(params, tensor);
            } اكسر;
        حالة GGML_OP_GET_ROWS_BACK:
            {
                ggml_compute_forward_get_rows_back(params, tensor);
            } اكسر;
        حالة GGML_OP_SET_ROWS:
            {
                ggml_compute_forward_set_rows(params, tensor);
            } اكسر;
        حالة GGML_OP_DIAG:
            {
                ggml_compute_forward_diag(params, tensor);
            } اكسر;
        حالة GGML_OP_DIAG_MASK_INF:
            {
                ggml_compute_forward_diag_mask_inf(params, tensor);
            } اكسر;
        حالة GGML_OP_DIAG_MASK_ZERO:
            {
                ggml_compute_forward_diag_mask_zero(params, tensor);
            } اكسر;
        حالة GGML_OP_SOFT_MAX:
            {
                ggml_compute_forward_soft_max(params, tensor);
            } اكسر;
        حالة GGML_OP_SOFT_MAX_BACK:
            {
                ggml_compute_forward_soft_max_ext_back(params, tensor);
            } اكسر;
        حالة GGML_OP_ROPE:
            {
                ggml_compute_forward_rope(params, tensor);
            } اكسر;
        حالة GGML_OP_ROPE_BACK:
            {
                ggml_compute_forward_rope_back(params, tensor);
            } اكسر;
        حالة GGML_OP_CLAMP:
            {
                ggml_compute_forward_clamp(params, tensor);
            } اكسر;
        حالة GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_compute_forward_conv_transpose_1d(params, tensor);
            } اكسر;
        حالة GGML_OP_IM2COL:
            {
                ggml_compute_forward_im2col(params, tensor);
            } اكسر;
        حالة GGML_OP_IM2COL_BACK:
            {
                ggml_compute_forward_im2col_back_f32(params, tensor);
            } اكسر;
        حالة GGML_OP_IM2COL_3D:
            {
                ggml_compute_forward_im2col_3d(params, tensor);
            } اكسر;
        حالة GGML_OP_COL2IM_1D:
            {
                ggml_compute_forward_col2im_1d(params, tensor);
            } اكسر;
        حالة GGML_OP_CONV_2D:
            {
                ggml_compute_forward_conv_2d(params, tensor);
            } اكسر;
        حالة GGML_OP_CONV_3D:
            {
                ggml_compute_forward_conv_3d(params, tensor);
            } اكسر;
        حالة GGML_OP_CONV_2D_DW:
            {
                ggml_compute_forward_conv_2d_dw(params, tensor);
            } اكسر;
        حالة GGML_OP_CONV_TRANSPOSE_2D:
            {
                ggml_compute_forward_conv_transpose_2d(params, tensor);
            } اكسر;
        حالة GGML_OP_POOL_1D:
            {
                ggml_compute_forward_pool_1d(params, tensor);
            } اكسر;
        حالة GGML_OP_POOL_2D:
            {
                ggml_compute_forward_pool_2d(params, tensor);
            } اكسر;
        حالة GGML_OP_POOL_2D_BACK:
            {
                ggml_compute_forward_pool_2d_back(params, tensor);
            } اكسر;
        حالة GGML_OP_UPSCALE:
            {
                ggml_compute_forward_upscale(params, tensor);
            } اكسر;
        حالة GGML_OP_PAD:
            {
                ggml_compute_forward_pad(params, tensor);
            } اكسر;
        حالة GGML_OP_PAD_REFLECT_1D:
            {
                ggml_compute_forward_pad_reflect_1d(params, tensor);
            } اكسر;
        حالة GGML_OP_ROLL:
            {
                ggml_compute_forward_roll(params, tensor);
            } اكسر;
        حالة GGML_OP_ARANGE:
            {
                ggml_compute_forward_arange(params, tensor);
            } اكسر;
        حالة GGML_OP_TIMESTEP_EMBEDDING:
            {
                ggml_compute_forward_timestep_embedding(params, tensor);
            } اكسر;
        حالة GGML_OP_ARGSORT:
            {
                ggml_compute_forward_argsort(params, tensor);
            } اكسر;
        حالة GGML_OP_TOP_K:
            {
                ggml_compute_forward_top_k(params, tensor);
            } اكسر;
        حالة GGML_OP_LEAKY_RELU:
            {
                ggml_compute_forward_leaky_relu(params, tensor);
            } اكسر;
        حالة GGML_OP_TRI:
            {
                ggml_compute_forward_tri(params, tensor);
            } اكسر;
        حالة GGML_OP_FILL:
            {
                ggml_compute_forward_fill(params, tensor);
            } اكسر;
        حالة GGML_OP_FLASH_ATTN_EXT:
            {
                ggml_compute_forward_flash_attn_ext(params, tensor);
            } اكسر;
        حالة GGML_OP_FLASH_ATTN_BACK:
            {
                int32_t t = ggml_get_op_params_i32(tensor, 0);
                GGML_ASSERT(t == 0 || t == 1);
                bool masked = t != 0;
                ggml_compute_forward_flash_attn_back(params, masked, tensor);
            } اكسر;
        حالة GGML_OP_SSM_CONV:
            {
                ggml_compute_forward_ssm_conv(params, tensor);
            } اكسر;
        حالة GGML_OP_SSM_SCAN:
            {
                ggml_compute_forward_ssm_scan(params, tensor);
            } اكسر;
        حالة GGML_OP_WIN_PART:
            {
                ggml_compute_forward_win_part(params, tensor);
            } اكسر;
        حالة GGML_OP_WIN_UNPART:
            {
                ggml_compute_forward_win_unpart(params, tensor);
            } اكسر;
        حالة GGML_OP_UNARY:
            {
                ggml_compute_forward_unary(params, tensor);
            } اكسر;
        حالة GGML_OP_GLU:
            {
                ggml_compute_forward_glu(params, tensor);
            } اكسر;
        حالة GGML_OP_GET_REL_POS:
            {
                ggml_compute_forward_get_rel_pos(params, tensor);
            } اكسر;
        حالة GGML_OP_ADD_REL_POS:
            {
                ggml_compute_forward_add_rel_pos(params, tensor);
            } اكسر;
        حالة GGML_OP_RWKV_WKV6:
            {
                ggml_compute_forward_rwkv_wkv6(params, tensor);
            } اكسر;
        حالة GGML_OP_GATED_LINEAR_ATTN:
            {
                ggml_compute_forward_gla(params, tensor);
            } اكسر;
        حالة GGML_OP_RWKV_WKV7:
            {
                ggml_compute_forward_rwkv_wkv7(params, tensor);
            } اكسر;
        حالة GGML_OP_SOLVE_TRI:
            {
                ggml_compute_forward_solve_tri(params, tensor);
            } اكسر;
        حالة GGML_OP_GATED_DELTA_NET:
            {
                ggml_compute_forward_gated_delta_net(params, tensor);
            } اكسر;
        حالة GGML_OP_LIGHTNING_INDEXER:
            {
                ggml_compute_forward_lightning_indexer(params, tensor);
            } اكسر;
        حالة GGML_OP_DSV4_HC_COMB:
            {
                ggml_compute_forward_dsv4_hc_comb(params, tensor);
            } اكسر;
        حالة GGML_OP_DSV4_HC_PRE:
            {
                ggml_compute_forward_dsv4_hc_pre(params, tensor);
            } اكسر;
        حالة GGML_OP_DSV4_HC_POST:
            {
                ggml_compute_forward_dsv4_hc_post(params, tensor);
            } اكسر;
        حالة GGML_OP_TOP_K_MASK:
            {
                ggml_compute_forward_top_k_mask(params, tensor);
            } اكسر;
        حالة GGML_OP_MAP_CUSTOM1:
            {
                ggml_compute_forward_map_custom1(params, tensor);
            }
            اكسر;
        حالة GGML_OP_MAP_CUSTOM2:
            {
                ggml_compute_forward_map_custom2(params, tensor);
            }
            اكسر;
        حالة GGML_OP_MAP_CUSTOM3:
            {
                ggml_compute_forward_map_custom3(params, tensor);
            }
            اكسر;
        حالة GGML_OP_CUSTOM:
            {
                ggml_compute_forward_custom(params, tensor);
            }
            اكسر;
        حالة GGML_OP_CROSS_ENTROPY_LOSS:
            {
                ggml_compute_forward_cross_entropy_loss(params, tensor);
            }
            اكسر;
        حالة GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            {
                ggml_compute_forward_cross_entropy_loss_back(params, tensor);
            }
            اكسر;
        حالة GGML_OP_OPT_STEP_ADAMW:
            {
                ggml_compute_forward_opt_step_adamw(params, tensor);
            }
            اكسر;
        حالة GGML_OP_OPT_STEP_SGD:
            {
                ggml_compute_forward_opt_step_sgd(params, tensor);
            }
            اكسر;
        حالة GGML_OP_NONE:
            {
                // nop
            } اكسر;
        حالة GGML_OP_RESHAPE:
            {
                // nop
            } اكسر;
        حالة GGML_OP_PERMUTE:
            {
                // nop
            } اكسر;
        حالة GGML_OP_VIEW:
            {
                // nop
            } اكسر;
        حالة GGML_OP_TRANSPOSE:
            {
                // nop
            } اكسر;
        حالة GGML_OP_COUNT:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// Android's libc implementation "bionic" does not support setting affinity
#𐤀𐤌 defined(__gnu_linux__)
ساكن فراغ set_numa_thread_affinity(صحيح thread_n) {
    𐤀𐤌 (!ggml_is_numa()) {
        أرجع;
    }

    صحيح node_num;
    صحيح rv;
    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    بدل(g_state.numa.numa_strategy) {
        حالة GGML_NUMA_STRATEGY_DISTRIBUTE:
            // run thread on node_num thread_n / (threads per node)
            node_num = thread_n % g_state.numa.n_nodes;
            اكسر;
        حالة GGML_NUMA_STRATEGY_ISOLATE:
            // run thread on current_node
            node_num = g_state.numa.current_node;
            اكسر;
        حالة GGML_NUMA_STRATEGY_NUMACTL:
            // use the cpuset that numactl gave us
            rv = pthread_setaffinity_np(pthread_self(), setsize, &g_state.numa.cpuset);
            𐤀𐤌 (rv) {
                fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n",strerror(rv));
            }
            أرجع;
        افتراضي:
            أرجع;
    }

    بنية ggml_numa_node * node = &g_state.numa.nodes[node_num];

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    لكل (size_t i = 0; i < node->n_cpus; ++i) {
        CPU_SET_S(node->cpus[i], setsize, cpus);
    }

    rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    𐤀𐤌 (rv) {
            fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}

ساكن فراغ clear_numa_thread_affinity(فراغ) {
    𐤀𐤌 (!ggml_is_numa()) {
        أرجع;
    }

    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    لكل (unsigned i = 0; i < g_state.numa.total_cpus; ++i) {
        CPU_SET_S(i, setsize, cpus);
    }

    صحيح rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    𐤀𐤌 (rv) {
        fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}
#وإلا
// TODO: Windows etc.
// (the linux implementation may also work on BSD, someone should test)
ساكن فراغ set_numa_thread_affinity(صحيح thread_n) { UNUSED(thread_n);  }
ساكن فراغ clear_numa_thread_affinity(فراغ) {}
#endif

ساكن صحيح ggml_get_n_tasks(بنية ggml_tensor * node, صحيح n_threads) {
    صحيح n_tasks = 0;

    𐤀𐤌 (ggml_is_empty(node)) {
        // no need to multi-thread a no-op
        n_tasks = 1;
        أرجع n_tasks;
    }

    بدل (node->op) {
        حالة GGML_OP_CPY:
        حالة GGML_OP_DUP:
        حالة GGML_OP_CONT:
        حالة GGML_OP_ADD:
        حالة GGML_OP_ADD_ID:
        حالة GGML_OP_ADD1:
        حالة GGML_OP_ACC:
        حالة GGML_OP_CUMSUM:
        حالة GGML_OP_TRI:
        حالة GGML_OP_FILL:
            {
                n_tasks = n_threads;
            } اكسر;
        حالة GGML_OP_SUB:
        حالة GGML_OP_SQR:
        حالة GGML_OP_SQRT:
        حالة GGML_OP_LOG:
        حالة GGML_OP_SIN:
        حالة GGML_OP_COS:
        حالة GGML_OP_SUM:
        حالة GGML_OP_SUM_ROWS:
        حالة GGML_OP_MEAN:
        حالة GGML_OP_ARGMAX:
            {
                n_tasks = 1;
            } اكسر;
        حالة GGML_OP_COUNT_EQUAL:
        حالة GGML_OP_SOLVE_TRI:
        حالة GGML_OP_GATED_DELTA_NET:
        حالة GGML_OP_DSV4_HC_COMB:
        حالة GGML_OP_DSV4_HC_PRE:
        حالة GGML_OP_DSV4_HC_POST:
        حالة GGML_OP_TOP_K_MASK:
            {
                n_tasks = n_threads;
            } اكسر;
        حالة GGML_OP_REPEAT:
        حالة GGML_OP_REPEAT_BACK:
        حالة GGML_OP_LEAKY_RELU:
            {
                n_tasks = 1;
            } اكسر;
        حالة GGML_OP_UNARY:
            بدل (ggml_get_unary_op(node)) {
                حالة GGML_UNARY_OP_ABS:
                حالة GGML_UNARY_OP_SGN:
                حالة GGML_UNARY_OP_NEG:
                حالة GGML_UNARY_OP_STEP:
                حالة GGML_UNARY_OP_TANH:
                حالة GGML_UNARY_OP_ELU:
                حالة GGML_UNARY_OP_RELU:
                حالة GGML_UNARY_OP_SIGMOID:
                حالة GGML_UNARY_OP_HARDSWISH:
                حالة GGML_UNARY_OP_HARDSIGMOID:
                حالة GGML_UNARY_OP_EXP:
                حالة GGML_UNARY_OP_SOFTPLUS:
                حالة GGML_UNARY_OP_EXPM1:
                حالة GGML_UNARY_OP_FLOOR:
                حالة GGML_UNARY_OP_CEIL:
                حالة GGML_UNARY_OP_ROUND:
                حالة GGML_UNARY_OP_TRUNC:
                    {
                        n_tasks = 1;
                    } اكسر;

                حالة GGML_UNARY_OP_GELU:
                حالة GGML_UNARY_OP_GELU_ERF:
                حالة GGML_UNARY_OP_GELU_QUICK:
                حالة GGML_UNARY_OP_SILU:
                حالة GGML_UNARY_OP_XIELU:
                    {
                        n_tasks = n_threads;
                    } اكسر;
                افتراضي:
                    GGML_ABORT("fatal error");
            }
            اكسر;
        حالة GGML_OP_GLU:
            بدل (ggml_get_glu_op(node)) {
                حالة GGML_GLU_OP_REGLU:
                حالة GGML_GLU_OP_GEGLU:
                حالة GGML_GLU_OP_SWIGLU:
                حالة GGML_GLU_OP_SWIGLU_OAI:
                حالة GGML_GLU_OP_GEGLU_ERF:
                حالة GGML_GLU_OP_GEGLU_QUICK:
                حالة GGML_GLU_OP_SWIGLU_CLAMP:
                    {
                        n_tasks = n_threads;
                    } اكسر;
                افتراضي:
                    GGML_ABORT("fatal error");
            }
            اكسر;
        حالة GGML_OP_SILU_BACK:
        حالة GGML_OP_MUL:
        حالة GGML_OP_DIV:
        حالة GGML_OP_NORM:
        حالة GGML_OP_RMS_NORM:
        حالة GGML_OP_RMS_NORM_BACK:
        حالة GGML_OP_L2_NORM:
        حالة GGML_OP_GROUP_NORM:
        حالة GGML_OP_CONCAT:
        حالة GGML_OP_MUL_MAT:
        حالة GGML_OP_MUL_MAT_ID:
        حالة GGML_OP_OUT_PROD:
        حالة GGML_OP_SET_ROWS:
            {
                n_tasks = n_threads;
            } اكسر;
        حالة GGML_OP_GET_ROWS:
            {
                // FIXME: get_rows can use additional threads, but the cost of launching additional threads
                // decreases performance with GPU offloading
                //n_tasks = n_threads;
                n_tasks = 1;
            } اكسر;
        حالة GGML_OP_SCALE:
        حالة GGML_OP_SET:
        حالة GGML_OP_RESHAPE:
        حالة GGML_OP_VIEW:
        حالة GGML_OP_PERMUTE:
        حالة GGML_OP_TRANSPOSE:
        حالة GGML_OP_GET_ROWS_BACK:
        حالة GGML_OP_DIAG:
            {
                n_tasks = 1;
            } اكسر;
        حالة GGML_OP_DIAG_MASK_ZERO:
        حالة GGML_OP_DIAG_MASK_INF:
        حالة GGML_OP_SOFT_MAX_BACK:
        حالة GGML_OP_ROPE:
        حالة GGML_OP_ROPE_BACK:
        حالة GGML_OP_ADD_REL_POS:
            {
                n_tasks = n_threads;
            } اكسر;
        حالة GGML_OP_CLAMP:
            {
                n_tasks = 1; //TODO
            } اكسر;
        حالة GGML_OP_SOFT_MAX:
            {
                n_tasks = MIN(n_threads, ggml_nrows(node->src[0]));
            } اكسر;
        حالة GGML_OP_IM2COL:
        حالة GGML_OP_IM2COL_BACK:
        حالة GGML_OP_IM2COL_3D:
        حالة GGML_OP_CONV_2D:
        حالة GGML_OP_CONV_3D:
        حالة GGML_OP_CONV_2D_DW:
        حالة GGML_OP_COL2IM_1D:
        حالة GGML_OP_CONV_TRANSPOSE_1D:
        حالة GGML_OP_CONV_TRANSPOSE_2D:
            {
                n_tasks = n_threads;
            } اكسر;
        حالة GGML_OP_POOL_1D:
        حالة GGML_OP_POOL_2D:
        حالة GGML_OP_POOL_2D_BACK:
            {
                n_tasks = 1;
            } اكسر;
        حالة GGML_OP_UPSCALE:
        حالة GGML_OP_PAD:
        حالة GGML_OP_PAD_REFLECT_1D:
        حالة GGML_OP_ROLL:
        حالة GGML_OP_ARANGE:
        حالة GGML_OP_TIMESTEP_EMBEDDING:
        حالة GGML_OP_ARGSORT:
        حالة GGML_OP_TOP_K:
        حالة GGML_OP_FLASH_ATTN_EXT:
        حالة GGML_OP_FLASH_ATTN_BACK:
        حالة GGML_OP_SSM_CONV:
        حالة GGML_OP_SSM_SCAN:
        حالة GGML_OP_LIGHTNING_INDEXER:
            {
                n_tasks = n_threads;
            } اكسر;
        حالة GGML_OP_RWKV_WKV6:
        حالة GGML_OP_GATED_LINEAR_ATTN:
        حالة GGML_OP_RWKV_WKV7:
            {
                ثابت int64_t n_heads = node->src[1]->ne[1];
                n_tasks = MIN(n_threads, n_heads);
            } اكسر;
        حالة GGML_OP_WIN_PART:
        حالة GGML_OP_WIN_UNPART:
        حالة GGML_OP_GET_REL_POS:
            {
                n_tasks = 1;
            } اكسر;
        حالة GGML_OP_MAP_CUSTOM1:
            {
                بنية ggml_map_custom1_op_params p;
                memcpy(&p, node->op_params, حجم(p));
                𐤀𐤌 (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } وإلا {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } اكسر;
        حالة GGML_OP_MAP_CUSTOM2:
            {
                بنية ggml_map_custom2_op_params p;
                memcpy(&p, node->op_params, حجم(p));
                𐤀𐤌 (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } وإلا {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } اكسر;
        حالة GGML_OP_MAP_CUSTOM3:
            {
                بنية ggml_map_custom3_op_params p;
                memcpy(&p, node->op_params, حجم(p));
                𐤀𐤌 (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } وإلا {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } اكسر;
        حالة GGML_OP_CUSTOM:
            {
                بنية ggml_custom_op_params p;
                memcpy(&p, node->op_params, حجم(p));
                𐤀𐤌 (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } وإلا {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } اكسر;
        حالة GGML_OP_CROSS_ENTROPY_LOSS:
        حالة GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        حالة GGML_OP_OPT_STEP_ADAMW:
        حالة GGML_OP_OPT_STEP_SGD:
            {
                n_tasks = n_threads;
            } اكسر;
        حالة GGML_OP_NONE:
            {
                n_tasks = 1;
            } اكسر;
        حالة GGML_OP_COUNT:
            {
                GGML_ABORT("fatal error");
            }
        افتراضي:
            {
                fprintf(stderr, "%s: op not implemented: ", __func__);
                𐤀𐤌 (node->op < GGML_OP_COUNT) {
                    fprintf(stderr, "%s\n", ggml_op_name(node->op));
                } وإلا {
                    fprintf(stderr, "%d\n", node->op);
                }
                GGML_ABORT("fatal error");
            }
    }

    أكد(n_tasks > 0);

    أرجع n_tasks;
}

ساكن thread_ret_t ggml_graph_compute_secondary_thread(فراغ* data);

#𐤀𐤌 defined(_WIN32)
#ضمن "windows.h"

// TODO: support > 64 CPUs
ساكن bool ggml_thread_apply_affinity(bool * mask) {
    HANDLE    h = GetCurrentThread();
    uint64_t  bitmask = 0ULL;

    أكد(GGML_MAX_N_THREADS >= 64);

    لكل (int32_t i = 0; i < 8; i++) {
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

    لكل (int32_t i = 64; i < GGML_MAX_N_THREADS; i++) {
        𐤀𐤌 (mask[i]) {
            fprintf(stderr, "warn: setting thread-affinity for > 64 CPUs isn't supported on windows!\n");
            اكسر;
        }
    }

    DWORD_PTR m = (DWORD_PTR)bitmask;

    m = SetThreadAffinityMask(h, m);

    أرجع m != 0;
}

ساكن bool ggml_thread_apply_priority(int32_t prio) {
    // Note that on Windows the Process Priority Class must be updated in order to set Thread priority.
    // This is up to the applications.
    DWORD p = THREAD_PRIORITY_NORMAL;
    بدل (prio) {
        حالة GGML_SCHED_PRIO_LOW:      p = THREAD_PRIORITY_BELOW_NORMAL;  اكسر;
        حالة GGML_SCHED_PRIO_NORMAL:   p = THREAD_PRIORITY_NORMAL;        اكسر;
        حالة GGML_SCHED_PRIO_MEDIUM:   p = THREAD_PRIORITY_ABOVE_NORMAL;  اكسر;
        حالة GGML_SCHED_PRIO_HIGH:     p = THREAD_PRIORITY_HIGHEST;       اكسر;
        حالة GGML_SCHED_PRIO_REALTIME: p = THREAD_PRIORITY_TIME_CRITICAL; اكسر;
    }

    𐤀𐤌 (prio != GGML_SCHED_PRIO_LOW) {
        // Tell Windows that this thread should not be throttled (needs its own CPU core).
        // Newer Windows 11 versions aggressively park (offline) CPU cores and often place
        // all our threads onto the first 4 cores which results in terrible performance with
        // n_threads > 4
        #𐤀𐤌 _WIN32_WINNT >= 0x0602
        THREAD_POWER_THROTTLING_STATE t;
        ZeroMemory(&t, حجم(t));
        t.Version     = THREAD_POWER_THROTTLING_CURRENT_VERSION;
        t.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
        t.StateMask   = 0;

        𐤀𐤌 (!SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &t, حجم(t))) {
            GGML_LOG_DEBUG("failed to disable thread power throttling %d : (%d)\n", prio, (صحيح) GetLastError());
            أرجع false;
        }
        #endif
    }

    𐤀𐤌 (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        أرجع true;
    }

    𐤀𐤌 (!SetThreadPriority(GetCurrentThread(), p)) {
        fprintf(stderr, "warn: failed to set thread priority %d : (%d)\n", prio, (صحيح) GetLastError());
        أرجع false;
    }

    أرجع true;
}

#وإلاإذا defined(__APPLE__)
#ضمن <sys/types.h>
#ضمن <sys/resource.h>

ساكن bool ggml_thread_apply_affinity(ثابت bool * mask) {
    // Not supported on Apple platforms
    UNUSED(mask);
    أرجع true;
}

ساكن bool ggml_thread_apply_priority(int32_t prio) {
    بنية sched_param p;
    int32_t policy = SCHED_OTHER;
    بدل (prio) {
        // TODO: there seems to be no way to set lower prio on Apple platforms
        حالة GGML_SCHED_PRIO_LOW:      policy = SCHED_OTHER; p.sched_priority = 0;  اكسر;
        حالة GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  اكسر;
        حالة GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; اكسر;
        حالة GGML_SCHED_PRIO_HIGH:     policy = SCHED_FIFO;  p.sched_priority = 80; اكسر;
        حالة GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; اكسر;
    }

    𐤀𐤌 (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        أرجع true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    𐤀𐤌 (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        أرجع false;
    }

    أرجع true;
}

#وإلاإذا defined(__linux__)
// TODO: this may not work on BSD, to be verified

ساكن bool ggml_thread_apply_affinity(ثابت bool * mask) {
    cpu_set_t cpuset;
    صحيح err;

    CPU_ZERO(&cpuset);

    لكل (uint32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
        𐤀𐤌 (mask[i]) {
            GGML_PRINT_DEBUG("Thread %lx: adding %d to cpuset\n", pthread_self(), i);
            CPU_SET(i, &cpuset);
        }
    }

#ifdef __ANDROID__
    err = sched_setaffinity(0, حجم(cpuset), &cpuset);
    𐤀𐤌 (err < 0) {
        err = errno;
    }
#وإلا
    err = pthread_setaffinity_np(pthread_self(), حجم(cpuset), &cpuset);
#endif
    𐤀𐤌 (err != 0) {
        fprintf(stderr, "warn: failed to set affinity mask 0x%llx : %s (%d)\n", (unsigned long long)mask, strerror(err), err);
        أرجع false;
    }

    أرجع true;
}

ساكن bool ggml_thread_apply_priority(int32_t prio) {
    بنية sched_param p;
    int32_t policy = SCHED_OTHER;
    بدل (prio) {
        حالة GGML_SCHED_PRIO_LOW:      policy = SCHED_BATCH; p.sched_priority = 0;  اكسر;
        حالة GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  اكسر;
        حالة GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; اكسر;
        حالة GGML_SCHED_PRIO_HIGH:     policy = SCHED_FIFO;  p.sched_priority = 80; اكسر;
        حالة GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; اكسر;
    }

    𐤀𐤌 (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        أرجع true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    𐤀𐤌 (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        أرجع false;
    }

    أرجع true;
}

#وإلا // unsupported platforms

ساكن bool ggml_thread_apply_affinity(ثابت bool * mask) {
    UNUSED(mask);
    أرجع true;
}

ساكن bool ggml_thread_apply_priority(int32_t prio) {
    UNUSED(prio);
    أرجع true;
}

#endif

ساكن bool ggml_thread_cpumask_is_valid(ثابت bool * mask) {
    لكل (صحيح i = 0; i < GGML_MAX_N_THREADS; i++) {
        𐤀𐤌 (mask[i]) { أرجع true; }
    }
    أرجع false;
}

ساكن فراغ ggml_thread_cpumask_next(ثابت bool * global_mask, bool * local_mask, bool strict, int32_t* iter) {
    𐤀𐤌 (!strict) {
        memcpy(local_mask, global_mask, GGML_MAX_N_THREADS);
        أرجع;
    } وإلا {
        memset(local_mask, 0, GGML_MAX_N_THREADS);
        int32_t base_idx = *iter;
        لكل (int32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
            int32_t idx = base_idx + i;
            𐤀𐤌 (idx >= GGML_MAX_N_THREADS) {
                // Just a cheaper modulo
                idx -= GGML_MAX_N_THREADS;
            }
            𐤀𐤌 (global_mask[idx]) {
                local_mask[idx] = 1;
                *iter = idx + 1;
                أرجع;
            }
        }
    }
}

فراغ ggml_threadpool_free(بنية ggml_threadpool* threadpool) {
    𐤀𐤌 (!threadpool) أرجع;

    ثابت صحيح n_threads = threadpool->n_threads;

#ifndef GGML_USE_OPENMP
    بنية ggml_compute_state* workers = threadpool->workers;

    ggml_mutex_lock(&threadpool->mutex);

    threadpool->stop = true;
    threadpool->pause = false;

    ggml_cond_broadcast(&threadpool->cond);
    ggml_mutex_unlock(&threadpool->mutex);

    لكل (صحيح j = 1; j < n_threads; j++) {
        int32_t rc = ggml_thread_join(workers[j].thrd, عدم);
        GGML_ASSERT(rc == GGML_EXIT_SUCCESS || rc == GGML_EXIT_ABORTED);
        UNUSED(rc);
    }

    ggml_mutex_destroy(&threadpool->mutex);
    ggml_cond_destroy(&threadpool->cond);
#endif // GGML_USE_OPENMP

    ثابت size_t workers_size = حجم(بنية ggml_compute_state) * n_threads;
    ggml_aligned_free(threadpool->workers, workers_size);
    ggml_aligned_free(threadpool, حجم(بنية ggml_threadpool));
}

#ifndef GGML_USE_OPENMP
// pause/resume must be called under mutex
ساكن فراغ ggml_threadpool_pause_locked(بنية ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Pausing threadpool\n");
    threadpool->pause = true;
    ggml_cond_broadcast(&threadpool->cond);
}

ساكن فراغ ggml_threadpool_resume_locked(بنية ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Resuming threadpool\n");
    threadpool->pause = false;
    ggml_cond_broadcast(&threadpool->cond);
}
#endif

فراغ ggml_threadpool_pause(بنية ggml_threadpool * threadpool) {
#ifndef GGML_USE_OPENMP
    ggml_mutex_lock(&threadpool->mutex);
    𐤀𐤌 (!threadpool->pause) {
       ggml_threadpool_pause_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#وإلا
    UNUSED(threadpool);
#endif
}

فراغ ggml_threadpool_resume(بنية ggml_threadpool * threadpool) {
#ifndef GGML_USE_OPENMP
    ggml_mutex_lock(&threadpool->mutex);
    𐤀𐤌 (threadpool->pause) {
       ggml_threadpool_resume_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#وإلا
    UNUSED(threadpool);
#endif
}

بنية ggml_cplan ggml_graph_plan(
          ثابت بنية ggml_cgraph * cgraph,
                               صحيح   n_threads,
            بنية ggml_threadpool * threadpool) {

    𐤀𐤌 (threadpool == عدم) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
    }
    𐤀𐤌 (n_threads <= 0) {
        n_threads = threadpool ? threadpool->n_threads : GGML_DEFAULT_N_THREADS;
    }

#𐤀𐤌 defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
    // Emscripten without pthreads support can only use a single thread
    n_threads = 1;
#endif

#𐤀𐤌 defined(__wasi__)
    // WASI doesn't support parallelism yet
    n_threads = 1;
#endif

    size_t work_size = 0;

    بنية ggml_cplan cplan;
    memset(&cplan, 0, حجم(بنية ggml_cplan));

    صحيح max_tasks = 1;

    // thread scheduling for the different operations + work buffer size estimation
    لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
        بنية ggml_tensor * node = cgraph->nodes[i];

        ثابت صحيح n_tasks = ggml_get_n_tasks(node, n_threads);

        max_tasks = MAX(max_tasks, n_tasks);

        size_t cur = 0;

        𐤀𐤌 (!ggml_cpu_extra_work_size(n_threads, node, &cur)) {
            بدل (node->op) {
                حالة GGML_OP_CPY:
                حالة GGML_OP_DUP:
                    {
                        𐤀𐤌 (ggml_is_quantized(node->type) ||
                            // F16 -> BF16 and BF16 -> F16 copies go through intermediate F32
                            (node->src[0]->type == GGML_TYPE_F16  && node->src[1] && node->src[1]->type == GGML_TYPE_BF16) ||
                            (node->src[0]->type == GGML_TYPE_BF16 && node->src[1] && node->src[1]->type == GGML_TYPE_F16) ||
                            // conversion between F32 and I32
                            (node->src[0]->type == GGML_TYPE_F32 && node->src[1] && node->src[1]->type == GGML_TYPE_I32) ||
                            (node->src[0]->type == GGML_TYPE_I32 && node->src[1] && node->src[1]->type == GGML_TYPE_F32)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->ne[0] * n_tasks;
                        }
                    } اكسر;
                حالة GGML_OP_ADD:
                حالة GGML_OP_ADD_ID:
                حالة GGML_OP_ADD1:
                    {
                        𐤀𐤌 (ggml_is_quantized(node->src[0]->type)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[0]->ne[0] * n_tasks;
                        }
                    } اكسر;
                حالة GGML_OP_ACC:
                    {
                        𐤀𐤌 (ggml_is_quantized(node->src[0]->type)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[1]->ne[0] * n_tasks;
                        }
                    } اكسر;
                حالة GGML_OP_COUNT_EQUAL:
                    {
                        cur = ggml_type_size(node->type)*n_tasks;
                    } اكسر;
                حالة GGML_OP_MUL_MAT:
                    {
                        ثابت تعداد ggml_type vec_dot_type = type_traits_cpu[node->src[0]->type].vec_dot_type;
                        𐤀𐤌 (node->src[1]->type != vec_dot_type) {
                            cur = ggml_row_size(vec_dot_type, ggml_nelements(node->src[1]));
                        }
                        // Workspace for tiled (see tiled.h)
                        cur = GGML_PAD(cur, 64);
                        cur += ggml_tiled_wdata_size(n_tasks, node);
                    } اكسر;
                حالة GGML_OP_MUL_MAT_ID:
                    {
                        cur = 0;
                        ثابت بنية ggml_tensor * src0 = node->src[0];
                        ثابت بنية ggml_tensor * src1 = node->src[1];
                        ثابت بنية ggml_tensor * ids = node->src[2];
                        ثابت تعداد ggml_type vec_dot_type = type_traits_cpu[src0->type].vec_dot_type;
                        ثابت صحيح n_as = src0->ne[2];
                        // src1
                        𐤀𐤌 (src1->type != vec_dot_type) {
                            cur += ggml_row_size(vec_dot_type, ggml_nelements(src1)) + حجم(int64_t);
                        }
                        // matrix_row_counts
                        cur += n_as * حجم(int64_t) + حجم(int64_t);
                        // matrix_rows
                        cur += n_as*ids->ne[0]*ids->ne[1]*حجم(بنية mmid_row_mapping) + حجم(int64_t);
                        // atomic_current_chunk
                        cur += CACHE_LINE_SIZE*n_as + CACHE_LINE_SIZE;
                        // Workspace for tiled (see tiled.h)
                        cur = GGML_PAD(cur, 64);
                        cur += ggml_tiled_wdata_size(n_tasks, node);
                    } اكسر;
                حالة GGML_OP_OUT_PROD:
                    {
                        𐤀𐤌 (ggml_is_quantized(node->src[0]->type) ||
                            node->src[0]->type == GGML_TYPE_F16) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[0]->ne[0] * n_tasks;
                        }
                    } اكسر;
                حالة GGML_OP_SET_ROWS:
                    {
                        𐤀𐤌 (node->src[0]->type == GGML_TYPE_F16 && node->type != GGML_TYPE_F16) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[0]->ne[0] * n_tasks;
                        }
                    } اكسر;
                حالة GGML_OP_SOFT_MAX:
                حالة GGML_OP_ROPE:
                حالة GGML_OP_ROPE_BACK:
                    {
                        cur = ggml_type_size(GGML_TYPE_F32) * node->ne[0] * n_tasks;
                    } اكسر;
                حالة GGML_OP_CONV_TRANSPOSE_1D:
                    {
                        GGML_ASSERT(node->src[0]->ne[3] == 1);
                        GGML_ASSERT(node->src[1]->ne[2] == 1);
                        GGML_ASSERT(node->src[1]->ne[3] == 1);

                        ثابت int64_t ne00 = node->src[0]->ne[0];  // K
                        ثابت int64_t ne01 = node->src[0]->ne[1];  // Cout
                        ثابت int64_t ne02 = node->src[0]->ne[2];  // Cin
                        ثابت int64_t ne10 = node->src[1]->ne[0];  // L
                        ثابت int64_t ne11 = node->src[1]->ne[1];  // Cin

                        𐤀𐤌 ((node->src[0]->type == GGML_TYPE_F16 ||
                             node->src[0]->type == GGML_TYPE_BF16) &&
                            node->src[1]->type == GGML_TYPE_F32) {
                            cur += حجم(ggml_fp16_t)*ne00*ne01*ne02;
                            cur += حجم(ggml_fp16_t)*ne10*ne11;
                        } وإلا 𐤀𐤌 (node->src[0]->type == GGML_TYPE_F32 &&
                                   node->src[1]->type == GGML_TYPE_F32) {
                            cur += حجم(عائم)*ne00*ne01*ne02;
                            cur += حجم(عائم)*ne10*ne11;
                        } وإلا {
                            GGML_ABORT("fatal error");
                        }
                    } اكسر;
                حالة GGML_OP_CONV_2D:
                حالة GGML_OP_CONV_3D:
                    {
                        cur = GGML_IM2COL_WORK_SIZE;
                    } اكسر;
                حالة GGML_OP_CONV_TRANSPOSE_2D:
                    {
                        ثابت int64_t ne00 = node->src[0]->ne[0]; // W
                        ثابت int64_t ne01 = node->src[0]->ne[1]; // H
                        ثابت int64_t ne02 = node->src[0]->ne[2]; // Channels Out
                        ثابت int64_t ne03 = node->src[0]->ne[3]; // Channels In

                        ثابت int64_t ne10 = node->src[1]->ne[0]; // W
                        ثابت int64_t ne11 = node->src[1]->ne[1]; // H
                        ثابت int64_t ne12 = node->src[1]->ne[2]; // Channels In
                        ثابت int64_t ne13 = node->src[1]->ne[3]; // Batch

                        GGML_ASSERT(node->src[0]->type == GGML_TYPE_F16 || node->src[0]->type == GGML_TYPE_F32);
                        GGML_ASSERT(node->src[1]->type == GGML_TYPE_F32);

                        cur += ggml_type_size(node->src[0]->type) * ne00 * ne01 * ne02 * ne03;
                        cur += ggml_type_size(node->src[0]->type) * ne10 * ne11 * ne12 * ne13;

                    } اكسر;
                حالة GGML_OP_TOP_K:
                    {
                        cur += حجم(int32_t)*node->src[0]->ne[0]*n_tasks;
                    } اكسر;
                حالة GGML_OP_FLASH_ATTN_EXT:
                    {
                        ثابت int64_t neq2 = node->src[0]->ne[2]; // number of query heads
                        ثابت int64_t DK = node->src[1]->ne[0];
                        ثابت int64_t DV = node->src[2]->ne[0];

                        // Tiled flash attention scratch (tile sizes defined in common.h)
                        // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
                        size_t prefill  = حجم(عائم)*(GGML_FA_TILE_Q*DK + 2*GGML_FA_TILE_Q*GGML_FA_TILE_KV + GGML_FA_TILE_Q*DV + GGML_FA_TILE_KV*DV + GGML_FA_TILE_KV*DK)*n_tasks;

                        // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
                        // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
                        size_t n_chunks = n_tasks;
                        size_t decode   = حجم(عائم)*(neq2*n_chunks*(2+DV) + n_tasks*(DK + 2*DV));

                        cur += MAX(prefill, decode);
                    } اكسر;
                حالة GGML_OP_FLASH_ATTN_BACK:
                    {
                        ثابت int64_t    D = node->src[0]->ne[0];
                        ثابت int64_t ne11 = ggml_up(node->src[1]->ne[1], GGML_SOFT_MAX_UNROLL);
                        ثابت int64_t mxDn = MAX(D, ne11) * 2; // *2 because of S and SM in ggml_compute_forward_flash_attn_back
                        𐤀𐤌 (node->src[1]->type == GGML_TYPE_F32) {
                            cur  = حجم(عائم)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += حجم(عائم)*mxDn*n_tasks; // this is overestimated by x2
                        } وإلا 𐤀𐤌 (node->src[1]->type == GGML_TYPE_F16) {
                            cur  = حجم(عائم)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += حجم(عائم)*mxDn*n_tasks; // this is overestimated by x2
                        } وإلا 𐤀𐤌 (node->src[1]->type == GGML_TYPE_BF16) {
                            cur  = حجم(عائم)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += حجم(عائم)*mxDn*n_tasks; // this is overestimated by x2
                        }
                    } اكسر;

                حالة GGML_OP_CROSS_ENTROPY_LOSS:
                    {
                        cur = ggml_type_size(node->type)*(n_tasks + node->src[0]->ne[0]*n_tasks);
                    } اكسر;
                حالة GGML_OP_GATED_DELTA_NET:
                    {
                        ثابت int64_t S_v = node->src[2]->ne[0];
                        ثابت int64_t K   = ggml_get_op_params_i32(node, 0);
                        ثابت int64_t per_thread = S_v + (K > 1 ? S_v * S_v : 0);
                        cur = per_thread * حجم(عائم) * n_tasks;
                    } اكسر;
                حالة GGML_OP_COUNT:
                    {
                        GGML_ABORT("fatal error");
                    }
                حالة GGML_OP_LIGHTNING_INDEXER:
                    {
                        // temp buffer for dequantizing lightning indexer keys
                        ثابت int64_t ne10 = node->src[1]->ne[0];
                        cur += حجم(عائم)*ne10*n_tasks;
                    } اكسر;
                افتراضي:
                    اكسر;
            }
        }

        work_size = MAX(work_size, cur);
    }

    𐤀𐤌 (work_size > 0) {
        work_size += CACHE_LINE_SIZE*(n_threads);
    }

    cplan.threadpool = threadpool;
    cplan.n_threads  = MIN(max_tasks, n_threads);
    cplan.work_size  = work_size;
    cplan.work_data  = عدم;

    أرجع cplan;
}


// Try to fuse the current node with subsequent nodes for better performance.
// Returns the number of nodes skipped by fusion (>=1), or 0 if no fusion was applied.
ساكن bool ggml_cpu_disable_fusion = false;  // initialized once in ggml_cpu_init(), read-only afterwards

ساكن صحيح ggml_cpu_try_fuse_ops(
        ثابت بنية ggml_cgraph * cgraph,
        ثابت صحيح node_n,
        ثابت بنية ggml_compute_params * params,
        ثابت بنية ggml_cplan * cplan) {

    𐤀𐤌 (ggml_cpu_disable_fusion || cplan->use_ref) {
        أرجع 0;
    }

    بنية ggml_tensor * node = cgraph->nodes[node_n];

    𐤀𐤌 (node->op == GGML_OP_RMS_NORM) {
        // RMS_NORM + MUL fusion
        ثابت تعداد ggml_op fuse_ops[] = { GGML_OP_RMS_NORM, GGML_OP_MUL };
        𐤀𐤌 (ggml_can_fuse(cgraph, node_n, fuse_ops, 2)) {
            بنية ggml_tensor * mul_node = cgraph->nodes[node_n + 1];
            ثابت بنية ggml_tensor * mul_w = (mul_node->src[0] == node)
                ? mul_node->src[1] : mul_node->src[0];
            𐤀𐤌 (node->src[0]->type  == GGML_TYPE_F32 &&
                mul_node->type      == GGML_TYPE_F32 &&
                mul_w->type         == GGML_TYPE_F32 &&
                mul_w->ne[0]        == node->ne[0]   &&
                mul_w->nb[0]        == حجم(عائم)) {

                ggml_compute_forward_rms_norm_mul_fused(params, node, mul_node);
                أرجع 1;
            }
        }
    }

    أرجع 0;
}

ساكن thread_ret_t ggml_graph_compute_thread(فراغ * data) {
    بنية ggml_compute_state * state = (بنية ggml_compute_state *) data;
    بنية ggml_threadpool    * tp    = state->threadpool;

    ثابت بنية ggml_cgraph * cgraph = tp->cgraph;
    ثابت بنية ggml_cplan  * cplan  = tp->cplan;

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(state->ith);
#وإلا
    set_numa_thread_affinity(state->ith);
#endif

    بنية ggml_compute_params params = {
        /*.ith        =*/ state->ith,
        /*.nth        =*/ atomic_load_explicit(&tp->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK,
        /*.wsize      =*/ cplan->work_size,
        /*.wdata      =*/ cplan->work_data,
        /*.threadpool =*/ tp,
        /*.use_ref    =*/ cplan->use_ref,
    };

#ifdef GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p\n", state->ith, (ثابت فراغ *)cplan);
#وإلا
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p last-graph %d\n", state->ith, (ثابت فراغ *)cplan, state->last_graph);
#endif

    لكل (صحيح node_n = 0; node_n < cgraph->n_nodes && atomic_load_explicit(&tp->abort, memory_order_relaxed) != node_n; node_n++) {
        بنية ggml_tensor * node = cgraph->nodes[node_n];

        𐤀𐤌 (ggml_op_is_empty(node->op)) {
            // skip NOPs
            استمر;
        }

        𐤀𐤌 ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            استمر;
        }

        // TODO: move fused-op detection into ggml_graph_plan so fusion decisions are made once at planning time
        // Try fused ops, fall back to normal compute
        ثابت صحيح n_fused = ggml_cpu_try_fuse_ops(cgraph, node_n, &params, cplan);
        𐤀𐤌 (n_fused > 0) {
            node_n += n_fused;
        } وإلا {
            ggml_compute_forward(&params, node);
        }

        𐤀𐤌 (state->ith == 0 && cplan->abort_callback &&
                cplan->abort_callback(cplan->abort_callback_data)) {
            atomic_store_explicit(&tp->abort, node_n + 1, memory_order_relaxed);
            tp->ec    = GGML_STATUS_ABORTED;
        }

        𐤀𐤌 (node_n + 1 < cgraph->n_nodes) {
            ggml_barrier(state->threadpool);
        }
    }

#ifdef GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p\n", state->ith, (ثابت فراغ *)cplan);
#وإلا
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p last-graph %d\n", state->ith, (ثابت فراغ *)cplan, state->last_graph);
#endif

    ggml_barrier(state->threadpool);

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(state->ith);
#endif

    أرجع 0;
}

#ifndef GGML_USE_OPENMP

// check if thread is ready to proceed (exit from polling or sleeping)
// returns true if loops should exit, sets state->pending to indicate new work
ساكن مضمن bool ggml_graph_compute_thread_ready(بنية ggml_compute_state * state) {
    بنية ggml_threadpool * threadpool = state->threadpool;

    𐤀𐤌 (state->pending || threadpool->stop || threadpool->pause) { أرجع true; }

    // check for new graph/work
    صحيح n_graph   = atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed);
    صحيح n_threads = n_graph & GGML_THREADPOOL_N_THREADS_MASK;
    𐤀𐤌 (n_graph != state->last_graph) {
        state->pending    = (state->ith < n_threads);
        state->last_graph = n_graph;
        أرجع true;
    }

    أرجع false;
}

// sync thread state after polling
ساكن مضمن فراغ ggml_graph_compute_thread_sync(بنية ggml_compute_state * state) {
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&state->threadpool->n_graph, 0, memory_order_seq_cst);
    #وإلا
    atomic_thread_fence(memory_order_seq_cst);
    #endif
    UNUSED(state);
}

ساكن مضمن bool ggml_graph_compute_poll_for_work(بنية ggml_compute_state * state) {
    بنية ggml_threadpool * threadpool = state->threadpool;

    // This seems to make 0 ... 100 a decent range for polling level across modern processors.
    // Perhaps, we can adjust it dynamically based on load and things.
    ثابت uint64_t n_rounds = 1024UL * 128 * threadpool->poll;

    لكل (uint64_t i=0; !ggml_graph_compute_thread_ready(state) && i < n_rounds; i++) {
        // No new work. Keep polling.
        ggml_thread_cpu_relax();
    }

    أرجع state->pending;
}

ساكن مضمن bool ggml_graph_compute_check_for_work(بنية ggml_compute_state * state) {
    بنية ggml_threadpool * threadpool = state->threadpool;

    𐤀𐤌 (ggml_graph_compute_poll_for_work(state)) {
        ggml_graph_compute_thread_sync(state);
        أرجع state->pending;
    }

    ggml_mutex_lock_shared(&threadpool->mutex);
    طالما (!ggml_graph_compute_thread_ready(state)) {
        // No new work. Wait for the signal.
        GGML_PRINT_DEBUG("thread #%d waiting for work (sleeping)\n", state->ith);
        ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
    }
    ggml_mutex_unlock_shared(&threadpool->mutex);

    أرجع state->pending;
}

ساكن thread_ret_t ggml_graph_compute_secondary_thread(فراغ* data) {
    بنية ggml_compute_state * state = (بنية ggml_compute_state *) data;
    بنية ggml_threadpool * threadpool = state->threadpool;

    ggml_thread_apply_priority(threadpool->prio);
    𐤀𐤌 (ggml_thread_cpumask_is_valid(state->cpumask)) {
        ggml_thread_apply_affinity(state->cpumask);
    }

    طالما (true) {
        // Check if we need to sleep
        طالما (threadpool->pause) {
            GGML_PRINT_DEBUG("thread #%d inside pause loop\n", state->ith);
            ggml_mutex_lock_shared(&threadpool->mutex);
            𐤀𐤌 (threadpool->pause) {
                ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
            }
            GGML_PRINT_DEBUG("thread #%d resuming after wait\n", state->ith);
            ggml_mutex_unlock_shared(&threadpool->mutex);
        }

        // This needs to be checked for after the cond_wait
        𐤀𐤌 (threadpool->stop) اكسر;

        // Check if there is new work
        // The main thread is the only one that can dispatch new work

        ggml_graph_compute_check_for_work(state);
        𐤀𐤌 (state->pending) {
            state->pending = false;
            ggml_graph_compute_thread(state);
        }
    }

    أرجع (thread_ret_t) 0;
}

// Start processing new graph
ساكن فراغ ggml_graph_compute_kickoff(بنية ggml_threadpool * threadpool, صحيح n_threads)
{
    // Always take the mutex here because the worker threads are doing hybrid poll/wait

    ggml_mutex_lock(&threadpool->mutex);

    // Update the number of active threads and the graph count
    صحيح n_graph = atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed) >> GGML_THREADPOOL_N_THREADS_BITS;
    n_graph = ((n_graph + 1) << GGML_THREADPOOL_N_THREADS_BITS) | (n_threads & GGML_THREADPOOL_N_THREADS_MASK);

    GGML_PRINT_DEBUG("compute-kickoff: n_threads %d n_graph %d\n", n_threads, n_graph);

    // Indicate the graph is ready to be processed
    // We need the full seq-cst fence here because of the polling threads (used in thread_sync)
    atomic_store_explicit(&threadpool->n_graph, n_graph, memory_order_seq_cst);

    𐤀𐤌 (threadpool->pause) {
       // Update main thread prio and affinity to match the threadpool settings
       ggml_thread_apply_priority(threadpool->prio);
       𐤀𐤌 (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
           ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
       }

       // resume does cond broadcast
       ggml_threadpool_resume_locked(threadpool);
    } وإلا {
       ggml_cond_broadcast(&threadpool->cond);
    }

    ggml_mutex_unlock(&threadpool->mutex);
}

#endif // GGML_USE_OPENMP

ساكن بنية ggml_threadpool * ggml_threadpool_new_impl(
    بنية ggml_threadpool_params * tpp,
               بنية ggml_cgraph * cgraph,
                بنية ggml_cplan * cplan) {

    بنية ggml_threadpool * threadpool =
        ggml_aligned_malloc(حجم(بنية ggml_threadpool));
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
        threadpool->workers          = عدم;
        threadpool->n_threads        = tpp->n_threads;
        threadpool->poll             = tpp->poll;
        threadpool->prio             = tpp->prio;
        threadpool->ec               = GGML_STATUS_SUCCESS;
    }

    // Allocate and init workers state
    ثابت size_t workers_size = حجم(بنية ggml_compute_state) * tpp->n_threads;
    بنية ggml_compute_state * workers = ggml_aligned_malloc(workers_size);

    memset(workers, 0, workers_size);
    لكل (صحيح j = 0; j < tpp->n_threads; j++) {
        workers[j].threadpool = threadpool;
        workers[j].ith        = j;
    }

    threadpool->workers = workers;

#ifdef GGML_USE_OPENMP
    int32_t cpumask_iter = 0;

    // Compute CPU masks for each thread
    لكل (صحيح j = 0; j < tpp->n_threads; j++) {
        ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);
    }
#وإلا // GGML_USE_OPENMP
    ggml_mutex_init(&threadpool->mutex);
    ggml_cond_init(&threadpool->cond);

    // Spin the threads for all workers, and update CPU placements.
    // Place the main thread last (towards the higher numbered CPU cores).

    int32_t cpumask_iter = 0;

    لكل (صحيح j = 1; j < tpp->n_threads; j++) {
        ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);

        int32_t rc = ggml_thread_create(&workers[j].thrd, عدم, ggml_graph_compute_secondary_thread, &workers[j]);
        GGML_ASSERT(rc == 0);
    }

    ggml_thread_cpumask_next(tpp->cpumask, workers[0].cpumask, tpp->strict_cpu, &cpumask_iter);

    𐤀𐤌 (!threadpool->pause) {
        // Update main thread prio and affinity at the start, otherwise we'll do it in resume
        ggml_thread_apply_priority(threadpool->prio);
        𐤀𐤌 (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
            ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
        }
    }
#endif // GGML_USE_OPENMP

    أرجع threadpool;
}

بنية ggml_threadpool * ggml_threadpool_new(بنية ggml_threadpool_params * tpp) {
    أرجع ggml_threadpool_new_impl(tpp, عدم, عدم);
}

تعداد ggml_status ggml_graph_compute(بنية ggml_cgraph * cgraph, بنية ggml_cplan * cplan) {
    ggml_cpu_init();

    GGML_ASSERT(cplan);
    GGML_ASSERT(cplan->n_threads > 0);
    GGML_ASSERT(cplan->work_size == 0 || cplan->work_data != عدم);

    صحيح n_threads                               = cplan->n_threads;
    بنية ggml_threadpool * threadpool = cplan->threadpool;

    bool disposable_threadpool = false;

    𐤀𐤌 (threadpool == عدم) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
        disposable_threadpool = true;

        بنية ggml_threadpool_params ttp = ggml_threadpool_params_default(n_threads);
        threadpool = ggml_threadpool_new_impl(&ttp, cgraph, cplan);
    } وإلا {
        // Reset some of the parameters that need resetting
        // No worker threads should be accessing the parameters below at this stage
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->current_chunk    = 0;
        threadpool->abort            = -1;
        threadpool->ec               = GGML_STATUS_SUCCESS;
    }

#ifdef GGML_USE_OPENMP
    𐤀𐤌 (n_threads > 1) {
        #pragma omp parallel num_threads(n_threads)
        {
            #pragma omp single
            {
                // update the number of threads from the actual number of threads that we got from OpenMP
                n_threads = omp_get_num_threads();
                atomic_store_explicit(&threadpool->n_graph, n_threads, memory_order_relaxed);
            }

            // Apply thread CPU mask and priority
            صحيح ith = omp_get_thread_num();

            ggml_thread_apply_priority(threadpool->prio);
            𐤀𐤌 (ggml_thread_cpumask_is_valid(threadpool->workers[ith].cpumask)) {
                ggml_thread_apply_affinity(threadpool->workers[ith].cpumask);
            }
            ggml_graph_compute_thread(&threadpool->workers[ith]);
        }
    } وإلا {
        atomic_store_explicit(&threadpool->n_graph, 1, memory_order_relaxed);
        ggml_graph_compute_thread(&threadpool->workers[0]);
    }
#وإلا
    𐤀𐤌 (n_threads > threadpool->n_threads) {
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

    تعداد ggml_status ret = threadpool->ec;

    𐤀𐤌 (disposable_threadpool) {
        ggml_threadpool_free(threadpool);
    }

    أرجع ret;
}

تعداد ggml_status ggml_graph_compute_with_ctx(بنية ggml_context * ctx, بنية ggml_cgraph * cgraph, صحيح n_threads) {
    بنية ggml_cplan cplan = ggml_graph_plan(cgraph, n_threads, عدم);

    cplan.work_data = (uint8_t *)ggml_new_buffer(ctx, cplan.work_size);

    أرجع ggml_graph_compute(cgraph, &cplan);
}

فراغ ggml_cpu_fp32_to_fp32(ثابت عائم * x, عائم * y, int64_t n) {
    memcpy(y, x, n * حجم(عائم));
}

فراغ ggml_cpu_fp32_to_fp16(ثابت عائم * x, ggml_fp16_t * y, int64_t n) {
    int64_t i = 0;
#𐤀𐤌 defined(__F16C__)
#𐤀𐤌 defined(__AVX512F__)
    لكل (; i + 15 < n; i += 16) {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        __m256i y_vec = _mm512_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256((__m256i *)(y + i), y_vec);
    }
#endif
    لكل (; i + 7 < n; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m128i y_vec = _mm256_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i *)(y + i), y_vec);
    }
    لكل (; i + 3 < n; i += 4) {
        __m128 x_vec = _mm_loadu_ps(x + i);
        __m128i y_vec = _mm_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storel_epi64((__m128i *)(y + i), y_vec);
    }
#وإلاإذا defined(__riscv_zvfh)
    لكل (صحيح vl; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(&x[i], vl);
        vfloat16m1_t vy = __riscv_vfncvt_f_f_w_f16m1(vx, vl);
        __riscv_vse16_v_f16m1((_Float16 *)&y[i], vy, vl);
    }
#وإلاإذا defined(__VXE__) || defined(__VXE2__)
    لكل (; i + 7 < n; i += 8) {
        ثابت uint32x4_t v_yl = __lzs_f32cx4_to_f16(vec_xl(0, x + i + 0));
        ثابت uint32x4_t v_yh = __lzs_f32cx4_to_f16(vec_xl(0, x + i + 4));
        vec_xst(vec_pack(v_yl, v_yh), 0, (uint16_t *)(y + i));
    }
#endif
    لكل (; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(x[i]);
    }
}

فراغ ggml_cpu_fp16_to_fp32(ثابت ggml_fp16_t * x, عائم * y, int64_t n) {
    int64_t i = 0;
#𐤀𐤌 defined(__F16C__)
#𐤀𐤌 defined(__AVX512F__)
    لكل (; i + 15 < n; i += 16) {
        __m256i x_vec = _mm256_loadu_si256((ثابت __m256i *)(x + i));
        __m512 y_vec = _mm512_cvtph_ps(x_vec);
        _mm512_storeu_ps(y + i, y_vec);
    }
#endif
    لكل (; i + 7 < n; i += 8) {
        __m128i x_vec = _mm_loadu_si128((ثابت __m128i *)(x + i));
        __m256 y_vec = _mm256_cvtph_ps(x_vec);
        _mm256_storeu_ps(y + i, y_vec);
    }
    لكل (; i + 3 < n; i += 4) {
        __m128i x_vec = _mm_loadl_epi64((ثابت __m128i *)(x + i));
        __m128 y_vec = _mm_cvtph_ps(x_vec);
        _mm_storeu_ps(y + i, y_vec);
    }

#وإلاإذا defined(__riscv_v_intrinsic) && defined(__riscv_zvfhmin)
    // calculate step size
    ثابت صحيح epr = __riscv_vsetvlmax_e16m2();
    ثابت صحيح step = epr * 2;
    ثابت صحيح np = (n & ~(step - 1));

    // unroll by 2
    لكل (; i < np; i += step) {
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((ثابت _Float16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vfloat16m2_t ax1 = __riscv_vle16_v_f16m2((ثابت _Float16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvt_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    صحيح vl;
    لكل (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((ثابت _Float16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }

#endif

    لكل (; i < n; ++i) {
        y[i] = GGML_CPU_FP16_TO_FP32(x[i]);
    }
}

فراغ ggml_cpu_fp32_to_bf16(ثابت عائم * x, ggml_bf16_t * y, int64_t n) {
    int64_t i = 0;
    لكل (; i < n; ++i) {
        y[i] = GGML_FP32_TO_BF16(x[i]);
    }
}

فراغ ggml_cpu_fp32_to_i32(ثابت عائم * x, int32_t * y, int64_t n) {
    int64_t i = 0;
    لكل (; i < n; ++i) {
        y[i] = x[i];
    }
}

فراغ ggml_cpu_bf16_to_fp32(ثابت ggml_bf16_t * x, عائم * y, int64_t n) {
    int64_t i = 0;
#𐤀𐤌 defined(__AVX2__)
#𐤀𐤌 defined(__AVX512F__)
    لكل (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i,
                        _mm512_castsi512_ps(
                            _mm512_slli_epi32(
                                _mm512_cvtepu16_epi32(
                                    _mm256_loadu_si256(
                                        (ثابت __m256i *)(x + i))),
                                16)));
    }
#endif
    لكل (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i,
                        _mm256_castsi256_ps(
                            _mm256_slli_epi32(
                                _mm256_cvtepu16_epi32(
                                    _mm_loadu_si128(
                                        (ثابت __m128i *)(x + i))),
                                16)));
    }
#وإلاإذا defined(__riscv_v_intrinsic) && defined(__riscv_zvfbfmin)
    // calculate step size
    ثابت صحيح epr = __riscv_vsetvlmax_e16m2();
    ثابت صحيح step = epr * 2;
    ثابت صحيح np = (n & ~(step - 1));

    // unroll by 2
    لكل (; i < np; i += step) {
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((ثابت __bf16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vbfloat16m2_t ax1 = __riscv_vle16_v_bf16m2((ثابت __bf16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    صحيح vl;
    لكل (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((ثابت __bf16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }
#endif
    لكل (; i < n; i++) {
        y[i] = GGML_BF16_TO_FP32(x[i]);
    }
}

صحيح ggml_cpu_has_avx(فراغ) {
#𐤀𐤌 defined(__AVX__)
    أرجع 1;
#وإلا
    أرجع 0;
#endif
}

صحيح ggml_cpu_has_avx_vnni(فراغ) {
#𐤀𐤌 defined(__AVXVNNI__)
    أرجع 1;
#وإلا
    أرجع 0;
#endif
}

صحيح ggml_cpu_has_avx2(فراغ) {
#𐤀𐤌 defined(__AVX2__)
    أرجع 1;
#وإلا
    أرجع 0;
#endif
}

صحيح ggml_cpu_has_avx512(فراغ) {
#𐤀𐤌 defined(__AVX512F__)
    أرجع 1;
#وإلا
    أرجع 0;
#endif
}

صحيح ggml_cpu_has_avx512_vbmi(فراغ) {
#𐤀𐤌 defined(__AVX512VBMI__)
    أرجع 1;
#وإلا
    أرجع 0;
#endif
}

صحيح ggml_cpu_has_avx512_vnni(فراغ) {
#𐤀𐤌 defined(__AVX512VNNI__)
    أرجع 1;
#وإلا
    أرجع 0;
#endif
}

صحيح ggml_cpu_has_avx512_bf16(فراغ) {
#𐤀𐤌 defined(__AVX512BF16__)
    أرجع 1;
#وإلا
    أرجع 0;
#endif
}

صحيح ggml_cpu_has_amx_int8(فراغ) {
#𐤀𐤌 defined(__AMX_INT8__)
    أرجع 1;
#وإلا
    أرجع 0;
#endif
}

صحيح ggml_cpu_has_bmi2(فراغ) {
#𐤀𐤌 defined(__BMI2__)
    أرجع 1;
#وإلا
    أرجع 0;
#endif
}

صحيح ggml_cpu_has_fma(فراغ) {
#𐤀𐤌 defined(__FMA__)
    أرجع 1;
#وإلا
    أرجع 0;
#endif
}

صحيح ggml_cpu_has_arm_fma(فراغ) {
#𐤀𐤌 defined(__ARM_FEATURE_FMA)
    أرجع 1;
#وإلا
    أرجع 0;
#endif
}

صحيح ggml_cpu_has_riscv_v(فراغ) {
#𐤀𐤌 defined(__riscv_v_intrinsic)
    أرجع 1;
#وإلا
    أرجع 0;
#endif
}

صحيح ggml_cpu_get_rvv_vlen(فراغ) {
#𐤀𐤌 defined(__riscv) && defined(__riscv_v_intrinsic)
    أرجع ggml_riscv_arch_features.rvv_vlen;
#وإلا
    أرجع 0;
#endif
}

صحيح ggml_cpu_has_f16c(فراغ) {
#𐤀𐤌 defined(__F16C__)
    أرجع 1;
#وإلا
    أرجع 0;
#endif
}

صحيح ggml_cpu_has_fp16_va(فراغ) {
#𐤀𐤌 defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    أرجع 1;
#وإلا
    أرجع 0;
#endif
}

صحيح ggml_cpu_has_wasm_simd(فراغ) {
#𐤀𐤌 defined(__wasm_simd128__)
    أرجع 1;
#وإلا
    أرجع 0;
#endif
}

صحيح ggml_cpu_has_llamafile(فراغ) {
#𐤀𐤌 defined(GGML_USE_LLAMAFILE)
    أرجع 1;
#وإلا
    أرجع 0;
#endif
}

صحيح ggml_cpu_has_sse3(فراغ) {
#𐤀𐤌 defined(__SSE3__)
    أرجع 1;
#وإلا
    أرجع 0;
#endif
}

صحيح ggml_cpu_has_ssse3(فراغ) {
#𐤀𐤌 defined(__SSSE3__)
    أرجع 1;
#وإلا
    أرجع 0;
#endif
}

صحيح ggml_cpu_has_vsx(فراغ) {
#𐤀𐤌 defined(__POWER9_VECTOR__)
    أرجع 1;
#وإلا
    أرجع 0;
#endif
}

صحيح ggml_cpu_has_vxe(فراغ) {
#𐤀𐤌 defined(__VXE__) || defined(__VXE2__)
    أرجع 1;
#وإلا
    أرجع 0;
#endif
}

صحيح ggml_cpu_has_neon(فراغ) {
#𐤀𐤌 defined(__ARM_ARCH) && defined(__ARM_NEON)
    أرجع 1;
#وإلا
    أرجع 0;
#endif
}

صحيح ggml_cpu_has_dotprod(فراغ) {
#𐤀𐤌 defined(__ARM_ARCH) && defined(__ARM_FEATURE_DOTPROD)
    أرجع 1;
#وإلا
    أرجع 0;
#endif
}

صحيح ggml_cpu_has_sve(فراغ) {
#𐤀𐤌 defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    أرجع 1;
#وإلا
    أرجع 0;
#endif
}

صحيح ggml_cpu_has_matmul_int8(فراغ) {
#𐤀𐤌 defined(__ARM_ARCH) && defined(__ARM_FEATURE_MATMUL_INT8)
    أرجع 1;
#وإلا
    أرجع 0;
#endif
}

صحيح ggml_cpu_get_sve_cnt(فراغ) {
#𐤀𐤌 defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    أرجع ggml_arm_arch_features.sve_cnt;
#وإلا
    أرجع 0;
#endif
}

صحيح ggml_cpu_has_sme(فراغ) {
#𐤀𐤌 defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME)
    أرجع 1;
#وإلا
    أرجع 0;
#endif
}

صحيح ggml_cpu_has_sme2(فراغ) {
#𐤀𐤌 defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME2)
    أرجع 1;
#وإلا
    أرجع 0;
#endif
}

فراغ ggml_cpu_init(فراغ) {
    // needed to initialize ggml_time
    {
        بنية ggml_init_params params = { 0, عدم, false };
        بنية ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    ggml_critical_section_start();

    ساكن bool is_first_call = true;

    𐤀𐤌 (is_first_call) {
        // initialize GELU, Quick GELU, SILU and EXP F32 tables
        {
            ثابت uint64_t t_start = ggml_time_us(); UNUSED(t_start);

            لكل (صحيح i = 0; i < (1 << 16); ++i) {
                اتحاد {
                    uint16_t u16;
                    ggml_fp16_t fp16;
                } u = {i};
                عائم f = GGML_COMPUTE_FP16_TO_FP32(u.fp16);
                ggml_table_f32_f16[i] = f;
                ggml_table_gelu_f16[i] = GGML_CPU_FP32_TO_FP16(ggml_gelu_f32(f));
                ggml_table_gelu_quick_f16[i] = GGML_CPU_FP32_TO_FP16(ggml_gelu_quick_f32(f));
            }

            // initialize E8M0 half table (256 entries)
            لكل (صحيح i = 0; i < (1 << 8); ++i) {
                ggml_table_f32_e8m0_half[i] = GGML_E8M0_TO_FP32_HALF(i);
            }

            // initialize UE4M3 table (256 entries)
            لكل (صحيح i = 0; i < (1 << 8); ++i) {
                ggml_table_f32_ue4m3[i] = ggml_ue4m3_to_fp32(i);
            }

            ثابت uint64_t t_end = ggml_time_us(); UNUSED(t_end);

            GGML_PRINT_DEBUG("%s: GELU, Quick GELU, SILU and EXP tables initialized in %f ms\n", __func__, (t_end - t_start)/1000.0);

#ifdef GGML_USE_OPENMP
            //if (!getenv("OMP_WAIT_POLICY")) {
            //    // set the wait policy to active, so that OpenMP threads don't sleep
            //    setenv("OMP_WAIT_POLICY", "active", 0)
            //}

            𐤀𐤌 (!getenv("KMP_BLOCKTIME")) {
                // set the time to wait before sleeping a thread
                // this is less aggressive than setting the wait policy to active, but should achieve similar results in most cases
#ifdef _WIN32
                _putenv_s("KMP_BLOCKTIME", "200"); // 200ms
#وإلا
                setenv("KMP_BLOCKTIME", "200", 0); // 200ms
#endif
            }
#endif
        }

#𐤀𐤌 defined(__ARM_ARCH)
        ggml_init_arm_arch_features();
#endif

#𐤀𐤌 defined(__riscv)
        ggml_init_riscv_arch_features();
#endif

        {
            ثابت حرف * env = getenv("GGML_CPU_DISABLE_FUSION");
            ggml_cpu_disable_fusion = (env != عدم && atoi(env) == 1);
        }

        is_first_call = false;
    }

    ggml_critical_section_end();
}
