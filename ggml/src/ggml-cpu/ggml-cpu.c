#define _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#define _USE_MATH_DEFINES // For M_PI on MSVC

#अन्तर्भाव "ggml-backend-impl.h"
#अन्तर्भाव "ggml-backend.h"
#अन्तर्भाव "traits.h"
#अन्तर्भाव "ggml-cpu-impl.h"
#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "quants.h"
#अन्तर्भाव "ggml-threading.h"
#अन्तर्भाव "unary-ops.h"
#अन्तर्भाव "binary-ops.h"
#अन्तर्भाव "vec.h"
#अन्तर्भाव "ops.h"
#अन्तर्भाव "ggml.h"
#अन्तर्भाव "common.h"
#अन्तर्भाव "tiled/tiled.h"

#जर defined(_MSC_VER) || defined(__MINGW32__)
#अन्तर्भाव <malloc.h> // using malloc.h with MSC/MINGW
#किंवाजर !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#अन्तर्भाव <alloca.h>
#endif

#अन्तर्भाव <assert.h>
#अन्तर्भाव <errno.h>
#अन्तर्भाव <time.h>
#अन्तर्भाव <math.h>
#अन्तर्भाव <stdlib.h>
#अन्तर्भाव <string.h>
#अन्तर्भाव <stdint.h>
#अन्तर्भाव <inttypes.h>
#अन्तर्भाव <stdio.h>
#अन्तर्भाव <float.h>
#अन्तर्भाव <limits.h>
#अन्तर्भाव <stdarg.h>
#अन्तर्भाव <signal.h>
#जर defined(__gnu_linux__)
#अन्तर्भाव <syscall.h>
#endif

#ifdef GGML_USE_OPENMP
#अन्तर्भाव <omp.h>
#endif

#जर defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_MATMUL_INT8)
#undef GGML_USE_LLAMAFILE
#endif

#ifdef GGML_USE_LLAMAFILE
#अन्तर्भाव "llamafile/sgemm.h"
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
#    include "spacemit/ime.h"
#endif

// Note: once we move threading into a separate C++ file
// will use std::hardware_destructive_interference_size instead of hardcoding it here
// and we'll use C++ attribute syntax.
#define GGML_CACHE_ओळ  64

#जर defined(__clang__) || defined(__GNUC__)
#define GGML_CACHE_ALIGN __attribute__((aligned(GGML_CACHE_ओळ)))
#endif

#जर defined(__has_feature)
#जर __has_feature(thread_sanitizer)
#define GGML_TSAN_ENABLED 1
#endif
#अन्यथा  // __has_feature
#जर defined(__SANITIZE_THREAD__)
#define GGML_TSAN_ENABLED 1
#endif
#endif // __has_feature

#define UNUSED GGML_UNUSED
#define SWAP(x, y, T) करा { T SWAP = x; (x) = y; (y) = SWAP; } पर्यंत (0)

// precomputed f32 table for f16 (256 KB) (simd-mappings.h)
दशांश ggml_table_f32_f16[1 << 16];

// precomputed f32 table for e8m0 half (1 KB) (simd-mappings.h)
दशांश ggml_table_f32_e8m0_half[1 << 8];

// precomputed f32 table for ue4m3 (1 KB) (simd-mappings.h)
दशांश ggml_table_f32_ue4m3[1 << 8];

#जर defined(__ARM_ARCH)
संरचना ggml_arm_arch_features_type {
    पूर्णांक sve_cnt;
} ggml_arm_arch_features = { 0 };
#endif

#जर defined(__riscv)
संरचना ggml_riscv_arch_features_type {
    पूर्णांक rvv_vlen;
} ggml_riscv_arch_features = { 0 };
#endif

#जर defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#अन्तर्भाव <windows.h>

#जर defined(_MSC_VER) && !defined(__clang__)
#define GGML_CACHE_ALIGN __declspec(align(GGML_CACHE_ओळ))

प्रकारव्याख्या अस्थिर LONG atomic_int;
प्रकारव्याख्या atomic_int atomic_bool;
प्रकारव्याख्या atomic_int atomic_flag;

#define ATOMIC_FLAG_INIT 0

प्रकारव्याख्या गणना {
    memory_order_relaxed,
    memory_order_consume,
    memory_order_acquire,
    memory_order_release,
    memory_order_acq_rel,
    memory_order_seq_cst
} memory_order;

स्थिर शून्य atomic_store(atomic_int * ptr, LONG val) {
    InterlockedExchange(ptr, val);
}
स्थिर शून्य atomic_store_explicit(atomic_int * ptr, LONG val, memory_order mo) {
    // TODO: add support for explicit memory order
    InterlockedExchange(ptr, val);
}
स्थिर LONG atomic_load(atomic_int * ptr) {
    परत InterlockedCompareExchange(ptr, 0, 0);
}
स्थिर LONG atomic_load_explicit(atomic_int * ptr, memory_order mo) {
    // TODO: add support for explicit memory order
    परत InterlockedCompareExchange(ptr, 0, 0);
}
स्थिर LONG atomic_fetch_add(atomic_int * ptr, LONG inc) {
    परत InterlockedExchangeAdd(ptr, inc);
}
स्थिर LONG atomic_fetch_add_explicit(atomic_int * ptr, LONG inc, memory_order mo) {
    // TODO: add support for explicit memory order
    परत InterlockedExchangeAdd(ptr, inc);
}
स्थिर atomic_bool atomic_flag_test_and_set(atomic_flag * ptr) {
    परत InterlockedExchange(ptr, 1);
}
स्थिर शून्य atomic_flag_clear(atomic_flag * ptr) {
    InterlockedExchange(ptr, 0);
}
स्थिर शून्य atomic_thread_fence(memory_order mo) {
    MemoryBarrier();
}
#अन्यथा // clang
#अन्तर्भाव <stdatomic.h>
#endif

प्रकारव्याख्या HANDLE pthread_t;

प्रकारव्याख्या DWORD thread_ret_t;
स्थिर पूर्णांक pthread_create(pthread_t * out, शून्य * unused, thread_ret_t(*func)(शून्य *), शून्य * arg) {
    (शून्य) unused;
    HANDLE handle = CreateThread(रिक्त, 0, (LPTHREAD_START_ROUTINE) func, arg, 0, रिक्त);
    जर (handle == रिक्त)
    {
        परत EAGAIN;
    }

    *out = handle;
    परत 0;
}

स्थिर पूर्णांक pthread_join(pthread_t thread, शून्य * unused) {
    (शून्य) unused;
    पूर्णांक ret = (पूर्णांक) WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    परत ret;
}

स्थिर पूर्णांक sched_yield (शून्य) {
    Sleep (0);
    परत 0;
}
#अन्यथा

#अन्तर्भाव <pthread.h>
#अन्तर्भाव <stdatomic.h>
#अन्तर्भाव <sched.h>
#जर defined(__FreeBSD__)
#अन्तर्भाव <pthread_np.h>
#endif

प्रकारव्याख्या शून्य * thread_ret_t;

#अन्तर्भाव <sys/types.h>
#अन्तर्भाव <sys/stat.h>
#अन्तर्भाव <unistd.h>

#endif

प्रकारव्याख्या pthread_t ggml_thread_t;

#define GGML_THREADPOOL_N_THREADS_MASK (0xffffU)
#define GGML_THREADPOOL_N_THREADS_BITS (16)

#जर defined(__APPLE__)
#अन्तर्भाव <unistd.h>
#अन्तर्भाव <mach/mach.h>
#अन्तर्भाव <TargetConditionals.h>
#endif

स्थिर स्थिरांक संरचना ggml_type_traits_cpu type_traits_cpu[GGML_TYPE_मोजणी] = {
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
#जर defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#अन्यथा
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q4_1] = {
        .from_float               = quantize_row_q4_1,
        .vec_dot                  = ggml_vec_dot_q4_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
#जर defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#अन्यथा
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
#जर defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#अन्यथा
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
#जर defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#अन्यथा
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
#जर defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#अन्यथा
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_IQ2_XXS] = {
        .from_float               = रिक्त,
        .vec_dot                  = ggml_vec_dot_iq2_xxs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_XS] = {
        .from_float               = रिक्त,
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
        .from_float               = रिक्त,
        .vec_dot                  = ggml_vec_dot_iq1_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_M] = {
        .from_float               = रिक्त,
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

स्थिरांक संरचना ggml_type_traits_cpu * ggml_get_type_traits_cpu(गणना ggml_type type) {
    परत &type_traits_cpu[type];
}

//
// Threading defs
//

प्रकारव्याख्या pthread_t          ggml_thread_t;

#जर defined(_WIN32)

प्रकारव्याख्या CONDITION_VARIABLE ggml_cond_t;
प्रकारव्याख्या SRWLOCK            ggml_mutex_t;

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

#अन्यथा

प्रकारव्याख्या pthread_cond_t     ggml_cond_t;
प्रकारव्याख्या pthread_mutex_t    ggml_mutex_t;

#define ggml_mutex_init(m)          pthread_mutex_init(m, रिक्त)
#define ggml_mutex_destroy(m)       pthread_mutex_destroy(m)
#define ggml_mutex_lock(m)          pthread_mutex_lock(m)
#define ggml_mutex_unlock(m)        pthread_mutex_unlock(m)
#define ggml_mutex_lock_shared(m)   pthread_mutex_lock(m)
#define ggml_mutex_unlock_shared(m) pthread_mutex_unlock(m)

#define ggml_lock_init(x)    UNUSED(x)
#define ggml_lock_destroy(x) UNUSED(x)
#जर defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64) && !defined(_M_ARM64EC))
#define ggml_lock_lock(x)    _mm_pause()
#अन्यथा
#define ggml_lock_lock(x)    UNUSED(x)
#endif
#define ggml_lock_unlock(x)  UNUSED(x)

#define GGML_LOCK_INITIALIZER 0
#define ggml_cond_init(c)      pthread_cond_init(c, रिक्त)
#define ggml_cond_destroy(c)   pthread_cond_destroy(c)
#define ggml_cond_wait(c, m)   pthread_cond_wait(c, m)
#define ggml_cond_broadcast(c) pthread_cond_broadcast(c)

#define ggml_thread_create pthread_create
#define ggml_thread_join   pthread_join

#endif

// Threadpool def
संरचना ggml_threadpool {
    ggml_mutex_t mutex;       // mutex for cond.var
    ggml_cond_t  cond;        // cond.var for waiting for new work

    संरचना ggml_cgraph * cgraph;
    संरचना ggml_cplan  * cplan;

    // synchronization primitives
    atomic_int n_graph;       // updated when there is work to be done (i.e each graph) holds graph and active thread counts.
    atomic_int GGML_CACHE_ALIGN n_barrier;
    atomic_int GGML_CACHE_ALIGN n_barrier_passed;
    atomic_int GGML_CACHE_ALIGN current_chunk; // currently processing chunk during Mat_Mul, shared between all the threads.

    // these are atomic as an annotation for thread-sanitizer
    atomic_bool stop;         // Used for stopping the threadpool altogether
    atomic_bool छोड़ो;        // Used for pausing the threadpool or individual threads
    atomic_int  पात;        // Used for aborting processing of a graph

    संरचना ggml_compute_state * workers;   // per thread state
    पूर्णांक          n_threads;   // Number of threads in the pool
    int32_t      prio;        // Scheduling priority
    uint32_t     poll;        // Polling level (0 - no polling)

    गणना ggml_status ec;
};

// Per-thread state
संरचना ggml_compute_state {
#ifndef GGML_USE_OPENMP
    ggml_thread_t thrd;
    पूर्णांक  last_graph;
    bool pending;
#endif
    bool cpumask[GGML_MAX_N_THREADS];
    संरचना ggml_threadpool * threadpool;
    पूर्णांक ith;
};

// Helpers for polling loops
#जर defined(__aarch64__) && ( defined(__clang__) || defined(__GNUC__) )
स्थिर ओळीत शून्य ggml_thread_cpu_relax(शून्य) {
    __asm__ अस्थिर("yield" ::: "memory");
}
#किंवाजर defined(__x86_64__)
स्थिर ओळीत शून्य ggml_thread_cpu_relax(शून्य) {
    _mm_pause();
}
#किंवाजर defined(__riscv)
स्थिर ओळीत शून्य ggml_thread_cpu_relax(शून्य) {
    #ifdef __riscv_zihintpause
        __asm__ __volatile__ ("pause");
    #अन्यथा
        /* Encoding of the pause instruction */
        __asm__ __volatile__ (".4byte 0x100000F");
    #endif
}
#अन्यथा
स्थिर ओळीत शून्य ggml_thread_cpu_relax(शून्य) {;}
#endif

//
// NUMA support
//

#define GGML_NUMA_MAX_NODES 8
#define GGML_NUMA_MAX_CPUS 512

संरचना ggml_numa_गाठ {
    uint32_t cpus[GGML_NUMA_MAX_CPUS]; // hardware threads on this node
    uint32_t n_cpus;
};

संरचना ggml_numa_nodes {
    गणना ggml_numa_strategy numa_strategy;
    संरचना ggml_numa_गाठ nodes[GGML_NUMA_MAX_NODES];
    uint32_t n_nodes;
    uint32_t total_cpus; // hardware threads on system
    uint32_t current_गाठ; // node on which main process is execting
#जर defined(__gnu_linux__)
    cpu_set_t cpuset; // cpuset from numactl
#अन्यथा
    uint32_t cpuset; // no NUMA support outside of Linux at this time. Use a portable datatype
#endif
};

//
// ggml state
//

संरचना ggml_state {
    संरचना ggml_numa_nodes numa;
};

स्थिर संरचना ggml_state g_state = {0};

शून्य ggml_barrier(संरचना ggml_threadpool * tp) {
    पूर्णांक n_threads = atomic_load_explicit(&tp->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK;
    जर (n_threads == 1) {
        परत;
    }

#ifdef GGML_USE_OPENMP
    #pragma omp barrier
#अन्यथा
    पूर्णांक n_passed = atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed);

    // enter barrier (full seq-cst fence)
    पूर्णांक n_barrier = atomic_fetch_add_explicit(&tp->n_barrier, 1, memory_order_seq_cst);

    जर (n_barrier == (n_threads - 1)) {
        // last thread
        atomic_store_explicit(&tp->n_barrier, 0, memory_order_relaxed);

        // exit barrier (full seq-cst fence)
        atomic_fetch_add_explicit(&tp->n_barrier_passed, 1, memory_order_seq_cst);
        परत;
    }

    // wait for other threads
    पर्यंत (atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed) == n_passed) {
        ggml_thread_cpu_relax();
    }

    // exit barrier (full seq-cst fence)
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&tp->n_barrier_passed, 0, memory_order_seq_cst);
    #अन्यथा
    atomic_thread_fence(memory_order_seq_cst);
    #endif
#endif
}

शून्य ggml_threadpool_chunk_set(संरचना ggml_threadpool * tp, पूर्णांक मूल्य) {
    atomic_store_explicit(&tp->current_chunk, मूल्य, memory_order_relaxed);
}

पूर्णांक ggml_threadpool_chunk_add(संरचना ggml_threadpool * tp, पूर्णांक मूल्य) {
    परत atomic_fetch_add_explicit(&tp->current_chunk, मूल्य, memory_order_relaxed);
}

#जर defined(__gnu_linux__)
स्थिर cpu_set_t ggml_get_numa_affinity(शून्य) {
    cpu_set_t cpuset;
    pthread_t thread;
    thread = pthread_self();
    CPU_ZERO(&cpuset);
    pthread_getaffinity_np(thread, आकार(cpu_set_t), &cpuset);
    परत cpuset;
}
#अन्यथा
स्थिर uint32_t ggml_get_numa_affinity(शून्य) {
    परत 0; // no NUMA support
}
#endif

शून्य ggml_numa_init(गणना ggml_numa_strategy numa_flag) {
    जर (g_state.numa.n_nodes > 0) {
        ख_लिखो(मानक_त्रुटि, "ggml_numa_init: NUMA already initialized\n");

        परत;
    }

#जर defined(__gnu_linux__)
    संरचना stat st;
    अक्षर path[256];
    पूर्णांक rv;

    // set numa scheme
    g_state.numa.numa_strategy = numa_flag;

    GGML_PRINT_DEBUG("numa strategy %u\n",g_state.numa.numa_strategy);

    g_state.numa.cpuset = ggml_get_numa_affinity();

    // enumerate nodes
    पर्यंत (g_state.numa.n_nodes < GGML_NUMA_MAX_NODES) {
        rv = snprintf(path, आकार(path), "/sys/devices/system/node/node%u", g_state.numa.n_nodes);
        GGML_ASSERT(rv > 0 && (अचिन्हित)rv < आकार(path));
        जर (stat(path, &st) != 0) { थांबा; }
        ++g_state.numa.n_nodes;
    }

    // enumerate CPUs
    पर्यंत (g_state.numa.total_cpus < GGML_NUMA_MAX_CPUS) {
        rv = snprintf(path, आकार(path), "/sys/devices/system/cpu/cpu%u", g_state.numa.total_cpus);
        GGML_ASSERT(rv > 0 && (अचिन्हित)rv < आकार(path));
        जर (stat(path, &st) != 0) { थांबा; }
        ++g_state.numa.total_cpus;
    }

    GGML_PRINT_DEBUG("found %u numa nodes, %u CPUs\n", g_state.numa.n_nodes, g_state.numa.total_cpus);

    // figure out which node we're on
    uint current_cpu;
    पूर्णांक getcpu_ret = 0;
#जर __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ > 33) || defined(__COSMOPOLITAN__)
    getcpu_ret = getcpu(&current_cpu, &g_state.numa.current_गाठ);
#अन्यथा
    // old glibc doesn't have a wrapper for this call. Fall back on direct syscall
#   जर !defined(SYS_getcpu) && defined(SYS_get_cpu)
#       define SYS_getcpu SYS_get_cpu // some older glibc versions use this name
#   endif
    getcpu_ret = syscall(SYS_getcpu, &current_cpu, &g_state.numa.current_गाठ);
#endif

    जर (g_state.numa.n_nodes < 1 || g_state.numa.total_cpus < 1 || getcpu_ret != 0) {
        g_state.numa.n_nodes = 0;
        परत;
    }

    GGML_PRINT_DEBUG("found our process on numa node %u, CPU %u\n", g_state.numa.current_गाठ, current_cpu);

    साठी (uint32_t n = 0; n < g_state.numa.n_nodes; ++n) {
        संरचना ggml_numa_गाठ * गाठ = &g_state.numa.nodes[n];
        GGML_PRINT_DEBUG("CPUs on node %u:", n);
        गाठ->n_cpus = 0;
        साठी (uint32_t c = 0; c < g_state.numa.total_cpus; ++c) {
            rv = snprintf(path, आकार(path), "/sys/devices/system/node/node%u/cpu%u", n, c);
            GGML_ASSERT(rv > 0 && (अचिन्हित)rv < आकार(path));
            जर (stat(path, &st) == 0) {
                गाठ->cpus[गाठ->n_cpus++] = c;
                GGML_PRINT_DEBUG(" %u", c);
            }
        }
        GGML_PRINT_DEBUG("\n");
    }

    जर (ggml_is_numa()) {
        खाता *fptr = ख_खोलो("/proc/sys/kernel/numa_balancing", "r");
        जर (fptr != रिक्त) {
            अक्षर buf[42];
            जर (ख_माला_लो(buf, आकार(buf), fptr) && म_भेदन(buf, "0\n", आकार(buf)) != 0) {
                GGML_LOG_WARN("/proc/sys/kernel/numa_balancing is enabled, this has been observed to impair performance\n");
            }
            ख_बंद(fptr);
        }
    }
#अन्यथा
    UNUSED(numa_flag);
    // TODO
#endif
}

bool ggml_is_numa(शून्य) {
    परत g_state.numa.n_nodes > 1;
}

#जर defined(__ARM_ARCH)
#जर defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
#अन्तर्भाव <arm_sve.h>
स्थिर शून्य ggml_init_arm_arch_features(शून्य) {
    ggml_arm_arch_features.sve_cnt = svcntb();
}
#अन्यथा
स्थिर शून्य ggml_init_arm_arch_features(शून्य) {}
#endif
#endif // __ARM_ARCH

#जर defined(__riscv) && defined(__riscv_v_intrinsic)
#अन्तर्भाव <riscv_vector.h>
स्थिर शून्य ggml_init_riscv_arch_features(शून्य) {
    ggml_riscv_arch_features.rvv_vlen = __riscv_vlenb();
}
#अन्यथा
स्थिर शून्य ggml_init_riscv_arch_features(शून्य) {}
#endif

संरचना ggml_tensor * ggml_new_i32(संरचना ggml_context * ctx, int32_t मूल्य) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    संरचना ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

    ggml_set_i32(result, मूल्य);

    परत result;
}

संरचना ggml_tensor * ggml_new_f32(संरचना ggml_context * ctx, दशांश मूल्य) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    संरचना ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    ggml_set_f32(result, मूल्य);

    परत result;
}

संरचना ggml_tensor * ggml_set_i32 (संरचना ggml_tensor * tensor, int32_t मूल्य) {
    स्थिरांक पूर्णांक n     = ggml_nrows(tensor);
    स्थिरांक पूर्णांक nc    = tensor->ne[0];
    स्थिरांक माप_प्रकार n1 = tensor->nb[1];

    अक्षर * स्थिरांक data = tensor->data;

    निवड (tensor->type) {
        स्थिती GGML_TYPE_I8:
            {
                प्रतिज्ञा(tensor->nb[0] == आकार(int8_t));
                साठी (पूर्णांक i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), मूल्य);
                }
            } थांबा;
        स्थिती GGML_TYPE_I16:
            {
                प्रतिज्ञा(tensor->nb[0] == आकार(int16_t));
                साठी (पूर्णांक i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), मूल्य);
                }
            } थांबा;
        स्थिती GGML_TYPE_I32:
            {
                प्रतिज्ञा(tensor->nb[0] == आकार(int32_t));
                साठी (पूर्णांक i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), मूल्य);
                }
            } थांबा;
        स्थिती GGML_TYPE_F16:
            {
                प्रतिज्ञा(tensor->nb[0] == आकार(ggml_fp16_t));
                साठी (पूर्णांक i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), GGML_CPU_FP32_TO_FP16(मूल्य));
                }
            } थांबा;
        स्थिती GGML_TYPE_BF16:
            {
                प्रतिज्ञा(tensor->nb[0] == आकार(ggml_fp16_t));
                साठी (पूर्णांक i = 0; i < n; i++) {
                    ggml_vec_set_bf16(nc, (ggml_bf16_t *)(data + i*n1), GGML_FP32_TO_BF16(मूल्य));
                }
            } थांबा;
        स्थिती GGML_TYPE_F32:
            {
                प्रतिज्ञा(tensor->nb[0] == आकार(दशांश));
                साठी (पूर्णांक i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (दशांश *)(data + i*n1), मूल्य);
                }
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }

    परत tensor;
}

संरचना ggml_tensor * ggml_set_f32(संरचना ggml_tensor * tensor, दशांश मूल्य) {
    स्थिरांक पूर्णांक n     = ggml_nrows(tensor);
    स्थिरांक पूर्णांक nc    = tensor->ne[0];
    स्थिरांक माप_प्रकार n1 = tensor->nb[1];

    अक्षर * स्थिरांक data = tensor->data;

    निवड (tensor->type) {
        स्थिती GGML_TYPE_I8:
            {
                प्रतिज्ञा(tensor->nb[0] == आकार(int8_t));
                साठी (पूर्णांक i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), मूल्य);
                }
            } थांबा;
        स्थिती GGML_TYPE_I16:
            {
                प्रतिज्ञा(tensor->nb[0] == आकार(int16_t));
                साठी (पूर्णांक i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), मूल्य);
                }
            } थांबा;
        स्थिती GGML_TYPE_I32:
            {
                प्रतिज्ञा(tensor->nb[0] == आकार(int32_t));
                साठी (पूर्णांक i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), मूल्य);
                }
            } थांबा;
        स्थिती GGML_TYPE_F16:
            {
                प्रतिज्ञा(tensor->nb[0] == आकार(ggml_fp16_t));
                साठी (पूर्णांक i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), GGML_CPU_FP32_TO_FP16(मूल्य));
                }
            } थांबा;
        स्थिती GGML_TYPE_BF16:
            {
                प्रतिज्ञा(tensor->nb[0] == आकार(ggml_bf16_t));
                साठी (पूर्णांक i = 0; i < n; i++) {
                    ggml_vec_set_bf16(nc, (ggml_bf16_t *)(data + i*n1), GGML_FP32_TO_BF16(मूल्य));
                }
            } थांबा;
        स्थिती GGML_TYPE_F32:
            {
                प्रतिज्ञा(tensor->nb[0] == आकार(दशांश));
                साठी (पूर्णांक i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (दशांश *)(data + i*n1), मूल्य);
                }
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }

    परत tensor;
}

int32_t ggml_get_i32_1d(स्थिरांक संरचना ggml_tensor * tensor, पूर्णांक i) {
    जर (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        परत ggml_get_i32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    निवड (tensor->type) {
        स्थिती GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == आकार(int8_t));
                परत ((int8_t *)(tensor->data))[i];
            }
        स्थिती GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == आकार(int16_t));
                परत ((int16_t *)(tensor->data))[i];
            }
        स्थिती GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == आकार(int32_t));
                परत ((int32_t *)(tensor->data))[i];
            }
        स्थिती GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == आकार(ggml_fp16_t));
                परत GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *)(tensor->data))[i]);
            }
        स्थिती GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == आकार(ggml_bf16_t));
                परत GGML_BF16_TO_FP32(((ggml_bf16_t *)(tensor->data))[i]);
            }
        स्थिती GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == आकार(दशांश));
                परत ((दशांश *)(tensor->data))[i];
            }
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

शून्य ggml_set_i32_1d(स्थिरांक संरचना ggml_tensor * tensor, पूर्णांक i, int32_t मूल्य) {
    जर (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_set_i32_nd(tensor, id[0], id[1], id[2], id[3], मूल्य);
        परत;
    }
    निवड (tensor->type) {
        स्थिती GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == आकार(int8_t));
                ((int8_t *)(tensor->data))[i] = मूल्य;
            } थांबा;
        स्थिती GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == आकार(int16_t));
                ((int16_t *)(tensor->data))[i] = मूल्य;
            } थांबा;
        स्थिती GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == आकार(int32_t));
                ((int32_t *)(tensor->data))[i] = मूल्य;
            } थांबा;
        स्थिती GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == आकार(ggml_fp16_t));
                ((ggml_fp16_t *)(tensor->data))[i] = GGML_CPU_FP32_TO_FP16(मूल्य);
            } थांबा;
        स्थिती GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == आकार(ggml_bf16_t));
                ((ggml_bf16_t *)(tensor->data))[i] = GGML_FP32_TO_BF16(मूल्य);
            } थांबा;
        स्थिती GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == आकार(दशांश));
                ((दशांश *)(tensor->data))[i] = मूल्य;
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

int32_t ggml_get_i32_nd(स्थिरांक संरचना ggml_tensor * tensor, पूर्णांक i0, पूर्णांक i1, पूर्णांक i2, पूर्णांक i3) {
    शून्य * data   = (अक्षर *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    निवड (tensor->type) {
        स्थिती GGML_TYPE_I8:
            परत ((int8_t *) data)[0];
        स्थिती GGML_TYPE_I16:
            परत ((int16_t *) data)[0];
        स्थिती GGML_TYPE_I32:
            परत ((int32_t *) data)[0];
        स्थिती GGML_TYPE_F16:
            परत GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *) data)[0]);
        स्थिती GGML_TYPE_BF16:
            परत GGML_BF16_TO_FP32(((ggml_bf16_t *) data)[0]);
        स्थिती GGML_TYPE_F32:
            परत ((दशांश *) data)[0];
        पूर्वनिर्धारित:
            GGML_ABORT("fatal error");
    }
}

शून्य ggml_set_i32_nd(स्थिरांक संरचना ggml_tensor * tensor, पूर्णांक i0, पूर्णांक i1, पूर्णांक i2, पूर्णांक i3, int32_t मूल्य) {
    शून्य * data   = (अक्षर *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    निवड (tensor->type) {
        स्थिती GGML_TYPE_I8:
            {
                ((int8_t *)(data))[0] = मूल्य;
            } थांबा;
        स्थिती GGML_TYPE_I16:
            {
                ((int16_t *)(data))[0] = मूल्य;
            } थांबा;
        स्थिती GGML_TYPE_I32:
            {
                ((int32_t *)(data))[0] = मूल्य;
            } थांबा;
        स्थिती GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(data))[0] = GGML_CPU_FP32_TO_FP16(मूल्य);
            } थांबा;
        स्थिती GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(data))[0] = GGML_FP32_TO_BF16(मूल्य);
            } थांबा;
        स्थिती GGML_TYPE_F32:
            {
                ((दशांश *)(data))[0] = मूल्य;
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

दशांश ggml_get_f32_1d(स्थिरांक संरचना ggml_tensor * tensor, पूर्णांक i) {
    जर (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        परत ggml_get_f32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    निवड (tensor->type) {
        स्थिती GGML_TYPE_I8:
            {
                परत ((int8_t *)(tensor->data))[i];
            }
        स्थिती GGML_TYPE_I16:
            {
                परत ((int16_t *)(tensor->data))[i];
            }
        स्थिती GGML_TYPE_I32:
            {
                परत ((int32_t *)(tensor->data))[i];
            }
        स्थिती GGML_TYPE_F16:
            {
                परत GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *)(tensor->data))[i]);
            }
        स्थिती GGML_TYPE_BF16:
            {
                परत GGML_BF16_TO_FP32(((ggml_bf16_t *)(tensor->data))[i]);
            }
        स्थिती GGML_TYPE_F32:
            {
                परत ((दशांश *)(tensor->data))[i];
            }
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

शून्य ggml_set_f32_1d(स्थिरांक संरचना ggml_tensor * tensor, पूर्णांक i, दशांश मूल्य) {
    जर (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_set_f32_nd(tensor, id[0], id[1], id[2], id[3], मूल्य);
        परत;
    }
    निवड (tensor->type) {
        स्थिती GGML_TYPE_I8:
            {
                ((int8_t *)(tensor->data))[i] = मूल्य;
            } थांबा;
        स्थिती GGML_TYPE_I16:
            {
                ((int16_t *)(tensor->data))[i] = मूल्य;
            } थांबा;
        स्थिती GGML_TYPE_I32:
            {
                ((int32_t *)(tensor->data))[i] = मूल्य;
            } थांबा;
        स्थिती GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(tensor->data))[i] = GGML_CPU_FP32_TO_FP16(मूल्य);
            } थांबा;
        स्थिती GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(tensor->data))[i] = GGML_FP32_TO_BF16(मूल्य);
            } थांबा;
        स्थिती GGML_TYPE_F32:
            {
                ((दशांश *)(tensor->data))[i] = मूल्य;
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

दशांश ggml_get_f32_nd(स्थिरांक संरचना ggml_tensor * tensor, पूर्णांक i0, पूर्णांक i1, पूर्णांक i2, पूर्णांक i3) {
    शून्य * data   = (अक्षर *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    निवड (tensor->type) {
        स्थिती GGML_TYPE_I8:
            परत ((int8_t *) data)[0];
        स्थिती GGML_TYPE_I16:
            परत ((int16_t *) data)[0];
        स्थिती GGML_TYPE_I32:
            परत ((int32_t *) data)[0];
        स्थिती GGML_TYPE_F16:
            परत GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *) data)[0]);
        स्थिती GGML_TYPE_BF16:
            परत GGML_BF16_TO_FP32(((ggml_bf16_t *) data)[0]);
        स्थिती GGML_TYPE_F32:
            परत ((दशांश *) data)[0];
        पूर्वनिर्धारित:
            GGML_ABORT("fatal error");
    }
}

शून्य ggml_set_f32_nd(स्थिरांक संरचना ggml_tensor * tensor, पूर्णांक i0, पूर्णांक i1, पूर्णांक i2, पूर्णांक i3, दशांश मूल्य) {
    शून्य * data   = (अक्षर *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    निवड (tensor->type) {
        स्थिती GGML_TYPE_I8:
            {
                ((int8_t *)(data))[0] = मूल्य;
            } थांबा;
        स्थिती GGML_TYPE_I16:
            {
                ((int16_t *)(data))[0] = मूल्य;
            } थांबा;
        स्थिती GGML_TYPE_I32:
            {
                ((int32_t *)(data))[0] = मूल्य;
            } थांबा;
        स्थिती GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(data))[0] = GGML_CPU_FP32_TO_FP16(मूल्य);
            } थांबा;
        स्थिती GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(data))[0] = GGML_FP32_TO_BF16(मूल्य);
            } थांबा;
        स्थिती GGML_TYPE_F32:
            {
                ((दशांश *)(data))[0] = मूल्य;
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

////////////////////////////////////////////////////////////////////////////////

// ggml_compute_forward_mul_mat

स्थिर शून्य ggml_compute_forward_mul_mat_one_chunk(
    स्थिरांक संरचना ggml_compute_params * params,
    संरचना ggml_tensor * dst,
    स्थिरांक गणना ggml_type type,
    स्थिरांक int64_t अंक_rows_per_vec_dot,
    स्थिरांक int64_t ir0_start,
    स्थिरांक int64_t ir0_end,
    स्थिरांक int64_t ir1_start,
    स्थिरांक int64_t ir1_end) {

    स्थिरांक संरचना ggml_tensor * src0 = dst->src[0];
    स्थिरांक संरचना ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विमान_OP_LOCALS

    स्थिरांक bool src1_cont = ggml_is_contiguous(src1);

    ggml_vec_dot_t स्थिरांक vec_dot      = type_traits_cpu[type].vec_dot;
    गणना ggml_type स्थिरांक vec_dot_type = type_traits_cpu[type].vec_dot_type;

    // broadcast factors
    स्थिरांक int64_t r2 = ne12 / ne02;
    स्थिरांक int64_t r3 = ne13 / ne03;

    //printf("ir0_start = %6lld, ir0_end = %6lld, ir1_start = %6lld, ir1_end = %6lld\n", ir0_start, ir0_end, ir1_start, ir1_end);

    // threads with no work simply yield (not sure if it helps)
    जर (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        परत;
    }

    स्थिरांक शून्य * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
    स्थिरांक माप_प्रकार row_size = ggml_row_size(vec_dot_type, ne10);

    प्रतिज्ञा(ne12 % ne02 == 0);
    प्रतिज्ञा(ne13 % ne03 == 0);

    // block-tiling attempt
    स्थिरांक int64_t blck_0 = 16;
    स्थिरांक int64_t blck_1 = 16;

    स्थिरांक माप_प्रकार src1_col_stride = src1_cont || src1->type != vec_dot_type ? row_size : nb11;

    // attempt to reduce false-sharing (does not seem to make a difference)
    // 16 * 2, accounting for mmla kernels
    दशांश tmp[32];

    साठी (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        साठी (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            साठी (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += अंक_rows_per_vec_dot) {
                स्थिरांक int64_t i13 = (ir1 / (ne12 * ne1));
                स्थिरांक int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                स्थिरांक int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                // broadcast src0 into src1
                स्थिरांक int64_t i03 = i13 / r3;
                स्थिरांक int64_t i02 = i12 / r2;

                स्थिरांक int64_t i1 = i11;
                स्थिरांक int64_t i2 = i12;
                स्थिरांक int64_t i3 = i13;

                स्थिरांक अक्षर * src0_row = (स्थिरांक अक्षर*)src0->data + (0 + i02 * nb02 + i03 * nb03);

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                स्थिरांक अक्षर * src1_col = (स्थिरांक अक्षर*)wdata +
                    (src1_cont || src1->type != vec_dot_type
                        ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                        : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                दशांश * dst_col = (दशांश*)((अक्षर*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

                //for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                //    vec_dot(ne00, &dst_col[ir0], src0_row + ir0*nb01, src1_col);
                //}

                साठी (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += अंक_rows_per_vec_dot) {
                    vec_dot(ne00, &tmp[ir0 - iir0], (अंक_rows_per_vec_dot > 1 ? 16 : 0), src0_row + ir0 * nb01, (अंक_rows_per_vec_dot > 1 ? nb01 : 0), src1_col, (अंक_rows_per_vec_dot > 1 ? src1_col_stride : 0), अंक_rows_per_vec_dot);
                }

                साठी (पूर्णांक cn = 0; cn < अंक_rows_per_vec_dot; ++cn) {
                    स_नकल(&dst_col[iir0 + cn * nb1 / nb0], tmp + (cn * 16), (MIN(iir0 + blck_0, ir0_end) - iir0) * आकार(दशांश));
                }
            }
        }
    }
}

शून्य ggml_compute_forward_mul_mat(
        स्थिरांक संरचना ggml_compute_params * params,
              संरचना ggml_tensor * dst) {

    स्थिरांक संरचना ggml_tensor * src0 = dst->src[0];
    स्थिरांक संरचना ggml_tensor * src1 = dst->src[1];

    स्थिरांक int32_t hint = ggml_get_op_params_i32(dst, 1);
    जर (hint == GGML_HINT_SRC0_IS_HADAMARD && !params->use_ref) {
        ggml_compute_forward_fwht(params, dst);
        परत;
    }

    // If tiled is supported, it will execute the full op here and we return
    जर (ggml_compute_forward_mul_mat_tiled(params, dst)) {
        परत;
    }

    GGML_TENSOR_द्विमान_OP_LOCALS

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    गणना ggml_type           स्थिरांक vec_dot_type         = type_traits_cpu[src0->type].vec_dot_type;
    ggml_from_float_t        स्थिरांक from_float           = type_traits_cpu[vec_dot_type].from_float;
    int64_t                  स्थिरांक vec_dot_अंक_rows     = type_traits_cpu[src0->type].nrows;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == आकार(दशांश));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    // TODO: extract to "extra_op"
#जर GGML_USE_LLAMAFILE
    // broadcast factors
    स्थिरांक int64_t r2 = ne12 / ne02;
    स्थिरांक int64_t r3 = ne13 / ne03;

    स्थिरांक bool src1_cont = ggml_is_contiguous(src1);

    जर (!params->use_ref && src1_cont) {
        साठी (int64_t i13 = 0; i13 < ne13; i13++)
            साठी (int64_t i12 = 0; i12 < ne12; i12++)
                जर (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_size(src0->type),
                                     (स्थिरांक अक्षर *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_size(src0->type),
                                     (स्थिरांक अक्षर *)src1->data + i12*nb12 + i13*nb13,
                                     nb11/ggml_type_size(src1->type),
                                     (अक्षर *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_size(dst->type),
                                     src0->type,
                                     src1->type,
                                     dst->type))
                    जा UseGgmlGemm1;
        परत;
    }
UseGgmlGemm1:;
#endif

    जर (src1->type != vec_dot_type) {
        अक्षर * wdata = params->wdata;

        स्थिरांक माप_प्रकार nbw0 = ggml_type_size(vec_dot_type);
        स्थिरांक माप_प्रकार nbw1 = ggml_row_size(vec_dot_type, ne10);
        स्थिरांक माप_प्रकार nbw2 = nbw1*ne11;
        स्थिरांक माप_प्रकार nbw3 = nbw2*ne12;

        प्रतिज्ञा(params->wsize >= ne13*nbw3);
        // src1 is either packed from F32 into vec_dot_type, or widened from F16 or BF16 into the F32 work buffer
        स्थिरांक bool widen = src1->type != GGML_TYPE_F32;

        GGML_ASSERT(!widen || vec_dot_type == GGML_TYPE_F32);
        GGML_ASSERT(!widen || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_BF16);

    #जर 0
        साठी (int64_t i13 = 0; i13 < ne13; ++i13) {
            साठी (int64_t i12 = 0; i12 < ne12; ++i12) {
                साठी (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                    from_float((दशांश *)((अक्षर *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (शून्य *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                                ne10);
                }
            }
        }
    #अन्यथा
        स्थिरांक int64_t bs = ggml_blck_size(vec_dot_type);

        साठी (int64_t i13 = 0; i13 < ne13; ++i13) {
            साठी (int64_t i12 = 0; i12 < ne12; ++i12) {
                साठी (int64_t i11 = 0; i11 < ne11; ++i11) {
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    स्थिरांक शून्य * src1_row  = (स्थिरांक अक्षर *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10;
                    शून्य       * wdata_row =                    wdata  + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0;

                    स्थिरांक int64_t ne10_block_size = (ne10_block_end - ne10_block_start) * bs;

                    जर (src1->type == GGML_TYPE_F16) {
                        ggml_cpu_fp16_to_fp32((स्थिरांक ggml_fp16_t *) src1_row, (दशांश *) wdata_row, ne10_block_size);
                    } अन्यथा जर (src1->type == GGML_TYPE_BF16) {
                        ggml_cpu_bf16_to_fp32((स्थिरांक ggml_bf16_t *) src1_row, (दशांश *) wdata_row, ne10_block_size);
                    } अन्यथा {
                        from_float((स्थिरांक दशांश *) src1_row, wdata_row, ne10_block_size);
                    }
                }
            }
        }
    #endif
    }

    जर (ith == 0) {
        // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
        atomic_store_explicit(&params->threadpool->current_chunk, nth, memory_order_relaxed);
    }

    ggml_barrier(params->threadpool);

#जर GGML_USE_LLAMAFILE
    जर (!params->use_ref && src1->type != vec_dot_type) {
        स्थिरांक शून्य* wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        स्थिरांक माप_प्रकार row_size = ggml_row_size(vec_dot_type, ne10);

        साठी (int64_t i13 = 0; i13 < ne13; i13++)
            साठी (int64_t i12 = 0; i12 < ne12; i12++)
                जर (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_size(src0->type),
                                     (स्थिरांक अक्षर *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_size(src0->type),
                                     (स्थिरांक अक्षर *)wdata + (i12*ne11 + i13*ne12*ne11)*row_size,
                                     row_size/ggml_type_size(vec_dot_type),
                                     (अक्षर *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_size(dst->type),
                                     src0->type,
                                     vec_dot_type,
                                     dst->type))
                    जा UseGgmlGemm2;
        परत;
    }
UseGgmlGemm2:;
#endif

    // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
    स्थिरांक int64_t nr0 = ne0;

    // This is the size of the rest of the dimensions of the result
    स्थिरांक int64_t nr1 = ne1 * ne2 * ne3;

    // Now select a reasonable chunk size.
    पूर्णांक chunk_size = 16;

    // We need to step up the size if it's small
    जर (nr0 == 1 || nr1 == 1) {
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
    जर (nchunk0 * nchunk1 < nth * 4 || ggml_is_numa()) {
        // distribute the thread work across the inner or outer loop based on which one is larger
        nchunk0 = nr0 > nr1 ? nth : 1; // parallelize by src0 rows
        nchunk1 = nr0 > nr1 ? 1 : nth; // parallelize by src1 rows
    }

    // The number of elements in each chunk
    स्थिरांक int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    स्थिरांक int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    // The first chunk comes from our thread_id, the rest will get auto-assigned.
    पूर्णांक current_chunk = ith;

    पर्यंत (current_chunk < nchunk0 * nchunk1) {
        स्थिरांक int64_t ith0 = current_chunk % nchunk0;
        स्थिरांक int64_t ith1 = current_chunk / nchunk0;

        स्थिरांक int64_t ir0_start = dr0 * ith0;
        स्थिरांक int64_t ir0_end = MIN(ir0_start + dr0, nr0);

        स्थिरांक int64_t ir1_start = dr1 * ith1;
        स्थिरांक int64_t ir1_end = MIN(ir1_start + dr1, nr1);

        // dot kernels can handle 1 row and col at a time, but mmla kernels can process 2 rows and cols
        int64_t अंक_rows_per_vec_dot = vec_dot_अंक_rows;

        // these checks are needed to avoid crossing dim1 boundaries
        // can be optimized, but the logic would become more complicated, so keeping it like this for simplicity
        जर ((nr0 % 2 != 0) || (ne11 % 2 != 0) || ((ir0_end - ir0_start) % 2 != 0) || ((ir1_end - ir1_start) % 2 != 0)) {
            अंक_rows_per_vec_dot = 1;
        }
        ggml_compute_forward_mul_mat_one_chunk(params, dst, src0->type, अंक_rows_per_vec_dot, ir0_start, ir0_end, ir1_start, ir1_end);

        जर (nth >= nchunk0 * nchunk1) {
            थांबा;
        }

        current_chunk = atomic_fetch_add_explicit(&params->threadpool->current_chunk, 1, memory_order_relaxed);
    }
}

// ggml_compute_forward_mul_mat_id

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id)*ids->ne[0]*ids->ne[1] + (i1)]

संरचना mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

स्थिर शून्य ggml_compute_forward_mul_mat_id_one_chunk(
    संरचना ggml_tensor * dst,
    स्थिरांक संरचना ggml_tensor * src0,
    स्थिरांक संरचना ggml_tensor * src1,
    स्थिरांक संरचना ggml_tensor * ids,
    स्थिरांक int64_t cur_a,
    स्थिरांक int64_t ir0_start,
    स्थिरांक int64_t ir0_end,
    स्थिरांक int64_t ir1_start,
    स्थिरांक int64_t ir1_end,
    स्थिरांक अक्षर * src0_cur,
    स्थिरांक संरचना mmid_row_mapping * matrix_rows,
    स्थिरांक माप_प्रकार row_size,
    स्थिरांक bool src1_cont,
    स्थिरांक शून्य * wdata) {

    GGML_TENSOR_द्विमान_OP_LOCALS

    स्थिरांक गणना ggml_type type = src0->type;

    ggml_vec_dot_t    स्थिरांक vec_dot      = type_traits_cpu[type].vec_dot;
    गणना ggml_type    स्थिरांक vec_dot_type = type_traits_cpu[type].vec_dot_type;

    स्थिरांक int64_t blck_0 = 16;
    स्थिरांक int64_t blck_1 = 16;

    दशांश tmp[16];

    साठी (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        साठी (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            साठी (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ++ir1) {
                स्थिरांक int64_t _i12 = ir1; // logical row index for this expert

                संरचना mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, _i12);
                स्थिरांक पूर्णांक id       = row_mapping.i1; // selected expert index

                स्थिरांक int64_t  i11 = id % ne11;
                स्थिरांक int64_t  i12 = row_mapping.i2; // row index in src1

                स्थिरांक int64_t  i1 = id;  // selected expert index
                स्थिरांक int64_t  i2 = i12; // row

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                स्थिरांक अक्षर * src1_col = (स्थिरांक अक्षर *) wdata +
                    (src1_cont || src1->type != vec_dot_type
                    ? (i11      + i12*ne11)*row_size
                    : (i11*nb11 + i12*nb12));

                दशांश * dst_col = (दशांश *) ((अक्षर *) dst->data + (i1*nb1 + i2*nb2));

                साठी (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                    vec_dot(ne00, &tmp[ir0 - iir0], 0, src0_cur + ir0*nb01, 0, src1_col, 0, 1);
                }

                स_नकल(&dst_col[iir0], tmp, (MIN(iir0 + blck_0, ir0_end) - iir0)*आकार(दशांश));
            }
        }
    }
}

स्थिर शून्य * incr_ptr_aligned(शून्य ** p, माप_प्रकार size, माप_प्रकार align) {

    शून्य * ptr = *p;
    ptr = (शून्य *) GGML_PAD((uintptr_t) ptr, align);
    *p = (शून्य *) ((अक्षर *) ptr + size);
    परत ptr;
}

स्थिर शून्य ggml_compute_forward_mul_mat_id(
        स्थिरांक संरचना ggml_compute_params * params,
              संरचना ggml_tensor * dst) {

    स्थिरांक संरचना ggml_tensor * src0 = dst->src[0];
    स्थिरांक संरचना ggml_tensor * src1 = dst->src[1];
    स्थिरांक संरचना ggml_tensor * ids = dst->src[2];

    GGML_TENSOR_द्विमान_OP_LOCALS

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक गणना ggml_type type = src0->type;

    स्थिरांक bool src1_cont = ggml_is_contiguous(src1);

    गणना ggml_type    स्थिरांक vec_dot_type    = type_traits_cpu[type].vec_dot_type;
    ggml_from_float_t स्थिरांक from_float      = type_traits_cpu[vec_dot_type].from_float;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == आकार(दशांश));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // row groups
    स्थिरांक पूर्णांक n_ids = ids->ne[0]; // n_expert_used
    स्थिरांक पूर्णांक n_as  = ne02;       // n_expert

    शून्य * wdata_cur = params->wdata;

    जर (src1->type != vec_dot_type) {
        incr_ptr_aligned(&wdata_cur, ggml_row_size(vec_dot_type, ggml_nelements(src1)), आकार(int64_t));
    }

    int64_t * matrix_row_counts = // [n_as]
        incr_ptr_aligned(&wdata_cur, n_as*आकार(int64_t), आकार(int64_t));

    संरचना mmid_row_mapping * matrix_rows = // [n_as][ids->ne[0]*ids->ne[1]]
        incr_ptr_aligned(&wdata_cur, n_as*ids->ne[0]*ids->ne[1]*आकार(संरचना mmid_row_mapping), आकार(int64_t));

    अक्षर (*atomic_current_chunk)[CACHE_ओळ_SIZE] = // [n_as]
        incr_ptr_aligned(&wdata_cur, CACHE_ओळ_SIZE * n_as, CACHE_ओळ_SIZE);

    // Tiled matmul (see tiled.h); per-thread work buffers, 0 bytes when disabled. The
    // reservation is unconditional, the per expert eligibility is decided at dispatch time
    अक्षर * tiled_scratch = incr_ptr_aligned(&wdata_cur, ggml_tiled_wdata_size(nth, dst), 64);

    GGML_ASSERT(params->wsize >= (माप_प्रकार)((अक्षर *) wdata_cur - (अक्षर *) params->wdata));

    जर (src1->type != vec_dot_type) {
        अक्षर * wdata = params->wdata;

        स्थिरांक माप_प्रकार nbw0 = ggml_type_size(vec_dot_type);
        स्थिरांक माप_प्रकार nbw1 = ggml_row_size(vec_dot_type, ne10);
        स्थिरांक माप_प्रकार nbw2 = nbw1*ne11;
        स्थिरांक माप_प्रकार nbw3 = nbw2*ne12;

        प्रतिज्ञा(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

#जर 0
        साठी (int64_t i13 = 0; i13 < ne13; ++i13) {
            साठी (int64_t i12 = ith; i12 < ne12; i12 += nth) {
                साठी (int64_t i11 = 0; i11 < ne11; ++i11) {
                    from_float((दशांश *)((अक्षर *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (शून्य *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                               ne10);
                }
            }
        }
#अन्यथा
        साठी (int64_t i13 = 0; i13 < ne13; ++i13) {
            साठी (int64_t i12 = 0; i12 < ne12; ++i12) {
                साठी (int64_t i11 = 0; i11 < ne11; ++i11) {
                    माप_प्रकार bs = ggml_blck_size(vec_dot_type);
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    from_float((दशांश *)((अक्षर *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10),
                               (शून्य *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0),
                               (ne10_block_end - ne10_block_start) * bs);
                }
            }
        }
#endif
    }

    जर (ith == 0) {
        // initialize matrix_row_counts
        स_रखो(matrix_row_counts, 0, n_as*आकार(int64_t));

        // group rows by src0 matrix
        साठी (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            साठी (पूर्णांक id = 0; id < n_ids; ++id) {
                स्थिरांक int32_t i02 = *(स्थिरांक int32_t *) ((स्थिरांक अक्षर *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);

                प्रतिज्ञा(i02 >= 0 && i02 < n_as);

                MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = (संरचना mmid_row_mapping) {id, iid1};
                matrix_row_counts[i02] += 1;
            }
        }
    }

    // reset current_chunk
    साठी (पूर्णांक cur_a = ith; cur_a < n_as; cur_a += nth) {
        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);
        *current_chunk_ctr = nth;
    }

    ggml_barrier(params->threadpool);

    साठी (पूर्णांक cur_a = 0; cur_a < n_as; ++cur_a) {
        स्थिरांक int64_t cne1 = matrix_row_counts[cur_a];

        जर (cne1 == 0) {
            सुरूठेवा;
        }

        // tiled takes over if profitable for this expert (see tiled.h)
        जर (ggml_compute_forward_mul_mat_id_tiled(params, dst, cur_a, cne1, (स्थिरांक int32_t *) &MMID_MATRIX_ROW(cur_a, 0), tiled_scratch)) {
            सुरूठेवा;
        }

        स्थिरांक अक्षर * src0_cur = (स्थिरांक अक्षर *) src0->data + cur_a * nb02;
        स्थिरांक शून्य * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        स्थिरांक माप_प्रकार row_size = ggml_row_size(vec_dot_type, ne10);

        स्थिरांक int64_t nr0 = ne01;
        स्थिरांक int64_t nr1 = cne1;

        पूर्णांक chunk_size = 16;
        जर (nr0 == 1 || nr1 == 1) {
            chunk_size = 64;
        }

        // disable for NUMA
        स्थिरांक bool disable_chunking = ggml_is_numa();

        int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
        int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

        जर (nchunk0 * nchunk1 < nth * 4 || disable_chunking) {
            nchunk0 = nr0 > nr1 ? nth : 1;
            nchunk1 = nr0 > nr1 ? 1 : nth;
        }

        स्थिरांक int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        स्थिरांक int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

        पूर्णांक current_chunk = ith;

        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);

        पर्यंत (current_chunk < nchunk0 * nchunk1) {
            स्थिरांक int64_t ith0 = current_chunk % nchunk0;
            स्थिरांक int64_t ith1 = current_chunk / nchunk0;

            स्थिरांक int64_t ir0_start = dr0 * ith0;
            स्थिरांक int64_t ir0_end = MIN(ir0_start + dr0, nr0);

            स्थिरांक int64_t ir1_start = dr1 * ith1;
            स्थिरांक int64_t ir1_end = MIN(ir1_start + dr1, nr1);

            ggml_compute_forward_mul_mat_id_one_chunk(
                dst, src0, src1, ids, cur_a,
                ir0_start, ir0_end, ir1_start, ir1_end,
                src0_cur, matrix_rows, row_size, src1_cont, wdata
            );

            जर (nth >= nchunk0 * nchunk1) {
                थांबा;
            }

            current_chunk = atomic_fetch_add_explicit(current_chunk_ctr, 1, memory_order_relaxed);
        }
    }
}

/////////////////////////////////

स्थिर शून्य ggml_compute_forward(संरचना ggml_compute_params * params, संरचना ggml_tensor * tensor) {
    GGML_ASSERT(params);

    जर (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {
        परत;
    }

    // extra_buffer op?
    जर (ggml_cpu_extra_compute_forward(params, tensor)) {
        परत;
    }

    निवड (tensor->op) {
        स्थिती GGML_OP_DUP:
            {
                ggml_compute_forward_dup(params, tensor);
            } थांबा;
        स्थिती GGML_OP_ADD:
            {
                ggml_compute_forward_add(params, tensor);
            } थांबा;
        स्थिती GGML_OP_ADD_ID:
            {
                ggml_compute_forward_add_id(params, tensor);
            } थांबा;
        स्थिती GGML_OP_ADD1:
            {
                ggml_compute_forward_add1(params, tensor);
            } थांबा;
        स्थिती GGML_OP_ACC:
            {
                ggml_compute_forward_acc(params, tensor);
            } थांबा;
        स्थिती GGML_OP_SUB:
            {
                ggml_compute_forward_sub(params, tensor);
            } थांबा;
        स्थिती GGML_OP_MUL:
            {
                ggml_compute_forward_mul(params, tensor);
            } थांबा;
        स्थिती GGML_OP_DIV:
            {
                ggml_compute_forward_div(params, tensor);
            } थांबा;
        स्थिती GGML_OP_SQR:
            {
                ggml_compute_forward_sqr(params, tensor);
            } थांबा;
        स्थिती GGML_OP_SQRT:
            {
                ggml_compute_forward_sqrt(params, tensor);
            } थांबा;
        स्थिती GGML_OP_LOG:
            {
                ggml_compute_forward_log(params, tensor);
            } थांबा;
        स्थिती GGML_OP_SIN:
            {
                ggml_compute_forward_sin(params, tensor);
            } थांबा;
        स्थिती GGML_OP_COS:
            {
                ggml_compute_forward_cos(params, tensor);
            } थांबा;
        स्थिती GGML_OP_SUM:
            {
                ggml_compute_forward_sum(params, tensor);
            } थांबा;
        स्थिती GGML_OP_SUM_ROWS:
            {
                ggml_compute_forward_sum_rows(params, tensor);
            } थांबा;
        स्थिती GGML_OP_CUMSUM:
            {
                ggml_compute_forward_cumsum(params, tensor);
            } थांबा;
        स्थिती GGML_OP_MEAN:
            {
                ggml_compute_forward_mean(params, tensor);
            } थांबा;
        स्थिती GGML_OP_ARGMAX:
            {
                ggml_compute_forward_argmax(params, tensor);
            } थांबा;
        स्थिती GGML_OP_मोजणी_EQUAL:
            {
                ggml_compute_forward_मोजणी_equal(params, tensor);
            } थांबा;
        स्थिती GGML_OP_REPEAT:
            {
                ggml_compute_forward_repeat(params, tensor);
            } थांबा;
        स्थिती GGML_OP_REPEAT_BACK:
            {
                ggml_compute_forward_repeat_back(params, tensor);
            } थांबा;
        स्थिती GGML_OP_CONCAT:
            {
                ggml_compute_forward_concat(params, tensor);
            } थांबा;
        स्थिती GGML_OP_SILU_BACK:
            {
                ggml_compute_forward_silu_back(params, tensor);
            } थांबा;
        स्थिती GGML_OP_NORM:
            {
                ggml_compute_forward_norm(params, tensor);
            } थांबा;
        स्थिती GGML_OP_RMS_NORM:
            {
                ggml_compute_forward_rms_norm(params, tensor);
            } थांबा;
        स्थिती GGML_OP_RMS_NORM_BACK:
            {
                ggml_compute_forward_rms_norm_back(params, tensor);
            } थांबा;
        स्थिती GGML_OP_GROUP_NORM:
            {
                ggml_compute_forward_group_norm(params, tensor);
            } थांबा;
        स्थिती GGML_OP_L2_NORM:
            {
                ggml_compute_forward_l2_norm(params, tensor);
            } थांबा;
        स्थिती GGML_OP_MUL_MAT:
            {
                ggml_compute_forward_mul_mat(params, tensor);
            } थांबा;
        स्थिती GGML_OP_MUL_MAT_ID:
            {
                ggml_compute_forward_mul_mat_id(params, tensor);
            } थांबा;
        स्थिती GGML_OP_OUT_PROD:
            {
                ggml_compute_forward_out_prod(params, tensor);
            } थांबा;
        स्थिती GGML_OP_SCALE:
            {
                ggml_compute_forward_scale(params, tensor);
            } थांबा;
        स्थिती GGML_OP_SET:
            {
                ggml_compute_forward_set(params, tensor);
            } थांबा;
        स्थिती GGML_OP_CPY:
            {
                ggml_compute_forward_cpy(params, tensor);
            } थांबा;
        स्थिती GGML_OP_CONT:
            {
                ggml_compute_forward_cont(params, tensor);
            } थांबा;
        स्थिती GGML_OP_GET_ROWS:
            {
                ggml_compute_forward_get_rows(params, tensor);
            } थांबा;
        स्थिती GGML_OP_GET_ROWS_BACK:
            {
                ggml_compute_forward_get_rows_back(params, tensor);
            } थांबा;
        स्थिती GGML_OP_SET_ROWS:
            {
                ggml_compute_forward_set_rows(params, tensor);
            } थांबा;
        स्थिती GGML_OP_DIAG:
            {
                ggml_compute_forward_diag(params, tensor);
            } थांबा;
        स्थिती GGML_OP_DIAG_MASK_INF:
            {
                ggml_compute_forward_diag_mask_inf(params, tensor);
            } थांबा;
        स्थिती GGML_OP_DIAG_MASK_ZERO:
            {
                ggml_compute_forward_diag_mask_zero(params, tensor);
            } थांबा;
        स्थिती GGML_OP_SOFT_MAX:
            {
                ggml_compute_forward_soft_max(params, tensor);
            } थांबा;
        स्थिती GGML_OP_SOFT_MAX_BACK:
            {
                ggml_compute_forward_soft_max_ext_back(params, tensor);
            } थांबा;
        स्थिती GGML_OP_ROPE:
            {
                ggml_compute_forward_rope(params, tensor);
            } थांबा;
        स्थिती GGML_OP_ROPE_BACK:
            {
                ggml_compute_forward_rope_back(params, tensor);
            } थांबा;
        स्थिती GGML_OP_CLAMP:
            {
                ggml_compute_forward_clamp(params, tensor);
            } थांबा;
        स्थिती GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_compute_forward_conv_transpose_1d(params, tensor);
            } थांबा;
        स्थिती GGML_OP_IM2COL:
            {
                ggml_compute_forward_im2col(params, tensor);
            } थांबा;
        स्थिती GGML_OP_IM2COL_BACK:
            {
                ggml_compute_forward_im2col_back_f32(params, tensor);
            } थांबा;
        स्थिती GGML_OP_IM2COL_3D:
            {
                ggml_compute_forward_im2col_3d(params, tensor);
            } थांबा;
        स्थिती GGML_OP_COL2IM_1D:
            {
                ggml_compute_forward_col2im_1d(params, tensor);
            } थांबा;
        स्थिती GGML_OP_CONV_2D:
            {
                ggml_compute_forward_conv_2d(params, tensor);
            } थांबा;
        स्थिती GGML_OP_CONV_3D:
            {
                ggml_compute_forward_conv_3d(params, tensor);
            } थांबा;
        स्थिती GGML_OP_CONV_2D_DW:
            {
                ggml_compute_forward_conv_2d_dw(params, tensor);
            } थांबा;
        स्थिती GGML_OP_CONV_TRANSPOSE_2D:
            {
                ggml_compute_forward_conv_transpose_2d(params, tensor);
            } थांबा;
        स्थिती GGML_OP_POOL_1D:
            {
                ggml_compute_forward_pool_1d(params, tensor);
            } थांबा;
        स्थिती GGML_OP_POOL_2D:
            {
                ggml_compute_forward_pool_2d(params, tensor);
            } थांबा;
        स्थिती GGML_OP_POOL_2D_BACK:
            {
                ggml_compute_forward_pool_2d_back(params, tensor);
            } थांबा;
        स्थिती GGML_OP_UPSCALE:
            {
                ggml_compute_forward_upscale(params, tensor);
            } थांबा;
        स्थिती GGML_OP_PAD:
            {
                ggml_compute_forward_pad(params, tensor);
            } थांबा;
        स्थिती GGML_OP_PAD_REFLECT_1D:
            {
                ggml_compute_forward_pad_reflect_1d(params, tensor);
            } थांबा;
        स्थिती GGML_OP_ROLL:
            {
                ggml_compute_forward_roll(params, tensor);
            } थांबा;
        स्थिती GGML_OP_ARANGE:
            {
                ggml_compute_forward_arange(params, tensor);
            } थांबा;
        स्थिती GGML_OP_TIMESTEP_EMBEDDING:
            {
                ggml_compute_forward_timestep_embedding(params, tensor);
            } थांबा;
        स्थिती GGML_OP_ARGSORT:
            {
                ggml_compute_forward_argsort(params, tensor);
            } थांबा;
        स्थिती GGML_OP_शीर्ष_K:
            {
                ggml_compute_forward_शीर्ष_k(params, tensor);
            } थांबा;
        स्थिती GGML_OP_LEAKY_RELU:
            {
                ggml_compute_forward_leaky_relu(params, tensor);
            } थांबा;
        स्थिती GGML_OP_TRI:
            {
                ggml_compute_forward_tri(params, tensor);
            } थांबा;
        स्थिती GGML_OP_FILL:
            {
                ggml_compute_forward_fill(params, tensor);
            } थांबा;
        स्थिती GGML_OP_FLASH_ATTN_EXT:
            {
                ggml_compute_forward_flash_attn_ext(params, tensor);
            } थांबा;
        स्थिती GGML_OP_FLASH_ATTN_BACK:
            {
                int32_t t = ggml_get_op_params_i32(tensor, 0);
                GGML_ASSERT(t == 0 || t == 1);
                bool masked = t != 0;
                ggml_compute_forward_flash_attn_back(params, masked, tensor);
            } थांबा;
        स्थिती GGML_OP_SSM_CONV:
            {
                ggml_compute_forward_ssm_conv(params, tensor);
            } थांबा;
        स्थिती GGML_OP_SSM_SCAN:
            {
                ggml_compute_forward_ssm_scan(params, tensor);
            } थांबा;
        स्थिती GGML_OP_WIN_PART:
            {
                ggml_compute_forward_win_part(params, tensor);
            } थांबा;
        स्थिती GGML_OP_WIN_UNPART:
            {
                ggml_compute_forward_win_unpart(params, tensor);
            } थांबा;
        स्थिती GGML_OP_UNARY:
            {
                ggml_compute_forward_unary(params, tensor);
            } थांबा;
        स्थिती GGML_OP_GLU:
            {
                ggml_compute_forward_glu(params, tensor);
            } थांबा;
        स्थिती GGML_OP_GET_REL_POS:
            {
                ggml_compute_forward_get_rel_pos(params, tensor);
            } थांबा;
        स्थिती GGML_OP_ADD_REL_POS:
            {
                ggml_compute_forward_add_rel_pos(params, tensor);
            } थांबा;
        स्थिती GGML_OP_RWKV_WKV6:
            {
                ggml_compute_forward_rwkv_wkv6(params, tensor);
            } थांबा;
        स्थिती GGML_OP_GATED_LINEAR_ATTN:
            {
                ggml_compute_forward_gla(params, tensor);
            } थांबा;
        स्थिती GGML_OP_RWKV_WKV7:
            {
                ggml_compute_forward_rwkv_wkv7(params, tensor);
            } थांबा;
        स्थिती GGML_OP_SOLVE_TRI:
            {
                ggml_compute_forward_solve_tri(params, tensor);
            } थांबा;
        स्थिती GGML_OP_GATED_DELTA_NET:
            {
                ggml_compute_forward_gated_delta_net(params, tensor);
            } थांबा;
        स्थिती GGML_OP_LIGHTNING_INDEXER:
            {
                ggml_compute_forward_lightning_indexer(params, tensor);
            } थांबा;
        स्थिती GGML_OP_DSV4_HC_COMB:
            {
                ggml_compute_forward_dsv4_hc_comb(params, tensor);
            } थांबा;
        स्थिती GGML_OP_DSV4_HC_PRE:
            {
                ggml_compute_forward_dsv4_hc_pre(params, tensor);
            } थांबा;
        स्थिती GGML_OP_DSV4_HC_POST:
            {
                ggml_compute_forward_dsv4_hc_post(params, tensor);
            } थांबा;
        स्थिती GGML_OP_शीर्ष_K_MASK:
            {
                ggml_compute_forward_शीर्ष_k_mask(params, tensor);
            } थांबा;
        स्थिती GGML_OP_MAP_CUSTOM1:
            {
                ggml_compute_forward_map_custom1(params, tensor);
            }
            थांबा;
        स्थिती GGML_OP_MAP_CUSTOM2:
            {
                ggml_compute_forward_map_custom2(params, tensor);
            }
            थांबा;
        स्थिती GGML_OP_MAP_CUSTOM3:
            {
                ggml_compute_forward_map_custom3(params, tensor);
            }
            थांबा;
        स्थिती GGML_OP_CUSTOM:
            {
                ggml_compute_forward_custom(params, tensor);
            }
            थांबा;
        स्थिती GGML_OP_CROSS_ENTROPY_LOSS:
            {
                ggml_compute_forward_cross_entropy_loss(params, tensor);
            }
            थांबा;
        स्थिती GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            {
                ggml_compute_forward_cross_entropy_loss_back(params, tensor);
            }
            थांबा;
        स्थिती GGML_OP_OPT_STEP_ADAMW:
            {
                ggml_compute_forward_opt_step_adamw(params, tensor);
            }
            थांबा;
        स्थिती GGML_OP_OPT_STEP_SGD:
            {
                ggml_compute_forward_opt_step_sgd(params, tensor);
            }
            थांबा;
        स्थिती GGML_OP_NONE:
            {
                // nop
            } थांबा;
        स्थिती GGML_OP_RESHAPE:
            {
                // nop
            } थांबा;
        स्थिती GGML_OP_PERMUTE:
            {
                // nop
            } थांबा;
        स्थिती GGML_OP_VIEW:
            {
                // nop
            } थांबा;
        स्थिती GGML_OP_TRANSPOSE:
            {
                // nop
            } थांबा;
        स्थिती GGML_OP_मोजणी:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// Android's libc implementation "bionic" does not support setting affinity
#जर defined(__gnu_linux__)
स्थिर शून्य set_numa_thread_affinity(पूर्णांक thread_n) {
    जर (!ggml_is_numa()) {
        परत;
    }

    पूर्णांक गाठ_अंक;
    पूर्णांक rv;
    माप_प्रकार setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    निवड(g_state.numa.numa_strategy) {
        स्थिती GGML_NUMA_STRATEGY_DISTRIBUTE:
            // run thread on node_num thread_n / (threads per node)
            गाठ_अंक = thread_n % g_state.numa.n_nodes;
            थांबा;
        स्थिती GGML_NUMA_STRATEGY_ISOLATE:
            // run thread on current_node
            गाठ_अंक = g_state.numa.current_गाठ;
            थांबा;
        स्थिती GGML_NUMA_STRATEGY_NUMACTL:
            // use the cpuset that numactl gave us
            rv = pthread_setaffinity_np(pthread_self(), setsize, &g_state.numa.cpuset);
            जर (rv) {
                ख_लिखो(मानक_त्रुटि, "warning: pthread_setaffinity_np() failed: %s\n",म_त्रुटि(rv));
            }
            परत;
        पूर्वनिर्धारित:
            परत;
    }

    संरचना ggml_numa_गाठ * गाठ = &g_state.numa.nodes[गाठ_अंक];

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    साठी (माप_प्रकार i = 0; i < गाठ->n_cpus; ++i) {
        CPU_SET_S(गाठ->cpus[i], setsize, cpus);
    }

    rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    जर (rv) {
            ख_लिखो(मानक_त्रुटि, "warning: pthread_setaffinity_np() failed: %s\n", म_त्रुटि(rv));
    }

    CPU_FREE(cpus);
}

स्थिर शून्य clear_numa_thread_affinity(शून्य) {
    जर (!ggml_is_numa()) {
        परत;
    }

    माप_प्रकार setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    साठी (अचिन्हित i = 0; i < g_state.numa.total_cpus; ++i) {
        CPU_SET_S(i, setsize, cpus);
    }

    पूर्णांक rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    जर (rv) {
        ख_लिखो(मानक_त्रुटि, "warning: pthread_setaffinity_np() failed: %s\n", म_त्रुटि(rv));
    }

    CPU_FREE(cpus);
}
#अन्यथा
// TODO: Windows etc.
// (the linux implementation may also work on BSD, someone should test)
स्थिर शून्य set_numa_thread_affinity(पूर्णांक thread_n) { UNUSED(thread_n);  }
स्थिर शून्य clear_numa_thread_affinity(शून्य) {}
#endif

स्थिर पूर्णांक ggml_get_n_tasks(संरचना ggml_tensor * गाठ, पूर्णांक n_threads) {
    पूर्णांक n_tasks = 0;

    जर (ggml_is_empty(गाठ)) {
        // no need to multi-thread a no-op
        n_tasks = 1;
        परत n_tasks;
    }

    निवड (गाठ->op) {
        स्थिती GGML_OP_CPY:
        स्थिती GGML_OP_DUP:
        स्थिती GGML_OP_CONT:
        स्थिती GGML_OP_ADD:
        स्थिती GGML_OP_ADD_ID:
        स्थिती GGML_OP_ADD1:
        स्थिती GGML_OP_ACC:
        स्थिती GGML_OP_CUMSUM:
        स्थिती GGML_OP_TRI:
        स्थिती GGML_OP_FILL:
            {
                n_tasks = n_threads;
            } थांबा;
        स्थिती GGML_OP_SUB:
        स्थिती GGML_OP_SQR:
        स्थिती GGML_OP_SQRT:
        स्थिती GGML_OP_LOG:
        स्थिती GGML_OP_SIN:
        स्थिती GGML_OP_COS:
        स्थिती GGML_OP_SUM:
        स्थिती GGML_OP_SUM_ROWS:
        स्थिती GGML_OP_MEAN:
        स्थिती GGML_OP_ARGMAX:
            {
                n_tasks = 1;
            } थांबा;
        स्थिती GGML_OP_मोजणी_EQUAL:
        स्थिती GGML_OP_SOLVE_TRI:
        स्थिती GGML_OP_GATED_DELTA_NET:
        स्थिती GGML_OP_DSV4_HC_COMB:
        स्थिती GGML_OP_DSV4_HC_PRE:
        स्थिती GGML_OP_DSV4_HC_POST:
        स्थिती GGML_OP_शीर्ष_K_MASK:
            {
                n_tasks = n_threads;
            } थांबा;
        स्थिती GGML_OP_REPEAT:
        स्थिती GGML_OP_REPEAT_BACK:
        स्थिती GGML_OP_LEAKY_RELU:
            {
                n_tasks = 1;
            } थांबा;
        स्थिती GGML_OP_UNARY:
            निवड (ggml_get_unary_op(गाठ)) {
                स्थिती GGML_UNARY_OP_ABS:
                स्थिती GGML_UNARY_OP_SGN:
                स्थिती GGML_UNARY_OP_NEG:
                स्थिती GGML_UNARY_OP_STEP:
                स्थिती GGML_UNARY_OP_TANH:
                स्थिती GGML_UNARY_OP_ELU:
                स्थिती GGML_UNARY_OP_RELU:
                स्थिती GGML_UNARY_OP_SIGMOID:
                स्थिती GGML_UNARY_OP_HARDSWISH:
                स्थिती GGML_UNARY_OP_HARDSIGMOID:
                स्थिती GGML_UNARY_OP_EXP:
                स्थिती GGML_UNARY_OP_SOFTPLUS:
                स्थिती GGML_UNARY_OP_EXPM1:
                स्थिती GGML_UNARY_OP_FLOOR:
                स्थिती GGML_UNARY_OP_CEIL:
                स्थिती GGML_UNARY_OP_ROUND:
                स्थिती GGML_UNARY_OP_TRUNC:
                    {
                        n_tasks = 1;
                    } थांबा;

                स्थिती GGML_UNARY_OP_GELU:
                स्थिती GGML_UNARY_OP_GELU_ERF:
                स्थिती GGML_UNARY_OP_GELU_QUICK:
                स्थिती GGML_UNARY_OP_SILU:
                स्थिती GGML_UNARY_OP_XIELU:
                    {
                        n_tasks = n_threads;
                    } थांबा;
                पूर्वनिर्धारित:
                    GGML_ABORT("fatal error");
            }
            थांबा;
        स्थिती GGML_OP_GLU:
            निवड (ggml_get_glu_op(गाठ)) {
                स्थिती GGML_GLU_OP_REGLU:
                स्थिती GGML_GLU_OP_GEGLU:
                स्थिती GGML_GLU_OP_SWIGLU:
                स्थिती GGML_GLU_OP_SWIGLU_OAI:
                स्थिती GGML_GLU_OP_GEGLU_ERF:
                स्थिती GGML_GLU_OP_GEGLU_QUICK:
                स्थिती GGML_GLU_OP_SWIGLU_CLAMP:
                    {
                        n_tasks = n_threads;
                    } थांबा;
                पूर्वनिर्धारित:
                    GGML_ABORT("fatal error");
            }
            थांबा;
        स्थिती GGML_OP_SILU_BACK:
        स्थिती GGML_OP_MUL:
        स्थिती GGML_OP_DIV:
        स्थिती GGML_OP_NORM:
        स्थिती GGML_OP_RMS_NORM:
        स्थिती GGML_OP_RMS_NORM_BACK:
        स्थिती GGML_OP_L2_NORM:
        स्थिती GGML_OP_GROUP_NORM:
        स्थिती GGML_OP_CONCAT:
        स्थिती GGML_OP_MUL_MAT:
        स्थिती GGML_OP_MUL_MAT_ID:
        स्थिती GGML_OP_OUT_PROD:
        स्थिती GGML_OP_SET_ROWS:
            {
                n_tasks = n_threads;
            } थांबा;
        स्थिती GGML_OP_GET_ROWS:
            {
                // FIXME: get_rows can use additional threads, but the cost of launching additional threads
                // decreases performance with GPU offloading
                //n_tasks = n_threads;
                n_tasks = 1;
            } थांबा;
        स्थिती GGML_OP_SCALE:
        स्थिती GGML_OP_SET:
        स्थिती GGML_OP_RESHAPE:
        स्थिती GGML_OP_VIEW:
        स्थिती GGML_OP_PERMUTE:
        स्थिती GGML_OP_TRANSPOSE:
        स्थिती GGML_OP_GET_ROWS_BACK:
        स्थिती GGML_OP_DIAG:
            {
                n_tasks = 1;
            } थांबा;
        स्थिती GGML_OP_DIAG_MASK_ZERO:
        स्थिती GGML_OP_DIAG_MASK_INF:
        स्थिती GGML_OP_SOFT_MAX_BACK:
        स्थिती GGML_OP_ROPE:
        स्थिती GGML_OP_ROPE_BACK:
        स्थिती GGML_OP_ADD_REL_POS:
            {
                n_tasks = n_threads;
            } थांबा;
        स्थिती GGML_OP_CLAMP:
            {
                n_tasks = 1; //TODO
            } थांबा;
        स्थिती GGML_OP_SOFT_MAX:
            {
                n_tasks = MIN(n_threads, ggml_nrows(गाठ->src[0]));
            } थांबा;
        स्थिती GGML_OP_IM2COL:
        स्थिती GGML_OP_IM2COL_BACK:
        स्थिती GGML_OP_IM2COL_3D:
        स्थिती GGML_OP_CONV_2D:
        स्थिती GGML_OP_CONV_3D:
        स्थिती GGML_OP_CONV_2D_DW:
        स्थिती GGML_OP_COL2IM_1D:
        स्थिती GGML_OP_CONV_TRANSPOSE_1D:
        स्थिती GGML_OP_CONV_TRANSPOSE_2D:
            {
                n_tasks = n_threads;
            } थांबा;
        स्थिती GGML_OP_POOL_1D:
        स्थिती GGML_OP_POOL_2D:
        स्थिती GGML_OP_POOL_2D_BACK:
            {
                n_tasks = 1;
            } थांबा;
        स्थिती GGML_OP_UPSCALE:
        स्थिती GGML_OP_PAD:
        स्थिती GGML_OP_PAD_REFLECT_1D:
        स्थिती GGML_OP_ROLL:
        स्थिती GGML_OP_ARANGE:
        स्थिती GGML_OP_TIMESTEP_EMBEDDING:
        स्थिती GGML_OP_ARGSORT:
        स्थिती GGML_OP_शीर्ष_K:
        स्थिती GGML_OP_FLASH_ATTN_EXT:
        स्थिती GGML_OP_FLASH_ATTN_BACK:
        स्थिती GGML_OP_SSM_CONV:
        स्थिती GGML_OP_SSM_SCAN:
        स्थिती GGML_OP_LIGHTNING_INDEXER:
            {
                n_tasks = n_threads;
            } थांबा;
        स्थिती GGML_OP_RWKV_WKV6:
        स्थिती GGML_OP_GATED_LINEAR_ATTN:
        स्थिती GGML_OP_RWKV_WKV7:
            {
                स्थिरांक int64_t n_heads = गाठ->src[1]->ne[1];
                n_tasks = MIN(n_threads, n_heads);
            } थांबा;
        स्थिती GGML_OP_WIN_PART:
        स्थिती GGML_OP_WIN_UNPART:
        स्थिती GGML_OP_GET_REL_POS:
            {
                n_tasks = 1;
            } थांबा;
        स्थिती GGML_OP_MAP_CUSTOM1:
            {
                संरचना ggml_map_custom1_op_params p;
                स_नकल(&p, गाठ->op_params, आकार(p));
                जर (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } अन्यथा {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } थांबा;
        स्थिती GGML_OP_MAP_CUSTOM2:
            {
                संरचना ggml_map_custom2_op_params p;
                स_नकल(&p, गाठ->op_params, आकार(p));
                जर (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } अन्यथा {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } थांबा;
        स्थिती GGML_OP_MAP_CUSTOM3:
            {
                संरचना ggml_map_custom3_op_params p;
                स_नकल(&p, गाठ->op_params, आकार(p));
                जर (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } अन्यथा {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } थांबा;
        स्थिती GGML_OP_CUSTOM:
            {
                संरचना ggml_custom_op_params p;
                स_नकल(&p, गाठ->op_params, आकार(p));
                जर (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } अन्यथा {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } थांबा;
        स्थिती GGML_OP_CROSS_ENTROPY_LOSS:
        स्थिती GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        स्थिती GGML_OP_OPT_STEP_ADAMW:
        स्थिती GGML_OP_OPT_STEP_SGD:
            {
                n_tasks = n_threads;
            } थांबा;
        स्थिती GGML_OP_NONE:
            {
                n_tasks = 1;
            } थांबा;
        स्थिती GGML_OP_मोजणी:
            {
                GGML_ABORT("fatal error");
            }
        पूर्वनिर्धारित:
            {
                ख_लिखो(मानक_त्रुटि, "%s: op not implemented: ", __func__);
                जर (गाठ->op < GGML_OP_मोजणी) {
                    ख_लिखो(मानक_त्रुटि, "%s\n", ggml_op_name(गाठ->op));
                } अन्यथा {
                    ख_लिखो(मानक_त्रुटि, "%d\n", गाठ->op);
                }
                GGML_ABORT("fatal error");
            }
    }

    प्रतिज्ञा(n_tasks > 0);

    परत n_tasks;
}

स्थिर thread_ret_t ggml_graph_compute_secondary_thread(शून्य* data);

#जर defined(_WIN32)
#अन्तर्भाव "windows.h"

// TODO: support > 64 CPUs
स्थिर bool ggml_thread_apply_affinity(bool * mask) {
    HANDLE    h = GetCurrentThread();
    uint64_t  bitmask = 0ULL;

    प्रतिज्ञा(GGML_MAX_N_THREADS >= 64);

    साठी (int32_t i = 0; i < 8; i++) {
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

    साठी (int32_t i = 64; i < GGML_MAX_N_THREADS; i++) {
        जर (mask[i]) {
            ख_लिखो(मानक_त्रुटि, "warn: setting thread-affinity for > 64 CPUs isn't supported on windows!\n");
            थांबा;
        }
    }

    DWORD_PTR m = (DWORD_PTR)bitmask;

    m = SetThreadAffinityMask(h, m);

    परत m != 0;
}

स्थिर bool ggml_thread_apply_priority(int32_t prio) {
    // Note that on Windows the Process Priority Class must be updated in order to set Thread priority.
    // This is up to the applications.
    DWORD p = THREAD_PRIORITY_NORMAL;
    निवड (prio) {
        स्थिती GGML_SCHED_PRIO_खालचा:      p = THREAD_PRIORITY_BELOW_NORMAL;  थांबा;
        स्थिती GGML_SCHED_PRIO_NORMAL:   p = THREAD_PRIORITY_NORMAL;        थांबा;
        स्थिती GGML_SCHED_PRIO_MEDIUM:   p = THREAD_PRIORITY_ABOVE_NORMAL;  थांबा;
        स्थिती GGML_SCHED_PRIO_वरचा:     p = THREAD_PRIORITY_HIGHEST;       थांबा;
        स्थिती GGML_SCHED_PRIO_REALTIME: p = THREAD_PRIORITY_TIME_CRITICAL; थांबा;
    }

    जर (prio != GGML_SCHED_PRIO_खालचा) {
        // Tell Windows that this thread should not be throttled (needs its own CPU core).
        // Newer Windows 11 versions aggressively park (offline) CPU cores and often place
        // all our threads onto the first 4 cores which results in terrible performance with
        // n_threads > 4
        #जर _WIN32_WINNT >= 0x0602
        THREAD_POWER_THROTTLING_STATE t;
        ZeroMemory(&t, आकार(t));
        t.Version     = THREAD_POWER_THROTTLING_CURRENT_VERSION;
        t.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
        t.StateMask   = 0;

        जर (!SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &t, आकार(t))) {
            GGML_LOG_DEBUG("failed to disable thread power throttling %d : (%d)\n", prio, (पूर्णांक) GetLastError());
            परत false;
        }
        #endif
    }

    जर (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        परत true;
    }

    जर (!SetThreadPriority(GetCurrentThread(), p)) {
        ख_लिखो(मानक_त्रुटि, "warn: failed to set thread priority %d : (%d)\n", prio, (पूर्णांक) GetLastError());
        परत false;
    }

    परत true;
}

#किंवाजर defined(__APPLE__)
#अन्तर्भाव <sys/types.h>
#अन्तर्भाव <sys/resource.h>

स्थिर bool ggml_thread_apply_affinity(स्थिरांक bool * mask) {
    // Not supported on Apple platforms
    UNUSED(mask);
    परत true;
}

स्थिर bool ggml_thread_apply_priority(int32_t prio) {
    संरचना sched_param p;
    int32_t policy = SCHED_OTHER;
    निवड (prio) {
        // TODO: there seems to be no way to set lower prio on Apple platforms
        स्थिती GGML_SCHED_PRIO_खालचा:      policy = SCHED_OTHER; p.sched_priority = 0;  थांबा;
        स्थिती GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  थांबा;
        स्थिती GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; थांबा;
        स्थिती GGML_SCHED_PRIO_वरचा:     policy = SCHED_FIFO;  p.sched_priority = 80; थांबा;
        स्थिती GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; थांबा;
    }

    जर (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        परत true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    जर (err != 0) {
        ख_लिखो(मानक_त्रुटि, "warn: failed to set thread priority %d : %s (%d)\n", prio, म_त्रुटि(err), err);
        परत false;
    }

    परत true;
}

#किंवाजर defined(__linux__)
// TODO: this may not work on BSD, to be verified

स्थिर bool ggml_thread_apply_affinity(स्थिरांक bool * mask) {
    cpu_set_t cpuset;
    पूर्णांक err;

    CPU_ZERO(&cpuset);

    साठी (uint32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
        जर (mask[i]) {
            GGML_PRINT_DEBUG("Thread %lx: adding %d to cpuset\n", pthread_self(), i);
            CPU_SET(i, &cpuset);
        }
    }

#ifdef __ANDROID__
    err = sched_setaffinity(0, आकार(cpuset), &cpuset);
    जर (err < 0) {
        err = त्रुटि_सं;
    }
#अन्यथा
    err = pthread_setaffinity_np(pthread_self(), आकार(cpuset), &cpuset);
#endif
    जर (err != 0) {
        ख_लिखो(मानक_त्रुटि, "warn: failed to set affinity mask 0x%llx : %s (%d)\n", (अचिन्हित दीर्घ दीर्घ)mask, म_त्रुटि(err), err);
        परत false;
    }

    परत true;
}

स्थिर bool ggml_thread_apply_priority(int32_t prio) {
    संरचना sched_param p;
    int32_t policy = SCHED_OTHER;
    निवड (prio) {
        स्थिती GGML_SCHED_PRIO_खालचा:      policy = SCHED_BATCH; p.sched_priority = 0;  थांबा;
        स्थिती GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  थांबा;
        स्थिती GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; थांबा;
        स्थिती GGML_SCHED_PRIO_वरचा:     policy = SCHED_FIFO;  p.sched_priority = 80; थांबा;
        स्थिती GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; थांबा;
    }

    जर (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        परत true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    जर (err != 0) {
        ख_लिखो(मानक_त्रुटि, "warn: failed to set thread priority %d : %s (%d)\n", prio, म_त्रुटि(err), err);
        परत false;
    }

    परत true;
}

#अन्यथा // unsupported platforms

स्थिर bool ggml_thread_apply_affinity(स्थिरांक bool * mask) {
    UNUSED(mask);
    परत true;
}

स्थिर bool ggml_thread_apply_priority(int32_t prio) {
    UNUSED(prio);
    परत true;
}

#endif

स्थिर bool ggml_thread_cpumask_is_valid(स्थिरांक bool * mask) {
    साठी (पूर्णांक i = 0; i < GGML_MAX_N_THREADS; i++) {
        जर (mask[i]) { परत true; }
    }
    परत false;
}

स्थिर शून्य ggml_thread_cpumask_next(स्थिरांक bool * global_mask, bool * local_mask, bool strict, int32_t* iter) {
    जर (!strict) {
        स_नकल(local_mask, global_mask, GGML_MAX_N_THREADS);
        परत;
    } अन्यथा {
        स_रखो(local_mask, 0, GGML_MAX_N_THREADS);
        int32_t base_idx = *iter;
        साठी (int32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
            int32_t idx = base_idx + i;
            जर (idx >= GGML_MAX_N_THREADS) {
                // Just a cheaper modulo
                idx -= GGML_MAX_N_THREADS;
            }
            जर (global_mask[idx]) {
                local_mask[idx] = 1;
                *iter = idx + 1;
                परत;
            }
        }
    }
}

शून्य ggml_threadpool_free(संरचना ggml_threadpool* threadpool) {
    जर (!threadpool) परत;

    स्थिरांक पूर्णांक n_threads = threadpool->n_threads;

#ifndef GGML_USE_OPENMP
    संरचना ggml_compute_state* workers = threadpool->workers;

    ggml_mutex_lock(&threadpool->mutex);

    threadpool->stop = true;
    threadpool->छोड़ो = false;

    ggml_cond_broadcast(&threadpool->cond);
    ggml_mutex_unlock(&threadpool->mutex);

    साठी (पूर्णांक j = 1; j < n_threads; j++) {
        int32_t rc = ggml_thread_join(workers[j].thrd, रिक्त);
        GGML_ASSERT(rc == GGML_EXIT_SUCCESS || rc == GGML_EXIT_ABORTED);
        UNUSED(rc);
    }

    ggml_mutex_destroy(&threadpool->mutex);
    ggml_cond_destroy(&threadpool->cond);
#endif // GGML_USE_OPENMP

    स्थिरांक माप_प्रकार workers_size = आकार(संरचना ggml_compute_state) * n_threads;
    ggml_aligned_free(threadpool->workers, workers_size);
    ggml_aligned_free(threadpool, आकार(संरचना ggml_threadpool));
}

#ifndef GGML_USE_OPENMP
// pause/resume must be called under mutex
स्थिर शून्य ggml_threadpool_pause_locked(संरचना ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Pausing threadpool\n");
    threadpool->छोड़ो = true;
    ggml_cond_broadcast(&threadpool->cond);
}

स्थिर शून्य ggml_threadpool_resume_locked(संरचना ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Resuming threadpool\n");
    threadpool->छोड़ो = false;
    ggml_cond_broadcast(&threadpool->cond);
}
#endif

शून्य ggml_threadpool_pause(संरचना ggml_threadpool * threadpool) {
#ifndef GGML_USE_OPENMP
    ggml_mutex_lock(&threadpool->mutex);
    जर (!threadpool->छोड़ो) {
       ggml_threadpool_pause_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#अन्यथा
    UNUSED(threadpool);
#endif
}

शून्य ggml_threadpool_resume(संरचना ggml_threadpool * threadpool) {
#ifndef GGML_USE_OPENMP
    ggml_mutex_lock(&threadpool->mutex);
    जर (threadpool->छोड़ो) {
       ggml_threadpool_resume_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#अन्यथा
    UNUSED(threadpool);
#endif
}

संरचना ggml_cplan ggml_graph_plan(
          स्थिरांक संरचना ggml_cgraph * cgraph,
                               पूर्णांक   n_threads,
            संरचना ggml_threadpool * threadpool) {

    जर (threadpool == रिक्त) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
    }
    जर (n_threads <= 0) {
        n_threads = threadpool ? threadpool->n_threads : GGML_DEFAULT_N_THREADS;
    }

#जर defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
    // Emscripten without pthreads support can only use a single thread
    n_threads = 1;
#endif

#जर defined(__wasi__)
    // WASI doesn't support parallelism yet
    n_threads = 1;
#endif

    माप_प्रकार work_size = 0;

    संरचना ggml_cplan cplan;
    स_रखो(&cplan, 0, आकार(संरचना ggml_cplan));

    पूर्णांक max_tasks = 1;

    // thread scheduling for the different operations + work buffer size estimation
    साठी (पूर्णांक i = 0; i < cgraph->n_nodes; i++) {
        संरचना ggml_tensor * गाठ = cgraph->nodes[i];

        स्थिरांक पूर्णांक n_tasks = ggml_get_n_tasks(गाठ, n_threads);

        max_tasks = MAX(max_tasks, n_tasks);

        माप_प्रकार cur = 0;

        जर (!ggml_cpu_extra_work_size(n_threads, गाठ, &cur)) {
            निवड (गाठ->op) {
                स्थिती GGML_OP_CPY:
                स्थिती GGML_OP_DUP:
                    {
                        जर (ggml_is_quantized(गाठ->type) ||
                            // F16 -> BF16 and BF16 -> F16 copies go through intermediate F32
                            (गाठ->src[0]->type == GGML_TYPE_F16  && गाठ->src[1] && गाठ->src[1]->type == GGML_TYPE_BF16) ||
                            (गाठ->src[0]->type == GGML_TYPE_BF16 && गाठ->src[1] && गाठ->src[1]->type == GGML_TYPE_F16) ||
                            // conversion between F32 and I32
                            (गाठ->src[0]->type == GGML_TYPE_F32 && गाठ->src[1] && गाठ->src[1]->type == GGML_TYPE_I32) ||
                            (गाठ->src[0]->type == GGML_TYPE_I32 && गाठ->src[1] && गाठ->src[1]->type == GGML_TYPE_F32)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * गाठ->ne[0] * n_tasks;
                        }
                    } थांबा;
                स्थिती GGML_OP_ADD:
                स्थिती GGML_OP_ADD_ID:
                स्थिती GGML_OP_ADD1:
                    {
                        जर (ggml_is_quantized(गाठ->src[0]->type)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * गाठ->src[0]->ne[0] * n_tasks;
                        }
                    } थांबा;
                स्थिती GGML_OP_ACC:
                    {
                        जर (ggml_is_quantized(गाठ->src[0]->type)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * गाठ->src[1]->ne[0] * n_tasks;
                        }
                    } थांबा;
                स्थिती GGML_OP_मोजणी_EQUAL:
                    {
                        cur = ggml_type_size(गाठ->type)*n_tasks;
                    } थांबा;
                स्थिती GGML_OP_MUL_MAT:
                    {
                        स्थिरांक गणना ggml_type vec_dot_type = type_traits_cpu[गाठ->src[0]->type].vec_dot_type;
                        जर (गाठ->src[1]->type != vec_dot_type) {
                            cur = ggml_row_size(vec_dot_type, ggml_nelements(गाठ->src[1]));
                        }
                        // Workspace for tiled (see tiled.h)
                        cur = GGML_PAD(cur, 64);
                        cur += ggml_tiled_wdata_size(n_tasks, गाठ);
                    } थांबा;
                स्थिती GGML_OP_MUL_MAT_ID:
                    {
                        cur = 0;
                        स्थिरांक संरचना ggml_tensor * src0 = गाठ->src[0];
                        स्थिरांक संरचना ggml_tensor * src1 = गाठ->src[1];
                        स्थिरांक संरचना ggml_tensor * ids = गाठ->src[2];
                        स्थिरांक गणना ggml_type vec_dot_type = type_traits_cpu[src0->type].vec_dot_type;
                        स्थिरांक पूर्णांक n_as = src0->ne[2];
                        // src1
                        जर (src1->type != vec_dot_type) {
                            cur += ggml_row_size(vec_dot_type, ggml_nelements(src1)) + आकार(int64_t);
                        }
                        // matrix_row_counts
                        cur += n_as * आकार(int64_t) + आकार(int64_t);
                        // matrix_rows
                        cur += n_as*ids->ne[0]*ids->ne[1]*आकार(संरचना mmid_row_mapping) + आकार(int64_t);
                        // atomic_current_chunk
                        cur += CACHE_ओळ_SIZE*n_as + CACHE_ओळ_SIZE;
                        // Workspace for tiled (see tiled.h)
                        cur = GGML_PAD(cur, 64);
                        cur += ggml_tiled_wdata_size(n_tasks, गाठ);
                    } थांबा;
                स्थिती GGML_OP_OUT_PROD:
                    {
                        जर (ggml_is_quantized(गाठ->src[0]->type) ||
                            गाठ->src[0]->type == GGML_TYPE_F16) {
                            cur = ggml_type_size(GGML_TYPE_F32) * गाठ->src[0]->ne[0] * n_tasks;
                        }
                    } थांबा;
                स्थिती GGML_OP_SET_ROWS:
                    {
                        जर (गाठ->src[0]->type == GGML_TYPE_F16 && गाठ->type != GGML_TYPE_F16) {
                            cur = ggml_type_size(GGML_TYPE_F32) * गाठ->src[0]->ne[0] * n_tasks;
                        }
                    } थांबा;
                स्थिती GGML_OP_SOFT_MAX:
                स्थिती GGML_OP_ROPE:
                स्थिती GGML_OP_ROPE_BACK:
                    {
                        cur = ggml_type_size(GGML_TYPE_F32) * गाठ->ne[0] * n_tasks;
                    } थांबा;
                स्थिती GGML_OP_CONV_TRANSPOSE_1D:
                    {
                        GGML_ASSERT(गाठ->src[0]->ne[3] == 1);
                        GGML_ASSERT(गाठ->src[1]->ne[2] == 1);
                        GGML_ASSERT(गाठ->src[1]->ne[3] == 1);

                        स्थिरांक int64_t ne00 = गाठ->src[0]->ne[0];  // K
                        स्थिरांक int64_t ne01 = गाठ->src[0]->ne[1];  // Cout
                        स्थिरांक int64_t ne02 = गाठ->src[0]->ne[2];  // Cin
                        स्थिरांक int64_t ne10 = गाठ->src[1]->ne[0];  // L
                        स्थिरांक int64_t ne11 = गाठ->src[1]->ne[1];  // Cin

                        जर ((गाठ->src[0]->type == GGML_TYPE_F16 ||
                             गाठ->src[0]->type == GGML_TYPE_BF16) &&
                            गाठ->src[1]->type == GGML_TYPE_F32) {
                            cur += आकार(ggml_fp16_t)*ne00*ne01*ne02;
                            cur += आकार(ggml_fp16_t)*ne10*ne11;
                        } अन्यथा जर (गाठ->src[0]->type == GGML_TYPE_F32 &&
                                   गाठ->src[1]->type == GGML_TYPE_F32) {
                            cur += आकार(दशांश)*ne00*ne01*ne02;
                            cur += आकार(दशांश)*ne10*ne11;
                        } अन्यथा {
                            GGML_ABORT("fatal error");
                        }
                    } थांबा;
                स्थिती GGML_OP_CONV_2D:
                स्थिती GGML_OP_CONV_3D:
                    {
                        cur = GGML_IM2COL_WORK_SIZE;
                    } थांबा;
                स्थिती GGML_OP_CONV_TRANSPOSE_2D:
                    {
                        स्थिरांक int64_t ne00 = गाठ->src[0]->ne[0]; // W
                        स्थिरांक int64_t ne01 = गाठ->src[0]->ne[1]; // H
                        स्थिरांक int64_t ne02 = गाठ->src[0]->ne[2]; // Channels Out
                        स्थिरांक int64_t ne03 = गाठ->src[0]->ne[3]; // Channels In

                        स्थिरांक int64_t ne10 = गाठ->src[1]->ne[0]; // W
                        स्थिरांक int64_t ne11 = गाठ->src[1]->ne[1]; // H
                        स्थिरांक int64_t ne12 = गाठ->src[1]->ne[2]; // Channels In
                        स्थिरांक int64_t ne13 = गाठ->src[1]->ne[3]; // Batch

                        GGML_ASSERT(गाठ->src[0]->type == GGML_TYPE_F16 || गाठ->src[0]->type == GGML_TYPE_F32);
                        GGML_ASSERT(गाठ->src[1]->type == GGML_TYPE_F32);

                        cur += ggml_type_size(गाठ->src[0]->type) * ne00 * ne01 * ne02 * ne03;
                        cur += ggml_type_size(गाठ->src[0]->type) * ne10 * ne11 * ne12 * ne13;

                    } थांबा;
                स्थिती GGML_OP_शीर्ष_K:
                    {
                        cur += आकार(int32_t)*गाठ->src[0]->ne[0]*n_tasks;
                    } थांबा;
                स्थिती GGML_OP_FLASH_ATTN_EXT:
                    {
                        स्थिरांक int64_t neq2 = गाठ->src[0]->ne[2]; // number of query heads
                        स्थिरांक int64_t DK = गाठ->src[1]->ne[0];
                        स्थिरांक int64_t DV = गाठ->src[2]->ne[0];

                        // Tiled flash attention scratch (tile sizes defined in common.h)
                        // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
                        माप_प्रकार prefill  = आकार(दशांश)*(GGML_FA_TILE_Q*DK + 2*GGML_FA_TILE_Q*GGML_FA_TILE_KV + GGML_FA_TILE_Q*DV + GGML_FA_TILE_KV*DV + GGML_FA_TILE_KV*DK)*n_tasks;

                        // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
                        // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
                        माप_प्रकार n_chunks = n_tasks;
                        माप_प्रकार decode   = आकार(दशांश)*(neq2*n_chunks*(2+DV) + n_tasks*(DK + 2*DV));

                        cur += MAX(prefill, decode);
                    } थांबा;
                स्थिती GGML_OP_FLASH_ATTN_BACK:
                    {
                        स्थिरांक int64_t    D = गाठ->src[0]->ne[0];
                        स्थिरांक int64_t ne11 = ggml_up(गाठ->src[1]->ne[1], GGML_SOFT_MAX_UNROLL);
                        स्थिरांक int64_t mxDn = MAX(D, ne11) * 2; // *2 because of S and SM in ggml_compute_forward_flash_attn_back
                        जर (गाठ->src[1]->type == GGML_TYPE_F32) {
                            cur  = आकार(दशांश)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += आकार(दशांश)*mxDn*n_tasks; // this is overestimated by x2
                        } अन्यथा जर (गाठ->src[1]->type == GGML_TYPE_F16) {
                            cur  = आकार(दशांश)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += आकार(दशांश)*mxDn*n_tasks; // this is overestimated by x2
                        } अन्यथा जर (गाठ->src[1]->type == GGML_TYPE_BF16) {
                            cur  = आकार(दशांश)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += आकार(दशांश)*mxDn*n_tasks; // this is overestimated by x2
                        }
                    } थांबा;

                स्थिती GGML_OP_CROSS_ENTROPY_LOSS:
                    {
                        cur = ggml_type_size(गाठ->type)*(n_tasks + गाठ->src[0]->ne[0]*n_tasks);
                    } थांबा;
                स्थिती GGML_OP_GATED_DELTA_NET:
                    {
                        स्थिरांक int64_t S_v = गाठ->src[2]->ne[0];
                        स्थिरांक int64_t K   = ggml_get_op_params_i32(गाठ, 0);
                        स्थिरांक int64_t per_thread = S_v + (K > 1 ? S_v * S_v : 0);
                        cur = per_thread * आकार(दशांश) * n_tasks;
                    } थांबा;
                स्थिती GGML_OP_मोजणी:
                    {
                        GGML_ABORT("fatal error");
                    }
                स्थिती GGML_OP_LIGHTNING_INDEXER:
                    {
                        // temp buffer for dequantizing lightning indexer keys
                        स्थिरांक int64_t ne10 = गाठ->src[1]->ne[0];
                        cur += आकार(दशांश)*ne10*n_tasks;
                    } थांबा;
                पूर्वनिर्धारित:
                    थांबा;
            }
        }

        work_size = MAX(work_size, cur);
    }

    जर (work_size > 0) {
        work_size += CACHE_ओळ_SIZE*(n_threads);
    }

    cplan.threadpool = threadpool;
    cplan.n_threads  = MIN(max_tasks, n_threads);
    cplan.work_size  = work_size;
    cplan.work_data  = रिक्त;

    परत cplan;
}


// Try to fuse the current node with subsequent nodes for better performance.
// Returns the number of nodes skipped by fusion (>=1), or 0 if no fusion was applied.
स्थिर bool ggml_cpu_disable_fusion = false;  // initialized once in ggml_cpu_init(), read-only afterwards

स्थिर पूर्णांक ggml_cpu_try_fuse_ops(
        स्थिरांक संरचना ggml_cgraph * cgraph,
        स्थिरांक पूर्णांक गाठ_n,
        स्थिरांक संरचना ggml_compute_params * params,
        स्थिरांक संरचना ggml_cplan * cplan) {

    जर (ggml_cpu_disable_fusion || cplan->use_ref) {
        परत 0;
    }

    संरचना ggml_tensor * गाठ = cgraph->nodes[गाठ_n];

    जर (गाठ->op == GGML_OP_RMS_NORM) {
        // RMS_NORM + MUL fusion
        स्थिरांक गणना ggml_op fuse_ops[] = { GGML_OP_RMS_NORM, GGML_OP_MUL };
        जर (ggml_can_fuse(cgraph, गाठ_n, fuse_ops, 2)) {
            संरचना ggml_tensor * mul_गाठ = cgraph->nodes[गाठ_n + 1];
            स्थिरांक संरचना ggml_tensor * mul_w = (mul_गाठ->src[0] == गाठ)
                ? mul_गाठ->src[1] : mul_गाठ->src[0];
            जर (गाठ->src[0]->type  == GGML_TYPE_F32 &&
                mul_गाठ->type      == GGML_TYPE_F32 &&
                mul_w->type         == GGML_TYPE_F32 &&
                mul_w->ne[0]        == गाठ->ne[0]   &&
                mul_w->nb[0]        == आकार(दशांश)) {

                ggml_compute_forward_rms_norm_mul_fused(params, गाठ, mul_गाठ);
                परत 1;
            }
        }
    }

    परत 0;
}

स्थिर thread_ret_t ggml_graph_compute_thread(शून्य * data) {
    संरचना ggml_compute_state * state = (संरचना ggml_compute_state *) data;
    संरचना ggml_threadpool    * tp    = state->threadpool;

    स्थिरांक संरचना ggml_cgraph * cgraph = tp->cgraph;
    स्थिरांक संरचना ggml_cplan  * cplan  = tp->cplan;

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(state->ith);
#अन्यथा
    set_numa_thread_affinity(state->ith);
#endif

    संरचना ggml_compute_params params = {
        /*.ith        =*/ state->ith,
        /*.nth        =*/ atomic_load_explicit(&tp->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK,
        /*.wsize      =*/ cplan->work_size,
        /*.wdata      =*/ cplan->work_data,
        /*.threadpool =*/ tp,
        /*.use_ref    =*/ cplan->use_ref,
    };

#ifdef GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p\n", state->ith, (स्थिरांक शून्य *)cplan);
#अन्यथा
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p last-graph %d\n", state->ith, (स्थिरांक शून्य *)cplan, state->last_graph);
#endif

    साठी (पूर्णांक गाठ_n = 0; गाठ_n < cgraph->n_nodes && atomic_load_explicit(&tp->पात, memory_order_relaxed) != गाठ_n; गाठ_n++) {
        संरचना ggml_tensor * गाठ = cgraph->nodes[गाठ_n];

        जर (ggml_op_is_empty(गाठ->op)) {
            // skip NOPs
            सुरूठेवा;
        }

        जर ((गाठ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            सुरूठेवा;
        }

        // TODO: move fused-op detection into ggml_graph_plan so fusion decisions are made once at planning time
        // Try fused ops, fall back to normal compute
        स्थिरांक पूर्णांक n_fused = ggml_cpu_try_fuse_ops(cgraph, गाठ_n, &params, cplan);
        जर (n_fused > 0) {
            गाठ_n += n_fused;
        } अन्यथा {
            ggml_compute_forward(&params, गाठ);
        }

        जर (state->ith == 0 && cplan->abort_callback &&
                cplan->abort_callback(cplan->abort_callback_data)) {
            atomic_store_explicit(&tp->पात, गाठ_n + 1, memory_order_relaxed);
            tp->ec    = GGML_STATUS_ABORTED;
        }

        जर (गाठ_n + 1 < cgraph->n_nodes) {
            ggml_barrier(state->threadpool);
        }
    }

#ifdef GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p\n", state->ith, (स्थिरांक शून्य *)cplan);
#अन्यथा
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p last-graph %d\n", state->ith, (स्थिरांक शून्य *)cplan, state->last_graph);
#endif

    ggml_barrier(state->threadpool);

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(state->ith);
#endif

    परत 0;
}

#ifndef GGML_USE_OPENMP

// check if thread is ready to proceed (exit from polling or sleeping)
// returns true if loops should exit, sets state->pending to indicate new work
स्थिर ओळीत bool ggml_graph_compute_thread_ready(संरचना ggml_compute_state * state) {
    संरचना ggml_threadpool * threadpool = state->threadpool;

    जर (state->pending || threadpool->stop || threadpool->छोड़ो) { परत true; }

    // check for new graph/work
    पूर्णांक n_graph   = atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed);
    पूर्णांक n_threads = n_graph & GGML_THREADPOOL_N_THREADS_MASK;
    जर (n_graph != state->last_graph) {
        state->pending    = (state->ith < n_threads);
        state->last_graph = n_graph;
        परत true;
    }

    परत false;
}

// sync thread state after polling
स्थिर ओळीत शून्य ggml_graph_compute_thread_sync(संरचना ggml_compute_state * state) {
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&state->threadpool->n_graph, 0, memory_order_seq_cst);
    #अन्यथा
    atomic_thread_fence(memory_order_seq_cst);
    #endif
    UNUSED(state);
}

स्थिर ओळीत bool ggml_graph_compute_poll_for_work(संरचना ggml_compute_state * state) {
    संरचना ggml_threadpool * threadpool = state->threadpool;

    // This seems to make 0 ... 100 a decent range for polling level across modern processors.
    // Perhaps, we can adjust it dynamically based on load and things.
    स्थिरांक uint64_t n_rounds = 1024UL * 128 * threadpool->poll;

    साठी (uint64_t i=0; !ggml_graph_compute_thread_ready(state) && i < n_rounds; i++) {
        // No new work. Keep polling.
        ggml_thread_cpu_relax();
    }

    परत state->pending;
}

स्थिर ओळीत bool ggml_graph_compute_check_for_work(संरचना ggml_compute_state * state) {
    संरचना ggml_threadpool * threadpool = state->threadpool;

    जर (ggml_graph_compute_poll_for_work(state)) {
        ggml_graph_compute_thread_sync(state);
        परत state->pending;
    }

    ggml_mutex_lock_shared(&threadpool->mutex);
    पर्यंत (!ggml_graph_compute_thread_ready(state)) {
        // No new work. Wait for the signal.
        GGML_PRINT_DEBUG("thread #%d waiting for work (sleeping)\n", state->ith);
        ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
    }
    ggml_mutex_unlock_shared(&threadpool->mutex);

    परत state->pending;
}

स्थिर thread_ret_t ggml_graph_compute_secondary_thread(शून्य* data) {
    संरचना ggml_compute_state * state = (संरचना ggml_compute_state *) data;
    संरचना ggml_threadpool * threadpool = state->threadpool;

    ggml_thread_apply_priority(threadpool->prio);
    जर (ggml_thread_cpumask_is_valid(state->cpumask)) {
        ggml_thread_apply_affinity(state->cpumask);
    }

    पर्यंत (true) {
        // Check if we need to sleep
        पर्यंत (threadpool->छोड़ो) {
            GGML_PRINT_DEBUG("thread #%d inside pause loop\n", state->ith);
            ggml_mutex_lock_shared(&threadpool->mutex);
            जर (threadpool->छोड़ो) {
                ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
            }
            GGML_PRINT_DEBUG("thread #%d resuming after wait\n", state->ith);
            ggml_mutex_unlock_shared(&threadpool->mutex);
        }

        // This needs to be checked for after the cond_wait
        जर (threadpool->stop) थांबा;

        // Check if there is new work
        // The main thread is the only one that can dispatch new work

        ggml_graph_compute_check_for_work(state);
        जर (state->pending) {
            state->pending = false;
            ggml_graph_compute_thread(state);
        }
    }

    परत (thread_ret_t) 0;
}

// Start processing new graph
स्थिर शून्य ggml_graph_compute_kickoff(संरचना ggml_threadpool * threadpool, पूर्णांक n_threads)
{
    // Always take the mutex here because the worker threads are doing hybrid poll/wait

    ggml_mutex_lock(&threadpool->mutex);

    // Update the number of active threads and the graph count
    पूर्णांक n_graph = atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed) >> GGML_THREADPOOL_N_THREADS_BITS;
    n_graph = ((n_graph + 1) << GGML_THREADPOOL_N_THREADS_BITS) | (n_threads & GGML_THREADPOOL_N_THREADS_MASK);

    GGML_PRINT_DEBUG("compute-kickoff: n_threads %d n_graph %d\n", n_threads, n_graph);

    // Indicate the graph is ready to be processed
    // We need the full seq-cst fence here because of the polling threads (used in thread_sync)
    atomic_store_explicit(&threadpool->n_graph, n_graph, memory_order_seq_cst);

    जर (threadpool->छोड़ो) {
       // Update main thread prio and affinity to match the threadpool settings
       ggml_thread_apply_priority(threadpool->prio);
       जर (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
           ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
       }

       // resume does cond broadcast
       ggml_threadpool_resume_locked(threadpool);
    } अन्यथा {
       ggml_cond_broadcast(&threadpool->cond);
    }

    ggml_mutex_unlock(&threadpool->mutex);
}

#endif // GGML_USE_OPENMP

स्थिर संरचना ggml_threadpool * ggml_threadpool_new_impl(
    संरचना ggml_threadpool_params * tpp,
               संरचना ggml_cgraph * cgraph,
                संरचना ggml_cplan * cplan) {

    संरचना ggml_threadpool * threadpool =
        ggml_aligned_malloc(आकार(संरचना ggml_threadpool));
    {
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->n_graph          = 0;
        threadpool->n_barrier        = 0;
        threadpool->n_barrier_passed = 0;
        threadpool->current_chunk    = 0;
        threadpool->stop             = false;
        threadpool->छोड़ो            = tpp->paused;
        threadpool->पात            = -1;
        threadpool->workers          = रिक्त;
        threadpool->n_threads        = tpp->n_threads;
        threadpool->poll             = tpp->poll;
        threadpool->prio             = tpp->prio;
        threadpool->ec               = GGML_STATUS_SUCCESS;
    }

    // Allocate and init workers state
    स्थिरांक माप_प्रकार workers_size = आकार(संरचना ggml_compute_state) * tpp->n_threads;
    संरचना ggml_compute_state * workers = ggml_aligned_malloc(workers_size);

    स_रखो(workers, 0, workers_size);
    साठी (पूर्णांक j = 0; j < tpp->n_threads; j++) {
        workers[j].threadpool = threadpool;
        workers[j].ith        = j;
    }

    threadpool->workers = workers;

#ifdef GGML_USE_OPENMP
    int32_t cpumask_iter = 0;

    // Compute CPU masks for each thread
    साठी (पूर्णांक j = 0; j < tpp->n_threads; j++) {
        ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);
    }
#अन्यथा // GGML_USE_OPENMP
    ggml_mutex_init(&threadpool->mutex);
    ggml_cond_init(&threadpool->cond);

    // Spin the threads for all workers, and update CPU placements.
    // Place the main thread last (towards the higher numbered CPU cores).

    int32_t cpumask_iter = 0;

    साठी (पूर्णांक j = 1; j < tpp->n_threads; j++) {
        ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);

        int32_t rc = ggml_thread_create(&workers[j].thrd, रिक्त, ggml_graph_compute_secondary_thread, &workers[j]);
        GGML_ASSERT(rc == 0);
    }

    ggml_thread_cpumask_next(tpp->cpumask, workers[0].cpumask, tpp->strict_cpu, &cpumask_iter);

    जर (!threadpool->छोड़ो) {
        // Update main thread prio and affinity at the start, otherwise we'll do it in resume
        ggml_thread_apply_priority(threadpool->prio);
        जर (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
            ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
        }
    }
#endif // GGML_USE_OPENMP

    परत threadpool;
}

संरचना ggml_threadpool * ggml_threadpool_new(संरचना ggml_threadpool_params * tpp) {
    परत ggml_threadpool_new_impl(tpp, रिक्त, रिक्त);
}

गणना ggml_status ggml_graph_compute(संरचना ggml_cgraph * cgraph, संरचना ggml_cplan * cplan) {
    ggml_cpu_init();

    GGML_ASSERT(cplan);
    GGML_ASSERT(cplan->n_threads > 0);
    GGML_ASSERT(cplan->work_size == 0 || cplan->work_data != रिक्त);

    पूर्णांक n_threads                               = cplan->n_threads;
    संरचना ggml_threadpool * threadpool = cplan->threadpool;

    bool disposable_threadpool = false;

    जर (threadpool == रिक्त) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
        disposable_threadpool = true;

        संरचना ggml_threadpool_params ttp = ggml_threadpool_params_default(n_threads);
        threadpool = ggml_threadpool_new_impl(&ttp, cgraph, cplan);
    } अन्यथा {
        // Reset some of the parameters that need resetting
        // No worker threads should be accessing the parameters below at this stage
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->current_chunk    = 0;
        threadpool->पात            = -1;
        threadpool->ec               = GGML_STATUS_SUCCESS;
    }

#ifdef GGML_USE_OPENMP
    जर (n_threads > 1) {
        #pragma omp parallel अंक_threads(n_threads)
        {
            #pragma omp single
            {
                // update the number of threads from the actual number of threads that we got from OpenMP
                n_threads = omp_get_अंक_threads();
                atomic_store_explicit(&threadpool->n_graph, n_threads, memory_order_relaxed);
            }

            // Apply thread CPU mask and priority
            पूर्णांक ith = omp_get_thread_अंक();

            ggml_thread_apply_priority(threadpool->prio);
            जर (ggml_thread_cpumask_is_valid(threadpool->workers[ith].cpumask)) {
                ggml_thread_apply_affinity(threadpool->workers[ith].cpumask);
            }
            ggml_graph_compute_thread(&threadpool->workers[ith]);
        }
    } अन्यथा {
        atomic_store_explicit(&threadpool->n_graph, 1, memory_order_relaxed);
        ggml_graph_compute_thread(&threadpool->workers[0]);
    }
#अन्यथा
    जर (n_threads > threadpool->n_threads) {
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

    गणना ggml_status ret = threadpool->ec;

    जर (disposable_threadpool) {
        ggml_threadpool_free(threadpool);
    }

    परत ret;
}

गणना ggml_status ggml_graph_compute_with_ctx(संरचना ggml_context * ctx, संरचना ggml_cgraph * cgraph, पूर्णांक n_threads) {
    संरचना ggml_cplan cplan = ggml_graph_plan(cgraph, n_threads, रिक्त);

    cplan.work_data = (uint8_t *)ggml_new_buffer(ctx, cplan.work_size);

    परत ggml_graph_compute(cgraph, &cplan);
}

शून्य ggml_cpu_fp32_to_fp32(स्थिरांक दशांश * x, दशांश * y, int64_t n) {
    स_नकल(y, x, n * आकार(दशांश));
}

शून्य ggml_cpu_fp32_to_fp16(स्थिरांक दशांश * x, ggml_fp16_t * y, int64_t n) {
    int64_t i = 0;
#जर defined(__F16C__)
#जर defined(__AVX512F__)
    साठी (; i + 15 < n; i += 16) {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        __m256i y_vec = _mm512_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256((__m256i *)(y + i), y_vec);
    }
#endif
    साठी (; i + 7 < n; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m128i y_vec = _mm256_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i *)(y + i), y_vec);
    }
    साठी (; i + 3 < n; i += 4) {
        __m128 x_vec = _mm_loadu_ps(x + i);
        __m128i y_vec = _mm_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storel_epi64((__m128i *)(y + i), y_vec);
    }
#किंवाजर defined(__riscv_zvfh)
    साठी (पूर्णांक vl; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(&x[i], vl);
        vfloat16m1_t vy = __riscv_vfncvt_f_f_w_f16m1(vx, vl);
        __riscv_vse16_v_f16m1((_Float16 *)&y[i], vy, vl);
    }
#किंवाजर defined(__VXE__) || defined(__VXE2__)
    साठी (; i + 7 < n; i += 8) {
        स्थिरांक uint32x4_t v_yl = __lzs_f32cx4_to_f16(vec_xl(0, x + i + 0));
        स्थिरांक uint32x4_t v_yh = __lzs_f32cx4_to_f16(vec_xl(0, x + i + 4));
        vec_xst(vec_pack(v_yl, v_yh), 0, (uint16_t *)(y + i));
    }
#endif
    साठी (; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(x[i]);
    }
}

शून्य ggml_cpu_fp16_to_fp32(स्थिरांक ggml_fp16_t * x, दशांश * y, int64_t n) {
    int64_t i = 0;
#जर defined(__F16C__)
#जर defined(__AVX512F__)
    साठी (; i + 15 < n; i += 16) {
        __m256i x_vec = _mm256_loadu_si256((स्थिरांक __m256i *)(x + i));
        __m512 y_vec = _mm512_cvtph_ps(x_vec);
        _mm512_storeu_ps(y + i, y_vec);
    }
#endif
    साठी (; i + 7 < n; i += 8) {
        __m128i x_vec = _mm_loadu_si128((स्थिरांक __m128i *)(x + i));
        __m256 y_vec = _mm256_cvtph_ps(x_vec);
        _mm256_storeu_ps(y + i, y_vec);
    }
    साठी (; i + 3 < n; i += 4) {
        __m128i x_vec = _mm_loadl_epi64((स्थिरांक __m128i *)(x + i));
        __m128 y_vec = _mm_cvtph_ps(x_vec);
        _mm_storeu_ps(y + i, y_vec);
    }

#किंवाजर defined(__riscv_v_intrinsic) && defined(__riscv_zvfhmin)
    // calculate step size
    स्थिरांक पूर्णांक epr = __riscv_vsetvlmax_e16m2();
    स्थिरांक पूर्णांक step = epr * 2;
    स्थिरांक पूर्णांक np = (n & ~(step - 1));

    // unroll by 2
    साठी (; i < np; i += step) {
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((स्थिरांक _Float16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vfloat16m2_t ax1 = __riscv_vle16_v_f16m2((स्थिरांक _Float16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvt_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    पूर्णांक vl;
    साठी (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((स्थिरांक _Float16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }

#endif

    साठी (; i < n; ++i) {
        y[i] = GGML_CPU_FP16_TO_FP32(x[i]);
    }
}

शून्य ggml_cpu_fp32_to_bf16(स्थिरांक दशांश * x, ggml_bf16_t * y, int64_t n) {
    int64_t i = 0;
    साठी (; i < n; ++i) {
        y[i] = GGML_FP32_TO_BF16(x[i]);
    }
}

शून्य ggml_cpu_fp32_to_i32(स्थिरांक दशांश * x, int32_t * y, int64_t n) {
    int64_t i = 0;
    साठी (; i < n; ++i) {
        y[i] = x[i];
    }
}

शून्य ggml_cpu_bf16_to_fp32(स्थिरांक ggml_bf16_t * x, दशांश * y, int64_t n) {
    int64_t i = 0;
#जर defined(__AVX2__)
#जर defined(__AVX512F__)
    साठी (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i,
                        _mm512_castsi512_ps(
                            _mm512_slli_epi32(
                                _mm512_cvtepu16_epi32(
                                    _mm256_loadu_si256(
                                        (स्थिरांक __m256i *)(x + i))),
                                16)));
    }
#endif
    साठी (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i,
                        _mm256_castsi256_ps(
                            _mm256_slli_epi32(
                                _mm256_cvtepu16_epi32(
                                    _mm_loadu_si128(
                                        (स्थिरांक __m128i *)(x + i))),
                                16)));
    }
#किंवाजर defined(__riscv_v_intrinsic) && defined(__riscv_zvfbfmin)
    // calculate step size
    स्थिरांक पूर्णांक epr = __riscv_vsetvlmax_e16m2();
    स्थिरांक पूर्णांक step = epr * 2;
    स्थिरांक पूर्णांक np = (n & ~(step - 1));

    // unroll by 2
    साठी (; i < np; i += step) {
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((स्थिरांक __bf16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vbfloat16m2_t ax1 = __riscv_vle16_v_bf16m2((स्थिरांक __bf16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    पूर्णांक vl;
    साठी (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((स्थिरांक __bf16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }
#endif
    साठी (; i < n; i++) {
        y[i] = GGML_BF16_TO_FP32(x[i]);
    }
}

पूर्णांक ggml_cpu_has_avx(शून्य) {
#जर defined(__AVX__)
    परत 1;
#अन्यथा
    परत 0;
#endif
}

पूर्णांक ggml_cpu_has_avx_vnni(शून्य) {
#जर defined(__AVXVNNI__)
    परत 1;
#अन्यथा
    परत 0;
#endif
}

पूर्णांक ggml_cpu_has_avx2(शून्य) {
#जर defined(__AVX2__)
    परत 1;
#अन्यथा
    परत 0;
#endif
}

पूर्णांक ggml_cpu_has_avx512(शून्य) {
#जर defined(__AVX512F__)
    परत 1;
#अन्यथा
    परत 0;
#endif
}

पूर्णांक ggml_cpu_has_avx512_vbmi(शून्य) {
#जर defined(__AVX512VBMI__)
    परत 1;
#अन्यथा
    परत 0;
#endif
}

पूर्णांक ggml_cpu_has_avx512_vnni(शून्य) {
#जर defined(__AVX512VNNI__)
    परत 1;
#अन्यथा
    परत 0;
#endif
}

पूर्णांक ggml_cpu_has_avx512_bf16(शून्य) {
#जर defined(__AVX512BF16__)
    परत 1;
#अन्यथा
    परत 0;
#endif
}

पूर्णांक ggml_cpu_has_amx_int8(शून्य) {
#जर defined(__AMX_INT8__)
    परत 1;
#अन्यथा
    परत 0;
#endif
}

पूर्णांक ggml_cpu_has_bmi2(शून्य) {
#जर defined(__BMI2__)
    परत 1;
#अन्यथा
    परत 0;
#endif
}

पूर्णांक ggml_cpu_has_fma(शून्य) {
#जर defined(__FMA__)
    परत 1;
#अन्यथा
    परत 0;
#endif
}

पूर्णांक ggml_cpu_has_arm_fma(शून्य) {
#जर defined(__ARM_FEATURE_FMA)
    परत 1;
#अन्यथा
    परत 0;
#endif
}

पूर्णांक ggml_cpu_has_riscv_v(शून्य) {
#जर defined(__riscv_v_intrinsic)
    परत 1;
#अन्यथा
    परत 0;
#endif
}

पूर्णांक ggml_cpu_get_rvv_vlen(शून्य) {
#जर defined(__riscv) && defined(__riscv_v_intrinsic)
    परत ggml_riscv_arch_features.rvv_vlen;
#अन्यथा
    परत 0;
#endif
}

पूर्णांक ggml_cpu_has_f16c(शून्य) {
#जर defined(__F16C__)
    परत 1;
#अन्यथा
    परत 0;
#endif
}

पूर्णांक ggml_cpu_has_fp16_va(शून्य) {
#जर defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    परत 1;
#अन्यथा
    परत 0;
#endif
}

पूर्णांक ggml_cpu_has_wasm_simd(शून्य) {
#जर defined(__wasm_simd128__)
    परत 1;
#अन्यथा
    परत 0;
#endif
}

पूर्णांक ggml_cpu_has_llamafile(शून्य) {
#जर defined(GGML_USE_LLAMAFILE)
    परत 1;
#अन्यथा
    परत 0;
#endif
}

पूर्णांक ggml_cpu_has_sse3(शून्य) {
#जर defined(__SSE3__)
    परत 1;
#अन्यथा
    परत 0;
#endif
}

पूर्णांक ggml_cpu_has_ssse3(शून्य) {
#जर defined(__SSSE3__)
    परत 1;
#अन्यथा
    परत 0;
#endif
}

पूर्णांक ggml_cpu_has_vsx(शून्य) {
#जर defined(__POWER9_VECTOR__)
    परत 1;
#अन्यथा
    परत 0;
#endif
}

पूर्णांक ggml_cpu_has_vxe(शून्य) {
#जर defined(__VXE__) || defined(__VXE2__)
    परत 1;
#अन्यथा
    परत 0;
#endif
}

पूर्णांक ggml_cpu_has_neon(शून्य) {
#जर defined(__ARM_ARCH) && defined(__ARM_NEON)
    परत 1;
#अन्यथा
    परत 0;
#endif
}

पूर्णांक ggml_cpu_has_dotprod(शून्य) {
#जर defined(__ARM_ARCH) && defined(__ARM_FEATURE_DOTPROD)
    परत 1;
#अन्यथा
    परत 0;
#endif
}

पूर्णांक ggml_cpu_has_sve(शून्य) {
#जर defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    परत 1;
#अन्यथा
    परत 0;
#endif
}

पूर्णांक ggml_cpu_has_matmul_int8(शून्य) {
#जर defined(__ARM_ARCH) && defined(__ARM_FEATURE_MATMUL_INT8)
    परत 1;
#अन्यथा
    परत 0;
#endif
}

पूर्णांक ggml_cpu_get_sve_cnt(शून्य) {
#जर defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    परत ggml_arm_arch_features.sve_cnt;
#अन्यथा
    परत 0;
#endif
}

पूर्णांक ggml_cpu_has_sme(शून्य) {
#जर defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME)
    परत 1;
#अन्यथा
    परत 0;
#endif
}

पूर्णांक ggml_cpu_has_sme2(शून्य) {
#जर defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME2)
    परत 1;
#अन्यथा
    परत 0;
#endif
}

शून्य ggml_cpu_init(शून्य) {
    // needed to initialize ggml_time
    {
        संरचना ggml_init_params params = { 0, रिक्त, false };
        संरचना ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    ggml_critical_section_start();

    स्थिर bool is_first_call = true;

    जर (is_first_call) {
        // initialize GELU, Quick GELU, SILU and EXP F32 tables
        {
            स्थिरांक uint64_t t_start = ggml_time_us(); UNUSED(t_start);

            साठी (पूर्णांक i = 0; i < (1 << 16); ++i) {
                संघ {
                    uint16_t u16;
                    ggml_fp16_t fp16;
                } u = {i};
                दशांश f = GGML_COMPUTE_FP16_TO_FP32(u.fp16);
                ggml_table_f32_f16[i] = f;
                ggml_table_gelu_f16[i] = GGML_CPU_FP32_TO_FP16(ggml_gelu_f32(f));
                ggml_table_gelu_quick_f16[i] = GGML_CPU_FP32_TO_FP16(ggml_gelu_quick_f32(f));
            }

            // initialize E8M0 half table (256 entries)
            साठी (पूर्णांक i = 0; i < (1 << 8); ++i) {
                ggml_table_f32_e8m0_half[i] = GGML_E8M0_TO_FP32_HALF(i);
            }

            // initialize UE4M3 table (256 entries)
            साठी (पूर्णांक i = 0; i < (1 << 8); ++i) {
                ggml_table_f32_ue4m3[i] = ggml_ue4m3_to_fp32(i);
            }

            स्थिरांक uint64_t t_end = ggml_time_us(); UNUSED(t_end);

            GGML_PRINT_DEBUG("%s: GELU, Quick GELU, SILU and EXP tables initialized in %f ms\n", __func__, (t_end - t_start)/1000.0);

#ifdef GGML_USE_OPENMP
            //if (!getenv("OMP_WAIT_POLICY")) {
            //    // set the wait policy to active, so that OpenMP threads don't sleep
            //    setenv("OMP_WAIT_POLICY", "active", 0)
            //}

            जर (!दो_पर्या("KMP_BLOCKTIME")) {
                // set the time to wait before sleeping a thread
                // this is less aggressive than setting the wait policy to active, but should achieve similar results in most cases
#ifdef _WIN32
                _putenv_s("KMP_BLOCKTIME", "200"); // 200ms
#अन्यथा
                setenv("KMP_BLOCKTIME", "200", 0); // 200ms
#endif
            }
#endif
        }

#जर defined(__ARM_ARCH)
        ggml_init_arm_arch_features();
#endif

#जर defined(__riscv)
        ggml_init_riscv_arch_features();
#endif

        {
            स्थिरांक अक्षर * env = दो_पर्या("GGML_CPU_DISABLE_FUSION");
            ggml_cpu_disable_fusion = (env != रिक्त && म_से_प(env) == 1);
        }

        is_first_call = false;
    }

    ggml_critical_section_end();
}
