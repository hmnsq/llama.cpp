#घोषणा _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#घोषणा _USE_MATH_DEFINES // For M_PI on MSVC

#समावेश "ggml-backend-impl.h"
#समावेश "ggml-backend.h"
#समावेश "traits.h"
#समावेश "ggml-cpu-impl.h"
#समावेश "ggml-impl.h"
#समावेश "quants.h"
#समावेश "ggml-threading.h"
#समावेश "unary-ops.h"
#समावेश "binary-ops.h"
#समावेश "vec.h"
#समावेश "ops.h"
#समावेश "ggml.h"
#समावेश "common.h"
#समावेश "tiled/tiled.h"

#अगर defined(_MSC_VER) || defined(__MINGW32__)
#समावेश <malloc.h> // using malloc.h with MSC/MINGW
#या_अगर !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#समावेश <alloca.h>
#पूर्ण_अगर

#समावेश <assert.h>
#समावेश <errno.h>
#समावेश <time.h>
#समावेश <math.h>
#समावेश <stdlib.h>
#समावेश <string.h>
#समावेश <stdint.h>
#समावेश <inttypes.h>
#समावेश <stdio.h>
#समावेश <float.h>
#समावेश <limits.h>
#समावेश <stdarg.h>
#समावेश <signal.h>
#अगर defined(__gnu_linux__)
#समावेश <syscall.h>
#पूर्ण_अगर

#अगर_घोषित GGML_USE_OPENMP
#समावेश <omp.h>
#पूर्ण_अगर

#अगर defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_MATMUL_INT8)
#अघोषित GGML_USE_LLAMAFILE
#पूर्ण_अगर

#अगर_घोषित GGML_USE_LLAMAFILE
#समावेश "llamafile/sgemm.h"
#पूर्ण_अगर

#अगर_घोषित GGML_USE_CPU_RISCV64_SPACEMIT
#    include "spacemit/ime.h"
#पूर्ण_अगर

// Note: once we move threading into a separate C++ file
// will use std::hardware_destructive_interference_size instead of hardcoding it here
// and we'll use C++ attribute syntax.
#घोषणा GGML_CACHE_पंक्ति  64

#अगर defined(__clang__) || defined(__GNUC__)
#घोषणा GGML_CACHE_ALIGN __attribute__((aligned(GGML_CACHE_पंक्ति)))
#पूर्ण_अगर

#अगर defined(__has_feature)
#अगर __has_feature(thread_sanitizer)
#घोषणा GGML_TSAN_ENABLED 1
#पूर्ण_अगर
#अन्यथा  // __has_feature
#अगर defined(__SANITIZE_THREAD__)
#घोषणा GGML_TSAN_ENABLED 1
#पूर्ण_अगर
#पूर्ण_अगर // __has_feature

#घोषणा UNUSED GGML_UNUSED
#घोषणा SWAP(x, y, T) करो { T SWAP = x; (x) = y; (y) = SWAP; } जबतक (0)

// precomputed f32 table for f16 (256 KB) (simd-mappings.h)
भग्न ggml_table_f32_f16[1 << 16];

// precomputed f32 table for e8m0 half (1 KB) (simd-mappings.h)
भग्न ggml_table_f32_e8m0_half[1 << 8];

// precomputed f32 table for ue4m3 (1 KB) (simd-mappings.h)
भग्न ggml_table_f32_ue4m3[1 << 8];

#अगर defined(__ARM_ARCH)
काष्ठा ggml_arm_arch_features_type {
    पूर्णांक sve_cnt;
} ggml_arm_arch_features = { 0 };
#पूर्ण_अगर

#अगर defined(__riscv)
काष्ठा ggml_riscv_arch_features_type {
    पूर्णांक rvv_vlen;
} ggml_riscv_arch_features = { 0 };
#पूर्ण_अगर

#अगर defined(_WIN32)

#घोषणा WIN32_LEAN_AND_MEAN
#अगर_अघोषित NOMINMAX
    #घोषणा NOMINMAX
#पूर्ण_अगर
#समावेश <windows.h>

#अगर defined(_MSC_VER) && !defined(__clang__)
#घोषणा GGML_CACHE_ALIGN __declspec(align(GGML_CACHE_पंक्ति))

प्रकार अस्थिर LONG atomic_int;
प्रकार atomic_int atomic_bool;
प्रकार atomic_int atomic_flag;

#घोषणा ATOMIC_FLAG_INIT 0

प्रकार क्रमागत {
    memory_order_relaxed,
    memory_order_consume,
    memory_order_acquire,
    memory_order_release,
    memory_order_acq_rel,
    memory_order_seq_cst
} memory_order;

अटल व्योम atomic_store(atomic_int * ptr, LONG val) {
    InterlockedExchange(ptr, val);
}
अटल व्योम atomic_store_explicit(atomic_int * ptr, LONG val, memory_order mo) {
    // TODO: add support for explicit memory order
    InterlockedExchange(ptr, val);
}
अटल LONG atomic_load(atomic_int * ptr) {
    वापस InterlockedCompareExchange(ptr, 0, 0);
}
अटल LONG atomic_load_explicit(atomic_int * ptr, memory_order mo) {
    // TODO: add support for explicit memory order
    वापस InterlockedCompareExchange(ptr, 0, 0);
}
अटल LONG atomic_fetch_add(atomic_int * ptr, LONG inc) {
    वापस InterlockedExchangeAdd(ptr, inc);
}
अटल LONG atomic_fetch_add_explicit(atomic_int * ptr, LONG inc, memory_order mo) {
    // TODO: add support for explicit memory order
    वापस InterlockedExchangeAdd(ptr, inc);
}
अटल atomic_bool atomic_flag_test_and_set(atomic_flag * ptr) {
    वापस InterlockedExchange(ptr, 1);
}
अटल व्योम atomic_flag_clear(atomic_flag * ptr) {
    InterlockedExchange(ptr, 0);
}
अटल व्योम atomic_thread_fence(memory_order mo) {
    MemoryBarrier();
}
#अन्यथा // clang
#समावेश <stdatomic.h>
#पूर्ण_अगर

प्रकार HANDLE pthread_t;

प्रकार DWORD thread_ret_t;
अटल पूर्णांक pthread_create(pthread_t * out, व्योम * unused, thread_ret_t(*func)(व्योम *), व्योम * arg) {
    (व्योम) unused;
    HANDLE handle = CreateThread(शून्य, 0, (LPTHREAD_START_ROUTINE) func, arg, 0, शून्य);
    अगर (handle == शून्य)
    {
        वापस EAGAIN;
    }

    *out = handle;
    वापस 0;
}

अटल पूर्णांक pthread_join(pthread_t thread, व्योम * unused) {
    (व्योम) unused;
    पूर्णांक ret = (पूर्णांक) WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    वापस ret;
}

अटल पूर्णांक sched_yield (व्योम) {
    Sleep (0);
    वापस 0;
}
#अन्यथा

#समावेश <pthread.h>
#समावेश <stdatomic.h>
#समावेश <sched.h>
#अगर defined(__FreeBSD__)
#समावेश <pthread_np.h>
#पूर्ण_अगर

प्रकार व्योम * thread_ret_t;

#समावेश <sys/types.h>
#समावेश <sys/stat.h>
#समावेश <unistd.h>

#पूर्ण_अगर

प्रकार pthread_t ggml_thread_t;

#घोषणा GGML_THREADPOOL_N_THREADS_MASK (0xffffU)
#घोषणा GGML_THREADPOOL_N_THREADS_BITS (16)

#अगर defined(__APPLE__)
#समावेश <unistd.h>
#समावेश <mach/mach.h>
#समावेश <TargetConditionals.h>
#पूर्ण_अगर

अटल स्थिर काष्ठा ggml_type_traits_cpu type_traits_cpu[GGML_TYPE_गिनती] = {
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
#अगर defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#अन्यथा
        .nrows                    = 1,
#पूर्ण_अगर
    },
    [GGML_TYPE_Q4_1] = {
        .from_float               = quantize_row_q4_1,
        .vec_dot                  = ggml_vec_dot_q4_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
#अगर defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#अन्यथा
        .nrows                    = 1,
#पूर्ण_अगर
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
#अगर defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#अन्यथा
        .nrows                    = 1,
#पूर्ण_अगर
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
#अगर defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#अन्यथा
        .nrows                    = 1,
#पूर्ण_अगर
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
#अगर defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#अन्यथा
        .nrows                    = 1,
#पूर्ण_अगर
    },
    [GGML_TYPE_IQ2_XXS] = {
        .from_float               = शून्य,
        .vec_dot                  = ggml_vec_dot_iq2_xxs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_XS] = {
        .from_float               = शून्य,
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
        .from_float               = शून्य,
        .vec_dot                  = ggml_vec_dot_iq1_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_M] = {
        .from_float               = शून्य,
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

स्थिर काष्ठा ggml_type_traits_cpu * ggml_get_type_traits_cpu(क्रमागत ggml_type type) {
    वापस &type_traits_cpu[type];
}

//
// Threading defs
//

प्रकार pthread_t          ggml_thread_t;

#अगर defined(_WIN32)

प्रकार CONDITION_VARIABLE ggml_cond_t;
प्रकार SRWLOCK            ggml_mutex_t;

#घोषणा ggml_mutex_init(m)   InitializeSRWLock(m)
#घोषणा ggml_mutex_destroy(m)
#घोषणा ggml_mutex_lock(m)   AcquireSRWLockExclusive(m)
#घोषणा ggml_mutex_unlock(m) ReleaseSRWLockExclusive(m)
#घोषणा ggml_mutex_lock_shared(m)   AcquireSRWLockShared(m)
#घोषणा ggml_mutex_unlock_shared(m) ReleaseSRWLockShared(m)

#घोषणा ggml_cond_init(c)    InitializeConditionVariable(c)
#घोषणा ggml_cond_destroy(c)
#घोषणा ggml_cond_wait(c, m) SleepConditionVariableSRW(c, m, INFINITE, CONDITION_VARIABLE_LOCKMODE_SHARED)
#घोषणा ggml_cond_broadcast(c) WakeAllConditionVariable(c)

#घोषणा ggml_thread_create pthread_create
#घोषणा ggml_thread_join   pthread_join

#अन्यथा

प्रकार pthread_cond_t     ggml_cond_t;
प्रकार pthread_mutex_t    ggml_mutex_t;

#घोषणा ggml_mutex_init(m)          pthread_mutex_init(m, शून्य)
#घोषणा ggml_mutex_destroy(m)       pthread_mutex_destroy(m)
#घोषणा ggml_mutex_lock(m)          pthread_mutex_lock(m)
#घोषणा ggml_mutex_unlock(m)        pthread_mutex_unlock(m)
#घोषणा ggml_mutex_lock_shared(m)   pthread_mutex_lock(m)
#घोषणा ggml_mutex_unlock_shared(m) pthread_mutex_unlock(m)

#घोषणा ggml_lock_init(x)    UNUSED(x)
#घोषणा ggml_lock_destroy(x) UNUSED(x)
#अगर defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64) && !defined(_M_ARM64EC))
#घोषणा ggml_lock_lock(x)    _mm_pause()
#अन्यथा
#घोषणा ggml_lock_lock(x)    UNUSED(x)
#पूर्ण_अगर
#घोषणा ggml_lock_unlock(x)  UNUSED(x)

#घोषणा GGML_LOCK_INITIALIZER 0
#घोषणा ggml_cond_init(c)      pthread_cond_init(c, शून्य)
#घोषणा ggml_cond_destroy(c)   pthread_cond_destroy(c)
#घोषणा ggml_cond_wait(c, m)   pthread_cond_wait(c, m)
#घोषणा ggml_cond_broadcast(c) pthread_cond_broadcast(c)

#घोषणा ggml_thread_create pthread_create
#घोषणा ggml_thread_join   pthread_join

#पूर्ण_अगर

// Threadpool def
काष्ठा ggml_threadpool {
    ggml_mutex_t mutex;       // mutex for cond.var
    ggml_cond_t  cond;        // cond.var for waiting for new work

    काष्ठा ggml_cgraph * cgraph;
    काष्ठा ggml_cplan  * cplan;

    // synchronization primitives
    atomic_int n_graph;       // updated when there is work to be done (i.e each graph) holds graph and active thread counts.
    atomic_int GGML_CACHE_ALIGN n_barrier;
    atomic_int GGML_CACHE_ALIGN n_barrier_passed;
    atomic_int GGML_CACHE_ALIGN current_chunk; // currently processing chunk during Mat_Mul, shared between all the threads.

    // these are atomic as an annotation for thread-sanitizer
    atomic_bool stop;         // Used for stopping the threadpool altogether
    atomic_bool छोड़ो;        // Used for pausing the threadpool or individual threads
    atomic_int  पात;        // Used for aborting processing of a graph

    काष्ठा ggml_compute_state * workers;   // per thread state
    पूर्णांक          n_threads;   // Number of threads in the pool
    int32_t      prio;        // Scheduling priority
    uint32_t     poll;        // Polling level (0 - no polling)

    क्रमागत ggml_status ec;
};

// Per-thread state
काष्ठा ggml_compute_state {
#अगर_अघोषित GGML_USE_OPENMP
    ggml_thread_t thrd;
    पूर्णांक  last_graph;
    bool pending;
#पूर्ण_अगर
    bool cpumask[GGML_MAX_N_THREADS];
    काष्ठा ggml_threadpool * threadpool;
    पूर्णांक ith;
};

// Helpers for polling loops
#अगर defined(__aarch64__) && ( defined(__clang__) || defined(__GNUC__) )
अटल अंतरभूत व्योम ggml_thread_cpu_relax(व्योम) {
    __asm__ अस्थिर("yield" ::: "memory");
}
#या_अगर defined(__x86_64__)
अटल अंतरभूत व्योम ggml_thread_cpu_relax(व्योम) {
    _mm_pause();
}
#या_अगर defined(__riscv)
अटल अंतरभूत व्योम ggml_thread_cpu_relax(व्योम) {
    #अगर_घोषित __riscv_zihintpause
        __asm__ __volatile__ ("pause");
    #अन्यथा
        /* Encoding of the pause instruction */
        __asm__ __volatile__ (".4byte 0x100000F");
    #पूर्ण_अगर
}
#अन्यथा
अटल अंतरभूत व्योम ggml_thread_cpu_relax(व्योम) {;}
#पूर्ण_अगर

//
// NUMA support
//

#घोषणा GGML_NUMA_MAX_NODES 8
#घोषणा GGML_NUMA_MAX_CPUS 512

काष्ठा ggml_numa_गाँठ {
    uint32_t cpus[GGML_NUMA_MAX_CPUS]; // hardware threads on this node
    uint32_t n_cpus;
};

काष्ठा ggml_numa_nodes {
    क्रमागत ggml_numa_strategy numa_strategy;
    काष्ठा ggml_numa_गाँठ nodes[GGML_NUMA_MAX_NODES];
    uint32_t n_nodes;
    uint32_t योग_cpus; // hardware threads on system
    uint32_t current_गाँठ; // node on which main process is execting
#अगर defined(__gnu_linux__)
    cpu_set_t cpuset; // cpuset from numactl
#अन्यथा
    uint32_t cpuset; // no NUMA support outside of Linux at this time. Use a portable datatype
#पूर्ण_अगर
};

//
// ggml state
//

काष्ठा ggml_state {
    काष्ठा ggml_numa_nodes numa;
};

अटल काष्ठा ggml_state g_state = {0};

व्योम ggml_barrier(काष्ठा ggml_threadpool * tp) {
    पूर्णांक n_threads = atomic_load_explicit(&tp->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK;
    अगर (n_threads == 1) {
        वापस;
    }

#अगर_घोषित GGML_USE_OPENMP
    #आशय omp barrier
#अन्यथा
    पूर्णांक n_passed = atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed);

    // enter barrier (full seq-cst fence)
    पूर्णांक n_barrier = atomic_fetch_add_explicit(&tp->n_barrier, 1, memory_order_seq_cst);

    अगर (n_barrier == (n_threads - 1)) {
        // last thread
        atomic_store_explicit(&tp->n_barrier, 0, memory_order_relaxed);

        // exit barrier (full seq-cst fence)
        atomic_fetch_add_explicit(&tp->n_barrier_passed, 1, memory_order_seq_cst);
        वापस;
    }

    // wait for other threads
    जबतक (atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed) == n_passed) {
        ggml_thread_cpu_relax();
    }

    // exit barrier (full seq-cst fence)
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #अगर_घोषित GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&tp->n_barrier_passed, 0, memory_order_seq_cst);
    #अन्यथा
    atomic_thread_fence(memory_order_seq_cst);
    #पूर्ण_अगर
#पूर्ण_अगर
}

व्योम ggml_threadpool_chunk_set(काष्ठा ggml_threadpool * tp, पूर्णांक मूल्य) {
    atomic_store_explicit(&tp->current_chunk, मूल्य, memory_order_relaxed);
}

पूर्णांक ggml_threadpool_chunk_add(काष्ठा ggml_threadpool * tp, पूर्णांक मूल्य) {
    वापस atomic_fetch_add_explicit(&tp->current_chunk, मूल्य, memory_order_relaxed);
}

#अगर defined(__gnu_linux__)
अटल cpu_set_t ggml_get_numa_affinity(व्योम) {
    cpu_set_t cpuset;
    pthread_t thread;
    thread = pthread_self();
    CPU_ZERO(&cpuset);
    pthread_getaffinity_np(thread, माप(cpu_set_t), &cpuset);
    वापस cpuset;
}
#अन्यथा
अटल uint32_t ggml_get_numa_affinity(व्योम) {
    वापस 0; // no NUMA support
}
#पूर्ण_अगर

व्योम ggml_numa_init(क्रमागत ggml_numa_strategy numa_flag) {
    अगर (g_state.numa.n_nodes > 0) {
        ख_लिखो(मानक_त्रुटि, "ggml_numa_init: NUMA already initialized\n");

        वापस;
    }

#अगर defined(__gnu_linux__)
    काष्ठा stat st;
    अक्षर path[256];
    पूर्णांक rv;

    // set numa scheme
    g_state.numa.numa_strategy = numa_flag;

    GGML_PRINT_DEBUG("numa strategy %u\n",g_state.numa.numa_strategy);

    g_state.numa.cpuset = ggml_get_numa_affinity();

    // enumerate nodes
    जबतक (g_state.numa.n_nodes < GGML_NUMA_MAX_NODES) {
        rv = snprintf(path, माप(path), "/sys/devices/system/node/node%u", g_state.numa.n_nodes);
        GGML_ASSERT(rv > 0 && (अचिन्हित)rv < माप(path));
        अगर (stat(path, &st) != 0) { अवरोध; }
        ++g_state.numa.n_nodes;
    }

    // enumerate CPUs
    जबतक (g_state.numa.योग_cpus < GGML_NUMA_MAX_CPUS) {
        rv = snprintf(path, माप(path), "/sys/devices/system/cpu/cpu%u", g_state.numa.योग_cpus);
        GGML_ASSERT(rv > 0 && (अचिन्हित)rv < माप(path));
        अगर (stat(path, &st) != 0) { अवरोध; }
        ++g_state.numa.योग_cpus;
    }

    GGML_PRINT_DEBUG("found %u numa nodes, %u CPUs\n", g_state.numa.n_nodes, g_state.numa.योग_cpus);

    // figure out which node we're on
    uint current_cpu;
    पूर्णांक getcpu_ret = 0;
#अगर __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ > 33) || defined(__COSMOPOLITAN__)
    getcpu_ret = getcpu(&current_cpu, &g_state.numa.current_गाँठ);
#अन्यथा
    // old glibc doesn't have a wrapper for this call. Fall back on direct syscall
#   अगर !defined(SYS_getcpu) && defined(SYS_get_cpu)
#       define SYS_getcpu SYS_get_cpu // some older glibc versions use this name
#   endif
    getcpu_ret = syscall(SYS_getcpu, &current_cpu, &g_state.numa.current_गाँठ);
#पूर्ण_अगर

    अगर (g_state.numa.n_nodes < 1 || g_state.numa.योग_cpus < 1 || getcpu_ret != 0) {
        g_state.numa.n_nodes = 0;
        वापस;
    }

    GGML_PRINT_DEBUG("found our process on numa node %u, CPU %u\n", g_state.numa.current_गाँठ, current_cpu);

    क्रम (uint32_t n = 0; n < g_state.numa.n_nodes; ++n) {
        काष्ठा ggml_numa_गाँठ * गाँठ = &g_state.numa.nodes[n];
        GGML_PRINT_DEBUG("CPUs on node %u:", n);
        गाँठ->n_cpus = 0;
        क्रम (uint32_t c = 0; c < g_state.numa.योग_cpus; ++c) {
            rv = snprintf(path, माप(path), "/sys/devices/system/node/node%u/cpu%u", n, c);
            GGML_ASSERT(rv > 0 && (अचिन्हित)rv < माप(path));
            अगर (stat(path, &st) == 0) {
                गाँठ->cpus[गाँठ->n_cpus++] = c;
                GGML_PRINT_DEBUG(" %u", c);
            }
        }
        GGML_PRINT_DEBUG("\n");
    }

    अगर (ggml_is_numa()) {
        खाता *fptr = ख_खोलो("/proc/sys/kernel/numa_balancing", "r");
        अगर (fptr != शून्य) {
            अक्षर buf[42];
            अगर (ख_माला_लो(buf, माप(buf), fptr) && म_भेदन(buf, "0\n", माप(buf)) != 0) {
                GGML_LOG_WARN("/proc/sys/kernel/numa_balancing is enabled, this has been observed to impair performance\n");
            }
            ख_बंद(fptr);
        }
    }
#अन्यथा
    UNUSED(numa_flag);
    // TODO
#पूर्ण_अगर
}

bool ggml_is_numa(व्योम) {
    वापस g_state.numa.n_nodes > 1;
}

#अगर defined(__ARM_ARCH)
#अगर defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
#समावेश <arm_sve.h>
अटल व्योम ggml_init_arm_arch_features(व्योम) {
    ggml_arm_arch_features.sve_cnt = svcntb();
}
#अन्यथा
अटल व्योम ggml_init_arm_arch_features(व्योम) {}
#पूर्ण_अगर
#पूर्ण_अगर // __ARM_ARCH

#अगर defined(__riscv) && defined(__riscv_v_intrinsic)
#समावेश <riscv_vector.h>
अटल व्योम ggml_init_riscv_arch_features(व्योम) {
    ggml_riscv_arch_features.rvv_vlen = __riscv_vlenb();
}
#अन्यथा
अटल व्योम ggml_init_riscv_arch_features(व्योम) {}
#पूर्ण_अगर

काष्ठा ggml_tensor * ggml_new_i32(काष्ठा ggml_context * ctx, int32_t मूल्य) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

    ggml_set_i32(परिणाम, मूल्य);

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_new_f32(काष्ठा ggml_context * ctx, भग्न मूल्य) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    ggml_set_f32(परिणाम, मूल्य);

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_set_i32 (काष्ठा ggml_tensor * tensor, int32_t मूल्य) {
    स्थिर पूर्णांक n     = ggml_nrows(tensor);
    स्थिर पूर्णांक nc    = tensor->ne[0];
    स्थिर माप_प्रकार n1 = tensor->nb[1];

    अक्षर * स्थिर तथ्य = tensor->तथ्य;

    चयन (tensor->type) {
        हाल GGML_TYPE_I8:
            {
                निश्चित(tensor->nb[0] == माप(int8_t));
                क्रम (पूर्णांक i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(तथ्य + i*n1), मूल्य);
                }
            } अवरोध;
        हाल GGML_TYPE_I16:
            {
                निश्चित(tensor->nb[0] == माप(int16_t));
                क्रम (पूर्णांक i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(तथ्य + i*n1), मूल्य);
                }
            } अवरोध;
        हाल GGML_TYPE_I32:
            {
                निश्चित(tensor->nb[0] == माप(int32_t));
                क्रम (पूर्णांक i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(तथ्य + i*n1), मूल्य);
                }
            } अवरोध;
        हाल GGML_TYPE_F16:
            {
                निश्चित(tensor->nb[0] == माप(ggml_fp16_t));
                क्रम (पूर्णांक i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(तथ्य + i*n1), GGML_CPU_FP32_TO_FP16(मूल्य));
                }
            } अवरोध;
        हाल GGML_TYPE_BF16:
            {
                निश्चित(tensor->nb[0] == माप(ggml_fp16_t));
                क्रम (पूर्णांक i = 0; i < n; i++) {
                    ggml_vec_set_bf16(nc, (ggml_bf16_t *)(तथ्य + i*n1), GGML_FP32_TO_BF16(मूल्य));
                }
            } अवरोध;
        हाल GGML_TYPE_F32:
            {
                निश्चित(tensor->nb[0] == माप(भग्न));
                क्रम (पूर्णांक i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (भग्न *)(तथ्य + i*n1), मूल्य);
                }
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }

    वापस tensor;
}

काष्ठा ggml_tensor * ggml_set_f32(काष्ठा ggml_tensor * tensor, भग्न मूल्य) {
    स्थिर पूर्णांक n     = ggml_nrows(tensor);
    स्थिर पूर्णांक nc    = tensor->ne[0];
    स्थिर माप_प्रकार n1 = tensor->nb[1];

    अक्षर * स्थिर तथ्य = tensor->तथ्य;

    चयन (tensor->type) {
        हाल GGML_TYPE_I8:
            {
                निश्चित(tensor->nb[0] == माप(int8_t));
                क्रम (पूर्णांक i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(तथ्य + i*n1), मूल्य);
                }
            } अवरोध;
        हाल GGML_TYPE_I16:
            {
                निश्चित(tensor->nb[0] == माप(int16_t));
                क्रम (पूर्णांक i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(तथ्य + i*n1), मूल्य);
                }
            } अवरोध;
        हाल GGML_TYPE_I32:
            {
                निश्चित(tensor->nb[0] == माप(int32_t));
                क्रम (पूर्णांक i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(तथ्य + i*n1), मूल्य);
                }
            } अवरोध;
        हाल GGML_TYPE_F16:
            {
                निश्चित(tensor->nb[0] == माप(ggml_fp16_t));
                क्रम (पूर्णांक i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(तथ्य + i*n1), GGML_CPU_FP32_TO_FP16(मूल्य));
                }
            } अवरोध;
        हाल GGML_TYPE_BF16:
            {
                निश्चित(tensor->nb[0] == माप(ggml_bf16_t));
                क्रम (पूर्णांक i = 0; i < n; i++) {
                    ggml_vec_set_bf16(nc, (ggml_bf16_t *)(तथ्य + i*n1), GGML_FP32_TO_BF16(मूल्य));
                }
            } अवरोध;
        हाल GGML_TYPE_F32:
            {
                निश्चित(tensor->nb[0] == माप(भग्न));
                क्रम (पूर्णांक i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (भग्न *)(तथ्य + i*n1), मूल्य);
                }
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }

    वापस tensor;
}

int32_t ggml_get_i32_1d(स्थिर काष्ठा ggml_tensor * tensor, पूर्णांक i) {
    अगर (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_सूचकांक(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        वापस ggml_get_i32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    चयन (tensor->type) {
        हाल GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == माप(int8_t));
                वापस ((int8_t *)(tensor->तथ्य))[i];
            }
        हाल GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == माप(int16_t));
                वापस ((int16_t *)(tensor->तथ्य))[i];
            }
        हाल GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == माप(int32_t));
                वापस ((int32_t *)(tensor->तथ्य))[i];
            }
        हाल GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == माप(ggml_fp16_t));
                वापस GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *)(tensor->तथ्य))[i]);
            }
        हाल GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == माप(ggml_bf16_t));
                वापस GGML_BF16_TO_FP32(((ggml_bf16_t *)(tensor->तथ्य))[i]);
            }
        हाल GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == माप(भग्न));
                वापस ((भग्न *)(tensor->तथ्य))[i];
            }
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

व्योम ggml_set_i32_1d(स्थिर काष्ठा ggml_tensor * tensor, पूर्णांक i, int32_t मूल्य) {
    अगर (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_सूचकांक(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_set_i32_nd(tensor, id[0], id[1], id[2], id[3], मूल्य);
        वापस;
    }
    चयन (tensor->type) {
        हाल GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == माप(int8_t));
                ((int8_t *)(tensor->तथ्य))[i] = मूल्य;
            } अवरोध;
        हाल GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == माप(int16_t));
                ((int16_t *)(tensor->तथ्य))[i] = मूल्य;
            } अवरोध;
        हाल GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == माप(int32_t));
                ((int32_t *)(tensor->तथ्य))[i] = मूल्य;
            } अवरोध;
        हाल GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == माप(ggml_fp16_t));
                ((ggml_fp16_t *)(tensor->तथ्य))[i] = GGML_CPU_FP32_TO_FP16(मूल्य);
            } अवरोध;
        हाल GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == माप(ggml_bf16_t));
                ((ggml_bf16_t *)(tensor->तथ्य))[i] = GGML_FP32_TO_BF16(मूल्य);
            } अवरोध;
        हाल GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == माप(भग्न));
                ((भग्न *)(tensor->तथ्य))[i] = मूल्य;
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

int32_t ggml_get_i32_nd(स्थिर काष्ठा ggml_tensor * tensor, पूर्णांक i0, पूर्णांक i1, पूर्णांक i2, पूर्णांक i3) {
    व्योम * तथ्य   = (अक्षर *) tensor->तथ्य + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    चयन (tensor->type) {
        हाल GGML_TYPE_I8:
            वापस ((int8_t *) तथ्य)[0];
        हाल GGML_TYPE_I16:
            वापस ((int16_t *) तथ्य)[0];
        हाल GGML_TYPE_I32:
            वापस ((int32_t *) तथ्य)[0];
        हाल GGML_TYPE_F16:
            वापस GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *) तथ्य)[0]);
        हाल GGML_TYPE_BF16:
            वापस GGML_BF16_TO_FP32(((ggml_bf16_t *) तथ्य)[0]);
        हाल GGML_TYPE_F32:
            वापस ((भग्न *) तथ्य)[0];
        शेष:
            GGML_ABORT("fatal error");
    }
}

व्योम ggml_set_i32_nd(स्थिर काष्ठा ggml_tensor * tensor, पूर्णांक i0, पूर्णांक i1, पूर्णांक i2, पूर्णांक i3, int32_t मूल्य) {
    व्योम * तथ्य   = (अक्षर *) tensor->तथ्य + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    चयन (tensor->type) {
        हाल GGML_TYPE_I8:
            {
                ((int8_t *)(तथ्य))[0] = मूल्य;
            } अवरोध;
        हाल GGML_TYPE_I16:
            {
                ((int16_t *)(तथ्य))[0] = मूल्य;
            } अवरोध;
        हाल GGML_TYPE_I32:
            {
                ((int32_t *)(तथ्य))[0] = मूल्य;
            } अवरोध;
        हाल GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(तथ्य))[0] = GGML_CPU_FP32_TO_FP16(मूल्य);
            } अवरोध;
        हाल GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(तथ्य))[0] = GGML_FP32_TO_BF16(मूल्य);
            } अवरोध;
        हाल GGML_TYPE_F32:
            {
                ((भग्न *)(तथ्य))[0] = मूल्य;
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

भग्न ggml_get_f32_1d(स्थिर काष्ठा ggml_tensor * tensor, पूर्णांक i) {
    अगर (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_सूचकांक(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        वापस ggml_get_f32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    चयन (tensor->type) {
        हाल GGML_TYPE_I8:
            {
                वापस ((int8_t *)(tensor->तथ्य))[i];
            }
        हाल GGML_TYPE_I16:
            {
                वापस ((int16_t *)(tensor->तथ्य))[i];
            }
        हाल GGML_TYPE_I32:
            {
                वापस ((int32_t *)(tensor->तथ्य))[i];
            }
        हाल GGML_TYPE_F16:
            {
                वापस GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *)(tensor->तथ्य))[i]);
            }
        हाल GGML_TYPE_BF16:
            {
                वापस GGML_BF16_TO_FP32(((ggml_bf16_t *)(tensor->तथ्य))[i]);
            }
        हाल GGML_TYPE_F32:
            {
                वापस ((भग्न *)(tensor->तथ्य))[i];
            }
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

व्योम ggml_set_f32_1d(स्थिर काष्ठा ggml_tensor * tensor, पूर्णांक i, भग्न मूल्य) {
    अगर (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_सूचकांक(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_set_f32_nd(tensor, id[0], id[1], id[2], id[3], मूल्य);
        वापस;
    }
    चयन (tensor->type) {
        हाल GGML_TYPE_I8:
            {
                ((int8_t *)(tensor->तथ्य))[i] = मूल्य;
            } अवरोध;
        हाल GGML_TYPE_I16:
            {
                ((int16_t *)(tensor->तथ्य))[i] = मूल्य;
            } अवरोध;
        हाल GGML_TYPE_I32:
            {
                ((int32_t *)(tensor->तथ्य))[i] = मूल्य;
            } अवरोध;
        हाल GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(tensor->तथ्य))[i] = GGML_CPU_FP32_TO_FP16(मूल्य);
            } अवरोध;
        हाल GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(tensor->तथ्य))[i] = GGML_FP32_TO_BF16(मूल्य);
            } अवरोध;
        हाल GGML_TYPE_F32:
            {
                ((भग्न *)(tensor->तथ्य))[i] = मूल्य;
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

भग्न ggml_get_f32_nd(स्थिर काष्ठा ggml_tensor * tensor, पूर्णांक i0, पूर्णांक i1, पूर्णांक i2, पूर्णांक i3) {
    व्योम * तथ्य   = (अक्षर *) tensor->तथ्य + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    चयन (tensor->type) {
        हाल GGML_TYPE_I8:
            वापस ((int8_t *) तथ्य)[0];
        हाल GGML_TYPE_I16:
            वापस ((int16_t *) तथ्य)[0];
        हाल GGML_TYPE_I32:
            वापस ((int32_t *) तथ्य)[0];
        हाल GGML_TYPE_F16:
            वापस GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *) तथ्य)[0]);
        हाल GGML_TYPE_BF16:
            वापस GGML_BF16_TO_FP32(((ggml_bf16_t *) तथ्य)[0]);
        हाल GGML_TYPE_F32:
            वापस ((भग्न *) तथ्य)[0];
        शेष:
            GGML_ABORT("fatal error");
    }
}

व्योम ggml_set_f32_nd(स्थिर काष्ठा ggml_tensor * tensor, पूर्णांक i0, पूर्णांक i1, पूर्णांक i2, पूर्णांक i3, भग्न मूल्य) {
    व्योम * तथ्य   = (अक्षर *) tensor->तथ्य + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    चयन (tensor->type) {
        हाल GGML_TYPE_I8:
            {
                ((int8_t *)(तथ्य))[0] = मूल्य;
            } अवरोध;
        हाल GGML_TYPE_I16:
            {
                ((int16_t *)(तथ्य))[0] = मूल्य;
            } अवरोध;
        हाल GGML_TYPE_I32:
            {
                ((int32_t *)(तथ्य))[0] = मूल्य;
            } अवरोध;
        हाल GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(तथ्य))[0] = GGML_CPU_FP32_TO_FP16(मूल्य);
            } अवरोध;
        हाल GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(तथ्य))[0] = GGML_FP32_TO_BF16(मूल्य);
            } अवरोध;
        हाल GGML_TYPE_F32:
            {
                ((भग्न *)(तथ्य))[0] = मूल्य;
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

////////////////////////////////////////////////////////////////////////////////

// ggml_compute_forward_mul_mat

अटल व्योम ggml_compute_forward_mul_mat_one_chunk(
    स्थिर काष्ठा ggml_compute_params * params,
    काष्ठा ggml_tensor * dst,
    स्थिर क्रमागत ggml_type type,
    स्थिर int64_t अंक_rows_per_vec_dot,
    स्थिर int64_t ir0_start,
    स्थिर int64_t ir0_end,
    स्थिर int64_t ir1_start,
    स्थिर int64_t ir1_end) {

    स्थिर काष्ठा ggml_tensor * src0 = dst->src[0];
    स्थिर काष्ठा ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विआधारी_OP_LOCALS

    स्थिर bool src1_cont = ggml_is_contiguous(src1);

    ggml_vec_dot_t स्थिर vec_dot      = type_traits_cpu[type].vec_dot;
    क्रमागत ggml_type स्थिर vec_dot_type = type_traits_cpu[type].vec_dot_type;

    // broadcast factors
    स्थिर int64_t r2 = ne12 / ne02;
    स्थिर int64_t r3 = ne13 / ne03;

    //printf("ir0_start = %6lld, ir0_end = %6lld, ir1_start = %6lld, ir1_end = %6lld\n", ir0_start, ir0_end, ir1_start, ir1_end);

    // threads with no work simply yield (not sure if it helps)
    अगर (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        वापस;
    }

    स्थिर व्योम * wdata = (src1->type == vec_dot_type) ? src1->तथ्य : params->wdata;
    स्थिर माप_प्रकार row_आकार = ggml_row_आकार(vec_dot_type, ne10);

    निश्चित(ne12 % ne02 == 0);
    निश्चित(ne13 % ne03 == 0);

    // block-tiling attempt
    स्थिर int64_t blck_0 = 16;
    स्थिर int64_t blck_1 = 16;

    स्थिर माप_प्रकार src1_col_stride = src1_cont || src1->type != vec_dot_type ? row_आकार : nb11;

    // attempt to reduce false-sharing (does not seem to make a difference)
    // 16 * 2, accounting for mmla kernels
    भग्न tmp[32];

    क्रम (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        क्रम (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            क्रम (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += अंक_rows_per_vec_dot) {
                स्थिर int64_t i13 = (ir1 / (ne12 * ne1));
                स्थिर int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                स्थिर int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                // broadcast src0 into src1
                स्थिर int64_t i03 = i13 / r3;
                स्थिर int64_t i02 = i12 / r2;

                स्थिर int64_t i1 = i11;
                स्थिर int64_t i2 = i12;
                स्थिर int64_t i3 = i13;

                स्थिर अक्षर * src0_row = (स्थिर अक्षर*)src0->तथ्य + (0 + i02 * nb02 + i03 * nb03);

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                स्थिर अक्षर * src1_col = (स्थिर अक्षर*)wdata +
                    (src1_cont || src1->type != vec_dot_type
                        ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_आकार
                        : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                भग्न * dst_col = (भग्न*)((अक्षर*)dst->तथ्य + (i1 * nb1 + i2 * nb2 + i3 * nb3));

                //for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                //    vec_dot(ne00, &dst_col[ir0], src0_row + ir0*nb01, src1_col);
                //}

                क्रम (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += अंक_rows_per_vec_dot) {
                    vec_dot(ne00, &tmp[ir0 - iir0], (अंक_rows_per_vec_dot > 1 ? 16 : 0), src0_row + ir0 * nb01, (अंक_rows_per_vec_dot > 1 ? nb01 : 0), src1_col, (अंक_rows_per_vec_dot > 1 ? src1_col_stride : 0), अंक_rows_per_vec_dot);
                }

                क्रम (पूर्णांक cn = 0; cn < अंक_rows_per_vec_dot; ++cn) {
                    स_नकल(&dst_col[iir0 + cn * nb1 / nb0], tmp + (cn * 16), (MIN(iir0 + blck_0, ir0_end) - iir0) * माप(भग्न));
                }
            }
        }
    }
}

व्योम ggml_compute_forward_mul_mat(
        स्थिर काष्ठा ggml_compute_params * params,
              काष्ठा ggml_tensor * dst) {

    स्थिर काष्ठा ggml_tensor * src0 = dst->src[0];
    स्थिर काष्ठा ggml_tensor * src1 = dst->src[1];

    स्थिर int32_t hint = ggml_get_op_params_i32(dst, 1);
    अगर (hint == GGML_HINT_SRC0_IS_HADAMARD && !params->use_ref) {
        ggml_compute_forward_fwht(params, dst);
        वापस;
    }

    // If tiled is supported, it will execute the full op here and we return
    अगर (ggml_compute_forward_mul_mat_tiled(params, dst)) {
        वापस;
    }

    GGML_TENSOR_द्विआधारी_OP_LOCALS

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    क्रमागत ggml_type           स्थिर vec_dot_type         = type_traits_cpu[src0->type].vec_dot_type;
    ggml_from_float_t        स्थिर from_float           = type_traits_cpu[vec_dot_type].from_float;
    int64_t                  स्थिर vec_dot_अंक_rows     = type_traits_cpu[src0->type].nrows;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_आकार(src0->type));
    GGML_ASSERT(nb10 == ggml_type_आकार(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == माप(भग्न));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    // TODO: extract to "extra_op"
#अगर GGML_USE_LLAMAFILE
    // broadcast factors
    स्थिर int64_t r2 = ne12 / ne02;
    स्थिर int64_t r3 = ne13 / ne03;

    स्थिर bool src1_cont = ggml_is_contiguous(src1);

    अगर (!params->use_ref && src1_cont) {
        क्रम (int64_t i13 = 0; i13 < ne13; i13++)
            क्रम (int64_t i12 = 0; i12 < ne12; i12++)
                अगर (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_आकार(src0->type),
                                     (स्थिर अक्षर *)src0->तथ्य + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_आकार(src0->type),
                                     (स्थिर अक्षर *)src1->तथ्य + i12*nb12 + i13*nb13,
                                     nb11/ggml_type_आकार(src1->type),
                                     (अक्षर *)dst->तथ्य + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_आकार(dst->type),
                                     src0->type,
                                     src1->type,
                                     dst->type))
                    जाओ UseGgmlGemm1;
        वापस;
    }
UseGgmlGemm1:;
#पूर्ण_अगर

    अगर (src1->type != vec_dot_type) {
        अक्षर * wdata = params->wdata;

        स्थिर माप_प्रकार nbw0 = ggml_type_आकार(vec_dot_type);
        स्थिर माप_प्रकार nbw1 = ggml_row_आकार(vec_dot_type, ne10);
        स्थिर माप_प्रकार nbw2 = nbw1*ne11;
        स्थिर माप_प्रकार nbw3 = nbw2*ne12;

        निश्चित(params->wsize >= ne13*nbw3);
        // src1 is either packed from F32 into vec_dot_type, or widened from F16 or BF16 into the F32 work buffer
        स्थिर bool widen = src1->type != GGML_TYPE_F32;

        GGML_ASSERT(!widen || vec_dot_type == GGML_TYPE_F32);
        GGML_ASSERT(!widen || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_BF16);

    #अगर 0
        क्रम (int64_t i13 = 0; i13 < ne13; ++i13) {
            क्रम (int64_t i12 = 0; i12 < ne12; ++i12) {
                क्रम (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                    from_float((भग्न *)((अक्षर *) src1->तथ्य + i13*nb13 + i12*nb12 + i11*nb11),
                               (व्योम *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                                ne10);
                }
            }
        }
    #अन्यथा
        स्थिर int64_t bs = ggml_blck_आकार(vec_dot_type);

        क्रम (int64_t i13 = 0; i13 < ne13; ++i13) {
            क्रम (int64_t i12 = 0; i12 < ne12; ++i12) {
                क्रम (int64_t i11 = 0; i11 < ne11; ++i11) {
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    स्थिर व्योम * src1_row  = (स्थिर अक्षर *) src1->तथ्य + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10;
                    व्योम       * wdata_row =                    wdata  + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0;

                    स्थिर int64_t ne10_block_आकार = (ne10_block_end - ne10_block_start) * bs;

                    अगर (src1->type == GGML_TYPE_F16) {
                        ggml_cpu_fp16_to_fp32((स्थिर ggml_fp16_t *) src1_row, (भग्न *) wdata_row, ne10_block_आकार);
                    } अन्यथा अगर (src1->type == GGML_TYPE_BF16) {
                        ggml_cpu_bf16_to_fp32((स्थिर ggml_bf16_t *) src1_row, (भग्न *) wdata_row, ne10_block_आकार);
                    } अन्यथा {
                        from_float((स्थिर भग्न *) src1_row, wdata_row, ne10_block_आकार);
                    }
                }
            }
        }
    #पूर्ण_अगर
    }

    अगर (ith == 0) {
        // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
        atomic_store_explicit(&params->threadpool->current_chunk, nth, memory_order_relaxed);
    }

    ggml_barrier(params->threadpool);

#अगर GGML_USE_LLAMAFILE
    अगर (!params->use_ref && src1->type != vec_dot_type) {
        स्थिर व्योम* wdata = (src1->type == vec_dot_type) ? src1->तथ्य : params->wdata;
        स्थिर माप_प्रकार row_आकार = ggml_row_आकार(vec_dot_type, ne10);

        क्रम (int64_t i13 = 0; i13 < ne13; i13++)
            क्रम (int64_t i12 = 0; i12 < ne12; i12++)
                अगर (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_आकार(src0->type),
                                     (स्थिर अक्षर *)src0->तथ्य + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_आकार(src0->type),
                                     (स्थिर अक्षर *)wdata + (i12*ne11 + i13*ne12*ne11)*row_आकार,
                                     row_आकार/ggml_type_आकार(vec_dot_type),
                                     (अक्षर *)dst->तथ्य + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_आकार(dst->type),
                                     src0->type,
                                     vec_dot_type,
                                     dst->type))
                    जाओ UseGgmlGemm2;
        वापस;
    }
UseGgmlGemm2:;
#पूर्ण_अगर

    // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
    स्थिर int64_t nr0 = ne0;

    // This is the size of the rest of the dimensions of the result
    स्थिर int64_t nr1 = ne1 * ne2 * ne3;

    // Now select a reasonable chunk size.
    पूर्णांक chunk_आकार = 16;

    // We need to step up the size if it's small
    अगर (nr0 == 1 || nr1 == 1) {
        chunk_आकार = 64;
    }

    // distribute the work across the inner or outer loop based on which one is larger
    // The number of chunks in the 0/1 dim.
    // CEIL(nr0/chunk_size)
    int64_t nchunk0 = (nr0 + chunk_आकार - 1) / chunk_आकार;
    int64_t nchunk1 = (nr1 + chunk_आकार - 1) / chunk_आकार;

    // If the chunking is poor for the number of threads on this setup, scrap the whole plan.  Re-chunk it by thread.
    //   Also, chunking by thread was measured to have perform better on NUMA systems.  See https://github.com/ggml-org/llama.cpp/pull/6915
    //   In theory, chunking should be just as useful on NUMA and non NUMA systems, but testing disagreed with that.
    अगर (nchunk0 * nchunk1 < nth * 4 || ggml_is_numa()) {
        // distribute the thread work across the inner or outer loop based on which one is larger
        nchunk0 = nr0 > nr1 ? nth : 1; // parallelize by src0 rows
        nchunk1 = nr0 > nr1 ? 1 : nth; // parallelize by src1 rows
    }

    // The number of elements in each chunk
    स्थिर int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    स्थिर int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    // The first chunk comes from our thread_id, the rest will get auto-assigned.
    पूर्णांक current_chunk = ith;

    जबतक (current_chunk < nchunk0 * nchunk1) {
        स्थिर int64_t ith0 = current_chunk % nchunk0;
        स्थिर int64_t ith1 = current_chunk / nchunk0;

        स्थिर int64_t ir0_start = dr0 * ith0;
        स्थिर int64_t ir0_end = MIN(ir0_start + dr0, nr0);

        स्थिर int64_t ir1_start = dr1 * ith1;
        स्थिर int64_t ir1_end = MIN(ir1_start + dr1, nr1);

        // dot kernels can handle 1 row and col at a time, but mmla kernels can process 2 rows and cols
        int64_t अंक_rows_per_vec_dot = vec_dot_अंक_rows;

        // these checks are needed to avoid crossing dim1 boundaries
        // can be optimized, but the logic would become more complicated, so keeping it like this for simplicity
        अगर ((nr0 % 2 != 0) || (ne11 % 2 != 0) || ((ir0_end - ir0_start) % 2 != 0) || ((ir1_end - ir1_start) % 2 != 0)) {
            अंक_rows_per_vec_dot = 1;
        }
        ggml_compute_forward_mul_mat_one_chunk(params, dst, src0->type, अंक_rows_per_vec_dot, ir0_start, ir0_end, ir1_start, ir1_end);

        अगर (nth >= nchunk0 * nchunk1) {
            अवरोध;
        }

        current_chunk = atomic_fetch_add_explicit(&params->threadpool->current_chunk, 1, memory_order_relaxed);
    }
}

// ggml_compute_forward_mul_mat_id

#घोषणा MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id)*ids->ne[0]*ids->ne[1] + (i1)]

काष्ठा mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

अटल व्योम ggml_compute_forward_mul_mat_id_one_chunk(
    काष्ठा ggml_tensor * dst,
    स्थिर काष्ठा ggml_tensor * src0,
    स्थिर काष्ठा ggml_tensor * src1,
    स्थिर काष्ठा ggml_tensor * ids,
    स्थिर int64_t cur_a,
    स्थिर int64_t ir0_start,
    स्थिर int64_t ir0_end,
    स्थिर int64_t ir1_start,
    स्थिर int64_t ir1_end,
    स्थिर अक्षर * src0_cur,
    स्थिर काष्ठा mmid_row_mapping * matrix_rows,
    स्थिर माप_प्रकार row_आकार,
    स्थिर bool src1_cont,
    स्थिर व्योम * wdata) {

    GGML_TENSOR_द्विआधारी_OP_LOCALS

    स्थिर क्रमागत ggml_type type = src0->type;

    ggml_vec_dot_t    स्थिर vec_dot      = type_traits_cpu[type].vec_dot;
    क्रमागत ggml_type    स्थिर vec_dot_type = type_traits_cpu[type].vec_dot_type;

    स्थिर int64_t blck_0 = 16;
    स्थिर int64_t blck_1 = 16;

    भग्न tmp[16];

    क्रम (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        क्रम (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            क्रम (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ++ir1) {
                स्थिर int64_t _i12 = ir1; // logical row index for this expert

                काष्ठा mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, _i12);
                स्थिर पूर्णांक id       = row_mapping.i1; // selected expert index

                स्थिर int64_t  i11 = id % ne11;
                स्थिर int64_t  i12 = row_mapping.i2; // row index in src1

                स्थिर int64_t  i1 = id;  // selected expert index
                स्थिर int64_t  i2 = i12; // row

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                स्थिर अक्षर * src1_col = (स्थिर अक्षर *) wdata +
                    (src1_cont || src1->type != vec_dot_type
                    ? (i11      + i12*ne11)*row_आकार
                    : (i11*nb11 + i12*nb12));

                भग्न * dst_col = (भग्न *) ((अक्षर *) dst->तथ्य + (i1*nb1 + i2*nb2));

                क्रम (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                    vec_dot(ne00, &tmp[ir0 - iir0], 0, src0_cur + ir0*nb01, 0, src1_col, 0, 1);
                }

                स_नकल(&dst_col[iir0], tmp, (MIN(iir0 + blck_0, ir0_end) - iir0)*माप(भग्न));
            }
        }
    }
}

अटल व्योम * incr_ptr_aligned(व्योम ** p, माप_प्रकार आकार, माप_प्रकार align) {

    व्योम * ptr = *p;
    ptr = (व्योम *) GGML_PAD((uintptr_t) ptr, align);
    *p = (व्योम *) ((अक्षर *) ptr + आकार);
    वापस ptr;
}

अटल व्योम ggml_compute_forward_mul_mat_id(
        स्थिर काष्ठा ggml_compute_params * params,
              काष्ठा ggml_tensor * dst) {

    स्थिर काष्ठा ggml_tensor * src0 = dst->src[0];
    स्थिर काष्ठा ggml_tensor * src1 = dst->src[1];
    स्थिर काष्ठा ggml_tensor * ids = dst->src[2];

    GGML_TENSOR_द्विआधारी_OP_LOCALS

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर क्रमागत ggml_type type = src0->type;

    स्थिर bool src1_cont = ggml_is_contiguous(src1);

    क्रमागत ggml_type    स्थिर vec_dot_type    = type_traits_cpu[type].vec_dot_type;
    ggml_from_float_t स्थिर from_float      = type_traits_cpu[vec_dot_type].from_float;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_आकार(type));
    GGML_ASSERT(nb10 == ggml_type_आकार(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == माप(भग्न));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // row groups
    स्थिर पूर्णांक n_ids = ids->ne[0]; // n_expert_used
    स्थिर पूर्णांक n_as  = ne02;       // n_expert

    व्योम * wdata_cur = params->wdata;

    अगर (src1->type != vec_dot_type) {
        incr_ptr_aligned(&wdata_cur, ggml_row_आकार(vec_dot_type, ggml_nelements(src1)), माप(int64_t));
    }

    int64_t * matrix_row_counts = // [n_as]
        incr_ptr_aligned(&wdata_cur, n_as*माप(int64_t), माप(int64_t));

    काष्ठा mmid_row_mapping * matrix_rows = // [n_as][ids->ne[0]*ids->ne[1]]
        incr_ptr_aligned(&wdata_cur, n_as*ids->ne[0]*ids->ne[1]*माप(काष्ठा mmid_row_mapping), माप(int64_t));

    अक्षर (*atomic_current_chunk)[CACHE_पंक्ति_आकार] = // [n_as]
        incr_ptr_aligned(&wdata_cur, CACHE_पंक्ति_आकार * n_as, CACHE_पंक्ति_आकार);

    // Tiled matmul (see tiled.h); per-thread work buffers, 0 bytes when disabled. The
    // reservation is unconditional, the per expert eligibility is decided at dispatch time
    अक्षर * tiled_scratch = incr_ptr_aligned(&wdata_cur, ggml_tiled_wdata_आकार(nth, dst), 64);

    GGML_ASSERT(params->wsize >= (माप_प्रकार)((अक्षर *) wdata_cur - (अक्षर *) params->wdata));

    अगर (src1->type != vec_dot_type) {
        अक्षर * wdata = params->wdata;

        स्थिर माप_प्रकार nbw0 = ggml_type_आकार(vec_dot_type);
        स्थिर माप_प्रकार nbw1 = ggml_row_आकार(vec_dot_type, ne10);
        स्थिर माप_प्रकार nbw2 = nbw1*ne11;
        स्थिर माप_प्रकार nbw3 = nbw2*ne12;

        निश्चित(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

#अगर 0
        क्रम (int64_t i13 = 0; i13 < ne13; ++i13) {
            क्रम (int64_t i12 = ith; i12 < ne12; i12 += nth) {
                क्रम (int64_t i11 = 0; i11 < ne11; ++i11) {
                    from_float((भग्न *)((अक्षर *) src1->तथ्य + i13*nb13 + i12*nb12 + i11*nb11),
                               (व्योम *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                               ne10);
                }
            }
        }
#अन्यथा
        क्रम (int64_t i13 = 0; i13 < ne13; ++i13) {
            क्रम (int64_t i12 = 0; i12 < ne12; ++i12) {
                क्रम (int64_t i11 = 0; i11 < ne11; ++i11) {
                    माप_प्रकार bs = ggml_blck_आकार(vec_dot_type);
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    from_float((भग्न *)((अक्षर *) src1->तथ्य + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10),
                               (व्योम *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0),
                               (ne10_block_end - ne10_block_start) * bs);
                }
            }
        }
#पूर्ण_अगर
    }

    अगर (ith == 0) {
        // initialize matrix_row_counts
        स_रखो(matrix_row_counts, 0, n_as*माप(int64_t));

        // group rows by src0 matrix
        क्रम (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            क्रम (पूर्णांक id = 0; id < n_ids; ++id) {
                स्थिर int32_t i02 = *(स्थिर int32_t *) ((स्थिर अक्षर *) ids->तथ्य + iid1*ids->nb[1] + id*ids->nb[0]);

                निश्चित(i02 >= 0 && i02 < n_as);

                MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = (काष्ठा mmid_row_mapping) {id, iid1};
                matrix_row_counts[i02] += 1;
            }
        }
    }

    // reset current_chunk
    क्रम (पूर्णांक cur_a = ith; cur_a < n_as; cur_a += nth) {
        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);
        *current_chunk_ctr = nth;
    }

    ggml_barrier(params->threadpool);

    क्रम (पूर्णांक cur_a = 0; cur_a < n_as; ++cur_a) {
        स्थिर int64_t cne1 = matrix_row_counts[cur_a];

        अगर (cne1 == 0) {
            जारी;
        }

        // tiled takes over if profitable for this expert (see tiled.h)
        अगर (ggml_compute_forward_mul_mat_id_tiled(params, dst, cur_a, cne1, (स्थिर int32_t *) &MMID_MATRIX_ROW(cur_a, 0), tiled_scratch)) {
            जारी;
        }

        स्थिर अक्षर * src0_cur = (स्थिर अक्षर *) src0->तथ्य + cur_a * nb02;
        स्थिर व्योम * wdata = (src1->type == vec_dot_type) ? src1->तथ्य : params->wdata;
        स्थिर माप_प्रकार row_आकार = ggml_row_आकार(vec_dot_type, ne10);

        स्थिर int64_t nr0 = ne01;
        स्थिर int64_t nr1 = cne1;

        पूर्णांक chunk_आकार = 16;
        अगर (nr0 == 1 || nr1 == 1) {
            chunk_आकार = 64;
        }

        // disable for NUMA
        स्थिर bool disable_chunking = ggml_is_numa();

        int64_t nchunk0 = (nr0 + chunk_आकार - 1) / chunk_आकार;
        int64_t nchunk1 = (nr1 + chunk_आकार - 1) / chunk_आकार;

        अगर (nchunk0 * nchunk1 < nth * 4 || disable_chunking) {
            nchunk0 = nr0 > nr1 ? nth : 1;
            nchunk1 = nr0 > nr1 ? 1 : nth;
        }

        स्थिर int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        स्थिर int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

        पूर्णांक current_chunk = ith;

        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);

        जबतक (current_chunk < nchunk0 * nchunk1) {
            स्थिर int64_t ith0 = current_chunk % nchunk0;
            स्थिर int64_t ith1 = current_chunk / nchunk0;

            स्थिर int64_t ir0_start = dr0 * ith0;
            स्थिर int64_t ir0_end = MIN(ir0_start + dr0, nr0);

            स्थिर int64_t ir1_start = dr1 * ith1;
            स्थिर int64_t ir1_end = MIN(ir1_start + dr1, nr1);

            ggml_compute_forward_mul_mat_id_one_chunk(
                dst, src0, src1, ids, cur_a,
                ir0_start, ir0_end, ir1_start, ir1_end,
                src0_cur, matrix_rows, row_आकार, src1_cont, wdata
            );

            अगर (nth >= nchunk0 * nchunk1) {
                अवरोध;
            }

            current_chunk = atomic_fetch_add_explicit(current_chunk_ctr, 1, memory_order_relaxed);
        }
    }
}

/////////////////////////////////

अटल व्योम ggml_compute_forward(काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * tensor) {
    GGML_ASSERT(params);

    अगर (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {
        वापस;
    }

    // extra_buffer op?
    अगर (ggml_cpu_extra_compute_forward(params, tensor)) {
        वापस;
    }

    चयन (tensor->op) {
        हाल GGML_OP_DUP:
            {
                ggml_compute_forward_dup(params, tensor);
            } अवरोध;
        हाल GGML_OP_ADD:
            {
                ggml_compute_forward_add(params, tensor);
            } अवरोध;
        हाल GGML_OP_ADD_ID:
            {
                ggml_compute_forward_add_id(params, tensor);
            } अवरोध;
        हाल GGML_OP_ADD1:
            {
                ggml_compute_forward_add1(params, tensor);
            } अवरोध;
        हाल GGML_OP_ACC:
            {
                ggml_compute_forward_acc(params, tensor);
            } अवरोध;
        हाल GGML_OP_SUB:
            {
                ggml_compute_forward_sub(params, tensor);
            } अवरोध;
        हाल GGML_OP_MUL:
            {
                ggml_compute_forward_mul(params, tensor);
            } अवरोध;
        हाल GGML_OP_DIV:
            {
                ggml_compute_forward_div(params, tensor);
            } अवरोध;
        हाल GGML_OP_SQR:
            {
                ggml_compute_forward_sqr(params, tensor);
            } अवरोध;
        हाल GGML_OP_SQRT:
            {
                ggml_compute_forward_sqrt(params, tensor);
            } अवरोध;
        हाल GGML_OP_LOG:
            {
                ggml_compute_forward_log(params, tensor);
            } अवरोध;
        हाल GGML_OP_SIN:
            {
                ggml_compute_forward_sin(params, tensor);
            } अवरोध;
        हाल GGML_OP_COS:
            {
                ggml_compute_forward_cos(params, tensor);
            } अवरोध;
        हाल GGML_OP_योग:
            {
                ggml_compute_forward_योग(params, tensor);
            } अवरोध;
        हाल GGML_OP_योग_ROWS:
            {
                ggml_compute_forward_योग_rows(params, tensor);
            } अवरोध;
        हाल GGML_OP_CUMSUM:
            {
                ggml_compute_forward_cumsum(params, tensor);
            } अवरोध;
        हाल GGML_OP_MEAN:
            {
                ggml_compute_forward_mean(params, tensor);
            } अवरोध;
        हाल GGML_OP_ARGMAX:
            {
                ggml_compute_forward_argmax(params, tensor);
            } अवरोध;
        हाल GGML_OP_गिनती_EQUAL:
            {
                ggml_compute_forward_गिनती_equal(params, tensor);
            } अवरोध;
        हाल GGML_OP_REPEAT:
            {
                ggml_compute_forward_repeat(params, tensor);
            } अवरोध;
        हाल GGML_OP_REPEAT_BACK:
            {
                ggml_compute_forward_repeat_back(params, tensor);
            } अवरोध;
        हाल GGML_OP_CONCAT:
            {
                ggml_compute_forward_concat(params, tensor);
            } अवरोध;
        हाल GGML_OP_SILU_BACK:
            {
                ggml_compute_forward_silu_back(params, tensor);
            } अवरोध;
        हाल GGML_OP_NORM:
            {
                ggml_compute_forward_norm(params, tensor);
            } अवरोध;
        हाल GGML_OP_RMS_NORM:
            {
                ggml_compute_forward_rms_norm(params, tensor);
            } अवरोध;
        हाल GGML_OP_RMS_NORM_BACK:
            {
                ggml_compute_forward_rms_norm_back(params, tensor);
            } अवरोध;
        हाल GGML_OP_GROUP_NORM:
            {
                ggml_compute_forward_group_norm(params, tensor);
            } अवरोध;
        हाल GGML_OP_L2_NORM:
            {
                ggml_compute_forward_l2_norm(params, tensor);
            } अवरोध;
        हाल GGML_OP_MUL_MAT:
            {
                ggml_compute_forward_mul_mat(params, tensor);
            } अवरोध;
        हाल GGML_OP_MUL_MAT_ID:
            {
                ggml_compute_forward_mul_mat_id(params, tensor);
            } अवरोध;
        हाल GGML_OP_OUT_PROD:
            {
                ggml_compute_forward_out_prod(params, tensor);
            } अवरोध;
        हाल GGML_OP_SCALE:
            {
                ggml_compute_forward_scale(params, tensor);
            } अवरोध;
        हाल GGML_OP_SET:
            {
                ggml_compute_forward_set(params, tensor);
            } अवरोध;
        हाल GGML_OP_CPY:
            {
                ggml_compute_forward_cpy(params, tensor);
            } अवरोध;
        हाल GGML_OP_CONT:
            {
                ggml_compute_forward_cont(params, tensor);
            } अवरोध;
        हाल GGML_OP_GET_ROWS:
            {
                ggml_compute_forward_get_rows(params, tensor);
            } अवरोध;
        हाल GGML_OP_GET_ROWS_BACK:
            {
                ggml_compute_forward_get_rows_back(params, tensor);
            } अवरोध;
        हाल GGML_OP_SET_ROWS:
            {
                ggml_compute_forward_set_rows(params, tensor);
            } अवरोध;
        हाल GGML_OP_DIAG:
            {
                ggml_compute_forward_diag(params, tensor);
            } अवरोध;
        हाल GGML_OP_DIAG_MASK_INF:
            {
                ggml_compute_forward_diag_mask_inf(params, tensor);
            } अवरोध;
        हाल GGML_OP_DIAG_MASK_ZERO:
            {
                ggml_compute_forward_diag_mask_zero(params, tensor);
            } अवरोध;
        हाल GGML_OP_SOFT_MAX:
            {
                ggml_compute_forward_soft_max(params, tensor);
            } अवरोध;
        हाल GGML_OP_SOFT_MAX_BACK:
            {
                ggml_compute_forward_soft_max_ext_back(params, tensor);
            } अवरोध;
        हाल GGML_OP_ROPE:
            {
                ggml_compute_forward_rope(params, tensor);
            } अवरोध;
        हाल GGML_OP_ROPE_BACK:
            {
                ggml_compute_forward_rope_back(params, tensor);
            } अवरोध;
        हाल GGML_OP_CLAMP:
            {
                ggml_compute_forward_clamp(params, tensor);
            } अवरोध;
        हाल GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_compute_forward_conv_transpose_1d(params, tensor);
            } अवरोध;
        हाल GGML_OP_IM2COL:
            {
                ggml_compute_forward_im2col(params, tensor);
            } अवरोध;
        हाल GGML_OP_IM2COL_BACK:
            {
                ggml_compute_forward_im2col_back_f32(params, tensor);
            } अवरोध;
        हाल GGML_OP_IM2COL_3D:
            {
                ggml_compute_forward_im2col_3d(params, tensor);
            } अवरोध;
        हाल GGML_OP_COL2IM_1D:
            {
                ggml_compute_forward_col2im_1d(params, tensor);
            } अवरोध;
        हाल GGML_OP_CONV_2D:
            {
                ggml_compute_forward_conv_2d(params, tensor);
            } अवरोध;
        हाल GGML_OP_CONV_3D:
            {
                ggml_compute_forward_conv_3d(params, tensor);
            } अवरोध;
        हाल GGML_OP_CONV_2D_DW:
            {
                ggml_compute_forward_conv_2d_dw(params, tensor);
            } अवरोध;
        हाल GGML_OP_CONV_TRANSPOSE_2D:
            {
                ggml_compute_forward_conv_transpose_2d(params, tensor);
            } अवरोध;
        हाल GGML_OP_POOL_1D:
            {
                ggml_compute_forward_pool_1d(params, tensor);
            } अवरोध;
        हाल GGML_OP_POOL_2D:
            {
                ggml_compute_forward_pool_2d(params, tensor);
            } अवरोध;
        हाल GGML_OP_POOL_2D_BACK:
            {
                ggml_compute_forward_pool_2d_back(params, tensor);
            } अवरोध;
        हाल GGML_OP_UPSCALE:
            {
                ggml_compute_forward_upscale(params, tensor);
            } अवरोध;
        हाल GGML_OP_PAD:
            {
                ggml_compute_forward_pad(params, tensor);
            } अवरोध;
        हाल GGML_OP_PAD_REFLECT_1D:
            {
                ggml_compute_forward_pad_reflect_1d(params, tensor);
            } अवरोध;
        हाल GGML_OP_ROLL:
            {
                ggml_compute_forward_roll(params, tensor);
            } अवरोध;
        हाल GGML_OP_ARANGE:
            {
                ggml_compute_forward_arange(params, tensor);
            } अवरोध;
        हाल GGML_OP_TIMESTEP_EMBEDDING:
            {
                ggml_compute_forward_timestep_embedding(params, tensor);
            } अवरोध;
        हाल GGML_OP_ARGSORT:
            {
                ggml_compute_forward_argsort(params, tensor);
            } अवरोध;
        हाल GGML_OP_शीर्ष_K:
            {
                ggml_compute_forward_शीर्ष_k(params, tensor);
            } अवरोध;
        हाल GGML_OP_LEAKY_RELU:
            {
                ggml_compute_forward_leaky_relu(params, tensor);
            } अवरोध;
        हाल GGML_OP_TRI:
            {
                ggml_compute_forward_tri(params, tensor);
            } अवरोध;
        हाल GGML_OP_FILL:
            {
                ggml_compute_forward_fill(params, tensor);
            } अवरोध;
        हाल GGML_OP_FLASH_ATTN_EXT:
            {
                ggml_compute_forward_flash_attn_ext(params, tensor);
            } अवरोध;
        हाल GGML_OP_FLASH_ATTN_BACK:
            {
                int32_t t = ggml_get_op_params_i32(tensor, 0);
                GGML_ASSERT(t == 0 || t == 1);
                bool masked = t != 0;
                ggml_compute_forward_flash_attn_back(params, masked, tensor);
            } अवरोध;
        हाल GGML_OP_SSM_CONV:
            {
                ggml_compute_forward_ssm_conv(params, tensor);
            } अवरोध;
        हाल GGML_OP_SSM_SCAN:
            {
                ggml_compute_forward_ssm_scan(params, tensor);
            } अवरोध;
        हाल GGML_OP_WIN_PART:
            {
                ggml_compute_forward_win_part(params, tensor);
            } अवरोध;
        हाल GGML_OP_WIN_UNPART:
            {
                ggml_compute_forward_win_unpart(params, tensor);
            } अवरोध;
        हाल GGML_OP_UNARY:
            {
                ggml_compute_forward_unary(params, tensor);
            } अवरोध;
        हाल GGML_OP_GLU:
            {
                ggml_compute_forward_glu(params, tensor);
            } अवरोध;
        हाल GGML_OP_GET_REL_POS:
            {
                ggml_compute_forward_get_rel_pos(params, tensor);
            } अवरोध;
        हाल GGML_OP_ADD_REL_POS:
            {
                ggml_compute_forward_add_rel_pos(params, tensor);
            } अवरोध;
        हाल GGML_OP_RWKV_WKV6:
            {
                ggml_compute_forward_rwkv_wkv6(params, tensor);
            } अवरोध;
        हाल GGML_OP_GATED_LINEAR_ATTN:
            {
                ggml_compute_forward_gla(params, tensor);
            } अवरोध;
        हाल GGML_OP_RWKV_WKV7:
            {
                ggml_compute_forward_rwkv_wkv7(params, tensor);
            } अवरोध;
        हाल GGML_OP_SOLVE_TRI:
            {
                ggml_compute_forward_solve_tri(params, tensor);
            } अवरोध;
        हाल GGML_OP_GATED_DELTA_NET:
            {
                ggml_compute_forward_gated_delta_net(params, tensor);
            } अवरोध;
        हाल GGML_OP_LIGHTNING_INDEXER:
            {
                ggml_compute_forward_lightning_indexer(params, tensor);
            } अवरोध;
        हाल GGML_OP_DSV4_HC_COMB:
            {
                ggml_compute_forward_dsv4_hc_comb(params, tensor);
            } अवरोध;
        हाल GGML_OP_DSV4_HC_PRE:
            {
                ggml_compute_forward_dsv4_hc_pre(params, tensor);
            } अवरोध;
        हाल GGML_OP_DSV4_HC_POST:
            {
                ggml_compute_forward_dsv4_hc_post(params, tensor);
            } अवरोध;
        हाल GGML_OP_शीर्ष_K_MASK:
            {
                ggml_compute_forward_शीर्ष_k_mask(params, tensor);
            } अवरोध;
        हाल GGML_OP_MAP_CUSTOM1:
            {
                ggml_compute_forward_map_custom1(params, tensor);
            }
            अवरोध;
        हाल GGML_OP_MAP_CUSTOM2:
            {
                ggml_compute_forward_map_custom2(params, tensor);
            }
            अवरोध;
        हाल GGML_OP_MAP_CUSTOM3:
            {
                ggml_compute_forward_map_custom3(params, tensor);
            }
            अवरोध;
        हाल GGML_OP_CUSTOM:
            {
                ggml_compute_forward_custom(params, tensor);
            }
            अवरोध;
        हाल GGML_OP_CROSS_ENTROPY_LOSS:
            {
                ggml_compute_forward_cross_entropy_loss(params, tensor);
            }
            अवरोध;
        हाल GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            {
                ggml_compute_forward_cross_entropy_loss_back(params, tensor);
            }
            अवरोध;
        हाल GGML_OP_OPT_STEP_ADAMW:
            {
                ggml_compute_forward_opt_step_adamw(params, tensor);
            }
            अवरोध;
        हाल GGML_OP_OPT_STEP_SGD:
            {
                ggml_compute_forward_opt_step_sgd(params, tensor);
            }
            अवरोध;
        हाल GGML_OP_NONE:
            {
                // nop
            } अवरोध;
        हाल GGML_OP_RESHAPE:
            {
                // nop
            } अवरोध;
        हाल GGML_OP_PERMUTE:
            {
                // nop
            } अवरोध;
        हाल GGML_OP_VIEW:
            {
                // nop
            } अवरोध;
        हाल GGML_OP_TRANSPOSE:
            {
                // nop
            } अवरोध;
        हाल GGML_OP_गिनती:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// Android's libc implementation "bionic" does not support setting affinity
#अगर defined(__gnu_linux__)
अटल व्योम set_numa_thread_affinity(पूर्णांक thread_n) {
    अगर (!ggml_is_numa()) {
        वापस;
    }

    पूर्णांक गाँठ_अंक;
    पूर्णांक rv;
    माप_प्रकार setsize = CPU_ALLOC_आकार(g_state.numa.योग_cpus);

    चयन(g_state.numa.numa_strategy) {
        हाल GGML_NUMA_STRATEGY_DISTRIBUTE:
            // run thread on node_num thread_n / (threads per node)
            गाँठ_अंक = thread_n % g_state.numa.n_nodes;
            अवरोध;
        हाल GGML_NUMA_STRATEGY_ISOLATE:
            // run thread on current_node
            गाँठ_अंक = g_state.numa.current_गाँठ;
            अवरोध;
        हाल GGML_NUMA_STRATEGY_NUMACTL:
            // use the cpuset that numactl gave us
            rv = pthread_setaffinity_np(pthread_self(), setsize, &g_state.numa.cpuset);
            अगर (rv) {
                ख_लिखो(मानक_त्रुटि, "warning: pthread_setaffinity_np() failed: %s\n",म_त्रुटि(rv));
            }
            वापस;
        शेष:
            वापस;
    }

    काष्ठा ggml_numa_गाँठ * गाँठ = &g_state.numa.nodes[गाँठ_अंक];

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.योग_cpus);
    CPU_ZERO_S(setsize, cpus);
    क्रम (माप_प्रकार i = 0; i < गाँठ->n_cpus; ++i) {
        CPU_SET_S(गाँठ->cpus[i], setsize, cpus);
    }

    rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    अगर (rv) {
            ख_लिखो(मानक_त्रुटि, "warning: pthread_setaffinity_np() failed: %s\n", म_त्रुटि(rv));
    }

    CPU_FREE(cpus);
}

अटल व्योम clear_numa_thread_affinity(व्योम) {
    अगर (!ggml_is_numa()) {
        वापस;
    }

    माप_प्रकार setsize = CPU_ALLOC_आकार(g_state.numa.योग_cpus);

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.योग_cpus);
    CPU_ZERO_S(setsize, cpus);
    क्रम (अचिन्हित i = 0; i < g_state.numa.योग_cpus; ++i) {
        CPU_SET_S(i, setsize, cpus);
    }

    पूर्णांक rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    अगर (rv) {
        ख_लिखो(मानक_त्रुटि, "warning: pthread_setaffinity_np() failed: %s\n", म_त्रुटि(rv));
    }

    CPU_FREE(cpus);
}
#अन्यथा
// TODO: Windows etc.
// (the linux implementation may also work on BSD, someone should test)
अटल व्योम set_numa_thread_affinity(पूर्णांक thread_n) { UNUSED(thread_n);  }
अटल व्योम clear_numa_thread_affinity(व्योम) {}
#पूर्ण_अगर

अटल पूर्णांक ggml_get_n_tasks(काष्ठा ggml_tensor * गाँठ, पूर्णांक n_threads) {
    पूर्णांक n_tasks = 0;

    अगर (ggml_is_empty(गाँठ)) {
        // no need to multi-thread a no-op
        n_tasks = 1;
        वापस n_tasks;
    }

    चयन (गाँठ->op) {
        हाल GGML_OP_CPY:
        हाल GGML_OP_DUP:
        हाल GGML_OP_CONT:
        हाल GGML_OP_ADD:
        हाल GGML_OP_ADD_ID:
        हाल GGML_OP_ADD1:
        हाल GGML_OP_ACC:
        हाल GGML_OP_CUMSUM:
        हाल GGML_OP_TRI:
        हाल GGML_OP_FILL:
            {
                n_tasks = n_threads;
            } अवरोध;
        हाल GGML_OP_SUB:
        हाल GGML_OP_SQR:
        हाल GGML_OP_SQRT:
        हाल GGML_OP_LOG:
        हाल GGML_OP_SIN:
        हाल GGML_OP_COS:
        हाल GGML_OP_योग:
        हाल GGML_OP_योग_ROWS:
        हाल GGML_OP_MEAN:
        हाल GGML_OP_ARGMAX:
            {
                n_tasks = 1;
            } अवरोध;
        हाल GGML_OP_गिनती_EQUAL:
        हाल GGML_OP_SOLVE_TRI:
        हाल GGML_OP_GATED_DELTA_NET:
        हाल GGML_OP_DSV4_HC_COMB:
        हाल GGML_OP_DSV4_HC_PRE:
        हाल GGML_OP_DSV4_HC_POST:
        हाल GGML_OP_शीर्ष_K_MASK:
            {
                n_tasks = n_threads;
            } अवरोध;
        हाल GGML_OP_REPEAT:
        हाल GGML_OP_REPEAT_BACK:
        हाल GGML_OP_LEAKY_RELU:
            {
                n_tasks = 1;
            } अवरोध;
        हाल GGML_OP_UNARY:
            चयन (ggml_get_unary_op(गाँठ)) {
                हाल GGML_UNARY_OP_ABS:
                हाल GGML_UNARY_OP_SGN:
                हाल GGML_UNARY_OP_NEG:
                हाल GGML_UNARY_OP_STEP:
                हाल GGML_UNARY_OP_TANH:
                हाल GGML_UNARY_OP_ELU:
                हाल GGML_UNARY_OP_RELU:
                हाल GGML_UNARY_OP_SIGMOID:
                हाल GGML_UNARY_OP_HARDSWISH:
                हाल GGML_UNARY_OP_HARDSIGMOID:
                हाल GGML_UNARY_OP_EXP:
                हाल GGML_UNARY_OP_SOFTPLUS:
                हाल GGML_UNARY_OP_EXPM1:
                हाल GGML_UNARY_OP_FLOOR:
                हाल GGML_UNARY_OP_CEIL:
                हाल GGML_UNARY_OP_ROUND:
                हाल GGML_UNARY_OP_TRUNC:
                    {
                        n_tasks = 1;
                    } अवरोध;

                हाल GGML_UNARY_OP_GELU:
                हाल GGML_UNARY_OP_GELU_ERF:
                हाल GGML_UNARY_OP_GELU_QUICK:
                हाल GGML_UNARY_OP_SILU:
                हाल GGML_UNARY_OP_XIELU:
                    {
                        n_tasks = n_threads;
                    } अवरोध;
                शेष:
                    GGML_ABORT("fatal error");
            }
            अवरोध;
        हाल GGML_OP_GLU:
            चयन (ggml_get_glu_op(गाँठ)) {
                हाल GGML_GLU_OP_REGLU:
                हाल GGML_GLU_OP_GEGLU:
                हाल GGML_GLU_OP_SWIGLU:
                हाल GGML_GLU_OP_SWIGLU_OAI:
                हाल GGML_GLU_OP_GEGLU_ERF:
                हाल GGML_GLU_OP_GEGLU_QUICK:
                हाल GGML_GLU_OP_SWIGLU_CLAMP:
                    {
                        n_tasks = n_threads;
                    } अवरोध;
                शेष:
                    GGML_ABORT("fatal error");
            }
            अवरोध;
        हाल GGML_OP_SILU_BACK:
        हाल GGML_OP_MUL:
        हाल GGML_OP_DIV:
        हाल GGML_OP_NORM:
        हाल GGML_OP_RMS_NORM:
        हाल GGML_OP_RMS_NORM_BACK:
        हाल GGML_OP_L2_NORM:
        हाल GGML_OP_GROUP_NORM:
        हाल GGML_OP_CONCAT:
        हाल GGML_OP_MUL_MAT:
        हाल GGML_OP_MUL_MAT_ID:
        हाल GGML_OP_OUT_PROD:
        हाल GGML_OP_SET_ROWS:
            {
                n_tasks = n_threads;
            } अवरोध;
        हाल GGML_OP_GET_ROWS:
            {
                // FIXME: get_rows can use additional threads, but the cost of launching additional threads
                // decreases performance with GPU offloading
                //n_tasks = n_threads;
                n_tasks = 1;
            } अवरोध;
        हाल GGML_OP_SCALE:
        हाल GGML_OP_SET:
        हाल GGML_OP_RESHAPE:
        हाल GGML_OP_VIEW:
        हाल GGML_OP_PERMUTE:
        हाल GGML_OP_TRANSPOSE:
        हाल GGML_OP_GET_ROWS_BACK:
        हाल GGML_OP_DIAG:
            {
                n_tasks = 1;
            } अवरोध;
        हाल GGML_OP_DIAG_MASK_ZERO:
        हाल GGML_OP_DIAG_MASK_INF:
        हाल GGML_OP_SOFT_MAX_BACK:
        हाल GGML_OP_ROPE:
        हाल GGML_OP_ROPE_BACK:
        हाल GGML_OP_ADD_REL_POS:
            {
                n_tasks = n_threads;
            } अवरोध;
        हाल GGML_OP_CLAMP:
            {
                n_tasks = 1; //TODO
            } अवरोध;
        हाल GGML_OP_SOFT_MAX:
            {
                n_tasks = MIN(n_threads, ggml_nrows(गाँठ->src[0]));
            } अवरोध;
        हाल GGML_OP_IM2COL:
        हाल GGML_OP_IM2COL_BACK:
        हाल GGML_OP_IM2COL_3D:
        हाल GGML_OP_CONV_2D:
        हाल GGML_OP_CONV_3D:
        हाल GGML_OP_CONV_2D_DW:
        हाल GGML_OP_COL2IM_1D:
        हाल GGML_OP_CONV_TRANSPOSE_1D:
        हाल GGML_OP_CONV_TRANSPOSE_2D:
            {
                n_tasks = n_threads;
            } अवरोध;
        हाल GGML_OP_POOL_1D:
        हाल GGML_OP_POOL_2D:
        हाल GGML_OP_POOL_2D_BACK:
            {
                n_tasks = 1;
            } अवरोध;
        हाल GGML_OP_UPSCALE:
        हाल GGML_OP_PAD:
        हाल GGML_OP_PAD_REFLECT_1D:
        हाल GGML_OP_ROLL:
        हाल GGML_OP_ARANGE:
        हाल GGML_OP_TIMESTEP_EMBEDDING:
        हाल GGML_OP_ARGSORT:
        हाल GGML_OP_शीर्ष_K:
        हाल GGML_OP_FLASH_ATTN_EXT:
        हाल GGML_OP_FLASH_ATTN_BACK:
        हाल GGML_OP_SSM_CONV:
        हाल GGML_OP_SSM_SCAN:
        हाल GGML_OP_LIGHTNING_INDEXER:
            {
                n_tasks = n_threads;
            } अवरोध;
        हाल GGML_OP_RWKV_WKV6:
        हाल GGML_OP_GATED_LINEAR_ATTN:
        हाल GGML_OP_RWKV_WKV7:
            {
                स्थिर int64_t n_heads = गाँठ->src[1]->ne[1];
                n_tasks = MIN(n_threads, n_heads);
            } अवरोध;
        हाल GGML_OP_WIN_PART:
        हाल GGML_OP_WIN_UNPART:
        हाल GGML_OP_GET_REL_POS:
            {
                n_tasks = 1;
            } अवरोध;
        हाल GGML_OP_MAP_CUSTOM1:
            {
                काष्ठा ggml_map_custom1_op_params p;
                स_नकल(&p, गाँठ->op_params, माप(p));
                अगर (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } अन्यथा {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } अवरोध;
        हाल GGML_OP_MAP_CUSTOM2:
            {
                काष्ठा ggml_map_custom2_op_params p;
                स_नकल(&p, गाँठ->op_params, माप(p));
                अगर (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } अन्यथा {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } अवरोध;
        हाल GGML_OP_MAP_CUSTOM3:
            {
                काष्ठा ggml_map_custom3_op_params p;
                स_नकल(&p, गाँठ->op_params, माप(p));
                अगर (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } अन्यथा {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } अवरोध;
        हाल GGML_OP_CUSTOM:
            {
                काष्ठा ggml_custom_op_params p;
                स_नकल(&p, गाँठ->op_params, माप(p));
                अगर (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } अन्यथा {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } अवरोध;
        हाल GGML_OP_CROSS_ENTROPY_LOSS:
        हाल GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        हाल GGML_OP_OPT_STEP_ADAMW:
        हाल GGML_OP_OPT_STEP_SGD:
            {
                n_tasks = n_threads;
            } अवरोध;
        हाल GGML_OP_NONE:
            {
                n_tasks = 1;
            } अवरोध;
        हाल GGML_OP_गिनती:
            {
                GGML_ABORT("fatal error");
            }
        शेष:
            {
                ख_लिखो(मानक_त्रुटि, "%s: op not implemented: ", __func__);
                अगर (गाँठ->op < GGML_OP_गिनती) {
                    ख_लिखो(मानक_त्रुटि, "%s\n", ggml_op_name(गाँठ->op));
                } अन्यथा {
                    ख_लिखो(मानक_त्रुटि, "%d\n", गाँठ->op);
                }
                GGML_ABORT("fatal error");
            }
    }

    निश्चित(n_tasks > 0);

    वापस n_tasks;
}

अटल thread_ret_t ggml_graph_compute_secondary_thread(व्योम* तथ्य);

#अगर defined(_WIN32)
#समावेश "windows.h"

// TODO: support > 64 CPUs
अटल bool ggml_thread_apply_affinity(bool * mask) {
    HANDLE    h = GetCurrentThread();
    uint64_t  bitmask = 0ULL;

    निश्चित(GGML_MAX_N_THREADS >= 64);

    क्रम (int32_t i = 0; i < 8; i++) {
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

    क्रम (int32_t i = 64; i < GGML_MAX_N_THREADS; i++) {
        अगर (mask[i]) {
            ख_लिखो(मानक_त्रुटि, "warn: setting thread-affinity for > 64 CPUs isn't supported on windows!\n");
            अवरोध;
        }
    }

    DWORD_PTR m = (DWORD_PTR)bitmask;

    m = SetThreadAffinityMask(h, m);

    वापस m != 0;
}

अटल bool ggml_thread_apply_priority(int32_t prio) {
    // Note that on Windows the Process Priority Class must be updated in order to set Thread priority.
    // This is up to the applications.
    DWORD p = THREAD_PRIORITY_NORMAL;
    चयन (prio) {
        हाल GGML_SCHED_PRIO_निचला:      p = THREAD_PRIORITY_BELOW_NORMAL;  अवरोध;
        हाल GGML_SCHED_PRIO_NORMAL:   p = THREAD_PRIORITY_NORMAL;        अवरोध;
        हाल GGML_SCHED_PRIO_MEDIUM:   p = THREAD_PRIORITY_ABOVE_NORMAL;  अवरोध;
        हाल GGML_SCHED_PRIO_ऊपरी:     p = THREAD_PRIORITY_HIGHEST;       अवरोध;
        हाल GGML_SCHED_PRIO_REALTIME: p = THREAD_PRIORITY_TIME_CRITICAL; अवरोध;
    }

    अगर (prio != GGML_SCHED_PRIO_निचला) {
        // Tell Windows that this thread should not be throttled (needs its own CPU core).
        // Newer Windows 11 versions aggressively park (offline) CPU cores and often place
        // all our threads onto the first 4 cores which results in terrible performance with
        // n_threads > 4
        #अगर _WIN32_WINNT >= 0x0602
        THREAD_POWER_THROTTLING_STATE t;
        ZeroMemory(&t, माप(t));
        t.Version     = THREAD_POWER_THROTTLING_CURRENT_VERSION;
        t.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
        t.StateMask   = 0;

        अगर (!SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &t, माप(t))) {
            GGML_LOG_DEBUG("failed to disable thread power throttling %d : (%d)\n", prio, (पूर्णांक) GetLastError());
            वापस false;
        }
        #पूर्ण_अगर
    }

    अगर (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        वापस true;
    }

    अगर (!SetThreadPriority(GetCurrentThread(), p)) {
        ख_लिखो(मानक_त्रुटि, "warn: failed to set thread priority %d : (%d)\n", prio, (पूर्णांक) GetLastError());
        वापस false;
    }

    वापस true;
}

#या_अगर defined(__APPLE__)
#समावेश <sys/types.h>
#समावेश <sys/resource.h>

अटल bool ggml_thread_apply_affinity(स्थिर bool * mask) {
    // Not supported on Apple platforms
    UNUSED(mask);
    वापस true;
}

अटल bool ggml_thread_apply_priority(int32_t prio) {
    काष्ठा sched_param p;
    int32_t policy = SCHED_OTHER;
    चयन (prio) {
        // TODO: there seems to be no way to set lower prio on Apple platforms
        हाल GGML_SCHED_PRIO_निचला:      policy = SCHED_OTHER; p.sched_priority = 0;  अवरोध;
        हाल GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  अवरोध;
        हाल GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; अवरोध;
        हाल GGML_SCHED_PRIO_ऊपरी:     policy = SCHED_FIFO;  p.sched_priority = 80; अवरोध;
        हाल GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; अवरोध;
    }

    अगर (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        वापस true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    अगर (err != 0) {
        ख_लिखो(मानक_त्रुटि, "warn: failed to set thread priority %d : %s (%d)\n", prio, म_त्रुटि(err), err);
        वापस false;
    }

    वापस true;
}

#या_अगर defined(__linux__)
// TODO: this may not work on BSD, to be verified

अटल bool ggml_thread_apply_affinity(स्थिर bool * mask) {
    cpu_set_t cpuset;
    पूर्णांक err;

    CPU_ZERO(&cpuset);

    क्रम (uint32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
        अगर (mask[i]) {
            GGML_PRINT_DEBUG("Thread %lx: adding %d to cpuset\n", pthread_self(), i);
            CPU_SET(i, &cpuset);
        }
    }

#अगर_घोषित __ANDROID__
    err = sched_setaffinity(0, माप(cpuset), &cpuset);
    अगर (err < 0) {
        err = त्रुटि_सं;
    }
#अन्यथा
    err = pthread_setaffinity_np(pthread_self(), माप(cpuset), &cpuset);
#पूर्ण_अगर
    अगर (err != 0) {
        ख_लिखो(मानक_त्रुटि, "warn: failed to set affinity mask 0x%llx : %s (%d)\n", (अचिन्हित दीर्घ दीर्घ)mask, म_त्रुटि(err), err);
        वापस false;
    }

    वापस true;
}

अटल bool ggml_thread_apply_priority(int32_t prio) {
    काष्ठा sched_param p;
    int32_t policy = SCHED_OTHER;
    चयन (prio) {
        हाल GGML_SCHED_PRIO_निचला:      policy = SCHED_BATCH; p.sched_priority = 0;  अवरोध;
        हाल GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  अवरोध;
        हाल GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; अवरोध;
        हाल GGML_SCHED_PRIO_ऊपरी:     policy = SCHED_FIFO;  p.sched_priority = 80; अवरोध;
        हाल GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; अवरोध;
    }

    अगर (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        वापस true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    अगर (err != 0) {
        ख_लिखो(मानक_त्रुटि, "warn: failed to set thread priority %d : %s (%d)\n", prio, म_त्रुटि(err), err);
        वापस false;
    }

    वापस true;
}

#अन्यथा // unsupported platforms

अटल bool ggml_thread_apply_affinity(स्थिर bool * mask) {
    UNUSED(mask);
    वापस true;
}

अटल bool ggml_thread_apply_priority(int32_t prio) {
    UNUSED(prio);
    वापस true;
}

#पूर्ण_अगर

अटल bool ggml_thread_cpumask_is_valid(स्थिर bool * mask) {
    क्रम (पूर्णांक i = 0; i < GGML_MAX_N_THREADS; i++) {
        अगर (mask[i]) { वापस true; }
    }
    वापस false;
}

अटल व्योम ggml_thread_cpumask_अग्र(स्थिर bool * global_mask, bool * local_mask, bool strict, int32_t* iter) {
    अगर (!strict) {
        स_नकल(local_mask, global_mask, GGML_MAX_N_THREADS);
        वापस;
    } अन्यथा {
        स_रखो(local_mask, 0, GGML_MAX_N_THREADS);
        int32_t base_idx = *iter;
        क्रम (int32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
            int32_t idx = base_idx + i;
            अगर (idx >= GGML_MAX_N_THREADS) {
                // Just a cheaper modulo
                idx -= GGML_MAX_N_THREADS;
            }
            अगर (global_mask[idx]) {
                local_mask[idx] = 1;
                *iter = idx + 1;
                वापस;
            }
        }
    }
}

व्योम ggml_threadpool_free(काष्ठा ggml_threadpool* threadpool) {
    अगर (!threadpool) वापस;

    स्थिर पूर्णांक n_threads = threadpool->n_threads;

#अगर_अघोषित GGML_USE_OPENMP
    काष्ठा ggml_compute_state* workers = threadpool->workers;

    ggml_mutex_lock(&threadpool->mutex);

    threadpool->stop = true;
    threadpool->छोड़ो = false;

    ggml_cond_broadcast(&threadpool->cond);
    ggml_mutex_unlock(&threadpool->mutex);

    क्रम (पूर्णांक j = 1; j < n_threads; j++) {
        int32_t rc = ggml_thread_join(workers[j].thrd, शून्य);
        GGML_ASSERT(rc == GGML_EXIT_SUCCESS || rc == GGML_EXIT_ABORTED);
        UNUSED(rc);
    }

    ggml_mutex_destroy(&threadpool->mutex);
    ggml_cond_destroy(&threadpool->cond);
#पूर्ण_अगर // GGML_USE_OPENMP

    स्थिर माप_प्रकार workers_आकार = माप(काष्ठा ggml_compute_state) * n_threads;
    ggml_aligned_free(threadpool->workers, workers_आकार);
    ggml_aligned_free(threadpool, माप(काष्ठा ggml_threadpool));
}

#अगर_अघोषित GGML_USE_OPENMP
// pause/resume must be called under mutex
अटल व्योम ggml_threadpool_pause_locked(काष्ठा ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Pausing threadpool\n");
    threadpool->छोड़ो = true;
    ggml_cond_broadcast(&threadpool->cond);
}

अटल व्योम ggml_threadpool_resume_locked(काष्ठा ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Resuming threadpool\n");
    threadpool->छोड़ो = false;
    ggml_cond_broadcast(&threadpool->cond);
}
#पूर्ण_अगर

व्योम ggml_threadpool_pause(काष्ठा ggml_threadpool * threadpool) {
#अगर_अघोषित GGML_USE_OPENMP
    ggml_mutex_lock(&threadpool->mutex);
    अगर (!threadpool->छोड़ो) {
       ggml_threadpool_pause_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#अन्यथा
    UNUSED(threadpool);
#पूर्ण_अगर
}

व्योम ggml_threadpool_resume(काष्ठा ggml_threadpool * threadpool) {
#अगर_अघोषित GGML_USE_OPENMP
    ggml_mutex_lock(&threadpool->mutex);
    अगर (threadpool->छोड़ो) {
       ggml_threadpool_resume_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#अन्यथा
    UNUSED(threadpool);
#पूर्ण_अगर
}

काष्ठा ggml_cplan ggml_graph_plan(
          स्थिर काष्ठा ggml_cgraph * cgraph,
                               पूर्णांक   n_threads,
            काष्ठा ggml_threadpool * threadpool) {

    अगर (threadpool == शून्य) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
    }
    अगर (n_threads <= 0) {
        n_threads = threadpool ? threadpool->n_threads : GGML_DEFAULT_N_THREADS;
    }

#अगर defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
    // Emscripten without pthreads support can only use a single thread
    n_threads = 1;
#पूर्ण_अगर

#अगर defined(__wasi__)
    // WASI doesn't support parallelism yet
    n_threads = 1;
#पूर्ण_अगर

    माप_प्रकार work_आकार = 0;

    काष्ठा ggml_cplan cplan;
    स_रखो(&cplan, 0, माप(काष्ठा ggml_cplan));

    पूर्णांक max_tasks = 1;

    // thread scheduling for the different operations + work buffer size estimation
    क्रम (पूर्णांक i = 0; i < cgraph->n_nodes; i++) {
        काष्ठा ggml_tensor * गाँठ = cgraph->nodes[i];

        स्थिर पूर्णांक n_tasks = ggml_get_n_tasks(गाँठ, n_threads);

        max_tasks = MAX(max_tasks, n_tasks);

        माप_प्रकार cur = 0;

        अगर (!ggml_cpu_extra_work_आकार(n_threads, गाँठ, &cur)) {
            चयन (गाँठ->op) {
                हाल GGML_OP_CPY:
                हाल GGML_OP_DUP:
                    {
                        अगर (ggml_is_quantized(गाँठ->type) ||
                            // F16 -> BF16 and BF16 -> F16 copies go through intermediate F32
                            (गाँठ->src[0]->type == GGML_TYPE_F16  && गाँठ->src[1] && गाँठ->src[1]->type == GGML_TYPE_BF16) ||
                            (गाँठ->src[0]->type == GGML_TYPE_BF16 && गाँठ->src[1] && गाँठ->src[1]->type == GGML_TYPE_F16) ||
                            // conversion between F32 and I32
                            (गाँठ->src[0]->type == GGML_TYPE_F32 && गाँठ->src[1] && गाँठ->src[1]->type == GGML_TYPE_I32) ||
                            (गाँठ->src[0]->type == GGML_TYPE_I32 && गाँठ->src[1] && गाँठ->src[1]->type == GGML_TYPE_F32)) {
                            cur = ggml_type_आकार(GGML_TYPE_F32) * गाँठ->ne[0] * n_tasks;
                        }
                    } अवरोध;
                हाल GGML_OP_ADD:
                हाल GGML_OP_ADD_ID:
                हाल GGML_OP_ADD1:
                    {
                        अगर (ggml_is_quantized(गाँठ->src[0]->type)) {
                            cur = ggml_type_आकार(GGML_TYPE_F32) * गाँठ->src[0]->ne[0] * n_tasks;
                        }
                    } अवरोध;
                हाल GGML_OP_ACC:
                    {
                        अगर (ggml_is_quantized(गाँठ->src[0]->type)) {
                            cur = ggml_type_आकार(GGML_TYPE_F32) * गाँठ->src[1]->ne[0] * n_tasks;
                        }
                    } अवरोध;
                हाल GGML_OP_गिनती_EQUAL:
                    {
                        cur = ggml_type_आकार(गाँठ->type)*n_tasks;
                    } अवरोध;
                हाल GGML_OP_MUL_MAT:
                    {
                        स्थिर क्रमागत ggml_type vec_dot_type = type_traits_cpu[गाँठ->src[0]->type].vec_dot_type;
                        अगर (गाँठ->src[1]->type != vec_dot_type) {
                            cur = ggml_row_आकार(vec_dot_type, ggml_nelements(गाँठ->src[1]));
                        }
                        // Workspace for tiled (see tiled.h)
                        cur = GGML_PAD(cur, 64);
                        cur += ggml_tiled_wdata_आकार(n_tasks, गाँठ);
                    } अवरोध;
                हाल GGML_OP_MUL_MAT_ID:
                    {
                        cur = 0;
                        स्थिर काष्ठा ggml_tensor * src0 = गाँठ->src[0];
                        स्थिर काष्ठा ggml_tensor * src1 = गाँठ->src[1];
                        स्थिर काष्ठा ggml_tensor * ids = गाँठ->src[2];
                        स्थिर क्रमागत ggml_type vec_dot_type = type_traits_cpu[src0->type].vec_dot_type;
                        स्थिर पूर्णांक n_as = src0->ne[2];
                        // src1
                        अगर (src1->type != vec_dot_type) {
                            cur += ggml_row_आकार(vec_dot_type, ggml_nelements(src1)) + माप(int64_t);
                        }
                        // matrix_row_counts
                        cur += n_as * माप(int64_t) + माप(int64_t);
                        // matrix_rows
                        cur += n_as*ids->ne[0]*ids->ne[1]*माप(काष्ठा mmid_row_mapping) + माप(int64_t);
                        // atomic_current_chunk
                        cur += CACHE_पंक्ति_आकार*n_as + CACHE_पंक्ति_आकार;
                        // Workspace for tiled (see tiled.h)
                        cur = GGML_PAD(cur, 64);
                        cur += ggml_tiled_wdata_आकार(n_tasks, गाँठ);
                    } अवरोध;
                हाल GGML_OP_OUT_PROD:
                    {
                        अगर (ggml_is_quantized(गाँठ->src[0]->type) ||
                            गाँठ->src[0]->type == GGML_TYPE_F16) {
                            cur = ggml_type_आकार(GGML_TYPE_F32) * गाँठ->src[0]->ne[0] * n_tasks;
                        }
                    } अवरोध;
                हाल GGML_OP_SET_ROWS:
                    {
                        अगर (गाँठ->src[0]->type == GGML_TYPE_F16 && गाँठ->type != GGML_TYPE_F16) {
                            cur = ggml_type_आकार(GGML_TYPE_F32) * गाँठ->src[0]->ne[0] * n_tasks;
                        }
                    } अवरोध;
                हाल GGML_OP_SOFT_MAX:
                हाल GGML_OP_ROPE:
                हाल GGML_OP_ROPE_BACK:
                    {
                        cur = ggml_type_आकार(GGML_TYPE_F32) * गाँठ->ne[0] * n_tasks;
                    } अवरोध;
                हाल GGML_OP_CONV_TRANSPOSE_1D:
                    {
                        GGML_ASSERT(गाँठ->src[0]->ne[3] == 1);
                        GGML_ASSERT(गाँठ->src[1]->ne[2] == 1);
                        GGML_ASSERT(गाँठ->src[1]->ne[3] == 1);

                        स्थिर int64_t ne00 = गाँठ->src[0]->ne[0];  // K
                        स्थिर int64_t ne01 = गाँठ->src[0]->ne[1];  // Cout
                        स्थिर int64_t ne02 = गाँठ->src[0]->ne[2];  // Cin
                        स्थिर int64_t ne10 = गाँठ->src[1]->ne[0];  // L
                        स्थिर int64_t ne11 = गाँठ->src[1]->ne[1];  // Cin

                        अगर ((गाँठ->src[0]->type == GGML_TYPE_F16 ||
                             गाँठ->src[0]->type == GGML_TYPE_BF16) &&
                            गाँठ->src[1]->type == GGML_TYPE_F32) {
                            cur += माप(ggml_fp16_t)*ne00*ne01*ne02;
                            cur += माप(ggml_fp16_t)*ne10*ne11;
                        } अन्यथा अगर (गाँठ->src[0]->type == GGML_TYPE_F32 &&
                                   गाँठ->src[1]->type == GGML_TYPE_F32) {
                            cur += माप(भग्न)*ne00*ne01*ne02;
                            cur += माप(भग्न)*ne10*ne11;
                        } अन्यथा {
                            GGML_ABORT("fatal error");
                        }
                    } अवरोध;
                हाल GGML_OP_CONV_2D:
                हाल GGML_OP_CONV_3D:
                    {
                        cur = GGML_IM2_COL_WORK_आकार;
                    } अवरोध;
                हाल GGML_OP_CONV_TRANSPOSE_2D:
                    {
                        स्थिर int64_t ne00 = गाँठ->src[0]->ne[0]; // W
                        स्थिर int64_t ne01 = गाँठ->src[0]->ne[1]; // H
                        स्थिर int64_t ne02 = गाँठ->src[0]->ne[2]; // Channels Out
                        स्थिर int64_t ne03 = गाँठ->src[0]->ne[3]; // Channels In

                        स्थिर int64_t ne10 = गाँठ->src[1]->ne[0]; // W
                        स्थिर int64_t ne11 = गाँठ->src[1]->ne[1]; // H
                        स्थिर int64_t ne12 = गाँठ->src[1]->ne[2]; // Channels In
                        स्थिर int64_t ne13 = गाँठ->src[1]->ne[3]; // Batch

                        GGML_ASSERT(गाँठ->src[0]->type == GGML_TYPE_F16 || गाँठ->src[0]->type == GGML_TYPE_F32);
                        GGML_ASSERT(गाँठ->src[1]->type == GGML_TYPE_F32);

                        cur += ggml_type_आकार(गाँठ->src[0]->type) * ne00 * ne01 * ne02 * ne03;
                        cur += ggml_type_आकार(गाँठ->src[0]->type) * ne10 * ne11 * ne12 * ne13;

                    } अवरोध;
                हाल GGML_OP_शीर्ष_K:
                    {
                        cur += माप(int32_t)*गाँठ->src[0]->ne[0]*n_tasks;
                    } अवरोध;
                हाल GGML_OP_FLASH_ATTN_EXT:
                    {
                        स्थिर int64_t neq2 = गाँठ->src[0]->ne[2]; // number of query heads
                        स्थिर int64_t DK = गाँठ->src[1]->ne[0];
                        स्थिर int64_t DV = गाँठ->src[2]->ne[0];

                        // Tiled flash attention scratch (tile sizes defined in common.h)
                        // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
                        माप_प्रकार prefill  = माप(भग्न)*(GGML_FA_TILE_Q*DK + 2*GGML_FA_TILE_Q*GGML_FA_TILE_KV + GGML_FA_TILE_Q*DV + GGML_FA_TILE_KV*DV + GGML_FA_TILE_KV*DK)*n_tasks;

                        // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
                        // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
                        माप_प्रकार n_chunks = n_tasks;
                        माप_प्रकार decode   = माप(भग्न)*(neq2*n_chunks*(2+DV) + n_tasks*(DK + 2*DV));

                        cur += MAX(prefill, decode);
                    } अवरोध;
                हाल GGML_OP_FLASH_ATTN_BACK:
                    {
                        स्थिर int64_t    D = गाँठ->src[0]->ne[0];
                        स्थिर int64_t ne11 = ggml_up(गाँठ->src[1]->ne[1], GGML_SOFT_MAX_UNROLL);
                        स्थिर int64_t mxDn = MAX(D, ne11) * 2; // *2 because of S and SM in ggml_compute_forward_flash_attn_back
                        अगर (गाँठ->src[1]->type == GGML_TYPE_F32) {
                            cur  = माप(भग्न)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += माप(भग्न)*mxDn*n_tasks; // this is overestimated by x2
                        } अन्यथा अगर (गाँठ->src[1]->type == GGML_TYPE_F16) {
                            cur  = माप(भग्न)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += माप(भग्न)*mxDn*n_tasks; // this is overestimated by x2
                        } अन्यथा अगर (गाँठ->src[1]->type == GGML_TYPE_BF16) {
                            cur  = माप(भग्न)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += माप(भग्न)*mxDn*n_tasks; // this is overestimated by x2
                        }
                    } अवरोध;

                हाल GGML_OP_CROSS_ENTROPY_LOSS:
                    {
                        cur = ggml_type_आकार(गाँठ->type)*(n_tasks + गाँठ->src[0]->ne[0]*n_tasks);
                    } अवरोध;
                हाल GGML_OP_GATED_DELTA_NET:
                    {
                        स्थिर int64_t S_v = गाँठ->src[2]->ne[0];
                        स्थिर int64_t K   = ggml_get_op_params_i32(गाँठ, 0);
                        स्थिर int64_t per_thread = S_v + (K > 1 ? S_v * S_v : 0);
                        cur = per_thread * माप(भग्न) * n_tasks;
                    } अवरोध;
                हाल GGML_OP_गिनती:
                    {
                        GGML_ABORT("fatal error");
                    }
                हाल GGML_OP_LIGHTNING_INDEXER:
                    {
                        // temp buffer for dequantizing lightning indexer keys
                        स्थिर int64_t ne10 = गाँठ->src[1]->ne[0];
                        cur += माप(भग्न)*ne10*n_tasks;
                    } अवरोध;
                शेष:
                    अवरोध;
            }
        }

        work_आकार = MAX(work_आकार, cur);
    }

    अगर (work_आकार > 0) {
        work_आकार += CACHE_पंक्ति_आकार*(n_threads);
    }

    cplan.threadpool = threadpool;
    cplan.n_threads  = MIN(max_tasks, n_threads);
    cplan.work_आकार  = work_आकार;
    cplan.work_तथ्य  = शून्य;

    वापस cplan;
}


// Try to fuse the current node with subsequent nodes for better performance.
// Returns the number of nodes skipped by fusion (>=1), or 0 if no fusion was applied.
अटल bool ggml_cpu_disable_fusion = false;  // initialized once in ggml_cpu_init(), read-only afterwards

अटल पूर्णांक ggml_cpu_try_fuse_ops(
        स्थिर काष्ठा ggml_cgraph * cgraph,
        स्थिर पूर्णांक गाँठ_n,
        स्थिर काष्ठा ggml_compute_params * params,
        स्थिर काष्ठा ggml_cplan * cplan) {

    अगर (ggml_cpu_disable_fusion || cplan->use_ref) {
        वापस 0;
    }

    काष्ठा ggml_tensor * गाँठ = cgraph->nodes[गाँठ_n];

    अगर (गाँठ->op == GGML_OP_RMS_NORM) {
        // RMS_NORM + MUL fusion
        स्थिर क्रमागत ggml_op fuse_ops[] = { GGML_OP_RMS_NORM, GGML_OP_MUL };
        अगर (ggml_can_fuse(cgraph, गाँठ_n, fuse_ops, 2)) {
            काष्ठा ggml_tensor * mul_गाँठ = cgraph->nodes[गाँठ_n + 1];
            स्थिर काष्ठा ggml_tensor * mul_w = (mul_गाँठ->src[0] == गाँठ)
                ? mul_गाँठ->src[1] : mul_गाँठ->src[0];
            अगर (गाँठ->src[0]->type  == GGML_TYPE_F32 &&
                mul_गाँठ->type      == GGML_TYPE_F32 &&
                mul_w->type         == GGML_TYPE_F32 &&
                mul_w->ne[0]        == गाँठ->ne[0]   &&
                mul_w->nb[0]        == माप(भग्न)) {

                ggml_compute_forward_rms_norm_mul_fused(params, गाँठ, mul_गाँठ);
                वापस 1;
            }
        }
    }

    वापस 0;
}

अटल thread_ret_t ggml_graph_compute_thread(व्योम * तथ्य) {
    काष्ठा ggml_compute_state * state = (काष्ठा ggml_compute_state *) तथ्य;
    काष्ठा ggml_threadpool    * tp    = state->threadpool;

    स्थिर काष्ठा ggml_cgraph * cgraph = tp->cgraph;
    स्थिर काष्ठा ggml_cplan  * cplan  = tp->cplan;

#अगर_घोषित GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(state->ith);
#अन्यथा
    set_numa_thread_affinity(state->ith);
#पूर्ण_अगर

    काष्ठा ggml_compute_params params = {
        /*.ith        =*/ state->ith,
        /*.nth        =*/ atomic_load_explicit(&tp->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK,
        /*.wsize      =*/ cplan->work_आकार,
        /*.wdata      =*/ cplan->work_तथ्य,
        /*.threadpool =*/ tp,
        /*.use_ref    =*/ cplan->use_ref,
    };

#अगर_घोषित GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p\n", state->ith, (स्थिर व्योम *)cplan);
#अन्यथा
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p last-graph %d\n", state->ith, (स्थिर व्योम *)cplan, state->last_graph);
#पूर्ण_अगर

    क्रम (पूर्णांक गाँठ_n = 0; गाँठ_n < cgraph->n_nodes && atomic_load_explicit(&tp->पात, memory_order_relaxed) != गाँठ_n; गाँठ_n++) {
        काष्ठा ggml_tensor * गाँठ = cgraph->nodes[गाँठ_n];

        अगर (ggml_op_is_empty(गाँठ->op)) {
            // skip NOPs
            जारी;
        }

        अगर ((गाँठ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            जारी;
        }

        // TODO: move fused-op detection into ggml_graph_plan so fusion decisions are made once at planning time
        // Try fused ops, fall back to normal compute
        स्थिर पूर्णांक n_fused = ggml_cpu_try_fuse_ops(cgraph, गाँठ_n, &params, cplan);
        अगर (n_fused > 0) {
            गाँठ_n += n_fused;
        } अन्यथा {
            ggml_compute_forward(&params, गाँठ);
        }

        अगर (state->ith == 0 && cplan->abort_callback &&
                cplan->abort_callback(cplan->abort_callback_तथ्य)) {
            atomic_store_explicit(&tp->पात, गाँठ_n + 1, memory_order_relaxed);
            tp->ec    = GGML_STATUS_ABORTED;
        }

        अगर (गाँठ_n + 1 < cgraph->n_nodes) {
            ggml_barrier(state->threadpool);
        }
    }

#अगर_घोषित GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p\n", state->ith, (स्थिर व्योम *)cplan);
#अन्यथा
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p last-graph %d\n", state->ith, (स्थिर व्योम *)cplan, state->last_graph);
#पूर्ण_अगर

    ggml_barrier(state->threadpool);

#अगर_घोषित GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(state->ith);
#पूर्ण_अगर

    वापस 0;
}

#अगर_अघोषित GGML_USE_OPENMP

// check if thread is ready to proceed (exit from polling or sleeping)
// returns true if loops should exit, sets state->pending to indicate new work
अटल अंतरभूत bool ggml_graph_compute_thread_ready(काष्ठा ggml_compute_state * state) {
    काष्ठा ggml_threadpool * threadpool = state->threadpool;

    अगर (state->pending || threadpool->stop || threadpool->छोड़ो) { वापस true; }

    // check for new graph/work
    पूर्णांक n_graph   = atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed);
    पूर्णांक n_threads = n_graph & GGML_THREADPOOL_N_THREADS_MASK;
    अगर (n_graph != state->last_graph) {
        state->pending    = (state->ith < n_threads);
        state->last_graph = n_graph;
        वापस true;
    }

    वापस false;
}

// sync thread state after polling
अटल अंतरभूत व्योम ggml_graph_compute_thread_sync(काष्ठा ggml_compute_state * state) {
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #अगर_घोषित GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&state->threadpool->n_graph, 0, memory_order_seq_cst);
    #अन्यथा
    atomic_thread_fence(memory_order_seq_cst);
    #पूर्ण_अगर
    UNUSED(state);
}

अटल अंतरभूत bool ggml_graph_compute_poll_for_work(काष्ठा ggml_compute_state * state) {
    काष्ठा ggml_threadpool * threadpool = state->threadpool;

    // This seems to make 0 ... 100 a decent range for polling level across modern processors.
    // Perhaps, we can adjust it dynamically based on load and things.
    स्थिर uint64_t n_rounds = 1024UL * 128 * threadpool->poll;

    क्रम (uint64_t i=0; !ggml_graph_compute_thread_ready(state) && i < n_rounds; i++) {
        // No new work. Keep polling.
        ggml_thread_cpu_relax();
    }

    वापस state->pending;
}

अटल अंतरभूत bool ggml_graph_compute_check_for_work(काष्ठा ggml_compute_state * state) {
    काष्ठा ggml_threadpool * threadpool = state->threadpool;

    अगर (ggml_graph_compute_poll_for_work(state)) {
        ggml_graph_compute_thread_sync(state);
        वापस state->pending;
    }

    ggml_mutex_lock_shared(&threadpool->mutex);
    जबतक (!ggml_graph_compute_thread_ready(state)) {
        // No new work. Wait for the signal.
        GGML_PRINT_DEBUG("thread #%d waiting for work (sleeping)\n", state->ith);
        ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
    }
    ggml_mutex_unlock_shared(&threadpool->mutex);

    वापस state->pending;
}

अटल thread_ret_t ggml_graph_compute_secondary_thread(व्योम* तथ्य) {
    काष्ठा ggml_compute_state * state = (काष्ठा ggml_compute_state *) तथ्य;
    काष्ठा ggml_threadpool * threadpool = state->threadpool;

    ggml_thread_apply_priority(threadpool->prio);
    अगर (ggml_thread_cpumask_is_valid(state->cpumask)) {
        ggml_thread_apply_affinity(state->cpumask);
    }

    जबतक (true) {
        // Check if we need to sleep
        जबतक (threadpool->छोड़ो) {
            GGML_PRINT_DEBUG("thread #%d inside pause loop\n", state->ith);
            ggml_mutex_lock_shared(&threadpool->mutex);
            अगर (threadpool->छोड़ो) {
                ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
            }
            GGML_PRINT_DEBUG("thread #%d resuming after wait\n", state->ith);
            ggml_mutex_unlock_shared(&threadpool->mutex);
        }

        // This needs to be checked for after the cond_wait
        अगर (threadpool->stop) अवरोध;

        // Check if there is new work
        // The main thread is the only one that can dispatch new work

        ggml_graph_compute_check_for_work(state);
        अगर (state->pending) {
            state->pending = false;
            ggml_graph_compute_thread(state);
        }
    }

    वापस (thread_ret_t) 0;
}

// Start processing new graph
अटल व्योम ggml_graph_compute_kickoff(काष्ठा ggml_threadpool * threadpool, पूर्णांक n_threads)
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

    अगर (threadpool->छोड़ो) {
       // Update main thread prio and affinity to match the threadpool settings
       ggml_thread_apply_priority(threadpool->prio);
       अगर (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
           ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
       }

       // resume does cond broadcast
       ggml_threadpool_resume_locked(threadpool);
    } अन्यथा {
       ggml_cond_broadcast(&threadpool->cond);
    }

    ggml_mutex_unlock(&threadpool->mutex);
}

#पूर्ण_अगर // GGML_USE_OPENMP

अटल काष्ठा ggml_threadpool * ggml_threadpool_new_impl(
    काष्ठा ggml_threadpool_params * tpp,
               काष्ठा ggml_cgraph * cgraph,
                काष्ठा ggml_cplan * cplan) {

    काष्ठा ggml_threadpool * threadpool =
        ggml_aligned_malloc(माप(काष्ठा ggml_threadpool));
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
        threadpool->workers          = शून्य;
        threadpool->n_threads        = tpp->n_threads;
        threadpool->poll             = tpp->poll;
        threadpool->prio             = tpp->prio;
        threadpool->ec               = GGML_STATUS_SUCCESS;
    }

    // Allocate and init workers state
    स्थिर माप_प्रकार workers_आकार = माप(काष्ठा ggml_compute_state) * tpp->n_threads;
    काष्ठा ggml_compute_state * workers = ggml_aligned_malloc(workers_आकार);

    स_रखो(workers, 0, workers_आकार);
    क्रम (पूर्णांक j = 0; j < tpp->n_threads; j++) {
        workers[j].threadpool = threadpool;
        workers[j].ith        = j;
    }

    threadpool->workers = workers;

#अगर_घोषित GGML_USE_OPENMP
    int32_t cpumask_iter = 0;

    // Compute CPU masks for each thread
    क्रम (पूर्णांक j = 0; j < tpp->n_threads; j++) {
        ggml_thread_cpumask_अग्र(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);
    }
#अन्यथा // GGML_USE_OPENMP
    ggml_mutex_init(&threadpool->mutex);
    ggml_cond_init(&threadpool->cond);

    // Spin the threads for all workers, and update CPU placements.
    // Place the main thread last (towards the higher numbered CPU cores).

    int32_t cpumask_iter = 0;

    क्रम (पूर्णांक j = 1; j < tpp->n_threads; j++) {
        ggml_thread_cpumask_अग्र(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);

        int32_t rc = ggml_thread_create(&workers[j].thrd, शून्य, ggml_graph_compute_secondary_thread, &workers[j]);
        GGML_ASSERT(rc == 0);
    }

    ggml_thread_cpumask_अग्र(tpp->cpumask, workers[0].cpumask, tpp->strict_cpu, &cpumask_iter);

    अगर (!threadpool->छोड़ो) {
        // Update main thread prio and affinity at the start, otherwise we'll do it in resume
        ggml_thread_apply_priority(threadpool->prio);
        अगर (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
            ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
        }
    }
#पूर्ण_अगर // GGML_USE_OPENMP

    वापस threadpool;
}

काष्ठा ggml_threadpool * ggml_threadpool_new(काष्ठा ggml_threadpool_params * tpp) {
    वापस ggml_threadpool_new_impl(tpp, शून्य, शून्य);
}

क्रमागत ggml_status ggml_graph_compute(काष्ठा ggml_cgraph * cgraph, काष्ठा ggml_cplan * cplan) {
    ggml_cpu_init();

    GGML_ASSERT(cplan);
    GGML_ASSERT(cplan->n_threads > 0);
    GGML_ASSERT(cplan->work_आकार == 0 || cplan->work_तथ्य != शून्य);

    पूर्णांक n_threads                               = cplan->n_threads;
    काष्ठा ggml_threadpool * threadpool = cplan->threadpool;

    bool disposable_threadpool = false;

    अगर (threadpool == शून्य) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
        disposable_threadpool = true;

        काष्ठा ggml_threadpool_params ttp = ggml_threadpool_params_default(n_threads);
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

#अगर_घोषित GGML_USE_OPENMP
    अगर (n_threads > 1) {
        #आशय omp parallel अंक_threads(n_threads)
        {
            #आशय omp single
            {
                // update the number of threads from the actual number of threads that we got from OpenMP
                n_threads = omp_get_अंक_threads();
                atomic_store_explicit(&threadpool->n_graph, n_threads, memory_order_relaxed);
            }

            // Apply thread CPU mask and priority
            पूर्णांक ith = omp_get_thread_अंक();

            ggml_thread_apply_priority(threadpool->prio);
            अगर (ggml_thread_cpumask_is_valid(threadpool->workers[ith].cpumask)) {
                ggml_thread_apply_affinity(threadpool->workers[ith].cpumask);
            }
            ggml_graph_compute_thread(&threadpool->workers[ith]);
        }
    } अन्यथा {
        atomic_store_explicit(&threadpool->n_graph, 1, memory_order_relaxed);
        ggml_graph_compute_thread(&threadpool->workers[0]);
    }
#अन्यथा
    अगर (n_threads > threadpool->n_threads) {
        GGML_LOG_WARN("cplan requested more threads (%d) than available (%d)\n", n_threads, threadpool->n_threads);
        n_threads = threadpool->n_threads;
    }

    // Kick all threads to start the new graph
    ggml_graph_compute_kickoff(threadpool, n_threads);

    // This is a work thread too
    ggml_graph_compute_thread(&threadpool->workers[0]);
#पूर्ण_अगर

    // don't leave affinity set on the main thread
    clear_numa_thread_affinity();

    क्रमागत ggml_status ret = threadpool->ec;

    अगर (disposable_threadpool) {
        ggml_threadpool_free(threadpool);
    }

    वापस ret;
}

क्रमागत ggml_status ggml_graph_compute_with_ctx(काष्ठा ggml_context * ctx, काष्ठा ggml_cgraph * cgraph, पूर्णांक n_threads) {
    काष्ठा ggml_cplan cplan = ggml_graph_plan(cgraph, n_threads, शून्य);

    cplan.work_तथ्य = (uint8_t *)ggml_new_बफर(ctx, cplan.work_आकार);

    वापस ggml_graph_compute(cgraph, &cplan);
}

व्योम ggml_cpu_fp32_to_fp32(स्थिर भग्न * x, भग्न * y, int64_t n) {
    स_नकल(y, x, n * माप(भग्न));
}

व्योम ggml_cpu_fp32_to_fp16(स्थिर भग्न * x, ggml_fp16_t * y, int64_t n) {
    int64_t i = 0;
#अगर defined(__F16C__)
#अगर defined(__AVX512F__)
    क्रम (; i + 15 < n; i += 16) {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        __m256i y_vec = _mm512_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256((__m256i *)(y + i), y_vec);
    }
#पूर्ण_अगर
    क्रम (; i + 7 < n; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m128i y_vec = _mm256_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i *)(y + i), y_vec);
    }
    क्रम (; i + 3 < n; i += 4) {
        __m128 x_vec = _mm_loadu_ps(x + i);
        __m128i y_vec = _mm_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storel_epi64((__m128i *)(y + i), y_vec);
    }
#या_अगर defined(__riscv_zvfh)
    क्रम (पूर्णांक vl; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(&x[i], vl);
        vfloat16m1_t vy = __riscv_vfncvt_f_f_w_f16m1(vx, vl);
        __riscv_vse16_v_f16m1((_Float16 *)&y[i], vy, vl);
    }
#या_अगर defined(__VXE__) || defined(__VXE2__)
    क्रम (; i + 7 < n; i += 8) {
        स्थिर uint32x4_t v_yl = __lzs_f32cx4_to_f16(vec_xl(0, x + i + 0));
        स्थिर uint32x4_t v_yh = __lzs_f32cx4_to_f16(vec_xl(0, x + i + 4));
        vec_xst(vec_pack(v_yl, v_yh), 0, (uint16_t *)(y + i));
    }
#पूर्ण_अगर
    क्रम (; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(x[i]);
    }
}

व्योम ggml_cpu_fp16_to_fp32(स्थिर ggml_fp16_t * x, भग्न * y, int64_t n) {
    int64_t i = 0;
#अगर defined(__F16C__)
#अगर defined(__AVX512F__)
    क्रम (; i + 15 < n; i += 16) {
        __m256i x_vec = _mm256_loadu_si256((स्थिर __m256i *)(x + i));
        __m512 y_vec = _mm512_cvtph_ps(x_vec);
        _mm512_storeu_ps(y + i, y_vec);
    }
#पूर्ण_अगर
    क्रम (; i + 7 < n; i += 8) {
        __m128i x_vec = _mm_loadu_si128((स्थिर __m128i *)(x + i));
        __m256 y_vec = _mm256_cvtph_ps(x_vec);
        _mm256_storeu_ps(y + i, y_vec);
    }
    क्रम (; i + 3 < n; i += 4) {
        __m128i x_vec = _mm_loadl_epi64((स्थिर __m128i *)(x + i));
        __m128 y_vec = _mm_cvtph_ps(x_vec);
        _mm_storeu_ps(y + i, y_vec);
    }

#या_अगर defined(__riscv_v_intrinsic) && defined(__riscv_zvfhmin)
    // calculate step size
    स्थिर पूर्णांक epr = __riscv_vsetvlmax_e16m2();
    स्थिर पूर्णांक step = epr * 2;
    स्थिर पूर्णांक np = (n & ~(step - 1));

    // unroll by 2
    क्रम (; i < np; i += step) {
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((स्थिर _Float16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vfloat16m2_t ax1 = __riscv_vle16_v_f16m2((स्थिर _Float16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvt_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    पूर्णांक vl;
    क्रम (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((स्थिर _Float16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }

#पूर्ण_अगर

    क्रम (; i < n; ++i) {
        y[i] = GGML_CPU_FP16_TO_FP32(x[i]);
    }
}

व्योम ggml_cpu_fp32_to_bf16(स्थिर भग्न * x, ggml_bf16_t * y, int64_t n) {
    int64_t i = 0;
    क्रम (; i < n; ++i) {
        y[i] = GGML_FP32_TO_BF16(x[i]);
    }
}

व्योम ggml_cpu_fp32_to_i32(स्थिर भग्न * x, int32_t * y, int64_t n) {
    int64_t i = 0;
    क्रम (; i < n; ++i) {
        y[i] = x[i];
    }
}

व्योम ggml_cpu_bf16_to_fp32(स्थिर ggml_bf16_t * x, भग्न * y, int64_t n) {
    int64_t i = 0;
#अगर defined(__AVX2__)
#अगर defined(__AVX512F__)
    क्रम (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i,
                        _mm512_castsi512_ps(
                            _mm512_slli_epi32(
                                _mm512_cvtepu16_epi32(
                                    _mm256_loadu_si256(
                                        (स्थिर __m256i *)(x + i))),
                                16)));
    }
#पूर्ण_अगर
    क्रम (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i,
                        _mm256_castsi256_ps(
                            _mm256_slli_epi32(
                                _mm256_cvtepu16_epi32(
                                    _mm_loadu_si128(
                                        (स्थिर __m128i *)(x + i))),
                                16)));
    }
#या_अगर defined(__riscv_v_intrinsic) && defined(__riscv_zvfbfmin)
    // calculate step size
    स्थिर पूर्णांक epr = __riscv_vsetvlmax_e16m2();
    स्थिर पूर्णांक step = epr * 2;
    स्थिर पूर्णांक np = (n & ~(step - 1));

    // unroll by 2
    क्रम (; i < np; i += step) {
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((स्थिर __bf16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vbfloat16m2_t ax1 = __riscv_vle16_v_bf16m2((स्थिर __bf16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    पूर्णांक vl;
    क्रम (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((स्थिर __bf16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }
#पूर्ण_अगर
    क्रम (; i < n; i++) {
        y[i] = GGML_BF16_TO_FP32(x[i]);
    }
}

पूर्णांक ggml_cpu_has_avx(व्योम) {
#अगर defined(__AVX__)
    वापस 1;
#अन्यथा
    वापस 0;
#पूर्ण_अगर
}

पूर्णांक ggml_cpu_has_avx_vnni(व्योम) {
#अगर defined(__AVXVNNI__)
    वापस 1;
#अन्यथा
    वापस 0;
#पूर्ण_अगर
}

पूर्णांक ggml_cpu_has_avx2(व्योम) {
#अगर defined(__AVX2__)
    वापस 1;
#अन्यथा
    वापस 0;
#पूर्ण_अगर
}

पूर्णांक ggml_cpu_has_avx512(व्योम) {
#अगर defined(__AVX512F__)
    वापस 1;
#अन्यथा
    वापस 0;
#पूर्ण_अगर
}

पूर्णांक ggml_cpu_has_avx512_vbmi(व्योम) {
#अगर defined(__AVX512VBMI__)
    वापस 1;
#अन्यथा
    वापस 0;
#पूर्ण_अगर
}

पूर्णांक ggml_cpu_has_avx512_vnni(व्योम) {
#अगर defined(__AVX512VNNI__)
    वापस 1;
#अन्यथा
    वापस 0;
#पूर्ण_अगर
}

पूर्णांक ggml_cpu_has_avx512_bf16(व्योम) {
#अगर defined(__AVX512BF16__)
    वापस 1;
#अन्यथा
    वापस 0;
#पूर्ण_अगर
}

पूर्णांक ggml_cpu_has_amx_int8(व्योम) {
#अगर defined(__AMX_INT8__)
    वापस 1;
#अन्यथा
    वापस 0;
#पूर्ण_अगर
}

पूर्णांक ggml_cpu_has_bmi2(व्योम) {
#अगर defined(__BMI2__)
    वापस 1;
#अन्यथा
    वापस 0;
#पूर्ण_अगर
}

पूर्णांक ggml_cpu_has_fma(व्योम) {
#अगर defined(__FMA__)
    वापस 1;
#अन्यथा
    वापस 0;
#पूर्ण_अगर
}

पूर्णांक ggml_cpu_has_arm_fma(व्योम) {
#अगर defined(__ARM_FEATURE_FMA)
    वापस 1;
#अन्यथा
    वापस 0;
#पूर्ण_अगर
}

पूर्णांक ggml_cpu_has_riscv_v(व्योम) {
#अगर defined(__riscv_v_intrinsic)
    वापस 1;
#अन्यथा
    वापस 0;
#पूर्ण_अगर
}

पूर्णांक ggml_cpu_get_rvv_vlen(व्योम) {
#अगर defined(__riscv) && defined(__riscv_v_intrinsic)
    वापस ggml_riscv_arch_features.rvv_vlen;
#अन्यथा
    वापस 0;
#पूर्ण_अगर
}

पूर्णांक ggml_cpu_has_f16c(व्योम) {
#अगर defined(__F16C__)
    वापस 1;
#अन्यथा
    वापस 0;
#पूर्ण_अगर
}

पूर्णांक ggml_cpu_has_fp16_va(व्योम) {
#अगर defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    वापस 1;
#अन्यथा
    वापस 0;
#पूर्ण_अगर
}

पूर्णांक ggml_cpu_has_wasm_simd(व्योम) {
#अगर defined(__wasm_simd128__)
    वापस 1;
#अन्यथा
    वापस 0;
#पूर्ण_अगर
}

पूर्णांक ggml_cpu_has_llamafile(व्योम) {
#अगर defined(GGML_USE_LLAMAFILE)
    वापस 1;
#अन्यथा
    वापस 0;
#पूर्ण_अगर
}

पूर्णांक ggml_cpu_has_sse3(व्योम) {
#अगर defined(__SSE3__)
    वापस 1;
#अन्यथा
    वापस 0;
#पूर्ण_अगर
}

पूर्णांक ggml_cpu_has_ssse3(व्योम) {
#अगर defined(__SSSE3__)
    वापस 1;
#अन्यथा
    वापस 0;
#पूर्ण_अगर
}

पूर्णांक ggml_cpu_has_vsx(व्योम) {
#अगर defined(__POWER9_VECTOR__)
    वापस 1;
#अन्यथा
    वापस 0;
#पूर्ण_अगर
}

पूर्णांक ggml_cpu_has_vxe(व्योम) {
#अगर defined(__VXE__) || defined(__VXE2__)
    वापस 1;
#अन्यथा
    वापस 0;
#पूर्ण_अगर
}

पूर्णांक ggml_cpu_has_neon(व्योम) {
#अगर defined(__ARM_ARCH) && defined(__ARM_NEON)
    वापस 1;
#अन्यथा
    वापस 0;
#पूर्ण_अगर
}

पूर्णांक ggml_cpu_has_dotprod(व्योम) {
#अगर defined(__ARM_ARCH) && defined(__ARM_FEATURE_DOTPROD)
    वापस 1;
#अन्यथा
    वापस 0;
#पूर्ण_अगर
}

पूर्णांक ggml_cpu_has_sve(व्योम) {
#अगर defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    वापस 1;
#अन्यथा
    वापस 0;
#पूर्ण_अगर
}

पूर्णांक ggml_cpu_has_matmul_int8(व्योम) {
#अगर defined(__ARM_ARCH) && defined(__ARM_FEATURE_MATMUL_INT8)
    वापस 1;
#अन्यथा
    वापस 0;
#पूर्ण_अगर
}

पूर्णांक ggml_cpu_get_sve_cnt(व्योम) {
#अगर defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    वापस ggml_arm_arch_features.sve_cnt;
#अन्यथा
    वापस 0;
#पूर्ण_अगर
}

पूर्णांक ggml_cpu_has_sme(व्योम) {
#अगर defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME)
    वापस 1;
#अन्यथा
    वापस 0;
#पूर्ण_अगर
}

पूर्णांक ggml_cpu_has_sme2(व्योम) {
#अगर defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME2)
    वापस 1;
#अन्यथा
    वापस 0;
#पूर्ण_अगर
}

व्योम ggml_cpu_init(व्योम) {
    // needed to initialize ggml_time
    {
        काष्ठा ggml_init_params params = { 0, शून्य, false };
        काष्ठा ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    ggml_critical_section_start();

    अटल bool is_first_call = true;

    अगर (is_first_call) {
        // initialize GELU, Quick GELU, SILU and EXP F32 tables
        {
            स्थिर uint64_t t_start = ggml_time_us(); UNUSED(t_start);

            क्रम (पूर्णांक i = 0; i < (1 << 16); ++i) {
                जोड़ {
                    uint16_t u16;
                    ggml_fp16_t fp16;
                } u = {i};
                भग्न f = GGML_COMPUTE_FP16_TO_FP32(u.fp16);
                ggml_table_f32_f16[i] = f;
                ggml_table_gelu_f16[i] = GGML_CPU_FP32_TO_FP16(ggml_gelu_f32(f));
                ggml_table_gelu_quick_f16[i] = GGML_CPU_FP32_TO_FP16(ggml_gelu_quick_f32(f));
            }

            // initialize E8M0 half table (256 entries)
            क्रम (पूर्णांक i = 0; i < (1 << 8); ++i) {
                ggml_table_f32_e8m0_half[i] = GGML_E8M0_TO_FP32_HALF(i);
            }

            // initialize UE4M3 table (256 entries)
            क्रम (पूर्णांक i = 0; i < (1 << 8); ++i) {
                ggml_table_f32_ue4m3[i] = ggml_ue4m3_to_fp32(i);
            }

            स्थिर uint64_t t_end = ggml_time_us(); UNUSED(t_end);

            GGML_PRINT_DEBUG("%s: GELU, Quick GELU, SILU and EXP tables initialized in %f ms\n", __func__, (t_end - t_start)/1000.0);

#अगर_घोषित GGML_USE_OPENMP
            //if (!getenv("OMP_WAIT_POLICY")) {
            //    // set the wait policy to active, so that OpenMP threads don't sleep
            //    setenv("OMP_WAIT_POLICY", "active", 0)
            //}

            अगर (!दो_पर्या("KMP_BLOCKTIME")) {
                // set the time to wait before sleeping a thread
                // this is less aggressive than setting the wait policy to active, but should achieve similar results in most cases
#अगर_घोषित _WIN32
                _putenv_s("KMP_BLOCKTIME", "200"); // 200ms
#अन्यथा
                setenv("KMP_BLOCKTIME", "200", 0); // 200ms
#पूर्ण_अगर
            }
#पूर्ण_अगर
        }

#अगर defined(__ARM_ARCH)
        ggml_init_arm_arch_features();
#पूर्ण_अगर

#अगर defined(__riscv)
        ggml_init_riscv_arch_features();
#पूर्ण_अगर

        {
            स्थिर अक्षर * env = दो_पर्या("GGML_CPU_DISABLE_FUSION");
            ggml_cpu_disable_fusion = (env != शून्य && म_से_प(env) == 1);
        }

        is_first_call = false;
    }

    ggml_critical_section_end();
}
