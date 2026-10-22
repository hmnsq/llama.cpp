#define _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#define _USE_MATH_DEFINES // For M_PI on MSVC

#અન્તર્ભાવ "ggml-backend-impl.h"
#અન્તર્ભાવ "ggml-backend.h"
#અન્તર્ભાવ "traits.h"
#અન્તર્ભાવ "ggml-cpu-impl.h"
#અન્તર્ભાવ "ggml-impl.h"
#અન્તર્ભાવ "quants.h"
#અન્તર્ભાવ "ggml-threading.h"
#અન્તર્ભાવ "unary-ops.h"
#અન્તર્ભાવ "binary-ops.h"
#અન્તર્ભાવ "vec.h"
#અન્તર્ભાવ "ops.h"
#અન્તર્ભાવ "ggml.h"
#અન્તર્ભાવ "common.h"
#અન્તર્ભાવ "tiled/tiled.h"

#જો defined(_MSC_VER) || defined(__MINGW32__)
#અન્તર્ભાવ <malloc.h> // using malloc.h with MSC/MINGW
#અથવાજો !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#અન્તર્ભાવ <alloca.h>
#endif

#અન્તર્ભાવ <assert.h>
#અન્તર્ભાવ <errno.h>
#અન્તર્ભાવ <time.h>
#અન્તર્ભાવ <math.h>
#અન્તર્ભાવ <stdlib.h>
#અન્તર્ભાવ <string.h>
#અન્તર્ભાવ <stdint.h>
#અન્તર્ભાવ <inttypes.h>
#અન્તર્ભાવ <stdio.h>
#અન્તર્ભાવ <float.h>
#અન્તર્ભાવ <limits.h>
#અન્તર્ભાવ <stdarg.h>
#અન્તર્ભાવ <signal.h>
#જો defined(__gnu_linux__)
#અન્તર્ભાવ <syscall.h>
#endif

#ifdef GGML_USE_OPENMP
#અન્તર્ભાવ <omp.h>
#endif

#જો defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_MATMUL_INT8)
#undef GGML_USE_LLAMAFILE
#endif

#ifdef GGML_USE_LLAMAFILE
#અન્તર્ભાવ "llamafile/sgemm.h"
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
#    include "spacemit/ime.h"
#endif

// Note: once we move threading into a separate C++ file
// will use std::hardware_destructive_interference_size instead of hardcoding it here
// and we'll use C++ attribute syntax.
#define GGML_CACHE_લીટી  64

#જો defined(__clang__) || defined(__GNUC__)
#define GGML_CACHE_ALIGN __attribute__((aligned(GGML_CACHE_લીટી)))
#endif

#જો defined(__has_feature)
#જો __has_feature(thread_sanitizer)
#define GGML_TSAN_ENABLED 1
#endif
#નહીંતર  // __has_feature
#જો defined(__SANITIZE_THREAD__)
#define GGML_TSAN_ENABLED 1
#endif
#endif // __has_feature

#define UNUSED GGML_UNUSED
#define SWAP(x, y, T) કરો { T SWAP = x; (x) = y; (y) = SWAP; } જ્યારે (0)

// precomputed f32 table for f16 (256 KB) (simd-mappings.h)
દશાંશ ggml_table_f32_f16[1 << 16];

// precomputed f32 table for e8m0 half (1 KB) (simd-mappings.h)
દશાંશ ggml_table_f32_e8m0_half[1 << 8];

// precomputed f32 table for ue4m3 (1 KB) (simd-mappings.h)
દશાંશ ggml_table_f32_ue4m3[1 << 8];

#જો defined(__ARM_ARCH)
સંરચના ggml_arm_arch_features_type {
    પૂર્ણાંક sve_cnt;
} ggml_arm_arch_features = { 0 };
#endif

#જો defined(__riscv)
સંરચના ggml_riscv_arch_features_type {
    પૂર્ણાંક rvv_vlen;
} ggml_riscv_arch_features = { 0 };
#endif

#જો defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#અન્તર્ભાવ <windows.h>

#જો defined(_MSC_VER) && !defined(__clang__)
#define GGML_CACHE_ALIGN __declspec(align(GGML_CACHE_લીટી))

પ્રકારવ્યાખ્યા અસ્થિર LONG atomic_int;
પ્રકારવ્યાખ્યા atomic_int atomic_bool;
પ્રકારવ્યાખ્યા atomic_int atomic_flag;

#define ATOMIC_FLAG_INIT 0

પ્રકારવ્યાખ્યા ગણના {
    memory_order_relaxed,
    memory_order_consume,
    memory_order_acquire,
    memory_order_release,
    memory_order_acq_rel,
    memory_order_seq_cst
} memory_order;

સ્થિર શૂન્ય atomic_store(atomic_int * ptr, LONG val) {
    InterlockedExchange(ptr, val);
}
સ્થિર શૂન્ય atomic_store_explicit(atomic_int * ptr, LONG val, memory_order mo) {
    // TODO: add support for explicit memory order
    InterlockedExchange(ptr, val);
}
સ્થિર LONG atomic_load(atomic_int * ptr) {
    પાછા InterlockedCompareExchange(ptr, 0, 0);
}
સ્થિર LONG atomic_load_explicit(atomic_int * ptr, memory_order mo) {
    // TODO: add support for explicit memory order
    પાછા InterlockedCompareExchange(ptr, 0, 0);
}
સ્થિર LONG atomic_fetch_add(atomic_int * ptr, LONG inc) {
    પાછા InterlockedExchangeAdd(ptr, inc);
}
સ્થિર LONG atomic_fetch_add_explicit(atomic_int * ptr, LONG inc, memory_order mo) {
    // TODO: add support for explicit memory order
    પાછા InterlockedExchangeAdd(ptr, inc);
}
સ્થિર atomic_bool atomic_flag_test_and_set(atomic_flag * ptr) {
    પાછા InterlockedExchange(ptr, 1);
}
સ્થિર શૂન્ય atomic_flag_clear(atomic_flag * ptr) {
    InterlockedExchange(ptr, 0);
}
સ્થિર શૂન્ય atomic_thread_fence(memory_order mo) {
    MemoryBarrier();
}
#નહીંતર // clang
#અન્તર્ભાવ <stdatomic.h>
#endif

પ્રકારવ્યાખ્યા HANDLE pthread_t;

પ્રકારવ્યાખ્યા DWORD thread_ret_t;
સ્થિર પૂર્ણાંક pthread_create(pthread_t * out, શૂન્ય * unused, thread_ret_t(*func)(શૂન્ય *), શૂન્ય * arg) {
    (શૂન્ય) unused;
    HANDLE handle = CreateThread(ખાલી, 0, (LPTHREAD_START_ROUTINE) func, arg, 0, ખાલી);
    જો (handle == ખાલી)
    {
        પાછા EAGAIN;
    }

    *out = handle;
    પાછા 0;
}

સ્થિર પૂર્ણાંક pthread_join(pthread_t thread, શૂન્ય * unused) {
    (શૂન્ય) unused;
    પૂર્ણાંક ret = (પૂર્ણાંક) WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    પાછા ret;
}

સ્થિર પૂર્ણાંક sched_yield (શૂન્ય) {
    Sleep (0);
    પાછા 0;
}
#નહીંતર

#અન્તર્ભાવ <pthread.h>
#અન્તર્ભાવ <stdatomic.h>
#અન્તર્ભાવ <sched.h>
#જો defined(__FreeBSD__)
#અન્તર્ભાવ <pthread_np.h>
#endif

પ્રકારવ્યાખ્યા શૂન્ય * thread_ret_t;

#અન્તર્ભાવ <sys/types.h>
#અન્તર્ભાવ <sys/stat.h>
#અન્તર્ભાવ <unistd.h>

#endif

પ્રકારવ્યાખ્યા pthread_t ggml_thread_t;

#define GGML_THREADPOOL_N_THREADS_MASK (0xffffU)
#define GGML_THREADPOOL_N_THREADS_BITS (16)

#જો defined(__APPLE__)
#અન્તર્ભાવ <unistd.h>
#અન્તર્ભાવ <mach/mach.h>
#અન્તર્ભાવ <TargetConditionals.h>
#endif

સ્થિર અચલ સંરચના ggml_type_traits_cpu type_traits_cpu[GGML_TYPE_ગણતરી] = {
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
#જો defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#નહીંતર
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q4_1] = {
        .from_float               = quantize_row_q4_1,
        .vec_dot                  = ggml_vec_dot_q4_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
#જો defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#નહીંતર
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
#જો defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#નહીંતર
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
#જો defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#નહીંતર
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
#જો defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#નહીંતર
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_IQ2_XXS] = {
        .from_float               = ખાલી,
        .vec_dot                  = ggml_vec_dot_iq2_xxs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_XS] = {
        .from_float               = ખાલી,
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
        .from_float               = ખાલી,
        .vec_dot                  = ggml_vec_dot_iq1_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_M] = {
        .from_float               = ખાલી,
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

અચલ સંરચના ggml_type_traits_cpu * ggml_get_type_traits_cpu(ગણના ggml_type type) {
    પાછા &type_traits_cpu[type];
}

//
// Threading defs
//

પ્રકારવ્યાખ્યા pthread_t          ggml_thread_t;

#જો defined(_WIN32)

પ્રકારવ્યાખ્યા CONDITION_VARIABLE ggml_cond_t;
પ્રકારવ્યાખ્યા SRWLOCK            ggml_mutex_t;

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

#નહીંતર

પ્રકારવ્યાખ્યા pthread_cond_t     ggml_cond_t;
પ્રકારવ્યાખ્યા pthread_mutex_t    ggml_mutex_t;

#define ggml_mutex_init(m)          pthread_mutex_init(m, ખાલી)
#define ggml_mutex_destroy(m)       pthread_mutex_destroy(m)
#define ggml_mutex_lock(m)          pthread_mutex_lock(m)
#define ggml_mutex_unlock(m)        pthread_mutex_unlock(m)
#define ggml_mutex_lock_shared(m)   pthread_mutex_lock(m)
#define ggml_mutex_unlock_shared(m) pthread_mutex_unlock(m)

#define ggml_lock_init(x)    UNUSED(x)
#define ggml_lock_destroy(x) UNUSED(x)
#જો defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64) && !defined(_M_ARM64EC))
#define ggml_lock_lock(x)    _mm_pause()
#નહીંતર
#define ggml_lock_lock(x)    UNUSED(x)
#endif
#define ggml_lock_unlock(x)  UNUSED(x)

#define GGML_LOCK_INITIALIZER 0
#define ggml_cond_init(c)      pthread_cond_init(c, ખાલી)
#define ggml_cond_destroy(c)   pthread_cond_destroy(c)
#define ggml_cond_wait(c, m)   pthread_cond_wait(c, m)
#define ggml_cond_broadcast(c) pthread_cond_broadcast(c)

#define ggml_thread_create pthread_create
#define ggml_thread_join   pthread_join

#endif

// Threadpool def
સંરચના ggml_threadpool {
    ggml_mutex_t mutex;       // mutex for cond.var
    ggml_cond_t  cond;        // cond.var for waiting for new work

    સંરચના ggml_cgraph * cgraph;
    સંરચના ggml_cplan  * cplan;

    // synchronization primitives
    atomic_int n_graph;       // updated when there is work to be done (i.e each graph) holds graph and active thread counts.
    atomic_int GGML_CACHE_ALIGN n_barrier;
    atomic_int GGML_CACHE_ALIGN n_barrier_passed;
    atomic_int GGML_CACHE_ALIGN current_chunk; // currently processing chunk during Mat_Mul, shared between all the threads.

    // these are atomic as an annotation for thread-sanitizer
    atomic_bool stop;         // Used for stopping the threadpool altogether
    atomic_bool pause;        // Used for pausing the threadpool or individual threads
    atomic_int  abort;        // Used for aborting processing of a graph

    સંરચના ggml_compute_state * workers;   // per thread state
    પૂર્ણાંક          n_threads;   // Number of threads in the pool
    int32_t      prio;        // Scheduling priority
    uint32_t     poll;        // Polling level (0 - no polling)

    ગણના ggml_status ec;
};

// Per-thread state
સંરચના ggml_compute_state {
#ifndef GGML_USE_OPENMP
    ggml_thread_t thrd;
    પૂર્ણાંક  last_graph;
    bool pending;
#endif
    bool cpumask[GGML_MAX_N_THREADS];
    સંરચના ggml_threadpool * threadpool;
    પૂર્ણાંક ith;
};

// Helpers for polling loops
#જો defined(__aarch64__) && ( defined(__clang__) || defined(__GNUC__) )
સ્થિર અંતઃરેખ શૂન્ય ggml_thread_cpu_relax(શૂન્ય) {
    __asm__ અસ્થિર("yield" ::: "memory");
}
#અથવાજો defined(__x86_64__)
સ્થિર અંતઃરેખ શૂન્ય ggml_thread_cpu_relax(શૂન્ય) {
    _mm_pause();
}
#અથવાજો defined(__riscv)
સ્થિર અંતઃરેખ શૂન્ય ggml_thread_cpu_relax(શૂન્ય) {
    #ifdef __riscv_zihintpause
        __asm__ __volatile__ ("pause");
    #નહીંતર
        /* Encoding of the pause instruction */
        __asm__ __volatile__ (".4byte 0x100000F");
    #endif
}
#નહીંતર
સ્થિર અંતઃરેખ શૂન્ય ggml_thread_cpu_relax(શૂન્ય) {;}
#endif

//
// NUMA support
//

#define GGML_NUMA_MAX_NODES 8
#define GGML_NUMA_MAX_CPUS 512

સંરચના ggml_numa_ગાંઠ {
    uint32_t cpus[GGML_NUMA_MAX_CPUS]; // hardware threads on this node
    uint32_t n_cpus;
};

સંરચના ggml_numa_nodes {
    ગણના ggml_numa_strategy numa_strategy;
    સંરચના ggml_numa_ગાંઠ nodes[GGML_NUMA_MAX_NODES];
    uint32_t n_nodes;
    uint32_t total_cpus; // hardware threads on system
    uint32_t current_ગાંઠ; // node on which main process is execting
#જો defined(__gnu_linux__)
    cpu_set_t cpuset; // cpuset from numactl
#નહીંતર
    uint32_t cpuset; // no NUMA support outside of Linux at this time. Use a portable datatype
#endif
};

//
// ggml state
//

સંરચના ggml_state {
    સંરચના ggml_numa_nodes numa;
};

સ્થિર સંરચના ggml_state g_state = {0};

શૂન્ય ggml_barrier(સંરચના ggml_threadpool * tp) {
    પૂર્ણાંક n_threads = atomic_load_explicit(&tp->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK;
    જો (n_threads == 1) {
        પાછા;
    }

#ifdef GGML_USE_OPENMP
    #pragma omp barrier
#નહીંતર
    પૂર્ણાંક n_passed = atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed);

    // enter barrier (full seq-cst fence)
    પૂર્ણાંક n_barrier = atomic_fetch_add_explicit(&tp->n_barrier, 1, memory_order_seq_cst);

    જો (n_barrier == (n_threads - 1)) {
        // last thread
        atomic_store_explicit(&tp->n_barrier, 0, memory_order_relaxed);

        // exit barrier (full seq-cst fence)
        atomic_fetch_add_explicit(&tp->n_barrier_passed, 1, memory_order_seq_cst);
        પાછા;
    }

    // wait for other threads
    જ્યારે (atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed) == n_passed) {
        ggml_thread_cpu_relax();
    }

    // exit barrier (full seq-cst fence)
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&tp->n_barrier_passed, 0, memory_order_seq_cst);
    #નહીંતર
    atomic_thread_fence(memory_order_seq_cst);
    #endif
#endif
}

શૂન્ય ggml_threadpool_chunk_set(સંરચના ggml_threadpool * tp, પૂર્ણાંક કિંમત) {
    atomic_store_explicit(&tp->current_chunk, કિંમત, memory_order_relaxed);
}

પૂર્ણાંક ggml_threadpool_chunk_add(સંરચના ggml_threadpool * tp, પૂર્ણાંક કિંમત) {
    પાછા atomic_fetch_add_explicit(&tp->current_chunk, કિંમત, memory_order_relaxed);
}

#જો defined(__gnu_linux__)
સ્થિર cpu_set_t ggml_get_numa_affinity(શૂન્ય) {
    cpu_set_t cpuset;
    pthread_t thread;
    thread = pthread_self();
    CPU_ZERO(&cpuset);
    pthread_getaffinity_np(thread, કદ(cpu_set_t), &cpuset);
    પાછા cpuset;
}
#નહીંતર
સ્થિર uint32_t ggml_get_numa_affinity(શૂન્ય) {
    પાછા 0; // no NUMA support
}
#endif

શૂન્ય ggml_numa_init(ગણના ggml_numa_strategy numa_flag) {
    જો (g_state.numa.n_nodes > 0) {
        fprintf(stderr, "ggml_numa_init: NUMA already initialized\n");

        પાછા;
    }

#જો defined(__gnu_linux__)
    સંરચના stat st;
    અક્ષર path[256];
    પૂર્ણાંક rv;

    // set numa scheme
    g_state.numa.numa_strategy = numa_flag;

    GGML_PRINT_DEBUG("numa strategy %u\n",g_state.numa.numa_strategy);

    g_state.numa.cpuset = ggml_get_numa_affinity();

    // enumerate nodes
    જ્યારે (g_state.numa.n_nodes < GGML_NUMA_MAX_NODES) {
        rv = snprintf(path, કદ(path), "/sys/devices/system/node/node%u", g_state.numa.n_nodes);
        GGML_ASSERT(rv > 0 && (unsigned)rv < કદ(path));
        જો (stat(path, &st) != 0) { તોડો; }
        ++g_state.numa.n_nodes;
    }

    // enumerate CPUs
    જ્યારે (g_state.numa.total_cpus < GGML_NUMA_MAX_CPUS) {
        rv = snprintf(path, કદ(path), "/sys/devices/system/cpu/cpu%u", g_state.numa.total_cpus);
        GGML_ASSERT(rv > 0 && (unsigned)rv < કદ(path));
        જો (stat(path, &st) != 0) { તોડો; }
        ++g_state.numa.total_cpus;
    }

    GGML_PRINT_DEBUG("found %u numa nodes, %u CPUs\n", g_state.numa.n_nodes, g_state.numa.total_cpus);

    // figure out which node we're on
    uint current_cpu;
    પૂર્ણાંક getcpu_ret = 0;
#જો __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ > 33) || defined(__COSMOPOLITAN__)
    getcpu_ret = getcpu(&current_cpu, &g_state.numa.current_ગાંઠ);
#નહીંતર
    // old glibc doesn't have a wrapper for this call. Fall back on direct syscall
#   જો !defined(SYS_getcpu) && defined(SYS_get_cpu)
#       define SYS_getcpu SYS_get_cpu // some older glibc versions use this name
#   endif
    getcpu_ret = syscall(SYS_getcpu, &current_cpu, &g_state.numa.current_ગાંઠ);
#endif

    જો (g_state.numa.n_nodes < 1 || g_state.numa.total_cpus < 1 || getcpu_ret != 0) {
        g_state.numa.n_nodes = 0;
        પાછા;
    }

    GGML_PRINT_DEBUG("found our process on numa node %u, CPU %u\n", g_state.numa.current_ગાંઠ, current_cpu);

    માટે (uint32_t n = 0; n < g_state.numa.n_nodes; ++n) {
        સંરચના ggml_numa_ગાંઠ * ગાંઠ = &g_state.numa.nodes[n];
        GGML_PRINT_DEBUG("CPUs on node %u:", n);
        ગાંઠ->n_cpus = 0;
        માટે (uint32_t c = 0; c < g_state.numa.total_cpus; ++c) {
            rv = snprintf(path, કદ(path), "/sys/devices/system/node/node%u/cpu%u", n, c);
            GGML_ASSERT(rv > 0 && (unsigned)rv < કદ(path));
            જો (stat(path, &st) == 0) {
                ગાંઠ->cpus[ગાંઠ->n_cpus++] = c;
                GGML_PRINT_DEBUG(" %u", c);
            }
        }
        GGML_PRINT_DEBUG("\n");
    }

    જો (ggml_is_numa()) {
        FILE *fptr = fopen("/proc/sys/kernel/numa_balancing", "r");
        જો (fptr != ખાલી) {
            અક્ષર buf[42];
            જો (fgets(buf, કદ(buf), fptr) && strncmp(buf, "0\n", કદ(buf)) != 0) {
                GGML_LOG_WARN("/proc/sys/kernel/numa_balancing is enabled, this has been observed to impair performance\n");
            }
            fclose(fptr);
        }
    }
#નહીંતર
    UNUSED(numa_flag);
    // TODO
#endif
}

bool ggml_is_numa(શૂન્ય) {
    પાછા g_state.numa.n_nodes > 1;
}

#જો defined(__ARM_ARCH)
#જો defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
#અન્તર્ભાવ <arm_sve.h>
સ્થિર શૂન્ય ggml_init_arm_arch_features(શૂન્ય) {
    ggml_arm_arch_features.sve_cnt = svcntb();
}
#નહીંતર
સ્થિર શૂન્ય ggml_init_arm_arch_features(શૂન્ય) {}
#endif
#endif // __ARM_ARCH

#જો defined(__riscv) && defined(__riscv_v_intrinsic)
#અન્તર્ભાવ <riscv_vector.h>
સ્થિર શૂન્ય ggml_init_riscv_arch_features(શૂન્ય) {
    ggml_riscv_arch_features.rvv_vlen = __riscv_vlenb();
}
#નહીંતર
સ્થિર શૂન્ય ggml_init_riscv_arch_features(શૂન્ય) {}
#endif

સંરચના ggml_tensor * ggml_new_i32(સંરચના ggml_context * ctx, int32_t કિંમત) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    સંરચના ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

    ggml_set_i32(result, કિંમત);

    પાછા result;
}

સંરચના ggml_tensor * ggml_new_f32(સંરચના ggml_context * ctx, દશાંશ કિંમત) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    સંરચના ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    ggml_set_f32(result, કિંમત);

    પાછા result;
}

સંરચના ggml_tensor * ggml_set_i32 (સંરચના ggml_tensor * tensor, int32_t કિંમત) {
    અચલ પૂર્ણાંક n     = ggml_nrows(tensor);
    અચલ પૂર્ણાંક nc    = tensor->ne[0];
    અચલ size_t n1 = tensor->nb[1];

    અક્ષર * અચલ data = tensor->data;

    પસંદગી (tensor->type) {
        સ્થિતિ GGML_TYPE_I8:
            {
                પ્રતિજ્ઞા(tensor->nb[0] == કદ(int8_t));
                માટે (પૂર્ણાંક i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), કિંમત);
                }
            } તોડો;
        સ્થિતિ GGML_TYPE_I16:
            {
                પ્રતિજ્ઞા(tensor->nb[0] == કદ(int16_t));
                માટે (પૂર્ણાંક i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), કિંમત);
                }
            } તોડો;
        સ્થિતિ GGML_TYPE_I32:
            {
                પ્રતિજ્ઞા(tensor->nb[0] == કદ(int32_t));
                માટે (પૂર્ણાંક i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), કિંમત);
                }
            } તોડો;
        સ્થિતિ GGML_TYPE_F16:
            {
                પ્રતિજ્ઞા(tensor->nb[0] == કદ(ggml_fp16_t));
                માટે (પૂર્ણાંક i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), GGML_CPU_FP32_TO_FP16(કિંમત));
                }
            } તોડો;
        સ્થિતિ GGML_TYPE_BF16:
            {
                પ્રતિજ્ઞા(tensor->nb[0] == કદ(ggml_fp16_t));
                માટે (પૂર્ણાંક i = 0; i < n; i++) {
                    ggml_vec_set_bf16(nc, (ggml_bf16_t *)(data + i*n1), GGML_FP32_TO_BF16(કિંમત));
                }
            } તોડો;
        સ્થિતિ GGML_TYPE_F32:
            {
                પ્રતિજ્ઞા(tensor->nb[0] == કદ(દશાંશ));
                માટે (પૂર્ણાંક i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (દશાંશ *)(data + i*n1), કિંમત);
                }
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }

    પાછા tensor;
}

સંરચના ggml_tensor * ggml_set_f32(સંરચના ggml_tensor * tensor, દશાંશ કિંમત) {
    અચલ પૂર્ણાંક n     = ggml_nrows(tensor);
    અચલ પૂર્ણાંક nc    = tensor->ne[0];
    અચલ size_t n1 = tensor->nb[1];

    અક્ષર * અચલ data = tensor->data;

    પસંદગી (tensor->type) {
        સ્થિતિ GGML_TYPE_I8:
            {
                પ્રતિજ્ઞા(tensor->nb[0] == કદ(int8_t));
                માટે (પૂર્ણાંક i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), કિંમત);
                }
            } તોડો;
        સ્થિતિ GGML_TYPE_I16:
            {
                પ્રતિજ્ઞા(tensor->nb[0] == કદ(int16_t));
                માટે (પૂર્ણાંક i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), કિંમત);
                }
            } તોડો;
        સ્થિતિ GGML_TYPE_I32:
            {
                પ્રતિજ્ઞા(tensor->nb[0] == કદ(int32_t));
                માટે (પૂર્ણાંક i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), કિંમત);
                }
            } તોડો;
        સ્થિતિ GGML_TYPE_F16:
            {
                પ્રતિજ્ઞા(tensor->nb[0] == કદ(ggml_fp16_t));
                માટે (પૂર્ણાંક i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), GGML_CPU_FP32_TO_FP16(કિંમત));
                }
            } તોડો;
        સ્થિતિ GGML_TYPE_BF16:
            {
                પ્રતિજ્ઞા(tensor->nb[0] == કદ(ggml_bf16_t));
                માટે (પૂર્ણાંક i = 0; i < n; i++) {
                    ggml_vec_set_bf16(nc, (ggml_bf16_t *)(data + i*n1), GGML_FP32_TO_BF16(કિંમત));
                }
            } તોડો;
        સ્થિતિ GGML_TYPE_F32:
            {
                પ્રતિજ્ઞા(tensor->nb[0] == કદ(દશાંશ));
                માટે (પૂર્ણાંક i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (દશાંશ *)(data + i*n1), કિંમત);
                }
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }

    પાછા tensor;
}

int32_t ggml_get_i32_1d(અચલ સંરચના ggml_tensor * tensor, પૂર્ણાંક i) {
    જો (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        પાછા ggml_get_i32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    પસંદગી (tensor->type) {
        સ્થિતિ GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == કદ(int8_t));
                પાછા ((int8_t *)(tensor->data))[i];
            }
        સ્થિતિ GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == કદ(int16_t));
                પાછા ((int16_t *)(tensor->data))[i];
            }
        સ્થિતિ GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == કદ(int32_t));
                પાછા ((int32_t *)(tensor->data))[i];
            }
        સ્થિતિ GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == કદ(ggml_fp16_t));
                પાછા GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *)(tensor->data))[i]);
            }
        સ્થિતિ GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == કદ(ggml_bf16_t));
                પાછા GGML_BF16_TO_FP32(((ggml_bf16_t *)(tensor->data))[i]);
            }
        સ્થિતિ GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == કદ(દશાંશ));
                પાછા ((દશાંશ *)(tensor->data))[i];
            }
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

શૂન્ય ggml_set_i32_1d(અચલ સંરચના ggml_tensor * tensor, પૂર્ણાંક i, int32_t કિંમત) {
    જો (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_set_i32_nd(tensor, id[0], id[1], id[2], id[3], કિંમત);
        પાછા;
    }
    પસંદગી (tensor->type) {
        સ્થિતિ GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == કદ(int8_t));
                ((int8_t *)(tensor->data))[i] = કિંમત;
            } તોડો;
        સ્થિતિ GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == કદ(int16_t));
                ((int16_t *)(tensor->data))[i] = કિંમત;
            } તોડો;
        સ્થિતિ GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == કદ(int32_t));
                ((int32_t *)(tensor->data))[i] = કિંમત;
            } તોડો;
        સ્થિતિ GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == કદ(ggml_fp16_t));
                ((ggml_fp16_t *)(tensor->data))[i] = GGML_CPU_FP32_TO_FP16(કિંમત);
            } તોડો;
        સ્થિતિ GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == કદ(ggml_bf16_t));
                ((ggml_bf16_t *)(tensor->data))[i] = GGML_FP32_TO_BF16(કિંમત);
            } તોડો;
        સ્થિતિ GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == કદ(દશાંશ));
                ((દશાંશ *)(tensor->data))[i] = કિંમત;
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

int32_t ggml_get_i32_nd(અચલ સંરચના ggml_tensor * tensor, પૂર્ણાંક i0, પૂર્ણાંક i1, પૂર્ણાંક i2, પૂર્ણાંક i3) {
    શૂન્ય * data   = (અક્ષર *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    પસંદગી (tensor->type) {
        સ્થિતિ GGML_TYPE_I8:
            પાછા ((int8_t *) data)[0];
        સ્થિતિ GGML_TYPE_I16:
            પાછા ((int16_t *) data)[0];
        સ્થિતિ GGML_TYPE_I32:
            પાછા ((int32_t *) data)[0];
        સ્થિતિ GGML_TYPE_F16:
            પાછા GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *) data)[0]);
        સ્થિતિ GGML_TYPE_BF16:
            પાછા GGML_BF16_TO_FP32(((ggml_bf16_t *) data)[0]);
        સ્થિતિ GGML_TYPE_F32:
            પાછા ((દશાંશ *) data)[0];
        મૂળભૂત:
            GGML_ABORT("fatal error");
    }
}

શૂન્ય ggml_set_i32_nd(અચલ સંરચના ggml_tensor * tensor, પૂર્ણાંક i0, પૂર્ણાંક i1, પૂર્ણાંક i2, પૂર્ણાંક i3, int32_t કિંમત) {
    શૂન્ય * data   = (અક્ષર *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    પસંદગી (tensor->type) {
        સ્થિતિ GGML_TYPE_I8:
            {
                ((int8_t *)(data))[0] = કિંમત;
            } તોડો;
        સ્થિતિ GGML_TYPE_I16:
            {
                ((int16_t *)(data))[0] = કિંમત;
            } તોડો;
        સ્થિતિ GGML_TYPE_I32:
            {
                ((int32_t *)(data))[0] = કિંમત;
            } તોડો;
        સ્થિતિ GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(data))[0] = GGML_CPU_FP32_TO_FP16(કિંમત);
            } તોડો;
        સ્થિતિ GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(data))[0] = GGML_FP32_TO_BF16(કિંમત);
            } તોડો;
        સ્થિતિ GGML_TYPE_F32:
            {
                ((દશાંશ *)(data))[0] = કિંમત;
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

દશાંશ ggml_get_f32_1d(અચલ સંરચના ggml_tensor * tensor, પૂર્ણાંક i) {
    જો (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        પાછા ggml_get_f32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    પસંદગી (tensor->type) {
        સ્થિતિ GGML_TYPE_I8:
            {
                પાછા ((int8_t *)(tensor->data))[i];
            }
        સ્થિતિ GGML_TYPE_I16:
            {
                પાછા ((int16_t *)(tensor->data))[i];
            }
        સ્થિતિ GGML_TYPE_I32:
            {
                પાછા ((int32_t *)(tensor->data))[i];
            }
        સ્થિતિ GGML_TYPE_F16:
            {
                પાછા GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *)(tensor->data))[i]);
            }
        સ્થિતિ GGML_TYPE_BF16:
            {
                પાછા GGML_BF16_TO_FP32(((ggml_bf16_t *)(tensor->data))[i]);
            }
        સ્થિતિ GGML_TYPE_F32:
            {
                પાછા ((દશાંશ *)(tensor->data))[i];
            }
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

શૂન્ય ggml_set_f32_1d(અચલ સંરચના ggml_tensor * tensor, પૂર્ણાંક i, દશાંશ કિંમત) {
    જો (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_set_f32_nd(tensor, id[0], id[1], id[2], id[3], કિંમત);
        પાછા;
    }
    પસંદગી (tensor->type) {
        સ્થિતિ GGML_TYPE_I8:
            {
                ((int8_t *)(tensor->data))[i] = કિંમત;
            } તોડો;
        સ્થિતિ GGML_TYPE_I16:
            {
                ((int16_t *)(tensor->data))[i] = કિંમત;
            } તોડો;
        સ્થિતિ GGML_TYPE_I32:
            {
                ((int32_t *)(tensor->data))[i] = કિંમત;
            } તોડો;
        સ્થિતિ GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(tensor->data))[i] = GGML_CPU_FP32_TO_FP16(કિંમત);
            } તોડો;
        સ્થિતિ GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(tensor->data))[i] = GGML_FP32_TO_BF16(કિંમત);
            } તોડો;
        સ્થિતિ GGML_TYPE_F32:
            {
                ((દશાંશ *)(tensor->data))[i] = કિંમત;
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

દશાંશ ggml_get_f32_nd(અચલ સંરચના ggml_tensor * tensor, પૂર્ણાંક i0, પૂર્ણાંક i1, પૂર્ણાંક i2, પૂર્ણાંક i3) {
    શૂન્ય * data   = (અક્ષર *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    પસંદગી (tensor->type) {
        સ્થિતિ GGML_TYPE_I8:
            પાછા ((int8_t *) data)[0];
        સ્થિતિ GGML_TYPE_I16:
            પાછા ((int16_t *) data)[0];
        સ્થિતિ GGML_TYPE_I32:
            પાછા ((int32_t *) data)[0];
        સ્થિતિ GGML_TYPE_F16:
            પાછા GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *) data)[0]);
        સ્થિતિ GGML_TYPE_BF16:
            પાછા GGML_BF16_TO_FP32(((ggml_bf16_t *) data)[0]);
        સ્થિતિ GGML_TYPE_F32:
            પાછા ((દશાંશ *) data)[0];
        મૂળભૂત:
            GGML_ABORT("fatal error");
    }
}

શૂન્ય ggml_set_f32_nd(અચલ સંરચના ggml_tensor * tensor, પૂર્ણાંક i0, પૂર્ણાંક i1, પૂર્ણાંક i2, પૂર્ણાંક i3, દશાંશ કિંમત) {
    શૂન્ય * data   = (અક્ષર *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    પસંદગી (tensor->type) {
        સ્થિતિ GGML_TYPE_I8:
            {
                ((int8_t *)(data))[0] = કિંમત;
            } તોડો;
        સ્થિતિ GGML_TYPE_I16:
            {
                ((int16_t *)(data))[0] = કિંમત;
            } તોડો;
        સ્થિતિ GGML_TYPE_I32:
            {
                ((int32_t *)(data))[0] = કિંમત;
            } તોડો;
        સ્થિતિ GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(data))[0] = GGML_CPU_FP32_TO_FP16(કિંમત);
            } તોડો;
        સ્થિતિ GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(data))[0] = GGML_FP32_TO_BF16(કિંમત);
            } તોડો;
        સ્થિતિ GGML_TYPE_F32:
            {
                ((દશાંશ *)(data))[0] = કિંમત;
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

////////////////////////////////////////////////////////////////////////////////

// ggml_compute_forward_mul_mat

સ્થિર શૂન્ય ggml_compute_forward_mul_mat_one_chunk(
    અચલ સંરચના ggml_compute_params * params,
    સંરચના ggml_tensor * dst,
    અચલ ગણના ggml_type type,
    અચલ int64_t અંક_rows_per_vec_dot,
    અચલ int64_t ir0_start,
    અચલ int64_t ir0_end,
    અચલ int64_t ir1_start,
    અચલ int64_t ir1_end) {

    અચલ સંરચના ggml_tensor * src0 = dst->src[0];
    અચલ સંરચના ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_દ્વિઅંકી_OP_LOCALS

    અચલ bool src1_cont = ggml_is_contiguous(src1);

    ggml_vec_dot_t અચલ vec_dot      = type_traits_cpu[type].vec_dot;
    ગણના ggml_type અચલ vec_dot_type = type_traits_cpu[type].vec_dot_type;

    // broadcast factors
    અચલ int64_t r2 = ne12 / ne02;
    અચલ int64_t r3 = ne13 / ne03;

    //printf("ir0_start = %6lld, ir0_end = %6lld, ir1_start = %6lld, ir1_end = %6lld\n", ir0_start, ir0_end, ir1_start, ir1_end);

    // threads with no work simply yield (not sure if it helps)
    જો (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        પાછા;
    }

    અચલ શૂન્ય * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
    અચલ size_t row_size = ggml_row_size(vec_dot_type, ne10);

    પ્રતિજ્ઞા(ne12 % ne02 == 0);
    પ્રતિજ્ઞા(ne13 % ne03 == 0);

    // block-tiling attempt
    અચલ int64_t blck_0 = 16;
    અચલ int64_t blck_1 = 16;

    અચલ size_t src1_col_stride = src1_cont || src1->type != vec_dot_type ? row_size : nb11;

    // attempt to reduce false-sharing (does not seem to make a difference)
    // 16 * 2, accounting for mmla kernels
    દશાંશ tmp[32];

    માટે (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        માટે (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            માટે (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += અંક_rows_per_vec_dot) {
                અચલ int64_t i13 = (ir1 / (ne12 * ne1));
                અચલ int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                અચલ int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                // broadcast src0 into src1
                અચલ int64_t i03 = i13 / r3;
                અચલ int64_t i02 = i12 / r2;

                અચલ int64_t i1 = i11;
                અચલ int64_t i2 = i12;
                અચલ int64_t i3 = i13;

                અચલ અક્ષર * src0_row = (અચલ અક્ષર*)src0->data + (0 + i02 * nb02 + i03 * nb03);

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                અચલ અક્ષર * src1_col = (અચલ અક્ષર*)wdata +
                    (src1_cont || src1->type != vec_dot_type
                        ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                        : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                દશાંશ * dst_col = (દશાંશ*)((અક્ષર*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

                //for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                //    vec_dot(ne00, &dst_col[ir0], src0_row + ir0*nb01, src1_col);
                //}

                માટે (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += અંક_rows_per_vec_dot) {
                    vec_dot(ne00, &tmp[ir0 - iir0], (અંક_rows_per_vec_dot > 1 ? 16 : 0), src0_row + ir0 * nb01, (અંક_rows_per_vec_dot > 1 ? nb01 : 0), src1_col, (અંક_rows_per_vec_dot > 1 ? src1_col_stride : 0), અંક_rows_per_vec_dot);
                }

                માટે (પૂર્ણાંક cn = 0; cn < અંક_rows_per_vec_dot; ++cn) {
                    memcpy(&dst_col[iir0 + cn * nb1 / nb0], tmp + (cn * 16), (MIN(iir0 + blck_0, ir0_end) - iir0) * કદ(દશાંશ));
                }
            }
        }
    }
}

શૂન્ય ggml_compute_forward_mul_mat(
        અચલ સંરચના ggml_compute_params * params,
              સંરચના ggml_tensor * dst) {

    અચલ સંરચના ggml_tensor * src0 = dst->src[0];
    અચલ સંરચના ggml_tensor * src1 = dst->src[1];

    અચલ int32_t hint = ggml_get_op_params_i32(dst, 1);
    જો (hint == GGML_HINT_SRC0_IS_HADAMARD && !params->use_ref) {
        ggml_compute_forward_fwht(params, dst);
        પાછા;
    }

    // If tiled is supported, it will execute the full op here and we return
    જો (ggml_compute_forward_mul_mat_tiled(params, dst)) {
        પાછા;
    }

    GGML_TENSOR_દ્વિઅંકી_OP_LOCALS

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    ગણના ggml_type           અચલ vec_dot_type         = type_traits_cpu[src0->type].vec_dot_type;
    ggml_from_float_t        અચલ from_float           = type_traits_cpu[vec_dot_type].from_float;
    int64_t                  અચલ vec_dot_અંક_rows     = type_traits_cpu[src0->type].nrows;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == કદ(દશાંશ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    // TODO: extract to "extra_op"
#જો GGML_USE_LLAMAFILE
    // broadcast factors
    અચલ int64_t r2 = ne12 / ne02;
    અચલ int64_t r3 = ne13 / ne03;

    અચલ bool src1_cont = ggml_is_contiguous(src1);

    જો (!params->use_ref && src1_cont) {
        માટે (int64_t i13 = 0; i13 < ne13; i13++)
            માટે (int64_t i12 = 0; i12 < ne12; i12++)
                જો (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_size(src0->type),
                                     (અચલ અક્ષર *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_size(src0->type),
                                     (અચલ અક્ષર *)src1->data + i12*nb12 + i13*nb13,
                                     nb11/ggml_type_size(src1->type),
                                     (અક્ષર *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_size(dst->type),
                                     src0->type,
                                     src1->type,
                                     dst->type))
                    જાઓ UseGgmlGemm1;
        પાછા;
    }
UseGgmlGemm1:;
#endif

    જો (src1->type != vec_dot_type) {
        અક્ષર * wdata = params->wdata;

        અચલ size_t nbw0 = ggml_type_size(vec_dot_type);
        અચલ size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        અચલ size_t nbw2 = nbw1*ne11;
        અચલ size_t nbw3 = nbw2*ne12;

        પ્રતિજ્ઞા(params->wsize >= ne13*nbw3);
        // src1 is either packed from F32 into vec_dot_type, or widened from F16 or BF16 into the F32 work buffer
        અચલ bool widen = src1->type != GGML_TYPE_F32;

        GGML_ASSERT(!widen || vec_dot_type == GGML_TYPE_F32);
        GGML_ASSERT(!widen || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_BF16);

    #જો 0
        માટે (int64_t i13 = 0; i13 < ne13; ++i13) {
            માટે (int64_t i12 = 0; i12 < ne12; ++i12) {
                માટે (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                    from_float((દશાંશ *)((અક્ષર *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (શૂન્ય *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                                ne10);
                }
            }
        }
    #નહીંતર
        અચલ int64_t bs = ggml_blck_size(vec_dot_type);

        માટે (int64_t i13 = 0; i13 < ne13; ++i13) {
            માટે (int64_t i12 = 0; i12 < ne12; ++i12) {
                માટે (int64_t i11 = 0; i11 < ne11; ++i11) {
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    અચલ શૂન્ય * src1_row  = (અચલ અક્ષર *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10;
                    શૂન્ય       * wdata_row =                    wdata  + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0;

                    અચલ int64_t ne10_block_size = (ne10_block_end - ne10_block_start) * bs;

                    જો (src1->type == GGML_TYPE_F16) {
                        ggml_cpu_fp16_to_fp32((અચલ ggml_fp16_t *) src1_row, (દશાંશ *) wdata_row, ne10_block_size);
                    } નહીંતર જો (src1->type == GGML_TYPE_BF16) {
                        ggml_cpu_bf16_to_fp32((અચલ ggml_bf16_t *) src1_row, (દશાંશ *) wdata_row, ne10_block_size);
                    } નહીંતર {
                        from_float((અચલ દશાંશ *) src1_row, wdata_row, ne10_block_size);
                    }
                }
            }
        }
    #endif
    }

    જો (ith == 0) {
        // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
        atomic_store_explicit(&params->threadpool->current_chunk, nth, memory_order_relaxed);
    }

    ggml_barrier(params->threadpool);

#જો GGML_USE_LLAMAFILE
    જો (!params->use_ref && src1->type != vec_dot_type) {
        અચલ શૂન્ય* wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        અચલ size_t row_size = ggml_row_size(vec_dot_type, ne10);

        માટે (int64_t i13 = 0; i13 < ne13; i13++)
            માટે (int64_t i12 = 0; i12 < ne12; i12++)
                જો (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_size(src0->type),
                                     (અચલ અક્ષર *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_size(src0->type),
                                     (અચલ અક્ષર *)wdata + (i12*ne11 + i13*ne12*ne11)*row_size,
                                     row_size/ggml_type_size(vec_dot_type),
                                     (અક્ષર *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_size(dst->type),
                                     src0->type,
                                     vec_dot_type,
                                     dst->type))
                    જાઓ UseGgmlGemm2;
        પાછા;
    }
UseGgmlGemm2:;
#endif

    // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
    અચલ int64_t nr0 = ne0;

    // This is the size of the rest of the dimensions of the result
    અચલ int64_t nr1 = ne1 * ne2 * ne3;

    // Now select a reasonable chunk size.
    પૂર્ણાંક chunk_size = 16;

    // We need to step up the size if it's small
    જો (nr0 == 1 || nr1 == 1) {
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
    જો (nchunk0 * nchunk1 < nth * 4 || ggml_is_numa()) {
        // distribute the thread work across the inner or outer loop based on which one is larger
        nchunk0 = nr0 > nr1 ? nth : 1; // parallelize by src0 rows
        nchunk1 = nr0 > nr1 ? 1 : nth; // parallelize by src1 rows
    }

    // The number of elements in each chunk
    અચલ int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    અચલ int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    // The first chunk comes from our thread_id, the rest will get auto-assigned.
    પૂર્ણાંક current_chunk = ith;

    જ્યારે (current_chunk < nchunk0 * nchunk1) {
        અચલ int64_t ith0 = current_chunk % nchunk0;
        અચલ int64_t ith1 = current_chunk / nchunk0;

        અચલ int64_t ir0_start = dr0 * ith0;
        અચલ int64_t ir0_end = MIN(ir0_start + dr0, nr0);

        અચલ int64_t ir1_start = dr1 * ith1;
        અચલ int64_t ir1_end = MIN(ir1_start + dr1, nr1);

        // dot kernels can handle 1 row and col at a time, but mmla kernels can process 2 rows and cols
        int64_t અંક_rows_per_vec_dot = vec_dot_અંક_rows;

        // these checks are needed to avoid crossing dim1 boundaries
        // can be optimized, but the logic would become more complicated, so keeping it like this for simplicity
        જો ((nr0 % 2 != 0) || (ne11 % 2 != 0) || ((ir0_end - ir0_start) % 2 != 0) || ((ir1_end - ir1_start) % 2 != 0)) {
            અંક_rows_per_vec_dot = 1;
        }
        ggml_compute_forward_mul_mat_one_chunk(params, dst, src0->type, અંક_rows_per_vec_dot, ir0_start, ir0_end, ir1_start, ir1_end);

        જો (nth >= nchunk0 * nchunk1) {
            તોડો;
        }

        current_chunk = atomic_fetch_add_explicit(&params->threadpool->current_chunk, 1, memory_order_relaxed);
    }
}

// ggml_compute_forward_mul_mat_id

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id)*ids->ne[0]*ids->ne[1] + (i1)]

સંરચના mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

સ્થિર શૂન્ય ggml_compute_forward_mul_mat_id_one_chunk(
    સંરચના ggml_tensor * dst,
    અચલ સંરચના ggml_tensor * src0,
    અચલ સંરચના ggml_tensor * src1,
    અચલ સંરચના ggml_tensor * ids,
    અચલ int64_t cur_a,
    અચલ int64_t ir0_start,
    અચલ int64_t ir0_end,
    અચલ int64_t ir1_start,
    અચલ int64_t ir1_end,
    અચલ અક્ષર * src0_cur,
    અચલ સંરચના mmid_row_mapping * matrix_rows,
    અચલ size_t row_size,
    અચલ bool src1_cont,
    અચલ શૂન્ય * wdata) {

    GGML_TENSOR_દ્વિઅંકી_OP_LOCALS

    અચલ ગણના ggml_type type = src0->type;

    ggml_vec_dot_t    અચલ vec_dot      = type_traits_cpu[type].vec_dot;
    ગણના ggml_type    અચલ vec_dot_type = type_traits_cpu[type].vec_dot_type;

    અચલ int64_t blck_0 = 16;
    અચલ int64_t blck_1 = 16;

    દશાંશ tmp[16];

    માટે (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        માટે (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            માટે (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ++ir1) {
                અચલ int64_t _i12 = ir1; // logical row index for this expert

                સંરચના mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, _i12);
                અચલ પૂર્ણાંક id       = row_mapping.i1; // selected expert index

                અચલ int64_t  i11 = id % ne11;
                અચલ int64_t  i12 = row_mapping.i2; // row index in src1

                અચલ int64_t  i1 = id;  // selected expert index
                અચલ int64_t  i2 = i12; // row

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                અચલ અક્ષર * src1_col = (અચલ અક્ષર *) wdata +
                    (src1_cont || src1->type != vec_dot_type
                    ? (i11      + i12*ne11)*row_size
                    : (i11*nb11 + i12*nb12));

                દશાંશ * dst_col = (દશાંશ *) ((અક્ષર *) dst->data + (i1*nb1 + i2*nb2));

                માટે (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                    vec_dot(ne00, &tmp[ir0 - iir0], 0, src0_cur + ir0*nb01, 0, src1_col, 0, 1);
                }

                memcpy(&dst_col[iir0], tmp, (MIN(iir0 + blck_0, ir0_end) - iir0)*કદ(દશાંશ));
            }
        }
    }
}

સ્થિર શૂન્ય * incr_ptr_aligned(શૂન્ય ** p, size_t size, size_t align) {

    શૂન્ય * ptr = *p;
    ptr = (શૂન્ય *) GGML_PAD((uintptr_t) ptr, align);
    *p = (શૂન્ય *) ((અક્ષર *) ptr + size);
    પાછા ptr;
}

સ્થિર શૂન્ય ggml_compute_forward_mul_mat_id(
        અચલ સંરચના ggml_compute_params * params,
              સંરચના ggml_tensor * dst) {

    અચલ સંરચના ggml_tensor * src0 = dst->src[0];
    અચલ સંરચના ggml_tensor * src1 = dst->src[1];
    અચલ સંરચના ggml_tensor * ids = dst->src[2];

    GGML_TENSOR_દ્વિઅંકી_OP_LOCALS

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ ગણના ggml_type type = src0->type;

    અચલ bool src1_cont = ggml_is_contiguous(src1);

    ગણના ggml_type    અચલ vec_dot_type    = type_traits_cpu[type].vec_dot_type;
    ggml_from_float_t અચલ from_float      = type_traits_cpu[vec_dot_type].from_float;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == કદ(દશાંશ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // row groups
    અચલ પૂર્ણાંક n_ids = ids->ne[0]; // n_expert_used
    અચલ પૂર્ણાંક n_as  = ne02;       // n_expert

    શૂન્ય * wdata_cur = params->wdata;

    જો (src1->type != vec_dot_type) {
        incr_ptr_aligned(&wdata_cur, ggml_row_size(vec_dot_type, ggml_nelements(src1)), કદ(int64_t));
    }

    int64_t * matrix_row_counts = // [n_as]
        incr_ptr_aligned(&wdata_cur, n_as*કદ(int64_t), કદ(int64_t));

    સંરચના mmid_row_mapping * matrix_rows = // [n_as][ids->ne[0]*ids->ne[1]]
        incr_ptr_aligned(&wdata_cur, n_as*ids->ne[0]*ids->ne[1]*કદ(સંરચના mmid_row_mapping), કદ(int64_t));

    અક્ષર (*atomic_current_chunk)[CACHE_લીટી_SIZE] = // [n_as]
        incr_ptr_aligned(&wdata_cur, CACHE_લીટી_SIZE * n_as, CACHE_લીટી_SIZE);

    // Tiled matmul (see tiled.h); per-thread work buffers, 0 bytes when disabled. The
    // reservation is unconditional, the per expert eligibility is decided at dispatch time
    અક્ષર * tiled_scratch = incr_ptr_aligned(&wdata_cur, ggml_tiled_wdata_size(nth, dst), 64);

    GGML_ASSERT(params->wsize >= (size_t)((અક્ષર *) wdata_cur - (અક્ષર *) params->wdata));

    જો (src1->type != vec_dot_type) {
        અક્ષર * wdata = params->wdata;

        અચલ size_t nbw0 = ggml_type_size(vec_dot_type);
        અચલ size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        અચલ size_t nbw2 = nbw1*ne11;
        અચલ size_t nbw3 = nbw2*ne12;

        પ્રતિજ્ઞા(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

#જો 0
        માટે (int64_t i13 = 0; i13 < ne13; ++i13) {
            માટે (int64_t i12 = ith; i12 < ne12; i12 += nth) {
                માટે (int64_t i11 = 0; i11 < ne11; ++i11) {
                    from_float((દશાંશ *)((અક્ષર *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (શૂન્ય *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                               ne10);
                }
            }
        }
#નહીંતર
        માટે (int64_t i13 = 0; i13 < ne13; ++i13) {
            માટે (int64_t i12 = 0; i12 < ne12; ++i12) {
                માટે (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = ggml_blck_size(vec_dot_type);
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    from_float((દશાંશ *)((અક્ષર *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10),
                               (શૂન્ય *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0),
                               (ne10_block_end - ne10_block_start) * bs);
                }
            }
        }
#endif
    }

    જો (ith == 0) {
        // initialize matrix_row_counts
        memset(matrix_row_counts, 0, n_as*કદ(int64_t));

        // group rows by src0 matrix
        માટે (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            માટે (પૂર્ણાંક id = 0; id < n_ids; ++id) {
                અચલ int32_t i02 = *(અચલ int32_t *) ((અચલ અક્ષર *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);

                પ્રતિજ્ઞા(i02 >= 0 && i02 < n_as);

                MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = (સંરચના mmid_row_mapping) {id, iid1};
                matrix_row_counts[i02] += 1;
            }
        }
    }

    // reset current_chunk
    માટે (પૂર્ણાંક cur_a = ith; cur_a < n_as; cur_a += nth) {
        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);
        *current_chunk_ctr = nth;
    }

    ggml_barrier(params->threadpool);

    માટે (પૂર્ણાંક cur_a = 0; cur_a < n_as; ++cur_a) {
        અચલ int64_t cne1 = matrix_row_counts[cur_a];

        જો (cne1 == 0) {
            ચાલુ;
        }

        // tiled takes over if profitable for this expert (see tiled.h)
        જો (ggml_compute_forward_mul_mat_id_tiled(params, dst, cur_a, cne1, (અચલ int32_t *) &MMID_MATRIX_ROW(cur_a, 0), tiled_scratch)) {
            ચાલુ;
        }

        અચલ અક્ષર * src0_cur = (અચલ અક્ષર *) src0->data + cur_a * nb02;
        અચલ શૂન્ય * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        અચલ size_t row_size = ggml_row_size(vec_dot_type, ne10);

        અચલ int64_t nr0 = ne01;
        અચલ int64_t nr1 = cne1;

        પૂર્ણાંક chunk_size = 16;
        જો (nr0 == 1 || nr1 == 1) {
            chunk_size = 64;
        }

        // disable for NUMA
        અચલ bool disable_chunking = ggml_is_numa();

        int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
        int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

        જો (nchunk0 * nchunk1 < nth * 4 || disable_chunking) {
            nchunk0 = nr0 > nr1 ? nth : 1;
            nchunk1 = nr0 > nr1 ? 1 : nth;
        }

        અચલ int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        અચલ int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

        પૂર્ણાંક current_chunk = ith;

        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);

        જ્યારે (current_chunk < nchunk0 * nchunk1) {
            અચલ int64_t ith0 = current_chunk % nchunk0;
            અચલ int64_t ith1 = current_chunk / nchunk0;

            અચલ int64_t ir0_start = dr0 * ith0;
            અચલ int64_t ir0_end = MIN(ir0_start + dr0, nr0);

            અચલ int64_t ir1_start = dr1 * ith1;
            અચલ int64_t ir1_end = MIN(ir1_start + dr1, nr1);

            ggml_compute_forward_mul_mat_id_one_chunk(
                dst, src0, src1, ids, cur_a,
                ir0_start, ir0_end, ir1_start, ir1_end,
                src0_cur, matrix_rows, row_size, src1_cont, wdata
            );

            જો (nth >= nchunk0 * nchunk1) {
                તોડો;
            }

            current_chunk = atomic_fetch_add_explicit(current_chunk_ctr, 1, memory_order_relaxed);
        }
    }
}

/////////////////////////////////

સ્થિર શૂન્ય ggml_compute_forward(સંરચના ggml_compute_params * params, સંરચના ggml_tensor * tensor) {
    GGML_ASSERT(params);

    જો (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {
        પાછા;
    }

    // extra_buffer op?
    જો (ggml_cpu_extra_compute_forward(params, tensor)) {
        પાછા;
    }

    પસંદગી (tensor->op) {
        સ્થિતિ GGML_OP_DUP:
            {
                ggml_compute_forward_dup(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_ADD:
            {
                ggml_compute_forward_add(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_ADD_ID:
            {
                ggml_compute_forward_add_id(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_ADD1:
            {
                ggml_compute_forward_add1(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_ACC:
            {
                ggml_compute_forward_acc(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_SUB:
            {
                ggml_compute_forward_sub(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_MUL:
            {
                ggml_compute_forward_mul(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_DIV:
            {
                ggml_compute_forward_div(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_SQR:
            {
                ggml_compute_forward_sqr(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_SQRT:
            {
                ggml_compute_forward_sqrt(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_LOG:
            {
                ggml_compute_forward_log(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_SIN:
            {
                ggml_compute_forward_sin(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_COS:
            {
                ggml_compute_forward_cos(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_SUM:
            {
                ggml_compute_forward_sum(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_SUM_ROWS:
            {
                ggml_compute_forward_sum_rows(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_CUMSUM:
            {
                ggml_compute_forward_cumsum(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_MEAN:
            {
                ggml_compute_forward_mean(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_ARGMAX:
            {
                ggml_compute_forward_argmax(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_ગણતરી_EQUAL:
            {
                ggml_compute_forward_ગણતરી_equal(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_REPEAT:
            {
                ggml_compute_forward_repeat(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_REPEAT_BACK:
            {
                ggml_compute_forward_repeat_back(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_CONCAT:
            {
                ggml_compute_forward_concat(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_SILU_BACK:
            {
                ggml_compute_forward_silu_back(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_NORM:
            {
                ggml_compute_forward_norm(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_RMS_NORM:
            {
                ggml_compute_forward_rms_norm(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_RMS_NORM_BACK:
            {
                ggml_compute_forward_rms_norm_back(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_GROUP_NORM:
            {
                ggml_compute_forward_group_norm(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_L2_NORM:
            {
                ggml_compute_forward_l2_norm(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_MUL_MAT:
            {
                ggml_compute_forward_mul_mat(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_MUL_MAT_ID:
            {
                ggml_compute_forward_mul_mat_id(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_OUT_PROD:
            {
                ggml_compute_forward_out_prod(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_SCALE:
            {
                ggml_compute_forward_scale(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_SET:
            {
                ggml_compute_forward_set(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_CPY:
            {
                ggml_compute_forward_cpy(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_CONT:
            {
                ggml_compute_forward_cont(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_GET_ROWS:
            {
                ggml_compute_forward_get_rows(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_GET_ROWS_BACK:
            {
                ggml_compute_forward_get_rows_back(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_SET_ROWS:
            {
                ggml_compute_forward_set_rows(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_DIAG:
            {
                ggml_compute_forward_diag(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_DIAG_MASK_INF:
            {
                ggml_compute_forward_diag_mask_inf(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_DIAG_MASK_ZERO:
            {
                ggml_compute_forward_diag_mask_zero(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_SOFT_MAX:
            {
                ggml_compute_forward_soft_max(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_SOFT_MAX_BACK:
            {
                ggml_compute_forward_soft_max_ext_back(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_ROPE:
            {
                ggml_compute_forward_rope(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_ROPE_BACK:
            {
                ggml_compute_forward_rope_back(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_CLAMP:
            {
                ggml_compute_forward_clamp(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_compute_forward_conv_transpose_1d(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_IM2COL:
            {
                ggml_compute_forward_im2col(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_IM2COL_BACK:
            {
                ggml_compute_forward_im2col_back_f32(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_IM2COL_3D:
            {
                ggml_compute_forward_im2col_3d(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_COL2IM_1D:
            {
                ggml_compute_forward_col2im_1d(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_CONV_2D:
            {
                ggml_compute_forward_conv_2d(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_CONV_3D:
            {
                ggml_compute_forward_conv_3d(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_CONV_2D_DW:
            {
                ggml_compute_forward_conv_2d_dw(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_CONV_TRANSPOSE_2D:
            {
                ggml_compute_forward_conv_transpose_2d(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_POOL_1D:
            {
                ggml_compute_forward_pool_1d(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_POOL_2D:
            {
                ggml_compute_forward_pool_2d(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_POOL_2D_BACK:
            {
                ggml_compute_forward_pool_2d_back(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_UPSCALE:
            {
                ggml_compute_forward_upscale(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_PAD:
            {
                ggml_compute_forward_pad(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_PAD_REFLECT_1D:
            {
                ggml_compute_forward_pad_reflect_1d(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_ROLL:
            {
                ggml_compute_forward_roll(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_ARANGE:
            {
                ggml_compute_forward_arange(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_TIMESTEP_EMBEDDING:
            {
                ggml_compute_forward_timestep_embedding(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_ARGSORT:
            {
                ggml_compute_forward_argsort(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_ટોચ_K:
            {
                ggml_compute_forward_ટોચ_k(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_LEAKY_RELU:
            {
                ggml_compute_forward_leaky_relu(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_TRI:
            {
                ggml_compute_forward_tri(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_FILL:
            {
                ggml_compute_forward_fill(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_FLASH_ATTN_EXT:
            {
                ggml_compute_forward_flash_attn_ext(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_FLASH_ATTN_BACK:
            {
                int32_t t = ggml_get_op_params_i32(tensor, 0);
                GGML_ASSERT(t == 0 || t == 1);
                bool masked = t != 0;
                ggml_compute_forward_flash_attn_back(params, masked, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_SSM_CONV:
            {
                ggml_compute_forward_ssm_conv(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_SSM_SCAN:
            {
                ggml_compute_forward_ssm_scan(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_WIN_PART:
            {
                ggml_compute_forward_win_part(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_WIN_UNPART:
            {
                ggml_compute_forward_win_unpart(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_UNARY:
            {
                ggml_compute_forward_unary(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_GLU:
            {
                ggml_compute_forward_glu(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_GET_REL_POS:
            {
                ggml_compute_forward_get_rel_pos(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_ADD_REL_POS:
            {
                ggml_compute_forward_add_rel_pos(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_RWKV_WKV6:
            {
                ggml_compute_forward_rwkv_wkv6(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_GATED_LINEAR_ATTN:
            {
                ggml_compute_forward_gla(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_RWKV_WKV7:
            {
                ggml_compute_forward_rwkv_wkv7(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_SOLVE_TRI:
            {
                ggml_compute_forward_solve_tri(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_GATED_DELTA_NET:
            {
                ggml_compute_forward_gated_delta_net(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_LIGHTNING_INDEXER:
            {
                ggml_compute_forward_lightning_indexer(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_DSV4_HC_COMB:
            {
                ggml_compute_forward_dsv4_hc_comb(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_DSV4_HC_PRE:
            {
                ggml_compute_forward_dsv4_hc_pre(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_DSV4_HC_POST:
            {
                ggml_compute_forward_dsv4_hc_post(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_ટોચ_K_MASK:
            {
                ggml_compute_forward_ટોચ_k_mask(params, tensor);
            } તોડો;
        સ્થિતિ GGML_OP_MAP_CUSTOM1:
            {
                ggml_compute_forward_map_custom1(params, tensor);
            }
            તોડો;
        સ્થિતિ GGML_OP_MAP_CUSTOM2:
            {
                ggml_compute_forward_map_custom2(params, tensor);
            }
            તોડો;
        સ્થિતિ GGML_OP_MAP_CUSTOM3:
            {
                ggml_compute_forward_map_custom3(params, tensor);
            }
            તોડો;
        સ્થિતિ GGML_OP_CUSTOM:
            {
                ggml_compute_forward_custom(params, tensor);
            }
            તોડો;
        સ્થિતિ GGML_OP_CROSS_ENTROPY_LOSS:
            {
                ggml_compute_forward_cross_entropy_loss(params, tensor);
            }
            તોડો;
        સ્થિતિ GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            {
                ggml_compute_forward_cross_entropy_loss_back(params, tensor);
            }
            તોડો;
        સ્થિતિ GGML_OP_OPT_STEP_ADAMW:
            {
                ggml_compute_forward_opt_step_adamw(params, tensor);
            }
            તોડો;
        સ્થિતિ GGML_OP_OPT_STEP_SGD:
            {
                ggml_compute_forward_opt_step_sgd(params, tensor);
            }
            તોડો;
        સ્થિતિ GGML_OP_NONE:
            {
                // nop
            } તોડો;
        સ્થિતિ GGML_OP_RESHAPE:
            {
                // nop
            } તોડો;
        સ્થિતિ GGML_OP_PERMUTE:
            {
                // nop
            } તોડો;
        સ્થિતિ GGML_OP_VIEW:
            {
                // nop
            } તોડો;
        સ્થિતિ GGML_OP_TRANSPOSE:
            {
                // nop
            } તોડો;
        સ્થિતિ GGML_OP_ગણતરી:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// Android's libc implementation "bionic" does not support setting affinity
#જો defined(__gnu_linux__)
સ્થિર શૂન્ય set_numa_thread_affinity(પૂર્ણાંક thread_n) {
    જો (!ggml_is_numa()) {
        પાછા;
    }

    પૂર્ણાંક ગાંઠ_અંક;
    પૂર્ણાંક rv;
    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    પસંદગી(g_state.numa.numa_strategy) {
        સ્થિતિ GGML_NUMA_STRATEGY_DISTRIBUTE:
            // run thread on node_num thread_n / (threads per node)
            ગાંઠ_અંક = thread_n % g_state.numa.n_nodes;
            તોડો;
        સ્થિતિ GGML_NUMA_STRATEGY_ISOLATE:
            // run thread on current_node
            ગાંઠ_અંક = g_state.numa.current_ગાંઠ;
            તોડો;
        સ્થિતિ GGML_NUMA_STRATEGY_NUMACTL:
            // use the cpuset that numactl gave us
            rv = pthread_setaffinity_np(pthread_self(), setsize, &g_state.numa.cpuset);
            જો (rv) {
                fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n",strerror(rv));
            }
            પાછા;
        મૂળભૂત:
            પાછા;
    }

    સંરચના ggml_numa_ગાંઠ * ગાંઠ = &g_state.numa.nodes[ગાંઠ_અંક];

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    માટે (size_t i = 0; i < ગાંઠ->n_cpus; ++i) {
        CPU_SET_S(ગાંઠ->cpus[i], setsize, cpus);
    }

    rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    જો (rv) {
            fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}

સ્થિર શૂન્ય clear_numa_thread_affinity(શૂન્ય) {
    જો (!ggml_is_numa()) {
        પાછા;
    }

    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    માટે (unsigned i = 0; i < g_state.numa.total_cpus; ++i) {
        CPU_SET_S(i, setsize, cpus);
    }

    પૂર્ણાંક rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    જો (rv) {
        fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}
#નહીંતર
// TODO: Windows etc.
// (the linux implementation may also work on BSD, someone should test)
સ્થિર શૂન્ય set_numa_thread_affinity(પૂર્ણાંક thread_n) { UNUSED(thread_n);  }
સ્થિર શૂન્ય clear_numa_thread_affinity(શૂન્ય) {}
#endif

સ્થિર પૂર્ણાંક ggml_get_n_tasks(સંરચના ggml_tensor * ગાંઠ, પૂર્ણાંક n_threads) {
    પૂર્ણાંક n_tasks = 0;

    જો (ggml_is_empty(ગાંઠ)) {
        // no need to multi-thread a no-op
        n_tasks = 1;
        પાછા n_tasks;
    }

    પસંદગી (ગાંઠ->op) {
        સ્થિતિ GGML_OP_CPY:
        સ્થિતિ GGML_OP_DUP:
        સ્થિતિ GGML_OP_CONT:
        સ્થિતિ GGML_OP_ADD:
        સ્થિતિ GGML_OP_ADD_ID:
        સ્થિતિ GGML_OP_ADD1:
        સ્થિતિ GGML_OP_ACC:
        સ્થિતિ GGML_OP_CUMSUM:
        સ્થિતિ GGML_OP_TRI:
        સ્થિતિ GGML_OP_FILL:
            {
                n_tasks = n_threads;
            } તોડો;
        સ્થિતિ GGML_OP_SUB:
        સ્થિતિ GGML_OP_SQR:
        સ્થિતિ GGML_OP_SQRT:
        સ્થિતિ GGML_OP_LOG:
        સ્થિતિ GGML_OP_SIN:
        સ્થિતિ GGML_OP_COS:
        સ્થિતિ GGML_OP_SUM:
        સ્થિતિ GGML_OP_SUM_ROWS:
        સ્થિતિ GGML_OP_MEAN:
        સ્થિતિ GGML_OP_ARGMAX:
            {
                n_tasks = 1;
            } તોડો;
        સ્થિતિ GGML_OP_ગણતરી_EQUAL:
        સ્થિતિ GGML_OP_SOLVE_TRI:
        સ્થિતિ GGML_OP_GATED_DELTA_NET:
        સ્થિતિ GGML_OP_DSV4_HC_COMB:
        સ્થિતિ GGML_OP_DSV4_HC_PRE:
        સ્થિતિ GGML_OP_DSV4_HC_POST:
        સ્થિતિ GGML_OP_ટોચ_K_MASK:
            {
                n_tasks = n_threads;
            } તોડો;
        સ્થિતિ GGML_OP_REPEAT:
        સ્થિતિ GGML_OP_REPEAT_BACK:
        સ્થિતિ GGML_OP_LEAKY_RELU:
            {
                n_tasks = 1;
            } તોડો;
        સ્થિતિ GGML_OP_UNARY:
            પસંદગી (ggml_get_unary_op(ગાંઠ)) {
                સ્થિતિ GGML_UNARY_OP_ABS:
                સ્થિતિ GGML_UNARY_OP_SGN:
                સ્થિતિ GGML_UNARY_OP_NEG:
                સ્થિતિ GGML_UNARY_OP_STEP:
                સ્થિતિ GGML_UNARY_OP_TANH:
                સ્થિતિ GGML_UNARY_OP_ELU:
                સ્થિતિ GGML_UNARY_OP_RELU:
                સ્થિતિ GGML_UNARY_OP_SIGMOID:
                સ્થિતિ GGML_UNARY_OP_HARDSWISH:
                સ્થિતિ GGML_UNARY_OP_HARDSIGMOID:
                સ્થિતિ GGML_UNARY_OP_EXP:
                સ્થિતિ GGML_UNARY_OP_SOFTPLUS:
                સ્થિતિ GGML_UNARY_OP_EXPM1:
                સ્થિતિ GGML_UNARY_OP_FLOOR:
                સ્થિતિ GGML_UNARY_OP_CEIL:
                સ્થિતિ GGML_UNARY_OP_ROUND:
                સ્થિતિ GGML_UNARY_OP_TRUNC:
                    {
                        n_tasks = 1;
                    } તોડો;

                સ્થિતિ GGML_UNARY_OP_GELU:
                સ્થિતિ GGML_UNARY_OP_GELU_ERF:
                સ્થિતિ GGML_UNARY_OP_GELU_QUICK:
                સ્થિતિ GGML_UNARY_OP_SILU:
                સ્થિતિ GGML_UNARY_OP_XIELU:
                    {
                        n_tasks = n_threads;
                    } તોડો;
                મૂળભૂત:
                    GGML_ABORT("fatal error");
            }
            તોડો;
        સ્થિતિ GGML_OP_GLU:
            પસંદગી (ggml_get_glu_op(ગાંઠ)) {
                સ્થિતિ GGML_GLU_OP_REGLU:
                સ્થિતિ GGML_GLU_OP_GEGLU:
                સ્થિતિ GGML_GLU_OP_SWIGLU:
                સ્થિતિ GGML_GLU_OP_SWIGLU_OAI:
                સ્થિતિ GGML_GLU_OP_GEGLU_ERF:
                સ્થિતિ GGML_GLU_OP_GEGLU_QUICK:
                સ્થિતિ GGML_GLU_OP_SWIGLU_CLAMP:
                    {
                        n_tasks = n_threads;
                    } તોડો;
                મૂળભૂત:
                    GGML_ABORT("fatal error");
            }
            તોડો;
        સ્થિતિ GGML_OP_SILU_BACK:
        સ્થિતિ GGML_OP_MUL:
        સ્થિતિ GGML_OP_DIV:
        સ્થિતિ GGML_OP_NORM:
        સ્થિતિ GGML_OP_RMS_NORM:
        સ્થિતિ GGML_OP_RMS_NORM_BACK:
        સ્થિતિ GGML_OP_L2_NORM:
        સ્થિતિ GGML_OP_GROUP_NORM:
        સ્થિતિ GGML_OP_CONCAT:
        સ્થિતિ GGML_OP_MUL_MAT:
        સ્થિતિ GGML_OP_MUL_MAT_ID:
        સ્થિતિ GGML_OP_OUT_PROD:
        સ્થિતિ GGML_OP_SET_ROWS:
            {
                n_tasks = n_threads;
            } તોડો;
        સ્થિતિ GGML_OP_GET_ROWS:
            {
                // FIXME: get_rows can use additional threads, but the cost of launching additional threads
                // decreases performance with GPU offloading
                //n_tasks = n_threads;
                n_tasks = 1;
            } તોડો;
        સ્થિતિ GGML_OP_SCALE:
        સ્થિતિ GGML_OP_SET:
        સ્થિતિ GGML_OP_RESHAPE:
        સ્થિતિ GGML_OP_VIEW:
        સ્થિતિ GGML_OP_PERMUTE:
        સ્થિતિ GGML_OP_TRANSPOSE:
        સ્થિતિ GGML_OP_GET_ROWS_BACK:
        સ્થિતિ GGML_OP_DIAG:
            {
                n_tasks = 1;
            } તોડો;
        સ્થિતિ GGML_OP_DIAG_MASK_ZERO:
        સ્થિતિ GGML_OP_DIAG_MASK_INF:
        સ્થિતિ GGML_OP_SOFT_MAX_BACK:
        સ્થિતિ GGML_OP_ROPE:
        સ્થિતિ GGML_OP_ROPE_BACK:
        સ્થિતિ GGML_OP_ADD_REL_POS:
            {
                n_tasks = n_threads;
            } તોડો;
        સ્થિતિ GGML_OP_CLAMP:
            {
                n_tasks = 1; //TODO
            } તોડો;
        સ્થિતિ GGML_OP_SOFT_MAX:
            {
                n_tasks = MIN(n_threads, ggml_nrows(ગાંઠ->src[0]));
            } તોડો;
        સ્થિતિ GGML_OP_IM2COL:
        સ્થિતિ GGML_OP_IM2COL_BACK:
        સ્થિતિ GGML_OP_IM2COL_3D:
        સ્થિતિ GGML_OP_CONV_2D:
        સ્થિતિ GGML_OP_CONV_3D:
        સ્થિતિ GGML_OP_CONV_2D_DW:
        સ્થિતિ GGML_OP_COL2IM_1D:
        સ્થિતિ GGML_OP_CONV_TRANSPOSE_1D:
        સ્થિતિ GGML_OP_CONV_TRANSPOSE_2D:
            {
                n_tasks = n_threads;
            } તોડો;
        સ્થિતિ GGML_OP_POOL_1D:
        સ્થિતિ GGML_OP_POOL_2D:
        સ્થિતિ GGML_OP_POOL_2D_BACK:
            {
                n_tasks = 1;
            } તોડો;
        સ્થિતિ GGML_OP_UPSCALE:
        સ્થિતિ GGML_OP_PAD:
        સ્થિતિ GGML_OP_PAD_REFLECT_1D:
        સ્થિતિ GGML_OP_ROLL:
        સ્થિતિ GGML_OP_ARANGE:
        સ્થિતિ GGML_OP_TIMESTEP_EMBEDDING:
        સ્થિતિ GGML_OP_ARGSORT:
        સ્થિતિ GGML_OP_ટોચ_K:
        સ્થિતિ GGML_OP_FLASH_ATTN_EXT:
        સ્થિતિ GGML_OP_FLASH_ATTN_BACK:
        સ્થિતિ GGML_OP_SSM_CONV:
        સ્થિતિ GGML_OP_SSM_SCAN:
        સ્થિતિ GGML_OP_LIGHTNING_INDEXER:
            {
                n_tasks = n_threads;
            } તોડો;
        સ્થિતિ GGML_OP_RWKV_WKV6:
        સ્થિતિ GGML_OP_GATED_LINEAR_ATTN:
        સ્થિતિ GGML_OP_RWKV_WKV7:
            {
                અચલ int64_t n_heads = ગાંઠ->src[1]->ne[1];
                n_tasks = MIN(n_threads, n_heads);
            } તોડો;
        સ્થિતિ GGML_OP_WIN_PART:
        સ્થિતિ GGML_OP_WIN_UNPART:
        સ્થિતિ GGML_OP_GET_REL_POS:
            {
                n_tasks = 1;
            } તોડો;
        સ્થિતિ GGML_OP_MAP_CUSTOM1:
            {
                સંરચના ggml_map_custom1_op_params p;
                memcpy(&p, ગાંઠ->op_params, કદ(p));
                જો (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } નહીંતર {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } તોડો;
        સ્થિતિ GGML_OP_MAP_CUSTOM2:
            {
                સંરચના ggml_map_custom2_op_params p;
                memcpy(&p, ગાંઠ->op_params, કદ(p));
                જો (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } નહીંતર {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } તોડો;
        સ્થિતિ GGML_OP_MAP_CUSTOM3:
            {
                સંરચના ggml_map_custom3_op_params p;
                memcpy(&p, ગાંઠ->op_params, કદ(p));
                જો (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } નહીંતર {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } તોડો;
        સ્થિતિ GGML_OP_CUSTOM:
            {
                સંરચના ggml_custom_op_params p;
                memcpy(&p, ગાંઠ->op_params, કદ(p));
                જો (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } નહીંતર {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } તોડો;
        સ્થિતિ GGML_OP_CROSS_ENTROPY_LOSS:
        સ્થિતિ GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        સ્થિતિ GGML_OP_OPT_STEP_ADAMW:
        સ્થિતિ GGML_OP_OPT_STEP_SGD:
            {
                n_tasks = n_threads;
            } તોડો;
        સ્થિતિ GGML_OP_NONE:
            {
                n_tasks = 1;
            } તોડો;
        સ્થિતિ GGML_OP_ગણતરી:
            {
                GGML_ABORT("fatal error");
            }
        મૂળભૂત:
            {
                fprintf(stderr, "%s: op not implemented: ", __func__);
                જો (ગાંઠ->op < GGML_OP_ગણતરી) {
                    fprintf(stderr, "%s\n", ggml_op_name(ગાંઠ->op));
                } નહીંતર {
                    fprintf(stderr, "%d\n", ગાંઠ->op);
                }
                GGML_ABORT("fatal error");
            }
    }

    પ્રતિજ્ઞા(n_tasks > 0);

    પાછા n_tasks;
}

સ્થિર thread_ret_t ggml_graph_compute_secondary_thread(શૂન્ય* data);

#જો defined(_WIN32)
#અન્તર્ભાવ "windows.h"

// TODO: support > 64 CPUs
સ્થિર bool ggml_thread_apply_affinity(bool * mask) {
    HANDLE    h = GetCurrentThread();
    uint64_t  bitmask = 0ULL;

    પ્રતિજ્ઞા(GGML_MAX_N_THREADS >= 64);

    માટે (int32_t i = 0; i < 8; i++) {
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

    માટે (int32_t i = 64; i < GGML_MAX_N_THREADS; i++) {
        જો (mask[i]) {
            fprintf(stderr, "warn: setting thread-affinity for > 64 CPUs isn't supported on windows!\n");
            તોડો;
        }
    }

    DWORD_PTR m = (DWORD_PTR)bitmask;

    m = SetThreadAffinityMask(h, m);

    પાછા m != 0;
}

સ્થિર bool ggml_thread_apply_priority(int32_t prio) {
    // Note that on Windows the Process Priority Class must be updated in order to set Thread priority.
    // This is up to the applications.
    DWORD p = THREAD_PRIORITY_NORMAL;
    પસંદગી (prio) {
        સ્થિતિ GGML_SCHED_PRIO_નીચલું:      p = THREAD_PRIORITY_BELOW_NORMAL;  તોડો;
        સ્થિતિ GGML_SCHED_PRIO_NORMAL:   p = THREAD_PRIORITY_NORMAL;        તોડો;
        સ્થિતિ GGML_SCHED_PRIO_MEDIUM:   p = THREAD_PRIORITY_ABOVE_NORMAL;  તોડો;
        સ્થિતિ GGML_SCHED_PRIO_ઉપલું:     p = THREAD_PRIORITY_HIGHEST;       તોડો;
        સ્થિતિ GGML_SCHED_PRIO_REALTIME: p = THREAD_PRIORITY_TIME_CRITICAL; તોડો;
    }

    જો (prio != GGML_SCHED_PRIO_નીચલું) {
        // Tell Windows that this thread should not be throttled (needs its own CPU core).
        // Newer Windows 11 versions aggressively park (offline) CPU cores and often place
        // all our threads onto the first 4 cores which results in terrible performance with
        // n_threads > 4
        #જો _WIN32_WINNT >= 0x0602
        THREAD_POWER_THROTTLING_STATE t;
        ZeroMemory(&t, કદ(t));
        t.Version     = THREAD_POWER_THROTTLING_CURRENT_VERSION;
        t.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
        t.StateMask   = 0;

        જો (!SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &t, કદ(t))) {
            GGML_LOG_DEBUG("failed to disable thread power throttling %d : (%d)\n", prio, (પૂર્ણાંક) GetLastError());
            પાછા false;
        }
        #endif
    }

    જો (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        પાછા true;
    }

    જો (!SetThreadPriority(GetCurrentThread(), p)) {
        fprintf(stderr, "warn: failed to set thread priority %d : (%d)\n", prio, (પૂર્ણાંક) GetLastError());
        પાછા false;
    }

    પાછા true;
}

#અથવાજો defined(__APPLE__)
#અન્તર્ભાવ <sys/types.h>
#અન્તર્ભાવ <sys/resource.h>

સ્થિર bool ggml_thread_apply_affinity(અચલ bool * mask) {
    // Not supported on Apple platforms
    UNUSED(mask);
    પાછા true;
}

સ્થિર bool ggml_thread_apply_priority(int32_t prio) {
    સંરચના sched_param p;
    int32_t policy = SCHED_OTHER;
    પસંદગી (prio) {
        // TODO: there seems to be no way to set lower prio on Apple platforms
        સ્થિતિ GGML_SCHED_PRIO_નીચલું:      policy = SCHED_OTHER; p.sched_priority = 0;  તોડો;
        સ્થિતિ GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  તોડો;
        સ્થિતિ GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; તોડો;
        સ્થિતિ GGML_SCHED_PRIO_ઉપલું:     policy = SCHED_FIFO;  p.sched_priority = 80; તોડો;
        સ્થિતિ GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; તોડો;
    }

    જો (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        પાછા true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    જો (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        પાછા false;
    }

    પાછા true;
}

#અથવાજો defined(__linux__)
// TODO: this may not work on BSD, to be verified

સ્થિર bool ggml_thread_apply_affinity(અચલ bool * mask) {
    cpu_set_t cpuset;
    પૂર્ણાંક err;

    CPU_ZERO(&cpuset);

    માટે (uint32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
        જો (mask[i]) {
            GGML_PRINT_DEBUG("Thread %lx: adding %d to cpuset\n", pthread_self(), i);
            CPU_SET(i, &cpuset);
        }
    }

#ifdef __ANDROID__
    err = sched_setaffinity(0, કદ(cpuset), &cpuset);
    જો (err < 0) {
        err = errno;
    }
#નહીંતર
    err = pthread_setaffinity_np(pthread_self(), કદ(cpuset), &cpuset);
#endif
    જો (err != 0) {
        fprintf(stderr, "warn: failed to set affinity mask 0x%llx : %s (%d)\n", (unsigned long long)mask, strerror(err), err);
        પાછા false;
    }

    પાછા true;
}

સ્થિર bool ggml_thread_apply_priority(int32_t prio) {
    સંરચના sched_param p;
    int32_t policy = SCHED_OTHER;
    પસંદગી (prio) {
        સ્થિતિ GGML_SCHED_PRIO_નીચલું:      policy = SCHED_BATCH; p.sched_priority = 0;  તોડો;
        સ્થિતિ GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  તોડો;
        સ્થિતિ GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; તોડો;
        સ્થિતિ GGML_SCHED_PRIO_ઉપલું:     policy = SCHED_FIFO;  p.sched_priority = 80; તોડો;
        સ્થિતિ GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; તોડો;
    }

    જો (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        પાછા true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    જો (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        પાછા false;
    }

    પાછા true;
}

#નહીંતર // unsupported platforms

સ્થિર bool ggml_thread_apply_affinity(અચલ bool * mask) {
    UNUSED(mask);
    પાછા true;
}

સ્થિર bool ggml_thread_apply_priority(int32_t prio) {
    UNUSED(prio);
    પાછા true;
}

#endif

સ્થિર bool ggml_thread_cpumask_is_valid(અચલ bool * mask) {
    માટે (પૂર્ણાંક i = 0; i < GGML_MAX_N_THREADS; i++) {
        જો (mask[i]) { પાછા true; }
    }
    પાછા false;
}

સ્થિર શૂન્ય ggml_thread_cpumask_next(અચલ bool * global_mask, bool * local_mask, bool strict, int32_t* iter) {
    જો (!strict) {
        memcpy(local_mask, global_mask, GGML_MAX_N_THREADS);
        પાછા;
    } નહીંતર {
        memset(local_mask, 0, GGML_MAX_N_THREADS);
        int32_t base_idx = *iter;
        માટે (int32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
            int32_t idx = base_idx + i;
            જો (idx >= GGML_MAX_N_THREADS) {
                // Just a cheaper modulo
                idx -= GGML_MAX_N_THREADS;
            }
            જો (global_mask[idx]) {
                local_mask[idx] = 1;
                *iter = idx + 1;
                પાછા;
            }
        }
    }
}

શૂન્ય ggml_threadpool_free(સંરચના ggml_threadpool* threadpool) {
    જો (!threadpool) પાછા;

    અચલ પૂર્ણાંક n_threads = threadpool->n_threads;

#ifndef GGML_USE_OPENMP
    સંરચના ggml_compute_state* workers = threadpool->workers;

    ggml_mutex_lock(&threadpool->mutex);

    threadpool->stop = true;
    threadpool->pause = false;

    ggml_cond_broadcast(&threadpool->cond);
    ggml_mutex_unlock(&threadpool->mutex);

    માટે (પૂર્ણાંક j = 1; j < n_threads; j++) {
        int32_t rc = ggml_thread_join(workers[j].thrd, ખાલી);
        GGML_ASSERT(rc == GGML_EXIT_SUCCESS || rc == GGML_EXIT_ABORTED);
        UNUSED(rc);
    }

    ggml_mutex_destroy(&threadpool->mutex);
    ggml_cond_destroy(&threadpool->cond);
#endif // GGML_USE_OPENMP

    અચલ size_t workers_size = કદ(સંરચના ggml_compute_state) * n_threads;
    ggml_aligned_free(threadpool->workers, workers_size);
    ggml_aligned_free(threadpool, કદ(સંરચના ggml_threadpool));
}

#ifndef GGML_USE_OPENMP
// pause/resume must be called under mutex
સ્થિર શૂન્ય ggml_threadpool_pause_locked(સંરચના ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Pausing threadpool\n");
    threadpool->pause = true;
    ggml_cond_broadcast(&threadpool->cond);
}

સ્થિર શૂન્ય ggml_threadpool_resume_locked(સંરચના ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Resuming threadpool\n");
    threadpool->pause = false;
    ggml_cond_broadcast(&threadpool->cond);
}
#endif

શૂન્ય ggml_threadpool_pause(સંરચના ggml_threadpool * threadpool) {
#ifndef GGML_USE_OPENMP
    ggml_mutex_lock(&threadpool->mutex);
    જો (!threadpool->pause) {
       ggml_threadpool_pause_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#નહીંતર
    UNUSED(threadpool);
#endif
}

શૂન્ય ggml_threadpool_resume(સંરચના ggml_threadpool * threadpool) {
#ifndef GGML_USE_OPENMP
    ggml_mutex_lock(&threadpool->mutex);
    જો (threadpool->pause) {
       ggml_threadpool_resume_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#નહીંતર
    UNUSED(threadpool);
#endif
}

સંરચના ggml_cplan ggml_graph_plan(
          અચલ સંરચના ggml_cgraph * cgraph,
                               પૂર્ણાંક   n_threads,
            સંરચના ggml_threadpool * threadpool) {

    જો (threadpool == ખાલી) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
    }
    જો (n_threads <= 0) {
        n_threads = threadpool ? threadpool->n_threads : GGML_DEFAULT_N_THREADS;
    }

#જો defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
    // Emscripten without pthreads support can only use a single thread
    n_threads = 1;
#endif

#જો defined(__wasi__)
    // WASI doesn't support parallelism yet
    n_threads = 1;
#endif

    size_t work_size = 0;

    સંરચના ggml_cplan cplan;
    memset(&cplan, 0, કદ(સંરચના ggml_cplan));

    પૂર્ણાંક max_tasks = 1;

    // thread scheduling for the different operations + work buffer size estimation
    માટે (પૂર્ણાંક i = 0; i < cgraph->n_nodes; i++) {
        સંરચના ggml_tensor * ગાંઠ = cgraph->nodes[i];

        અચલ પૂર્ણાંક n_tasks = ggml_get_n_tasks(ગાંઠ, n_threads);

        max_tasks = MAX(max_tasks, n_tasks);

        size_t cur = 0;

        જો (!ggml_cpu_extra_work_size(n_threads, ગાંઠ, &cur)) {
            પસંદગી (ગાંઠ->op) {
                સ્થિતિ GGML_OP_CPY:
                સ્થિતિ GGML_OP_DUP:
                    {
                        જો (ggml_is_quantized(ગાંઠ->type) ||
                            // F16 -> BF16 and BF16 -> F16 copies go through intermediate F32
                            (ગાંઠ->src[0]->type == GGML_TYPE_F16  && ગાંઠ->src[1] && ગાંઠ->src[1]->type == GGML_TYPE_BF16) ||
                            (ગાંઠ->src[0]->type == GGML_TYPE_BF16 && ગાંઠ->src[1] && ગાંઠ->src[1]->type == GGML_TYPE_F16) ||
                            // conversion between F32 and I32
                            (ગાંઠ->src[0]->type == GGML_TYPE_F32 && ગાંઠ->src[1] && ગાંઠ->src[1]->type == GGML_TYPE_I32) ||
                            (ગાંઠ->src[0]->type == GGML_TYPE_I32 && ગાંઠ->src[1] && ગાંઠ->src[1]->type == GGML_TYPE_F32)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * ગાંઠ->ne[0] * n_tasks;
                        }
                    } તોડો;
                સ્થિતિ GGML_OP_ADD:
                સ્થિતિ GGML_OP_ADD_ID:
                સ્થિતિ GGML_OP_ADD1:
                    {
                        જો (ggml_is_quantized(ગાંઠ->src[0]->type)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * ગાંઠ->src[0]->ne[0] * n_tasks;
                        }
                    } તોડો;
                સ્થિતિ GGML_OP_ACC:
                    {
                        જો (ggml_is_quantized(ગાંઠ->src[0]->type)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * ગાંઠ->src[1]->ne[0] * n_tasks;
                        }
                    } તોડો;
                સ્થિતિ GGML_OP_ગણતરી_EQUAL:
                    {
                        cur = ggml_type_size(ગાંઠ->type)*n_tasks;
                    } તોડો;
                સ્થિતિ GGML_OP_MUL_MAT:
                    {
                        અચલ ગણના ggml_type vec_dot_type = type_traits_cpu[ગાંઠ->src[0]->type].vec_dot_type;
                        જો (ગાંઠ->src[1]->type != vec_dot_type) {
                            cur = ggml_row_size(vec_dot_type, ggml_nelements(ગાંઠ->src[1]));
                        }
                        // Workspace for tiled (see tiled.h)
                        cur = GGML_PAD(cur, 64);
                        cur += ggml_tiled_wdata_size(n_tasks, ગાંઠ);
                    } તોડો;
                સ્થિતિ GGML_OP_MUL_MAT_ID:
                    {
                        cur = 0;
                        અચલ સંરચના ggml_tensor * src0 = ગાંઠ->src[0];
                        અચલ સંરચના ggml_tensor * src1 = ગાંઠ->src[1];
                        અચલ સંરચના ggml_tensor * ids = ગાંઠ->src[2];
                        અચલ ગણના ggml_type vec_dot_type = type_traits_cpu[src0->type].vec_dot_type;
                        અચલ પૂર્ણાંક n_as = src0->ne[2];
                        // src1
                        જો (src1->type != vec_dot_type) {
                            cur += ggml_row_size(vec_dot_type, ggml_nelements(src1)) + કદ(int64_t);
                        }
                        // matrix_row_counts
                        cur += n_as * કદ(int64_t) + કદ(int64_t);
                        // matrix_rows
                        cur += n_as*ids->ne[0]*ids->ne[1]*કદ(સંરચના mmid_row_mapping) + કદ(int64_t);
                        // atomic_current_chunk
                        cur += CACHE_લીટી_SIZE*n_as + CACHE_લીટી_SIZE;
                        // Workspace for tiled (see tiled.h)
                        cur = GGML_PAD(cur, 64);
                        cur += ggml_tiled_wdata_size(n_tasks, ગાંઠ);
                    } તોડો;
                સ્થિતિ GGML_OP_OUT_PROD:
                    {
                        જો (ggml_is_quantized(ગાંઠ->src[0]->type) ||
                            ગાંઠ->src[0]->type == GGML_TYPE_F16) {
                            cur = ggml_type_size(GGML_TYPE_F32) * ગાંઠ->src[0]->ne[0] * n_tasks;
                        }
                    } તોડો;
                સ્થિતિ GGML_OP_SET_ROWS:
                    {
                        જો (ગાંઠ->src[0]->type == GGML_TYPE_F16 && ગાંઠ->type != GGML_TYPE_F16) {
                            cur = ggml_type_size(GGML_TYPE_F32) * ગાંઠ->src[0]->ne[0] * n_tasks;
                        }
                    } તોડો;
                સ્થિતિ GGML_OP_SOFT_MAX:
                સ્થિતિ GGML_OP_ROPE:
                સ્થિતિ GGML_OP_ROPE_BACK:
                    {
                        cur = ggml_type_size(GGML_TYPE_F32) * ગાંઠ->ne[0] * n_tasks;
                    } તોડો;
                સ્થિતિ GGML_OP_CONV_TRANSPOSE_1D:
                    {
                        GGML_ASSERT(ગાંઠ->src[0]->ne[3] == 1);
                        GGML_ASSERT(ગાંઠ->src[1]->ne[2] == 1);
                        GGML_ASSERT(ગાંઠ->src[1]->ne[3] == 1);

                        અચલ int64_t ne00 = ગાંઠ->src[0]->ne[0];  // K
                        અચલ int64_t ne01 = ગાંઠ->src[0]->ne[1];  // Cout
                        અચલ int64_t ne02 = ગાંઠ->src[0]->ne[2];  // Cin
                        અચલ int64_t ne10 = ગાંઠ->src[1]->ne[0];  // L
                        અચલ int64_t ne11 = ગાંઠ->src[1]->ne[1];  // Cin

                        જો ((ગાંઠ->src[0]->type == GGML_TYPE_F16 ||
                             ગાંઠ->src[0]->type == GGML_TYPE_BF16) &&
                            ગાંઠ->src[1]->type == GGML_TYPE_F32) {
                            cur += કદ(ggml_fp16_t)*ne00*ne01*ne02;
                            cur += કદ(ggml_fp16_t)*ne10*ne11;
                        } નહીંતર જો (ગાંઠ->src[0]->type == GGML_TYPE_F32 &&
                                   ગાંઠ->src[1]->type == GGML_TYPE_F32) {
                            cur += કદ(દશાંશ)*ne00*ne01*ne02;
                            cur += કદ(દશાંશ)*ne10*ne11;
                        } નહીંતર {
                            GGML_ABORT("fatal error");
                        }
                    } તોડો;
                સ્થિતિ GGML_OP_CONV_2D:
                સ્થિતિ GGML_OP_CONV_3D:
                    {
                        cur = GGML_IM2COL_WORK_SIZE;
                    } તોડો;
                સ્થિતિ GGML_OP_CONV_TRANSPOSE_2D:
                    {
                        અચલ int64_t ne00 = ગાંઠ->src[0]->ne[0]; // W
                        અચલ int64_t ne01 = ગાંઠ->src[0]->ne[1]; // H
                        અચલ int64_t ne02 = ગાંઠ->src[0]->ne[2]; // Channels Out
                        અચલ int64_t ne03 = ગાંઠ->src[0]->ne[3]; // Channels In

                        અચલ int64_t ne10 = ગાંઠ->src[1]->ne[0]; // W
                        અચલ int64_t ne11 = ગાંઠ->src[1]->ne[1]; // H
                        અચલ int64_t ne12 = ગાંઠ->src[1]->ne[2]; // Channels In
                        અચલ int64_t ne13 = ગાંઠ->src[1]->ne[3]; // Batch

                        GGML_ASSERT(ગાંઠ->src[0]->type == GGML_TYPE_F16 || ગાંઠ->src[0]->type == GGML_TYPE_F32);
                        GGML_ASSERT(ગાંઠ->src[1]->type == GGML_TYPE_F32);

                        cur += ggml_type_size(ગાંઠ->src[0]->type) * ne00 * ne01 * ne02 * ne03;
                        cur += ggml_type_size(ગાંઠ->src[0]->type) * ne10 * ne11 * ne12 * ne13;

                    } તોડો;
                સ્થિતિ GGML_OP_ટોચ_K:
                    {
                        cur += કદ(int32_t)*ગાંઠ->src[0]->ne[0]*n_tasks;
                    } તોડો;
                સ્થિતિ GGML_OP_FLASH_ATTN_EXT:
                    {
                        અચલ int64_t neq2 = ગાંઠ->src[0]->ne[2]; // number of query heads
                        અચલ int64_t DK = ગાંઠ->src[1]->ne[0];
                        અચલ int64_t DV = ગાંઠ->src[2]->ne[0];

                        // Tiled flash attention scratch (tile sizes defined in common.h)
                        // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
                        size_t prefill  = કદ(દશાંશ)*(GGML_FA_TILE_Q*DK + 2*GGML_FA_TILE_Q*GGML_FA_TILE_KV + GGML_FA_TILE_Q*DV + GGML_FA_TILE_KV*DV + GGML_FA_TILE_KV*DK)*n_tasks;

                        // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
                        // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
                        size_t n_chunks = n_tasks;
                        size_t decode   = કદ(દશાંશ)*(neq2*n_chunks*(2+DV) + n_tasks*(DK + 2*DV));

                        cur += MAX(prefill, decode);
                    } તોડો;
                સ્થિતિ GGML_OP_FLASH_ATTN_BACK:
                    {
                        અચલ int64_t    D = ગાંઠ->src[0]->ne[0];
                        અચલ int64_t ne11 = ggml_up(ગાંઠ->src[1]->ne[1], GGML_SOFT_MAX_UNROLL);
                        અચલ int64_t mxDn = MAX(D, ne11) * 2; // *2 because of S and SM in ggml_compute_forward_flash_attn_back
                        જો (ગાંઠ->src[1]->type == GGML_TYPE_F32) {
                            cur  = કદ(દશાંશ)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += કદ(દશાંશ)*mxDn*n_tasks; // this is overestimated by x2
                        } નહીંતર જો (ગાંઠ->src[1]->type == GGML_TYPE_F16) {
                            cur  = કદ(દશાંશ)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += કદ(દશાંશ)*mxDn*n_tasks; // this is overestimated by x2
                        } નહીંતર જો (ગાંઠ->src[1]->type == GGML_TYPE_BF16) {
                            cur  = કદ(દશાંશ)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += કદ(દશાંશ)*mxDn*n_tasks; // this is overestimated by x2
                        }
                    } તોડો;

                સ્થિતિ GGML_OP_CROSS_ENTROPY_LOSS:
                    {
                        cur = ggml_type_size(ગાંઠ->type)*(n_tasks + ગાંઠ->src[0]->ne[0]*n_tasks);
                    } તોડો;
                સ્થિતિ GGML_OP_GATED_DELTA_NET:
                    {
                        અચલ int64_t S_v = ગાંઠ->src[2]->ne[0];
                        અચલ int64_t K   = ggml_get_op_params_i32(ગાંઠ, 0);
                        અચલ int64_t per_thread = S_v + (K > 1 ? S_v * S_v : 0);
                        cur = per_thread * કદ(દશાંશ) * n_tasks;
                    } તોડો;
                સ્થિતિ GGML_OP_ગણતરી:
                    {
                        GGML_ABORT("fatal error");
                    }
                સ્થિતિ GGML_OP_LIGHTNING_INDEXER:
                    {
                        // temp buffer for dequantizing lightning indexer keys
                        અચલ int64_t ne10 = ગાંઠ->src[1]->ne[0];
                        cur += કદ(દશાંશ)*ne10*n_tasks;
                    } તોડો;
                મૂળભૂત:
                    તોડો;
            }
        }

        work_size = MAX(work_size, cur);
    }

    જો (work_size > 0) {
        work_size += CACHE_લીટી_SIZE*(n_threads);
    }

    cplan.threadpool = threadpool;
    cplan.n_threads  = MIN(max_tasks, n_threads);
    cplan.work_size  = work_size;
    cplan.work_data  = ખાલી;

    પાછા cplan;
}


// Try to fuse the current node with subsequent nodes for better performance.
// Returns the number of nodes skipped by fusion (>=1), or 0 if no fusion was applied.
સ્થિર bool ggml_cpu_disable_fusion = false;  // initialized once in ggml_cpu_init(), read-only afterwards

સ્થિર પૂર્ણાંક ggml_cpu_try_fuse_ops(
        અચલ સંરચના ggml_cgraph * cgraph,
        અચલ પૂર્ણાંક ગાંઠ_n,
        અચલ સંરચના ggml_compute_params * params,
        અચલ સંરચના ggml_cplan * cplan) {

    જો (ggml_cpu_disable_fusion || cplan->use_ref) {
        પાછા 0;
    }

    સંરચના ggml_tensor * ગાંઠ = cgraph->nodes[ગાંઠ_n];

    જો (ગાંઠ->op == GGML_OP_RMS_NORM) {
        // RMS_NORM + MUL fusion
        અચલ ગણના ggml_op fuse_ops[] = { GGML_OP_RMS_NORM, GGML_OP_MUL };
        જો (ggml_can_fuse(cgraph, ગાંઠ_n, fuse_ops, 2)) {
            સંરચના ggml_tensor * mul_ગાંઠ = cgraph->nodes[ગાંઠ_n + 1];
            અચલ સંરચના ggml_tensor * mul_w = (mul_ગાંઠ->src[0] == ગાંઠ)
                ? mul_ગાંઠ->src[1] : mul_ગાંઠ->src[0];
            જો (ગાંઠ->src[0]->type  == GGML_TYPE_F32 &&
                mul_ગાંઠ->type      == GGML_TYPE_F32 &&
                mul_w->type         == GGML_TYPE_F32 &&
                mul_w->ne[0]        == ગાંઠ->ne[0]   &&
                mul_w->nb[0]        == કદ(દશાંશ)) {

                ggml_compute_forward_rms_norm_mul_fused(params, ગાંઠ, mul_ગાંઠ);
                પાછા 1;
            }
        }
    }

    પાછા 0;
}

સ્થિર thread_ret_t ggml_graph_compute_thread(શૂન્ય * data) {
    સંરચના ggml_compute_state * state = (સંરચના ggml_compute_state *) data;
    સંરચના ggml_threadpool    * tp    = state->threadpool;

    અચલ સંરચના ggml_cgraph * cgraph = tp->cgraph;
    અચલ સંરચના ggml_cplan  * cplan  = tp->cplan;

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(state->ith);
#નહીંતર
    set_numa_thread_affinity(state->ith);
#endif

    સંરચના ggml_compute_params params = {
        /*.ith        =*/ state->ith,
        /*.nth        =*/ atomic_load_explicit(&tp->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK,
        /*.wsize      =*/ cplan->work_size,
        /*.wdata      =*/ cplan->work_data,
        /*.threadpool =*/ tp,
        /*.use_ref    =*/ cplan->use_ref,
    };

#ifdef GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p\n", state->ith, (અચલ શૂન્ય *)cplan);
#નહીંતર
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p last-graph %d\n", state->ith, (અચલ શૂન્ય *)cplan, state->last_graph);
#endif

    માટે (પૂર્ણાંક ગાંઠ_n = 0; ગાંઠ_n < cgraph->n_nodes && atomic_load_explicit(&tp->abort, memory_order_relaxed) != ગાંઠ_n; ગાંઠ_n++) {
        સંરચના ggml_tensor * ગાંઠ = cgraph->nodes[ગાંઠ_n];

        જો (ggml_op_is_empty(ગાંઠ->op)) {
            // skip NOPs
            ચાલુ;
        }

        જો ((ગાંઠ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ચાલુ;
        }

        // TODO: move fused-op detection into ggml_graph_plan so fusion decisions are made once at planning time
        // Try fused ops, fall back to normal compute
        અચલ પૂર્ણાંક n_fused = ggml_cpu_try_fuse_ops(cgraph, ગાંઠ_n, &params, cplan);
        જો (n_fused > 0) {
            ગાંઠ_n += n_fused;
        } નહીંતર {
            ggml_compute_forward(&params, ગાંઠ);
        }

        જો (state->ith == 0 && cplan->abort_callback &&
                cplan->abort_callback(cplan->abort_callback_data)) {
            atomic_store_explicit(&tp->abort, ગાંઠ_n + 1, memory_order_relaxed);
            tp->ec    = GGML_STATUS_ABORTED;
        }

        જો (ગાંઠ_n + 1 < cgraph->n_nodes) {
            ggml_barrier(state->threadpool);
        }
    }

#ifdef GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p\n", state->ith, (અચલ શૂન્ય *)cplan);
#નહીંતર
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p last-graph %d\n", state->ith, (અચલ શૂન્ય *)cplan, state->last_graph);
#endif

    ggml_barrier(state->threadpool);

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(state->ith);
#endif

    પાછા 0;
}

#ifndef GGML_USE_OPENMP

// check if thread is ready to proceed (exit from polling or sleeping)
// returns true if loops should exit, sets state->pending to indicate new work
સ્થિર અંતઃરેખ bool ggml_graph_compute_thread_ready(સંરચના ggml_compute_state * state) {
    સંરચના ggml_threadpool * threadpool = state->threadpool;

    જો (state->pending || threadpool->stop || threadpool->pause) { પાછા true; }

    // check for new graph/work
    પૂર્ણાંક n_graph   = atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed);
    પૂર્ણાંક n_threads = n_graph & GGML_THREADPOOL_N_THREADS_MASK;
    જો (n_graph != state->last_graph) {
        state->pending    = (state->ith < n_threads);
        state->last_graph = n_graph;
        પાછા true;
    }

    પાછા false;
}

// sync thread state after polling
સ્થિર અંતઃરેખ શૂન્ય ggml_graph_compute_thread_sync(સંરચના ggml_compute_state * state) {
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&state->threadpool->n_graph, 0, memory_order_seq_cst);
    #નહીંતર
    atomic_thread_fence(memory_order_seq_cst);
    #endif
    UNUSED(state);
}

સ્થિર અંતઃરેખ bool ggml_graph_compute_poll_for_work(સંરચના ggml_compute_state * state) {
    સંરચના ggml_threadpool * threadpool = state->threadpool;

    // This seems to make 0 ... 100 a decent range for polling level across modern processors.
    // Perhaps, we can adjust it dynamically based on load and things.
    અચલ uint64_t n_rounds = 1024UL * 128 * threadpool->poll;

    માટે (uint64_t i=0; !ggml_graph_compute_thread_ready(state) && i < n_rounds; i++) {
        // No new work. Keep polling.
        ggml_thread_cpu_relax();
    }

    પાછા state->pending;
}

સ્થિર અંતઃરેખ bool ggml_graph_compute_check_for_work(સંરચના ggml_compute_state * state) {
    સંરચના ggml_threadpool * threadpool = state->threadpool;

    જો (ggml_graph_compute_poll_for_work(state)) {
        ggml_graph_compute_thread_sync(state);
        પાછા state->pending;
    }

    ggml_mutex_lock_shared(&threadpool->mutex);
    જ્યારે (!ggml_graph_compute_thread_ready(state)) {
        // No new work. Wait for the signal.
        GGML_PRINT_DEBUG("thread #%d waiting for work (sleeping)\n", state->ith);
        ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
    }
    ggml_mutex_unlock_shared(&threadpool->mutex);

    પાછા state->pending;
}

સ્થિર thread_ret_t ggml_graph_compute_secondary_thread(શૂન્ય* data) {
    સંરચના ggml_compute_state * state = (સંરચના ggml_compute_state *) data;
    સંરચના ggml_threadpool * threadpool = state->threadpool;

    ggml_thread_apply_priority(threadpool->prio);
    જો (ggml_thread_cpumask_is_valid(state->cpumask)) {
        ggml_thread_apply_affinity(state->cpumask);
    }

    જ્યારે (true) {
        // Check if we need to sleep
        જ્યારે (threadpool->pause) {
            GGML_PRINT_DEBUG("thread #%d inside pause loop\n", state->ith);
            ggml_mutex_lock_shared(&threadpool->mutex);
            જો (threadpool->pause) {
                ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
            }
            GGML_PRINT_DEBUG("thread #%d resuming after wait\n", state->ith);
            ggml_mutex_unlock_shared(&threadpool->mutex);
        }

        // This needs to be checked for after the cond_wait
        જો (threadpool->stop) તોડો;

        // Check if there is new work
        // The main thread is the only one that can dispatch new work

        ggml_graph_compute_check_for_work(state);
        જો (state->pending) {
            state->pending = false;
            ggml_graph_compute_thread(state);
        }
    }

    પાછા (thread_ret_t) 0;
}

// Start processing new graph
સ્થિર શૂન્ય ggml_graph_compute_kickoff(સંરચના ggml_threadpool * threadpool, પૂર્ણાંક n_threads)
{
    // Always take the mutex here because the worker threads are doing hybrid poll/wait

    ggml_mutex_lock(&threadpool->mutex);

    // Update the number of active threads and the graph count
    પૂર્ણાંક n_graph = atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed) >> GGML_THREADPOOL_N_THREADS_BITS;
    n_graph = ((n_graph + 1) << GGML_THREADPOOL_N_THREADS_BITS) | (n_threads & GGML_THREADPOOL_N_THREADS_MASK);

    GGML_PRINT_DEBUG("compute-kickoff: n_threads %d n_graph %d\n", n_threads, n_graph);

    // Indicate the graph is ready to be processed
    // We need the full seq-cst fence here because of the polling threads (used in thread_sync)
    atomic_store_explicit(&threadpool->n_graph, n_graph, memory_order_seq_cst);

    જો (threadpool->pause) {
       // Update main thread prio and affinity to match the threadpool settings
       ggml_thread_apply_priority(threadpool->prio);
       જો (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
           ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
       }

       // resume does cond broadcast
       ggml_threadpool_resume_locked(threadpool);
    } નહીંતર {
       ggml_cond_broadcast(&threadpool->cond);
    }

    ggml_mutex_unlock(&threadpool->mutex);
}

#endif // GGML_USE_OPENMP

સ્થિર સંરચના ggml_threadpool * ggml_threadpool_new_impl(
    સંરચના ggml_threadpool_params * tpp,
               સંરચના ggml_cgraph * cgraph,
                સંરચના ggml_cplan * cplan) {

    સંરચના ggml_threadpool * threadpool =
        ggml_aligned_malloc(કદ(સંરચના ggml_threadpool));
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
        threadpool->workers          = ખાલી;
        threadpool->n_threads        = tpp->n_threads;
        threadpool->poll             = tpp->poll;
        threadpool->prio             = tpp->prio;
        threadpool->ec               = GGML_STATUS_SUCCESS;
    }

    // Allocate and init workers state
    અચલ size_t workers_size = કદ(સંરચના ggml_compute_state) * tpp->n_threads;
    સંરચના ggml_compute_state * workers = ggml_aligned_malloc(workers_size);

    memset(workers, 0, workers_size);
    માટે (પૂર્ણાંક j = 0; j < tpp->n_threads; j++) {
        workers[j].threadpool = threadpool;
        workers[j].ith        = j;
    }

    threadpool->workers = workers;

#ifdef GGML_USE_OPENMP
    int32_t cpumask_iter = 0;

    // Compute CPU masks for each thread
    માટે (પૂર્ણાંક j = 0; j < tpp->n_threads; j++) {
        ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);
    }
#નહીંતર // GGML_USE_OPENMP
    ggml_mutex_init(&threadpool->mutex);
    ggml_cond_init(&threadpool->cond);

    // Spin the threads for all workers, and update CPU placements.
    // Place the main thread last (towards the higher numbered CPU cores).

    int32_t cpumask_iter = 0;

    માટે (પૂર્ણાંક j = 1; j < tpp->n_threads; j++) {
        ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);

        int32_t rc = ggml_thread_create(&workers[j].thrd, ખાલી, ggml_graph_compute_secondary_thread, &workers[j]);
        GGML_ASSERT(rc == 0);
    }

    ggml_thread_cpumask_next(tpp->cpumask, workers[0].cpumask, tpp->strict_cpu, &cpumask_iter);

    જો (!threadpool->pause) {
        // Update main thread prio and affinity at the start, otherwise we'll do it in resume
        ggml_thread_apply_priority(threadpool->prio);
        જો (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
            ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
        }
    }
#endif // GGML_USE_OPENMP

    પાછા threadpool;
}

સંરચના ggml_threadpool * ggml_threadpool_new(સંરચના ggml_threadpool_params * tpp) {
    પાછા ggml_threadpool_new_impl(tpp, ખાલી, ખાલી);
}

ગણના ggml_status ggml_graph_compute(સંરચના ggml_cgraph * cgraph, સંરચના ggml_cplan * cplan) {
    ggml_cpu_init();

    GGML_ASSERT(cplan);
    GGML_ASSERT(cplan->n_threads > 0);
    GGML_ASSERT(cplan->work_size == 0 || cplan->work_data != ખાલી);

    પૂર્ણાંક n_threads                               = cplan->n_threads;
    સંરચના ggml_threadpool * threadpool = cplan->threadpool;

    bool disposable_threadpool = false;

    જો (threadpool == ખાલી) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
        disposable_threadpool = true;

        સંરચના ggml_threadpool_params ttp = ggml_threadpool_params_default(n_threads);
        threadpool = ggml_threadpool_new_impl(&ttp, cgraph, cplan);
    } નહીંતર {
        // Reset some of the parameters that need resetting
        // No worker threads should be accessing the parameters below at this stage
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->current_chunk    = 0;
        threadpool->abort            = -1;
        threadpool->ec               = GGML_STATUS_SUCCESS;
    }

#ifdef GGML_USE_OPENMP
    જો (n_threads > 1) {
        #pragma omp parallel અંક_threads(n_threads)
        {
            #pragma omp single
            {
                // update the number of threads from the actual number of threads that we got from OpenMP
                n_threads = omp_get_અંક_threads();
                atomic_store_explicit(&threadpool->n_graph, n_threads, memory_order_relaxed);
            }

            // Apply thread CPU mask and priority
            પૂર્ણાંક ith = omp_get_thread_અંક();

            ggml_thread_apply_priority(threadpool->prio);
            જો (ggml_thread_cpumask_is_valid(threadpool->workers[ith].cpumask)) {
                ggml_thread_apply_affinity(threadpool->workers[ith].cpumask);
            }
            ggml_graph_compute_thread(&threadpool->workers[ith]);
        }
    } નહીંતર {
        atomic_store_explicit(&threadpool->n_graph, 1, memory_order_relaxed);
        ggml_graph_compute_thread(&threadpool->workers[0]);
    }
#નહીંતર
    જો (n_threads > threadpool->n_threads) {
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

    ગણના ggml_status ret = threadpool->ec;

    જો (disposable_threadpool) {
        ggml_threadpool_free(threadpool);
    }

    પાછા ret;
}

ગણના ggml_status ggml_graph_compute_with_ctx(સંરચના ggml_context * ctx, સંરચના ggml_cgraph * cgraph, પૂર્ણાંક n_threads) {
    સંરચના ggml_cplan cplan = ggml_graph_plan(cgraph, n_threads, ખાલી);

    cplan.work_data = (uint8_t *)ggml_new_buffer(ctx, cplan.work_size);

    પાછા ggml_graph_compute(cgraph, &cplan);
}

શૂન્ય ggml_cpu_fp32_to_fp32(અચલ દશાંશ * x, દશાંશ * y, int64_t n) {
    memcpy(y, x, n * કદ(દશાંશ));
}

શૂન્ય ggml_cpu_fp32_to_fp16(અચલ દશાંશ * x, ggml_fp16_t * y, int64_t n) {
    int64_t i = 0;
#જો defined(__F16C__)
#જો defined(__AVX512F__)
    માટે (; i + 15 < n; i += 16) {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        __m256i y_vec = _mm512_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256((__m256i *)(y + i), y_vec);
    }
#endif
    માટે (; i + 7 < n; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m128i y_vec = _mm256_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i *)(y + i), y_vec);
    }
    માટે (; i + 3 < n; i += 4) {
        __m128 x_vec = _mm_loadu_ps(x + i);
        __m128i y_vec = _mm_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storel_epi64((__m128i *)(y + i), y_vec);
    }
#અથવાજો defined(__riscv_zvfh)
    માટે (પૂર્ણાંક vl; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(&x[i], vl);
        vfloat16m1_t vy = __riscv_vfncvt_f_f_w_f16m1(vx, vl);
        __riscv_vse16_v_f16m1((_Float16 *)&y[i], vy, vl);
    }
#અથવાજો defined(__VXE__) || defined(__VXE2__)
    માટે (; i + 7 < n; i += 8) {
        અચલ uint32x4_t v_yl = __lzs_f32cx4_to_f16(vec_xl(0, x + i + 0));
        અચલ uint32x4_t v_yh = __lzs_f32cx4_to_f16(vec_xl(0, x + i + 4));
        vec_xst(vec_pack(v_yl, v_yh), 0, (uint16_t *)(y + i));
    }
#endif
    માટે (; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(x[i]);
    }
}

શૂન્ય ggml_cpu_fp16_to_fp32(અચલ ggml_fp16_t * x, દશાંશ * y, int64_t n) {
    int64_t i = 0;
#જો defined(__F16C__)
#જો defined(__AVX512F__)
    માટે (; i + 15 < n; i += 16) {
        __m256i x_vec = _mm256_loadu_si256((અચલ __m256i *)(x + i));
        __m512 y_vec = _mm512_cvtph_ps(x_vec);
        _mm512_storeu_ps(y + i, y_vec);
    }
#endif
    માટે (; i + 7 < n; i += 8) {
        __m128i x_vec = _mm_loadu_si128((અચલ __m128i *)(x + i));
        __m256 y_vec = _mm256_cvtph_ps(x_vec);
        _mm256_storeu_ps(y + i, y_vec);
    }
    માટે (; i + 3 < n; i += 4) {
        __m128i x_vec = _mm_loadl_epi64((અચલ __m128i *)(x + i));
        __m128 y_vec = _mm_cvtph_ps(x_vec);
        _mm_storeu_ps(y + i, y_vec);
    }

#અથવાજો defined(__riscv_v_intrinsic) && defined(__riscv_zvfhmin)
    // calculate step size
    અચલ પૂર્ણાંક epr = __riscv_vsetvlmax_e16m2();
    અચલ પૂર્ણાંક step = epr * 2;
    અચલ પૂર્ણાંક np = (n & ~(step - 1));

    // unroll by 2
    માટે (; i < np; i += step) {
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((અચલ _Float16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vfloat16m2_t ax1 = __riscv_vle16_v_f16m2((અચલ _Float16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvt_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    પૂર્ણાંક vl;
    માટે (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((અચલ _Float16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }

#endif

    માટે (; i < n; ++i) {
        y[i] = GGML_CPU_FP16_TO_FP32(x[i]);
    }
}

શૂન્ય ggml_cpu_fp32_to_bf16(અચલ દશાંશ * x, ggml_bf16_t * y, int64_t n) {
    int64_t i = 0;
    માટે (; i < n; ++i) {
        y[i] = GGML_FP32_TO_BF16(x[i]);
    }
}

શૂન્ય ggml_cpu_fp32_to_i32(અચલ દશાંશ * x, int32_t * y, int64_t n) {
    int64_t i = 0;
    માટે (; i < n; ++i) {
        y[i] = x[i];
    }
}

શૂન્ય ggml_cpu_bf16_to_fp32(અચલ ggml_bf16_t * x, દશાંશ * y, int64_t n) {
    int64_t i = 0;
#જો defined(__AVX2__)
#જો defined(__AVX512F__)
    માટે (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i,
                        _mm512_castsi512_ps(
                            _mm512_slli_epi32(
                                _mm512_cvtepu16_epi32(
                                    _mm256_loadu_si256(
                                        (અચલ __m256i *)(x + i))),
                                16)));
    }
#endif
    માટે (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i,
                        _mm256_castsi256_ps(
                            _mm256_slli_epi32(
                                _mm256_cvtepu16_epi32(
                                    _mm_loadu_si128(
                                        (અચલ __m128i *)(x + i))),
                                16)));
    }
#અથવાજો defined(__riscv_v_intrinsic) && defined(__riscv_zvfbfmin)
    // calculate step size
    અચલ પૂર્ણાંક epr = __riscv_vsetvlmax_e16m2();
    અચલ પૂર્ણાંક step = epr * 2;
    અચલ પૂર્ણાંક np = (n & ~(step - 1));

    // unroll by 2
    માટે (; i < np; i += step) {
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((અચલ __bf16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vbfloat16m2_t ax1 = __riscv_vle16_v_bf16m2((અચલ __bf16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    પૂર્ણાંક vl;
    માટે (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((અચલ __bf16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }
#endif
    માટે (; i < n; i++) {
        y[i] = GGML_BF16_TO_FP32(x[i]);
    }
}

પૂર્ણાંક ggml_cpu_has_avx(શૂન્ય) {
#જો defined(__AVX__)
    પાછા 1;
#નહીંતર
    પાછા 0;
#endif
}

પૂર્ણાંક ggml_cpu_has_avx_vnni(શૂન્ય) {
#જો defined(__AVXVNNI__)
    પાછા 1;
#નહીંતર
    પાછા 0;
#endif
}

પૂર્ણાંક ggml_cpu_has_avx2(શૂન્ય) {
#જો defined(__AVX2__)
    પાછા 1;
#નહીંતર
    પાછા 0;
#endif
}

પૂર્ણાંક ggml_cpu_has_avx512(શૂન્ય) {
#જો defined(__AVX512F__)
    પાછા 1;
#નહીંતર
    પાછા 0;
#endif
}

પૂર્ણાંક ggml_cpu_has_avx512_vbmi(શૂન્ય) {
#જો defined(__AVX512VBMI__)
    પાછા 1;
#નહીંતર
    પાછા 0;
#endif
}

પૂર્ણાંક ggml_cpu_has_avx512_vnni(શૂન્ય) {
#જો defined(__AVX512VNNI__)
    પાછા 1;
#નહીંતર
    પાછા 0;
#endif
}

પૂર્ણાંક ggml_cpu_has_avx512_bf16(શૂન્ય) {
#જો defined(__AVX512BF16__)
    પાછા 1;
#નહીંતર
    પાછા 0;
#endif
}

પૂર્ણાંક ggml_cpu_has_amx_int8(શૂન્ય) {
#જો defined(__AMX_INT8__)
    પાછા 1;
#નહીંતર
    પાછા 0;
#endif
}

પૂર્ણાંક ggml_cpu_has_bmi2(શૂન્ય) {
#જો defined(__BMI2__)
    પાછા 1;
#નહીંતર
    પાછા 0;
#endif
}

પૂર્ણાંક ggml_cpu_has_fma(શૂન્ય) {
#જો defined(__FMA__)
    પાછા 1;
#નહીંતર
    પાછા 0;
#endif
}

પૂર્ણાંક ggml_cpu_has_arm_fma(શૂન્ય) {
#જો defined(__ARM_FEATURE_FMA)
    પાછા 1;
#નહીંતર
    પાછા 0;
#endif
}

પૂર્ણાંક ggml_cpu_has_riscv_v(શૂન્ય) {
#જો defined(__riscv_v_intrinsic)
    પાછા 1;
#નહીંતર
    પાછા 0;
#endif
}

પૂર્ણાંક ggml_cpu_get_rvv_vlen(શૂન્ય) {
#જો defined(__riscv) && defined(__riscv_v_intrinsic)
    પાછા ggml_riscv_arch_features.rvv_vlen;
#નહીંતર
    પાછા 0;
#endif
}

પૂર્ણાંક ggml_cpu_has_f16c(શૂન્ય) {
#જો defined(__F16C__)
    પાછા 1;
#નહીંતર
    પાછા 0;
#endif
}

પૂર્ણાંક ggml_cpu_has_fp16_va(શૂન્ય) {
#જો defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    પાછા 1;
#નહીંતર
    પાછા 0;
#endif
}

પૂર્ણાંક ggml_cpu_has_wasm_simd(શૂન્ય) {
#જો defined(__wasm_simd128__)
    પાછા 1;
#નહીંતર
    પાછા 0;
#endif
}

પૂર્ણાંક ggml_cpu_has_llamafile(શૂન્ય) {
#જો defined(GGML_USE_LLAMAFILE)
    પાછા 1;
#નહીંતર
    પાછા 0;
#endif
}

પૂર્ણાંક ggml_cpu_has_sse3(શૂન્ય) {
#જો defined(__SSE3__)
    પાછા 1;
#નહીંતર
    પાછા 0;
#endif
}

પૂર્ણાંક ggml_cpu_has_ssse3(શૂન્ય) {
#જો defined(__SSSE3__)
    પાછા 1;
#નહીંતર
    પાછા 0;
#endif
}

પૂર્ણાંક ggml_cpu_has_vsx(શૂન્ય) {
#જો defined(__POWER9_VECTOR__)
    પાછા 1;
#નહીંતર
    પાછા 0;
#endif
}

પૂર્ણાંક ggml_cpu_has_vxe(શૂન્ય) {
#જો defined(__VXE__) || defined(__VXE2__)
    પાછા 1;
#નહીંતર
    પાછા 0;
#endif
}

પૂર્ણાંક ggml_cpu_has_neon(શૂન્ય) {
#જો defined(__ARM_ARCH) && defined(__ARM_NEON)
    પાછા 1;
#નહીંતર
    પાછા 0;
#endif
}

પૂર્ણાંક ggml_cpu_has_dotprod(શૂન્ય) {
#જો defined(__ARM_ARCH) && defined(__ARM_FEATURE_DOTPROD)
    પાછા 1;
#નહીંતર
    પાછા 0;
#endif
}

પૂર્ણાંક ggml_cpu_has_sve(શૂન્ય) {
#જો defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    પાછા 1;
#નહીંતર
    પાછા 0;
#endif
}

પૂર્ણાંક ggml_cpu_has_matmul_int8(શૂન્ય) {
#જો defined(__ARM_ARCH) && defined(__ARM_FEATURE_MATMUL_INT8)
    પાછા 1;
#નહીંતર
    પાછા 0;
#endif
}

પૂર્ણાંક ggml_cpu_get_sve_cnt(શૂન્ય) {
#જો defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    પાછા ggml_arm_arch_features.sve_cnt;
#નહીંતર
    પાછા 0;
#endif
}

પૂર્ણાંક ggml_cpu_has_sme(શૂન્ય) {
#જો defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME)
    પાછા 1;
#નહીંતર
    પાછા 0;
#endif
}

પૂર્ણાંક ggml_cpu_has_sme2(શૂન્ય) {
#જો defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME2)
    પાછા 1;
#નહીંતર
    પાછા 0;
#endif
}

શૂન્ય ggml_cpu_init(શૂન્ય) {
    // needed to initialize ggml_time
    {
        સંરચના ggml_init_params params = { 0, ખાલી, false };
        સંરચના ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    ggml_critical_section_start();

    સ્થિર bool is_first_call = true;

    જો (is_first_call) {
        // initialize GELU, Quick GELU, SILU and EXP F32 tables
        {
            અચલ uint64_t t_start = ggml_time_us(); UNUSED(t_start);

            માટે (પૂર્ણાંક i = 0; i < (1 << 16); ++i) {
                સંઘ {
                    uint16_t u16;
                    ggml_fp16_t fp16;
                } u = {i};
                દશાંશ f = GGML_COMPUTE_FP16_TO_FP32(u.fp16);
                ggml_table_f32_f16[i] = f;
                ggml_table_gelu_f16[i] = GGML_CPU_FP32_TO_FP16(ggml_gelu_f32(f));
                ggml_table_gelu_quick_f16[i] = GGML_CPU_FP32_TO_FP16(ggml_gelu_quick_f32(f));
            }

            // initialize E8M0 half table (256 entries)
            માટે (પૂર્ણાંક i = 0; i < (1 << 8); ++i) {
                ggml_table_f32_e8m0_half[i] = GGML_E8M0_TO_FP32_HALF(i);
            }

            // initialize UE4M3 table (256 entries)
            માટે (પૂર્ણાંક i = 0; i < (1 << 8); ++i) {
                ggml_table_f32_ue4m3[i] = ggml_ue4m3_to_fp32(i);
            }

            અચલ uint64_t t_end = ggml_time_us(); UNUSED(t_end);

            GGML_PRINT_DEBUG("%s: GELU, Quick GELU, SILU and EXP tables initialized in %f ms\n", __func__, (t_end - t_start)/1000.0);

#ifdef GGML_USE_OPENMP
            //if (!getenv("OMP_WAIT_POLICY")) {
            //    // set the wait policy to active, so that OpenMP threads don't sleep
            //    setenv("OMP_WAIT_POLICY", "active", 0)
            //}

            જો (!getenv("KMP_BLOCKTIME")) {
                // set the time to wait before sleeping a thread
                // this is less aggressive than setting the wait policy to active, but should achieve similar results in most cases
#ifdef _WIN32
                _putenv_s("KMP_BLOCKTIME", "200"); // 200ms
#નહીંતર
                setenv("KMP_BLOCKTIME", "200", 0); // 200ms
#endif
            }
#endif
        }

#જો defined(__ARM_ARCH)
        ggml_init_arm_arch_features();
#endif

#જો defined(__riscv)
        ggml_init_riscv_arch_features();
#endif

        {
            અચલ અક્ષર * env = getenv("GGML_CPU_DISABLE_FUSION");
            ggml_cpu_disable_fusion = (env != ખાલી && atoi(env) == 1);
        }

        is_first_call = false;
    }

    ggml_critical_section_end();
}
