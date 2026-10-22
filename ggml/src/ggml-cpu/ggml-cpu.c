#تعریف_کرو _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#تعریف_کرو USE_MATH_متعین_کرتا // For M_PI on MSVC

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
#وگرنہ_اگر !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#شامل <alloca.h>
#اگر_ختم

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
#اگر_ختم

#اگر_تعریف_شدہ GGML_USE_OPENMP
#شامل <omp.h>
#اگر_ختم

#اگر defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_MATMUL_INT8)
#تعریف_منسوخ GGML_USE_LLAMAFILE
#اگر_ختم

#اگر_تعریف_شدہ GGML_USE_LLAMAFILE
#شامل "llamafile/sgemm.h"
#اگر_ختم

#اگر_تعریف_شدہ GGML_USE_CPU_RISCV64_SPACEMIT
#    include "spacemit/ime.h"
#اگر_ختم

// Note: once we move threading into a separate C++ file
// will use std::hardware_destructive_interference_size instead of hardcoding it here
// and we'll use C++ attribute syntax.
#تعریف_کرو GGML_CACHE_سطر  64

#اگر defined(__clang__) || defined(__GNUC__)
#تعریف_کرو GGML_CACHE_ALIGN __attribute__((aligned(GGML_CACHE_سطر)))
#اگر_ختم

#اگر defined(__has_feature)
#اگر __has_feature(thread_sanitizer)
#تعریف_کرو GGML_TSAN_ENABLED 1
#اگر_ختم
#ورنہ  // __has_feature
#اگر defined(__SANITIZE_THREAD__)
#تعریف_کرو GGML_TSAN_ENABLED 1
#اگر_ختم
#اگر_ختم // __has_feature

#تعریف_کرو غیر_مستعمل GGML_غیر_مستعمل
#تعریف_کرو تبادلہ(x, y, T) کرو { T تبادلہ = x; (x) = y; (y) = تبادلہ; } جب (0)

// precomputed f32 table for f16 (256 KB) (simd-mappings.h)
اعشاری ggml_جدول_f32_f16[1 << 16];

// precomputed f32 table for e8m0 half (1 KB) (simd-mappings.h)
اعشاری ggml_جدول_f32_e8m0_half[1 << 8];

// precomputed f32 table for ue4m3 (1 KB) (simd-mappings.h)
اعشاری ggml_جدول_f32_ue4m3[1 << 8];

#اگر defined(__ARM_ARCH)
ساخت ggml_arm_arch_features_type {
    صحیح sve_cnt;
} ggml_arm_arch_features = { 0 };
#اگر_ختم

#اگر defined(__riscv)
ساخت ggml_riscv_arch_features_type {
    صحیح rvv_vlen;
} ggml_riscv_arch_features = { 0 };
#اگر_ختم

#اگر defined(_WIN32)

#تعریف_کرو WIN32_LEAN_AND_اوسط
#اگر_غیر_تعریف_شدہ NOMINMAX
    #تعریف_کرو NOMINMAX
#اگر_ختم
#شامل <windows.h>

#اگر defined(_MSC_VER) && !defined(__clang__)
#تعریف_کرو GGML_CACHE_ALIGN __declspec(align(GGML_CACHE_سطر))

قسمتعریف متلون LONG atomic_int;
قسمتعریف atomic_int atomic_bool;
قسمتعریف atomic_int atomic_flag;

#تعریف_کرو ATOMIC_FLAG_INIT 0

قسمتعریف شمار {
    memory_order_relaxed,
    memory_order_consume,
    memory_order_acquire,
    memory_order_release,
    memory_order_acq_rel,
    memory_order_seq_cst
} memory_order;

ساکن خالی atomic_store(atomic_int * ptr, LONG val) {
    InterlockedExchange(ptr, val);
}
ساکن خالی atomic_store_explicit(atomic_int * ptr, LONG val, memory_order mo) {
    // TODO: add support for explicit memory order
    InterlockedExchange(ptr, val);
}
ساکن LONG atomic_بھرو(atomic_int * ptr) {
    واپس InterlockedCompareExchange(ptr, 0, 0);
}
ساکن LONG atomic_بھرو_explicit(atomic_int * ptr, memory_order mo) {
    // TODO: add support for explicit memory order
    واپس InterlockedCompareExchange(ptr, 0, 0);
}
ساکن LONG atomic_fetch_جوڑو(atomic_int * ptr, LONG inc) {
    واپس Interlocked_Exchange_جوڑو(ptr, inc);
}
ساکن LONG atomic_fetch_جوڑو_explicit(atomic_int * ptr, LONG inc, memory_order mo) {
    // TODO: add support for explicit memory order
    واپس Interlocked_Exchange_جوڑو(ptr, inc);
}
ساکن atomic_bool atomic_flag_test_and_سیٹ(atomic_flag * ptr) {
    واپس InterlockedExchange(ptr, 1);
}
ساکن خالی atomic_flag_clear(atomic_flag * ptr) {
    InterlockedExchange(ptr, 0);
}
ساکن خالی atomic_thread_fence(memory_order mo) {
    حافظہ_Barrier();
}
#ورنہ // clang
#شامل <stdatomic.h>
#اگر_ختم

قسمتعریف HANDLE pthread_t;

قسمتعریف DWORD thread_ret_t;
ساکن صحیح pthread_create(pthread_t * برآمد, خالی * غیر_مستعمل_2, thread_ret_t(*func)(خالی *), خالی * arg) {
    (خالی) غیر_مستعمل_2;
    HANDLE handle = CreateThread(تہی, 0, (LPTHREAD_آغاز_ROUTINE) func, arg, 0, تہی);
    اگر (handle == تہی)
    {
        واپس EAGAIN;
    }

    *برآمد = handle;
    واپس 0;
}

ساکن صحیح pthread_join(pthread_t thread, خالی * غیر_مستعمل_2) {
    (خالی) غیر_مستعمل_2;
    صحیح ret = (صحیح) WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    واپس ret;
}

ساکن صحیح sched_yield (خالی) {
    Sleep (0);
    واپس 0;
}
#ورنہ

#شامل <pthread.h>
#شامل <stdatomic.h>
#شامل <sched.h>
#اگر defined(__FreeBSD__)
#شامل <pthread_np.h>
#اگر_ختم

قسمتعریف خالی * thread_ret_t;

#شامل <sys/types.h>
#شامل <sys/stat.h>
#شامل <unistd.h>

#اگر_ختم

قسمتعریف pthread_t ggml_thread_t;

#تعریف_کرو GGML_THREADPOOL_N_دھاگے_نقاب (0xffffU)
#تعریف_کرو GGML_THREADPOOL_N_دھاگے_بٹ (16)

#اگر defined(__APPLE__)
#شامل <unistd.h>
#شامل <mach/mach.h>
#شامل <TargetConditionals.h>
#اگر_ختم

ساکن ثابت ساخت ggml_type_traits_cpu type_traits_cpu[GGML_TYPE_گنتی] = {
    [GGML_TYPE_F32] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_مقصد_fp32,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_f32,
        .vec_dot_type             = GGML_TYPE_F32,
        .nrows                    = 1,
    },
    [GGML_TYPE_F16] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_مقصد_fp16,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_f16,
        .vec_dot_type             = GGML_TYPE_F16,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q1_0] = {
        .from_float               = quantize_صف_q1_0,
        .vec_dot                  = ggml_vec_dot_q1_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q2_0] = {
        .from_float               = quantize_صف_q2_0,
        .vec_dot                  = ggml_vec_dot_q2_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q4_0] = {
        .from_float               = quantize_صف_q4_0,
        .vec_dot                  = ggml_vec_dot_q4_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
#اگر defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#ورنہ
        .nrows                    = 1,
#اگر_ختم
    },
    [GGML_TYPE_Q4_1] = {
        .from_float               = quantize_صف_q4_1,
        .vec_dot                  = ggml_vec_dot_q4_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
#اگر defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#ورنہ
        .nrows                    = 1,
#اگر_ختم
    },
    [GGML_TYPE_Q5_0] = {
        .from_float               = quantize_صف_q5_0,
        .vec_dot                  = ggml_vec_dot_q5_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q5_1] = {
        .from_float               = quantize_صف_q5_1,
        .vec_dot                  = ggml_vec_dot_q5_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q8_0] = {
        .from_float               = quantize_صف_q8_0,
        .vec_dot                  = ggml_vec_dot_q8_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
#اگر defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#ورنہ
        .nrows                    = 1,
#اگر_ختم
    },
    [GGML_TYPE_Q8_1] = {
        .from_float               = quantize_صف_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
        .nrows                    = 1,
    },
    [GGML_TYPE_MXFP4] = {
        .from_float               = quantize_صف_mxfp4,
        .vec_dot                  = ggml_vec_dot_mxfp4_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_NVFP4] = {
        .from_float               = quantize_صف_nvfp4,
        .vec_dot                  = ggml_vec_dot_nvfp4_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q2_K] = {
        .from_float               = quantize_صف_q2_K,
        .vec_dot                  = ggml_vec_dot_q2_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q3_K] = {
        .from_float               = quantize_صف_q3_K,
        .vec_dot                  = ggml_vec_dot_q3_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q4_K] = {
        .from_float               = quantize_صف_q4_K,
        .vec_dot                  = ggml_vec_dot_q4_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
#اگر defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#ورنہ
        .nrows                    = 1,
#اگر_ختم
    },
    [GGML_TYPE_Q5_K] = {
        .from_float               = quantize_صف_q5_K,
        .vec_dot                  = ggml_vec_dot_q5_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q6_K] = {
        .from_float               = quantize_صف_q6_K,
        .vec_dot                  = ggml_vec_dot_q6_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
#اگر defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#ورنہ
        .nrows                    = 1,
#اگر_ختم
    },
    [GGML_TYPE_IQ2_XXS] = {
        .from_float               = تہی,
        .vec_dot                  = ggml_vec_dot_iq2_xxs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_XS] = {
        .from_float               = تہی,
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
        .from_float               = تہی,
        .vec_dot                  = ggml_vec_dot_iq1_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_M] = {
        .from_float               = تہی,
        .vec_dot                  = ggml_vec_dot_iq1_m_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_NL] = {
        .from_float               = quantize_صف_iq4_nl,
        .vec_dot                  = ggml_vec_dot_iq4_nl_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_XS] = {
        .from_float               = quantize_صف_iq4_xs,
        .vec_dot                  = ggml_vec_dot_iq4_xs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q8_K] = {
        .from_float               = quantize_صف_q8_K,
    },
    [GGML_TYPE_BF16] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_مقصد_bf16,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_bf16,
        .vec_dot_type             = GGML_TYPE_BF16,
        .nrows                    = 1,
    },
    [GGML_TYPE_TQ1_0] = {
        .from_float               = quantize_صف_tq1_0,
        .vec_dot                  = ggml_vec_dot_tq1_0_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_TQ2_0] = {
        .from_float               = quantize_صف_tq2_0,
        .vec_dot                  = ggml_vec_dot_tq2_0_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_I32] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_مقصد_i32,
    },
};

ثابت ساخت ggml_type_traits_cpu * ggml_get_type_traits_cpu(شمار ggml_type type) {
    واپس &type_traits_cpu[type];
}

//
// Threading defs
//

قسمتعریف pthread_t          ggml_thread_t;

#اگر defined(_WIN32)

قسمتعریف CONDITION_VARIABLE ggml_cond_t;
قسمتعریف SRWLOCK            ggml_mutex_t;

#تعریف_کرو ggml_mutex_init(m)   InitializeSRWLock(m)
#تعریف_کرو ggml_mutex_destroy(m)
#تعریف_کرو ggml_mutex_تالا(m)   AcquireSRWLockExclusive(m)
#تعریف_کرو ggml_mutex_unlock(m) ReleaseSRWLockExclusive(m)
#تعریف_کرو ggml_mutex_تالا_shared(m)   AcquireSRWLockShared(m)
#تعریف_کرو ggml_mutex_unlock_shared(m) ReleaseSRWLockShared(m)

#تعریف_کرو ggml_cond_init(c)    InitializeConditionVariable(c)
#تعریف_کرو ggml_cond_destroy(c)
#تعریف_کرو ggml_cond_wait(c, m) SleepConditionVariableSRW(c, m, INFINITE, CONDITION_VARIABLE_LOCKMODE_SHARED)
#تعریف_کرو ggml_cond_نشر(c) WakeAllConditionVariable(c)

#تعریف_کرو ggml_thread_create pthread_create
#تعریف_کرو ggml_thread_join   pthread_join

#ورنہ

قسمتعریف pthread_cond_t     ggml_cond_t;
قسمتعریف pthread_mutex_t    ggml_mutex_t;

#تعریف_کرو ggml_mutex_init(m)          pthread_mutex_init(m, تہی)
#تعریف_کرو ggml_mutex_destroy(m)       pthread_mutex_destroy(m)
#تعریف_کرو ggml_mutex_تالا(m)          pthread_mutex_تالا(m)
#تعریف_کرو ggml_mutex_unlock(m)        pthread_mutex_unlock(m)
#تعریف_کرو ggml_mutex_تالا_shared(m)   pthread_mutex_تالا(m)
#تعریف_کرو ggml_mutex_unlock_shared(m) pthread_mutex_unlock(m)

#تعریف_کرو ggml_تالا_init(x)    غیر_مستعمل(x)
#تعریف_کرو ggml_تالا_destroy(x) غیر_مستعمل(x)
#اگر defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64) && !defined(_M_ARM64EC))
#تعریف_کرو ggml_تالا_تالا(x)    _mm_pause()
#ورنہ
#تعریف_کرو ggml_تالا_تالا(x)    غیر_مستعمل(x)
#اگر_ختم
#تعریف_کرو ggml_تالا_unlock(x)  غیر_مستعمل(x)

#تعریف_کرو GGML_تالا_INITIALIZER 0
#تعریف_کرو ggml_cond_init(c)      pthread_cond_init(c, تہی)
#تعریف_کرو ggml_cond_destroy(c)   pthread_cond_destroy(c)
#تعریف_کرو ggml_cond_wait(c, m)   pthread_cond_wait(c, m)
#تعریف_کرو ggml_cond_نشر(c) pthread_cond_نشر(c)

#تعریف_کرو ggml_thread_create pthread_create
#تعریف_کرو ggml_thread_join   pthread_join

#اگر_ختم

// Threadpool def
ساخت ggml_threadpool {
    ggml_mutex_t mutex;       // mutex for cond.var
    ggml_cond_t  cond;        // cond.var for waiting for new work

    ساخت ggml_cgraph * cgraph;
    ساخت ggml_cplan  * cplan;

    // synchronization primitives
    atomic_int n_گراف;       // updated when there is work to be done (i.e each graph) holds graph and active thread counts.
    atomic_int GGML_CACHE_ALIGN n_barrier;
    atomic_int GGML_CACHE_ALIGN n_barrier_passed;
    atomic_int GGML_CACHE_ALIGN current_chunk; // currently processing chunk during Mat_Mul, shared between all the threads.

    // these are atomic as an annotation for thread-sanitizer
    atomic_bool stop;         // Used for stopping the threadpool altogether
    atomic_bool pause;        // Used for pausing the threadpool or individual threads
    atomic_int  abort;        // Used for aborting processing of a graph

    ساخت ggml_compute_state * workers;   // per thread state
    صحیح          n_دھاگے;   // Number of threads in the pool
    int32_t      prio;        // Scheduling priority
    uint32_t     poll;        // Polling level (0 - no polling)

    شمار ggml_حالت ec;
};

// Per-thread state
ساخت ggml_compute_state {
#اگر_غیر_تعریف_شدہ GGML_USE_OPENMP
    ggml_thread_t thrd;
    صحیح  آخری_گراف;
    bool pending;
#اگر_ختم
    bool cpumask[GGML_MAX_N_دھاگے];
    ساخت ggml_threadpool * threadpool;
    صحیح ith;
};

// Helpers for polling loops
#اگر defined(__aarch64__) && ( defined(__clang__) || defined(__GNUC__) )
ساکن اندرونی خالی ggml_thread_cpu_relax(خالی) {
    __asm__ متلون("yield" ::: "memory");
}
#وگرنہ_اگر defined(__x86_64__)
ساکن اندرونی خالی ggml_thread_cpu_relax(خالی) {
    _mm_pause();
}
#وگرنہ_اگر defined(__riscv)
ساکن اندرونی خالی ggml_thread_cpu_relax(خالی) {
    #اگر_تعریف_شدہ __riscv_zihintpause
        __asm__ __volatile__ ("pause");
    #ورنہ
        /* Encoding of the pause instruction */
        __asm__ __volatile__ (".4byte 0x100000F");
    #اگر_ختم
}
#ورنہ
ساکن اندرونی خالی ggml_thread_cpu_relax(خالی) {;}
#اگر_ختم

//
// NUMA support
//

#تعریف_کرو GGML_NUMA_MAX_NODES 8
#تعریف_کرو GGML_NUMA_MAX_CPUS 512

ساخت ggml_numa_گرہ {
    uint32_t cpus[GGML_NUMA_MAX_CPUS]; // hardware threads on this node
    uint32_t n_cpus;
};

ساخت ggml_numa_nodes {
    شمار ggml_numa_strategy numa_strategy;
    ساخت ggml_numa_گرہ nodes[GGML_NUMA_MAX_NODES];
    uint32_t n_nodes;
    uint32_t کل_cpus; // hardware threads on system
    uint32_t current_گرہ; // node on which main process is execting
#اگر defined(__gnu_linux__)
    cpu_سیٹ_t cpuset; // cpuset from numactl
#ورنہ
    uint32_t cpuset; // no NUMA support outside of Linux at this time. Use a portable datatype
#اگر_ختم
};

//
// ggml state
//

ساخت ggml_state {
    ساخت ggml_numa_nodes numa;
};

ساکن ساخت ggml_state g_state = {0};

خالی ggml_barrier(ساخت ggml_threadpool * tp) {
    صحیح n_دھاگے = atomic_بھرو_explicit(&tp->n_گراف, memory_order_relaxed) & GGML_THREADPOOL_N_دھاگے_نقاب;
    اگر (n_دھاگے == 1) {
        واپس;
    }

#اگر_تعریف_شدہ GGML_USE_OPENMP
    #ہدایت omp barrier
#ورنہ
    صحیح n_passed = atomic_بھرو_explicit(&tp->n_barrier_passed, memory_order_relaxed);

    // enter barrier (full seq-cst fence)
    صحیح n_barrier = atomic_fetch_جوڑو_explicit(&tp->n_barrier, 1, memory_order_seq_cst);

    اگر (n_barrier == (n_دھاگے - 1)) {
        // last thread
        atomic_store_explicit(&tp->n_barrier, 0, memory_order_relaxed);

        // exit barrier (full seq-cst fence)
        atomic_fetch_جوڑو_explicit(&tp->n_barrier_passed, 1, memory_order_seq_cst);
        واپس;
    }

    // wait for other threads
    جب (atomic_بھرو_explicit(&tp->n_barrier_passed, memory_order_relaxed) == n_passed) {
        ggml_thread_cpu_relax();
    }

    // exit barrier (full seq-cst fence)
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #اگر_تعریف_شدہ GGML_TSAN_ENABLED
    atomic_fetch_جوڑو_explicit(&tp->n_barrier_passed, 0, memory_order_seq_cst);
    #ورنہ
    atomic_thread_fence(memory_order_seq_cst);
    #اگر_ختم
#اگر_ختم
}

خالی ggml_threadpool_chunk_سیٹ(ساخت ggml_threadpool * tp, صحیح قدر) {
    atomic_store_explicit(&tp->current_chunk, قدر, memory_order_relaxed);
}

صحیح ggml_threadpool_chunk_جوڑو(ساخت ggml_threadpool * tp, صحیح قدر) {
    واپس atomic_fetch_جوڑو_explicit(&tp->current_chunk, قدر, memory_order_relaxed);
}

#اگر defined(__gnu_linux__)
ساکن cpu_سیٹ_t ggml_get_numa_affinity(خالی) {
    cpu_سیٹ_t cpuset;
    pthread_t thread;
    thread = pthread_self();
    CPU_ZERO(&cpuset);
    pthread_getaffinity_np(thread, حجم(cpu_سیٹ_t), &cpuset);
    واپس cpuset;
}
#ورنہ
ساکن uint32_t ggml_get_numa_affinity(خالی) {
    واپس 0; // no NUMA support
}
#اگر_ختم

خالی ggml_numa_init(شمار ggml_numa_strategy numa_flag) {
    اگر (g_state.numa.n_nodes > 0) {
        fprintf(stderr, "ggml_numa_init: NUMA already initialized\n");

        واپس;
    }

#اگر defined(__gnu_linux__)
    ساخت stat st;
    حرف راستہ[256];
    صحیح rv;

    // set numa scheme
    g_state.numa.numa_strategy = numa_flag;

    GGML_PRINT_DEBUG("numa strategy %u\n",g_state.numa.numa_strategy);

    g_state.numa.cpuset = ggml_get_numa_affinity();

    // enumerate nodes
    جب (g_state.numa.n_nodes < GGML_NUMA_MAX_NODES) {
        rv = snprintf(راستہ, حجم(راستہ), "/sys/devices/system/node/node%u", g_state.numa.n_nodes);
        GGML_ASSERT(rv > 0 && (غیرعلامتی)rv < حجم(راستہ));
        اگر (stat(راستہ, &st) != 0) { توڑو; }
        ++g_state.numa.n_nodes;
    }

    // enumerate CPUs
    جب (g_state.numa.کل_cpus < GGML_NUMA_MAX_CPUS) {
        rv = snprintf(راستہ, حجم(راستہ), "/sys/devices/system/cpu/cpu%u", g_state.numa.کل_cpus);
        GGML_ASSERT(rv > 0 && (غیرعلامتی)rv < حجم(راستہ));
        اگر (stat(راستہ, &st) != 0) { توڑو; }
        ++g_state.numa.کل_cpus;
    }

    GGML_PRINT_DEBUG("found %u numa nodes, %u CPUs\n", g_state.numa.n_nodes, g_state.numa.کل_cpus);

    // figure out which node we're on
    uint current_cpu;
    صحیح getcpu_ret = 0;
#اگر __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ > 33) || defined(__COSMOPOLITAN__)
    getcpu_ret = getcpu(&current_cpu, &g_state.numa.current_گرہ);
#ورنہ
    // old glibc doesn't have a wrapper for this call. Fall back on direct syscall
#   اگر !defined(SYS_getcpu) && defined(SYS_get_cpu)
#       define SYS_getcpu SYS_get_cpu // some older glibc versions use this name
#   endif
    getcpu_ret = syscall(SYS_getcpu, &current_cpu, &g_state.numa.current_گرہ);
#اگر_ختم

    اگر (g_state.numa.n_nodes < 1 || g_state.numa.کل_cpus < 1 || getcpu_ret != 0) {
        g_state.numa.n_nodes = 0;
        واپس;
    }

    GGML_PRINT_DEBUG("found our process on numa node %u, CPU %u\n", g_state.numa.current_گرہ, current_cpu);

    برائے (uint32_t n = 0; n < g_state.numa.n_nodes; ++n) {
        ساخت ggml_numa_گرہ * گرہ = &g_state.numa.nodes[n];
        GGML_PRINT_DEBUG("CPUs on node %u:", n);
        گرہ->n_cpus = 0;
        برائے (uint32_t c = 0; c < g_state.numa.کل_cpus; ++c) {
            rv = snprintf(راستہ, حجم(راستہ), "/sys/devices/system/node/node%u/cpu%u", n, c);
            GGML_ASSERT(rv > 0 && (غیرعلامتی)rv < حجم(راستہ));
            اگر (stat(راستہ, &st) == 0) {
                گرہ->cpus[گرہ->n_cpus++] = c;
                GGML_PRINT_DEBUG(" %u", c);
            }
        }
        GGML_PRINT_DEBUG("\n");
    }

    اگر (ggml_is_numa()) {
        FILE *fptr = fopen("/proc/sys/kernel/numa_balancing", "r");
        اگر (fptr != تہی) {
            حرف buf[42];
            اگر (fgets(buf, حجم(buf), fptr) && لڑی_موازنہ(buf, "0\n", حجم(buf)) != 0) {
                GGML_LOG_WARN("/proc/sys/kernel/numa_balancing is enabled, this has been observed to impair performance\n");
            }
            fclose(fptr);
        }
    }
#ورنہ
    غیر_مستعمل(numa_flag);
    // TODO
#اگر_ختم
}

bool ggml_is_numa(خالی) {
    واپس g_state.numa.n_nodes > 1;
}

#اگر defined(__ARM_ARCH)
#اگر defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
#شامل <arm_sve.h>
ساکن خالی ggml_init_arm_arch_features(خالی) {
    ggml_arm_arch_features.sve_cnt = svcntb();
}
#ورنہ
ساکن خالی ggml_init_arm_arch_features(خالی) {}
#اگر_ختم
#اگر_ختم // __ARM_ARCH

#اگر defined(__riscv) && defined(__riscv_v_intrinsic)
#شامل <riscv_vector.h>
ساکن خالی ggml_init_riscv_arch_features(خالی) {
    ggml_riscv_arch_features.rvv_vlen = __riscv_vlenb();
}
#ورنہ
ساکن خالی ggml_init_riscv_arch_features(خالی) {}
#اگر_ختم

ساخت ggml_tensor * ggml_new_i32(ساخت ggml_context * ctx, int32_t قدر) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    ساخت ggml_tensor * نتیجہ = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

    ggml_سیٹ_i32(نتیجہ, قدر);

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_new_f32(ساخت ggml_context * ctx, اعشاری قدر) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    ساخت ggml_tensor * نتیجہ = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    ggml_سیٹ_f32(نتیجہ, قدر);

    واپس نتیجہ;
}

ساخت ggml_tensor * ggml_سیٹ_i32 (ساخت ggml_tensor * tensor, int32_t قدر) {
    ثابت صحیح n     = ggml_nrows(tensor);
    ثابت صحیح nc    = tensor->ne[0];
    ثابت size_t n1 = tensor->nb[1];

    حرف * ثابت ڈیٹا = tensor->ڈیٹا;

    انتخاب (tensor->type) {
        حالت GGML_TYPE_I8:
            {
                أكد(tensor->nb[0] == حجم(int8_t));
                برائے (صحیح i = 0; i < n; i++) {
                    ggml_vec_سیٹ_i8(nc, (int8_t *)(ڈیٹا + i*n1), قدر);
                }
            } توڑو;
        حالت GGML_TYPE_I16:
            {
                أكد(tensor->nb[0] == حجم(int16_t));
                برائے (صحیح i = 0; i < n; i++) {
                    ggml_vec_سیٹ_i16(nc, (int16_t *)(ڈیٹا + i*n1), قدر);
                }
            } توڑو;
        حالت GGML_TYPE_I32:
            {
                أكد(tensor->nb[0] == حجم(int32_t));
                برائے (صحیح i = 0; i < n; i++) {
                    ggml_vec_سیٹ_i32(nc, (int32_t *)(ڈیٹا + i*n1), قدر);
                }
            } توڑو;
        حالت GGML_TYPE_F16:
            {
                أكد(tensor->nb[0] == حجم(ggml_fp16_t));
                برائے (صحیح i = 0; i < n; i++) {
                    ggml_vec_سیٹ_f16(nc, (ggml_fp16_t *)(ڈیٹا + i*n1), GGML_CPU_FP32_مقصد_FP16(قدر));
                }
            } توڑو;
        حالت GGML_TYPE_BF16:
            {
                أكد(tensor->nb[0] == حجم(ggml_fp16_t));
                برائے (صحیح i = 0; i < n; i++) {
                    ggml_vec_سیٹ_bf16(nc, (ggml_bf16_t *)(ڈیٹا + i*n1), GGML_FP32_مقصد_BF16(قدر));
                }
            } توڑو;
        حالت GGML_TYPE_F32:
            {
                أكد(tensor->nb[0] == حجم(اعشاری));
                برائے (صحیح i = 0; i < n; i++) {
                    ggml_vec_سیٹ_f32(nc, (اعشاری *)(ڈیٹا + i*n1), قدر);
                }
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }

    واپس tensor;
}

ساخت ggml_tensor * ggml_سیٹ_f32(ساخت ggml_tensor * tensor, اعشاری قدر) {
    ثابت صحیح n     = ggml_nrows(tensor);
    ثابت صحیح nc    = tensor->ne[0];
    ثابت size_t n1 = tensor->nb[1];

    حرف * ثابت ڈیٹا = tensor->ڈیٹا;

    انتخاب (tensor->type) {
        حالت GGML_TYPE_I8:
            {
                أكد(tensor->nb[0] == حجم(int8_t));
                برائے (صحیح i = 0; i < n; i++) {
                    ggml_vec_سیٹ_i8(nc, (int8_t *)(ڈیٹا + i*n1), قدر);
                }
            } توڑو;
        حالت GGML_TYPE_I16:
            {
                أكد(tensor->nb[0] == حجم(int16_t));
                برائے (صحیح i = 0; i < n; i++) {
                    ggml_vec_سیٹ_i16(nc, (int16_t *)(ڈیٹا + i*n1), قدر);
                }
            } توڑو;
        حالت GGML_TYPE_I32:
            {
                أكد(tensor->nb[0] == حجم(int32_t));
                برائے (صحیح i = 0; i < n; i++) {
                    ggml_vec_سیٹ_i32(nc, (int32_t *)(ڈیٹا + i*n1), قدر);
                }
            } توڑو;
        حالت GGML_TYPE_F16:
            {
                أكد(tensor->nb[0] == حجم(ggml_fp16_t));
                برائے (صحیح i = 0; i < n; i++) {
                    ggml_vec_سیٹ_f16(nc, (ggml_fp16_t *)(ڈیٹا + i*n1), GGML_CPU_FP32_مقصد_FP16(قدر));
                }
            } توڑو;
        حالت GGML_TYPE_BF16:
            {
                أكد(tensor->nb[0] == حجم(ggml_bf16_t));
                برائے (صحیح i = 0; i < n; i++) {
                    ggml_vec_سیٹ_bf16(nc, (ggml_bf16_t *)(ڈیٹا + i*n1), GGML_FP32_مقصد_BF16(قدر));
                }
            } توڑو;
        حالت GGML_TYPE_F32:
            {
                أكد(tensor->nb[0] == حجم(اعشاری));
                برائے (صحیح i = 0; i < n; i++) {
                    ggml_vec_سیٹ_f32(nc, (اعشاری *)(ڈیٹا + i*n1), قدر);
                }
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }

    واپس tensor;
}

int32_t ggml_get_i32_1d(ثابت ساخت ggml_tensor * tensor, صحیح i) {
    اگر (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_اشاریہ(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        واپس ggml_get_i32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    انتخاب (tensor->type) {
        حالت GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == حجم(int8_t));
                واپس ((int8_t *)(tensor->ڈیٹا))[i];
            }
        حالت GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == حجم(int16_t));
                واپس ((int16_t *)(tensor->ڈیٹا))[i];
            }
        حالت GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == حجم(int32_t));
                واپس ((int32_t *)(tensor->ڈیٹا))[i];
            }
        حالت GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == حجم(ggml_fp16_t));
                واپس GGML_CPU_FP16_مقصد_FP32(((ggml_fp16_t *)(tensor->ڈیٹا))[i]);
            }
        حالت GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == حجم(ggml_bf16_t));
                واپس GGML_BF16_مقصد_FP32(((ggml_bf16_t *)(tensor->ڈیٹا))[i]);
            }
        حالت GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == حجم(اعشاری));
                واپس ((اعشاری *)(tensor->ڈیٹا))[i];
            }
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

خالی ggml_سیٹ_i32_1d(ثابت ساخت ggml_tensor * tensor, صحیح i, int32_t قدر) {
    اگر (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_اشاریہ(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_سیٹ_i32_nd(tensor, id[0], id[1], id[2], id[3], قدر);
        واپس;
    }
    انتخاب (tensor->type) {
        حالت GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == حجم(int8_t));
                ((int8_t *)(tensor->ڈیٹا))[i] = قدر;
            } توڑو;
        حالت GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == حجم(int16_t));
                ((int16_t *)(tensor->ڈیٹا))[i] = قدر;
            } توڑو;
        حالت GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == حجم(int32_t));
                ((int32_t *)(tensor->ڈیٹا))[i] = قدر;
            } توڑو;
        حالت GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == حجم(ggml_fp16_t));
                ((ggml_fp16_t *)(tensor->ڈیٹا))[i] = GGML_CPU_FP32_مقصد_FP16(قدر);
            } توڑو;
        حالت GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == حجم(ggml_bf16_t));
                ((ggml_bf16_t *)(tensor->ڈیٹا))[i] = GGML_FP32_مقصد_BF16(قدر);
            } توڑو;
        حالت GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == حجم(اعشاری));
                ((اعشاری *)(tensor->ڈیٹا))[i] = قدر;
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

int32_t ggml_get_i32_nd(ثابت ساخت ggml_tensor * tensor, صحیح i0, صحیح i1, صحیح i2, صحیح i3) {
    خالی * ڈیٹا   = (حرف *) tensor->ڈیٹا + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    انتخاب (tensor->type) {
        حالت GGML_TYPE_I8:
            واپس ((int8_t *) ڈیٹا)[0];
        حالت GGML_TYPE_I16:
            واپس ((int16_t *) ڈیٹا)[0];
        حالت GGML_TYPE_I32:
            واپس ((int32_t *) ڈیٹا)[0];
        حالت GGML_TYPE_F16:
            واپس GGML_CPU_FP16_مقصد_FP32(((ggml_fp16_t *) ڈیٹا)[0]);
        حالت GGML_TYPE_BF16:
            واپس GGML_BF16_مقصد_FP32(((ggml_bf16_t *) ڈیٹا)[0]);
        حالت GGML_TYPE_F32:
            واپس ((اعشاری *) ڈیٹا)[0];
        طےشدہ:
            GGML_ABORT("fatal error");
    }
}

خالی ggml_سیٹ_i32_nd(ثابت ساخت ggml_tensor * tensor, صحیح i0, صحیح i1, صحیح i2, صحیح i3, int32_t قدر) {
    خالی * ڈیٹا   = (حرف *) tensor->ڈیٹا + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    انتخاب (tensor->type) {
        حالت GGML_TYPE_I8:
            {
                ((int8_t *)(ڈیٹا))[0] = قدر;
            } توڑو;
        حالت GGML_TYPE_I16:
            {
                ((int16_t *)(ڈیٹا))[0] = قدر;
            } توڑو;
        حالت GGML_TYPE_I32:
            {
                ((int32_t *)(ڈیٹا))[0] = قدر;
            } توڑو;
        حالت GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(ڈیٹا))[0] = GGML_CPU_FP32_مقصد_FP16(قدر);
            } توڑو;
        حالت GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(ڈیٹا))[0] = GGML_FP32_مقصد_BF16(قدر);
            } توڑو;
        حالت GGML_TYPE_F32:
            {
                ((اعشاری *)(ڈیٹا))[0] = قدر;
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

اعشاری ggml_get_f32_1d(ثابت ساخت ggml_tensor * tensor, صحیح i) {
    اگر (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_اشاریہ(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        واپس ggml_get_f32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    انتخاب (tensor->type) {
        حالت GGML_TYPE_I8:
            {
                واپس ((int8_t *)(tensor->ڈیٹا))[i];
            }
        حالت GGML_TYPE_I16:
            {
                واپس ((int16_t *)(tensor->ڈیٹا))[i];
            }
        حالت GGML_TYPE_I32:
            {
                واپس ((int32_t *)(tensor->ڈیٹا))[i];
            }
        حالت GGML_TYPE_F16:
            {
                واپس GGML_CPU_FP16_مقصد_FP32(((ggml_fp16_t *)(tensor->ڈیٹا))[i]);
            }
        حالت GGML_TYPE_BF16:
            {
                واپس GGML_BF16_مقصد_FP32(((ggml_bf16_t *)(tensor->ڈیٹا))[i]);
            }
        حالت GGML_TYPE_F32:
            {
                واپس ((اعشاری *)(tensor->ڈیٹا))[i];
            }
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

خالی ggml_سیٹ_f32_1d(ثابت ساخت ggml_tensor * tensor, صحیح i, اعشاری قدر) {
    اگر (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_اشاریہ(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_سیٹ_f32_nd(tensor, id[0], id[1], id[2], id[3], قدر);
        واپس;
    }
    انتخاب (tensor->type) {
        حالت GGML_TYPE_I8:
            {
                ((int8_t *)(tensor->ڈیٹا))[i] = قدر;
            } توڑو;
        حالت GGML_TYPE_I16:
            {
                ((int16_t *)(tensor->ڈیٹا))[i] = قدر;
            } توڑو;
        حالت GGML_TYPE_I32:
            {
                ((int32_t *)(tensor->ڈیٹا))[i] = قدر;
            } توڑو;
        حالت GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(tensor->ڈیٹا))[i] = GGML_CPU_FP32_مقصد_FP16(قدر);
            } توڑو;
        حالت GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(tensor->ڈیٹا))[i] = GGML_FP32_مقصد_BF16(قدر);
            } توڑو;
        حالت GGML_TYPE_F32:
            {
                ((اعشاری *)(tensor->ڈیٹا))[i] = قدر;
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

اعشاری ggml_get_f32_nd(ثابت ساخت ggml_tensor * tensor, صحیح i0, صحیح i1, صحیح i2, صحیح i3) {
    خالی * ڈیٹا   = (حرف *) tensor->ڈیٹا + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    انتخاب (tensor->type) {
        حالت GGML_TYPE_I8:
            واپس ((int8_t *) ڈیٹا)[0];
        حالت GGML_TYPE_I16:
            واپس ((int16_t *) ڈیٹا)[0];
        حالت GGML_TYPE_I32:
            واپس ((int32_t *) ڈیٹا)[0];
        حالت GGML_TYPE_F16:
            واپس GGML_CPU_FP16_مقصد_FP32(((ggml_fp16_t *) ڈیٹا)[0]);
        حالت GGML_TYPE_BF16:
            واپس GGML_BF16_مقصد_FP32(((ggml_bf16_t *) ڈیٹا)[0]);
        حالت GGML_TYPE_F32:
            واپس ((اعشاری *) ڈیٹا)[0];
        طےشدہ:
            GGML_ABORT("fatal error");
    }
}

خالی ggml_سیٹ_f32_nd(ثابت ساخت ggml_tensor * tensor, صحیح i0, صحیح i1, صحیح i2, صحیح i3, اعشاری قدر) {
    خالی * ڈیٹا   = (حرف *) tensor->ڈیٹا + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    انتخاب (tensor->type) {
        حالت GGML_TYPE_I8:
            {
                ((int8_t *)(ڈیٹا))[0] = قدر;
            } توڑو;
        حالت GGML_TYPE_I16:
            {
                ((int16_t *)(ڈیٹا))[0] = قدر;
            } توڑو;
        حالت GGML_TYPE_I32:
            {
                ((int32_t *)(ڈیٹا))[0] = قدر;
            } توڑو;
        حالت GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(ڈیٹا))[0] = GGML_CPU_FP32_مقصد_FP16(قدر);
            } توڑو;
        حالت GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(ڈیٹا))[0] = GGML_FP32_مقصد_BF16(قدر);
            } توڑو;
        حالت GGML_TYPE_F32:
            {
                ((اعشاری *)(ڈیٹا))[0] = قدر;
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

////////////////////////////////////////////////////////////////////////////////

// ggml_compute_forward_mul_mat

ساکن خالی ggml_compute_forward_mul_mat_one_chunk(
    ثابت ساخت ggml_compute_params * params,
    ساخت ggml_tensor * dst,
    ثابت شمار ggml_type type,
    ثابت int64_t عدد_صفیں_فی_vec_dot,
    ثابت int64_t ir0_آغاز,
    ثابت int64_t ir0_انتہا,
    ثابت int64_t ir1_آغاز,
    ثابت int64_t ir1_انتہا) {

    ثابت ساخت ggml_tensor * src0 = dst->src[0];
    ثابت ساخت ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ثنائی_OP_LOCALS

    ثابت bool src1_cont = ggml_is_contiguous(src1);

    ggml_vec_dot_t ثابت vec_dot      = type_traits_cpu[type].vec_dot;
    شمار ggml_type ثابت vec_dot_type = type_traits_cpu[type].vec_dot_type;

    // broadcast factors
    ثابت int64_t r2 = ne12 / ne02;
    ثابت int64_t r3 = ne13 / ne03;

    //printf("ir0_start = %6lld, ir0_end = %6lld, ir1_start = %6lld, ir1_end = %6lld\n", ir0_start, ir0_end, ir1_start, ir1_end);

    // threads with no work simply yield (not sure if it helps)
    اگر (ir0_آغاز >= ir0_انتہا || ir1_آغاز >= ir1_انتہا) {
        واپس;
    }

    ثابت خالی * wdata = (src1->type == vec_dot_type) ? src1->ڈیٹا : params->wdata;
    ثابت size_t صف_سائز = ggml_صف_سائز(vec_dot_type, ne10);

    أكد(ne12 % ne02 == 0);
    أكد(ne13 % ne03 == 0);

    // block-tiling attempt
    ثابت int64_t blck_0 = 16;
    ثابت int64_t blck_1 = 16;

    ثابت size_t src1_ستون_stride = src1_cont || src1->type != vec_dot_type ? صف_سائز : nb11;

    // attempt to reduce false-sharing (does not seem to make a difference)
    // 16 * 2, accounting for mmla kernels
    اعشاری tmp[32];

    برائے (int64_t iir1 = ir1_آغاز; iir1 < ir1_انتہا; iir1 += blck_1) {
        برائے (int64_t iir0 = ir0_آغاز; iir0 < ir0_انتہا; iir0 += blck_0) {
            برائے (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_انتہا; ir1 += عدد_صفیں_فی_vec_dot) {
                ثابت int64_t i13 = (ir1 / (ne12 * ne1));
                ثابت int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                ثابت int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                // broadcast src0 into src1
                ثابت int64_t i03 = i13 / r3;
                ثابت int64_t i02 = i12 / r2;

                ثابت int64_t i1 = i11;
                ثابت int64_t i2 = i12;
                ثابت int64_t i3 = i13;

                ثابت حرف * src0_صف = (ثابت حرف*)src0->ڈیٹا + (0 + i02 * nb02 + i03 * nb03);

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                ثابت حرف * src1_ستون = (ثابت حرف*)wdata +
                    (src1_cont || src1->type != vec_dot_type
                        ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * صف_سائز
                        : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                اعشاری * dst_ستون = (اعشاری*)((حرف*)dst->ڈیٹا + (i1 * nb1 + i2 * nb2 + i3 * nb3));

                //for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                //    vec_dot(ne00, &dst_col[ir0], src0_row + ir0*nb01, src1_col);
                //}

                برائے (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_انتہا; ir0 += عدد_صفیں_فی_vec_dot) {
                    vec_dot(ne00, &tmp[ir0 - iir0], (عدد_صفیں_فی_vec_dot > 1 ? 16 : 0), src0_صف + ir0 * nb01, (عدد_صفیں_فی_vec_dot > 1 ? nb01 : 0), src1_ستون, (عدد_صفیں_فی_vec_dot > 1 ? src1_ستون_stride : 0), عدد_صفیں_فی_vec_dot);
                }

                برائے (صحیح cn = 0; cn < عدد_صفیں_فی_vec_dot; ++cn) {
                    حافظہ_نقل(&dst_ستون[iir0 + cn * nb1 / nb0], tmp + (cn * 16), (MIN(iir0 + blck_0, ir0_انتہا) - iir0) * حجم(اعشاری));
                }
            }
        }
    }
}

خالی ggml_compute_forward_mul_mat(
        ثابت ساخت ggml_compute_params * params,
              ساخت ggml_tensor * dst) {

    ثابت ساخت ggml_tensor * src0 = dst->src[0];
    ثابت ساخت ggml_tensor * src1 = dst->src[1];

    ثابت int32_t hint = ggml_get_op_params_i32(dst, 1);
    اگر (hint == GGML_HINT_SRC0_IS_HADAMARD && !params->use_ref) {
        ggml_compute_forward_fwht(params, dst);
        واپس;
    }

    // If tiled is supported, it will execute the full op here and we return
    اگر (ggml_compute_forward_mul_mat_tiled(params, dst)) {
        واپس;
    }

    GGML_TENSOR_ثنائی_OP_LOCALS

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    شمار ggml_type           ثابت vec_dot_type         = type_traits_cpu[src0->type].vec_dot_type;
    ggml_from_float_t        ثابت from_float           = type_traits_cpu[vec_dot_type].from_float;
    int64_t                  ثابت vec_dot_عدد_صفیں     = type_traits_cpu[src0->type].nrows;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_سائز(src0->type));
    GGML_ASSERT(nb10 == ggml_type_سائز(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == حجم(اعشاری));
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
        برائے (int64_t i13 = 0; i13 < ne13; i13++)
            برائے (int64_t i12 = 0; i12 < ne12; i12++)
                اگر (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_سائز(src0->type),
                                     (ثابت حرف *)src0->ڈیٹا + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_سائز(src0->type),
                                     (ثابت حرف *)src1->ڈیٹا + i12*nb12 + i13*nb13,
                                     nb11/ggml_type_سائز(src1->type),
                                     (حرف *)dst->ڈیٹا + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_سائز(dst->type),
                                     src0->type,
                                     src1->type,
                                     dst->type))
                    جاؤ UseGgmlGemm1;
        واپس;
    }
UseGgmlGemm1:;
#اگر_ختم

    اگر (src1->type != vec_dot_type) {
        حرف * wdata = params->wdata;

        ثابت size_t nbw0 = ggml_type_سائز(vec_dot_type);
        ثابت size_t nbw1 = ggml_صف_سائز(vec_dot_type, ne10);
        ثابت size_t nbw2 = nbw1*ne11;
        ثابت size_t nbw3 = nbw2*ne12;

        أكد(params->wsize >= ne13*nbw3);
        // src1 is either packed from F32 into vec_dot_type, or widened from F16 or BF16 into the F32 work buffer
        ثابت bool widen = src1->type != GGML_TYPE_F32;

        GGML_ASSERT(!widen || vec_dot_type == GGML_TYPE_F32);
        GGML_ASSERT(!widen || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_BF16);

    #اگر 0
        برائے (int64_t i13 = 0; i13 < ne13; ++i13) {
            برائے (int64_t i12 = 0; i12 < ne12; ++i12) {
                برائے (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                    from_float((اعشاری *)((حرف *) src1->ڈیٹا + i13*nb13 + i12*nb12 + i11*nb11),
                               (خالی *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                                ne10);
                }
            }
        }
    #ورنہ
        ثابت int64_t bs = ggml_blck_سائز(vec_dot_type);

        برائے (int64_t i13 = 0; i13 < ne13; ++i13) {
            برائے (int64_t i12 = 0; i12 < ne12; ++i12) {
                برائے (int64_t i11 = 0; i11 < ne11; ++i11) {
                    int64_t ne10_بلاک_آغاز = (ith * ne10/bs) / nth;
                    int64_t ne10_بلاک_انتہا   = ((ith + 1) * ne10/bs) / nth;
                    ثابت خالی * src1_صف  = (ثابت حرف *) src1->ڈیٹا + i13*nb13 + i12*nb12 + i11*nb11 + ne10_بلاک_آغاز*bs*nb10;
                    خالی       * wdata_صف =                    wdata  + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_بلاک_آغاز*nbw0;

                    ثابت int64_t ne10_بلاک_سائز = (ne10_بلاک_انتہا - ne10_بلاک_آغاز) * bs;

                    اگر (src1->type == GGML_TYPE_F16) {
                        ggml_cpu_fp16_مقصد_fp32((ثابت ggml_fp16_t *) src1_صف, (اعشاری *) wdata_صف, ne10_بلاک_سائز);
                    } ورنہ اگر (src1->type == GGML_TYPE_BF16) {
                        ggml_cpu_bf16_مقصد_fp32((ثابت ggml_bf16_t *) src1_صف, (اعشاری *) wdata_صف, ne10_بلاک_سائز);
                    } ورنہ {
                        from_float((ثابت اعشاری *) src1_صف, wdata_صف, ne10_بلاک_سائز);
                    }
                }
            }
        }
    #اگر_ختم
    }

    اگر (ith == 0) {
        // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
        atomic_store_explicit(&params->threadpool->current_chunk, nth, memory_order_relaxed);
    }

    ggml_barrier(params->threadpool);

#اگر GGML_USE_LLAMAFILE
    اگر (!params->use_ref && src1->type != vec_dot_type) {
        ثابت خالی* wdata = (src1->type == vec_dot_type) ? src1->ڈیٹا : params->wdata;
        ثابت size_t صف_سائز = ggml_صف_سائز(vec_dot_type, ne10);

        برائے (int64_t i13 = 0; i13 < ne13; i13++)
            برائے (int64_t i12 = 0; i12 < ne12; i12++)
                اگر (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_سائز(src0->type),
                                     (ثابت حرف *)src0->ڈیٹا + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_سائز(src0->type),
                                     (ثابت حرف *)wdata + (i12*ne11 + i13*ne12*ne11)*صف_سائز,
                                     صف_سائز/ggml_type_سائز(vec_dot_type),
                                     (حرف *)dst->ڈیٹا + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_سائز(dst->type),
                                     src0->type,
                                     vec_dot_type,
                                     dst->type))
                    جاؤ UseGgmlGemm2;
        واپس;
    }
UseGgmlGemm2:;
#اگر_ختم

    // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
    ثابت int64_t nr0 = ne0;

    // This is the size of the rest of the dimensions of the result
    ثابت int64_t nr1 = ne1 * ne2 * ne3;

    // Now select a reasonable chunk size.
    صحیح chunk_سائز = 16;

    // We need to step up the size if it's small
    اگر (nr0 == 1 || nr1 == 1) {
        chunk_سائز = 64;
    }

    // distribute the work across the inner or outer loop based on which one is larger
    // The number of chunks in the 0/1 dim.
    // CEIL(nr0/chunk_size)
    int64_t nchunk0 = (nr0 + chunk_سائز - 1) / chunk_سائز;
    int64_t nchunk1 = (nr1 + chunk_سائز - 1) / chunk_سائز;

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

    جب (current_chunk < nchunk0 * nchunk1) {
        ثابت int64_t ith0 = current_chunk % nchunk0;
        ثابت int64_t ith1 = current_chunk / nchunk0;

        ثابت int64_t ir0_آغاز = dr0 * ith0;
        ثابت int64_t ir0_انتہا = MIN(ir0_آغاز + dr0, nr0);

        ثابت int64_t ir1_آغاز = dr1 * ith1;
        ثابت int64_t ir1_انتہا = MIN(ir1_آغاز + dr1, nr1);

        // dot kernels can handle 1 row and col at a time, but mmla kernels can process 2 rows and cols
        int64_t عدد_صفیں_فی_vec_dot = vec_dot_عدد_صفیں;

        // these checks are needed to avoid crossing dim1 boundaries
        // can be optimized, but the logic would become more complicated, so keeping it like this for simplicity
        اگر ((nr0 % 2 != 0) || (ne11 % 2 != 0) || ((ir0_انتہا - ir0_آغاز) % 2 != 0) || ((ir1_انتہا - ir1_آغاز) % 2 != 0)) {
            عدد_صفیں_فی_vec_dot = 1;
        }
        ggml_compute_forward_mul_mat_one_chunk(params, dst, src0->type, عدد_صفیں_فی_vec_dot, ir0_آغاز, ir0_انتہا, ir1_آغاز, ir1_انتہا);

        اگر (nth >= nchunk0 * nchunk1) {
            توڑو;
        }

        current_chunk = atomic_fetch_جوڑو_explicit(&params->threadpool->current_chunk, 1, memory_order_relaxed);
    }
}

// ggml_compute_forward_mul_mat_id

#تعریف_کرو MMID_میٹرکس_صف(صف_id, i1) میٹرکس_صفیں[(صف_id)*ids->ne[0]*ids->ne[1] + (i1)]

ساخت mmid_صف_mapping {
    int32_t i1;
    int32_t i2;
};

ساکن خالی ggml_compute_forward_mul_mat_id_one_chunk(
    ساخت ggml_tensor * dst,
    ثابت ساخت ggml_tensor * src0,
    ثابت ساخت ggml_tensor * src1,
    ثابت ساخت ggml_tensor * ids,
    ثابت int64_t cur_a,
    ثابت int64_t ir0_آغاز,
    ثابت int64_t ir0_انتہا,
    ثابت int64_t ir1_آغاز,
    ثابت int64_t ir1_انتہا,
    ثابت حرف * src0_cur,
    ثابت ساخت mmid_صف_mapping * میٹرکس_صفیں,
    ثابت size_t صف_سائز,
    ثابت bool src1_cont,
    ثابت خالی * wdata) {

    GGML_TENSOR_ثنائی_OP_LOCALS

    ثابت شمار ggml_type type = src0->type;

    ggml_vec_dot_t    ثابت vec_dot      = type_traits_cpu[type].vec_dot;
    شمار ggml_type    ثابت vec_dot_type = type_traits_cpu[type].vec_dot_type;

    ثابت int64_t blck_0 = 16;
    ثابت int64_t blck_1 = 16;

    اعشاری tmp[16];

    برائے (int64_t iir1 = ir1_آغاز; iir1 < ir1_انتہا; iir1 += blck_1) {
        برائے (int64_t iir0 = ir0_آغاز; iir0 < ir0_انتہا; iir0 += blck_0) {
            برائے (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_انتہا; ++ir1) {
                ثابت int64_t _i12 = ir1; // logical row index for this expert

                ساخت mmid_صف_mapping صف_mapping = MMID_میٹرکس_صف(cur_a, _i12);
                ثابت صحیح id       = صف_mapping.i1; // selected expert index

                ثابت int64_t  i11 = id % ne11;
                ثابت int64_t  i12 = صف_mapping.i2; // row index in src1

                ثابت int64_t  i1 = id;  // selected expert index
                ثابت int64_t  i2 = i12; //صف

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                ثابت حرف * src1_ستون = (ثابت حرف *) wdata +
                    (src1_cont || src1->type != vec_dot_type
                    ? (i11      + i12*ne11)*صف_سائز
                    : (i11*nb11 + i12*nb12));

                اعشاری * dst_ستون = (اعشاری *) ((حرف *) dst->ڈیٹا + (i1*nb1 + i2*nb2));

                برائے (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_انتہا; ++ir0) {
                    vec_dot(ne00, &tmp[ir0 - iir0], 0, src0_cur + ir0*nb01, 0, src1_ستون, 0, 1);
                }

                حافظہ_نقل(&dst_ستون[iir0], tmp, (MIN(iir0 + blck_0, ir0_انتہا) - iir0)*حجم(اعشاری));
            }
        }
    }
}

ساکن خالی * incr_ptr_aligned(خالی ** p, size_t سائز, size_t align) {

    خالی * ptr = *p;
    ptr = (خالی *) GGML_PAD((uintptr_t) ptr, align);
    *p = (خالی *) ((حرف *) ptr + سائز);
    واپس ptr;
}

ساکن خالی ggml_compute_forward_mul_mat_id(
        ثابت ساخت ggml_compute_params * params,
              ساخت ggml_tensor * dst) {

    ثابت ساخت ggml_tensor * src0 = dst->src[0];
    ثابت ساخت ggml_tensor * src1 = dst->src[1];
    ثابت ساخت ggml_tensor * ids = dst->src[2];

    GGML_TENSOR_ثنائی_OP_LOCALS

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت شمار ggml_type type = src0->type;

    ثابت bool src1_cont = ggml_is_contiguous(src1);

    شمار ggml_type    ثابت vec_dot_type    = type_traits_cpu[type].vec_dot_type;
    ggml_from_float_t ثابت from_float      = type_traits_cpu[vec_dot_type].from_float;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_سائز(type));
    GGML_ASSERT(nb10 == ggml_type_سائز(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == حجم(اعشاری));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // row groups
    ثابت صحیح n_ids = ids->ne[0]; // n_expert_used
    ثابت صحیح n_as  = ne02;       // n_expert

    خالی * wdata_cur = params->wdata;

    اگر (src1->type != vec_dot_type) {
        incr_ptr_aligned(&wdata_cur, ggml_صف_سائز(vec_dot_type, ggml_nelements(src1)), حجم(int64_t));
    }

    int64_t * میٹرکس_صف_counts = // [n_as]
        incr_ptr_aligned(&wdata_cur, n_as*حجم(int64_t), حجم(int64_t));

    ساخت mmid_صف_mapping * میٹرکس_صفیں = // [n_as][ids->ne[0]*ids->ne[1]]
        incr_ptr_aligned(&wdata_cur, n_as*ids->ne[0]*ids->ne[1]*حجم(ساخت mmid_صف_mapping), حجم(int64_t));

    حرف (*atomic_current_chunk)[CACHE_سطر_سائز] = // [n_as]
        incr_ptr_aligned(&wdata_cur, CACHE_سطر_سائز * n_as, CACHE_سطر_سائز);

    // Tiled matmul (see tiled.h); per-thread work buffers, 0 bytes when disabled. The
    // reservation is unconditional, the per expert eligibility is decided at dispatch time
    حرف * tiled_scratch = incr_ptr_aligned(&wdata_cur, ggml_tiled_wdata_سائز(nth, dst), 64);

    GGML_ASSERT(params->wsize >= (size_t)((حرف *) wdata_cur - (حرف *) params->wdata));

    اگر (src1->type != vec_dot_type) {
        حرف * wdata = params->wdata;

        ثابت size_t nbw0 = ggml_type_سائز(vec_dot_type);
        ثابت size_t nbw1 = ggml_صف_سائز(vec_dot_type, ne10);
        ثابت size_t nbw2 = nbw1*ne11;
        ثابت size_t nbw3 = nbw2*ne12;

        أكد(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

#اگر 0
        برائے (int64_t i13 = 0; i13 < ne13; ++i13) {
            برائے (int64_t i12 = ith; i12 < ne12; i12 += nth) {
                برائے (int64_t i11 = 0; i11 < ne11; ++i11) {
                    from_float((اعشاری *)((حرف *) src1->ڈیٹا + i13*nb13 + i12*nb12 + i11*nb11),
                               (خالی *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                               ne10);
                }
            }
        }
#ورنہ
        برائے (int64_t i13 = 0; i13 < ne13; ++i13) {
            برائے (int64_t i12 = 0; i12 < ne12; ++i12) {
                برائے (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = ggml_blck_سائز(vec_dot_type);
                    int64_t ne10_بلاک_آغاز = (ith * ne10/bs) / nth;
                    int64_t ne10_بلاک_انتہا   = ((ith + 1) * ne10/bs) / nth;
                    from_float((اعشاری *)((حرف *) src1->ڈیٹا + i13*nb13 + i12*nb12 + i11*nb11 + ne10_بلاک_آغاز*bs*nb10),
                               (خالی *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_بلاک_آغاز*nbw0),
                               (ne10_بلاک_انتہا - ne10_بلاک_آغاز) * bs);
                }
            }
        }
#اگر_ختم
    }

    اگر (ith == 0) {
        // initialize matrix_row_counts
        memset(میٹرکس_صف_counts, 0, n_as*حجم(int64_t));

        // group rows by src0 matrix
        برائے (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            برائے (صحیح id = 0; id < n_ids; ++id) {
                ثابت int32_t i02 = *(ثابت int32_t *) ((ثابت حرف *) ids->ڈیٹا + iid1*ids->nb[1] + id*ids->nb[0]);

                أكد(i02 >= 0 && i02 < n_as);

                MMID_میٹرکس_صف(i02, میٹرکس_صف_counts[i02]) = (ساخت mmid_صف_mapping) {id, iid1};
                میٹرکس_صف_counts[i02] += 1;
            }
        }
    }

    // reset current_chunk
    برائے (صحیح cur_a = ith; cur_a < n_as; cur_a += nth) {
        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);
        *current_chunk_ctr = nth;
    }

    ggml_barrier(params->threadpool);

    برائے (صحیح cur_a = 0; cur_a < n_as; ++cur_a) {
        ثابت int64_t cne1 = میٹرکس_صف_counts[cur_a];

        اگر (cne1 == 0) {
            جاری;
        }

        // tiled takes over if profitable for this expert (see tiled.h)
        اگر (ggml_compute_forward_mul_mat_id_tiled(params, dst, cur_a, cne1, (ثابت int32_t *) &MMID_میٹرکس_صف(cur_a, 0), tiled_scratch)) {
            جاری;
        }

        ثابت حرف * src0_cur = (ثابت حرف *) src0->ڈیٹا + cur_a * nb02;
        ثابت خالی * wdata = (src1->type == vec_dot_type) ? src1->ڈیٹا : params->wdata;
        ثابت size_t صف_سائز = ggml_صف_سائز(vec_dot_type, ne10);

        ثابت int64_t nr0 = ne01;
        ثابت int64_t nr1 = cne1;

        صحیح chunk_سائز = 16;
        اگر (nr0 == 1 || nr1 == 1) {
            chunk_سائز = 64;
        }

        // disable for NUMA
        ثابت bool disable_chunking = ggml_is_numa();

        int64_t nchunk0 = (nr0 + chunk_سائز - 1) / chunk_سائز;
        int64_t nchunk1 = (nr1 + chunk_سائز - 1) / chunk_سائز;

        اگر (nchunk0 * nchunk1 < nth * 4 || disable_chunking) {
            nchunk0 = nr0 > nr1 ? nth : 1;
            nchunk1 = nr0 > nr1 ? 1 : nth;
        }

        ثابت int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        ثابت int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

        صحیح current_chunk = ith;

        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);

        جب (current_chunk < nchunk0 * nchunk1) {
            ثابت int64_t ith0 = current_chunk % nchunk0;
            ثابت int64_t ith1 = current_chunk / nchunk0;

            ثابت int64_t ir0_آغاز = dr0 * ith0;
            ثابت int64_t ir0_انتہا = MIN(ir0_آغاز + dr0, nr0);

            ثابت int64_t ir1_آغاز = dr1 * ith1;
            ثابت int64_t ir1_انتہا = MIN(ir1_آغاز + dr1, nr1);

            ggml_compute_forward_mul_mat_id_one_chunk(
                dst, src0, src1, ids, cur_a,
                ir0_آغاز, ir0_انتہا, ir1_آغاز, ir1_انتہا,
                src0_cur, میٹرکس_صفیں, صف_سائز, src1_cont, wdata
            );

            اگر (nth >= nchunk0 * nchunk1) {
                توڑو;
            }

            current_chunk = atomic_fetch_جوڑو_explicit(current_chunk_ctr, 1, memory_order_relaxed);
        }
    }
}

/////////////////////////////////

ساکن خالی ggml_compute_forward(ساخت ggml_compute_params * params, ساخت ggml_tensor * tensor) {
    GGML_ASSERT(params);

    اگر (tensor->op == GGML_OP_NONE || ggml_is_خالی_خانے(tensor)) {
        واپس;
    }

    // extra_buffer op?
    اگر (ggml_cpu_extra_compute_forward(params, tensor)) {
        واپس;
    }

    انتخاب (tensor->op) {
        حالت GGML_OP_DUP:
            {
                ggml_compute_forward_dup(params, tensor);
            } توڑو;
        حالت GGML_OP_جوڑو:
            {
                ggml_compute_forward_جوڑو(params, tensor);
            } توڑو;
        حالت GGML_OP_جوڑو_ID:
            {
                ggml_compute_forward_جوڑو_id(params, tensor);
            } توڑو;
        حالت GGML_OP_ADD1:
            {
                ggml_compute_forward_add1(params, tensor);
            } توڑو;
        حالت GGML_OP_ACC:
            {
                ggml_compute_forward_acc(params, tensor);
            } توڑو;
        حالت GGML_OP_SUB:
            {
                ggml_compute_forward_sub(params, tensor);
            } توڑو;
        حالت GGML_OP_MUL:
            {
                ggml_compute_forward_mul(params, tensor);
            } توڑو;
        حالت GGML_OP_DIV:
            {
                ggml_compute_forward_div(params, tensor);
            } توڑو;
        حالت GGML_OP_SQR:
            {
                ggml_compute_forward_sqr(params, tensor);
            } توڑو;
        حالت GGML_OP_SQRT:
            {
                ggml_compute_forward_sqrt(params, tensor);
            } توڑو;
        حالت GGML_OP_LOG:
            {
                ggml_compute_forward_log(params, tensor);
            } توڑو;
        حالت GGML_OP_SIN:
            {
                ggml_compute_forward_sin(params, tensor);
            } توڑو;
        حالت GGML_OP_COS:
            {
                ggml_compute_forward_cos(params, tensor);
            } توڑو;
        حالت GGML_OP_مجموع:
            {
                ggml_compute_forward_مجموع(params, tensor);
            } توڑو;
        حالت GGML_OP_مجموع_صفیں:
            {
                ggml_compute_forward_مجموع_صفیں(params, tensor);
            } توڑو;
        حالت GGML_OP_CUMSUM:
            {
                ggml_compute_forward_cumsum(params, tensor);
            } توڑو;
        حالت GGML_OP_اوسط:
            {
                ggml_compute_forward_اوسط(params, tensor);
            } توڑو;
        حالت GGML_OP_ARGMAX:
            {
                ggml_compute_forward_argmax(params, tensor);
            } توڑو;
        حالت GGML_OP_گنتی_EQUAL:
            {
                ggml_compute_forward_گنتی_equal(params, tensor);
            } توڑو;
        حالت GGML_OP_REPEAT:
            {
                ggml_compute_forward_repeat(params, tensor);
            } توڑو;
        حالت GGML_OP_REPEAT_عقب:
            {
                ggml_compute_forward_repeat_عقب(params, tensor);
            } توڑو;
        حالت GGML_OP_CONCAT:
            {
                ggml_compute_forward_concat(params, tensor);
            } توڑو;
        حالت GGML_OP_SILU_عقب:
            {
                ggml_compute_forward_silu_عقب(params, tensor);
            } توڑو;
        حالت GGML_OP_NORM:
            {
                ggml_compute_forward_norm(params, tensor);
            } توڑو;
        حالت GGML_OP_RMS_NORM:
            {
                ggml_compute_forward_rms_norm(params, tensor);
            } توڑو;
        حالت GGML_OP_RMS_NORM_عقب:
            {
                ggml_compute_forward_rms_norm_عقب(params, tensor);
            } توڑو;
        حالت GGML_OP_گروہ_NORM:
            {
                ggml_compute_forward_گروہ_norm(params, tensor);
            } توڑو;
        حالت GGML_OP_L2_NORM:
            {
                ggml_compute_forward_l2_norm(params, tensor);
            } توڑو;
        حالت GGML_OP_MUL_MAT:
            {
                ggml_compute_forward_mul_mat(params, tensor);
            } توڑو;
        حالت GGML_OP_MUL_MAT_ID:
            {
                ggml_compute_forward_mul_mat_id(params, tensor);
            } توڑو;
        حالت GGML_OP_برآمد_PROD:
            {
                ggml_compute_forward_برآمد_prod(params, tensor);
            } توڑو;
        حالت GGML_OP_SCALE:
            {
                ggml_compute_forward_scale(params, tensor);
            } توڑو;
        حالت GGML_OP_سیٹ:
            {
                ggml_compute_forward_سیٹ(params, tensor);
            } توڑو;
        حالت GGML_OP_CPY:
            {
                ggml_compute_forward_cpy(params, tensor);
            } توڑو;
        حالت GGML_OP_CONT:
            {
                ggml_compute_forward_cont(params, tensor);
            } توڑو;
        حالت GGML_OP_GET_صفیں:
            {
                ggml_compute_forward_get_صفیں(params, tensor);
            } توڑو;
        حالت GGML_OP_GET_صفیں_عقب:
            {
                ggml_compute_forward_get_صفیں_عقب(params, tensor);
            } توڑو;
        حالت GGML_OP_سیٹ_صفیں:
            {
                ggml_compute_forward_سیٹ_صفیں(params, tensor);
            } توڑو;
        حالت GGML_OP_DIAG:
            {
                ggml_compute_forward_diag(params, tensor);
            } توڑو;
        حالت GGML_OP_DIAG_نقاب_INF:
            {
                ggml_compute_forward_diag_نقاب_inf(params, tensor);
            } توڑو;
        حالت GGML_OP_DIAG_نقاب_ZERO:
            {
                ggml_compute_forward_diag_نقاب_zero(params, tensor);
            } توڑو;
        حالت GGML_OP_SOFT_MAX:
            {
                ggml_compute_forward_soft_max(params, tensor);
            } توڑو;
        حالت GGML_OP_SOFT_MAX_عقب:
            {
                ggml_compute_forward_soft_max_ext_عقب(params, tensor);
            } توڑو;
        حالت GGML_OP_ROPE:
            {
                ggml_compute_forward_rope(params, tensor);
            } توڑو;
        حالت GGML_OP_ROPE_عقب:
            {
                ggml_compute_forward_rope_عقب(params, tensor);
            } توڑو;
        حالت GGML_OP_CLAMP:
            {
                ggml_compute_forward_clamp(params, tensor);
            } توڑو;
        حالت GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_compute_forward_conv_transpose_1d(params, tensor);
            } توڑو;
        حالت GGML_OP_IM2_ستون:
            {
                ggml_compute_forward_im2col(params, tensor);
            } توڑو;
        حالت GGML_OP_IM2_ستون_عقب:
            {
                ggml_compute_forward_im2col_عقب_f32(params, tensor);
            } توڑو;
        حالت GGML_OP_IM2_ستون_3_D:
            {
                ggml_compute_forward_im2col_3d(params, tensor);
            } توڑو;
        حالت GGML_OP_COL2IM_1D:
            {
                ggml_compute_forward_col2im_1d(params, tensor);
            } توڑو;
        حالت GGML_OP_CONV_2D:
            {
                ggml_compute_forward_conv_2d(params, tensor);
            } توڑو;
        حالت GGML_OP_CONV_3D:
            {
                ggml_compute_forward_conv_3d(params, tensor);
            } توڑو;
        حالت GGML_OP_CONV_2D_DW:
            {
                ggml_compute_forward_conv_2d_dw(params, tensor);
            } توڑو;
        حالت GGML_OP_CONV_TRANSPOSE_2D:
            {
                ggml_compute_forward_conv_transpose_2d(params, tensor);
            } توڑو;
        حالت GGML_OP_POOL_1D:
            {
                ggml_compute_forward_pool_1d(params, tensor);
            } توڑو;
        حالت GGML_OP_POOL_2D:
            {
                ggml_compute_forward_pool_2d(params, tensor);
            } توڑو;
        حالت GGML_OP_POOL_2_D_عقب:
            {
                ggml_compute_forward_pool_2d_عقب(params, tensor);
            } توڑو;
        حالت GGML_OP_UPSCALE:
            {
                ggml_compute_forward_upscale(params, tensor);
            } توڑو;
        حالت GGML_OP_PAD:
            {
                ggml_compute_forward_pad(params, tensor);
            } توڑو;
        حالت GGML_OP_PAD_REFLECT_1D:
            {
                ggml_compute_forward_pad_reflect_1d(params, tensor);
            } توڑو;
        حالت GGML_OP_ROLL:
            {
                ggml_compute_forward_roll(params, tensor);
            } توڑو;
        حالت GGML_OP_ARANGE:
            {
                ggml_compute_forward_arange(params, tensor);
            } توڑو;
        حالت GGML_OP_TIMESTEP_EMBEDDING:
            {
                ggml_compute_forward_timestep_embedding(params, tensor);
            } توڑو;
        حالت GGML_OP_ARGSORT:
            {
                ggml_compute_forward_argsort(params, tensor);
            } توڑو;
        حالت GGML_OP_چوٹی_K:
            {
                ggml_compute_forward_چوٹی_k(params, tensor);
            } توڑو;
        حالت GGML_OP_LEAKY_RELU:
            {
                ggml_compute_forward_leaky_relu(params, tensor);
            } توڑو;
        حالت GGML_OP_TRI:
            {
                ggml_compute_forward_tri(params, tensor);
            } توڑو;
        حالت GGML_OP_FILL:
            {
                ggml_compute_forward_fill(params, tensor);
            } توڑو;
        حالت GGML_OP_FLASH_ATTN_EXT:
            {
                ggml_compute_forward_flash_attn_ext(params, tensor);
            } توڑو;
        حالت GGML_OP_FLASH_ATTN_عقب:
            {
                int32_t t = ggml_get_op_params_i32(tensor, 0);
                GGML_ASSERT(t == 0 || t == 1);
                bool masked = t != 0;
                ggml_compute_forward_flash_attn_عقب(params, masked, tensor);
            } توڑو;
        حالت GGML_OP_SSM_CONV:
            {
                ggml_compute_forward_ssm_conv(params, tensor);
            } توڑو;
        حالت GGML_OP_SSM_جائزہ:
            {
                ggml_compute_forward_ssm_جائزہ(params, tensor);
            } توڑو;
        حالت GGML_OP_WIN_PART:
            {
                ggml_compute_forward_win_part(params, tensor);
            } توڑو;
        حالت GGML_OP_WIN_UNPART:
            {
                ggml_compute_forward_win_unpart(params, tensor);
            } توڑو;
        حالت GGML_OP_UNARY:
            {
                ggml_compute_forward_unary(params, tensor);
            } توڑو;
        حالت GGML_OP_GLU:
            {
                ggml_compute_forward_glu(params, tensor);
            } توڑو;
        حالت GGML_OP_GET_REL_POS:
            {
                ggml_compute_forward_get_rel_pos(params, tensor);
            } توڑو;
        حالت GGML_OP_جوڑو_REL_POS:
            {
                ggml_compute_forward_جوڑو_rel_pos(params, tensor);
            } توڑو;
        حالت GGML_OP_RWKV_WKV6:
            {
                ggml_compute_forward_rwkv_wkv6(params, tensor);
            } توڑو;
        حالت GGML_OP_GATED_خطی_ATTN:
            {
                ggml_compute_forward_gla(params, tensor);
            } توڑو;
        حالت GGML_OP_RWKV_WKV7:
            {
                ggml_compute_forward_rwkv_wkv7(params, tensor);
            } توڑو;
        حالت GGML_OP_SOLVE_TRI:
            {
                ggml_compute_forward_solve_tri(params, tensor);
            } توڑو;
        حالت GGML_OP_GATED_DELTA_NET:
            {
                ggml_compute_forward_gated_delta_net(params, tensor);
            } توڑو;
        حالت GGML_OP_LIGHTNING_INDEXER:
            {
                ggml_compute_forward_lightning_indexer(params, tensor);
            } توڑو;
        حالت GGML_OP_DSV4_HC_COMB:
            {
                ggml_compute_forward_dsv4_hc_comb(params, tensor);
            } توڑو;
        حالت GGML_OP_DSV4_HC_PRE:
            {
                ggml_compute_forward_dsv4_hc_pre(params, tensor);
            } توڑو;
        حالت GGML_OP_DSV4_HC_POST:
            {
                ggml_compute_forward_dsv4_hc_post(params, tensor);
            } توڑو;
        حالت GGML_OP_چوٹی_K_نقاب:
            {
                ggml_compute_forward_چوٹی_k_نقاب(params, tensor);
            } توڑو;
        حالت GGML_OP_MAP_CUSTOM1:
            {
                ggml_compute_forward_map_custom1(params, tensor);
            }
            توڑو;
        حالت GGML_OP_MAP_CUSTOM2:
            {
                ggml_compute_forward_map_custom2(params, tensor);
            }
            توڑو;
        حالت GGML_OP_MAP_CUSTOM3:
            {
                ggml_compute_forward_map_custom3(params, tensor);
            }
            توڑو;
        حالت GGML_OP_CUSTOM:
            {
                ggml_compute_forward_custom(params, tensor);
            }
            توڑو;
        حالت GGML_OP_CROSS_ENTROPY_LOSS:
            {
                ggml_compute_forward_cross_entropy_loss(params, tensor);
            }
            توڑو;
        حالت GGML_OP_CROSS_ENTROPY_LOSS_عقب:
            {
                ggml_compute_forward_cross_entropy_loss_عقب(params, tensor);
            }
            توڑو;
        حالت GGML_OP_OPT_قدم_ADAMW:
            {
                ggml_compute_forward_opt_قدم_adamw(params, tensor);
            }
            توڑو;
        حالت GGML_OP_OPT_قدم_SGD:
            {
                ggml_compute_forward_opt_قدم_sgd(params, tensor);
            }
            توڑو;
        حالت GGML_OP_NONE:
            {
                // nop
            } توڑو;
        حالت GGML_OP_RESHAPE:
            {
                // nop
            } توڑو;
        حالت GGML_OP_ترتیب_بدلو:
            {
                // nop
            } توڑو;
        حالت GGML_OP_VIEW:
            {
                // nop
            } توڑو;
        حالت GGML_OP_TRANSPOSE:
            {
                // nop
            } توڑو;
        حالت GGML_OP_گنتی:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// Android's libc implementation "bionic" does not support setting affinity
#اگر defined(__gnu_linux__)
ساکن خالی سیٹ_numa_thread_affinity(صحیح thread_n) {
    اگر (!ggml_is_numa()) {
        واپس;
    }

    صحیح گرہ_عدد;
    صحیح rv;
    size_t setsize = CPU_ALLOC_سائز(g_state.numa.کل_cpus);

    انتخاب(g_state.numa.numa_strategy) {
        حالت GGML_NUMA_STRATEGY_DISTRIBUTE:
            // run thread on node_num thread_n / (threads per node)
            گرہ_عدد = thread_n % g_state.numa.n_nodes;
            توڑو;
        حالت GGML_NUMA_STRATEGY_ISOLATE:
            // run thread on current_node
            گرہ_عدد = g_state.numa.current_گرہ;
            توڑو;
        حالت GGML_NUMA_STRATEGY_NUMACTL:
            // use the cpuset that numactl gave us
            rv = pthread_setaffinity_np(pthread_self(), setsize, &g_state.numa.cpuset);
            اگر (rv) {
                fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n",strerror(rv));
            }
            واپس;
        طےشدہ:
            واپس;
    }

    ساخت ggml_numa_گرہ * گرہ = &g_state.numa.nodes[گرہ_عدد];

    cpu_سیٹ_t * cpus = CPU_ALLOC(g_state.numa.کل_cpus);
    CPU_ZERO_S(setsize, cpus);
    برائے (size_t i = 0; i < گرہ->n_cpus; ++i) {
        CPU_سیٹ_S(گرہ->cpus[i], setsize, cpus);
    }

    rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    اگر (rv) {
            fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}

ساکن خالی clear_numa_thread_affinity(خالی) {
    اگر (!ggml_is_numa()) {
        واپس;
    }

    size_t setsize = CPU_ALLOC_سائز(g_state.numa.کل_cpus);

    cpu_سیٹ_t * cpus = CPU_ALLOC(g_state.numa.کل_cpus);
    CPU_ZERO_S(setsize, cpus);
    برائے (غیرعلامتی i = 0; i < g_state.numa.کل_cpus; ++i) {
        CPU_سیٹ_S(i, setsize, cpus);
    }

    صحیح rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    اگر (rv) {
        fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}
#ورنہ
// TODO: Windows etc.
// (the linux implementation may also work on BSD, someone should test)
ساکن خالی سیٹ_numa_thread_affinity(صحیح thread_n) { غیر_مستعمل(thread_n);  }
ساکن خالی clear_numa_thread_affinity(خالی) {}
#اگر_ختم

ساکن صحیح ggml_get_n_tasks(ساخت ggml_tensor * گرہ, صحیح n_دھاگے) {
    صحیح n_tasks = 0;

    اگر (ggml_is_خالی_خانے(گرہ)) {
        // no need to multi-thread a no-op
        n_tasks = 1;
        واپس n_tasks;
    }

    انتخاب (گرہ->op) {
        حالت GGML_OP_CPY:
        حالت GGML_OP_DUP:
        حالت GGML_OP_CONT:
        حالت GGML_OP_جوڑو:
        حالت GGML_OP_جوڑو_ID:
        حالت GGML_OP_ADD1:
        حالت GGML_OP_ACC:
        حالت GGML_OP_CUMSUM:
        حالت GGML_OP_TRI:
        حالت GGML_OP_FILL:
            {
                n_tasks = n_دھاگے;
            } توڑو;
        حالت GGML_OP_SUB:
        حالت GGML_OP_SQR:
        حالت GGML_OP_SQRT:
        حالت GGML_OP_LOG:
        حالت GGML_OP_SIN:
        حالت GGML_OP_COS:
        حالت GGML_OP_مجموع:
        حالت GGML_OP_مجموع_صفیں:
        حالت GGML_OP_اوسط:
        حالت GGML_OP_ARGMAX:
            {
                n_tasks = 1;
            } توڑو;
        حالت GGML_OP_گنتی_EQUAL:
        حالت GGML_OP_SOLVE_TRI:
        حالت GGML_OP_GATED_DELTA_NET:
        حالت GGML_OP_DSV4_HC_COMB:
        حالت GGML_OP_DSV4_HC_PRE:
        حالت GGML_OP_DSV4_HC_POST:
        حالت GGML_OP_چوٹی_K_نقاب:
            {
                n_tasks = n_دھاگے;
            } توڑو;
        حالت GGML_OP_REPEAT:
        حالت GGML_OP_REPEAT_عقب:
        حالت GGML_OP_LEAKY_RELU:
            {
                n_tasks = 1;
            } توڑو;
        حالت GGML_OP_UNARY:
            انتخاب (ggml_get_unary_op(گرہ)) {
                حالت GGML_UNARY_OP_ABS:
                حالت GGML_UNARY_OP_SGN:
                حالت GGML_UNARY_OP_NEG:
                حالت GGML_UNARY_OP_قدم:
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
                حالت GGML_UNARY_OP_باری:
                حالت GGML_UNARY_OP_TRUNC:
                    {
                        n_tasks = 1;
                    } توڑو;

                حالت GGML_UNARY_OP_GELU:
                حالت GGML_UNARY_OP_GELU_ERF:
                حالت GGML_UNARY_OP_GELU_تیز:
                حالت GGML_UNARY_OP_SILU:
                حالت GGML_UNARY_OP_XIELU:
                    {
                        n_tasks = n_دھاگے;
                    } توڑو;
                طےشدہ:
                    GGML_ABORT("fatal error");
            }
            توڑو;
        حالت GGML_OP_GLU:
            انتخاب (ggml_get_glu_op(گرہ)) {
                حالت GGML_GLU_OP_REGLU:
                حالت GGML_GLU_OP_GEGLU:
                حالت GGML_GLU_OP_SWIGLU:
                حالت GGML_GLU_OP_SWIGLU_OAI:
                حالت GGML_GLU_OP_GEGLU_ERF:
                حالت GGML_GLU_OP_GEGLU_تیز:
                حالت GGML_GLU_OP_SWIGLU_CLAMP:
                    {
                        n_tasks = n_دھاگے;
                    } توڑو;
                طےشدہ:
                    GGML_ABORT("fatal error");
            }
            توڑو;
        حالت GGML_OP_SILU_عقب:
        حالت GGML_OP_MUL:
        حالت GGML_OP_DIV:
        حالت GGML_OP_NORM:
        حالت GGML_OP_RMS_NORM:
        حالت GGML_OP_RMS_NORM_عقب:
        حالت GGML_OP_L2_NORM:
        حالت GGML_OP_گروہ_NORM:
        حالت GGML_OP_CONCAT:
        حالت GGML_OP_MUL_MAT:
        حالت GGML_OP_MUL_MAT_ID:
        حالت GGML_OP_برآمد_PROD:
        حالت GGML_OP_سیٹ_صفیں:
            {
                n_tasks = n_دھاگے;
            } توڑو;
        حالت GGML_OP_GET_صفیں:
            {
                // FIXME: get_rows can use additional threads, but the cost of launching additional threads
                // decreases performance with GPU offloading
                //n_tasks = n_threads;
                n_tasks = 1;
            } توڑو;
        حالت GGML_OP_SCALE:
        حالت GGML_OP_سیٹ:
        حالت GGML_OP_RESHAPE:
        حالت GGML_OP_VIEW:
        حالت GGML_OP_ترتیب_بدلو:
        حالت GGML_OP_TRANSPOSE:
        حالت GGML_OP_GET_صفیں_عقب:
        حالت GGML_OP_DIAG:
            {
                n_tasks = 1;
            } توڑو;
        حالت GGML_OP_DIAG_نقاب_ZERO:
        حالت GGML_OP_DIAG_نقاب_INF:
        حالت GGML_OP_SOFT_MAX_عقب:
        حالت GGML_OP_ROPE:
        حالت GGML_OP_ROPE_عقب:
        حالت GGML_OP_جوڑو_REL_POS:
            {
                n_tasks = n_دھاگے;
            } توڑو;
        حالت GGML_OP_CLAMP:
            {
                n_tasks = 1; //TODO
            } توڑو;
        حالت GGML_OP_SOFT_MAX:
            {
                n_tasks = MIN(n_دھاگے, ggml_nrows(گرہ->src[0]));
            } توڑو;
        حالت GGML_OP_IM2_ستون:
        حالت GGML_OP_IM2_ستون_عقب:
        حالت GGML_OP_IM2_ستون_3_D:
        حالت GGML_OP_CONV_2D:
        حالت GGML_OP_CONV_3D:
        حالت GGML_OP_CONV_2D_DW:
        حالت GGML_OP_COL2IM_1D:
        حالت GGML_OP_CONV_TRANSPOSE_1D:
        حالت GGML_OP_CONV_TRANSPOSE_2D:
            {
                n_tasks = n_دھاگے;
            } توڑو;
        حالت GGML_OP_POOL_1D:
        حالت GGML_OP_POOL_2D:
        حالت GGML_OP_POOL_2_D_عقب:
            {
                n_tasks = 1;
            } توڑو;
        حالت GGML_OP_UPSCALE:
        حالت GGML_OP_PAD:
        حالت GGML_OP_PAD_REFLECT_1D:
        حالت GGML_OP_ROLL:
        حالت GGML_OP_ARANGE:
        حالت GGML_OP_TIMESTEP_EMBEDDING:
        حالت GGML_OP_ARGSORT:
        حالت GGML_OP_چوٹی_K:
        حالت GGML_OP_FLASH_ATTN_EXT:
        حالت GGML_OP_FLASH_ATTN_عقب:
        حالت GGML_OP_SSM_CONV:
        حالت GGML_OP_SSM_جائزہ:
        حالت GGML_OP_LIGHTNING_INDEXER:
            {
                n_tasks = n_دھاگے;
            } توڑو;
        حالت GGML_OP_RWKV_WKV6:
        حالت GGML_OP_GATED_خطی_ATTN:
        حالت GGML_OP_RWKV_WKV7:
            {
                ثابت int64_t n_heads = گرہ->src[1]->ne[1];
                n_tasks = MIN(n_دھاگے, n_heads);
            } توڑو;
        حالت GGML_OP_WIN_PART:
        حالت GGML_OP_WIN_UNPART:
        حالت GGML_OP_GET_REL_POS:
            {
                n_tasks = 1;
            } توڑو;
        حالت GGML_OP_MAP_CUSTOM1:
            {
                ساخت ggml_map_custom1_op_params p;
                حافظہ_نقل(&p, گرہ->op_params, حجم(p));
                اگر (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_دھاگے;
                } ورنہ {
                    n_tasks = MIN(p.n_tasks, n_دھاگے);
                }
            } توڑو;
        حالت GGML_OP_MAP_CUSTOM2:
            {
                ساخت ggml_map_custom2_op_params p;
                حافظہ_نقل(&p, گرہ->op_params, حجم(p));
                اگر (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_دھاگے;
                } ورنہ {
                    n_tasks = MIN(p.n_tasks, n_دھاگے);
                }
            } توڑو;
        حالت GGML_OP_MAP_CUSTOM3:
            {
                ساخت ggml_map_custom3_op_params p;
                حافظہ_نقل(&p, گرہ->op_params, حجم(p));
                اگر (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_دھاگے;
                } ورنہ {
                    n_tasks = MIN(p.n_tasks, n_دھاگے);
                }
            } توڑو;
        حالت GGML_OP_CUSTOM:
            {
                ساخت ggml_custom_op_params p;
                حافظہ_نقل(&p, گرہ->op_params, حجم(p));
                اگر (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_دھاگے;
                } ورنہ {
                    n_tasks = MIN(p.n_tasks, n_دھاگے);
                }
            } توڑو;
        حالت GGML_OP_CROSS_ENTROPY_LOSS:
        حالت GGML_OP_CROSS_ENTROPY_LOSS_عقب:
        حالت GGML_OP_OPT_قدم_ADAMW:
        حالت GGML_OP_OPT_قدم_SGD:
            {
                n_tasks = n_دھاگے;
            } توڑو;
        حالت GGML_OP_NONE:
            {
                n_tasks = 1;
            } توڑو;
        حالت GGML_OP_گنتی:
            {
                GGML_ABORT("fatal error");
            }
        طےشدہ:
            {
                fprintf(stderr, "%s: op not implemented: ", __func__);
                اگر (گرہ->op < GGML_OP_گنتی) {
                    fprintf(stderr, "%s\n", ggml_op_name(گرہ->op));
                } ورنہ {
                    fprintf(stderr, "%d\n", گرہ->op);
                }
                GGML_ABORT("fatal error");
            }
    }

    أكد(n_tasks > 0);

    واپس n_tasks;
}

ساکن thread_ret_t ggml_گراف_compute_secondary_thread(خالی* ڈیٹا);

#اگر defined(_WIN32)
#شامل "windows.h"

// TODO: support > 64 CPUs
ساکن bool ggml_thread_apply_affinity(bool * نقاب) {
    HANDLE    h = GetCurrentThread();
    uint64_t  bitmask = 0ULL;

    أكد(GGML_MAX_N_دھاگے >= 64);

    برائے (int32_t i = 0; i < 8; i++) {
        int32_t idx = i * 8;
        uint8_t val = 0;
        val |= نقاب[idx + 0] << 0;
        val |= نقاب[idx + 1] << 1;
        val |= نقاب[idx + 2] << 2;
        val |= نقاب[idx + 3] << 3;
        val |= نقاب[idx + 4] << 4;
        val |= نقاب[idx + 5] << 5;
        val |= نقاب[idx + 6] << 6;
        val |= نقاب[idx + 7] << 7;
        bitmask |= (uint64_t)val << idx;
    }

    برائے (int32_t i = 64; i < GGML_MAX_N_دھاگے; i++) {
        اگر (نقاب[i]) {
            fprintf(stderr, "warn: setting thread-affinity for > 64 CPUs isn't supported on windows!\n");
            توڑو;
        }
    }

    DWORD_PTR m = (DWORD_PTR)bitmask;

    m = سیٹ_Thread_Affinity_نقاب(h, m);

    واپس m != 0;
}

ساکن bool ggml_thread_apply_ترجیح(int32_t prio) {
    // Note that on Windows the Process Priority Class must be updated in order to set Thread priority.
    // This is up to the applications.
    DWORD p = THREAD_ترجیح_معمول;
    انتخاب (prio) {
        حالت GGML_SCHED_PRIO_کم:      p = THREAD_ترجیح_نیچے_معمول;  توڑو;
        حالت GGML_SCHED_PRIO_معمول:   p = THREAD_ترجیح_معمول;        توڑو;
        حالت GGML_SCHED_PRIO_MEDIUM:   p = THREAD_ترجیح_ABOVE_معمول;  توڑو;
        حالت GGML_SCHED_PRIO_بلند:     p = THREAD_ترجیح_HIGHEST;       توڑو;
        حالت GGML_SCHED_PRIO_REALTIME: p = THREAD_ترجیح_وقت_CRITICAL; توڑو;
    }

    اگر (prio != GGML_SCHED_PRIO_کم) {
        // Tell Windows that this thread should not be throttled (needs its own CPU core).
        // Newer Windows 11 versions aggressively park (offline) CPU cores and often place
        // all our threads onto the first 4 cores which results in terrible performance with
        // n_threads > 4
        #اگر _WIN32_WINNT >= 0x0602
        THREAD_POWER_THROTTLING_STATE t;
        Zero_حافظہ(&t, حجم(t));
        t.نسخہ     = THREAD_POWER_THROTTLING_CURRENT_نسخہ;
        t.Control_نقاب = THREAD_POWER_THROTTLING_EXECUTION_رفتار;
        t.State_نقاب   = 0;

        اگر (!سیٹ_Thread_Information(GetCurrentThread(), ThreadPowerThrottling, &t, حجم(t))) {
            GGML_LOG_DEBUG("failed to disable thread power throttling %d : (%d)\n", prio, (صحیح) Get_آخری_Error());
            واپس false;
        }
        #اگر_ختم
    }

    اگر (prio == GGML_SCHED_PRIO_معمول) {
        // Keep inherited policy/priority
        واپس true;
    }

    اگر (!سیٹ_Thread_ترجیح(GetCurrentThread(), p)) {
        fprintf(stderr, "warn: failed to set thread priority %d : (%d)\n", prio, (صحیح) Get_آخری_Error());
        واپس false;
    }

    واپس true;
}

#وگرنہ_اگر defined(__APPLE__)
#شامل <sys/types.h>
#شامل <sys/resource.h>

ساکن bool ggml_thread_apply_affinity(ثابت bool * نقاب) {
    // Not supported on Apple platforms
    غیر_مستعمل(نقاب);
    واپس true;
}

ساکن bool ggml_thread_apply_ترجیح(int32_t prio) {
    ساخت sched_param p;
    int32_t policy = SCHED_OTHER;
    انتخاب (prio) {
        // TODO: there seems to be no way to set lower prio on Apple platforms
        حالت GGML_SCHED_PRIO_کم:      policy = SCHED_OTHER; p.sched_ترجیح = 0;  توڑو;
        حالت GGML_SCHED_PRIO_معمول:   policy = SCHED_OTHER; p.sched_ترجیح = 0;  توڑو;
        حالت GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_ترجیح = 40; توڑو;
        حالت GGML_SCHED_PRIO_بلند:     policy = SCHED_FIFO;  p.sched_ترجیح = 80; توڑو;
        حالت GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_ترجیح = 90; توڑو;
    }

    اگر (prio == GGML_SCHED_PRIO_معمول) {
        // Keep inherited policy/priority
        واپس true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    اگر (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        واپس false;
    }

    واپس true;
}

#وگرنہ_اگر defined(__linux__)
// TODO: this may not work on BSD, to be verified

ساکن bool ggml_thread_apply_affinity(ثابت bool * نقاب) {
    cpu_سیٹ_t cpuset;
    صحیح err;

    CPU_ZERO(&cpuset);

    برائے (uint32_t i = 0; i < GGML_MAX_N_دھاگے; i++) {
        اگر (نقاب[i]) {
            GGML_PRINT_DEBUG("Thread %lx: adding %d to cpuset\n", pthread_self(), i);
            CPU_سیٹ(i, &cpuset);
        }
    }

#اگر_تعریف_شدہ __ANDROID__
    err = sched_setaffinity(0, حجم(cpuset), &cpuset);
    اگر (err < 0) {
        err = errno;
    }
#ورنہ
    err = pthread_setaffinity_np(pthread_self(), حجم(cpuset), &cpuset);
#اگر_ختم
    اگر (err != 0) {
        fprintf(stderr, "warn: failed to set affinity mask 0x%llx : %s (%d)\n", (غیرعلامتی طویل طویل)نقاب, strerror(err), err);
        واپس false;
    }

    واپس true;
}

ساکن bool ggml_thread_apply_ترجیح(int32_t prio) {
    ساخت sched_param p;
    int32_t policy = SCHED_OTHER;
    انتخاب (prio) {
        حالت GGML_SCHED_PRIO_کم:      policy = SCHED_BATCH; p.sched_ترجیح = 0;  توڑو;
        حالت GGML_SCHED_PRIO_معمول:   policy = SCHED_OTHER; p.sched_ترجیح = 0;  توڑو;
        حالت GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_ترجیح = 40; توڑو;
        حالت GGML_SCHED_PRIO_بلند:     policy = SCHED_FIFO;  p.sched_ترجیح = 80; توڑو;
        حالت GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_ترجیح = 90; توڑو;
    }

    اگر (prio == GGML_SCHED_PRIO_معمول) {
        // Keep inherited policy/priority
        واپس true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    اگر (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        واپس false;
    }

    واپس true;
}

#ورنہ // unsupported platforms

ساکن bool ggml_thread_apply_affinity(ثابت bool * نقاب) {
    غیر_مستعمل(نقاب);
    واپس true;
}

ساکن bool ggml_thread_apply_ترجیح(int32_t prio) {
    غیر_مستعمل(prio);
    واپس true;
}

#اگر_ختم

ساکن bool ggml_thread_cpumask_is_valid(ثابت bool * نقاب) {
    برائے (صحیح i = 0; i < GGML_MAX_N_دھاگے; i++) {
        اگر (نقاب[i]) { واپس true; }
    }
    واپس false;
}

ساکن خالی ggml_thread_cpumask_اگلا(ثابت bool * global_نقاب, bool * local_نقاب, bool strict, int32_t* iter) {
    اگر (!strict) {
        حافظہ_نقل(local_نقاب, global_نقاب, GGML_MAX_N_دھاگے);
        واپس;
    } ورنہ {
        memset(local_نقاب, 0, GGML_MAX_N_دھاگے);
        int32_t بنیاد_idx = *iter;
        برائے (int32_t i = 0; i < GGML_MAX_N_دھاگے; i++) {
            int32_t idx = بنیاد_idx + i;
            اگر (idx >= GGML_MAX_N_دھاگے) {
                // Just a cheaper modulo
                idx -= GGML_MAX_N_دھاگے;
            }
            اگر (global_نقاب[idx]) {
                local_نقاب[idx] = 1;
                *iter = idx + 1;
                واپس;
            }
        }
    }
}

خالی ggml_threadpool_free(ساخت ggml_threadpool* threadpool) {
    اگر (!threadpool) واپس;

    ثابت صحیح n_دھاگے = threadpool->n_دھاگے;

#اگر_غیر_تعریف_شدہ GGML_USE_OPENMP
    ساخت ggml_compute_state* workers = threadpool->workers;

    ggml_mutex_تالا(&threadpool->mutex);

    threadpool->stop = true;
    threadpool->pause = false;

    ggml_cond_نشر(&threadpool->cond);
    ggml_mutex_unlock(&threadpool->mutex);

    برائے (صحیح j = 1; j < n_دھاگے; j++) {
        int32_t rc = ggml_thread_join(workers[j].thrd, تہی);
        GGML_ASSERT(rc == GGML_EXIT_SUCCESS || rc == GGML_EXIT_ABORTED);
        غیر_مستعمل(rc);
    }

    ggml_mutex_destroy(&threadpool->mutex);
    ggml_cond_destroy(&threadpool->cond);
#اگر_ختم // GGML_USE_OPENMP

    ثابت size_t workers_سائز = حجم(ساخت ggml_compute_state) * n_دھاگے;
    ggml_aligned_free(threadpool->workers, workers_سائز);
    ggml_aligned_free(threadpool, حجم(ساخت ggml_threadpool));
}

#اگر_غیر_تعریف_شدہ GGML_USE_OPENMP
// pause/resume must be called under mutex
ساکن خالی ggml_threadpool_pause_locked(ساخت ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Pausing threadpool\n");
    threadpool->pause = true;
    ggml_cond_نشر(&threadpool->cond);
}

ساکن خالی ggml_threadpool_resume_locked(ساخت ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Resuming threadpool\n");
    threadpool->pause = false;
    ggml_cond_نشر(&threadpool->cond);
}
#اگر_ختم

خالی ggml_threadpool_pause(ساخت ggml_threadpool * threadpool) {
#اگر_غیر_تعریف_شدہ GGML_USE_OPENMP
    ggml_mutex_تالا(&threadpool->mutex);
    اگر (!threadpool->pause) {
       ggml_threadpool_pause_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#ورنہ
    غیر_مستعمل(threadpool);
#اگر_ختم
}

خالی ggml_threadpool_resume(ساخت ggml_threadpool * threadpool) {
#اگر_غیر_تعریف_شدہ GGML_USE_OPENMP
    ggml_mutex_تالا(&threadpool->mutex);
    اگر (threadpool->pause) {
       ggml_threadpool_resume_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#ورنہ
    غیر_مستعمل(threadpool);
#اگر_ختم
}

ساخت ggml_cplan ggml_گراف_plan(
          ثابت ساخت ggml_cgraph * cgraph,
                               صحیح   n_دھاگے,
            ساخت ggml_threadpool * threadpool) {

    اگر (threadpool == تہی) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
    }
    اگر (n_دھاگے <= 0) {
        n_دھاگے = threadpool ? threadpool->n_دھاگے : GGML_DEFAULT_N_دھاگے;
    }

#اگر defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
    // Emscripten without pthreads support can only use a single thread
    n_دھاگے = 1;
#اگر_ختم

#اگر defined(__wasi__)
    // WASI doesn't support parallelism yet
    n_دھاگے = 1;
#اگر_ختم

    size_t work_سائز = 0;

    ساخت ggml_cplan cplan;
    memset(&cplan, 0, حجم(ساخت ggml_cplan));

    صحیح max_tasks = 1;

    // thread scheduling for the different operations + work buffer size estimation
    برائے (صحیح i = 0; i < cgraph->n_nodes; i++) {
        ساخت ggml_tensor * گرہ = cgraph->nodes[i];

        ثابت صحیح n_tasks = ggml_get_n_tasks(گرہ, n_دھاگے);

        max_tasks = MAX(max_tasks, n_tasks);

        size_t cur = 0;

        اگر (!ggml_cpu_extra_work_سائز(n_دھاگے, گرہ, &cur)) {
            انتخاب (گرہ->op) {
                حالت GGML_OP_CPY:
                حالت GGML_OP_DUP:
                    {
                        اگر (ggml_is_quantized(گرہ->type) ||
                            // F16 -> BF16 and BF16 -> F16 copies go through intermediate F32
                            (گرہ->src[0]->type == GGML_TYPE_F16  && گرہ->src[1] && گرہ->src[1]->type == GGML_TYPE_BF16) ||
                            (گرہ->src[0]->type == GGML_TYPE_BF16 && گرہ->src[1] && گرہ->src[1]->type == GGML_TYPE_F16) ||
                            // conversion between F32 and I32
                            (گرہ->src[0]->type == GGML_TYPE_F32 && گرہ->src[1] && گرہ->src[1]->type == GGML_TYPE_I32) ||
                            (گرہ->src[0]->type == GGML_TYPE_I32 && گرہ->src[1] && گرہ->src[1]->type == GGML_TYPE_F32)) {
                            cur = ggml_type_سائز(GGML_TYPE_F32) * گرہ->ne[0] * n_tasks;
                        }
                    } توڑو;
                حالت GGML_OP_جوڑو:
                حالت GGML_OP_جوڑو_ID:
                حالت GGML_OP_ADD1:
                    {
                        اگر (ggml_is_quantized(گرہ->src[0]->type)) {
                            cur = ggml_type_سائز(GGML_TYPE_F32) * گرہ->src[0]->ne[0] * n_tasks;
                        }
                    } توڑو;
                حالت GGML_OP_ACC:
                    {
                        اگر (ggml_is_quantized(گرہ->src[0]->type)) {
                            cur = ggml_type_سائز(GGML_TYPE_F32) * گرہ->src[1]->ne[0] * n_tasks;
                        }
                    } توڑو;
                حالت GGML_OP_گنتی_EQUAL:
                    {
                        cur = ggml_type_سائز(گرہ->type)*n_tasks;
                    } توڑو;
                حالت GGML_OP_MUL_MAT:
                    {
                        ثابت شمار ggml_type vec_dot_type = type_traits_cpu[گرہ->src[0]->type].vec_dot_type;
                        اگر (گرہ->src[1]->type != vec_dot_type) {
                            cur = ggml_صف_سائز(vec_dot_type, ggml_nelements(گرہ->src[1]));
                        }
                        // Workspace for tiled (see tiled.h)
                        cur = GGML_PAD(cur, 64);
                        cur += ggml_tiled_wdata_سائز(n_tasks, گرہ);
                    } توڑو;
                حالت GGML_OP_MUL_MAT_ID:
                    {
                        cur = 0;
                        ثابت ساخت ggml_tensor * src0 = گرہ->src[0];
                        ثابت ساخت ggml_tensor * src1 = گرہ->src[1];
                        ثابت ساخت ggml_tensor * ids = گرہ->src[2];
                        ثابت شمار ggml_type vec_dot_type = type_traits_cpu[src0->type].vec_dot_type;
                        ثابت صحیح n_as = src0->ne[2];
                        // src1
                        اگر (src1->type != vec_dot_type) {
                            cur += ggml_صف_سائز(vec_dot_type, ggml_nelements(src1)) + حجم(int64_t);
                        }
                        // matrix_row_counts
                        cur += n_as * حجم(int64_t) + حجم(int64_t);
                        // matrix_rows
                        cur += n_as*ids->ne[0]*ids->ne[1]*حجم(ساخت mmid_صف_mapping) + حجم(int64_t);
                        // atomic_current_chunk
                        cur += CACHE_سطر_سائز*n_as + CACHE_سطر_سائز;
                        // Workspace for tiled (see tiled.h)
                        cur = GGML_PAD(cur, 64);
                        cur += ggml_tiled_wdata_سائز(n_tasks, گرہ);
                    } توڑو;
                حالت GGML_OP_برآمد_PROD:
                    {
                        اگر (ggml_is_quantized(گرہ->src[0]->type) ||
                            گرہ->src[0]->type == GGML_TYPE_F16) {
                            cur = ggml_type_سائز(GGML_TYPE_F32) * گرہ->src[0]->ne[0] * n_tasks;
                        }
                    } توڑو;
                حالت GGML_OP_سیٹ_صفیں:
                    {
                        اگر (گرہ->src[0]->type == GGML_TYPE_F16 && گرہ->type != GGML_TYPE_F16) {
                            cur = ggml_type_سائز(GGML_TYPE_F32) * گرہ->src[0]->ne[0] * n_tasks;
                        }
                    } توڑو;
                حالت GGML_OP_SOFT_MAX:
                حالت GGML_OP_ROPE:
                حالت GGML_OP_ROPE_عقب:
                    {
                        cur = ggml_type_سائز(GGML_TYPE_F32) * گرہ->ne[0] * n_tasks;
                    } توڑو;
                حالت GGML_OP_CONV_TRANSPOSE_1D:
                    {
                        GGML_ASSERT(گرہ->src[0]->ne[3] == 1);
                        GGML_ASSERT(گرہ->src[1]->ne[2] == 1);
                        GGML_ASSERT(گرہ->src[1]->ne[3] == 1);

                        ثابت int64_t ne00 = گرہ->src[0]->ne[0];  // K
                        ثابت int64_t ne01 = گرہ->src[0]->ne[1];  // Cout
                        ثابت int64_t ne02 = گرہ->src[0]->ne[2];  // Cin
                        ثابت int64_t ne10 = گرہ->src[1]->ne[0];  // L
                        ثابت int64_t ne11 = گرہ->src[1]->ne[1];  // Cin

                        اگر ((گرہ->src[0]->type == GGML_TYPE_F16 ||
                             گرہ->src[0]->type == GGML_TYPE_BF16) &&
                            گرہ->src[1]->type == GGML_TYPE_F32) {
                            cur += حجم(ggml_fp16_t)*ne00*ne01*ne02;
                            cur += حجم(ggml_fp16_t)*ne10*ne11;
                        } ورنہ اگر (گرہ->src[0]->type == GGML_TYPE_F32 &&
                                   گرہ->src[1]->type == GGML_TYPE_F32) {
                            cur += حجم(اعشاری)*ne00*ne01*ne02;
                            cur += حجم(اعشاری)*ne10*ne11;
                        } ورنہ {
                            GGML_ABORT("fatal error");
                        }
                    } توڑو;
                حالت GGML_OP_CONV_2D:
                حالت GGML_OP_CONV_3D:
                    {
                        cur = GGML_IM2_ستون_WORK_سائز;
                    } توڑو;
                حالت GGML_OP_CONV_TRANSPOSE_2D:
                    {
                        ثابت int64_t ne00 = گرہ->src[0]->ne[0]; // W
                        ثابت int64_t ne01 = گرہ->src[0]->ne[1]; // H
                        ثابت int64_t ne02 = گرہ->src[0]->ne[2]; // Channels Out
                        ثابت int64_t ne03 = گرہ->src[0]->ne[3]; // Channels In

                        ثابت int64_t ne10 = گرہ->src[1]->ne[0]; // W
                        ثابت int64_t ne11 = گرہ->src[1]->ne[1]; // H
                        ثابت int64_t ne12 = گرہ->src[1]->ne[2]; // Channels In
                        ثابت int64_t ne13 = گرہ->src[1]->ne[3]; // Batch

                        GGML_ASSERT(گرہ->src[0]->type == GGML_TYPE_F16 || گرہ->src[0]->type == GGML_TYPE_F32);
                        GGML_ASSERT(گرہ->src[1]->type == GGML_TYPE_F32);

                        cur += ggml_type_سائز(گرہ->src[0]->type) * ne00 * ne01 * ne02 * ne03;
                        cur += ggml_type_سائز(گرہ->src[0]->type) * ne10 * ne11 * ne12 * ne13;

                    } توڑو;
                حالت GGML_OP_چوٹی_K:
                    {
                        cur += حجم(int32_t)*گرہ->src[0]->ne[0]*n_tasks;
                    } توڑو;
                حالت GGML_OP_FLASH_ATTN_EXT:
                    {
                        ثابت int64_t neq2 = گرہ->src[0]->ne[2]; // number of query heads
                        ثابت int64_t DK = گرہ->src[1]->ne[0];
                        ثابت int64_t DV = گرہ->src[2]->ne[0];

                        // Tiled flash attention scratch (tile sizes defined in common.h)
                        // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
                        size_t prefill  = حجم(اعشاری)*(GGML_FA_TILE_Q*DK + 2*GGML_FA_TILE_Q*GGML_FA_TILE_KV + GGML_FA_TILE_Q*DV + GGML_FA_TILE_KV*DV + GGML_FA_TILE_KV*DK)*n_tasks;

                        // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
                        // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
                        size_t n_chunks = n_tasks;
                        size_t decode   = حجم(اعشاری)*(neq2*n_chunks*(2+DV) + n_tasks*(DK + 2*DV));

                        cur += MAX(prefill, decode);
                    } توڑو;
                حالت GGML_OP_FLASH_ATTN_عقب:
                    {
                        ثابت int64_t    D = گرہ->src[0]->ne[0];
                        ثابت int64_t ne11 = ggml_اوپر(گرہ->src[1]->ne[1], GGML_SOFT_MAX_UNROLL);
                        ثابت int64_t mxDn = MAX(D, ne11) * 2; // *2 because of S and SM in ggml_compute_forward_flash_attn_back
                        اگر (گرہ->src[1]->type == GGML_TYPE_F32) {
                            cur  = حجم(اعشاری)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += حجم(اعشاری)*mxDn*n_tasks; // this is overestimated by x2
                        } ورنہ اگر (گرہ->src[1]->type == GGML_TYPE_F16) {
                            cur  = حجم(اعشاری)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += حجم(اعشاری)*mxDn*n_tasks; // this is overestimated by x2
                        } ورنہ اگر (گرہ->src[1]->type == GGML_TYPE_BF16) {
                            cur  = حجم(اعشاری)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += حجم(اعشاری)*mxDn*n_tasks; // this is overestimated by x2
                        }
                    } توڑو;

                حالت GGML_OP_CROSS_ENTROPY_LOSS:
                    {
                        cur = ggml_type_سائز(گرہ->type)*(n_tasks + گرہ->src[0]->ne[0]*n_tasks);
                    } توڑو;
                حالت GGML_OP_GATED_DELTA_NET:
                    {
                        ثابت int64_t S_v = گرہ->src[2]->ne[0];
                        ثابت int64_t K   = ggml_get_op_params_i32(گرہ, 0);
                        ثابت int64_t فی_thread = S_v + (K > 1 ? S_v * S_v : 0);
                        cur = فی_thread * حجم(اعشاری) * n_tasks;
                    } توڑو;
                حالت GGML_OP_گنتی:
                    {
                        GGML_ABORT("fatal error");
                    }
                حالت GGML_OP_LIGHTNING_INDEXER:
                    {
                        // temp buffer for dequantizing lightning indexer keys
                        ثابت int64_t ne10 = گرہ->src[1]->ne[0];
                        cur += حجم(اعشاری)*ne10*n_tasks;
                    } توڑو;
                طےشدہ:
                    توڑو;
            }
        }

        work_سائز = MAX(work_سائز, cur);
    }

    اگر (work_سائز > 0) {
        work_سائز += CACHE_سطر_سائز*(n_دھاگے);
    }

    cplan.threadpool = threadpool;
    cplan.n_دھاگے  = MIN(max_tasks, n_دھاگے);
    cplan.work_سائز  = work_سائز;
    cplan.work_ڈیٹا  = تہی;

    واپس cplan;
}


// Try to fuse the current node with subsequent nodes for better performance.
// Returns the number of nodes skipped by fusion (>=1), or 0 if no fusion was applied.
ساکن bool ggml_cpu_disable_fusion = false;  // initialized once in ggml_cpu_init(), read-only afterwards

ساکن صحیح ggml_cpu_try_fuse_ops(
        ثابت ساخت ggml_cgraph * cgraph,
        ثابت صحیح گرہ_n,
        ثابت ساخت ggml_compute_params * params,
        ثابت ساخت ggml_cplan * cplan) {

    اگر (ggml_cpu_disable_fusion || cplan->use_ref) {
        واپس 0;
    }

    ساخت ggml_tensor * گرہ = cgraph->nodes[گرہ_n];

    اگر (گرہ->op == GGML_OP_RMS_NORM) {
        // RMS_NORM + MUL fusion
        ثابت شمار ggml_op fuse_ops[] = { GGML_OP_RMS_NORM, GGML_OP_MUL };
        اگر (ggml_can_fuse(cgraph, گرہ_n, fuse_ops, 2)) {
            ساخت ggml_tensor * mul_گرہ = cgraph->nodes[گرہ_n + 1];
            ثابت ساخت ggml_tensor * mul_w = (mul_گرہ->src[0] == گرہ)
                ? mul_گرہ->src[1] : mul_گرہ->src[0];
            اگر (گرہ->src[0]->type  == GGML_TYPE_F32 &&
                mul_گرہ->type      == GGML_TYPE_F32 &&
                mul_w->type         == GGML_TYPE_F32 &&
                mul_w->ne[0]        == گرہ->ne[0]   &&
                mul_w->nb[0]        == حجم(اعشاری)) {

                ggml_compute_forward_rms_norm_mul_fused(params, گرہ, mul_گرہ);
                واپس 1;
            }
        }
    }

    واپس 0;
}

ساکن thread_ret_t ggml_گراف_compute_thread(خالی * ڈیٹا) {
    ساخت ggml_compute_state * state = (ساخت ggml_compute_state *) ڈیٹا;
    ساخت ggml_threadpool    * tp    = state->threadpool;

    ثابت ساخت ggml_cgraph * cgraph = tp->cgraph;
    ثابت ساخت ggml_cplan  * cplan  = tp->cplan;

#اگر_تعریف_شدہ GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_سیٹ_numa_thread_affinity(state->ith);
#ورنہ
    سیٹ_numa_thread_affinity(state->ith);
#اگر_ختم

    ساخت ggml_compute_params params = {
        /*.ith        =*/ state->ith,
        /*.nth        =*/ atomic_بھرو_explicit(&tp->n_گراف, memory_order_relaxed) & GGML_THREADPOOL_N_دھاگے_نقاب,
        /*.wsize      =*/ cplan->work_سائز,
        /*.wdata      =*/ cplan->work_ڈیٹا,
        /*.threadpool =*/ tp,
        /*.use_ref    =*/ cplan->use_ref,
    };

#اگر_تعریف_شدہ GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p\n", state->ith, (ثابت خالی *)cplan);
#ورنہ
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p last-graph %d\n", state->ith, (ثابت خالی *)cplan, state->آخری_گراف);
#اگر_ختم

    برائے (صحیح گرہ_n = 0; گرہ_n < cgraph->n_nodes && atomic_بھرو_explicit(&tp->abort, memory_order_relaxed) != گرہ_n; گرہ_n++) {
        ساخت ggml_tensor * گرہ = cgraph->nodes[گرہ_n];

        اگر (ggml_op_is_خالی_خانے(گرہ->op)) {
            // skip NOPs
            جاری;
        }

        اگر ((گرہ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            جاری;
        }

        // TODO: move fused-op detection into ggml_graph_plan so fusion decisions are made once at planning time
        // Try fused ops, fall back to normal compute
        ثابت صحیح n_fused = ggml_cpu_try_fuse_ops(cgraph, گرہ_n, &params, cplan);
        اگر (n_fused > 0) {
            گرہ_n += n_fused;
        } ورنہ {
            ggml_compute_forward(&params, گرہ);
        }

        اگر (state->ith == 0 && cplan->abort_callback &&
                cplan->abort_callback(cplan->abort_callback_ڈیٹا)) {
            atomic_store_explicit(&tp->abort, گرہ_n + 1, memory_order_relaxed);
            tp->ec    = GGML_حالت_ABORTED;
        }

        اگر (گرہ_n + 1 < cgraph->n_nodes) {
            ggml_barrier(state->threadpool);
        }
    }

#اگر_تعریف_شدہ GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p\n", state->ith, (ثابت خالی *)cplan);
#ورنہ
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p last-graph %d\n", state->ith, (ثابت خالی *)cplan, state->آخری_گراف);
#اگر_ختم

    ggml_barrier(state->threadpool);

#اگر_تعریف_شدہ GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(state->ith);
#اگر_ختم

    واپس 0;
}

#اگر_غیر_تعریف_شدہ GGML_USE_OPENMP

// check if thread is ready to proceed (exit from polling or sleeping)
// returns true if loops should exit, sets state->pending to indicate new work
ساکن اندرونی bool ggml_گراف_compute_thread_ready(ساخت ggml_compute_state * state) {
    ساخت ggml_threadpool * threadpool = state->threadpool;

    اگر (state->pending || threadpool->stop || threadpool->pause) { واپس true; }

    // check for new graph/work
    صحیح n_گراف   = atomic_بھرو_explicit(&threadpool->n_گراف, memory_order_relaxed);
    صحیح n_دھاگے = n_گراف & GGML_THREADPOOL_N_دھاگے_نقاب;
    اگر (n_گراف != state->آخری_گراف) {
        state->pending    = (state->ith < n_دھاگے);
        state->آخری_گراف = n_گراف;
        واپس true;
    }

    واپس false;
}

// sync thread state after polling
ساکن اندرونی خالی ggml_گراف_compute_thread_sync(ساخت ggml_compute_state * state) {
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #اگر_تعریف_شدہ GGML_TSAN_ENABLED
    atomic_fetch_جوڑو_explicit(&state->threadpool->n_گراف, 0, memory_order_seq_cst);
    #ورنہ
    atomic_thread_fence(memory_order_seq_cst);
    #اگر_ختم
    غیر_مستعمل(state);
}

ساکن اندرونی bool ggml_گراف_compute_poll_for_work(ساخت ggml_compute_state * state) {
    ساخت ggml_threadpool * threadpool = state->threadpool;

    // This seems to make 0 ... 100 a decent range for polling level across modern processors.
    // Perhaps, we can adjust it dynamically based on load and things.
    ثابت uint64_t n_دور = 1024UL * 128 * threadpool->poll;

    برائے (uint64_t i=0; !ggml_گراف_compute_thread_ready(state) && i < n_دور; i++) {
        // No new work. Keep polling.
        ggml_thread_cpu_relax();
    }

    واپس state->pending;
}

ساکن اندرونی bool ggml_گراف_compute_check_for_work(ساخت ggml_compute_state * state) {
    ساخت ggml_threadpool * threadpool = state->threadpool;

    اگر (ggml_گراف_compute_poll_for_work(state)) {
        ggml_گراف_compute_thread_sync(state);
        واپس state->pending;
    }

    ggml_mutex_تالا_shared(&threadpool->mutex);
    جب (!ggml_گراف_compute_thread_ready(state)) {
        // No new work. Wait for the signal.
        GGML_PRINT_DEBUG("thread #%d waiting for work (sleeping)\n", state->ith);
        ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
    }
    ggml_mutex_unlock_shared(&threadpool->mutex);

    واپس state->pending;
}

ساکن thread_ret_t ggml_گراف_compute_secondary_thread(خالی* ڈیٹا) {
    ساخت ggml_compute_state * state = (ساخت ggml_compute_state *) ڈیٹا;
    ساخت ggml_threadpool * threadpool = state->threadpool;

    ggml_thread_apply_ترجیح(threadpool->prio);
    اگر (ggml_thread_cpumask_is_valid(state->cpumask)) {
        ggml_thread_apply_affinity(state->cpumask);
    }

    جب (true) {
        // Check if we need to sleep
        جب (threadpool->pause) {
            GGML_PRINT_DEBUG("thread #%d inside pause loop\n", state->ith);
            ggml_mutex_تالا_shared(&threadpool->mutex);
            اگر (threadpool->pause) {
                ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
            }
            GGML_PRINT_DEBUG("thread #%d resuming after wait\n", state->ith);
            ggml_mutex_unlock_shared(&threadpool->mutex);
        }

        // This needs to be checked for after the cond_wait
        اگر (threadpool->stop) توڑو;

        // Check if there is new work
        // The main thread is the only one that can dispatch new work

        ggml_گراف_compute_check_for_work(state);
        اگر (state->pending) {
            state->pending = false;
            ggml_گراف_compute_thread(state);
        }
    }

    واپس (thread_ret_t) 0;
}

// Start processing new graph
ساکن خالی ggml_گراف_compute_kickoff(ساخت ggml_threadpool * threadpool, صحیح n_دھاگے)
{
    // Always take the mutex here because the worker threads are doing hybrid poll/wait

    ggml_mutex_تالا(&threadpool->mutex);

    // Update the number of active threads and the graph count
    صحیح n_گراف = atomic_بھرو_explicit(&threadpool->n_گراف, memory_order_relaxed) >> GGML_THREADPOOL_N_دھاگے_بٹ;
    n_گراف = ((n_گراف + 1) << GGML_THREADPOOL_N_دھاگے_بٹ) | (n_دھاگے & GGML_THREADPOOL_N_دھاگے_نقاب);

    GGML_PRINT_DEBUG("compute-kickoff: n_threads %d n_graph %d\n", n_دھاگے, n_گراف);

    // Indicate the graph is ready to be processed
    // We need the full seq-cst fence here because of the polling threads (used in thread_sync)
    atomic_store_explicit(&threadpool->n_گراف, n_گراف, memory_order_seq_cst);

    اگر (threadpool->pause) {
       // Update main thread prio and affinity to match the threadpool settings
       ggml_thread_apply_ترجیح(threadpool->prio);
       اگر (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
           ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
       }

       // resume does cond broadcast
       ggml_threadpool_resume_locked(threadpool);
    } ورنہ {
       ggml_cond_نشر(&threadpool->cond);
    }

    ggml_mutex_unlock(&threadpool->mutex);
}

#اگر_ختم // GGML_USE_OPENMP

ساکن ساخت ggml_threadpool * ggml_threadpool_new_impl(
    ساخت ggml_threadpool_params * tpp,
               ساخت ggml_cgraph * cgraph,
                ساخت ggml_cplan * cplan) {

    ساخت ggml_threadpool * threadpool =
        ggml_aligned_malloc(حجم(ساخت ggml_threadpool));
    {
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->n_گراف          = 0;
        threadpool->n_barrier        = 0;
        threadpool->n_barrier_passed = 0;
        threadpool->current_chunk    = 0;
        threadpool->stop             = false;
        threadpool->pause            = tpp->paused;
        threadpool->abort            = -1;
        threadpool->workers          = تہی;
        threadpool->n_دھاگے        = tpp->n_دھاگے;
        threadpool->poll             = tpp->poll;
        threadpool->prio             = tpp->prio;
        threadpool->ec               = GGML_حالت_SUCCESS;
    }

    // Allocate and init workers state
    ثابت size_t workers_سائز = حجم(ساخت ggml_compute_state) * tpp->n_دھاگے;
    ساخت ggml_compute_state * workers = ggml_aligned_malloc(workers_سائز);

    memset(workers, 0, workers_سائز);
    برائے (صحیح j = 0; j < tpp->n_دھاگے; j++) {
        workers[j].threadpool = threadpool;
        workers[j].ith        = j;
    }

    threadpool->workers = workers;

#اگر_تعریف_شدہ GGML_USE_OPENMP
    int32_t cpumask_iter = 0;

    // Compute CPU masks for each thread
    برائے (صحیح j = 0; j < tpp->n_دھاگے; j++) {
        ggml_thread_cpumask_اگلا(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);
    }
#ورنہ // GGML_USE_OPENMP
    ggml_mutex_init(&threadpool->mutex);
    ggml_cond_init(&threadpool->cond);

    // Spin the threads for all workers, and update CPU placements.
    // Place the main thread last (towards the higher numbered CPU cores).

    int32_t cpumask_iter = 0;

    برائے (صحیح j = 1; j < tpp->n_دھاگے; j++) {
        ggml_thread_cpumask_اگلا(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);

        int32_t rc = ggml_thread_create(&workers[j].thrd, تہی, ggml_گراف_compute_secondary_thread, &workers[j]);
        GGML_ASSERT(rc == 0);
    }

    ggml_thread_cpumask_اگلا(tpp->cpumask, workers[0].cpumask, tpp->strict_cpu, &cpumask_iter);

    اگر (!threadpool->pause) {
        // Update main thread prio and affinity at the start, otherwise we'll do it in resume
        ggml_thread_apply_ترجیح(threadpool->prio);
        اگر (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
            ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
        }
    }
#اگر_ختم // GGML_USE_OPENMP

    واپس threadpool;
}

ساخت ggml_threadpool * ggml_threadpool_new(ساخت ggml_threadpool_params * tpp) {
    واپس ggml_threadpool_new_impl(tpp, تہی, تہی);
}

شمار ggml_حالت ggml_گراف_compute(ساخت ggml_cgraph * cgraph, ساخت ggml_cplan * cplan) {
    ggml_cpu_init();

    GGML_ASSERT(cplan);
    GGML_ASSERT(cplan->n_دھاگے > 0);
    GGML_ASSERT(cplan->work_سائز == 0 || cplan->work_ڈیٹا != تہی);

    صحیح n_دھاگے                               = cplan->n_دھاگے;
    ساخت ggml_threadpool * threadpool = cplan->threadpool;

    bool disposable_threadpool = false;

    اگر (threadpool == تہی) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
        disposable_threadpool = true;

        ساخت ggml_threadpool_params ttp = ggml_threadpool_params_default(n_دھاگے);
        threadpool = ggml_threadpool_new_impl(&ttp, cgraph, cplan);
    } ورنہ {
        // Reset some of the parameters that need resetting
        // No worker threads should be accessing the parameters below at this stage
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->current_chunk    = 0;
        threadpool->abort            = -1;
        threadpool->ec               = GGML_حالت_SUCCESS;
    }

#اگر_تعریف_شدہ GGML_USE_OPENMP
    اگر (n_دھاگے > 1) {
        #ہدایت omp parallel عدد_دھاگے(n_دھاگے)
        {
            #ہدایت omp single
            {
                // update the number of threads from the actual number of threads that we got from OpenMP
                n_دھاگے = omp_get_عدد_دھاگے();
                atomic_store_explicit(&threadpool->n_گراف, n_دھاگے, memory_order_relaxed);
            }

            // Apply thread CPU mask and priority
            صحیح ith = omp_get_thread_عدد();

            ggml_thread_apply_ترجیح(threadpool->prio);
            اگر (ggml_thread_cpumask_is_valid(threadpool->workers[ith].cpumask)) {
                ggml_thread_apply_affinity(threadpool->workers[ith].cpumask);
            }
            ggml_گراف_compute_thread(&threadpool->workers[ith]);
        }
    } ورنہ {
        atomic_store_explicit(&threadpool->n_گراف, 1, memory_order_relaxed);
        ggml_گراف_compute_thread(&threadpool->workers[0]);
    }
#ورنہ
    اگر (n_دھاگے > threadpool->n_دھاگے) {
        GGML_LOG_WARN("cplan requested more threads (%d) than available (%d)\n", n_دھاگے, threadpool->n_دھاگے);
        n_دھاگے = threadpool->n_دھاگے;
    }

    // Kick all threads to start the new graph
    ggml_گراف_compute_kickoff(threadpool, n_دھاگے);

    // This is a work thread too
    ggml_گراف_compute_thread(&threadpool->workers[0]);
#اگر_ختم

    // don't leave affinity set on the main thread
    clear_numa_thread_affinity();

    شمار ggml_حالت ret = threadpool->ec;

    اگر (disposable_threadpool) {
        ggml_threadpool_free(threadpool);
    }

    واپس ret;
}

شمار ggml_حالت ggml_گراف_compute_with_ctx(ساخت ggml_context * ctx, ساخت ggml_cgraph * cgraph, صحیح n_دھاگے) {
    ساخت ggml_cplan cplan = ggml_گراف_plan(cgraph, n_دھاگے, تہی);

    cplan.work_ڈیٹا = (uint8_t *)ggml_new_بفر(ctx, cplan.work_سائز);

    واپس ggml_گراف_compute(cgraph, &cplan);
}

خالی ggml_cpu_fp32_مقصد_fp32(ثابت اعشاری * x, اعشاری * y, int64_t n) {
    حافظہ_نقل(y, x, n * حجم(اعشاری));
}

خالی ggml_cpu_fp32_مقصد_fp16(ثابت اعشاری * x, ggml_fp16_t * y, int64_t n) {
    int64_t i = 0;
#اگر defined(__F16C__)
#اگر defined(__AVX512F__)
    برائے (; i + 15 < n; i += 16) {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        __m256i y_vec = _mm512_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256((__m256i *)(y + i), y_vec);
    }
#اگر_ختم
    برائے (; i + 7 < n; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m128i y_vec = _mm256_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i *)(y + i), y_vec);
    }
    برائے (; i + 3 < n; i += 4) {
        __m128 x_vec = _mm_loadu_ps(x + i);
        __m128i y_vec = _mm_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storel_epi64((__m128i *)(y + i), y_vec);
    }
#وگرنہ_اگر defined(__riscv_zvfh)
    برائے (صحیح vl; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(&x[i], vl);
        vfloat16m1_t vy = __riscv_vfncvt_f_f_w_f16m1(vx, vl);
        __riscv_vse16_v_f16m1((_Float16 *)&y[i], vy, vl);
    }
#وگرنہ_اگر defined(__VXE__) || defined(__VXE2__)
    برائے (; i + 7 < n; i += 8) {
        ثابت uint32x4_t v_yl = __lzs_f32cx4_to_f16(vec_xl(0, x + i + 0));
        ثابت uint32x4_t v_yh = __lzs_f32cx4_to_f16(vec_xl(0, x + i + 4));
        vec_xst(vec_pack(v_yl, v_yh), 0, (uint16_t *)(y + i));
    }
#اگر_ختم
    برائے (; i < n; ++i) {
        y[i] = GGML_CPU_FP32_مقصد_FP16(x[i]);
    }
}

خالی ggml_cpu_fp16_مقصد_fp32(ثابت ggml_fp16_t * x, اعشاری * y, int64_t n) {
    int64_t i = 0;
#اگر defined(__F16C__)
#اگر defined(__AVX512F__)
    برائے (; i + 15 < n; i += 16) {
        __m256i x_vec = _mm256_loadu_si256((ثابت __m256i *)(x + i));
        __m512 y_vec = _mm512_cvtph_ps(x_vec);
        _mm512_storeu_ps(y + i, y_vec);
    }
#اگر_ختم
    برائے (; i + 7 < n; i += 8) {
        __m128i x_vec = _mm_loadu_si128((ثابت __m128i *)(x + i));
        __m256 y_vec = _mm256_cvtph_ps(x_vec);
        _mm256_storeu_ps(y + i, y_vec);
    }
    برائے (; i + 3 < n; i += 4) {
        __m128i x_vec = _mm_loadl_epi64((ثابت __m128i *)(x + i));
        __m128 y_vec = _mm_cvtph_ps(x_vec);
        _mm_storeu_ps(y + i, y_vec);
    }

#وگرنہ_اگر defined(__riscv_v_intrinsic) && defined(__riscv_zvfhmin)
    // calculate step size
    ثابت صحیح epr = __riscv_vsetvlmax_e16m2();
    ثابت صحیح قدم = epr * 2;
    ثابت صحیح np = (n & ~(قدم - 1));

    // unroll by 2
    برائے (; i < np; i += قدم) {
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((ثابت _Float16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vfloat16m2_t ax1 = __riscv_vle16_v_f16m2((ثابت _Float16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvt_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    صحیح vl;
    برائے (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((ثابت _Float16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }

#اگر_ختم

    برائے (; i < n; ++i) {
        y[i] = GGML_CPU_FP16_مقصد_FP32(x[i]);
    }
}

خالی ggml_cpu_fp32_مقصد_bf16(ثابت اعشاری * x, ggml_bf16_t * y, int64_t n) {
    int64_t i = 0;
    برائے (; i < n; ++i) {
        y[i] = GGML_FP32_مقصد_BF16(x[i]);
    }
}

خالی ggml_cpu_fp32_مقصد_i32(ثابت اعشاری * x, int32_t * y, int64_t n) {
    int64_t i = 0;
    برائے (; i < n; ++i) {
        y[i] = x[i];
    }
}

خالی ggml_cpu_bf16_مقصد_fp32(ثابت ggml_bf16_t * x, اعشاری * y, int64_t n) {
    int64_t i = 0;
#اگر defined(__AVX2__)
#اگر defined(__AVX512F__)
    برائے (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i,
                        _mm512_castsi512_ps(
                            _mm512_slli_epi32(
                                _mm512_cvtepu16_epi32(
                                    _mm256_loadu_si256(
                                        (ثابت __m256i *)(x + i))),
                                16)));
    }
#اگر_ختم
    برائے (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i,
                        _mm256_castsi256_ps(
                            _mm256_slli_epi32(
                                _mm256_cvtepu16_epi32(
                                    _mm_loadu_si128(
                                        (ثابت __m128i *)(x + i))),
                                16)));
    }
#وگرنہ_اگر defined(__riscv_v_intrinsic) && defined(__riscv_zvfbfmin)
    // calculate step size
    ثابت صحیح epr = __riscv_vsetvlmax_e16m2();
    ثابت صحیح قدم = epr * 2;
    ثابت صحیح np = (n & ~(قدم - 1));

    // unroll by 2
    برائے (; i < np; i += قدم) {
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((ثابت __bf16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vbfloat16m2_t ax1 = __riscv_vle16_v_bf16m2((ثابت __bf16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    صحیح vl;
    برائے (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((ثابت __bf16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }
#اگر_ختم
    برائے (; i < n; i++) {
        y[i] = GGML_BF16_مقصد_FP32(x[i]);
    }
}

صحیح ggml_cpu_has_avx(خالی) {
#اگر defined(__AVX__)
    واپس 1;
#ورنہ
    واپس 0;
#اگر_ختم
}

صحیح ggml_cpu_has_avx_vnni(خالی) {
#اگر defined(__AVXVNNI__)
    واپس 1;
#ورنہ
    واپس 0;
#اگر_ختم
}

صحیح ggml_cpu_has_avx2(خالی) {
#اگر defined(__AVX2__)
    واپس 1;
#ورنہ
    واپس 0;
#اگر_ختم
}

صحیح ggml_cpu_has_avx512(خالی) {
#اگر defined(__AVX512F__)
    واپس 1;
#ورنہ
    واپس 0;
#اگر_ختم
}

صحیح ggml_cpu_has_avx512_vbmi(خالی) {
#اگر defined(__AVX512VBMI__)
    واپس 1;
#ورنہ
    واپس 0;
#اگر_ختم
}

صحیح ggml_cpu_has_avx512_vnni(خالی) {
#اگر defined(__AVX512VNNI__)
    واپس 1;
#ورنہ
    واپس 0;
#اگر_ختم
}

صحیح ggml_cpu_has_avx512_bf16(خالی) {
#اگر defined(__AVX512BF16__)
    واپس 1;
#ورنہ
    واپس 0;
#اگر_ختم
}

صحیح ggml_cpu_has_amx_int8(خالی) {
#اگر defined(__AMX_INT8__)
    واپس 1;
#ورنہ
    واپس 0;
#اگر_ختم
}

صحیح ggml_cpu_has_bmi2(خالی) {
#اگر defined(__BMI2__)
    واپس 1;
#ورنہ
    واپس 0;
#اگر_ختم
}

صحیح ggml_cpu_has_fma(خالی) {
#اگر defined(__FMA__)
    واپس 1;
#ورنہ
    واپس 0;
#اگر_ختم
}

صحیح ggml_cpu_has_arm_fma(خالی) {
#اگر defined(__ARM_FEATURE_FMA)
    واپس 1;
#ورنہ
    واپس 0;
#اگر_ختم
}

صحیح ggml_cpu_has_riscv_v(خالی) {
#اگر defined(__riscv_v_intrinsic)
    واپس 1;
#ورنہ
    واپس 0;
#اگر_ختم
}

صحیح ggml_cpu_get_rvv_vlen(خالی) {
#اگر defined(__riscv) && defined(__riscv_v_intrinsic)
    واپس ggml_riscv_arch_features.rvv_vlen;
#ورنہ
    واپس 0;
#اگر_ختم
}

صحیح ggml_cpu_has_f16c(خالی) {
#اگر defined(__F16C__)
    واپس 1;
#ورنہ
    واپس 0;
#اگر_ختم
}

صحیح ggml_cpu_has_fp16_va(خالی) {
#اگر defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    واپس 1;
#ورنہ
    واپس 0;
#اگر_ختم
}

صحیح ggml_cpu_has_wasm_simd(خالی) {
#اگر defined(__wasm_simd128__)
    واپس 1;
#ورنہ
    واپس 0;
#اگر_ختم
}

صحیح ggml_cpu_has_llamafile(خالی) {
#اگر defined(GGML_USE_LLAMAFILE)
    واپس 1;
#ورنہ
    واپس 0;
#اگر_ختم
}

صحیح ggml_cpu_has_sse3(خالی) {
#اگر defined(__SSE3__)
    واپس 1;
#ورنہ
    واپس 0;
#اگر_ختم
}

صحیح ggml_cpu_has_ssse3(خالی) {
#اگر defined(__SSSE3__)
    واپس 1;
#ورنہ
    واپس 0;
#اگر_ختم
}

صحیح ggml_cpu_has_vsx(خالی) {
#اگر defined(__POWER9_VECTOR__)
    واپس 1;
#ورنہ
    واپس 0;
#اگر_ختم
}

صحیح ggml_cpu_has_vxe(خالی) {
#اگر defined(__VXE__) || defined(__VXE2__)
    واپس 1;
#ورنہ
    واپس 0;
#اگر_ختم
}

صحیح ggml_cpu_has_neon(خالی) {
#اگر defined(__ARM_ARCH) && defined(__ARM_NEON)
    واپس 1;
#ورنہ
    واپس 0;
#اگر_ختم
}

صحیح ggml_cpu_has_dotprod(خالی) {
#اگر defined(__ARM_ARCH) && defined(__ARM_FEATURE_DOTPROD)
    واپس 1;
#ورنہ
    واپس 0;
#اگر_ختم
}

صحیح ggml_cpu_has_sve(خالی) {
#اگر defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    واپس 1;
#ورنہ
    واپس 0;
#اگر_ختم
}

صحیح ggml_cpu_has_matmul_int8(خالی) {
#اگر defined(__ARM_ARCH) && defined(__ARM_FEATURE_MATMUL_INT8)
    واپس 1;
#ورنہ
    واپس 0;
#اگر_ختم
}

صحیح ggml_cpu_get_sve_cnt(خالی) {
#اگر defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    واپس ggml_arm_arch_features.sve_cnt;
#ورنہ
    واپس 0;
#اگر_ختم
}

صحیح ggml_cpu_has_sme(خالی) {
#اگر defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME)
    واپس 1;
#ورنہ
    واپس 0;
#اگر_ختم
}

صحیح ggml_cpu_has_sme2(خالی) {
#اگر defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME2)
    واپس 1;
#ورنہ
    واپس 0;
#اگر_ختم
}

خالی ggml_cpu_init(خالی) {
    // needed to initialize ggml_time
    {
        ساخت ggml_init_params params = { 0, تہی, false };
        ساخت ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    ggml_critical_section_آغاز();

    ساکن bool is_پہلا_call = true;

    اگر (is_پہلا_call) {
        // initialize GELU, Quick GELU, SILU and EXP F32 tables
        {
            ثابت uint64_t t_آغاز = ggml_وقت_us(); غیر_مستعمل(t_آغاز);

            برائے (صحیح i = 0; i < (1 << 16); ++i) {
                اتحاد {
                    uint16_t u16;
                    ggml_fp16_t fp16;
                } u = {i};
                اعشاری f = GGML_COMPUTE_FP16_مقصد_FP32(u.fp16);
                ggml_جدول_f32_f16[i] = f;
                ggml_جدول_gelu_f16[i] = GGML_CPU_FP32_مقصد_FP16(ggml_gelu_f32(f));
                ggml_جدول_gelu_تیز_f16[i] = GGML_CPU_FP32_مقصد_FP16(ggml_gelu_تیز_f32(f));
            }

            // initialize E8M0 half table (256 entries)
            برائے (صحیح i = 0; i < (1 << 8); ++i) {
                ggml_جدول_f32_e8m0_half[i] = GGML_E8_M0_مقصد_FP32_HALF(i);
            }

            // initialize UE4M3 table (256 entries)
            برائے (صحیح i = 0; i < (1 << 8); ++i) {
                ggml_جدول_f32_ue4m3[i] = ggml_ue4m3_مقصد_fp32(i);
            }

            ثابت uint64_t t_انتہا = ggml_وقت_us(); غیر_مستعمل(t_انتہا);

            GGML_PRINT_DEBUG("%s: GELU, Quick GELU, SILU and EXP tables initialized in %f ms\n", __func__, (t_انتہا - t_آغاز)/1000.0);

#اگر_تعریف_شدہ GGML_USE_OPENMP
            //if (!getenv("OMP_WAIT_POLICY")) {
            //    // set the wait policy to active, so that OpenMP threads don't sleep
            //    setenv("OMP_WAIT_POLICY", "active", 0)
            //}

            اگر (!getenv("KMP_BLOCKTIME")) {
                // set the time to wait before sleeping a thread
                // this is less aggressive than setting the wait policy to active, but should achieve similar results in most cases
#اگر_تعریف_شدہ _WIN32
                _putenv_s("KMP_BLOCKTIME", "200"); // 200ms
#ورنہ
                setenv("KMP_BLOCKTIME", "200", 0); // 200ms
#اگر_ختم
            }
#اگر_ختم
        }

#اگر defined(__ARM_ARCH)
        ggml_init_arm_arch_features();
#اگر_ختم

#اگر defined(__riscv)
        ggml_init_riscv_arch_features();
#اگر_ختم

        {
            ثابت حرف * env = getenv("GGML_CPU_DISABLE_FUSION");
            ggml_cpu_disable_fusion = (env != تہی && atoi(env) == 1);
        }

        is_پہلا_call = false;
    }

    ggml_critical_section_انتہا();
}
