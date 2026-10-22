#define _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#define _USE_MATH_DEFINES // For M_PI on MSVC

#ଅନ୍ତର୍ଭାଵ "ggml-backend-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-backend.h"
#ଅନ୍ତର୍ଭାଵ "traits.h"
#ଅନ୍ତର୍ଭାଵ "ggml-cpu-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-impl.h"
#ଅନ୍ତର୍ଭାଵ "quants.h"
#ଅନ୍ତର୍ଭାଵ "ggml-threading.h"
#ଅନ୍ତର୍ଭାଵ "unary-ops.h"
#ଅନ୍ତର୍ଭାଵ "binary-ops.h"
#ଅନ୍ତର୍ଭାଵ "vec.h"
#ଅନ୍ତର୍ଭାଵ "ops.h"
#ଅନ୍ତର୍ଭାଵ "ggml.h"
#ଅନ୍ତର୍ଭାଵ "common.h"
#ଅନ୍ତର୍ଭାଵ "tiled/tiled.h"

#ଯଦି defined(_MSC_VER) || defined(__MINGW32__)
#ଅନ୍ତର୍ଭାଵ <malloc.h> // using malloc.h with MSC/MINGW
#ନଚେତଯଦି !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#ଅନ୍ତର୍ଭାଵ <alloca.h>
#endif

#ଅନ୍ତର୍ଭାଵ <assert.h>
#ଅନ୍ତର୍ଭାଵ <errno.h>
#ଅନ୍ତର୍ଭାଵ <time.h>
#ଅନ୍ତର୍ଭାଵ <math.h>
#ଅନ୍ତର୍ଭାଵ <stdlib.h>
#ଅନ୍ତର୍ଭାଵ <string.h>
#ଅନ୍ତର୍ଭାଵ <stdint.h>
#ଅନ୍ତର୍ଭାଵ <inttypes.h>
#ଅନ୍ତର୍ଭାଵ <stdio.h>
#ଅନ୍ତର୍ଭାଵ <float.h>
#ଅନ୍ତର୍ଭାଵ <limits.h>
#ଅନ୍ତର୍ଭାଵ <stdarg.h>
#ଅନ୍ତର୍ଭାଵ <signal.h>
#ଯଦି defined(__gnu_linux__)
#ଅନ୍ତର୍ଭାଵ <syscall.h>
#endif

#ifdef GGML_USE_OPENMP
#ଅନ୍ତର୍ଭାଵ <omp.h>
#endif

#ଯଦି defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_MATMUL_INT8)
#undef GGML_USE_LLAMAFILE
#endif

#ifdef GGML_USE_LLAMAFILE
#ଅନ୍ତର୍ଭାଵ "llamafile/sgemm.h"
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
#    include "spacemit/ime.h"
#endif

// Note: once we move threading into a separate C++ file
// will use std::hardware_destructive_interference_size instead of hardcoding it here
// and we'll use C++ attribute syntax.
#define GGML_CACHE_ଧାଡ଼ି  64

#ଯଦି defined(__clang__) || defined(__GNUC__)
#define GGML_CACHE_ALIGN __attribute__((aligned(GGML_CACHE_ଧାଡ଼ି)))
#endif

#ଯଦି defined(__has_feature)
#ଯଦି __has_feature(thread_sanitizer)
#define GGML_TSAN_ENABLED 1
#endif
#ଅନ୍ୟଥା  // __has_feature
#ଯଦି defined(__SANITIZE_THREAD__)
#define GGML_TSAN_ENABLED 1
#endif
#endif // __has_feature

#define UNUSED GGML_UNUSED
#define SWAP(x, y, T) କର { T SWAP = x; (x) = y; (y) = SWAP; } ଯେତେବେଳେ (0)

// precomputed f32 table for f16 (256 KB) (simd-mappings.h)
ଦଶମିକ ggml_table_f32_f16[1 << 16];

// precomputed f32 table for e8m0 half (1 KB) (simd-mappings.h)
ଦଶମିକ ggml_table_f32_e8m0_half[1 << 8];

// precomputed f32 table for ue4m3 (1 KB) (simd-mappings.h)
ଦଶମିକ ggml_table_f32_ue4m3[1 << 8];

#ଯଦି defined(__ARM_ARCH)
ସଂରଚନା ggml_arm_arch_features_type {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା sve_cnt;
} ggml_arm_arch_features = { 0 };
#endif

#ଯଦି defined(__riscv)
ସଂରଚନା ggml_riscv_arch_features_type {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା rvv_vlen;
} ggml_riscv_arch_features = { 0 };
#endif

#ଯଦି defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#ଅନ୍ତର୍ଭାଵ <windows.h>

#ଯଦି defined(_MSC_VER) && !defined(__clang__)
#define GGML_CACHE_ALIGN __declspec(align(GGML_CACHE_ଧାଡ଼ି))

ପ୍ରକାରବ୍ୟାଖ୍ୟା ଅସ୍ଥିର LONG atomic_int;
ପ୍ରକାରବ୍ୟାଖ୍ୟା atomic_int atomic_bool;
ପ୍ରକାରବ୍ୟାଖ୍ୟା atomic_int atomic_flag;

#define ATOMIC_FLAG_INIT 0

ପ୍ରକାରବ୍ୟାଖ୍ୟା ଗଣନା {
    memory_order_relaxed,
    memory_order_consume,
    memory_order_acquire,
    memory_order_release,
    memory_order_acq_rel,
    memory_order_seq_cst
} memory_order;

ସ୍ଥିର ଶୂନ୍ୟ atomic_store(atomic_int * ptr, LONG val) {
    InterlockedExchange(ptr, val);
}
ସ୍ଥିର ଶୂନ୍ୟ atomic_store_explicit(atomic_int * ptr, LONG val, memory_order mo) {
    // TODO: add support for explicit memory order
    InterlockedExchange(ptr, val);
}
ସ୍ଥିର LONG atomic_load(atomic_int * ptr) {
    ଫେରନ୍ତୁ InterlockedCompareExchange(ptr, 0, 0);
}
ସ୍ଥିର LONG atomic_load_explicit(atomic_int * ptr, memory_order mo) {
    // TODO: add support for explicit memory order
    ଫେରନ୍ତୁ InterlockedCompareExchange(ptr, 0, 0);
}
ସ୍ଥିର LONG atomic_fetch_add(atomic_int * ptr, LONG inc) {
    ଫେରନ୍ତୁ InterlockedExchangeAdd(ptr, inc);
}
ସ୍ଥିର LONG atomic_fetch_add_explicit(atomic_int * ptr, LONG inc, memory_order mo) {
    // TODO: add support for explicit memory order
    ଫେରନ୍ତୁ InterlockedExchangeAdd(ptr, inc);
}
ସ୍ଥିର atomic_bool atomic_flag_test_and_set(atomic_flag * ptr) {
    ଫେରନ୍ତୁ InterlockedExchange(ptr, 1);
}
ସ୍ଥିର ଶୂନ୍ୟ atomic_flag_clear(atomic_flag * ptr) {
    InterlockedExchange(ptr, 0);
}
ସ୍ଥିର ଶୂନ୍ୟ atomic_thread_fence(memory_order mo) {
    MemoryBarrier();
}
#ଅନ୍ୟଥା // clang
#ଅନ୍ତର୍ଭାଵ <stdatomic.h>
#endif

ପ୍ରକାରବ୍ୟାଖ୍ୟା HANDLE pthread_t;

ପ୍ରକାରବ୍ୟାଖ୍ୟା DWORD thread_ret_t;
ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା pthread_create(pthread_t * out, ଶୂନ୍ୟ * unused, thread_ret_t(*func)(ଶୂନ୍ୟ *), ଶୂନ୍ୟ * arg) {
    (ଶୂନ୍ୟ) unused;
    HANDLE handle = CreateThread(ଖାଲି, 0, (LPTHREAD_START_ROUTINE) func, arg, 0, ଖାଲି);
    ଯଦି (handle == ଖାଲି)
    {
        ଫେରନ୍ତୁ EAGAIN;
    }

    *out = handle;
    ଫେରନ୍ତୁ 0;
}

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା pthread_join(pthread_t thread, ଶୂନ୍ୟ * unused) {
    (ଶୂନ୍ୟ) unused;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ret = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    ଫେରନ୍ତୁ ret;
}

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା sched_yield (ଶୂନ୍ୟ) {
    Sleep (0);
    ଫେରନ୍ତୁ 0;
}
#ଅନ୍ୟଥା

#ଅନ୍ତର୍ଭାଵ <pthread.h>
#ଅନ୍ତର୍ଭାଵ <stdatomic.h>
#ଅନ୍ତର୍ଭାଵ <sched.h>
#ଯଦି defined(__FreeBSD__)
#ଅନ୍ତର୍ଭାଵ <pthread_np.h>
#endif

ପ୍ରକାରବ୍ୟାଖ୍ୟା ଶୂନ୍ୟ * thread_ret_t;

#ଅନ୍ତର୍ଭାଵ <sys/types.h>
#ଅନ୍ତର୍ଭାଵ <sys/stat.h>
#ଅନ୍ତର୍ଭାଵ <unistd.h>

#endif

ପ୍ରକାରବ୍ୟାଖ୍ୟା pthread_t ggml_thread_t;

#define GGML_THREADPOOL_N_THREADS_MASK (0xffffU)
#define GGML_THREADPOOL_N_THREADS_BITS (16)

#ଯଦି defined(__APPLE__)
#ଅନ୍ତର୍ଭାଵ <unistd.h>
#ଅନ୍ତର୍ଭାଵ <mach/mach.h>
#ଅନ୍ତର୍ଭାଵ <TargetConditionals.h>
#endif

ସ୍ଥିର ଅଚଳ ସଂରଚନା ggml_type_traits_cpu type_traits_cpu[GGML_TYPE_ଗଣନା] = {
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
#ଯଦି defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#ଅନ୍ୟଥା
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q4_1] = {
        .from_float               = quantize_row_q4_1,
        .vec_dot                  = ggml_vec_dot_q4_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
#ଯଦି defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#ଅନ୍ୟଥା
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
#ଯଦି defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#ଅନ୍ୟଥା
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
#ଯଦି defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#ଅନ୍ୟଥା
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
#ଯଦି defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#ଅନ୍ୟଥା
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_IQ2_XXS] = {
        .from_float               = ଖାଲି,
        .vec_dot                  = ggml_vec_dot_iq2_xxs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_XS] = {
        .from_float               = ଖାଲି,
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
        .from_float               = ଖାଲି,
        .vec_dot                  = ggml_vec_dot_iq1_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_M] = {
        .from_float               = ଖାଲି,
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

ଅଚଳ ସଂରଚନା ggml_type_traits_cpu * ggml_get_type_traits_cpu(ଗଣନା ggml_type type) {
    ଫେରନ୍ତୁ &type_traits_cpu[type];
}

//
// Threading defs
//

ପ୍ରକାରବ୍ୟାଖ୍ୟା pthread_t          ggml_thread_t;

#ଯଦି defined(_WIN32)

ପ୍ରକାରବ୍ୟାଖ୍ୟା CONDITION_VARIABLE ggml_cond_t;
ପ୍ରକାରବ୍ୟାଖ୍ୟା SRWLOCK            ggml_mutex_t;

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

#ଅନ୍ୟଥା

ପ୍ରକାରବ୍ୟାଖ୍ୟା pthread_cond_t     ggml_cond_t;
ପ୍ରକାରବ୍ୟାଖ୍ୟା pthread_mutex_t    ggml_mutex_t;

#define ggml_mutex_init(m)          pthread_mutex_init(m, ଖାଲି)
#define ggml_mutex_destroy(m)       pthread_mutex_destroy(m)
#define ggml_mutex_lock(m)          pthread_mutex_lock(m)
#define ggml_mutex_unlock(m)        pthread_mutex_unlock(m)
#define ggml_mutex_lock_shared(m)   pthread_mutex_lock(m)
#define ggml_mutex_unlock_shared(m) pthread_mutex_unlock(m)

#define ggml_lock_init(x)    UNUSED(x)
#define ggml_lock_destroy(x) UNUSED(x)
#ଯଦି defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64) && !defined(_M_ARM64EC))
#define ggml_lock_lock(x)    _mm_pause()
#ଅନ୍ୟଥା
#define ggml_lock_lock(x)    UNUSED(x)
#endif
#define ggml_lock_unlock(x)  UNUSED(x)

#define GGML_LOCK_INITIALIZER 0
#define ggml_cond_init(c)      pthread_cond_init(c, ଖାଲି)
#define ggml_cond_destroy(c)   pthread_cond_destroy(c)
#define ggml_cond_wait(c, m)   pthread_cond_wait(c, m)
#define ggml_cond_broadcast(c) pthread_cond_broadcast(c)

#define ggml_thread_create pthread_create
#define ggml_thread_join   pthread_join

#endif

// Threadpool def
ସଂରଚନା ggml_threadpool {
    ggml_mutex_t mutex;       // mutex for cond.var
    ggml_cond_t  cond;        // cond.var for waiting for new work

    ସଂରଚନା ggml_cgraph * cgraph;
    ସଂରଚନା ggml_cplan  * cplan;

    // synchronization primitives
    atomic_int n_graph;       // updated when there is work to be done (i.e each graph) holds graph and active thread counts.
    atomic_int GGML_CACHE_ALIGN n_barrier;
    atomic_int GGML_CACHE_ALIGN n_barrier_passed;
    atomic_int GGML_CACHE_ALIGN current_chunk; // currently processing chunk during Mat_Mul, shared between all the threads.

    // these are atomic as an annotation for thread-sanitizer
    atomic_bool stop;         // Used for stopping the threadpool altogether
    atomic_bool pause;        // Used for pausing the threadpool or individual threads
    atomic_int  abort;        // Used for aborting processing of a graph

    ସଂରଚନା ggml_compute_state * workers;   // per thread state
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା          n_threads;   // Number of threads in the pool
    int32_t      prio;        // Scheduling priority
    uint32_t     poll;        // Polling level (0 - no polling)

    ଗଣନା ggml_status ec;
};

// Per-thread state
ସଂରଚନା ggml_compute_state {
#ifndef GGML_USE_OPENMP
    ggml_thread_t thrd;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା  last_graph;
    bool pending;
#endif
    bool cpumask[GGML_MAX_N_THREADS];
    ସଂରଚନା ggml_threadpool * threadpool;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith;
};

// Helpers for polling loops
#ଯଦି defined(__aarch64__) && ( defined(__clang__) || defined(__GNUC__) )
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_thread_cpu_relax(ଶୂନ୍ୟ) {
    __asm__ ଅସ୍ଥିର("yield" ::: "memory");
}
#ନଚେତଯଦି defined(__x86_64__)
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_thread_cpu_relax(ଶୂନ୍ୟ) {
    _mm_pause();
}
#ନଚେତଯଦି defined(__riscv)
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_thread_cpu_relax(ଶୂନ୍ୟ) {
    #ifdef __riscv_zihintpause
        __asm__ __volatile__ ("pause");
    #ଅନ୍ୟଥା
        /* Encoding of the pause instruction */
        __asm__ __volatile__ (".4byte 0x100000F");
    #endif
}
#ଅନ୍ୟଥା
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_thread_cpu_relax(ଶୂନ୍ୟ) {;}
#endif

//
// NUMA support
//

#define GGML_NUMA_MAX_NODES 8
#define GGML_NUMA_MAX_CPUS 512

ସଂରଚନା ggml_numa_ଗଣ୍ଠି {
    uint32_t cpus[GGML_NUMA_MAX_CPUS]; // hardware threads on this node
    uint32_t n_cpus;
};

ସଂରଚନା ggml_numa_nodes {
    ଗଣନା ggml_numa_strategy numa_strategy;
    ସଂରଚନା ggml_numa_ଗଣ୍ଠି nodes[GGML_NUMA_MAX_NODES];
    uint32_t n_nodes;
    uint32_t total_cpus; // hardware threads on system
    uint32_t current_ଗଣ୍ଠି; // node on which main process is execting
#ଯଦି defined(__gnu_linux__)
    cpu_set_t cpuset; // cpuset from numactl
#ଅନ୍ୟଥା
    uint32_t cpuset; // no NUMA support outside of Linux at this time. Use a portable datatype
#endif
};

//
// ggml state
//

ସଂରଚନା ggml_state {
    ସଂରଚନା ggml_numa_nodes numa;
};

ସ୍ଥିର ସଂରଚନା ggml_state g_state = {0};

ଶୂନ୍ୟ ggml_barrier(ସଂରଚନା ggml_threadpool * tp) {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_threads = atomic_load_explicit(&tp->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK;
    ଯଦି (n_threads == 1) {
        ଫେରନ୍ତୁ;
    }

#ifdef GGML_USE_OPENMP
    #pragma omp barrier
#ଅନ୍ୟଥା
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_passed = atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed);

    // enter barrier (full seq-cst fence)
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_barrier = atomic_fetch_add_explicit(&tp->n_barrier, 1, memory_order_seq_cst);

    ଯଦି (n_barrier == (n_threads - 1)) {
        // last thread
        atomic_store_explicit(&tp->n_barrier, 0, memory_order_relaxed);

        // exit barrier (full seq-cst fence)
        atomic_fetch_add_explicit(&tp->n_barrier_passed, 1, memory_order_seq_cst);
        ଫେରନ୍ତୁ;
    }

    // wait for other threads
    ଯେତେବେଳେ (atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed) == n_passed) {
        ggml_thread_cpu_relax();
    }

    // exit barrier (full seq-cst fence)
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&tp->n_barrier_passed, 0, memory_order_seq_cst);
    #ଅନ୍ୟଥା
    atomic_thread_fence(memory_order_seq_cst);
    #endif
#endif
}

ଶୂନ୍ୟ ggml_threadpool_chunk_set(ସଂରଚନା ggml_threadpool * tp, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ମୂଲ୍ୟ) {
    atomic_store_explicit(&tp->current_chunk, ମୂଲ୍ୟ, memory_order_relaxed);
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_threadpool_chunk_add(ସଂରଚନା ggml_threadpool * tp, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ମୂଲ୍ୟ) {
    ଫେରନ୍ତୁ atomic_fetch_add_explicit(&tp->current_chunk, ମୂଲ୍ୟ, memory_order_relaxed);
}

#ଯଦି defined(__gnu_linux__)
ସ୍ଥିର cpu_set_t ggml_get_numa_affinity(ଶୂନ୍ୟ) {
    cpu_set_t cpuset;
    pthread_t thread;
    thread = pthread_self();
    CPU_ZERO(&cpuset);
    pthread_getaffinity_np(thread, ଆକାର(cpu_set_t), &cpuset);
    ଫେରନ୍ତୁ cpuset;
}
#ଅନ୍ୟଥା
ସ୍ଥିର uint32_t ggml_get_numa_affinity(ଶୂନ୍ୟ) {
    ଫେରନ୍ତୁ 0; // no NUMA support
}
#endif

ଶୂନ୍ୟ ggml_numa_init(ଗଣନା ggml_numa_strategy numa_flag) {
    ଯଦି (g_state.numa.n_nodes > 0) {
        fprintf(stderr, "ggml_numa_init: NUMA already initialized\n");

        ଫେରନ୍ତୁ;
    }

#ଯଦି defined(__gnu_linux__)
    ସଂରଚନା stat st;
    ଅକ୍ଷର path[256];
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା rv;

    // set numa scheme
    g_state.numa.numa_strategy = numa_flag;

    GGML_PRINT_DEBUG("numa strategy %u\n",g_state.numa.numa_strategy);

    g_state.numa.cpuset = ggml_get_numa_affinity();

    // enumerate nodes
    ଯେତେବେଳେ (g_state.numa.n_nodes < GGML_NUMA_MAX_NODES) {
        rv = snprintf(path, ଆକାର(path), "/sys/devices/system/node/node%u", g_state.numa.n_nodes);
        GGML_ASSERT(rv > 0 && (unsigned)rv < ଆକାର(path));
        ଯଦି (stat(path, &st) != 0) { ଭାଙ୍ଗ; }
        ++g_state.numa.n_nodes;
    }

    // enumerate CPUs
    ଯେତେବେଳେ (g_state.numa.total_cpus < GGML_NUMA_MAX_CPUS) {
        rv = snprintf(path, ଆକାର(path), "/sys/devices/system/cpu/cpu%u", g_state.numa.total_cpus);
        GGML_ASSERT(rv > 0 && (unsigned)rv < ଆକାର(path));
        ଯଦି (stat(path, &st) != 0) { ଭାଙ୍ଗ; }
        ++g_state.numa.total_cpus;
    }

    GGML_PRINT_DEBUG("found %u numa nodes, %u CPUs\n", g_state.numa.n_nodes, g_state.numa.total_cpus);

    // figure out which node we're on
    uint current_cpu;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା getcpu_ret = 0;
#ଯଦି __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ > 33) || defined(__COSMOPOLITAN__)
    getcpu_ret = getcpu(&current_cpu, &g_state.numa.current_ଗଣ୍ଠି);
#ଅନ୍ୟଥା
    // old glibc doesn't have a wrapper for this call. Fall back on direct syscall
#   ଯଦି !defined(SYS_getcpu) && defined(SYS_get_cpu)
#       define SYS_getcpu SYS_get_cpu // some older glibc versions use this name
#   endif
    getcpu_ret = syscall(SYS_getcpu, &current_cpu, &g_state.numa.current_ଗଣ୍ଠି);
#endif

    ଯଦି (g_state.numa.n_nodes < 1 || g_state.numa.total_cpus < 1 || getcpu_ret != 0) {
        g_state.numa.n_nodes = 0;
        ଫେରନ୍ତୁ;
    }

    GGML_PRINT_DEBUG("found our process on numa node %u, CPU %u\n", g_state.numa.current_ଗଣ୍ଠି, current_cpu);

    ପାଇଁ (uint32_t n = 0; n < g_state.numa.n_nodes; ++n) {
        ସଂରଚନା ggml_numa_ଗଣ୍ଠି * ଗଣ୍ଠି = &g_state.numa.nodes[n];
        GGML_PRINT_DEBUG("CPUs on node %u:", n);
        ଗଣ୍ଠି->n_cpus = 0;
        ପାଇଁ (uint32_t c = 0; c < g_state.numa.total_cpus; ++c) {
            rv = snprintf(path, ଆକାର(path), "/sys/devices/system/node/node%u/cpu%u", n, c);
            GGML_ASSERT(rv > 0 && (unsigned)rv < ଆକାର(path));
            ଯଦି (stat(path, &st) == 0) {
                ଗଣ୍ଠି->cpus[ଗଣ୍ଠି->n_cpus++] = c;
                GGML_PRINT_DEBUG(" %u", c);
            }
        }
        GGML_PRINT_DEBUG("\n");
    }

    ଯଦି (ggml_is_numa()) {
        FILE *fptr = fopen("/proc/sys/kernel/numa_balancing", "r");
        ଯଦି (fptr != ଖାଲି) {
            ଅକ୍ଷର buf[42];
            ଯଦି (fgets(buf, ଆକାର(buf), fptr) && strncmp(buf, "0\n", ଆକାର(buf)) != 0) {
                GGML_LOG_WARN("/proc/sys/kernel/numa_balancing is enabled, this has been observed to impair performance\n");
            }
            fclose(fptr);
        }
    }
#ଅନ୍ୟଥା
    UNUSED(numa_flag);
    // TODO
#endif
}

bool ggml_is_numa(ଶୂନ୍ୟ) {
    ଫେରନ୍ତୁ g_state.numa.n_nodes > 1;
}

#ଯଦି defined(__ARM_ARCH)
#ଯଦି defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
#ଅନ୍ତର୍ଭାଵ <arm_sve.h>
ସ୍ଥିର ଶୂନ୍ୟ ggml_init_arm_arch_features(ଶୂନ୍ୟ) {
    ggml_arm_arch_features.sve_cnt = svcntb();
}
#ଅନ୍ୟଥା
ସ୍ଥିର ଶୂନ୍ୟ ggml_init_arm_arch_features(ଶୂନ୍ୟ) {}
#endif
#endif // __ARM_ARCH

#ଯଦି defined(__riscv) && defined(__riscv_v_intrinsic)
#ଅନ୍ତର୍ଭାଵ <riscv_vector.h>
ସ୍ଥିର ଶୂନ୍ୟ ggml_init_riscv_arch_features(ଶୂନ୍ୟ) {
    ggml_riscv_arch_features.rvv_vlen = __riscv_vlenb();
}
#ଅନ୍ୟଥା
ସ୍ଥିର ଶୂନ୍ୟ ggml_init_riscv_arch_features(ଶୂନ୍ୟ) {}
#endif

ସଂରଚନା ggml_tensor * ggml_new_i32(ସଂରଚନା ggml_context * ctx, int32_t ମୂଲ୍ୟ) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    ସଂରଚନା ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

    ggml_set_i32(result, ମୂଲ୍ୟ);

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_new_f32(ସଂରଚନା ggml_context * ctx, ଦଶମିକ ମୂଲ୍ୟ) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    ସଂରଚନା ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    ggml_set_f32(result, ମୂଲ୍ୟ);

    ଫେରନ୍ତୁ result;
}

ସଂରଚନା ggml_tensor * ggml_set_i32 (ସଂରଚନା ggml_tensor * tensor, int32_t ମୂଲ୍ୟ) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n     = ggml_nrows(tensor);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nc    = tensor->ne[0];
    ଅଚଳ size_t n1 = tensor->nb[1];

    ଅକ୍ଷର * ଅଚଳ data = tensor->data;

    ବାଛ (tensor->type) {
        ସ୍ଥିତି GGML_TYPE_I8:
            {
                ପ୍ରତିଜ୍ଞା(tensor->nb[0] == ଆକାର(int8_t));
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), ମୂଲ୍ୟ);
                }
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_I16:
            {
                ପ୍ରତିଜ୍ଞା(tensor->nb[0] == ଆକାର(int16_t));
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), ମୂଲ୍ୟ);
                }
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_I32:
            {
                ପ୍ରତିଜ୍ଞା(tensor->nb[0] == ଆକାର(int32_t));
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), ମୂଲ୍ୟ);
                }
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F16:
            {
                ପ୍ରତିଜ୍ଞା(tensor->nb[0] == ଆକାର(ggml_fp16_t));
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), GGML_CPU_FP32_TO_FP16(ମୂଲ୍ୟ));
                }
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_BF16:
            {
                ପ୍ରତିଜ୍ଞା(tensor->nb[0] == ଆକାର(ggml_fp16_t));
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; i++) {
                    ggml_vec_set_bf16(nc, (ggml_bf16_t *)(data + i*n1), GGML_FP32_TO_BF16(ମୂଲ୍ୟ));
                }
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ପ୍ରତିଜ୍ଞା(tensor->nb[0] == ଆକାର(ଦଶମିକ));
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (ଦଶମିକ *)(data + i*n1), ମୂଲ୍ୟ);
                }
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }

    ଫେରନ୍ତୁ tensor;
}

ସଂରଚନା ggml_tensor * ggml_set_f32(ସଂରଚନା ggml_tensor * tensor, ଦଶମିକ ମୂଲ୍ୟ) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n     = ggml_nrows(tensor);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nc    = tensor->ne[0];
    ଅଚଳ size_t n1 = tensor->nb[1];

    ଅକ୍ଷର * ଅଚଳ data = tensor->data;

    ବାଛ (tensor->type) {
        ସ୍ଥିତି GGML_TYPE_I8:
            {
                ପ୍ରତିଜ୍ଞା(tensor->nb[0] == ଆକାର(int8_t));
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), ମୂଲ୍ୟ);
                }
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_I16:
            {
                ପ୍ରତିଜ୍ଞା(tensor->nb[0] == ଆକାର(int16_t));
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), ମୂଲ୍ୟ);
                }
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_I32:
            {
                ପ୍ରତିଜ୍ଞା(tensor->nb[0] == ଆକାର(int32_t));
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), ମୂଲ୍ୟ);
                }
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F16:
            {
                ପ୍ରତିଜ୍ଞା(tensor->nb[0] == ଆକାର(ggml_fp16_t));
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), GGML_CPU_FP32_TO_FP16(ମୂଲ୍ୟ));
                }
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_BF16:
            {
                ପ୍ରତିଜ୍ଞା(tensor->nb[0] == ଆକାର(ggml_bf16_t));
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; i++) {
                    ggml_vec_set_bf16(nc, (ggml_bf16_t *)(data + i*n1), GGML_FP32_TO_BF16(ମୂଲ୍ୟ));
                }
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ପ୍ରତିଜ୍ଞା(tensor->nb[0] == ଆକାର(ଦଶମିକ));
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (ଦଶମିକ *)(data + i*n1), ମୂଲ୍ୟ);
                }
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }

    ଫେରନ୍ତୁ tensor;
}

int32_t ggml_get_i32_1d(ଅଚଳ ସଂରଚନା ggml_tensor * tensor, ପୂର୍ଣ୍ଣସଂଖ୍ୟା i) {
    ଯଦି (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ଫେରନ୍ତୁ ggml_get_i32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    ବାଛ (tensor->type) {
        ସ୍ଥିତି GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == ଆକାର(int8_t));
                ଫେରନ୍ତୁ ((int8_t *)(tensor->data))[i];
            }
        ସ୍ଥିତି GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == ଆକାର(int16_t));
                ଫେରନ୍ତୁ ((int16_t *)(tensor->data))[i];
            }
        ସ୍ଥିତି GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == ଆକାର(int32_t));
                ଫେରନ୍ତୁ ((int32_t *)(tensor->data))[i];
            }
        ସ୍ଥିତି GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == ଆକାର(ggml_fp16_t));
                ଫେରନ୍ତୁ GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *)(tensor->data))[i]);
            }
        ସ୍ଥିତି GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == ଆକାର(ggml_bf16_t));
                ଫେରନ୍ତୁ GGML_BF16_TO_FP32(((ggml_bf16_t *)(tensor->data))[i]);
            }
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == ଆକାର(ଦଶମିକ));
                ଫେରନ୍ତୁ ((ଦଶମିକ *)(tensor->data))[i];
            }
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ଶୂନ୍ୟ ggml_set_i32_1d(ଅଚଳ ସଂରଚନା ggml_tensor * tensor, ପୂର୍ଣ୍ଣସଂଖ୍ୟା i, int32_t ମୂଲ୍ୟ) {
    ଯଦି (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_set_i32_nd(tensor, id[0], id[1], id[2], id[3], ମୂଲ୍ୟ);
        ଫେରନ୍ତୁ;
    }
    ବାଛ (tensor->type) {
        ସ୍ଥିତି GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == ଆକାର(int8_t));
                ((int8_t *)(tensor->data))[i] = ମୂଲ୍ୟ;
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == ଆକାର(int16_t));
                ((int16_t *)(tensor->data))[i] = ମୂଲ୍ୟ;
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == ଆକାର(int32_t));
                ((int32_t *)(tensor->data))[i] = ମୂଲ୍ୟ;
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == ଆକାର(ggml_fp16_t));
                ((ggml_fp16_t *)(tensor->data))[i] = GGML_CPU_FP32_TO_FP16(ମୂଲ୍ୟ);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == ଆକାର(ggml_bf16_t));
                ((ggml_bf16_t *)(tensor->data))[i] = GGML_FP32_TO_BF16(ମୂଲ୍ୟ);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == ଆକାର(ଦଶମିକ));
                ((ଦଶମିକ *)(tensor->data))[i] = ମୂଲ୍ୟ;
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

int32_t ggml_get_i32_nd(ଅଚଳ ସଂରଚନା ggml_tensor * tensor, ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1, ପୂର୍ଣ୍ଣସଂଖ୍ୟା i2, ପୂର୍ଣ୍ଣସଂଖ୍ୟା i3) {
    ଶୂନ୍ୟ * data   = (ଅକ୍ଷର *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    ବାଛ (tensor->type) {
        ସ୍ଥିତି GGML_TYPE_I8:
            ଫେରନ୍ତୁ ((int8_t *) data)[0];
        ସ୍ଥିତି GGML_TYPE_I16:
            ଫେରନ୍ତୁ ((int16_t *) data)[0];
        ସ୍ଥିତି GGML_TYPE_I32:
            ଫେରନ୍ତୁ ((int32_t *) data)[0];
        ସ୍ଥିତି GGML_TYPE_F16:
            ଫେରନ୍ତୁ GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *) data)[0]);
        ସ୍ଥିତି GGML_TYPE_BF16:
            ଫେରନ୍ତୁ GGML_BF16_TO_FP32(((ggml_bf16_t *) data)[0]);
        ସ୍ଥିତି GGML_TYPE_F32:
            ଫେରନ୍ତୁ ((ଦଶମିକ *) data)[0];
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            GGML_ABORT("fatal error");
    }
}

ଶୂନ୍ୟ ggml_set_i32_nd(ଅଚଳ ସଂରଚନା ggml_tensor * tensor, ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1, ପୂର୍ଣ୍ଣସଂଖ୍ୟା i2, ପୂର୍ଣ୍ଣସଂଖ୍ୟା i3, int32_t ମୂଲ୍ୟ) {
    ଶୂନ୍ୟ * data   = (ଅକ୍ଷର *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    ବାଛ (tensor->type) {
        ସ୍ଥିତି GGML_TYPE_I8:
            {
                ((int8_t *)(data))[0] = ମୂଲ୍ୟ;
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_I16:
            {
                ((int16_t *)(data))[0] = ମୂଲ୍ୟ;
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_I32:
            {
                ((int32_t *)(data))[0] = ମୂଲ୍ୟ;
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(data))[0] = GGML_CPU_FP32_TO_FP16(ମୂଲ୍ୟ);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(data))[0] = GGML_FP32_TO_BF16(ମୂଲ୍ୟ);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ((ଦଶମିକ *)(data))[0] = ମୂଲ୍ୟ;
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ଦଶମିକ ggml_get_f32_1d(ଅଚଳ ସଂରଚନା ggml_tensor * tensor, ପୂର୍ଣ୍ଣସଂଖ୍ୟା i) {
    ଯଦି (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ଫେରନ୍ତୁ ggml_get_f32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    ବାଛ (tensor->type) {
        ସ୍ଥିତି GGML_TYPE_I8:
            {
                ଫେରନ୍ତୁ ((int8_t *)(tensor->data))[i];
            }
        ସ୍ଥିତି GGML_TYPE_I16:
            {
                ଫେରନ୍ତୁ ((int16_t *)(tensor->data))[i];
            }
        ସ୍ଥିତି GGML_TYPE_I32:
            {
                ଫେରନ୍ତୁ ((int32_t *)(tensor->data))[i];
            }
        ସ୍ଥିତି GGML_TYPE_F16:
            {
                ଫେରନ୍ତୁ GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *)(tensor->data))[i]);
            }
        ସ୍ଥିତି GGML_TYPE_BF16:
            {
                ଫେରନ୍ତୁ GGML_BF16_TO_FP32(((ggml_bf16_t *)(tensor->data))[i]);
            }
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ଫେରନ୍ତୁ ((ଦଶମିକ *)(tensor->data))[i];
            }
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ଶୂନ୍ୟ ggml_set_f32_1d(ଅଚଳ ସଂରଚନା ggml_tensor * tensor, ପୂର୍ଣ୍ଣସଂଖ୍ୟା i, ଦଶମିକ ମୂଲ୍ୟ) {
    ଯଦି (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_set_f32_nd(tensor, id[0], id[1], id[2], id[3], ମୂଲ୍ୟ);
        ଫେରନ୍ତୁ;
    }
    ବାଛ (tensor->type) {
        ସ୍ଥିତି GGML_TYPE_I8:
            {
                ((int8_t *)(tensor->data))[i] = ମୂଲ୍ୟ;
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_I16:
            {
                ((int16_t *)(tensor->data))[i] = ମୂଲ୍ୟ;
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_I32:
            {
                ((int32_t *)(tensor->data))[i] = ମୂଲ୍ୟ;
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(tensor->data))[i] = GGML_CPU_FP32_TO_FP16(ମୂଲ୍ୟ);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(tensor->data))[i] = GGML_FP32_TO_BF16(ମୂଲ୍ୟ);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ((ଦଶମିକ *)(tensor->data))[i] = ମୂଲ୍ୟ;
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ଦଶମିକ ggml_get_f32_nd(ଅଚଳ ସଂରଚନା ggml_tensor * tensor, ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1, ପୂର୍ଣ୍ଣସଂଖ୍ୟା i2, ପୂର୍ଣ୍ଣସଂଖ୍ୟା i3) {
    ଶୂନ୍ୟ * data   = (ଅକ୍ଷର *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    ବାଛ (tensor->type) {
        ସ୍ଥିତି GGML_TYPE_I8:
            ଫେରନ୍ତୁ ((int8_t *) data)[0];
        ସ୍ଥିତି GGML_TYPE_I16:
            ଫେରନ୍ତୁ ((int16_t *) data)[0];
        ସ୍ଥିତି GGML_TYPE_I32:
            ଫେରନ୍ତୁ ((int32_t *) data)[0];
        ସ୍ଥିତି GGML_TYPE_F16:
            ଫେରନ୍ତୁ GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *) data)[0]);
        ସ୍ଥିତି GGML_TYPE_BF16:
            ଫେରନ୍ତୁ GGML_BF16_TO_FP32(((ggml_bf16_t *) data)[0]);
        ସ୍ଥିତି GGML_TYPE_F32:
            ଫେରନ୍ତୁ ((ଦଶମିକ *) data)[0];
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            GGML_ABORT("fatal error");
    }
}

ଶୂନ୍ୟ ggml_set_f32_nd(ଅଚଳ ସଂରଚନା ggml_tensor * tensor, ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1, ପୂର୍ଣ୍ଣସଂଖ୍ୟା i2, ପୂର୍ଣ୍ଣସଂଖ୍ୟା i3, ଦଶମିକ ମୂଲ୍ୟ) {
    ଶୂନ୍ୟ * data   = (ଅକ୍ଷର *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    ବାଛ (tensor->type) {
        ସ୍ଥିତି GGML_TYPE_I8:
            {
                ((int8_t *)(data))[0] = ମୂଲ୍ୟ;
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_I16:
            {
                ((int16_t *)(data))[0] = ମୂଲ୍ୟ;
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_I32:
            {
                ((int32_t *)(data))[0] = ମୂଲ୍ୟ;
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(data))[0] = GGML_CPU_FP32_TO_FP16(ମୂଲ୍ୟ);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(data))[0] = GGML_FP32_TO_BF16(ମୂଲ୍ୟ);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ((ଦଶମିକ *)(data))[0] = ମୂଲ୍ୟ;
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

////////////////////////////////////////////////////////////////////////////////

// ggml_compute_forward_mul_mat

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_mul_mat_one_chunk(
    ଅଚଳ ସଂରଚନା ggml_compute_params * params,
    ସଂରଚନା ggml_tensor * dst,
    ଅଚଳ ଗଣନା ggml_type type,
    ଅଚଳ int64_t ଅଙ୍କ_rows_per_vec_dot,
    ଅଚଳ int64_t ir0_start,
    ଅଚଳ int64_t ir0_end,
    ଅଚଳ int64_t ir1_start,
    ଅଚଳ int64_t ir1_end) {

    ଅଚଳ ସଂରଚନା ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ସଂରଚନା ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS

    ଅଚଳ bool src1_cont = ggml_is_contiguous(src1);

    ggml_vec_dot_t ଅଚଳ vec_dot      = type_traits_cpu[type].vec_dot;
    ଗଣନା ggml_type ଅଚଳ vec_dot_type = type_traits_cpu[type].vec_dot_type;

    // broadcast factors
    ଅଚଳ int64_t r2 = ne12 / ne02;
    ଅଚଳ int64_t r3 = ne13 / ne03;

    //printf("ir0_start = %6lld, ir0_end = %6lld, ir1_start = %6lld, ir1_end = %6lld\n", ir0_start, ir0_end, ir1_start, ir1_end);

    // threads with no work simply yield (not sure if it helps)
    ଯଦି (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ ଶୂନ୍ୟ * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
    ଅଚଳ size_t row_size = ggml_row_size(vec_dot_type, ne10);

    ପ୍ରତିଜ୍ଞା(ne12 % ne02 == 0);
    ପ୍ରତିଜ୍ଞା(ne13 % ne03 == 0);

    // block-tiling attempt
    ଅଚଳ int64_t blck_0 = 16;
    ଅଚଳ int64_t blck_1 = 16;

    ଅଚଳ size_t src1_col_stride = src1_cont || src1->type != vec_dot_type ? row_size : nb11;

    // attempt to reduce false-sharing (does not seem to make a difference)
    // 16 * 2, accounting for mmla kernels
    ଦଶମିକ tmp[32];

    ପାଇଁ (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        ପାଇଁ (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            ପାଇଁ (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += ଅଙ୍କ_rows_per_vec_dot) {
                ଅଚଳ int64_t i13 = (ir1 / (ne12 * ne1));
                ଅଚଳ int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                ଅଚଳ int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                // broadcast src0 into src1
                ଅଚଳ int64_t i03 = i13 / r3;
                ଅଚଳ int64_t i02 = i12 / r2;

                ଅଚଳ int64_t i1 = i11;
                ଅଚଳ int64_t i2 = i12;
                ଅଚଳ int64_t i3 = i13;

                ଅଚଳ ଅକ୍ଷର * src0_row = (ଅଚଳ ଅକ୍ଷର*)src0->data + (0 + i02 * nb02 + i03 * nb03);

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                ଅଚଳ ଅକ୍ଷର * src1_col = (ଅଚଳ ଅକ୍ଷର*)wdata +
                    (src1_cont || src1->type != vec_dot_type
                        ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                        : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                ଦଶମିକ * dst_col = (ଦଶମିକ*)((ଅକ୍ଷର*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

                //for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                //    vec_dot(ne00, &dst_col[ir0], src0_row + ir0*nb01, src1_col);
                //}

                ପାଇଁ (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += ଅଙ୍କ_rows_per_vec_dot) {
                    vec_dot(ne00, &tmp[ir0 - iir0], (ଅଙ୍କ_rows_per_vec_dot > 1 ? 16 : 0), src0_row + ir0 * nb01, (ଅଙ୍କ_rows_per_vec_dot > 1 ? nb01 : 0), src1_col, (ଅଙ୍କ_rows_per_vec_dot > 1 ? src1_col_stride : 0), ଅଙ୍କ_rows_per_vec_dot);
                }

                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା cn = 0; cn < ଅଙ୍କ_rows_per_vec_dot; ++cn) {
                    memcpy(&dst_col[iir0 + cn * nb1 / nb0], tmp + (cn * 16), (MIN(iir0 + blck_0, ir0_end) - iir0) * ଆକାର(ଦଶମିକ));
                }
            }
        }
    }
}

ଶୂନ୍ୟ ggml_compute_forward_mul_mat(
        ଅଚଳ ସଂରଚନା ggml_compute_params * params,
              ସଂରଚନା ggml_tensor * dst) {

    ଅଚଳ ସଂରଚନା ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ସଂରଚନା ggml_tensor * src1 = dst->src[1];

    ଅଚଳ int32_t hint = ggml_get_op_params_i32(dst, 1);
    ଯଦି (hint == GGML_HINT_SRC0_IS_HADAMARD && !params->use_ref) {
        ggml_compute_forward_fwht(params, dst);
        ଫେରନ୍ତୁ;
    }

    // If tiled is supported, it will execute the full op here and we return
    ଯଦି (ggml_compute_forward_mul_mat_tiled(params, dst)) {
        ଫେରନ୍ତୁ;
    }

    GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଗଣନା ggml_type           ଅଚଳ vec_dot_type         = type_traits_cpu[src0->type].vec_dot_type;
    ggml_from_float_t        ଅଚଳ from_float           = type_traits_cpu[vec_dot_type].from_float;
    int64_t                  ଅଚଳ vec_dot_ଅଙ୍କ_rows     = type_traits_cpu[src0->type].nrows;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == ଆକାର(ଦଶମିକ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    // TODO: extract to "extra_op"
#ଯଦି GGML_USE_LLAMAFILE
    // broadcast factors
    ଅଚଳ int64_t r2 = ne12 / ne02;
    ଅଚଳ int64_t r3 = ne13 / ne03;

    ଅଚଳ bool src1_cont = ggml_is_contiguous(src1);

    ଯଦି (!params->use_ref && src1_cont) {
        ପାଇଁ (int64_t i13 = 0; i13 < ne13; i13++)
            ପାଇଁ (int64_t i12 = 0; i12 < ne12; i12++)
                ଯଦି (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_size(src0->type),
                                     (ଅଚଳ ଅକ୍ଷର *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_size(src0->type),
                                     (ଅଚଳ ଅକ୍ଷର *)src1->data + i12*nb12 + i13*nb13,
                                     nb11/ggml_type_size(src1->type),
                                     (ଅକ୍ଷର *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_size(dst->type),
                                     src0->type,
                                     src1->type,
                                     dst->type))
                    ଯାଅ UseGgmlGemm1;
        ଫେରନ୍ତୁ;
    }
UseGgmlGemm1:;
#endif

    ଯଦି (src1->type != vec_dot_type) {
        ଅକ୍ଷର * wdata = params->wdata;

        ଅଚଳ size_t nbw0 = ggml_type_size(vec_dot_type);
        ଅଚଳ size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        ଅଚଳ size_t nbw2 = nbw1*ne11;
        ଅଚଳ size_t nbw3 = nbw2*ne12;

        ପ୍ରତିଜ୍ଞା(params->wsize >= ne13*nbw3);
        // src1 is either packed from F32 into vec_dot_type, or widened from F16 or BF16 into the F32 work buffer
        ଅଚଳ bool widen = src1->type != GGML_TYPE_F32;

        GGML_ASSERT(!widen || vec_dot_type == GGML_TYPE_F32);
        GGML_ASSERT(!widen || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_BF16);

    #ଯଦି 0
        ପାଇଁ (int64_t i13 = 0; i13 < ne13; ++i13) {
            ପାଇଁ (int64_t i12 = 0; i12 < ne12; ++i12) {
                ପାଇଁ (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                    from_float((ଦଶମିକ *)((ଅକ୍ଷର *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (ଶୂନ୍ୟ *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                                ne10);
                }
            }
        }
    #ଅନ୍ୟଥା
        ଅଚଳ int64_t bs = ggml_blck_size(vec_dot_type);

        ପାଇଁ (int64_t i13 = 0; i13 < ne13; ++i13) {
            ପାଇଁ (int64_t i12 = 0; i12 < ne12; ++i12) {
                ପାଇଁ (int64_t i11 = 0; i11 < ne11; ++i11) {
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    ଅଚଳ ଶୂନ୍ୟ * src1_row  = (ଅଚଳ ଅକ୍ଷର *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10;
                    ଶୂନ୍ୟ       * wdata_row =                    wdata  + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0;

                    ଅଚଳ int64_t ne10_block_size = (ne10_block_end - ne10_block_start) * bs;

                    ଯଦି (src1->type == GGML_TYPE_F16) {
                        ggml_cpu_fp16_to_fp32((ଅଚଳ ggml_fp16_t *) src1_row, (ଦଶମିକ *) wdata_row, ne10_block_size);
                    } ଅନ୍ୟଥା ଯଦି (src1->type == GGML_TYPE_BF16) {
                        ggml_cpu_bf16_to_fp32((ଅଚଳ ggml_bf16_t *) src1_row, (ଦଶମିକ *) wdata_row, ne10_block_size);
                    } ଅନ୍ୟଥା {
                        from_float((ଅଚଳ ଦଶମିକ *) src1_row, wdata_row, ne10_block_size);
                    }
                }
            }
        }
    #endif
    }

    ଯଦି (ith == 0) {
        // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
        atomic_store_explicit(&params->threadpool->current_chunk, nth, memory_order_relaxed);
    }

    ggml_barrier(params->threadpool);

#ଯଦି GGML_USE_LLAMAFILE
    ଯଦି (!params->use_ref && src1->type != vec_dot_type) {
        ଅଚଳ ଶୂନ୍ୟ* wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        ଅଚଳ size_t row_size = ggml_row_size(vec_dot_type, ne10);

        ପାଇଁ (int64_t i13 = 0; i13 < ne13; i13++)
            ପାଇଁ (int64_t i12 = 0; i12 < ne12; i12++)
                ଯଦି (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_size(src0->type),
                                     (ଅଚଳ ଅକ୍ଷର *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_size(src0->type),
                                     (ଅଚଳ ଅକ୍ଷର *)wdata + (i12*ne11 + i13*ne12*ne11)*row_size,
                                     row_size/ggml_type_size(vec_dot_type),
                                     (ଅକ୍ଷର *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_size(dst->type),
                                     src0->type,
                                     vec_dot_type,
                                     dst->type))
                    ଯାଅ UseGgmlGemm2;
        ଫେରନ୍ତୁ;
    }
UseGgmlGemm2:;
#endif

    // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
    ଅଚଳ int64_t nr0 = ne0;

    // This is the size of the rest of the dimensions of the result
    ଅଚଳ int64_t nr1 = ne1 * ne2 * ne3;

    // Now select a reasonable chunk size.
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା chunk_size = 16;

    // We need to step up the size if it's small
    ଯଦି (nr0 == 1 || nr1 == 1) {
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
    ଯଦି (nchunk0 * nchunk1 < nth * 4 || ggml_is_numa()) {
        // distribute the thread work across the inner or outer loop based on which one is larger
        nchunk0 = nr0 > nr1 ? nth : 1; // parallelize by src0 rows
        nchunk1 = nr0 > nr1 ? 1 : nth; // parallelize by src1 rows
    }

    // The number of elements in each chunk
    ଅଚଳ int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    ଅଚଳ int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    // The first chunk comes from our thread_id, the rest will get auto-assigned.
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା current_chunk = ith;

    ଯେତେବେଳେ (current_chunk < nchunk0 * nchunk1) {
        ଅଚଳ int64_t ith0 = current_chunk % nchunk0;
        ଅଚଳ int64_t ith1 = current_chunk / nchunk0;

        ଅଚଳ int64_t ir0_start = dr0 * ith0;
        ଅଚଳ int64_t ir0_end = MIN(ir0_start + dr0, nr0);

        ଅଚଳ int64_t ir1_start = dr1 * ith1;
        ଅଚଳ int64_t ir1_end = MIN(ir1_start + dr1, nr1);

        // dot kernels can handle 1 row and col at a time, but mmla kernels can process 2 rows and cols
        int64_t ଅଙ୍କ_rows_per_vec_dot = vec_dot_ଅଙ୍କ_rows;

        // these checks are needed to avoid crossing dim1 boundaries
        // can be optimized, but the logic would become more complicated, so keeping it like this for simplicity
        ଯଦି ((nr0 % 2 != 0) || (ne11 % 2 != 0) || ((ir0_end - ir0_start) % 2 != 0) || ((ir1_end - ir1_start) % 2 != 0)) {
            ଅଙ୍କ_rows_per_vec_dot = 1;
        }
        ggml_compute_forward_mul_mat_one_chunk(params, dst, src0->type, ଅଙ୍କ_rows_per_vec_dot, ir0_start, ir0_end, ir1_start, ir1_end);

        ଯଦି (nth >= nchunk0 * nchunk1) {
            ଭାଙ୍ଗ;
        }

        current_chunk = atomic_fetch_add_explicit(&params->threadpool->current_chunk, 1, memory_order_relaxed);
    }
}

// ggml_compute_forward_mul_mat_id

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id)*ids->ne[0]*ids->ne[1] + (i1)]

ସଂରଚନା mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_mul_mat_id_one_chunk(
    ସଂରଚନା ggml_tensor * dst,
    ଅଚଳ ସଂରଚନା ggml_tensor * src0,
    ଅଚଳ ସଂରଚନା ggml_tensor * src1,
    ଅଚଳ ସଂରଚନା ggml_tensor * ids,
    ଅଚଳ int64_t cur_a,
    ଅଚଳ int64_t ir0_start,
    ଅଚଳ int64_t ir0_end,
    ଅଚଳ int64_t ir1_start,
    ଅଚଳ int64_t ir1_end,
    ଅଚଳ ଅକ୍ଷର * src0_cur,
    ଅଚଳ ସଂରଚନା mmid_row_mapping * matrix_rows,
    ଅଚଳ size_t row_size,
    ଅଚଳ bool src1_cont,
    ଅଚଳ ଶୂନ୍ୟ * wdata) {

    GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS

    ଅଚଳ ଗଣନା ggml_type type = src0->type;

    ggml_vec_dot_t    ଅଚଳ vec_dot      = type_traits_cpu[type].vec_dot;
    ଗଣନା ggml_type    ଅଚଳ vec_dot_type = type_traits_cpu[type].vec_dot_type;

    ଅଚଳ int64_t blck_0 = 16;
    ଅଚଳ int64_t blck_1 = 16;

    ଦଶମିକ tmp[16];

    ପାଇଁ (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        ପାଇଁ (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            ପାଇଁ (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ++ir1) {
                ଅଚଳ int64_t _i12 = ir1; // logical row index for this expert

                ସଂରଚନା mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, _i12);
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା id       = row_mapping.i1; // selected expert index

                ଅଚଳ int64_t  i11 = id % ne11;
                ଅଚଳ int64_t  i12 = row_mapping.i2; // row index in src1

                ଅଚଳ int64_t  i1 = id;  // selected expert index
                ଅଚଳ int64_t  i2 = i12; // row

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                ଅଚଳ ଅକ୍ଷର * src1_col = (ଅଚଳ ଅକ୍ଷର *) wdata +
                    (src1_cont || src1->type != vec_dot_type
                    ? (i11      + i12*ne11)*row_size
                    : (i11*nb11 + i12*nb12));

                ଦଶମିକ * dst_col = (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + (i1*nb1 + i2*nb2));

                ପାଇଁ (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                    vec_dot(ne00, &tmp[ir0 - iir0], 0, src0_cur + ir0*nb01, 0, src1_col, 0, 1);
                }

                memcpy(&dst_col[iir0], tmp, (MIN(iir0 + blck_0, ir0_end) - iir0)*ଆକାର(ଦଶମିକ));
            }
        }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ * incr_ptr_aligned(ଶୂନ୍ୟ ** p, size_t size, size_t align) {

    ଶୂନ୍ୟ * ptr = *p;
    ptr = (ଶୂନ୍ୟ *) GGML_PAD((uintptr_t) ptr, align);
    *p = (ଶୂନ୍ୟ *) ((ଅକ୍ଷର *) ptr + size);
    ଫେରନ୍ତୁ ptr;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_mul_mat_id(
        ଅଚଳ ସଂରଚନା ggml_compute_params * params,
              ସଂରଚନା ggml_tensor * dst) {

    ଅଚଳ ସଂରଚନା ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ସଂରଚନା ggml_tensor * src1 = dst->src[1];
    ଅଚଳ ସଂରଚନା ggml_tensor * ids = dst->src[2];

    GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ଗଣନା ggml_type type = src0->type;

    ଅଚଳ bool src1_cont = ggml_is_contiguous(src1);

    ଗଣନା ggml_type    ଅଚଳ vec_dot_type    = type_traits_cpu[type].vec_dot_type;
    ggml_from_float_t ଅଚଳ from_float      = type_traits_cpu[vec_dot_type].from_float;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == ଆକାର(ଦଶମିକ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // row groups
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_ids = ids->ne[0]; // n_expert_used
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_as  = ne02;       // n_expert

    ଶୂନ୍ୟ * wdata_cur = params->wdata;

    ଯଦି (src1->type != vec_dot_type) {
        incr_ptr_aligned(&wdata_cur, ggml_row_size(vec_dot_type, ggml_nelements(src1)), ଆକାର(int64_t));
    }

    int64_t * matrix_row_counts = // [n_as]
        incr_ptr_aligned(&wdata_cur, n_as*ଆକାର(int64_t), ଆକାର(int64_t));

    ସଂରଚନା mmid_row_mapping * matrix_rows = // [n_as][ids->ne[0]*ids->ne[1]]
        incr_ptr_aligned(&wdata_cur, n_as*ids->ne[0]*ids->ne[1]*ଆକାର(ସଂରଚନା mmid_row_mapping), ଆକାର(int64_t));

    ଅକ୍ଷର (*atomic_current_chunk)[CACHE_ଧାଡ଼ି_SIZE] = // [n_as]
        incr_ptr_aligned(&wdata_cur, CACHE_ଧାଡ଼ି_SIZE * n_as, CACHE_ଧାଡ଼ି_SIZE);

    // Tiled matmul (see tiled.h); per-thread work buffers, 0 bytes when disabled. The
    // reservation is unconditional, the per expert eligibility is decided at dispatch time
    ଅକ୍ଷର * tiled_scratch = incr_ptr_aligned(&wdata_cur, ggml_tiled_wdata_size(nth, dst), 64);

    GGML_ASSERT(params->wsize >= (size_t)((ଅକ୍ଷର *) wdata_cur - (ଅକ୍ଷର *) params->wdata));

    ଯଦି (src1->type != vec_dot_type) {
        ଅକ୍ଷର * wdata = params->wdata;

        ଅଚଳ size_t nbw0 = ggml_type_size(vec_dot_type);
        ଅଚଳ size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        ଅଚଳ size_t nbw2 = nbw1*ne11;
        ଅଚଳ size_t nbw3 = nbw2*ne12;

        ପ୍ରତିଜ୍ଞା(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

#ଯଦି 0
        ପାଇଁ (int64_t i13 = 0; i13 < ne13; ++i13) {
            ପାଇଁ (int64_t i12 = ith; i12 < ne12; i12 += nth) {
                ପାଇଁ (int64_t i11 = 0; i11 < ne11; ++i11) {
                    from_float((ଦଶମିକ *)((ଅକ୍ଷର *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (ଶୂନ୍ୟ *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                               ne10);
                }
            }
        }
#ଅନ୍ୟଥା
        ପାଇଁ (int64_t i13 = 0; i13 < ne13; ++i13) {
            ପାଇଁ (int64_t i12 = 0; i12 < ne12; ++i12) {
                ପାଇଁ (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = ggml_blck_size(vec_dot_type);
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    from_float((ଦଶମିକ *)((ଅକ୍ଷର *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10),
                               (ଶୂନ୍ୟ *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0),
                               (ne10_block_end - ne10_block_start) * bs);
                }
            }
        }
#endif
    }

    ଯଦି (ith == 0) {
        // initialize matrix_row_counts
        memset(matrix_row_counts, 0, n_as*ଆକାର(int64_t));

        // group rows by src0 matrix
        ପାଇଁ (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା id = 0; id < n_ids; ++id) {
                ଅଚଳ int32_t i02 = *(ଅଚଳ int32_t *) ((ଅଚଳ ଅକ୍ଷର *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);

                ପ୍ରତିଜ୍ଞା(i02 >= 0 && i02 < n_as);

                MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = (ସଂରଚନା mmid_row_mapping) {id, iid1};
                matrix_row_counts[i02] += 1;
            }
        }
    }

    // reset current_chunk
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା cur_a = ith; cur_a < n_as; cur_a += nth) {
        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);
        *current_chunk_ctr = nth;
    }

    ggml_barrier(params->threadpool);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା cur_a = 0; cur_a < n_as; ++cur_a) {
        ଅଚଳ int64_t cne1 = matrix_row_counts[cur_a];

        ଯଦି (cne1 == 0) {
            ଜାରି;
        }

        // tiled takes over if profitable for this expert (see tiled.h)
        ଯଦି (ggml_compute_forward_mul_mat_id_tiled(params, dst, cur_a, cne1, (ଅଚଳ int32_t *) &MMID_MATRIX_ROW(cur_a, 0), tiled_scratch)) {
            ଜାରି;
        }

        ଅଚଳ ଅକ୍ଷର * src0_cur = (ଅଚଳ ଅକ୍ଷର *) src0->data + cur_a * nb02;
        ଅଚଳ ଶୂନ୍ୟ * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        ଅଚଳ size_t row_size = ggml_row_size(vec_dot_type, ne10);

        ଅଚଳ int64_t nr0 = ne01;
        ଅଚଳ int64_t nr1 = cne1;

        ପୂର୍ଣ୍ଣସଂଖ୍ୟା chunk_size = 16;
        ଯଦି (nr0 == 1 || nr1 == 1) {
            chunk_size = 64;
        }

        // disable for NUMA
        ଅଚଳ bool disable_chunking = ggml_is_numa();

        int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
        int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

        ଯଦି (nchunk0 * nchunk1 < nth * 4 || disable_chunking) {
            nchunk0 = nr0 > nr1 ? nth : 1;
            nchunk1 = nr0 > nr1 ? 1 : nth;
        }

        ଅଚଳ int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        ଅଚଳ int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

        ପୂର୍ଣ୍ଣସଂଖ୍ୟା current_chunk = ith;

        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);

        ଯେତେବେଳେ (current_chunk < nchunk0 * nchunk1) {
            ଅଚଳ int64_t ith0 = current_chunk % nchunk0;
            ଅଚଳ int64_t ith1 = current_chunk / nchunk0;

            ଅଚଳ int64_t ir0_start = dr0 * ith0;
            ଅଚଳ int64_t ir0_end = MIN(ir0_start + dr0, nr0);

            ଅଚଳ int64_t ir1_start = dr1 * ith1;
            ଅଚଳ int64_t ir1_end = MIN(ir1_start + dr1, nr1);

            ggml_compute_forward_mul_mat_id_one_chunk(
                dst, src0, src1, ids, cur_a,
                ir0_start, ir0_end, ir1_start, ir1_end,
                src0_cur, matrix_rows, row_size, src1_cont, wdata
            );

            ଯଦି (nth >= nchunk0 * nchunk1) {
                ଭାଙ୍ଗ;
            }

            current_chunk = atomic_fetch_add_explicit(current_chunk_ctr, 1, memory_order_relaxed);
        }
    }
}

/////////////////////////////////

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward(ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * tensor) {
    GGML_ASSERT(params);

    ଯଦି (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {
        ଫେରନ୍ତୁ;
    }

    // extra_buffer op?
    ଯଦି (ggml_cpu_extra_compute_forward(params, tensor)) {
        ଫେରନ୍ତୁ;
    }

    ବାଛ (tensor->op) {
        ସ୍ଥିତି GGML_OP_DUP:
            {
                ggml_compute_forward_dup(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_ADD:
            {
                ggml_compute_forward_add(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_ADD_ID:
            {
                ggml_compute_forward_add_id(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_ADD1:
            {
                ggml_compute_forward_add1(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_ACC:
            {
                ggml_compute_forward_acc(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SUB:
            {
                ggml_compute_forward_sub(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_MUL:
            {
                ggml_compute_forward_mul(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_DIV:
            {
                ggml_compute_forward_div(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SQR:
            {
                ggml_compute_forward_sqr(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SQRT:
            {
                ggml_compute_forward_sqrt(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_LOG:
            {
                ggml_compute_forward_log(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SIN:
            {
                ggml_compute_forward_sin(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_COS:
            {
                ggml_compute_forward_cos(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SUM:
            {
                ggml_compute_forward_sum(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SUM_ROWS:
            {
                ggml_compute_forward_sum_rows(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_CUMSUM:
            {
                ggml_compute_forward_cumsum(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_MEAN:
            {
                ggml_compute_forward_mean(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_ARGMAX:
            {
                ggml_compute_forward_argmax(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_ଗଣନା_EQUAL:
            {
                ggml_compute_forward_ଗଣନା_equal(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_REPEAT:
            {
                ggml_compute_forward_repeat(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_REPEAT_BACK:
            {
                ggml_compute_forward_repeat_back(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_CONCAT:
            {
                ggml_compute_forward_concat(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SILU_BACK:
            {
                ggml_compute_forward_silu_back(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_NORM:
            {
                ggml_compute_forward_norm(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_RMS_NORM:
            {
                ggml_compute_forward_rms_norm(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_RMS_NORM_BACK:
            {
                ggml_compute_forward_rms_norm_back(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_GROUP_NORM:
            {
                ggml_compute_forward_group_norm(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_L2_NORM:
            {
                ggml_compute_forward_l2_norm(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_MUL_MAT:
            {
                ggml_compute_forward_mul_mat(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_MUL_MAT_ID:
            {
                ggml_compute_forward_mul_mat_id(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_OUT_PROD:
            {
                ggml_compute_forward_out_prod(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SCALE:
            {
                ggml_compute_forward_scale(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SET:
            {
                ggml_compute_forward_set(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_CPY:
            {
                ggml_compute_forward_cpy(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_CONT:
            {
                ggml_compute_forward_cont(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_GET_ROWS:
            {
                ggml_compute_forward_get_rows(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_GET_ROWS_BACK:
            {
                ggml_compute_forward_get_rows_back(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SET_ROWS:
            {
                ggml_compute_forward_set_rows(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_DIAG:
            {
                ggml_compute_forward_diag(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_DIAG_MASK_INF:
            {
                ggml_compute_forward_diag_mask_inf(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_DIAG_MASK_ZERO:
            {
                ggml_compute_forward_diag_mask_zero(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SOFT_MAX:
            {
                ggml_compute_forward_soft_max(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SOFT_MAX_BACK:
            {
                ggml_compute_forward_soft_max_ext_back(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_ROPE:
            {
                ggml_compute_forward_rope(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_ROPE_BACK:
            {
                ggml_compute_forward_rope_back(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_CLAMP:
            {
                ggml_compute_forward_clamp(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_compute_forward_conv_transpose_1d(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_IM2COL:
            {
                ggml_compute_forward_im2col(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_IM2COL_BACK:
            {
                ggml_compute_forward_im2col_back_f32(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_IM2COL_3D:
            {
                ggml_compute_forward_im2col_3d(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_COL2IM_1D:
            {
                ggml_compute_forward_col2im_1d(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_CONV_2D:
            {
                ggml_compute_forward_conv_2d(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_CONV_3D:
            {
                ggml_compute_forward_conv_3d(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_CONV_2D_DW:
            {
                ggml_compute_forward_conv_2d_dw(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_CONV_TRANSPOSE_2D:
            {
                ggml_compute_forward_conv_transpose_2d(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_POOL_1D:
            {
                ggml_compute_forward_pool_1d(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_POOL_2D:
            {
                ggml_compute_forward_pool_2d(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_POOL_2D_BACK:
            {
                ggml_compute_forward_pool_2d_back(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_UPSCALE:
            {
                ggml_compute_forward_upscale(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_PAD:
            {
                ggml_compute_forward_pad(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_PAD_REFLECT_1D:
            {
                ggml_compute_forward_pad_reflect_1d(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_ROLL:
            {
                ggml_compute_forward_roll(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_ARANGE:
            {
                ggml_compute_forward_arange(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_TIMESTEP_EMBEDDING:
            {
                ggml_compute_forward_timestep_embedding(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_ARGSORT:
            {
                ggml_compute_forward_argsort(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_ଶୀର୍ଷ_K:
            {
                ggml_compute_forward_ଶୀର୍ଷ_k(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_LEAKY_RELU:
            {
                ggml_compute_forward_leaky_relu(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_TRI:
            {
                ggml_compute_forward_tri(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_FILL:
            {
                ggml_compute_forward_fill(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_FLASH_ATTN_EXT:
            {
                ggml_compute_forward_flash_attn_ext(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_FLASH_ATTN_BACK:
            {
                int32_t t = ggml_get_op_params_i32(tensor, 0);
                GGML_ASSERT(t == 0 || t == 1);
                bool masked = t != 0;
                ggml_compute_forward_flash_attn_back(params, masked, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SSM_CONV:
            {
                ggml_compute_forward_ssm_conv(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SSM_SCAN:
            {
                ggml_compute_forward_ssm_scan(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_WIN_PART:
            {
                ggml_compute_forward_win_part(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_WIN_UNPART:
            {
                ggml_compute_forward_win_unpart(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_UNARY:
            {
                ggml_compute_forward_unary(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_GLU:
            {
                ggml_compute_forward_glu(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_GET_REL_POS:
            {
                ggml_compute_forward_get_rel_pos(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_ADD_REL_POS:
            {
                ggml_compute_forward_add_rel_pos(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_RWKV_WKV6:
            {
                ggml_compute_forward_rwkv_wkv6(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_GATED_LINEAR_ATTN:
            {
                ggml_compute_forward_gla(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_RWKV_WKV7:
            {
                ggml_compute_forward_rwkv_wkv7(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SOLVE_TRI:
            {
                ggml_compute_forward_solve_tri(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_GATED_DELTA_NET:
            {
                ggml_compute_forward_gated_delta_net(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_LIGHTNING_INDEXER:
            {
                ggml_compute_forward_lightning_indexer(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_DSV4_HC_COMB:
            {
                ggml_compute_forward_dsv4_hc_comb(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_DSV4_HC_PRE:
            {
                ggml_compute_forward_dsv4_hc_pre(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_DSV4_HC_POST:
            {
                ggml_compute_forward_dsv4_hc_post(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_ଶୀର୍ଷ_K_MASK:
            {
                ggml_compute_forward_ଶୀର୍ଷ_k_mask(params, tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_MAP_CUSTOM1:
            {
                ggml_compute_forward_map_custom1(params, tensor);
            }
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_MAP_CUSTOM2:
            {
                ggml_compute_forward_map_custom2(params, tensor);
            }
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_MAP_CUSTOM3:
            {
                ggml_compute_forward_map_custom3(params, tensor);
            }
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_CUSTOM:
            {
                ggml_compute_forward_custom(params, tensor);
            }
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_CROSS_ENTROPY_LOSS:
            {
                ggml_compute_forward_cross_entropy_loss(params, tensor);
            }
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            {
                ggml_compute_forward_cross_entropy_loss_back(params, tensor);
            }
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_OPT_STEP_ADAMW:
            {
                ggml_compute_forward_opt_step_adamw(params, tensor);
            }
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_OPT_STEP_SGD:
            {
                ggml_compute_forward_opt_step_sgd(params, tensor);
            }
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_NONE:
            {
                // nop
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_RESHAPE:
            {
                // nop
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_PERMUTE:
            {
                // nop
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_VIEW:
            {
                // nop
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_TRANSPOSE:
            {
                // nop
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_ଗଣନା:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// Android's libc implementation "bionic" does not support setting affinity
#ଯଦି defined(__gnu_linux__)
ସ୍ଥିର ଶୂନ୍ୟ set_numa_thread_affinity(ପୂର୍ଣ୍ଣସଂଖ୍ୟା thread_n) {
    ଯଦି (!ggml_is_numa()) {
        ଫେରନ୍ତୁ;
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_ଅଙ୍କ;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା rv;
    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    ବାଛ(g_state.numa.numa_strategy) {
        ସ୍ଥିତି GGML_NUMA_STRATEGY_DISTRIBUTE:
            // run thread on node_num thread_n / (threads per node)
            ଗଣ୍ଠି_ଅଙ୍କ = thread_n % g_state.numa.n_nodes;
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_NUMA_STRATEGY_ISOLATE:
            // run thread on current_node
            ଗଣ୍ଠି_ଅଙ୍କ = g_state.numa.current_ଗଣ୍ଠି;
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_NUMA_STRATEGY_NUMACTL:
            // use the cpuset that numactl gave us
            rv = pthread_setaffinity_np(pthread_self(), setsize, &g_state.numa.cpuset);
            ଯଦି (rv) {
                fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n",strerror(rv));
            }
            ଫେରନ୍ତୁ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ;
    }

    ସଂରଚନା ggml_numa_ଗଣ୍ଠି * ଗଣ୍ଠି = &g_state.numa.nodes[ଗଣ୍ଠି_ଅଙ୍କ];

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    ପାଇଁ (size_t i = 0; i < ଗଣ୍ଠି->n_cpus; ++i) {
        CPU_SET_S(ଗଣ୍ଠି->cpus[i], setsize, cpus);
    }

    rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    ଯଦି (rv) {
            fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}

ସ୍ଥିର ଶୂନ୍ୟ clear_numa_thread_affinity(ଶୂନ୍ୟ) {
    ଯଦି (!ggml_is_numa()) {
        ଫେରନ୍ତୁ;
    }

    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    ପାଇଁ (unsigned i = 0; i < g_state.numa.total_cpus; ++i) {
        CPU_SET_S(i, setsize, cpus);
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    ଯଦି (rv) {
        fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}
#ଅନ୍ୟଥା
// TODO: Windows etc.
// (the linux implementation may also work on BSD, someone should test)
ସ୍ଥିର ଶୂନ୍ୟ set_numa_thread_affinity(ପୂର୍ଣ୍ଣସଂଖ୍ୟା thread_n) { UNUSED(thread_n);  }
ସ୍ଥିର ଶୂନ୍ୟ clear_numa_thread_affinity(ଶୂନ୍ୟ) {}
#endif

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_get_n_tasks(ସଂରଚନା ggml_tensor * ଗଣ୍ଠି, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_threads) {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_tasks = 0;

    ଯଦି (ggml_is_empty(ଗଣ୍ଠି)) {
        // no need to multi-thread a no-op
        n_tasks = 1;
        ଫେରନ୍ତୁ n_tasks;
    }

    ବାଛ (ଗଣ୍ଠି->op) {
        ସ୍ଥିତି GGML_OP_CPY:
        ସ୍ଥିତି GGML_OP_DUP:
        ସ୍ଥିତି GGML_OP_CONT:
        ସ୍ଥିତି GGML_OP_ADD:
        ସ୍ଥିତି GGML_OP_ADD_ID:
        ସ୍ଥିତି GGML_OP_ADD1:
        ସ୍ଥିତି GGML_OP_ACC:
        ସ୍ଥିତି GGML_OP_CUMSUM:
        ସ୍ଥିତି GGML_OP_TRI:
        ସ୍ଥିତି GGML_OP_FILL:
            {
                n_tasks = n_threads;
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SUB:
        ସ୍ଥିତି GGML_OP_SQR:
        ସ୍ଥିତି GGML_OP_SQRT:
        ସ୍ଥିତି GGML_OP_LOG:
        ସ୍ଥିତି GGML_OP_SIN:
        ସ୍ଥିତି GGML_OP_COS:
        ସ୍ଥିତି GGML_OP_SUM:
        ସ୍ଥିତି GGML_OP_SUM_ROWS:
        ସ୍ଥିତି GGML_OP_MEAN:
        ସ୍ଥିତି GGML_OP_ARGMAX:
            {
                n_tasks = 1;
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_ଗଣନା_EQUAL:
        ସ୍ଥିତି GGML_OP_SOLVE_TRI:
        ସ୍ଥିତି GGML_OP_GATED_DELTA_NET:
        ସ୍ଥିତି GGML_OP_DSV4_HC_COMB:
        ସ୍ଥିତି GGML_OP_DSV4_HC_PRE:
        ସ୍ଥିତି GGML_OP_DSV4_HC_POST:
        ସ୍ଥିତି GGML_OP_ଶୀର୍ଷ_K_MASK:
            {
                n_tasks = n_threads;
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_REPEAT:
        ସ୍ଥିତି GGML_OP_REPEAT_BACK:
        ସ୍ଥିତି GGML_OP_LEAKY_RELU:
            {
                n_tasks = 1;
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_UNARY:
            ବାଛ (ggml_get_unary_op(ଗଣ୍ଠି)) {
                ସ୍ଥିତି GGML_UNARY_OP_ABS:
                ସ୍ଥିତି GGML_UNARY_OP_SGN:
                ସ୍ଥିତି GGML_UNARY_OP_NEG:
                ସ୍ଥିତି GGML_UNARY_OP_STEP:
                ସ୍ଥିତି GGML_UNARY_OP_TANH:
                ସ୍ଥିତି GGML_UNARY_OP_ELU:
                ସ୍ଥିତି GGML_UNARY_OP_RELU:
                ସ୍ଥିତି GGML_UNARY_OP_SIGMOID:
                ସ୍ଥିତି GGML_UNARY_OP_HARDSWISH:
                ସ୍ଥିତି GGML_UNARY_OP_HARDSIGMOID:
                ସ୍ଥିତି GGML_UNARY_OP_EXP:
                ସ୍ଥିତି GGML_UNARY_OP_SOFTPLUS:
                ସ୍ଥିତି GGML_UNARY_OP_EXPM1:
                ସ୍ଥିତି GGML_UNARY_OP_FLOOR:
                ସ୍ଥିତି GGML_UNARY_OP_CEIL:
                ସ୍ଥିତି GGML_UNARY_OP_ROUND:
                ସ୍ଥିତି GGML_UNARY_OP_TRUNC:
                    {
                        n_tasks = 1;
                    } ଭାଙ୍ଗ;

                ସ୍ଥିତି GGML_UNARY_OP_GELU:
                ସ୍ଥିତି GGML_UNARY_OP_GELU_ERF:
                ସ୍ଥିତି GGML_UNARY_OP_GELU_QUICK:
                ସ୍ଥିତି GGML_UNARY_OP_SILU:
                ସ୍ଥିତି GGML_UNARY_OP_XIELU:
                    {
                        n_tasks = n_threads;
                    } ଭାଙ୍ଗ;
                ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                    GGML_ABORT("fatal error");
            }
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_GLU:
            ବାଛ (ggml_get_glu_op(ଗଣ୍ଠି)) {
                ସ୍ଥିତି GGML_GLU_OP_REGLU:
                ସ୍ଥିତି GGML_GLU_OP_GEGLU:
                ସ୍ଥିତି GGML_GLU_OP_SWIGLU:
                ସ୍ଥିତି GGML_GLU_OP_SWIGLU_OAI:
                ସ୍ଥିତି GGML_GLU_OP_GEGLU_ERF:
                ସ୍ଥିତି GGML_GLU_OP_GEGLU_QUICK:
                ସ୍ଥିତି GGML_GLU_OP_SWIGLU_CLAMP:
                    {
                        n_tasks = n_threads;
                    } ଭାଙ୍ଗ;
                ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                    GGML_ABORT("fatal error");
            }
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SILU_BACK:
        ସ୍ଥିତି GGML_OP_MUL:
        ସ୍ଥିତି GGML_OP_DIV:
        ସ୍ଥିତି GGML_OP_NORM:
        ସ୍ଥିତି GGML_OP_RMS_NORM:
        ସ୍ଥିତି GGML_OP_RMS_NORM_BACK:
        ସ୍ଥିତି GGML_OP_L2_NORM:
        ସ୍ଥିତି GGML_OP_GROUP_NORM:
        ସ୍ଥିତି GGML_OP_CONCAT:
        ସ୍ଥିତି GGML_OP_MUL_MAT:
        ସ୍ଥିତି GGML_OP_MUL_MAT_ID:
        ସ୍ଥିତି GGML_OP_OUT_PROD:
        ସ୍ଥିତି GGML_OP_SET_ROWS:
            {
                n_tasks = n_threads;
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_GET_ROWS:
            {
                // FIXME: get_rows can use additional threads, but the cost of launching additional threads
                // decreases performance with GPU offloading
                //n_tasks = n_threads;
                n_tasks = 1;
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SCALE:
        ସ୍ଥିତି GGML_OP_SET:
        ସ୍ଥିତି GGML_OP_RESHAPE:
        ସ୍ଥିତି GGML_OP_VIEW:
        ସ୍ଥିତି GGML_OP_PERMUTE:
        ସ୍ଥିତି GGML_OP_TRANSPOSE:
        ସ୍ଥିତି GGML_OP_GET_ROWS_BACK:
        ସ୍ଥିତି GGML_OP_DIAG:
            {
                n_tasks = 1;
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_DIAG_MASK_ZERO:
        ସ୍ଥିତି GGML_OP_DIAG_MASK_INF:
        ସ୍ଥିତି GGML_OP_SOFT_MAX_BACK:
        ସ୍ଥିତି GGML_OP_ROPE:
        ସ୍ଥିତି GGML_OP_ROPE_BACK:
        ସ୍ଥିତି GGML_OP_ADD_REL_POS:
            {
                n_tasks = n_threads;
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_CLAMP:
            {
                n_tasks = 1; //TODO
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SOFT_MAX:
            {
                n_tasks = MIN(n_threads, ggml_nrows(ଗଣ୍ଠି->src[0]));
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_IM2COL:
        ସ୍ଥିତି GGML_OP_IM2COL_BACK:
        ସ୍ଥିତି GGML_OP_IM2COL_3D:
        ସ୍ଥିତି GGML_OP_CONV_2D:
        ସ୍ଥିତି GGML_OP_CONV_3D:
        ସ୍ଥିତି GGML_OP_CONV_2D_DW:
        ସ୍ଥିତି GGML_OP_COL2IM_1D:
        ସ୍ଥିତି GGML_OP_CONV_TRANSPOSE_1D:
        ସ୍ଥିତି GGML_OP_CONV_TRANSPOSE_2D:
            {
                n_tasks = n_threads;
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_POOL_1D:
        ସ୍ଥିତି GGML_OP_POOL_2D:
        ସ୍ଥିତି GGML_OP_POOL_2D_BACK:
            {
                n_tasks = 1;
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_UPSCALE:
        ସ୍ଥିତି GGML_OP_PAD:
        ସ୍ଥିତି GGML_OP_PAD_REFLECT_1D:
        ସ୍ଥିତି GGML_OP_ROLL:
        ସ୍ଥିତି GGML_OP_ARANGE:
        ସ୍ଥିତି GGML_OP_TIMESTEP_EMBEDDING:
        ସ୍ଥିତି GGML_OP_ARGSORT:
        ସ୍ଥିତି GGML_OP_ଶୀର୍ଷ_K:
        ସ୍ଥିତି GGML_OP_FLASH_ATTN_EXT:
        ସ୍ଥିତି GGML_OP_FLASH_ATTN_BACK:
        ସ୍ଥିତି GGML_OP_SSM_CONV:
        ସ୍ଥିତି GGML_OP_SSM_SCAN:
        ସ୍ଥିତି GGML_OP_LIGHTNING_INDEXER:
            {
                n_tasks = n_threads;
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_RWKV_WKV6:
        ସ୍ଥିତି GGML_OP_GATED_LINEAR_ATTN:
        ସ୍ଥିତି GGML_OP_RWKV_WKV7:
            {
                ଅଚଳ int64_t n_heads = ଗଣ୍ଠି->src[1]->ne[1];
                n_tasks = MIN(n_threads, n_heads);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_WIN_PART:
        ସ୍ଥିତି GGML_OP_WIN_UNPART:
        ସ୍ଥିତି GGML_OP_GET_REL_POS:
            {
                n_tasks = 1;
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_MAP_CUSTOM1:
            {
                ସଂରଚନା ggml_map_custom1_op_params p;
                memcpy(&p, ଗଣ୍ଠି->op_params, ଆକାର(p));
                ଯଦି (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } ଅନ୍ୟଥା {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_MAP_CUSTOM2:
            {
                ସଂରଚନା ggml_map_custom2_op_params p;
                memcpy(&p, ଗଣ୍ଠି->op_params, ଆକାର(p));
                ଯଦି (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } ଅନ୍ୟଥା {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_MAP_CUSTOM3:
            {
                ସଂରଚନା ggml_map_custom3_op_params p;
                memcpy(&p, ଗଣ୍ଠି->op_params, ଆକାର(p));
                ଯଦି (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } ଅନ୍ୟଥା {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_CUSTOM:
            {
                ସଂରଚନା ggml_custom_op_params p;
                memcpy(&p, ଗଣ୍ଠି->op_params, ଆକାର(p));
                ଯଦି (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } ଅନ୍ୟଥା {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_CROSS_ENTROPY_LOSS:
        ସ୍ଥିତି GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        ସ୍ଥିତି GGML_OP_OPT_STEP_ADAMW:
        ସ୍ଥିତି GGML_OP_OPT_STEP_SGD:
            {
                n_tasks = n_threads;
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_NONE:
            {
                n_tasks = 1;
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_ଗଣନା:
            {
                GGML_ABORT("fatal error");
            }
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                fprintf(stderr, "%s: op not implemented: ", __func__);
                ଯଦି (ଗଣ୍ଠି->op < GGML_OP_ଗଣନା) {
                    fprintf(stderr, "%s\n", ggml_op_name(ଗଣ୍ଠି->op));
                } ଅନ୍ୟଥା {
                    fprintf(stderr, "%d\n", ଗଣ୍ଠି->op);
                }
                GGML_ABORT("fatal error");
            }
    }

    ପ୍ରତିଜ୍ଞା(n_tasks > 0);

    ଫେରନ୍ତୁ n_tasks;
}

ସ୍ଥିର thread_ret_t ggml_graph_compute_secondary_thread(ଶୂନ୍ୟ* data);

#ଯଦି defined(_WIN32)
#ଅନ୍ତର୍ଭାଵ "windows.h"

// TODO: support > 64 CPUs
ସ୍ଥିର bool ggml_thread_apply_affinity(bool * mask) {
    HANDLE    h = GetCurrentThread();
    uint64_t  bitmask = 0ULL;

    ପ୍ରତିଜ୍ଞା(GGML_MAX_N_THREADS >= 64);

    ପାଇଁ (int32_t i = 0; i < 8; i++) {
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

    ପାଇଁ (int32_t i = 64; i < GGML_MAX_N_THREADS; i++) {
        ଯଦି (mask[i]) {
            fprintf(stderr, "warn: setting thread-affinity for > 64 CPUs isn't supported on windows!\n");
            ଭାଙ୍ଗ;
        }
    }

    DWORD_PTR m = (DWORD_PTR)bitmask;

    m = SetThreadAffinityMask(h, m);

    ଫେରନ୍ତୁ m != 0;
}

ସ୍ଥିର bool ggml_thread_apply_priority(int32_t prio) {
    // Note that on Windows the Process Priority Class must be updated in order to set Thread priority.
    // This is up to the applications.
    DWORD p = THREAD_PRIORITY_NORMAL;
    ବାଛ (prio) {
        ସ୍ଥିତି GGML_SCHED_PRIO_ତଳ:      p = THREAD_PRIORITY_BELOW_NORMAL;  ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_SCHED_PRIO_NORMAL:   p = THREAD_PRIORITY_NORMAL;        ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_SCHED_PRIO_MEDIUM:   p = THREAD_PRIORITY_ABOVE_NORMAL;  ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_SCHED_PRIO_ଉପର:     p = THREAD_PRIORITY_HIGHEST;       ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_SCHED_PRIO_REALTIME: p = THREAD_PRIORITY_TIME_CRITICAL; ଭାଙ୍ଗ;
    }

    ଯଦି (prio != GGML_SCHED_PRIO_ତଳ) {
        // Tell Windows that this thread should not be throttled (needs its own CPU core).
        // Newer Windows 11 versions aggressively park (offline) CPU cores and often place
        // all our threads onto the first 4 cores which results in terrible performance with
        // n_threads > 4
        #ଯଦି _WIN32_WINNT >= 0x0602
        THREAD_POWER_THROTTLING_STATE t;
        ZeroMemory(&t, ଆକାର(t));
        t.Version     = THREAD_POWER_THROTTLING_CURRENT_VERSION;
        t.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
        t.StateMask   = 0;

        ଯଦି (!SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &t, ଆକାର(t))) {
            GGML_LOG_DEBUG("failed to disable thread power throttling %d : (%d)\n", prio, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) GetLastError());
            ଫେରନ୍ତୁ false;
        }
        #endif
    }

    ଯଦି (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        ଫେରନ୍ତୁ true;
    }

    ଯଦି (!SetThreadPriority(GetCurrentThread(), p)) {
        fprintf(stderr, "warn: failed to set thread priority %d : (%d)\n", prio, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) GetLastError());
        ଫେରନ୍ତୁ false;
    }

    ଫେରନ୍ତୁ true;
}

#ନଚେତଯଦି defined(__APPLE__)
#ଅନ୍ତର୍ଭାଵ <sys/types.h>
#ଅନ୍ତର୍ଭାଵ <sys/resource.h>

ସ୍ଥିର bool ggml_thread_apply_affinity(ଅଚଳ bool * mask) {
    // Not supported on Apple platforms
    UNUSED(mask);
    ଫେରନ୍ତୁ true;
}

ସ୍ଥିର bool ggml_thread_apply_priority(int32_t prio) {
    ସଂରଚନା sched_param p;
    int32_t policy = SCHED_OTHER;
    ବାଛ (prio) {
        // TODO: there seems to be no way to set lower prio on Apple platforms
        ସ୍ଥିତି GGML_SCHED_PRIO_ତଳ:      policy = SCHED_OTHER; p.sched_priority = 0;  ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_SCHED_PRIO_ଉପର:     policy = SCHED_FIFO;  p.sched_priority = 80; ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; ଭାଙ୍ଗ;
    }

    ଯଦି (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        ଫେରନ୍ତୁ true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    ଯଦି (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        ଫେରନ୍ତୁ false;
    }

    ଫେରନ୍ତୁ true;
}

#ନଚେତଯଦି defined(__linux__)
// TODO: this may not work on BSD, to be verified

ସ୍ଥିର bool ggml_thread_apply_affinity(ଅଚଳ bool * mask) {
    cpu_set_t cpuset;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା err;

    CPU_ZERO(&cpuset);

    ପାଇଁ (uint32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
        ଯଦି (mask[i]) {
            GGML_PRINT_DEBUG("Thread %lx: adding %d to cpuset\n", pthread_self(), i);
            CPU_SET(i, &cpuset);
        }
    }

#ifdef __ANDROID__
    err = sched_setaffinity(0, ଆକାର(cpuset), &cpuset);
    ଯଦି (err < 0) {
        err = errno;
    }
#ଅନ୍ୟଥା
    err = pthread_setaffinity_np(pthread_self(), ଆକାର(cpuset), &cpuset);
#endif
    ଯଦି (err != 0) {
        fprintf(stderr, "warn: failed to set affinity mask 0x%llx : %s (%d)\n", (unsigned long long)mask, strerror(err), err);
        ଫେରନ୍ତୁ false;
    }

    ଫେରନ୍ତୁ true;
}

ସ୍ଥିର bool ggml_thread_apply_priority(int32_t prio) {
    ସଂରଚନା sched_param p;
    int32_t policy = SCHED_OTHER;
    ବାଛ (prio) {
        ସ୍ଥିତି GGML_SCHED_PRIO_ତଳ:      policy = SCHED_BATCH; p.sched_priority = 0;  ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_SCHED_PRIO_ଉପର:     policy = SCHED_FIFO;  p.sched_priority = 80; ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; ଭାଙ୍ଗ;
    }

    ଯଦି (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        ଫେରନ୍ତୁ true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    ଯଦି (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        ଫେରନ୍ତୁ false;
    }

    ଫେରନ୍ତୁ true;
}

#ଅନ୍ୟଥା // unsupported platforms

ସ୍ଥିର bool ggml_thread_apply_affinity(ଅଚଳ bool * mask) {
    UNUSED(mask);
    ଫେରନ୍ତୁ true;
}

ସ୍ଥିର bool ggml_thread_apply_priority(int32_t prio) {
    UNUSED(prio);
    ଫେରନ୍ତୁ true;
}

#endif

ସ୍ଥିର bool ggml_thread_cpumask_is_valid(ଅଚଳ bool * mask) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_N_THREADS; i++) {
        ଯଦି (mask[i]) { ଫେରନ୍ତୁ true; }
    }
    ଫେରନ୍ତୁ false;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_thread_cpumask_next(ଅଚଳ bool * global_mask, bool * local_mask, bool strict, int32_t* iter) {
    ଯଦି (!strict) {
        memcpy(local_mask, global_mask, GGML_MAX_N_THREADS);
        ଫେରନ୍ତୁ;
    } ଅନ୍ୟଥା {
        memset(local_mask, 0, GGML_MAX_N_THREADS);
        int32_t base_idx = *iter;
        ପାଇଁ (int32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
            int32_t idx = base_idx + i;
            ଯଦି (idx >= GGML_MAX_N_THREADS) {
                // Just a cheaper modulo
                idx -= GGML_MAX_N_THREADS;
            }
            ଯଦି (global_mask[idx]) {
                local_mask[idx] = 1;
                *iter = idx + 1;
                ଫେରନ୍ତୁ;
            }
        }
    }
}

ଶୂନ୍ୟ ggml_threadpool_free(ସଂରଚନା ggml_threadpool* threadpool) {
    ଯଦି (!threadpool) ଫେରନ୍ତୁ;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_threads = threadpool->n_threads;

#ifndef GGML_USE_OPENMP
    ସଂରଚନା ggml_compute_state* workers = threadpool->workers;

    ggml_mutex_lock(&threadpool->mutex);

    threadpool->stop = true;
    threadpool->pause = false;

    ggml_cond_broadcast(&threadpool->cond);
    ggml_mutex_unlock(&threadpool->mutex);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 1; j < n_threads; j++) {
        int32_t rc = ggml_thread_join(workers[j].thrd, ଖାଲି);
        GGML_ASSERT(rc == GGML_EXIT_SUCCESS || rc == GGML_EXIT_ABORTED);
        UNUSED(rc);
    }

    ggml_mutex_destroy(&threadpool->mutex);
    ggml_cond_destroy(&threadpool->cond);
#endif // GGML_USE_OPENMP

    ଅଚଳ size_t workers_size = ଆକାର(ସଂରଚନା ggml_compute_state) * n_threads;
    ggml_aligned_free(threadpool->workers, workers_size);
    ggml_aligned_free(threadpool, ଆକାର(ସଂରଚନା ggml_threadpool));
}

#ifndef GGML_USE_OPENMP
// pause/resume must be called under mutex
ସ୍ଥିର ଶୂନ୍ୟ ggml_threadpool_pause_locked(ସଂରଚନା ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Pausing threadpool\n");
    threadpool->pause = true;
    ggml_cond_broadcast(&threadpool->cond);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_threadpool_resume_locked(ସଂରଚନା ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Resuming threadpool\n");
    threadpool->pause = false;
    ggml_cond_broadcast(&threadpool->cond);
}
#endif

ଶୂନ୍ୟ ggml_threadpool_pause(ସଂରଚନା ggml_threadpool * threadpool) {
#ifndef GGML_USE_OPENMP
    ggml_mutex_lock(&threadpool->mutex);
    ଯଦି (!threadpool->pause) {
       ggml_threadpool_pause_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#ଅନ୍ୟଥା
    UNUSED(threadpool);
#endif
}

ଶୂନ୍ୟ ggml_threadpool_resume(ସଂରଚନା ggml_threadpool * threadpool) {
#ifndef GGML_USE_OPENMP
    ggml_mutex_lock(&threadpool->mutex);
    ଯଦି (threadpool->pause) {
       ggml_threadpool_resume_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#ଅନ୍ୟଥା
    UNUSED(threadpool);
#endif
}

ସଂରଚନା ggml_cplan ggml_graph_plan(
          ଅଚଳ ସଂରଚନା ggml_cgraph * cgraph,
                               ପୂର୍ଣ୍ଣସଂଖ୍ୟା   n_threads,
            ସଂରଚନା ggml_threadpool * threadpool) {

    ଯଦି (threadpool == ଖାଲି) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
    }
    ଯଦି (n_threads <= 0) {
        n_threads = threadpool ? threadpool->n_threads : GGML_DEFAULT_N_THREADS;
    }

#ଯଦି defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
    // Emscripten without pthreads support can only use a single thread
    n_threads = 1;
#endif

#ଯଦି defined(__wasi__)
    // WASI doesn't support parallelism yet
    n_threads = 1;
#endif

    size_t work_size = 0;

    ସଂରଚନା ggml_cplan cplan;
    memset(&cplan, 0, ଆକାର(ସଂରଚନା ggml_cplan));

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା max_tasks = 1;

    // thread scheduling for the different operations + work buffer size estimation
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < cgraph->n_nodes; i++) {
        ସଂରଚନା ggml_tensor * ଗଣ୍ଠି = cgraph->nodes[i];

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_tasks = ggml_get_n_tasks(ଗଣ୍ଠି, n_threads);

        max_tasks = MAX(max_tasks, n_tasks);

        size_t cur = 0;

        ଯଦି (!ggml_cpu_extra_work_size(n_threads, ଗଣ୍ଠି, &cur)) {
            ବାଛ (ଗଣ୍ଠି->op) {
                ସ୍ଥିତି GGML_OP_CPY:
                ସ୍ଥିତି GGML_OP_DUP:
                    {
                        ଯଦି (ggml_is_quantized(ଗଣ୍ଠି->type) ||
                            // F16 -> BF16 and BF16 -> F16 copies go through intermediate F32
                            (ଗଣ୍ଠି->src[0]->type == GGML_TYPE_F16  && ଗଣ୍ଠି->src[1] && ଗଣ୍ଠି->src[1]->type == GGML_TYPE_BF16) ||
                            (ଗଣ୍ଠି->src[0]->type == GGML_TYPE_BF16 && ଗଣ୍ଠି->src[1] && ଗଣ୍ଠି->src[1]->type == GGML_TYPE_F16) ||
                            // conversion between F32 and I32
                            (ଗଣ୍ଠି->src[0]->type == GGML_TYPE_F32 && ଗଣ୍ଠି->src[1] && ଗଣ୍ଠି->src[1]->type == GGML_TYPE_I32) ||
                            (ଗଣ୍ଠି->src[0]->type == GGML_TYPE_I32 && ଗଣ୍ଠି->src[1] && ଗଣ୍ଠି->src[1]->type == GGML_TYPE_F32)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * ଗଣ୍ଠି->ne[0] * n_tasks;
                        }
                    } ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_OP_ADD:
                ସ୍ଥିତି GGML_OP_ADD_ID:
                ସ୍ଥିତି GGML_OP_ADD1:
                    {
                        ଯଦି (ggml_is_quantized(ଗଣ୍ଠି->src[0]->type)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * ଗଣ୍ଠି->src[0]->ne[0] * n_tasks;
                        }
                    } ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_OP_ACC:
                    {
                        ଯଦି (ggml_is_quantized(ଗଣ୍ଠି->src[0]->type)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * ଗଣ୍ଠି->src[1]->ne[0] * n_tasks;
                        }
                    } ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_OP_ଗଣନା_EQUAL:
                    {
                        cur = ggml_type_size(ଗଣ୍ଠି->type)*n_tasks;
                    } ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_OP_MUL_MAT:
                    {
                        ଅଚଳ ଗଣନା ggml_type vec_dot_type = type_traits_cpu[ଗଣ୍ଠି->src[0]->type].vec_dot_type;
                        ଯଦି (ଗଣ୍ଠି->src[1]->type != vec_dot_type) {
                            cur = ggml_row_size(vec_dot_type, ggml_nelements(ଗଣ୍ଠି->src[1]));
                        }
                        // Workspace for tiled (see tiled.h)
                        cur = GGML_PAD(cur, 64);
                        cur += ggml_tiled_wdata_size(n_tasks, ଗଣ୍ଠି);
                    } ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_OP_MUL_MAT_ID:
                    {
                        cur = 0;
                        ଅଚଳ ସଂରଚନା ggml_tensor * src0 = ଗଣ୍ଠି->src[0];
                        ଅଚଳ ସଂରଚନା ggml_tensor * src1 = ଗଣ୍ଠି->src[1];
                        ଅଚଳ ସଂରଚନା ggml_tensor * ids = ଗଣ୍ଠି->src[2];
                        ଅଚଳ ଗଣନା ggml_type vec_dot_type = type_traits_cpu[src0->type].vec_dot_type;
                        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_as = src0->ne[2];
                        // src1
                        ଯଦି (src1->type != vec_dot_type) {
                            cur += ggml_row_size(vec_dot_type, ggml_nelements(src1)) + ଆକାର(int64_t);
                        }
                        // matrix_row_counts
                        cur += n_as * ଆକାର(int64_t) + ଆକାର(int64_t);
                        // matrix_rows
                        cur += n_as*ids->ne[0]*ids->ne[1]*ଆକାର(ସଂରଚନା mmid_row_mapping) + ଆକାର(int64_t);
                        // atomic_current_chunk
                        cur += CACHE_ଧାଡ଼ି_SIZE*n_as + CACHE_ଧାଡ଼ି_SIZE;
                        // Workspace for tiled (see tiled.h)
                        cur = GGML_PAD(cur, 64);
                        cur += ggml_tiled_wdata_size(n_tasks, ଗଣ୍ଠି);
                    } ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_OP_OUT_PROD:
                    {
                        ଯଦି (ggml_is_quantized(ଗଣ୍ଠି->src[0]->type) ||
                            ଗଣ୍ଠି->src[0]->type == GGML_TYPE_F16) {
                            cur = ggml_type_size(GGML_TYPE_F32) * ଗଣ୍ଠି->src[0]->ne[0] * n_tasks;
                        }
                    } ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_OP_SET_ROWS:
                    {
                        ଯଦି (ଗଣ୍ଠି->src[0]->type == GGML_TYPE_F16 && ଗଣ୍ଠି->type != GGML_TYPE_F16) {
                            cur = ggml_type_size(GGML_TYPE_F32) * ଗଣ୍ଠି->src[0]->ne[0] * n_tasks;
                        }
                    } ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_OP_SOFT_MAX:
                ସ୍ଥିତି GGML_OP_ROPE:
                ସ୍ଥିତି GGML_OP_ROPE_BACK:
                    {
                        cur = ggml_type_size(GGML_TYPE_F32) * ଗଣ୍ଠି->ne[0] * n_tasks;
                    } ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_OP_CONV_TRANSPOSE_1D:
                    {
                        GGML_ASSERT(ଗଣ୍ଠି->src[0]->ne[3] == 1);
                        GGML_ASSERT(ଗଣ୍ଠି->src[1]->ne[2] == 1);
                        GGML_ASSERT(ଗଣ୍ଠି->src[1]->ne[3] == 1);

                        ଅଚଳ int64_t ne00 = ଗଣ୍ଠି->src[0]->ne[0];  // K
                        ଅଚଳ int64_t ne01 = ଗଣ୍ଠି->src[0]->ne[1];  // Cout
                        ଅଚଳ int64_t ne02 = ଗଣ୍ଠି->src[0]->ne[2];  // Cin
                        ଅଚଳ int64_t ne10 = ଗଣ୍ଠି->src[1]->ne[0];  // L
                        ଅଚଳ int64_t ne11 = ଗଣ୍ଠି->src[1]->ne[1];  // Cin

                        ଯଦି ((ଗଣ୍ଠି->src[0]->type == GGML_TYPE_F16 ||
                             ଗଣ୍ଠି->src[0]->type == GGML_TYPE_BF16) &&
                            ଗଣ୍ଠି->src[1]->type == GGML_TYPE_F32) {
                            cur += ଆକାର(ggml_fp16_t)*ne00*ne01*ne02;
                            cur += ଆକାର(ggml_fp16_t)*ne10*ne11;
                        } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି->src[0]->type == GGML_TYPE_F32 &&
                                   ଗଣ୍ଠି->src[1]->type == GGML_TYPE_F32) {
                            cur += ଆକାର(ଦଶମିକ)*ne00*ne01*ne02;
                            cur += ଆକାର(ଦଶମିକ)*ne10*ne11;
                        } ଅନ୍ୟଥା {
                            GGML_ABORT("fatal error");
                        }
                    } ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_OP_CONV_2D:
                ସ୍ଥିତି GGML_OP_CONV_3D:
                    {
                        cur = GGML_IM2COL_WORK_SIZE;
                    } ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_OP_CONV_TRANSPOSE_2D:
                    {
                        ଅଚଳ int64_t ne00 = ଗଣ୍ଠି->src[0]->ne[0]; // W
                        ଅଚଳ int64_t ne01 = ଗଣ୍ଠି->src[0]->ne[1]; // H
                        ଅଚଳ int64_t ne02 = ଗଣ୍ଠି->src[0]->ne[2]; // Channels Out
                        ଅଚଳ int64_t ne03 = ଗଣ୍ଠି->src[0]->ne[3]; // Channels In

                        ଅଚଳ int64_t ne10 = ଗଣ୍ଠି->src[1]->ne[0]; // W
                        ଅଚଳ int64_t ne11 = ଗଣ୍ଠି->src[1]->ne[1]; // H
                        ଅଚଳ int64_t ne12 = ଗଣ୍ଠି->src[1]->ne[2]; // Channels In
                        ଅଚଳ int64_t ne13 = ଗଣ୍ଠି->src[1]->ne[3]; // Batch

                        GGML_ASSERT(ଗଣ୍ଠି->src[0]->type == GGML_TYPE_F16 || ଗଣ୍ଠି->src[0]->type == GGML_TYPE_F32);
                        GGML_ASSERT(ଗଣ୍ଠି->src[1]->type == GGML_TYPE_F32);

                        cur += ggml_type_size(ଗଣ୍ଠି->src[0]->type) * ne00 * ne01 * ne02 * ne03;
                        cur += ggml_type_size(ଗଣ୍ଠି->src[0]->type) * ne10 * ne11 * ne12 * ne13;

                    } ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_OP_ଶୀର୍ଷ_K:
                    {
                        cur += ଆକାର(int32_t)*ଗଣ୍ଠି->src[0]->ne[0]*n_tasks;
                    } ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_OP_FLASH_ATTN_EXT:
                    {
                        ଅଚଳ int64_t neq2 = ଗଣ୍ଠି->src[0]->ne[2]; // number of query heads
                        ଅଚଳ int64_t DK = ଗଣ୍ଠି->src[1]->ne[0];
                        ଅଚଳ int64_t DV = ଗଣ୍ଠି->src[2]->ne[0];

                        // Tiled flash attention scratch (tile sizes defined in common.h)
                        // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
                        size_t prefill  = ଆକାର(ଦଶମିକ)*(GGML_FA_TILE_Q*DK + 2*GGML_FA_TILE_Q*GGML_FA_TILE_KV + GGML_FA_TILE_Q*DV + GGML_FA_TILE_KV*DV + GGML_FA_TILE_KV*DK)*n_tasks;

                        // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
                        // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
                        size_t n_chunks = n_tasks;
                        size_t decode   = ଆକାର(ଦଶମିକ)*(neq2*n_chunks*(2+DV) + n_tasks*(DK + 2*DV));

                        cur += MAX(prefill, decode);
                    } ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_OP_FLASH_ATTN_BACK:
                    {
                        ଅଚଳ int64_t    D = ଗଣ୍ଠି->src[0]->ne[0];
                        ଅଚଳ int64_t ne11 = ggml_up(ଗଣ୍ଠି->src[1]->ne[1], GGML_SOFT_MAX_UNROLL);
                        ଅଚଳ int64_t mxDn = MAX(D, ne11) * 2; // *2 because of S and SM in ggml_compute_forward_flash_attn_back
                        ଯଦି (ଗଣ୍ଠି->src[1]->type == GGML_TYPE_F32) {
                            cur  = ଆକାର(ଦଶମିକ)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += ଆକାର(ଦଶମିକ)*mxDn*n_tasks; // this is overestimated by x2
                        } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି->src[1]->type == GGML_TYPE_F16) {
                            cur  = ଆକାର(ଦଶମିକ)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += ଆକାର(ଦଶମିକ)*mxDn*n_tasks; // this is overestimated by x2
                        } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି->src[1]->type == GGML_TYPE_BF16) {
                            cur  = ଆକାର(ଦଶମିକ)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += ଆକାର(ଦଶମିକ)*mxDn*n_tasks; // this is overestimated by x2
                        }
                    } ଭାଙ୍ଗ;

                ସ୍ଥିତି GGML_OP_CROSS_ENTROPY_LOSS:
                    {
                        cur = ggml_type_size(ଗଣ୍ଠି->type)*(n_tasks + ଗଣ୍ଠି->src[0]->ne[0]*n_tasks);
                    } ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_OP_GATED_DELTA_NET:
                    {
                        ଅଚଳ int64_t S_v = ଗଣ୍ଠି->src[2]->ne[0];
                        ଅଚଳ int64_t K   = ggml_get_op_params_i32(ଗଣ୍ଠି, 0);
                        ଅଚଳ int64_t per_thread = S_v + (K > 1 ? S_v * S_v : 0);
                        cur = per_thread * ଆକାର(ଦଶମିକ) * n_tasks;
                    } ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_OP_ଗଣନା:
                    {
                        GGML_ABORT("fatal error");
                    }
                ସ୍ଥିତି GGML_OP_LIGHTNING_INDEXER:
                    {
                        // temp buffer for dequantizing lightning indexer keys
                        ଅଚଳ int64_t ne10 = ଗଣ୍ଠି->src[1]->ne[0];
                        cur += ଆକାର(ଦଶମିକ)*ne10*n_tasks;
                    } ଭାଙ୍ଗ;
                ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                    ଭାଙ୍ଗ;
            }
        }

        work_size = MAX(work_size, cur);
    }

    ଯଦି (work_size > 0) {
        work_size += CACHE_ଧାଡ଼ି_SIZE*(n_threads);
    }

    cplan.threadpool = threadpool;
    cplan.n_threads  = MIN(max_tasks, n_threads);
    cplan.work_size  = work_size;
    cplan.work_data  = ଖାଲି;

    ଫେରନ୍ତୁ cplan;
}


// Try to fuse the current node with subsequent nodes for better performance.
// Returns the number of nodes skipped by fusion (>=1), or 0 if no fusion was applied.
ସ୍ଥିର bool ggml_cpu_disable_fusion = false;  // initialized once in ggml_cpu_init(), read-only afterwards

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_cpu_try_fuse_ops(
        ଅଚଳ ସଂରଚନା ggml_cgraph * cgraph,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_n,
        ଅଚଳ ସଂରଚନା ggml_compute_params * params,
        ଅଚଳ ସଂରଚନା ggml_cplan * cplan) {

    ଯଦି (ggml_cpu_disable_fusion || cplan->use_ref) {
        ଫେରନ୍ତୁ 0;
    }

    ସଂରଚନା ggml_tensor * ଗଣ୍ଠି = cgraph->nodes[ଗଣ୍ଠି_n];

    ଯଦି (ଗଣ୍ଠି->op == GGML_OP_RMS_NORM) {
        // RMS_NORM + MUL fusion
        ଅଚଳ ଗଣନା ggml_op fuse_ops[] = { GGML_OP_RMS_NORM, GGML_OP_MUL };
        ଯଦି (ggml_can_fuse(cgraph, ଗଣ୍ଠି_n, fuse_ops, 2)) {
            ସଂରଚନା ggml_tensor * mul_ଗଣ୍ଠି = cgraph->nodes[ଗଣ୍ଠି_n + 1];
            ଅଚଳ ସଂରଚନା ggml_tensor * mul_w = (mul_ଗଣ୍ଠି->src[0] == ଗଣ୍ଠି)
                ? mul_ଗଣ୍ଠି->src[1] : mul_ଗଣ୍ଠି->src[0];
            ଯଦି (ଗଣ୍ଠି->src[0]->type  == GGML_TYPE_F32 &&
                mul_ଗଣ୍ଠି->type      == GGML_TYPE_F32 &&
                mul_w->type         == GGML_TYPE_F32 &&
                mul_w->ne[0]        == ଗଣ୍ଠି->ne[0]   &&
                mul_w->nb[0]        == ଆକାର(ଦଶମିକ)) {

                ggml_compute_forward_rms_norm_mul_fused(params, ଗଣ୍ଠି, mul_ଗଣ୍ଠି);
                ଫେରନ୍ତୁ 1;
            }
        }
    }

    ଫେରନ୍ତୁ 0;
}

ସ୍ଥିର thread_ret_t ggml_graph_compute_thread(ଶୂନ୍ୟ * data) {
    ସଂରଚନା ggml_compute_state * state = (ସଂରଚନା ggml_compute_state *) data;
    ସଂରଚନା ggml_threadpool    * tp    = state->threadpool;

    ଅଚଳ ସଂରଚନା ggml_cgraph * cgraph = tp->cgraph;
    ଅଚଳ ସଂରଚନା ggml_cplan  * cplan  = tp->cplan;

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(state->ith);
#ଅନ୍ୟଥା
    set_numa_thread_affinity(state->ith);
#endif

    ସଂରଚନା ggml_compute_params params = {
        /*.ith        =*/ state->ith,
        /*.nth        =*/ atomic_load_explicit(&tp->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK,
        /*.wsize      =*/ cplan->work_size,
        /*.wdata      =*/ cplan->work_data,
        /*.threadpool =*/ tp,
        /*.use_ref    =*/ cplan->use_ref,
    };

#ifdef GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p\n", state->ith, (ଅଚଳ ଶୂନ୍ୟ *)cplan);
#ଅନ୍ୟଥା
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p last-graph %d\n", state->ith, (ଅଚଳ ଶୂନ୍ୟ *)cplan, state->last_graph);
#endif

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_n = 0; ଗଣ୍ଠି_n < cgraph->n_nodes && atomic_load_explicit(&tp->abort, memory_order_relaxed) != ଗଣ୍ଠି_n; ଗଣ୍ଠି_n++) {
        ସଂରଚନା ggml_tensor * ଗଣ୍ଠି = cgraph->nodes[ଗଣ୍ଠି_n];

        ଯଦି (ggml_op_is_empty(ଗଣ୍ଠି->op)) {
            // skip NOPs
            ଜାରି;
        }

        ଯଦି ((ଗଣ୍ଠି->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ଜାରି;
        }

        // TODO: move fused-op detection into ggml_graph_plan so fusion decisions are made once at planning time
        // Try fused ops, fall back to normal compute
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_fused = ggml_cpu_try_fuse_ops(cgraph, ଗଣ୍ଠି_n, &params, cplan);
        ଯଦି (n_fused > 0) {
            ଗଣ୍ଠି_n += n_fused;
        } ଅନ୍ୟଥା {
            ggml_compute_forward(&params, ଗଣ୍ଠି);
        }

        ଯଦି (state->ith == 0 && cplan->abort_callback &&
                cplan->abort_callback(cplan->abort_callback_data)) {
            atomic_store_explicit(&tp->abort, ଗଣ୍ଠି_n + 1, memory_order_relaxed);
            tp->ec    = GGML_STATUS_ABORTED;
        }

        ଯଦି (ଗଣ୍ଠି_n + 1 < cgraph->n_nodes) {
            ggml_barrier(state->threadpool);
        }
    }

#ifdef GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p\n", state->ith, (ଅଚଳ ଶୂନ୍ୟ *)cplan);
#ଅନ୍ୟଥା
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p last-graph %d\n", state->ith, (ଅଚଳ ଶୂନ୍ୟ *)cplan, state->last_graph);
#endif

    ggml_barrier(state->threadpool);

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(state->ith);
#endif

    ଫେରନ୍ତୁ 0;
}

#ifndef GGML_USE_OPENMP

// check if thread is ready to proceed (exit from polling or sleeping)
// returns true if loops should exit, sets state->pending to indicate new work
ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool ggml_graph_compute_thread_ready(ସଂରଚନା ggml_compute_state * state) {
    ସଂରଚନା ggml_threadpool * threadpool = state->threadpool;

    ଯଦି (state->pending || threadpool->stop || threadpool->pause) { ଫେରନ୍ତୁ true; }

    // check for new graph/work
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_graph   = atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed);
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_threads = n_graph & GGML_THREADPOOL_N_THREADS_MASK;
    ଯଦି (n_graph != state->last_graph) {
        state->pending    = (state->ith < n_threads);
        state->last_graph = n_graph;
        ଫେରନ୍ତୁ true;
    }

    ଫେରନ୍ତୁ false;
}

// sync thread state after polling
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_graph_compute_thread_sync(ସଂରଚନା ggml_compute_state * state) {
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&state->threadpool->n_graph, 0, memory_order_seq_cst);
    #ଅନ୍ୟଥା
    atomic_thread_fence(memory_order_seq_cst);
    #endif
    UNUSED(state);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool ggml_graph_compute_poll_for_work(ସଂରଚନା ggml_compute_state * state) {
    ସଂରଚନା ggml_threadpool * threadpool = state->threadpool;

    // This seems to make 0 ... 100 a decent range for polling level across modern processors.
    // Perhaps, we can adjust it dynamically based on load and things.
    ଅଚଳ uint64_t n_rounds = 1024UL * 128 * threadpool->poll;

    ପାଇଁ (uint64_t i=0; !ggml_graph_compute_thread_ready(state) && i < n_rounds; i++) {
        // No new work. Keep polling.
        ggml_thread_cpu_relax();
    }

    ଫେରନ୍ତୁ state->pending;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool ggml_graph_compute_check_for_work(ସଂରଚନା ggml_compute_state * state) {
    ସଂରଚନା ggml_threadpool * threadpool = state->threadpool;

    ଯଦି (ggml_graph_compute_poll_for_work(state)) {
        ggml_graph_compute_thread_sync(state);
        ଫେରନ୍ତୁ state->pending;
    }

    ggml_mutex_lock_shared(&threadpool->mutex);
    ଯେତେବେଳେ (!ggml_graph_compute_thread_ready(state)) {
        // No new work. Wait for the signal.
        GGML_PRINT_DEBUG("thread #%d waiting for work (sleeping)\n", state->ith);
        ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
    }
    ggml_mutex_unlock_shared(&threadpool->mutex);

    ଫେରନ୍ତୁ state->pending;
}

ସ୍ଥିର thread_ret_t ggml_graph_compute_secondary_thread(ଶୂନ୍ୟ* data) {
    ସଂରଚନା ggml_compute_state * state = (ସଂରଚନା ggml_compute_state *) data;
    ସଂରଚନା ggml_threadpool * threadpool = state->threadpool;

    ggml_thread_apply_priority(threadpool->prio);
    ଯଦି (ggml_thread_cpumask_is_valid(state->cpumask)) {
        ggml_thread_apply_affinity(state->cpumask);
    }

    ଯେତେବେଳେ (true) {
        // Check if we need to sleep
        ଯେତେବେଳେ (threadpool->pause) {
            GGML_PRINT_DEBUG("thread #%d inside pause loop\n", state->ith);
            ggml_mutex_lock_shared(&threadpool->mutex);
            ଯଦି (threadpool->pause) {
                ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
            }
            GGML_PRINT_DEBUG("thread #%d resuming after wait\n", state->ith);
            ggml_mutex_unlock_shared(&threadpool->mutex);
        }

        // This needs to be checked for after the cond_wait
        ଯଦି (threadpool->stop) ଭାଙ୍ଗ;

        // Check if there is new work
        // The main thread is the only one that can dispatch new work

        ggml_graph_compute_check_for_work(state);
        ଯଦି (state->pending) {
            state->pending = false;
            ggml_graph_compute_thread(state);
        }
    }

    ଫେରନ୍ତୁ (thread_ret_t) 0;
}

// Start processing new graph
ସ୍ଥିର ଶୂନ୍ୟ ggml_graph_compute_kickoff(ସଂରଚନା ggml_threadpool * threadpool, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_threads)
{
    // Always take the mutex here because the worker threads are doing hybrid poll/wait

    ggml_mutex_lock(&threadpool->mutex);

    // Update the number of active threads and the graph count
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_graph = atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed) >> GGML_THREADPOOL_N_THREADS_BITS;
    n_graph = ((n_graph + 1) << GGML_THREADPOOL_N_THREADS_BITS) | (n_threads & GGML_THREADPOOL_N_THREADS_MASK);

    GGML_PRINT_DEBUG("compute-kickoff: n_threads %d n_graph %d\n", n_threads, n_graph);

    // Indicate the graph is ready to be processed
    // We need the full seq-cst fence here because of the polling threads (used in thread_sync)
    atomic_store_explicit(&threadpool->n_graph, n_graph, memory_order_seq_cst);

    ଯଦି (threadpool->pause) {
       // Update main thread prio and affinity to match the threadpool settings
       ggml_thread_apply_priority(threadpool->prio);
       ଯଦି (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
           ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
       }

       // resume does cond broadcast
       ggml_threadpool_resume_locked(threadpool);
    } ଅନ୍ୟଥା {
       ggml_cond_broadcast(&threadpool->cond);
    }

    ggml_mutex_unlock(&threadpool->mutex);
}

#endif // GGML_USE_OPENMP

ସ୍ଥିର ସଂରଚନା ggml_threadpool * ggml_threadpool_new_impl(
    ସଂରଚନା ggml_threadpool_params * tpp,
               ସଂରଚନା ggml_cgraph * cgraph,
                ସଂରଚନା ggml_cplan * cplan) {

    ସଂରଚନା ggml_threadpool * threadpool =
        ggml_aligned_malloc(ଆକାର(ସଂରଚନା ggml_threadpool));
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
        threadpool->workers          = ଖାଲି;
        threadpool->n_threads        = tpp->n_threads;
        threadpool->poll             = tpp->poll;
        threadpool->prio             = tpp->prio;
        threadpool->ec               = GGML_STATUS_SUCCESS;
    }

    // Allocate and init workers state
    ଅଚଳ size_t workers_size = ଆକାର(ସଂରଚନା ggml_compute_state) * tpp->n_threads;
    ସଂରଚନା ggml_compute_state * workers = ggml_aligned_malloc(workers_size);

    memset(workers, 0, workers_size);
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < tpp->n_threads; j++) {
        workers[j].threadpool = threadpool;
        workers[j].ith        = j;
    }

    threadpool->workers = workers;

#ifdef GGML_USE_OPENMP
    int32_t cpumask_iter = 0;

    // Compute CPU masks for each thread
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < tpp->n_threads; j++) {
        ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);
    }
#ଅନ୍ୟଥା // GGML_USE_OPENMP
    ggml_mutex_init(&threadpool->mutex);
    ggml_cond_init(&threadpool->cond);

    // Spin the threads for all workers, and update CPU placements.
    // Place the main thread last (towards the higher numbered CPU cores).

    int32_t cpumask_iter = 0;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 1; j < tpp->n_threads; j++) {
        ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);

        int32_t rc = ggml_thread_create(&workers[j].thrd, ଖାଲି, ggml_graph_compute_secondary_thread, &workers[j]);
        GGML_ASSERT(rc == 0);
    }

    ggml_thread_cpumask_next(tpp->cpumask, workers[0].cpumask, tpp->strict_cpu, &cpumask_iter);

    ଯଦି (!threadpool->pause) {
        // Update main thread prio and affinity at the start, otherwise we'll do it in resume
        ggml_thread_apply_priority(threadpool->prio);
        ଯଦି (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
            ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
        }
    }
#endif // GGML_USE_OPENMP

    ଫେରନ୍ତୁ threadpool;
}

ସଂରଚନା ggml_threadpool * ggml_threadpool_new(ସଂରଚନା ggml_threadpool_params * tpp) {
    ଫେରନ୍ତୁ ggml_threadpool_new_impl(tpp, ଖାଲି, ଖାଲି);
}

ଗଣନା ggml_status ggml_graph_compute(ସଂରଚନା ggml_cgraph * cgraph, ସଂରଚନା ggml_cplan * cplan) {
    ggml_cpu_init();

    GGML_ASSERT(cplan);
    GGML_ASSERT(cplan->n_threads > 0);
    GGML_ASSERT(cplan->work_size == 0 || cplan->work_data != ଖାଲି);

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_threads                               = cplan->n_threads;
    ସଂରଚନା ggml_threadpool * threadpool = cplan->threadpool;

    bool disposable_threadpool = false;

    ଯଦି (threadpool == ଖାଲି) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
        disposable_threadpool = true;

        ସଂରଚନା ggml_threadpool_params ttp = ggml_threadpool_params_default(n_threads);
        threadpool = ggml_threadpool_new_impl(&ttp, cgraph, cplan);
    } ଅନ୍ୟଥା {
        // Reset some of the parameters that need resetting
        // No worker threads should be accessing the parameters below at this stage
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->current_chunk    = 0;
        threadpool->abort            = -1;
        threadpool->ec               = GGML_STATUS_SUCCESS;
    }

#ifdef GGML_USE_OPENMP
    ଯଦି (n_threads > 1) {
        #pragma omp parallel ଅଙ୍କ_threads(n_threads)
        {
            #pragma omp single
            {
                // update the number of threads from the actual number of threads that we got from OpenMP
                n_threads = omp_get_ଅଙ୍କ_threads();
                atomic_store_explicit(&threadpool->n_graph, n_threads, memory_order_relaxed);
            }

            // Apply thread CPU mask and priority
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = omp_get_thread_ଅଙ୍କ();

            ggml_thread_apply_priority(threadpool->prio);
            ଯଦି (ggml_thread_cpumask_is_valid(threadpool->workers[ith].cpumask)) {
                ggml_thread_apply_affinity(threadpool->workers[ith].cpumask);
            }
            ggml_graph_compute_thread(&threadpool->workers[ith]);
        }
    } ଅନ୍ୟଥା {
        atomic_store_explicit(&threadpool->n_graph, 1, memory_order_relaxed);
        ggml_graph_compute_thread(&threadpool->workers[0]);
    }
#ଅନ୍ୟଥା
    ଯଦି (n_threads > threadpool->n_threads) {
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

    ଗଣନା ggml_status ret = threadpool->ec;

    ଯଦି (disposable_threadpool) {
        ggml_threadpool_free(threadpool);
    }

    ଫେରନ୍ତୁ ret;
}

ଗଣନା ggml_status ggml_graph_compute_with_ctx(ସଂରଚନା ggml_context * ctx, ସଂରଚନା ggml_cgraph * cgraph, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_threads) {
    ସଂରଚନା ggml_cplan cplan = ggml_graph_plan(cgraph, n_threads, ଖାଲି);

    cplan.work_data = (uint8_t *)ggml_new_buffer(ctx, cplan.work_size);

    ଫେରନ୍ତୁ ggml_graph_compute(cgraph, &cplan);
}

ଶୂନ୍ୟ ggml_cpu_fp32_to_fp32(ଅଚଳ ଦଶମିକ * x, ଦଶମିକ * y, int64_t n) {
    memcpy(y, x, n * ଆକାର(ଦଶମିକ));
}

ଶୂନ୍ୟ ggml_cpu_fp32_to_fp16(ଅଚଳ ଦଶମିକ * x, ggml_fp16_t * y, int64_t n) {
    int64_t i = 0;
#ଯଦି defined(__F16C__)
#ଯଦି defined(__AVX512F__)
    ପାଇଁ (; i + 15 < n; i += 16) {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        __m256i y_vec = _mm512_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256((__m256i *)(y + i), y_vec);
    }
#endif
    ପାଇଁ (; i + 7 < n; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m128i y_vec = _mm256_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i *)(y + i), y_vec);
    }
    ପାଇଁ (; i + 3 < n; i += 4) {
        __m128 x_vec = _mm_loadu_ps(x + i);
        __m128i y_vec = _mm_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storel_epi64((__m128i *)(y + i), y_vec);
    }
#ନଚେତଯଦି defined(__riscv_zvfh)
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା vl; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(&x[i], vl);
        vfloat16m1_t vy = __riscv_vfncvt_f_f_w_f16m1(vx, vl);
        __riscv_vse16_v_f16m1((_Float16 *)&y[i], vy, vl);
    }
#ନଚେତଯଦି defined(__VXE__) || defined(__VXE2__)
    ପାଇଁ (; i + 7 < n; i += 8) {
        ଅଚଳ uint32x4_t v_yl = __lzs_f32cx4_to_f16(vec_xl(0, x + i + 0));
        ଅଚଳ uint32x4_t v_yh = __lzs_f32cx4_to_f16(vec_xl(0, x + i + 4));
        vec_xst(vec_pack(v_yl, v_yh), 0, (uint16_t *)(y + i));
    }
#endif
    ପାଇଁ (; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(x[i]);
    }
}

ଶୂନ୍ୟ ggml_cpu_fp16_to_fp32(ଅଚଳ ggml_fp16_t * x, ଦଶମିକ * y, int64_t n) {
    int64_t i = 0;
#ଯଦି defined(__F16C__)
#ଯଦି defined(__AVX512F__)
    ପାଇଁ (; i + 15 < n; i += 16) {
        __m256i x_vec = _mm256_loadu_si256((ଅଚଳ __m256i *)(x + i));
        __m512 y_vec = _mm512_cvtph_ps(x_vec);
        _mm512_storeu_ps(y + i, y_vec);
    }
#endif
    ପାଇଁ (; i + 7 < n; i += 8) {
        __m128i x_vec = _mm_loadu_si128((ଅଚଳ __m128i *)(x + i));
        __m256 y_vec = _mm256_cvtph_ps(x_vec);
        _mm256_storeu_ps(y + i, y_vec);
    }
    ପାଇଁ (; i + 3 < n; i += 4) {
        __m128i x_vec = _mm_loadl_epi64((ଅଚଳ __m128i *)(x + i));
        __m128 y_vec = _mm_cvtph_ps(x_vec);
        _mm_storeu_ps(y + i, y_vec);
    }

#ନଚେତଯଦି defined(__riscv_v_intrinsic) && defined(__riscv_zvfhmin)
    // calculate step size
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା epr = __riscv_vsetvlmax_e16m2();
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା step = epr * 2;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା np = (n & ~(step - 1));

    // unroll by 2
    ପାଇଁ (; i < np; i += step) {
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((ଅଚଳ _Float16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vfloat16m2_t ax1 = __riscv_vle16_v_f16m2((ଅଚଳ _Float16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvt_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା vl;
    ପାଇଁ (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((ଅଚଳ _Float16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }

#endif

    ପାଇଁ (; i < n; ++i) {
        y[i] = GGML_CPU_FP16_TO_FP32(x[i]);
    }
}

ଶୂନ୍ୟ ggml_cpu_fp32_to_bf16(ଅଚଳ ଦଶମିକ * x, ggml_bf16_t * y, int64_t n) {
    int64_t i = 0;
    ପାଇଁ (; i < n; ++i) {
        y[i] = GGML_FP32_TO_BF16(x[i]);
    }
}

ଶୂନ୍ୟ ggml_cpu_fp32_to_i32(ଅଚଳ ଦଶମିକ * x, int32_t * y, int64_t n) {
    int64_t i = 0;
    ପାଇଁ (; i < n; ++i) {
        y[i] = x[i];
    }
}

ଶୂନ୍ୟ ggml_cpu_bf16_to_fp32(ଅଚଳ ggml_bf16_t * x, ଦଶମିକ * y, int64_t n) {
    int64_t i = 0;
#ଯଦି defined(__AVX2__)
#ଯଦି defined(__AVX512F__)
    ପାଇଁ (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i,
                        _mm512_castsi512_ps(
                            _mm512_slli_epi32(
                                _mm512_cvtepu16_epi32(
                                    _mm256_loadu_si256(
                                        (ଅଚଳ __m256i *)(x + i))),
                                16)));
    }
#endif
    ପାଇଁ (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i,
                        _mm256_castsi256_ps(
                            _mm256_slli_epi32(
                                _mm256_cvtepu16_epi32(
                                    _mm_loadu_si128(
                                        (ଅଚଳ __m128i *)(x + i))),
                                16)));
    }
#ନଚେତଯଦି defined(__riscv_v_intrinsic) && defined(__riscv_zvfbfmin)
    // calculate step size
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା epr = __riscv_vsetvlmax_e16m2();
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା step = epr * 2;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା np = (n & ~(step - 1));

    // unroll by 2
    ପାଇଁ (; i < np; i += step) {
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((ଅଚଳ __bf16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vbfloat16m2_t ax1 = __riscv_vle16_v_bf16m2((ଅଚଳ __bf16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା vl;
    ପାଇଁ (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((ଅଚଳ __bf16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }
#endif
    ପାଇଁ (; i < n; i++) {
        y[i] = GGML_BF16_TO_FP32(x[i]);
    }
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_cpu_has_avx(ଶୂନ୍ୟ) {
#ଯଦି defined(__AVX__)
    ଫେରନ୍ତୁ 1;
#ଅନ୍ୟଥା
    ଫେରନ୍ତୁ 0;
#endif
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_cpu_has_avx_vnni(ଶୂନ୍ୟ) {
#ଯଦି defined(__AVXVNNI__)
    ଫେରନ୍ତୁ 1;
#ଅନ୍ୟଥା
    ଫେରନ୍ତୁ 0;
#endif
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_cpu_has_avx2(ଶୂନ୍ୟ) {
#ଯଦି defined(__AVX2__)
    ଫେରନ୍ତୁ 1;
#ଅନ୍ୟଥା
    ଫେରନ୍ତୁ 0;
#endif
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_cpu_has_avx512(ଶୂନ୍ୟ) {
#ଯଦି defined(__AVX512F__)
    ଫେରନ୍ତୁ 1;
#ଅନ୍ୟଥା
    ଫେରନ୍ତୁ 0;
#endif
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_cpu_has_avx512_vbmi(ଶୂନ୍ୟ) {
#ଯଦି defined(__AVX512VBMI__)
    ଫେରନ୍ତୁ 1;
#ଅନ୍ୟଥା
    ଫେରନ୍ତୁ 0;
#endif
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_cpu_has_avx512_vnni(ଶୂନ୍ୟ) {
#ଯଦି defined(__AVX512VNNI__)
    ଫେରନ୍ତୁ 1;
#ଅନ୍ୟଥା
    ଫେରନ୍ତୁ 0;
#endif
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_cpu_has_avx512_bf16(ଶୂନ୍ୟ) {
#ଯଦି defined(__AVX512BF16__)
    ଫେରନ୍ତୁ 1;
#ଅନ୍ୟଥା
    ଫେରନ୍ତୁ 0;
#endif
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_cpu_has_amx_int8(ଶୂନ୍ୟ) {
#ଯଦି defined(__AMX_INT8__)
    ଫେରନ୍ତୁ 1;
#ଅନ୍ୟଥା
    ଫେରନ୍ତୁ 0;
#endif
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_cpu_has_bmi2(ଶୂନ୍ୟ) {
#ଯଦି defined(__BMI2__)
    ଫେରନ୍ତୁ 1;
#ଅନ୍ୟଥା
    ଫେରନ୍ତୁ 0;
#endif
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_cpu_has_fma(ଶୂନ୍ୟ) {
#ଯଦି defined(__FMA__)
    ଫେରନ୍ତୁ 1;
#ଅନ୍ୟଥା
    ଫେରନ୍ତୁ 0;
#endif
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_cpu_has_arm_fma(ଶୂନ୍ୟ) {
#ଯଦି defined(__ARM_FEATURE_FMA)
    ଫେରନ୍ତୁ 1;
#ଅନ୍ୟଥା
    ଫେରନ୍ତୁ 0;
#endif
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_cpu_has_riscv_v(ଶୂନ୍ୟ) {
#ଯଦି defined(__riscv_v_intrinsic)
    ଫେରନ୍ତୁ 1;
#ଅନ୍ୟଥା
    ଫେରନ୍ତୁ 0;
#endif
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_cpu_get_rvv_vlen(ଶୂନ୍ୟ) {
#ଯଦି defined(__riscv) && defined(__riscv_v_intrinsic)
    ଫେରନ୍ତୁ ggml_riscv_arch_features.rvv_vlen;
#ଅନ୍ୟଥା
    ଫେରନ୍ତୁ 0;
#endif
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_cpu_has_f16c(ଶୂନ୍ୟ) {
#ଯଦି defined(__F16C__)
    ଫେରନ୍ତୁ 1;
#ଅନ୍ୟଥା
    ଫେରନ୍ତୁ 0;
#endif
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_cpu_has_fp16_va(ଶୂନ୍ୟ) {
#ଯଦି defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    ଫେରନ୍ତୁ 1;
#ଅନ୍ୟଥା
    ଫେରନ୍ତୁ 0;
#endif
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_cpu_has_wasm_simd(ଶୂନ୍ୟ) {
#ଯଦି defined(__wasm_simd128__)
    ଫେରନ୍ତୁ 1;
#ଅନ୍ୟଥା
    ଫେରନ୍ତୁ 0;
#endif
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_cpu_has_llamafile(ଶୂନ୍ୟ) {
#ଯଦି defined(GGML_USE_LLAMAFILE)
    ଫେରନ୍ତୁ 1;
#ଅନ୍ୟଥା
    ଫେରନ୍ତୁ 0;
#endif
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_cpu_has_sse3(ଶୂନ୍ୟ) {
#ଯଦି defined(__SSE3__)
    ଫେରନ୍ତୁ 1;
#ଅନ୍ୟଥା
    ଫେରନ୍ତୁ 0;
#endif
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_cpu_has_ssse3(ଶୂନ୍ୟ) {
#ଯଦି defined(__SSSE3__)
    ଫେରନ୍ତୁ 1;
#ଅନ୍ୟଥା
    ଫେରନ୍ତୁ 0;
#endif
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_cpu_has_vsx(ଶୂନ୍ୟ) {
#ଯଦି defined(__POWER9_VECTOR__)
    ଫେରନ୍ତୁ 1;
#ଅନ୍ୟଥା
    ଫେରନ୍ତୁ 0;
#endif
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_cpu_has_vxe(ଶୂନ୍ୟ) {
#ଯଦି defined(__VXE__) || defined(__VXE2__)
    ଫେରନ୍ତୁ 1;
#ଅନ୍ୟଥା
    ଫେରନ୍ତୁ 0;
#endif
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_cpu_has_neon(ଶୂନ୍ୟ) {
#ଯଦି defined(__ARM_ARCH) && defined(__ARM_NEON)
    ଫେରନ୍ତୁ 1;
#ଅନ୍ୟଥା
    ଫେରନ୍ତୁ 0;
#endif
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_cpu_has_dotprod(ଶୂନ୍ୟ) {
#ଯଦି defined(__ARM_ARCH) && defined(__ARM_FEATURE_DOTPROD)
    ଫେରନ୍ତୁ 1;
#ଅନ୍ୟଥା
    ଫେରନ୍ତୁ 0;
#endif
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_cpu_has_sve(ଶୂନ୍ୟ) {
#ଯଦି defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    ଫେରନ୍ତୁ 1;
#ଅନ୍ୟଥା
    ଫେରନ୍ତୁ 0;
#endif
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_cpu_has_matmul_int8(ଶୂନ୍ୟ) {
#ଯଦି defined(__ARM_ARCH) && defined(__ARM_FEATURE_MATMUL_INT8)
    ଫେରନ୍ତୁ 1;
#ଅନ୍ୟଥା
    ଫେରନ୍ତୁ 0;
#endif
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_cpu_get_sve_cnt(ଶୂନ୍ୟ) {
#ଯଦି defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    ଫେରନ୍ତୁ ggml_arm_arch_features.sve_cnt;
#ଅନ୍ୟଥା
    ଫେରନ୍ତୁ 0;
#endif
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_cpu_has_sme(ଶୂନ୍ୟ) {
#ଯଦି defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME)
    ଫେରନ୍ତୁ 1;
#ଅନ୍ୟଥା
    ଫେରନ୍ତୁ 0;
#endif
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_cpu_has_sme2(ଶୂନ୍ୟ) {
#ଯଦି defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME2)
    ଫେରନ୍ତୁ 1;
#ଅନ୍ୟଥା
    ଫେରନ୍ତୁ 0;
#endif
}

ଶୂନ୍ୟ ggml_cpu_init(ଶୂନ୍ୟ) {
    // needed to initialize ggml_time
    {
        ସଂରଚନା ggml_init_params params = { 0, ଖାଲି, false };
        ସଂରଚନା ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    ggml_critical_section_start();

    ସ୍ଥିର bool is_first_call = true;

    ଯଦି (is_first_call) {
        // initialize GELU, Quick GELU, SILU and EXP F32 tables
        {
            ଅଚଳ uint64_t t_start = ggml_time_us(); UNUSED(t_start);

            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < (1 << 16); ++i) {
                ସଂଘ {
                    uint16_t u16;
                    ggml_fp16_t fp16;
                } u = {i};
                ଦଶମିକ f = GGML_COMPUTE_FP16_TO_FP32(u.fp16);
                ggml_table_f32_f16[i] = f;
                ggml_table_gelu_f16[i] = GGML_CPU_FP32_TO_FP16(ggml_gelu_f32(f));
                ggml_table_gelu_quick_f16[i] = GGML_CPU_FP32_TO_FP16(ggml_gelu_quick_f32(f));
            }

            // initialize E8M0 half table (256 entries)
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < (1 << 8); ++i) {
                ggml_table_f32_e8m0_half[i] = GGML_E8M0_TO_FP32_HALF(i);
            }

            // initialize UE4M3 table (256 entries)
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < (1 << 8); ++i) {
                ggml_table_f32_ue4m3[i] = ggml_ue4m3_to_fp32(i);
            }

            ଅଚଳ uint64_t t_end = ggml_time_us(); UNUSED(t_end);

            GGML_PRINT_DEBUG("%s: GELU, Quick GELU, SILU and EXP tables initialized in %f ms\n", __func__, (t_end - t_start)/1000.0);

#ifdef GGML_USE_OPENMP
            //if (!getenv("OMP_WAIT_POLICY")) {
            //    // set the wait policy to active, so that OpenMP threads don't sleep
            //    setenv("OMP_WAIT_POLICY", "active", 0)
            //}

            ଯଦି (!getenv("KMP_BLOCKTIME")) {
                // set the time to wait before sleeping a thread
                // this is less aggressive than setting the wait policy to active, but should achieve similar results in most cases
#ifdef _WIN32
                _putenv_s("KMP_BLOCKTIME", "200"); // 200ms
#ଅନ୍ୟଥା
                setenv("KMP_BLOCKTIME", "200", 0); // 200ms
#endif
            }
#endif
        }

#ଯଦି defined(__ARM_ARCH)
        ggml_init_arm_arch_features();
#endif

#ଯଦି defined(__riscv)
        ggml_init_riscv_arch_features();
#endif

        {
            ଅଚଳ ଅକ୍ଷର * env = getenv("GGML_CPU_DISABLE_FUSION");
            ggml_cpu_disable_fusion = (env != ଖାଲି && atoi(env) == 1);
        }

        is_first_call = false;
    }

    ggml_critical_section_end();
}
