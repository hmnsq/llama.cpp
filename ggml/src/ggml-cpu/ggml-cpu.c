#నిర్వచించు _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#నిర్వచించు USE_MATH_నిర్వచిస్తుంది // For M_PI on MSVC

#అన్తర్భావ "ggml-backend-impl.h"
#అన్తర్భావ "ggml-backend.h"
#అన్తర్భావ "traits.h"
#అన్తర్భావ "ggml-cpu-impl.h"
#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "quants.h"
#అన్తర్భావ "ggml-threading.h"
#అన్తర్భావ "unary-ops.h"
#అన్తర్భావ "binary-ops.h"
#అన్తర్భావ "vec.h"
#అన్తర్భావ "ops.h"
#అన్తర్భావ "ggml.h"
#అన్తర్భావ "common.h"
#అన్తర్భావ "tiled/tiled.h"

#ఒకవేళ defined(_MSC_VER) || defined(__MINGW32__)
#అన్తర్భావ <malloc.h> // using malloc.h with MSC/MINGW
#లేదావేళ !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#అన్తర్భావ <alloca.h>
#అయితే_ముగింపు

#అన్తర్భావ <assert.h>
#అన్తర్భావ <errno.h>
#అన్తర్భావ <time.h>
#అన్తర్భావ <math.h>
#అన్తర్భావ <stdlib.h>
#అన్తర్భావ <string.h>
#అన్తర్భావ <stdint.h>
#అన్తర్భావ <inttypes.h>
#అన్తర్భావ <stdio.h>
#అన్తర్భావ <float.h>
#అన్తర్భావ <limits.h>
#అన్తర్భావ <stdarg.h>
#అన్తర్భావ <signal.h>
#ఒకవేళ defined(__gnu_linux__)
#అన్తర్భావ <syscall.h>
#అయితే_ముగింపు

#నిర్వచితమైతే GGML_USE_OPENMP
#అన్తర్భావ <omp.h>
#అయితే_ముగింపు

#ఒకవేళ defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_MATMUL_INT8)
#నిర్వచనం_రద్దు GGML_USE_LLAMAFILE
#అయితే_ముగింపు

#నిర్వచితమైతే GGML_USE_LLAMAFILE
#అన్తర్భావ "llamafile/sgemm.h"
#అయితే_ముగింపు

#నిర్వచితమైతే GGML_USE_CPU_RISCV64_SPACEMIT
#    include "spacemit/ime.h"
#అయితే_ముగింపు

// Note: once we move threading into a separate C++ file
// will use std::hardware_destructive_interference_size instead of hardcoding it here
// and we'll use C++ attribute syntax.
#నిర్వచించు GGML_CACHE_పంక్తి  64

#ఒకవేళ defined(__clang__) || defined(__GNUC__)
#నిర్వచించు GGML_CACHE_ALIGN __attribute__((aligned(GGML_CACHE_పంక్తి)))
#అయితే_ముగింపు

#ఒకవేళ defined(__has_feature)
#ఒకవేళ __has_feature(thread_sanitizer)
#నిర్వచించు GGML_TSAN_ENABLED 1
#అయితే_ముగింపు
#లేకపోతే  // __has_feature
#ఒకవేళ defined(__SANITIZE_THREAD__)
#నిర్వచించు GGML_TSAN_ENABLED 1
#అయితే_ముగింపు
#అయితే_ముగింపు // __has_feature

#నిర్వచించు వాడని GGML_వాడని
#నిర్వచించు మార్పిడి(x, y, T) చేయి { T మార్పిడి = x; (x) = y; (y) = మార్పిడి; } అయితే (0)

// precomputed f32 table for f16 (256 KB) (simd-mappings.h)
దశాంశం ggml_పట్టిక_f32_f16[1 << 16];

// precomputed f32 table for e8m0 half (1 KB) (simd-mappings.h)
దశాంశం ggml_పట్టిక_f32_e8m0_half[1 << 8];

// precomputed f32 table for ue4m3 (1 KB) (simd-mappings.h)
దశాంశం ggml_పట్టిక_f32_ue4m3[1 << 8];

#ఒకవేళ defined(__ARM_ARCH)
నిర్మాణం ggml_arm_arch_features_type {
    పూర్ణాంకం sve_cnt;
} ggml_arm_arch_features = { 0 };
#అయితే_ముగింపు

#ఒకవేళ defined(__riscv)
నిర్మాణం ggml_riscv_arch_features_type {
    పూర్ణాంకం rvv_vlen;
} ggml_riscv_arch_features = { 0 };
#అయితే_ముగింపు

#ఒకవేళ defined(_WIN32)

#నిర్వచించు WIN32_LEAN_AND_సగటు
#నిర్వచితం_కాకపోతే NOMINMAX
    #నిర్వచించు NOMINMAX
#అయితే_ముగింపు
#అన్తర్భావ <windows.h>

#ఒకవేళ defined(_MSC_VER) && !defined(__clang__)
#నిర్వచించు GGML_CACHE_ALIGN __declspec(align(GGML_CACHE_పంక్తి))

రకనిర్వచనం అస్థిర LONG atomic_int;
రకనిర్వచనం atomic_int atomic_bool;
రకనిర్వచనం atomic_int atomic_flag;

#నిర్వచించు ATOMIC_FLAG_INIT 0

రకనిర్వచనం గణన {
    memory_order_relaxed,
    memory_order_consume,
    memory_order_acquire,
    memory_order_release,
    memory_order_acq_rel,
    memory_order_seq_cst
} memory_order;

స్థిర శూన్య atomic_store(atomic_int * ptr, LONG val) {
    InterlockedExchange(ptr, val);
}
స్థిర శూన్య atomic_store_explicit(atomic_int * ptr, LONG val, memory_order mo) {
    // TODO: add support for explicit memory order
    InterlockedExchange(ptr, val);
}
స్థిర LONG atomic_ఎక్కించు(atomic_int * ptr) {
    తిరిగి InterlockedCompareExchange(ptr, 0, 0);
}
స్థిర LONG atomic_ఎక్కించు_explicit(atomic_int * ptr, memory_order mo) {
    // TODO: add support for explicit memory order
    తిరిగి InterlockedCompareExchange(ptr, 0, 0);
}
స్థిర LONG atomic_fetch_కూడు(atomic_int * ptr, LONG inc) {
    తిరిగి Interlocked_Exchange_కూడు(ptr, inc);
}
స్థిర LONG atomic_fetch_కూడు_explicit(atomic_int * ptr, LONG inc, memory_order mo) {
    // TODO: add support for explicit memory order
    తిరిగి Interlocked_Exchange_కూడు(ptr, inc);
}
స్థిర atomic_bool atomic_flag_test_and_సమితి(atomic_flag * ptr) {
    తిరిగి InterlockedExchange(ptr, 1);
}
స్థిర శూన్య atomic_flag_clear(atomic_flag * ptr) {
    InterlockedExchange(ptr, 0);
}
స్థిర శూన్య atomic_thread_fence(memory_order mo) {
    జ్ఞాపకం_Barrier();
}
#లేకపోతే // clang
#అన్తర్భావ <stdatomic.h>
#అయితే_ముగింపు

రకనిర్వచనం HANDLE pthread_t;

రకనిర్వచనం DWORD thread_ret_t;
స్థిర పూర్ణాంకం pthread_create(pthread_t * బహిర్గతం, శూన్య * వాడని_2, thread_ret_t(*func)(శూన్య *), శూన్య * arg) {
    (శూన్య) వాడని_2;
    HANDLE handle = CreateThread(ఖాళీ, 0, (LPTHREAD_ప్రారంభం_ROUTINE) func, arg, 0, ఖాళీ);
    ఒకవేళ (handle == ఖాళీ)
    {
        తిరిగి EAGAIN;
    }

    *బహిర్గతం = handle;
    తిరిగి 0;
}

స్థిర పూర్ణాంకం pthread_join(pthread_t thread, శూన్య * వాడని_2) {
    (శూన్య) వాడని_2;
    పూర్ణాంకం ret = (పూర్ణాంకం) WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    తిరిగి ret;
}

స్థిర పూర్ణాంకం sched_yield (శూన్య) {
    Sleep (0);
    తిరిగి 0;
}
#లేకపోతే

#అన్తర్భావ <pthread.h>
#అన్తర్భావ <stdatomic.h>
#అన్తర్భావ <sched.h>
#ఒకవేళ defined(__FreeBSD__)
#అన్తర్భావ <pthread_np.h>
#అయితే_ముగింపు

రకనిర్వచనం శూన్య * thread_ret_t;

#అన్తర్భావ <sys/types.h>
#అన్తర్భావ <sys/stat.h>
#అన్తర్భావ <unistd.h>

#అయితే_ముగింపు

రకనిర్వచనం pthread_t ggml_thread_t;

#నిర్వచించు GGML_THREADPOOL_N_దారాలు_ముసుగు (0xffffU)
#నిర్వచించు GGML_THREADPOOL_N_దారాలు_బిట్లు (16)

#ఒకవేళ defined(__APPLE__)
#అన్తర్భావ <unistd.h>
#అన్తర్భావ <mach/mach.h>
#అన్తర్భావ <TargetConditionals.h>
#అయితే_ముగింపు

స్థిర స్థిరరాశి నిర్మాణం ggml_type_traits_cpu type_traits_cpu[GGML_TYPE_లెక్క] = {
    [GGML_TYPE_F32] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_గమ్యం_fp32,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_f32,
        .vec_dot_type             = GGML_TYPE_F32,
        .nrows                    = 1,
    },
    [GGML_TYPE_F16] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_గమ్యం_fp16,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_f16,
        .vec_dot_type             = GGML_TYPE_F16,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q1_0] = {
        .from_float               = quantize_అడ్డువరుస_q1_0,
        .vec_dot                  = ggml_vec_dot_q1_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q2_0] = {
        .from_float               = quantize_అడ్డువరుస_q2_0,
        .vec_dot                  = ggml_vec_dot_q2_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q4_0] = {
        .from_float               = quantize_అడ్డువరుస_q4_0,
        .vec_dot                  = ggml_vec_dot_q4_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
#ఒకవేళ defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#లేకపోతే
        .nrows                    = 1,
#అయితే_ముగింపు
    },
    [GGML_TYPE_Q4_1] = {
        .from_float               = quantize_అడ్డువరుస_q4_1,
        .vec_dot                  = ggml_vec_dot_q4_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
#ఒకవేళ defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#లేకపోతే
        .nrows                    = 1,
#అయితే_ముగింపు
    },
    [GGML_TYPE_Q5_0] = {
        .from_float               = quantize_అడ్డువరుస_q5_0,
        .vec_dot                  = ggml_vec_dot_q5_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q5_1] = {
        .from_float               = quantize_అడ్డువరుస_q5_1,
        .vec_dot                  = ggml_vec_dot_q5_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q8_0] = {
        .from_float               = quantize_అడ్డువరుస_q8_0,
        .vec_dot                  = ggml_vec_dot_q8_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
#ఒకవేళ defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#లేకపోతే
        .nrows                    = 1,
#అయితే_ముగింపు
    },
    [GGML_TYPE_Q8_1] = {
        .from_float               = quantize_అడ్డువరుస_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
        .nrows                    = 1,
    },
    [GGML_TYPE_MXFP4] = {
        .from_float               = quantize_అడ్డువరుస_mxfp4,
        .vec_dot                  = ggml_vec_dot_mxfp4_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_NVFP4] = {
        .from_float               = quantize_అడ్డువరుస_nvfp4,
        .vec_dot                  = ggml_vec_dot_nvfp4_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q2_K] = {
        .from_float               = quantize_అడ్డువరుస_q2_K,
        .vec_dot                  = ggml_vec_dot_q2_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q3_K] = {
        .from_float               = quantize_అడ్డువరుస_q3_K,
        .vec_dot                  = ggml_vec_dot_q3_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q4_K] = {
        .from_float               = quantize_అడ్డువరుస_q4_K,
        .vec_dot                  = ggml_vec_dot_q4_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
#ఒకవేళ defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#లేకపోతే
        .nrows                    = 1,
#అయితే_ముగింపు
    },
    [GGML_TYPE_Q5_K] = {
        .from_float               = quantize_అడ్డువరుస_q5_K,
        .vec_dot                  = ggml_vec_dot_q5_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q6_K] = {
        .from_float               = quantize_అడ్డువరుస_q6_K,
        .vec_dot                  = ggml_vec_dot_q6_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
#ఒకవేళ defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#లేకపోతే
        .nrows                    = 1,
#అయితే_ముగింపు
    },
    [GGML_TYPE_IQ2_XXS] = {
        .from_float               = ఖాళీ,
        .vec_dot                  = ggml_vec_dot_iq2_xxs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_XS] = {
        .from_float               = ఖాళీ,
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
        .from_float               = ఖాళీ,
        .vec_dot                  = ggml_vec_dot_iq1_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_M] = {
        .from_float               = ఖాళీ,
        .vec_dot                  = ggml_vec_dot_iq1_m_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_NL] = {
        .from_float               = quantize_అడ్డువరుస_iq4_nl,
        .vec_dot                  = ggml_vec_dot_iq4_nl_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_XS] = {
        .from_float               = quantize_అడ్డువరుస_iq4_xs,
        .vec_dot                  = ggml_vec_dot_iq4_xs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q8_K] = {
        .from_float               = quantize_అడ్డువరుస_q8_K,
    },
    [GGML_TYPE_BF16] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_గమ్యం_bf16,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_bf16,
        .vec_dot_type             = GGML_TYPE_BF16,
        .nrows                    = 1,
    },
    [GGML_TYPE_TQ1_0] = {
        .from_float               = quantize_అడ్డువరుస_tq1_0,
        .vec_dot                  = ggml_vec_dot_tq1_0_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_TQ2_0] = {
        .from_float               = quantize_అడ్డువరుస_tq2_0,
        .vec_dot                  = ggml_vec_dot_tq2_0_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_I32] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_గమ్యం_i32,
    },
};

స్థిరరాశి నిర్మాణం ggml_type_traits_cpu * ggml_get_type_traits_cpu(గణన ggml_type type) {
    తిరిగి &type_traits_cpu[type];
}

//
// Threading defs
//

రకనిర్వచనం pthread_t          ggml_thread_t;

#ఒకవేళ defined(_WIN32)

రకనిర్వచనం CONDITION_VARIABLE ggml_cond_t;
రకనిర్వచనం SRWLOCK            ggml_mutex_t;

#నిర్వచించు ggml_mutex_init(m)   InitializeSRWLock(m)
#నిర్వచించు ggml_mutex_destroy(m)
#నిర్వచించు ggml_mutex_తాళం(m)   AcquireSRWLockExclusive(m)
#నిర్వచించు ggml_mutex_unlock(m) ReleaseSRWLockExclusive(m)
#నిర్వచించు ggml_mutex_తాళం_shared(m)   AcquireSRWLockShared(m)
#నిర్వచించు ggml_mutex_unlock_shared(m) ReleaseSRWLockShared(m)

#నిర్వచించు ggml_cond_init(c)    InitializeConditionVariable(c)
#నిర్వచించు ggml_cond_destroy(c)
#నిర్వచించు ggml_cond_wait(c, m) SleepConditionVariableSRW(c, m, INFINITE, CONDITION_VARIABLE_LOCKMODE_SHARED)
#నిర్వచించు ggml_cond_ప్రసారం(c) WakeAllConditionVariable(c)

#నిర్వచించు ggml_thread_create pthread_create
#నిర్వచించు ggml_thread_join   pthread_join

#లేకపోతే

రకనిర్వచనం pthread_cond_t     ggml_cond_t;
రకనిర్వచనం pthread_mutex_t    ggml_mutex_t;

#నిర్వచించు ggml_mutex_init(m)          pthread_mutex_init(m, ఖాళీ)
#నిర్వచించు ggml_mutex_destroy(m)       pthread_mutex_destroy(m)
#నిర్వచించు ggml_mutex_తాళం(m)          pthread_mutex_తాళం(m)
#నిర్వచించు ggml_mutex_unlock(m)        pthread_mutex_unlock(m)
#నిర్వచించు ggml_mutex_తాళం_shared(m)   pthread_mutex_తాళం(m)
#నిర్వచించు ggml_mutex_unlock_shared(m) pthread_mutex_unlock(m)

#నిర్వచించు ggml_తాళం_init(x)    వాడని(x)
#నిర్వచించు ggml_తాళం_destroy(x) వాడని(x)
#ఒకవేళ defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64) && !defined(_M_ARM64EC))
#నిర్వచించు ggml_తాళం_తాళం(x)    _mm_pause()
#లేకపోతే
#నిర్వచించు ggml_తాళం_తాళం(x)    వాడని(x)
#అయితే_ముగింపు
#నిర్వచించు ggml_తాళం_unlock(x)  వాడని(x)

#నిర్వచించు GGML_తాళం_INITIALIZER 0
#నిర్వచించు ggml_cond_init(c)      pthread_cond_init(c, ఖాళీ)
#నిర్వచించు ggml_cond_destroy(c)   pthread_cond_destroy(c)
#నిర్వచించు ggml_cond_wait(c, m)   pthread_cond_wait(c, m)
#నిర్వచించు ggml_cond_ప్రసారం(c) pthread_cond_ప్రసారం(c)

#నిర్వచించు ggml_thread_create pthread_create
#నిర్వచించు ggml_thread_join   pthread_join

#అయితే_ముగింపు

// Threadpool def
నిర్మాణం ggml_threadpool {
    ggml_mutex_t mutex;       // mutex for cond.var
    ggml_cond_t  cond;        // cond.var for waiting for new work

    నిర్మాణం ggml_cgraph * cgraph;
    నిర్మాణం ggml_cplan  * cplan;

    // synchronization primitives
    atomic_int n_లేఖాచిత్రం;       // updated when there is work to be done (i.e each graph) holds graph and active thread counts.
    atomic_int GGML_CACHE_ALIGN n_barrier;
    atomic_int GGML_CACHE_ALIGN n_barrier_passed;
    atomic_int GGML_CACHE_ALIGN current_chunk; // currently processing chunk during Mat_Mul, shared between all the threads.

    // these are atomic as an annotation for thread-sanitizer
    atomic_bool stop;         // Used for stopping the threadpool altogether
    atomic_bool pause;        // Used for pausing the threadpool or individual threads
    atomic_int  abort;        // Used for aborting processing of a graph

    నిర్మాణం ggml_compute_state * workers;   // per thread state
    పూర్ణాంకం          n_దారాలు;   // Number of threads in the pool
    int32_t      prio;        // Scheduling priority
    uint32_t     poll;        // Polling level (0 - no polling)

    గణన ggml_స్థితి ec;
};

// Per-thread state
నిర్మాణం ggml_compute_state {
#నిర్వచితం_కాకపోతే GGML_USE_OPENMP
    ggml_thread_t thrd;
    పూర్ణాంకం  చివరి_లేఖాచిత్రం;
    bool pending;
#అయితే_ముగింపు
    bool cpumask[GGML_MAX_N_దారాలు];
    నిర్మాణం ggml_threadpool * threadpool;
    పూర్ణాంకం ith;
};

// Helpers for polling loops
#ఒకవేళ defined(__aarch64__) && ( defined(__clang__) || defined(__GNUC__) )
స్థిర అంతఃవరుస శూన్య ggml_thread_cpu_relax(శూన్య) {
    __asm__ అస్థిర("yield" ::: "memory");
}
#లేదావేళ defined(__x86_64__)
స్థిర అంతఃవరుస శూన్య ggml_thread_cpu_relax(శూన్య) {
    _mm_pause();
}
#లేదావేళ defined(__riscv)
స్థిర అంతఃవరుస శూన్య ggml_thread_cpu_relax(శూన్య) {
    #నిర్వచితమైతే __riscv_zihintpause
        __asm__ __volatile__ ("pause");
    #లేకపోతే
        /* Encoding of the pause instruction */
        __asm__ __volatile__ (".4byte 0x100000F");
    #అయితే_ముగింపు
}
#లేకపోతే
స్థిర అంతఃవరుస శూన్య ggml_thread_cpu_relax(శూన్య) {;}
#అయితే_ముగింపు

//
// NUMA support
//

#నిర్వచించు GGML_NUMA_MAX_NODES 8
#నిర్వచించు GGML_NUMA_MAX_CPUS 512

నిర్మాణం ggml_numa_కణుపు {
    uint32_t cpus[GGML_NUMA_MAX_CPUS]; // hardware threads on this node
    uint32_t n_cpus;
};

నిర్మాణం ggml_numa_nodes {
    గణన ggml_numa_strategy numa_strategy;
    నిర్మాణం ggml_numa_కణుపు nodes[GGML_NUMA_MAX_NODES];
    uint32_t n_nodes;
    uint32_t మొత్తం_cpus; // hardware threads on system
    uint32_t current_కణుపు; // node on which main process is execting
#ఒకవేళ defined(__gnu_linux__)
    cpu_సమితి_t cpuset; // cpuset from numactl
#లేకపోతే
    uint32_t cpuset; // no NUMA support outside of Linux at this time. Use a portable datatype
#అయితే_ముగింపు
};

//
// ggml state
//

నిర్మాణం ggml_state {
    నిర్మాణం ggml_numa_nodes numa;
};

స్థిర నిర్మాణం ggml_state g_state = {0};

శూన్య ggml_barrier(నిర్మాణం ggml_threadpool * tp) {
    పూర్ణాంకం n_దారాలు = atomic_ఎక్కించు_explicit(&tp->n_లేఖాచిత్రం, memory_order_relaxed) & GGML_THREADPOOL_N_దారాలు_ముసుగు;
    ఒకవేళ (n_దారాలు == 1) {
        తిరిగి;
    }

#నిర్వచితమైతే GGML_USE_OPENMP
    #సూచన omp barrier
#లేకపోతే
    పూర్ణాంకం n_passed = atomic_ఎక్కించు_explicit(&tp->n_barrier_passed, memory_order_relaxed);

    // enter barrier (full seq-cst fence)
    పూర్ణాంకం n_barrier = atomic_fetch_కూడు_explicit(&tp->n_barrier, 1, memory_order_seq_cst);

    ఒకవేళ (n_barrier == (n_దారాలు - 1)) {
        // last thread
        atomic_store_explicit(&tp->n_barrier, 0, memory_order_relaxed);

        // exit barrier (full seq-cst fence)
        atomic_fetch_కూడు_explicit(&tp->n_barrier_passed, 1, memory_order_seq_cst);
        తిరిగి;
    }

    // wait for other threads
    అయితే (atomic_ఎక్కించు_explicit(&tp->n_barrier_passed, memory_order_relaxed) == n_passed) {
        ggml_thread_cpu_relax();
    }

    // exit barrier (full seq-cst fence)
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #నిర్వచితమైతే GGML_TSAN_ENABLED
    atomic_fetch_కూడు_explicit(&tp->n_barrier_passed, 0, memory_order_seq_cst);
    #లేకపోతే
    atomic_thread_fence(memory_order_seq_cst);
    #అయితే_ముగింపు
#అయితే_ముగింపు
}

శూన్య ggml_threadpool_chunk_సమితి(నిర్మాణం ggml_threadpool * tp, పూర్ణాంకం విలువ) {
    atomic_store_explicit(&tp->current_chunk, విలువ, memory_order_relaxed);
}

పూర్ణాంకం ggml_threadpool_chunk_కూడు(నిర్మాణం ggml_threadpool * tp, పూర్ణాంకం విలువ) {
    తిరిగి atomic_fetch_కూడు_explicit(&tp->current_chunk, విలువ, memory_order_relaxed);
}

#ఒకవేళ defined(__gnu_linux__)
స్థిర cpu_సమితి_t ggml_get_numa_affinity(శూన్య) {
    cpu_సమితి_t cpuset;
    pthread_t thread;
    thread = pthread_self();
    CPU_ZERO(&cpuset);
    pthread_getaffinity_np(thread, పరిమాణం(cpu_సమితి_t), &cpuset);
    తిరిగి cpuset;
}
#లేకపోతే
స్థిర uint32_t ggml_get_numa_affinity(శూన్య) {
    తిరిగి 0; // no NUMA support
}
#అయితే_ముగింపు

శూన్య ggml_numa_init(గణన ggml_numa_strategy numa_flag) {
    ఒకవేళ (g_state.numa.n_nodes > 0) {
        fprintf(stderr, "ggml_numa_init: NUMA already initialized\n");

        తిరిగి;
    }

#ఒకవేళ defined(__gnu_linux__)
    నిర్మాణం stat st;
    అక్షరం మార్గం[256];
    పూర్ణాంకం rv;

    // set numa scheme
    g_state.numa.numa_strategy = numa_flag;

    GGML_PRINT_DEBUG("numa strategy %u\n",g_state.numa.numa_strategy);

    g_state.numa.cpuset = ggml_get_numa_affinity();

    // enumerate nodes
    అయితే (g_state.numa.n_nodes < GGML_NUMA_MAX_NODES) {
        rv = snprintf(మార్గం, పరిమాణం(మార్గం), "/sys/devices/system/node/node%u", g_state.numa.n_nodes);
        GGML_ASSERT(rv > 0 && (గుర్తులేని)rv < పరిమాణం(మార్గం));
        ఒకవేళ (stat(మార్గం, &st) != 0) { ఆపు; }
        ++g_state.numa.n_nodes;
    }

    // enumerate CPUs
    అయితే (g_state.numa.మొత్తం_cpus < GGML_NUMA_MAX_CPUS) {
        rv = snprintf(మార్గం, పరిమాణం(మార్గం), "/sys/devices/system/cpu/cpu%u", g_state.numa.మొత్తం_cpus);
        GGML_ASSERT(rv > 0 && (గుర్తులేని)rv < పరిమాణం(మార్గం));
        ఒకవేళ (stat(మార్గం, &st) != 0) { ఆపు; }
        ++g_state.numa.మొత్తం_cpus;
    }

    GGML_PRINT_DEBUG("found %u numa nodes, %u CPUs\n", g_state.numa.n_nodes, g_state.numa.మొత్తం_cpus);

    // figure out which node we're on
    uint current_cpu;
    పూర్ణాంకం getcpu_ret = 0;
#ఒకవేళ __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ > 33) || defined(__COSMOPOLITAN__)
    getcpu_ret = getcpu(&current_cpu, &g_state.numa.current_కణుపు);
#లేకపోతే
    // old glibc doesn't have a wrapper for this call. Fall back on direct syscall
#   ఒకవేళ !defined(SYS_getcpu) && defined(SYS_get_cpu)
#       define SYS_getcpu SYS_get_cpu // some older glibc versions use this name
#   endif
    getcpu_ret = syscall(SYS_getcpu, &current_cpu, &g_state.numa.current_కణుపు);
#అయితే_ముగింపు

    ఒకవేళ (g_state.numa.n_nodes < 1 || g_state.numa.మొత్తం_cpus < 1 || getcpu_ret != 0) {
        g_state.numa.n_nodes = 0;
        తిరిగి;
    }

    GGML_PRINT_DEBUG("found our process on numa node %u, CPU %u\n", g_state.numa.current_కణుపు, current_cpu);

    కొరకు (uint32_t n = 0; n < g_state.numa.n_nodes; ++n) {
        నిర్మాణం ggml_numa_కణుపు * కణుపు = &g_state.numa.nodes[n];
        GGML_PRINT_DEBUG("CPUs on node %u:", n);
        కణుపు->n_cpus = 0;
        కొరకు (uint32_t c = 0; c < g_state.numa.మొత్తం_cpus; ++c) {
            rv = snprintf(మార్గం, పరిమాణం(మార్గం), "/sys/devices/system/node/node%u/cpu%u", n, c);
            GGML_ASSERT(rv > 0 && (గుర్తులేని)rv < పరిమాణం(మార్గం));
            ఒకవేళ (stat(మార్గం, &st) == 0) {
                కణుపు->cpus[కణుపు->n_cpus++] = c;
                GGML_PRINT_DEBUG(" %u", c);
            }
        }
        GGML_PRINT_DEBUG("\n");
    }

    ఒకవేళ (ggml_is_numa()) {
        FILE *fptr = fopen("/proc/sys/kernel/numa_balancing", "r");
        ఒకవేళ (fptr != ఖాళీ) {
            అక్షరం buf[42];
            ఒకవేళ (fgets(buf, పరిమాణం(buf), fptr) && అక్షరశ్రేణి_పోలిక(buf, "0\n", పరిమాణం(buf)) != 0) {
                GGML_LOG_WARN("/proc/sys/kernel/numa_balancing is enabled, this has been observed to impair performance\n");
            }
            fclose(fptr);
        }
    }
#లేకపోతే
    వాడని(numa_flag);
    // TODO
#అయితే_ముగింపు
}

bool ggml_is_numa(శూన్య) {
    తిరిగి g_state.numa.n_nodes > 1;
}

#ఒకవేళ defined(__ARM_ARCH)
#ఒకవేళ defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
#అన్తర్భావ <arm_sve.h>
స్థిర శూన్య ggml_init_arm_arch_features(శూన్య) {
    ggml_arm_arch_features.sve_cnt = svcntb();
}
#లేకపోతే
స్థిర శూన్య ggml_init_arm_arch_features(శూన్య) {}
#అయితే_ముగింపు
#అయితే_ముగింపు // __ARM_ARCH

#ఒకవేళ defined(__riscv) && defined(__riscv_v_intrinsic)
#అన్తర్భావ <riscv_vector.h>
స్థిర శూన్య ggml_init_riscv_arch_features(శూన్య) {
    ggml_riscv_arch_features.rvv_vlen = __riscv_vlenb();
}
#లేకపోతే
స్థిర శూన్య ggml_init_riscv_arch_features(శూన్య) {}
#అయితే_ముగింపు

నిర్మాణం ggml_tensor * ggml_new_i32(నిర్మాణం ggml_context * ctx, int32_t విలువ) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

    ggml_సమితి_i32(ఫలితం, విలువ);

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_new_f32(నిర్మాణం ggml_context * ctx, దశాంశం విలువ) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    ggml_సమితి_f32(ఫలితం, విలువ);

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_సమితి_i32 (నిర్మాణం ggml_tensor * tensor, int32_t విలువ) {
    స్థిరరాశి పూర్ణాంకం n     = ggml_nrows(tensor);
    స్థిరరాశి పూర్ణాంకం nc    = tensor->ne[0];
    స్థిరరాశి size_t n1 = tensor->nb[1];

    అక్షరం * స్థిరరాశి దత్తాంశం = tensor->దత్తాంశం;

    ఎంపిక (tensor->type) {
        స్థితి GGML_TYPE_I8:
            {
                ప్రతిజ్ఞా(tensor->nb[0] == పరిమాణం(int8_t));
                కొరకు (పూర్ణాంకం i = 0; i < n; i++) {
                    ggml_vec_సమితి_i8(nc, (int8_t *)(దత్తాంశం + i*n1), విలువ);
                }
            } ఆపు;
        స్థితి GGML_TYPE_I16:
            {
                ప్రతిజ్ఞా(tensor->nb[0] == పరిమాణం(int16_t));
                కొరకు (పూర్ణాంకం i = 0; i < n; i++) {
                    ggml_vec_సమితి_i16(nc, (int16_t *)(దత్తాంశం + i*n1), విలువ);
                }
            } ఆపు;
        స్థితి GGML_TYPE_I32:
            {
                ప్రతిజ్ఞా(tensor->nb[0] == పరిమాణం(int32_t));
                కొరకు (పూర్ణాంకం i = 0; i < n; i++) {
                    ggml_vec_సమితి_i32(nc, (int32_t *)(దత్తాంశం + i*n1), విలువ);
                }
            } ఆపు;
        స్థితి GGML_TYPE_F16:
            {
                ప్రతిజ్ఞా(tensor->nb[0] == పరిమాణం(ggml_fp16_t));
                కొరకు (పూర్ణాంకం i = 0; i < n; i++) {
                    ggml_vec_సమితి_f16(nc, (ggml_fp16_t *)(దత్తాంశం + i*n1), GGML_CPU_FP32_గమ్యం_FP16(విలువ));
                }
            } ఆపు;
        స్థితి GGML_TYPE_BF16:
            {
                ప్రతిజ్ఞా(tensor->nb[0] == పరిమాణం(ggml_fp16_t));
                కొరకు (పూర్ణాంకం i = 0; i < n; i++) {
                    ggml_vec_సమితి_bf16(nc, (ggml_bf16_t *)(దత్తాంశం + i*n1), GGML_FP32_గమ్యం_BF16(విలువ));
                }
            } ఆపు;
        స్థితి GGML_TYPE_F32:
            {
                ప్రతిజ్ఞా(tensor->nb[0] == పరిమాణం(దశాంశం));
                కొరకు (పూర్ణాంకం i = 0; i < n; i++) {
                    ggml_vec_సమితి_f32(nc, (దశాంశం *)(దత్తాంశం + i*n1), విలువ);
                }
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }

    తిరిగి tensor;
}

నిర్మాణం ggml_tensor * ggml_సమితి_f32(నిర్మాణం ggml_tensor * tensor, దశాంశం విలువ) {
    స్థిరరాశి పూర్ణాంకం n     = ggml_nrows(tensor);
    స్థిరరాశి పూర్ణాంకం nc    = tensor->ne[0];
    స్థిరరాశి size_t n1 = tensor->nb[1];

    అక్షరం * స్థిరరాశి దత్తాంశం = tensor->దత్తాంశం;

    ఎంపిక (tensor->type) {
        స్థితి GGML_TYPE_I8:
            {
                ప్రతిజ్ఞా(tensor->nb[0] == పరిమాణం(int8_t));
                కొరకు (పూర్ణాంకం i = 0; i < n; i++) {
                    ggml_vec_సమితి_i8(nc, (int8_t *)(దత్తాంశం + i*n1), విలువ);
                }
            } ఆపు;
        స్థితి GGML_TYPE_I16:
            {
                ప్రతిజ్ఞా(tensor->nb[0] == పరిమాణం(int16_t));
                కొరకు (పూర్ణాంకం i = 0; i < n; i++) {
                    ggml_vec_సమితి_i16(nc, (int16_t *)(దత్తాంశం + i*n1), విలువ);
                }
            } ఆపు;
        స్థితి GGML_TYPE_I32:
            {
                ప్రతిజ్ఞా(tensor->nb[0] == పరిమాణం(int32_t));
                కొరకు (పూర్ణాంకం i = 0; i < n; i++) {
                    ggml_vec_సమితి_i32(nc, (int32_t *)(దత్తాంశం + i*n1), విలువ);
                }
            } ఆపు;
        స్థితి GGML_TYPE_F16:
            {
                ప్రతిజ్ఞా(tensor->nb[0] == పరిమాణం(ggml_fp16_t));
                కొరకు (పూర్ణాంకం i = 0; i < n; i++) {
                    ggml_vec_సమితి_f16(nc, (ggml_fp16_t *)(దత్తాంశం + i*n1), GGML_CPU_FP32_గమ్యం_FP16(విలువ));
                }
            } ఆపు;
        స్థితి GGML_TYPE_BF16:
            {
                ప్రతిజ్ఞా(tensor->nb[0] == పరిమాణం(ggml_bf16_t));
                కొరకు (పూర్ణాంకం i = 0; i < n; i++) {
                    ggml_vec_సమితి_bf16(nc, (ggml_bf16_t *)(దత్తాంశం + i*n1), GGML_FP32_గమ్యం_BF16(విలువ));
                }
            } ఆపు;
        స్థితి GGML_TYPE_F32:
            {
                ప్రతిజ్ఞా(tensor->nb[0] == పరిమాణం(దశాంశం));
                కొరకు (పూర్ణాంకం i = 0; i < n; i++) {
                    ggml_vec_సమితి_f32(nc, (దశాంశం *)(దత్తాంశం + i*n1), విలువ);
                }
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }

    తిరిగి tensor;
}

int32_t ggml_get_i32_1d(స్థిరరాశి నిర్మాణం ggml_tensor * tensor, పూర్ణాంకం i) {
    ఒకవేళ (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_సూచీ(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        తిరిగి ggml_get_i32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    ఎంపిక (tensor->type) {
        స్థితి GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == పరిమాణం(int8_t));
                తిరిగి ((int8_t *)(tensor->దత్తాంశం))[i];
            }
        స్థితి GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == పరిమాణం(int16_t));
                తిరిగి ((int16_t *)(tensor->దత్తాంశం))[i];
            }
        స్థితి GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == పరిమాణం(int32_t));
                తిరిగి ((int32_t *)(tensor->దత్తాంశం))[i];
            }
        స్థితి GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == పరిమాణం(ggml_fp16_t));
                తిరిగి GGML_CPU_FP16_గమ్యం_FP32(((ggml_fp16_t *)(tensor->దత్తాంశం))[i]);
            }
        స్థితి GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == పరిమాణం(ggml_bf16_t));
                తిరిగి GGML_BF16_గమ్యం_FP32(((ggml_bf16_t *)(tensor->దత్తాంశం))[i]);
            }
        స్థితి GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == పరిమాణం(దశాంశం));
                తిరిగి ((దశాంశం *)(tensor->దత్తాంశం))[i];
            }
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

శూన్య ggml_సమితి_i32_1d(స్థిరరాశి నిర్మాణం ggml_tensor * tensor, పూర్ణాంకం i, int32_t విలువ) {
    ఒకవేళ (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_సూచీ(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_సమితి_i32_nd(tensor, id[0], id[1], id[2], id[3], విలువ);
        తిరిగి;
    }
    ఎంపిక (tensor->type) {
        స్థితి GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == పరిమాణం(int8_t));
                ((int8_t *)(tensor->దత్తాంశం))[i] = విలువ;
            } ఆపు;
        స్థితి GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == పరిమాణం(int16_t));
                ((int16_t *)(tensor->దత్తాంశం))[i] = విలువ;
            } ఆపు;
        స్థితి GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == పరిమాణం(int32_t));
                ((int32_t *)(tensor->దత్తాంశం))[i] = విలువ;
            } ఆపు;
        స్థితి GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == పరిమాణం(ggml_fp16_t));
                ((ggml_fp16_t *)(tensor->దత్తాంశం))[i] = GGML_CPU_FP32_గమ్యం_FP16(విలువ);
            } ఆపు;
        స్థితి GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == పరిమాణం(ggml_bf16_t));
                ((ggml_bf16_t *)(tensor->దత్తాంశం))[i] = GGML_FP32_గమ్యం_BF16(విలువ);
            } ఆపు;
        స్థితి GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == పరిమాణం(దశాంశం));
                ((దశాంశం *)(tensor->దత్తాంశం))[i] = విలువ;
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

int32_t ggml_get_i32_nd(స్థిరరాశి నిర్మాణం ggml_tensor * tensor, పూర్ణాంకం i0, పూర్ణాంకం i1, పూర్ణాంకం i2, పూర్ణాంకం i3) {
    శూన్య * దత్తాంశం   = (అక్షరం *) tensor->దత్తాంశం + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    ఎంపిక (tensor->type) {
        స్థితి GGML_TYPE_I8:
            తిరిగి ((int8_t *) దత్తాంశం)[0];
        స్థితి GGML_TYPE_I16:
            తిరిగి ((int16_t *) దత్తాంశం)[0];
        స్థితి GGML_TYPE_I32:
            తిరిగి ((int32_t *) దత్తాంశం)[0];
        స్థితి GGML_TYPE_F16:
            తిరిగి GGML_CPU_FP16_గమ్యం_FP32(((ggml_fp16_t *) దత్తాంశం)[0]);
        స్థితి GGML_TYPE_BF16:
            తిరిగి GGML_BF16_గమ్యం_FP32(((ggml_bf16_t *) దత్తాంశం)[0]);
        స్థితి GGML_TYPE_F32:
            తిరిగి ((దశాంశం *) దత్తాంశం)[0];
        అప్రమేయ:
            GGML_ABORT("fatal error");
    }
}

శూన్య ggml_సమితి_i32_nd(స్థిరరాశి నిర్మాణం ggml_tensor * tensor, పూర్ణాంకం i0, పూర్ణాంకం i1, పూర్ణాంకం i2, పూర్ణాంకం i3, int32_t విలువ) {
    శూన్య * దత్తాంశం   = (అక్షరం *) tensor->దత్తాంశం + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    ఎంపిక (tensor->type) {
        స్థితి GGML_TYPE_I8:
            {
                ((int8_t *)(దత్తాంశం))[0] = విలువ;
            } ఆపు;
        స్థితి GGML_TYPE_I16:
            {
                ((int16_t *)(దత్తాంశం))[0] = విలువ;
            } ఆపు;
        స్థితి GGML_TYPE_I32:
            {
                ((int32_t *)(దత్తాంశం))[0] = విలువ;
            } ఆపు;
        స్థితి GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(దత్తాంశం))[0] = GGML_CPU_FP32_గమ్యం_FP16(విలువ);
            } ఆపు;
        స్థితి GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(దత్తాంశం))[0] = GGML_FP32_గమ్యం_BF16(విలువ);
            } ఆపు;
        స్థితి GGML_TYPE_F32:
            {
                ((దశాంశం *)(దత్తాంశం))[0] = విలువ;
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

దశాంశం ggml_get_f32_1d(స్థిరరాశి నిర్మాణం ggml_tensor * tensor, పూర్ణాంకం i) {
    ఒకవేళ (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_సూచీ(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        తిరిగి ggml_get_f32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    ఎంపిక (tensor->type) {
        స్థితి GGML_TYPE_I8:
            {
                తిరిగి ((int8_t *)(tensor->దత్తాంశం))[i];
            }
        స్థితి GGML_TYPE_I16:
            {
                తిరిగి ((int16_t *)(tensor->దత్తాంశం))[i];
            }
        స్థితి GGML_TYPE_I32:
            {
                తిరిగి ((int32_t *)(tensor->దత్తాంశం))[i];
            }
        స్థితి GGML_TYPE_F16:
            {
                తిరిగి GGML_CPU_FP16_గమ్యం_FP32(((ggml_fp16_t *)(tensor->దత్తాంశం))[i]);
            }
        స్థితి GGML_TYPE_BF16:
            {
                తిరిగి GGML_BF16_గమ్యం_FP32(((ggml_bf16_t *)(tensor->దత్తాంశం))[i]);
            }
        స్థితి GGML_TYPE_F32:
            {
                తిరిగి ((దశాంశం *)(tensor->దత్తాంశం))[i];
            }
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

శూన్య ggml_సమితి_f32_1d(స్థిరరాశి నిర్మాణం ggml_tensor * tensor, పూర్ణాంకం i, దశాంశం విలువ) {
    ఒకవేళ (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_సూచీ(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_సమితి_f32_nd(tensor, id[0], id[1], id[2], id[3], విలువ);
        తిరిగి;
    }
    ఎంపిక (tensor->type) {
        స్థితి GGML_TYPE_I8:
            {
                ((int8_t *)(tensor->దత్తాంశం))[i] = విలువ;
            } ఆపు;
        స్థితి GGML_TYPE_I16:
            {
                ((int16_t *)(tensor->దత్తాంశం))[i] = విలువ;
            } ఆపు;
        స్థితి GGML_TYPE_I32:
            {
                ((int32_t *)(tensor->దత్తాంశం))[i] = విలువ;
            } ఆపు;
        స్థితి GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(tensor->దత్తాంశం))[i] = GGML_CPU_FP32_గమ్యం_FP16(విలువ);
            } ఆపు;
        స్థితి GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(tensor->దత్తాంశం))[i] = GGML_FP32_గమ్యం_BF16(విలువ);
            } ఆపు;
        స్థితి GGML_TYPE_F32:
            {
                ((దశాంశం *)(tensor->దత్తాంశం))[i] = విలువ;
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

దశాంశం ggml_get_f32_nd(స్థిరరాశి నిర్మాణం ggml_tensor * tensor, పూర్ణాంకం i0, పూర్ణాంకం i1, పూర్ణాంకం i2, పూర్ణాంకం i3) {
    శూన్య * దత్తాంశం   = (అక్షరం *) tensor->దత్తాంశం + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    ఎంపిక (tensor->type) {
        స్థితి GGML_TYPE_I8:
            తిరిగి ((int8_t *) దత్తాంశం)[0];
        స్థితి GGML_TYPE_I16:
            తిరిగి ((int16_t *) దత్తాంశం)[0];
        స్థితి GGML_TYPE_I32:
            తిరిగి ((int32_t *) దత్తాంశం)[0];
        స్థితి GGML_TYPE_F16:
            తిరిగి GGML_CPU_FP16_గమ్యం_FP32(((ggml_fp16_t *) దత్తాంశం)[0]);
        స్థితి GGML_TYPE_BF16:
            తిరిగి GGML_BF16_గమ్యం_FP32(((ggml_bf16_t *) దత్తాంశం)[0]);
        స్థితి GGML_TYPE_F32:
            తిరిగి ((దశాంశం *) దత్తాంశం)[0];
        అప్రమేయ:
            GGML_ABORT("fatal error");
    }
}

శూన్య ggml_సమితి_f32_nd(స్థిరరాశి నిర్మాణం ggml_tensor * tensor, పూర్ణాంకం i0, పూర్ణాంకం i1, పూర్ణాంకం i2, పూర్ణాంకం i3, దశాంశం విలువ) {
    శూన్య * దత్తాంశం   = (అక్షరం *) tensor->దత్తాంశం + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    ఎంపిక (tensor->type) {
        స్థితి GGML_TYPE_I8:
            {
                ((int8_t *)(దత్తాంశం))[0] = విలువ;
            } ఆపు;
        స్థితి GGML_TYPE_I16:
            {
                ((int16_t *)(దత్తాంశం))[0] = విలువ;
            } ఆపు;
        స్థితి GGML_TYPE_I32:
            {
                ((int32_t *)(దత్తాంశం))[0] = విలువ;
            } ఆపు;
        స్థితి GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(దత్తాంశం))[0] = GGML_CPU_FP32_గమ్యం_FP16(విలువ);
            } ఆపు;
        స్థితి GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(దత్తాంశం))[0] = GGML_FP32_గమ్యం_BF16(విలువ);
            } ఆపు;
        స్థితి GGML_TYPE_F32:
            {
                ((దశాంశం *)(దత్తాంశం))[0] = విలువ;
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

////////////////////////////////////////////////////////////////////////////////

// ggml_compute_forward_mul_mat

స్థిర శూన్య ggml_compute_forward_mul_mat_one_chunk(
    స్థిరరాశి నిర్మాణం ggml_compute_params * params,
    నిర్మాణం ggml_tensor * dst,
    స్థిరరాశి గణన ggml_type type,
    స్థిరరాశి int64_t అంకె_అడ్డువరుసలు_చొప్పున_vec_dot,
    స్థిరరాశి int64_t ir0_ప్రారంభం,
    స్థిరరాశి int64_t ir0_చివర,
    స్థిరరాశి int64_t ir1_ప్రారంభం,
    స్థిరరాశి int64_t ir1_చివర) {

    స్థిరరాశి నిర్మాణం ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి నిర్మాణం ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ద్విమాన_OP_LOCALS

    స్థిరరాశి bool src1_cont = ggml_is_contiguous(src1);

    ggml_vec_dot_t స్థిరరాశి vec_dot      = type_traits_cpu[type].vec_dot;
    గణన ggml_type స్థిరరాశి vec_dot_type = type_traits_cpu[type].vec_dot_type;

    // broadcast factors
    స్థిరరాశి int64_t r2 = ne12 / ne02;
    స్థిరరాశి int64_t r3 = ne13 / ne03;

    //printf("ir0_start = %6lld, ir0_end = %6lld, ir1_start = %6lld, ir1_end = %6lld\n", ir0_start, ir0_end, ir1_start, ir1_end);

    // threads with no work simply yield (not sure if it helps)
    ఒకవేళ (ir0_ప్రారంభం >= ir0_చివర || ir1_ప్రారంభం >= ir1_చివర) {
        తిరిగి;
    }

    స్థిరరాశి శూన్య * wdata = (src1->type == vec_dot_type) ? src1->దత్తాంశం : params->wdata;
    స్థిరరాశి size_t అడ్డువరుస_కొలత = ggml_అడ్డువరుస_కొలత(vec_dot_type, ne10);

    ప్రతిజ్ఞా(ne12 % ne02 == 0);
    ప్రతిజ్ఞా(ne13 % ne03 == 0);

    // block-tiling attempt
    స్థిరరాశి int64_t blck_0 = 16;
    స్థిరరాశి int64_t blck_1 = 16;

    స్థిరరాశి size_t src1_నిలువు_stride = src1_cont || src1->type != vec_dot_type ? అడ్డువరుస_కొలత : nb11;

    // attempt to reduce false-sharing (does not seem to make a difference)
    // 16 * 2, accounting for mmla kernels
    దశాంశం tmp[32];

    కొరకు (int64_t iir1 = ir1_ప్రారంభం; iir1 < ir1_చివర; iir1 += blck_1) {
        కొరకు (int64_t iir0 = ir0_ప్రారంభం; iir0 < ir0_చివర; iir0 += blck_0) {
            కొరకు (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_చివర; ir1 += అంకె_అడ్డువరుసలు_చొప్పున_vec_dot) {
                స్థిరరాశి int64_t i13 = (ir1 / (ne12 * ne1));
                స్థిరరాశి int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                స్థిరరాశి int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                // broadcast src0 into src1
                స్థిరరాశి int64_t i03 = i13 / r3;
                స్థిరరాశి int64_t i02 = i12 / r2;

                స్థిరరాశి int64_t i1 = i11;
                స్థిరరాశి int64_t i2 = i12;
                స్థిరరాశి int64_t i3 = i13;

                స్థిరరాశి అక్షరం * src0_అడ్డువరుస = (స్థిరరాశి అక్షరం*)src0->దత్తాంశం + (0 + i02 * nb02 + i03 * nb03);

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                స్థిరరాశి అక్షరం * src1_నిలువు = (స్థిరరాశి అక్షరం*)wdata +
                    (src1_cont || src1->type != vec_dot_type
                        ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * అడ్డువరుస_కొలత
                        : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                దశాంశం * dst_నిలువు = (దశాంశం*)((అక్షరం*)dst->దత్తాంశం + (i1 * nb1 + i2 * nb2 + i3 * nb3));

                //for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                //    vec_dot(ne00, &dst_col[ir0], src0_row + ir0*nb01, src1_col);
                //}

                కొరకు (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_చివర; ir0 += అంకె_అడ్డువరుసలు_చొప్పున_vec_dot) {
                    vec_dot(ne00, &tmp[ir0 - iir0], (అంకె_అడ్డువరుసలు_చొప్పున_vec_dot > 1 ? 16 : 0), src0_అడ్డువరుస + ir0 * nb01, (అంకె_అడ్డువరుసలు_చొప్పున_vec_dot > 1 ? nb01 : 0), src1_నిలువు, (అంకె_అడ్డువరుసలు_చొప్పున_vec_dot > 1 ? src1_నిలువు_stride : 0), అంకె_అడ్డువరుసలు_చొప్పున_vec_dot);
                }

                కొరకు (పూర్ణాంకం cn = 0; cn < అంకె_అడ్డువరుసలు_చొప్పున_vec_dot; ++cn) {
                    జ్ఞాపకం_నకలు(&dst_నిలువు[iir0 + cn * nb1 / nb0], tmp + (cn * 16), (MIN(iir0 + blck_0, ir0_చివర) - iir0) * పరిమాణం(దశాంశం));
                }
            }
        }
    }
}

శూన్య ggml_compute_forward_mul_mat(
        స్థిరరాశి నిర్మాణం ggml_compute_params * params,
              నిర్మాణం ggml_tensor * dst) {

    స్థిరరాశి నిర్మాణం ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి నిర్మాణం ggml_tensor * src1 = dst->src[1];

    స్థిరరాశి int32_t hint = ggml_get_op_params_i32(dst, 1);
    ఒకవేళ (hint == GGML_HINT_SRC0_IS_HADAMARD && !params->use_ref) {
        ggml_compute_forward_fwht(params, dst);
        తిరిగి;
    }

    // If tiled is supported, it will execute the full op here and we return
    ఒకవేళ (ggml_compute_forward_mul_mat_tiled(params, dst)) {
        తిరిగి;
    }

    GGML_TENSOR_ద్విమాన_OP_LOCALS

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    గణన ggml_type           స్థిరరాశి vec_dot_type         = type_traits_cpu[src0->type].vec_dot_type;
    ggml_from_float_t        స్థిరరాశి from_float           = type_traits_cpu[vec_dot_type].from_float;
    int64_t                  స్థిరరాశి vec_dot_అంకె_అడ్డువరుసలు     = type_traits_cpu[src0->type].nrows;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_కొలత(src0->type));
    GGML_ASSERT(nb10 == ggml_type_కొలత(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == పరిమాణం(దశాంశం));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    // TODO: extract to "extra_op"
#ఒకవేళ GGML_USE_LLAMAFILE
    // broadcast factors
    స్థిరరాశి int64_t r2 = ne12 / ne02;
    స్థిరరాశి int64_t r3 = ne13 / ne03;

    స్థిరరాశి bool src1_cont = ggml_is_contiguous(src1);

    ఒకవేళ (!params->use_ref && src1_cont) {
        కొరకు (int64_t i13 = 0; i13 < ne13; i13++)
            కొరకు (int64_t i12 = 0; i12 < ne12; i12++)
                ఒకవేళ (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_కొలత(src0->type),
                                     (స్థిరరాశి అక్షరం *)src0->దత్తాంశం + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_కొలత(src0->type),
                                     (స్థిరరాశి అక్షరం *)src1->దత్తాంశం + i12*nb12 + i13*nb13,
                                     nb11/ggml_type_కొలత(src1->type),
                                     (అక్షరం *)dst->దత్తాంశం + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_కొలత(dst->type),
                                     src0->type,
                                     src1->type,
                                     dst->type))
                    వెళ్ళు UseGgmlGemm1;
        తిరిగి;
    }
UseGgmlGemm1:;
#అయితే_ముగింపు

    ఒకవేళ (src1->type != vec_dot_type) {
        అక్షరం * wdata = params->wdata;

        స్థిరరాశి size_t nbw0 = ggml_type_కొలత(vec_dot_type);
        స్థిరరాశి size_t nbw1 = ggml_అడ్డువరుస_కొలత(vec_dot_type, ne10);
        స్థిరరాశి size_t nbw2 = nbw1*ne11;
        స్థిరరాశి size_t nbw3 = nbw2*ne12;

        ప్రతిజ్ఞా(params->wsize >= ne13*nbw3);
        // src1 is either packed from F32 into vec_dot_type, or widened from F16 or BF16 into the F32 work buffer
        స్థిరరాశి bool widen = src1->type != GGML_TYPE_F32;

        GGML_ASSERT(!widen || vec_dot_type == GGML_TYPE_F32);
        GGML_ASSERT(!widen || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_BF16);

    #ఒకవేళ 0
        కొరకు (int64_t i13 = 0; i13 < ne13; ++i13) {
            కొరకు (int64_t i12 = 0; i12 < ne12; ++i12) {
                కొరకు (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                    from_float((దశాంశం *)((అక్షరం *) src1->దత్తాంశం + i13*nb13 + i12*nb12 + i11*nb11),
                               (శూన్య *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                                ne10);
                }
            }
        }
    #లేకపోతే
        స్థిరరాశి int64_t bs = ggml_blck_కొలత(vec_dot_type);

        కొరకు (int64_t i13 = 0; i13 < ne13; ++i13) {
            కొరకు (int64_t i12 = 0; i12 < ne12; ++i12) {
                కొరకు (int64_t i11 = 0; i11 < ne11; ++i11) {
                    int64_t ne10_ఖండం_ప్రారంభం = (ith * ne10/bs) / nth;
                    int64_t ne10_ఖండం_చివర   = ((ith + 1) * ne10/bs) / nth;
                    స్థిరరాశి శూన్య * src1_అడ్డువరుస  = (స్థిరరాశి అక్షరం *) src1->దత్తాంశం + i13*nb13 + i12*nb12 + i11*nb11 + ne10_ఖండం_ప్రారంభం*bs*nb10;
                    శూన్య       * wdata_అడ్డువరుస =                    wdata  + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_ఖండం_ప్రారంభం*nbw0;

                    స్థిరరాశి int64_t ne10_ఖండం_కొలత = (ne10_ఖండం_చివర - ne10_ఖండం_ప్రారంభం) * bs;

                    ఒకవేళ (src1->type == GGML_TYPE_F16) {
                        ggml_cpu_fp16_గమ్యం_fp32((స్థిరరాశి ggml_fp16_t *) src1_అడ్డువరుస, (దశాంశం *) wdata_అడ్డువరుస, ne10_ఖండం_కొలత);
                    } లేకపోతే ఒకవేళ (src1->type == GGML_TYPE_BF16) {
                        ggml_cpu_bf16_గమ్యం_fp32((స్థిరరాశి ggml_bf16_t *) src1_అడ్డువరుస, (దశాంశం *) wdata_అడ్డువరుస, ne10_ఖండం_కొలత);
                    } లేకపోతే {
                        from_float((స్థిరరాశి దశాంశం *) src1_అడ్డువరుస, wdata_అడ్డువరుస, ne10_ఖండం_కొలత);
                    }
                }
            }
        }
    #అయితే_ముగింపు
    }

    ఒకవేళ (ith == 0) {
        // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
        atomic_store_explicit(&params->threadpool->current_chunk, nth, memory_order_relaxed);
    }

    ggml_barrier(params->threadpool);

#ఒకవేళ GGML_USE_LLAMAFILE
    ఒకవేళ (!params->use_ref && src1->type != vec_dot_type) {
        స్థిరరాశి శూన్య* wdata = (src1->type == vec_dot_type) ? src1->దత్తాంశం : params->wdata;
        స్థిరరాశి size_t అడ్డువరుస_కొలత = ggml_అడ్డువరుస_కొలత(vec_dot_type, ne10);

        కొరకు (int64_t i13 = 0; i13 < ne13; i13++)
            కొరకు (int64_t i12 = 0; i12 < ne12; i12++)
                ఒకవేళ (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_కొలత(src0->type),
                                     (స్థిరరాశి అక్షరం *)src0->దత్తాంశం + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_కొలత(src0->type),
                                     (స్థిరరాశి అక్షరం *)wdata + (i12*ne11 + i13*ne12*ne11)*అడ్డువరుస_కొలత,
                                     అడ్డువరుస_కొలత/ggml_type_కొలత(vec_dot_type),
                                     (అక్షరం *)dst->దత్తాంశం + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_కొలత(dst->type),
                                     src0->type,
                                     vec_dot_type,
                                     dst->type))
                    వెళ్ళు UseGgmlGemm2;
        తిరిగి;
    }
UseGgmlGemm2:;
#అయితే_ముగింపు

    // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
    స్థిరరాశి int64_t nr0 = ne0;

    // This is the size of the rest of the dimensions of the result
    స్థిరరాశి int64_t nr1 = ne1 * ne2 * ne3;

    // Now select a reasonable chunk size.
    పూర్ణాంకం chunk_కొలత = 16;

    // We need to step up the size if it's small
    ఒకవేళ (nr0 == 1 || nr1 == 1) {
        chunk_కొలత = 64;
    }

    // distribute the work across the inner or outer loop based on which one is larger
    // The number of chunks in the 0/1 dim.
    // CEIL(nr0/chunk_size)
    int64_t nchunk0 = (nr0 + chunk_కొలత - 1) / chunk_కొలత;
    int64_t nchunk1 = (nr1 + chunk_కొలత - 1) / chunk_కొలత;

    // If the chunking is poor for the number of threads on this setup, scrap the whole plan.  Re-chunk it by thread.
    //   Also, chunking by thread was measured to have perform better on NUMA systems.  See https://github.com/ggml-org/llama.cpp/pull/6915
    //   In theory, chunking should be just as useful on NUMA and non NUMA systems, but testing disagreed with that.
    ఒకవేళ (nchunk0 * nchunk1 < nth * 4 || ggml_is_numa()) {
        // distribute the thread work across the inner or outer loop based on which one is larger
        nchunk0 = nr0 > nr1 ? nth : 1; // parallelize by src0 rows
        nchunk1 = nr0 > nr1 ? 1 : nth; // parallelize by src1 rows
    }

    // The number of elements in each chunk
    స్థిరరాశి int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    స్థిరరాశి int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    // The first chunk comes from our thread_id, the rest will get auto-assigned.
    పూర్ణాంకం current_chunk = ith;

    అయితే (current_chunk < nchunk0 * nchunk1) {
        స్థిరరాశి int64_t ith0 = current_chunk % nchunk0;
        స్థిరరాశి int64_t ith1 = current_chunk / nchunk0;

        స్థిరరాశి int64_t ir0_ప్రారంభం = dr0 * ith0;
        స్థిరరాశి int64_t ir0_చివర = MIN(ir0_ప్రారంభం + dr0, nr0);

        స్థిరరాశి int64_t ir1_ప్రారంభం = dr1 * ith1;
        స్థిరరాశి int64_t ir1_చివర = MIN(ir1_ప్రారంభం + dr1, nr1);

        // dot kernels can handle 1 row and col at a time, but mmla kernels can process 2 rows and cols
        int64_t అంకె_అడ్డువరుసలు_చొప్పున_vec_dot = vec_dot_అంకె_అడ్డువరుసలు;

        // these checks are needed to avoid crossing dim1 boundaries
        // can be optimized, but the logic would become more complicated, so keeping it like this for simplicity
        ఒకవేళ ((nr0 % 2 != 0) || (ne11 % 2 != 0) || ((ir0_చివర - ir0_ప్రారంభం) % 2 != 0) || ((ir1_చివర - ir1_ప్రారంభం) % 2 != 0)) {
            అంకె_అడ్డువరుసలు_చొప్పున_vec_dot = 1;
        }
        ggml_compute_forward_mul_mat_one_chunk(params, dst, src0->type, అంకె_అడ్డువరుసలు_చొప్పున_vec_dot, ir0_ప్రారంభం, ir0_చివర, ir1_ప్రారంభం, ir1_చివర);

        ఒకవేళ (nth >= nchunk0 * nchunk1) {
            ఆపు;
        }

        current_chunk = atomic_fetch_కూడు_explicit(&params->threadpool->current_chunk, 1, memory_order_relaxed);
    }
}

// ggml_compute_forward_mul_mat_id

#నిర్వచించు MMID_మాతృక_అడ్డువరుస(అడ్డువరుస_id, i1) మాతృక_అడ్డువరుసలు[(అడ్డువరుస_id)*ids->ne[0]*ids->ne[1] + (i1)]

నిర్మాణం mmid_అడ్డువరుస_mapping {
    int32_t i1;
    int32_t i2;
};

స్థిర శూన్య ggml_compute_forward_mul_mat_id_one_chunk(
    నిర్మాణం ggml_tensor * dst,
    స్థిరరాశి నిర్మాణం ggml_tensor * src0,
    స్థిరరాశి నిర్మాణం ggml_tensor * src1,
    స్థిరరాశి నిర్మాణం ggml_tensor * ids,
    స్థిరరాశి int64_t cur_a,
    స్థిరరాశి int64_t ir0_ప్రారంభం,
    స్థిరరాశి int64_t ir0_చివర,
    స్థిరరాశి int64_t ir1_ప్రారంభం,
    స్థిరరాశి int64_t ir1_చివర,
    స్థిరరాశి అక్షరం * src0_cur,
    స్థిరరాశి నిర్మాణం mmid_అడ్డువరుస_mapping * మాతృక_అడ్డువరుసలు,
    స్థిరరాశి size_t అడ్డువరుస_కొలత,
    స్థిరరాశి bool src1_cont,
    స్థిరరాశి శూన్య * wdata) {

    GGML_TENSOR_ద్విమాన_OP_LOCALS

    స్థిరరాశి గణన ggml_type type = src0->type;

    ggml_vec_dot_t    స్థిరరాశి vec_dot      = type_traits_cpu[type].vec_dot;
    గణన ggml_type    స్థిరరాశి vec_dot_type = type_traits_cpu[type].vec_dot_type;

    స్థిరరాశి int64_t blck_0 = 16;
    స్థిరరాశి int64_t blck_1 = 16;

    దశాంశం tmp[16];

    కొరకు (int64_t iir1 = ir1_ప్రారంభం; iir1 < ir1_చివర; iir1 += blck_1) {
        కొరకు (int64_t iir0 = ir0_ప్రారంభం; iir0 < ir0_చివర; iir0 += blck_0) {
            కొరకు (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_చివర; ++ir1) {
                స్థిరరాశి int64_t _i12 = ir1; // logical row index for this expert

                నిర్మాణం mmid_అడ్డువరుస_mapping అడ్డువరుస_mapping = MMID_మాతృక_అడ్డువరుస(cur_a, _i12);
                స్థిరరాశి పూర్ణాంకం id       = అడ్డువరుస_mapping.i1; // selected expert index

                స్థిరరాశి int64_t  i11 = id % ne11;
                స్థిరరాశి int64_t  i12 = అడ్డువరుస_mapping.i2; // row index in src1

                స్థిరరాశి int64_t  i1 = id;  // selected expert index
                స్థిరరాశి int64_t  i2 = i12; //అడ్డువరుస

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                స్థిరరాశి అక్షరం * src1_నిలువు = (స్థిరరాశి అక్షరం *) wdata +
                    (src1_cont || src1->type != vec_dot_type
                    ? (i11      + i12*ne11)*అడ్డువరుస_కొలత
                    : (i11*nb11 + i12*nb12));

                దశాంశం * dst_నిలువు = (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + (i1*nb1 + i2*nb2));

                కొరకు (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_చివర; ++ir0) {
                    vec_dot(ne00, &tmp[ir0 - iir0], 0, src0_cur + ir0*nb01, 0, src1_నిలువు, 0, 1);
                }

                జ్ఞాపకం_నకలు(&dst_నిలువు[iir0], tmp, (MIN(iir0 + blck_0, ir0_చివర) - iir0)*పరిమాణం(దశాంశం));
            }
        }
    }
}

స్థిర శూన్య * incr_ptr_aligned(శూన్య ** p, size_t కొలత, size_t align) {

    శూన్య * ptr = *p;
    ptr = (శూన్య *) GGML_PAD((uintptr_t) ptr, align);
    *p = (శూన్య *) ((అక్షరం *) ptr + కొలత);
    తిరిగి ptr;
}

స్థిర శూన్య ggml_compute_forward_mul_mat_id(
        స్థిరరాశి నిర్మాణం ggml_compute_params * params,
              నిర్మాణం ggml_tensor * dst) {

    స్థిరరాశి నిర్మాణం ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి నిర్మాణం ggml_tensor * src1 = dst->src[1];
    స్థిరరాశి నిర్మాణం ggml_tensor * ids = dst->src[2];

    GGML_TENSOR_ద్విమాన_OP_LOCALS

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి గణన ggml_type type = src0->type;

    స్థిరరాశి bool src1_cont = ggml_is_contiguous(src1);

    గణన ggml_type    స్థిరరాశి vec_dot_type    = type_traits_cpu[type].vec_dot_type;
    ggml_from_float_t స్థిరరాశి from_float      = type_traits_cpu[vec_dot_type].from_float;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_కొలత(type));
    GGML_ASSERT(nb10 == ggml_type_కొలత(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == పరిమాణం(దశాంశం));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // row groups
    స్థిరరాశి పూర్ణాంకం n_ids = ids->ne[0]; // n_expert_used
    స్థిరరాశి పూర్ణాంకం n_as  = ne02;       // n_expert

    శూన్య * wdata_cur = params->wdata;

    ఒకవేళ (src1->type != vec_dot_type) {
        incr_ptr_aligned(&wdata_cur, ggml_అడ్డువరుస_కొలత(vec_dot_type, ggml_nelements(src1)), పరిమాణం(int64_t));
    }

    int64_t * మాతృక_అడ్డువరుస_counts = // [n_as]
        incr_ptr_aligned(&wdata_cur, n_as*పరిమాణం(int64_t), పరిమాణం(int64_t));

    నిర్మాణం mmid_అడ్డువరుస_mapping * మాతృక_అడ్డువరుసలు = // [n_as][ids->ne[0]*ids->ne[1]]
        incr_ptr_aligned(&wdata_cur, n_as*ids->ne[0]*ids->ne[1]*పరిమాణం(నిర్మాణం mmid_అడ్డువరుస_mapping), పరిమాణం(int64_t));

    అక్షరం (*atomic_current_chunk)[CACHE_పంక్తి_కొలత] = // [n_as]
        incr_ptr_aligned(&wdata_cur, CACHE_పంక్తి_కొలత * n_as, CACHE_పంక్తి_కొలత);

    // Tiled matmul (see tiled.h); per-thread work buffers, 0 bytes when disabled. The
    // reservation is unconditional, the per expert eligibility is decided at dispatch time
    అక్షరం * tiled_scratch = incr_ptr_aligned(&wdata_cur, ggml_tiled_wdata_కొలత(nth, dst), 64);

    GGML_ASSERT(params->wsize >= (size_t)((అక్షరం *) wdata_cur - (అక్షరం *) params->wdata));

    ఒకవేళ (src1->type != vec_dot_type) {
        అక్షరం * wdata = params->wdata;

        స్థిరరాశి size_t nbw0 = ggml_type_కొలత(vec_dot_type);
        స్థిరరాశి size_t nbw1 = ggml_అడ్డువరుస_కొలత(vec_dot_type, ne10);
        స్థిరరాశి size_t nbw2 = nbw1*ne11;
        స్థిరరాశి size_t nbw3 = nbw2*ne12;

        ప్రతిజ్ఞా(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

#ఒకవేళ 0
        కొరకు (int64_t i13 = 0; i13 < ne13; ++i13) {
            కొరకు (int64_t i12 = ith; i12 < ne12; i12 += nth) {
                కొరకు (int64_t i11 = 0; i11 < ne11; ++i11) {
                    from_float((దశాంశం *)((అక్షరం *) src1->దత్తాంశం + i13*nb13 + i12*nb12 + i11*nb11),
                               (శూన్య *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                               ne10);
                }
            }
        }
#లేకపోతే
        కొరకు (int64_t i13 = 0; i13 < ne13; ++i13) {
            కొరకు (int64_t i12 = 0; i12 < ne12; ++i12) {
                కొరకు (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = ggml_blck_కొలత(vec_dot_type);
                    int64_t ne10_ఖండం_ప్రారంభం = (ith * ne10/bs) / nth;
                    int64_t ne10_ఖండం_చివర   = ((ith + 1) * ne10/bs) / nth;
                    from_float((దశాంశం *)((అక్షరం *) src1->దత్తాంశం + i13*nb13 + i12*nb12 + i11*nb11 + ne10_ఖండం_ప్రారంభం*bs*nb10),
                               (శూన్య *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_ఖండం_ప్రారంభం*nbw0),
                               (ne10_ఖండం_చివర - ne10_ఖండం_ప్రారంభం) * bs);
                }
            }
        }
#అయితే_ముగింపు
    }

    ఒకవేళ (ith == 0) {
        // initialize matrix_row_counts
        memset(మాతృక_అడ్డువరుస_counts, 0, n_as*పరిమాణం(int64_t));

        // group rows by src0 matrix
        కొరకు (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            కొరకు (పూర్ణాంకం id = 0; id < n_ids; ++id) {
                స్థిరరాశి int32_t i02 = *(స్థిరరాశి int32_t *) ((స్థిరరాశి అక్షరం *) ids->దత్తాంశం + iid1*ids->nb[1] + id*ids->nb[0]);

                ప్రతిజ్ఞా(i02 >= 0 && i02 < n_as);

                MMID_మాతృక_అడ్డువరుస(i02, మాతృక_అడ్డువరుస_counts[i02]) = (నిర్మాణం mmid_అడ్డువరుస_mapping) {id, iid1};
                మాతృక_అడ్డువరుస_counts[i02] += 1;
            }
        }
    }

    // reset current_chunk
    కొరకు (పూర్ణాంకం cur_a = ith; cur_a < n_as; cur_a += nth) {
        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);
        *current_chunk_ctr = nth;
    }

    ggml_barrier(params->threadpool);

    కొరకు (పూర్ణాంకం cur_a = 0; cur_a < n_as; ++cur_a) {
        స్థిరరాశి int64_t cne1 = మాతృక_అడ్డువరుస_counts[cur_a];

        ఒకవేళ (cne1 == 0) {
            కొనసాగు;
        }

        // tiled takes over if profitable for this expert (see tiled.h)
        ఒకవేళ (ggml_compute_forward_mul_mat_id_tiled(params, dst, cur_a, cne1, (స్థిరరాశి int32_t *) &MMID_మాతృక_అడ్డువరుస(cur_a, 0), tiled_scratch)) {
            కొనసాగు;
        }

        స్థిరరాశి అక్షరం * src0_cur = (స్థిరరాశి అక్షరం *) src0->దత్తాంశం + cur_a * nb02;
        స్థిరరాశి శూన్య * wdata = (src1->type == vec_dot_type) ? src1->దత్తాంశం : params->wdata;
        స్థిరరాశి size_t అడ్డువరుస_కొలత = ggml_అడ్డువరుస_కొలత(vec_dot_type, ne10);

        స్థిరరాశి int64_t nr0 = ne01;
        స్థిరరాశి int64_t nr1 = cne1;

        పూర్ణాంకం chunk_కొలత = 16;
        ఒకవేళ (nr0 == 1 || nr1 == 1) {
            chunk_కొలత = 64;
        }

        // disable for NUMA
        స్థిరరాశి bool disable_chunking = ggml_is_numa();

        int64_t nchunk0 = (nr0 + chunk_కొలత - 1) / chunk_కొలత;
        int64_t nchunk1 = (nr1 + chunk_కొలత - 1) / chunk_కొలత;

        ఒకవేళ (nchunk0 * nchunk1 < nth * 4 || disable_chunking) {
            nchunk0 = nr0 > nr1 ? nth : 1;
            nchunk1 = nr0 > nr1 ? 1 : nth;
        }

        స్థిరరాశి int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        స్థిరరాశి int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

        పూర్ణాంకం current_chunk = ith;

        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);

        అయితే (current_chunk < nchunk0 * nchunk1) {
            స్థిరరాశి int64_t ith0 = current_chunk % nchunk0;
            స్థిరరాశి int64_t ith1 = current_chunk / nchunk0;

            స్థిరరాశి int64_t ir0_ప్రారంభం = dr0 * ith0;
            స్థిరరాశి int64_t ir0_చివర = MIN(ir0_ప్రారంభం + dr0, nr0);

            స్థిరరాశి int64_t ir1_ప్రారంభం = dr1 * ith1;
            స్థిరరాశి int64_t ir1_చివర = MIN(ir1_ప్రారంభం + dr1, nr1);

            ggml_compute_forward_mul_mat_id_one_chunk(
                dst, src0, src1, ids, cur_a,
                ir0_ప్రారంభం, ir0_చివర, ir1_ప్రారంభం, ir1_చివర,
                src0_cur, మాతృక_అడ్డువరుసలు, అడ్డువరుస_కొలత, src1_cont, wdata
            );

            ఒకవేళ (nth >= nchunk0 * nchunk1) {
                ఆపు;
            }

            current_chunk = atomic_fetch_కూడు_explicit(current_chunk_ctr, 1, memory_order_relaxed);
        }
    }
}

/////////////////////////////////

స్థిర శూన్య ggml_compute_forward(నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * tensor) {
    GGML_ASSERT(params);

    ఒకవేళ (tensor->op == GGML_OP_NONE || ggml_is_ఖాళీలు(tensor)) {
        తిరిగి;
    }

    // extra_buffer op?
    ఒకవేళ (ggml_cpu_extra_compute_forward(params, tensor)) {
        తిరిగి;
    }

    ఎంపిక (tensor->op) {
        స్థితి GGML_OP_DUP:
            {
                ggml_compute_forward_dup(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_కూడు:
            {
                ggml_compute_forward_కూడు(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_కూడు_ID:
            {
                ggml_compute_forward_కూడు_id(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_ADD1:
            {
                ggml_compute_forward_add1(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_ACC:
            {
                ggml_compute_forward_acc(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_SUB:
            {
                ggml_compute_forward_sub(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_MUL:
            {
                ggml_compute_forward_mul(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_DIV:
            {
                ggml_compute_forward_div(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_SQR:
            {
                ggml_compute_forward_sqr(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_SQRT:
            {
                ggml_compute_forward_sqrt(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_LOG:
            {
                ggml_compute_forward_log(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_SIN:
            {
                ggml_compute_forward_sin(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_COS:
            {
                ggml_compute_forward_cos(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_కూడిక:
            {
                ggml_compute_forward_కూడిక(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_కూడిక_అడ్డువరుసలు:
            {
                ggml_compute_forward_కూడిక_అడ్డువరుసలు(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_CUMSUM:
            {
                ggml_compute_forward_cumsum(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_సగటు:
            {
                ggml_compute_forward_సగటు(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_ARGMAX:
            {
                ggml_compute_forward_argmax(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_లెక్క_EQUAL:
            {
                ggml_compute_forward_లెక్క_equal(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_REPEAT:
            {
                ggml_compute_forward_repeat(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_REPEAT_వెనుక:
            {
                ggml_compute_forward_repeat_వెనుక(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_CONCAT:
            {
                ggml_compute_forward_concat(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_SILU_వెనుక:
            {
                ggml_compute_forward_silu_వెనుక(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_NORM:
            {
                ggml_compute_forward_norm(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_RMS_NORM:
            {
                ggml_compute_forward_rms_norm(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_RMS_NORM_వెనుక:
            {
                ggml_compute_forward_rms_norm_వెనుక(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_సమూహం_NORM:
            {
                ggml_compute_forward_సమూహం_norm(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_L2_NORM:
            {
                ggml_compute_forward_l2_norm(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_MUL_MAT:
            {
                ggml_compute_forward_mul_mat(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_MUL_MAT_ID:
            {
                ggml_compute_forward_mul_mat_id(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_బహిర్గతం_PROD:
            {
                ggml_compute_forward_బహిర్గతం_prod(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_SCALE:
            {
                ggml_compute_forward_scale(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_సమితి:
            {
                ggml_compute_forward_సమితి(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_CPY:
            {
                ggml_compute_forward_cpy(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_CONT:
            {
                ggml_compute_forward_cont(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_GET_అడ్డువరుసలు:
            {
                ggml_compute_forward_get_అడ్డువరుసలు(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_GET_అడ్డువరుసలు_వెనుక:
            {
                ggml_compute_forward_get_అడ్డువరుసలు_వెనుక(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_సమితి_అడ్డువరుసలు:
            {
                ggml_compute_forward_సమితి_అడ్డువరుసలు(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_DIAG:
            {
                ggml_compute_forward_diag(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_DIAG_ముసుగు_INF:
            {
                ggml_compute_forward_diag_ముసుగు_inf(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_DIAG_ముసుగు_ZERO:
            {
                ggml_compute_forward_diag_ముసుగు_zero(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_SOFT_MAX:
            {
                ggml_compute_forward_soft_max(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_SOFT_MAX_వెనుక:
            {
                ggml_compute_forward_soft_max_ext_వెనుక(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_ROPE:
            {
                ggml_compute_forward_rope(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_ROPE_వెనుక:
            {
                ggml_compute_forward_rope_వెనుక(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_CLAMP:
            {
                ggml_compute_forward_clamp(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_compute_forward_conv_transpose_1d(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_IM2_నిలువు:
            {
                ggml_compute_forward_im2col(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_IM2_నిలువు_వెనుక:
            {
                ggml_compute_forward_im2col_వెనుక_f32(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_IM2_నిలువు_3_D:
            {
                ggml_compute_forward_im2col_3d(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_COL2IM_1D:
            {
                ggml_compute_forward_col2im_1d(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_CONV_2D:
            {
                ggml_compute_forward_conv_2d(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_CONV_3D:
            {
                ggml_compute_forward_conv_3d(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_CONV_2D_DW:
            {
                ggml_compute_forward_conv_2d_dw(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_CONV_TRANSPOSE_2D:
            {
                ggml_compute_forward_conv_transpose_2d(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_POOL_1D:
            {
                ggml_compute_forward_pool_1d(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_POOL_2D:
            {
                ggml_compute_forward_pool_2d(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_POOL_2_D_వెనుక:
            {
                ggml_compute_forward_pool_2d_వెనుక(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_UPSCALE:
            {
                ggml_compute_forward_upscale(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_PAD:
            {
                ggml_compute_forward_pad(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_PAD_REFLECT_1D:
            {
                ggml_compute_forward_pad_reflect_1d(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_ROLL:
            {
                ggml_compute_forward_roll(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_ARANGE:
            {
                ggml_compute_forward_arange(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_TIMESTEP_EMBEDDING:
            {
                ggml_compute_forward_timestep_embedding(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_ARGSORT:
            {
                ggml_compute_forward_argsort(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_శిఖరం_K:
            {
                ggml_compute_forward_శిఖరం_k(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_LEAKY_RELU:
            {
                ggml_compute_forward_leaky_relu(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_TRI:
            {
                ggml_compute_forward_tri(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_FILL:
            {
                ggml_compute_forward_fill(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_FLASH_ATTN_EXT:
            {
                ggml_compute_forward_flash_attn_ext(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_FLASH_ATTN_వెనుక:
            {
                int32_t t = ggml_get_op_params_i32(tensor, 0);
                GGML_ASSERT(t == 0 || t == 1);
                bool masked = t != 0;
                ggml_compute_forward_flash_attn_వెనుక(params, masked, tensor);
            } ఆపు;
        స్థితి GGML_OP_SSM_CONV:
            {
                ggml_compute_forward_ssm_conv(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_SSM_పరిశీలన:
            {
                ggml_compute_forward_ssm_పరిశీలన(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_WIN_PART:
            {
                ggml_compute_forward_win_part(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_WIN_UNPART:
            {
                ggml_compute_forward_win_unpart(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_UNARY:
            {
                ggml_compute_forward_unary(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_GLU:
            {
                ggml_compute_forward_glu(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_GET_REL_POS:
            {
                ggml_compute_forward_get_rel_pos(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_కూడు_REL_POS:
            {
                ggml_compute_forward_కూడు_rel_pos(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_RWKV_WKV6:
            {
                ggml_compute_forward_rwkv_wkv6(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_GATED_రేఖీయ_ATTN:
            {
                ggml_compute_forward_gla(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_RWKV_WKV7:
            {
                ggml_compute_forward_rwkv_wkv7(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_SOLVE_TRI:
            {
                ggml_compute_forward_solve_tri(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_GATED_DELTA_NET:
            {
                ggml_compute_forward_gated_delta_net(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_LIGHTNING_INDEXER:
            {
                ggml_compute_forward_lightning_indexer(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_DSV4_HC_COMB:
            {
                ggml_compute_forward_dsv4_hc_comb(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_DSV4_HC_PRE:
            {
                ggml_compute_forward_dsv4_hc_pre(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_DSV4_HC_POST:
            {
                ggml_compute_forward_dsv4_hc_post(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_శిఖరం_K_ముసుగు:
            {
                ggml_compute_forward_శిఖరం_k_ముసుగు(params, tensor);
            } ఆపు;
        స్థితి GGML_OP_MAP_CUSTOM1:
            {
                ggml_compute_forward_map_custom1(params, tensor);
            }
            ఆపు;
        స్థితి GGML_OP_MAP_CUSTOM2:
            {
                ggml_compute_forward_map_custom2(params, tensor);
            }
            ఆపు;
        స్థితి GGML_OP_MAP_CUSTOM3:
            {
                ggml_compute_forward_map_custom3(params, tensor);
            }
            ఆపు;
        స్థితి GGML_OP_CUSTOM:
            {
                ggml_compute_forward_custom(params, tensor);
            }
            ఆపు;
        స్థితి GGML_OP_CROSS_ENTROPY_LOSS:
            {
                ggml_compute_forward_cross_entropy_loss(params, tensor);
            }
            ఆపు;
        స్థితి GGML_OP_CROSS_ENTROPY_LOSS_వెనుక:
            {
                ggml_compute_forward_cross_entropy_loss_వెనుక(params, tensor);
            }
            ఆపు;
        స్థితి GGML_OP_OPT_అడుగు_ADAMW:
            {
                ggml_compute_forward_opt_అడుగు_adamw(params, tensor);
            }
            ఆపు;
        స్థితి GGML_OP_OPT_అడుగు_SGD:
            {
                ggml_compute_forward_opt_అడుగు_sgd(params, tensor);
            }
            ఆపు;
        స్థితి GGML_OP_NONE:
            {
                // nop
            } ఆపు;
        స్థితి GGML_OP_RESHAPE:
            {
                // nop
            } ఆపు;
        స్థితి GGML_OP_క్రమమార్పు:
            {
                // nop
            } ఆపు;
        స్థితి GGML_OP_VIEW:
            {
                // nop
            } ఆపు;
        స్థితి GGML_OP_TRANSPOSE:
            {
                // nop
            } ఆపు;
        స్థితి GGML_OP_లెక్క:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// Android's libc implementation "bionic" does not support setting affinity
#ఒకవేళ defined(__gnu_linux__)
స్థిర శూన్య సమితి_numa_thread_affinity(పూర్ణాంకం thread_n) {
    ఒకవేళ (!ggml_is_numa()) {
        తిరిగి;
    }

    పూర్ణాంకం కణుపు_అంకె;
    పూర్ణాంకం rv;
    size_t setsize = CPU_ALLOC_కొలత(g_state.numa.మొత్తం_cpus);

    ఎంపిక(g_state.numa.numa_strategy) {
        స్థితి GGML_NUMA_STRATEGY_DISTRIBUTE:
            // run thread on node_num thread_n / (threads per node)
            కణుపు_అంకె = thread_n % g_state.numa.n_nodes;
            ఆపు;
        స్థితి GGML_NUMA_STRATEGY_ISOLATE:
            // run thread on current_node
            కణుపు_అంకె = g_state.numa.current_కణుపు;
            ఆపు;
        స్థితి GGML_NUMA_STRATEGY_NUMACTL:
            // use the cpuset that numactl gave us
            rv = pthread_setaffinity_np(pthread_self(), setsize, &g_state.numa.cpuset);
            ఒకవేళ (rv) {
                fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n",strerror(rv));
            }
            తిరిగి;
        అప్రమేయ:
            తిరిగి;
    }

    నిర్మాణం ggml_numa_కణుపు * కణుపు = &g_state.numa.nodes[కణుపు_అంకె];

    cpu_సమితి_t * cpus = CPU_ALLOC(g_state.numa.మొత్తం_cpus);
    CPU_ZERO_S(setsize, cpus);
    కొరకు (size_t i = 0; i < కణుపు->n_cpus; ++i) {
        CPU_సమితి_S(కణుపు->cpus[i], setsize, cpus);
    }

    rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    ఒకవేళ (rv) {
            fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}

స్థిర శూన్య clear_numa_thread_affinity(శూన్య) {
    ఒకవేళ (!ggml_is_numa()) {
        తిరిగి;
    }

    size_t setsize = CPU_ALLOC_కొలత(g_state.numa.మొత్తం_cpus);

    cpu_సమితి_t * cpus = CPU_ALLOC(g_state.numa.మొత్తం_cpus);
    CPU_ZERO_S(setsize, cpus);
    కొరకు (గుర్తులేని i = 0; i < g_state.numa.మొత్తం_cpus; ++i) {
        CPU_సమితి_S(i, setsize, cpus);
    }

    పూర్ణాంకం rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    ఒకవేళ (rv) {
        fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}
#లేకపోతే
// TODO: Windows etc.
// (the linux implementation may also work on BSD, someone should test)
స్థిర శూన్య సమితి_numa_thread_affinity(పూర్ణాంకం thread_n) { వాడని(thread_n);  }
స్థిర శూన్య clear_numa_thread_affinity(శూన్య) {}
#అయితే_ముగింపు

స్థిర పూర్ణాంకం ggml_get_n_tasks(నిర్మాణం ggml_tensor * కణుపు, పూర్ణాంకం n_దారాలు) {
    పూర్ణాంకం n_tasks = 0;

    ఒకవేళ (ggml_is_ఖాళీలు(కణుపు)) {
        // no need to multi-thread a no-op
        n_tasks = 1;
        తిరిగి n_tasks;
    }

    ఎంపిక (కణుపు->op) {
        స్థితి GGML_OP_CPY:
        స్థితి GGML_OP_DUP:
        స్థితి GGML_OP_CONT:
        స్థితి GGML_OP_కూడు:
        స్థితి GGML_OP_కూడు_ID:
        స్థితి GGML_OP_ADD1:
        స్థితి GGML_OP_ACC:
        స్థితి GGML_OP_CUMSUM:
        స్థితి GGML_OP_TRI:
        స్థితి GGML_OP_FILL:
            {
                n_tasks = n_దారాలు;
            } ఆపు;
        స్థితి GGML_OP_SUB:
        స్థితి GGML_OP_SQR:
        స్థితి GGML_OP_SQRT:
        స్థితి GGML_OP_LOG:
        స్థితి GGML_OP_SIN:
        స్థితి GGML_OP_COS:
        స్థితి GGML_OP_కూడిక:
        స్థితి GGML_OP_కూడిక_అడ్డువరుసలు:
        స్థితి GGML_OP_సగటు:
        స్థితి GGML_OP_ARGMAX:
            {
                n_tasks = 1;
            } ఆపు;
        స్థితి GGML_OP_లెక్క_EQUAL:
        స్థితి GGML_OP_SOLVE_TRI:
        స్థితి GGML_OP_GATED_DELTA_NET:
        స్థితి GGML_OP_DSV4_HC_COMB:
        స్థితి GGML_OP_DSV4_HC_PRE:
        స్థితి GGML_OP_DSV4_HC_POST:
        స్థితి GGML_OP_శిఖరం_K_ముసుగు:
            {
                n_tasks = n_దారాలు;
            } ఆపు;
        స్థితి GGML_OP_REPEAT:
        స్థితి GGML_OP_REPEAT_వెనుక:
        స్థితి GGML_OP_LEAKY_RELU:
            {
                n_tasks = 1;
            } ఆపు;
        స్థితి GGML_OP_UNARY:
            ఎంపిక (ggml_get_unary_op(కణుపు)) {
                స్థితి GGML_UNARY_OP_ABS:
                స్థితి GGML_UNARY_OP_SGN:
                స్థితి GGML_UNARY_OP_NEG:
                స్థితి GGML_UNARY_OP_అడుగు:
                స్థితి GGML_UNARY_OP_TANH:
                స్థితి GGML_UNARY_OP_ELU:
                స్థితి GGML_UNARY_OP_RELU:
                స్థితి GGML_UNARY_OP_SIGMOID:
                స్థితి GGML_UNARY_OP_HARDSWISH:
                స్థితి GGML_UNARY_OP_HARDSIGMOID:
                స్థితి GGML_UNARY_OP_EXP:
                స్థితి GGML_UNARY_OP_SOFTPLUS:
                స్థితి GGML_UNARY_OP_EXPM1:
                స్థితి GGML_UNARY_OP_FLOOR:
                స్థితి GGML_UNARY_OP_CEIL:
                స్థితి GGML_UNARY_OP_విడత:
                స్థితి GGML_UNARY_OP_TRUNC:
                    {
                        n_tasks = 1;
                    } ఆపు;

                స్థితి GGML_UNARY_OP_GELU:
                స్థితి GGML_UNARY_OP_GELU_ERF:
                స్థితి GGML_UNARY_OP_GELU_శీఘ్ర:
                స్థితి GGML_UNARY_OP_SILU:
                స్థితి GGML_UNARY_OP_XIELU:
                    {
                        n_tasks = n_దారాలు;
                    } ఆపు;
                అప్రమేయ:
                    GGML_ABORT("fatal error");
            }
            ఆపు;
        స్థితి GGML_OP_GLU:
            ఎంపిక (ggml_get_glu_op(కణుపు)) {
                స్థితి GGML_GLU_OP_REGLU:
                స్థితి GGML_GLU_OP_GEGLU:
                స్థితి GGML_GLU_OP_SWIGLU:
                స్థితి GGML_GLU_OP_SWIGLU_OAI:
                స్థితి GGML_GLU_OP_GEGLU_ERF:
                స్థితి GGML_GLU_OP_GEGLU_శీఘ్ర:
                స్థితి GGML_GLU_OP_SWIGLU_CLAMP:
                    {
                        n_tasks = n_దారాలు;
                    } ఆపు;
                అప్రమేయ:
                    GGML_ABORT("fatal error");
            }
            ఆపు;
        స్థితి GGML_OP_SILU_వెనుక:
        స్థితి GGML_OP_MUL:
        స్థితి GGML_OP_DIV:
        స్థితి GGML_OP_NORM:
        స్థితి GGML_OP_RMS_NORM:
        స్థితి GGML_OP_RMS_NORM_వెనుక:
        స్థితి GGML_OP_L2_NORM:
        స్థితి GGML_OP_సమూహం_NORM:
        స్థితి GGML_OP_CONCAT:
        స్థితి GGML_OP_MUL_MAT:
        స్థితి GGML_OP_MUL_MAT_ID:
        స్థితి GGML_OP_బహిర్గతం_PROD:
        స్థితి GGML_OP_సమితి_అడ్డువరుసలు:
            {
                n_tasks = n_దారాలు;
            } ఆపు;
        స్థితి GGML_OP_GET_అడ్డువరుసలు:
            {
                // FIXME: get_rows can use additional threads, but the cost of launching additional threads
                // decreases performance with GPU offloading
                //n_tasks = n_threads;
                n_tasks = 1;
            } ఆపు;
        స్థితి GGML_OP_SCALE:
        స్థితి GGML_OP_సమితి:
        స్థితి GGML_OP_RESHAPE:
        స్థితి GGML_OP_VIEW:
        స్థితి GGML_OP_క్రమమార్పు:
        స్థితి GGML_OP_TRANSPOSE:
        స్థితి GGML_OP_GET_అడ్డువరుసలు_వెనుక:
        స్థితి GGML_OP_DIAG:
            {
                n_tasks = 1;
            } ఆపు;
        స్థితి GGML_OP_DIAG_ముసుగు_ZERO:
        స్థితి GGML_OP_DIAG_ముసుగు_INF:
        స్థితి GGML_OP_SOFT_MAX_వెనుక:
        స్థితి GGML_OP_ROPE:
        స్థితి GGML_OP_ROPE_వెనుక:
        స్థితి GGML_OP_కూడు_REL_POS:
            {
                n_tasks = n_దారాలు;
            } ఆపు;
        స్థితి GGML_OP_CLAMP:
            {
                n_tasks = 1; //TODO
            } ఆపు;
        స్థితి GGML_OP_SOFT_MAX:
            {
                n_tasks = MIN(n_దారాలు, ggml_nrows(కణుపు->src[0]));
            } ఆపు;
        స్థితి GGML_OP_IM2_నిలువు:
        స్థితి GGML_OP_IM2_నిలువు_వెనుక:
        స్థితి GGML_OP_IM2_నిలువు_3_D:
        స్థితి GGML_OP_CONV_2D:
        స్థితి GGML_OP_CONV_3D:
        స్థితి GGML_OP_CONV_2D_DW:
        స్థితి GGML_OP_COL2IM_1D:
        స్థితి GGML_OP_CONV_TRANSPOSE_1D:
        స్థితి GGML_OP_CONV_TRANSPOSE_2D:
            {
                n_tasks = n_దారాలు;
            } ఆపు;
        స్థితి GGML_OP_POOL_1D:
        స్థితి GGML_OP_POOL_2D:
        స్థితి GGML_OP_POOL_2_D_వెనుక:
            {
                n_tasks = 1;
            } ఆపు;
        స్థితి GGML_OP_UPSCALE:
        స్థితి GGML_OP_PAD:
        స్థితి GGML_OP_PAD_REFLECT_1D:
        స్థితి GGML_OP_ROLL:
        స్థితి GGML_OP_ARANGE:
        స్థితి GGML_OP_TIMESTEP_EMBEDDING:
        స్థితి GGML_OP_ARGSORT:
        స్థితి GGML_OP_శిఖరం_K:
        స్థితి GGML_OP_FLASH_ATTN_EXT:
        స్థితి GGML_OP_FLASH_ATTN_వెనుక:
        స్థితి GGML_OP_SSM_CONV:
        స్థితి GGML_OP_SSM_పరిశీలన:
        స్థితి GGML_OP_LIGHTNING_INDEXER:
            {
                n_tasks = n_దారాలు;
            } ఆపు;
        స్థితి GGML_OP_RWKV_WKV6:
        స్థితి GGML_OP_GATED_రేఖీయ_ATTN:
        స్థితి GGML_OP_RWKV_WKV7:
            {
                స్థిరరాశి int64_t n_heads = కణుపు->src[1]->ne[1];
                n_tasks = MIN(n_దారాలు, n_heads);
            } ఆపు;
        స్థితి GGML_OP_WIN_PART:
        స్థితి GGML_OP_WIN_UNPART:
        స్థితి GGML_OP_GET_REL_POS:
            {
                n_tasks = 1;
            } ఆపు;
        స్థితి GGML_OP_MAP_CUSTOM1:
            {
                నిర్మాణం ggml_map_custom1_op_params p;
                జ్ఞాపకం_నకలు(&p, కణుపు->op_params, పరిమాణం(p));
                ఒకవేళ (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_దారాలు;
                } లేకపోతే {
                    n_tasks = MIN(p.n_tasks, n_దారాలు);
                }
            } ఆపు;
        స్థితి GGML_OP_MAP_CUSTOM2:
            {
                నిర్మాణం ggml_map_custom2_op_params p;
                జ్ఞాపకం_నకలు(&p, కణుపు->op_params, పరిమాణం(p));
                ఒకవేళ (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_దారాలు;
                } లేకపోతే {
                    n_tasks = MIN(p.n_tasks, n_దారాలు);
                }
            } ఆపు;
        స్థితి GGML_OP_MAP_CUSTOM3:
            {
                నిర్మాణం ggml_map_custom3_op_params p;
                జ్ఞాపకం_నకలు(&p, కణుపు->op_params, పరిమాణం(p));
                ఒకవేళ (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_దారాలు;
                } లేకపోతే {
                    n_tasks = MIN(p.n_tasks, n_దారాలు);
                }
            } ఆపు;
        స్థితి GGML_OP_CUSTOM:
            {
                నిర్మాణం ggml_custom_op_params p;
                జ్ఞాపకం_నకలు(&p, కణుపు->op_params, పరిమాణం(p));
                ఒకవేళ (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_దారాలు;
                } లేకపోతే {
                    n_tasks = MIN(p.n_tasks, n_దారాలు);
                }
            } ఆపు;
        స్థితి GGML_OP_CROSS_ENTROPY_LOSS:
        స్థితి GGML_OP_CROSS_ENTROPY_LOSS_వెనుక:
        స్థితి GGML_OP_OPT_అడుగు_ADAMW:
        స్థితి GGML_OP_OPT_అడుగు_SGD:
            {
                n_tasks = n_దారాలు;
            } ఆపు;
        స్థితి GGML_OP_NONE:
            {
                n_tasks = 1;
            } ఆపు;
        స్థితి GGML_OP_లెక్క:
            {
                GGML_ABORT("fatal error");
            }
        అప్రమేయ:
            {
                fprintf(stderr, "%s: op not implemented: ", __func__);
                ఒకవేళ (కణుపు->op < GGML_OP_లెక్క) {
                    fprintf(stderr, "%s\n", ggml_op_name(కణుపు->op));
                } లేకపోతే {
                    fprintf(stderr, "%d\n", కణుపు->op);
                }
                GGML_ABORT("fatal error");
            }
    }

    ప్రతిజ్ఞా(n_tasks > 0);

    తిరిగి n_tasks;
}

స్థిర thread_ret_t ggml_లేఖాచిత్రం_compute_secondary_thread(శూన్య* దత్తాంశం);

#ఒకవేళ defined(_WIN32)
#అన్తర్భావ "windows.h"

// TODO: support > 64 CPUs
స్థిర bool ggml_thread_apply_affinity(bool * ముసుగు) {
    HANDLE    h = GetCurrentThread();
    uint64_t  bitmask = 0ULL;

    ప్రతిజ్ఞా(GGML_MAX_N_దారాలు >= 64);

    కొరకు (int32_t i = 0; i < 8; i++) {
        int32_t idx = i * 8;
        uint8_t val = 0;
        val |= ముసుగు[idx + 0] << 0;
        val |= ముసుగు[idx + 1] << 1;
        val |= ముసుగు[idx + 2] << 2;
        val |= ముసుగు[idx + 3] << 3;
        val |= ముసుగు[idx + 4] << 4;
        val |= ముసుగు[idx + 5] << 5;
        val |= ముసుగు[idx + 6] << 6;
        val |= ముసుగు[idx + 7] << 7;
        bitmask |= (uint64_t)val << idx;
    }

    కొరకు (int32_t i = 64; i < GGML_MAX_N_దారాలు; i++) {
        ఒకవేళ (ముసుగు[i]) {
            fprintf(stderr, "warn: setting thread-affinity for > 64 CPUs isn't supported on windows!\n");
            ఆపు;
        }
    }

    DWORD_PTR m = (DWORD_PTR)bitmask;

    m = సమితి_Thread_Affinity_ముసుగు(h, m);

    తిరిగి m != 0;
}

స్థిర bool ggml_thread_apply_ప్రాధాన్యత(int32_t prio) {
    // Note that on Windows the Process Priority Class must be updated in order to set Thread priority.
    // This is up to the applications.
    DWORD p = THREAD_ప్రాధాన్యత_సాధారణ;
    ఎంపిక (prio) {
        స్థితి GGML_SCHED_PRIO_దిగువ:      p = THREAD_ప్రాధాన్యత_క్రింద_సాధారణ;  ఆపు;
        స్థితి GGML_SCHED_PRIO_సాధారణ:   p = THREAD_ప్రాధాన్యత_సాధారణ;        ఆపు;
        స్థితి GGML_SCHED_PRIO_MEDIUM:   p = THREAD_ప్రాధాన్యత_ABOVE_సాధారణ;  ఆపు;
        స్థితి GGML_SCHED_PRIO_ఎగువ:     p = THREAD_ప్రాధాన్యత_HIGHEST;       ఆపు;
        స్థితి GGML_SCHED_PRIO_REALTIME: p = THREAD_ప్రాధాన్యత_సమయం_CRITICAL; ఆపు;
    }

    ఒకవేళ (prio != GGML_SCHED_PRIO_దిగువ) {
        // Tell Windows that this thread should not be throttled (needs its own CPU core).
        // Newer Windows 11 versions aggressively park (offline) CPU cores and often place
        // all our threads onto the first 4 cores which results in terrible performance with
        // n_threads > 4
        #ఒకవేళ _WIN32_WINNT >= 0x0602
        THREAD_POWER_THROTTLING_STATE t;
        Zero_జ్ఞాపకం(&t, పరిమాణం(t));
        t.సంస్కరణ     = THREAD_POWER_THROTTLING_CURRENT_సంస్కరణ;
        t.Control_ముసుగు = THREAD_POWER_THROTTLING_EXECUTION_వేగం;
        t.State_ముసుగు   = 0;

        ఒకవేళ (!సమితి_Thread_Information(GetCurrentThread(), ThreadPowerThrottling, &t, పరిమాణం(t))) {
            GGML_LOG_DEBUG("failed to disable thread power throttling %d : (%d)\n", prio, (పూర్ణాంకం) Get_చివరి_Error());
            తిరిగి false;
        }
        #అయితే_ముగింపు
    }

    ఒకవేళ (prio == GGML_SCHED_PRIO_సాధారణ) {
        // Keep inherited policy/priority
        తిరిగి true;
    }

    ఒకవేళ (!సమితి_Thread_ప్రాధాన్యత(GetCurrentThread(), p)) {
        fprintf(stderr, "warn: failed to set thread priority %d : (%d)\n", prio, (పూర్ణాంకం) Get_చివరి_Error());
        తిరిగి false;
    }

    తిరిగి true;
}

#లేదావేళ defined(__APPLE__)
#అన్తర్భావ <sys/types.h>
#అన్తర్భావ <sys/resource.h>

స్థిర bool ggml_thread_apply_affinity(స్థిరరాశి bool * ముసుగు) {
    // Not supported on Apple platforms
    వాడని(ముసుగు);
    తిరిగి true;
}

స్థిర bool ggml_thread_apply_ప్రాధాన్యత(int32_t prio) {
    నిర్మాణం sched_param p;
    int32_t policy = SCHED_OTHER;
    ఎంపిక (prio) {
        // TODO: there seems to be no way to set lower prio on Apple platforms
        స్థితి GGML_SCHED_PRIO_దిగువ:      policy = SCHED_OTHER; p.sched_ప్రాధాన్యత = 0;  ఆపు;
        స్థితి GGML_SCHED_PRIO_సాధారణ:   policy = SCHED_OTHER; p.sched_ప్రాధాన్యత = 0;  ఆపు;
        స్థితి GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_ప్రాధాన్యత = 40; ఆపు;
        స్థితి GGML_SCHED_PRIO_ఎగువ:     policy = SCHED_FIFO;  p.sched_ప్రాధాన్యత = 80; ఆపు;
        స్థితి GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_ప్రాధాన్యత = 90; ఆపు;
    }

    ఒకవేళ (prio == GGML_SCHED_PRIO_సాధారణ) {
        // Keep inherited policy/priority
        తిరిగి true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    ఒకవేళ (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        తిరిగి false;
    }

    తిరిగి true;
}

#లేదావేళ defined(__linux__)
// TODO: this may not work on BSD, to be verified

స్థిర bool ggml_thread_apply_affinity(స్థిరరాశి bool * ముసుగు) {
    cpu_సమితి_t cpuset;
    పూర్ణాంకం err;

    CPU_ZERO(&cpuset);

    కొరకు (uint32_t i = 0; i < GGML_MAX_N_దారాలు; i++) {
        ఒకవేళ (ముసుగు[i]) {
            GGML_PRINT_DEBUG("Thread %lx: adding %d to cpuset\n", pthread_self(), i);
            CPU_సమితి(i, &cpuset);
        }
    }

#నిర్వచితమైతే __ANDROID__
    err = sched_setaffinity(0, పరిమాణం(cpuset), &cpuset);
    ఒకవేళ (err < 0) {
        err = errno;
    }
#లేకపోతే
    err = pthread_setaffinity_np(pthread_self(), పరిమాణం(cpuset), &cpuset);
#అయితే_ముగింపు
    ఒకవేళ (err != 0) {
        fprintf(stderr, "warn: failed to set affinity mask 0x%llx : %s (%d)\n", (గుర్తులేని దీర్ఘ దీర్ఘ)ముసుగు, strerror(err), err);
        తిరిగి false;
    }

    తిరిగి true;
}

స్థిర bool ggml_thread_apply_ప్రాధాన్యత(int32_t prio) {
    నిర్మాణం sched_param p;
    int32_t policy = SCHED_OTHER;
    ఎంపిక (prio) {
        స్థితి GGML_SCHED_PRIO_దిగువ:      policy = SCHED_BATCH; p.sched_ప్రాధాన్యత = 0;  ఆపు;
        స్థితి GGML_SCHED_PRIO_సాధారణ:   policy = SCHED_OTHER; p.sched_ప్రాధాన్యత = 0;  ఆపు;
        స్థితి GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_ప్రాధాన్యత = 40; ఆపు;
        స్థితి GGML_SCHED_PRIO_ఎగువ:     policy = SCHED_FIFO;  p.sched_ప్రాధాన్యత = 80; ఆపు;
        స్థితి GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_ప్రాధాన్యత = 90; ఆపు;
    }

    ఒకవేళ (prio == GGML_SCHED_PRIO_సాధారణ) {
        // Keep inherited policy/priority
        తిరిగి true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    ఒకవేళ (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        తిరిగి false;
    }

    తిరిగి true;
}

#లేకపోతే // unsupported platforms

స్థిర bool ggml_thread_apply_affinity(స్థిరరాశి bool * ముసుగు) {
    వాడని(ముసుగు);
    తిరిగి true;
}

స్థిర bool ggml_thread_apply_ప్రాధాన్యత(int32_t prio) {
    వాడని(prio);
    తిరిగి true;
}

#అయితే_ముగింపు

స్థిర bool ggml_thread_cpumask_is_valid(స్థిరరాశి bool * ముసుగు) {
    కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_N_దారాలు; i++) {
        ఒకవేళ (ముసుగు[i]) { తిరిగి true; }
    }
    తిరిగి false;
}

స్థిర శూన్య ggml_thread_cpumask_తదుపరి(స్థిరరాశి bool * global_ముసుగు, bool * local_ముసుగు, bool strict, int32_t* iter) {
    ఒకవేళ (!strict) {
        జ్ఞాపకం_నకలు(local_ముసుగు, global_ముసుగు, GGML_MAX_N_దారాలు);
        తిరిగి;
    } లేకపోతే {
        memset(local_ముసుగు, 0, GGML_MAX_N_దారాలు);
        int32_t పునాది_idx = *iter;
        కొరకు (int32_t i = 0; i < GGML_MAX_N_దారాలు; i++) {
            int32_t idx = పునాది_idx + i;
            ఒకవేళ (idx >= GGML_MAX_N_దారాలు) {
                // Just a cheaper modulo
                idx -= GGML_MAX_N_దారాలు;
            }
            ఒకవేళ (global_ముసుగు[idx]) {
                local_ముసుగు[idx] = 1;
                *iter = idx + 1;
                తిరిగి;
            }
        }
    }
}

శూన్య ggml_threadpool_free(నిర్మాణం ggml_threadpool* threadpool) {
    ఒకవేళ (!threadpool) తిరిగి;

    స్థిరరాశి పూర్ణాంకం n_దారాలు = threadpool->n_దారాలు;

#నిర్వచితం_కాకపోతే GGML_USE_OPENMP
    నిర్మాణం ggml_compute_state* workers = threadpool->workers;

    ggml_mutex_తాళం(&threadpool->mutex);

    threadpool->stop = true;
    threadpool->pause = false;

    ggml_cond_ప్రసారం(&threadpool->cond);
    ggml_mutex_unlock(&threadpool->mutex);

    కొరకు (పూర్ణాంకం j = 1; j < n_దారాలు; j++) {
        int32_t rc = ggml_thread_join(workers[j].thrd, ఖాళీ);
        GGML_ASSERT(rc == GGML_EXIT_SUCCESS || rc == GGML_EXIT_ABORTED);
        వాడని(rc);
    }

    ggml_mutex_destroy(&threadpool->mutex);
    ggml_cond_destroy(&threadpool->cond);
#అయితే_ముగింపు // GGML_USE_OPENMP

    స్థిరరాశి size_t workers_కొలత = పరిమాణం(నిర్మాణం ggml_compute_state) * n_దారాలు;
    ggml_aligned_free(threadpool->workers, workers_కొలత);
    ggml_aligned_free(threadpool, పరిమాణం(నిర్మాణం ggml_threadpool));
}

#నిర్వచితం_కాకపోతే GGML_USE_OPENMP
// pause/resume must be called under mutex
స్థిర శూన్య ggml_threadpool_pause_locked(నిర్మాణం ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Pausing threadpool\n");
    threadpool->pause = true;
    ggml_cond_ప్రసారం(&threadpool->cond);
}

స్థిర శూన్య ggml_threadpool_resume_locked(నిర్మాణం ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Resuming threadpool\n");
    threadpool->pause = false;
    ggml_cond_ప్రసారం(&threadpool->cond);
}
#అయితే_ముగింపు

శూన్య ggml_threadpool_pause(నిర్మాణం ggml_threadpool * threadpool) {
#నిర్వచితం_కాకపోతే GGML_USE_OPENMP
    ggml_mutex_తాళం(&threadpool->mutex);
    ఒకవేళ (!threadpool->pause) {
       ggml_threadpool_pause_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#లేకపోతే
    వాడని(threadpool);
#అయితే_ముగింపు
}

శూన్య ggml_threadpool_resume(నిర్మాణం ggml_threadpool * threadpool) {
#నిర్వచితం_కాకపోతే GGML_USE_OPENMP
    ggml_mutex_తాళం(&threadpool->mutex);
    ఒకవేళ (threadpool->pause) {
       ggml_threadpool_resume_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#లేకపోతే
    వాడని(threadpool);
#అయితే_ముగింపు
}

నిర్మాణం ggml_cplan ggml_లేఖాచిత్రం_plan(
          స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph,
                               పూర్ణాంకం   n_దారాలు,
            నిర్మాణం ggml_threadpool * threadpool) {

    ఒకవేళ (threadpool == ఖాళీ) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
    }
    ఒకవేళ (n_దారాలు <= 0) {
        n_దారాలు = threadpool ? threadpool->n_దారాలు : GGML_DEFAULT_N_దారాలు;
    }

#ఒకవేళ defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
    // Emscripten without pthreads support can only use a single thread
    n_దారాలు = 1;
#అయితే_ముగింపు

#ఒకవేళ defined(__wasi__)
    // WASI doesn't support parallelism yet
    n_దారాలు = 1;
#అయితే_ముగింపు

    size_t work_కొలత = 0;

    నిర్మాణం ggml_cplan cplan;
    memset(&cplan, 0, పరిమాణం(నిర్మాణం ggml_cplan));

    పూర్ణాంకం max_tasks = 1;

    // thread scheduling for the different operations + work buffer size estimation
    కొరకు (పూర్ణాంకం i = 0; i < cgraph->n_nodes; i++) {
        నిర్మాణం ggml_tensor * కణుపు = cgraph->nodes[i];

        స్థిరరాశి పూర్ణాంకం n_tasks = ggml_get_n_tasks(కణుపు, n_దారాలు);

        max_tasks = MAX(max_tasks, n_tasks);

        size_t cur = 0;

        ఒకవేళ (!ggml_cpu_extra_work_కొలత(n_దారాలు, కణుపు, &cur)) {
            ఎంపిక (కణుపు->op) {
                స్థితి GGML_OP_CPY:
                స్థితి GGML_OP_DUP:
                    {
                        ఒకవేళ (ggml_is_quantized(కణుపు->type) ||
                            // F16 -> BF16 and BF16 -> F16 copies go through intermediate F32
                            (కణుపు->src[0]->type == GGML_TYPE_F16  && కణుపు->src[1] && కణుపు->src[1]->type == GGML_TYPE_BF16) ||
                            (కణుపు->src[0]->type == GGML_TYPE_BF16 && కణుపు->src[1] && కణుపు->src[1]->type == GGML_TYPE_F16) ||
                            // conversion between F32 and I32
                            (కణుపు->src[0]->type == GGML_TYPE_F32 && కణుపు->src[1] && కణుపు->src[1]->type == GGML_TYPE_I32) ||
                            (కణుపు->src[0]->type == GGML_TYPE_I32 && కణుపు->src[1] && కణుపు->src[1]->type == GGML_TYPE_F32)) {
                            cur = ggml_type_కొలత(GGML_TYPE_F32) * కణుపు->ne[0] * n_tasks;
                        }
                    } ఆపు;
                స్థితి GGML_OP_కూడు:
                స్థితి GGML_OP_కూడు_ID:
                స్థితి GGML_OP_ADD1:
                    {
                        ఒకవేళ (ggml_is_quantized(కణుపు->src[0]->type)) {
                            cur = ggml_type_కొలత(GGML_TYPE_F32) * కణుపు->src[0]->ne[0] * n_tasks;
                        }
                    } ఆపు;
                స్థితి GGML_OP_ACC:
                    {
                        ఒకవేళ (ggml_is_quantized(కణుపు->src[0]->type)) {
                            cur = ggml_type_కొలత(GGML_TYPE_F32) * కణుపు->src[1]->ne[0] * n_tasks;
                        }
                    } ఆపు;
                స్థితి GGML_OP_లెక్క_EQUAL:
                    {
                        cur = ggml_type_కొలత(కణుపు->type)*n_tasks;
                    } ఆపు;
                స్థితి GGML_OP_MUL_MAT:
                    {
                        స్థిరరాశి గణన ggml_type vec_dot_type = type_traits_cpu[కణుపు->src[0]->type].vec_dot_type;
                        ఒకవేళ (కణుపు->src[1]->type != vec_dot_type) {
                            cur = ggml_అడ్డువరుస_కొలత(vec_dot_type, ggml_nelements(కణుపు->src[1]));
                        }
                        // Workspace for tiled (see tiled.h)
                        cur = GGML_PAD(cur, 64);
                        cur += ggml_tiled_wdata_కొలత(n_tasks, కణుపు);
                    } ఆపు;
                స్థితి GGML_OP_MUL_MAT_ID:
                    {
                        cur = 0;
                        స్థిరరాశి నిర్మాణం ggml_tensor * src0 = కణుపు->src[0];
                        స్థిరరాశి నిర్మాణం ggml_tensor * src1 = కణుపు->src[1];
                        స్థిరరాశి నిర్మాణం ggml_tensor * ids = కణుపు->src[2];
                        స్థిరరాశి గణన ggml_type vec_dot_type = type_traits_cpu[src0->type].vec_dot_type;
                        స్థిరరాశి పూర్ణాంకం n_as = src0->ne[2];
                        // src1
                        ఒకవేళ (src1->type != vec_dot_type) {
                            cur += ggml_అడ్డువరుస_కొలత(vec_dot_type, ggml_nelements(src1)) + పరిమాణం(int64_t);
                        }
                        // matrix_row_counts
                        cur += n_as * పరిమాణం(int64_t) + పరిమాణం(int64_t);
                        // matrix_rows
                        cur += n_as*ids->ne[0]*ids->ne[1]*పరిమాణం(నిర్మాణం mmid_అడ్డువరుస_mapping) + పరిమాణం(int64_t);
                        // atomic_current_chunk
                        cur += CACHE_పంక్తి_కొలత*n_as + CACHE_పంక్తి_కొలత;
                        // Workspace for tiled (see tiled.h)
                        cur = GGML_PAD(cur, 64);
                        cur += ggml_tiled_wdata_కొలత(n_tasks, కణుపు);
                    } ఆపు;
                స్థితి GGML_OP_బహిర్గతం_PROD:
                    {
                        ఒకవేళ (ggml_is_quantized(కణుపు->src[0]->type) ||
                            కణుపు->src[0]->type == GGML_TYPE_F16) {
                            cur = ggml_type_కొలత(GGML_TYPE_F32) * కణుపు->src[0]->ne[0] * n_tasks;
                        }
                    } ఆపు;
                స్థితి GGML_OP_సమితి_అడ్డువరుసలు:
                    {
                        ఒకవేళ (కణుపు->src[0]->type == GGML_TYPE_F16 && కణుపు->type != GGML_TYPE_F16) {
                            cur = ggml_type_కొలత(GGML_TYPE_F32) * కణుపు->src[0]->ne[0] * n_tasks;
                        }
                    } ఆపు;
                స్థితి GGML_OP_SOFT_MAX:
                స్థితి GGML_OP_ROPE:
                స్థితి GGML_OP_ROPE_వెనుక:
                    {
                        cur = ggml_type_కొలత(GGML_TYPE_F32) * కణుపు->ne[0] * n_tasks;
                    } ఆపు;
                స్థితి GGML_OP_CONV_TRANSPOSE_1D:
                    {
                        GGML_ASSERT(కణుపు->src[0]->ne[3] == 1);
                        GGML_ASSERT(కణుపు->src[1]->ne[2] == 1);
                        GGML_ASSERT(కణుపు->src[1]->ne[3] == 1);

                        స్థిరరాశి int64_t ne00 = కణుపు->src[0]->ne[0];  // K
                        స్థిరరాశి int64_t ne01 = కణుపు->src[0]->ne[1];  // Cout
                        స్థిరరాశి int64_t ne02 = కణుపు->src[0]->ne[2];  // Cin
                        స్థిరరాశి int64_t ne10 = కణుపు->src[1]->ne[0];  // L
                        స్థిరరాశి int64_t ne11 = కణుపు->src[1]->ne[1];  // Cin

                        ఒకవేళ ((కణుపు->src[0]->type == GGML_TYPE_F16 ||
                             కణుపు->src[0]->type == GGML_TYPE_BF16) &&
                            కణుపు->src[1]->type == GGML_TYPE_F32) {
                            cur += పరిమాణం(ggml_fp16_t)*ne00*ne01*ne02;
                            cur += పరిమాణం(ggml_fp16_t)*ne10*ne11;
                        } లేకపోతే ఒకవేళ (కణుపు->src[0]->type == GGML_TYPE_F32 &&
                                   కణుపు->src[1]->type == GGML_TYPE_F32) {
                            cur += పరిమాణం(దశాంశం)*ne00*ne01*ne02;
                            cur += పరిమాణం(దశాంశం)*ne10*ne11;
                        } లేకపోతే {
                            GGML_ABORT("fatal error");
                        }
                    } ఆపు;
                స్థితి GGML_OP_CONV_2D:
                స్థితి GGML_OP_CONV_3D:
                    {
                        cur = GGML_IM2_నిలువు_WORK_కొలత;
                    } ఆపు;
                స్థితి GGML_OP_CONV_TRANSPOSE_2D:
                    {
                        స్థిరరాశి int64_t ne00 = కణుపు->src[0]->ne[0]; // W
                        స్థిరరాశి int64_t ne01 = కణుపు->src[0]->ne[1]; // H
                        స్థిరరాశి int64_t ne02 = కణుపు->src[0]->ne[2]; // Channels Out
                        స్థిరరాశి int64_t ne03 = కణుపు->src[0]->ne[3]; // Channels In

                        స్థిరరాశి int64_t ne10 = కణుపు->src[1]->ne[0]; // W
                        స్థిరరాశి int64_t ne11 = కణుపు->src[1]->ne[1]; // H
                        స్థిరరాశి int64_t ne12 = కణుపు->src[1]->ne[2]; // Channels In
                        స్థిరరాశి int64_t ne13 = కణుపు->src[1]->ne[3]; // Batch

                        GGML_ASSERT(కణుపు->src[0]->type == GGML_TYPE_F16 || కణుపు->src[0]->type == GGML_TYPE_F32);
                        GGML_ASSERT(కణుపు->src[1]->type == GGML_TYPE_F32);

                        cur += ggml_type_కొలత(కణుపు->src[0]->type) * ne00 * ne01 * ne02 * ne03;
                        cur += ggml_type_కొలత(కణుపు->src[0]->type) * ne10 * ne11 * ne12 * ne13;

                    } ఆపు;
                స్థితి GGML_OP_శిఖరం_K:
                    {
                        cur += పరిమాణం(int32_t)*కణుపు->src[0]->ne[0]*n_tasks;
                    } ఆపు;
                స్థితి GGML_OP_FLASH_ATTN_EXT:
                    {
                        స్థిరరాశి int64_t neq2 = కణుపు->src[0]->ne[2]; // number of query heads
                        స్థిరరాశి int64_t DK = కణుపు->src[1]->ne[0];
                        స్థిరరాశి int64_t DV = కణుపు->src[2]->ne[0];

                        // Tiled flash attention scratch (tile sizes defined in common.h)
                        // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
                        size_t prefill  = పరిమాణం(దశాంశం)*(GGML_FA_TILE_Q*DK + 2*GGML_FA_TILE_Q*GGML_FA_TILE_KV + GGML_FA_TILE_Q*DV + GGML_FA_TILE_KV*DV + GGML_FA_TILE_KV*DK)*n_tasks;

                        // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
                        // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
                        size_t n_chunks = n_tasks;
                        size_t decode   = పరిమాణం(దశాంశం)*(neq2*n_chunks*(2+DV) + n_tasks*(DK + 2*DV));

                        cur += MAX(prefill, decode);
                    } ఆపు;
                స్థితి GGML_OP_FLASH_ATTN_వెనుక:
                    {
                        స్థిరరాశి int64_t    D = కణుపు->src[0]->ne[0];
                        స్థిరరాశి int64_t ne11 = ggml_పైకి(కణుపు->src[1]->ne[1], GGML_SOFT_MAX_UNROLL);
                        స్థిరరాశి int64_t mxDn = MAX(D, ne11) * 2; // *2 because of S and SM in ggml_compute_forward_flash_attn_back
                        ఒకవేళ (కణుపు->src[1]->type == GGML_TYPE_F32) {
                            cur  = పరిమాణం(దశాంశం)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += పరిమాణం(దశాంశం)*mxDn*n_tasks; // this is overestimated by x2
                        } లేకపోతే ఒకవేళ (కణుపు->src[1]->type == GGML_TYPE_F16) {
                            cur  = పరిమాణం(దశాంశం)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += పరిమాణం(దశాంశం)*mxDn*n_tasks; // this is overestimated by x2
                        } లేకపోతే ఒకవేళ (కణుపు->src[1]->type == GGML_TYPE_BF16) {
                            cur  = పరిమాణం(దశాంశం)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += పరిమాణం(దశాంశం)*mxDn*n_tasks; // this is overestimated by x2
                        }
                    } ఆపు;

                స్థితి GGML_OP_CROSS_ENTROPY_LOSS:
                    {
                        cur = ggml_type_కొలత(కణుపు->type)*(n_tasks + కణుపు->src[0]->ne[0]*n_tasks);
                    } ఆపు;
                స్థితి GGML_OP_GATED_DELTA_NET:
                    {
                        స్థిరరాశి int64_t S_v = కణుపు->src[2]->ne[0];
                        స్థిరరాశి int64_t K   = ggml_get_op_params_i32(కణుపు, 0);
                        స్థిరరాశి int64_t చొప్పున_thread = S_v + (K > 1 ? S_v * S_v : 0);
                        cur = చొప్పున_thread * పరిమాణం(దశాంశం) * n_tasks;
                    } ఆపు;
                స్థితి GGML_OP_లెక్క:
                    {
                        GGML_ABORT("fatal error");
                    }
                స్థితి GGML_OP_LIGHTNING_INDEXER:
                    {
                        // temp buffer for dequantizing lightning indexer keys
                        స్థిరరాశి int64_t ne10 = కణుపు->src[1]->ne[0];
                        cur += పరిమాణం(దశాంశం)*ne10*n_tasks;
                    } ఆపు;
                అప్రమేయ:
                    ఆపు;
            }
        }

        work_కొలత = MAX(work_కొలత, cur);
    }

    ఒకవేళ (work_కొలత > 0) {
        work_కొలత += CACHE_పంక్తి_కొలత*(n_దారాలు);
    }

    cplan.threadpool = threadpool;
    cplan.n_దారాలు  = MIN(max_tasks, n_దారాలు);
    cplan.work_కొలత  = work_కొలత;
    cplan.work_దత్తాంశం  = ఖాళీ;

    తిరిగి cplan;
}


// Try to fuse the current node with subsequent nodes for better performance.
// Returns the number of nodes skipped by fusion (>=1), or 0 if no fusion was applied.
స్థిర bool ggml_cpu_disable_fusion = false;  // initialized once in ggml_cpu_init(), read-only afterwards

స్థిర పూర్ణాంకం ggml_cpu_try_fuse_ops(
        స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph,
        స్థిరరాశి పూర్ణాంకం కణుపు_n,
        స్థిరరాశి నిర్మాణం ggml_compute_params * params,
        స్థిరరాశి నిర్మాణం ggml_cplan * cplan) {

    ఒకవేళ (ggml_cpu_disable_fusion || cplan->use_ref) {
        తిరిగి 0;
    }

    నిర్మాణం ggml_tensor * కణుపు = cgraph->nodes[కణుపు_n];

    ఒకవేళ (కణుపు->op == GGML_OP_RMS_NORM) {
        // RMS_NORM + MUL fusion
        స్థిరరాశి గణన ggml_op fuse_ops[] = { GGML_OP_RMS_NORM, GGML_OP_MUL };
        ఒకవేళ (ggml_can_fuse(cgraph, కణుపు_n, fuse_ops, 2)) {
            నిర్మాణం ggml_tensor * mul_కణుపు = cgraph->nodes[కణుపు_n + 1];
            స్థిరరాశి నిర్మాణం ggml_tensor * mul_w = (mul_కణుపు->src[0] == కణుపు)
                ? mul_కణుపు->src[1] : mul_కణుపు->src[0];
            ఒకవేళ (కణుపు->src[0]->type  == GGML_TYPE_F32 &&
                mul_కణుపు->type      == GGML_TYPE_F32 &&
                mul_w->type         == GGML_TYPE_F32 &&
                mul_w->ne[0]        == కణుపు->ne[0]   &&
                mul_w->nb[0]        == పరిమాణం(దశాంశం)) {

                ggml_compute_forward_rms_norm_mul_fused(params, కణుపు, mul_కణుపు);
                తిరిగి 1;
            }
        }
    }

    తిరిగి 0;
}

స్థిర thread_ret_t ggml_లేఖాచిత్రం_compute_thread(శూన్య * దత్తాంశం) {
    నిర్మాణం ggml_compute_state * state = (నిర్మాణం ggml_compute_state *) దత్తాంశం;
    నిర్మాణం ggml_threadpool    * tp    = state->threadpool;

    స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph = tp->cgraph;
    స్థిరరాశి నిర్మాణం ggml_cplan  * cplan  = tp->cplan;

#నిర్వచితమైతే GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_సమితి_numa_thread_affinity(state->ith);
#లేకపోతే
    సమితి_numa_thread_affinity(state->ith);
#అయితే_ముగింపు

    నిర్మాణం ggml_compute_params params = {
        /*.ith        =*/ state->ith,
        /*.nth        =*/ atomic_ఎక్కించు_explicit(&tp->n_లేఖాచిత్రం, memory_order_relaxed) & GGML_THREADPOOL_N_దారాలు_ముసుగు,
        /*.wsize      =*/ cplan->work_కొలత,
        /*.wdata      =*/ cplan->work_దత్తాంశం,
        /*.threadpool =*/ tp,
        /*.use_ref    =*/ cplan->use_ref,
    };

#నిర్వచితమైతే GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p\n", state->ith, (స్థిరరాశి శూన్య *)cplan);
#లేకపోతే
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p last-graph %d\n", state->ith, (స్థిరరాశి శూన్య *)cplan, state->చివరి_లేఖాచిత్రం);
#అయితే_ముగింపు

    కొరకు (పూర్ణాంకం కణుపు_n = 0; కణుపు_n < cgraph->n_nodes && atomic_ఎక్కించు_explicit(&tp->abort, memory_order_relaxed) != కణుపు_n; కణుపు_n++) {
        నిర్మాణం ggml_tensor * కణుపు = cgraph->nodes[కణుపు_n];

        ఒకవేళ (ggml_op_is_ఖాళీలు(కణుపు->op)) {
            // skip NOPs
            కొనసాగు;
        }

        ఒకవేళ ((కణుపు->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            కొనసాగు;
        }

        // TODO: move fused-op detection into ggml_graph_plan so fusion decisions are made once at planning time
        // Try fused ops, fall back to normal compute
        స్థిరరాశి పూర్ణాంకం n_fused = ggml_cpu_try_fuse_ops(cgraph, కణుపు_n, &params, cplan);
        ఒకవేళ (n_fused > 0) {
            కణుపు_n += n_fused;
        } లేకపోతే {
            ggml_compute_forward(&params, కణుపు);
        }

        ఒకవేళ (state->ith == 0 && cplan->abort_callback &&
                cplan->abort_callback(cplan->abort_callback_దత్తాంశం)) {
            atomic_store_explicit(&tp->abort, కణుపు_n + 1, memory_order_relaxed);
            tp->ec    = GGML_స్థితి_ABORTED;
        }

        ఒకవేళ (కణుపు_n + 1 < cgraph->n_nodes) {
            ggml_barrier(state->threadpool);
        }
    }

#నిర్వచితమైతే GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p\n", state->ith, (స్థిరరాశి శూన్య *)cplan);
#లేకపోతే
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p last-graph %d\n", state->ith, (స్థిరరాశి శూన్య *)cplan, state->చివరి_లేఖాచిత్రం);
#అయితే_ముగింపు

    ggml_barrier(state->threadpool);

#నిర్వచితమైతే GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(state->ith);
#అయితే_ముగింపు

    తిరిగి 0;
}

#నిర్వచితం_కాకపోతే GGML_USE_OPENMP

// check if thread is ready to proceed (exit from polling or sleeping)
// returns true if loops should exit, sets state->pending to indicate new work
స్థిర అంతఃవరుస bool ggml_లేఖాచిత్రం_compute_thread_ready(నిర్మాణం ggml_compute_state * state) {
    నిర్మాణం ggml_threadpool * threadpool = state->threadpool;

    ఒకవేళ (state->pending || threadpool->stop || threadpool->pause) { తిరిగి true; }

    // check for new graph/work
    పూర్ణాంకం n_లేఖాచిత్రం   = atomic_ఎక్కించు_explicit(&threadpool->n_లేఖాచిత్రం, memory_order_relaxed);
    పూర్ణాంకం n_దారాలు = n_లేఖాచిత్రం & GGML_THREADPOOL_N_దారాలు_ముసుగు;
    ఒకవేళ (n_లేఖాచిత్రం != state->చివరి_లేఖాచిత్రం) {
        state->pending    = (state->ith < n_దారాలు);
        state->చివరి_లేఖాచిత్రం = n_లేఖాచిత్రం;
        తిరిగి true;
    }

    తిరిగి false;
}

// sync thread state after polling
స్థిర అంతఃవరుస శూన్య ggml_లేఖాచిత్రం_compute_thread_sync(నిర్మాణం ggml_compute_state * state) {
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #నిర్వచితమైతే GGML_TSAN_ENABLED
    atomic_fetch_కూడు_explicit(&state->threadpool->n_లేఖాచిత్రం, 0, memory_order_seq_cst);
    #లేకపోతే
    atomic_thread_fence(memory_order_seq_cst);
    #అయితే_ముగింపు
    వాడని(state);
}

స్థిర అంతఃవరుస bool ggml_లేఖాచిత్రం_compute_poll_for_work(నిర్మాణం ggml_compute_state * state) {
    నిర్మాణం ggml_threadpool * threadpool = state->threadpool;

    // This seems to make 0 ... 100 a decent range for polling level across modern processors.
    // Perhaps, we can adjust it dynamically based on load and things.
    స్థిరరాశి uint64_t n_విడతలు = 1024UL * 128 * threadpool->poll;

    కొరకు (uint64_t i=0; !ggml_లేఖాచిత్రం_compute_thread_ready(state) && i < n_విడతలు; i++) {
        // No new work. Keep polling.
        ggml_thread_cpu_relax();
    }

    తిరిగి state->pending;
}

స్థిర అంతఃవరుస bool ggml_లేఖాచిత్రం_compute_check_for_work(నిర్మాణం ggml_compute_state * state) {
    నిర్మాణం ggml_threadpool * threadpool = state->threadpool;

    ఒకవేళ (ggml_లేఖాచిత్రం_compute_poll_for_work(state)) {
        ggml_లేఖాచిత్రం_compute_thread_sync(state);
        తిరిగి state->pending;
    }

    ggml_mutex_తాళం_shared(&threadpool->mutex);
    అయితే (!ggml_లేఖాచిత్రం_compute_thread_ready(state)) {
        // No new work. Wait for the signal.
        GGML_PRINT_DEBUG("thread #%d waiting for work (sleeping)\n", state->ith);
        ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
    }
    ggml_mutex_unlock_shared(&threadpool->mutex);

    తిరిగి state->pending;
}

స్థిర thread_ret_t ggml_లేఖాచిత్రం_compute_secondary_thread(శూన్య* దత్తాంశం) {
    నిర్మాణం ggml_compute_state * state = (నిర్మాణం ggml_compute_state *) దత్తాంశం;
    నిర్మాణం ggml_threadpool * threadpool = state->threadpool;

    ggml_thread_apply_ప్రాధాన్యత(threadpool->prio);
    ఒకవేళ (ggml_thread_cpumask_is_valid(state->cpumask)) {
        ggml_thread_apply_affinity(state->cpumask);
    }

    అయితే (true) {
        // Check if we need to sleep
        అయితే (threadpool->pause) {
            GGML_PRINT_DEBUG("thread #%d inside pause loop\n", state->ith);
            ggml_mutex_తాళం_shared(&threadpool->mutex);
            ఒకవేళ (threadpool->pause) {
                ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
            }
            GGML_PRINT_DEBUG("thread #%d resuming after wait\n", state->ith);
            ggml_mutex_unlock_shared(&threadpool->mutex);
        }

        // This needs to be checked for after the cond_wait
        ఒకవేళ (threadpool->stop) ఆపు;

        // Check if there is new work
        // The main thread is the only one that can dispatch new work

        ggml_లేఖాచిత్రం_compute_check_for_work(state);
        ఒకవేళ (state->pending) {
            state->pending = false;
            ggml_లేఖాచిత్రం_compute_thread(state);
        }
    }

    తిరిగి (thread_ret_t) 0;
}

// Start processing new graph
స్థిర శూన్య ggml_లేఖాచిత్రం_compute_kickoff(నిర్మాణం ggml_threadpool * threadpool, పూర్ణాంకం n_దారాలు)
{
    // Always take the mutex here because the worker threads are doing hybrid poll/wait

    ggml_mutex_తాళం(&threadpool->mutex);

    // Update the number of active threads and the graph count
    పూర్ణాంకం n_లేఖాచిత్రం = atomic_ఎక్కించు_explicit(&threadpool->n_లేఖాచిత్రం, memory_order_relaxed) >> GGML_THREADPOOL_N_దారాలు_బిట్లు;
    n_లేఖాచిత్రం = ((n_లేఖాచిత్రం + 1) << GGML_THREADPOOL_N_దారాలు_బిట్లు) | (n_దారాలు & GGML_THREADPOOL_N_దారాలు_ముసుగు);

    GGML_PRINT_DEBUG("compute-kickoff: n_threads %d n_graph %d\n", n_దారాలు, n_లేఖాచిత్రం);

    // Indicate the graph is ready to be processed
    // We need the full seq-cst fence here because of the polling threads (used in thread_sync)
    atomic_store_explicit(&threadpool->n_లేఖాచిత్రం, n_లేఖాచిత్రం, memory_order_seq_cst);

    ఒకవేళ (threadpool->pause) {
       // Update main thread prio and affinity to match the threadpool settings
       ggml_thread_apply_ప్రాధాన్యత(threadpool->prio);
       ఒకవేళ (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
           ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
       }

       // resume does cond broadcast
       ggml_threadpool_resume_locked(threadpool);
    } లేకపోతే {
       ggml_cond_ప్రసారం(&threadpool->cond);
    }

    ggml_mutex_unlock(&threadpool->mutex);
}

#అయితే_ముగింపు // GGML_USE_OPENMP

స్థిర నిర్మాణం ggml_threadpool * ggml_threadpool_new_impl(
    నిర్మాణం ggml_threadpool_params * tpp,
               నిర్మాణం ggml_cgraph * cgraph,
                నిర్మాణం ggml_cplan * cplan) {

    నిర్మాణం ggml_threadpool * threadpool =
        ggml_aligned_malloc(పరిమాణం(నిర్మాణం ggml_threadpool));
    {
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->n_లేఖాచిత్రం          = 0;
        threadpool->n_barrier        = 0;
        threadpool->n_barrier_passed = 0;
        threadpool->current_chunk    = 0;
        threadpool->stop             = false;
        threadpool->pause            = tpp->paused;
        threadpool->abort            = -1;
        threadpool->workers          = ఖాళీ;
        threadpool->n_దారాలు        = tpp->n_దారాలు;
        threadpool->poll             = tpp->poll;
        threadpool->prio             = tpp->prio;
        threadpool->ec               = GGML_స్థితి_SUCCESS;
    }

    // Allocate and init workers state
    స్థిరరాశి size_t workers_కొలత = పరిమాణం(నిర్మాణం ggml_compute_state) * tpp->n_దారాలు;
    నిర్మాణం ggml_compute_state * workers = ggml_aligned_malloc(workers_కొలత);

    memset(workers, 0, workers_కొలత);
    కొరకు (పూర్ణాంకం j = 0; j < tpp->n_దారాలు; j++) {
        workers[j].threadpool = threadpool;
        workers[j].ith        = j;
    }

    threadpool->workers = workers;

#నిర్వచితమైతే GGML_USE_OPENMP
    int32_t cpumask_iter = 0;

    // Compute CPU masks for each thread
    కొరకు (పూర్ణాంకం j = 0; j < tpp->n_దారాలు; j++) {
        ggml_thread_cpumask_తదుపరి(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);
    }
#లేకపోతే // GGML_USE_OPENMP
    ggml_mutex_init(&threadpool->mutex);
    ggml_cond_init(&threadpool->cond);

    // Spin the threads for all workers, and update CPU placements.
    // Place the main thread last (towards the higher numbered CPU cores).

    int32_t cpumask_iter = 0;

    కొరకు (పూర్ణాంకం j = 1; j < tpp->n_దారాలు; j++) {
        ggml_thread_cpumask_తదుపరి(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);

        int32_t rc = ggml_thread_create(&workers[j].thrd, ఖాళీ, ggml_లేఖాచిత్రం_compute_secondary_thread, &workers[j]);
        GGML_ASSERT(rc == 0);
    }

    ggml_thread_cpumask_తదుపరి(tpp->cpumask, workers[0].cpumask, tpp->strict_cpu, &cpumask_iter);

    ఒకవేళ (!threadpool->pause) {
        // Update main thread prio and affinity at the start, otherwise we'll do it in resume
        ggml_thread_apply_ప్రాధాన్యత(threadpool->prio);
        ఒకవేళ (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
            ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
        }
    }
#అయితే_ముగింపు // GGML_USE_OPENMP

    తిరిగి threadpool;
}

నిర్మాణం ggml_threadpool * ggml_threadpool_new(నిర్మాణం ggml_threadpool_params * tpp) {
    తిరిగి ggml_threadpool_new_impl(tpp, ఖాళీ, ఖాళీ);
}

గణన ggml_స్థితి ggml_లేఖాచిత్రం_compute(నిర్మాణం ggml_cgraph * cgraph, నిర్మాణం ggml_cplan * cplan) {
    ggml_cpu_init();

    GGML_ASSERT(cplan);
    GGML_ASSERT(cplan->n_దారాలు > 0);
    GGML_ASSERT(cplan->work_కొలత == 0 || cplan->work_దత్తాంశం != ఖాళీ);

    పూర్ణాంకం n_దారాలు                               = cplan->n_దారాలు;
    నిర్మాణం ggml_threadpool * threadpool = cplan->threadpool;

    bool disposable_threadpool = false;

    ఒకవేళ (threadpool == ఖాళీ) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
        disposable_threadpool = true;

        నిర్మాణం ggml_threadpool_params ttp = ggml_threadpool_params_default(n_దారాలు);
        threadpool = ggml_threadpool_new_impl(&ttp, cgraph, cplan);
    } లేకపోతే {
        // Reset some of the parameters that need resetting
        // No worker threads should be accessing the parameters below at this stage
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->current_chunk    = 0;
        threadpool->abort            = -1;
        threadpool->ec               = GGML_స్థితి_SUCCESS;
    }

#నిర్వచితమైతే GGML_USE_OPENMP
    ఒకవేళ (n_దారాలు > 1) {
        #సూచన omp parallel అంకె_దారాలు(n_దారాలు)
        {
            #సూచన omp single
            {
                // update the number of threads from the actual number of threads that we got from OpenMP
                n_దారాలు = omp_get_అంకె_దారాలు();
                atomic_store_explicit(&threadpool->n_లేఖాచిత్రం, n_దారాలు, memory_order_relaxed);
            }

            // Apply thread CPU mask and priority
            పూర్ణాంకం ith = omp_get_thread_అంకె();

            ggml_thread_apply_ప్రాధాన్యత(threadpool->prio);
            ఒకవేళ (ggml_thread_cpumask_is_valid(threadpool->workers[ith].cpumask)) {
                ggml_thread_apply_affinity(threadpool->workers[ith].cpumask);
            }
            ggml_లేఖాచిత్రం_compute_thread(&threadpool->workers[ith]);
        }
    } లేకపోతే {
        atomic_store_explicit(&threadpool->n_లేఖాచిత్రం, 1, memory_order_relaxed);
        ggml_లేఖాచిత్రం_compute_thread(&threadpool->workers[0]);
    }
#లేకపోతే
    ఒకవేళ (n_దారాలు > threadpool->n_దారాలు) {
        GGML_LOG_WARN("cplan requested more threads (%d) than available (%d)\n", n_దారాలు, threadpool->n_దారాలు);
        n_దారాలు = threadpool->n_దారాలు;
    }

    // Kick all threads to start the new graph
    ggml_లేఖాచిత్రం_compute_kickoff(threadpool, n_దారాలు);

    // This is a work thread too
    ggml_లేఖాచిత్రం_compute_thread(&threadpool->workers[0]);
#అయితే_ముగింపు

    // don't leave affinity set on the main thread
    clear_numa_thread_affinity();

    గణన ggml_స్థితి ret = threadpool->ec;

    ఒకవేళ (disposable_threadpool) {
        ggml_threadpool_free(threadpool);
    }

    తిరిగి ret;
}

గణన ggml_స్థితి ggml_లేఖాచిత్రం_compute_with_ctx(నిర్మాణం ggml_context * ctx, నిర్మాణం ggml_cgraph * cgraph, పూర్ణాంకం n_దారాలు) {
    నిర్మాణం ggml_cplan cplan = ggml_లేఖాచిత్రం_plan(cgraph, n_దారాలు, ఖాళీ);

    cplan.work_దత్తాంశం = (uint8_t *)ggml_new_బఫరు(ctx, cplan.work_కొలత);

    తిరిగి ggml_లేఖాచిత్రం_compute(cgraph, &cplan);
}

శూన్య ggml_cpu_fp32_గమ్యం_fp32(స్థిరరాశి దశాంశం * x, దశాంశం * y, int64_t n) {
    జ్ఞాపకం_నకలు(y, x, n * పరిమాణం(దశాంశం));
}

శూన్య ggml_cpu_fp32_గమ్యం_fp16(స్థిరరాశి దశాంశం * x, ggml_fp16_t * y, int64_t n) {
    int64_t i = 0;
#ఒకవేళ defined(__F16C__)
#ఒకవేళ defined(__AVX512F__)
    కొరకు (; i + 15 < n; i += 16) {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        __m256i y_vec = _mm512_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256((__m256i *)(y + i), y_vec);
    }
#అయితే_ముగింపు
    కొరకు (; i + 7 < n; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m128i y_vec = _mm256_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i *)(y + i), y_vec);
    }
    కొరకు (; i + 3 < n; i += 4) {
        __m128 x_vec = _mm_loadu_ps(x + i);
        __m128i y_vec = _mm_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storel_epi64((__m128i *)(y + i), y_vec);
    }
#లేదావేళ defined(__riscv_zvfh)
    కొరకు (పూర్ణాంకం vl; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(&x[i], vl);
        vfloat16m1_t vy = __riscv_vfncvt_f_f_w_f16m1(vx, vl);
        __riscv_vse16_v_f16m1((_Float16 *)&y[i], vy, vl);
    }
#లేదావేళ defined(__VXE__) || defined(__VXE2__)
    కొరకు (; i + 7 < n; i += 8) {
        స్థిరరాశి uint32x4_t v_yl = __lzs_f32cx4_to_f16(vec_xl(0, x + i + 0));
        స్థిరరాశి uint32x4_t v_yh = __lzs_f32cx4_to_f16(vec_xl(0, x + i + 4));
        vec_xst(vec_pack(v_yl, v_yh), 0, (uint16_t *)(y + i));
    }
#అయితే_ముగింపు
    కొరకు (; i < n; ++i) {
        y[i] = GGML_CPU_FP32_గమ్యం_FP16(x[i]);
    }
}

శూన్య ggml_cpu_fp16_గమ్యం_fp32(స్థిరరాశి ggml_fp16_t * x, దశాంశం * y, int64_t n) {
    int64_t i = 0;
#ఒకవేళ defined(__F16C__)
#ఒకవేళ defined(__AVX512F__)
    కొరకు (; i + 15 < n; i += 16) {
        __m256i x_vec = _mm256_loadu_si256((స్థిరరాశి __m256i *)(x + i));
        __m512 y_vec = _mm512_cvtph_ps(x_vec);
        _mm512_storeu_ps(y + i, y_vec);
    }
#అయితే_ముగింపు
    కొరకు (; i + 7 < n; i += 8) {
        __m128i x_vec = _mm_loadu_si128((స్థిరరాశి __m128i *)(x + i));
        __m256 y_vec = _mm256_cvtph_ps(x_vec);
        _mm256_storeu_ps(y + i, y_vec);
    }
    కొరకు (; i + 3 < n; i += 4) {
        __m128i x_vec = _mm_loadl_epi64((స్థిరరాశి __m128i *)(x + i));
        __m128 y_vec = _mm_cvtph_ps(x_vec);
        _mm_storeu_ps(y + i, y_vec);
    }

#లేదావేళ defined(__riscv_v_intrinsic) && defined(__riscv_zvfhmin)
    // calculate step size
    స్థిరరాశి పూర్ణాంకం epr = __riscv_vsetvlmax_e16m2();
    స్థిరరాశి పూర్ణాంకం అడుగు = epr * 2;
    స్థిరరాశి పూర్ణాంకం np = (n & ~(అడుగు - 1));

    // unroll by 2
    కొరకు (; i < np; i += అడుగు) {
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((స్థిరరాశి _Float16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vfloat16m2_t ax1 = __riscv_vle16_v_f16m2((స్థిరరాశి _Float16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvt_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    పూర్ణాంకం vl;
    కొరకు (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((స్థిరరాశి _Float16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }

#అయితే_ముగింపు

    కొరకు (; i < n; ++i) {
        y[i] = GGML_CPU_FP16_గమ్యం_FP32(x[i]);
    }
}

శూన్య ggml_cpu_fp32_గమ్యం_bf16(స్థిరరాశి దశాంశం * x, ggml_bf16_t * y, int64_t n) {
    int64_t i = 0;
    కొరకు (; i < n; ++i) {
        y[i] = GGML_FP32_గమ్యం_BF16(x[i]);
    }
}

శూన్య ggml_cpu_fp32_గమ్యం_i32(స్థిరరాశి దశాంశం * x, int32_t * y, int64_t n) {
    int64_t i = 0;
    కొరకు (; i < n; ++i) {
        y[i] = x[i];
    }
}

శూన్య ggml_cpu_bf16_గమ్యం_fp32(స్థిరరాశి ggml_bf16_t * x, దశాంశం * y, int64_t n) {
    int64_t i = 0;
#ఒకవేళ defined(__AVX2__)
#ఒకవేళ defined(__AVX512F__)
    కొరకు (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i,
                        _mm512_castsi512_ps(
                            _mm512_slli_epi32(
                                _mm512_cvtepu16_epi32(
                                    _mm256_loadu_si256(
                                        (స్థిరరాశి __m256i *)(x + i))),
                                16)));
    }
#అయితే_ముగింపు
    కొరకు (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i,
                        _mm256_castsi256_ps(
                            _mm256_slli_epi32(
                                _mm256_cvtepu16_epi32(
                                    _mm_loadu_si128(
                                        (స్థిరరాశి __m128i *)(x + i))),
                                16)));
    }
#లేదావేళ defined(__riscv_v_intrinsic) && defined(__riscv_zvfbfmin)
    // calculate step size
    స్థిరరాశి పూర్ణాంకం epr = __riscv_vsetvlmax_e16m2();
    స్థిరరాశి పూర్ణాంకం అడుగు = epr * 2;
    స్థిరరాశి పూర్ణాంకం np = (n & ~(అడుగు - 1));

    // unroll by 2
    కొరకు (; i < np; i += అడుగు) {
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((స్థిరరాశి __bf16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vbfloat16m2_t ax1 = __riscv_vle16_v_bf16m2((స్థిరరాశి __bf16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    పూర్ణాంకం vl;
    కొరకు (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((స్థిరరాశి __bf16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }
#అయితే_ముగింపు
    కొరకు (; i < n; i++) {
        y[i] = GGML_BF16_గమ్యం_FP32(x[i]);
    }
}

పూర్ణాంకం ggml_cpu_has_avx(శూన్య) {
#ఒకవేళ defined(__AVX__)
    తిరిగి 1;
#లేకపోతే
    తిరిగి 0;
#అయితే_ముగింపు
}

పూర్ణాంకం ggml_cpu_has_avx_vnni(శూన్య) {
#ఒకవేళ defined(__AVXVNNI__)
    తిరిగి 1;
#లేకపోతే
    తిరిగి 0;
#అయితే_ముగింపు
}

పూర్ణాంకం ggml_cpu_has_avx2(శూన్య) {
#ఒకవేళ defined(__AVX2__)
    తిరిగి 1;
#లేకపోతే
    తిరిగి 0;
#అయితే_ముగింపు
}

పూర్ణాంకం ggml_cpu_has_avx512(శూన్య) {
#ఒకవేళ defined(__AVX512F__)
    తిరిగి 1;
#లేకపోతే
    తిరిగి 0;
#అయితే_ముగింపు
}

పూర్ణాంకం ggml_cpu_has_avx512_vbmi(శూన్య) {
#ఒకవేళ defined(__AVX512VBMI__)
    తిరిగి 1;
#లేకపోతే
    తిరిగి 0;
#అయితే_ముగింపు
}

పూర్ణాంకం ggml_cpu_has_avx512_vnni(శూన్య) {
#ఒకవేళ defined(__AVX512VNNI__)
    తిరిగి 1;
#లేకపోతే
    తిరిగి 0;
#అయితే_ముగింపు
}

పూర్ణాంకం ggml_cpu_has_avx512_bf16(శూన్య) {
#ఒకవేళ defined(__AVX512BF16__)
    తిరిగి 1;
#లేకపోతే
    తిరిగి 0;
#అయితే_ముగింపు
}

పూర్ణాంకం ggml_cpu_has_amx_int8(శూన్య) {
#ఒకవేళ defined(__AMX_INT8__)
    తిరిగి 1;
#లేకపోతే
    తిరిగి 0;
#అయితే_ముగింపు
}

పూర్ణాంకం ggml_cpu_has_bmi2(శూన్య) {
#ఒకవేళ defined(__BMI2__)
    తిరిగి 1;
#లేకపోతే
    తిరిగి 0;
#అయితే_ముగింపు
}

పూర్ణాంకం ggml_cpu_has_fma(శూన్య) {
#ఒకవేళ defined(__FMA__)
    తిరిగి 1;
#లేకపోతే
    తిరిగి 0;
#అయితే_ముగింపు
}

పూర్ణాంకం ggml_cpu_has_arm_fma(శూన్య) {
#ఒకవేళ defined(__ARM_FEATURE_FMA)
    తిరిగి 1;
#లేకపోతే
    తిరిగి 0;
#అయితే_ముగింపు
}

పూర్ణాంకం ggml_cpu_has_riscv_v(శూన్య) {
#ఒకవేళ defined(__riscv_v_intrinsic)
    తిరిగి 1;
#లేకపోతే
    తిరిగి 0;
#అయితే_ముగింపు
}

పూర్ణాంకం ggml_cpu_get_rvv_vlen(శూన్య) {
#ఒకవేళ defined(__riscv) && defined(__riscv_v_intrinsic)
    తిరిగి ggml_riscv_arch_features.rvv_vlen;
#లేకపోతే
    తిరిగి 0;
#అయితే_ముగింపు
}

పూర్ణాంకం ggml_cpu_has_f16c(శూన్య) {
#ఒకవేళ defined(__F16C__)
    తిరిగి 1;
#లేకపోతే
    తిరిగి 0;
#అయితే_ముగింపు
}

పూర్ణాంకం ggml_cpu_has_fp16_va(శూన్య) {
#ఒకవేళ defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    తిరిగి 1;
#లేకపోతే
    తిరిగి 0;
#అయితే_ముగింపు
}

పూర్ణాంకం ggml_cpu_has_wasm_simd(శూన్య) {
#ఒకవేళ defined(__wasm_simd128__)
    తిరిగి 1;
#లేకపోతే
    తిరిగి 0;
#అయితే_ముగింపు
}

పూర్ణాంకం ggml_cpu_has_llamafile(శూన్య) {
#ఒకవేళ defined(GGML_USE_LLAMAFILE)
    తిరిగి 1;
#లేకపోతే
    తిరిగి 0;
#అయితే_ముగింపు
}

పూర్ణాంకం ggml_cpu_has_sse3(శూన్య) {
#ఒకవేళ defined(__SSE3__)
    తిరిగి 1;
#లేకపోతే
    తిరిగి 0;
#అయితే_ముగింపు
}

పూర్ణాంకం ggml_cpu_has_ssse3(శూన్య) {
#ఒకవేళ defined(__SSSE3__)
    తిరిగి 1;
#లేకపోతే
    తిరిగి 0;
#అయితే_ముగింపు
}

పూర్ణాంకం ggml_cpu_has_vsx(శూన్య) {
#ఒకవేళ defined(__POWER9_VECTOR__)
    తిరిగి 1;
#లేకపోతే
    తిరిగి 0;
#అయితే_ముగింపు
}

పూర్ణాంకం ggml_cpu_has_vxe(శూన్య) {
#ఒకవేళ defined(__VXE__) || defined(__VXE2__)
    తిరిగి 1;
#లేకపోతే
    తిరిగి 0;
#అయితే_ముగింపు
}

పూర్ణాంకం ggml_cpu_has_neon(శూన్య) {
#ఒకవేళ defined(__ARM_ARCH) && defined(__ARM_NEON)
    తిరిగి 1;
#లేకపోతే
    తిరిగి 0;
#అయితే_ముగింపు
}

పూర్ణాంకం ggml_cpu_has_dotprod(శూన్య) {
#ఒకవేళ defined(__ARM_ARCH) && defined(__ARM_FEATURE_DOTPROD)
    తిరిగి 1;
#లేకపోతే
    తిరిగి 0;
#అయితే_ముగింపు
}

పూర్ణాంకం ggml_cpu_has_sve(శూన్య) {
#ఒకవేళ defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    తిరిగి 1;
#లేకపోతే
    తిరిగి 0;
#అయితే_ముగింపు
}

పూర్ణాంకం ggml_cpu_has_matmul_int8(శూన్య) {
#ఒకవేళ defined(__ARM_ARCH) && defined(__ARM_FEATURE_MATMUL_INT8)
    తిరిగి 1;
#లేకపోతే
    తిరిగి 0;
#అయితే_ముగింపు
}

పూర్ణాంకం ggml_cpu_get_sve_cnt(శూన్య) {
#ఒకవేళ defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    తిరిగి ggml_arm_arch_features.sve_cnt;
#లేకపోతే
    తిరిగి 0;
#అయితే_ముగింపు
}

పూర్ణాంకం ggml_cpu_has_sme(శూన్య) {
#ఒకవేళ defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME)
    తిరిగి 1;
#లేకపోతే
    తిరిగి 0;
#అయితే_ముగింపు
}

పూర్ణాంకం ggml_cpu_has_sme2(శూన్య) {
#ఒకవేళ defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME2)
    తిరిగి 1;
#లేకపోతే
    తిరిగి 0;
#అయితే_ముగింపు
}

శూన్య ggml_cpu_init(శూన్య) {
    // needed to initialize ggml_time
    {
        నిర్మాణం ggml_init_params params = { 0, ఖాళీ, false };
        నిర్మాణం ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    ggml_critical_section_ప్రారంభం();

    స్థిర bool is_మొదటి_call = true;

    ఒకవేళ (is_మొదటి_call) {
        // initialize GELU, Quick GELU, SILU and EXP F32 tables
        {
            స్థిరరాశి uint64_t t_ప్రారంభం = ggml_సమయం_us(); వాడని(t_ప్రారంభం);

            కొరకు (పూర్ణాంకం i = 0; i < (1 << 16); ++i) {
                సంఘం {
                    uint16_t u16;
                    ggml_fp16_t fp16;
                } u = {i};
                దశాంశం f = GGML_COMPUTE_FP16_గమ్యం_FP32(u.fp16);
                ggml_పట్టిక_f32_f16[i] = f;
                ggml_పట్టిక_gelu_f16[i] = GGML_CPU_FP32_గమ్యం_FP16(ggml_gelu_f32(f));
                ggml_పట్టిక_gelu_శీఘ్ర_f16[i] = GGML_CPU_FP32_గమ్యం_FP16(ggml_gelu_శీఘ్ర_f32(f));
            }

            // initialize E8M0 half table (256 entries)
            కొరకు (పూర్ణాంకం i = 0; i < (1 << 8); ++i) {
                ggml_పట్టిక_f32_e8m0_half[i] = GGML_E8_M0_గమ్యం_FP32_HALF(i);
            }

            // initialize UE4M3 table (256 entries)
            కొరకు (పూర్ణాంకం i = 0; i < (1 << 8); ++i) {
                ggml_పట్టిక_f32_ue4m3[i] = ggml_ue4m3_గమ్యం_fp32(i);
            }

            స్థిరరాశి uint64_t t_చివర = ggml_సమయం_us(); వాడని(t_చివర);

            GGML_PRINT_DEBUG("%s: GELU, Quick GELU, SILU and EXP tables initialized in %f ms\n", __func__, (t_చివర - t_ప్రారంభం)/1000.0);

#నిర్వచితమైతే GGML_USE_OPENMP
            //if (!getenv("OMP_WAIT_POLICY")) {
            //    // set the wait policy to active, so that OpenMP threads don't sleep
            //    setenv("OMP_WAIT_POLICY", "active", 0)
            //}

            ఒకవేళ (!getenv("KMP_BLOCKTIME")) {
                // set the time to wait before sleeping a thread
                // this is less aggressive than setting the wait policy to active, but should achieve similar results in most cases
#నిర్వచితమైతే _WIN32
                _putenv_s("KMP_BLOCKTIME", "200"); // 200ms
#లేకపోతే
                setenv("KMP_BLOCKTIME", "200", 0); // 200ms
#అయితే_ముగింపు
            }
#అయితే_ముగింపు
        }

#ఒకవేళ defined(__ARM_ARCH)
        ggml_init_arm_arch_features();
#అయితే_ముగింపు

#ఒకవేళ defined(__riscv)
        ggml_init_riscv_arch_features();
#అయితే_ముగింపు

        {
            స్థిరరాశి అక్షరం * env = getenv("GGML_CPU_DISABLE_FUSION");
            ggml_cpu_disable_fusion = (env != ఖాళీ && atoi(env) == 1);
        }

        is_మొదటి_call = false;
    }

    ggml_critical_section_చివర();
}
