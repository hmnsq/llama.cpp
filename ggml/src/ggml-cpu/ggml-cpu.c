#परिभाषय _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#परिभाषय USE_MATH_परिभाषते // For M_PI on MSVC

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

#यदि defined(_MSC_VER) || defined(__MINGW32__)
#अन्तर्भाव <malloc.h> // using malloc.h with MSC/MINGW
#नोचेद्यदि !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#अन्तर्भाव <alloca.h>
#यदि_अन्तः

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
#यदि defined(__gnu_linux__)
#अन्तर्भाव <syscall.h>
#यदि_अन्तः

#यदि_परिभाषितम् GGML_USE_OPENMP
#अन्तर्भाव <omp.h>
#यदि_अन्तः

#यदि defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_MATMUL_INT8)
#अपरिभाषय GGML_USE_LLAMAFILE
#यदि_अन्तः

#यदि_परिभाषितम् GGML_USE_LLAMAFILE
#अन्तर्भाव "llamafile/sgemm.h"
#यदि_अन्तः

#यदि_परिभाषितम् GGML_USE_CPU_RISCV64_SPACEMIT
#    include "spacemit/ime.h"
#यदि_अन्तः

// Note: once we move threading into a separate C++ file
// will use std::hardware_destructive_interference_size instead of hardcoding it here
// and we'll use C++ attribute syntax.
#परिभाषय GGML_CACHE_पङ्क्तिः  64

#यदि defined(__clang__) || defined(__GNUC__)
#परिभाषय GGML_CACHE_ALIGN __attribute__((aligned(GGML_CACHE_पङ्क्तिः)))
#यदि_अन्तः

#यदि defined(__has_feature)
#यदि __has_feature(thread_sanitizer)
#परिभाषय GGML_TSAN_ENABLED 1
#यदि_अन्तः
#अन्यथा  // __has_feature
#यदि defined(__SANITIZE_THREAD__)
#परिभाषय GGML_TSAN_ENABLED 1
#यदि_अन्तः
#यदि_अन्तः // __has_feature

#परिभाषय अप्रयुक्तम् GGML_अप्रयुक्तम्
#परिभाषय SWAP(x, y, T) कुरु { T SWAP = x; (x) = y; (y) = SWAP; } यावत् (0)

// precomputed f32 table for f16 (256 KB) (simd-mappings.h)
भिन्न ggml_सारणी_f32_f16[1 << 16];

// precomputed f32 table for e8m0 half (1 KB) (simd-mappings.h)
भिन्न ggml_सारणी_f32_e8m0_half[1 << 8];

// precomputed f32 table for ue4m3 (1 KB) (simd-mappings.h)
भिन्न ggml_सारणी_f32_ue4m3[1 << 8];

#यदि defined(__ARM_ARCH)
संरचना ggml_arm_arch_features_type {
    पूर्णाङ्क sve_cnt;
} ggml_arm_arch_features = { 0 };
#यदि_अन्तः

#यदि defined(__riscv)
संरचना ggml_riscv_arch_features_type {
    पूर्णाङ्क rvv_vlen;
} ggml_riscv_arch_features = { 0 };
#यदि_अन्तः

#यदि defined(_WIN32)

#परिभाषय WIN32_LEAN_AND_माध्यम्
#यदि_अपरिभाषितम् NOMINMAX
    #परिभाषय NOMINMAX
#यदि_अन्तः
#अन्तर्भाव <windows.h>

#यदि defined(_MSC_VER) && !defined(__clang__)
#परिभाषय GGML_CACHE_ALIGN __declspec(align(GGML_CACHE_पङ्क्तिः))

प्रकारनिर्देश चञ्चल LONG atomic_int;
प्रकारनिर्देश atomic_int atomic_bool;
प्रकारनिर्देश atomic_int atomic_flag;

#परिभाषय ATOMIC_FLAG_INIT 0

प्रकारनिर्देश गणना {
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
स्थिर LONG atomic_आरोपय(atomic_int * ptr) {
    प्रत्यागम InterlockedCompareExchange(ptr, 0, 0);
}
स्थिर LONG atomic_आरोपय_explicit(atomic_int * ptr, memory_order mo) {
    // TODO: add support for explicit memory order
    प्रत्यागम InterlockedCompareExchange(ptr, 0, 0);
}
स्थिर LONG atomic_fetch_सङ्कलय(atomic_int * ptr, LONG inc) {
    प्रत्यागम Interlocked_Exchange_सङ्कलय(ptr, inc);
}
स्थिर LONG atomic_fetch_सङ्कलय_explicit(atomic_int * ptr, LONG inc, memory_order mo) {
    // TODO: add support for explicit memory order
    प्रत्यागम Interlocked_Exchange_सङ्कलय(ptr, inc);
}
स्थिर atomic_bool atomic_flag_test_and_समुच्चयः(atomic_flag * ptr) {
    प्रत्यागम InterlockedExchange(ptr, 1);
}
स्थिर शून्य atomic_flag_clear(atomic_flag * ptr) {
    InterlockedExchange(ptr, 0);
}
स्थिर शून्य atomic_thread_fence(memory_order mo) {
    स्मृतिः_Barrier();
}
#अन्यथा // clang
#अन्तर्भाव <stdatomic.h>
#यदि_अन्तः

प्रकारनिर्देश HANDLE pthread_t;

प्रकारनिर्देश DWORD thread_ret_t;
स्थिर पूर्णाङ्क pthread_create(pthread_t * बहिर्गमनम्, शून्य * अप्रयुक्तम्_2, thread_ret_t(*func)(शून्य *), शून्य * arg) {
    (शून्य) अप्रयुक्तम्_2;
    HANDLE handle = CreateThread(रिक्त, 0, (LPTHREAD_START_ROUTINE) func, arg, 0, रिक्त);
    यदि (handle == रिक्त)
    {
        प्रत्यागम EAGAIN;
    }

    *बहिर्गमनम् = handle;
    प्रत्यागम 0;
}

स्थिर पूर्णाङ्क pthread_join(pthread_t thread, शून्य * अप्रयुक्तम्_2) {
    (शून्य) अप्रयुक्तम्_2;
    पूर्णाङ्क ret = (पूर्णाङ्क) WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    प्रत्यागम ret;
}

स्थिर पूर्णाङ्क sched_yield (शून्य) {
    Sleep (0);
    प्रत्यागम 0;
}
#अन्यथा

#अन्तर्भाव <pthread.h>
#अन्तर्भाव <stdatomic.h>
#अन्तर्भाव <sched.h>
#यदि defined(__FreeBSD__)
#अन्तर्भाव <pthread_np.h>
#यदि_अन्तः

प्रकारनिर्देश शून्य * thread_ret_t;

#अन्तर्भाव <sys/types.h>
#अन्तर्भाव <sys/stat.h>
#अन्तर्भाव <unistd.h>

#यदि_अन्तः

प्रकारनिर्देश pthread_t ggml_thread_t;

#परिभाषय GGML_THREADPOOL_N_तन्तवः_आवरणम् (0xffffU)
#परिभाषय GGML_THREADPOOL_N_तन्तवः_द्वयङ्काः (16)

#यदि defined(__APPLE__)
#अन्तर्भाव <unistd.h>
#अन्तर्भाव <mach/mach.h>
#अन्तर्भाव <TargetConditionals.h>
#यदि_अन्तः

स्थिर ध्रुव संरचना ggml_type_traits_cpu type_traits_cpu[GGML_TYPE_गणनम्] = {
    [GGML_TYPE_F32] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_गन्तव्यम्_fp32,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_f32,
        .vec_dot_type             = GGML_TYPE_F32,
        .nrows                    = 1,
    },
    [GGML_TYPE_F16] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_गन्तव्यम्_fp16,
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
#यदि defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#अन्यथा
        .nrows                    = 1,
#यदि_अन्तः
    },
    [GGML_TYPE_Q4_1] = {
        .from_float               = quantize_row_q4_1,
        .vec_dot                  = ggml_vec_dot_q4_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
#यदि defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#अन्यथा
        .nrows                    = 1,
#यदि_अन्तः
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
#यदि defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#अन्यथा
        .nrows                    = 1,
#यदि_अन्तः
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
#यदि defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#अन्यथा
        .nrows                    = 1,
#यदि_अन्तः
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
#यदि defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#अन्यथा
        .nrows                    = 1,
#यदि_अन्तः
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
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_गन्तव्यम्_bf16,
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
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_गन्तव्यम्_i32,
    },
};

ध्रुव संरचना ggml_type_traits_cpu * ggml_get_type_traits_cpu(गणना ggml_type type) {
    प्रत्यागम &type_traits_cpu[type];
}

//
// Threading defs
//

प्रकारनिर्देश pthread_t          ggml_thread_t;

#यदि defined(_WIN32)

प्रकारनिर्देश CONDITION_VARIABLE ggml_cond_t;
प्रकारनिर्देश SRWLOCK            ggml_mutex_t;

#परिभाषय ggml_mutex_init(m)   InitializeSRWLock(m)
#परिभाषय ggml_mutex_destroy(m)
#परिभाषय ggml_mutex_तालकम्(m)   AcquireSRWLockExclusive(m)
#परिभाषय ggml_mutex_unlock(m) ReleaseSRWLockExclusive(m)
#परिभाषय ggml_mutex_तालकम्_shared(m)   AcquireSRWLockShared(m)
#परिभाषय ggml_mutex_unlock_shared(m) ReleaseSRWLockShared(m)

#परिभाषय ggml_cond_init(c)    InitializeConditionVariable(c)
#परिभाषय ggml_cond_destroy(c)
#परिभाषय ggml_cond_wait(c, m) SleepConditionVariableSRW(c, m, INFINITE, CONDITION_VARIABLE_LOCKMODE_SHARED)
#परिभाषय ggml_cond_प्रसारणम्(c) WakeAllConditionVariable(c)

#परिभाषय ggml_thread_create pthread_create
#परिभाषय ggml_thread_join   pthread_join

#अन्यथा

प्रकारनिर्देश pthread_cond_t     ggml_cond_t;
प्रकारनिर्देश pthread_mutex_t    ggml_mutex_t;

#परिभाषय ggml_mutex_init(m)          pthread_mutex_init(m, रिक्त)
#परिभाषय ggml_mutex_destroy(m)       pthread_mutex_destroy(m)
#परिभाषय ggml_mutex_तालकम्(m)          pthread_mutex_तालकम्(m)
#परिभाषय ggml_mutex_unlock(m)        pthread_mutex_unlock(m)
#परिभाषय ggml_mutex_तालकम्_shared(m)   pthread_mutex_तालकम्(m)
#परिभाषय ggml_mutex_unlock_shared(m) pthread_mutex_unlock(m)

#परिभाषय ggml_तालकम्_init(x)    अप्रयुक्तम्(x)
#परिभाषय ggml_तालकम्_destroy(x) अप्रयुक्तम्(x)
#यदि defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64) && !defined(_M_ARM64EC))
#परिभाषय ggml_तालकम्_तालकम्(x)    _mm_pause()
#अन्यथा
#परिभाषय ggml_तालकम्_तालकम्(x)    अप्रयुक्तम्(x)
#यदि_अन्तः
#परिभाषय ggml_तालकम्_unlock(x)  अप्रयुक्तम्(x)

#परिभाषय GGML_तालकम्_INITIALIZER 0
#परिभाषय ggml_cond_init(c)      pthread_cond_init(c, रिक्त)
#परिभाषय ggml_cond_destroy(c)   pthread_cond_destroy(c)
#परिभाषय ggml_cond_wait(c, m)   pthread_cond_wait(c, m)
#परिभाषय ggml_cond_प्रसारणम्(c) pthread_cond_प्रसारणम्(c)

#परिभाषय ggml_thread_create pthread_create
#परिभाषय ggml_thread_join   pthread_join

#यदि_अन्तः

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
    atomic_bool pause;        // Used for pausing the threadpool or individual threads
    atomic_int  abort;        // Used for aborting processing of a graph

    संरचना ggml_compute_state * workers;   // per thread state
    पूर्णाङ्क          n_तन्तवः;   // Number of threads in the pool
    int32_t      prio;        // Scheduling priority
    uint32_t     poll;        // Polling level (0 - no polling)

    गणना ggml_स्थितिः ec;
};

// Per-thread state
संरचना ggml_compute_state {
#यदि_अपरिभाषितम् GGML_USE_OPENMP
    ggml_thread_t thrd;
    पूर्णाङ्क  चरमः_graph;
    bool pending;
#यदि_अन्तः
    bool cpumask[GGML_MAX_N_तन्तवः];
    संरचना ggml_threadpool * threadpool;
    पूर्णाङ्क ith;
};

// Helpers for polling loops
#यदि defined(__aarch64__) && ( defined(__clang__) || defined(__GNUC__) )
स्थिर अन्तःस्थ शून्य ggml_thread_cpu_relax(शून्य) {
    __asm__ चञ्चल("yield" ::: "memory");
}
#नोचेद्यदि defined(__x86_64__)
स्थिर अन्तःस्थ शून्य ggml_thread_cpu_relax(शून्य) {
    _mm_pause();
}
#नोचेद्यदि defined(__riscv)
स्थिर अन्तःस्थ शून्य ggml_thread_cpu_relax(शून्य) {
    #यदि_परिभाषितम् __riscv_zihintpause
        __asm__ __volatile__ ("pause");
    #अन्यथा
        /* Encoding of the pause instruction */
        __asm__ __volatile__ (".4byte 0x100000F");
    #यदि_अन्तः
}
#अन्यथा
स्थिर अन्तःस्थ शून्य ggml_thread_cpu_relax(शून्य) {;}
#यदि_अन्तः

//
// NUMA support
//

#परिभाषय GGML_NUMA_MAX_NODES 8
#परिभाषय GGML_NUMA_MAX_CPUS 512

संरचना ggml_numa_ग्रन्थिः {
    uint32_t cpus[GGML_NUMA_MAX_CPUS]; // hardware threads on this node
    uint32_t n_cpus;
};

संरचना ggml_numa_nodes {
    गणना ggml_numa_strategy numa_strategy;
    संरचना ggml_numa_ग्रन्थिः nodes[GGML_NUMA_MAX_NODES];
    uint32_t n_nodes;
    uint32_t योग_cpus; // hardware threads on system
    uint32_t current_ग्रन्थिः; // node on which main process is execting
#यदि defined(__gnu_linux__)
    cpu_समुच्चयः_t cpuset; // cpuset from numactl
#अन्यथा
    uint32_t cpuset; // no NUMA support outside of Linux at this time. Use a portable datatype
#यदि_अन्तः
};

//
// ggml state
//

संरचना ggml_state {
    संरचना ggml_numa_nodes numa;
};

स्थिर संरचना ggml_state g_state = {0};

शून्य ggml_barrier(संरचना ggml_threadpool * tp) {
    पूर्णाङ्क n_तन्तवः = atomic_आरोपय_explicit(&tp->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_तन्तवः_आवरणम्;
    यदि (n_तन्तवः == 1) {
        प्रत्यागम;
    }

#यदि_परिभाषितम् GGML_USE_OPENMP
    #निर्देशः omp barrier
#अन्यथा
    पूर्णाङ्क n_passed = atomic_आरोपय_explicit(&tp->n_barrier_passed, memory_order_relaxed);

    // enter barrier (full seq-cst fence)
    पूर्णाङ्क n_barrier = atomic_fetch_सङ्कलय_explicit(&tp->n_barrier, 1, memory_order_seq_cst);

    यदि (n_barrier == (n_तन्तवः - 1)) {
        // last thread
        atomic_store_explicit(&tp->n_barrier, 0, memory_order_relaxed);

        // exit barrier (full seq-cst fence)
        atomic_fetch_सङ्कलय_explicit(&tp->n_barrier_passed, 1, memory_order_seq_cst);
        प्रत्यागम;
    }

    // wait for other threads
    यावत् (atomic_आरोपय_explicit(&tp->n_barrier_passed, memory_order_relaxed) == n_passed) {
        ggml_thread_cpu_relax();
    }

    // exit barrier (full seq-cst fence)
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #यदि_परिभाषितम् GGML_TSAN_ENABLED
    atomic_fetch_सङ्कलय_explicit(&tp->n_barrier_passed, 0, memory_order_seq_cst);
    #अन्यथा
    atomic_thread_fence(memory_order_seq_cst);
    #यदि_अन्तः
#यदि_अन्तः
}

शून्य ggml_threadpool_chunk_समुच्चयः(संरचना ggml_threadpool * tp, पूर्णाङ्क मानम्) {
    atomic_store_explicit(&tp->current_chunk, मानम्, memory_order_relaxed);
}

पूर्णाङ्क ggml_threadpool_chunk_सङ्कलय(संरचना ggml_threadpool * tp, पूर्णाङ्क मानम्) {
    प्रत्यागम atomic_fetch_सङ्कलय_explicit(&tp->current_chunk, मानम्, memory_order_relaxed);
}

#यदि defined(__gnu_linux__)
स्थिर cpu_समुच्चयः_t ggml_get_numa_affinity(शून्य) {
    cpu_समुच्चयः_t cpuset;
    pthread_t thread;
    thread = pthread_self();
    CPU_ZERO(&cpuset);
    pthread_getaffinity_np(thread, परिमाण(cpu_समुच्चयः_t), &cpuset);
    प्रत्यागम cpuset;
}
#अन्यथा
स्थिर uint32_t ggml_get_numa_affinity(शून्य) {
    प्रत्यागम 0; // no NUMA support
}
#यदि_अन्तः

शून्य ggml_numa_init(गणना ggml_numa_strategy numa_flag) {
    यदि (g_state.numa.n_nodes > 0) {
        fprintf(stderr, "ggml_numa_init: NUMA already initialized\n");

        प्रत्यागम;
    }

#यदि defined(__gnu_linux__)
    संरचना stat st;
    वर्ण मार्गः[256];
    पूर्णाङ्क rv;

    // set numa scheme
    g_state.numa.numa_strategy = numa_flag;

    GGML_PRINT_DEBUG("numa strategy %u\n",g_state.numa.numa_strategy);

    g_state.numa.cpuset = ggml_get_numa_affinity();

    // enumerate nodes
    यावत् (g_state.numa.n_nodes < GGML_NUMA_MAX_NODES) {
        rv = snprintf(मार्गः, परिमाण(मार्गः), "/sys/devices/system/node/node%u", g_state.numa.n_nodes);
        GGML_ASSERT(rv > 0 && (अचिह्नित)rv < परिमाण(मार्गः));
        यदि (stat(मार्गः, &st) != 0) { भङ्ग; }
        ++g_state.numa.n_nodes;
    }

    // enumerate CPUs
    यावत् (g_state.numa.योग_cpus < GGML_NUMA_MAX_CPUS) {
        rv = snprintf(मार्गः, परिमाण(मार्गः), "/sys/devices/system/cpu/cpu%u", g_state.numa.योग_cpus);
        GGML_ASSERT(rv > 0 && (अचिह्नित)rv < परिमाण(मार्गः));
        यदि (stat(मार्गः, &st) != 0) { भङ्ग; }
        ++g_state.numa.योग_cpus;
    }

    GGML_PRINT_DEBUG("found %u numa nodes, %u CPUs\n", g_state.numa.n_nodes, g_state.numa.योग_cpus);

    // figure out which node we're on
    uint current_cpu;
    पूर्णाङ्क getcpu_ret = 0;
#यदि __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ > 33) || defined(__COSMOPOLITAN__)
    getcpu_ret = getcpu(&current_cpu, &g_state.numa.current_ग्रन्थिः);
#अन्यथा
    // old glibc doesn't have a wrapper for this call. Fall back on direct syscall
#   यदि !defined(SYS_getcpu) && defined(SYS_get_cpu)
#       define SYS_getcpu SYS_get_cpu // some older glibc versions use this name
#   endif
    getcpu_ret = syscall(SYS_getcpu, &current_cpu, &g_state.numa.current_ग्रन्थिः);
#यदि_अन्तः

    यदि (g_state.numa.n_nodes < 1 || g_state.numa.योग_cpus < 1 || getcpu_ret != 0) {
        g_state.numa.n_nodes = 0;
        प्रत्यागम;
    }

    GGML_PRINT_DEBUG("found our process on numa node %u, CPU %u\n", g_state.numa.current_ग्रन्थिः, current_cpu);

    प्रति (uint32_t n = 0; n < g_state.numa.n_nodes; ++n) {
        संरचना ggml_numa_ग्रन्थिः * ग्रन्थिः = &g_state.numa.nodes[n];
        GGML_PRINT_DEBUG("CPUs on node %u:", n);
        ग्रन्थिः->n_cpus = 0;
        प्रति (uint32_t c = 0; c < g_state.numa.योग_cpus; ++c) {
            rv = snprintf(मार्गः, परिमाण(मार्गः), "/sys/devices/system/node/node%u/cpu%u", n, c);
            GGML_ASSERT(rv > 0 && (अचिह्नित)rv < परिमाण(मार्गः));
            यदि (stat(मार्गः, &st) == 0) {
                ग्रन्थिः->cpus[ग्रन्थिः->n_cpus++] = c;
                GGML_PRINT_DEBUG(" %u", c);
            }
        }
        GGML_PRINT_DEBUG("\n");
    }

    यदि (ggml_is_numa()) {
        FILE *fptr = fopen("/proc/sys/kernel/numa_balancing", "r");
        यदि (fptr != रिक्त) {
            वर्ण buf[42];
            यदि (fgets(buf, परिमाण(buf), fptr) && वाक्ये_तुलय(buf, "0\n", परिमाण(buf)) != 0) {
                GGML_LOG_WARN("/proc/sys/kernel/numa_balancing is enabled, this has been observed to impair performance\n");
            }
            fclose(fptr);
        }
    }
#अन्यथा
    अप्रयुक्तम्(numa_flag);
    // TODO
#यदि_अन्तः
}

bool ggml_is_numa(शून्य) {
    प्रत्यागम g_state.numa.n_nodes > 1;
}

#यदि defined(__ARM_ARCH)
#यदि defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
#अन्तर्भाव <arm_sve.h>
स्थिर शून्य ggml_init_arm_arch_features(शून्य) {
    ggml_arm_arch_features.sve_cnt = svcntb();
}
#अन्यथा
स्थिर शून्य ggml_init_arm_arch_features(शून्य) {}
#यदि_अन्तः
#यदि_अन्तः // __ARM_ARCH

#यदि defined(__riscv) && defined(__riscv_v_intrinsic)
#अन्तर्भाव <riscv_vector.h>
स्थिर शून्य ggml_init_riscv_arch_features(शून्य) {
    ggml_riscv_arch_features.rvv_vlen = __riscv_vlenb();
}
#अन्यथा
स्थिर शून्य ggml_init_riscv_arch_features(शून्य) {}
#यदि_अन्तः

संरचना ggml_tensor * ggml_new_i32(संरचना ggml_context * ctx, int32_t मानम्) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    संरचना ggml_tensor * परिणाम = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

    ggml_समुच्चयः_i32(परिणाम, मानम्);

    प्रत्यागम परिणाम;
}

संरचना ggml_tensor * ggml_new_f32(संरचना ggml_context * ctx, भिन्न मानम्) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    संरचना ggml_tensor * परिणाम = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    ggml_समुच्चयः_f32(परिणाम, मानम्);

    प्रत्यागम परिणाम;
}

संरचना ggml_tensor * ggml_समुच्चयः_i32 (संरचना ggml_tensor * tensor, int32_t मानम्) {
    ध्रुव पूर्णाङ्क n     = ggml_nrows(tensor);
    ध्रुव पूर्णाङ्क nc    = tensor->ne[0];
    ध्रुव size_t n1 = tensor->nb[1];

    वर्ण * ध्रुव दत्त = tensor->दत्त;

    विकल्प (tensor->type) {
        पक्ष GGML_TYPE_I8:
            {
                प्रतिज्ञा(tensor->nb[0] == परिमाण(int8_t));
                प्रति (पूर्णाङ्क i = 0; i < n; i++) {
                    ggml_vec_समुच्चयः_i8(nc, (int8_t *)(दत्त + i*n1), मानम्);
                }
            } भङ्ग;
        पक्ष GGML_TYPE_I16:
            {
                प्रतिज्ञा(tensor->nb[0] == परिमाण(int16_t));
                प्रति (पूर्णाङ्क i = 0; i < n; i++) {
                    ggml_vec_समुच्चयः_i16(nc, (int16_t *)(दत्त + i*n1), मानम्);
                }
            } भङ्ग;
        पक्ष GGML_TYPE_I32:
            {
                प्रतिज्ञा(tensor->nb[0] == परिमाण(int32_t));
                प्रति (पूर्णाङ्क i = 0; i < n; i++) {
                    ggml_vec_समुच्चयः_i32(nc, (int32_t *)(दत्त + i*n1), मानम्);
                }
            } भङ्ग;
        पक्ष GGML_TYPE_F16:
            {
                प्रतिज्ञा(tensor->nb[0] == परिमाण(ggml_fp16_t));
                प्रति (पूर्णाङ्क i = 0; i < n; i++) {
                    ggml_vec_समुच्चयः_f16(nc, (ggml_fp16_t *)(दत्त + i*n1), GGML_CPU_FP32_गन्तव्यम्_FP16(मानम्));
                }
            } भङ्ग;
        पक्ष GGML_TYPE_BF16:
            {
                प्रतिज्ञा(tensor->nb[0] == परिमाण(ggml_fp16_t));
                प्रति (पूर्णाङ्क i = 0; i < n; i++) {
                    ggml_vec_समुच्चयः_bf16(nc, (ggml_bf16_t *)(दत्त + i*n1), GGML_FP32_गन्तव्यम्_BF16(मानम्));
                }
            } भङ्ग;
        पक्ष GGML_TYPE_F32:
            {
                प्रतिज्ञा(tensor->nb[0] == परिमाण(भिन्न));
                प्रति (पूर्णाङ्क i = 0; i < n; i++) {
                    ggml_vec_समुच्चयः_f32(nc, (भिन्न *)(दत्त + i*n1), मानम्);
                }
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }

    प्रत्यागम tensor;
}

संरचना ggml_tensor * ggml_समुच्चयः_f32(संरचना ggml_tensor * tensor, भिन्न मानम्) {
    ध्रुव पूर्णाङ्क n     = ggml_nrows(tensor);
    ध्रुव पूर्णाङ्क nc    = tensor->ne[0];
    ध्रुव size_t n1 = tensor->nb[1];

    वर्ण * ध्रुव दत्त = tensor->दत्त;

    विकल्प (tensor->type) {
        पक्ष GGML_TYPE_I8:
            {
                प्रतिज्ञा(tensor->nb[0] == परिमाण(int8_t));
                प्रति (पूर्णाङ्क i = 0; i < n; i++) {
                    ggml_vec_समुच्चयः_i8(nc, (int8_t *)(दत्त + i*n1), मानम्);
                }
            } भङ्ग;
        पक्ष GGML_TYPE_I16:
            {
                प्रतिज्ञा(tensor->nb[0] == परिमाण(int16_t));
                प्रति (पूर्णाङ्क i = 0; i < n; i++) {
                    ggml_vec_समुच्चयः_i16(nc, (int16_t *)(दत्त + i*n1), मानम्);
                }
            } भङ्ग;
        पक्ष GGML_TYPE_I32:
            {
                प्रतिज्ञा(tensor->nb[0] == परिमाण(int32_t));
                प्रति (पूर्णाङ्क i = 0; i < n; i++) {
                    ggml_vec_समुच्चयः_i32(nc, (int32_t *)(दत्त + i*n1), मानम्);
                }
            } भङ्ग;
        पक्ष GGML_TYPE_F16:
            {
                प्रतिज्ञा(tensor->nb[0] == परिमाण(ggml_fp16_t));
                प्रति (पूर्णाङ्क i = 0; i < n; i++) {
                    ggml_vec_समुच्चयः_f16(nc, (ggml_fp16_t *)(दत्त + i*n1), GGML_CPU_FP32_गन्तव्यम्_FP16(मानम्));
                }
            } भङ्ग;
        पक्ष GGML_TYPE_BF16:
            {
                प्रतिज्ञा(tensor->nb[0] == परिमाण(ggml_bf16_t));
                प्रति (पूर्णाङ्क i = 0; i < n; i++) {
                    ggml_vec_समुच्चयः_bf16(nc, (ggml_bf16_t *)(दत्त + i*n1), GGML_FP32_गन्तव्यम्_BF16(मानम्));
                }
            } भङ्ग;
        पक्ष GGML_TYPE_F32:
            {
                प्रतिज्ञा(tensor->nb[0] == परिमाण(भिन्न));
                प्रति (पूर्णाङ्क i = 0; i < n; i++) {
                    ggml_vec_समुच्चयः_f32(nc, (भिन्न *)(दत्त + i*n1), मानम्);
                }
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }

    प्रत्यागम tensor;
}

int32_t ggml_get_i32_1d(ध्रुव संरचना ggml_tensor * tensor, पूर्णाङ्क i) {
    यदि (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_सूचकांक(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        प्रत्यागम ggml_get_i32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    विकल्प (tensor->type) {
        पक्ष GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == परिमाण(int8_t));
                प्रत्यागम ((int8_t *)(tensor->दत्त))[i];
            }
        पक्ष GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == परिमाण(int16_t));
                प्रत्यागम ((int16_t *)(tensor->दत्त))[i];
            }
        पक्ष GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == परिमाण(int32_t));
                प्रत्यागम ((int32_t *)(tensor->दत्त))[i];
            }
        पक्ष GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == परिमाण(ggml_fp16_t));
                प्रत्यागम GGML_CPU_FP16_गन्तव्यम्_FP32(((ggml_fp16_t *)(tensor->दत्त))[i]);
            }
        पक्ष GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == परिमाण(ggml_bf16_t));
                प्रत्यागम GGML_BF16_गन्तव्यम्_FP32(((ggml_bf16_t *)(tensor->दत्त))[i]);
            }
        पक्ष GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == परिमाण(भिन्न));
                प्रत्यागम ((भिन्न *)(tensor->दत्त))[i];
            }
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

शून्य ggml_समुच्चयः_i32_1d(ध्रुव संरचना ggml_tensor * tensor, पूर्णाङ्क i, int32_t मानम्) {
    यदि (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_सूचकांक(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_समुच्चयः_i32_nd(tensor, id[0], id[1], id[2], id[3], मानम्);
        प्रत्यागम;
    }
    विकल्प (tensor->type) {
        पक्ष GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == परिमाण(int8_t));
                ((int8_t *)(tensor->दत्त))[i] = मानम्;
            } भङ्ग;
        पक्ष GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == परिमाण(int16_t));
                ((int16_t *)(tensor->दत्त))[i] = मानम्;
            } भङ्ग;
        पक्ष GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == परिमाण(int32_t));
                ((int32_t *)(tensor->दत्त))[i] = मानम्;
            } भङ्ग;
        पक्ष GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == परिमाण(ggml_fp16_t));
                ((ggml_fp16_t *)(tensor->दत्त))[i] = GGML_CPU_FP32_गन्तव्यम्_FP16(मानम्);
            } भङ्ग;
        पक्ष GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == परिमाण(ggml_bf16_t));
                ((ggml_bf16_t *)(tensor->दत्त))[i] = GGML_FP32_गन्तव्यम्_BF16(मानम्);
            } भङ्ग;
        पक्ष GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == परिमाण(भिन्न));
                ((भिन्न *)(tensor->दत्त))[i] = मानम्;
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

int32_t ggml_get_i32_nd(ध्रुव संरचना ggml_tensor * tensor, पूर्णाङ्क i0, पूर्णाङ्क i1, पूर्णाङ्क i2, पूर्णाङ्क i3) {
    शून्य * दत्त   = (वर्ण *) tensor->दत्त + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    विकल्प (tensor->type) {
        पक्ष GGML_TYPE_I8:
            प्रत्यागम ((int8_t *) दत्त)[0];
        पक्ष GGML_TYPE_I16:
            प्रत्यागम ((int16_t *) दत्त)[0];
        पक्ष GGML_TYPE_I32:
            प्रत्यागम ((int32_t *) दत्त)[0];
        पक्ष GGML_TYPE_F16:
            प्रत्यागम GGML_CPU_FP16_गन्तव्यम्_FP32(((ggml_fp16_t *) दत्त)[0]);
        पक्ष GGML_TYPE_BF16:
            प्रत्यागम GGML_BF16_गन्तव्यम्_FP32(((ggml_bf16_t *) दत्त)[0]);
        पक्ष GGML_TYPE_F32:
            प्रत्यागम ((भिन्न *) दत्त)[0];
        स्वभाव:
            GGML_ABORT("fatal error");
    }
}

शून्य ggml_समुच्चयः_i32_nd(ध्रुव संरचना ggml_tensor * tensor, पूर्णाङ्क i0, पूर्णाङ्क i1, पूर्णाङ्क i2, पूर्णाङ्क i3, int32_t मानम्) {
    शून्य * दत्त   = (वर्ण *) tensor->दत्त + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    विकल्प (tensor->type) {
        पक्ष GGML_TYPE_I8:
            {
                ((int8_t *)(दत्त))[0] = मानम्;
            } भङ्ग;
        पक्ष GGML_TYPE_I16:
            {
                ((int16_t *)(दत्त))[0] = मानम्;
            } भङ्ग;
        पक्ष GGML_TYPE_I32:
            {
                ((int32_t *)(दत्त))[0] = मानम्;
            } भङ्ग;
        पक्ष GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(दत्त))[0] = GGML_CPU_FP32_गन्तव्यम्_FP16(मानम्);
            } भङ्ग;
        पक्ष GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(दत्त))[0] = GGML_FP32_गन्तव्यम्_BF16(मानम्);
            } भङ्ग;
        पक्ष GGML_TYPE_F32:
            {
                ((भिन्न *)(दत्त))[0] = मानम्;
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

भिन्न ggml_get_f32_1d(ध्रुव संरचना ggml_tensor * tensor, पूर्णाङ्क i) {
    यदि (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_सूचकांक(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        प्रत्यागम ggml_get_f32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    विकल्प (tensor->type) {
        पक्ष GGML_TYPE_I8:
            {
                प्रत्यागम ((int8_t *)(tensor->दत्त))[i];
            }
        पक्ष GGML_TYPE_I16:
            {
                प्रत्यागम ((int16_t *)(tensor->दत्त))[i];
            }
        पक्ष GGML_TYPE_I32:
            {
                प्रत्यागम ((int32_t *)(tensor->दत्त))[i];
            }
        पक्ष GGML_TYPE_F16:
            {
                प्रत्यागम GGML_CPU_FP16_गन्तव्यम्_FP32(((ggml_fp16_t *)(tensor->दत्त))[i]);
            }
        पक्ष GGML_TYPE_BF16:
            {
                प्रत्यागम GGML_BF16_गन्तव्यम्_FP32(((ggml_bf16_t *)(tensor->दत्त))[i]);
            }
        पक्ष GGML_TYPE_F32:
            {
                प्रत्यागम ((भिन्न *)(tensor->दत्त))[i];
            }
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

शून्य ggml_समुच्चयः_f32_1d(ध्रुव संरचना ggml_tensor * tensor, पूर्णाङ्क i, भिन्न मानम्) {
    यदि (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_सूचकांक(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_समुच्चयः_f32_nd(tensor, id[0], id[1], id[2], id[3], मानम्);
        प्रत्यागम;
    }
    विकल्प (tensor->type) {
        पक्ष GGML_TYPE_I8:
            {
                ((int8_t *)(tensor->दत्त))[i] = मानम्;
            } भङ्ग;
        पक्ष GGML_TYPE_I16:
            {
                ((int16_t *)(tensor->दत्त))[i] = मानम्;
            } भङ्ग;
        पक्ष GGML_TYPE_I32:
            {
                ((int32_t *)(tensor->दत्त))[i] = मानम्;
            } भङ्ग;
        पक्ष GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(tensor->दत्त))[i] = GGML_CPU_FP32_गन्तव्यम्_FP16(मानम्);
            } भङ्ग;
        पक्ष GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(tensor->दत्त))[i] = GGML_FP32_गन्तव्यम्_BF16(मानम्);
            } भङ्ग;
        पक्ष GGML_TYPE_F32:
            {
                ((भिन्न *)(tensor->दत्त))[i] = मानम्;
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

भिन्न ggml_get_f32_nd(ध्रुव संरचना ggml_tensor * tensor, पूर्णाङ्क i0, पूर्णाङ्क i1, पूर्णाङ्क i2, पूर्णाङ्क i3) {
    शून्य * दत्त   = (वर्ण *) tensor->दत्त + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    विकल्प (tensor->type) {
        पक्ष GGML_TYPE_I8:
            प्रत्यागम ((int8_t *) दत्त)[0];
        पक्ष GGML_TYPE_I16:
            प्रत्यागम ((int16_t *) दत्त)[0];
        पक्ष GGML_TYPE_I32:
            प्रत्यागम ((int32_t *) दत्त)[0];
        पक्ष GGML_TYPE_F16:
            प्रत्यागम GGML_CPU_FP16_गन्तव्यम्_FP32(((ggml_fp16_t *) दत्त)[0]);
        पक्ष GGML_TYPE_BF16:
            प्रत्यागम GGML_BF16_गन्तव्यम्_FP32(((ggml_bf16_t *) दत्त)[0]);
        पक्ष GGML_TYPE_F32:
            प्रत्यागम ((भिन्न *) दत्त)[0];
        स्वभाव:
            GGML_ABORT("fatal error");
    }
}

शून्य ggml_समुच्चयः_f32_nd(ध्रुव संरचना ggml_tensor * tensor, पूर्णाङ्क i0, पूर्णाङ्क i1, पूर्णाङ्क i2, पूर्णाङ्क i3, भिन्न मानम्) {
    शून्य * दत्त   = (वर्ण *) tensor->दत्त + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    विकल्प (tensor->type) {
        पक्ष GGML_TYPE_I8:
            {
                ((int8_t *)(दत्त))[0] = मानम्;
            } भङ्ग;
        पक्ष GGML_TYPE_I16:
            {
                ((int16_t *)(दत्त))[0] = मानम्;
            } भङ्ग;
        पक्ष GGML_TYPE_I32:
            {
                ((int32_t *)(दत्त))[0] = मानम्;
            } भङ्ग;
        पक्ष GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(दत्त))[0] = GGML_CPU_FP32_गन्तव्यम्_FP16(मानम्);
            } भङ्ग;
        पक्ष GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(दत्त))[0] = GGML_FP32_गन्तव्यम्_BF16(मानम्);
            } भङ्ग;
        पक्ष GGML_TYPE_F32:
            {
                ((भिन्न *)(दत्त))[0] = मानम्;
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

////////////////////////////////////////////////////////////////////////////////

// ggml_compute_forward_mul_mat

स्थिर शून्य ggml_compute_forward_mul_mat_one_chunk(
    ध्रुव संरचना ggml_compute_params * params,
    संरचना ggml_tensor * dst,
    ध्रुव गणना ggml_type type,
    ध्रुव int64_t अङ्कः_पङ्क्तिकाः_प्रत्येकम्_vec_dot,
    ध्रुव int64_t ir0_start,
    ध्रुव int64_t ir0_अन्तः,
    ध्रुव int64_t ir1_start,
    ध्रुव int64_t ir1_अन्तः) {

    ध्रुव संरचना ggml_tensor * src0 = dst->src[0];
    ध्रुव संरचना ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विमान_OP_LOCALS

    ध्रुव bool src1_cont = ggml_is_contiguous(src1);

    ggml_vec_dot_t ध्रुव vec_dot      = type_traits_cpu[type].vec_dot;
    गणना ggml_type ध्रुव vec_dot_type = type_traits_cpu[type].vec_dot_type;

    // broadcast factors
    ध्रुव int64_t r2 = ne12 / ne02;
    ध्रुव int64_t r3 = ne13 / ne03;

    //printf("ir0_start = %6lld, ir0_end = %6lld, ir1_start = %6lld, ir1_end = %6lld\n", ir0_start, ir0_end, ir1_start, ir1_end);

    // threads with no work simply yield (not sure if it helps)
    यदि (ir0_start >= ir0_अन्तः || ir1_start >= ir1_अन्तः) {
        प्रत्यागम;
    }

    ध्रुव शून्य * wdata = (src1->type == vec_dot_type) ? src1->दत्त : params->wdata;
    ध्रुव size_t row_आकार = ggml_row_आकार(vec_dot_type, ne10);

    प्रतिज्ञा(ne12 % ne02 == 0);
    प्रतिज्ञा(ne13 % ne03 == 0);

    // block-tiling attempt
    ध्रुव int64_t blck_0 = 16;
    ध्रुव int64_t blck_1 = 16;

    ध्रुव size_t src1_col_stride = src1_cont || src1->type != vec_dot_type ? row_आकार : nb11;

    // attempt to reduce false-sharing (does not seem to make a difference)
    // 16 * 2, accounting for mmla kernels
    भिन्न tmp[32];

    प्रति (int64_t iir1 = ir1_start; iir1 < ir1_अन्तः; iir1 += blck_1) {
        प्रति (int64_t iir0 = ir0_start; iir0 < ir0_अन्तः; iir0 += blck_0) {
            प्रति (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_अन्तः; ir1 += अङ्कः_पङ्क्तिकाः_प्रत्येकम्_vec_dot) {
                ध्रुव int64_t i13 = (ir1 / (ne12 * ne1));
                ध्रुव int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                ध्रुव int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                // broadcast src0 into src1
                ध्रुव int64_t i03 = i13 / r3;
                ध्रुव int64_t i02 = i12 / r2;

                ध्रुव int64_t i1 = i11;
                ध्रुव int64_t i2 = i12;
                ध्रुव int64_t i3 = i13;

                ध्रुव वर्ण * src0_row = (ध्रुव वर्ण*)src0->दत्त + (0 + i02 * nb02 + i03 * nb03);

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                ध्रुव वर्ण * src1_col = (ध्रुव वर्ण*)wdata +
                    (src1_cont || src1->type != vec_dot_type
                        ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_आकार
                        : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                भिन्न * dst_col = (भिन्न*)((वर्ण*)dst->दत्त + (i1 * nb1 + i2 * nb2 + i3 * nb3));

                //for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                //    vec_dot(ne00, &dst_col[ir0], src0_row + ir0*nb01, src1_col);
                //}

                प्रति (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_अन्तः; ir0 += अङ्कः_पङ्क्तिकाः_प्रत्येकम्_vec_dot) {
                    vec_dot(ne00, &tmp[ir0 - iir0], (अङ्कः_पङ्क्तिकाः_प्रत्येकम्_vec_dot > 1 ? 16 : 0), src0_row + ir0 * nb01, (अङ्कः_पङ्क्तिकाः_प्रत्येकम्_vec_dot > 1 ? nb01 : 0), src1_col, (अङ्कः_पङ्क्तिकाः_प्रत्येकम्_vec_dot > 1 ? src1_col_stride : 0), अङ्कः_पङ्क्तिकाः_प्रत्येकम्_vec_dot);
                }

                प्रति (पूर्णाङ्क cn = 0; cn < अङ्कः_पङ्क्तिकाः_प्रत्येकम्_vec_dot; ++cn) {
                    स्मृतिं_प्रतिलिख(&dst_col[iir0 + cn * nb1 / nb0], tmp + (cn * 16), (MIN(iir0 + blck_0, ir0_अन्तः) - iir0) * परिमाण(भिन्न));
                }
            }
        }
    }
}

शून्य ggml_compute_forward_mul_mat(
        ध्रुव संरचना ggml_compute_params * params,
              संरचना ggml_tensor * dst) {

    ध्रुव संरचना ggml_tensor * src0 = dst->src[0];
    ध्रुव संरचना ggml_tensor * src1 = dst->src[1];

    ध्रुव int32_t hint = ggml_get_op_params_i32(dst, 1);
    यदि (hint == GGML_HINT_SRC0_IS_HADAMARD && !params->use_ref) {
        ggml_compute_forward_fwht(params, dst);
        प्रत्यागम;
    }

    // If tiled is supported, it will execute the full op here and we return
    यदि (ggml_compute_forward_mul_mat_tiled(params, dst)) {
        प्रत्यागम;
    }

    GGML_TENSOR_द्विमान_OP_LOCALS

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    गणना ggml_type           ध्रुव vec_dot_type         = type_traits_cpu[src0->type].vec_dot_type;
    ggml_from_float_t        ध्रुव from_float           = type_traits_cpu[vec_dot_type].from_float;
    int64_t                  ध्रुव vec_dot_अङ्कः_पङ्क्तिकाः     = type_traits_cpu[src0->type].nrows;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_आकार(src0->type));
    GGML_ASSERT(nb10 == ggml_type_आकार(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == परिमाण(भिन्न));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    // TODO: extract to "extra_op"
#यदि GGML_USE_LLAMAFILE
    // broadcast factors
    ध्रुव int64_t r2 = ne12 / ne02;
    ध्रुव int64_t r3 = ne13 / ne03;

    ध्रुव bool src1_cont = ggml_is_contiguous(src1);

    यदि (!params->use_ref && src1_cont) {
        प्रति (int64_t i13 = 0; i13 < ne13; i13++)
            प्रति (int64_t i12 = 0; i12 < ne12; i12++)
                यदि (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_आकार(src0->type),
                                     (ध्रुव वर्ण *)src0->दत्त + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_आकार(src0->type),
                                     (ध्रुव वर्ण *)src1->दत्त + i12*nb12 + i13*nb13,
                                     nb11/ggml_type_आकार(src1->type),
                                     (वर्ण *)dst->दत्त + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_आकार(dst->type),
                                     src0->type,
                                     src1->type,
                                     dst->type))
                    गच्छ UseGgmlGemm1;
        प्रत्यागम;
    }
UseGgmlGemm1:;
#यदि_अन्तः

    यदि (src1->type != vec_dot_type) {
        वर्ण * wdata = params->wdata;

        ध्रुव size_t nbw0 = ggml_type_आकार(vec_dot_type);
        ध्रुव size_t nbw1 = ggml_row_आकार(vec_dot_type, ne10);
        ध्रुव size_t nbw2 = nbw1*ne11;
        ध्रुव size_t nbw3 = nbw2*ne12;

        प्रतिज्ञा(params->wsize >= ne13*nbw3);
        // src1 is either packed from F32 into vec_dot_type, or widened from F16 or BF16 into the F32 work buffer
        ध्रुव bool widen = src1->type != GGML_TYPE_F32;

        GGML_ASSERT(!widen || vec_dot_type == GGML_TYPE_F32);
        GGML_ASSERT(!widen || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_BF16);

    #यदि 0
        प्रति (int64_t i13 = 0; i13 < ne13; ++i13) {
            प्रति (int64_t i12 = 0; i12 < ne12; ++i12) {
                प्रति (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                    from_float((भिन्न *)((वर्ण *) src1->दत्त + i13*nb13 + i12*nb12 + i11*nb11),
                               (शून्य *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                                ne10);
                }
            }
        }
    #अन्यथा
        ध्रुव int64_t bs = ggml_blck_आकार(vec_dot_type);

        प्रति (int64_t i13 = 0; i13 < ne13; ++i13) {
            प्रति (int64_t i12 = 0; i12 < ne12; ++i12) {
                प्रति (int64_t i11 = 0; i11 < ne11; ++i11) {
                    int64_t ne10_खण्डम्_start = (ith * ne10/bs) / nth;
                    int64_t ne10_खण्डम्_अन्तः   = ((ith + 1) * ne10/bs) / nth;
                    ध्रुव शून्य * src1_row  = (ध्रुव वर्ण *) src1->दत्त + i13*nb13 + i12*nb12 + i11*nb11 + ne10_खण्डम्_start*bs*nb10;
                    शून्य       * wdata_row =                    wdata  + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_खण्डम्_start*nbw0;

                    ध्रुव int64_t ne10_खण्डम्_आकार = (ne10_खण्डम्_अन्तः - ne10_खण्डम्_start) * bs;

                    यदि (src1->type == GGML_TYPE_F16) {
                        ggml_cpu_fp16_गन्तव्यम्_fp32((ध्रुव ggml_fp16_t *) src1_row, (भिन्न *) wdata_row, ne10_खण्डम्_आकार);
                    } अन्यथा यदि (src1->type == GGML_TYPE_BF16) {
                        ggml_cpu_bf16_गन्तव्यम्_fp32((ध्रुव ggml_bf16_t *) src1_row, (भिन्न *) wdata_row, ne10_खण्डम्_आकार);
                    } अन्यथा {
                        from_float((ध्रुव भिन्न *) src1_row, wdata_row, ne10_खण्डम्_आकार);
                    }
                }
            }
        }
    #यदि_अन्तः
    }

    यदि (ith == 0) {
        // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
        atomic_store_explicit(&params->threadpool->current_chunk, nth, memory_order_relaxed);
    }

    ggml_barrier(params->threadpool);

#यदि GGML_USE_LLAMAFILE
    यदि (!params->use_ref && src1->type != vec_dot_type) {
        ध्रुव शून्य* wdata = (src1->type == vec_dot_type) ? src1->दत्त : params->wdata;
        ध्रुव size_t row_आकार = ggml_row_आकार(vec_dot_type, ne10);

        प्रति (int64_t i13 = 0; i13 < ne13; i13++)
            प्रति (int64_t i12 = 0; i12 < ne12; i12++)
                यदि (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_आकार(src0->type),
                                     (ध्रुव वर्ण *)src0->दत्त + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_आकार(src0->type),
                                     (ध्रुव वर्ण *)wdata + (i12*ne11 + i13*ne12*ne11)*row_आकार,
                                     row_आकार/ggml_type_आकार(vec_dot_type),
                                     (वर्ण *)dst->दत्त + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_आकार(dst->type),
                                     src0->type,
                                     vec_dot_type,
                                     dst->type))
                    गच्छ UseGgmlGemm2;
        प्रत्यागम;
    }
UseGgmlGemm2:;
#यदि_अन्तः

    // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
    ध्रुव int64_t nr0 = ne0;

    // This is the size of the rest of the dimensions of the result
    ध्रुव int64_t nr1 = ne1 * ne2 * ne3;

    // Now select a reasonable chunk size.
    पूर्णाङ्क chunk_आकार = 16;

    // We need to step up the size if it's small
    यदि (nr0 == 1 || nr1 == 1) {
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
    यदि (nchunk0 * nchunk1 < nth * 4 || ggml_is_numa()) {
        // distribute the thread work across the inner or outer loop based on which one is larger
        nchunk0 = nr0 > nr1 ? nth : 1; // parallelize by src0 rows
        nchunk1 = nr0 > nr1 ? 1 : nth; // parallelize by src1 rows
    }

    // The number of elements in each chunk
    ध्रुव int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    ध्रुव int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    // The first chunk comes from our thread_id, the rest will get auto-assigned.
    पूर्णाङ्क current_chunk = ith;

    यावत् (current_chunk < nchunk0 * nchunk1) {
        ध्रुव int64_t ith0 = current_chunk % nchunk0;
        ध्रुव int64_t ith1 = current_chunk / nchunk0;

        ध्रुव int64_t ir0_start = dr0 * ith0;
        ध्रुव int64_t ir0_अन्तः = MIN(ir0_start + dr0, nr0);

        ध्रुव int64_t ir1_start = dr1 * ith1;
        ध्रुव int64_t ir1_अन्तः = MIN(ir1_start + dr1, nr1);

        // dot kernels can handle 1 row and col at a time, but mmla kernels can process 2 rows and cols
        int64_t अङ्कः_पङ्क्तिकाः_प्रत्येकम्_vec_dot = vec_dot_अङ्कः_पङ्क्तिकाः;

        // these checks are needed to avoid crossing dim1 boundaries
        // can be optimized, but the logic would become more complicated, so keeping it like this for simplicity
        यदि ((nr0 % 2 != 0) || (ne11 % 2 != 0) || ((ir0_अन्तः - ir0_start) % 2 != 0) || ((ir1_अन्तः - ir1_start) % 2 != 0)) {
            अङ्कः_पङ्क्तिकाः_प्रत्येकम्_vec_dot = 1;
        }
        ggml_compute_forward_mul_mat_one_chunk(params, dst, src0->type, अङ्कः_पङ्क्तिकाः_प्रत्येकम्_vec_dot, ir0_start, ir0_अन्तः, ir1_start, ir1_अन्तः);

        यदि (nth >= nchunk0 * nchunk1) {
            भङ्ग;
        }

        current_chunk = atomic_fetch_सङ्कलय_explicit(&params->threadpool->current_chunk, 1, memory_order_relaxed);
    }
}

// ggml_compute_forward_mul_mat_id

#परिभाषय MMID_MATRIX_ROW(row_id, i1) matrix_पङ्क्तिकाः[(row_id)*ids->ne[0]*ids->ne[1] + (i1)]

संरचना mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

स्थिर शून्य ggml_compute_forward_mul_mat_id_one_chunk(
    संरचना ggml_tensor * dst,
    ध्रुव संरचना ggml_tensor * src0,
    ध्रुव संरचना ggml_tensor * src1,
    ध्रुव संरचना ggml_tensor * ids,
    ध्रुव int64_t cur_a,
    ध्रुव int64_t ir0_start,
    ध्रुव int64_t ir0_अन्तः,
    ध्रुव int64_t ir1_start,
    ध्रुव int64_t ir1_अन्तः,
    ध्रुव वर्ण * src0_cur,
    ध्रुव संरचना mmid_row_mapping * matrix_पङ्क्तिकाः,
    ध्रुव size_t row_आकार,
    ध्रुव bool src1_cont,
    ध्रुव शून्य * wdata) {

    GGML_TENSOR_द्विमान_OP_LOCALS

    ध्रुव गणना ggml_type type = src0->type;

    ggml_vec_dot_t    ध्रुव vec_dot      = type_traits_cpu[type].vec_dot;
    गणना ggml_type    ध्रुव vec_dot_type = type_traits_cpu[type].vec_dot_type;

    ध्रुव int64_t blck_0 = 16;
    ध्रुव int64_t blck_1 = 16;

    भिन्न tmp[16];

    प्रति (int64_t iir1 = ir1_start; iir1 < ir1_अन्तः; iir1 += blck_1) {
        प्रति (int64_t iir0 = ir0_start; iir0 < ir0_अन्तः; iir0 += blck_0) {
            प्रति (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_अन्तः; ++ir1) {
                ध्रुव int64_t _i12 = ir1; // logical row index for this expert

                संरचना mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, _i12);
                ध्रुव पूर्णाङ्क id       = row_mapping.i1; // selected expert index

                ध्रुव int64_t  i11 = id % ne11;
                ध्रुव int64_t  i12 = row_mapping.i2; // row index in src1

                ध्रुव int64_t  i1 = id;  // selected expert index
                ध्रुव int64_t  i2 = i12; // row

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                ध्रुव वर्ण * src1_col = (ध्रुव वर्ण *) wdata +
                    (src1_cont || src1->type != vec_dot_type
                    ? (i11      + i12*ne11)*row_आकार
                    : (i11*nb11 + i12*nb12));

                भिन्न * dst_col = (भिन्न *) ((वर्ण *) dst->दत्त + (i1*nb1 + i2*nb2));

                प्रति (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_अन्तः; ++ir0) {
                    vec_dot(ne00, &tmp[ir0 - iir0], 0, src0_cur + ir0*nb01, 0, src1_col, 0, 1);
                }

                स्मृतिं_प्रतिलिख(&dst_col[iir0], tmp, (MIN(iir0 + blck_0, ir0_अन्तः) - iir0)*परिमाण(भिन्न));
            }
        }
    }
}

स्थिर शून्य * incr_ptr_aligned(शून्य ** p, size_t आकार, size_t align) {

    शून्य * ptr = *p;
    ptr = (शून्य *) GGML_PAD((uintptr_t) ptr, align);
    *p = (शून्य *) ((वर्ण *) ptr + आकार);
    प्रत्यागम ptr;
}

स्थिर शून्य ggml_compute_forward_mul_mat_id(
        ध्रुव संरचना ggml_compute_params * params,
              संरचना ggml_tensor * dst) {

    ध्रुव संरचना ggml_tensor * src0 = dst->src[0];
    ध्रुव संरचना ggml_tensor * src1 = dst->src[1];
    ध्रुव संरचना ggml_tensor * ids = dst->src[2];

    GGML_TENSOR_द्विमान_OP_LOCALS

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव गणना ggml_type type = src0->type;

    ध्रुव bool src1_cont = ggml_is_contiguous(src1);

    गणना ggml_type    ध्रुव vec_dot_type    = type_traits_cpu[type].vec_dot_type;
    ggml_from_float_t ध्रुव from_float      = type_traits_cpu[vec_dot_type].from_float;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_आकार(type));
    GGML_ASSERT(nb10 == ggml_type_आकार(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == परिमाण(भिन्न));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // row groups
    ध्रुव पूर्णाङ्क n_ids = ids->ne[0]; // n_expert_used
    ध्रुव पूर्णाङ्क n_as  = ne02;       // n_expert

    शून्य * wdata_cur = params->wdata;

    यदि (src1->type != vec_dot_type) {
        incr_ptr_aligned(&wdata_cur, ggml_row_आकार(vec_dot_type, ggml_nelements(src1)), परिमाण(int64_t));
    }

    int64_t * matrix_row_counts = // [n_as]
        incr_ptr_aligned(&wdata_cur, n_as*परिमाण(int64_t), परिमाण(int64_t));

    संरचना mmid_row_mapping * matrix_पङ्क्तिकाः = // [n_as][ids->ne[0]*ids->ne[1]]
        incr_ptr_aligned(&wdata_cur, n_as*ids->ne[0]*ids->ne[1]*परिमाण(संरचना mmid_row_mapping), परिमाण(int64_t));

    वर्ण (*atomic_current_chunk)[CACHE_पङ्क्तिः_आकार] = // [n_as]
        incr_ptr_aligned(&wdata_cur, CACHE_पङ्क्तिः_आकार * n_as, CACHE_पङ्क्तिः_आकार);

    // Tiled matmul (see tiled.h); per-thread work buffers, 0 bytes when disabled. The
    // reservation is unconditional, the per expert eligibility is decided at dispatch time
    वर्ण * tiled_scratch = incr_ptr_aligned(&wdata_cur, ggml_tiled_wdata_आकार(nth, dst), 64);

    GGML_ASSERT(params->wsize >= (size_t)((वर्ण *) wdata_cur - (वर्ण *) params->wdata));

    यदि (src1->type != vec_dot_type) {
        वर्ण * wdata = params->wdata;

        ध्रुव size_t nbw0 = ggml_type_आकार(vec_dot_type);
        ध्रुव size_t nbw1 = ggml_row_आकार(vec_dot_type, ne10);
        ध्रुव size_t nbw2 = nbw1*ne11;
        ध्रुव size_t nbw3 = nbw2*ne12;

        प्रतिज्ञा(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

#यदि 0
        प्रति (int64_t i13 = 0; i13 < ne13; ++i13) {
            प्रति (int64_t i12 = ith; i12 < ne12; i12 += nth) {
                प्रति (int64_t i11 = 0; i11 < ne11; ++i11) {
                    from_float((भिन्न *)((वर्ण *) src1->दत्त + i13*nb13 + i12*nb12 + i11*nb11),
                               (शून्य *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                               ne10);
                }
            }
        }
#अन्यथा
        प्रति (int64_t i13 = 0; i13 < ne13; ++i13) {
            प्रति (int64_t i12 = 0; i12 < ne12; ++i12) {
                प्रति (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = ggml_blck_आकार(vec_dot_type);
                    int64_t ne10_खण्डम्_start = (ith * ne10/bs) / nth;
                    int64_t ne10_खण्डम्_अन्तः   = ((ith + 1) * ne10/bs) / nth;
                    from_float((भिन्न *)((वर्ण *) src1->दत्त + i13*nb13 + i12*nb12 + i11*nb11 + ne10_खण्डम्_start*bs*nb10),
                               (शून्य *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_खण्डम्_start*nbw0),
                               (ne10_खण्डम्_अन्तः - ne10_खण्डम्_start) * bs);
                }
            }
        }
#यदि_अन्तः
    }

    यदि (ith == 0) {
        // initialize matrix_row_counts
        स्मृतिं_पूरय(matrix_row_counts, 0, n_as*परिमाण(int64_t));

        // group rows by src0 matrix
        प्रति (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            प्रति (पूर्णाङ्क id = 0; id < n_ids; ++id) {
                ध्रुव int32_t i02 = *(ध्रुव int32_t *) ((ध्रुव वर्ण *) ids->दत्त + iid1*ids->nb[1] + id*ids->nb[0]);

                प्रतिज्ञा(i02 >= 0 && i02 < n_as);

                MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = (संरचना mmid_row_mapping) {id, iid1};
                matrix_row_counts[i02] += 1;
            }
        }
    }

    // reset current_chunk
    प्रति (पूर्णाङ्क cur_a = ith; cur_a < n_as; cur_a += nth) {
        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);
        *current_chunk_ctr = nth;
    }

    ggml_barrier(params->threadpool);

    प्रति (पूर्णाङ्क cur_a = 0; cur_a < n_as; ++cur_a) {
        ध्रुव int64_t cne1 = matrix_row_counts[cur_a];

        यदि (cne1 == 0) {
            अनुवर्त;
        }

        // tiled takes over if profitable for this expert (see tiled.h)
        यदि (ggml_compute_forward_mul_mat_id_tiled(params, dst, cur_a, cne1, (ध्रुव int32_t *) &MMID_MATRIX_ROW(cur_a, 0), tiled_scratch)) {
            अनुवर्त;
        }

        ध्रुव वर्ण * src0_cur = (ध्रुव वर्ण *) src0->दत्त + cur_a * nb02;
        ध्रुव शून्य * wdata = (src1->type == vec_dot_type) ? src1->दत्त : params->wdata;
        ध्रुव size_t row_आकार = ggml_row_आकार(vec_dot_type, ne10);

        ध्रुव int64_t nr0 = ne01;
        ध्रुव int64_t nr1 = cne1;

        पूर्णाङ्क chunk_आकार = 16;
        यदि (nr0 == 1 || nr1 == 1) {
            chunk_आकार = 64;
        }

        // disable for NUMA
        ध्रुव bool disable_chunking = ggml_is_numa();

        int64_t nchunk0 = (nr0 + chunk_आकार - 1) / chunk_आकार;
        int64_t nchunk1 = (nr1 + chunk_आकार - 1) / chunk_आकार;

        यदि (nchunk0 * nchunk1 < nth * 4 || disable_chunking) {
            nchunk0 = nr0 > nr1 ? nth : 1;
            nchunk1 = nr0 > nr1 ? 1 : nth;
        }

        ध्रुव int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        ध्रुव int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

        पूर्णाङ्क current_chunk = ith;

        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);

        यावत् (current_chunk < nchunk0 * nchunk1) {
            ध्रुव int64_t ith0 = current_chunk % nchunk0;
            ध्रुव int64_t ith1 = current_chunk / nchunk0;

            ध्रुव int64_t ir0_start = dr0 * ith0;
            ध्रुव int64_t ir0_अन्तः = MIN(ir0_start + dr0, nr0);

            ध्रुव int64_t ir1_start = dr1 * ith1;
            ध्रुव int64_t ir1_अन्तः = MIN(ir1_start + dr1, nr1);

            ggml_compute_forward_mul_mat_id_one_chunk(
                dst, src0, src1, ids, cur_a,
                ir0_start, ir0_अन्तः, ir1_start, ir1_अन्तः,
                src0_cur, matrix_पङ्क्तिकाः, row_आकार, src1_cont, wdata
            );

            यदि (nth >= nchunk0 * nchunk1) {
                भङ्ग;
            }

            current_chunk = atomic_fetch_सङ्कलय_explicit(current_chunk_ctr, 1, memory_order_relaxed);
        }
    }
}

/////////////////////////////////

स्थिर शून्य ggml_compute_forward(संरचना ggml_compute_params * params, संरचना ggml_tensor * tensor) {
    GGML_ASSERT(params);

    यदि (tensor->op == GGML_OP_NONE || ggml_is_रिक्तानि(tensor)) {
        प्रत्यागम;
    }

    // extra_buffer op?
    यदि (ggml_cpu_extra_compute_forward(params, tensor)) {
        प्रत्यागम;
    }

    विकल्प (tensor->op) {
        पक्ष GGML_OP_DUP:
            {
                ggml_compute_forward_dup(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_सङ्कलय:
            {
                ggml_compute_forward_सङ्कलय(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_सङ्कलय_ID:
            {
                ggml_compute_forward_सङ्कलय_id(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_ADD1:
            {
                ggml_compute_forward_add1(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_ACC:
            {
                ggml_compute_forward_acc(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_SUB:
            {
                ggml_compute_forward_sub(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_MUL:
            {
                ggml_compute_forward_mul(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_DIV:
            {
                ggml_compute_forward_div(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_SQR:
            {
                ggml_compute_forward_sqr(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_SQRT:
            {
                ggml_compute_forward_sqrt(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_LOG:
            {
                ggml_compute_forward_log(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_SIN:
            {
                ggml_compute_forward_sin(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_COS:
            {
                ggml_compute_forward_cos(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_योग:
            {
                ggml_compute_forward_योग(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_योग_पङ्क्तिकाः:
            {
                ggml_compute_forward_योग_पङ्क्तिकाः(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_CUMSUM:
            {
                ggml_compute_forward_cumsum(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_माध्यम्:
            {
                ggml_compute_forward_माध्यम्(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_ARGMAX:
            {
                ggml_compute_forward_argmax(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_गणनम्_EQUAL:
            {
                ggml_compute_forward_गणनम्_equal(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_REPEAT:
            {
                ggml_compute_forward_repeat(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_REPEAT_पृष्ठभागः:
            {
                ggml_compute_forward_repeat_पृष्ठभागः(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_CONCAT:
            {
                ggml_compute_forward_concat(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_SILU_पृष्ठभागः:
            {
                ggml_compute_forward_silu_पृष्ठभागः(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_NORM:
            {
                ggml_compute_forward_norm(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_RMS_NORM:
            {
                ggml_compute_forward_rms_norm(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_RMS_NORM_पृष्ठभागः:
            {
                ggml_compute_forward_rms_norm_पृष्ठभागः(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_गणः_NORM:
            {
                ggml_compute_forward_गणः_norm(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_L2_NORM:
            {
                ggml_compute_forward_l2_norm(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_MUL_MAT:
            {
                ggml_compute_forward_mul_mat(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_MUL_MAT_ID:
            {
                ggml_compute_forward_mul_mat_id(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_बहिर्गमनम्_PROD:
            {
                ggml_compute_forward_बहिर्गमनम्_prod(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_SCALE:
            {
                ggml_compute_forward_scale(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_समुच्चयः:
            {
                ggml_compute_forward_समुच्चयः(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_CPY:
            {
                ggml_compute_forward_cpy(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_CONT:
            {
                ggml_compute_forward_cont(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_GET_पङ्क्तिकाः:
            {
                ggml_compute_forward_get_पङ्क्तिकाः(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_GET_पङ्क्तिकाः_पृष्ठभागः:
            {
                ggml_compute_forward_get_पङ्क्तिकाः_पृष्ठभागः(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_समुच्चयः_पङ्क्तिकाः:
            {
                ggml_compute_forward_समुच्चयः_पङ्क्तिकाः(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_DIAG:
            {
                ggml_compute_forward_diag(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_DIAG_आवरणम्_INF:
            {
                ggml_compute_forward_diag_आवरणम्_inf(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_DIAG_आवरणम्_ZERO:
            {
                ggml_compute_forward_diag_आवरणम्_zero(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_SOFT_MAX:
            {
                ggml_compute_forward_soft_max(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_SOFT_MAX_पृष्ठभागः:
            {
                ggml_compute_forward_soft_max_ext_पृष्ठभागः(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_ROPE:
            {
                ggml_compute_forward_rope(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_ROPE_पृष्ठभागः:
            {
                ggml_compute_forward_rope_पृष्ठभागः(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_CLAMP:
            {
                ggml_compute_forward_clamp(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_compute_forward_conv_transpose_1d(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_IM2COL:
            {
                ggml_compute_forward_im2col(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_IM2_COL_पृष्ठभागः:
            {
                ggml_compute_forward_im2col_पृष्ठभागः_f32(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_IM2COL_3D:
            {
                ggml_compute_forward_im2col_3d(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_COL2IM_1D:
            {
                ggml_compute_forward_col2im_1d(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_CONV_2D:
            {
                ggml_compute_forward_conv_2d(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_CONV_3D:
            {
                ggml_compute_forward_conv_3d(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_CONV_2D_DW:
            {
                ggml_compute_forward_conv_2d_dw(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_CONV_TRANSPOSE_2D:
            {
                ggml_compute_forward_conv_transpose_2d(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_POOL_1D:
            {
                ggml_compute_forward_pool_1d(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_POOL_2D:
            {
                ggml_compute_forward_pool_2d(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_POOL_2_D_पृष्ठभागः:
            {
                ggml_compute_forward_pool_2d_पृष्ठभागः(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_UPSCALE:
            {
                ggml_compute_forward_upscale(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_PAD:
            {
                ggml_compute_forward_pad(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_PAD_REFLECT_1D:
            {
                ggml_compute_forward_pad_reflect_1d(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_ROLL:
            {
                ggml_compute_forward_roll(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_ARANGE:
            {
                ggml_compute_forward_arange(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_TIMESTEP_EMBEDDING:
            {
                ggml_compute_forward_timestep_embedding(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_ARGSORT:
            {
                ggml_compute_forward_argsort(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_शिखरम्_K:
            {
                ggml_compute_forward_शिखरम्_k(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_LEAKY_RELU:
            {
                ggml_compute_forward_leaky_relu(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_TRI:
            {
                ggml_compute_forward_tri(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_FILL:
            {
                ggml_compute_forward_fill(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_FLASH_ATTN_EXT:
            {
                ggml_compute_forward_flash_attn_ext(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_FLASH_ATTN_पृष्ठभागः:
            {
                int32_t t = ggml_get_op_params_i32(tensor, 0);
                GGML_ASSERT(t == 0 || t == 1);
                bool masked = t != 0;
                ggml_compute_forward_flash_attn_पृष्ठभागः(params, masked, tensor);
            } भङ्ग;
        पक्ष GGML_OP_SSM_CONV:
            {
                ggml_compute_forward_ssm_conv(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_SSM_अवलोकनम्:
            {
                ggml_compute_forward_ssm_अवलोकनम्(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_WIN_PART:
            {
                ggml_compute_forward_win_part(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_WIN_UNPART:
            {
                ggml_compute_forward_win_unpart(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_UNARY:
            {
                ggml_compute_forward_unary(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_GLU:
            {
                ggml_compute_forward_glu(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_GET_REL_POS:
            {
                ggml_compute_forward_get_rel_pos(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_सङ्कलय_REL_POS:
            {
                ggml_compute_forward_सङ्कलय_rel_pos(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_RWKV_WKV6:
            {
                ggml_compute_forward_rwkv_wkv6(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_GATED_रैखिकम्_ATTN:
            {
                ggml_compute_forward_gla(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_RWKV_WKV7:
            {
                ggml_compute_forward_rwkv_wkv7(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_SOLVE_TRI:
            {
                ggml_compute_forward_solve_tri(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_GATED_DELTA_NET:
            {
                ggml_compute_forward_gated_delta_net(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_LIGHTNING_INDEXER:
            {
                ggml_compute_forward_lightning_indexer(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_DSV4_HC_COMB:
            {
                ggml_compute_forward_dsv4_hc_comb(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_DSV4_HC_PRE:
            {
                ggml_compute_forward_dsv4_hc_pre(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_DSV4_HC_POST:
            {
                ggml_compute_forward_dsv4_hc_post(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_शिखरम्_K_आवरणम्:
            {
                ggml_compute_forward_शिखरम्_k_आवरणम्(params, tensor);
            } भङ्ग;
        पक्ष GGML_OP_MAP_CUSTOM1:
            {
                ggml_compute_forward_map_custom1(params, tensor);
            }
            भङ्ग;
        पक्ष GGML_OP_MAP_CUSTOM2:
            {
                ggml_compute_forward_map_custom2(params, tensor);
            }
            भङ्ग;
        पक्ष GGML_OP_MAP_CUSTOM3:
            {
                ggml_compute_forward_map_custom3(params, tensor);
            }
            भङ्ग;
        पक्ष GGML_OP_CUSTOM:
            {
                ggml_compute_forward_custom(params, tensor);
            }
            भङ्ग;
        पक्ष GGML_OP_CROSS_ENTROPY_LOSS:
            {
                ggml_compute_forward_cross_entropy_loss(params, tensor);
            }
            भङ्ग;
        पक्ष GGML_OP_CROSS_ENTROPY_LOSS_पृष्ठभागः:
            {
                ggml_compute_forward_cross_entropy_loss_पृष्ठभागः(params, tensor);
            }
            भङ्ग;
        पक्ष GGML_OP_OPT_चरणः_ADAMW:
            {
                ggml_compute_forward_opt_चरणः_adamw(params, tensor);
            }
            भङ्ग;
        पक्ष GGML_OP_OPT_चरणः_SGD:
            {
                ggml_compute_forward_opt_चरणः_sgd(params, tensor);
            }
            भङ्ग;
        पक्ष GGML_OP_NONE:
            {
                // nop
            } भङ्ग;
        पक्ष GGML_OP_RESHAPE:
            {
                // nop
            } भङ्ग;
        पक्ष GGML_OP_PERMUTE:
            {
                // nop
            } भङ्ग;
        पक्ष GGML_OP_VIEW:
            {
                // nop
            } भङ्ग;
        पक्ष GGML_OP_TRANSPOSE:
            {
                // nop
            } भङ्ग;
        पक्ष GGML_OP_गणनम्:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// Android's libc implementation "bionic" does not support setting affinity
#यदि defined(__gnu_linux__)
स्थिर शून्य समुच्चयः_numa_thread_affinity(पूर्णाङ्क thread_n) {
    यदि (!ggml_is_numa()) {
        प्रत्यागम;
    }

    पूर्णाङ्क ग्रन्थिः_अङ्कः;
    पूर्णाङ्क rv;
    size_t setsize = CPU_ALLOC_आकार(g_state.numa.योग_cpus);

    विकल्प(g_state.numa.numa_strategy) {
        पक्ष GGML_NUMA_STRATEGY_DISTRIBUTE:
            // run thread on node_num thread_n / (threads per node)
            ग्रन्थिः_अङ्कः = thread_n % g_state.numa.n_nodes;
            भङ्ग;
        पक्ष GGML_NUMA_STRATEGY_ISOLATE:
            // run thread on current_node
            ग्रन्थिः_अङ्कः = g_state.numa.current_ग्रन्थिः;
            भङ्ग;
        पक्ष GGML_NUMA_STRATEGY_NUMACTL:
            // use the cpuset that numactl gave us
            rv = pthread_setaffinity_np(pthread_self(), setsize, &g_state.numa.cpuset);
            यदि (rv) {
                fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n",strerror(rv));
            }
            प्रत्यागम;
        स्वभाव:
            प्रत्यागम;
    }

    संरचना ggml_numa_ग्रन्थिः * ग्रन्थिः = &g_state.numa.nodes[ग्रन्थिः_अङ्कः];

    cpu_समुच्चयः_t * cpus = CPU_ALLOC(g_state.numa.योग_cpus);
    CPU_ZERO_S(setsize, cpus);
    प्रति (size_t i = 0; i < ग्रन्थिः->n_cpus; ++i) {
        CPU_समुच्चयः_S(ग्रन्थिः->cpus[i], setsize, cpus);
    }

    rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    यदि (rv) {
            fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}

स्थिर शून्य clear_numa_thread_affinity(शून्य) {
    यदि (!ggml_is_numa()) {
        प्रत्यागम;
    }

    size_t setsize = CPU_ALLOC_आकार(g_state.numa.योग_cpus);

    cpu_समुच्चयः_t * cpus = CPU_ALLOC(g_state.numa.योग_cpus);
    CPU_ZERO_S(setsize, cpus);
    प्रति (अचिह्नित i = 0; i < g_state.numa.योग_cpus; ++i) {
        CPU_समुच्चयः_S(i, setsize, cpus);
    }

    पूर्णाङ्क rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    यदि (rv) {
        fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}
#अन्यथा
// TODO: Windows etc.
// (the linux implementation may also work on BSD, someone should test)
स्थिर शून्य समुच्चयः_numa_thread_affinity(पूर्णाङ्क thread_n) { अप्रयुक्तम्(thread_n);  }
स्थिर शून्य clear_numa_thread_affinity(शून्य) {}
#यदि_अन्तः

स्थिर पूर्णाङ्क ggml_get_n_tasks(संरचना ggml_tensor * ग्रन्थिः, पूर्णाङ्क n_तन्तवः) {
    पूर्णाङ्क n_tasks = 0;

    यदि (ggml_is_रिक्तानि(ग्रन्थिः)) {
        // no need to multi-thread a no-op
        n_tasks = 1;
        प्रत्यागम n_tasks;
    }

    विकल्प (ग्रन्थिः->op) {
        पक्ष GGML_OP_CPY:
        पक्ष GGML_OP_DUP:
        पक्ष GGML_OP_CONT:
        पक्ष GGML_OP_सङ्कलय:
        पक्ष GGML_OP_सङ्कलय_ID:
        पक्ष GGML_OP_ADD1:
        पक्ष GGML_OP_ACC:
        पक्ष GGML_OP_CUMSUM:
        पक्ष GGML_OP_TRI:
        पक्ष GGML_OP_FILL:
            {
                n_tasks = n_तन्तवः;
            } भङ्ग;
        पक्ष GGML_OP_SUB:
        पक्ष GGML_OP_SQR:
        पक्ष GGML_OP_SQRT:
        पक्ष GGML_OP_LOG:
        पक्ष GGML_OP_SIN:
        पक्ष GGML_OP_COS:
        पक्ष GGML_OP_योग:
        पक्ष GGML_OP_योग_पङ्क्तिकाः:
        पक्ष GGML_OP_माध्यम्:
        पक्ष GGML_OP_ARGMAX:
            {
                n_tasks = 1;
            } भङ्ग;
        पक्ष GGML_OP_गणनम्_EQUAL:
        पक्ष GGML_OP_SOLVE_TRI:
        पक्ष GGML_OP_GATED_DELTA_NET:
        पक्ष GGML_OP_DSV4_HC_COMB:
        पक्ष GGML_OP_DSV4_HC_PRE:
        पक्ष GGML_OP_DSV4_HC_POST:
        पक्ष GGML_OP_शिखरम्_K_आवरणम्:
            {
                n_tasks = n_तन्तवः;
            } भङ्ग;
        पक्ष GGML_OP_REPEAT:
        पक्ष GGML_OP_REPEAT_पृष्ठभागः:
        पक्ष GGML_OP_LEAKY_RELU:
            {
                n_tasks = 1;
            } भङ्ग;
        पक्ष GGML_OP_UNARY:
            विकल्प (ggml_get_unary_op(ग्रन्थिः)) {
                पक्ष GGML_UNARY_OP_ABS:
                पक्ष GGML_UNARY_OP_SGN:
                पक्ष GGML_UNARY_OP_NEG:
                पक्ष GGML_UNARY_OP_चरणः:
                पक्ष GGML_UNARY_OP_TANH:
                पक्ष GGML_UNARY_OP_ELU:
                पक्ष GGML_UNARY_OP_RELU:
                पक्ष GGML_UNARY_OP_SIGMOID:
                पक्ष GGML_UNARY_OP_HARDSWISH:
                पक्ष GGML_UNARY_OP_HARDSIGMOID:
                पक्ष GGML_UNARY_OP_EXP:
                पक्ष GGML_UNARY_OP_SOFTPLUS:
                पक्ष GGML_UNARY_OP_EXPM1:
                पक्ष GGML_UNARY_OP_FLOOR:
                पक्ष GGML_UNARY_OP_CEIL:
                पक्ष GGML_UNARY_OP_आवर्तनम्:
                पक्ष GGML_UNARY_OP_TRUNC:
                    {
                        n_tasks = 1;
                    } भङ्ग;

                पक्ष GGML_UNARY_OP_GELU:
                पक्ष GGML_UNARY_OP_GELU_ERF:
                पक्ष GGML_UNARY_OP_GELU_शीघ्र:
                पक्ष GGML_UNARY_OP_SILU:
                पक्ष GGML_UNARY_OP_XIELU:
                    {
                        n_tasks = n_तन्तवः;
                    } भङ्ग;
                स्वभाव:
                    GGML_ABORT("fatal error");
            }
            भङ्ग;
        पक्ष GGML_OP_GLU:
            विकल्प (ggml_get_glu_op(ग्रन्थिः)) {
                पक्ष GGML_GLU_OP_REGLU:
                पक्ष GGML_GLU_OP_GEGLU:
                पक्ष GGML_GLU_OP_SWIGLU:
                पक्ष GGML_GLU_OP_SWIGLU_OAI:
                पक्ष GGML_GLU_OP_GEGLU_ERF:
                पक्ष GGML_GLU_OP_GEGLU_शीघ्र:
                पक्ष GGML_GLU_OP_SWIGLU_CLAMP:
                    {
                        n_tasks = n_तन्तवः;
                    } भङ्ग;
                स्वभाव:
                    GGML_ABORT("fatal error");
            }
            भङ्ग;
        पक्ष GGML_OP_SILU_पृष्ठभागः:
        पक्ष GGML_OP_MUL:
        पक्ष GGML_OP_DIV:
        पक्ष GGML_OP_NORM:
        पक्ष GGML_OP_RMS_NORM:
        पक्ष GGML_OP_RMS_NORM_पृष्ठभागः:
        पक्ष GGML_OP_L2_NORM:
        पक्ष GGML_OP_गणः_NORM:
        पक्ष GGML_OP_CONCAT:
        पक्ष GGML_OP_MUL_MAT:
        पक्ष GGML_OP_MUL_MAT_ID:
        पक्ष GGML_OP_बहिर्गमनम्_PROD:
        पक्ष GGML_OP_समुच्चयः_पङ्क्तिकाः:
            {
                n_tasks = n_तन्तवः;
            } भङ्ग;
        पक्ष GGML_OP_GET_पङ्क्तिकाः:
            {
                // FIXME: get_rows can use additional threads, but the cost of launching additional threads
                // decreases performance with GPU offloading
                //n_tasks = n_threads;
                n_tasks = 1;
            } भङ्ग;
        पक्ष GGML_OP_SCALE:
        पक्ष GGML_OP_समुच्चयः:
        पक्ष GGML_OP_RESHAPE:
        पक्ष GGML_OP_VIEW:
        पक्ष GGML_OP_PERMUTE:
        पक्ष GGML_OP_TRANSPOSE:
        पक्ष GGML_OP_GET_पङ्क्तिकाः_पृष्ठभागः:
        पक्ष GGML_OP_DIAG:
            {
                n_tasks = 1;
            } भङ्ग;
        पक्ष GGML_OP_DIAG_आवरणम्_ZERO:
        पक्ष GGML_OP_DIAG_आवरणम्_INF:
        पक्ष GGML_OP_SOFT_MAX_पृष्ठभागः:
        पक्ष GGML_OP_ROPE:
        पक्ष GGML_OP_ROPE_पृष्ठभागः:
        पक्ष GGML_OP_सङ्कलय_REL_POS:
            {
                n_tasks = n_तन्तवः;
            } भङ्ग;
        पक्ष GGML_OP_CLAMP:
            {
                n_tasks = 1; //TODO
            } भङ्ग;
        पक्ष GGML_OP_SOFT_MAX:
            {
                n_tasks = MIN(n_तन्तवः, ggml_nrows(ग्रन्थिः->src[0]));
            } भङ्ग;
        पक्ष GGML_OP_IM2COL:
        पक्ष GGML_OP_IM2_COL_पृष्ठभागः:
        पक्ष GGML_OP_IM2COL_3D:
        पक्ष GGML_OP_CONV_2D:
        पक्ष GGML_OP_CONV_3D:
        पक्ष GGML_OP_CONV_2D_DW:
        पक्ष GGML_OP_COL2IM_1D:
        पक्ष GGML_OP_CONV_TRANSPOSE_1D:
        पक्ष GGML_OP_CONV_TRANSPOSE_2D:
            {
                n_tasks = n_तन्तवः;
            } भङ्ग;
        पक्ष GGML_OP_POOL_1D:
        पक्ष GGML_OP_POOL_2D:
        पक्ष GGML_OP_POOL_2_D_पृष्ठभागः:
            {
                n_tasks = 1;
            } भङ्ग;
        पक्ष GGML_OP_UPSCALE:
        पक्ष GGML_OP_PAD:
        पक्ष GGML_OP_PAD_REFLECT_1D:
        पक्ष GGML_OP_ROLL:
        पक्ष GGML_OP_ARANGE:
        पक्ष GGML_OP_TIMESTEP_EMBEDDING:
        पक्ष GGML_OP_ARGSORT:
        पक्ष GGML_OP_शिखरम्_K:
        पक्ष GGML_OP_FLASH_ATTN_EXT:
        पक्ष GGML_OP_FLASH_ATTN_पृष्ठभागः:
        पक्ष GGML_OP_SSM_CONV:
        पक्ष GGML_OP_SSM_अवलोकनम्:
        पक्ष GGML_OP_LIGHTNING_INDEXER:
            {
                n_tasks = n_तन्तवः;
            } भङ्ग;
        पक्ष GGML_OP_RWKV_WKV6:
        पक्ष GGML_OP_GATED_रैखिकम्_ATTN:
        पक्ष GGML_OP_RWKV_WKV7:
            {
                ध्रुव int64_t n_heads = ग्रन्थिः->src[1]->ne[1];
                n_tasks = MIN(n_तन्तवः, n_heads);
            } भङ्ग;
        पक्ष GGML_OP_WIN_PART:
        पक्ष GGML_OP_WIN_UNPART:
        पक्ष GGML_OP_GET_REL_POS:
            {
                n_tasks = 1;
            } भङ्ग;
        पक्ष GGML_OP_MAP_CUSTOM1:
            {
                संरचना ggml_map_custom1_op_params p;
                स्मृतिं_प्रतिलिख(&p, ग्रन्थिः->op_params, परिमाण(p));
                यदि (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_तन्तवः;
                } अन्यथा {
                    n_tasks = MIN(p.n_tasks, n_तन्तवः);
                }
            } भङ्ग;
        पक्ष GGML_OP_MAP_CUSTOM2:
            {
                संरचना ggml_map_custom2_op_params p;
                स्मृतिं_प्रतिलिख(&p, ग्रन्थिः->op_params, परिमाण(p));
                यदि (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_तन्तवः;
                } अन्यथा {
                    n_tasks = MIN(p.n_tasks, n_तन्तवः);
                }
            } भङ्ग;
        पक्ष GGML_OP_MAP_CUSTOM3:
            {
                संरचना ggml_map_custom3_op_params p;
                स्मृतिं_प्रतिलिख(&p, ग्रन्थिः->op_params, परिमाण(p));
                यदि (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_तन्तवः;
                } अन्यथा {
                    n_tasks = MIN(p.n_tasks, n_तन्तवः);
                }
            } भङ्ग;
        पक्ष GGML_OP_CUSTOM:
            {
                संरचना ggml_custom_op_params p;
                स्मृतिं_प्रतिलिख(&p, ग्रन्थिः->op_params, परिमाण(p));
                यदि (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_तन्तवः;
                } अन्यथा {
                    n_tasks = MIN(p.n_tasks, n_तन्तवः);
                }
            } भङ्ग;
        पक्ष GGML_OP_CROSS_ENTROPY_LOSS:
        पक्ष GGML_OP_CROSS_ENTROPY_LOSS_पृष्ठभागः:
        पक्ष GGML_OP_OPT_चरणः_ADAMW:
        पक्ष GGML_OP_OPT_चरणः_SGD:
            {
                n_tasks = n_तन्तवः;
            } भङ्ग;
        पक्ष GGML_OP_NONE:
            {
                n_tasks = 1;
            } भङ्ग;
        पक्ष GGML_OP_गणनम्:
            {
                GGML_ABORT("fatal error");
            }
        स्वभाव:
            {
                fprintf(stderr, "%s: op not implemented: ", __func__);
                यदि (ग्रन्थिः->op < GGML_OP_गणनम्) {
                    fprintf(stderr, "%s\n", ggml_op_name(ग्रन्थिः->op));
                } अन्यथा {
                    fprintf(stderr, "%d\n", ग्रन्थिः->op);
                }
                GGML_ABORT("fatal error");
            }
    }

    प्रतिज्ञा(n_tasks > 0);

    प्रत्यागम n_tasks;
}

स्थिर thread_ret_t ggml_graph_compute_secondary_thread(शून्य* दत्त);

#यदि defined(_WIN32)
#अन्तर्भाव "windows.h"

// TODO: support > 64 CPUs
स्थिर bool ggml_thread_apply_affinity(bool * आवरणम्) {
    HANDLE    h = GetCurrentThread();
    uint64_t  bitmask = 0ULL;

    प्रतिज्ञा(GGML_MAX_N_तन्तवः >= 64);

    प्रति (int32_t i = 0; i < 8; i++) {
        int32_t idx = i * 8;
        uint8_t val = 0;
        val |= आवरणम्[idx + 0] << 0;
        val |= आवरणम्[idx + 1] << 1;
        val |= आवरणम्[idx + 2] << 2;
        val |= आवरणम्[idx + 3] << 3;
        val |= आवरणम्[idx + 4] << 4;
        val |= आवरणम्[idx + 5] << 5;
        val |= आवरणम्[idx + 6] << 6;
        val |= आवरणम्[idx + 7] << 7;
        bitmask |= (uint64_t)val << idx;
    }

    प्रति (int32_t i = 64; i < GGML_MAX_N_तन्तवः; i++) {
        यदि (आवरणम्[i]) {
            fprintf(stderr, "warn: setting thread-affinity for > 64 CPUs isn't supported on windows!\n");
            भङ्ग;
        }
    }

    DWORD_PTR m = (DWORD_PTR)bitmask;

    m = समुच्चयः_Thread_Affinity_आवरणम्(h, m);

    प्रत्यागम m != 0;
}

स्थिर bool ggml_thread_apply_प्राथमिकता(int32_t prio) {
    // Note that on Windows the Process Priority Class must be updated in order to set Thread priority.
    // This is up to the applications.
    DWORD p = THREAD_प्राथमिकता_प्रसामान्यम्;
    विकल्प (prio) {
        पक्ष GGML_SCHED_PRIO_नीचः:      p = THREAD_प्राथमिकता_अधः_प्रसामान्यम्;  भङ्ग;
        पक्ष GGML_SCHED_PRIO_प्रसामान्यम्:   p = THREAD_प्राथमिकता_प्रसामान्यम्;        भङ्ग;
        पक्ष GGML_SCHED_PRIO_MEDIUM:   p = THREAD_प्राथमिकता_ABOVE_प्रसामान्यम्;  भङ्ग;
        पक्ष GGML_SCHED_PRIO_उच्चः:     p = THREAD_प्राथमिकता_HIGHEST;       भङ्ग;
        पक्ष GGML_SCHED_PRIO_REALTIME: p = THREAD_प्राथमिकता_TIME_CRITICAL; भङ्ग;
    }

    यदि (prio != GGML_SCHED_PRIO_नीचः) {
        // Tell Windows that this thread should not be throttled (needs its own CPU core).
        // Newer Windows 11 versions aggressively park (offline) CPU cores and often place
        // all our threads onto the first 4 cores which results in terrible performance with
        // n_threads > 4
        #यदि _WIN32_WINNT >= 0x0602
        THREAD_POWER_THROTTLING_STATE t;
        Zero_स्मृतिः(&t, परिमाण(t));
        t.संस्करणम्     = THREAD_POWER_THROTTLING_CURRENT_संस्करणम्;
        t.Control_आवरणम् = THREAD_POWER_THROTTLING_EXECUTION_जवः;
        t.State_आवरणम्   = 0;

        यदि (!समुच्चयः_Thread_Information(GetCurrentThread(), ThreadPowerThrottling, &t, परिमाण(t))) {
            GGML_LOG_DEBUG("failed to disable thread power throttling %d : (%d)\n", prio, (पूर्णाङ्क) Get_चरमः_Error());
            प्रत्यागम false;
        }
        #यदि_अन्तः
    }

    यदि (prio == GGML_SCHED_PRIO_प्रसामान्यम्) {
        // Keep inherited policy/priority
        प्रत्यागम true;
    }

    यदि (!समुच्चयः_Thread_प्राथमिकता(GetCurrentThread(), p)) {
        fprintf(stderr, "warn: failed to set thread priority %d : (%d)\n", prio, (पूर्णाङ्क) Get_चरमः_Error());
        प्रत्यागम false;
    }

    प्रत्यागम true;
}

#नोचेद्यदि defined(__APPLE__)
#अन्तर्भाव <sys/types.h>
#अन्तर्भाव <sys/resource.h>

स्थिर bool ggml_thread_apply_affinity(ध्रुव bool * आवरणम्) {
    // Not supported on Apple platforms
    अप्रयुक्तम्(आवरणम्);
    प्रत्यागम true;
}

स्थिर bool ggml_thread_apply_प्राथमिकता(int32_t prio) {
    संरचना sched_param p;
    int32_t policy = SCHED_OTHER;
    विकल्प (prio) {
        // TODO: there seems to be no way to set lower prio on Apple platforms
        पक्ष GGML_SCHED_PRIO_नीचः:      policy = SCHED_OTHER; p.sched_प्राथमिकता = 0;  भङ्ग;
        पक्ष GGML_SCHED_PRIO_प्रसामान्यम्:   policy = SCHED_OTHER; p.sched_प्राथमिकता = 0;  भङ्ग;
        पक्ष GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_प्राथमिकता = 40; भङ्ग;
        पक्ष GGML_SCHED_PRIO_उच्चः:     policy = SCHED_FIFO;  p.sched_प्राथमिकता = 80; भङ्ग;
        पक्ष GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_प्राथमिकता = 90; भङ्ग;
    }

    यदि (prio == GGML_SCHED_PRIO_प्रसामान्यम्) {
        // Keep inherited policy/priority
        प्रत्यागम true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    यदि (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        प्रत्यागम false;
    }

    प्रत्यागम true;
}

#नोचेद्यदि defined(__linux__)
// TODO: this may not work on BSD, to be verified

स्थिर bool ggml_thread_apply_affinity(ध्रुव bool * आवरणम्) {
    cpu_समुच्चयः_t cpuset;
    पूर्णाङ्क err;

    CPU_ZERO(&cpuset);

    प्रति (uint32_t i = 0; i < GGML_MAX_N_तन्तवः; i++) {
        यदि (आवरणम्[i]) {
            GGML_PRINT_DEBUG("Thread %lx: adding %d to cpuset\n", pthread_self(), i);
            CPU_समुच्चयः(i, &cpuset);
        }
    }

#यदि_परिभाषितम् __ANDROID__
    err = sched_setaffinity(0, परिमाण(cpuset), &cpuset);
    यदि (err < 0) {
        err = errno;
    }
#अन्यथा
    err = pthread_setaffinity_np(pthread_self(), परिमाण(cpuset), &cpuset);
#यदि_अन्तः
    यदि (err != 0) {
        fprintf(stderr, "warn: failed to set affinity mask 0x%llx : %s (%d)\n", (अचिह्नित दीर्घ दीर्घ)आवरणम्, strerror(err), err);
        प्रत्यागम false;
    }

    प्रत्यागम true;
}

स्थिर bool ggml_thread_apply_प्राथमिकता(int32_t prio) {
    संरचना sched_param p;
    int32_t policy = SCHED_OTHER;
    विकल्प (prio) {
        पक्ष GGML_SCHED_PRIO_नीचः:      policy = SCHED_BATCH; p.sched_प्राथमिकता = 0;  भङ्ग;
        पक्ष GGML_SCHED_PRIO_प्रसामान्यम्:   policy = SCHED_OTHER; p.sched_प्राथमिकता = 0;  भङ्ग;
        पक्ष GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_प्राथमिकता = 40; भङ्ग;
        पक्ष GGML_SCHED_PRIO_उच्चः:     policy = SCHED_FIFO;  p.sched_प्राथमिकता = 80; भङ्ग;
        पक्ष GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_प्राथमिकता = 90; भङ्ग;
    }

    यदि (prio == GGML_SCHED_PRIO_प्रसामान्यम्) {
        // Keep inherited policy/priority
        प्रत्यागम true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    यदि (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        प्रत्यागम false;
    }

    प्रत्यागम true;
}

#अन्यथा // unsupported platforms

स्थिर bool ggml_thread_apply_affinity(ध्रुव bool * आवरणम्) {
    अप्रयुक्तम्(आवरणम्);
    प्रत्यागम true;
}

स्थिर bool ggml_thread_apply_प्राथमिकता(int32_t prio) {
    अप्रयुक्तम्(prio);
    प्रत्यागम true;
}

#यदि_अन्तः

स्थिर bool ggml_thread_cpumask_is_valid(ध्रुव bool * आवरणम्) {
    प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_N_तन्तवः; i++) {
        यदि (आवरणम्[i]) { प्रत्यागम true; }
    }
    प्रत्यागम false;
}

स्थिर शून्य ggml_thread_cpumask_अग्र(ध्रुव bool * global_आवरणम्, bool * local_आवरणम्, bool strict, int32_t* iter) {
    यदि (!strict) {
        स्मृतिं_प्रतिलिख(local_आवरणम्, global_आवरणम्, GGML_MAX_N_तन्तवः);
        प्रत्यागम;
    } अन्यथा {
        स्मृतिं_पूरय(local_आवरणम्, 0, GGML_MAX_N_तन्तवः);
        int32_t आधारः_idx = *iter;
        प्रति (int32_t i = 0; i < GGML_MAX_N_तन्तवः; i++) {
            int32_t idx = आधारः_idx + i;
            यदि (idx >= GGML_MAX_N_तन्तवः) {
                // Just a cheaper modulo
                idx -= GGML_MAX_N_तन्तवः;
            }
            यदि (global_आवरणम्[idx]) {
                local_आवरणम्[idx] = 1;
                *iter = idx + 1;
                प्रत्यागम;
            }
        }
    }
}

शून्य ggml_threadpool_free(संरचना ggml_threadpool* threadpool) {
    यदि (!threadpool) प्रत्यागम;

    ध्रुव पूर्णाङ्क n_तन्तवः = threadpool->n_तन्तवः;

#यदि_अपरिभाषितम् GGML_USE_OPENMP
    संरचना ggml_compute_state* workers = threadpool->workers;

    ggml_mutex_तालकम्(&threadpool->mutex);

    threadpool->stop = true;
    threadpool->pause = false;

    ggml_cond_प्रसारणम्(&threadpool->cond);
    ggml_mutex_unlock(&threadpool->mutex);

    प्रति (पूर्णाङ्क j = 1; j < n_तन्तवः; j++) {
        int32_t rc = ggml_thread_join(workers[j].thrd, रिक्त);
        GGML_ASSERT(rc == GGML_EXIT_SUCCESS || rc == GGML_EXIT_ABORTED);
        अप्रयुक्तम्(rc);
    }

    ggml_mutex_destroy(&threadpool->mutex);
    ggml_cond_destroy(&threadpool->cond);
#यदि_अन्तः // GGML_USE_OPENMP

    ध्रुव size_t workers_आकार = परिमाण(संरचना ggml_compute_state) * n_तन्तवः;
    ggml_aligned_free(threadpool->workers, workers_आकार);
    ggml_aligned_free(threadpool, परिमाण(संरचना ggml_threadpool));
}

#यदि_अपरिभाषितम् GGML_USE_OPENMP
// pause/resume must be called under mutex
स्थिर शून्य ggml_threadpool_pause_locked(संरचना ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Pausing threadpool\n");
    threadpool->pause = true;
    ggml_cond_प्रसारणम्(&threadpool->cond);
}

स्थिर शून्य ggml_threadpool_resume_locked(संरचना ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Resuming threadpool\n");
    threadpool->pause = false;
    ggml_cond_प्रसारणम्(&threadpool->cond);
}
#यदि_अन्तः

शून्य ggml_threadpool_pause(संरचना ggml_threadpool * threadpool) {
#यदि_अपरिभाषितम् GGML_USE_OPENMP
    ggml_mutex_तालकम्(&threadpool->mutex);
    यदि (!threadpool->pause) {
       ggml_threadpool_pause_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#अन्यथा
    अप्रयुक्तम्(threadpool);
#यदि_अन्तः
}

शून्य ggml_threadpool_resume(संरचना ggml_threadpool * threadpool) {
#यदि_अपरिभाषितम् GGML_USE_OPENMP
    ggml_mutex_तालकम्(&threadpool->mutex);
    यदि (threadpool->pause) {
       ggml_threadpool_resume_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#अन्यथा
    अप्रयुक्तम्(threadpool);
#यदि_अन्तः
}

संरचना ggml_cplan ggml_graph_plan(
          ध्रुव संरचना ggml_cgraph * cgraph,
                               पूर्णाङ्क   n_तन्तवः,
            संरचना ggml_threadpool * threadpool) {

    यदि (threadpool == रिक्त) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
    }
    यदि (n_तन्तवः <= 0) {
        n_तन्तवः = threadpool ? threadpool->n_तन्तवः : GGML_DEFAULT_N_तन्तवः;
    }

#यदि defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
    // Emscripten without pthreads support can only use a single thread
    n_तन्तवः = 1;
#यदि_अन्तः

#यदि defined(__wasi__)
    // WASI doesn't support parallelism yet
    n_तन्तवः = 1;
#यदि_अन्तः

    size_t work_आकार = 0;

    संरचना ggml_cplan cplan;
    स्मृतिं_पूरय(&cplan, 0, परिमाण(संरचना ggml_cplan));

    पूर्णाङ्क max_tasks = 1;

    // thread scheduling for the different operations + work buffer size estimation
    प्रति (पूर्णाङ्क i = 0; i < cgraph->n_nodes; i++) {
        संरचना ggml_tensor * ग्रन्थिः = cgraph->nodes[i];

        ध्रुव पूर्णाङ्क n_tasks = ggml_get_n_tasks(ग्रन्थिः, n_तन्तवः);

        max_tasks = MAX(max_tasks, n_tasks);

        size_t cur = 0;

        यदि (!ggml_cpu_extra_work_आकार(n_तन्तवः, ग्रन्थिः, &cur)) {
            विकल्प (ग्रन्थिः->op) {
                पक्ष GGML_OP_CPY:
                पक्ष GGML_OP_DUP:
                    {
                        यदि (ggml_is_quantized(ग्रन्थिः->type) ||
                            // F16 -> BF16 and BF16 -> F16 copies go through intermediate F32
                            (ग्रन्थिः->src[0]->type == GGML_TYPE_F16  && ग्रन्थिः->src[1] && ग्रन्थिः->src[1]->type == GGML_TYPE_BF16) ||
                            (ग्रन्थिः->src[0]->type == GGML_TYPE_BF16 && ग्रन्थिः->src[1] && ग्रन्थिः->src[1]->type == GGML_TYPE_F16) ||
                            // conversion between F32 and I32
                            (ग्रन्थिः->src[0]->type == GGML_TYPE_F32 && ग्रन्थिः->src[1] && ग्रन्थिः->src[1]->type == GGML_TYPE_I32) ||
                            (ग्रन्थिः->src[0]->type == GGML_TYPE_I32 && ग्रन्थिः->src[1] && ग्रन्थिः->src[1]->type == GGML_TYPE_F32)) {
                            cur = ggml_type_आकार(GGML_TYPE_F32) * ग्रन्थिः->ne[0] * n_tasks;
                        }
                    } भङ्ग;
                पक्ष GGML_OP_सङ्कलय:
                पक्ष GGML_OP_सङ्कलय_ID:
                पक्ष GGML_OP_ADD1:
                    {
                        यदि (ggml_is_quantized(ग्रन्थिः->src[0]->type)) {
                            cur = ggml_type_आकार(GGML_TYPE_F32) * ग्रन्थिः->src[0]->ne[0] * n_tasks;
                        }
                    } भङ्ग;
                पक्ष GGML_OP_ACC:
                    {
                        यदि (ggml_is_quantized(ग्रन्थिः->src[0]->type)) {
                            cur = ggml_type_आकार(GGML_TYPE_F32) * ग्रन्थिः->src[1]->ne[0] * n_tasks;
                        }
                    } भङ्ग;
                पक्ष GGML_OP_गणनम्_EQUAL:
                    {
                        cur = ggml_type_आकार(ग्रन्थिः->type)*n_tasks;
                    } भङ्ग;
                पक्ष GGML_OP_MUL_MAT:
                    {
                        ध्रुव गणना ggml_type vec_dot_type = type_traits_cpu[ग्रन्थिः->src[0]->type].vec_dot_type;
                        यदि (ग्रन्थिः->src[1]->type != vec_dot_type) {
                            cur = ggml_row_आकार(vec_dot_type, ggml_nelements(ग्रन्थिः->src[1]));
                        }
                        // Workspace for tiled (see tiled.h)
                        cur = GGML_PAD(cur, 64);
                        cur += ggml_tiled_wdata_आकार(n_tasks, ग्रन्थिः);
                    } भङ्ग;
                पक्ष GGML_OP_MUL_MAT_ID:
                    {
                        cur = 0;
                        ध्रुव संरचना ggml_tensor * src0 = ग्रन्थिः->src[0];
                        ध्रुव संरचना ggml_tensor * src1 = ग्रन्थिः->src[1];
                        ध्रुव संरचना ggml_tensor * ids = ग्रन्थिः->src[2];
                        ध्रुव गणना ggml_type vec_dot_type = type_traits_cpu[src0->type].vec_dot_type;
                        ध्रुव पूर्णाङ्क n_as = src0->ne[2];
                        // src1
                        यदि (src1->type != vec_dot_type) {
                            cur += ggml_row_आकार(vec_dot_type, ggml_nelements(src1)) + परिमाण(int64_t);
                        }
                        // matrix_row_counts
                        cur += n_as * परिमाण(int64_t) + परिमाण(int64_t);
                        // matrix_rows
                        cur += n_as*ids->ne[0]*ids->ne[1]*परिमाण(संरचना mmid_row_mapping) + परिमाण(int64_t);
                        // atomic_current_chunk
                        cur += CACHE_पङ्क्तिः_आकार*n_as + CACHE_पङ्क्तिः_आकार;
                        // Workspace for tiled (see tiled.h)
                        cur = GGML_PAD(cur, 64);
                        cur += ggml_tiled_wdata_आकार(n_tasks, ग्रन्थिः);
                    } भङ्ग;
                पक्ष GGML_OP_बहिर्गमनम्_PROD:
                    {
                        यदि (ggml_is_quantized(ग्रन्थिः->src[0]->type) ||
                            ग्रन्थिः->src[0]->type == GGML_TYPE_F16) {
                            cur = ggml_type_आकार(GGML_TYPE_F32) * ग्रन्थिः->src[0]->ne[0] * n_tasks;
                        }
                    } भङ्ग;
                पक्ष GGML_OP_समुच्चयः_पङ्क्तिकाः:
                    {
                        यदि (ग्रन्थिः->src[0]->type == GGML_TYPE_F16 && ग्रन्थिः->type != GGML_TYPE_F16) {
                            cur = ggml_type_आकार(GGML_TYPE_F32) * ग्रन्थिः->src[0]->ne[0] * n_tasks;
                        }
                    } भङ्ग;
                पक्ष GGML_OP_SOFT_MAX:
                पक्ष GGML_OP_ROPE:
                पक्ष GGML_OP_ROPE_पृष्ठभागः:
                    {
                        cur = ggml_type_आकार(GGML_TYPE_F32) * ग्रन्थिः->ne[0] * n_tasks;
                    } भङ्ग;
                पक्ष GGML_OP_CONV_TRANSPOSE_1D:
                    {
                        GGML_ASSERT(ग्रन्थिः->src[0]->ne[3] == 1);
                        GGML_ASSERT(ग्रन्थिः->src[1]->ne[2] == 1);
                        GGML_ASSERT(ग्रन्थिः->src[1]->ne[3] == 1);

                        ध्रुव int64_t ne00 = ग्रन्थिः->src[0]->ne[0];  // K
                        ध्रुव int64_t ne01 = ग्रन्थिः->src[0]->ne[1];  // Cout
                        ध्रुव int64_t ne02 = ग्रन्थिः->src[0]->ne[2];  // Cin
                        ध्रुव int64_t ne10 = ग्रन्थिः->src[1]->ne[0];  // L
                        ध्रुव int64_t ne11 = ग्रन्थिः->src[1]->ne[1];  // Cin

                        यदि ((ग्रन्थिः->src[0]->type == GGML_TYPE_F16 ||
                             ग्रन्थिः->src[0]->type == GGML_TYPE_BF16) &&
                            ग्रन्थिः->src[1]->type == GGML_TYPE_F32) {
                            cur += परिमाण(ggml_fp16_t)*ne00*ne01*ne02;
                            cur += परिमाण(ggml_fp16_t)*ne10*ne11;
                        } अन्यथा यदि (ग्रन्थिः->src[0]->type == GGML_TYPE_F32 &&
                                   ग्रन्थिः->src[1]->type == GGML_TYPE_F32) {
                            cur += परिमाण(भिन्न)*ne00*ne01*ne02;
                            cur += परिमाण(भिन्न)*ne10*ne11;
                        } अन्यथा {
                            GGML_ABORT("fatal error");
                        }
                    } भङ्ग;
                पक्ष GGML_OP_CONV_2D:
                पक्ष GGML_OP_CONV_3D:
                    {
                        cur = GGML_IM2_COL_WORK_आकार;
                    } भङ्ग;
                पक्ष GGML_OP_CONV_TRANSPOSE_2D:
                    {
                        ध्रुव int64_t ne00 = ग्रन्थिः->src[0]->ne[0]; // W
                        ध्रुव int64_t ne01 = ग्रन्थिः->src[0]->ne[1]; // H
                        ध्रुव int64_t ne02 = ग्रन्थिः->src[0]->ne[2]; // Channels Out
                        ध्रुव int64_t ne03 = ग्रन्थिः->src[0]->ne[3]; // Channels In

                        ध्रुव int64_t ne10 = ग्रन्थिः->src[1]->ne[0]; // W
                        ध्रुव int64_t ne11 = ग्रन्थिः->src[1]->ne[1]; // H
                        ध्रुव int64_t ne12 = ग्रन्थिः->src[1]->ne[2]; // Channels In
                        ध्रुव int64_t ne13 = ग्रन्थिः->src[1]->ne[3]; // Batch

                        GGML_ASSERT(ग्रन्थिः->src[0]->type == GGML_TYPE_F16 || ग्रन्थिः->src[0]->type == GGML_TYPE_F32);
                        GGML_ASSERT(ग्रन्थिः->src[1]->type == GGML_TYPE_F32);

                        cur += ggml_type_आकार(ग्रन्थिः->src[0]->type) * ne00 * ne01 * ne02 * ne03;
                        cur += ggml_type_आकार(ग्रन्थिः->src[0]->type) * ne10 * ne11 * ne12 * ne13;

                    } भङ्ग;
                पक्ष GGML_OP_शिखरम्_K:
                    {
                        cur += परिमाण(int32_t)*ग्रन्थिः->src[0]->ne[0]*n_tasks;
                    } भङ्ग;
                पक्ष GGML_OP_FLASH_ATTN_EXT:
                    {
                        ध्रुव int64_t neq2 = ग्रन्थिः->src[0]->ne[2]; // number of query heads
                        ध्रुव int64_t DK = ग्रन्थिः->src[1]->ne[0];
                        ध्रुव int64_t DV = ग्रन्थिः->src[2]->ne[0];

                        // Tiled flash attention scratch (tile sizes defined in common.h)
                        // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
                        size_t prefill  = परिमाण(भिन्न)*(GGML_FA_TILE_Q*DK + 2*GGML_FA_TILE_Q*GGML_FA_TILE_KV + GGML_FA_TILE_Q*DV + GGML_FA_TILE_KV*DV + GGML_FA_TILE_KV*DK)*n_tasks;

                        // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
                        // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
                        size_t n_chunks = n_tasks;
                        size_t decode   = परिमाण(भिन्न)*(neq2*n_chunks*(2+DV) + n_tasks*(DK + 2*DV));

                        cur += MAX(prefill, decode);
                    } भङ्ग;
                पक्ष GGML_OP_FLASH_ATTN_पृष्ठभागः:
                    {
                        ध्रुव int64_t    D = ग्रन्थिः->src[0]->ne[0];
                        ध्रुव int64_t ne11 = ggml_ऊर्ध्वम्(ग्रन्थिः->src[1]->ne[1], GGML_SOFT_MAX_UNROLL);
                        ध्रुव int64_t mxDn = MAX(D, ne11) * 2; // *2 because of S and SM in ggml_compute_forward_flash_attn_back
                        यदि (ग्रन्थिः->src[1]->type == GGML_TYPE_F32) {
                            cur  = परिमाण(भिन्न)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += परिमाण(भिन्न)*mxDn*n_tasks; // this is overestimated by x2
                        } अन्यथा यदि (ग्रन्थिः->src[1]->type == GGML_TYPE_F16) {
                            cur  = परिमाण(भिन्न)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += परिमाण(भिन्न)*mxDn*n_tasks; // this is overestimated by x2
                        } अन्यथा यदि (ग्रन्थिः->src[1]->type == GGML_TYPE_BF16) {
                            cur  = परिमाण(भिन्न)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += परिमाण(भिन्न)*mxDn*n_tasks; // this is overestimated by x2
                        }
                    } भङ्ग;

                पक्ष GGML_OP_CROSS_ENTROPY_LOSS:
                    {
                        cur = ggml_type_आकार(ग्रन्थिः->type)*(n_tasks + ग्रन्थिः->src[0]->ne[0]*n_tasks);
                    } भङ्ग;
                पक्ष GGML_OP_GATED_DELTA_NET:
                    {
                        ध्रुव int64_t S_v = ग्रन्थिः->src[2]->ne[0];
                        ध्रुव int64_t K   = ggml_get_op_params_i32(ग्रन्थिः, 0);
                        ध्रुव int64_t प्रत्येकम्_thread = S_v + (K > 1 ? S_v * S_v : 0);
                        cur = प्रत्येकम्_thread * परिमाण(भिन्न) * n_tasks;
                    } भङ्ग;
                पक्ष GGML_OP_गणनम्:
                    {
                        GGML_ABORT("fatal error");
                    }
                पक्ष GGML_OP_LIGHTNING_INDEXER:
                    {
                        // temp buffer for dequantizing lightning indexer keys
                        ध्रुव int64_t ne10 = ग्रन्थिः->src[1]->ne[0];
                        cur += परिमाण(भिन्न)*ne10*n_tasks;
                    } भङ्ग;
                स्वभाव:
                    भङ्ग;
            }
        }

        work_आकार = MAX(work_आकार, cur);
    }

    यदि (work_आकार > 0) {
        work_आकार += CACHE_पङ्क्तिः_आकार*(n_तन्तवः);
    }

    cplan.threadpool = threadpool;
    cplan.n_तन्तवः  = MIN(max_tasks, n_तन्तवः);
    cplan.work_आकार  = work_आकार;
    cplan.work_दत्त  = रिक्त;

    प्रत्यागम cplan;
}


// Try to fuse the current node with subsequent nodes for better performance.
// Returns the number of nodes skipped by fusion (>=1), or 0 if no fusion was applied.
स्थिर bool ggml_cpu_disable_fusion = false;  // initialized once in ggml_cpu_init(), read-only afterwards

स्थिर पूर्णाङ्क ggml_cpu_try_fuse_ops(
        ध्रुव संरचना ggml_cgraph * cgraph,
        ध्रुव पूर्णाङ्क ग्रन्थिः_n,
        ध्रुव संरचना ggml_compute_params * params,
        ध्रुव संरचना ggml_cplan * cplan) {

    यदि (ggml_cpu_disable_fusion || cplan->use_ref) {
        प्रत्यागम 0;
    }

    संरचना ggml_tensor * ग्रन्थिः = cgraph->nodes[ग्रन्थिः_n];

    यदि (ग्रन्थिः->op == GGML_OP_RMS_NORM) {
        // RMS_NORM + MUL fusion
        ध्रुव गणना ggml_op fuse_ops[] = { GGML_OP_RMS_NORM, GGML_OP_MUL };
        यदि (ggml_can_fuse(cgraph, ग्रन्थिः_n, fuse_ops, 2)) {
            संरचना ggml_tensor * mul_ग्रन्थिः = cgraph->nodes[ग्रन्थिः_n + 1];
            ध्रुव संरचना ggml_tensor * mul_w = (mul_ग्रन्थिः->src[0] == ग्रन्थिः)
                ? mul_ग्रन्थिः->src[1] : mul_ग्रन्थिः->src[0];
            यदि (ग्रन्थिः->src[0]->type  == GGML_TYPE_F32 &&
                mul_ग्रन्थिः->type      == GGML_TYPE_F32 &&
                mul_w->type         == GGML_TYPE_F32 &&
                mul_w->ne[0]        == ग्रन्थिः->ne[0]   &&
                mul_w->nb[0]        == परिमाण(भिन्न)) {

                ggml_compute_forward_rms_norm_mul_fused(params, ग्रन्थिः, mul_ग्रन्थिः);
                प्रत्यागम 1;
            }
        }
    }

    प्रत्यागम 0;
}

स्थिर thread_ret_t ggml_graph_compute_thread(शून्य * दत्त) {
    संरचना ggml_compute_state * state = (संरचना ggml_compute_state *) दत्त;
    संरचना ggml_threadpool    * tp    = state->threadpool;

    ध्रुव संरचना ggml_cgraph * cgraph = tp->cgraph;
    ध्रुव संरचना ggml_cplan  * cplan  = tp->cplan;

#यदि_परिभाषितम् GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_समुच्चयः_numa_thread_affinity(state->ith);
#अन्यथा
    समुच्चयः_numa_thread_affinity(state->ith);
#यदि_अन्तः

    संरचना ggml_compute_params params = {
        /*.ith        =*/ state->ith,
        /*.nth        =*/ atomic_आरोपय_explicit(&tp->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_तन्तवः_आवरणम्,
        /*.wsize      =*/ cplan->work_आकार,
        /*.wdata      =*/ cplan->work_दत्त,
        /*.threadpool =*/ tp,
        /*.use_ref    =*/ cplan->use_ref,
    };

#यदि_परिभाषितम् GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p\n", state->ith, (ध्रुव शून्य *)cplan);
#अन्यथा
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p last-graph %d\n", state->ith, (ध्रुव शून्य *)cplan, state->चरमः_graph);
#यदि_अन्तः

    प्रति (पूर्णाङ्क ग्रन्थिः_n = 0; ग्रन्थिः_n < cgraph->n_nodes && atomic_आरोपय_explicit(&tp->abort, memory_order_relaxed) != ग्रन्थिः_n; ग्रन्थिः_n++) {
        संरचना ggml_tensor * ग्रन्थिः = cgraph->nodes[ग्रन्थिः_n];

        यदि (ggml_op_is_रिक्तानि(ग्रन्थिः->op)) {
            // skip NOPs
            अनुवर्त;
        }

        यदि ((ग्रन्थिः->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            अनुवर्त;
        }

        // TODO: move fused-op detection into ggml_graph_plan so fusion decisions are made once at planning time
        // Try fused ops, fall back to normal compute
        ध्रुव पूर्णाङ्क n_fused = ggml_cpu_try_fuse_ops(cgraph, ग्रन्थिः_n, &params, cplan);
        यदि (n_fused > 0) {
            ग्रन्थिः_n += n_fused;
        } अन्यथा {
            ggml_compute_forward(&params, ग्रन्थिः);
        }

        यदि (state->ith == 0 && cplan->abort_callback &&
                cplan->abort_callback(cplan->abort_callback_दत्त)) {
            atomic_store_explicit(&tp->abort, ग्रन्थिः_n + 1, memory_order_relaxed);
            tp->ec    = GGML_स्थितिः_ABORTED;
        }

        यदि (ग्रन्थिः_n + 1 < cgraph->n_nodes) {
            ggml_barrier(state->threadpool);
        }
    }

#यदि_परिभाषितम् GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p\n", state->ith, (ध्रुव शून्य *)cplan);
#अन्यथा
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p last-graph %d\n", state->ith, (ध्रुव शून्य *)cplan, state->चरमः_graph);
#यदि_अन्तः

    ggml_barrier(state->threadpool);

#यदि_परिभाषितम् GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(state->ith);
#यदि_अन्तः

    प्रत्यागम 0;
}

#यदि_अपरिभाषितम् GGML_USE_OPENMP

// check if thread is ready to proceed (exit from polling or sleeping)
// returns true if loops should exit, sets state->pending to indicate new work
स्थिर अन्तःस्थ bool ggml_graph_compute_thread_ready(संरचना ggml_compute_state * state) {
    संरचना ggml_threadpool * threadpool = state->threadpool;

    यदि (state->pending || threadpool->stop || threadpool->pause) { प्रत्यागम true; }

    // check for new graph/work
    पूर्णाङ्क n_graph   = atomic_आरोपय_explicit(&threadpool->n_graph, memory_order_relaxed);
    पूर्णाङ्क n_तन्तवः = n_graph & GGML_THREADPOOL_N_तन्तवः_आवरणम्;
    यदि (n_graph != state->चरमः_graph) {
        state->pending    = (state->ith < n_तन्तवः);
        state->चरमः_graph = n_graph;
        प्रत्यागम true;
    }

    प्रत्यागम false;
}

// sync thread state after polling
स्थिर अन्तःस्थ शून्य ggml_graph_compute_thread_sync(संरचना ggml_compute_state * state) {
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #यदि_परिभाषितम् GGML_TSAN_ENABLED
    atomic_fetch_सङ्कलय_explicit(&state->threadpool->n_graph, 0, memory_order_seq_cst);
    #अन्यथा
    atomic_thread_fence(memory_order_seq_cst);
    #यदि_अन्तः
    अप्रयुक्तम्(state);
}

स्थिर अन्तःस्थ bool ggml_graph_compute_poll_for_work(संरचना ggml_compute_state * state) {
    संरचना ggml_threadpool * threadpool = state->threadpool;

    // This seems to make 0 ... 100 a decent range for polling level across modern processors.
    // Perhaps, we can adjust it dynamically based on load and things.
    ध्रुव uint64_t n_आवर्तनानि = 1024UL * 128 * threadpool->poll;

    प्रति (uint64_t i=0; !ggml_graph_compute_thread_ready(state) && i < n_आवर्तनानि; i++) {
        // No new work. Keep polling.
        ggml_thread_cpu_relax();
    }

    प्रत्यागम state->pending;
}

स्थिर अन्तःस्थ bool ggml_graph_compute_check_for_work(संरचना ggml_compute_state * state) {
    संरचना ggml_threadpool * threadpool = state->threadpool;

    यदि (ggml_graph_compute_poll_for_work(state)) {
        ggml_graph_compute_thread_sync(state);
        प्रत्यागम state->pending;
    }

    ggml_mutex_तालकम्_shared(&threadpool->mutex);
    यावत् (!ggml_graph_compute_thread_ready(state)) {
        // No new work. Wait for the signal.
        GGML_PRINT_DEBUG("thread #%d waiting for work (sleeping)\n", state->ith);
        ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
    }
    ggml_mutex_unlock_shared(&threadpool->mutex);

    प्रत्यागम state->pending;
}

स्थिर thread_ret_t ggml_graph_compute_secondary_thread(शून्य* दत्त) {
    संरचना ggml_compute_state * state = (संरचना ggml_compute_state *) दत्त;
    संरचना ggml_threadpool * threadpool = state->threadpool;

    ggml_thread_apply_प्राथमिकता(threadpool->prio);
    यदि (ggml_thread_cpumask_is_valid(state->cpumask)) {
        ggml_thread_apply_affinity(state->cpumask);
    }

    यावत् (true) {
        // Check if we need to sleep
        यावत् (threadpool->pause) {
            GGML_PRINT_DEBUG("thread #%d inside pause loop\n", state->ith);
            ggml_mutex_तालकम्_shared(&threadpool->mutex);
            यदि (threadpool->pause) {
                ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
            }
            GGML_PRINT_DEBUG("thread #%d resuming after wait\n", state->ith);
            ggml_mutex_unlock_shared(&threadpool->mutex);
        }

        // This needs to be checked for after the cond_wait
        यदि (threadpool->stop) भङ्ग;

        // Check if there is new work
        // The main thread is the only one that can dispatch new work

        ggml_graph_compute_check_for_work(state);
        यदि (state->pending) {
            state->pending = false;
            ggml_graph_compute_thread(state);
        }
    }

    प्रत्यागम (thread_ret_t) 0;
}

// Start processing new graph
स्थिर शून्य ggml_graph_compute_kickoff(संरचना ggml_threadpool * threadpool, पूर्णाङ्क n_तन्तवः)
{
    // Always take the mutex here because the worker threads are doing hybrid poll/wait

    ggml_mutex_तालकम्(&threadpool->mutex);

    // Update the number of active threads and the graph count
    पूर्णाङ्क n_graph = atomic_आरोपय_explicit(&threadpool->n_graph, memory_order_relaxed) >> GGML_THREADPOOL_N_तन्तवः_द्वयङ्काः;
    n_graph = ((n_graph + 1) << GGML_THREADPOOL_N_तन्तवः_द्वयङ्काः) | (n_तन्तवः & GGML_THREADPOOL_N_तन्तवः_आवरणम्);

    GGML_PRINT_DEBUG("compute-kickoff: n_threads %d n_graph %d\n", n_तन्तवः, n_graph);

    // Indicate the graph is ready to be processed
    // We need the full seq-cst fence here because of the polling threads (used in thread_sync)
    atomic_store_explicit(&threadpool->n_graph, n_graph, memory_order_seq_cst);

    यदि (threadpool->pause) {
       // Update main thread prio and affinity to match the threadpool settings
       ggml_thread_apply_प्राथमिकता(threadpool->prio);
       यदि (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
           ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
       }

       // resume does cond broadcast
       ggml_threadpool_resume_locked(threadpool);
    } अन्यथा {
       ggml_cond_प्रसारणम्(&threadpool->cond);
    }

    ggml_mutex_unlock(&threadpool->mutex);
}

#यदि_अन्तः // GGML_USE_OPENMP

स्थिर संरचना ggml_threadpool * ggml_threadpool_new_impl(
    संरचना ggml_threadpool_params * tpp,
               संरचना ggml_cgraph * cgraph,
                संरचना ggml_cplan * cplan) {

    संरचना ggml_threadpool * threadpool =
        ggml_aligned_malloc(परिमाण(संरचना ggml_threadpool));
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
        threadpool->workers          = रिक्त;
        threadpool->n_तन्तवः        = tpp->n_तन्तवः;
        threadpool->poll             = tpp->poll;
        threadpool->prio             = tpp->prio;
        threadpool->ec               = GGML_स्थितिः_SUCCESS;
    }

    // Allocate and init workers state
    ध्रुव size_t workers_आकार = परिमाण(संरचना ggml_compute_state) * tpp->n_तन्तवः;
    संरचना ggml_compute_state * workers = ggml_aligned_malloc(workers_आकार);

    स्मृतिं_पूरय(workers, 0, workers_आकार);
    प्रति (पूर्णाङ्क j = 0; j < tpp->n_तन्तवः; j++) {
        workers[j].threadpool = threadpool;
        workers[j].ith        = j;
    }

    threadpool->workers = workers;

#यदि_परिभाषितम् GGML_USE_OPENMP
    int32_t cpumask_iter = 0;

    // Compute CPU masks for each thread
    प्रति (पूर्णाङ्क j = 0; j < tpp->n_तन्तवः; j++) {
        ggml_thread_cpumask_अग्र(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);
    }
#अन्यथा // GGML_USE_OPENMP
    ggml_mutex_init(&threadpool->mutex);
    ggml_cond_init(&threadpool->cond);

    // Spin the threads for all workers, and update CPU placements.
    // Place the main thread last (towards the higher numbered CPU cores).

    int32_t cpumask_iter = 0;

    प्रति (पूर्णाङ्क j = 1; j < tpp->n_तन्तवः; j++) {
        ggml_thread_cpumask_अग्र(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);

        int32_t rc = ggml_thread_create(&workers[j].thrd, रिक्त, ggml_graph_compute_secondary_thread, &workers[j]);
        GGML_ASSERT(rc == 0);
    }

    ggml_thread_cpumask_अग्र(tpp->cpumask, workers[0].cpumask, tpp->strict_cpu, &cpumask_iter);

    यदि (!threadpool->pause) {
        // Update main thread prio and affinity at the start, otherwise we'll do it in resume
        ggml_thread_apply_प्राथमिकता(threadpool->prio);
        यदि (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
            ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
        }
    }
#यदि_अन्तः // GGML_USE_OPENMP

    प्रत्यागम threadpool;
}

संरचना ggml_threadpool * ggml_threadpool_new(संरचना ggml_threadpool_params * tpp) {
    प्रत्यागम ggml_threadpool_new_impl(tpp, रिक्त, रिक्त);
}

गणना ggml_स्थितिः ggml_graph_compute(संरचना ggml_cgraph * cgraph, संरचना ggml_cplan * cplan) {
    ggml_cpu_init();

    GGML_ASSERT(cplan);
    GGML_ASSERT(cplan->n_तन्तवः > 0);
    GGML_ASSERT(cplan->work_आकार == 0 || cplan->work_दत्त != रिक्त);

    पूर्णाङ्क n_तन्तवः                               = cplan->n_तन्तवः;
    संरचना ggml_threadpool * threadpool = cplan->threadpool;

    bool disposable_threadpool = false;

    यदि (threadpool == रिक्त) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
        disposable_threadpool = true;

        संरचना ggml_threadpool_params ttp = ggml_threadpool_params_default(n_तन्तवः);
        threadpool = ggml_threadpool_new_impl(&ttp, cgraph, cplan);
    } अन्यथा {
        // Reset some of the parameters that need resetting
        // No worker threads should be accessing the parameters below at this stage
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->current_chunk    = 0;
        threadpool->abort            = -1;
        threadpool->ec               = GGML_स्थितिः_SUCCESS;
    }

#यदि_परिभाषितम् GGML_USE_OPENMP
    यदि (n_तन्तवः > 1) {
        #निर्देशः omp parallel अङ्कः_तन्तवः(n_तन्तवः)
        {
            #निर्देशः omp single
            {
                // update the number of threads from the actual number of threads that we got from OpenMP
                n_तन्तवः = omp_get_अङ्कः_तन्तवः();
                atomic_store_explicit(&threadpool->n_graph, n_तन्तवः, memory_order_relaxed);
            }

            // Apply thread CPU mask and priority
            पूर्णाङ्क ith = omp_get_thread_अङ्कः();

            ggml_thread_apply_प्राथमिकता(threadpool->prio);
            यदि (ggml_thread_cpumask_is_valid(threadpool->workers[ith].cpumask)) {
                ggml_thread_apply_affinity(threadpool->workers[ith].cpumask);
            }
            ggml_graph_compute_thread(&threadpool->workers[ith]);
        }
    } अन्यथा {
        atomic_store_explicit(&threadpool->n_graph, 1, memory_order_relaxed);
        ggml_graph_compute_thread(&threadpool->workers[0]);
    }
#अन्यथा
    यदि (n_तन्तवः > threadpool->n_तन्तवः) {
        GGML_LOG_WARN("cplan requested more threads (%d) than available (%d)\n", n_तन्तवः, threadpool->n_तन्तवः);
        n_तन्तवः = threadpool->n_तन्तवः;
    }

    // Kick all threads to start the new graph
    ggml_graph_compute_kickoff(threadpool, n_तन्तवः);

    // This is a work thread too
    ggml_graph_compute_thread(&threadpool->workers[0]);
#यदि_अन्तः

    // don't leave affinity set on the main thread
    clear_numa_thread_affinity();

    गणना ggml_स्थितिः ret = threadpool->ec;

    यदि (disposable_threadpool) {
        ggml_threadpool_free(threadpool);
    }

    प्रत्यागम ret;
}

गणना ggml_स्थितिः ggml_graph_compute_with_ctx(संरचना ggml_context * ctx, संरचना ggml_cgraph * cgraph, पूर्णाङ्क n_तन्तवः) {
    संरचना ggml_cplan cplan = ggml_graph_plan(cgraph, n_तन्तवः, रिक्त);

    cplan.work_दत्त = (uint8_t *)ggml_new_बफर(ctx, cplan.work_आकार);

    प्रत्यागम ggml_graph_compute(cgraph, &cplan);
}

शून्य ggml_cpu_fp32_गन्तव्यम्_fp32(ध्रुव भिन्न * x, भिन्न * y, int64_t n) {
    स्मृतिं_प्रतिलिख(y, x, n * परिमाण(भिन्न));
}

शून्य ggml_cpu_fp32_गन्तव्यम्_fp16(ध्रुव भिन्न * x, ggml_fp16_t * y, int64_t n) {
    int64_t i = 0;
#यदि defined(__F16C__)
#यदि defined(__AVX512F__)
    प्रति (; i + 15 < n; i += 16) {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        __m256i y_vec = _mm512_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256((__m256i *)(y + i), y_vec);
    }
#यदि_अन्तः
    प्रति (; i + 7 < n; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m128i y_vec = _mm256_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i *)(y + i), y_vec);
    }
    प्रति (; i + 3 < n; i += 4) {
        __m128 x_vec = _mm_loadu_ps(x + i);
        __m128i y_vec = _mm_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storel_epi64((__m128i *)(y + i), y_vec);
    }
#नोचेद्यदि defined(__riscv_zvfh)
    प्रति (पूर्णाङ्क vl; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(&x[i], vl);
        vfloat16m1_t vy = __riscv_vfncvt_f_f_w_f16m1(vx, vl);
        __riscv_vse16_v_f16m1((_Float16 *)&y[i], vy, vl);
    }
#नोचेद्यदि defined(__VXE__) || defined(__VXE2__)
    प्रति (; i + 7 < n; i += 8) {
        ध्रुव uint32x4_t v_yl = __lzs_f32cx4_to_f16(vec_xl(0, x + i + 0));
        ध्रुव uint32x4_t v_yh = __lzs_f32cx4_to_f16(vec_xl(0, x + i + 4));
        vec_xst(vec_pack(v_yl, v_yh), 0, (uint16_t *)(y + i));
    }
#यदि_अन्तः
    प्रति (; i < n; ++i) {
        y[i] = GGML_CPU_FP32_गन्तव्यम्_FP16(x[i]);
    }
}

शून्य ggml_cpu_fp16_गन्तव्यम्_fp32(ध्रुव ggml_fp16_t * x, भिन्न * y, int64_t n) {
    int64_t i = 0;
#यदि defined(__F16C__)
#यदि defined(__AVX512F__)
    प्रति (; i + 15 < n; i += 16) {
        __m256i x_vec = _mm256_loadu_si256((ध्रुव __m256i *)(x + i));
        __m512 y_vec = _mm512_cvtph_ps(x_vec);
        _mm512_storeu_ps(y + i, y_vec);
    }
#यदि_अन्तः
    प्रति (; i + 7 < n; i += 8) {
        __m128i x_vec = _mm_loadu_si128((ध्रुव __m128i *)(x + i));
        __m256 y_vec = _mm256_cvtph_ps(x_vec);
        _mm256_storeu_ps(y + i, y_vec);
    }
    प्रति (; i + 3 < n; i += 4) {
        __m128i x_vec = _mm_loadl_epi64((ध्रुव __m128i *)(x + i));
        __m128 y_vec = _mm_cvtph_ps(x_vec);
        _mm_storeu_ps(y + i, y_vec);
    }

#नोचेद्यदि defined(__riscv_v_intrinsic) && defined(__riscv_zvfhmin)
    // calculate step size
    ध्रुव पूर्णाङ्क epr = __riscv_vsetvlmax_e16m2();
    ध्रुव पूर्णाङ्क चरणः = epr * 2;
    ध्रुव पूर्णाङ्क np = (n & ~(चरणः - 1));

    // unroll by 2
    प्रति (; i < np; i += चरणः) {
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((ध्रुव _Float16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vfloat16m2_t ax1 = __riscv_vle16_v_f16m2((ध्रुव _Float16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvt_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    पूर्णाङ्क vl;
    प्रति (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((ध्रुव _Float16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }

#यदि_अन्तः

    प्रति (; i < n; ++i) {
        y[i] = GGML_CPU_FP16_गन्तव्यम्_FP32(x[i]);
    }
}

शून्य ggml_cpu_fp32_गन्तव्यम्_bf16(ध्रुव भिन्न * x, ggml_bf16_t * y, int64_t n) {
    int64_t i = 0;
    प्रति (; i < n; ++i) {
        y[i] = GGML_FP32_गन्तव्यम्_BF16(x[i]);
    }
}

शून्य ggml_cpu_fp32_गन्तव्यम्_i32(ध्रुव भिन्न * x, int32_t * y, int64_t n) {
    int64_t i = 0;
    प्रति (; i < n; ++i) {
        y[i] = x[i];
    }
}

शून्य ggml_cpu_bf16_गन्तव्यम्_fp32(ध्रुव ggml_bf16_t * x, भिन्न * y, int64_t n) {
    int64_t i = 0;
#यदि defined(__AVX2__)
#यदि defined(__AVX512F__)
    प्रति (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i,
                        _mm512_castsi512_ps(
                            _mm512_slli_epi32(
                                _mm512_cvtepu16_epi32(
                                    _mm256_loadu_si256(
                                        (ध्रुव __m256i *)(x + i))),
                                16)));
    }
#यदि_अन्तः
    प्रति (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i,
                        _mm256_castsi256_ps(
                            _mm256_slli_epi32(
                                _mm256_cvtepu16_epi32(
                                    _mm_loadu_si128(
                                        (ध्रुव __m128i *)(x + i))),
                                16)));
    }
#नोचेद्यदि defined(__riscv_v_intrinsic) && defined(__riscv_zvfbfmin)
    // calculate step size
    ध्रुव पूर्णाङ्क epr = __riscv_vsetvlmax_e16m2();
    ध्रुव पूर्णाङ्क चरणः = epr * 2;
    ध्रुव पूर्णाङ्क np = (n & ~(चरणः - 1));

    // unroll by 2
    प्रति (; i < np; i += चरणः) {
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((ध्रुव __bf16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vbfloat16m2_t ax1 = __riscv_vle16_v_bf16m2((ध्रुव __bf16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    पूर्णाङ्क vl;
    प्रति (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((ध्रुव __bf16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }
#यदि_अन्तः
    प्रति (; i < n; i++) {
        y[i] = GGML_BF16_गन्तव्यम्_FP32(x[i]);
    }
}

पूर्णाङ्क ggml_cpu_has_avx(शून्य) {
#यदि defined(__AVX__)
    प्रत्यागम 1;
#अन्यथा
    प्रत्यागम 0;
#यदि_अन्तः
}

पूर्णाङ्क ggml_cpu_has_avx_vnni(शून्य) {
#यदि defined(__AVXVNNI__)
    प्रत्यागम 1;
#अन्यथा
    प्रत्यागम 0;
#यदि_अन्तः
}

पूर्णाङ्क ggml_cpu_has_avx2(शून्य) {
#यदि defined(__AVX2__)
    प्रत्यागम 1;
#अन्यथा
    प्रत्यागम 0;
#यदि_अन्तः
}

पूर्णाङ्क ggml_cpu_has_avx512(शून्य) {
#यदि defined(__AVX512F__)
    प्रत्यागम 1;
#अन्यथा
    प्रत्यागम 0;
#यदि_अन्तः
}

पूर्णाङ्क ggml_cpu_has_avx512_vbmi(शून्य) {
#यदि defined(__AVX512VBMI__)
    प्रत्यागम 1;
#अन्यथा
    प्रत्यागम 0;
#यदि_अन्तः
}

पूर्णाङ्क ggml_cpu_has_avx512_vnni(शून्य) {
#यदि defined(__AVX512VNNI__)
    प्रत्यागम 1;
#अन्यथा
    प्रत्यागम 0;
#यदि_अन्तः
}

पूर्णाङ्क ggml_cpu_has_avx512_bf16(शून्य) {
#यदि defined(__AVX512BF16__)
    प्रत्यागम 1;
#अन्यथा
    प्रत्यागम 0;
#यदि_अन्तः
}

पूर्णाङ्क ggml_cpu_has_amx_int8(शून्य) {
#यदि defined(__AMX_INT8__)
    प्रत्यागम 1;
#अन्यथा
    प्रत्यागम 0;
#यदि_अन्तः
}

पूर्णाङ्क ggml_cpu_has_bmi2(शून्य) {
#यदि defined(__BMI2__)
    प्रत्यागम 1;
#अन्यथा
    प्रत्यागम 0;
#यदि_अन्तः
}

पूर्णाङ्क ggml_cpu_has_fma(शून्य) {
#यदि defined(__FMA__)
    प्रत्यागम 1;
#अन्यथा
    प्रत्यागम 0;
#यदि_अन्तः
}

पूर्णाङ्क ggml_cpu_has_arm_fma(शून्य) {
#यदि defined(__ARM_FEATURE_FMA)
    प्रत्यागम 1;
#अन्यथा
    प्रत्यागम 0;
#यदि_अन्तः
}

पूर्णाङ्क ggml_cpu_has_riscv_v(शून्य) {
#यदि defined(__riscv_v_intrinsic)
    प्रत्यागम 1;
#अन्यथा
    प्रत्यागम 0;
#यदि_अन्तः
}

पूर्णाङ्क ggml_cpu_get_rvv_vlen(शून्य) {
#यदि defined(__riscv) && defined(__riscv_v_intrinsic)
    प्रत्यागम ggml_riscv_arch_features.rvv_vlen;
#अन्यथा
    प्रत्यागम 0;
#यदि_अन्तः
}

पूर्णाङ्क ggml_cpu_has_f16c(शून्य) {
#यदि defined(__F16C__)
    प्रत्यागम 1;
#अन्यथा
    प्रत्यागम 0;
#यदि_अन्तः
}

पूर्णाङ्क ggml_cpu_has_fp16_va(शून्य) {
#यदि defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    प्रत्यागम 1;
#अन्यथा
    प्रत्यागम 0;
#यदि_अन्तः
}

पूर्णाङ्क ggml_cpu_has_wasm_simd(शून्य) {
#यदि defined(__wasm_simd128__)
    प्रत्यागम 1;
#अन्यथा
    प्रत्यागम 0;
#यदि_अन्तः
}

पूर्णाङ्क ggml_cpu_has_llamafile(शून्य) {
#यदि defined(GGML_USE_LLAMAFILE)
    प्रत्यागम 1;
#अन्यथा
    प्रत्यागम 0;
#यदि_अन्तः
}

पूर्णाङ्क ggml_cpu_has_sse3(शून्य) {
#यदि defined(__SSE3__)
    प्रत्यागम 1;
#अन्यथा
    प्रत्यागम 0;
#यदि_अन्तः
}

पूर्णाङ्क ggml_cpu_has_ssse3(शून्य) {
#यदि defined(__SSSE3__)
    प्रत्यागम 1;
#अन्यथा
    प्रत्यागम 0;
#यदि_अन्तः
}

पूर्णाङ्क ggml_cpu_has_vsx(शून्य) {
#यदि defined(__POWER9_VECTOR__)
    प्रत्यागम 1;
#अन्यथा
    प्रत्यागम 0;
#यदि_अन्तः
}

पूर्णाङ्क ggml_cpu_has_vxe(शून्य) {
#यदि defined(__VXE__) || defined(__VXE2__)
    प्रत्यागम 1;
#अन्यथा
    प्रत्यागम 0;
#यदि_अन्तः
}

पूर्णाङ्क ggml_cpu_has_neon(शून्य) {
#यदि defined(__ARM_ARCH) && defined(__ARM_NEON)
    प्रत्यागम 1;
#अन्यथा
    प्रत्यागम 0;
#यदि_अन्तः
}

पूर्णाङ्क ggml_cpu_has_dotprod(शून्य) {
#यदि defined(__ARM_ARCH) && defined(__ARM_FEATURE_DOTPROD)
    प्रत्यागम 1;
#अन्यथा
    प्रत्यागम 0;
#यदि_अन्तः
}

पूर्णाङ्क ggml_cpu_has_sve(शून्य) {
#यदि defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    प्रत्यागम 1;
#अन्यथा
    प्रत्यागम 0;
#यदि_अन्तः
}

पूर्णाङ्क ggml_cpu_has_matmul_int8(शून्य) {
#यदि defined(__ARM_ARCH) && defined(__ARM_FEATURE_MATMUL_INT8)
    प्रत्यागम 1;
#अन्यथा
    प्रत्यागम 0;
#यदि_अन्तः
}

पूर्णाङ्क ggml_cpu_get_sve_cnt(शून्य) {
#यदि defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    प्रत्यागम ggml_arm_arch_features.sve_cnt;
#अन्यथा
    प्रत्यागम 0;
#यदि_अन्तः
}

पूर्णाङ्क ggml_cpu_has_sme(शून्य) {
#यदि defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME)
    प्रत्यागम 1;
#अन्यथा
    प्रत्यागम 0;
#यदि_अन्तः
}

पूर्णाङ्क ggml_cpu_has_sme2(शून्य) {
#यदि defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME2)
    प्रत्यागम 1;
#अन्यथा
    प्रत्यागम 0;
#यदि_अन्तः
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

    यदि (is_first_call) {
        // initialize GELU, Quick GELU, SILU and EXP F32 tables
        {
            ध्रुव uint64_t t_start = ggml_time_us(); अप्रयुक्तम्(t_start);

            प्रति (पूर्णाङ्क i = 0; i < (1 << 16); ++i) {
                सङ्घ {
                    uint16_t u16;
                    ggml_fp16_t fp16;
                } u = {i};
                भिन्न f = GGML_COMPUTE_FP16_गन्तव्यम्_FP32(u.fp16);
                ggml_सारणी_f32_f16[i] = f;
                ggml_सारणी_gelu_f16[i] = GGML_CPU_FP32_गन्तव्यम्_FP16(ggml_gelu_f32(f));
                ggml_सारणी_gelu_शीघ्र_f16[i] = GGML_CPU_FP32_गन्तव्यम्_FP16(ggml_gelu_शीघ्र_f32(f));
            }

            // initialize E8M0 half table (256 entries)
            प्रति (पूर्णाङ्क i = 0; i < (1 << 8); ++i) {
                ggml_सारणी_f32_e8m0_half[i] = GGML_E8_M0_गन्तव्यम्_FP32_HALF(i);
            }

            // initialize UE4M3 table (256 entries)
            प्रति (पूर्णाङ्क i = 0; i < (1 << 8); ++i) {
                ggml_सारणी_f32_ue4m3[i] = ggml_ue4m3_गन्तव्यम्_fp32(i);
            }

            ध्रुव uint64_t t_अन्तः = ggml_time_us(); अप्रयुक्तम्(t_अन्तः);

            GGML_PRINT_DEBUG("%s: GELU, Quick GELU, SILU and EXP tables initialized in %f ms\n", __func__, (t_अन्तः - t_start)/1000.0);

#यदि_परिभाषितम् GGML_USE_OPENMP
            //if (!getenv("OMP_WAIT_POLICY")) {
            //    // set the wait policy to active, so that OpenMP threads don't sleep
            //    setenv("OMP_WAIT_POLICY", "active", 0)
            //}

            यदि (!getenv("KMP_BLOCKTIME")) {
                // set the time to wait before sleeping a thread
                // this is less aggressive than setting the wait policy to active, but should achieve similar results in most cases
#यदि_परिभाषितम् _WIN32
                _putenv_s("KMP_BLOCKTIME", "200"); // 200ms
#अन्यथा
                setenv("KMP_BLOCKTIME", "200", 0); // 200ms
#यदि_अन्तः
            }
#यदि_अन्तः
        }

#यदि defined(__ARM_ARCH)
        ggml_init_arm_arch_features();
#यदि_अन्तः

#यदि defined(__riscv)
        ggml_init_riscv_arch_features();
#यदि_अन्तः

        {
            ध्रुव वर्ण * env = getenv("GGML_CPU_DISABLE_FUSION");
            ggml_cpu_disable_fusion = (env != रिक्त && atoi(env) == 1);
        }

        is_first_call = false;
    }

    ggml_critical_section_अन्तः();
}
