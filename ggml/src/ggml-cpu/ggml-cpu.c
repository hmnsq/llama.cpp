#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ USE_MATH_ਨਿਰਧਾਰਤ_ਕਰਦਾ // For M_PI on MSVC

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "traits.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-cpu-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "quants.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-threading.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "unary-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "binary-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "vec.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "tiled/tiled.h"

#ਜੇ defined(_MSC_VER) || defined(__MINGW32__)
#ਅਨ੍ਤਰ੍ਭਾਵ <malloc.h> // using malloc.h with MSC/MINGW
#ਜਾਂਜੇ !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#ਅਨ੍ਤਰ੍ਭਾਵ <alloca.h>
#ਜੇ_ਅੰਤ

#ਅਨ੍ਤਰ੍ਭਾਵ <assert.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <errno.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <time.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <math.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdlib.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <inttypes.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdio.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <float.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <limits.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdarg.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <signal.h>
#ਜੇ defined(__gnu_linux__)
#ਅਨ੍ਤਰ੍ਭਾਵ <syscall.h>
#ਜੇ_ਅੰਤ

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_OPENMP
#ਅਨ੍ਤਰ੍ਭਾਵ <omp.h>
#ਜੇ_ਅੰਤ

#ਜੇ defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_MATMUL_INT8)
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ GGML_USE_LLAMAFILE
#ਜੇ_ਅੰਤ

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_LLAMAFILE
#ਅਨ੍ਤਰ੍ਭਾਵ "llamafile/sgemm.h"
#ਜੇ_ਅੰਤ

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_CPU_RISCV64_SPACEMIT
#    include "spacemit/ime.h"
#ਜੇ_ਅੰਤ

// Note: once we move threading into a separate C++ file
// will use std::hardware_destructive_interference_size instead of hardcoding it here
// and we'll use C++ attribute syntax.
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_CACHE_ਸਤਰ  64

#ਜੇ defined(__clang__) || defined(__GNUC__)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_CACHE_ALIGN __attribute__((aligned(GGML_CACHE_ਸਤਰ)))
#ਜੇ_ਅੰਤ

#ਜੇ defined(__has_feature)
#ਜੇ __has_feature(thread_sanitizer)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_TSAN_ENABLED 1
#ਜੇ_ਅੰਤ
#ਨਹੀਂਤਾਂ  // __has_feature
#ਜੇ defined(__SANITIZE_THREAD__)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_TSAN_ENABLED 1
#ਜੇ_ਅੰਤ
#ਜੇ_ਅੰਤ // __has_feature

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਅਣਵਰਤਿਆ GGML_ਅਣਵਰਤਿਆ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਅਦਲਾ_ਬਦਲੀ(x, y, T) ਕਰੋ { T ਅਦਲਾ_ਬਦਲੀ = x; (x) = y; (y) = ਅਦਲਾ_ਬਦਲੀ; } ਜਦੋਂ (0)

// precomputed f32 table for f16 (256 KB) (simd-mappings.h)
ਦਸ਼ਮਲਵ ggml_ਸਾਰਣੀ_f32_f16[1 << 16];

// precomputed f32 table for e8m0 half (1 KB) (simd-mappings.h)
ਦਸ਼ਮਲਵ ggml_ਸਾਰਣੀ_f32_e8m0_half[1 << 8];

// precomputed f32 table for ue4m3 (1 KB) (simd-mappings.h)
ਦਸ਼ਮਲਵ ggml_ਸਾਰਣੀ_f32_ue4m3[1 << 8];

#ਜੇ defined(__ARM_ARCH)
ਬਣਤਰ ggml_arm_arch_features_type {
    ਪੂਰਨ sve_cnt;
} ggml_arm_arch_features = { 0 };
#ਜੇ_ਅੰਤ

#ਜੇ defined(__riscv)
ਬਣਤਰ ggml_riscv_arch_features_type {
    ਪੂਰਨ rvv_vlen;
} ggml_riscv_arch_features = { 0 };
#ਜੇ_ਅੰਤ

#ਜੇ defined(_WIN32)

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ WIN32_LEAN_AND_ਔਸਤ
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NOMINMAX
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NOMINMAX
#ਜੇ_ਅੰਤ
#ਅਨ੍ਤਰ੍ਭਾਵ <windows.h>

#ਜੇ defined(_MSC_VER) && !defined(__clang__)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_CACHE_ALIGN __declspec(align(GGML_CACHE_ਸਤਰ))

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਅਸਥਿਰ LONG atomic_int;
ਕਿਸਮਪਰਿਭਾਸ਼ਾ atomic_int atomic_bool;
ਕਿਸਮਪਰਿਭਾਸ਼ਾ atomic_int atomic_flag;

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ATOMIC_FLAG_INIT 0

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਗਿਣਤੀ {
    memory_order_relaxed,
    memory_order_consume,
    memory_order_acquire,
    memory_order_release,
    memory_order_acq_rel,
    memory_order_seq_cst
} memory_order;

ਸਥਿਰ ਖਾਲੀ atomic_store(atomic_int * ptr, LONG val) {
    InterlockedExchange(ptr, val);
}
ਸਥਿਰ ਖਾਲੀ atomic_store_explicit(atomic_int * ptr, LONG val, memory_order mo) {
    // TODO: add support for explicit memory order
    InterlockedExchange(ptr, val);
}
ਸਥਿਰ LONG atomic_ਲੱਦੋ(atomic_int * ptr) {
    ਵਾਪਸ InterlockedCompareExchange(ptr, 0, 0);
}
ਸਥਿਰ LONG atomic_ਲੱਦੋ_explicit(atomic_int * ptr, memory_order mo) {
    // TODO: add support for explicit memory order
    ਵਾਪਸ InterlockedCompareExchange(ptr, 0, 0);
}
ਸਥਿਰ LONG atomic_fetch_ਜੋੜੋ(atomic_int * ptr, LONG inc) {
    ਵਾਪਸ Interlocked_Exchange_ਜੋੜੋ(ptr, inc);
}
ਸਥਿਰ LONG atomic_fetch_ਜੋੜੋ_explicit(atomic_int * ptr, LONG inc, memory_order mo) {
    // TODO: add support for explicit memory order
    ਵਾਪਸ Interlocked_Exchange_ਜੋੜੋ(ptr, inc);
}
ਸਥਿਰ atomic_bool atomic_flag_test_and_ਸੈੱਟ(atomic_flag * ptr) {
    ਵਾਪਸ InterlockedExchange(ptr, 1);
}
ਸਥਿਰ ਖਾਲੀ atomic_flag_clear(atomic_flag * ptr) {
    InterlockedExchange(ptr, 0);
}
ਸਥਿਰ ਖਾਲੀ atomic_thread_fence(memory_order mo) {
    ਸਮ੍ਰਿਤੀ_Barrier();
}
#ਨਹੀਂਤਾਂ // clang
#ਅਨ੍ਤਰ੍ਭਾਵ <stdatomic.h>
#ਜੇ_ਅੰਤ

ਕਿਸਮਪਰਿਭਾਸ਼ਾ HANDLE pthread_t;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ DWORD thread_ret_t;
ਸਥਿਰ ਪੂਰਨ pthread_create(pthread_t * ਬਾਹਰਲਾ, ਖਾਲੀ * ਅਣਵਰਤਿਆ_2, thread_ret_t(*func)(ਖਾਲੀ *), ਖਾਲੀ * arg) {
    (ਖਾਲੀ) ਅਣਵਰਤਿਆ_2;
    HANDLE handle = CreateThread(ਨਿਰਾ, 0, (LPTHREAD_ਸ਼ੁਰੂ_ROUTINE) func, arg, 0, ਨਿਰਾ);
    ਜੇ (handle == ਨਿਰਾ)
    {
        ਵਾਪਸ EAGAIN;
    }

    *ਬਾਹਰਲਾ = handle;
    ਵਾਪਸ 0;
}

ਸਥਿਰ ਪੂਰਨ pthread_join(pthread_t thread, ਖਾਲੀ * ਅਣਵਰਤਿਆ_2) {
    (ਖਾਲੀ) ਅਣਵਰਤਿਆ_2;
    ਪੂਰਨ ret = (ਪੂਰਨ) WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    ਵਾਪਸ ret;
}

ਸਥਿਰ ਪੂਰਨ sched_yield (ਖਾਲੀ) {
    Sleep (0);
    ਵਾਪਸ 0;
}
#ਨਹੀਂਤਾਂ

#ਅਨ੍ਤਰ੍ਭਾਵ <pthread.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdatomic.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <sched.h>
#ਜੇ defined(__FreeBSD__)
#ਅਨ੍ਤਰ੍ਭਾਵ <pthread_np.h>
#ਜੇ_ਅੰਤ

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਖਾਲੀ * thread_ret_t;

#ਅਨ੍ਤਰ੍ਭਾਵ <sys/types.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <sys/stat.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <unistd.h>

#ਜੇ_ਅੰਤ

ਕਿਸਮਪਰਿਭਾਸ਼ਾ pthread_t ggml_thread_t;

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_THREADPOOL_N_ਧਾਗੇ_ਨਕਾਬ (0xffffU)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_THREADPOOL_N_ਧਾਗੇ_ਬਿੱਟ (16)

#ਜੇ defined(__APPLE__)
#ਅਨ੍ਤਰ੍ਭਾਵ <unistd.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <mach/mach.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <TargetConditionals.h>
#ਜੇ_ਅੰਤ

ਸਥਿਰ ਅਚਲ ਬਣਤਰ ggml_type_traits_cpu type_traits_cpu[GGML_TYPE_ਗਿਣਤੀ] = {
    [GGML_TYPE_F32] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_ਮੰਜ਼ਿਲ_fp32,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_f32,
        .vec_dot_type             = GGML_TYPE_F32,
        .nrows                    = 1,
    },
    [GGML_TYPE_F16] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_ਮੰਜ਼ਿਲ_fp16,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_f16,
        .vec_dot_type             = GGML_TYPE_F16,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q1_0] = {
        .from_float               = quantize_ਪੰਕਤੀ_q1_0,
        .vec_dot                  = ggml_vec_dot_q1_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q2_0] = {
        .from_float               = quantize_ਪੰਕਤੀ_q2_0,
        .vec_dot                  = ggml_vec_dot_q2_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q4_0] = {
        .from_float               = quantize_ਪੰਕਤੀ_q4_0,
        .vec_dot                  = ggml_vec_dot_q4_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
#ਜੇ defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#ਨਹੀਂਤਾਂ
        .nrows                    = 1,
#ਜੇ_ਅੰਤ
    },
    [GGML_TYPE_Q4_1] = {
        .from_float               = quantize_ਪੰਕਤੀ_q4_1,
        .vec_dot                  = ggml_vec_dot_q4_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
#ਜੇ defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#ਨਹੀਂਤਾਂ
        .nrows                    = 1,
#ਜੇ_ਅੰਤ
    },
    [GGML_TYPE_Q5_0] = {
        .from_float               = quantize_ਪੰਕਤੀ_q5_0,
        .vec_dot                  = ggml_vec_dot_q5_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q5_1] = {
        .from_float               = quantize_ਪੰਕਤੀ_q5_1,
        .vec_dot                  = ggml_vec_dot_q5_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q8_0] = {
        .from_float               = quantize_ਪੰਕਤੀ_q8_0,
        .vec_dot                  = ggml_vec_dot_q8_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
#ਜੇ defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#ਨਹੀਂਤਾਂ
        .nrows                    = 1,
#ਜੇ_ਅੰਤ
    },
    [GGML_TYPE_Q8_1] = {
        .from_float               = quantize_ਪੰਕਤੀ_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
        .nrows                    = 1,
    },
    [GGML_TYPE_MXFP4] = {
        .from_float               = quantize_ਪੰਕਤੀ_mxfp4,
        .vec_dot                  = ggml_vec_dot_mxfp4_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_NVFP4] = {
        .from_float               = quantize_ਪੰਕਤੀ_nvfp4,
        .vec_dot                  = ggml_vec_dot_nvfp4_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q2_K] = {
        .from_float               = quantize_ਪੰਕਤੀ_q2_K,
        .vec_dot                  = ggml_vec_dot_q2_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q3_K] = {
        .from_float               = quantize_ਪੰਕਤੀ_q3_K,
        .vec_dot                  = ggml_vec_dot_q3_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q4_K] = {
        .from_float               = quantize_ਪੰਕਤੀ_q4_K,
        .vec_dot                  = ggml_vec_dot_q4_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
#ਜੇ defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#ਨਹੀਂਤਾਂ
        .nrows                    = 1,
#ਜੇ_ਅੰਤ
    },
    [GGML_TYPE_Q5_K] = {
        .from_float               = quantize_ਪੰਕਤੀ_q5_K,
        .vec_dot                  = ggml_vec_dot_q5_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q6_K] = {
        .from_float               = quantize_ਪੰਕਤੀ_q6_K,
        .vec_dot                  = ggml_vec_dot_q6_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
#ਜੇ defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#ਨਹੀਂਤਾਂ
        .nrows                    = 1,
#ਜੇ_ਅੰਤ
    },
    [GGML_TYPE_IQ2_XXS] = {
        .from_float               = ਨਿਰਾ,
        .vec_dot                  = ggml_vec_dot_iq2_xxs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_XS] = {
        .from_float               = ਨਿਰਾ,
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
        .from_float               = ਨਿਰਾ,
        .vec_dot                  = ggml_vec_dot_iq1_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_M] = {
        .from_float               = ਨਿਰਾ,
        .vec_dot                  = ggml_vec_dot_iq1_m_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_NL] = {
        .from_float               = quantize_ਪੰਕਤੀ_iq4_nl,
        .vec_dot                  = ggml_vec_dot_iq4_nl_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_XS] = {
        .from_float               = quantize_ਪੰਕਤੀ_iq4_xs,
        .vec_dot                  = ggml_vec_dot_iq4_xs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q8_K] = {
        .from_float               = quantize_ਪੰਕਤੀ_q8_K,
    },
    [GGML_TYPE_BF16] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_ਮੰਜ਼ਿਲ_bf16,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_bf16,
        .vec_dot_type             = GGML_TYPE_BF16,
        .nrows                    = 1,
    },
    [GGML_TYPE_TQ1_0] = {
        .from_float               = quantize_ਪੰਕਤੀ_tq1_0,
        .vec_dot                  = ggml_vec_dot_tq1_0_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_TQ2_0] = {
        .from_float               = quantize_ਪੰਕਤੀ_tq2_0,
        .vec_dot                  = ggml_vec_dot_tq2_0_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_I32] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_ਮੰਜ਼ਿਲ_i32,
    },
};

ਅਚਲ ਬਣਤਰ ggml_type_traits_cpu * ggml_get_type_traits_cpu(ਗਿਣਤੀ ggml_type type) {
    ਵਾਪਸ &type_traits_cpu[type];
}

//
// Threading defs
//

ਕਿਸਮਪਰਿਭਾਸ਼ਾ pthread_t          ggml_thread_t;

#ਜੇ defined(_WIN32)

ਕਿਸਮਪਰਿਭਾਸ਼ਾ CONDITION_VARIABLE ggml_cond_t;
ਕਿਸਮਪਰਿਭਾਸ਼ਾ SRWLOCK            ggml_mutex_t;

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_mutex_init(m)   InitializeSRWLock(m)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_mutex_destroy(m)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_mutex_ਤਾਲਾ(m)   AcquireSRWLockExclusive(m)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_mutex_unlock(m) ReleaseSRWLockExclusive(m)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_mutex_ਤਾਲਾ_shared(m)   AcquireSRWLockShared(m)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_mutex_unlock_shared(m) ReleaseSRWLockShared(m)

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_cond_init(c)    InitializeConditionVariable(c)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_cond_destroy(c)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_cond_wait(c, m) SleepConditionVariableSRW(c, m, INFINITE, CONDITION_VARIABLE_LOCKMODE_SHARED)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_cond_ਪ੍ਰਸਾਰਣ(c) WakeAllConditionVariable(c)

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_thread_create pthread_create
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_thread_join   pthread_join

#ਨਹੀਂਤਾਂ

ਕਿਸਮਪਰਿਭਾਸ਼ਾ pthread_cond_t     ggml_cond_t;
ਕਿਸਮਪਰਿਭਾਸ਼ਾ pthread_mutex_t    ggml_mutex_t;

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_mutex_init(m)          pthread_mutex_init(m, ਨਿਰਾ)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_mutex_destroy(m)       pthread_mutex_destroy(m)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_mutex_ਤਾਲਾ(m)          pthread_mutex_ਤਾਲਾ(m)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_mutex_unlock(m)        pthread_mutex_unlock(m)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_mutex_ਤਾਲਾ_shared(m)   pthread_mutex_ਤਾਲਾ(m)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_mutex_unlock_shared(m) pthread_mutex_unlock(m)

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_ਤਾਲਾ_init(x)    ਅਣਵਰਤਿਆ(x)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_ਤਾਲਾ_destroy(x) ਅਣਵਰਤਿਆ(x)
#ਜੇ defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64) && !defined(_M_ARM64EC))
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_ਤਾਲਾ_ਤਾਲਾ(x)    _mm_pause()
#ਨਹੀਂਤਾਂ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_ਤਾਲਾ_ਤਾਲਾ(x)    ਅਣਵਰਤਿਆ(x)
#ਜੇ_ਅੰਤ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_ਤਾਲਾ_unlock(x)  ਅਣਵਰਤਿਆ(x)

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_ਤਾਲਾ_INITIALIZER 0
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_cond_init(c)      pthread_cond_init(c, ਨਿਰਾ)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_cond_destroy(c)   pthread_cond_destroy(c)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_cond_wait(c, m)   pthread_cond_wait(c, m)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_cond_ਪ੍ਰਸਾਰਣ(c) pthread_cond_ਪ੍ਰਸਾਰਣ(c)

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_thread_create pthread_create
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_thread_join   pthread_join

#ਜੇ_ਅੰਤ

// Threadpool def
ਬਣਤਰ ggml_threadpool {
    ggml_mutex_t mutex;       // mutex for cond.var
    ggml_cond_t  cond;        // cond.var for waiting for new work

    ਬਣਤਰ ggml_cgraph * cgraph;
    ਬਣਤਰ ggml_cplan  * cplan;

    // synchronization primitives
    atomic_int n_ਗ੍ਰਾਫ;       // updated when there is work to be done (i.e each graph) holds graph and active thread counts.
    atomic_int GGML_CACHE_ALIGN n_barrier;
    atomic_int GGML_CACHE_ALIGN n_barrier_passed;
    atomic_int GGML_CACHE_ALIGN current_chunk; // currently processing chunk during Mat_Mul, shared between all the threads.

    // these are atomic as an annotation for thread-sanitizer
    atomic_bool stop;         // Used for stopping the threadpool altogether
    atomic_bool pause;        // Used for pausing the threadpool or individual threads
    atomic_int  abort;        // Used for aborting processing of a graph

    ਬਣਤਰ ggml_compute_state * workers;   // per thread state
    ਪੂਰਨ          n_ਧਾਗੇ;   // Number of threads in the pool
    int32_t      prio;        // Scheduling priority
    uint32_t     poll;        // Polling level (0 - no polling)

    ਗਿਣਤੀ ggml_ਸਥਿਤੀ ec;
};

// Per-thread state
ਬਣਤਰ ggml_compute_state {
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_USE_OPENMP
    ggml_thread_t thrd;
    ਪੂਰਨ  ਆਖਰੀ_ਗ੍ਰਾਫ;
    bool pending;
#ਜੇ_ਅੰਤ
    bool cpumask[GGML_MAX_N_ਧਾਗੇ];
    ਬਣਤਰ ggml_threadpool * threadpool;
    ਪੂਰਨ ith;
};

// Helpers for polling loops
#ਜੇ defined(__aarch64__) && ( defined(__clang__) || defined(__GNUC__) )
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_thread_cpu_relax(ਖਾਲੀ) {
    __asm__ ਅਸਥਿਰ("yield" ::: "memory");
}
#ਜਾਂਜੇ defined(__x86_64__)
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_thread_cpu_relax(ਖਾਲੀ) {
    _mm_pause();
}
#ਜਾਂਜੇ defined(__riscv)
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_thread_cpu_relax(ਖਾਲੀ) {
    #ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __riscv_zihintpause
        __asm__ __volatile__ ("pause");
    #ਨਹੀਂਤਾਂ
        /* Encoding of the pause instruction */
        __asm__ __volatile__ (".4byte 0x100000F");
    #ਜੇ_ਅੰਤ
}
#ਨਹੀਂਤਾਂ
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_thread_cpu_relax(ਖਾਲੀ) {;}
#ਜੇ_ਅੰਤ

//
// NUMA support
//

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_NUMA_MAX_NODES 8
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_NUMA_MAX_CPUS 512

ਬਣਤਰ ggml_numa_ਗੰਢ {
    uint32_t cpus[GGML_NUMA_MAX_CPUS]; // hardware threads on this node
    uint32_t n_cpus;
};

ਬਣਤਰ ggml_numa_nodes {
    ਗਿਣਤੀ ggml_numa_strategy numa_strategy;
    ਬਣਤਰ ggml_numa_ਗੰਢ nodes[GGML_NUMA_MAX_NODES];
    uint32_t n_nodes;
    uint32_t ਕੁੱਲ_cpus; // hardware threads on system
    uint32_t current_ਗੰਢ; // node on which main process is execting
#ਜੇ defined(__gnu_linux__)
    cpu_ਸੈੱਟ_t cpuset; // cpuset from numactl
#ਨਹੀਂਤਾਂ
    uint32_t cpuset; // no NUMA support outside of Linux at this time. Use a portable datatype
#ਜੇ_ਅੰਤ
};

//
// ggml state
//

ਬਣਤਰ ggml_state {
    ਬਣਤਰ ggml_numa_nodes numa;
};

ਸਥਿਰ ਬਣਤਰ ggml_state g_state = {0};

ਖਾਲੀ ggml_barrier(ਬਣਤਰ ggml_threadpool * tp) {
    ਪੂਰਨ n_ਧਾਗੇ = atomic_ਲੱਦੋ_explicit(&tp->n_ਗ੍ਰਾਫ, memory_order_relaxed) & GGML_THREADPOOL_N_ਧਾਗੇ_ਨਕਾਬ;
    ਜੇ (n_ਧਾਗੇ == 1) {
        ਵਾਪਸ;
    }

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_OPENMP
    #ਹਦਾਇਤ omp barrier
#ਨਹੀਂਤਾਂ
    ਪੂਰਨ n_passed = atomic_ਲੱਦੋ_explicit(&tp->n_barrier_passed, memory_order_relaxed);

    // enter barrier (full seq-cst fence)
    ਪੂਰਨ n_barrier = atomic_fetch_ਜੋੜੋ_explicit(&tp->n_barrier, 1, memory_order_seq_cst);

    ਜੇ (n_barrier == (n_ਧਾਗੇ - 1)) {
        // last thread
        atomic_store_explicit(&tp->n_barrier, 0, memory_order_relaxed);

        // exit barrier (full seq-cst fence)
        atomic_fetch_ਜੋੜੋ_explicit(&tp->n_barrier_passed, 1, memory_order_seq_cst);
        ਵਾਪਸ;
    }

    // wait for other threads
    ਜਦੋਂ (atomic_ਲੱਦੋ_explicit(&tp->n_barrier_passed, memory_order_relaxed) == n_passed) {
        ggml_thread_cpu_relax();
    }

    // exit barrier (full seq-cst fence)
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_TSAN_ENABLED
    atomic_fetch_ਜੋੜੋ_explicit(&tp->n_barrier_passed, 0, memory_order_seq_cst);
    #ਨਹੀਂਤਾਂ
    atomic_thread_fence(memory_order_seq_cst);
    #ਜੇ_ਅੰਤ
#ਜੇ_ਅੰਤ
}

ਖਾਲੀ ggml_threadpool_chunk_ਸੈੱਟ(ਬਣਤਰ ggml_threadpool * tp, ਪੂਰਨ ਮੁੱਲ) {
    atomic_store_explicit(&tp->current_chunk, ਮੁੱਲ, memory_order_relaxed);
}

ਪੂਰਨ ggml_threadpool_chunk_ਜੋੜੋ(ਬਣਤਰ ggml_threadpool * tp, ਪੂਰਨ ਮੁੱਲ) {
    ਵਾਪਸ atomic_fetch_ਜੋੜੋ_explicit(&tp->current_chunk, ਮੁੱਲ, memory_order_relaxed);
}

#ਜੇ defined(__gnu_linux__)
ਸਥਿਰ cpu_ਸੈੱਟ_t ggml_get_numa_affinity(ਖਾਲੀ) {
    cpu_ਸੈੱਟ_t cpuset;
    pthread_t thread;
    thread = pthread_self();
    CPU_ZERO(&cpuset);
    pthread_getaffinity_np(thread, ਆਕਾਰ(cpu_ਸੈੱਟ_t), &cpuset);
    ਵਾਪਸ cpuset;
}
#ਨਹੀਂਤਾਂ
ਸਥਿਰ uint32_t ggml_get_numa_affinity(ਖਾਲੀ) {
    ਵਾਪਸ 0; // no NUMA support
}
#ਜੇ_ਅੰਤ

ਖਾਲੀ ggml_numa_init(ਗਿਣਤੀ ggml_numa_strategy numa_flag) {
    ਜੇ (g_state.numa.n_nodes > 0) {
        fprintf(stderr, "ggml_numa_init: NUMA already initialized\n");

        ਵਾਪਸ;
    }

#ਜੇ defined(__gnu_linux__)
    ਬਣਤਰ stat st;
    ਅੱਖਰ ਰਸਤਾ[256];
    ਪੂਰਨ rv;

    // set numa scheme
    g_state.numa.numa_strategy = numa_flag;

    GGML_PRINT_DEBUG("numa strategy %u\n",g_state.numa.numa_strategy);

    g_state.numa.cpuset = ggml_get_numa_affinity();

    // enumerate nodes
    ਜਦੋਂ (g_state.numa.n_nodes < GGML_NUMA_MAX_NODES) {
        rv = snprintf(ਰਸਤਾ, ਆਕਾਰ(ਰਸਤਾ), "/sys/devices/system/node/node%u", g_state.numa.n_nodes);
        GGML_ASSERT(rv > 0 && (ਬਿਨਾਨਿਸ਼ਾਨੀ)rv < ਆਕਾਰ(ਰਸਤਾ));
        ਜੇ (stat(ਰਸਤਾ, &st) != 0) { ਰੋਕੋ; }
        ++g_state.numa.n_nodes;
    }

    // enumerate CPUs
    ਜਦੋਂ (g_state.numa.ਕੁੱਲ_cpus < GGML_NUMA_MAX_CPUS) {
        rv = snprintf(ਰਸਤਾ, ਆਕਾਰ(ਰਸਤਾ), "/sys/devices/system/cpu/cpu%u", g_state.numa.ਕੁੱਲ_cpus);
        GGML_ASSERT(rv > 0 && (ਬਿਨਾਨਿਸ਼ਾਨੀ)rv < ਆਕਾਰ(ਰਸਤਾ));
        ਜੇ (stat(ਰਸਤਾ, &st) != 0) { ਰੋਕੋ; }
        ++g_state.numa.ਕੁੱਲ_cpus;
    }

    GGML_PRINT_DEBUG("found %u numa nodes, %u CPUs\n", g_state.numa.n_nodes, g_state.numa.ਕੁੱਲ_cpus);

    // figure out which node we're on
    uint current_cpu;
    ਪੂਰਨ getcpu_ret = 0;
#ਜੇ __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ > 33) || defined(__COSMOPOLITAN__)
    getcpu_ret = getcpu(&current_cpu, &g_state.numa.current_ਗੰਢ);
#ਨਹੀਂਤਾਂ
    // old glibc doesn't have a wrapper for this call. Fall back on direct syscall
#   ਜੇ !defined(SYS_getcpu) && defined(SYS_get_cpu)
#       define SYS_getcpu SYS_get_cpu // some older glibc versions use this name
#   endif
    getcpu_ret = syscall(SYS_getcpu, &current_cpu, &g_state.numa.current_ਗੰਢ);
#ਜੇ_ਅੰਤ

    ਜੇ (g_state.numa.n_nodes < 1 || g_state.numa.ਕੁੱਲ_cpus < 1 || getcpu_ret != 0) {
        g_state.numa.n_nodes = 0;
        ਵਾਪਸ;
    }

    GGML_PRINT_DEBUG("found our process on numa node %u, CPU %u\n", g_state.numa.current_ਗੰਢ, current_cpu);

    ਲਈ (uint32_t n = 0; n < g_state.numa.n_nodes; ++n) {
        ਬਣਤਰ ggml_numa_ਗੰਢ * ਗੰਢ = &g_state.numa.nodes[n];
        GGML_PRINT_DEBUG("CPUs on node %u:", n);
        ਗੰਢ->n_cpus = 0;
        ਲਈ (uint32_t c = 0; c < g_state.numa.ਕੁੱਲ_cpus; ++c) {
            rv = snprintf(ਰਸਤਾ, ਆਕਾਰ(ਰਸਤਾ), "/sys/devices/system/node/node%u/cpu%u", n, c);
            GGML_ASSERT(rv > 0 && (ਬਿਨਾਨਿਸ਼ਾਨੀ)rv < ਆਕਾਰ(ਰਸਤਾ));
            ਜੇ (stat(ਰਸਤਾ, &st) == 0) {
                ਗੰਢ->cpus[ਗੰਢ->n_cpus++] = c;
                GGML_PRINT_DEBUG(" %u", c);
            }
        }
        GGML_PRINT_DEBUG("\n");
    }

    ਜੇ (ggml_is_numa()) {
        FILE *fptr = fopen("/proc/sys/kernel/numa_balancing", "r");
        ਜੇ (fptr != ਨਿਰਾ) {
            ਅੱਖਰ buf[42];
            ਜੇ (fgets(buf, ਆਕਾਰ(buf), fptr) && ਲੜੀ_ਤੁਲਨਾ(buf, "0\n", ਆਕਾਰ(buf)) != 0) {
                GGML_LOG_WARN("/proc/sys/kernel/numa_balancing is enabled, this has been observed to impair performance\n");
            }
            fclose(fptr);
        }
    }
#ਨਹੀਂਤਾਂ
    ਅਣਵਰਤਿਆ(numa_flag);
    // TODO
#ਜੇ_ਅੰਤ
}

bool ggml_is_numa(ਖਾਲੀ) {
    ਵਾਪਸ g_state.numa.n_nodes > 1;
}

#ਜੇ defined(__ARM_ARCH)
#ਜੇ defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
#ਅਨ੍ਤਰ੍ਭਾਵ <arm_sve.h>
ਸਥਿਰ ਖਾਲੀ ggml_init_arm_arch_features(ਖਾਲੀ) {
    ggml_arm_arch_features.sve_cnt = svcntb();
}
#ਨਹੀਂਤਾਂ
ਸਥਿਰ ਖਾਲੀ ggml_init_arm_arch_features(ਖਾਲੀ) {}
#ਜੇ_ਅੰਤ
#ਜੇ_ਅੰਤ // __ARM_ARCH

#ਜੇ defined(__riscv) && defined(__riscv_v_intrinsic)
#ਅਨ੍ਤਰ੍ਭਾਵ <riscv_vector.h>
ਸਥਿਰ ਖਾਲੀ ggml_init_riscv_arch_features(ਖਾਲੀ) {
    ggml_riscv_arch_features.rvv_vlen = __riscv_vlenb();
}
#ਨਹੀਂਤਾਂ
ਸਥਿਰ ਖਾਲੀ ggml_init_riscv_arch_features(ਖਾਲੀ) {}
#ਜੇ_ਅੰਤ

ਬਣਤਰ ggml_tensor * ggml_new_i32(ਬਣਤਰ ggml_context * ctx, int32_t ਮੁੱਲ) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

    ggml_ਸੈੱਟ_i32(ਨਤੀਜਾ, ਮੁੱਲ);

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_new_f32(ਬਣਤਰ ggml_context * ctx, ਦਸ਼ਮਲਵ ਮੁੱਲ) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    ggml_ਸੈੱਟ_f32(ਨਤੀਜਾ, ਮੁੱਲ);

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਬਣਤਰ ggml_tensor * ggml_ਸੈੱਟ_i32 (ਬਣਤਰ ggml_tensor * tensor, int32_t ਮੁੱਲ) {
    ਅਚਲ ਪੂਰਨ n     = ggml_nrows(tensor);
    ਅਚਲ ਪੂਰਨ nc    = tensor->ne[0];
    ਅਚਲ size_t n1 = tensor->nb[1];

    ਅੱਖਰ * ਅਚਲ ਅੰਕੜੇ = tensor->ਅੰਕੜੇ;

    ਚੋਣ (tensor->type) {
        ਹਾਲਤ GGML_TYPE_I8:
            {
                ਪ੍ਰਤਿਜ੍ਞਾ(tensor->nb[0] == ਆਕਾਰ(int8_t));
                ਲਈ (ਪੂਰਨ i = 0; i < n; i++) {
                    ggml_vec_ਸੈੱਟ_i8(nc, (int8_t *)(ਅੰਕੜੇ + i*n1), ਮੁੱਲ);
                }
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_I16:
            {
                ਪ੍ਰਤਿਜ੍ਞਾ(tensor->nb[0] == ਆਕਾਰ(int16_t));
                ਲਈ (ਪੂਰਨ i = 0; i < n; i++) {
                    ggml_vec_ਸੈੱਟ_i16(nc, (int16_t *)(ਅੰਕੜੇ + i*n1), ਮੁੱਲ);
                }
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_I32:
            {
                ਪ੍ਰਤਿਜ੍ਞਾ(tensor->nb[0] == ਆਕਾਰ(int32_t));
                ਲਈ (ਪੂਰਨ i = 0; i < n; i++) {
                    ggml_vec_ਸੈੱਟ_i32(nc, (int32_t *)(ਅੰਕੜੇ + i*n1), ਮੁੱਲ);
                }
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F16:
            {
                ਪ੍ਰਤਿਜ੍ਞਾ(tensor->nb[0] == ਆਕਾਰ(ggml_fp16_t));
                ਲਈ (ਪੂਰਨ i = 0; i < n; i++) {
                    ggml_vec_ਸੈੱਟ_f16(nc, (ggml_fp16_t *)(ਅੰਕੜੇ + i*n1), GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(ਮੁੱਲ));
                }
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_BF16:
            {
                ਪ੍ਰਤਿਜ੍ਞਾ(tensor->nb[0] == ਆਕਾਰ(ggml_fp16_t));
                ਲਈ (ਪੂਰਨ i = 0; i < n; i++) {
                    ggml_vec_ਸੈੱਟ_bf16(nc, (ggml_bf16_t *)(ਅੰਕੜੇ + i*n1), GGML_FP32_ਮੰਜ਼ਿਲ_BF16(ਮੁੱਲ));
                }
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ਪ੍ਰਤਿਜ੍ਞਾ(tensor->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                ਲਈ (ਪੂਰਨ i = 0; i < n; i++) {
                    ggml_vec_ਸੈੱਟ_f32(nc, (ਦਸ਼ਮਲਵ *)(ਅੰਕੜੇ + i*n1), ਮੁੱਲ);
                }
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }

    ਵਾਪਸ tensor;
}

ਬਣਤਰ ggml_tensor * ggml_ਸੈੱਟ_f32(ਬਣਤਰ ggml_tensor * tensor, ਦਸ਼ਮਲਵ ਮੁੱਲ) {
    ਅਚਲ ਪੂਰਨ n     = ggml_nrows(tensor);
    ਅਚਲ ਪੂਰਨ nc    = tensor->ne[0];
    ਅਚਲ size_t n1 = tensor->nb[1];

    ਅੱਖਰ * ਅਚਲ ਅੰਕੜੇ = tensor->ਅੰਕੜੇ;

    ਚੋਣ (tensor->type) {
        ਹਾਲਤ GGML_TYPE_I8:
            {
                ਪ੍ਰਤਿਜ੍ਞਾ(tensor->nb[0] == ਆਕਾਰ(int8_t));
                ਲਈ (ਪੂਰਨ i = 0; i < n; i++) {
                    ggml_vec_ਸੈੱਟ_i8(nc, (int8_t *)(ਅੰਕੜੇ + i*n1), ਮੁੱਲ);
                }
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_I16:
            {
                ਪ੍ਰਤਿਜ੍ਞਾ(tensor->nb[0] == ਆਕਾਰ(int16_t));
                ਲਈ (ਪੂਰਨ i = 0; i < n; i++) {
                    ggml_vec_ਸੈੱਟ_i16(nc, (int16_t *)(ਅੰਕੜੇ + i*n1), ਮੁੱਲ);
                }
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_I32:
            {
                ਪ੍ਰਤਿਜ੍ਞਾ(tensor->nb[0] == ਆਕਾਰ(int32_t));
                ਲਈ (ਪੂਰਨ i = 0; i < n; i++) {
                    ggml_vec_ਸੈੱਟ_i32(nc, (int32_t *)(ਅੰਕੜੇ + i*n1), ਮੁੱਲ);
                }
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F16:
            {
                ਪ੍ਰਤਿਜ੍ਞਾ(tensor->nb[0] == ਆਕਾਰ(ggml_fp16_t));
                ਲਈ (ਪੂਰਨ i = 0; i < n; i++) {
                    ggml_vec_ਸੈੱਟ_f16(nc, (ggml_fp16_t *)(ਅੰਕੜੇ + i*n1), GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(ਮੁੱਲ));
                }
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_BF16:
            {
                ਪ੍ਰਤਿਜ੍ਞਾ(tensor->nb[0] == ਆਕਾਰ(ggml_bf16_t));
                ਲਈ (ਪੂਰਨ i = 0; i < n; i++) {
                    ggml_vec_ਸੈੱਟ_bf16(nc, (ggml_bf16_t *)(ਅੰਕੜੇ + i*n1), GGML_FP32_ਮੰਜ਼ਿਲ_BF16(ਮੁੱਲ));
                }
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ਪ੍ਰਤਿਜ੍ਞਾ(tensor->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                ਲਈ (ਪੂਰਨ i = 0; i < n; i++) {
                    ggml_vec_ਸੈੱਟ_f32(nc, (ਦਸ਼ਮਲਵ *)(ਅੰਕੜੇ + i*n1), ਮੁੱਲ);
                }
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }

    ਵਾਪਸ tensor;
}

int32_t ggml_get_i32_1d(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor, ਪੂਰਨ i) {
    ਜੇ (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_ਸੂਚਕਾਂਕ(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ਵਾਪਸ ggml_get_i32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    ਚੋਣ (tensor->type) {
        ਹਾਲਤ GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == ਆਕਾਰ(int8_t));
                ਵਾਪਸ ((int8_t *)(tensor->ਅੰਕੜੇ))[i];
            }
        ਹਾਲਤ GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == ਆਕਾਰ(int16_t));
                ਵਾਪਸ ((int16_t *)(tensor->ਅੰਕੜੇ))[i];
            }
        ਹਾਲਤ GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == ਆਕਾਰ(int32_t));
                ਵਾਪਸ ((int32_t *)(tensor->ਅੰਕੜੇ))[i];
            }
        ਹਾਲਤ GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == ਆਕਾਰ(ggml_fp16_t));
                ਵਾਪਸ GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(((ggml_fp16_t *)(tensor->ਅੰਕੜੇ))[i]);
            }
        ਹਾਲਤ GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == ਆਕਾਰ(ggml_bf16_t));
                ਵਾਪਸ GGML_BF16_ਮੰਜ਼ਿਲ_FP32(((ggml_bf16_t *)(tensor->ਅੰਕੜੇ))[i]);
            }
        ਹਾਲਤ GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                ਵਾਪਸ ((ਦਸ਼ਮਲਵ *)(tensor->ਅੰਕੜੇ))[i];
            }
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ਖਾਲੀ ggml_ਸੈੱਟ_i32_1d(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor, ਪੂਰਨ i, int32_t ਮੁੱਲ) {
    ਜੇ (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_ਸੂਚਕਾਂਕ(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_ਸੈੱਟ_i32_nd(tensor, id[0], id[1], id[2], id[3], ਮੁੱਲ);
        ਵਾਪਸ;
    }
    ਚੋਣ (tensor->type) {
        ਹਾਲਤ GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == ਆਕਾਰ(int8_t));
                ((int8_t *)(tensor->ਅੰਕੜੇ))[i] = ਮੁੱਲ;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == ਆਕਾਰ(int16_t));
                ((int16_t *)(tensor->ਅੰਕੜੇ))[i] = ਮੁੱਲ;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == ਆਕਾਰ(int32_t));
                ((int32_t *)(tensor->ਅੰਕੜੇ))[i] = ਮੁੱਲ;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == ਆਕਾਰ(ggml_fp16_t));
                ((ggml_fp16_t *)(tensor->ਅੰਕੜੇ))[i] = GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(ਮੁੱਲ);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == ਆਕਾਰ(ggml_bf16_t));
                ((ggml_bf16_t *)(tensor->ਅੰਕੜੇ))[i] = GGML_FP32_ਮੰਜ਼ਿਲ_BF16(ਮੁੱਲ);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                ((ਦਸ਼ਮਲਵ *)(tensor->ਅੰਕੜੇ))[i] = ਮੁੱਲ;
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

int32_t ggml_get_i32_nd(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor, ਪੂਰਨ i0, ਪੂਰਨ i1, ਪੂਰਨ i2, ਪੂਰਨ i3) {
    ਖਾਲੀ * ਅੰਕੜੇ   = (ਅੱਖਰ *) tensor->ਅੰਕੜੇ + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    ਚੋਣ (tensor->type) {
        ਹਾਲਤ GGML_TYPE_I8:
            ਵਾਪਸ ((int8_t *) ਅੰਕੜੇ)[0];
        ਹਾਲਤ GGML_TYPE_I16:
            ਵਾਪਸ ((int16_t *) ਅੰਕੜੇ)[0];
        ਹਾਲਤ GGML_TYPE_I32:
            ਵਾਪਸ ((int32_t *) ਅੰਕੜੇ)[0];
        ਹਾਲਤ GGML_TYPE_F16:
            ਵਾਪਸ GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(((ggml_fp16_t *) ਅੰਕੜੇ)[0]);
        ਹਾਲਤ GGML_TYPE_BF16:
            ਵਾਪਸ GGML_BF16_ਮੰਜ਼ਿਲ_FP32(((ggml_bf16_t *) ਅੰਕੜੇ)[0]);
        ਹਾਲਤ GGML_TYPE_F32:
            ਵਾਪਸ ((ਦਸ਼ਮਲਵ *) ਅੰਕੜੇ)[0];
        ਮੂਲ:
            GGML_ABORT("fatal error");
    }
}

ਖਾਲੀ ggml_ਸੈੱਟ_i32_nd(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor, ਪੂਰਨ i0, ਪੂਰਨ i1, ਪੂਰਨ i2, ਪੂਰਨ i3, int32_t ਮੁੱਲ) {
    ਖਾਲੀ * ਅੰਕੜੇ   = (ਅੱਖਰ *) tensor->ਅੰਕੜੇ + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    ਚੋਣ (tensor->type) {
        ਹਾਲਤ GGML_TYPE_I8:
            {
                ((int8_t *)(ਅੰਕੜੇ))[0] = ਮੁੱਲ;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_I16:
            {
                ((int16_t *)(ਅੰਕੜੇ))[0] = ਮੁੱਲ;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_I32:
            {
                ((int32_t *)(ਅੰਕੜੇ))[0] = ਮੁੱਲ;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(ਅੰਕੜੇ))[0] = GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(ਮੁੱਲ);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(ਅੰਕੜੇ))[0] = GGML_FP32_ਮੰਜ਼ਿਲ_BF16(ਮੁੱਲ);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ((ਦਸ਼ਮਲਵ *)(ਅੰਕੜੇ))[0] = ਮੁੱਲ;
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ਦਸ਼ਮਲਵ ggml_get_f32_1d(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor, ਪੂਰਨ i) {
    ਜੇ (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_ਸੂਚਕਾਂਕ(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ਵਾਪਸ ggml_get_f32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    ਚੋਣ (tensor->type) {
        ਹਾਲਤ GGML_TYPE_I8:
            {
                ਵਾਪਸ ((int8_t *)(tensor->ਅੰਕੜੇ))[i];
            }
        ਹਾਲਤ GGML_TYPE_I16:
            {
                ਵਾਪਸ ((int16_t *)(tensor->ਅੰਕੜੇ))[i];
            }
        ਹਾਲਤ GGML_TYPE_I32:
            {
                ਵਾਪਸ ((int32_t *)(tensor->ਅੰਕੜੇ))[i];
            }
        ਹਾਲਤ GGML_TYPE_F16:
            {
                ਵਾਪਸ GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(((ggml_fp16_t *)(tensor->ਅੰਕੜੇ))[i]);
            }
        ਹਾਲਤ GGML_TYPE_BF16:
            {
                ਵਾਪਸ GGML_BF16_ਮੰਜ਼ਿਲ_FP32(((ggml_bf16_t *)(tensor->ਅੰਕੜੇ))[i]);
            }
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ਵਾਪਸ ((ਦਸ਼ਮਲਵ *)(tensor->ਅੰਕੜੇ))[i];
            }
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ਖਾਲੀ ggml_ਸੈੱਟ_f32_1d(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor, ਪੂਰਨ i, ਦਸ਼ਮਲਵ ਮੁੱਲ) {
    ਜੇ (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_ਸੂਚਕਾਂਕ(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_ਸੈੱਟ_f32_nd(tensor, id[0], id[1], id[2], id[3], ਮੁੱਲ);
        ਵਾਪਸ;
    }
    ਚੋਣ (tensor->type) {
        ਹਾਲਤ GGML_TYPE_I8:
            {
                ((int8_t *)(tensor->ਅੰਕੜੇ))[i] = ਮੁੱਲ;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_I16:
            {
                ((int16_t *)(tensor->ਅੰਕੜੇ))[i] = ਮੁੱਲ;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_I32:
            {
                ((int32_t *)(tensor->ਅੰਕੜੇ))[i] = ਮੁੱਲ;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(tensor->ਅੰਕੜੇ))[i] = GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(ਮੁੱਲ);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(tensor->ਅੰਕੜੇ))[i] = GGML_FP32_ਮੰਜ਼ਿਲ_BF16(ਮੁੱਲ);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ((ਦਸ਼ਮਲਵ *)(tensor->ਅੰਕੜੇ))[i] = ਮੁੱਲ;
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ਦਸ਼ਮਲਵ ggml_get_f32_nd(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor, ਪੂਰਨ i0, ਪੂਰਨ i1, ਪੂਰਨ i2, ਪੂਰਨ i3) {
    ਖਾਲੀ * ਅੰਕੜੇ   = (ਅੱਖਰ *) tensor->ਅੰਕੜੇ + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    ਚੋਣ (tensor->type) {
        ਹਾਲਤ GGML_TYPE_I8:
            ਵਾਪਸ ((int8_t *) ਅੰਕੜੇ)[0];
        ਹਾਲਤ GGML_TYPE_I16:
            ਵਾਪਸ ((int16_t *) ਅੰਕੜੇ)[0];
        ਹਾਲਤ GGML_TYPE_I32:
            ਵਾਪਸ ((int32_t *) ਅੰਕੜੇ)[0];
        ਹਾਲਤ GGML_TYPE_F16:
            ਵਾਪਸ GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(((ggml_fp16_t *) ਅੰਕੜੇ)[0]);
        ਹਾਲਤ GGML_TYPE_BF16:
            ਵਾਪਸ GGML_BF16_ਮੰਜ਼ਿਲ_FP32(((ggml_bf16_t *) ਅੰਕੜੇ)[0]);
        ਹਾਲਤ GGML_TYPE_F32:
            ਵਾਪਸ ((ਦਸ਼ਮਲਵ *) ਅੰਕੜੇ)[0];
        ਮੂਲ:
            GGML_ABORT("fatal error");
    }
}

ਖਾਲੀ ggml_ਸੈੱਟ_f32_nd(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor, ਪੂਰਨ i0, ਪੂਰਨ i1, ਪੂਰਨ i2, ਪੂਰਨ i3, ਦਸ਼ਮਲਵ ਮੁੱਲ) {
    ਖਾਲੀ * ਅੰਕੜੇ   = (ਅੱਖਰ *) tensor->ਅੰਕੜੇ + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    ਚੋਣ (tensor->type) {
        ਹਾਲਤ GGML_TYPE_I8:
            {
                ((int8_t *)(ਅੰਕੜੇ))[0] = ਮੁੱਲ;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_I16:
            {
                ((int16_t *)(ਅੰਕੜੇ))[0] = ਮੁੱਲ;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_I32:
            {
                ((int32_t *)(ਅੰਕੜੇ))[0] = ਮੁੱਲ;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(ਅੰਕੜੇ))[0] = GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(ਮੁੱਲ);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(ਅੰਕੜੇ))[0] = GGML_FP32_ਮੰਜ਼ਿਲ_BF16(ਮੁੱਲ);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ((ਦਸ਼ਮਲਵ *)(ਅੰਕੜੇ))[0] = ਮੁੱਲ;
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

////////////////////////////////////////////////////////////////////////////////

// ggml_compute_forward_mul_mat

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_mul_mat_one_chunk(
    ਅਚਲ ਬਣਤਰ ggml_compute_params * params,
    ਬਣਤਰ ggml_tensor * dst,
    ਅਚਲ ਗਿਣਤੀ ggml_type type,
    ਅਚਲ int64_t ਅੰਕ_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_vec_dot,
    ਅਚਲ int64_t ir0_ਸ਼ੁਰੂ,
    ਅਚਲ int64_t ir0_ਸਿਰਾ,
    ਅਚਲ int64_t ir1_ਸ਼ੁਰੂ,
    ਅਚਲ int64_t ir1_ਸਿਰਾ) {

    ਅਚਲ ਬਣਤਰ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ਬਣਤਰ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

    ਅਚਲ bool src1_cont = ggml_is_contiguous(src1);

    ggml_vec_dot_t ਅਚਲ vec_dot      = type_traits_cpu[type].vec_dot;
    ਗਿਣਤੀ ggml_type ਅਚਲ vec_dot_type = type_traits_cpu[type].vec_dot_type;

    // broadcast factors
    ਅਚਲ int64_t r2 = ne12 / ne02;
    ਅਚਲ int64_t r3 = ne13 / ne03;

    //printf("ir0_start = %6lld, ir0_end = %6lld, ir1_start = %6lld, ir1_end = %6lld\n", ir0_start, ir0_end, ir1_start, ir1_end);

    // threads with no work simply yield (not sure if it helps)
    ਜੇ (ir0_ਸ਼ੁਰੂ >= ir0_ਸਿਰਾ || ir1_ਸ਼ੁਰੂ >= ir1_ਸਿਰਾ) {
        ਵਾਪਸ;
    }

    ਅਚਲ ਖਾਲੀ * wdata = (src1->type == vec_dot_type) ? src1->ਅੰਕੜੇ : params->wdata;
    ਅਚਲ size_t ਪੰਕਤੀ_ਮਾਪ = ggml_ਪੰਕਤੀ_ਮਾਪ(vec_dot_type, ne10);

    ਪ੍ਰਤਿਜ੍ਞਾ(ne12 % ne02 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(ne13 % ne03 == 0);

    // block-tiling attempt
    ਅਚਲ int64_t blck_0 = 16;
    ਅਚਲ int64_t blck_1 = 16;

    ਅਚਲ size_t src1_ਖਾਨਾ_stride = src1_cont || src1->type != vec_dot_type ? ਪੰਕਤੀ_ਮਾਪ : nb11;

    // attempt to reduce false-sharing (does not seem to make a difference)
    // 16 * 2, accounting for mmla kernels
    ਦਸ਼ਮਲਵ tmp[32];

    ਲਈ (int64_t iir1 = ir1_ਸ਼ੁਰੂ; iir1 < ir1_ਸਿਰਾ; iir1 += blck_1) {
        ਲਈ (int64_t iir0 = ir0_ਸ਼ੁਰੂ; iir0 < ir0_ਸਿਰਾ; iir0 += blck_0) {
            ਲਈ (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_ਸਿਰਾ; ir1 += ਅੰਕ_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_vec_dot) {
                ਅਚਲ int64_t i13 = (ir1 / (ne12 * ne1));
                ਅਚਲ int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                ਅਚਲ int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                // broadcast src0 into src1
                ਅਚਲ int64_t i03 = i13 / r3;
                ਅਚਲ int64_t i02 = i12 / r2;

                ਅਚਲ int64_t i1 = i11;
                ਅਚਲ int64_t i2 = i12;
                ਅਚਲ int64_t i3 = i13;

                ਅਚਲ ਅੱਖਰ * src0_ਪੰਕਤੀ = (ਅਚਲ ਅੱਖਰ*)src0->ਅੰਕੜੇ + (0 + i02 * nb02 + i03 * nb03);

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                ਅਚਲ ਅੱਖਰ * src1_ਖਾਨਾ = (ਅਚਲ ਅੱਖਰ*)wdata +
                    (src1_cont || src1->type != vec_dot_type
                        ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * ਪੰਕਤੀ_ਮਾਪ
                        : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                ਦਸ਼ਮਲਵ * dst_ਖਾਨਾ = (ਦਸ਼ਮਲਵ*)((ਅੱਖਰ*)dst->ਅੰਕੜੇ + (i1 * nb1 + i2 * nb2 + i3 * nb3));

                //for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                //    vec_dot(ne00, &dst_col[ir0], src0_row + ir0*nb01, src1_col);
                //}

                ਲਈ (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_ਸਿਰਾ; ir0 += ਅੰਕ_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_vec_dot) {
                    vec_dot(ne00, &tmp[ir0 - iir0], (ਅੰਕ_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_vec_dot > 1 ? 16 : 0), src0_ਪੰਕਤੀ + ir0 * nb01, (ਅੰਕ_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_vec_dot > 1 ? nb01 : 0), src1_ਖਾਨਾ, (ਅੰਕ_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_vec_dot > 1 ? src1_ਖਾਨਾ_stride : 0), ਅੰਕ_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_vec_dot);
                }

                ਲਈ (ਪੂਰਨ cn = 0; cn < ਅੰਕ_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_vec_dot; ++cn) {
                    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&dst_ਖਾਨਾ[iir0 + cn * nb1 / nb0], tmp + (cn * 16), (MIN(iir0 + blck_0, ir0_ਸਿਰਾ) - iir0) * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                }
            }
        }
    }
}

ਖਾਲੀ ggml_compute_forward_mul_mat(
        ਅਚਲ ਬਣਤਰ ggml_compute_params * params,
              ਬਣਤਰ ggml_tensor * dst) {

    ਅਚਲ ਬਣਤਰ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ਬਣਤਰ ggml_tensor * src1 = dst->src[1];

    ਅਚਲ int32_t hint = ggml_get_op_params_i32(dst, 1);
    ਜੇ (hint == GGML_HINT_SRC0_IS_HADAMARD && !params->use_ref) {
        ggml_compute_forward_fwht(params, dst);
        ਵਾਪਸ;
    }

    // If tiled is supported, it will execute the full op here and we return
    ਜੇ (ggml_compute_forward_mul_mat_tiled(params, dst)) {
        ਵਾਪਸ;
    }

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਗਿਣਤੀ ggml_type           ਅਚਲ vec_dot_type         = type_traits_cpu[src0->type].vec_dot_type;
    ggml_from_float_t        ਅਚਲ from_float           = type_traits_cpu[vec_dot_type].from_float;
    int64_t                  ਅਚਲ vec_dot_ਅੰਕ_ਪੰਕਤੀਆਂ     = type_traits_cpu[src0->type].nrows;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_ਮਾਪ(src0->type));
    GGML_ASSERT(nb10 == ggml_type_ਮਾਪ(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    // TODO: extract to "extra_op"
#ਜੇ GGML_USE_LLAMAFILE
    // broadcast factors
    ਅਚਲ int64_t r2 = ne12 / ne02;
    ਅਚਲ int64_t r3 = ne13 / ne03;

    ਅਚਲ bool src1_cont = ggml_is_contiguous(src1);

    ਜੇ (!params->use_ref && src1_cont) {
        ਲਈ (int64_t i13 = 0; i13 < ne13; i13++)
            ਲਈ (int64_t i12 = 0; i12 < ne12; i12++)
                ਜੇ (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_ਮਾਪ(src0->type),
                                     (ਅਚਲ ਅੱਖਰ *)src0->ਅੰਕੜੇ + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_ਮਾਪ(src0->type),
                                     (ਅਚਲ ਅੱਖਰ *)src1->ਅੰਕੜੇ + i12*nb12 + i13*nb13,
                                     nb11/ggml_type_ਮਾਪ(src1->type),
                                     (ਅੱਖਰ *)dst->ਅੰਕੜੇ + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_ਮਾਪ(dst->type),
                                     src0->type,
                                     src1->type,
                                     dst->type))
                    ਜਾਓ UseGgmlGemm1;
        ਵਾਪਸ;
    }
UseGgmlGemm1:;
#ਜੇ_ਅੰਤ

    ਜੇ (src1->type != vec_dot_type) {
        ਅੱਖਰ * wdata = params->wdata;

        ਅਚਲ size_t nbw0 = ggml_type_ਮਾਪ(vec_dot_type);
        ਅਚਲ size_t nbw1 = ggml_ਪੰਕਤੀ_ਮਾਪ(vec_dot_type, ne10);
        ਅਚਲ size_t nbw2 = nbw1*ne11;
        ਅਚਲ size_t nbw3 = nbw2*ne12;

        ਪ੍ਰਤਿਜ੍ਞਾ(params->wsize >= ne13*nbw3);
        // src1 is either packed from F32 into vec_dot_type, or widened from F16 or BF16 into the F32 work buffer
        ਅਚਲ bool widen = src1->type != GGML_TYPE_F32;

        GGML_ASSERT(!widen || vec_dot_type == GGML_TYPE_F32);
        GGML_ASSERT(!widen || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_BF16);

    #ਜੇ 0
        ਲਈ (int64_t i13 = 0; i13 < ne13; ++i13) {
            ਲਈ (int64_t i12 = 0; i12 < ne12; ++i12) {
                ਲਈ (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                    from_float((ਦਸ਼ਮਲਵ *)((ਅੱਖਰ *) src1->ਅੰਕੜੇ + i13*nb13 + i12*nb12 + i11*nb11),
                               (ਖਾਲੀ *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                                ne10);
                }
            }
        }
    #ਨਹੀਂਤਾਂ
        ਅਚਲ int64_t bs = ggml_blck_ਮਾਪ(vec_dot_type);

        ਲਈ (int64_t i13 = 0; i13 < ne13; ++i13) {
            ਲਈ (int64_t i12 = 0; i12 < ne12; ++i12) {
                ਲਈ (int64_t i11 = 0; i11 < ne11; ++i11) {
                    int64_t ne10_ਖੰਡ_ਸ਼ੁਰੂ = (ith * ne10/bs) / nth;
                    int64_t ne10_ਖੰਡ_ਸਿਰਾ   = ((ith + 1) * ne10/bs) / nth;
                    ਅਚਲ ਖਾਲੀ * src1_ਪੰਕਤੀ  = (ਅਚਲ ਅੱਖਰ *) src1->ਅੰਕੜੇ + i13*nb13 + i12*nb12 + i11*nb11 + ne10_ਖੰਡ_ਸ਼ੁਰੂ*bs*nb10;
                    ਖਾਲੀ       * wdata_ਪੰਕਤੀ =                    wdata  + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_ਖੰਡ_ਸ਼ੁਰੂ*nbw0;

                    ਅਚਲ int64_t ne10_ਖੰਡ_ਮਾਪ = (ne10_ਖੰਡ_ਸਿਰਾ - ne10_ਖੰਡ_ਸ਼ੁਰੂ) * bs;

                    ਜੇ (src1->type == GGML_TYPE_F16) {
                        ggml_cpu_fp16_ਮੰਜ਼ਿਲ_fp32((ਅਚਲ ggml_fp16_t *) src1_ਪੰਕਤੀ, (ਦਸ਼ਮਲਵ *) wdata_ਪੰਕਤੀ, ne10_ਖੰਡ_ਮਾਪ);
                    } ਨਹੀਂਤਾਂ ਜੇ (src1->type == GGML_TYPE_BF16) {
                        ggml_cpu_bf16_ਮੰਜ਼ਿਲ_fp32((ਅਚਲ ggml_bf16_t *) src1_ਪੰਕਤੀ, (ਦਸ਼ਮਲਵ *) wdata_ਪੰਕਤੀ, ne10_ਖੰਡ_ਮਾਪ);
                    } ਨਹੀਂਤਾਂ {
                        from_float((ਅਚਲ ਦਸ਼ਮਲਵ *) src1_ਪੰਕਤੀ, wdata_ਪੰਕਤੀ, ne10_ਖੰਡ_ਮਾਪ);
                    }
                }
            }
        }
    #ਜੇ_ਅੰਤ
    }

    ਜੇ (ith == 0) {
        // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
        atomic_store_explicit(&params->threadpool->current_chunk, nth, memory_order_relaxed);
    }

    ggml_barrier(params->threadpool);

#ਜੇ GGML_USE_LLAMAFILE
    ਜੇ (!params->use_ref && src1->type != vec_dot_type) {
        ਅਚਲ ਖਾਲੀ* wdata = (src1->type == vec_dot_type) ? src1->ਅੰਕੜੇ : params->wdata;
        ਅਚਲ size_t ਪੰਕਤੀ_ਮਾਪ = ggml_ਪੰਕਤੀ_ਮਾਪ(vec_dot_type, ne10);

        ਲਈ (int64_t i13 = 0; i13 < ne13; i13++)
            ਲਈ (int64_t i12 = 0; i12 < ne12; i12++)
                ਜੇ (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_ਮਾਪ(src0->type),
                                     (ਅਚਲ ਅੱਖਰ *)src0->ਅੰਕੜੇ + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_ਮਾਪ(src0->type),
                                     (ਅਚਲ ਅੱਖਰ *)wdata + (i12*ne11 + i13*ne12*ne11)*ਪੰਕਤੀ_ਮਾਪ,
                                     ਪੰਕਤੀ_ਮਾਪ/ggml_type_ਮਾਪ(vec_dot_type),
                                     (ਅੱਖਰ *)dst->ਅੰਕੜੇ + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_ਮਾਪ(dst->type),
                                     src0->type,
                                     vec_dot_type,
                                     dst->type))
                    ਜਾਓ UseGgmlGemm2;
        ਵਾਪਸ;
    }
UseGgmlGemm2:;
#ਜੇ_ਅੰਤ

    // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
    ਅਚਲ int64_t nr0 = ne0;

    // This is the size of the rest of the dimensions of the result
    ਅਚਲ int64_t nr1 = ne1 * ne2 * ne3;

    // Now select a reasonable chunk size.
    ਪੂਰਨ chunk_ਮਾਪ = 16;

    // We need to step up the size if it's small
    ਜੇ (nr0 == 1 || nr1 == 1) {
        chunk_ਮਾਪ = 64;
    }

    // distribute the work across the inner or outer loop based on which one is larger
    // The number of chunks in the 0/1 dim.
    // CEIL(nr0/chunk_size)
    int64_t nchunk0 = (nr0 + chunk_ਮਾਪ - 1) / chunk_ਮਾਪ;
    int64_t nchunk1 = (nr1 + chunk_ਮਾਪ - 1) / chunk_ਮਾਪ;

    // If the chunking is poor for the number of threads on this setup, scrap the whole plan.  Re-chunk it by thread.
    //   Also, chunking by thread was measured to have perform better on NUMA systems.  See https://github.com/ggml-org/llama.cpp/pull/6915
    //   In theory, chunking should be just as useful on NUMA and non NUMA systems, but testing disagreed with that.
    ਜੇ (nchunk0 * nchunk1 < nth * 4 || ggml_is_numa()) {
        // distribute the thread work across the inner or outer loop based on which one is larger
        nchunk0 = nr0 > nr1 ? nth : 1; // parallelize by src0 rows
        nchunk1 = nr0 > nr1 ? 1 : nth; // parallelize by src1 rows
    }

    // The number of elements in each chunk
    ਅਚਲ int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    ਅਚਲ int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    // The first chunk comes from our thread_id, the rest will get auto-assigned.
    ਪੂਰਨ current_chunk = ith;

    ਜਦੋਂ (current_chunk < nchunk0 * nchunk1) {
        ਅਚਲ int64_t ith0 = current_chunk % nchunk0;
        ਅਚਲ int64_t ith1 = current_chunk / nchunk0;

        ਅਚਲ int64_t ir0_ਸ਼ੁਰੂ = dr0 * ith0;
        ਅਚਲ int64_t ir0_ਸਿਰਾ = MIN(ir0_ਸ਼ੁਰੂ + dr0, nr0);

        ਅਚਲ int64_t ir1_ਸ਼ੁਰੂ = dr1 * ith1;
        ਅਚਲ int64_t ir1_ਸਿਰਾ = MIN(ir1_ਸ਼ੁਰੂ + dr1, nr1);

        // dot kernels can handle 1 row and col at a time, but mmla kernels can process 2 rows and cols
        int64_t ਅੰਕ_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_vec_dot = vec_dot_ਅੰਕ_ਪੰਕਤੀਆਂ;

        // these checks are needed to avoid crossing dim1 boundaries
        // can be optimized, but the logic would become more complicated, so keeping it like this for simplicity
        ਜੇ ((nr0 % 2 != 0) || (ne11 % 2 != 0) || ((ir0_ਸਿਰਾ - ir0_ਸ਼ੁਰੂ) % 2 != 0) || ((ir1_ਸਿਰਾ - ir1_ਸ਼ੁਰੂ) % 2 != 0)) {
            ਅੰਕ_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_vec_dot = 1;
        }
        ggml_compute_forward_mul_mat_one_chunk(params, dst, src0->type, ਅੰਕ_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_vec_dot, ir0_ਸ਼ੁਰੂ, ir0_ਸਿਰਾ, ir1_ਸ਼ੁਰੂ, ir1_ਸਿਰਾ);

        ਜੇ (nth >= nchunk0 * nchunk1) {
            ਰੋਕੋ;
        }

        current_chunk = atomic_fetch_ਜੋੜੋ_explicit(&params->threadpool->current_chunk, 1, memory_order_relaxed);
    }
}

// ggml_compute_forward_mul_mat_id

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MMID_ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ(ਪੰਕਤੀ_id, i1) ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ[(ਪੰਕਤੀ_id)*ids->ne[0]*ids->ne[1] + (i1)]

ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping {
    int32_t i1;
    int32_t i2;
};

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_mul_mat_id_one_chunk(
    ਬਣਤਰ ggml_tensor * dst,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src1,
    ਅਚਲ ਬਣਤਰ ggml_tensor * ids,
    ਅਚਲ int64_t cur_a,
    ਅਚਲ int64_t ir0_ਸ਼ੁਰੂ,
    ਅਚਲ int64_t ir0_ਸਿਰਾ,
    ਅਚਲ int64_t ir1_ਸ਼ੁਰੂ,
    ਅਚਲ int64_t ir1_ਸਿਰਾ,
    ਅਚਲ ਅੱਖਰ * src0_cur,
    ਅਚਲ ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping * ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ,
    ਅਚਲ size_t ਪੰਕਤੀ_ਮਾਪ,
    ਅਚਲ bool src1_cont,
    ਅਚਲ ਖਾਲੀ * wdata) {

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

    ਅਚਲ ਗਿਣਤੀ ggml_type type = src0->type;

    ggml_vec_dot_t    ਅਚਲ vec_dot      = type_traits_cpu[type].vec_dot;
    ਗਿਣਤੀ ggml_type    ਅਚਲ vec_dot_type = type_traits_cpu[type].vec_dot_type;

    ਅਚਲ int64_t blck_0 = 16;
    ਅਚਲ int64_t blck_1 = 16;

    ਦਸ਼ਮਲਵ tmp[16];

    ਲਈ (int64_t iir1 = ir1_ਸ਼ੁਰੂ; iir1 < ir1_ਸਿਰਾ; iir1 += blck_1) {
        ਲਈ (int64_t iir0 = ir0_ਸ਼ੁਰੂ; iir0 < ir0_ਸਿਰਾ; iir0 += blck_0) {
            ਲਈ (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_ਸਿਰਾ; ++ir1) {
                ਅਚਲ int64_t _i12 = ir1; // logical row index for this expert

                ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping ਪੰਕਤੀ_mapping = MMID_ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ(cur_a, _i12);
                ਅਚਲ ਪੂਰਨ id       = ਪੰਕਤੀ_mapping.i1; // selected expert index

                ਅਚਲ int64_t  i11 = id % ne11;
                ਅਚਲ int64_t  i12 = ਪੰਕਤੀ_mapping.i2; // row index in src1

                ਅਚਲ int64_t  i1 = id;  // selected expert index
                ਅਚਲ int64_t  i2 = i12; //ਪੰਕਤੀ

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                ਅਚਲ ਅੱਖਰ * src1_ਖਾਨਾ = (ਅਚਲ ਅੱਖਰ *) wdata +
                    (src1_cont || src1->type != vec_dot_type
                    ? (i11      + i12*ne11)*ਪੰਕਤੀ_ਮਾਪ
                    : (i11*nb11 + i12*nb12));

                ਦਸ਼ਮਲਵ * dst_ਖਾਨਾ = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + (i1*nb1 + i2*nb2));

                ਲਈ (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_ਸਿਰਾ; ++ir0) {
                    vec_dot(ne00, &tmp[ir0 - iir0], 0, src0_cur + ir0*nb01, 0, src1_ਖਾਨਾ, 0, 1);
                }

                ਸਮ੍ਰਿਤੀ_ਨਕਲ(&dst_ਖਾਨਾ[iir0], tmp, (MIN(iir0 + blck_0, ir0_ਸਿਰਾ) - iir0)*ਆਕਾਰ(ਦਸ਼ਮਲਵ));
            }
        }
    }
}

ਸਥਿਰ ਖਾਲੀ * incr_ptr_aligned(ਖਾਲੀ ** p, size_t ਮਾਪ, size_t align) {

    ਖਾਲੀ * ptr = *p;
    ptr = (ਖਾਲੀ *) GGML_PAD((uintptr_t) ptr, align);
    *p = (ਖਾਲੀ *) ((ਅੱਖਰ *) ptr + ਮਾਪ);
    ਵਾਪਸ ptr;
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_mul_mat_id(
        ਅਚਲ ਬਣਤਰ ggml_compute_params * params,
              ਬਣਤਰ ggml_tensor * dst) {

    ਅਚਲ ਬਣਤਰ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ਬਣਤਰ ggml_tensor * src1 = dst->src[1];
    ਅਚਲ ਬਣਤਰ ggml_tensor * ids = dst->src[2];

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਗਿਣਤੀ ggml_type type = src0->type;

    ਅਚਲ bool src1_cont = ggml_is_contiguous(src1);

    ਗਿਣਤੀ ggml_type    ਅਚਲ vec_dot_type    = type_traits_cpu[type].vec_dot_type;
    ggml_from_float_t ਅਚਲ from_float      = type_traits_cpu[vec_dot_type].from_float;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_ਮਾਪ(type));
    GGML_ASSERT(nb10 == ggml_type_ਮਾਪ(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // row groups
    ਅਚਲ ਪੂਰਨ n_ids = ids->ne[0]; // n_expert_used
    ਅਚਲ ਪੂਰਨ n_as  = ne02;       // n_expert

    ਖਾਲੀ * wdata_cur = params->wdata;

    ਜੇ (src1->type != vec_dot_type) {
        incr_ptr_aligned(&wdata_cur, ggml_ਪੰਕਤੀ_ਮਾਪ(vec_dot_type, ggml_nelements(src1)), ਆਕਾਰ(int64_t));
    }

    int64_t * ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts = // [n_as]
        incr_ptr_aligned(&wdata_cur, n_as*ਆਕਾਰ(int64_t), ਆਕਾਰ(int64_t));

    ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping * ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ = // [n_as][ids->ne[0]*ids->ne[1]]
        incr_ptr_aligned(&wdata_cur, n_as*ids->ne[0]*ids->ne[1]*ਆਕਾਰ(ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping), ਆਕਾਰ(int64_t));

    ਅੱਖਰ (*atomic_current_chunk)[CACHE_ਸਤਰ_ਮਾਪ] = // [n_as]
        incr_ptr_aligned(&wdata_cur, CACHE_ਸਤਰ_ਮਾਪ * n_as, CACHE_ਸਤਰ_ਮਾਪ);

    // Tiled matmul (see tiled.h); per-thread work buffers, 0 bytes when disabled. The
    // reservation is unconditional, the per expert eligibility is decided at dispatch time
    ਅੱਖਰ * tiled_scratch = incr_ptr_aligned(&wdata_cur, ggml_tiled_wdata_ਮਾਪ(nth, dst), 64);

    GGML_ASSERT(params->wsize >= (size_t)((ਅੱਖਰ *) wdata_cur - (ਅੱਖਰ *) params->wdata));

    ਜੇ (src1->type != vec_dot_type) {
        ਅੱਖਰ * wdata = params->wdata;

        ਅਚਲ size_t nbw0 = ggml_type_ਮਾਪ(vec_dot_type);
        ਅਚਲ size_t nbw1 = ggml_ਪੰਕਤੀ_ਮਾਪ(vec_dot_type, ne10);
        ਅਚਲ size_t nbw2 = nbw1*ne11;
        ਅਚਲ size_t nbw3 = nbw2*ne12;

        ਪ੍ਰਤਿਜ੍ਞਾ(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

#ਜੇ 0
        ਲਈ (int64_t i13 = 0; i13 < ne13; ++i13) {
            ਲਈ (int64_t i12 = ith; i12 < ne12; i12 += nth) {
                ਲਈ (int64_t i11 = 0; i11 < ne11; ++i11) {
                    from_float((ਦਸ਼ਮਲਵ *)((ਅੱਖਰ *) src1->ਅੰਕੜੇ + i13*nb13 + i12*nb12 + i11*nb11),
                               (ਖਾਲੀ *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                               ne10);
                }
            }
        }
#ਨਹੀਂਤਾਂ
        ਲਈ (int64_t i13 = 0; i13 < ne13; ++i13) {
            ਲਈ (int64_t i12 = 0; i12 < ne12; ++i12) {
                ਲਈ (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = ggml_blck_ਮਾਪ(vec_dot_type);
                    int64_t ne10_ਖੰਡ_ਸ਼ੁਰੂ = (ith * ne10/bs) / nth;
                    int64_t ne10_ਖੰਡ_ਸਿਰਾ   = ((ith + 1) * ne10/bs) / nth;
                    from_float((ਦਸ਼ਮਲਵ *)((ਅੱਖਰ *) src1->ਅੰਕੜੇ + i13*nb13 + i12*nb12 + i11*nb11 + ne10_ਖੰਡ_ਸ਼ੁਰੂ*bs*nb10),
                               (ਖਾਲੀ *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_ਖੰਡ_ਸ਼ੁਰੂ*nbw0),
                               (ne10_ਖੰਡ_ਸਿਰਾ - ne10_ਖੰਡ_ਸ਼ੁਰੂ) * bs);
                }
            }
        }
#ਜੇ_ਅੰਤ
    }

    ਜੇ (ith == 0) {
        // initialize matrix_row_counts
        memset(ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts, 0, n_as*ਆਕਾਰ(int64_t));

        // group rows by src0 matrix
        ਲਈ (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            ਲਈ (ਪੂਰਨ id = 0; id < n_ids; ++id) {
                ਅਚਲ int32_t i02 = *(ਅਚਲ int32_t *) ((ਅਚਲ ਅੱਖਰ *) ids->ਅੰਕੜੇ + iid1*ids->nb[1] + id*ids->nb[0]);

                ਪ੍ਰਤਿਜ੍ਞਾ(i02 >= 0 && i02 < n_as);

                MMID_ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ(i02, ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts[i02]) = (ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping) {id, iid1};
                ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts[i02] += 1;
            }
        }
    }

    // reset current_chunk
    ਲਈ (ਪੂਰਨ cur_a = ith; cur_a < n_as; cur_a += nth) {
        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);
        *current_chunk_ctr = nth;
    }

    ggml_barrier(params->threadpool);

    ਲਈ (ਪੂਰਨ cur_a = 0; cur_a < n_as; ++cur_a) {
        ਅਚਲ int64_t cne1 = ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts[cur_a];

        ਜੇ (cne1 == 0) {
            ਜਾਰੀ;
        }

        // tiled takes over if profitable for this expert (see tiled.h)
        ਜੇ (ggml_compute_forward_mul_mat_id_tiled(params, dst, cur_a, cne1, (ਅਚਲ int32_t *) &MMID_ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ(cur_a, 0), tiled_scratch)) {
            ਜਾਰੀ;
        }

        ਅਚਲ ਅੱਖਰ * src0_cur = (ਅਚਲ ਅੱਖਰ *) src0->ਅੰਕੜੇ + cur_a * nb02;
        ਅਚਲ ਖਾਲੀ * wdata = (src1->type == vec_dot_type) ? src1->ਅੰਕੜੇ : params->wdata;
        ਅਚਲ size_t ਪੰਕਤੀ_ਮਾਪ = ggml_ਪੰਕਤੀ_ਮਾਪ(vec_dot_type, ne10);

        ਅਚਲ int64_t nr0 = ne01;
        ਅਚਲ int64_t nr1 = cne1;

        ਪੂਰਨ chunk_ਮਾਪ = 16;
        ਜੇ (nr0 == 1 || nr1 == 1) {
            chunk_ਮਾਪ = 64;
        }

        // disable for NUMA
        ਅਚਲ bool disable_chunking = ggml_is_numa();

        int64_t nchunk0 = (nr0 + chunk_ਮਾਪ - 1) / chunk_ਮਾਪ;
        int64_t nchunk1 = (nr1 + chunk_ਮਾਪ - 1) / chunk_ਮਾਪ;

        ਜੇ (nchunk0 * nchunk1 < nth * 4 || disable_chunking) {
            nchunk0 = nr0 > nr1 ? nth : 1;
            nchunk1 = nr0 > nr1 ? 1 : nth;
        }

        ਅਚਲ int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        ਅਚਲ int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

        ਪੂਰਨ current_chunk = ith;

        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);

        ਜਦੋਂ (current_chunk < nchunk0 * nchunk1) {
            ਅਚਲ int64_t ith0 = current_chunk % nchunk0;
            ਅਚਲ int64_t ith1 = current_chunk / nchunk0;

            ਅਚਲ int64_t ir0_ਸ਼ੁਰੂ = dr0 * ith0;
            ਅਚਲ int64_t ir0_ਸਿਰਾ = MIN(ir0_ਸ਼ੁਰੂ + dr0, nr0);

            ਅਚਲ int64_t ir1_ਸ਼ੁਰੂ = dr1 * ith1;
            ਅਚਲ int64_t ir1_ਸਿਰਾ = MIN(ir1_ਸ਼ੁਰੂ + dr1, nr1);

            ggml_compute_forward_mul_mat_id_one_chunk(
                dst, src0, src1, ids, cur_a,
                ir0_ਸ਼ੁਰੂ, ir0_ਸਿਰਾ, ir1_ਸ਼ੁਰੂ, ir1_ਸਿਰਾ,
                src0_cur, ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ, ਪੰਕਤੀ_ਮਾਪ, src1_cont, wdata
            );

            ਜੇ (nth >= nchunk0 * nchunk1) {
                ਰੋਕੋ;
            }

            current_chunk = atomic_fetch_ਜੋੜੋ_explicit(current_chunk_ctr, 1, memory_order_relaxed);
        }
    }
}

/////////////////////////////////

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward(ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * tensor) {
    GGML_ASSERT(params);

    ਜੇ (tensor->op == GGML_OP_NONE || ggml_is_ਖਾਲੀ_ਥਾਵਾਂ(tensor)) {
        ਵਾਪਸ;
    }

    // extra_buffer op?
    ਜੇ (ggml_cpu_extra_compute_forward(params, tensor)) {
        ਵਾਪਸ;
    }

    ਚੋਣ (tensor->op) {
        ਹਾਲਤ GGML_OP_DUP:
            {
                ggml_compute_forward_dup(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਜੋੜੋ:
            {
                ggml_compute_forward_ਜੋੜੋ(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਜੋੜੋ_ID:
            {
                ggml_compute_forward_ਜੋੜੋ_id(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ADD1:
            {
                ggml_compute_forward_add1(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ACC:
            {
                ggml_compute_forward_acc(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SUB:
            {
                ggml_compute_forward_sub(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_MUL:
            {
                ggml_compute_forward_mul(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_DIV:
            {
                ggml_compute_forward_div(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SQR:
            {
                ggml_compute_forward_sqr(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SQRT:
            {
                ggml_compute_forward_sqrt(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_LOG:
            {
                ggml_compute_forward_log(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SIN:
            {
                ggml_compute_forward_sin(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_COS:
            {
                ggml_compute_forward_cos(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਜੋੜ:
            {
                ggml_compute_forward_ਜੋੜ(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਜੋੜ_ਪੰਕਤੀਆਂ:
            {
                ggml_compute_forward_ਜੋੜ_ਪੰਕਤੀਆਂ(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CUMSUM:
            {
                ggml_compute_forward_cumsum(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਔਸਤ:
            {
                ggml_compute_forward_ਔਸਤ(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ARGMAX:
            {
                ggml_compute_forward_argmax(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਗਿਣਤੀ_EQUAL:
            {
                ggml_compute_forward_ਗਿਣਤੀ_equal(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_REPEAT:
            {
                ggml_compute_forward_repeat(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_REPEAT_ਪਿੱਠ:
            {
                ggml_compute_forward_repeat_ਪਿੱਠ(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CONCAT:
            {
                ggml_compute_forward_concat(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SILU_ਪਿੱਠ:
            {
                ggml_compute_forward_silu_ਪਿੱਠ(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_NORM:
            {
                ggml_compute_forward_norm(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_RMS_NORM:
            {
                ggml_compute_forward_rms_norm(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_RMS_NORM_ਪਿੱਠ:
            {
                ggml_compute_forward_rms_norm_ਪਿੱਠ(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਸਮੂਹ_NORM:
            {
                ggml_compute_forward_ਸਮੂਹ_norm(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_L2_NORM:
            {
                ggml_compute_forward_l2_norm(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_MUL_MAT:
            {
                ggml_compute_forward_mul_mat(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_MUL_MAT_ID:
            {
                ggml_compute_forward_mul_mat_id(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਬਾਹਰਲਾ_PROD:
            {
                ggml_compute_forward_ਬਾਹਰਲਾ_prod(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SCALE:
            {
                ggml_compute_forward_scale(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਸੈੱਟ:
            {
                ggml_compute_forward_ਸੈੱਟ(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CPY:
            {
                ggml_compute_forward_cpy(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CONT:
            {
                ggml_compute_forward_cont(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_GET_ਪੰਕਤੀਆਂ:
            {
                ggml_compute_forward_get_ਪੰਕਤੀਆਂ(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_GET_ਪੰਕਤੀਆਂ_ਪਿੱਠ:
            {
                ggml_compute_forward_get_ਪੰਕਤੀਆਂ_ਪਿੱਠ(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਸੈੱਟ_ਪੰਕਤੀਆਂ:
            {
                ggml_compute_forward_ਸੈੱਟ_ਪੰਕਤੀਆਂ(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_DIAG:
            {
                ggml_compute_forward_diag(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_DIAG_ਨਕਾਬ_INF:
            {
                ggml_compute_forward_diag_ਨਕਾਬ_inf(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_DIAG_ਨਕਾਬ_ZERO:
            {
                ggml_compute_forward_diag_ਨਕਾਬ_zero(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SOFT_MAX:
            {
                ggml_compute_forward_soft_max(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SOFT_MAX_ਪਿੱਠ:
            {
                ggml_compute_forward_soft_max_ext_ਪਿੱਠ(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ROPE:
            {
                ggml_compute_forward_rope(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ROPE_ਪਿੱਠ:
            {
                ggml_compute_forward_rope_ਪਿੱਠ(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CLAMP:
            {
                ggml_compute_forward_clamp(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_compute_forward_conv_transpose_1d(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_IM2_ਖਾਨਾ:
            {
                ggml_compute_forward_im2col(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_IM2_ਖਾਨਾ_ਪਿੱਠ:
            {
                ggml_compute_forward_im2col_ਪਿੱਠ_f32(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_IM2_ਖਾਨਾ_3_D:
            {
                ggml_compute_forward_im2col_3d(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_COL2IM_1D:
            {
                ggml_compute_forward_col2im_1d(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CONV_2D:
            {
                ggml_compute_forward_conv_2d(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CONV_3D:
            {
                ggml_compute_forward_conv_3d(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CONV_2D_DW:
            {
                ggml_compute_forward_conv_2d_dw(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CONV_TRANSPOSE_2D:
            {
                ggml_compute_forward_conv_transpose_2d(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_POOL_1D:
            {
                ggml_compute_forward_pool_1d(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_POOL_2D:
            {
                ggml_compute_forward_pool_2d(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_POOL_2_D_ਪਿੱਠ:
            {
                ggml_compute_forward_pool_2d_ਪਿੱਠ(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_UPSCALE:
            {
                ggml_compute_forward_upscale(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_PAD:
            {
                ggml_compute_forward_pad(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_PAD_REFLECT_1D:
            {
                ggml_compute_forward_pad_reflect_1d(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ROLL:
            {
                ggml_compute_forward_roll(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ARANGE:
            {
                ggml_compute_forward_arange(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_TIMESTEP_EMBEDDING:
            {
                ggml_compute_forward_timestep_embedding(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ARGSORT:
            {
                ggml_compute_forward_argsort(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਸਿਖਰ_K:
            {
                ggml_compute_forward_ਸਿਖਰ_k(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_LEAKY_RELU:
            {
                ggml_compute_forward_leaky_relu(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_TRI:
            {
                ggml_compute_forward_tri(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_FILL:
            {
                ggml_compute_forward_fill(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_FLASH_ATTN_EXT:
            {
                ggml_compute_forward_flash_attn_ext(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_FLASH_ATTN_ਪਿੱਠ:
            {
                int32_t t = ggml_get_op_params_i32(tensor, 0);
                GGML_ASSERT(t == 0 || t == 1);
                bool masked = t != 0;
                ggml_compute_forward_flash_attn_ਪਿੱਠ(params, masked, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SSM_CONV:
            {
                ggml_compute_forward_ssm_conv(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SSM_ਜਾਂਚ:
            {
                ggml_compute_forward_ssm_ਜਾਂਚ(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_WIN_PART:
            {
                ggml_compute_forward_win_part(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_WIN_UNPART:
            {
                ggml_compute_forward_win_unpart(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_UNARY:
            {
                ggml_compute_forward_unary(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_GLU:
            {
                ggml_compute_forward_glu(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_GET_REL_POS:
            {
                ggml_compute_forward_get_rel_pos(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਜੋੜੋ_REL_POS:
            {
                ggml_compute_forward_ਜੋੜੋ_rel_pos(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_RWKV_WKV6:
            {
                ggml_compute_forward_rwkv_wkv6(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_GATED_ਰੇਖਿਕ_ATTN:
            {
                ggml_compute_forward_gla(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_RWKV_WKV7:
            {
                ggml_compute_forward_rwkv_wkv7(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SOLVE_TRI:
            {
                ggml_compute_forward_solve_tri(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_GATED_DELTA_NET:
            {
                ggml_compute_forward_gated_delta_net(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_LIGHTNING_INDEXER:
            {
                ggml_compute_forward_lightning_indexer(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_DSV4_HC_COMB:
            {
                ggml_compute_forward_dsv4_hc_comb(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_DSV4_HC_PRE:
            {
                ggml_compute_forward_dsv4_hc_pre(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_DSV4_HC_POST:
            {
                ggml_compute_forward_dsv4_hc_post(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਸਿਖਰ_K_ਨਕਾਬ:
            {
                ggml_compute_forward_ਸਿਖਰ_k_ਨਕਾਬ(params, tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_MAP_CUSTOM1:
            {
                ggml_compute_forward_map_custom1(params, tensor);
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_MAP_CUSTOM2:
            {
                ggml_compute_forward_map_custom2(params, tensor);
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_MAP_CUSTOM3:
            {
                ggml_compute_forward_map_custom3(params, tensor);
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CUSTOM:
            {
                ggml_compute_forward_custom(params, tensor);
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CROSS_ENTROPY_LOSS:
            {
                ggml_compute_forward_cross_entropy_loss(params, tensor);
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CROSS_ENTROPY_LOSS_ਪਿੱਠ:
            {
                ggml_compute_forward_cross_entropy_loss_ਪਿੱਠ(params, tensor);
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_OPT_ਕਦਮ_ADAMW:
            {
                ggml_compute_forward_opt_ਕਦਮ_adamw(params, tensor);
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_OPT_ਕਦਮ_SGD:
            {
                ggml_compute_forward_opt_ਕਦਮ_sgd(params, tensor);
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_NONE:
            {
                // nop
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_RESHAPE:
            {
                // nop
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਕ੍ਰਮ_ਬਦਲੋ:
            {
                // nop
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_VIEW:
            {
                // nop
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_TRANSPOSE:
            {
                // nop
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਗਿਣਤੀ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// Android's libc implementation "bionic" does not support setting affinity
#ਜੇ defined(__gnu_linux__)
ਸਥਿਰ ਖਾਲੀ ਸੈੱਟ_numa_thread_affinity(ਪੂਰਨ thread_n) {
    ਜੇ (!ggml_is_numa()) {
        ਵਾਪਸ;
    }

    ਪੂਰਨ ਗੰਢ_ਅੰਕ;
    ਪੂਰਨ rv;
    size_t setsize = CPU_ALLOC_ਮਾਪ(g_state.numa.ਕੁੱਲ_cpus);

    ਚੋਣ(g_state.numa.numa_strategy) {
        ਹਾਲਤ GGML_NUMA_STRATEGY_DISTRIBUTE:
            // run thread on node_num thread_n / (threads per node)
            ਗੰਢ_ਅੰਕ = thread_n % g_state.numa.n_nodes;
            ਰੋਕੋ;
        ਹਾਲਤ GGML_NUMA_STRATEGY_ISOLATE:
            // run thread on current_node
            ਗੰਢ_ਅੰਕ = g_state.numa.current_ਗੰਢ;
            ਰੋਕੋ;
        ਹਾਲਤ GGML_NUMA_STRATEGY_NUMACTL:
            // use the cpuset that numactl gave us
            rv = pthread_setaffinity_np(pthread_self(), setsize, &g_state.numa.cpuset);
            ਜੇ (rv) {
                fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n",strerror(rv));
            }
            ਵਾਪਸ;
        ਮੂਲ:
            ਵਾਪਸ;
    }

    ਬਣਤਰ ggml_numa_ਗੰਢ * ਗੰਢ = &g_state.numa.nodes[ਗੰਢ_ਅੰਕ];

    cpu_ਸੈੱਟ_t * cpus = CPU_ALLOC(g_state.numa.ਕੁੱਲ_cpus);
    CPU_ZERO_S(setsize, cpus);
    ਲਈ (size_t i = 0; i < ਗੰਢ->n_cpus; ++i) {
        CPU_ਸੈੱਟ_S(ਗੰਢ->cpus[i], setsize, cpus);
    }

    rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    ਜੇ (rv) {
            fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}

ਸਥਿਰ ਖਾਲੀ clear_numa_thread_affinity(ਖਾਲੀ) {
    ਜੇ (!ggml_is_numa()) {
        ਵਾਪਸ;
    }

    size_t setsize = CPU_ALLOC_ਮਾਪ(g_state.numa.ਕੁੱਲ_cpus);

    cpu_ਸੈੱਟ_t * cpus = CPU_ALLOC(g_state.numa.ਕੁੱਲ_cpus);
    CPU_ZERO_S(setsize, cpus);
    ਲਈ (ਬਿਨਾਨਿਸ਼ਾਨੀ i = 0; i < g_state.numa.ਕੁੱਲ_cpus; ++i) {
        CPU_ਸੈੱਟ_S(i, setsize, cpus);
    }

    ਪੂਰਨ rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    ਜੇ (rv) {
        fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}
#ਨਹੀਂਤਾਂ
// TODO: Windows etc.
// (the linux implementation may also work on BSD, someone should test)
ਸਥਿਰ ਖਾਲੀ ਸੈੱਟ_numa_thread_affinity(ਪੂਰਨ thread_n) { ਅਣਵਰਤਿਆ(thread_n);  }
ਸਥਿਰ ਖਾਲੀ clear_numa_thread_affinity(ਖਾਲੀ) {}
#ਜੇ_ਅੰਤ

ਸਥਿਰ ਪੂਰਨ ggml_get_n_tasks(ਬਣਤਰ ggml_tensor * ਗੰਢ, ਪੂਰਨ n_ਧਾਗੇ) {
    ਪੂਰਨ n_tasks = 0;

    ਜੇ (ggml_is_ਖਾਲੀ_ਥਾਵਾਂ(ਗੰਢ)) {
        // no need to multi-thread a no-op
        n_tasks = 1;
        ਵਾਪਸ n_tasks;
    }

    ਚੋਣ (ਗੰਢ->op) {
        ਹਾਲਤ GGML_OP_CPY:
        ਹਾਲਤ GGML_OP_DUP:
        ਹਾਲਤ GGML_OP_CONT:
        ਹਾਲਤ GGML_OP_ਜੋੜੋ:
        ਹਾਲਤ GGML_OP_ਜੋੜੋ_ID:
        ਹਾਲਤ GGML_OP_ADD1:
        ਹਾਲਤ GGML_OP_ACC:
        ਹਾਲਤ GGML_OP_CUMSUM:
        ਹਾਲਤ GGML_OP_TRI:
        ਹਾਲਤ GGML_OP_FILL:
            {
                n_tasks = n_ਧਾਗੇ;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SUB:
        ਹਾਲਤ GGML_OP_SQR:
        ਹਾਲਤ GGML_OP_SQRT:
        ਹਾਲਤ GGML_OP_LOG:
        ਹਾਲਤ GGML_OP_SIN:
        ਹਾਲਤ GGML_OP_COS:
        ਹਾਲਤ GGML_OP_ਜੋੜ:
        ਹਾਲਤ GGML_OP_ਜੋੜ_ਪੰਕਤੀਆਂ:
        ਹਾਲਤ GGML_OP_ਔਸਤ:
        ਹਾਲਤ GGML_OP_ARGMAX:
            {
                n_tasks = 1;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਗਿਣਤੀ_EQUAL:
        ਹਾਲਤ GGML_OP_SOLVE_TRI:
        ਹਾਲਤ GGML_OP_GATED_DELTA_NET:
        ਹਾਲਤ GGML_OP_DSV4_HC_COMB:
        ਹਾਲਤ GGML_OP_DSV4_HC_PRE:
        ਹਾਲਤ GGML_OP_DSV4_HC_POST:
        ਹਾਲਤ GGML_OP_ਸਿਖਰ_K_ਨਕਾਬ:
            {
                n_tasks = n_ਧਾਗੇ;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_REPEAT:
        ਹਾਲਤ GGML_OP_REPEAT_ਪਿੱਠ:
        ਹਾਲਤ GGML_OP_LEAKY_RELU:
            {
                n_tasks = 1;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_UNARY:
            ਚੋਣ (ggml_get_unary_op(ਗੰਢ)) {
                ਹਾਲਤ GGML_UNARY_OP_ABS:
                ਹਾਲਤ GGML_UNARY_OP_SGN:
                ਹਾਲਤ GGML_UNARY_OP_NEG:
                ਹਾਲਤ GGML_UNARY_OP_ਕਦਮ:
                ਹਾਲਤ GGML_UNARY_OP_TANH:
                ਹਾਲਤ GGML_UNARY_OP_ELU:
                ਹਾਲਤ GGML_UNARY_OP_RELU:
                ਹਾਲਤ GGML_UNARY_OP_SIGMOID:
                ਹਾਲਤ GGML_UNARY_OP_HARDSWISH:
                ਹਾਲਤ GGML_UNARY_OP_HARDSIGMOID:
                ਹਾਲਤ GGML_UNARY_OP_EXP:
                ਹਾਲਤ GGML_UNARY_OP_SOFTPLUS:
                ਹਾਲਤ GGML_UNARY_OP_EXPM1:
                ਹਾਲਤ GGML_UNARY_OP_FLOOR:
                ਹਾਲਤ GGML_UNARY_OP_CEIL:
                ਹਾਲਤ GGML_UNARY_OP_ਗੇੜਾ:
                ਹਾਲਤ GGML_UNARY_OP_TRUNC:
                    {
                        n_tasks = 1;
                    } ਰੋਕੋ;

                ਹਾਲਤ GGML_UNARY_OP_GELU:
                ਹਾਲਤ GGML_UNARY_OP_GELU_ERF:
                ਹਾਲਤ GGML_UNARY_OP_GELU_ਤੇਜ਼:
                ਹਾਲਤ GGML_UNARY_OP_SILU:
                ਹਾਲਤ GGML_UNARY_OP_XIELU:
                    {
                        n_tasks = n_ਧਾਗੇ;
                    } ਰੋਕੋ;
                ਮੂਲ:
                    GGML_ABORT("fatal error");
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_GLU:
            ਚੋਣ (ggml_get_glu_op(ਗੰਢ)) {
                ਹਾਲਤ GGML_GLU_OP_REGLU:
                ਹਾਲਤ GGML_GLU_OP_GEGLU:
                ਹਾਲਤ GGML_GLU_OP_SWIGLU:
                ਹਾਲਤ GGML_GLU_OP_SWIGLU_OAI:
                ਹਾਲਤ GGML_GLU_OP_GEGLU_ERF:
                ਹਾਲਤ GGML_GLU_OP_GEGLU_ਤੇਜ਼:
                ਹਾਲਤ GGML_GLU_OP_SWIGLU_CLAMP:
                    {
                        n_tasks = n_ਧਾਗੇ;
                    } ਰੋਕੋ;
                ਮੂਲ:
                    GGML_ABORT("fatal error");
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SILU_ਪਿੱਠ:
        ਹਾਲਤ GGML_OP_MUL:
        ਹਾਲਤ GGML_OP_DIV:
        ਹਾਲਤ GGML_OP_NORM:
        ਹਾਲਤ GGML_OP_RMS_NORM:
        ਹਾਲਤ GGML_OP_RMS_NORM_ਪਿੱਠ:
        ਹਾਲਤ GGML_OP_L2_NORM:
        ਹਾਲਤ GGML_OP_ਸਮੂਹ_NORM:
        ਹਾਲਤ GGML_OP_CONCAT:
        ਹਾਲਤ GGML_OP_MUL_MAT:
        ਹਾਲਤ GGML_OP_MUL_MAT_ID:
        ਹਾਲਤ GGML_OP_ਬਾਹਰਲਾ_PROD:
        ਹਾਲਤ GGML_OP_ਸੈੱਟ_ਪੰਕਤੀਆਂ:
            {
                n_tasks = n_ਧਾਗੇ;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_GET_ਪੰਕਤੀਆਂ:
            {
                // FIXME: get_rows can use additional threads, but the cost of launching additional threads
                // decreases performance with GPU offloading
                //n_tasks = n_threads;
                n_tasks = 1;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SCALE:
        ਹਾਲਤ GGML_OP_ਸੈੱਟ:
        ਹਾਲਤ GGML_OP_RESHAPE:
        ਹਾਲਤ GGML_OP_VIEW:
        ਹਾਲਤ GGML_OP_ਕ੍ਰਮ_ਬਦਲੋ:
        ਹਾਲਤ GGML_OP_TRANSPOSE:
        ਹਾਲਤ GGML_OP_GET_ਪੰਕਤੀਆਂ_ਪਿੱਠ:
        ਹਾਲਤ GGML_OP_DIAG:
            {
                n_tasks = 1;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_DIAG_ਨਕਾਬ_ZERO:
        ਹਾਲਤ GGML_OP_DIAG_ਨਕਾਬ_INF:
        ਹਾਲਤ GGML_OP_SOFT_MAX_ਪਿੱਠ:
        ਹਾਲਤ GGML_OP_ROPE:
        ਹਾਲਤ GGML_OP_ROPE_ਪਿੱਠ:
        ਹਾਲਤ GGML_OP_ਜੋੜੋ_REL_POS:
            {
                n_tasks = n_ਧਾਗੇ;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CLAMP:
            {
                n_tasks = 1; //TODO
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SOFT_MAX:
            {
                n_tasks = MIN(n_ਧਾਗੇ, ggml_nrows(ਗੰਢ->src[0]));
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_IM2_ਖਾਨਾ:
        ਹਾਲਤ GGML_OP_IM2_ਖਾਨਾ_ਪਿੱਠ:
        ਹਾਲਤ GGML_OP_IM2_ਖਾਨਾ_3_D:
        ਹਾਲਤ GGML_OP_CONV_2D:
        ਹਾਲਤ GGML_OP_CONV_3D:
        ਹਾਲਤ GGML_OP_CONV_2D_DW:
        ਹਾਲਤ GGML_OP_COL2IM_1D:
        ਹਾਲਤ GGML_OP_CONV_TRANSPOSE_1D:
        ਹਾਲਤ GGML_OP_CONV_TRANSPOSE_2D:
            {
                n_tasks = n_ਧਾਗੇ;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_POOL_1D:
        ਹਾਲਤ GGML_OP_POOL_2D:
        ਹਾਲਤ GGML_OP_POOL_2_D_ਪਿੱਠ:
            {
                n_tasks = 1;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_UPSCALE:
        ਹਾਲਤ GGML_OP_PAD:
        ਹਾਲਤ GGML_OP_PAD_REFLECT_1D:
        ਹਾਲਤ GGML_OP_ROLL:
        ਹਾਲਤ GGML_OP_ARANGE:
        ਹਾਲਤ GGML_OP_TIMESTEP_EMBEDDING:
        ਹਾਲਤ GGML_OP_ARGSORT:
        ਹਾਲਤ GGML_OP_ਸਿਖਰ_K:
        ਹਾਲਤ GGML_OP_FLASH_ATTN_EXT:
        ਹਾਲਤ GGML_OP_FLASH_ATTN_ਪਿੱਠ:
        ਹਾਲਤ GGML_OP_SSM_CONV:
        ਹਾਲਤ GGML_OP_SSM_ਜਾਂਚ:
        ਹਾਲਤ GGML_OP_LIGHTNING_INDEXER:
            {
                n_tasks = n_ਧਾਗੇ;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_RWKV_WKV6:
        ਹਾਲਤ GGML_OP_GATED_ਰੇਖਿਕ_ATTN:
        ਹਾਲਤ GGML_OP_RWKV_WKV7:
            {
                ਅਚਲ int64_t n_heads = ਗੰਢ->src[1]->ne[1];
                n_tasks = MIN(n_ਧਾਗੇ, n_heads);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_WIN_PART:
        ਹਾਲਤ GGML_OP_WIN_UNPART:
        ਹਾਲਤ GGML_OP_GET_REL_POS:
            {
                n_tasks = 1;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_MAP_CUSTOM1:
            {
                ਬਣਤਰ ggml_map_custom1_op_params p;
                ਸਮ੍ਰਿਤੀ_ਨਕਲ(&p, ਗੰਢ->op_params, ਆਕਾਰ(p));
                ਜੇ (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_ਧਾਗੇ;
                } ਨਹੀਂਤਾਂ {
                    n_tasks = MIN(p.n_tasks, n_ਧਾਗੇ);
                }
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_MAP_CUSTOM2:
            {
                ਬਣਤਰ ggml_map_custom2_op_params p;
                ਸਮ੍ਰਿਤੀ_ਨਕਲ(&p, ਗੰਢ->op_params, ਆਕਾਰ(p));
                ਜੇ (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_ਧਾਗੇ;
                } ਨਹੀਂਤਾਂ {
                    n_tasks = MIN(p.n_tasks, n_ਧਾਗੇ);
                }
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_MAP_CUSTOM3:
            {
                ਬਣਤਰ ggml_map_custom3_op_params p;
                ਸਮ੍ਰਿਤੀ_ਨਕਲ(&p, ਗੰਢ->op_params, ਆਕਾਰ(p));
                ਜੇ (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_ਧਾਗੇ;
                } ਨਹੀਂਤਾਂ {
                    n_tasks = MIN(p.n_tasks, n_ਧਾਗੇ);
                }
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CUSTOM:
            {
                ਬਣਤਰ ggml_custom_op_params p;
                ਸਮ੍ਰਿਤੀ_ਨਕਲ(&p, ਗੰਢ->op_params, ਆਕਾਰ(p));
                ਜੇ (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_ਧਾਗੇ;
                } ਨਹੀਂਤਾਂ {
                    n_tasks = MIN(p.n_tasks, n_ਧਾਗੇ);
                }
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CROSS_ENTROPY_LOSS:
        ਹਾਲਤ GGML_OP_CROSS_ENTROPY_LOSS_ਪਿੱਠ:
        ਹਾਲਤ GGML_OP_OPT_ਕਦਮ_ADAMW:
        ਹਾਲਤ GGML_OP_OPT_ਕਦਮ_SGD:
            {
                n_tasks = n_ਧਾਗੇ;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_NONE:
            {
                n_tasks = 1;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਗਿਣਤੀ:
            {
                GGML_ABORT("fatal error");
            }
        ਮੂਲ:
            {
                fprintf(stderr, "%s: op not implemented: ", __func__);
                ਜੇ (ਗੰਢ->op < GGML_OP_ਗਿਣਤੀ) {
                    fprintf(stderr, "%s\n", ggml_op_name(ਗੰਢ->op));
                } ਨਹੀਂਤਾਂ {
                    fprintf(stderr, "%d\n", ਗੰਢ->op);
                }
                GGML_ABORT("fatal error");
            }
    }

    ਪ੍ਰਤਿਜ੍ਞਾ(n_tasks > 0);

    ਵਾਪਸ n_tasks;
}

ਸਥਿਰ thread_ret_t ggml_ਗ੍ਰਾਫ_compute_secondary_thread(ਖਾਲੀ* ਅੰਕੜੇ);

#ਜੇ defined(_WIN32)
#ਅਨ੍ਤਰ੍ਭਾਵ "windows.h"

// TODO: support > 64 CPUs
ਸਥਿਰ bool ggml_thread_apply_affinity(bool * ਨਕਾਬ) {
    HANDLE    h = GetCurrentThread();
    uint64_t  bitmask = 0ULL;

    ਪ੍ਰਤਿਜ੍ਞਾ(GGML_MAX_N_ਧਾਗੇ >= 64);

    ਲਈ (int32_t i = 0; i < 8; i++) {
        int32_t idx = i * 8;
        uint8_t val = 0;
        val |= ਨਕਾਬ[idx + 0] << 0;
        val |= ਨਕਾਬ[idx + 1] << 1;
        val |= ਨਕਾਬ[idx + 2] << 2;
        val |= ਨਕਾਬ[idx + 3] << 3;
        val |= ਨਕਾਬ[idx + 4] << 4;
        val |= ਨਕਾਬ[idx + 5] << 5;
        val |= ਨਕਾਬ[idx + 6] << 6;
        val |= ਨਕਾਬ[idx + 7] << 7;
        bitmask |= (uint64_t)val << idx;
    }

    ਲਈ (int32_t i = 64; i < GGML_MAX_N_ਧਾਗੇ; i++) {
        ਜੇ (ਨਕਾਬ[i]) {
            fprintf(stderr, "warn: setting thread-affinity for > 64 CPUs isn't supported on windows!\n");
            ਰੋਕੋ;
        }
    }

    DWORD_PTR m = (DWORD_PTR)bitmask;

    m = ਸੈੱਟ_Thread_Affinity_ਨਕਾਬ(h, m);

    ਵਾਪਸ m != 0;
}

ਸਥਿਰ bool ggml_thread_apply_ਤਰਜੀਹ(int32_t prio) {
    // Note that on Windows the Process Priority Class must be updated in order to set Thread priority.
    // This is up to the applications.
    DWORD p = THREAD_ਤਰਜੀਹ_ਸਾਧਾਰਨ;
    ਚੋਣ (prio) {
        ਹਾਲਤ GGML_SCHED_PRIO_ਨੀਵਾਂ:      p = THREAD_ਤਰਜੀਹ_ਹੇਠਾਂ_ਸਾਧਾਰਨ;  ਰੋਕੋ;
        ਹਾਲਤ GGML_SCHED_PRIO_ਸਾਧਾਰਨ:   p = THREAD_ਤਰਜੀਹ_ਸਾਧਾਰਨ;        ਰੋਕੋ;
        ਹਾਲਤ GGML_SCHED_PRIO_MEDIUM:   p = THREAD_ਤਰਜੀਹ_ABOVE_ਸਾਧਾਰਨ;  ਰੋਕੋ;
        ਹਾਲਤ GGML_SCHED_PRIO_ਉੱਚਾ:     p = THREAD_ਤਰਜੀਹ_HIGHEST;       ਰੋਕੋ;
        ਹਾਲਤ GGML_SCHED_PRIO_REALTIME: p = THREAD_ਤਰਜੀਹ_ਸਮਾਂ_CRITICAL; ਰੋਕੋ;
    }

    ਜੇ (prio != GGML_SCHED_PRIO_ਨੀਵਾਂ) {
        // Tell Windows that this thread should not be throttled (needs its own CPU core).
        // Newer Windows 11 versions aggressively park (offline) CPU cores and often place
        // all our threads onto the first 4 cores which results in terrible performance with
        // n_threads > 4
        #ਜੇ _WIN32_WINNT >= 0x0602
        THREAD_POWER_THROTTLING_STATE t;
        Zero_ਸਮ੍ਰਿਤੀ(&t, ਆਕਾਰ(t));
        t.ਸੰਸਕਰਣ     = THREAD_POWER_THROTTLING_CURRENT_ਸੰਸਕਰਣ;
        t.Control_ਨਕਾਬ = THREAD_POWER_THROTTLING_EXECUTION_ਰਫਤਾਰ;
        t.State_ਨਕਾਬ   = 0;

        ਜੇ (!ਸੈੱਟ_Thread_Information(GetCurrentThread(), ThreadPowerThrottling, &t, ਆਕਾਰ(t))) {
            GGML_LOG_DEBUG("failed to disable thread power throttling %d : (%d)\n", prio, (ਪੂਰਨ) Get_ਆਖਰੀ_Error());
            ਵਾਪਸ false;
        }
        #ਜੇ_ਅੰਤ
    }

    ਜੇ (prio == GGML_SCHED_PRIO_ਸਾਧਾਰਨ) {
        // Keep inherited policy/priority
        ਵਾਪਸ true;
    }

    ਜੇ (!ਸੈੱਟ_Thread_ਤਰਜੀਹ(GetCurrentThread(), p)) {
        fprintf(stderr, "warn: failed to set thread priority %d : (%d)\n", prio, (ਪੂਰਨ) Get_ਆਖਰੀ_Error());
        ਵਾਪਸ false;
    }

    ਵਾਪਸ true;
}

#ਜਾਂਜੇ defined(__APPLE__)
#ਅਨ੍ਤਰ੍ਭਾਵ <sys/types.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <sys/resource.h>

ਸਥਿਰ bool ggml_thread_apply_affinity(ਅਚਲ bool * ਨਕਾਬ) {
    // Not supported on Apple platforms
    ਅਣਵਰਤਿਆ(ਨਕਾਬ);
    ਵਾਪਸ true;
}

ਸਥਿਰ bool ggml_thread_apply_ਤਰਜੀਹ(int32_t prio) {
    ਬਣਤਰ sched_param p;
    int32_t policy = SCHED_OTHER;
    ਚੋਣ (prio) {
        // TODO: there seems to be no way to set lower prio on Apple platforms
        ਹਾਲਤ GGML_SCHED_PRIO_ਨੀਵਾਂ:      policy = SCHED_OTHER; p.sched_ਤਰਜੀਹ = 0;  ਰੋਕੋ;
        ਹਾਲਤ GGML_SCHED_PRIO_ਸਾਧਾਰਨ:   policy = SCHED_OTHER; p.sched_ਤਰਜੀਹ = 0;  ਰੋਕੋ;
        ਹਾਲਤ GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_ਤਰਜੀਹ = 40; ਰੋਕੋ;
        ਹਾਲਤ GGML_SCHED_PRIO_ਉੱਚਾ:     policy = SCHED_FIFO;  p.sched_ਤਰਜੀਹ = 80; ਰੋਕੋ;
        ਹਾਲਤ GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_ਤਰਜੀਹ = 90; ਰੋਕੋ;
    }

    ਜੇ (prio == GGML_SCHED_PRIO_ਸਾਧਾਰਨ) {
        // Keep inherited policy/priority
        ਵਾਪਸ true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    ਜੇ (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        ਵਾਪਸ false;
    }

    ਵਾਪਸ true;
}

#ਜਾਂਜੇ defined(__linux__)
// TODO: this may not work on BSD, to be verified

ਸਥਿਰ bool ggml_thread_apply_affinity(ਅਚਲ bool * ਨਕਾਬ) {
    cpu_ਸੈੱਟ_t cpuset;
    ਪੂਰਨ err;

    CPU_ZERO(&cpuset);

    ਲਈ (uint32_t i = 0; i < GGML_MAX_N_ਧਾਗੇ; i++) {
        ਜੇ (ਨਕਾਬ[i]) {
            GGML_PRINT_DEBUG("Thread %lx: adding %d to cpuset\n", pthread_self(), i);
            CPU_ਸੈੱਟ(i, &cpuset);
        }
    }

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __ANDROID__
    err = sched_setaffinity(0, ਆਕਾਰ(cpuset), &cpuset);
    ਜੇ (err < 0) {
        err = errno;
    }
#ਨਹੀਂਤਾਂ
    err = pthread_setaffinity_np(pthread_self(), ਆਕਾਰ(cpuset), &cpuset);
#ਜੇ_ਅੰਤ
    ਜੇ (err != 0) {
        fprintf(stderr, "warn: failed to set affinity mask 0x%llx : %s (%d)\n", (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ)ਨਕਾਬ, strerror(err), err);
        ਵਾਪਸ false;
    }

    ਵਾਪਸ true;
}

ਸਥਿਰ bool ggml_thread_apply_ਤਰਜੀਹ(int32_t prio) {
    ਬਣਤਰ sched_param p;
    int32_t policy = SCHED_OTHER;
    ਚੋਣ (prio) {
        ਹਾਲਤ GGML_SCHED_PRIO_ਨੀਵਾਂ:      policy = SCHED_BATCH; p.sched_ਤਰਜੀਹ = 0;  ਰੋਕੋ;
        ਹਾਲਤ GGML_SCHED_PRIO_ਸਾਧਾਰਨ:   policy = SCHED_OTHER; p.sched_ਤਰਜੀਹ = 0;  ਰੋਕੋ;
        ਹਾਲਤ GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_ਤਰਜੀਹ = 40; ਰੋਕੋ;
        ਹਾਲਤ GGML_SCHED_PRIO_ਉੱਚਾ:     policy = SCHED_FIFO;  p.sched_ਤਰਜੀਹ = 80; ਰੋਕੋ;
        ਹਾਲਤ GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_ਤਰਜੀਹ = 90; ਰੋਕੋ;
    }

    ਜੇ (prio == GGML_SCHED_PRIO_ਸਾਧਾਰਨ) {
        // Keep inherited policy/priority
        ਵਾਪਸ true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    ਜੇ (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        ਵਾਪਸ false;
    }

    ਵਾਪਸ true;
}

#ਨਹੀਂਤਾਂ // unsupported platforms

ਸਥਿਰ bool ggml_thread_apply_affinity(ਅਚਲ bool * ਨਕਾਬ) {
    ਅਣਵਰਤਿਆ(ਨਕਾਬ);
    ਵਾਪਸ true;
}

ਸਥਿਰ bool ggml_thread_apply_ਤਰਜੀਹ(int32_t prio) {
    ਅਣਵਰਤਿਆ(prio);
    ਵਾਪਸ true;
}

#ਜੇ_ਅੰਤ

ਸਥਿਰ bool ggml_thread_cpumask_is_valid(ਅਚਲ bool * ਨਕਾਬ) {
    ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_N_ਧਾਗੇ; i++) {
        ਜੇ (ਨਕਾਬ[i]) { ਵਾਪਸ true; }
    }
    ਵਾਪਸ false;
}

ਸਥਿਰ ਖਾਲੀ ggml_thread_cpumask_ਅਗਲਾ(ਅਚਲ bool * global_ਨਕਾਬ, bool * local_ਨਕਾਬ, bool strict, int32_t* iter) {
    ਜੇ (!strict) {
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(local_ਨਕਾਬ, global_ਨਕਾਬ, GGML_MAX_N_ਧਾਗੇ);
        ਵਾਪਸ;
    } ਨਹੀਂਤਾਂ {
        memset(local_ਨਕਾਬ, 0, GGML_MAX_N_ਧਾਗੇ);
        int32_t ਅਧਾਰ_idx = *iter;
        ਲਈ (int32_t i = 0; i < GGML_MAX_N_ਧਾਗੇ; i++) {
            int32_t idx = ਅਧਾਰ_idx + i;
            ਜੇ (idx >= GGML_MAX_N_ਧਾਗੇ) {
                // Just a cheaper modulo
                idx -= GGML_MAX_N_ਧਾਗੇ;
            }
            ਜੇ (global_ਨਕਾਬ[idx]) {
                local_ਨਕਾਬ[idx] = 1;
                *iter = idx + 1;
                ਵਾਪਸ;
            }
        }
    }
}

ਖਾਲੀ ggml_threadpool_free(ਬਣਤਰ ggml_threadpool* threadpool) {
    ਜੇ (!threadpool) ਵਾਪਸ;

    ਅਚਲ ਪੂਰਨ n_ਧਾਗੇ = threadpool->n_ਧਾਗੇ;

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_USE_OPENMP
    ਬਣਤਰ ggml_compute_state* workers = threadpool->workers;

    ggml_mutex_ਤਾਲਾ(&threadpool->mutex);

    threadpool->stop = true;
    threadpool->pause = false;

    ggml_cond_ਪ੍ਰਸਾਰਣ(&threadpool->cond);
    ggml_mutex_unlock(&threadpool->mutex);

    ਲਈ (ਪੂਰਨ j = 1; j < n_ਧਾਗੇ; j++) {
        int32_t rc = ggml_thread_join(workers[j].thrd, ਨਿਰਾ);
        GGML_ASSERT(rc == GGML_EXIT_SUCCESS || rc == GGML_EXIT_ABORTED);
        ਅਣਵਰਤਿਆ(rc);
    }

    ggml_mutex_destroy(&threadpool->mutex);
    ggml_cond_destroy(&threadpool->cond);
#ਜੇ_ਅੰਤ // GGML_USE_OPENMP

    ਅਚਲ size_t workers_ਮਾਪ = ਆਕਾਰ(ਬਣਤਰ ggml_compute_state) * n_ਧਾਗੇ;
    ggml_aligned_free(threadpool->workers, workers_ਮਾਪ);
    ggml_aligned_free(threadpool, ਆਕਾਰ(ਬਣਤਰ ggml_threadpool));
}

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_USE_OPENMP
// pause/resume must be called under mutex
ਸਥਿਰ ਖਾਲੀ ggml_threadpool_pause_locked(ਬਣਤਰ ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Pausing threadpool\n");
    threadpool->pause = true;
    ggml_cond_ਪ੍ਰਸਾਰਣ(&threadpool->cond);
}

ਸਥਿਰ ਖਾਲੀ ggml_threadpool_resume_locked(ਬਣਤਰ ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Resuming threadpool\n");
    threadpool->pause = false;
    ggml_cond_ਪ੍ਰਸਾਰਣ(&threadpool->cond);
}
#ਜੇ_ਅੰਤ

ਖਾਲੀ ggml_threadpool_pause(ਬਣਤਰ ggml_threadpool * threadpool) {
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_USE_OPENMP
    ggml_mutex_ਤਾਲਾ(&threadpool->mutex);
    ਜੇ (!threadpool->pause) {
       ggml_threadpool_pause_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#ਨਹੀਂਤਾਂ
    ਅਣਵਰਤਿਆ(threadpool);
#ਜੇ_ਅੰਤ
}

ਖਾਲੀ ggml_threadpool_resume(ਬਣਤਰ ggml_threadpool * threadpool) {
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_USE_OPENMP
    ggml_mutex_ਤਾਲਾ(&threadpool->mutex);
    ਜੇ (threadpool->pause) {
       ggml_threadpool_resume_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#ਨਹੀਂਤਾਂ
    ਅਣਵਰਤਿਆ(threadpool);
#ਜੇ_ਅੰਤ
}

ਬਣਤਰ ggml_cplan ggml_ਗ੍ਰਾਫ_plan(
          ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph,
                               ਪੂਰਨ   n_ਧਾਗੇ,
            ਬਣਤਰ ggml_threadpool * threadpool) {

    ਜੇ (threadpool == ਨਿਰਾ) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
    }
    ਜੇ (n_ਧਾਗੇ <= 0) {
        n_ਧਾਗੇ = threadpool ? threadpool->n_ਧਾਗੇ : GGML_DEFAULT_N_ਧਾਗੇ;
    }

#ਜੇ defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
    // Emscripten without pthreads support can only use a single thread
    n_ਧਾਗੇ = 1;
#ਜੇ_ਅੰਤ

#ਜੇ defined(__wasi__)
    // WASI doesn't support parallelism yet
    n_ਧਾਗੇ = 1;
#ਜੇ_ਅੰਤ

    size_t work_ਮਾਪ = 0;

    ਬਣਤਰ ggml_cplan cplan;
    memset(&cplan, 0, ਆਕਾਰ(ਬਣਤਰ ggml_cplan));

    ਪੂਰਨ max_tasks = 1;

    // thread scheduling for the different operations + work buffer size estimation
    ਲਈ (ਪੂਰਨ i = 0; i < cgraph->n_nodes; i++) {
        ਬਣਤਰ ggml_tensor * ਗੰਢ = cgraph->nodes[i];

        ਅਚਲ ਪੂਰਨ n_tasks = ggml_get_n_tasks(ਗੰਢ, n_ਧਾਗੇ);

        max_tasks = MAX(max_tasks, n_tasks);

        size_t cur = 0;

        ਜੇ (!ggml_cpu_extra_work_ਮਾਪ(n_ਧਾਗੇ, ਗੰਢ, &cur)) {
            ਚੋਣ (ਗੰਢ->op) {
                ਹਾਲਤ GGML_OP_CPY:
                ਹਾਲਤ GGML_OP_DUP:
                    {
                        ਜੇ (ggml_is_quantized(ਗੰਢ->type) ||
                            // F16 -> BF16 and BF16 -> F16 copies go through intermediate F32
                            (ਗੰਢ->src[0]->type == GGML_TYPE_F16  && ਗੰਢ->src[1] && ਗੰਢ->src[1]->type == GGML_TYPE_BF16) ||
                            (ਗੰਢ->src[0]->type == GGML_TYPE_BF16 && ਗੰਢ->src[1] && ਗੰਢ->src[1]->type == GGML_TYPE_F16) ||
                            // conversion between F32 and I32
                            (ਗੰਢ->src[0]->type == GGML_TYPE_F32 && ਗੰਢ->src[1] && ਗੰਢ->src[1]->type == GGML_TYPE_I32) ||
                            (ਗੰਢ->src[0]->type == GGML_TYPE_I32 && ਗੰਢ->src[1] && ਗੰਢ->src[1]->type == GGML_TYPE_F32)) {
                            cur = ggml_type_ਮਾਪ(GGML_TYPE_F32) * ਗੰਢ->ne[0] * n_tasks;
                        }
                    } ਰੋਕੋ;
                ਹਾਲਤ GGML_OP_ਜੋੜੋ:
                ਹਾਲਤ GGML_OP_ਜੋੜੋ_ID:
                ਹਾਲਤ GGML_OP_ADD1:
                    {
                        ਜੇ (ggml_is_quantized(ਗੰਢ->src[0]->type)) {
                            cur = ggml_type_ਮਾਪ(GGML_TYPE_F32) * ਗੰਢ->src[0]->ne[0] * n_tasks;
                        }
                    } ਰੋਕੋ;
                ਹਾਲਤ GGML_OP_ACC:
                    {
                        ਜੇ (ggml_is_quantized(ਗੰਢ->src[0]->type)) {
                            cur = ggml_type_ਮਾਪ(GGML_TYPE_F32) * ਗੰਢ->src[1]->ne[0] * n_tasks;
                        }
                    } ਰੋਕੋ;
                ਹਾਲਤ GGML_OP_ਗਿਣਤੀ_EQUAL:
                    {
                        cur = ggml_type_ਮਾਪ(ਗੰਢ->type)*n_tasks;
                    } ਰੋਕੋ;
                ਹਾਲਤ GGML_OP_MUL_MAT:
                    {
                        ਅਚਲ ਗਿਣਤੀ ggml_type vec_dot_type = type_traits_cpu[ਗੰਢ->src[0]->type].vec_dot_type;
                        ਜੇ (ਗੰਢ->src[1]->type != vec_dot_type) {
                            cur = ggml_ਪੰਕਤੀ_ਮਾਪ(vec_dot_type, ggml_nelements(ਗੰਢ->src[1]));
                        }
                        // Workspace for tiled (see tiled.h)
                        cur = GGML_PAD(cur, 64);
                        cur += ggml_tiled_wdata_ਮਾਪ(n_tasks, ਗੰਢ);
                    } ਰੋਕੋ;
                ਹਾਲਤ GGML_OP_MUL_MAT_ID:
                    {
                        cur = 0;
                        ਅਚਲ ਬਣਤਰ ggml_tensor * src0 = ਗੰਢ->src[0];
                        ਅਚਲ ਬਣਤਰ ggml_tensor * src1 = ਗੰਢ->src[1];
                        ਅਚਲ ਬਣਤਰ ggml_tensor * ids = ਗੰਢ->src[2];
                        ਅਚਲ ਗਿਣਤੀ ggml_type vec_dot_type = type_traits_cpu[src0->type].vec_dot_type;
                        ਅਚਲ ਪੂਰਨ n_as = src0->ne[2];
                        // src1
                        ਜੇ (src1->type != vec_dot_type) {
                            cur += ggml_ਪੰਕਤੀ_ਮਾਪ(vec_dot_type, ggml_nelements(src1)) + ਆਕਾਰ(int64_t);
                        }
                        // matrix_row_counts
                        cur += n_as * ਆਕਾਰ(int64_t) + ਆਕਾਰ(int64_t);
                        // matrix_rows
                        cur += n_as*ids->ne[0]*ids->ne[1]*ਆਕਾਰ(ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping) + ਆਕਾਰ(int64_t);
                        // atomic_current_chunk
                        cur += CACHE_ਸਤਰ_ਮਾਪ*n_as + CACHE_ਸਤਰ_ਮਾਪ;
                        // Workspace for tiled (see tiled.h)
                        cur = GGML_PAD(cur, 64);
                        cur += ggml_tiled_wdata_ਮਾਪ(n_tasks, ਗੰਢ);
                    } ਰੋਕੋ;
                ਹਾਲਤ GGML_OP_ਬਾਹਰਲਾ_PROD:
                    {
                        ਜੇ (ggml_is_quantized(ਗੰਢ->src[0]->type) ||
                            ਗੰਢ->src[0]->type == GGML_TYPE_F16) {
                            cur = ggml_type_ਮਾਪ(GGML_TYPE_F32) * ਗੰਢ->src[0]->ne[0] * n_tasks;
                        }
                    } ਰੋਕੋ;
                ਹਾਲਤ GGML_OP_ਸੈੱਟ_ਪੰਕਤੀਆਂ:
                    {
                        ਜੇ (ਗੰਢ->src[0]->type == GGML_TYPE_F16 && ਗੰਢ->type != GGML_TYPE_F16) {
                            cur = ggml_type_ਮਾਪ(GGML_TYPE_F32) * ਗੰਢ->src[0]->ne[0] * n_tasks;
                        }
                    } ਰੋਕੋ;
                ਹਾਲਤ GGML_OP_SOFT_MAX:
                ਹਾਲਤ GGML_OP_ROPE:
                ਹਾਲਤ GGML_OP_ROPE_ਪਿੱਠ:
                    {
                        cur = ggml_type_ਮਾਪ(GGML_TYPE_F32) * ਗੰਢ->ne[0] * n_tasks;
                    } ਰੋਕੋ;
                ਹਾਲਤ GGML_OP_CONV_TRANSPOSE_1D:
                    {
                        GGML_ASSERT(ਗੰਢ->src[0]->ne[3] == 1);
                        GGML_ASSERT(ਗੰਢ->src[1]->ne[2] == 1);
                        GGML_ASSERT(ਗੰਢ->src[1]->ne[3] == 1);

                        ਅਚਲ int64_t ne00 = ਗੰਢ->src[0]->ne[0];  // K
                        ਅਚਲ int64_t ne01 = ਗੰਢ->src[0]->ne[1];  // Cout
                        ਅਚਲ int64_t ne02 = ਗੰਢ->src[0]->ne[2];  // Cin
                        ਅਚਲ int64_t ne10 = ਗੰਢ->src[1]->ne[0];  // L
                        ਅਚਲ int64_t ne11 = ਗੰਢ->src[1]->ne[1];  // Cin

                        ਜੇ ((ਗੰਢ->src[0]->type == GGML_TYPE_F16 ||
                             ਗੰਢ->src[0]->type == GGML_TYPE_BF16) &&
                            ਗੰਢ->src[1]->type == GGML_TYPE_F32) {
                            cur += ਆਕਾਰ(ggml_fp16_t)*ne00*ne01*ne02;
                            cur += ਆਕਾਰ(ggml_fp16_t)*ne10*ne11;
                        } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ->src[0]->type == GGML_TYPE_F32 &&
                                   ਗੰਢ->src[1]->type == GGML_TYPE_F32) {
                            cur += ਆਕਾਰ(ਦਸ਼ਮਲਵ)*ne00*ne01*ne02;
                            cur += ਆਕਾਰ(ਦਸ਼ਮਲਵ)*ne10*ne11;
                        } ਨਹੀਂਤਾਂ {
                            GGML_ABORT("fatal error");
                        }
                    } ਰੋਕੋ;
                ਹਾਲਤ GGML_OP_CONV_2D:
                ਹਾਲਤ GGML_OP_CONV_3D:
                    {
                        cur = GGML_IM2_ਖਾਨਾ_WORK_ਮਾਪ;
                    } ਰੋਕੋ;
                ਹਾਲਤ GGML_OP_CONV_TRANSPOSE_2D:
                    {
                        ਅਚਲ int64_t ne00 = ਗੰਢ->src[0]->ne[0]; // W
                        ਅਚਲ int64_t ne01 = ਗੰਢ->src[0]->ne[1]; // H
                        ਅਚਲ int64_t ne02 = ਗੰਢ->src[0]->ne[2]; // Channels Out
                        ਅਚਲ int64_t ne03 = ਗੰਢ->src[0]->ne[3]; // Channels In

                        ਅਚਲ int64_t ne10 = ਗੰਢ->src[1]->ne[0]; // W
                        ਅਚਲ int64_t ne11 = ਗੰਢ->src[1]->ne[1]; // H
                        ਅਚਲ int64_t ne12 = ਗੰਢ->src[1]->ne[2]; // Channels In
                        ਅਚਲ int64_t ne13 = ਗੰਢ->src[1]->ne[3]; // Batch

                        GGML_ASSERT(ਗੰਢ->src[0]->type == GGML_TYPE_F16 || ਗੰਢ->src[0]->type == GGML_TYPE_F32);
                        GGML_ASSERT(ਗੰਢ->src[1]->type == GGML_TYPE_F32);

                        cur += ggml_type_ਮਾਪ(ਗੰਢ->src[0]->type) * ne00 * ne01 * ne02 * ne03;
                        cur += ggml_type_ਮਾਪ(ਗੰਢ->src[0]->type) * ne10 * ne11 * ne12 * ne13;

                    } ਰੋਕੋ;
                ਹਾਲਤ GGML_OP_ਸਿਖਰ_K:
                    {
                        cur += ਆਕਾਰ(int32_t)*ਗੰਢ->src[0]->ne[0]*n_tasks;
                    } ਰੋਕੋ;
                ਹਾਲਤ GGML_OP_FLASH_ATTN_EXT:
                    {
                        ਅਚਲ int64_t neq2 = ਗੰਢ->src[0]->ne[2]; // number of query heads
                        ਅਚਲ int64_t DK = ਗੰਢ->src[1]->ne[0];
                        ਅਚਲ int64_t DV = ਗੰਢ->src[2]->ne[0];

                        // Tiled flash attention scratch (tile sizes defined in common.h)
                        // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
                        size_t prefill  = ਆਕਾਰ(ਦਸ਼ਮਲਵ)*(GGML_FA_TILE_Q*DK + 2*GGML_FA_TILE_Q*GGML_FA_TILE_KV + GGML_FA_TILE_Q*DV + GGML_FA_TILE_KV*DV + GGML_FA_TILE_KV*DK)*n_tasks;

                        // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
                        // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
                        size_t n_chunks = n_tasks;
                        size_t decode   = ਆਕਾਰ(ਦਸ਼ਮਲਵ)*(neq2*n_chunks*(2+DV) + n_tasks*(DK + 2*DV));

                        cur += MAX(prefill, decode);
                    } ਰੋਕੋ;
                ਹਾਲਤ GGML_OP_FLASH_ATTN_ਪਿੱਠ:
                    {
                        ਅਚਲ int64_t    D = ਗੰਢ->src[0]->ne[0];
                        ਅਚਲ int64_t ne11 = ggml_ਉੱਪਰ(ਗੰਢ->src[1]->ne[1], GGML_SOFT_MAX_UNROLL);
                        ਅਚਲ int64_t mxDn = MAX(D, ne11) * 2; // *2 because of S and SM in ggml_compute_forward_flash_attn_back
                        ਜੇ (ਗੰਢ->src[1]->type == GGML_TYPE_F32) {
                            cur  = ਆਕਾਰ(ਦਸ਼ਮਲਵ)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += ਆਕਾਰ(ਦਸ਼ਮਲਵ)*mxDn*n_tasks; // this is overestimated by x2
                        } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ->src[1]->type == GGML_TYPE_F16) {
                            cur  = ਆਕਾਰ(ਦਸ਼ਮਲਵ)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += ਆਕਾਰ(ਦਸ਼ਮਲਵ)*mxDn*n_tasks; // this is overestimated by x2
                        } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ->src[1]->type == GGML_TYPE_BF16) {
                            cur  = ਆਕਾਰ(ਦਸ਼ਮਲਵ)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += ਆਕਾਰ(ਦਸ਼ਮਲਵ)*mxDn*n_tasks; // this is overestimated by x2
                        }
                    } ਰੋਕੋ;

                ਹਾਲਤ GGML_OP_CROSS_ENTROPY_LOSS:
                    {
                        cur = ggml_type_ਮਾਪ(ਗੰਢ->type)*(n_tasks + ਗੰਢ->src[0]->ne[0]*n_tasks);
                    } ਰੋਕੋ;
                ਹਾਲਤ GGML_OP_GATED_DELTA_NET:
                    {
                        ਅਚਲ int64_t S_v = ਗੰਢ->src[2]->ne[0];
                        ਅਚਲ int64_t K   = ggml_get_op_params_i32(ਗੰਢ, 0);
                        ਅਚਲ int64_t ਪ੍ਰਤੀ_thread = S_v + (K > 1 ? S_v * S_v : 0);
                        cur = ਪ੍ਰਤੀ_thread * ਆਕਾਰ(ਦਸ਼ਮਲਵ) * n_tasks;
                    } ਰੋਕੋ;
                ਹਾਲਤ GGML_OP_ਗਿਣਤੀ:
                    {
                        GGML_ABORT("fatal error");
                    }
                ਹਾਲਤ GGML_OP_LIGHTNING_INDEXER:
                    {
                        // temp buffer for dequantizing lightning indexer keys
                        ਅਚਲ int64_t ne10 = ਗੰਢ->src[1]->ne[0];
                        cur += ਆਕਾਰ(ਦਸ਼ਮਲਵ)*ne10*n_tasks;
                    } ਰੋਕੋ;
                ਮੂਲ:
                    ਰੋਕੋ;
            }
        }

        work_ਮਾਪ = MAX(work_ਮਾਪ, cur);
    }

    ਜੇ (work_ਮਾਪ > 0) {
        work_ਮਾਪ += CACHE_ਸਤਰ_ਮਾਪ*(n_ਧਾਗੇ);
    }

    cplan.threadpool = threadpool;
    cplan.n_ਧਾਗੇ  = MIN(max_tasks, n_ਧਾਗੇ);
    cplan.work_ਮਾਪ  = work_ਮਾਪ;
    cplan.work_ਅੰਕੜੇ  = ਨਿਰਾ;

    ਵਾਪਸ cplan;
}


// Try to fuse the current node with subsequent nodes for better performance.
// Returns the number of nodes skipped by fusion (>=1), or 0 if no fusion was applied.
ਸਥਿਰ bool ggml_cpu_disable_fusion = false;  // initialized once in ggml_cpu_init(), read-only afterwards

ਸਥਿਰ ਪੂਰਨ ggml_cpu_try_fuse_ops(
        ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph,
        ਅਚਲ ਪੂਰਨ ਗੰਢ_n,
        ਅਚਲ ਬਣਤਰ ggml_compute_params * params,
        ਅਚਲ ਬਣਤਰ ggml_cplan * cplan) {

    ਜੇ (ggml_cpu_disable_fusion || cplan->use_ref) {
        ਵਾਪਸ 0;
    }

    ਬਣਤਰ ggml_tensor * ਗੰਢ = cgraph->nodes[ਗੰਢ_n];

    ਜੇ (ਗੰਢ->op == GGML_OP_RMS_NORM) {
        // RMS_NORM + MUL fusion
        ਅਚਲ ਗਿਣਤੀ ggml_op fuse_ops[] = { GGML_OP_RMS_NORM, GGML_OP_MUL };
        ਜੇ (ggml_can_fuse(cgraph, ਗੰਢ_n, fuse_ops, 2)) {
            ਬਣਤਰ ggml_tensor * mul_ਗੰਢ = cgraph->nodes[ਗੰਢ_n + 1];
            ਅਚਲ ਬਣਤਰ ggml_tensor * mul_w = (mul_ਗੰਢ->src[0] == ਗੰਢ)
                ? mul_ਗੰਢ->src[1] : mul_ਗੰਢ->src[0];
            ਜੇ (ਗੰਢ->src[0]->type  == GGML_TYPE_F32 &&
                mul_ਗੰਢ->type      == GGML_TYPE_F32 &&
                mul_w->type         == GGML_TYPE_F32 &&
                mul_w->ne[0]        == ਗੰਢ->ne[0]   &&
                mul_w->nb[0]        == ਆਕਾਰ(ਦਸ਼ਮਲਵ)) {

                ggml_compute_forward_rms_norm_mul_fused(params, ਗੰਢ, mul_ਗੰਢ);
                ਵਾਪਸ 1;
            }
        }
    }

    ਵਾਪਸ 0;
}

ਸਥਿਰ thread_ret_t ggml_ਗ੍ਰਾਫ_compute_thread(ਖਾਲੀ * ਅੰਕੜੇ) {
    ਬਣਤਰ ggml_compute_state * state = (ਬਣਤਰ ggml_compute_state *) ਅੰਕੜੇ;
    ਬਣਤਰ ggml_threadpool    * tp    = state->threadpool;

    ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph = tp->cgraph;
    ਅਚਲ ਬਣਤਰ ggml_cplan  * cplan  = tp->cplan;

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_ਸੈੱਟ_numa_thread_affinity(state->ith);
#ਨਹੀਂਤਾਂ
    ਸੈੱਟ_numa_thread_affinity(state->ith);
#ਜੇ_ਅੰਤ

    ਬਣਤਰ ggml_compute_params params = {
        /*.ith        =*/ state->ith,
        /*.nth        =*/ atomic_ਲੱਦੋ_explicit(&tp->n_ਗ੍ਰਾਫ, memory_order_relaxed) & GGML_THREADPOOL_N_ਧਾਗੇ_ਨਕਾਬ,
        /*.wsize      =*/ cplan->work_ਮਾਪ,
        /*.wdata      =*/ cplan->work_ਅੰਕੜੇ,
        /*.threadpool =*/ tp,
        /*.use_ref    =*/ cplan->use_ref,
    };

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p\n", state->ith, (ਅਚਲ ਖਾਲੀ *)cplan);
#ਨਹੀਂਤਾਂ
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p last-graph %d\n", state->ith, (ਅਚਲ ਖਾਲੀ *)cplan, state->ਆਖਰੀ_ਗ੍ਰਾਫ);
#ਜੇ_ਅੰਤ

    ਲਈ (ਪੂਰਨ ਗੰਢ_n = 0; ਗੰਢ_n < cgraph->n_nodes && atomic_ਲੱਦੋ_explicit(&tp->abort, memory_order_relaxed) != ਗੰਢ_n; ਗੰਢ_n++) {
        ਬਣਤਰ ggml_tensor * ਗੰਢ = cgraph->nodes[ਗੰਢ_n];

        ਜੇ (ggml_op_is_ਖਾਲੀ_ਥਾਵਾਂ(ਗੰਢ->op)) {
            // skip NOPs
            ਜਾਰੀ;
        }

        ਜੇ ((ਗੰਢ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ਜਾਰੀ;
        }

        // TODO: move fused-op detection into ggml_graph_plan so fusion decisions are made once at planning time
        // Try fused ops, fall back to normal compute
        ਅਚਲ ਪੂਰਨ n_fused = ggml_cpu_try_fuse_ops(cgraph, ਗੰਢ_n, &params, cplan);
        ਜੇ (n_fused > 0) {
            ਗੰਢ_n += n_fused;
        } ਨਹੀਂਤਾਂ {
            ggml_compute_forward(&params, ਗੰਢ);
        }

        ਜੇ (state->ith == 0 && cplan->abort_callback &&
                cplan->abort_callback(cplan->abort_callback_ਅੰਕੜੇ)) {
            atomic_store_explicit(&tp->abort, ਗੰਢ_n + 1, memory_order_relaxed);
            tp->ec    = GGML_ਸਥਿਤੀ_ABORTED;
        }

        ਜੇ (ਗੰਢ_n + 1 < cgraph->n_nodes) {
            ggml_barrier(state->threadpool);
        }
    }

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p\n", state->ith, (ਅਚਲ ਖਾਲੀ *)cplan);
#ਨਹੀਂਤਾਂ
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p last-graph %d\n", state->ith, (ਅਚਲ ਖਾਲੀ *)cplan, state->ਆਖਰੀ_ਗ੍ਰਾਫ);
#ਜੇ_ਅੰਤ

    ggml_barrier(state->threadpool);

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(state->ith);
#ਜੇ_ਅੰਤ

    ਵਾਪਸ 0;
}

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_USE_OPENMP

// check if thread is ready to proceed (exit from polling or sleeping)
// returns true if loops should exit, sets state->pending to indicate new work
ਸਥਿਰ ਸਤਰਅੰਦਰ bool ggml_ਗ੍ਰਾਫ_compute_thread_ready(ਬਣਤਰ ggml_compute_state * state) {
    ਬਣਤਰ ggml_threadpool * threadpool = state->threadpool;

    ਜੇ (state->pending || threadpool->stop || threadpool->pause) { ਵਾਪਸ true; }

    // check for new graph/work
    ਪੂਰਨ n_ਗ੍ਰਾਫ   = atomic_ਲੱਦੋ_explicit(&threadpool->n_ਗ੍ਰਾਫ, memory_order_relaxed);
    ਪੂਰਨ n_ਧਾਗੇ = n_ਗ੍ਰਾਫ & GGML_THREADPOOL_N_ਧਾਗੇ_ਨਕਾਬ;
    ਜੇ (n_ਗ੍ਰਾਫ != state->ਆਖਰੀ_ਗ੍ਰਾਫ) {
        state->pending    = (state->ith < n_ਧਾਗੇ);
        state->ਆਖਰੀ_ਗ੍ਰਾਫ = n_ਗ੍ਰਾਫ;
        ਵਾਪਸ true;
    }

    ਵਾਪਸ false;
}

// sync thread state after polling
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_ਗ੍ਰਾਫ_compute_thread_sync(ਬਣਤਰ ggml_compute_state * state) {
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_TSAN_ENABLED
    atomic_fetch_ਜੋੜੋ_explicit(&state->threadpool->n_ਗ੍ਰਾਫ, 0, memory_order_seq_cst);
    #ਨਹੀਂਤਾਂ
    atomic_thread_fence(memory_order_seq_cst);
    #ਜੇ_ਅੰਤ
    ਅਣਵਰਤਿਆ(state);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ bool ggml_ਗ੍ਰਾਫ_compute_poll_for_work(ਬਣਤਰ ggml_compute_state * state) {
    ਬਣਤਰ ggml_threadpool * threadpool = state->threadpool;

    // This seems to make 0 ... 100 a decent range for polling level across modern processors.
    // Perhaps, we can adjust it dynamically based on load and things.
    ਅਚਲ uint64_t n_ਗੇੜ = 1024UL * 128 * threadpool->poll;

    ਲਈ (uint64_t i=0; !ggml_ਗ੍ਰਾਫ_compute_thread_ready(state) && i < n_ਗੇੜ; i++) {
        // No new work. Keep polling.
        ggml_thread_cpu_relax();
    }

    ਵਾਪਸ state->pending;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ bool ggml_ਗ੍ਰਾਫ_compute_check_for_work(ਬਣਤਰ ggml_compute_state * state) {
    ਬਣਤਰ ggml_threadpool * threadpool = state->threadpool;

    ਜੇ (ggml_ਗ੍ਰਾਫ_compute_poll_for_work(state)) {
        ggml_ਗ੍ਰਾਫ_compute_thread_sync(state);
        ਵਾਪਸ state->pending;
    }

    ggml_mutex_ਤਾਲਾ_shared(&threadpool->mutex);
    ਜਦੋਂ (!ggml_ਗ੍ਰਾਫ_compute_thread_ready(state)) {
        // No new work. Wait for the signal.
        GGML_PRINT_DEBUG("thread #%d waiting for work (sleeping)\n", state->ith);
        ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
    }
    ggml_mutex_unlock_shared(&threadpool->mutex);

    ਵਾਪਸ state->pending;
}

ਸਥਿਰ thread_ret_t ggml_ਗ੍ਰਾਫ_compute_secondary_thread(ਖਾਲੀ* ਅੰਕੜੇ) {
    ਬਣਤਰ ggml_compute_state * state = (ਬਣਤਰ ggml_compute_state *) ਅੰਕੜੇ;
    ਬਣਤਰ ggml_threadpool * threadpool = state->threadpool;

    ggml_thread_apply_ਤਰਜੀਹ(threadpool->prio);
    ਜੇ (ggml_thread_cpumask_is_valid(state->cpumask)) {
        ggml_thread_apply_affinity(state->cpumask);
    }

    ਜਦੋਂ (true) {
        // Check if we need to sleep
        ਜਦੋਂ (threadpool->pause) {
            GGML_PRINT_DEBUG("thread #%d inside pause loop\n", state->ith);
            ggml_mutex_ਤਾਲਾ_shared(&threadpool->mutex);
            ਜੇ (threadpool->pause) {
                ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
            }
            GGML_PRINT_DEBUG("thread #%d resuming after wait\n", state->ith);
            ggml_mutex_unlock_shared(&threadpool->mutex);
        }

        // This needs to be checked for after the cond_wait
        ਜੇ (threadpool->stop) ਰੋਕੋ;

        // Check if there is new work
        // The main thread is the only one that can dispatch new work

        ggml_ਗ੍ਰਾਫ_compute_check_for_work(state);
        ਜੇ (state->pending) {
            state->pending = false;
            ggml_ਗ੍ਰਾਫ_compute_thread(state);
        }
    }

    ਵਾਪਸ (thread_ret_t) 0;
}

// Start processing new graph
ਸਥਿਰ ਖਾਲੀ ggml_ਗ੍ਰਾਫ_compute_kickoff(ਬਣਤਰ ggml_threadpool * threadpool, ਪੂਰਨ n_ਧਾਗੇ)
{
    // Always take the mutex here because the worker threads are doing hybrid poll/wait

    ggml_mutex_ਤਾਲਾ(&threadpool->mutex);

    // Update the number of active threads and the graph count
    ਪੂਰਨ n_ਗ੍ਰਾਫ = atomic_ਲੱਦੋ_explicit(&threadpool->n_ਗ੍ਰਾਫ, memory_order_relaxed) >> GGML_THREADPOOL_N_ਧਾਗੇ_ਬਿੱਟ;
    n_ਗ੍ਰਾਫ = ((n_ਗ੍ਰਾਫ + 1) << GGML_THREADPOOL_N_ਧਾਗੇ_ਬਿੱਟ) | (n_ਧਾਗੇ & GGML_THREADPOOL_N_ਧਾਗੇ_ਨਕਾਬ);

    GGML_PRINT_DEBUG("compute-kickoff: n_threads %d n_graph %d\n", n_ਧਾਗੇ, n_ਗ੍ਰਾਫ);

    // Indicate the graph is ready to be processed
    // We need the full seq-cst fence here because of the polling threads (used in thread_sync)
    atomic_store_explicit(&threadpool->n_ਗ੍ਰਾਫ, n_ਗ੍ਰਾਫ, memory_order_seq_cst);

    ਜੇ (threadpool->pause) {
       // Update main thread prio and affinity to match the threadpool settings
       ggml_thread_apply_ਤਰਜੀਹ(threadpool->prio);
       ਜੇ (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
           ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
       }

       // resume does cond broadcast
       ggml_threadpool_resume_locked(threadpool);
    } ਨਹੀਂਤਾਂ {
       ggml_cond_ਪ੍ਰਸਾਰਣ(&threadpool->cond);
    }

    ggml_mutex_unlock(&threadpool->mutex);
}

#ਜੇ_ਅੰਤ // GGML_USE_OPENMP

ਸਥਿਰ ਬਣਤਰ ggml_threadpool * ggml_threadpool_new_impl(
    ਬਣਤਰ ggml_threadpool_params * tpp,
               ਬਣਤਰ ggml_cgraph * cgraph,
                ਬਣਤਰ ggml_cplan * cplan) {

    ਬਣਤਰ ggml_threadpool * threadpool =
        ggml_aligned_malloc(ਆਕਾਰ(ਬਣਤਰ ggml_threadpool));
    {
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->n_ਗ੍ਰਾਫ          = 0;
        threadpool->n_barrier        = 0;
        threadpool->n_barrier_passed = 0;
        threadpool->current_chunk    = 0;
        threadpool->stop             = false;
        threadpool->pause            = tpp->paused;
        threadpool->abort            = -1;
        threadpool->workers          = ਨਿਰਾ;
        threadpool->n_ਧਾਗੇ        = tpp->n_ਧਾਗੇ;
        threadpool->poll             = tpp->poll;
        threadpool->prio             = tpp->prio;
        threadpool->ec               = GGML_ਸਥਿਤੀ_SUCCESS;
    }

    // Allocate and init workers state
    ਅਚਲ size_t workers_ਮਾਪ = ਆਕਾਰ(ਬਣਤਰ ggml_compute_state) * tpp->n_ਧਾਗੇ;
    ਬਣਤਰ ggml_compute_state * workers = ggml_aligned_malloc(workers_ਮਾਪ);

    memset(workers, 0, workers_ਮਾਪ);
    ਲਈ (ਪੂਰਨ j = 0; j < tpp->n_ਧਾਗੇ; j++) {
        workers[j].threadpool = threadpool;
        workers[j].ith        = j;
    }

    threadpool->workers = workers;

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_OPENMP
    int32_t cpumask_iter = 0;

    // Compute CPU masks for each thread
    ਲਈ (ਪੂਰਨ j = 0; j < tpp->n_ਧਾਗੇ; j++) {
        ggml_thread_cpumask_ਅਗਲਾ(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);
    }
#ਨਹੀਂਤਾਂ // GGML_USE_OPENMP
    ggml_mutex_init(&threadpool->mutex);
    ggml_cond_init(&threadpool->cond);

    // Spin the threads for all workers, and update CPU placements.
    // Place the main thread last (towards the higher numbered CPU cores).

    int32_t cpumask_iter = 0;

    ਲਈ (ਪੂਰਨ j = 1; j < tpp->n_ਧਾਗੇ; j++) {
        ggml_thread_cpumask_ਅਗਲਾ(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);

        int32_t rc = ggml_thread_create(&workers[j].thrd, ਨਿਰਾ, ggml_ਗ੍ਰਾਫ_compute_secondary_thread, &workers[j]);
        GGML_ASSERT(rc == 0);
    }

    ggml_thread_cpumask_ਅਗਲਾ(tpp->cpumask, workers[0].cpumask, tpp->strict_cpu, &cpumask_iter);

    ਜੇ (!threadpool->pause) {
        // Update main thread prio and affinity at the start, otherwise we'll do it in resume
        ggml_thread_apply_ਤਰਜੀਹ(threadpool->prio);
        ਜੇ (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
            ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
        }
    }
#ਜੇ_ਅੰਤ // GGML_USE_OPENMP

    ਵਾਪਸ threadpool;
}

ਬਣਤਰ ggml_threadpool * ggml_threadpool_new(ਬਣਤਰ ggml_threadpool_params * tpp) {
    ਵਾਪਸ ggml_threadpool_new_impl(tpp, ਨਿਰਾ, ਨਿਰਾ);
}

ਗਿਣਤੀ ggml_ਸਥਿਤੀ ggml_ਗ੍ਰਾਫ_compute(ਬਣਤਰ ggml_cgraph * cgraph, ਬਣਤਰ ggml_cplan * cplan) {
    ggml_cpu_init();

    GGML_ASSERT(cplan);
    GGML_ASSERT(cplan->n_ਧਾਗੇ > 0);
    GGML_ASSERT(cplan->work_ਮਾਪ == 0 || cplan->work_ਅੰਕੜੇ != ਨਿਰਾ);

    ਪੂਰਨ n_ਧਾਗੇ                               = cplan->n_ਧਾਗੇ;
    ਬਣਤਰ ggml_threadpool * threadpool = cplan->threadpool;

    bool disposable_threadpool = false;

    ਜੇ (threadpool == ਨਿਰਾ) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
        disposable_threadpool = true;

        ਬਣਤਰ ggml_threadpool_params ttp = ggml_threadpool_params_default(n_ਧਾਗੇ);
        threadpool = ggml_threadpool_new_impl(&ttp, cgraph, cplan);
    } ਨਹੀਂਤਾਂ {
        // Reset some of the parameters that need resetting
        // No worker threads should be accessing the parameters below at this stage
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->current_chunk    = 0;
        threadpool->abort            = -1;
        threadpool->ec               = GGML_ਸਥਿਤੀ_SUCCESS;
    }

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_OPENMP
    ਜੇ (n_ਧਾਗੇ > 1) {
        #ਹਦਾਇਤ omp parallel ਅੰਕ_ਧਾਗੇ(n_ਧਾਗੇ)
        {
            #ਹਦਾਇਤ omp single
            {
                // update the number of threads from the actual number of threads that we got from OpenMP
                n_ਧਾਗੇ = omp_get_ਅੰਕ_ਧਾਗੇ();
                atomic_store_explicit(&threadpool->n_ਗ੍ਰਾਫ, n_ਧਾਗੇ, memory_order_relaxed);
            }

            // Apply thread CPU mask and priority
            ਪੂਰਨ ith = omp_get_thread_ਅੰਕ();

            ggml_thread_apply_ਤਰਜੀਹ(threadpool->prio);
            ਜੇ (ggml_thread_cpumask_is_valid(threadpool->workers[ith].cpumask)) {
                ggml_thread_apply_affinity(threadpool->workers[ith].cpumask);
            }
            ggml_ਗ੍ਰਾਫ_compute_thread(&threadpool->workers[ith]);
        }
    } ਨਹੀਂਤਾਂ {
        atomic_store_explicit(&threadpool->n_ਗ੍ਰਾਫ, 1, memory_order_relaxed);
        ggml_ਗ੍ਰਾਫ_compute_thread(&threadpool->workers[0]);
    }
#ਨਹੀਂਤਾਂ
    ਜੇ (n_ਧਾਗੇ > threadpool->n_ਧਾਗੇ) {
        GGML_LOG_WARN("cplan requested more threads (%d) than available (%d)\n", n_ਧਾਗੇ, threadpool->n_ਧਾਗੇ);
        n_ਧਾਗੇ = threadpool->n_ਧਾਗੇ;
    }

    // Kick all threads to start the new graph
    ggml_ਗ੍ਰਾਫ_compute_kickoff(threadpool, n_ਧਾਗੇ);

    // This is a work thread too
    ggml_ਗ੍ਰਾਫ_compute_thread(&threadpool->workers[0]);
#ਜੇ_ਅੰਤ

    // don't leave affinity set on the main thread
    clear_numa_thread_affinity();

    ਗਿਣਤੀ ggml_ਸਥਿਤੀ ret = threadpool->ec;

    ਜੇ (disposable_threadpool) {
        ggml_threadpool_free(threadpool);
    }

    ਵਾਪਸ ret;
}

ਗਿਣਤੀ ggml_ਸਥਿਤੀ ggml_ਗ੍ਰਾਫ_compute_with_ctx(ਬਣਤਰ ggml_context * ctx, ਬਣਤਰ ggml_cgraph * cgraph, ਪੂਰਨ n_ਧਾਗੇ) {
    ਬਣਤਰ ggml_cplan cplan = ggml_ਗ੍ਰਾਫ_plan(cgraph, n_ਧਾਗੇ, ਨਿਰਾ);

    cplan.work_ਅੰਕੜੇ = (uint8_t *)ggml_new_ਬਫਰ(ctx, cplan.work_ਮਾਪ);

    ਵਾਪਸ ggml_ਗ੍ਰਾਫ_compute(cgraph, &cplan);
}

ਖਾਲੀ ggml_cpu_fp32_ਮੰਜ਼ਿਲ_fp32(ਅਚਲ ਦਸ਼ਮਲਵ * x, ਦਸ਼ਮਲਵ * y, int64_t n) {
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(y, x, n * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
}

ਖਾਲੀ ggml_cpu_fp32_ਮੰਜ਼ਿਲ_fp16(ਅਚਲ ਦਸ਼ਮਲਵ * x, ggml_fp16_t * y, int64_t n) {
    int64_t i = 0;
#ਜੇ defined(__F16C__)
#ਜੇ defined(__AVX512F__)
    ਲਈ (; i + 15 < n; i += 16) {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        __m256i y_vec = _mm512_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256((__m256i *)(y + i), y_vec);
    }
#ਜੇ_ਅੰਤ
    ਲਈ (; i + 7 < n; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m128i y_vec = _mm256_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i *)(y + i), y_vec);
    }
    ਲਈ (; i + 3 < n; i += 4) {
        __m128 x_vec = _mm_loadu_ps(x + i);
        __m128i y_vec = _mm_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storel_epi64((__m128i *)(y + i), y_vec);
    }
#ਜਾਂਜੇ defined(__riscv_zvfh)
    ਲਈ (ਪੂਰਨ vl; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(&x[i], vl);
        vfloat16m1_t vy = __riscv_vfncvt_f_f_w_f16m1(vx, vl);
        __riscv_vse16_v_f16m1((_Float16 *)&y[i], vy, vl);
    }
#ਜਾਂਜੇ defined(__VXE__) || defined(__VXE2__)
    ਲਈ (; i + 7 < n; i += 8) {
        ਅਚਲ uint32x4_t v_yl = __lzs_f32cx4_to_f16(vec_xl(0, x + i + 0));
        ਅਚਲ uint32x4_t v_yh = __lzs_f32cx4_to_f16(vec_xl(0, x + i + 4));
        vec_xst(vec_pack(v_yl, v_yh), 0, (uint16_t *)(y + i));
    }
#ਜੇ_ਅੰਤ
    ਲਈ (; i < n; ++i) {
        y[i] = GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(x[i]);
    }
}

ਖਾਲੀ ggml_cpu_fp16_ਮੰਜ਼ਿਲ_fp32(ਅਚਲ ggml_fp16_t * x, ਦਸ਼ਮਲਵ * y, int64_t n) {
    int64_t i = 0;
#ਜੇ defined(__F16C__)
#ਜੇ defined(__AVX512F__)
    ਲਈ (; i + 15 < n; i += 16) {
        __m256i x_vec = _mm256_loadu_si256((ਅਚਲ __m256i *)(x + i));
        __m512 y_vec = _mm512_cvtph_ps(x_vec);
        _mm512_storeu_ps(y + i, y_vec);
    }
#ਜੇ_ਅੰਤ
    ਲਈ (; i + 7 < n; i += 8) {
        __m128i x_vec = _mm_loadu_si128((ਅਚਲ __m128i *)(x + i));
        __m256 y_vec = _mm256_cvtph_ps(x_vec);
        _mm256_storeu_ps(y + i, y_vec);
    }
    ਲਈ (; i + 3 < n; i += 4) {
        __m128i x_vec = _mm_loadl_epi64((ਅਚਲ __m128i *)(x + i));
        __m128 y_vec = _mm_cvtph_ps(x_vec);
        _mm_storeu_ps(y + i, y_vec);
    }

#ਜਾਂਜੇ defined(__riscv_v_intrinsic) && defined(__riscv_zvfhmin)
    // calculate step size
    ਅਚਲ ਪੂਰਨ epr = __riscv_vsetvlmax_e16m2();
    ਅਚਲ ਪੂਰਨ ਕਦਮ = epr * 2;
    ਅਚਲ ਪੂਰਨ np = (n & ~(ਕਦਮ - 1));

    // unroll by 2
    ਲਈ (; i < np; i += ਕਦਮ) {
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((ਅਚਲ _Float16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vfloat16m2_t ax1 = __riscv_vle16_v_f16m2((ਅਚਲ _Float16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvt_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    ਪੂਰਨ vl;
    ਲਈ (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((ਅਚਲ _Float16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }

#ਜੇ_ਅੰਤ

    ਲਈ (; i < n; ++i) {
        y[i] = GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[i]);
    }
}

ਖਾਲੀ ggml_cpu_fp32_ਮੰਜ਼ਿਲ_bf16(ਅਚਲ ਦਸ਼ਮਲਵ * x, ggml_bf16_t * y, int64_t n) {
    int64_t i = 0;
    ਲਈ (; i < n; ++i) {
        y[i] = GGML_FP32_ਮੰਜ਼ਿਲ_BF16(x[i]);
    }
}

ਖਾਲੀ ggml_cpu_fp32_ਮੰਜ਼ਿਲ_i32(ਅਚਲ ਦਸ਼ਮਲਵ * x, int32_t * y, int64_t n) {
    int64_t i = 0;
    ਲਈ (; i < n; ++i) {
        y[i] = x[i];
    }
}

ਖਾਲੀ ggml_cpu_bf16_ਮੰਜ਼ਿਲ_fp32(ਅਚਲ ggml_bf16_t * x, ਦਸ਼ਮਲਵ * y, int64_t n) {
    int64_t i = 0;
#ਜੇ defined(__AVX2__)
#ਜੇ defined(__AVX512F__)
    ਲਈ (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i,
                        _mm512_castsi512_ps(
                            _mm512_slli_epi32(
                                _mm512_cvtepu16_epi32(
                                    _mm256_loadu_si256(
                                        (ਅਚਲ __m256i *)(x + i))),
                                16)));
    }
#ਜੇ_ਅੰਤ
    ਲਈ (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i,
                        _mm256_castsi256_ps(
                            _mm256_slli_epi32(
                                _mm256_cvtepu16_epi32(
                                    _mm_loadu_si128(
                                        (ਅਚਲ __m128i *)(x + i))),
                                16)));
    }
#ਜਾਂਜੇ defined(__riscv_v_intrinsic) && defined(__riscv_zvfbfmin)
    // calculate step size
    ਅਚਲ ਪੂਰਨ epr = __riscv_vsetvlmax_e16m2();
    ਅਚਲ ਪੂਰਨ ਕਦਮ = epr * 2;
    ਅਚਲ ਪੂਰਨ np = (n & ~(ਕਦਮ - 1));

    // unroll by 2
    ਲਈ (; i < np; i += ਕਦਮ) {
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((ਅਚਲ __bf16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vbfloat16m2_t ax1 = __riscv_vle16_v_bf16m2((ਅਚਲ __bf16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    ਪੂਰਨ vl;
    ਲਈ (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((ਅਚਲ __bf16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }
#ਜੇ_ਅੰਤ
    ਲਈ (; i < n; i++) {
        y[i] = GGML_BF16_ਮੰਜ਼ਿਲ_FP32(x[i]);
    }
}

ਪੂਰਨ ggml_cpu_has_avx(ਖਾਲੀ) {
#ਜੇ defined(__AVX__)
    ਵਾਪਸ 1;
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ 0;
#ਜੇ_ਅੰਤ
}

ਪੂਰਨ ggml_cpu_has_avx_vnni(ਖਾਲੀ) {
#ਜੇ defined(__AVXVNNI__)
    ਵਾਪਸ 1;
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ 0;
#ਜੇ_ਅੰਤ
}

ਪੂਰਨ ggml_cpu_has_avx2(ਖਾਲੀ) {
#ਜੇ defined(__AVX2__)
    ਵਾਪਸ 1;
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ 0;
#ਜੇ_ਅੰਤ
}

ਪੂਰਨ ggml_cpu_has_avx512(ਖਾਲੀ) {
#ਜੇ defined(__AVX512F__)
    ਵਾਪਸ 1;
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ 0;
#ਜੇ_ਅੰਤ
}

ਪੂਰਨ ggml_cpu_has_avx512_vbmi(ਖਾਲੀ) {
#ਜੇ defined(__AVX512VBMI__)
    ਵਾਪਸ 1;
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ 0;
#ਜੇ_ਅੰਤ
}

ਪੂਰਨ ggml_cpu_has_avx512_vnni(ਖਾਲੀ) {
#ਜੇ defined(__AVX512VNNI__)
    ਵਾਪਸ 1;
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ 0;
#ਜੇ_ਅੰਤ
}

ਪੂਰਨ ggml_cpu_has_avx512_bf16(ਖਾਲੀ) {
#ਜੇ defined(__AVX512BF16__)
    ਵਾਪਸ 1;
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ 0;
#ਜੇ_ਅੰਤ
}

ਪੂਰਨ ggml_cpu_has_amx_int8(ਖਾਲੀ) {
#ਜੇ defined(__AMX_INT8__)
    ਵਾਪਸ 1;
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ 0;
#ਜੇ_ਅੰਤ
}

ਪੂਰਨ ggml_cpu_has_bmi2(ਖਾਲੀ) {
#ਜੇ defined(__BMI2__)
    ਵਾਪਸ 1;
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ 0;
#ਜੇ_ਅੰਤ
}

ਪੂਰਨ ggml_cpu_has_fma(ਖਾਲੀ) {
#ਜੇ defined(__FMA__)
    ਵਾਪਸ 1;
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ 0;
#ਜੇ_ਅੰਤ
}

ਪੂਰਨ ggml_cpu_has_arm_fma(ਖਾਲੀ) {
#ਜੇ defined(__ARM_FEATURE_FMA)
    ਵਾਪਸ 1;
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ 0;
#ਜੇ_ਅੰਤ
}

ਪੂਰਨ ggml_cpu_has_riscv_v(ਖਾਲੀ) {
#ਜੇ defined(__riscv_v_intrinsic)
    ਵਾਪਸ 1;
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ 0;
#ਜੇ_ਅੰਤ
}

ਪੂਰਨ ggml_cpu_get_rvv_vlen(ਖਾਲੀ) {
#ਜੇ defined(__riscv) && defined(__riscv_v_intrinsic)
    ਵਾਪਸ ggml_riscv_arch_features.rvv_vlen;
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ 0;
#ਜੇ_ਅੰਤ
}

ਪੂਰਨ ggml_cpu_has_f16c(ਖਾਲੀ) {
#ਜੇ defined(__F16C__)
    ਵਾਪਸ 1;
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ 0;
#ਜੇ_ਅੰਤ
}

ਪੂਰਨ ggml_cpu_has_fp16_va(ਖਾਲੀ) {
#ਜੇ defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    ਵਾਪਸ 1;
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ 0;
#ਜੇ_ਅੰਤ
}

ਪੂਰਨ ggml_cpu_has_wasm_simd(ਖਾਲੀ) {
#ਜੇ defined(__wasm_simd128__)
    ਵਾਪਸ 1;
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ 0;
#ਜੇ_ਅੰਤ
}

ਪੂਰਨ ggml_cpu_has_llamafile(ਖਾਲੀ) {
#ਜੇ defined(GGML_USE_LLAMAFILE)
    ਵਾਪਸ 1;
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ 0;
#ਜੇ_ਅੰਤ
}

ਪੂਰਨ ggml_cpu_has_sse3(ਖਾਲੀ) {
#ਜੇ defined(__SSE3__)
    ਵਾਪਸ 1;
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ 0;
#ਜੇ_ਅੰਤ
}

ਪੂਰਨ ggml_cpu_has_ssse3(ਖਾਲੀ) {
#ਜੇ defined(__SSSE3__)
    ਵਾਪਸ 1;
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ 0;
#ਜੇ_ਅੰਤ
}

ਪੂਰਨ ggml_cpu_has_vsx(ਖਾਲੀ) {
#ਜੇ defined(__POWER9_VECTOR__)
    ਵਾਪਸ 1;
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ 0;
#ਜੇ_ਅੰਤ
}

ਪੂਰਨ ggml_cpu_has_vxe(ਖਾਲੀ) {
#ਜੇ defined(__VXE__) || defined(__VXE2__)
    ਵਾਪਸ 1;
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ 0;
#ਜੇ_ਅੰਤ
}

ਪੂਰਨ ggml_cpu_has_neon(ਖਾਲੀ) {
#ਜੇ defined(__ARM_ARCH) && defined(__ARM_NEON)
    ਵਾਪਸ 1;
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ 0;
#ਜੇ_ਅੰਤ
}

ਪੂਰਨ ggml_cpu_has_dotprod(ਖਾਲੀ) {
#ਜੇ defined(__ARM_ARCH) && defined(__ARM_FEATURE_DOTPROD)
    ਵਾਪਸ 1;
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ 0;
#ਜੇ_ਅੰਤ
}

ਪੂਰਨ ggml_cpu_has_sve(ਖਾਲੀ) {
#ਜੇ defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    ਵਾਪਸ 1;
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ 0;
#ਜੇ_ਅੰਤ
}

ਪੂਰਨ ggml_cpu_has_matmul_int8(ਖਾਲੀ) {
#ਜੇ defined(__ARM_ARCH) && defined(__ARM_FEATURE_MATMUL_INT8)
    ਵਾਪਸ 1;
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ 0;
#ਜੇ_ਅੰਤ
}

ਪੂਰਨ ggml_cpu_get_sve_cnt(ਖਾਲੀ) {
#ਜੇ defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    ਵਾਪਸ ggml_arm_arch_features.sve_cnt;
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ 0;
#ਜੇ_ਅੰਤ
}

ਪੂਰਨ ggml_cpu_has_sme(ਖਾਲੀ) {
#ਜੇ defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME)
    ਵਾਪਸ 1;
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ 0;
#ਜੇ_ਅੰਤ
}

ਪੂਰਨ ggml_cpu_has_sme2(ਖਾਲੀ) {
#ਜੇ defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME2)
    ਵਾਪਸ 1;
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ 0;
#ਜੇ_ਅੰਤ
}

ਖਾਲੀ ggml_cpu_init(ਖਾਲੀ) {
    // needed to initialize ggml_time
    {
        ਬਣਤਰ ggml_init_params params = { 0, ਨਿਰਾ, false };
        ਬਣਤਰ ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    ggml_critical_section_ਸ਼ੁਰੂ();

    ਸਥਿਰ bool is_ਪਹਿਲਾ_call = true;

    ਜੇ (is_ਪਹਿਲਾ_call) {
        // initialize GELU, Quick GELU, SILU and EXP F32 tables
        {
            ਅਚਲ uint64_t t_ਸ਼ੁਰੂ = ggml_ਸਮਾਂ_us(); ਅਣਵਰਤਿਆ(t_ਸ਼ੁਰੂ);

            ਲਈ (ਪੂਰਨ i = 0; i < (1 << 16); ++i) {
                ਸੰਘ {
                    uint16_t u16;
                    ggml_fp16_t fp16;
                } u = {i};
                ਦਸ਼ਮਲਵ f = GGML_COMPUTE_FP16_ਮੰਜ਼ਿਲ_FP32(u.fp16);
                ggml_ਸਾਰਣੀ_f32_f16[i] = f;
                ggml_ਸਾਰਣੀ_gelu_f16[i] = GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(ggml_gelu_f32(f));
                ggml_ਸਾਰਣੀ_gelu_ਤੇਜ਼_f16[i] = GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(ggml_gelu_ਤੇਜ਼_f32(f));
            }

            // initialize E8M0 half table (256 entries)
            ਲਈ (ਪੂਰਨ i = 0; i < (1 << 8); ++i) {
                ggml_ਸਾਰਣੀ_f32_e8m0_half[i] = GGML_E8_M0_ਮੰਜ਼ਿਲ_FP32_HALF(i);
            }

            // initialize UE4M3 table (256 entries)
            ਲਈ (ਪੂਰਨ i = 0; i < (1 << 8); ++i) {
                ggml_ਸਾਰਣੀ_f32_ue4m3[i] = ggml_ue4m3_ਮੰਜ਼ਿਲ_fp32(i);
            }

            ਅਚਲ uint64_t t_ਸਿਰਾ = ggml_ਸਮਾਂ_us(); ਅਣਵਰਤਿਆ(t_ਸਿਰਾ);

            GGML_PRINT_DEBUG("%s: GELU, Quick GELU, SILU and EXP tables initialized in %f ms\n", __func__, (t_ਸਿਰਾ - t_ਸ਼ੁਰੂ)/1000.0);

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_OPENMP
            //if (!getenv("OMP_WAIT_POLICY")) {
            //    // set the wait policy to active, so that OpenMP threads don't sleep
            //    setenv("OMP_WAIT_POLICY", "active", 0)
            //}

            ਜੇ (!getenv("KMP_BLOCKTIME")) {
                // set the time to wait before sleeping a thread
                // this is less aggressive than setting the wait policy to active, but should achieve similar results in most cases
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ _WIN32
                _putenv_s("KMP_BLOCKTIME", "200"); // 200ms
#ਨਹੀਂਤਾਂ
                setenv("KMP_BLOCKTIME", "200", 0); // 200ms
#ਜੇ_ਅੰਤ
            }
#ਜੇ_ਅੰਤ
        }

#ਜੇ defined(__ARM_ARCH)
        ggml_init_arm_arch_features();
#ਜੇ_ਅੰਤ

#ਜੇ defined(__riscv)
        ggml_init_riscv_arch_features();
#ਜੇ_ਅੰਤ

        {
            ਅਚਲ ਅੱਖਰ * env = getenv("GGML_CPU_DISABLE_FUSION");
            ggml_cpu_disable_fusion = (env != ਨਿਰਾ && atoi(env) == 1);
        }

        is_ਪਹਿਲਾ_call = false;
    }

    ggml_critical_section_ਸਿਰਾ();
}
