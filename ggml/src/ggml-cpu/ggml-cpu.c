#ವ್ಯಾಖ್ಯಾನಿಸು _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#ವ್ಯಾಖ್ಯಾನಿಸು USE_MATH_ನಿರೂಪಿಸುತ್ತದೆ // For M_PI on MSVC

#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend.h"
#ಅನ್ತರ್ಭಾವ "traits.h"
#ಅನ್ತರ್ಭಾವ "ggml-cpu-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "quants.h"
#ಅನ್ತರ್ಭಾವ "ggml-threading.h"
#ಅನ್ತರ್ಭಾವ "unary-ops.h"
#ಅನ್ತರ್ಭಾವ "binary-ops.h"
#ಅನ್ತರ್ಭಾವ "vec.h"
#ಅನ್ತರ್ಭಾವ "ops.h"
#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "common.h"
#ಅನ್ತರ್ಭಾವ "tiled/tiled.h"

#ಆದರೆ defined(_MSC_VER) || defined(__MINGW32__)
#ಅನ್ತರ್ಭಾವ <malloc.h> // using malloc.h with MSC/MINGW
#ಅಥವಾಆದರೆ !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#ಅನ್ತರ್ಭಾವ <alloca.h>
#ಆದರೆ_ಅಂತ್ಯ

#ಅನ್ತರ್ಭಾವ <assert.h>
#ಅನ್ತರ್ಭಾವ <errno.h>
#ಅನ್ತರ್ಭಾವ <time.h>
#ಅನ್ತರ್ಭಾವ <math.h>
#ಅನ್ತರ್ಭಾವ <stdlib.h>
#ಅನ್ತರ್ಭಾವ <string.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <inttypes.h>
#ಅನ್ತರ್ಭಾವ <stdio.h>
#ಅನ್ತರ್ಭಾವ <float.h>
#ಅನ್ತರ್ಭಾವ <limits.h>
#ಅನ್ತರ್ಭಾವ <stdarg.h>
#ಅನ್ತರ್ಭಾವ <signal.h>
#ಆದರೆ defined(__gnu_linux__)
#ಅನ್ತರ್ಭಾವ <syscall.h>
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_OPENMP
#ಅನ್ತರ್ಭಾವ <omp.h>
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_MATMUL_INT8)
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು GGML_USE_LLAMAFILE
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_LLAMAFILE
#ಅನ್ತರ್ಭಾವ "llamafile/sgemm.h"
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_CPU_RISCV64_SPACEMIT
#    include "spacemit/ime.h"
#ಆದರೆ_ಅಂತ್ಯ

// Note: once we move threading into a separate C++ file
// will use std::hardware_destructive_interference_size instead of hardcoding it here
// and we'll use C++ attribute syntax.
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_CACHE_ಸಾಲು  64

#ಆದರೆ defined(__clang__) || defined(__GNUC__)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_CACHE_ALIGN __attribute__((aligned(GGML_CACHE_ಸಾಲು)))
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(__has_feature)
#ಆದರೆ __has_feature(thread_sanitizer)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_TSAN_ENABLED 1
#ಆದರೆ_ಅಂತ್ಯ
#ಇಲ್ಲದಿದ್ದರೆ  // __has_feature
#ಆದರೆ defined(__SANITIZE_THREAD__)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_TSAN_ENABLED 1
#ಆದರೆ_ಅಂತ್ಯ
#ಆದರೆ_ಅಂತ್ಯ // __has_feature

#ವ್ಯಾಖ್ಯಾನಿಸು ಬಳಸದ GGML_ಬಳಸದ
#ವ್ಯಾಖ್ಯಾನಿಸು ಅದಲುಬದಲು(x, y, T) ಮಾಡು { T ಅದಲುಬದಲು = x; (x) = y; (y) = ಅದಲುಬದಲು; } ಆಗ (0)

// precomputed f32 table for f16 (256 KB) (simd-mappings.h)
ದಶಮಾಂಶ ggml_ಕೋಷ್ಟಕ_f32_f16[1 << 16];

// precomputed f32 table for e8m0 half (1 KB) (simd-mappings.h)
ದಶಮಾಂಶ ggml_ಕೋಷ್ಟಕ_f32_e8m0_half[1 << 8];

// precomputed f32 table for ue4m3 (1 KB) (simd-mappings.h)
ದಶಮಾಂಶ ggml_ಕೋಷ್ಟಕ_f32_ue4m3[1 << 8];

#ಆದರೆ defined(__ARM_ARCH)
ರಚನೆ ggml_arm_arch_features_type {
    ಪೂರ್ಣಾಂಕ sve_cnt;
} ggml_arm_arch_features = { 0 };
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(__riscv)
ರಚನೆ ggml_riscv_arch_features_type {
    ಪೂರ್ಣಾಂಕ rvv_vlen;
} ggml_riscv_arch_features = { 0 };
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(_WIN32)

#ವ್ಯಾಖ್ಯಾನಿಸು WIN32_LEAN_AND_ಸರಾಸರಿ
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NOMINMAX
    #ವ್ಯಾಖ್ಯಾನಿಸು NOMINMAX
#ಆದರೆ_ಅಂತ್ಯ
#ಅನ್ತರ್ಭಾವ <windows.h>

#ಆದರೆ defined(_MSC_VER) && !defined(__clang__)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_CACHE_ALIGN __declspec(align(GGML_CACHE_ಸಾಲು))

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಅಸ್ಥಿರ LONG atomic_int;
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ atomic_int atomic_bool;
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ atomic_int atomic_flag;

#ವ್ಯಾಖ್ಯಾನಿಸು ATOMIC_FLAG_INIT 0

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಎಣಿಕೆ {
    memory_order_relaxed,
    memory_order_consume,
    memory_order_acquire,
    memory_order_release,
    memory_order_acq_rel,
    memory_order_seq_cst
} memory_order;

ಸ್ಥಿರ ಶೂನ್ಯ atomic_store(atomic_int * ptr, LONG val) {
    InterlockedExchange(ptr, val);
}
ಸ್ಥಿರ ಶೂನ್ಯ atomic_store_explicit(atomic_int * ptr, LONG val, memory_order mo) {
    // TODO: add support for explicit memory order
    InterlockedExchange(ptr, val);
}
ಸ್ಥಿರ LONG atomic_ತುಂಬು(atomic_int * ptr) {
    ಹಿಂತಿರುಗಿ InterlockedCompareExchange(ptr, 0, 0);
}
ಸ್ಥಿರ LONG atomic_ತುಂಬು_explicit(atomic_int * ptr, memory_order mo) {
    // TODO: add support for explicit memory order
    ಹಿಂತಿರುಗಿ InterlockedCompareExchange(ptr, 0, 0);
}
ಸ್ಥಿರ LONG atomic_fetch_ಕೂಡಿಸು(atomic_int * ptr, LONG inc) {
    ಹಿಂತಿರುಗಿ Interlocked_Exchange_ಕೂಡಿಸು(ptr, inc);
}
ಸ್ಥಿರ LONG atomic_fetch_ಕೂಡಿಸು_explicit(atomic_int * ptr, LONG inc, memory_order mo) {
    // TODO: add support for explicit memory order
    ಹಿಂತಿರುಗಿ Interlocked_Exchange_ಕೂಡಿಸು(ptr, inc);
}
ಸ್ಥಿರ atomic_bool atomic_flag_test_and_ಗಣ(atomic_flag * ptr) {
    ಹಿಂತಿರುಗಿ InterlockedExchange(ptr, 1);
}
ಸ್ಥಿರ ಶೂನ್ಯ atomic_flag_clear(atomic_flag * ptr) {
    InterlockedExchange(ptr, 0);
}
ಸ್ಥಿರ ಶೂನ್ಯ atomic_thread_fence(memory_order mo) {
    ಸ್ಮರಣೆ_Barrier();
}
#ಇಲ್ಲದಿದ್ದರೆ // clang
#ಅನ್ತರ್ಭಾವ <stdatomic.h>
#ಆದರೆ_ಅಂತ್ಯ

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ HANDLE pthread_t;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ DWORD thread_ret_t;
ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ pthread_create(pthread_t * ಹೊರಹೋಗು, ಶೂನ್ಯ * ಬಳಸದ_2, thread_ret_t(*func)(ಶೂನ್ಯ *), ಶೂನ್ಯ * arg) {
    (ಶೂನ್ಯ) ಬಳಸದ_2;
    HANDLE handle = CreateThread(ಖಾಲಿ, 0, (LPTHREAD_ಆರಂಭ_ROUTINE) func, arg, 0, ಖಾಲಿ);
    ಆದರೆ (handle == ಖಾಲಿ)
    {
        ಹಿಂತಿರುಗಿ EAGAIN;
    }

    *ಹೊರಹೋಗು = handle;
    ಹಿಂತಿರುಗಿ 0;
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ pthread_join(pthread_t thread, ಶೂನ್ಯ * ಬಳಸದ_2) {
    (ಶೂನ್ಯ) ಬಳಸದ_2;
    ಪೂರ್ಣಾಂಕ ret = (ಪೂರ್ಣಾಂಕ) WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    ಹಿಂತಿರುಗಿ ret;
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ sched_yield (ಶೂನ್ಯ) {
    Sleep (0);
    ಹಿಂತಿರುಗಿ 0;
}
#ಇಲ್ಲದಿದ್ದರೆ

#ಅನ್ತರ್ಭಾವ <pthread.h>
#ಅನ್ತರ್ಭಾವ <stdatomic.h>
#ಅನ್ತರ್ಭಾವ <sched.h>
#ಆದರೆ defined(__FreeBSD__)
#ಅನ್ತರ್ಭಾವ <pthread_np.h>
#ಆದರೆ_ಅಂತ್ಯ

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಶೂನ್ಯ * thread_ret_t;

#ಅನ್ತರ್ಭಾವ <sys/types.h>
#ಅನ್ತರ್ಭಾವ <sys/stat.h>
#ಅನ್ತರ್ಭಾವ <unistd.h>

#ಆದರೆ_ಅಂತ್ಯ

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ pthread_t ggml_thread_t;

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_THREADPOOL_N_ಎಳೆಗಳು_ಮುಖವಾಡ (0xffffU)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_THREADPOOL_N_ಎಳೆಗಳು_ಬಿಟ್ಟುಗಳು (16)

#ಆದರೆ defined(__APPLE__)
#ಅನ್ತರ್ಭಾವ <unistd.h>
#ಅನ್ತರ್ಭಾವ <mach/mach.h>
#ಅನ್ತರ್ಭಾವ <TargetConditionals.h>
#ಆದರೆ_ಅಂತ್ಯ

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_type_traits_cpu type_traits_cpu[GGML_TYPE_ಎಣಿಕೆ] = {
    [GGML_TYPE_F32] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_ಗಮ್ಯ_fp32,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_f32,
        .vec_dot_type             = GGML_TYPE_F32,
        .nrows                    = 1,
    },
    [GGML_TYPE_F16] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_ಗಮ್ಯ_fp16,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_f16,
        .vec_dot_type             = GGML_TYPE_F16,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q1_0] = {
        .from_float               = quantize_ಅಡ್ಡಸಾಲು_q1_0,
        .vec_dot                  = ggml_vec_dot_q1_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q2_0] = {
        .from_float               = quantize_ಅಡ್ಡಸಾಲು_q2_0,
        .vec_dot                  = ggml_vec_dot_q2_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q4_0] = {
        .from_float               = quantize_ಅಡ್ಡಸಾಲು_q4_0,
        .vec_dot                  = ggml_vec_dot_q4_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
#ಆದರೆ defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#ಇಲ್ಲದಿದ್ದರೆ
        .nrows                    = 1,
#ಆದರೆ_ಅಂತ್ಯ
    },
    [GGML_TYPE_Q4_1] = {
        .from_float               = quantize_ಅಡ್ಡಸಾಲು_q4_1,
        .vec_dot                  = ggml_vec_dot_q4_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
#ಆದರೆ defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#ಇಲ್ಲದಿದ್ದರೆ
        .nrows                    = 1,
#ಆದರೆ_ಅಂತ್ಯ
    },
    [GGML_TYPE_Q5_0] = {
        .from_float               = quantize_ಅಡ್ಡಸಾಲು_q5_0,
        .vec_dot                  = ggml_vec_dot_q5_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q5_1] = {
        .from_float               = quantize_ಅಡ್ಡಸಾಲು_q5_1,
        .vec_dot                  = ggml_vec_dot_q5_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q8_0] = {
        .from_float               = quantize_ಅಡ್ಡಸಾಲು_q8_0,
        .vec_dot                  = ggml_vec_dot_q8_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
#ಆದರೆ defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#ಇಲ್ಲದಿದ್ದರೆ
        .nrows                    = 1,
#ಆದರೆ_ಅಂತ್ಯ
    },
    [GGML_TYPE_Q8_1] = {
        .from_float               = quantize_ಅಡ್ಡಸಾಲು_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
        .nrows                    = 1,
    },
    [GGML_TYPE_MXFP4] = {
        .from_float               = quantize_ಅಡ್ಡಸಾಲು_mxfp4,
        .vec_dot                  = ggml_vec_dot_mxfp4_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_NVFP4] = {
        .from_float               = quantize_ಅಡ್ಡಸಾಲು_nvfp4,
        .vec_dot                  = ggml_vec_dot_nvfp4_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q2_K] = {
        .from_float               = quantize_ಅಡ್ಡಸಾಲು_q2_K,
        .vec_dot                  = ggml_vec_dot_q2_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q3_K] = {
        .from_float               = quantize_ಅಡ್ಡಸಾಲು_q3_K,
        .vec_dot                  = ggml_vec_dot_q3_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q4_K] = {
        .from_float               = quantize_ಅಡ್ಡಸಾಲು_q4_K,
        .vec_dot                  = ggml_vec_dot_q4_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
#ಆದರೆ defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#ಇಲ್ಲದಿದ್ದರೆ
        .nrows                    = 1,
#ಆದರೆ_ಅಂತ್ಯ
    },
    [GGML_TYPE_Q5_K] = {
        .from_float               = quantize_ಅಡ್ಡಸಾಲು_q5_K,
        .vec_dot                  = ggml_vec_dot_q5_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q6_K] = {
        .from_float               = quantize_ಅಡ್ಡಸಾಲು_q6_K,
        .vec_dot                  = ggml_vec_dot_q6_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
#ಆದರೆ defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#ಇಲ್ಲದಿದ್ದರೆ
        .nrows                    = 1,
#ಆದರೆ_ಅಂತ್ಯ
    },
    [GGML_TYPE_IQ2_XXS] = {
        .from_float               = ಖಾಲಿ,
        .vec_dot                  = ggml_vec_dot_iq2_xxs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_XS] = {
        .from_float               = ಖಾಲಿ,
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
        .from_float               = ಖಾಲಿ,
        .vec_dot                  = ggml_vec_dot_iq1_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_M] = {
        .from_float               = ಖಾಲಿ,
        .vec_dot                  = ggml_vec_dot_iq1_m_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_NL] = {
        .from_float               = quantize_ಅಡ್ಡಸಾಲು_iq4_nl,
        .vec_dot                  = ggml_vec_dot_iq4_nl_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_XS] = {
        .from_float               = quantize_ಅಡ್ಡಸಾಲು_iq4_xs,
        .vec_dot                  = ggml_vec_dot_iq4_xs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q8_K] = {
        .from_float               = quantize_ಅಡ್ಡಸಾಲು_q8_K,
    },
    [GGML_TYPE_BF16] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_ಗಮ್ಯ_bf16,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_bf16,
        .vec_dot_type             = GGML_TYPE_BF16,
        .nrows                    = 1,
    },
    [GGML_TYPE_TQ1_0] = {
        .from_float               = quantize_ಅಡ್ಡಸಾಲು_tq1_0,
        .vec_dot                  = ggml_vec_dot_tq1_0_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_TQ2_0] = {
        .from_float               = quantize_ಅಡ್ಡಸಾಲು_tq2_0,
        .vec_dot                  = ggml_vec_dot_tq2_0_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_I32] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_ಗಮ್ಯ_i32,
    },
};

ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_type_traits_cpu * ggml_get_type_traits_cpu(ಎಣಿಕೆ ggml_type type) {
    ಹಿಂತಿರುಗಿ &type_traits_cpu[type];
}

//
// Threading defs
//

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ pthread_t          ggml_thread_t;

#ಆದರೆ defined(_WIN32)

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ CONDITION_VARIABLE ggml_cond_t;
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ SRWLOCK            ggml_mutex_t;

#ವ್ಯಾಖ್ಯಾನಿಸು ggml_mutex_init(m)   InitializeSRWLock(m)
#ವ್ಯಾಖ್ಯಾನಿಸು ggml_mutex_destroy(m)
#ವ್ಯಾಖ್ಯಾನಿಸು ggml_mutex_ಬೀಗ(m)   AcquireSRWLockExclusive(m)
#ವ್ಯಾಖ್ಯಾನಿಸು ggml_mutex_unlock(m) ReleaseSRWLockExclusive(m)
#ವ್ಯಾಖ್ಯಾನಿಸು ggml_mutex_ಬೀಗ_shared(m)   AcquireSRWLockShared(m)
#ವ್ಯಾಖ್ಯಾನಿಸು ggml_mutex_unlock_shared(m) ReleaseSRWLockShared(m)

#ವ್ಯಾಖ್ಯಾನಿಸು ggml_cond_init(c)    InitializeConditionVariable(c)
#ವ್ಯಾಖ್ಯಾನಿಸು ggml_cond_destroy(c)
#ವ್ಯಾಖ್ಯಾನಿಸು ggml_cond_wait(c, m) SleepConditionVariableSRW(c, m, INFINITE, CONDITION_VARIABLE_LOCKMODE_SHARED)
#ವ್ಯಾಖ್ಯಾನಿಸು ggml_cond_ಪ್ರಸಾರ(c) WakeAllConditionVariable(c)

#ವ್ಯಾಖ್ಯಾನಿಸು ggml_thread_create pthread_create
#ವ್ಯಾಖ್ಯಾನಿಸು ggml_thread_join   pthread_join

#ಇಲ್ಲದಿದ್ದರೆ

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ pthread_cond_t     ggml_cond_t;
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ pthread_mutex_t    ggml_mutex_t;

#ವ್ಯಾಖ್ಯಾನಿಸು ggml_mutex_init(m)          pthread_mutex_init(m, ಖಾಲಿ)
#ವ್ಯಾಖ್ಯಾನಿಸು ggml_mutex_destroy(m)       pthread_mutex_destroy(m)
#ವ್ಯಾಖ್ಯಾನಿಸು ggml_mutex_ಬೀಗ(m)          pthread_mutex_ಬೀಗ(m)
#ವ್ಯಾಖ್ಯಾನಿಸು ggml_mutex_unlock(m)        pthread_mutex_unlock(m)
#ವ್ಯಾಖ್ಯಾನಿಸು ggml_mutex_ಬೀಗ_shared(m)   pthread_mutex_ಬೀಗ(m)
#ವ್ಯಾಖ್ಯಾನಿಸು ggml_mutex_unlock_shared(m) pthread_mutex_unlock(m)

#ವ್ಯಾಖ್ಯಾನಿಸು ggml_ಬೀಗ_init(x)    ಬಳಸದ(x)
#ವ್ಯಾಖ್ಯಾನಿಸು ggml_ಬೀಗ_destroy(x) ಬಳಸದ(x)
#ಆದರೆ defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64) && !defined(_M_ARM64EC))
#ವ್ಯಾಖ್ಯಾನಿಸು ggml_ಬೀಗ_ಬೀಗ(x)    _mm_pause()
#ಇಲ್ಲದಿದ್ದರೆ
#ವ್ಯಾಖ್ಯಾನಿಸು ggml_ಬೀಗ_ಬೀಗ(x)    ಬಳಸದ(x)
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿಸು ggml_ಬೀಗ_unlock(x)  ಬಳಸದ(x)

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_ಬೀಗ_INITIALIZER 0
#ವ್ಯಾಖ್ಯಾನಿಸು ggml_cond_init(c)      pthread_cond_init(c, ಖಾಲಿ)
#ವ್ಯಾಖ್ಯಾನಿಸು ggml_cond_destroy(c)   pthread_cond_destroy(c)
#ವ್ಯಾಖ್ಯಾನಿಸು ggml_cond_wait(c, m)   pthread_cond_wait(c, m)
#ವ್ಯಾಖ್ಯಾನಿಸು ggml_cond_ಪ್ರಸಾರ(c) pthread_cond_ಪ್ರಸಾರ(c)

#ವ್ಯಾಖ್ಯಾನಿಸು ggml_thread_create pthread_create
#ವ್ಯಾಖ್ಯಾನಿಸು ggml_thread_join   pthread_join

#ಆದರೆ_ಅಂತ್ಯ

// Threadpool def
ರಚನೆ ggml_threadpool {
    ggml_mutex_t mutex;       // mutex for cond.var
    ggml_cond_t  cond;        // cond.var for waiting for new work

    ರಚನೆ ggml_cgraph * cgraph;
    ರಚನೆ ggml_cplan  * cplan;

    // synchronization primitives
    atomic_int n_ನಕ್ಷೆ;       // updated when there is work to be done (i.e each graph) holds graph and active thread counts.
    atomic_int GGML_CACHE_ALIGN n_barrier;
    atomic_int GGML_CACHE_ALIGN n_barrier_passed;
    atomic_int GGML_CACHE_ALIGN current_chunk; // currently processing chunk during Mat_Mul, shared between all the threads.

    // these are atomic as an annotation for thread-sanitizer
    atomic_bool stop;         // Used for stopping the threadpool altogether
    atomic_bool pause;        // Used for pausing the threadpool or individual threads
    atomic_int  abort;        // Used for aborting processing of a graph

    ರಚನೆ ggml_compute_state * workers;   // per thread state
    ಪೂರ್ಣಾಂಕ          n_ಎಳೆಗಳು;   // Number of threads in the pool
    int32_t      prio;        // Scheduling priority
    uint32_t     poll;        // Polling level (0 - no polling)

    ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ ec;
};

// Per-thread state
ರಚನೆ ggml_compute_state {
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_USE_OPENMP
    ggml_thread_t thrd;
    ಪೂರ್ಣಾಂಕ  ಕೊನೆಯ_ನಕ್ಷೆ;
    bool pending;
#ಆದರೆ_ಅಂತ್ಯ
    bool cpumask[GGML_MAX_N_ಎಳೆಗಳು];
    ರಚನೆ ggml_threadpool * threadpool;
    ಪೂರ್ಣಾಂಕ ith;
};

// Helpers for polling loops
#ಆದರೆ defined(__aarch64__) && ( defined(__clang__) || defined(__GNUC__) )
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_thread_cpu_relax(ಶೂನ್ಯ) {
    __asm__ ಅಸ್ಥಿರ("yield" ::: "memory");
}
#ಅಥವಾಆದರೆ defined(__x86_64__)
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_thread_cpu_relax(ಶೂನ್ಯ) {
    _mm_pause();
}
#ಅಥವಾಆದರೆ defined(__riscv)
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_thread_cpu_relax(ಶೂನ್ಯ) {
    #ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __riscv_zihintpause
        __asm__ __volatile__ ("pause");
    #ಇಲ್ಲದಿದ್ದರೆ
        /* Encoding of the pause instruction */
        __asm__ __volatile__ (".4byte 0x100000F");
    #ಆದರೆ_ಅಂತ್ಯ
}
#ಇಲ್ಲದಿದ್ದರೆ
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_thread_cpu_relax(ಶೂನ್ಯ) {;}
#ಆದರೆ_ಅಂತ್ಯ

//
// NUMA support
//

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_NUMA_MAX_NODES 8
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_NUMA_MAX_CPUS 512

ರಚನೆ ggml_numa_ಗಂಟು {
    uint32_t cpus[GGML_NUMA_MAX_CPUS]; // hardware threads on this node
    uint32_t n_cpus;
};

ರಚನೆ ggml_numa_nodes {
    ಎಣಿಕೆ ggml_numa_strategy numa_strategy;
    ರಚನೆ ggml_numa_ಗಂಟು nodes[GGML_NUMA_MAX_NODES];
    uint32_t n_nodes;
    uint32_t ಒಟ್ಟು_cpus; // hardware threads on system
    uint32_t current_ಗಂಟು; // node on which main process is execting
#ಆದರೆ defined(__gnu_linux__)
    cpu_ಗಣ_t cpuset; // cpuset from numactl
#ಇಲ್ಲದಿದ್ದರೆ
    uint32_t cpuset; // no NUMA support outside of Linux at this time. Use a portable datatype
#ಆದರೆ_ಅಂತ್ಯ
};

//
// ggml state
//

ರಚನೆ ggml_state {
    ರಚನೆ ggml_numa_nodes numa;
};

ಸ್ಥಿರ ರಚನೆ ggml_state g_state = {0};

ಶೂನ್ಯ ggml_barrier(ರಚನೆ ggml_threadpool * tp) {
    ಪೂರ್ಣಾಂಕ n_ಎಳೆಗಳು = atomic_ತುಂಬು_explicit(&tp->n_ನಕ್ಷೆ, memory_order_relaxed) & GGML_THREADPOOL_N_ಎಳೆಗಳು_ಮುಖವಾಡ;
    ಆದರೆ (n_ಎಳೆಗಳು == 1) {
        ಹಿಂತಿರುಗಿ;
    }

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_OPENMP
    #ಸೂಚನೆ omp barrier
#ಇಲ್ಲದಿದ್ದರೆ
    ಪೂರ್ಣಾಂಕ n_passed = atomic_ತುಂಬು_explicit(&tp->n_barrier_passed, memory_order_relaxed);

    // enter barrier (full seq-cst fence)
    ಪೂರ್ಣಾಂಕ n_barrier = atomic_fetch_ಕೂಡಿಸು_explicit(&tp->n_barrier, 1, memory_order_seq_cst);

    ಆದರೆ (n_barrier == (n_ಎಳೆಗಳು - 1)) {
        // last thread
        atomic_store_explicit(&tp->n_barrier, 0, memory_order_relaxed);

        // exit barrier (full seq-cst fence)
        atomic_fetch_ಕೂಡಿಸು_explicit(&tp->n_barrier_passed, 1, memory_order_seq_cst);
        ಹಿಂತಿರುಗಿ;
    }

    // wait for other threads
    ಆಗ (atomic_ತುಂಬು_explicit(&tp->n_barrier_passed, memory_order_relaxed) == n_passed) {
        ggml_thread_cpu_relax();
    }

    // exit barrier (full seq-cst fence)
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_TSAN_ENABLED
    atomic_fetch_ಕೂಡಿಸು_explicit(&tp->n_barrier_passed, 0, memory_order_seq_cst);
    #ಇಲ್ಲದಿದ್ದರೆ
    atomic_thread_fence(memory_order_seq_cst);
    #ಆದರೆ_ಅಂತ್ಯ
#ಆದರೆ_ಅಂತ್ಯ
}

ಶೂನ್ಯ ggml_threadpool_chunk_ಗಣ(ರಚನೆ ggml_threadpool * tp, ಪೂರ್ಣಾಂಕ ಮೌಲ್ಯ) {
    atomic_store_explicit(&tp->current_chunk, ಮೌಲ್ಯ, memory_order_relaxed);
}

ಪೂರ್ಣಾಂಕ ggml_threadpool_chunk_ಕೂಡಿಸು(ರಚನೆ ggml_threadpool * tp, ಪೂರ್ಣಾಂಕ ಮೌಲ್ಯ) {
    ಹಿಂತಿರುಗಿ atomic_fetch_ಕೂಡಿಸು_explicit(&tp->current_chunk, ಮೌಲ್ಯ, memory_order_relaxed);
}

#ಆದರೆ defined(__gnu_linux__)
ಸ್ಥಿರ cpu_ಗಣ_t ggml_get_numa_affinity(ಶೂನ್ಯ) {
    cpu_ಗಣ_t cpuset;
    pthread_t thread;
    thread = pthread_self();
    CPU_ZERO(&cpuset);
    pthread_getaffinity_np(thread, ಗಾತ್ರ(cpu_ಗಣ_t), &cpuset);
    ಹಿಂತಿರುಗಿ cpuset;
}
#ಇಲ್ಲದಿದ್ದರೆ
ಸ್ಥಿರ uint32_t ggml_get_numa_affinity(ಶೂನ್ಯ) {
    ಹಿಂತಿರುಗಿ 0; // no NUMA support
}
#ಆದರೆ_ಅಂತ್ಯ

ಶೂನ್ಯ ggml_numa_init(ಎಣಿಕೆ ggml_numa_strategy numa_flag) {
    ಆದರೆ (g_state.numa.n_nodes > 0) {
        fprintf(stderr, "ggml_numa_init: NUMA already initialized\n");

        ಹಿಂತಿರುಗಿ;
    }

#ಆದರೆ defined(__gnu_linux__)
    ರಚನೆ stat st;
    ಅಕ್ಷರ ಮಾರ್ಗ[256];
    ಪೂರ್ಣಾಂಕ rv;

    // set numa scheme
    g_state.numa.numa_strategy = numa_flag;

    GGML_PRINT_DEBUG("numa strategy %u\n",g_state.numa.numa_strategy);

    g_state.numa.cpuset = ggml_get_numa_affinity();

    // enumerate nodes
    ಆಗ (g_state.numa.n_nodes < GGML_NUMA_MAX_NODES) {
        rv = snprintf(ಮಾರ್ಗ, ಗಾತ್ರ(ಮಾರ್ಗ), "/sys/devices/system/node/node%u", g_state.numa.n_nodes);
        GGML_ASSERT(rv > 0 && (ಚಿಹ್ನೆರಹಿತ)rv < ಗಾತ್ರ(ಮಾರ್ಗ));
        ಆದರೆ (stat(ಮಾರ್ಗ, &st) != 0) { ನಿಲ್ಲಿಸು; }
        ++g_state.numa.n_nodes;
    }

    // enumerate CPUs
    ಆಗ (g_state.numa.ಒಟ್ಟು_cpus < GGML_NUMA_MAX_CPUS) {
        rv = snprintf(ಮಾರ್ಗ, ಗಾತ್ರ(ಮಾರ್ಗ), "/sys/devices/system/cpu/cpu%u", g_state.numa.ಒಟ್ಟು_cpus);
        GGML_ASSERT(rv > 0 && (ಚಿಹ್ನೆರಹಿತ)rv < ಗಾತ್ರ(ಮಾರ್ಗ));
        ಆದರೆ (stat(ಮಾರ್ಗ, &st) != 0) { ನಿಲ್ಲಿಸು; }
        ++g_state.numa.ಒಟ್ಟು_cpus;
    }

    GGML_PRINT_DEBUG("found %u numa nodes, %u CPUs\n", g_state.numa.n_nodes, g_state.numa.ಒಟ್ಟು_cpus);

    // figure out which node we're on
    uint current_cpu;
    ಪೂರ್ಣಾಂಕ getcpu_ret = 0;
#ಆದರೆ __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ > 33) || defined(__COSMOPOLITAN__)
    getcpu_ret = getcpu(&current_cpu, &g_state.numa.current_ಗಂಟು);
#ಇಲ್ಲದಿದ್ದರೆ
    // old glibc doesn't have a wrapper for this call. Fall back on direct syscall
#   ಆದರೆ !defined(SYS_getcpu) && defined(SYS_get_cpu)
#       define SYS_getcpu SYS_get_cpu // some older glibc versions use this name
#   endif
    getcpu_ret = syscall(SYS_getcpu, &current_cpu, &g_state.numa.current_ಗಂಟು);
#ಆದರೆ_ಅಂತ್ಯ

    ಆದರೆ (g_state.numa.n_nodes < 1 || g_state.numa.ಒಟ್ಟು_cpus < 1 || getcpu_ret != 0) {
        g_state.numa.n_nodes = 0;
        ಹಿಂತಿರುಗಿ;
    }

    GGML_PRINT_DEBUG("found our process on numa node %u, CPU %u\n", g_state.numa.current_ಗಂಟು, current_cpu);

    ಗಾಗಿ (uint32_t n = 0; n < g_state.numa.n_nodes; ++n) {
        ರಚನೆ ggml_numa_ಗಂಟು * ಗಂಟು = &g_state.numa.nodes[n];
        GGML_PRINT_DEBUG("CPUs on node %u:", n);
        ಗಂಟು->n_cpus = 0;
        ಗಾಗಿ (uint32_t c = 0; c < g_state.numa.ಒಟ್ಟು_cpus; ++c) {
            rv = snprintf(ಮಾರ್ಗ, ಗಾತ್ರ(ಮಾರ್ಗ), "/sys/devices/system/node/node%u/cpu%u", n, c);
            GGML_ASSERT(rv > 0 && (ಚಿಹ್ನೆರಹಿತ)rv < ಗಾತ್ರ(ಮಾರ್ಗ));
            ಆದರೆ (stat(ಮಾರ್ಗ, &st) == 0) {
                ಗಂಟು->cpus[ಗಂಟು->n_cpus++] = c;
                GGML_PRINT_DEBUG(" %u", c);
            }
        }
        GGML_PRINT_DEBUG("\n");
    }

    ಆದರೆ (ggml_is_numa()) {
        FILE *fptr = fopen("/proc/sys/kernel/numa_balancing", "r");
        ಆದರೆ (fptr != ಖಾಲಿ) {
            ಅಕ್ಷರ buf[42];
            ಆದರೆ (fgets(buf, ಗಾತ್ರ(buf), fptr) && ಅಕ್ಷರಸರಣಿ_ಹೋಲಿಕೆ(buf, "0\n", ಗಾತ್ರ(buf)) != 0) {
                GGML_LOG_WARN("/proc/sys/kernel/numa_balancing is enabled, this has been observed to impair performance\n");
            }
            fclose(fptr);
        }
    }
#ಇಲ್ಲದಿದ್ದರೆ
    ಬಳಸದ(numa_flag);
    // TODO
#ಆದರೆ_ಅಂತ್ಯ
}

bool ggml_is_numa(ಶೂನ್ಯ) {
    ಹಿಂತಿರುಗಿ g_state.numa.n_nodes > 1;
}

#ಆದರೆ defined(__ARM_ARCH)
#ಆದರೆ defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
#ಅನ್ತರ್ಭಾವ <arm_sve.h>
ಸ್ಥಿರ ಶೂನ್ಯ ggml_init_arm_arch_features(ಶೂನ್ಯ) {
    ggml_arm_arch_features.sve_cnt = svcntb();
}
#ಇಲ್ಲದಿದ್ದರೆ
ಸ್ಥಿರ ಶೂನ್ಯ ggml_init_arm_arch_features(ಶೂನ್ಯ) {}
#ಆದರೆ_ಅಂತ್ಯ
#ಆದರೆ_ಅಂತ್ಯ // __ARM_ARCH

#ಆದರೆ defined(__riscv) && defined(__riscv_v_intrinsic)
#ಅನ್ತರ್ಭಾವ <riscv_vector.h>
ಸ್ಥಿರ ಶೂನ್ಯ ggml_init_riscv_arch_features(ಶೂನ್ಯ) {
    ggml_riscv_arch_features.rvv_vlen = __riscv_vlenb();
}
#ಇಲ್ಲದಿದ್ದರೆ
ಸ್ಥಿರ ಶೂನ್ಯ ggml_init_riscv_arch_features(ಶೂನ್ಯ) {}
#ಆದರೆ_ಅಂತ್ಯ

ರಚನೆ ggml_tensor * ggml_new_i32(ರಚನೆ ggml_context * ctx, int32_t ಮೌಲ್ಯ) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

    ggml_ಗಣ_i32(ಫಲಿತಾಂಶ, ಮೌಲ್ಯ);

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_new_f32(ರಚನೆ ggml_context * ctx, ದಶಮಾಂಶ ಮೌಲ್ಯ) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    ggml_ಗಣ_f32(ಫಲಿತಾಂಶ, ಮೌಲ್ಯ);

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ರಚನೆ ggml_tensor * ggml_ಗಣ_i32 (ರಚನೆ ggml_tensor * tensor, int32_t ಮೌಲ್ಯ) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n     = ggml_nrows(tensor);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nc    = tensor->ne[0];
    ಸ್ಥಿರಾಂಕ size_t n1 = tensor->nb[1];

    ಅಕ್ಷರ * ಸ್ಥಿರಾಂಕ ದತ್ತಾಂಶ = tensor->ದತ್ತಾಂಶ;

    ಆಯ್ಕೆ (tensor->type) {
        ಸ್ಥಿತಿ GGML_TYPE_I8:
            {
                ಪ್ರತಿಜ್ಞಾ(tensor->nb[0] == ಗಾತ್ರ(int8_t));
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n; i++) {
                    ggml_vec_ಗಣ_i8(nc, (int8_t *)(ದತ್ತಾಂಶ + i*n1), ಮೌಲ್ಯ);
                }
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_I16:
            {
                ಪ್ರತಿಜ್ಞಾ(tensor->nb[0] == ಗಾತ್ರ(int16_t));
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n; i++) {
                    ggml_vec_ಗಣ_i16(nc, (int16_t *)(ದತ್ತಾಂಶ + i*n1), ಮೌಲ್ಯ);
                }
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_I32:
            {
                ಪ್ರತಿಜ್ಞಾ(tensor->nb[0] == ಗಾತ್ರ(int32_t));
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n; i++) {
                    ggml_vec_ಗಣ_i32(nc, (int32_t *)(ದತ್ತಾಂಶ + i*n1), ಮೌಲ್ಯ);
                }
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            {
                ಪ್ರತಿಜ್ಞಾ(tensor->nb[0] == ಗಾತ್ರ(ggml_fp16_t));
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n; i++) {
                    ggml_vec_ಗಣ_f16(nc, (ggml_fp16_t *)(ದತ್ತಾಂಶ + i*n1), GGML_CPU_FP32_ಗಮ್ಯ_FP16(ಮೌಲ್ಯ));
                }
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            {
                ಪ್ರತಿಜ್ಞಾ(tensor->nb[0] == ಗಾತ್ರ(ggml_fp16_t));
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n; i++) {
                    ggml_vec_ಗಣ_bf16(nc, (ggml_bf16_t *)(ದತ್ತಾಂಶ + i*n1), GGML_FP32_ಗಮ್ಯ_BF16(ಮೌಲ್ಯ));
                }
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ಪ್ರತಿಜ್ಞಾ(tensor->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ));
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n; i++) {
                    ggml_vec_ಗಣ_f32(nc, (ದಶಮಾಂಶ *)(ದತ್ತಾಂಶ + i*n1), ಮೌಲ್ಯ);
                }
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }

    ಹಿಂತಿರುಗಿ tensor;
}

ರಚನೆ ggml_tensor * ggml_ಗಣ_f32(ರಚನೆ ggml_tensor * tensor, ದಶಮಾಂಶ ಮೌಲ್ಯ) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n     = ggml_nrows(tensor);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nc    = tensor->ne[0];
    ಸ್ಥಿರಾಂಕ size_t n1 = tensor->nb[1];

    ಅಕ್ಷರ * ಸ್ಥಿರಾಂಕ ದತ್ತಾಂಶ = tensor->ದತ್ತಾಂಶ;

    ಆಯ್ಕೆ (tensor->type) {
        ಸ್ಥಿತಿ GGML_TYPE_I8:
            {
                ಪ್ರತಿಜ್ಞಾ(tensor->nb[0] == ಗಾತ್ರ(int8_t));
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n; i++) {
                    ggml_vec_ಗಣ_i8(nc, (int8_t *)(ದತ್ತಾಂಶ + i*n1), ಮೌಲ್ಯ);
                }
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_I16:
            {
                ಪ್ರತಿಜ್ಞಾ(tensor->nb[0] == ಗಾತ್ರ(int16_t));
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n; i++) {
                    ggml_vec_ಗಣ_i16(nc, (int16_t *)(ದತ್ತಾಂಶ + i*n1), ಮೌಲ್ಯ);
                }
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_I32:
            {
                ಪ್ರತಿಜ್ಞಾ(tensor->nb[0] == ಗಾತ್ರ(int32_t));
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n; i++) {
                    ggml_vec_ಗಣ_i32(nc, (int32_t *)(ದತ್ತಾಂಶ + i*n1), ಮೌಲ್ಯ);
                }
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            {
                ಪ್ರತಿಜ್ಞಾ(tensor->nb[0] == ಗಾತ್ರ(ggml_fp16_t));
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n; i++) {
                    ggml_vec_ಗಣ_f16(nc, (ggml_fp16_t *)(ದತ್ತಾಂಶ + i*n1), GGML_CPU_FP32_ಗಮ್ಯ_FP16(ಮೌಲ್ಯ));
                }
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            {
                ಪ್ರತಿಜ್ಞಾ(tensor->nb[0] == ಗಾತ್ರ(ggml_bf16_t));
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n; i++) {
                    ggml_vec_ಗಣ_bf16(nc, (ggml_bf16_t *)(ದತ್ತಾಂಶ + i*n1), GGML_FP32_ಗಮ್ಯ_BF16(ಮೌಲ್ಯ));
                }
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ಪ್ರತಿಜ್ಞಾ(tensor->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ));
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n; i++) {
                    ggml_vec_ಗಣ_f32(nc, (ದಶಮಾಂಶ *)(ದತ್ತಾಂಶ + i*n1), ಮೌಲ್ಯ);
                }
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }

    ಹಿಂತಿರುಗಿ tensor;
}

int32_t ggml_get_i32_1d(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor, ಪೂರ್ಣಾಂಕ i) {
    ಆದರೆ (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_ಸೂಚ್ಯಂಕ(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ಹಿಂತಿರುಗಿ ggml_get_i32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    ಆಯ್ಕೆ (tensor->type) {
        ಸ್ಥಿತಿ GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == ಗಾತ್ರ(int8_t));
                ಹಿಂತಿರುಗಿ ((int8_t *)(tensor->ದತ್ತಾಂಶ))[i];
            }
        ಸ್ಥಿತಿ GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == ಗಾತ್ರ(int16_t));
                ಹಿಂತಿರುಗಿ ((int16_t *)(tensor->ದತ್ತಾಂಶ))[i];
            }
        ಸ್ಥಿತಿ GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == ಗಾತ್ರ(int32_t));
                ಹಿಂತಿರುಗಿ ((int32_t *)(tensor->ದತ್ತಾಂಶ))[i];
            }
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == ಗಾತ್ರ(ggml_fp16_t));
                ಹಿಂತಿರುಗಿ GGML_CPU_FP16_ಗಮ್ಯ_FP32(((ggml_fp16_t *)(tensor->ದತ್ತಾಂಶ))[i]);
            }
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == ಗಾತ್ರ(ggml_bf16_t));
                ಹಿಂತಿರುಗಿ GGML_BF16_ಗಮ್ಯ_FP32(((ggml_bf16_t *)(tensor->ದತ್ತಾಂಶ))[i]);
            }
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ));
                ಹಿಂತಿರುಗಿ ((ದಶಮಾಂಶ *)(tensor->ದತ್ತಾಂಶ))[i];
            }
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ಶೂನ್ಯ ggml_ಗಣ_i32_1d(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor, ಪೂರ್ಣಾಂಕ i, int32_t ಮೌಲ್ಯ) {
    ಆದರೆ (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_ಸೂಚ್ಯಂಕ(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_ಗಣ_i32_nd(tensor, id[0], id[1], id[2], id[3], ಮೌಲ್ಯ);
        ಹಿಂತಿರುಗಿ;
    }
    ಆಯ್ಕೆ (tensor->type) {
        ಸ್ಥಿತಿ GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == ಗಾತ್ರ(int8_t));
                ((int8_t *)(tensor->ದತ್ತಾಂಶ))[i] = ಮೌಲ್ಯ;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == ಗಾತ್ರ(int16_t));
                ((int16_t *)(tensor->ದತ್ತಾಂಶ))[i] = ಮೌಲ್ಯ;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == ಗಾತ್ರ(int32_t));
                ((int32_t *)(tensor->ದತ್ತಾಂಶ))[i] = ಮೌಲ್ಯ;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == ಗಾತ್ರ(ggml_fp16_t));
                ((ggml_fp16_t *)(tensor->ದತ್ತಾಂಶ))[i] = GGML_CPU_FP32_ಗಮ್ಯ_FP16(ಮೌಲ್ಯ);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == ಗಾತ್ರ(ggml_bf16_t));
                ((ggml_bf16_t *)(tensor->ದತ್ತಾಂಶ))[i] = GGML_FP32_ಗಮ್ಯ_BF16(ಮೌಲ್ಯ);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ));
                ((ದಶಮಾಂಶ *)(tensor->ದತ್ತಾಂಶ))[i] = ಮೌಲ್ಯ;
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

int32_t ggml_get_i32_nd(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor, ಪೂರ್ಣಾಂಕ i0, ಪೂರ್ಣಾಂಕ i1, ಪೂರ್ಣಾಂಕ i2, ಪೂರ್ಣಾಂಕ i3) {
    ಶೂನ್ಯ * ದತ್ತಾಂಶ   = (ಅಕ್ಷರ *) tensor->ದತ್ತಾಂಶ + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    ಆಯ್ಕೆ (tensor->type) {
        ಸ್ಥಿತಿ GGML_TYPE_I8:
            ಹಿಂತಿರುಗಿ ((int8_t *) ದತ್ತಾಂಶ)[0];
        ಸ್ಥಿತಿ GGML_TYPE_I16:
            ಹಿಂತಿರುಗಿ ((int16_t *) ದತ್ತಾಂಶ)[0];
        ಸ್ಥಿತಿ GGML_TYPE_I32:
            ಹಿಂತಿರುಗಿ ((int32_t *) ದತ್ತಾಂಶ)[0];
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            ಹಿಂತಿರುಗಿ GGML_CPU_FP16_ಗಮ್ಯ_FP32(((ggml_fp16_t *) ದತ್ತಾಂಶ)[0]);
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            ಹಿಂತಿರುಗಿ GGML_BF16_ಗಮ್ಯ_FP32(((ggml_bf16_t *) ದತ್ತಾಂಶ)[0]);
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            ಹಿಂತಿರುಗಿ ((ದಶಮಾಂಶ *) ದತ್ತಾಂಶ)[0];
        ಪೂರ್ವನಿಯೋಜಿತ:
            GGML_ABORT("fatal error");
    }
}

ಶೂನ್ಯ ggml_ಗಣ_i32_nd(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor, ಪೂರ್ಣಾಂಕ i0, ಪೂರ್ಣಾಂಕ i1, ಪೂರ್ಣಾಂಕ i2, ಪೂರ್ಣಾಂಕ i3, int32_t ಮೌಲ್ಯ) {
    ಶೂನ್ಯ * ದತ್ತಾಂಶ   = (ಅಕ್ಷರ *) tensor->ದತ್ತಾಂಶ + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    ಆಯ್ಕೆ (tensor->type) {
        ಸ್ಥಿತಿ GGML_TYPE_I8:
            {
                ((int8_t *)(ದತ್ತಾಂಶ))[0] = ಮೌಲ್ಯ;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_I16:
            {
                ((int16_t *)(ದತ್ತಾಂಶ))[0] = ಮೌಲ್ಯ;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_I32:
            {
                ((int32_t *)(ದತ್ತಾಂಶ))[0] = ಮೌಲ್ಯ;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(ದತ್ತಾಂಶ))[0] = GGML_CPU_FP32_ಗಮ್ಯ_FP16(ಮೌಲ್ಯ);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(ದತ್ತಾಂಶ))[0] = GGML_FP32_ಗಮ್ಯ_BF16(ಮೌಲ್ಯ);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ((ದಶಮಾಂಶ *)(ದತ್ತಾಂಶ))[0] = ಮೌಲ್ಯ;
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ದಶಮಾಂಶ ggml_get_f32_1d(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor, ಪೂರ್ಣಾಂಕ i) {
    ಆದರೆ (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_ಸೂಚ್ಯಂಕ(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ಹಿಂತಿರುಗಿ ggml_get_f32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    ಆಯ್ಕೆ (tensor->type) {
        ಸ್ಥಿತಿ GGML_TYPE_I8:
            {
                ಹಿಂತಿರುಗಿ ((int8_t *)(tensor->ದತ್ತಾಂಶ))[i];
            }
        ಸ್ಥಿತಿ GGML_TYPE_I16:
            {
                ಹಿಂತಿರುಗಿ ((int16_t *)(tensor->ದತ್ತಾಂಶ))[i];
            }
        ಸ್ಥಿತಿ GGML_TYPE_I32:
            {
                ಹಿಂತಿರುಗಿ ((int32_t *)(tensor->ದತ್ತಾಂಶ))[i];
            }
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            {
                ಹಿಂತಿರುಗಿ GGML_CPU_FP16_ಗಮ್ಯ_FP32(((ggml_fp16_t *)(tensor->ದತ್ತಾಂಶ))[i]);
            }
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            {
                ಹಿಂತಿರುಗಿ GGML_BF16_ಗಮ್ಯ_FP32(((ggml_bf16_t *)(tensor->ದತ್ತಾಂಶ))[i]);
            }
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ಹಿಂತಿರುಗಿ ((ದಶಮಾಂಶ *)(tensor->ದತ್ತಾಂಶ))[i];
            }
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ಶೂನ್ಯ ggml_ಗಣ_f32_1d(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor, ಪೂರ್ಣಾಂಕ i, ದಶಮಾಂಶ ಮೌಲ್ಯ) {
    ಆದರೆ (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_ಸೂಚ್ಯಂಕ(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_ಗಣ_f32_nd(tensor, id[0], id[1], id[2], id[3], ಮೌಲ್ಯ);
        ಹಿಂತಿರುಗಿ;
    }
    ಆಯ್ಕೆ (tensor->type) {
        ಸ್ಥಿತಿ GGML_TYPE_I8:
            {
                ((int8_t *)(tensor->ದತ್ತಾಂಶ))[i] = ಮೌಲ್ಯ;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_I16:
            {
                ((int16_t *)(tensor->ದತ್ತಾಂಶ))[i] = ಮೌಲ್ಯ;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_I32:
            {
                ((int32_t *)(tensor->ದತ್ತಾಂಶ))[i] = ಮೌಲ್ಯ;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(tensor->ದತ್ತಾಂಶ))[i] = GGML_CPU_FP32_ಗಮ್ಯ_FP16(ಮೌಲ್ಯ);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(tensor->ದತ್ತಾಂಶ))[i] = GGML_FP32_ಗಮ್ಯ_BF16(ಮೌಲ್ಯ);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ((ದಶಮಾಂಶ *)(tensor->ದತ್ತಾಂಶ))[i] = ಮೌಲ್ಯ;
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ದಶಮಾಂಶ ggml_get_f32_nd(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor, ಪೂರ್ಣಾಂಕ i0, ಪೂರ್ಣಾಂಕ i1, ಪೂರ್ಣಾಂಕ i2, ಪೂರ್ಣಾಂಕ i3) {
    ಶೂನ್ಯ * ದತ್ತಾಂಶ   = (ಅಕ್ಷರ *) tensor->ದತ್ತಾಂಶ + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    ಆಯ್ಕೆ (tensor->type) {
        ಸ್ಥಿತಿ GGML_TYPE_I8:
            ಹಿಂತಿರುಗಿ ((int8_t *) ದತ್ತಾಂಶ)[0];
        ಸ್ಥಿತಿ GGML_TYPE_I16:
            ಹಿಂತಿರುಗಿ ((int16_t *) ದತ್ತಾಂಶ)[0];
        ಸ್ಥಿತಿ GGML_TYPE_I32:
            ಹಿಂತಿರುಗಿ ((int32_t *) ದತ್ತಾಂಶ)[0];
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            ಹಿಂತಿರುಗಿ GGML_CPU_FP16_ಗಮ್ಯ_FP32(((ggml_fp16_t *) ದತ್ತಾಂಶ)[0]);
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            ಹಿಂತಿರುಗಿ GGML_BF16_ಗಮ್ಯ_FP32(((ggml_bf16_t *) ದತ್ತಾಂಶ)[0]);
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            ಹಿಂತಿರುಗಿ ((ದಶಮಾಂಶ *) ದತ್ತಾಂಶ)[0];
        ಪೂರ್ವನಿಯೋಜಿತ:
            GGML_ABORT("fatal error");
    }
}

ಶೂನ್ಯ ggml_ಗಣ_f32_nd(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor, ಪೂರ್ಣಾಂಕ i0, ಪೂರ್ಣಾಂಕ i1, ಪೂರ್ಣಾಂಕ i2, ಪೂರ್ಣಾಂಕ i3, ದಶಮಾಂಶ ಮೌಲ್ಯ) {
    ಶೂನ್ಯ * ದತ್ತಾಂಶ   = (ಅಕ್ಷರ *) tensor->ದತ್ತಾಂಶ + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    ಆಯ್ಕೆ (tensor->type) {
        ಸ್ಥಿತಿ GGML_TYPE_I8:
            {
                ((int8_t *)(ದತ್ತಾಂಶ))[0] = ಮೌಲ್ಯ;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_I16:
            {
                ((int16_t *)(ದತ್ತಾಂಶ))[0] = ಮೌಲ್ಯ;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_I32:
            {
                ((int32_t *)(ದತ್ತಾಂಶ))[0] = ಮೌಲ್ಯ;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(ದತ್ತಾಂಶ))[0] = GGML_CPU_FP32_ಗಮ್ಯ_FP16(ಮೌಲ್ಯ);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(ದತ್ತಾಂಶ))[0] = GGML_FP32_ಗಮ್ಯ_BF16(ಮೌಲ್ಯ);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ((ದಶಮಾಂಶ *)(ದತ್ತಾಂಶ))[0] = ಮೌಲ್ಯ;
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

////////////////////////////////////////////////////////////////////////////////

// ggml_compute_forward_mul_mat

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_mul_mat_one_chunk(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params,
    ರಚನೆ ggml_tensor * dst,
    ಸ್ಥಿರಾಂಕ ಎಣಿಕೆ ggml_type type,
    ಸ್ಥಿರಾಂಕ int64_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_vec_dot,
    ಸ್ಥಿರಾಂಕ int64_t ir0_ಆರಂಭ,
    ಸ್ಥಿರಾಂಕ int64_t ir0_ಕೊನೆ,
    ಸ್ಥಿರಾಂಕ int64_t ir1_ಆರಂಭ,
    ಸ್ಥಿರಾಂಕ int64_t ir1_ಕೊನೆ) {

    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    ಸ್ಥಿರಾಂಕ bool src1_cont = ggml_is_contiguous(src1);

    ggml_vec_dot_t ಸ್ಥಿರಾಂಕ vec_dot      = type_traits_cpu[type].vec_dot;
    ಎಣಿಕೆ ggml_type ಸ್ಥಿರಾಂಕ vec_dot_type = type_traits_cpu[type].vec_dot_type;

    // broadcast factors
    ಸ್ಥಿರಾಂಕ int64_t r2 = ne12 / ne02;
    ಸ್ಥಿರಾಂಕ int64_t r3 = ne13 / ne03;

    //printf("ir0_start = %6lld, ir0_end = %6lld, ir1_start = %6lld, ir1_end = %6lld\n", ir0_start, ir0_end, ir1_start, ir1_end);

    // threads with no work simply yield (not sure if it helps)
    ಆದರೆ (ir0_ಆರಂಭ >= ir0_ಕೊನೆ || ir1_ಆರಂಭ >= ir1_ಕೊನೆ) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * wdata = (src1->type == vec_dot_type) ? src1->ದತ್ತಾಂಶ : params->wdata;
    ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(vec_dot_type, ne10);

    ಪ್ರತಿಜ್ಞಾ(ne12 % ne02 == 0);
    ಪ್ರತಿಜ್ಞಾ(ne13 % ne03 == 0);

    // block-tiling attempt
    ಸ್ಥಿರಾಂಕ int64_t blck_0 = 16;
    ಸ್ಥಿರಾಂಕ int64_t blck_1 = 16;

    ಸ್ಥಿರಾಂಕ size_t src1_ಲಂಬಸಾಲು_stride = src1_cont || src1->type != vec_dot_type ? ಅಡ್ಡಸಾಲು_ಗಾತ್ರ : nb11;

    // attempt to reduce false-sharing (does not seem to make a difference)
    // 16 * 2, accounting for mmla kernels
    ದಶಮಾಂಶ tmp[32];

    ಗಾಗಿ (int64_t iir1 = ir1_ಆರಂಭ; iir1 < ir1_ಕೊನೆ; iir1 += blck_1) {
        ಗಾಗಿ (int64_t iir0 = ir0_ಆರಂಭ; iir0 < ir0_ಕೊನೆ; iir0 += blck_0) {
            ಗಾಗಿ (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_ಕೊನೆ; ir1 += ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_vec_dot) {
                ಸ್ಥಿರಾಂಕ int64_t i13 = (ir1 / (ne12 * ne1));
                ಸ್ಥಿರಾಂಕ int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                ಸ್ಥಿರಾಂಕ int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                // broadcast src0 into src1
                ಸ್ಥಿರಾಂಕ int64_t i03 = i13 / r3;
                ಸ್ಥಿರಾಂಕ int64_t i02 = i12 / r2;

                ಸ್ಥಿರಾಂಕ int64_t i1 = i11;
                ಸ್ಥಿರಾಂಕ int64_t i2 = i12;
                ಸ್ಥಿರಾಂಕ int64_t i3 = i13;

                ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * src0_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ*)src0->ದತ್ತಾಂಶ + (0 + i02 * nb02 + i03 * nb03);

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * src1_ಲಂಬಸಾಲು = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ*)wdata +
                    (src1_cont || src1->type != vec_dot_type
                        ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ
                        : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                ದಶಮಾಂಶ * dst_ಲಂಬಸಾಲು = (ದಶಮಾಂಶ*)((ಅಕ್ಷರ*)dst->ದತ್ತಾಂಶ + (i1 * nb1 + i2 * nb2 + i3 * nb3));

                //for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                //    vec_dot(ne00, &dst_col[ir0], src0_row + ir0*nb01, src1_col);
                //}

                ಗಾಗಿ (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_ಕೊನೆ; ir0 += ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_vec_dot) {
                    vec_dot(ne00, &tmp[ir0 - iir0], (ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_vec_dot > 1 ? 16 : 0), src0_ಅಡ್ಡಸಾಲು + ir0 * nb01, (ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_vec_dot > 1 ? nb01 : 0), src1_ಲಂಬಸಾಲು, (ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_vec_dot > 1 ? src1_ಲಂಬಸಾಲು_stride : 0), ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_vec_dot);
                }

                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ cn = 0; cn < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_vec_dot; ++cn) {
                    ಸ್ಮರಣೆ_ನಕಲು(&dst_ಲಂಬಸಾಲು[iir0 + cn * nb1 / nb0], tmp + (cn * 16), (MIN(iir0 + blck_0, ir0_ಕೊನೆ) - iir0) * ಗಾತ್ರ(ದಶಮಾಂಶ));
                }
            }
        }
    }
}

ಶೂನ್ಯ ggml_compute_forward_mul_mat(
        ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params,
              ರಚನೆ ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1 = dst->src[1];

    ಸ್ಥಿರಾಂಕ int32_t hint = ggml_get_op_params_i32(dst, 1);
    ಆದರೆ (hint == GGML_HINT_SRC0_IS_HADAMARD && !params->use_ref) {
        ggml_compute_forward_fwht(params, dst);
        ಹಿಂತಿರುಗಿ;
    }

    // If tiled is supported, it will execute the full op here and we return
    ಆದರೆ (ggml_compute_forward_mul_mat_tiled(params, dst)) {
        ಹಿಂತಿರುಗಿ;
    }

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಎಣಿಕೆ ggml_type           ಸ್ಥಿರಾಂಕ vec_dot_type         = type_traits_cpu[src0->type].vec_dot_type;
    ggml_from_float_t        ಸ್ಥಿರಾಂಕ from_float           = type_traits_cpu[vec_dot_type].from_float;
    int64_t                  ಸ್ಥಿರಾಂಕ vec_dot_ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು     = type_traits_cpu[src0->type].nrows;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_ಗಾತ್ರ(src0->type));
    GGML_ASSERT(nb10 == ggml_type_ಗಾತ್ರ(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == ಗಾತ್ರ(ದಶಮಾಂಶ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    // TODO: extract to "extra_op"
#ಆದರೆ GGML_USE_LLAMAFILE
    // broadcast factors
    ಸ್ಥಿರಾಂಕ int64_t r2 = ne12 / ne02;
    ಸ್ಥಿರಾಂಕ int64_t r3 = ne13 / ne03;

    ಸ್ಥಿರಾಂಕ bool src1_cont = ggml_is_contiguous(src1);

    ಆದರೆ (!params->use_ref && src1_cont) {
        ಗಾಗಿ (int64_t i13 = 0; i13 < ne13; i13++)
            ಗಾಗಿ (int64_t i12 = 0; i12 < ne12; i12++)
                ಆದರೆ (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_ಗಾತ್ರ(src0->type),
                                     (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)src0->ದತ್ತಾಂಶ + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_ಗಾತ್ರ(src0->type),
                                     (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)src1->ದತ್ತಾಂಶ + i12*nb12 + i13*nb13,
                                     nb11/ggml_type_ಗಾತ್ರ(src1->type),
                                     (ಅಕ್ಷರ *)dst->ದತ್ತಾಂಶ + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_ಗಾತ್ರ(dst->type),
                                     src0->type,
                                     src1->type,
                                     dst->type))
                    ಹೋಗು UseGgmlGemm1;
        ಹಿಂತಿರುಗಿ;
    }
UseGgmlGemm1:;
#ಆದರೆ_ಅಂತ್ಯ

    ಆದರೆ (src1->type != vec_dot_type) {
        ಅಕ್ಷರ * wdata = params->wdata;

        ಸ್ಥಿರಾಂಕ size_t nbw0 = ggml_type_ಗಾತ್ರ(vec_dot_type);
        ಸ್ಥಿರಾಂಕ size_t nbw1 = ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(vec_dot_type, ne10);
        ಸ್ಥಿರಾಂಕ size_t nbw2 = nbw1*ne11;
        ಸ್ಥಿರಾಂಕ size_t nbw3 = nbw2*ne12;

        ಪ್ರತಿಜ್ಞಾ(params->wsize >= ne13*nbw3);
        // src1 is either packed from F32 into vec_dot_type, or widened from F16 or BF16 into the F32 work buffer
        ಸ್ಥಿರಾಂಕ bool widen = src1->type != GGML_TYPE_F32;

        GGML_ASSERT(!widen || vec_dot_type == GGML_TYPE_F32);
        GGML_ASSERT(!widen || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_BF16);

    #ಆದರೆ 0
        ಗಾಗಿ (int64_t i13 = 0; i13 < ne13; ++i13) {
            ಗಾಗಿ (int64_t i12 = 0; i12 < ne12; ++i12) {
                ಗಾಗಿ (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                    from_float((ದಶಮಾಂಶ *)((ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + i13*nb13 + i12*nb12 + i11*nb11),
                               (ಶೂನ್ಯ *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                                ne10);
                }
            }
        }
    #ಇಲ್ಲದಿದ್ದರೆ
        ಸ್ಥಿರಾಂಕ int64_t bs = ggml_blck_ಗಾತ್ರ(vec_dot_type);

        ಗಾಗಿ (int64_t i13 = 0; i13 < ne13; ++i13) {
            ಗಾಗಿ (int64_t i12 = 0; i12 < ne12; ++i12) {
                ಗಾಗಿ (int64_t i11 = 0; i11 < ne11; ++i11) {
                    int64_t ne10_ಖಂಡ_ಆರಂಭ = (ith * ne10/bs) / nth;
                    int64_t ne10_ಖಂಡ_ಕೊನೆ   = ((ith + 1) * ne10/bs) / nth;
                    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src1_ಅಡ್ಡಸಾಲು  = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + i13*nb13 + i12*nb12 + i11*nb11 + ne10_ಖಂಡ_ಆರಂಭ*bs*nb10;
                    ಶೂನ್ಯ       * wdata_ಅಡ್ಡಸಾಲು =                    wdata  + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_ಖಂಡ_ಆರಂಭ*nbw0;

                    ಸ್ಥಿರಾಂಕ int64_t ne10_ಖಂಡ_ಗಾತ್ರ = (ne10_ಖಂಡ_ಕೊನೆ - ne10_ಖಂಡ_ಆರಂಭ) * bs;

                    ಆದರೆ (src1->type == GGML_TYPE_F16) {
                        ggml_cpu_fp16_ಗಮ್ಯ_fp32((ಸ್ಥಿರಾಂಕ ggml_fp16_t *) src1_ಅಡ್ಡಸಾಲು, (ದಶಮಾಂಶ *) wdata_ಅಡ್ಡಸಾಲು, ne10_ಖಂಡ_ಗಾತ್ರ);
                    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src1->type == GGML_TYPE_BF16) {
                        ggml_cpu_bf16_ಗಮ್ಯ_fp32((ಸ್ಥಿರಾಂಕ ggml_bf16_t *) src1_ಅಡ್ಡಸಾಲು, (ದಶಮಾಂಶ *) wdata_ಅಡ್ಡಸಾಲು, ne10_ಖಂಡ_ಗಾತ್ರ);
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        from_float((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src1_ಅಡ್ಡಸಾಲು, wdata_ಅಡ್ಡಸಾಲು, ne10_ಖಂಡ_ಗಾತ್ರ);
                    }
                }
            }
        }
    #ಆದರೆ_ಅಂತ್ಯ
    }

    ಆದರೆ (ith == 0) {
        // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
        atomic_store_explicit(&params->threadpool->current_chunk, nth, memory_order_relaxed);
    }

    ggml_barrier(params->threadpool);

#ಆದರೆ GGML_USE_LLAMAFILE
    ಆದರೆ (!params->use_ref && src1->type != vec_dot_type) {
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ* wdata = (src1->type == vec_dot_type) ? src1->ದತ್ತಾಂಶ : params->wdata;
        ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(vec_dot_type, ne10);

        ಗಾಗಿ (int64_t i13 = 0; i13 < ne13; i13++)
            ಗಾಗಿ (int64_t i12 = 0; i12 < ne12; i12++)
                ಆದರೆ (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_ಗಾತ್ರ(src0->type),
                                     (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)src0->ದತ್ತಾಂಶ + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_ಗಾತ್ರ(src0->type),
                                     (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)wdata + (i12*ne11 + i13*ne12*ne11)*ಅಡ್ಡಸಾಲು_ಗಾತ್ರ,
                                     ಅಡ್ಡಸಾಲು_ಗಾತ್ರ/ggml_type_ಗಾತ್ರ(vec_dot_type),
                                     (ಅಕ್ಷರ *)dst->ದತ್ತಾಂಶ + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_ಗಾತ್ರ(dst->type),
                                     src0->type,
                                     vec_dot_type,
                                     dst->type))
                    ಹೋಗು UseGgmlGemm2;
        ಹಿಂತಿರುಗಿ;
    }
UseGgmlGemm2:;
#ಆದರೆ_ಅಂತ್ಯ

    // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
    ಸ್ಥಿರಾಂಕ int64_t nr0 = ne0;

    // This is the size of the rest of the dimensions of the result
    ಸ್ಥಿರಾಂಕ int64_t nr1 = ne1 * ne2 * ne3;

    // Now select a reasonable chunk size.
    ಪೂರ್ಣಾಂಕ chunk_ಗಾತ್ರ = 16;

    // We need to step up the size if it's small
    ಆದರೆ (nr0 == 1 || nr1 == 1) {
        chunk_ಗಾತ್ರ = 64;
    }

    // distribute the work across the inner or outer loop based on which one is larger
    // The number of chunks in the 0/1 dim.
    // CEIL(nr0/chunk_size)
    int64_t nchunk0 = (nr0 + chunk_ಗಾತ್ರ - 1) / chunk_ಗಾತ್ರ;
    int64_t nchunk1 = (nr1 + chunk_ಗಾತ್ರ - 1) / chunk_ಗಾತ್ರ;

    // If the chunking is poor for the number of threads on this setup, scrap the whole plan.  Re-chunk it by thread.
    //   Also, chunking by thread was measured to have perform better on NUMA systems.  See https://github.com/ggml-org/llama.cpp/pull/6915
    //   In theory, chunking should be just as useful on NUMA and non NUMA systems, but testing disagreed with that.
    ಆದರೆ (nchunk0 * nchunk1 < nth * 4 || ggml_is_numa()) {
        // distribute the thread work across the inner or outer loop based on which one is larger
        nchunk0 = nr0 > nr1 ? nth : 1; // parallelize by src0 rows
        nchunk1 = nr0 > nr1 ? 1 : nth; // parallelize by src1 rows
    }

    // The number of elements in each chunk
    ಸ್ಥಿರಾಂಕ int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    ಸ್ಥಿರಾಂಕ int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    // The first chunk comes from our thread_id, the rest will get auto-assigned.
    ಪೂರ್ಣಾಂಕ current_chunk = ith;

    ಆಗ (current_chunk < nchunk0 * nchunk1) {
        ಸ್ಥಿರಾಂಕ int64_t ith0 = current_chunk % nchunk0;
        ಸ್ಥಿರಾಂಕ int64_t ith1 = current_chunk / nchunk0;

        ಸ್ಥಿರಾಂಕ int64_t ir0_ಆರಂಭ = dr0 * ith0;
        ಸ್ಥಿರಾಂಕ int64_t ir0_ಕೊನೆ = MIN(ir0_ಆರಂಭ + dr0, nr0);

        ಸ್ಥಿರಾಂಕ int64_t ir1_ಆರಂಭ = dr1 * ith1;
        ಸ್ಥಿರಾಂಕ int64_t ir1_ಕೊನೆ = MIN(ir1_ಆರಂಭ + dr1, nr1);

        // dot kernels can handle 1 row and col at a time, but mmla kernels can process 2 rows and cols
        int64_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_vec_dot = vec_dot_ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು;

        // these checks are needed to avoid crossing dim1 boundaries
        // can be optimized, but the logic would become more complicated, so keeping it like this for simplicity
        ಆದರೆ ((nr0 % 2 != 0) || (ne11 % 2 != 0) || ((ir0_ಕೊನೆ - ir0_ಆರಂಭ) % 2 != 0) || ((ir1_ಕೊನೆ - ir1_ಆರಂಭ) % 2 != 0)) {
            ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_vec_dot = 1;
        }
        ggml_compute_forward_mul_mat_one_chunk(params, dst, src0->type, ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_vec_dot, ir0_ಆರಂಭ, ir0_ಕೊನೆ, ir1_ಆರಂಭ, ir1_ಕೊನೆ);

        ಆದರೆ (nth >= nchunk0 * nchunk1) {
            ನಿಲ್ಲಿಸು;
        }

        current_chunk = atomic_fetch_ಕೂಡಿಸು_explicit(&params->threadpool->current_chunk, 1, memory_order_relaxed);
    }
}

// ggml_compute_forward_mul_mat_id

#ವ್ಯಾಖ್ಯಾನಿಸು MMID_ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು(ಅಡ್ಡಸಾಲು_id, i1) ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು[(ಅಡ್ಡಸಾಲು_id)*ids->ne[0]*ids->ne[1] + (i1)]

ರಚನೆ mmid_ಅಡ್ಡಸಾಲು_mapping {
    int32_t i1;
    int32_t i2;
};

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_mul_mat_id_one_chunk(
    ರಚನೆ ggml_tensor * dst,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * ids,
    ಸ್ಥಿರಾಂಕ int64_t cur_a,
    ಸ್ಥಿರಾಂಕ int64_t ir0_ಆರಂಭ,
    ಸ್ಥಿರಾಂಕ int64_t ir0_ಕೊನೆ,
    ಸ್ಥಿರಾಂಕ int64_t ir1_ಆರಂಭ,
    ಸ್ಥಿರಾಂಕ int64_t ir1_ಕೊನೆ,
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * src0_cur,
    ಸ್ಥಿರಾಂಕ ರಚನೆ mmid_ಅಡ್ಡಸಾಲು_mapping * ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು,
    ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು_ಗಾತ್ರ,
    ಸ್ಥಿರಾಂಕ bool src1_cont,
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * wdata) {

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    ಸ್ಥಿರಾಂಕ ಎಣಿಕೆ ggml_type type = src0->type;

    ggml_vec_dot_t    ಸ್ಥಿರಾಂಕ vec_dot      = type_traits_cpu[type].vec_dot;
    ಎಣಿಕೆ ggml_type    ಸ್ಥಿರಾಂಕ vec_dot_type = type_traits_cpu[type].vec_dot_type;

    ಸ್ಥಿರಾಂಕ int64_t blck_0 = 16;
    ಸ್ಥಿರಾಂಕ int64_t blck_1 = 16;

    ದಶಮಾಂಶ tmp[16];

    ಗಾಗಿ (int64_t iir1 = ir1_ಆರಂಭ; iir1 < ir1_ಕೊನೆ; iir1 += blck_1) {
        ಗಾಗಿ (int64_t iir0 = ir0_ಆರಂಭ; iir0 < ir0_ಕೊನೆ; iir0 += blck_0) {
            ಗಾಗಿ (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_ಕೊನೆ; ++ir1) {
                ಸ್ಥಿರಾಂಕ int64_t _i12 = ir1; // logical row index for this expert

                ರಚನೆ mmid_ಅಡ್ಡಸಾಲು_mapping ಅಡ್ಡಸಾಲು_mapping = MMID_ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು(cur_a, _i12);
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ id       = ಅಡ್ಡಸಾಲು_mapping.i1; // selected expert index

                ಸ್ಥಿರಾಂಕ int64_t  i11 = id % ne11;
                ಸ್ಥಿರಾಂಕ int64_t  i12 = ಅಡ್ಡಸಾಲು_mapping.i2; // row index in src1

                ಸ್ಥಿರಾಂಕ int64_t  i1 = id;  // selected expert index
                ಸ್ಥಿರಾಂಕ int64_t  i2 = i12; //ಅಡ್ಡಸಾಲು

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * src1_ಲಂಬಸಾಲು = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) wdata +
                    (src1_cont || src1->type != vec_dot_type
                    ? (i11      + i12*ne11)*ಅಡ್ಡಸಾಲು_ಗಾತ್ರ
                    : (i11*nb11 + i12*nb12));

                ದಶಮಾಂಶ * dst_ಲಂಬಸಾಲು = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + (i1*nb1 + i2*nb2));

                ಗಾಗಿ (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_ಕೊನೆ; ++ir0) {
                    vec_dot(ne00, &tmp[ir0 - iir0], 0, src0_cur + ir0*nb01, 0, src1_ಲಂಬಸಾಲು, 0, 1);
                }

                ಸ್ಮರಣೆ_ನಕಲು(&dst_ಲಂಬಸಾಲು[iir0], tmp, (MIN(iir0 + blck_0, ir0_ಕೊನೆ) - iir0)*ಗಾತ್ರ(ದಶಮಾಂಶ));
            }
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ * incr_ptr_aligned(ಶೂನ್ಯ ** p, size_t ಗಾತ್ರ_2, size_t align) {

    ಶೂನ್ಯ * ptr = *p;
    ptr = (ಶೂನ್ಯ *) GGML_PAD((uintptr_t) ptr, align);
    *p = (ಶೂನ್ಯ *) ((ಅಕ್ಷರ *) ptr + ಗಾತ್ರ_2);
    ಹಿಂತಿರುಗಿ ptr;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_mul_mat_id(
        ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params,
              ರಚನೆ ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1 = dst->src[1];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * ids = dst->src[2];

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಎಣಿಕೆ ggml_type type = src0->type;

    ಸ್ಥಿರಾಂಕ bool src1_cont = ggml_is_contiguous(src1);

    ಎಣಿಕೆ ggml_type    ಸ್ಥಿರಾಂಕ vec_dot_type    = type_traits_cpu[type].vec_dot_type;
    ggml_from_float_t ಸ್ಥಿರಾಂಕ from_float      = type_traits_cpu[vec_dot_type].from_float;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_ಗಾತ್ರ(type));
    GGML_ASSERT(nb10 == ggml_type_ಗಾತ್ರ(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == ಗಾತ್ರ(ದಶಮಾಂಶ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // row groups
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ids = ids->ne[0]; // n_expert_used
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_as  = ne02;       // n_expert

    ಶೂನ್ಯ * wdata_cur = params->wdata;

    ಆದರೆ (src1->type != vec_dot_type) {
        incr_ptr_aligned(&wdata_cur, ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(vec_dot_type, ggml_nelements(src1)), ಗಾತ್ರ(int64_t));
    }

    int64_t * ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts = // [n_as]
        incr_ptr_aligned(&wdata_cur, n_as*ಗಾತ್ರ(int64_t), ಗಾತ್ರ(int64_t));

    ರಚನೆ mmid_ಅಡ್ಡಸಾಲು_mapping * ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು = // [n_as][ids->ne[0]*ids->ne[1]]
        incr_ptr_aligned(&wdata_cur, n_as*ids->ne[0]*ids->ne[1]*ಗಾತ್ರ(ರಚನೆ mmid_ಅಡ್ಡಸಾಲು_mapping), ಗಾತ್ರ(int64_t));

    ಅಕ್ಷರ (*atomic_current_chunk)[CACHE_ಸಾಲು_ಗಾತ್ರ] = // [n_as]
        incr_ptr_aligned(&wdata_cur, CACHE_ಸಾಲು_ಗಾತ್ರ * n_as, CACHE_ಸಾಲು_ಗಾತ್ರ);

    // Tiled matmul (see tiled.h); per-thread work buffers, 0 bytes when disabled. The
    // reservation is unconditional, the per expert eligibility is decided at dispatch time
    ಅಕ್ಷರ * tiled_scratch = incr_ptr_aligned(&wdata_cur, ggml_tiled_wdata_ಗಾತ್ರ(nth, dst), 64);

    GGML_ASSERT(params->wsize >= (size_t)((ಅಕ್ಷರ *) wdata_cur - (ಅಕ್ಷರ *) params->wdata));

    ಆದರೆ (src1->type != vec_dot_type) {
        ಅಕ್ಷರ * wdata = params->wdata;

        ಸ್ಥಿರಾಂಕ size_t nbw0 = ggml_type_ಗಾತ್ರ(vec_dot_type);
        ಸ್ಥಿರಾಂಕ size_t nbw1 = ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(vec_dot_type, ne10);
        ಸ್ಥಿರಾಂಕ size_t nbw2 = nbw1*ne11;
        ಸ್ಥಿರಾಂಕ size_t nbw3 = nbw2*ne12;

        ಪ್ರತಿಜ್ಞಾ(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

#ಆದರೆ 0
        ಗಾಗಿ (int64_t i13 = 0; i13 < ne13; ++i13) {
            ಗಾಗಿ (int64_t i12 = ith; i12 < ne12; i12 += nth) {
                ಗಾಗಿ (int64_t i11 = 0; i11 < ne11; ++i11) {
                    from_float((ದಶಮಾಂಶ *)((ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + i13*nb13 + i12*nb12 + i11*nb11),
                               (ಶೂನ್ಯ *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                               ne10);
                }
            }
        }
#ಇಲ್ಲದಿದ್ದರೆ
        ಗಾಗಿ (int64_t i13 = 0; i13 < ne13; ++i13) {
            ಗಾಗಿ (int64_t i12 = 0; i12 < ne12; ++i12) {
                ಗಾಗಿ (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = ggml_blck_ಗಾತ್ರ(vec_dot_type);
                    int64_t ne10_ಖಂಡ_ಆರಂಭ = (ith * ne10/bs) / nth;
                    int64_t ne10_ಖಂಡ_ಕೊನೆ   = ((ith + 1) * ne10/bs) / nth;
                    from_float((ದಶಮಾಂಶ *)((ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + i13*nb13 + i12*nb12 + i11*nb11 + ne10_ಖಂಡ_ಆರಂಭ*bs*nb10),
                               (ಶೂನ್ಯ *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_ಖಂಡ_ಆರಂಭ*nbw0),
                               (ne10_ಖಂಡ_ಕೊನೆ - ne10_ಖಂಡ_ಆರಂಭ) * bs);
                }
            }
        }
#ಆದರೆ_ಅಂತ್ಯ
    }

    ಆದರೆ (ith == 0) {
        // initialize matrix_row_counts
        memset(ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts, 0, n_as*ಗಾತ್ರ(int64_t));

        // group rows by src0 matrix
        ಗಾಗಿ (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ id = 0; id < n_ids; ++id) {
                ಸ್ಥಿರಾಂಕ int32_t i02 = *(ಸ್ಥಿರಾಂಕ int32_t *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) ids->ದತ್ತಾಂಶ + iid1*ids->nb[1] + id*ids->nb[0]);

                ಪ್ರತಿಜ್ಞಾ(i02 >= 0 && i02 < n_as);

                MMID_ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು(i02, ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts[i02]) = (ರಚನೆ mmid_ಅಡ್ಡಸಾಲು_mapping) {id, iid1};
                ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts[i02] += 1;
            }
        }
    }

    // reset current_chunk
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ cur_a = ith; cur_a < n_as; cur_a += nth) {
        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);
        *current_chunk_ctr = nth;
    }

    ggml_barrier(params->threadpool);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ cur_a = 0; cur_a < n_as; ++cur_a) {
        ಸ್ಥಿರಾಂಕ int64_t cne1 = ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts[cur_a];

        ಆದರೆ (cne1 == 0) {
            ಮುಂದುವರಿ;
        }

        // tiled takes over if profitable for this expert (see tiled.h)
        ಆದರೆ (ggml_compute_forward_mul_mat_id_tiled(params, dst, cur_a, cne1, (ಸ್ಥಿರಾಂಕ int32_t *) &MMID_ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು(cur_a, 0), tiled_scratch)) {
            ಮುಂದುವರಿ;
        }

        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * src0_cur = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + cur_a * nb02;
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * wdata = (src1->type == vec_dot_type) ? src1->ದತ್ತಾಂಶ : params->wdata;
        ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(vec_dot_type, ne10);

        ಸ್ಥಿರಾಂಕ int64_t nr0 = ne01;
        ಸ್ಥಿರಾಂಕ int64_t nr1 = cne1;

        ಪೂರ್ಣಾಂಕ chunk_ಗಾತ್ರ = 16;
        ಆದರೆ (nr0 == 1 || nr1 == 1) {
            chunk_ಗಾತ್ರ = 64;
        }

        // disable for NUMA
        ಸ್ಥಿರಾಂಕ bool disable_chunking = ggml_is_numa();

        int64_t nchunk0 = (nr0 + chunk_ಗಾತ್ರ - 1) / chunk_ಗಾತ್ರ;
        int64_t nchunk1 = (nr1 + chunk_ಗಾತ್ರ - 1) / chunk_ಗಾತ್ರ;

        ಆದರೆ (nchunk0 * nchunk1 < nth * 4 || disable_chunking) {
            nchunk0 = nr0 > nr1 ? nth : 1;
            nchunk1 = nr0 > nr1 ? 1 : nth;
        }

        ಸ್ಥಿರಾಂಕ int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        ಸ್ಥಿರಾಂಕ int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

        ಪೂರ್ಣಾಂಕ current_chunk = ith;

        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);

        ಆಗ (current_chunk < nchunk0 * nchunk1) {
            ಸ್ಥಿರಾಂಕ int64_t ith0 = current_chunk % nchunk0;
            ಸ್ಥಿರಾಂಕ int64_t ith1 = current_chunk / nchunk0;

            ಸ್ಥಿರಾಂಕ int64_t ir0_ಆರಂಭ = dr0 * ith0;
            ಸ್ಥಿರಾಂಕ int64_t ir0_ಕೊನೆ = MIN(ir0_ಆರಂಭ + dr0, nr0);

            ಸ್ಥಿರಾಂಕ int64_t ir1_ಆರಂಭ = dr1 * ith1;
            ಸ್ಥಿರಾಂಕ int64_t ir1_ಕೊನೆ = MIN(ir1_ಆರಂಭ + dr1, nr1);

            ggml_compute_forward_mul_mat_id_one_chunk(
                dst, src0, src1, ids, cur_a,
                ir0_ಆರಂಭ, ir0_ಕೊನೆ, ir1_ಆರಂಭ, ir1_ಕೊನೆ,
                src0_cur, ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, src1_cont, wdata
            );

            ಆದರೆ (nth >= nchunk0 * nchunk1) {
                ನಿಲ್ಲಿಸು;
            }

            current_chunk = atomic_fetch_ಕೂಡಿಸು_explicit(current_chunk_ctr, 1, memory_order_relaxed);
        }
    }
}

/////////////////////////////////

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward(ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * tensor) {
    GGML_ASSERT(params);

    ಆದರೆ (tensor->op == GGML_OP_NONE || ggml_is_ಖಾಲಿಗಳು(tensor)) {
        ಹಿಂತಿರುಗಿ;
    }

    // extra_buffer op?
    ಆದರೆ (ggml_cpu_extra_compute_forward(params, tensor)) {
        ಹಿಂತಿರುಗಿ;
    }

    ಆಯ್ಕೆ (tensor->op) {
        ಸ್ಥಿತಿ GGML_OP_DUP:
            {
                ggml_compute_forward_dup(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು:
            {
                ggml_compute_forward_ಕೂಡಿಸು(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು_ID:
            {
                ggml_compute_forward_ಕೂಡಿಸು_id(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ADD1:
            {
                ggml_compute_forward_add1(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ACC:
            {
                ggml_compute_forward_acc(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SUB:
            {
                ggml_compute_forward_sub(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_MUL:
            {
                ggml_compute_forward_mul(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_DIV:
            {
                ggml_compute_forward_div(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SQR:
            {
                ggml_compute_forward_sqr(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SQRT:
            {
                ggml_compute_forward_sqrt(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_LOG:
            {
                ggml_compute_forward_log(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SIN:
            {
                ggml_compute_forward_sin(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_COS:
            {
                ggml_compute_forward_cos(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಮೊತ್ತ:
            {
                ggml_compute_forward_ಮೊತ್ತ(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು:
            {
                ggml_compute_forward_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CUMSUM:
            {
                ggml_compute_forward_cumsum(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಸರಾಸರಿ:
            {
                ggml_compute_forward_ಸರಾಸರಿ(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ARGMAX:
            {
                ggml_compute_forward_argmax(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಎಣಿಕೆ_EQUAL:
            {
                ggml_compute_forward_ಎಣಿಕೆ_equal(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_REPEAT:
            {
                ggml_compute_forward_repeat(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_REPEAT_ಹಿಂಭಾಗ:
            {
                ggml_compute_forward_repeat_ಹಿಂಭಾಗ(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CONCAT:
            {
                ggml_compute_forward_concat(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SILU_ಹಿಂಭಾಗ:
            {
                ggml_compute_forward_silu_ಹಿಂಭಾಗ(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_NORM:
            {
                ggml_compute_forward_norm(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_RMS_NORM:
            {
                ggml_compute_forward_rms_norm(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_RMS_NORM_ಹಿಂಭಾಗ:
            {
                ggml_compute_forward_rms_norm_ಹಿಂಭಾಗ(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಗುಂಪು_NORM:
            {
                ggml_compute_forward_ಗುಂಪು_norm(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_L2_NORM:
            {
                ggml_compute_forward_l2_norm(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_MUL_MAT:
            {
                ggml_compute_forward_mul_mat(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_MUL_MAT_ID:
            {
                ggml_compute_forward_mul_mat_id(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಹೊರಹೋಗು_PROD:
            {
                ggml_compute_forward_ಹೊರಹೋಗು_prod(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SCALE:
            {
                ggml_compute_forward_scale(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಗಣ:
            {
                ggml_compute_forward_ಗಣ(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CPY:
            {
                ggml_compute_forward_cpy(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CONT:
            {
                ggml_compute_forward_cont(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು:
            {
                ggml_compute_forward_get_ಅಡ್ಡಸಾಲುಗಳು(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು_ಹಿಂಭಾಗ:
            {
                ggml_compute_forward_get_ಅಡ್ಡಸಾಲುಗಳು_ಹಿಂಭಾಗ(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು:
            {
                ggml_compute_forward_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_DIAG:
            {
                ggml_compute_forward_diag(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_DIAG_ಮುಖವಾಡ_INF:
            {
                ggml_compute_forward_diag_ಮುಖವಾಡ_inf(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_DIAG_ಮುಖವಾಡ_ZERO:
            {
                ggml_compute_forward_diag_ಮುಖವಾಡ_zero(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SOFT_MAX:
            {
                ggml_compute_forward_soft_max(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SOFT_MAX_ಹಿಂಭಾಗ:
            {
                ggml_compute_forward_soft_max_ext_ಹಿಂಭಾಗ(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ROPE:
            {
                ggml_compute_forward_rope(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ROPE_ಹಿಂಭಾಗ:
            {
                ggml_compute_forward_rope_ಹಿಂಭಾಗ(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CLAMP:
            {
                ggml_compute_forward_clamp(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_compute_forward_conv_transpose_1d(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_IM2_ಲಂಬಸಾಲು:
            {
                ggml_compute_forward_im2col(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_IM2_ಲಂಬಸಾಲು_ಹಿಂಭಾಗ:
            {
                ggml_compute_forward_im2col_ಹಿಂಭಾಗ_f32(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_IM2_ಲಂಬಸಾಲು_3_D:
            {
                ggml_compute_forward_im2col_3d(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_COL2IM_1D:
            {
                ggml_compute_forward_col2im_1d(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CONV_2D:
            {
                ggml_compute_forward_conv_2d(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CONV_3D:
            {
                ggml_compute_forward_conv_3d(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CONV_2D_DW:
            {
                ggml_compute_forward_conv_2d_dw(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CONV_TRANSPOSE_2D:
            {
                ggml_compute_forward_conv_transpose_2d(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_POOL_1D:
            {
                ggml_compute_forward_pool_1d(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_POOL_2D:
            {
                ggml_compute_forward_pool_2d(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_POOL_2_D_ಹಿಂಭಾಗ:
            {
                ggml_compute_forward_pool_2d_ಹಿಂಭಾಗ(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_UPSCALE:
            {
                ggml_compute_forward_upscale(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_PAD:
            {
                ggml_compute_forward_pad(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_PAD_REFLECT_1D:
            {
                ggml_compute_forward_pad_reflect_1d(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ROLL:
            {
                ggml_compute_forward_roll(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ARANGE:
            {
                ggml_compute_forward_arange(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_TIMESTEP_EMBEDDING:
            {
                ggml_compute_forward_timestep_embedding(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ARGSORT:
            {
                ggml_compute_forward_argsort(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ತುದಿ_K:
            {
                ggml_compute_forward_ತುದಿ_k(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_LEAKY_RELU:
            {
                ggml_compute_forward_leaky_relu(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_TRI:
            {
                ggml_compute_forward_tri(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_FILL:
            {
                ggml_compute_forward_fill(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_FLASH_ATTN_EXT:
            {
                ggml_compute_forward_flash_attn_ext(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_FLASH_ATTN_ಹಿಂಭಾಗ:
            {
                int32_t t = ggml_get_op_params_i32(tensor, 0);
                GGML_ASSERT(t == 0 || t == 1);
                bool masked = t != 0;
                ggml_compute_forward_flash_attn_ಹಿಂಭಾಗ(params, masked, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SSM_CONV:
            {
                ggml_compute_forward_ssm_conv(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SSM_ಪರಿಶೀಲನೆ:
            {
                ggml_compute_forward_ssm_ಪರಿಶೀಲನೆ(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_WIN_PART:
            {
                ggml_compute_forward_win_part(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_WIN_UNPART:
            {
                ggml_compute_forward_win_unpart(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_UNARY:
            {
                ggml_compute_forward_unary(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_GLU:
            {
                ggml_compute_forward_glu(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_GET_REL_POS:
            {
                ggml_compute_forward_get_rel_pos(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು_REL_POS:
            {
                ggml_compute_forward_ಕೂಡಿಸು_rel_pos(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_RWKV_WKV6:
            {
                ggml_compute_forward_rwkv_wkv6(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_GATED_ರೇಖೀಯ_ATTN:
            {
                ggml_compute_forward_gla(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_RWKV_WKV7:
            {
                ggml_compute_forward_rwkv_wkv7(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SOLVE_TRI:
            {
                ggml_compute_forward_solve_tri(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_GATED_DELTA_NET:
            {
                ggml_compute_forward_gated_delta_net(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_LIGHTNING_INDEXER:
            {
                ggml_compute_forward_lightning_indexer(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_DSV4_HC_COMB:
            {
                ggml_compute_forward_dsv4_hc_comb(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_DSV4_HC_PRE:
            {
                ggml_compute_forward_dsv4_hc_pre(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_DSV4_HC_POST:
            {
                ggml_compute_forward_dsv4_hc_post(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ತುದಿ_K_ಮುಖವಾಡ:
            {
                ggml_compute_forward_ತುದಿ_k_ಮುಖವಾಡ(params, tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_MAP_CUSTOM1:
            {
                ggml_compute_forward_map_custom1(params, tensor);
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_MAP_CUSTOM2:
            {
                ggml_compute_forward_map_custom2(params, tensor);
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_MAP_CUSTOM3:
            {
                ggml_compute_forward_map_custom3(params, tensor);
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CUSTOM:
            {
                ggml_compute_forward_custom(params, tensor);
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CROSS_ENTROPY_LOSS:
            {
                ggml_compute_forward_cross_entropy_loss(params, tensor);
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CROSS_ENTROPY_LOSS_ಹಿಂಭಾಗ:
            {
                ggml_compute_forward_cross_entropy_loss_ಹಿಂಭಾಗ(params, tensor);
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_OPT_ಹೆಜ್ಜೆ_ADAMW:
            {
                ggml_compute_forward_opt_ಹೆಜ್ಜೆ_adamw(params, tensor);
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_OPT_ಹೆಜ್ಜೆ_SGD:
            {
                ggml_compute_forward_opt_ಹೆಜ್ಜೆ_sgd(params, tensor);
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_NONE:
            {
                // nop
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_RESHAPE:
            {
                // nop
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಕ್ರಮಬದಲಾವಣೆ:
            {
                // nop
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_VIEW:
            {
                // nop
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_TRANSPOSE:
            {
                // nop
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಎಣಿಕೆ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// Android's libc implementation "bionic" does not support setting affinity
#ಆದರೆ defined(__gnu_linux__)
ಸ್ಥಿರ ಶೂನ್ಯ ಗಣ_numa_thread_affinity(ಪೂರ್ಣಾಂಕ thread_n) {
    ಆದರೆ (!ggml_is_numa()) {
        ಹಿಂತಿರುಗಿ;
    }

    ಪೂರ್ಣಾಂಕ ಗಂಟು_ಅಂಕೆ;
    ಪೂರ್ಣಾಂಕ rv;
    size_t setsize = CPU_ALLOC_ಗಾತ್ರ(g_state.numa.ಒಟ್ಟು_cpus);

    ಆಯ್ಕೆ(g_state.numa.numa_strategy) {
        ಸ್ಥಿತಿ GGML_NUMA_STRATEGY_DISTRIBUTE:
            // run thread on node_num thread_n / (threads per node)
            ಗಂಟು_ಅಂಕೆ = thread_n % g_state.numa.n_nodes;
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_NUMA_STRATEGY_ISOLATE:
            // run thread on current_node
            ಗಂಟು_ಅಂಕೆ = g_state.numa.current_ಗಂಟು;
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_NUMA_STRATEGY_NUMACTL:
            // use the cpuset that numactl gave us
            rv = pthread_setaffinity_np(pthread_self(), setsize, &g_state.numa.cpuset);
            ಆದರೆ (rv) {
                fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n",strerror(rv));
            }
            ಹಿಂತಿರುಗಿ;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ;
    }

    ರಚನೆ ggml_numa_ಗಂಟು * ಗಂಟು = &g_state.numa.nodes[ಗಂಟು_ಅಂಕೆ];

    cpu_ಗಣ_t * cpus = CPU_ALLOC(g_state.numa.ಒಟ್ಟು_cpus);
    CPU_ZERO_S(setsize, cpus);
    ಗಾಗಿ (size_t i = 0; i < ಗಂಟು->n_cpus; ++i) {
        CPU_ಗಣ_S(ಗಂಟು->cpus[i], setsize, cpus);
    }

    rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    ಆದರೆ (rv) {
            fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}

ಸ್ಥಿರ ಶೂನ್ಯ clear_numa_thread_affinity(ಶೂನ್ಯ) {
    ಆದರೆ (!ggml_is_numa()) {
        ಹಿಂತಿರುಗಿ;
    }

    size_t setsize = CPU_ALLOC_ಗಾತ್ರ(g_state.numa.ಒಟ್ಟು_cpus);

    cpu_ಗಣ_t * cpus = CPU_ALLOC(g_state.numa.ಒಟ್ಟು_cpus);
    CPU_ZERO_S(setsize, cpus);
    ಗಾಗಿ (ಚಿಹ್ನೆರಹಿತ i = 0; i < g_state.numa.ಒಟ್ಟು_cpus; ++i) {
        CPU_ಗಣ_S(i, setsize, cpus);
    }

    ಪೂರ್ಣಾಂಕ rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    ಆದರೆ (rv) {
        fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}
#ಇಲ್ಲದಿದ್ದರೆ
// TODO: Windows etc.
// (the linux implementation may also work on BSD, someone should test)
ಸ್ಥಿರ ಶೂನ್ಯ ಗಣ_numa_thread_affinity(ಪೂರ್ಣಾಂಕ thread_n) { ಬಳಸದ(thread_n);  }
ಸ್ಥಿರ ಶೂನ್ಯ clear_numa_thread_affinity(ಶೂನ್ಯ) {}
#ಆದರೆ_ಅಂತ್ಯ

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ ggml_get_n_tasks(ರಚನೆ ggml_tensor * ಗಂಟು, ಪೂರ್ಣಾಂಕ n_ಎಳೆಗಳು) {
    ಪೂರ್ಣಾಂಕ n_tasks = 0;

    ಆದರೆ (ggml_is_ಖಾಲಿಗಳು(ಗಂಟು)) {
        // no need to multi-thread a no-op
        n_tasks = 1;
        ಹಿಂತಿರುಗಿ n_tasks;
    }

    ಆಯ್ಕೆ (ಗಂಟು->op) {
        ಸ್ಥಿತಿ GGML_OP_CPY:
        ಸ್ಥಿತಿ GGML_OP_DUP:
        ಸ್ಥಿತಿ GGML_OP_CONT:
        ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು:
        ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು_ID:
        ಸ್ಥಿತಿ GGML_OP_ADD1:
        ಸ್ಥಿತಿ GGML_OP_ACC:
        ಸ್ಥಿತಿ GGML_OP_CUMSUM:
        ಸ್ಥಿತಿ GGML_OP_TRI:
        ಸ್ಥಿತಿ GGML_OP_FILL:
            {
                n_tasks = n_ಎಳೆಗಳು;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SUB:
        ಸ್ಥಿತಿ GGML_OP_SQR:
        ಸ್ಥಿತಿ GGML_OP_SQRT:
        ಸ್ಥಿತಿ GGML_OP_LOG:
        ಸ್ಥಿತಿ GGML_OP_SIN:
        ಸ್ಥಿತಿ GGML_OP_COS:
        ಸ್ಥಿತಿ GGML_OP_ಮೊತ್ತ:
        ಸ್ಥಿತಿ GGML_OP_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು:
        ಸ್ಥಿತಿ GGML_OP_ಸರಾಸರಿ:
        ಸ್ಥಿತಿ GGML_OP_ARGMAX:
            {
                n_tasks = 1;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಎಣಿಕೆ_EQUAL:
        ಸ್ಥಿತಿ GGML_OP_SOLVE_TRI:
        ಸ್ಥಿತಿ GGML_OP_GATED_DELTA_NET:
        ಸ್ಥಿತಿ GGML_OP_DSV4_HC_COMB:
        ಸ್ಥಿತಿ GGML_OP_DSV4_HC_PRE:
        ಸ್ಥಿತಿ GGML_OP_DSV4_HC_POST:
        ಸ್ಥಿತಿ GGML_OP_ತುದಿ_K_ಮುಖವಾಡ:
            {
                n_tasks = n_ಎಳೆಗಳು;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_REPEAT:
        ಸ್ಥಿತಿ GGML_OP_REPEAT_ಹಿಂಭಾಗ:
        ಸ್ಥಿತಿ GGML_OP_LEAKY_RELU:
            {
                n_tasks = 1;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_UNARY:
            ಆಯ್ಕೆ (ggml_get_unary_op(ಗಂಟು)) {
                ಸ್ಥಿತಿ GGML_UNARY_OP_ABS:
                ಸ್ಥಿತಿ GGML_UNARY_OP_SGN:
                ಸ್ಥಿತಿ GGML_UNARY_OP_NEG:
                ಸ್ಥಿತಿ GGML_UNARY_OP_ಹೆಜ್ಜೆ:
                ಸ್ಥಿತಿ GGML_UNARY_OP_TANH:
                ಸ್ಥಿತಿ GGML_UNARY_OP_ELU:
                ಸ್ಥಿತಿ GGML_UNARY_OP_RELU:
                ಸ್ಥಿತಿ GGML_UNARY_OP_SIGMOID:
                ಸ್ಥಿತಿ GGML_UNARY_OP_HARDSWISH:
                ಸ್ಥಿತಿ GGML_UNARY_OP_HARDSIGMOID:
                ಸ್ಥಿತಿ GGML_UNARY_OP_EXP:
                ಸ್ಥಿತಿ GGML_UNARY_OP_SOFTPLUS:
                ಸ್ಥಿತಿ GGML_UNARY_OP_EXPM1:
                ಸ್ಥಿತಿ GGML_UNARY_OP_FLOOR:
                ಸ್ಥಿತಿ GGML_UNARY_OP_CEIL:
                ಸ್ಥಿತಿ GGML_UNARY_OP_ಸುತ್ತು:
                ಸ್ಥಿತಿ GGML_UNARY_OP_TRUNC:
                    {
                        n_tasks = 1;
                    } ನಿಲ್ಲಿಸು;

                ಸ್ಥಿತಿ GGML_UNARY_OP_GELU:
                ಸ್ಥಿತಿ GGML_UNARY_OP_GELU_ERF:
                ಸ್ಥಿತಿ GGML_UNARY_OP_GELU_ಶೀಘ್ರ:
                ಸ್ಥಿತಿ GGML_UNARY_OP_SILU:
                ಸ್ಥಿತಿ GGML_UNARY_OP_XIELU:
                    {
                        n_tasks = n_ಎಳೆಗಳು;
                    } ನಿಲ್ಲಿಸು;
                ಪೂರ್ವನಿಯೋಜಿತ:
                    GGML_ABORT("fatal error");
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_GLU:
            ಆಯ್ಕೆ (ggml_get_glu_op(ಗಂಟು)) {
                ಸ್ಥಿತಿ GGML_GLU_OP_REGLU:
                ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU:
                ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU:
                ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU_OAI:
                ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU_ERF:
                ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU_ಶೀಘ್ರ:
                ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU_CLAMP:
                    {
                        n_tasks = n_ಎಳೆಗಳು;
                    } ನಿಲ್ಲಿಸು;
                ಪೂರ್ವನಿಯೋಜಿತ:
                    GGML_ABORT("fatal error");
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SILU_ಹಿಂಭಾಗ:
        ಸ್ಥಿತಿ GGML_OP_MUL:
        ಸ್ಥಿತಿ GGML_OP_DIV:
        ಸ್ಥಿತಿ GGML_OP_NORM:
        ಸ್ಥಿತಿ GGML_OP_RMS_NORM:
        ಸ್ಥಿತಿ GGML_OP_RMS_NORM_ಹಿಂಭಾಗ:
        ಸ್ಥಿತಿ GGML_OP_L2_NORM:
        ಸ್ಥಿತಿ GGML_OP_ಗುಂಪು_NORM:
        ಸ್ಥಿತಿ GGML_OP_CONCAT:
        ಸ್ಥಿತಿ GGML_OP_MUL_MAT:
        ಸ್ಥಿತಿ GGML_OP_MUL_MAT_ID:
        ಸ್ಥಿತಿ GGML_OP_ಹೊರಹೋಗು_PROD:
        ಸ್ಥಿತಿ GGML_OP_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು:
            {
                n_tasks = n_ಎಳೆಗಳು;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು:
            {
                // FIXME: get_rows can use additional threads, but the cost of launching additional threads
                // decreases performance with GPU offloading
                //n_tasks = n_threads;
                n_tasks = 1;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SCALE:
        ಸ್ಥಿತಿ GGML_OP_ಗಣ:
        ಸ್ಥಿತಿ GGML_OP_RESHAPE:
        ಸ್ಥಿತಿ GGML_OP_VIEW:
        ಸ್ಥಿತಿ GGML_OP_ಕ್ರಮಬದಲಾವಣೆ:
        ಸ್ಥಿತಿ GGML_OP_TRANSPOSE:
        ಸ್ಥಿತಿ GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು_ಹಿಂಭಾಗ:
        ಸ್ಥಿತಿ GGML_OP_DIAG:
            {
                n_tasks = 1;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_DIAG_ಮುಖವಾಡ_ZERO:
        ಸ್ಥಿತಿ GGML_OP_DIAG_ಮುಖವಾಡ_INF:
        ಸ್ಥಿತಿ GGML_OP_SOFT_MAX_ಹಿಂಭಾಗ:
        ಸ್ಥಿತಿ GGML_OP_ROPE:
        ಸ್ಥಿತಿ GGML_OP_ROPE_ಹಿಂಭಾಗ:
        ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು_REL_POS:
            {
                n_tasks = n_ಎಳೆಗಳು;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CLAMP:
            {
                n_tasks = 1; //TODO
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SOFT_MAX:
            {
                n_tasks = MIN(n_ಎಳೆಗಳು, ggml_nrows(ಗಂಟು->src[0]));
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_IM2_ಲಂಬಸಾಲು:
        ಸ್ಥಿತಿ GGML_OP_IM2_ಲಂಬಸಾಲು_ಹಿಂಭಾಗ:
        ಸ್ಥಿತಿ GGML_OP_IM2_ಲಂಬಸಾಲು_3_D:
        ಸ್ಥಿತಿ GGML_OP_CONV_2D:
        ಸ್ಥಿತಿ GGML_OP_CONV_3D:
        ಸ್ಥಿತಿ GGML_OP_CONV_2D_DW:
        ಸ್ಥಿತಿ GGML_OP_COL2IM_1D:
        ಸ್ಥಿತಿ GGML_OP_CONV_TRANSPOSE_1D:
        ಸ್ಥಿತಿ GGML_OP_CONV_TRANSPOSE_2D:
            {
                n_tasks = n_ಎಳೆಗಳು;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_POOL_1D:
        ಸ್ಥಿತಿ GGML_OP_POOL_2D:
        ಸ್ಥಿತಿ GGML_OP_POOL_2_D_ಹಿಂಭಾಗ:
            {
                n_tasks = 1;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_UPSCALE:
        ಸ್ಥಿತಿ GGML_OP_PAD:
        ಸ್ಥಿತಿ GGML_OP_PAD_REFLECT_1D:
        ಸ್ಥಿತಿ GGML_OP_ROLL:
        ಸ್ಥಿತಿ GGML_OP_ARANGE:
        ಸ್ಥಿತಿ GGML_OP_TIMESTEP_EMBEDDING:
        ಸ್ಥಿತಿ GGML_OP_ARGSORT:
        ಸ್ಥಿತಿ GGML_OP_ತುದಿ_K:
        ಸ್ಥಿತಿ GGML_OP_FLASH_ATTN_EXT:
        ಸ್ಥಿತಿ GGML_OP_FLASH_ATTN_ಹಿಂಭಾಗ:
        ಸ್ಥಿತಿ GGML_OP_SSM_CONV:
        ಸ್ಥಿತಿ GGML_OP_SSM_ಪರಿಶೀಲನೆ:
        ಸ್ಥಿತಿ GGML_OP_LIGHTNING_INDEXER:
            {
                n_tasks = n_ಎಳೆಗಳು;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_RWKV_WKV6:
        ಸ್ಥಿತಿ GGML_OP_GATED_ರೇಖೀಯ_ATTN:
        ಸ್ಥಿತಿ GGML_OP_RWKV_WKV7:
            {
                ಸ್ಥಿರಾಂಕ int64_t n_heads = ಗಂಟು->src[1]->ne[1];
                n_tasks = MIN(n_ಎಳೆಗಳು, n_heads);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_WIN_PART:
        ಸ್ಥಿತಿ GGML_OP_WIN_UNPART:
        ಸ್ಥಿತಿ GGML_OP_GET_REL_POS:
            {
                n_tasks = 1;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_MAP_CUSTOM1:
            {
                ರಚನೆ ggml_map_custom1_op_params p;
                ಸ್ಮರಣೆ_ನಕಲು(&p, ಗಂಟು->op_params, ಗಾತ್ರ(p));
                ಆದರೆ (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_ಎಳೆಗಳು;
                } ಇಲ್ಲದಿದ್ದರೆ {
                    n_tasks = MIN(p.n_tasks, n_ಎಳೆಗಳು);
                }
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_MAP_CUSTOM2:
            {
                ರಚನೆ ggml_map_custom2_op_params p;
                ಸ್ಮರಣೆ_ನಕಲು(&p, ಗಂಟು->op_params, ಗಾತ್ರ(p));
                ಆದರೆ (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_ಎಳೆಗಳು;
                } ಇಲ್ಲದಿದ್ದರೆ {
                    n_tasks = MIN(p.n_tasks, n_ಎಳೆಗಳು);
                }
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_MAP_CUSTOM3:
            {
                ರಚನೆ ggml_map_custom3_op_params p;
                ಸ್ಮರಣೆ_ನಕಲು(&p, ಗಂಟು->op_params, ಗಾತ್ರ(p));
                ಆದರೆ (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_ಎಳೆಗಳು;
                } ಇಲ್ಲದಿದ್ದರೆ {
                    n_tasks = MIN(p.n_tasks, n_ಎಳೆಗಳು);
                }
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CUSTOM:
            {
                ರಚನೆ ggml_custom_op_params p;
                ಸ್ಮರಣೆ_ನಕಲು(&p, ಗಂಟು->op_params, ಗಾತ್ರ(p));
                ಆದರೆ (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_ಎಳೆಗಳು;
                } ಇಲ್ಲದಿದ್ದರೆ {
                    n_tasks = MIN(p.n_tasks, n_ಎಳೆಗಳು);
                }
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CROSS_ENTROPY_LOSS:
        ಸ್ಥಿತಿ GGML_OP_CROSS_ENTROPY_LOSS_ಹಿಂಭಾಗ:
        ಸ್ಥಿತಿ GGML_OP_OPT_ಹೆಜ್ಜೆ_ADAMW:
        ಸ್ಥಿತಿ GGML_OP_OPT_ಹೆಜ್ಜೆ_SGD:
            {
                n_tasks = n_ಎಳೆಗಳು;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_NONE:
            {
                n_tasks = 1;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಎಣಿಕೆ:
            {
                GGML_ABORT("fatal error");
            }
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                fprintf(stderr, "%s: op not implemented: ", __func__);
                ಆದರೆ (ಗಂಟು->op < GGML_OP_ಎಣಿಕೆ) {
                    fprintf(stderr, "%s\n", ggml_op_name(ಗಂಟು->op));
                } ಇಲ್ಲದಿದ್ದರೆ {
                    fprintf(stderr, "%d\n", ಗಂಟು->op);
                }
                GGML_ABORT("fatal error");
            }
    }

    ಪ್ರತಿಜ್ಞಾ(n_tasks > 0);

    ಹಿಂತಿರುಗಿ n_tasks;
}

ಸ್ಥಿರ thread_ret_t ggml_ನಕ್ಷೆ_compute_secondary_thread(ಶೂನ್ಯ* ದತ್ತಾಂಶ);

#ಆದರೆ defined(_WIN32)
#ಅನ್ತರ್ಭಾವ "windows.h"

// TODO: support > 64 CPUs
ಸ್ಥಿರ bool ggml_thread_apply_affinity(bool * ಮುಖವಾಡ) {
    HANDLE    h = GetCurrentThread();
    uint64_t  bitmask = 0ULL;

    ಪ್ರತಿಜ್ಞಾ(GGML_MAX_N_ಎಳೆಗಳು >= 64);

    ಗಾಗಿ (int32_t i = 0; i < 8; i++) {
        int32_t idx = i * 8;
        uint8_t val = 0;
        val |= ಮುಖವಾಡ[idx + 0] << 0;
        val |= ಮುಖವಾಡ[idx + 1] << 1;
        val |= ಮುಖವಾಡ[idx + 2] << 2;
        val |= ಮುಖವಾಡ[idx + 3] << 3;
        val |= ಮುಖವಾಡ[idx + 4] << 4;
        val |= ಮುಖವಾಡ[idx + 5] << 5;
        val |= ಮುಖವಾಡ[idx + 6] << 6;
        val |= ಮುಖವಾಡ[idx + 7] << 7;
        bitmask |= (uint64_t)val << idx;
    }

    ಗಾಗಿ (int32_t i = 64; i < GGML_MAX_N_ಎಳೆಗಳು; i++) {
        ಆದರೆ (ಮುಖವಾಡ[i]) {
            fprintf(stderr, "warn: setting thread-affinity for > 64 CPUs isn't supported on windows!\n");
            ನಿಲ್ಲಿಸು;
        }
    }

    DWORD_PTR m = (DWORD_PTR)bitmask;

    m = ಗಣ_Thread_Affinity_ಮುಖವಾಡ(h, m);

    ಹಿಂತಿರುಗಿ m != 0;
}

ಸ್ಥಿರ bool ggml_thread_apply_ಆದ್ಯತೆ(int32_t prio) {
    // Note that on Windows the Process Priority Class must be updated in order to set Thread priority.
    // This is up to the applications.
    DWORD p = THREAD_ಆದ್ಯತೆ_ಸಾಮಾನ್ಯ;
    ಆಯ್ಕೆ (prio) {
        ಸ್ಥಿತಿ GGML_SCHED_PRIO_ಕೆಳಗಿನ:      p = THREAD_ಆದ್ಯತೆ_ಕೆಳಗೆ_ಸಾಮಾನ್ಯ;  ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_SCHED_PRIO_ಸಾಮಾನ್ಯ:   p = THREAD_ಆದ್ಯತೆ_ಸಾಮಾನ್ಯ;        ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_SCHED_PRIO_MEDIUM:   p = THREAD_ಆದ್ಯತೆ_ABOVE_ಸಾಮಾನ್ಯ;  ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_SCHED_PRIO_ಮೇಲಿನ:     p = THREAD_ಆದ್ಯತೆ_HIGHEST;       ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_SCHED_PRIO_REALTIME: p = THREAD_ಆದ್ಯತೆ_ಸಮಯ_CRITICAL; ನಿಲ್ಲಿಸು;
    }

    ಆದರೆ (prio != GGML_SCHED_PRIO_ಕೆಳಗಿನ) {
        // Tell Windows that this thread should not be throttled (needs its own CPU core).
        // Newer Windows 11 versions aggressively park (offline) CPU cores and often place
        // all our threads onto the first 4 cores which results in terrible performance with
        // n_threads > 4
        #ಆದರೆ _WIN32_WINNT >= 0x0602
        THREAD_POWER_THROTTLING_STATE t;
        Zero_ಸ್ಮರಣೆ(&t, ಗಾತ್ರ(t));
        t.ಆವೃತ್ತಿ     = THREAD_POWER_THROTTLING_CURRENT_ಆವೃತ್ತಿ;
        t.Control_ಮುಖವಾಡ = THREAD_POWER_THROTTLING_EXECUTION_ವೇಗ;
        t.State_ಮುಖವಾಡ   = 0;

        ಆದರೆ (!ಗಣ_Thread_Information(GetCurrentThread(), ThreadPowerThrottling, &t, ಗಾತ್ರ(t))) {
            GGML_LOG_DEBUG("failed to disable thread power throttling %d : (%d)\n", prio, (ಪೂರ್ಣಾಂಕ) Get_ಕೊನೆಯ_Error());
            ಹಿಂತಿರುಗಿ false;
        }
        #ಆದರೆ_ಅಂತ್ಯ
    }

    ಆದರೆ (prio == GGML_SCHED_PRIO_ಸಾಮಾನ್ಯ) {
        // Keep inherited policy/priority
        ಹಿಂತಿರುಗಿ true;
    }

    ಆದರೆ (!ಗಣ_Thread_ಆದ್ಯತೆ(GetCurrentThread(), p)) {
        fprintf(stderr, "warn: failed to set thread priority %d : (%d)\n", prio, (ಪೂರ್ಣಾಂಕ) Get_ಕೊನೆಯ_Error());
        ಹಿಂತಿರುಗಿ false;
    }

    ಹಿಂತಿರುಗಿ true;
}

#ಅಥವಾಆದರೆ defined(__APPLE__)
#ಅನ್ತರ್ಭಾವ <sys/types.h>
#ಅನ್ತರ್ಭಾವ <sys/resource.h>

ಸ್ಥಿರ bool ggml_thread_apply_affinity(ಸ್ಥಿರಾಂಕ bool * ಮುಖವಾಡ) {
    // Not supported on Apple platforms
    ಬಳಸದ(ಮುಖವಾಡ);
    ಹಿಂತಿರುಗಿ true;
}

ಸ್ಥಿರ bool ggml_thread_apply_ಆದ್ಯತೆ(int32_t prio) {
    ರಚನೆ sched_param p;
    int32_t policy = SCHED_OTHER;
    ಆಯ್ಕೆ (prio) {
        // TODO: there seems to be no way to set lower prio on Apple platforms
        ಸ್ಥಿತಿ GGML_SCHED_PRIO_ಕೆಳಗಿನ:      policy = SCHED_OTHER; p.sched_ಆದ್ಯತೆ = 0;  ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_SCHED_PRIO_ಸಾಮಾನ್ಯ:   policy = SCHED_OTHER; p.sched_ಆದ್ಯತೆ = 0;  ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_ಆದ್ಯತೆ = 40; ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_SCHED_PRIO_ಮೇಲಿನ:     policy = SCHED_FIFO;  p.sched_ಆದ್ಯತೆ = 80; ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_ಆದ್ಯತೆ = 90; ನಿಲ್ಲಿಸು;
    }

    ಆದರೆ (prio == GGML_SCHED_PRIO_ಸಾಮಾನ್ಯ) {
        // Keep inherited policy/priority
        ಹಿಂತಿರುಗಿ true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    ಆದರೆ (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        ಹಿಂತಿರುಗಿ false;
    }

    ಹಿಂತಿರುಗಿ true;
}

#ಅಥವಾಆದರೆ defined(__linux__)
// TODO: this may not work on BSD, to be verified

ಸ್ಥಿರ bool ggml_thread_apply_affinity(ಸ್ಥಿರಾಂಕ bool * ಮುಖವಾಡ) {
    cpu_ಗಣ_t cpuset;
    ಪೂರ್ಣಾಂಕ err;

    CPU_ZERO(&cpuset);

    ಗಾಗಿ (uint32_t i = 0; i < GGML_MAX_N_ಎಳೆಗಳು; i++) {
        ಆದರೆ (ಮುಖವಾಡ[i]) {
            GGML_PRINT_DEBUG("Thread %lx: adding %d to cpuset\n", pthread_self(), i);
            CPU_ಗಣ(i, &cpuset);
        }
    }

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __ANDROID__
    err = sched_setaffinity(0, ಗಾತ್ರ(cpuset), &cpuset);
    ಆದರೆ (err < 0) {
        err = errno;
    }
#ಇಲ್ಲದಿದ್ದರೆ
    err = pthread_setaffinity_np(pthread_self(), ಗಾತ್ರ(cpuset), &cpuset);
#ಆದರೆ_ಅಂತ್ಯ
    ಆದರೆ (err != 0) {
        fprintf(stderr, "warn: failed to set affinity mask 0x%llx : %s (%d)\n", (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ)ಮುಖವಾಡ, strerror(err), err);
        ಹಿಂತಿರುಗಿ false;
    }

    ಹಿಂತಿರುಗಿ true;
}

ಸ್ಥಿರ bool ggml_thread_apply_ಆದ್ಯತೆ(int32_t prio) {
    ರಚನೆ sched_param p;
    int32_t policy = SCHED_OTHER;
    ಆಯ್ಕೆ (prio) {
        ಸ್ಥಿತಿ GGML_SCHED_PRIO_ಕೆಳಗಿನ:      policy = SCHED_BATCH; p.sched_ಆದ್ಯತೆ = 0;  ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_SCHED_PRIO_ಸಾಮಾನ್ಯ:   policy = SCHED_OTHER; p.sched_ಆದ್ಯತೆ = 0;  ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_ಆದ್ಯತೆ = 40; ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_SCHED_PRIO_ಮೇಲಿನ:     policy = SCHED_FIFO;  p.sched_ಆದ್ಯತೆ = 80; ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_ಆದ್ಯತೆ = 90; ನಿಲ್ಲಿಸು;
    }

    ಆದರೆ (prio == GGML_SCHED_PRIO_ಸಾಮಾನ್ಯ) {
        // Keep inherited policy/priority
        ಹಿಂತಿರುಗಿ true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    ಆದರೆ (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        ಹಿಂತಿರುಗಿ false;
    }

    ಹಿಂತಿರುಗಿ true;
}

#ಇಲ್ಲದಿದ್ದರೆ // unsupported platforms

ಸ್ಥಿರ bool ggml_thread_apply_affinity(ಸ್ಥಿರಾಂಕ bool * ಮುಖವಾಡ) {
    ಬಳಸದ(ಮುಖವಾಡ);
    ಹಿಂತಿರುಗಿ true;
}

ಸ್ಥಿರ bool ggml_thread_apply_ಆದ್ಯತೆ(int32_t prio) {
    ಬಳಸದ(prio);
    ಹಿಂತಿರುಗಿ true;
}

#ಆದರೆ_ಅಂತ್ಯ

ಸ್ಥಿರ bool ggml_thread_cpumask_is_valid(ಸ್ಥಿರಾಂಕ bool * ಮುಖವಾಡ) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_N_ಎಳೆಗಳು; i++) {
        ಆದರೆ (ಮುಖವಾಡ[i]) { ಹಿಂತಿರುಗಿ true; }
    }
    ಹಿಂತಿರುಗಿ false;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_thread_cpumask_ಮುಂದಿನ(ಸ್ಥಿರಾಂಕ bool * global_ಮುಖವಾಡ, bool * local_ಮುಖವಾಡ, bool strict, int32_t* iter) {
    ಆದರೆ (!strict) {
        ಸ್ಮರಣೆ_ನಕಲು(local_ಮುಖವಾಡ, global_ಮುಖವಾಡ, GGML_MAX_N_ಎಳೆಗಳು);
        ಹಿಂತಿರುಗಿ;
    } ಇಲ್ಲದಿದ್ದರೆ {
        memset(local_ಮುಖವಾಡ, 0, GGML_MAX_N_ಎಳೆಗಳು);
        int32_t ತಳ_idx = *iter;
        ಗಾಗಿ (int32_t i = 0; i < GGML_MAX_N_ಎಳೆಗಳು; i++) {
            int32_t idx = ತಳ_idx + i;
            ಆದರೆ (idx >= GGML_MAX_N_ಎಳೆಗಳು) {
                // Just a cheaper modulo
                idx -= GGML_MAX_N_ಎಳೆಗಳು;
            }
            ಆದರೆ (global_ಮುಖವಾಡ[idx]) {
                local_ಮುಖವಾಡ[idx] = 1;
                *iter = idx + 1;
                ಹಿಂತಿರುಗಿ;
            }
        }
    }
}

ಶೂನ್ಯ ggml_threadpool_free(ರಚನೆ ggml_threadpool* threadpool) {
    ಆದರೆ (!threadpool) ಹಿಂತಿರುಗಿ;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ಎಳೆಗಳು = threadpool->n_ಎಳೆಗಳು;

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_USE_OPENMP
    ರಚನೆ ggml_compute_state* workers = threadpool->workers;

    ggml_mutex_ಬೀಗ(&threadpool->mutex);

    threadpool->stop = true;
    threadpool->pause = false;

    ggml_cond_ಪ್ರಸಾರ(&threadpool->cond);
    ggml_mutex_unlock(&threadpool->mutex);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 1; j < n_ಎಳೆಗಳು; j++) {
        int32_t rc = ggml_thread_join(workers[j].thrd, ಖಾಲಿ);
        GGML_ASSERT(rc == GGML_EXIT_SUCCESS || rc == GGML_EXIT_ABORTED);
        ಬಳಸದ(rc);
    }

    ggml_mutex_destroy(&threadpool->mutex);
    ggml_cond_destroy(&threadpool->cond);
#ಆದರೆ_ಅಂತ್ಯ // GGML_USE_OPENMP

    ಸ್ಥಿರಾಂಕ size_t workers_ಗಾತ್ರ = ಗಾತ್ರ(ರಚನೆ ggml_compute_state) * n_ಎಳೆಗಳು;
    ggml_aligned_free(threadpool->workers, workers_ಗಾತ್ರ);
    ggml_aligned_free(threadpool, ಗಾತ್ರ(ರಚನೆ ggml_threadpool));
}

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_USE_OPENMP
// pause/resume must be called under mutex
ಸ್ಥಿರ ಶೂನ್ಯ ggml_threadpool_pause_locked(ರಚನೆ ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Pausing threadpool\n");
    threadpool->pause = true;
    ggml_cond_ಪ್ರಸಾರ(&threadpool->cond);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_threadpool_resume_locked(ರಚನೆ ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Resuming threadpool\n");
    threadpool->pause = false;
    ggml_cond_ಪ್ರಸಾರ(&threadpool->cond);
}
#ಆದರೆ_ಅಂತ್ಯ

ಶೂನ್ಯ ggml_threadpool_pause(ರಚನೆ ggml_threadpool * threadpool) {
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_USE_OPENMP
    ggml_mutex_ಬೀಗ(&threadpool->mutex);
    ಆದರೆ (!threadpool->pause) {
       ggml_threadpool_pause_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#ಇಲ್ಲದಿದ್ದರೆ
    ಬಳಸದ(threadpool);
#ಆದರೆ_ಅಂತ್ಯ
}

ಶೂನ್ಯ ggml_threadpool_resume(ರಚನೆ ggml_threadpool * threadpool) {
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_USE_OPENMP
    ggml_mutex_ಬೀಗ(&threadpool->mutex);
    ಆದರೆ (threadpool->pause) {
       ggml_threadpool_resume_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#ಇಲ್ಲದಿದ್ದರೆ
    ಬಳಸದ(threadpool);
#ಆದರೆ_ಅಂತ್ಯ
}

ರಚನೆ ggml_cplan ggml_ನಕ್ಷೆ_plan(
          ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph,
                               ಪೂರ್ಣಾಂಕ   n_ಎಳೆಗಳು,
            ರಚನೆ ggml_threadpool * threadpool) {

    ಆದರೆ (threadpool == ಖಾಲಿ) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
    }
    ಆದರೆ (n_ಎಳೆಗಳು <= 0) {
        n_ಎಳೆಗಳು = threadpool ? threadpool->n_ಎಳೆಗಳು : GGML_DEFAULT_N_ಎಳೆಗಳು;
    }

#ಆದರೆ defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
    // Emscripten without pthreads support can only use a single thread
    n_ಎಳೆಗಳು = 1;
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(__wasi__)
    // WASI doesn't support parallelism yet
    n_ಎಳೆಗಳು = 1;
#ಆದರೆ_ಅಂತ್ಯ

    size_t work_ಗಾತ್ರ = 0;

    ರಚನೆ ggml_cplan cplan;
    memset(&cplan, 0, ಗಾತ್ರ(ರಚನೆ ggml_cplan));

    ಪೂರ್ಣಾಂಕ max_tasks = 1;

    // thread scheduling for the different operations + work buffer size estimation
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < cgraph->n_nodes; i++) {
        ರಚನೆ ggml_tensor * ಗಂಟು = cgraph->nodes[i];

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_tasks = ggml_get_n_tasks(ಗಂಟು, n_ಎಳೆಗಳು);

        max_tasks = MAX(max_tasks, n_tasks);

        size_t cur = 0;

        ಆದರೆ (!ggml_cpu_extra_work_ಗಾತ್ರ(n_ಎಳೆಗಳು, ಗಂಟು, &cur)) {
            ಆಯ್ಕೆ (ಗಂಟು->op) {
                ಸ್ಥಿತಿ GGML_OP_CPY:
                ಸ್ಥಿತಿ GGML_OP_DUP:
                    {
                        ಆದರೆ (ggml_is_quantized(ಗಂಟು->type) ||
                            // F16 -> BF16 and BF16 -> F16 copies go through intermediate F32
                            (ಗಂಟು->src[0]->type == GGML_TYPE_F16  && ಗಂಟು->src[1] && ಗಂಟು->src[1]->type == GGML_TYPE_BF16) ||
                            (ಗಂಟು->src[0]->type == GGML_TYPE_BF16 && ಗಂಟು->src[1] && ಗಂಟು->src[1]->type == GGML_TYPE_F16) ||
                            // conversion between F32 and I32
                            (ಗಂಟು->src[0]->type == GGML_TYPE_F32 && ಗಂಟು->src[1] && ಗಂಟು->src[1]->type == GGML_TYPE_I32) ||
                            (ಗಂಟು->src[0]->type == GGML_TYPE_I32 && ಗಂಟು->src[1] && ಗಂಟು->src[1]->type == GGML_TYPE_F32)) {
                            cur = ggml_type_ಗಾತ್ರ(GGML_TYPE_F32) * ಗಂಟು->ne[0] * n_tasks;
                        }
                    } ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು:
                ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು_ID:
                ಸ್ಥಿತಿ GGML_OP_ADD1:
                    {
                        ಆದರೆ (ggml_is_quantized(ಗಂಟು->src[0]->type)) {
                            cur = ggml_type_ಗಾತ್ರ(GGML_TYPE_F32) * ಗಂಟು->src[0]->ne[0] * n_tasks;
                        }
                    } ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_OP_ACC:
                    {
                        ಆದರೆ (ggml_is_quantized(ಗಂಟು->src[0]->type)) {
                            cur = ggml_type_ಗಾತ್ರ(GGML_TYPE_F32) * ಗಂಟು->src[1]->ne[0] * n_tasks;
                        }
                    } ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_OP_ಎಣಿಕೆ_EQUAL:
                    {
                        cur = ggml_type_ಗಾತ್ರ(ಗಂಟು->type)*n_tasks;
                    } ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_OP_MUL_MAT:
                    {
                        ಸ್ಥಿರಾಂಕ ಎಣಿಕೆ ggml_type vec_dot_type = type_traits_cpu[ಗಂಟು->src[0]->type].vec_dot_type;
                        ಆದರೆ (ಗಂಟು->src[1]->type != vec_dot_type) {
                            cur = ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(vec_dot_type, ggml_nelements(ಗಂಟು->src[1]));
                        }
                        // Workspace for tiled (see tiled.h)
                        cur = GGML_PAD(cur, 64);
                        cur += ggml_tiled_wdata_ಗಾತ್ರ(n_tasks, ಗಂಟು);
                    } ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_OP_MUL_MAT_ID:
                    {
                        cur = 0;
                        ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0 = ಗಂಟು->src[0];
                        ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1 = ಗಂಟು->src[1];
                        ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * ids = ಗಂಟು->src[2];
                        ಸ್ಥಿರಾಂಕ ಎಣಿಕೆ ggml_type vec_dot_type = type_traits_cpu[src0->type].vec_dot_type;
                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_as = src0->ne[2];
                        // src1
                        ಆದರೆ (src1->type != vec_dot_type) {
                            cur += ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(vec_dot_type, ggml_nelements(src1)) + ಗಾತ್ರ(int64_t);
                        }
                        // matrix_row_counts
                        cur += n_as * ಗಾತ್ರ(int64_t) + ಗಾತ್ರ(int64_t);
                        // matrix_rows
                        cur += n_as*ids->ne[0]*ids->ne[1]*ಗಾತ್ರ(ರಚನೆ mmid_ಅಡ್ಡಸಾಲು_mapping) + ಗಾತ್ರ(int64_t);
                        // atomic_current_chunk
                        cur += CACHE_ಸಾಲು_ಗಾತ್ರ*n_as + CACHE_ಸಾಲು_ಗಾತ್ರ;
                        // Workspace for tiled (see tiled.h)
                        cur = GGML_PAD(cur, 64);
                        cur += ggml_tiled_wdata_ಗಾತ್ರ(n_tasks, ಗಂಟು);
                    } ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_OP_ಹೊರಹೋಗು_PROD:
                    {
                        ಆದರೆ (ggml_is_quantized(ಗಂಟು->src[0]->type) ||
                            ಗಂಟು->src[0]->type == GGML_TYPE_F16) {
                            cur = ggml_type_ಗಾತ್ರ(GGML_TYPE_F32) * ಗಂಟು->src[0]->ne[0] * n_tasks;
                        }
                    } ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_OP_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು:
                    {
                        ಆದರೆ (ಗಂಟು->src[0]->type == GGML_TYPE_F16 && ಗಂಟು->type != GGML_TYPE_F16) {
                            cur = ggml_type_ಗಾತ್ರ(GGML_TYPE_F32) * ಗಂಟು->src[0]->ne[0] * n_tasks;
                        }
                    } ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_OP_SOFT_MAX:
                ಸ್ಥಿತಿ GGML_OP_ROPE:
                ಸ್ಥಿತಿ GGML_OP_ROPE_ಹಿಂಭಾಗ:
                    {
                        cur = ggml_type_ಗಾತ್ರ(GGML_TYPE_F32) * ಗಂಟು->ne[0] * n_tasks;
                    } ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_OP_CONV_TRANSPOSE_1D:
                    {
                        GGML_ASSERT(ಗಂಟು->src[0]->ne[3] == 1);
                        GGML_ASSERT(ಗಂಟು->src[1]->ne[2] == 1);
                        GGML_ASSERT(ಗಂಟು->src[1]->ne[3] == 1);

                        ಸ್ಥಿರಾಂಕ int64_t ne00 = ಗಂಟು->src[0]->ne[0];  // K
                        ಸ್ಥಿರಾಂಕ int64_t ne01 = ಗಂಟು->src[0]->ne[1];  // Cout
                        ಸ್ಥಿರಾಂಕ int64_t ne02 = ಗಂಟು->src[0]->ne[2];  // Cin
                        ಸ್ಥಿರಾಂಕ int64_t ne10 = ಗಂಟು->src[1]->ne[0];  // L
                        ಸ್ಥಿರಾಂಕ int64_t ne11 = ಗಂಟು->src[1]->ne[1];  // Cin

                        ಆದರೆ ((ಗಂಟು->src[0]->type == GGML_TYPE_F16 ||
                             ಗಂಟು->src[0]->type == GGML_TYPE_BF16) &&
                            ಗಂಟು->src[1]->type == GGML_TYPE_F32) {
                            cur += ಗಾತ್ರ(ggml_fp16_t)*ne00*ne01*ne02;
                            cur += ಗಾತ್ರ(ggml_fp16_t)*ne10*ne11;
                        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು->src[0]->type == GGML_TYPE_F32 &&
                                   ಗಂಟು->src[1]->type == GGML_TYPE_F32) {
                            cur += ಗಾತ್ರ(ದಶಮಾಂಶ)*ne00*ne01*ne02;
                            cur += ಗಾತ್ರ(ದಶಮಾಂಶ)*ne10*ne11;
                        } ಇಲ್ಲದಿದ್ದರೆ {
                            GGML_ABORT("fatal error");
                        }
                    } ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_OP_CONV_2D:
                ಸ್ಥಿತಿ GGML_OP_CONV_3D:
                    {
                        cur = GGML_IM2_ಲಂಬಸಾಲು_WORK_ಗಾತ್ರ;
                    } ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_OP_CONV_TRANSPOSE_2D:
                    {
                        ಸ್ಥಿರಾಂಕ int64_t ne00 = ಗಂಟು->src[0]->ne[0]; // W
                        ಸ್ಥಿರಾಂಕ int64_t ne01 = ಗಂಟು->src[0]->ne[1]; // H
                        ಸ್ಥಿರಾಂಕ int64_t ne02 = ಗಂಟು->src[0]->ne[2]; // Channels Out
                        ಸ್ಥಿರಾಂಕ int64_t ne03 = ಗಂಟು->src[0]->ne[3]; // Channels In

                        ಸ್ಥಿರಾಂಕ int64_t ne10 = ಗಂಟು->src[1]->ne[0]; // W
                        ಸ್ಥಿರಾಂಕ int64_t ne11 = ಗಂಟು->src[1]->ne[1]; // H
                        ಸ್ಥಿರಾಂಕ int64_t ne12 = ಗಂಟು->src[1]->ne[2]; // Channels In
                        ಸ್ಥಿರಾಂಕ int64_t ne13 = ಗಂಟು->src[1]->ne[3]; // Batch

                        GGML_ASSERT(ಗಂಟು->src[0]->type == GGML_TYPE_F16 || ಗಂಟು->src[0]->type == GGML_TYPE_F32);
                        GGML_ASSERT(ಗಂಟು->src[1]->type == GGML_TYPE_F32);

                        cur += ggml_type_ಗಾತ್ರ(ಗಂಟು->src[0]->type) * ne00 * ne01 * ne02 * ne03;
                        cur += ggml_type_ಗಾತ್ರ(ಗಂಟು->src[0]->type) * ne10 * ne11 * ne12 * ne13;

                    } ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_OP_ತುದಿ_K:
                    {
                        cur += ಗಾತ್ರ(int32_t)*ಗಂಟು->src[0]->ne[0]*n_tasks;
                    } ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_OP_FLASH_ATTN_EXT:
                    {
                        ಸ್ಥಿರಾಂಕ int64_t neq2 = ಗಂಟು->src[0]->ne[2]; // number of query heads
                        ಸ್ಥಿರಾಂಕ int64_t DK = ಗಂಟು->src[1]->ne[0];
                        ಸ್ಥಿರಾಂಕ int64_t DV = ಗಂಟು->src[2]->ne[0];

                        // Tiled flash attention scratch (tile sizes defined in common.h)
                        // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
                        size_t prefill  = ಗಾತ್ರ(ದಶಮಾಂಶ)*(GGML_FA_TILE_Q*DK + 2*GGML_FA_TILE_Q*GGML_FA_TILE_KV + GGML_FA_TILE_Q*DV + GGML_FA_TILE_KV*DV + GGML_FA_TILE_KV*DK)*n_tasks;

                        // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
                        // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
                        size_t n_chunks = n_tasks;
                        size_t decode   = ಗಾತ್ರ(ದಶಮಾಂಶ)*(neq2*n_chunks*(2+DV) + n_tasks*(DK + 2*DV));

                        cur += MAX(prefill, decode);
                    } ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_OP_FLASH_ATTN_ಹಿಂಭಾಗ:
                    {
                        ಸ್ಥಿರಾಂಕ int64_t    D = ಗಂಟು->src[0]->ne[0];
                        ಸ್ಥಿರಾಂಕ int64_t ne11 = ggml_ಮೇಲೆ(ಗಂಟು->src[1]->ne[1], GGML_SOFT_MAX_UNROLL);
                        ಸ್ಥಿರಾಂಕ int64_t mxDn = MAX(D, ne11) * 2; // *2 because of S and SM in ggml_compute_forward_flash_attn_back
                        ಆದರೆ (ಗಂಟು->src[1]->type == GGML_TYPE_F32) {
                            cur  = ಗಾತ್ರ(ದಶಮಾಂಶ)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += ಗಾತ್ರ(ದಶಮಾಂಶ)*mxDn*n_tasks; // this is overestimated by x2
                        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು->src[1]->type == GGML_TYPE_F16) {
                            cur  = ಗಾತ್ರ(ದಶಮಾಂಶ)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += ಗಾತ್ರ(ದಶಮಾಂಶ)*mxDn*n_tasks; // this is overestimated by x2
                        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು->src[1]->type == GGML_TYPE_BF16) {
                            cur  = ಗಾತ್ರ(ದಶಮಾಂಶ)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += ಗಾತ್ರ(ದಶಮಾಂಶ)*mxDn*n_tasks; // this is overestimated by x2
                        }
                    } ನಿಲ್ಲಿಸು;

                ಸ್ಥಿತಿ GGML_OP_CROSS_ENTROPY_LOSS:
                    {
                        cur = ggml_type_ಗಾತ್ರ(ಗಂಟು->type)*(n_tasks + ಗಂಟು->src[0]->ne[0]*n_tasks);
                    } ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_OP_GATED_DELTA_NET:
                    {
                        ಸ್ಥಿರಾಂಕ int64_t S_v = ಗಂಟು->src[2]->ne[0];
                        ಸ್ಥಿರಾಂಕ int64_t K   = ggml_get_op_params_i32(ಗಂಟು, 0);
                        ಸ್ಥಿರಾಂಕ int64_t ಪ್ರತಿ_thread = S_v + (K > 1 ? S_v * S_v : 0);
                        cur = ಪ್ರತಿ_thread * ಗಾತ್ರ(ದಶಮಾಂಶ) * n_tasks;
                    } ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_OP_ಎಣಿಕೆ:
                    {
                        GGML_ABORT("fatal error");
                    }
                ಸ್ಥಿತಿ GGML_OP_LIGHTNING_INDEXER:
                    {
                        // temp buffer for dequantizing lightning indexer keys
                        ಸ್ಥಿರಾಂಕ int64_t ne10 = ಗಂಟು->src[1]->ne[0];
                        cur += ಗಾತ್ರ(ದಶಮಾಂಶ)*ne10*n_tasks;
                    } ನಿಲ್ಲಿಸು;
                ಪೂರ್ವನಿಯೋಜಿತ:
                    ನಿಲ್ಲಿಸು;
            }
        }

        work_ಗಾತ್ರ = MAX(work_ಗಾತ್ರ, cur);
    }

    ಆದರೆ (work_ಗಾತ್ರ > 0) {
        work_ಗಾತ್ರ += CACHE_ಸಾಲು_ಗಾತ್ರ*(n_ಎಳೆಗಳು);
    }

    cplan.threadpool = threadpool;
    cplan.n_ಎಳೆಗಳು  = MIN(max_tasks, n_ಎಳೆಗಳು);
    cplan.work_ಗಾತ್ರ  = work_ಗಾತ್ರ;
    cplan.work_ದತ್ತಾಂಶ  = ಖಾಲಿ;

    ಹಿಂತಿರುಗಿ cplan;
}


// Try to fuse the current node with subsequent nodes for better performance.
// Returns the number of nodes skipped by fusion (>=1), or 0 if no fusion was applied.
ಸ್ಥಿರ bool ggml_cpu_disable_fusion = false;  // initialized once in ggml_cpu_init(), read-only afterwards

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ ggml_cpu_try_fuse_ops(
        ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಗಂಟು_n,
        ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params,
        ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cplan * cplan) {

    ಆದರೆ (ggml_cpu_disable_fusion || cplan->use_ref) {
        ಹಿಂತಿರುಗಿ 0;
    }

    ರಚನೆ ggml_tensor * ಗಂಟು = cgraph->nodes[ಗಂಟು_n];

    ಆದರೆ (ಗಂಟು->op == GGML_OP_RMS_NORM) {
        // RMS_NORM + MUL fusion
        ಸ್ಥಿರಾಂಕ ಎಣಿಕೆ ggml_op fuse_ops[] = { GGML_OP_RMS_NORM, GGML_OP_MUL };
        ಆದರೆ (ggml_can_fuse(cgraph, ಗಂಟು_n, fuse_ops, 2)) {
            ರಚನೆ ggml_tensor * mul_ಗಂಟು = cgraph->nodes[ಗಂಟು_n + 1];
            ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * mul_w = (mul_ಗಂಟು->src[0] == ಗಂಟು)
                ? mul_ಗಂಟು->src[1] : mul_ಗಂಟು->src[0];
            ಆದರೆ (ಗಂಟು->src[0]->type  == GGML_TYPE_F32 &&
                mul_ಗಂಟು->type      == GGML_TYPE_F32 &&
                mul_w->type         == GGML_TYPE_F32 &&
                mul_w->ne[0]        == ಗಂಟು->ne[0]   &&
                mul_w->nb[0]        == ಗಾತ್ರ(ದಶಮಾಂಶ)) {

                ggml_compute_forward_rms_norm_mul_fused(params, ಗಂಟು, mul_ಗಂಟು);
                ಹಿಂತಿರುಗಿ 1;
            }
        }
    }

    ಹಿಂತಿರುಗಿ 0;
}

ಸ್ಥಿರ thread_ret_t ggml_ನಕ್ಷೆ_compute_thread(ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    ರಚನೆ ggml_compute_state * state = (ರಚನೆ ggml_compute_state *) ದತ್ತಾಂಶ;
    ರಚನೆ ggml_threadpool    * tp    = state->threadpool;

    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph = tp->cgraph;
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cplan  * cplan  = tp->cplan;

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_ಗಣ_numa_thread_affinity(state->ith);
#ಇಲ್ಲದಿದ್ದರೆ
    ಗಣ_numa_thread_affinity(state->ith);
#ಆದರೆ_ಅಂತ್ಯ

    ರಚನೆ ggml_compute_params params = {
        /*.ith        =*/ state->ith,
        /*.nth        =*/ atomic_ತುಂಬು_explicit(&tp->n_ನಕ್ಷೆ, memory_order_relaxed) & GGML_THREADPOOL_N_ಎಳೆಗಳು_ಮುಖವಾಡ,
        /*.wsize      =*/ cplan->work_ಗಾತ್ರ,
        /*.wdata      =*/ cplan->work_ದತ್ತಾಂಶ,
        /*.threadpool =*/ tp,
        /*.use_ref    =*/ cplan->use_ref,
    };

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p\n", state->ith, (ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *)cplan);
#ಇಲ್ಲದಿದ್ದರೆ
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p last-graph %d\n", state->ith, (ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *)cplan, state->ಕೊನೆಯ_ನಕ್ಷೆ);
#ಆದರೆ_ಅಂತ್ಯ

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಗಂಟು_n = 0; ಗಂಟು_n < cgraph->n_nodes && atomic_ತುಂಬು_explicit(&tp->abort, memory_order_relaxed) != ಗಂಟು_n; ಗಂಟು_n++) {
        ರಚನೆ ggml_tensor * ಗಂಟು = cgraph->nodes[ಗಂಟು_n];

        ಆದರೆ (ggml_op_is_ಖಾಲಿಗಳು(ಗಂಟು->op)) {
            // skip NOPs
            ಮುಂದುವರಿ;
        }

        ಆದರೆ ((ಗಂಟು->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ಮುಂದುವರಿ;
        }

        // TODO: move fused-op detection into ggml_graph_plan so fusion decisions are made once at planning time
        // Try fused ops, fall back to normal compute
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_fused = ggml_cpu_try_fuse_ops(cgraph, ಗಂಟು_n, &params, cplan);
        ಆದರೆ (n_fused > 0) {
            ಗಂಟು_n += n_fused;
        } ಇಲ್ಲದಿದ್ದರೆ {
            ggml_compute_forward(&params, ಗಂಟು);
        }

        ಆದರೆ (state->ith == 0 && cplan->abort_callback &&
                cplan->abort_callback(cplan->abort_callback_ದತ್ತಾಂಶ)) {
            atomic_store_explicit(&tp->abort, ಗಂಟು_n + 1, memory_order_relaxed);
            tp->ec    = GGML_ಸ್ಥಿತಿ_ABORTED;
        }

        ಆದರೆ (ಗಂಟು_n + 1 < cgraph->n_nodes) {
            ggml_barrier(state->threadpool);
        }
    }

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p\n", state->ith, (ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *)cplan);
#ಇಲ್ಲದಿದ್ದರೆ
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p last-graph %d\n", state->ith, (ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *)cplan, state->ಕೊನೆಯ_ನಕ್ಷೆ);
#ಆದರೆ_ಅಂತ್ಯ

    ggml_barrier(state->threadpool);

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(state->ith);
#ಆದರೆ_ಅಂತ್ಯ

    ಹಿಂತಿರುಗಿ 0;
}

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_USE_OPENMP

// check if thread is ready to proceed (exit from polling or sleeping)
// returns true if loops should exit, sets state->pending to indicate new work
ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool ggml_ನಕ್ಷೆ_compute_thread_ready(ರಚನೆ ggml_compute_state * state) {
    ರಚನೆ ggml_threadpool * threadpool = state->threadpool;

    ಆದರೆ (state->pending || threadpool->stop || threadpool->pause) { ಹಿಂತಿರುಗಿ true; }

    // check for new graph/work
    ಪೂರ್ಣಾಂಕ n_ನಕ್ಷೆ   = atomic_ತುಂಬು_explicit(&threadpool->n_ನಕ್ಷೆ, memory_order_relaxed);
    ಪೂರ್ಣಾಂಕ n_ಎಳೆಗಳು = n_ನಕ್ಷೆ & GGML_THREADPOOL_N_ಎಳೆಗಳು_ಮುಖವಾಡ;
    ಆದರೆ (n_ನಕ್ಷೆ != state->ಕೊನೆಯ_ನಕ್ಷೆ) {
        state->pending    = (state->ith < n_ಎಳೆಗಳು);
        state->ಕೊನೆಯ_ನಕ್ಷೆ = n_ನಕ್ಷೆ;
        ಹಿಂತಿರುಗಿ true;
    }

    ಹಿಂತಿರುಗಿ false;
}

// sync thread state after polling
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_ನಕ್ಷೆ_compute_thread_sync(ರಚನೆ ggml_compute_state * state) {
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_TSAN_ENABLED
    atomic_fetch_ಕೂಡಿಸು_explicit(&state->threadpool->n_ನಕ್ಷೆ, 0, memory_order_seq_cst);
    #ಇಲ್ಲದಿದ್ದರೆ
    atomic_thread_fence(memory_order_seq_cst);
    #ಆದರೆ_ಅಂತ್ಯ
    ಬಳಸದ(state);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool ggml_ನಕ್ಷೆ_compute_poll_for_work(ರಚನೆ ggml_compute_state * state) {
    ರಚನೆ ggml_threadpool * threadpool = state->threadpool;

    // This seems to make 0 ... 100 a decent range for polling level across modern processors.
    // Perhaps, we can adjust it dynamically based on load and things.
    ಸ್ಥಿರಾಂಕ uint64_t n_ಸುತ್ತುಗಳು = 1024UL * 128 * threadpool->poll;

    ಗಾಗಿ (uint64_t i=0; !ggml_ನಕ್ಷೆ_compute_thread_ready(state) && i < n_ಸುತ್ತುಗಳು; i++) {
        // No new work. Keep polling.
        ggml_thread_cpu_relax();
    }

    ಹಿಂತಿರುಗಿ state->pending;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool ggml_ನಕ್ಷೆ_compute_check_for_work(ರಚನೆ ggml_compute_state * state) {
    ರಚನೆ ggml_threadpool * threadpool = state->threadpool;

    ಆದರೆ (ggml_ನಕ್ಷೆ_compute_poll_for_work(state)) {
        ggml_ನಕ್ಷೆ_compute_thread_sync(state);
        ಹಿಂತಿರುಗಿ state->pending;
    }

    ggml_mutex_ಬೀಗ_shared(&threadpool->mutex);
    ಆಗ (!ggml_ನಕ್ಷೆ_compute_thread_ready(state)) {
        // No new work. Wait for the signal.
        GGML_PRINT_DEBUG("thread #%d waiting for work (sleeping)\n", state->ith);
        ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
    }
    ggml_mutex_unlock_shared(&threadpool->mutex);

    ಹಿಂತಿರುಗಿ state->pending;
}

ಸ್ಥಿರ thread_ret_t ggml_ನಕ್ಷೆ_compute_secondary_thread(ಶೂನ್ಯ* ದತ್ತಾಂಶ) {
    ರಚನೆ ggml_compute_state * state = (ರಚನೆ ggml_compute_state *) ದತ್ತಾಂಶ;
    ರಚನೆ ggml_threadpool * threadpool = state->threadpool;

    ggml_thread_apply_ಆದ್ಯತೆ(threadpool->prio);
    ಆದರೆ (ggml_thread_cpumask_is_valid(state->cpumask)) {
        ggml_thread_apply_affinity(state->cpumask);
    }

    ಆಗ (true) {
        // Check if we need to sleep
        ಆಗ (threadpool->pause) {
            GGML_PRINT_DEBUG("thread #%d inside pause loop\n", state->ith);
            ggml_mutex_ಬೀಗ_shared(&threadpool->mutex);
            ಆದರೆ (threadpool->pause) {
                ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
            }
            GGML_PRINT_DEBUG("thread #%d resuming after wait\n", state->ith);
            ggml_mutex_unlock_shared(&threadpool->mutex);
        }

        // This needs to be checked for after the cond_wait
        ಆದರೆ (threadpool->stop) ನಿಲ್ಲಿಸು;

        // Check if there is new work
        // The main thread is the only one that can dispatch new work

        ggml_ನಕ್ಷೆ_compute_check_for_work(state);
        ಆದರೆ (state->pending) {
            state->pending = false;
            ggml_ನಕ್ಷೆ_compute_thread(state);
        }
    }

    ಹಿಂತಿರುಗಿ (thread_ret_t) 0;
}

// Start processing new graph
ಸ್ಥಿರ ಶೂನ್ಯ ggml_ನಕ್ಷೆ_compute_kickoff(ರಚನೆ ggml_threadpool * threadpool, ಪೂರ್ಣಾಂಕ n_ಎಳೆಗಳು)
{
    // Always take the mutex here because the worker threads are doing hybrid poll/wait

    ggml_mutex_ಬೀಗ(&threadpool->mutex);

    // Update the number of active threads and the graph count
    ಪೂರ್ಣಾಂಕ n_ನಕ್ಷೆ = atomic_ತುಂಬು_explicit(&threadpool->n_ನಕ್ಷೆ, memory_order_relaxed) >> GGML_THREADPOOL_N_ಎಳೆಗಳು_ಬಿಟ್ಟುಗಳು;
    n_ನಕ್ಷೆ = ((n_ನಕ್ಷೆ + 1) << GGML_THREADPOOL_N_ಎಳೆಗಳು_ಬಿಟ್ಟುಗಳು) | (n_ಎಳೆಗಳು & GGML_THREADPOOL_N_ಎಳೆಗಳು_ಮುಖವಾಡ);

    GGML_PRINT_DEBUG("compute-kickoff: n_threads %d n_graph %d\n", n_ಎಳೆಗಳು, n_ನಕ್ಷೆ);

    // Indicate the graph is ready to be processed
    // We need the full seq-cst fence here because of the polling threads (used in thread_sync)
    atomic_store_explicit(&threadpool->n_ನಕ್ಷೆ, n_ನಕ್ಷೆ, memory_order_seq_cst);

    ಆದರೆ (threadpool->pause) {
       // Update main thread prio and affinity to match the threadpool settings
       ggml_thread_apply_ಆದ್ಯತೆ(threadpool->prio);
       ಆದರೆ (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
           ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
       }

       // resume does cond broadcast
       ggml_threadpool_resume_locked(threadpool);
    } ಇಲ್ಲದಿದ್ದರೆ {
       ggml_cond_ಪ್ರಸಾರ(&threadpool->cond);
    }

    ggml_mutex_unlock(&threadpool->mutex);
}

#ಆದರೆ_ಅಂತ್ಯ // GGML_USE_OPENMP

ಸ್ಥಿರ ರಚನೆ ggml_threadpool * ggml_threadpool_new_impl(
    ರಚನೆ ggml_threadpool_params * tpp,
               ರಚನೆ ggml_cgraph * cgraph,
                ರಚನೆ ggml_cplan * cplan) {

    ರಚನೆ ggml_threadpool * threadpool =
        ggml_aligned_malloc(ಗಾತ್ರ(ರಚನೆ ggml_threadpool));
    {
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->n_ನಕ್ಷೆ          = 0;
        threadpool->n_barrier        = 0;
        threadpool->n_barrier_passed = 0;
        threadpool->current_chunk    = 0;
        threadpool->stop             = false;
        threadpool->pause            = tpp->paused;
        threadpool->abort            = -1;
        threadpool->workers          = ಖಾಲಿ;
        threadpool->n_ಎಳೆಗಳು        = tpp->n_ಎಳೆಗಳು;
        threadpool->poll             = tpp->poll;
        threadpool->prio             = tpp->prio;
        threadpool->ec               = GGML_ಸ್ಥಿತಿ_SUCCESS;
    }

    // Allocate and init workers state
    ಸ್ಥಿರಾಂಕ size_t workers_ಗಾತ್ರ = ಗಾತ್ರ(ರಚನೆ ggml_compute_state) * tpp->n_ಎಳೆಗಳು;
    ರಚನೆ ggml_compute_state * workers = ggml_aligned_malloc(workers_ಗಾತ್ರ);

    memset(workers, 0, workers_ಗಾತ್ರ);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < tpp->n_ಎಳೆಗಳು; j++) {
        workers[j].threadpool = threadpool;
        workers[j].ith        = j;
    }

    threadpool->workers = workers;

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_OPENMP
    int32_t cpumask_iter = 0;

    // Compute CPU masks for each thread
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < tpp->n_ಎಳೆಗಳು; j++) {
        ggml_thread_cpumask_ಮುಂದಿನ(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);
    }
#ಇಲ್ಲದಿದ್ದರೆ // GGML_USE_OPENMP
    ggml_mutex_init(&threadpool->mutex);
    ggml_cond_init(&threadpool->cond);

    // Spin the threads for all workers, and update CPU placements.
    // Place the main thread last (towards the higher numbered CPU cores).

    int32_t cpumask_iter = 0;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 1; j < tpp->n_ಎಳೆಗಳು; j++) {
        ggml_thread_cpumask_ಮುಂದಿನ(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);

        int32_t rc = ggml_thread_create(&workers[j].thrd, ಖಾಲಿ, ggml_ನಕ್ಷೆ_compute_secondary_thread, &workers[j]);
        GGML_ASSERT(rc == 0);
    }

    ggml_thread_cpumask_ಮುಂದಿನ(tpp->cpumask, workers[0].cpumask, tpp->strict_cpu, &cpumask_iter);

    ಆದರೆ (!threadpool->pause) {
        // Update main thread prio and affinity at the start, otherwise we'll do it in resume
        ggml_thread_apply_ಆದ್ಯತೆ(threadpool->prio);
        ಆದರೆ (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
            ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
        }
    }
#ಆದರೆ_ಅಂತ್ಯ // GGML_USE_OPENMP

    ಹಿಂತಿರುಗಿ threadpool;
}

ರಚನೆ ggml_threadpool * ggml_threadpool_new(ರಚನೆ ggml_threadpool_params * tpp) {
    ಹಿಂತಿರುಗಿ ggml_threadpool_new_impl(tpp, ಖಾಲಿ, ಖಾಲಿ);
}

ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ ggml_ನಕ್ಷೆ_compute(ರಚನೆ ggml_cgraph * cgraph, ರಚನೆ ggml_cplan * cplan) {
    ggml_cpu_init();

    GGML_ASSERT(cplan);
    GGML_ASSERT(cplan->n_ಎಳೆಗಳು > 0);
    GGML_ASSERT(cplan->work_ಗಾತ್ರ == 0 || cplan->work_ದತ್ತಾಂಶ != ಖಾಲಿ);

    ಪೂರ್ಣಾಂಕ n_ಎಳೆಗಳು                               = cplan->n_ಎಳೆಗಳು;
    ರಚನೆ ggml_threadpool * threadpool = cplan->threadpool;

    bool disposable_threadpool = false;

    ಆದರೆ (threadpool == ಖಾಲಿ) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
        disposable_threadpool = true;

        ರಚನೆ ggml_threadpool_params ttp = ggml_threadpool_params_default(n_ಎಳೆಗಳು);
        threadpool = ggml_threadpool_new_impl(&ttp, cgraph, cplan);
    } ಇಲ್ಲದಿದ್ದರೆ {
        // Reset some of the parameters that need resetting
        // No worker threads should be accessing the parameters below at this stage
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->current_chunk    = 0;
        threadpool->abort            = -1;
        threadpool->ec               = GGML_ಸ್ಥಿತಿ_SUCCESS;
    }

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_OPENMP
    ಆದರೆ (n_ಎಳೆಗಳು > 1) {
        #ಸೂಚನೆ omp parallel ಅಂಕೆ_ಎಳೆಗಳು(n_ಎಳೆಗಳು)
        {
            #ಸೂಚನೆ omp single
            {
                // update the number of threads from the actual number of threads that we got from OpenMP
                n_ಎಳೆಗಳು = omp_get_ಅಂಕೆ_ಎಳೆಗಳು();
                atomic_store_explicit(&threadpool->n_ನಕ್ಷೆ, n_ಎಳೆಗಳು, memory_order_relaxed);
            }

            // Apply thread CPU mask and priority
            ಪೂರ್ಣಾಂಕ ith = omp_get_thread_ಅಂಕೆ();

            ggml_thread_apply_ಆದ್ಯತೆ(threadpool->prio);
            ಆದರೆ (ggml_thread_cpumask_is_valid(threadpool->workers[ith].cpumask)) {
                ggml_thread_apply_affinity(threadpool->workers[ith].cpumask);
            }
            ggml_ನಕ್ಷೆ_compute_thread(&threadpool->workers[ith]);
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        atomic_store_explicit(&threadpool->n_ನಕ್ಷೆ, 1, memory_order_relaxed);
        ggml_ನಕ್ಷೆ_compute_thread(&threadpool->workers[0]);
    }
#ಇಲ್ಲದಿದ್ದರೆ
    ಆದರೆ (n_ಎಳೆಗಳು > threadpool->n_ಎಳೆಗಳು) {
        GGML_LOG_WARN("cplan requested more threads (%d) than available (%d)\n", n_ಎಳೆಗಳು, threadpool->n_ಎಳೆಗಳು);
        n_ಎಳೆಗಳು = threadpool->n_ಎಳೆಗಳು;
    }

    // Kick all threads to start the new graph
    ggml_ನಕ್ಷೆ_compute_kickoff(threadpool, n_ಎಳೆಗಳು);

    // This is a work thread too
    ggml_ನಕ್ಷೆ_compute_thread(&threadpool->workers[0]);
#ಆದರೆ_ಅಂತ್ಯ

    // don't leave affinity set on the main thread
    clear_numa_thread_affinity();

    ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ ret = threadpool->ec;

    ಆದರೆ (disposable_threadpool) {
        ggml_threadpool_free(threadpool);
    }

    ಹಿಂತಿರುಗಿ ret;
}

ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ ggml_ನಕ್ಷೆ_compute_with_ctx(ರಚನೆ ggml_context * ctx, ರಚನೆ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ n_ಎಳೆಗಳು) {
    ರಚನೆ ggml_cplan cplan = ggml_ನಕ್ಷೆ_plan(cgraph, n_ಎಳೆಗಳು, ಖಾಲಿ);

    cplan.work_ದತ್ತಾಂಶ = (uint8_t *)ggml_new_ಬಫರ್(ctx, cplan.work_ಗಾತ್ರ);

    ಹಿಂತಿರುಗಿ ggml_ನಕ್ಷೆ_compute(cgraph, &cplan);
}

ಶೂನ್ಯ ggml_cpu_fp32_ಗಮ್ಯ_fp32(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x, ದಶಮಾಂಶ * y, int64_t n) {
    ಸ್ಮರಣೆ_ನಕಲು(y, x, n * ಗಾತ್ರ(ದಶಮಾಂಶ));
}

ಶೂನ್ಯ ggml_cpu_fp32_ಗಮ್ಯ_fp16(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x, ggml_fp16_t * y, int64_t n) {
    int64_t i = 0;
#ಆದರೆ defined(__F16C__)
#ಆದರೆ defined(__AVX512F__)
    ಗಾಗಿ (; i + 15 < n; i += 16) {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        __m256i y_vec = _mm512_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256((__m256i *)(y + i), y_vec);
    }
#ಆದರೆ_ಅಂತ್ಯ
    ಗಾಗಿ (; i + 7 < n; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m128i y_vec = _mm256_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i *)(y + i), y_vec);
    }
    ಗಾಗಿ (; i + 3 < n; i += 4) {
        __m128 x_vec = _mm_loadu_ps(x + i);
        __m128i y_vec = _mm_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storel_epi64((__m128i *)(y + i), y_vec);
    }
#ಅಥವಾಆದರೆ defined(__riscv_zvfh)
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ vl; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(&x[i], vl);
        vfloat16m1_t vy = __riscv_vfncvt_f_f_w_f16m1(vx, vl);
        __riscv_vse16_v_f16m1((_Float16 *)&y[i], vy, vl);
    }
#ಅಥವಾಆದರೆ defined(__VXE__) || defined(__VXE2__)
    ಗಾಗಿ (; i + 7 < n; i += 8) {
        ಸ್ಥಿರಾಂಕ uint32x4_t v_yl = __lzs_f32cx4_to_f16(vec_xl(0, x + i + 0));
        ಸ್ಥಿರಾಂಕ uint32x4_t v_yh = __lzs_f32cx4_to_f16(vec_xl(0, x + i + 4));
        vec_xst(vec_pack(v_yl, v_yh), 0, (uint16_t *)(y + i));
    }
#ಆದರೆ_ಅಂತ್ಯ
    ಗಾಗಿ (; i < n; ++i) {
        y[i] = GGML_CPU_FP32_ಗಮ್ಯ_FP16(x[i]);
    }
}

ಶೂನ್ಯ ggml_cpu_fp16_ಗಮ್ಯ_fp32(ಸ್ಥಿರಾಂಕ ggml_fp16_t * x, ದಶಮಾಂಶ * y, int64_t n) {
    int64_t i = 0;
#ಆದರೆ defined(__F16C__)
#ಆದರೆ defined(__AVX512F__)
    ಗಾಗಿ (; i + 15 < n; i += 16) {
        __m256i x_vec = _mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)(x + i));
        __m512 y_vec = _mm512_cvtph_ps(x_vec);
        _mm512_storeu_ps(y + i, y_vec);
    }
#ಆದರೆ_ಅಂತ್ಯ
    ಗಾಗಿ (; i + 7 < n; i += 8) {
        __m128i x_vec = _mm_loadu_si128((ಸ್ಥಿರಾಂಕ __m128i *)(x + i));
        __m256 y_vec = _mm256_cvtph_ps(x_vec);
        _mm256_storeu_ps(y + i, y_vec);
    }
    ಗಾಗಿ (; i + 3 < n; i += 4) {
        __m128i x_vec = _mm_loadl_epi64((ಸ್ಥಿರಾಂಕ __m128i *)(x + i));
        __m128 y_vec = _mm_cvtph_ps(x_vec);
        _mm_storeu_ps(y + i, y_vec);
    }

#ಅಥವಾಆದರೆ defined(__riscv_v_intrinsic) && defined(__riscv_zvfhmin)
    // calculate step size
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ epr = __riscv_vsetvlmax_e16m2();
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಹೆಜ್ಜೆ = epr * 2;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ np = (n & ~(ಹೆಜ್ಜೆ - 1));

    // unroll by 2
    ಗಾಗಿ (; i < np; i += ಹೆಜ್ಜೆ) {
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((ಸ್ಥಿರಾಂಕ _Float16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vfloat16m2_t ax1 = __riscv_vle16_v_f16m2((ಸ್ಥಿರಾಂಕ _Float16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvt_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    ಪೂರ್ಣಾಂಕ vl;
    ಗಾಗಿ (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((ಸ್ಥಿರಾಂಕ _Float16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }

#ಆದರೆ_ಅಂತ್ಯ

    ಗಾಗಿ (; i < n; ++i) {
        y[i] = GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i]);
    }
}

ಶೂನ್ಯ ggml_cpu_fp32_ಗಮ್ಯ_bf16(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x, ggml_bf16_t * y, int64_t n) {
    int64_t i = 0;
    ಗಾಗಿ (; i < n; ++i) {
        y[i] = GGML_FP32_ಗಮ್ಯ_BF16(x[i]);
    }
}

ಶೂನ್ಯ ggml_cpu_fp32_ಗಮ್ಯ_i32(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x, int32_t * y, int64_t n) {
    int64_t i = 0;
    ಗಾಗಿ (; i < n; ++i) {
        y[i] = x[i];
    }
}

ಶೂನ್ಯ ggml_cpu_bf16_ಗಮ್ಯ_fp32(ಸ್ಥಿರಾಂಕ ggml_bf16_t * x, ದಶಮಾಂಶ * y, int64_t n) {
    int64_t i = 0;
#ಆದರೆ defined(__AVX2__)
#ಆದರೆ defined(__AVX512F__)
    ಗಾಗಿ (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i,
                        _mm512_castsi512_ps(
                            _mm512_slli_epi32(
                                _mm512_cvtepu16_epi32(
                                    _mm256_loadu_si256(
                                        (ಸ್ಥಿರಾಂಕ __m256i *)(x + i))),
                                16)));
    }
#ಆದರೆ_ಅಂತ್ಯ
    ಗಾಗಿ (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i,
                        _mm256_castsi256_ps(
                            _mm256_slli_epi32(
                                _mm256_cvtepu16_epi32(
                                    _mm_loadu_si128(
                                        (ಸ್ಥಿರಾಂಕ __m128i *)(x + i))),
                                16)));
    }
#ಅಥವಾಆದರೆ defined(__riscv_v_intrinsic) && defined(__riscv_zvfbfmin)
    // calculate step size
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ epr = __riscv_vsetvlmax_e16m2();
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಹೆಜ್ಜೆ = epr * 2;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ np = (n & ~(ಹೆಜ್ಜೆ - 1));

    // unroll by 2
    ಗಾಗಿ (; i < np; i += ಹೆಜ್ಜೆ) {
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((ಸ್ಥಿರಾಂಕ __bf16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vbfloat16m2_t ax1 = __riscv_vle16_v_bf16m2((ಸ್ಥಿರಾಂಕ __bf16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    ಪೂರ್ಣಾಂಕ vl;
    ಗಾಗಿ (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((ಸ್ಥಿರಾಂಕ __bf16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }
#ಆದರೆ_ಅಂತ್ಯ
    ಗಾಗಿ (; i < n; i++) {
        y[i] = GGML_BF16_ಗಮ್ಯ_FP32(x[i]);
    }
}

ಪೂರ್ಣಾಂಕ ggml_cpu_has_avx(ಶೂನ್ಯ) {
#ಆದರೆ defined(__AVX__)
    ಹಿಂತಿರುಗಿ 1;
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ 0;
#ಆದರೆ_ಅಂತ್ಯ
}

ಪೂರ್ಣಾಂಕ ggml_cpu_has_avx_vnni(ಶೂನ್ಯ) {
#ಆದರೆ defined(__AVXVNNI__)
    ಹಿಂತಿರುಗಿ 1;
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ 0;
#ಆದರೆ_ಅಂತ್ಯ
}

ಪೂರ್ಣಾಂಕ ggml_cpu_has_avx2(ಶೂನ್ಯ) {
#ಆದರೆ defined(__AVX2__)
    ಹಿಂತಿರುಗಿ 1;
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ 0;
#ಆದರೆ_ಅಂತ್ಯ
}

ಪೂರ್ಣಾಂಕ ggml_cpu_has_avx512(ಶೂನ್ಯ) {
#ಆದರೆ defined(__AVX512F__)
    ಹಿಂತಿರುಗಿ 1;
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ 0;
#ಆದರೆ_ಅಂತ್ಯ
}

ಪೂರ್ಣಾಂಕ ggml_cpu_has_avx512_vbmi(ಶೂನ್ಯ) {
#ಆದರೆ defined(__AVX512VBMI__)
    ಹಿಂತಿರುಗಿ 1;
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ 0;
#ಆದರೆ_ಅಂತ್ಯ
}

ಪೂರ್ಣಾಂಕ ggml_cpu_has_avx512_vnni(ಶೂನ್ಯ) {
#ಆದರೆ defined(__AVX512VNNI__)
    ಹಿಂತಿರುಗಿ 1;
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ 0;
#ಆದರೆ_ಅಂತ್ಯ
}

ಪೂರ್ಣಾಂಕ ggml_cpu_has_avx512_bf16(ಶೂನ್ಯ) {
#ಆದರೆ defined(__AVX512BF16__)
    ಹಿಂತಿರುಗಿ 1;
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ 0;
#ಆದರೆ_ಅಂತ್ಯ
}

ಪೂರ್ಣಾಂಕ ggml_cpu_has_amx_int8(ಶೂನ್ಯ) {
#ಆದರೆ defined(__AMX_INT8__)
    ಹಿಂತಿರುಗಿ 1;
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ 0;
#ಆದರೆ_ಅಂತ್ಯ
}

ಪೂರ್ಣಾಂಕ ggml_cpu_has_bmi2(ಶೂನ್ಯ) {
#ಆದರೆ defined(__BMI2__)
    ಹಿಂತಿರುಗಿ 1;
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ 0;
#ಆದರೆ_ಅಂತ್ಯ
}

ಪೂರ್ಣಾಂಕ ggml_cpu_has_fma(ಶೂನ್ಯ) {
#ಆದರೆ defined(__FMA__)
    ಹಿಂತಿರುಗಿ 1;
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ 0;
#ಆದರೆ_ಅಂತ್ಯ
}

ಪೂರ್ಣಾಂಕ ggml_cpu_has_arm_fma(ಶೂನ್ಯ) {
#ಆದರೆ defined(__ARM_FEATURE_FMA)
    ಹಿಂತಿರುಗಿ 1;
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ 0;
#ಆದರೆ_ಅಂತ್ಯ
}

ಪೂರ್ಣಾಂಕ ggml_cpu_has_riscv_v(ಶೂನ್ಯ) {
#ಆದರೆ defined(__riscv_v_intrinsic)
    ಹಿಂತಿರುಗಿ 1;
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ 0;
#ಆದರೆ_ಅಂತ್ಯ
}

ಪೂರ್ಣಾಂಕ ggml_cpu_get_rvv_vlen(ಶೂನ್ಯ) {
#ಆದರೆ defined(__riscv) && defined(__riscv_v_intrinsic)
    ಹಿಂತಿರುಗಿ ggml_riscv_arch_features.rvv_vlen;
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ 0;
#ಆದರೆ_ಅಂತ್ಯ
}

ಪೂರ್ಣಾಂಕ ggml_cpu_has_f16c(ಶೂನ್ಯ) {
#ಆದರೆ defined(__F16C__)
    ಹಿಂತಿರುಗಿ 1;
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ 0;
#ಆದರೆ_ಅಂತ್ಯ
}

ಪೂರ್ಣಾಂಕ ggml_cpu_has_fp16_va(ಶೂನ್ಯ) {
#ಆದರೆ defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    ಹಿಂತಿರುಗಿ 1;
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ 0;
#ಆದರೆ_ಅಂತ್ಯ
}

ಪೂರ್ಣಾಂಕ ggml_cpu_has_wasm_simd(ಶೂನ್ಯ) {
#ಆದರೆ defined(__wasm_simd128__)
    ಹಿಂತಿರುಗಿ 1;
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ 0;
#ಆದರೆ_ಅಂತ್ಯ
}

ಪೂರ್ಣಾಂಕ ggml_cpu_has_llamafile(ಶೂನ್ಯ) {
#ಆದರೆ defined(GGML_USE_LLAMAFILE)
    ಹಿಂತಿರುಗಿ 1;
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ 0;
#ಆದರೆ_ಅಂತ್ಯ
}

ಪೂರ್ಣಾಂಕ ggml_cpu_has_sse3(ಶೂನ್ಯ) {
#ಆದರೆ defined(__SSE3__)
    ಹಿಂತಿರುಗಿ 1;
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ 0;
#ಆದರೆ_ಅಂತ್ಯ
}

ಪೂರ್ಣಾಂಕ ggml_cpu_has_ssse3(ಶೂನ್ಯ) {
#ಆದರೆ defined(__SSSE3__)
    ಹಿಂತಿರುಗಿ 1;
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ 0;
#ಆದರೆ_ಅಂತ್ಯ
}

ಪೂರ್ಣಾಂಕ ggml_cpu_has_vsx(ಶೂನ್ಯ) {
#ಆದರೆ defined(__POWER9_VECTOR__)
    ಹಿಂತಿರುಗಿ 1;
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ 0;
#ಆದರೆ_ಅಂತ್ಯ
}

ಪೂರ್ಣಾಂಕ ggml_cpu_has_vxe(ಶೂನ್ಯ) {
#ಆದರೆ defined(__VXE__) || defined(__VXE2__)
    ಹಿಂತಿರುಗಿ 1;
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ 0;
#ಆದರೆ_ಅಂತ್ಯ
}

ಪೂರ್ಣಾಂಕ ggml_cpu_has_neon(ಶೂನ್ಯ) {
#ಆದರೆ defined(__ARM_ARCH) && defined(__ARM_NEON)
    ಹಿಂತಿರುಗಿ 1;
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ 0;
#ಆದರೆ_ಅಂತ್ಯ
}

ಪೂರ್ಣಾಂಕ ggml_cpu_has_dotprod(ಶೂನ್ಯ) {
#ಆದರೆ defined(__ARM_ARCH) && defined(__ARM_FEATURE_DOTPROD)
    ಹಿಂತಿರುಗಿ 1;
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ 0;
#ಆದರೆ_ಅಂತ್ಯ
}

ಪೂರ್ಣಾಂಕ ggml_cpu_has_sve(ಶೂನ್ಯ) {
#ಆದರೆ defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    ಹಿಂತಿರುಗಿ 1;
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ 0;
#ಆದರೆ_ಅಂತ್ಯ
}

ಪೂರ್ಣಾಂಕ ggml_cpu_has_matmul_int8(ಶೂನ್ಯ) {
#ಆದರೆ defined(__ARM_ARCH) && defined(__ARM_FEATURE_MATMUL_INT8)
    ಹಿಂತಿರುಗಿ 1;
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ 0;
#ಆದರೆ_ಅಂತ್ಯ
}

ಪೂರ್ಣಾಂಕ ggml_cpu_get_sve_cnt(ಶೂನ್ಯ) {
#ಆದರೆ defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    ಹಿಂತಿರುಗಿ ggml_arm_arch_features.sve_cnt;
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ 0;
#ಆದರೆ_ಅಂತ್ಯ
}

ಪೂರ್ಣಾಂಕ ggml_cpu_has_sme(ಶೂನ್ಯ) {
#ಆದರೆ defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME)
    ಹಿಂತಿರುಗಿ 1;
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ 0;
#ಆದರೆ_ಅಂತ್ಯ
}

ಪೂರ್ಣಾಂಕ ggml_cpu_has_sme2(ಶೂನ್ಯ) {
#ಆದರೆ defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME2)
    ಹಿಂತಿರುಗಿ 1;
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ 0;
#ಆದರೆ_ಅಂತ್ಯ
}

ಶೂನ್ಯ ggml_cpu_init(ಶೂನ್ಯ) {
    // needed to initialize ggml_time
    {
        ರಚನೆ ggml_init_params params = { 0, ಖಾಲಿ, false };
        ರಚನೆ ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    ggml_critical_section_ಆರಂಭ();

    ಸ್ಥಿರ bool is_ಮೊದಲ_call = true;

    ಆದರೆ (is_ಮೊದಲ_call) {
        // initialize GELU, Quick GELU, SILU and EXP F32 tables
        {
            ಸ್ಥಿರಾಂಕ uint64_t t_ಆರಂಭ = ggml_ಸಮಯ_us(); ಬಳಸದ(t_ಆರಂಭ);

            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < (1 << 16); ++i) {
                ಒಕ್ಕೂಟ {
                    uint16_t u16;
                    ggml_fp16_t fp16;
                } u = {i};
                ದಶಮಾಂಶ f = GGML_COMPUTE_FP16_ಗಮ್ಯ_FP32(u.fp16);
                ggml_ಕೋಷ್ಟಕ_f32_f16[i] = f;
                ggml_ಕೋಷ್ಟಕ_gelu_f16[i] = GGML_CPU_FP32_ಗಮ್ಯ_FP16(ggml_gelu_f32(f));
                ggml_ಕೋಷ್ಟಕ_gelu_ಶೀಘ್ರ_f16[i] = GGML_CPU_FP32_ಗಮ್ಯ_FP16(ggml_gelu_ಶೀಘ್ರ_f32(f));
            }

            // initialize E8M0 half table (256 entries)
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < (1 << 8); ++i) {
                ggml_ಕೋಷ್ಟಕ_f32_e8m0_half[i] = GGML_E8_M0_ಗಮ್ಯ_FP32_HALF(i);
            }

            // initialize UE4M3 table (256 entries)
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < (1 << 8); ++i) {
                ggml_ಕೋಷ್ಟಕ_f32_ue4m3[i] = ggml_ue4m3_ಗಮ್ಯ_fp32(i);
            }

            ಸ್ಥಿರಾಂಕ uint64_t t_ಕೊನೆ = ggml_ಸಮಯ_us(); ಬಳಸದ(t_ಕೊನೆ);

            GGML_PRINT_DEBUG("%s: GELU, Quick GELU, SILU and EXP tables initialized in %f ms\n", __func__, (t_ಕೊನೆ - t_ಆರಂಭ)/1000.0);

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_OPENMP
            //if (!getenv("OMP_WAIT_POLICY")) {
            //    // set the wait policy to active, so that OpenMP threads don't sleep
            //    setenv("OMP_WAIT_POLICY", "active", 0)
            //}

            ಆದರೆ (!getenv("KMP_BLOCKTIME")) {
                // set the time to wait before sleeping a thread
                // this is less aggressive than setting the wait policy to active, but should achieve similar results in most cases
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ _WIN32
                _putenv_s("KMP_BLOCKTIME", "200"); // 200ms
#ಇಲ್ಲದಿದ್ದರೆ
                setenv("KMP_BLOCKTIME", "200", 0); // 200ms
#ಆದರೆ_ಅಂತ್ಯ
            }
#ಆದರೆ_ಅಂತ್ಯ
        }

#ಆದರೆ defined(__ARM_ARCH)
        ggml_init_arm_arch_features();
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(__riscv)
        ggml_init_riscv_arch_features();
#ಆದರೆ_ಅಂತ್ಯ

        {
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * env = getenv("GGML_CPU_DISABLE_FUSION");
            ggml_cpu_disable_fusion = (env != ಖಾಲಿ && atoi(env) == 1);
        }

        is_ಮೊದಲ_call = false;
    }

    ggml_critical_section_ಕೊನೆ();
}
