#ಆದರೆ defined(__GNUC__)
#ಸೂಚನೆ GCC diagnostic ignored "-Wpedantic"
#ಸೂಚನೆ GCC diagnostic ignored "-Wunused-local-typedefs"
#ಆದರೆ_ಅಂತ್ಯ

#ಅನ್ತರ್ಭಾವ "amx.h"
#ಅನ್ತರ್ಭಾವ "mmq.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-cpu-impl.h"
#ಅನ್ತರ್ಭಾವ "simd-mappings.h"
#ಅನ್ತರ್ಭಾವ "quants.h"
#ಅನ್ತರ್ಭಾವ "ggml-quants.h"
#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <type_traits>

#ಆದರೆ defined(__gnu_linux__)
#ಅನ್ತರ್ಭಾವ <sys/syscall.h>
#ಅನ್ತರ್ಭಾವ <unistd.h>
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ (defined(_WIN32) || defined(_WIN64))
#ವ್ಯಾಖ್ಯಾನಿಸು RESTRICT __restrict
#ಇಲ್ಲದಿದ್ದರೆ
#ವ್ಯಾಖ್ಯಾನಿಸು RESTRICT __restrict__
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ (defined(_WIN32) || defined(_WIN64))
#ವ್ಯಾಖ್ಯಾನಿಸು ALWAYS_INLINE __forceinline
#ಅಥವಾಆದರೆ __has_attribute(always_inline) || defined(__GNUC__)
#ವ್ಯಾಖ್ಯಾನಿಸು ALWAYS_INLINE __attribute__((__always_inline__)) ಸಾಲೊಳಗೆ
#ಇಲ್ಲದಿದ್ದರೆ
#ವ್ಯಾಖ್ಯಾನಿಸು ALWAYS_INLINE ಸಾಲೊಳಗೆ
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(__AMX_INT8__) && defined(__AVX512VNNI__)

ನಾಮಾಕಾಶ {

// Forced unrolling
ಮಾದರಿ <ಪೂರ್ಣಾಂಕ n>
ರಚನೆ Unroll {
    ಮಾದರಿ <ಬಗೆ_ಹೆಸರು Func, ಬಗೆ_ಹೆಸರು... ವಾದಗಳು>
    ALWAYS_INLINE ಶೂನ್ಯ ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ Func& f, ವಾದಗಳು... ವಾದಗಳು_2) ಸ್ಥಿರಾಂಕ {
        Unroll<n - 1>{}(f, ವಾದಗಳು_2...);
        f(std::integral_constant<ಪೂರ್ಣಾಂಕ, n - 1>{}, ವಾದಗಳು_2...);
    }
};

ಮಾದರಿ <>
ರಚನೆ Unroll<1> {
    ಮಾದರಿ <ಬಗೆ_ಹೆಸರು Func, ಬಗೆ_ಹೆಸರು... ವಾದಗಳು>
    ALWAYS_INLINE ಶೂನ್ಯ ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ Func& f, ವಾದಗಳು... ವಾದಗಳು_2) ಸ್ಥಿರಾಂಕ {
        f(std::integral_constant<ಪೂರ್ಣಾಂಕ, 0>{}, ವಾದಗಳು_2...);
    }
};

// type traits
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T> ರಚನೆ PackedTypes {};
ಮಾದರಿ <> ರಚನೆ PackedTypes<ಖಂಡ_q4_0> { ಬಳಸಿ type = int8_t; };
ಮಾದರಿ <> ರಚನೆ PackedTypes<ಖಂಡ_q4_1> { ಬಳಸಿ type = uint8_t; };
ಮಾದರಿ <> ರಚನೆ PackedTypes<ಖಂಡ_q8_0> { ಬಳಸಿ type = int8_t; };
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T> ಬಳಸಿ packed_B_type = ಬಗೆ_ಹೆಸರು PackedTypes<T>::type;

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
ರಚನೆ do_compensate : std::integral_constant<ತಾರ್ಕಿಕ,
    std::is_same<T, ಖಂಡ_q8_0>::value> {};

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
ರಚನೆ do_unpack : std::integral_constant<ತಾರ್ಕಿಕ,
    std::is_same<T, ಖಂಡ_q4_0>::value ||
    std::is_same<T, ಖಂಡ_q4_1>::value> {};

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
ರಚನೆ is_type_qkk : std::integral_constant<ತಾರ್ಕಿಕ,
    std::is_same<T, ಖಂಡ_q4_K>::value ||
    std::is_same<T, ಖಂಡ_q5_K>::value ||
    std::is_same<T, ಖಂಡ_q6_K>::value ||
    std::is_same<T, ಖಂಡ_iq4_xs>::value> {};

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_DISPATCH_FLOATING_TYPES(TYPE, ...)                                        \
    [&] {                                                                              \
        ಆಯ್ಕೆ (TYPE) {                                                                \
            ಸ್ಥಿತಿ GGML_TYPE_F16: {                                                      \
                ಬಳಸಿ type = ggml_fp16_t;                                              \
                constexpr ಪೂರ್ಣಾಂಕ blck_ಗಾತ್ರ = 16;                                          \
                ಹಿಂತಿರುಗಿ __VA_ARGS__();                                                  \
            }                                                                          \
            ಸ್ಥಿತಿ GGML_TYPE_BF16: {                                                     \
                ಬಳಸಿ type = ggml_bf16_t;                                              \
                constexpr ಪೂರ್ಣಾಂಕ blck_ಗಾತ್ರ = 32;                                          \
                ಹಿಂತಿರುಗಿ __VA_ARGS__();                                                  \
            }                                                                          \
            ಪೂರ್ವನಿಯೋಜಿತ:                                                                   \
                fprintf(stderr, "Unsupported floating data type\n");                   \
        }                                                                              \
    }()

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_DISPATCH_QTYPES(QT, ...)                                                  \
    [&] {                                                                              \
        ಆಯ್ಕೆ (QT) {                                                                  \
            ಸ್ಥಿತಿ GGML_TYPE_Q4_0: {                                                     \
                ಬಳಸಿ type = ಖಂಡ_q4_0;                                               \
                ಬಳಸಿ vec_dot_type = ಖಂಡ_q8_0;                                       \
                constexpr ಪೂರ್ಣಾಂಕ blck_ಗಾತ್ರ = QK4_0;                                       \
                ಹಿಂತಿರುಗಿ __VA_ARGS__();                                                  \
            }                                                                          \
            ಸ್ಥಿತಿ GGML_TYPE_Q4_1: {                                                     \
                ಬಳಸಿ type = ಖಂಡ_q4_1;                                               \
                ಬಳಸಿ vec_dot_type = ಖಂಡ_q8_1;                                       \
                constexpr ಪೂರ್ಣಾಂಕ blck_ಗಾತ್ರ = QK4_1;                                       \
                ಹಿಂತಿರುಗಿ __VA_ARGS__();                                                  \
            }                                                                          \
            ಸ್ಥಿತಿ GGML_TYPE_Q8_0: {                                                     \
                ಬಳಸಿ type = ಖಂಡ_q8_0;                                               \
                ಬಳಸಿ vec_dot_type = ಖಂಡ_q8_0;                                       \
                constexpr ಪೂರ್ಣಾಂಕ blck_ಗಾತ್ರ = QK8_0;                                       \
                ಹಿಂತಿರುಗಿ __VA_ARGS__();                                                  \
            }                                                                          \
            ಸ್ಥಿತಿ GGML_TYPE_Q4_K: {                                                     \
                ಬಳಸಿ type = ಖಂಡ_q4_K;                                               \
                ಬಳಸಿ vec_dot_type = ಖಂಡ_q8_K;                                       \
                constexpr ಪೂರ್ಣಾಂಕ blck_ಗಾತ್ರ = QK_K;                                        \
                ಹಿಂತಿರುಗಿ __VA_ARGS__();                                                  \
            }                                                                          \
            ಸ್ಥಿತಿ GGML_TYPE_Q5_K: {                                                     \
                ಬಳಸಿ type = ಖಂಡ_q5_K;                                               \
                ಬಳಸಿ vec_dot_type = ಖಂಡ_q8_K;                                       \
                constexpr ಪೂರ್ಣಾಂಕ blck_ಗಾತ್ರ = QK_K;                                        \
                ಹಿಂತಿರುಗಿ __VA_ARGS__();                                                  \
            }                                                                          \
            ಸ್ಥಿತಿ GGML_TYPE_Q6_K: {                                                     \
                ಬಳಸಿ type = ಖಂಡ_q6_K;                                               \
                ಬಳಸಿ vec_dot_type = ಖಂಡ_q8_K;                                       \
                constexpr ಪೂರ್ಣಾಂಕ blck_ಗಾತ್ರ = QK_K;                                        \
                ಹಿಂತಿರುಗಿ __VA_ARGS__();                                                  \
            }                                                                          \
            ಸ್ಥಿತಿ GGML_TYPE_IQ4_XS: {                                                   \
                ಬಳಸಿ type = ಖಂಡ_iq4_xs;                                             \
                ಬಳಸಿ vec_dot_type = ಖಂಡ_q8_K;                                       \
                constexpr ಪೂರ್ಣಾಂಕ blck_ಗಾತ್ರ = QK_K;                                        \
                ಹಿಂತಿರುಗಿ __VA_ARGS__();                                                  \
            }                                                                          \
            ಪೂರ್ವನಿಯೋಜಿತ:                                                                   \
                fprintf(stderr, "Unsupported quantized data type: %d\n", ಪೂರ್ಣಾಂಕ(TYPE));   \
        }                                                                              \
    }()

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_DISPATCH_BOOL(BOOL_V, BOOL_NAME, ...)                                     \
    [&] {                                                                              \
        ಆದರೆ (BOOL_V) {                                                                  \
            constexpr ತಾರ್ಕಿಕ BOOL_NAME = ನಿಜ;                                           \
            ಹಿಂತಿರುಗಿ __VA_ARGS__();                                                      \
        } ಇಲ್ಲದಿದ್ದರೆ {                                                                       \
            constexpr ತಾರ್ಕಿಕ BOOL_NAME = ಸುಳ್ಳು;                                          \
            ಹಿಂತಿರುಗಿ __VA_ARGS__();                                                      \
        }                                                                              \
    }()

// define amx tile config data structure
ರಚನೆ tile_config_t{
    uint8_t palette_id = 0;
    uint8_t ಆರಂಭ_ಅಡ್ಡಸಾಲು = 0;
    uint8_t reserved_0[14] = {0};
    uint16_t colsb[16] = {0};
    uint8_t ಅಡ್ಡಸಾಲುಗಳು[16] = {0};
};

// Notes: amx tile config
//
// Typically, TMUL calculates A and B of size 16 x 64 containing INT8 values,
// and accumulate the result to a 16 x 16 matrix C containing INT32 values,
//
// As many GGUF quantized types as `block_size` of 32, so a 16-16-32 config is used
// instead of the normally used 16-16-64 config.
//
//    Block A: {16, 32}, dtype = int8_t
//    Block B: {16, 32}, dtype = uint8_t/int8_t
//    Block C: {16, 16}, dtype = int32_t
//
// Block B needs to be prepacked to vnni format before feeding into  TMUL:
//    packed_B: from {n, k} to {k/vnni_blk, n, vnni_blck}, viewed in 2d, we get {8, 64}
//
// Therefore, we get tileconfig:
//             A    B    C
//    rows    16    8   16
//    colsb   32   64   16
//
// For tile distribution, follow a 2-2-4 pattern, e.g. A used TMM2-TMM3, B used TMM0-TMM1,
// C used TMM4-TMM7:
//            B TMM0  B TMM1
//    A TMM2  C TMM4  C TMM6
//    A TMM3  C TMM5  C TMM7
//
// Each `amx` kernel handles 4 blocks at a time: 2MB * 2NB, when m < 2 * BLOCK_M, unpack A
// will be needed.
//
// Here another commonly used pattern 1-3-3 is skipped, as it is mostly used when m <=16;
// and the single batch gemm (m=1) has a special fast path with `avx512-vnni`.
//
// ref: https://www.intel.com/content/www/us/en/developer/articles/code-sample/
//    advanced-matrix-extensions-intrinsics-functions.html
//

ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_tile_config_init(ಶೂನ್ಯ) {
    ಸ್ಥಿರ thread_local ತಾರ್ಕಿಕ ಮುಗಿದ = ಸುಳ್ಳು;

    ಆದರೆ (ಮುಗಿದ) {
        ಹಿಂತಿರುಗಿ;
    }

    alignas(64) tile_config_t tc = {};
    tc.palette_id = 1;
    tc.ಆರಂಭ_ಅಡ್ಡಸಾಲು = 0;
    tc.ಅಡ್ಡಸಾಲುಗಳು[0] = 8;   tc.colsb[0] = 64;
    tc.ಅಡ್ಡಸಾಲುಗಳು[1] = 8;   tc.colsb[1] = 64;
    tc.ಅಡ್ಡಸಾಲುಗಳು[2] = 16;  tc.colsb[2] = 32;
    tc.ಅಡ್ಡಸಾಲುಗಳು[3] = 16;  tc.colsb[3] = 32;
    tc.ಅಡ್ಡಸಾಲುಗಳು[4] = 16;  tc.colsb[4] = 64;
    tc.ಅಡ್ಡಸಾಲುಗಳು[5] = 16;  tc.colsb[5] = 64;
    tc.ಅಡ್ಡಸಾಲುಗಳು[6] = 16;  tc.colsb[6] = 64;
    tc.ಅಡ್ಡಸಾಲುಗಳು[7] = 16;  tc.colsb[7] = 64;

    _tile_loadconfig(&tc);
    ಮುಗಿದ = ನಿಜ;
}

// we need an extra 16 * 4B (TILE_N * int32_t) for each NB/KB block for compensation.
// See the notes `s8s8 igemm compensation in avx512-vnni` for detail.
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು TB>
ಪೂರ್ಣಾಂಕ get_tile_ಗಾತ್ರ() {
    ಪೂರ್ಣಾಂಕ tile_ಗಾತ್ರ = TILE_N * ಗಾತ್ರ(TB);
    ಆದರೆ (do_compensate<TB>::value) {
        tile_ಗಾತ್ರ += TILE_N * ಗಾತ್ರ(int32_t);
    }
    ಆದರೆ (std::is_same<TB, ಖಂಡ_q4_K>::value ||
        std::is_same<TB, ಖಂಡ_q5_K>::value) {
        tile_ಗಾತ್ರ += TILE_N * 4;
    }
    ಆದರೆ (std::is_same<TB, ಖಂಡ_iq4_xs>::value) {
        tile_ಗಾತ್ರ += TILE_N * 2;
    }
    ಹಿಂತಿರುಗಿ tile_ಗಾತ್ರ;
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು TB, ಪೂರ್ಣಾಂಕ ಖಂಡ_K>
ಪೂರ್ಣಾಂಕ get_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ K) {
    ಪೂರ್ಣಾಂಕ KB = K / ಖಂಡ_K;
    ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = KB * ಗಾತ್ರ(TB);
    ಆದರೆ (do_compensate<TB>::value) {
        ಅಡ್ಡಸಾಲು_ಗಾತ್ರ += KB * ಗಾತ್ರ(int32_t);
    }
    ಆದರೆ (std::is_same<TB, ಖಂಡ_q4_K>::value ||
        std::is_same<TB, ಖಂಡ_q5_K>::value) {
        ಅಡ್ಡಸಾಲು_ಗಾತ್ರ += KB * 4;
    }
    ಆದರೆ (std::is_same<TB, ಖಂಡ_iq4_xs>::value) {
        ಅಡ್ಡಸಾಲು_ಗಾತ್ರ += KB * 2;
    }
    ಹಿಂತಿರುಗಿ ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
}

// transpose utils
#ವ್ಯಾಖ್ಯಾನಿಸು SHUFFLE_EPI32(a, b, ಮುಖವಾಡ) \
    _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b), ಮುಖವಾಡ))
ಸಾಲೊಳಗೆ ಶೂನ್ಯ transpose_8x8_32bit(__m256i * v, __m256i * v1) {
    // unpacking and 32-bit elements
    v1[0] = _mm256_unpacklo_epi32(v[0], v[1]);
    v1[1] = _mm256_unpackhi_epi32(v[0], v[1]);
    v1[2] = _mm256_unpacklo_epi32(v[2], v[3]);
    v1[3] = _mm256_unpackhi_epi32(v[2], v[3]);
    v1[4] = _mm256_unpacklo_epi32(v[4], v[5]);
    v1[5] = _mm256_unpackhi_epi32(v[4], v[5]);
    v1[6] = _mm256_unpacklo_epi32(v[6], v[7]);
    v1[7] = _mm256_unpackhi_epi32(v[6], v[7]);

    // shuffling the 32-bit elements
    v[0] = SHUFFLE_EPI32(v1[0], v1[2], 0x44);
    v[1] = SHUFFLE_EPI32(v1[0], v1[2], 0xee);
    v[2] = SHUFFLE_EPI32(v1[4], v1[6], 0x44);
    v[3] = SHUFFLE_EPI32(v1[4], v1[6], 0xee);
    v[4] = SHUFFLE_EPI32(v1[1], v1[3], 0x44);
    v[5] = SHUFFLE_EPI32(v1[1], v1[3], 0xee);
    v[6] = SHUFFLE_EPI32(v1[5], v1[7], 0x44);
    v[7] = SHUFFLE_EPI32(v1[5], v1[7], 0xee);

    // shuffling 128-bit elements
    v1[0] = _mm256_permute2f128_si256(v[2], v[0], 0x02);
    v1[1] = _mm256_permute2f128_si256(v[3], v[1], 0x02);
    v1[2] = _mm256_permute2f128_si256(v[6], v[4], 0x02);
    v1[3] = _mm256_permute2f128_si256(v[7], v[5], 0x02);
    v1[4] = _mm256_permute2f128_si256(v[2], v[0], 0x13);
    v1[5] = _mm256_permute2f128_si256(v[3], v[1], 0x13);
    v1[6] = _mm256_permute2f128_si256(v[6], v[4], 0x13);
    v1[7] = _mm256_permute2f128_si256(v[7], v[5], 0x13);
}

ಸಾಲೊಳಗೆ ಶೂನ್ಯ transpose_16x4_32bit(__m512i * r, __m512i * d) {

    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ __m512i index1 = mm512_ಗಣ_epi32(
        0x0f, 0x0b, 0x07, 0x03,
        0x0e, 0x0a, 0x06, 0x02,
        0x0d, 0x09, 0x05, 0x01,
        0x0c, 0x08, 0x04, 0x00);

    d[0] = _mm512_permutexvar_epi32(index1, r[0]);
    d[1] = _mm512_permutexvar_epi32(index1, r[1]);
    d[2] = _mm512_permutexvar_epi32(index1, r[2]);
    d[3] = _mm512_permutexvar_epi32(index1, r[3]);

    r[0] = _mm512_shuffle_i32x4(d[0], d[1], 0x44);
    r[1] = _mm512_shuffle_i32x4(d[0], d[1], 0xee);
    r[2] = _mm512_shuffle_i32x4(d[2], d[3], 0x44);
    r[3] = _mm512_shuffle_i32x4(d[2], d[3], 0xee);

    d[0] = _mm512_shuffle_i32x4(r[0], r[2], 0x88);
    d[1] = _mm512_shuffle_i32x4(r[0], r[2], 0xdd);
    d[2] = _mm512_shuffle_i32x4(r[1], r[3], 0x88);
    d[3] = _mm512_shuffle_i32x4(r[1], r[3], 0xdd);
}

ಸಾಲೊಳಗೆ ಶೂನ್ಯ transpose_16x16_32bit(__m512i * v) {
    __m512i v1[16];
    v1[0] = _mm512_unpacklo_epi32(v[0], v[1]);
    v1[1] = _mm512_unpackhi_epi32(v[0], v[1]);
    v1[2] = _mm512_unpacklo_epi32(v[2], v[3]);
    v1[3] = _mm512_unpackhi_epi32(v[2], v[3]);
    v1[4] = _mm512_unpacklo_epi32(v[4], v[5]);
    v1[5] = _mm512_unpackhi_epi32(v[4], v[5]);
    v1[6] = _mm512_unpacklo_epi32(v[6], v[7]);
    v1[7] = _mm512_unpackhi_epi32(v[6], v[7]);
    v1[8] = _mm512_unpacklo_epi32(v[8], v[9]);
    v1[9] = _mm512_unpackhi_epi32(v[8], v[9]);
    v1[10] = _mm512_unpacklo_epi32(v[10], v[11]);
    v1[11] = _mm512_unpackhi_epi32(v[10], v[11]);
    v1[12] = _mm512_unpacklo_epi32(v[12], v[13]);
    v1[13] = _mm512_unpackhi_epi32(v[12], v[13]);
    v1[14] = _mm512_unpacklo_epi32(v[14], v[15]);
    v1[15] = _mm512_unpackhi_epi32(v[14], v[15]);

    v[0] = _mm512_unpacklo_epi64(v1[0], v1[2]);
    v[1] = _mm512_unpackhi_epi64(v1[0], v1[2]);
    v[2] = _mm512_unpacklo_epi64(v1[1], v1[3]);
    v[3] = _mm512_unpackhi_epi64(v1[1], v1[3]);
    v[4] = _mm512_unpacklo_epi64(v1[4], v1[6]);
    v[5] = _mm512_unpackhi_epi64(v1[4], v1[6]);
    v[6] = _mm512_unpacklo_epi64(v1[5], v1[7]);
    v[7] = _mm512_unpackhi_epi64(v1[5], v1[7]);
    v[8] = _mm512_unpacklo_epi64(v1[8], v1[10]);
    v[9] = _mm512_unpackhi_epi64(v1[8], v1[10]);
    v[10] = _mm512_unpacklo_epi64(v1[9], v1[11]);
    v[11] = _mm512_unpackhi_epi64(v1[9], v1[11]);
    v[12] = _mm512_unpacklo_epi64(v1[12], v1[14]);
    v[13] = _mm512_unpackhi_epi64(v1[12], v1[14]);
    v[14] = _mm512_unpacklo_epi64(v1[13], v1[15]);
    v[15] = _mm512_unpackhi_epi64(v1[13], v1[15]);

    v1[0] = _mm512_shuffle_i32x4(v[0], v[4], 0x88);
    v1[1] = _mm512_shuffle_i32x4(v[1], v[5], 0x88);
    v1[2] = _mm512_shuffle_i32x4(v[2], v[6], 0x88);
    v1[3] = _mm512_shuffle_i32x4(v[3], v[7], 0x88);
    v1[4] = _mm512_shuffle_i32x4(v[0], v[4], 0xdd);
    v1[5] = _mm512_shuffle_i32x4(v[1], v[5], 0xdd);
    v1[6] = _mm512_shuffle_i32x4(v[2], v[6], 0xdd);
    v1[7] = _mm512_shuffle_i32x4(v[3], v[7], 0xdd);
    v1[8] = _mm512_shuffle_i32x4(v[8], v[12], 0x88);
    v1[9] = _mm512_shuffle_i32x4(v[9], v[13], 0x88);
    v1[10] = _mm512_shuffle_i32x4(v[10], v[14], 0x88);
    v1[11] = _mm512_shuffle_i32x4(v[11], v[15], 0x88);
    v1[12] = _mm512_shuffle_i32x4(v[8], v[12], 0xdd);
    v1[13] = _mm512_shuffle_i32x4(v[9], v[13], 0xdd);
    v1[14] = _mm512_shuffle_i32x4(v[10], v[14], 0xdd);
    v1[15] = _mm512_shuffle_i32x4(v[11], v[15], 0xdd);

    v[0] = _mm512_shuffle_i32x4(v1[0], v1[8], 0x88);
    v[1] = _mm512_shuffle_i32x4(v1[1], v1[9], 0x88);
    v[2] = _mm512_shuffle_i32x4(v1[2], v1[10], 0x88);
    v[3] = _mm512_shuffle_i32x4(v1[3], v1[11], 0x88);
    v[4] = _mm512_shuffle_i32x4(v1[4], v1[12], 0x88);
    v[5] = _mm512_shuffle_i32x4(v1[5], v1[13], 0x88);
    v[6] = _mm512_shuffle_i32x4(v1[6], v1[14], 0x88);
    v[7] = _mm512_shuffle_i32x4(v1[7], v1[15], 0x88);
    v[8] = _mm512_shuffle_i32x4(v1[0], v1[8], 0xdd);
    v[9] = _mm512_shuffle_i32x4(v1[1], v1[9], 0xdd);
    v[10] = _mm512_shuffle_i32x4(v1[2], v1[10], 0xdd);
    v[11] = _mm512_shuffle_i32x4(v1[3], v1[11], 0xdd);
    v[12] = _mm512_shuffle_i32x4(v1[4], v1[12], 0xdd);
    v[13] = _mm512_shuffle_i32x4(v1[5], v1[13], 0xdd);
    v[14] = _mm512_shuffle_i32x4(v1[6], v1[14], 0xdd);
    v[15] = _mm512_shuffle_i32x4(v1[7], v1[15], 0xdd);
}

ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_q8_K_vnni(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * RESTRICT x, ಶೂನ್ಯ * RESTRICT vy, int64_t k) {
    ಪ್ರತಿಜ್ಞಾ(k % QK_K == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ KB = k / QK_K;
    constexpr ಪೂರ್ಣಾಂಕ kVecs = QK_K / 16;

    ಖಂಡ_q8_K * y = reinterpret_cast<ಖಂಡ_q8_K *>(vy);

    // hold 16 float vecs from x
    __m512  v[kVecs];

    // hold the quants vecs
    __m512i vq[kVecs / 4];

    // hold the packed quants vecs
    __m512i vq_packed[kVecs / 4];

    ಸ್ಥಿರಾಂಕ __m512 sign_ಬಿಟ್ = _mm512_set1_ps(-0.f);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < KB; ++i) {
        // Compute max(abs(e)) for the block
        __m512 vamax = _mm512_set1_ps(0.f);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < kVecs; ++j) {
            v[j] = _mm512_loadu_ps(x); x += 16;
            vamax = _mm512_max_ps(vamax, _mm512_andnot_ps(sign_ಬಿಟ್, v[j]));
        }
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ amax = _mm512_reduce_max_ps(vamax);

        // Quantize these floats
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ iscale = 127.f / amax;
        y[i].d = GGML_CPU_FP32_ಗಮ್ಯ_FP16(1 / iscale);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ id = ( amax != 0.0f ) ? iscale : 0.f;
        ಸ್ಥಿರಾಂಕ __m512 vscale = _mm512_set1_ps(id);

        // Apply multiplier and round to nearest integer
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < kVecs; ++j) {
            v[j] = _mm512_mul_ps(v[j], vscale);
            v[j] = _mm512_roundscale_ps(v[j], (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        }

        // Pack to epi8 vecs
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < kVecs / 4; ++j) {
            __m128i q8_0 = _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(v[j * 4 + 0]));
            __m128i q8_1 = _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(v[j * 4 + 1]));
            __m128i q8_2 = _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(v[j * 4 + 2]));
            __m128i q8_3 = _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(v[j * 4 + 3]));

            __m256i q8_01 = _mm256_insertf128_si256(_mm256_castsi128_si256(q8_0), (q8_1), 1);
            __m256i q8_23 = _mm256_insertf128_si256(_mm256_castsi128_si256(q8_2), (q8_3), 1);

            vq[j] = _mm512_inserti32x8(_mm512_castsi256_si512(q8_01), q8_23, 1);
            _mm512_storeu_si512((__m512i *)(y[i].qs + j * 64), vq[j]);
        }

        // Compute the bsums with vnni
        transpose_16x4_32bit(vq, vq_packed);

        ಸ್ಥಿರಾಂಕ __m512i one = _mm512_set1_epi8(1);
        __m512i ಮೊತ್ತ = _mm512_setzero_si512();
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 4; ++k) {
            ಮೊತ್ತ = _mm512_dpbusd_epi32(ಮೊತ್ತ, one, vq_packed[k]);
        }
        _mm256_storeu_si256((__m256i *)(y[i].bsums), _mm512_cvtepi32_epi16(ಮೊತ್ತ));
    }
}

// quantize A from float to `vec_dot_type`
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
ಸಾಲೊಳಗೆ ಶೂನ್ಯ from_float(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x, ಅಕ್ಷರ * vy, int64_t k);

ಮಾದರಿ <>
ಸಾಲೊಳಗೆ ಶೂನ್ಯ from_float<ಖಂಡ_q8_0>(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x, ಅಕ್ಷರ * vy, int64_t k) {
    quantize_ಅಡ್ಡಸಾಲು_q8_0(x, (ಖಂಡ_q8_0 *)vy, k);
}

ಮಾದರಿ <>
ಸಾಲೊಳಗೆ ಶೂನ್ಯ from_float<ಖಂಡ_q8_1>(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x, ಅಕ್ಷರ * vy, int64_t k) {
    quantize_ಅಡ್ಡಸಾಲು_q8_1(x, (ಖಂಡ_q8_1 *)vy, k);
}

ಮಾದರಿ <>
ಸಾಲೊಳಗೆ ಶೂನ್ಯ from_float<ಖಂಡ_q8_K>(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x, ಅಕ್ಷರ * vy, int64_t k) {
#ಆದರೆ 1
    // TODO: this is reference impl!
    quantize_ಅಡ್ಡಸಾಲು_q8_K_ref(x, (ಖಂಡ_q8_K *)vy, k);
#ಇಲ್ಲದಿದ್ದರೆ
    quantize_ಅಡ್ಡಸಾಲು_q8_K_vnni(x, vy, k);
#ಆದರೆ_ಅಂತ್ಯ
}

// load A from memory to array when nrows can not fill in whole tile
ಶೂನ್ಯ unpack_A(int8_t * RESTRICT tile, ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * RESTRICT A, ಪೂರ್ಣಾಂಕ lda, ಪೂರ್ಣಾಂಕ nr) {
    ಪ್ರತಿಜ್ಞಾ(nr != TILE_M);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ m = 0; m < nr; ++m) {
        ಸ್ಥಿರಾಂಕ __m256i v = _mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)(A[m * lda].qs));
        _mm256_storeu_si256((__m256i *)(tile + m * TILE_K), v);
    }
}

ಶೂನ್ಯ unpack_A(int8_t * RESTRICT tile, ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 * RESTRICT A, ಪೂರ್ಣಾಂಕ lda, ಪೂರ್ಣಾಂಕ nr) {
    ಪ್ರತಿಜ್ಞಾ(nr != TILE_M);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ m = 0; m < nr; ++m) {
        ಸ್ಥಿರಾಂಕ __m256i v = _mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)(A[m * lda].qs));
        _mm256_storeu_si256((__m256i *)(tile + m * TILE_K), v);
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು TB>
ಶೂನ್ಯ unpack_A(int8_t * RESTRICT tile, ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K * RESTRICT A, ಪೂರ್ಣಾಂಕ lda, ಪೂರ್ಣಾಂಕ k, ಪೂರ್ಣಾಂಕ nr) {
    ಪ್ರತಿಜ್ಞಾ(nr <= TILE_M);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ m = 0; m < nr; ++m) {
        ಸ್ಥಿರಾಂಕ __m256i v = _mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)(A[m * lda].qs + k * 32));
        _mm256_storeu_si256((__m256i *)(tile + m * TILE_K), v);
    }
}

ಮಾದರಿ <>
ಶೂನ್ಯ unpack_A<ಖಂಡ_q6_K>(int8_t * RESTRICT tile, ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K * RESTRICT A, ಪೂರ್ಣಾಂಕ lda, ಪೂರ್ಣಾಂಕ k, ಪೂರ್ಣಾಂಕ nr) {
    ಪ್ರತಿಜ್ಞಾ(nr <= TILE_M);
    // zero padding k from 16 to 32, so that we don't have to re-config amx
    ಸ್ಥಿರಾಂಕ __m128i zero = _mm_setzero_si128();
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ m = 0; m < nr; ++m) {
        ಸ್ಥಿರಾಂಕ __m128i v = _mm_loadu_si128((ಸ್ಥಿರಾಂಕ __m128i *)(A[m * lda].qs + k * 16));
        ಸ್ಥಿರಾಂಕ __m256i r = _mm256_insertf128_si256(_mm256_castsi128_si256(v), zero, 1);
        _mm256_storeu_si256((__m256i *)(tile + m * TILE_K), r);
    }
}

#ವ್ಯಾಖ್ಯಾನಿಸು MM256_ಗಣ_M128_I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)
ಸಾಲೊಳಗೆ __m256i ಬೈಟುಗಳು_from_nibbles_32(ಸ್ಥಿರಾಂಕ uint8_t * rsi) {
    ಸ್ಥಿರಾಂಕ __m128i tmp = _mm_loadu_si128((ಸ್ಥಿರಾಂಕ __m128i *)rsi);
    ಸ್ಥಿರಾಂಕ __m256i ಬೈಟುಗಳು = MM256_ಗಣ_M128_I(_mm_srli_epi16(tmp, 4), tmp);
    ಸ್ಥಿರಾಂಕ __m256i ಕೆಳಗಿನ_ಮುಖವಾಡ = _mm256_set1_epi8(0xF);
    ಹಿಂತಿರುಗಿ _mm256_and_si256(ಕೆಳಗಿನ_ಮುಖವಾಡ, ಬೈಟುಗಳು);
}

// used for block_q4_K
ಸಾಲೊಳಗೆ __m512i ಬೈಟುಗಳು_from_nibbles_64(ಸ್ಥಿರಾಂಕ uint8_t * rsi) {
    ಸ್ಥಿರಾಂಕ __m256i tmp = _mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)rsi);
    ಸ್ಥಿರಾಂಕ __m256i ಕೆಳಗಿನ_ಮುಖವಾಡ = _mm256_set1_epi8(0xF);
    ಸ್ಥಿರಾಂಕ __m256i q4l = _mm256_and_si256(tmp, ಕೆಳಗಿನ_ಮುಖವಾಡ);
    ಸ್ಥಿರಾಂಕ __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(tmp, 4), ಕೆಳಗಿನ_ಮುಖವಾಡ);
    ಹಿಂತಿರುಗಿ _mm512_inserti32x8(_mm512_castsi256_si512(q4l), q4h, 1);
}

// used for block_q5_K
ಸಾಲೊಳಗೆ __m512i ಬೈಟುಗಳು_from_nibbles_64(ಸ್ಥಿರಾಂಕ uint8_t * qs, ಸ್ಥಿರಾಂಕ uint8_t * qh, ಪೂರ್ಣಾಂಕ k) {
    ಸ್ಥಿರಾಂಕ __m256i ಕೆಳಗಿನ_ಮುಖವಾಡ = _mm256_set1_epi8(0xF);
    __m256i hmask = _mm256_set1_epi8(1);
    hmask = _mm256_slli_epi16(hmask, k);

    ಸ್ಥಿರಾಂಕ __m256i q5bits = _mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)qs);
    ಸ್ಥಿರಾಂಕ __m256i hbits = _mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)qh);

    ಸ್ಥಿರಾಂಕ __m256i q5l_0 = _mm256_and_si256(q5bits, ಕೆಳಗಿನ_ಮುಖವಾಡ);
    ಸ್ಥಿರಾಂಕ __m256i q5h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), k + 0), 4);
    ಸ್ಥಿರಾಂಕ __m256i q5_0  = mm256_ಕೂಡಿಸು_epi8(q5l_0, q5h_0);
    hmask = _mm256_slli_epi16(hmask, 1);

    ಸ್ಥಿರಾಂಕ __m256i q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), ಕೆಳಗಿನ_ಮುಖವಾಡ);
    ಸ್ಥಿರಾಂಕ __m256i q5h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), k + 1), 4);
    ಸ್ಥಿರಾಂಕ __m256i q5_1  = mm256_ಕೂಡಿಸು_epi8(q5l_1, q5h_1);

    ಹಿಂತಿರುಗಿ _mm512_inserti32x8(_mm512_castsi256_si512(q5_0), q5_1, 1);
}

// used for block_q6_K
ಸಾಲೊಳಗೆ ಶೂನ್ಯ ಬೈಟುಗಳು_from_nibbles_128(__m512i& r0, __m512i& r1, ಸ್ಥಿರಾಂಕ uint8_t * qs, ಸ್ಥಿರಾಂಕ uint8_t * qh) {
    ಸ್ಥಿರಾಂಕ __m256i m4 = _mm256_set1_epi8(0xF);
    ಸ್ಥಿರಾಂಕ __m256i m2 = _mm256_set1_epi8(0x3);

    ಸ್ಥಿರಾಂಕ __m256i q6bits1 = _mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)qs);
    ಸ್ಥಿರಾಂಕ __m256i q6bits2 = _mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)(qs + 32));
    ಸ್ಥಿರಾಂಕ __m256i q6bitsH = _mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)qh);

    ಸ್ಥಿರಾಂಕ __m256i q6h_0 = _mm256_slli_epi16(_mm256_and_si256(                  q6bitsH,     m2), 4);
    ಸ್ಥಿರಾಂಕ __m256i q6h_1 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q6bitsH, 2), m2), 4);
    ಸ್ಥಿರಾಂಕ __m256i q6h_2 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q6bitsH, 4), m2), 4);
    ಸ್ಥಿರಾಂಕ __m256i q6h_3 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q6bitsH, 6), m2), 4);

    ಸ್ಥಿರಾಂಕ __m256i q6_0 = _mm256_or_si256(_mm256_and_si256(q6bits1, m4), q6h_0);
    ಸ್ಥಿರಾಂಕ __m256i q6_1 = _mm256_or_si256(_mm256_and_si256(q6bits2, m4), q6h_1);
    ಸ್ಥಿರಾಂಕ __m256i q6_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q6bits1, 4), m4), q6h_2);
    ಸ್ಥಿರಾಂಕ __m256i q6_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q6bits2, 4), m4), q6h_3);

    r0 = _mm512_inserti32x8(_mm512_castsi256_si512(q6_0), q6_1, 1);
    r1 = _mm512_inserti32x8(_mm512_castsi256_si512(q6_2), q6_3, 1);
}

ಸಾಲೊಳಗೆ __m512i packNibbles(__m512i r0, __m512i r1) {
    ಹಿಂತಿರುಗಿ _mm512_or_si512(r0, _mm512_slli_epi16(r1, 4));
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು TB>
ಸಾಲೊಳಗೆ ಶೂನ್ಯ pack_qs(ಶೂನ್ಯ * RESTRICT packed_B, ಸ್ಥಿರಾಂಕ TB * RESTRICT B, ಪೂರ್ಣಾಂಕ KB) {
    int8_t tmp[8 * 64];
    __m256i v[8], v2[8];
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ n = 0; n < 8; ++n) {
        v[n] = ಬೈಟುಗಳು_from_nibbles_32(B[n * KB].qs);
    }
    transpose_8x8_32bit(v, v2);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ n = 0; n < 8; ++n) {
        _mm256_storeu_si256((__m256i *)(tmp + n * 64), v2[n]);
    }
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ n = 0; n < 8; ++n) {
        v[n] = ಬೈಟುಗಳು_from_nibbles_32(B[(n + 8) * KB].qs);
    }
    transpose_8x8_32bit(v, v2);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ n = 0; n < 8; ++n) {
        _mm256_storeu_si256((__m256i *)(tmp + n * 64 + 32), v2[n]);
    }

    // pack again with 128 to fully utilize vector length
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ n = 0; n < 8; n += 2) {
        __m512i r0 = _mm512_loadu_si512((ಸ್ಥಿರಾಂಕ __m512i *)(tmp + n * 64));
        __m512i r1 = _mm512_loadu_si512((ಸ್ಥಿರಾಂಕ __m512i *)(tmp + n * 64 + 64));
        __m512i r1r0 = packNibbles(r0, r1);
        _mm512_storeu_si512((__m512i *)((ಅಕ್ಷರ *)packed_B + n * 32), r1r0);
    }
}

ಮಾದರಿ <>
ಸಾಲೊಳಗೆ ಶೂನ್ಯ pack_qs<ಖಂಡ_q8_0>(ಶೂನ್ಯ * RESTRICT packed_B, ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * RESTRICT B, ಪೂರ್ಣಾಂಕ KB) {
    __m256i v[8], v2[8];
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ n = 0; n < 8; ++n) {
        v[n] = _mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)(B[n * KB].qs));
    }
    transpose_8x8_32bit(v, v2);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ n = 0; n < 8; ++n) {
        _mm256_storeu_si256((__m256i *)((ಅಕ್ಷರ *)packed_B + n * 64), v2[n]);
    }
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ n = 0; n < 8; ++n) {
        v[n] = _mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)(B[(n + 8) * KB].qs));
    }
    transpose_8x8_32bit(v, v2);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ n = 0; n < 8; ++n) {
        _mm256_storeu_si256((__m256i *)((ಅಕ್ಷರ *)packed_B + n * 64 + 32), v2[n]);
    }
}

ಮಾದರಿ <>
ಸಾಲೊಳಗೆ ಶೂನ್ಯ pack_qs<ಖಂಡ_q4_K>(ಶೂನ್ಯ * RESTRICT packed_B, ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_K * RESTRICT B, ಪೂರ್ಣಾಂಕ KB) {
    __m512i v[16];
    // QK_K 256 with 8 groups, handle 2 groups at a time
    ಅಕ್ಷರ * pb = (ಅಕ್ಷರ *)packed_B;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < QK_K / 64; ++k) {
        // pack 2 groups { n, g,  k} to {g, k/4, 4n}
        //          e.g. {16, 2, 32} to {2,   8, 64}
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ n = 0; n < TILE_N; ++n) {
            v[n] = ಬೈಟುಗಳು_from_nibbles_64(B[n * KB].qs + k * 32);
        }

        transpose_16x16_32bit(v);

        // pack again with 128 to fully utilize vector length
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ n = 0; n < TILE_N; n += 2) {
            _mm512_storeu_si512((__m512i *)pb, packNibbles(v[n], v[n + 1]));
            pb += 64;
        }
    }
}

ಮಾದರಿ <>
ಸಾಲೊಳಗೆ ಶೂನ್ಯ pack_qs<ಖಂಡ_q5_K>(ಶೂನ್ಯ * RESTRICT packed_B, ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_K * RESTRICT B, ಪೂರ್ಣಾಂಕ KB) {
    __m512i v[16];
    ಸ್ಥಿರಾಂಕ __m512i ಕೆಳಗಿನ_ಮುಖವಾಡ = _mm512_set1_epi8(0xF);
    // QK_K 256 with 8 groups, handle 2 groups at a time
    ಅಕ್ಷರ * pb = (ಅಕ್ಷರ *)packed_B;
    ಅಕ್ಷರ * ph = (ಅಕ್ಷರ *)packed_B + (QK_K / 2) * TILE_N;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < QK_K / 64; ++k) {
        // pack 2 groups { n, g,  k} to {g, k/4, 4n}
        //          e.g. {16, 2, 32} to {2,   8, 64}
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ n = 0; n < TILE_N; ++n) {
            v[n] = ಬೈಟುಗಳು_from_nibbles_64(B[n * KB].qs + k * 32, B[n * KB].qh, /*ಗುಂಪು*/2 * k);
        }

        transpose_16x16_32bit(v);

        // 1. pack lower 4bits with 2 groups
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ n = 0; n < TILE_N; n += 2) {
            // get lower 4 bits
            ಸ್ಥಿರಾಂಕ __m512i r0 = _mm512_and_si512(v[n], ಕೆಳಗಿನ_ಮುಖವಾಡ);
            ಸ್ಥಿರಾಂಕ __m512i r1 = _mm512_and_si512(v[n + 1], ಕೆಳಗಿನ_ಮುಖವಾಡ);
            _mm512_storeu_si512((__m512i *)pb, packNibbles(r0, r1)); pb += 64;
        }

        // 2. pack higher 1bit with 2 groups
        ಸ್ಥಿರಾಂಕ __m512i hmask = _mm512_set1_epi8(0x10);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ g = 0; g < 2; ++g) {
            __m512i hbits = _mm512_setzero_si512();
            hbits = mm512_ಕೂಡಿಸು_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 8 + 0], hmask), 4));
            hbits = mm512_ಕೂಡಿಸು_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 8 + 1], hmask), 3));
            hbits = mm512_ಕೂಡಿಸು_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 8 + 2], hmask), 2));
            hbits = mm512_ಕೂಡಿಸು_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 8 + 3], hmask), 1));
            hbits = mm512_ಕೂಡಿಸು_epi8(hbits,                   _mm512_and_si512(v[g * 8 + 4], hmask)    );
            hbits = mm512_ಕೂಡಿಸು_epi8(hbits, _mm512_slli_epi16(_mm512_and_si512(v[g * 8 + 5], hmask), 1));
            hbits = mm512_ಕೂಡಿಸು_epi8(hbits, _mm512_slli_epi16(_mm512_and_si512(v[g * 8 + 6], hmask), 2));
            hbits = mm512_ಕೂಡಿಸು_epi8(hbits, _mm512_slli_epi16(_mm512_and_si512(v[g * 8 + 7], hmask), 3));
            _mm512_storeu_si512((__m512i *)ph, hbits); ph += 64;
        }
    }
}

ಮಾದರಿ <>
ಸಾಲೊಳಗೆ ಶೂನ್ಯ pack_qs<ಖಂಡ_q6_K>(ಶೂನ್ಯ * RESTRICT packed_B, ಸ್ಥಿರಾಂಕ ಖಂಡ_q6_K * RESTRICT B, ಪೂರ್ಣಾಂಕ KB) {
    __m512i v[32];
    ಸ್ಥಿರಾಂಕ __m512i ಕೆಳಗಿನ_ಮುಖವಾಡ = _mm512_set1_epi8(0xF);
    // QK_K 256 with 8 groups, handle 4 groups at a time
    ಅಕ್ಷರ * pb = (ಅಕ್ಷರ *)packed_B;
    ಅಕ್ಷರ * ph = (ಅಕ್ಷರ *)packed_B + (QK_K / 2) * TILE_N;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < QK_K / 128; ++k) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ n = 0; n < TILE_N; ++n) {
            ಬೈಟುಗಳು_from_nibbles_128(v[n], v[n + 16], B[n * KB].ql + k * 64, B[n * KB].qh + k * 32);
        }

        // top half: group 0,1 or 4,5; bottom half: group 2,3 or 6,7
        transpose_16x16_32bit(v);
        transpose_16x16_32bit(v + 16);

        // 1. pack lower 4bits with 4 groups
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ n = 0; n < 32; n += 2) {
            ಸ್ಥಿರಾಂಕ __m512i r0 = _mm512_and_si512(v[n], ಕೆಳಗಿನ_ಮುಖವಾಡ);
            ಸ್ಥಿರಾಂಕ __m512i r1 = _mm512_and_si512(v[n + 1], ಕೆಳಗಿನ_ಮುಖವಾಡ);
            _mm512_storeu_si512((__m512i *)pb, packNibbles(r0, r1)); pb += 64;
        }

        // 2. pack higher 2bit with 4 groups
        ಸ್ಥಿರಾಂಕ __m512i hmask = _mm512_set1_epi8(0x30);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ g = 0; g < 8; ++g) {
            __m512i hbits = _mm512_setzero_si512();
            hbits = mm512_ಕೂಡಿಸು_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 4 + 0], hmask), 4));
            hbits = mm512_ಕೂಡಿಸು_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 4 + 1], hmask), 2));
            hbits = mm512_ಕೂಡಿಸು_epi8(hbits,                   _mm512_and_si512(v[g * 4 + 2], hmask)    );
            hbits = mm512_ಕೂಡಿಸು_epi8(hbits, _mm512_slli_epi16(_mm512_and_si512(v[g * 4 + 3], hmask), 2));
            _mm512_storeu_si512((__m512i *)ph, hbits); ph += 64;
        }
    }
}

ಮಾದರಿ <>
ಸಾಲೊಳಗೆ ಶೂನ್ಯ pack_qs<ಖಂಡ_iq4_xs>(ಶೂನ್ಯ * RESTRICT packed_B, ಸ್ಥಿರಾಂಕ ಖಂಡ_iq4_xs * RESTRICT B, ಪೂರ್ಣಾಂಕ KB) {
    __m512i v[16];
    ಅಕ್ಷರ * pb = (ಅಕ್ಷರ *)packed_B;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < QK_K / 64; ++k) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ n = 0; n < TILE_N; ++n) {
            __m256i r0 = ಬೈಟುಗಳು_from_nibbles_32(B[n * KB].qs + k * 32 +  0);
            __m256i r1 = ಬೈಟುಗಳು_from_nibbles_32(B[n * KB].qs + k * 32 + 16);
            v[n] = _mm512_inserti32x8(_mm512_castsi256_si512(r0), r1, 1);
        }

        transpose_16x16_32bit(v);

        // pack again with 128 to fully utilize vector length
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ n = 0; n < TILE_N; n += 2) {
            _mm512_storeu_si512((__m512i *)pb, packNibbles(v[n], v[n + 1]));
            pb += 64;
        }
    }
}

// pack B to vnni formats in 4bits or 8 bits
ಶೂನ್ಯ pack_B(ಶೂನ್ಯ * RESTRICT packed_B, ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 * RESTRICT B, ಪೂರ್ಣಾಂಕ KB) {
    pack_qs(packed_B, B, KB);
    ggml_half * d0 = reinterpret_cast<ggml_half *>((ಅಕ್ಷರ *)packed_B + TILE_N * TILE_K / 2);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ n = 0; n < TILE_N; ++n) {
        d0[n] = B[n * KB].d;
    }
}

ಶೂನ್ಯ pack_B(ಶೂನ್ಯ * RESTRICT packed_B, ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_1 * RESTRICT B, ಪೂರ್ಣಾಂಕ KB) {
    pack_qs(packed_B, B, KB);
    ggml_half * d0 = reinterpret_cast<ggml_half *>((ಅಕ್ಷರ *)packed_B + TILE_N * TILE_K / 2);
    ggml_half * m0 = d0 + TILE_N;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ n = 0; n < TILE_N; ++n) {
        d0[n] = B[n * KB].d;
        m0[n] = B[n * KB].m;
    }
}

ಸಾಲೊಳಗೆ ಶೂನ್ಯ s8s8_compensation(ಶೂನ್ಯ * RESTRICT packed_B) {
    // packed_B layout:
    //   quants {TILE_N, TILEK}  int8_t
    //   d0     {TILE_N}      ggml_half
    //   comp   {TILE_N}        int32_t
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ = TILE_N * TILE_K + TILE_N * ಗಾತ್ರ(ggml_half);
    __m512i vcomp = _mm512_setzero_si512();
    ಸ್ಥಿರಾಂಕ __m512i off = _mm512_set1_epi8(static_cast<ಅಕ್ಷರ>(0x80));
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 8; ++k) {
        __m512i vb = _mm512_loadu_si512((ಸ್ಥಿರಾಂಕ __m512i *)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)packed_B + k * 64));
        vcomp = _mm512_dpbusd_epi32(vcomp, off, vb);
    }
    _mm512_storeu_si512((__m512i *)((ಅಕ್ಷರ *)(packed_B) + ಸ್ಥಾನಾಂತರ), vcomp);
}

ಶೂನ್ಯ pack_B(ಶೂನ್ಯ * RESTRICT packed_B, ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * RESTRICT B, ಪೂರ್ಣಾಂಕ KB) {
    pack_qs(packed_B, B, KB);
    ggml_half * d0 = reinterpret_cast<ggml_half *>((ಅಕ್ಷರ *)packed_B + TILE_N * TILE_K);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ n = 0; n < TILE_N; ++n) {
        d0[n] = B[n * KB].d;
    }
    s8s8_compensation(packed_B);
}

// convert 8 * {min, scale} from int6 to int8
ಸಾಲೊಳಗೆ ಶೂನ್ಯ unpack_mins_and_scales(ಸ್ಥಿರಾಂಕ uint8_t * scales, uint32_t * utmp) {
    ಸ್ಥಿರಾಂಕ uint32_t kmask1 = 0x3f3f3f3f;
    ಸ್ಥಿರಾಂಕ uint32_t kmask2 = 0x0f0f0f0f;
    ಸ್ಥಿರಾಂಕ uint32_t kmask3 = 0x03030303;

    ಸ್ಮರಣೆ_ನಕಲು(utmp, scales, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    ಸ್ಥಿರಾಂಕ uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
}

// packed_B layout:
//   quants {8, TILE_N, 16}  uint8
//   scales {8, TILE_N}      uint8
//   mins   {8, TILE_N}      uint8
//   d      {TILE_N}     ggml_half
//   dmin   {TILE_N}     ggml_half
ಶೂನ್ಯ pack_B(ಶೂನ್ಯ * RESTRICT packed_B, ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_K * RESTRICT B, ಪೂರ್ಣಾಂಕ KB) {
    pack_qs(packed_B, B, KB);

    uint8_t * scales = reinterpret_cast<uint8_t *>((ಅಕ್ಷರ *)packed_B + (QK_K / 2) * TILE_N);
    uint8_t * mins = scales + 8 * TILE_N;
    ggml_half * d = reinterpret_cast<ggml_half *>(mins + 8 * TILE_N);
    ggml_half * dmin = d + TILE_N;

    ಒಕ್ಕೂಟ {
        uint32_t u32[4];
        uint8_t  u8[16];
    } s;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ n = 0; n < TILE_N; ++n) {
        unpack_mins_and_scales(B[n * KB].scales, s.u32);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 8; ++k) {
            scales[k * TILE_N + n] = s.u8[k];
            mins[(k >> 1) * TILE_N * 2 + n * 2 + (k & 0x1)] = s.u8[k + 8];
        }
        d[n] = B[n * KB].d;
        dmin[n] = B[n * KB].dmin;
    }
}

// packed_B layout:
//   quants {8, TILE_N, 16}  uint8
//   qh     {8, TILE_N,  4}  uint8
//   scales {8, TILE_N}      uint8
//   mins   {8, TILE_N}      uint8
//   d      {TILE_N}     ggml_half
//   dmin   {TILE_N}     ggml_half
ಶೂನ್ಯ pack_B(ಶೂನ್ಯ * RESTRICT packed_B, ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_K * RESTRICT B, ಪೂರ್ಣಾಂಕ KB) {
    pack_qs(packed_B, B, KB);

    uint8_t * scales = reinterpret_cast<uint8_t *>((ಅಕ್ಷರ *)packed_B + (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N);
    uint8_t * mins = scales + 8 * TILE_N;
    ggml_half * d = reinterpret_cast<ggml_half *>(mins + 8 * TILE_N);
    ggml_half * dmin = d + TILE_N;

    ಒಕ್ಕೂಟ {
        uint32_t u32[4];
        uint8_t  u8[16];
    } s;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ n = 0; n < TILE_N; ++n) {
        unpack_mins_and_scales(B[n * KB].scales, s.u32);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 8; ++k) {
            scales[k * TILE_N + n] = s.u8[k];
            mins[(k >> 1) * TILE_N * 2 + n * 2 + (k & 0x1)] = s.u8[k + 8];
        }
        d[n] = B[n * KB].d;
        dmin[n] = B[n * KB].dmin;
    }
}

// packed_B layout:
//   quants {16, TILE_N, 8}  uint8
//   qh     {16, TILE_N, 4}  uint8
//   scales {16, TILE_N}      uint8
//   d      {TILE_N}     ggml_half
ಶೂನ್ಯ pack_B(ಶೂನ್ಯ * RESTRICT packed_B, ಸ್ಥಿರಾಂಕ ಖಂಡ_q6_K * RESTRICT B, ಪೂರ್ಣಾಂಕ KB) {
    pack_qs(packed_B, B, KB);

    uint8_t * scales = reinterpret_cast<uint8_t *>((ಅಕ್ಷರ *)packed_B + (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N);
    ggml_half * d = reinterpret_cast<ggml_half *>(scales + 16 * TILE_N);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ n = 0; n < TILE_N; ++n) {
        ಸ್ಥಿರಾಂಕ int8_t * ps = B[n * KB].scales;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 16; ++k) {
            scales[k * TILE_N + n] = ps[k];
        }
        d[n] = B[n * KB].d;
    }
}

// packed_B layout:
//   quants {8, TILE_N, 16}  uint8
//   scales {8, TILE_N}       int8
//   d      {TILE_N}     ggml_half
ಶೂನ್ಯ pack_B(ಶೂನ್ಯ * RESTRICT packed_B, ಸ್ಥಿರಾಂಕ ಖಂಡ_iq4_xs * RESTRICT B, ಪೂರ್ಣಾಂಕ KB) {
    pack_qs(packed_B, B, KB);

    int8_t * scales = reinterpret_cast<int8_t *>((ಅಕ್ಷರ *)packed_B + (QK_K / 2) * TILE_N);
    ggml_half * d = reinterpret_cast<ggml_half *>(scales + 8 * TILE_N);

    // pack the scales
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ n = 0; n < TILE_N; ++n) {
        uint16_t sh = B[n * KB].scales_h;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 8; k += 2) {
            ಸ್ಥಿರಾಂಕ int16_t ls1 = ((B[n * KB].scales_l[k / 2] & 0xf) | ((sh << 4) & 0x30)) - 32;
            ಸ್ಥಿರಾಂಕ int16_t ls2 = ((B[n * KB].scales_l[k / 2] >>  4) | ((sh << 2) & 0x30)) - 32;
            scales[(k + 0) * TILE_N + n] = ls1;
            scales[(k + 1) * TILE_N + n] = ls2;
            sh >>= 4;
        }
        d[n] = B[n * KB].d;
    }
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು TB, ಬಗೆ_ಹೆಸರು packed_B_t = packed_B_type<TB>>
ಶೂನ್ಯ unpack_B(packed_B_t * RESTRICT tile, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * RESTRICT packed_B) {
    GGML_ಬಳಸದ(tile);
    GGML_ಬಳಸದ(packed_B);
}

ಮಾದರಿ <>
ಶೂನ್ಯ unpack_B<ಖಂಡ_q4_0>(int8_t * RESTRICT tile, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * RESTRICT packed_B) {
  ಸ್ಥಿರಾಂಕ __m512i off = _mm512_set1_epi8(8);
  ಸ್ಥಿರಾಂಕ __m512i ಕೆಳಗಿನ_ಮುಖವಾಡ = _mm512_set1_epi8(0xF);
  ಗಾಗಿ (ಪೂರ್ಣಾಂಕ n = 0; n < 8; n += 2) {
    __m512i ಬೈಟುಗಳು = _mm512_loadu_si512((ಸ್ಥಿರಾಂಕ __m512i *)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)packed_B + n * 32));
    ಸ್ಥಿರಾಂಕ __m512i r0 = _mm512_sub_epi8(_mm512_and_si512(ಬೈಟುಗಳು, ಕೆಳಗಿನ_ಮುಖವಾಡ), off);
    ಸ್ಥಿರಾಂಕ __m512i r1 = _mm512_sub_epi8(_mm512_and_si512(_mm512_srli_epi16(ಬೈಟುಗಳು, 4), ಕೆಳಗಿನ_ಮುಖವಾಡ), off);
    _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
    _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
  }
}

ಮಾದರಿ <>
ಶೂನ್ಯ unpack_B<ಖಂಡ_q4_1>(uint8_t * RESTRICT tile, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * RESTRICT packed_B) {
    ಸ್ಥಿರಾಂಕ __m512i ಕೆಳಗಿನ_ಮುಖವಾಡ = _mm512_set1_epi8(0xF);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ n = 0; n < 8; n += 2) {
        __m512i ಬೈಟುಗಳು = _mm512_loadu_si512((ಸ್ಥಿರಾಂಕ __m512i *)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)packed_B + n * 32));
        ಸ್ಥಿರಾಂಕ __m512i r0 = _mm512_and_si512(ಬೈಟುಗಳು, ಕೆಳಗಿನ_ಮುಖವಾಡ);
        ಸ್ಥಿರಾಂಕ __m512i r1 = _mm512_and_si512(_mm512_srli_epi16(ಬೈಟುಗಳು, 4), ಕೆಳಗಿನ_ಮುಖವಾಡ);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
    }
}

// packed_B_t for QKK is int8_t
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು TB>
ಶೂನ್ಯ unpack_B(int8_t * RESTRICT tile, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * RESTRICT packed_B, ಪೂರ್ಣಾಂಕ k) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ packed_B_ಗುಂಪು_ಗಾತ್ರ = QK_K / 2 * TILE_N / 8;
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * packed_B_ಗುಂಪು = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)packed_B + k * packed_B_ಗುಂಪು_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ __m512i ಕೆಳಗಿನ_ಮುಖವಾಡ = _mm512_set1_epi8(0xF);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ n = 0; n < 8; n += 2) {
        __m512i ಬೈಟುಗಳು = _mm512_loadu_si512(packed_B_ಗುಂಪು + n * 32);
        ಸ್ಥಿರಾಂಕ __m512i r0 = _mm512_and_si512(ಬೈಟುಗಳು, ಕೆಳಗಿನ_ಮುಖವಾಡ);
        ಸ್ಥಿರಾಂಕ __m512i r1 = _mm512_and_si512(_mm512_srli_epi16(ಬೈಟುಗಳು, 4), ಕೆಳಗಿನ_ಮುಖವಾಡ);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
    }
}

ಮಾದರಿ <>
ಶೂನ್ಯ unpack_B<ಖಂಡ_q5_K>(int8_t * RESTRICT tile, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * RESTRICT packed_B, ಪೂರ್ಣಾಂಕ k) {
    // lower 4bits, stride 256 bytes
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ packed_l4_ಗುಂಪು_ಗಾತ್ರ = QK_K / 2 * TILE_N / 8;
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * pb = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)packed_B + k * packed_l4_ಗುಂಪು_ಗಾತ್ರ;

    // higher 1bit, stride 64 bytes
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ packed_h1_ಗುಂಪು_ಗಾತ್ರ = QK_K / 8 * TILE_N / 8;
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ph = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)packed_B + (QK_K / 2) * TILE_N + k * packed_h1_ಗುಂಪು_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ __m512i hbits = _mm512_loadu_si512(ph);

    ಸ್ಥಿರಾಂಕ __m512i ಕೆಳಗಿನ_ಮುಖವಾಡ = _mm512_set1_epi8(0xF);
    __m512i hmask0 = _mm512_set1_epi8(0x1);
    __m512i hmask1 = _mm512_set1_epi8(0x2);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ n = 0; n < 8; n += 2) {
        __m512i ಬೈಟುಗಳು = _mm512_loadu_si512(pb + n * 32);
        __m512i r0 = _mm512_and_si512(ಬೈಟುಗಳು, ಕೆಳಗಿನ_ಮುಖವಾಡ);
        __m512i r1 = _mm512_and_si512(_mm512_srli_epi16(ಬೈಟುಗಳು, 4), ಕೆಳಗಿನ_ಮುಖವಾಡ);
        __m512i h0 = _mm512_slli_epi16(_mm512_srli_epi16(_mm512_and_si512(hbits, hmask0), n), 4);
        __m512i h1 = _mm512_slli_epi16(_mm512_srli_epi16(_mm512_and_si512(hbits, hmask1), n + 1), 4);

        hmask0 = _mm512_slli_epi16(hmask0, 2);
        hmask1 = _mm512_slli_epi16(hmask1, 2);
        r0 = mm512_ಕೂಡಿಸು_epi8(r0, h0);
        r1 = mm512_ಕೂಡಿಸು_epi8(r1, h1);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
    }
}

ಮಾದರಿ <>
ಶೂನ್ಯ unpack_B<ಖಂಡ_q6_K>(int8_t * RESTRICT tile, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * RESTRICT packed_B, ಪೂರ್ಣಾಂಕ k) {
    // lower 4bits, stride 128 bytes
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ packed_l4_ಗುಂಪು_ಗಾತ್ರ = QK_K / 2 * TILE_N / 16;
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * pb = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)packed_B + k * packed_l4_ಗುಂಪು_ಗಾತ್ರ;

    // higher 2bits, stride 64 bytes
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ packed_h2_ಗುಂಪು_ಗಾತ್ರ = QK_K / 4 * TILE_N / 16;
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ph = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)packed_B + (QK_K / 2) * TILE_N + k * packed_h2_ಗುಂಪು_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ __m512i hbits = _mm512_loadu_si512(ph);

    ಸ್ಥಿರಾಂಕ __m512i off = _mm512_set1_epi8(32);
    ಸ್ಥಿರಾಂಕ __m512i ಕೆಳಗಿನ_ಮುಖವಾಡ = _mm512_set1_epi8(0xF);
    __m512i hmask0 = _mm512_set1_epi8(0x3); // 0011
    __m512i hmask1 = _mm512_set1_epi8(0xC); // 1100

    // notes: skip zero padding from row4 to row7 as we have done so in `unpack_A`
    __m512i ಬೈಟುಗಳು = _mm512_loadu_si512(pb);
    __m512i r0 = _mm512_and_si512(ಬೈಟುಗಳು, ಕೆಳಗಿನ_ಮುಖವಾಡ);
    __m512i r1 = _mm512_and_si512(_mm512_srli_epi16(ಬೈಟುಗಳು, 4), ಕೆಳಗಿನ_ಮುಖವಾಡ);
    __m512i h0 = _mm512_slli_epi16(_mm512_and_si512(hbits, hmask0), 4);
    __m512i h1 = _mm512_slli_epi16(_mm512_and_si512(hbits, hmask1), 2);
    _mm512_storeu_si512((__m512i *)(tile +  0), _mm512_sub_epi8(mm512_ಕೂಡಿಸು_epi8(r0, h0), off));
    _mm512_storeu_si512((__m512i *)(tile + 64), _mm512_sub_epi8(mm512_ಕೂಡಿಸು_epi8(r1, h1), off));

    hmask0 = _mm512_slli_epi16(hmask0, 4);
    hmask1 = _mm512_slli_epi16(hmask1, 4);

    ಬೈಟುಗಳು = _mm512_loadu_si512(pb + 64);
    r0 = _mm512_and_si512(ಬೈಟುಗಳು, ಕೆಳಗಿನ_ಮುಖವಾಡ);
    r1 = _mm512_and_si512(_mm512_srli_epi16(ಬೈಟುಗಳು, 4), ಕೆಳಗಿನ_ಮುಖವಾಡ);
    h0 =                   _mm512_and_si512(hbits, hmask0);
    h1 = _mm512_srli_epi16(_mm512_and_si512(hbits, hmask1), 2);
    _mm512_storeu_si512((__m512i *)(tile + 128), _mm512_sub_epi8(mm512_ಕೂಡಿಸು_epi8(r0, h0), off));
    _mm512_storeu_si512((__m512i *)(tile + 192), _mm512_sub_epi8(mm512_ಕೂಡಿಸು_epi8(r1, h1), off));
}

ಮಾದರಿ <>
ಶೂನ್ಯ unpack_B<ಖಂಡ_iq4_xs>(int8_t * RESTRICT tile, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * RESTRICT packed_B, ಪೂರ್ಣಾಂಕ k) {
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ __m512i values128 = mm512_ಗಣ_epi8(
        113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
        113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
        113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
        113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127
    );

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ packed_B_ಗುಂಪು_ಗಾತ್ರ = QK_K / 2 * TILE_N / 8;
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * pb = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)packed_B + k * packed_B_ಗುಂಪು_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ __m512i ಕೆಳಗಿನ_ಮುಖವಾಡ = _mm512_set1_epi8(0xF);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ n = 0; n < 8; n += 2) {
        __m512i ಬೈಟುಗಳು = _mm512_loadu_si512(pb + n * 32);
        ಸ್ಥಿರಾಂಕ __m512i r0 = _mm512_shuffle_epi8(values128, _mm512_and_si512(ಬೈಟುಗಳು, ಕೆಳಗಿನ_ಮುಖವಾಡ));
        ಸ್ಥಿರಾಂಕ __m512i r1 = _mm512_shuffle_epi8(values128, _mm512_and_si512(_mm512_srli_epi16(ಬೈಟುಗಳು, 4), ಕೆಳಗಿನ_ಮುಖವಾಡ));
        _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು TA, ಬಗೆ_ಹೆಸರು TB, ತಾರ್ಕಿಕ is_acc>
ರಚನೆ acc_C {};

ಮಾದರಿ <ತಾರ್ಕಿಕ is_acc>
ರಚನೆ acc_C<ಖಂಡ_q8_0, ಖಂಡ_q4_0, is_acc> {
    ಸ್ಥಿರ ಶೂನ್ಯ apply(ದಶಮಾಂಶ * RESTRICT C, ಪೂರ್ಣಾಂಕ ldc, ಸ್ಥಿರಾಂಕ int32_t * RESTRICT tile, ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * A, ಪೂರ್ಣಾಂಕ lda, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * packed_B, ಪೂರ್ಣಾಂಕ nr) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ = TILE_N * TILE_K / 2;
        ಸ್ಥಿರಾಂಕ __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)packed_B + ಸ್ಥಾನಾಂತರ)));

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ m = 0; m < nr; ++m) {
            ಸ್ಥಿರಾಂಕ __m512 vd1 = _mm512_set1_ps(GGML_CPU_FP16_ಗಮ್ಯ_FP32(A[m * lda].d));
            ಸ್ಥಿರಾಂಕ __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            ಆದರೆ (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } ಇಲ್ಲದಿದ್ದರೆ {
                vsum = _mm512_set1_ps(0.f);
            }
            vsum = _mm512_fmadd_ps(vtile, _mm512_mul_ps(vd0, vd1), vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

ಮಾದರಿ <ತಾರ್ಕಿಕ is_acc>
ರಚನೆ acc_C<ಖಂಡ_q8_1, ಖಂಡ_q4_1, is_acc> {
    ಸ್ಥಿರ ಶೂನ್ಯ apply(ದಶಮಾಂಶ * RESTRICT C, ಪೂರ್ಣಾಂಕ ldc, ಸ್ಥಿರಾಂಕ int32_t * RESTRICT tile, ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 * A, ಪೂರ್ಣಾಂಕ lda, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * packed_B, ಪೂರ್ಣಾಂಕ nr) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ = TILE_N * TILE_K / 2;
        ಸ್ಥಿರಾಂಕ __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)packed_B + ಸ್ಥಾನಾಂತರ)));
        ಸ್ಥಿರಾಂಕ __m512 vm0 = _mm512_cvtph_ps(_mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)packed_B + ಸ್ಥಾನಾಂತರ + TILE_N * ಗಾತ್ರ(ggml_half))));

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ m = 0; m < nr; ++m) {
            ಸ್ಥಿರಾಂಕ __m512 vd1 = _mm512_set1_ps(GGML_CPU_FP16_ಗಮ್ಯ_FP32(A[m * lda].d));
            ಸ್ಥಿರಾಂಕ __m512 vs1 = _mm512_set1_ps(GGML_CPU_FP16_ಗಮ್ಯ_FP32(A[m * lda].s));
            ಸ್ಥಿರಾಂಕ __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            ಆದರೆ (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } ಇಲ್ಲದಿದ್ದರೆ {
                vsum = _mm512_set1_ps(0.f);
            }
            vsum = _mm512_fmadd_ps(vtile, _mm512_mul_ps(vd0, vd1), vsum);
            vsum = _mm512_fmadd_ps(vm0, vs1, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

ಮಾದರಿ <ತಾರ್ಕಿಕ is_acc>
ರಚನೆ acc_C<ಖಂಡ_q8_0, ಖಂಡ_q8_0, is_acc> {
    ಸ್ಥಿರ ಶೂನ್ಯ apply(ದಶಮಾಂಶ * RESTRICT C, ಪೂರ್ಣಾಂಕ ldc, ಸ್ಥಿರಾಂಕ int32_t * RESTRICT tile, ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * A, ಪೂರ್ಣಾಂಕ lda, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * packed_B, ಪೂರ್ಣಾಂಕ nr) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ = TILE_N * TILE_K;
        ಸ್ಥಿರಾಂಕ __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)packed_B + ಸ್ಥಾನಾಂತರ)));

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ m = 0; m < nr; ++m) {
            ಸ್ಥಿರಾಂಕ __m512 vd1 = _mm512_set1_ps(GGML_CPU_FP16_ಗಮ್ಯ_FP32(A[m * lda].d));
            ಸ್ಥಿರಾಂಕ __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            ಆದರೆ (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } ಇಲ್ಲದಿದ್ದರೆ {
                vsum = _mm512_set1_ps(0.f);
            }
            vsum = _mm512_fmadd_ps(vtile, _mm512_mul_ps(vd0, vd1), vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

ಮಾದರಿ <ತಾರ್ಕಿಕ is_acc>
ರಚನೆ acc_C<ಖಂಡ_q8_K, ಖಂಡ_q4_K, is_acc> {
    ಸ್ಥಿರ ಶೂನ್ಯ apply(ದಶಮಾಂಶ * RESTRICT C, ಪೂರ್ಣಾಂಕ ldc, ಸ್ಥಿರಾಂಕ int32_t * RESTRICT tile, ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K * A, ಪೂರ್ಣಾಂಕ lda, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * packed_B, ಪೂರ್ಣಾಂಕ nr) {
        ಸ್ಥಿರಾಂಕ uint8_t * scales = reinterpret_cast<ಸ್ಥಿರಾಂಕ uint8_t *>((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)packed_B + (QK_K / 2) * TILE_N);
        ಸ್ಥಿರಾಂಕ uint8_t * mins = scales + 8 * TILE_N;
        ಸ್ಥಿರಾಂಕ ggml_half * d0 = reinterpret_cast<ಸ್ಥಿರಾಂಕ ggml_half *>(mins + 8 * TILE_N);
        ಸ್ಥಿರಾಂಕ ggml_half * dmin = d0 + TILE_N;

        ಸ್ಥಿರಾಂಕ __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)d0));
        ಸ್ಥಿರಾಂಕ __m512 vdmin = _mm512_cvtph_ps(_mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)dmin));

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ m = 0; m < nr; ++m) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d1 = A[m * lda].d;
            ಸ್ಥಿರಾಂಕ __m512 vd = _mm512_mul_ps(_mm512_set1_ps(d1), vd0);
            ಸ್ಥಿರಾಂಕ __m512 vdm = _mm512_mul_ps(_mm512_set1_ps(-d1), vdmin);
            ಸ್ಥಿರಾಂಕ __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            ಆದರೆ (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } ಇಲ್ಲದಿದ್ದರೆ {
                vsum = _mm512_set1_ps(0.f);
            }

            ಸ್ಥಿರಾಂಕ __m256i q8sums = _mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)A[m * lda].bsums);
            ಸ್ಥಿರಾಂಕ __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));

            __m512i acc_m = _mm512_setzero_si512();
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 4; ++k) {
                __m512i vmask = _mm512_set1_epi32(k);
                __m512i va = _mm512_permutexvar_epi32(vmask, _mm512_castsi128_si512(q8s));
                __m512i vb = _mm512_cvtepi8_epi16(_mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)(mins + k * 32)));
                acc_m = _mm512_dpwssds_epi32(acc_m, va, vb);
            }

            vsum = _mm512_fmadd_ps(vtile, vd, vsum);
            vsum = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc_m), vdm, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

ಮಾದರಿ <ತಾರ್ಕಿಕ is_acc>
ರಚನೆ acc_C<ಖಂಡ_q8_K, ಖಂಡ_q5_K, is_acc> {
    ಸ್ಥಿರ ಶೂನ್ಯ apply(ದಶಮಾಂಶ * RESTRICT C, ಪೂರ್ಣಾಂಕ ldc, ಸ್ಥಿರಾಂಕ int32_t * RESTRICT tile, ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K * A, ಪೂರ್ಣಾಂಕ lda, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * packed_B, ಪೂರ್ಣಾಂಕ nr) {
        ಸ್ಥಿರಾಂಕ uint8_t * scales = reinterpret_cast<ಸ್ಥಿರಾಂಕ uint8_t *>((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)packed_B + (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N);
        ಸ್ಥಿರಾಂಕ uint8_t * mins = scales + 8 * TILE_N;
        ಸ್ಥಿರಾಂಕ ggml_half * d0 = reinterpret_cast<ಸ್ಥಿರಾಂಕ ggml_half *>(mins + 8 * TILE_N);
        ಸ್ಥಿರಾಂಕ ggml_half * dmin = d0 + TILE_N;

        ಸ್ಥಿರಾಂಕ __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)d0));
        ಸ್ಥಿರಾಂಕ __m512 vdmin = _mm512_cvtph_ps(_mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)dmin));

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ m = 0; m < nr; ++m) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d1 = A[m * lda].d;
            ಸ್ಥಿರಾಂಕ __m512 vd = _mm512_mul_ps(_mm512_set1_ps(d1), vd0);
            ಸ್ಥಿರಾಂಕ __m512 vdm = _mm512_mul_ps(_mm512_set1_ps(-d1), vdmin);
            ಸ್ಥಿರಾಂಕ __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            ಆದರೆ (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } ಇಲ್ಲದಿದ್ದರೆ {
                vsum = _mm512_set1_ps(0.f);
            }

            ಸ್ಥಿರಾಂಕ __m256i q8sums = _mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)A[m * lda].bsums);
            ಸ್ಥಿರಾಂಕ __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));

            __m512i acc_m = _mm512_setzero_si512();
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 4; ++k) {
                __m512i vmask = _mm512_set1_epi32(k);
                __m512i va = _mm512_permutexvar_epi32(vmask, _mm512_castsi128_si512(q8s));
                __m512i vb = _mm512_cvtepi8_epi16(_mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)(mins + k * 32)));
                acc_m = _mm512_dpwssds_epi32(acc_m, va, vb);
            }

            vsum = _mm512_fmadd_ps(vtile, vd, vsum);
            vsum = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc_m), vdm, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

ಮಾದರಿ <ತಾರ್ಕಿಕ is_acc>
ರಚನೆ acc_C<ಖಂಡ_q8_K, ಖಂಡ_q6_K, is_acc> {
    ಸ್ಥಿರ ಶೂನ್ಯ apply(ದಶಮಾಂಶ * RESTRICT C, ಪೂರ್ಣಾಂಕ ldc, ಸ್ಥಿರಾಂಕ int32_t * RESTRICT tile, ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K * A, ಪೂರ್ಣಾಂಕ lda, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * packed_B, ಪೂರ್ಣಾಂಕ nr) {
        ಸ್ಥಿರಾಂಕ uint8_t * scales = reinterpret_cast<ಸ್ಥಿರಾಂಕ uint8_t *>((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)packed_B + (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N);
        ಸ್ಥಿರಾಂಕ ggml_half * d0 = reinterpret_cast<ಸ್ಥಿರಾಂಕ ggml_half *>(scales + 16 * TILE_N);

        ಸ್ಥಿರಾಂಕ __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)d0));

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ m = 0; m < nr; ++m) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d1 = A[m * lda].d;
            ಸ್ಥಿರಾಂಕ __m512 vd = _mm512_mul_ps(_mm512_set1_ps(d1), vd0);
            ಸ್ಥಿರಾಂಕ __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            ಆದರೆ (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } ಇಲ್ಲದಿದ್ದರೆ {
                vsum = _mm512_set1_ps(0.f);
            }

            vsum = _mm512_fmadd_ps(vtile, vd, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

ಮಾದರಿ <ತಾರ್ಕಿಕ is_acc>
ರಚನೆ acc_C<ಖಂಡ_q8_K, ಖಂಡ_iq4_xs, is_acc> {
    ಸ್ಥಿರ ಶೂನ್ಯ apply(ದಶಮಾಂಶ * RESTRICT C, ಪೂರ್ಣಾಂಕ ldc, ಸ್ಥಿರಾಂಕ int32_t * RESTRICT tile, ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K * A, ಪೂರ್ಣಾಂಕ lda, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * packed_B, ಪೂರ್ಣಾಂಕ nr) {
        ಸ್ಥಿರಾಂಕ int8_t * scales = reinterpret_cast<ಸ್ಥಿರಾಂಕ int8_t *>((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)packed_B + (QK_K / 2) * TILE_N);
        ಸ್ಥಿರಾಂಕ ggml_half * d0 = reinterpret_cast<ಸ್ಥಿರಾಂಕ ggml_half *>(scales + 8 * TILE_N);

        ಸ್ಥಿರಾಂಕ __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)d0));

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ m = 0; m < nr; ++m) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d1 = A[m * lda].d;
            ಸ್ಥಿರಾಂಕ __m512 vd = _mm512_mul_ps(_mm512_set1_ps(d1), vd0);
            ಸ್ಥಿರಾಂಕ __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            ಆದರೆ (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } ಇಲ್ಲದಿದ್ದರೆ {
                vsum = _mm512_set1_ps(0.f);
            }

            vsum = _mm512_fmadd_ps(vtile, vd, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು TB> constexpr ಪೂರ್ಣಾಂಕ get_quants_ಗಾತ್ರ();
ಮಾದರಿ <> constexpr ಪೂರ್ಣಾಂಕ get_quants_ಗಾತ್ರ<ಖಂಡ_q4_K>() { ಹಿಂತಿರುಗಿ (QK_K / 2) * TILE_N; }
ಮಾದರಿ <> constexpr ಪೂರ್ಣಾಂಕ get_quants_ಗಾತ್ರ<ಖಂಡ_q5_K>() { ಹಿಂತಿರುಗಿ (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N; }
ಮಾದರಿ <> constexpr ಪೂರ್ಣಾಂಕ get_quants_ಗಾತ್ರ<ಖಂಡ_q6_K>() { ಹಿಂತಿರುಗಿ (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N; }
ಮಾದರಿ <> constexpr ಪೂರ್ಣಾಂಕ get_quants_ಗಾತ್ರ<ಖಂಡ_iq4_xs>() { ಹಿಂತಿರುಗಿ (QK_K / 2) * TILE_N; }

// used for QKK format
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು TB, ತಾರ್ಕಿಕ is_acc,
          ಬಗೆ_ಹೆಸರು std::enable_if<is_type_qkk<TB>::value, ಪೂರ್ಣಾಂಕ>::type = 0>
ಸಾಲೊಳಗೆ ಶೂನ್ಯ scale_C(ಸ್ಥಿರಾಂಕ int32_t * RESTRICT tile, int32_t * RESTRICT sumi, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * packed_B, ಪೂರ್ಣಾಂಕ k, ಪೂರ್ಣಾಂಕ nr) {
    ಸ್ಥಿರಾಂಕ uint8_t * scales = reinterpret_cast<ಸ್ಥಿರಾಂಕ uint8_t *>((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)packed_B + get_quants_ಗಾತ್ರ<TB>());
    ಸ್ಥಿರಾಂಕ __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((ಸ್ಥಿರಾಂಕ __m128i *)(scales + k * TILE_N)));

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ m = 0; m < nr; ++m) {
        __m512i vsumi;
        ಆದರೆ (is_acc) {
            vsumi = _mm512_loadu_si512(sumi + m * TILE_N);
        } ಇಲ್ಲದಿದ್ದರೆ {
            vsumi = _mm512_setzero_si512();
        }
        __m512i vtile = _mm512_loadu_si512(tile + m * TILE_N);
        vsumi = mm512_ಕೂಡಿಸು_epi32(vsumi, _mm512_mullo_epi32(vtile, vscale));
        _mm512_storeu_si512((__m512i *)(sumi + m * TILE_N), vsumi);
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು TA, ಬಗೆ_ಹೆಸರು TB, ಬಗೆ_ಹೆಸರು TC, ಪೂರ್ಣಾಂಕ ಖಂಡ_M, ಪೂರ್ಣಾಂಕ ಖಂಡ_N, ಪೂರ್ಣಾಂಕ ಖಂಡ_K>
ರಚನೆ tinygemm_kernel_avx {
    ಸ್ಥಿರ ಶೂನ್ಯ apply(ಪೂರ್ಣಾಂಕ K, ಸ್ಥಿರಾಂಕ TA * RESTRICT A, ಸ್ಥಿರಾಂಕ TB * RESTRICT B, TC * RESTRICT C, ಪೂರ್ಣಾಂಕ ldc) {
        GGML_ಬಳಸದ(K);
        GGML_ಬಳಸದ(A);
        GGML_ಬಳಸದ(B);
        GGML_ಬಳಸದ(C);
        GGML_ಬಳಸದ(ldc);
    }
};

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ ಖಂಡ_M, ಪೂರ್ಣಾಂಕ ಖಂಡ_N, ಪೂರ್ಣಾಂಕ ಖಂಡ_K>
ರಚನೆ tinygemm_kernel_avx<ದಶಮಾಂಶ, ggml_fp16_t, ದಶಮಾಂಶ, ಖಂಡ_M, ಖಂಡ_N, ಖಂಡ_K> {
    ಸ್ಥಿರ ಶೂನ್ಯ apply(ಪೂರ್ಣಾಂಕ K, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * RESTRICT A, ಸ್ಥಿರಾಂಕ ggml_fp16_t * RESTRICT B, ದಶಮಾಂಶ * RESTRICT C, ಪೂರ್ಣಾಂಕ ldc) {
        constexpr ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲುಗಳು_2 = ಖಂಡ_M;
        constexpr ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = ಖಂಡ_N;
        ಪ್ರತಿಜ್ಞಾ(ಖಂಡ_K == 16);

        __m512 va;
        __m512 vb[ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ];
        __m512 vc[ಅಡ್ಡಸಾಲುಗಳು_2 * ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ];

        ಸ್ವಯಂಚಾಲಿತ loadc = [&](ಸ್ವಯಂಚಾಲಿತ idx) {
            vc[idx] = _mm512_setzero_ps();
        };
        Unroll<ಅಡ್ಡಸಾಲುಗಳು_2 * ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ>{}(loadc);

        ಸ್ವಯಂಚಾಲಿತ compute = [&](ಸ್ವಯಂಚಾಲಿತ idx, ಸ್ವಯಂಚಾಲಿತ k) {
            constexpr ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = idx / ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
            constexpr ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು = idx % ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;

            ಆದರೆ constexpr (ಲಂಬಸಾಲು == 0) {
                va = _mm512_loadu_ps(A + ಅಡ್ಡಸಾಲು * K + k);
            }
            ಆದರೆ constexpr (ಅಡ್ಡಸಾಲು == 0) {
                vb[ಲಂಬಸಾಲು] =  _mm512_cvtph_ps(_mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)(B + ಲಂಬಸಾಲು * K + k)));
            }
            vc[idx] = _mm512_fmadd_ps(va, vb[ಲಂಬಸಾಲು], vc[idx]);
        };

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < K; k += 16) {
            Unroll<ಅಡ್ಡಸಾಲುಗಳು_2 * ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ>{}(compute, k);
        }

        ಸ್ವಯಂಚಾಲಿತ storec = [&](ಸ್ವಯಂಚಾಲಿತ idx) {
            constexpr ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = idx / ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
            constexpr ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು = idx % ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
            C[ಅಡ್ಡಸಾಲು * ldc + ಲಂಬಸಾಲು] = mm512_reduce_ಕೂಡಿಸು_ps(vc[idx]);
        };
        Unroll<ಅಡ್ಡಸಾಲುಗಳು_2 * ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ>{}(storec);
    }
};

#ವ್ಯಾಖ್ಯಾನಿಸು LAUNCH_TINYGEMM_KERNEL_AVX(MB_ಗಾತ್ರ, NB_ಗಾತ್ರ)                                \
    tinygemm_kernel_avx<ದಶಮಾಂಶ, type, ದಶಮಾಂಶ, MB_ಗಾತ್ರ, NB_ಗಾತ್ರ, blck_ಗಾತ್ರ>::apply(    \
        K, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)src1->ದತ್ತಾಂಶ + src1_ಸ್ಥಾನಾಂತರ + mb_ಆರಂಭ * K,                  \
        (ಸ್ಥಿರಾಂಕ type *)src0->ದತ್ತಾಂಶ + src0_ಸ್ಥಾನಾಂತರ + nb_ಆರಂಭ * K,                      \
        (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ + dst_ಸ್ಥಾನಾಂತರ + mb_ಆರಂಭ * ldc + nb_ಆರಂಭ, ldc)


// re-organize in the format {NB, KB, TILE_SIZE}:
#ವ್ಯಾಖ್ಯಾನಿಸು PACKED_ಸೂಚ್ಯಂಕ(n, k, KB, tile_ಗಾತ್ರ) (n * KB + k) * tile_ಗಾತ್ರ

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು TB, ಪೂರ್ಣಾಂಕ ಖಂಡ_K>
ಶೂನ್ಯ convert_B_packed_format(ಶೂನ್ಯ * RESTRICT packed_B, ಸ್ಥಿರಾಂಕ TB * RESTRICT B, ಪೂರ್ಣಾಂಕ N, ಪೂರ್ಣಾಂಕ K) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ NB = N / TILE_N;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ KB = K / ಖಂಡ_K;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ TILE_ಗಾತ್ರ = get_tile_ಗಾತ್ರ<TB>();

    // parallel on NB should be enough
    parallel_for(NB, [&](ಪೂರ್ಣಾಂಕ begin, ಪೂರ್ಣಾಂಕ end) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ n = begin; n < end; ++n) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < KB; ++k) {
                ಪೂರ್ಣಾಂಕ n0 = n * TILE_N;
                pack_B((ಅಕ್ಷರ *)packed_B + PACKED_ಸೂಚ್ಯಂಕ(n, k, KB, TILE_ಗಾತ್ರ), &B[n0 * KB + k], KB);
            }
        }
    });
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು TA, ಬಗೆ_ಹೆಸರು TB, ಬಗೆ_ಹೆಸರು TC, ಪೂರ್ಣಾಂಕ ಖಂಡ_M, ಪೂರ್ಣಾಂಕ ಖಂಡ_N, ಪೂರ್ಣಾಂಕ ಖಂಡ_K>
ರಚನೆ tinygemm_kernel_vnni {};

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ ಖಂಡ_M, ಪೂರ್ಣಾಂಕ ಖಂಡ_N, ಪೂರ್ಣಾಂಕ ಖಂಡ_K>
ರಚನೆ tinygemm_kernel_vnni<ಖಂಡ_q8_0, ಖಂಡ_q4_0, ದಶಮಾಂಶ, ಖಂಡ_M, ಖಂಡ_N, ಖಂಡ_K> {
    ಸ್ಥಿರ ಶೂನ್ಯ apply(ಪೂರ್ಣಾಂಕ KB, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * RESTRICT _A, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * RESTRICT _B, ದಶಮಾಂಶ * RESTRICT C, ಪೂರ್ಣಾಂಕ ldc) {

        constexpr ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = ಖಂಡ_N / 16;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ TILE_ಗಾತ್ರ = TILE_N * ಗಾತ್ರ(ಖಂಡ_q4_0);

        ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * RESTRICT A = static_cast<ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *>(_A);
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * RESTRICT B = static_cast<ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *>(_B);

        __m512i va[8];
        __m512 vc[ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ];
        __m512 vd1;

        // sum of offsets, shared across COLS
        //
        // avx512-vnni does not have `_mm512_dpbssd_epi32`,
        // need to transform ss to us:
        //   a * (b - 8) is equivalent to b * a - 8 * a
        //   s    u   u                   u   s   u   s
        //
        __m512i vcomp;

        ಸ್ಥಿರಾಂಕ __m512i off = _mm512_set1_epi8(8);
        ಸ್ಥಿರಾಂಕ __m512i ಕೆಳಗಿನ_ಮುಖವಾಡ = _mm512_set1_epi8(0xF);

        ಸ್ವಯಂಚಾಲಿತ loadc = [&](ಸ್ವಯಂಚಾಲಿತ ಲಂಬಸಾಲು) {
            vc[ಲಂಬಸಾಲು] = _mm512_setzero_ps();
        };
        Unroll<ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ>{}(loadc);

        ಸ್ವಯಂಚಾಲಿತ compute = [&](ಸ್ವಯಂಚಾಲಿತ ಲಂಬಸಾಲು, ಸ್ವಯಂಚಾಲಿತ i) {
            // load a and compute compensation
            ಆದರೆ constexpr (ಲಂಬಸಾಲು == 0) {
                ಸ್ಥಿರಾಂಕ int32_t * a_ptr = reinterpret_cast<ಸ್ಥಿರಾಂಕ int32_t *>(A[0 * KB + i].qs);
                vcomp = _mm512_setzero_si512();
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 8; ++k) {
                    va[k] = _mm512_set1_epi32(a_ptr[k]);
                    vcomp = _mm512_dpbusd_epi32(vcomp, off, va[k]);
                }
                vd1 = _mm512_set1_ps(GGML_CPU_FP16_ಗಮ್ಯ_FP32(A[0 * KB + i].d));
            }

            // load b
            __m512i vsum = _mm512_setzero_si512();
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * b_ptr = B + PACKED_ಸೂಚ್ಯಂಕ(ಲಂಬಸಾಲು, i, KB, TILE_ಗಾತ್ರ);
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 8; k += 2) {
                __m512i ಬೈಟುಗಳು = _mm512_loadu_si512((ಸ್ಥಿರಾಂಕ __m512i *)(b_ptr + k * 32));
                __m512i vb0 = _mm512_and_si512(ಬೈಟುಗಳು, ಕೆಳಗಿನ_ಮುಖವಾಡ);
                vsum = _mm512_dpbusd_epi32(vsum, vb0, va[k + 0]);
                __m512i vb1 = _mm512_and_si512(_mm512_srli_epi16(ಬೈಟುಗಳು, 4), ಕೆಳಗಿನ_ಮುಖವಾಡ);
                vsum = _mm512_dpbusd_epi32(vsum, vb1, va[k + 1]);
            }
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ = TILE_N * TILE_K / 2;
            ಸ್ಥಿರಾಂಕ __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)(b_ptr + ಸ್ಥಾನಾಂತರ)));
            vsum = _mm512_sub_epi32(vsum, vcomp);

            vc[ಲಂಬಸಾಲು] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vsum), _mm512_mul_ps(vd0, vd1), vc[ಲಂಬಸಾಲು]);
        };

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < KB; ++i) {
            Unroll<ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ>{}(compute, i);
        }

        //store to C
        ಸ್ವಯಂಚಾಲಿತ storec = [&](ಸ್ವಯಂಚಾಲಿತ ಲಂಬಸಾಲು) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + ಲಂಬಸಾಲು * 16), vc[ಲಂಬಸಾಲು]);
        };
        Unroll<ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ>{}(storec);
    }
};

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ ಖಂಡ_N, ಪೂರ್ಣಾಂಕ ಖಂಡ_K>
ರಚನೆ tinygemm_kernel_vnni<ಖಂಡ_q8_1, ಖಂಡ_q4_1, ದಶಮಾಂಶ, 1, ಖಂಡ_N, ಖಂಡ_K> {
    ಸ್ಥಿರ ಶೂನ್ಯ apply(ಪೂರ್ಣಾಂಕ KB, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * RESTRICT _A, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * RESTRICT _B, ದಶಮಾಂಶ * RESTRICT C, ಪೂರ್ಣಾಂಕ ldc) {

        constexpr ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = ಖಂಡ_N / 16;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ TILE_ಗಾತ್ರ = TILE_N * ಗಾತ್ರ(ಖಂಡ_q4_1);

        ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 * RESTRICT A = static_cast<ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 *>(_A);
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * RESTRICT B = static_cast<ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *>(_B);

        __m512i va[8];
        __m512i vb[8];
        __m512 vc[ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ];
        __m512 vd1, vs1;

        ಸ್ಥಿರಾಂಕ __m512i ಕೆಳಗಿನ_ಮುಖವಾಡ = _mm512_set1_epi8(0xF);

        ಸ್ವಯಂಚಾಲಿತ loadc = [&](ಸ್ವಯಂಚಾಲಿತ ಲಂಬಸಾಲು) {
            vc[ಲಂಬಸಾಲು] = _mm512_setzero_ps();
        };
        Unroll<ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ>{}(loadc);

        ಸ್ವಯಂಚಾಲಿತ compute = [&](ಸ್ವಯಂಚಾಲಿತ ಲಂಬಸಾಲು, ಸ್ವಯಂಚಾಲಿತ i) {
            // load a
            ಆದರೆ constexpr (ಲಂಬಸಾಲು == 0) {
                ಸ್ಥಿರಾಂಕ int32_t * a_ptr = reinterpret_cast<ಸ್ಥಿರಾಂಕ int32_t *>(A[0 * KB + i].qs);
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 8; ++k) {
                    va[k] = _mm512_set1_epi32(a_ptr[k]);
                }
                vd1 = _mm512_set1_ps(GGML_CPU_FP16_ಗಮ್ಯ_FP32(A[0 * KB + i].d));
                vs1 = _mm512_set1_ps(GGML_CPU_FP16_ಗಮ್ಯ_FP32(A[0 * KB + i].s));
            }

            // load b
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * b_ptr = B + PACKED_ಸೂಚ್ಯಂಕ(ಲಂಬಸಾಲು, i, KB, TILE_ಗಾತ್ರ);
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 8; k += 2) {
                __m512i ಬೈಟುಗಳು = _mm512_loadu_si512((ಸ್ಥಿರಾಂಕ __m512i *)(b_ptr + k * 32));
                vb[k + 0] = _mm512_and_si512(ಬೈಟುಗಳು, ಕೆಳಗಿನ_ಮುಖವಾಡ);
                vb[k + 1] = _mm512_and_si512(_mm512_srli_epi16(ಬೈಟುಗಳು, 4), ಕೆಳಗಿನ_ಮುಖವಾಡ);
            }
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ = TILE_N * TILE_K / 2;
            ಸ್ಥಿರಾಂಕ __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)(b_ptr + ಸ್ಥಾನಾಂತರ)));
            ಸ್ಥಿರಾಂಕ __m512 vm0 = _mm512_cvtph_ps(_mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)(b_ptr + ಸ್ಥಾನಾಂತರ + TILE_N * ಗಾತ್ರ(ggml_half))));

            __m512i vsum = _mm512_setzero_si512();
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 8; ++k) {
                vsum = _mm512_dpbusd_epi32(vsum, vb[k], va[k]);
            }

            vc[ಲಂಬಸಾಲು] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vsum), _mm512_mul_ps(vd0, vd1), vc[ಲಂಬಸಾಲು]);
            vc[ಲಂಬಸಾಲು] = _mm512_fmadd_ps(vm0, vs1, vc[ಲಂಬಸಾಲು]);
        };

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < KB; ++i) {
            Unroll<ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ>{}(compute, i);
        }

        //store to C
        ಸ್ವಯಂಚಾಲಿತ storec = [&](ಸ್ವಯಂಚಾಲಿತ ಲಂಬಸಾಲು) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + ಲಂಬಸಾಲು * 16), vc[ಲಂಬಸಾಲು]);
        };
        Unroll<ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ>{}(storec);
    }
};

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ ಖಂಡ_M, ಪೂರ್ಣಾಂಕ ಖಂಡ_N, ಪೂರ್ಣಾಂಕ ಖಂಡ_K>
ರಚನೆ tinygemm_kernel_vnni<ಖಂಡ_q8_0, ಖಂಡ_q8_0, ದಶಮಾಂಶ, ಖಂಡ_M, ಖಂಡ_N, ಖಂಡ_K> {
    ಸ್ಥಿರ ಶೂನ್ಯ apply(ಪೂರ್ಣಾಂಕ KB, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * RESTRICT _A, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * RESTRICT _B, ದಶಮಾಂಶ * RESTRICT C, ಪೂರ್ಣಾಂಕ ldc) {

        constexpr ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = ಖಂಡ_N / 16;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ TILE_ಗಾತ್ರ = TILE_N * ಗಾತ್ರ(ಖಂಡ_q8_0) + TILE_N * ಗಾತ್ರ(int32_t);

        ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * RESTRICT A = static_cast<ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *>(_A);
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * RESTRICT B = static_cast<ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *>(_B);

        __m512i va[8];
        __m512i vb[8];
        __m512 vc[ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ];
        __m512 vd1;

        // Notes: s8s8 igemm compensation in avx512-vnni
        // change s8s8 to u8s8 with compensate
        //   a * b = (a + 128) * b - 128 * b
        //   s   s       u       s    u    s
        //
        // (128 * b is pre-computed when packing B to vnni formats)
        //
        ಸ್ಥಿರಾಂಕ __m512i off = _mm512_set1_epi8(static_cast<ಅಕ್ಷರ>(0x80));

        ಸ್ವಯಂಚಾಲಿತ loadc = [&](ಸ್ವಯಂಚಾಲಿತ ಲಂಬಸಾಲು) {
            vc[ಲಂಬಸಾಲು] = _mm512_setzero_ps();
        };
        Unroll<ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ>{}(loadc);

        ಸ್ವಯಂಚಾಲಿತ compute = [&](ಸ್ವಯಂಚಾಲಿತ ಲಂಬಸಾಲು, ಸ್ವಯಂಚಾಲಿತ i) {
            // load a and add offset 128
            ಆದರೆ constexpr (ಲಂಬಸಾಲು == 0) {
                ಸ್ಥಿರಾಂಕ int32_t * a_ptr = reinterpret_cast<ಸ್ಥಿರಾಂಕ int32_t *>(A[0 * KB + i].qs);
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 8; ++k) {
                    va[k] = _mm512_set1_epi32(a_ptr[k]);
                    va[k] = mm512_ಕೂಡಿಸು_epi8(va[k], off);
                }
                vd1 = _mm512_set1_ps(GGML_CPU_FP16_ಗಮ್ಯ_FP32(A[0 * KB + i].d));
            }

            // load b
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * b_ptr = B + PACKED_ಸೂಚ್ಯಂಕ(ಲಂಬಸಾಲು, i, KB, TILE_ಗಾತ್ರ);
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 8; ++k) {
                vb[k] = _mm512_loadu_si512((ಸ್ಥಿರಾಂಕ __m512i *)(b_ptr + k * 64));
            }
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ = TILE_N * TILE_K;
            ಸ್ಥಿರಾಂಕ __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)(b_ptr + ಸ್ಥಾನಾಂತರ)));
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ offset2 = TILE_N * TILE_K + TILE_N * ಗಾತ್ರ(ggml_half);
            ಸ್ಥಿರಾಂಕ __m512i vcomp = _mm512_loadu_si512((ಸ್ಥಿರಾಂಕ __m512i *)(b_ptr + offset2));

            __m512i vsum = _mm512_setzero_si512();
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 8; ++k) {
                vsum = _mm512_dpbusd_epi32(vsum, va[k], vb[k]);
            }
            vsum = _mm512_sub_epi32(vsum, vcomp);

            vc[ಲಂಬಸಾಲು] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vsum), _mm512_mul_ps(vd0, vd1), vc[ಲಂಬಸಾಲು]);
        };

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < KB; ++i) {
            Unroll<ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ>{}(compute, i);
        }

        //store to C
        ಸ್ವಯಂಚಾಲಿತ storec = [&](ಸ್ವಯಂಚಾಲಿತ ಲಂಬಸಾಲು) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + ಲಂಬಸಾಲು * 16), vc[ಲಂಬಸಾಲು]);
        };
        Unroll<ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ>{}(storec);
    }
};

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ ಖಂಡ_M, ಪೂರ್ಣಾಂಕ ಖಂಡ_N, ಪೂರ್ಣಾಂಕ ಖಂಡ_K>
ರಚನೆ tinygemm_kernel_vnni<ಖಂಡ_q8_K, ಖಂಡ_q4_K, ದಶಮಾಂಶ, ಖಂಡ_M, ಖಂಡ_N, ಖಂಡ_K> {
    ಸ್ಥಿರ ಶೂನ್ಯ apply(ಪೂರ್ಣಾಂಕ KB, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * RESTRICT _A, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * RESTRICT _B, ದಶಮಾಂಶ * RESTRICT C, ಪೂರ್ಣಾಂಕ ldc) {

        constexpr ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = ಖಂಡ_N / 16;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ TILE_ಗಾತ್ರ = TILE_N * ಗಾತ್ರ(ಖಂಡ_q4_K) + TILE_N * 4;

        ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K * RESTRICT A = static_cast<ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K *>(_A);
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * RESTRICT B = static_cast<ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *>(_B);

        // a.qs:   8 groups, 32 bytes each group (m256i)
        __m512i va[8];
        // a.bsum: 8 groups,  2 bytes each group (m128i)
        __m512i va_bsum;
        __m512 vc[ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ];
        __m512 vd1;

        // packed_B:
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ_scales = (QK_K / 2) * TILE_N;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ_mins   = (QK_K / 2) * TILE_N +  8 * TILE_N;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ_d0     = (QK_K / 2) * TILE_N + 16 * TILE_N;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ_dmin   = (QK_K / 2) * TILE_N + 16 * TILE_N + TILE_N * ಗಾತ್ರ(ggml_half);

        ಸ್ಥಿರಾಂಕ __m512i ಕೆಳಗಿನ_ಮುಖವಾಡ = _mm512_set1_epi8(0xF);

        ಸ್ವಯಂಚಾಲಿತ loadc = [&](ಸ್ವಯಂಚಾಲಿತ ಲಂಬಸಾಲು) {
            vc[ಲಂಬಸಾಲು] = _mm512_setzero_ps();
        };
        Unroll<ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ>{}(loadc);

        // Notes: vnni formats in QK_K
        //   a) quants vnni format
        //     int8  {k/4, n, 4}, viewed as 2d {k/4, 4n}, k = 32
        //     from {16, 32} to {8, 64}
        //
        //   b) min vnni format
        //     int16 {k/2, n, 2}, viewed as 2d {k/2, 2n}, k = 8
        //     from {16,  8} to {4, 32}
        //
        ಸ್ವಯಂಚಾಲಿತ compute = [&](ಸ್ವಯಂಚಾಲಿತ ಲಂಬಸಾಲು, ಸ್ವಯಂಚಾಲಿತ i) {
            // load a
            ಆದರೆ constexpr (ಲಂಬಸಾಲು == 0) {
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k_ಗುಂಪು = 0; k_ಗುಂಪು < QK_K / 32; ++k_ಗುಂಪು) {
                    va[k_ಗುಂಪು] = _mm512_castsi256_si512(_mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)(A[0 * KB + i].qs + k_ಗುಂಪು * 32)));
                }
                ಸ್ಥಿರಾಂಕ __m256i q8sums = _mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)A[0 * KB + i].bsums);
                ಸ್ಥಿರಾಂಕ __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
                va_bsum = _mm512_castsi128_si512(q8s);
                vd1 = _mm512_set1_ps(A[0 * KB + i].d);
            }

            // step 1: accumultate the quants
            __m512i acc = _mm512_setzero_si512();
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * b_ptr = B + PACKED_ಸೂಚ್ಯಂಕ(ಲಂಬಸಾಲು, i, KB, TILE_ಗಾತ್ರ);
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * b_qs  = b_ptr;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k_ಗುಂಪು = 0; k_ಗುಂಪು < QK_K / 32; ++k_ಗುಂಪು) {
                __m512i vsum = _mm512_setzero_si512();
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 8; k += 2) {
                    __m512i va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(k + 0), va[k_ಗುಂಪು]);
                    __m512i va1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(k + 1), va[k_ಗುಂಪು]);

                    __m512i ಬೈಟುಗಳು = _mm512_loadu_si512((ಸ್ಥಿರಾಂಕ __m512i *)b_qs);
                    __m512i vb0 = _mm512_and_si512(ಬೈಟುಗಳು, ಕೆಳಗಿನ_ಮುಖವಾಡ);
                    vsum = _mm512_dpbusd_epi32(vsum, vb0, va0);
                    __m512i vb1 = _mm512_and_si512(_mm512_srli_epi16(ಬೈಟುಗಳು, 4), ಕೆಳಗಿನ_ಮುಖವಾಡ);
                    vsum = _mm512_dpbusd_epi32(vsum, vb1, va1);

                    b_qs += 64;
                }
                // vacc += scale * (q8 @ q4)
                ಸ್ಥಿರಾಂಕ __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((ಸ್ಥಿರಾಂಕ __m128i *)(b_ptr + ಸ್ಥಾನಾಂತರ_scales + k_ಗುಂಪು * TILE_N)));
                acc = mm512_ಕೂಡಿಸು_epi32(acc, _mm512_mullo_epi32(vsum, vscale));
            }
            ಸ್ಥಿರಾಂಕ __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)(b_ptr + ಸ್ಥಾನಾಂತರ_d0)));
            vc[ಲಂಬಸಾಲು] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(vd0, vd1), vc[ಲಂಬಸಾಲು]);

            // step 2: accumulate the mins
            __m512i acc_m = _mm512_setzero_si512();
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 4; ++k) {
                __m512i vmask = _mm512_set1_epi32(k);
                __m512i va = _mm512_permutexvar_epi32(vmask, va_bsum);
                __m512i vb = _mm512_cvtepi8_epi16(_mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)(b_ptr + ಸ್ಥಾನಾಂತರ_mins + k * 32)));
                acc_m = _mm512_dpwssds_epi32(acc_m, va, vb);
            }
            ಸ್ಥಿರಾಂಕ __m512 vdmin = _mm512_cvtph_ps(_mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)(b_ptr + ಸ್ಥಾನಾಂತರ_dmin)));
            vc[ಲಂಬಸಾಲು] = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(acc_m), _mm512_mul_ps(vdmin, vd1), vc[ಲಂಬಸಾಲು]);
        };

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < KB; ++i) {
            Unroll<ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ>{}(compute, i);
        }

        //store to C
        ಸ್ವಯಂಚಾಲಿತ storec = [&](ಸ್ವಯಂಚಾಲಿತ ಲಂಬಸಾಲು) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + ಲಂಬಸಾಲು * 16), vc[ಲಂಬಸಾಲು]);
        };
        Unroll<ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ>{}(storec);
    }
};

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ ಖಂಡ_M, ಪೂರ್ಣಾಂಕ ಖಂಡ_N, ಪೂರ್ಣಾಂಕ ಖಂಡ_K>
ರಚನೆ tinygemm_kernel_vnni<ಖಂಡ_q8_K, ಖಂಡ_q5_K, ದಶಮಾಂಶ, ಖಂಡ_M, ಖಂಡ_N, ಖಂಡ_K> {
    ಸ್ಥಿರ ಶೂನ್ಯ apply(ಪೂರ್ಣಾಂಕ KB, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * RESTRICT _A, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * RESTRICT _B, ದಶಮಾಂಶ * RESTRICT C, ಪೂರ್ಣಾಂಕ ldc) {

        constexpr ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = ಖಂಡ_N / 16;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ TILE_ಗಾತ್ರ = TILE_N * ಗಾತ್ರ(ಖಂಡ_q5_K) + TILE_N * 4;

        ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K * RESTRICT A = static_cast<ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K *>(_A);
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * RESTRICT B = static_cast<ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *>(_B);

        // a.qs:   8 groups, 32 bytes each group (m256i)
        __m512i va[8];
        // a.bsum: 8 groups,  2 bytes each group (m128i)
        __m512i va_bsum;
        __m512 vc[ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ];
        __m512 vd1;

        // packed_B:
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ_qh     = (QK_K / 2) * TILE_N;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ_scales = (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ_mins   = (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N +  8 * TILE_N;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ_d0     = (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N + 16 * TILE_N;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ_dmin   = (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N + 16 * TILE_N + TILE_N * ಗಾತ್ರ(ggml_half);

        ಸ್ಥಿರಾಂಕ __m512i ಕೆಳಗಿನ_ಮುಖವಾಡ = _mm512_set1_epi8(0xF);

        ಸ್ವಯಂಚಾಲಿತ loadc = [&](ಸ್ವಯಂಚಾಲಿತ ಲಂಬಸಾಲು) {
            vc[ಲಂಬಸಾಲು] = _mm512_setzero_ps();
        };
        Unroll<ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ>{}(loadc);

        // Q5_K and Q4_K shares the same vnni formats, refer to notes above.
        ಸ್ವಯಂಚಾಲಿತ compute = [&](ಸ್ವಯಂಚಾಲಿತ ಲಂಬಸಾಲು, ಸ್ವಯಂಚಾಲಿತ i) {
            // load a
            ಆದರೆ constexpr (ಲಂಬಸಾಲು == 0) {
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k_ಗುಂಪು = 0; k_ಗುಂಪು < QK_K / 32; ++k_ಗುಂಪು) {
                    va[k_ಗುಂಪು] = _mm512_castsi256_si512(_mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)(A[0 * KB + i].qs + k_ಗುಂಪು * 32)));
                }
                ಸ್ಥಿರಾಂಕ __m256i q8sums = _mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)A[0 * KB + i].bsums);
                ಸ್ಥಿರಾಂಕ __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
                va_bsum = _mm512_castsi128_si512(q8s);
                vd1 = _mm512_set1_ps(A[0 * KB + i].d);
            }

            // step 1: accumultate the quants
            __m512i acc = _mm512_setzero_si512();
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * b_ptr = B + PACKED_ಸೂಚ್ಯಂಕ(ಲಂಬಸಾಲು, i, KB, TILE_ಗಾತ್ರ);
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * b_qs  = b_ptr;
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * b_qh  = b_ptr + ಸ್ಥಾನಾಂತರ_qh;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k_ಗುಂಪು = 0; k_ಗುಂಪು < QK_K / 32; ++k_ಗುಂಪು) {
                __m512i vsum = _mm512_setzero_si512();
                __m512i hmask0 = _mm512_set1_epi8(0x1);
                __m512i hmask1 = _mm512_set1_epi8(0x2);
                __m512i hbits = _mm512_loadu_si512((ಸ್ಥಿರಾಂಕ __m512i *)(b_qh + k_ಗುಂಪು * 64));
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 8; k += 2) {
                    __m512i va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(k + 0), va[k_ಗುಂಪು]);
                    __m512i va1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(k + 1), va[k_ಗುಂಪು]);

                    __m512i ಬೈಟುಗಳು = _mm512_loadu_si512((ಸ್ಥಿರಾಂಕ __m512i *)b_qs);
                    __m512i vb0 = _mm512_and_si512(ಬೈಟುಗಳು, ಕೆಳಗಿನ_ಮುಖವಾಡ);
                    __m512i vb1 = _mm512_and_si512(_mm512_srli_epi16(ಬೈಟುಗಳು, 4), ಕೆಳಗಿನ_ಮುಖವಾಡ);

                    __m512i vh0 = _mm512_slli_epi16(_mm512_srli_epi16(_mm512_and_si512(hbits, hmask0), k), 4);
                    __m512i vh1 = _mm512_slli_epi16(_mm512_srli_epi16(_mm512_and_si512(hbits, hmask1), k + 1), 4);

                    hmask0 = _mm512_slli_epi16(hmask0, 2);
                    hmask1 = _mm512_slli_epi16(hmask1, 2);
                    vb0 = mm512_ಕೂಡಿಸು_epi8(vb0, vh0);
                    vb1 = mm512_ಕೂಡಿಸು_epi8(vb1, vh1);

                    vsum = _mm512_dpbusd_epi32(vsum, vb0, va0);
                    vsum = _mm512_dpbusd_epi32(vsum, vb1, va1);

                    b_qs += 64;
                }
                // vacc += scale * (q8 @ q5)
                ಸ್ಥಿರಾಂಕ __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((ಸ್ಥಿರಾಂಕ __m128i *)(b_ptr + ಸ್ಥಾನಾಂತರ_scales + k_ಗುಂಪು * TILE_N)));
                acc = mm512_ಕೂಡಿಸು_epi32(acc, _mm512_mullo_epi32(vsum, vscale));
            }
            ಸ್ಥಿರಾಂಕ __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)(b_ptr + ಸ್ಥಾನಾಂತರ_d0)));
            vc[ಲಂಬಸಾಲು] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(vd0, vd1), vc[ಲಂಬಸಾಲು]);

            // step 2: accumulate the mins
            __m512i acc_m = _mm512_setzero_si512();
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 4; ++k) {
                __m512i vmask = _mm512_set1_epi32(k);
                __m512i va = _mm512_permutexvar_epi32(vmask, va_bsum);
                __m512i vb = _mm512_cvtepi8_epi16(_mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)(b_ptr + ಸ್ಥಾನಾಂತರ_mins + k * 32)));
                acc_m = _mm512_dpwssds_epi32(acc_m, va, vb);
            }
            ಸ್ಥಿರಾಂಕ __m512 vdmin = _mm512_cvtph_ps(_mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)(b_ptr + ಸ್ಥಾನಾಂತರ_dmin)));
            vc[ಲಂಬಸಾಲು] = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(acc_m), _mm512_mul_ps(vdmin, vd1), vc[ಲಂಬಸಾಲು]);
        };

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < KB; ++i) {
            Unroll<ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ>{}(compute, i);
        }

        //store to C
        ಸ್ವಯಂಚಾಲಿತ storec = [&](ಸ್ವಯಂಚಾಲಿತ ಲಂಬಸಾಲು) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + ಲಂಬಸಾಲು * 16), vc[ಲಂಬಸಾಲು]);
        };
        Unroll<ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ>{}(storec);
    }
};

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ ಖಂಡ_M, ಪೂರ್ಣಾಂಕ ಖಂಡ_N, ಪೂರ್ಣಾಂಕ ಖಂಡ_K>
ರಚನೆ tinygemm_kernel_vnni<ಖಂಡ_q8_K, ಖಂಡ_q6_K, ದಶಮಾಂಶ, ಖಂಡ_M, ಖಂಡ_N, ಖಂಡ_K> {
    ಸ್ಥಿರ ಶೂನ್ಯ apply(ಪೂರ್ಣಾಂಕ KB, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * RESTRICT _A, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * RESTRICT _B, ದಶಮಾಂಶ * RESTRICT C, ಪೂರ್ಣಾಂಕ ldc) {

        constexpr ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = ಖಂಡ_N / 16;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ TILE_ಗಾತ್ರ = TILE_N * ಗಾತ್ರ(ಖಂಡ_q6_K);

        ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K * RESTRICT A = static_cast<ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K *>(_A);
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * RESTRICT B = static_cast<ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *>(_B);

        // load the 256 bytes from A to 4 avx512 vectors
        __m512i va[4];
        __m512 vc[ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ];
        __m512 vd1;

        // packed_B:
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ_qh     = (QK_K / 2) * TILE_N;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ_scales = (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ_d0     = (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N + 16 * TILE_N;

        // compensation
        __m512i vcomp;

        ಸ್ಥಿರಾಂಕ __m512i m32s = _mm512_set1_epi32(32);
        ಸ್ಥಿರಾಂಕ __m512i ಕೆಳಗಿನ_ಮುಖವಾಡ = _mm512_set1_epi8(0xF);

        ಸ್ವಯಂಚಾಲಿತ loadc = [&](ಸ್ವಯಂಚಾಲಿತ ಲಂಬಸಾಲು) {
            vc[ಲಂಬಸಾಲು] = _mm512_setzero_ps();
        };
        Unroll<ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ>{}(loadc);

        ಸ್ವಯಂಚಾಲಿತ compute = [&](ಸ್ವಯಂಚಾಲಿತ ಲಂಬಸಾಲು, ಸ್ವಯಂಚಾಲಿತ i) {
            ಆದರೆ constexpr (ಲಂಬಸಾಲು == 0) {
                // load a
                va[0] = _mm512_loadu_si512((ಸ್ಥಿರಾಂಕ __m512i *)(A[0 * KB + i].qs +   0));
                va[1] = _mm512_loadu_si512((ಸ್ಥಿರಾಂಕ __m512i *)(A[0 * KB + i].qs +  64));
                va[2] = _mm512_loadu_si512((ಸ್ಥಿರಾಂಕ __m512i *)(A[0 * KB + i].qs + 128));
                va[3] = _mm512_loadu_si512((ಸ್ಥಿರಾಂಕ __m512i *)(A[0 * KB + i].qs + 192));

                ಸ್ಥಿರಾಂಕ __m256i q8sums = _mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)A[0 * KB + i].bsums);
                vcomp = _mm512_mullo_epi32(_mm512_cvtepi16_epi32(q8sums), m32s);
                vd1 = _mm512_set1_ps(A[0 * KB + i].d);
            }

            // accmulate the quants
            __m512i acc = _mm512_setzero_si512();
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * b_ptr = B + PACKED_ಸೂಚ್ಯಂಕ(ಲಂಬಸಾಲು, i, KB, TILE_ಗಾತ್ರ);
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * b_qs = b_ptr;
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * b_qh = b_ptr + ಸ್ಥಾನಾಂತರ_qh;
            ಪೂರ್ಣಾಂಕ ಮುಖವಾಡ = 0;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k_ಗುಂಪು = 0; k_ಗುಂಪು < QK_K / 16; ++k_ಗುಂಪು) {
                ಪೂರ್ಣಾಂಕ r = k_ಗುಂಪು >> 2;
                __m512i va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(ಮುಖವಾಡ++), va[r]);
                __m512i va1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(ಮುಖವಾಡ++), va[r]);

                __m512i vsum = _mm512_setzero_si512();
                __m512i hmask = _mm512_set1_epi8(0x3);

                __m512i ಬೈಟುಗಳು = _mm512_loadu_si512(b_qs);
                __m512i hbits = _mm512_loadu_si512(b_qh);
                __m512i vb0 = _mm512_and_si512(ಬೈಟುಗಳು, ಕೆಳಗಿನ_ಮುಖವಾಡ);
                __m512i vb1 = _mm512_and_si512(_mm512_srli_epi16(ಬೈಟುಗಳು, 4), ಕೆಳಗಿನ_ಮುಖವಾಡ);
                __m512i vh0 = _mm512_slli_epi16(_mm512_and_si512(hbits, hmask), 4);
                __m512i vh1 = _mm512_slli_epi16(_mm512_and_si512(hbits, _mm512_slli_epi16(hmask, 2)), 2);

                vb0 = mm512_ಕೂಡಿಸು_epi8(vb0, vh0);
                vb1 = mm512_ಕೂಡಿಸು_epi8(vb1, vh1);
                vsum = _mm512_dpbusd_epi32(vsum, vb0, va0);
                vsum = _mm512_dpbusd_epi32(vsum, vb1, va1);
                b_qs += 64;

                va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(ಮುಖವಾಡ++), va[r]);
                va1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(ಮುಖವಾಡ++), va[r]);

                ಬೈಟುಗಳು = _mm512_loadu_si512(b_qs);
                vb0 = _mm512_and_si512(ಬೈಟುಗಳು, ಕೆಳಗಿನ_ಮುಖವಾಡ);
                vb1 = _mm512_and_si512(_mm512_srli_epi16(ಬೈಟುಗಳು, 4), ಕೆಳಗಿನ_ಮುಖವಾಡ);
                vh0 =                   _mm512_and_si512(hbits, _mm512_slli_epi16(hmask, 4));
                vh1 = _mm512_srli_epi16(_mm512_and_si512(hbits, _mm512_slli_epi16(hmask, 6)), 2);
                vb0 = mm512_ಕೂಡಿಸು_epi8(vb0, vh0);
                vb1 = mm512_ಕೂಡಿಸು_epi8(vb1, vh1);
                vsum = _mm512_dpbusd_epi32(vsum, vb0, va0);
                vsum = _mm512_dpbusd_epi32(vsum, vb1, va1);
                b_qs += 64;
                b_qh += 64;

                // B * A - 32 * A
                __m512i vmask = _mm512_set1_epi32(k_ಗುಂಪು);
                vsum = _mm512_sub_epi32(vsum, _mm512_permutexvar_epi32(vmask, vcomp));

                // vacc += scale * (q8 @ q6)
                ಸ್ಥಿರಾಂಕ __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((ಸ್ಥಿರಾಂಕ __m128i *)(b_ptr + ಸ್ಥಾನಾಂತರ_scales + k_ಗುಂಪು * TILE_N)));
                acc = mm512_ಕೂಡಿಸು_epi32(acc, _mm512_mullo_epi32(vsum, vscale));
            }
            ಸ್ಥಿರಾಂಕ __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)(b_ptr + ಸ್ಥಾನಾಂತರ_d0)));
            vc[ಲಂಬಸಾಲು] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(vd0, vd1), vc[ಲಂಬಸಾಲು]);
        };

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < KB; ++i) {
            Unroll<ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ>{}(compute, i);
        }

        //store to C
        ಸ್ವಯಂಚಾಲಿತ storec = [&](ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + ಲಂಬಸಾಲು * 16), vc[ಲಂಬಸಾಲು]);
        };
        Unroll<ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ>{}(storec);
    }
};

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ ಖಂಡ_M, ಪೂರ್ಣಾಂಕ ಖಂಡ_N, ಪೂರ್ಣಾಂಕ ಖಂಡ_K>
ರಚನೆ tinygemm_kernel_vnni<ಖಂಡ_q8_K, ಖಂಡ_iq4_xs, ದಶಮಾಂಶ, ಖಂಡ_M, ಖಂಡ_N, ಖಂಡ_K> {
    ಸ್ಥಿರ ಶೂನ್ಯ apply(ಪೂರ್ಣಾಂಕ KB, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * RESTRICT _A, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * RESTRICT _B, ದಶಮಾಂಶ * RESTRICT C, ಪೂರ್ಣಾಂಕ ldc) {

        constexpr ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = ಖಂಡ_N / 16;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ TILE_ಗಾತ್ರ = TILE_N * ಗಾತ್ರ(ಖಂಡ_iq4_xs) + TILE_N * 2;

        ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K * RESTRICT A = static_cast<ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K *>(_A);
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * RESTRICT B = static_cast<ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *>(_B);

        // load the 256 bytes from A to 4 avx512 vectors
        __m512i va[4];
        __m512 vc[ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ];
        __m512 vd1;

        // packed_B:
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ_scales = (QK_K / 2) * TILE_N ;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ_d0     = (QK_K / 2) * TILE_N + 8 * TILE_N;

        // compensation
        __m512i vcomp;

        ಸ್ಥಿರಾಂಕ __m256i m128s = _mm256_set1_epi16(128);
        ಸ್ಥಿರಾಂಕ __m512i ಕೆಳಗಿನ_ಮುಖವಾಡ = _mm512_set1_epi8(0xF);

        ಸ್ಥಿರಾಂಕ __m512i values128 = mm512_ಗಣ_epi8(
            113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
            113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
            113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
            113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127
        );
        ಸ್ಥಿರಾಂಕ __m512i off = _mm512_set1_epi8(static_cast<ಅಕ್ಷರ>(0x80));
        ಸ್ಥಿರಾಂಕ __m512i values256 = mm512_ಕೂಡಿಸು_epi8(values128, off);

        ಸ್ವಯಂಚಾಲಿತ loadc = [&](ಸ್ವಯಂಚಾಲಿತ ಲಂಬಸಾಲು) {
            vc[ಲಂಬಸಾಲು] = _mm512_setzero_ps();
        };
        Unroll<ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ>{}(loadc);

        ಸ್ವಯಂಚಾಲಿತ compute = [&](ಸ್ವಯಂಚಾಲಿತ ಲಂಬಸಾಲು, ಸ್ವಯಂಚಾಲಿತ i) {
            ಆದರೆ constexpr (ಲಂಬಸಾಲು == 0) {
                // load a
                va[0] = _mm512_loadu_si512((ಸ್ಥಿರಾಂಕ __m512i *)(A[0 * KB + i].qs +   0));
                va[1] = _mm512_loadu_si512((ಸ್ಥಿರಾಂಕ __m512i *)(A[0 * KB + i].qs +  64));
                va[2] = _mm512_loadu_si512((ಸ್ಥಿರಾಂಕ __m512i *)(A[0 * KB + i].qs + 128));
                va[3] = _mm512_loadu_si512((ಸ್ಥಿರಾಂಕ __m512i *)(A[0 * KB + i].qs + 192));

                // compensation: 128 * A
                ಸ್ಥಿರಾಂಕ __m256i q8sums = _mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)A[0 * KB + i].bsums);
                vcomp = _mm512_castsi256_si512(_mm256_madd_epi16(q8sums, m128s));
                vd1 = _mm512_set1_ps(A[0 * KB + i].d);
            }

            // accmulate the quants
            __m512i acc = _mm512_setzero_si512();
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * b_ptr = B + PACKED_ಸೂಚ್ಯಂಕ(ಲಂಬಸಾಲು, i, KB, TILE_ಗಾತ್ರ);
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * b_qs = b_ptr;
            ಪೂರ್ಣಾಂಕ ಮುಖವಾಡ = 0;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k_ಗುಂಪು = 0; k_ಗುಂಪು < QK_K / 32; ++k_ಗುಂಪು) {
                ಪೂರ್ಣಾಂಕ r = k_ಗುಂಪು >> 1;
                __m512i vmask = _mm512_set1_epi32(k_ಗುಂಪು);
                __m512i vsum = _mm512_setzero_si512();
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 8; k += 2) {
                    __m512i va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(ಮುಖವಾಡ++), va[r]);
                    __m512i va1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(ಮುಖವಾಡ++), va[r]);

                    __m512i ಬೈಟುಗಳು = _mm512_loadu_si512(b_qs);
                    __m512i vb0 = _mm512_shuffle_epi8(values256, _mm512_and_si512(ಬೈಟುಗಳು, ಕೆಳಗಿನ_ಮುಖವಾಡ));
                    __m512i vb1 = _mm512_shuffle_epi8(values256, _mm512_and_si512(_mm512_srli_epi16(ಬೈಟುಗಳು, 4), ಕೆಳಗಿನ_ಮುಖವಾಡ));

                    vsum = _mm512_dpbusd_epi32(vsum, vb0, va0);
                    vsum = _mm512_dpbusd_epi32(vsum, vb1, va1);
                    b_qs += 64;
                }
                // (B + 128) * A - 128 * A
                vsum = _mm512_sub_epi32(vsum, _mm512_permutexvar_epi32(vmask, vcomp));

                // vacc += scale * (q8 @ q4)
                ಸ್ಥಿರಾಂಕ __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((ಸ್ಥಿರಾಂಕ __m128i *)(b_ptr + ಸ್ಥಾನಾಂತರ_scales + k_ಗುಂಪು * TILE_N)));
                acc = mm512_ಕೂಡಿಸು_epi32(acc, _mm512_mullo_epi32(vsum, vscale));
            }
            ಸ್ಥಿರಾಂಕ __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)(b_ptr + ಸ್ಥಾನಾಂತರ_d0)));
            vc[ಲಂಬಸಾಲು] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(vd0, vd1), vc[ಲಂಬಸಾಲು]);
        };

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < KB; ++i) {
            Unroll<ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ>{}(compute, i);
        }

        //store to C
        ಸ್ವಯಂಚಾಲಿತ storec = [&](ಸ್ವಯಂಚಾಲಿತ ಲಂಬಸಾಲು) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + ಲಂಬಸಾಲು * 16), vc[ಲಂಬಸಾಲು]);
        };
        Unroll<ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ>{}(storec);
    }
};

#ವ್ಯಾಖ್ಯಾನಿಸು LAUNCH_TINYGEMM_KERNEL_VNNI(NB_ಗಾತ್ರ)                                                   \
    tinygemm_kernel_vnni<vec_dot_type, type, ದಶಮಾಂಶ, 1, NB_ಗಾತ್ರ, blck_ಗಾತ್ರ>::apply(             \
        KB, wdata_batch,                                                                       \
        (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)src0->ದತ್ತಾಂಶ + src0_ಸ್ಥಾನಾಂತರ + PACKED_ಸೂಚ್ಯಂಕ(nb * kTilesN, 0, KB, TILE_ಗಾತ್ರ), \
        (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ + dst_ಸ್ಥಾನಾಂತರ + nb_ಆರಂಭ, ldc)

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು TA, ಬಗೆ_ಹೆಸರು TB, ಬಗೆ_ಹೆಸರು TC, ಪೂರ್ಣಾಂಕ ಖಂಡ_K,
          ಬಗೆ_ಹೆಸರು std::enable_if<!is_type_qkk<TB>::value, ಪೂರ್ಣಾಂಕ>::type = 0>
ಶೂನ್ಯ tinygemm_kernel_amx(ಪೂರ್ಣಾಂಕ M, ಪೂರ್ಣಾಂಕ N, ಪೂರ್ಣಾಂಕ KB, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * RESTRICT _A, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * RESTRICT _B, TC * RESTRICT C, ಪೂರ್ಣಾಂಕ ldc) {
    ಬಳಸಿ packed_B_t = packed_B_type<TB>;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ TILE_ಗಾತ್ರ = get_tile_ಗಾತ್ರ<TB>();
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ need_unpack = do_unpack<TB>::value;

    GGML_ASSERT(M <= 2 * TILE_M && N == 2 * TILE_N);
    ಸ್ಥಿರಾಂಕ TA * RESTRICT A = static_cast<ಸ್ಥಿರಾಂಕ TA *>(_A);
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * RESTRICT B = static_cast<ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *>(_B);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ m0 = std::min(M, TILE_M);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ m1 = std::max(M - TILE_M, 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ lda = KB * ಗಾತ್ರ(TA);
    //const int ldb = KB * sizeof(TB);

    alignas(64) ಸ್ಥಿರ thread_local packed_B_t Tile0[TILE_N * TILE_K];
    alignas(64) ಸ್ಥಿರ thread_local packed_B_t Tile1[TILE_N * TILE_K];
    alignas(64) ಸ್ಥಿರ thread_local int8_t Tile23[TILE_M * TILE_K];

    alignas(64) ಸ್ಥಿರ thread_local int32_t TileC0[TILE_M * TILE_N * 4];
    alignas(64) ಸ್ಥಿರ thread_local int32_t TileC1[TILE_M * TILE_N * 4];

    // double buffering C to interleave avx512 and amx
    int32_t * C_cur = TileC0;
    int32_t * C_pre = TileC1;

    ಸ್ವಯಂಚಾಲಿತ Tile4 = [&](int32_t * ತಳ) { ಹಿಂತಿರುಗಿ ತಳ; };
    ಸ್ವಯಂಚಾಲಿತ Tile5 = [&](int32_t * ತಳ) { ಹಿಂತಿರುಗಿ ತಳ + TILE_M * TILE_N; };
    ಸ್ವಯಂಚಾಲಿತ Tile6 = [&](int32_t * ತಳ) { ಹಿಂತಿರುಗಿ ತಳ + 2 * TILE_M * TILE_N; };
    ಸ್ವಯಂಚಾಲಿತ Tile7 = [&](int32_t * ತಳ) { ಹಿಂತಿರುಗಿ ತಳ + 3 * TILE_M * TILE_N; };

    ಆದರೆ (M == 2 * TILE_M) {
        // i = 0
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * B_blk0 = B + PACKED_ಸೂಚ್ಯಂಕ(0, 0, KB, TILE_ಗಾತ್ರ);
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * B_blk1 = B + PACKED_ಸೂಚ್ಯಂಕ(1, 0, KB, TILE_ಗಾತ್ರ);
        ಆದರೆ (need_unpack) {
            unpack_B<TB>(Tile0, B_blk0);
            _tile_loadd(TMM0, Tile0, TILE_N * VNNI_BLK);
        } ಇಲ್ಲದಿದ್ದರೆ {
            _tile_loadd(TMM0, B_blk0, TILE_N * VNNI_BLK);
        }

        _tile_zero(TMM4);
        _tile_loadd(TMM2, A[0].qs, lda);
        _tile_dpbssd(TMM4, TMM2, TMM0);
        _tile_stored(TMM4, Tile4(C_pre), TILE_N * ಗಾತ್ರ(int32_t));

        _tile_zero(TMM5);
        _tile_loadd(TMM3, A[TILE_M * KB + 0].qs, lda);
        _tile_dpbssd(TMM5, TMM3, TMM0);
        _tile_stored(TMM5, Tile5(C_pre), TILE_N * ಗಾತ್ರ(int32_t));

        ಆದರೆ (need_unpack) {
            unpack_B<TB>(Tile1, B_blk1);
            _tile_loadd(TMM1, Tile1, TILE_N * VNNI_BLK);
        } ಇಲ್ಲದಿದ್ದರೆ {
            _tile_loadd(TMM1, B_blk1, TILE_N * VNNI_BLK);
        }

        _tile_zero(TMM6);
        _tile_dpbssd(TMM6, TMM2, TMM1);
        _tile_stored(TMM6, Tile6(C_pre), TILE_N * ಗಾತ್ರ(int32_t));

        _tile_zero(TMM7);
        _tile_dpbssd(TMM7, TMM3, TMM1);
        _tile_stored(TMM7, Tile7(C_pre), TILE_N * ಗಾತ್ರ(int32_t));

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < KB; ++i) {
            // index of previous iter
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ii = i - 1;
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * B_blk0 = B + PACKED_ಸೂಚ್ಯಂಕ(0, i, KB, TILE_ಗಾತ್ರ);
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * B_blk1 = B + PACKED_ಸೂಚ್ಯಂಕ(1, i, KB, TILE_ಗಾತ್ರ);
            GGML_DISPATCH_BOOL(ii > 0, is_acc, [&] {
                ಆದರೆ (need_unpack) {
                    unpack_B<TB>(Tile0, B_blk0);
                    _tile_loadd(TMM0, Tile0, TILE_N * VNNI_BLK);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    _tile_loadd(TMM0, B_blk0, TILE_N * VNNI_BLK);
                }
                _tile_zero(TMM4);
                _tile_loadd(TMM2, A[i].qs, lda);
                acc_C<TA, TB, is_acc>::apply(C, ldc, Tile4(C_pre), &A[ii], KB, B + PACKED_ಸೂಚ್ಯಂಕ(0, ii, KB, TILE_ಗಾತ್ರ), TILE_M);

                _tile_dpbssd(TMM4, TMM2, TMM0);
                _tile_stored(TMM4, Tile4(C_cur), TILE_N * ಗಾತ್ರ(int32_t));

                _tile_zero(TMM5);
                _tile_loadd(TMM3, A[TILE_M * KB + i].qs, lda);
                acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc, ldc, Tile5(C_pre), &A[TILE_M * KB + ii], KB, B + PACKED_ಸೂಚ್ಯಂಕ(0, ii, KB, TILE_ಗಾತ್ರ), TILE_M);

                _tile_dpbssd(TMM5, TMM3, TMM0);
                _tile_stored(TMM5, Tile5(C_cur), TILE_N * ಗಾತ್ರ(int32_t));

                ಆದರೆ (need_unpack) {
                    unpack_B<TB>(Tile1, B_blk1);
                    _tile_loadd(TMM1, Tile1, TILE_N * VNNI_BLK);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    _tile_loadd(TMM1, B_blk1, TILE_N * VNNI_BLK);
                }
                _tile_zero(TMM6);
                acc_C<TA, TB, is_acc>::apply(C + TILE_N, ldc, Tile6(C_pre), &A[ii], KB, B + PACKED_ಸೂಚ್ಯಂಕ(1, ii, KB, TILE_ಗಾತ್ರ), TILE_M);

                _tile_dpbssd(TMM6, TMM2, TMM1);
                _tile_stored(TMM6, Tile6(C_cur), TILE_N * ಗಾತ್ರ(int32_t));

                _tile_zero(TMM7);
                acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc + TILE_N, ldc, Tile7(C_pre), &A[TILE_M * KB + ii], KB, B + PACKED_ಸೂಚ್ಯಂಕ(1, ii, KB, TILE_ಗಾತ್ರ), TILE_M);

                _tile_dpbssd(TMM7, TMM3, TMM1);
                _tile_stored(TMM7, Tile7(C_cur), TILE_N * ಗಾತ್ರ(int32_t));

                std::swap(C_cur, C_pre);
            });
        }
        // final accumulation
        {
            ಪೂರ್ಣಾಂಕ ii = KB - 1;
            acc_C<TA, TB, ನಿಜ>::apply(C, ldc, Tile4(C_pre), &A[ii], KB, B + PACKED_ಸೂಚ್ಯಂಕ(0, ii, KB, TILE_ಗಾತ್ರ), TILE_M);
            acc_C<TA, TB, ನಿಜ>::apply(C + TILE_M * ldc, ldc, Tile5(C_pre), &A[TILE_M * KB + ii], KB, B + PACKED_ಸೂಚ್ಯಂಕ(0, ii, KB, TILE_ಗಾತ್ರ), TILE_M);
            acc_C<TA, TB, ನಿಜ>::apply(C + TILE_N, ldc, Tile6(C_pre), &A[ii], KB, B + PACKED_ಸೂಚ್ಯಂಕ(1, ii, KB, TILE_ಗಾತ್ರ), TILE_M);
            acc_C<TA, TB, ನಿಜ>::apply(C + TILE_M * ldc + TILE_N, ldc, Tile7(C_pre), &A[TILE_M * KB + ii], KB, B + PACKED_ಸೂಚ್ಯಂಕ(1, ii, KB, TILE_ಗಾತ್ರ), TILE_M);
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < KB; ++i) {
            _tile_zero(TMM4);
            _tile_zero(TMM6);
            ಆದರೆ (m1 != 0) {
                _tile_zero(TMM5);
                _tile_zero(TMM7);
            }

            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * B_blk0 = B + PACKED_ಸೂಚ್ಯಂಕ(0, i, KB, TILE_ಗಾತ್ರ);
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * B_blk1 = B + PACKED_ಸೂಚ್ಯಂಕ(1, i, KB, TILE_ಗಾತ್ರ);
            ಆದರೆ (need_unpack) {
                unpack_B<TB>(Tile0, B_blk0);
                _tile_loadd(TMM0, Tile0, TILE_N * VNNI_BLK);
            } ಇಲ್ಲದಿದ್ದರೆ {
                _tile_loadd(TMM0, B_blk0, TILE_N * VNNI_BLK);
            }

            ಆದರೆ (need_unpack) {
                unpack_B<TB>(Tile1, B_blk1);
                _tile_loadd(TMM1, Tile1, TILE_N * VNNI_BLK);
            } ಇಲ್ಲದಿದ್ದರೆ {
                _tile_loadd(TMM1, B_blk1, TILE_N * VNNI_BLK);
            }

            ಆದರೆ (m0 == TILE_M) {
                _tile_loadd(TMM2, A[i].qs, lda);
            } ಇಲ್ಲದಿದ್ದರೆ {
                unpack_A(Tile23, &A[i], KB, m0);
                _tile_loadd(TMM2, Tile23, TILE_K);
            }

            _tile_dpbssd(TMM4, TMM2, TMM0);
            _tile_dpbssd(TMM6, TMM2, TMM1);

            _tile_stored(TMM4, Tile4(C_cur), TILE_N * ಗಾತ್ರ(int32_t));
            _tile_stored(TMM6, Tile6(C_cur), TILE_N * ಗಾತ್ರ(int32_t));

            GGML_DISPATCH_BOOL(i > 0, is_acc, [&] {
                acc_C<TA, TB, is_acc>::apply(C,          ldc, Tile4(C_cur), &A[i], KB, B + PACKED_ಸೂಚ್ಯಂಕ(0, i, KB, TILE_ಗಾತ್ರ), m0);
                acc_C<TA, TB, is_acc>::apply(C + TILE_N, ldc, Tile6(C_cur), &A[i], KB, B + PACKED_ಸೂಚ್ಯಂಕ(1, i, KB, TILE_ಗಾತ್ರ), m0);
            });

            ಆದರೆ (m1 != 0) {
                unpack_A(Tile23, &A[TILE_M * KB + i], KB, m1);
                _tile_loadd(TMM3, Tile23, TILE_K);

                _tile_dpbssd(TMM5, TMM3, TMM0);
                _tile_dpbssd(TMM7, TMM3, TMM1);
                _tile_stored(TMM5, Tile5(C_cur), TILE_N * ಗಾತ್ರ(int32_t));
                _tile_stored(TMM7, Tile7(C_cur), TILE_N * ಗಾತ್ರ(int32_t));
                GGML_DISPATCH_BOOL(i > 0, is_acc, [&] {
                    acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc,          ldc, Tile5(C_cur), &A[TILE_M * KB + i], KB, B + PACKED_ಸೂಚ್ಯಂಕ(0, i, KB, TILE_ಗಾತ್ರ), m1);
                    acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc + TILE_N, ldc, Tile7(C_cur), &A[TILE_M * KB + i], KB, B + PACKED_ಸೂಚ್ಯಂಕ(1, i, KB, TILE_ಗಾತ್ರ), m1);
                });
            }
        }
    }
    ಹಿಂತಿರುಗಿ;
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು TA, ಬಗೆ_ಹೆಸರು TB, ಬಗೆ_ಹೆಸರು TC, ಪೂರ್ಣಾಂಕ ಖಂಡ_K,
          ಬಗೆ_ಹೆಸರು std::enable_if<is_type_qkk<TB>::value, ಪೂರ್ಣಾಂಕ>::type = 0>
ಶೂನ್ಯ tinygemm_kernel_amx(ಪೂರ್ಣಾಂಕ M, ಪೂರ್ಣಾಂಕ N, ಪೂರ್ಣಾಂಕ KB, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * RESTRICT _A, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * RESTRICT _B, ದಶಮಾಂಶ * RESTRICT C, ಪೂರ್ಣಾಂಕ ldc) {
    static_assert(std::is_same<TA, ಖಂಡ_q8_K>::value);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ TILE_ಗಾತ್ರ = get_tile_ಗಾತ್ರ<TB>();

    GGML_ASSERT(M <= 2 * TILE_M && N == 2 * TILE_N);
    ಸ್ಥಿರಾಂಕ TA * RESTRICT A = static_cast<ಸ್ಥಿರಾಂಕ TA *>(_A);
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * RESTRICT B = static_cast<ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *>(_B);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ m0 = std::min(M, TILE_M);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ m1 = std::max(M - TILE_M, 0);
    //const int lda = KB * sizeof(TA);

    alignas(64) ಸ್ಥಿರ thread_local int8_t Tile0[TILE_N * TILE_K];
    alignas(64) ಸ್ಥಿರ thread_local int8_t Tile1[TILE_N * TILE_K];
    alignas(64) ಸ್ಥಿರ thread_local int8_t Tile23[TILE_M * TILE_K];

    // mat mul result for each group
    alignas(64) ಸ್ಥಿರ thread_local int32_t Tile4[TILE_M * TILE_N];
    alignas(64) ಸ್ಥಿರ thread_local int32_t Tile5[TILE_M * TILE_N];
    alignas(64) ಸ್ಥಿರ thread_local int32_t Tile6[TILE_M * TILE_N];
    alignas(64) ಸ್ಥಿರ thread_local int32_t Tile7[TILE_M * TILE_N];

    // sum of each QK_K block, contains 8 groups, int32
    alignas(64) ಸ್ಥಿರ thread_local int32_t Sumi4[TILE_M * TILE_N];
    alignas(64) ಸ್ಥಿರ thread_local int32_t Sumi5[TILE_M * TILE_N];
    alignas(64) ಸ್ಥಿರ thread_local int32_t Sumi6[TILE_M * TILE_N];
    alignas(64) ಸ್ಥಿರ thread_local int32_t Sumi7[TILE_M * TILE_N];

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k_ಗುಂಪು_ಗಾತ್ರ = std::is_same<TB, ಖಂಡ_q6_K>::value ? 16 : 32;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < KB; ++i) {
        // step 1: accumulate the quants across 8 groups, each group with 32
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < QK_K / k_ಗುಂಪು_ಗಾತ್ರ; ++k) {
            GGML_DISPATCH_BOOL(k > 0, is_acc, [&] {
                _tile_zero(TMM4);
                _tile_zero(TMM6);

                unpack_B<TB>(Tile0, B + PACKED_ಸೂಚ್ಯಂಕ(0, i, KB, TILE_ಗಾತ್ರ), k);
                _tile_loadd(TMM0, Tile0, TILE_N * VNNI_BLK);

                unpack_B<TB>(Tile1, B + PACKED_ಸೂಚ್ಯಂಕ(1, i, KB, TILE_ಗಾತ್ರ), k);
                _tile_loadd(TMM1, Tile1, TILE_N * VNNI_BLK);

                unpack_A<TB>(Tile23, &A[i], KB, k, m0);
                _tile_loadd(TMM2, Tile23, TILE_K);

                _tile_dpbssd(TMM4, TMM2, TMM0);
                _tile_dpbssd(TMM6, TMM2, TMM1);

                _tile_stored(TMM4, Tile4, TILE_N * ಗಾತ್ರ(int32_t));
                _tile_stored(TMM6, Tile6, TILE_N * ಗಾತ್ರ(int32_t));

                scale_C<TB, is_acc>(Tile4, Sumi4, B + PACKED_ಸೂಚ್ಯಂಕ(0, i, KB, TILE_ಗಾತ್ರ), k, m0);
                scale_C<TB, is_acc>(Tile6, Sumi6, B + PACKED_ಸೂಚ್ಯಂಕ(1, i, KB, TILE_ಗಾತ್ರ), k, m0);

                ಆದರೆ (m1 != 0) {
                    _tile_zero(TMM5);
                    _tile_zero(TMM7);

                    unpack_A<TB>(Tile23, &A[TILE_M * KB + i], KB, k, m1);
                    _tile_loadd(TMM3, Tile23, TILE_K);

                    _tile_dpbssd(TMM5, TMM3, TMM0);
                    _tile_dpbssd(TMM7, TMM3, TMM1);

                    _tile_stored(TMM5, Tile5, TILE_N * ಗಾತ್ರ(int32_t));
                    _tile_stored(TMM7, Tile7, TILE_N * ಗಾತ್ರ(int32_t));

                    scale_C<TB, is_acc>(Tile5, Sumi5, B + PACKED_ಸೂಚ್ಯಂಕ(0, i, KB, TILE_ಗಾತ್ರ), k, m1);
                    scale_C<TB, is_acc>(Tile7, Sumi7, B + PACKED_ಸೂಚ್ಯಂಕ(1, i, KB, TILE_ಗಾತ್ರ), k, m1);
                }
            });
        }

        // step 2: accmulate the mins
        GGML_DISPATCH_BOOL(i > 0, is_acc, [&] {
            acc_C<TA, TB, is_acc>::apply(C,          ldc, Sumi4, &A[i], KB, B + PACKED_ಸೂಚ್ಯಂಕ(0, i, KB, TILE_ಗಾತ್ರ), m0);
            acc_C<TA, TB, is_acc>::apply(C + TILE_N, ldc, Sumi6, &A[i], KB, B + PACKED_ಸೂಚ್ಯಂಕ(1, i, KB, TILE_ಗಾತ್ರ), m0);
            ಆದರೆ (m1 != 0) {
                acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc,          ldc, Sumi5, &A[TILE_M * KB + i], KB, B + PACKED_ಸೂಚ್ಯಂಕ(0, i, KB, TILE_ಗಾತ್ರ), m1);
                acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc + TILE_N, ldc, Sumi7, &A[TILE_M * KB + i], KB, B + PACKED_ಸೂಚ್ಯಂಕ(1, i, KB, TILE_ಗಾತ್ರ), m1);
            }
        });
    }
    ಹಿಂತಿರುಗಿ;
}

} // anonymous namespace

// get the packed tensor size for quantized weights
size_t ggml_backend_amx_get_alloc_ಗಾತ್ರ(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor) {
    ಸ್ಥಿರಾಂಕ ಎಣಿಕೆ ggml_type TYPE = tensor->type;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ K = tensor->ne[0]; // ne0: in_features
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ N = tensor->ne[1]; // ne1: out_features

    ಸ್ವಯಂಚಾಲಿತ get_tensor_ಗಾತ್ರ = [&] {
        size_t ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_B{0};
        GGML_DISPATCH_QTYPES(TYPE, [&] {
            ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_B = get_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ<type, blck_ಗಾತ್ರ>(K);
        });
        ಹಿಂತಿರುಗಿ N * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_B;
    };

    ಆದರೆ (qtype_has_amx_kernels(TYPE)) {
        ಹಿಂತಿರುಗಿ get_tensor_ಗಾತ್ರ();
    } ಇಲ್ಲದಿದ್ದರೆ {
        // for f16, bf16 we don't do packing
        ಹಿಂತಿರುಗಿ ggml_nbytes(tensor);
    }
}

// pack weight to vnni format
ಶೂನ್ಯ ggml_backend_amx_convert_ತೂಕ(ರಚನೆ ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    GGML_ASSERT(ಸ್ಥಾನಾಂತರ == 0 && size == ggml_nbytes(tensor)); // only full tensor conversion is supported for now

    ಸ್ಥಿರಾಂಕ ಎಣಿಕೆ ggml_type TYPE = tensor->type;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ K = tensor->ne[0]; // ne0: in_features
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ N = tensor->ne[1]; // ne1: out_features

    GGML_DISPATCH_QTYPES(TYPE, [&] {
        convert_B_packed_format<type, blck_ಗಾತ್ರ>((ಶೂನ್ಯ *)((ಅಕ್ಷರ *)tensor->ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ), (ಸ್ಥಿರಾಂಕ type *)ದತ್ತಾಂಶ, N, K);
    });
}

// ne2 is passed explicitly to help compiler optimize repeated calls
ಸಾಲೊಳಗೆ int64_t ggml_batch_ಸ್ಥಾನಾಂತರ(ಸ್ಥಿರಾಂಕ ggml_tensor * t, int64_t batch_idx, int64_t ne2) {
    ಸ್ಥಿರಾಂಕ int64_t i2 = batch_idx % ne2;
    ಸ್ಥಿರಾಂಕ int64_t i3 = batch_idx / ne2;
    ಹಿಂತಿರುಗಿ i3 * t->nb[3] + i2 * t->nb[2];
}

size_t ggml_backend_amx_desired_wsize(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst) {
    ರಚನೆ ggml_tensor * src0 = dst->src[0];

    ಸ್ಥಿರಾಂಕ ಎಣಿಕೆ ggml_type TYPE = src0->type;

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_floating_type = TYPE == GGML_TYPE_F16;
    ಆದರೆ (is_floating_type) {
        ಹಿಂತಿರುಗಿ 0;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ M = dst->ne[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ K = src0->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t n_batch = dst->ne[2] * dst->ne[3];

    size_t desired_wsize = 0;

    GGML_DISPATCH_QTYPES(TYPE, [&] {
        ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_A = K / blck_ಗಾತ್ರ * ಗಾತ್ರ(vec_dot_type);
        desired_wsize = n_batch * M * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_A;
    });

    ಹಿಂತಿರುಗಿ desired_wsize;
}

// NB: mixed dtype gemm with Advanced Matrix Extensions (Intel AMX)
//
// src0: weight in shape of {N, K}, quantized
// src1: input  in shape of {M, K}, float32
// dst:  output in shape of {M, N}, float32
//
// the function performs: dst = src1 @ src0.T for each batch
//
ಶೂನ್ಯ ggml_backend_amx_mul_mat(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst) {
    ರಚನೆ ggml_tensor * src0 = dst->src[0];
    ರಚನೆ ggml_tensor * src1 = dst->src[1];

    ಸ್ಥಿರಾಂಕ ಎಣಿಕೆ ggml_type TYPE = src0->type;

    // f16 only has avx512 kernels for now,
    // amx kernels will be added once 6th gen xeon is released.
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_floating_type = TYPE == GGML_TYPE_F16;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ M = dst->ne[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ N = dst->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ K = src0->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ldc = dst->nb[1] / dst->nb[0];

    ಸ್ಥಿರಾಂಕ int64_t ne2 = dst->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t n_batch = ne2 * dst->ne[3];

    ಆದರೆ (is_floating_type) {
        constexpr ಪೂರ್ಣಾಂಕ ಖಂಡ_M = 4;
        constexpr ಪೂರ್ಣಾಂಕ ಖಂಡ_N = 6;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ MB = div_ಮೇಲೆ(M, ಖಂಡ_M);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ NB = div_ಮೇಲೆ(N, ಖಂಡ_N);

        parallel_for_ggml(params, n_batch * MB * NB, [&](ಪೂರ್ಣಾಂಕ begin, ಪೂರ್ಣಾಂಕ end) {
            GGML_DISPATCH_FLOATING_TYPES(TYPE, [&] {
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = begin; i < end; ++i) {
                    ಪೂರ್ಣಾಂಕ batch_idx = i / (MB * NB);
                    ಪೂರ್ಣಾಂಕ ಉಳಿದ = i % (MB * NB);
                    ಪೂರ್ಣಾಂಕ mb = ಉಳಿದ / NB;
                    ಪೂರ್ಣಾಂಕ nb = ಉಳಿದ % NB;

                    int64_t src0_ಸ್ಥಾನಾಂತರ = ggml_batch_ಸ್ಥಾನಾಂತರ(src0, batch_idx, ne2);
                    int64_t src1_ಸ್ಥಾನಾಂತರ = ggml_batch_ಸ್ಥಾನಾಂತರ(src1, batch_idx, ne2);
                    int64_t dst_ಸ್ಥಾನಾಂತರ  = ggml_batch_ಸ್ಥಾನಾಂತರ(dst,  batch_idx, ne2);

                    ಪೂರ್ಣಾಂಕ mb_ಆರಂಭ = mb * ಖಂಡ_M;
                    ಪೂರ್ಣಾಂಕ mb_ಗಾತ್ರ = std::min(ಖಂಡ_M, M - mb_ಆರಂಭ);
                    ಪೂರ್ಣಾಂಕ nb_ಆರಂಭ = nb * ಖಂಡ_N;
                    ಪೂರ್ಣಾಂಕ nb_ಗಾತ್ರ = std::min(ಖಂಡ_N, N - nb_ಆರಂಭ);

                    ಆಯ್ಕೆ (mb_ಗಾತ್ರ << 4 | nb_ಗಾತ್ರ) {
                        ಸ್ಥಿತಿ 0x12: LAUNCH_TINYGEMM_KERNEL_AVX(1, 2); ನಿಲ್ಲಿಸು;
                        ಸ್ಥಿತಿ 0x14: LAUNCH_TINYGEMM_KERNEL_AVX(1, 4); ನಿಲ್ಲಿಸು;
                        ಸ್ಥಿತಿ 0x16: LAUNCH_TINYGEMM_KERNEL_AVX(1, 6); ನಿಲ್ಲಿಸು;
                        ಸ್ಥಿತಿ 0x22: LAUNCH_TINYGEMM_KERNEL_AVX(2, 2); ನಿಲ್ಲಿಸು;
                        ಸ್ಥಿತಿ 0x24: LAUNCH_TINYGEMM_KERNEL_AVX(2, 4); ನಿಲ್ಲಿಸು;
                        ಸ್ಥಿತಿ 0x26: LAUNCH_TINYGEMM_KERNEL_AVX(2, 6); ನಿಲ್ಲಿಸು;
                        ಸ್ಥಿತಿ 0x32: LAUNCH_TINYGEMM_KERNEL_AVX(3, 2); ನಿಲ್ಲಿಸು;
                        ಸ್ಥಿತಿ 0x34: LAUNCH_TINYGEMM_KERNEL_AVX(3, 4); ನಿಲ್ಲಿಸು;
                        ಸ್ಥಿತಿ 0x36: LAUNCH_TINYGEMM_KERNEL_AVX(3, 6); ನಿಲ್ಲಿಸು;
                        ಸ್ಥಿತಿ 0x42: LAUNCH_TINYGEMM_KERNEL_AVX(4, 2); ನಿಲ್ಲಿಸು;
                        ಸ್ಥಿತಿ 0x44: LAUNCH_TINYGEMM_KERNEL_AVX(4, 4); ನಿಲ್ಲಿಸು;
                        ಸ್ಥಿತಿ 0x46: LAUNCH_TINYGEMM_KERNEL_AVX(4, 6); ನಿಲ್ಲಿಸು;
                        ಪೂರ್ವನಿಯೋಜಿತ: fprintf(stderr, "Unexpected block size!\n");
                    }
                }
            });
        });
        ಹಿಂತಿರುಗಿ;
    }

    // pointer to work space, used convert A from float to quantized type
    ಶೂನ್ಯ * wdata = params->wdata;

    //TODO: performance improvement: merge quant A
 // if (params->ith == 0) {
        GGML_DISPATCH_QTYPES(TYPE, [&] {
            ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_A = K / blck_ಗಾತ್ರ * ಗಾತ್ರ(vec_dot_type);
            ಸ್ಥಿರಾಂಕ size_t desired_wsize = n_batch * M * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_A;
            ಆದರೆ (params->wsize < desired_wsize) {
                GGML_ABORT("insufficient work space size");
            }

            // Q4_0, Q4_1, Q8_0 handles 1 TILE_K per blck_size
            // Q4_K, Q5_K, Q6_K, IQ4_XS handles 8 TILE_K per blck_size
            GGML_ASSERT(TILE_K == blck_ಗಾತ್ರ || TILE_K * 8 == blck_ಗಾತ್ರ);

            parallel_for_ggml(params, n_batch * M, [&](ಪೂರ್ಣಾಂಕ begin, ಪೂರ್ಣಾಂಕ end) {
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ idx = begin; idx < end; ++idx) {
                    ಪೂರ್ಣಾಂಕ batch_idx = idx / M;
                    ಪೂರ್ಣಾಂಕ m         = idx % M;
                    int64_t src1_ಸ್ಥಾನಾಂತರ = ggml_batch_ಸ್ಥಾನಾಂತರ(src1, batch_idx, ne2);
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * A_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)src1->ದತ್ತಾಂಶ + src1_ಸ್ಥಾನಾಂತರ);
                    ಅಕ್ಷರ * wdata_batch = (ಅಕ್ಷರ *)wdata + batch_idx * M * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_A;
                    from_float<vec_dot_type>(A_ದತ್ತಾಂಶ + m * K, wdata_batch + m * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_A, K);
                }
            });
        });
 // }

    ggml_barrier(params->threadpool);

    ಆದರೆ (M == 1) {
        // MB = 1 and handle 8 tiles in each block
        constexpr ಪೂರ್ಣಾಂಕ kTilesN = 4;
        constexpr ಪೂರ್ಣಾಂಕ ಖಂಡ_N = TILE_N * kTilesN;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ NB = div_ಮೇಲೆ(N, ಖಂಡ_N);

        parallel_for_ggml(params, n_batch * NB, [&](ಪೂರ್ಣಾಂಕ begin, ಪೂರ್ಣಾಂಕ end) {
            GGML_DISPATCH_QTYPES(TYPE, [&] {
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ KB = K / blck_ಗಾತ್ರ;
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ TILE_ಗಾತ್ರ = get_tile_ಗಾತ್ರ<type>();
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_A = KB * ಗಾತ್ರ(vec_dot_type);
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = begin; i < end; ++i) {
                    ಪೂರ್ಣಾಂಕ batch_idx = i / NB;
                    ಪೂರ್ಣಾಂಕ nb = i % NB;

                    int64_t src0_ಸ್ಥಾನಾಂತರ = ggml_batch_ಸ್ಥಾನಾಂತರ(src0, batch_idx, ne2);
                    int64_t dst_ಸ್ಥಾನಾಂತರ  = ggml_batch_ಸ್ಥಾನಾಂತರ(dst,  batch_idx, ne2);
                    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * wdata_batch = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)wdata + batch_idx * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_A;

                    ಪೂರ್ಣಾಂಕ nb_ಆರಂಭ = nb * ಖಂಡ_N;
                    ಪೂರ್ಣಾಂಕ nb_ಗಾತ್ರ = std::min(ಖಂಡ_N, N - nb_ಆರಂಭ); // 32, 64, 96

                    ಆಯ್ಕೆ (nb_ಗಾತ್ರ) {
                        //case 160: LAUNCH_TINYGEMM_KERNEL_VNNI(160); break;
                        ಸ್ಥಿತಿ 128: LAUNCH_TINYGEMM_KERNEL_VNNI(128); ನಿಲ್ಲಿಸು;
                        ಸ್ಥಿತಿ 96: LAUNCH_TINYGEMM_KERNEL_VNNI(96); ನಿಲ್ಲಿಸು;
                        ಸ್ಥಿತಿ 64: LAUNCH_TINYGEMM_KERNEL_VNNI(64); ನಿಲ್ಲಿಸು;
                        ಸ್ಥಿತಿ 32: LAUNCH_TINYGEMM_KERNEL_VNNI(32); ನಿಲ್ಲಿಸು;
                        ಪೂರ್ವನಿಯೋಜಿತ: fprintf(stderr, "Unexpected n block size!\n");
                    }
                }
            });
        });
        ಹಿಂತಿರುಗಿ;
    }

    // handle 4 tiles at a tile
    constexpr ಪೂರ್ಣಾಂಕ ಖಂಡ_M = TILE_M * 2;
    constexpr ಪೂರ್ಣಾಂಕ ಖಂಡ_N = TILE_N * 2;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ MB = div_ಮೇಲೆ(M, ಖಂಡ_M);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ NB = div_ಮೇಲೆ(N, ಖಂಡ_N);

    parallel_for_ggml(params, n_batch * MB * NB, [&](ಪೂರ್ಣಾಂಕ begin, ಪೂರ್ಣಾಂಕ end) {
        // init tile config for each thread
        ggml_tile_config_init();

        GGML_DISPATCH_QTYPES(TYPE, [&] {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ KB = K / blck_ಗಾತ್ರ;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ TILE_ಗಾತ್ರ = get_tile_ಗಾತ್ರ<type>();
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_A = KB * ಗಾತ್ರ(vec_dot_type);

            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = begin; i < end; ++i) {
                ಪೂರ್ಣಾಂಕ batch_idx = i / (MB * NB);
                ಪೂರ್ಣಾಂಕ ಉಳಿದ = i % (MB * NB);
                ಪೂರ್ಣಾಂಕ mb = ಉಳಿದ / NB;
                ಪೂರ್ಣಾಂಕ nb = ಉಳಿದ % NB;

                int64_t src0_ಸ್ಥಾನಾಂತರ = ggml_batch_ಸ್ಥಾನಾಂತರ(src0, batch_idx, ne2);
                int64_t dst_ಸ್ಥಾನಾಂತರ  = ggml_batch_ಸ್ಥಾನಾಂತರ(dst,  batch_idx, ne2);
                ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * wdata_batch = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)wdata + batch_idx * M * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_A;

                ಪೂರ್ಣಾಂಕ mb_ಆರಂಭ = mb * ಖಂಡ_M;
                ಪೂರ್ಣಾಂಕ mb_ಗಾತ್ರ = std::min(ಖಂಡ_M, M - mb_ಆರಂಭ);
                ಪೂರ್ಣಾಂಕ nb_ಆರಂಭ = nb * ಖಂಡ_N;
                ಪೂರ್ಣಾಂಕ nb_ಗಾತ್ರ = ಖಂಡ_N;

                tinygemm_kernel_amx<vec_dot_type, type, ದಶಮಾಂಶ, blck_ಗಾತ್ರ>(
                    mb_ಗಾತ್ರ, nb_ಗಾತ್ರ, KB,
                    wdata_batch + mb_ಆರಂಭ * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_A,
                    (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)src0->ದತ್ತಾಂಶ + src0_ಸ್ಥಾನಾಂತರ + PACKED_ಸೂಚ್ಯಂಕ(nb * 2, 0, KB, TILE_ಗಾತ್ರ),
                    (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ + dst_ಸ್ಥಾನಾಂತರ + mb_ಆರಂಭ * N + nb_ಆರಂಭ, ldc);
            }
        });
    });
}

#ಆದರೆ_ಅಂತ್ಯ // if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
