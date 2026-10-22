#ਜੇ defined(__GNUC__)
#ਹਦਾਇਤ GCC diagnostic ignored "-Wpedantic"
#ਹਦਾਇਤ GCC diagnostic ignored "-Wunused-local-typedefs"
#ਜੇ_ਅੰਤ

#ਅਨ੍ਤਰ੍ਭਾਵ "amx.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "mmq.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-cpu-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "simd-mappings.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "quants.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-quants.h"
#ਅਨ੍ਤਰ੍ਭਾਵ <algorithm>
#ਅਨ੍ਤਰ੍ਭਾਵ <type_traits>

#ਜੇ defined(__gnu_linux__)
#ਅਨ੍ਤਰ੍ਭਾਵ <sys/syscall.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <unistd.h>
#ਜੇ_ਅੰਤ

#ਜੇ (defined(_WIN32) || defined(_WIN64))
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ RESTRICT __restrict
#ਨਹੀਂਤਾਂ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ RESTRICT __restrict__
#ਜੇ_ਅੰਤ

#ਜੇ (defined(_WIN32) || defined(_WIN64))
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ALWAYS_INLINE __forceinline
#ਜਾਂਜੇ __has_attribute(always_inline) || defined(__GNUC__)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ALWAYS_INLINE __attribute__((__always_inline__)) ਸਤਰਅੰਦਰ
#ਨਹੀਂਤਾਂ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ALWAYS_INLINE ਸਤਰਅੰਦਰ
#ਜੇ_ਅੰਤ

#ਜੇ defined(__AMX_INT8__) && defined(__AVX512VNNI__)

ਨਾਮਾਕਾਸ਼ {

// Forced unrolling
ਸਾਂਚਾ <ਪੂਰਨ n>
ਬਣਤਰ Unroll {
    ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ Func, ਕਿਸਮ_ਨਾਂ... ਦਲੀਲਾਂ>
    ALWAYS_INLINE ਖਾਲੀ ਸੰਚਾਲਕ()(ਅਚਲ Func& f, ਦਲੀਲਾਂ... ਦਲੀਲਾਂ_2) ਅਚਲ {
        Unroll<n - 1>{}(f, ਦਲੀਲਾਂ_2...);
        f(std::integral_constant<ਪੂਰਨ, n - 1>{}, ਦਲੀਲਾਂ_2...);
    }
};

ਸਾਂਚਾ <>
ਬਣਤਰ Unroll<1> {
    ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ Func, ਕਿਸਮ_ਨਾਂ... ਦਲੀਲਾਂ>
    ALWAYS_INLINE ਖਾਲੀ ਸੰਚਾਲਕ()(ਅਚਲ Func& f, ਦਲੀਲਾਂ... ਦਲੀਲਾਂ_2) ਅਚਲ {
        f(std::integral_constant<ਪੂਰਨ, 0>{}, ਦਲੀਲਾਂ_2...);
    }
};

// type traits
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T> ਬਣਤਰ PackedTypes {};
ਸਾਂਚਾ <> ਬਣਤਰ PackedTypes<ਖੰਡ_q4_0> { ਵਰਤੋਂ type = int8_t; };
ਸਾਂਚਾ <> ਬਣਤਰ PackedTypes<ਖੰਡ_q4_1> { ਵਰਤੋਂ type = uint8_t; };
ਸਾਂਚਾ <> ਬਣਤਰ PackedTypes<ਖੰਡ_q8_0> { ਵਰਤੋਂ type = int8_t; };
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T> ਵਰਤੋਂ packed_B_type = ਕਿਸਮ_ਨਾਂ PackedTypes<T>::type;

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
ਬਣਤਰ do_compensate : std::integral_constant<ਤਾਰਕਿਕ,
    std::is_same<T, ਖੰਡ_q8_0>::value> {};

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
ਬਣਤਰ do_unpack : std::integral_constant<ਤਾਰਕਿਕ,
    std::is_same<T, ਖੰਡ_q4_0>::value ||
    std::is_same<T, ਖੰਡ_q4_1>::value> {};

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
ਬਣਤਰ is_type_qkk : std::integral_constant<ਤਾਰਕਿਕ,
    std::is_same<T, ਖੰਡ_q4_K>::value ||
    std::is_same<T, ਖੰਡ_q5_K>::value ||
    std::is_same<T, ਖੰਡ_q6_K>::value ||
    std::is_same<T, ਖੰਡ_iq4_xs>::value> {};

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_DISPATCH_FLOATING_TYPES(TYPE, ...)                                        \
    [&] {                                                                              \
        ਚੋਣ (TYPE) {                                                                \
            ਹਾਲਤ GGML_TYPE_F16: {                                                      \
                ਵਰਤੋਂ type = ggml_fp16_t;                                              \
                constexpr ਪੂਰਨ blck_ਮਾਪ = 16;                                          \
                ਵਾਪਸ __VA_ARGS__();                                                  \
            }                                                                          \
            ਹਾਲਤ GGML_TYPE_BF16: {                                                     \
                ਵਰਤੋਂ type = ggml_bf16_t;                                              \
                constexpr ਪੂਰਨ blck_ਮਾਪ = 32;                                          \
                ਵਾਪਸ __VA_ARGS__();                                                  \
            }                                                                          \
            ਮੂਲ:                                                                   \
                fprintf(stderr, "Unsupported floating data type\n");                   \
        }                                                                              \
    }()

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_DISPATCH_QTYPES(QT, ...)                                                  \
    [&] {                                                                              \
        ਚੋਣ (QT) {                                                                  \
            ਹਾਲਤ GGML_TYPE_Q4_0: {                                                     \
                ਵਰਤੋਂ type = ਖੰਡ_q4_0;                                               \
                ਵਰਤੋਂ vec_dot_type = ਖੰਡ_q8_0;                                       \
                constexpr ਪੂਰਨ blck_ਮਾਪ = QK4_0;                                       \
                ਵਾਪਸ __VA_ARGS__();                                                  \
            }                                                                          \
            ਹਾਲਤ GGML_TYPE_Q4_1: {                                                     \
                ਵਰਤੋਂ type = ਖੰਡ_q4_1;                                               \
                ਵਰਤੋਂ vec_dot_type = ਖੰਡ_q8_1;                                       \
                constexpr ਪੂਰਨ blck_ਮਾਪ = QK4_1;                                       \
                ਵਾਪਸ __VA_ARGS__();                                                  \
            }                                                                          \
            ਹਾਲਤ GGML_TYPE_Q8_0: {                                                     \
                ਵਰਤੋਂ type = ਖੰਡ_q8_0;                                               \
                ਵਰਤੋਂ vec_dot_type = ਖੰਡ_q8_0;                                       \
                constexpr ਪੂਰਨ blck_ਮਾਪ = QK8_0;                                       \
                ਵਾਪਸ __VA_ARGS__();                                                  \
            }                                                                          \
            ਹਾਲਤ GGML_TYPE_Q4_K: {                                                     \
                ਵਰਤੋਂ type = ਖੰਡ_q4_K;                                               \
                ਵਰਤੋਂ vec_dot_type = ਖੰਡ_q8_K;                                       \
                constexpr ਪੂਰਨ blck_ਮਾਪ = QK_K;                                        \
                ਵਾਪਸ __VA_ARGS__();                                                  \
            }                                                                          \
            ਹਾਲਤ GGML_TYPE_Q5_K: {                                                     \
                ਵਰਤੋਂ type = ਖੰਡ_q5_K;                                               \
                ਵਰਤੋਂ vec_dot_type = ਖੰਡ_q8_K;                                       \
                constexpr ਪੂਰਨ blck_ਮਾਪ = QK_K;                                        \
                ਵਾਪਸ __VA_ARGS__();                                                  \
            }                                                                          \
            ਹਾਲਤ GGML_TYPE_Q6_K: {                                                     \
                ਵਰਤੋਂ type = ਖੰਡ_q6_K;                                               \
                ਵਰਤੋਂ vec_dot_type = ਖੰਡ_q8_K;                                       \
                constexpr ਪੂਰਨ blck_ਮਾਪ = QK_K;                                        \
                ਵਾਪਸ __VA_ARGS__();                                                  \
            }                                                                          \
            ਹਾਲਤ GGML_TYPE_IQ4_XS: {                                                   \
                ਵਰਤੋਂ type = ਖੰਡ_iq4_xs;                                             \
                ਵਰਤੋਂ vec_dot_type = ਖੰਡ_q8_K;                                       \
                constexpr ਪੂਰਨ blck_ਮਾਪ = QK_K;                                        \
                ਵਾਪਸ __VA_ARGS__();                                                  \
            }                                                                          \
            ਮੂਲ:                                                                   \
                fprintf(stderr, "Unsupported quantized data type: %d\n", ਪੂਰਨ(TYPE));   \
        }                                                                              \
    }()

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_DISPATCH_BOOL(BOOL_V, BOOL_NAME, ...)                                     \
    [&] {                                                                              \
        ਜੇ (BOOL_V) {                                                                  \
            constexpr ਤਾਰਕਿਕ BOOL_NAME = ਸੱਚ;                                           \
            ਵਾਪਸ __VA_ARGS__();                                                      \
        } ਨਹੀਂਤਾਂ {                                                                       \
            constexpr ਤਾਰਕਿਕ BOOL_NAME = ਝੂਠ;                                          \
            ਵਾਪਸ __VA_ARGS__();                                                      \
        }                                                                              \
    }()

// define amx tile config data structure
ਬਣਤਰ tile_config_t{
    uint8_t palette_id = 0;
    uint8_t ਸ਼ੁਰੂ_ਪੰਕਤੀ = 0;
    uint8_t reserved_0[14] = {0};
    uint16_t colsb[16] = {0};
    uint8_t ਪੰਕਤੀਆਂ[16] = {0};
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

ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_tile_config_init(ਖਾਲੀ) {
    ਸਥਿਰ thread_local ਤਾਰਕਿਕ ਪੂਰਾ = ਝੂਠ;

    ਜੇ (ਪੂਰਾ) {
        ਵਾਪਸ;
    }

    alignas(64) tile_config_t tc = {};
    tc.palette_id = 1;
    tc.ਸ਼ੁਰੂ_ਪੰਕਤੀ = 0;
    tc.ਪੰਕਤੀਆਂ[0] = 8;   tc.colsb[0] = 64;
    tc.ਪੰਕਤੀਆਂ[1] = 8;   tc.colsb[1] = 64;
    tc.ਪੰਕਤੀਆਂ[2] = 16;  tc.colsb[2] = 32;
    tc.ਪੰਕਤੀਆਂ[3] = 16;  tc.colsb[3] = 32;
    tc.ਪੰਕਤੀਆਂ[4] = 16;  tc.colsb[4] = 64;
    tc.ਪੰਕਤੀਆਂ[5] = 16;  tc.colsb[5] = 64;
    tc.ਪੰਕਤੀਆਂ[6] = 16;  tc.colsb[6] = 64;
    tc.ਪੰਕਤੀਆਂ[7] = 16;  tc.colsb[7] = 64;

    _tile_loadconfig(&tc);
    ਪੂਰਾ = ਸੱਚ;
}

// we need an extra 16 * 4B (TILE_N * int32_t) for each NB/KB block for compensation.
// See the notes `s8s8 igemm compensation in avx512-vnni` for detail.
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ TB>
ਪੂਰਨ get_tile_ਮਾਪ() {
    ਪੂਰਨ tile_ਮਾਪ = TILE_N * ਆਕਾਰ(TB);
    ਜੇ (do_compensate<TB>::value) {
        tile_ਮਾਪ += TILE_N * ਆਕਾਰ(int32_t);
    }
    ਜੇ (std::is_same<TB, ਖੰਡ_q4_K>::value ||
        std::is_same<TB, ਖੰਡ_q5_K>::value) {
        tile_ਮਾਪ += TILE_N * 4;
    }
    ਜੇ (std::is_same<TB, ਖੰਡ_iq4_xs>::value) {
        tile_ਮਾਪ += TILE_N * 2;
    }
    ਵਾਪਸ tile_ਮਾਪ;
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ TB, ਪੂਰਨ ਖੰਡ_K>
ਪੂਰਨ get_ਪੰਕਤੀ_ਮਾਪ(ਪੂਰਨ K) {
    ਪੂਰਨ KB = K / ਖੰਡ_K;
    ਪੂਰਨ ਪੰਕਤੀ_ਮਾਪ = KB * ਆਕਾਰ(TB);
    ਜੇ (do_compensate<TB>::value) {
        ਪੰਕਤੀ_ਮਾਪ += KB * ਆਕਾਰ(int32_t);
    }
    ਜੇ (std::is_same<TB, ਖੰਡ_q4_K>::value ||
        std::is_same<TB, ਖੰਡ_q5_K>::value) {
        ਪੰਕਤੀ_ਮਾਪ += KB * 4;
    }
    ਜੇ (std::is_same<TB, ਖੰਡ_iq4_xs>::value) {
        ਪੰਕਤੀ_ਮਾਪ += KB * 2;
    }
    ਵਾਪਸ ਪੰਕਤੀ_ਮਾਪ;
}

// transpose utils
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SHUFFLE_EPI32(a, b, ਨਕਾਬ) \
    _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b), ਨਕਾਬ))
ਸਤਰਅੰਦਰ ਖਾਲੀ transpose_8x8_32bit(__m256i * v, __m256i * v1) {
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

ਸਤਰਅੰਦਰ ਖਾਲੀ transpose_16x4_32bit(__m512i * r, __m512i * d) {

    ਸਥਿਰ ਅਚਲ __m512i index1 = mm512_ਸੈੱਟ_epi32(
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

ਸਤਰਅੰਦਰ ਖਾਲੀ transpose_16x16_32bit(__m512i * v) {
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

ਖਾਲੀ quantize_ਪੰਕਤੀ_q8_K_vnni(ਅਚਲ ਦਸ਼ਮਲਵ * RESTRICT x, ਖਾਲੀ * RESTRICT vy, int64_t k) {
    ਪ੍ਰਤਿਜ੍ਞਾ(k % QK_K == 0);
    ਅਚਲ ਪੂਰਨ KB = k / QK_K;
    constexpr ਪੂਰਨ kVecs = QK_K / 16;

    ਖੰਡ_q8_K * y = reinterpret_cast<ਖੰਡ_q8_K *>(vy);

    // hold 16 float vecs from x
    __m512  v[kVecs];

    // hold the quants vecs
    __m512i vq[kVecs / 4];

    // hold the packed quants vecs
    __m512i vq_packed[kVecs / 4];

    ਅਚਲ __m512 sign_ਦੋਆਧਾਰੀ_ਅੰਕ = _mm512_set1_ps(-0.f);

    ਲਈ (ਪੂਰਨ i = 0; i < KB; ++i) {
        // Compute max(abs(e)) for the block
        __m512 vamax = _mm512_set1_ps(0.f);
        ਲਈ (ਪੂਰਨ j = 0; j < kVecs; ++j) {
            v[j] = _mm512_loadu_ps(x); x += 16;
            vamax = _mm512_max_ps(vamax, _mm512_andnot_ps(sign_ਦੋਆਧਾਰੀ_ਅੰਕ, v[j]));
        }
        ਅਚਲ ਦਸ਼ਮਲਵ amax = _mm512_reduce_max_ps(vamax);

        // Quantize these floats
        ਅਚਲ ਦਸ਼ਮਲਵ iscale = 127.f / amax;
        y[i].d = GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(1 / iscale);
        ਅਚਲ ਦਸ਼ਮਲਵ id = ( amax != 0.0f ) ? iscale : 0.f;
        ਅਚਲ __m512 vscale = _mm512_set1_ps(id);

        // Apply multiplier and round to nearest integer
        ਲਈ (ਪੂਰਨ j = 0; j < kVecs; ++j) {
            v[j] = _mm512_mul_ps(v[j], vscale);
            v[j] = _mm512_roundscale_ps(v[j], (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        }

        // Pack to epi8 vecs
        ਲਈ (ਪੂਰਨ j = 0; j < kVecs / 4; ++j) {
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

        ਅਚਲ __m512i one = _mm512_set1_epi8(1);
        __m512i ਜੋੜ = _mm512_setzero_si512();
        ਲਈ (ਪੂਰਨ k = 0; k < 4; ++k) {
            ਜੋੜ = _mm512_dpbusd_epi32(ਜੋੜ, one, vq_packed[k]);
        }
        _mm256_storeu_si256((__m256i *)(y[i].bsums), _mm512_cvtepi32_epi16(ਜੋੜ));
    }
}

// quantize A from float to `vec_dot_type`
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
ਸਤਰਅੰਦਰ ਖਾਲੀ from_float(ਅਚਲ ਦਸ਼ਮਲਵ * x, ਅੱਖਰ * vy, int64_t k);

ਸਾਂਚਾ <>
ਸਤਰਅੰਦਰ ਖਾਲੀ from_float<ਖੰਡ_q8_0>(ਅਚਲ ਦਸ਼ਮਲਵ * x, ਅੱਖਰ * vy, int64_t k) {
    quantize_ਪੰਕਤੀ_q8_0(x, (ਖੰਡ_q8_0 *)vy, k);
}

ਸਾਂਚਾ <>
ਸਤਰਅੰਦਰ ਖਾਲੀ from_float<ਖੰਡ_q8_1>(ਅਚਲ ਦਸ਼ਮਲਵ * x, ਅੱਖਰ * vy, int64_t k) {
    quantize_ਪੰਕਤੀ_q8_1(x, (ਖੰਡ_q8_1 *)vy, k);
}

ਸਾਂਚਾ <>
ਸਤਰਅੰਦਰ ਖਾਲੀ from_float<ਖੰਡ_q8_K>(ਅਚਲ ਦਸ਼ਮਲਵ * x, ਅੱਖਰ * vy, int64_t k) {
#ਜੇ 1
    // TODO: this is reference impl!
    quantize_ਪੰਕਤੀ_q8_K_ref(x, (ਖੰਡ_q8_K *)vy, k);
#ਨਹੀਂਤਾਂ
    quantize_ਪੰਕਤੀ_q8_K_vnni(x, vy, k);
#ਜੇ_ਅੰਤ
}

// load A from memory to array when nrows can not fill in whole tile
ਖਾਲੀ unpack_A(int8_t * RESTRICT tile, ਅਚਲ ਖੰਡ_q8_0 * RESTRICT A, ਪੂਰਨ lda, ਪੂਰਨ nr) {
    ਪ੍ਰਤਿਜ੍ਞਾ(nr != TILE_M);
    ਲਈ (ਪੂਰਨ m = 0; m < nr; ++m) {
        ਅਚਲ __m256i v = _mm256_loadu_si256((ਅਚਲ __m256i *)(A[m * lda].qs));
        _mm256_storeu_si256((__m256i *)(tile + m * TILE_K), v);
    }
}

ਖਾਲੀ unpack_A(int8_t * RESTRICT tile, ਅਚਲ ਖੰਡ_q8_1 * RESTRICT A, ਪੂਰਨ lda, ਪੂਰਨ nr) {
    ਪ੍ਰਤਿਜ੍ਞਾ(nr != TILE_M);
    ਲਈ (ਪੂਰਨ m = 0; m < nr; ++m) {
        ਅਚਲ __m256i v = _mm256_loadu_si256((ਅਚਲ __m256i *)(A[m * lda].qs));
        _mm256_storeu_si256((__m256i *)(tile + m * TILE_K), v);
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ TB>
ਖਾਲੀ unpack_A(int8_t * RESTRICT tile, ਅਚਲ ਖੰਡ_q8_K * RESTRICT A, ਪੂਰਨ lda, ਪੂਰਨ k, ਪੂਰਨ nr) {
    ਪ੍ਰਤਿਜ੍ਞਾ(nr <= TILE_M);
    ਲਈ (ਪੂਰਨ m = 0; m < nr; ++m) {
        ਅਚਲ __m256i v = _mm256_loadu_si256((ਅਚਲ __m256i *)(A[m * lda].qs + k * 32));
        _mm256_storeu_si256((__m256i *)(tile + m * TILE_K), v);
    }
}

ਸਾਂਚਾ <>
ਖਾਲੀ unpack_A<ਖੰਡ_q6_K>(int8_t * RESTRICT tile, ਅਚਲ ਖੰਡ_q8_K * RESTRICT A, ਪੂਰਨ lda, ਪੂਰਨ k, ਪੂਰਨ nr) {
    ਪ੍ਰਤਿਜ੍ਞਾ(nr <= TILE_M);
    // zero padding k from 16 to 32, so that we don't have to re-config amx
    ਅਚਲ __m128i zero = _mm_setzero_si128();
    ਲਈ (ਪੂਰਨ m = 0; m < nr; ++m) {
        ਅਚਲ __m128i v = _mm_loadu_si128((ਅਚਲ __m128i *)(A[m * lda].qs + k * 16));
        ਅਚਲ __m256i r = _mm256_insertf128_si256(_mm256_castsi128_si256(v), zero, 1);
        _mm256_storeu_si256((__m256i *)(tile + m * TILE_K), r);
    }
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MM256_ਸੈੱਟ_M128_I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)
ਸਤਰਅੰਦਰ __m256i ਬਾਈਟਾਂ_from_nibbles_32(ਅਚਲ uint8_t * rsi) {
    ਅਚਲ __m128i tmp = _mm_loadu_si128((ਅਚਲ __m128i *)rsi);
    ਅਚਲ __m256i ਬਾਈਟਾਂ = MM256_ਸੈੱਟ_M128_I(_mm_srli_epi16(tmp, 4), tmp);
    ਅਚਲ __m256i ਨੀਵਾਂ_ਨਕਾਬ = _mm256_set1_epi8(0xF);
    ਵਾਪਸ _mm256_and_si256(ਨੀਵਾਂ_ਨਕਾਬ, ਬਾਈਟਾਂ);
}

// used for block_q4_K
ਸਤਰਅੰਦਰ __m512i ਬਾਈਟਾਂ_from_nibbles_64(ਅਚਲ uint8_t * rsi) {
    ਅਚਲ __m256i tmp = _mm256_loadu_si256((ਅਚਲ __m256i *)rsi);
    ਅਚਲ __m256i ਨੀਵਾਂ_ਨਕਾਬ = _mm256_set1_epi8(0xF);
    ਅਚਲ __m256i q4l = _mm256_and_si256(tmp, ਨੀਵਾਂ_ਨਕਾਬ);
    ਅਚਲ __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(tmp, 4), ਨੀਵਾਂ_ਨਕਾਬ);
    ਵਾਪਸ _mm512_inserti32x8(_mm512_castsi256_si512(q4l), q4h, 1);
}

// used for block_q5_K
ਸਤਰਅੰਦਰ __m512i ਬਾਈਟਾਂ_from_nibbles_64(ਅਚਲ uint8_t * qs, ਅਚਲ uint8_t * qh, ਪੂਰਨ k) {
    ਅਚਲ __m256i ਨੀਵਾਂ_ਨਕਾਬ = _mm256_set1_epi8(0xF);
    __m256i hmask = _mm256_set1_epi8(1);
    hmask = _mm256_slli_epi16(hmask, k);

    ਅਚਲ __m256i q5bits = _mm256_loadu_si256((ਅਚਲ __m256i *)qs);
    ਅਚਲ __m256i hbits = _mm256_loadu_si256((ਅਚਲ __m256i *)qh);

    ਅਚਲ __m256i q5l_0 = _mm256_and_si256(q5bits, ਨੀਵਾਂ_ਨਕਾਬ);
    ਅਚਲ __m256i q5h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), k + 0), 4);
    ਅਚਲ __m256i q5_0  = mm256_ਜੋੜੋ_epi8(q5l_0, q5h_0);
    hmask = _mm256_slli_epi16(hmask, 1);

    ਅਚਲ __m256i q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), ਨੀਵਾਂ_ਨਕਾਬ);
    ਅਚਲ __m256i q5h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), k + 1), 4);
    ਅਚਲ __m256i q5_1  = mm256_ਜੋੜੋ_epi8(q5l_1, q5h_1);

    ਵਾਪਸ _mm512_inserti32x8(_mm512_castsi256_si512(q5_0), q5_1, 1);
}

// used for block_q6_K
ਸਤਰਅੰਦਰ ਖਾਲੀ ਬਾਈਟਾਂ_from_nibbles_128(__m512i& r0, __m512i& r1, ਅਚਲ uint8_t * qs, ਅਚਲ uint8_t * qh) {
    ਅਚਲ __m256i m4 = _mm256_set1_epi8(0xF);
    ਅਚਲ __m256i m2 = _mm256_set1_epi8(0x3);

    ਅਚਲ __m256i q6bits1 = _mm256_loadu_si256((ਅਚਲ __m256i *)qs);
    ਅਚਲ __m256i q6bits2 = _mm256_loadu_si256((ਅਚਲ __m256i *)(qs + 32));
    ਅਚਲ __m256i q6bitsH = _mm256_loadu_si256((ਅਚਲ __m256i *)qh);

    ਅਚਲ __m256i q6h_0 = _mm256_slli_epi16(_mm256_and_si256(                  q6bitsH,     m2), 4);
    ਅਚਲ __m256i q6h_1 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q6bitsH, 2), m2), 4);
    ਅਚਲ __m256i q6h_2 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q6bitsH, 4), m2), 4);
    ਅਚਲ __m256i q6h_3 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q6bitsH, 6), m2), 4);

    ਅਚਲ __m256i q6_0 = _mm256_or_si256(_mm256_and_si256(q6bits1, m4), q6h_0);
    ਅਚਲ __m256i q6_1 = _mm256_or_si256(_mm256_and_si256(q6bits2, m4), q6h_1);
    ਅਚਲ __m256i q6_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q6bits1, 4), m4), q6h_2);
    ਅਚਲ __m256i q6_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q6bits2, 4), m4), q6h_3);

    r0 = _mm512_inserti32x8(_mm512_castsi256_si512(q6_0), q6_1, 1);
    r1 = _mm512_inserti32x8(_mm512_castsi256_si512(q6_2), q6_3, 1);
}

ਸਤਰਅੰਦਰ __m512i packNibbles(__m512i r0, __m512i r1) {
    ਵਾਪਸ _mm512_or_si512(r0, _mm512_slli_epi16(r1, 4));
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ TB>
ਸਤਰਅੰਦਰ ਖਾਲੀ pack_qs(ਖਾਲੀ * RESTRICT packed_B, ਅਚਲ TB * RESTRICT B, ਪੂਰਨ KB) {
    int8_t tmp[8 * 64];
    __m256i v[8], v2[8];
    ਲਈ (ਪੂਰਨ n = 0; n < 8; ++n) {
        v[n] = ਬਾਈਟਾਂ_from_nibbles_32(B[n * KB].qs);
    }
    transpose_8x8_32bit(v, v2);
    ਲਈ (ਪੂਰਨ n = 0; n < 8; ++n) {
        _mm256_storeu_si256((__m256i *)(tmp + n * 64), v2[n]);
    }
    ਲਈ (ਪੂਰਨ n = 0; n < 8; ++n) {
        v[n] = ਬਾਈਟਾਂ_from_nibbles_32(B[(n + 8) * KB].qs);
    }
    transpose_8x8_32bit(v, v2);
    ਲਈ (ਪੂਰਨ n = 0; n < 8; ++n) {
        _mm256_storeu_si256((__m256i *)(tmp + n * 64 + 32), v2[n]);
    }

    // pack again with 128 to fully utilize vector length
    ਲਈ (ਪੂਰਨ n = 0; n < 8; n += 2) {
        __m512i r0 = _mm512_loadu_si512((ਅਚਲ __m512i *)(tmp + n * 64));
        __m512i r1 = _mm512_loadu_si512((ਅਚਲ __m512i *)(tmp + n * 64 + 64));
        __m512i r1r0 = packNibbles(r0, r1);
        _mm512_storeu_si512((__m512i *)((ਅੱਖਰ *)packed_B + n * 32), r1r0);
    }
}

ਸਾਂਚਾ <>
ਸਤਰਅੰਦਰ ਖਾਲੀ pack_qs<ਖੰਡ_q8_0>(ਖਾਲੀ * RESTRICT packed_B, ਅਚਲ ਖੰਡ_q8_0 * RESTRICT B, ਪੂਰਨ KB) {
    __m256i v[8], v2[8];
    ਲਈ (ਪੂਰਨ n = 0; n < 8; ++n) {
        v[n] = _mm256_loadu_si256((ਅਚਲ __m256i *)(B[n * KB].qs));
    }
    transpose_8x8_32bit(v, v2);
    ਲਈ (ਪੂਰਨ n = 0; n < 8; ++n) {
        _mm256_storeu_si256((__m256i *)((ਅੱਖਰ *)packed_B + n * 64), v2[n]);
    }
    ਲਈ (ਪੂਰਨ n = 0; n < 8; ++n) {
        v[n] = _mm256_loadu_si256((ਅਚਲ __m256i *)(B[(n + 8) * KB].qs));
    }
    transpose_8x8_32bit(v, v2);
    ਲਈ (ਪੂਰਨ n = 0; n < 8; ++n) {
        _mm256_storeu_si256((__m256i *)((ਅੱਖਰ *)packed_B + n * 64 + 32), v2[n]);
    }
}

ਸਾਂਚਾ <>
ਸਤਰਅੰਦਰ ਖਾਲੀ pack_qs<ਖੰਡ_q4_K>(ਖਾਲੀ * RESTRICT packed_B, ਅਚਲ ਖੰਡ_q4_K * RESTRICT B, ਪੂਰਨ KB) {
    __m512i v[16];
    // QK_K 256 with 8 groups, handle 2 groups at a time
    ਅੱਖਰ * pb = (ਅੱਖਰ *)packed_B;
    ਲਈ (ਪੂਰਨ k = 0; k < QK_K / 64; ++k) {
        // pack 2 groups { n, g,  k} to {g, k/4, 4n}
        //          e.g. {16, 2, 32} to {2,   8, 64}
        ਲਈ (ਪੂਰਨ n = 0; n < TILE_N; ++n) {
            v[n] = ਬਾਈਟਾਂ_from_nibbles_64(B[n * KB].qs + k * 32);
        }

        transpose_16x16_32bit(v);

        // pack again with 128 to fully utilize vector length
        ਲਈ (ਪੂਰਨ n = 0; n < TILE_N; n += 2) {
            _mm512_storeu_si512((__m512i *)pb, packNibbles(v[n], v[n + 1]));
            pb += 64;
        }
    }
}

ਸਾਂਚਾ <>
ਸਤਰਅੰਦਰ ਖਾਲੀ pack_qs<ਖੰਡ_q5_K>(ਖਾਲੀ * RESTRICT packed_B, ਅਚਲ ਖੰਡ_q5_K * RESTRICT B, ਪੂਰਨ KB) {
    __m512i v[16];
    ਅਚਲ __m512i ਨੀਵਾਂ_ਨਕਾਬ = _mm512_set1_epi8(0xF);
    // QK_K 256 with 8 groups, handle 2 groups at a time
    ਅੱਖਰ * pb = (ਅੱਖਰ *)packed_B;
    ਅੱਖਰ * ph = (ਅੱਖਰ *)packed_B + (QK_K / 2) * TILE_N;
    ਲਈ (ਪੂਰਨ k = 0; k < QK_K / 64; ++k) {
        // pack 2 groups { n, g,  k} to {g, k/4, 4n}
        //          e.g. {16, 2, 32} to {2,   8, 64}
        ਲਈ (ਪੂਰਨ n = 0; n < TILE_N; ++n) {
            v[n] = ਬਾਈਟਾਂ_from_nibbles_64(B[n * KB].qs + k * 32, B[n * KB].qh, /*ਸਮੂਹ*/2 * k);
        }

        transpose_16x16_32bit(v);

        // 1. pack lower 4bits with 2 groups
        ਲਈ (ਪੂਰਨ n = 0; n < TILE_N; n += 2) {
            // get lower 4 bits
            ਅਚਲ __m512i r0 = _mm512_and_si512(v[n], ਨੀਵਾਂ_ਨਕਾਬ);
            ਅਚਲ __m512i r1 = _mm512_and_si512(v[n + 1], ਨੀਵਾਂ_ਨਕਾਬ);
            _mm512_storeu_si512((__m512i *)pb, packNibbles(r0, r1)); pb += 64;
        }

        // 2. pack higher 1bit with 2 groups
        ਅਚਲ __m512i hmask = _mm512_set1_epi8(0x10);
        ਲਈ (ਪੂਰਨ g = 0; g < 2; ++g) {
            __m512i hbits = _mm512_setzero_si512();
            hbits = mm512_ਜੋੜੋ_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 8 + 0], hmask), 4));
            hbits = mm512_ਜੋੜੋ_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 8 + 1], hmask), 3));
            hbits = mm512_ਜੋੜੋ_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 8 + 2], hmask), 2));
            hbits = mm512_ਜੋੜੋ_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 8 + 3], hmask), 1));
            hbits = mm512_ਜੋੜੋ_epi8(hbits,                   _mm512_and_si512(v[g * 8 + 4], hmask)    );
            hbits = mm512_ਜੋੜੋ_epi8(hbits, _mm512_slli_epi16(_mm512_and_si512(v[g * 8 + 5], hmask), 1));
            hbits = mm512_ਜੋੜੋ_epi8(hbits, _mm512_slli_epi16(_mm512_and_si512(v[g * 8 + 6], hmask), 2));
            hbits = mm512_ਜੋੜੋ_epi8(hbits, _mm512_slli_epi16(_mm512_and_si512(v[g * 8 + 7], hmask), 3));
            _mm512_storeu_si512((__m512i *)ph, hbits); ph += 64;
        }
    }
}

ਸਾਂਚਾ <>
ਸਤਰਅੰਦਰ ਖਾਲੀ pack_qs<ਖੰਡ_q6_K>(ਖਾਲੀ * RESTRICT packed_B, ਅਚਲ ਖੰਡ_q6_K * RESTRICT B, ਪੂਰਨ KB) {
    __m512i v[32];
    ਅਚਲ __m512i ਨੀਵਾਂ_ਨਕਾਬ = _mm512_set1_epi8(0xF);
    // QK_K 256 with 8 groups, handle 4 groups at a time
    ਅੱਖਰ * pb = (ਅੱਖਰ *)packed_B;
    ਅੱਖਰ * ph = (ਅੱਖਰ *)packed_B + (QK_K / 2) * TILE_N;
    ਲਈ (ਪੂਰਨ k = 0; k < QK_K / 128; ++k) {
        ਲਈ (ਪੂਰਨ n = 0; n < TILE_N; ++n) {
            ਬਾਈਟਾਂ_from_nibbles_128(v[n], v[n + 16], B[n * KB].ql + k * 64, B[n * KB].qh + k * 32);
        }

        // top half: group 0,1 or 4,5; bottom half: group 2,3 or 6,7
        transpose_16x16_32bit(v);
        transpose_16x16_32bit(v + 16);

        // 1. pack lower 4bits with 4 groups
        ਲਈ (ਪੂਰਨ n = 0; n < 32; n += 2) {
            ਅਚਲ __m512i r0 = _mm512_and_si512(v[n], ਨੀਵਾਂ_ਨਕਾਬ);
            ਅਚਲ __m512i r1 = _mm512_and_si512(v[n + 1], ਨੀਵਾਂ_ਨਕਾਬ);
            _mm512_storeu_si512((__m512i *)pb, packNibbles(r0, r1)); pb += 64;
        }

        // 2. pack higher 2bit with 4 groups
        ਅਚਲ __m512i hmask = _mm512_set1_epi8(0x30);
        ਲਈ (ਪੂਰਨ g = 0; g < 8; ++g) {
            __m512i hbits = _mm512_setzero_si512();
            hbits = mm512_ਜੋੜੋ_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 4 + 0], hmask), 4));
            hbits = mm512_ਜੋੜੋ_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 4 + 1], hmask), 2));
            hbits = mm512_ਜੋੜੋ_epi8(hbits,                   _mm512_and_si512(v[g * 4 + 2], hmask)    );
            hbits = mm512_ਜੋੜੋ_epi8(hbits, _mm512_slli_epi16(_mm512_and_si512(v[g * 4 + 3], hmask), 2));
            _mm512_storeu_si512((__m512i *)ph, hbits); ph += 64;
        }
    }
}

ਸਾਂਚਾ <>
ਸਤਰਅੰਦਰ ਖਾਲੀ pack_qs<ਖੰਡ_iq4_xs>(ਖਾਲੀ * RESTRICT packed_B, ਅਚਲ ਖੰਡ_iq4_xs * RESTRICT B, ਪੂਰਨ KB) {
    __m512i v[16];
    ਅੱਖਰ * pb = (ਅੱਖਰ *)packed_B;
    ਲਈ (ਪੂਰਨ k = 0; k < QK_K / 64; ++k) {
        ਲਈ (ਪੂਰਨ n = 0; n < TILE_N; ++n) {
            __m256i r0 = ਬਾਈਟਾਂ_from_nibbles_32(B[n * KB].qs + k * 32 +  0);
            __m256i r1 = ਬਾਈਟਾਂ_from_nibbles_32(B[n * KB].qs + k * 32 + 16);
            v[n] = _mm512_inserti32x8(_mm512_castsi256_si512(r0), r1, 1);
        }

        transpose_16x16_32bit(v);

        // pack again with 128 to fully utilize vector length
        ਲਈ (ਪੂਰਨ n = 0; n < TILE_N; n += 2) {
            _mm512_storeu_si512((__m512i *)pb, packNibbles(v[n], v[n + 1]));
            pb += 64;
        }
    }
}

// pack B to vnni formats in 4bits or 8 bits
ਖਾਲੀ pack_B(ਖਾਲੀ * RESTRICT packed_B, ਅਚਲ ਖੰਡ_q4_0 * RESTRICT B, ਪੂਰਨ KB) {
    pack_qs(packed_B, B, KB);
    ggml_half * d0 = reinterpret_cast<ggml_half *>((ਅੱਖਰ *)packed_B + TILE_N * TILE_K / 2);
    ਲਈ (ਪੂਰਨ n = 0; n < TILE_N; ++n) {
        d0[n] = B[n * KB].d;
    }
}

ਖਾਲੀ pack_B(ਖਾਲੀ * RESTRICT packed_B, ਅਚਲ ਖੰਡ_q4_1 * RESTRICT B, ਪੂਰਨ KB) {
    pack_qs(packed_B, B, KB);
    ggml_half * d0 = reinterpret_cast<ggml_half *>((ਅੱਖਰ *)packed_B + TILE_N * TILE_K / 2);
    ggml_half * m0 = d0 + TILE_N;
    ਲਈ (ਪੂਰਨ n = 0; n < TILE_N; ++n) {
        d0[n] = B[n * KB].d;
        m0[n] = B[n * KB].m;
    }
}

ਸਤਰਅੰਦਰ ਖਾਲੀ s8s8_compensation(ਖਾਲੀ * RESTRICT packed_B) {
    // packed_B layout:
    //   quants {TILE_N, TILEK}  int8_t
    //   d0     {TILE_N}      ggml_half
    //   comp   {TILE_N}        int32_t
    ਅਚਲ ਪੂਰਨ ਵਿੱਥ = TILE_N * TILE_K + TILE_N * ਆਕਾਰ(ggml_half);
    __m512i vcomp = _mm512_setzero_si512();
    ਅਚਲ __m512i off = _mm512_set1_epi8(static_cast<ਅੱਖਰ>(0x80));
    ਲਈ (ਪੂਰਨ k = 0; k < 8; ++k) {
        __m512i vb = _mm512_loadu_si512((ਅਚਲ __m512i *)((ਅਚਲ ਅੱਖਰ *)packed_B + k * 64));
        vcomp = _mm512_dpbusd_epi32(vcomp, off, vb);
    }
    _mm512_storeu_si512((__m512i *)((ਅੱਖਰ *)(packed_B) + ਵਿੱਥ), vcomp);
}

ਖਾਲੀ pack_B(ਖਾਲੀ * RESTRICT packed_B, ਅਚਲ ਖੰਡ_q8_0 * RESTRICT B, ਪੂਰਨ KB) {
    pack_qs(packed_B, B, KB);
    ggml_half * d0 = reinterpret_cast<ggml_half *>((ਅੱਖਰ *)packed_B + TILE_N * TILE_K);
    ਲਈ (ਪੂਰਨ n = 0; n < TILE_N; ++n) {
        d0[n] = B[n * KB].d;
    }
    s8s8_compensation(packed_B);
}

// convert 8 * {min, scale} from int6 to int8
ਸਤਰਅੰਦਰ ਖਾਲੀ unpack_mins_and_scales(ਅਚਲ uint8_t * scales, uint32_t * utmp) {
    ਅਚਲ uint32_t kmask1 = 0x3f3f3f3f;
    ਅਚਲ uint32_t kmask2 = 0x0f0f0f0f;
    ਅਚਲ uint32_t kmask3 = 0x03030303;

    ਸਮ੍ਰਿਤੀ_ਨਕਲ(utmp, scales, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    ਅਚਲ uint32_t uaux = utmp[1] & kmask1;
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
ਖਾਲੀ pack_B(ਖਾਲੀ * RESTRICT packed_B, ਅਚਲ ਖੰਡ_q4_K * RESTRICT B, ਪੂਰਨ KB) {
    pack_qs(packed_B, B, KB);

    uint8_t * scales = reinterpret_cast<uint8_t *>((ਅੱਖਰ *)packed_B + (QK_K / 2) * TILE_N);
    uint8_t * mins = scales + 8 * TILE_N;
    ggml_half * d = reinterpret_cast<ggml_half *>(mins + 8 * TILE_N);
    ggml_half * dmin = d + TILE_N;

    ਸੰਘ {
        uint32_t u32[4];
        uint8_t  u8[16];
    } s;

    ਲਈ (ਪੂਰਨ n = 0; n < TILE_N; ++n) {
        unpack_mins_and_scales(B[n * KB].scales, s.u32);
        ਲਈ (ਪੂਰਨ k = 0; k < 8; ++k) {
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
ਖਾਲੀ pack_B(ਖਾਲੀ * RESTRICT packed_B, ਅਚਲ ਖੰਡ_q5_K * RESTRICT B, ਪੂਰਨ KB) {
    pack_qs(packed_B, B, KB);

    uint8_t * scales = reinterpret_cast<uint8_t *>((ਅੱਖਰ *)packed_B + (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N);
    uint8_t * mins = scales + 8 * TILE_N;
    ggml_half * d = reinterpret_cast<ggml_half *>(mins + 8 * TILE_N);
    ggml_half * dmin = d + TILE_N;

    ਸੰਘ {
        uint32_t u32[4];
        uint8_t  u8[16];
    } s;

    ਲਈ (ਪੂਰਨ n = 0; n < TILE_N; ++n) {
        unpack_mins_and_scales(B[n * KB].scales, s.u32);
        ਲਈ (ਪੂਰਨ k = 0; k < 8; ++k) {
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
ਖਾਲੀ pack_B(ਖਾਲੀ * RESTRICT packed_B, ਅਚਲ ਖੰਡ_q6_K * RESTRICT B, ਪੂਰਨ KB) {
    pack_qs(packed_B, B, KB);

    uint8_t * scales = reinterpret_cast<uint8_t *>((ਅੱਖਰ *)packed_B + (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N);
    ggml_half * d = reinterpret_cast<ggml_half *>(scales + 16 * TILE_N);
    ਲਈ (ਪੂਰਨ n = 0; n < TILE_N; ++n) {
        ਅਚਲ int8_t * ps = B[n * KB].scales;
        ਲਈ (ਪੂਰਨ k = 0; k < 16; ++k) {
            scales[k * TILE_N + n] = ps[k];
        }
        d[n] = B[n * KB].d;
    }
}

// packed_B layout:
//   quants {8, TILE_N, 16}  uint8
//   scales {8, TILE_N}       int8
//   d      {TILE_N}     ggml_half
ਖਾਲੀ pack_B(ਖਾਲੀ * RESTRICT packed_B, ਅਚਲ ਖੰਡ_iq4_xs * RESTRICT B, ਪੂਰਨ KB) {
    pack_qs(packed_B, B, KB);

    int8_t * scales = reinterpret_cast<int8_t *>((ਅੱਖਰ *)packed_B + (QK_K / 2) * TILE_N);
    ggml_half * d = reinterpret_cast<ggml_half *>(scales + 8 * TILE_N);

    // pack the scales
    ਲਈ (ਪੂਰਨ n = 0; n < TILE_N; ++n) {
        uint16_t sh = B[n * KB].scales_h;
        ਲਈ (ਪੂਰਨ k = 0; k < 8; k += 2) {
            ਅਚਲ int16_t ls1 = ((B[n * KB].scales_l[k / 2] & 0xf) | ((sh << 4) & 0x30)) - 32;
            ਅਚਲ int16_t ls2 = ((B[n * KB].scales_l[k / 2] >>  4) | ((sh << 2) & 0x30)) - 32;
            scales[(k + 0) * TILE_N + n] = ls1;
            scales[(k + 1) * TILE_N + n] = ls2;
            sh >>= 4;
        }
        d[n] = B[n * KB].d;
    }
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ TB, ਕਿਸਮ_ਨਾਂ packed_B_t = packed_B_type<TB>>
ਖਾਲੀ unpack_B(packed_B_t * RESTRICT tile, ਅਚਲ ਖਾਲੀ * RESTRICT packed_B) {
    GGML_ਅਣਵਰਤਿਆ(tile);
    GGML_ਅਣਵਰਤਿਆ(packed_B);
}

ਸਾਂਚਾ <>
ਖਾਲੀ unpack_B<ਖੰਡ_q4_0>(int8_t * RESTRICT tile, ਅਚਲ ਖਾਲੀ * RESTRICT packed_B) {
  ਅਚਲ __m512i off = _mm512_set1_epi8(8);
  ਅਚਲ __m512i ਨੀਵਾਂ_ਨਕਾਬ = _mm512_set1_epi8(0xF);
  ਲਈ (ਪੂਰਨ n = 0; n < 8; n += 2) {
    __m512i ਬਾਈਟਾਂ = _mm512_loadu_si512((ਅਚਲ __m512i *)((ਅਚਲ ਅੱਖਰ *)packed_B + n * 32));
    ਅਚਲ __m512i r0 = _mm512_sub_epi8(_mm512_and_si512(ਬਾਈਟਾਂ, ਨੀਵਾਂ_ਨਕਾਬ), off);
    ਅਚਲ __m512i r1 = _mm512_sub_epi8(_mm512_and_si512(_mm512_srli_epi16(ਬਾਈਟਾਂ, 4), ਨੀਵਾਂ_ਨਕਾਬ), off);
    _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
    _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
  }
}

ਸਾਂਚਾ <>
ਖਾਲੀ unpack_B<ਖੰਡ_q4_1>(uint8_t * RESTRICT tile, ਅਚਲ ਖਾਲੀ * RESTRICT packed_B) {
    ਅਚਲ __m512i ਨੀਵਾਂ_ਨਕਾਬ = _mm512_set1_epi8(0xF);
    ਲਈ (ਪੂਰਨ n = 0; n < 8; n += 2) {
        __m512i ਬਾਈਟਾਂ = _mm512_loadu_si512((ਅਚਲ __m512i *)((ਅਚਲ ਅੱਖਰ *)packed_B + n * 32));
        ਅਚਲ __m512i r0 = _mm512_and_si512(ਬਾਈਟਾਂ, ਨੀਵਾਂ_ਨਕਾਬ);
        ਅਚਲ __m512i r1 = _mm512_and_si512(_mm512_srli_epi16(ਬਾਈਟਾਂ, 4), ਨੀਵਾਂ_ਨਕਾਬ);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
    }
}

// packed_B_t for QKK is int8_t
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ TB>
ਖਾਲੀ unpack_B(int8_t * RESTRICT tile, ਅਚਲ ਖਾਲੀ * RESTRICT packed_B, ਪੂਰਨ k) {
    ਅਚਲ ਪੂਰਨ packed_B_ਸਮੂਹ_ਮਾਪ = QK_K / 2 * TILE_N / 8;
    ਅਚਲ ਅੱਖਰ * packed_B_ਸਮੂਹ = (ਅਚਲ ਅੱਖਰ *)packed_B + k * packed_B_ਸਮੂਹ_ਮਾਪ;
    ਅਚਲ __m512i ਨੀਵਾਂ_ਨਕਾਬ = _mm512_set1_epi8(0xF);
    ਲਈ (ਪੂਰਨ n = 0; n < 8; n += 2) {
        __m512i ਬਾਈਟਾਂ = _mm512_loadu_si512(packed_B_ਸਮੂਹ + n * 32);
        ਅਚਲ __m512i r0 = _mm512_and_si512(ਬਾਈਟਾਂ, ਨੀਵਾਂ_ਨਕਾਬ);
        ਅਚਲ __m512i r1 = _mm512_and_si512(_mm512_srli_epi16(ਬਾਈਟਾਂ, 4), ਨੀਵਾਂ_ਨਕਾਬ);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
    }
}

ਸਾਂਚਾ <>
ਖਾਲੀ unpack_B<ਖੰਡ_q5_K>(int8_t * RESTRICT tile, ਅਚਲ ਖਾਲੀ * RESTRICT packed_B, ਪੂਰਨ k) {
    // lower 4bits, stride 256 bytes
    ਅਚਲ ਪੂਰਨ packed_l4_ਸਮੂਹ_ਮਾਪ = QK_K / 2 * TILE_N / 8;
    ਅਚਲ ਅੱਖਰ * pb = (ਅਚਲ ਅੱਖਰ *)packed_B + k * packed_l4_ਸਮੂਹ_ਮਾਪ;

    // higher 1bit, stride 64 bytes
    ਅਚਲ ਪੂਰਨ packed_h1_ਸਮੂਹ_ਮਾਪ = QK_K / 8 * TILE_N / 8;
    ਅਚਲ ਅੱਖਰ * ph = (ਅਚਲ ਅੱਖਰ *)packed_B + (QK_K / 2) * TILE_N + k * packed_h1_ਸਮੂਹ_ਮਾਪ;
    ਅਚਲ __m512i hbits = _mm512_loadu_si512(ph);

    ਅਚਲ __m512i ਨੀਵਾਂ_ਨਕਾਬ = _mm512_set1_epi8(0xF);
    __m512i hmask0 = _mm512_set1_epi8(0x1);
    __m512i hmask1 = _mm512_set1_epi8(0x2);

    ਲਈ (ਪੂਰਨ n = 0; n < 8; n += 2) {
        __m512i ਬਾਈਟਾਂ = _mm512_loadu_si512(pb + n * 32);
        __m512i r0 = _mm512_and_si512(ਬਾਈਟਾਂ, ਨੀਵਾਂ_ਨਕਾਬ);
        __m512i r1 = _mm512_and_si512(_mm512_srli_epi16(ਬਾਈਟਾਂ, 4), ਨੀਵਾਂ_ਨਕਾਬ);
        __m512i h0 = _mm512_slli_epi16(_mm512_srli_epi16(_mm512_and_si512(hbits, hmask0), n), 4);
        __m512i h1 = _mm512_slli_epi16(_mm512_srli_epi16(_mm512_and_si512(hbits, hmask1), n + 1), 4);

        hmask0 = _mm512_slli_epi16(hmask0, 2);
        hmask1 = _mm512_slli_epi16(hmask1, 2);
        r0 = mm512_ਜੋੜੋ_epi8(r0, h0);
        r1 = mm512_ਜੋੜੋ_epi8(r1, h1);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
    }
}

ਸਾਂਚਾ <>
ਖਾਲੀ unpack_B<ਖੰਡ_q6_K>(int8_t * RESTRICT tile, ਅਚਲ ਖਾਲੀ * RESTRICT packed_B, ਪੂਰਨ k) {
    // lower 4bits, stride 128 bytes
    ਅਚਲ ਪੂਰਨ packed_l4_ਸਮੂਹ_ਮਾਪ = QK_K / 2 * TILE_N / 16;
    ਅਚਲ ਅੱਖਰ * pb = (ਅਚਲ ਅੱਖਰ *)packed_B + k * packed_l4_ਸਮੂਹ_ਮਾਪ;

    // higher 2bits, stride 64 bytes
    ਅਚਲ ਪੂਰਨ packed_h2_ਸਮੂਹ_ਮਾਪ = QK_K / 4 * TILE_N / 16;
    ਅਚਲ ਅੱਖਰ * ph = (ਅਚਲ ਅੱਖਰ *)packed_B + (QK_K / 2) * TILE_N + k * packed_h2_ਸਮੂਹ_ਮਾਪ;
    ਅਚਲ __m512i hbits = _mm512_loadu_si512(ph);

    ਅਚਲ __m512i off = _mm512_set1_epi8(32);
    ਅਚਲ __m512i ਨੀਵਾਂ_ਨਕਾਬ = _mm512_set1_epi8(0xF);
    __m512i hmask0 = _mm512_set1_epi8(0x3); // 0011
    __m512i hmask1 = _mm512_set1_epi8(0xC); // 1100

    // notes: skip zero padding from row4 to row7 as we have done so in `unpack_A`
    __m512i ਬਾਈਟਾਂ = _mm512_loadu_si512(pb);
    __m512i r0 = _mm512_and_si512(ਬਾਈਟਾਂ, ਨੀਵਾਂ_ਨਕਾਬ);
    __m512i r1 = _mm512_and_si512(_mm512_srli_epi16(ਬਾਈਟਾਂ, 4), ਨੀਵਾਂ_ਨਕਾਬ);
    __m512i h0 = _mm512_slli_epi16(_mm512_and_si512(hbits, hmask0), 4);
    __m512i h1 = _mm512_slli_epi16(_mm512_and_si512(hbits, hmask1), 2);
    _mm512_storeu_si512((__m512i *)(tile +  0), _mm512_sub_epi8(mm512_ਜੋੜੋ_epi8(r0, h0), off));
    _mm512_storeu_si512((__m512i *)(tile + 64), _mm512_sub_epi8(mm512_ਜੋੜੋ_epi8(r1, h1), off));

    hmask0 = _mm512_slli_epi16(hmask0, 4);
    hmask1 = _mm512_slli_epi16(hmask1, 4);

    ਬਾਈਟਾਂ = _mm512_loadu_si512(pb + 64);
    r0 = _mm512_and_si512(ਬਾਈਟਾਂ, ਨੀਵਾਂ_ਨਕਾਬ);
    r1 = _mm512_and_si512(_mm512_srli_epi16(ਬਾਈਟਾਂ, 4), ਨੀਵਾਂ_ਨਕਾਬ);
    h0 =                   _mm512_and_si512(hbits, hmask0);
    h1 = _mm512_srli_epi16(_mm512_and_si512(hbits, hmask1), 2);
    _mm512_storeu_si512((__m512i *)(tile + 128), _mm512_sub_epi8(mm512_ਜੋੜੋ_epi8(r0, h0), off));
    _mm512_storeu_si512((__m512i *)(tile + 192), _mm512_sub_epi8(mm512_ਜੋੜੋ_epi8(r1, h1), off));
}

ਸਾਂਚਾ <>
ਖਾਲੀ unpack_B<ਖੰਡ_iq4_xs>(int8_t * RESTRICT tile, ਅਚਲ ਖਾਲੀ * RESTRICT packed_B, ਪੂਰਨ k) {
    ਸਥਿਰ ਅਚਲ __m512i values128 = mm512_ਸੈੱਟ_epi8(
        113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
        113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
        113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
        113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127
    );

    ਅਚਲ ਪੂਰਨ packed_B_ਸਮੂਹ_ਮਾਪ = QK_K / 2 * TILE_N / 8;
    ਅਚਲ ਅੱਖਰ * pb = (ਅਚਲ ਅੱਖਰ *)packed_B + k * packed_B_ਸਮੂਹ_ਮਾਪ;
    ਅਚਲ __m512i ਨੀਵਾਂ_ਨਕਾਬ = _mm512_set1_epi8(0xF);

    ਲਈ (ਪੂਰਨ n = 0; n < 8; n += 2) {
        __m512i ਬਾਈਟਾਂ = _mm512_loadu_si512(pb + n * 32);
        ਅਚਲ __m512i r0 = _mm512_shuffle_epi8(values128, _mm512_and_si512(ਬਾਈਟਾਂ, ਨੀਵਾਂ_ਨਕਾਬ));
        ਅਚਲ __m512i r1 = _mm512_shuffle_epi8(values128, _mm512_and_si512(_mm512_srli_epi16(ਬਾਈਟਾਂ, 4), ਨੀਵਾਂ_ਨਕਾਬ));
        _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ TA, ਕਿਸਮ_ਨਾਂ TB, ਤਾਰਕਿਕ is_acc>
ਬਣਤਰ acc_C {};

ਸਾਂਚਾ <ਤਾਰਕਿਕ is_acc>
ਬਣਤਰ acc_C<ਖੰਡ_q8_0, ਖੰਡ_q4_0, is_acc> {
    ਸਥਿਰ ਖਾਲੀ apply(ਦਸ਼ਮਲਵ * RESTRICT C, ਪੂਰਨ ldc, ਅਚਲ int32_t * RESTRICT tile, ਅਚਲ ਖੰਡ_q8_0 * A, ਪੂਰਨ lda, ਅਚਲ ਖਾਲੀ * packed_B, ਪੂਰਨ nr) {
        ਅਚਲ ਪੂਰਨ ਵਿੱਥ = TILE_N * TILE_K / 2;
        ਅਚਲ __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ਅਚਲ __m256i *)((ਅਚਲ ਅੱਖਰ *)packed_B + ਵਿੱਥ)));

        ਲਈ (ਪੂਰਨ m = 0; m < nr; ++m) {
            ਅਚਲ __m512 vd1 = _mm512_set1_ps(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(A[m * lda].d));
            ਅਚਲ __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            ਜੇ (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } ਨਹੀਂਤਾਂ {
                vsum = _mm512_set1_ps(0.f);
            }
            vsum = _mm512_fmadd_ps(vtile, _mm512_mul_ps(vd0, vd1), vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

ਸਾਂਚਾ <ਤਾਰਕਿਕ is_acc>
ਬਣਤਰ acc_C<ਖੰਡ_q8_1, ਖੰਡ_q4_1, is_acc> {
    ਸਥਿਰ ਖਾਲੀ apply(ਦਸ਼ਮਲਵ * RESTRICT C, ਪੂਰਨ ldc, ਅਚਲ int32_t * RESTRICT tile, ਅਚਲ ਖੰਡ_q8_1 * A, ਪੂਰਨ lda, ਅਚਲ ਖਾਲੀ * packed_B, ਪੂਰਨ nr) {
        ਅਚਲ ਪੂਰਨ ਵਿੱਥ = TILE_N * TILE_K / 2;
        ਅਚਲ __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ਅਚਲ __m256i *)((ਅਚਲ ਅੱਖਰ *)packed_B + ਵਿੱਥ)));
        ਅਚਲ __m512 vm0 = _mm512_cvtph_ps(_mm256_loadu_si256((ਅਚਲ __m256i *)((ਅਚਲ ਅੱਖਰ *)packed_B + ਵਿੱਥ + TILE_N * ਆਕਾਰ(ggml_half))));

        ਲਈ (ਪੂਰਨ m = 0; m < nr; ++m) {
            ਅਚਲ __m512 vd1 = _mm512_set1_ps(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(A[m * lda].d));
            ਅਚਲ __m512 vs1 = _mm512_set1_ps(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(A[m * lda].s));
            ਅਚਲ __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            ਜੇ (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } ਨਹੀਂਤਾਂ {
                vsum = _mm512_set1_ps(0.f);
            }
            vsum = _mm512_fmadd_ps(vtile, _mm512_mul_ps(vd0, vd1), vsum);
            vsum = _mm512_fmadd_ps(vm0, vs1, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

ਸਾਂਚਾ <ਤਾਰਕਿਕ is_acc>
ਬਣਤਰ acc_C<ਖੰਡ_q8_0, ਖੰਡ_q8_0, is_acc> {
    ਸਥਿਰ ਖਾਲੀ apply(ਦਸ਼ਮਲਵ * RESTRICT C, ਪੂਰਨ ldc, ਅਚਲ int32_t * RESTRICT tile, ਅਚਲ ਖੰਡ_q8_0 * A, ਪੂਰਨ lda, ਅਚਲ ਖਾਲੀ * packed_B, ਪੂਰਨ nr) {
        ਅਚਲ ਪੂਰਨ ਵਿੱਥ = TILE_N * TILE_K;
        ਅਚਲ __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ਅਚਲ __m256i *)((ਅਚਲ ਅੱਖਰ *)packed_B + ਵਿੱਥ)));

        ਲਈ (ਪੂਰਨ m = 0; m < nr; ++m) {
            ਅਚਲ __m512 vd1 = _mm512_set1_ps(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(A[m * lda].d));
            ਅਚਲ __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            ਜੇ (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } ਨਹੀਂਤਾਂ {
                vsum = _mm512_set1_ps(0.f);
            }
            vsum = _mm512_fmadd_ps(vtile, _mm512_mul_ps(vd0, vd1), vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

ਸਾਂਚਾ <ਤਾਰਕਿਕ is_acc>
ਬਣਤਰ acc_C<ਖੰਡ_q8_K, ਖੰਡ_q4_K, is_acc> {
    ਸਥਿਰ ਖਾਲੀ apply(ਦਸ਼ਮਲਵ * RESTRICT C, ਪੂਰਨ ldc, ਅਚਲ int32_t * RESTRICT tile, ਅਚਲ ਖੰਡ_q8_K * A, ਪੂਰਨ lda, ਅਚਲ ਖਾਲੀ * packed_B, ਪੂਰਨ nr) {
        ਅਚਲ uint8_t * scales = reinterpret_cast<ਅਚਲ uint8_t *>((ਅਚਲ ਅੱਖਰ *)packed_B + (QK_K / 2) * TILE_N);
        ਅਚਲ uint8_t * mins = scales + 8 * TILE_N;
        ਅਚਲ ggml_half * d0 = reinterpret_cast<ਅਚਲ ggml_half *>(mins + 8 * TILE_N);
        ਅਚਲ ggml_half * dmin = d0 + TILE_N;

        ਅਚਲ __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ਅਚਲ __m256i *)d0));
        ਅਚਲ __m512 vdmin = _mm512_cvtph_ps(_mm256_loadu_si256((ਅਚਲ __m256i *)dmin));

        ਲਈ (ਪੂਰਨ m = 0; m < nr; ++m) {
            ਅਚਲ ਦਸ਼ਮਲਵ d1 = A[m * lda].d;
            ਅਚਲ __m512 vd = _mm512_mul_ps(_mm512_set1_ps(d1), vd0);
            ਅਚਲ __m512 vdm = _mm512_mul_ps(_mm512_set1_ps(-d1), vdmin);
            ਅਚਲ __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            ਜੇ (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } ਨਹੀਂਤਾਂ {
                vsum = _mm512_set1_ps(0.f);
            }

            ਅਚਲ __m256i q8sums = _mm256_loadu_si256((ਅਚਲ __m256i *)A[m * lda].bsums);
            ਅਚਲ __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));

            __m512i acc_m = _mm512_setzero_si512();
            ਲਈ (ਪੂਰਨ k = 0; k < 4; ++k) {
                __m512i vmask = _mm512_set1_epi32(k);
                __m512i va = _mm512_permutexvar_epi32(vmask, _mm512_castsi128_si512(q8s));
                __m512i vb = _mm512_cvtepi8_epi16(_mm256_loadu_si256((ਅਚਲ __m256i *)(mins + k * 32)));
                acc_m = _mm512_dpwssds_epi32(acc_m, va, vb);
            }

            vsum = _mm512_fmadd_ps(vtile, vd, vsum);
            vsum = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc_m), vdm, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

ਸਾਂਚਾ <ਤਾਰਕਿਕ is_acc>
ਬਣਤਰ acc_C<ਖੰਡ_q8_K, ਖੰਡ_q5_K, is_acc> {
    ਸਥਿਰ ਖਾਲੀ apply(ਦਸ਼ਮਲਵ * RESTRICT C, ਪੂਰਨ ldc, ਅਚਲ int32_t * RESTRICT tile, ਅਚਲ ਖੰਡ_q8_K * A, ਪੂਰਨ lda, ਅਚਲ ਖਾਲੀ * packed_B, ਪੂਰਨ nr) {
        ਅਚਲ uint8_t * scales = reinterpret_cast<ਅਚਲ uint8_t *>((ਅਚਲ ਅੱਖਰ *)packed_B + (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N);
        ਅਚਲ uint8_t * mins = scales + 8 * TILE_N;
        ਅਚਲ ggml_half * d0 = reinterpret_cast<ਅਚਲ ggml_half *>(mins + 8 * TILE_N);
        ਅਚਲ ggml_half * dmin = d0 + TILE_N;

        ਅਚਲ __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ਅਚਲ __m256i *)d0));
        ਅਚਲ __m512 vdmin = _mm512_cvtph_ps(_mm256_loadu_si256((ਅਚਲ __m256i *)dmin));

        ਲਈ (ਪੂਰਨ m = 0; m < nr; ++m) {
            ਅਚਲ ਦਸ਼ਮਲਵ d1 = A[m * lda].d;
            ਅਚਲ __m512 vd = _mm512_mul_ps(_mm512_set1_ps(d1), vd0);
            ਅਚਲ __m512 vdm = _mm512_mul_ps(_mm512_set1_ps(-d1), vdmin);
            ਅਚਲ __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            ਜੇ (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } ਨਹੀਂਤਾਂ {
                vsum = _mm512_set1_ps(0.f);
            }

            ਅਚਲ __m256i q8sums = _mm256_loadu_si256((ਅਚਲ __m256i *)A[m * lda].bsums);
            ਅਚਲ __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));

            __m512i acc_m = _mm512_setzero_si512();
            ਲਈ (ਪੂਰਨ k = 0; k < 4; ++k) {
                __m512i vmask = _mm512_set1_epi32(k);
                __m512i va = _mm512_permutexvar_epi32(vmask, _mm512_castsi128_si512(q8s));
                __m512i vb = _mm512_cvtepi8_epi16(_mm256_loadu_si256((ਅਚਲ __m256i *)(mins + k * 32)));
                acc_m = _mm512_dpwssds_epi32(acc_m, va, vb);
            }

            vsum = _mm512_fmadd_ps(vtile, vd, vsum);
            vsum = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc_m), vdm, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

ਸਾਂਚਾ <ਤਾਰਕਿਕ is_acc>
ਬਣਤਰ acc_C<ਖੰਡ_q8_K, ਖੰਡ_q6_K, is_acc> {
    ਸਥਿਰ ਖਾਲੀ apply(ਦਸ਼ਮਲਵ * RESTRICT C, ਪੂਰਨ ldc, ਅਚਲ int32_t * RESTRICT tile, ਅਚਲ ਖੰਡ_q8_K * A, ਪੂਰਨ lda, ਅਚਲ ਖਾਲੀ * packed_B, ਪੂਰਨ nr) {
        ਅਚਲ uint8_t * scales = reinterpret_cast<ਅਚਲ uint8_t *>((ਅਚਲ ਅੱਖਰ *)packed_B + (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N);
        ਅਚਲ ggml_half * d0 = reinterpret_cast<ਅਚਲ ggml_half *>(scales + 16 * TILE_N);

        ਅਚਲ __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ਅਚਲ __m256i *)d0));

        ਲਈ (ਪੂਰਨ m = 0; m < nr; ++m) {
            ਅਚਲ ਦਸ਼ਮਲਵ d1 = A[m * lda].d;
            ਅਚਲ __m512 vd = _mm512_mul_ps(_mm512_set1_ps(d1), vd0);
            ਅਚਲ __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            ਜੇ (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } ਨਹੀਂਤਾਂ {
                vsum = _mm512_set1_ps(0.f);
            }

            vsum = _mm512_fmadd_ps(vtile, vd, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

ਸਾਂਚਾ <ਤਾਰਕਿਕ is_acc>
ਬਣਤਰ acc_C<ਖੰਡ_q8_K, ਖੰਡ_iq4_xs, is_acc> {
    ਸਥਿਰ ਖਾਲੀ apply(ਦਸ਼ਮਲਵ * RESTRICT C, ਪੂਰਨ ldc, ਅਚਲ int32_t * RESTRICT tile, ਅਚਲ ਖੰਡ_q8_K * A, ਪੂਰਨ lda, ਅਚਲ ਖਾਲੀ * packed_B, ਪੂਰਨ nr) {
        ਅਚਲ int8_t * scales = reinterpret_cast<ਅਚਲ int8_t *>((ਅਚਲ ਅੱਖਰ *)packed_B + (QK_K / 2) * TILE_N);
        ਅਚਲ ggml_half * d0 = reinterpret_cast<ਅਚਲ ggml_half *>(scales + 8 * TILE_N);

        ਅਚਲ __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ਅਚਲ __m256i *)d0));

        ਲਈ (ਪੂਰਨ m = 0; m < nr; ++m) {
            ਅਚਲ ਦਸ਼ਮਲਵ d1 = A[m * lda].d;
            ਅਚਲ __m512 vd = _mm512_mul_ps(_mm512_set1_ps(d1), vd0);
            ਅਚਲ __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            ਜੇ (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } ਨਹੀਂਤਾਂ {
                vsum = _mm512_set1_ps(0.f);
            }

            vsum = _mm512_fmadd_ps(vtile, vd, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ TB> constexpr ਪੂਰਨ get_quants_ਮਾਪ();
ਸਾਂਚਾ <> constexpr ਪੂਰਨ get_quants_ਮਾਪ<ਖੰਡ_q4_K>() { ਵਾਪਸ (QK_K / 2) * TILE_N; }
ਸਾਂਚਾ <> constexpr ਪੂਰਨ get_quants_ਮਾਪ<ਖੰਡ_q5_K>() { ਵਾਪਸ (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N; }
ਸਾਂਚਾ <> constexpr ਪੂਰਨ get_quants_ਮਾਪ<ਖੰਡ_q6_K>() { ਵਾਪਸ (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N; }
ਸਾਂਚਾ <> constexpr ਪੂਰਨ get_quants_ਮਾਪ<ਖੰਡ_iq4_xs>() { ਵਾਪਸ (QK_K / 2) * TILE_N; }

// used for QKK format
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ TB, ਤਾਰਕਿਕ is_acc,
          ਕਿਸਮ_ਨਾਂ std::enable_if<is_type_qkk<TB>::value, ਪੂਰਨ>::type = 0>
ਸਤਰਅੰਦਰ ਖਾਲੀ scale_C(ਅਚਲ int32_t * RESTRICT tile, int32_t * RESTRICT sumi, ਅਚਲ ਖਾਲੀ * packed_B, ਪੂਰਨ k, ਪੂਰਨ nr) {
    ਅਚਲ uint8_t * scales = reinterpret_cast<ਅਚਲ uint8_t *>((ਅਚਲ ਅੱਖਰ *)packed_B + get_quants_ਮਾਪ<TB>());
    ਅਚਲ __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((ਅਚਲ __m128i *)(scales + k * TILE_N)));

    ਲਈ (ਪੂਰਨ m = 0; m < nr; ++m) {
        __m512i vsumi;
        ਜੇ (is_acc) {
            vsumi = _mm512_loadu_si512(sumi + m * TILE_N);
        } ਨਹੀਂਤਾਂ {
            vsumi = _mm512_setzero_si512();
        }
        __m512i vtile = _mm512_loadu_si512(tile + m * TILE_N);
        vsumi = mm512_ਜੋੜੋ_epi32(vsumi, _mm512_mullo_epi32(vtile, vscale));
        _mm512_storeu_si512((__m512i *)(sumi + m * TILE_N), vsumi);
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ TA, ਕਿਸਮ_ਨਾਂ TB, ਕਿਸਮ_ਨਾਂ TC, ਪੂਰਨ ਖੰਡ_M, ਪੂਰਨ ਖੰਡ_N, ਪੂਰਨ ਖੰਡ_K>
ਬਣਤਰ tinygemm_kernel_avx {
    ਸਥਿਰ ਖਾਲੀ apply(ਪੂਰਨ K, ਅਚਲ TA * RESTRICT A, ਅਚਲ TB * RESTRICT B, TC * RESTRICT C, ਪੂਰਨ ldc) {
        GGML_ਅਣਵਰਤਿਆ(K);
        GGML_ਅਣਵਰਤਿਆ(A);
        GGML_ਅਣਵਰਤਿਆ(B);
        GGML_ਅਣਵਰਤਿਆ(C);
        GGML_ਅਣਵਰਤਿਆ(ldc);
    }
};

ਸਾਂਚਾ <ਪੂਰਨ ਖੰਡ_M, ਪੂਰਨ ਖੰਡ_N, ਪੂਰਨ ਖੰਡ_K>
ਬਣਤਰ tinygemm_kernel_avx<ਦਸ਼ਮਲਵ, ggml_fp16_t, ਦਸ਼ਮਲਵ, ਖੰਡ_M, ਖੰਡ_N, ਖੰਡ_K> {
    ਸਥਿਰ ਖਾਲੀ apply(ਪੂਰਨ K, ਅਚਲ ਦਸ਼ਮਲਵ * RESTRICT A, ਅਚਲ ggml_fp16_t * RESTRICT B, ਦਸ਼ਮਲਵ * RESTRICT C, ਪੂਰਨ ldc) {
        constexpr ਪੂਰਨ ਪੰਕਤੀਆਂ_2 = ਖੰਡ_M;
        constexpr ਪੂਰਨ ਖਾਨੇ_ਗਿਣਤੀ = ਖੰਡ_N;
        ਪ੍ਰਤਿਜ੍ਞਾ(ਖੰਡ_K == 16);

        __m512 va;
        __m512 vb[ਖਾਨੇ_ਗਿਣਤੀ];
        __m512 vc[ਪੰਕਤੀਆਂ_2 * ਖਾਨੇ_ਗਿਣਤੀ];

        ਸਵੈਚਾਲਿਤ loadc = [&](ਸਵੈਚਾਲਿਤ idx) {
            vc[idx] = _mm512_setzero_ps();
        };
        Unroll<ਪੰਕਤੀਆਂ_2 * ਖਾਨੇ_ਗਿਣਤੀ>{}(loadc);

        ਸਵੈਚਾਲਿਤ compute = [&](ਸਵੈਚਾਲਿਤ idx, ਸਵੈਚਾਲਿਤ k) {
            constexpr ਪੂਰਨ ਪੰਕਤੀ = idx / ਖਾਨੇ_ਗਿਣਤੀ;
            constexpr ਪੂਰਨ ਖਾਨਾ = idx % ਖਾਨੇ_ਗਿਣਤੀ;

            ਜੇ constexpr (ਖਾਨਾ == 0) {
                va = _mm512_loadu_ps(A + ਪੰਕਤੀ * K + k);
            }
            ਜੇ constexpr (ਪੰਕਤੀ == 0) {
                vb[ਖਾਨਾ] =  _mm512_cvtph_ps(_mm256_loadu_si256((ਅਚਲ __m256i *)(B + ਖਾਨਾ * K + k)));
            }
            vc[idx] = _mm512_fmadd_ps(va, vb[ਖਾਨਾ], vc[idx]);
        };

        ਲਈ (ਪੂਰਨ k = 0; k < K; k += 16) {
            Unroll<ਪੰਕਤੀਆਂ_2 * ਖਾਨੇ_ਗਿਣਤੀ>{}(compute, k);
        }

        ਸਵੈਚਾਲਿਤ storec = [&](ਸਵੈਚਾਲਿਤ idx) {
            constexpr ਪੂਰਨ ਪੰਕਤੀ = idx / ਖਾਨੇ_ਗਿਣਤੀ;
            constexpr ਪੂਰਨ ਖਾਨਾ = idx % ਖਾਨੇ_ਗਿਣਤੀ;
            C[ਪੰਕਤੀ * ldc + ਖਾਨਾ] = mm512_reduce_ਜੋੜੋ_ps(vc[idx]);
        };
        Unroll<ਪੰਕਤੀਆਂ_2 * ਖਾਨੇ_ਗਿਣਤੀ>{}(storec);
    }
};

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ LAUNCH_TINYGEMM_KERNEL_AVX(MB_ਮਾਪ, NB_ਮਾਪ)                                \
    tinygemm_kernel_avx<ਦਸ਼ਮਲਵ, type, ਦਸ਼ਮਲਵ, MB_ਮਾਪ, NB_ਮਾਪ, blck_ਮਾਪ>::apply(    \
        K, (ਅਚਲ ਦਸ਼ਮਲਵ *)src1->ਅੰਕੜੇ + src1_ਵਿੱਥ + mb_ਸ਼ੁਰੂ * K,                  \
        (ਅਚਲ type *)src0->ਅੰਕੜੇ + src0_ਵਿੱਥ + nb_ਸ਼ੁਰੂ * K,                      \
        (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ + dst_ਵਿੱਥ + mb_ਸ਼ੁਰੂ * ldc + nb_ਸ਼ੁਰੂ, ldc)


// re-organize in the format {NB, KB, TILE_SIZE}:
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ PACKED_ਸੂਚਕਾਂਕ(n, k, KB, tile_ਮਾਪ) (n * KB + k) * tile_ਮਾਪ

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ TB, ਪੂਰਨ ਖੰਡ_K>
ਖਾਲੀ convert_B_packed_format(ਖਾਲੀ * RESTRICT packed_B, ਅਚਲ TB * RESTRICT B, ਪੂਰਨ N, ਪੂਰਨ K) {
    ਅਚਲ ਪੂਰਨ NB = N / TILE_N;
    ਅਚਲ ਪੂਰਨ KB = K / ਖੰਡ_K;
    ਅਚਲ ਪੂਰਨ TILE_ਮਾਪ = get_tile_ਮਾਪ<TB>();

    // parallel on NB should be enough
    parallel_for(NB, [&](ਪੂਰਨ begin, ਪੂਰਨ end) {
        ਲਈ (ਪੂਰਨ n = begin; n < end; ++n) {
            ਲਈ (ਪੂਰਨ k = 0; k < KB; ++k) {
                ਪੂਰਨ n0 = n * TILE_N;
                pack_B((ਅੱਖਰ *)packed_B + PACKED_ਸੂਚਕਾਂਕ(n, k, KB, TILE_ਮਾਪ), &B[n0 * KB + k], KB);
            }
        }
    });
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ TA, ਕਿਸਮ_ਨਾਂ TB, ਕਿਸਮ_ਨਾਂ TC, ਪੂਰਨ ਖੰਡ_M, ਪੂਰਨ ਖੰਡ_N, ਪੂਰਨ ਖੰਡ_K>
ਬਣਤਰ tinygemm_kernel_vnni {};

ਸਾਂਚਾ <ਪੂਰਨ ਖੰਡ_M, ਪੂਰਨ ਖੰਡ_N, ਪੂਰਨ ਖੰਡ_K>
ਬਣਤਰ tinygemm_kernel_vnni<ਖੰਡ_q8_0, ਖੰਡ_q4_0, ਦਸ਼ਮਲਵ, ਖੰਡ_M, ਖੰਡ_N, ਖੰਡ_K> {
    ਸਥਿਰ ਖਾਲੀ apply(ਪੂਰਨ KB, ਅਚਲ ਖਾਲੀ * RESTRICT _A, ਅਚਲ ਖਾਲੀ * RESTRICT _B, ਦਸ਼ਮਲਵ * RESTRICT C, ਪੂਰਨ ldc) {

        constexpr ਪੂਰਨ ਖਾਨੇ_ਗਿਣਤੀ = ਖੰਡ_N / 16;
        ਅਚਲ ਪੂਰਨ TILE_ਮਾਪ = TILE_N * ਆਕਾਰ(ਖੰਡ_q4_0);

        ਅਚਲ ਖੰਡ_q8_0 * RESTRICT A = static_cast<ਅਚਲ ਖੰਡ_q8_0 *>(_A);
        ਅਚਲ ਅੱਖਰ * RESTRICT B = static_cast<ਅਚਲ ਅੱਖਰ *>(_B);

        __m512i va[8];
        __m512 vc[ਖਾਨੇ_ਗਿਣਤੀ];
        __m512 vd1;

        // sum of offsets, shared across COLS
        //
        // avx512-vnni does not have `_mm512_dpbssd_epi32`,
        // need to transform ss to us:
        //   a * (b - 8) is equivalent to b * a - 8 * a
        //   s    u   u                   u   s   u   s
        //
        __m512i vcomp;

        ਅਚਲ __m512i off = _mm512_set1_epi8(8);
        ਅਚਲ __m512i ਨੀਵਾਂ_ਨਕਾਬ = _mm512_set1_epi8(0xF);

        ਸਵੈਚਾਲਿਤ loadc = [&](ਸਵੈਚਾਲਿਤ ਖਾਨਾ) {
            vc[ਖਾਨਾ] = _mm512_setzero_ps();
        };
        Unroll<ਖਾਨੇ_ਗਿਣਤੀ>{}(loadc);

        ਸਵੈਚਾਲਿਤ compute = [&](ਸਵੈਚਾਲਿਤ ਖਾਨਾ, ਸਵੈਚਾਲਿਤ i) {
            // load a and compute compensation
            ਜੇ constexpr (ਖਾਨਾ == 0) {
                ਅਚਲ int32_t * a_ptr = reinterpret_cast<ਅਚਲ int32_t *>(A[0 * KB + i].qs);
                vcomp = _mm512_setzero_si512();
                ਲਈ (ਪੂਰਨ k = 0; k < 8; ++k) {
                    va[k] = _mm512_set1_epi32(a_ptr[k]);
                    vcomp = _mm512_dpbusd_epi32(vcomp, off, va[k]);
                }
                vd1 = _mm512_set1_ps(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(A[0 * KB + i].d));
            }

            // load b
            __m512i vsum = _mm512_setzero_si512();
            ਅਚਲ ਅੱਖਰ * b_ptr = B + PACKED_ਸੂਚਕਾਂਕ(ਖਾਨਾ, i, KB, TILE_ਮਾਪ);
            ਲਈ (ਪੂਰਨ k = 0; k < 8; k += 2) {
                __m512i ਬਾਈਟਾਂ = _mm512_loadu_si512((ਅਚਲ __m512i *)(b_ptr + k * 32));
                __m512i vb0 = _mm512_and_si512(ਬਾਈਟਾਂ, ਨੀਵਾਂ_ਨਕਾਬ);
                vsum = _mm512_dpbusd_epi32(vsum, vb0, va[k + 0]);
                __m512i vb1 = _mm512_and_si512(_mm512_srli_epi16(ਬਾਈਟਾਂ, 4), ਨੀਵਾਂ_ਨਕਾਬ);
                vsum = _mm512_dpbusd_epi32(vsum, vb1, va[k + 1]);
            }
            ਅਚਲ ਪੂਰਨ ਵਿੱਥ = TILE_N * TILE_K / 2;
            ਅਚਲ __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ਅਚਲ __m256i *)(b_ptr + ਵਿੱਥ)));
            vsum = _mm512_sub_epi32(vsum, vcomp);

            vc[ਖਾਨਾ] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vsum), _mm512_mul_ps(vd0, vd1), vc[ਖਾਨਾ]);
        };

        ਲਈ (ਪੂਰਨ i = 0; i < KB; ++i) {
            Unroll<ਖਾਨੇ_ਗਿਣਤੀ>{}(compute, i);
        }

        //store to C
        ਸਵੈਚਾਲਿਤ storec = [&](ਸਵੈਚਾਲਿਤ ਖਾਨਾ) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + ਖਾਨਾ * 16), vc[ਖਾਨਾ]);
        };
        Unroll<ਖਾਨੇ_ਗਿਣਤੀ>{}(storec);
    }
};

ਸਾਂਚਾ <ਪੂਰਨ ਖੰਡ_N, ਪੂਰਨ ਖੰਡ_K>
ਬਣਤਰ tinygemm_kernel_vnni<ਖੰਡ_q8_1, ਖੰਡ_q4_1, ਦਸ਼ਮਲਵ, 1, ਖੰਡ_N, ਖੰਡ_K> {
    ਸਥਿਰ ਖਾਲੀ apply(ਪੂਰਨ KB, ਅਚਲ ਖਾਲੀ * RESTRICT _A, ਅਚਲ ਖਾਲੀ * RESTRICT _B, ਦਸ਼ਮਲਵ * RESTRICT C, ਪੂਰਨ ldc) {

        constexpr ਪੂਰਨ ਖਾਨੇ_ਗਿਣਤੀ = ਖੰਡ_N / 16;
        ਅਚਲ ਪੂਰਨ TILE_ਮਾਪ = TILE_N * ਆਕਾਰ(ਖੰਡ_q4_1);

        ਅਚਲ ਖੰਡ_q8_1 * RESTRICT A = static_cast<ਅਚਲ ਖੰਡ_q8_1 *>(_A);
        ਅਚਲ ਅੱਖਰ * RESTRICT B = static_cast<ਅਚਲ ਅੱਖਰ *>(_B);

        __m512i va[8];
        __m512i vb[8];
        __m512 vc[ਖਾਨੇ_ਗਿਣਤੀ];
        __m512 vd1, vs1;

        ਅਚਲ __m512i ਨੀਵਾਂ_ਨਕਾਬ = _mm512_set1_epi8(0xF);

        ਸਵੈਚਾਲਿਤ loadc = [&](ਸਵੈਚਾਲਿਤ ਖਾਨਾ) {
            vc[ਖਾਨਾ] = _mm512_setzero_ps();
        };
        Unroll<ਖਾਨੇ_ਗਿਣਤੀ>{}(loadc);

        ਸਵੈਚਾਲਿਤ compute = [&](ਸਵੈਚਾਲਿਤ ਖਾਨਾ, ਸਵੈਚਾਲਿਤ i) {
            // load a
            ਜੇ constexpr (ਖਾਨਾ == 0) {
                ਅਚਲ int32_t * a_ptr = reinterpret_cast<ਅਚਲ int32_t *>(A[0 * KB + i].qs);
                ਲਈ (ਪੂਰਨ k = 0; k < 8; ++k) {
                    va[k] = _mm512_set1_epi32(a_ptr[k]);
                }
                vd1 = _mm512_set1_ps(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(A[0 * KB + i].d));
                vs1 = _mm512_set1_ps(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(A[0 * KB + i].s));
            }

            // load b
            ਅਚਲ ਅੱਖਰ * b_ptr = B + PACKED_ਸੂਚਕਾਂਕ(ਖਾਨਾ, i, KB, TILE_ਮਾਪ);
            ਲਈ (ਪੂਰਨ k = 0; k < 8; k += 2) {
                __m512i ਬਾਈਟਾਂ = _mm512_loadu_si512((ਅਚਲ __m512i *)(b_ptr + k * 32));
                vb[k + 0] = _mm512_and_si512(ਬਾਈਟਾਂ, ਨੀਵਾਂ_ਨਕਾਬ);
                vb[k + 1] = _mm512_and_si512(_mm512_srli_epi16(ਬਾਈਟਾਂ, 4), ਨੀਵਾਂ_ਨਕਾਬ);
            }
            ਅਚਲ ਪੂਰਨ ਵਿੱਥ = TILE_N * TILE_K / 2;
            ਅਚਲ __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ਅਚਲ __m256i *)(b_ptr + ਵਿੱਥ)));
            ਅਚਲ __m512 vm0 = _mm512_cvtph_ps(_mm256_loadu_si256((ਅਚਲ __m256i *)(b_ptr + ਵਿੱਥ + TILE_N * ਆਕਾਰ(ggml_half))));

            __m512i vsum = _mm512_setzero_si512();
            ਲਈ (ਪੂਰਨ k = 0; k < 8; ++k) {
                vsum = _mm512_dpbusd_epi32(vsum, vb[k], va[k]);
            }

            vc[ਖਾਨਾ] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vsum), _mm512_mul_ps(vd0, vd1), vc[ਖਾਨਾ]);
            vc[ਖਾਨਾ] = _mm512_fmadd_ps(vm0, vs1, vc[ਖਾਨਾ]);
        };

        ਲਈ (ਪੂਰਨ i = 0; i < KB; ++i) {
            Unroll<ਖਾਨੇ_ਗਿਣਤੀ>{}(compute, i);
        }

        //store to C
        ਸਵੈਚਾਲਿਤ storec = [&](ਸਵੈਚਾਲਿਤ ਖਾਨਾ) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + ਖਾਨਾ * 16), vc[ਖਾਨਾ]);
        };
        Unroll<ਖਾਨੇ_ਗਿਣਤੀ>{}(storec);
    }
};

ਸਾਂਚਾ <ਪੂਰਨ ਖੰਡ_M, ਪੂਰਨ ਖੰਡ_N, ਪੂਰਨ ਖੰਡ_K>
ਬਣਤਰ tinygemm_kernel_vnni<ਖੰਡ_q8_0, ਖੰਡ_q8_0, ਦਸ਼ਮਲਵ, ਖੰਡ_M, ਖੰਡ_N, ਖੰਡ_K> {
    ਸਥਿਰ ਖਾਲੀ apply(ਪੂਰਨ KB, ਅਚਲ ਖਾਲੀ * RESTRICT _A, ਅਚਲ ਖਾਲੀ * RESTRICT _B, ਦਸ਼ਮਲਵ * RESTRICT C, ਪੂਰਨ ldc) {

        constexpr ਪੂਰਨ ਖਾਨੇ_ਗਿਣਤੀ = ਖੰਡ_N / 16;
        ਅਚਲ ਪੂਰਨ TILE_ਮਾਪ = TILE_N * ਆਕਾਰ(ਖੰਡ_q8_0) + TILE_N * ਆਕਾਰ(int32_t);

        ਅਚਲ ਖੰਡ_q8_0 * RESTRICT A = static_cast<ਅਚਲ ਖੰਡ_q8_0 *>(_A);
        ਅਚਲ ਅੱਖਰ * RESTRICT B = static_cast<ਅਚਲ ਅੱਖਰ *>(_B);

        __m512i va[8];
        __m512i vb[8];
        __m512 vc[ਖਾਨੇ_ਗਿਣਤੀ];
        __m512 vd1;

        // Notes: s8s8 igemm compensation in avx512-vnni
        // change s8s8 to u8s8 with compensate
        //   a * b = (a + 128) * b - 128 * b
        //   s   s       u       s    u    s
        //
        // (128 * b is pre-computed when packing B to vnni formats)
        //
        ਅਚਲ __m512i off = _mm512_set1_epi8(static_cast<ਅੱਖਰ>(0x80));

        ਸਵੈਚਾਲਿਤ loadc = [&](ਸਵੈਚਾਲਿਤ ਖਾਨਾ) {
            vc[ਖਾਨਾ] = _mm512_setzero_ps();
        };
        Unroll<ਖਾਨੇ_ਗਿਣਤੀ>{}(loadc);

        ਸਵੈਚਾਲਿਤ compute = [&](ਸਵੈਚਾਲਿਤ ਖਾਨਾ, ਸਵੈਚਾਲਿਤ i) {
            // load a and add offset 128
            ਜੇ constexpr (ਖਾਨਾ == 0) {
                ਅਚਲ int32_t * a_ptr = reinterpret_cast<ਅਚਲ int32_t *>(A[0 * KB + i].qs);
                ਲਈ (ਪੂਰਨ k = 0; k < 8; ++k) {
                    va[k] = _mm512_set1_epi32(a_ptr[k]);
                    va[k] = mm512_ਜੋੜੋ_epi8(va[k], off);
                }
                vd1 = _mm512_set1_ps(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(A[0 * KB + i].d));
            }

            // load b
            ਅਚਲ ਅੱਖਰ * b_ptr = B + PACKED_ਸੂਚਕਾਂਕ(ਖਾਨਾ, i, KB, TILE_ਮਾਪ);
            ਲਈ (ਪੂਰਨ k = 0; k < 8; ++k) {
                vb[k] = _mm512_loadu_si512((ਅਚਲ __m512i *)(b_ptr + k * 64));
            }
            ਅਚਲ ਪੂਰਨ ਵਿੱਥ = TILE_N * TILE_K;
            ਅਚਲ __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ਅਚਲ __m256i *)(b_ptr + ਵਿੱਥ)));
            ਅਚਲ ਪੂਰਨ offset2 = TILE_N * TILE_K + TILE_N * ਆਕਾਰ(ggml_half);
            ਅਚਲ __m512i vcomp = _mm512_loadu_si512((ਅਚਲ __m512i *)(b_ptr + offset2));

            __m512i vsum = _mm512_setzero_si512();
            ਲਈ (ਪੂਰਨ k = 0; k < 8; ++k) {
                vsum = _mm512_dpbusd_epi32(vsum, va[k], vb[k]);
            }
            vsum = _mm512_sub_epi32(vsum, vcomp);

            vc[ਖਾਨਾ] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vsum), _mm512_mul_ps(vd0, vd1), vc[ਖਾਨਾ]);
        };

        ਲਈ (ਪੂਰਨ i = 0; i < KB; ++i) {
            Unroll<ਖਾਨੇ_ਗਿਣਤੀ>{}(compute, i);
        }

        //store to C
        ਸਵੈਚਾਲਿਤ storec = [&](ਸਵੈਚਾਲਿਤ ਖਾਨਾ) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + ਖਾਨਾ * 16), vc[ਖਾਨਾ]);
        };
        Unroll<ਖਾਨੇ_ਗਿਣਤੀ>{}(storec);
    }
};

ਸਾਂਚਾ <ਪੂਰਨ ਖੰਡ_M, ਪੂਰਨ ਖੰਡ_N, ਪੂਰਨ ਖੰਡ_K>
ਬਣਤਰ tinygemm_kernel_vnni<ਖੰਡ_q8_K, ਖੰਡ_q4_K, ਦਸ਼ਮਲਵ, ਖੰਡ_M, ਖੰਡ_N, ਖੰਡ_K> {
    ਸਥਿਰ ਖਾਲੀ apply(ਪੂਰਨ KB, ਅਚਲ ਖਾਲੀ * RESTRICT _A, ਅਚਲ ਖਾਲੀ * RESTRICT _B, ਦਸ਼ਮਲਵ * RESTRICT C, ਪੂਰਨ ldc) {

        constexpr ਪੂਰਨ ਖਾਨੇ_ਗਿਣਤੀ = ਖੰਡ_N / 16;
        ਅਚਲ ਪੂਰਨ TILE_ਮਾਪ = TILE_N * ਆਕਾਰ(ਖੰਡ_q4_K) + TILE_N * 4;

        ਅਚਲ ਖੰਡ_q8_K * RESTRICT A = static_cast<ਅਚਲ ਖੰਡ_q8_K *>(_A);
        ਅਚਲ ਅੱਖਰ * RESTRICT B = static_cast<ਅਚਲ ਅੱਖਰ *>(_B);

        // a.qs:   8 groups, 32 bytes each group (m256i)
        __m512i va[8];
        // a.bsum: 8 groups,  2 bytes each group (m128i)
        __m512i va_bsum;
        __m512 vc[ਖਾਨੇ_ਗਿਣਤੀ];
        __m512 vd1;

        // packed_B:
        ਅਚਲ ਪੂਰਨ ਵਿੱਥ_scales = (QK_K / 2) * TILE_N;
        ਅਚਲ ਪੂਰਨ ਵਿੱਥ_mins   = (QK_K / 2) * TILE_N +  8 * TILE_N;
        ਅਚਲ ਪੂਰਨ ਵਿੱਥ_d0     = (QK_K / 2) * TILE_N + 16 * TILE_N;
        ਅਚਲ ਪੂਰਨ ਵਿੱਥ_dmin   = (QK_K / 2) * TILE_N + 16 * TILE_N + TILE_N * ਆਕਾਰ(ggml_half);

        ਅਚਲ __m512i ਨੀਵਾਂ_ਨਕਾਬ = _mm512_set1_epi8(0xF);

        ਸਵੈਚਾਲਿਤ loadc = [&](ਸਵੈਚਾਲਿਤ ਖਾਨਾ) {
            vc[ਖਾਨਾ] = _mm512_setzero_ps();
        };
        Unroll<ਖਾਨੇ_ਗਿਣਤੀ>{}(loadc);

        // Notes: vnni formats in QK_K
        //   a) quants vnni format
        //     int8  {k/4, n, 4}, viewed as 2d {k/4, 4n}, k = 32
        //     from {16, 32} to {8, 64}
        //
        //   b) min vnni format
        //     int16 {k/2, n, 2}, viewed as 2d {k/2, 2n}, k = 8
        //     from {16,  8} to {4, 32}
        //
        ਸਵੈਚਾਲਿਤ compute = [&](ਸਵੈਚਾਲਿਤ ਖਾਨਾ, ਸਵੈਚਾਲਿਤ i) {
            // load a
            ਜੇ constexpr (ਖਾਨਾ == 0) {
                ਲਈ (ਪੂਰਨ k_ਸਮੂਹ = 0; k_ਸਮੂਹ < QK_K / 32; ++k_ਸਮੂਹ) {
                    va[k_ਸਮੂਹ] = _mm512_castsi256_si512(_mm256_loadu_si256((ਅਚਲ __m256i *)(A[0 * KB + i].qs + k_ਸਮੂਹ * 32)));
                }
                ਅਚਲ __m256i q8sums = _mm256_loadu_si256((ਅਚਲ __m256i *)A[0 * KB + i].bsums);
                ਅਚਲ __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
                va_bsum = _mm512_castsi128_si512(q8s);
                vd1 = _mm512_set1_ps(A[0 * KB + i].d);
            }

            // step 1: accumultate the quants
            __m512i acc = _mm512_setzero_si512();
            ਅਚਲ ਅੱਖਰ * b_ptr = B + PACKED_ਸੂਚਕਾਂਕ(ਖਾਨਾ, i, KB, TILE_ਮਾਪ);
            ਅਚਲ ਅੱਖਰ * b_qs  = b_ptr;
            ਲਈ (ਪੂਰਨ k_ਸਮੂਹ = 0; k_ਸਮੂਹ < QK_K / 32; ++k_ਸਮੂਹ) {
                __m512i vsum = _mm512_setzero_si512();
                ਲਈ (ਪੂਰਨ k = 0; k < 8; k += 2) {
                    __m512i va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(k + 0), va[k_ਸਮੂਹ]);
                    __m512i va1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(k + 1), va[k_ਸਮੂਹ]);

                    __m512i ਬਾਈਟਾਂ = _mm512_loadu_si512((ਅਚਲ __m512i *)b_qs);
                    __m512i vb0 = _mm512_and_si512(ਬਾਈਟਾਂ, ਨੀਵਾਂ_ਨਕਾਬ);
                    vsum = _mm512_dpbusd_epi32(vsum, vb0, va0);
                    __m512i vb1 = _mm512_and_si512(_mm512_srli_epi16(ਬਾਈਟਾਂ, 4), ਨੀਵਾਂ_ਨਕਾਬ);
                    vsum = _mm512_dpbusd_epi32(vsum, vb1, va1);

                    b_qs += 64;
                }
                // vacc += scale * (q8 @ q4)
                ਅਚਲ __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((ਅਚਲ __m128i *)(b_ptr + ਵਿੱਥ_scales + k_ਸਮੂਹ * TILE_N)));
                acc = mm512_ਜੋੜੋ_epi32(acc, _mm512_mullo_epi32(vsum, vscale));
            }
            ਅਚਲ __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ਅਚਲ __m256i *)(b_ptr + ਵਿੱਥ_d0)));
            vc[ਖਾਨਾ] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(vd0, vd1), vc[ਖਾਨਾ]);

            // step 2: accumulate the mins
            __m512i acc_m = _mm512_setzero_si512();
            ਲਈ (ਪੂਰਨ k = 0; k < 4; ++k) {
                __m512i vmask = _mm512_set1_epi32(k);
                __m512i va = _mm512_permutexvar_epi32(vmask, va_bsum);
                __m512i vb = _mm512_cvtepi8_epi16(_mm256_loadu_si256((ਅਚਲ __m256i *)(b_ptr + ਵਿੱਥ_mins + k * 32)));
                acc_m = _mm512_dpwssds_epi32(acc_m, va, vb);
            }
            ਅਚਲ __m512 vdmin = _mm512_cvtph_ps(_mm256_loadu_si256((ਅਚਲ __m256i *)(b_ptr + ਵਿੱਥ_dmin)));
            vc[ਖਾਨਾ] = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(acc_m), _mm512_mul_ps(vdmin, vd1), vc[ਖਾਨਾ]);
        };

        ਲਈ (ਪੂਰਨ i = 0; i < KB; ++i) {
            Unroll<ਖਾਨੇ_ਗਿਣਤੀ>{}(compute, i);
        }

        //store to C
        ਸਵੈਚਾਲਿਤ storec = [&](ਸਵੈਚਾਲਿਤ ਖਾਨਾ) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + ਖਾਨਾ * 16), vc[ਖਾਨਾ]);
        };
        Unroll<ਖਾਨੇ_ਗਿਣਤੀ>{}(storec);
    }
};

ਸਾਂਚਾ <ਪੂਰਨ ਖੰਡ_M, ਪੂਰਨ ਖੰਡ_N, ਪੂਰਨ ਖੰਡ_K>
ਬਣਤਰ tinygemm_kernel_vnni<ਖੰਡ_q8_K, ਖੰਡ_q5_K, ਦਸ਼ਮਲਵ, ਖੰਡ_M, ਖੰਡ_N, ਖੰਡ_K> {
    ਸਥਿਰ ਖਾਲੀ apply(ਪੂਰਨ KB, ਅਚਲ ਖਾਲੀ * RESTRICT _A, ਅਚਲ ਖਾਲੀ * RESTRICT _B, ਦਸ਼ਮਲਵ * RESTRICT C, ਪੂਰਨ ldc) {

        constexpr ਪੂਰਨ ਖਾਨੇ_ਗਿਣਤੀ = ਖੰਡ_N / 16;
        ਅਚਲ ਪੂਰਨ TILE_ਮਾਪ = TILE_N * ਆਕਾਰ(ਖੰਡ_q5_K) + TILE_N * 4;

        ਅਚਲ ਖੰਡ_q8_K * RESTRICT A = static_cast<ਅਚਲ ਖੰਡ_q8_K *>(_A);
        ਅਚਲ ਅੱਖਰ * RESTRICT B = static_cast<ਅਚਲ ਅੱਖਰ *>(_B);

        // a.qs:   8 groups, 32 bytes each group (m256i)
        __m512i va[8];
        // a.bsum: 8 groups,  2 bytes each group (m128i)
        __m512i va_bsum;
        __m512 vc[ਖਾਨੇ_ਗਿਣਤੀ];
        __m512 vd1;

        // packed_B:
        ਅਚਲ ਪੂਰਨ ਵਿੱਥ_qh     = (QK_K / 2) * TILE_N;
        ਅਚਲ ਪੂਰਨ ਵਿੱਥ_scales = (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N;
        ਅਚਲ ਪੂਰਨ ਵਿੱਥ_mins   = (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N +  8 * TILE_N;
        ਅਚਲ ਪੂਰਨ ਵਿੱਥ_d0     = (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N + 16 * TILE_N;
        ਅਚਲ ਪੂਰਨ ਵਿੱਥ_dmin   = (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N + 16 * TILE_N + TILE_N * ਆਕਾਰ(ggml_half);

        ਅਚਲ __m512i ਨੀਵਾਂ_ਨਕਾਬ = _mm512_set1_epi8(0xF);

        ਸਵੈਚਾਲਿਤ loadc = [&](ਸਵੈਚਾਲਿਤ ਖਾਨਾ) {
            vc[ਖਾਨਾ] = _mm512_setzero_ps();
        };
        Unroll<ਖਾਨੇ_ਗਿਣਤੀ>{}(loadc);

        // Q5_K and Q4_K shares the same vnni formats, refer to notes above.
        ਸਵੈਚਾਲਿਤ compute = [&](ਸਵੈਚਾਲਿਤ ਖਾਨਾ, ਸਵੈਚਾਲਿਤ i) {
            // load a
            ਜੇ constexpr (ਖਾਨਾ == 0) {
                ਲਈ (ਪੂਰਨ k_ਸਮੂਹ = 0; k_ਸਮੂਹ < QK_K / 32; ++k_ਸਮੂਹ) {
                    va[k_ਸਮੂਹ] = _mm512_castsi256_si512(_mm256_loadu_si256((ਅਚਲ __m256i *)(A[0 * KB + i].qs + k_ਸਮੂਹ * 32)));
                }
                ਅਚਲ __m256i q8sums = _mm256_loadu_si256((ਅਚਲ __m256i *)A[0 * KB + i].bsums);
                ਅਚਲ __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
                va_bsum = _mm512_castsi128_si512(q8s);
                vd1 = _mm512_set1_ps(A[0 * KB + i].d);
            }

            // step 1: accumultate the quants
            __m512i acc = _mm512_setzero_si512();
            ਅਚਲ ਅੱਖਰ * b_ptr = B + PACKED_ਸੂਚਕਾਂਕ(ਖਾਨਾ, i, KB, TILE_ਮਾਪ);
            ਅਚਲ ਅੱਖਰ * b_qs  = b_ptr;
            ਅਚਲ ਅੱਖਰ * b_qh  = b_ptr + ਵਿੱਥ_qh;
            ਲਈ (ਪੂਰਨ k_ਸਮੂਹ = 0; k_ਸਮੂਹ < QK_K / 32; ++k_ਸਮੂਹ) {
                __m512i vsum = _mm512_setzero_si512();
                __m512i hmask0 = _mm512_set1_epi8(0x1);
                __m512i hmask1 = _mm512_set1_epi8(0x2);
                __m512i hbits = _mm512_loadu_si512((ਅਚਲ __m512i *)(b_qh + k_ਸਮੂਹ * 64));
                ਲਈ (ਪੂਰਨ k = 0; k < 8; k += 2) {
                    __m512i va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(k + 0), va[k_ਸਮੂਹ]);
                    __m512i va1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(k + 1), va[k_ਸਮੂਹ]);

                    __m512i ਬਾਈਟਾਂ = _mm512_loadu_si512((ਅਚਲ __m512i *)b_qs);
                    __m512i vb0 = _mm512_and_si512(ਬਾਈਟਾਂ, ਨੀਵਾਂ_ਨਕਾਬ);
                    __m512i vb1 = _mm512_and_si512(_mm512_srli_epi16(ਬਾਈਟਾਂ, 4), ਨੀਵਾਂ_ਨਕਾਬ);

                    __m512i vh0 = _mm512_slli_epi16(_mm512_srli_epi16(_mm512_and_si512(hbits, hmask0), k), 4);
                    __m512i vh1 = _mm512_slli_epi16(_mm512_srli_epi16(_mm512_and_si512(hbits, hmask1), k + 1), 4);

                    hmask0 = _mm512_slli_epi16(hmask0, 2);
                    hmask1 = _mm512_slli_epi16(hmask1, 2);
                    vb0 = mm512_ਜੋੜੋ_epi8(vb0, vh0);
                    vb1 = mm512_ਜੋੜੋ_epi8(vb1, vh1);

                    vsum = _mm512_dpbusd_epi32(vsum, vb0, va0);
                    vsum = _mm512_dpbusd_epi32(vsum, vb1, va1);

                    b_qs += 64;
                }
                // vacc += scale * (q8 @ q5)
                ਅਚਲ __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((ਅਚਲ __m128i *)(b_ptr + ਵਿੱਥ_scales + k_ਸਮੂਹ * TILE_N)));
                acc = mm512_ਜੋੜੋ_epi32(acc, _mm512_mullo_epi32(vsum, vscale));
            }
            ਅਚਲ __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ਅਚਲ __m256i *)(b_ptr + ਵਿੱਥ_d0)));
            vc[ਖਾਨਾ] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(vd0, vd1), vc[ਖਾਨਾ]);

            // step 2: accumulate the mins
            __m512i acc_m = _mm512_setzero_si512();
            ਲਈ (ਪੂਰਨ k = 0; k < 4; ++k) {
                __m512i vmask = _mm512_set1_epi32(k);
                __m512i va = _mm512_permutexvar_epi32(vmask, va_bsum);
                __m512i vb = _mm512_cvtepi8_epi16(_mm256_loadu_si256((ਅਚਲ __m256i *)(b_ptr + ਵਿੱਥ_mins + k * 32)));
                acc_m = _mm512_dpwssds_epi32(acc_m, va, vb);
            }
            ਅਚਲ __m512 vdmin = _mm512_cvtph_ps(_mm256_loadu_si256((ਅਚਲ __m256i *)(b_ptr + ਵਿੱਥ_dmin)));
            vc[ਖਾਨਾ] = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(acc_m), _mm512_mul_ps(vdmin, vd1), vc[ਖਾਨਾ]);
        };

        ਲਈ (ਪੂਰਨ i = 0; i < KB; ++i) {
            Unroll<ਖਾਨੇ_ਗਿਣਤੀ>{}(compute, i);
        }

        //store to C
        ਸਵੈਚਾਲਿਤ storec = [&](ਸਵੈਚਾਲਿਤ ਖਾਨਾ) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + ਖਾਨਾ * 16), vc[ਖਾਨਾ]);
        };
        Unroll<ਖਾਨੇ_ਗਿਣਤੀ>{}(storec);
    }
};

ਸਾਂਚਾ <ਪੂਰਨ ਖੰਡ_M, ਪੂਰਨ ਖੰਡ_N, ਪੂਰਨ ਖੰਡ_K>
ਬਣਤਰ tinygemm_kernel_vnni<ਖੰਡ_q8_K, ਖੰਡ_q6_K, ਦਸ਼ਮਲਵ, ਖੰਡ_M, ਖੰਡ_N, ਖੰਡ_K> {
    ਸਥਿਰ ਖਾਲੀ apply(ਪੂਰਨ KB, ਅਚਲ ਖਾਲੀ * RESTRICT _A, ਅਚਲ ਖਾਲੀ * RESTRICT _B, ਦਸ਼ਮਲਵ * RESTRICT C, ਪੂਰਨ ldc) {

        constexpr ਪੂਰਨ ਖਾਨੇ_ਗਿਣਤੀ = ਖੰਡ_N / 16;
        ਅਚਲ ਪੂਰਨ TILE_ਮਾਪ = TILE_N * ਆਕਾਰ(ਖੰਡ_q6_K);

        ਅਚਲ ਖੰਡ_q8_K * RESTRICT A = static_cast<ਅਚਲ ਖੰਡ_q8_K *>(_A);
        ਅਚਲ ਅੱਖਰ * RESTRICT B = static_cast<ਅਚਲ ਅੱਖਰ *>(_B);

        // load the 256 bytes from A to 4 avx512 vectors
        __m512i va[4];
        __m512 vc[ਖਾਨੇ_ਗਿਣਤੀ];
        __m512 vd1;

        // packed_B:
        ਅਚਲ ਪੂਰਨ ਵਿੱਥ_qh     = (QK_K / 2) * TILE_N;
        ਅਚਲ ਪੂਰਨ ਵਿੱਥ_scales = (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N;
        ਅਚਲ ਪੂਰਨ ਵਿੱਥ_d0     = (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N + 16 * TILE_N;

        // compensation
        __m512i vcomp;

        ਅਚਲ __m512i m32s = _mm512_set1_epi32(32);
        ਅਚਲ __m512i ਨੀਵਾਂ_ਨਕਾਬ = _mm512_set1_epi8(0xF);

        ਸਵੈਚਾਲਿਤ loadc = [&](ਸਵੈਚਾਲਿਤ ਖਾਨਾ) {
            vc[ਖਾਨਾ] = _mm512_setzero_ps();
        };
        Unroll<ਖਾਨੇ_ਗਿਣਤੀ>{}(loadc);

        ਸਵੈਚਾਲਿਤ compute = [&](ਸਵੈਚਾਲਿਤ ਖਾਨਾ, ਸਵੈਚਾਲਿਤ i) {
            ਜੇ constexpr (ਖਾਨਾ == 0) {
                // load a
                va[0] = _mm512_loadu_si512((ਅਚਲ __m512i *)(A[0 * KB + i].qs +   0));
                va[1] = _mm512_loadu_si512((ਅਚਲ __m512i *)(A[0 * KB + i].qs +  64));
                va[2] = _mm512_loadu_si512((ਅਚਲ __m512i *)(A[0 * KB + i].qs + 128));
                va[3] = _mm512_loadu_si512((ਅਚਲ __m512i *)(A[0 * KB + i].qs + 192));

                ਅਚਲ __m256i q8sums = _mm256_loadu_si256((ਅਚਲ __m256i *)A[0 * KB + i].bsums);
                vcomp = _mm512_mullo_epi32(_mm512_cvtepi16_epi32(q8sums), m32s);
                vd1 = _mm512_set1_ps(A[0 * KB + i].d);
            }

            // accmulate the quants
            __m512i acc = _mm512_setzero_si512();
            ਅਚਲ ਅੱਖਰ * b_ptr = B + PACKED_ਸੂਚਕਾਂਕ(ਖਾਨਾ, i, KB, TILE_ਮਾਪ);
            ਅਚਲ ਅੱਖਰ * b_qs = b_ptr;
            ਅਚਲ ਅੱਖਰ * b_qh = b_ptr + ਵਿੱਥ_qh;
            ਪੂਰਨ ਨਕਾਬ = 0;
            ਲਈ (ਪੂਰਨ k_ਸਮੂਹ = 0; k_ਸਮੂਹ < QK_K / 16; ++k_ਸਮੂਹ) {
                ਪੂਰਨ r = k_ਸਮੂਹ >> 2;
                __m512i va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(ਨਕਾਬ++), va[r]);
                __m512i va1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(ਨਕਾਬ++), va[r]);

                __m512i vsum = _mm512_setzero_si512();
                __m512i hmask = _mm512_set1_epi8(0x3);

                __m512i ਬਾਈਟਾਂ = _mm512_loadu_si512(b_qs);
                __m512i hbits = _mm512_loadu_si512(b_qh);
                __m512i vb0 = _mm512_and_si512(ਬਾਈਟਾਂ, ਨੀਵਾਂ_ਨਕਾਬ);
                __m512i vb1 = _mm512_and_si512(_mm512_srli_epi16(ਬਾਈਟਾਂ, 4), ਨੀਵਾਂ_ਨਕਾਬ);
                __m512i vh0 = _mm512_slli_epi16(_mm512_and_si512(hbits, hmask), 4);
                __m512i vh1 = _mm512_slli_epi16(_mm512_and_si512(hbits, _mm512_slli_epi16(hmask, 2)), 2);

                vb0 = mm512_ਜੋੜੋ_epi8(vb0, vh0);
                vb1 = mm512_ਜੋੜੋ_epi8(vb1, vh1);
                vsum = _mm512_dpbusd_epi32(vsum, vb0, va0);
                vsum = _mm512_dpbusd_epi32(vsum, vb1, va1);
                b_qs += 64;

                va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(ਨਕਾਬ++), va[r]);
                va1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(ਨਕਾਬ++), va[r]);

                ਬਾਈਟਾਂ = _mm512_loadu_si512(b_qs);
                vb0 = _mm512_and_si512(ਬਾਈਟਾਂ, ਨੀਵਾਂ_ਨਕਾਬ);
                vb1 = _mm512_and_si512(_mm512_srli_epi16(ਬਾਈਟਾਂ, 4), ਨੀਵਾਂ_ਨਕਾਬ);
                vh0 =                   _mm512_and_si512(hbits, _mm512_slli_epi16(hmask, 4));
                vh1 = _mm512_srli_epi16(_mm512_and_si512(hbits, _mm512_slli_epi16(hmask, 6)), 2);
                vb0 = mm512_ਜੋੜੋ_epi8(vb0, vh0);
                vb1 = mm512_ਜੋੜੋ_epi8(vb1, vh1);
                vsum = _mm512_dpbusd_epi32(vsum, vb0, va0);
                vsum = _mm512_dpbusd_epi32(vsum, vb1, va1);
                b_qs += 64;
                b_qh += 64;

                // B * A - 32 * A
                __m512i vmask = _mm512_set1_epi32(k_ਸਮੂਹ);
                vsum = _mm512_sub_epi32(vsum, _mm512_permutexvar_epi32(vmask, vcomp));

                // vacc += scale * (q8 @ q6)
                ਅਚਲ __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((ਅਚਲ __m128i *)(b_ptr + ਵਿੱਥ_scales + k_ਸਮੂਹ * TILE_N)));
                acc = mm512_ਜੋੜੋ_epi32(acc, _mm512_mullo_epi32(vsum, vscale));
            }
            ਅਚਲ __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ਅਚਲ __m256i *)(b_ptr + ਵਿੱਥ_d0)));
            vc[ਖਾਨਾ] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(vd0, vd1), vc[ਖਾਨਾ]);
        };

        ਲਈ (ਪੂਰਨ i = 0; i < KB; ++i) {
            Unroll<ਖਾਨੇ_ਗਿਣਤੀ>{}(compute, i);
        }

        //store to C
        ਸਵੈਚਾਲਿਤ storec = [&](ਪੂਰਨ ਖਾਨਾ) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + ਖਾਨਾ * 16), vc[ਖਾਨਾ]);
        };
        Unroll<ਖਾਨੇ_ਗਿਣਤੀ>{}(storec);
    }
};

ਸਾਂਚਾ <ਪੂਰਨ ਖੰਡ_M, ਪੂਰਨ ਖੰਡ_N, ਪੂਰਨ ਖੰਡ_K>
ਬਣਤਰ tinygemm_kernel_vnni<ਖੰਡ_q8_K, ਖੰਡ_iq4_xs, ਦਸ਼ਮਲਵ, ਖੰਡ_M, ਖੰਡ_N, ਖੰਡ_K> {
    ਸਥਿਰ ਖਾਲੀ apply(ਪੂਰਨ KB, ਅਚਲ ਖਾਲੀ * RESTRICT _A, ਅਚਲ ਖਾਲੀ * RESTRICT _B, ਦਸ਼ਮਲਵ * RESTRICT C, ਪੂਰਨ ldc) {

        constexpr ਪੂਰਨ ਖਾਨੇ_ਗਿਣਤੀ = ਖੰਡ_N / 16;
        ਅਚਲ ਪੂਰਨ TILE_ਮਾਪ = TILE_N * ਆਕਾਰ(ਖੰਡ_iq4_xs) + TILE_N * 2;

        ਅਚਲ ਖੰਡ_q8_K * RESTRICT A = static_cast<ਅਚਲ ਖੰਡ_q8_K *>(_A);
        ਅਚਲ ਅੱਖਰ * RESTRICT B = static_cast<ਅਚਲ ਅੱਖਰ *>(_B);

        // load the 256 bytes from A to 4 avx512 vectors
        __m512i va[4];
        __m512 vc[ਖਾਨੇ_ਗਿਣਤੀ];
        __m512 vd1;

        // packed_B:
        ਅਚਲ ਪੂਰਨ ਵਿੱਥ_scales = (QK_K / 2) * TILE_N ;
        ਅਚਲ ਪੂਰਨ ਵਿੱਥ_d0     = (QK_K / 2) * TILE_N + 8 * TILE_N;

        // compensation
        __m512i vcomp;

        ਅਚਲ __m256i m128s = _mm256_set1_epi16(128);
        ਅਚਲ __m512i ਨੀਵਾਂ_ਨਕਾਬ = _mm512_set1_epi8(0xF);

        ਅਚਲ __m512i values128 = mm512_ਸੈੱਟ_epi8(
            113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
            113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
            113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
            113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127
        );
        ਅਚਲ __m512i off = _mm512_set1_epi8(static_cast<ਅੱਖਰ>(0x80));
        ਅਚਲ __m512i values256 = mm512_ਜੋੜੋ_epi8(values128, off);

        ਸਵੈਚਾਲਿਤ loadc = [&](ਸਵੈਚਾਲਿਤ ਖਾਨਾ) {
            vc[ਖਾਨਾ] = _mm512_setzero_ps();
        };
        Unroll<ਖਾਨੇ_ਗਿਣਤੀ>{}(loadc);

        ਸਵੈਚਾਲਿਤ compute = [&](ਸਵੈਚਾਲਿਤ ਖਾਨਾ, ਸਵੈਚਾਲਿਤ i) {
            ਜੇ constexpr (ਖਾਨਾ == 0) {
                // load a
                va[0] = _mm512_loadu_si512((ਅਚਲ __m512i *)(A[0 * KB + i].qs +   0));
                va[1] = _mm512_loadu_si512((ਅਚਲ __m512i *)(A[0 * KB + i].qs +  64));
                va[2] = _mm512_loadu_si512((ਅਚਲ __m512i *)(A[0 * KB + i].qs + 128));
                va[3] = _mm512_loadu_si512((ਅਚਲ __m512i *)(A[0 * KB + i].qs + 192));

                // compensation: 128 * A
                ਅਚਲ __m256i q8sums = _mm256_loadu_si256((ਅਚਲ __m256i *)A[0 * KB + i].bsums);
                vcomp = _mm512_castsi256_si512(_mm256_madd_epi16(q8sums, m128s));
                vd1 = _mm512_set1_ps(A[0 * KB + i].d);
            }

            // accmulate the quants
            __m512i acc = _mm512_setzero_si512();
            ਅਚਲ ਅੱਖਰ * b_ptr = B + PACKED_ਸੂਚਕਾਂਕ(ਖਾਨਾ, i, KB, TILE_ਮਾਪ);
            ਅਚਲ ਅੱਖਰ * b_qs = b_ptr;
            ਪੂਰਨ ਨਕਾਬ = 0;
            ਲਈ (ਪੂਰਨ k_ਸਮੂਹ = 0; k_ਸਮੂਹ < QK_K / 32; ++k_ਸਮੂਹ) {
                ਪੂਰਨ r = k_ਸਮੂਹ >> 1;
                __m512i vmask = _mm512_set1_epi32(k_ਸਮੂਹ);
                __m512i vsum = _mm512_setzero_si512();
                ਲਈ (ਪੂਰਨ k = 0; k < 8; k += 2) {
                    __m512i va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(ਨਕਾਬ++), va[r]);
                    __m512i va1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(ਨਕਾਬ++), va[r]);

                    __m512i ਬਾਈਟਾਂ = _mm512_loadu_si512(b_qs);
                    __m512i vb0 = _mm512_shuffle_epi8(values256, _mm512_and_si512(ਬਾਈਟਾਂ, ਨੀਵਾਂ_ਨਕਾਬ));
                    __m512i vb1 = _mm512_shuffle_epi8(values256, _mm512_and_si512(_mm512_srli_epi16(ਬਾਈਟਾਂ, 4), ਨੀਵਾਂ_ਨਕਾਬ));

                    vsum = _mm512_dpbusd_epi32(vsum, vb0, va0);
                    vsum = _mm512_dpbusd_epi32(vsum, vb1, va1);
                    b_qs += 64;
                }
                // (B + 128) * A - 128 * A
                vsum = _mm512_sub_epi32(vsum, _mm512_permutexvar_epi32(vmask, vcomp));

                // vacc += scale * (q8 @ q4)
                ਅਚਲ __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((ਅਚਲ __m128i *)(b_ptr + ਵਿੱਥ_scales + k_ਸਮੂਹ * TILE_N)));
                acc = mm512_ਜੋੜੋ_epi32(acc, _mm512_mullo_epi32(vsum, vscale));
            }
            ਅਚਲ __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ਅਚਲ __m256i *)(b_ptr + ਵਿੱਥ_d0)));
            vc[ਖਾਨਾ] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(vd0, vd1), vc[ਖਾਨਾ]);
        };

        ਲਈ (ਪੂਰਨ i = 0; i < KB; ++i) {
            Unroll<ਖਾਨੇ_ਗਿਣਤੀ>{}(compute, i);
        }

        //store to C
        ਸਵੈਚਾਲਿਤ storec = [&](ਸਵੈਚਾਲਿਤ ਖਾਨਾ) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + ਖਾਨਾ * 16), vc[ਖਾਨਾ]);
        };
        Unroll<ਖਾਨੇ_ਗਿਣਤੀ>{}(storec);
    }
};

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ LAUNCH_TINYGEMM_KERNEL_VNNI(NB_ਮਾਪ)                                                   \
    tinygemm_kernel_vnni<vec_dot_type, type, ਦਸ਼ਮਲਵ, 1, NB_ਮਾਪ, blck_ਮਾਪ>::apply(             \
        KB, wdata_batch,                                                                       \
        (ਅਚਲ ਅੱਖਰ *)src0->ਅੰਕੜੇ + src0_ਵਿੱਥ + PACKED_ਸੂਚਕਾਂਕ(nb * kTilesN, 0, KB, TILE_ਮਾਪ), \
        (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ + dst_ਵਿੱਥ + nb_ਸ਼ੁਰੂ, ldc)

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ TA, ਕਿਸਮ_ਨਾਂ TB, ਕਿਸਮ_ਨਾਂ TC, ਪੂਰਨ ਖੰਡ_K,
          ਕਿਸਮ_ਨਾਂ std::enable_if<!is_type_qkk<TB>::value, ਪੂਰਨ>::type = 0>
ਖਾਲੀ tinygemm_kernel_amx(ਪੂਰਨ M, ਪੂਰਨ N, ਪੂਰਨ KB, ਅਚਲ ਖਾਲੀ * RESTRICT _A, ਅਚਲ ਖਾਲੀ * RESTRICT _B, TC * RESTRICT C, ਪੂਰਨ ldc) {
    ਵਰਤੋਂ packed_B_t = packed_B_type<TB>;
    ਅਚਲ ਪੂਰਨ TILE_ਮਾਪ = get_tile_ਮਾਪ<TB>();
    ਅਚਲ ਤਾਰਕਿਕ need_unpack = do_unpack<TB>::value;

    GGML_ASSERT(M <= 2 * TILE_M && N == 2 * TILE_N);
    ਅਚਲ TA * RESTRICT A = static_cast<ਅਚਲ TA *>(_A);
    ਅਚਲ ਅੱਖਰ * RESTRICT B = static_cast<ਅਚਲ ਅੱਖਰ *>(_B);

    ਅਚਲ ਪੂਰਨ m0 = std::min(M, TILE_M);
    ਅਚਲ ਪੂਰਨ m1 = std::max(M - TILE_M, 0);
    ਅਚਲ ਪੂਰਨ lda = KB * ਆਕਾਰ(TA);
    //const int ldb = KB * sizeof(TB);

    alignas(64) ਸਥਿਰ thread_local packed_B_t Tile0[TILE_N * TILE_K];
    alignas(64) ਸਥਿਰ thread_local packed_B_t Tile1[TILE_N * TILE_K];
    alignas(64) ਸਥਿਰ thread_local int8_t Tile23[TILE_M * TILE_K];

    alignas(64) ਸਥਿਰ thread_local int32_t TileC0[TILE_M * TILE_N * 4];
    alignas(64) ਸਥਿਰ thread_local int32_t TileC1[TILE_M * TILE_N * 4];

    // double buffering C to interleave avx512 and amx
    int32_t * C_cur = TileC0;
    int32_t * C_pre = TileC1;

    ਸਵੈਚਾਲਿਤ Tile4 = [&](int32_t * ਅਧਾਰ) { ਵਾਪਸ ਅਧਾਰ; };
    ਸਵੈਚਾਲਿਤ Tile5 = [&](int32_t * ਅਧਾਰ) { ਵਾਪਸ ਅਧਾਰ + TILE_M * TILE_N; };
    ਸਵੈਚਾਲਿਤ Tile6 = [&](int32_t * ਅਧਾਰ) { ਵਾਪਸ ਅਧਾਰ + 2 * TILE_M * TILE_N; };
    ਸਵੈਚਾਲਿਤ Tile7 = [&](int32_t * ਅਧਾਰ) { ਵਾਪਸ ਅਧਾਰ + 3 * TILE_M * TILE_N; };

    ਜੇ (M == 2 * TILE_M) {
        // i = 0
        ਅਚਲ ਅੱਖਰ * B_blk0 = B + PACKED_ਸੂਚਕਾਂਕ(0, 0, KB, TILE_ਮਾਪ);
        ਅਚਲ ਅੱਖਰ * B_blk1 = B + PACKED_ਸੂਚਕਾਂਕ(1, 0, KB, TILE_ਮਾਪ);
        ਜੇ (need_unpack) {
            unpack_B<TB>(Tile0, B_blk0);
            _tile_loadd(TMM0, Tile0, TILE_N * VNNI_BLK);
        } ਨਹੀਂਤਾਂ {
            _tile_loadd(TMM0, B_blk0, TILE_N * VNNI_BLK);
        }

        _tile_zero(TMM4);
        _tile_loadd(TMM2, A[0].qs, lda);
        _tile_dpbssd(TMM4, TMM2, TMM0);
        _tile_stored(TMM4, Tile4(C_pre), TILE_N * ਆਕਾਰ(int32_t));

        _tile_zero(TMM5);
        _tile_loadd(TMM3, A[TILE_M * KB + 0].qs, lda);
        _tile_dpbssd(TMM5, TMM3, TMM0);
        _tile_stored(TMM5, Tile5(C_pre), TILE_N * ਆਕਾਰ(int32_t));

        ਜੇ (need_unpack) {
            unpack_B<TB>(Tile1, B_blk1);
            _tile_loadd(TMM1, Tile1, TILE_N * VNNI_BLK);
        } ਨਹੀਂਤਾਂ {
            _tile_loadd(TMM1, B_blk1, TILE_N * VNNI_BLK);
        }

        _tile_zero(TMM6);
        _tile_dpbssd(TMM6, TMM2, TMM1);
        _tile_stored(TMM6, Tile6(C_pre), TILE_N * ਆਕਾਰ(int32_t));

        _tile_zero(TMM7);
        _tile_dpbssd(TMM7, TMM3, TMM1);
        _tile_stored(TMM7, Tile7(C_pre), TILE_N * ਆਕਾਰ(int32_t));

        ਲਈ (ਪੂਰਨ i = 1; i < KB; ++i) {
            // index of previous iter
            ਅਚਲ ਪੂਰਨ ii = i - 1;
            ਅਚਲ ਅੱਖਰ * B_blk0 = B + PACKED_ਸੂਚਕਾਂਕ(0, i, KB, TILE_ਮਾਪ);
            ਅਚਲ ਅੱਖਰ * B_blk1 = B + PACKED_ਸੂਚਕਾਂਕ(1, i, KB, TILE_ਮਾਪ);
            GGML_DISPATCH_BOOL(ii > 0, is_acc, [&] {
                ਜੇ (need_unpack) {
                    unpack_B<TB>(Tile0, B_blk0);
                    _tile_loadd(TMM0, Tile0, TILE_N * VNNI_BLK);
                } ਨਹੀਂਤਾਂ {
                    _tile_loadd(TMM0, B_blk0, TILE_N * VNNI_BLK);
                }
                _tile_zero(TMM4);
                _tile_loadd(TMM2, A[i].qs, lda);
                acc_C<TA, TB, is_acc>::apply(C, ldc, Tile4(C_pre), &A[ii], KB, B + PACKED_ਸੂਚਕਾਂਕ(0, ii, KB, TILE_ਮਾਪ), TILE_M);

                _tile_dpbssd(TMM4, TMM2, TMM0);
                _tile_stored(TMM4, Tile4(C_cur), TILE_N * ਆਕਾਰ(int32_t));

                _tile_zero(TMM5);
                _tile_loadd(TMM3, A[TILE_M * KB + i].qs, lda);
                acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc, ldc, Tile5(C_pre), &A[TILE_M * KB + ii], KB, B + PACKED_ਸੂਚਕਾਂਕ(0, ii, KB, TILE_ਮਾਪ), TILE_M);

                _tile_dpbssd(TMM5, TMM3, TMM0);
                _tile_stored(TMM5, Tile5(C_cur), TILE_N * ਆਕਾਰ(int32_t));

                ਜੇ (need_unpack) {
                    unpack_B<TB>(Tile1, B_blk1);
                    _tile_loadd(TMM1, Tile1, TILE_N * VNNI_BLK);
                } ਨਹੀਂਤਾਂ {
                    _tile_loadd(TMM1, B_blk1, TILE_N * VNNI_BLK);
                }
                _tile_zero(TMM6);
                acc_C<TA, TB, is_acc>::apply(C + TILE_N, ldc, Tile6(C_pre), &A[ii], KB, B + PACKED_ਸੂਚਕਾਂਕ(1, ii, KB, TILE_ਮਾਪ), TILE_M);

                _tile_dpbssd(TMM6, TMM2, TMM1);
                _tile_stored(TMM6, Tile6(C_cur), TILE_N * ਆਕਾਰ(int32_t));

                _tile_zero(TMM7);
                acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc + TILE_N, ldc, Tile7(C_pre), &A[TILE_M * KB + ii], KB, B + PACKED_ਸੂਚਕਾਂਕ(1, ii, KB, TILE_ਮਾਪ), TILE_M);

                _tile_dpbssd(TMM7, TMM3, TMM1);
                _tile_stored(TMM7, Tile7(C_cur), TILE_N * ਆਕਾਰ(int32_t));

                std::swap(C_cur, C_pre);
            });
        }
        // final accumulation
        {
            ਪੂਰਨ ii = KB - 1;
            acc_C<TA, TB, ਸੱਚ>::apply(C, ldc, Tile4(C_pre), &A[ii], KB, B + PACKED_ਸੂਚਕਾਂਕ(0, ii, KB, TILE_ਮਾਪ), TILE_M);
            acc_C<TA, TB, ਸੱਚ>::apply(C + TILE_M * ldc, ldc, Tile5(C_pre), &A[TILE_M * KB + ii], KB, B + PACKED_ਸੂਚਕਾਂਕ(0, ii, KB, TILE_ਮਾਪ), TILE_M);
            acc_C<TA, TB, ਸੱਚ>::apply(C + TILE_N, ldc, Tile6(C_pre), &A[ii], KB, B + PACKED_ਸੂਚਕਾਂਕ(1, ii, KB, TILE_ਮਾਪ), TILE_M);
            acc_C<TA, TB, ਸੱਚ>::apply(C + TILE_M * ldc + TILE_N, ldc, Tile7(C_pre), &A[TILE_M * KB + ii], KB, B + PACKED_ਸੂਚਕਾਂਕ(1, ii, KB, TILE_ਮਾਪ), TILE_M);
        }
    } ਨਹੀਂਤਾਂ {
        ਲਈ (ਪੂਰਨ i = 0; i < KB; ++i) {
            _tile_zero(TMM4);
            _tile_zero(TMM6);
            ਜੇ (m1 != 0) {
                _tile_zero(TMM5);
                _tile_zero(TMM7);
            }

            ਅਚਲ ਅੱਖਰ * B_blk0 = B + PACKED_ਸੂਚਕਾਂਕ(0, i, KB, TILE_ਮਾਪ);
            ਅਚਲ ਅੱਖਰ * B_blk1 = B + PACKED_ਸੂਚਕਾਂਕ(1, i, KB, TILE_ਮਾਪ);
            ਜੇ (need_unpack) {
                unpack_B<TB>(Tile0, B_blk0);
                _tile_loadd(TMM0, Tile0, TILE_N * VNNI_BLK);
            } ਨਹੀਂਤਾਂ {
                _tile_loadd(TMM0, B_blk0, TILE_N * VNNI_BLK);
            }

            ਜੇ (need_unpack) {
                unpack_B<TB>(Tile1, B_blk1);
                _tile_loadd(TMM1, Tile1, TILE_N * VNNI_BLK);
            } ਨਹੀਂਤਾਂ {
                _tile_loadd(TMM1, B_blk1, TILE_N * VNNI_BLK);
            }

            ਜੇ (m0 == TILE_M) {
                _tile_loadd(TMM2, A[i].qs, lda);
            } ਨਹੀਂਤਾਂ {
                unpack_A(Tile23, &A[i], KB, m0);
                _tile_loadd(TMM2, Tile23, TILE_K);
            }

            _tile_dpbssd(TMM4, TMM2, TMM0);
            _tile_dpbssd(TMM6, TMM2, TMM1);

            _tile_stored(TMM4, Tile4(C_cur), TILE_N * ਆਕਾਰ(int32_t));
            _tile_stored(TMM6, Tile6(C_cur), TILE_N * ਆਕਾਰ(int32_t));

            GGML_DISPATCH_BOOL(i > 0, is_acc, [&] {
                acc_C<TA, TB, is_acc>::apply(C,          ldc, Tile4(C_cur), &A[i], KB, B + PACKED_ਸੂਚਕਾਂਕ(0, i, KB, TILE_ਮਾਪ), m0);
                acc_C<TA, TB, is_acc>::apply(C + TILE_N, ldc, Tile6(C_cur), &A[i], KB, B + PACKED_ਸੂਚਕਾਂਕ(1, i, KB, TILE_ਮਾਪ), m0);
            });

            ਜੇ (m1 != 0) {
                unpack_A(Tile23, &A[TILE_M * KB + i], KB, m1);
                _tile_loadd(TMM3, Tile23, TILE_K);

                _tile_dpbssd(TMM5, TMM3, TMM0);
                _tile_dpbssd(TMM7, TMM3, TMM1);
                _tile_stored(TMM5, Tile5(C_cur), TILE_N * ਆਕਾਰ(int32_t));
                _tile_stored(TMM7, Tile7(C_cur), TILE_N * ਆਕਾਰ(int32_t));
                GGML_DISPATCH_BOOL(i > 0, is_acc, [&] {
                    acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc,          ldc, Tile5(C_cur), &A[TILE_M * KB + i], KB, B + PACKED_ਸੂਚਕਾਂਕ(0, i, KB, TILE_ਮਾਪ), m1);
                    acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc + TILE_N, ldc, Tile7(C_cur), &A[TILE_M * KB + i], KB, B + PACKED_ਸੂਚਕਾਂਕ(1, i, KB, TILE_ਮਾਪ), m1);
                });
            }
        }
    }
    ਵਾਪਸ;
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ TA, ਕਿਸਮ_ਨਾਂ TB, ਕਿਸਮ_ਨਾਂ TC, ਪੂਰਨ ਖੰਡ_K,
          ਕਿਸਮ_ਨਾਂ std::enable_if<is_type_qkk<TB>::value, ਪੂਰਨ>::type = 0>
ਖਾਲੀ tinygemm_kernel_amx(ਪੂਰਨ M, ਪੂਰਨ N, ਪੂਰਨ KB, ਅਚਲ ਖਾਲੀ * RESTRICT _A, ਅਚਲ ਖਾਲੀ * RESTRICT _B, ਦਸ਼ਮਲਵ * RESTRICT C, ਪੂਰਨ ldc) {
    static_assert(std::is_same<TA, ਖੰਡ_q8_K>::value);
    ਅਚਲ ਪੂਰਨ TILE_ਮਾਪ = get_tile_ਮਾਪ<TB>();

    GGML_ASSERT(M <= 2 * TILE_M && N == 2 * TILE_N);
    ਅਚਲ TA * RESTRICT A = static_cast<ਅਚਲ TA *>(_A);
    ਅਚਲ ਅੱਖਰ * RESTRICT B = static_cast<ਅਚਲ ਅੱਖਰ *>(_B);

    ਅਚਲ ਪੂਰਨ m0 = std::min(M, TILE_M);
    ਅਚਲ ਪੂਰਨ m1 = std::max(M - TILE_M, 0);
    //const int lda = KB * sizeof(TA);

    alignas(64) ਸਥਿਰ thread_local int8_t Tile0[TILE_N * TILE_K];
    alignas(64) ਸਥਿਰ thread_local int8_t Tile1[TILE_N * TILE_K];
    alignas(64) ਸਥਿਰ thread_local int8_t Tile23[TILE_M * TILE_K];

    // mat mul result for each group
    alignas(64) ਸਥਿਰ thread_local int32_t Tile4[TILE_M * TILE_N];
    alignas(64) ਸਥਿਰ thread_local int32_t Tile5[TILE_M * TILE_N];
    alignas(64) ਸਥਿਰ thread_local int32_t Tile6[TILE_M * TILE_N];
    alignas(64) ਸਥਿਰ thread_local int32_t Tile7[TILE_M * TILE_N];

    // sum of each QK_K block, contains 8 groups, int32
    alignas(64) ਸਥਿਰ thread_local int32_t Sumi4[TILE_M * TILE_N];
    alignas(64) ਸਥਿਰ thread_local int32_t Sumi5[TILE_M * TILE_N];
    alignas(64) ਸਥਿਰ thread_local int32_t Sumi6[TILE_M * TILE_N];
    alignas(64) ਸਥਿਰ thread_local int32_t Sumi7[TILE_M * TILE_N];

    ਅਚਲ ਪੂਰਨ k_ਸਮੂਹ_ਮਾਪ = std::is_same<TB, ਖੰਡ_q6_K>::value ? 16 : 32;
    ਲਈ (ਪੂਰਨ i = 0; i < KB; ++i) {
        // step 1: accumulate the quants across 8 groups, each group with 32
        ਲਈ (ਪੂਰਨ k = 0; k < QK_K / k_ਸਮੂਹ_ਮਾਪ; ++k) {
            GGML_DISPATCH_BOOL(k > 0, is_acc, [&] {
                _tile_zero(TMM4);
                _tile_zero(TMM6);

                unpack_B<TB>(Tile0, B + PACKED_ਸੂਚਕਾਂਕ(0, i, KB, TILE_ਮਾਪ), k);
                _tile_loadd(TMM0, Tile0, TILE_N * VNNI_BLK);

                unpack_B<TB>(Tile1, B + PACKED_ਸੂਚਕਾਂਕ(1, i, KB, TILE_ਮਾਪ), k);
                _tile_loadd(TMM1, Tile1, TILE_N * VNNI_BLK);

                unpack_A<TB>(Tile23, &A[i], KB, k, m0);
                _tile_loadd(TMM2, Tile23, TILE_K);

                _tile_dpbssd(TMM4, TMM2, TMM0);
                _tile_dpbssd(TMM6, TMM2, TMM1);

                _tile_stored(TMM4, Tile4, TILE_N * ਆਕਾਰ(int32_t));
                _tile_stored(TMM6, Tile6, TILE_N * ਆਕਾਰ(int32_t));

                scale_C<TB, is_acc>(Tile4, Sumi4, B + PACKED_ਸੂਚਕਾਂਕ(0, i, KB, TILE_ਮਾਪ), k, m0);
                scale_C<TB, is_acc>(Tile6, Sumi6, B + PACKED_ਸੂਚਕਾਂਕ(1, i, KB, TILE_ਮਾਪ), k, m0);

                ਜੇ (m1 != 0) {
                    _tile_zero(TMM5);
                    _tile_zero(TMM7);

                    unpack_A<TB>(Tile23, &A[TILE_M * KB + i], KB, k, m1);
                    _tile_loadd(TMM3, Tile23, TILE_K);

                    _tile_dpbssd(TMM5, TMM3, TMM0);
                    _tile_dpbssd(TMM7, TMM3, TMM1);

                    _tile_stored(TMM5, Tile5, TILE_N * ਆਕਾਰ(int32_t));
                    _tile_stored(TMM7, Tile7, TILE_N * ਆਕਾਰ(int32_t));

                    scale_C<TB, is_acc>(Tile5, Sumi5, B + PACKED_ਸੂਚਕਾਂਕ(0, i, KB, TILE_ਮਾਪ), k, m1);
                    scale_C<TB, is_acc>(Tile7, Sumi7, B + PACKED_ਸੂਚਕਾਂਕ(1, i, KB, TILE_ਮਾਪ), k, m1);
                }
            });
        }

        // step 2: accmulate the mins
        GGML_DISPATCH_BOOL(i > 0, is_acc, [&] {
            acc_C<TA, TB, is_acc>::apply(C,          ldc, Sumi4, &A[i], KB, B + PACKED_ਸੂਚਕਾਂਕ(0, i, KB, TILE_ਮਾਪ), m0);
            acc_C<TA, TB, is_acc>::apply(C + TILE_N, ldc, Sumi6, &A[i], KB, B + PACKED_ਸੂਚਕਾਂਕ(1, i, KB, TILE_ਮਾਪ), m0);
            ਜੇ (m1 != 0) {
                acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc,          ldc, Sumi5, &A[TILE_M * KB + i], KB, B + PACKED_ਸੂਚਕਾਂਕ(0, i, KB, TILE_ਮਾਪ), m1);
                acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc + TILE_N, ldc, Sumi7, &A[TILE_M * KB + i], KB, B + PACKED_ਸੂਚਕਾਂਕ(1, i, KB, TILE_ਮਾਪ), m1);
            }
        });
    }
    ਵਾਪਸ;
}

} // anonymous namespace

// get the packed tensor size for quantized weights
size_t ggml_backend_amx_get_alloc_ਮਾਪ(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor) {
    ਅਚਲ ਗਿਣਤੀ ggml_type TYPE = tensor->type;

    ਅਚਲ ਪੂਰਨ K = tensor->ne[0]; // ne0: in_features
    ਅਚਲ ਪੂਰਨ N = tensor->ne[1]; // ne1: out_features

    ਸਵੈਚਾਲਿਤ get_tensor_ਮਾਪ = [&] {
        size_t ਪੰਕਤੀ_ਮਾਪ_B{0};
        GGML_DISPATCH_QTYPES(TYPE, [&] {
            ਪੰਕਤੀ_ਮਾਪ_B = get_ਪੰਕਤੀ_ਮਾਪ<type, blck_ਮਾਪ>(K);
        });
        ਵਾਪਸ N * ਪੰਕਤੀ_ਮਾਪ_B;
    };

    ਜੇ (qtype_has_amx_kernels(TYPE)) {
        ਵਾਪਸ get_tensor_ਮਾਪ();
    } ਨਹੀਂਤਾਂ {
        // for f16, bf16 we don't do packing
        ਵਾਪਸ ggml_nbytes(tensor);
    }
}

// pack weight to vnni format
ਖਾਲੀ ggml_backend_amx_convert_ਭਾਰ(ਬਣਤਰ ggml_tensor * tensor, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size) {
    GGML_ASSERT(ਵਿੱਥ == 0 && size == ggml_nbytes(tensor)); // only full tensor conversion is supported for now

    ਅਚਲ ਗਿਣਤੀ ggml_type TYPE = tensor->type;

    ਅਚਲ ਪੂਰਨ K = tensor->ne[0]; // ne0: in_features
    ਅਚਲ ਪੂਰਨ N = tensor->ne[1]; // ne1: out_features

    GGML_DISPATCH_QTYPES(TYPE, [&] {
        convert_B_packed_format<type, blck_ਮਾਪ>((ਖਾਲੀ *)((ਅੱਖਰ *)tensor->ਅੰਕੜੇ + ਵਿੱਥ), (ਅਚਲ type *)ਅੰਕੜੇ, N, K);
    });
}

// ne2 is passed explicitly to help compiler optimize repeated calls
ਸਤਰਅੰਦਰ int64_t ggml_batch_ਵਿੱਥ(ਅਚਲ ggml_tensor * t, int64_t batch_idx, int64_t ne2) {
    ਅਚਲ int64_t i2 = batch_idx % ne2;
    ਅਚਲ int64_t i3 = batch_idx / ne2;
    ਵਾਪਸ i3 * t->nb[3] + i2 * t->nb[2];
}

size_t ggml_backend_amx_desired_wsize(ਅਚਲ ਬਣਤਰ ggml_tensor * dst) {
    ਬਣਤਰ ggml_tensor * src0 = dst->src[0];

    ਅਚਲ ਗਿਣਤੀ ggml_type TYPE = src0->type;

    ਅਚਲ ਤਾਰਕਿਕ is_floating_type = TYPE == GGML_TYPE_F16;
    ਜੇ (is_floating_type) {
        ਵਾਪਸ 0;
    }

    ਅਚਲ ਪੂਰਨ M = dst->ne[1];
    ਅਚਲ ਪੂਰਨ K = src0->ne[0];
    ਅਚਲ int64_t n_batch = dst->ne[2] * dst->ne[3];

    size_t desired_wsize = 0;

    GGML_DISPATCH_QTYPES(TYPE, [&] {
        ਅਚਲ size_t ਪੰਕਤੀ_ਮਾਪ_A = K / blck_ਮਾਪ * ਆਕਾਰ(vec_dot_type);
        desired_wsize = n_batch * M * ਪੰਕਤੀ_ਮਾਪ_A;
    });

    ਵਾਪਸ desired_wsize;
}

// NB: mixed dtype gemm with Advanced Matrix Extensions (Intel AMX)
//
// src0: weight in shape of {N, K}, quantized
// src1: input  in shape of {M, K}, float32
// dst:  output in shape of {M, N}, float32
//
// the function performs: dst = src1 @ src0.T for each batch
//
ਖਾਲੀ ggml_backend_amx_mul_mat(ਅਚਲ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst) {
    ਬਣਤਰ ggml_tensor * src0 = dst->src[0];
    ਬਣਤਰ ggml_tensor * src1 = dst->src[1];

    ਅਚਲ ਗਿਣਤੀ ggml_type TYPE = src0->type;

    // f16 only has avx512 kernels for now,
    // amx kernels will be added once 6th gen xeon is released.
    ਅਚਲ ਤਾਰਕਿਕ is_floating_type = TYPE == GGML_TYPE_F16;

    ਅਚਲ ਪੂਰਨ M = dst->ne[1];
    ਅਚਲ ਪੂਰਨ N = dst->ne[0];
    ਅਚਲ ਪੂਰਨ K = src0->ne[0];
    ਅਚਲ ਪੂਰਨ ldc = dst->nb[1] / dst->nb[0];

    ਅਚਲ int64_t ne2 = dst->ne[2];
    ਅਚਲ int64_t n_batch = ne2 * dst->ne[3];

    ਜੇ (is_floating_type) {
        constexpr ਪੂਰਨ ਖੰਡ_M = 4;
        constexpr ਪੂਰਨ ਖੰਡ_N = 6;
        ਅਚਲ ਪੂਰਨ MB = div_ਉੱਪਰ(M, ਖੰਡ_M);
        ਅਚਲ ਪੂਰਨ NB = div_ਉੱਪਰ(N, ਖੰਡ_N);

        parallel_for_ggml(params, n_batch * MB * NB, [&](ਪੂਰਨ begin, ਪੂਰਨ end) {
            GGML_DISPATCH_FLOATING_TYPES(TYPE, [&] {
                ਲਈ (ਪੂਰਨ i = begin; i < end; ++i) {
                    ਪੂਰਨ batch_idx = i / (MB * NB);
                    ਪੂਰਨ ਬਚਿਆ = i % (MB * NB);
                    ਪੂਰਨ mb = ਬਚਿਆ / NB;
                    ਪੂਰਨ nb = ਬਚਿਆ % NB;

                    int64_t src0_ਵਿੱਥ = ggml_batch_ਵਿੱਥ(src0, batch_idx, ne2);
                    int64_t src1_ਵਿੱਥ = ggml_batch_ਵਿੱਥ(src1, batch_idx, ne2);
                    int64_t dst_ਵਿੱਥ  = ggml_batch_ਵਿੱਥ(dst,  batch_idx, ne2);

                    ਪੂਰਨ mb_ਸ਼ੁਰੂ = mb * ਖੰਡ_M;
                    ਪੂਰਨ mb_ਮਾਪ = std::min(ਖੰਡ_M, M - mb_ਸ਼ੁਰੂ);
                    ਪੂਰਨ nb_ਸ਼ੁਰੂ = nb * ਖੰਡ_N;
                    ਪੂਰਨ nb_ਮਾਪ = std::min(ਖੰਡ_N, N - nb_ਸ਼ੁਰੂ);

                    ਚੋਣ (mb_ਮਾਪ << 4 | nb_ਮਾਪ) {
                        ਹਾਲਤ 0x12: LAUNCH_TINYGEMM_KERNEL_AVX(1, 2); ਰੋਕੋ;
                        ਹਾਲਤ 0x14: LAUNCH_TINYGEMM_KERNEL_AVX(1, 4); ਰੋਕੋ;
                        ਹਾਲਤ 0x16: LAUNCH_TINYGEMM_KERNEL_AVX(1, 6); ਰੋਕੋ;
                        ਹਾਲਤ 0x22: LAUNCH_TINYGEMM_KERNEL_AVX(2, 2); ਰੋਕੋ;
                        ਹਾਲਤ 0x24: LAUNCH_TINYGEMM_KERNEL_AVX(2, 4); ਰੋਕੋ;
                        ਹਾਲਤ 0x26: LAUNCH_TINYGEMM_KERNEL_AVX(2, 6); ਰੋਕੋ;
                        ਹਾਲਤ 0x32: LAUNCH_TINYGEMM_KERNEL_AVX(3, 2); ਰੋਕੋ;
                        ਹਾਲਤ 0x34: LAUNCH_TINYGEMM_KERNEL_AVX(3, 4); ਰੋਕੋ;
                        ਹਾਲਤ 0x36: LAUNCH_TINYGEMM_KERNEL_AVX(3, 6); ਰੋਕੋ;
                        ਹਾਲਤ 0x42: LAUNCH_TINYGEMM_KERNEL_AVX(4, 2); ਰੋਕੋ;
                        ਹਾਲਤ 0x44: LAUNCH_TINYGEMM_KERNEL_AVX(4, 4); ਰੋਕੋ;
                        ਹਾਲਤ 0x46: LAUNCH_TINYGEMM_KERNEL_AVX(4, 6); ਰੋਕੋ;
                        ਮੂਲ: fprintf(stderr, "Unexpected block size!\n");
                    }
                }
            });
        });
        ਵਾਪਸ;
    }

    // pointer to work space, used convert A from float to quantized type
    ਖਾਲੀ * wdata = params->wdata;

    //TODO: performance improvement: merge quant A
 // if (params->ith == 0) {
        GGML_DISPATCH_QTYPES(TYPE, [&] {
            ਅਚਲ size_t ਪੰਕਤੀ_ਮਾਪ_A = K / blck_ਮਾਪ * ਆਕਾਰ(vec_dot_type);
            ਅਚਲ size_t desired_wsize = n_batch * M * ਪੰਕਤੀ_ਮਾਪ_A;
            ਜੇ (params->wsize < desired_wsize) {
                GGML_ABORT("insufficient work space size");
            }

            // Q4_0, Q4_1, Q8_0 handles 1 TILE_K per blck_size
            // Q4_K, Q5_K, Q6_K, IQ4_XS handles 8 TILE_K per blck_size
            GGML_ASSERT(TILE_K == blck_ਮਾਪ || TILE_K * 8 == blck_ਮਾਪ);

            parallel_for_ggml(params, n_batch * M, [&](ਪੂਰਨ begin, ਪੂਰਨ end) {
                ਲਈ (ਪੂਰਨ idx = begin; idx < end; ++idx) {
                    ਪੂਰਨ batch_idx = idx / M;
                    ਪੂਰਨ m         = idx % M;
                    int64_t src1_ਵਿੱਥ = ggml_batch_ਵਿੱਥ(src1, batch_idx, ne2);
                    ਅਚਲ ਦਸ਼ਮਲਵ * A_ਅੰਕੜੇ = (ਅਚਲ ਦਸ਼ਮਲਵ *)((ਅਚਲ ਅੱਖਰ *)src1->ਅੰਕੜੇ + src1_ਵਿੱਥ);
                    ਅੱਖਰ * wdata_batch = (ਅੱਖਰ *)wdata + batch_idx * M * ਪੰਕਤੀ_ਮਾਪ_A;
                    from_float<vec_dot_type>(A_ਅੰਕੜੇ + m * K, wdata_batch + m * ਪੰਕਤੀ_ਮਾਪ_A, K);
                }
            });
        });
 // }

    ggml_barrier(params->threadpool);

    ਜੇ (M == 1) {
        // MB = 1 and handle 8 tiles in each block
        constexpr ਪੂਰਨ kTilesN = 4;
        constexpr ਪੂਰਨ ਖੰਡ_N = TILE_N * kTilesN;
        ਅਚਲ ਪੂਰਨ NB = div_ਉੱਪਰ(N, ਖੰਡ_N);

        parallel_for_ggml(params, n_batch * NB, [&](ਪੂਰਨ begin, ਪੂਰਨ end) {
            GGML_DISPATCH_QTYPES(TYPE, [&] {
                ਅਚਲ ਪੂਰਨ KB = K / blck_ਮਾਪ;
                ਅਚਲ ਪੂਰਨ TILE_ਮਾਪ = get_tile_ਮਾਪ<type>();
                ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ_ਮਾਪ_A = KB * ਆਕਾਰ(vec_dot_type);
                ਲਈ (ਪੂਰਨ i = begin; i < end; ++i) {
                    ਪੂਰਨ batch_idx = i / NB;
                    ਪੂਰਨ nb = i % NB;

                    int64_t src0_ਵਿੱਥ = ggml_batch_ਵਿੱਥ(src0, batch_idx, ne2);
                    int64_t dst_ਵਿੱਥ  = ggml_batch_ਵਿੱਥ(dst,  batch_idx, ne2);
                    ਅਚਲ ਅੱਖਰ * wdata_batch = (ਅਚਲ ਅੱਖਰ *)wdata + batch_idx * ਪੰਕਤੀ_ਮਾਪ_A;

                    ਪੂਰਨ nb_ਸ਼ੁਰੂ = nb * ਖੰਡ_N;
                    ਪੂਰਨ nb_ਮਾਪ = std::min(ਖੰਡ_N, N - nb_ਸ਼ੁਰੂ); // 32, 64, 96

                    ਚੋਣ (nb_ਮਾਪ) {
                        //case 160: LAUNCH_TINYGEMM_KERNEL_VNNI(160); break;
                        ਹਾਲਤ 128: LAUNCH_TINYGEMM_KERNEL_VNNI(128); ਰੋਕੋ;
                        ਹਾਲਤ 96: LAUNCH_TINYGEMM_KERNEL_VNNI(96); ਰੋਕੋ;
                        ਹਾਲਤ 64: LAUNCH_TINYGEMM_KERNEL_VNNI(64); ਰੋਕੋ;
                        ਹਾਲਤ 32: LAUNCH_TINYGEMM_KERNEL_VNNI(32); ਰੋਕੋ;
                        ਮੂਲ: fprintf(stderr, "Unexpected n block size!\n");
                    }
                }
            });
        });
        ਵਾਪਸ;
    }

    // handle 4 tiles at a tile
    constexpr ਪੂਰਨ ਖੰਡ_M = TILE_M * 2;
    constexpr ਪੂਰਨ ਖੰਡ_N = TILE_N * 2;
    ਅਚਲ ਪੂਰਨ MB = div_ਉੱਪਰ(M, ਖੰਡ_M);
    ਅਚਲ ਪੂਰਨ NB = div_ਉੱਪਰ(N, ਖੰਡ_N);

    parallel_for_ggml(params, n_batch * MB * NB, [&](ਪੂਰਨ begin, ਪੂਰਨ end) {
        // init tile config for each thread
        ggml_tile_config_init();

        GGML_DISPATCH_QTYPES(TYPE, [&] {
            ਅਚਲ ਪੂਰਨ KB = K / blck_ਮਾਪ;
            ਅਚਲ ਪੂਰਨ TILE_ਮਾਪ = get_tile_ਮਾਪ<type>();
            ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ_ਮਾਪ_A = KB * ਆਕਾਰ(vec_dot_type);

            ਲਈ (ਪੂਰਨ i = begin; i < end; ++i) {
                ਪੂਰਨ batch_idx = i / (MB * NB);
                ਪੂਰਨ ਬਚਿਆ = i % (MB * NB);
                ਪੂਰਨ mb = ਬਚਿਆ / NB;
                ਪੂਰਨ nb = ਬਚਿਆ % NB;

                int64_t src0_ਵਿੱਥ = ggml_batch_ਵਿੱਥ(src0, batch_idx, ne2);
                int64_t dst_ਵਿੱਥ  = ggml_batch_ਵਿੱਥ(dst,  batch_idx, ne2);
                ਅਚਲ ਅੱਖਰ * wdata_batch = (ਅਚਲ ਅੱਖਰ *)wdata + batch_idx * M * ਪੰਕਤੀ_ਮਾਪ_A;

                ਪੂਰਨ mb_ਸ਼ੁਰੂ = mb * ਖੰਡ_M;
                ਪੂਰਨ mb_ਮਾਪ = std::min(ਖੰਡ_M, M - mb_ਸ਼ੁਰੂ);
                ਪੂਰਨ nb_ਸ਼ੁਰੂ = nb * ਖੰਡ_N;
                ਪੂਰਨ nb_ਮਾਪ = ਖੰਡ_N;

                tinygemm_kernel_amx<vec_dot_type, type, ਦਸ਼ਮਲਵ, blck_ਮਾਪ>(
                    mb_ਮਾਪ, nb_ਮਾਪ, KB,
                    wdata_batch + mb_ਸ਼ੁਰੂ * ਪੰਕਤੀ_ਮਾਪ_A,
                    (ਅਚਲ ਅੱਖਰ *)src0->ਅੰਕੜੇ + src0_ਵਿੱਥ + PACKED_ਸੂਚਕਾਂਕ(nb * 2, 0, KB, TILE_ਮਾਪ),
                    (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ + dst_ਵਿੱਥ + mb_ਸ਼ੁਰੂ * N + nb_ਸ਼ੁਰੂ, ldc);
            }
        });
    });
}

#ਜੇ_ਅੰਤ // if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
