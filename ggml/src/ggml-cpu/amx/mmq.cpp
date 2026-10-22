#ఒకవేళ defined(__GNUC__)
#సూచన GCC diagnostic ignored "-Wpedantic"
#సూచన GCC diagnostic ignored "-Wunused-local-typedefs"
#అయితే_ముగింపు

#అన్తర్భావ "amx.h"
#అన్తర్భావ "mmq.h"
#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "ggml-cpu-impl.h"
#అన్తర్భావ "simd-mappings.h"
#అన్తర్భావ "quants.h"
#అన్తర్భావ "ggml-quants.h"
#అన్తర్భావ <algorithm>
#అన్తర్భావ <type_traits>

#ఒకవేళ defined(__gnu_linux__)
#అన్తర్భావ <sys/syscall.h>
#అన్తర్భావ <unistd.h>
#అయితే_ముగింపు

#ఒకవేళ (defined(_WIN32) || defined(_WIN64))
#నిర్వచించు RESTRICT __restrict
#లేకపోతే
#నిర్వచించు RESTRICT __restrict__
#అయితే_ముగింపు

#ఒకవేళ (defined(_WIN32) || defined(_WIN64))
#నిర్వచించు ALWAYS_INLINE __forceinline
#లేదావేళ __has_attribute(always_inline) || defined(__GNUC__)
#నిర్వచించు ALWAYS_INLINE __attribute__((__always_inline__)) అంతఃవరుస
#లేకపోతే
#నిర్వచించు ALWAYS_INLINE అంతఃవరుస
#అయితే_ముగింపు

#ఒకవేళ defined(__AMX_INT8__) && defined(__AVX512VNNI__)

నామాకాశ {

// Forced unrolling
మూస <పూర్ణాంకం n>
నిర్మాణం Unroll {
    మూస <రకం_పేరు Func, రకం_పేరు... వాదనలు>
    ALWAYS_INLINE శూన్య క్రియాకారకం()(స్థిరరాశి Func& f, వాదనలు... వాదనలు_2) స్థిరరాశి {
        Unroll<n - 1>{}(f, వాదనలు_2...);
        f(std::integral_constant<పూర్ణాంకం, n - 1>{}, వాదనలు_2...);
    }
};

మూస <>
నిర్మాణం Unroll<1> {
    మూస <రకం_పేరు Func, రకం_పేరు... వాదనలు>
    ALWAYS_INLINE శూన్య క్రియాకారకం()(స్థిరరాశి Func& f, వాదనలు... వాదనలు_2) స్థిరరాశి {
        f(std::integral_constant<పూర్ణాంకం, 0>{}, వాదనలు_2...);
    }
};

// type traits
మూస <రకం_పేరు T> నిర్మాణం PackedTypes {};
మూస <> నిర్మాణం PackedTypes<ఖండం_q4_0> { ఉపయోగించి type = int8_t; };
మూస <> నిర్మాణం PackedTypes<ఖండం_q4_1> { ఉపయోగించి type = uint8_t; };
మూస <> నిర్మాణం PackedTypes<ఖండం_q8_0> { ఉపయోగించి type = int8_t; };
మూస <రకం_పేరు T> ఉపయోగించి packed_B_type = రకం_పేరు PackedTypes<T>::type;

మూస <రకం_పేరు T>
నిర్మాణం do_compensate : std::integral_constant<తార్కిక,
    std::is_same<T, ఖండం_q8_0>::value> {};

మూస <రకం_పేరు T>
నిర్మాణం do_unpack : std::integral_constant<తార్కిక,
    std::is_same<T, ఖండం_q4_0>::value ||
    std::is_same<T, ఖండం_q4_1>::value> {};

మూస <రకం_పేరు T>
నిర్మాణం is_type_qkk : std::integral_constant<తార్కిక,
    std::is_same<T, ఖండం_q4_K>::value ||
    std::is_same<T, ఖండం_q5_K>::value ||
    std::is_same<T, ఖండం_q6_K>::value ||
    std::is_same<T, ఖండం_iq4_xs>::value> {};

#నిర్వచించు GGML_DISPATCH_FLOATING_TYPES(TYPE, ...)                                        \
    [&] {                                                                              \
        ఎంపిక (TYPE) {                                                                \
            స్థితి GGML_TYPE_F16: {                                                      \
                ఉపయోగించి type = ggml_fp16_t;                                              \
                constexpr పూర్ణాంకం blck_కొలత = 16;                                          \
                తిరిగి __VA_ARGS__();                                                  \
            }                                                                          \
            స్థితి GGML_TYPE_BF16: {                                                     \
                ఉపయోగించి type = ggml_bf16_t;                                              \
                constexpr పూర్ణాంకం blck_కొలత = 32;                                          \
                తిరిగి __VA_ARGS__();                                                  \
            }                                                                          \
            అప్రమేయ:                                                                   \
                fprintf(stderr, "Unsupported floating data type\n");                   \
        }                                                                              \
    }()

#నిర్వచించు GGML_DISPATCH_QTYPES(QT, ...)                                                  \
    [&] {                                                                              \
        ఎంపిక (QT) {                                                                  \
            స్థితి GGML_TYPE_Q4_0: {                                                     \
                ఉపయోగించి type = ఖండం_q4_0;                                               \
                ఉపయోగించి vec_dot_type = ఖండం_q8_0;                                       \
                constexpr పూర్ణాంకం blck_కొలత = QK4_0;                                       \
                తిరిగి __VA_ARGS__();                                                  \
            }                                                                          \
            స్థితి GGML_TYPE_Q4_1: {                                                     \
                ఉపయోగించి type = ఖండం_q4_1;                                               \
                ఉపయోగించి vec_dot_type = ఖండం_q8_1;                                       \
                constexpr పూర్ణాంకం blck_కొలత = QK4_1;                                       \
                తిరిగి __VA_ARGS__();                                                  \
            }                                                                          \
            స్థితి GGML_TYPE_Q8_0: {                                                     \
                ఉపయోగించి type = ఖండం_q8_0;                                               \
                ఉపయోగించి vec_dot_type = ఖండం_q8_0;                                       \
                constexpr పూర్ణాంకం blck_కొలత = QK8_0;                                       \
                తిరిగి __VA_ARGS__();                                                  \
            }                                                                          \
            స్థితి GGML_TYPE_Q4_K: {                                                     \
                ఉపయోగించి type = ఖండం_q4_K;                                               \
                ఉపయోగించి vec_dot_type = ఖండం_q8_K;                                       \
                constexpr పూర్ణాంకం blck_కొలత = QK_K;                                        \
                తిరిగి __VA_ARGS__();                                                  \
            }                                                                          \
            స్థితి GGML_TYPE_Q5_K: {                                                     \
                ఉపయోగించి type = ఖండం_q5_K;                                               \
                ఉపయోగించి vec_dot_type = ఖండం_q8_K;                                       \
                constexpr పూర్ణాంకం blck_కొలత = QK_K;                                        \
                తిరిగి __VA_ARGS__();                                                  \
            }                                                                          \
            స్థితి GGML_TYPE_Q6_K: {                                                     \
                ఉపయోగించి type = ఖండం_q6_K;                                               \
                ఉపయోగించి vec_dot_type = ఖండం_q8_K;                                       \
                constexpr పూర్ణాంకం blck_కొలత = QK_K;                                        \
                తిరిగి __VA_ARGS__();                                                  \
            }                                                                          \
            స్థితి GGML_TYPE_IQ4_XS: {                                                   \
                ఉపయోగించి type = ఖండం_iq4_xs;                                             \
                ఉపయోగించి vec_dot_type = ఖండం_q8_K;                                       \
                constexpr పూర్ణాంకం blck_కొలత = QK_K;                                        \
                తిరిగి __VA_ARGS__();                                                  \
            }                                                                          \
            అప్రమేయ:                                                                   \
                fprintf(stderr, "Unsupported quantized data type: %d\n", పూర్ణాంకం(TYPE));   \
        }                                                                              \
    }()

#నిర్వచించు GGML_DISPATCH_BOOL(BOOL_V, BOOL_NAME, ...)                                     \
    [&] {                                                                              \
        ఒకవేళ (BOOL_V) {                                                                  \
            constexpr తార్కిక BOOL_NAME = నిజం;                                           \
            తిరిగి __VA_ARGS__();                                                      \
        } లేకపోతే {                                                                       \
            constexpr తార్కిక BOOL_NAME = అబద్ధం;                                          \
            తిరిగి __VA_ARGS__();                                                      \
        }                                                                              \
    }()

// define amx tile config data structure
నిర్మాణం tile_config_t{
    uint8_t palette_id = 0;
    uint8_t ప్రారంభం_అడ్డువరుస = 0;
    uint8_t reserved_0[14] = {0};
    uint16_t colsb[16] = {0};
    uint8_t అడ్డువరుసలు[16] = {0};
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

అంతఃవరుస శూన్య ggml_tile_config_init(శూన్య) {
    స్థిర thread_local తార్కిక పూర్తి = అబద్ధం;

    ఒకవేళ (పూర్తి) {
        తిరిగి;
    }

    alignas(64) tile_config_t tc = {};
    tc.palette_id = 1;
    tc.ప్రారంభం_అడ్డువరుస = 0;
    tc.అడ్డువరుసలు[0] = 8;   tc.colsb[0] = 64;
    tc.అడ్డువరుసలు[1] = 8;   tc.colsb[1] = 64;
    tc.అడ్డువరుసలు[2] = 16;  tc.colsb[2] = 32;
    tc.అడ్డువరుసలు[3] = 16;  tc.colsb[3] = 32;
    tc.అడ్డువరుసలు[4] = 16;  tc.colsb[4] = 64;
    tc.అడ్డువరుసలు[5] = 16;  tc.colsb[5] = 64;
    tc.అడ్డువరుసలు[6] = 16;  tc.colsb[6] = 64;
    tc.అడ్డువరుసలు[7] = 16;  tc.colsb[7] = 64;

    _tile_loadconfig(&tc);
    పూర్తి = నిజం;
}

// we need an extra 16 * 4B (TILE_N * int32_t) for each NB/KB block for compensation.
// See the notes `s8s8 igemm compensation in avx512-vnni` for detail.
మూస <రకం_పేరు TB>
పూర్ణాంకం get_tile_కొలత() {
    పూర్ణాంకం tile_కొలత = TILE_N * పరిమాణం(TB);
    ఒకవేళ (do_compensate<TB>::value) {
        tile_కొలత += TILE_N * పరిమాణం(int32_t);
    }
    ఒకవేళ (std::is_same<TB, ఖండం_q4_K>::value ||
        std::is_same<TB, ఖండం_q5_K>::value) {
        tile_కొలత += TILE_N * 4;
    }
    ఒకవేళ (std::is_same<TB, ఖండం_iq4_xs>::value) {
        tile_కొలత += TILE_N * 2;
    }
    తిరిగి tile_కొలత;
}

మూస <రకం_పేరు TB, పూర్ణాంకం ఖండం_K>
పూర్ణాంకం get_అడ్డువరుస_కొలత(పూర్ణాంకం K) {
    పూర్ణాంకం KB = K / ఖండం_K;
    పూర్ణాంకం అడ్డువరుస_కొలత = KB * పరిమాణం(TB);
    ఒకవేళ (do_compensate<TB>::value) {
        అడ్డువరుస_కొలత += KB * పరిమాణం(int32_t);
    }
    ఒకవేళ (std::is_same<TB, ఖండం_q4_K>::value ||
        std::is_same<TB, ఖండం_q5_K>::value) {
        అడ్డువరుస_కొలత += KB * 4;
    }
    ఒకవేళ (std::is_same<TB, ఖండం_iq4_xs>::value) {
        అడ్డువరుస_కొలత += KB * 2;
    }
    తిరిగి అడ్డువరుస_కొలత;
}

// transpose utils
#నిర్వచించు SHUFFLE_EPI32(a, b, ముసుగు) \
    _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b), ముసుగు))
అంతఃవరుస శూన్య transpose_8x8_32bit(__m256i * v, __m256i * v1) {
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

అంతఃవరుస శూన్య transpose_16x4_32bit(__m512i * r, __m512i * d) {

    స్థిర స్థిరరాశి __m512i index1 = mm512_సమితి_epi32(
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

అంతఃవరుస శూన్య transpose_16x16_32bit(__m512i * v) {
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

శూన్య quantize_అడ్డువరుస_q8_K_vnni(స్థిరరాశి దశాంశం * RESTRICT x, శూన్య * RESTRICT vy, int64_t k) {
    ప్రతిజ్ఞా(k % QK_K == 0);
    స్థిరరాశి పూర్ణాంకం KB = k / QK_K;
    constexpr పూర్ణాంకం kVecs = QK_K / 16;

    ఖండం_q8_K * y = reinterpret_cast<ఖండం_q8_K *>(vy);

    // hold 16 float vecs from x
    __m512  v[kVecs];

    // hold the quants vecs
    __m512i vq[kVecs / 4];

    // hold the packed quants vecs
    __m512i vq_packed[kVecs / 4];

    స్థిరరాశి __m512 sign_బిట్టు = _mm512_set1_ps(-0.f);

    కొరకు (పూర్ణాంకం i = 0; i < KB; ++i) {
        // Compute max(abs(e)) for the block
        __m512 vamax = _mm512_set1_ps(0.f);
        కొరకు (పూర్ణాంకం j = 0; j < kVecs; ++j) {
            v[j] = _mm512_loadu_ps(x); x += 16;
            vamax = _mm512_max_ps(vamax, _mm512_andnot_ps(sign_బిట్టు, v[j]));
        }
        స్థిరరాశి దశాంశం amax = _mm512_reduce_max_ps(vamax);

        // Quantize these floats
        స్థిరరాశి దశాంశం iscale = 127.f / amax;
        y[i].d = GGML_CPU_FP32_గమ్యం_FP16(1 / iscale);
        స్థిరరాశి దశాంశం id = ( amax != 0.0f ) ? iscale : 0.f;
        స్థిరరాశి __m512 vscale = _mm512_set1_ps(id);

        // Apply multiplier and round to nearest integer
        కొరకు (పూర్ణాంకం j = 0; j < kVecs; ++j) {
            v[j] = _mm512_mul_ps(v[j], vscale);
            v[j] = _mm512_roundscale_ps(v[j], (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        }

        // Pack to epi8 vecs
        కొరకు (పూర్ణాంకం j = 0; j < kVecs / 4; ++j) {
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

        స్థిరరాశి __m512i one = _mm512_set1_epi8(1);
        __m512i కూడిక = _mm512_setzero_si512();
        కొరకు (పూర్ణాంకం k = 0; k < 4; ++k) {
            కూడిక = _mm512_dpbusd_epi32(కూడిక, one, vq_packed[k]);
        }
        _mm256_storeu_si256((__m256i *)(y[i].bsums), _mm512_cvtepi32_epi16(కూడిక));
    }
}

// quantize A from float to `vec_dot_type`
మూస <రకం_పేరు T>
అంతఃవరుస శూన్య from_float(స్థిరరాశి దశాంశం * x, అక్షరం * vy, int64_t k);

మూస <>
అంతఃవరుస శూన్య from_float<ఖండం_q8_0>(స్థిరరాశి దశాంశం * x, అక్షరం * vy, int64_t k) {
    quantize_అడ్డువరుస_q8_0(x, (ఖండం_q8_0 *)vy, k);
}

మూస <>
అంతఃవరుస శూన్య from_float<ఖండం_q8_1>(స్థిరరాశి దశాంశం * x, అక్షరం * vy, int64_t k) {
    quantize_అడ్డువరుస_q8_1(x, (ఖండం_q8_1 *)vy, k);
}

మూస <>
అంతఃవరుస శూన్య from_float<ఖండం_q8_K>(స్థిరరాశి దశాంశం * x, అక్షరం * vy, int64_t k) {
#ఒకవేళ 1
    // TODO: this is reference impl!
    quantize_అడ్డువరుస_q8_K_ref(x, (ఖండం_q8_K *)vy, k);
#లేకపోతే
    quantize_అడ్డువరుస_q8_K_vnni(x, vy, k);
#అయితే_ముగింపు
}

// load A from memory to array when nrows can not fill in whole tile
శూన్య unpack_A(int8_t * RESTRICT tile, స్థిరరాశి ఖండం_q8_0 * RESTRICT A, పూర్ణాంకం lda, పూర్ణాంకం nr) {
    ప్రతిజ్ఞా(nr != TILE_M);
    కొరకు (పూర్ణాంకం m = 0; m < nr; ++m) {
        స్థిరరాశి __m256i v = _mm256_loadu_si256((స్థిరరాశి __m256i *)(A[m * lda].qs));
        _mm256_storeu_si256((__m256i *)(tile + m * TILE_K), v);
    }
}

శూన్య unpack_A(int8_t * RESTRICT tile, స్థిరరాశి ఖండం_q8_1 * RESTRICT A, పూర్ణాంకం lda, పూర్ణాంకం nr) {
    ప్రతిజ్ఞా(nr != TILE_M);
    కొరకు (పూర్ణాంకం m = 0; m < nr; ++m) {
        స్థిరరాశి __m256i v = _mm256_loadu_si256((స్థిరరాశి __m256i *)(A[m * lda].qs));
        _mm256_storeu_si256((__m256i *)(tile + m * TILE_K), v);
    }
}

మూస <రకం_పేరు TB>
శూన్య unpack_A(int8_t * RESTRICT tile, స్థిరరాశి ఖండం_q8_K * RESTRICT A, పూర్ణాంకం lda, పూర్ణాంకం k, పూర్ణాంకం nr) {
    ప్రతిజ్ఞా(nr <= TILE_M);
    కొరకు (పూర్ణాంకం m = 0; m < nr; ++m) {
        స్థిరరాశి __m256i v = _mm256_loadu_si256((స్థిరరాశి __m256i *)(A[m * lda].qs + k * 32));
        _mm256_storeu_si256((__m256i *)(tile + m * TILE_K), v);
    }
}

మూస <>
శూన్య unpack_A<ఖండం_q6_K>(int8_t * RESTRICT tile, స్థిరరాశి ఖండం_q8_K * RESTRICT A, పూర్ణాంకం lda, పూర్ణాంకం k, పూర్ణాంకం nr) {
    ప్రతిజ్ఞా(nr <= TILE_M);
    // zero padding k from 16 to 32, so that we don't have to re-config amx
    స్థిరరాశి __m128i zero = _mm_setzero_si128();
    కొరకు (పూర్ణాంకం m = 0; m < nr; ++m) {
        స్థిరరాశి __m128i v = _mm_loadu_si128((స్థిరరాశి __m128i *)(A[m * lda].qs + k * 16));
        స్థిరరాశి __m256i r = _mm256_insertf128_si256(_mm256_castsi128_si256(v), zero, 1);
        _mm256_storeu_si256((__m256i *)(tile + m * TILE_K), r);
    }
}

#నిర్వచించు MM256_సమితి_M128_I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)
అంతఃవరుస __m256i బైట్లు_from_nibbles_32(స్థిరరాశి uint8_t * rsi) {
    స్థిరరాశి __m128i tmp = _mm_loadu_si128((స్థిరరాశి __m128i *)rsi);
    స్థిరరాశి __m256i బైట్లు = MM256_సమితి_M128_I(_mm_srli_epi16(tmp, 4), tmp);
    స్థిరరాశి __m256i దిగువ_ముసుగు = _mm256_set1_epi8(0xF);
    తిరిగి _mm256_and_si256(దిగువ_ముసుగు, బైట్లు);
}

// used for block_q4_K
అంతఃవరుస __m512i బైట్లు_from_nibbles_64(స్థిరరాశి uint8_t * rsi) {
    స్థిరరాశి __m256i tmp = _mm256_loadu_si256((స్థిరరాశి __m256i *)rsi);
    స్థిరరాశి __m256i దిగువ_ముసుగు = _mm256_set1_epi8(0xF);
    స్థిరరాశి __m256i q4l = _mm256_and_si256(tmp, దిగువ_ముసుగు);
    స్థిరరాశి __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(tmp, 4), దిగువ_ముసుగు);
    తిరిగి _mm512_inserti32x8(_mm512_castsi256_si512(q4l), q4h, 1);
}

// used for block_q5_K
అంతఃవరుస __m512i బైట్లు_from_nibbles_64(స్థిరరాశి uint8_t * qs, స్థిరరాశి uint8_t * qh, పూర్ణాంకం k) {
    స్థిరరాశి __m256i దిగువ_ముసుగు = _mm256_set1_epi8(0xF);
    __m256i hmask = _mm256_set1_epi8(1);
    hmask = _mm256_slli_epi16(hmask, k);

    స్థిరరాశి __m256i q5bits = _mm256_loadu_si256((స్థిరరాశి __m256i *)qs);
    స్థిరరాశి __m256i hbits = _mm256_loadu_si256((స్థిరరాశి __m256i *)qh);

    స్థిరరాశి __m256i q5l_0 = _mm256_and_si256(q5bits, దిగువ_ముసుగు);
    స్థిరరాశి __m256i q5h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), k + 0), 4);
    స్థిరరాశి __m256i q5_0  = mm256_కూడు_epi8(q5l_0, q5h_0);
    hmask = _mm256_slli_epi16(hmask, 1);

    స్థిరరాశి __m256i q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), దిగువ_ముసుగు);
    స్థిరరాశి __m256i q5h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), k + 1), 4);
    స్థిరరాశి __m256i q5_1  = mm256_కూడు_epi8(q5l_1, q5h_1);

    తిరిగి _mm512_inserti32x8(_mm512_castsi256_si512(q5_0), q5_1, 1);
}

// used for block_q6_K
అంతఃవరుస శూన్య బైట్లు_from_nibbles_128(__m512i& r0, __m512i& r1, స్థిరరాశి uint8_t * qs, స్థిరరాశి uint8_t * qh) {
    స్థిరరాశి __m256i m4 = _mm256_set1_epi8(0xF);
    స్థిరరాశి __m256i m2 = _mm256_set1_epi8(0x3);

    స్థిరరాశి __m256i q6bits1 = _mm256_loadu_si256((స్థిరరాశి __m256i *)qs);
    స్థిరరాశి __m256i q6bits2 = _mm256_loadu_si256((స్థిరరాశి __m256i *)(qs + 32));
    స్థిరరాశి __m256i q6bitsH = _mm256_loadu_si256((స్థిరరాశి __m256i *)qh);

    స్థిరరాశి __m256i q6h_0 = _mm256_slli_epi16(_mm256_and_si256(                  q6bitsH,     m2), 4);
    స్థిరరాశి __m256i q6h_1 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q6bitsH, 2), m2), 4);
    స్థిరరాశి __m256i q6h_2 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q6bitsH, 4), m2), 4);
    స్థిరరాశి __m256i q6h_3 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q6bitsH, 6), m2), 4);

    స్థిరరాశి __m256i q6_0 = _mm256_or_si256(_mm256_and_si256(q6bits1, m4), q6h_0);
    స్థిరరాశి __m256i q6_1 = _mm256_or_si256(_mm256_and_si256(q6bits2, m4), q6h_1);
    స్థిరరాశి __m256i q6_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q6bits1, 4), m4), q6h_2);
    స్థిరరాశి __m256i q6_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q6bits2, 4), m4), q6h_3);

    r0 = _mm512_inserti32x8(_mm512_castsi256_si512(q6_0), q6_1, 1);
    r1 = _mm512_inserti32x8(_mm512_castsi256_si512(q6_2), q6_3, 1);
}

అంతఃవరుస __m512i packNibbles(__m512i r0, __m512i r1) {
    తిరిగి _mm512_or_si512(r0, _mm512_slli_epi16(r1, 4));
}

మూస <రకం_పేరు TB>
అంతఃవరుస శూన్య pack_qs(శూన్య * RESTRICT packed_B, స్థిరరాశి TB * RESTRICT B, పూర్ణాంకం KB) {
    int8_t tmp[8 * 64];
    __m256i v[8], v2[8];
    కొరకు (పూర్ణాంకం n = 0; n < 8; ++n) {
        v[n] = బైట్లు_from_nibbles_32(B[n * KB].qs);
    }
    transpose_8x8_32bit(v, v2);
    కొరకు (పూర్ణాంకం n = 0; n < 8; ++n) {
        _mm256_storeu_si256((__m256i *)(tmp + n * 64), v2[n]);
    }
    కొరకు (పూర్ణాంకం n = 0; n < 8; ++n) {
        v[n] = బైట్లు_from_nibbles_32(B[(n + 8) * KB].qs);
    }
    transpose_8x8_32bit(v, v2);
    కొరకు (పూర్ణాంకం n = 0; n < 8; ++n) {
        _mm256_storeu_si256((__m256i *)(tmp + n * 64 + 32), v2[n]);
    }

    // pack again with 128 to fully utilize vector length
    కొరకు (పూర్ణాంకం n = 0; n < 8; n += 2) {
        __m512i r0 = _mm512_loadu_si512((స్థిరరాశి __m512i *)(tmp + n * 64));
        __m512i r1 = _mm512_loadu_si512((స్థిరరాశి __m512i *)(tmp + n * 64 + 64));
        __m512i r1r0 = packNibbles(r0, r1);
        _mm512_storeu_si512((__m512i *)((అక్షరం *)packed_B + n * 32), r1r0);
    }
}

మూస <>
అంతఃవరుస శూన్య pack_qs<ఖండం_q8_0>(శూన్య * RESTRICT packed_B, స్థిరరాశి ఖండం_q8_0 * RESTRICT B, పూర్ణాంకం KB) {
    __m256i v[8], v2[8];
    కొరకు (పూర్ణాంకం n = 0; n < 8; ++n) {
        v[n] = _mm256_loadu_si256((స్థిరరాశి __m256i *)(B[n * KB].qs));
    }
    transpose_8x8_32bit(v, v2);
    కొరకు (పూర్ణాంకం n = 0; n < 8; ++n) {
        _mm256_storeu_si256((__m256i *)((అక్షరం *)packed_B + n * 64), v2[n]);
    }
    కొరకు (పూర్ణాంకం n = 0; n < 8; ++n) {
        v[n] = _mm256_loadu_si256((స్థిరరాశి __m256i *)(B[(n + 8) * KB].qs));
    }
    transpose_8x8_32bit(v, v2);
    కొరకు (పూర్ణాంకం n = 0; n < 8; ++n) {
        _mm256_storeu_si256((__m256i *)((అక్షరం *)packed_B + n * 64 + 32), v2[n]);
    }
}

మూస <>
అంతఃవరుస శూన్య pack_qs<ఖండం_q4_K>(శూన్య * RESTRICT packed_B, స్థిరరాశి ఖండం_q4_K * RESTRICT B, పూర్ణాంకం KB) {
    __m512i v[16];
    // QK_K 256 with 8 groups, handle 2 groups at a time
    అక్షరం * pb = (అక్షరం *)packed_B;
    కొరకు (పూర్ణాంకం k = 0; k < QK_K / 64; ++k) {
        // pack 2 groups { n, g,  k} to {g, k/4, 4n}
        //          e.g. {16, 2, 32} to {2,   8, 64}
        కొరకు (పూర్ణాంకం n = 0; n < TILE_N; ++n) {
            v[n] = బైట్లు_from_nibbles_64(B[n * KB].qs + k * 32);
        }

        transpose_16x16_32bit(v);

        // pack again with 128 to fully utilize vector length
        కొరకు (పూర్ణాంకం n = 0; n < TILE_N; n += 2) {
            _mm512_storeu_si512((__m512i *)pb, packNibbles(v[n], v[n + 1]));
            pb += 64;
        }
    }
}

మూస <>
అంతఃవరుస శూన్య pack_qs<ఖండం_q5_K>(శూన్య * RESTRICT packed_B, స్థిరరాశి ఖండం_q5_K * RESTRICT B, పూర్ణాంకం KB) {
    __m512i v[16];
    స్థిరరాశి __m512i దిగువ_ముసుగు = _mm512_set1_epi8(0xF);
    // QK_K 256 with 8 groups, handle 2 groups at a time
    అక్షరం * pb = (అక్షరం *)packed_B;
    అక్షరం * ph = (అక్షరం *)packed_B + (QK_K / 2) * TILE_N;
    కొరకు (పూర్ణాంకం k = 0; k < QK_K / 64; ++k) {
        // pack 2 groups { n, g,  k} to {g, k/4, 4n}
        //          e.g. {16, 2, 32} to {2,   8, 64}
        కొరకు (పూర్ణాంకం n = 0; n < TILE_N; ++n) {
            v[n] = బైట్లు_from_nibbles_64(B[n * KB].qs + k * 32, B[n * KB].qh, /*సమూహం*/2 * k);
        }

        transpose_16x16_32bit(v);

        // 1. pack lower 4bits with 2 groups
        కొరకు (పూర్ణాంకం n = 0; n < TILE_N; n += 2) {
            // get lower 4 bits
            స్థిరరాశి __m512i r0 = _mm512_and_si512(v[n], దిగువ_ముసుగు);
            స్థిరరాశి __m512i r1 = _mm512_and_si512(v[n + 1], దిగువ_ముసుగు);
            _mm512_storeu_si512((__m512i *)pb, packNibbles(r0, r1)); pb += 64;
        }

        // 2. pack higher 1bit with 2 groups
        స్థిరరాశి __m512i hmask = _mm512_set1_epi8(0x10);
        కొరకు (పూర్ణాంకం g = 0; g < 2; ++g) {
            __m512i hbits = _mm512_setzero_si512();
            hbits = mm512_కూడు_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 8 + 0], hmask), 4));
            hbits = mm512_కూడు_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 8 + 1], hmask), 3));
            hbits = mm512_కూడు_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 8 + 2], hmask), 2));
            hbits = mm512_కూడు_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 8 + 3], hmask), 1));
            hbits = mm512_కూడు_epi8(hbits,                   _mm512_and_si512(v[g * 8 + 4], hmask)    );
            hbits = mm512_కూడు_epi8(hbits, _mm512_slli_epi16(_mm512_and_si512(v[g * 8 + 5], hmask), 1));
            hbits = mm512_కూడు_epi8(hbits, _mm512_slli_epi16(_mm512_and_si512(v[g * 8 + 6], hmask), 2));
            hbits = mm512_కూడు_epi8(hbits, _mm512_slli_epi16(_mm512_and_si512(v[g * 8 + 7], hmask), 3));
            _mm512_storeu_si512((__m512i *)ph, hbits); ph += 64;
        }
    }
}

మూస <>
అంతఃవరుస శూన్య pack_qs<ఖండం_q6_K>(శూన్య * RESTRICT packed_B, స్థిరరాశి ఖండం_q6_K * RESTRICT B, పూర్ణాంకం KB) {
    __m512i v[32];
    స్థిరరాశి __m512i దిగువ_ముసుగు = _mm512_set1_epi8(0xF);
    // QK_K 256 with 8 groups, handle 4 groups at a time
    అక్షరం * pb = (అక్షరం *)packed_B;
    అక్షరం * ph = (అక్షరం *)packed_B + (QK_K / 2) * TILE_N;
    కొరకు (పూర్ణాంకం k = 0; k < QK_K / 128; ++k) {
        కొరకు (పూర్ణాంకం n = 0; n < TILE_N; ++n) {
            బైట్లు_from_nibbles_128(v[n], v[n + 16], B[n * KB].ql + k * 64, B[n * KB].qh + k * 32);
        }

        // top half: group 0,1 or 4,5; bottom half: group 2,3 or 6,7
        transpose_16x16_32bit(v);
        transpose_16x16_32bit(v + 16);

        // 1. pack lower 4bits with 4 groups
        కొరకు (పూర్ణాంకం n = 0; n < 32; n += 2) {
            స్థిరరాశి __m512i r0 = _mm512_and_si512(v[n], దిగువ_ముసుగు);
            స్థిరరాశి __m512i r1 = _mm512_and_si512(v[n + 1], దిగువ_ముసుగు);
            _mm512_storeu_si512((__m512i *)pb, packNibbles(r0, r1)); pb += 64;
        }

        // 2. pack higher 2bit with 4 groups
        స్థిరరాశి __m512i hmask = _mm512_set1_epi8(0x30);
        కొరకు (పూర్ణాంకం g = 0; g < 8; ++g) {
            __m512i hbits = _mm512_setzero_si512();
            hbits = mm512_కూడు_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 4 + 0], hmask), 4));
            hbits = mm512_కూడు_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 4 + 1], hmask), 2));
            hbits = mm512_కూడు_epi8(hbits,                   _mm512_and_si512(v[g * 4 + 2], hmask)    );
            hbits = mm512_కూడు_epi8(hbits, _mm512_slli_epi16(_mm512_and_si512(v[g * 4 + 3], hmask), 2));
            _mm512_storeu_si512((__m512i *)ph, hbits); ph += 64;
        }
    }
}

మూస <>
అంతఃవరుస శూన్య pack_qs<ఖండం_iq4_xs>(శూన్య * RESTRICT packed_B, స్థిరరాశి ఖండం_iq4_xs * RESTRICT B, పూర్ణాంకం KB) {
    __m512i v[16];
    అక్షరం * pb = (అక్షరం *)packed_B;
    కొరకు (పూర్ణాంకం k = 0; k < QK_K / 64; ++k) {
        కొరకు (పూర్ణాంకం n = 0; n < TILE_N; ++n) {
            __m256i r0 = బైట్లు_from_nibbles_32(B[n * KB].qs + k * 32 +  0);
            __m256i r1 = బైట్లు_from_nibbles_32(B[n * KB].qs + k * 32 + 16);
            v[n] = _mm512_inserti32x8(_mm512_castsi256_si512(r0), r1, 1);
        }

        transpose_16x16_32bit(v);

        // pack again with 128 to fully utilize vector length
        కొరకు (పూర్ణాంకం n = 0; n < TILE_N; n += 2) {
            _mm512_storeu_si512((__m512i *)pb, packNibbles(v[n], v[n + 1]));
            pb += 64;
        }
    }
}

// pack B to vnni formats in 4bits or 8 bits
శూన్య pack_B(శూన్య * RESTRICT packed_B, స్థిరరాశి ఖండం_q4_0 * RESTRICT B, పూర్ణాంకం KB) {
    pack_qs(packed_B, B, KB);
    ggml_half * d0 = reinterpret_cast<ggml_half *>((అక్షరం *)packed_B + TILE_N * TILE_K / 2);
    కొరకు (పూర్ణాంకం n = 0; n < TILE_N; ++n) {
        d0[n] = B[n * KB].d;
    }
}

శూన్య pack_B(శూన్య * RESTRICT packed_B, స్థిరరాశి ఖండం_q4_1 * RESTRICT B, పూర్ణాంకం KB) {
    pack_qs(packed_B, B, KB);
    ggml_half * d0 = reinterpret_cast<ggml_half *>((అక్షరం *)packed_B + TILE_N * TILE_K / 2);
    ggml_half * m0 = d0 + TILE_N;
    కొరకు (పూర్ణాంకం n = 0; n < TILE_N; ++n) {
        d0[n] = B[n * KB].d;
        m0[n] = B[n * KB].m;
    }
}

అంతఃవరుస శూన్య s8s8_compensation(శూన్య * RESTRICT packed_B) {
    // packed_B layout:
    //   quants {TILE_N, TILEK}  int8_t
    //   d0     {TILE_N}      ggml_half
    //   comp   {TILE_N}        int32_t
    స్థిరరాశి పూర్ణాంకం స్థానభ్రంశం = TILE_N * TILE_K + TILE_N * పరిమాణం(ggml_half);
    __m512i vcomp = _mm512_setzero_si512();
    స్థిరరాశి __m512i off = _mm512_set1_epi8(static_cast<అక్షరం>(0x80));
    కొరకు (పూర్ణాంకం k = 0; k < 8; ++k) {
        __m512i vb = _mm512_loadu_si512((స్థిరరాశి __m512i *)((స్థిరరాశి అక్షరం *)packed_B + k * 64));
        vcomp = _mm512_dpbusd_epi32(vcomp, off, vb);
    }
    _mm512_storeu_si512((__m512i *)((అక్షరం *)(packed_B) + స్థానభ్రంశం), vcomp);
}

శూన్య pack_B(శూన్య * RESTRICT packed_B, స్థిరరాశి ఖండం_q8_0 * RESTRICT B, పూర్ణాంకం KB) {
    pack_qs(packed_B, B, KB);
    ggml_half * d0 = reinterpret_cast<ggml_half *>((అక్షరం *)packed_B + TILE_N * TILE_K);
    కొరకు (పూర్ణాంకం n = 0; n < TILE_N; ++n) {
        d0[n] = B[n * KB].d;
    }
    s8s8_compensation(packed_B);
}

// convert 8 * {min, scale} from int6 to int8
అంతఃవరుస శూన్య unpack_mins_and_scales(స్థిరరాశి uint8_t * scales, uint32_t * utmp) {
    స్థిరరాశి uint32_t kmask1 = 0x3f3f3f3f;
    స్థిరరాశి uint32_t kmask2 = 0x0f0f0f0f;
    స్థిరరాశి uint32_t kmask3 = 0x03030303;

    జ్ఞాపకం_నకలు(utmp, scales, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    స్థిరరాశి uint32_t uaux = utmp[1] & kmask1;
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
శూన్య pack_B(శూన్య * RESTRICT packed_B, స్థిరరాశి ఖండం_q4_K * RESTRICT B, పూర్ణాంకం KB) {
    pack_qs(packed_B, B, KB);

    uint8_t * scales = reinterpret_cast<uint8_t *>((అక్షరం *)packed_B + (QK_K / 2) * TILE_N);
    uint8_t * mins = scales + 8 * TILE_N;
    ggml_half * d = reinterpret_cast<ggml_half *>(mins + 8 * TILE_N);
    ggml_half * dmin = d + TILE_N;

    సంఘం {
        uint32_t u32[4];
        uint8_t  u8[16];
    } s;

    కొరకు (పూర్ణాంకం n = 0; n < TILE_N; ++n) {
        unpack_mins_and_scales(B[n * KB].scales, s.u32);
        కొరకు (పూర్ణాంకం k = 0; k < 8; ++k) {
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
శూన్య pack_B(శూన్య * RESTRICT packed_B, స్థిరరాశి ఖండం_q5_K * RESTRICT B, పూర్ణాంకం KB) {
    pack_qs(packed_B, B, KB);

    uint8_t * scales = reinterpret_cast<uint8_t *>((అక్షరం *)packed_B + (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N);
    uint8_t * mins = scales + 8 * TILE_N;
    ggml_half * d = reinterpret_cast<ggml_half *>(mins + 8 * TILE_N);
    ggml_half * dmin = d + TILE_N;

    సంఘం {
        uint32_t u32[4];
        uint8_t  u8[16];
    } s;

    కొరకు (పూర్ణాంకం n = 0; n < TILE_N; ++n) {
        unpack_mins_and_scales(B[n * KB].scales, s.u32);
        కొరకు (పూర్ణాంకం k = 0; k < 8; ++k) {
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
శూన్య pack_B(శూన్య * RESTRICT packed_B, స్థిరరాశి ఖండం_q6_K * RESTRICT B, పూర్ణాంకం KB) {
    pack_qs(packed_B, B, KB);

    uint8_t * scales = reinterpret_cast<uint8_t *>((అక్షరం *)packed_B + (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N);
    ggml_half * d = reinterpret_cast<ggml_half *>(scales + 16 * TILE_N);
    కొరకు (పూర్ణాంకం n = 0; n < TILE_N; ++n) {
        స్థిరరాశి int8_t * ps = B[n * KB].scales;
        కొరకు (పూర్ణాంకం k = 0; k < 16; ++k) {
            scales[k * TILE_N + n] = ps[k];
        }
        d[n] = B[n * KB].d;
    }
}

// packed_B layout:
//   quants {8, TILE_N, 16}  uint8
//   scales {8, TILE_N}       int8
//   d      {TILE_N}     ggml_half
శూన్య pack_B(శూన్య * RESTRICT packed_B, స్థిరరాశి ఖండం_iq4_xs * RESTRICT B, పూర్ణాంకం KB) {
    pack_qs(packed_B, B, KB);

    int8_t * scales = reinterpret_cast<int8_t *>((అక్షరం *)packed_B + (QK_K / 2) * TILE_N);
    ggml_half * d = reinterpret_cast<ggml_half *>(scales + 8 * TILE_N);

    // pack the scales
    కొరకు (పూర్ణాంకం n = 0; n < TILE_N; ++n) {
        uint16_t sh = B[n * KB].scales_h;
        కొరకు (పూర్ణాంకం k = 0; k < 8; k += 2) {
            స్థిరరాశి int16_t ls1 = ((B[n * KB].scales_l[k / 2] & 0xf) | ((sh << 4) & 0x30)) - 32;
            స్థిరరాశి int16_t ls2 = ((B[n * KB].scales_l[k / 2] >>  4) | ((sh << 2) & 0x30)) - 32;
            scales[(k + 0) * TILE_N + n] = ls1;
            scales[(k + 1) * TILE_N + n] = ls2;
            sh >>= 4;
        }
        d[n] = B[n * KB].d;
    }
}

మూస<రకం_పేరు TB, రకం_పేరు packed_B_t = packed_B_type<TB>>
శూన్య unpack_B(packed_B_t * RESTRICT tile, స్థిరరాశి శూన్య * RESTRICT packed_B) {
    GGML_వాడని(tile);
    GGML_వాడని(packed_B);
}

మూస <>
శూన్య unpack_B<ఖండం_q4_0>(int8_t * RESTRICT tile, స్థిరరాశి శూన్య * RESTRICT packed_B) {
  స్థిరరాశి __m512i off = _mm512_set1_epi8(8);
  స్థిరరాశి __m512i దిగువ_ముసుగు = _mm512_set1_epi8(0xF);
  కొరకు (పూర్ణాంకం n = 0; n < 8; n += 2) {
    __m512i బైట్లు = _mm512_loadu_si512((స్థిరరాశి __m512i *)((స్థిరరాశి అక్షరం *)packed_B + n * 32));
    స్థిరరాశి __m512i r0 = _mm512_sub_epi8(_mm512_and_si512(బైట్లు, దిగువ_ముసుగు), off);
    స్థిరరాశి __m512i r1 = _mm512_sub_epi8(_mm512_and_si512(_mm512_srli_epi16(బైట్లు, 4), దిగువ_ముసుగు), off);
    _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
    _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
  }
}

మూస <>
శూన్య unpack_B<ఖండం_q4_1>(uint8_t * RESTRICT tile, స్థిరరాశి శూన్య * RESTRICT packed_B) {
    స్థిరరాశి __m512i దిగువ_ముసుగు = _mm512_set1_epi8(0xF);
    కొరకు (పూర్ణాంకం n = 0; n < 8; n += 2) {
        __m512i బైట్లు = _mm512_loadu_si512((స్థిరరాశి __m512i *)((స్థిరరాశి అక్షరం *)packed_B + n * 32));
        స్థిరరాశి __m512i r0 = _mm512_and_si512(బైట్లు, దిగువ_ముసుగు);
        స్థిరరాశి __m512i r1 = _mm512_and_si512(_mm512_srli_epi16(బైట్లు, 4), దిగువ_ముసుగు);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
    }
}

// packed_B_t for QKK is int8_t
మూస <రకం_పేరు TB>
శూన్య unpack_B(int8_t * RESTRICT tile, స్థిరరాశి శూన్య * RESTRICT packed_B, పూర్ణాంకం k) {
    స్థిరరాశి పూర్ణాంకం packed_B_సమూహం_కొలత = QK_K / 2 * TILE_N / 8;
    స్థిరరాశి అక్షరం * packed_B_సమూహం = (స్థిరరాశి అక్షరం *)packed_B + k * packed_B_సమూహం_కొలత;
    స్థిరరాశి __m512i దిగువ_ముసుగు = _mm512_set1_epi8(0xF);
    కొరకు (పూర్ణాంకం n = 0; n < 8; n += 2) {
        __m512i బైట్లు = _mm512_loadu_si512(packed_B_సమూహం + n * 32);
        స్థిరరాశి __m512i r0 = _mm512_and_si512(బైట్లు, దిగువ_ముసుగు);
        స్థిరరాశి __m512i r1 = _mm512_and_si512(_mm512_srli_epi16(బైట్లు, 4), దిగువ_ముసుగు);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
    }
}

మూస <>
శూన్య unpack_B<ఖండం_q5_K>(int8_t * RESTRICT tile, స్థిరరాశి శూన్య * RESTRICT packed_B, పూర్ణాంకం k) {
    // lower 4bits, stride 256 bytes
    స్థిరరాశి పూర్ణాంకం packed_l4_సమూహం_కొలత = QK_K / 2 * TILE_N / 8;
    స్థిరరాశి అక్షరం * pb = (స్థిరరాశి అక్షరం *)packed_B + k * packed_l4_సమూహం_కొలత;

    // higher 1bit, stride 64 bytes
    స్థిరరాశి పూర్ణాంకం packed_h1_సమూహం_కొలత = QK_K / 8 * TILE_N / 8;
    స్థిరరాశి అక్షరం * ph = (స్థిరరాశి అక్షరం *)packed_B + (QK_K / 2) * TILE_N + k * packed_h1_సమూహం_కొలత;
    స్థిరరాశి __m512i hbits = _mm512_loadu_si512(ph);

    స్థిరరాశి __m512i దిగువ_ముసుగు = _mm512_set1_epi8(0xF);
    __m512i hmask0 = _mm512_set1_epi8(0x1);
    __m512i hmask1 = _mm512_set1_epi8(0x2);

    కొరకు (పూర్ణాంకం n = 0; n < 8; n += 2) {
        __m512i బైట్లు = _mm512_loadu_si512(pb + n * 32);
        __m512i r0 = _mm512_and_si512(బైట్లు, దిగువ_ముసుగు);
        __m512i r1 = _mm512_and_si512(_mm512_srli_epi16(బైట్లు, 4), దిగువ_ముసుగు);
        __m512i h0 = _mm512_slli_epi16(_mm512_srli_epi16(_mm512_and_si512(hbits, hmask0), n), 4);
        __m512i h1 = _mm512_slli_epi16(_mm512_srli_epi16(_mm512_and_si512(hbits, hmask1), n + 1), 4);

        hmask0 = _mm512_slli_epi16(hmask0, 2);
        hmask1 = _mm512_slli_epi16(hmask1, 2);
        r0 = mm512_కూడు_epi8(r0, h0);
        r1 = mm512_కూడు_epi8(r1, h1);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
    }
}

మూస <>
శూన్య unpack_B<ఖండం_q6_K>(int8_t * RESTRICT tile, స్థిరరాశి శూన్య * RESTRICT packed_B, పూర్ణాంకం k) {
    // lower 4bits, stride 128 bytes
    స్థిరరాశి పూర్ణాంకం packed_l4_సమూహం_కొలత = QK_K / 2 * TILE_N / 16;
    స్థిరరాశి అక్షరం * pb = (స్థిరరాశి అక్షరం *)packed_B + k * packed_l4_సమూహం_కొలత;

    // higher 2bits, stride 64 bytes
    స్థిరరాశి పూర్ణాంకం packed_h2_సమూహం_కొలత = QK_K / 4 * TILE_N / 16;
    స్థిరరాశి అక్షరం * ph = (స్థిరరాశి అక్షరం *)packed_B + (QK_K / 2) * TILE_N + k * packed_h2_సమూహం_కొలత;
    స్థిరరాశి __m512i hbits = _mm512_loadu_si512(ph);

    స్థిరరాశి __m512i off = _mm512_set1_epi8(32);
    స్థిరరాశి __m512i దిగువ_ముసుగు = _mm512_set1_epi8(0xF);
    __m512i hmask0 = _mm512_set1_epi8(0x3); // 0011
    __m512i hmask1 = _mm512_set1_epi8(0xC); // 1100

    // notes: skip zero padding from row4 to row7 as we have done so in `unpack_A`
    __m512i బైట్లు = _mm512_loadu_si512(pb);
    __m512i r0 = _mm512_and_si512(బైట్లు, దిగువ_ముసుగు);
    __m512i r1 = _mm512_and_si512(_mm512_srli_epi16(బైట్లు, 4), దిగువ_ముసుగు);
    __m512i h0 = _mm512_slli_epi16(_mm512_and_si512(hbits, hmask0), 4);
    __m512i h1 = _mm512_slli_epi16(_mm512_and_si512(hbits, hmask1), 2);
    _mm512_storeu_si512((__m512i *)(tile +  0), _mm512_sub_epi8(mm512_కూడు_epi8(r0, h0), off));
    _mm512_storeu_si512((__m512i *)(tile + 64), _mm512_sub_epi8(mm512_కూడు_epi8(r1, h1), off));

    hmask0 = _mm512_slli_epi16(hmask0, 4);
    hmask1 = _mm512_slli_epi16(hmask1, 4);

    బైట్లు = _mm512_loadu_si512(pb + 64);
    r0 = _mm512_and_si512(బైట్లు, దిగువ_ముసుగు);
    r1 = _mm512_and_si512(_mm512_srli_epi16(బైట్లు, 4), దిగువ_ముసుగు);
    h0 =                   _mm512_and_si512(hbits, hmask0);
    h1 = _mm512_srli_epi16(_mm512_and_si512(hbits, hmask1), 2);
    _mm512_storeu_si512((__m512i *)(tile + 128), _mm512_sub_epi8(mm512_కూడు_epi8(r0, h0), off));
    _mm512_storeu_si512((__m512i *)(tile + 192), _mm512_sub_epi8(mm512_కూడు_epi8(r1, h1), off));
}

మూస <>
శూన్య unpack_B<ఖండం_iq4_xs>(int8_t * RESTRICT tile, స్థిరరాశి శూన్య * RESTRICT packed_B, పూర్ణాంకం k) {
    స్థిర స్థిరరాశి __m512i values128 = mm512_సమితి_epi8(
        113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
        113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
        113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
        113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127
    );

    స్థిరరాశి పూర్ణాంకం packed_B_సమూహం_కొలత = QK_K / 2 * TILE_N / 8;
    స్థిరరాశి అక్షరం * pb = (స్థిరరాశి అక్షరం *)packed_B + k * packed_B_సమూహం_కొలత;
    స్థిరరాశి __m512i దిగువ_ముసుగు = _mm512_set1_epi8(0xF);

    కొరకు (పూర్ణాంకం n = 0; n < 8; n += 2) {
        __m512i బైట్లు = _mm512_loadu_si512(pb + n * 32);
        స్థిరరాశి __m512i r0 = _mm512_shuffle_epi8(values128, _mm512_and_si512(బైట్లు, దిగువ_ముసుగు));
        స్థిరరాశి __m512i r1 = _mm512_shuffle_epi8(values128, _mm512_and_si512(_mm512_srli_epi16(బైట్లు, 4), దిగువ_ముసుగు));
        _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
    }
}

మూస <రకం_పేరు TA, రకం_పేరు TB, తార్కిక is_acc>
నిర్మాణం acc_C {};

మూస <తార్కిక is_acc>
నిర్మాణం acc_C<ఖండం_q8_0, ఖండం_q4_0, is_acc> {
    స్థిర శూన్య apply(దశాంశం * RESTRICT C, పూర్ణాంకం ldc, స్థిరరాశి int32_t * RESTRICT tile, స్థిరరాశి ఖండం_q8_0 * A, పూర్ణాంకం lda, స్థిరరాశి శూన్య * packed_B, పూర్ణాంకం nr) {
        స్థిరరాశి పూర్ణాంకం స్థానభ్రంశం = TILE_N * TILE_K / 2;
        స్థిరరాశి __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((స్థిరరాశి __m256i *)((స్థిరరాశి అక్షరం *)packed_B + స్థానభ్రంశం)));

        కొరకు (పూర్ణాంకం m = 0; m < nr; ++m) {
            స్థిరరాశి __m512 vd1 = _mm512_set1_ps(GGML_CPU_FP16_గమ్యం_FP32(A[m * lda].d));
            స్థిరరాశి __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            ఒకవేళ (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } లేకపోతే {
                vsum = _mm512_set1_ps(0.f);
            }
            vsum = _mm512_fmadd_ps(vtile, _mm512_mul_ps(vd0, vd1), vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

మూస <తార్కిక is_acc>
నిర్మాణం acc_C<ఖండం_q8_1, ఖండం_q4_1, is_acc> {
    స్థిర శూన్య apply(దశాంశం * RESTRICT C, పూర్ణాంకం ldc, స్థిరరాశి int32_t * RESTRICT tile, స్థిరరాశి ఖండం_q8_1 * A, పూర్ణాంకం lda, స్థిరరాశి శూన్య * packed_B, పూర్ణాంకం nr) {
        స్థిరరాశి పూర్ణాంకం స్థానభ్రంశం = TILE_N * TILE_K / 2;
        స్థిరరాశి __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((స్థిరరాశి __m256i *)((స్థిరరాశి అక్షరం *)packed_B + స్థానభ్రంశం)));
        స్థిరరాశి __m512 vm0 = _mm512_cvtph_ps(_mm256_loadu_si256((స్థిరరాశి __m256i *)((స్థిరరాశి అక్షరం *)packed_B + స్థానభ్రంశం + TILE_N * పరిమాణం(ggml_half))));

        కొరకు (పూర్ణాంకం m = 0; m < nr; ++m) {
            స్థిరరాశి __m512 vd1 = _mm512_set1_ps(GGML_CPU_FP16_గమ్యం_FP32(A[m * lda].d));
            స్థిరరాశి __m512 vs1 = _mm512_set1_ps(GGML_CPU_FP16_గమ్యం_FP32(A[m * lda].s));
            స్థిరరాశి __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            ఒకవేళ (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } లేకపోతే {
                vsum = _mm512_set1_ps(0.f);
            }
            vsum = _mm512_fmadd_ps(vtile, _mm512_mul_ps(vd0, vd1), vsum);
            vsum = _mm512_fmadd_ps(vm0, vs1, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

మూస <తార్కిక is_acc>
నిర్మాణం acc_C<ఖండం_q8_0, ఖండం_q8_0, is_acc> {
    స్థిర శూన్య apply(దశాంశం * RESTRICT C, పూర్ణాంకం ldc, స్థిరరాశి int32_t * RESTRICT tile, స్థిరరాశి ఖండం_q8_0 * A, పూర్ణాంకం lda, స్థిరరాశి శూన్య * packed_B, పూర్ణాంకం nr) {
        స్థిరరాశి పూర్ణాంకం స్థానభ్రంశం = TILE_N * TILE_K;
        స్థిరరాశి __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((స్థిరరాశి __m256i *)((స్థిరరాశి అక్షరం *)packed_B + స్థానభ్రంశం)));

        కొరకు (పూర్ణాంకం m = 0; m < nr; ++m) {
            స్థిరరాశి __m512 vd1 = _mm512_set1_ps(GGML_CPU_FP16_గమ్యం_FP32(A[m * lda].d));
            స్థిరరాశి __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            ఒకవేళ (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } లేకపోతే {
                vsum = _mm512_set1_ps(0.f);
            }
            vsum = _mm512_fmadd_ps(vtile, _mm512_mul_ps(vd0, vd1), vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

మూస <తార్కిక is_acc>
నిర్మాణం acc_C<ఖండం_q8_K, ఖండం_q4_K, is_acc> {
    స్థిర శూన్య apply(దశాంశం * RESTRICT C, పూర్ణాంకం ldc, స్థిరరాశి int32_t * RESTRICT tile, స్థిరరాశి ఖండం_q8_K * A, పూర్ణాంకం lda, స్థిరరాశి శూన్య * packed_B, పూర్ణాంకం nr) {
        స్థిరరాశి uint8_t * scales = reinterpret_cast<స్థిరరాశి uint8_t *>((స్థిరరాశి అక్షరం *)packed_B + (QK_K / 2) * TILE_N);
        స్థిరరాశి uint8_t * mins = scales + 8 * TILE_N;
        స్థిరరాశి ggml_half * d0 = reinterpret_cast<స్థిరరాశి ggml_half *>(mins + 8 * TILE_N);
        స్థిరరాశి ggml_half * dmin = d0 + TILE_N;

        స్థిరరాశి __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((స్థిరరాశి __m256i *)d0));
        స్థిరరాశి __m512 vdmin = _mm512_cvtph_ps(_mm256_loadu_si256((స్థిరరాశి __m256i *)dmin));

        కొరకు (పూర్ణాంకం m = 0; m < nr; ++m) {
            స్థిరరాశి దశాంశం d1 = A[m * lda].d;
            స్థిరరాశి __m512 vd = _mm512_mul_ps(_mm512_set1_ps(d1), vd0);
            స్థిరరాశి __m512 vdm = _mm512_mul_ps(_mm512_set1_ps(-d1), vdmin);
            స్థిరరాశి __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            ఒకవేళ (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } లేకపోతే {
                vsum = _mm512_set1_ps(0.f);
            }

            స్థిరరాశి __m256i q8sums = _mm256_loadu_si256((స్థిరరాశి __m256i *)A[m * lda].bsums);
            స్థిరరాశి __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));

            __m512i acc_m = _mm512_setzero_si512();
            కొరకు (పూర్ణాంకం k = 0; k < 4; ++k) {
                __m512i vmask = _mm512_set1_epi32(k);
                __m512i va = _mm512_permutexvar_epi32(vmask, _mm512_castsi128_si512(q8s));
                __m512i vb = _mm512_cvtepi8_epi16(_mm256_loadu_si256((స్థిరరాశి __m256i *)(mins + k * 32)));
                acc_m = _mm512_dpwssds_epi32(acc_m, va, vb);
            }

            vsum = _mm512_fmadd_ps(vtile, vd, vsum);
            vsum = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc_m), vdm, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

మూస <తార్కిక is_acc>
నిర్మాణం acc_C<ఖండం_q8_K, ఖండం_q5_K, is_acc> {
    స్థిర శూన్య apply(దశాంశం * RESTRICT C, పూర్ణాంకం ldc, స్థిరరాశి int32_t * RESTRICT tile, స్థిరరాశి ఖండం_q8_K * A, పూర్ణాంకం lda, స్థిరరాశి శూన్య * packed_B, పూర్ణాంకం nr) {
        స్థిరరాశి uint8_t * scales = reinterpret_cast<స్థిరరాశి uint8_t *>((స్థిరరాశి అక్షరం *)packed_B + (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N);
        స్థిరరాశి uint8_t * mins = scales + 8 * TILE_N;
        స్థిరరాశి ggml_half * d0 = reinterpret_cast<స్థిరరాశి ggml_half *>(mins + 8 * TILE_N);
        స్థిరరాశి ggml_half * dmin = d0 + TILE_N;

        స్థిరరాశి __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((స్థిరరాశి __m256i *)d0));
        స్థిరరాశి __m512 vdmin = _mm512_cvtph_ps(_mm256_loadu_si256((స్థిరరాశి __m256i *)dmin));

        కొరకు (పూర్ణాంకం m = 0; m < nr; ++m) {
            స్థిరరాశి దశాంశం d1 = A[m * lda].d;
            స్థిరరాశి __m512 vd = _mm512_mul_ps(_mm512_set1_ps(d1), vd0);
            స్థిరరాశి __m512 vdm = _mm512_mul_ps(_mm512_set1_ps(-d1), vdmin);
            స్థిరరాశి __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            ఒకవేళ (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } లేకపోతే {
                vsum = _mm512_set1_ps(0.f);
            }

            స్థిరరాశి __m256i q8sums = _mm256_loadu_si256((స్థిరరాశి __m256i *)A[m * lda].bsums);
            స్థిరరాశి __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));

            __m512i acc_m = _mm512_setzero_si512();
            కొరకు (పూర్ణాంకం k = 0; k < 4; ++k) {
                __m512i vmask = _mm512_set1_epi32(k);
                __m512i va = _mm512_permutexvar_epi32(vmask, _mm512_castsi128_si512(q8s));
                __m512i vb = _mm512_cvtepi8_epi16(_mm256_loadu_si256((స్థిరరాశి __m256i *)(mins + k * 32)));
                acc_m = _mm512_dpwssds_epi32(acc_m, va, vb);
            }

            vsum = _mm512_fmadd_ps(vtile, vd, vsum);
            vsum = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc_m), vdm, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

మూస <తార్కిక is_acc>
నిర్మాణం acc_C<ఖండం_q8_K, ఖండం_q6_K, is_acc> {
    స్థిర శూన్య apply(దశాంశం * RESTRICT C, పూర్ణాంకం ldc, స్థిరరాశి int32_t * RESTRICT tile, స్థిరరాశి ఖండం_q8_K * A, పూర్ణాంకం lda, స్థిరరాశి శూన్య * packed_B, పూర్ణాంకం nr) {
        స్థిరరాశి uint8_t * scales = reinterpret_cast<స్థిరరాశి uint8_t *>((స్థిరరాశి అక్షరం *)packed_B + (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N);
        స్థిరరాశి ggml_half * d0 = reinterpret_cast<స్థిరరాశి ggml_half *>(scales + 16 * TILE_N);

        స్థిరరాశి __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((స్థిరరాశి __m256i *)d0));

        కొరకు (పూర్ణాంకం m = 0; m < nr; ++m) {
            స్థిరరాశి దశాంశం d1 = A[m * lda].d;
            స్థిరరాశి __m512 vd = _mm512_mul_ps(_mm512_set1_ps(d1), vd0);
            స్థిరరాశి __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            ఒకవేళ (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } లేకపోతే {
                vsum = _mm512_set1_ps(0.f);
            }

            vsum = _mm512_fmadd_ps(vtile, vd, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

మూస <తార్కిక is_acc>
నిర్మాణం acc_C<ఖండం_q8_K, ఖండం_iq4_xs, is_acc> {
    స్థిర శూన్య apply(దశాంశం * RESTRICT C, పూర్ణాంకం ldc, స్థిరరాశి int32_t * RESTRICT tile, స్థిరరాశి ఖండం_q8_K * A, పూర్ణాంకం lda, స్థిరరాశి శూన్య * packed_B, పూర్ణాంకం nr) {
        స్థిరరాశి int8_t * scales = reinterpret_cast<స్థిరరాశి int8_t *>((స్థిరరాశి అక్షరం *)packed_B + (QK_K / 2) * TILE_N);
        స్థిరరాశి ggml_half * d0 = reinterpret_cast<స్థిరరాశి ggml_half *>(scales + 8 * TILE_N);

        స్థిరరాశి __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((స్థిరరాశి __m256i *)d0));

        కొరకు (పూర్ణాంకం m = 0; m < nr; ++m) {
            స్థిరరాశి దశాంశం d1 = A[m * lda].d;
            స్థిరరాశి __m512 vd = _mm512_mul_ps(_mm512_set1_ps(d1), vd0);
            స్థిరరాశి __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            ఒకవేళ (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } లేకపోతే {
                vsum = _mm512_set1_ps(0.f);
            }

            vsum = _mm512_fmadd_ps(vtile, vd, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

మూస <రకం_పేరు TB> constexpr పూర్ణాంకం get_quants_కొలత();
మూస <> constexpr పూర్ణాంకం get_quants_కొలత<ఖండం_q4_K>() { తిరిగి (QK_K / 2) * TILE_N; }
మూస <> constexpr పూర్ణాంకం get_quants_కొలత<ఖండం_q5_K>() { తిరిగి (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N; }
మూస <> constexpr పూర్ణాంకం get_quants_కొలత<ఖండం_q6_K>() { తిరిగి (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N; }
మూస <> constexpr పూర్ణాంకం get_quants_కొలత<ఖండం_iq4_xs>() { తిరిగి (QK_K / 2) * TILE_N; }

// used for QKK format
మూస <రకం_పేరు TB, తార్కిక is_acc,
          రకం_పేరు std::enable_if<is_type_qkk<TB>::value, పూర్ణాంకం>::type = 0>
అంతఃవరుస శూన్య scale_C(స్థిరరాశి int32_t * RESTRICT tile, int32_t * RESTRICT sumi, స్థిరరాశి శూన్య * packed_B, పూర్ణాంకం k, పూర్ణాంకం nr) {
    స్థిరరాశి uint8_t * scales = reinterpret_cast<స్థిరరాశి uint8_t *>((స్థిరరాశి అక్షరం *)packed_B + get_quants_కొలత<TB>());
    స్థిరరాశి __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((స్థిరరాశి __m128i *)(scales + k * TILE_N)));

    కొరకు (పూర్ణాంకం m = 0; m < nr; ++m) {
        __m512i vsumi;
        ఒకవేళ (is_acc) {
            vsumi = _mm512_loadu_si512(sumi + m * TILE_N);
        } లేకపోతే {
            vsumi = _mm512_setzero_si512();
        }
        __m512i vtile = _mm512_loadu_si512(tile + m * TILE_N);
        vsumi = mm512_కూడు_epi32(vsumi, _mm512_mullo_epi32(vtile, vscale));
        _mm512_storeu_si512((__m512i *)(sumi + m * TILE_N), vsumi);
    }
}

మూస <రకం_పేరు TA, రకం_పేరు TB, రకం_పేరు TC, పూర్ణాంకం ఖండం_M, పూర్ణాంకం ఖండం_N, పూర్ణాంకం ఖండం_K>
నిర్మాణం tinygemm_kernel_avx {
    స్థిర శూన్య apply(పూర్ణాంకం K, స్థిరరాశి TA * RESTRICT A, స్థిరరాశి TB * RESTRICT B, TC * RESTRICT C, పూర్ణాంకం ldc) {
        GGML_వాడని(K);
        GGML_వాడని(A);
        GGML_వాడని(B);
        GGML_వాడని(C);
        GGML_వాడని(ldc);
    }
};

మూస <పూర్ణాంకం ఖండం_M, పూర్ణాంకం ఖండం_N, పూర్ణాంకం ఖండం_K>
నిర్మాణం tinygemm_kernel_avx<దశాంశం, ggml_fp16_t, దశాంశం, ఖండం_M, ఖండం_N, ఖండం_K> {
    స్థిర శూన్య apply(పూర్ణాంకం K, స్థిరరాశి దశాంశం * RESTRICT A, స్థిరరాశి ggml_fp16_t * RESTRICT B, దశాంశం * RESTRICT C, పూర్ణాంకం ldc) {
        constexpr పూర్ణాంకం అడ్డువరుసలు_2 = ఖండం_M;
        constexpr పూర్ణాంకం నిలువుల_సంఖ్య = ఖండం_N;
        ప్రతిజ్ఞా(ఖండం_K == 16);

        __m512 va;
        __m512 vb[నిలువుల_సంఖ్య];
        __m512 vc[అడ్డువరుసలు_2 * నిలువుల_సంఖ్య];

        స్వయంచాలక loadc = [&](స్వయంచాలక idx) {
            vc[idx] = _mm512_setzero_ps();
        };
        Unroll<అడ్డువరుసలు_2 * నిలువుల_సంఖ్య>{}(loadc);

        స్వయంచాలక compute = [&](స్వయంచాలక idx, స్వయంచాలక k) {
            constexpr పూర్ణాంకం అడ్డువరుస = idx / నిలువుల_సంఖ్య;
            constexpr పూర్ణాంకం నిలువు = idx % నిలువుల_సంఖ్య;

            ఒకవేళ constexpr (నిలువు == 0) {
                va = _mm512_loadu_ps(A + అడ్డువరుస * K + k);
            }
            ఒకవేళ constexpr (అడ్డువరుస == 0) {
                vb[నిలువు] =  _mm512_cvtph_ps(_mm256_loadu_si256((స్థిరరాశి __m256i *)(B + నిలువు * K + k)));
            }
            vc[idx] = _mm512_fmadd_ps(va, vb[నిలువు], vc[idx]);
        };

        కొరకు (పూర్ణాంకం k = 0; k < K; k += 16) {
            Unroll<అడ్డువరుసలు_2 * నిలువుల_సంఖ్య>{}(compute, k);
        }

        స్వయంచాలక storec = [&](స్వయంచాలక idx) {
            constexpr పూర్ణాంకం అడ్డువరుస = idx / నిలువుల_సంఖ్య;
            constexpr పూర్ణాంకం నిలువు = idx % నిలువుల_సంఖ్య;
            C[అడ్డువరుస * ldc + నిలువు] = mm512_reduce_కూడు_ps(vc[idx]);
        };
        Unroll<అడ్డువరుసలు_2 * నిలువుల_సంఖ్య>{}(storec);
    }
};

#నిర్వచించు LAUNCH_TINYGEMM_KERNEL_AVX(MB_కొలత, NB_కొలత)                                \
    tinygemm_kernel_avx<దశాంశం, type, దశాంశం, MB_కొలత, NB_కొలత, blck_కొలత>::apply(    \
        K, (స్థిరరాశి దశాంశం *)src1->దత్తాంశం + src1_స్థానభ్రంశం + mb_ప్రారంభం * K,                  \
        (స్థిరరాశి type *)src0->దత్తాంశం + src0_స్థానభ్రంశం + nb_ప్రారంభం * K,                      \
        (దశాంశం *)dst->దత్తాంశం + dst_స్థానభ్రంశం + mb_ప్రారంభం * ldc + nb_ప్రారంభం, ldc)


// re-organize in the format {NB, KB, TILE_SIZE}:
#నిర్వచించు PACKED_సూచీ(n, k, KB, tile_కొలత) (n * KB + k) * tile_కొలత

మూస<రకం_పేరు TB, పూర్ణాంకం ఖండం_K>
శూన్య convert_B_packed_format(శూన్య * RESTRICT packed_B, స్థిరరాశి TB * RESTRICT B, పూర్ణాంకం N, పూర్ణాంకం K) {
    స్థిరరాశి పూర్ణాంకం NB = N / TILE_N;
    స్థిరరాశి పూర్ణాంకం KB = K / ఖండం_K;
    స్థిరరాశి పూర్ణాంకం TILE_కొలత = get_tile_కొలత<TB>();

    // parallel on NB should be enough
    parallel_for(NB, [&](పూర్ణాంకం begin, పూర్ణాంకం end) {
        కొరకు (పూర్ణాంకం n = begin; n < end; ++n) {
            కొరకు (పూర్ణాంకం k = 0; k < KB; ++k) {
                పూర్ణాంకం n0 = n * TILE_N;
                pack_B((అక్షరం *)packed_B + PACKED_సూచీ(n, k, KB, TILE_కొలత), &B[n0 * KB + k], KB);
            }
        }
    });
}

మూస <రకం_పేరు TA, రకం_పేరు TB, రకం_పేరు TC, పూర్ణాంకం ఖండం_M, పూర్ణాంకం ఖండం_N, పూర్ణాంకం ఖండం_K>
నిర్మాణం tinygemm_kernel_vnni {};

మూస <పూర్ణాంకం ఖండం_M, పూర్ణాంకం ఖండం_N, పూర్ణాంకం ఖండం_K>
నిర్మాణం tinygemm_kernel_vnni<ఖండం_q8_0, ఖండం_q4_0, దశాంశం, ఖండం_M, ఖండం_N, ఖండం_K> {
    స్థిర శూన్య apply(పూర్ణాంకం KB, స్థిరరాశి శూన్య * RESTRICT _A, స్థిరరాశి శూన్య * RESTRICT _B, దశాంశం * RESTRICT C, పూర్ణాంకం ldc) {

        constexpr పూర్ణాంకం నిలువుల_సంఖ్య = ఖండం_N / 16;
        స్థిరరాశి పూర్ణాంకం TILE_కొలత = TILE_N * పరిమాణం(ఖండం_q4_0);

        స్థిరరాశి ఖండం_q8_0 * RESTRICT A = static_cast<స్థిరరాశి ఖండం_q8_0 *>(_A);
        స్థిరరాశి అక్షరం * RESTRICT B = static_cast<స్థిరరాశి అక్షరం *>(_B);

        __m512i va[8];
        __m512 vc[నిలువుల_సంఖ్య];
        __m512 vd1;

        // sum of offsets, shared across COLS
        //
        // avx512-vnni does not have `_mm512_dpbssd_epi32`,
        // need to transform ss to us:
        //   a * (b - 8) is equivalent to b * a - 8 * a
        //   s    u   u                   u   s   u   s
        //
        __m512i vcomp;

        స్థిరరాశి __m512i off = _mm512_set1_epi8(8);
        స్థిరరాశి __m512i దిగువ_ముసుగు = _mm512_set1_epi8(0xF);

        స్వయంచాలక loadc = [&](స్వయంచాలక నిలువు) {
            vc[నిలువు] = _mm512_setzero_ps();
        };
        Unroll<నిలువుల_సంఖ్య>{}(loadc);

        స్వయంచాలక compute = [&](స్వయంచాలక నిలువు, స్వయంచాలక i) {
            // load a and compute compensation
            ఒకవేళ constexpr (నిలువు == 0) {
                స్థిరరాశి int32_t * a_ptr = reinterpret_cast<స్థిరరాశి int32_t *>(A[0 * KB + i].qs);
                vcomp = _mm512_setzero_si512();
                కొరకు (పూర్ణాంకం k = 0; k < 8; ++k) {
                    va[k] = _mm512_set1_epi32(a_ptr[k]);
                    vcomp = _mm512_dpbusd_epi32(vcomp, off, va[k]);
                }
                vd1 = _mm512_set1_ps(GGML_CPU_FP16_గమ్యం_FP32(A[0 * KB + i].d));
            }

            // load b
            __m512i vsum = _mm512_setzero_si512();
            స్థిరరాశి అక్షరం * b_ptr = B + PACKED_సూచీ(నిలువు, i, KB, TILE_కొలత);
            కొరకు (పూర్ణాంకం k = 0; k < 8; k += 2) {
                __m512i బైట్లు = _mm512_loadu_si512((స్థిరరాశి __m512i *)(b_ptr + k * 32));
                __m512i vb0 = _mm512_and_si512(బైట్లు, దిగువ_ముసుగు);
                vsum = _mm512_dpbusd_epi32(vsum, vb0, va[k + 0]);
                __m512i vb1 = _mm512_and_si512(_mm512_srli_epi16(బైట్లు, 4), దిగువ_ముసుగు);
                vsum = _mm512_dpbusd_epi32(vsum, vb1, va[k + 1]);
            }
            స్థిరరాశి పూర్ణాంకం స్థానభ్రంశం = TILE_N * TILE_K / 2;
            స్థిరరాశి __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((స్థిరరాశి __m256i *)(b_ptr + స్థానభ్రంశం)));
            vsum = _mm512_sub_epi32(vsum, vcomp);

            vc[నిలువు] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vsum), _mm512_mul_ps(vd0, vd1), vc[నిలువు]);
        };

        కొరకు (పూర్ణాంకం i = 0; i < KB; ++i) {
            Unroll<నిలువుల_సంఖ్య>{}(compute, i);
        }

        //store to C
        స్వయంచాలక storec = [&](స్వయంచాలక నిలువు) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + నిలువు * 16), vc[నిలువు]);
        };
        Unroll<నిలువుల_సంఖ్య>{}(storec);
    }
};

మూస <పూర్ణాంకం ఖండం_N, పూర్ణాంకం ఖండం_K>
నిర్మాణం tinygemm_kernel_vnni<ఖండం_q8_1, ఖండం_q4_1, దశాంశం, 1, ఖండం_N, ఖండం_K> {
    స్థిర శూన్య apply(పూర్ణాంకం KB, స్థిరరాశి శూన్య * RESTRICT _A, స్థిరరాశి శూన్య * RESTRICT _B, దశాంశం * RESTRICT C, పూర్ణాంకం ldc) {

        constexpr పూర్ణాంకం నిలువుల_సంఖ్య = ఖండం_N / 16;
        స్థిరరాశి పూర్ణాంకం TILE_కొలత = TILE_N * పరిమాణం(ఖండం_q4_1);

        స్థిరరాశి ఖండం_q8_1 * RESTRICT A = static_cast<స్థిరరాశి ఖండం_q8_1 *>(_A);
        స్థిరరాశి అక్షరం * RESTRICT B = static_cast<స్థిరరాశి అక్షరం *>(_B);

        __m512i va[8];
        __m512i vb[8];
        __m512 vc[నిలువుల_సంఖ్య];
        __m512 vd1, vs1;

        స్థిరరాశి __m512i దిగువ_ముసుగు = _mm512_set1_epi8(0xF);

        స్వయంచాలక loadc = [&](స్వయంచాలక నిలువు) {
            vc[నిలువు] = _mm512_setzero_ps();
        };
        Unroll<నిలువుల_సంఖ్య>{}(loadc);

        స్వయంచాలక compute = [&](స్వయంచాలక నిలువు, స్వయంచాలక i) {
            // load a
            ఒకవేళ constexpr (నిలువు == 0) {
                స్థిరరాశి int32_t * a_ptr = reinterpret_cast<స్థిరరాశి int32_t *>(A[0 * KB + i].qs);
                కొరకు (పూర్ణాంకం k = 0; k < 8; ++k) {
                    va[k] = _mm512_set1_epi32(a_ptr[k]);
                }
                vd1 = _mm512_set1_ps(GGML_CPU_FP16_గమ్యం_FP32(A[0 * KB + i].d));
                vs1 = _mm512_set1_ps(GGML_CPU_FP16_గమ్యం_FP32(A[0 * KB + i].s));
            }

            // load b
            స్థిరరాశి అక్షరం * b_ptr = B + PACKED_సూచీ(నిలువు, i, KB, TILE_కొలత);
            కొరకు (పూర్ణాంకం k = 0; k < 8; k += 2) {
                __m512i బైట్లు = _mm512_loadu_si512((స్థిరరాశి __m512i *)(b_ptr + k * 32));
                vb[k + 0] = _mm512_and_si512(బైట్లు, దిగువ_ముసుగు);
                vb[k + 1] = _mm512_and_si512(_mm512_srli_epi16(బైట్లు, 4), దిగువ_ముసుగు);
            }
            స్థిరరాశి పూర్ణాంకం స్థానభ్రంశం = TILE_N * TILE_K / 2;
            స్థిరరాశి __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((స్థిరరాశి __m256i *)(b_ptr + స్థానభ్రంశం)));
            స్థిరరాశి __m512 vm0 = _mm512_cvtph_ps(_mm256_loadu_si256((స్థిరరాశి __m256i *)(b_ptr + స్థానభ్రంశం + TILE_N * పరిమాణం(ggml_half))));

            __m512i vsum = _mm512_setzero_si512();
            కొరకు (పూర్ణాంకం k = 0; k < 8; ++k) {
                vsum = _mm512_dpbusd_epi32(vsum, vb[k], va[k]);
            }

            vc[నిలువు] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vsum), _mm512_mul_ps(vd0, vd1), vc[నిలువు]);
            vc[నిలువు] = _mm512_fmadd_ps(vm0, vs1, vc[నిలువు]);
        };

        కొరకు (పూర్ణాంకం i = 0; i < KB; ++i) {
            Unroll<నిలువుల_సంఖ్య>{}(compute, i);
        }

        //store to C
        స్వయంచాలక storec = [&](స్వయంచాలక నిలువు) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + నిలువు * 16), vc[నిలువు]);
        };
        Unroll<నిలువుల_సంఖ్య>{}(storec);
    }
};

మూస <పూర్ణాంకం ఖండం_M, పూర్ణాంకం ఖండం_N, పూర్ణాంకం ఖండం_K>
నిర్మాణం tinygemm_kernel_vnni<ఖండం_q8_0, ఖండం_q8_0, దశాంశం, ఖండం_M, ఖండం_N, ఖండం_K> {
    స్థిర శూన్య apply(పూర్ణాంకం KB, స్థిరరాశి శూన్య * RESTRICT _A, స్థిరరాశి శూన్య * RESTRICT _B, దశాంశం * RESTRICT C, పూర్ణాంకం ldc) {

        constexpr పూర్ణాంకం నిలువుల_సంఖ్య = ఖండం_N / 16;
        స్థిరరాశి పూర్ణాంకం TILE_కొలత = TILE_N * పరిమాణం(ఖండం_q8_0) + TILE_N * పరిమాణం(int32_t);

        స్థిరరాశి ఖండం_q8_0 * RESTRICT A = static_cast<స్థిరరాశి ఖండం_q8_0 *>(_A);
        స్థిరరాశి అక్షరం * RESTRICT B = static_cast<స్థిరరాశి అక్షరం *>(_B);

        __m512i va[8];
        __m512i vb[8];
        __m512 vc[నిలువుల_సంఖ్య];
        __m512 vd1;

        // Notes: s8s8 igemm compensation in avx512-vnni
        // change s8s8 to u8s8 with compensate
        //   a * b = (a + 128) * b - 128 * b
        //   s   s       u       s    u    s
        //
        // (128 * b is pre-computed when packing B to vnni formats)
        //
        స్థిరరాశి __m512i off = _mm512_set1_epi8(static_cast<అక్షరం>(0x80));

        స్వయంచాలక loadc = [&](స్వయంచాలక నిలువు) {
            vc[నిలువు] = _mm512_setzero_ps();
        };
        Unroll<నిలువుల_సంఖ్య>{}(loadc);

        స్వయంచాలక compute = [&](స్వయంచాలక నిలువు, స్వయంచాలక i) {
            // load a and add offset 128
            ఒకవేళ constexpr (నిలువు == 0) {
                స్థిరరాశి int32_t * a_ptr = reinterpret_cast<స్థిరరాశి int32_t *>(A[0 * KB + i].qs);
                కొరకు (పూర్ణాంకం k = 0; k < 8; ++k) {
                    va[k] = _mm512_set1_epi32(a_ptr[k]);
                    va[k] = mm512_కూడు_epi8(va[k], off);
                }
                vd1 = _mm512_set1_ps(GGML_CPU_FP16_గమ్యం_FP32(A[0 * KB + i].d));
            }

            // load b
            స్థిరరాశి అక్షరం * b_ptr = B + PACKED_సూచీ(నిలువు, i, KB, TILE_కొలత);
            కొరకు (పూర్ణాంకం k = 0; k < 8; ++k) {
                vb[k] = _mm512_loadu_si512((స్థిరరాశి __m512i *)(b_ptr + k * 64));
            }
            స్థిరరాశి పూర్ణాంకం స్థానభ్రంశం = TILE_N * TILE_K;
            స్థిరరాశి __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((స్థిరరాశి __m256i *)(b_ptr + స్థానభ్రంశం)));
            స్థిరరాశి పూర్ణాంకం offset2 = TILE_N * TILE_K + TILE_N * పరిమాణం(ggml_half);
            స్థిరరాశి __m512i vcomp = _mm512_loadu_si512((స్థిరరాశి __m512i *)(b_ptr + offset2));

            __m512i vsum = _mm512_setzero_si512();
            కొరకు (పూర్ణాంకం k = 0; k < 8; ++k) {
                vsum = _mm512_dpbusd_epi32(vsum, va[k], vb[k]);
            }
            vsum = _mm512_sub_epi32(vsum, vcomp);

            vc[నిలువు] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vsum), _mm512_mul_ps(vd0, vd1), vc[నిలువు]);
        };

        కొరకు (పూర్ణాంకం i = 0; i < KB; ++i) {
            Unroll<నిలువుల_సంఖ్య>{}(compute, i);
        }

        //store to C
        స్వయంచాలక storec = [&](స్వయంచాలక నిలువు) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + నిలువు * 16), vc[నిలువు]);
        };
        Unroll<నిలువుల_సంఖ్య>{}(storec);
    }
};

మూస <పూర్ణాంకం ఖండం_M, పూర్ణాంకం ఖండం_N, పూర్ణాంకం ఖండం_K>
నిర్మాణం tinygemm_kernel_vnni<ఖండం_q8_K, ఖండం_q4_K, దశాంశం, ఖండం_M, ఖండం_N, ఖండం_K> {
    స్థిర శూన్య apply(పూర్ణాంకం KB, స్థిరరాశి శూన్య * RESTRICT _A, స్థిరరాశి శూన్య * RESTRICT _B, దశాంశం * RESTRICT C, పూర్ణాంకం ldc) {

        constexpr పూర్ణాంకం నిలువుల_సంఖ్య = ఖండం_N / 16;
        స్థిరరాశి పూర్ణాంకం TILE_కొలత = TILE_N * పరిమాణం(ఖండం_q4_K) + TILE_N * 4;

        స్థిరరాశి ఖండం_q8_K * RESTRICT A = static_cast<స్థిరరాశి ఖండం_q8_K *>(_A);
        స్థిరరాశి అక్షరం * RESTRICT B = static_cast<స్థిరరాశి అక్షరం *>(_B);

        // a.qs:   8 groups, 32 bytes each group (m256i)
        __m512i va[8];
        // a.bsum: 8 groups,  2 bytes each group (m128i)
        __m512i va_bsum;
        __m512 vc[నిలువుల_సంఖ్య];
        __m512 vd1;

        // packed_B:
        స్థిరరాశి పూర్ణాంకం స్థానభ్రంశం_scales = (QK_K / 2) * TILE_N;
        స్థిరరాశి పూర్ణాంకం స్థానభ్రంశం_mins   = (QK_K / 2) * TILE_N +  8 * TILE_N;
        స్థిరరాశి పూర్ణాంకం స్థానభ్రంశం_d0     = (QK_K / 2) * TILE_N + 16 * TILE_N;
        స్థిరరాశి పూర్ణాంకం స్థానభ్రంశం_dmin   = (QK_K / 2) * TILE_N + 16 * TILE_N + TILE_N * పరిమాణం(ggml_half);

        స్థిరరాశి __m512i దిగువ_ముసుగు = _mm512_set1_epi8(0xF);

        స్వయంచాలక loadc = [&](స్వయంచాలక నిలువు) {
            vc[నిలువు] = _mm512_setzero_ps();
        };
        Unroll<నిలువుల_సంఖ్య>{}(loadc);

        // Notes: vnni formats in QK_K
        //   a) quants vnni format
        //     int8  {k/4, n, 4}, viewed as 2d {k/4, 4n}, k = 32
        //     from {16, 32} to {8, 64}
        //
        //   b) min vnni format
        //     int16 {k/2, n, 2}, viewed as 2d {k/2, 2n}, k = 8
        //     from {16,  8} to {4, 32}
        //
        స్వయంచాలక compute = [&](స్వయంచాలక నిలువు, స్వయంచాలక i) {
            // load a
            ఒకవేళ constexpr (నిలువు == 0) {
                కొరకు (పూర్ణాంకం k_సమూహం = 0; k_సమూహం < QK_K / 32; ++k_సమూహం) {
                    va[k_సమూహం] = _mm512_castsi256_si512(_mm256_loadu_si256((స్థిరరాశి __m256i *)(A[0 * KB + i].qs + k_సమూహం * 32)));
                }
                స్థిరరాశి __m256i q8sums = _mm256_loadu_si256((స్థిరరాశి __m256i *)A[0 * KB + i].bsums);
                స్థిరరాశి __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
                va_bsum = _mm512_castsi128_si512(q8s);
                vd1 = _mm512_set1_ps(A[0 * KB + i].d);
            }

            // step 1: accumultate the quants
            __m512i acc = _mm512_setzero_si512();
            స్థిరరాశి అక్షరం * b_ptr = B + PACKED_సూచీ(నిలువు, i, KB, TILE_కొలత);
            స్థిరరాశి అక్షరం * b_qs  = b_ptr;
            కొరకు (పూర్ణాంకం k_సమూహం = 0; k_సమూహం < QK_K / 32; ++k_సమూహం) {
                __m512i vsum = _mm512_setzero_si512();
                కొరకు (పూర్ణాంకం k = 0; k < 8; k += 2) {
                    __m512i va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(k + 0), va[k_సమూహం]);
                    __m512i va1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(k + 1), va[k_సమూహం]);

                    __m512i బైట్లు = _mm512_loadu_si512((స్థిరరాశి __m512i *)b_qs);
                    __m512i vb0 = _mm512_and_si512(బైట్లు, దిగువ_ముసుగు);
                    vsum = _mm512_dpbusd_epi32(vsum, vb0, va0);
                    __m512i vb1 = _mm512_and_si512(_mm512_srli_epi16(బైట్లు, 4), దిగువ_ముసుగు);
                    vsum = _mm512_dpbusd_epi32(vsum, vb1, va1);

                    b_qs += 64;
                }
                // vacc += scale * (q8 @ q4)
                స్థిరరాశి __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((స్థిరరాశి __m128i *)(b_ptr + స్థానభ్రంశం_scales + k_సమూహం * TILE_N)));
                acc = mm512_కూడు_epi32(acc, _mm512_mullo_epi32(vsum, vscale));
            }
            స్థిరరాశి __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((స్థిరరాశి __m256i *)(b_ptr + స్థానభ్రంశం_d0)));
            vc[నిలువు] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(vd0, vd1), vc[నిలువు]);

            // step 2: accumulate the mins
            __m512i acc_m = _mm512_setzero_si512();
            కొరకు (పూర్ణాంకం k = 0; k < 4; ++k) {
                __m512i vmask = _mm512_set1_epi32(k);
                __m512i va = _mm512_permutexvar_epi32(vmask, va_bsum);
                __m512i vb = _mm512_cvtepi8_epi16(_mm256_loadu_si256((స్థిరరాశి __m256i *)(b_ptr + స్థానభ్రంశం_mins + k * 32)));
                acc_m = _mm512_dpwssds_epi32(acc_m, va, vb);
            }
            స్థిరరాశి __m512 vdmin = _mm512_cvtph_ps(_mm256_loadu_si256((స్థిరరాశి __m256i *)(b_ptr + స్థానభ్రంశం_dmin)));
            vc[నిలువు] = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(acc_m), _mm512_mul_ps(vdmin, vd1), vc[నిలువు]);
        };

        కొరకు (పూర్ణాంకం i = 0; i < KB; ++i) {
            Unroll<నిలువుల_సంఖ్య>{}(compute, i);
        }

        //store to C
        స్వయంచాలక storec = [&](స్వయంచాలక నిలువు) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + నిలువు * 16), vc[నిలువు]);
        };
        Unroll<నిలువుల_సంఖ్య>{}(storec);
    }
};

మూస <పూర్ణాంకం ఖండం_M, పూర్ణాంకం ఖండం_N, పూర్ణాంకం ఖండం_K>
నిర్మాణం tinygemm_kernel_vnni<ఖండం_q8_K, ఖండం_q5_K, దశాంశం, ఖండం_M, ఖండం_N, ఖండం_K> {
    స్థిర శూన్య apply(పూర్ణాంకం KB, స్థిరరాశి శూన్య * RESTRICT _A, స్థిరరాశి శూన్య * RESTRICT _B, దశాంశం * RESTRICT C, పూర్ణాంకం ldc) {

        constexpr పూర్ణాంకం నిలువుల_సంఖ్య = ఖండం_N / 16;
        స్థిరరాశి పూర్ణాంకం TILE_కొలత = TILE_N * పరిమాణం(ఖండం_q5_K) + TILE_N * 4;

        స్థిరరాశి ఖండం_q8_K * RESTRICT A = static_cast<స్థిరరాశి ఖండం_q8_K *>(_A);
        స్థిరరాశి అక్షరం * RESTRICT B = static_cast<స్థిరరాశి అక్షరం *>(_B);

        // a.qs:   8 groups, 32 bytes each group (m256i)
        __m512i va[8];
        // a.bsum: 8 groups,  2 bytes each group (m128i)
        __m512i va_bsum;
        __m512 vc[నిలువుల_సంఖ్య];
        __m512 vd1;

        // packed_B:
        స్థిరరాశి పూర్ణాంకం స్థానభ్రంశం_qh     = (QK_K / 2) * TILE_N;
        స్థిరరాశి పూర్ణాంకం స్థానభ్రంశం_scales = (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N;
        స్థిరరాశి పూర్ణాంకం స్థానభ్రంశం_mins   = (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N +  8 * TILE_N;
        స్థిరరాశి పూర్ణాంకం స్థానభ్రంశం_d0     = (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N + 16 * TILE_N;
        స్థిరరాశి పూర్ణాంకం స్థానభ్రంశం_dmin   = (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N + 16 * TILE_N + TILE_N * పరిమాణం(ggml_half);

        స్థిరరాశి __m512i దిగువ_ముసుగు = _mm512_set1_epi8(0xF);

        స్వయంచాలక loadc = [&](స్వయంచాలక నిలువు) {
            vc[నిలువు] = _mm512_setzero_ps();
        };
        Unroll<నిలువుల_సంఖ్య>{}(loadc);

        // Q5_K and Q4_K shares the same vnni formats, refer to notes above.
        స్వయంచాలక compute = [&](స్వయంచాలక నిలువు, స్వయంచాలక i) {
            // load a
            ఒకవేళ constexpr (నిలువు == 0) {
                కొరకు (పూర్ణాంకం k_సమూహం = 0; k_సమూహం < QK_K / 32; ++k_సమూహం) {
                    va[k_సమూహం] = _mm512_castsi256_si512(_mm256_loadu_si256((స్థిరరాశి __m256i *)(A[0 * KB + i].qs + k_సమూహం * 32)));
                }
                స్థిరరాశి __m256i q8sums = _mm256_loadu_si256((స్థిరరాశి __m256i *)A[0 * KB + i].bsums);
                స్థిరరాశి __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
                va_bsum = _mm512_castsi128_si512(q8s);
                vd1 = _mm512_set1_ps(A[0 * KB + i].d);
            }

            // step 1: accumultate the quants
            __m512i acc = _mm512_setzero_si512();
            స్థిరరాశి అక్షరం * b_ptr = B + PACKED_సూచీ(నిలువు, i, KB, TILE_కొలత);
            స్థిరరాశి అక్షరం * b_qs  = b_ptr;
            స్థిరరాశి అక్షరం * b_qh  = b_ptr + స్థానభ్రంశం_qh;
            కొరకు (పూర్ణాంకం k_సమూహం = 0; k_సమూహం < QK_K / 32; ++k_సమూహం) {
                __m512i vsum = _mm512_setzero_si512();
                __m512i hmask0 = _mm512_set1_epi8(0x1);
                __m512i hmask1 = _mm512_set1_epi8(0x2);
                __m512i hbits = _mm512_loadu_si512((స్థిరరాశి __m512i *)(b_qh + k_సమూహం * 64));
                కొరకు (పూర్ణాంకం k = 0; k < 8; k += 2) {
                    __m512i va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(k + 0), va[k_సమూహం]);
                    __m512i va1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(k + 1), va[k_సమూహం]);

                    __m512i బైట్లు = _mm512_loadu_si512((స్థిరరాశి __m512i *)b_qs);
                    __m512i vb0 = _mm512_and_si512(బైట్లు, దిగువ_ముసుగు);
                    __m512i vb1 = _mm512_and_si512(_mm512_srli_epi16(బైట్లు, 4), దిగువ_ముసుగు);

                    __m512i vh0 = _mm512_slli_epi16(_mm512_srli_epi16(_mm512_and_si512(hbits, hmask0), k), 4);
                    __m512i vh1 = _mm512_slli_epi16(_mm512_srli_epi16(_mm512_and_si512(hbits, hmask1), k + 1), 4);

                    hmask0 = _mm512_slli_epi16(hmask0, 2);
                    hmask1 = _mm512_slli_epi16(hmask1, 2);
                    vb0 = mm512_కూడు_epi8(vb0, vh0);
                    vb1 = mm512_కూడు_epi8(vb1, vh1);

                    vsum = _mm512_dpbusd_epi32(vsum, vb0, va0);
                    vsum = _mm512_dpbusd_epi32(vsum, vb1, va1);

                    b_qs += 64;
                }
                // vacc += scale * (q8 @ q5)
                స్థిరరాశి __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((స్థిరరాశి __m128i *)(b_ptr + స్థానభ్రంశం_scales + k_సమూహం * TILE_N)));
                acc = mm512_కూడు_epi32(acc, _mm512_mullo_epi32(vsum, vscale));
            }
            స్థిరరాశి __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((స్థిరరాశి __m256i *)(b_ptr + స్థానభ్రంశం_d0)));
            vc[నిలువు] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(vd0, vd1), vc[నిలువు]);

            // step 2: accumulate the mins
            __m512i acc_m = _mm512_setzero_si512();
            కొరకు (పూర్ణాంకం k = 0; k < 4; ++k) {
                __m512i vmask = _mm512_set1_epi32(k);
                __m512i va = _mm512_permutexvar_epi32(vmask, va_bsum);
                __m512i vb = _mm512_cvtepi8_epi16(_mm256_loadu_si256((స్థిరరాశి __m256i *)(b_ptr + స్థానభ్రంశం_mins + k * 32)));
                acc_m = _mm512_dpwssds_epi32(acc_m, va, vb);
            }
            స్థిరరాశి __m512 vdmin = _mm512_cvtph_ps(_mm256_loadu_si256((స్థిరరాశి __m256i *)(b_ptr + స్థానభ్రంశం_dmin)));
            vc[నిలువు] = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(acc_m), _mm512_mul_ps(vdmin, vd1), vc[నిలువు]);
        };

        కొరకు (పూర్ణాంకం i = 0; i < KB; ++i) {
            Unroll<నిలువుల_సంఖ్య>{}(compute, i);
        }

        //store to C
        స్వయంచాలక storec = [&](స్వయంచాలక నిలువు) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + నిలువు * 16), vc[నిలువు]);
        };
        Unroll<నిలువుల_సంఖ్య>{}(storec);
    }
};

మూస <పూర్ణాంకం ఖండం_M, పూర్ణాంకం ఖండం_N, పూర్ణాంకం ఖండం_K>
నిర్మాణం tinygemm_kernel_vnni<ఖండం_q8_K, ఖండం_q6_K, దశాంశం, ఖండం_M, ఖండం_N, ఖండం_K> {
    స్థిర శూన్య apply(పూర్ణాంకం KB, స్థిరరాశి శూన్య * RESTRICT _A, స్థిరరాశి శూన్య * RESTRICT _B, దశాంశం * RESTRICT C, పూర్ణాంకం ldc) {

        constexpr పూర్ణాంకం నిలువుల_సంఖ్య = ఖండం_N / 16;
        స్థిరరాశి పూర్ణాంకం TILE_కొలత = TILE_N * పరిమాణం(ఖండం_q6_K);

        స్థిరరాశి ఖండం_q8_K * RESTRICT A = static_cast<స్థిరరాశి ఖండం_q8_K *>(_A);
        స్థిరరాశి అక్షరం * RESTRICT B = static_cast<స్థిరరాశి అక్షరం *>(_B);

        // load the 256 bytes from A to 4 avx512 vectors
        __m512i va[4];
        __m512 vc[నిలువుల_సంఖ్య];
        __m512 vd1;

        // packed_B:
        స్థిరరాశి పూర్ణాంకం స్థానభ్రంశం_qh     = (QK_K / 2) * TILE_N;
        స్థిరరాశి పూర్ణాంకం స్థానభ్రంశం_scales = (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N;
        స్థిరరాశి పూర్ణాంకం స్థానభ్రంశం_d0     = (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N + 16 * TILE_N;

        // compensation
        __m512i vcomp;

        స్థిరరాశి __m512i m32s = _mm512_set1_epi32(32);
        స్థిరరాశి __m512i దిగువ_ముసుగు = _mm512_set1_epi8(0xF);

        స్వయంచాలక loadc = [&](స్వయంచాలక నిలువు) {
            vc[నిలువు] = _mm512_setzero_ps();
        };
        Unroll<నిలువుల_సంఖ్య>{}(loadc);

        స్వయంచాలక compute = [&](స్వయంచాలక నిలువు, స్వయంచాలక i) {
            ఒకవేళ constexpr (నిలువు == 0) {
                // load a
                va[0] = _mm512_loadu_si512((స్థిరరాశి __m512i *)(A[0 * KB + i].qs +   0));
                va[1] = _mm512_loadu_si512((స్థిరరాశి __m512i *)(A[0 * KB + i].qs +  64));
                va[2] = _mm512_loadu_si512((స్థిరరాశి __m512i *)(A[0 * KB + i].qs + 128));
                va[3] = _mm512_loadu_si512((స్థిరరాశి __m512i *)(A[0 * KB + i].qs + 192));

                స్థిరరాశి __m256i q8sums = _mm256_loadu_si256((స్థిరరాశి __m256i *)A[0 * KB + i].bsums);
                vcomp = _mm512_mullo_epi32(_mm512_cvtepi16_epi32(q8sums), m32s);
                vd1 = _mm512_set1_ps(A[0 * KB + i].d);
            }

            // accmulate the quants
            __m512i acc = _mm512_setzero_si512();
            స్థిరరాశి అక్షరం * b_ptr = B + PACKED_సూచీ(నిలువు, i, KB, TILE_కొలత);
            స్థిరరాశి అక్షరం * b_qs = b_ptr;
            స్థిరరాశి అక్షరం * b_qh = b_ptr + స్థానభ్రంశం_qh;
            పూర్ణాంకం ముసుగు = 0;
            కొరకు (పూర్ణాంకం k_సమూహం = 0; k_సమూహం < QK_K / 16; ++k_సమూహం) {
                పూర్ణాంకం r = k_సమూహం >> 2;
                __m512i va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(ముసుగు++), va[r]);
                __m512i va1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(ముసుగు++), va[r]);

                __m512i vsum = _mm512_setzero_si512();
                __m512i hmask = _mm512_set1_epi8(0x3);

                __m512i బైట్లు = _mm512_loadu_si512(b_qs);
                __m512i hbits = _mm512_loadu_si512(b_qh);
                __m512i vb0 = _mm512_and_si512(బైట్లు, దిగువ_ముసుగు);
                __m512i vb1 = _mm512_and_si512(_mm512_srli_epi16(బైట్లు, 4), దిగువ_ముసుగు);
                __m512i vh0 = _mm512_slli_epi16(_mm512_and_si512(hbits, hmask), 4);
                __m512i vh1 = _mm512_slli_epi16(_mm512_and_si512(hbits, _mm512_slli_epi16(hmask, 2)), 2);

                vb0 = mm512_కూడు_epi8(vb0, vh0);
                vb1 = mm512_కూడు_epi8(vb1, vh1);
                vsum = _mm512_dpbusd_epi32(vsum, vb0, va0);
                vsum = _mm512_dpbusd_epi32(vsum, vb1, va1);
                b_qs += 64;

                va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(ముసుగు++), va[r]);
                va1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(ముసుగు++), va[r]);

                బైట్లు = _mm512_loadu_si512(b_qs);
                vb0 = _mm512_and_si512(బైట్లు, దిగువ_ముసుగు);
                vb1 = _mm512_and_si512(_mm512_srli_epi16(బైట్లు, 4), దిగువ_ముసుగు);
                vh0 =                   _mm512_and_si512(hbits, _mm512_slli_epi16(hmask, 4));
                vh1 = _mm512_srli_epi16(_mm512_and_si512(hbits, _mm512_slli_epi16(hmask, 6)), 2);
                vb0 = mm512_కూడు_epi8(vb0, vh0);
                vb1 = mm512_కూడు_epi8(vb1, vh1);
                vsum = _mm512_dpbusd_epi32(vsum, vb0, va0);
                vsum = _mm512_dpbusd_epi32(vsum, vb1, va1);
                b_qs += 64;
                b_qh += 64;

                // B * A - 32 * A
                __m512i vmask = _mm512_set1_epi32(k_సమూహం);
                vsum = _mm512_sub_epi32(vsum, _mm512_permutexvar_epi32(vmask, vcomp));

                // vacc += scale * (q8 @ q6)
                స్థిరరాశి __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((స్థిరరాశి __m128i *)(b_ptr + స్థానభ్రంశం_scales + k_సమూహం * TILE_N)));
                acc = mm512_కూడు_epi32(acc, _mm512_mullo_epi32(vsum, vscale));
            }
            స్థిరరాశి __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((స్థిరరాశి __m256i *)(b_ptr + స్థానభ్రంశం_d0)));
            vc[నిలువు] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(vd0, vd1), vc[నిలువు]);
        };

        కొరకు (పూర్ణాంకం i = 0; i < KB; ++i) {
            Unroll<నిలువుల_సంఖ్య>{}(compute, i);
        }

        //store to C
        స్వయంచాలక storec = [&](పూర్ణాంకం నిలువు) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + నిలువు * 16), vc[నిలువు]);
        };
        Unroll<నిలువుల_సంఖ్య>{}(storec);
    }
};

మూస <పూర్ణాంకం ఖండం_M, పూర్ణాంకం ఖండం_N, పూర్ణాంకం ఖండం_K>
నిర్మాణం tinygemm_kernel_vnni<ఖండం_q8_K, ఖండం_iq4_xs, దశాంశం, ఖండం_M, ఖండం_N, ఖండం_K> {
    స్థిర శూన్య apply(పూర్ణాంకం KB, స్థిరరాశి శూన్య * RESTRICT _A, స్థిరరాశి శూన్య * RESTRICT _B, దశాంశం * RESTRICT C, పూర్ణాంకం ldc) {

        constexpr పూర్ణాంకం నిలువుల_సంఖ్య = ఖండం_N / 16;
        స్థిరరాశి పూర్ణాంకం TILE_కొలత = TILE_N * పరిమాణం(ఖండం_iq4_xs) + TILE_N * 2;

        స్థిరరాశి ఖండం_q8_K * RESTRICT A = static_cast<స్థిరరాశి ఖండం_q8_K *>(_A);
        స్థిరరాశి అక్షరం * RESTRICT B = static_cast<స్థిరరాశి అక్షరం *>(_B);

        // load the 256 bytes from A to 4 avx512 vectors
        __m512i va[4];
        __m512 vc[నిలువుల_సంఖ్య];
        __m512 vd1;

        // packed_B:
        స్థిరరాశి పూర్ణాంకం స్థానభ్రంశం_scales = (QK_K / 2) * TILE_N ;
        స్థిరరాశి పూర్ణాంకం స్థానభ్రంశం_d0     = (QK_K / 2) * TILE_N + 8 * TILE_N;

        // compensation
        __m512i vcomp;

        స్థిరరాశి __m256i m128s = _mm256_set1_epi16(128);
        స్థిరరాశి __m512i దిగువ_ముసుగు = _mm512_set1_epi8(0xF);

        స్థిరరాశి __m512i values128 = mm512_సమితి_epi8(
            113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
            113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
            113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
            113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127
        );
        స్థిరరాశి __m512i off = _mm512_set1_epi8(static_cast<అక్షరం>(0x80));
        స్థిరరాశి __m512i values256 = mm512_కూడు_epi8(values128, off);

        స్వయంచాలక loadc = [&](స్వయంచాలక నిలువు) {
            vc[నిలువు] = _mm512_setzero_ps();
        };
        Unroll<నిలువుల_సంఖ్య>{}(loadc);

        స్వయంచాలక compute = [&](స్వయంచాలక నిలువు, స్వయంచాలక i) {
            ఒకవేళ constexpr (నిలువు == 0) {
                // load a
                va[0] = _mm512_loadu_si512((స్థిరరాశి __m512i *)(A[0 * KB + i].qs +   0));
                va[1] = _mm512_loadu_si512((స్థిరరాశి __m512i *)(A[0 * KB + i].qs +  64));
                va[2] = _mm512_loadu_si512((స్థిరరాశి __m512i *)(A[0 * KB + i].qs + 128));
                va[3] = _mm512_loadu_si512((స్థిరరాశి __m512i *)(A[0 * KB + i].qs + 192));

                // compensation: 128 * A
                స్థిరరాశి __m256i q8sums = _mm256_loadu_si256((స్థిరరాశి __m256i *)A[0 * KB + i].bsums);
                vcomp = _mm512_castsi256_si512(_mm256_madd_epi16(q8sums, m128s));
                vd1 = _mm512_set1_ps(A[0 * KB + i].d);
            }

            // accmulate the quants
            __m512i acc = _mm512_setzero_si512();
            స్థిరరాశి అక్షరం * b_ptr = B + PACKED_సూచీ(నిలువు, i, KB, TILE_కొలత);
            స్థిరరాశి అక్షరం * b_qs = b_ptr;
            పూర్ణాంకం ముసుగు = 0;
            కొరకు (పూర్ణాంకం k_సమూహం = 0; k_సమూహం < QK_K / 32; ++k_సమూహం) {
                పూర్ణాంకం r = k_సమూహం >> 1;
                __m512i vmask = _mm512_set1_epi32(k_సమూహం);
                __m512i vsum = _mm512_setzero_si512();
                కొరకు (పూర్ణాంకం k = 0; k < 8; k += 2) {
                    __m512i va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(ముసుగు++), va[r]);
                    __m512i va1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(ముసుగు++), va[r]);

                    __m512i బైట్లు = _mm512_loadu_si512(b_qs);
                    __m512i vb0 = _mm512_shuffle_epi8(values256, _mm512_and_si512(బైట్లు, దిగువ_ముసుగు));
                    __m512i vb1 = _mm512_shuffle_epi8(values256, _mm512_and_si512(_mm512_srli_epi16(బైట్లు, 4), దిగువ_ముసుగు));

                    vsum = _mm512_dpbusd_epi32(vsum, vb0, va0);
                    vsum = _mm512_dpbusd_epi32(vsum, vb1, va1);
                    b_qs += 64;
                }
                // (B + 128) * A - 128 * A
                vsum = _mm512_sub_epi32(vsum, _mm512_permutexvar_epi32(vmask, vcomp));

                // vacc += scale * (q8 @ q4)
                స్థిరరాశి __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((స్థిరరాశి __m128i *)(b_ptr + స్థానభ్రంశం_scales + k_సమూహం * TILE_N)));
                acc = mm512_కూడు_epi32(acc, _mm512_mullo_epi32(vsum, vscale));
            }
            స్థిరరాశి __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((స్థిరరాశి __m256i *)(b_ptr + స్థానభ్రంశం_d0)));
            vc[నిలువు] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(vd0, vd1), vc[నిలువు]);
        };

        కొరకు (పూర్ణాంకం i = 0; i < KB; ++i) {
            Unroll<నిలువుల_సంఖ్య>{}(compute, i);
        }

        //store to C
        స్వయంచాలక storec = [&](స్వయంచాలక నిలువు) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + నిలువు * 16), vc[నిలువు]);
        };
        Unroll<నిలువుల_సంఖ్య>{}(storec);
    }
};

#నిర్వచించు LAUNCH_TINYGEMM_KERNEL_VNNI(NB_కొలత)                                                   \
    tinygemm_kernel_vnni<vec_dot_type, type, దశాంశం, 1, NB_కొలత, blck_కొలత>::apply(             \
        KB, wdata_batch,                                                                       \
        (స్థిరరాశి అక్షరం *)src0->దత్తాంశం + src0_స్థానభ్రంశం + PACKED_సూచీ(nb * kTilesN, 0, KB, TILE_కొలత), \
        (దశాంశం *) dst->దత్తాంశం + dst_స్థానభ్రంశం + nb_ప్రారంభం, ldc)

మూస <రకం_పేరు TA, రకం_పేరు TB, రకం_పేరు TC, పూర్ణాంకం ఖండం_K,
          రకం_పేరు std::enable_if<!is_type_qkk<TB>::value, పూర్ణాంకం>::type = 0>
శూన్య tinygemm_kernel_amx(పూర్ణాంకం M, పూర్ణాంకం N, పూర్ణాంకం KB, స్థిరరాశి శూన్య * RESTRICT _A, స్థిరరాశి శూన్య * RESTRICT _B, TC * RESTRICT C, పూర్ణాంకం ldc) {
    ఉపయోగించి packed_B_t = packed_B_type<TB>;
    స్థిరరాశి పూర్ణాంకం TILE_కొలత = get_tile_కొలత<TB>();
    స్థిరరాశి తార్కిక need_unpack = do_unpack<TB>::value;

    GGML_ASSERT(M <= 2 * TILE_M && N == 2 * TILE_N);
    స్థిరరాశి TA * RESTRICT A = static_cast<స్థిరరాశి TA *>(_A);
    స్థిరరాశి అక్షరం * RESTRICT B = static_cast<స్థిరరాశి అక్షరం *>(_B);

    స్థిరరాశి పూర్ణాంకం m0 = std::min(M, TILE_M);
    స్థిరరాశి పూర్ణాంకం m1 = std::max(M - TILE_M, 0);
    స్థిరరాశి పూర్ణాంకం lda = KB * పరిమాణం(TA);
    //const int ldb = KB * sizeof(TB);

    alignas(64) స్థిర thread_local packed_B_t Tile0[TILE_N * TILE_K];
    alignas(64) స్థిర thread_local packed_B_t Tile1[TILE_N * TILE_K];
    alignas(64) స్థిర thread_local int8_t Tile23[TILE_M * TILE_K];

    alignas(64) స్థిర thread_local int32_t TileC0[TILE_M * TILE_N * 4];
    alignas(64) స్థిర thread_local int32_t TileC1[TILE_M * TILE_N * 4];

    // double buffering C to interleave avx512 and amx
    int32_t * C_cur = TileC0;
    int32_t * C_pre = TileC1;

    స్వయంచాలక Tile4 = [&](int32_t * పునాది) { తిరిగి పునాది; };
    స్వయంచాలక Tile5 = [&](int32_t * పునాది) { తిరిగి పునాది + TILE_M * TILE_N; };
    స్వయంచాలక Tile6 = [&](int32_t * పునాది) { తిరిగి పునాది + 2 * TILE_M * TILE_N; };
    స్వయంచాలక Tile7 = [&](int32_t * పునాది) { తిరిగి పునాది + 3 * TILE_M * TILE_N; };

    ఒకవేళ (M == 2 * TILE_M) {
        // i = 0
        స్థిరరాశి అక్షరం * B_blk0 = B + PACKED_సూచీ(0, 0, KB, TILE_కొలత);
        స్థిరరాశి అక్షరం * B_blk1 = B + PACKED_సూచీ(1, 0, KB, TILE_కొలత);
        ఒకవేళ (need_unpack) {
            unpack_B<TB>(Tile0, B_blk0);
            _tile_loadd(TMM0, Tile0, TILE_N * VNNI_BLK);
        } లేకపోతే {
            _tile_loadd(TMM0, B_blk0, TILE_N * VNNI_BLK);
        }

        _tile_zero(TMM4);
        _tile_loadd(TMM2, A[0].qs, lda);
        _tile_dpbssd(TMM4, TMM2, TMM0);
        _tile_stored(TMM4, Tile4(C_pre), TILE_N * పరిమాణం(int32_t));

        _tile_zero(TMM5);
        _tile_loadd(TMM3, A[TILE_M * KB + 0].qs, lda);
        _tile_dpbssd(TMM5, TMM3, TMM0);
        _tile_stored(TMM5, Tile5(C_pre), TILE_N * పరిమాణం(int32_t));

        ఒకవేళ (need_unpack) {
            unpack_B<TB>(Tile1, B_blk1);
            _tile_loadd(TMM1, Tile1, TILE_N * VNNI_BLK);
        } లేకపోతే {
            _tile_loadd(TMM1, B_blk1, TILE_N * VNNI_BLK);
        }

        _tile_zero(TMM6);
        _tile_dpbssd(TMM6, TMM2, TMM1);
        _tile_stored(TMM6, Tile6(C_pre), TILE_N * పరిమాణం(int32_t));

        _tile_zero(TMM7);
        _tile_dpbssd(TMM7, TMM3, TMM1);
        _tile_stored(TMM7, Tile7(C_pre), TILE_N * పరిమాణం(int32_t));

        కొరకు (పూర్ణాంకం i = 1; i < KB; ++i) {
            // index of previous iter
            స్థిరరాశి పూర్ణాంకం ii = i - 1;
            స్థిరరాశి అక్షరం * B_blk0 = B + PACKED_సూచీ(0, i, KB, TILE_కొలత);
            స్థిరరాశి అక్షరం * B_blk1 = B + PACKED_సూచీ(1, i, KB, TILE_కొలత);
            GGML_DISPATCH_BOOL(ii > 0, is_acc, [&] {
                ఒకవేళ (need_unpack) {
                    unpack_B<TB>(Tile0, B_blk0);
                    _tile_loadd(TMM0, Tile0, TILE_N * VNNI_BLK);
                } లేకపోతే {
                    _tile_loadd(TMM0, B_blk0, TILE_N * VNNI_BLK);
                }
                _tile_zero(TMM4);
                _tile_loadd(TMM2, A[i].qs, lda);
                acc_C<TA, TB, is_acc>::apply(C, ldc, Tile4(C_pre), &A[ii], KB, B + PACKED_సూచీ(0, ii, KB, TILE_కొలత), TILE_M);

                _tile_dpbssd(TMM4, TMM2, TMM0);
                _tile_stored(TMM4, Tile4(C_cur), TILE_N * పరిమాణం(int32_t));

                _tile_zero(TMM5);
                _tile_loadd(TMM3, A[TILE_M * KB + i].qs, lda);
                acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc, ldc, Tile5(C_pre), &A[TILE_M * KB + ii], KB, B + PACKED_సూచీ(0, ii, KB, TILE_కొలత), TILE_M);

                _tile_dpbssd(TMM5, TMM3, TMM0);
                _tile_stored(TMM5, Tile5(C_cur), TILE_N * పరిమాణం(int32_t));

                ఒకవేళ (need_unpack) {
                    unpack_B<TB>(Tile1, B_blk1);
                    _tile_loadd(TMM1, Tile1, TILE_N * VNNI_BLK);
                } లేకపోతే {
                    _tile_loadd(TMM1, B_blk1, TILE_N * VNNI_BLK);
                }
                _tile_zero(TMM6);
                acc_C<TA, TB, is_acc>::apply(C + TILE_N, ldc, Tile6(C_pre), &A[ii], KB, B + PACKED_సూచీ(1, ii, KB, TILE_కొలత), TILE_M);

                _tile_dpbssd(TMM6, TMM2, TMM1);
                _tile_stored(TMM6, Tile6(C_cur), TILE_N * పరిమాణం(int32_t));

                _tile_zero(TMM7);
                acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc + TILE_N, ldc, Tile7(C_pre), &A[TILE_M * KB + ii], KB, B + PACKED_సూచీ(1, ii, KB, TILE_కొలత), TILE_M);

                _tile_dpbssd(TMM7, TMM3, TMM1);
                _tile_stored(TMM7, Tile7(C_cur), TILE_N * పరిమాణం(int32_t));

                std::swap(C_cur, C_pre);
            });
        }
        // final accumulation
        {
            పూర్ణాంకం ii = KB - 1;
            acc_C<TA, TB, నిజం>::apply(C, ldc, Tile4(C_pre), &A[ii], KB, B + PACKED_సూచీ(0, ii, KB, TILE_కొలత), TILE_M);
            acc_C<TA, TB, నిజం>::apply(C + TILE_M * ldc, ldc, Tile5(C_pre), &A[TILE_M * KB + ii], KB, B + PACKED_సూచీ(0, ii, KB, TILE_కొలత), TILE_M);
            acc_C<TA, TB, నిజం>::apply(C + TILE_N, ldc, Tile6(C_pre), &A[ii], KB, B + PACKED_సూచీ(1, ii, KB, TILE_కొలత), TILE_M);
            acc_C<TA, TB, నిజం>::apply(C + TILE_M * ldc + TILE_N, ldc, Tile7(C_pre), &A[TILE_M * KB + ii], KB, B + PACKED_సూచీ(1, ii, KB, TILE_కొలత), TILE_M);
        }
    } లేకపోతే {
        కొరకు (పూర్ణాంకం i = 0; i < KB; ++i) {
            _tile_zero(TMM4);
            _tile_zero(TMM6);
            ఒకవేళ (m1 != 0) {
                _tile_zero(TMM5);
                _tile_zero(TMM7);
            }

            స్థిరరాశి అక్షరం * B_blk0 = B + PACKED_సూచీ(0, i, KB, TILE_కొలత);
            స్థిరరాశి అక్షరం * B_blk1 = B + PACKED_సూచీ(1, i, KB, TILE_కొలత);
            ఒకవేళ (need_unpack) {
                unpack_B<TB>(Tile0, B_blk0);
                _tile_loadd(TMM0, Tile0, TILE_N * VNNI_BLK);
            } లేకపోతే {
                _tile_loadd(TMM0, B_blk0, TILE_N * VNNI_BLK);
            }

            ఒకవేళ (need_unpack) {
                unpack_B<TB>(Tile1, B_blk1);
                _tile_loadd(TMM1, Tile1, TILE_N * VNNI_BLK);
            } లేకపోతే {
                _tile_loadd(TMM1, B_blk1, TILE_N * VNNI_BLK);
            }

            ఒకవేళ (m0 == TILE_M) {
                _tile_loadd(TMM2, A[i].qs, lda);
            } లేకపోతే {
                unpack_A(Tile23, &A[i], KB, m0);
                _tile_loadd(TMM2, Tile23, TILE_K);
            }

            _tile_dpbssd(TMM4, TMM2, TMM0);
            _tile_dpbssd(TMM6, TMM2, TMM1);

            _tile_stored(TMM4, Tile4(C_cur), TILE_N * పరిమాణం(int32_t));
            _tile_stored(TMM6, Tile6(C_cur), TILE_N * పరిమాణం(int32_t));

            GGML_DISPATCH_BOOL(i > 0, is_acc, [&] {
                acc_C<TA, TB, is_acc>::apply(C,          ldc, Tile4(C_cur), &A[i], KB, B + PACKED_సూచీ(0, i, KB, TILE_కొలత), m0);
                acc_C<TA, TB, is_acc>::apply(C + TILE_N, ldc, Tile6(C_cur), &A[i], KB, B + PACKED_సూచీ(1, i, KB, TILE_కొలత), m0);
            });

            ఒకవేళ (m1 != 0) {
                unpack_A(Tile23, &A[TILE_M * KB + i], KB, m1);
                _tile_loadd(TMM3, Tile23, TILE_K);

                _tile_dpbssd(TMM5, TMM3, TMM0);
                _tile_dpbssd(TMM7, TMM3, TMM1);
                _tile_stored(TMM5, Tile5(C_cur), TILE_N * పరిమాణం(int32_t));
                _tile_stored(TMM7, Tile7(C_cur), TILE_N * పరిమాణం(int32_t));
                GGML_DISPATCH_BOOL(i > 0, is_acc, [&] {
                    acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc,          ldc, Tile5(C_cur), &A[TILE_M * KB + i], KB, B + PACKED_సూచీ(0, i, KB, TILE_కొలత), m1);
                    acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc + TILE_N, ldc, Tile7(C_cur), &A[TILE_M * KB + i], KB, B + PACKED_సూచీ(1, i, KB, TILE_కొలత), m1);
                });
            }
        }
    }
    తిరిగి;
}

మూస <రకం_పేరు TA, రకం_పేరు TB, రకం_పేరు TC, పూర్ణాంకం ఖండం_K,
          రకం_పేరు std::enable_if<is_type_qkk<TB>::value, పూర్ణాంకం>::type = 0>
శూన్య tinygemm_kernel_amx(పూర్ణాంకం M, పూర్ణాంకం N, పూర్ణాంకం KB, స్థిరరాశి శూన్య * RESTRICT _A, స్థిరరాశి శూన్య * RESTRICT _B, దశాంశం * RESTRICT C, పూర్ణాంకం ldc) {
    static_assert(std::is_same<TA, ఖండం_q8_K>::value);
    స్థిరరాశి పూర్ణాంకం TILE_కొలత = get_tile_కొలత<TB>();

    GGML_ASSERT(M <= 2 * TILE_M && N == 2 * TILE_N);
    స్థిరరాశి TA * RESTRICT A = static_cast<స్థిరరాశి TA *>(_A);
    స్థిరరాశి అక్షరం * RESTRICT B = static_cast<స్థిరరాశి అక్షరం *>(_B);

    స్థిరరాశి పూర్ణాంకం m0 = std::min(M, TILE_M);
    స్థిరరాశి పూర్ణాంకం m1 = std::max(M - TILE_M, 0);
    //const int lda = KB * sizeof(TA);

    alignas(64) స్థిర thread_local int8_t Tile0[TILE_N * TILE_K];
    alignas(64) స్థిర thread_local int8_t Tile1[TILE_N * TILE_K];
    alignas(64) స్థిర thread_local int8_t Tile23[TILE_M * TILE_K];

    // mat mul result for each group
    alignas(64) స్థిర thread_local int32_t Tile4[TILE_M * TILE_N];
    alignas(64) స్థిర thread_local int32_t Tile5[TILE_M * TILE_N];
    alignas(64) స్థిర thread_local int32_t Tile6[TILE_M * TILE_N];
    alignas(64) స్థిర thread_local int32_t Tile7[TILE_M * TILE_N];

    // sum of each QK_K block, contains 8 groups, int32
    alignas(64) స్థిర thread_local int32_t Sumi4[TILE_M * TILE_N];
    alignas(64) స్థిర thread_local int32_t Sumi5[TILE_M * TILE_N];
    alignas(64) స్థిర thread_local int32_t Sumi6[TILE_M * TILE_N];
    alignas(64) స్థిర thread_local int32_t Sumi7[TILE_M * TILE_N];

    స్థిరరాశి పూర్ణాంకం k_సమూహం_కొలత = std::is_same<TB, ఖండం_q6_K>::value ? 16 : 32;
    కొరకు (పూర్ణాంకం i = 0; i < KB; ++i) {
        // step 1: accumulate the quants across 8 groups, each group with 32
        కొరకు (పూర్ణాంకం k = 0; k < QK_K / k_సమూహం_కొలత; ++k) {
            GGML_DISPATCH_BOOL(k > 0, is_acc, [&] {
                _tile_zero(TMM4);
                _tile_zero(TMM6);

                unpack_B<TB>(Tile0, B + PACKED_సూచీ(0, i, KB, TILE_కొలత), k);
                _tile_loadd(TMM0, Tile0, TILE_N * VNNI_BLK);

                unpack_B<TB>(Tile1, B + PACKED_సూచీ(1, i, KB, TILE_కొలత), k);
                _tile_loadd(TMM1, Tile1, TILE_N * VNNI_BLK);

                unpack_A<TB>(Tile23, &A[i], KB, k, m0);
                _tile_loadd(TMM2, Tile23, TILE_K);

                _tile_dpbssd(TMM4, TMM2, TMM0);
                _tile_dpbssd(TMM6, TMM2, TMM1);

                _tile_stored(TMM4, Tile4, TILE_N * పరిమాణం(int32_t));
                _tile_stored(TMM6, Tile6, TILE_N * పరిమాణం(int32_t));

                scale_C<TB, is_acc>(Tile4, Sumi4, B + PACKED_సూచీ(0, i, KB, TILE_కొలత), k, m0);
                scale_C<TB, is_acc>(Tile6, Sumi6, B + PACKED_సూచీ(1, i, KB, TILE_కొలత), k, m0);

                ఒకవేళ (m1 != 0) {
                    _tile_zero(TMM5);
                    _tile_zero(TMM7);

                    unpack_A<TB>(Tile23, &A[TILE_M * KB + i], KB, k, m1);
                    _tile_loadd(TMM3, Tile23, TILE_K);

                    _tile_dpbssd(TMM5, TMM3, TMM0);
                    _tile_dpbssd(TMM7, TMM3, TMM1);

                    _tile_stored(TMM5, Tile5, TILE_N * పరిమాణం(int32_t));
                    _tile_stored(TMM7, Tile7, TILE_N * పరిమాణం(int32_t));

                    scale_C<TB, is_acc>(Tile5, Sumi5, B + PACKED_సూచీ(0, i, KB, TILE_కొలత), k, m1);
                    scale_C<TB, is_acc>(Tile7, Sumi7, B + PACKED_సూచీ(1, i, KB, TILE_కొలత), k, m1);
                }
            });
        }

        // step 2: accmulate the mins
        GGML_DISPATCH_BOOL(i > 0, is_acc, [&] {
            acc_C<TA, TB, is_acc>::apply(C,          ldc, Sumi4, &A[i], KB, B + PACKED_సూచీ(0, i, KB, TILE_కొలత), m0);
            acc_C<TA, TB, is_acc>::apply(C + TILE_N, ldc, Sumi6, &A[i], KB, B + PACKED_సూచీ(1, i, KB, TILE_కొలత), m0);
            ఒకవేళ (m1 != 0) {
                acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc,          ldc, Sumi5, &A[TILE_M * KB + i], KB, B + PACKED_సూచీ(0, i, KB, TILE_కొలత), m1);
                acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc + TILE_N, ldc, Sumi7, &A[TILE_M * KB + i], KB, B + PACKED_సూచీ(1, i, KB, TILE_కొలత), m1);
            }
        });
    }
    తిరిగి;
}

} // anonymous namespace

// get the packed tensor size for quantized weights
size_t ggml_backend_amx_get_alloc_కొలత(స్థిరరాశి నిర్మాణం ggml_tensor * tensor) {
    స్థిరరాశి గణన ggml_type TYPE = tensor->type;

    స్థిరరాశి పూర్ణాంకం K = tensor->ne[0]; // ne0: in_features
    స్థిరరాశి పూర్ణాంకం N = tensor->ne[1]; // ne1: out_features

    స్వయంచాలక get_tensor_కొలత = [&] {
        size_t అడ్డువరుస_కొలత_B{0};
        GGML_DISPATCH_QTYPES(TYPE, [&] {
            అడ్డువరుస_కొలత_B = get_అడ్డువరుస_కొలత<type, blck_కొలత>(K);
        });
        తిరిగి N * అడ్డువరుస_కొలత_B;
    };

    ఒకవేళ (qtype_has_amx_kernels(TYPE)) {
        తిరిగి get_tensor_కొలత();
    } లేకపోతే {
        // for f16, bf16 we don't do packing
        తిరిగి ggml_nbytes(tensor);
    }
}

// pack weight to vnni format
శూన్య ggml_backend_amx_convert_బరువు(నిర్మాణం ggml_tensor * tensor, స్థిరరాశి శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size) {
    GGML_ASSERT(స్థానభ్రంశం == 0 && size == ggml_nbytes(tensor)); // only full tensor conversion is supported for now

    స్థిరరాశి గణన ggml_type TYPE = tensor->type;

    స్థిరరాశి పూర్ణాంకం K = tensor->ne[0]; // ne0: in_features
    స్థిరరాశి పూర్ణాంకం N = tensor->ne[1]; // ne1: out_features

    GGML_DISPATCH_QTYPES(TYPE, [&] {
        convert_B_packed_format<type, blck_కొలత>((శూన్య *)((అక్షరం *)tensor->దత్తాంశం + స్థానభ్రంశం), (స్థిరరాశి type *)దత్తాంశం, N, K);
    });
}

// ne2 is passed explicitly to help compiler optimize repeated calls
అంతఃవరుస int64_t ggml_batch_స్థానభ్రంశం(స్థిరరాశి ggml_tensor * t, int64_t batch_idx, int64_t ne2) {
    స్థిరరాశి int64_t i2 = batch_idx % ne2;
    స్థిరరాశి int64_t i3 = batch_idx / ne2;
    తిరిగి i3 * t->nb[3] + i2 * t->nb[2];
}

size_t ggml_backend_amx_desired_wsize(స్థిరరాశి నిర్మాణం ggml_tensor * dst) {
    నిర్మాణం ggml_tensor * src0 = dst->src[0];

    స్థిరరాశి గణన ggml_type TYPE = src0->type;

    స్థిరరాశి తార్కిక is_floating_type = TYPE == GGML_TYPE_F16;
    ఒకవేళ (is_floating_type) {
        తిరిగి 0;
    }

    స్థిరరాశి పూర్ణాంకం M = dst->ne[1];
    స్థిరరాశి పూర్ణాంకం K = src0->ne[0];
    స్థిరరాశి int64_t n_batch = dst->ne[2] * dst->ne[3];

    size_t desired_wsize = 0;

    GGML_DISPATCH_QTYPES(TYPE, [&] {
        స్థిరరాశి size_t అడ్డువరుస_కొలత_A = K / blck_కొలత * పరిమాణం(vec_dot_type);
        desired_wsize = n_batch * M * అడ్డువరుస_కొలత_A;
    });

    తిరిగి desired_wsize;
}

// NB: mixed dtype gemm with Advanced Matrix Extensions (Intel AMX)
//
// src0: weight in shape of {N, K}, quantized
// src1: input  in shape of {M, K}, float32
// dst:  output in shape of {M, N}, float32
//
// the function performs: dst = src1 @ src0.T for each batch
//
శూన్య ggml_backend_amx_mul_mat(స్థిరరాశి ggml_compute_params * params, నిర్మాణం ggml_tensor * dst) {
    నిర్మాణం ggml_tensor * src0 = dst->src[0];
    నిర్మాణం ggml_tensor * src1 = dst->src[1];

    స్థిరరాశి గణన ggml_type TYPE = src0->type;

    // f16 only has avx512 kernels for now,
    // amx kernels will be added once 6th gen xeon is released.
    స్థిరరాశి తార్కిక is_floating_type = TYPE == GGML_TYPE_F16;

    స్థిరరాశి పూర్ణాంకం M = dst->ne[1];
    స్థిరరాశి పూర్ణాంకం N = dst->ne[0];
    స్థిరరాశి పూర్ణాంకం K = src0->ne[0];
    స్థిరరాశి పూర్ణాంకం ldc = dst->nb[1] / dst->nb[0];

    స్థిరరాశి int64_t ne2 = dst->ne[2];
    స్థిరరాశి int64_t n_batch = ne2 * dst->ne[3];

    ఒకవేళ (is_floating_type) {
        constexpr పూర్ణాంకం ఖండం_M = 4;
        constexpr పూర్ణాంకం ఖండం_N = 6;
        స్థిరరాశి పూర్ణాంకం MB = div_పైకి(M, ఖండం_M);
        స్థిరరాశి పూర్ణాంకం NB = div_పైకి(N, ఖండం_N);

        parallel_for_ggml(params, n_batch * MB * NB, [&](పూర్ణాంకం begin, పూర్ణాంకం end) {
            GGML_DISPATCH_FLOATING_TYPES(TYPE, [&] {
                కొరకు (పూర్ణాంకం i = begin; i < end; ++i) {
                    పూర్ణాంకం batch_idx = i / (MB * NB);
                    పూర్ణాంకం మిగిలిన = i % (MB * NB);
                    పూర్ణాంకం mb = మిగిలిన / NB;
                    పూర్ణాంకం nb = మిగిలిన % NB;

                    int64_t src0_స్థానభ్రంశం = ggml_batch_స్థానభ్రంశం(src0, batch_idx, ne2);
                    int64_t src1_స్థానభ్రంశం = ggml_batch_స్థానభ్రంశం(src1, batch_idx, ne2);
                    int64_t dst_స్థానభ్రంశం  = ggml_batch_స్థానభ్రంశం(dst,  batch_idx, ne2);

                    పూర్ణాంకం mb_ప్రారంభం = mb * ఖండం_M;
                    పూర్ణాంకం mb_కొలత = std::min(ఖండం_M, M - mb_ప్రారంభం);
                    పూర్ణాంకం nb_ప్రారంభం = nb * ఖండం_N;
                    పూర్ణాంకం nb_కొలత = std::min(ఖండం_N, N - nb_ప్రారంభం);

                    ఎంపిక (mb_కొలత << 4 | nb_కొలత) {
                        స్థితి 0x12: LAUNCH_TINYGEMM_KERNEL_AVX(1, 2); ఆపు;
                        స్థితి 0x14: LAUNCH_TINYGEMM_KERNEL_AVX(1, 4); ఆపు;
                        స్థితి 0x16: LAUNCH_TINYGEMM_KERNEL_AVX(1, 6); ఆపు;
                        స్థితి 0x22: LAUNCH_TINYGEMM_KERNEL_AVX(2, 2); ఆపు;
                        స్థితి 0x24: LAUNCH_TINYGEMM_KERNEL_AVX(2, 4); ఆపు;
                        స్థితి 0x26: LAUNCH_TINYGEMM_KERNEL_AVX(2, 6); ఆపు;
                        స్థితి 0x32: LAUNCH_TINYGEMM_KERNEL_AVX(3, 2); ఆపు;
                        స్థితి 0x34: LAUNCH_TINYGEMM_KERNEL_AVX(3, 4); ఆపు;
                        స్థితి 0x36: LAUNCH_TINYGEMM_KERNEL_AVX(3, 6); ఆపు;
                        స్థితి 0x42: LAUNCH_TINYGEMM_KERNEL_AVX(4, 2); ఆపు;
                        స్థితి 0x44: LAUNCH_TINYGEMM_KERNEL_AVX(4, 4); ఆపు;
                        స్థితి 0x46: LAUNCH_TINYGEMM_KERNEL_AVX(4, 6); ఆపు;
                        అప్రమేయ: fprintf(stderr, "Unexpected block size!\n");
                    }
                }
            });
        });
        తిరిగి;
    }

    // pointer to work space, used convert A from float to quantized type
    శూన్య * wdata = params->wdata;

    //TODO: performance improvement: merge quant A
 // if (params->ith == 0) {
        GGML_DISPATCH_QTYPES(TYPE, [&] {
            స్థిరరాశి size_t అడ్డువరుస_కొలత_A = K / blck_కొలత * పరిమాణం(vec_dot_type);
            స్థిరరాశి size_t desired_wsize = n_batch * M * అడ్డువరుస_కొలత_A;
            ఒకవేళ (params->wsize < desired_wsize) {
                GGML_ABORT("insufficient work space size");
            }

            // Q4_0, Q4_1, Q8_0 handles 1 TILE_K per blck_size
            // Q4_K, Q5_K, Q6_K, IQ4_XS handles 8 TILE_K per blck_size
            GGML_ASSERT(TILE_K == blck_కొలత || TILE_K * 8 == blck_కొలత);

            parallel_for_ggml(params, n_batch * M, [&](పూర్ణాంకం begin, పూర్ణాంకం end) {
                కొరకు (పూర్ణాంకం idx = begin; idx < end; ++idx) {
                    పూర్ణాంకం batch_idx = idx / M;
                    పూర్ణాంకం m         = idx % M;
                    int64_t src1_స్థానభ్రంశం = ggml_batch_స్థానభ్రంశం(src1, batch_idx, ne2);
                    స్థిరరాశి దశాంశం * A_దత్తాంశం = (స్థిరరాశి దశాంశం *)((స్థిరరాశి అక్షరం *)src1->దత్తాంశం + src1_స్థానభ్రంశం);
                    అక్షరం * wdata_batch = (అక్షరం *)wdata + batch_idx * M * అడ్డువరుస_కొలత_A;
                    from_float<vec_dot_type>(A_దత్తాంశం + m * K, wdata_batch + m * అడ్డువరుస_కొలత_A, K);
                }
            });
        });
 // }

    ggml_barrier(params->threadpool);

    ఒకవేళ (M == 1) {
        // MB = 1 and handle 8 tiles in each block
        constexpr పూర్ణాంకం kTilesN = 4;
        constexpr పూర్ణాంకం ఖండం_N = TILE_N * kTilesN;
        స్థిరరాశి పూర్ణాంకం NB = div_పైకి(N, ఖండం_N);

        parallel_for_ggml(params, n_batch * NB, [&](పూర్ణాంకం begin, పూర్ణాంకం end) {
            GGML_DISPATCH_QTYPES(TYPE, [&] {
                స్థిరరాశి పూర్ణాంకం KB = K / blck_కొలత;
                స్థిరరాశి పూర్ణాంకం TILE_కొలత = get_tile_కొలత<type>();
                స్థిరరాశి పూర్ణాంకం అడ్డువరుస_కొలత_A = KB * పరిమాణం(vec_dot_type);
                కొరకు (పూర్ణాంకం i = begin; i < end; ++i) {
                    పూర్ణాంకం batch_idx = i / NB;
                    పూర్ణాంకం nb = i % NB;

                    int64_t src0_స్థానభ్రంశం = ggml_batch_స్థానభ్రంశం(src0, batch_idx, ne2);
                    int64_t dst_స్థానభ్రంశం  = ggml_batch_స్థానభ్రంశం(dst,  batch_idx, ne2);
                    స్థిరరాశి అక్షరం * wdata_batch = (స్థిరరాశి అక్షరం *)wdata + batch_idx * అడ్డువరుస_కొలత_A;

                    పూర్ణాంకం nb_ప్రారంభం = nb * ఖండం_N;
                    పూర్ణాంకం nb_కొలత = std::min(ఖండం_N, N - nb_ప్రారంభం); // 32, 64, 96

                    ఎంపిక (nb_కొలత) {
                        //case 160: LAUNCH_TINYGEMM_KERNEL_VNNI(160); break;
                        స్థితి 128: LAUNCH_TINYGEMM_KERNEL_VNNI(128); ఆపు;
                        స్థితి 96: LAUNCH_TINYGEMM_KERNEL_VNNI(96); ఆపు;
                        స్థితి 64: LAUNCH_TINYGEMM_KERNEL_VNNI(64); ఆపు;
                        స్థితి 32: LAUNCH_TINYGEMM_KERNEL_VNNI(32); ఆపు;
                        అప్రమేయ: fprintf(stderr, "Unexpected n block size!\n");
                    }
                }
            });
        });
        తిరిగి;
    }

    // handle 4 tiles at a tile
    constexpr పూర్ణాంకం ఖండం_M = TILE_M * 2;
    constexpr పూర్ణాంకం ఖండం_N = TILE_N * 2;
    స్థిరరాశి పూర్ణాంకం MB = div_పైకి(M, ఖండం_M);
    స్థిరరాశి పూర్ణాంకం NB = div_పైకి(N, ఖండం_N);

    parallel_for_ggml(params, n_batch * MB * NB, [&](పూర్ణాంకం begin, పూర్ణాంకం end) {
        // init tile config for each thread
        ggml_tile_config_init();

        GGML_DISPATCH_QTYPES(TYPE, [&] {
            స్థిరరాశి పూర్ణాంకం KB = K / blck_కొలత;
            స్థిరరాశి పూర్ణాంకం TILE_కొలత = get_tile_కొలత<type>();
            స్థిరరాశి పూర్ణాంకం అడ్డువరుస_కొలత_A = KB * పరిమాణం(vec_dot_type);

            కొరకు (పూర్ణాంకం i = begin; i < end; ++i) {
                పూర్ణాంకం batch_idx = i / (MB * NB);
                పూర్ణాంకం మిగిలిన = i % (MB * NB);
                పూర్ణాంకం mb = మిగిలిన / NB;
                పూర్ణాంకం nb = మిగిలిన % NB;

                int64_t src0_స్థానభ్రంశం = ggml_batch_స్థానభ్రంశం(src0, batch_idx, ne2);
                int64_t dst_స్థానభ్రంశం  = ggml_batch_స్థానభ్రంశం(dst,  batch_idx, ne2);
                స్థిరరాశి అక్షరం * wdata_batch = (స్థిరరాశి అక్షరం *)wdata + batch_idx * M * అడ్డువరుస_కొలత_A;

                పూర్ణాంకం mb_ప్రారంభం = mb * ఖండం_M;
                పూర్ణాంకం mb_కొలత = std::min(ఖండం_M, M - mb_ప్రారంభం);
                పూర్ణాంకం nb_ప్రారంభం = nb * ఖండం_N;
                పూర్ణాంకం nb_కొలత = ఖండం_N;

                tinygemm_kernel_amx<vec_dot_type, type, దశాంశం, blck_కొలత>(
                    mb_కొలత, nb_కొలత, KB,
                    wdata_batch + mb_ప్రారంభం * అడ్డువరుస_కొలత_A,
                    (స్థిరరాశి అక్షరం *)src0->దత్తాంశం + src0_స్థానభ్రంశం + PACKED_సూచీ(nb * 2, 0, KB, TILE_కొలత),
                    (దశాంశం *) dst->దత్తాంశం + dst_స్థానభ్రంశం + mb_ప్రారంభం * N + nb_ప్రారంభం, ldc);
            }
        });
    });
}

#అయితే_ముగింపు // if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
