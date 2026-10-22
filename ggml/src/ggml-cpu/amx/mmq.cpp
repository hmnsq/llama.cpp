#यदि defined(__GNUC__)
#निर्देशः GCC diagnostic ignored "-Wpedantic"
#निर्देशः GCC diagnostic ignored "-Wunused-local-typedefs"
#यदि_अन्तः

#अन्तर्भाव "amx.h"
#अन्तर्भाव "mmq.h"
#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "ggml-cpu-impl.h"
#अन्तर्भाव "simd-mappings.h"
#अन्तर्भाव "quants.h"
#अन्तर्भाव "ggml-quants.h"
#अन्तर्भाव <algorithm>
#अन्तर्भाव <type_traits>

#यदि defined(__gnu_linux__)
#अन्तर्भाव <sys/syscall.h>
#अन्तर्भाव <unistd.h>
#यदि_अन्तः

#यदि (defined(_WIN32) || defined(_WIN64))
#परिभाषय RESTRICT __restrict
#अन्यथा
#परिभाषय RESTRICT __restrict__
#यदि_अन्तः

#यदि (defined(_WIN32) || defined(_WIN64))
#परिभाषय ALWAYS_INLINE __forceinline
#नोचेद्यदि __has_attribute(always_inline) || defined(__GNUC__)
#परिभाषय ALWAYS_INLINE __attribute__((__always_inline__)) अन्तःस्थ
#अन्यथा
#परिभाषय ALWAYS_INLINE अन्तःस्थ
#यदि_अन्तः

#यदि defined(__AMX_INT8__) && defined(__AVX512VNNI__)

नामाकाश {

// Forced unrolling
प्रतिरूपम् <पूर्णाङ्क n>
संरचना Unroll {
    प्रतिरूपम् <प्रकारनाम Func, प्रकारनाम... तर्काः>
    ALWAYS_INLINE शून्य संक्रियकः()(ध्रुव Func& f, तर्काः... तर्काः_2) ध्रुव {
        Unroll<n - 1>{}(f, तर्काः_2...);
        f(std::integral_constant<पूर्णाङ्क, n - 1>{}, तर्काः_2...);
    }
};

प्रतिरूपम् <>
संरचना Unroll<1> {
    प्रतिरूपम् <प्रकारनाम Func, प्रकारनाम... तर्काः>
    ALWAYS_INLINE शून्य संक्रियकः()(ध्रुव Func& f, तर्काः... तर्काः_2) ध्रुव {
        f(std::integral_constant<पूर्णाङ्क, 0>{}, तर्काः_2...);
    }
};

// type traits
प्रतिरूपम् <प्रकारनाम T> संरचना PackedTypes {};
प्रतिरूपम् <> संरचना PackedTypes<खण्डम्_q4_0> { उपयुज्य type = int8_t; };
प्रतिरूपम् <> संरचना PackedTypes<खण्डम्_q4_1> { उपयुज्य type = uint8_t; };
प्रतिरूपम् <> संरचना PackedTypes<खण्डम्_q8_0> { उपयुज्य type = int8_t; };
प्रतिरूपम् <प्रकारनाम T> उपयुज्य packed_B_type = प्रकारनाम PackedTypes<T>::type;

प्रतिरूपम् <प्रकारनाम T>
संरचना do_compensate : std::integral_constant<तार्किक,
    std::is_same<T, खण्डम्_q8_0>::value> {};

प्रतिरूपम् <प्रकारनाम T>
संरचना do_unpack : std::integral_constant<तार्किक,
    std::is_same<T, खण्डम्_q4_0>::value ||
    std::is_same<T, खण्डम्_q4_1>::value> {};

प्रतिरूपम् <प्रकारनाम T>
संरचना is_type_qkk : std::integral_constant<तार्किक,
    std::is_same<T, खण्डम्_q4_K>::value ||
    std::is_same<T, खण्डम्_q5_K>::value ||
    std::is_same<T, खण्डम्_q6_K>::value ||
    std::is_same<T, खण्डम्_iq4_xs>::value> {};

#परिभाषय GGML_DISPATCH_FLOATING_TYPES(TYPE, ...)                                        \
    [&] {                                                                              \
        विकल्प (TYPE) {                                                                \
            पक्ष GGML_TYPE_F16: {                                                      \
                उपयुज्य type = ggml_fp16_t;                                              \
                constexpr पूर्णाङ्क blck_आकार = 16;                                          \
                प्रत्यागम __VA_ARGS__();                                                  \
            }                                                                          \
            पक्ष GGML_TYPE_BF16: {                                                     \
                उपयुज्य type = ggml_bf16_t;                                              \
                constexpr पूर्णाङ्क blck_आकार = 32;                                          \
                प्रत्यागम __VA_ARGS__();                                                  \
            }                                                                          \
            स्वभाव:                                                                   \
                fprintf(stderr, "Unsupported floating data type\n");                   \
        }                                                                              \
    }()

#परिभाषय GGML_DISPATCH_QTYPES(QT, ...)                                                  \
    [&] {                                                                              \
        विकल्प (QT) {                                                                  \
            पक्ष GGML_TYPE_Q4_0: {                                                     \
                उपयुज्य type = खण्डम्_q4_0;                                               \
                उपयुज्य vec_dot_type = खण्डम्_q8_0;                                       \
                constexpr पूर्णाङ्क blck_आकार = QK4_0;                                       \
                प्रत्यागम __VA_ARGS__();                                                  \
            }                                                                          \
            पक्ष GGML_TYPE_Q4_1: {                                                     \
                उपयुज्य type = खण्डम्_q4_1;                                               \
                उपयुज्य vec_dot_type = खण्डम्_q8_1;                                       \
                constexpr पूर्णाङ्क blck_आकार = QK4_1;                                       \
                प्रत्यागम __VA_ARGS__();                                                  \
            }                                                                          \
            पक्ष GGML_TYPE_Q8_0: {                                                     \
                उपयुज्य type = खण्डम्_q8_0;                                               \
                उपयुज्य vec_dot_type = खण्डम्_q8_0;                                       \
                constexpr पूर्णाङ्क blck_आकार = QK8_0;                                       \
                प्रत्यागम __VA_ARGS__();                                                  \
            }                                                                          \
            पक्ष GGML_TYPE_Q4_K: {                                                     \
                उपयुज्य type = खण्डम्_q4_K;                                               \
                उपयुज्य vec_dot_type = खण्डम्_q8_K;                                       \
                constexpr पूर्णाङ्क blck_आकार = QK_K;                                        \
                प्रत्यागम __VA_ARGS__();                                                  \
            }                                                                          \
            पक्ष GGML_TYPE_Q5_K: {                                                     \
                उपयुज्य type = खण्डम्_q5_K;                                               \
                उपयुज्य vec_dot_type = खण्डम्_q8_K;                                       \
                constexpr पूर्णाङ्क blck_आकार = QK_K;                                        \
                प्रत्यागम __VA_ARGS__();                                                  \
            }                                                                          \
            पक्ष GGML_TYPE_Q6_K: {                                                     \
                उपयुज्य type = खण्डम्_q6_K;                                               \
                उपयुज्य vec_dot_type = खण्डम्_q8_K;                                       \
                constexpr पूर्णाङ्क blck_आकार = QK_K;                                        \
                प्रत्यागम __VA_ARGS__();                                                  \
            }                                                                          \
            पक्ष GGML_TYPE_IQ4_XS: {                                                   \
                उपयुज्य type = खण्डम्_iq4_xs;                                             \
                उपयुज्य vec_dot_type = खण्डम्_q8_K;                                       \
                constexpr पूर्णाङ्क blck_आकार = QK_K;                                        \
                प्रत्यागम __VA_ARGS__();                                                  \
            }                                                                          \
            स्वभाव:                                                                   \
                fprintf(stderr, "Unsupported quantized data type: %d\n", पूर्णाङ्क(TYPE));   \
        }                                                                              \
    }()

#परिभाषय GGML_DISPATCH_BOOL(BOOL_V, BOOL_NAME, ...)                                     \
    [&] {                                                                              \
        यदि (BOOL_V) {                                                                  \
            constexpr तार्किक BOOL_NAME = सत्य;                                           \
            प्रत्यागम __VA_ARGS__();                                                      \
        } अन्यथा {                                                                       \
            constexpr तार्किक BOOL_NAME = असत्य;                                          \
            प्रत्यागम __VA_ARGS__();                                                      \
        }                                                                              \
    }()

// define amx tile config data structure
संरचना tile_config_t{
    uint8_t palette_id = 0;
    uint8_t start_row = 0;
    uint8_t reserved_0[14] = {0};
    uint16_t colsb[16] = {0};
    uint8_t पङ्क्तिकाः[16] = {0};
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

अन्तःस्थ शून्य ggml_tile_config_init(शून्य) {
    स्थिर thread_local तार्किक समाप्तम् = असत्य;

    यदि (समाप्तम्) {
        प्रत्यागम;
    }

    alignas(64) tile_config_t tc = {};
    tc.palette_id = 1;
    tc.start_row = 0;
    tc.पङ्क्तिकाः[0] = 8;   tc.colsb[0] = 64;
    tc.पङ्क्तिकाः[1] = 8;   tc.colsb[1] = 64;
    tc.पङ्क्तिकाः[2] = 16;  tc.colsb[2] = 32;
    tc.पङ्क्तिकाः[3] = 16;  tc.colsb[3] = 32;
    tc.पङ्क्तिकाः[4] = 16;  tc.colsb[4] = 64;
    tc.पङ्क्तिकाः[5] = 16;  tc.colsb[5] = 64;
    tc.पङ्क्तिकाः[6] = 16;  tc.colsb[6] = 64;
    tc.पङ्क्तिकाः[7] = 16;  tc.colsb[7] = 64;

    _tile_loadconfig(&tc);
    समाप्तम् = सत्य;
}

// we need an extra 16 * 4B (TILE_N * int32_t) for each NB/KB block for compensation.
// See the notes `s8s8 igemm compensation in avx512-vnni` for detail.
प्रतिरूपम् <प्रकारनाम TB>
पूर्णाङ्क get_tile_आकार() {
    पूर्णाङ्क tile_आकार = TILE_N * परिमाण(TB);
    यदि (do_compensate<TB>::value) {
        tile_आकार += TILE_N * परिमाण(int32_t);
    }
    यदि (std::is_same<TB, खण्डम्_q4_K>::value ||
        std::is_same<TB, खण्डम्_q5_K>::value) {
        tile_आकार += TILE_N * 4;
    }
    यदि (std::is_same<TB, खण्डम्_iq4_xs>::value) {
        tile_आकार += TILE_N * 2;
    }
    प्रत्यागम tile_आकार;
}

प्रतिरूपम् <प्रकारनाम TB, पूर्णाङ्क खण्डम्_K>
पूर्णाङ्क get_row_आकार(पूर्णाङ्क K) {
    पूर्णाङ्क KB = K / खण्डम्_K;
    पूर्णाङ्क row_आकार = KB * परिमाण(TB);
    यदि (do_compensate<TB>::value) {
        row_आकार += KB * परिमाण(int32_t);
    }
    यदि (std::is_same<TB, खण्डम्_q4_K>::value ||
        std::is_same<TB, खण्डम्_q5_K>::value) {
        row_आकार += KB * 4;
    }
    यदि (std::is_same<TB, खण्डम्_iq4_xs>::value) {
        row_आकार += KB * 2;
    }
    प्रत्यागम row_आकार;
}

// transpose utils
#परिभाषय SHUFFLE_EPI32(a, b, आवरणम्) \
    _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b), आवरणम्))
अन्तःस्थ शून्य transpose_8x8_32bit(__m256i * v, __m256i * v1) {
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

अन्तःस्थ शून्य transpose_16x4_32bit(__m512i * r, __m512i * d) {

    स्थिर ध्रुव __m512i index1 = mm512_समुच्चयः_epi32(
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

अन्तःस्थ शून्य transpose_16x16_32bit(__m512i * v) {
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

शून्य quantize_row_q8_K_vnni(ध्रुव भिन्न * RESTRICT x, शून्य * RESTRICT vy, int64_t k) {
    प्रतिज्ञा(k % QK_K == 0);
    ध्रुव पूर्णाङ्क KB = k / QK_K;
    constexpr पूर्णाङ्क kVecs = QK_K / 16;

    खण्डम्_q8_K * y = reinterpret_cast<खण्डम्_q8_K *>(vy);

    // hold 16 float vecs from x
    __m512  v[kVecs];

    // hold the quants vecs
    __m512i vq[kVecs / 4];

    // hold the packed quants vecs
    __m512i vq_packed[kVecs / 4];

    ध्रुव __m512 signBit = _mm512_set1_ps(-0.f);

    प्रति (पूर्णाङ्क i = 0; i < KB; ++i) {
        // Compute max(abs(e)) for the block
        __m512 vamax = _mm512_set1_ps(0.f);
        प्रति (पूर्णाङ्क j = 0; j < kVecs; ++j) {
            v[j] = _mm512_loadu_ps(x); x += 16;
            vamax = _mm512_max_ps(vamax, _mm512_andnot_ps(signBit, v[j]));
        }
        ध्रुव भिन्न amax = _mm512_reduce_max_ps(vamax);

        // Quantize these floats
        ध्रुव भिन्न iscale = 127.f / amax;
        y[i].d = GGML_CPU_FP32_गन्तव्यम्_FP16(1 / iscale);
        ध्रुव भिन्न id = ( amax != 0.0f ) ? iscale : 0.f;
        ध्रुव __m512 vscale = _mm512_set1_ps(id);

        // Apply multiplier and round to nearest integer
        प्रति (पूर्णाङ्क j = 0; j < kVecs; ++j) {
            v[j] = _mm512_mul_ps(v[j], vscale);
            v[j] = _mm512_roundscale_ps(v[j], (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        }

        // Pack to epi8 vecs
        प्रति (पूर्णाङ्क j = 0; j < kVecs / 4; ++j) {
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

        ध्रुव __m512i one = _mm512_set1_epi8(1);
        __m512i योग_2 = _mm512_setzero_si512();
        प्रति (पूर्णाङ्क k = 0; k < 4; ++k) {
            योग_2 = _mm512_dpbusd_epi32(योग_2, one, vq_packed[k]);
        }
        _mm256_storeu_si256((__m256i *)(y[i].bsums), _mm512_cvtepi32_epi16(योग_2));
    }
}

// quantize A from float to `vec_dot_type`
प्रतिरूपम् <प्रकारनाम T>
अन्तःस्थ शून्य from_float(ध्रुव भिन्न * x, वर्ण * vy, int64_t k);

प्रतिरूपम् <>
अन्तःस्थ शून्य from_float<खण्डम्_q8_0>(ध्रुव भिन्न * x, वर्ण * vy, int64_t k) {
    quantize_row_q8_0(x, (खण्डम्_q8_0 *)vy, k);
}

प्रतिरूपम् <>
अन्तःस्थ शून्य from_float<खण्डम्_q8_1>(ध्रुव भिन्न * x, वर्ण * vy, int64_t k) {
    quantize_row_q8_1(x, (खण्डम्_q8_1 *)vy, k);
}

प्रतिरूपम् <>
अन्तःस्थ शून्य from_float<खण्डम्_q8_K>(ध्रुव भिन्न * x, वर्ण * vy, int64_t k) {
#यदि 1
    // TODO: this is reference impl!
    quantize_row_q8_K_ref(x, (खण्डम्_q8_K *)vy, k);
#अन्यथा
    quantize_row_q8_K_vnni(x, vy, k);
#यदि_अन्तः
}

// load A from memory to array when nrows can not fill in whole tile
शून्य unpack_A(int8_t * RESTRICT tile, ध्रुव खण्डम्_q8_0 * RESTRICT A, पूर्णाङ्क lda, पूर्णाङ्क nr) {
    प्रतिज्ञा(nr != TILE_M);
    प्रति (पूर्णाङ्क m = 0; m < nr; ++m) {
        ध्रुव __m256i v = _mm256_loadu_si256((ध्रुव __m256i *)(A[m * lda].qs));
        _mm256_storeu_si256((__m256i *)(tile + m * TILE_K), v);
    }
}

शून्य unpack_A(int8_t * RESTRICT tile, ध्रुव खण्डम्_q8_1 * RESTRICT A, पूर्णाङ्क lda, पूर्णाङ्क nr) {
    प्रतिज्ञा(nr != TILE_M);
    प्रति (पूर्णाङ्क m = 0; m < nr; ++m) {
        ध्रुव __m256i v = _mm256_loadu_si256((ध्रुव __m256i *)(A[m * lda].qs));
        _mm256_storeu_si256((__m256i *)(tile + m * TILE_K), v);
    }
}

प्रतिरूपम् <प्रकारनाम TB>
शून्य unpack_A(int8_t * RESTRICT tile, ध्रुव खण्डम्_q8_K * RESTRICT A, पूर्णाङ्क lda, पूर्णाङ्क k, पूर्णाङ्क nr) {
    प्रतिज्ञा(nr <= TILE_M);
    प्रति (पूर्णाङ्क m = 0; m < nr; ++m) {
        ध्रुव __m256i v = _mm256_loadu_si256((ध्रुव __m256i *)(A[m * lda].qs + k * 32));
        _mm256_storeu_si256((__m256i *)(tile + m * TILE_K), v);
    }
}

प्रतिरूपम् <>
शून्य unpack_A<खण्डम्_q6_K>(int8_t * RESTRICT tile, ध्रुव खण्डम्_q8_K * RESTRICT A, पूर्णाङ्क lda, पूर्णाङ्क k, पूर्णाङ्क nr) {
    प्रतिज्ञा(nr <= TILE_M);
    // zero padding k from 16 to 32, so that we don't have to re-config amx
    ध्रुव __m128i zero = _mm_setzero_si128();
    प्रति (पूर्णाङ्क m = 0; m < nr; ++m) {
        ध्रुव __m128i v = _mm_loadu_si128((ध्रुव __m128i *)(A[m * lda].qs + k * 16));
        ध्रुव __m256i r = _mm256_insertf128_si256(_mm256_castsi128_si256(v), zero, 1);
        _mm256_storeu_si256((__m256i *)(tile + m * TILE_K), r);
    }
}

#परिभाषय MM256_समुच्चयः_M128_I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)
अन्तःस्थ __m256i अष्टकानि_from_nibbles_32(ध्रुव uint8_t * rsi) {
    ध्रुव __m128i tmp = _mm_loadu_si128((ध्रुव __m128i *)rsi);
    ध्रुव __m256i अष्टकानि = MM256_समुच्चयः_M128_I(_mm_srli_epi16(tmp, 4), tmp);
    ध्रुव __m256i नीचः_आवरणम् = _mm256_set1_epi8(0xF);
    प्रत्यागम _mm256_and_si256(नीचः_आवरणम्, अष्टकानि);
}

// used for block_q4_K
अन्तःस्थ __m512i अष्टकानि_from_nibbles_64(ध्रुव uint8_t * rsi) {
    ध्रुव __m256i tmp = _mm256_loadu_si256((ध्रुव __m256i *)rsi);
    ध्रुव __m256i नीचः_आवरणम् = _mm256_set1_epi8(0xF);
    ध्रुव __m256i q4l = _mm256_and_si256(tmp, नीचः_आवरणम्);
    ध्रुव __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(tmp, 4), नीचः_आवरणम्);
    प्रत्यागम _mm512_inserti32x8(_mm512_castsi256_si512(q4l), q4h, 1);
}

// used for block_q5_K
अन्तःस्थ __m512i अष्टकानि_from_nibbles_64(ध्रुव uint8_t * qs, ध्रुव uint8_t * qh, पूर्णाङ्क k) {
    ध्रुव __m256i नीचः_आवरणम् = _mm256_set1_epi8(0xF);
    __m256i hmask = _mm256_set1_epi8(1);
    hmask = _mm256_slli_epi16(hmask, k);

    ध्रुव __m256i q5bits = _mm256_loadu_si256((ध्रुव __m256i *)qs);
    ध्रुव __m256i hbits = _mm256_loadu_si256((ध्रुव __m256i *)qh);

    ध्रुव __m256i q5l_0 = _mm256_and_si256(q5bits, नीचः_आवरणम्);
    ध्रुव __m256i q5h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), k + 0), 4);
    ध्रुव __m256i q5_0  = mm256_सङ्कलय_epi8(q5l_0, q5h_0);
    hmask = _mm256_slli_epi16(hmask, 1);

    ध्रुव __m256i q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), नीचः_आवरणम्);
    ध्रुव __m256i q5h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), k + 1), 4);
    ध्रुव __m256i q5_1  = mm256_सङ्कलय_epi8(q5l_1, q5h_1);

    प्रत्यागम _mm512_inserti32x8(_mm512_castsi256_si512(q5_0), q5_1, 1);
}

// used for block_q6_K
अन्तःस्थ शून्य अष्टकानि_from_nibbles_128(__m512i& r0, __m512i& r1, ध्रुव uint8_t * qs, ध्रुव uint8_t * qh) {
    ध्रुव __m256i m4 = _mm256_set1_epi8(0xF);
    ध्रुव __m256i m2 = _mm256_set1_epi8(0x3);

    ध्रुव __m256i q6bits1 = _mm256_loadu_si256((ध्रुव __m256i *)qs);
    ध्रुव __m256i q6bits2 = _mm256_loadu_si256((ध्रुव __m256i *)(qs + 32));
    ध्रुव __m256i q6bitsH = _mm256_loadu_si256((ध्रुव __m256i *)qh);

    ध्रुव __m256i q6h_0 = _mm256_slli_epi16(_mm256_and_si256(                  q6bitsH,     m2), 4);
    ध्रुव __m256i q6h_1 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q6bitsH, 2), m2), 4);
    ध्रुव __m256i q6h_2 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q6bitsH, 4), m2), 4);
    ध्रुव __m256i q6h_3 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q6bitsH, 6), m2), 4);

    ध्रुव __m256i q6_0 = _mm256_or_si256(_mm256_and_si256(q6bits1, m4), q6h_0);
    ध्रुव __m256i q6_1 = _mm256_or_si256(_mm256_and_si256(q6bits2, m4), q6h_1);
    ध्रुव __m256i q6_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q6bits1, 4), m4), q6h_2);
    ध्रुव __m256i q6_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q6bits2, 4), m4), q6h_3);

    r0 = _mm512_inserti32x8(_mm512_castsi256_si512(q6_0), q6_1, 1);
    r1 = _mm512_inserti32x8(_mm512_castsi256_si512(q6_2), q6_3, 1);
}

अन्तःस्थ __m512i packNibbles(__m512i r0, __m512i r1) {
    प्रत्यागम _mm512_or_si512(r0, _mm512_slli_epi16(r1, 4));
}

प्रतिरूपम् <प्रकारनाम TB>
अन्तःस्थ शून्य pack_qs(शून्य * RESTRICT packed_B, ध्रुव TB * RESTRICT B, पूर्णाङ्क KB) {
    int8_t tmp[8 * 64];
    __m256i v[8], v2[8];
    प्रति (पूर्णाङ्क n = 0; n < 8; ++n) {
        v[n] = अष्टकानि_from_nibbles_32(B[n * KB].qs);
    }
    transpose_8x8_32bit(v, v2);
    प्रति (पूर्णाङ्क n = 0; n < 8; ++n) {
        _mm256_storeu_si256((__m256i *)(tmp + n * 64), v2[n]);
    }
    प्रति (पूर्णाङ्क n = 0; n < 8; ++n) {
        v[n] = अष्टकानि_from_nibbles_32(B[(n + 8) * KB].qs);
    }
    transpose_8x8_32bit(v, v2);
    प्रति (पूर्णाङ्क n = 0; n < 8; ++n) {
        _mm256_storeu_si256((__m256i *)(tmp + n * 64 + 32), v2[n]);
    }

    // pack again with 128 to fully utilize vector length
    प्रति (पूर्णाङ्क n = 0; n < 8; n += 2) {
        __m512i r0 = _mm512_loadu_si512((ध्रुव __m512i *)(tmp + n * 64));
        __m512i r1 = _mm512_loadu_si512((ध्रुव __m512i *)(tmp + n * 64 + 64));
        __m512i r1r0 = packNibbles(r0, r1);
        _mm512_storeu_si512((__m512i *)((वर्ण *)packed_B + n * 32), r1r0);
    }
}

प्रतिरूपम् <>
अन्तःस्थ शून्य pack_qs<खण्डम्_q8_0>(शून्य * RESTRICT packed_B, ध्रुव खण्डम्_q8_0 * RESTRICT B, पूर्णाङ्क KB) {
    __m256i v[8], v2[8];
    प्रति (पूर्णाङ्क n = 0; n < 8; ++n) {
        v[n] = _mm256_loadu_si256((ध्रुव __m256i *)(B[n * KB].qs));
    }
    transpose_8x8_32bit(v, v2);
    प्रति (पूर्णाङ्क n = 0; n < 8; ++n) {
        _mm256_storeu_si256((__m256i *)((वर्ण *)packed_B + n * 64), v2[n]);
    }
    प्रति (पूर्णाङ्क n = 0; n < 8; ++n) {
        v[n] = _mm256_loadu_si256((ध्रुव __m256i *)(B[(n + 8) * KB].qs));
    }
    transpose_8x8_32bit(v, v2);
    प्रति (पूर्णाङ्क n = 0; n < 8; ++n) {
        _mm256_storeu_si256((__m256i *)((वर्ण *)packed_B + n * 64 + 32), v2[n]);
    }
}

प्रतिरूपम् <>
अन्तःस्थ शून्य pack_qs<खण्डम्_q4_K>(शून्य * RESTRICT packed_B, ध्रुव खण्डम्_q4_K * RESTRICT B, पूर्णाङ्क KB) {
    __m512i v[16];
    // QK_K 256 with 8 groups, handle 2 groups at a time
    वर्ण * pb = (वर्ण *)packed_B;
    प्रति (पूर्णाङ्क k = 0; k < QK_K / 64; ++k) {
        // pack 2 groups { n, g,  k} to {g, k/4, 4n}
        //          e.g. {16, 2, 32} to {2,   8, 64}
        प्रति (पूर्णाङ्क n = 0; n < TILE_N; ++n) {
            v[n] = अष्टकानि_from_nibbles_64(B[n * KB].qs + k * 32);
        }

        transpose_16x16_32bit(v);

        // pack again with 128 to fully utilize vector length
        प्रति (पूर्णाङ्क n = 0; n < TILE_N; n += 2) {
            _mm512_storeu_si512((__m512i *)pb, packNibbles(v[n], v[n + 1]));
            pb += 64;
        }
    }
}

प्रतिरूपम् <>
अन्तःस्थ शून्य pack_qs<खण्डम्_q5_K>(शून्य * RESTRICT packed_B, ध्रुव खण्डम्_q5_K * RESTRICT B, पूर्णाङ्क KB) {
    __m512i v[16];
    ध्रुव __m512i नीचः_आवरणम् = _mm512_set1_epi8(0xF);
    // QK_K 256 with 8 groups, handle 2 groups at a time
    वर्ण * pb = (वर्ण *)packed_B;
    वर्ण * ph = (वर्ण *)packed_B + (QK_K / 2) * TILE_N;
    प्रति (पूर्णाङ्क k = 0; k < QK_K / 64; ++k) {
        // pack 2 groups { n, g,  k} to {g, k/4, 4n}
        //          e.g. {16, 2, 32} to {2,   8, 64}
        प्रति (पूर्णाङ्क n = 0; n < TILE_N; ++n) {
            v[n] = अष्टकानि_from_nibbles_64(B[n * KB].qs + k * 32, B[n * KB].qh, /*गणः*/2 * k);
        }

        transpose_16x16_32bit(v);

        // 1. pack lower 4bits with 2 groups
        प्रति (पूर्णाङ्क n = 0; n < TILE_N; n += 2) {
            // get lower 4 bits
            ध्रुव __m512i r0 = _mm512_and_si512(v[n], नीचः_आवरणम्);
            ध्रुव __m512i r1 = _mm512_and_si512(v[n + 1], नीचः_आवरणम्);
            _mm512_storeu_si512((__m512i *)pb, packNibbles(r0, r1)); pb += 64;
        }

        // 2. pack higher 1bit with 2 groups
        ध्रुव __m512i hmask = _mm512_set1_epi8(0x10);
        प्रति (पूर्णाङ्क g = 0; g < 2; ++g) {
            __m512i hbits = _mm512_setzero_si512();
            hbits = mm512_सङ्कलय_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 8 + 0], hmask), 4));
            hbits = mm512_सङ्कलय_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 8 + 1], hmask), 3));
            hbits = mm512_सङ्कलय_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 8 + 2], hmask), 2));
            hbits = mm512_सङ्कलय_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 8 + 3], hmask), 1));
            hbits = mm512_सङ्कलय_epi8(hbits,                   _mm512_and_si512(v[g * 8 + 4], hmask)    );
            hbits = mm512_सङ्कलय_epi8(hbits, _mm512_slli_epi16(_mm512_and_si512(v[g * 8 + 5], hmask), 1));
            hbits = mm512_सङ्कलय_epi8(hbits, _mm512_slli_epi16(_mm512_and_si512(v[g * 8 + 6], hmask), 2));
            hbits = mm512_सङ्कलय_epi8(hbits, _mm512_slli_epi16(_mm512_and_si512(v[g * 8 + 7], hmask), 3));
            _mm512_storeu_si512((__m512i *)ph, hbits); ph += 64;
        }
    }
}

प्रतिरूपम् <>
अन्तःस्थ शून्य pack_qs<खण्डम्_q6_K>(शून्य * RESTRICT packed_B, ध्रुव खण्डम्_q6_K * RESTRICT B, पूर्णाङ्क KB) {
    __m512i v[32];
    ध्रुव __m512i नीचः_आवरणम् = _mm512_set1_epi8(0xF);
    // QK_K 256 with 8 groups, handle 4 groups at a time
    वर्ण * pb = (वर्ण *)packed_B;
    वर्ण * ph = (वर्ण *)packed_B + (QK_K / 2) * TILE_N;
    प्रति (पूर्णाङ्क k = 0; k < QK_K / 128; ++k) {
        प्रति (पूर्णाङ्क n = 0; n < TILE_N; ++n) {
            अष्टकानि_from_nibbles_128(v[n], v[n + 16], B[n * KB].ql + k * 64, B[n * KB].qh + k * 32);
        }

        // top half: group 0,1 or 4,5; bottom half: group 2,3 or 6,7
        transpose_16x16_32bit(v);
        transpose_16x16_32bit(v + 16);

        // 1. pack lower 4bits with 4 groups
        प्रति (पूर्णाङ्क n = 0; n < 32; n += 2) {
            ध्रुव __m512i r0 = _mm512_and_si512(v[n], नीचः_आवरणम्);
            ध्रुव __m512i r1 = _mm512_and_si512(v[n + 1], नीचः_आवरणम्);
            _mm512_storeu_si512((__m512i *)pb, packNibbles(r0, r1)); pb += 64;
        }

        // 2. pack higher 2bit with 4 groups
        ध्रुव __m512i hmask = _mm512_set1_epi8(0x30);
        प्रति (पूर्णाङ्क g = 0; g < 8; ++g) {
            __m512i hbits = _mm512_setzero_si512();
            hbits = mm512_सङ्कलय_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 4 + 0], hmask), 4));
            hbits = mm512_सङ्कलय_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 4 + 1], hmask), 2));
            hbits = mm512_सङ्कलय_epi8(hbits,                   _mm512_and_si512(v[g * 4 + 2], hmask)    );
            hbits = mm512_सङ्कलय_epi8(hbits, _mm512_slli_epi16(_mm512_and_si512(v[g * 4 + 3], hmask), 2));
            _mm512_storeu_si512((__m512i *)ph, hbits); ph += 64;
        }
    }
}

प्रतिरूपम् <>
अन्तःस्थ शून्य pack_qs<खण्डम्_iq4_xs>(शून्य * RESTRICT packed_B, ध्रुव खण्डम्_iq4_xs * RESTRICT B, पूर्णाङ्क KB) {
    __m512i v[16];
    वर्ण * pb = (वर्ण *)packed_B;
    प्रति (पूर्णाङ्क k = 0; k < QK_K / 64; ++k) {
        प्रति (पूर्णाङ्क n = 0; n < TILE_N; ++n) {
            __m256i r0 = अष्टकानि_from_nibbles_32(B[n * KB].qs + k * 32 +  0);
            __m256i r1 = अष्टकानि_from_nibbles_32(B[n * KB].qs + k * 32 + 16);
            v[n] = _mm512_inserti32x8(_mm512_castsi256_si512(r0), r1, 1);
        }

        transpose_16x16_32bit(v);

        // pack again with 128 to fully utilize vector length
        प्रति (पूर्णाङ्क n = 0; n < TILE_N; n += 2) {
            _mm512_storeu_si512((__m512i *)pb, packNibbles(v[n], v[n + 1]));
            pb += 64;
        }
    }
}

// pack B to vnni formats in 4bits or 8 bits
शून्य pack_B(शून्य * RESTRICT packed_B, ध्रुव खण्डम्_q4_0 * RESTRICT B, पूर्णाङ्क KB) {
    pack_qs(packed_B, B, KB);
    ggml_half * d0 = reinterpret_cast<ggml_half *>((वर्ण *)packed_B + TILE_N * TILE_K / 2);
    प्रति (पूर्णाङ्क n = 0; n < TILE_N; ++n) {
        d0[n] = B[n * KB].d;
    }
}

शून्य pack_B(शून्य * RESTRICT packed_B, ध्रुव खण्डम्_q4_1 * RESTRICT B, पूर्णाङ्क KB) {
    pack_qs(packed_B, B, KB);
    ggml_half * d0 = reinterpret_cast<ggml_half *>((वर्ण *)packed_B + TILE_N * TILE_K / 2);
    ggml_half * m0 = d0 + TILE_N;
    प्रति (पूर्णाङ्क n = 0; n < TILE_N; ++n) {
        d0[n] = B[n * KB].d;
        m0[n] = B[n * KB].m;
    }
}

अन्तःस्थ शून्य s8s8_compensation(शून्य * RESTRICT packed_B) {
    // packed_B layout:
    //   quants {TILE_N, TILEK}  int8_t
    //   d0     {TILE_N}      ggml_half
    //   comp   {TILE_N}        int32_t
    ध्रुव पूर्णाङ्क विस्थापनम् = TILE_N * TILE_K + TILE_N * परिमाण(ggml_half);
    __m512i vcomp = _mm512_setzero_si512();
    ध्रुव __m512i off = _mm512_set1_epi8(static_cast<वर्ण>(0x80));
    प्रति (पूर्णाङ्क k = 0; k < 8; ++k) {
        __m512i vb = _mm512_loadu_si512((ध्रुव __m512i *)((ध्रुव वर्ण *)packed_B + k * 64));
        vcomp = _mm512_dpbusd_epi32(vcomp, off, vb);
    }
    _mm512_storeu_si512((__m512i *)((वर्ण *)(packed_B) + विस्थापनम्), vcomp);
}

शून्य pack_B(शून्य * RESTRICT packed_B, ध्रुव खण्डम्_q8_0 * RESTRICT B, पूर्णाङ्क KB) {
    pack_qs(packed_B, B, KB);
    ggml_half * d0 = reinterpret_cast<ggml_half *>((वर्ण *)packed_B + TILE_N * TILE_K);
    प्रति (पूर्णाङ्क n = 0; n < TILE_N; ++n) {
        d0[n] = B[n * KB].d;
    }
    s8s8_compensation(packed_B);
}

// convert 8 * {min, scale} from int6 to int8
अन्तःस्थ शून्य unpack_mins_and_scales(ध्रुव uint8_t * scales, uint32_t * utmp) {
    ध्रुव uint32_t kmask1 = 0x3f3f3f3f;
    ध्रुव uint32_t kmask2 = 0x0f0f0f0f;
    ध्रुव uint32_t kmask3 = 0x03030303;

    स्मृतिं_प्रतिलिख(utmp, scales, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    ध्रुव uint32_t uaux = utmp[1] & kmask1;
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
शून्य pack_B(शून्य * RESTRICT packed_B, ध्रुव खण्डम्_q4_K * RESTRICT B, पूर्णाङ्क KB) {
    pack_qs(packed_B, B, KB);

    uint8_t * scales = reinterpret_cast<uint8_t *>((वर्ण *)packed_B + (QK_K / 2) * TILE_N);
    uint8_t * mins = scales + 8 * TILE_N;
    ggml_half * d = reinterpret_cast<ggml_half *>(mins + 8 * TILE_N);
    ggml_half * dmin = d + TILE_N;

    सङ्घ {
        uint32_t u32[4];
        uint8_t  u8[16];
    } s;

    प्रति (पूर्णाङ्क n = 0; n < TILE_N; ++n) {
        unpack_mins_and_scales(B[n * KB].scales, s.u32);
        प्रति (पूर्णाङ्क k = 0; k < 8; ++k) {
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
शून्य pack_B(शून्य * RESTRICT packed_B, ध्रुव खण्डम्_q5_K * RESTRICT B, पूर्णाङ्क KB) {
    pack_qs(packed_B, B, KB);

    uint8_t * scales = reinterpret_cast<uint8_t *>((वर्ण *)packed_B + (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N);
    uint8_t * mins = scales + 8 * TILE_N;
    ggml_half * d = reinterpret_cast<ggml_half *>(mins + 8 * TILE_N);
    ggml_half * dmin = d + TILE_N;

    सङ्घ {
        uint32_t u32[4];
        uint8_t  u8[16];
    } s;

    प्रति (पूर्णाङ्क n = 0; n < TILE_N; ++n) {
        unpack_mins_and_scales(B[n * KB].scales, s.u32);
        प्रति (पूर्णाङ्क k = 0; k < 8; ++k) {
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
शून्य pack_B(शून्य * RESTRICT packed_B, ध्रुव खण्डम्_q6_K * RESTRICT B, पूर्णाङ्क KB) {
    pack_qs(packed_B, B, KB);

    uint8_t * scales = reinterpret_cast<uint8_t *>((वर्ण *)packed_B + (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N);
    ggml_half * d = reinterpret_cast<ggml_half *>(scales + 16 * TILE_N);
    प्रति (पूर्णाङ्क n = 0; n < TILE_N; ++n) {
        ध्रुव int8_t * ps = B[n * KB].scales;
        प्रति (पूर्णाङ्क k = 0; k < 16; ++k) {
            scales[k * TILE_N + n] = ps[k];
        }
        d[n] = B[n * KB].d;
    }
}

// packed_B layout:
//   quants {8, TILE_N, 16}  uint8
//   scales {8, TILE_N}       int8
//   d      {TILE_N}     ggml_half
शून्य pack_B(शून्य * RESTRICT packed_B, ध्रुव खण्डम्_iq4_xs * RESTRICT B, पूर्णाङ्क KB) {
    pack_qs(packed_B, B, KB);

    int8_t * scales = reinterpret_cast<int8_t *>((वर्ण *)packed_B + (QK_K / 2) * TILE_N);
    ggml_half * d = reinterpret_cast<ggml_half *>(scales + 8 * TILE_N);

    // pack the scales
    प्रति (पूर्णाङ्क n = 0; n < TILE_N; ++n) {
        uint16_t sh = B[n * KB].scales_h;
        प्रति (पूर्णाङ्क k = 0; k < 8; k += 2) {
            ध्रुव int16_t ls1 = ((B[n * KB].scales_l[k / 2] & 0xf) | ((sh << 4) & 0x30)) - 32;
            ध्रुव int16_t ls2 = ((B[n * KB].scales_l[k / 2] >>  4) | ((sh << 2) & 0x30)) - 32;
            scales[(k + 0) * TILE_N + n] = ls1;
            scales[(k + 1) * TILE_N + n] = ls2;
            sh >>= 4;
        }
        d[n] = B[n * KB].d;
    }
}

प्रतिरूपम्<प्रकारनाम TB, प्रकारनाम packed_B_t = packed_B_type<TB>>
शून्य unpack_B(packed_B_t * RESTRICT tile, ध्रुव शून्य * RESTRICT packed_B) {
    GGML_अप्रयुक्तम्(tile);
    GGML_अप्रयुक्तम्(packed_B);
}

प्रतिरूपम् <>
शून्य unpack_B<खण्डम्_q4_0>(int8_t * RESTRICT tile, ध्रुव शून्य * RESTRICT packed_B) {
  ध्रुव __m512i off = _mm512_set1_epi8(8);
  ध्रुव __m512i नीचः_आवरणम् = _mm512_set1_epi8(0xF);
  प्रति (पूर्णाङ्क n = 0; n < 8; n += 2) {
    __m512i अष्टकानि = _mm512_loadu_si512((ध्रुव __m512i *)((ध्रुव वर्ण *)packed_B + n * 32));
    ध्रुव __m512i r0 = _mm512_sub_epi8(_mm512_and_si512(अष्टकानि, नीचः_आवरणम्), off);
    ध्रुव __m512i r1 = _mm512_sub_epi8(_mm512_and_si512(_mm512_srli_epi16(अष्टकानि, 4), नीचः_आवरणम्), off);
    _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
    _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
  }
}

प्रतिरूपम् <>
शून्य unpack_B<खण्डम्_q4_1>(uint8_t * RESTRICT tile, ध्रुव शून्य * RESTRICT packed_B) {
    ध्रुव __m512i नीचः_आवरणम् = _mm512_set1_epi8(0xF);
    प्रति (पूर्णाङ्क n = 0; n < 8; n += 2) {
        __m512i अष्टकानि = _mm512_loadu_si512((ध्रुव __m512i *)((ध्रुव वर्ण *)packed_B + n * 32));
        ध्रुव __m512i r0 = _mm512_and_si512(अष्टकानि, नीचः_आवरणम्);
        ध्रुव __m512i r1 = _mm512_and_si512(_mm512_srli_epi16(अष्टकानि, 4), नीचः_आवरणम्);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
    }
}

// packed_B_t for QKK is int8_t
प्रतिरूपम् <प्रकारनाम TB>
शून्य unpack_B(int8_t * RESTRICT tile, ध्रुव शून्य * RESTRICT packed_B, पूर्णाङ्क k) {
    ध्रुव पूर्णाङ्क packed_B_गणः_आकार = QK_K / 2 * TILE_N / 8;
    ध्रुव वर्ण * packed_B_गणः = (ध्रुव वर्ण *)packed_B + k * packed_B_गणः_आकार;
    ध्रुव __m512i नीचः_आवरणम् = _mm512_set1_epi8(0xF);
    प्रति (पूर्णाङ्क n = 0; n < 8; n += 2) {
        __m512i अष्टकानि = _mm512_loadu_si512(packed_B_गणः + n * 32);
        ध्रुव __m512i r0 = _mm512_and_si512(अष्टकानि, नीचः_आवरणम्);
        ध्रुव __m512i r1 = _mm512_and_si512(_mm512_srli_epi16(अष्टकानि, 4), नीचः_आवरणम्);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
    }
}

प्रतिरूपम् <>
शून्य unpack_B<खण्डम्_q5_K>(int8_t * RESTRICT tile, ध्रुव शून्य * RESTRICT packed_B, पूर्णाङ्क k) {
    // lower 4bits, stride 256 bytes
    ध्रुव पूर्णाङ्क packed_l4_गणः_आकार = QK_K / 2 * TILE_N / 8;
    ध्रुव वर्ण * pb = (ध्रुव वर्ण *)packed_B + k * packed_l4_गणः_आकार;

    // higher 1bit, stride 64 bytes
    ध्रुव पूर्णाङ्क packed_h1_गणः_आकार = QK_K / 8 * TILE_N / 8;
    ध्रुव वर्ण * ph = (ध्रुव वर्ण *)packed_B + (QK_K / 2) * TILE_N + k * packed_h1_गणः_आकार;
    ध्रुव __m512i hbits = _mm512_loadu_si512(ph);

    ध्रुव __m512i नीचः_आवरणम् = _mm512_set1_epi8(0xF);
    __m512i hmask0 = _mm512_set1_epi8(0x1);
    __m512i hmask1 = _mm512_set1_epi8(0x2);

    प्रति (पूर्णाङ्क n = 0; n < 8; n += 2) {
        __m512i अष्टकानि = _mm512_loadu_si512(pb + n * 32);
        __m512i r0 = _mm512_and_si512(अष्टकानि, नीचः_आवरणम्);
        __m512i r1 = _mm512_and_si512(_mm512_srli_epi16(अष्टकानि, 4), नीचः_आवरणम्);
        __m512i h0 = _mm512_slli_epi16(_mm512_srli_epi16(_mm512_and_si512(hbits, hmask0), n), 4);
        __m512i h1 = _mm512_slli_epi16(_mm512_srli_epi16(_mm512_and_si512(hbits, hmask1), n + 1), 4);

        hmask0 = _mm512_slli_epi16(hmask0, 2);
        hmask1 = _mm512_slli_epi16(hmask1, 2);
        r0 = mm512_सङ्कलय_epi8(r0, h0);
        r1 = mm512_सङ्कलय_epi8(r1, h1);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
    }
}

प्रतिरूपम् <>
शून्य unpack_B<खण्डम्_q6_K>(int8_t * RESTRICT tile, ध्रुव शून्य * RESTRICT packed_B, पूर्णाङ्क k) {
    // lower 4bits, stride 128 bytes
    ध्रुव पूर्णाङ्क packed_l4_गणः_आकार = QK_K / 2 * TILE_N / 16;
    ध्रुव वर्ण * pb = (ध्रुव वर्ण *)packed_B + k * packed_l4_गणः_आकार;

    // higher 2bits, stride 64 bytes
    ध्रुव पूर्णाङ्क packed_h2_गणः_आकार = QK_K / 4 * TILE_N / 16;
    ध्रुव वर्ण * ph = (ध्रुव वर्ण *)packed_B + (QK_K / 2) * TILE_N + k * packed_h2_गणः_आकार;
    ध्रुव __m512i hbits = _mm512_loadu_si512(ph);

    ध्रुव __m512i off = _mm512_set1_epi8(32);
    ध्रुव __m512i नीचः_आवरणम् = _mm512_set1_epi8(0xF);
    __m512i hmask0 = _mm512_set1_epi8(0x3); // 0011
    __m512i hmask1 = _mm512_set1_epi8(0xC); // 1100

    // notes: skip zero padding from row4 to row7 as we have done so in `unpack_A`
    __m512i अष्टकानि = _mm512_loadu_si512(pb);
    __m512i r0 = _mm512_and_si512(अष्टकानि, नीचः_आवरणम्);
    __m512i r1 = _mm512_and_si512(_mm512_srli_epi16(अष्टकानि, 4), नीचः_आवरणम्);
    __m512i h0 = _mm512_slli_epi16(_mm512_and_si512(hbits, hmask0), 4);
    __m512i h1 = _mm512_slli_epi16(_mm512_and_si512(hbits, hmask1), 2);
    _mm512_storeu_si512((__m512i *)(tile +  0), _mm512_sub_epi8(mm512_सङ्कलय_epi8(r0, h0), off));
    _mm512_storeu_si512((__m512i *)(tile + 64), _mm512_sub_epi8(mm512_सङ्कलय_epi8(r1, h1), off));

    hmask0 = _mm512_slli_epi16(hmask0, 4);
    hmask1 = _mm512_slli_epi16(hmask1, 4);

    अष्टकानि = _mm512_loadu_si512(pb + 64);
    r0 = _mm512_and_si512(अष्टकानि, नीचः_आवरणम्);
    r1 = _mm512_and_si512(_mm512_srli_epi16(अष्टकानि, 4), नीचः_आवरणम्);
    h0 =                   _mm512_and_si512(hbits, hmask0);
    h1 = _mm512_srli_epi16(_mm512_and_si512(hbits, hmask1), 2);
    _mm512_storeu_si512((__m512i *)(tile + 128), _mm512_sub_epi8(mm512_सङ्कलय_epi8(r0, h0), off));
    _mm512_storeu_si512((__m512i *)(tile + 192), _mm512_sub_epi8(mm512_सङ्कलय_epi8(r1, h1), off));
}

प्रतिरूपम् <>
शून्य unpack_B<खण्डम्_iq4_xs>(int8_t * RESTRICT tile, ध्रुव शून्य * RESTRICT packed_B, पूर्णाङ्क k) {
    स्थिर ध्रुव __m512i values128 = mm512_समुच्चयः_epi8(
        113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
        113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
        113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
        113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127
    );

    ध्रुव पूर्णाङ्क packed_B_गणः_आकार = QK_K / 2 * TILE_N / 8;
    ध्रुव वर्ण * pb = (ध्रुव वर्ण *)packed_B + k * packed_B_गणः_आकार;
    ध्रुव __m512i नीचः_आवरणम् = _mm512_set1_epi8(0xF);

    प्रति (पूर्णाङ्क n = 0; n < 8; n += 2) {
        __m512i अष्टकानि = _mm512_loadu_si512(pb + n * 32);
        ध्रुव __m512i r0 = _mm512_shuffle_epi8(values128, _mm512_and_si512(अष्टकानि, नीचः_आवरणम्));
        ध्रुव __m512i r1 = _mm512_shuffle_epi8(values128, _mm512_and_si512(_mm512_srli_epi16(अष्टकानि, 4), नीचः_आवरणम्));
        _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
    }
}

प्रतिरूपम् <प्रकारनाम TA, प्रकारनाम TB, तार्किक is_acc>
संरचना acc_C {};

प्रतिरूपम् <तार्किक is_acc>
संरचना acc_C<खण्डम्_q8_0, खण्डम्_q4_0, is_acc> {
    स्थिर शून्य apply(भिन्न * RESTRICT C, पूर्णाङ्क ldc, ध्रुव int32_t * RESTRICT tile, ध्रुव खण्डम्_q8_0 * A, पूर्णाङ्क lda, ध्रुव शून्य * packed_B, पूर्णाङ्क nr) {
        ध्रुव पूर्णाङ्क विस्थापनम् = TILE_N * TILE_K / 2;
        ध्रुव __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ध्रुव __m256i *)((ध्रुव वर्ण *)packed_B + विस्थापनम्)));

        प्रति (पूर्णाङ्क m = 0; m < nr; ++m) {
            ध्रुव __m512 vd1 = _mm512_set1_ps(GGML_CPU_FP16_गन्तव्यम्_FP32(A[m * lda].d));
            ध्रुव __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            यदि (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } अन्यथा {
                vsum = _mm512_set1_ps(0.f);
            }
            vsum = _mm512_fmadd_ps(vtile, _mm512_mul_ps(vd0, vd1), vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

प्रतिरूपम् <तार्किक is_acc>
संरचना acc_C<खण्डम्_q8_1, खण्डम्_q4_1, is_acc> {
    स्थिर शून्य apply(भिन्न * RESTRICT C, पूर्णाङ्क ldc, ध्रुव int32_t * RESTRICT tile, ध्रुव खण्डम्_q8_1 * A, पूर्णाङ्क lda, ध्रुव शून्य * packed_B, पूर्णाङ्क nr) {
        ध्रुव पूर्णाङ्क विस्थापनम् = TILE_N * TILE_K / 2;
        ध्रुव __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ध्रुव __m256i *)((ध्रुव वर्ण *)packed_B + विस्थापनम्)));
        ध्रुव __m512 vm0 = _mm512_cvtph_ps(_mm256_loadu_si256((ध्रुव __m256i *)((ध्रुव वर्ण *)packed_B + विस्थापनम् + TILE_N * परिमाण(ggml_half))));

        प्रति (पूर्णाङ्क m = 0; m < nr; ++m) {
            ध्रुव __m512 vd1 = _mm512_set1_ps(GGML_CPU_FP16_गन्तव्यम्_FP32(A[m * lda].d));
            ध्रुव __m512 vs1 = _mm512_set1_ps(GGML_CPU_FP16_गन्तव्यम्_FP32(A[m * lda].s));
            ध्रुव __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            यदि (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } अन्यथा {
                vsum = _mm512_set1_ps(0.f);
            }
            vsum = _mm512_fmadd_ps(vtile, _mm512_mul_ps(vd0, vd1), vsum);
            vsum = _mm512_fmadd_ps(vm0, vs1, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

प्रतिरूपम् <तार्किक is_acc>
संरचना acc_C<खण्डम्_q8_0, खण्डम्_q8_0, is_acc> {
    स्थिर शून्य apply(भिन्न * RESTRICT C, पूर्णाङ्क ldc, ध्रुव int32_t * RESTRICT tile, ध्रुव खण्डम्_q8_0 * A, पूर्णाङ्क lda, ध्रुव शून्य * packed_B, पूर्णाङ्क nr) {
        ध्रुव पूर्णाङ्क विस्थापनम् = TILE_N * TILE_K;
        ध्रुव __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ध्रुव __m256i *)((ध्रुव वर्ण *)packed_B + विस्थापनम्)));

        प्रति (पूर्णाङ्क m = 0; m < nr; ++m) {
            ध्रुव __m512 vd1 = _mm512_set1_ps(GGML_CPU_FP16_गन्तव्यम्_FP32(A[m * lda].d));
            ध्रुव __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            यदि (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } अन्यथा {
                vsum = _mm512_set1_ps(0.f);
            }
            vsum = _mm512_fmadd_ps(vtile, _mm512_mul_ps(vd0, vd1), vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

प्रतिरूपम् <तार्किक is_acc>
संरचना acc_C<खण्डम्_q8_K, खण्डम्_q4_K, is_acc> {
    स्थिर शून्य apply(भिन्न * RESTRICT C, पूर्णाङ्क ldc, ध्रुव int32_t * RESTRICT tile, ध्रुव खण्डम्_q8_K * A, पूर्णाङ्क lda, ध्रुव शून्य * packed_B, पूर्णाङ्क nr) {
        ध्रुव uint8_t * scales = reinterpret_cast<ध्रुव uint8_t *>((ध्रुव वर्ण *)packed_B + (QK_K / 2) * TILE_N);
        ध्रुव uint8_t * mins = scales + 8 * TILE_N;
        ध्रुव ggml_half * d0 = reinterpret_cast<ध्रुव ggml_half *>(mins + 8 * TILE_N);
        ध्रुव ggml_half * dmin = d0 + TILE_N;

        ध्रुव __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ध्रुव __m256i *)d0));
        ध्रुव __m512 vdmin = _mm512_cvtph_ps(_mm256_loadu_si256((ध्रुव __m256i *)dmin));

        प्रति (पूर्णाङ्क m = 0; m < nr; ++m) {
            ध्रुव भिन्न d1 = A[m * lda].d;
            ध्रुव __m512 vd = _mm512_mul_ps(_mm512_set1_ps(d1), vd0);
            ध्रुव __m512 vdm = _mm512_mul_ps(_mm512_set1_ps(-d1), vdmin);
            ध्रुव __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            यदि (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } अन्यथा {
                vsum = _mm512_set1_ps(0.f);
            }

            ध्रुव __m256i q8sums = _mm256_loadu_si256((ध्रुव __m256i *)A[m * lda].bsums);
            ध्रुव __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));

            __m512i acc_m = _mm512_setzero_si512();
            प्रति (पूर्णाङ्क k = 0; k < 4; ++k) {
                __m512i vmask = _mm512_set1_epi32(k);
                __m512i va = _mm512_permutexvar_epi32(vmask, _mm512_castsi128_si512(q8s));
                __m512i vb = _mm512_cvtepi8_epi16(_mm256_loadu_si256((ध्रुव __m256i *)(mins + k * 32)));
                acc_m = _mm512_dpwssds_epi32(acc_m, va, vb);
            }

            vsum = _mm512_fmadd_ps(vtile, vd, vsum);
            vsum = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc_m), vdm, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

प्रतिरूपम् <तार्किक is_acc>
संरचना acc_C<खण्डम्_q8_K, खण्डम्_q5_K, is_acc> {
    स्थिर शून्य apply(भिन्न * RESTRICT C, पूर्णाङ्क ldc, ध्रुव int32_t * RESTRICT tile, ध्रुव खण्डम्_q8_K * A, पूर्णाङ्क lda, ध्रुव शून्य * packed_B, पूर्णाङ्क nr) {
        ध्रुव uint8_t * scales = reinterpret_cast<ध्रुव uint8_t *>((ध्रुव वर्ण *)packed_B + (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N);
        ध्रुव uint8_t * mins = scales + 8 * TILE_N;
        ध्रुव ggml_half * d0 = reinterpret_cast<ध्रुव ggml_half *>(mins + 8 * TILE_N);
        ध्रुव ggml_half * dmin = d0 + TILE_N;

        ध्रुव __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ध्रुव __m256i *)d0));
        ध्रुव __m512 vdmin = _mm512_cvtph_ps(_mm256_loadu_si256((ध्रुव __m256i *)dmin));

        प्रति (पूर्णाङ्क m = 0; m < nr; ++m) {
            ध्रुव भिन्न d1 = A[m * lda].d;
            ध्रुव __m512 vd = _mm512_mul_ps(_mm512_set1_ps(d1), vd0);
            ध्रुव __m512 vdm = _mm512_mul_ps(_mm512_set1_ps(-d1), vdmin);
            ध्रुव __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            यदि (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } अन्यथा {
                vsum = _mm512_set1_ps(0.f);
            }

            ध्रुव __m256i q8sums = _mm256_loadu_si256((ध्रुव __m256i *)A[m * lda].bsums);
            ध्रुव __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));

            __m512i acc_m = _mm512_setzero_si512();
            प्रति (पूर्णाङ्क k = 0; k < 4; ++k) {
                __m512i vmask = _mm512_set1_epi32(k);
                __m512i va = _mm512_permutexvar_epi32(vmask, _mm512_castsi128_si512(q8s));
                __m512i vb = _mm512_cvtepi8_epi16(_mm256_loadu_si256((ध्रुव __m256i *)(mins + k * 32)));
                acc_m = _mm512_dpwssds_epi32(acc_m, va, vb);
            }

            vsum = _mm512_fmadd_ps(vtile, vd, vsum);
            vsum = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc_m), vdm, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

प्रतिरूपम् <तार्किक is_acc>
संरचना acc_C<खण्डम्_q8_K, खण्डम्_q6_K, is_acc> {
    स्थिर शून्य apply(भिन्न * RESTRICT C, पूर्णाङ्क ldc, ध्रुव int32_t * RESTRICT tile, ध्रुव खण्डम्_q8_K * A, पूर्णाङ्क lda, ध्रुव शून्य * packed_B, पूर्णाङ्क nr) {
        ध्रुव uint8_t * scales = reinterpret_cast<ध्रुव uint8_t *>((ध्रुव वर्ण *)packed_B + (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N);
        ध्रुव ggml_half * d0 = reinterpret_cast<ध्रुव ggml_half *>(scales + 16 * TILE_N);

        ध्रुव __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ध्रुव __m256i *)d0));

        प्रति (पूर्णाङ्क m = 0; m < nr; ++m) {
            ध्रुव भिन्न d1 = A[m * lda].d;
            ध्रुव __m512 vd = _mm512_mul_ps(_mm512_set1_ps(d1), vd0);
            ध्रुव __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            यदि (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } अन्यथा {
                vsum = _mm512_set1_ps(0.f);
            }

            vsum = _mm512_fmadd_ps(vtile, vd, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

प्रतिरूपम् <तार्किक is_acc>
संरचना acc_C<खण्डम्_q8_K, खण्डम्_iq4_xs, is_acc> {
    स्थिर शून्य apply(भिन्न * RESTRICT C, पूर्णाङ्क ldc, ध्रुव int32_t * RESTRICT tile, ध्रुव खण्डम्_q8_K * A, पूर्णाङ्क lda, ध्रुव शून्य * packed_B, पूर्णाङ्क nr) {
        ध्रुव int8_t * scales = reinterpret_cast<ध्रुव int8_t *>((ध्रुव वर्ण *)packed_B + (QK_K / 2) * TILE_N);
        ध्रुव ggml_half * d0 = reinterpret_cast<ध्रुव ggml_half *>(scales + 8 * TILE_N);

        ध्रुव __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ध्रुव __m256i *)d0));

        प्रति (पूर्णाङ्क m = 0; m < nr; ++m) {
            ध्रुव भिन्न d1 = A[m * lda].d;
            ध्रुव __m512 vd = _mm512_mul_ps(_mm512_set1_ps(d1), vd0);
            ध्रुव __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            यदि (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } अन्यथा {
                vsum = _mm512_set1_ps(0.f);
            }

            vsum = _mm512_fmadd_ps(vtile, vd, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

प्रतिरूपम् <प्रकारनाम TB> constexpr पूर्णाङ्क get_quants_आकार();
प्रतिरूपम् <> constexpr पूर्णाङ्क get_quants_आकार<खण्डम्_q4_K>() { प्रत्यागम (QK_K / 2) * TILE_N; }
प्रतिरूपम् <> constexpr पूर्णाङ्क get_quants_आकार<खण्डम्_q5_K>() { प्रत्यागम (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N; }
प्रतिरूपम् <> constexpr पूर्णाङ्क get_quants_आकार<खण्डम्_q6_K>() { प्रत्यागम (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N; }
प्रतिरूपम् <> constexpr पूर्णाङ्क get_quants_आकार<खण्डम्_iq4_xs>() { प्रत्यागम (QK_K / 2) * TILE_N; }

// used for QKK format
प्रतिरूपम् <प्रकारनाम TB, तार्किक is_acc,
          प्रकारनाम std::enable_if<is_type_qkk<TB>::value, पूर्णाङ्क>::type = 0>
अन्तःस्थ शून्य scale_C(ध्रुव int32_t * RESTRICT tile, int32_t * RESTRICT sumi, ध्रुव शून्य * packed_B, पूर्णाङ्क k, पूर्णाङ्क nr) {
    ध्रुव uint8_t * scales = reinterpret_cast<ध्रुव uint8_t *>((ध्रुव वर्ण *)packed_B + get_quants_आकार<TB>());
    ध्रुव __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((ध्रुव __m128i *)(scales + k * TILE_N)));

    प्रति (पूर्णाङ्क m = 0; m < nr; ++m) {
        __m512i vsumi;
        यदि (is_acc) {
            vsumi = _mm512_loadu_si512(sumi + m * TILE_N);
        } अन्यथा {
            vsumi = _mm512_setzero_si512();
        }
        __m512i vtile = _mm512_loadu_si512(tile + m * TILE_N);
        vsumi = mm512_सङ्कलय_epi32(vsumi, _mm512_mullo_epi32(vtile, vscale));
        _mm512_storeu_si512((__m512i *)(sumi + m * TILE_N), vsumi);
    }
}

प्रतिरूपम् <प्रकारनाम TA, प्रकारनाम TB, प्रकारनाम TC, पूर्णाङ्क खण्डम्_M, पूर्णाङ्क खण्डम्_N, पूर्णाङ्क खण्डम्_K>
संरचना tinygemm_kernel_avx {
    स्थिर शून्य apply(पूर्णाङ्क K, ध्रुव TA * RESTRICT A, ध्रुव TB * RESTRICT B, TC * RESTRICT C, पूर्णाङ्क ldc) {
        GGML_अप्रयुक्तम्(K);
        GGML_अप्रयुक्तम्(A);
        GGML_अप्रयुक्तम्(B);
        GGML_अप्रयुक्तम्(C);
        GGML_अप्रयुक्तम्(ldc);
    }
};

प्रतिरूपम् <पूर्णाङ्क खण्डम्_M, पूर्णाङ्क खण्डम्_N, पूर्णाङ्क खण्डम्_K>
संरचना tinygemm_kernel_avx<भिन्न, ggml_fp16_t, भिन्न, खण्डम्_M, खण्डम्_N, खण्डम्_K> {
    स्थिर शून्य apply(पूर्णाङ्क K, ध्रुव भिन्न * RESTRICT A, ध्रुव ggml_fp16_t * RESTRICT B, भिन्न * RESTRICT C, पूर्णाङ्क ldc) {
        constexpr पूर्णाङ्क पङ्क्तिकाः_2 = खण्डम्_M;
        constexpr पूर्णाङ्क स्तम्भसङ्ख्या = खण्डम्_N;
        प्रतिज्ञा(खण्डम्_K == 16);

        __m512 va;
        __m512 vb[स्तम्भसङ्ख्या];
        __m512 vc[पङ्क्तिकाः_2 * स्तम्भसङ्ख्या];

        स्वचालित loadc = [&](स्वचालित idx) {
            vc[idx] = _mm512_setzero_ps();
        };
        Unroll<पङ्क्तिकाः_2 * स्तम्भसङ्ख्या>{}(loadc);

        स्वचालित compute = [&](स्वचालित idx, स्वचालित k) {
            constexpr पूर्णाङ्क row = idx / स्तम्भसङ्ख्या;
            constexpr पूर्णाङ्क col = idx % स्तम्भसङ्ख्या;

            यदि constexpr (col == 0) {
                va = _mm512_loadu_ps(A + row * K + k);
            }
            यदि constexpr (row == 0) {
                vb[col] =  _mm512_cvtph_ps(_mm256_loadu_si256((ध्रुव __m256i *)(B + col * K + k)));
            }
            vc[idx] = _mm512_fmadd_ps(va, vb[col], vc[idx]);
        };

        प्रति (पूर्णाङ्क k = 0; k < K; k += 16) {
            Unroll<पङ्क्तिकाः_2 * स्तम्भसङ्ख्या>{}(compute, k);
        }

        स्वचालित storec = [&](स्वचालित idx) {
            constexpr पूर्णाङ्क row = idx / स्तम्भसङ्ख्या;
            constexpr पूर्णाङ्क col = idx % स्तम्भसङ्ख्या;
            C[row * ldc + col] = mm512_reduce_सङ्कलय_ps(vc[idx]);
        };
        Unroll<पङ्क्तिकाः_2 * स्तम्भसङ्ख्या>{}(storec);
    }
};

#परिभाषय LAUNCH_TINYGEMM_KERNEL_AVX(MB_आकार, NB_आकार)                                \
    tinygemm_kernel_avx<भिन्न, type, भिन्न, MB_आकार, NB_आकार, blck_आकार>::apply(    \
        K, (ध्रुव भिन्न *)src1->दत्त + src1_विस्थापनम् + mb_start * K,                  \
        (ध्रुव type *)src0->दत्त + src0_विस्थापनम् + nb_start * K,                      \
        (भिन्न *)dst->दत्त + dst_विस्थापनम् + mb_start * ldc + nb_start, ldc)


// re-organize in the format {NB, KB, TILE_SIZE}:
#परिभाषय PACKED_सूचकांक(n, k, KB, tile_आकार) (n * KB + k) * tile_आकार

प्रतिरूपम्<प्रकारनाम TB, पूर्णाङ्क खण्डम्_K>
शून्य convert_B_packed_format(शून्य * RESTRICT packed_B, ध्रुव TB * RESTRICT B, पूर्णाङ्क N, पूर्णाङ्क K) {
    ध्रुव पूर्णाङ्क NB = N / TILE_N;
    ध्रुव पूर्णाङ्क KB = K / खण्डम्_K;
    ध्रुव पूर्णाङ्क TILE_आकार = get_tile_आकार<TB>();

    // parallel on NB should be enough
    parallel_for(NB, [&](पूर्णाङ्क begin, पूर्णाङ्क end) {
        प्रति (पूर्णाङ्क n = begin; n < end; ++n) {
            प्रति (पूर्णाङ्क k = 0; k < KB; ++k) {
                पूर्णाङ्क n0 = n * TILE_N;
                pack_B((वर्ण *)packed_B + PACKED_सूचकांक(n, k, KB, TILE_आकार), &B[n0 * KB + k], KB);
            }
        }
    });
}

प्रतिरूपम् <प्रकारनाम TA, प्रकारनाम TB, प्रकारनाम TC, पूर्णाङ्क खण्डम्_M, पूर्णाङ्क खण्डम्_N, पूर्णाङ्क खण्डम्_K>
संरचना tinygemm_kernel_vnni {};

प्रतिरूपम् <पूर्णाङ्क खण्डम्_M, पूर्णाङ्क खण्डम्_N, पूर्णाङ्क खण्डम्_K>
संरचना tinygemm_kernel_vnni<खण्डम्_q8_0, खण्डम्_q4_0, भिन्न, खण्डम्_M, खण्डम्_N, खण्डम्_K> {
    स्थिर शून्य apply(पूर्णाङ्क KB, ध्रुव शून्य * RESTRICT _A, ध्रुव शून्य * RESTRICT _B, भिन्न * RESTRICT C, पूर्णाङ्क ldc) {

        constexpr पूर्णाङ्क स्तम्भसङ्ख्या = खण्डम्_N / 16;
        ध्रुव पूर्णाङ्क TILE_आकार = TILE_N * परिमाण(खण्डम्_q4_0);

        ध्रुव खण्डम्_q8_0 * RESTRICT A = static_cast<ध्रुव खण्डम्_q8_0 *>(_A);
        ध्रुव वर्ण * RESTRICT B = static_cast<ध्रुव वर्ण *>(_B);

        __m512i va[8];
        __m512 vc[स्तम्भसङ्ख्या];
        __m512 vd1;

        // sum of offsets, shared across COLS
        //
        // avx512-vnni does not have `_mm512_dpbssd_epi32`,
        // need to transform ss to us:
        //   a * (b - 8) is equivalent to b * a - 8 * a
        //   s    u   u                   u   s   u   s
        //
        __m512i vcomp;

        ध्रुव __m512i off = _mm512_set1_epi8(8);
        ध्रुव __m512i नीचः_आवरणम् = _mm512_set1_epi8(0xF);

        स्वचालित loadc = [&](स्वचालित col) {
            vc[col] = _mm512_setzero_ps();
        };
        Unroll<स्तम्भसङ्ख्या>{}(loadc);

        स्वचालित compute = [&](स्वचालित col, स्वचालित i) {
            // load a and compute compensation
            यदि constexpr (col == 0) {
                ध्रुव int32_t * a_ptr = reinterpret_cast<ध्रुव int32_t *>(A[0 * KB + i].qs);
                vcomp = _mm512_setzero_si512();
                प्रति (पूर्णाङ्क k = 0; k < 8; ++k) {
                    va[k] = _mm512_set1_epi32(a_ptr[k]);
                    vcomp = _mm512_dpbusd_epi32(vcomp, off, va[k]);
                }
                vd1 = _mm512_set1_ps(GGML_CPU_FP16_गन्तव्यम्_FP32(A[0 * KB + i].d));
            }

            // load b
            __m512i vsum = _mm512_setzero_si512();
            ध्रुव वर्ण * b_ptr = B + PACKED_सूचकांक(col, i, KB, TILE_आकार);
            प्रति (पूर्णाङ्क k = 0; k < 8; k += 2) {
                __m512i अष्टकानि = _mm512_loadu_si512((ध्रुव __m512i *)(b_ptr + k * 32));
                __m512i vb0 = _mm512_and_si512(अष्टकानि, नीचः_आवरणम्);
                vsum = _mm512_dpbusd_epi32(vsum, vb0, va[k + 0]);
                __m512i vb1 = _mm512_and_si512(_mm512_srli_epi16(अष्टकानि, 4), नीचः_आवरणम्);
                vsum = _mm512_dpbusd_epi32(vsum, vb1, va[k + 1]);
            }
            ध्रुव पूर्णाङ्क विस्थापनम् = TILE_N * TILE_K / 2;
            ध्रुव __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ध्रुव __m256i *)(b_ptr + विस्थापनम्)));
            vsum = _mm512_sub_epi32(vsum, vcomp);

            vc[col] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vsum), _mm512_mul_ps(vd0, vd1), vc[col]);
        };

        प्रति (पूर्णाङ्क i = 0; i < KB; ++i) {
            Unroll<स्तम्भसङ्ख्या>{}(compute, i);
        }

        //store to C
        स्वचालित storec = [&](स्वचालित col) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + col * 16), vc[col]);
        };
        Unroll<स्तम्भसङ्ख्या>{}(storec);
    }
};

प्रतिरूपम् <पूर्णाङ्क खण्डम्_N, पूर्णाङ्क खण्डम्_K>
संरचना tinygemm_kernel_vnni<खण्डम्_q8_1, खण्डम्_q4_1, भिन्न, 1, खण्डम्_N, खण्डम्_K> {
    स्थिर शून्य apply(पूर्णाङ्क KB, ध्रुव शून्य * RESTRICT _A, ध्रुव शून्य * RESTRICT _B, भिन्न * RESTRICT C, पूर्णाङ्क ldc) {

        constexpr पूर्णाङ्क स्तम्भसङ्ख्या = खण्डम्_N / 16;
        ध्रुव पूर्णाङ्क TILE_आकार = TILE_N * परिमाण(खण्डम्_q4_1);

        ध्रुव खण्डम्_q8_1 * RESTRICT A = static_cast<ध्रुव खण्डम्_q8_1 *>(_A);
        ध्रुव वर्ण * RESTRICT B = static_cast<ध्रुव वर्ण *>(_B);

        __m512i va[8];
        __m512i vb[8];
        __m512 vc[स्तम्भसङ्ख्या];
        __m512 vd1, vs1;

        ध्रुव __m512i नीचः_आवरणम् = _mm512_set1_epi8(0xF);

        स्वचालित loadc = [&](स्वचालित col) {
            vc[col] = _mm512_setzero_ps();
        };
        Unroll<स्तम्भसङ्ख्या>{}(loadc);

        स्वचालित compute = [&](स्वचालित col, स्वचालित i) {
            // load a
            यदि constexpr (col == 0) {
                ध्रुव int32_t * a_ptr = reinterpret_cast<ध्रुव int32_t *>(A[0 * KB + i].qs);
                प्रति (पूर्णाङ्क k = 0; k < 8; ++k) {
                    va[k] = _mm512_set1_epi32(a_ptr[k]);
                }
                vd1 = _mm512_set1_ps(GGML_CPU_FP16_गन्तव्यम्_FP32(A[0 * KB + i].d));
                vs1 = _mm512_set1_ps(GGML_CPU_FP16_गन्तव्यम्_FP32(A[0 * KB + i].s));
            }

            // load b
            ध्रुव वर्ण * b_ptr = B + PACKED_सूचकांक(col, i, KB, TILE_आकार);
            प्रति (पूर्णाङ्क k = 0; k < 8; k += 2) {
                __m512i अष्टकानि = _mm512_loadu_si512((ध्रुव __m512i *)(b_ptr + k * 32));
                vb[k + 0] = _mm512_and_si512(अष्टकानि, नीचः_आवरणम्);
                vb[k + 1] = _mm512_and_si512(_mm512_srli_epi16(अष्टकानि, 4), नीचः_आवरणम्);
            }
            ध्रुव पूर्णाङ्क विस्थापनम् = TILE_N * TILE_K / 2;
            ध्रुव __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ध्रुव __m256i *)(b_ptr + विस्थापनम्)));
            ध्रुव __m512 vm0 = _mm512_cvtph_ps(_mm256_loadu_si256((ध्रुव __m256i *)(b_ptr + विस्थापनम् + TILE_N * परिमाण(ggml_half))));

            __m512i vsum = _mm512_setzero_si512();
            प्रति (पूर्णाङ्क k = 0; k < 8; ++k) {
                vsum = _mm512_dpbusd_epi32(vsum, vb[k], va[k]);
            }

            vc[col] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vsum), _mm512_mul_ps(vd0, vd1), vc[col]);
            vc[col] = _mm512_fmadd_ps(vm0, vs1, vc[col]);
        };

        प्रति (पूर्णाङ्क i = 0; i < KB; ++i) {
            Unroll<स्तम्भसङ्ख्या>{}(compute, i);
        }

        //store to C
        स्वचालित storec = [&](स्वचालित col) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + col * 16), vc[col]);
        };
        Unroll<स्तम्भसङ्ख्या>{}(storec);
    }
};

प्रतिरूपम् <पूर्णाङ्क खण्डम्_M, पूर्णाङ्क खण्डम्_N, पूर्णाङ्क खण्डम्_K>
संरचना tinygemm_kernel_vnni<खण्डम्_q8_0, खण्डम्_q8_0, भिन्न, खण्डम्_M, खण्डम्_N, खण्डम्_K> {
    स्थिर शून्य apply(पूर्णाङ्क KB, ध्रुव शून्य * RESTRICT _A, ध्रुव शून्य * RESTRICT _B, भिन्न * RESTRICT C, पूर्णाङ्क ldc) {

        constexpr पूर्णाङ्क स्तम्भसङ्ख्या = खण्डम्_N / 16;
        ध्रुव पूर्णाङ्क TILE_आकार = TILE_N * परिमाण(खण्डम्_q8_0) + TILE_N * परिमाण(int32_t);

        ध्रुव खण्डम्_q8_0 * RESTRICT A = static_cast<ध्रुव खण्डम्_q8_0 *>(_A);
        ध्रुव वर्ण * RESTRICT B = static_cast<ध्रुव वर्ण *>(_B);

        __m512i va[8];
        __m512i vb[8];
        __m512 vc[स्तम्भसङ्ख्या];
        __m512 vd1;

        // Notes: s8s8 igemm compensation in avx512-vnni
        // change s8s8 to u8s8 with compensate
        //   a * b = (a + 128) * b - 128 * b
        //   s   s       u       s    u    s
        //
        // (128 * b is pre-computed when packing B to vnni formats)
        //
        ध्रुव __m512i off = _mm512_set1_epi8(static_cast<वर्ण>(0x80));

        स्वचालित loadc = [&](स्वचालित col) {
            vc[col] = _mm512_setzero_ps();
        };
        Unroll<स्तम्भसङ्ख्या>{}(loadc);

        स्वचालित compute = [&](स्वचालित col, स्वचालित i) {
            // load a and add offset 128
            यदि constexpr (col == 0) {
                ध्रुव int32_t * a_ptr = reinterpret_cast<ध्रुव int32_t *>(A[0 * KB + i].qs);
                प्रति (पूर्णाङ्क k = 0; k < 8; ++k) {
                    va[k] = _mm512_set1_epi32(a_ptr[k]);
                    va[k] = mm512_सङ्कलय_epi8(va[k], off);
                }
                vd1 = _mm512_set1_ps(GGML_CPU_FP16_गन्तव्यम्_FP32(A[0 * KB + i].d));
            }

            // load b
            ध्रुव वर्ण * b_ptr = B + PACKED_सूचकांक(col, i, KB, TILE_आकार);
            प्रति (पूर्णाङ्क k = 0; k < 8; ++k) {
                vb[k] = _mm512_loadu_si512((ध्रुव __m512i *)(b_ptr + k * 64));
            }
            ध्रुव पूर्णाङ्क विस्थापनम् = TILE_N * TILE_K;
            ध्रुव __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ध्रुव __m256i *)(b_ptr + विस्थापनम्)));
            ध्रुव पूर्णाङ्क offset2 = TILE_N * TILE_K + TILE_N * परिमाण(ggml_half);
            ध्रुव __m512i vcomp = _mm512_loadu_si512((ध्रुव __m512i *)(b_ptr + offset2));

            __m512i vsum = _mm512_setzero_si512();
            प्रति (पूर्णाङ्क k = 0; k < 8; ++k) {
                vsum = _mm512_dpbusd_epi32(vsum, va[k], vb[k]);
            }
            vsum = _mm512_sub_epi32(vsum, vcomp);

            vc[col] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vsum), _mm512_mul_ps(vd0, vd1), vc[col]);
        };

        प्रति (पूर्णाङ्क i = 0; i < KB; ++i) {
            Unroll<स्तम्भसङ्ख्या>{}(compute, i);
        }

        //store to C
        स्वचालित storec = [&](स्वचालित col) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + col * 16), vc[col]);
        };
        Unroll<स्तम्भसङ्ख्या>{}(storec);
    }
};

प्रतिरूपम् <पूर्णाङ्क खण्डम्_M, पूर्णाङ्क खण्डम्_N, पूर्णाङ्क खण्डम्_K>
संरचना tinygemm_kernel_vnni<खण्डम्_q8_K, खण्डम्_q4_K, भिन्न, खण्डम्_M, खण्डम्_N, खण्डम्_K> {
    स्थिर शून्य apply(पूर्णाङ्क KB, ध्रुव शून्य * RESTRICT _A, ध्रुव शून्य * RESTRICT _B, भिन्न * RESTRICT C, पूर्णाङ्क ldc) {

        constexpr पूर्णाङ्क स्तम्भसङ्ख्या = खण्डम्_N / 16;
        ध्रुव पूर्णाङ्क TILE_आकार = TILE_N * परिमाण(खण्डम्_q4_K) + TILE_N * 4;

        ध्रुव खण्डम्_q8_K * RESTRICT A = static_cast<ध्रुव खण्डम्_q8_K *>(_A);
        ध्रुव वर्ण * RESTRICT B = static_cast<ध्रुव वर्ण *>(_B);

        // a.qs:   8 groups, 32 bytes each group (m256i)
        __m512i va[8];
        // a.bsum: 8 groups,  2 bytes each group (m128i)
        __m512i va_bsum;
        __m512 vc[स्तम्भसङ्ख्या];
        __m512 vd1;

        // packed_B:
        ध्रुव पूर्णाङ्क विस्थापनम्_scales = (QK_K / 2) * TILE_N;
        ध्रुव पूर्णाङ्क विस्थापनम्_mins   = (QK_K / 2) * TILE_N +  8 * TILE_N;
        ध्रुव पूर्णाङ्क विस्थापनम्_d0     = (QK_K / 2) * TILE_N + 16 * TILE_N;
        ध्रुव पूर्णाङ्क विस्थापनम्_dmin   = (QK_K / 2) * TILE_N + 16 * TILE_N + TILE_N * परिमाण(ggml_half);

        ध्रुव __m512i नीचः_आवरणम् = _mm512_set1_epi8(0xF);

        स्वचालित loadc = [&](स्वचालित col) {
            vc[col] = _mm512_setzero_ps();
        };
        Unroll<स्तम्भसङ्ख्या>{}(loadc);

        // Notes: vnni formats in QK_K
        //   a) quants vnni format
        //     int8  {k/4, n, 4}, viewed as 2d {k/4, 4n}, k = 32
        //     from {16, 32} to {8, 64}
        //
        //   b) min vnni format
        //     int16 {k/2, n, 2}, viewed as 2d {k/2, 2n}, k = 8
        //     from {16,  8} to {4, 32}
        //
        स्वचालित compute = [&](स्वचालित col, स्वचालित i) {
            // load a
            यदि constexpr (col == 0) {
                प्रति (पूर्णाङ्क k_गणः = 0; k_गणः < QK_K / 32; ++k_गणः) {
                    va[k_गणः] = _mm512_castsi256_si512(_mm256_loadu_si256((ध्रुव __m256i *)(A[0 * KB + i].qs + k_गणः * 32)));
                }
                ध्रुव __m256i q8sums = _mm256_loadu_si256((ध्रुव __m256i *)A[0 * KB + i].bsums);
                ध्रुव __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
                va_bsum = _mm512_castsi128_si512(q8s);
                vd1 = _mm512_set1_ps(A[0 * KB + i].d);
            }

            // step 1: accumultate the quants
            __m512i acc = _mm512_setzero_si512();
            ध्रुव वर्ण * b_ptr = B + PACKED_सूचकांक(col, i, KB, TILE_आकार);
            ध्रुव वर्ण * b_qs  = b_ptr;
            प्रति (पूर्णाङ्क k_गणः = 0; k_गणः < QK_K / 32; ++k_गणः) {
                __m512i vsum = _mm512_setzero_si512();
                प्रति (पूर्णाङ्क k = 0; k < 8; k += 2) {
                    __m512i va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(k + 0), va[k_गणः]);
                    __m512i va1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(k + 1), va[k_गणः]);

                    __m512i अष्टकानि = _mm512_loadu_si512((ध्रुव __m512i *)b_qs);
                    __m512i vb0 = _mm512_and_si512(अष्टकानि, नीचः_आवरणम्);
                    vsum = _mm512_dpbusd_epi32(vsum, vb0, va0);
                    __m512i vb1 = _mm512_and_si512(_mm512_srli_epi16(अष्टकानि, 4), नीचः_आवरणम्);
                    vsum = _mm512_dpbusd_epi32(vsum, vb1, va1);

                    b_qs += 64;
                }
                // vacc += scale * (q8 @ q4)
                ध्रुव __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((ध्रुव __m128i *)(b_ptr + विस्थापनम्_scales + k_गणः * TILE_N)));
                acc = mm512_सङ्कलय_epi32(acc, _mm512_mullo_epi32(vsum, vscale));
            }
            ध्रुव __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ध्रुव __m256i *)(b_ptr + विस्थापनम्_d0)));
            vc[col] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(vd0, vd1), vc[col]);

            // step 2: accumulate the mins
            __m512i acc_m = _mm512_setzero_si512();
            प्रति (पूर्णाङ्क k = 0; k < 4; ++k) {
                __m512i vmask = _mm512_set1_epi32(k);
                __m512i va = _mm512_permutexvar_epi32(vmask, va_bsum);
                __m512i vb = _mm512_cvtepi8_epi16(_mm256_loadu_si256((ध्रुव __m256i *)(b_ptr + विस्थापनम्_mins + k * 32)));
                acc_m = _mm512_dpwssds_epi32(acc_m, va, vb);
            }
            ध्रुव __m512 vdmin = _mm512_cvtph_ps(_mm256_loadu_si256((ध्रुव __m256i *)(b_ptr + विस्थापनम्_dmin)));
            vc[col] = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(acc_m), _mm512_mul_ps(vdmin, vd1), vc[col]);
        };

        प्रति (पूर्णाङ्क i = 0; i < KB; ++i) {
            Unroll<स्तम्भसङ्ख्या>{}(compute, i);
        }

        //store to C
        स्वचालित storec = [&](स्वचालित col) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + col * 16), vc[col]);
        };
        Unroll<स्तम्भसङ्ख्या>{}(storec);
    }
};

प्रतिरूपम् <पूर्णाङ्क खण्डम्_M, पूर्णाङ्क खण्डम्_N, पूर्णाङ्क खण्डम्_K>
संरचना tinygemm_kernel_vnni<खण्डम्_q8_K, खण्डम्_q5_K, भिन्न, खण्डम्_M, खण्डम्_N, खण्डम्_K> {
    स्थिर शून्य apply(पूर्णाङ्क KB, ध्रुव शून्य * RESTRICT _A, ध्रुव शून्य * RESTRICT _B, भिन्न * RESTRICT C, पूर्णाङ्क ldc) {

        constexpr पूर्णाङ्क स्तम्भसङ्ख्या = खण्डम्_N / 16;
        ध्रुव पूर्णाङ्क TILE_आकार = TILE_N * परिमाण(खण्डम्_q5_K) + TILE_N * 4;

        ध्रुव खण्डम्_q8_K * RESTRICT A = static_cast<ध्रुव खण्डम्_q8_K *>(_A);
        ध्रुव वर्ण * RESTRICT B = static_cast<ध्रुव वर्ण *>(_B);

        // a.qs:   8 groups, 32 bytes each group (m256i)
        __m512i va[8];
        // a.bsum: 8 groups,  2 bytes each group (m128i)
        __m512i va_bsum;
        __m512 vc[स्तम्भसङ्ख्या];
        __m512 vd1;

        // packed_B:
        ध्रुव पूर्णाङ्क विस्थापनम्_qh     = (QK_K / 2) * TILE_N;
        ध्रुव पूर्णाङ्क विस्थापनम्_scales = (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N;
        ध्रुव पूर्णाङ्क विस्थापनम्_mins   = (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N +  8 * TILE_N;
        ध्रुव पूर्णाङ्क विस्थापनम्_d0     = (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N + 16 * TILE_N;
        ध्रुव पूर्णाङ्क विस्थापनम्_dmin   = (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N + 16 * TILE_N + TILE_N * परिमाण(ggml_half);

        ध्रुव __m512i नीचः_आवरणम् = _mm512_set1_epi8(0xF);

        स्वचालित loadc = [&](स्वचालित col) {
            vc[col] = _mm512_setzero_ps();
        };
        Unroll<स्तम्भसङ्ख्या>{}(loadc);

        // Q5_K and Q4_K shares the same vnni formats, refer to notes above.
        स्वचालित compute = [&](स्वचालित col, स्वचालित i) {
            // load a
            यदि constexpr (col == 0) {
                प्रति (पूर्णाङ्क k_गणः = 0; k_गणः < QK_K / 32; ++k_गणः) {
                    va[k_गणः] = _mm512_castsi256_si512(_mm256_loadu_si256((ध्रुव __m256i *)(A[0 * KB + i].qs + k_गणः * 32)));
                }
                ध्रुव __m256i q8sums = _mm256_loadu_si256((ध्रुव __m256i *)A[0 * KB + i].bsums);
                ध्रुव __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
                va_bsum = _mm512_castsi128_si512(q8s);
                vd1 = _mm512_set1_ps(A[0 * KB + i].d);
            }

            // step 1: accumultate the quants
            __m512i acc = _mm512_setzero_si512();
            ध्रुव वर्ण * b_ptr = B + PACKED_सूचकांक(col, i, KB, TILE_आकार);
            ध्रुव वर्ण * b_qs  = b_ptr;
            ध्रुव वर्ण * b_qh  = b_ptr + विस्थापनम्_qh;
            प्रति (पूर्णाङ्क k_गणः = 0; k_गणः < QK_K / 32; ++k_गणः) {
                __m512i vsum = _mm512_setzero_si512();
                __m512i hmask0 = _mm512_set1_epi8(0x1);
                __m512i hmask1 = _mm512_set1_epi8(0x2);
                __m512i hbits = _mm512_loadu_si512((ध्रुव __m512i *)(b_qh + k_गणः * 64));
                प्रति (पूर्णाङ्क k = 0; k < 8; k += 2) {
                    __m512i va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(k + 0), va[k_गणः]);
                    __m512i va1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(k + 1), va[k_गणः]);

                    __m512i अष्टकानि = _mm512_loadu_si512((ध्रुव __m512i *)b_qs);
                    __m512i vb0 = _mm512_and_si512(अष्टकानि, नीचः_आवरणम्);
                    __m512i vb1 = _mm512_and_si512(_mm512_srli_epi16(अष्टकानि, 4), नीचः_आवरणम्);

                    __m512i vh0 = _mm512_slli_epi16(_mm512_srli_epi16(_mm512_and_si512(hbits, hmask0), k), 4);
                    __m512i vh1 = _mm512_slli_epi16(_mm512_srli_epi16(_mm512_and_si512(hbits, hmask1), k + 1), 4);

                    hmask0 = _mm512_slli_epi16(hmask0, 2);
                    hmask1 = _mm512_slli_epi16(hmask1, 2);
                    vb0 = mm512_सङ्कलय_epi8(vb0, vh0);
                    vb1 = mm512_सङ्कलय_epi8(vb1, vh1);

                    vsum = _mm512_dpbusd_epi32(vsum, vb0, va0);
                    vsum = _mm512_dpbusd_epi32(vsum, vb1, va1);

                    b_qs += 64;
                }
                // vacc += scale * (q8 @ q5)
                ध्रुव __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((ध्रुव __m128i *)(b_ptr + विस्थापनम्_scales + k_गणः * TILE_N)));
                acc = mm512_सङ्कलय_epi32(acc, _mm512_mullo_epi32(vsum, vscale));
            }
            ध्रुव __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ध्रुव __m256i *)(b_ptr + विस्थापनम्_d0)));
            vc[col] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(vd0, vd1), vc[col]);

            // step 2: accumulate the mins
            __m512i acc_m = _mm512_setzero_si512();
            प्रति (पूर्णाङ्क k = 0; k < 4; ++k) {
                __m512i vmask = _mm512_set1_epi32(k);
                __m512i va = _mm512_permutexvar_epi32(vmask, va_bsum);
                __m512i vb = _mm512_cvtepi8_epi16(_mm256_loadu_si256((ध्रुव __m256i *)(b_ptr + विस्थापनम्_mins + k * 32)));
                acc_m = _mm512_dpwssds_epi32(acc_m, va, vb);
            }
            ध्रुव __m512 vdmin = _mm512_cvtph_ps(_mm256_loadu_si256((ध्रुव __m256i *)(b_ptr + विस्थापनम्_dmin)));
            vc[col] = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(acc_m), _mm512_mul_ps(vdmin, vd1), vc[col]);
        };

        प्रति (पूर्णाङ्क i = 0; i < KB; ++i) {
            Unroll<स्तम्भसङ्ख्या>{}(compute, i);
        }

        //store to C
        स्वचालित storec = [&](स्वचालित col) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + col * 16), vc[col]);
        };
        Unroll<स्तम्भसङ्ख्या>{}(storec);
    }
};

प्रतिरूपम् <पूर्णाङ्क खण्डम्_M, पूर्णाङ्क खण्डम्_N, पूर्णाङ्क खण्डम्_K>
संरचना tinygemm_kernel_vnni<खण्डम्_q8_K, खण्डम्_q6_K, भिन्न, खण्डम्_M, खण्डम्_N, खण्डम्_K> {
    स्थिर शून्य apply(पूर्णाङ्क KB, ध्रुव शून्य * RESTRICT _A, ध्रुव शून्य * RESTRICT _B, भिन्न * RESTRICT C, पूर्णाङ्क ldc) {

        constexpr पूर्णाङ्क स्तम्भसङ्ख्या = खण्डम्_N / 16;
        ध्रुव पूर्णाङ्क TILE_आकार = TILE_N * परिमाण(खण्डम्_q6_K);

        ध्रुव खण्डम्_q8_K * RESTRICT A = static_cast<ध्रुव खण्डम्_q8_K *>(_A);
        ध्रुव वर्ण * RESTRICT B = static_cast<ध्रुव वर्ण *>(_B);

        // load the 256 bytes from A to 4 avx512 vectors
        __m512i va[4];
        __m512 vc[स्तम्भसङ्ख्या];
        __m512 vd1;

        // packed_B:
        ध्रुव पूर्णाङ्क विस्थापनम्_qh     = (QK_K / 2) * TILE_N;
        ध्रुव पूर्णाङ्क विस्थापनम्_scales = (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N;
        ध्रुव पूर्णाङ्क विस्थापनम्_d0     = (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N + 16 * TILE_N;

        // compensation
        __m512i vcomp;

        ध्रुव __m512i m32s = _mm512_set1_epi32(32);
        ध्रुव __m512i नीचः_आवरणम् = _mm512_set1_epi8(0xF);

        स्वचालित loadc = [&](स्वचालित col) {
            vc[col] = _mm512_setzero_ps();
        };
        Unroll<स्तम्भसङ्ख्या>{}(loadc);

        स्वचालित compute = [&](स्वचालित col, स्वचालित i) {
            यदि constexpr (col == 0) {
                // load a
                va[0] = _mm512_loadu_si512((ध्रुव __m512i *)(A[0 * KB + i].qs +   0));
                va[1] = _mm512_loadu_si512((ध्रुव __m512i *)(A[0 * KB + i].qs +  64));
                va[2] = _mm512_loadu_si512((ध्रुव __m512i *)(A[0 * KB + i].qs + 128));
                va[3] = _mm512_loadu_si512((ध्रुव __m512i *)(A[0 * KB + i].qs + 192));

                ध्रुव __m256i q8sums = _mm256_loadu_si256((ध्रुव __m256i *)A[0 * KB + i].bsums);
                vcomp = _mm512_mullo_epi32(_mm512_cvtepi16_epi32(q8sums), m32s);
                vd1 = _mm512_set1_ps(A[0 * KB + i].d);
            }

            // accmulate the quants
            __m512i acc = _mm512_setzero_si512();
            ध्रुव वर्ण * b_ptr = B + PACKED_सूचकांक(col, i, KB, TILE_आकार);
            ध्रुव वर्ण * b_qs = b_ptr;
            ध्रुव वर्ण * b_qh = b_ptr + विस्थापनम्_qh;
            पूर्णाङ्क आवरणम् = 0;
            प्रति (पूर्णाङ्क k_गणः = 0; k_गणः < QK_K / 16; ++k_गणः) {
                पूर्णाङ्क r = k_गणः >> 2;
                __m512i va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(आवरणम्++), va[r]);
                __m512i va1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(आवरणम्++), va[r]);

                __m512i vsum = _mm512_setzero_si512();
                __m512i hmask = _mm512_set1_epi8(0x3);

                __m512i अष्टकानि = _mm512_loadu_si512(b_qs);
                __m512i hbits = _mm512_loadu_si512(b_qh);
                __m512i vb0 = _mm512_and_si512(अष्टकानि, नीचः_आवरणम्);
                __m512i vb1 = _mm512_and_si512(_mm512_srli_epi16(अष्टकानि, 4), नीचः_आवरणम्);
                __m512i vh0 = _mm512_slli_epi16(_mm512_and_si512(hbits, hmask), 4);
                __m512i vh1 = _mm512_slli_epi16(_mm512_and_si512(hbits, _mm512_slli_epi16(hmask, 2)), 2);

                vb0 = mm512_सङ्कलय_epi8(vb0, vh0);
                vb1 = mm512_सङ्कलय_epi8(vb1, vh1);
                vsum = _mm512_dpbusd_epi32(vsum, vb0, va0);
                vsum = _mm512_dpbusd_epi32(vsum, vb1, va1);
                b_qs += 64;

                va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(आवरणम्++), va[r]);
                va1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(आवरणम्++), va[r]);

                अष्टकानि = _mm512_loadu_si512(b_qs);
                vb0 = _mm512_and_si512(अष्टकानि, नीचः_आवरणम्);
                vb1 = _mm512_and_si512(_mm512_srli_epi16(अष्टकानि, 4), नीचः_आवरणम्);
                vh0 =                   _mm512_and_si512(hbits, _mm512_slli_epi16(hmask, 4));
                vh1 = _mm512_srli_epi16(_mm512_and_si512(hbits, _mm512_slli_epi16(hmask, 6)), 2);
                vb0 = mm512_सङ्कलय_epi8(vb0, vh0);
                vb1 = mm512_सङ्कलय_epi8(vb1, vh1);
                vsum = _mm512_dpbusd_epi32(vsum, vb0, va0);
                vsum = _mm512_dpbusd_epi32(vsum, vb1, va1);
                b_qs += 64;
                b_qh += 64;

                // B * A - 32 * A
                __m512i vmask = _mm512_set1_epi32(k_गणः);
                vsum = _mm512_sub_epi32(vsum, _mm512_permutexvar_epi32(vmask, vcomp));

                // vacc += scale * (q8 @ q6)
                ध्रुव __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((ध्रुव __m128i *)(b_ptr + विस्थापनम्_scales + k_गणः * TILE_N)));
                acc = mm512_सङ्कलय_epi32(acc, _mm512_mullo_epi32(vsum, vscale));
            }
            ध्रुव __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ध्रुव __m256i *)(b_ptr + विस्थापनम्_d0)));
            vc[col] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(vd0, vd1), vc[col]);
        };

        प्रति (पूर्णाङ्क i = 0; i < KB; ++i) {
            Unroll<स्तम्भसङ्ख्या>{}(compute, i);
        }

        //store to C
        स्वचालित storec = [&](पूर्णाङ्क col) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + col * 16), vc[col]);
        };
        Unroll<स्तम्भसङ्ख्या>{}(storec);
    }
};

प्रतिरूपम् <पूर्णाङ्क खण्डम्_M, पूर्णाङ्क खण्डम्_N, पूर्णाङ्क खण्डम्_K>
संरचना tinygemm_kernel_vnni<खण्डम्_q8_K, खण्डम्_iq4_xs, भिन्न, खण्डम्_M, खण्डम्_N, खण्डम्_K> {
    स्थिर शून्य apply(पूर्णाङ्क KB, ध्रुव शून्य * RESTRICT _A, ध्रुव शून्य * RESTRICT _B, भिन्न * RESTRICT C, पूर्णाङ्क ldc) {

        constexpr पूर्णाङ्क स्तम्भसङ्ख्या = खण्डम्_N / 16;
        ध्रुव पूर्णाङ्क TILE_आकार = TILE_N * परिमाण(खण्डम्_iq4_xs) + TILE_N * 2;

        ध्रुव खण्डम्_q8_K * RESTRICT A = static_cast<ध्रुव खण्डम्_q8_K *>(_A);
        ध्रुव वर्ण * RESTRICT B = static_cast<ध्रुव वर्ण *>(_B);

        // load the 256 bytes from A to 4 avx512 vectors
        __m512i va[4];
        __m512 vc[स्तम्भसङ्ख्या];
        __m512 vd1;

        // packed_B:
        ध्रुव पूर्णाङ्क विस्थापनम्_scales = (QK_K / 2) * TILE_N ;
        ध्रुव पूर्णाङ्क विस्थापनम्_d0     = (QK_K / 2) * TILE_N + 8 * TILE_N;

        // compensation
        __m512i vcomp;

        ध्रुव __m256i m128s = _mm256_set1_epi16(128);
        ध्रुव __m512i नीचः_आवरणम् = _mm512_set1_epi8(0xF);

        ध्रुव __m512i values128 = mm512_समुच्चयः_epi8(
            113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
            113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
            113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
            113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127
        );
        ध्रुव __m512i off = _mm512_set1_epi8(static_cast<वर्ण>(0x80));
        ध्रुव __m512i values256 = mm512_सङ्कलय_epi8(values128, off);

        स्वचालित loadc = [&](स्वचालित col) {
            vc[col] = _mm512_setzero_ps();
        };
        Unroll<स्तम्भसङ्ख्या>{}(loadc);

        स्वचालित compute = [&](स्वचालित col, स्वचालित i) {
            यदि constexpr (col == 0) {
                // load a
                va[0] = _mm512_loadu_si512((ध्रुव __m512i *)(A[0 * KB + i].qs +   0));
                va[1] = _mm512_loadu_si512((ध्रुव __m512i *)(A[0 * KB + i].qs +  64));
                va[2] = _mm512_loadu_si512((ध्रुव __m512i *)(A[0 * KB + i].qs + 128));
                va[3] = _mm512_loadu_si512((ध्रुव __m512i *)(A[0 * KB + i].qs + 192));

                // compensation: 128 * A
                ध्रुव __m256i q8sums = _mm256_loadu_si256((ध्रुव __m256i *)A[0 * KB + i].bsums);
                vcomp = _mm512_castsi256_si512(_mm256_madd_epi16(q8sums, m128s));
                vd1 = _mm512_set1_ps(A[0 * KB + i].d);
            }

            // accmulate the quants
            __m512i acc = _mm512_setzero_si512();
            ध्रुव वर्ण * b_ptr = B + PACKED_सूचकांक(col, i, KB, TILE_आकार);
            ध्रुव वर्ण * b_qs = b_ptr;
            पूर्णाङ्क आवरणम् = 0;
            प्रति (पूर्णाङ्क k_गणः = 0; k_गणः < QK_K / 32; ++k_गणः) {
                पूर्णाङ्क r = k_गणः >> 1;
                __m512i vmask = _mm512_set1_epi32(k_गणः);
                __m512i vsum = _mm512_setzero_si512();
                प्रति (पूर्णाङ्क k = 0; k < 8; k += 2) {
                    __m512i va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(आवरणम्++), va[r]);
                    __m512i va1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(आवरणम्++), va[r]);

                    __m512i अष्टकानि = _mm512_loadu_si512(b_qs);
                    __m512i vb0 = _mm512_shuffle_epi8(values256, _mm512_and_si512(अष्टकानि, नीचः_आवरणम्));
                    __m512i vb1 = _mm512_shuffle_epi8(values256, _mm512_and_si512(_mm512_srli_epi16(अष्टकानि, 4), नीचः_आवरणम्));

                    vsum = _mm512_dpbusd_epi32(vsum, vb0, va0);
                    vsum = _mm512_dpbusd_epi32(vsum, vb1, va1);
                    b_qs += 64;
                }
                // (B + 128) * A - 128 * A
                vsum = _mm512_sub_epi32(vsum, _mm512_permutexvar_epi32(vmask, vcomp));

                // vacc += scale * (q8 @ q4)
                ध्रुव __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((ध्रुव __m128i *)(b_ptr + विस्थापनम्_scales + k_गणः * TILE_N)));
                acc = mm512_सङ्कलय_epi32(acc, _mm512_mullo_epi32(vsum, vscale));
            }
            ध्रुव __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ध्रुव __m256i *)(b_ptr + विस्थापनम्_d0)));
            vc[col] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(vd0, vd1), vc[col]);
        };

        प्रति (पूर्णाङ्क i = 0; i < KB; ++i) {
            Unroll<स्तम्भसङ्ख्या>{}(compute, i);
        }

        //store to C
        स्वचालित storec = [&](स्वचालित col) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + col * 16), vc[col]);
        };
        Unroll<स्तम्भसङ्ख्या>{}(storec);
    }
};

#परिभाषय LAUNCH_TINYGEMM_KERNEL_VNNI(NB_आकार)                                                   \
    tinygemm_kernel_vnni<vec_dot_type, type, भिन्न, 1, NB_आकार, blck_आकार>::apply(             \
        KB, wdata_batch,                                                                       \
        (ध्रुव वर्ण *)src0->दत्त + src0_विस्थापनम् + PACKED_सूचकांक(nb * kTilesN, 0, KB, TILE_आकार), \
        (भिन्न *) dst->दत्त + dst_विस्थापनम् + nb_start, ldc)

प्रतिरूपम् <प्रकारनाम TA, प्रकारनाम TB, प्रकारनाम TC, पूर्णाङ्क खण्डम्_K,
          प्रकारनाम std::enable_if<!is_type_qkk<TB>::value, पूर्णाङ्क>::type = 0>
शून्य tinygemm_kernel_amx(पूर्णाङ्क M, पूर्णाङ्क N, पूर्णाङ्क KB, ध्रुव शून्य * RESTRICT _A, ध्रुव शून्य * RESTRICT _B, TC * RESTRICT C, पूर्णाङ्क ldc) {
    उपयुज्य packed_B_t = packed_B_type<TB>;
    ध्रुव पूर्णाङ्क TILE_आकार = get_tile_आकार<TB>();
    ध्रुव तार्किक need_unpack = do_unpack<TB>::value;

    GGML_ASSERT(M <= 2 * TILE_M && N == 2 * TILE_N);
    ध्रुव TA * RESTRICT A = static_cast<ध्रुव TA *>(_A);
    ध्रुव वर्ण * RESTRICT B = static_cast<ध्रुव वर्ण *>(_B);

    ध्रुव पूर्णाङ्क m0 = std::min(M, TILE_M);
    ध्रुव पूर्णाङ्क m1 = std::max(M - TILE_M, 0);
    ध्रुव पूर्णाङ्क lda = KB * परिमाण(TA);
    //const int ldb = KB * sizeof(TB);

    alignas(64) स्थिर thread_local packed_B_t Tile0[TILE_N * TILE_K];
    alignas(64) स्थिर thread_local packed_B_t Tile1[TILE_N * TILE_K];
    alignas(64) स्थिर thread_local int8_t Tile23[TILE_M * TILE_K];

    alignas(64) स्थिर thread_local int32_t TileC0[TILE_M * TILE_N * 4];
    alignas(64) स्थिर thread_local int32_t TileC1[TILE_M * TILE_N * 4];

    // double buffering C to interleave avx512 and amx
    int32_t * C_cur = TileC0;
    int32_t * C_pre = TileC1;

    स्वचालित Tile4 = [&](int32_t * आधारः) { प्रत्यागम आधारः; };
    स्वचालित Tile5 = [&](int32_t * आधारः) { प्रत्यागम आधारः + TILE_M * TILE_N; };
    स्वचालित Tile6 = [&](int32_t * आधारः) { प्रत्यागम आधारः + 2 * TILE_M * TILE_N; };
    स्वचालित Tile7 = [&](int32_t * आधारः) { प्रत्यागम आधारः + 3 * TILE_M * TILE_N; };

    यदि (M == 2 * TILE_M) {
        // i = 0
        ध्रुव वर्ण * B_blk0 = B + PACKED_सूचकांक(0, 0, KB, TILE_आकार);
        ध्रुव वर्ण * B_blk1 = B + PACKED_सूचकांक(1, 0, KB, TILE_आकार);
        यदि (need_unpack) {
            unpack_B<TB>(Tile0, B_blk0);
            _tile_loadd(TMM0, Tile0, TILE_N * VNNI_BLK);
        } अन्यथा {
            _tile_loadd(TMM0, B_blk0, TILE_N * VNNI_BLK);
        }

        _tile_zero(TMM4);
        _tile_loadd(TMM2, A[0].qs, lda);
        _tile_dpbssd(TMM4, TMM2, TMM0);
        _tile_stored(TMM4, Tile4(C_pre), TILE_N * परिमाण(int32_t));

        _tile_zero(TMM5);
        _tile_loadd(TMM3, A[TILE_M * KB + 0].qs, lda);
        _tile_dpbssd(TMM5, TMM3, TMM0);
        _tile_stored(TMM5, Tile5(C_pre), TILE_N * परिमाण(int32_t));

        यदि (need_unpack) {
            unpack_B<TB>(Tile1, B_blk1);
            _tile_loadd(TMM1, Tile1, TILE_N * VNNI_BLK);
        } अन्यथा {
            _tile_loadd(TMM1, B_blk1, TILE_N * VNNI_BLK);
        }

        _tile_zero(TMM6);
        _tile_dpbssd(TMM6, TMM2, TMM1);
        _tile_stored(TMM6, Tile6(C_pre), TILE_N * परिमाण(int32_t));

        _tile_zero(TMM7);
        _tile_dpbssd(TMM7, TMM3, TMM1);
        _tile_stored(TMM7, Tile7(C_pre), TILE_N * परिमाण(int32_t));

        प्रति (पूर्णाङ्क i = 1; i < KB; ++i) {
            // index of previous iter
            ध्रुव पूर्णाङ्क ii = i - 1;
            ध्रुव वर्ण * B_blk0 = B + PACKED_सूचकांक(0, i, KB, TILE_आकार);
            ध्रुव वर्ण * B_blk1 = B + PACKED_सूचकांक(1, i, KB, TILE_आकार);
            GGML_DISPATCH_BOOL(ii > 0, is_acc, [&] {
                यदि (need_unpack) {
                    unpack_B<TB>(Tile0, B_blk0);
                    _tile_loadd(TMM0, Tile0, TILE_N * VNNI_BLK);
                } अन्यथा {
                    _tile_loadd(TMM0, B_blk0, TILE_N * VNNI_BLK);
                }
                _tile_zero(TMM4);
                _tile_loadd(TMM2, A[i].qs, lda);
                acc_C<TA, TB, is_acc>::apply(C, ldc, Tile4(C_pre), &A[ii], KB, B + PACKED_सूचकांक(0, ii, KB, TILE_आकार), TILE_M);

                _tile_dpbssd(TMM4, TMM2, TMM0);
                _tile_stored(TMM4, Tile4(C_cur), TILE_N * परिमाण(int32_t));

                _tile_zero(TMM5);
                _tile_loadd(TMM3, A[TILE_M * KB + i].qs, lda);
                acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc, ldc, Tile5(C_pre), &A[TILE_M * KB + ii], KB, B + PACKED_सूचकांक(0, ii, KB, TILE_आकार), TILE_M);

                _tile_dpbssd(TMM5, TMM3, TMM0);
                _tile_stored(TMM5, Tile5(C_cur), TILE_N * परिमाण(int32_t));

                यदि (need_unpack) {
                    unpack_B<TB>(Tile1, B_blk1);
                    _tile_loadd(TMM1, Tile1, TILE_N * VNNI_BLK);
                } अन्यथा {
                    _tile_loadd(TMM1, B_blk1, TILE_N * VNNI_BLK);
                }
                _tile_zero(TMM6);
                acc_C<TA, TB, is_acc>::apply(C + TILE_N, ldc, Tile6(C_pre), &A[ii], KB, B + PACKED_सूचकांक(1, ii, KB, TILE_आकार), TILE_M);

                _tile_dpbssd(TMM6, TMM2, TMM1);
                _tile_stored(TMM6, Tile6(C_cur), TILE_N * परिमाण(int32_t));

                _tile_zero(TMM7);
                acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc + TILE_N, ldc, Tile7(C_pre), &A[TILE_M * KB + ii], KB, B + PACKED_सूचकांक(1, ii, KB, TILE_आकार), TILE_M);

                _tile_dpbssd(TMM7, TMM3, TMM1);
                _tile_stored(TMM7, Tile7(C_cur), TILE_N * परिमाण(int32_t));

                std::swap(C_cur, C_pre);
            });
        }
        // final accumulation
        {
            पूर्णाङ्क ii = KB - 1;
            acc_C<TA, TB, सत्य>::apply(C, ldc, Tile4(C_pre), &A[ii], KB, B + PACKED_सूचकांक(0, ii, KB, TILE_आकार), TILE_M);
            acc_C<TA, TB, सत्य>::apply(C + TILE_M * ldc, ldc, Tile5(C_pre), &A[TILE_M * KB + ii], KB, B + PACKED_सूचकांक(0, ii, KB, TILE_आकार), TILE_M);
            acc_C<TA, TB, सत्य>::apply(C + TILE_N, ldc, Tile6(C_pre), &A[ii], KB, B + PACKED_सूचकांक(1, ii, KB, TILE_आकार), TILE_M);
            acc_C<TA, TB, सत्य>::apply(C + TILE_M * ldc + TILE_N, ldc, Tile7(C_pre), &A[TILE_M * KB + ii], KB, B + PACKED_सूचकांक(1, ii, KB, TILE_आकार), TILE_M);
        }
    } अन्यथा {
        प्रति (पूर्णाङ्क i = 0; i < KB; ++i) {
            _tile_zero(TMM4);
            _tile_zero(TMM6);
            यदि (m1 != 0) {
                _tile_zero(TMM5);
                _tile_zero(TMM7);
            }

            ध्रुव वर्ण * B_blk0 = B + PACKED_सूचकांक(0, i, KB, TILE_आकार);
            ध्रुव वर्ण * B_blk1 = B + PACKED_सूचकांक(1, i, KB, TILE_आकार);
            यदि (need_unpack) {
                unpack_B<TB>(Tile0, B_blk0);
                _tile_loadd(TMM0, Tile0, TILE_N * VNNI_BLK);
            } अन्यथा {
                _tile_loadd(TMM0, B_blk0, TILE_N * VNNI_BLK);
            }

            यदि (need_unpack) {
                unpack_B<TB>(Tile1, B_blk1);
                _tile_loadd(TMM1, Tile1, TILE_N * VNNI_BLK);
            } अन्यथा {
                _tile_loadd(TMM1, B_blk1, TILE_N * VNNI_BLK);
            }

            यदि (m0 == TILE_M) {
                _tile_loadd(TMM2, A[i].qs, lda);
            } अन्यथा {
                unpack_A(Tile23, &A[i], KB, m0);
                _tile_loadd(TMM2, Tile23, TILE_K);
            }

            _tile_dpbssd(TMM4, TMM2, TMM0);
            _tile_dpbssd(TMM6, TMM2, TMM1);

            _tile_stored(TMM4, Tile4(C_cur), TILE_N * परिमाण(int32_t));
            _tile_stored(TMM6, Tile6(C_cur), TILE_N * परिमाण(int32_t));

            GGML_DISPATCH_BOOL(i > 0, is_acc, [&] {
                acc_C<TA, TB, is_acc>::apply(C,          ldc, Tile4(C_cur), &A[i], KB, B + PACKED_सूचकांक(0, i, KB, TILE_आकार), m0);
                acc_C<TA, TB, is_acc>::apply(C + TILE_N, ldc, Tile6(C_cur), &A[i], KB, B + PACKED_सूचकांक(1, i, KB, TILE_आकार), m0);
            });

            यदि (m1 != 0) {
                unpack_A(Tile23, &A[TILE_M * KB + i], KB, m1);
                _tile_loadd(TMM3, Tile23, TILE_K);

                _tile_dpbssd(TMM5, TMM3, TMM0);
                _tile_dpbssd(TMM7, TMM3, TMM1);
                _tile_stored(TMM5, Tile5(C_cur), TILE_N * परिमाण(int32_t));
                _tile_stored(TMM7, Tile7(C_cur), TILE_N * परिमाण(int32_t));
                GGML_DISPATCH_BOOL(i > 0, is_acc, [&] {
                    acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc,          ldc, Tile5(C_cur), &A[TILE_M * KB + i], KB, B + PACKED_सूचकांक(0, i, KB, TILE_आकार), m1);
                    acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc + TILE_N, ldc, Tile7(C_cur), &A[TILE_M * KB + i], KB, B + PACKED_सूचकांक(1, i, KB, TILE_आकार), m1);
                });
            }
        }
    }
    प्रत्यागम;
}

प्रतिरूपम् <प्रकारनाम TA, प्रकारनाम TB, प्रकारनाम TC, पूर्णाङ्क खण्डम्_K,
          प्रकारनाम std::enable_if<is_type_qkk<TB>::value, पूर्णाङ्क>::type = 0>
शून्य tinygemm_kernel_amx(पूर्णाङ्क M, पूर्णाङ्क N, पूर्णाङ्क KB, ध्रुव शून्य * RESTRICT _A, ध्रुव शून्य * RESTRICT _B, भिन्न * RESTRICT C, पूर्णाङ्क ldc) {
    static_assert(std::is_same<TA, खण्डम्_q8_K>::value);
    ध्रुव पूर्णाङ्क TILE_आकार = get_tile_आकार<TB>();

    GGML_ASSERT(M <= 2 * TILE_M && N == 2 * TILE_N);
    ध्रुव TA * RESTRICT A = static_cast<ध्रुव TA *>(_A);
    ध्रुव वर्ण * RESTRICT B = static_cast<ध्रुव वर्ण *>(_B);

    ध्रुव पूर्णाङ्क m0 = std::min(M, TILE_M);
    ध्रुव पूर्णाङ्क m1 = std::max(M - TILE_M, 0);
    //const int lda = KB * sizeof(TA);

    alignas(64) स्थिर thread_local int8_t Tile0[TILE_N * TILE_K];
    alignas(64) स्थिर thread_local int8_t Tile1[TILE_N * TILE_K];
    alignas(64) स्थिर thread_local int8_t Tile23[TILE_M * TILE_K];

    // mat mul result for each group
    alignas(64) स्थिर thread_local int32_t Tile4[TILE_M * TILE_N];
    alignas(64) स्थिर thread_local int32_t Tile5[TILE_M * TILE_N];
    alignas(64) स्थिर thread_local int32_t Tile6[TILE_M * TILE_N];
    alignas(64) स्थिर thread_local int32_t Tile7[TILE_M * TILE_N];

    // sum of each QK_K block, contains 8 groups, int32
    alignas(64) स्थिर thread_local int32_t Sumi4[TILE_M * TILE_N];
    alignas(64) स्थिर thread_local int32_t Sumi5[TILE_M * TILE_N];
    alignas(64) स्थिर thread_local int32_t Sumi6[TILE_M * TILE_N];
    alignas(64) स्थिर thread_local int32_t Sumi7[TILE_M * TILE_N];

    ध्रुव पूर्णाङ्क k_गणः_आकार = std::is_same<TB, खण्डम्_q6_K>::value ? 16 : 32;
    प्रति (पूर्णाङ्क i = 0; i < KB; ++i) {
        // step 1: accumulate the quants across 8 groups, each group with 32
        प्रति (पूर्णाङ्क k = 0; k < QK_K / k_गणः_आकार; ++k) {
            GGML_DISPATCH_BOOL(k > 0, is_acc, [&] {
                _tile_zero(TMM4);
                _tile_zero(TMM6);

                unpack_B<TB>(Tile0, B + PACKED_सूचकांक(0, i, KB, TILE_आकार), k);
                _tile_loadd(TMM0, Tile0, TILE_N * VNNI_BLK);

                unpack_B<TB>(Tile1, B + PACKED_सूचकांक(1, i, KB, TILE_आकार), k);
                _tile_loadd(TMM1, Tile1, TILE_N * VNNI_BLK);

                unpack_A<TB>(Tile23, &A[i], KB, k, m0);
                _tile_loadd(TMM2, Tile23, TILE_K);

                _tile_dpbssd(TMM4, TMM2, TMM0);
                _tile_dpbssd(TMM6, TMM2, TMM1);

                _tile_stored(TMM4, Tile4, TILE_N * परिमाण(int32_t));
                _tile_stored(TMM6, Tile6, TILE_N * परिमाण(int32_t));

                scale_C<TB, is_acc>(Tile4, Sumi4, B + PACKED_सूचकांक(0, i, KB, TILE_आकार), k, m0);
                scale_C<TB, is_acc>(Tile6, Sumi6, B + PACKED_सूचकांक(1, i, KB, TILE_आकार), k, m0);

                यदि (m1 != 0) {
                    _tile_zero(TMM5);
                    _tile_zero(TMM7);

                    unpack_A<TB>(Tile23, &A[TILE_M * KB + i], KB, k, m1);
                    _tile_loadd(TMM3, Tile23, TILE_K);

                    _tile_dpbssd(TMM5, TMM3, TMM0);
                    _tile_dpbssd(TMM7, TMM3, TMM1);

                    _tile_stored(TMM5, Tile5, TILE_N * परिमाण(int32_t));
                    _tile_stored(TMM7, Tile7, TILE_N * परिमाण(int32_t));

                    scale_C<TB, is_acc>(Tile5, Sumi5, B + PACKED_सूचकांक(0, i, KB, TILE_आकार), k, m1);
                    scale_C<TB, is_acc>(Tile7, Sumi7, B + PACKED_सूचकांक(1, i, KB, TILE_आकार), k, m1);
                }
            });
        }

        // step 2: accmulate the mins
        GGML_DISPATCH_BOOL(i > 0, is_acc, [&] {
            acc_C<TA, TB, is_acc>::apply(C,          ldc, Sumi4, &A[i], KB, B + PACKED_सूचकांक(0, i, KB, TILE_आकार), m0);
            acc_C<TA, TB, is_acc>::apply(C + TILE_N, ldc, Sumi6, &A[i], KB, B + PACKED_सूचकांक(1, i, KB, TILE_आकार), m0);
            यदि (m1 != 0) {
                acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc,          ldc, Sumi5, &A[TILE_M * KB + i], KB, B + PACKED_सूचकांक(0, i, KB, TILE_आकार), m1);
                acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc + TILE_N, ldc, Sumi7, &A[TILE_M * KB + i], KB, B + PACKED_सूचकांक(1, i, KB, TILE_आकार), m1);
            }
        });
    }
    प्रत्यागम;
}

} // anonymous namespace

// get the packed tensor size for quantized weights
size_t ggml_backend_amx_get_alloc_आकार(ध्रुव संरचना ggml_tensor * tensor) {
    ध्रुव गणना ggml_type TYPE = tensor->type;

    ध्रुव पूर्णाङ्क K = tensor->ne[0]; // ne0: in_features
    ध्रुव पूर्णाङ्क N = tensor->ne[1]; // ne1: out_features

    स्वचालित get_tensor_आकार = [&] {
        size_t row_आकार_B{0};
        GGML_DISPATCH_QTYPES(TYPE, [&] {
            row_आकार_B = get_row_आकार<type, blck_आकार>(K);
        });
        प्रत्यागम N * row_आकार_B;
    };

    यदि (qtype_has_amx_kernels(TYPE)) {
        प्रत्यागम get_tensor_आकार();
    } अन्यथा {
        // for f16, bf16 we don't do packing
        प्रत्यागम ggml_nbytes(tensor);
    }
}

// pack weight to vnni format
शून्य ggml_backend_amx_convert_weight(संरचना ggml_tensor * tensor, ध्रुव शून्य * दत्त, size_t विस्थापनम्, size_t size) {
    GGML_ASSERT(विस्थापनम् == 0 && size == ggml_nbytes(tensor)); // only full tensor conversion is supported for now

    ध्रुव गणना ggml_type TYPE = tensor->type;

    ध्रुव पूर्णाङ्क K = tensor->ne[0]; // ne0: in_features
    ध्रुव पूर्णाङ्क N = tensor->ne[1]; // ne1: out_features

    GGML_DISPATCH_QTYPES(TYPE, [&] {
        convert_B_packed_format<type, blck_आकार>((शून्य *)((वर्ण *)tensor->दत्त + विस्थापनम्), (ध्रुव type *)दत्त, N, K);
    });
}

// ne2 is passed explicitly to help compiler optimize repeated calls
अन्तःस्थ int64_t ggml_batch_विस्थापनम्(ध्रुव ggml_tensor * t, int64_t batch_idx, int64_t ne2) {
    ध्रुव int64_t i2 = batch_idx % ne2;
    ध्रुव int64_t i3 = batch_idx / ne2;
    प्रत्यागम i3 * t->nb[3] + i2 * t->nb[2];
}

size_t ggml_backend_amx_desired_wsize(ध्रुव संरचना ggml_tensor * dst) {
    संरचना ggml_tensor * src0 = dst->src[0];

    ध्रुव गणना ggml_type TYPE = src0->type;

    ध्रुव तार्किक is_floating_type = TYPE == GGML_TYPE_F16;
    यदि (is_floating_type) {
        प्रत्यागम 0;
    }

    ध्रुव पूर्णाङ्क M = dst->ne[1];
    ध्रुव पूर्णाङ्क K = src0->ne[0];
    ध्रुव int64_t n_batch = dst->ne[2] * dst->ne[3];

    size_t desired_wsize = 0;

    GGML_DISPATCH_QTYPES(TYPE, [&] {
        ध्रुव size_t row_आकार_A = K / blck_आकार * परिमाण(vec_dot_type);
        desired_wsize = n_batch * M * row_आकार_A;
    });

    प्रत्यागम desired_wsize;
}

// NB: mixed dtype gemm with Advanced Matrix Extensions (Intel AMX)
//
// src0: weight in shape of {N, K}, quantized
// src1: input  in shape of {M, K}, float32
// dst:  output in shape of {M, N}, float32
//
// the function performs: dst = src1 @ src0.T for each batch
//
शून्य ggml_backend_amx_mul_mat(ध्रुव ggml_compute_params * params, संरचना ggml_tensor * dst) {
    संरचना ggml_tensor * src0 = dst->src[0];
    संरचना ggml_tensor * src1 = dst->src[1];

    ध्रुव गणना ggml_type TYPE = src0->type;

    // f16 only has avx512 kernels for now,
    // amx kernels will be added once 6th gen xeon is released.
    ध्रुव तार्किक is_floating_type = TYPE == GGML_TYPE_F16;

    ध्रुव पूर्णाङ्क M = dst->ne[1];
    ध्रुव पूर्णाङ्क N = dst->ne[0];
    ध्रुव पूर्णाङ्क K = src0->ne[0];
    ध्रुव पूर्णाङ्क ldc = dst->nb[1] / dst->nb[0];

    ध्रुव int64_t ne2 = dst->ne[2];
    ध्रुव int64_t n_batch = ne2 * dst->ne[3];

    यदि (is_floating_type) {
        constexpr पूर्णाङ्क खण्डम्_M = 4;
        constexpr पूर्णाङ्क खण्डम्_N = 6;
        ध्रुव पूर्णाङ्क MB = div_ऊर्ध्वम्(M, खण्डम्_M);
        ध्रुव पूर्णाङ्क NB = div_ऊर्ध्वम्(N, खण्डम्_N);

        parallel_for_ggml(params, n_batch * MB * NB, [&](पूर्णाङ्क begin, पूर्णाङ्क end) {
            GGML_DISPATCH_FLOATING_TYPES(TYPE, [&] {
                प्रति (पूर्णाङ्क i = begin; i < end; ++i) {
                    पूर्णाङ्क batch_idx = i / (MB * NB);
                    पूर्णाङ्क remaining = i % (MB * NB);
                    पूर्णाङ्क mb = remaining / NB;
                    पूर्णाङ्क nb = remaining % NB;

                    int64_t src0_विस्थापनम् = ggml_batch_विस्थापनम्(src0, batch_idx, ne2);
                    int64_t src1_विस्थापनम् = ggml_batch_विस्थापनम्(src1, batch_idx, ne2);
                    int64_t dst_विस्थापनम्  = ggml_batch_विस्थापनम्(dst,  batch_idx, ne2);

                    पूर्णाङ्क mb_start = mb * खण्डम्_M;
                    पूर्णाङ्क mb_आकार = std::min(खण्डम्_M, M - mb_start);
                    पूर्णाङ्क nb_start = nb * खण्डम्_N;
                    पूर्णाङ्क nb_आकार = std::min(खण्डम्_N, N - nb_start);

                    विकल्प (mb_आकार << 4 | nb_आकार) {
                        पक्ष 0x12: LAUNCH_TINYGEMM_KERNEL_AVX(1, 2); भङ्ग;
                        पक्ष 0x14: LAUNCH_TINYGEMM_KERNEL_AVX(1, 4); भङ्ग;
                        पक्ष 0x16: LAUNCH_TINYGEMM_KERNEL_AVX(1, 6); भङ्ग;
                        पक्ष 0x22: LAUNCH_TINYGEMM_KERNEL_AVX(2, 2); भङ्ग;
                        पक्ष 0x24: LAUNCH_TINYGEMM_KERNEL_AVX(2, 4); भङ्ग;
                        पक्ष 0x26: LAUNCH_TINYGEMM_KERNEL_AVX(2, 6); भङ्ग;
                        पक्ष 0x32: LAUNCH_TINYGEMM_KERNEL_AVX(3, 2); भङ्ग;
                        पक्ष 0x34: LAUNCH_TINYGEMM_KERNEL_AVX(3, 4); भङ्ग;
                        पक्ष 0x36: LAUNCH_TINYGEMM_KERNEL_AVX(3, 6); भङ्ग;
                        पक्ष 0x42: LAUNCH_TINYGEMM_KERNEL_AVX(4, 2); भङ्ग;
                        पक्ष 0x44: LAUNCH_TINYGEMM_KERNEL_AVX(4, 4); भङ्ग;
                        पक्ष 0x46: LAUNCH_TINYGEMM_KERNEL_AVX(4, 6); भङ्ग;
                        स्वभाव: fprintf(stderr, "Unexpected block size!\n");
                    }
                }
            });
        });
        प्रत्यागम;
    }

    // pointer to work space, used convert A from float to quantized type
    शून्य * wdata = params->wdata;

    //TODO: performance improvement: merge quant A
 // if (params->ith == 0) {
        GGML_DISPATCH_QTYPES(TYPE, [&] {
            ध्रुव size_t row_आकार_A = K / blck_आकार * परिमाण(vec_dot_type);
            ध्रुव size_t desired_wsize = n_batch * M * row_आकार_A;
            यदि (params->wsize < desired_wsize) {
                GGML_ABORT("insufficient work space size");
            }

            // Q4_0, Q4_1, Q8_0 handles 1 TILE_K per blck_size
            // Q4_K, Q5_K, Q6_K, IQ4_XS handles 8 TILE_K per blck_size
            GGML_ASSERT(TILE_K == blck_आकार || TILE_K * 8 == blck_आकार);

            parallel_for_ggml(params, n_batch * M, [&](पूर्णाङ्क begin, पूर्णाङ्क end) {
                प्रति (पूर्णाङ्क idx = begin; idx < end; ++idx) {
                    पूर्णाङ्क batch_idx = idx / M;
                    पूर्णाङ्क m         = idx % M;
                    int64_t src1_विस्थापनम् = ggml_batch_विस्थापनम्(src1, batch_idx, ne2);
                    ध्रुव भिन्न * A_दत्त = (ध्रुव भिन्न *)((ध्रुव वर्ण *)src1->दत्त + src1_विस्थापनम्);
                    वर्ण * wdata_batch = (वर्ण *)wdata + batch_idx * M * row_आकार_A;
                    from_float<vec_dot_type>(A_दत्त + m * K, wdata_batch + m * row_आकार_A, K);
                }
            });
        });
 // }

    ggml_barrier(params->threadpool);

    यदि (M == 1) {
        // MB = 1 and handle 8 tiles in each block
        constexpr पूर्णाङ्क kTilesN = 4;
        constexpr पूर्णाङ्क खण्डम्_N = TILE_N * kTilesN;
        ध्रुव पूर्णाङ्क NB = div_ऊर्ध्वम्(N, खण्डम्_N);

        parallel_for_ggml(params, n_batch * NB, [&](पूर्णाङ्क begin, पूर्णाङ्क end) {
            GGML_DISPATCH_QTYPES(TYPE, [&] {
                ध्रुव पूर्णाङ्क KB = K / blck_आकार;
                ध्रुव पूर्णाङ्क TILE_आकार = get_tile_आकार<type>();
                ध्रुव पूर्णाङ्क row_आकार_A = KB * परिमाण(vec_dot_type);
                प्रति (पूर्णाङ्क i = begin; i < end; ++i) {
                    पूर्णाङ्क batch_idx = i / NB;
                    पूर्णाङ्क nb = i % NB;

                    int64_t src0_विस्थापनम् = ggml_batch_विस्थापनम्(src0, batch_idx, ne2);
                    int64_t dst_विस्थापनम्  = ggml_batch_विस्थापनम्(dst,  batch_idx, ne2);
                    ध्रुव वर्ण * wdata_batch = (ध्रुव वर्ण *)wdata + batch_idx * row_आकार_A;

                    पूर्णाङ्क nb_start = nb * खण्डम्_N;
                    पूर्णाङ्क nb_आकार = std::min(खण्डम्_N, N - nb_start); // 32, 64, 96

                    विकल्प (nb_आकार) {
                        //case 160: LAUNCH_TINYGEMM_KERNEL_VNNI(160); break;
                        पक्ष 128: LAUNCH_TINYGEMM_KERNEL_VNNI(128); भङ्ग;
                        पक्ष 96: LAUNCH_TINYGEMM_KERNEL_VNNI(96); भङ्ग;
                        पक्ष 64: LAUNCH_TINYGEMM_KERNEL_VNNI(64); भङ्ग;
                        पक्ष 32: LAUNCH_TINYGEMM_KERNEL_VNNI(32); भङ्ग;
                        स्वभाव: fprintf(stderr, "Unexpected n block size!\n");
                    }
                }
            });
        });
        प्रत्यागम;
    }

    // handle 4 tiles at a tile
    constexpr पूर्णाङ्क खण्डम्_M = TILE_M * 2;
    constexpr पूर्णाङ्क खण्डम्_N = TILE_N * 2;
    ध्रुव पूर्णाङ्क MB = div_ऊर्ध्वम्(M, खण्डम्_M);
    ध्रुव पूर्णाङ्क NB = div_ऊर्ध्वम्(N, खण्डम्_N);

    parallel_for_ggml(params, n_batch * MB * NB, [&](पूर्णाङ्क begin, पूर्णाङ्क end) {
        // init tile config for each thread
        ggml_tile_config_init();

        GGML_DISPATCH_QTYPES(TYPE, [&] {
            ध्रुव पूर्णाङ्क KB = K / blck_आकार;
            ध्रुव पूर्णाङ्क TILE_आकार = get_tile_आकार<type>();
            ध्रुव पूर्णाङ्क row_आकार_A = KB * परिमाण(vec_dot_type);

            प्रति (पूर्णाङ्क i = begin; i < end; ++i) {
                पूर्णाङ्क batch_idx = i / (MB * NB);
                पूर्णाङ्क remaining = i % (MB * NB);
                पूर्णाङ्क mb = remaining / NB;
                पूर्णाङ्क nb = remaining % NB;

                int64_t src0_विस्थापनम् = ggml_batch_विस्थापनम्(src0, batch_idx, ne2);
                int64_t dst_विस्थापनम्  = ggml_batch_विस्थापनम्(dst,  batch_idx, ne2);
                ध्रुव वर्ण * wdata_batch = (ध्रुव वर्ण *)wdata + batch_idx * M * row_आकार_A;

                पूर्णाङ्क mb_start = mb * खण्डम्_M;
                पूर्णाङ्क mb_आकार = std::min(खण्डम्_M, M - mb_start);
                पूर्णाङ्क nb_start = nb * खण्डम्_N;
                पूर्णाङ्क nb_आकार = खण्डम्_N;

                tinygemm_kernel_amx<vec_dot_type, type, भिन्न, blck_आकार>(
                    mb_आकार, nb_आकार, KB,
                    wdata_batch + mb_start * row_आकार_A,
                    (ध्रुव वर्ण *)src0->दत्त + src0_विस्थापनम् + PACKED_सूचकांक(nb * 2, 0, KB, TILE_आकार),
                    (भिन्न *) dst->दत्त + dst_विस्थापनम् + mb_start * N + nb_start, ldc);
            }
        });
    });
}

#यदि_अन्तः // if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
