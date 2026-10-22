#अगर defined(__GNUC__)
#आशय GCC diagnostic ignored "-Wpedantic"
#आशय GCC diagnostic ignored "-Wunused-local-typedefs"
#पूर्ण_अगर

#समावेश "amx.h"
#समावेश "mmq.h"
#समावेश "ggml-impl.h"
#समावेश "ggml-cpu-impl.h"
#समावेश "simd-mappings.h"
#समावेश "quants.h"
#समावेश "ggml-quants.h"
#समावेश <algorithm>
#समावेश <type_traits>

#अगर defined(__gnu_linux__)
#समावेश <sys/syscall.h>
#समावेश <unistd.h>
#पूर्ण_अगर

#अगर (defined(_WIN32) || defined(_WIN64))
#घोषणा RESTRICT __restrict
#अन्यथा
#घोषणा RESTRICT __restrict__
#पूर्ण_अगर

#अगर (defined(_WIN32) || defined(_WIN64))
#घोषणा ALWAYS_INLINE __forceinline
#या_अगर __has_attribute(always_inline) || defined(__GNUC__)
#घोषणा ALWAYS_INLINE __attribute__((__always_inline__)) अंतरभूत
#अन्यथा
#घोषणा ALWAYS_INLINE अंतरभूत
#पूर्ण_अगर

#अगर defined(__AMX_INT8__) && defined(__AVX512VNNI__)

namespace {

// Forced unrolling
ढाँचा <पूर्णांक n>
काष्ठा Unroll {
    ढाँचा <typename Func, typename... Args>
    ALWAYS_INLINE व्योम चालक()(स्थिर Func& f, Args... args) स्थिर {
        Unroll<n - 1>{}(f, args...);
        f(std::integral_constant<पूर्णांक, n - 1>{}, args...);
    }
};

ढाँचा <>
काष्ठा Unroll<1> {
    ढाँचा <typename Func, typename... Args>
    ALWAYS_INLINE व्योम चालक()(स्थिर Func& f, Args... args) स्थिर {
        f(std::integral_constant<पूर्णांक, 0>{}, args...);
    }
};

// type traits
ढाँचा <typename T> काष्ठा PackedTypes {};
ढाँचा <> काष्ठा PackedTypes<block_q4_0> { using type = int8_t; };
ढाँचा <> काष्ठा PackedTypes<block_q4_1> { using type = uint8_t; };
ढाँचा <> काष्ठा PackedTypes<block_q8_0> { using type = int8_t; };
ढाँचा <typename T> using packed_B_type = typename PackedTypes<T>::type;

ढाँचा <typename T>
काष्ठा do_compensate : std::integral_constant<bool,
    std::is_same<T, block_q8_0>::value> {};

ढाँचा <typename T>
काष्ठा do_unpack : std::integral_constant<bool,
    std::is_same<T, block_q4_0>::value ||
    std::is_same<T, block_q4_1>::value> {};

ढाँचा <typename T>
काष्ठा is_type_qkk : std::integral_constant<bool,
    std::is_same<T, block_q4_K>::value ||
    std::is_same<T, block_q5_K>::value ||
    std::is_same<T, block_q6_K>::value ||
    std::is_same<T, block_iq4_xs>::value> {};

#घोषणा GGML_DISPATCH_FLOATING_TYPES(TYPE, ...)                                        \
    [&] {                                                                              \
        चयन (TYPE) {                                                                \
            हाल GGML_TYPE_F16: {                                                      \
                using type = ggml_fp16_t;                                              \
                constexpr पूर्णांक blck_आकार = 16;                                          \
                वापस __VA_ARGS__();                                                  \
            }                                                                          \
            हाल GGML_TYPE_BF16: {                                                     \
                using type = ggml_bf16_t;                                              \
                constexpr पूर्णांक blck_आकार = 32;                                          \
                वापस __VA_ARGS__();                                                  \
            }                                                                          \
            शेष:                                                                   \
                ख_लिखो(मानक_त्रुटि, "Unsupported floating data type\n");                   \
        }                                                                              \
    }()

#घोषणा GGML_DISPATCH_QTYPES(QT, ...)                                                  \
    [&] {                                                                              \
        चयन (QT) {                                                                  \
            हाल GGML_TYPE_Q4_0: {                                                     \
                using type = block_q4_0;                                               \
                using vec_dot_type = block_q8_0;                                       \
                constexpr पूर्णांक blck_आकार = QK4_0;                                       \
                वापस __VA_ARGS__();                                                  \
            }                                                                          \
            हाल GGML_TYPE_Q4_1: {                                                     \
                using type = block_q4_1;                                               \
                using vec_dot_type = block_q8_1;                                       \
                constexpr पूर्णांक blck_आकार = QK4_1;                                       \
                वापस __VA_ARGS__();                                                  \
            }                                                                          \
            हाल GGML_TYPE_Q8_0: {                                                     \
                using type = block_q8_0;                                               \
                using vec_dot_type = block_q8_0;                                       \
                constexpr पूर्णांक blck_आकार = QK8_0;                                       \
                वापस __VA_ARGS__();                                                  \
            }                                                                          \
            हाल GGML_TYPE_Q4_K: {                                                     \
                using type = block_q4_K;                                               \
                using vec_dot_type = block_q8_K;                                       \
                constexpr पूर्णांक blck_आकार = QK_K;                                        \
                वापस __VA_ARGS__();                                                  \
            }                                                                          \
            हाल GGML_TYPE_Q5_K: {                                                     \
                using type = block_q5_K;                                               \
                using vec_dot_type = block_q8_K;                                       \
                constexpr पूर्णांक blck_आकार = QK_K;                                        \
                वापस __VA_ARGS__();                                                  \
            }                                                                          \
            हाल GGML_TYPE_Q6_K: {                                                     \
                using type = block_q6_K;                                               \
                using vec_dot_type = block_q8_K;                                       \
                constexpr पूर्णांक blck_आकार = QK_K;                                        \
                वापस __VA_ARGS__();                                                  \
            }                                                                          \
            हाल GGML_TYPE_IQ4_XS: {                                                   \
                using type = block_iq4_xs;                                             \
                using vec_dot_type = block_q8_K;                                       \
                constexpr पूर्णांक blck_आकार = QK_K;                                        \
                वापस __VA_ARGS__();                                                  \
            }                                                                          \
            शेष:                                                                   \
                ख_लिखो(मानक_त्रुटि, "Unsupported quantized data type: %d\n", पूर्णांक(TYPE));   \
        }                                                                              \
    }()

#घोषणा GGML_DISPATCH_BOOL(BOOL_V, BOOL_NAME, ...)                                     \
    [&] {                                                                              \
        अगर (BOOL_V) {                                                                  \
            constexpr bool BOOL_NAME = true;                                           \
            वापस __VA_ARGS__();                                                      \
        } अन्यथा {                                                                       \
            constexpr bool BOOL_NAME = false;                                          \
            वापस __VA_ARGS__();                                                      \
        }                                                                              \
    }()

// define amx tile config data structure
काष्ठा tile_config_t{
    uint8_t palette_id = 0;
    uint8_t start_row = 0;
    uint8_t reserved_0[14] = {0};
    uint16_t colsb[16] = {0};
    uint8_t rows[16] = {0};
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

अंतरभूत व्योम ggml_tile_config_init(व्योम) {
    अटल thread_local bool done = false;

    अगर (done) {
        वापस;
    }

    alignas(64) tile_config_t tc = {};
    tc.palette_id = 1;
    tc.start_row = 0;
    tc.rows[0] = 8;   tc.colsb[0] = 64;
    tc.rows[1] = 8;   tc.colsb[1] = 64;
    tc.rows[2] = 16;  tc.colsb[2] = 32;
    tc.rows[3] = 16;  tc.colsb[3] = 32;
    tc.rows[4] = 16;  tc.colsb[4] = 64;
    tc.rows[5] = 16;  tc.colsb[5] = 64;
    tc.rows[6] = 16;  tc.colsb[6] = 64;
    tc.rows[7] = 16;  tc.colsb[7] = 64;

    _tile_loadconfig(&tc);
    done = true;
}

// we need an extra 16 * 4B (TILE_N * int32_t) for each NB/KB block for compensation.
// See the notes `s8s8 igemm compensation in avx512-vnni` for detail.
ढाँचा <typename TB>
पूर्णांक get_tile_आकार() {
    पूर्णांक tile_आकार = TILE_N * माप(TB);
    अगर (do_compensate<TB>::value) {
        tile_आकार += TILE_N * माप(int32_t);
    }
    अगर (std::is_same<TB, block_q4_K>::value ||
        std::is_same<TB, block_q5_K>::value) {
        tile_आकार += TILE_N * 4;
    }
    अगर (std::is_same<TB, block_iq4_xs>::value) {
        tile_आकार += TILE_N * 2;
    }
    वापस tile_आकार;
}

ढाँचा <typename TB, पूर्णांक BLOCK_K>
पूर्णांक get_row_आकार(पूर्णांक K) {
    पूर्णांक KB = K / BLOCK_K;
    पूर्णांक row_आकार = KB * माप(TB);
    अगर (do_compensate<TB>::value) {
        row_आकार += KB * माप(int32_t);
    }
    अगर (std::is_same<TB, block_q4_K>::value ||
        std::is_same<TB, block_q5_K>::value) {
        row_आकार += KB * 4;
    }
    अगर (std::is_same<TB, block_iq4_xs>::value) {
        row_आकार += KB * 2;
    }
    वापस row_आकार;
}

// transpose utils
#घोषणा SHUFFLE_EPI32(a, b, mask) \
    _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b), mask))
अंतरभूत व्योम transpose_8x8_32bit(__m256i * v, __m256i * v1) {
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

अंतरभूत व्योम transpose_16x4_32bit(__m512i * r, __m512i * d) {

    अटल स्थिर __m512i index1 = _mm512_set_epi32(
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

अंतरभूत व्योम transpose_16x16_32bit(__m512i * v) {
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

व्योम quantize_row_q8_K_vnni(स्थिर भग्न * RESTRICT x, व्योम * RESTRICT vy, int64_t k) {
    निश्चित(k % QK_K == 0);
    स्थिर पूर्णांक KB = k / QK_K;
    constexpr पूर्णांक kVecs = QK_K / 16;

    block_q8_K * y = reinterpret_cast<block_q8_K *>(vy);

    // hold 16 float vecs from x
    __m512  v[kVecs];

    // hold the quants vecs
    __m512i vq[kVecs / 4];

    // hold the packed quants vecs
    __m512i vq_packed[kVecs / 4];

    स्थिर __m512 signBit = _mm512_set1_ps(-0.f);

    क्रम (पूर्णांक i = 0; i < KB; ++i) {
        // Compute max(abs(e)) for the block
        __m512 vamax = _mm512_set1_ps(0.f);
        क्रम (पूर्णांक j = 0; j < kVecs; ++j) {
            v[j] = _mm512_loadu_ps(x); x += 16;
            vamax = _mm512_max_ps(vamax, _mm512_andnot_ps(signBit, v[j]));
        }
        स्थिर भग्न amax = _mm512_reduce_max_ps(vamax);

        // Quantize these floats
        स्थिर भग्न iscale = 127.f / amax;
        y[i].d = GGML_CPU_FP32_TO_FP16(1 / iscale);
        स्थिर भग्न id = ( amax != 0.0f ) ? iscale : 0.f;
        स्थिर __m512 vscale = _mm512_set1_ps(id);

        // Apply multiplier and round to nearest integer
        क्रम (पूर्णांक j = 0; j < kVecs; ++j) {
            v[j] = _mm512_mul_ps(v[j], vscale);
            v[j] = _mm512_roundscale_ps(v[j], (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        }

        // Pack to epi8 vecs
        क्रम (पूर्णांक j = 0; j < kVecs / 4; ++j) {
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

        स्थिर __m512i one = _mm512_set1_epi8(1);
        __m512i योग_2 = _mm512_setzero_si512();
        क्रम (पूर्णांक k = 0; k < 4; ++k) {
            योग_2 = _mm512_dpbusd_epi32(योग_2, one, vq_packed[k]);
        }
        _mm256_storeu_si256((__m256i *)(y[i].bsums), _mm512_cvtepi32_epi16(योग_2));
    }
}

// quantize A from float to `vec_dot_type`
ढाँचा <typename T>
अंतरभूत व्योम from_float(स्थिर भग्न * x, अक्षर * vy, int64_t k);

ढाँचा <>
अंतरभूत व्योम from_float<block_q8_0>(स्थिर भग्न * x, अक्षर * vy, int64_t k) {
    quantize_row_q8_0(x, (block_q8_0 *)vy, k);
}

ढाँचा <>
अंतरभूत व्योम from_float<block_q8_1>(स्थिर भग्न * x, अक्षर * vy, int64_t k) {
    quantize_row_q8_1(x, (block_q8_1 *)vy, k);
}

ढाँचा <>
अंतरभूत व्योम from_float<block_q8_K>(स्थिर भग्न * x, अक्षर * vy, int64_t k) {
#अगर 1
    // TODO: this is reference impl!
    quantize_row_q8_K_ref(x, (block_q8_K *)vy, k);
#अन्यथा
    quantize_row_q8_K_vnni(x, vy, k);
#पूर्ण_अगर
}

// load A from memory to array when nrows can not fill in whole tile
व्योम unpack_A(int8_t * RESTRICT tile, स्थिर block_q8_0 * RESTRICT A, पूर्णांक lda, पूर्णांक nr) {
    निश्चित(nr != TILE_M);
    क्रम (पूर्णांक m = 0; m < nr; ++m) {
        स्थिर __m256i v = _mm256_loadu_si256((स्थिर __m256i *)(A[m * lda].qs));
        _mm256_storeu_si256((__m256i *)(tile + m * TILE_K), v);
    }
}

व्योम unpack_A(int8_t * RESTRICT tile, स्थिर block_q8_1 * RESTRICT A, पूर्णांक lda, पूर्णांक nr) {
    निश्चित(nr != TILE_M);
    क्रम (पूर्णांक m = 0; m < nr; ++m) {
        स्थिर __m256i v = _mm256_loadu_si256((स्थिर __m256i *)(A[m * lda].qs));
        _mm256_storeu_si256((__m256i *)(tile + m * TILE_K), v);
    }
}

ढाँचा <typename TB>
व्योम unpack_A(int8_t * RESTRICT tile, स्थिर block_q8_K * RESTRICT A, पूर्णांक lda, पूर्णांक k, पूर्णांक nr) {
    निश्चित(nr <= TILE_M);
    क्रम (पूर्णांक m = 0; m < nr; ++m) {
        स्थिर __m256i v = _mm256_loadu_si256((स्थिर __m256i *)(A[m * lda].qs + k * 32));
        _mm256_storeu_si256((__m256i *)(tile + m * TILE_K), v);
    }
}

ढाँचा <>
व्योम unpack_A<block_q6_K>(int8_t * RESTRICT tile, स्थिर block_q8_K * RESTRICT A, पूर्णांक lda, पूर्णांक k, पूर्णांक nr) {
    निश्चित(nr <= TILE_M);
    // zero padding k from 16 to 32, so that we don't have to re-config amx
    स्थिर __m128i zero = _mm_setzero_si128();
    क्रम (पूर्णांक m = 0; m < nr; ++m) {
        स्थिर __m128i v = _mm_loadu_si128((स्थिर __m128i *)(A[m * lda].qs + k * 16));
        स्थिर __m256i r = _mm256_insertf128_si256(_mm256_castsi128_si256(v), zero, 1);
        _mm256_storeu_si256((__m256i *)(tile + m * TILE_K), r);
    }
}

#घोषणा MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)
अंतरभूत __m256i bytes_from_nibbles_32(स्थिर uint8_t * rsi) {
    स्थिर __m128i tmp = _mm_loadu_si128((स्थिर __m128i *)rsi);
    स्थिर __m256i bytes = MM256_SET_M128I(_mm_srli_epi16(tmp, 4), tmp);
    स्थिर __m256i निचला_Mask = _mm256_set1_epi8(0xF);
    वापस _mm256_and_si256(निचला_Mask, bytes);
}

// used for block_q4_K
अंतरभूत __m512i bytes_from_nibbles_64(स्थिर uint8_t * rsi) {
    स्थिर __m256i tmp = _mm256_loadu_si256((स्थिर __m256i *)rsi);
    स्थिर __m256i निचला_Mask = _mm256_set1_epi8(0xF);
    स्थिर __m256i q4l = _mm256_and_si256(tmp, निचला_Mask);
    स्थिर __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(tmp, 4), निचला_Mask);
    वापस _mm512_inserti32x8(_mm512_castsi256_si512(q4l), q4h, 1);
}

// used for block_q5_K
अंतरभूत __m512i bytes_from_nibbles_64(स्थिर uint8_t * qs, स्थिर uint8_t * qh, पूर्णांक k) {
    स्थिर __m256i निचला_Mask = _mm256_set1_epi8(0xF);
    __m256i hmask = _mm256_set1_epi8(1);
    hmask = _mm256_slli_epi16(hmask, k);

    स्थिर __m256i q5bits = _mm256_loadu_si256((स्थिर __m256i *)qs);
    स्थिर __m256i hbits = _mm256_loadu_si256((स्थिर __m256i *)qh);

    स्थिर __m256i q5l_0 = _mm256_and_si256(q5bits, निचला_Mask);
    स्थिर __m256i q5h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), k + 0), 4);
    स्थिर __m256i q5_0  = _mm256_add_epi8(q5l_0, q5h_0);
    hmask = _mm256_slli_epi16(hmask, 1);

    स्थिर __m256i q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), निचला_Mask);
    स्थिर __m256i q5h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), k + 1), 4);
    स्थिर __m256i q5_1  = _mm256_add_epi8(q5l_1, q5h_1);

    वापस _mm512_inserti32x8(_mm512_castsi256_si512(q5_0), q5_1, 1);
}

// used for block_q6_K
अंतरभूत व्योम bytes_from_nibbles_128(__m512i& r0, __m512i& r1, स्थिर uint8_t * qs, स्थिर uint8_t * qh) {
    स्थिर __m256i m4 = _mm256_set1_epi8(0xF);
    स्थिर __m256i m2 = _mm256_set1_epi8(0x3);

    स्थिर __m256i q6bits1 = _mm256_loadu_si256((स्थिर __m256i *)qs);
    स्थिर __m256i q6bits2 = _mm256_loadu_si256((स्थिर __m256i *)(qs + 32));
    स्थिर __m256i q6bitsH = _mm256_loadu_si256((स्थिर __m256i *)qh);

    स्थिर __m256i q6h_0 = _mm256_slli_epi16(_mm256_and_si256(                  q6bitsH,     m2), 4);
    स्थिर __m256i q6h_1 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q6bitsH, 2), m2), 4);
    स्थिर __m256i q6h_2 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q6bitsH, 4), m2), 4);
    स्थिर __m256i q6h_3 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q6bitsH, 6), m2), 4);

    स्थिर __m256i q6_0 = _mm256_or_si256(_mm256_and_si256(q6bits1, m4), q6h_0);
    स्थिर __m256i q6_1 = _mm256_or_si256(_mm256_and_si256(q6bits2, m4), q6h_1);
    स्थिर __m256i q6_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q6bits1, 4), m4), q6h_2);
    स्थिर __m256i q6_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q6bits2, 4), m4), q6h_3);

    r0 = _mm512_inserti32x8(_mm512_castsi256_si512(q6_0), q6_1, 1);
    r1 = _mm512_inserti32x8(_mm512_castsi256_si512(q6_2), q6_3, 1);
}

अंतरभूत __m512i packNibbles(__m512i r0, __m512i r1) {
    वापस _mm512_or_si512(r0, _mm512_slli_epi16(r1, 4));
}

ढाँचा <typename TB>
अंतरभूत व्योम pack_qs(व्योम * RESTRICT packed_B, स्थिर TB * RESTRICT B, पूर्णांक KB) {
    int8_t tmp[8 * 64];
    __m256i v[8], v2[8];
    क्रम (पूर्णांक n = 0; n < 8; ++n) {
        v[n] = bytes_from_nibbles_32(B[n * KB].qs);
    }
    transpose_8x8_32bit(v, v2);
    क्रम (पूर्णांक n = 0; n < 8; ++n) {
        _mm256_storeu_si256((__m256i *)(tmp + n * 64), v2[n]);
    }
    क्रम (पूर्णांक n = 0; n < 8; ++n) {
        v[n] = bytes_from_nibbles_32(B[(n + 8) * KB].qs);
    }
    transpose_8x8_32bit(v, v2);
    क्रम (पूर्णांक n = 0; n < 8; ++n) {
        _mm256_storeu_si256((__m256i *)(tmp + n * 64 + 32), v2[n]);
    }

    // pack again with 128 to fully utilize vector length
    क्रम (पूर्णांक n = 0; n < 8; n += 2) {
        __m512i r0 = _mm512_loadu_si512((स्थिर __m512i *)(tmp + n * 64));
        __m512i r1 = _mm512_loadu_si512((स्थिर __m512i *)(tmp + n * 64 + 64));
        __m512i r1r0 = packNibbles(r0, r1);
        _mm512_storeu_si512((__m512i *)((अक्षर *)packed_B + n * 32), r1r0);
    }
}

ढाँचा <>
अंतरभूत व्योम pack_qs<block_q8_0>(व्योम * RESTRICT packed_B, स्थिर block_q8_0 * RESTRICT B, पूर्णांक KB) {
    __m256i v[8], v2[8];
    क्रम (पूर्णांक n = 0; n < 8; ++n) {
        v[n] = _mm256_loadu_si256((स्थिर __m256i *)(B[n * KB].qs));
    }
    transpose_8x8_32bit(v, v2);
    क्रम (पूर्णांक n = 0; n < 8; ++n) {
        _mm256_storeu_si256((__m256i *)((अक्षर *)packed_B + n * 64), v2[n]);
    }
    क्रम (पूर्णांक n = 0; n < 8; ++n) {
        v[n] = _mm256_loadu_si256((स्थिर __m256i *)(B[(n + 8) * KB].qs));
    }
    transpose_8x8_32bit(v, v2);
    क्रम (पूर्णांक n = 0; n < 8; ++n) {
        _mm256_storeu_si256((__m256i *)((अक्षर *)packed_B + n * 64 + 32), v2[n]);
    }
}

ढाँचा <>
अंतरभूत व्योम pack_qs<block_q4_K>(व्योम * RESTRICT packed_B, स्थिर block_q4_K * RESTRICT B, पूर्णांक KB) {
    __m512i v[16];
    // QK_K 256 with 8 groups, handle 2 groups at a time
    अक्षर * pb = (अक्षर *)packed_B;
    क्रम (पूर्णांक k = 0; k < QK_K / 64; ++k) {
        // pack 2 groups { n, g,  k} to {g, k/4, 4n}
        //          e.g. {16, 2, 32} to {2,   8, 64}
        क्रम (पूर्णांक n = 0; n < TILE_N; ++n) {
            v[n] = bytes_from_nibbles_64(B[n * KB].qs + k * 32);
        }

        transpose_16x16_32bit(v);

        // pack again with 128 to fully utilize vector length
        क्रम (पूर्णांक n = 0; n < TILE_N; n += 2) {
            _mm512_storeu_si512((__m512i *)pb, packNibbles(v[n], v[n + 1]));
            pb += 64;
        }
    }
}

ढाँचा <>
अंतरभूत व्योम pack_qs<block_q5_K>(व्योम * RESTRICT packed_B, स्थिर block_q5_K * RESTRICT B, पूर्णांक KB) {
    __m512i v[16];
    स्थिर __m512i निचला_Mask = _mm512_set1_epi8(0xF);
    // QK_K 256 with 8 groups, handle 2 groups at a time
    अक्षर * pb = (अक्षर *)packed_B;
    अक्षर * ph = (अक्षर *)packed_B + (QK_K / 2) * TILE_N;
    क्रम (पूर्णांक k = 0; k < QK_K / 64; ++k) {
        // pack 2 groups { n, g,  k} to {g, k/4, 4n}
        //          e.g. {16, 2, 32} to {2,   8, 64}
        क्रम (पूर्णांक n = 0; n < TILE_N; ++n) {
            v[n] = bytes_from_nibbles_64(B[n * KB].qs + k * 32, B[n * KB].qh, /* group */2 * k);
        }

        transpose_16x16_32bit(v);

        // 1. pack lower 4bits with 2 groups
        क्रम (पूर्णांक n = 0; n < TILE_N; n += 2) {
            // get lower 4 bits
            स्थिर __m512i r0 = _mm512_and_si512(v[n], निचला_Mask);
            स्थिर __m512i r1 = _mm512_and_si512(v[n + 1], निचला_Mask);
            _mm512_storeu_si512((__m512i *)pb, packNibbles(r0, r1)); pb += 64;
        }

        // 2. pack higher 1bit with 2 groups
        स्थिर __m512i hmask = _mm512_set1_epi8(0x10);
        क्रम (पूर्णांक g = 0; g < 2; ++g) {
            __m512i hbits = _mm512_setzero_si512();
            hbits = _mm512_add_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 8 + 0], hmask), 4));
            hbits = _mm512_add_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 8 + 1], hmask), 3));
            hbits = _mm512_add_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 8 + 2], hmask), 2));
            hbits = _mm512_add_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 8 + 3], hmask), 1));
            hbits = _mm512_add_epi8(hbits,                   _mm512_and_si512(v[g * 8 + 4], hmask)    );
            hbits = _mm512_add_epi8(hbits, _mm512_slli_epi16(_mm512_and_si512(v[g * 8 + 5], hmask), 1));
            hbits = _mm512_add_epi8(hbits, _mm512_slli_epi16(_mm512_and_si512(v[g * 8 + 6], hmask), 2));
            hbits = _mm512_add_epi8(hbits, _mm512_slli_epi16(_mm512_and_si512(v[g * 8 + 7], hmask), 3));
            _mm512_storeu_si512((__m512i *)ph, hbits); ph += 64;
        }
    }
}

ढाँचा <>
अंतरभूत व्योम pack_qs<block_q6_K>(व्योम * RESTRICT packed_B, स्थिर block_q6_K * RESTRICT B, पूर्णांक KB) {
    __m512i v[32];
    स्थिर __m512i निचला_Mask = _mm512_set1_epi8(0xF);
    // QK_K 256 with 8 groups, handle 4 groups at a time
    अक्षर * pb = (अक्षर *)packed_B;
    अक्षर * ph = (अक्षर *)packed_B + (QK_K / 2) * TILE_N;
    क्रम (पूर्णांक k = 0; k < QK_K / 128; ++k) {
        क्रम (पूर्णांक n = 0; n < TILE_N; ++n) {
            bytes_from_nibbles_128(v[n], v[n + 16], B[n * KB].ql + k * 64, B[n * KB].qh + k * 32);
        }

        // top half: group 0,1 or 4,5; bottom half: group 2,3 or 6,7
        transpose_16x16_32bit(v);
        transpose_16x16_32bit(v + 16);

        // 1. pack lower 4bits with 4 groups
        क्रम (पूर्णांक n = 0; n < 32; n += 2) {
            स्थिर __m512i r0 = _mm512_and_si512(v[n], निचला_Mask);
            स्थिर __m512i r1 = _mm512_and_si512(v[n + 1], निचला_Mask);
            _mm512_storeu_si512((__m512i *)pb, packNibbles(r0, r1)); pb += 64;
        }

        // 2. pack higher 2bit with 4 groups
        स्थिर __m512i hmask = _mm512_set1_epi8(0x30);
        क्रम (पूर्णांक g = 0; g < 8; ++g) {
            __m512i hbits = _mm512_setzero_si512();
            hbits = _mm512_add_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 4 + 0], hmask), 4));
            hbits = _mm512_add_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 4 + 1], hmask), 2));
            hbits = _mm512_add_epi8(hbits,                   _mm512_and_si512(v[g * 4 + 2], hmask)    );
            hbits = _mm512_add_epi8(hbits, _mm512_slli_epi16(_mm512_and_si512(v[g * 4 + 3], hmask), 2));
            _mm512_storeu_si512((__m512i *)ph, hbits); ph += 64;
        }
    }
}

ढाँचा <>
अंतरभूत व्योम pack_qs<block_iq4_xs>(व्योम * RESTRICT packed_B, स्थिर block_iq4_xs * RESTRICT B, पूर्णांक KB) {
    __m512i v[16];
    अक्षर * pb = (अक्षर *)packed_B;
    क्रम (पूर्णांक k = 0; k < QK_K / 64; ++k) {
        क्रम (पूर्णांक n = 0; n < TILE_N; ++n) {
            __m256i r0 = bytes_from_nibbles_32(B[n * KB].qs + k * 32 +  0);
            __m256i r1 = bytes_from_nibbles_32(B[n * KB].qs + k * 32 + 16);
            v[n] = _mm512_inserti32x8(_mm512_castsi256_si512(r0), r1, 1);
        }

        transpose_16x16_32bit(v);

        // pack again with 128 to fully utilize vector length
        क्रम (पूर्णांक n = 0; n < TILE_N; n += 2) {
            _mm512_storeu_si512((__m512i *)pb, packNibbles(v[n], v[n + 1]));
            pb += 64;
        }
    }
}

// pack B to vnni formats in 4bits or 8 bits
व्योम pack_B(व्योम * RESTRICT packed_B, स्थिर block_q4_0 * RESTRICT B, पूर्णांक KB) {
    pack_qs(packed_B, B, KB);
    ggml_half * d0 = reinterpret_cast<ggml_half *>((अक्षर *)packed_B + TILE_N * TILE_K / 2);
    क्रम (पूर्णांक n = 0; n < TILE_N; ++n) {
        d0[n] = B[n * KB].d;
    }
}

व्योम pack_B(व्योम * RESTRICT packed_B, स्थिर block_q4_1 * RESTRICT B, पूर्णांक KB) {
    pack_qs(packed_B, B, KB);
    ggml_half * d0 = reinterpret_cast<ggml_half *>((अक्षर *)packed_B + TILE_N * TILE_K / 2);
    ggml_half * m0 = d0 + TILE_N;
    क्रम (पूर्णांक n = 0; n < TILE_N; ++n) {
        d0[n] = B[n * KB].d;
        m0[n] = B[n * KB].m;
    }
}

अंतरभूत व्योम s8s8_compensation(व्योम * RESTRICT packed_B) {
    // packed_B layout:
    //   quants {TILE_N, TILEK}  int8_t
    //   d0     {TILE_N}      ggml_half
    //   comp   {TILE_N}        int32_t
    स्थिर पूर्णांक offset = TILE_N * TILE_K + TILE_N * माप(ggml_half);
    __m512i vcomp = _mm512_setzero_si512();
    स्थिर __m512i off = _mm512_set1_epi8(static_cast<अक्षर>(0x80));
    क्रम (पूर्णांक k = 0; k < 8; ++k) {
        __m512i vb = _mm512_loadu_si512((स्थिर __m512i *)((स्थिर अक्षर *)packed_B + k * 64));
        vcomp = _mm512_dpbusd_epi32(vcomp, off, vb);
    }
    _mm512_storeu_si512((__m512i *)((अक्षर *)(packed_B) + offset), vcomp);
}

व्योम pack_B(व्योम * RESTRICT packed_B, स्थिर block_q8_0 * RESTRICT B, पूर्णांक KB) {
    pack_qs(packed_B, B, KB);
    ggml_half * d0 = reinterpret_cast<ggml_half *>((अक्षर *)packed_B + TILE_N * TILE_K);
    क्रम (पूर्णांक n = 0; n < TILE_N; ++n) {
        d0[n] = B[n * KB].d;
    }
    s8s8_compensation(packed_B);
}

// convert 8 * {min, scale} from int6 to int8
अंतरभूत व्योम unpack_mins_and_scales(स्थिर uint8_t * scales, uint32_t * utmp) {
    स्थिर uint32_t kmask1 = 0x3f3f3f3f;
    स्थिर uint32_t kmask2 = 0x0f0f0f0f;
    स्थिर uint32_t kmask3 = 0x03030303;

    स_नकल(utmp, scales, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    स्थिर uint32_t uaux = utmp[1] & kmask1;
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
व्योम pack_B(व्योम * RESTRICT packed_B, स्थिर block_q4_K * RESTRICT B, पूर्णांक KB) {
    pack_qs(packed_B, B, KB);

    uint8_t * scales = reinterpret_cast<uint8_t *>((अक्षर *)packed_B + (QK_K / 2) * TILE_N);
    uint8_t * mins = scales + 8 * TILE_N;
    ggml_half * d = reinterpret_cast<ggml_half *>(mins + 8 * TILE_N);
    ggml_half * dmin = d + TILE_N;

    जोड़ {
        uint32_t u32[4];
        uint8_t  u8[16];
    } s;

    क्रम (पूर्णांक n = 0; n < TILE_N; ++n) {
        unpack_mins_and_scales(B[n * KB].scales, s.u32);
        क्रम (पूर्णांक k = 0; k < 8; ++k) {
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
व्योम pack_B(व्योम * RESTRICT packed_B, स्थिर block_q5_K * RESTRICT B, पूर्णांक KB) {
    pack_qs(packed_B, B, KB);

    uint8_t * scales = reinterpret_cast<uint8_t *>((अक्षर *)packed_B + (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N);
    uint8_t * mins = scales + 8 * TILE_N;
    ggml_half * d = reinterpret_cast<ggml_half *>(mins + 8 * TILE_N);
    ggml_half * dmin = d + TILE_N;

    जोड़ {
        uint32_t u32[4];
        uint8_t  u8[16];
    } s;

    क्रम (पूर्णांक n = 0; n < TILE_N; ++n) {
        unpack_mins_and_scales(B[n * KB].scales, s.u32);
        क्रम (पूर्णांक k = 0; k < 8; ++k) {
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
व्योम pack_B(व्योम * RESTRICT packed_B, स्थिर block_q6_K * RESTRICT B, पूर्णांक KB) {
    pack_qs(packed_B, B, KB);

    uint8_t * scales = reinterpret_cast<uint8_t *>((अक्षर *)packed_B + (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N);
    ggml_half * d = reinterpret_cast<ggml_half *>(scales + 16 * TILE_N);
    क्रम (पूर्णांक n = 0; n < TILE_N; ++n) {
        स्थिर int8_t * ps = B[n * KB].scales;
        क्रम (पूर्णांक k = 0; k < 16; ++k) {
            scales[k * TILE_N + n] = ps[k];
        }
        d[n] = B[n * KB].d;
    }
}

// packed_B layout:
//   quants {8, TILE_N, 16}  uint8
//   scales {8, TILE_N}       int8
//   d      {TILE_N}     ggml_half
व्योम pack_B(व्योम * RESTRICT packed_B, स्थिर block_iq4_xs * RESTRICT B, पूर्णांक KB) {
    pack_qs(packed_B, B, KB);

    int8_t * scales = reinterpret_cast<int8_t *>((अक्षर *)packed_B + (QK_K / 2) * TILE_N);
    ggml_half * d = reinterpret_cast<ggml_half *>(scales + 8 * TILE_N);

    // pack the scales
    क्रम (पूर्णांक n = 0; n < TILE_N; ++n) {
        uint16_t sh = B[n * KB].scales_h;
        क्रम (पूर्णांक k = 0; k < 8; k += 2) {
            स्थिर int16_t ls1 = ((B[n * KB].scales_l[k / 2] & 0xf) | ((sh << 4) & 0x30)) - 32;
            स्थिर int16_t ls2 = ((B[n * KB].scales_l[k / 2] >>  4) | ((sh << 2) & 0x30)) - 32;
            scales[(k + 0) * TILE_N + n] = ls1;
            scales[(k + 1) * TILE_N + n] = ls2;
            sh >>= 4;
        }
        d[n] = B[n * KB].d;
    }
}

ढाँचा<typename TB, typename packed_B_t = packed_B_type<TB>>
व्योम unpack_B(packed_B_t * RESTRICT tile, स्थिर व्योम * RESTRICT packed_B) {
    GGML_UNUSED(tile);
    GGML_UNUSED(packed_B);
}

ढाँचा <>
व्योम unpack_B<block_q4_0>(int8_t * RESTRICT tile, स्थिर व्योम * RESTRICT packed_B) {
  स्थिर __m512i off = _mm512_set1_epi8(8);
  स्थिर __m512i निचला_Mask = _mm512_set1_epi8(0xF);
  क्रम (पूर्णांक n = 0; n < 8; n += 2) {
    __m512i bytes = _mm512_loadu_si512((स्थिर __m512i *)((स्थिर अक्षर *)packed_B + n * 32));
    स्थिर __m512i r0 = _mm512_sub_epi8(_mm512_and_si512(bytes, निचला_Mask), off);
    स्थिर __m512i r1 = _mm512_sub_epi8(_mm512_and_si512(_mm512_srli_epi16(bytes, 4), निचला_Mask), off);
    _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
    _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
  }
}

ढाँचा <>
व्योम unpack_B<block_q4_1>(uint8_t * RESTRICT tile, स्थिर व्योम * RESTRICT packed_B) {
    स्थिर __m512i निचला_Mask = _mm512_set1_epi8(0xF);
    क्रम (पूर्णांक n = 0; n < 8; n += 2) {
        __m512i bytes = _mm512_loadu_si512((स्थिर __m512i *)((स्थिर अक्षर *)packed_B + n * 32));
        स्थिर __m512i r0 = _mm512_and_si512(bytes, निचला_Mask);
        स्थिर __m512i r1 = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), निचला_Mask);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
    }
}

// packed_B_t for QKK is int8_t
ढाँचा <typename TB>
व्योम unpack_B(int8_t * RESTRICT tile, स्थिर व्योम * RESTRICT packed_B, पूर्णांक k) {
    स्थिर पूर्णांक packed_B_group_आकार = QK_K / 2 * TILE_N / 8;
    स्थिर अक्षर * packed_B_group = (स्थिर अक्षर *)packed_B + k * packed_B_group_आकार;
    स्थिर __m512i निचला_Mask = _mm512_set1_epi8(0xF);
    क्रम (पूर्णांक n = 0; n < 8; n += 2) {
        __m512i bytes = _mm512_loadu_si512(packed_B_group + n * 32);
        स्थिर __m512i r0 = _mm512_and_si512(bytes, निचला_Mask);
        स्थिर __m512i r1 = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), निचला_Mask);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
    }
}

ढाँचा <>
व्योम unpack_B<block_q5_K>(int8_t * RESTRICT tile, स्थिर व्योम * RESTRICT packed_B, पूर्णांक k) {
    // lower 4bits, stride 256 bytes
    स्थिर पूर्णांक packed_l4_group_आकार = QK_K / 2 * TILE_N / 8;
    स्थिर अक्षर * pb = (स्थिर अक्षर *)packed_B + k * packed_l4_group_आकार;

    // higher 1bit, stride 64 bytes
    स्थिर पूर्णांक packed_h1_group_आकार = QK_K / 8 * TILE_N / 8;
    स्थिर अक्षर * ph = (स्थिर अक्षर *)packed_B + (QK_K / 2) * TILE_N + k * packed_h1_group_आकार;
    स्थिर __m512i hbits = _mm512_loadu_si512(ph);

    स्थिर __m512i निचला_Mask = _mm512_set1_epi8(0xF);
    __m512i hmask0 = _mm512_set1_epi8(0x1);
    __m512i hmask1 = _mm512_set1_epi8(0x2);

    क्रम (पूर्णांक n = 0; n < 8; n += 2) {
        __m512i bytes = _mm512_loadu_si512(pb + n * 32);
        __m512i r0 = _mm512_and_si512(bytes, निचला_Mask);
        __m512i r1 = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), निचला_Mask);
        __m512i h0 = _mm512_slli_epi16(_mm512_srli_epi16(_mm512_and_si512(hbits, hmask0), n), 4);
        __m512i h1 = _mm512_slli_epi16(_mm512_srli_epi16(_mm512_and_si512(hbits, hmask1), n + 1), 4);

        hmask0 = _mm512_slli_epi16(hmask0, 2);
        hmask1 = _mm512_slli_epi16(hmask1, 2);
        r0 = _mm512_add_epi8(r0, h0);
        r1 = _mm512_add_epi8(r1, h1);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
    }
}

ढाँचा <>
व्योम unpack_B<block_q6_K>(int8_t * RESTRICT tile, स्थिर व्योम * RESTRICT packed_B, पूर्णांक k) {
    // lower 4bits, stride 128 bytes
    स्थिर पूर्णांक packed_l4_group_आकार = QK_K / 2 * TILE_N / 16;
    स्थिर अक्षर * pb = (स्थिर अक्षर *)packed_B + k * packed_l4_group_आकार;

    // higher 2bits, stride 64 bytes
    स्थिर पूर्णांक packed_h2_group_आकार = QK_K / 4 * TILE_N / 16;
    स्थिर अक्षर * ph = (स्थिर अक्षर *)packed_B + (QK_K / 2) * TILE_N + k * packed_h2_group_आकार;
    स्थिर __m512i hbits = _mm512_loadu_si512(ph);

    स्थिर __m512i off = _mm512_set1_epi8(32);
    स्थिर __m512i निचला_Mask = _mm512_set1_epi8(0xF);
    __m512i hmask0 = _mm512_set1_epi8(0x3); // 0011
    __m512i hmask1 = _mm512_set1_epi8(0xC); // 1100

    // notes: skip zero padding from row4 to row7 as we have done so in `unpack_A`
    __m512i bytes = _mm512_loadu_si512(pb);
    __m512i r0 = _mm512_and_si512(bytes, निचला_Mask);
    __m512i r1 = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), निचला_Mask);
    __m512i h0 = _mm512_slli_epi16(_mm512_and_si512(hbits, hmask0), 4);
    __m512i h1 = _mm512_slli_epi16(_mm512_and_si512(hbits, hmask1), 2);
    _mm512_storeu_si512((__m512i *)(tile +  0), _mm512_sub_epi8(_mm512_add_epi8(r0, h0), off));
    _mm512_storeu_si512((__m512i *)(tile + 64), _mm512_sub_epi8(_mm512_add_epi8(r1, h1), off));

    hmask0 = _mm512_slli_epi16(hmask0, 4);
    hmask1 = _mm512_slli_epi16(hmask1, 4);

    bytes = _mm512_loadu_si512(pb + 64);
    r0 = _mm512_and_si512(bytes, निचला_Mask);
    r1 = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), निचला_Mask);
    h0 =                   _mm512_and_si512(hbits, hmask0);
    h1 = _mm512_srli_epi16(_mm512_and_si512(hbits, hmask1), 2);
    _mm512_storeu_si512((__m512i *)(tile + 128), _mm512_sub_epi8(_mm512_add_epi8(r0, h0), off));
    _mm512_storeu_si512((__m512i *)(tile + 192), _mm512_sub_epi8(_mm512_add_epi8(r1, h1), off));
}

ढाँचा <>
व्योम unpack_B<block_iq4_xs>(int8_t * RESTRICT tile, स्थिर व्योम * RESTRICT packed_B, पूर्णांक k) {
    अटल स्थिर __m512i values128 = _mm512_set_epi8(
        113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
        113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
        113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
        113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127
    );

    स्थिर पूर्णांक packed_B_group_आकार = QK_K / 2 * TILE_N / 8;
    स्थिर अक्षर * pb = (स्थिर अक्षर *)packed_B + k * packed_B_group_आकार;
    स्थिर __m512i निचला_Mask = _mm512_set1_epi8(0xF);

    क्रम (पूर्णांक n = 0; n < 8; n += 2) {
        __m512i bytes = _mm512_loadu_si512(pb + n * 32);
        स्थिर __m512i r0 = _mm512_shuffle_epi8(values128, _mm512_and_si512(bytes, निचला_Mask));
        स्थिर __m512i r1 = _mm512_shuffle_epi8(values128, _mm512_and_si512(_mm512_srli_epi16(bytes, 4), निचला_Mask));
        _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
    }
}

ढाँचा <typename TA, typename TB, bool is_acc>
काष्ठा acc_C {};

ढाँचा <bool is_acc>
काष्ठा acc_C<block_q8_0, block_q4_0, is_acc> {
    अटल व्योम apply(भग्न * RESTRICT C, पूर्णांक ldc, स्थिर int32_t * RESTRICT tile, स्थिर block_q8_0 * A, पूर्णांक lda, स्थिर व्योम * packed_B, पूर्णांक nr) {
        स्थिर पूर्णांक offset = TILE_N * TILE_K / 2;
        स्थिर __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((स्थिर __m256i *)((स्थिर अक्षर *)packed_B + offset)));

        क्रम (पूर्णांक m = 0; m < nr; ++m) {
            स्थिर __m512 vd1 = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(A[m * lda].d));
            स्थिर __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            अगर (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } अन्यथा {
                vsum = _mm512_set1_ps(0.f);
            }
            vsum = _mm512_fmadd_ps(vtile, _mm512_mul_ps(vd0, vd1), vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

ढाँचा <bool is_acc>
काष्ठा acc_C<block_q8_1, block_q4_1, is_acc> {
    अटल व्योम apply(भग्न * RESTRICT C, पूर्णांक ldc, स्थिर int32_t * RESTRICT tile, स्थिर block_q8_1 * A, पूर्णांक lda, स्थिर व्योम * packed_B, पूर्णांक nr) {
        स्थिर पूर्णांक offset = TILE_N * TILE_K / 2;
        स्थिर __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((स्थिर __m256i *)((स्थिर अक्षर *)packed_B + offset)));
        स्थिर __m512 vm0 = _mm512_cvtph_ps(_mm256_loadu_si256((स्थिर __m256i *)((स्थिर अक्षर *)packed_B + offset + TILE_N * माप(ggml_half))));

        क्रम (पूर्णांक m = 0; m < nr; ++m) {
            स्थिर __m512 vd1 = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(A[m * lda].d));
            स्थिर __m512 vs1 = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(A[m * lda].s));
            स्थिर __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            अगर (is_acc) {
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

ढाँचा <bool is_acc>
काष्ठा acc_C<block_q8_0, block_q8_0, is_acc> {
    अटल व्योम apply(भग्न * RESTRICT C, पूर्णांक ldc, स्थिर int32_t * RESTRICT tile, स्थिर block_q8_0 * A, पूर्णांक lda, स्थिर व्योम * packed_B, पूर्णांक nr) {
        स्थिर पूर्णांक offset = TILE_N * TILE_K;
        स्थिर __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((स्थिर __m256i *)((स्थिर अक्षर *)packed_B + offset)));

        क्रम (पूर्णांक m = 0; m < nr; ++m) {
            स्थिर __m512 vd1 = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(A[m * lda].d));
            स्थिर __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            अगर (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } अन्यथा {
                vsum = _mm512_set1_ps(0.f);
            }
            vsum = _mm512_fmadd_ps(vtile, _mm512_mul_ps(vd0, vd1), vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

ढाँचा <bool is_acc>
काष्ठा acc_C<block_q8_K, block_q4_K, is_acc> {
    अटल व्योम apply(भग्न * RESTRICT C, पूर्णांक ldc, स्थिर int32_t * RESTRICT tile, स्थिर block_q8_K * A, पूर्णांक lda, स्थिर व्योम * packed_B, पूर्णांक nr) {
        स्थिर uint8_t * scales = reinterpret_cast<स्थिर uint8_t *>((स्थिर अक्षर *)packed_B + (QK_K / 2) * TILE_N);
        स्थिर uint8_t * mins = scales + 8 * TILE_N;
        स्थिर ggml_half * d0 = reinterpret_cast<स्थिर ggml_half *>(mins + 8 * TILE_N);
        स्थिर ggml_half * dmin = d0 + TILE_N;

        स्थिर __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((स्थिर __m256i *)d0));
        स्थिर __m512 vdmin = _mm512_cvtph_ps(_mm256_loadu_si256((स्थिर __m256i *)dmin));

        क्रम (पूर्णांक m = 0; m < nr; ++m) {
            स्थिर भग्न d1 = A[m * lda].d;
            स्थिर __m512 vd = _mm512_mul_ps(_mm512_set1_ps(d1), vd0);
            स्थिर __m512 vdm = _mm512_mul_ps(_mm512_set1_ps(-d1), vdmin);
            स्थिर __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            अगर (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } अन्यथा {
                vsum = _mm512_set1_ps(0.f);
            }

            स्थिर __m256i q8sums = _mm256_loadu_si256((स्थिर __m256i *)A[m * lda].bsums);
            स्थिर __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));

            __m512i acc_m = _mm512_setzero_si512();
            क्रम (पूर्णांक k = 0; k < 4; ++k) {
                __m512i vmask = _mm512_set1_epi32(k);
                __m512i va = _mm512_permutexvar_epi32(vmask, _mm512_castsi128_si512(q8s));
                __m512i vb = _mm512_cvtepi8_epi16(_mm256_loadu_si256((स्थिर __m256i *)(mins + k * 32)));
                acc_m = _mm512_dpwssds_epi32(acc_m, va, vb);
            }

            vsum = _mm512_fmadd_ps(vtile, vd, vsum);
            vsum = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc_m), vdm, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

ढाँचा <bool is_acc>
काष्ठा acc_C<block_q8_K, block_q5_K, is_acc> {
    अटल व्योम apply(भग्न * RESTRICT C, पूर्णांक ldc, स्थिर int32_t * RESTRICT tile, स्थिर block_q8_K * A, पूर्णांक lda, स्थिर व्योम * packed_B, पूर्णांक nr) {
        स्थिर uint8_t * scales = reinterpret_cast<स्थिर uint8_t *>((स्थिर अक्षर *)packed_B + (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N);
        स्थिर uint8_t * mins = scales + 8 * TILE_N;
        स्थिर ggml_half * d0 = reinterpret_cast<स्थिर ggml_half *>(mins + 8 * TILE_N);
        स्थिर ggml_half * dmin = d0 + TILE_N;

        स्थिर __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((स्थिर __m256i *)d0));
        स्थिर __m512 vdmin = _mm512_cvtph_ps(_mm256_loadu_si256((स्थिर __m256i *)dmin));

        क्रम (पूर्णांक m = 0; m < nr; ++m) {
            स्थिर भग्न d1 = A[m * lda].d;
            स्थिर __m512 vd = _mm512_mul_ps(_mm512_set1_ps(d1), vd0);
            स्थिर __m512 vdm = _mm512_mul_ps(_mm512_set1_ps(-d1), vdmin);
            स्थिर __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            अगर (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } अन्यथा {
                vsum = _mm512_set1_ps(0.f);
            }

            स्थिर __m256i q8sums = _mm256_loadu_si256((स्थिर __m256i *)A[m * lda].bsums);
            स्थिर __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));

            __m512i acc_m = _mm512_setzero_si512();
            क्रम (पूर्णांक k = 0; k < 4; ++k) {
                __m512i vmask = _mm512_set1_epi32(k);
                __m512i va = _mm512_permutexvar_epi32(vmask, _mm512_castsi128_si512(q8s));
                __m512i vb = _mm512_cvtepi8_epi16(_mm256_loadu_si256((स्थिर __m256i *)(mins + k * 32)));
                acc_m = _mm512_dpwssds_epi32(acc_m, va, vb);
            }

            vsum = _mm512_fmadd_ps(vtile, vd, vsum);
            vsum = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc_m), vdm, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

ढाँचा <bool is_acc>
काष्ठा acc_C<block_q8_K, block_q6_K, is_acc> {
    अटल व्योम apply(भग्न * RESTRICT C, पूर्णांक ldc, स्थिर int32_t * RESTRICT tile, स्थिर block_q8_K * A, पूर्णांक lda, स्थिर व्योम * packed_B, पूर्णांक nr) {
        स्थिर uint8_t * scales = reinterpret_cast<स्थिर uint8_t *>((स्थिर अक्षर *)packed_B + (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N);
        स्थिर ggml_half * d0 = reinterpret_cast<स्थिर ggml_half *>(scales + 16 * TILE_N);

        स्थिर __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((स्थिर __m256i *)d0));

        क्रम (पूर्णांक m = 0; m < nr; ++m) {
            स्थिर भग्न d1 = A[m * lda].d;
            स्थिर __m512 vd = _mm512_mul_ps(_mm512_set1_ps(d1), vd0);
            स्थिर __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            अगर (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } अन्यथा {
                vsum = _mm512_set1_ps(0.f);
            }

            vsum = _mm512_fmadd_ps(vtile, vd, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

ढाँचा <bool is_acc>
काष्ठा acc_C<block_q8_K, block_iq4_xs, is_acc> {
    अटल व्योम apply(भग्न * RESTRICT C, पूर्णांक ldc, स्थिर int32_t * RESTRICT tile, स्थिर block_q8_K * A, पूर्णांक lda, स्थिर व्योम * packed_B, पूर्णांक nr) {
        स्थिर int8_t * scales = reinterpret_cast<स्थिर int8_t *>((स्थिर अक्षर *)packed_B + (QK_K / 2) * TILE_N);
        स्थिर ggml_half * d0 = reinterpret_cast<स्थिर ggml_half *>(scales + 8 * TILE_N);

        स्थिर __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((स्थिर __m256i *)d0));

        क्रम (पूर्णांक m = 0; m < nr; ++m) {
            स्थिर भग्न d1 = A[m * lda].d;
            स्थिर __m512 vd = _mm512_mul_ps(_mm512_set1_ps(d1), vd0);
            स्थिर __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            अगर (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } अन्यथा {
                vsum = _mm512_set1_ps(0.f);
            }

            vsum = _mm512_fmadd_ps(vtile, vd, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

ढाँचा <typename TB> constexpr पूर्णांक get_quants_आकार();
ढाँचा <> constexpr पूर्णांक get_quants_आकार<block_q4_K>() { वापस (QK_K / 2) * TILE_N; }
ढाँचा <> constexpr पूर्णांक get_quants_आकार<block_q5_K>() { वापस (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N; }
ढाँचा <> constexpr पूर्णांक get_quants_आकार<block_q6_K>() { वापस (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N; }
ढाँचा <> constexpr पूर्णांक get_quants_आकार<block_iq4_xs>() { वापस (QK_K / 2) * TILE_N; }

// used for QKK format
ढाँचा <typename TB, bool is_acc,
          typename std::enable_if<is_type_qkk<TB>::value, पूर्णांक>::type = 0>
अंतरभूत व्योम scale_C(स्थिर int32_t * RESTRICT tile, int32_t * RESTRICT sumi, स्थिर व्योम * packed_B, पूर्णांक k, पूर्णांक nr) {
    स्थिर uint8_t * scales = reinterpret_cast<स्थिर uint8_t *>((स्थिर अक्षर *)packed_B + get_quants_आकार<TB>());
    स्थिर __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((स्थिर __m128i *)(scales + k * TILE_N)));

    क्रम (पूर्णांक m = 0; m < nr; ++m) {
        __m512i vsumi;
        अगर (is_acc) {
            vsumi = _mm512_loadu_si512(sumi + m * TILE_N);
        } अन्यथा {
            vsumi = _mm512_setzero_si512();
        }
        __m512i vtile = _mm512_loadu_si512(tile + m * TILE_N);
        vsumi = _mm512_add_epi32(vsumi, _mm512_mullo_epi32(vtile, vscale));
        _mm512_storeu_si512((__m512i *)(sumi + m * TILE_N), vsumi);
    }
}

ढाँचा <typename TA, typename TB, typename TC, पूर्णांक BLOCK_M, पूर्णांक BLOCK_N, पूर्णांक BLOCK_K>
काष्ठा tinygemm_kernel_avx {
    अटल व्योम apply(पूर्णांक K, स्थिर TA * RESTRICT A, स्थिर TB * RESTRICT B, TC * RESTRICT C, पूर्णांक ldc) {
        GGML_UNUSED(K);
        GGML_UNUSED(A);
        GGML_UNUSED(B);
        GGML_UNUSED(C);
        GGML_UNUSED(ldc);
    }
};

ढाँचा <पूर्णांक BLOCK_M, पूर्णांक BLOCK_N, पूर्णांक BLOCK_K>
काष्ठा tinygemm_kernel_avx<भग्न, ggml_fp16_t, भग्न, BLOCK_M, BLOCK_N, BLOCK_K> {
    अटल व्योम apply(पूर्णांक K, स्थिर भग्न * RESTRICT A, स्थिर ggml_fp16_t * RESTRICT B, भग्न * RESTRICT C, पूर्णांक ldc) {
        constexpr पूर्णांक ROWS = BLOCK_M;
        constexpr पूर्णांक COLS = BLOCK_N;
        निश्चित(BLOCK_K == 16);

        __m512 va;
        __m512 vb[COLS];
        __m512 vc[ROWS * COLS];

        स्वतः loadc = [&](स्वतः idx) {
            vc[idx] = _mm512_setzero_ps();
        };
        Unroll<ROWS * COLS>{}(loadc);

        स्वतः compute = [&](स्वतः idx, स्वतः k) {
            constexpr पूर्णांक row = idx / COLS;
            constexpr पूर्णांक col = idx % COLS;

            अगर constexpr (col == 0) {
                va = _mm512_loadu_ps(A + row * K + k);
            }
            अगर constexpr (row == 0) {
                vb[col] =  _mm512_cvtph_ps(_mm256_loadu_si256((स्थिर __m256i *)(B + col * K + k)));
            }
            vc[idx] = _mm512_fmadd_ps(va, vb[col], vc[idx]);
        };

        क्रम (पूर्णांक k = 0; k < K; k += 16) {
            Unroll<ROWS * COLS>{}(compute, k);
        }

        स्वतः storec = [&](स्वतः idx) {
            constexpr पूर्णांक row = idx / COLS;
            constexpr पूर्णांक col = idx % COLS;
            C[row * ldc + col] = _mm512_reduce_add_ps(vc[idx]);
        };
        Unroll<ROWS * COLS>{}(storec);
    }
};

#घोषणा LAUNCH_TINYGEMM_KERNEL_AVX(MB_आकार, NB_आकार)                                \
    tinygemm_kernel_avx<भग्न, type, भग्न, MB_आकार, NB_आकार, blck_आकार>::apply(    \
        K, (स्थिर भग्न *)src1->तथ्य + src1_offset + mb_start * K,                  \
        (स्थिर type *)src0->तथ्य + src0_offset + nb_start * K,                      \
        (भग्न *)dst->तथ्य + dst_offset + mb_start * ldc + nb_start, ldc)


// re-organize in the format {NB, KB, TILE_SIZE}:
#घोषणा PACKED_सूचकांक(n, k, KB, tile_आकार) (n * KB + k) * tile_आकार

ढाँचा<typename TB, पूर्णांक BLOCK_K>
व्योम convert_B_packed_format(व्योम * RESTRICT packed_B, स्थिर TB * RESTRICT B, पूर्णांक N, पूर्णांक K) {
    स्थिर पूर्णांक NB = N / TILE_N;
    स्थिर पूर्णांक KB = K / BLOCK_K;
    स्थिर पूर्णांक TILE_आकार = get_tile_आकार<TB>();

    // parallel on NB should be enough
    parallel_for(NB, [&](पूर्णांक begin, पूर्णांक end) {
        क्रम (पूर्णांक n = begin; n < end; ++n) {
            क्रम (पूर्णांक k = 0; k < KB; ++k) {
                पूर्णांक n0 = n * TILE_N;
                pack_B((अक्षर *)packed_B + PACKED_सूचकांक(n, k, KB, TILE_आकार), &B[n0 * KB + k], KB);
            }
        }
    });
}

ढाँचा <typename TA, typename TB, typename TC, पूर्णांक BLOCK_M, पूर्णांक BLOCK_N, पूर्णांक BLOCK_K>
काष्ठा tinygemm_kernel_vnni {};

ढाँचा <पूर्णांक BLOCK_M, पूर्णांक BLOCK_N, पूर्णांक BLOCK_K>
काष्ठा tinygemm_kernel_vnni<block_q8_0, block_q4_0, भग्न, BLOCK_M, BLOCK_N, BLOCK_K> {
    अटल व्योम apply(पूर्णांक KB, स्थिर व्योम * RESTRICT _A, स्थिर व्योम * RESTRICT _B, भग्न * RESTRICT C, पूर्णांक ldc) {

        constexpr पूर्णांक COLS = BLOCK_N / 16;
        स्थिर पूर्णांक TILE_आकार = TILE_N * माप(block_q4_0);

        स्थिर block_q8_0 * RESTRICT A = static_cast<स्थिर block_q8_0 *>(_A);
        स्थिर अक्षर * RESTRICT B = static_cast<स्थिर अक्षर *>(_B);

        __m512i va[8];
        __m512 vc[COLS];
        __m512 vd1;

        // sum of offsets, shared across COLS
        //
        // avx512-vnni does not have `_mm512_dpbssd_epi32`,
        // need to transform ss to us:
        //   a * (b - 8) is equivalent to b * a - 8 * a
        //   s    u   u                   u   s   u   s
        //
        __m512i vcomp;

        स्थिर __m512i off = _mm512_set1_epi8(8);
        स्थिर __m512i निचला_Mask = _mm512_set1_epi8(0xF);

        स्वतः loadc = [&](स्वतः col) {
            vc[col] = _mm512_setzero_ps();
        };
        Unroll<COLS>{}(loadc);

        स्वतः compute = [&](स्वतः col, स्वतः i) {
            // load a and compute compensation
            अगर constexpr (col == 0) {
                स्थिर int32_t * a_ptr = reinterpret_cast<स्थिर int32_t *>(A[0 * KB + i].qs);
                vcomp = _mm512_setzero_si512();
                क्रम (पूर्णांक k = 0; k < 8; ++k) {
                    va[k] = _mm512_set1_epi32(a_ptr[k]);
                    vcomp = _mm512_dpbusd_epi32(vcomp, off, va[k]);
                }
                vd1 = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(A[0 * KB + i].d));
            }

            // load b
            __m512i vsum = _mm512_setzero_si512();
            स्थिर अक्षर * b_ptr = B + PACKED_सूचकांक(col, i, KB, TILE_आकार);
            क्रम (पूर्णांक k = 0; k < 8; k += 2) {
                __m512i bytes = _mm512_loadu_si512((स्थिर __m512i *)(b_ptr + k * 32));
                __m512i vb0 = _mm512_and_si512(bytes, निचला_Mask);
                vsum = _mm512_dpbusd_epi32(vsum, vb0, va[k + 0]);
                __m512i vb1 = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), निचला_Mask);
                vsum = _mm512_dpbusd_epi32(vsum, vb1, va[k + 1]);
            }
            स्थिर पूर्णांक offset = TILE_N * TILE_K / 2;
            स्थिर __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((स्थिर __m256i *)(b_ptr + offset)));
            vsum = _mm512_sub_epi32(vsum, vcomp);

            vc[col] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vsum), _mm512_mul_ps(vd0, vd1), vc[col]);
        };

        क्रम (पूर्णांक i = 0; i < KB; ++i) {
            Unroll<COLS>{}(compute, i);
        }

        //store to C
        स्वतः storec = [&](स्वतः col) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + col * 16), vc[col]);
        };
        Unroll<COLS>{}(storec);
    }
};

ढाँचा <पूर्णांक BLOCK_N, पूर्णांक BLOCK_K>
काष्ठा tinygemm_kernel_vnni<block_q8_1, block_q4_1, भग्न, 1, BLOCK_N, BLOCK_K> {
    अटल व्योम apply(पूर्णांक KB, स्थिर व्योम * RESTRICT _A, स्थिर व्योम * RESTRICT _B, भग्न * RESTRICT C, पूर्णांक ldc) {

        constexpr पूर्णांक COLS = BLOCK_N / 16;
        स्थिर पूर्णांक TILE_आकार = TILE_N * माप(block_q4_1);

        स्थिर block_q8_1 * RESTRICT A = static_cast<स्थिर block_q8_1 *>(_A);
        स्थिर अक्षर * RESTRICT B = static_cast<स्थिर अक्षर *>(_B);

        __m512i va[8];
        __m512i vb[8];
        __m512 vc[COLS];
        __m512 vd1, vs1;

        स्थिर __m512i निचला_Mask = _mm512_set1_epi8(0xF);

        स्वतः loadc = [&](स्वतः col) {
            vc[col] = _mm512_setzero_ps();
        };
        Unroll<COLS>{}(loadc);

        स्वतः compute = [&](स्वतः col, स्वतः i) {
            // load a
            अगर constexpr (col == 0) {
                स्थिर int32_t * a_ptr = reinterpret_cast<स्थिर int32_t *>(A[0 * KB + i].qs);
                क्रम (पूर्णांक k = 0; k < 8; ++k) {
                    va[k] = _mm512_set1_epi32(a_ptr[k]);
                }
                vd1 = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(A[0 * KB + i].d));
                vs1 = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(A[0 * KB + i].s));
            }

            // load b
            स्थिर अक्षर * b_ptr = B + PACKED_सूचकांक(col, i, KB, TILE_आकार);
            क्रम (पूर्णांक k = 0; k < 8; k += 2) {
                __m512i bytes = _mm512_loadu_si512((स्थिर __m512i *)(b_ptr + k * 32));
                vb[k + 0] = _mm512_and_si512(bytes, निचला_Mask);
                vb[k + 1] = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), निचला_Mask);
            }
            स्थिर पूर्णांक offset = TILE_N * TILE_K / 2;
            स्थिर __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((स्थिर __m256i *)(b_ptr + offset)));
            स्थिर __m512 vm0 = _mm512_cvtph_ps(_mm256_loadu_si256((स्थिर __m256i *)(b_ptr + offset + TILE_N * माप(ggml_half))));

            __m512i vsum = _mm512_setzero_si512();
            क्रम (पूर्णांक k = 0; k < 8; ++k) {
                vsum = _mm512_dpbusd_epi32(vsum, vb[k], va[k]);
            }

            vc[col] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vsum), _mm512_mul_ps(vd0, vd1), vc[col]);
            vc[col] = _mm512_fmadd_ps(vm0, vs1, vc[col]);
        };

        क्रम (पूर्णांक i = 0; i < KB; ++i) {
            Unroll<COLS>{}(compute, i);
        }

        //store to C
        स्वतः storec = [&](स्वतः col) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + col * 16), vc[col]);
        };
        Unroll<COLS>{}(storec);
    }
};

ढाँचा <पूर्णांक BLOCK_M, पूर्णांक BLOCK_N, पूर्णांक BLOCK_K>
काष्ठा tinygemm_kernel_vnni<block_q8_0, block_q8_0, भग्न, BLOCK_M, BLOCK_N, BLOCK_K> {
    अटल व्योम apply(पूर्णांक KB, स्थिर व्योम * RESTRICT _A, स्थिर व्योम * RESTRICT _B, भग्न * RESTRICT C, पूर्णांक ldc) {

        constexpr पूर्णांक COLS = BLOCK_N / 16;
        स्थिर पूर्णांक TILE_आकार = TILE_N * माप(block_q8_0) + TILE_N * माप(int32_t);

        स्थिर block_q8_0 * RESTRICT A = static_cast<स्थिर block_q8_0 *>(_A);
        स्थिर अक्षर * RESTRICT B = static_cast<स्थिर अक्षर *>(_B);

        __m512i va[8];
        __m512i vb[8];
        __m512 vc[COLS];
        __m512 vd1;

        // Notes: s8s8 igemm compensation in avx512-vnni
        // change s8s8 to u8s8 with compensate
        //   a * b = (a + 128) * b - 128 * b
        //   s   s       u       s    u    s
        //
        // (128 * b is pre-computed when packing B to vnni formats)
        //
        स्थिर __m512i off = _mm512_set1_epi8(static_cast<अक्षर>(0x80));

        स्वतः loadc = [&](स्वतः col) {
            vc[col] = _mm512_setzero_ps();
        };
        Unroll<COLS>{}(loadc);

        स्वतः compute = [&](स्वतः col, स्वतः i) {
            // load a and add offset 128
            अगर constexpr (col == 0) {
                स्थिर int32_t * a_ptr = reinterpret_cast<स्थिर int32_t *>(A[0 * KB + i].qs);
                क्रम (पूर्णांक k = 0; k < 8; ++k) {
                    va[k] = _mm512_set1_epi32(a_ptr[k]);
                    va[k] = _mm512_add_epi8(va[k], off);
                }
                vd1 = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(A[0 * KB + i].d));
            }

            // load b
            स्थिर अक्षर * b_ptr = B + PACKED_सूचकांक(col, i, KB, TILE_आकार);
            क्रम (पूर्णांक k = 0; k < 8; ++k) {
                vb[k] = _mm512_loadu_si512((स्थिर __m512i *)(b_ptr + k * 64));
            }
            स्थिर पूर्णांक offset = TILE_N * TILE_K;
            स्थिर __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((स्थिर __m256i *)(b_ptr + offset)));
            स्थिर पूर्णांक offset2 = TILE_N * TILE_K + TILE_N * माप(ggml_half);
            स्थिर __m512i vcomp = _mm512_loadu_si512((स्थिर __m512i *)(b_ptr + offset2));

            __m512i vsum = _mm512_setzero_si512();
            क्रम (पूर्णांक k = 0; k < 8; ++k) {
                vsum = _mm512_dpbusd_epi32(vsum, va[k], vb[k]);
            }
            vsum = _mm512_sub_epi32(vsum, vcomp);

            vc[col] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vsum), _mm512_mul_ps(vd0, vd1), vc[col]);
        };

        क्रम (पूर्णांक i = 0; i < KB; ++i) {
            Unroll<COLS>{}(compute, i);
        }

        //store to C
        स्वतः storec = [&](स्वतः col) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + col * 16), vc[col]);
        };
        Unroll<COLS>{}(storec);
    }
};

ढाँचा <पूर्णांक BLOCK_M, पूर्णांक BLOCK_N, पूर्णांक BLOCK_K>
काष्ठा tinygemm_kernel_vnni<block_q8_K, block_q4_K, भग्न, BLOCK_M, BLOCK_N, BLOCK_K> {
    अटल व्योम apply(पूर्णांक KB, स्थिर व्योम * RESTRICT _A, स्थिर व्योम * RESTRICT _B, भग्न * RESTRICT C, पूर्णांक ldc) {

        constexpr पूर्णांक COLS = BLOCK_N / 16;
        स्थिर पूर्णांक TILE_आकार = TILE_N * माप(block_q4_K) + TILE_N * 4;

        स्थिर block_q8_K * RESTRICT A = static_cast<स्थिर block_q8_K *>(_A);
        स्थिर अक्षर * RESTRICT B = static_cast<स्थिर अक्षर *>(_B);

        // a.qs:   8 groups, 32 bytes each group (m256i)
        __m512i va[8];
        // a.bsum: 8 groups,  2 bytes each group (m128i)
        __m512i va_bsum;
        __m512 vc[COLS];
        __m512 vd1;

        // packed_B:
        स्थिर पूर्णांक offset_scales = (QK_K / 2) * TILE_N;
        स्थिर पूर्णांक offset_mins   = (QK_K / 2) * TILE_N +  8 * TILE_N;
        स्थिर पूर्णांक offset_d0     = (QK_K / 2) * TILE_N + 16 * TILE_N;
        स्थिर पूर्णांक offset_dmin   = (QK_K / 2) * TILE_N + 16 * TILE_N + TILE_N * माप(ggml_half);

        स्थिर __m512i निचला_Mask = _mm512_set1_epi8(0xF);

        स्वतः loadc = [&](स्वतः col) {
            vc[col] = _mm512_setzero_ps();
        };
        Unroll<COLS>{}(loadc);

        // Notes: vnni formats in QK_K
        //   a) quants vnni format
        //     int8  {k/4, n, 4}, viewed as 2d {k/4, 4n}, k = 32
        //     from {16, 32} to {8, 64}
        //
        //   b) min vnni format
        //     int16 {k/2, n, 2}, viewed as 2d {k/2, 2n}, k = 8
        //     from {16,  8} to {4, 32}
        //
        स्वतः compute = [&](स्वतः col, स्वतः i) {
            // load a
            अगर constexpr (col == 0) {
                क्रम (पूर्णांक k_group = 0; k_group < QK_K / 32; ++k_group) {
                    va[k_group] = _mm512_castsi256_si512(_mm256_loadu_si256((स्थिर __m256i *)(A[0 * KB + i].qs + k_group * 32)));
                }
                स्थिर __m256i q8sums = _mm256_loadu_si256((स्थिर __m256i *)A[0 * KB + i].bsums);
                स्थिर __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
                va_bsum = _mm512_castsi128_si512(q8s);
                vd1 = _mm512_set1_ps(A[0 * KB + i].d);
            }

            // step 1: accumultate the quants
            __m512i acc = _mm512_setzero_si512();
            स्थिर अक्षर * b_ptr = B + PACKED_सूचकांक(col, i, KB, TILE_आकार);
            स्थिर अक्षर * b_qs  = b_ptr;
            क्रम (पूर्णांक k_group = 0; k_group < QK_K / 32; ++k_group) {
                __m512i vsum = _mm512_setzero_si512();
                क्रम (पूर्णांक k = 0; k < 8; k += 2) {
                    __m512i va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(k + 0), va[k_group]);
                    __m512i va1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(k + 1), va[k_group]);

                    __m512i bytes = _mm512_loadu_si512((स्थिर __m512i *)b_qs);
                    __m512i vb0 = _mm512_and_si512(bytes, निचला_Mask);
                    vsum = _mm512_dpbusd_epi32(vsum, vb0, va0);
                    __m512i vb1 = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), निचला_Mask);
                    vsum = _mm512_dpbusd_epi32(vsum, vb1, va1);

                    b_qs += 64;
                }
                // vacc += scale * (q8 @ q4)
                स्थिर __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((स्थिर __m128i *)(b_ptr + offset_scales + k_group * TILE_N)));
                acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(vsum, vscale));
            }
            स्थिर __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((स्थिर __m256i *)(b_ptr + offset_d0)));
            vc[col] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(vd0, vd1), vc[col]);

            // step 2: accumulate the mins
            __m512i acc_m = _mm512_setzero_si512();
            क्रम (पूर्णांक k = 0; k < 4; ++k) {
                __m512i vmask = _mm512_set1_epi32(k);
                __m512i va = _mm512_permutexvar_epi32(vmask, va_bsum);
                __m512i vb = _mm512_cvtepi8_epi16(_mm256_loadu_si256((स्थिर __m256i *)(b_ptr + offset_mins + k * 32)));
                acc_m = _mm512_dpwssds_epi32(acc_m, va, vb);
            }
            स्थिर __m512 vdmin = _mm512_cvtph_ps(_mm256_loadu_si256((स्थिर __m256i *)(b_ptr + offset_dmin)));
            vc[col] = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(acc_m), _mm512_mul_ps(vdmin, vd1), vc[col]);
        };

        क्रम (पूर्णांक i = 0; i < KB; ++i) {
            Unroll<COLS>{}(compute, i);
        }

        //store to C
        स्वतः storec = [&](स्वतः col) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + col * 16), vc[col]);
        };
        Unroll<COLS>{}(storec);
    }
};

ढाँचा <पूर्णांक BLOCK_M, पूर्णांक BLOCK_N, पूर्णांक BLOCK_K>
काष्ठा tinygemm_kernel_vnni<block_q8_K, block_q5_K, भग्न, BLOCK_M, BLOCK_N, BLOCK_K> {
    अटल व्योम apply(पूर्णांक KB, स्थिर व्योम * RESTRICT _A, स्थिर व्योम * RESTRICT _B, भग्न * RESTRICT C, पूर्णांक ldc) {

        constexpr पूर्णांक COLS = BLOCK_N / 16;
        स्थिर पूर्णांक TILE_आकार = TILE_N * माप(block_q5_K) + TILE_N * 4;

        स्थिर block_q8_K * RESTRICT A = static_cast<स्थिर block_q8_K *>(_A);
        स्थिर अक्षर * RESTRICT B = static_cast<स्थिर अक्षर *>(_B);

        // a.qs:   8 groups, 32 bytes each group (m256i)
        __m512i va[8];
        // a.bsum: 8 groups,  2 bytes each group (m128i)
        __m512i va_bsum;
        __m512 vc[COLS];
        __m512 vd1;

        // packed_B:
        स्थिर पूर्णांक offset_qh     = (QK_K / 2) * TILE_N;
        स्थिर पूर्णांक offset_scales = (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N;
        स्थिर पूर्णांक offset_mins   = (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N +  8 * TILE_N;
        स्थिर पूर्णांक offset_d0     = (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N + 16 * TILE_N;
        स्थिर पूर्णांक offset_dmin   = (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N + 16 * TILE_N + TILE_N * माप(ggml_half);

        स्थिर __m512i निचला_Mask = _mm512_set1_epi8(0xF);

        स्वतः loadc = [&](स्वतः col) {
            vc[col] = _mm512_setzero_ps();
        };
        Unroll<COLS>{}(loadc);

        // Q5_K and Q4_K shares the same vnni formats, refer to notes above.
        स्वतः compute = [&](स्वतः col, स्वतः i) {
            // load a
            अगर constexpr (col == 0) {
                क्रम (पूर्णांक k_group = 0; k_group < QK_K / 32; ++k_group) {
                    va[k_group] = _mm512_castsi256_si512(_mm256_loadu_si256((स्थिर __m256i *)(A[0 * KB + i].qs + k_group * 32)));
                }
                स्थिर __m256i q8sums = _mm256_loadu_si256((स्थिर __m256i *)A[0 * KB + i].bsums);
                स्थिर __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
                va_bsum = _mm512_castsi128_si512(q8s);
                vd1 = _mm512_set1_ps(A[0 * KB + i].d);
            }

            // step 1: accumultate the quants
            __m512i acc = _mm512_setzero_si512();
            स्थिर अक्षर * b_ptr = B + PACKED_सूचकांक(col, i, KB, TILE_आकार);
            स्थिर अक्षर * b_qs  = b_ptr;
            स्थिर अक्षर * b_qh  = b_ptr + offset_qh;
            क्रम (पूर्णांक k_group = 0; k_group < QK_K / 32; ++k_group) {
                __m512i vsum = _mm512_setzero_si512();
                __m512i hmask0 = _mm512_set1_epi8(0x1);
                __m512i hmask1 = _mm512_set1_epi8(0x2);
                __m512i hbits = _mm512_loadu_si512((स्थिर __m512i *)(b_qh + k_group * 64));
                क्रम (पूर्णांक k = 0; k < 8; k += 2) {
                    __m512i va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(k + 0), va[k_group]);
                    __m512i va1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(k + 1), va[k_group]);

                    __m512i bytes = _mm512_loadu_si512((स्थिर __m512i *)b_qs);
                    __m512i vb0 = _mm512_and_si512(bytes, निचला_Mask);
                    __m512i vb1 = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), निचला_Mask);

                    __m512i vh0 = _mm512_slli_epi16(_mm512_srli_epi16(_mm512_and_si512(hbits, hmask0), k), 4);
                    __m512i vh1 = _mm512_slli_epi16(_mm512_srli_epi16(_mm512_and_si512(hbits, hmask1), k + 1), 4);

                    hmask0 = _mm512_slli_epi16(hmask0, 2);
                    hmask1 = _mm512_slli_epi16(hmask1, 2);
                    vb0 = _mm512_add_epi8(vb0, vh0);
                    vb1 = _mm512_add_epi8(vb1, vh1);

                    vsum = _mm512_dpbusd_epi32(vsum, vb0, va0);
                    vsum = _mm512_dpbusd_epi32(vsum, vb1, va1);

                    b_qs += 64;
                }
                // vacc += scale * (q8 @ q5)
                स्थिर __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((स्थिर __m128i *)(b_ptr + offset_scales + k_group * TILE_N)));
                acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(vsum, vscale));
            }
            स्थिर __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((स्थिर __m256i *)(b_ptr + offset_d0)));
            vc[col] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(vd0, vd1), vc[col]);

            // step 2: accumulate the mins
            __m512i acc_m = _mm512_setzero_si512();
            क्रम (पूर्णांक k = 0; k < 4; ++k) {
                __m512i vmask = _mm512_set1_epi32(k);
                __m512i va = _mm512_permutexvar_epi32(vmask, va_bsum);
                __m512i vb = _mm512_cvtepi8_epi16(_mm256_loadu_si256((स्थिर __m256i *)(b_ptr + offset_mins + k * 32)));
                acc_m = _mm512_dpwssds_epi32(acc_m, va, vb);
            }
            स्थिर __m512 vdmin = _mm512_cvtph_ps(_mm256_loadu_si256((स्थिर __m256i *)(b_ptr + offset_dmin)));
            vc[col] = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(acc_m), _mm512_mul_ps(vdmin, vd1), vc[col]);
        };

        क्रम (पूर्णांक i = 0; i < KB; ++i) {
            Unroll<COLS>{}(compute, i);
        }

        //store to C
        स्वतः storec = [&](स्वतः col) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + col * 16), vc[col]);
        };
        Unroll<COLS>{}(storec);
    }
};

ढाँचा <पूर्णांक BLOCK_M, पूर्णांक BLOCK_N, पूर्णांक BLOCK_K>
काष्ठा tinygemm_kernel_vnni<block_q8_K, block_q6_K, भग्न, BLOCK_M, BLOCK_N, BLOCK_K> {
    अटल व्योम apply(पूर्णांक KB, स्थिर व्योम * RESTRICT _A, स्थिर व्योम * RESTRICT _B, भग्न * RESTRICT C, पूर्णांक ldc) {

        constexpr पूर्णांक COLS = BLOCK_N / 16;
        स्थिर पूर्णांक TILE_आकार = TILE_N * माप(block_q6_K);

        स्थिर block_q8_K * RESTRICT A = static_cast<स्थिर block_q8_K *>(_A);
        स्थिर अक्षर * RESTRICT B = static_cast<स्थिर अक्षर *>(_B);

        // load the 256 bytes from A to 4 avx512 vectors
        __m512i va[4];
        __m512 vc[COLS];
        __m512 vd1;

        // packed_B:
        स्थिर पूर्णांक offset_qh     = (QK_K / 2) * TILE_N;
        स्थिर पूर्णांक offset_scales = (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N;
        स्थिर पूर्णांक offset_d0     = (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N + 16 * TILE_N;

        // compensation
        __m512i vcomp;

        स्थिर __m512i m32s = _mm512_set1_epi32(32);
        स्थिर __m512i निचला_Mask = _mm512_set1_epi8(0xF);

        स्वतः loadc = [&](स्वतः col) {
            vc[col] = _mm512_setzero_ps();
        };
        Unroll<COLS>{}(loadc);

        स्वतः compute = [&](स्वतः col, स्वतः i) {
            अगर constexpr (col == 0) {
                // load a
                va[0] = _mm512_loadu_si512((स्थिर __m512i *)(A[0 * KB + i].qs +   0));
                va[1] = _mm512_loadu_si512((स्थिर __m512i *)(A[0 * KB + i].qs +  64));
                va[2] = _mm512_loadu_si512((स्थिर __m512i *)(A[0 * KB + i].qs + 128));
                va[3] = _mm512_loadu_si512((स्थिर __m512i *)(A[0 * KB + i].qs + 192));

                स्थिर __m256i q8sums = _mm256_loadu_si256((स्थिर __m256i *)A[0 * KB + i].bsums);
                vcomp = _mm512_mullo_epi32(_mm512_cvtepi16_epi32(q8sums), m32s);
                vd1 = _mm512_set1_ps(A[0 * KB + i].d);
            }

            // accmulate the quants
            __m512i acc = _mm512_setzero_si512();
            स्थिर अक्षर * b_ptr = B + PACKED_सूचकांक(col, i, KB, TILE_आकार);
            स्थिर अक्षर * b_qs = b_ptr;
            स्थिर अक्षर * b_qh = b_ptr + offset_qh;
            पूर्णांक mask = 0;
            क्रम (पूर्णांक k_group = 0; k_group < QK_K / 16; ++k_group) {
                पूर्णांक r = k_group >> 2;
                __m512i va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(mask++), va[r]);
                __m512i va1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(mask++), va[r]);

                __m512i vsum = _mm512_setzero_si512();
                __m512i hmask = _mm512_set1_epi8(0x3);

                __m512i bytes = _mm512_loadu_si512(b_qs);
                __m512i hbits = _mm512_loadu_si512(b_qh);
                __m512i vb0 = _mm512_and_si512(bytes, निचला_Mask);
                __m512i vb1 = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), निचला_Mask);
                __m512i vh0 = _mm512_slli_epi16(_mm512_and_si512(hbits, hmask), 4);
                __m512i vh1 = _mm512_slli_epi16(_mm512_and_si512(hbits, _mm512_slli_epi16(hmask, 2)), 2);

                vb0 = _mm512_add_epi8(vb0, vh0);
                vb1 = _mm512_add_epi8(vb1, vh1);
                vsum = _mm512_dpbusd_epi32(vsum, vb0, va0);
                vsum = _mm512_dpbusd_epi32(vsum, vb1, va1);
                b_qs += 64;

                va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(mask++), va[r]);
                va1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(mask++), va[r]);

                bytes = _mm512_loadu_si512(b_qs);
                vb0 = _mm512_and_si512(bytes, निचला_Mask);
                vb1 = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), निचला_Mask);
                vh0 =                   _mm512_and_si512(hbits, _mm512_slli_epi16(hmask, 4));
                vh1 = _mm512_srli_epi16(_mm512_and_si512(hbits, _mm512_slli_epi16(hmask, 6)), 2);
                vb0 = _mm512_add_epi8(vb0, vh0);
                vb1 = _mm512_add_epi8(vb1, vh1);
                vsum = _mm512_dpbusd_epi32(vsum, vb0, va0);
                vsum = _mm512_dpbusd_epi32(vsum, vb1, va1);
                b_qs += 64;
                b_qh += 64;

                // B * A - 32 * A
                __m512i vmask = _mm512_set1_epi32(k_group);
                vsum = _mm512_sub_epi32(vsum, _mm512_permutexvar_epi32(vmask, vcomp));

                // vacc += scale * (q8 @ q6)
                स्थिर __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((स्थिर __m128i *)(b_ptr + offset_scales + k_group * TILE_N)));
                acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(vsum, vscale));
            }
            स्थिर __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((स्थिर __m256i *)(b_ptr + offset_d0)));
            vc[col] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(vd0, vd1), vc[col]);
        };

        क्रम (पूर्णांक i = 0; i < KB; ++i) {
            Unroll<COLS>{}(compute, i);
        }

        //store to C
        स्वतः storec = [&](पूर्णांक col) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + col * 16), vc[col]);
        };
        Unroll<COLS>{}(storec);
    }
};

ढाँचा <पूर्णांक BLOCK_M, पूर्णांक BLOCK_N, पूर्णांक BLOCK_K>
काष्ठा tinygemm_kernel_vnni<block_q8_K, block_iq4_xs, भग्न, BLOCK_M, BLOCK_N, BLOCK_K> {
    अटल व्योम apply(पूर्णांक KB, स्थिर व्योम * RESTRICT _A, स्थिर व्योम * RESTRICT _B, भग्न * RESTRICT C, पूर्णांक ldc) {

        constexpr पूर्णांक COLS = BLOCK_N / 16;
        स्थिर पूर्णांक TILE_आकार = TILE_N * माप(block_iq4_xs) + TILE_N * 2;

        स्थिर block_q8_K * RESTRICT A = static_cast<स्थिर block_q8_K *>(_A);
        स्थिर अक्षर * RESTRICT B = static_cast<स्थिर अक्षर *>(_B);

        // load the 256 bytes from A to 4 avx512 vectors
        __m512i va[4];
        __m512 vc[COLS];
        __m512 vd1;

        // packed_B:
        स्थिर पूर्णांक offset_scales = (QK_K / 2) * TILE_N ;
        स्थिर पूर्णांक offset_d0     = (QK_K / 2) * TILE_N + 8 * TILE_N;

        // compensation
        __m512i vcomp;

        स्थिर __m256i m128s = _mm256_set1_epi16(128);
        स्थिर __m512i निचला_Mask = _mm512_set1_epi8(0xF);

        स्थिर __m512i values128 = _mm512_set_epi8(
            113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
            113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
            113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
            113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127
        );
        स्थिर __m512i off = _mm512_set1_epi8(static_cast<अक्षर>(0x80));
        स्थिर __m512i values256 = _mm512_add_epi8(values128, off);

        स्वतः loadc = [&](स्वतः col) {
            vc[col] = _mm512_setzero_ps();
        };
        Unroll<COLS>{}(loadc);

        स्वतः compute = [&](स्वतः col, स्वतः i) {
            अगर constexpr (col == 0) {
                // load a
                va[0] = _mm512_loadu_si512((स्थिर __m512i *)(A[0 * KB + i].qs +   0));
                va[1] = _mm512_loadu_si512((स्थिर __m512i *)(A[0 * KB + i].qs +  64));
                va[2] = _mm512_loadu_si512((स्थिर __m512i *)(A[0 * KB + i].qs + 128));
                va[3] = _mm512_loadu_si512((स्थिर __m512i *)(A[0 * KB + i].qs + 192));

                // compensation: 128 * A
                स्थिर __m256i q8sums = _mm256_loadu_si256((स्थिर __m256i *)A[0 * KB + i].bsums);
                vcomp = _mm512_castsi256_si512(_mm256_madd_epi16(q8sums, m128s));
                vd1 = _mm512_set1_ps(A[0 * KB + i].d);
            }

            // accmulate the quants
            __m512i acc = _mm512_setzero_si512();
            स्थिर अक्षर * b_ptr = B + PACKED_सूचकांक(col, i, KB, TILE_आकार);
            स्थिर अक्षर * b_qs = b_ptr;
            पूर्णांक mask = 0;
            क्रम (पूर्णांक k_group = 0; k_group < QK_K / 32; ++k_group) {
                पूर्णांक r = k_group >> 1;
                __m512i vmask = _mm512_set1_epi32(k_group);
                __m512i vsum = _mm512_setzero_si512();
                क्रम (पूर्णांक k = 0; k < 8; k += 2) {
                    __m512i va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(mask++), va[r]);
                    __m512i va1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(mask++), va[r]);

                    __m512i bytes = _mm512_loadu_si512(b_qs);
                    __m512i vb0 = _mm512_shuffle_epi8(values256, _mm512_and_si512(bytes, निचला_Mask));
                    __m512i vb1 = _mm512_shuffle_epi8(values256, _mm512_and_si512(_mm512_srli_epi16(bytes, 4), निचला_Mask));

                    vsum = _mm512_dpbusd_epi32(vsum, vb0, va0);
                    vsum = _mm512_dpbusd_epi32(vsum, vb1, va1);
                    b_qs += 64;
                }
                // (B + 128) * A - 128 * A
                vsum = _mm512_sub_epi32(vsum, _mm512_permutexvar_epi32(vmask, vcomp));

                // vacc += scale * (q8 @ q4)
                स्थिर __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((स्थिर __m128i *)(b_ptr + offset_scales + k_group * TILE_N)));
                acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(vsum, vscale));
            }
            स्थिर __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((स्थिर __m256i *)(b_ptr + offset_d0)));
            vc[col] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(vd0, vd1), vc[col]);
        };

        क्रम (पूर्णांक i = 0; i < KB; ++i) {
            Unroll<COLS>{}(compute, i);
        }

        //store to C
        स्वतः storec = [&](स्वतः col) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + col * 16), vc[col]);
        };
        Unroll<COLS>{}(storec);
    }
};

#घोषणा LAUNCH_TINYGEMM_KERNEL_VNNI(NB_आकार)                                                   \
    tinygemm_kernel_vnni<vec_dot_type, type, भग्न, 1, NB_आकार, blck_आकार>::apply(             \
        KB, wdata_batch,                                                                       \
        (स्थिर अक्षर *)src0->तथ्य + src0_offset + PACKED_सूचकांक(nb * kTilesN, 0, KB, TILE_आकार), \
        (भग्न *) dst->तथ्य + dst_offset + nb_start, ldc)

ढाँचा <typename TA, typename TB, typename TC, पूर्णांक BLOCK_K,
          typename std::enable_if<!is_type_qkk<TB>::value, पूर्णांक>::type = 0>
व्योम tinygemm_kernel_amx(पूर्णांक M, पूर्णांक N, पूर्णांक KB, स्थिर व्योम * RESTRICT _A, स्थिर व्योम * RESTRICT _B, TC * RESTRICT C, पूर्णांक ldc) {
    using packed_B_t = packed_B_type<TB>;
    स्थिर पूर्णांक TILE_आकार = get_tile_आकार<TB>();
    स्थिर bool need_unpack = do_unpack<TB>::value;

    GGML_ASSERT(M <= 2 * TILE_M && N == 2 * TILE_N);
    स्थिर TA * RESTRICT A = static_cast<स्थिर TA *>(_A);
    स्थिर अक्षर * RESTRICT B = static_cast<स्थिर अक्षर *>(_B);

    स्थिर पूर्णांक m0 = std::min(M, TILE_M);
    स्थिर पूर्णांक m1 = std::max(M - TILE_M, 0);
    स्थिर पूर्णांक lda = KB * माप(TA);
    //const int ldb = KB * sizeof(TB);

    alignas(64) अटल thread_local packed_B_t Tile0[TILE_N * TILE_K];
    alignas(64) अटल thread_local packed_B_t Tile1[TILE_N * TILE_K];
    alignas(64) अटल thread_local int8_t Tile23[TILE_M * TILE_K];

    alignas(64) अटल thread_local int32_t TileC0[TILE_M * TILE_N * 4];
    alignas(64) अटल thread_local int32_t TileC1[TILE_M * TILE_N * 4];

    // double buffering C to interleave avx512 and amx
    int32_t * C_cur = TileC0;
    int32_t * C_pre = TileC1;

    स्वतः Tile4 = [&](int32_t * base) { वापस base; };
    स्वतः Tile5 = [&](int32_t * base) { वापस base + TILE_M * TILE_N; };
    स्वतः Tile6 = [&](int32_t * base) { वापस base + 2 * TILE_M * TILE_N; };
    स्वतः Tile7 = [&](int32_t * base) { वापस base + 3 * TILE_M * TILE_N; };

    अगर (M == 2 * TILE_M) {
        // i = 0
        स्थिर अक्षर * B_blk0 = B + PACKED_सूचकांक(0, 0, KB, TILE_आकार);
        स्थिर अक्षर * B_blk1 = B + PACKED_सूचकांक(1, 0, KB, TILE_आकार);
        अगर (need_unpack) {
            unpack_B<TB>(Tile0, B_blk0);
            _tile_loadd(TMM0, Tile0, TILE_N * VNNI_BLK);
        } अन्यथा {
            _tile_loadd(TMM0, B_blk0, TILE_N * VNNI_BLK);
        }

        _tile_zero(TMM4);
        _tile_loadd(TMM2, A[0].qs, lda);
        _tile_dpbssd(TMM4, TMM2, TMM0);
        _tile_stored(TMM4, Tile4(C_pre), TILE_N * माप(int32_t));

        _tile_zero(TMM5);
        _tile_loadd(TMM3, A[TILE_M * KB + 0].qs, lda);
        _tile_dpbssd(TMM5, TMM3, TMM0);
        _tile_stored(TMM5, Tile5(C_pre), TILE_N * माप(int32_t));

        अगर (need_unpack) {
            unpack_B<TB>(Tile1, B_blk1);
            _tile_loadd(TMM1, Tile1, TILE_N * VNNI_BLK);
        } अन्यथा {
            _tile_loadd(TMM1, B_blk1, TILE_N * VNNI_BLK);
        }

        _tile_zero(TMM6);
        _tile_dpbssd(TMM6, TMM2, TMM1);
        _tile_stored(TMM6, Tile6(C_pre), TILE_N * माप(int32_t));

        _tile_zero(TMM7);
        _tile_dpbssd(TMM7, TMM3, TMM1);
        _tile_stored(TMM7, Tile7(C_pre), TILE_N * माप(int32_t));

        क्रम (पूर्णांक i = 1; i < KB; ++i) {
            // index of previous iter
            स्थिर पूर्णांक ii = i - 1;
            स्थिर अक्षर * B_blk0 = B + PACKED_सूचकांक(0, i, KB, TILE_आकार);
            स्थिर अक्षर * B_blk1 = B + PACKED_सूचकांक(1, i, KB, TILE_आकार);
            GGML_DISPATCH_BOOL(ii > 0, is_acc, [&] {
                अगर (need_unpack) {
                    unpack_B<TB>(Tile0, B_blk0);
                    _tile_loadd(TMM0, Tile0, TILE_N * VNNI_BLK);
                } अन्यथा {
                    _tile_loadd(TMM0, B_blk0, TILE_N * VNNI_BLK);
                }
                _tile_zero(TMM4);
                _tile_loadd(TMM2, A[i].qs, lda);
                acc_C<TA, TB, is_acc>::apply(C, ldc, Tile4(C_pre), &A[ii], KB, B + PACKED_सूचकांक(0, ii, KB, TILE_आकार), TILE_M);

                _tile_dpbssd(TMM4, TMM2, TMM0);
                _tile_stored(TMM4, Tile4(C_cur), TILE_N * माप(int32_t));

                _tile_zero(TMM5);
                _tile_loadd(TMM3, A[TILE_M * KB + i].qs, lda);
                acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc, ldc, Tile5(C_pre), &A[TILE_M * KB + ii], KB, B + PACKED_सूचकांक(0, ii, KB, TILE_आकार), TILE_M);

                _tile_dpbssd(TMM5, TMM3, TMM0);
                _tile_stored(TMM5, Tile5(C_cur), TILE_N * माप(int32_t));

                अगर (need_unpack) {
                    unpack_B<TB>(Tile1, B_blk1);
                    _tile_loadd(TMM1, Tile1, TILE_N * VNNI_BLK);
                } अन्यथा {
                    _tile_loadd(TMM1, B_blk1, TILE_N * VNNI_BLK);
                }
                _tile_zero(TMM6);
                acc_C<TA, TB, is_acc>::apply(C + TILE_N, ldc, Tile6(C_pre), &A[ii], KB, B + PACKED_सूचकांक(1, ii, KB, TILE_आकार), TILE_M);

                _tile_dpbssd(TMM6, TMM2, TMM1);
                _tile_stored(TMM6, Tile6(C_cur), TILE_N * माप(int32_t));

                _tile_zero(TMM7);
                acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc + TILE_N, ldc, Tile7(C_pre), &A[TILE_M * KB + ii], KB, B + PACKED_सूचकांक(1, ii, KB, TILE_आकार), TILE_M);

                _tile_dpbssd(TMM7, TMM3, TMM1);
                _tile_stored(TMM7, Tile7(C_cur), TILE_N * माप(int32_t));

                std::swap(C_cur, C_pre);
            });
        }
        // final accumulation
        {
            पूर्णांक ii = KB - 1;
            acc_C<TA, TB, true>::apply(C, ldc, Tile4(C_pre), &A[ii], KB, B + PACKED_सूचकांक(0, ii, KB, TILE_आकार), TILE_M);
            acc_C<TA, TB, true>::apply(C + TILE_M * ldc, ldc, Tile5(C_pre), &A[TILE_M * KB + ii], KB, B + PACKED_सूचकांक(0, ii, KB, TILE_आकार), TILE_M);
            acc_C<TA, TB, true>::apply(C + TILE_N, ldc, Tile6(C_pre), &A[ii], KB, B + PACKED_सूचकांक(1, ii, KB, TILE_आकार), TILE_M);
            acc_C<TA, TB, true>::apply(C + TILE_M * ldc + TILE_N, ldc, Tile7(C_pre), &A[TILE_M * KB + ii], KB, B + PACKED_सूचकांक(1, ii, KB, TILE_आकार), TILE_M);
        }
    } अन्यथा {
        क्रम (पूर्णांक i = 0; i < KB; ++i) {
            _tile_zero(TMM4);
            _tile_zero(TMM6);
            अगर (m1 != 0) {
                _tile_zero(TMM5);
                _tile_zero(TMM7);
            }

            स्थिर अक्षर * B_blk0 = B + PACKED_सूचकांक(0, i, KB, TILE_आकार);
            स्थिर अक्षर * B_blk1 = B + PACKED_सूचकांक(1, i, KB, TILE_आकार);
            अगर (need_unpack) {
                unpack_B<TB>(Tile0, B_blk0);
                _tile_loadd(TMM0, Tile0, TILE_N * VNNI_BLK);
            } अन्यथा {
                _tile_loadd(TMM0, B_blk0, TILE_N * VNNI_BLK);
            }

            अगर (need_unpack) {
                unpack_B<TB>(Tile1, B_blk1);
                _tile_loadd(TMM1, Tile1, TILE_N * VNNI_BLK);
            } अन्यथा {
                _tile_loadd(TMM1, B_blk1, TILE_N * VNNI_BLK);
            }

            अगर (m0 == TILE_M) {
                _tile_loadd(TMM2, A[i].qs, lda);
            } अन्यथा {
                unpack_A(Tile23, &A[i], KB, m0);
                _tile_loadd(TMM2, Tile23, TILE_K);
            }

            _tile_dpbssd(TMM4, TMM2, TMM0);
            _tile_dpbssd(TMM6, TMM2, TMM1);

            _tile_stored(TMM4, Tile4(C_cur), TILE_N * माप(int32_t));
            _tile_stored(TMM6, Tile6(C_cur), TILE_N * माप(int32_t));

            GGML_DISPATCH_BOOL(i > 0, is_acc, [&] {
                acc_C<TA, TB, is_acc>::apply(C,          ldc, Tile4(C_cur), &A[i], KB, B + PACKED_सूचकांक(0, i, KB, TILE_आकार), m0);
                acc_C<TA, TB, is_acc>::apply(C + TILE_N, ldc, Tile6(C_cur), &A[i], KB, B + PACKED_सूचकांक(1, i, KB, TILE_आकार), m0);
            });

            अगर (m1 != 0) {
                unpack_A(Tile23, &A[TILE_M * KB + i], KB, m1);
                _tile_loadd(TMM3, Tile23, TILE_K);

                _tile_dpbssd(TMM5, TMM3, TMM0);
                _tile_dpbssd(TMM7, TMM3, TMM1);
                _tile_stored(TMM5, Tile5(C_cur), TILE_N * माप(int32_t));
                _tile_stored(TMM7, Tile7(C_cur), TILE_N * माप(int32_t));
                GGML_DISPATCH_BOOL(i > 0, is_acc, [&] {
                    acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc,          ldc, Tile5(C_cur), &A[TILE_M * KB + i], KB, B + PACKED_सूचकांक(0, i, KB, TILE_आकार), m1);
                    acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc + TILE_N, ldc, Tile7(C_cur), &A[TILE_M * KB + i], KB, B + PACKED_सूचकांक(1, i, KB, TILE_आकार), m1);
                });
            }
        }
    }
    वापस;
}

ढाँचा <typename TA, typename TB, typename TC, पूर्णांक BLOCK_K,
          typename std::enable_if<is_type_qkk<TB>::value, पूर्णांक>::type = 0>
व्योम tinygemm_kernel_amx(पूर्णांक M, पूर्णांक N, पूर्णांक KB, स्थिर व्योम * RESTRICT _A, स्थिर व्योम * RESTRICT _B, भग्न * RESTRICT C, पूर्णांक ldc) {
    static_assert(std::is_same<TA, block_q8_K>::value);
    स्थिर पूर्णांक TILE_आकार = get_tile_आकार<TB>();

    GGML_ASSERT(M <= 2 * TILE_M && N == 2 * TILE_N);
    स्थिर TA * RESTRICT A = static_cast<स्थिर TA *>(_A);
    स्थिर अक्षर * RESTRICT B = static_cast<स्थिर अक्षर *>(_B);

    स्थिर पूर्णांक m0 = std::min(M, TILE_M);
    स्थिर पूर्णांक m1 = std::max(M - TILE_M, 0);
    //const int lda = KB * sizeof(TA);

    alignas(64) अटल thread_local int8_t Tile0[TILE_N * TILE_K];
    alignas(64) अटल thread_local int8_t Tile1[TILE_N * TILE_K];
    alignas(64) अटल thread_local int8_t Tile23[TILE_M * TILE_K];

    // mat mul result for each group
    alignas(64) अटल thread_local int32_t Tile4[TILE_M * TILE_N];
    alignas(64) अटल thread_local int32_t Tile5[TILE_M * TILE_N];
    alignas(64) अटल thread_local int32_t Tile6[TILE_M * TILE_N];
    alignas(64) अटल thread_local int32_t Tile7[TILE_M * TILE_N];

    // sum of each QK_K block, contains 8 groups, int32
    alignas(64) अटल thread_local int32_t Sumi4[TILE_M * TILE_N];
    alignas(64) अटल thread_local int32_t Sumi5[TILE_M * TILE_N];
    alignas(64) अटल thread_local int32_t Sumi6[TILE_M * TILE_N];
    alignas(64) अटल thread_local int32_t Sumi7[TILE_M * TILE_N];

    स्थिर पूर्णांक k_group_आकार = std::is_same<TB, block_q6_K>::value ? 16 : 32;
    क्रम (पूर्णांक i = 0; i < KB; ++i) {
        // step 1: accumulate the quants across 8 groups, each group with 32
        क्रम (पूर्णांक k = 0; k < QK_K / k_group_आकार; ++k) {
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

                _tile_stored(TMM4, Tile4, TILE_N * माप(int32_t));
                _tile_stored(TMM6, Tile6, TILE_N * माप(int32_t));

                scale_C<TB, is_acc>(Tile4, Sumi4, B + PACKED_सूचकांक(0, i, KB, TILE_आकार), k, m0);
                scale_C<TB, is_acc>(Tile6, Sumi6, B + PACKED_सूचकांक(1, i, KB, TILE_आकार), k, m0);

                अगर (m1 != 0) {
                    _tile_zero(TMM5);
                    _tile_zero(TMM7);

                    unpack_A<TB>(Tile23, &A[TILE_M * KB + i], KB, k, m1);
                    _tile_loadd(TMM3, Tile23, TILE_K);

                    _tile_dpbssd(TMM5, TMM3, TMM0);
                    _tile_dpbssd(TMM7, TMM3, TMM1);

                    _tile_stored(TMM5, Tile5, TILE_N * माप(int32_t));
                    _tile_stored(TMM7, Tile7, TILE_N * माप(int32_t));

                    scale_C<TB, is_acc>(Tile5, Sumi5, B + PACKED_सूचकांक(0, i, KB, TILE_आकार), k, m1);
                    scale_C<TB, is_acc>(Tile7, Sumi7, B + PACKED_सूचकांक(1, i, KB, TILE_आकार), k, m1);
                }
            });
        }

        // step 2: accmulate the mins
        GGML_DISPATCH_BOOL(i > 0, is_acc, [&] {
            acc_C<TA, TB, is_acc>::apply(C,          ldc, Sumi4, &A[i], KB, B + PACKED_सूचकांक(0, i, KB, TILE_आकार), m0);
            acc_C<TA, TB, is_acc>::apply(C + TILE_N, ldc, Sumi6, &A[i], KB, B + PACKED_सूचकांक(1, i, KB, TILE_आकार), m0);
            अगर (m1 != 0) {
                acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc,          ldc, Sumi5, &A[TILE_M * KB + i], KB, B + PACKED_सूचकांक(0, i, KB, TILE_आकार), m1);
                acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc + TILE_N, ldc, Sumi7, &A[TILE_M * KB + i], KB, B + PACKED_सूचकांक(1, i, KB, TILE_आकार), m1);
            }
        });
    }
    वापस;
}

} // anonymous namespace

// get the packed tensor size for quantized weights
माप_प्रकार ggml_backend_amx_get_alloc_आकार(स्थिर काष्ठा ggml_tensor * tensor) {
    स्थिर क्रमागत ggml_type TYPE = tensor->type;

    स्थिर पूर्णांक K = tensor->ne[0]; // ne0: in_features
    स्थिर पूर्णांक N = tensor->ne[1]; // ne1: out_features

    स्वतः get_tensor_आकार = [&] {
        माप_प्रकार row_आकार_B{0};
        GGML_DISPATCH_QTYPES(TYPE, [&] {
            row_आकार_B = get_row_आकार<type, blck_आकार>(K);
        });
        वापस N * row_आकार_B;
    };

    अगर (qtype_has_amx_kernels(TYPE)) {
        वापस get_tensor_आकार();
    } अन्यथा {
        // for f16, bf16 we don't do packing
        वापस ggml_nbytes(tensor);
    }
}

// pack weight to vnni format
व्योम ggml_backend_amx_convert_weight(काष्ठा ggml_tensor * tensor, स्थिर व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार size) {
    GGML_ASSERT(offset == 0 && size == ggml_nbytes(tensor)); // only full tensor conversion is supported for now

    स्थिर क्रमागत ggml_type TYPE = tensor->type;

    स्थिर पूर्णांक K = tensor->ne[0]; // ne0: in_features
    स्थिर पूर्णांक N = tensor->ne[1]; // ne1: out_features

    GGML_DISPATCH_QTYPES(TYPE, [&] {
        convert_B_packed_format<type, blck_आकार>((व्योम *)((अक्षर *)tensor->तथ्य + offset), (स्थिर type *)तथ्य, N, K);
    });
}

// ne2 is passed explicitly to help compiler optimize repeated calls
अंतरभूत int64_t ggml_batch_offset(स्थिर ggml_tensor * t, int64_t batch_idx, int64_t ne2) {
    स्थिर int64_t i2 = batch_idx % ne2;
    स्थिर int64_t i3 = batch_idx / ne2;
    वापस i3 * t->nb[3] + i2 * t->nb[2];
}

माप_प्रकार ggml_backend_amx_desired_wsize(स्थिर काष्ठा ggml_tensor * dst) {
    काष्ठा ggml_tensor * src0 = dst->src[0];

    स्थिर क्रमागत ggml_type TYPE = src0->type;

    स्थिर bool is_floating_type = TYPE == GGML_TYPE_F16;
    अगर (is_floating_type) {
        वापस 0;
    }

    स्थिर पूर्णांक M = dst->ne[1];
    स्थिर पूर्णांक K = src0->ne[0];
    स्थिर int64_t n_batch = dst->ne[2] * dst->ne[3];

    माप_प्रकार desired_wsize = 0;

    GGML_DISPATCH_QTYPES(TYPE, [&] {
        स्थिर माप_प्रकार row_आकार_A = K / blck_आकार * माप(vec_dot_type);
        desired_wsize = n_batch * M * row_आकार_A;
    });

    वापस desired_wsize;
}

// NB: mixed dtype gemm with Advanced Matrix Extensions (Intel AMX)
//
// src0: weight in shape of {N, K}, quantized
// src1: input  in shape of {M, K}, float32
// dst:  output in shape of {M, N}, float32
//
// the function performs: dst = src1 @ src0.T for each batch
//
व्योम ggml_backend_amx_mul_mat(स्थिर ggml_compute_params * params, काष्ठा ggml_tensor * dst) {
    काष्ठा ggml_tensor * src0 = dst->src[0];
    काष्ठा ggml_tensor * src1 = dst->src[1];

    स्थिर क्रमागत ggml_type TYPE = src0->type;

    // f16 only has avx512 kernels for now,
    // amx kernels will be added once 6th gen xeon is released.
    स्थिर bool is_floating_type = TYPE == GGML_TYPE_F16;

    स्थिर पूर्णांक M = dst->ne[1];
    स्थिर पूर्णांक N = dst->ne[0];
    स्थिर पूर्णांक K = src0->ne[0];
    स्थिर पूर्णांक ldc = dst->nb[1] / dst->nb[0];

    स्थिर int64_t ne2 = dst->ne[2];
    स्थिर int64_t n_batch = ne2 * dst->ne[3];

    अगर (is_floating_type) {
        constexpr पूर्णांक BLOCK_M = 4;
        constexpr पूर्णांक BLOCK_N = 6;
        स्थिर पूर्णांक MB = div_up(M, BLOCK_M);
        स्थिर पूर्णांक NB = div_up(N, BLOCK_N);

        parallel_for_ggml(params, n_batch * MB * NB, [&](पूर्णांक begin, पूर्णांक end) {
            GGML_DISPATCH_FLOATING_TYPES(TYPE, [&] {
                क्रम (पूर्णांक i = begin; i < end; ++i) {
                    पूर्णांक batch_idx = i / (MB * NB);
                    पूर्णांक remaining = i % (MB * NB);
                    पूर्णांक mb = remaining / NB;
                    पूर्णांक nb = remaining % NB;

                    int64_t src0_offset = ggml_batch_offset(src0, batch_idx, ne2);
                    int64_t src1_offset = ggml_batch_offset(src1, batch_idx, ne2);
                    int64_t dst_offset  = ggml_batch_offset(dst,  batch_idx, ne2);

                    पूर्णांक mb_start = mb * BLOCK_M;
                    पूर्णांक mb_आकार = std::min(BLOCK_M, M - mb_start);
                    पूर्णांक nb_start = nb * BLOCK_N;
                    पूर्णांक nb_आकार = std::min(BLOCK_N, N - nb_start);

                    चयन (mb_आकार << 4 | nb_आकार) {
                        हाल 0x12: LAUNCH_TINYGEMM_KERNEL_AVX(1, 2); अवरोध;
                        हाल 0x14: LAUNCH_TINYGEMM_KERNEL_AVX(1, 4); अवरोध;
                        हाल 0x16: LAUNCH_TINYGEMM_KERNEL_AVX(1, 6); अवरोध;
                        हाल 0x22: LAUNCH_TINYGEMM_KERNEL_AVX(2, 2); अवरोध;
                        हाल 0x24: LAUNCH_TINYGEMM_KERNEL_AVX(2, 4); अवरोध;
                        हाल 0x26: LAUNCH_TINYGEMM_KERNEL_AVX(2, 6); अवरोध;
                        हाल 0x32: LAUNCH_TINYGEMM_KERNEL_AVX(3, 2); अवरोध;
                        हाल 0x34: LAUNCH_TINYGEMM_KERNEL_AVX(3, 4); अवरोध;
                        हाल 0x36: LAUNCH_TINYGEMM_KERNEL_AVX(3, 6); अवरोध;
                        हाल 0x42: LAUNCH_TINYGEMM_KERNEL_AVX(4, 2); अवरोध;
                        हाल 0x44: LAUNCH_TINYGEMM_KERNEL_AVX(4, 4); अवरोध;
                        हाल 0x46: LAUNCH_TINYGEMM_KERNEL_AVX(4, 6); अवरोध;
                        शेष: ख_लिखो(मानक_त्रुटि, "Unexpected block size!\n");
                    }
                }
            });
        });
        वापस;
    }

    // pointer to work space, used convert A from float to quantized type
    व्योम * wdata = params->wdata;

    //TODO: performance improvement: merge quant A
 // if (params->ith == 0) {
        GGML_DISPATCH_QTYPES(TYPE, [&] {
            स्थिर माप_प्रकार row_आकार_A = K / blck_आकार * माप(vec_dot_type);
            स्थिर माप_प्रकार desired_wsize = n_batch * M * row_आकार_A;
            अगर (params->wsize < desired_wsize) {
                GGML_ABORT("insufficient work space size");
            }

            // Q4_0, Q4_1, Q8_0 handles 1 TILE_K per blck_size
            // Q4_K, Q5_K, Q6_K, IQ4_XS handles 8 TILE_K per blck_size
            GGML_ASSERT(TILE_K == blck_आकार || TILE_K * 8 == blck_आकार);

            parallel_for_ggml(params, n_batch * M, [&](पूर्णांक begin, पूर्णांक end) {
                क्रम (पूर्णांक idx = begin; idx < end; ++idx) {
                    पूर्णांक batch_idx = idx / M;
                    पूर्णांक m         = idx % M;
                    int64_t src1_offset = ggml_batch_offset(src1, batch_idx, ne2);
                    स्थिर भग्न * A_तथ्य = (स्थिर भग्न *)((स्थिर अक्षर *)src1->तथ्य + src1_offset);
                    अक्षर * wdata_batch = (अक्षर *)wdata + batch_idx * M * row_आकार_A;
                    from_float<vec_dot_type>(A_तथ्य + m * K, wdata_batch + m * row_आकार_A, K);
                }
            });
        });
 // }

    ggml_barrier(params->threadpool);

    अगर (M == 1) {
        // MB = 1 and handle 8 tiles in each block
        constexpr पूर्णांक kTilesN = 4;
        constexpr पूर्णांक BLOCK_N = TILE_N * kTilesN;
        स्थिर पूर्णांक NB = div_up(N, BLOCK_N);

        parallel_for_ggml(params, n_batch * NB, [&](पूर्णांक begin, पूर्णांक end) {
            GGML_DISPATCH_QTYPES(TYPE, [&] {
                स्थिर पूर्णांक KB = K / blck_आकार;
                स्थिर पूर्णांक TILE_आकार = get_tile_आकार<type>();
                स्थिर पूर्णांक row_आकार_A = KB * माप(vec_dot_type);
                क्रम (पूर्णांक i = begin; i < end; ++i) {
                    पूर्णांक batch_idx = i / NB;
                    पूर्णांक nb = i % NB;

                    int64_t src0_offset = ggml_batch_offset(src0, batch_idx, ne2);
                    int64_t dst_offset  = ggml_batch_offset(dst,  batch_idx, ne2);
                    स्थिर अक्षर * wdata_batch = (स्थिर अक्षर *)wdata + batch_idx * row_आकार_A;

                    पूर्णांक nb_start = nb * BLOCK_N;
                    पूर्णांक nb_आकार = std::min(BLOCK_N, N - nb_start); // 32, 64, 96

                    चयन (nb_आकार) {
                        //case 160: LAUNCH_TINYGEMM_KERNEL_VNNI(160); break;
                        हाल 128: LAUNCH_TINYGEMM_KERNEL_VNNI(128); अवरोध;
                        हाल 96: LAUNCH_TINYGEMM_KERNEL_VNNI(96); अवरोध;
                        हाल 64: LAUNCH_TINYGEMM_KERNEL_VNNI(64); अवरोध;
                        हाल 32: LAUNCH_TINYGEMM_KERNEL_VNNI(32); अवरोध;
                        शेष: ख_लिखो(मानक_त्रुटि, "Unexpected n block size!\n");
                    }
                }
            });
        });
        वापस;
    }

    // handle 4 tiles at a tile
    constexpr पूर्णांक BLOCK_M = TILE_M * 2;
    constexpr पूर्णांक BLOCK_N = TILE_N * 2;
    स्थिर पूर्णांक MB = div_up(M, BLOCK_M);
    स्थिर पूर्णांक NB = div_up(N, BLOCK_N);

    parallel_for_ggml(params, n_batch * MB * NB, [&](पूर्णांक begin, पूर्णांक end) {
        // init tile config for each thread
        ggml_tile_config_init();

        GGML_DISPATCH_QTYPES(TYPE, [&] {
            स्थिर पूर्णांक KB = K / blck_आकार;
            स्थिर पूर्णांक TILE_आकार = get_tile_आकार<type>();
            स्थिर पूर्णांक row_आकार_A = KB * माप(vec_dot_type);

            क्रम (पूर्णांक i = begin; i < end; ++i) {
                पूर्णांक batch_idx = i / (MB * NB);
                पूर्णांक remaining = i % (MB * NB);
                पूर्णांक mb = remaining / NB;
                पूर्णांक nb = remaining % NB;

                int64_t src0_offset = ggml_batch_offset(src0, batch_idx, ne2);
                int64_t dst_offset  = ggml_batch_offset(dst,  batch_idx, ne2);
                स्थिर अक्षर * wdata_batch = (स्थिर अक्षर *)wdata + batch_idx * M * row_आकार_A;

                पूर्णांक mb_start = mb * BLOCK_M;
                पूर्णांक mb_आकार = std::min(BLOCK_M, M - mb_start);
                पूर्णांक nb_start = nb * BLOCK_N;
                पूर्णांक nb_आकार = BLOCK_N;

                tinygemm_kernel_amx<vec_dot_type, type, भग्न, blck_आकार>(
                    mb_आकार, nb_आकार, KB,
                    wdata_batch + mb_start * row_आकार_A,
                    (स्थिर अक्षर *)src0->तथ्य + src0_offset + PACKED_सूचकांक(nb * 2, 0, KB, TILE_आकार),
                    (भग्न *) dst->तथ्य + dst_offset + mb_start * N + nb_start, ldc);
            }
        });
    });
}

#पूर्ण_अगर // if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
