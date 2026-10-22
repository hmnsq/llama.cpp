#إذا defined(__GNUC__)
#هدايت GCC diagnostic ignored "-Wpedantic"
#هدايت GCC diagnostic ignored "-Wunused-local-typedefs"
#جيڪڏهن_پورو

#ضمن "amx.h"
#ضمن "mmq.h"
#ضمن "ggml-impl.h"
#ضمن "ggml-cpu-impl.h"
#ضمن "simd-mappings.h"
#ضمن "quants.h"
#ضمن "ggml-quants.h"
#ضمن <algorithm>
#ضمن <type_traits>

#إذا defined(__gnu_linux__)
#ضمن <sys/syscall.h>
#ضمن <unistd.h>
#جيڪڏهن_پورو

#إذا (defined(_WIN32) || defined(_WIN64))
#وصف_ڪريو RESTRICT __restrict
#وإلا
#وصف_ڪريو RESTRICT __restrict__
#جيڪڏهن_پورو

#إذا (defined(_WIN32) || defined(_WIN64))
#وصف_ڪريو ALWAYS_INLINE __forceinline
#وإلاإذا __has_attribute(always_inline) || defined(__GNUC__)
#وصف_ڪريو ALWAYS_INLINE __attribute__((__always_inline__)) مضمن
#وإلا
#وصف_ڪريو ALWAYS_INLINE مضمن
#جيڪڏهن_پورو

#إذا defined(__AMX_INT8__) && defined(__AVX512VNNI__)

نطاق {

// Forced unrolling
سانچو <صحيح n>
بنية Unroll {
    سانچو <قسم_نالو Func, قسم_نالو... دليلون>
    ALWAYS_INLINE فراغ عامل()(ثابت Func& f, دليلون... دليلون_2) ثابت {
        Unroll<n - 1>{}(f, دليلون_2...);
        f(std::integral_constant<صحيح, n - 1>{}, دليلون_2...);
    }
};

سانچو <>
بنية Unroll<1> {
    سانچو <قسم_نالو Func, قسم_نالو... دليلون>
    ALWAYS_INLINE فراغ عامل()(ثابت Func& f, دليلون... دليلون_2) ثابت {
        f(std::integral_constant<صحيح, 0>{}, دليلون_2...);
    }
};

// type traits
سانچو <قسم_نالو T> بنية PackedTypes {};
سانچو <> بنية PackedTypes<بلاڪ_q4_0> { استعمال type = int8_t; };
سانچو <> بنية PackedTypes<بلاڪ_q4_1> { استعمال type = uint8_t; };
سانچو <> بنية PackedTypes<بلاڪ_q8_0> { استعمال type = int8_t; };
سانچو <قسم_نالو T> استعمال packed_B_type = قسم_نالو PackedTypes<T>::type;

سانچو <قسم_نالو T>
بنية do_compensate : std::integral_constant<منطقي,
    std::is_same<T, بلاڪ_q8_0>::value> {};

سانچو <قسم_نالو T>
بنية do_unpack : std::integral_constant<منطقي,
    std::is_same<T, بلاڪ_q4_0>::value ||
    std::is_same<T, بلاڪ_q4_1>::value> {};

سانچو <قسم_نالو T>
بنية is_type_qkk : std::integral_constant<منطقي,
    std::is_same<T, بلاڪ_q4_K>::value ||
    std::is_same<T, بلاڪ_q5_K>::value ||
    std::is_same<T, بلاڪ_q6_K>::value ||
    std::is_same<T, بلاڪ_iq4_xs>::value> {};

#وصف_ڪريو GGML_DISPATCH_FLOATING_TYPES(TYPE, ...)                                        \
    [&] {                                                                              \
        بدل (TYPE) {                                                                \
            حالة GGML_TYPE_F16: {                                                      \
                استعمال type = ggml_fp16_t;                                              \
                constexpr صحيح blck_سائيز = 16;                                          \
                أرجع __VA_ARGS__();                                                  \
            }                                                                          \
            حالة GGML_TYPE_BF16: {                                                     \
                استعمال type = ggml_bf16_t;                                              \
                constexpr صحيح blck_سائيز = 32;                                          \
                أرجع __VA_ARGS__();                                                  \
            }                                                                          \
            افتراضي:                                                                   \
                fprintf(stderr, "Unsupported floating data type\n");                   \
        }                                                                              \
    }()

#وصف_ڪريو GGML_DISPATCH_QTYPES(QT, ...)                                                  \
    [&] {                                                                              \
        بدل (QT) {                                                                  \
            حالة GGML_TYPE_Q4_0: {                                                     \
                استعمال type = بلاڪ_q4_0;                                               \
                استعمال vec_dot_type = بلاڪ_q8_0;                                       \
                constexpr صحيح blck_سائيز = QK4_0;                                       \
                أرجع __VA_ARGS__();                                                  \
            }                                                                          \
            حالة GGML_TYPE_Q4_1: {                                                     \
                استعمال type = بلاڪ_q4_1;                                               \
                استعمال vec_dot_type = بلاڪ_q8_1;                                       \
                constexpr صحيح blck_سائيز = QK4_1;                                       \
                أرجع __VA_ARGS__();                                                  \
            }                                                                          \
            حالة GGML_TYPE_Q8_0: {                                                     \
                استعمال type = بلاڪ_q8_0;                                               \
                استعمال vec_dot_type = بلاڪ_q8_0;                                       \
                constexpr صحيح blck_سائيز = QK8_0;                                       \
                أرجع __VA_ARGS__();                                                  \
            }                                                                          \
            حالة GGML_TYPE_Q4_K: {                                                     \
                استعمال type = بلاڪ_q4_K;                                               \
                استعمال vec_dot_type = بلاڪ_q8_K;                                       \
                constexpr صحيح blck_سائيز = QK_K;                                        \
                أرجع __VA_ARGS__();                                                  \
            }                                                                          \
            حالة GGML_TYPE_Q5_K: {                                                     \
                استعمال type = بلاڪ_q5_K;                                               \
                استعمال vec_dot_type = بلاڪ_q8_K;                                       \
                constexpr صحيح blck_سائيز = QK_K;                                        \
                أرجع __VA_ARGS__();                                                  \
            }                                                                          \
            حالة GGML_TYPE_Q6_K: {                                                     \
                استعمال type = بلاڪ_q6_K;                                               \
                استعمال vec_dot_type = بلاڪ_q8_K;                                       \
                constexpr صحيح blck_سائيز = QK_K;                                        \
                أرجع __VA_ARGS__();                                                  \
            }                                                                          \
            حالة GGML_TYPE_IQ4_XS: {                                                   \
                استعمال type = بلاڪ_iq4_xs;                                             \
                استعمال vec_dot_type = بلاڪ_q8_K;                                       \
                constexpr صحيح blck_سائيز = QK_K;                                        \
                أرجع __VA_ARGS__();                                                  \
            }                                                                          \
            افتراضي:                                                                   \
                fprintf(stderr, "Unsupported quantized data type: %d\n", صحيح(TYPE));   \
        }                                                                              \
    }()

#وصف_ڪريو GGML_DISPATCH_BOOL(BOOL_V, BOOL_NAME, ...)                                     \
    [&] {                                                                              \
        إذا (BOOL_V) {                                                                  \
            constexpr منطقي BOOL_NAME = سچ;                                           \
            أرجع __VA_ARGS__();                                                      \
        } وإلا {                                                                       \
            constexpr منطقي BOOL_NAME = خطأ;                                          \
            أرجع __VA_ARGS__();                                                      \
        }                                                                              \
    }()

// define amx tile config data structure
بنية tile_config_t{
    uint8_t palette_id = 0;
    uint8_t شروعات_صف = 0;
    uint8_t reserved_0[14] = {0};
    uint16_t colsb[16] = {0};
    uint8_t صفون[16] = {0};
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

مضمن فراغ ggml_tile_config_init(فراغ) {
    ساكن thread_local منطقي پورو = خطأ;

    إذا (پورو) {
        أرجع;
    }

    alignas(64) tile_config_t tc = {};
    tc.palette_id = 1;
    tc.شروعات_صف = 0;
    tc.صفون[0] = 8;   tc.colsb[0] = 64;
    tc.صفون[1] = 8;   tc.colsb[1] = 64;
    tc.صفون[2] = 16;  tc.colsb[2] = 32;
    tc.صفون[3] = 16;  tc.colsb[3] = 32;
    tc.صفون[4] = 16;  tc.colsb[4] = 64;
    tc.صفون[5] = 16;  tc.colsb[5] = 64;
    tc.صفون[6] = 16;  tc.colsb[6] = 64;
    tc.صفون[7] = 16;  tc.colsb[7] = 64;

    _tile_loadconfig(&tc);
    پورو = سچ;
}

// we need an extra 16 * 4B (TILE_N * int32_t) for each NB/KB block for compensation.
// See the notes `s8s8 igemm compensation in avx512-vnni` for detail.
سانچو <قسم_نالو TB>
صحيح get_tile_سائيز() {
    صحيح tile_سائيز = TILE_N * حجم(TB);
    إذا (do_compensate<TB>::value) {
        tile_سائيز += TILE_N * حجم(int32_t);
    }
    إذا (std::is_same<TB, بلاڪ_q4_K>::value ||
        std::is_same<TB, بلاڪ_q5_K>::value) {
        tile_سائيز += TILE_N * 4;
    }
    إذا (std::is_same<TB, بلاڪ_iq4_xs>::value) {
        tile_سائيز += TILE_N * 2;
    }
    أرجع tile_سائيز;
}

سانچو <قسم_نالو TB, صحيح بلاڪ_K>
صحيح get_صف_سائيز(صحيح K) {
    صحيح KB = K / بلاڪ_K;
    صحيح صف_سائيز = KB * حجم(TB);
    إذا (do_compensate<TB>::value) {
        صف_سائيز += KB * حجم(int32_t);
    }
    إذا (std::is_same<TB, بلاڪ_q4_K>::value ||
        std::is_same<TB, بلاڪ_q5_K>::value) {
        صف_سائيز += KB * 4;
    }
    إذا (std::is_same<TB, بلاڪ_iq4_xs>::value) {
        صف_سائيز += KB * 2;
    }
    أرجع صف_سائيز;
}

// transpose utils
#وصف_ڪريو SHUFFLE_EPI32(a, b, نقاب) \
    _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b), نقاب))
مضمن فراغ transpose_8x8_32bit(__m256i * v, __m256i * v1) {
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

مضمن فراغ transpose_16x4_32bit(__m512i * r, __m512i * d) {

    ساكن ثابت __m512i index1 = mm512_سيٽ_epi32(
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

مضمن فراغ transpose_16x16_32bit(__m512i * v) {
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

فراغ quantize_صف_q8_K_vnni(ثابت عائم * RESTRICT x, فراغ * RESTRICT vy, int64_t k) {
    أكد(k % QK_K == 0);
    ثابت صحيح KB = k / QK_K;
    constexpr صحيح kVecs = QK_K / 16;

    بلاڪ_q8_K * y = reinterpret_cast<بلاڪ_q8_K *>(vy);

    // hold 16 float vecs from x
    __m512  v[kVecs];

    // hold the quants vecs
    __m512i vq[kVecs / 4];

    // hold the packed quants vecs
    __m512i vq_packed[kVecs / 4];

    ثابت __m512 sign_ٻٽو_انگ = _mm512_set1_ps(-0.f);

    لكل (صحيح i = 0; i < KB; ++i) {
        // Compute max(abs(e)) for the block
        __m512 vamax = _mm512_set1_ps(0.f);
        لكل (صحيح j = 0; j < kVecs; ++j) {
            v[j] = _mm512_loadu_ps(x); x += 16;
            vamax = _mm512_max_ps(vamax, _mm512_andnot_ps(sign_ٻٽو_انگ, v[j]));
        }
        ثابت عائم amax = _mm512_reduce_max_ps(vamax);

        // Quantize these floats
        ثابت عائم iscale = 127.f / amax;
        y[i].d = GGML_CPU_FP32_منزل_FP16(1 / iscale);
        ثابت عائم id = ( amax != 0.0f ) ? iscale : 0.f;
        ثابت __m512 vscale = _mm512_set1_ps(id);

        // Apply multiplier and round to nearest integer
        لكل (صحيح j = 0; j < kVecs; ++j) {
            v[j] = _mm512_mul_ps(v[j], vscale);
            v[j] = _mm512_roundscale_ps(v[j], (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        }

        // Pack to epi8 vecs
        لكل (صحيح j = 0; j < kVecs / 4; ++j) {
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

        ثابت __m512i one = _mm512_set1_epi8(1);
        __m512i جوڙ = _mm512_setzero_si512();
        لكل (صحيح k = 0; k < 4; ++k) {
            جوڙ = _mm512_dpbusd_epi32(جوڙ, one, vq_packed[k]);
        }
        _mm256_storeu_si256((__m256i *)(y[i].bsums), _mm512_cvtepi32_epi16(جوڙ));
    }
}

// quantize A from float to `vec_dot_type`
سانچو <قسم_نالو T>
مضمن فراغ from_float(ثابت عائم * x, حرف * vy, int64_t k);

سانچو <>
مضمن فراغ from_float<بلاڪ_q8_0>(ثابت عائم * x, حرف * vy, int64_t k) {
    quantize_صف_q8_0(x, (بلاڪ_q8_0 *)vy, k);
}

سانچو <>
مضمن فراغ from_float<بلاڪ_q8_1>(ثابت عائم * x, حرف * vy, int64_t k) {
    quantize_صف_q8_1(x, (بلاڪ_q8_1 *)vy, k);
}

سانچو <>
مضمن فراغ from_float<بلاڪ_q8_K>(ثابت عائم * x, حرف * vy, int64_t k) {
#إذا 1
    // TODO: this is reference impl!
    quantize_صف_q8_K_ref(x, (بلاڪ_q8_K *)vy, k);
#وإلا
    quantize_صف_q8_K_vnni(x, vy, k);
#جيڪڏهن_پورو
}

// load A from memory to array when nrows can not fill in whole tile
فراغ unpack_A(int8_t * RESTRICT tile, ثابت بلاڪ_q8_0 * RESTRICT A, صحيح lda, صحيح nr) {
    أكد(nr != TILE_M);
    لكل (صحيح m = 0; m < nr; ++m) {
        ثابت __m256i v = _mm256_loadu_si256((ثابت __m256i *)(A[m * lda].qs));
        _mm256_storeu_si256((__m256i *)(tile + m * TILE_K), v);
    }
}

فراغ unpack_A(int8_t * RESTRICT tile, ثابت بلاڪ_q8_1 * RESTRICT A, صحيح lda, صحيح nr) {
    أكد(nr != TILE_M);
    لكل (صحيح m = 0; m < nr; ++m) {
        ثابت __m256i v = _mm256_loadu_si256((ثابت __m256i *)(A[m * lda].qs));
        _mm256_storeu_si256((__m256i *)(tile + m * TILE_K), v);
    }
}

سانچو <قسم_نالو TB>
فراغ unpack_A(int8_t * RESTRICT tile, ثابت بلاڪ_q8_K * RESTRICT A, صحيح lda, صحيح k, صحيح nr) {
    أكد(nr <= TILE_M);
    لكل (صحيح m = 0; m < nr; ++m) {
        ثابت __m256i v = _mm256_loadu_si256((ثابت __m256i *)(A[m * lda].qs + k * 32));
        _mm256_storeu_si256((__m256i *)(tile + m * TILE_K), v);
    }
}

سانچو <>
فراغ unpack_A<بلاڪ_q6_K>(int8_t * RESTRICT tile, ثابت بلاڪ_q8_K * RESTRICT A, صحيح lda, صحيح k, صحيح nr) {
    أكد(nr <= TILE_M);
    // zero padding k from 16 to 32, so that we don't have to re-config amx
    ثابت __m128i zero = _mm_setzero_si128();
    لكل (صحيح m = 0; m < nr; ++m) {
        ثابت __m128i v = _mm_loadu_si128((ثابت __m128i *)(A[m * lda].qs + k * 16));
        ثابت __m256i r = _mm256_insertf128_si256(_mm256_castsi128_si256(v), zero, 1);
        _mm256_storeu_si256((__m256i *)(tile + m * TILE_K), r);
    }
}

#وصف_ڪريو MM256_سيٽ_M128_I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)
مضمن __m256i بائيٽون_from_nibbles_32(ثابت uint8_t * rsi) {
    ثابت __m128i tmp = _mm_loadu_si128((ثابت __m128i *)rsi);
    ثابت __m256i بائيٽون = MM256_سيٽ_M128_I(_mm_srli_epi16(tmp, 4), tmp);
    ثابت __m256i هيٺيون_نقاب = _mm256_set1_epi8(0xF);
    أرجع _mm256_and_si256(هيٺيون_نقاب, بائيٽون);
}

// used for block_q4_K
مضمن __m512i بائيٽون_from_nibbles_64(ثابت uint8_t * rsi) {
    ثابت __m256i tmp = _mm256_loadu_si256((ثابت __m256i *)rsi);
    ثابت __m256i هيٺيون_نقاب = _mm256_set1_epi8(0xF);
    ثابت __m256i q4l = _mm256_and_si256(tmp, هيٺيون_نقاب);
    ثابت __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(tmp, 4), هيٺيون_نقاب);
    أرجع _mm512_inserti32x8(_mm512_castsi256_si512(q4l), q4h, 1);
}

// used for block_q5_K
مضمن __m512i بائيٽون_from_nibbles_64(ثابت uint8_t * qs, ثابت uint8_t * qh, صحيح k) {
    ثابت __m256i هيٺيون_نقاب = _mm256_set1_epi8(0xF);
    __m256i hmask = _mm256_set1_epi8(1);
    hmask = _mm256_slli_epi16(hmask, k);

    ثابت __m256i q5bits = _mm256_loadu_si256((ثابت __m256i *)qs);
    ثابت __m256i hbits = _mm256_loadu_si256((ثابت __m256i *)qh);

    ثابت __m256i q5l_0 = _mm256_and_si256(q5bits, هيٺيون_نقاب);
    ثابت __m256i q5h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), k + 0), 4);
    ثابت __m256i q5_0  = mm256_جوڙيو_epi8(q5l_0, q5h_0);
    hmask = _mm256_slli_epi16(hmask, 1);

    ثابت __m256i q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), هيٺيون_نقاب);
    ثابت __m256i q5h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), k + 1), 4);
    ثابت __m256i q5_1  = mm256_جوڙيو_epi8(q5l_1, q5h_1);

    أرجع _mm512_inserti32x8(_mm512_castsi256_si512(q5_0), q5_1, 1);
}

// used for block_q6_K
مضمن فراغ بائيٽون_from_nibbles_128(__m512i& r0, __m512i& r1, ثابت uint8_t * qs, ثابت uint8_t * qh) {
    ثابت __m256i m4 = _mm256_set1_epi8(0xF);
    ثابت __m256i m2 = _mm256_set1_epi8(0x3);

    ثابت __m256i q6bits1 = _mm256_loadu_si256((ثابت __m256i *)qs);
    ثابت __m256i q6bits2 = _mm256_loadu_si256((ثابت __m256i *)(qs + 32));
    ثابت __m256i q6bitsH = _mm256_loadu_si256((ثابت __m256i *)qh);

    ثابت __m256i q6h_0 = _mm256_slli_epi16(_mm256_and_si256(                  q6bitsH,     m2), 4);
    ثابت __m256i q6h_1 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q6bitsH, 2), m2), 4);
    ثابت __m256i q6h_2 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q6bitsH, 4), m2), 4);
    ثابت __m256i q6h_3 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q6bitsH, 6), m2), 4);

    ثابت __m256i q6_0 = _mm256_or_si256(_mm256_and_si256(q6bits1, m4), q6h_0);
    ثابت __m256i q6_1 = _mm256_or_si256(_mm256_and_si256(q6bits2, m4), q6h_1);
    ثابت __m256i q6_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q6bits1, 4), m4), q6h_2);
    ثابت __m256i q6_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q6bits2, 4), m4), q6h_3);

    r0 = _mm512_inserti32x8(_mm512_castsi256_si512(q6_0), q6_1, 1);
    r1 = _mm512_inserti32x8(_mm512_castsi256_si512(q6_2), q6_3, 1);
}

مضمن __m512i packNibbles(__m512i r0, __m512i r1) {
    أرجع _mm512_or_si512(r0, _mm512_slli_epi16(r1, 4));
}

سانچو <قسم_نالو TB>
مضمن فراغ pack_qs(فراغ * RESTRICT packed_B, ثابت TB * RESTRICT B, صحيح KB) {
    int8_t tmp[8 * 64];
    __m256i v[8], v2[8];
    لكل (صحيح n = 0; n < 8; ++n) {
        v[n] = بائيٽون_from_nibbles_32(B[n * KB].qs);
    }
    transpose_8x8_32bit(v, v2);
    لكل (صحيح n = 0; n < 8; ++n) {
        _mm256_storeu_si256((__m256i *)(tmp + n * 64), v2[n]);
    }
    لكل (صحيح n = 0; n < 8; ++n) {
        v[n] = بائيٽون_from_nibbles_32(B[(n + 8) * KB].qs);
    }
    transpose_8x8_32bit(v, v2);
    لكل (صحيح n = 0; n < 8; ++n) {
        _mm256_storeu_si256((__m256i *)(tmp + n * 64 + 32), v2[n]);
    }

    // pack again with 128 to fully utilize vector length
    لكل (صحيح n = 0; n < 8; n += 2) {
        __m512i r0 = _mm512_loadu_si512((ثابت __m512i *)(tmp + n * 64));
        __m512i r1 = _mm512_loadu_si512((ثابت __m512i *)(tmp + n * 64 + 64));
        __m512i r1r0 = packNibbles(r0, r1);
        _mm512_storeu_si512((__m512i *)((حرف *)packed_B + n * 32), r1r0);
    }
}

سانچو <>
مضمن فراغ pack_qs<بلاڪ_q8_0>(فراغ * RESTRICT packed_B, ثابت بلاڪ_q8_0 * RESTRICT B, صحيح KB) {
    __m256i v[8], v2[8];
    لكل (صحيح n = 0; n < 8; ++n) {
        v[n] = _mm256_loadu_si256((ثابت __m256i *)(B[n * KB].qs));
    }
    transpose_8x8_32bit(v, v2);
    لكل (صحيح n = 0; n < 8; ++n) {
        _mm256_storeu_si256((__m256i *)((حرف *)packed_B + n * 64), v2[n]);
    }
    لكل (صحيح n = 0; n < 8; ++n) {
        v[n] = _mm256_loadu_si256((ثابت __m256i *)(B[(n + 8) * KB].qs));
    }
    transpose_8x8_32bit(v, v2);
    لكل (صحيح n = 0; n < 8; ++n) {
        _mm256_storeu_si256((__m256i *)((حرف *)packed_B + n * 64 + 32), v2[n]);
    }
}

سانچو <>
مضمن فراغ pack_qs<بلاڪ_q4_K>(فراغ * RESTRICT packed_B, ثابت بلاڪ_q4_K * RESTRICT B, صحيح KB) {
    __m512i v[16];
    // QK_K 256 with 8 groups, handle 2 groups at a time
    حرف * pb = (حرف *)packed_B;
    لكل (صحيح k = 0; k < QK_K / 64; ++k) {
        // pack 2 groups { n, g,  k} to {g, k/4, 4n}
        //          e.g. {16, 2, 32} to {2,   8, 64}
        لكل (صحيح n = 0; n < TILE_N; ++n) {
            v[n] = بائيٽون_from_nibbles_64(B[n * KB].qs + k * 32);
        }

        transpose_16x16_32bit(v);

        // pack again with 128 to fully utilize vector length
        لكل (صحيح n = 0; n < TILE_N; n += 2) {
            _mm512_storeu_si512((__m512i *)pb, packNibbles(v[n], v[n + 1]));
            pb += 64;
        }
    }
}

سانچو <>
مضمن فراغ pack_qs<بلاڪ_q5_K>(فراغ * RESTRICT packed_B, ثابت بلاڪ_q5_K * RESTRICT B, صحيح KB) {
    __m512i v[16];
    ثابت __m512i هيٺيون_نقاب = _mm512_set1_epi8(0xF);
    // QK_K 256 with 8 groups, handle 2 groups at a time
    حرف * pb = (حرف *)packed_B;
    حرف * ph = (حرف *)packed_B + (QK_K / 2) * TILE_N;
    لكل (صحيح k = 0; k < QK_K / 64; ++k) {
        // pack 2 groups { n, g,  k} to {g, k/4, 4n}
        //          e.g. {16, 2, 32} to {2,   8, 64}
        لكل (صحيح n = 0; n < TILE_N; ++n) {
            v[n] = بائيٽون_from_nibbles_64(B[n * KB].qs + k * 32, B[n * KB].qh, /*گروهه*/2 * k);
        }

        transpose_16x16_32bit(v);

        // 1. pack lower 4bits with 2 groups
        لكل (صحيح n = 0; n < TILE_N; n += 2) {
            // get lower 4 bits
            ثابت __m512i r0 = _mm512_and_si512(v[n], هيٺيون_نقاب);
            ثابت __m512i r1 = _mm512_and_si512(v[n + 1], هيٺيون_نقاب);
            _mm512_storeu_si512((__m512i *)pb, packNibbles(r0, r1)); pb += 64;
        }

        // 2. pack higher 1bit with 2 groups
        ثابت __m512i hmask = _mm512_set1_epi8(0x10);
        لكل (صحيح g = 0; g < 2; ++g) {
            __m512i hbits = _mm512_setzero_si512();
            hbits = mm512_جوڙيو_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 8 + 0], hmask), 4));
            hbits = mm512_جوڙيو_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 8 + 1], hmask), 3));
            hbits = mm512_جوڙيو_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 8 + 2], hmask), 2));
            hbits = mm512_جوڙيو_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 8 + 3], hmask), 1));
            hbits = mm512_جوڙيو_epi8(hbits,                   _mm512_and_si512(v[g * 8 + 4], hmask)    );
            hbits = mm512_جوڙيو_epi8(hbits, _mm512_slli_epi16(_mm512_and_si512(v[g * 8 + 5], hmask), 1));
            hbits = mm512_جوڙيو_epi8(hbits, _mm512_slli_epi16(_mm512_and_si512(v[g * 8 + 6], hmask), 2));
            hbits = mm512_جوڙيو_epi8(hbits, _mm512_slli_epi16(_mm512_and_si512(v[g * 8 + 7], hmask), 3));
            _mm512_storeu_si512((__m512i *)ph, hbits); ph += 64;
        }
    }
}

سانچو <>
مضمن فراغ pack_qs<بلاڪ_q6_K>(فراغ * RESTRICT packed_B, ثابت بلاڪ_q6_K * RESTRICT B, صحيح KB) {
    __m512i v[32];
    ثابت __m512i هيٺيون_نقاب = _mm512_set1_epi8(0xF);
    // QK_K 256 with 8 groups, handle 4 groups at a time
    حرف * pb = (حرف *)packed_B;
    حرف * ph = (حرف *)packed_B + (QK_K / 2) * TILE_N;
    لكل (صحيح k = 0; k < QK_K / 128; ++k) {
        لكل (صحيح n = 0; n < TILE_N; ++n) {
            بائيٽون_from_nibbles_128(v[n], v[n + 16], B[n * KB].ql + k * 64, B[n * KB].qh + k * 32);
        }

        // top half: group 0,1 or 4,5; bottom half: group 2,3 or 6,7
        transpose_16x16_32bit(v);
        transpose_16x16_32bit(v + 16);

        // 1. pack lower 4bits with 4 groups
        لكل (صحيح n = 0; n < 32; n += 2) {
            ثابت __m512i r0 = _mm512_and_si512(v[n], هيٺيون_نقاب);
            ثابت __m512i r1 = _mm512_and_si512(v[n + 1], هيٺيون_نقاب);
            _mm512_storeu_si512((__m512i *)pb, packNibbles(r0, r1)); pb += 64;
        }

        // 2. pack higher 2bit with 4 groups
        ثابت __m512i hmask = _mm512_set1_epi8(0x30);
        لكل (صحيح g = 0; g < 8; ++g) {
            __m512i hbits = _mm512_setzero_si512();
            hbits = mm512_جوڙيو_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 4 + 0], hmask), 4));
            hbits = mm512_جوڙيو_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 4 + 1], hmask), 2));
            hbits = mm512_جوڙيو_epi8(hbits,                   _mm512_and_si512(v[g * 4 + 2], hmask)    );
            hbits = mm512_جوڙيو_epi8(hbits, _mm512_slli_epi16(_mm512_and_si512(v[g * 4 + 3], hmask), 2));
            _mm512_storeu_si512((__m512i *)ph, hbits); ph += 64;
        }
    }
}

سانچو <>
مضمن فراغ pack_qs<بلاڪ_iq4_xs>(فراغ * RESTRICT packed_B, ثابت بلاڪ_iq4_xs * RESTRICT B, صحيح KB) {
    __m512i v[16];
    حرف * pb = (حرف *)packed_B;
    لكل (صحيح k = 0; k < QK_K / 64; ++k) {
        لكل (صحيح n = 0; n < TILE_N; ++n) {
            __m256i r0 = بائيٽون_from_nibbles_32(B[n * KB].qs + k * 32 +  0);
            __m256i r1 = بائيٽون_from_nibbles_32(B[n * KB].qs + k * 32 + 16);
            v[n] = _mm512_inserti32x8(_mm512_castsi256_si512(r0), r1, 1);
        }

        transpose_16x16_32bit(v);

        // pack again with 128 to fully utilize vector length
        لكل (صحيح n = 0; n < TILE_N; n += 2) {
            _mm512_storeu_si512((__m512i *)pb, packNibbles(v[n], v[n + 1]));
            pb += 64;
        }
    }
}

// pack B to vnni formats in 4bits or 8 bits
فراغ pack_B(فراغ * RESTRICT packed_B, ثابت بلاڪ_q4_0 * RESTRICT B, صحيح KB) {
    pack_qs(packed_B, B, KB);
    ggml_half * d0 = reinterpret_cast<ggml_half *>((حرف *)packed_B + TILE_N * TILE_K / 2);
    لكل (صحيح n = 0; n < TILE_N; ++n) {
        d0[n] = B[n * KB].d;
    }
}

فراغ pack_B(فراغ * RESTRICT packed_B, ثابت بلاڪ_q4_1 * RESTRICT B, صحيح KB) {
    pack_qs(packed_B, B, KB);
    ggml_half * d0 = reinterpret_cast<ggml_half *>((حرف *)packed_B + TILE_N * TILE_K / 2);
    ggml_half * m0 = d0 + TILE_N;
    لكل (صحيح n = 0; n < TILE_N; ++n) {
        d0[n] = B[n * KB].d;
        m0[n] = B[n * KB].m;
    }
}

مضمن فراغ s8s8_compensation(فراغ * RESTRICT packed_B) {
    // packed_B layout:
    //   quants {TILE_N, TILEK}  int8_t
    //   d0     {TILE_N}      ggml_half
    //   comp   {TILE_N}        int32_t
    ثابت صحيح وٿي = TILE_N * TILE_K + TILE_N * حجم(ggml_half);
    __m512i vcomp = _mm512_setzero_si512();
    ثابت __m512i off = _mm512_set1_epi8(static_cast<حرف>(0x80));
    لكل (صحيح k = 0; k < 8; ++k) {
        __m512i vb = _mm512_loadu_si512((ثابت __m512i *)((ثابت حرف *)packed_B + k * 64));
        vcomp = _mm512_dpbusd_epi32(vcomp, off, vb);
    }
    _mm512_storeu_si512((__m512i *)((حرف *)(packed_B) + وٿي), vcomp);
}

فراغ pack_B(فراغ * RESTRICT packed_B, ثابت بلاڪ_q8_0 * RESTRICT B, صحيح KB) {
    pack_qs(packed_B, B, KB);
    ggml_half * d0 = reinterpret_cast<ggml_half *>((حرف *)packed_B + TILE_N * TILE_K);
    لكل (صحيح n = 0; n < TILE_N; ++n) {
        d0[n] = B[n * KB].d;
    }
    s8s8_compensation(packed_B);
}

// convert 8 * {min, scale} from int6 to int8
مضمن فراغ unpack_mins_and_scales(ثابت uint8_t * scales, uint32_t * utmp) {
    ثابت uint32_t kmask1 = 0x3f3f3f3f;
    ثابت uint32_t kmask2 = 0x0f0f0f0f;
    ثابت uint32_t kmask3 = 0x03030303;

    يادگيري_نقل(utmp, scales, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    ثابت uint32_t uaux = utmp[1] & kmask1;
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
فراغ pack_B(فراغ * RESTRICT packed_B, ثابت بلاڪ_q4_K * RESTRICT B, صحيح KB) {
    pack_qs(packed_B, B, KB);

    uint8_t * scales = reinterpret_cast<uint8_t *>((حرف *)packed_B + (QK_K / 2) * TILE_N);
    uint8_t * mins = scales + 8 * TILE_N;
    ggml_half * d = reinterpret_cast<ggml_half *>(mins + 8 * TILE_N);
    ggml_half * dmin = d + TILE_N;

    اتحاد {
        uint32_t u32[4];
        uint8_t  u8[16];
    } s;

    لكل (صحيح n = 0; n < TILE_N; ++n) {
        unpack_mins_and_scales(B[n * KB].scales, s.u32);
        لكل (صحيح k = 0; k < 8; ++k) {
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
فراغ pack_B(فراغ * RESTRICT packed_B, ثابت بلاڪ_q5_K * RESTRICT B, صحيح KB) {
    pack_qs(packed_B, B, KB);

    uint8_t * scales = reinterpret_cast<uint8_t *>((حرف *)packed_B + (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N);
    uint8_t * mins = scales + 8 * TILE_N;
    ggml_half * d = reinterpret_cast<ggml_half *>(mins + 8 * TILE_N);
    ggml_half * dmin = d + TILE_N;

    اتحاد {
        uint32_t u32[4];
        uint8_t  u8[16];
    } s;

    لكل (صحيح n = 0; n < TILE_N; ++n) {
        unpack_mins_and_scales(B[n * KB].scales, s.u32);
        لكل (صحيح k = 0; k < 8; ++k) {
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
فراغ pack_B(فراغ * RESTRICT packed_B, ثابت بلاڪ_q6_K * RESTRICT B, صحيح KB) {
    pack_qs(packed_B, B, KB);

    uint8_t * scales = reinterpret_cast<uint8_t *>((حرف *)packed_B + (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N);
    ggml_half * d = reinterpret_cast<ggml_half *>(scales + 16 * TILE_N);
    لكل (صحيح n = 0; n < TILE_N; ++n) {
        ثابت int8_t * ps = B[n * KB].scales;
        لكل (صحيح k = 0; k < 16; ++k) {
            scales[k * TILE_N + n] = ps[k];
        }
        d[n] = B[n * KB].d;
    }
}

// packed_B layout:
//   quants {8, TILE_N, 16}  uint8
//   scales {8, TILE_N}       int8
//   d      {TILE_N}     ggml_half
فراغ pack_B(فراغ * RESTRICT packed_B, ثابت بلاڪ_iq4_xs * RESTRICT B, صحيح KB) {
    pack_qs(packed_B, B, KB);

    int8_t * scales = reinterpret_cast<int8_t *>((حرف *)packed_B + (QK_K / 2) * TILE_N);
    ggml_half * d = reinterpret_cast<ggml_half *>(scales + 8 * TILE_N);

    // pack the scales
    لكل (صحيح n = 0; n < TILE_N; ++n) {
        uint16_t sh = B[n * KB].scales_h;
        لكل (صحيح k = 0; k < 8; k += 2) {
            ثابت int16_t ls1 = ((B[n * KB].scales_l[k / 2] & 0xf) | ((sh << 4) & 0x30)) - 32;
            ثابت int16_t ls2 = ((B[n * KB].scales_l[k / 2] >>  4) | ((sh << 2) & 0x30)) - 32;
            scales[(k + 0) * TILE_N + n] = ls1;
            scales[(k + 1) * TILE_N + n] = ls2;
            sh >>= 4;
        }
        d[n] = B[n * KB].d;
    }
}

سانچو<قسم_نالو TB, قسم_نالو packed_B_t = packed_B_type<TB>>
فراغ unpack_B(packed_B_t * RESTRICT tile, ثابت فراغ * RESTRICT packed_B) {
    GGML_اڻ_استعمال_ٿيل(tile);
    GGML_اڻ_استعمال_ٿيل(packed_B);
}

سانچو <>
فراغ unpack_B<بلاڪ_q4_0>(int8_t * RESTRICT tile, ثابت فراغ * RESTRICT packed_B) {
  ثابت __m512i off = _mm512_set1_epi8(8);
  ثابت __m512i هيٺيون_نقاب = _mm512_set1_epi8(0xF);
  لكل (صحيح n = 0; n < 8; n += 2) {
    __m512i بائيٽون = _mm512_loadu_si512((ثابت __m512i *)((ثابت حرف *)packed_B + n * 32));
    ثابت __m512i r0 = _mm512_sub_epi8(_mm512_and_si512(بائيٽون, هيٺيون_نقاب), off);
    ثابت __m512i r1 = _mm512_sub_epi8(_mm512_and_si512(_mm512_srli_epi16(بائيٽون, 4), هيٺيون_نقاب), off);
    _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
    _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
  }
}

سانچو <>
فراغ unpack_B<بلاڪ_q4_1>(uint8_t * RESTRICT tile, ثابت فراغ * RESTRICT packed_B) {
    ثابت __m512i هيٺيون_نقاب = _mm512_set1_epi8(0xF);
    لكل (صحيح n = 0; n < 8; n += 2) {
        __m512i بائيٽون = _mm512_loadu_si512((ثابت __m512i *)((ثابت حرف *)packed_B + n * 32));
        ثابت __m512i r0 = _mm512_and_si512(بائيٽون, هيٺيون_نقاب);
        ثابت __m512i r1 = _mm512_and_si512(_mm512_srli_epi16(بائيٽون, 4), هيٺيون_نقاب);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
    }
}

// packed_B_t for QKK is int8_t
سانچو <قسم_نالو TB>
فراغ unpack_B(int8_t * RESTRICT tile, ثابت فراغ * RESTRICT packed_B, صحيح k) {
    ثابت صحيح packed_B_گروهه_سائيز = QK_K / 2 * TILE_N / 8;
    ثابت حرف * packed_B_گروهه = (ثابت حرف *)packed_B + k * packed_B_گروهه_سائيز;
    ثابت __m512i هيٺيون_نقاب = _mm512_set1_epi8(0xF);
    لكل (صحيح n = 0; n < 8; n += 2) {
        __m512i بائيٽون = _mm512_loadu_si512(packed_B_گروهه + n * 32);
        ثابت __m512i r0 = _mm512_and_si512(بائيٽون, هيٺيون_نقاب);
        ثابت __m512i r1 = _mm512_and_si512(_mm512_srli_epi16(بائيٽون, 4), هيٺيون_نقاب);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
    }
}

سانچو <>
فراغ unpack_B<بلاڪ_q5_K>(int8_t * RESTRICT tile, ثابت فراغ * RESTRICT packed_B, صحيح k) {
    // lower 4bits, stride 256 bytes
    ثابت صحيح packed_l4_گروهه_سائيز = QK_K / 2 * TILE_N / 8;
    ثابت حرف * pb = (ثابت حرف *)packed_B + k * packed_l4_گروهه_سائيز;

    // higher 1bit, stride 64 bytes
    ثابت صحيح packed_h1_گروهه_سائيز = QK_K / 8 * TILE_N / 8;
    ثابت حرف * ph = (ثابت حرف *)packed_B + (QK_K / 2) * TILE_N + k * packed_h1_گروهه_سائيز;
    ثابت __m512i hbits = _mm512_loadu_si512(ph);

    ثابت __m512i هيٺيون_نقاب = _mm512_set1_epi8(0xF);
    __m512i hmask0 = _mm512_set1_epi8(0x1);
    __m512i hmask1 = _mm512_set1_epi8(0x2);

    لكل (صحيح n = 0; n < 8; n += 2) {
        __m512i بائيٽون = _mm512_loadu_si512(pb + n * 32);
        __m512i r0 = _mm512_and_si512(بائيٽون, هيٺيون_نقاب);
        __m512i r1 = _mm512_and_si512(_mm512_srli_epi16(بائيٽون, 4), هيٺيون_نقاب);
        __m512i h0 = _mm512_slli_epi16(_mm512_srli_epi16(_mm512_and_si512(hbits, hmask0), n), 4);
        __m512i h1 = _mm512_slli_epi16(_mm512_srli_epi16(_mm512_and_si512(hbits, hmask1), n + 1), 4);

        hmask0 = _mm512_slli_epi16(hmask0, 2);
        hmask1 = _mm512_slli_epi16(hmask1, 2);
        r0 = mm512_جوڙيو_epi8(r0, h0);
        r1 = mm512_جوڙيو_epi8(r1, h1);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
    }
}

سانچو <>
فراغ unpack_B<بلاڪ_q6_K>(int8_t * RESTRICT tile, ثابت فراغ * RESTRICT packed_B, صحيح k) {
    // lower 4bits, stride 128 bytes
    ثابت صحيح packed_l4_گروهه_سائيز = QK_K / 2 * TILE_N / 16;
    ثابت حرف * pb = (ثابت حرف *)packed_B + k * packed_l4_گروهه_سائيز;

    // higher 2bits, stride 64 bytes
    ثابت صحيح packed_h2_گروهه_سائيز = QK_K / 4 * TILE_N / 16;
    ثابت حرف * ph = (ثابت حرف *)packed_B + (QK_K / 2) * TILE_N + k * packed_h2_گروهه_سائيز;
    ثابت __m512i hbits = _mm512_loadu_si512(ph);

    ثابت __m512i off = _mm512_set1_epi8(32);
    ثابت __m512i هيٺيون_نقاب = _mm512_set1_epi8(0xF);
    __m512i hmask0 = _mm512_set1_epi8(0x3); // 0011
    __m512i hmask1 = _mm512_set1_epi8(0xC); // 1100

    // notes: skip zero padding from row4 to row7 as we have done so in `unpack_A`
    __m512i بائيٽون = _mm512_loadu_si512(pb);
    __m512i r0 = _mm512_and_si512(بائيٽون, هيٺيون_نقاب);
    __m512i r1 = _mm512_and_si512(_mm512_srli_epi16(بائيٽون, 4), هيٺيون_نقاب);
    __m512i h0 = _mm512_slli_epi16(_mm512_and_si512(hbits, hmask0), 4);
    __m512i h1 = _mm512_slli_epi16(_mm512_and_si512(hbits, hmask1), 2);
    _mm512_storeu_si512((__m512i *)(tile +  0), _mm512_sub_epi8(mm512_جوڙيو_epi8(r0, h0), off));
    _mm512_storeu_si512((__m512i *)(tile + 64), _mm512_sub_epi8(mm512_جوڙيو_epi8(r1, h1), off));

    hmask0 = _mm512_slli_epi16(hmask0, 4);
    hmask1 = _mm512_slli_epi16(hmask1, 4);

    بائيٽون = _mm512_loadu_si512(pb + 64);
    r0 = _mm512_and_si512(بائيٽون, هيٺيون_نقاب);
    r1 = _mm512_and_si512(_mm512_srli_epi16(بائيٽون, 4), هيٺيون_نقاب);
    h0 =                   _mm512_and_si512(hbits, hmask0);
    h1 = _mm512_srli_epi16(_mm512_and_si512(hbits, hmask1), 2);
    _mm512_storeu_si512((__m512i *)(tile + 128), _mm512_sub_epi8(mm512_جوڙيو_epi8(r0, h0), off));
    _mm512_storeu_si512((__m512i *)(tile + 192), _mm512_sub_epi8(mm512_جوڙيو_epi8(r1, h1), off));
}

سانچو <>
فراغ unpack_B<بلاڪ_iq4_xs>(int8_t * RESTRICT tile, ثابت فراغ * RESTRICT packed_B, صحيح k) {
    ساكن ثابت __m512i values128 = mm512_سيٽ_epi8(
        113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
        113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
        113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
        113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127
    );

    ثابت صحيح packed_B_گروهه_سائيز = QK_K / 2 * TILE_N / 8;
    ثابت حرف * pb = (ثابت حرف *)packed_B + k * packed_B_گروهه_سائيز;
    ثابت __m512i هيٺيون_نقاب = _mm512_set1_epi8(0xF);

    لكل (صحيح n = 0; n < 8; n += 2) {
        __m512i بائيٽون = _mm512_loadu_si512(pb + n * 32);
        ثابت __m512i r0 = _mm512_shuffle_epi8(values128, _mm512_and_si512(بائيٽون, هيٺيون_نقاب));
        ثابت __m512i r1 = _mm512_shuffle_epi8(values128, _mm512_and_si512(_mm512_srli_epi16(بائيٽون, 4), هيٺيون_نقاب));
        _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
    }
}

سانچو <قسم_نالو TA, قسم_نالو TB, منطقي is_acc>
بنية acc_C {};

سانچو <منطقي is_acc>
بنية acc_C<بلاڪ_q8_0, بلاڪ_q4_0, is_acc> {
    ساكن فراغ apply(عائم * RESTRICT C, صحيح ldc, ثابت int32_t * RESTRICT tile, ثابت بلاڪ_q8_0 * A, صحيح lda, ثابت فراغ * packed_B, صحيح nr) {
        ثابت صحيح وٿي = TILE_N * TILE_K / 2;
        ثابت __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ثابت __m256i *)((ثابت حرف *)packed_B + وٿي)));

        لكل (صحيح m = 0; m < nr; ++m) {
            ثابت __m512 vd1 = _mm512_set1_ps(GGML_CPU_FP16_منزل_FP32(A[m * lda].d));
            ثابت __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            إذا (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } وإلا {
                vsum = _mm512_set1_ps(0.f);
            }
            vsum = _mm512_fmadd_ps(vtile, _mm512_mul_ps(vd0, vd1), vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

سانچو <منطقي is_acc>
بنية acc_C<بلاڪ_q8_1, بلاڪ_q4_1, is_acc> {
    ساكن فراغ apply(عائم * RESTRICT C, صحيح ldc, ثابت int32_t * RESTRICT tile, ثابت بلاڪ_q8_1 * A, صحيح lda, ثابت فراغ * packed_B, صحيح nr) {
        ثابت صحيح وٿي = TILE_N * TILE_K / 2;
        ثابت __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ثابت __m256i *)((ثابت حرف *)packed_B + وٿي)));
        ثابت __m512 vm0 = _mm512_cvtph_ps(_mm256_loadu_si256((ثابت __m256i *)((ثابت حرف *)packed_B + وٿي + TILE_N * حجم(ggml_half))));

        لكل (صحيح m = 0; m < nr; ++m) {
            ثابت __m512 vd1 = _mm512_set1_ps(GGML_CPU_FP16_منزل_FP32(A[m * lda].d));
            ثابت __m512 vs1 = _mm512_set1_ps(GGML_CPU_FP16_منزل_FP32(A[m * lda].s));
            ثابت __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            إذا (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } وإلا {
                vsum = _mm512_set1_ps(0.f);
            }
            vsum = _mm512_fmadd_ps(vtile, _mm512_mul_ps(vd0, vd1), vsum);
            vsum = _mm512_fmadd_ps(vm0, vs1, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

سانچو <منطقي is_acc>
بنية acc_C<بلاڪ_q8_0, بلاڪ_q8_0, is_acc> {
    ساكن فراغ apply(عائم * RESTRICT C, صحيح ldc, ثابت int32_t * RESTRICT tile, ثابت بلاڪ_q8_0 * A, صحيح lda, ثابت فراغ * packed_B, صحيح nr) {
        ثابت صحيح وٿي = TILE_N * TILE_K;
        ثابت __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ثابت __m256i *)((ثابت حرف *)packed_B + وٿي)));

        لكل (صحيح m = 0; m < nr; ++m) {
            ثابت __m512 vd1 = _mm512_set1_ps(GGML_CPU_FP16_منزل_FP32(A[m * lda].d));
            ثابت __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            إذا (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } وإلا {
                vsum = _mm512_set1_ps(0.f);
            }
            vsum = _mm512_fmadd_ps(vtile, _mm512_mul_ps(vd0, vd1), vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

سانچو <منطقي is_acc>
بنية acc_C<بلاڪ_q8_K, بلاڪ_q4_K, is_acc> {
    ساكن فراغ apply(عائم * RESTRICT C, صحيح ldc, ثابت int32_t * RESTRICT tile, ثابت بلاڪ_q8_K * A, صحيح lda, ثابت فراغ * packed_B, صحيح nr) {
        ثابت uint8_t * scales = reinterpret_cast<ثابت uint8_t *>((ثابت حرف *)packed_B + (QK_K / 2) * TILE_N);
        ثابت uint8_t * mins = scales + 8 * TILE_N;
        ثابت ggml_half * d0 = reinterpret_cast<ثابت ggml_half *>(mins + 8 * TILE_N);
        ثابت ggml_half * dmin = d0 + TILE_N;

        ثابت __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ثابت __m256i *)d0));
        ثابت __m512 vdmin = _mm512_cvtph_ps(_mm256_loadu_si256((ثابت __m256i *)dmin));

        لكل (صحيح m = 0; m < nr; ++m) {
            ثابت عائم d1 = A[m * lda].d;
            ثابت __m512 vd = _mm512_mul_ps(_mm512_set1_ps(d1), vd0);
            ثابت __m512 vdm = _mm512_mul_ps(_mm512_set1_ps(-d1), vdmin);
            ثابت __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            إذا (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } وإلا {
                vsum = _mm512_set1_ps(0.f);
            }

            ثابت __m256i q8sums = _mm256_loadu_si256((ثابت __m256i *)A[m * lda].bsums);
            ثابت __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));

            __m512i acc_m = _mm512_setzero_si512();
            لكل (صحيح k = 0; k < 4; ++k) {
                __m512i vmask = _mm512_set1_epi32(k);
                __m512i va = _mm512_permutexvar_epi32(vmask, _mm512_castsi128_si512(q8s));
                __m512i vb = _mm512_cvtepi8_epi16(_mm256_loadu_si256((ثابت __m256i *)(mins + k * 32)));
                acc_m = _mm512_dpwssds_epi32(acc_m, va, vb);
            }

            vsum = _mm512_fmadd_ps(vtile, vd, vsum);
            vsum = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc_m), vdm, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

سانچو <منطقي is_acc>
بنية acc_C<بلاڪ_q8_K, بلاڪ_q5_K, is_acc> {
    ساكن فراغ apply(عائم * RESTRICT C, صحيح ldc, ثابت int32_t * RESTRICT tile, ثابت بلاڪ_q8_K * A, صحيح lda, ثابت فراغ * packed_B, صحيح nr) {
        ثابت uint8_t * scales = reinterpret_cast<ثابت uint8_t *>((ثابت حرف *)packed_B + (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N);
        ثابت uint8_t * mins = scales + 8 * TILE_N;
        ثابت ggml_half * d0 = reinterpret_cast<ثابت ggml_half *>(mins + 8 * TILE_N);
        ثابت ggml_half * dmin = d0 + TILE_N;

        ثابت __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ثابت __m256i *)d0));
        ثابت __m512 vdmin = _mm512_cvtph_ps(_mm256_loadu_si256((ثابت __m256i *)dmin));

        لكل (صحيح m = 0; m < nr; ++m) {
            ثابت عائم d1 = A[m * lda].d;
            ثابت __m512 vd = _mm512_mul_ps(_mm512_set1_ps(d1), vd0);
            ثابت __m512 vdm = _mm512_mul_ps(_mm512_set1_ps(-d1), vdmin);
            ثابت __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            إذا (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } وإلا {
                vsum = _mm512_set1_ps(0.f);
            }

            ثابت __m256i q8sums = _mm256_loadu_si256((ثابت __m256i *)A[m * lda].bsums);
            ثابت __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));

            __m512i acc_m = _mm512_setzero_si512();
            لكل (صحيح k = 0; k < 4; ++k) {
                __m512i vmask = _mm512_set1_epi32(k);
                __m512i va = _mm512_permutexvar_epi32(vmask, _mm512_castsi128_si512(q8s));
                __m512i vb = _mm512_cvtepi8_epi16(_mm256_loadu_si256((ثابت __m256i *)(mins + k * 32)));
                acc_m = _mm512_dpwssds_epi32(acc_m, va, vb);
            }

            vsum = _mm512_fmadd_ps(vtile, vd, vsum);
            vsum = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc_m), vdm, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

سانچو <منطقي is_acc>
بنية acc_C<بلاڪ_q8_K, بلاڪ_q6_K, is_acc> {
    ساكن فراغ apply(عائم * RESTRICT C, صحيح ldc, ثابت int32_t * RESTRICT tile, ثابت بلاڪ_q8_K * A, صحيح lda, ثابت فراغ * packed_B, صحيح nr) {
        ثابت uint8_t * scales = reinterpret_cast<ثابت uint8_t *>((ثابت حرف *)packed_B + (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N);
        ثابت ggml_half * d0 = reinterpret_cast<ثابت ggml_half *>(scales + 16 * TILE_N);

        ثابت __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ثابت __m256i *)d0));

        لكل (صحيح m = 0; m < nr; ++m) {
            ثابت عائم d1 = A[m * lda].d;
            ثابت __m512 vd = _mm512_mul_ps(_mm512_set1_ps(d1), vd0);
            ثابت __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            إذا (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } وإلا {
                vsum = _mm512_set1_ps(0.f);
            }

            vsum = _mm512_fmadd_ps(vtile, vd, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

سانچو <منطقي is_acc>
بنية acc_C<بلاڪ_q8_K, بلاڪ_iq4_xs, is_acc> {
    ساكن فراغ apply(عائم * RESTRICT C, صحيح ldc, ثابت int32_t * RESTRICT tile, ثابت بلاڪ_q8_K * A, صحيح lda, ثابت فراغ * packed_B, صحيح nr) {
        ثابت int8_t * scales = reinterpret_cast<ثابت int8_t *>((ثابت حرف *)packed_B + (QK_K / 2) * TILE_N);
        ثابت ggml_half * d0 = reinterpret_cast<ثابت ggml_half *>(scales + 8 * TILE_N);

        ثابت __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ثابت __m256i *)d0));

        لكل (صحيح m = 0; m < nr; ++m) {
            ثابت عائم d1 = A[m * lda].d;
            ثابت __m512 vd = _mm512_mul_ps(_mm512_set1_ps(d1), vd0);
            ثابت __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            إذا (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } وإلا {
                vsum = _mm512_set1_ps(0.f);
            }

            vsum = _mm512_fmadd_ps(vtile, vd, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

سانچو <قسم_نالو TB> constexpr صحيح get_quants_سائيز();
سانچو <> constexpr صحيح get_quants_سائيز<بلاڪ_q4_K>() { أرجع (QK_K / 2) * TILE_N; }
سانچو <> constexpr صحيح get_quants_سائيز<بلاڪ_q5_K>() { أرجع (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N; }
سانچو <> constexpr صحيح get_quants_سائيز<بلاڪ_q6_K>() { أرجع (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N; }
سانچو <> constexpr صحيح get_quants_سائيز<بلاڪ_iq4_xs>() { أرجع (QK_K / 2) * TILE_N; }

// used for QKK format
سانچو <قسم_نالو TB, منطقي is_acc,
          قسم_نالو std::enable_if<is_type_qkk<TB>::value, صحيح>::type = 0>
مضمن فراغ scale_C(ثابت int32_t * RESTRICT tile, int32_t * RESTRICT sumi, ثابت فراغ * packed_B, صحيح k, صحيح nr) {
    ثابت uint8_t * scales = reinterpret_cast<ثابت uint8_t *>((ثابت حرف *)packed_B + get_quants_سائيز<TB>());
    ثابت __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((ثابت __m128i *)(scales + k * TILE_N)));

    لكل (صحيح m = 0; m < nr; ++m) {
        __m512i vsumi;
        إذا (is_acc) {
            vsumi = _mm512_loadu_si512(sumi + m * TILE_N);
        } وإلا {
            vsumi = _mm512_setzero_si512();
        }
        __m512i vtile = _mm512_loadu_si512(tile + m * TILE_N);
        vsumi = mm512_جوڙيو_epi32(vsumi, _mm512_mullo_epi32(vtile, vscale));
        _mm512_storeu_si512((__m512i *)(sumi + m * TILE_N), vsumi);
    }
}

سانچو <قسم_نالو TA, قسم_نالو TB, قسم_نالو TC, صحيح بلاڪ_M, صحيح بلاڪ_N, صحيح بلاڪ_K>
بنية tinygemm_kernel_avx {
    ساكن فراغ apply(صحيح K, ثابت TA * RESTRICT A, ثابت TB * RESTRICT B, TC * RESTRICT C, صحيح ldc) {
        GGML_اڻ_استعمال_ٿيل(K);
        GGML_اڻ_استعمال_ٿيل(A);
        GGML_اڻ_استعمال_ٿيل(B);
        GGML_اڻ_استعمال_ٿيل(C);
        GGML_اڻ_استعمال_ٿيل(ldc);
    }
};

سانچو <صحيح بلاڪ_M, صحيح بلاڪ_N, صحيح بلاڪ_K>
بنية tinygemm_kernel_avx<عائم, ggml_fp16_t, عائم, بلاڪ_M, بلاڪ_N, بلاڪ_K> {
    ساكن فراغ apply(صحيح K, ثابت عائم * RESTRICT A, ثابت ggml_fp16_t * RESTRICT B, عائم * RESTRICT C, صحيح ldc) {
        constexpr صحيح صفون_2 = بلاڪ_M;
        constexpr صحيح کانن_ڳڻپ = بلاڪ_N;
        أكد(بلاڪ_K == 16);

        __m512 va;
        __m512 vb[کانن_ڳڻپ];
        __m512 vc[صفون_2 * کانن_ڳڻپ];

        خودڪار loadc = [&](خودڪار idx) {
            vc[idx] = _mm512_setzero_ps();
        };
        Unroll<صفون_2 * کانن_ڳڻپ>{}(loadc);

        خودڪار compute = [&](خودڪار idx, خودڪار k) {
            constexpr صحيح صف = idx / کانن_ڳڻپ;
            constexpr صحيح کانو = idx % کانن_ڳڻپ;

            إذا constexpr (کانو == 0) {
                va = _mm512_loadu_ps(A + صف * K + k);
            }
            إذا constexpr (صف == 0) {
                vb[کانو] =  _mm512_cvtph_ps(_mm256_loadu_si256((ثابت __m256i *)(B + کانو * K + k)));
            }
            vc[idx] = _mm512_fmadd_ps(va, vb[کانو], vc[idx]);
        };

        لكل (صحيح k = 0; k < K; k += 16) {
            Unroll<صفون_2 * کانن_ڳڻپ>{}(compute, k);
        }

        خودڪار storec = [&](خودڪار idx) {
            constexpr صحيح صف = idx / کانن_ڳڻپ;
            constexpr صحيح کانو = idx % کانن_ڳڻپ;
            C[صف * ldc + کانو] = mm512_reduce_جوڙيو_ps(vc[idx]);
        };
        Unroll<صفون_2 * کانن_ڳڻپ>{}(storec);
    }
};

#وصف_ڪريو LAUNCH_TINYGEMM_KERNEL_AVX(MB_سائيز, NB_سائيز)                                \
    tinygemm_kernel_avx<عائم, type, عائم, MB_سائيز, NB_سائيز, blck_سائيز>::apply(    \
        K, (ثابت عائم *)src1->معلومات + src1_وٿي + mb_شروعات * K,                  \
        (ثابت type *)src0->معلومات + src0_وٿي + nb_شروعات * K,                      \
        (عائم *)dst->معلومات + dst_وٿي + mb_شروعات * ldc + nb_شروعات, ldc)


// re-organize in the format {NB, KB, TILE_SIZE}:
#وصف_ڪريو PACKED_اشاريو(n, k, KB, tile_سائيز) (n * KB + k) * tile_سائيز

سانچو<قسم_نالو TB, صحيح بلاڪ_K>
فراغ convert_B_packed_format(فراغ * RESTRICT packed_B, ثابت TB * RESTRICT B, صحيح N, صحيح K) {
    ثابت صحيح NB = N / TILE_N;
    ثابت صحيح KB = K / بلاڪ_K;
    ثابت صحيح TILE_سائيز = get_tile_سائيز<TB>();

    // parallel on NB should be enough
    parallel_for(NB, [&](صحيح begin, صحيح end) {
        لكل (صحيح n = begin; n < end; ++n) {
            لكل (صحيح k = 0; k < KB; ++k) {
                صحيح n0 = n * TILE_N;
                pack_B((حرف *)packed_B + PACKED_اشاريو(n, k, KB, TILE_سائيز), &B[n0 * KB + k], KB);
            }
        }
    });
}

سانچو <قسم_نالو TA, قسم_نالو TB, قسم_نالو TC, صحيح بلاڪ_M, صحيح بلاڪ_N, صحيح بلاڪ_K>
بنية tinygemm_kernel_vnni {};

سانچو <صحيح بلاڪ_M, صحيح بلاڪ_N, صحيح بلاڪ_K>
بنية tinygemm_kernel_vnni<بلاڪ_q8_0, بلاڪ_q4_0, عائم, بلاڪ_M, بلاڪ_N, بلاڪ_K> {
    ساكن فراغ apply(صحيح KB, ثابت فراغ * RESTRICT _A, ثابت فراغ * RESTRICT _B, عائم * RESTRICT C, صحيح ldc) {

        constexpr صحيح کانن_ڳڻپ = بلاڪ_N / 16;
        ثابت صحيح TILE_سائيز = TILE_N * حجم(بلاڪ_q4_0);

        ثابت بلاڪ_q8_0 * RESTRICT A = static_cast<ثابت بلاڪ_q8_0 *>(_A);
        ثابت حرف * RESTRICT B = static_cast<ثابت حرف *>(_B);

        __m512i va[8];
        __m512 vc[کانن_ڳڻپ];
        __m512 vd1;

        // sum of offsets, shared across COLS
        //
        // avx512-vnni does not have `_mm512_dpbssd_epi32`,
        // need to transform ss to us:
        //   a * (b - 8) is equivalent to b * a - 8 * a
        //   s    u   u                   u   s   u   s
        //
        __m512i vcomp;

        ثابت __m512i off = _mm512_set1_epi8(8);
        ثابت __m512i هيٺيون_نقاب = _mm512_set1_epi8(0xF);

        خودڪار loadc = [&](خودڪار کانو) {
            vc[کانو] = _mm512_setzero_ps();
        };
        Unroll<کانن_ڳڻپ>{}(loadc);

        خودڪار compute = [&](خودڪار کانو, خودڪار i) {
            // load a and compute compensation
            إذا constexpr (کانو == 0) {
                ثابت int32_t * a_ptr = reinterpret_cast<ثابت int32_t *>(A[0 * KB + i].qs);
                vcomp = _mm512_setzero_si512();
                لكل (صحيح k = 0; k < 8; ++k) {
                    va[k] = _mm512_set1_epi32(a_ptr[k]);
                    vcomp = _mm512_dpbusd_epi32(vcomp, off, va[k]);
                }
                vd1 = _mm512_set1_ps(GGML_CPU_FP16_منزل_FP32(A[0 * KB + i].d));
            }

            // load b
            __m512i vsum = _mm512_setzero_si512();
            ثابت حرف * b_ptr = B + PACKED_اشاريو(کانو, i, KB, TILE_سائيز);
            لكل (صحيح k = 0; k < 8; k += 2) {
                __m512i بائيٽون = _mm512_loadu_si512((ثابت __m512i *)(b_ptr + k * 32));
                __m512i vb0 = _mm512_and_si512(بائيٽون, هيٺيون_نقاب);
                vsum = _mm512_dpbusd_epi32(vsum, vb0, va[k + 0]);
                __m512i vb1 = _mm512_and_si512(_mm512_srli_epi16(بائيٽون, 4), هيٺيون_نقاب);
                vsum = _mm512_dpbusd_epi32(vsum, vb1, va[k + 1]);
            }
            ثابت صحيح وٿي = TILE_N * TILE_K / 2;
            ثابت __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ثابت __m256i *)(b_ptr + وٿي)));
            vsum = _mm512_sub_epi32(vsum, vcomp);

            vc[کانو] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vsum), _mm512_mul_ps(vd0, vd1), vc[کانو]);
        };

        لكل (صحيح i = 0; i < KB; ++i) {
            Unroll<کانن_ڳڻپ>{}(compute, i);
        }

        //store to C
        خودڪار storec = [&](خودڪار کانو) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + کانو * 16), vc[کانو]);
        };
        Unroll<کانن_ڳڻپ>{}(storec);
    }
};

سانچو <صحيح بلاڪ_N, صحيح بلاڪ_K>
بنية tinygemm_kernel_vnni<بلاڪ_q8_1, بلاڪ_q4_1, عائم, 1, بلاڪ_N, بلاڪ_K> {
    ساكن فراغ apply(صحيح KB, ثابت فراغ * RESTRICT _A, ثابت فراغ * RESTRICT _B, عائم * RESTRICT C, صحيح ldc) {

        constexpr صحيح کانن_ڳڻپ = بلاڪ_N / 16;
        ثابت صحيح TILE_سائيز = TILE_N * حجم(بلاڪ_q4_1);

        ثابت بلاڪ_q8_1 * RESTRICT A = static_cast<ثابت بلاڪ_q8_1 *>(_A);
        ثابت حرف * RESTRICT B = static_cast<ثابت حرف *>(_B);

        __m512i va[8];
        __m512i vb[8];
        __m512 vc[کانن_ڳڻپ];
        __m512 vd1, vs1;

        ثابت __m512i هيٺيون_نقاب = _mm512_set1_epi8(0xF);

        خودڪار loadc = [&](خودڪار کانو) {
            vc[کانو] = _mm512_setzero_ps();
        };
        Unroll<کانن_ڳڻپ>{}(loadc);

        خودڪار compute = [&](خودڪار کانو, خودڪار i) {
            // load a
            إذا constexpr (کانو == 0) {
                ثابت int32_t * a_ptr = reinterpret_cast<ثابت int32_t *>(A[0 * KB + i].qs);
                لكل (صحيح k = 0; k < 8; ++k) {
                    va[k] = _mm512_set1_epi32(a_ptr[k]);
                }
                vd1 = _mm512_set1_ps(GGML_CPU_FP16_منزل_FP32(A[0 * KB + i].d));
                vs1 = _mm512_set1_ps(GGML_CPU_FP16_منزل_FP32(A[0 * KB + i].s));
            }

            // load b
            ثابت حرف * b_ptr = B + PACKED_اشاريو(کانو, i, KB, TILE_سائيز);
            لكل (صحيح k = 0; k < 8; k += 2) {
                __m512i بائيٽون = _mm512_loadu_si512((ثابت __m512i *)(b_ptr + k * 32));
                vb[k + 0] = _mm512_and_si512(بائيٽون, هيٺيون_نقاب);
                vb[k + 1] = _mm512_and_si512(_mm512_srli_epi16(بائيٽون, 4), هيٺيون_نقاب);
            }
            ثابت صحيح وٿي = TILE_N * TILE_K / 2;
            ثابت __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ثابت __m256i *)(b_ptr + وٿي)));
            ثابت __m512 vm0 = _mm512_cvtph_ps(_mm256_loadu_si256((ثابت __m256i *)(b_ptr + وٿي + TILE_N * حجم(ggml_half))));

            __m512i vsum = _mm512_setzero_si512();
            لكل (صحيح k = 0; k < 8; ++k) {
                vsum = _mm512_dpbusd_epi32(vsum, vb[k], va[k]);
            }

            vc[کانو] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vsum), _mm512_mul_ps(vd0, vd1), vc[کانو]);
            vc[کانو] = _mm512_fmadd_ps(vm0, vs1, vc[کانو]);
        };

        لكل (صحيح i = 0; i < KB; ++i) {
            Unroll<کانن_ڳڻپ>{}(compute, i);
        }

        //store to C
        خودڪار storec = [&](خودڪار کانو) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + کانو * 16), vc[کانو]);
        };
        Unroll<کانن_ڳڻپ>{}(storec);
    }
};

سانچو <صحيح بلاڪ_M, صحيح بلاڪ_N, صحيح بلاڪ_K>
بنية tinygemm_kernel_vnni<بلاڪ_q8_0, بلاڪ_q8_0, عائم, بلاڪ_M, بلاڪ_N, بلاڪ_K> {
    ساكن فراغ apply(صحيح KB, ثابت فراغ * RESTRICT _A, ثابت فراغ * RESTRICT _B, عائم * RESTRICT C, صحيح ldc) {

        constexpr صحيح کانن_ڳڻپ = بلاڪ_N / 16;
        ثابت صحيح TILE_سائيز = TILE_N * حجم(بلاڪ_q8_0) + TILE_N * حجم(int32_t);

        ثابت بلاڪ_q8_0 * RESTRICT A = static_cast<ثابت بلاڪ_q8_0 *>(_A);
        ثابت حرف * RESTRICT B = static_cast<ثابت حرف *>(_B);

        __m512i va[8];
        __m512i vb[8];
        __m512 vc[کانن_ڳڻپ];
        __m512 vd1;

        // Notes: s8s8 igemm compensation in avx512-vnni
        // change s8s8 to u8s8 with compensate
        //   a * b = (a + 128) * b - 128 * b
        //   s   s       u       s    u    s
        //
        // (128 * b is pre-computed when packing B to vnni formats)
        //
        ثابت __m512i off = _mm512_set1_epi8(static_cast<حرف>(0x80));

        خودڪار loadc = [&](خودڪار کانو) {
            vc[کانو] = _mm512_setzero_ps();
        };
        Unroll<کانن_ڳڻپ>{}(loadc);

        خودڪار compute = [&](خودڪار کانو, خودڪار i) {
            // load a and add offset 128
            إذا constexpr (کانو == 0) {
                ثابت int32_t * a_ptr = reinterpret_cast<ثابت int32_t *>(A[0 * KB + i].qs);
                لكل (صحيح k = 0; k < 8; ++k) {
                    va[k] = _mm512_set1_epi32(a_ptr[k]);
                    va[k] = mm512_جوڙيو_epi8(va[k], off);
                }
                vd1 = _mm512_set1_ps(GGML_CPU_FP16_منزل_FP32(A[0 * KB + i].d));
            }

            // load b
            ثابت حرف * b_ptr = B + PACKED_اشاريو(کانو, i, KB, TILE_سائيز);
            لكل (صحيح k = 0; k < 8; ++k) {
                vb[k] = _mm512_loadu_si512((ثابت __m512i *)(b_ptr + k * 64));
            }
            ثابت صحيح وٿي = TILE_N * TILE_K;
            ثابت __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ثابت __m256i *)(b_ptr + وٿي)));
            ثابت صحيح offset2 = TILE_N * TILE_K + TILE_N * حجم(ggml_half);
            ثابت __m512i vcomp = _mm512_loadu_si512((ثابت __m512i *)(b_ptr + offset2));

            __m512i vsum = _mm512_setzero_si512();
            لكل (صحيح k = 0; k < 8; ++k) {
                vsum = _mm512_dpbusd_epi32(vsum, va[k], vb[k]);
            }
            vsum = _mm512_sub_epi32(vsum, vcomp);

            vc[کانو] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vsum), _mm512_mul_ps(vd0, vd1), vc[کانو]);
        };

        لكل (صحيح i = 0; i < KB; ++i) {
            Unroll<کانن_ڳڻپ>{}(compute, i);
        }

        //store to C
        خودڪار storec = [&](خودڪار کانو) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + کانو * 16), vc[کانو]);
        };
        Unroll<کانن_ڳڻپ>{}(storec);
    }
};

سانچو <صحيح بلاڪ_M, صحيح بلاڪ_N, صحيح بلاڪ_K>
بنية tinygemm_kernel_vnni<بلاڪ_q8_K, بلاڪ_q4_K, عائم, بلاڪ_M, بلاڪ_N, بلاڪ_K> {
    ساكن فراغ apply(صحيح KB, ثابت فراغ * RESTRICT _A, ثابت فراغ * RESTRICT _B, عائم * RESTRICT C, صحيح ldc) {

        constexpr صحيح کانن_ڳڻپ = بلاڪ_N / 16;
        ثابت صحيح TILE_سائيز = TILE_N * حجم(بلاڪ_q4_K) + TILE_N * 4;

        ثابت بلاڪ_q8_K * RESTRICT A = static_cast<ثابت بلاڪ_q8_K *>(_A);
        ثابت حرف * RESTRICT B = static_cast<ثابت حرف *>(_B);

        // a.qs:   8 groups, 32 bytes each group (m256i)
        __m512i va[8];
        // a.bsum: 8 groups,  2 bytes each group (m128i)
        __m512i va_bsum;
        __m512 vc[کانن_ڳڻپ];
        __m512 vd1;

        // packed_B:
        ثابت صحيح وٿي_scales = (QK_K / 2) * TILE_N;
        ثابت صحيح وٿي_mins   = (QK_K / 2) * TILE_N +  8 * TILE_N;
        ثابت صحيح وٿي_d0     = (QK_K / 2) * TILE_N + 16 * TILE_N;
        ثابت صحيح وٿي_dmin   = (QK_K / 2) * TILE_N + 16 * TILE_N + TILE_N * حجم(ggml_half);

        ثابت __m512i هيٺيون_نقاب = _mm512_set1_epi8(0xF);

        خودڪار loadc = [&](خودڪار کانو) {
            vc[کانو] = _mm512_setzero_ps();
        };
        Unroll<کانن_ڳڻپ>{}(loadc);

        // Notes: vnni formats in QK_K
        //   a) quants vnni format
        //     int8  {k/4, n, 4}, viewed as 2d {k/4, 4n}, k = 32
        //     from {16, 32} to {8, 64}
        //
        //   b) min vnni format
        //     int16 {k/2, n, 2}, viewed as 2d {k/2, 2n}, k = 8
        //     from {16,  8} to {4, 32}
        //
        خودڪار compute = [&](خودڪار کانو, خودڪار i) {
            // load a
            إذا constexpr (کانو == 0) {
                لكل (صحيح k_گروهه = 0; k_گروهه < QK_K / 32; ++k_گروهه) {
                    va[k_گروهه] = _mm512_castsi256_si512(_mm256_loadu_si256((ثابت __m256i *)(A[0 * KB + i].qs + k_گروهه * 32)));
                }
                ثابت __m256i q8sums = _mm256_loadu_si256((ثابت __m256i *)A[0 * KB + i].bsums);
                ثابت __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
                va_bsum = _mm512_castsi128_si512(q8s);
                vd1 = _mm512_set1_ps(A[0 * KB + i].d);
            }

            // step 1: accumultate the quants
            __m512i acc = _mm512_setzero_si512();
            ثابت حرف * b_ptr = B + PACKED_اشاريو(کانو, i, KB, TILE_سائيز);
            ثابت حرف * b_qs  = b_ptr;
            لكل (صحيح k_گروهه = 0; k_گروهه < QK_K / 32; ++k_گروهه) {
                __m512i vsum = _mm512_setzero_si512();
                لكل (صحيح k = 0; k < 8; k += 2) {
                    __m512i va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(k + 0), va[k_گروهه]);
                    __m512i va1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(k + 1), va[k_گروهه]);

                    __m512i بائيٽون = _mm512_loadu_si512((ثابت __m512i *)b_qs);
                    __m512i vb0 = _mm512_and_si512(بائيٽون, هيٺيون_نقاب);
                    vsum = _mm512_dpbusd_epi32(vsum, vb0, va0);
                    __m512i vb1 = _mm512_and_si512(_mm512_srli_epi16(بائيٽون, 4), هيٺيون_نقاب);
                    vsum = _mm512_dpbusd_epi32(vsum, vb1, va1);

                    b_qs += 64;
                }
                // vacc += scale * (q8 @ q4)
                ثابت __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((ثابت __m128i *)(b_ptr + وٿي_scales + k_گروهه * TILE_N)));
                acc = mm512_جوڙيو_epi32(acc, _mm512_mullo_epi32(vsum, vscale));
            }
            ثابت __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ثابت __m256i *)(b_ptr + وٿي_d0)));
            vc[کانو] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(vd0, vd1), vc[کانو]);

            // step 2: accumulate the mins
            __m512i acc_m = _mm512_setzero_si512();
            لكل (صحيح k = 0; k < 4; ++k) {
                __m512i vmask = _mm512_set1_epi32(k);
                __m512i va = _mm512_permutexvar_epi32(vmask, va_bsum);
                __m512i vb = _mm512_cvtepi8_epi16(_mm256_loadu_si256((ثابت __m256i *)(b_ptr + وٿي_mins + k * 32)));
                acc_m = _mm512_dpwssds_epi32(acc_m, va, vb);
            }
            ثابت __m512 vdmin = _mm512_cvtph_ps(_mm256_loadu_si256((ثابت __m256i *)(b_ptr + وٿي_dmin)));
            vc[کانو] = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(acc_m), _mm512_mul_ps(vdmin, vd1), vc[کانو]);
        };

        لكل (صحيح i = 0; i < KB; ++i) {
            Unroll<کانن_ڳڻپ>{}(compute, i);
        }

        //store to C
        خودڪار storec = [&](خودڪار کانو) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + کانو * 16), vc[کانو]);
        };
        Unroll<کانن_ڳڻپ>{}(storec);
    }
};

سانچو <صحيح بلاڪ_M, صحيح بلاڪ_N, صحيح بلاڪ_K>
بنية tinygemm_kernel_vnni<بلاڪ_q8_K, بلاڪ_q5_K, عائم, بلاڪ_M, بلاڪ_N, بلاڪ_K> {
    ساكن فراغ apply(صحيح KB, ثابت فراغ * RESTRICT _A, ثابت فراغ * RESTRICT _B, عائم * RESTRICT C, صحيح ldc) {

        constexpr صحيح کانن_ڳڻپ = بلاڪ_N / 16;
        ثابت صحيح TILE_سائيز = TILE_N * حجم(بلاڪ_q5_K) + TILE_N * 4;

        ثابت بلاڪ_q8_K * RESTRICT A = static_cast<ثابت بلاڪ_q8_K *>(_A);
        ثابت حرف * RESTRICT B = static_cast<ثابت حرف *>(_B);

        // a.qs:   8 groups, 32 bytes each group (m256i)
        __m512i va[8];
        // a.bsum: 8 groups,  2 bytes each group (m128i)
        __m512i va_bsum;
        __m512 vc[کانن_ڳڻپ];
        __m512 vd1;

        // packed_B:
        ثابت صحيح وٿي_qh     = (QK_K / 2) * TILE_N;
        ثابت صحيح وٿي_scales = (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N;
        ثابت صحيح وٿي_mins   = (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N +  8 * TILE_N;
        ثابت صحيح وٿي_d0     = (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N + 16 * TILE_N;
        ثابت صحيح وٿي_dmin   = (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N + 16 * TILE_N + TILE_N * حجم(ggml_half);

        ثابت __m512i هيٺيون_نقاب = _mm512_set1_epi8(0xF);

        خودڪار loadc = [&](خودڪار کانو) {
            vc[کانو] = _mm512_setzero_ps();
        };
        Unroll<کانن_ڳڻپ>{}(loadc);

        // Q5_K and Q4_K shares the same vnni formats, refer to notes above.
        خودڪار compute = [&](خودڪار کانو, خودڪار i) {
            // load a
            إذا constexpr (کانو == 0) {
                لكل (صحيح k_گروهه = 0; k_گروهه < QK_K / 32; ++k_گروهه) {
                    va[k_گروهه] = _mm512_castsi256_si512(_mm256_loadu_si256((ثابت __m256i *)(A[0 * KB + i].qs + k_گروهه * 32)));
                }
                ثابت __m256i q8sums = _mm256_loadu_si256((ثابت __m256i *)A[0 * KB + i].bsums);
                ثابت __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
                va_bsum = _mm512_castsi128_si512(q8s);
                vd1 = _mm512_set1_ps(A[0 * KB + i].d);
            }

            // step 1: accumultate the quants
            __m512i acc = _mm512_setzero_si512();
            ثابت حرف * b_ptr = B + PACKED_اشاريو(کانو, i, KB, TILE_سائيز);
            ثابت حرف * b_qs  = b_ptr;
            ثابت حرف * b_qh  = b_ptr + وٿي_qh;
            لكل (صحيح k_گروهه = 0; k_گروهه < QK_K / 32; ++k_گروهه) {
                __m512i vsum = _mm512_setzero_si512();
                __m512i hmask0 = _mm512_set1_epi8(0x1);
                __m512i hmask1 = _mm512_set1_epi8(0x2);
                __m512i hbits = _mm512_loadu_si512((ثابت __m512i *)(b_qh + k_گروهه * 64));
                لكل (صحيح k = 0; k < 8; k += 2) {
                    __m512i va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(k + 0), va[k_گروهه]);
                    __m512i va1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(k + 1), va[k_گروهه]);

                    __m512i بائيٽون = _mm512_loadu_si512((ثابت __m512i *)b_qs);
                    __m512i vb0 = _mm512_and_si512(بائيٽون, هيٺيون_نقاب);
                    __m512i vb1 = _mm512_and_si512(_mm512_srli_epi16(بائيٽون, 4), هيٺيون_نقاب);

                    __m512i vh0 = _mm512_slli_epi16(_mm512_srli_epi16(_mm512_and_si512(hbits, hmask0), k), 4);
                    __m512i vh1 = _mm512_slli_epi16(_mm512_srli_epi16(_mm512_and_si512(hbits, hmask1), k + 1), 4);

                    hmask0 = _mm512_slli_epi16(hmask0, 2);
                    hmask1 = _mm512_slli_epi16(hmask1, 2);
                    vb0 = mm512_جوڙيو_epi8(vb0, vh0);
                    vb1 = mm512_جوڙيو_epi8(vb1, vh1);

                    vsum = _mm512_dpbusd_epi32(vsum, vb0, va0);
                    vsum = _mm512_dpbusd_epi32(vsum, vb1, va1);

                    b_qs += 64;
                }
                // vacc += scale * (q8 @ q5)
                ثابت __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((ثابت __m128i *)(b_ptr + وٿي_scales + k_گروهه * TILE_N)));
                acc = mm512_جوڙيو_epi32(acc, _mm512_mullo_epi32(vsum, vscale));
            }
            ثابت __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ثابت __m256i *)(b_ptr + وٿي_d0)));
            vc[کانو] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(vd0, vd1), vc[کانو]);

            // step 2: accumulate the mins
            __m512i acc_m = _mm512_setzero_si512();
            لكل (صحيح k = 0; k < 4; ++k) {
                __m512i vmask = _mm512_set1_epi32(k);
                __m512i va = _mm512_permutexvar_epi32(vmask, va_bsum);
                __m512i vb = _mm512_cvtepi8_epi16(_mm256_loadu_si256((ثابت __m256i *)(b_ptr + وٿي_mins + k * 32)));
                acc_m = _mm512_dpwssds_epi32(acc_m, va, vb);
            }
            ثابت __m512 vdmin = _mm512_cvtph_ps(_mm256_loadu_si256((ثابت __m256i *)(b_ptr + وٿي_dmin)));
            vc[کانو] = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(acc_m), _mm512_mul_ps(vdmin, vd1), vc[کانو]);
        };

        لكل (صحيح i = 0; i < KB; ++i) {
            Unroll<کانن_ڳڻپ>{}(compute, i);
        }

        //store to C
        خودڪار storec = [&](خودڪار کانو) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + کانو * 16), vc[کانو]);
        };
        Unroll<کانن_ڳڻپ>{}(storec);
    }
};

سانچو <صحيح بلاڪ_M, صحيح بلاڪ_N, صحيح بلاڪ_K>
بنية tinygemm_kernel_vnni<بلاڪ_q8_K, بلاڪ_q6_K, عائم, بلاڪ_M, بلاڪ_N, بلاڪ_K> {
    ساكن فراغ apply(صحيح KB, ثابت فراغ * RESTRICT _A, ثابت فراغ * RESTRICT _B, عائم * RESTRICT C, صحيح ldc) {

        constexpr صحيح کانن_ڳڻپ = بلاڪ_N / 16;
        ثابت صحيح TILE_سائيز = TILE_N * حجم(بلاڪ_q6_K);

        ثابت بلاڪ_q8_K * RESTRICT A = static_cast<ثابت بلاڪ_q8_K *>(_A);
        ثابت حرف * RESTRICT B = static_cast<ثابت حرف *>(_B);

        // load the 256 bytes from A to 4 avx512 vectors
        __m512i va[4];
        __m512 vc[کانن_ڳڻپ];
        __m512 vd1;

        // packed_B:
        ثابت صحيح وٿي_qh     = (QK_K / 2) * TILE_N;
        ثابت صحيح وٿي_scales = (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N;
        ثابت صحيح وٿي_d0     = (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N + 16 * TILE_N;

        // compensation
        __m512i vcomp;

        ثابت __m512i m32s = _mm512_set1_epi32(32);
        ثابت __m512i هيٺيون_نقاب = _mm512_set1_epi8(0xF);

        خودڪار loadc = [&](خودڪار کانو) {
            vc[کانو] = _mm512_setzero_ps();
        };
        Unroll<کانن_ڳڻپ>{}(loadc);

        خودڪار compute = [&](خودڪار کانو, خودڪار i) {
            إذا constexpr (کانو == 0) {
                // load a
                va[0] = _mm512_loadu_si512((ثابت __m512i *)(A[0 * KB + i].qs +   0));
                va[1] = _mm512_loadu_si512((ثابت __m512i *)(A[0 * KB + i].qs +  64));
                va[2] = _mm512_loadu_si512((ثابت __m512i *)(A[0 * KB + i].qs + 128));
                va[3] = _mm512_loadu_si512((ثابت __m512i *)(A[0 * KB + i].qs + 192));

                ثابت __m256i q8sums = _mm256_loadu_si256((ثابت __m256i *)A[0 * KB + i].bsums);
                vcomp = _mm512_mullo_epi32(_mm512_cvtepi16_epi32(q8sums), m32s);
                vd1 = _mm512_set1_ps(A[0 * KB + i].d);
            }

            // accmulate the quants
            __m512i acc = _mm512_setzero_si512();
            ثابت حرف * b_ptr = B + PACKED_اشاريو(کانو, i, KB, TILE_سائيز);
            ثابت حرف * b_qs = b_ptr;
            ثابت حرف * b_qh = b_ptr + وٿي_qh;
            صحيح نقاب = 0;
            لكل (صحيح k_گروهه = 0; k_گروهه < QK_K / 16; ++k_گروهه) {
                صحيح r = k_گروهه >> 2;
                __m512i va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(نقاب++), va[r]);
                __m512i va1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(نقاب++), va[r]);

                __m512i vsum = _mm512_setzero_si512();
                __m512i hmask = _mm512_set1_epi8(0x3);

                __m512i بائيٽون = _mm512_loadu_si512(b_qs);
                __m512i hbits = _mm512_loadu_si512(b_qh);
                __m512i vb0 = _mm512_and_si512(بائيٽون, هيٺيون_نقاب);
                __m512i vb1 = _mm512_and_si512(_mm512_srli_epi16(بائيٽون, 4), هيٺيون_نقاب);
                __m512i vh0 = _mm512_slli_epi16(_mm512_and_si512(hbits, hmask), 4);
                __m512i vh1 = _mm512_slli_epi16(_mm512_and_si512(hbits, _mm512_slli_epi16(hmask, 2)), 2);

                vb0 = mm512_جوڙيو_epi8(vb0, vh0);
                vb1 = mm512_جوڙيو_epi8(vb1, vh1);
                vsum = _mm512_dpbusd_epi32(vsum, vb0, va0);
                vsum = _mm512_dpbusd_epi32(vsum, vb1, va1);
                b_qs += 64;

                va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(نقاب++), va[r]);
                va1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(نقاب++), va[r]);

                بائيٽون = _mm512_loadu_si512(b_qs);
                vb0 = _mm512_and_si512(بائيٽون, هيٺيون_نقاب);
                vb1 = _mm512_and_si512(_mm512_srli_epi16(بائيٽون, 4), هيٺيون_نقاب);
                vh0 =                   _mm512_and_si512(hbits, _mm512_slli_epi16(hmask, 4));
                vh1 = _mm512_srli_epi16(_mm512_and_si512(hbits, _mm512_slli_epi16(hmask, 6)), 2);
                vb0 = mm512_جوڙيو_epi8(vb0, vh0);
                vb1 = mm512_جوڙيو_epi8(vb1, vh1);
                vsum = _mm512_dpbusd_epi32(vsum, vb0, va0);
                vsum = _mm512_dpbusd_epi32(vsum, vb1, va1);
                b_qs += 64;
                b_qh += 64;

                // B * A - 32 * A
                __m512i vmask = _mm512_set1_epi32(k_گروهه);
                vsum = _mm512_sub_epi32(vsum, _mm512_permutexvar_epi32(vmask, vcomp));

                // vacc += scale * (q8 @ q6)
                ثابت __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((ثابت __m128i *)(b_ptr + وٿي_scales + k_گروهه * TILE_N)));
                acc = mm512_جوڙيو_epi32(acc, _mm512_mullo_epi32(vsum, vscale));
            }
            ثابت __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ثابت __m256i *)(b_ptr + وٿي_d0)));
            vc[کانو] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(vd0, vd1), vc[کانو]);
        };

        لكل (صحيح i = 0; i < KB; ++i) {
            Unroll<کانن_ڳڻپ>{}(compute, i);
        }

        //store to C
        خودڪار storec = [&](صحيح کانو) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + کانو * 16), vc[کانو]);
        };
        Unroll<کانن_ڳڻپ>{}(storec);
    }
};

سانچو <صحيح بلاڪ_M, صحيح بلاڪ_N, صحيح بلاڪ_K>
بنية tinygemm_kernel_vnni<بلاڪ_q8_K, بلاڪ_iq4_xs, عائم, بلاڪ_M, بلاڪ_N, بلاڪ_K> {
    ساكن فراغ apply(صحيح KB, ثابت فراغ * RESTRICT _A, ثابت فراغ * RESTRICT _B, عائم * RESTRICT C, صحيح ldc) {

        constexpr صحيح کانن_ڳڻپ = بلاڪ_N / 16;
        ثابت صحيح TILE_سائيز = TILE_N * حجم(بلاڪ_iq4_xs) + TILE_N * 2;

        ثابت بلاڪ_q8_K * RESTRICT A = static_cast<ثابت بلاڪ_q8_K *>(_A);
        ثابت حرف * RESTRICT B = static_cast<ثابت حرف *>(_B);

        // load the 256 bytes from A to 4 avx512 vectors
        __m512i va[4];
        __m512 vc[کانن_ڳڻپ];
        __m512 vd1;

        // packed_B:
        ثابت صحيح وٿي_scales = (QK_K / 2) * TILE_N ;
        ثابت صحيح وٿي_d0     = (QK_K / 2) * TILE_N + 8 * TILE_N;

        // compensation
        __m512i vcomp;

        ثابت __m256i m128s = _mm256_set1_epi16(128);
        ثابت __m512i هيٺيون_نقاب = _mm512_set1_epi8(0xF);

        ثابت __m512i values128 = mm512_سيٽ_epi8(
            113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
            113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
            113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
            113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127
        );
        ثابت __m512i off = _mm512_set1_epi8(static_cast<حرف>(0x80));
        ثابت __m512i values256 = mm512_جوڙيو_epi8(values128, off);

        خودڪار loadc = [&](خودڪار کانو) {
            vc[کانو] = _mm512_setzero_ps();
        };
        Unroll<کانن_ڳڻپ>{}(loadc);

        خودڪار compute = [&](خودڪار کانو, خودڪار i) {
            إذا constexpr (کانو == 0) {
                // load a
                va[0] = _mm512_loadu_si512((ثابت __m512i *)(A[0 * KB + i].qs +   0));
                va[1] = _mm512_loadu_si512((ثابت __m512i *)(A[0 * KB + i].qs +  64));
                va[2] = _mm512_loadu_si512((ثابت __m512i *)(A[0 * KB + i].qs + 128));
                va[3] = _mm512_loadu_si512((ثابت __m512i *)(A[0 * KB + i].qs + 192));

                // compensation: 128 * A
                ثابت __m256i q8sums = _mm256_loadu_si256((ثابت __m256i *)A[0 * KB + i].bsums);
                vcomp = _mm512_castsi256_si512(_mm256_madd_epi16(q8sums, m128s));
                vd1 = _mm512_set1_ps(A[0 * KB + i].d);
            }

            // accmulate the quants
            __m512i acc = _mm512_setzero_si512();
            ثابت حرف * b_ptr = B + PACKED_اشاريو(کانو, i, KB, TILE_سائيز);
            ثابت حرف * b_qs = b_ptr;
            صحيح نقاب = 0;
            لكل (صحيح k_گروهه = 0; k_گروهه < QK_K / 32; ++k_گروهه) {
                صحيح r = k_گروهه >> 1;
                __m512i vmask = _mm512_set1_epi32(k_گروهه);
                __m512i vsum = _mm512_setzero_si512();
                لكل (صحيح k = 0; k < 8; k += 2) {
                    __m512i va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(نقاب++), va[r]);
                    __m512i va1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(نقاب++), va[r]);

                    __m512i بائيٽون = _mm512_loadu_si512(b_qs);
                    __m512i vb0 = _mm512_shuffle_epi8(values256, _mm512_and_si512(بائيٽون, هيٺيون_نقاب));
                    __m512i vb1 = _mm512_shuffle_epi8(values256, _mm512_and_si512(_mm512_srli_epi16(بائيٽون, 4), هيٺيون_نقاب));

                    vsum = _mm512_dpbusd_epi32(vsum, vb0, va0);
                    vsum = _mm512_dpbusd_epi32(vsum, vb1, va1);
                    b_qs += 64;
                }
                // (B + 128) * A - 128 * A
                vsum = _mm512_sub_epi32(vsum, _mm512_permutexvar_epi32(vmask, vcomp));

                // vacc += scale * (q8 @ q4)
                ثابت __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((ثابت __m128i *)(b_ptr + وٿي_scales + k_گروهه * TILE_N)));
                acc = mm512_جوڙيو_epi32(acc, _mm512_mullo_epi32(vsum, vscale));
            }
            ثابت __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((ثابت __m256i *)(b_ptr + وٿي_d0)));
            vc[کانو] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(vd0, vd1), vc[کانو]);
        };

        لكل (صحيح i = 0; i < KB; ++i) {
            Unroll<کانن_ڳڻپ>{}(compute, i);
        }

        //store to C
        خودڪار storec = [&](خودڪار کانو) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + کانو * 16), vc[کانو]);
        };
        Unroll<کانن_ڳڻپ>{}(storec);
    }
};

#وصف_ڪريو LAUNCH_TINYGEMM_KERNEL_VNNI(NB_سائيز)                                                   \
    tinygemm_kernel_vnni<vec_dot_type, type, عائم, 1, NB_سائيز, blck_سائيز>::apply(             \
        KB, wdata_batch,                                                                       \
        (ثابت حرف *)src0->معلومات + src0_وٿي + PACKED_اشاريو(nb * kTilesN, 0, KB, TILE_سائيز), \
        (عائم *) dst->معلومات + dst_وٿي + nb_شروعات, ldc)

سانچو <قسم_نالو TA, قسم_نالو TB, قسم_نالو TC, صحيح بلاڪ_K,
          قسم_نالو std::enable_if<!is_type_qkk<TB>::value, صحيح>::type = 0>
فراغ tinygemm_kernel_amx(صحيح M, صحيح N, صحيح KB, ثابت فراغ * RESTRICT _A, ثابت فراغ * RESTRICT _B, TC * RESTRICT C, صحيح ldc) {
    استعمال packed_B_t = packed_B_type<TB>;
    ثابت صحيح TILE_سائيز = get_tile_سائيز<TB>();
    ثابت منطقي need_unpack = do_unpack<TB>::value;

    GGML_ASSERT(M <= 2 * TILE_M && N == 2 * TILE_N);
    ثابت TA * RESTRICT A = static_cast<ثابت TA *>(_A);
    ثابت حرف * RESTRICT B = static_cast<ثابت حرف *>(_B);

    ثابت صحيح m0 = std::min(M, TILE_M);
    ثابت صحيح m1 = std::max(M - TILE_M, 0);
    ثابت صحيح lda = KB * حجم(TA);
    //const int ldb = KB * sizeof(TB);

    alignas(64) ساكن thread_local packed_B_t Tile0[TILE_N * TILE_K];
    alignas(64) ساكن thread_local packed_B_t Tile1[TILE_N * TILE_K];
    alignas(64) ساكن thread_local int8_t Tile23[TILE_M * TILE_K];

    alignas(64) ساكن thread_local int32_t TileC0[TILE_M * TILE_N * 4];
    alignas(64) ساكن thread_local int32_t TileC1[TILE_M * TILE_N * 4];

    // double buffering C to interleave avx512 and amx
    int32_t * C_cur = TileC0;
    int32_t * C_pre = TileC1;

    خودڪار Tile4 = [&](int32_t * بنياد) { أرجع بنياد; };
    خودڪار Tile5 = [&](int32_t * بنياد) { أرجع بنياد + TILE_M * TILE_N; };
    خودڪار Tile6 = [&](int32_t * بنياد) { أرجع بنياد + 2 * TILE_M * TILE_N; };
    خودڪار Tile7 = [&](int32_t * بنياد) { أرجع بنياد + 3 * TILE_M * TILE_N; };

    إذا (M == 2 * TILE_M) {
        // i = 0
        ثابت حرف * B_blk0 = B + PACKED_اشاريو(0, 0, KB, TILE_سائيز);
        ثابت حرف * B_blk1 = B + PACKED_اشاريو(1, 0, KB, TILE_سائيز);
        إذا (need_unpack) {
            unpack_B<TB>(Tile0, B_blk0);
            _tile_loadd(TMM0, Tile0, TILE_N * VNNI_BLK);
        } وإلا {
            _tile_loadd(TMM0, B_blk0, TILE_N * VNNI_BLK);
        }

        _tile_zero(TMM4);
        _tile_loadd(TMM2, A[0].qs, lda);
        _tile_dpbssd(TMM4, TMM2, TMM0);
        _tile_stored(TMM4, Tile4(C_pre), TILE_N * حجم(int32_t));

        _tile_zero(TMM5);
        _tile_loadd(TMM3, A[TILE_M * KB + 0].qs, lda);
        _tile_dpbssd(TMM5, TMM3, TMM0);
        _tile_stored(TMM5, Tile5(C_pre), TILE_N * حجم(int32_t));

        إذا (need_unpack) {
            unpack_B<TB>(Tile1, B_blk1);
            _tile_loadd(TMM1, Tile1, TILE_N * VNNI_BLK);
        } وإلا {
            _tile_loadd(TMM1, B_blk1, TILE_N * VNNI_BLK);
        }

        _tile_zero(TMM6);
        _tile_dpbssd(TMM6, TMM2, TMM1);
        _tile_stored(TMM6, Tile6(C_pre), TILE_N * حجم(int32_t));

        _tile_zero(TMM7);
        _tile_dpbssd(TMM7, TMM3, TMM1);
        _tile_stored(TMM7, Tile7(C_pre), TILE_N * حجم(int32_t));

        لكل (صحيح i = 1; i < KB; ++i) {
            // index of previous iter
            ثابت صحيح ii = i - 1;
            ثابت حرف * B_blk0 = B + PACKED_اشاريو(0, i, KB, TILE_سائيز);
            ثابت حرف * B_blk1 = B + PACKED_اشاريو(1, i, KB, TILE_سائيز);
            GGML_DISPATCH_BOOL(ii > 0, is_acc, [&] {
                إذا (need_unpack) {
                    unpack_B<TB>(Tile0, B_blk0);
                    _tile_loadd(TMM0, Tile0, TILE_N * VNNI_BLK);
                } وإلا {
                    _tile_loadd(TMM0, B_blk0, TILE_N * VNNI_BLK);
                }
                _tile_zero(TMM4);
                _tile_loadd(TMM2, A[i].qs, lda);
                acc_C<TA, TB, is_acc>::apply(C, ldc, Tile4(C_pre), &A[ii], KB, B + PACKED_اشاريو(0, ii, KB, TILE_سائيز), TILE_M);

                _tile_dpbssd(TMM4, TMM2, TMM0);
                _tile_stored(TMM4, Tile4(C_cur), TILE_N * حجم(int32_t));

                _tile_zero(TMM5);
                _tile_loadd(TMM3, A[TILE_M * KB + i].qs, lda);
                acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc, ldc, Tile5(C_pre), &A[TILE_M * KB + ii], KB, B + PACKED_اشاريو(0, ii, KB, TILE_سائيز), TILE_M);

                _tile_dpbssd(TMM5, TMM3, TMM0);
                _tile_stored(TMM5, Tile5(C_cur), TILE_N * حجم(int32_t));

                إذا (need_unpack) {
                    unpack_B<TB>(Tile1, B_blk1);
                    _tile_loadd(TMM1, Tile1, TILE_N * VNNI_BLK);
                } وإلا {
                    _tile_loadd(TMM1, B_blk1, TILE_N * VNNI_BLK);
                }
                _tile_zero(TMM6);
                acc_C<TA, TB, is_acc>::apply(C + TILE_N, ldc, Tile6(C_pre), &A[ii], KB, B + PACKED_اشاريو(1, ii, KB, TILE_سائيز), TILE_M);

                _tile_dpbssd(TMM6, TMM2, TMM1);
                _tile_stored(TMM6, Tile6(C_cur), TILE_N * حجم(int32_t));

                _tile_zero(TMM7);
                acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc + TILE_N, ldc, Tile7(C_pre), &A[TILE_M * KB + ii], KB, B + PACKED_اشاريو(1, ii, KB, TILE_سائيز), TILE_M);

                _tile_dpbssd(TMM7, TMM3, TMM1);
                _tile_stored(TMM7, Tile7(C_cur), TILE_N * حجم(int32_t));

                std::swap(C_cur, C_pre);
            });
        }
        // final accumulation
        {
            صحيح ii = KB - 1;
            acc_C<TA, TB, سچ>::apply(C, ldc, Tile4(C_pre), &A[ii], KB, B + PACKED_اشاريو(0, ii, KB, TILE_سائيز), TILE_M);
            acc_C<TA, TB, سچ>::apply(C + TILE_M * ldc, ldc, Tile5(C_pre), &A[TILE_M * KB + ii], KB, B + PACKED_اشاريو(0, ii, KB, TILE_سائيز), TILE_M);
            acc_C<TA, TB, سچ>::apply(C + TILE_N, ldc, Tile6(C_pre), &A[ii], KB, B + PACKED_اشاريو(1, ii, KB, TILE_سائيز), TILE_M);
            acc_C<TA, TB, سچ>::apply(C + TILE_M * ldc + TILE_N, ldc, Tile7(C_pre), &A[TILE_M * KB + ii], KB, B + PACKED_اشاريو(1, ii, KB, TILE_سائيز), TILE_M);
        }
    } وإلا {
        لكل (صحيح i = 0; i < KB; ++i) {
            _tile_zero(TMM4);
            _tile_zero(TMM6);
            إذا (m1 != 0) {
                _tile_zero(TMM5);
                _tile_zero(TMM7);
            }

            ثابت حرف * B_blk0 = B + PACKED_اشاريو(0, i, KB, TILE_سائيز);
            ثابت حرف * B_blk1 = B + PACKED_اشاريو(1, i, KB, TILE_سائيز);
            إذا (need_unpack) {
                unpack_B<TB>(Tile0, B_blk0);
                _tile_loadd(TMM0, Tile0, TILE_N * VNNI_BLK);
            } وإلا {
                _tile_loadd(TMM0, B_blk0, TILE_N * VNNI_BLK);
            }

            إذا (need_unpack) {
                unpack_B<TB>(Tile1, B_blk1);
                _tile_loadd(TMM1, Tile1, TILE_N * VNNI_BLK);
            } وإلا {
                _tile_loadd(TMM1, B_blk1, TILE_N * VNNI_BLK);
            }

            إذا (m0 == TILE_M) {
                _tile_loadd(TMM2, A[i].qs, lda);
            } وإلا {
                unpack_A(Tile23, &A[i], KB, m0);
                _tile_loadd(TMM2, Tile23, TILE_K);
            }

            _tile_dpbssd(TMM4, TMM2, TMM0);
            _tile_dpbssd(TMM6, TMM2, TMM1);

            _tile_stored(TMM4, Tile4(C_cur), TILE_N * حجم(int32_t));
            _tile_stored(TMM6, Tile6(C_cur), TILE_N * حجم(int32_t));

            GGML_DISPATCH_BOOL(i > 0, is_acc, [&] {
                acc_C<TA, TB, is_acc>::apply(C,          ldc, Tile4(C_cur), &A[i], KB, B + PACKED_اشاريو(0, i, KB, TILE_سائيز), m0);
                acc_C<TA, TB, is_acc>::apply(C + TILE_N, ldc, Tile6(C_cur), &A[i], KB, B + PACKED_اشاريو(1, i, KB, TILE_سائيز), m0);
            });

            إذا (m1 != 0) {
                unpack_A(Tile23, &A[TILE_M * KB + i], KB, m1);
                _tile_loadd(TMM3, Tile23, TILE_K);

                _tile_dpbssd(TMM5, TMM3, TMM0);
                _tile_dpbssd(TMM7, TMM3, TMM1);
                _tile_stored(TMM5, Tile5(C_cur), TILE_N * حجم(int32_t));
                _tile_stored(TMM7, Tile7(C_cur), TILE_N * حجم(int32_t));
                GGML_DISPATCH_BOOL(i > 0, is_acc, [&] {
                    acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc,          ldc, Tile5(C_cur), &A[TILE_M * KB + i], KB, B + PACKED_اشاريو(0, i, KB, TILE_سائيز), m1);
                    acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc + TILE_N, ldc, Tile7(C_cur), &A[TILE_M * KB + i], KB, B + PACKED_اشاريو(1, i, KB, TILE_سائيز), m1);
                });
            }
        }
    }
    أرجع;
}

سانچو <قسم_نالو TA, قسم_نالو TB, قسم_نالو TC, صحيح بلاڪ_K,
          قسم_نالو std::enable_if<is_type_qkk<TB>::value, صحيح>::type = 0>
فراغ tinygemm_kernel_amx(صحيح M, صحيح N, صحيح KB, ثابت فراغ * RESTRICT _A, ثابت فراغ * RESTRICT _B, عائم * RESTRICT C, صحيح ldc) {
    static_assert(std::is_same<TA, بلاڪ_q8_K>::value);
    ثابت صحيح TILE_سائيز = get_tile_سائيز<TB>();

    GGML_ASSERT(M <= 2 * TILE_M && N == 2 * TILE_N);
    ثابت TA * RESTRICT A = static_cast<ثابت TA *>(_A);
    ثابت حرف * RESTRICT B = static_cast<ثابت حرف *>(_B);

    ثابت صحيح m0 = std::min(M, TILE_M);
    ثابت صحيح m1 = std::max(M - TILE_M, 0);
    //const int lda = KB * sizeof(TA);

    alignas(64) ساكن thread_local int8_t Tile0[TILE_N * TILE_K];
    alignas(64) ساكن thread_local int8_t Tile1[TILE_N * TILE_K];
    alignas(64) ساكن thread_local int8_t Tile23[TILE_M * TILE_K];

    // mat mul result for each group
    alignas(64) ساكن thread_local int32_t Tile4[TILE_M * TILE_N];
    alignas(64) ساكن thread_local int32_t Tile5[TILE_M * TILE_N];
    alignas(64) ساكن thread_local int32_t Tile6[TILE_M * TILE_N];
    alignas(64) ساكن thread_local int32_t Tile7[TILE_M * TILE_N];

    // sum of each QK_K block, contains 8 groups, int32
    alignas(64) ساكن thread_local int32_t Sumi4[TILE_M * TILE_N];
    alignas(64) ساكن thread_local int32_t Sumi5[TILE_M * TILE_N];
    alignas(64) ساكن thread_local int32_t Sumi6[TILE_M * TILE_N];
    alignas(64) ساكن thread_local int32_t Sumi7[TILE_M * TILE_N];

    ثابت صحيح k_گروهه_سائيز = std::is_same<TB, بلاڪ_q6_K>::value ? 16 : 32;
    لكل (صحيح i = 0; i < KB; ++i) {
        // step 1: accumulate the quants across 8 groups, each group with 32
        لكل (صحيح k = 0; k < QK_K / k_گروهه_سائيز; ++k) {
            GGML_DISPATCH_BOOL(k > 0, is_acc, [&] {
                _tile_zero(TMM4);
                _tile_zero(TMM6);

                unpack_B<TB>(Tile0, B + PACKED_اشاريو(0, i, KB, TILE_سائيز), k);
                _tile_loadd(TMM0, Tile0, TILE_N * VNNI_BLK);

                unpack_B<TB>(Tile1, B + PACKED_اشاريو(1, i, KB, TILE_سائيز), k);
                _tile_loadd(TMM1, Tile1, TILE_N * VNNI_BLK);

                unpack_A<TB>(Tile23, &A[i], KB, k, m0);
                _tile_loadd(TMM2, Tile23, TILE_K);

                _tile_dpbssd(TMM4, TMM2, TMM0);
                _tile_dpbssd(TMM6, TMM2, TMM1);

                _tile_stored(TMM4, Tile4, TILE_N * حجم(int32_t));
                _tile_stored(TMM6, Tile6, TILE_N * حجم(int32_t));

                scale_C<TB, is_acc>(Tile4, Sumi4, B + PACKED_اشاريو(0, i, KB, TILE_سائيز), k, m0);
                scale_C<TB, is_acc>(Tile6, Sumi6, B + PACKED_اشاريو(1, i, KB, TILE_سائيز), k, m0);

                إذا (m1 != 0) {
                    _tile_zero(TMM5);
                    _tile_zero(TMM7);

                    unpack_A<TB>(Tile23, &A[TILE_M * KB + i], KB, k, m1);
                    _tile_loadd(TMM3, Tile23, TILE_K);

                    _tile_dpbssd(TMM5, TMM3, TMM0);
                    _tile_dpbssd(TMM7, TMM3, TMM1);

                    _tile_stored(TMM5, Tile5, TILE_N * حجم(int32_t));
                    _tile_stored(TMM7, Tile7, TILE_N * حجم(int32_t));

                    scale_C<TB, is_acc>(Tile5, Sumi5, B + PACKED_اشاريو(0, i, KB, TILE_سائيز), k, m1);
                    scale_C<TB, is_acc>(Tile7, Sumi7, B + PACKED_اشاريو(1, i, KB, TILE_سائيز), k, m1);
                }
            });
        }

        // step 2: accmulate the mins
        GGML_DISPATCH_BOOL(i > 0, is_acc, [&] {
            acc_C<TA, TB, is_acc>::apply(C,          ldc, Sumi4, &A[i], KB, B + PACKED_اشاريو(0, i, KB, TILE_سائيز), m0);
            acc_C<TA, TB, is_acc>::apply(C + TILE_N, ldc, Sumi6, &A[i], KB, B + PACKED_اشاريو(1, i, KB, TILE_سائيز), m0);
            إذا (m1 != 0) {
                acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc,          ldc, Sumi5, &A[TILE_M * KB + i], KB, B + PACKED_اشاريو(0, i, KB, TILE_سائيز), m1);
                acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc + TILE_N, ldc, Sumi7, &A[TILE_M * KB + i], KB, B + PACKED_اشاريو(1, i, KB, TILE_سائيز), m1);
            }
        });
    }
    أرجع;
}

} // anonymous namespace

// get the packed tensor size for quantized weights
size_t ggml_backend_amx_get_alloc_سائيز(ثابت بنية ggml_tensor * tensor) {
    ثابت تعداد ggml_type TYPE = tensor->type;

    ثابت صحيح K = tensor->ne[0]; // ne0: in_features
    ثابت صحيح N = tensor->ne[1]; // ne1: out_features

    خودڪار get_tensor_سائيز = [&] {
        size_t صف_سائيز_B{0};
        GGML_DISPATCH_QTYPES(TYPE, [&] {
            صف_سائيز_B = get_صف_سائيز<type, blck_سائيز>(K);
        });
        أرجع N * صف_سائيز_B;
    };

    إذا (qtype_has_amx_kernels(TYPE)) {
        أرجع get_tensor_سائيز();
    } وإلا {
        // for f16, bf16 we don't do packing
        أرجع ggml_nbytes(tensor);
    }
}

// pack weight to vnni format
فراغ ggml_backend_amx_convert_وزن(بنية ggml_tensor * tensor, ثابت فراغ * معلومات, size_t وٿي, size_t size) {
    GGML_ASSERT(وٿي == 0 && size == ggml_nbytes(tensor)); // only full tensor conversion is supported for now

    ثابت تعداد ggml_type TYPE = tensor->type;

    ثابت صحيح K = tensor->ne[0]; // ne0: in_features
    ثابت صحيح N = tensor->ne[1]; // ne1: out_features

    GGML_DISPATCH_QTYPES(TYPE, [&] {
        convert_B_packed_format<type, blck_سائيز>((فراغ *)((حرف *)tensor->معلومات + وٿي), (ثابت type *)معلومات, N, K);
    });
}

// ne2 is passed explicitly to help compiler optimize repeated calls
مضمن int64_t ggml_batch_وٿي(ثابت ggml_tensor * t, int64_t batch_idx, int64_t ne2) {
    ثابت int64_t i2 = batch_idx % ne2;
    ثابت int64_t i3 = batch_idx / ne2;
    أرجع i3 * t->nb[3] + i2 * t->nb[2];
}

size_t ggml_backend_amx_desired_wsize(ثابت بنية ggml_tensor * dst) {
    بنية ggml_tensor * src0 = dst->src[0];

    ثابت تعداد ggml_type TYPE = src0->type;

    ثابت منطقي is_floating_type = TYPE == GGML_TYPE_F16;
    إذا (is_floating_type) {
        أرجع 0;
    }

    ثابت صحيح M = dst->ne[1];
    ثابت صحيح K = src0->ne[0];
    ثابت int64_t n_batch = dst->ne[2] * dst->ne[3];

    size_t desired_wsize = 0;

    GGML_DISPATCH_QTYPES(TYPE, [&] {
        ثابت size_t صف_سائيز_A = K / blck_سائيز * حجم(vec_dot_type);
        desired_wsize = n_batch * M * صف_سائيز_A;
    });

    أرجع desired_wsize;
}

// NB: mixed dtype gemm with Advanced Matrix Extensions (Intel AMX)
//
// src0: weight in shape of {N, K}, quantized
// src1: input  in shape of {M, K}, float32
// dst:  output in shape of {M, N}, float32
//
// the function performs: dst = src1 @ src0.T for each batch
//
فراغ ggml_backend_amx_mul_mat(ثابت ggml_compute_params * params, بنية ggml_tensor * dst) {
    بنية ggml_tensor * src0 = dst->src[0];
    بنية ggml_tensor * src1 = dst->src[1];

    ثابت تعداد ggml_type TYPE = src0->type;

    // f16 only has avx512 kernels for now,
    // amx kernels will be added once 6th gen xeon is released.
    ثابت منطقي is_floating_type = TYPE == GGML_TYPE_F16;

    ثابت صحيح M = dst->ne[1];
    ثابت صحيح N = dst->ne[0];
    ثابت صحيح K = src0->ne[0];
    ثابت صحيح ldc = dst->nb[1] / dst->nb[0];

    ثابت int64_t ne2 = dst->ne[2];
    ثابت int64_t n_batch = ne2 * dst->ne[3];

    إذا (is_floating_type) {
        constexpr صحيح بلاڪ_M = 4;
        constexpr صحيح بلاڪ_N = 6;
        ثابت صحيح MB = div_مٿي(M, بلاڪ_M);
        ثابت صحيح NB = div_مٿي(N, بلاڪ_N);

        parallel_for_ggml(params, n_batch * MB * NB, [&](صحيح begin, صحيح end) {
            GGML_DISPATCH_FLOATING_TYPES(TYPE, [&] {
                لكل (صحيح i = begin; i < end; ++i) {
                    صحيح batch_idx = i / (MB * NB);
                    صحيح بچيل = i % (MB * NB);
                    صحيح mb = بچيل / NB;
                    صحيح nb = بچيل % NB;

                    int64_t src0_وٿي = ggml_batch_وٿي(src0, batch_idx, ne2);
                    int64_t src1_وٿي = ggml_batch_وٿي(src1, batch_idx, ne2);
                    int64_t dst_وٿي  = ggml_batch_وٿي(dst,  batch_idx, ne2);

                    صحيح mb_شروعات = mb * بلاڪ_M;
                    صحيح mb_سائيز = std::min(بلاڪ_M, M - mb_شروعات);
                    صحيح nb_شروعات = nb * بلاڪ_N;
                    صحيح nb_سائيز = std::min(بلاڪ_N, N - nb_شروعات);

                    بدل (mb_سائيز << 4 | nb_سائيز) {
                        حالة 0x12: LAUNCH_TINYGEMM_KERNEL_AVX(1, 2); اكسر;
                        حالة 0x14: LAUNCH_TINYGEMM_KERNEL_AVX(1, 4); اكسر;
                        حالة 0x16: LAUNCH_TINYGEMM_KERNEL_AVX(1, 6); اكسر;
                        حالة 0x22: LAUNCH_TINYGEMM_KERNEL_AVX(2, 2); اكسر;
                        حالة 0x24: LAUNCH_TINYGEMM_KERNEL_AVX(2, 4); اكسر;
                        حالة 0x26: LAUNCH_TINYGEMM_KERNEL_AVX(2, 6); اكسر;
                        حالة 0x32: LAUNCH_TINYGEMM_KERNEL_AVX(3, 2); اكسر;
                        حالة 0x34: LAUNCH_TINYGEMM_KERNEL_AVX(3, 4); اكسر;
                        حالة 0x36: LAUNCH_TINYGEMM_KERNEL_AVX(3, 6); اكسر;
                        حالة 0x42: LAUNCH_TINYGEMM_KERNEL_AVX(4, 2); اكسر;
                        حالة 0x44: LAUNCH_TINYGEMM_KERNEL_AVX(4, 4); اكسر;
                        حالة 0x46: LAUNCH_TINYGEMM_KERNEL_AVX(4, 6); اكسر;
                        افتراضي: fprintf(stderr, "Unexpected block size!\n");
                    }
                }
            });
        });
        أرجع;
    }

    // pointer to work space, used convert A from float to quantized type
    فراغ * wdata = params->wdata;

    //TODO: performance improvement: merge quant A
 // if (params->ith == 0) {
        GGML_DISPATCH_QTYPES(TYPE, [&] {
            ثابت size_t صف_سائيز_A = K / blck_سائيز * حجم(vec_dot_type);
            ثابت size_t desired_wsize = n_batch * M * صف_سائيز_A;
            إذا (params->wsize < desired_wsize) {
                GGML_ABORT("insufficient work space size");
            }

            // Q4_0, Q4_1, Q8_0 handles 1 TILE_K per blck_size
            // Q4_K, Q5_K, Q6_K, IQ4_XS handles 8 TILE_K per blck_size
            GGML_ASSERT(TILE_K == blck_سائيز || TILE_K * 8 == blck_سائيز);

            parallel_for_ggml(params, n_batch * M, [&](صحيح begin, صحيح end) {
                لكل (صحيح idx = begin; idx < end; ++idx) {
                    صحيح batch_idx = idx / M;
                    صحيح m         = idx % M;
                    int64_t src1_وٿي = ggml_batch_وٿي(src1, batch_idx, ne2);
                    ثابت عائم * A_معلومات = (ثابت عائم *)((ثابت حرف *)src1->معلومات + src1_وٿي);
                    حرف * wdata_batch = (حرف *)wdata + batch_idx * M * صف_سائيز_A;
                    from_float<vec_dot_type>(A_معلومات + m * K, wdata_batch + m * صف_سائيز_A, K);
                }
            });
        });
 // }

    ggml_barrier(params->threadpool);

    إذا (M == 1) {
        // MB = 1 and handle 8 tiles in each block
        constexpr صحيح kTilesN = 4;
        constexpr صحيح بلاڪ_N = TILE_N * kTilesN;
        ثابت صحيح NB = div_مٿي(N, بلاڪ_N);

        parallel_for_ggml(params, n_batch * NB, [&](صحيح begin, صحيح end) {
            GGML_DISPATCH_QTYPES(TYPE, [&] {
                ثابت صحيح KB = K / blck_سائيز;
                ثابت صحيح TILE_سائيز = get_tile_سائيز<type>();
                ثابت صحيح صف_سائيز_A = KB * حجم(vec_dot_type);
                لكل (صحيح i = begin; i < end; ++i) {
                    صحيح batch_idx = i / NB;
                    صحيح nb = i % NB;

                    int64_t src0_وٿي = ggml_batch_وٿي(src0, batch_idx, ne2);
                    int64_t dst_وٿي  = ggml_batch_وٿي(dst,  batch_idx, ne2);
                    ثابت حرف * wdata_batch = (ثابت حرف *)wdata + batch_idx * صف_سائيز_A;

                    صحيح nb_شروعات = nb * بلاڪ_N;
                    صحيح nb_سائيز = std::min(بلاڪ_N, N - nb_شروعات); // 32, 64, 96

                    بدل (nb_سائيز) {
                        //case 160: LAUNCH_TINYGEMM_KERNEL_VNNI(160); break;
                        حالة 128: LAUNCH_TINYGEMM_KERNEL_VNNI(128); اكسر;
                        حالة 96: LAUNCH_TINYGEMM_KERNEL_VNNI(96); اكسر;
                        حالة 64: LAUNCH_TINYGEMM_KERNEL_VNNI(64); اكسر;
                        حالة 32: LAUNCH_TINYGEMM_KERNEL_VNNI(32); اكسر;
                        افتراضي: fprintf(stderr, "Unexpected n block size!\n");
                    }
                }
            });
        });
        أرجع;
    }

    // handle 4 tiles at a tile
    constexpr صحيح بلاڪ_M = TILE_M * 2;
    constexpr صحيح بلاڪ_N = TILE_N * 2;
    ثابت صحيح MB = div_مٿي(M, بلاڪ_M);
    ثابت صحيح NB = div_مٿي(N, بلاڪ_N);

    parallel_for_ggml(params, n_batch * MB * NB, [&](صحيح begin, صحيح end) {
        // init tile config for each thread
        ggml_tile_config_init();

        GGML_DISPATCH_QTYPES(TYPE, [&] {
            ثابت صحيح KB = K / blck_سائيز;
            ثابت صحيح TILE_سائيز = get_tile_سائيز<type>();
            ثابت صحيح صف_سائيز_A = KB * حجم(vec_dot_type);

            لكل (صحيح i = begin; i < end; ++i) {
                صحيح batch_idx = i / (MB * NB);
                صحيح بچيل = i % (MB * NB);
                صحيح mb = بچيل / NB;
                صحيح nb = بچيل % NB;

                int64_t src0_وٿي = ggml_batch_وٿي(src0, batch_idx, ne2);
                int64_t dst_وٿي  = ggml_batch_وٿي(dst,  batch_idx, ne2);
                ثابت حرف * wdata_batch = (ثابت حرف *)wdata + batch_idx * M * صف_سائيز_A;

                صحيح mb_شروعات = mb * بلاڪ_M;
                صحيح mb_سائيز = std::min(بلاڪ_M, M - mb_شروعات);
                صحيح nb_شروعات = nb * بلاڪ_N;
                صحيح nb_سائيز = بلاڪ_N;

                tinygemm_kernel_amx<vec_dot_type, type, عائم, blck_سائيز>(
                    mb_سائيز, nb_سائيز, KB,
                    wdata_batch + mb_شروعات * صف_سائيز_A,
                    (ثابت حرف *)src0->معلومات + src0_وٿي + PACKED_اشاريو(nb * 2, 0, KB, TILE_سائيز),
                    (عائم *) dst->معلومات + dst_وٿي + mb_شروعات * N + nb_شروعات, ldc);
            }
        });
    });
}

#جيڪڏهن_پورو // if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
