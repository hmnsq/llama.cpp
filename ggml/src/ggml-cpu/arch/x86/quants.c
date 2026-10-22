#تعریف_کرو GGML_COMMON_IMPL_C
#شامل "ggml-common.h"
#شامل "ggml-quants.h"
#شامل "ggml-impl.h"
#شامل "ggml-cpu.h"
#شامل "simd-mappings.h"

#شامل "../../quants.h"
#شامل "../../ggml-cpu-impl.h"

#شامل <math.h>
#شامل <string.h>
#شامل <assert.h>
#شامل <stdlib.h> // for qsort
#شامل <stdio.h>  // for GGML_ASSERT

#تعریف_کرو گروہ_MAX_EPS 1e-15f
#تعریف_کرو گروہ_MAX_EPS_IQ3_XXS 1e-8f
#تعریف_کرو گروہ_MAX_EPS_IQ2_S 1e-8f
#تعریف_کرو گروہ_MAX_EPS_IQ1_M 1e-7f
#تعریف_کرو گروہ_MAX_EPS_IQ1_S 1e-12f

#تعریف_کرو غیر_مستعمل GGML_غیر_مستعمل

// some compilers don't provide _mm256_set_m128i, e.g. gcc 7
#تعریف_کرو MM256_سیٹ_M128_I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

#اگر defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)
// multiply int8_t, add results pairwise twice
ساکن اندرونی __m128i mul_مجموع_i8_جوڑے(ثابت __m128i x, ثابت __m128i y) {
    // Get absolute values of x vectors
    ثابت __m128i ax = _mm_sign_epi8(x, x);
    // Sign the values of the y vectors
    ثابت __m128i sy = _mm_sign_epi8(y, x);
    // Perform multiplication and create 16-bit values
    ثابت __m128i dot = _mm_maddubs_epi16(ax, sy);
    ثابت __m128i ones = _mm_set1_epi16(1);
    واپس _mm_madd_epi16(ones, dot);
}

#اگر __AVX__ || __AVX2__ || __AVX512F__
// horizontally add 8 floats
ساکن اندرونی اعشاری hsum_float_8(ثابت __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = mm_جوڑو_ps(res, _mm256_castps256_ps128(x));
    res = mm_جوڑو_ps(res, _mm_movehl_ps(res, res));
    res = mm_جوڑو_ss(res, _mm_movehdup_ps(res));
    واپس _mm_cvtss_f32(res);
}

// horizontally add 8 int32_t
ساکن اندرونی صحیح hsum_i32_8(ثابت __m256i a) {
    ثابت __m128i sum128 = mm_جوڑو_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
    ثابت __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    ثابت __m128i sum64 = mm_جوڑو_epi32(hi64, sum128);
    ثابت __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    واپس _mm_cvtsi128_si32(mm_جوڑو_epi32(sum64, hi32));
}

// horizontally add 4 int32_t
ساکن اندرونی صحیح hsum_i32_4(ثابت __m128i a) {
    ثابت __m128i hi64 = _mm_unpackhi_epi64(a, a);
    ثابت __m128i sum64 = mm_جوڑو_epi32(hi64, a);
    ثابت __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    واپس _mm_cvtsi128_si32(mm_جوڑو_epi32(sum64, hi32));
}

#اگر defined(__AVX2__) || defined(__AVX512F__)
ساکن اندرونی __m256i mul_جوڑو_epi8(ثابت __m256i x, ثابت __m256i y) {
    ثابت __m256i ax = _mm256_sign_epi8(x, x);
    ثابت __m256i sy = _mm256_sign_epi8(y, x);
    واپس _mm256_maddubs_epi16(ax, sy);
}

// spread 32 bits to 32 bytes { 0x00, 0xFF }
ساکن اندرونی __m256i بائٹیں_from_بٹ_32(ثابت uint8_t * x) {
    uint32_t x32;
    حافظہ_نقل(&x32, x, حجم(uint32_t));
    ثابت __m256i shuf_نقاب = mm256_سیٹ_epi64x(
            0x0303030303030303, 0x0202020202020202,
            0x0101010101010101, 0x0000000000000000);
    __m256i بائٹیں = _mm256_shuffle_epi8(_mm256_set1_epi32(x32), shuf_نقاب);
    ثابت __m256i ثنائی_ہندسہ_نقاب = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe);
    بائٹیں = _mm256_or_si256(بائٹیں, ثنائی_ہندسہ_نقاب);
    واپس _mm256_cmpeq_epi8(بائٹیں, _mm256_set1_epi64x(-1));
}

// Unpack 32 4-bit fields into 32 bytes
// The output vector contains 32 bytes, each one in [ 0 .. 15 ] interval
ساکن اندرونی __m256i بائٹیں_from_nibbles_32(ثابت uint8_t * rsi)
{
    ثابت __m128i tmp = _mm_loadu_si128((ثابت __m128i *)rsi);
    ثابت __m256i بائٹیں = MM256_سیٹ_M128_I(_mm_srli_epi16(tmp, 4), tmp);
    ثابت __m256i کم_نقاب = _mm256_set1_epi8( 0xF );
    واپس _mm256_and_si256(کم_نقاب, بائٹیں);
}

// add int16_t pairwise and return as float vector
ساکن اندرونی __m256 مجموع_i16_جوڑے_float(ثابت __m256i x) {
    ثابت __m256i ones = _mm256_set1_epi16(1);
    ثابت __m256i summed_جوڑے = _mm256_madd_epi16(ones, x);
    واپس _mm256_cvtepi32_ps(summed_جوڑے);
}

ساکن اندرونی __m256 mul_مجموع_us8_جوڑے_float(ثابت __m256i ax, ثابت __m256i sy) {
#اگر defined(__AVX512VNNI__) && defined(__AVX512VL__)
    ثابت __m256i zero = _mm256_setzero_si256();
    ثابت __m256i summed_جوڑے = _mm256_dpbusd_epi32(zero, ax, sy);
    واپس _mm256_cvtepi32_ps(summed_جوڑے);
#وگرنہ_اگر defined(__AVXVNNI__)
    ثابت __m256i zero = _mm256_setzero_si256();
    ثابت __m256i summed_جوڑے = _mm256_dpbusd_avx_epi32(zero, ax, sy);
    واپس _mm256_cvtepi32_ps(summed_جوڑے);
#ورنہ
    // Perform multiplication and create 16-bit values
    ثابت __m256i dot = _mm256_maddubs_epi16(ax, sy);
    واپس مجموع_i16_جوڑے_float(dot);
#اگر_ختم
}

// multiply int8_t, add results pairwise twice and return as float vector
ساکن اندرونی __m256 mul_مجموع_i8_جوڑے_float(ثابت __m256i x, ثابت __m256i y) {
#اگر __AVXVNNIINT8__
    ثابت __m256i zero = _mm256_setzero_si256();
    ثابت __m256i summed_جوڑے = _mm256_dpbssd_epi32(zero, x, y);
    واپس _mm256_cvtepi32_ps(summed_جوڑے);
#ورنہ
    // Get absolute values of x vectors
    ثابت __m256i ax = _mm256_sign_epi8(x, x);
    // Sign the values of the y vectors
    ثابت __m256i sy = _mm256_sign_epi8(y, x);
    واپس mul_مجموع_us8_جوڑے_float(ax, sy);
#اگر_ختم
}

ساکن اندرونی __m128i packNibbles( __m256i بائٹیں )
{
    // Move bits within 16-bit lanes from 0000_abcd_0000_efgh into 0000_0000_abcd_efgh
#اگر __AVX512F__
    ثابت __m256i بائٹیں_srli_4 = _mm256_srli_epi16(بائٹیں, 4);   // 0000_0000_abcd_0000
    بائٹیں = _mm256_or_si256(بائٹیں, بائٹیں_srli_4);               // 0000_abcd_abcd_efgh
    واپس _mm256_cvtepi16_epi8(بائٹیں);                         // abcd_efgh
#ورنہ
    ثابت __m256i کم_Byte = _mm256_set1_epi16( 0xFF );
    __m256i بلند = _mm256_andnot_si256( کم_Byte, بائٹیں );
    __m256i کم = _mm256_and_si256( کم_Byte, بائٹیں );
    بلند = _mm256_srli_epi16( بلند, 4 );
    بائٹیں = _mm256_or_si256( کم, بلند );

    // Compress uint16_t lanes into bytes
    __m128i r0 = _mm256_castsi256_si128( بائٹیں );
    __m128i r1 = _mm256_extracti128_si256( بائٹیں, 1 );
    واپس _mm_packus_epi16( r0, r1 );
#اگر_ختم
}
#وگرنہ_اگر defined(__AVX__)
ساکن اندرونی __m128i packNibbles( __m128i bytes1, __m128i bytes2 )
{
    // Move bits within 16-bit lanes from 0000_abcd_0000_efgh into 0000_0000_abcd_efgh
    ثابت __m128i کم_Byte = _mm_set1_epi16( 0xFF );
    __m128i بلند = _mm_andnot_si128( کم_Byte, bytes1 );
    __m128i کم = _mm_and_si128( کم_Byte, bytes1 );
    بلند = _mm_srli_epi16( بلند, 4 );
    bytes1 = _mm_or_si128( کم, بلند );
    بلند = _mm_andnot_si128( کم_Byte, bytes2 );
    کم = _mm_and_si128( کم_Byte, bytes2 );
    بلند = _mm_srli_epi16( بلند, 4 );
    bytes2 = _mm_or_si128( کم, بلند );

    واپس _mm_packus_epi16( bytes1, bytes2);
}

ساکن اندرونی __m128i mul_جوڑو_epi8_sse(ثابت __m128i x, ثابت __m128i y) {
    ثابت __m128i ax = _mm_sign_epi8(x, x);
    ثابت __m128i sy = _mm_sign_epi8(y, x);
    واپس _mm_maddubs_epi16(ax, sy);
}

// spread 32 bits to 32 bytes { 0x00, 0xFF }
ساکن اندرونی __m256i بائٹیں_from_بٹ_32(ثابت uint8_t * x) {
    uint32_t x32;
    حافظہ_نقل(&x32, x, حجم(uint32_t));
    ثابت __m128i shuf_maskl = mm_سیٹ_epi64x(0x0101010101010101, 0x0000000000000000);
    ثابت __m128i shuf_maskh = mm_سیٹ_epi64x(0x0303030303030303, 0x0202020202020202);
    __m128i bytesl = _mm_shuffle_epi8(_mm_set1_epi32(x32), shuf_maskl);
    __m128i bytesh = _mm_shuffle_epi8(_mm_set1_epi32(x32), shuf_maskh);
    ثابت __m128i ثنائی_ہندسہ_نقاب = _mm_set1_epi64x(0x7fbfdfeff7fbfdfe);
    bytesl = _mm_or_si128(bytesl, ثنائی_ہندسہ_نقاب);
    bytesh = _mm_or_si128(bytesh, ثنائی_ہندسہ_نقاب);
    bytesl = _mm_cmpeq_epi8(bytesl, _mm_set1_epi64x(-1));
    bytesh = _mm_cmpeq_epi8(bytesh, _mm_set1_epi64x(-1));
    واپس MM256_سیٹ_M128_I(bytesh, bytesl);
}

// Unpack 32 4-bit fields into 32 bytes
// The output vector contains 32 bytes, each one in [ 0 .. 15 ] interval
ساکن اندرونی __m256i بائٹیں_from_nibbles_32(ثابت uint8_t * rsi)
{
    // Load 16 bytes from memory
    __m128i tmpl = _mm_loadu_si128((ثابت __m128i *)rsi);
    __m128i tmph = _mm_srli_epi16(tmpl, 4);
    ثابت __m128i کم_نقاب = _mm_set1_epi8(0xF);
    tmpl = _mm_and_si128(کم_نقاب, tmpl);
    tmph = _mm_and_si128(کم_نقاب, tmph);
    واپس MM256_سیٹ_M128_I(tmph, tmpl);
}

// add int16_t pairwise and return as float vector
ساکن اندرونی __m256 مجموع_i16_جوڑے_float(ثابت __m128i xh, ثابت __m128i xl) {
    ثابت __m128i ones = _mm_set1_epi16(1);
    ثابت __m128i summed_pairsl = _mm_madd_epi16(ones, xl);
    ثابت __m128i summed_pairsh = _mm_madd_epi16(ones, xh);
    ثابت __m256i summed_جوڑے = MM256_سیٹ_M128_I(summed_pairsh, summed_pairsl);
    واپس _mm256_cvtepi32_ps(summed_جوڑے);
}

ساکن اندرونی __m256 mul_مجموع_us8_جوڑے_float(ثابت __m256i ax, ثابت __m256i sy) {
    ثابت __m128i axl = _mm256_castsi256_si128(ax);
    ثابت __m128i axh = _mm256_extractf128_si256(ax, 1);
    ثابت __m128i syl = _mm256_castsi256_si128(sy);
    ثابت __m128i syh = _mm256_extractf128_si256(sy, 1);
    // Perform multiplication and create 16-bit values
    ثابت __m128i dotl = _mm_maddubs_epi16(axl, syl);
    ثابت __m128i doth = _mm_maddubs_epi16(axh, syh);
    واپس مجموع_i16_جوڑے_float(doth, dotl);
}

// multiply int8_t, add results pairwise twice and return as float vector
ساکن اندرونی __m256 mul_مجموع_i8_جوڑے_float(ثابت __m256i x, ثابت __m256i y) {
    ثابت __m128i xl = _mm256_castsi256_si128(x);
    ثابت __m128i xh = _mm256_extractf128_si256(x, 1);
    ثابت __m128i yl = _mm256_castsi256_si128(y);
    ثابت __m128i yh = _mm256_extractf128_si256(y, 1);
    // Get absolute values of x vectors
    ثابت __m128i axl = _mm_sign_epi8(xl, xl);
    ثابت __m128i axh = _mm_sign_epi8(xh, xh);
    // Sign the values of the y vectors
    ثابت __m128i syl = _mm_sign_epi8(yl, xl);
    ثابت __m128i syh = _mm_sign_epi8(yh, xh);
    // Perform multiplication and create 16-bit values
    ثابت __m128i dotl = _mm_maddubs_epi16(axl, syl);
    ثابت __m128i doth = _mm_maddubs_epi16(axh, syh);
    واپس مجموع_i16_جوڑے_float(doth, dotl);
}

// larger version of mul_sum_i8_pairs_float where x and y are each represented by four 128-bit vectors
ساکن اندرونی __m256 mul_مجموع_i8_quad_float(ثابت __m128i x_1_0, ثابت __m128i x_1_1, ثابت __m128i x_2_0, ثابت __m128i x_2_1,
                                           ثابت __m128i y_1_0, ثابت __m128i y_1_1, ثابت __m128i y_2_0, ثابت __m128i y_2_1) {
    ثابت __m128i mone = _mm_set1_epi16(1);

    ثابت __m128i p16_1_0 = mul_جوڑو_epi8_sse(x_1_0, y_1_0);
    ثابت __m128i p16_1_1 = mul_جوڑو_epi8_sse(x_1_1, y_1_1);
    ثابت __m128i p16_2_0 = mul_جوڑو_epi8_sse(x_2_0, y_2_0);
    ثابت __m128i p16_2_1 = mul_جوڑو_epi8_sse(x_2_1, y_2_1);
    ثابت __m128i p_1_0 = _mm_madd_epi16(p16_1_0, mone);
    ثابت __m128i p_1_1 = _mm_madd_epi16(p16_1_1, mone);
    ثابت __m128i p_2_0 = _mm_madd_epi16(p16_2_0, mone);
    ثابت __m128i p_2_1 = _mm_madd_epi16(p16_2_1, mone);
    ثابت __m128i p_1 = mm_جوڑو_epi32(p_1_0, p_1_1);
    ثابت __m128i p_2 = mm_جوڑو_epi32(p_2_0, p_2_1);
    واپس _mm256_cvtepi32_ps(MM256_سیٹ_M128_I(p_2, p_1));
}

// quad fp16 delta calculation
ساکن اندرونی __m256 quad_fp16_delta_float(ثابت اعشاری x0, ثابت اعشاری y0, ثابت اعشاری x1, ثابت اعشاری y1) {
    // GGML_CPU_FP16_TO_FP32 is faster than Intel F16C
    واپس mm256_سیٹ_m128(_mm_set1_ps(GGML_CPU_FP16_مقصد_FP32(x1) * GGML_CPU_FP16_مقصد_FP32(y1)),
                           _mm_set1_ps(GGML_CPU_FP16_مقصد_FP32(x0) * GGML_CPU_FP16_مقصد_FP32(y0)));
}

ساکن اندرونی __m256 quad_mx_delta_float(ثابت uint8_t x0, ثابت اعشاری y0, ثابت uint8_t x1, ثابت اعشاری y1) {
    واپس mm256_سیٹ_m128(_mm_set1_ps(GGML_CPU_E8_M0_مقصد_FP32_HALF(x1) * GGML_CPU_FP16_مقصد_FP32(y1)),
                           _mm_set1_ps(GGML_CPU_E8_M0_مقصد_FP32_HALF(x0) * GGML_CPU_FP16_مقصد_FP32(y0)));
}
#اگر_ختم
#وگرنہ_اگر defined(__SSSE3__)
ساکن اندرونی __m128i بائٹیں_from_بٹ_16(ثابت uint8_t * x) {
    uint16_t x16;
    حافظہ_نقل(&x16, x, حجم(uint16_t));

    ثابت __m128i shuf_نقاب = mm_سیٹ_epi64x(0x0101010101010101, 0x0000000000000000);
    __m128i بائٹیں = _mm_shuffle_epi8(_mm_set1_epi16((مختصر) x16), shuf_نقاب);
    ثابت __m128i ثنائی_ہندسہ_نقاب = mm_سیٹ_epi64x(0x7fbfdfeff7fbfdfe, 0x7fbfdfeff7fbfdfe);
    بائٹیں = _mm_or_si128(بائٹیں, ثنائی_ہندسہ_نقاب);

    واپس _mm_cmpeq_epi8(بائٹیں, _mm_set1_epi64x(-1));
}

// horizontally add 4x4 floats
ساکن اندرونی اعشاری hsum_float_4x4(ثابت __m128 a, ثابت __m128 b, ثابت __m128 c, ثابت __m128 d) {
    __m128 res_0 =_mm_hadd_ps(a, b);
    __m128 res_1 =_mm_hadd_ps(c, d);
    __m128 res =_mm_hadd_ps(res_0, res_1);
    res =_mm_hadd_ps(res, res);
    res =_mm_hadd_ps(res, res);

    واپس _mm_cvtss_f32(res);
}
#اگر_ختم // __AVX__ || __AVX2__ || __AVX512F__
#اگر_ختم // defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)

خالی quantize_صف_q8_0(ثابت اعشاری * GGML_RESTRICT x, خالی * GGML_RESTRICT vy, int64_t k) {
    أكد(QK8_0 == 32);
    أكد(k % QK8_0 == 0);
    ثابت صحیح nb = k / QK8_0;

    بلاک_q8_0 * GGML_RESTRICT y = vy;

#اگر defined(__AVX2__) || defined(__AVX__)
    برائے (صحیح i = 0; i < nb; i++) {
        // Load elements into 4 AVX vectors
        __m256 v0 = _mm256_loadu_ps( x );
        __m256 v1 = _mm256_loadu_ps( x + 8 );
        __m256 v2 = _mm256_loadu_ps( x + 16 );
        __m256 v3 = _mm256_loadu_ps( x + 24 );
        x += 32;

        // Compute max(abs(e)) for the block
        ثابت __m256 sign_ثنائی_ہندسہ = _mm256_set1_ps( -0.0f );
        __m256 maxAbs = _mm256_andnot_ps( sign_ثنائی_ہندسہ, v0 );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( sign_ثنائی_ہندسہ, v1 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( sign_ثنائی_ہندسہ, v2 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( sign_ثنائی_ہندسہ, v3 ) );

        __m128 max4 = _mm_max_ps( _mm256_extractf128_ps( maxAbs, 1 ), _mm256_castps256_ps128( maxAbs ) );
        max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
        max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
        ثابت اعشاری maxScalar = _mm_cvtss_f32( max4 );

        // Quantize these floats
        ثابت اعشاری d = maxScalar / 127.f;
        y[i].d = GGML_CPU_FP32_مقصد_FP16(d);
        ثابت اعشاری id = ( maxScalar != 0.0f ) ? 127.f / maxScalar : 0.0f;
        ثابت __m256 mul = _mm256_set1_ps( id );

        // Apply the multiplier
        v0 = _mm256_mul_ps( v0, mul );
        v1 = _mm256_mul_ps( v1, mul );
        v2 = _mm256_mul_ps( v2, mul );
        v3 = _mm256_mul_ps( v3, mul );

        // Round to nearest integer
        v0 = _mm256_round_ps( v0, _MM_ROUND_NEAREST );
        v1 = _mm256_round_ps( v1, _MM_ROUND_NEAREST );
        v2 = _mm256_round_ps( v2, _MM_ROUND_NEAREST );
        v3 = _mm256_round_ps( v3, _MM_ROUND_NEAREST );

        // Convert floats to integers
        __m256i i0 = _mm256_cvtps_epi32( v0 );
        __m256i i1 = _mm256_cvtps_epi32( v1 );
        __m256i i2 = _mm256_cvtps_epi32( v2 );
        __m256i i3 = _mm256_cvtps_epi32( v3 );

#اگر defined(__AVX2__)
        // Convert int32 to int16
        i0 = _mm256_packs_epi32( i0, i1 );	// 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 12, 13, 14, 15
        i2 = _mm256_packs_epi32( i2, i3 );	// 16, 17, 18, 19,  24, 25, 26, 27,  20, 21, 22, 23, 28, 29, 30, 31
                                            // Convert int16 to int8
        i0 = _mm256_packs_epi16( i0, i2 );	// 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24, 25, 26, 27,  4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31

        // We got our precious signed bytes, but the order is now wrong
        // These AVX2 pack instructions process 16-byte pieces independently
        // The following instruction is fixing the order
        ثابت __m256i perm = _mm256_setr_epi32( 0, 4, 1, 5, 2, 6, 3, 7 );
        i0 = _mm256_permutevar8x32_epi32( i0, perm );

        _mm256_storeu_si256((__m256i *)y[i].qs, i0);
#ورنہ
        // Since we don't have in AVX some necessary functions,
        // we split the registers in half and call AVX2 analogs from SSE
        __m128i ni0 = _mm256_castsi256_si128( i0 );
        __m128i ni1 = _mm256_extractf128_si256( i0, 1);
        __m128i ni2 = _mm256_castsi256_si128( i1 );
        __m128i ni3 = _mm256_extractf128_si256( i1, 1);
        __m128i ni4 = _mm256_castsi256_si128( i2 );
        __m128i ni5 = _mm256_extractf128_si256( i2, 1);
        __m128i ni6 = _mm256_castsi256_si128( i3 );
        __m128i ni7 = _mm256_extractf128_si256( i3, 1);

        // Convert int32 to int16
        ni0 = _mm_packs_epi32( ni0, ni1 );
        ni2 = _mm_packs_epi32( ni2, ni3 );
        ni4 = _mm_packs_epi32( ni4, ni5 );
        ni6 = _mm_packs_epi32( ni6, ni7 );
        // Convert int16 to int8
        ni0 = _mm_packs_epi16( ni0, ni2 );
        ni4 = _mm_packs_epi16( ni4, ni6 );

        _mm_storeu_si128((__m128i *)(y[i].qs +  0), ni0);
        _mm_storeu_si128((__m128i *)(y[i].qs + 16), ni4);
#اگر_ختم
    }
#ورنہ
    GGML_غیر_مستعمل(nb);
    // scalar
    quantize_صف_q8_0_ref(x, y, k);
#اگر_ختم
}

خالی quantize_صف_q8_1(ثابت اعشاری * GGML_RESTRICT x, خالی * GGML_RESTRICT vy, int64_t k) {
    أكد(k % QK8_1 == 0);
    ثابت صحیح nb = k / QK8_1;

    بلاک_q8_1 * GGML_RESTRICT y = vy;
#اگر defined(__AVX2__) || defined(__AVX__)
    برائے (صحیح i = 0; i < nb; i++) {
        // Load elements into 4 AVX vectors
        __m256 v0 = _mm256_loadu_ps( x );
        __m256 v1 = _mm256_loadu_ps( x + 8 );
        __m256 v2 = _mm256_loadu_ps( x + 16 );
        __m256 v3 = _mm256_loadu_ps( x + 24 );
        x += 32;

        // Compute max(abs(e)) for the block
        ثابت __m256 sign_ثنائی_ہندسہ = _mm256_set1_ps( -0.0f );
        __m256 maxAbs = _mm256_andnot_ps( sign_ثنائی_ہندسہ, v0 );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( sign_ثنائی_ہندسہ, v1 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( sign_ثنائی_ہندسہ, v2 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( sign_ثنائی_ہندسہ, v3 ) );

        __m128 max4 = _mm_max_ps( _mm256_extractf128_ps( maxAbs, 1 ), _mm256_castps256_ps128( maxAbs ) );
        max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
        max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
        ثابت اعشاری max_scalar = _mm_cvtss_f32( max4 );

        // Quantize these floats
        ثابت اعشاری d = max_scalar / 127.f;
        y[i].d = GGML_CPU_FP32_مقصد_FP16(d);
        ثابت اعشاری id = ( max_scalar != 0.0f ) ? 127.f / max_scalar : 0.0f;
        ثابت __m256 mul = _mm256_set1_ps( id );

        // Apply the multiplier
        v0 = _mm256_mul_ps( v0, mul );
        v1 = _mm256_mul_ps( v1, mul );
        v2 = _mm256_mul_ps( v2, mul );
        v3 = _mm256_mul_ps( v3, mul );

        // Round to nearest integer
        v0 = _mm256_round_ps( v0, _MM_ROUND_NEAREST );
        v1 = _mm256_round_ps( v1, _MM_ROUND_NEAREST );
        v2 = _mm256_round_ps( v2, _MM_ROUND_NEAREST );
        v3 = _mm256_round_ps( v3, _MM_ROUND_NEAREST );

        // Convert floats to integers
        __m256i i0 = _mm256_cvtps_epi32( v0 );
        __m256i i1 = _mm256_cvtps_epi32( v1 );
        __m256i i2 = _mm256_cvtps_epi32( v2 );
        __m256i i3 = _mm256_cvtps_epi32( v3 );

#اگر defined(__AVX2__)
        // Compute the sum of the quants and set y[i].s
        y[i].s = GGML_CPU_FP32_مقصد_FP16(d * hsum_i32_8(mm256_جوڑو_epi32(mm256_جوڑو_epi32(i0, i1), mm256_جوڑو_epi32(i2, i3))));

        // Convert int32 to int16
        i0 = _mm256_packs_epi32( i0, i1 );	// 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 12, 13, 14, 15
        i2 = _mm256_packs_epi32( i2, i3 );	// 16, 17, 18, 19,  24, 25, 26, 27,  20, 21, 22, 23, 28, 29, 30, 31
                                            // Convert int16 to int8
        i0 = _mm256_packs_epi16( i0, i2 );	// 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24, 25, 26, 27,  4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31

        // We got our precious signed bytes, but the order is now wrong
        // These AVX2 pack instructions process 16-byte pieces independently
        // The following instruction is fixing the order
        ثابت __m256i perm = _mm256_setr_epi32( 0, 4, 1, 5, 2, 6, 3, 7 );
        i0 = _mm256_permutevar8x32_epi32( i0, perm );

        _mm256_storeu_si256((__m256i *)y[i].qs, i0);
#ورنہ
        // Since we don't have in AVX some necessary functions,
        // we split the registers in half and call AVX2 analogs from SSE
        __m128i ni0 = _mm256_castsi256_si128( i0 );
        __m128i ni1 = _mm256_extractf128_si256( i0, 1);
        __m128i ni2 = _mm256_castsi256_si128( i1 );
        __m128i ni3 = _mm256_extractf128_si256( i1, 1);
        __m128i ni4 = _mm256_castsi256_si128( i2 );
        __m128i ni5 = _mm256_extractf128_si256( i2, 1);
        __m128i ni6 = _mm256_castsi256_si128( i3 );
        __m128i ni7 = _mm256_extractf128_si256( i3, 1);

        // Compute the sum of the quants and set y[i].s
        ثابت __m128i s0 = mm_جوڑو_epi32(mm_جوڑو_epi32(ni0, ni1), mm_جوڑو_epi32(ni2, ni3));
        ثابت __m128i s1 = mm_جوڑو_epi32(mm_جوڑو_epi32(ni4, ni5), mm_جوڑو_epi32(ni6, ni7));
        y[i].s = GGML_CPU_FP32_مقصد_FP16(d * hsum_i32_4(mm_جوڑو_epi32(s0, s1)));

        // Convert int32 to int16
        ni0 = _mm_packs_epi32( ni0, ni1 );
        ni2 = _mm_packs_epi32( ni2, ni3 );
        ni4 = _mm_packs_epi32( ni4, ni5 );
        ni6 = _mm_packs_epi32( ni6, ni7 );
        // Convert int16 to int8
        ni0 = _mm_packs_epi16( ni0, ni2 );
        ni4 = _mm_packs_epi16( ni4, ni6 );

        _mm_storeu_si128((__m128i *)(y[i].qs +  0), ni0);
        _mm_storeu_si128((__m128i *)(y[i].qs + 16), ni4);
#اگر_ختم
    }
#ورنہ
    GGML_غیر_مستعمل(nb);
    // scalar
    quantize_صف_q8_1_ref(x, y, k);
#اگر_ختم
}

// placeholder implementation for Apple targets
خالی quantize_صف_q8_K(ثابت اعشاری * GGML_RESTRICT x, خالی * GGML_RESTRICT y, int64_t k) {
    quantize_صف_q8_K_ref(x, y, k);
}

//===================================== Dot products =================================

//
// Helper functions
//

#اگر __AVX__ || __AVX2__ || __AVX512F__

// shuffles to pick the required scales in dot products
ساکن اندرونی __m256i get_scale_shuffle_q3k(صحیح i) {
    ساکن ثابت uint8_t k_shuffle[128] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,     2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,     6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,    10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,    14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
    };
    واپس _mm256_loadu_si256((ثابت __m256i*)k_shuffle + i);
}
ساکن اندرونی __m256i get_scale_shuffle_k4(صحیح i) {
    ساکن ثابت uint8_t k_shuffle[256] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
         2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
         6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
        14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15
    };
    واپس _mm256_loadu_si256((ثابت __m256i*)k_shuffle + i);
}
ساکن اندرونی __m128i get_scale_shuffle(صحیح i) {
    ساکن ثابت uint8_t k_shuffle[128] = {
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
         2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
         4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
         6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
         8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
        10,10,10,10,10,10,10,10, 11,11,11,11,11,11,11,11,
        12,12,12,12,12,12,12,12, 13,13,13,13,13,13,13,13,
        14,14,14,14,14,14,14,14, 15,15,15,15,15,15,15,15
    };
    واپس _mm_loadu_si128((ثابت __m128i*)k_shuffle + i);
}
#اگر_ختم

خالی ggml_vec_dot_q1_0_q8_0(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc) {
    ثابت صحیح qk = QK1_0;
    ثابت صحیح nb = n / qk;

    أكد(n % qk == 0);
    أكد(nrc == 1);
    غیر_مستعمل(nrc);
    غیر_مستعمل(bx);
    غیر_مستعمل(by);
    غیر_مستعمل(bs);

    ثابت بلاک_q1_0 * GGML_RESTRICT x = vx;
    ثابت بلاک_q8_0 * GGML_RESTRICT y = vy;

#اگر defined(__AVX2__)
    ثابت __m256i ones_8 = _mm256_set1_epi8(1);
    ثابت __m256i ones_16 = _mm256_set1_epi16(1);
    ثابت __m256i byte_shuf = _mm256_setr_epi8(
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
            2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    ثابت __m256i ثنائی_ہندسہ_masks = _mm256_setr_epi8(
            1, 2, 4, 8, 16, 32, 64, (حرف) -128, 1, 2, 4, 8, 16, 32, 64, (حرف) -128,
            1, 2, 4, 8, 16, 32, 64, (حرف) -128, 1, 2, 4, 8, 16, 32, 64, (حرف) -128);
    ثابت __m256i zero = _mm256_setzero_si256();
    __m256 acc = _mm256_setzero_ps();

    برائے (صحیح ib = 0; ib < nb; ++ib) {
        ثابت اعشاری d0 = GGML_CPU_FP16_مقصد_FP32(x[ib].d);
        ثابت uint32_t * GGML_RESTRICT qs32 = (ثابت uint32_t *) x[ib].qs;
        ثابت بلاک_q8_0 * GGML_RESTRICT y_ptr = &y[ib * 4];

        __m256 acc_بلاک;
        {
            ثابت __m256i qy = _mm256_loadu_si256((ثابت __m256i *) y_ptr[0].qs);
            ثابت __m256i sm = _mm256_cmpeq_epi8(
                    _mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((صحیح) qs32[0]), byte_shuf), ثنائی_ہندسہ_masks), zero);
            ثابت __m256i sy = _mm256_sub_epi8(_mm256_xor_si256(qy, sm), sm);
            ثابت __m256i s32 = _mm256_madd_epi16(_mm256_maddubs_epi16(ones_8, sy), ones_16);
            acc_بلاک = _mm256_mul_ps(_mm256_set1_ps(GGML_CPU_FP16_مقصد_FP32(y_ptr[0].d)), _mm256_cvtepi32_ps(s32));
        }
        برائے (صحیح K = 1; K < 4; ++K) {
            ثابت __m256i qy = _mm256_loadu_si256((ثابت __m256i *) y_ptr[K].qs);
            ثابت __m256i sm = _mm256_cmpeq_epi8(
                    _mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((صحیح) qs32[K]), byte_shuf), ثنائی_ہندسہ_masks), zero);
            ثابت __m256i sy = _mm256_sub_epi8(_mm256_xor_si256(qy, sm), sm);
            ثابت __m256i s32 = _mm256_madd_epi16(_mm256_maddubs_epi16(ones_8, sy), ones_16);
            acc_بلاک = _mm256_fmadd_ps(_mm256_set1_ps(GGML_CPU_FP16_مقصد_FP32(y_ptr[K].d)), _mm256_cvtepi32_ps(s32), acc_بلاک);
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d0), acc_بلاک, acc);
    }

    *s = hsum_float_8(acc);
#وگرنہ_اگر defined(__AVX__)
    ثابت __m128i ones_8 = _mm_set1_epi8(1);
    ثابت __m128i ones_16 = _mm_set1_epi16(1);
    ثابت __m128i zero = _mm_setzero_si128();
    __m256 acc = _mm256_setzero_ps();

    برائے (صحیح ib = 0; ib < nb; ++ib) {
        ثابت اعشاری d0 = GGML_CPU_FP16_مقصد_FP32(x[ib].d);
        ثابت بلاک_q8_0 * GGML_RESTRICT y_ptr = &y[ib * 4];
        __m256 acc_بلاک;
        {
            ثابت __m256i ثنائی_ہندسہ_نقاب = بائٹیں_from_بٹ_32(&x[ib].qs[0]);
            ثابت __m128i ثنائی_ہندسہ_نقاب_0 = _mm256_castsi256_si128(ثنائی_ہندسہ_نقاب);
            ثابت __m128i ثنائی_ہندسہ_نقاب_1 = _mm256_extractf128_si256(ثنائی_ہندسہ_نقاب, 1);
            ثابت __m128i qy_0 = _mm_loadu_si128((ثابت __m128i *) &y_ptr[0].qs[0]);
            ثابت __m128i qy_1 = _mm_loadu_si128((ثابت __m128i *) &y_ptr[0].qs[16]);
            ثابت __m128i sign_نقاب_0 = _mm_cmpeq_epi8(ثنائی_ہندسہ_نقاب_0, zero);
            ثابت __m128i sign_نقاب_1 = _mm_cmpeq_epi8(ثنائی_ہندسہ_نقاب_1, zero);
            ثابت __m128i sy_0 = _mm_sub_epi8(_mm_xor_si128(qy_0, sign_نقاب_0), sign_نقاب_0);
            ثابت __m128i sy_1 = _mm_sub_epi8(_mm_xor_si128(qy_1, sign_نقاب_1), sign_نقاب_1);
            ثابت __m128i sum16_0 = _mm_maddubs_epi16(ones_8, sy_0);
            ثابت __m128i sum16_1 = _mm_maddubs_epi16(ones_8, sy_1);
            ثابت __m128i sum32_0 = _mm_madd_epi16(sum16_0, ones_16);
            ثابت __m128i sum32_1 = _mm_madd_epi16(sum16_1, ones_16);
            ثابت __m256 q = _mm256_cvtepi32_ps(MM256_سیٹ_M128_I(sum32_1, sum32_0));
            acc_بلاک = _mm256_mul_ps(_mm256_set1_ps(GGML_CPU_FP16_مقصد_FP32(y_ptr[0].d)), q);
        }
        برائے(صحیح K = 1; K < 4; ++K) {
            ثابت __m256i ثنائی_ہندسہ_نقاب = بائٹیں_from_بٹ_32(&x[ib].qs[(K) * 4]);
            ثابت __m128i ثنائی_ہندسہ_نقاب_0 = _mm256_castsi256_si128(ثنائی_ہندسہ_نقاب);
            ثابت __m128i ثنائی_ہندسہ_نقاب_1 = _mm256_extractf128_si256(ثنائی_ہندسہ_نقاب, 1);
            ثابت __m128i qy_0 = _mm_loadu_si128((ثابت __m128i *) &y_ptr[(K)].qs[0]);
            ثابت __m128i qy_1 = _mm_loadu_si128((ثابت __m128i *) &y_ptr[(K)].qs[16]);
            ثابت __m128i sign_نقاب_0 = _mm_cmpeq_epi8(ثنائی_ہندسہ_نقاب_0, zero);
            ثابت __m128i sign_نقاب_1 = _mm_cmpeq_epi8(ثنائی_ہندسہ_نقاب_1, zero);
            ثابت __m128i sy_0 = _mm_sub_epi8(_mm_xor_si128(qy_0, sign_نقاب_0), sign_نقاب_0);
            ثابت __m128i sy_1 = _mm_sub_epi8(_mm_xor_si128(qy_1, sign_نقاب_1), sign_نقاب_1);
            ثابت __m128i sum16_0 = _mm_maddubs_epi16(ones_8, sy_0);
            ثابت __m128i sum16_1 = _mm_maddubs_epi16(ones_8, sy_1);
            ثابت __m128i sum32_0 = _mm_madd_epi16(sum16_0, ones_16);
            ثابت __m128i sum32_1 = _mm_madd_epi16(sum16_1, ones_16);
            ثابت __m256 q = _mm256_cvtepi32_ps(MM256_سیٹ_M128_I(sum32_1, sum32_0));
            acc_بلاک = mm256_جوڑو_ps(acc_بلاک, _mm256_mul_ps(_mm256_set1_ps(GGML_CPU_FP16_مقصد_FP32(y_ptr[(K)].d)), q));
        }
#تعریف_منسوخ Q1_AVX_بلاک

        acc = mm256_جوڑو_ps(acc, _mm256_mul_ps(_mm256_set1_ps(d0), acc_بلاک));
    }

    *s = hsum_float_8(acc);
#وگرنہ_اگر defined(__SSSE3__)
    ثابت __m128i ones_8 = _mm_set1_epi8(1);
    ثابت __m128i ones_16 = _mm_set1_epi16(1);
    ثابت __m128i zero = _mm_setzero_si128();
    __m128 acc_0 = _mm_setzero_ps();
    __m128 acc_1 = _mm_setzero_ps();
    __m128 acc_2 = _mm_setzero_ps();
    __m128 acc_3 = _mm_setzero_ps();

    برائے (صحیح ib = 0; ib < nb; ++ib) {
        ثابت __m128 d0 = _mm_set1_ps(GGML_CPU_FP16_مقصد_FP32(x[ib].d));
        ثابت بلاک_q8_0 * GGML_RESTRICT y_ptr = &y[ib * 4];

#تعریف_کرو Q1_SSSE3_بلاک(QS_OFF, Y_IDX, ACC) \
        { \
            ثابت __m128i ثنائی_ہندسہ_نقاب_0 = بائٹیں_from_بٹ_16(&x[ib].qs[(QS_OFF) + 0]); \
            ثابت __m128i ثنائی_ہندسہ_نقاب_1 = بائٹیں_from_بٹ_16(&x[ib].qs[(QS_OFF) + 2]); \
            ثابت __m128i qy_0 = _mm_loadu_si128((ثابت __m128i *) &y_ptr[(Y_IDX)].qs[0]); \
            ثابت __m128i qy_1 = _mm_loadu_si128((ثابت __m128i *) &y_ptr[(Y_IDX)].qs[16]); \
            ثابت __m128i sign_نقاب_0 = _mm_cmpeq_epi8(ثنائی_ہندسہ_نقاب_0, zero); \
            ثابت __m128i sign_نقاب_1 = _mm_cmpeq_epi8(ثنائی_ہندسہ_نقاب_1, zero); \
            ثابت __m128i sy_0 = _mm_sub_epi8(_mm_xor_si128(qy_0, sign_نقاب_0), sign_نقاب_0); \
            ثابت __m128i sy_1 = _mm_sub_epi8(_mm_xor_si128(qy_1, sign_نقاب_1), sign_نقاب_1); \
            ثابت __m128i مجموع_0 = _mm_madd_epi16(_mm_maddubs_epi16(ones_8, sy_0), ones_16); \
            ثابت __m128i مجموع_1 = _mm_madd_epi16(_mm_maddubs_epi16(ones_8, sy_1), ones_16); \
            ثابت __m128 q = _mm_cvtepi32_ps(mm_جوڑو_epi32(مجموع_0, مجموع_1)); \
            (ACC) = mm_جوڑو_ps((ACC), _mm_mul_ps(_mm_mul_ps(d0, _mm_set1_ps(GGML_CPU_FP16_مقصد_FP32(y_ptr[(Y_IDX)].d))), q)); \
        }
        Q1_SSSE3_بلاک(0,  0, acc_0)
        Q1_SSSE3_بلاک(4,  1, acc_1)
        Q1_SSSE3_بلاک(8,  2, acc_2)
        Q1_SSSE3_بلاک(12, 3, acc_3)
#تعریف_منسوخ Q1_SSSE3_بلاک
    }

    *s = hsum_float_4x4(acc_0, acc_1, acc_2, acc_3);
#ورنہ
    غیر_مستعمل(nb);
    غیر_مستعمل(x);
    غیر_مستعمل(y);
    ggml_vec_dot_q1_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#اگر_ختم
}

#اگر defined(__AVX2__)
// four consecutive Q2_0 weights per byte: byte b becomes b & 3, (b >> 2) & 3, (b >> 4) & 3, (b >> 6) & 3 (OR, never add: the overlapping copies only touch masked-off bits)
ساکن اندرونی __m256i q2_0_unpack_32_avx2(ثابت uint8_t * qs) {
    __m256i v = _mm256_cvtepu8_epi32(_mm_loadl_epi64((ثابت __m128i *) qs));
    v = _mm256_or_si256(v, _mm256_slli_epi32(v, 12));
    v = _mm256_or_si256(v, _mm256_slli_epi32(v,  6));
    واپس _mm256_and_si256(v, _mm256_set1_epi32(0x03030303));
}

// 32 weights against one Q8_0 block: sum((v - 1) * y) = sum(v * y) - sum(y), exact for every int8 y
ساکن اندرونی __m256 q2_0_dot_32_avx2(ثابت uint8_t * qs, ثابت int8_t * qy) {
    ثابت __m256i v = q2_0_unpack_32_avx2(qs);
    ثابت __m256i y = _mm256_loadu_si256((ثابت __m256i *) qy);
    ثابت __m256i p = _mm256_sub_epi16(_mm256_maddubs_epi16(v, y), _mm256_maddubs_epi16(_mm256_set1_epi8(1), y));
    واپس _mm256_cvtepi32_ps(_mm256_madd_epi16(p, _mm256_set1_epi16(1)));
}

ساکن اندرونی __m256 q2_0_acc_بلاک_avx2(ثابت بلاک_q2_0 * x, ثابت بلاک_q8_0 * y, __m256 acc) {
    ثابت اعشاری  dx = GGML_CPU_FP16_مقصد_FP32(x->d);
    ثابت __m256 d0 = _mm256_set1_ps(dx * GGML_CPU_FP16_مقصد_FP32(y[0].d));
    ثابت __m256 d1 = _mm256_set1_ps(dx * GGML_CPU_FP16_مقصد_FP32(y[1].d));
    ثابت __m256 p0 = q2_0_dot_32_avx2(x->qs + 0, y[0].qs);
    ثابت __m256 p1 = q2_0_dot_32_avx2(x->qs + 8, y[1].qs);
#اگر defined(__FMA__)
    acc = _mm256_fmadd_ps(d0, p0, acc);
    واپس _mm256_fmadd_ps(d1, p1, acc);
#ورنہ
    acc = mm256_جوڑو_ps(acc, _mm256_mul_ps(d0, p0));
    واپس mm256_جوڑو_ps(acc, _mm256_mul_ps(d1, p1));
#اگر_ختم
}
#اگر_ختم

خالی ggml_vec_dot_q2_0_q8_0(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc) {
#اگر defined(__AVX2__)
    أكد(n % QK2_0 == 0);
    أكد(nrc == 1);
    غیر_مستعمل(nrc);
    غیر_مستعمل(bx);
    غیر_مستعمل(by);
    غیر_مستعمل(bs);

    ثابت بلاک_q2_0 * GGML_RESTRICT x = vx;
    ثابت بلاک_q8_0 * GGML_RESTRICT y = vy;
    ثابت صحیح nb = n / QK2_0;

    // two Q2_0 blocks per step, each into its own accumulator (two FMA chains)
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    صحیح i = 0;
    برائے (; i + 1 < nb; i += 2) {
        acc0 = q2_0_acc_بلاک_avx2(x + i + 0, y + 2*i + 0, acc0);
        acc1 = q2_0_acc_بلاک_avx2(x + i + 1, y + 2*i + 2, acc1);
    }
    اگر (i < nb) {
        acc0 = q2_0_acc_بلاک_avx2(x + i, y + 2*i, acc0);
    }

    *s = hsum_float_8(mm256_جوڑو_ps(acc0, acc1));
#ورنہ
    ggml_vec_dot_q2_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#اگر_ختم
}

خالی ggml_vec_dot_q4_0_q8_0(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc) {
    ثابت صحیح qk = QK8_0;
    ثابت صحیح nb = n / qk;

    أكد(n % qk == 0);
    أكد(nrc == 1);
    غیر_مستعمل(nrc);
    غیر_مستعمل(bx);
    غیر_مستعمل(by);
    غیر_مستعمل(bs);

    ثابت بلاک_q4_0 * GGML_RESTRICT x = vx;
    ثابت بلاک_q8_0 * GGML_RESTRICT y = vy;

    صحیح ib = 0;
    اعشاری sumf = 0;

#اگر defined(__AVX2__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    // Main loop
    برائے (; ib < nb; ++ib) {
        /* Compute combined scale for the block */
        ثابت __m256 d = _mm256_set1_ps( GGML_CPU_FP16_مقصد_FP32(x[ib].d) * GGML_CPU_FP16_مقصد_FP32(y[ib].d) );

        __m256i qx = بائٹیں_from_nibbles_32(x[ib].qs);

        // Now we have a vector with bytes in [ 0 .. 15 ] interval. Offset them into [ -8 .. +7 ] interval.
        ثابت __m256i off = _mm256_set1_epi8( 8 );
        qx = _mm256_sub_epi8( qx, off );

        __m256i qy = _mm256_loadu_si256((ثابت __m256i *)y[ib].qs);

        ثابت __m256 q = mul_مجموع_i8_جوڑے_float(qx, qy);

        /* Multiply q with scale and accumulate */
        acc = _mm256_fmadd_ps( d, q, acc );
    }

    sumf = hsum_float_8(acc);
#وگرنہ_اگر defined(__AVX__)
    __m256 accum = _mm256_setzero_ps();
    برائے (; ib + 1 < nb; ib += 2) {
        ثابت __m128i q4bits_1 = _mm_loadu_si128((ثابت __m128i *)x[ib + 0].qs);
        ثابت __m128i q4bits_2 = _mm_loadu_si128((ثابت __m128i *)x[ib + 1].qs);
        ثابت __m128i q8b_1_0 = _mm_loadu_si128((ثابت __m128i *)y[ib + 0].qs);
        ثابت __m128i q8b_1_1 = _mm_loadu_si128((ثابت __m128i *)y[ib + 0].qs + 1);
        ثابت __m128i q8b_2_0 = _mm_loadu_si128((ثابت __m128i *)y[ib + 1].qs);
        ثابت __m128i q8b_2_1 = _mm_loadu_si128((ثابت __m128i *)y[ib + 1].qs + 1);

        ثابت __m128i q4b_1_0 = _mm_sub_epi8(_mm_and_si128(_mm_set1_epi8(15), q4bits_1), _mm_set1_epi8(8));
        ثابت __m128i q4b_1_1 = _mm_sub_epi8(_mm_and_si128(_mm_set1_epi8(15), _mm_srli_epi16(q4bits_1, 4)), _mm_set1_epi8(8));
        ثابت __m128i q4b_2_0 = _mm_sub_epi8(_mm_and_si128(_mm_set1_epi8(15), q4bits_2), _mm_set1_epi8(8));
        ثابت __m128i q4b_2_1 = _mm_sub_epi8(_mm_and_si128(_mm_set1_epi8(15), _mm_srli_epi16(q4bits_2, 4)), _mm_set1_epi8(8));

        ثابت __m128i p16_1_0 = mul_جوڑو_epi8_sse(q4b_1_0, q8b_1_0);
        ثابت __m128i p16_1_1 = mul_جوڑو_epi8_sse(q4b_1_1, q8b_1_1);
        ثابت __m128i p16_2_0 = mul_جوڑو_epi8_sse(q4b_2_0, q8b_2_0);
        ثابت __m128i p16_2_1 = mul_جوڑو_epi8_sse(q4b_2_1, q8b_2_1);
        ثابت __m128i p_1 = mm_جوڑو_epi16(p16_1_0, p16_1_1);
        ثابت __m128i p_2 = mm_جوڑو_epi16(p16_2_0, p16_2_1);
        ثابت __m256 p =  مجموع_i16_جوڑے_float(p_2, p_1);

        ثابت __m256 deltas = quad_fp16_delta_float(x[ib].d, y[ib].d, x[ib + 1].d, y[ib + 1].d);
        accum = mm256_جوڑو_ps(_mm256_mul_ps(deltas, p), accum);
    }

    sumf = hsum_float_8(accum);
#وگرنہ_اگر defined(__SSSE3__)
    // set constants
    ثابت __m128i کم_نقاب = _mm_set1_epi8(0xF);
    ثابت __m128i off = _mm_set1_epi8(8);

    // Initialize accumulator with zeros
    __m128 acc_0 = _mm_setzero_ps();
    __m128 acc_1 = _mm_setzero_ps();
    __m128 acc_2 = _mm_setzero_ps();
    __m128 acc_3 = _mm_setzero_ps();

    برائے (; ib + 1 < nb; ib += 2) {
        _mm_prefetch(&x[ib] + حجم(بلاک_q4_0), _MM_HINT_T0);
        _mm_prefetch(&y[ib] + حجم(بلاک_q8_0), _MM_HINT_T0);

        // Compute combined scale for the block 0 and 1
        ثابت __m128 d_0_1 = _mm_set1_ps( GGML_CPU_FP16_مقصد_FP32(x[ib].d) * GGML_CPU_FP16_مقصد_FP32(y[ib].d) );

        ثابت __m128i tmp_0_1 = _mm_loadu_si128((ثابت __m128i *)x[ib].qs);

        __m128i bx_0 = _mm_and_si128(کم_نقاب, tmp_0_1);
        __m128i by_0 = _mm_loadu_si128((ثابت __m128i *)y[ib].qs);
        bx_0 = _mm_sub_epi8(bx_0, off);
        ثابت __m128i i32_0 = mul_مجموع_i8_جوڑے(bx_0, by_0);

        __m128i bx_1 = _mm_and_si128(کم_نقاب, _mm_srli_epi64(tmp_0_1, 4));
        __m128i by_1 = _mm_loadu_si128((ثابت __m128i *)(y[ib].qs + 16));
        bx_1 = _mm_sub_epi8(bx_1, off);
        ثابت __m128i i32_1 = mul_مجموع_i8_جوڑے(bx_1, by_1);

        _mm_prefetch(&x[ib] + 2 * حجم(بلاک_q4_0), _MM_HINT_T0);
        _mm_prefetch(&y[ib] + 2 * حجم(بلاک_q8_0), _MM_HINT_T0);

        // Compute combined scale for the block 2 and 3
        ثابت __m128 d_2_3 = _mm_set1_ps( GGML_CPU_FP16_مقصد_FP32(x[ib + 1].d) * GGML_CPU_FP16_مقصد_FP32(y[ib + 1].d) );

        ثابت __m128i tmp_2_3 = _mm_loadu_si128((ثابت __m128i *)x[ib + 1].qs);

        __m128i bx_2 = _mm_and_si128(کم_نقاب, tmp_2_3);
        __m128i by_2 = _mm_loadu_si128((ثابت __m128i *)y[ib + 1].qs);
        bx_2 = _mm_sub_epi8(bx_2, off);
        ثابت __m128i i32_2 = mul_مجموع_i8_جوڑے(bx_2, by_2);

        __m128i bx_3 = _mm_and_si128(کم_نقاب, _mm_srli_epi64(tmp_2_3, 4));
        __m128i by_3 = _mm_loadu_si128((ثابت __m128i *)(y[ib + 1].qs + 16));
        bx_3 = _mm_sub_epi8(bx_3, off);
        ثابت __m128i i32_3 = mul_مجموع_i8_جوڑے(bx_3, by_3);

        // Convert int32_t to float
        __m128 p0 = _mm_cvtepi32_ps(i32_0);
        __m128 p1 = _mm_cvtepi32_ps(i32_1);
        __m128 p2 = _mm_cvtepi32_ps(i32_2);
        __m128 p3 = _mm_cvtepi32_ps(i32_3);

        // Apply the scale
        __m128 p0_d = _mm_mul_ps( d_0_1, p0 );
        __m128 p1_d = _mm_mul_ps( d_0_1, p1 );
        __m128 p2_d = _mm_mul_ps( d_2_3, p2 );
        __m128 p3_d = _mm_mul_ps( d_2_3, p3 );

        // Acummulate
        acc_0 = mm_جوڑو_ps(p0_d, acc_0);
        acc_1 = mm_جوڑو_ps(p1_d, acc_1);
        acc_2 = mm_جوڑو_ps(p2_d, acc_2);
        acc_3 = mm_جوڑو_ps(p3_d, acc_3);
    }

    sumf = hsum_float_4x4(acc_0, acc_1, acc_2, acc_3);

#اگر_ختم
    برائے (; ib < nb; ++ib) {
        صحیح sumi0 = 0;
        صحیح sumi1 = 0;

        برائے (صحیح j = 0; j < qk/2; ++j) {
            ثابت صحیح v0 = (x[ib].qs[j] & 0x0F) - 8;
            ثابت صحیح v1 = (x[ib].qs[j] >>   4) - 8;

            sumi0 += (v0 * y[ib].qs[j]);
            sumi1 += (v1 * y[ib].qs[j + qk/2]);
        }

        صحیح sumi = sumi0 + sumi1;
        sumf += sumi*GGML_CPU_FP16_مقصد_FP32(x[ib].d)*GGML_CPU_FP16_مقصد_FP32(y[ib].d);
    }

    *s = sumf;
}

خالی ggml_vec_dot_q4_1_q8_1(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc) {
    ثابت صحیح qk = QK8_1;
    ثابت صحیح nb = n / qk;

    أكد(n % qk == 0);
    أكد(nrc == 1);
    غیر_مستعمل(nrc);
    غیر_مستعمل(bx);
    غیر_مستعمل(by);
    غیر_مستعمل(bs);

    ثابت بلاک_q4_1 * GGML_RESTRICT x = vx;
    ثابت بلاک_q8_1 * GGML_RESTRICT y = vy;

    صحیح ib = 0;

#اگر defined(__AVX2__) || defined(__AVX__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    اعشاری summs = 0;

    // Main loop
    برائے (; ib < nb; ++ib) {
        ثابت اعشاری d0 = GGML_CPU_FP16_مقصد_FP32(x[ib].d);
        ثابت اعشاری d1 = GGML_CPU_FP16_مقصد_FP32(y[ib].d);

        summs += GGML_CPU_FP16_مقصد_FP32(x[ib].m) * GGML_CPU_FP16_مقصد_FP32(y[ib].s);

        ثابت __m256 d0v = _mm256_set1_ps( d0 );
        ثابت __m256 d1v = _mm256_set1_ps( d1 );

        // Compute combined scales
        ثابت __m256 d0d1 = _mm256_mul_ps( d0v, d1v );

        // Load 16 bytes, and unpack 4 bit fields into bytes, making 32 bytes
        ثابت __m256i qx = بائٹیں_from_nibbles_32(x[ib].qs);
        ثابت __m256i qy = _mm256_loadu_si256( (ثابت __m256i *)y[ib].qs );

        ثابت __m256 xy = mul_مجموع_us8_جوڑے_float(qx, qy);

        // Accumulate d0*d1*x*y
#اگر defined(__AVX2__)
        acc = _mm256_fmadd_ps( d0d1, xy, acc );
#ورنہ
        acc = mm256_جوڑو_ps( _mm256_mul_ps( d0d1, xy ), acc );
#اگر_ختم
    }

    *s = hsum_float_8(acc) + summs;
#ورنہ
    غیر_مستعمل(nb);
    غیر_مستعمل(x);
    غیر_مستعمل(y);
    غیر_مستعمل(ib);
    ggml_vec_dot_q4_1_q8_1_generic(n, s, bs, vx, bx, vy, by, nrc);
#اگر_ختم
}

خالی ggml_vec_dot_mxfp4_q8_0(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc) {
    أكد(nrc == 1);
    غیر_مستعمل(nrc);
    غیر_مستعمل(bx);
    غیر_مستعمل(by);
    غیر_مستعمل(bs);
    أكد(n % QK_MXFP4 == 0);
    static_assert(QK_MXFP4 == QK8_0, "QK_MXFP4 and QK8_0 must be the same");

    ثابت بلاک_mxfp4 * GGML_RESTRICT x = vx;
    ثابت بلاک_q8_0 * GGML_RESTRICT y = vy;

    ثابت صحیح nb = n / QK_MXFP4;

    صحیح ib = 0;
    اعشاری sumf = 0;

#اگر defined __AVX2__

    ثابت __m128i values128 = _mm_loadu_si128((ثابت __m128i*)kvalues_fp4);
    ثابت __m128i m4b  = _mm_set1_epi8(0x0f);
    ثابت __m256i mone = _mm256_set1_epi16(1);

    __m256 accum1 = _mm256_setzero_ps();
    __m256 accum2 = _mm256_setzero_ps();

    برائے (; ib + 1 < nb; ib += 2) {
        ثابت __m128i q4bits_1 = _mm_loadu_si128((ثابت __m128i*)x[ib + 0].qs);
        ثابت __m128i q4bits_2 = _mm_loadu_si128((ثابت __m128i*)x[ib + 1].qs);
        ثابت __m256i q8b_1 = _mm256_loadu_si256((ثابت __m256i *)y[ib + 0].qs);
        ثابت __m256i q8b_2 = _mm256_loadu_si256((ثابت __m256i *)y[ib + 1].qs);
        ثابت __m256i q4b_1 = MM256_سیٹ_M128_I(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b)),
                                              _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b)));
        ثابت __m256i q4b_2 = MM256_سیٹ_M128_I(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b)),
                                              _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b)));
        ثابت __m256i p16_1 = mul_جوڑو_epi8(q4b_1, q8b_1);
        ثابت __m256i p16_2 = mul_جوڑو_epi8(q4b_2, q8b_2);
        ثابت __m256i p_1 = _mm256_madd_epi16(p16_1, mone);
        ثابت __m256i p_2 = _mm256_madd_epi16(p16_2, mone);
        ثابت __m256 scale0 = _mm256_set1_ps(GGML_CPU_FP16_مقصد_FP32(y[ib + 0].d)*GGML_CPU_E8_M0_مقصد_FP32_HALF(x[ib + 0].e));
        ثابت __m256 scale1 = _mm256_set1_ps(GGML_CPU_FP16_مقصد_FP32(y[ib + 1].d)*GGML_CPU_E8_M0_مقصد_FP32_HALF(x[ib + 1].e));
        accum1 = _mm256_fmadd_ps(scale0, _mm256_cvtepi32_ps(p_1), accum1);
        accum2 = _mm256_fmadd_ps(scale1, _mm256_cvtepi32_ps(p_2), accum2);
    }

    sumf = hsum_float_8(mm256_جوڑو_ps(accum1, accum2));

#وگرنہ_اگر defined __AVX__
    ثابت __m128i values128 = _mm_loadu_si128((ثابت __m128i*)kvalues_fp4);
    ثابت __m128i m4b  = _mm_set1_epi8(0x0f);

    __m256 accum = _mm256_setzero_ps();
    برائے (; ib + 1 < nb; ib += 2) {
        ثابت __m128i q4bits_1 = _mm_loadu_si128((ثابت __m128i *)x[ib + 0].qs);
        ثابت __m128i q4bits_2 = _mm_loadu_si128((ثابت __m128i *)x[ib + 1].qs);
        ثابت __m128i q8b_1_0 = _mm_loadu_si128((ثابت __m128i *)y[ib + 0].qs);
        ثابت __m128i q8b_1_1 = _mm_loadu_si128((ثابت __m128i *)y[ib + 0].qs + 1);
        ثابت __m128i q8b_2_0 = _mm_loadu_si128((ثابت __m128i *)y[ib + 1].qs);
        ثابت __m128i q8b_2_1 = _mm_loadu_si128((ثابت __m128i *)y[ib + 1].qs + 1);

        ثابت __m128i q4b_1_0 = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b));
        ثابت __m128i q4b_1_1 = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b));
        ثابت __m128i q4b_2_0 = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b));
        ثابت __m128i q4b_2_1 = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b));

        ثابت __m256 p = mul_مجموع_i8_quad_float(q4b_1_0, q4b_1_1, q4b_2_0, q4b_2_1, q8b_1_0, q8b_1_1, q8b_2_0, q8b_2_1);
        ثابت __m256 deltas = quad_mx_delta_float(x[ib].e, y[ib].d, x[ib + 1].e, y[ib + 1].d);
        accum = mm256_جوڑو_ps(_mm256_mul_ps(deltas, p), accum);
    }

    sumf = hsum_float_8(accum);

#اگر_ختم
    برائے (; ib < nb; ++ib) {
        ثابت اعشاری d = GGML_CPU_FP16_مقصد_FP32(y[ib].d)*GGML_CPU_E8_M0_مقصد_FP32_HALF(x[ib].e);
        صحیح sumi1 = 0;
        صحیح sumi2 = 0;
        برائے (صحیح j = 0; j < QK_MXFP4/2; ++j) {
            sumi1 += y[ib].qs[j +          0] * kvalues_fp4[x[ib].qs[j] & 0xf];
            sumi2 += y[ib].qs[j + QK_MXFP4/2] * kvalues_fp4[x[ib].qs[j] >>  4];
        }
        sumf += d * (sumi1 + sumi2);
    }
    *s = sumf;
}

خالی ggml_vec_dot_nvfp4_q8_0(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc) {
    أكد(nrc == 1);
    غیر_مستعمل(nrc);
    غیر_مستعمل(bx);
    غیر_مستعمل(by);
    غیر_مستعمل(bs);
    أكد(n % QK_NVFP4 == 0);

    ثابت بلاک_nvfp4 * GGML_RESTRICT x = vx;
    ثابت بلاک_q8_0 * GGML_RESTRICT y = vy;

    ثابت صحیح nb = n / QK_NVFP4;
    صحیح ib = 0;
    اعشاری sumf = 0;

#اگر defined(__AVX2__)

    ثابت __m128i values128 = _mm_loadu_si128((ثابت __m128i*)kvalues_fp4);
    ثابت __m128i m4b  = _mm_set1_epi8(0x0f);
    ثابت __m256i mone = _mm256_set1_epi16(1);

    __m256 accum = _mm256_setzero_ps();
    برائے(; ib < nb; ib++){

        ثابت __m128i q4bits_01 = _mm_loadu_si128((ثابت __m128i *)(x[ib].qs + 0));
        ثابت __m128i q4bits_23 = _mm_loadu_si128((ثابت __m128i *)(x[ib].qs + 16));

        ثابت __m256i q8_01 = _mm256_loadu_si256((ثابت __m256i *)y[2*ib + 0].qs);
        ثابت __m256i q8_23 = _mm256_loadu_si256((ثابت __m256i *)y[2*ib + 1].qs);

        ثابت __m128i q4_01_lo = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_01, m4b));
        ثابت __m128i q4_01_hi = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_01, 4), m4b));
        ثابت __m128i q4_23_lo = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_23, m4b));
        ثابت __m128i q4_23_hi = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_23, 4), m4b));

        //reordering
        ثابت __m256i q4_01 = MM256_سیٹ_M128_I(_mm_unpackhi_epi64(q4_01_lo,q4_01_hi), _mm_unpacklo_epi64(q4_01_lo,q4_01_hi));
        ثابت __m256i q4_23 = MM256_سیٹ_M128_I(_mm_unpackhi_epi64(q4_23_lo,q4_23_hi),_mm_unpacklo_epi64(q4_23_lo,q4_23_hi));

        ثابت __m256i p01 = mul_جوڑو_epi8(q4_01,q8_01);
        ثابت __m256i p_1 = _mm256_madd_epi16(p01, mone);

        ثابت __m256i p23 = mul_جوڑو_epi8(q4_23,q8_23);
        ثابت __m256i p_2 = _mm256_madd_epi16(p23, mone);

        ثابت اعشاری dy0 = GGML_CPU_FP16_مقصد_FP32(y[2*ib].d);
        ثابت اعشاری dy1 = GGML_CPU_FP16_مقصد_FP32(y[2*ib+1].d);

        ثابت اعشاری s0 = GGML_CPU_UE4_M3_مقصد_FP32(x[ib].d[0]) * dy0;
        ثابت اعشاری s1 = GGML_CPU_UE4_M3_مقصد_FP32(x[ib].d[1]) * dy0;
        ثابت اعشاری s2 = GGML_CPU_UE4_M3_مقصد_FP32(x[ib].d[2]) * dy1;
        ثابت اعشاری s3 = GGML_CPU_UE4_M3_مقصد_FP32(x[ib].d[3]) * dy1;

        ثابت __m256 scales01 = mm256_سیٹ_m128(_mm_set1_ps(s1), _mm_set1_ps(s0));
        ثابت __m256 scales23 = mm256_سیٹ_m128(_mm_set1_ps(s3), _mm_set1_ps(s2));

        accum = _mm256_fmadd_ps(scales01, _mm256_cvtepi32_ps(p_1), accum);
        accum = _mm256_fmadd_ps(scales23, _mm256_cvtepi32_ps(p_2), accum);
    }
    sumf = hsum_float_8(accum);

#وگرنہ_اگر defined(__AVX__)

    ثابت __m128i values128 = _mm_loadu_si128((ثابت __m128i*)kvalues_fp4);
    ثابت __m128i m4b  = _mm_set1_epi8(0x0f);

    __m256 accum = _mm256_setzero_ps();
    برائے(; ib < nb; ib++){

        ثابت __m128i q4bits_01 = _mm_loadu_si128((ثابت __m128i *)(x[ib].qs + 0));
        ثابت __m128i q4bits_23 = _mm_loadu_si128((ثابت __m128i *)(x[ib].qs + 16));

        ثابت __m128i q8_0 = _mm_loadu_si128((ثابت __m128i *)(y[2*ib + 0].qs + 0));
        ثابت __m128i q8_1 = _mm_loadu_si128((ثابت __m128i *)(y[2*ib + 0].qs + 16));
        ثابت __m128i q8_2 = _mm_loadu_si128((ثابت __m128i *)(y[2*ib + 1].qs + 0));
        ثابت __m128i q8_3 = _mm_loadu_si128((ثابت __m128i *)(y[2*ib + 1].qs + 16));

        ثابت __m128i q4_01_lo = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_01, m4b));
        ثابت __m128i q4_01_hi = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_01, 4), m4b));
        ثابت __m128i q4_23_lo = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_23, m4b));
        ثابت __m128i q4_23_hi = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_23, 4), m4b));

        ثابت __m128i q4_0 = _mm_unpacklo_epi64(q4_01_lo, q4_01_hi);
        ثابت __m128i q4_1 = _mm_unpackhi_epi64(q4_01_lo, q4_01_hi);
        ثابت __m128i q4_2 = _mm_unpacklo_epi64(q4_23_lo, q4_23_hi);
        ثابت __m128i q4_3 = _mm_unpackhi_epi64(q4_23_lo, q4_23_hi);

        ثابت __m128i p0_i32 = mul_مجموع_i8_جوڑے(q4_0, q8_0);
        ثابت __m128i p1_i32 = mul_مجموع_i8_جوڑے(q4_1, q8_1);
        ثابت __m128i p2_i32 = mul_مجموع_i8_جوڑے(q4_2, q8_2);
        ثابت __m128i p3_i32 = mul_مجموع_i8_جوڑے(q4_3, q8_3);

        ثابت __m128 p0 = _mm_cvtepi32_ps(p0_i32);
        ثابت __m128 p1 = _mm_cvtepi32_ps(p1_i32);
        ثابت __m128 p2 = _mm_cvtepi32_ps(p2_i32);
        ثابت __m128 p3 = _mm_cvtepi32_ps(p3_i32);

        ثابت __m256 p01 = mm256_سیٹ_m128(p1, p0);
        ثابت __m256 p23 = mm256_سیٹ_m128(p3, p2);

        ثابت اعشاری dy0 = GGML_CPU_FP16_مقصد_FP32(y[2*ib].d);
        ثابت اعشاری dy1 = GGML_CPU_FP16_مقصد_FP32(y[2*ib+1].d);

        ثابت اعشاری s0 = GGML_CPU_UE4_M3_مقصد_FP32(x[ib].d[0]) * dy0;
        ثابت اعشاری s1 = GGML_CPU_UE4_M3_مقصد_FP32(x[ib].d[1]) * dy0;
        ثابت اعشاری s2 = GGML_CPU_UE4_M3_مقصد_FP32(x[ib].d[2]) * dy1;
        ثابت اعشاری s3 = GGML_CPU_UE4_M3_مقصد_FP32(x[ib].d[3]) * dy1;

        ثابت __m256 scales01 = mm256_سیٹ_m128(_mm_set1_ps(s1), _mm_set1_ps(s0));
        ثابت __m256 scales23 = mm256_سیٹ_m128(_mm_set1_ps(s3), _mm_set1_ps(s2));

        accum = mm256_جوڑو_ps(accum, _mm256_mul_ps(p01, scales01));
        accum = mm256_جوڑو_ps(accum, _mm256_mul_ps(p23, scales23));
    }
    sumf = hsum_float_8(accum);

#اگر_ختم

    برائے (;ib < nb; ++ib) {
        برائے (صحیح s_idx = 0; s_idx < 4; ++s_idx) {
            ثابت اعشاری d = GGML_CPU_UE4_M3_مقصد_FP32(x[ib].d[s_idx]);
            ثابت صحیح q8_بلاک = s_idx / 2;
            ثابت صحیح q8_off   = (s_idx % 2) * QK_NVFP4_SUB;
            ثابت اعشاری dy = GGML_CPU_FP16_مقصد_FP32(y[2*ib + q8_بلاک].d);

            صحیح sumi_lo = 0, sumi_hi = 0;
            برائے (صحیح j = 0; j < QK_NVFP4_SUB/2; ++j) {
                ثابت uint8_t qv = x[ib].qs[s_idx*(QK_NVFP4_SUB/2) + j];
                sumi_lo += y[2*ib + q8_بلاک].qs[q8_off + j +               0] * kvalues_fp4[qv & 0xf];
                sumi_hi += y[2*ib + q8_بلاک].qs[q8_off + j + QK_NVFP4_SUB/2] * kvalues_fp4[qv >>  4];
            }

            sumf += dy * d * (sumi_lo + sumi_hi);
        }
    }
    *s = sumf;
}

خالی ggml_vec_dot_q5_0_q8_0(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc) {
    ثابت صحیح qk = QK8_0;
    ثابت صحیح nb = n / qk;

    صحیح ib = 0;

    أكد(n % qk == 0);
    أكد(qk == QK5_0);
    أكد(nrc == 1);
    غیر_مستعمل(nrc);
    غیر_مستعمل(bx);
    غیر_مستعمل(by);
    غیر_مستعمل(bs);

    ثابت بلاک_q5_0 * GGML_RESTRICT x = vx;
    ثابت بلاک_q8_0 * GGML_RESTRICT y = vy;

#اگر defined(__AVX2__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    // Main loop
    برائے (; ib < nb; ++ib) {
        /* Compute combined scale for the block */
        ثابت __m256 d = _mm256_set1_ps(GGML_CPU_FP16_مقصد_FP32(x[ib].d) * GGML_CPU_FP16_مقصد_FP32(y[ib].d));

        __m256i qx = بائٹیں_from_nibbles_32(x[ib].qs);
        __m256i bxhi = بائٹیں_from_بٹ_32(x[ib].qh);
        bxhi = _mm256_andnot_si256(bxhi, _mm256_set1_epi8((حرف)0xF0));
        qx = _mm256_or_si256(qx, bxhi);

        __m256i qy = _mm256_loadu_si256((ثابت __m256i *)y[ib].qs);

        ثابت __m256 q = mul_مجموع_i8_جوڑے_float(qx, qy);

        /* Multiply q with scale and accumulate */
        acc = _mm256_fmadd_ps(d, q, acc);
    }

    *s = hsum_float_8(acc);
#وگرنہ_اگر defined(__AVX__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();
    __m128i نقاب = _mm_set1_epi8((حرف)0xF0);

    // Main loop
    برائے (; ib < nb; ++ib) {
        /* Compute combined scale for the block */
        ثابت __m256 d = _mm256_set1_ps(GGML_CPU_FP16_مقصد_FP32(x[ib].d) * GGML_CPU_FP16_مقصد_FP32(y[ib].d));

        __m256i bx_0 = بائٹیں_from_nibbles_32(x[ib].qs);
        ثابت __m256i bxhi = بائٹیں_from_بٹ_32(x[ib].qh);
        __m128i bxhil = _mm256_castsi256_si128(bxhi);
        __m128i bxhih = _mm256_extractf128_si256(bxhi, 1);
        bxhil = _mm_andnot_si128(bxhil, نقاب);
        bxhih = _mm_andnot_si128(bxhih, نقاب);
        __m128i bxl = _mm256_castsi256_si128(bx_0);
        __m128i bxh = _mm256_extractf128_si256(bx_0, 1);
        bxl = _mm_or_si128(bxl, bxhil);
        bxh = _mm_or_si128(bxh, bxhih);
        bx_0 = MM256_سیٹ_M128_I(bxh, bxl);

        ثابت __m256i by_0 = _mm256_loadu_si256((ثابت __m256i *)y[ib].qs);

        ثابت __m256 q = mul_مجموع_i8_جوڑے_float(bx_0, by_0);

        /* Multiply q with scale and accumulate */
        acc = mm256_جوڑو_ps(_mm256_mul_ps(d, q), acc);
    }

    *s = hsum_float_8(acc);
#ورنہ
    غیر_مستعمل(nb);
    غیر_مستعمل(ib);
    غیر_مستعمل(x);
    غیر_مستعمل(y);
    ggml_vec_dot_q5_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#اگر_ختم
}

خالی ggml_vec_dot_q5_1_q8_1(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc) {
    ثابت صحیح qk = QK8_1;
    ثابت صحیح nb = n / qk;

    صحیح ib = 0;

    أكد(n % qk == 0);
    أكد(qk == QK5_1);
    أكد(nrc == 1);
    غیر_مستعمل(nrc);
    غیر_مستعمل(bx);
    غیر_مستعمل(by);
    غیر_مستعمل(bs);

    ثابت بلاک_q5_1 * GGML_RESTRICT x = vx;
    ثابت بلاک_q8_1 * GGML_RESTRICT y = vy;

#اگر defined(__AVX2__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    اعشاری summs = 0.0f;

    // Main loop
    برائے (; ib < nb; ++ib) {
        ثابت __m256 dx = _mm256_set1_ps(GGML_CPU_FP16_مقصد_FP32(x[ib].d));

        summs += GGML_CPU_FP16_مقصد_FP32(x[ib].m) * GGML_CPU_FP16_مقصد_FP32(y[ib].s);

        __m256i qx = بائٹیں_from_nibbles_32(x[ib].qs);
        __m256i bxhi = بائٹیں_from_بٹ_32(x[ib].qh);
        bxhi = _mm256_and_si256(bxhi, _mm256_set1_epi8(0x10));
        qx = _mm256_or_si256(qx, bxhi);

        ثابت __m256 dy = _mm256_set1_ps(GGML_CPU_FP16_مقصد_FP32(y[ib].d));
        ثابت __m256i qy = _mm256_loadu_si256((ثابت __m256i *)y[ib].qs);

        ثابت __m256 q = mul_مجموع_us8_جوڑے_float(qx, qy);

        acc = _mm256_fmadd_ps(q, _mm256_mul_ps(dx, dy), acc);
    }

    *s = hsum_float_8(acc) + summs;
#وگرنہ_اگر defined(__AVX__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();
    __m128i نقاب = _mm_set1_epi8(0x10);

    اعشاری summs = 0.0f;

    // Main loop
    برائے (; ib < nb; ++ib) {
        ثابت __m256 dx = _mm256_set1_ps(GGML_CPU_FP16_مقصد_FP32(x[ib].d));

        summs += GGML_CPU_FP16_مقصد_FP32(x[ib].m) * GGML_CPU_FP16_مقصد_FP32(y[ib].s);

        __m256i bx_0 = بائٹیں_from_nibbles_32(x[ib].qs);
        ثابت __m256i bxhi = بائٹیں_from_بٹ_32(x[ib].qh);
        __m128i bxhil = _mm256_castsi256_si128(bxhi);
        __m128i bxhih = _mm256_extractf128_si256(bxhi, 1);
        bxhil = _mm_and_si128(bxhil, نقاب);
        bxhih = _mm_and_si128(bxhih, نقاب);
        __m128i bxl = _mm256_castsi256_si128(bx_0);
        __m128i bxh = _mm256_extractf128_si256(bx_0, 1);
        bxl = _mm_or_si128(bxl, bxhil);
        bxh = _mm_or_si128(bxh, bxhih);
        bx_0 = MM256_سیٹ_M128_I(bxh, bxl);

        ثابت __m256 dy = _mm256_set1_ps(GGML_CPU_FP16_مقصد_FP32(y[ib].d));
        ثابت __m256i by_0 = _mm256_loadu_si256((ثابت __m256i *)y[ib].qs);

        ثابت __m256 q = mul_مجموع_us8_جوڑے_float(bx_0, by_0);

        acc = mm256_جوڑو_ps(_mm256_mul_ps(q, _mm256_mul_ps(dx, dy)), acc);
    }

    *s = hsum_float_8(acc) + summs;
#ورنہ
    غیر_مستعمل(nb);
    غیر_مستعمل(ib);
    غیر_مستعمل(x);
    غیر_مستعمل(y);
    ggml_vec_dot_q5_1_q8_1_generic(n, s, bs, vx, bx, vy, by, nrc);
#اگر_ختم
}

خالی ggml_vec_dot_q8_0_q8_0(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc) {
    ثابت صحیح qk = QK8_0;
    ثابت صحیح nb = n / qk;

    أكد(n % qk == 0);
    أكد(nrc == 1);
    غیر_مستعمل(nrc);
    غیر_مستعمل(bx);
    غیر_مستعمل(by);
    غیر_مستعمل(bs);

    ثابت بلاک_q8_0 * GGML_RESTRICT x = vx;
    ثابت بلاک_q8_0 * GGML_RESTRICT y = vy;

    صحیح ib = 0;
    اعشاری sumf = 0;

#اگر defined(__AVX2__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    // Main loop
    برائے (; ib < nb; ++ib) {
        // Compute combined scale for the block
        ثابت __m256 d = _mm256_set1_ps(GGML_CPU_FP16_مقصد_FP32(x[ib].d) * GGML_CPU_FP16_مقصد_FP32(y[ib].d));
        __m256i qx = _mm256_loadu_si256((ثابت __m256i *)x[ib].qs);
        __m256i qy = _mm256_loadu_si256((ثابت __m256i *)y[ib].qs);

        ثابت __m256 q = mul_مجموع_i8_جوڑے_float(qx, qy);

        // Multiply q with scale and accumulate
        acc = _mm256_fmadd_ps( d, q, acc );
    }

    sumf = hsum_float_8(acc);
#وگرنہ_اگر defined(__AVX__)
    __m256 accum = _mm256_setzero_ps();

    برائے (; ib + 1 < nb; ib += 2) {
        ثابت __m128i qx_1_0 = _mm_loadu_si128((ثابت __m128i *)x[ib].qs);
        ثابت __m128i qx_1_1 = _mm_loadu_si128((ثابت __m128i *)x[ib].qs + 1);
        ثابت __m128i qx_2_0 = _mm_loadu_si128((ثابت __m128i *)x[ib + 1].qs);
        ثابت __m128i qx_2_1 = _mm_loadu_si128((ثابت __m128i *)x[ib + 1].qs + 1);
        ثابت __m128i qy_1_0 = _mm_loadu_si128((ثابت __m128i *)y[ib].qs);
        ثابت __m128i qy_1_1 = _mm_loadu_si128((ثابت __m128i *)y[ib].qs + 1);
        ثابت __m128i qy_2_0 = _mm_loadu_si128((ثابت __m128i *)y[ib + 1].qs);
        ثابت __m128i qy_2_1 = _mm_loadu_si128((ثابت __m128i *)y[ib + 1].qs + 1);

        ثابت __m256 p = mul_مجموع_i8_quad_float(qx_1_0, qx_1_1, qx_2_0, qx_2_1, qy_1_0, qy_1_1, qy_2_0, qy_2_1);
        ثابت __m256 deltas = quad_fp16_delta_float(x[ib].d, y[ib].d, x[ib + 1].d, y[ib + 1].d);
        accum = mm256_جوڑو_ps(_mm256_mul_ps(deltas, p), accum);
    }

    sumf = hsum_float_8(accum);
#اگر_ختم
    برائے (; ib < nb; ++ib) {
        صحیح sumi = 0;

        برائے (صحیح j = 0; j < qk; j++) {
            sumi += x[ib].qs[j]*y[ib].qs[j];
        }

        sumf += sumi*(GGML_CPU_FP16_مقصد_FP32(x[ib].d)*GGML_CPU_FP16_مقصد_FP32(y[ib].d));
    }

    *s = sumf;
}

خالی ggml_vec_dot_tq1_0_q8_K(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc) {
    أكد(nrc == 1);
    غیر_مستعمل(nrc);
    غیر_مستعمل(bx);
    غیر_مستعمل(by);
    غیر_مستعمل(bs);

    ثابت بلاک_tq1_0 * GGML_RESTRICT x = vx;
    ثابت بلاک_q8_K  * GGML_RESTRICT y = vy;

    ثابت صحیح nb = n / QK_K;

#اگر defined(__AVX2__)
    __m256 sumf = _mm256_setzero_ps();

    برائے (صحیح i = 0; i < nb; ++i) {
        // 16-bit sums
        __m256i sumi0 = _mm256_setzero_si256();
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();

        // first 32 bytes of 5 elements
        {
            __m256i qx0 = _mm256_loadu_si256((ثابت __m256i *) (x[i].qs));
            // 8-bit multiplies with shifts, masks and adds
            __m256i qx1 = mm256_جوڑو_epi8(qx0, mm256_جوڑو_epi8(qx0, qx0)); // 1 * 3
            __m256i qx2 = mm256_جوڑو_epi8(_mm256_and_si256(_mm256_slli_epi16(qx0, 3), _mm256_set1_epi8(-8)), qx0); // 1 * 9
            __m256i qx3 = mm256_جوڑو_epi8(_mm256_and_si256(_mm256_slli_epi16(qx1, 3), _mm256_set1_epi8(-8)), qx1); // 3 * 9
            __m256i qx4 = mm256_جوڑو_epi8(_mm256_and_si256(_mm256_slli_epi16(qx2, 3), _mm256_set1_epi8(-8)), qx2); // 9 * 9

            // TODO: can _mm256_mulhi_epu16 be faster even if 16-bits?

            // Cancel the +1 from avg so that it behaves like a halving add
            qx0 = _mm256_subs_epu8(qx0, _mm256_set1_epi8(1));
            qx1 = _mm256_subs_epu8(qx1, _mm256_set1_epi8(1));
            qx2 = _mm256_subs_epu8(qx2, _mm256_set1_epi8(1));
            qx3 = _mm256_subs_epu8(qx3, _mm256_set1_epi8(1));
            qx4 = _mm256_subs_epu8(qx4, _mm256_set1_epi8(1));
            // Multiply by 3 and get the top 2 bits
            qx0 = _mm256_avg_epu8(qx0, _mm256_avg_epu8(qx0, _mm256_setzero_si256()));
            qx1 = _mm256_avg_epu8(qx1, _mm256_avg_epu8(qx1, _mm256_setzero_si256()));
            qx2 = _mm256_avg_epu8(qx2, _mm256_avg_epu8(qx2, _mm256_setzero_si256()));
            qx3 = _mm256_avg_epu8(qx3, _mm256_avg_epu8(qx3, _mm256_setzero_si256()));
            qx4 = _mm256_avg_epu8(qx4, _mm256_avg_epu8(qx4, _mm256_setzero_si256()));
            qx0 = _mm256_and_si256(_mm256_srli_epi16(qx0, 6), _mm256_set1_epi8(3));
            qx1 = _mm256_and_si256(_mm256_srli_epi16(qx1, 6), _mm256_set1_epi8(3));
            qx2 = _mm256_and_si256(_mm256_srli_epi16(qx2, 6), _mm256_set1_epi8(3));
            qx3 = _mm256_and_si256(_mm256_srli_epi16(qx3, 6), _mm256_set1_epi8(3));
            qx4 = _mm256_and_si256(_mm256_srli_epi16(qx4, 6), _mm256_set1_epi8(3));

            ثابت __m256i qy0 = _mm256_loadu_si256((ثابت __m256i *) (y[i].qs +   0));
            ثابت __m256i qy1 = _mm256_loadu_si256((ثابت __m256i *) (y[i].qs +  32));
            ثابت __m256i qy2 = _mm256_loadu_si256((ثابت __m256i *) (y[i].qs +  64));
            ثابت __m256i qy3 = _mm256_loadu_si256((ثابت __m256i *) (y[i].qs +  96));
            ثابت __m256i qy4 = _mm256_loadu_si256((ثابت __m256i *) (y[i].qs + 128));

            qx0 = _mm256_maddubs_epi16(qx0, qy0);
            qx1 = _mm256_maddubs_epi16(qx1, qy1);
            qx2 = _mm256_maddubs_epi16(qx2, qy2);
            qx3 = _mm256_maddubs_epi16(qx3, qy3);
            qx4 = _mm256_maddubs_epi16(qx4, qy4);

            sumi0 = mm256_جوڑو_epi16(sumi0, mm256_جوڑو_epi16(qx0, qx1));
            sumi1 = mm256_جوڑو_epi16(sumi1, mm256_جوڑو_epi16(qx2, qx3));
            sumi2 = mm256_جوڑو_epi16(sumi2, qx4);
        }

        // last 16 bytes of 5-element, along with the 4 bytes of 4 elements
        {
            __m128i qx0 = _mm_loadu_si128((ثابت __m128i *) (x[i].qs + 32));
            uint32_t qh;
            حافظہ_نقل(&qh, x[i].qh, حجم(qh)); // potentially unaligned
            __m256i qx5_l = _mm256_cvtepu8_epi16(_mm_set1_epi32(qh));
            __m128i qx1 = mm_جوڑو_epi8(qx0, mm_جوڑو_epi8(qx0, qx0)); // 1 * 3
            __m128i qx2 = mm_جوڑو_epi8(_mm_and_si128(_mm_slli_epi16(qx0, 3), _mm_set1_epi8(-8)), qx0); // 1 * 9
            __m128i qx3 = mm_جوڑو_epi8(_mm_and_si128(_mm_slli_epi16(qx1, 3), _mm_set1_epi8(-8)), qx1); // 3 * 9
            __m128i qx4 = mm_جوڑو_epi8(_mm_and_si128(_mm_slli_epi16(qx2, 3), _mm_set1_epi8(-8)), qx2); // 9 * 9
            __m256i qx01 = MM256_سیٹ_M128_I(qx1, qx0);
            __m256i qx23 = MM256_سیٹ_M128_I(qx3, qx2);

            // avx2 does not have 8-bit multiplies, so 16-bit it is.
            qx5_l = _mm256_mullo_epi16(qx5_l, mm256_سیٹ_epi16(27, 27, 27, 27, 9, 9, 9, 9, 3, 3, 3, 3, 1, 1, 1, 1));
            qx5_l = _mm256_and_si256(qx5_l, _mm256_set1_epi16(0xFF));
            __m128i qx5 = _mm_packus_epi16(_mm256_castsi256_si128(qx5_l), _mm256_extracti128_si256(qx5_l, 1));

            __m256i qx45 = MM256_سیٹ_M128_I(qx5, qx4);

            // Cancel the +1 from avg so that it behaves like a halving add
            qx01 = _mm256_subs_epu8(qx01, _mm256_set1_epi8(1));
            qx23 = _mm256_subs_epu8(qx23, _mm256_set1_epi8(1));
            qx45 = _mm256_subs_epu8(qx45, _mm256_set1_epi8(1));
            // Multiply by 3 and get the top 2 bits
            qx01 = _mm256_avg_epu8(qx01, _mm256_avg_epu8(qx01, _mm256_setzero_si256()));
            qx23 = _mm256_avg_epu8(qx23, _mm256_avg_epu8(qx23, _mm256_setzero_si256()));
            qx45 = _mm256_avg_epu8(qx45, _mm256_avg_epu8(qx45, _mm256_setzero_si256()));
            qx01 = _mm256_and_si256(_mm256_srli_epi16(qx01, 6), _mm256_set1_epi8(3));
            qx23 = _mm256_and_si256(_mm256_srli_epi16(qx23, 6), _mm256_set1_epi8(3));
            qx45 = _mm256_and_si256(_mm256_srli_epi16(qx45, 6), _mm256_set1_epi8(3));

            ثابت __m256i qy01 = _mm256_loadu_si256((ثابت __m256i *) (y[i].qs + 160));
            ثابت __m256i qy23 = _mm256_loadu_si256((ثابت __m256i *) (y[i].qs + 192));
            ثابت __m256i qy45 = _mm256_loadu_si256((ثابت __m256i *) (y[i].qs + 224));

            qx01 = _mm256_maddubs_epi16(qx01, qy01);
            qx23 = _mm256_maddubs_epi16(qx23, qy23);
            qx45 = _mm256_maddubs_epi16(qx45, qy45);

            sumi0 = mm256_جوڑو_epi16(sumi0, qx01);
            sumi1 = mm256_جوڑو_epi16(sumi1, qx23);
            sumi2 = mm256_جوڑو_epi16(sumi2, qx45);
        }

        ثابت __m256i ysum = _mm256_loadu_si256((ثابت __m256i *) y[i].bsums);
        ثابت __m256 d = _mm256_set1_ps(y[i].d * GGML_CPU_FP16_مقصد_FP32(x[i].d));

        sumi0 = _mm256_sub_epi16(sumi0, ysum);
        sumi0 = mm256_جوڑو_epi16(sumi0, mm256_جوڑو_epi16(sumi1, sumi2));
        sumi0 = _mm256_madd_epi16(sumi0, _mm256_set1_epi16(1));

        sumf = mm256_جوڑو_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(sumi0), d), sumf);
    }

    *s = hsum_float_8(sumf);

#ورنہ
    غیر_مستعمل(x);
    غیر_مستعمل(y);
    غیر_مستعمل(nb);
    ggml_vec_dot_tq1_0_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#اگر_ختم
}

خالی ggml_vec_dot_tq2_0_q8_K(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc) {
    أكد(nrc == 1);
    غیر_مستعمل(nrc);
    غیر_مستعمل(bx);
    غیر_مستعمل(by);
    غیر_مستعمل(bs);

    ثابت بلاک_tq2_0 * GGML_RESTRICT x = vx;
    ثابت بلاک_q8_K  * GGML_RESTRICT y = vy;

    ثابت صحیح nb = n / QK_K;

#اگر defined(__AVX2__)
    __m256 sumf = _mm256_setzero_ps();

    برائے (صحیح i = 0; i < nb; ++i) {
        // 16-bit sums, because 256*127 still fits
        __m256i sumi0 = _mm256_setzero_si256();
        __m256i sumi1 = _mm256_setzero_si256();

        برائے (size_t j = 0; j < حجم(x->qs); j += 32) {
            __m256i qx0 = _mm256_loadu_si256((ثابت __m256i *) (x[i].qs + j));
            __m256i qx1 = _mm256_srli_epi16(qx0, 2);
            __m256i qx2 = _mm256_srli_epi16(qx0, 4);
            __m256i qx3 = _mm256_srli_epi16(qx0, 6);

            // 0, 1, 2 (should not be 3)
            qx0 = _mm256_and_si256(qx0, _mm256_set1_epi8(3));
            qx1 = _mm256_and_si256(qx1, _mm256_set1_epi8(3));
            qx2 = _mm256_and_si256(qx2, _mm256_set1_epi8(3));
            qx3 = _mm256_and_si256(qx3, _mm256_set1_epi8(3));

            ثابت __m256i qy0 = _mm256_loadu_si256((ثابت __m256i *) (y[i].qs + j*4 +  0));
            ثابت __m256i qy1 = _mm256_loadu_si256((ثابت __m256i *) (y[i].qs + j*4 + 32));
            ثابت __m256i qy2 = _mm256_loadu_si256((ثابت __m256i *) (y[i].qs + j*4 + 64));
            ثابت __m256i qy3 = _mm256_loadu_si256((ثابت __m256i *) (y[i].qs + j*4 + 96));

            qx0 = _mm256_maddubs_epi16(qx0, qy0);
            qx1 = _mm256_maddubs_epi16(qx1, qy1);
            qx2 = _mm256_maddubs_epi16(qx2, qy2);
            qx3 = _mm256_maddubs_epi16(qx3, qy3);

            sumi0 = mm256_جوڑو_epi16(sumi0, mm256_جوڑو_epi16(qx0, qx1));
            sumi1 = mm256_جوڑو_epi16(sumi1, mm256_جوڑو_epi16(qx2, qx3));
        }

        ثابت __m256i ysum = _mm256_loadu_si256((ثابت __m256i *) y[i].bsums);
        ثابت __m256 d = _mm256_set1_ps(y[i].d * GGML_CPU_FP16_مقصد_FP32(x[i].d));

        sumi0 = mm256_جوڑو_epi16(sumi0, sumi1);
        sumi0 = _mm256_sub_epi16(sumi0, ysum);
        sumi0 = _mm256_madd_epi16(sumi0, _mm256_set1_epi16(1));

        sumf = mm256_جوڑو_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(sumi0), d), sumf);
    }

    *s = hsum_float_8(sumf);

#ورنہ
    غیر_مستعمل(x);
    غیر_مستعمل(y);
    غیر_مستعمل(nb);
    ggml_vec_dot_tq2_0_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#اگر_ختم
}

خالی ggml_vec_dot_q2_K_q8_K(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc) {
    أكد(nrc == 1);
    غیر_مستعمل(nrc);
    غیر_مستعمل(bx);
    غیر_مستعمل(by);
    غیر_مستعمل(bs);

    ثابت بلاک_q2_K * GGML_RESTRICT x = vx;
    ثابت بلاک_q8_K * GGML_RESTRICT y = vy;

    ثابت صحیح nb = n / QK_K;

#اگر defined __AVX2__

    ثابت __m256i m3 = _mm256_set1_epi8(3);
    ثابت __m128i m4 = _mm_set1_epi8(0xF);

    __m256 acc = _mm256_setzero_ps();

    برائے (صحیح i = 0; i < nb; ++i) {

        ثابت اعشاری d = y[i].d * GGML_CPU_FP16_مقصد_FP32(x[i].d);
        ثابت اعشاری dmin = -y[i].d * GGML_CPU_FP16_مقصد_FP32(x[i].dmin);

        ثابت uint8_t * GGML_RESTRICT q2 = x[i].qs;
        ثابت int8_t  * GGML_RESTRICT q8 = y[i].qs;

        ثابت __m128i mins_and_scales = _mm_loadu_si128((ثابت __m128i*)x[i].scales);
        ثابت __m128i scales8 = _mm_and_si128(mins_and_scales, m4);
        ثابت __m128i mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
        ثابت __m256i mins = _mm256_cvtepi8_epi16(mins8);
        ثابت __m256i prod = _mm256_madd_epi16(mins, _mm256_loadu_si256((ثابت __m256i*)y[i].bsums));

        acc = _mm256_fmadd_ps(mm256_نشر_ss(&dmin), _mm256_cvtepi32_ps(prod), acc);

        ثابت __m256i all_scales = _mm256_cvtepi8_epi16(scales8);
        ثابت __m128i l_scales = _mm256_extracti128_si256(all_scales, 0);
        ثابت __m128i h_scales = _mm256_extracti128_si256(all_scales, 1);
        ثابت __m256i scales[2] = {MM256_سیٹ_M128_I(l_scales, l_scales), MM256_سیٹ_M128_I(h_scales, h_scales)};

        __m256i sumi = _mm256_setzero_si256();

        برائے (صحیح j = 0; j < QK_K/128; ++j) {

            ثابت __m256i q2bits = _mm256_loadu_si256((ثابت __m256i*)q2); q2 += 32;

            ثابت __m256i q8_0 = _mm256_loadu_si256((ثابت __m256i*)q8); q8 += 32;
            ثابت __m256i q8_1 = _mm256_loadu_si256((ثابت __m256i*)q8); q8 += 32;
            ثابت __m256i q8_2 = _mm256_loadu_si256((ثابت __m256i*)q8); q8 += 32;
            ثابت __m256i q8_3 = _mm256_loadu_si256((ثابت __m256i*)q8); q8 += 32;

            ثابت __m256i q2_0 = _mm256_and_si256(q2bits, m3);
            ثابت __m256i q2_1 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 2), m3);
            ثابت __m256i q2_2 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 4), m3);
            ثابت __m256i q2_3 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 6), m3);

            __m256i p0 = _mm256_maddubs_epi16(q2_0, q8_0);
            __m256i p1 = _mm256_maddubs_epi16(q2_1, q8_1);
            __m256i p2 = _mm256_maddubs_epi16(q2_2, q8_2);
            __m256i p3 = _mm256_maddubs_epi16(q2_3, q8_3);

            p0 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(0)), p0);
            p1 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(1)), p1);
            p2 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(2)), p2);
            p3 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(3)), p3);

            p0 = mm256_جوڑو_epi32(p0, p1);
            p2 = mm256_جوڑو_epi32(p2, p3);

            sumi = mm256_جوڑو_epi32(sumi, mm256_جوڑو_epi32(p0, p2));
        }

        acc = _mm256_fmadd_ps(mm256_نشر_ss(&d), _mm256_cvtepi32_ps(sumi), acc);

    }

    *s = hsum_float_8(acc);

#وگرنہ_اگر defined __AVX__

    ثابت __m128i m3 = _mm_set1_epi8(0x3);
    ثابت __m128i m4 = _mm_set1_epi8(0xF);
    ثابت __m128i m2 = _mm_set1_epi8(0x2);

    __m256 acc = _mm256_setzero_ps();

    برائے (صحیح i = 0; i < nb; ++i) {

        ثابت اعشاری dall = y[i].d * GGML_CPU_FP16_مقصد_FP32(x[i].d);
        ثابت اعشاری dmin = -y[i].d * GGML_CPU_FP16_مقصد_FP32(x[i].dmin);

        ثابت uint8_t * GGML_RESTRICT q2 = x[i].qs;
        ثابت int8_t  * GGML_RESTRICT q8 = y[i].qs;

        // load mins and scales from block_q2_K.scales[QK_K/16]
        ثابت __m128i mins_and_scales = _mm_loadu_si128((ثابت __m128i*)x[i].scales);
        ثابت __m128i scales16 = _mm_and_si128(mins_and_scales, m4);
        ثابت __m128i mins16 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
        ثابت __m128i mins_0 = _mm_cvtepi8_epi16(mins16);
        ثابت __m128i mins_1 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(mins16, mins16));

        // summs = y[i].bsums * (x[i].scales >> 4) in 16bits*8*2 to 32bits*4*2
        ثابت __m128i summs_0 = _mm_madd_epi16(mins_0, _mm_loadu_si128((ثابت __m128i*)&y[i].bsums[0]));
        ثابت __m128i summs_1 = _mm_madd_epi16(mins_1, _mm_loadu_si128((ثابت __m128i*)&y[i].bsums[8]));

        // sumf += -dmin * summs in 32bits*8
        acc = mm256_جوڑو_ps(_mm256_mul_ps(mm256_نشر_ss(&dmin), _mm256_cvtepi32_ps(MM256_سیٹ_M128_I(summs_1, summs_0))), acc);

        ثابت __m128i scales_0 = _mm_cvtepi8_epi16(scales16);
        ثابت __m128i scales_1 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(scales16, scales16));
        ثابت __m128i scales[2] = { scales_0, scales_1 };

        __m128i sumi_0 = _mm_setzero_si128();
        __m128i sumi_1 = _mm_setzero_si128();

        برائے (صحیح j = 0; j < QK_K/128; ++j) {

            // load Q8 quants int8*16*8 from block_q8_K.qs[QK_K]
            ثابت __m128i q8_0 = _mm_loadu_si128((ثابت __m128i*)q8); q8 += 16;
            ثابت __m128i q8_1 = _mm_loadu_si128((ثابت __m128i*)q8); q8 += 16;
            ثابت __m128i q8_2 = _mm_loadu_si128((ثابت __m128i*)q8); q8 += 16;
            ثابت __m128i q8_3 = _mm_loadu_si128((ثابت __m128i*)q8); q8 += 16;
            ثابت __m128i q8_4 = _mm_loadu_si128((ثابت __m128i*)q8); q8 += 16;
            ثابت __m128i q8_5 = _mm_loadu_si128((ثابت __m128i*)q8); q8 += 16;
            ثابت __m128i q8_6 = _mm_loadu_si128((ثابت __m128i*)q8); q8 += 16;
            ثابت __m128i q8_7 = _mm_loadu_si128((ثابت __m128i*)q8); q8 += 16;

            // load 2bits*16*8 from block_q2_K.qs[QK_K/4]
            __m128i q2bits = _mm_loadu_si128((ثابت __m128i*)q2); q2 += 16;
            ثابت __m128i q2_0 = _mm_and_si128(q2bits, m3);
            ثابت __m128i q2_2 = _mm_and_si128(_mm_srli_epi16(q2bits, 2), m3);
            ثابت __m128i q2_4 = _mm_and_si128(_mm_srli_epi16(q2bits, 4), m3);
            ثابت __m128i q2_6 = _mm_and_si128(_mm_srli_epi16(q2bits, 6), m3);
            q2bits = _mm_loadu_si128((ثابت __m128i*)q2); q2 += 16;
            ثابت __m128i q2_1 = _mm_and_si128(q2bits, m3);
            ثابت __m128i q2_3 = _mm_and_si128(_mm_srli_epi16(q2bits, 2), m3);
            ثابت __m128i q2_5 = _mm_and_si128(_mm_srli_epi16(q2bits, 4), m3);
            ثابت __m128i q2_7 = _mm_and_si128(_mm_srli_epi16(q2bits, 6), m3);

            // isuml = q8[l] * ((q2[l] >> shift) & 3) in 8bits*16*8 to 16bits*8*8
            __m128i p0 = _mm_maddubs_epi16(q2_0, q8_0);
            __m128i p1 = _mm_maddubs_epi16(q2_1, q8_1);
            __m128i p2 = _mm_maddubs_epi16(q2_2, q8_2);
            __m128i p3 = _mm_maddubs_epi16(q2_3, q8_3);
            __m128i p4 = _mm_maddubs_epi16(q2_4, q8_4);
            __m128i p5 = _mm_maddubs_epi16(q2_5, q8_5);
            __m128i p6 = _mm_maddubs_epi16(q2_6, q8_6);
            __m128i p7 = _mm_maddubs_epi16(q2_7, q8_7);

            // isum += (x[i].scales[is++] & 0xF) * isuml in 16bits*8*8 to 32bits*4*8
            __m128i shuffle = _mm_set1_epi16(0x0100);
            p0 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p0);
            shuffle = mm_جوڑو_epi16(shuffle, m2);
            p1 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p1);
            shuffle = mm_جوڑو_epi16(shuffle, m2);
            p2 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p2);
            shuffle = mm_جوڑو_epi16(shuffle, m2);
            p3 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p3);
            shuffle = mm_جوڑو_epi16(shuffle, m2);
            p4 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p4);
            shuffle = mm_جوڑو_epi16(shuffle, m2);
            p5 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p5);
            shuffle = mm_جوڑو_epi16(shuffle, m2);
            p6 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p6);
            shuffle = mm_جوڑو_epi16(shuffle, m2);
            p7 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p7);

            p0 = mm_جوڑو_epi32(p0, p1);
            p2 = mm_جوڑو_epi32(p2, p3);
            p4 = mm_جوڑو_epi32(p4, p5);
            p6 = mm_جوڑو_epi32(p6, p7);

            // isum in 32bits*4*2
            sumi_0 = mm_جوڑو_epi32(sumi_0, mm_جوڑو_epi32(p0, p2));
            sumi_1 = mm_جوڑو_epi32(sumi_1, mm_جوڑو_epi32(p4, p6));
        }

        // sumf += dall * isum - dmin * summs in 32bits
        __m256i sumi = MM256_سیٹ_M128_I(sumi_1, sumi_0);
        acc = mm256_جوڑو_ps(_mm256_mul_ps(mm256_نشر_ss(&dall), _mm256_cvtepi32_ps(sumi)), acc);
    }

    *s = hsum_float_8(acc);

#ورنہ
    غیر_مستعمل(x);
    غیر_مستعمل(y);
    غیر_مستعمل(nb);
    ggml_vec_dot_q2_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#اگر_ختم
}

خالی ggml_vec_dot_q3_K_q8_K(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc) {
    أكد(n % QK_K == 0);
    أكد(nrc == 1);
    غیر_مستعمل(nrc);
    غیر_مستعمل(bx);
    غیر_مستعمل(by);
    غیر_مستعمل(bs);

    ثابت uint32_t kmask1 = 0x03030303;
    ثابت uint32_t kmask2 = 0x0f0f0f0f;

    ثابت بلاک_q3_K * GGML_RESTRICT x = vx;
    ثابت بلاک_q8_K * GGML_RESTRICT y = vy;

    ثابت صحیح nb = n / QK_K;

#اگر defined __AVX2__

    ثابت __m256i m3 = _mm256_set1_epi8(3);
    ثابت __m256i mone = _mm256_set1_epi8(1);
    ثابت __m128i m32 = _mm_set1_epi8(32);

    __m256 acc = _mm256_setzero_ps();

    uint32_t aux[3];

    برائے (صحیح i = 0; i < nb; ++i) {

        ثابت اعشاری d = y[i].d * GGML_CPU_FP16_مقصد_FP32(x[i].d);

        ثابت uint8_t * GGML_RESTRICT q3 = x[i].qs;
        ثابت int8_t  * GGML_RESTRICT q8 = y[i].qs;

        // Set up scales
        حافظہ_نقل(aux, x[i].scales, 12);
        __m128i scales128 = mm_سیٹ_epi32(
                ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4),
                ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4),
                (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4),
                (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4));
        scales128 = _mm_sub_epi8(scales128, m32);
        ثابت __m256i all_scales = _mm256_cvtepi8_epi16(scales128);
        ثابت __m128i l_scales = _mm256_extracti128_si256(all_scales, 0);
        ثابت __m128i h_scales = _mm256_extracti128_si256(all_scales, 1);
        ثابت __m256i scales[2] = {MM256_سیٹ_M128_I(l_scales, l_scales), MM256_سیٹ_M128_I(h_scales, h_scales)};

        // high bit
        ثابت __m256i hbits = _mm256_loadu_si256((ثابت __m256i*)x[i].hmask);

        // integer accumulator
        __m256i sumi = _mm256_setzero_si256();

        صحیح ثنائی_ہندسہ = 0;
        صحیح is  = 0;

        برائے (صحیح j = 0; j < QK_K/128; ++j) {
            // load low 2 bits
            ثابت __m256i q3bits = _mm256_loadu_si256((ثابت __m256i*)q3); q3 += 32;

            // prepare low and high bits
            ثابت __m256i q3l_0 = _mm256_and_si256(q3bits, m3);
            ثابت __m256i q3h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, ثنائی_ہندسہ)), ثنائی_ہندسہ), 2);
            ++ثنائی_ہندسہ;

            ثابت __m256i q3l_1 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 2), m3);
            ثابت __m256i q3h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, ثنائی_ہندسہ)), ثنائی_ہندسہ), 2);
            ++ثنائی_ہندسہ;

            ثابت __m256i q3l_2 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 4), m3);
            ثابت __m256i q3h_2 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, ثنائی_ہندسہ)), ثنائی_ہندسہ), 2);
            ++ثنائی_ہندسہ;

            ثابت __m256i q3l_3 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 6), m3);
            ثابت __m256i q3h_3 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, ثنائی_ہندسہ)), ثنائی_ہندسہ), 2);
            ++ثنائی_ہندسہ;

            // load Q8 quants
            ثابت __m256i q8_0 = _mm256_loadu_si256((ثابت __m256i*)q8); q8 += 32;
            ثابت __m256i q8_1 = _mm256_loadu_si256((ثابت __m256i*)q8); q8 += 32;
            ثابت __m256i q8_2 = _mm256_loadu_si256((ثابت __m256i*)q8); q8 += 32;
            ثابت __m256i q8_3 = _mm256_loadu_si256((ثابت __m256i*)q8); q8 += 32;

            // Dot product: we multiply the 2 low bits and 1 high bit part separately, so we can use _mm256_maddubs_epi16,
            // and then subtract. The high bit part has the 2 already subtracted (and so, it is zero if the high bit was not set,
            // and 2 if the high bit was set)
            __m256i q8s_0 = _mm256_maddubs_epi16(q3h_0, q8_0);
            __m256i q8s_1 = _mm256_maddubs_epi16(q3h_1, q8_1);
            __m256i q8s_2 = _mm256_maddubs_epi16(q3h_2, q8_2);
            __m256i q8s_3 = _mm256_maddubs_epi16(q3h_3, q8_3);

            __m256i p16_0 = _mm256_maddubs_epi16(q3l_0, q8_0);
            __m256i p16_1 = _mm256_maddubs_epi16(q3l_1, q8_1);
            __m256i p16_2 = _mm256_maddubs_epi16(q3l_2, q8_2);
            __m256i p16_3 = _mm256_maddubs_epi16(q3l_3, q8_3);

            p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
            p16_1 = _mm256_sub_epi16(p16_1, q8s_1);
            p16_2 = _mm256_sub_epi16(p16_2, q8s_2);
            p16_3 = _mm256_sub_epi16(p16_3, q8s_3);

            // multiply with scales
            p16_0 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(is + 0)), p16_0);
            p16_1 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(is + 1)), p16_1);
            p16_2 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(is + 2)), p16_2);
            p16_3 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(is + 3)), p16_3);

            // accumulate
            p16_0 = mm256_جوڑو_epi32(p16_0, p16_1);
            p16_2 = mm256_جوڑو_epi32(p16_2, p16_3);
            sumi  = mm256_جوڑو_epi32(sumi, mm256_جوڑو_epi32(p16_0, p16_2));

        }

        // multiply with block scale and accumulate
        acc = _mm256_fmadd_ps(mm256_نشر_ss(&d), _mm256_cvtepi32_ps(sumi), acc);

    }

    *s = hsum_float_8(acc);

#وگرنہ_اگر defined __AVX__

    ثابت __m128i m3 = _mm_set1_epi8(3);
    ثابت __m128i mone = _mm_set1_epi8(1);
    ثابت __m128i m32 = _mm_set1_epi8(32);
    ثابت __m128i m2 = _mm_set1_epi8(2);

    __m256 acc = _mm256_setzero_ps();

    ثابت uint32_t *aux;

    برائے (صحیح i = 0; i < nb; ++i) {

        ثابت اعشاری d = y[i].d * GGML_CPU_FP16_مقصد_FP32(x[i].d);

        ثابت uint8_t * GGML_RESTRICT q3 = x[i].qs;
        ثابت int8_t  * GGML_RESTRICT q8 = y[i].qs;

        // Set up scales
        aux = (ثابت uint32_t *)x[i].scales;
        __m128i scales128 = mm_سیٹ_epi32(
                ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4),
                ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4),
                (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4),
                (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4));
        scales128 = _mm_sub_epi8(scales128, m32);
        ثابت __m128i scales_0 = _mm_cvtepi8_epi16(scales128);
        ثابت __m128i scales_1 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(scales128, scales128));
        ثابت __m128i scales[2] = { scales_0, scales_1 };

        // high bit *128*2 from block_q3_K.hmask[QK_K/8]
        ثابت __m128i hbits_0 = _mm_loadu_si128((ثابت __m128i*)&x[i].hmask[0]);
        ثابت __m128i hbits_1 = _mm_loadu_si128((ثابت __m128i*)&x[i].hmask[16]);

        // integer accumulator
        __m128i sumi_0 = _mm_setzero_si128();
        __m128i sumi_1 = _mm_setzero_si128();

        برائے (صحیح j = 0; j < QK_K/128; ++j) {
            // load low 2 bits *64*2 from block_q3_K.qs[QK_K/4]
            ثابت __m128i q3bits_0 = _mm_loadu_si128((ثابت __m128i*)q3); q3 += 16;
            ثابت __m128i q3bits_1 = _mm_loadu_si128((ثابت __m128i*)q3); q3 += 16;

            // prepare low and high bits
            ثابت صحیح ثنائی_ہندسہ = j << 2;

            ثابت __m128i q3l_0 = _mm_and_si128(q3bits_0, m3);
            ثابت __m128i q3l_1 = _mm_and_si128(q3bits_1, m3);
            ثابت __m128i q3h_0 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_0, _mm_slli_epi16(mone, ثنائی_ہندسہ)), ثنائی_ہندسہ), 2);
            ثابت __m128i q3h_1 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_1, _mm_slli_epi16(mone, ثنائی_ہندسہ)), ثنائی_ہندسہ), 2);

            ثابت __m128i q3l_2 = _mm_and_si128(_mm_srli_epi16(q3bits_0, 2), m3);
            ثابت __m128i q3l_3 = _mm_and_si128(_mm_srli_epi16(q3bits_1, 2), m3);
            ثابت __m128i q3h_2 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_0, _mm_slli_epi16(mone, ثنائی_ہندسہ+1)), ثنائی_ہندسہ+1), 2);
            ثابت __m128i q3h_3 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_1, _mm_slli_epi16(mone, ثنائی_ہندسہ+1)), ثنائی_ہندسہ+1), 2);

            ثابت __m128i q3l_4 = _mm_and_si128(_mm_srli_epi16(q3bits_0, 4), m3);
            ثابت __m128i q3l_5 = _mm_and_si128(_mm_srli_epi16(q3bits_1, 4), m3);
            ثابت __m128i q3h_4 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_0, _mm_slli_epi16(mone, ثنائی_ہندسہ+2)), ثنائی_ہندسہ+2), 2);
            ثابت __m128i q3h_5 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_1, _mm_slli_epi16(mone, ثنائی_ہندسہ+2)), ثنائی_ہندسہ+2), 2);

            ثابت __m128i q3l_6 = _mm_and_si128(_mm_srli_epi16(q3bits_0, 6), m3);
            ثابت __m128i q3l_7 = _mm_and_si128(_mm_srli_epi16(q3bits_1, 6), m3);
            ثابت __m128i q3h_6 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_0, _mm_slli_epi16(mone, ثنائی_ہندسہ+3)), ثنائی_ہندسہ+3), 2);
            ثابت __m128i q3h_7 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_1, _mm_slli_epi16(mone, ثنائی_ہندسہ+3)), ثنائی_ہندسہ+3), 2);

            // load Q8 quants from block_q8_K.qs[QK_K]
            ثابت __m128i q8_0 = _mm_loadu_si128((ثابت __m128i*)q8); q8 += 16;
            ثابت __m128i q8_1 = _mm_loadu_si128((ثابت __m128i*)q8); q8 += 16;
            ثابت __m128i q8_2 = _mm_loadu_si128((ثابت __m128i*)q8); q8 += 16;
            ثابت __m128i q8_3 = _mm_loadu_si128((ثابت __m128i*)q8); q8 += 16;
            ثابت __m128i q8_4 = _mm_loadu_si128((ثابت __m128i*)q8); q8 += 16;
            ثابت __m128i q8_5 = _mm_loadu_si128((ثابت __m128i*)q8); q8 += 16;
            ثابت __m128i q8_6 = _mm_loadu_si128((ثابت __m128i*)q8); q8 += 16;
            ثابت __m128i q8_7 = _mm_loadu_si128((ثابت __m128i*)q8); q8 += 16;

            // Dot product: we multiply the 2 low bits and 1 high bit part separately, so we can use _mm256_maddubs_epi16,
            // and then subtract. The high bit part has the 2 already subtracted (and so, it is zero if the high bit was not set,
            // and 2 if the high bit was set)
            __m128i q8s_0 = _mm_maddubs_epi16(q3h_0, q8_0);
            __m128i q8s_1 = _mm_maddubs_epi16(q3h_1, q8_1);
            __m128i q8s_2 = _mm_maddubs_epi16(q3h_2, q8_2);
            __m128i q8s_3 = _mm_maddubs_epi16(q3h_3, q8_3);
            __m128i q8s_4 = _mm_maddubs_epi16(q3h_4, q8_4);
            __m128i q8s_5 = _mm_maddubs_epi16(q3h_5, q8_5);
            __m128i q8s_6 = _mm_maddubs_epi16(q3h_6, q8_6);
            __m128i q8s_7 = _mm_maddubs_epi16(q3h_7, q8_7);

            __m128i p16_0 = _mm_maddubs_epi16(q3l_0, q8_0);
            __m128i p16_1 = _mm_maddubs_epi16(q3l_1, q8_1);
            __m128i p16_2 = _mm_maddubs_epi16(q3l_2, q8_2);
            __m128i p16_3 = _mm_maddubs_epi16(q3l_3, q8_3);
            __m128i p16_4 = _mm_maddubs_epi16(q3l_4, q8_4);
            __m128i p16_5 = _mm_maddubs_epi16(q3l_5, q8_5);
            __m128i p16_6 = _mm_maddubs_epi16(q3l_6, q8_6);
            __m128i p16_7 = _mm_maddubs_epi16(q3l_7, q8_7);

            p16_0 = _mm_sub_epi16(p16_0, q8s_0);
            p16_1 = _mm_sub_epi16(p16_1, q8s_1);
            p16_2 = _mm_sub_epi16(p16_2, q8s_2);
            p16_3 = _mm_sub_epi16(p16_3, q8s_3);
            p16_4 = _mm_sub_epi16(p16_4, q8s_4);
            p16_5 = _mm_sub_epi16(p16_5, q8s_5);
            p16_6 = _mm_sub_epi16(p16_6, q8s_6);
            p16_7 = _mm_sub_epi16(p16_7, q8s_7);

            // multiply with scales
            __m128i shuffle = _mm_set1_epi16(0x0100);
            p16_0 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_0);
            shuffle = mm_جوڑو_epi16(shuffle, m2);
            p16_1 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_1);
            shuffle = mm_جوڑو_epi16(shuffle, m2);
            p16_2 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_2);
            shuffle = mm_جوڑو_epi16(shuffle, m2);
            p16_3 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_3);
            shuffle = mm_جوڑو_epi16(shuffle, m2);
            p16_4 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_4);
            shuffle = mm_جوڑو_epi16(shuffle, m2);
            p16_5 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_5);
            shuffle = mm_جوڑو_epi16(shuffle, m2);
            p16_6 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_6);
            shuffle = mm_جوڑو_epi16(shuffle, m2);
            p16_7 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_7);

            // accumulate
            p16_0 = mm_جوڑو_epi32(p16_0, p16_1);
            p16_2 = mm_جوڑو_epi32(p16_2, p16_3);
            p16_4 = mm_جوڑو_epi32(p16_4, p16_5);
            p16_6 = mm_جوڑو_epi32(p16_6, p16_7);
            sumi_0 = mm_جوڑو_epi32(sumi_0, mm_جوڑو_epi32(p16_0, p16_2));
            sumi_1 = mm_جوڑو_epi32(sumi_1, mm_جوڑو_epi32(p16_4, p16_6));

        }

        // multiply with block scale and accumulate
        __m256i sumi = MM256_سیٹ_M128_I(sumi_1, sumi_0);
        acc = mm256_جوڑو_ps(_mm256_mul_ps(mm256_نشر_ss(&d), _mm256_cvtepi32_ps(sumi)), acc);

    }

    *s = hsum_float_8(acc);

#ورنہ
    غیر_مستعمل(kmask1);
    غیر_مستعمل(kmask2);
    غیر_مستعمل(x);
    غیر_مستعمل(y);
    غیر_مستعمل(nb);
    ggml_vec_dot_q3_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#اگر_ختم
}

خالی ggml_vec_dot_q4_K_q8_K(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc) {
    أكد(n % QK_K == 0);
    أكد(nrc == 1);
    غیر_مستعمل(nrc);
    غیر_مستعمل(bx);
    غیر_مستعمل(by);
    غیر_مستعمل(bs);

    ثابت بلاک_q4_K * GGML_RESTRICT x = vx;
    ثابت بلاک_q8_K * GGML_RESTRICT y = vy;

    ثابت صحیح nb = n / QK_K;

    ساکن ثابت uint32_t kmask1 = 0x3f3f3f3f;
    ساکن ثابت uint32_t kmask2 = 0x0f0f0f0f;
    ساکن ثابت uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

#اگر defined __AVX2__

    ثابت __m256i m4 = _mm256_set1_epi8(0xF);

    __m256 acc = _mm256_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();

   برائے (صحیح i = 0; i < nb; ++i) {

        ثابت اعشاری d = y[i].d * GGML_CPU_FP16_مقصد_FP32(x[i].d);
        ثابت اعشاری dmin = -y[i].d * GGML_CPU_FP16_مقصد_FP32(x[i].dmin);

        حافظہ_نقل(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        ثابت uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        ثابت uint8_t * GGML_RESTRICT q4 = x[i].qs;
        ثابت int8_t  * GGML_RESTRICT q8 = y[i].qs;

        ثابت __m256i mins_and_scales = _mm256_cvtepu8_epi16(mm_سیٹ_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));

        ثابت __m256i q8sums = _mm256_loadu_si256((ثابت __m256i*)y[i].bsums);
        ثابت __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        ثابت __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

        ثابت __m128i sc128  = _mm256_extracti128_si256(mins_and_scales, 0);
        ثابت __m256i scales = MM256_سیٹ_M128_I(sc128, sc128);

        __m256i sumi = _mm256_setzero_si256();

        برائے (صحیح j = 0; j < QK_K/64; ++j) {

            ثابت __m256i scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+0));
            ثابت __m256i scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+1));

            ثابت __m256i q4bits = _mm256_loadu_si256((ثابت __m256i*)q4); q4 += 32;
            ثابت __m256i q4l = _mm256_and_si256(q4bits, m4);
            ثابت __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

            ثابت __m256i q8l = _mm256_loadu_si256((ثابت __m256i*)q8); q8 += 32;
            __m256i p16l = _mm256_maddubs_epi16(q4l, q8l);
            p16l = _mm256_madd_epi16(scale_l, p16l);

            ثابت __m256i q8h = _mm256_loadu_si256((ثابت __m256i*)q8); q8 += 32;
            __m256i p16h = _mm256_maddubs_epi16(q4h, q8h);
            p16h = _mm256_madd_epi16(scale_h, p16h);
            ثابت __m256i sumj = mm256_جوڑو_epi32(p16l, p16h);

            sumi = mm256_جوڑو_epi32(sumi, sumj);
        }

        __m256 vd = _mm256_set1_ps(d);
        acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);

    }

    acc_m = mm_جوڑو_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = mm_جوڑو_ss(acc_m, _mm_movehdup_ps(acc_m));

    *s = hsum_float_8(acc) + _mm_cvtss_f32(acc_m);

#وگرنہ_اگر defined __AVX__

    ثابت __m128i m4 = _mm_set1_epi8(0xF);
    ثابت __m128i m2 = _mm_set1_epi8(0x2);

    __m256 acc = _mm256_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();

   برائے (صحیح i = 0; i < nb; ++i) {

        ثابت اعشاری d = y[i].d * GGML_CPU_FP16_مقصد_FP32(x[i].d);
        ثابت اعشاری dmin = -y[i].d * GGML_CPU_FP16_مقصد_FP32(x[i].dmin);

        ثابت uint8_t * GGML_RESTRICT q4 = x[i].qs;
        ثابت int8_t  * GGML_RESTRICT q8 = y[i].qs;

        حافظہ_نقل(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        ثابت uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        ثابت __m128i utmps = mm_سیٹ_epi32(utmp[3], utmp[2], utmp[1], utmp[0]);
        ثابت __m128i scales = _mm_cvtepu8_epi16(utmps);
        ثابت __m128i mins = _mm_cvtepu8_epi16(_mm_unpackhi_epi64(utmps, utmps));

        ثابت __m128i q8sums_0 = _mm_loadu_si128((ثابت __m128i*)&y[i].bsums[0]);
        ثابت __m128i q8sums_1 = _mm_loadu_si128((ثابت __m128i*)&y[i].bsums[8]);
        ثابت __m128i q8s = _mm_hadd_epi16(q8sums_0, q8sums_1);
        ثابت __m128i prod = _mm_madd_epi16(mins, q8s);
        acc_m = mm_جوڑو_ps(_mm_mul_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod)), acc_m);

        __m128i sumi_0 = _mm_setzero_si128();
        __m128i sumi_1 = _mm_setzero_si128();

        __m128i shuffle = _mm_set1_epi16(0x0100);
        برائے (صحیح j = 0; j < QK_K/64; ++j) {

            ثابت __m128i scale_l = _mm_shuffle_epi8(scales, shuffle);
            shuffle = mm_جوڑو_epi16(shuffle, m2);
            ثابت __m128i scale_h = _mm_shuffle_epi8(scales, shuffle);
            shuffle = mm_جوڑو_epi16(shuffle, m2);

            __m128i q4bits = _mm_loadu_si128((ثابت __m128i*)q4); q4 += 16;
            ثابت __m128i q4l_0 = _mm_and_si128(q4bits, m4);
            ثابت __m128i q4h_0 = _mm_and_si128(_mm_srli_epi16(q4bits, 4), m4);
            q4bits = _mm_loadu_si128((ثابت __m128i*)q4); q4 += 16;
            ثابت __m128i q4l_1 = _mm_and_si128(q4bits, m4);
            ثابت __m128i q4h_1 = _mm_and_si128(_mm_srli_epi16(q4bits, 4), m4);

            ثابت __m128i q8l_0 = _mm_loadu_si128((ثابت __m128i*)q8); q8 += 16;
            __m128i p16l = _mm_maddubs_epi16(q4l_0, q8l_0);
            p16l = _mm_madd_epi16(scale_l, p16l);
            sumi_0 = mm_جوڑو_epi32(sumi_0, p16l);
            ثابت __m128i q8l_1 = _mm_loadu_si128((ثابت __m128i*)q8); q8 += 16;
            p16l = _mm_maddubs_epi16(q4l_1, q8l_1);
            p16l = _mm_madd_epi16(scale_l, p16l);
            sumi_1 = mm_جوڑو_epi32(sumi_1, p16l);

            ثابت __m128i q8h_0 = _mm_loadu_si128((ثابت __m128i*)q8); q8 += 16;
            __m128i p16h = _mm_maddubs_epi16(q4h_0, q8h_0);
            p16h = _mm_madd_epi16(scale_h, p16h);
            sumi_0 = mm_جوڑو_epi32(sumi_0, p16h);
            ثابت __m128i q8h_1 = _mm_loadu_si128((ثابت __m128i*)q8); q8 += 16;
            p16h = _mm_maddubs_epi16(q4h_1, q8h_1);
            p16h = _mm_madd_epi16(scale_h, p16h);
            sumi_1 = mm_جوڑو_epi32(sumi_1, p16h);

        }

        __m256 vd = _mm256_set1_ps(d);
        __m256i sumi = MM256_سیٹ_M128_I(sumi_1, sumi_0);
        acc = mm256_جوڑو_ps(_mm256_mul_ps(vd, _mm256_cvtepi32_ps(sumi)), acc);

    }

    acc_m = mm_جوڑو_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = mm_جوڑو_ss(acc_m, _mm_movehdup_ps(acc_m));

    *s = hsum_float_8(acc) + _mm_cvtss_f32(acc_m);

#ورنہ
    غیر_مستعمل(x);
    غیر_مستعمل(y);
    غیر_مستعمل(nb);
    غیر_مستعمل(kmask1);
    غیر_مستعمل(kmask2);
    غیر_مستعمل(kmask3);
    غیر_مستعمل(utmp);
    ggml_vec_dot_q4_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#اگر_ختم
}

خالی ggml_vec_dot_q5_K_q8_K(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy,  size_t by, صحیح nrc) {
    أكد(n % QK_K == 0);
    أكد(nrc == 1);
    غیر_مستعمل(nrc);
    غیر_مستعمل(bx);
    غیر_مستعمل(by);
    غیر_مستعمل(bs);

    ثابت بلاک_q5_K * GGML_RESTRICT x = vx;
    ثابت بلاک_q8_K * GGML_RESTRICT y = vy;

    ثابت صحیح nb = n / QK_K;

    ساکن ثابت uint32_t kmask1 = 0x3f3f3f3f;
    ساکن ثابت uint32_t kmask2 = 0x0f0f0f0f;
    ساکن ثابت uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

#اگر defined __AVX2__

    ثابت __m256i m4 = _mm256_set1_epi8(0xF);
    ثابت __m128i mzero = _mm_setzero_si128();
    ثابت __m256i mone  = _mm256_set1_epi8(1);

    __m256 acc = _mm256_setzero_ps();

    اعشاری summs = 0.f;

    برائے (صحیح i = 0; i < nb; ++i) {
        ثابت uint8_t * GGML_RESTRICT q5 = x[i].qs;
        ثابت int8_t  * GGML_RESTRICT q8 = y[i].qs;

        ثابت اعشاری d = y[i].d * GGML_CPU_FP16_مقصد_FP32(x[i].d);
        ثابت اعشاری dmin = -y[i].d * GGML_CPU_FP16_مقصد_FP32(x[i].dmin);

        حافظہ_نقل(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        ثابت uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        ثابت __m256i mins_and_scales = _mm256_cvtepu8_epi16(mm_سیٹ_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));

        ثابت __m256i q8sums = _mm256_loadu_si256((ثابت __m256i*)y[i].bsums);
        ثابت __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        ثابت __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        ثابت __m128i hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
        summs += dmin * _mm_extract_epi32(hsum, 0);

        ثابت __m128i sc128  = _mm256_extracti128_si256(mins_and_scales, 0);
        ثابت __m256i scales = MM256_سیٹ_M128_I(sc128, sc128);

        ثابت __m256i hbits = _mm256_loadu_si256((ثابت __m256i*)x[i].qh);
        __m256i hmask = mone;

        __m256i sumi = _mm256_setzero_si256();

        صحیح ثنائی_ہندسہ = 0;

        برائے (صحیح j = 0; j < QK_K/64; ++j) {

            ثابت __m256i scale_0 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+0));
            ثابت __m256i scale_1 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+1));

            ثابت __m256i q5bits = _mm256_loadu_si256((ثابت __m256i*)q5); q5 += 32;

            ثابت __m256i q5l_0 = _mm256_and_si256(q5bits, m4);
            ثابت __m256i q5h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), ثنائی_ہندسہ++), 4);
            ثابت __m256i q5_0  = mm256_جوڑو_epi8(q5l_0, q5h_0);
            hmask = _mm256_slli_epi16(hmask, 1);

            ثابت __m256i q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4);
            ثابت __m256i q5h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), ثنائی_ہندسہ++), 4);
            ثابت __m256i q5_1  = mm256_جوڑو_epi8(q5l_1, q5h_1);
            hmask = _mm256_slli_epi16(hmask, 1);

            ثابت __m256i q8_0 = _mm256_loadu_si256((ثابت __m256i*)q8); q8 += 32;
            ثابت __m256i q8_1 = _mm256_loadu_si256((ثابت __m256i*)q8); q8 += 32;

            __m256i p16_0 = _mm256_maddubs_epi16(q5_0, q8_0);
            __m256i p16_1 = _mm256_maddubs_epi16(q5_1, q8_1);

            p16_0 = _mm256_madd_epi16(scale_0, p16_0);
            p16_1 = _mm256_madd_epi16(scale_1, p16_1);

            sumi = mm256_جوڑو_epi32(sumi, mm256_جوڑو_epi32(p16_0, p16_1));

        }

        __m256 vd = _mm256_set1_ps(d);
        acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);

    }

    *s = hsum_float_8(acc) + summs;

#وگرنہ_اگر defined __AVX__

    ثابت __m128i m4 = _mm_set1_epi8(0xF);
    ثابت __m128i mzero = _mm_setzero_si128();
    ثابت __m128i mone  = _mm_set1_epi8(1);
    ثابت __m128i m2 = _mm_set1_epi8(2);

    __m256 acc = _mm256_setzero_ps();

    اعشاری summs = 0.f;

    برائے (صحیح i = 0; i < nb; ++i) {

        ثابت اعشاری d = y[i].d * GGML_CPU_FP16_مقصد_FP32(x[i].d);
        ثابت اعشاری dmin = -y[i].d * GGML_CPU_FP16_مقصد_FP32(x[i].dmin);

        ثابت uint8_t * GGML_RESTRICT q5 = x[i].qs;
        ثابت int8_t  * GGML_RESTRICT q8 = y[i].qs;

        حافظہ_نقل(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        ثابت uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        ثابت __m128i utmps = mm_سیٹ_epi32(utmp[3], utmp[2], utmp[1], utmp[0]);
        ثابت __m128i scales = _mm_cvtepu8_epi16(utmps);
        ثابت __m128i mins = _mm_cvtepu8_epi16(_mm_unpackhi_epi64(utmps, utmps));

        ثابت __m128i q8sums_0 = _mm_loadu_si128((ثابت __m128i*)&y[i].bsums[0]);
        ثابت __m128i q8sums_1 = _mm_loadu_si128((ثابت __m128i*)&y[i].bsums[8]);
        ثابت __m128i q8s = _mm_hadd_epi16(q8sums_0, q8sums_1);
        ثابت __m128i prod = _mm_madd_epi16(mins, q8s);
        ثابت __m128i hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
        summs += dmin * _mm_extract_epi32(hsum, 0);

        ثابت __m128i hbits_0 = _mm_loadu_si128((ثابت __m128i*)&x[i].qh[0]);
        ثابت __m128i hbits_1 = _mm_loadu_si128((ثابت __m128i*)&x[i].qh[16]);
        __m128i hmask = mone;

        __m128i sumi_0 = _mm_setzero_si128();
        __m128i sumi_1 = _mm_setzero_si128();

        صحیح ثنائی_ہندسہ = 0;

        __m128i shuffle = _mm_set1_epi16(0x0100);
        برائے (صحیح j = 0; j < QK_K/64; ++j) {

            ثابت __m128i scale_0 = _mm_shuffle_epi8(scales, shuffle);
            shuffle = mm_جوڑو_epi16(shuffle, m2);
            ثابت __m128i scale_1 = _mm_shuffle_epi8(scales, shuffle);
            shuffle = mm_جوڑو_epi16(shuffle, m2);

            ثابت __m128i q5bits_0 = _mm_loadu_si128((ثابت __m128i*)q5); q5 += 16;
            ثابت __m128i q5bits_1 = _mm_loadu_si128((ثابت __m128i*)q5); q5 += 16;

            __m128i q5l_0 = _mm_and_si128(q5bits_0, m4);
            __m128i q5l_1 = _mm_and_si128(q5bits_1, m4);
            __m128i q5h_0 = _mm_slli_epi16(_mm_srli_epi16(_mm_and_si128(hbits_0, hmask), ثنائی_ہندسہ), 4);
            __m128i q5h_1 = _mm_slli_epi16(_mm_srli_epi16(_mm_and_si128(hbits_1, hmask), ثنائی_ہندسہ++), 4);
            __m128i q5_0  = mm_جوڑو_epi8(q5l_0, q5h_0);
            __m128i q5_1  = mm_جوڑو_epi8(q5l_1, q5h_1);
            hmask = _mm_slli_epi16(hmask, 1);

            __m128i q8_0 = _mm_loadu_si128((ثابت __m128i*)q8); q8 += 16;
            __m128i q8_1 = _mm_loadu_si128((ثابت __m128i*)q8); q8 += 16;
            __m128i p16_0 = _mm_maddubs_epi16(q5_0, q8_0);
            __m128i p16_1 = _mm_maddubs_epi16(q5_1, q8_1);
            p16_0 = _mm_madd_epi16(scale_0, p16_0);
            p16_1 = _mm_madd_epi16(scale_0, p16_1);

            q5l_0 = _mm_and_si128(_mm_srli_epi16(q5bits_0, 4), m4);
            q5l_1 = _mm_and_si128(_mm_srli_epi16(q5bits_1, 4), m4);
            q5h_0 = _mm_slli_epi16(_mm_srli_epi16(_mm_and_si128(hbits_0, hmask), ثنائی_ہندسہ), 4);
            q5h_1 = _mm_slli_epi16(_mm_srli_epi16(_mm_and_si128(hbits_1, hmask), ثنائی_ہندسہ++), 4);
            q5_0  = mm_جوڑو_epi8(q5l_0, q5h_0);
            q5_1  = mm_جوڑو_epi8(q5l_1, q5h_1);
            hmask = _mm_slli_epi16(hmask, 1);

            q8_0 = _mm_loadu_si128((ثابت __m128i*)q8); q8 += 16;
            q8_1 = _mm_loadu_si128((ثابت __m128i*)q8); q8 += 16;
            __m128i p16_2 = _mm_maddubs_epi16(q5_0, q8_0);
            __m128i p16_3 = _mm_maddubs_epi16(q5_1, q8_1);
            p16_2 = _mm_madd_epi16(scale_1, p16_2);
            p16_3 = _mm_madd_epi16(scale_1, p16_3);

            sumi_0 = mm_جوڑو_epi32(sumi_0, mm_جوڑو_epi32(p16_0, p16_2));
            sumi_1 = mm_جوڑو_epi32(sumi_1, mm_جوڑو_epi32(p16_1, p16_3));

        }

        __m256 vd = _mm256_set1_ps(d);
        __m256i sumi = MM256_سیٹ_M128_I(sumi_1, sumi_0);
        acc = mm256_جوڑو_ps(_mm256_mul_ps(vd, _mm256_cvtepi32_ps(sumi)), acc);

    }

    *s = hsum_float_8(acc) + summs;

#ورنہ
    غیر_مستعمل(x);
    غیر_مستعمل(y);
    غیر_مستعمل(nb);
    غیر_مستعمل(kmask1);
    غیر_مستعمل(kmask2);
    غیر_مستعمل(kmask3);
    غیر_مستعمل(utmp);
    ggml_vec_dot_q5_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#اگر_ختم
}

خالی ggml_vec_dot_q6_K_q8_K(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc) {
    أكد(n % QK_K == 0);
    أكد(nrc == 1);
    غیر_مستعمل(nrc);
    غیر_مستعمل(bx);
    غیر_مستعمل(by);
    غیر_مستعمل(bs);

    ثابت بلاک_q6_K * GGML_RESTRICT x = vx;
    ثابت بلاک_q8_K * GGML_RESTRICT y = vy;

    ثابت صحیح nb = n / QK_K;

#اگر defined __AVX2__

    ثابت __m256i m3 = _mm256_set1_epi8(3);
    ثابت __m256i m15 = _mm256_set1_epi8(15);

    __m256 acc = _mm256_setzero_ps();

    برائے (صحیح i = 0; i < nb; ++i) {

        ثابت اعشاری d = y[i].d * GGML_CPU_FP16_مقصد_FP32(x[i].d);

        ثابت uint8_t * GGML_RESTRICT q4 = x[i].ql;
        ثابت uint8_t * GGML_RESTRICT qh = x[i].qh;
        ثابت int8_t  * GGML_RESTRICT q8 = y[i].qs;

        ثابت __m256i q8sums = _mm256_loadu_si256((ثابت __m256i*)y[i].bsums);
        ثابت __m128i scales = _mm_loadu_si128((ثابت __m128i*)x[i].scales);
        ثابت __m256i scales_16 = _mm256_cvtepi8_epi16(scales);
        ثابت __m256i q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales_16), 5);

        __m256i sumi = _mm256_setzero_si256();

        صحیح is = 0;

        برائے (صحیح j = 0; j < QK_K/128; ++j) {
            ثابت __m256i q4bits1 = _mm256_loadu_si256((ثابت __m256i*)q4); q4 += 32;
            ثابت __m256i q4bits2 = _mm256_loadu_si256((ثابت __m256i*)q4); q4 += 32;
            ثابت __m256i q4bitsH = _mm256_loadu_si256((ثابت __m256i*)qh); qh += 32;

            ثابت __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m3), 4);
            ثابت __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(12)), 2);
            ثابت __m256i q4h_2 = _mm256_and_si256(q4bitsH, _mm256_set1_epi8(48));
            ثابت __m256i q4h_3 = _mm256_srli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(-64)), 2);

            ثابت __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
            ثابت __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
            ثابت __m256i q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
            ثابت __m256i q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h_3);

            ثابت __m256i q8_0 = _mm256_loadu_si256((ثابت __m256i*)q8); q8 += 32;
            ثابت __m256i q8_1 = _mm256_loadu_si256((ثابت __m256i*)q8); q8 += 32;
            ثابت __m256i q8_2 = _mm256_loadu_si256((ثابت __m256i*)q8); q8 += 32;
            ثابت __m256i q8_3 = _mm256_loadu_si256((ثابت __m256i*)q8); q8 += 32;

            __m256i p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
            __m256i p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
            __m256i p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
            __m256i p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

            ثابت __m128i scale_0 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 0));
            ثابت __m128i scale_1 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 1));
            ثابت __m128i scale_2 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 2));
            ثابت __m128i scale_3 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 3));
            is += 4;

            p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
            p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
            p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
            p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);

            sumi = mm256_جوڑو_epi32(sumi, mm256_جوڑو_epi32(p16_0, p16_1));
            sumi = mm256_جوڑو_epi32(sumi, mm256_جوڑو_epi32(p16_2, p16_3));

        }

        sumi = _mm256_sub_epi32(sumi, q8sclsub);
        acc = _mm256_fmadd_ps(mm256_نشر_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }

    *s = hsum_float_8(acc);

#وگرنہ_اگر defined __AVX__

    ثابت __m128i m3 = _mm_set1_epi8(3);
    ثابت __m128i m15 = _mm_set1_epi8(15);

    __m256 acc = _mm256_setzero_ps();

    برائے (صحیح i = 0; i < nb; ++i) {

        ثابت اعشاری d = y[i].d * GGML_CPU_FP16_مقصد_FP32(x[i].d);

        ثابت uint8_t * GGML_RESTRICT q4 = x[i].ql;
        ثابت uint8_t * GGML_RESTRICT qh = x[i].qh;
        ثابت int8_t  * GGML_RESTRICT q8 = y[i].qs;

        // handle the q6_k -32 offset separately using bsums
        ثابت __m128i q8sums_0 = _mm_loadu_si128((ثابت __m128i*)y[i].bsums);
        ثابت __m128i q8sums_1 = _mm_loadu_si128((ثابت __m128i*)y[i].bsums + 1);
        ثابت __m128i scales = _mm_loadu_si128((ثابت __m128i*)x[i].scales);
        ثابت __m128i scales_16_0 = _mm_cvtepi8_epi16(scales);
        ثابت __m128i scales_16_1 = _mm_cvtepi8_epi16(_mm_bsrli_si128(scales, 8));
        ثابت __m128i q8sclsub_0 = _mm_slli_epi32(_mm_madd_epi16(q8sums_0, scales_16_0), 5);
        ثابت __m128i q8sclsub_1 = _mm_slli_epi32(_mm_madd_epi16(q8sums_1, scales_16_1), 5);

        __m128i sumi_0 = _mm_setzero_si128();
        __m128i sumi_1 = _mm_setzero_si128();

        صحیح is = 0;

        برائے (صحیح j = 0; j < QK_K/128; ++j) {

            ثابت __m128i q4bitsH_0 = _mm_loadu_si128((ثابت __m128i*)qh); qh += 16;
            ثابت __m128i q4bitsH_1 = _mm_loadu_si128((ثابت __m128i*)qh); qh += 16;

            ثابت __m128i q4h_0 = _mm_slli_epi16(_mm_and_si128(q4bitsH_0, m3), 4);
            ثابت __m128i q4h_1 = _mm_slli_epi16(_mm_and_si128(q4bitsH_1, m3), 4);
            ثابت __m128i q4h_2 = _mm_slli_epi16(_mm_and_si128(q4bitsH_0, _mm_set1_epi8(12)), 2);
            ثابت __m128i q4h_3 = _mm_slli_epi16(_mm_and_si128(q4bitsH_1, _mm_set1_epi8(12)), 2);
            ثابت __m128i q4h_4 = _mm_and_si128(q4bitsH_0, _mm_set1_epi8(48));
            ثابت __m128i q4h_5 = _mm_and_si128(q4bitsH_1, _mm_set1_epi8(48));
            ثابت __m128i q4h_6 = _mm_srli_epi16(_mm_and_si128(q4bitsH_0, _mm_set1_epi8(-64)), 2);
            ثابت __m128i q4h_7 = _mm_srli_epi16(_mm_and_si128(q4bitsH_1, _mm_set1_epi8(-64)), 2);

            ثابت __m128i q4bits1_0 = _mm_loadu_si128((ثابت __m128i*)q4); q4 += 16;
            ثابت __m128i q4bits1_1 = _mm_loadu_si128((ثابت __m128i*)q4); q4 += 16;
            ثابت __m128i q4bits2_0 = _mm_loadu_si128((ثابت __m128i*)q4); q4 += 16;
            ثابت __m128i q4bits2_1 = _mm_loadu_si128((ثابت __m128i*)q4); q4 += 16;

            ثابت __m128i q4_0 = _mm_or_si128(_mm_and_si128(q4bits1_0, m15), q4h_0);
            ثابت __m128i q4_1 = _mm_or_si128(_mm_and_si128(q4bits1_1, m15), q4h_1);
            ثابت __m128i q4_2 = _mm_or_si128(_mm_and_si128(q4bits2_0, m15), q4h_2);
            ثابت __m128i q4_3 = _mm_or_si128(_mm_and_si128(q4bits2_1, m15), q4h_3);
            ثابت __m128i q4_4 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits1_0, 4), m15), q4h_4);
            ثابت __m128i q4_5 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits1_1, 4), m15), q4h_5);
            ثابت __m128i q4_6 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits2_0, 4), m15), q4h_6);
            ثابت __m128i q4_7 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits2_1, 4), m15), q4h_7);

            ثابت __m128i q8_0 = _mm_loadu_si128((ثابت __m128i*)q8); q8 += 16;
            ثابت __m128i q8_1 = _mm_loadu_si128((ثابت __m128i*)q8); q8 += 16;
            ثابت __m128i q8_2 = _mm_loadu_si128((ثابت __m128i*)q8); q8 += 16;
            ثابت __m128i q8_3 = _mm_loadu_si128((ثابت __m128i*)q8); q8 += 16;
            ثابت __m128i q8_4 = _mm_loadu_si128((ثابت __m128i*)q8); q8 += 16;
            ثابت __m128i q8_5 = _mm_loadu_si128((ثابت __m128i*)q8); q8 += 16;
            ثابت __m128i q8_6 = _mm_loadu_si128((ثابت __m128i*)q8); q8 += 16;
            ثابت __m128i q8_7 = _mm_loadu_si128((ثابت __m128i*)q8); q8 += 16;

            __m128i p16_0 = _mm_maddubs_epi16(q4_0, q8_0);
            __m128i p16_1 = _mm_maddubs_epi16(q4_1, q8_1);
            __m128i p16_2 = _mm_maddubs_epi16(q4_2, q8_2);
            __m128i p16_3 = _mm_maddubs_epi16(q4_3, q8_3);
            __m128i p16_4 = _mm_maddubs_epi16(q4_4, q8_4);
            __m128i p16_5 = _mm_maddubs_epi16(q4_5, q8_5);
            __m128i p16_6 = _mm_maddubs_epi16(q4_6, q8_6);
            __m128i p16_7 = _mm_maddubs_epi16(q4_7, q8_7);

            ثابت __m128i scale_0 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 0));
            ثابت __m128i scale_1 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 1));
            ثابت __m128i scale_2 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 2));
            ثابت __m128i scale_3 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 3));
            is += 4;

            p16_0 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_0), p16_0);
            p16_1 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_bsrli_si128(scale_0, 8)), p16_1);
            p16_2 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_1), p16_2);
            p16_3 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_bsrli_si128(scale_1, 8)), p16_3);
            p16_4 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_2), p16_4);
            p16_5 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_bsrli_si128(scale_2, 8)), p16_5);
            p16_6 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_3), p16_6);
            p16_7 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_bsrli_si128(scale_3, 8)), p16_7);

            sumi_0 = mm_جوڑو_epi32(sumi_0, mm_جوڑو_epi32(p16_0, p16_2));
            sumi_1 = mm_جوڑو_epi32(sumi_1, mm_جوڑو_epi32(p16_1, p16_3));
            sumi_0 = mm_جوڑو_epi32(sumi_0, mm_جوڑو_epi32(p16_4, p16_6));
            sumi_1 = mm_جوڑو_epi32(sumi_1, mm_جوڑو_epi32(p16_5, p16_7));

        }

        sumi_0 = _mm_sub_epi32(sumi_0, q8sclsub_0);
        sumi_1 = _mm_sub_epi32(sumi_1, q8sclsub_1);
        ثابت __m256i sumi = MM256_سیٹ_M128_I(sumi_1, sumi_0);
        acc = mm256_جوڑو_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi)), acc);
    }

    *s = hsum_float_8(acc);

#ورنہ
    غیر_مستعمل(x);
    غیر_مستعمل(y);
    غیر_مستعمل(nb);
    ggml_vec_dot_q6_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#اگر_ختم
}

#اگر defined (__AVX__) || defined (__AVX2__)
ساکن ثابت int8_t keven_signs_q2xs[1024] = {
     1,  1,  1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1,  1,  1, -1,  1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1,  1,
     1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1, -1,
     1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1,  1,  1,  1, -1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1, -1,
     1,  1, -1, -1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1, -1,  1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1,  1,
     1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1,  1, -1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1, -1,
     1,  1, -1,  1, -1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1,  1,
     1,  1,  1, -1, -1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1,  1,
     1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1, -1,
     1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1,  1, -1,  1,  1,  1, -1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1,  1, -1,  1, -1,
     1,  1, -1,  1,  1, -1,  1,  1, -1,  1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1,  1,
     1,  1,  1, -1,  1, -1,  1,  1, -1,  1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1,  1,
     1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1,  1,  1, -1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1, -1,
     1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1,  1,
     1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1,  1,  1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1, -1,
     1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1,  1,  1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1, -1,
     1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1, -1,  1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1,  1,
     1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1,  1, -1, -1,
     1,  1, -1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1,  1, -1, -1,  1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1,  1,
     1,  1,  1, -1,  1,  1, -1,  1, -1,  1,  1, -1,  1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1,  1,
     1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1,  1,  1, -1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1, -1,
     1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1,  1,
     1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1,  1,  1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1, -1,
     1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1, -1,
     1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1, -1,  1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1,  1,
     1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1,  1, -1, -1, -1,  1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1,  1,
     1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1, -1,
     1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1, -1,
     1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1, -1,  1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1,  1,
     1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1,  1,  1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1, -1,
     1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1, -1,  1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1,  1,
     1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1, -1,  1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1,  1,
     1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1,  1,  1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1, -1,
};
#اگر_ختم

خالی ggml_vec_dot_iq2_xxs_q8_K(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc) {
    أكد(n % QK_K == 0);
    أكد(nrc == 1);
    غیر_مستعمل(nrc);
    غیر_مستعمل(bx);
    غیر_مستعمل(by);
    غیر_مستعمل(bs);

    ثابت بلاک_iq2_xxs * GGML_RESTRICT x = vx;
    ثابت بلاک_q8_K    * GGML_RESTRICT y = vy;

    ثابت صحیح nb = n / QK_K;

#اگر defined(__AVX2__)

    ثابت uint64_t * signs64 = (ثابت uint64_t *)keven_signs_q2xs;

    uint32_t aux32[4];
    ثابت uint8_t * aux8 = (ثابت uint8_t *)aux32;

    __m256 accumf = _mm256_setzero_ps();
    برائے (صحیح i = 0; i < nb; ++i) {
        ثابت اعشاری d = GGML_CPU_FP16_مقصد_FP32(x[i].d) * y[i].d;
        ثابت uint16_t * GGML_RESTRICT q2 = x[i].qs;
        ثابت int8_t   * GGML_RESTRICT q8 = y[i].qs;
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        برائے (صحیح ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            ثابت __m256i q8_1 = _mm256_loadu_si256((ثابت __m256i *)q8); q8 += 32;
            ثابت __m256i q8_2 = _mm256_loadu_si256((ثابت __m256i *)q8); q8 += 32;
            حافظہ_نقل(aux32, q2, 4*حجم(uint32_t)); q2 += 8;
            ثابت __m256i q2_1 = mm256_سیٹ_epi64x(iq2xxs_grid[aux8[ 3]], iq2xxs_grid[aux8[ 2]], iq2xxs_grid[aux8[1]], iq2xxs_grid[aux8[0]]);
            ثابت __m256i q2_2 = mm256_سیٹ_epi64x(iq2xxs_grid[aux8[11]], iq2xxs_grid[aux8[10]], iq2xxs_grid[aux8[9]], iq2xxs_grid[aux8[8]]);
            ثابت __m256i s2_1 = mm256_سیٹ_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127],
                                                   signs64[(aux32[1] >>  7) & 127], signs64[(aux32[1] >>  0) & 127]);
            ثابت __m256i s2_2 = mm256_سیٹ_epi64x(signs64[(aux32[3] >> 21) & 127], signs64[(aux32[3] >> 14) & 127],
                                                   signs64[(aux32[3] >>  7) & 127], signs64[(aux32[3] >>  0) & 127]);
            ثابت __m256i q8s_1 = _mm256_sign_epi8(q8_1, s2_1);
            ثابت __m256i q8s_2 = _mm256_sign_epi8(q8_2, s2_2);
            ثابت __m256i dot1  = _mm256_maddubs_epi16(q2_1, q8s_1);
            ثابت __m256i dot2  = _mm256_maddubs_epi16(q2_2, q8s_2);
            ثابت uint16_t ls1 = aux32[1] >> 28;
            ثابت uint16_t ls2 = aux32[3] >> 28;
            ثابت __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16(2*ls1+1));
            ثابت __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16(2*ls2+1));
            sumi1 = mm256_جوڑو_epi32(sumi1, p1);
            sumi2 = mm256_جوڑو_epi32(sumi2, p2);
        }

        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(mm256_جوڑو_epi32(sumi1, sumi2)), accumf);

    }

    *s = 0.125f * hsum_float_8(accumf);

#وگرنہ_اگر defined(__AVX__)
    ثابت uint64_t * signs64 = (ثابت uint64_t *)keven_signs_q2xs;

    uint32_t aux32[4];
    ثابت uint8_t * aux8 = (ثابت uint8_t *)aux32;

    __m256 accumf = _mm256_setzero_ps();
    برائے (صحیح i = 0; i < nb; ++i) {
        ثابت اعشاری d = GGML_CPU_FP16_مقصد_FP32(x[i].d) * y[i].d;
        ثابت uint16_t * GGML_RESTRICT q2 = x[i].qs;
        ثابت int8_t   * GGML_RESTRICT q8 = y[i].qs;
        __m128i sumi1_0 = _mm_setzero_si128();
        __m128i sumi1_1 = _mm_setzero_si128();
        __m128i sumi2_0 = _mm_setzero_si128();
        __m128i sumi2_1 = _mm_setzero_si128();
        برائے (صحیح ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            ثابت __m128i q8_1_0 = _mm_loadu_si128((ثابت __m128i *)q8); q8 += 16;
            ثابت __m128i q8_1_1 = _mm_loadu_si128((ثابت __m128i *)q8); q8 += 16;
            ثابت __m128i q8_2_0 = _mm_loadu_si128((ثابت __m128i *)q8); q8 += 16;
            ثابت __m128i q8_2_1 = _mm_loadu_si128((ثابت __m128i *)q8); q8 += 16;
            حافظہ_نقل(aux32, q2, 4*حجم(uint32_t)); q2 += 8;
            ثابت __m128i q2_1_0 = mm_سیٹ_epi64x(iq2xxs_grid[aux8[1]], iq2xxs_grid[aux8[0]]);
            ثابت __m128i q2_1_1 = mm_سیٹ_epi64x(iq2xxs_grid[aux8[3]], iq2xxs_grid[aux8[2]]);
            ثابت __m128i q2_2_0 = mm_سیٹ_epi64x(iq2xxs_grid[aux8[9]], iq2xxs_grid[aux8[8]]);
            ثابت __m128i q2_2_1 = mm_سیٹ_epi64x(iq2xxs_grid[aux8[11]], iq2xxs_grid[aux8[10]]);
            ثابت __m128i s2_1_0 = mm_سیٹ_epi64x(signs64[(aux32[1] >>  7) & 127], signs64[(aux32[1] >>  0) & 127]);
            ثابت __m128i s2_1_1 = mm_سیٹ_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127]);
            ثابت __m128i s2_2_0 = mm_سیٹ_epi64x(signs64[(aux32[3] >>  7) & 127], signs64[(aux32[3] >>  0) & 127]);
            ثابت __m128i s2_2_1 = mm_سیٹ_epi64x(signs64[(aux32[3] >> 21) & 127], signs64[(aux32[3] >> 14) & 127]);
            ثابت __m128i q8s_1_0 = _mm_sign_epi8(q8_1_0, s2_1_0);
            ثابت __m128i q8s_1_1 = _mm_sign_epi8(q8_1_1, s2_1_1);
            ثابت __m128i q8s_2_0 = _mm_sign_epi8(q8_2_0, s2_2_0);
            ثابت __m128i q8s_2_1 = _mm_sign_epi8(q8_2_1, s2_2_1);
            ثابت __m128i dot1_0  = _mm_maddubs_epi16(q2_1_0, q8s_1_0);
            ثابت __m128i dot1_1  = _mm_maddubs_epi16(q2_1_1, q8s_1_1);
            ثابت __m128i dot2_0  = _mm_maddubs_epi16(q2_2_0, q8s_2_0);
            ثابت __m128i dot2_1  = _mm_maddubs_epi16(q2_2_1, q8s_2_1);
            ثابت uint16_t ls1 = aux32[1] >> 28;
            ثابت uint16_t ls2 = aux32[3] >> 28;
            ثابت __m128i p1_0 = _mm_madd_epi16(dot1_0, _mm_set1_epi16(2*ls1+1));
            ثابت __m128i p1_1 = _mm_madd_epi16(dot1_1, _mm_set1_epi16(2*ls1+1));
            ثابت __m128i p2_0 = _mm_madd_epi16(dot2_0, _mm_set1_epi16(2*ls2+1));
            ثابت __m128i p2_1 = _mm_madd_epi16(dot2_1, _mm_set1_epi16(2*ls2+1));
            sumi1_0 = mm_جوڑو_epi32(sumi1_0, p1_0);
            sumi1_1 = mm_جوڑو_epi32(sumi1_1, p1_1);
            sumi2_0 = mm_جوڑو_epi32(sumi2_0, p2_0);
            sumi2_1 = mm_جوڑو_epi32(sumi2_1, p2_1);
        }

        accumf = mm256_جوڑو_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(MM256_سیٹ_M128_I(mm_جوڑو_epi32(sumi1_1, sumi2_1), mm_جوڑو_epi32(sumi1_0, sumi2_0)))), accumf);

    }

    *s = 0.125f * hsum_float_8(accumf);

#ورنہ
    غیر_مستعمل(x);
    غیر_مستعمل(y);
    غیر_مستعمل(nb);
    ggml_vec_dot_iq2_xxs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#اگر_ختم
}

خالی ggml_vec_dot_iq2_xs_q8_K(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc) {
    أكد(n % QK_K == 0);
    أكد(nrc == 1);
    غیر_مستعمل(nrc);
    غیر_مستعمل(bx);
    غیر_مستعمل(by);
    غیر_مستعمل(bs);

    ثابت بلاک_iq2_xs * GGML_RESTRICT x = vx;
    ثابت بلاک_q8_K   * GGML_RESTRICT y = vy;

    ثابت صحیح nb = n / QK_K;

#اگر defined(__AVX2__)

    ثابت __m256i mone = _mm256_set1_epi8(1);
    ساکن ثابت حرف بلاک_sign_shuffle_نقاب_1[32] = {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
        0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06,
    };
    ساکن ثابت حرف بلاک_sign_shuffle_نقاب_2[32] = {
        0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a,
        0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e,
    };
    ساکن ثابت uint8_t ثنائی_ہندسہ_selector_نقاب_بائٹیں[32] = {
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };

    ثابت __m256i ثنائی_ہندسہ_selector_نقاب = _mm256_loadu_si256((ثابت __m256i*)ثنائی_ہندسہ_selector_نقاب_بائٹیں);
    ثابت __m256i بلاک_sign_shuffle_1 = _mm256_loadu_si256((ثابت __m256i*)بلاک_sign_shuffle_نقاب_1);
    ثابت __m256i بلاک_sign_shuffle_2 = _mm256_loadu_si256((ثابت __m256i*)بلاک_sign_shuffle_نقاب_2);

    ساکن ثابت uint8_t k_ثنائی_ہندسہ_helper[32] = {
        0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
        0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
    };
    ثابت __m256i ثنائی_ہندسہ_helper = _mm256_loadu_si256((ثابت __m256i*)k_ثنائی_ہندسہ_helper);
    ثابت __m256i m511 = _mm256_set1_epi16(511);
    ثابت __m128i m4 = _mm_set1_epi8(0xf);
    ثابت __m128i m1 = _mm_set1_epi8(1);

    uint64_t aux64;

    // somewhat hacky, but gives a significant boost in performance
    __m256i aux_gindex;
    ثابت uint16_t * gindex = (ثابت uint16_t *)&aux_gindex;

    __m256 accumf = _mm256_setzero_ps();
    برائے (صحیح i = 0; i < nb; ++i) {
        ثابت اعشاری d = GGML_CPU_FP16_مقصد_FP32(x[i].d) * y[i].d;
        ثابت uint16_t * GGML_RESTRICT q2 = x[i].qs;
        ثابت int8_t   * GGML_RESTRICT q8 = y[i].qs;

        حافظہ_نقل(&aux64, x[i].scales, 8);
        __m128i stmp = _mm_set1_epi64x(aux64);
        stmp = _mm_unpacklo_epi8(_mm_and_si128(stmp, m4), _mm_and_si128(_mm_srli_epi16(stmp, 4), m4));
        ثابت __m128i scales = mm_جوڑو_epi8(_mm_slli_epi16(stmp, 1), m1);

        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        برائے (صحیح ib32 = 0; ib32 < QK_K/32; ib32 += 4) {

            ثابت __m256i q2_ڈیٹا = _mm256_loadu_si256((ثابت __m256i*)q2);  q2 += 16;
            aux_gindex = _mm256_and_si256(q2_ڈیٹا, m511);

            ثابت __m256i partial_sign_بٹ = _mm256_srli_epi16(q2_ڈیٹا, 9);
            ثابت __m256i partial_sign_بٹ_بالائی = _mm256_srli_epi16(q2_ڈیٹا, 13);
            ثابت __m256i partial_sign_بٹ_for_counting = _mm256_xor_si256(partial_sign_بٹ, partial_sign_بٹ_بالائی);

            ثابت __m256i odd_بٹ = _mm256_shuffle_epi8(ثنائی_ہندسہ_helper, partial_sign_بٹ_for_counting);
            ثابت __m256i بھرا_sign_بٹ = _mm256_or_si256(partial_sign_بٹ, odd_بٹ);

            ثابت __m256i q8_1 = _mm256_loadu_si256((ثابت __m256i *)q8); q8 += 32;
            ثابت __m256i q8_2 = _mm256_loadu_si256((ثابت __m256i *)q8); q8 += 32;
            ثابت __m256i q8_3 = _mm256_loadu_si256((ثابت __m256i *)q8); q8 += 32;
            ثابت __m256i q8_4 = _mm256_loadu_si256((ثابت __m256i *)q8); q8 += 32;

            ثابت __m256i q2_1 = mm256_سیٹ_epi64x(iq2xs_grid[gindex[ 3]], iq2xs_grid[gindex[ 2]],
                                                   iq2xs_grid[gindex[ 1]], iq2xs_grid[gindex[ 0]]);
            ثابت __m256i q2_2 = mm256_سیٹ_epi64x(iq2xs_grid[gindex[ 7]], iq2xs_grid[gindex[ 6]],
                                                   iq2xs_grid[gindex[ 5]], iq2xs_grid[gindex[ 4]]);
            ثابت __m256i q2_3 = mm256_سیٹ_epi64x(iq2xs_grid[gindex[11]], iq2xs_grid[gindex[10]],
                                                   iq2xs_grid[gindex[ 9]], iq2xs_grid[gindex[ 8]]);
            ثابت __m256i q2_4 = mm256_سیٹ_epi64x(iq2xs_grid[gindex[15]], iq2xs_grid[gindex[14]],
                                                   iq2xs_grid[gindex[13]], iq2xs_grid[gindex[12]]);

            ثابت __m128i بھرا_signs_l = _mm256_castsi256_si128(بھرا_sign_بٹ);
            ثابت __m128i بھرا_signs_h = _mm256_extractf128_si256(بھرا_sign_بٹ, 1);
            ثابت __m256i بھرا_signs_1 = MM256_سیٹ_M128_I(بھرا_signs_l, بھرا_signs_l);
            ثابت __m256i بھرا_signs_2 = MM256_سیٹ_M128_I(بھرا_signs_h, بھرا_signs_h);

            __m256i signs;
            signs = _mm256_shuffle_epi8(بھرا_signs_1, بلاک_sign_shuffle_1);
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, ثنائی_ہندسہ_selector_نقاب), ثنائی_ہندسہ_selector_نقاب);
            ثابت __m256i q8s_1 = _mm256_sign_epi8(q8_1, _mm256_or_si256(signs, mone));

            signs = _mm256_shuffle_epi8(بھرا_signs_1, بلاک_sign_shuffle_2);
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, ثنائی_ہندسہ_selector_نقاب), ثنائی_ہندسہ_selector_نقاب);
            ثابت __m256i q8s_2 = _mm256_sign_epi8(q8_2, _mm256_or_si256(signs, mone));

            signs = _mm256_shuffle_epi8(بھرا_signs_2, بلاک_sign_shuffle_1);
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, ثنائی_ہندسہ_selector_نقاب), ثنائی_ہندسہ_selector_نقاب);
            ثابت __m256i q8s_3 = _mm256_sign_epi8(q8_3, _mm256_or_si256(signs, mone));

            signs = _mm256_shuffle_epi8(بھرا_signs_2, بلاک_sign_shuffle_2);
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, ثنائی_ہندسہ_selector_نقاب), ثنائی_ہندسہ_selector_نقاب);
            ثابت __m256i q8s_4 = _mm256_sign_epi8(q8_4, _mm256_or_si256(signs, mone));

            ثابت __m256i dot1  = _mm256_maddubs_epi16(q2_1, q8s_1);
            ثابت __m256i dot2  = _mm256_maddubs_epi16(q2_2, q8s_2);
            ثابت __m256i dot3  = _mm256_maddubs_epi16(q2_3, q8s_3);
            ثابت __m256i dot4  = _mm256_maddubs_epi16(q2_4, q8s_4);

            ثابت __m256i sc1 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32+0)));
            ثابت __m256i sc2 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32+1)));
            ثابت __m256i sc3 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32+2)));
            ثابت __m256i sc4 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32+3)));

            sumi1 = mm256_جوڑو_epi32(sumi1, _mm256_madd_epi16(dot1, sc1));
            sumi2 = mm256_جوڑو_epi32(sumi2, _mm256_madd_epi16(dot2, sc2));
            sumi1 = mm256_جوڑو_epi32(sumi1, _mm256_madd_epi16(dot3, sc3));
            sumi2 = mm256_جوڑو_epi32(sumi2, _mm256_madd_epi16(dot4, sc4));
        }

        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(mm256_جوڑو_epi32(sumi1, sumi2)), accumf);

    }

    *s = 0.125f * hsum_float_8(accumf);

#وگرنہ_اگر defined(__AVX__)
    ثابت __m128i mone = _mm_set1_epi8(1);
    ساکن ثابت حرف بلاک_sign_shuffle_نقاب_1[32] = {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
        0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06,
    };
    ساکن ثابت حرف بلاک_sign_shuffle_نقاب_2[32] = {
        0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a,
        0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e,
    };
    ساکن ثابت uint8_t ثنائی_ہندسہ_selector_نقاب_بائٹیں[32] = {
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };

    ثابت __m128i ثنائی_ہندسہ_selector_نقاب_0 = _mm_loadu_si128((ثابت __m128i*)ثنائی_ہندسہ_selector_نقاب_بائٹیں);
    ثابت __m128i ثنائی_ہندسہ_selector_نقاب_1 = _mm_loadu_si128((ثابت __m128i*)ثنائی_ہندسہ_selector_نقاب_بائٹیں + 1);
    ثابت __m128i بلاک_sign_shuffle_1_0 = _mm_loadu_si128((ثابت __m128i*)بلاک_sign_shuffle_نقاب_1);
    ثابت __m128i بلاک_sign_shuffle_1_1 = _mm_loadu_si128((ثابت __m128i*)بلاک_sign_shuffle_نقاب_1 + 1);
    ثابت __m128i بلاک_sign_shuffle_2_0 = _mm_loadu_si128((ثابت __m128i*)بلاک_sign_shuffle_نقاب_2);
    ثابت __m128i بلاک_sign_shuffle_2_1 = _mm_loadu_si128((ثابت __m128i*)بلاک_sign_shuffle_نقاب_2 + 1);

    ساکن ثابت uint8_t k_ثنائی_ہندسہ_helper[32] = {
        0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
        0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
    };
    ثابت __m128i ثنائی_ہندسہ_helper_0 = _mm_loadu_si128((ثابت __m128i*)k_ثنائی_ہندسہ_helper);
    ثابت __m128i ثنائی_ہندسہ_helper_1 = _mm_loadu_si128((ثابت __m128i*)k_ثنائی_ہندسہ_helper + 1);
    ثابت __m128i m511 = _mm_set1_epi16(511);
    ثابت __m128i m4 = _mm_set1_epi8(0xf);
    ثابت __m128i m1 = _mm_set1_epi8(1);

    uint64_t aux64;

    // somewhat hacky, but gives a significant boost in performance
    __m256i aux_gindex;
    ثابت uint16_t * gindex = (ثابت uint16_t *)&aux_gindex;

    __m256 accumf = _mm256_setzero_ps();
    برائے (صحیح i = 0; i < nb; ++i) {
        ثابت اعشاری d = GGML_CPU_FP16_مقصد_FP32(x[i].d) * y[i].d;
        ثابت uint16_t * GGML_RESTRICT q2 = x[i].qs;
        ثابت int8_t   * GGML_RESTRICT q8 = y[i].qs;

        حافظہ_نقل(&aux64, x[i].scales, 8);
        __m128i stmp = _mm_set1_epi64x(aux64);
        stmp = _mm_unpacklo_epi8(_mm_and_si128(stmp, m4), _mm_and_si128(_mm_srli_epi16(stmp, 4), m4));
        ثابت __m128i scales = mm_جوڑو_epi8(_mm_slli_epi16(stmp, 1), m1);

        __m128i sumi1_0 = _mm_setzero_si128();
        __m128i sumi1_1 = _mm_setzero_si128();
        __m128i sumi2_0 = _mm_setzero_si128();
        __m128i sumi2_1 = _mm_setzero_si128();
        برائے (صحیح ib32 = 0; ib32 < QK_K/32; ib32 += 4) {

            ثابت __m128i q2_ڈیٹا_0 = _mm_loadu_si128((ثابت __m128i*)q2);
            ثابت __m128i q2_ڈیٹا_1 = _mm_loadu_si128((ثابت __m128i*)q2 + 1);  q2 += 16;
            aux_gindex = MM256_سیٹ_M128_I(_mm_and_si128(q2_ڈیٹا_1, m511), _mm_and_si128(q2_ڈیٹا_0, m511));

            ثابت __m128i partial_sign_بٹ_0 = _mm_srli_epi16(q2_ڈیٹا_0, 9);
            ثابت __m128i partial_sign_بٹ_1 = _mm_srli_epi16(q2_ڈیٹا_1, 9);
            ثابت __m128i partial_sign_بٹ_بالائی_0 = _mm_srli_epi16(q2_ڈیٹا_0, 13);
            ثابت __m128i partial_sign_بٹ_بالائی_1 = _mm_srli_epi16(q2_ڈیٹا_1, 13);
            ثابت __m128i partial_sign_بٹ_for_counting_0 = _mm_xor_si128(partial_sign_بٹ_0, partial_sign_بٹ_بالائی_0);
            ثابت __m128i partial_sign_بٹ_for_counting_1 = _mm_xor_si128(partial_sign_بٹ_1, partial_sign_بٹ_بالائی_1);

            ثابت __m128i odd_بٹ_0 = _mm_shuffle_epi8(ثنائی_ہندسہ_helper_0, partial_sign_بٹ_for_counting_0);
            ثابت __m128i odd_بٹ_1 = _mm_shuffle_epi8(ثنائی_ہندسہ_helper_1, partial_sign_بٹ_for_counting_1);
            ثابت __m128i بھرا_sign_بٹ_0 = _mm_or_si128(partial_sign_بٹ_0, odd_بٹ_0);
            ثابت __m128i بھرا_sign_بٹ_1 = _mm_or_si128(partial_sign_بٹ_1, odd_بٹ_1);

            ثابت __m128i q8_1_0 = _mm_loadu_si128((ثابت __m128i *)q8); q8 += 16;
            ثابت __m128i q8_1_1 = _mm_loadu_si128((ثابت __m128i *)q8); q8 += 16;
            ثابت __m128i q8_2_0 = _mm_loadu_si128((ثابت __m128i *)q8); q8 += 16;
            ثابت __m128i q8_2_1 = _mm_loadu_si128((ثابت __m128i *)q8); q8 += 16;
            ثابت __m128i q8_3_0 = _mm_loadu_si128((ثابت __m128i *)q8); q8 += 16;
            ثابت __m128i q8_3_1 = _mm_loadu_si128((ثابت __m128i *)q8); q8 += 16;
            ثابت __m128i q8_4_0 = _mm_loadu_si128((ثابت __m128i *)q8); q8 += 16;
            ثابت __m128i q8_4_1 = _mm_loadu_si128((ثابت __m128i *)q8); q8 += 16;

            ثابت __m128i q2_1_0 = mm_سیٹ_epi64x(iq2xs_grid[gindex[1]], iq2xs_grid[gindex[0]]);
            ثابت __m128i q2_1_1 = mm_سیٹ_epi64x(iq2xs_grid[gindex[3]], iq2xs_grid[gindex[2]]);
            ثابت __m128i q2_2_0 = mm_سیٹ_epi64x(iq2xs_grid[gindex[5]], iq2xs_grid[gindex[4]]);
            ثابت __m128i q2_2_1 = mm_سیٹ_epi64x(iq2xs_grid[gindex[7]], iq2xs_grid[gindex[6]]);
            ثابت __m128i q2_3_0 = mm_سیٹ_epi64x(iq2xs_grid[gindex[9]], iq2xs_grid[gindex[8]]);
            ثابت __m128i q2_3_1 = mm_سیٹ_epi64x(iq2xs_grid[gindex[11]], iq2xs_grid[gindex[10]]);
            ثابت __m128i q2_4_0 = mm_سیٹ_epi64x(iq2xs_grid[gindex[13]], iq2xs_grid[gindex[12]]);
            ثابت __m128i q2_4_1 = mm_سیٹ_epi64x(iq2xs_grid[gindex[15]], iq2xs_grid[gindex[14]]);

            // AVX2 full_signs_1 is full_sign_bits_0 here
            // AVX2 full_signs_2 is full_sign_bits_1 here
            __m128i signs_0, signs_1;
            signs_0 = _mm_shuffle_epi8(بھرا_sign_بٹ_0, بلاک_sign_shuffle_1_0);
            signs_1 = _mm_shuffle_epi8(بھرا_sign_بٹ_0, بلاک_sign_shuffle_1_1);
            signs_0 = _mm_cmpeq_epi8(_mm_and_si128(signs_0, ثنائی_ہندسہ_selector_نقاب_0), ثنائی_ہندسہ_selector_نقاب_0);
            signs_1 = _mm_cmpeq_epi8(_mm_and_si128(signs_1, ثنائی_ہندسہ_selector_نقاب_1), ثنائی_ہندسہ_selector_نقاب_1);
            ثابت __m128i q8s_1_0 = _mm_sign_epi8(q8_1_0, _mm_or_si128(signs_0, mone));
            ثابت __m128i q8s_1_1 = _mm_sign_epi8(q8_1_1, _mm_or_si128(signs_1, mone));

            signs_0 = _mm_shuffle_epi8(بھرا_sign_بٹ_0, بلاک_sign_shuffle_2_0);
            signs_1 = _mm_shuffle_epi8(بھرا_sign_بٹ_0, بلاک_sign_shuffle_2_1);
            signs_0 = _mm_cmpeq_epi8(_mm_and_si128(signs_0, ثنائی_ہندسہ_selector_نقاب_0), ثنائی_ہندسہ_selector_نقاب_0);
            signs_1 = _mm_cmpeq_epi8(_mm_and_si128(signs_1, ثنائی_ہندسہ_selector_نقاب_1), ثنائی_ہندسہ_selector_نقاب_1);
            ثابت __m128i q8s_2_0 = _mm_sign_epi8(q8_2_0, _mm_or_si128(signs_0, mone));
            ثابت __m128i q8s_2_1 = _mm_sign_epi8(q8_2_1, _mm_or_si128(signs_1, mone));

            signs_0 = _mm_shuffle_epi8(بھرا_sign_بٹ_1, بلاک_sign_shuffle_1_0);
            signs_1 = _mm_shuffle_epi8(بھرا_sign_بٹ_1, بلاک_sign_shuffle_1_1);
            signs_0 = _mm_cmpeq_epi8(_mm_and_si128(signs_0, ثنائی_ہندسہ_selector_نقاب_0), ثنائی_ہندسہ_selector_نقاب_0);
            signs_1 = _mm_cmpeq_epi8(_mm_and_si128(signs_1, ثنائی_ہندسہ_selector_نقاب_1), ثنائی_ہندسہ_selector_نقاب_1);
            ثابت __m128i q8s_3_0 = _mm_sign_epi8(q8_3_0, _mm_or_si128(signs_0, mone));
            ثابت __m128i q8s_3_1 = _mm_sign_epi8(q8_3_1, _mm_or_si128(signs_1, mone));

            signs_0 = _mm_shuffle_epi8(بھرا_sign_بٹ_1, بلاک_sign_shuffle_2_0);
            signs_1 = _mm_shuffle_epi8(بھرا_sign_بٹ_1, بلاک_sign_shuffle_2_1);
            signs_0 = _mm_cmpeq_epi8(_mm_and_si128(signs_0, ثنائی_ہندسہ_selector_نقاب_0), ثنائی_ہندسہ_selector_نقاب_0);
            signs_1 = _mm_cmpeq_epi8(_mm_and_si128(signs_1, ثنائی_ہندسہ_selector_نقاب_1), ثنائی_ہندسہ_selector_نقاب_1);
            ثابت __m128i q8s_4_0 = _mm_sign_epi8(q8_4_0, _mm_or_si128(signs_0, mone));
            ثابت __m128i q8s_4_1 = _mm_sign_epi8(q8_4_1, _mm_or_si128(signs_1, mone));

            ثابت __m128i dot1_0  = _mm_maddubs_epi16(q2_1_0, q8s_1_0);
            ثابت __m128i dot1_1  = _mm_maddubs_epi16(q2_1_1, q8s_1_1);
            ثابت __m128i dot2_0  = _mm_maddubs_epi16(q2_2_0, q8s_2_0);
            ثابت __m128i dot2_1  = _mm_maddubs_epi16(q2_2_1, q8s_2_1);
            ثابت __m128i dot3_0  = _mm_maddubs_epi16(q2_3_0, q8s_3_0);
            ثابت __m128i dot3_1  = _mm_maddubs_epi16(q2_3_1, q8s_3_1);
            ثابت __m128i dot4_0  = _mm_maddubs_epi16(q2_4_0, q8s_4_0);
            ثابت __m128i dot4_1  = _mm_maddubs_epi16(q2_4_1, q8s_4_1);

            __m128i sc_tmp = _mm_shuffle_epi8(scales, get_scale_shuffle(ib32+0));
            ثابت __m128i sc1_0 = _mm_cvtepi8_epi16(sc_tmp);
            ثابت __m128i sc1_1 = _mm_cvtepi8_epi16(_mm_srli_si128(sc_tmp, 8));
            sc_tmp = _mm_shuffle_epi8(scales, get_scale_shuffle(ib32+1));
            ثابت __m128i sc2_0 = _mm_cvtepi8_epi16(sc_tmp);
            ثابت __m128i sc2_1 = _mm_cvtepi8_epi16(_mm_srli_si128(sc_tmp, 8));
            sc_tmp = _mm_shuffle_epi8(scales, get_scale_shuffle(ib32+2));
            ثابت __m128i sc3_0 = _mm_cvtepi8_epi16(sc_tmp);
            ثابت __m128i sc3_1 = _mm_cvtepi8_epi16(_mm_srli_si128(sc_tmp, 8));
            sc_tmp = _mm_shuffle_epi8(scales, get_scale_shuffle(ib32+3));
            ثابت __m128i sc4_0 = _mm_cvtepi8_epi16(sc_tmp);
            ثابت __m128i sc4_1 = _mm_cvtepi8_epi16(_mm_srli_si128(sc_tmp, 8));

            sumi1_0 = mm_جوڑو_epi32(sumi1_0, _mm_madd_epi16(dot1_0, sc1_0));
            sumi1_1 = mm_جوڑو_epi32(sumi1_1, _mm_madd_epi16(dot1_1, sc1_1));
            sumi2_0 = mm_جوڑو_epi32(sumi2_0, _mm_madd_epi16(dot2_0, sc2_0));
            sumi2_1 = mm_جوڑو_epi32(sumi2_1, _mm_madd_epi16(dot2_1, sc2_1));
            sumi1_0 = mm_جوڑو_epi32(sumi1_0, _mm_madd_epi16(dot3_0, sc3_0));
            sumi1_1 = mm_جوڑو_epi32(sumi1_1, _mm_madd_epi16(dot3_1, sc3_1));
            sumi2_0 = mm_جوڑو_epi32(sumi2_0, _mm_madd_epi16(dot4_0, sc4_0));
            sumi2_1 = mm_جوڑو_epi32(sumi2_1, _mm_madd_epi16(dot4_1, sc4_1));
        }

        accumf = mm256_جوڑو_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(MM256_سیٹ_M128_I(mm_جوڑو_epi32(sumi1_1, sumi2_1), mm_جوڑو_epi32(sumi1_0, sumi2_0)))), accumf);

    }

    *s = 0.125f * hsum_float_8(accumf);

#ورنہ
    غیر_مستعمل(x);
    غیر_مستعمل(y);
    غیر_مستعمل(nb);
    ggml_vec_dot_iq2_xs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#اگر_ختم
}

خالی ggml_vec_dot_iq2_s_q8_K(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc) {
    أكد(n % QK_K == 0);
    أكد(nrc == 1);
    غیر_مستعمل(nrc);
    غیر_مستعمل(bx);
    غیر_مستعمل(by);
    غیر_مستعمل(bs);

    ثابت بلاک_iq2_s * GGML_RESTRICT x = vx;
    ثابت بلاک_q8_K  * GGML_RESTRICT y = vy;

    ثابت صحیح nb = n / QK_K;

#اگر defined(__AVX2__)

   ساکن ثابت uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                       0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
   };

    ساکن ثابت uint8_t k_mask2[32] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };

    ثابت __m128i m4 = _mm_set1_epi8(0xf);
    ثابت __m128i m1 = _mm_set1_epi8(1);

    ثابت __m256i mask1 = _mm256_loadu_si256((ثابت __m256i*)k_mask1);
    ثابت __m256i mask2 = _mm256_loadu_si256((ثابت __m256i*)k_mask2);

    uint64_t aux64;

    __m256 accumf = _mm256_setzero_ps();
    برائے (صحیح i = 0; i < nb; ++i) {
        ثابت اعشاری d = GGML_CPU_FP16_مقصد_FP32(x[i].d) * y[i].d;
        ثابت uint8_t * GGML_RESTRICT qs = x[i].qs;
        ثابت uint8_t * GGML_RESTRICT qh = x[i].qh;
        ثابت uint16_t * GGML_RESTRICT signs = (ثابت uint16_t *)(x[i].qs + QK_K/8);
        ثابت int8_t  * GGML_RESTRICT q8 = y[i].qs;

        حافظہ_نقل(&aux64, x[i].scales, 8);
        ثابت __m128i scales8 = mm_جوڑو_epi8(_mm_slli_epi16(_mm_and_si128(mm_سیٹ_epi64x(aux64 >> 4, aux64), m4), 1), m1);
        ثابت __m256i scales16 = _mm256_cvtepi8_epi16(scales8); // 0 2 4 6 8 10 12 14 1 3 5 7 9 11 13 15

        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        برائے (صحیح ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            ثابت __m256i q8_1 = _mm256_loadu_si256((ثابت __m256i *)q8); q8 += 32;
            ثابت __m256i q8_2 = _mm256_loadu_si256((ثابت __m256i *)q8); q8 += 32;
            ثابت __m256i q2_1 = mm256_سیٹ_epi64x(iq2s_grid[qs[3] | ((qh[ib32+0] << 2) & 0x300)],
                                                   iq2s_grid[qs[2] | ((qh[ib32+0] << 4) & 0x300)],
                                                   iq2s_grid[qs[1] | ((qh[ib32+0] << 6) & 0x300)],
                                                   iq2s_grid[qs[0] | ((qh[ib32+0] << 8) & 0x300)]);
            ثابت __m256i q2_2 = mm256_سیٹ_epi64x(iq2s_grid[qs[7] | ((qh[ib32+1] << 2) & 0x300)],
                                                   iq2s_grid[qs[6] | ((qh[ib32+1] << 4) & 0x300)],
                                                   iq2s_grid[qs[5] | ((qh[ib32+1] << 6) & 0x300)],
                                                   iq2s_grid[qs[4] | ((qh[ib32+1] << 8) & 0x300)]);
            qs += 8;

            __m256i aux256 = _mm256_set1_epi32(signs[0] | ((uint32_t) signs[1] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256,mask1), mask2);
            ثابت __m256i s2_1 = _mm256_cmpeq_epi8(aux256, mask2);
            ثابت __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, q8_1), s2_1);

            aux256 = _mm256_set1_epi32(signs[2] | ((uint32_t) signs[3] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256,mask1), mask2);
            ثابت __m256i s2_2 = _mm256_cmpeq_epi8(aux256, mask2);
            ثابت __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, q8_2), s2_2);

            signs += 4;

            ثابت __m256i dot1  = _mm256_maddubs_epi16(q2_1, q8s_1); // blocks 2*ib32+0, 2*ib32+1
            ثابت __m256i dot2  = _mm256_maddubs_epi16(q2_2, q8s_2); // blocks 2*ib32+2, 2*ib32+3

            ثابت __m256i p1 = _mm256_madd_epi16(dot1, _mm256_shuffle_epi8(scales16, get_scale_shuffle_k4(ib32+0)));
            ثابت __m256i p2 = _mm256_madd_epi16(dot2, _mm256_shuffle_epi8(scales16, get_scale_shuffle_k4(ib32+1)));
            sumi1 = mm256_جوڑو_epi32(sumi1, p1);
            sumi2 = mm256_جوڑو_epi32(sumi2, p2);
        }

        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(mm256_جوڑو_epi32(sumi1, sumi2)), accumf);

    }

    *s = 0.125f * hsum_float_8(accumf);

#وگرنہ_اگر defined(__AVX__)
   ساکن ثابت uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                       0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
   };

    ساکن ثابت uint8_t k_mask2[32] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };

    ثابت __m128i m4 = _mm_set1_epi8(0xf);
    ثابت __m128i m1 = _mm_set1_epi8(1);

    ثابت __m128i mask1_0 = _mm_loadu_si128((ثابت __m128i*)k_mask1);
    ثابت __m128i mask1_1 = _mm_loadu_si128((ثابت __m128i*)k_mask1 + 1);
    ثابت __m128i mask2_0 = _mm_loadu_si128((ثابت __m128i*)k_mask2);
    ثابت __m128i mask2_1 = _mm_loadu_si128((ثابت __m128i*)k_mask2 + 1);

    uint64_t aux64;

    __m256 accumf = _mm256_setzero_ps();
    برائے (صحیح i = 0; i < nb; ++i) {
        ثابت اعشاری d = GGML_CPU_FP16_مقصد_FP32(x[i].d) * y[i].d;
        ثابت uint8_t * GGML_RESTRICT qs = x[i].qs;
        ثابت uint8_t * GGML_RESTRICT qh = x[i].qh;
        ثابت uint16_t * GGML_RESTRICT signs = (ثابت uint16_t *)(x[i].qs + QK_K/8);
        ثابت int8_t  * GGML_RESTRICT q8 = y[i].qs;

        حافظہ_نقل(&aux64, x[i].scales, 8);
        ثابت __m128i scales8 = mm_جوڑو_epi8(_mm_slli_epi16(_mm_and_si128(mm_سیٹ_epi64x(aux64 >> 4, aux64), m4), 1), m1);
        ثابت __m128i scales16_0 = _mm_cvtepi8_epi16(scales8);
        ثابت __m128i scales16_1 = _mm_cvtepi8_epi16(_mm_srli_si128(scales8, 8));

        __m128i sumi1_0 = _mm_setzero_si128();
        __m128i sumi1_1 = _mm_setzero_si128();
        __m128i sumi2_0 = _mm_setzero_si128();
        __m128i sumi2_1 = _mm_setzero_si128();
        برائے (صحیح ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            ثابت __m128i q8_1_0 = _mm_loadu_si128((ثابت __m128i *)q8); q8 += 16;
            ثابت __m128i q8_1_1 = _mm_loadu_si128((ثابت __m128i *)q8); q8 += 16;
            ثابت __m128i q8_2_0 = _mm_loadu_si128((ثابت __m128i *)q8); q8 += 16;
            ثابت __m128i q8_2_1 = _mm_loadu_si128((ثابت __m128i *)q8); q8 += 16;
            ثابت __m128i q2_1_0 = mm_سیٹ_epi64x(iq2s_grid[qs[1] | ((qh[ib32+0] << 6) & 0x300)],
                                                  iq2s_grid[qs[0] | ((qh[ib32+0] << 8) & 0x300)]);
            ثابت __m128i q2_1_1 = mm_سیٹ_epi64x(iq2s_grid[qs[3] | ((qh[ib32+0] << 2) & 0x300)],
                                                  iq2s_grid[qs[2] | ((qh[ib32+0] << 4) & 0x300)]);
            ثابت __m128i q2_2_0 = mm_سیٹ_epi64x(iq2s_grid[qs[5] | ((qh[ib32+1] << 6) & 0x300)],
                                                  iq2s_grid[qs[4] | ((qh[ib32+1] << 8) & 0x300)]);
            ثابت __m128i q2_2_1 = mm_سیٹ_epi64x(iq2s_grid[qs[7] | ((qh[ib32+1] << 2) & 0x300)],
                                                  iq2s_grid[qs[6] | ((qh[ib32+1] << 4) & 0x300)]);
            qs += 8;

            __m128i aux128_0 = _mm_set1_epi32(signs[0] | ((uint32_t) signs[1] << 16));
            __m128i aux128_1 = aux128_0;
            aux128_0 = _mm_and_si128(_mm_shuffle_epi8(aux128_0,mask1_0), mask2_0);
            aux128_1 = _mm_and_si128(_mm_shuffle_epi8(aux128_1,mask1_1), mask2_1);
            ثابت __m128i s2_1_0 = _mm_cmpeq_epi8(aux128_0, mask2_0);
            ثابت __m128i s2_1_1 = _mm_cmpeq_epi8(aux128_1, mask2_1);
            ثابت __m128i q8s_1_0 = _mm_sub_epi8(_mm_xor_si128(s2_1_0, q8_1_0), s2_1_0);
            ثابت __m128i q8s_1_1 = _mm_sub_epi8(_mm_xor_si128(s2_1_1, q8_1_1), s2_1_1);

            aux128_0 = _mm_set1_epi32(signs[2] | ((uint32_t) signs[3] << 16));
            aux128_1 = aux128_0;
            aux128_0 = _mm_and_si128(_mm_shuffle_epi8(aux128_0,mask1_0), mask2_0);
            aux128_1 = _mm_and_si128(_mm_shuffle_epi8(aux128_1,mask1_1), mask2_1);
            ثابت __m128i s2_2_0 = _mm_cmpeq_epi8(aux128_0, mask2_0);
            ثابت __m128i s2_2_1 = _mm_cmpeq_epi8(aux128_1, mask2_1);
            ثابت __m128i q8s_2_0 = _mm_sub_epi8(_mm_xor_si128(s2_2_0, q8_2_0), s2_2_0);
            ثابت __m128i q8s_2_1 = _mm_sub_epi8(_mm_xor_si128(s2_2_1, q8_2_1), s2_2_1);

            signs += 4;

            ثابت __m128i dot1_0  = _mm_maddubs_epi16(q2_1_0, q8s_1_0);
            ثابت __m128i dot1_1  = _mm_maddubs_epi16(q2_1_1, q8s_1_1);
            ثابت __m128i dot2_0  = _mm_maddubs_epi16(q2_2_0, q8s_2_0);
            ثابت __m128i dot2_1  = _mm_maddubs_epi16(q2_2_1, q8s_2_1);

            ثابت __m128i p1_0 = _mm_madd_epi16(dot1_0, _mm_shuffle_epi8(scales16_0, _mm256_extractf128_si256(get_scale_shuffle_k4(ib32+0), 0)));
            ثابت __m128i p1_1 = _mm_madd_epi16(dot1_1, _mm_shuffle_epi8(scales16_1, _mm256_extractf128_si256(get_scale_shuffle_k4(ib32+0), 1)));
            ثابت __m128i p2_0 = _mm_madd_epi16(dot2_0, _mm_shuffle_epi8(scales16_0, _mm256_extractf128_si256(get_scale_shuffle_k4(ib32+1), 0)));
            ثابت __m128i p2_1 = _mm_madd_epi16(dot2_1, _mm_shuffle_epi8(scales16_1, _mm256_extractf128_si256(get_scale_shuffle_k4(ib32+1), 1)));
            sumi1_0 = mm_جوڑو_epi32(sumi1_0, p1_0);
            sumi1_1 = mm_جوڑو_epi32(sumi1_1, p1_1);
            sumi2_0 = mm_جوڑو_epi32(sumi2_0, p2_0);
            sumi2_1 = mm_جوڑو_epi32(sumi2_1, p2_1);
        }

        accumf = mm256_جوڑو_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(MM256_سیٹ_M128_I(mm_جوڑو_epi32(sumi1_1, sumi2_1), mm_جوڑو_epi32(sumi1_0, sumi2_0)))), accumf);

    }

    *s = 0.125f * hsum_float_8(accumf);

#ورنہ
    غیر_مستعمل(x);
    غیر_مستعمل(y);
    غیر_مستعمل(nb);
    ggml_vec_dot_iq2_s_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#اگر_ختم
}

خالی ggml_vec_dot_iq3_xxs_q8_K(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc) {
    أكد(n % QK_K == 0);
    أكد(nrc == 1);
    غیر_مستعمل(nrc);
    غیر_مستعمل(bx);
    غیر_مستعمل(by);
    غیر_مستعمل(bs);

    ثابت بلاک_iq3_xxs * GGML_RESTRICT x = vx;
    ثابت بلاک_q8_K    * GGML_RESTRICT y = vy;

    ثابت صحیح nb = n / QK_K;

#اگر defined(__AVX2__)

    ثابت uint64_t * signs64 = (ثابت uint64_t *)keven_signs_q2xs;

    uint32_t aux32[2];

    __m256 accumf = _mm256_setzero_ps();
    برائے (صحیح i = 0; i < nb; ++i) {
        ثابت اعشاری d = GGML_CPU_FP16_مقصد_FP32(x[i].d) * y[i].d;
        ثابت uint8_t * GGML_RESTRICT q3 = x[i].qs;
        ثابت uint8_t * GGML_RESTRICT gas = x[i].qs + QK_K/4;
        ثابت int8_t  * GGML_RESTRICT q8 = y[i].qs;
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        برائے (صحیح ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            ثابت __m256i q8_1 = _mm256_loadu_si256((ثابت __m256i *)q8); q8 += 32;
            ثابت __m256i q8_2 = _mm256_loadu_si256((ثابت __m256i *)q8); q8 += 32;
            ثابت __m256i q2_1 = mm256_سیٹ_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]],
                                                  iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
            q3 += 8;
            ثابت __m256i q2_2 = mm256_سیٹ_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]],
                                                  iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
            q3 += 8;
            حافظہ_نقل(aux32, gas, 8); gas += 8;
            ثابت __m256i s2_1 = mm256_سیٹ_epi64x(signs64[(aux32[0] >> 21) & 127], signs64[(aux32[0] >> 14) & 127],
                                                   signs64[(aux32[0] >>  7) & 127], signs64[(aux32[0] >>  0) & 127]);
            ثابت __m256i s2_2 = mm256_سیٹ_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127],
                                                   signs64[(aux32[1] >>  7) & 127], signs64[(aux32[1] >>  0) & 127]);
            ثابت __m256i q8s_1 = _mm256_sign_epi8(q8_1, s2_1);
            ثابت __m256i q8s_2 = _mm256_sign_epi8(q8_2, s2_2);
            ثابت __m256i dot1  = _mm256_maddubs_epi16(q2_1, q8s_1);
            ثابت __m256i dot2  = _mm256_maddubs_epi16(q2_2, q8s_2);
            ثابت uint16_t ls1 = aux32[0] >> 28;
            ثابت uint16_t ls2 = aux32[1] >> 28;
            ثابت __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16(2*ls1+1));
            ثابت __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16(2*ls2+1));
            sumi1 = mm256_جوڑو_epi32(sumi1, p1);
            sumi2 = mm256_جوڑو_epi32(sumi2, p2);
        }

        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(mm256_جوڑو_epi32(sumi1, sumi2)), accumf);

    }

    *s = 0.25f * hsum_float_8(accumf);

#وگرنہ_اگر defined(__AVX__)
    ثابت uint64_t * signs64 = (ثابت uint64_t *)keven_signs_q2xs;

    uint32_t aux32[2];

    __m256 accumf = _mm256_setzero_ps();
    برائے (صحیح i = 0; i < nb; ++i) {
        ثابت اعشاری d = GGML_CPU_FP16_مقصد_FP32(x[i].d) * y[i].d;
        ثابت uint8_t * GGML_RESTRICT q3 = x[i].qs;
        ثابت uint8_t * GGML_RESTRICT gas = x[i].qs + QK_K/4;
        ثابت int8_t  * GGML_RESTRICT q8 = y[i].qs;
        __m128i sumi1_0 = _mm_setzero_si128();
        __m128i sumi1_1 = _mm_setzero_si128();
        __m128i sumi2_0 = _mm_setzero_si128();
        __m128i sumi2_1 = _mm_setzero_si128();
        برائے (صحیح ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            ثابت __m128i q8_1_0 = _mm_loadu_si128((ثابت __m128i *)q8); q8 += 16;
            ثابت __m128i q8_1_1 = _mm_loadu_si128((ثابت __m128i *)q8); q8 += 16;
            ثابت __m128i q8_2_0 = _mm_loadu_si128((ثابت __m128i *)q8); q8 += 16;
            ثابت __m128i q8_2_1 = _mm_loadu_si128((ثابت __m128i *)q8); q8 += 16;
            ثابت __m128i q2_1_0 = mm_سیٹ_epi32(iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
            ثابت __m128i q2_1_1 = mm_سیٹ_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]]);
            q3 += 8;
            ثابت __m128i q2_2_0 = mm_سیٹ_epi32(iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
            ثابت __m128i q2_2_1 = mm_سیٹ_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]]);
            q3 += 8;
            حافظہ_نقل(aux32, gas, 8); gas += 8;
            ثابت __m128i s2_1_0 = mm_سیٹ_epi64x(signs64[(aux32[0] >>  7) & 127], signs64[(aux32[0] >>  0) & 127]);
            ثابت __m128i s2_1_1 = mm_سیٹ_epi64x(signs64[(aux32[0] >> 21) & 127], signs64[(aux32[0] >> 14) & 127]);
            ثابت __m128i s2_2_0 = mm_سیٹ_epi64x(signs64[(aux32[1] >>  7) & 127], signs64[(aux32[1] >>  0) & 127]);
            ثابت __m128i s2_2_1 = mm_سیٹ_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127]);
            ثابت __m128i q8s_1_0 = _mm_sign_epi8(q8_1_0, s2_1_0);
            ثابت __m128i q8s_1_1 = _mm_sign_epi8(q8_1_1, s2_1_1);
            ثابت __m128i q8s_2_0 = _mm_sign_epi8(q8_2_0, s2_2_0);
            ثابت __m128i q8s_2_1 = _mm_sign_epi8(q8_2_1, s2_2_1);
            ثابت __m128i dot1_0  = _mm_maddubs_epi16(q2_1_0, q8s_1_0);
            ثابت __m128i dot1_1  = _mm_maddubs_epi16(q2_1_1, q8s_1_1);
            ثابت __m128i dot2_0  = _mm_maddubs_epi16(q2_2_0, q8s_2_0);
            ثابت __m128i dot2_1  = _mm_maddubs_epi16(q2_2_1, q8s_2_1);
            ثابت uint16_t ls1 = aux32[0] >> 28;
            ثابت uint16_t ls2 = aux32[1] >> 28;
            ثابت __m128i p1_0 = _mm_madd_epi16(dot1_0, _mm_set1_epi16(2*ls1+1));
            ثابت __m128i p1_1 = _mm_madd_epi16(dot1_1, _mm_set1_epi16(2*ls1+1));
            ثابت __m128i p2_0 = _mm_madd_epi16(dot2_0, _mm_set1_epi16(2*ls2+1));
            ثابت __m128i p2_1 = _mm_madd_epi16(dot2_1, _mm_set1_epi16(2*ls2+1));
            sumi1_0 = mm_جوڑو_epi32(sumi1_0, p1_0);
            sumi1_1 = mm_جوڑو_epi32(sumi1_1, p1_1);
            sumi2_0 = mm_جوڑو_epi32(sumi2_0, p2_0);
            sumi2_1 = mm_جوڑو_epi32(sumi2_1, p2_1);
        }

        accumf = mm256_جوڑو_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(MM256_سیٹ_M128_I(mm_جوڑو_epi32(sumi1_1, sumi2_1), mm_جوڑو_epi32(sumi1_0, sumi2_0)))), accumf);

    }

    *s = 0.25f * hsum_float_8(accumf);

#ورنہ
    غیر_مستعمل(x);
    غیر_مستعمل(y);
    غیر_مستعمل(nb);
    ggml_vec_dot_iq3_xxs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#اگر_ختم
}

خالی ggml_vec_dot_iq3_s_q8_K(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc) {
    أكد(n % QK_K == 0);
    أكد(nrc == 1);
    غیر_مستعمل(nrc);
    غیر_مستعمل(bx);
    غیر_مستعمل(by);
    غیر_مستعمل(bs);

    ثابت بلاک_iq3_s * GGML_RESTRICT x = vx;
    ثابت بلاک_q8_K  * GGML_RESTRICT y = vy;

    ثابت صحیح nb = n / QK_K;

#اگر defined(__AVX2__)

   ساکن ثابت uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                       0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
   };

    ساکن ثابت uint8_t k_mask2[32] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };

    ثابت __m256i mask1 = _mm256_loadu_si256((ثابت __m256i*)k_mask1);
    ثابت __m256i mask2 = _mm256_loadu_si256((ثابت __m256i*)k_mask2);

    ثابت __m256i idx_سرکاؤ = mm256_سیٹ_epi32(1, 2, 3, 4, 5, 6, 7, 8);
    ثابت __m256i idx_نقاب  = _mm256_set1_epi32(256);

    قسمتعریف اتحاد {
        __m256i  vec[2];
        uint32_t اشاریہ[16];
    } اشاریہ_t;

    اشاریہ_t idx;

    __m256 accumf = _mm256_setzero_ps();
    برائے (صحیح i = 0; i < nb; ++i) {
        ثابت اعشاری d = GGML_CPU_FP16_مقصد_FP32(x[i].d) * y[i].d;
        ثابت uint8_t * GGML_RESTRICT qs = x[i].qs;
        ثابت uint8_t * GGML_RESTRICT qh = x[i].qh;
        ثابت uint16_t * GGML_RESTRICT signs = (ثابت uint16_t *)x[i].signs;
        ثابت int8_t  * GGML_RESTRICT q8 = y[i].qs;
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        برائے (صحیح ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            ثابت __m256i q8_1 = _mm256_loadu_si256((ثابت __m256i *)q8); q8 += 32;
            ثابت __m256i q8_2 = _mm256_loadu_si256((ثابت __m256i *)q8); q8 += 32;
            ثابت __m256i idx_l = _mm256_cvtepu8_epi16(_mm_loadu_si128((ثابت __m128i *)qs)); qs += 16;
            idx.vec[0] = _mm256_set1_epi32(qh[ib32+0]);
            idx.vec[1] = _mm256_set1_epi32(qh[ib32+1]);
            idx.vec[0] = _mm256_and_si256(_mm256_sllv_epi32(idx.vec[0], idx_سرکاؤ), idx_نقاب);
            idx.vec[1] = _mm256_and_si256(_mm256_sllv_epi32(idx.vec[1], idx_سرکاؤ), idx_نقاب);
            idx.vec[0] = _mm256_or_si256(idx.vec[0], _mm256_cvtepi16_epi32(_mm256_castsi256_si128(idx_l)));
            idx.vec[1] = _mm256_or_si256(idx.vec[1], _mm256_cvtepi16_epi32(_mm256_extractf128_si256(idx_l, 1)));

            // At leat on my CPU (Ryzen 7950X), using _mm256_i32gather_epi32 is slower than _mm256_set_epi32. Strange.
            //const __m256i q2_1 = _mm256_i32gather_epi32((const int *)iq3s_grid, idx.vec[0], 4);
            //const __m256i q2_2 = _mm256_i32gather_epi32((const int *)iq3s_grid, idx.vec[1], 4);
            ثابت __m256i q2_1 = mm256_سیٹ_epi32(
                    iq3s_grid[idx.اشاریہ[7]], iq3s_grid[idx.اشاریہ[6]], iq3s_grid[idx.اشاریہ[5]], iq3s_grid[idx.اشاریہ[4]],
                    iq3s_grid[idx.اشاریہ[3]], iq3s_grid[idx.اشاریہ[2]], iq3s_grid[idx.اشاریہ[1]], iq3s_grid[idx.اشاریہ[0]]
            );
            ثابت __m256i q2_2 = mm256_سیٹ_epi32(
                    iq3s_grid[idx.اشاریہ[15]], iq3s_grid[idx.اشاریہ[14]], iq3s_grid[idx.اشاریہ[13]], iq3s_grid[idx.اشاریہ[12]],
                    iq3s_grid[idx.اشاریہ[11]], iq3s_grid[idx.اشاریہ[10]], iq3s_grid[idx.اشاریہ[ 9]], iq3s_grid[idx.اشاریہ[ 8]]
            );

            __m256i aux256 = _mm256_set1_epi32(signs[0] | (signs[1] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256,mask1), mask2);
            ثابت __m256i s2_1 = _mm256_cmpeq_epi8(aux256, mask2);
            ثابت __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, q8_1), s2_1);

            aux256 = _mm256_set1_epi32(signs[2] | (signs[3] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256,mask1), mask2);
            ثابت __m256i s2_2 = _mm256_cmpeq_epi8(aux256, mask2);
            ثابت __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, q8_2), s2_2);

            signs += 4;

            ثابت __m256i dot1  = _mm256_maddubs_epi16(q2_1, q8s_1);
            ثابت __m256i dot2  = _mm256_maddubs_epi16(q2_2, q8s_2);
            ثابت uint16_t ls1 = x[i].scales[ib32/2] & 0xf;
            ثابت uint16_t ls2 = x[i].scales[ib32/2] >>  4;
            ثابت __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16(2*ls1+1));
            ثابت __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16(2*ls2+1));
            sumi1 = mm256_جوڑو_epi32(sumi1, p1);
            sumi2 = mm256_جوڑو_epi32(sumi2, p2);
        }

        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(mm256_جوڑو_epi32(sumi1, sumi2)), accumf);

    }

    *s = hsum_float_8(accumf);

#وگرنہ_اگر defined(__AVX__)
   ساکن ثابت uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                       0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
   };

    ساکن ثابت uint8_t k_mask2[32] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };

    ثابت __m128i mask1_0 = _mm_loadu_si128((ثابت __m128i*)k_mask1);
    ثابت __m128i mask1_1 = _mm_loadu_si128((ثابت __m128i*)k_mask1 + 1);
    ثابت __m128i mask2_0 = _mm_loadu_si128((ثابت __m128i*)k_mask2);
    ثابت __m128i mask2_1 = _mm_loadu_si128((ثابت __m128i*)k_mask2 + 1);

    ثابت __m128i idx_mul_0 = mm_سیٹ_epi32(32, 64, 128, 256);
    ثابت __m128i idx_mul_1 = mm_سیٹ_epi32(2, 4, 8, 16);
    ثابت __m128i idx_نقاب  = _mm_set1_epi32(256);

    قسمتعریف اتحاد {
        __m128i  vec[4];
        uint32_t اشاریہ[16];
    } اشاریہ_t;

    اشاریہ_t idx;

    __m256 accumf = _mm256_setzero_ps();
    برائے (صحیح i = 0; i < nb; ++i) {
        ثابت اعشاری d = GGML_CPU_FP16_مقصد_FP32(x[i].d) * y[i].d;
        ثابت uint8_t * GGML_RESTRICT qs = x[i].qs;
        ثابت uint8_t * GGML_RESTRICT qh = x[i].qh;
        ثابت uint16_t * GGML_RESTRICT signs = (ثابت uint16_t *)x[i].signs;
        ثابت int8_t  * GGML_RESTRICT q8 = y[i].qs;
        __m128i sumi1_0 = _mm_setzero_si128();
        __m128i sumi1_1 = _mm_setzero_si128();
        __m128i sumi2_0 = _mm_setzero_si128();
        __m128i sumi2_1 = _mm_setzero_si128();
        برائے (صحیح ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            ثابت __m128i q8_1_0 = _mm_loadu_si128((ثابت __m128i *)q8); q8 += 16;
            ثابت __m128i q8_1_1 = _mm_loadu_si128((ثابت __m128i *)q8); q8 += 16;
            ثابت __m128i q8_2_0 = _mm_loadu_si128((ثابت __m128i *)q8); q8 += 16;
            ثابت __m128i q8_2_1 = _mm_loadu_si128((ثابت __m128i *)q8); q8 += 16;
            ثابت __m128i qs_tmp = _mm_loadu_si128((ثابت __m128i *)qs);
            ثابت __m128i idx_l_0 = _mm_cvtepu8_epi16(qs_tmp);
            ثابت __m128i idx_l_1 = _mm_cvtepu8_epi16(_mm_srli_si128(qs_tmp, 8)); qs += 16;
            idx.vec[0] = _mm_set1_epi32(qh[ib32+0]);
            idx.vec[1] = idx.vec[0];
            idx.vec[2] = _mm_set1_epi32(qh[ib32+1]);
            idx.vec[3] = idx.vec[2];

            idx.vec[0] = _mm_and_si128(_mm_mullo_epi32(idx.vec[0], idx_mul_0), idx_نقاب);
            idx.vec[1] = _mm_and_si128(_mm_mullo_epi32(idx.vec[1], idx_mul_1), idx_نقاب);
            idx.vec[2] = _mm_and_si128(_mm_mullo_epi32(idx.vec[2], idx_mul_0), idx_نقاب);
            idx.vec[3] = _mm_and_si128(_mm_mullo_epi32(idx.vec[3], idx_mul_1), idx_نقاب);

            idx.vec[0] = _mm_or_si128(idx.vec[0], _mm_cvtepi16_epi32(idx_l_0));
            idx.vec[1] = _mm_or_si128(idx.vec[1], _mm_cvtepi16_epi32(_mm_srli_si128(idx_l_0, 8)));
            idx.vec[2] = _mm_or_si128(idx.vec[2], _mm_cvtepi16_epi32(idx_l_1));
            idx.vec[3] = _mm_or_si128(idx.vec[3], _mm_cvtepi16_epi32(_mm_srli_si128(idx_l_1, 8)));

            ثابت __m128i q2_1_0 = mm_سیٹ_epi32(iq3s_grid[idx.اشاریہ[3]], iq3s_grid[idx.اشاریہ[2]], iq3s_grid[idx.اشاریہ[1]], iq3s_grid[idx.اشاریہ[0]]);
            ثابت __m128i q2_1_1 = mm_سیٹ_epi32(iq3s_grid[idx.اشاریہ[7]], iq3s_grid[idx.اشاریہ[6]], iq3s_grid[idx.اشاریہ[5]], iq3s_grid[idx.اشاریہ[4]]);
            ثابت __m128i q2_2_0 = mm_سیٹ_epi32(iq3s_grid[idx.اشاریہ[11]], iq3s_grid[idx.اشاریہ[10]], iq3s_grid[idx.اشاریہ[9]], iq3s_grid[idx.اشاریہ[8]]);
            ثابت __m128i q2_2_1 = mm_سیٹ_epi32(iq3s_grid[idx.اشاریہ[15]], iq3s_grid[idx.اشاریہ[14]], iq3s_grid[idx.اشاریہ[13]], iq3s_grid[idx.اشاریہ[12]]);

            __m128i aux128_0 = _mm_set1_epi32(signs[0] | (signs[1] << 16));
            __m128i aux128_1 = aux128_0;
            aux128_0 = _mm_and_si128(_mm_shuffle_epi8(aux128_0,mask1_0), mask2_0);
            aux128_1 = _mm_and_si128(_mm_shuffle_epi8(aux128_1,mask1_1), mask2_1);
            ثابت __m128i s2_1_0 = _mm_cmpeq_epi8(aux128_0, mask2_0);
            ثابت __m128i s2_1_1 = _mm_cmpeq_epi8(aux128_1, mask2_1);
            ثابت __m128i q8s_1_0 = _mm_sub_epi8(_mm_xor_si128(s2_1_0, q8_1_0), s2_1_0);
            ثابت __m128i q8s_1_1 = _mm_sub_epi8(_mm_xor_si128(s2_1_1, q8_1_1), s2_1_1);

            aux128_0 = _mm_set1_epi32(signs[2] | (signs[3] << 16));
            aux128_1 = aux128_0;
            aux128_0 = _mm_and_si128(_mm_shuffle_epi8(aux128_0,mask1_0), mask2_0);
            aux128_1 = _mm_and_si128(_mm_shuffle_epi8(aux128_1,mask1_1), mask2_1);
            ثابت __m128i s2_2_0 = _mm_cmpeq_epi8(aux128_0, mask2_0);
            ثابت __m128i s2_2_1 = _mm_cmpeq_epi8(aux128_1, mask2_1);
            ثابت __m128i q8s_2_0 = _mm_sub_epi8(_mm_xor_si128(s2_2_0, q8_2_0), s2_2_0);
            ثابت __m128i q8s_2_1 = _mm_sub_epi8(_mm_xor_si128(s2_2_1, q8_2_1), s2_2_1);

            signs += 4;

            ثابت __m128i dot1_0  = _mm_maddubs_epi16(q2_1_0, q8s_1_0);
            ثابت __m128i dot1_1  = _mm_maddubs_epi16(q2_1_1, q8s_1_1);
            ثابت __m128i dot2_0  = _mm_maddubs_epi16(q2_2_0, q8s_2_0);
            ثابت __m128i dot2_1  = _mm_maddubs_epi16(q2_2_1, q8s_2_1);
            ثابت uint16_t ls1 = x[i].scales[ib32/2] & 0xf;
            ثابت uint16_t ls2 = x[i].scales[ib32/2] >>  4;
            ثابت __m128i p1_0 = _mm_madd_epi16(dot1_0, _mm_set1_epi16(2*ls1+1));
            ثابت __m128i p1_1 = _mm_madd_epi16(dot1_1, _mm_set1_epi16(2*ls1+1));
            ثابت __m128i p2_0 = _mm_madd_epi16(dot2_0, _mm_set1_epi16(2*ls2+1));
            ثابت __m128i p2_1 = _mm_madd_epi16(dot2_1, _mm_set1_epi16(2*ls2+1));
            sumi1_0 = mm_جوڑو_epi32(sumi1_0, p1_0);
            sumi1_1 = mm_جوڑو_epi32(sumi1_1, p1_1);
            sumi2_0 = mm_جوڑو_epi32(sumi2_0, p2_0);
            sumi2_1 = mm_جوڑو_epi32(sumi2_1, p2_1);
        }

        accumf = mm256_جوڑو_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(MM256_سیٹ_M128_I(mm_جوڑو_epi32(sumi1_1, sumi2_1), mm_جوڑو_epi32(sumi1_0, sumi2_0)))), accumf);

    }

    *s = hsum_float_8(accumf);

#ورنہ
    غیر_مستعمل(x);
    غیر_مستعمل(y);
    غیر_مستعمل(nb);
    ggml_vec_dot_iq3_s_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#اگر_ختم
}

خالی ggml_vec_dot_iq1_s_q8_K(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc) {
    أكد(n % QK_K == 0);
    أكد(nrc == 1);
    غیر_مستعمل(nrc);
    غیر_مستعمل(bx);
    غیر_مستعمل(by);
    غیر_مستعمل(bs);

    ثابت بلاک_iq1_s * GGML_RESTRICT x = vx;
    ثابت بلاک_q8_K  * GGML_RESTRICT y = vy;

    ثابت صحیح nb = n / QK_K;

#اگر defined __AVX2__

    __m256 accum = _mm256_setzero_ps();
    اعشاری accum1 = 0;
    برائے (صحیح i = 0; i < nb; ++i) {

        ثابت int8_t   * q8 = y[i].qs;
        ثابت uint8_t  * qs = x[i].qs;
        ثابت uint16_t * qh = x[i].qh;

        __m256i sumi = _mm256_setzero_si256();
        صحیح sumi1 = 0;
        برائے (صحیح ib = 0; ib < QK_K/32; ib += 2) {
#اگر_تعریف_شدہ __BMI2__
            ثابت uint64_t packed_idx1 = _pdep_u64(*(ثابت uint32_t *)qs, 0x00ff00ff00ff00ffULL) | _pdep_u64(qh[ib], 0x700070007000700ULL);
            ثابت uint64_t packed_idx2 = _pdep_u64(*(ثابت uint32_t *)(qs + 4), 0x00ff00ff00ff00ffULL) | _pdep_u64(qh[ib + 1], 0x700070007000700ULL);
            ثابت uint16_t *idx1 = (ثابت uint16_t *)(&packed_idx1);
            ثابت uint16_t *idx2 = (ثابت uint16_t *)(&packed_idx2);
            ثابت __m256i q1b_1 = mm256_سیٹ_epi64x(iq1s_grid[idx1[3]], iq1s_grid[idx1[2]], iq1s_grid[idx1[1]], iq1s_grid[idx1[0]]);
            ثابت __m256i q1b_2 = mm256_سیٹ_epi64x(iq1s_grid[idx2[3]], iq1s_grid[idx2[2]], iq1s_grid[idx2[1]], iq1s_grid[idx2[0]]);
#ورنہ
            ثابت __m256i q1b_1 = mm256_سیٹ_epi64x(iq1s_grid[qs[3] | ((qh[ib+0] >> 1) & 0x700)], iq1s_grid[qs[2] | ((qh[ib+0] << 2) & 0x700)],
                                                    iq1s_grid[qs[1] | ((qh[ib+0] << 5) & 0x700)], iq1s_grid[qs[0] | ((qh[ib+0] << 8) & 0x700)]);
            ثابت __m256i q1b_2 = mm256_سیٹ_epi64x(iq1s_grid[qs[7] | ((qh[ib+1] >> 1) & 0x700)], iq1s_grid[qs[6] | ((qh[ib+1] << 2) & 0x700)],
                                                    iq1s_grid[qs[5] | ((qh[ib+1] << 5) & 0x700)], iq1s_grid[qs[4] | ((qh[ib+1] << 8) & 0x700)]);
#اگر_ختم
            qs += 8;
            ثابت __m256i q8b_1 = _mm256_loadu_si256((ثابت __m256i*)q8); q8 += 32;
            ثابت __m256i q8b_2 = _mm256_loadu_si256((ثابت __m256i*)q8); q8 += 32;

            ثابت __m256i dot1 = mul_جوڑو_epi8(q1b_1, q8b_1);
            ثابت __m256i dot2 = mul_جوڑو_epi8(q1b_2, q8b_2);
            ثابت int16_t ls1 = 2*((qh[ib+0] >> 12) & 7) + 1;
            ثابت int16_t ls2 = 2*((qh[ib+1] >> 12) & 7) + 1;
            ثابت __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16(ls1));
            ثابت __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16(ls2));

            sumi = mm256_جوڑو_epi32(sumi, mm256_جوڑو_epi32(p1, p2));
            sumi1 += (y[i].bsums[2*ib+0] + y[i].bsums[2*ib+1]) * (qh[ib+0] & 0x8000 ? -1 : 1) * ls1
                   + (y[i].bsums[2*ib+2] + y[i].bsums[2*ib+3]) * (qh[ib+1] & 0x8000 ? -1 : 1) * ls2;
        }

        ثابت اعشاری d = y[i].d * GGML_CPU_FP16_مقصد_FP32(x[i].d);
        accum = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), accum);
        accum1 += d * sumi1;

    }

    *s = hsum_float_8(accum) + IQ1S_DELTA * accum1;

#وگرنہ_اگر defined __AVX__
    __m256 accum = _mm256_setzero_ps();
    اعشاری accum1 = 0;
    برائے (صحیح i = 0; i < nb; ++i) {

        ثابت int8_t   * q8 = y[i].qs;
        ثابت uint8_t  * qs = x[i].qs;
        ثابت uint16_t * qh = x[i].qh;

        __m128i sumi1_0 = _mm_setzero_si128();
        __m128i sumi1_1 = _mm_setzero_si128();
        صحیح sumi1 = 0;
        برائے (صحیح ib = 0; ib < QK_K/32; ib += 2) {
            ثابت __m128i q1b_1_0 = mm_سیٹ_epi64x(iq1s_grid[qs[1] | ((qh[ib+0] << 5) & 0x700)], iq1s_grid[qs[0] | ((qh[ib+0] << 8) & 0x700)]);
            ثابت __m128i q1b_1_1 = mm_سیٹ_epi64x(iq1s_grid[qs[3] | ((qh[ib+0] >> 1) & 0x700)], iq1s_grid[qs[2] | ((qh[ib+0] << 2) & 0x700)]);
            ثابت __m128i q1b_2_0 = mm_سیٹ_epi64x(iq1s_grid[qs[5] | ((qh[ib+1] << 5) & 0x700)], iq1s_grid[qs[4] | ((qh[ib+1] << 8) & 0x700)]);
            ثابت __m128i q1b_2_1 = mm_سیٹ_epi64x(iq1s_grid[qs[7] | ((qh[ib+1] >> 1) & 0x700)], iq1s_grid[qs[6] | ((qh[ib+1] << 2) & 0x700)]);
            qs += 8;
            ثابت __m128i q8b_1_0 = _mm_loadu_si128((ثابت __m128i *)q8); q8 += 16;
            ثابت __m128i q8b_1_1 = _mm_loadu_si128((ثابت __m128i *)q8); q8 += 16;
            ثابت __m128i q8b_2_0 = _mm_loadu_si128((ثابت __m128i *)q8); q8 += 16;
            ثابت __m128i q8b_2_1 = _mm_loadu_si128((ثابت __m128i *)q8); q8 += 16;

            ثابت __m128i dot1_0 = mul_جوڑو_epi8_sse(q1b_1_0, q8b_1_0);
            ثابت __m128i dot1_1 = mul_جوڑو_epi8_sse(q1b_1_1, q8b_1_1);
            ثابت __m128i dot2_0 = mul_جوڑو_epi8_sse(q1b_2_0, q8b_2_0);
            ثابت __m128i dot2_1 = mul_جوڑو_epi8_sse(q1b_2_1, q8b_2_1);
            ثابت int16_t ls1 = 2*((qh[ib+0] >> 12) & 7) + 1;
            ثابت int16_t ls2 = 2*((qh[ib+1] >> 12) & 7) + 1;
            ثابت __m128i p1_0 = _mm_madd_epi16(dot1_0, _mm_set1_epi16(ls1));
            ثابت __m128i p1_1 = _mm_madd_epi16(dot1_1, _mm_set1_epi16(ls1));
            ثابت __m128i p2_0 = _mm_madd_epi16(dot2_0, _mm_set1_epi16(ls2));
            ثابت __m128i p2_1 = _mm_madd_epi16(dot2_1, _mm_set1_epi16(ls2));

            sumi1_0 = mm_جوڑو_epi32(sumi1_0, mm_جوڑو_epi32(p1_0, p2_0));
            sumi1_1 = mm_جوڑو_epi32(sumi1_1, mm_جوڑو_epi32(p1_1, p2_1));
            sumi1 += (y[i].bsums[2*ib+0] + y[i].bsums[2*ib+1]) * (qh[ib+0] & 0x8000 ? -1 : 1) * ls1
                   + (y[i].bsums[2*ib+2] + y[i].bsums[2*ib+3]) * (qh[ib+1] & 0x8000 ? -1 : 1) * ls2;
        }

        ثابت اعشاری d = y[i].d * GGML_CPU_FP16_مقصد_FP32(x[i].d);
        accum = mm256_جوڑو_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(MM256_سیٹ_M128_I(sumi1_1, sumi1_0))), accum);
        accum1 += d * sumi1;

    }

    *s = hsum_float_8(accum) + IQ1S_DELTA * accum1;

#ورنہ
    غیر_مستعمل(x);
    غیر_مستعمل(y);
    غیر_مستعمل(nb);
    ggml_vec_dot_iq1_s_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#اگر_ختم
}

خالی ggml_vec_dot_iq1_m_q8_K(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc) {
    أكد(n % QK_K == 0);
    أكد(nrc == 1);
    غیر_مستعمل(nrc);
    غیر_مستعمل(bx);
    غیر_مستعمل(by);
    غیر_مستعمل(bs);

    ثابت بلاک_iq1_m * GGML_RESTRICT x = vx;
    ثابت بلاک_q8_K  * GGML_RESTRICT y = vy;

    ثابت صحیح nb = n / QK_K;

    iq1m_scale_t scale;

#اگر defined __AVX2__

    ثابت __m256i نقاب = _mm256_set1_epi16(0x7);
    ثابت __m256i mone = _mm256_set1_epi16(1);
    ثابت __m256i mone8 = _mm256_set1_epi8(1);
    ثابت __m256i mtwo8 = _mm256_set1_epi8(2);
    // VPSHUFB cannot cross 128-bit lanes so odd shifts go to upper half.
    ثابت __m256i scales_سرکاؤ = mm256_سیٹ_epi64x(9, 3, 6, 0);

    __m256 accum1 = _mm256_setzero_ps();
    __m256 accum2 = _mm256_setzero_ps();
    برائے (صحیح i = 0; i < nb; ++i) {

        ثابت int8_t   * q8 = y[i].qs;
        ثابت uint8_t  * qs = x[i].qs;
        ثابت uint8_t  * qh = x[i].qh;
        ثابت uint16_t * sc = (ثابت uint16_t *)x[i].scales;

        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        // Extract 3-bit scales (16 values)
        __m256i scales = _mm256_set1_epi64x(*(ثابت uint64_t*)sc);
        scales = _mm256_srlv_epi64(scales, scales_سرکاؤ);
        scales = mm256_جوڑو_epi16(_mm256_slli_epi16(_mm256_and_si256(scales, نقاب), 1), mone);

        // Indices to repeat each scale 8 times.
        __m256i scales_idx1 = _mm256_set1_epi16(0x0100);
        __m256i scales_idx2 = mm256_جوڑو_epi8(scales_idx1, _mm256_set1_epi8(8));

        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        برائے (صحیح ib = 0; ib < QK_K/32; ib += 2) {
#اگر_تعریف_شدہ __BMI2__
            ثابت uint64_t packed_idx1 = _pdep_u64(*(ثابت uint32_t *)qs, 0x00ff00ff00ff00ffULL)
                                       | _pdep_u64(*(ثابت uint16_t*)(qh) & 0x7777, 0xf000f000f000f00ULL);
            ثابت uint64_t packed_idx2 = _pdep_u64(*(ثابت uint32_t *)(qs + 4), 0x00ff00ff00ff00ffULL)
                                       | _pdep_u64(*(ثابت uint16_t*)(qh + 2) & 0x7777, 0xf000f000f000f00ULL);
            ثابت uint16_t *idx1 = (ثابت uint16_t *)(&packed_idx1);
            ثابت uint16_t *idx2 = (ثابت uint16_t *)(&packed_idx2);
            ثابت __m256i q1b_1 = mm256_سیٹ_epi64x(iq1s_grid[idx1[3]], iq1s_grid[idx1[2]], iq1s_grid[idx1[1]], iq1s_grid[idx1[0]]);
            ثابت __m256i q1b_2 = mm256_سیٹ_epi64x(iq1s_grid[idx2[3]], iq1s_grid[idx2[2]], iq1s_grid[idx2[1]], iq1s_grid[idx2[0]]);

            // Convert signs to bytes 0x81 (negative) or 0x01 (positive)
            ثابت uint64_t delta_sign = _pdep_u64(*(ثابت uint32_t*)(qh) & 0x88888888, 0xf0f0f0f0f0f0f0f0ULL);
            ثابت __m256i delta1 = _mm256_or_si256(mone8, _mm256_cvtepi8_epi64(_mm_set1_epi32(delta_sign)));
            ثابت __m256i delta2 = _mm256_or_si256(mone8, _mm256_cvtepi8_epi64(_mm_set1_epi32(delta_sign >> 32)));
#ورنہ
            ثابت __m256i q1b_1 = mm256_سیٹ_epi64x(
                    iq1s_grid[qs[3] | (((uint16_t)qh[1] << 4) & 0x700)], iq1s_grid[qs[2] | (((uint16_t)qh[1] << 8) & 0x700)],
                    iq1s_grid[qs[1] | (((uint16_t)qh[0] << 4) & 0x700)], iq1s_grid[qs[0] | (((uint16_t)qh[0] << 8) & 0x700)]
            );
            ثابت __m256i q1b_2 = mm256_سیٹ_epi64x(
                    iq1s_grid[qs[7] | (((uint16_t)qh[3] << 4) & 0x700)], iq1s_grid[qs[6] | (((uint16_t)qh[3] << 8) & 0x700)],
                    iq1s_grid[qs[5] | (((uint16_t)qh[2] << 4) & 0x700)], iq1s_grid[qs[4] | (((uint16_t)qh[2] << 8) & 0x700)]
            );

            ثابت __m256i delta1 = mm256_سیٹ_epi64x(qh[1] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[1] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[0] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[0] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101);
            ثابت __m256i delta2 = mm256_سیٹ_epi64x(qh[3] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[3] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[2] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[2] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101);
#اگر_ختم
            ثابت __m256i q8b_1 = _mm256_loadu_si256((ثابت __m256i*)q8); q8 += 32;
            ثابت __m256i q8b_2 = _mm256_loadu_si256((ثابت __m256i*)q8); q8 += 32;

            ثابت __m256i dot1 = mul_جوڑو_epi8(q1b_1, q8b_1);
            ثابت __m256i dot2 = mul_جوڑو_epi8(q1b_2, q8b_2);
            ثابت __m256i dot3 = _mm256_maddubs_epi16(mone8, _mm256_sign_epi8(q8b_1, delta1));
            ثابت __m256i dot4 = _mm256_maddubs_epi16(mone8, _mm256_sign_epi8(q8b_2, delta2));

            __m256i scale1 = _mm256_shuffle_epi8(scales, scales_idx1);
            __m256i scale2 = _mm256_shuffle_epi8(scales, scales_idx2);

            scales_idx1 = mm256_جوڑو_epi8(scales_idx1, mtwo8);
            scales_idx2 = mm256_جوڑو_epi8(scales_idx2, mtwo8);

            ثابت __m256i p1 = _mm256_madd_epi16(dot1, scale1);
            ثابت __m256i p2 = _mm256_madd_epi16(dot2, scale2);
            ثابت __m256i p3 = _mm256_madd_epi16(dot3, scale1);
            ثابت __m256i p4 = _mm256_madd_epi16(dot4, scale2);

            sumi1 = mm256_جوڑو_epi32(sumi1, mm256_جوڑو_epi32(p1, p2));
            sumi2 = mm256_جوڑو_epi32(sumi2, mm256_جوڑو_epi32(p3, p4));

            qs += 8; qh += 4;
        }

        ثابت __m256 d = _mm256_set1_ps(y[i].d * GGML_CPU_FP16_مقصد_FP32(scale.f16));

        accum1 = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sumi1), accum1);
        accum2 = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sumi2), accum2);
    }

    *s = hsum_float_8(accum1) + IQ1M_DELTA * hsum_float_8(accum2);

#وگرنہ_اگر defined __AVX__
    ثابت __m128i نقاب = _mm_set1_epi16(0x7);
    ثابت __m128i mone = _mm_set1_epi16(1);

    __m256 accum1 = _mm256_setzero_ps();
    __m256 accum2 = _mm256_setzero_ps();
    برائے (صحیح i = 0; i < nb; ++i) {

        ثابت int8_t   * q8 = y[i].qs;
        ثابت uint8_t  * qs = x[i].qs;
        ثابت uint8_t  * qh = x[i].qh;
        ثابت uint16_t * sc = (ثابت uint16_t *)x[i].scales;

        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

        __m128i sumi1_0 = _mm_setzero_si128();
        __m128i sumi1_1 = _mm_setzero_si128();
        __m128i sumi2_0 = _mm_setzero_si128();
        __m128i sumi2_1 = _mm_setzero_si128();
        برائے (صحیح ib = 0; ib < QK_K/32; ib += 2) {
            ثابت __m128i q1b_1_0 = mm_سیٹ_epi64x(
                    iq1s_grid[qs[1] | (((uint16_t)qh[0] << 4) & 0x700)], iq1s_grid[qs[0] | (((uint16_t)qh[0] << 8) & 0x700)]);
            ثابت __m128i q1b_1_1 = mm_سیٹ_epi64x(
                    iq1s_grid[qs[3] | (((uint16_t)qh[1] << 4) & 0x700)], iq1s_grid[qs[2] | (((uint16_t)qh[1] << 8) & 0x700)]);
            ثابت __m128i q1b_2_0 = mm_سیٹ_epi64x(
                    iq1s_grid[qs[5] | (((uint16_t)qh[2] << 4) & 0x700)], iq1s_grid[qs[4] | (((uint16_t)qh[2] << 8) & 0x700)]);
            ثابت __m128i q1b_2_1 = mm_سیٹ_epi64x(
                    iq1s_grid[qs[7] | (((uint16_t)qh[3] << 4) & 0x700)], iq1s_grid[qs[6] | (((uint16_t)qh[3] << 8) & 0x700)]);
            ثابت __m128i q8b_1_0 = _mm_loadu_si128((ثابت __m128i *)q8); q8 += 16;
            ثابت __m128i q8b_1_1 = _mm_loadu_si128((ثابت __m128i *)q8); q8 += 16;
            ثابت __m128i q8b_2_0 = _mm_loadu_si128((ثابت __m128i *)q8); q8 += 16;
            ثابت __m128i q8b_2_1 = _mm_loadu_si128((ثابت __m128i *)q8); q8 += 16;

            ثابت __m128i dot1_0 = mul_جوڑو_epi8_sse(q1b_1_0, q8b_1_0);
            ثابت __m128i dot1_1 = mul_جوڑو_epi8_sse(q1b_1_1, q8b_1_1);
            ثابت __m128i dot2_0 = mul_جوڑو_epi8_sse(q1b_2_0, q8b_2_0);
            ثابت __m128i dot2_1 = mul_جوڑو_epi8_sse(q1b_2_1, q8b_2_1);

            ثابت __m128i delta1_0 = mm_سیٹ_epi64x(qh[0] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[0] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101);
            ثابت __m128i delta1_1 = mm_سیٹ_epi64x(qh[1] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[1] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101);
            ثابت __m128i delta2_0 = mm_سیٹ_epi64x(qh[2] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[2] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101);
            ثابت __m128i delta2_1 = mm_سیٹ_epi64x(qh[3] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[3] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101);

            ثابت __m128i dot3_0 = mul_جوڑو_epi8_sse(delta1_0, q8b_1_0);
            ثابت __m128i dot3_1 = mul_جوڑو_epi8_sse(delta1_1, q8b_1_1);
            ثابت __m128i dot4_0 = mul_جوڑو_epi8_sse(delta2_0, q8b_2_0);
            ثابت __m128i dot4_1 = mul_جوڑو_epi8_sse(delta2_1, q8b_2_1);

            __m128i scale1_0 = _mm_set1_epi16(sc[ib/2] >> 0);
            __m128i scale1_1 = _mm_set1_epi16(sc[ib/2] >> 3);
            __m128i scale2_0 = _mm_set1_epi16(sc[ib/2] >> 6);
            __m128i scale2_1 = _mm_set1_epi16(sc[ib/2] >> 9);

            scale1_0 = mm_جوڑو_epi16(_mm_slli_epi16(_mm_and_si128(scale1_0, نقاب), 1), mone);
            scale1_1 = mm_جوڑو_epi16(_mm_slli_epi16(_mm_and_si128(scale1_1, نقاب), 1), mone);
            scale2_0 = mm_جوڑو_epi16(_mm_slli_epi16(_mm_and_si128(scale2_0, نقاب), 1), mone);
            scale2_1 = mm_جوڑو_epi16(_mm_slli_epi16(_mm_and_si128(scale2_1, نقاب), 1), mone);
            ثابت __m128i p1_0 = _mm_madd_epi16(dot1_0, scale1_0);
            ثابت __m128i p1_1 = _mm_madd_epi16(dot1_1, scale1_1);
            ثابت __m128i p2_0 = _mm_madd_epi16(dot2_0, scale2_0);
            ثابت __m128i p2_1 = _mm_madd_epi16(dot2_1, scale2_1);
            ثابت __m128i p3_0 = _mm_madd_epi16(dot3_0, scale1_0);
            ثابت __m128i p3_1 = _mm_madd_epi16(dot3_1, scale1_1);
            ثابت __m128i p4_0 = _mm_madd_epi16(dot4_0, scale2_0);
            ثابت __m128i p4_1 = _mm_madd_epi16(dot4_1, scale2_1);

            sumi1_0 = mm_جوڑو_epi32(sumi1_0, mm_جوڑو_epi32(p1_0, p2_0));
            sumi1_1 = mm_جوڑو_epi32(sumi1_1, mm_جوڑو_epi32(p1_1, p2_1));
            sumi2_0 = mm_جوڑو_epi32(sumi2_0, mm_جوڑو_epi32(p3_0, p4_0));
            sumi2_1 = mm_جوڑو_epi32(sumi2_1, mm_جوڑو_epi32(p3_1, p4_1));

            qs += 8; qh += 4;
        }

        ثابت __m256 d = _mm256_set1_ps(y[i].d * GGML_CPU_FP16_مقصد_FP32(scale.f16));

        accum1 = mm256_جوڑو_ps(_mm256_mul_ps(d, _mm256_cvtepi32_ps(MM256_سیٹ_M128_I(sumi1_1, sumi1_0))), accum1);
        accum2 = mm256_جوڑو_ps(_mm256_mul_ps(d, _mm256_cvtepi32_ps(MM256_سیٹ_M128_I(sumi2_1, sumi2_0))), accum2);
    }

    *s = hsum_float_8(accum1) + IQ1M_DELTA * hsum_float_8(accum2);

#ورنہ
    غیر_مستعمل(x);
    غیر_مستعمل(y);
    غیر_مستعمل(nb);
    غیر_مستعمل(scale);
    ggml_vec_dot_iq1_m_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#اگر_ختم
}

خالی ggml_vec_dot_iq4_nl_q8_0(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc) {
    أكد(nrc == 1);
    غیر_مستعمل(nrc);
    غیر_مستعمل(bx);
    غیر_مستعمل(by);
    غیر_مستعمل(bs);
    أكد(n % QK4_NL == 0);
    static_assert(QK4_NL == QK8_0, "QK4_NL and QK8_0 must be the same");

    ثابت بلاک_iq4_nl * GGML_RESTRICT x = vx;
    ثابت بلاک_q8_0   * GGML_RESTRICT y = vy;

    ثابت صحیح nb = n / QK4_NL;

    صحیح ib = 0;
    اعشاری sumf = 0;

#اگر defined __AVX2__

    ثابت __m128i values128 = _mm_loadu_si128((ثابت __m128i*)kvalues_iq4nl);
    ثابت __m128i m4b  = _mm_set1_epi8(0x0f);
    ثابت __m256i mone = _mm256_set1_epi16(1);

    __m256 accum1 = _mm256_setzero_ps();
    __m256 accum2 = _mm256_setzero_ps();
    برائے (; ib + 1 < nb; ib += 2) {
        ثابت __m128i q4bits_1 = _mm_loadu_si128((ثابت __m128i*)x[ib + 0].qs);
        ثابت __m128i q4bits_2 = _mm_loadu_si128((ثابت __m128i*)x[ib + 1].qs);
        ثابت __m256i q8b_1 = _mm256_loadu_si256((ثابت __m256i *)y[ib + 0].qs);
        ثابت __m256i q8b_2 = _mm256_loadu_si256((ثابت __m256i *)y[ib + 1].qs);
        ثابت __m256i q4b_1 = MM256_سیٹ_M128_I(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b)),
                                              _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b)));
        ثابت __m256i q4b_2 = MM256_سیٹ_M128_I(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b)),
                                              _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b)));
        ثابت __m256i p16_1 = mul_جوڑو_epi8(q4b_1, q8b_1);
        ثابت __m256i p16_2 = mul_جوڑو_epi8(q4b_2, q8b_2);
        ثابت __m256i p_1 = _mm256_madd_epi16(p16_1, mone);
        ثابت __m256i p_2 = _mm256_madd_epi16(p16_2, mone);
        accum1 = _mm256_fmadd_ps(_mm256_set1_ps(GGML_CPU_FP16_مقصد_FP32(y[ib + 0].d)*GGML_CPU_FP16_مقصد_FP32(x[ib + 0].d)),
                _mm256_cvtepi32_ps(p_1), accum1);
        accum2 = _mm256_fmadd_ps(_mm256_set1_ps(GGML_CPU_FP16_مقصد_FP32(y[ib + 1].d)*GGML_CPU_FP16_مقصد_FP32(x[ib + 1].d)),
                _mm256_cvtepi32_ps(p_2), accum2);
    }

    sumf = hsum_float_8(mm256_جوڑو_ps(accum1, accum2));

#وگرنہ_اگر defined __AVX__
    ثابت __m128i values128 = _mm_loadu_si128((ثابت __m128i*)kvalues_iq4nl);
    ثابت __m128i m4b  = _mm_set1_epi8(0x0f);

    __m256 accum = _mm256_setzero_ps();
    برائے (; ib + 1 < nb; ib += 2) {
        ثابت __m128i q4bits_1 = _mm_loadu_si128((ثابت __m128i *)x[ib + 0].qs);
        ثابت __m128i q4bits_2 = _mm_loadu_si128((ثابت __m128i *)x[ib + 1].qs);
        ثابت __m128i q8b_1_0 = _mm_loadu_si128((ثابت __m128i *)y[ib + 0].qs);
        ثابت __m128i q8b_1_1 = _mm_loadu_si128((ثابت __m128i *)y[ib + 0].qs + 1);
        ثابت __m128i q8b_2_0 = _mm_loadu_si128((ثابت __m128i *)y[ib + 1].qs);
        ثابت __m128i q8b_2_1 = _mm_loadu_si128((ثابت __m128i *)y[ib + 1].qs + 1);

        ثابت __m128i q4b_1_0 = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b));
        ثابت __m128i q4b_1_1 = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b));
        ثابت __m128i q4b_2_0 = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b));
        ثابت __m128i q4b_2_1 = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b));

        ثابت __m256 p = mul_مجموع_i8_quad_float(q4b_1_0, q4b_1_1, q4b_2_0, q4b_2_1, q8b_1_0, q8b_1_1, q8b_2_0, q8b_2_1);
        ثابت __m256 deltas = quad_fp16_delta_float(x[ib].d, y[ib].d, x[ib + 1].d, y[ib + 1].d);
        accum = mm256_جوڑو_ps(_mm256_mul_ps(deltas, p), accum);
    }

    sumf = hsum_float_8(accum);

#اگر_ختم
    برائے (; ib < nb; ++ib) {
        ثابت اعشاری d = GGML_CPU_FP16_مقصد_FP32(y[ib].d)*GGML_CPU_FP16_مقصد_FP32(x[ib].d);
        صحیح sumi1 = 0, sumi2 = 0;
        برائے (صحیح j = 0; j < QK4_NL/2; ++j) {
            sumi1 += y[ib].qs[j+       0] * kvalues_iq4nl[x[ib].qs[j] & 0xf];
            sumi2 += y[ib].qs[j+QK4_NL/2] * kvalues_iq4nl[x[ib].qs[j] >>  4];
        }
        sumf += d * (sumi1 + sumi2);
    }
    *s = sumf;
}

خالی ggml_vec_dot_iq4_xs_q8_K(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc) {
    أكد(nrc == 1);
    غیر_مستعمل(nrc);
    غیر_مستعمل(bx);
    غیر_مستعمل(by);
    غیر_مستعمل(bs);
    أكد(n % QK_K == 0);

    ثابت بلاک_iq4_xs * GGML_RESTRICT x = vx;
    ثابت بلاک_q8_K   * GGML_RESTRICT y = vy;

    ثابت صحیح nb = n / QK_K;

#اگر defined __AVX2__

    ثابت __m128i values128 = _mm_loadu_si128((ثابت __m128i*)kvalues_iq4nl);
    ثابت __m128i m4b  = _mm_set1_epi8(0x0f);

    __m256 accum = _mm256_setzero_ps();
    برائے (صحیح ibl = 0; ibl < nb; ++ibl) {
        ثابت uint8_t * qs = x[ibl].qs;
        ثابت int8_t  * q8 = y[ibl].qs;
        uint16_t sh = x[ibl].scales_h;
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        برائے (صحیح ib = 0; ib < QK_K/32; ib += 2) {
            ثابت __m128i q4bits_1 = _mm_loadu_si128((ثابت __m128i*)qs);  qs += 16;
            ثابت __m128i q4bits_2 = _mm_loadu_si128((ثابت __m128i*)qs);  qs += 16;
            ثابت __m256i q8b_1 = _mm256_loadu_si256((ثابت __m256i *)q8); q8 += 32;
            ثابت __m256i q8b_2 = _mm256_loadu_si256((ثابت __m256i *)q8); q8 += 32;
            ثابت __m256i q4b_1 = MM256_سیٹ_M128_I(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b)),
                                                  _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b)));
            ثابت __m256i q4b_2 = MM256_سیٹ_M128_I(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b)),
                                                  _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b)));
            ثابت __m256i p16_1 = mul_جوڑو_epi8(q4b_1, q8b_1);
            ثابت __m256i p16_2 = mul_جوڑو_epi8(q4b_2, q8b_2);
            ثابت int16_t ls1 = ((x[ibl].scales_l[ib/2] & 0xf) | ((sh << 4) & 0x30)) - 32;
            ثابت int16_t ls2 = ((x[ibl].scales_l[ib/2] >>  4) | ((sh << 2) & 0x30)) - 32;
            sh >>= 4;
            ثابت __m256i p_1 = _mm256_madd_epi16(p16_1, _mm256_set1_epi16(ls1));
            ثابت __m256i p_2 = _mm256_madd_epi16(p16_2, _mm256_set1_epi16(ls2));
            sumi1 = mm256_جوڑو_epi32(p_1, sumi1);
            sumi2 = mm256_جوڑو_epi32(p_2, sumi2);
        }
        accum = _mm256_fmadd_ps(_mm256_set1_ps(GGML_CPU_FP16_مقصد_FP32(x[ibl].d)*y[ibl].d),
                _mm256_cvtepi32_ps(mm256_جوڑو_epi32(sumi1, sumi2)), accum);
    }

    *s = hsum_float_8(accum);

#وگرنہ_اگر defined __AVX__
    ثابت __m128i values128 = _mm_loadu_si128((ثابت __m128i*)kvalues_iq4nl);
    ثابت __m128i m4b  = _mm_set1_epi8(0x0f);

    __m256 accum = _mm256_setzero_ps();
    برائے (صحیح ibl = 0; ibl < nb; ++ibl) {
        ثابت uint8_t * qs = x[ibl].qs;
        ثابت int8_t  * q8 = y[ibl].qs;
        uint16_t sh = x[ibl].scales_h;
        __m128i sumi1_0 = _mm_setzero_si128();
        __m128i sumi1_1 = _mm_setzero_si128();
        __m128i sumi2_0 = _mm_setzero_si128();
        __m128i sumi2_1 = _mm_setzero_si128();
        برائے (صحیح ib = 0; ib < QK_K/32; ib += 2) {
            ثابت __m128i q4bits_1 = _mm_loadu_si128((ثابت __m128i *)qs); qs += 16;
            ثابت __m128i q4bits_2 = _mm_loadu_si128((ثابت __m128i *)qs); qs += 16;
            ثابت __m128i q8b_1_0 = _mm_loadu_si128((ثابت __m128i *)q8); q8 += 16;
            ثابت __m128i q8b_1_1 = _mm_loadu_si128((ثابت __m128i *)q8); q8 += 16;
            ثابت __m128i q8b_2_0 = _mm_loadu_si128((ثابت __m128i *)q8); q8 += 16;
            ثابت __m128i q8b_2_1 = _mm_loadu_si128((ثابت __m128i *)q8); q8 += 16;
            ثابت __m128i q4b_1_0 = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b));
            ثابت __m128i q4b_1_1 = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b));
            ثابت __m128i q4b_2_0 = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b));
            ثابت __m128i q4b_2_1 = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b));
            ثابت __m128i p16_1_0 = mul_جوڑو_epi8_sse(q4b_1_0, q8b_1_0);
            ثابت __m128i p16_1_1 = mul_جوڑو_epi8_sse(q4b_1_1, q8b_1_1);
            ثابت __m128i p16_2_0 = mul_جوڑو_epi8_sse(q4b_2_0, q8b_2_0);
            ثابت __m128i p16_2_1 = mul_جوڑو_epi8_sse(q4b_2_1, q8b_2_1);
            ثابت int16_t ls1 = ((x[ibl].scales_l[ib/2] & 0xf) | ((sh << 4) & 0x30)) - 32;
            ثابت int16_t ls2 = ((x[ibl].scales_l[ib/2] >>  4) | ((sh << 2) & 0x30)) - 32;
            sh >>= 4;
            ثابت __m128i p_1_0 = _mm_madd_epi16(p16_1_0, _mm_set1_epi16(ls1));
            ثابت __m128i p_1_1 = _mm_madd_epi16(p16_1_1, _mm_set1_epi16(ls1));
            ثابت __m128i p_2_0 = _mm_madd_epi16(p16_2_0, _mm_set1_epi16(ls2));
            ثابت __m128i p_2_1 = _mm_madd_epi16(p16_2_1, _mm_set1_epi16(ls2));
            sumi1_0 = mm_جوڑو_epi32(p_1_0, sumi1_0);
            sumi1_1 = mm_جوڑو_epi32(p_1_1, sumi1_1);
            sumi2_0 = mm_جوڑو_epi32(p_2_0, sumi2_0);
            sumi2_1 = mm_جوڑو_epi32(p_2_1, sumi2_1);
        }
        __m128i sumi12_0 = mm_جوڑو_epi32(sumi1_0, sumi2_0);
        __m128i sumi12_1 = mm_جوڑو_epi32(sumi1_1, sumi2_1);
        accum = mm256_جوڑو_ps(_mm256_mul_ps(_mm256_set1_ps(GGML_CPU_FP16_مقصد_FP32(x[ibl].d)*y[ibl].d),
                _mm256_cvtepi32_ps(MM256_سیٹ_M128_I(sumi12_1, sumi12_0))), accum);
    }

    *s = hsum_float_8(accum);

#ورنہ
    غیر_مستعمل(x);
    غیر_مستعمل(y);
    غیر_مستعمل(nb);
    ggml_vec_dot_iq4_xs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#اگر_ختم
}
