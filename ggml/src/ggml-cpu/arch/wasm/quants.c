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
#شامل <float.h>
#شامل <stdlib.h> // for qsort
#شامل <stdio.h>  // for GGML_ASSERT

#تعریف_کرو گروہ_MAX_EPS 1e-15f
#تعریف_کرو گروہ_MAX_EPS_IQ3_XXS 1e-8f
#تعریف_کرو گروہ_MAX_EPS_IQ2_S 1e-8f
#تعریف_کرو گروہ_MAX_EPS_IQ1_M 1e-7f
#تعریف_کرو گروہ_MAX_EPS_IQ1_S 1e-12f

#تعریف_کرو غیر_مستعمل GGML_غیر_مستعمل

#اگر defined(__wasm_simd128__)
#تعریف_کرو B1(c,s,n)  0x ## n ## c ,  0x ## n ## s
#تعریف_کرو B2(c,s,n) B1(c,s,n ## c), B1(c,s,n ## s)
#تعریف_کرو B3(c,s,n) B2(c,s,n ## c), B2(c,s,n ## s)
#تعریف_کرو B4(c,s,n) B3(c,s,n ## c), B3(c,s,n ## s)
#تعریف_کرو B5(c,s,n) B4(c,s,n ## c), B4(c,s,n ## s)
#تعریف_کرو B6(c,s,n) B5(c,s,n ## c), B5(c,s,n ## s)
#تعریف_کرو B7(c,s,n) B6(c,s,n ## c), B6(c,s,n ## s)
#تعریف_کرو B8(c,s  ) B7(c,s,     c), B7(c,s,     s)

// precomputed tables for expanding 8bits to 8 bytes:
ساکن ثابت uint64_t جدول_b2b_0[1 << 8] = { B8(00, 10) }; // ( b) << 4
ساکن ثابت uint64_t جدول_b2b_1[1 << 8] = { B8(10, 00) }; // (!b) << 4
#اگر_ختم

خالی quantize_صف_q8_0(ثابت اعشاری * GGML_RESTRICT x, خالی * GGML_RESTRICT vy, int64_t k) {
    أكد(QK8_0 == 32);
    أكد(k % QK8_0 == 0);
    ثابت صحیح nb = k / QK8_0;

    بلاک_q8_0 * GGML_RESTRICT y = vy;

#اگر defined __wasm_simd128__
    برائے (صحیح i = 0; i < nb; i++) {
        v128_t srcv [8];
        v128_t asrcv[8];
        v128_t amaxv[8];

        برائے (صحیح j = 0; j < 8; j++) srcv[j]  = wasm_v128_بھرو(x + i*32 + 4*j);
        برائے (صحیح j = 0; j < 8; j++) asrcv[j] = wasm_f32x4_abs(srcv[j]);

        برائے (صحیح j = 0; j < 4; j++) amaxv[2*j] = wasm_f32x4_max(asrcv[2*j], asrcv[2*j+1]);
        برائے (صحیح j = 0; j < 2; j++) amaxv[4*j] = wasm_f32x4_max(amaxv[4*j], amaxv[4*j+2]);
        برائے (صحیح j = 0; j < 1; j++) amaxv[8*j] = wasm_f32x4_max(amaxv[8*j], amaxv[8*j+4]);

        ثابت اعشاری amax = MAX(MAX(wasm_f32x4_نکالو_lane(amaxv[0], 0),
                                   wasm_f32x4_نکالو_lane(amaxv[0], 1)),
                               MAX(wasm_f32x4_نکالو_lane(amaxv[0], 2),
                                   wasm_f32x4_نکالو_lane(amaxv[0], 3)));

        ثابت اعشاری d = amax / ((1 << 7) - 1);
        ثابت اعشاری id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_CPU_FP32_مقصد_FP16(d);

        برائے (صحیح j = 0; j < 8; j++) {
            ثابت v128_t v  = wasm_f32x4_mul(srcv[j], wasm_f32x4_splat(id));
            ثابت v128_t vi = wasm_i32x4_trunc_sat_f32x4(v);

            y[i].qs[4*j + 0] = wasm_i32x4_نکالو_lane(vi, 0);
            y[i].qs[4*j + 1] = wasm_i32x4_نکالو_lane(vi, 1);
            y[i].qs[4*j + 2] = wasm_i32x4_نکالو_lane(vi, 2);
            y[i].qs[4*j + 3] = wasm_i32x4_نکالو_lane(vi, 3);
        }
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
#اگر defined __wasm_simd128__
    برائے (صحیح i = 0; i < nb; i++) {
        v128_t srcv [8];
        v128_t asrcv[8];
        v128_t amaxv[8];

        برائے (صحیح j = 0; j < 8; j++) srcv[j]  = wasm_v128_بھرو(x + i*32 + 4*j);
        برائے (صحیح j = 0; j < 8; j++) asrcv[j] = wasm_f32x4_abs(srcv[j]);

        برائے (صحیح j = 0; j < 4; j++) amaxv[2*j] = wasm_f32x4_max(asrcv[2*j], asrcv[2*j+1]);
        برائے (صحیح j = 0; j < 2; j++) amaxv[4*j] = wasm_f32x4_max(amaxv[4*j], amaxv[4*j+2]);
        برائے (صحیح j = 0; j < 1; j++) amaxv[8*j] = wasm_f32x4_max(amaxv[8*j], amaxv[8*j+4]);

        ثابت اعشاری amax = MAX(MAX(wasm_f32x4_نکالو_lane(amaxv[0], 0),
                                   wasm_f32x4_نکالو_lane(amaxv[0], 1)),
                               MAX(wasm_f32x4_نکالو_lane(amaxv[0], 2),
                                   wasm_f32x4_نکالو_lane(amaxv[0], 3)));

        ثابت اعشاری d = amax / ((1 << 7) - 1);
        ثابت اعشاری id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_CPU_FP32_مقصد_FP16(d);

        v128_t accv = wasm_i32x4_splat(0);

        برائے (صحیح j = 0; j < 8; j++) {
            ثابت v128_t v  = wasm_f32x4_mul(srcv[j], wasm_f32x4_splat(id));
            ثابت v128_t vi = wasm_i32x4_trunc_sat_f32x4(v);

            y[i].qs[4*j + 0] = wasm_i32x4_نکالو_lane(vi, 0);
            y[i].qs[4*j + 1] = wasm_i32x4_نکالو_lane(vi, 1);
            y[i].qs[4*j + 2] = wasm_i32x4_نکالو_lane(vi, 2);
            y[i].qs[4*j + 3] = wasm_i32x4_نکالو_lane(vi, 3);

            accv = wasm_i32x4_جوڑو(accv, vi);
        }

        y[i].s = GGML_CPU_FP32_مقصد_FP16(
                d * (wasm_i32x4_نکالو_lane(accv, 0) +
                     wasm_i32x4_نکالو_lane(accv, 1) +
                     wasm_i32x4_نکالو_lane(accv, 2) +
                     wasm_i32x4_نکالو_lane(accv, 3)));
    }
#ورنہ
    GGML_غیر_مستعمل(nb);
    // scalar
    quantize_صف_q8_1_ref(x, y, k);
#اگر_ختم
}

//===================================== Q8_K ==============================================

خالی quantize_صف_q8_K(ثابت اعشاری * GGML_RESTRICT x, خالی * GGML_RESTRICT y, int64_t k) {
#اگر_تعریف_شدہ __wasm_simd128__
    أكد(k % QK_K == 0);
    ثابت int64_t nb = k / QK_K;
    بلاک_q8_K * GGML_RESTRICT yc = y; // Cast to proper type

    برائے (صحیح i = 0; i < nb; i++) {
        ثابت اعشاری * x_بلاک = x + i * QK_K;

        v128_t min_vec = wasm_v128_بھرو(x_بلاک);
        v128_t max_vec = min_vec;

        برائے (صحیح j = 4; j < QK_K; j += 4) {
            v128_t x_vec = wasm_v128_بھرو(x_بلاک + j);
            max_vec = wasm_f32x4_pmax(max_vec, x_vec);
            min_vec = wasm_f32x4_pmin(min_vec, x_vec);
        }
        max_vec = wasm_f32x4_pmax(max_vec, wasm_i32x4_shuffle(max_vec, max_vec, 2, 3, 0, 1));
        max_vec = wasm_f32x4_pmax(max_vec, wasm_i32x4_shuffle(max_vec, max_vec, 1, 0, 3, 2));
        min_vec = wasm_f32x4_pmin(min_vec, wasm_i32x4_shuffle(min_vec, min_vec, 2, 3, 0, 1));
        min_vec = wasm_f32x4_pmin(min_vec, wasm_i32x4_shuffle(min_vec, min_vec, 1, 0, 3, 2));
        اعشاری max = wasm_f32x4_نکالو_lane(max_vec, 0);
        اعشاری min = wasm_f32x4_نکالو_lane(min_vec, 0);
        اعشاری amax = -min > max ? min : max;

        اگر (amax == 0.0f) {
            yc[i].d = 0.0f;
            ثابت v128_t zero = wasm_i8x16_splat(0);
            برائے (صحیح j = 0; j < QK_K; j += 16) {
                wasm_v128_store(yc[i].qs + j, zero);
            }
            جاری;
        }

        ثابت اعشاری iscale = -127.0f / amax;
        ثابت v128_t scale_vec = wasm_f32x4_splat(iscale);

        // Process 16 elements per iteration
        برائے (صحیح j = 0, jb = 0; j < QK_K; j += 16, jb++) {
            // Load and quantize 16 floats
            v128_t x0 = wasm_v128_بھرو(x_بلاک + j);
            v128_t x1 = wasm_v128_بھرو(x_بلاک + j + 4);
            v128_t x2 = wasm_v128_بھرو(x_بلاک + j + 8);
            v128_t x3 = wasm_v128_بھرو(x_بلاک + j + 12);

            v128_t q0 = wasm_f32x4_قریب_ترین(wasm_f32x4_mul(x0, scale_vec));
            v128_t q1 = wasm_f32x4_قریب_ترین(wasm_f32x4_mul(x1, scale_vec));
            v128_t q2 = wasm_f32x4_قریب_ترین(wasm_f32x4_mul(x2, scale_vec));
            v128_t q3 = wasm_f32x4_قریب_ترین(wasm_f32x4_mul(x3, scale_vec));

            // Convert to i32 with saturation
            v128_t i0 = wasm_i32x4_trunc_sat_f32x4(q0);
            v128_t i1 = wasm_i32x4_trunc_sat_f32x4(q1);
            v128_t i2 = wasm_i32x4_trunc_sat_f32x4(q2);
            v128_t i3 = wasm_i32x4_trunc_sat_f32x4(q3);

            // Pack into 16 i8 values
            v128_t i8 = wasm_i8x16_narrow_i16x8(
                wasm_i16x8_narrow_i32x4(i0, i1),
                wasm_i16x8_narrow_i32x4(i2, i3)
            );
            wasm_v128_store(yc[i].qs + j, i8);

            // Calculate bsums using SIMD
            v128_t sum16 = wasm_i16x8_جوڑو(
                wasm_i16x8_extend_کم_i8x16(i8),
                wasm_i16x8_extend_بلند_i8x16(i8)
            );
            v128_t sum32 = wasm_i32x4_جوڑو(
                wasm_i32x4_extend_کم_i16x8(sum16),
                wasm_i32x4_extend_بلند_i16x8(sum16)
            );
            sum32 = wasm_i32x4_جوڑو(sum32, wasm_i32x4_shuffle(sum32, sum32, 2, 3, 0, 1));
            sum32 = wasm_i32x4_جوڑو(sum32, wasm_i32x4_shuffle(sum32, sum32, 1, 0, 3, 2));
            yc[i].bsums[jb] = wasm_i32x4_نکالو_lane(sum32, 0);
        }

        yc[i].d = 1.0f / iscale;
    }
#ورنہ
    quantize_صف_q8_K_ref(x, y, k);
#اگر_ختم
}


//===================================== Dot products =================================

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

#اگر defined __wasm_simd128__
    v128_t sumv = wasm_f32x4_splat(0.0f);

    ثابت v128_t m4b = wasm_i8x16_splat(0x0F);
    ثابت v128_t s8b = wasm_i8x16_splat(0x8);

    برائے (; ib + 1 < nb; ib += 2) {
        ثابت بلاک_q4_0 * GGML_RESTRICT x0 = &x[ib];
        ثابت بلاک_q4_0 * GGML_RESTRICT x1 = &x[ib + 1];
        ثابت بلاک_q8_0 * GGML_RESTRICT y0 = &y[ib];
        ثابت بلاک_q8_0 * GGML_RESTRICT y1 = &y[ib + 1];

        // Load and process x0
        v128_t v0_0 = wasm_v128_بھرو(x0->qs);
        v128_t v0_0l = wasm_v128_and(v0_0, m4b);
        v128_t v0_0h = wasm_u8x16_shr(v0_0, 4);
        v128_t v0_0ls = wasm_i8x16_sub(v0_0l, s8b);
        v128_t v0_0hs = wasm_i8x16_sub(v0_0h, s8b);

        // Load y0 vectors
        v128_t y0_l = wasm_v128_بھرو(y0->qs);
        v128_t y0_h = wasm_v128_بھرو(y0->qs + 16);

        // Extend to i16x8 and compute dot products
        v128_t dx0l = wasm_i16x8_extend_کم_i8x16(v0_0ls);
        v128_t dx0h = wasm_i16x8_extend_بلند_i8x16(v0_0ls);
        v128_t dx0hl = wasm_i16x8_extend_کم_i8x16(v0_0hs);
        v128_t dx0hh = wasm_i16x8_extend_بلند_i8x16(v0_0hs);

        v128_t dy0ll = wasm_i16x8_extend_کم_i8x16(y0_l);
        v128_t dy0lh = wasm_i16x8_extend_بلند_i8x16(y0_l);
        v128_t dy0hl = wasm_i16x8_extend_کم_i8x16(y0_h);
        v128_t dy0hh = wasm_i16x8_extend_بلند_i8x16(y0_h);

        v128_t dp0 = wasm_i32x4_جوڑو(
            wasm_i32x4_جوڑو(
                wasm_i32x4_dot_i16x8(dx0l, dy0ll),
                wasm_i32x4_dot_i16x8(dx0h, dy0lh)
            ),
            wasm_i32x4_جوڑو(
                wasm_i32x4_dot_i16x8(dx0hl, dy0hl),
                wasm_i32x4_dot_i16x8(dx0hh, dy0hh)
            )
        );

        // Load and process x1
        v128_t v0_1 = wasm_v128_بھرو(x1->qs);
        v128_t v0_1l = wasm_v128_and(v0_1, m4b);
        v128_t v0_1h = wasm_u8x16_shr(v0_1, 4);
        v128_t v0_1ls = wasm_i8x16_sub(v0_1l, s8b);
        v128_t v0_1hs = wasm_i8x16_sub(v0_1h, s8b);

        // Load y1 vectors
        v128_t y1_l = wasm_v128_بھرو(y1->qs);
        v128_t y1_h = wasm_v128_بھرو(y1->qs + 16);

        // Extend to i16x8 and compute dot products
        v128_t dx1l = wasm_i16x8_extend_کم_i8x16(v0_1ls);
        v128_t dx1h = wasm_i16x8_extend_بلند_i8x16(v0_1ls);
        v128_t dx1hl = wasm_i16x8_extend_کم_i8x16(v0_1hs);
        v128_t dx1hh = wasm_i16x8_extend_بلند_i8x16(v0_1hs);

        v128_t dy1ll = wasm_i16x8_extend_کم_i8x16(y1_l);
        v128_t dy1lh = wasm_i16x8_extend_بلند_i8x16(y1_l);
        v128_t dy1hl = wasm_i16x8_extend_کم_i8x16(y1_h);
        v128_t dy1hh = wasm_i16x8_extend_بلند_i8x16(y1_h);

        v128_t dp1 = wasm_i32x4_جوڑو(
            wasm_i32x4_جوڑو(
                wasm_i32x4_dot_i16x8(dx1l, dy1ll),
                wasm_i32x4_dot_i16x8(dx1h, dy1lh)
            ),
            wasm_i32x4_جوڑو(
                wasm_i32x4_dot_i16x8(dx1hl, dy1hl),
                wasm_i32x4_dot_i16x8(dx1hh, dy1hh)
            )
        );

        // Accumulate results with scaling
        اعشاری scale0 = GGML_CPU_FP16_مقصد_FP32(x0->d) * GGML_CPU_FP16_مقصد_FP32(y0->d);
        اعشاری scale1 = GGML_CPU_FP16_مقصد_FP32(x1->d) * GGML_CPU_FP16_مقصد_FP32(y1->d);

        sumv = wasm_f32x4_جوڑو(sumv, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(dp0), wasm_f32x4_splat(scale0)));
        sumv = wasm_f32x4_جوڑو(sumv, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(dp1), wasm_f32x4_splat(scale1)));
    }

    sumf = wasm_f32x4_نکالو_lane(sumv, 0) + wasm_f32x4_نکالو_lane(sumv, 1) +
           wasm_f32x4_نکالو_lane(sumv, 2) + wasm_f32x4_نکالو_lane(sumv, 3);

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

    اعشاری sumf = 0;

#اگر defined __wasm_simd128__
    v128_t sumv = wasm_f32x4_splat(0.0f);
    اعشاری summs = 0.0f;

    برائے (صحیح ib = 0; ib < nb; ++ib) {
        ثابت بلاک_q4_1 * GGML_RESTRICT x0 = &x[ib];
        ثابت بلاک_q8_1 * GGML_RESTRICT y0 = &y[ib];

        summs += GGML_CPU_FP16_مقصد_FP32(x0->m) * GGML_CPU_FP16_مقصد_FP32(y0->s);

        ثابت v128_t raw  = wasm_v128_بھرو(x0->qs);
        ثابت v128_t v0s  = wasm_v128_and(raw, wasm_i8x16_splat(0x0F));
        ثابت v128_t v1s  = wasm_u8x16_shr(raw, 4);

        ثابت v128_t ys_lo = wasm_v128_بھرو(y0->qs);
        ثابت v128_t ys_hi = wasm_v128_بھرو(y0->qs + 16);

        ثابت v128_t v0s_l = wasm_u16x8_extend_کم_u8x16(v0s);
        ثابت v128_t v0s_h = wasm_u16x8_extend_بلند_u8x16(v0s);
        ثابت v128_t ylo_l = wasm_i16x8_extend_کم_i8x16(ys_lo);
        ثابت v128_t ylo_h = wasm_i16x8_extend_بلند_i8x16(ys_lo);
        ثابت v128_t v1s_l = wasm_u16x8_extend_کم_u8x16(v1s);
        ثابت v128_t v1s_h = wasm_u16x8_extend_بلند_u8x16(v1s);
        ثابت v128_t yhi_l = wasm_i16x8_extend_کم_i8x16(ys_hi);
        ثابت v128_t yhi_h = wasm_i16x8_extend_بلند_i8x16(ys_hi);

        ثابت v128_t acc = wasm_i32x4_جوڑو(
            wasm_i32x4_جوڑو(
                wasm_i32x4_dot_i16x8(v0s_l, ylo_l),
                wasm_i32x4_dot_i16x8(v0s_h, ylo_h)),
            wasm_i32x4_جوڑو(
                wasm_i32x4_dot_i16x8(v1s_l, yhi_l),
                wasm_i32x4_dot_i16x8(v1s_h, yhi_h)));

        sumv = wasm_f32x4_جوڑو(sumv,
            wasm_f32x4_mul(
                wasm_f32x4_convert_i32x4(acc),
                wasm_f32x4_splat(GGML_CPU_FP16_مقصد_FP32(x0->d) * GGML_CPU_FP16_مقصد_FP32(y0->d))));
    }

    sumf = wasm_f32x4_نکالو_lane(sumv, 0) + wasm_f32x4_نکالو_lane(sumv, 1) +
           wasm_f32x4_نکالو_lane(sumv, 2) + wasm_f32x4_نکالو_lane(sumv, 3) + summs;

    *s = sumf;

#ورنہ
    غیر_مستعمل(nb);
    غیر_مستعمل(x);
    غیر_مستعمل(y);
    غیر_مستعمل(sumf);

    ggml_vec_dot_q4_1_q8_1_generic(
        n, s, bs, vx, bx, vy, by, nrc);
#اگر_ختم
}

خالی ggml_vec_dot_q5_0_q8_0(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc) {
    ثابت صحیح qk = QK8_0;
    ثابت صحیح nb = n / qk;

    صحیح ib = 0;
    اعشاری sumf = 0;

    أكد(n % qk == 0);
    أكد(qk == QK5_0);
    أكد(nrc == 1);
    غیر_مستعمل(nrc);
    غیر_مستعمل(bx);
    غیر_مستعمل(by);
    غیر_مستعمل(bs);

    ثابت بلاک_q5_0 * GGML_RESTRICT x = vx;
    ثابت بلاک_q8_0 * GGML_RESTRICT y = vy;

#اگر defined __wasm_simd128__
    v128_t sumv = wasm_f32x4_splat(0.0f);

    uint32_t qh_;
    uint64_t tmp[4];

    // TODO: check if unrolling this is better
    برائے (; ib < nb; ++ib) {
        ثابت بلاک_q5_0 * GGML_RESTRICT x0 = &x[ib];
        ثابت بلاک_q8_0 * GGML_RESTRICT y0 = &y[ib];

        ثابت v128_t m4b  = wasm_i8x16_splat(0x0F);

        // extract the 5th bit
        حافظہ_نقل(&qh_, x0->qh, حجم(qh_));

        tmp[0] = جدول_b2b_1[(qh_ >>  0) & 0xFF];
        tmp[1] = جدول_b2b_1[(qh_ >>  8) & 0xFF];
        tmp[2] = جدول_b2b_1[(qh_ >> 16) & 0xFF];
        tmp[3] = جدول_b2b_1[(qh_ >> 24)       ];

        ثابت v128_t qhl = wasm_v128_بھرو(tmp + 0);
        ثابت v128_t qhh = wasm_v128_بھرو(tmp + 2);

        ثابت v128_t v0 = wasm_v128_بھرو(x0->qs);

        // 4-bit -> 8-bit
        ثابت v128_t v0l = wasm_v128_and (v0, m4b);
        ثابت v128_t v0h = wasm_u8x16_shr(v0, 4);

        // add high bit and sub 16 (equivalent to sub 0x10 when bit is zero)
        ثابت v128_t v0lf = wasm_i8x16_sub(v0l, qhl);
        ثابت v128_t v0hf = wasm_i8x16_sub(v0h, qhh);

        // load y
        ثابت v128_t v1l = wasm_v128_بھرو(y0->qs);
        ثابت v128_t v1h = wasm_v128_بھرو(y0->qs + 16);

        // int8x16 -> int16x8
        ثابت v128_t v0lfl = wasm_i16x8_extend_کم_i8x16 (v0lf);
        ثابت v128_t v0lfh = wasm_i16x8_extend_بلند_i8x16(v0lf);
        ثابت v128_t v0hfl = wasm_i16x8_extend_کم_i8x16 (v0hf);
        ثابت v128_t v0hfh = wasm_i16x8_extend_بلند_i8x16(v0hf);

        ثابت v128_t v1ll = wasm_i16x8_extend_کم_i8x16 (v1l);
        ثابت v128_t v1lh = wasm_i16x8_extend_بلند_i8x16(v1l);
        ثابت v128_t v1hl = wasm_i16x8_extend_کم_i8x16 (v1h);
        ثابت v128_t v1hh = wasm_i16x8_extend_بلند_i8x16(v1h);

        // dot product
        sumv = wasm_f32x4_جوڑو(sumv, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(
                        wasm_i32x4_جوڑو(
                            wasm_i32x4_جوڑو(wasm_i32x4_dot_i16x8(v0lfl, v1ll),
                                           wasm_i32x4_dot_i16x8(v0lfh, v1lh)),
                            wasm_i32x4_جوڑو(wasm_i32x4_dot_i16x8(v0hfl, v1hl),
                                           wasm_i32x4_dot_i16x8(v0hfh, v1hh)))),
                    wasm_f32x4_splat(GGML_CPU_FP16_مقصد_FP32(x0->d) * GGML_CPU_FP16_مقصد_FP32(y0->d))));
    }

    sumf = wasm_f32x4_نکالو_lane(sumv, 0) + wasm_f32x4_نکالو_lane(sumv, 1) +
           wasm_f32x4_نکالو_lane(sumv, 2) + wasm_f32x4_نکالو_lane(sumv, 3);

    *s = sumf;
#ورنہ
    غیر_مستعمل(nb);
    غیر_مستعمل(ib);
    غیر_مستعمل(sumf);
    غیر_مستعمل(x);
    غیر_مستعمل(y);
    ggml_vec_dot_q5_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#اگر_ختم
}

خالی ggml_vec_dot_q5_1_q8_1(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc) {
    ثابت صحیح qk = QK8_1;
    ثابت صحیح nb = n / qk;

    صحیح ib = 0;
    اعشاری sumf = 0;

    أكد(n % qk == 0);
    أكد(qk == QK5_1);
    أكد(nrc == 1);
    غیر_مستعمل(nrc);
    غیر_مستعمل(bx);
    غیر_مستعمل(by);
    غیر_مستعمل(bs);

    ثابت بلاک_q5_1 * GGML_RESTRICT x = vx;
    ثابت بلاک_q8_1 * GGML_RESTRICT y = vy;

#اگر defined __wasm_simd128__
    v128_t sumv = wasm_f32x4_splat(0.0f);

    اعشاری summs = 0.0f;

    uint32_t qh_;
    uint64_t tmp[4];

    // TODO: check if unrolling this is better
    برائے (; ib < nb; ++ib) {
        ثابت بلاک_q5_1 * GGML_RESTRICT x0 = &x[ib];
        ثابت بلاک_q8_1 * GGML_RESTRICT y0 = &y[ib];

        summs += GGML_CPU_FP16_مقصد_FP32(x0->m) * GGML_CPU_FP16_مقصد_FP32(y0->s);

        ثابت v128_t m4b = wasm_i8x16_splat(0x0F);

        // extract the 5th bit
        حافظہ_نقل(&qh_, x0->qh, حجم(qh_));

        tmp[0] = جدول_b2b_0[(qh_ >>  0) & 0xFF];
        tmp[1] = جدول_b2b_0[(qh_ >>  8) & 0xFF];
        tmp[2] = جدول_b2b_0[(qh_ >> 16) & 0xFF];
        tmp[3] = جدول_b2b_0[(qh_ >> 24)       ];

        ثابت v128_t qhl = wasm_v128_بھرو(tmp + 0);
        ثابت v128_t qhh = wasm_v128_بھرو(tmp + 2);

        ثابت v128_t v0 = wasm_v128_بھرو(x0->qs);

        // 4-bit -> 8-bit
        ثابت v128_t v0l = wasm_v128_and (v0, m4b);
        ثابت v128_t v0h = wasm_u8x16_shr(v0, 4);

        // add high bit
        ثابت v128_t v0lf = wasm_v128_or(v0l, qhl);
        ثابت v128_t v0hf = wasm_v128_or(v0h, qhh);

        // load y
        ثابت v128_t v1l = wasm_v128_بھرو(y0->qs);
        ثابت v128_t v1h = wasm_v128_بھرو(y0->qs + 16);

        // int8x16 -> int16x8
        ثابت v128_t v0lfl = wasm_i16x8_extend_کم_i8x16 (v0lf);
        ثابت v128_t v0lfh = wasm_i16x8_extend_بلند_i8x16(v0lf);
        ثابت v128_t v0hfl = wasm_i16x8_extend_کم_i8x16 (v0hf);
        ثابت v128_t v0hfh = wasm_i16x8_extend_بلند_i8x16(v0hf);

        ثابت v128_t v1ll = wasm_i16x8_extend_کم_i8x16 (v1l);
        ثابت v128_t v1lh = wasm_i16x8_extend_بلند_i8x16(v1l);
        ثابت v128_t v1hl = wasm_i16x8_extend_کم_i8x16 (v1h);
        ثابت v128_t v1hh = wasm_i16x8_extend_بلند_i8x16(v1h);

        // dot product
        sumv = wasm_f32x4_جوڑو(sumv,
                wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_جوڑو(
                            wasm_i32x4_جوڑو(wasm_i32x4_dot_i16x8(v0lfl, v1ll),
                                           wasm_i32x4_dot_i16x8(v0lfh, v1lh)),
                            wasm_i32x4_جوڑو(wasm_i32x4_dot_i16x8(v0hfl, v1hl),
                                           wasm_i32x4_dot_i16x8(v0hfh, v1hh)))),
                    wasm_f32x4_splat(GGML_CPU_FP16_مقصد_FP32(x0->d) * GGML_CPU_FP16_مقصد_FP32(y0->d))));
    }

    sumf = wasm_f32x4_نکالو_lane(sumv, 0) + wasm_f32x4_نکالو_lane(sumv, 1) +
           wasm_f32x4_نکالو_lane(sumv, 2) + wasm_f32x4_نکالو_lane(sumv, 3) + summs;

    *s = sumf;
#ورنہ
    غیر_مستعمل(nb);
    غیر_مستعمل(ib);
    غیر_مستعمل(sumf);
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

#اگر defined __wasm_simd128__
    v128_t sumv = wasm_f32x4_splat(0.0f);

    برائے (; ib < nb; ++ib) {
        ثابت بلاک_q8_0 * GGML_RESTRICT x0 = &x[ib];
        ثابت بلاک_q8_0 * GGML_RESTRICT y0 = &y[ib];

        ثابت v128_t x0_0 = wasm_v128_بھرو(x0->qs);
        ثابت v128_t x0_1 = wasm_v128_بھرو(x0->qs + 16);
        ثابت v128_t y0_0 = wasm_v128_بھرو(y0->qs);
        ثابت v128_t y0_1 = wasm_v128_بھرو(y0->qs + 16);

        // Extend 8-bit to 16-bit
        ثابت v128_t x0_0l = wasm_i16x8_extend_کم_i8x16(x0_0);
        ثابت v128_t x0_0h = wasm_i16x8_extend_بلند_i8x16(x0_0);
        ثابت v128_t x0_1l = wasm_i16x8_extend_کم_i8x16(x0_1);
        ثابت v128_t x0_1h = wasm_i16x8_extend_بلند_i8x16(x0_1);

        ثابت v128_t y0_0l = wasm_i16x8_extend_کم_i8x16(y0_0);
        ثابت v128_t y0_0h = wasm_i16x8_extend_بلند_i8x16(y0_0);
        ثابت v128_t y0_1l = wasm_i16x8_extend_کم_i8x16(y0_1);
        ثابت v128_t y0_1h = wasm_i16x8_extend_بلند_i8x16(y0_1);

        // Compute dot products
        ثابت v128_t dx0_0 = wasm_i32x4_dot_i16x8(x0_0l, y0_0l);
        ثابت v128_t dx0_1 = wasm_i32x4_dot_i16x8(x0_0h, y0_0h);
        ثابت v128_t dx1_0 = wasm_i32x4_dot_i16x8(x0_1l, y0_1l);
        ثابت v128_t dx1_1 = wasm_i32x4_dot_i16x8(x0_1h, y0_1h);

        // Sum all dot products
        ثابت v128_t مجموع_dots = wasm_i32x4_جوڑو(wasm_i32x4_جوڑو(dx0_0, dx0_1), wasm_i32x4_جوڑو(dx1_0, dx1_1));

        // Convert to float and accumulate
        ثابت اعشاری scale = GGML_CPU_FP16_مقصد_FP32(x0->d) * GGML_CPU_FP16_مقصد_FP32(y0->d);
        sumv = wasm_f32x4_جوڑو(sumv, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(مجموع_dots), wasm_f32x4_splat(scale)));
    }

    sumf = wasm_f32x4_نکالو_lane(sumv, 0) + wasm_f32x4_نکالو_lane(sumv, 1) +
           wasm_f32x4_نکالو_lane(sumv, 2) + wasm_f32x4_نکالو_lane(sumv, 3);

    *s = sumf;
#ورنہ
    غیر_مستعمل(nb);
    غیر_مستعمل(x);
    غیر_مستعمل(y);
    غیر_مستعمل(ib);
    غیر_مستعمل(sumf);
    ggml_vec_dot_q8_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
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

#اگر defined __wasm_simd128__
    اعشاری sumf = 0;

    برائے (صحیح i = 0; i < nb; ++i) {
        ثابت uint8_t * q2 = x[i].qs;
        ثابت int8_t * q8 = y[i].qs;
        ثابت uint8_t * sc = x[i].scales;

        // Vectorized summs calculation
        v128_t summs_vec = wasm_i32x4_splat(0);
        {
            v128_t sc_vec = wasm_v128_بھرو(sc);
            v128_t sc_بالائی = wasm_u8x16_shr(sc_vec, 4);

            v128_t sc_کم = wasm_u16x8_extend_کم_u8x16(sc_بالائی);
            v128_t sc_بلند = wasm_u16x8_extend_بلند_u8x16(sc_بالائی);

            v128_t bsums1 = wasm_v128_بھرو(&y[i].bsums[0]);
            v128_t bsums2 = wasm_v128_بھرو(&y[i].bsums[8]);

            summs_vec = wasm_i32x4_جوڑو(
                wasm_i32x4_جوڑو(wasm_i32x4_dot_i16x8(sc_کم, bsums1),
                               wasm_i32x4_dot_i16x8(sc_بلند, bsums2)),
                summs_vec
            );

            summs_vec = wasm_i32x4_جوڑو(summs_vec, wasm_i32x4_shuffle(summs_vec, summs_vec, 2, 3, 0, 1));
            summs_vec = wasm_i32x4_جوڑو(summs_vec, wasm_i32x4_shuffle(summs_vec, summs_vec, 1, 0, 3, 2));
        }
        int32_t summs = wasm_i32x4_نکالو_lane(summs_vec, 0);

        // Vectorized isum calculation
        int32_t isum = 0;
        ثابت uint8_t * sc_ptr = sc;
        ثابت صحیح k_iters = QK_K/128;

        برائے (صحیح k = 0; k < k_iters; ++k) {
            v128_t isum_vec = wasm_i32x4_splat(0);
            صحیح سرکاؤ = 0;

            برائے (صحیح j = 0; j < 4; ++j) {
                ثابت صحیح d0 = (sc_ptr[0] & 0xF);
                ثابت صحیح d1 = (sc_ptr[1] & 0xF);
                sc_ptr += 2;

                // Process first 16 elements
                v128_t q2_0 = wasm_v128_بھرو(q2);
                v128_t q8_0 = wasm_v128_بھرو(q8);
                v128_t q2_سرکاؤ_0 = wasm_u8x16_shr(q2_0, سرکاؤ);
                v128_t q2_بٹ_0 = wasm_v128_and(q2_سرکاؤ_0, wasm_i8x16_splat(0x03));

                // Process next 16 elements
                v128_t q2_1 = wasm_v128_بھرو(q2 + 16);
                v128_t q8_1 = wasm_v128_بھرو(q8 + 16);
                v128_t q2_سرکاؤ_1 = wasm_u8x16_shr(q2_1, سرکاؤ);
                v128_t q2_بٹ_1 = wasm_v128_and(q2_سرکاؤ_1, wasm_i8x16_splat(0x03));

                // Calculate dot products
                v128_t p0 = wasm_i32x4_dot_i16x8(
                    wasm_i16x8_extend_کم_i8x16(q8_0),
                    wasm_i16x8_extend_کم_i8x16(q2_بٹ_0)
                );
                v128_t p1 = wasm_i32x4_dot_i16x8(
                    wasm_i16x8_extend_بلند_i8x16(q8_0),
                    wasm_i16x8_extend_بلند_i8x16(q2_بٹ_0)
                );
                v128_t p2 = wasm_i32x4_dot_i16x8(
                    wasm_i16x8_extend_کم_i8x16(q8_1),
                    wasm_i16x8_extend_کم_i8x16(q2_بٹ_1)
                );
                v128_t p3 = wasm_i32x4_dot_i16x8(
                    wasm_i16x8_extend_بلند_i8x16(q8_1),
                    wasm_i16x8_extend_بلند_i8x16(q2_بٹ_1)
                );

                // Accumulate scaled results
                v128_t scaled = wasm_i32x4_جوڑو(
                    wasm_i32x4_mul(wasm_i32x4_جوڑو(p0, p1), wasm_i32x4_splat(d0)),
                    wasm_i32x4_mul(wasm_i32x4_جوڑو(p2, p3), wasm_i32x4_splat(d1))
                );

                isum_vec = wasm_i32x4_جوڑو(isum_vec, scaled);
                q8 += 32;
                سرکاؤ += 2;
            }
            q2 += 32;

            // Horizontal sum of isum_vec
            isum_vec = wasm_i32x4_جوڑو(isum_vec, wasm_i32x4_shuffle(isum_vec, isum_vec, 2, 3, 0, 1));
            isum_vec = wasm_i32x4_جوڑو(isum_vec, wasm_i32x4_shuffle(isum_vec, isum_vec, 1, 0, 3, 2));
            isum += wasm_i32x4_نکالو_lane(isum_vec, 0);
        }

        ثابت اعشاری dall = GGML_CPU_FP16_مقصد_FP32(x[i].d) * y[i].d;
        ثابت اعشاری dmin = GGML_CPU_FP16_مقصد_FP32(x[i].dmin) * y[i].d;
        sumf += dall * isum - dmin * summs;
    }

    *s = sumf;

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

#اگر defined __wasm_simd128__
    int8_t  aux8[QK_K];
    اعشاری   sums[8] = {0};
    uint32_t auxs[4];

    اعشاری sumf = 0;
    برائے (صحیح i = 0; i < nb; ++i) {
        ثابت uint8_t * GGML_RESTRICT q3 = x[i].qs;
        ثابت uint8_t * GGML_RESTRICT hm = x[i].hmask;
        ثابت int8_t  * GGML_RESTRICT q8 = y[i].qs;

        // Process blocks with SIMD
        int8_t * a = aux8;
        uint8_t m = 1;
        برائے (صحیح j = 0; j < QK_K; j += 128) {
            برائے (صحیح سرکاؤ = 0; سرکاؤ <= 6; سرکاؤ += 2) {
                v128_t v_m = wasm_i8x16_splat(m);
                برائے (صحیح l = 0; l < 32; l += 16) {
                    v128_t v_q3 = wasm_v128_بھرو(q3 + l);
                    v128_t v_سرکاؤ = wasm_i8x16_shr(v_q3, سرکاؤ);
                    v128_t v_low2 = wasm_v128_and(v_سرکاؤ, wasm_i8x16_splat(0x03));

                    v128_t v_hm = wasm_v128_بھرو(hm + l);
                    v128_t v_نقاب = wasm_v128_and(v_hm, v_m);
                    v_نقاب = wasm_i8x16_ne(v_نقاب, wasm_i8x16_splat(0));

                    v_low2 = wasm_i8x16_sub(v_low2, wasm_v128_and(wasm_i8x16_splat(4), wasm_v128_not(v_نقاب)));
                    wasm_v128_store(a + l, v_low2);
                }
                a += 32;
                m <<= 1;
            }
            q3 += 32;
        }

        // Extract scales
        حافظہ_نقل(auxs, x[i].scales, 12);
        uint32_t tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        ثابت int8_t * scales = (ثابت int8_t *)auxs;

        // SIMD dot product with register accumulators
        v128_t v_acc0 = wasm_i32x4_splat(0);
        v128_t v_acc1 = wasm_i32x4_splat(0);
        a = aux8;
        برائے (صحیح j = 0; j < QK_K/16; ++j) {
            ثابت v128_t v_scale = wasm_i16x8_splat(scales[j] - 32);

            // Process 16 elements per iteration
            برائے (صحیح k = 0; k < 2; ++k) {
                ثابت v128_t v_q8 = wasm_i16x8_load8x8(q8);
                ثابت v128_t v_a = wasm_i16x8_load8x8(a);

                v128_t v_prod = wasm_i16x8_mul(v_q8, v_a);
                v_prod = wasm_i16x8_mul(v_prod, v_scale);

                v_acc0 = wasm_i32x4_جوڑو(v_acc0, wasm_i32x4_extend_کم_i16x8(v_prod));
                v_acc1 = wasm_i32x4_جوڑو(v_acc1, wasm_i32x4_extend_بلند_i16x8(v_prod));

                q8 += 8;
                a += 8;
            }
        }

        // Accumulate results
        ثابت اعشاری d = GGML_CPU_FP16_مقصد_FP32(x[i].d) * y[i].d;
        ثابت v128_t v_d = wasm_f32x4_splat(d);
        v128_t v_مجموع = wasm_f32x4_جوڑو(
            wasm_f32x4_mul(wasm_f32x4_convert_i32x4(v_acc0), v_d),
            wasm_f32x4_mul(wasm_f32x4_convert_i32x4(v_acc1), v_d)
        );

        // Accumulate into sums vector
        wasm_v128_store(sums, wasm_f32x4_جوڑو(wasm_v128_بھرو(sums), v_مجموع));
    }

    // Horizontal sum
    v128_t v_مجموع = wasm_f32x4_جوڑو(wasm_v128_بھرو(sums), wasm_v128_بھرو(sums + 4));
    sumf = wasm_f32x4_نکالو_lane(v_مجموع, 0) +
           wasm_f32x4_نکالو_lane(v_مجموع, 1) +
           wasm_f32x4_نکالو_lane(v_مجموع, 2) +
           wasm_f32x4_نکالو_lane(v_مجموع, 3);

    *s = sumf;

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

#اگر defined __wasm_simd128__
    ثابت uint8_t * scales = (ثابت uint8_t*)&utmp[0];
    اعشاری sumf = 0;

    برائے (صحیح i = 0; i < nb; ++i) {
        ثابت اعشاری d = y[i].d * GGML_CPU_FP16_مقصد_FP32(x[i].d);
        ثابت اعشاری dmin = y[i].d * GGML_CPU_FP16_مقصد_FP32(x[i].dmin); // Corrected sign

        ثابت uint8_t * GGML_RESTRICT q4 = x[i].qs;
        ثابت int8_t  * GGML_RESTRICT q8 = y[i].qs;

        // Process scales and mins
        حافظہ_نقل(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        ثابت uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        // Sum mins * q8sums
        int32_t sumi = 0;
        ثابت int16_t * GGML_RESTRICT q8sums = y[i].bsums;
        ثابت uint8_t * m = (ثابت uint8_t *)&utmp[2];
        برائے (صحیح j = 0; j < 16; j += 2) {
            sumi += (q8sums[j] + q8sums[j+1]) * m[j/2];
        }
        sumf -= dmin * sumi;

        int32_t sumi1 = 0;
        int32_t sumi2 = 0;

        برائے (صحیح j = 0; j < QK_K/64; ++j) {
            // Load 64 4-bit weights (32 bytes)
            ثابت v128_t q4x0 = wasm_v128_بھرو(q4);
            ثابت v128_t q4x1 = wasm_v128_بھرو(q4 + 16);
            q4 += 32;

            // Split into low/high nibbles
            ثابت v128_t q4l0 = wasm_v128_and(q4x0, wasm_i8x16_splat(0x0F));
            ثابت v128_t q4h0 = wasm_u8x16_shr(q4x0, 4);
            ثابت v128_t q4l1 = wasm_v128_and(q4x1, wasm_i8x16_splat(0x0F));
            ثابت v128_t q4h1 = wasm_u8x16_shr(q4x1, 4);

            // Load 64 8-bit values (64 bytes)
            ثابت v128_t q8x0 = wasm_v128_بھرو(q8);
            ثابت v128_t q8x1 = wasm_v128_بھرو(q8 + 16);
            ثابت v128_t q8x2 = wasm_v128_بھرو(q8 + 32);
            ثابت v128_t q8x3 = wasm_v128_بھرو(q8 + 48);
            q8 += 64;

            // Low nibble products
            v128_t vacc1 = wasm_i32x4_dot_i16x8(
                wasm_i16x8_extend_کم_i8x16(q4l0),
                wasm_i16x8_extend_کم_i8x16(q8x0)
            );
            vacc1 = wasm_i32x4_جوڑو(vacc1, wasm_i32x4_dot_i16x8(
                wasm_i16x8_extend_بلند_i8x16(q4l0),
                wasm_i16x8_extend_بلند_i8x16(q8x0)
            ));
            vacc1 = wasm_i32x4_جوڑو(vacc1, wasm_i32x4_dot_i16x8(
                wasm_i16x8_extend_کم_i8x16(q4l1),
                wasm_i16x8_extend_کم_i8x16(q8x1)
            ));
            vacc1 = wasm_i32x4_جوڑو(vacc1, wasm_i32x4_dot_i16x8(
                wasm_i16x8_extend_بلند_i8x16(q4l1),
                wasm_i16x8_extend_بلند_i8x16(q8x1)
            ));

            // High nibble products
            v128_t vacc2 = wasm_i32x4_dot_i16x8(
                wasm_i16x8_extend_کم_i8x16(q4h0),
                wasm_i16x8_extend_کم_i8x16(q8x2)
            );
            vacc2 = wasm_i32x4_جوڑو(vacc2, wasm_i32x4_dot_i16x8(
                wasm_i16x8_extend_بلند_i8x16(q4h0),
                wasm_i16x8_extend_بلند_i8x16(q8x2)
            ));
            vacc2 = wasm_i32x4_جوڑو(vacc2, wasm_i32x4_dot_i16x8(
                wasm_i16x8_extend_کم_i8x16(q4h1),
                wasm_i16x8_extend_کم_i8x16(q8x3)
            ));
            vacc2 = wasm_i32x4_جوڑو(vacc2, wasm_i32x4_dot_i16x8(
                wasm_i16x8_extend_بلند_i8x16(q4h1),
                wasm_i16x8_extend_بلند_i8x16(q8x3)
            ));

            // Accumulate scaled results
            int32_t vacc1_مجموع = wasm_i32x4_نکالو_lane(vacc1, 0) + wasm_i32x4_نکالو_lane(vacc1, 1) +
                                wasm_i32x4_نکالو_lane(vacc1, 2) + wasm_i32x4_نکالو_lane(vacc1, 3);
            sumi1 += vacc1_مجموع * scales[2*j];

            int32_t vacc2_مجموع = wasm_i32x4_نکالو_lane(vacc2, 0) + wasm_i32x4_نکالو_lane(vacc2, 1) +
                                wasm_i32x4_نکالو_lane(vacc2, 2) + wasm_i32x4_نکالو_lane(vacc2, 3);
            sumi2 += vacc2_مجموع * scales[2*j+1];
        }

        sumf += d * (sumi1 + sumi2);
    }

    *s = sumf;

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

#اگر defined __wasm_simd128__
    //const uint8_t * scales = (const uint8_t*)&utmp[0];
    اعشاری sumf = 0;

    برائے (صحیح i = 0; i < nb; ++i) {
        ثابت اعشاری d = y[i].d * GGML_CPU_FP16_مقصد_FP32(x[i].d);
        ثابت اعشاری dmin = y[i].d * GGML_CPU_FP16_مقصد_FP32(x[i].dmin); // Fixed sign

        ثابت uint8_t * GGML_RESTRICT q5 = x[i].qs;
        ثابت uint8_t * GGML_RESTRICT qh = x[i].qh;
        ثابت int8_t  * GGML_RESTRICT q8 = y[i].qs;

        // Process scales and mins
        حافظہ_نقل(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        ثابت uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        // Sum mins * q8sums
        int32_t sumi_mins = 0;
        ثابت int16_t * GGML_RESTRICT q8sums = y[i].bsums;
        ثابت uint8_t * m = (ثابت uint8_t *)&utmp[2];
        برائے (صحیح j = 0; j < 16; j += 2) {
            sumi_mins += (q8sums[j] + q8sums[j+1]) * m[j/2];
        }
        sumf -= dmin * sumi_mins; // Correct subtraction

        v128_t qh0 = wasm_v128_بھرو(qh);
        v128_t qh1 = wasm_v128_بھرو(qh + 16);
        ثابت uint8_t * sc = (ثابت uint8_t *)utmp;

        int32_t sumi = 0;

        برائے (صحیح j = 0; j < QK_K/64; ++j) {
            ثابت صحیح سرکاؤ = j * 2;
            v128_t qh_shift0 = wasm_u8x16_shr(qh0, سرکاؤ);
            v128_t qh_shift1 = wasm_u8x16_shr(qh1, سرکاؤ);

            v128_t qh_low0 = wasm_i8x16_shl(wasm_v128_and(qh_shift0, wasm_i8x16_splat(0x01)), 4);
            v128_t qh_high0 = wasm_i8x16_shl(wasm_v128_and(qh_shift0, wasm_i8x16_splat(0x02)), 3);
            v128_t qh_low1 = wasm_i8x16_shl(wasm_v128_and(qh_shift1, wasm_i8x16_splat(0x01)), 4);
            v128_t qh_high1 = wasm_i8x16_shl(wasm_v128_and(qh_shift1, wasm_i8x16_splat(0x02)), 3);

            v128_t q5_0 = wasm_v128_بھرو(q5);
            v128_t q5_1 = wasm_v128_بھرو(q5 + 16);
            q5 += 32;

            v128_t q5l_0 = wasm_v128_or(wasm_v128_and(q5_0, wasm_i8x16_splat(0x0F)), qh_low0);
            v128_t q5h_0 = wasm_v128_or(wasm_u8x16_shr(q5_0, 4), qh_high0);
            v128_t q5l_1 = wasm_v128_or(wasm_v128_and(q5_1, wasm_i8x16_splat(0x0F)), qh_low1);
            v128_t q5h_1 = wasm_v128_or(wasm_u8x16_shr(q5_1, 4), qh_high1);

            v128_t q8_0 = wasm_v128_بھرو(q8);
            v128_t q8_1 = wasm_v128_بھرو(q8 + 16);
            v128_t q8_2 = wasm_v128_بھرو(q8 + 32);
            v128_t q8_3 = wasm_v128_بھرو(q8 + 48);
            q8 += 64;

            // Process low quants
            v128_t pl0 = wasm_i32x4_dot_i16x8(
                wasm_i16x8_extend_کم_i8x16(q5l_0),
                wasm_i16x8_extend_کم_i8x16(q8_0)
            );
            pl0 = wasm_i32x4_جوڑو(pl0, wasm_i32x4_dot_i16x8(
                wasm_i16x8_extend_بلند_i8x16(q5l_0),
                wasm_i16x8_extend_بلند_i8x16(q8_0)
            ));
            v128_t pl1 = wasm_i32x4_dot_i16x8(
                wasm_i16x8_extend_کم_i8x16(q5l_1),
                wasm_i16x8_extend_کم_i8x16(q8_1)
            );
            pl1 = wasm_i32x4_جوڑو(pl1, wasm_i32x4_dot_i16x8(
                wasm_i16x8_extend_بلند_i8x16(q5l_1),
                wasm_i16x8_extend_بلند_i8x16(q8_1)
            ));
            v128_t مجموع_کم = wasm_i32x4_جوڑو(pl0, pl1);

            // Process high quants
            v128_t ph0 = wasm_i32x4_dot_i16x8(
                wasm_i16x8_extend_کم_i8x16(q5h_0),
                wasm_i16x8_extend_کم_i8x16(q8_2)
            );
            ph0 = wasm_i32x4_جوڑو(ph0, wasm_i32x4_dot_i16x8(
                wasm_i16x8_extend_بلند_i8x16(q5h_0),
                wasm_i16x8_extend_بلند_i8x16(q8_2)
            ));
            v128_t ph1 = wasm_i32x4_dot_i16x8(
                wasm_i16x8_extend_کم_i8x16(q5h_1),
                wasm_i16x8_extend_کم_i8x16(q8_3)
            );
            ph1 = wasm_i32x4_جوڑو(ph1, wasm_i32x4_dot_i16x8(
                wasm_i16x8_extend_بلند_i8x16(q5h_1),
                wasm_i16x8_extend_بلند_i8x16(q8_3)
            ));
            v128_t مجموع_بلند = wasm_i32x4_جوڑو(ph0, ph1);

            // Accumulate with scale factors
            int32_t sl = wasm_i32x4_نکالو_lane(مجموع_کم, 0) + wasm_i32x4_نکالو_lane(مجموع_کم, 1) +
                        wasm_i32x4_نکالو_lane(مجموع_کم, 2) + wasm_i32x4_نکالو_lane(مجموع_کم, 3);
            int32_t sh = wasm_i32x4_نکالو_lane(مجموع_بلند, 0) + wasm_i32x4_نکالو_lane(مجموع_بلند, 1) +
                        wasm_i32x4_نکالو_lane(مجموع_بلند, 2) + wasm_i32x4_نکالو_lane(مجموع_بلند, 3);

            sumi += sl * sc[2*j] + sh * sc[2*j+1];
        }

        sumf += d * sumi;
    }

    *s = sumf;

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

#اگر defined __wasm_simd128__
    int8_t aux8[QK_K] __attribute__((aligned(16)));
    int32_t aux32[8] __attribute__((aligned(16))) = {0};
    اعشاری sums[8] __attribute__((aligned(16))) = {0};

    برائے (صحیح i = 0; i < nb; ++i) {
        // Unpack 6-bit quantized data into aux8 (unchanged)
        ثابت uint8_t * GGML_RESTRICT q4 = x[i].ql;
        ثابت uint8_t * GGML_RESTRICT qh = x[i].qh;
        int8_t * a = aux8;
        برائے (صحیح j = 0; j < QK_K; j += 128) {
            برائے (صحیح l = 0; l < 32; ++l) {
                a[l +  0] = (int8_t)((q4[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                a[l + 32] = (int8_t)((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                a[l + 64] = (int8_t)((q4[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                a[l + 96] = (int8_t)((q4[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            }
            a += 128;
            q4 += 64;
            qh += 32;
        }

        ثابت int8_t * GGML_RESTRICT a_ptr = aux8;
        ثابت int8_t * GGML_RESTRICT q8 = y[i].qs;
        v128_t acc0 = wasm_i32x4_splat(0);
        v128_t acc1 = wasm_i32x4_splat(0);

        برائے (صحیح j = 0; j < QK_K/16; ++j) {
            ثابت صحیح scale = x[i].scales[j];
            ثابت v128_t vscale = wasm_i32x4_splat(scale);

            // Load 16 elements from a and q8
            ثابت v128_t a_vec = wasm_v128_بھرو(a_ptr);
            ثابت v128_t q8_vec = wasm_v128_بھرو(q8);

            // Process low 8 elements
            v128_t a_کم = wasm_i16x8_extend_کم_i8x16(a_vec);
            v128_t q8_کم = wasm_i16x8_extend_کم_i8x16(q8_vec);
            v128_t prod_کم = wasm_i16x8_mul(a_کم, q8_کم);
            v128_t prod_lo_lo = wasm_i32x4_extend_کم_i16x8(prod_کم);
            v128_t prod_lo_hi = wasm_i32x4_extend_بلند_i16x8(prod_کم);

            // Process high 8 elements
            v128_t a_بلند = wasm_i16x8_extend_بلند_i8x16(a_vec);
            v128_t q8_بلند = wasm_i16x8_extend_بلند_i8x16(q8_vec);
            v128_t prod_بلند = wasm_i16x8_mul(a_بلند, q8_بلند);
            v128_t prod_hi_lo = wasm_i32x4_extend_کم_i16x8(prod_بلند);
            v128_t prod_hi_hi = wasm_i32x4_extend_بلند_i16x8(prod_بلند);

            // Scale and accumulate
            prod_lo_lo = wasm_i32x4_mul(prod_lo_lo, vscale);
            prod_lo_hi = wasm_i32x4_mul(prod_lo_hi, vscale);
            prod_hi_lo = wasm_i32x4_mul(prod_hi_lo, vscale);
            prod_hi_hi = wasm_i32x4_mul(prod_hi_hi, vscale);

            acc0 = wasm_i32x4_جوڑو(acc0, wasm_i32x4_جوڑو(prod_lo_lo, prod_hi_lo));
            acc1 = wasm_i32x4_جوڑو(acc1, wasm_i32x4_جوڑو(prod_lo_hi, prod_hi_hi));

            a_ptr += 16;
            q8 += 16;
        }

        // Store accumulated results
        wasm_v128_store(&aux32[0], acc0);
        wasm_v128_store(&aux32[4], acc1);

        ثابت اعشاری d = GGML_CPU_FP16_مقصد_FP32(x[i].d) * y[i].d;
        برائے (صحیح l = 0; l < 8; ++l) {
            sums[l] += d * aux32[l];
        }
    }

    // Sum final results
    اعشاری sumf = 0;
    برائے (صحیح l = 0; l < 8; ++l) {
        sumf += sums[l];
    }
    *s = sumf;

#ورنہ
    غیر_مستعمل(x);
    غیر_مستعمل(y);
    غیر_مستعمل(nb);
    ggml_vec_dot_q6_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#اگر_ختم
}
