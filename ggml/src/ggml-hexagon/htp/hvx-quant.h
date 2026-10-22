#اگر_تعریف_نشده HVX_QUANT_H
#تعریف_کن HVX_QUANT_H

#ضمن <math.h>
#ضمن <stdint.h>
#ضمن <string.h>

#ضمن "htp-tensor.h"
#ضمن "hvx-arith.h"
#ضمن "hvx-base.h"
#ضمن "hvx-reduce.h"
#ضمن "hvx-repl.h"
#ضمن "hvx-utils.h"

#اگر_تعریف_نشده GGML_COMMON_DECL_C
#تعریف_کن GGML_COMMON_DECL_C
#پایان_اگر
#ضمن "ggml-common.h"
#ضمن "ggml-impl.h"

ساكن مضمن فراغ hvx_quantize_ردیف_q8_0_f32(فراغ * محدود dst_ptr, ثابت عائم * محدود src_ptr, صحيح n) {
    ثابت صحيح nb = n / QK8_0;
    بلوک_q8_0 * dst = (بلوک_q8_0 *) dst_ptr;
    HVX_وکتور zero = Q6_V_vzero();

    صحيح i = 0;
    لكل (; i + 3 < nb; i += 4) {
        HVX_وکتور * vx = (HVX_وکتور *) (src_ptr + i * QK8_0);

        HVX_وکتور vmax0_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[0]));
        HVX_وکتور vmax1_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[1]));
        HVX_وکتور vmax2_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[2]));
        HVX_وکتور vmax3_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[3]));

        HVX_وکتور vx0_qf = Q6_Vqf32_vsub_VsfVsf(vx[0], zero);
        HVX_وکتور vx1_qf = Q6_Vqf32_vsub_VsfVsf(vx[1], zero);
        HVX_وکتور vx2_qf = Q6_Vqf32_vsub_VsfVsf(vx[2], zero);
        HVX_وکتور vx3_qf = Q6_Vqf32_vsub_VsfVsf(vx[3], zero);

        HVX_وکتور vmax0_qf = Q6_Vqf32_vsub_VsfVsf(vmax0_sf, zero);
        HVX_وکتور vmax1_qf = Q6_Vqf32_vsub_VsfVsf(vmax1_sf, zero);
        HVX_وکتور vmax2_qf = Q6_Vqf32_vsub_VsfVsf(vmax2_sf, zero);
        HVX_وکتور vmax3_qf = Q6_Vqf32_vsub_VsfVsf(vmax3_sf, zero);

        HVX_وکتور vmax01_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vmax1_qf, vmax0_qf)));
        HVX_وکتور vmax23_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vmax3_qf, vmax2_qf)));

        HVX_وکتور vx01_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx1_qf, vx0_qf)));
        HVX_وکتور vx23_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx3_qf, vx2_qf)));

        HVX_وکتور vd01_qf16 = Q6_Vqf16_vmpy_VhfVhf(vmax01_hf, Q6_Vh_vsplat_R(0x2008));  // 1.0 / 127.0
        HVX_وکتور vd23_qf16 = Q6_Vqf16_vmpy_VhfVhf(vmax23_hf, Q6_Vh_vsplat_R(0x2008));  // 1.0 / 127.0
        HVX_وکتور vd01_hf   = Q6_Vhf_equals_Vqf16(vd01_qf16);
        HVX_وکتور vd23_hf   = Q6_Vhf_equals_Vqf16(vd23_qf16);

        HVX_وکتور vd01_inv_hf = hvx_vec_inverse_f16(vd01_hf);
        HVX_وکتور vd23_inv_hf = hvx_vec_inverse_f16(vd23_hf);
        vx01_hf              = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx01_hf, vd01_inv_hf));
        vx23_hf              = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx23_hf, vd23_inv_hf));

        HVX_وکتور vx01_i16 = hvx_vec_i16_from_hf_rnd_sat(vx01_hf);
        HVX_وکتور vx23_i16 = hvx_vec_i16_from_hf_rnd_sat(vx23_hf);
        HVX_وکتور vx_i8    = Q6_Vb_vpack_VhVh_sat(vx23_i16, vx01_i16);

        hvx_vec_store_u(&dst[i + 0].d, 2, vd01_hf);
        hvx_vec_store_u(dst[i + 0].qs, 32, vx_i8);

        hvx_vec_store_u(&dst[i + 1].d, 2, Q6_V_vror_VR(vd01_hf, 64));
        hvx_vec_store_u(dst[i + 1].qs, 32, Q6_V_vror_VR(vx_i8, 32));

        hvx_vec_store_u(&dst[i + 2].d, 2, vd23_hf);
        hvx_vec_store_u(dst[i + 2].qs, 32, Q6_V_vror_VR(vx_i8, 64));

        hvx_vec_store_u(&dst[i + 3].d, 2, Q6_V_vror_VR(vd23_hf, 64));
        hvx_vec_store_u(dst[i + 3].qs, 32, Q6_V_vror_VR(vx_i8, 96));
    }

    لكل (; i < nb; i++) {
        ثابت عائم * بلوک_src = src_ptr + i * QK8_0;
        HVX_وکتور vx = *(ثابت HVX_UVector *) بلوک_src;
        HVX_وکتور v_abs = hvx_vec_abs_f32(vx);
        HVX_وکتور v_max = hvx_vec_reduce_max_f32(v_abs);
        عائم amax = hvx_vec_get_f32(v_max);

        ثابت عائم d = amax / 127.0f;
        ثابت عائم id = d ? (1.0f / d) : 0.0f;
        dst[i].d = GGML_FP32_مقصد_FP16(d);

        HVX_وکتور vid = hvx_vec_splat_f32(id);
        HVX_وکتور v_scaled = hvx_vec_mul_f32_f32(vx, vid);
        HVX_وکتور v_scaled_qf = Q6_Vqf32_vsub_VsfVsf(v_scaled, zero);
        HVX_وکتور v_scaled_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(zero, v_scaled_qf)));
        HVX_وکتور v_i16 = hvx_vec_i16_from_hf_rnd_sat(v_scaled_hf);
        HVX_وکتور v_i8  = Q6_Vb_vpack_VhVh_sat(zero, v_i16);

        hvx_vec_store_u(dst[i].qs, 32, v_i8);
    }
}

ساكن مضمن فراغ hvx_dequantize_ردیف_flat_q8_0_f32(عائم * محدود dst_ptr, ثابت فراغ * محدود src_ptr, uint32_t n) {
    ثابت صحيح nb = n / QK8_0;
    ثابت بلوک_q8_0 * src = (ثابت بلوک_q8_0 *) src_ptr;

    لكل (صحيح i = 0; i < nb; i++) {
        HVX_وکتور vd_f16     = Q6_Vh_vsplat_R(*(ثابت int16_t *) &src[i].d);
        HVX_وکتور_Pair vp_f32 = hvx_vec_f16_مقصد_f32(vd_f16);
        HVX_وکتور vd         = Q6_V_lo_W(vp_f32);

        HVX_وکتور vq_i8 = *(ثابت HVX_UVector *) src[i].qs;

        HVX_وکتور_Pair p16   = Q6_Wh_vunpack_Vb(vq_i8);
        HVX_وکتور     v_i16 = Q6_V_lo_W(p16);
        HVX_وکتور_Pair p32   = Q6_Ww_vunpack_Vh(v_i16);
        HVX_وکتور     v_i32 = Q6_V_lo_W(p32);

        HVX_وکتور v_f32 = Q6_Vsf_equals_Vw(v_i32);
        HVX_وکتور res   = hvx_vec_mul_f32_f32(v_f32, vd);

        عائم * بلوک_dst = dst_ptr + i * QK8_0;
        hvx_vmem(بلوک_dst) = res;
    }
}

ساكن مضمن فراغ hvx_dequantize_ردیف_flat_q8_0_f16(__fp16 * محدود dst_ptr, ثابت فراغ * محدود src_ptr, uint32_t n) {
    ثابت صحيح nb = n / QK8_0;
    ثابت بلوک_q8_0 * src = (ثابت بلوک_q8_0 *) src_ptr;

    لكل (صحيح i = nb - 1; i >= 0; i--) {
        HVX_وکتور vd_f16     = Q6_Vh_vsplat_R(*(ثابت int16_t *) &src[i].d);
        HVX_وکتور_Pair vp_f32 = hvx_vec_f16_مقصد_f32(vd_f16);
        HVX_وکتور vd         = Q6_V_lo_W(vp_f32);

        HVX_وکتور vq_i8 = *(ثابت HVX_UVector *) src[i].qs;

        HVX_وکتور_Pair p16   = Q6_Wh_vunpack_Vb(vq_i8);
        HVX_وکتور     v_i16 = Q6_V_lo_W(p16);
        HVX_وکتور_Pair p32   = Q6_Ww_vunpack_Vh(v_i16);
        HVX_وکتور     v_i32 = Q6_V_lo_W(p32);

        HVX_وکتور v_f32 = Q6_Vsf_equals_Vw(v_i32);
        HVX_وکتور res_f32 = hvx_vec_mul_f32_f32(v_f32, vd);

        HVX_وکتور res_f16 = hvx_vec_f32_مقصد_f16(res_f32, Q6_V_vzero());

        __fp16 * بلوک_dst = dst_ptr + i * QK8_0;
        hvx_vec_store_u(بلوک_dst, QK8_0 * حجم(__fp16), res_f16);
    }
}

ساكن مضمن فراغ hvx_dequantize_ردیف_flat_f16_f32(عائم * محدود dst_ptr, ثابت فراغ * محدود src_ptr, uint32_t n) {
    ثابت uint32_t nvec = n / VLEN_FP32;
    ثابت uint32_t nloe = n % VLEN_FP32;
    ثابت _Float16 * src = (ثابت _Float16 *) src_ptr;

    لكل (uint32_t i = 0; i < nvec; i++) {
        HVX_وکتور v_f16 = *(ثابت HVX_UVector *) (src + i * VLEN_FP32);
        HVX_وکتور_Pair vp_f32 = hvx_vec_f16_مقصد_f32(v_f16);
        HVX_وکتور res = Q6_V_lo_W(vp_f32);

        hvx_vmem(dst_ptr + i * VLEN_FP32) = res;
    }

    إذا (nloe) {
        HVX_وکتور v_f16 = *(ثابت HVX_UVector *) (src + nvec * VLEN_FP32);
        HVX_وکتور_Pair vp_f32 = hvx_vec_f16_مقصد_f32(v_f16);
        HVX_وکتور res = Q6_V_lo_W(vp_f32);

        hvx_vec_store_a(dst_ptr + nvec * VLEN_FP32, nloe * حجم(عائم), res);
    }
}

ساكن مضمن فراغ hvx_dequantize_ردیف_flat_bf16_f32(عائم * محدود dst_ptr, ثابت فراغ * محدود src_ptr, uint32_t n) {
    ثابت uint32_t nvec = n / VLEN_FP32;
    ثابت uint32_t nloe = n % VLEN_FP32;
    ثابت uint16_t * src = (ثابت uint16_t *) src_ptr;

    لكل (uint32_t i = 0; i < nvec; i++) {
        HVX_وکتور v_bf16 = *(ثابت HVX_UVector *) (src + i * VLEN_FP32);
        HVX_وکتور_Pair p = Q6_Ww_vunpack_Vh(v_bf16);
        HVX_وکتور res = Q6_Vw_vasl_VwR(Q6_V_lo_W(p), 16);

        hvx_vmem(dst_ptr + i * VLEN_FP32) = res;
    }

    إذا (nloe) {
        HVX_وکتور v_bf16 = *(ثابت HVX_UVector *) (src + nvec * VLEN_FP32);
        HVX_وکتور_Pair p = Q6_Ww_vunpack_Vh(v_bf16);
        HVX_وکتور res = Q6_Vw_vasl_VwR(Q6_V_lo_W(p), 16);

        hvx_vec_store_a(dst_ptr + nvec * VLEN_FP32, nloe * حجم(عائم), res);
    }
}

ساكن مضمن فراغ hvx_dequantize_ردیف_flat_q1_0_f32(عائم * محدود dst_ptr, ثابت فراغ * محدود src_ptr, uint32_t n) {
    ساكن ثابت uint32_t __attribute__((aligned(128))) بیت_ماسک[32] = {
        1u <<  0, 1u <<  1, 1u <<  2, 1u <<  3, 1u <<  4, 1u <<  5, 1u <<  6, 1u <<  7,
        1u <<  8, 1u <<  9, 1u << 10, 1u << 11, 1u << 12, 1u << 13, 1u << 14, 1u << 15,
        1u << 16, 1u << 17, 1u << 18, 1u << 19, 1u << 20, 1u << 21, 1u << 22, 1u << 23,
        1u << 24, 1u << 25, 1u << 26, 1u << 27, 1u << 28, 1u << 29, 1u << 30, 1u << 31,
    };
    ثابت صحيح nb = n / QK1_0;
    ثابت بلوک_q1_0 * src = (ثابت بلوک_q1_0 *) src_ptr;
    ثابت HVX_وکتور v_بیت_ماسک = *(ثابت HVX_وکتور *) بیت_ماسک;
    ثابت HVX_وکتور v_zero     = Q6_V_vzero();
    ثابت HVX_وکتور v_sign_بیت = Q6_V_vsplat_R(0x80000000);

    لكل (صحيح i = 0; i < nb; i++) {
        HVX_وکتور vd_f16 = Q6_Vh_vsplat_R(*(ثابت int16_t *) &src[i].d);
        HVX_وکتور vd     = Q6_V_lo_W(hvx_vec_f16_مقصد_f32(vd_f16));
        HVX_وکتور neg_vd = Q6_V_vxor_VV(vd, v_sign_بیت);
        HVX_وکتور v_qs   = *(ثابت HVX_UVector *) src[i].qs;

        عائم * بلوک_dst = dst_ptr + i * QK1_0;
        #رهنمود unroll
        لكل (صحيح q = 0; q < 4; q++) {
            HVX_وکتور v_بیتها = hvx_vec_repl_u32(q == 0 ? v_qs : Q6_V_vror_VR(v_qs, 4 * q));
            HVX_وکتور_Pred q_zero = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(v_بیتها, v_بیت_ماسک), v_zero);
            hvx_vmem(بلوک_dst + q * 32) = Q6_V_vmux_QVV(q_zero, neg_vd, vd);
        }
    }
}

ساكن مضمن فراغ hvx_dequantize_ردیف_flat_q2_0_f32(عائم * محدود dst_ptr, ثابت فراغ * محدود src_ptr, uint32_t n) {
    ثابت صحيح nb = n / QK2_0;
    ثابت بلوک_q2_0 * src = (ثابت بلوک_q2_0 *) src_ptr;
    ثابت HVX_وکتور ماسک_03 = Q6_Vb_vsplat_R(0x03);
    ثابت HVX_وکتور v_one   = Q6_Vb_vsplat_R(1);

    لكل (صحيح i = 0; i < nb; i++) {
        HVX_وکتور vd_f16 = Q6_Vh_vsplat_R(*(ثابت int16_t *) &src[i].d);
        HVX_وکتور vd     = Q6_V_lo_W(hvx_vec_f16_مقصد_f32(vd_f16));

        HVX_وکتور vq  = *(ثابت HVX_UVector *) src[i].qs;
        HVX_وکتور v0  = Q6_V_vand_VV(vq, ماسک_03);
        HVX_وکتور v1  = Q6_V_vand_VV(Q6_Vub_vlsr_VubR(vq, 2), ماسک_03);
        HVX_وکتور v2  = Q6_V_vand_VV(Q6_Vub_vlsr_VubR(vq, 4), ماسک_03);
        HVX_وکتور v3  = Q6_Vub_vlsr_VubR(vq, 6);

        HVX_وکتور v01 = Q6_V_lo_W(Q6_W_vshuff_VVR(v1, v0, -1));
        HVX_وکتور v23 = Q6_V_lo_W(Q6_W_vshuff_VVR(v3, v2, -1));
        HVX_وکتور v64 = Q6_Vb_vsub_VbVb(Q6_V_lo_W(Q6_W_vshuff_VVR(v23, v01, -2)), v_one);

        HVX_وکتور_Pair p16 = Q6_Wh_vunpack_Vb(v64);
        HVX_وکتور_Pair p32 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16));

        عائم * بلوک_dst = dst_ptr + i * QK2_0;
        hvx_vmem(بلوک_dst + 0)  = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_lo_W(p32)), vd);
        hvx_vmem(بلوک_dst + 32) = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_hi_W(p32)), vd);
    }
}

ساكن مضمن فراغ hvx_dequantize_ردیف_flat_q4_0_f32(عائم * محدود dst_ptr, ثابت فراغ * محدود src_ptr, uint32_t n) {
    ثابت صحيح nb = n / QK4_0;
    ثابت بلوک_q4_0 * src = (ثابت بلوک_q4_0 *) src_ptr;

    ثابت HVX_وکتور_Pred q16 = Q6_Q_vsetq_R(16);
    ثابت HVX_وکتور v_eight = Q6_Vb_vsplat_R(8);
    ثابت HVX_وکتور v_zero  = Q6_V_vzero();

    لكل (صحيح i = 0; i < nb; i++) {
        HVX_وکتور vd_f16     = Q6_Vh_vsplat_R(*(ثابت int16_t *) &src[i].d);
        HVX_وکتور_Pair vp_f32 = hvx_vec_f16_مقصد_f32(vd_f16);
        HVX_وکتور vd         = Q6_V_lo_W(vp_f32);

        HVX_وکتور vq = *(ثابت HVX_UVector *) src[i].qs;
        HVX_وکتور v_lo = Q6_V_vand_VV(vq, Q6_Vb_vsplat_R(0x0F));
        HVX_وکتور v_hi = Q6_Vub_vlsr_VubR(vq, 4);

        HVX_وکتور v_lo_s8 = Q6_Vb_vsub_VbVb(v_lo, v_eight);
        HVX_وکتور v_hi_s8 = Q6_Vb_vsub_VbVb(v_hi, v_eight);

        HVX_وکتور v32 = Q6_V_vmux_QVV(q16, v_lo_s8, Q6_V_vlalign_VVR(v_hi_s8, v_zero, 16));

        HVX_وکتور_Pair p16   = Q6_Wh_vunpack_Vb(v32);
        HVX_وکتور     v_i16 = Q6_V_lo_W(p16);
        HVX_وکتور_Pair p32   = Q6_Ww_vunpack_Vh(v_i16);
        HVX_وکتور     v_i32 = Q6_V_lo_W(p32);

        HVX_وکتور v_f32 = Q6_Vsf_equals_Vw(v_i32);
        HVX_وکتور res   = hvx_vec_mul_f32_f32(v_f32, vd);

        hvx_vmem(dst_ptr + i * QK4_0) = res;
    }
}

ساكن مضمن فراغ hvx_dequantize_ردیف_flat_q4_k_f32(عائم * محدود dst_ptr, ثابت فراغ * محدود src_ptr, uint32_t n) {
    ثابت صحيح nb = n / QK_K;
    ثابت بلوک_q4_K * src = (ثابت بلوک_q4_K *) src_ptr;

    لكل (صحيح i = 0; i < nb; i++) {
        ثابت عائم d   = GGML_FP16_مقصد_FP32(src[i].d);
        ثابت عائم min = GGML_FP16_مقصد_FP32(src[i].dmin);
        ثابت uint8_t * sc = src[i].scales;

        عائم * y = dst_ptr + i * QK_K;

        لكل (صحيح گام = 0; گام < 4; ++گام) {
            uint8_t sc1, m1_u, sc2, m2_u;
            إذا (گام < 2) {
                sc1  = sc[2 * گام + 0] & 63;
                m1_u = sc[2 * گام + 4] & 63;
                sc2  = sc[2 * گام + 1] & 63;
                m2_u = sc[2 * گام + 5] & 63;
            } وإلا {
                ثابت صحيح s = گام - 2;
                sc1  = (sc[8 + 2 * s + 0] & 0xF) | ((sc[2 * s + 0] >> 6) << 4);
                m1_u = (sc[8 + 2 * s + 0] >>  4) | ((sc[4 + 2 * s + 0] >> 6) << 4);
                sc2  = (sc[8 + 2 * s + 1] & 0xF) | ((sc[2 * s + 1] >> 6) << 4);
                m2_u = (sc[8 + 2 * s + 1] >>  4) | ((sc[4 + 2 * s + 1] >> 6) << 4);
            }

            HVX_وکتور vq = *(ثابت HVX_UVector *) (src[i].qs + گام * 32);
            HVX_وکتور v_lo = Q6_V_vand_VV(vq, Q6_Vb_vsplat_R(0x0F));
            HVX_وکتور v_hi = Q6_Vub_vlsr_VubR(vq, 4);

            HVX_وکتور_Pair p16_lo = Q6_Wuh_vunpack_Vub(v_lo);
            HVX_وکتور_Pair p32_lo = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16_lo));
            HVX_وکتور     v_i32_lo = Q6_V_lo_W(p32_lo);
            HVX_وکتور     v_f32_lo = Q6_Vsf_equals_Vw(v_i32_lo);

            HVX_وکتور vd1 = hvx_vec_splat_f32(d * sc1);
            HVX_وکتور vm1 = hvx_vec_splat_f32(min * m1_u);
            HVX_وکتور v_out1 = hvx_vec_sub_f32_f32(hvx_vec_mul_f32_f32(v_f32_lo, vd1), vm1);
            hvx_vmem(y + 0) = v_out1;

            HVX_وکتور_Pair p16_hi = Q6_Wuh_vunpack_Vub(v_hi);
            HVX_وکتور_Pair p32_hi = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16_hi));
            HVX_وکتور     v_i32_hi = Q6_V_lo_W(p32_hi);
            HVX_وکتور     v_f32_hi = Q6_Vsf_equals_Vw(v_i32_hi);

            HVX_وکتور vd2 = hvx_vec_splat_f32(d * sc2);
            HVX_وکتور vm2 = hvx_vec_splat_f32(min * m2_u);
            HVX_وکتور v_out2 = hvx_vec_sub_f32_f32(hvx_vec_mul_f32_f32(v_f32_hi, vd2), vm2);
            hvx_vmem(y + 32) = v_out2;

            y += 64;
        }
    }
}

ساكن مضمن فراغ hvx_dequantize_ردیف_flat_q5_k_f32(عائم * محدود dst_ptr, ثابت فراغ * محدود src_ptr, uint32_t n) {
    ثابت صحيح nb = n / QK_K;
    ثابت بلوک_q5_K * src = (ثابت بلوک_q5_K *) src_ptr;

    لكل (صحيح i = 0; i < nb; i++) {
        ثابت عائم d   = GGML_FP16_مقصد_FP32(src[i].d);
        ثابت عائم min = GGML_FP16_مقصد_FP32(src[i].dmin);
        ثابت uint8_t * sc = src[i].scales;

        HVX_وکتور v_qh = *(ثابت HVX_UVector *) src[i].qh;
        عائم * y = dst_ptr + i * QK_K;

        لكل (صحيح گام = 0; گام < 4; ++گام) {
            uint8_t sc1, m1_u, sc2, m2_u;
            إذا (گام < 2) {
                sc1  = sc[2 * گام + 0] & 63;
                m1_u = sc[2 * گام + 4] & 63;
                sc2  = sc[2 * گام + 1] & 63;
                m2_u = sc[2 * گام + 5] & 63;
            } وإلا {
                ثابت صحيح s = گام - 2;
                sc1  = (sc[8 + 2 * s + 0] & 0xF) | ((sc[2 * s + 0] >> 6) << 4);
                m1_u = (sc[8 + 2 * s + 0] >>  4) | ((sc[4 + 2 * s + 0] >> 6) << 4);
                sc2  = (sc[8 + 2 * s + 1] & 0xF) | ((sc[2 * s + 1] >> 6) << 4);
                m2_u = (sc[8 + 2 * s + 1] >>  4) | ((sc[4 + 2 * s + 1] >> 6) << 4);
            }

            HVX_وکتور v_ql = *(ثابت HVX_UVector *) (src[i].qs + گام * 32);
            HVX_وکتور v_lo = Q6_V_vand_VV(v_ql, Q6_Vb_vsplat_R(0x0F));
            HVX_وکتور v_hi = Q6_Vub_vlsr_VubR(v_ql, 4);

            HVX_وکتور v_h1 = Q6_Vw_vasl_VwR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v_qh, 2 * گام), Q6_Vb_vsplat_R(1)), 4);
            HVX_وکتور v_val1 = Q6_V_vor_VV(v_lo, v_h1);

            HVX_وکتور_Pair p16_1 = Q6_Wuh_vunpack_Vub(v_val1);
            HVX_وکتور_Pair p32_1 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16_1));
            HVX_وکتور     v_i32_1 = Q6_V_lo_W(p32_1);
            HVX_وکتور     v_f32_1 = Q6_Vsf_equals_Vw(v_i32_1);

            HVX_وکتور vd1 = hvx_vec_splat_f32(d * sc1);
            HVX_وکتور vm1 = hvx_vec_splat_f32(min * m1_u);
            HVX_وکتور v_out1 = hvx_vec_sub_f32_f32(hvx_vec_mul_f32_f32(v_f32_1, vd1), vm1);
            hvx_vmem(y + 0) = v_out1;

            HVX_وکتور v_h2 = Q6_Vw_vasl_VwR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v_qh, 2 * گام + 1), Q6_Vb_vsplat_R(1)), 4);
            HVX_وکتور v_val2 = Q6_V_vor_VV(v_hi, v_h2);

            HVX_وکتور_Pair p16_2 = Q6_Wuh_vunpack_Vub(v_val2);
            HVX_وکتور_Pair p32_2 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16_2));
            HVX_وکتور     v_i32_2 = Q6_V_lo_W(p32_2);
            HVX_وکتور     v_f32_2 = Q6_Vsf_equals_Vw(v_i32_2);

            HVX_وکتور vd2 = hvx_vec_splat_f32(d * sc2);
            HVX_وکتور vm2 = hvx_vec_splat_f32(min * m2_u);
            HVX_وکتور v_out2 = hvx_vec_sub_f32_f32(hvx_vec_mul_f32_f32(v_f32_2, vd2), vm2);
            hvx_vmem(y + 32) = v_out2;

            y += 64;
        }
    }
}

ساكن مضمن فراغ hvx_dequantize_ردیف_flat_q6_k_f32(عائم * محدود dst_ptr, ثابت فراغ * محدود src_ptr, uint32_t n) {
    ثابت صحيح nb = n / QK_K;
    ثابت بلوک_q6_K * src = (ثابت بلوک_q6_K *) src_ptr;

    ثابت HVX_وکتور_Pred q64 = Q6_Q_vsetq_R(64);
    ثابت HVX_وکتور v32_b   = Q6_Vb_vsplat_R(32);
    ثابت HVX_وکتور ماسک_0f = Q6_Vb_vsplat_R(0x0F);
    ثابت HVX_وکتور ماسک_03 = Q6_Vb_vsplat_R(0x03);

    لكل (صحيح i = 0; i < nb; i++) {
        ثابت عائم d = GGML_FP16_مقصد_FP32(src[i].d);

        ثابت uint8_t * محدود ql = src[i].ql;
        ثابت uint8_t * محدود qh = src[i].qh;
        ثابت int8_t  * محدود sc = src[i].scales;

        عائم * y = dst_ptr + i * QK_K;

        لكل (صحيح half = 0; half < 2; ++half) {
            عائم s[8];
            لكل (صحيح j = 0; j < 8; ++j) {
                s[j] = d * sc[j];
            }

            HVX_وکتور vscale1 = Q6_V_vmux_QVV(q64, hvx_vec_splat_f32(s[0]), hvx_vec_splat_f32(s[1]));
            HVX_وکتور vscale2 = Q6_V_vmux_QVV(q64, hvx_vec_splat_f32(s[2]), hvx_vec_splat_f32(s[3]));
            HVX_وکتور vscale3 = Q6_V_vmux_QVV(q64, hvx_vec_splat_f32(s[4]), hvx_vec_splat_f32(s[5]));
            HVX_وکتور vscale4 = Q6_V_vmux_QVV(q64, hvx_vec_splat_f32(s[6]), hvx_vec_splat_f32(s[7]));

            HVX_وکتور v_ql0 = *(ثابت HVX_UVector *) (ql);
            HVX_وکتور v_ql1 = *(ثابت HVX_UVector *) (ql + 32);
            HVX_وکتور v_qh  = *(ثابت HVX_UVector *) (qh);

            HVX_وکتور ql0_lo = Q6_V_vand_VV(v_ql0, ماسک_0f);
            HVX_وکتور qh_0   = Q6_Vw_vasl_VwR(Q6_V_vand_VV(v_qh, ماسک_03), 4);
            HVX_وکتور v_q1   = Q6_Vb_vsub_VbVb(Q6_V_vor_VV(ql0_lo, qh_0), v32_b);

            HVX_وکتور ql1_lo = Q6_V_vand_VV(v_ql1, ماسک_0f);
            HVX_وکتور qh_1   = Q6_Vw_vasl_VwR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v_qh, 2), ماسک_03), 4);
            HVX_وکتور v_q2   = Q6_Vb_vsub_VbVb(Q6_V_vor_VV(ql1_lo, qh_1), v32_b);

            HVX_وکتور ql0_hi = Q6_Vub_vlsr_VubR(v_ql0, 4);
            HVX_وکتور qh_2   = Q6_Vw_vasl_VwR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v_qh, 4), ماسک_03), 4);
            HVX_وکتور v_q3   = Q6_Vb_vsub_VbVb(Q6_V_vor_VV(ql0_hi, qh_2), v32_b);

            HVX_وکتور ql1_hi = Q6_Vub_vlsr_VubR(v_ql1, 4);
            HVX_وکتور qh_3   = Q6_Vw_vasl_VwR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v_qh, 6), ماسک_03), 4);
            HVX_وکتور v_q4   = Q6_Vb_vsub_VbVb(Q6_V_vor_VV(ql1_hi, qh_3), v32_b);

            HVX_وکتور_Pair p16_1 = Q6_Wh_vunpack_Vb(v_q1);
            HVX_وکتور_Pair p32_1 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16_1));
            HVX_وکتور     v_f32_1 = Q6_Vsf_equals_Vw(Q6_V_lo_W(p32_1));
            hvx_vmem(y + 0) = hvx_vec_mul_f32_f32(v_f32_1, vscale1);

            HVX_وکتور_Pair p16_2 = Q6_Wh_vunpack_Vb(v_q2);
            HVX_وکتور_Pair p32_2 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16_2));
            HVX_وکتور     v_f32_2 = Q6_Vsf_equals_Vw(Q6_V_lo_W(p32_2));
            hvx_vmem(y + 32) = hvx_vec_mul_f32_f32(v_f32_2, vscale2);

            HVX_وکتور_Pair p16_3 = Q6_Wh_vunpack_Vb(v_q3);
            HVX_وکتور_Pair p32_3 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16_3));
            HVX_وکتور     v_f32_3 = Q6_Vsf_equals_Vw(Q6_V_lo_W(p32_3));
            hvx_vmem(y + 64) = hvx_vec_mul_f32_f32(v_f32_3, vscale3);

            HVX_وکتور_Pair p16_4 = Q6_Wh_vunpack_Vb(v_q4);
            HVX_وکتور_Pair p32_4 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16_4));
            HVX_وکتور     v_f32_4 = Q6_Vsf_equals_Vw(Q6_V_lo_W(p32_4));
            hvx_vmem(y + 96) = hvx_vec_mul_f32_f32(v_f32_4, vscale4);

            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

// bytes row, row + 32, row + 64 and row + 96 of v, to bytes 0..3
ساكن مضمن HVX_وکتور hvx_dequantize_gather4(HVX_وکتور v, uint32_t ردیف) {
    v = Q6_V_vror_VR(v, ردیف);
    HVX_وکتور packed = Q6_V_vmux_QVV(Q6_Q_vsetq_R(1), v, Q6_V_vror_VR(v, 31));
    packed = Q6_V_vmux_QVV(Q6_Q_vsetq_R(2), packed, Q6_V_vror_VR(v, 62));
    أرجع Q6_V_vmux_QVV(Q6_Q_vsetq_R(3), packed, Q6_V_vror_VR(v, 93));
}

ساكن مضمن فراغ hvx_dequantize_ردیف_tiled_q8_0_f32(عائم * dst_spad, ثابت uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t ردیف) {
    ثابت HVX_وکتور_Pred first2 = Q6_Q_vsetq_R(2);
    ثابت HVX_وکتور_Pred first4 = Q6_Q_vsetq_R(4);

    لكل (uint32_t k_tile = 0; k_tile < n_k_tiles; ++k_tile) {
        ثابت uint8_t * tile = src_spad + k_tile * tile_stride;
        عائم * dst = dst_spad + k_tile * HTP_TENSOR_TILE_N_تعداد_ستون;

        HVX_وکتور vq = Q6_V_vzero();
        لكل (صحيح گروه = 7; گروه >= 0; --گروه) {
            ثابت HVX_وکتور v = Q6_V_vror_VR(hvx_vmem(tile + گروه * VLEN), 2 * ردیف);
            ثابت HVX_وکتور packed = Q6_V_vmux_QVV(first2, v, Q6_V_vror_VR(v, 62));
            vq = Q6_V_vmux_QVV(first4, packed, Q6_V_vror_VR(vq, VLEN - 4));
        }

        ثابت HVX_وکتور scales = hvx_vmem(tile + 1024);
        ثابت HVX_وکتور scale_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 2 * ردیف));
        ثابت HVX_وکتور scale = Q6_V_lo_W(hvx_vec_f16_مقصد_f32(scale_hf));
        ثابت HVX_وکتور_Pair p16 = Q6_Wh_vunpack_Vb(vq);
        ثابت HVX_وکتور_Pair p32 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16));
        HVX_وکتور مقادیر = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_lo_W(p32)), scale);
        *(HVX_وکتور *) dst = مقادیر;
    }
}

ساكن مضمن فراغ hvx_dequantize_ردیف_tiled_q4_0_f32(عائم * dst_spad, ثابت uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t ردیف) {
    ثابت HVX_وکتور_Pred first4 = Q6_Q_vsetq_R(4);

    لكل (uint32_t k_tile = 0; k_tile < n_k_tiles; ++k_tile) {
        ثابت uint8_t * tile = src_spad + k_tile * tile_stride;
        عائم * dst = dst_spad + k_tile * HTP_TENSOR_TILE_N_تعداد_ستون;

        HVX_وکتور vq = Q6_V_vzero();
        لكل (صحيح گروه = 3; گروه >= 0; --گروه) {
            vq = Q6_V_vmux_QVV(first4, hvx_dequantize_gather4(hvx_vmem(tile + گروه * VLEN), ردیف), Q6_V_vror_VR(vq, VLEN - 4));
        }

        ثابت HVX_وکتور lo = Q6_V_vand_VV(vq, Q6_Vb_vsplat_R(0x0F));
        ثابت HVX_وکتور hi = Q6_Vub_vlsr_VubR(vq, 4);
        vq = Q6_V_lo_W(Q6_W_vshuff_VVR(hi, lo, -1));
        vq = Q6_Vb_vsub_VbVb(vq, Q6_Vb_vsplat_R(8));

        ثابت HVX_وکتور scales = hvx_vmem(tile + 512);
        ثابت HVX_وکتور scale_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 2 * ردیف));
        ثابت HVX_وکتور scale = Q6_V_lo_W(hvx_vec_f16_مقصد_f32(scale_hf));
        ثابت HVX_وکتور_Pair p16 = Q6_Wh_vunpack_Vb(vq);
        ثابت HVX_وکتور_Pair p32 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16));
        HVX_وکتور مقادیر = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_lo_W(p32)), scale);
        *(HVX_وکتور *) dst = مقادیر;
    }
}

ساكن مضمن فراغ hvx_dequantize_ردیف_tiled_q4_k_f32(عائم * dst_spad, ثابت uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t ردیف) {
    ثابت HVX_وکتور_Pred first4 = Q6_Q_vsetq_R(4);

    لكل (uint32_t k_tile = 0; k_tile < n_k_tiles; ++k_tile) {
        ثابت uint8_t * tile = src_spad + k_tile * tile_stride;
        عائم * dst = dst_spad + k_tile * HTP_TENSOR_TILE_N_تعداد_ستون;

        HVX_وکتور vq = Q6_V_vzero();
        لكل (صحيح گروه = 3; گروه >= 0; --گروه) {
            vq = Q6_V_vmux_QVV(first4, hvx_dequantize_gather4(hvx_vmem(tile + گروه * VLEN), ردیف), Q6_V_vror_VR(vq, VLEN - 4));
        }

        ثابت HVX_وکتور lo = Q6_V_vand_VV(vq, Q6_Vb_vsplat_R(0x0F));
        ثابت HVX_وکتور hi = Q6_Vub_vlsr_VubR(vq, 4);
        vq = Q6_V_lo_W(Q6_W_vshuff_VVR(hi, lo, -1));

        ثابت HVX_وکتور scales = hvx_vmem(tile + 512);
        ثابت HVX_وکتور scale_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 4 * ردیف));
        ثابت HVX_وکتور scale = Q6_V_lo_W(hvx_vec_f16_مقصد_f32(scale_hf));
        ثابت HVX_وکتور جابه_جایی_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 4 * ردیف + 2));
        ثابت HVX_وکتور جابه_جایی = Q6_V_lo_W(hvx_vec_f16_مقصد_f32(جابه_جایی_hf));

        ثابت HVX_وکتور_Pair p16 = Q6_Wh_vunpack_Vb(vq);
        ثابت HVX_وکتور_Pair p32 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16));
        HVX_وکتور مقادیر = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_lo_W(p32)), scale);
        *(HVX_وکتور *) dst = hvx_vec_جمع_کن_f32_f32(مقادیر, جابه_جایی);
    }
}

ساكن مضمن فراغ hvx_dequantize_ردیف_tiled_q5_k_f32(عائم * dst_spad, ثابت uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t ردیف) {
    ثابت HVX_وکتور_Pred first4 = Q6_Q_vsetq_R(4);
    ثابت HVX_وکتور one = Q6_Vb_vsplat_R(1);

    لكل (uint32_t k_tile = 0; k_tile < n_k_tiles; ++k_tile) {
        ثابت uint8_t * tile = src_spad + k_tile * tile_stride;
        عائم * dst = dst_spad + k_tile * HTP_TENSOR_TILE_N_تعداد_ستون;

        ثابت HVX_وکتور p5 = hvx_dequantize_gather4(hvx_vmem(tile + 640), ردیف);
        HVX_وکتور vq = Q6_V_vzero();
        HVX_وکتور vh = Q6_V_vzero();
        لكل (صحيح گروه = 3; گروه >= 0; --گروه) {
            vq = Q6_V_vmux_QVV(first4, hvx_dequantize_gather4(hvx_vmem(tile + گروه * VLEN), ردیف), Q6_V_vror_VR(vq, VLEN - 4));
            vh = Q6_V_vmux_QVV(first4, Q6_Vub_vlsr_VubR(p5, 2 * گروه), Q6_V_vror_VR(vh, VLEN - 4));
        }

        ثابت HVX_وکتور lo = Q6_V_vor_VV(Q6_V_vand_VV(vq, Q6_Vb_vsplat_R(0x0F)), Q6_Vw_vasl_VwR(Q6_V_vand_VV(vh, one), 4));
        ثابت HVX_وکتور hi = Q6_V_vor_VV(Q6_Vub_vlsr_VubR(vq, 4), Q6_Vw_vasl_VwR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(vh, 1), one), 4));
        vq = Q6_V_lo_W(Q6_W_vshuff_VVR(hi, lo, -1));

        ثابت HVX_وکتور scales = hvx_vmem(tile + 512);
        ثابت HVX_وکتور scale_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 4 * ردیف));
        ثابت HVX_وکتور scale = Q6_V_lo_W(hvx_vec_f16_مقصد_f32(scale_hf));
        ثابت HVX_وکتور جابه_جایی_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 4 * ردیف + 2));
        ثابت HVX_وکتور جابه_جایی = Q6_V_lo_W(hvx_vec_f16_مقصد_f32(جابه_جایی_hf));

        ثابت HVX_وکتور_Pair p16 = Q6_Wh_vunpack_Vb(vq);
        ثابت HVX_وکتور_Pair p32 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16));
        HVX_وکتور مقادیر = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_lo_W(p32)), scale);
        *(HVX_وکتور *) dst = hvx_vec_جمع_کن_f32_f32(مقادیر, جابه_جایی);
    }
}

ساكن مضمن فراغ hvx_dequantize_ردیف_tiled_q6_k_f32(عائم * dst_spad, ثابت uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t ردیف) {
    ثابت HVX_وکتور_Pred first4 = Q6_Q_vsetq_R(4);
    ثابت HVX_وکتور_Pred first16 = Q6_Q_vsetq_R(16 * حجم(عائم));
    ثابت HVX_وکتور ماسک_0f = Q6_Vb_vsplat_R(0x0F);
    ثابت HVX_وکتور ماسک_03 = Q6_Vb_vsplat_R(0x03);

    لكل (uint32_t k_tile = 0; k_tile < n_k_tiles; ++k_tile) {
        ثابت uint8_t * tile = src_spad + k_tile * tile_stride;
        عائم * dst = dst_spad + k_tile * HTP_TENSOR_TILE_N_تعداد_ستون;

        HVX_وکتور vq = Q6_V_vzero();
        لكل (صحيح گروه = 7; گروه >= 0; --گروه) {
            ثابت HVX_وکتور lo_plane = Q6_V_vror_VR(hvx_vmem(tile + (گروه >> 1) * VLEN), 4 * ردیف);
            ثابت HVX_وکتور hi_plane = Q6_V_vror_VR(hvx_vmem(tile + 512 + (گروه >> 2) * VLEN), 4 * ردیف);
            ثابت HVX_وکتور lo = (گروه & 1) ? Q6_Vub_vlsr_VubR(lo_plane, 4) : Q6_V_vand_VV(lo_plane, ماسک_0f);
            ثابت HVX_وکتور hi = Q6_Vub_vlsr_VubR(hi_plane, 2 * (گروه & 3));
            ثابت HVX_وکتور packed = Q6_V_vor_VV(lo, Q6_Vw_vasl_VwR(Q6_V_vand_VV(hi, ماسک_03), 4));
            vq = Q6_V_vmux_QVV(first4, packed, Q6_V_vror_VR(vq, VLEN - 4));
        }

        ثابت HVX_وکتور scales = hvx_vmem(tile + 768);
        ثابت HVX_وکتور scale_lo_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 2 * ردیف));
        ثابت HVX_وکتور scale_hi_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 64 + 2 * ردیف));
        ثابت HVX_وکتور scale_lo = Q6_V_lo_W(hvx_vec_f16_مقصد_f32(scale_lo_hf));
        ثابت HVX_وکتور scale_hi = Q6_V_lo_W(hvx_vec_f16_مقصد_f32(scale_hi_hf));
        ثابت HVX_وکتور scale = Q6_V_vmux_QVV(first16, scale_lo, scale_hi);
        ثابت HVX_وکتور_Pair p16 = Q6_Wh_vunpack_Vb(Q6_Vb_vsub_VbVb(vq, Q6_Vb_vsplat_R(32)));
        ثابت HVX_وکتور_Pair p32 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16));
        *(HVX_وکتور *) dst = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_lo_W(p32)), scale);
    }
}

ساكن مضمن HVX_وکتور hvx_dequantize_tiled_f16_gather4(ثابت uint8_t * p, uint32_t r_byte) {
    HVX_وکتور w0  = Q6_V_vror_VR(hvx_vmem(p + 0 * VLEN), r_byte);
    HVX_وکتور w1  = Q6_V_vror_VR(hvx_vmem(p + 1 * VLEN), r_byte);
    HVX_وکتور w01 = Q6_V_lo_W(Q6_W_vshuff_VVR(w1, w0, 4));
    HVX_وکتور w2  = Q6_V_vror_VR(hvx_vmem(p + 2 * VLEN), r_byte);
    HVX_وکتور w3  = Q6_V_vror_VR(hvx_vmem(p + 3 * VLEN), r_byte);
    HVX_وکتور w23 = Q6_V_lo_W(Q6_W_vshuff_VVR(w3, w2, 4));
    أرجع Q6_V_lo_W(Q6_W_vshuff_VVR(w23, w01, 8));
}

ساكن مضمن فراغ hvx_dequantize_ردیف_tiled_f16_f32(عائم * dst_spad, ثابت uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t ردیف) {
    ثابت uint32_t r_byte = 4 * ردیف;

    لكل (uint32_t k_tile = 0; k_tile < n_k_tiles; ++k_tile) {
        ثابت uint8_t * tile = src_spad + k_tile * tile_stride;
        عائم * dst = dst_spad + k_tile * HTP_TENSOR_TILE_N_تعداد_ستون;

        HVX_وکتور g0 = hvx_dequantize_tiled_f16_gather4(tile + 0 * VLEN, r_byte);
        HVX_وکتور g1 = hvx_dequantize_tiled_f16_gather4(tile + 4 * VLEN, r_byte);
        HVX_وکتور g01 = Q6_V_lo_W(Q6_W_vshuff_VVR(g1, g0, 16));

        HVX_وکتور g2 = hvx_dequantize_tiled_f16_gather4(tile + 8 * VLEN, r_byte);
        HVX_وکتور g3 = hvx_dequantize_tiled_f16_gather4(tile + 12 * VLEN, r_byte);
        HVX_وکتور g23 = Q6_V_lo_W(Q6_W_vshuff_VVR(g3, g2, 16));

        HVX_وکتور g = Q6_V_lo_W(Q6_W_vshuff_VVR(g23, g01, 32));
        *(HVX_وکتور *) dst = Q6_V_lo_W(hvx_vec_f16_مقصد_f32(g));
    }
}

ساكن مضمن فراغ hvx_dequantize_ردیف_tiled_q1_0_f32(عائم * dst_spad, ثابت uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t ردیف) {
    ساكن ثابت uint32_t __attribute__((aligned(128))) بیت_ماسک[32] = {
        1u <<  0, 1u <<  8, 1u << 16, 1u << 24,
        1u <<  1, 1u <<  9, 1u << 17, 1u << 25,
        1u <<  2, 1u << 10, 1u << 18, 1u << 26,
        1u <<  3, 1u << 11, 1u << 19, 1u << 27,
        1u <<  4, 1u << 12, 1u << 20, 1u << 28,
        1u <<  5, 1u << 13, 1u << 21, 1u << 29,
        1u <<  6, 1u << 14, 1u << 22, 1u << 30,
        1u <<  7, 1u << 15, 1u << 23, 1u << 31,
    };
    ثابت HVX_وکتور v_بیت_ماسک = *(ثابت HVX_وکتور *) بیت_ماسک;
    ثابت HVX_وکتور v_zero     = Q6_V_vzero();
    ثابت HVX_وکتور v_sign_بیت = Q6_V_vsplat_R(0x80000000);

    لكل (uint32_t k_tile = 0; k_tile < n_k_tiles; ++k_tile) {
        ثابت uint8_t * tile = src_spad + k_tile * tile_stride;
        عائم * dst = dst_spad + k_tile * HTP_TENSOR_TILE_K_Q1_0;

        ثابت HVX_وکتور scales   = hvx_vmem(tile + 512);
        ثابت HVX_وکتور scale_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 2 * ردیف));
        ثابت HVX_وکتور vd       = Q6_V_lo_W(hvx_vec_f16_مقصد_f32(scale_hf));
        ثابت HVX_وکتور neg_vd   = Q6_V_vxor_VV(vd, v_sign_بیت);

        #رهنمود unroll
        لكل (صحيح q = 0; q < 4; ++q) {
            ثابت HVX_وکتور v_بیتها = hvx_vec_repl_u32(Q6_V_vror_VR(hvx_vmem(tile + q * VLEN), 4 * ردیف));
            ثابت HVX_وکتور_Pred q_zero = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(v_بیتها, v_بیت_ماسک), v_zero);
            hvx_vmem(dst + q * 32) = Q6_V_vmux_QVV(q_zero, neg_vd, vd);
        }
    }
}

ساكن مضمن فراغ hvx_dequantize_ردیف_tiled_q2_0_f32(عائم * dst_spad, ثابت uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t ردیف) {
    ثابت HVX_وکتور ماسک_03 = Q6_Vb_vsplat_R(0x03);
    ثابت HVX_وکتور v_one   = Q6_Vb_vsplat_R(1);

    لكل (uint32_t k_tile = 0; k_tile < n_k_tiles; ++k_tile) {
        ثابت uint8_t * tile = src_spad + k_tile * tile_stride;
        عائم * dst = dst_spad + k_tile * HTP_TENSOR_TILE_K_Q2_0;

        ثابت HVX_وکتور v_packed = hvx_dequantize_tiled_f16_gather4(tile, 4 * ردیف);
        ثابت HVX_وکتور p0 = Q6_V_vand_VV(v_packed, ماسک_03);
        ثابت HVX_وکتور p1 = Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v_packed, 2), ماسک_03);
        ثابت HVX_وکتور p2 = Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v_packed, 4), ماسک_03);
        ثابت HVX_وکتور p3 = Q6_Vub_vlsr_VubR(v_packed, 6);

        ثابت HVX_وکتور p01 = Q6_V_lo_W(Q6_W_vshuff_VVR(p1, p0, -4));
        ثابت HVX_وکتور p23 = Q6_V_lo_W(Q6_W_vshuff_VVR(p3, p2, -4));
        ثابت HVX_وکتور vq  = Q6_Vb_vsub_VbVb(Q6_V_lo_W(Q6_W_vshuff_VVR(p23, p01, -8)), v_one);

        ثابت HVX_وکتور scales   = hvx_vmem(tile + 512);
        ثابت HVX_وکتور scale_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 2 * ردیف));
        ثابت HVX_وکتور scale    = Q6_V_lo_W(hvx_vec_f16_مقصد_f32(scale_hf));

        ثابت HVX_وکتور_Pair p16 = Q6_Wh_vunpack_Vb(vq);
        ثابت HVX_وکتور_Pair p32 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16));
        hvx_vmem(dst + 0)  = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_lo_W(p32)), scale);
        hvx_vmem(dst + 32) = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_hi_W(p32)), scale);
    }
}

#پایان_اگر // HVX_QUANT_H
