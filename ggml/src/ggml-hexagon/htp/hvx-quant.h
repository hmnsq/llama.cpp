#جيڪڏهن_وصف_نه_ٿيل HVX_QUANT_H
#وصف_ڪريو HVX_QUANT_H

#ضمن <math.h>
#ضمن <stdint.h>
#ضمن <string.h>

#ضمن "htp-tensor.h"
#ضمن "hvx-arith.h"
#ضمن "hvx-base.h"
#ضمن "hvx-reduce.h"
#ضمن "hvx-repl.h"
#ضمن "hvx-utils.h"

#جيڪڏهن_وصف_نه_ٿيل GGML_COMMON_DECL_C
#وصف_ڪريو GGML_COMMON_DECL_C
#جيڪڏهن_پورو
#ضمن "ggml-common.h"
#ضمن "ggml-impl.h"

ساكن مضمن فراغ hvx_quantize_صف_q8_0_f32(فراغ * محدود dst_ptr, ثابت عائم * محدود src_ptr, صحيح n) {
    ثابت صحيح nb = n / QK8_0;
    بلاڪ_q8_0 * dst = (بلاڪ_q8_0 *) dst_ptr;
    HVX_ويڪٽر zero = Q6_V_vzero();

    صحيح i = 0;
    لكل (; i + 3 < nb; i += 4) {
        HVX_ويڪٽر * vx = (HVX_ويڪٽر *) (src_ptr + i * QK8_0);

        HVX_ويڪٽر vmax0_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[0]));
        HVX_ويڪٽر vmax1_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[1]));
        HVX_ويڪٽر vmax2_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[2]));
        HVX_ويڪٽر vmax3_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[3]));

        HVX_ويڪٽر vx0_qf = Q6_Vqf32_vsub_VsfVsf(vx[0], zero);
        HVX_ويڪٽر vx1_qf = Q6_Vqf32_vsub_VsfVsf(vx[1], zero);
        HVX_ويڪٽر vx2_qf = Q6_Vqf32_vsub_VsfVsf(vx[2], zero);
        HVX_ويڪٽر vx3_qf = Q6_Vqf32_vsub_VsfVsf(vx[3], zero);

        HVX_ويڪٽر vmax0_qf = Q6_Vqf32_vsub_VsfVsf(vmax0_sf, zero);
        HVX_ويڪٽر vmax1_qf = Q6_Vqf32_vsub_VsfVsf(vmax1_sf, zero);
        HVX_ويڪٽر vmax2_qf = Q6_Vqf32_vsub_VsfVsf(vmax2_sf, zero);
        HVX_ويڪٽر vmax3_qf = Q6_Vqf32_vsub_VsfVsf(vmax3_sf, zero);

        HVX_ويڪٽر vmax01_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vmax1_qf, vmax0_qf)));
        HVX_ويڪٽر vmax23_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vmax3_qf, vmax2_qf)));

        HVX_ويڪٽر vx01_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx1_qf, vx0_qf)));
        HVX_ويڪٽر vx23_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx3_qf, vx2_qf)));

        HVX_ويڪٽر vd01_qf16 = Q6_Vqf16_vmpy_VhfVhf(vmax01_hf, Q6_Vh_vsplat_R(0x2008));  // 1.0 / 127.0
        HVX_ويڪٽر vd23_qf16 = Q6_Vqf16_vmpy_VhfVhf(vmax23_hf, Q6_Vh_vsplat_R(0x2008));  // 1.0 / 127.0
        HVX_ويڪٽر vd01_hf   = Q6_Vhf_equals_Vqf16(vd01_qf16);
        HVX_ويڪٽر vd23_hf   = Q6_Vhf_equals_Vqf16(vd23_qf16);

        HVX_ويڪٽر vd01_inv_hf = hvx_vec_inverse_f16(vd01_hf);
        HVX_ويڪٽر vd23_inv_hf = hvx_vec_inverse_f16(vd23_hf);
        vx01_hf              = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx01_hf, vd01_inv_hf));
        vx23_hf              = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx23_hf, vd23_inv_hf));

        HVX_ويڪٽر vx01_i16 = hvx_vec_i16_from_hf_rnd_sat(vx01_hf);
        HVX_ويڪٽر vx23_i16 = hvx_vec_i16_from_hf_rnd_sat(vx23_hf);
        HVX_ويڪٽر vx_i8    = Q6_Vb_vpack_VhVh_sat(vx23_i16, vx01_i16);

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
        ثابت عائم * بلاڪ_src = src_ptr + i * QK8_0;
        HVX_ويڪٽر vx = *(ثابت HVX_UVector *) بلاڪ_src;
        HVX_ويڪٽر v_abs = hvx_vec_abs_f32(vx);
        HVX_ويڪٽر v_max = hvx_vec_reduce_max_f32(v_abs);
        عائم amax = hvx_vec_get_f32(v_max);

        ثابت عائم d = amax / 127.0f;
        ثابت عائم id = d ? (1.0f / d) : 0.0f;
        dst[i].d = GGML_FP32_منزل_FP16(d);

        HVX_ويڪٽر vid = hvx_vec_splat_f32(id);
        HVX_ويڪٽر v_scaled = hvx_vec_mul_f32_f32(vx, vid);
        HVX_ويڪٽر v_scaled_qf = Q6_Vqf32_vsub_VsfVsf(v_scaled, zero);
        HVX_ويڪٽر v_scaled_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(zero, v_scaled_qf)));
        HVX_ويڪٽر v_i16 = hvx_vec_i16_from_hf_rnd_sat(v_scaled_hf);
        HVX_ويڪٽر v_i8  = Q6_Vb_vpack_VhVh_sat(zero, v_i16);

        hvx_vec_store_u(dst[i].qs, 32, v_i8);
    }
}

ساكن مضمن فراغ hvx_dequantize_صف_flat_q8_0_f32(عائم * محدود dst_ptr, ثابت فراغ * محدود src_ptr, uint32_t n) {
    ثابت صحيح nb = n / QK8_0;
    ثابت بلاڪ_q8_0 * src = (ثابت بلاڪ_q8_0 *) src_ptr;

    لكل (صحيح i = 0; i < nb; i++) {
        HVX_ويڪٽر vd_f16     = Q6_Vh_vsplat_R(*(ثابت int16_t *) &src[i].d);
        HVX_ويڪٽر_Pair vp_f32 = hvx_vec_f16_منزل_f32(vd_f16);
        HVX_ويڪٽر vd         = Q6_V_lo_W(vp_f32);

        HVX_ويڪٽر vq_i8 = *(ثابت HVX_UVector *) src[i].qs;

        HVX_ويڪٽر_Pair p16   = Q6_Wh_vunpack_Vb(vq_i8);
        HVX_ويڪٽر     v_i16 = Q6_V_lo_W(p16);
        HVX_ويڪٽر_Pair p32   = Q6_Ww_vunpack_Vh(v_i16);
        HVX_ويڪٽر     v_i32 = Q6_V_lo_W(p32);

        HVX_ويڪٽر v_f32 = Q6_Vsf_equals_Vw(v_i32);
        HVX_ويڪٽر res   = hvx_vec_mul_f32_f32(v_f32, vd);

        عائم * بلاڪ_dst = dst_ptr + i * QK8_0;
        hvx_vmem(بلاڪ_dst) = res;
    }
}

ساكن مضمن فراغ hvx_dequantize_صف_flat_q8_0_f16(__fp16 * محدود dst_ptr, ثابت فراغ * محدود src_ptr, uint32_t n) {
    ثابت صحيح nb = n / QK8_0;
    ثابت بلاڪ_q8_0 * src = (ثابت بلاڪ_q8_0 *) src_ptr;

    لكل (صحيح i = nb - 1; i >= 0; i--) {
        HVX_ويڪٽر vd_f16     = Q6_Vh_vsplat_R(*(ثابت int16_t *) &src[i].d);
        HVX_ويڪٽر_Pair vp_f32 = hvx_vec_f16_منزل_f32(vd_f16);
        HVX_ويڪٽر vd         = Q6_V_lo_W(vp_f32);

        HVX_ويڪٽر vq_i8 = *(ثابت HVX_UVector *) src[i].qs;

        HVX_ويڪٽر_Pair p16   = Q6_Wh_vunpack_Vb(vq_i8);
        HVX_ويڪٽر     v_i16 = Q6_V_lo_W(p16);
        HVX_ويڪٽر_Pair p32   = Q6_Ww_vunpack_Vh(v_i16);
        HVX_ويڪٽر     v_i32 = Q6_V_lo_W(p32);

        HVX_ويڪٽر v_f32 = Q6_Vsf_equals_Vw(v_i32);
        HVX_ويڪٽر res_f32 = hvx_vec_mul_f32_f32(v_f32, vd);

        HVX_ويڪٽر res_f16 = hvx_vec_f32_منزل_f16(res_f32, Q6_V_vzero());

        __fp16 * بلاڪ_dst = dst_ptr + i * QK8_0;
        hvx_vec_store_u(بلاڪ_dst, QK8_0 * حجم(__fp16), res_f16);
    }
}

ساكن مضمن فراغ hvx_dequantize_صف_flat_f16_f32(عائم * محدود dst_ptr, ثابت فراغ * محدود src_ptr, uint32_t n) {
    ثابت uint32_t nvec = n / VLEN_FP32;
    ثابت uint32_t nloe = n % VLEN_FP32;
    ثابت _Float16 * src = (ثابت _Float16 *) src_ptr;

    لكل (uint32_t i = 0; i < nvec; i++) {
        HVX_ويڪٽر v_f16 = *(ثابت HVX_UVector *) (src + i * VLEN_FP32);
        HVX_ويڪٽر_Pair vp_f32 = hvx_vec_f16_منزل_f32(v_f16);
        HVX_ويڪٽر res = Q6_V_lo_W(vp_f32);

        hvx_vmem(dst_ptr + i * VLEN_FP32) = res;
    }

    إذا (nloe) {
        HVX_ويڪٽر v_f16 = *(ثابت HVX_UVector *) (src + nvec * VLEN_FP32);
        HVX_ويڪٽر_Pair vp_f32 = hvx_vec_f16_منزل_f32(v_f16);
        HVX_ويڪٽر res = Q6_V_lo_W(vp_f32);

        hvx_vec_store_a(dst_ptr + nvec * VLEN_FP32, nloe * حجم(عائم), res);
    }
}

ساكن مضمن فراغ hvx_dequantize_صف_flat_bf16_f32(عائم * محدود dst_ptr, ثابت فراغ * محدود src_ptr, uint32_t n) {
    ثابت uint32_t nvec = n / VLEN_FP32;
    ثابت uint32_t nloe = n % VLEN_FP32;
    ثابت uint16_t * src = (ثابت uint16_t *) src_ptr;

    لكل (uint32_t i = 0; i < nvec; i++) {
        HVX_ويڪٽر v_bf16 = *(ثابت HVX_UVector *) (src + i * VLEN_FP32);
        HVX_ويڪٽر_Pair p = Q6_Ww_vunpack_Vh(v_bf16);
        HVX_ويڪٽر res = Q6_Vw_vasl_VwR(Q6_V_lo_W(p), 16);

        hvx_vmem(dst_ptr + i * VLEN_FP32) = res;
    }

    إذا (nloe) {
        HVX_ويڪٽر v_bf16 = *(ثابت HVX_UVector *) (src + nvec * VLEN_FP32);
        HVX_ويڪٽر_Pair p = Q6_Ww_vunpack_Vh(v_bf16);
        HVX_ويڪٽر res = Q6_Vw_vasl_VwR(Q6_V_lo_W(p), 16);

        hvx_vec_store_a(dst_ptr + nvec * VLEN_FP32, nloe * حجم(عائم), res);
    }
}

ساكن مضمن فراغ hvx_dequantize_صف_flat_q1_0_f32(عائم * محدود dst_ptr, ثابت فراغ * محدود src_ptr, uint32_t n) {
    ساكن ثابت uint32_t __attribute__((aligned(128))) ٻٽو_انگ_نقاب[32] = {
        1u <<  0, 1u <<  1, 1u <<  2, 1u <<  3, 1u <<  4, 1u <<  5, 1u <<  6, 1u <<  7,
        1u <<  8, 1u <<  9, 1u << 10, 1u << 11, 1u << 12, 1u << 13, 1u << 14, 1u << 15,
        1u << 16, 1u << 17, 1u << 18, 1u << 19, 1u << 20, 1u << 21, 1u << 22, 1u << 23,
        1u << 24, 1u << 25, 1u << 26, 1u << 27, 1u << 28, 1u << 29, 1u << 30, 1u << 31,
    };
    ثابت صحيح nb = n / QK1_0;
    ثابت بلاڪ_q1_0 * src = (ثابت بلاڪ_q1_0 *) src_ptr;
    ثابت HVX_ويڪٽر v_ٻٽو_انگ_نقاب = *(ثابت HVX_ويڪٽر *) ٻٽو_انگ_نقاب;
    ثابت HVX_ويڪٽر v_zero     = Q6_V_vzero();
    ثابت HVX_ويڪٽر v_sign_ٻٽو_انگ = Q6_V_vsplat_R(0x80000000);

    لكل (صحيح i = 0; i < nb; i++) {
        HVX_ويڪٽر vd_f16 = Q6_Vh_vsplat_R(*(ثابت int16_t *) &src[i].d);
        HVX_ويڪٽر vd     = Q6_V_lo_W(hvx_vec_f16_منزل_f32(vd_f16));
        HVX_ويڪٽر neg_vd = Q6_V_vxor_VV(vd, v_sign_ٻٽو_انگ);
        HVX_ويڪٽر v_qs   = *(ثابت HVX_UVector *) src[i].qs;

        عائم * بلاڪ_dst = dst_ptr + i * QK1_0;
        #هدايت unroll
        لكل (صحيح q = 0; q < 4; q++) {
            HVX_ويڪٽر v_بٽ = hvx_vec_repl_u32(q == 0 ? v_qs : Q6_V_vror_VR(v_qs, 4 * q));
            HVX_ويڪٽر_Pred q_zero = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(v_بٽ, v_ٻٽو_انگ_نقاب), v_zero);
            hvx_vmem(بلاڪ_dst + q * 32) = Q6_V_vmux_QVV(q_zero, neg_vd, vd);
        }
    }
}

ساكن مضمن فراغ hvx_dequantize_صف_flat_q2_0_f32(عائم * محدود dst_ptr, ثابت فراغ * محدود src_ptr, uint32_t n) {
    ثابت صحيح nb = n / QK2_0;
    ثابت بلاڪ_q2_0 * src = (ثابت بلاڪ_q2_0 *) src_ptr;
    ثابت HVX_ويڪٽر نقاب_03 = Q6_Vb_vsplat_R(0x03);
    ثابت HVX_ويڪٽر v_one   = Q6_Vb_vsplat_R(1);

    لكل (صحيح i = 0; i < nb; i++) {
        HVX_ويڪٽر vd_f16 = Q6_Vh_vsplat_R(*(ثابت int16_t *) &src[i].d);
        HVX_ويڪٽر vd     = Q6_V_lo_W(hvx_vec_f16_منزل_f32(vd_f16));

        HVX_ويڪٽر vq  = *(ثابت HVX_UVector *) src[i].qs;
        HVX_ويڪٽر v0  = Q6_V_vand_VV(vq, نقاب_03);
        HVX_ويڪٽر v1  = Q6_V_vand_VV(Q6_Vub_vlsr_VubR(vq, 2), نقاب_03);
        HVX_ويڪٽر v2  = Q6_V_vand_VV(Q6_Vub_vlsr_VubR(vq, 4), نقاب_03);
        HVX_ويڪٽر v3  = Q6_Vub_vlsr_VubR(vq, 6);

        HVX_ويڪٽر v01 = Q6_V_lo_W(Q6_W_vshuff_VVR(v1, v0, -1));
        HVX_ويڪٽر v23 = Q6_V_lo_W(Q6_W_vshuff_VVR(v3, v2, -1));
        HVX_ويڪٽر v64 = Q6_Vb_vsub_VbVb(Q6_V_lo_W(Q6_W_vshuff_VVR(v23, v01, -2)), v_one);

        HVX_ويڪٽر_Pair p16 = Q6_Wh_vunpack_Vb(v64);
        HVX_ويڪٽر_Pair p32 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16));

        عائم * بلاڪ_dst = dst_ptr + i * QK2_0;
        hvx_vmem(بلاڪ_dst + 0)  = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_lo_W(p32)), vd);
        hvx_vmem(بلاڪ_dst + 32) = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_hi_W(p32)), vd);
    }
}

ساكن مضمن فراغ hvx_dequantize_صف_flat_q4_0_f32(عائم * محدود dst_ptr, ثابت فراغ * محدود src_ptr, uint32_t n) {
    ثابت صحيح nb = n / QK4_0;
    ثابت بلاڪ_q4_0 * src = (ثابت بلاڪ_q4_0 *) src_ptr;

    ثابت HVX_ويڪٽر_Pred q16 = Q6_Q_vsetq_R(16);
    ثابت HVX_ويڪٽر v_eight = Q6_Vb_vsplat_R(8);
    ثابت HVX_ويڪٽر v_zero  = Q6_V_vzero();

    لكل (صحيح i = 0; i < nb; i++) {
        HVX_ويڪٽر vd_f16     = Q6_Vh_vsplat_R(*(ثابت int16_t *) &src[i].d);
        HVX_ويڪٽر_Pair vp_f32 = hvx_vec_f16_منزل_f32(vd_f16);
        HVX_ويڪٽر vd         = Q6_V_lo_W(vp_f32);

        HVX_ويڪٽر vq = *(ثابت HVX_UVector *) src[i].qs;
        HVX_ويڪٽر v_lo = Q6_V_vand_VV(vq, Q6_Vb_vsplat_R(0x0F));
        HVX_ويڪٽر v_hi = Q6_Vub_vlsr_VubR(vq, 4);

        HVX_ويڪٽر v_lo_s8 = Q6_Vb_vsub_VbVb(v_lo, v_eight);
        HVX_ويڪٽر v_hi_s8 = Q6_Vb_vsub_VbVb(v_hi, v_eight);

        HVX_ويڪٽر v32 = Q6_V_vmux_QVV(q16, v_lo_s8, Q6_V_vlalign_VVR(v_hi_s8, v_zero, 16));

        HVX_ويڪٽر_Pair p16   = Q6_Wh_vunpack_Vb(v32);
        HVX_ويڪٽر     v_i16 = Q6_V_lo_W(p16);
        HVX_ويڪٽر_Pair p32   = Q6_Ww_vunpack_Vh(v_i16);
        HVX_ويڪٽر     v_i32 = Q6_V_lo_W(p32);

        HVX_ويڪٽر v_f32 = Q6_Vsf_equals_Vw(v_i32);
        HVX_ويڪٽر res   = hvx_vec_mul_f32_f32(v_f32, vd);

        hvx_vmem(dst_ptr + i * QK4_0) = res;
    }
}

ساكن مضمن فراغ hvx_dequantize_صف_flat_q4_k_f32(عائم * محدود dst_ptr, ثابت فراغ * محدود src_ptr, uint32_t n) {
    ثابت صحيح nb = n / QK_K;
    ثابت بلاڪ_q4_K * src = (ثابت بلاڪ_q4_K *) src_ptr;

    لكل (صحيح i = 0; i < nb; i++) {
        ثابت عائم d   = GGML_FP16_منزل_FP32(src[i].d);
        ثابت عائم min = GGML_FP16_منزل_FP32(src[i].dmin);
        ثابت uint8_t * sc = src[i].scales;

        عائم * y = dst_ptr + i * QK_K;

        لكل (صحيح قدم = 0; قدم < 4; ++قدم) {
            uint8_t sc1, m1_u, sc2, m2_u;
            إذا (قدم < 2) {
                sc1  = sc[2 * قدم + 0] & 63;
                m1_u = sc[2 * قدم + 4] & 63;
                sc2  = sc[2 * قدم + 1] & 63;
                m2_u = sc[2 * قدم + 5] & 63;
            } وإلا {
                ثابت صحيح s = قدم - 2;
                sc1  = (sc[8 + 2 * s + 0] & 0xF) | ((sc[2 * s + 0] >> 6) << 4);
                m1_u = (sc[8 + 2 * s + 0] >>  4) | ((sc[4 + 2 * s + 0] >> 6) << 4);
                sc2  = (sc[8 + 2 * s + 1] & 0xF) | ((sc[2 * s + 1] >> 6) << 4);
                m2_u = (sc[8 + 2 * s + 1] >>  4) | ((sc[4 + 2 * s + 1] >> 6) << 4);
            }

            HVX_ويڪٽر vq = *(ثابت HVX_UVector *) (src[i].qs + قدم * 32);
            HVX_ويڪٽر v_lo = Q6_V_vand_VV(vq, Q6_Vb_vsplat_R(0x0F));
            HVX_ويڪٽر v_hi = Q6_Vub_vlsr_VubR(vq, 4);

            HVX_ويڪٽر_Pair p16_lo = Q6_Wuh_vunpack_Vub(v_lo);
            HVX_ويڪٽر_Pair p32_lo = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16_lo));
            HVX_ويڪٽر     v_i32_lo = Q6_V_lo_W(p32_lo);
            HVX_ويڪٽر     v_f32_lo = Q6_Vsf_equals_Vw(v_i32_lo);

            HVX_ويڪٽر vd1 = hvx_vec_splat_f32(d * sc1);
            HVX_ويڪٽر vm1 = hvx_vec_splat_f32(min * m1_u);
            HVX_ويڪٽر v_out1 = hvx_vec_sub_f32_f32(hvx_vec_mul_f32_f32(v_f32_lo, vd1), vm1);
            hvx_vmem(y + 0) = v_out1;

            HVX_ويڪٽر_Pair p16_hi = Q6_Wuh_vunpack_Vub(v_hi);
            HVX_ويڪٽر_Pair p32_hi = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16_hi));
            HVX_ويڪٽر     v_i32_hi = Q6_V_lo_W(p32_hi);
            HVX_ويڪٽر     v_f32_hi = Q6_Vsf_equals_Vw(v_i32_hi);

            HVX_ويڪٽر vd2 = hvx_vec_splat_f32(d * sc2);
            HVX_ويڪٽر vm2 = hvx_vec_splat_f32(min * m2_u);
            HVX_ويڪٽر v_out2 = hvx_vec_sub_f32_f32(hvx_vec_mul_f32_f32(v_f32_hi, vd2), vm2);
            hvx_vmem(y + 32) = v_out2;

            y += 64;
        }
    }
}

ساكن مضمن فراغ hvx_dequantize_صف_flat_q5_k_f32(عائم * محدود dst_ptr, ثابت فراغ * محدود src_ptr, uint32_t n) {
    ثابت صحيح nb = n / QK_K;
    ثابت بلاڪ_q5_K * src = (ثابت بلاڪ_q5_K *) src_ptr;

    لكل (صحيح i = 0; i < nb; i++) {
        ثابت عائم d   = GGML_FP16_منزل_FP32(src[i].d);
        ثابت عائم min = GGML_FP16_منزل_FP32(src[i].dmin);
        ثابت uint8_t * sc = src[i].scales;

        HVX_ويڪٽر v_qh = *(ثابت HVX_UVector *) src[i].qh;
        عائم * y = dst_ptr + i * QK_K;

        لكل (صحيح قدم = 0; قدم < 4; ++قدم) {
            uint8_t sc1, m1_u, sc2, m2_u;
            إذا (قدم < 2) {
                sc1  = sc[2 * قدم + 0] & 63;
                m1_u = sc[2 * قدم + 4] & 63;
                sc2  = sc[2 * قدم + 1] & 63;
                m2_u = sc[2 * قدم + 5] & 63;
            } وإلا {
                ثابت صحيح s = قدم - 2;
                sc1  = (sc[8 + 2 * s + 0] & 0xF) | ((sc[2 * s + 0] >> 6) << 4);
                m1_u = (sc[8 + 2 * s + 0] >>  4) | ((sc[4 + 2 * s + 0] >> 6) << 4);
                sc2  = (sc[8 + 2 * s + 1] & 0xF) | ((sc[2 * s + 1] >> 6) << 4);
                m2_u = (sc[8 + 2 * s + 1] >>  4) | ((sc[4 + 2 * s + 1] >> 6) << 4);
            }

            HVX_ويڪٽر v_ql = *(ثابت HVX_UVector *) (src[i].qs + قدم * 32);
            HVX_ويڪٽر v_lo = Q6_V_vand_VV(v_ql, Q6_Vb_vsplat_R(0x0F));
            HVX_ويڪٽر v_hi = Q6_Vub_vlsr_VubR(v_ql, 4);

            HVX_ويڪٽر v_h1 = Q6_Vw_vasl_VwR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v_qh, 2 * قدم), Q6_Vb_vsplat_R(1)), 4);
            HVX_ويڪٽر v_val1 = Q6_V_vor_VV(v_lo, v_h1);

            HVX_ويڪٽر_Pair p16_1 = Q6_Wuh_vunpack_Vub(v_val1);
            HVX_ويڪٽر_Pair p32_1 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16_1));
            HVX_ويڪٽر     v_i32_1 = Q6_V_lo_W(p32_1);
            HVX_ويڪٽر     v_f32_1 = Q6_Vsf_equals_Vw(v_i32_1);

            HVX_ويڪٽر vd1 = hvx_vec_splat_f32(d * sc1);
            HVX_ويڪٽر vm1 = hvx_vec_splat_f32(min * m1_u);
            HVX_ويڪٽر v_out1 = hvx_vec_sub_f32_f32(hvx_vec_mul_f32_f32(v_f32_1, vd1), vm1);
            hvx_vmem(y + 0) = v_out1;

            HVX_ويڪٽر v_h2 = Q6_Vw_vasl_VwR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v_qh, 2 * قدم + 1), Q6_Vb_vsplat_R(1)), 4);
            HVX_ويڪٽر v_val2 = Q6_V_vor_VV(v_hi, v_h2);

            HVX_ويڪٽر_Pair p16_2 = Q6_Wuh_vunpack_Vub(v_val2);
            HVX_ويڪٽر_Pair p32_2 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16_2));
            HVX_ويڪٽر     v_i32_2 = Q6_V_lo_W(p32_2);
            HVX_ويڪٽر     v_f32_2 = Q6_Vsf_equals_Vw(v_i32_2);

            HVX_ويڪٽر vd2 = hvx_vec_splat_f32(d * sc2);
            HVX_ويڪٽر vm2 = hvx_vec_splat_f32(min * m2_u);
            HVX_ويڪٽر v_out2 = hvx_vec_sub_f32_f32(hvx_vec_mul_f32_f32(v_f32_2, vd2), vm2);
            hvx_vmem(y + 32) = v_out2;

            y += 64;
        }
    }
}

ساكن مضمن فراغ hvx_dequantize_صف_flat_q6_k_f32(عائم * محدود dst_ptr, ثابت فراغ * محدود src_ptr, uint32_t n) {
    ثابت صحيح nb = n / QK_K;
    ثابت بلاڪ_q6_K * src = (ثابت بلاڪ_q6_K *) src_ptr;

    ثابت HVX_ويڪٽر_Pred q64 = Q6_Q_vsetq_R(64);
    ثابت HVX_ويڪٽر v32_b   = Q6_Vb_vsplat_R(32);
    ثابت HVX_ويڪٽر نقاب_0f = Q6_Vb_vsplat_R(0x0F);
    ثابت HVX_ويڪٽر نقاب_03 = Q6_Vb_vsplat_R(0x03);

    لكل (صحيح i = 0; i < nb; i++) {
        ثابت عائم d = GGML_FP16_منزل_FP32(src[i].d);

        ثابت uint8_t * محدود ql = src[i].ql;
        ثابت uint8_t * محدود qh = src[i].qh;
        ثابت int8_t  * محدود sc = src[i].scales;

        عائم * y = dst_ptr + i * QK_K;

        لكل (صحيح half = 0; half < 2; ++half) {
            عائم s[8];
            لكل (صحيح j = 0; j < 8; ++j) {
                s[j] = d * sc[j];
            }

            HVX_ويڪٽر vscale1 = Q6_V_vmux_QVV(q64, hvx_vec_splat_f32(s[0]), hvx_vec_splat_f32(s[1]));
            HVX_ويڪٽر vscale2 = Q6_V_vmux_QVV(q64, hvx_vec_splat_f32(s[2]), hvx_vec_splat_f32(s[3]));
            HVX_ويڪٽر vscale3 = Q6_V_vmux_QVV(q64, hvx_vec_splat_f32(s[4]), hvx_vec_splat_f32(s[5]));
            HVX_ويڪٽر vscale4 = Q6_V_vmux_QVV(q64, hvx_vec_splat_f32(s[6]), hvx_vec_splat_f32(s[7]));

            HVX_ويڪٽر v_ql0 = *(ثابت HVX_UVector *) (ql);
            HVX_ويڪٽر v_ql1 = *(ثابت HVX_UVector *) (ql + 32);
            HVX_ويڪٽر v_qh  = *(ثابت HVX_UVector *) (qh);

            HVX_ويڪٽر ql0_lo = Q6_V_vand_VV(v_ql0, نقاب_0f);
            HVX_ويڪٽر qh_0   = Q6_Vw_vasl_VwR(Q6_V_vand_VV(v_qh, نقاب_03), 4);
            HVX_ويڪٽر v_q1   = Q6_Vb_vsub_VbVb(Q6_V_vor_VV(ql0_lo, qh_0), v32_b);

            HVX_ويڪٽر ql1_lo = Q6_V_vand_VV(v_ql1, نقاب_0f);
            HVX_ويڪٽر qh_1   = Q6_Vw_vasl_VwR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v_qh, 2), نقاب_03), 4);
            HVX_ويڪٽر v_q2   = Q6_Vb_vsub_VbVb(Q6_V_vor_VV(ql1_lo, qh_1), v32_b);

            HVX_ويڪٽر ql0_hi = Q6_Vub_vlsr_VubR(v_ql0, 4);
            HVX_ويڪٽر qh_2   = Q6_Vw_vasl_VwR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v_qh, 4), نقاب_03), 4);
            HVX_ويڪٽر v_q3   = Q6_Vb_vsub_VbVb(Q6_V_vor_VV(ql0_hi, qh_2), v32_b);

            HVX_ويڪٽر ql1_hi = Q6_Vub_vlsr_VubR(v_ql1, 4);
            HVX_ويڪٽر qh_3   = Q6_Vw_vasl_VwR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v_qh, 6), نقاب_03), 4);
            HVX_ويڪٽر v_q4   = Q6_Vb_vsub_VbVb(Q6_V_vor_VV(ql1_hi, qh_3), v32_b);

            HVX_ويڪٽر_Pair p16_1 = Q6_Wh_vunpack_Vb(v_q1);
            HVX_ويڪٽر_Pair p32_1 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16_1));
            HVX_ويڪٽر     v_f32_1 = Q6_Vsf_equals_Vw(Q6_V_lo_W(p32_1));
            hvx_vmem(y + 0) = hvx_vec_mul_f32_f32(v_f32_1, vscale1);

            HVX_ويڪٽر_Pair p16_2 = Q6_Wh_vunpack_Vb(v_q2);
            HVX_ويڪٽر_Pair p32_2 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16_2));
            HVX_ويڪٽر     v_f32_2 = Q6_Vsf_equals_Vw(Q6_V_lo_W(p32_2));
            hvx_vmem(y + 32) = hvx_vec_mul_f32_f32(v_f32_2, vscale2);

            HVX_ويڪٽر_Pair p16_3 = Q6_Wh_vunpack_Vb(v_q3);
            HVX_ويڪٽر_Pair p32_3 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16_3));
            HVX_ويڪٽر     v_f32_3 = Q6_Vsf_equals_Vw(Q6_V_lo_W(p32_3));
            hvx_vmem(y + 64) = hvx_vec_mul_f32_f32(v_f32_3, vscale3);

            HVX_ويڪٽر_Pair p16_4 = Q6_Wh_vunpack_Vb(v_q4);
            HVX_ويڪٽر_Pair p32_4 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16_4));
            HVX_ويڪٽر     v_f32_4 = Q6_Vsf_equals_Vw(Q6_V_lo_W(p32_4));
            hvx_vmem(y + 96) = hvx_vec_mul_f32_f32(v_f32_4, vscale4);

            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

// bytes row, row + 32, row + 64 and row + 96 of v, to bytes 0..3
ساكن مضمن HVX_ويڪٽر hvx_dequantize_gather4(HVX_ويڪٽر v, uint32_t صف) {
    v = Q6_V_vror_VR(v, صف);
    HVX_ويڪٽر packed = Q6_V_vmux_QVV(Q6_Q_vsetq_R(1), v, Q6_V_vror_VR(v, 31));
    packed = Q6_V_vmux_QVV(Q6_Q_vsetq_R(2), packed, Q6_V_vror_VR(v, 62));
    أرجع Q6_V_vmux_QVV(Q6_Q_vsetq_R(3), packed, Q6_V_vror_VR(v, 93));
}

ساكن مضمن فراغ hvx_dequantize_صف_tiled_q8_0_f32(عائم * dst_spad, ثابت uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t صف) {
    ثابت HVX_ويڪٽر_Pred first2 = Q6_Q_vsetq_R(2);
    ثابت HVX_ويڪٽر_Pred first4 = Q6_Q_vsetq_R(4);

    لكل (uint32_t k_tile = 0; k_tile < n_k_tiles; ++k_tile) {
        ثابت uint8_t * tile = src_spad + k_tile * tile_stride;
        عائم * dst = dst_spad + k_tile * HTP_TENSOR_TILE_N_کانن_ڳڻپ;

        HVX_ويڪٽر vq = Q6_V_vzero();
        لكل (صحيح گروهه = 7; گروهه >= 0; --گروهه) {
            ثابت HVX_ويڪٽر v = Q6_V_vror_VR(hvx_vmem(tile + گروهه * VLEN), 2 * صف);
            ثابت HVX_ويڪٽر packed = Q6_V_vmux_QVV(first2, v, Q6_V_vror_VR(v, 62));
            vq = Q6_V_vmux_QVV(first4, packed, Q6_V_vror_VR(vq, VLEN - 4));
        }

        ثابت HVX_ويڪٽر scales = hvx_vmem(tile + 1024);
        ثابت HVX_ويڪٽر scale_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 2 * صف));
        ثابت HVX_ويڪٽر scale = Q6_V_lo_W(hvx_vec_f16_منزل_f32(scale_hf));
        ثابت HVX_ويڪٽر_Pair p16 = Q6_Wh_vunpack_Vb(vq);
        ثابت HVX_ويڪٽر_Pair p32 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16));
        HVX_ويڪٽر قدرون = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_lo_W(p32)), scale);
        *(HVX_ويڪٽر *) dst = قدرون;
    }
}

ساكن مضمن فراغ hvx_dequantize_صف_tiled_q4_0_f32(عائم * dst_spad, ثابت uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t صف) {
    ثابت HVX_ويڪٽر_Pred first4 = Q6_Q_vsetq_R(4);

    لكل (uint32_t k_tile = 0; k_tile < n_k_tiles; ++k_tile) {
        ثابت uint8_t * tile = src_spad + k_tile * tile_stride;
        عائم * dst = dst_spad + k_tile * HTP_TENSOR_TILE_N_کانن_ڳڻپ;

        HVX_ويڪٽر vq = Q6_V_vzero();
        لكل (صحيح گروهه = 3; گروهه >= 0; --گروهه) {
            vq = Q6_V_vmux_QVV(first4, hvx_dequantize_gather4(hvx_vmem(tile + گروهه * VLEN), صف), Q6_V_vror_VR(vq, VLEN - 4));
        }

        ثابت HVX_ويڪٽر lo = Q6_V_vand_VV(vq, Q6_Vb_vsplat_R(0x0F));
        ثابت HVX_ويڪٽر hi = Q6_Vub_vlsr_VubR(vq, 4);
        vq = Q6_V_lo_W(Q6_W_vshuff_VVR(hi, lo, -1));
        vq = Q6_Vb_vsub_VbVb(vq, Q6_Vb_vsplat_R(8));

        ثابت HVX_ويڪٽر scales = hvx_vmem(tile + 512);
        ثابت HVX_ويڪٽر scale_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 2 * صف));
        ثابت HVX_ويڪٽر scale = Q6_V_lo_W(hvx_vec_f16_منزل_f32(scale_hf));
        ثابت HVX_ويڪٽر_Pair p16 = Q6_Wh_vunpack_Vb(vq);
        ثابت HVX_ويڪٽر_Pair p32 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16));
        HVX_ويڪٽر قدرون = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_lo_W(p32)), scale);
        *(HVX_ويڪٽر *) dst = قدرون;
    }
}

ساكن مضمن فراغ hvx_dequantize_صف_tiled_q4_k_f32(عائم * dst_spad, ثابت uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t صف) {
    ثابت HVX_ويڪٽر_Pred first4 = Q6_Q_vsetq_R(4);

    لكل (uint32_t k_tile = 0; k_tile < n_k_tiles; ++k_tile) {
        ثابت uint8_t * tile = src_spad + k_tile * tile_stride;
        عائم * dst = dst_spad + k_tile * HTP_TENSOR_TILE_N_کانن_ڳڻپ;

        HVX_ويڪٽر vq = Q6_V_vzero();
        لكل (صحيح گروهه = 3; گروهه >= 0; --گروهه) {
            vq = Q6_V_vmux_QVV(first4, hvx_dequantize_gather4(hvx_vmem(tile + گروهه * VLEN), صف), Q6_V_vror_VR(vq, VLEN - 4));
        }

        ثابت HVX_ويڪٽر lo = Q6_V_vand_VV(vq, Q6_Vb_vsplat_R(0x0F));
        ثابت HVX_ويڪٽر hi = Q6_Vub_vlsr_VubR(vq, 4);
        vq = Q6_V_lo_W(Q6_W_vshuff_VVR(hi, lo, -1));

        ثابت HVX_ويڪٽر scales = hvx_vmem(tile + 512);
        ثابت HVX_ويڪٽر scale_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 4 * صف));
        ثابت HVX_ويڪٽر scale = Q6_V_lo_W(hvx_vec_f16_منزل_f32(scale_hf));
        ثابت HVX_ويڪٽر وٿي_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 4 * صف + 2));
        ثابت HVX_ويڪٽر وٿي = Q6_V_lo_W(hvx_vec_f16_منزل_f32(وٿي_hf));

        ثابت HVX_ويڪٽر_Pair p16 = Q6_Wh_vunpack_Vb(vq);
        ثابت HVX_ويڪٽر_Pair p32 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16));
        HVX_ويڪٽر قدرون = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_lo_W(p32)), scale);
        *(HVX_ويڪٽر *) dst = hvx_vec_جوڙيو_f32_f32(قدرون, وٿي);
    }
}

ساكن مضمن فراغ hvx_dequantize_صف_tiled_q5_k_f32(عائم * dst_spad, ثابت uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t صف) {
    ثابت HVX_ويڪٽر_Pred first4 = Q6_Q_vsetq_R(4);
    ثابت HVX_ويڪٽر one = Q6_Vb_vsplat_R(1);

    لكل (uint32_t k_tile = 0; k_tile < n_k_tiles; ++k_tile) {
        ثابت uint8_t * tile = src_spad + k_tile * tile_stride;
        عائم * dst = dst_spad + k_tile * HTP_TENSOR_TILE_N_کانن_ڳڻپ;

        ثابت HVX_ويڪٽر p5 = hvx_dequantize_gather4(hvx_vmem(tile + 640), صف);
        HVX_ويڪٽر vq = Q6_V_vzero();
        HVX_ويڪٽر vh = Q6_V_vzero();
        لكل (صحيح گروهه = 3; گروهه >= 0; --گروهه) {
            vq = Q6_V_vmux_QVV(first4, hvx_dequantize_gather4(hvx_vmem(tile + گروهه * VLEN), صف), Q6_V_vror_VR(vq, VLEN - 4));
            vh = Q6_V_vmux_QVV(first4, Q6_Vub_vlsr_VubR(p5, 2 * گروهه), Q6_V_vror_VR(vh, VLEN - 4));
        }

        ثابت HVX_ويڪٽر lo = Q6_V_vor_VV(Q6_V_vand_VV(vq, Q6_Vb_vsplat_R(0x0F)), Q6_Vw_vasl_VwR(Q6_V_vand_VV(vh, one), 4));
        ثابت HVX_ويڪٽر hi = Q6_V_vor_VV(Q6_Vub_vlsr_VubR(vq, 4), Q6_Vw_vasl_VwR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(vh, 1), one), 4));
        vq = Q6_V_lo_W(Q6_W_vshuff_VVR(hi, lo, -1));

        ثابت HVX_ويڪٽر scales = hvx_vmem(tile + 512);
        ثابت HVX_ويڪٽر scale_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 4 * صف));
        ثابت HVX_ويڪٽر scale = Q6_V_lo_W(hvx_vec_f16_منزل_f32(scale_hf));
        ثابت HVX_ويڪٽر وٿي_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 4 * صف + 2));
        ثابت HVX_ويڪٽر وٿي = Q6_V_lo_W(hvx_vec_f16_منزل_f32(وٿي_hf));

        ثابت HVX_ويڪٽر_Pair p16 = Q6_Wh_vunpack_Vb(vq);
        ثابت HVX_ويڪٽر_Pair p32 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16));
        HVX_ويڪٽر قدرون = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_lo_W(p32)), scale);
        *(HVX_ويڪٽر *) dst = hvx_vec_جوڙيو_f32_f32(قدرون, وٿي);
    }
}

ساكن مضمن فراغ hvx_dequantize_صف_tiled_q6_k_f32(عائم * dst_spad, ثابت uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t صف) {
    ثابت HVX_ويڪٽر_Pred first4 = Q6_Q_vsetq_R(4);
    ثابت HVX_ويڪٽر_Pred first16 = Q6_Q_vsetq_R(16 * حجم(عائم));
    ثابت HVX_ويڪٽر نقاب_0f = Q6_Vb_vsplat_R(0x0F);
    ثابت HVX_ويڪٽر نقاب_03 = Q6_Vb_vsplat_R(0x03);

    لكل (uint32_t k_tile = 0; k_tile < n_k_tiles; ++k_tile) {
        ثابت uint8_t * tile = src_spad + k_tile * tile_stride;
        عائم * dst = dst_spad + k_tile * HTP_TENSOR_TILE_N_کانن_ڳڻپ;

        HVX_ويڪٽر vq = Q6_V_vzero();
        لكل (صحيح گروهه = 7; گروهه >= 0; --گروهه) {
            ثابت HVX_ويڪٽر lo_plane = Q6_V_vror_VR(hvx_vmem(tile + (گروهه >> 1) * VLEN), 4 * صف);
            ثابت HVX_ويڪٽر hi_plane = Q6_V_vror_VR(hvx_vmem(tile + 512 + (گروهه >> 2) * VLEN), 4 * صف);
            ثابت HVX_ويڪٽر lo = (گروهه & 1) ? Q6_Vub_vlsr_VubR(lo_plane, 4) : Q6_V_vand_VV(lo_plane, نقاب_0f);
            ثابت HVX_ويڪٽر hi = Q6_Vub_vlsr_VubR(hi_plane, 2 * (گروهه & 3));
            ثابت HVX_ويڪٽر packed = Q6_V_vor_VV(lo, Q6_Vw_vasl_VwR(Q6_V_vand_VV(hi, نقاب_03), 4));
            vq = Q6_V_vmux_QVV(first4, packed, Q6_V_vror_VR(vq, VLEN - 4));
        }

        ثابت HVX_ويڪٽر scales = hvx_vmem(tile + 768);
        ثابت HVX_ويڪٽر scale_lo_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 2 * صف));
        ثابت HVX_ويڪٽر scale_hi_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 64 + 2 * صف));
        ثابت HVX_ويڪٽر scale_lo = Q6_V_lo_W(hvx_vec_f16_منزل_f32(scale_lo_hf));
        ثابت HVX_ويڪٽر scale_hi = Q6_V_lo_W(hvx_vec_f16_منزل_f32(scale_hi_hf));
        ثابت HVX_ويڪٽر scale = Q6_V_vmux_QVV(first16, scale_lo, scale_hi);
        ثابت HVX_ويڪٽر_Pair p16 = Q6_Wh_vunpack_Vb(Q6_Vb_vsub_VbVb(vq, Q6_Vb_vsplat_R(32)));
        ثابت HVX_ويڪٽر_Pair p32 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16));
        *(HVX_ويڪٽر *) dst = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_lo_W(p32)), scale);
    }
}

ساكن مضمن HVX_ويڪٽر hvx_dequantize_tiled_f16_gather4(ثابت uint8_t * p, uint32_t r_byte) {
    HVX_ويڪٽر w0  = Q6_V_vror_VR(hvx_vmem(p + 0 * VLEN), r_byte);
    HVX_ويڪٽر w1  = Q6_V_vror_VR(hvx_vmem(p + 1 * VLEN), r_byte);
    HVX_ويڪٽر w01 = Q6_V_lo_W(Q6_W_vshuff_VVR(w1, w0, 4));
    HVX_ويڪٽر w2  = Q6_V_vror_VR(hvx_vmem(p + 2 * VLEN), r_byte);
    HVX_ويڪٽر w3  = Q6_V_vror_VR(hvx_vmem(p + 3 * VLEN), r_byte);
    HVX_ويڪٽر w23 = Q6_V_lo_W(Q6_W_vshuff_VVR(w3, w2, 4));
    أرجع Q6_V_lo_W(Q6_W_vshuff_VVR(w23, w01, 8));
}

ساكن مضمن فراغ hvx_dequantize_صف_tiled_f16_f32(عائم * dst_spad, ثابت uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t صف) {
    ثابت uint32_t r_byte = 4 * صف;

    لكل (uint32_t k_tile = 0; k_tile < n_k_tiles; ++k_tile) {
        ثابت uint8_t * tile = src_spad + k_tile * tile_stride;
        عائم * dst = dst_spad + k_tile * HTP_TENSOR_TILE_N_کانن_ڳڻپ;

        HVX_ويڪٽر g0 = hvx_dequantize_tiled_f16_gather4(tile + 0 * VLEN, r_byte);
        HVX_ويڪٽر g1 = hvx_dequantize_tiled_f16_gather4(tile + 4 * VLEN, r_byte);
        HVX_ويڪٽر g01 = Q6_V_lo_W(Q6_W_vshuff_VVR(g1, g0, 16));

        HVX_ويڪٽر g2 = hvx_dequantize_tiled_f16_gather4(tile + 8 * VLEN, r_byte);
        HVX_ويڪٽر g3 = hvx_dequantize_tiled_f16_gather4(tile + 12 * VLEN, r_byte);
        HVX_ويڪٽر g23 = Q6_V_lo_W(Q6_W_vshuff_VVR(g3, g2, 16));

        HVX_ويڪٽر g = Q6_V_lo_W(Q6_W_vshuff_VVR(g23, g01, 32));
        *(HVX_ويڪٽر *) dst = Q6_V_lo_W(hvx_vec_f16_منزل_f32(g));
    }
}

ساكن مضمن فراغ hvx_dequantize_صف_tiled_q1_0_f32(عائم * dst_spad, ثابت uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t صف) {
    ساكن ثابت uint32_t __attribute__((aligned(128))) ٻٽو_انگ_نقاب[32] = {
        1u <<  0, 1u <<  8, 1u << 16, 1u << 24,
        1u <<  1, 1u <<  9, 1u << 17, 1u << 25,
        1u <<  2, 1u << 10, 1u << 18, 1u << 26,
        1u <<  3, 1u << 11, 1u << 19, 1u << 27,
        1u <<  4, 1u << 12, 1u << 20, 1u << 28,
        1u <<  5, 1u << 13, 1u << 21, 1u << 29,
        1u <<  6, 1u << 14, 1u << 22, 1u << 30,
        1u <<  7, 1u << 15, 1u << 23, 1u << 31,
    };
    ثابت HVX_ويڪٽر v_ٻٽو_انگ_نقاب = *(ثابت HVX_ويڪٽر *) ٻٽو_انگ_نقاب;
    ثابت HVX_ويڪٽر v_zero     = Q6_V_vzero();
    ثابت HVX_ويڪٽر v_sign_ٻٽو_انگ = Q6_V_vsplat_R(0x80000000);

    لكل (uint32_t k_tile = 0; k_tile < n_k_tiles; ++k_tile) {
        ثابت uint8_t * tile = src_spad + k_tile * tile_stride;
        عائم * dst = dst_spad + k_tile * HTP_TENSOR_TILE_K_Q1_0;

        ثابت HVX_ويڪٽر scales   = hvx_vmem(tile + 512);
        ثابت HVX_ويڪٽر scale_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 2 * صف));
        ثابت HVX_ويڪٽر vd       = Q6_V_lo_W(hvx_vec_f16_منزل_f32(scale_hf));
        ثابت HVX_ويڪٽر neg_vd   = Q6_V_vxor_VV(vd, v_sign_ٻٽو_انگ);

        #هدايت unroll
        لكل (صحيح q = 0; q < 4; ++q) {
            ثابت HVX_ويڪٽر v_بٽ = hvx_vec_repl_u32(Q6_V_vror_VR(hvx_vmem(tile + q * VLEN), 4 * صف));
            ثابت HVX_ويڪٽر_Pred q_zero = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(v_بٽ, v_ٻٽو_انگ_نقاب), v_zero);
            hvx_vmem(dst + q * 32) = Q6_V_vmux_QVV(q_zero, neg_vd, vd);
        }
    }
}

ساكن مضمن فراغ hvx_dequantize_صف_tiled_q2_0_f32(عائم * dst_spad, ثابت uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t صف) {
    ثابت HVX_ويڪٽر نقاب_03 = Q6_Vb_vsplat_R(0x03);
    ثابت HVX_ويڪٽر v_one   = Q6_Vb_vsplat_R(1);

    لكل (uint32_t k_tile = 0; k_tile < n_k_tiles; ++k_tile) {
        ثابت uint8_t * tile = src_spad + k_tile * tile_stride;
        عائم * dst = dst_spad + k_tile * HTP_TENSOR_TILE_K_Q2_0;

        ثابت HVX_ويڪٽر v_packed = hvx_dequantize_tiled_f16_gather4(tile, 4 * صف);
        ثابت HVX_ويڪٽر p0 = Q6_V_vand_VV(v_packed, نقاب_03);
        ثابت HVX_ويڪٽر p1 = Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v_packed, 2), نقاب_03);
        ثابت HVX_ويڪٽر p2 = Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v_packed, 4), نقاب_03);
        ثابت HVX_ويڪٽر p3 = Q6_Vub_vlsr_VubR(v_packed, 6);

        ثابت HVX_ويڪٽر p01 = Q6_V_lo_W(Q6_W_vshuff_VVR(p1, p0, -4));
        ثابت HVX_ويڪٽر p23 = Q6_V_lo_W(Q6_W_vshuff_VVR(p3, p2, -4));
        ثابت HVX_ويڪٽر vq  = Q6_Vb_vsub_VbVb(Q6_V_lo_W(Q6_W_vshuff_VVR(p23, p01, -8)), v_one);

        ثابت HVX_ويڪٽر scales   = hvx_vmem(tile + 512);
        ثابت HVX_ويڪٽر scale_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 2 * صف));
        ثابت HVX_ويڪٽر scale    = Q6_V_lo_W(hvx_vec_f16_منزل_f32(scale_hf));

        ثابت HVX_ويڪٽر_Pair p16 = Q6_Wh_vunpack_Vb(vq);
        ثابت HVX_ويڪٽر_Pair p32 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16));
        hvx_vmem(dst + 0)  = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_lo_W(p32)), scale);
        hvx_vmem(dst + 32) = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_hi_W(p32)), scale);
    }
}

#جيڪڏهن_پورو // HVX_QUANT_H
