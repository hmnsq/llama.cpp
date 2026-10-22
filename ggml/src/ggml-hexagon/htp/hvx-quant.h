#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HVX_QUANT_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_QUANT_H

#ਅਨ੍ਤਰ੍ਭਾਵ <math.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>

#ਅਨ੍ਤਰ੍ਭਾਵ "htp-tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-arith.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-base.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-reduce.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-repl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-utils.h"

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_COMMON_DECL_C
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_DECL_C
#ਜੇ_ਅੰਤ
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_quantize_ਪੰਕਤੀ_q8_0_f32(ਖਾਲੀ * ਸੀਮਿਤ dst_ptr, ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ src_ptr, ਪੂਰਨ n) {
    ਅਚਲ ਪੂਰਨ nb = n / QK8_0;
    ਖੰਡ_q8_0 * dst = (ਖੰਡ_q8_0 *) dst_ptr;
    HVX_ਸਦਿਸ਼ zero = Q6_V_vzero();

    ਪੂਰਨ i = 0;
    ਲਈ (; i + 3 < nb; i += 4) {
        HVX_ਸਦਿਸ਼ * vx = (HVX_ਸਦਿਸ਼ *) (src_ptr + i * QK8_0);

        HVX_ਸਦਿਸ਼ vmax0_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[0]));
        HVX_ਸਦਿਸ਼ vmax1_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[1]));
        HVX_ਸਦਿਸ਼ vmax2_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[2]));
        HVX_ਸਦਿਸ਼ vmax3_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[3]));

        HVX_ਸਦਿਸ਼ vx0_qf = Q6_Vqf32_vsub_VsfVsf(vx[0], zero);
        HVX_ਸਦਿਸ਼ vx1_qf = Q6_Vqf32_vsub_VsfVsf(vx[1], zero);
        HVX_ਸਦਿਸ਼ vx2_qf = Q6_Vqf32_vsub_VsfVsf(vx[2], zero);
        HVX_ਸਦਿਸ਼ vx3_qf = Q6_Vqf32_vsub_VsfVsf(vx[3], zero);

        HVX_ਸਦਿਸ਼ vmax0_qf = Q6_Vqf32_vsub_VsfVsf(vmax0_sf, zero);
        HVX_ਸਦਿਸ਼ vmax1_qf = Q6_Vqf32_vsub_VsfVsf(vmax1_sf, zero);
        HVX_ਸਦਿਸ਼ vmax2_qf = Q6_Vqf32_vsub_VsfVsf(vmax2_sf, zero);
        HVX_ਸਦਿਸ਼ vmax3_qf = Q6_Vqf32_vsub_VsfVsf(vmax3_sf, zero);

        HVX_ਸਦਿਸ਼ vmax01_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vmax1_qf, vmax0_qf)));
        HVX_ਸਦਿਸ਼ vmax23_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vmax3_qf, vmax2_qf)));

        HVX_ਸਦਿਸ਼ vx01_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx1_qf, vx0_qf)));
        HVX_ਸਦਿਸ਼ vx23_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx3_qf, vx2_qf)));

        HVX_ਸਦਿਸ਼ vd01_qf16 = Q6_Vqf16_vmpy_VhfVhf(vmax01_hf, Q6_Vh_vsplat_R(0x2008));  // 1.0 / 127.0
        HVX_ਸਦਿਸ਼ vd23_qf16 = Q6_Vqf16_vmpy_VhfVhf(vmax23_hf, Q6_Vh_vsplat_R(0x2008));  // 1.0 / 127.0
        HVX_ਸਦਿਸ਼ vd01_hf   = Q6_Vhf_equals_Vqf16(vd01_qf16);
        HVX_ਸਦਿਸ਼ vd23_hf   = Q6_Vhf_equals_Vqf16(vd23_qf16);

        HVX_ਸਦਿਸ਼ vd01_inv_hf = hvx_vec_inverse_f16(vd01_hf);
        HVX_ਸਦਿਸ਼ vd23_inv_hf = hvx_vec_inverse_f16(vd23_hf);
        vx01_hf              = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx01_hf, vd01_inv_hf));
        vx23_hf              = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx23_hf, vd23_inv_hf));

        HVX_ਸਦਿਸ਼ vx01_i16 = hvx_vec_i16_from_hf_rnd_sat(vx01_hf);
        HVX_ਸਦਿਸ਼ vx23_i16 = hvx_vec_i16_from_hf_rnd_sat(vx23_hf);
        HVX_ਸਦਿਸ਼ vx_i8    = Q6_Vb_vpack_VhVh_sat(vx23_i16, vx01_i16);

        hvx_vec_store_u(&dst[i + 0].d, 2, vd01_hf);
        hvx_vec_store_u(dst[i + 0].qs, 32, vx_i8);

        hvx_vec_store_u(&dst[i + 1].d, 2, Q6_V_vror_VR(vd01_hf, 64));
        hvx_vec_store_u(dst[i + 1].qs, 32, Q6_V_vror_VR(vx_i8, 32));

        hvx_vec_store_u(&dst[i + 2].d, 2, vd23_hf);
        hvx_vec_store_u(dst[i + 2].qs, 32, Q6_V_vror_VR(vx_i8, 64));

        hvx_vec_store_u(&dst[i + 3].d, 2, Q6_V_vror_VR(vd23_hf, 64));
        hvx_vec_store_u(dst[i + 3].qs, 32, Q6_V_vror_VR(vx_i8, 96));
    }

    ਲਈ (; i < nb; i++) {
        ਅਚਲ ਦਸ਼ਮਲਵ * ਖੰਡ_src = src_ptr + i * QK8_0;
        HVX_ਸਦਿਸ਼ vx = *(ਅਚਲ HVX_UVector *) ਖੰਡ_src;
        HVX_ਸਦਿਸ਼ v_abs = hvx_vec_abs_f32(vx);
        HVX_ਸਦਿਸ਼ v_max = hvx_vec_reduce_max_f32(v_abs);
        ਦਸ਼ਮਲਵ amax = hvx_vec_get_f32(v_max);

        ਅਚਲ ਦਸ਼ਮਲਵ d = amax / 127.0f;
        ਅਚਲ ਦਸ਼ਮਲਵ id = d ? (1.0f / d) : 0.0f;
        dst[i].d = GGML_FP32_ਮੰਜ਼ਿਲ_FP16(d);

        HVX_ਸਦਿਸ਼ vid = hvx_vec_splat_f32(id);
        HVX_ਸਦਿਸ਼ v_scaled = hvx_vec_mul_f32_f32(vx, vid);
        HVX_ਸਦਿਸ਼ v_scaled_qf = Q6_Vqf32_vsub_VsfVsf(v_scaled, zero);
        HVX_ਸਦਿਸ਼ v_scaled_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(zero, v_scaled_qf)));
        HVX_ਸਦਿਸ਼ v_i16 = hvx_vec_i16_from_hf_rnd_sat(v_scaled_hf);
        HVX_ਸਦਿਸ਼ v_i8  = Q6_Vb_vpack_VhVh_sat(zero, v_i16);

        hvx_vec_store_u(dst[i].qs, 32, v_i8);
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_dequantize_ਪੰਕਤੀ_flat_q8_0_f32(ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst_ptr, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src_ptr, uint32_t n) {
    ਅਚਲ ਪੂਰਨ nb = n / QK8_0;
    ਅਚਲ ਖੰਡ_q8_0 * src = (ਅਚਲ ਖੰਡ_q8_0 *) src_ptr;

    ਲਈ (ਪੂਰਨ i = 0; i < nb; i++) {
        HVX_ਸਦਿਸ਼ vd_f16     = Q6_Vh_vsplat_R(*(ਅਚਲ int16_t *) &src[i].d);
        HVX_ਸਦਿਸ਼_Pair vp_f32 = hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(vd_f16);
        HVX_ਸਦਿਸ਼ vd         = Q6_V_lo_W(vp_f32);

        HVX_ਸਦਿਸ਼ vq_i8 = *(ਅਚਲ HVX_UVector *) src[i].qs;

        HVX_ਸਦਿਸ਼_Pair p16   = Q6_Wh_vunpack_Vb(vq_i8);
        HVX_ਸਦਿਸ਼     v_i16 = Q6_V_lo_W(p16);
        HVX_ਸਦਿਸ਼_Pair p32   = Q6_Ww_vunpack_Vh(v_i16);
        HVX_ਸਦਿਸ਼     v_i32 = Q6_V_lo_W(p32);

        HVX_ਸਦਿਸ਼ v_f32 = Q6_Vsf_equals_Vw(v_i32);
        HVX_ਸਦਿਸ਼ res   = hvx_vec_mul_f32_f32(v_f32, vd);

        ਦਸ਼ਮਲਵ * ਖੰਡ_dst = dst_ptr + i * QK8_0;
        hvx_vmem(ਖੰਡ_dst) = res;
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_dequantize_ਪੰਕਤੀ_flat_q8_0_f16(__fp16 * ਸੀਮਿਤ dst_ptr, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src_ptr, uint32_t n) {
    ਅਚਲ ਪੂਰਨ nb = n / QK8_0;
    ਅਚਲ ਖੰਡ_q8_0 * src = (ਅਚਲ ਖੰਡ_q8_0 *) src_ptr;

    ਲਈ (ਪੂਰਨ i = nb - 1; i >= 0; i--) {
        HVX_ਸਦਿਸ਼ vd_f16     = Q6_Vh_vsplat_R(*(ਅਚਲ int16_t *) &src[i].d);
        HVX_ਸਦਿਸ਼_Pair vp_f32 = hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(vd_f16);
        HVX_ਸਦਿਸ਼ vd         = Q6_V_lo_W(vp_f32);

        HVX_ਸਦਿਸ਼ vq_i8 = *(ਅਚਲ HVX_UVector *) src[i].qs;

        HVX_ਸਦਿਸ਼_Pair p16   = Q6_Wh_vunpack_Vb(vq_i8);
        HVX_ਸਦਿਸ਼     v_i16 = Q6_V_lo_W(p16);
        HVX_ਸਦਿਸ਼_Pair p32   = Q6_Ww_vunpack_Vh(v_i16);
        HVX_ਸਦਿਸ਼     v_i32 = Q6_V_lo_W(p32);

        HVX_ਸਦਿਸ਼ v_f32 = Q6_Vsf_equals_Vw(v_i32);
        HVX_ਸਦਿਸ਼ res_f32 = hvx_vec_mul_f32_f32(v_f32, vd);

        HVX_ਸਦਿਸ਼ res_f16 = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16(res_f32, Q6_V_vzero());

        __fp16 * ਖੰਡ_dst = dst_ptr + i * QK8_0;
        hvx_vec_store_u(ਖੰਡ_dst, QK8_0 * ਆਕਾਰ(__fp16), res_f16);
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_dequantize_ਪੰਕਤੀ_flat_f16_f32(ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst_ptr, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src_ptr, uint32_t n) {
    ਅਚਲ uint32_t nvec = n / VLEN_FP32;
    ਅਚਲ uint32_t nloe = n % VLEN_FP32;
    ਅਚਲ _Float16 * src = (ਅਚਲ _Float16 *) src_ptr;

    ਲਈ (uint32_t i = 0; i < nvec; i++) {
        HVX_ਸਦਿਸ਼ v_f16 = *(ਅਚਲ HVX_UVector *) (src + i * VLEN_FP32);
        HVX_ਸਦਿਸ਼_Pair vp_f32 = hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(v_f16);
        HVX_ਸਦਿਸ਼ res = Q6_V_lo_W(vp_f32);

        hvx_vmem(dst_ptr + i * VLEN_FP32) = res;
    }

    ਜੇ (nloe) {
        HVX_ਸਦਿਸ਼ v_f16 = *(ਅਚਲ HVX_UVector *) (src + nvec * VLEN_FP32);
        HVX_ਸਦਿਸ਼_Pair vp_f32 = hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(v_f16);
        HVX_ਸਦਿਸ਼ res = Q6_V_lo_W(vp_f32);

        hvx_vec_store_a(dst_ptr + nvec * VLEN_FP32, nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), res);
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_dequantize_ਪੰਕਤੀ_flat_bf16_f32(ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst_ptr, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src_ptr, uint32_t n) {
    ਅਚਲ uint32_t nvec = n / VLEN_FP32;
    ਅਚਲ uint32_t nloe = n % VLEN_FP32;
    ਅਚਲ uint16_t * src = (ਅਚਲ uint16_t *) src_ptr;

    ਲਈ (uint32_t i = 0; i < nvec; i++) {
        HVX_ਸਦਿਸ਼ v_bf16 = *(ਅਚਲ HVX_UVector *) (src + i * VLEN_FP32);
        HVX_ਸਦਿਸ਼_Pair p = Q6_Ww_vunpack_Vh(v_bf16);
        HVX_ਸਦਿਸ਼ res = Q6_Vw_vasl_VwR(Q6_V_lo_W(p), 16);

        hvx_vmem(dst_ptr + i * VLEN_FP32) = res;
    }

    ਜੇ (nloe) {
        HVX_ਸਦਿਸ਼ v_bf16 = *(ਅਚਲ HVX_UVector *) (src + nvec * VLEN_FP32);
        HVX_ਸਦਿਸ਼_Pair p = Q6_Ww_vunpack_Vh(v_bf16);
        HVX_ਸਦਿਸ਼ res = Q6_Vw_vasl_VwR(Q6_V_lo_W(p), 16);

        hvx_vec_store_a(dst_ptr + nvec * VLEN_FP32, nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), res);
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_dequantize_ਪੰਕਤੀ_flat_q1_0_f32(ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst_ptr, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src_ptr, uint32_t n) {
    ਸਥਿਰ ਅਚਲ uint32_t __attribute__((aligned(128))) ਦੋਆਧਾਰੀ_ਅੰਕ_ਨਕਾਬ[32] = {
        1u <<  0, 1u <<  1, 1u <<  2, 1u <<  3, 1u <<  4, 1u <<  5, 1u <<  6, 1u <<  7,
        1u <<  8, 1u <<  9, 1u << 10, 1u << 11, 1u << 12, 1u << 13, 1u << 14, 1u << 15,
        1u << 16, 1u << 17, 1u << 18, 1u << 19, 1u << 20, 1u << 21, 1u << 22, 1u << 23,
        1u << 24, 1u << 25, 1u << 26, 1u << 27, 1u << 28, 1u << 29, 1u << 30, 1u << 31,
    };
    ਅਚਲ ਪੂਰਨ nb = n / QK1_0;
    ਅਚਲ ਖੰਡ_q1_0 * src = (ਅਚਲ ਖੰਡ_q1_0 *) src_ptr;
    ਅਚਲ HVX_ਸਦਿਸ਼ v_ਦੋਆਧਾਰੀ_ਅੰਕ_ਨਕਾਬ = *(ਅਚਲ HVX_ਸਦਿਸ਼ *) ਦੋਆਧਾਰੀ_ਅੰਕ_ਨਕਾਬ;
    ਅਚਲ HVX_ਸਦਿਸ਼ v_zero     = Q6_V_vzero();
    ਅਚਲ HVX_ਸਦਿਸ਼ v_sign_ਦੋਆਧਾਰੀ_ਅੰਕ = Q6_V_vsplat_R(0x80000000);

    ਲਈ (ਪੂਰਨ i = 0; i < nb; i++) {
        HVX_ਸਦਿਸ਼ vd_f16 = Q6_Vh_vsplat_R(*(ਅਚਲ int16_t *) &src[i].d);
        HVX_ਸਦਿਸ਼ vd     = Q6_V_lo_W(hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(vd_f16));
        HVX_ਸਦਿਸ਼ neg_vd = Q6_V_vxor_VV(vd, v_sign_ਦੋਆਧਾਰੀ_ਅੰਕ);
        HVX_ਸਦਿਸ਼ v_qs   = *(ਅਚਲ HVX_UVector *) src[i].qs;

        ਦਸ਼ਮਲਵ * ਖੰਡ_dst = dst_ptr + i * QK1_0;
        #ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ q = 0; q < 4; q++) {
            HVX_ਸਦਿਸ਼ v_ਬਿੱਟ = hvx_vec_repl_u32(q == 0 ? v_qs : Q6_V_vror_VR(v_qs, 4 * q));
            HVX_ਸਦਿਸ਼_Pred q_zero = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(v_ਬਿੱਟ, v_ਦੋਆਧਾਰੀ_ਅੰਕ_ਨਕਾਬ), v_zero);
            hvx_vmem(ਖੰਡ_dst + q * 32) = Q6_V_vmux_QVV(q_zero, neg_vd, vd);
        }
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_dequantize_ਪੰਕਤੀ_flat_q2_0_f32(ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst_ptr, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src_ptr, uint32_t n) {
    ਅਚਲ ਪੂਰਨ nb = n / QK2_0;
    ਅਚਲ ਖੰਡ_q2_0 * src = (ਅਚਲ ਖੰਡ_q2_0 *) src_ptr;
    ਅਚਲ HVX_ਸਦਿਸ਼ ਨਕਾਬ_03 = Q6_Vb_vsplat_R(0x03);
    ਅਚਲ HVX_ਸਦਿਸ਼ v_one   = Q6_Vb_vsplat_R(1);

    ਲਈ (ਪੂਰਨ i = 0; i < nb; i++) {
        HVX_ਸਦਿਸ਼ vd_f16 = Q6_Vh_vsplat_R(*(ਅਚਲ int16_t *) &src[i].d);
        HVX_ਸਦਿਸ਼ vd     = Q6_V_lo_W(hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(vd_f16));

        HVX_ਸਦਿਸ਼ vq  = *(ਅਚਲ HVX_UVector *) src[i].qs;
        HVX_ਸਦਿਸ਼ v0  = Q6_V_vand_VV(vq, ਨਕਾਬ_03);
        HVX_ਸਦਿਸ਼ v1  = Q6_V_vand_VV(Q6_Vub_vlsr_VubR(vq, 2), ਨਕਾਬ_03);
        HVX_ਸਦਿਸ਼ v2  = Q6_V_vand_VV(Q6_Vub_vlsr_VubR(vq, 4), ਨਕਾਬ_03);
        HVX_ਸਦਿਸ਼ v3  = Q6_Vub_vlsr_VubR(vq, 6);

        HVX_ਸਦਿਸ਼ v01 = Q6_V_lo_W(Q6_W_vshuff_VVR(v1, v0, -1));
        HVX_ਸਦਿਸ਼ v23 = Q6_V_lo_W(Q6_W_vshuff_VVR(v3, v2, -1));
        HVX_ਸਦਿਸ਼ v64 = Q6_Vb_vsub_VbVb(Q6_V_lo_W(Q6_W_vshuff_VVR(v23, v01, -2)), v_one);

        HVX_ਸਦਿਸ਼_Pair p16 = Q6_Wh_vunpack_Vb(v64);
        HVX_ਸਦਿਸ਼_Pair p32 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16));

        ਦਸ਼ਮਲਵ * ਖੰਡ_dst = dst_ptr + i * QK2_0;
        hvx_vmem(ਖੰਡ_dst + 0)  = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_lo_W(p32)), vd);
        hvx_vmem(ਖੰਡ_dst + 32) = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_hi_W(p32)), vd);
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_dequantize_ਪੰਕਤੀ_flat_q4_0_f32(ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst_ptr, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src_ptr, uint32_t n) {
    ਅਚਲ ਪੂਰਨ nb = n / QK4_0;
    ਅਚਲ ਖੰਡ_q4_0 * src = (ਅਚਲ ਖੰਡ_q4_0 *) src_ptr;

    ਅਚਲ HVX_ਸਦਿਸ਼_Pred q16 = Q6_Q_vsetq_R(16);
    ਅਚਲ HVX_ਸਦਿਸ਼ v_eight = Q6_Vb_vsplat_R(8);
    ਅਚਲ HVX_ਸਦਿਸ਼ v_zero  = Q6_V_vzero();

    ਲਈ (ਪੂਰਨ i = 0; i < nb; i++) {
        HVX_ਸਦਿਸ਼ vd_f16     = Q6_Vh_vsplat_R(*(ਅਚਲ int16_t *) &src[i].d);
        HVX_ਸਦਿਸ਼_Pair vp_f32 = hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(vd_f16);
        HVX_ਸਦਿਸ਼ vd         = Q6_V_lo_W(vp_f32);

        HVX_ਸਦਿਸ਼ vq = *(ਅਚਲ HVX_UVector *) src[i].qs;
        HVX_ਸਦਿਸ਼ v_lo = Q6_V_vand_VV(vq, Q6_Vb_vsplat_R(0x0F));
        HVX_ਸਦਿਸ਼ v_hi = Q6_Vub_vlsr_VubR(vq, 4);

        HVX_ਸਦਿਸ਼ v_lo_s8 = Q6_Vb_vsub_VbVb(v_lo, v_eight);
        HVX_ਸਦਿਸ਼ v_hi_s8 = Q6_Vb_vsub_VbVb(v_hi, v_eight);

        HVX_ਸਦਿਸ਼ v32 = Q6_V_vmux_QVV(q16, v_lo_s8, Q6_V_vlalign_VVR(v_hi_s8, v_zero, 16));

        HVX_ਸਦਿਸ਼_Pair p16   = Q6_Wh_vunpack_Vb(v32);
        HVX_ਸਦਿਸ਼     v_i16 = Q6_V_lo_W(p16);
        HVX_ਸਦਿਸ਼_Pair p32   = Q6_Ww_vunpack_Vh(v_i16);
        HVX_ਸਦਿਸ਼     v_i32 = Q6_V_lo_W(p32);

        HVX_ਸਦਿਸ਼ v_f32 = Q6_Vsf_equals_Vw(v_i32);
        HVX_ਸਦਿਸ਼ res   = hvx_vec_mul_f32_f32(v_f32, vd);

        hvx_vmem(dst_ptr + i * QK4_0) = res;
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_dequantize_ਪੰਕਤੀ_flat_q4_k_f32(ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst_ptr, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src_ptr, uint32_t n) {
    ਅਚਲ ਪੂਰਨ nb = n / QK_K;
    ਅਚਲ ਖੰਡ_q4_K * src = (ਅਚਲ ਖੰਡ_q4_K *) src_ptr;

    ਲਈ (ਪੂਰਨ i = 0; i < nb; i++) {
        ਅਚਲ ਦਸ਼ਮਲਵ d   = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(src[i].d);
        ਅਚਲ ਦਸ਼ਮਲਵ min = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(src[i].dmin);
        ਅਚਲ uint8_t * sc = src[i].scales;

        ਦਸ਼ਮਲਵ * y = dst_ptr + i * QK_K;

        ਲਈ (ਪੂਰਨ ਕਦਮ = 0; ਕਦਮ < 4; ++ਕਦਮ) {
            uint8_t sc1, m1_u, sc2, m2_u;
            ਜੇ (ਕਦਮ < 2) {
                sc1  = sc[2 * ਕਦਮ + 0] & 63;
                m1_u = sc[2 * ਕਦਮ + 4] & 63;
                sc2  = sc[2 * ਕਦਮ + 1] & 63;
                m2_u = sc[2 * ਕਦਮ + 5] & 63;
            } ਨਹੀਂਤਾਂ {
                ਅਚਲ ਪੂਰਨ s = ਕਦਮ - 2;
                sc1  = (sc[8 + 2 * s + 0] & 0xF) | ((sc[2 * s + 0] >> 6) << 4);
                m1_u = (sc[8 + 2 * s + 0] >>  4) | ((sc[4 + 2 * s + 0] >> 6) << 4);
                sc2  = (sc[8 + 2 * s + 1] & 0xF) | ((sc[2 * s + 1] >> 6) << 4);
                m2_u = (sc[8 + 2 * s + 1] >>  4) | ((sc[4 + 2 * s + 1] >> 6) << 4);
            }

            HVX_ਸਦਿਸ਼ vq = *(ਅਚਲ HVX_UVector *) (src[i].qs + ਕਦਮ * 32);
            HVX_ਸਦਿਸ਼ v_lo = Q6_V_vand_VV(vq, Q6_Vb_vsplat_R(0x0F));
            HVX_ਸਦਿਸ਼ v_hi = Q6_Vub_vlsr_VubR(vq, 4);

            HVX_ਸਦਿਸ਼_Pair p16_lo = Q6_Wuh_vunpack_Vub(v_lo);
            HVX_ਸਦਿਸ਼_Pair p32_lo = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16_lo));
            HVX_ਸਦਿਸ਼     v_i32_lo = Q6_V_lo_W(p32_lo);
            HVX_ਸਦਿਸ਼     v_f32_lo = Q6_Vsf_equals_Vw(v_i32_lo);

            HVX_ਸਦਿਸ਼ vd1 = hvx_vec_splat_f32(d * sc1);
            HVX_ਸਦਿਸ਼ vm1 = hvx_vec_splat_f32(min * m1_u);
            HVX_ਸਦਿਸ਼ v_out1 = hvx_vec_sub_f32_f32(hvx_vec_mul_f32_f32(v_f32_lo, vd1), vm1);
            hvx_vmem(y + 0) = v_out1;

            HVX_ਸਦਿਸ਼_Pair p16_hi = Q6_Wuh_vunpack_Vub(v_hi);
            HVX_ਸਦਿਸ਼_Pair p32_hi = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16_hi));
            HVX_ਸਦਿਸ਼     v_i32_hi = Q6_V_lo_W(p32_hi);
            HVX_ਸਦਿਸ਼     v_f32_hi = Q6_Vsf_equals_Vw(v_i32_hi);

            HVX_ਸਦਿਸ਼ vd2 = hvx_vec_splat_f32(d * sc2);
            HVX_ਸਦਿਸ਼ vm2 = hvx_vec_splat_f32(min * m2_u);
            HVX_ਸਦਿਸ਼ v_out2 = hvx_vec_sub_f32_f32(hvx_vec_mul_f32_f32(v_f32_hi, vd2), vm2);
            hvx_vmem(y + 32) = v_out2;

            y += 64;
        }
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_dequantize_ਪੰਕਤੀ_flat_q5_k_f32(ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst_ptr, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src_ptr, uint32_t n) {
    ਅਚਲ ਪੂਰਨ nb = n / QK_K;
    ਅਚਲ ਖੰਡ_q5_K * src = (ਅਚਲ ਖੰਡ_q5_K *) src_ptr;

    ਲਈ (ਪੂਰਨ i = 0; i < nb; i++) {
        ਅਚਲ ਦਸ਼ਮਲਵ d   = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(src[i].d);
        ਅਚਲ ਦਸ਼ਮਲਵ min = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(src[i].dmin);
        ਅਚਲ uint8_t * sc = src[i].scales;

        HVX_ਸਦਿਸ਼ v_qh = *(ਅਚਲ HVX_UVector *) src[i].qh;
        ਦਸ਼ਮਲਵ * y = dst_ptr + i * QK_K;

        ਲਈ (ਪੂਰਨ ਕਦਮ = 0; ਕਦਮ < 4; ++ਕਦਮ) {
            uint8_t sc1, m1_u, sc2, m2_u;
            ਜੇ (ਕਦਮ < 2) {
                sc1  = sc[2 * ਕਦਮ + 0] & 63;
                m1_u = sc[2 * ਕਦਮ + 4] & 63;
                sc2  = sc[2 * ਕਦਮ + 1] & 63;
                m2_u = sc[2 * ਕਦਮ + 5] & 63;
            } ਨਹੀਂਤਾਂ {
                ਅਚਲ ਪੂਰਨ s = ਕਦਮ - 2;
                sc1  = (sc[8 + 2 * s + 0] & 0xF) | ((sc[2 * s + 0] >> 6) << 4);
                m1_u = (sc[8 + 2 * s + 0] >>  4) | ((sc[4 + 2 * s + 0] >> 6) << 4);
                sc2  = (sc[8 + 2 * s + 1] & 0xF) | ((sc[2 * s + 1] >> 6) << 4);
                m2_u = (sc[8 + 2 * s + 1] >>  4) | ((sc[4 + 2 * s + 1] >> 6) << 4);
            }

            HVX_ਸਦਿਸ਼ v_ql = *(ਅਚਲ HVX_UVector *) (src[i].qs + ਕਦਮ * 32);
            HVX_ਸਦਿਸ਼ v_lo = Q6_V_vand_VV(v_ql, Q6_Vb_vsplat_R(0x0F));
            HVX_ਸਦਿਸ਼ v_hi = Q6_Vub_vlsr_VubR(v_ql, 4);

            HVX_ਸਦਿਸ਼ v_h1 = Q6_Vw_vasl_VwR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v_qh, 2 * ਕਦਮ), Q6_Vb_vsplat_R(1)), 4);
            HVX_ਸਦਿਸ਼ v_val1 = Q6_V_vor_VV(v_lo, v_h1);

            HVX_ਸਦਿਸ਼_Pair p16_1 = Q6_Wuh_vunpack_Vub(v_val1);
            HVX_ਸਦਿਸ਼_Pair p32_1 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16_1));
            HVX_ਸਦਿਸ਼     v_i32_1 = Q6_V_lo_W(p32_1);
            HVX_ਸਦਿਸ਼     v_f32_1 = Q6_Vsf_equals_Vw(v_i32_1);

            HVX_ਸਦਿਸ਼ vd1 = hvx_vec_splat_f32(d * sc1);
            HVX_ਸਦਿਸ਼ vm1 = hvx_vec_splat_f32(min * m1_u);
            HVX_ਸਦਿਸ਼ v_out1 = hvx_vec_sub_f32_f32(hvx_vec_mul_f32_f32(v_f32_1, vd1), vm1);
            hvx_vmem(y + 0) = v_out1;

            HVX_ਸਦਿਸ਼ v_h2 = Q6_Vw_vasl_VwR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v_qh, 2 * ਕਦਮ + 1), Q6_Vb_vsplat_R(1)), 4);
            HVX_ਸਦਿਸ਼ v_val2 = Q6_V_vor_VV(v_hi, v_h2);

            HVX_ਸਦਿਸ਼_Pair p16_2 = Q6_Wuh_vunpack_Vub(v_val2);
            HVX_ਸਦਿਸ਼_Pair p32_2 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16_2));
            HVX_ਸਦਿਸ਼     v_i32_2 = Q6_V_lo_W(p32_2);
            HVX_ਸਦਿਸ਼     v_f32_2 = Q6_Vsf_equals_Vw(v_i32_2);

            HVX_ਸਦਿਸ਼ vd2 = hvx_vec_splat_f32(d * sc2);
            HVX_ਸਦਿਸ਼ vm2 = hvx_vec_splat_f32(min * m2_u);
            HVX_ਸਦਿਸ਼ v_out2 = hvx_vec_sub_f32_f32(hvx_vec_mul_f32_f32(v_f32_2, vd2), vm2);
            hvx_vmem(y + 32) = v_out2;

            y += 64;
        }
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_dequantize_ਪੰਕਤੀ_flat_q6_k_f32(ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst_ptr, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src_ptr, uint32_t n) {
    ਅਚਲ ਪੂਰਨ nb = n / QK_K;
    ਅਚਲ ਖੰਡ_q6_K * src = (ਅਚਲ ਖੰਡ_q6_K *) src_ptr;

    ਅਚਲ HVX_ਸਦਿਸ਼_Pred q64 = Q6_Q_vsetq_R(64);
    ਅਚਲ HVX_ਸਦਿਸ਼ v32_b   = Q6_Vb_vsplat_R(32);
    ਅਚਲ HVX_ਸਦਿਸ਼ ਨਕਾਬ_0f = Q6_Vb_vsplat_R(0x0F);
    ਅਚਲ HVX_ਸਦਿਸ਼ ਨਕਾਬ_03 = Q6_Vb_vsplat_R(0x03);

    ਲਈ (ਪੂਰਨ i = 0; i < nb; i++) {
        ਅਚਲ ਦਸ਼ਮਲਵ d = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(src[i].d);

        ਅਚਲ uint8_t * ਸੀਮਿਤ ql = src[i].ql;
        ਅਚਲ uint8_t * ਸੀਮਿਤ qh = src[i].qh;
        ਅਚਲ int8_t  * ਸੀਮਿਤ sc = src[i].scales;

        ਦਸ਼ਮਲਵ * y = dst_ptr + i * QK_K;

        ਲਈ (ਪੂਰਨ half = 0; half < 2; ++half) {
            ਦਸ਼ਮਲਵ s[8];
            ਲਈ (ਪੂਰਨ j = 0; j < 8; ++j) {
                s[j] = d * sc[j];
            }

            HVX_ਸਦਿਸ਼ vscale1 = Q6_V_vmux_QVV(q64, hvx_vec_splat_f32(s[0]), hvx_vec_splat_f32(s[1]));
            HVX_ਸਦਿਸ਼ vscale2 = Q6_V_vmux_QVV(q64, hvx_vec_splat_f32(s[2]), hvx_vec_splat_f32(s[3]));
            HVX_ਸਦਿਸ਼ vscale3 = Q6_V_vmux_QVV(q64, hvx_vec_splat_f32(s[4]), hvx_vec_splat_f32(s[5]));
            HVX_ਸਦਿਸ਼ vscale4 = Q6_V_vmux_QVV(q64, hvx_vec_splat_f32(s[6]), hvx_vec_splat_f32(s[7]));

            HVX_ਸਦਿਸ਼ v_ql0 = *(ਅਚਲ HVX_UVector *) (ql);
            HVX_ਸਦਿਸ਼ v_ql1 = *(ਅਚਲ HVX_UVector *) (ql + 32);
            HVX_ਸਦਿਸ਼ v_qh  = *(ਅਚਲ HVX_UVector *) (qh);

            HVX_ਸਦਿਸ਼ ql0_lo = Q6_V_vand_VV(v_ql0, ਨਕਾਬ_0f);
            HVX_ਸਦਿਸ਼ qh_0   = Q6_Vw_vasl_VwR(Q6_V_vand_VV(v_qh, ਨਕਾਬ_03), 4);
            HVX_ਸਦਿਸ਼ v_q1   = Q6_Vb_vsub_VbVb(Q6_V_vor_VV(ql0_lo, qh_0), v32_b);

            HVX_ਸਦਿਸ਼ ql1_lo = Q6_V_vand_VV(v_ql1, ਨਕਾਬ_0f);
            HVX_ਸਦਿਸ਼ qh_1   = Q6_Vw_vasl_VwR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v_qh, 2), ਨਕਾਬ_03), 4);
            HVX_ਸਦਿਸ਼ v_q2   = Q6_Vb_vsub_VbVb(Q6_V_vor_VV(ql1_lo, qh_1), v32_b);

            HVX_ਸਦਿਸ਼ ql0_hi = Q6_Vub_vlsr_VubR(v_ql0, 4);
            HVX_ਸਦਿਸ਼ qh_2   = Q6_Vw_vasl_VwR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v_qh, 4), ਨਕਾਬ_03), 4);
            HVX_ਸਦਿਸ਼ v_q3   = Q6_Vb_vsub_VbVb(Q6_V_vor_VV(ql0_hi, qh_2), v32_b);

            HVX_ਸਦਿਸ਼ ql1_hi = Q6_Vub_vlsr_VubR(v_ql1, 4);
            HVX_ਸਦਿਸ਼ qh_3   = Q6_Vw_vasl_VwR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v_qh, 6), ਨਕਾਬ_03), 4);
            HVX_ਸਦਿਸ਼ v_q4   = Q6_Vb_vsub_VbVb(Q6_V_vor_VV(ql1_hi, qh_3), v32_b);

            HVX_ਸਦਿਸ਼_Pair p16_1 = Q6_Wh_vunpack_Vb(v_q1);
            HVX_ਸਦਿਸ਼_Pair p32_1 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16_1));
            HVX_ਸਦਿਸ਼     v_f32_1 = Q6_Vsf_equals_Vw(Q6_V_lo_W(p32_1));
            hvx_vmem(y + 0) = hvx_vec_mul_f32_f32(v_f32_1, vscale1);

            HVX_ਸਦਿਸ਼_Pair p16_2 = Q6_Wh_vunpack_Vb(v_q2);
            HVX_ਸਦਿਸ਼_Pair p32_2 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16_2));
            HVX_ਸਦਿਸ਼     v_f32_2 = Q6_Vsf_equals_Vw(Q6_V_lo_W(p32_2));
            hvx_vmem(y + 32) = hvx_vec_mul_f32_f32(v_f32_2, vscale2);

            HVX_ਸਦਿਸ਼_Pair p16_3 = Q6_Wh_vunpack_Vb(v_q3);
            HVX_ਸਦਿਸ਼_Pair p32_3 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16_3));
            HVX_ਸਦਿਸ਼     v_f32_3 = Q6_Vsf_equals_Vw(Q6_V_lo_W(p32_3));
            hvx_vmem(y + 64) = hvx_vec_mul_f32_f32(v_f32_3, vscale3);

            HVX_ਸਦਿਸ਼_Pair p16_4 = Q6_Wh_vunpack_Vb(v_q4);
            HVX_ਸਦਿਸ਼_Pair p32_4 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16_4));
            HVX_ਸਦਿਸ਼     v_f32_4 = Q6_Vsf_equals_Vw(Q6_V_lo_W(p32_4));
            hvx_vmem(y + 96) = hvx_vec_mul_f32_f32(v_f32_4, vscale4);

            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

// bytes row, row + 32, row + 64 and row + 96 of v, to bytes 0..3
ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_dequantize_gather4(HVX_ਸਦਿਸ਼ v, uint32_t ਪੰਕਤੀ) {
    v = Q6_V_vror_VR(v, ਪੰਕਤੀ);
    HVX_ਸਦਿਸ਼ packed = Q6_V_vmux_QVV(Q6_Q_vsetq_R(1), v, Q6_V_vror_VR(v, 31));
    packed = Q6_V_vmux_QVV(Q6_Q_vsetq_R(2), packed, Q6_V_vror_VR(v, 62));
    ਵਾਪਸ Q6_V_vmux_QVV(Q6_Q_vsetq_R(3), packed, Q6_V_vror_VR(v, 93));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_dequantize_ਪੰਕਤੀ_tiled_q8_0_f32(ਦਸ਼ਮਲਵ * dst_spad, ਅਚਲ uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t ਪੰਕਤੀ) {
    ਅਚਲ HVX_ਸਦਿਸ਼_Pred first2 = Q6_Q_vsetq_R(2);
    ਅਚਲ HVX_ਸਦਿਸ਼_Pred first4 = Q6_Q_vsetq_R(4);

    ਲਈ (uint32_t k_tile = 0; k_tile < n_k_tiles; ++k_tile) {
        ਅਚਲ uint8_t * tile = src_spad + k_tile * tile_stride;
        ਦਸ਼ਮਲਵ * dst = dst_spad + k_tile * HTP_TENSOR_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ;

        HVX_ਸਦਿਸ਼ vq = Q6_V_vzero();
        ਲਈ (ਪੂਰਨ ਸਮੂਹ = 7; ਸਮੂਹ >= 0; --ਸਮੂਹ) {
            ਅਚਲ HVX_ਸਦਿਸ਼ v = Q6_V_vror_VR(hvx_vmem(tile + ਸਮੂਹ * VLEN), 2 * ਪੰਕਤੀ);
            ਅਚਲ HVX_ਸਦਿਸ਼ packed = Q6_V_vmux_QVV(first2, v, Q6_V_vror_VR(v, 62));
            vq = Q6_V_vmux_QVV(first4, packed, Q6_V_vror_VR(vq, VLEN - 4));
        }

        ਅਚਲ HVX_ਸਦਿਸ਼ scales = hvx_vmem(tile + 1024);
        ਅਚਲ HVX_ਸਦਿਸ਼ scale_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 2 * ਪੰਕਤੀ));
        ਅਚਲ HVX_ਸਦਿਸ਼ scale = Q6_V_lo_W(hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(scale_hf));
        ਅਚਲ HVX_ਸਦਿਸ਼_Pair p16 = Q6_Wh_vunpack_Vb(vq);
        ਅਚਲ HVX_ਸਦਿਸ਼_Pair p32 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16));
        HVX_ਸਦਿਸ਼ ਮੁੱਲਾਂ = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_lo_W(p32)), scale);
        *(HVX_ਸਦਿਸ਼ *) dst = ਮੁੱਲਾਂ;
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_dequantize_ਪੰਕਤੀ_tiled_q4_0_f32(ਦਸ਼ਮਲਵ * dst_spad, ਅਚਲ uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t ਪੰਕਤੀ) {
    ਅਚਲ HVX_ਸਦਿਸ਼_Pred first4 = Q6_Q_vsetq_R(4);

    ਲਈ (uint32_t k_tile = 0; k_tile < n_k_tiles; ++k_tile) {
        ਅਚਲ uint8_t * tile = src_spad + k_tile * tile_stride;
        ਦਸ਼ਮਲਵ * dst = dst_spad + k_tile * HTP_TENSOR_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ;

        HVX_ਸਦਿਸ਼ vq = Q6_V_vzero();
        ਲਈ (ਪੂਰਨ ਸਮੂਹ = 3; ਸਮੂਹ >= 0; --ਸਮੂਹ) {
            vq = Q6_V_vmux_QVV(first4, hvx_dequantize_gather4(hvx_vmem(tile + ਸਮੂਹ * VLEN), ਪੰਕਤੀ), Q6_V_vror_VR(vq, VLEN - 4));
        }

        ਅਚਲ HVX_ਸਦਿਸ਼ lo = Q6_V_vand_VV(vq, Q6_Vb_vsplat_R(0x0F));
        ਅਚਲ HVX_ਸਦਿਸ਼ hi = Q6_Vub_vlsr_VubR(vq, 4);
        vq = Q6_V_lo_W(Q6_W_vshuff_VVR(hi, lo, -1));
        vq = Q6_Vb_vsub_VbVb(vq, Q6_Vb_vsplat_R(8));

        ਅਚਲ HVX_ਸਦਿਸ਼ scales = hvx_vmem(tile + 512);
        ਅਚਲ HVX_ਸਦਿਸ਼ scale_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 2 * ਪੰਕਤੀ));
        ਅਚਲ HVX_ਸਦਿਸ਼ scale = Q6_V_lo_W(hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(scale_hf));
        ਅਚਲ HVX_ਸਦਿਸ਼_Pair p16 = Q6_Wh_vunpack_Vb(vq);
        ਅਚਲ HVX_ਸਦਿਸ਼_Pair p32 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16));
        HVX_ਸਦਿਸ਼ ਮੁੱਲਾਂ = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_lo_W(p32)), scale);
        *(HVX_ਸਦਿਸ਼ *) dst = ਮੁੱਲਾਂ;
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_dequantize_ਪੰਕਤੀ_tiled_q4_k_f32(ਦਸ਼ਮਲਵ * dst_spad, ਅਚਲ uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t ਪੰਕਤੀ) {
    ਅਚਲ HVX_ਸਦਿਸ਼_Pred first4 = Q6_Q_vsetq_R(4);

    ਲਈ (uint32_t k_tile = 0; k_tile < n_k_tiles; ++k_tile) {
        ਅਚਲ uint8_t * tile = src_spad + k_tile * tile_stride;
        ਦਸ਼ਮਲਵ * dst = dst_spad + k_tile * HTP_TENSOR_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ;

        HVX_ਸਦਿਸ਼ vq = Q6_V_vzero();
        ਲਈ (ਪੂਰਨ ਸਮੂਹ = 3; ਸਮੂਹ >= 0; --ਸਮੂਹ) {
            vq = Q6_V_vmux_QVV(first4, hvx_dequantize_gather4(hvx_vmem(tile + ਸਮੂਹ * VLEN), ਪੰਕਤੀ), Q6_V_vror_VR(vq, VLEN - 4));
        }

        ਅਚਲ HVX_ਸਦਿਸ਼ lo = Q6_V_vand_VV(vq, Q6_Vb_vsplat_R(0x0F));
        ਅਚਲ HVX_ਸਦਿਸ਼ hi = Q6_Vub_vlsr_VubR(vq, 4);
        vq = Q6_V_lo_W(Q6_W_vshuff_VVR(hi, lo, -1));

        ਅਚਲ HVX_ਸਦਿਸ਼ scales = hvx_vmem(tile + 512);
        ਅਚਲ HVX_ਸਦਿਸ਼ scale_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 4 * ਪੰਕਤੀ));
        ਅਚਲ HVX_ਸਦਿਸ਼ scale = Q6_V_lo_W(hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(scale_hf));
        ਅਚਲ HVX_ਸਦਿਸ਼ ਵਿੱਥ_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 4 * ਪੰਕਤੀ + 2));
        ਅਚਲ HVX_ਸਦਿਸ਼ ਵਿੱਥ = Q6_V_lo_W(hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(ਵਿੱਥ_hf));

        ਅਚਲ HVX_ਸਦਿਸ਼_Pair p16 = Q6_Wh_vunpack_Vb(vq);
        ਅਚਲ HVX_ਸਦਿਸ਼_Pair p32 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16));
        HVX_ਸਦਿਸ਼ ਮੁੱਲਾਂ = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_lo_W(p32)), scale);
        *(HVX_ਸਦਿਸ਼ *) dst = hvx_vec_ਜੋੜੋ_f32_f32(ਮੁੱਲਾਂ, ਵਿੱਥ);
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_dequantize_ਪੰਕਤੀ_tiled_q5_k_f32(ਦਸ਼ਮਲਵ * dst_spad, ਅਚਲ uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t ਪੰਕਤੀ) {
    ਅਚਲ HVX_ਸਦਿਸ਼_Pred first4 = Q6_Q_vsetq_R(4);
    ਅਚਲ HVX_ਸਦਿਸ਼ one = Q6_Vb_vsplat_R(1);

    ਲਈ (uint32_t k_tile = 0; k_tile < n_k_tiles; ++k_tile) {
        ਅਚਲ uint8_t * tile = src_spad + k_tile * tile_stride;
        ਦਸ਼ਮਲਵ * dst = dst_spad + k_tile * HTP_TENSOR_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ;

        ਅਚਲ HVX_ਸਦਿਸ਼ p5 = hvx_dequantize_gather4(hvx_vmem(tile + 640), ਪੰਕਤੀ);
        HVX_ਸਦਿਸ਼ vq = Q6_V_vzero();
        HVX_ਸਦਿਸ਼ vh = Q6_V_vzero();
        ਲਈ (ਪੂਰਨ ਸਮੂਹ = 3; ਸਮੂਹ >= 0; --ਸਮੂਹ) {
            vq = Q6_V_vmux_QVV(first4, hvx_dequantize_gather4(hvx_vmem(tile + ਸਮੂਹ * VLEN), ਪੰਕਤੀ), Q6_V_vror_VR(vq, VLEN - 4));
            vh = Q6_V_vmux_QVV(first4, Q6_Vub_vlsr_VubR(p5, 2 * ਸਮੂਹ), Q6_V_vror_VR(vh, VLEN - 4));
        }

        ਅਚਲ HVX_ਸਦਿਸ਼ lo = Q6_V_vor_VV(Q6_V_vand_VV(vq, Q6_Vb_vsplat_R(0x0F)), Q6_Vw_vasl_VwR(Q6_V_vand_VV(vh, one), 4));
        ਅਚਲ HVX_ਸਦਿਸ਼ hi = Q6_V_vor_VV(Q6_Vub_vlsr_VubR(vq, 4), Q6_Vw_vasl_VwR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(vh, 1), one), 4));
        vq = Q6_V_lo_W(Q6_W_vshuff_VVR(hi, lo, -1));

        ਅਚਲ HVX_ਸਦਿਸ਼ scales = hvx_vmem(tile + 512);
        ਅਚਲ HVX_ਸਦਿਸ਼ scale_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 4 * ਪੰਕਤੀ));
        ਅਚਲ HVX_ਸਦਿਸ਼ scale = Q6_V_lo_W(hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(scale_hf));
        ਅਚਲ HVX_ਸਦਿਸ਼ ਵਿੱਥ_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 4 * ਪੰਕਤੀ + 2));
        ਅਚਲ HVX_ਸਦਿਸ਼ ਵਿੱਥ = Q6_V_lo_W(hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(ਵਿੱਥ_hf));

        ਅਚਲ HVX_ਸਦਿਸ਼_Pair p16 = Q6_Wh_vunpack_Vb(vq);
        ਅਚਲ HVX_ਸਦਿਸ਼_Pair p32 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16));
        HVX_ਸਦਿਸ਼ ਮੁੱਲਾਂ = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_lo_W(p32)), scale);
        *(HVX_ਸਦਿਸ਼ *) dst = hvx_vec_ਜੋੜੋ_f32_f32(ਮੁੱਲਾਂ, ਵਿੱਥ);
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_dequantize_ਪੰਕਤੀ_tiled_q6_k_f32(ਦਸ਼ਮਲਵ * dst_spad, ਅਚਲ uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t ਪੰਕਤੀ) {
    ਅਚਲ HVX_ਸਦਿਸ਼_Pred first4 = Q6_Q_vsetq_R(4);
    ਅਚਲ HVX_ਸਦਿਸ਼_Pred first16 = Q6_Q_vsetq_R(16 * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਅਚਲ HVX_ਸਦਿਸ਼ ਨਕਾਬ_0f = Q6_Vb_vsplat_R(0x0F);
    ਅਚਲ HVX_ਸਦਿਸ਼ ਨਕਾਬ_03 = Q6_Vb_vsplat_R(0x03);

    ਲਈ (uint32_t k_tile = 0; k_tile < n_k_tiles; ++k_tile) {
        ਅਚਲ uint8_t * tile = src_spad + k_tile * tile_stride;
        ਦਸ਼ਮਲਵ * dst = dst_spad + k_tile * HTP_TENSOR_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ;

        HVX_ਸਦਿਸ਼ vq = Q6_V_vzero();
        ਲਈ (ਪੂਰਨ ਸਮੂਹ = 7; ਸਮੂਹ >= 0; --ਸਮੂਹ) {
            ਅਚਲ HVX_ਸਦਿਸ਼ lo_plane = Q6_V_vror_VR(hvx_vmem(tile + (ਸਮੂਹ >> 1) * VLEN), 4 * ਪੰਕਤੀ);
            ਅਚਲ HVX_ਸਦਿਸ਼ hi_plane = Q6_V_vror_VR(hvx_vmem(tile + 512 + (ਸਮੂਹ >> 2) * VLEN), 4 * ਪੰਕਤੀ);
            ਅਚਲ HVX_ਸਦਿਸ਼ lo = (ਸਮੂਹ & 1) ? Q6_Vub_vlsr_VubR(lo_plane, 4) : Q6_V_vand_VV(lo_plane, ਨਕਾਬ_0f);
            ਅਚਲ HVX_ਸਦਿਸ਼ hi = Q6_Vub_vlsr_VubR(hi_plane, 2 * (ਸਮੂਹ & 3));
            ਅਚਲ HVX_ਸਦਿਸ਼ packed = Q6_V_vor_VV(lo, Q6_Vw_vasl_VwR(Q6_V_vand_VV(hi, ਨਕਾਬ_03), 4));
            vq = Q6_V_vmux_QVV(first4, packed, Q6_V_vror_VR(vq, VLEN - 4));
        }

        ਅਚਲ HVX_ਸਦਿਸ਼ scales = hvx_vmem(tile + 768);
        ਅਚਲ HVX_ਸਦਿਸ਼ scale_lo_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 2 * ਪੰਕਤੀ));
        ਅਚਲ HVX_ਸਦਿਸ਼ scale_hi_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 64 + 2 * ਪੰਕਤੀ));
        ਅਚਲ HVX_ਸਦਿਸ਼ scale_lo = Q6_V_lo_W(hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(scale_lo_hf));
        ਅਚਲ HVX_ਸਦਿਸ਼ scale_hi = Q6_V_lo_W(hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(scale_hi_hf));
        ਅਚਲ HVX_ਸਦਿਸ਼ scale = Q6_V_vmux_QVV(first16, scale_lo, scale_hi);
        ਅਚਲ HVX_ਸਦਿਸ਼_Pair p16 = Q6_Wh_vunpack_Vb(Q6_Vb_vsub_VbVb(vq, Q6_Vb_vsplat_R(32)));
        ਅਚਲ HVX_ਸਦਿਸ਼_Pair p32 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16));
        *(HVX_ਸਦਿਸ਼ *) dst = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_lo_W(p32)), scale);
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_dequantize_tiled_f16_gather4(ਅਚਲ uint8_t * p, uint32_t r_byte) {
    HVX_ਸਦਿਸ਼ w0  = Q6_V_vror_VR(hvx_vmem(p + 0 * VLEN), r_byte);
    HVX_ਸਦਿਸ਼ w1  = Q6_V_vror_VR(hvx_vmem(p + 1 * VLEN), r_byte);
    HVX_ਸਦਿਸ਼ w01 = Q6_V_lo_W(Q6_W_vshuff_VVR(w1, w0, 4));
    HVX_ਸਦਿਸ਼ w2  = Q6_V_vror_VR(hvx_vmem(p + 2 * VLEN), r_byte);
    HVX_ਸਦਿਸ਼ w3  = Q6_V_vror_VR(hvx_vmem(p + 3 * VLEN), r_byte);
    HVX_ਸਦਿਸ਼ w23 = Q6_V_lo_W(Q6_W_vshuff_VVR(w3, w2, 4));
    ਵਾਪਸ Q6_V_lo_W(Q6_W_vshuff_VVR(w23, w01, 8));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_dequantize_ਪੰਕਤੀ_tiled_f16_f32(ਦਸ਼ਮਲਵ * dst_spad, ਅਚਲ uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t ਪੰਕਤੀ) {
    ਅਚਲ uint32_t r_byte = 4 * ਪੰਕਤੀ;

    ਲਈ (uint32_t k_tile = 0; k_tile < n_k_tiles; ++k_tile) {
        ਅਚਲ uint8_t * tile = src_spad + k_tile * tile_stride;
        ਦਸ਼ਮਲਵ * dst = dst_spad + k_tile * HTP_TENSOR_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ;

        HVX_ਸਦਿਸ਼ g0 = hvx_dequantize_tiled_f16_gather4(tile + 0 * VLEN, r_byte);
        HVX_ਸਦਿਸ਼ g1 = hvx_dequantize_tiled_f16_gather4(tile + 4 * VLEN, r_byte);
        HVX_ਸਦਿਸ਼ g01 = Q6_V_lo_W(Q6_W_vshuff_VVR(g1, g0, 16));

        HVX_ਸਦਿਸ਼ g2 = hvx_dequantize_tiled_f16_gather4(tile + 8 * VLEN, r_byte);
        HVX_ਸਦਿਸ਼ g3 = hvx_dequantize_tiled_f16_gather4(tile + 12 * VLEN, r_byte);
        HVX_ਸਦਿਸ਼ g23 = Q6_V_lo_W(Q6_W_vshuff_VVR(g3, g2, 16));

        HVX_ਸਦਿਸ਼ g = Q6_V_lo_W(Q6_W_vshuff_VVR(g23, g01, 32));
        *(HVX_ਸਦਿਸ਼ *) dst = Q6_V_lo_W(hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(g));
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_dequantize_ਪੰਕਤੀ_tiled_q1_0_f32(ਦਸ਼ਮਲਵ * dst_spad, ਅਚਲ uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t ਪੰਕਤੀ) {
    ਸਥਿਰ ਅਚਲ uint32_t __attribute__((aligned(128))) ਦੋਆਧਾਰੀ_ਅੰਕ_ਨਕਾਬ[32] = {
        1u <<  0, 1u <<  8, 1u << 16, 1u << 24,
        1u <<  1, 1u <<  9, 1u << 17, 1u << 25,
        1u <<  2, 1u << 10, 1u << 18, 1u << 26,
        1u <<  3, 1u << 11, 1u << 19, 1u << 27,
        1u <<  4, 1u << 12, 1u << 20, 1u << 28,
        1u <<  5, 1u << 13, 1u << 21, 1u << 29,
        1u <<  6, 1u << 14, 1u << 22, 1u << 30,
        1u <<  7, 1u << 15, 1u << 23, 1u << 31,
    };
    ਅਚਲ HVX_ਸਦਿਸ਼ v_ਦੋਆਧਾਰੀ_ਅੰਕ_ਨਕਾਬ = *(ਅਚਲ HVX_ਸਦਿਸ਼ *) ਦੋਆਧਾਰੀ_ਅੰਕ_ਨਕਾਬ;
    ਅਚਲ HVX_ਸਦਿਸ਼ v_zero     = Q6_V_vzero();
    ਅਚਲ HVX_ਸਦਿਸ਼ v_sign_ਦੋਆਧਾਰੀ_ਅੰਕ = Q6_V_vsplat_R(0x80000000);

    ਲਈ (uint32_t k_tile = 0; k_tile < n_k_tiles; ++k_tile) {
        ਅਚਲ uint8_t * tile = src_spad + k_tile * tile_stride;
        ਦਸ਼ਮਲਵ * dst = dst_spad + k_tile * HTP_TENSOR_TILE_K_Q1_0;

        ਅਚਲ HVX_ਸਦਿਸ਼ scales   = hvx_vmem(tile + 512);
        ਅਚਲ HVX_ਸਦਿਸ਼ scale_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 2 * ਪੰਕਤੀ));
        ਅਚਲ HVX_ਸਦਿਸ਼ vd       = Q6_V_lo_W(hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(scale_hf));
        ਅਚਲ HVX_ਸਦਿਸ਼ neg_vd   = Q6_V_vxor_VV(vd, v_sign_ਦੋਆਧਾਰੀ_ਅੰਕ);

        #ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ q = 0; q < 4; ++q) {
            ਅਚਲ HVX_ਸਦਿਸ਼ v_ਬਿੱਟ = hvx_vec_repl_u32(Q6_V_vror_VR(hvx_vmem(tile + q * VLEN), 4 * ਪੰਕਤੀ));
            ਅਚਲ HVX_ਸਦਿਸ਼_Pred q_zero = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(v_ਬਿੱਟ, v_ਦੋਆਧਾਰੀ_ਅੰਕ_ਨਕਾਬ), v_zero);
            hvx_vmem(dst + q * 32) = Q6_V_vmux_QVV(q_zero, neg_vd, vd);
        }
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_dequantize_ਪੰਕਤੀ_tiled_q2_0_f32(ਦਸ਼ਮਲਵ * dst_spad, ਅਚਲ uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t ਪੰਕਤੀ) {
    ਅਚਲ HVX_ਸਦਿਸ਼ ਨਕਾਬ_03 = Q6_Vb_vsplat_R(0x03);
    ਅਚਲ HVX_ਸਦਿਸ਼ v_one   = Q6_Vb_vsplat_R(1);

    ਲਈ (uint32_t k_tile = 0; k_tile < n_k_tiles; ++k_tile) {
        ਅਚਲ uint8_t * tile = src_spad + k_tile * tile_stride;
        ਦਸ਼ਮਲਵ * dst = dst_spad + k_tile * HTP_TENSOR_TILE_K_Q2_0;

        ਅਚਲ HVX_ਸਦਿਸ਼ v_packed = hvx_dequantize_tiled_f16_gather4(tile, 4 * ਪੰਕਤੀ);
        ਅਚਲ HVX_ਸਦਿਸ਼ p0 = Q6_V_vand_VV(v_packed, ਨਕਾਬ_03);
        ਅਚਲ HVX_ਸਦਿਸ਼ p1 = Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v_packed, 2), ਨਕਾਬ_03);
        ਅਚਲ HVX_ਸਦਿਸ਼ p2 = Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v_packed, 4), ਨਕਾਬ_03);
        ਅਚਲ HVX_ਸਦਿਸ਼ p3 = Q6_Vub_vlsr_VubR(v_packed, 6);

        ਅਚਲ HVX_ਸਦਿਸ਼ p01 = Q6_V_lo_W(Q6_W_vshuff_VVR(p1, p0, -4));
        ਅਚਲ HVX_ਸਦਿਸ਼ p23 = Q6_V_lo_W(Q6_W_vshuff_VVR(p3, p2, -4));
        ਅਚਲ HVX_ਸਦਿਸ਼ vq  = Q6_Vb_vsub_VbVb(Q6_V_lo_W(Q6_W_vshuff_VVR(p23, p01, -8)), v_one);

        ਅਚਲ HVX_ਸਦਿਸ਼ scales   = hvx_vmem(tile + 512);
        ਅਚਲ HVX_ਸਦਿਸ਼ scale_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 2 * ਪੰਕਤੀ));
        ਅਚਲ HVX_ਸਦਿਸ਼ scale    = Q6_V_lo_W(hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(scale_hf));

        ਅਚਲ HVX_ਸਦਿਸ਼_Pair p16 = Q6_Wh_vunpack_Vb(vq);
        ਅਚਲ HVX_ਸਦਿਸ਼_Pair p32 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16));
        hvx_vmem(dst + 0)  = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_lo_W(p32)), scale);
        hvx_vmem(dst + 32) = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_hi_W(p32)), scale);
    }
}

#ਜੇ_ਅੰਤ // HVX_QUANT_H
