#ہدایت clang diagnostic ignored "-Wunused-variable"
#ہدایت clang diagnostic ignored "-Wunused-function"
#ہدایت clang diagnostic ignored "-Wunused-but-set-variable"

#شامل <HAP_farf.h>

#شامل <math.h>
#شامل <string.h>

#شامل "dma-queue.h"
#شامل "hvx-utils.h"

#تعریف_کرو GGML_COMMON_DECL_C
#شامل "ggml-common.h"
#شامل "htp-ctx.h"
#شامل "htp-ops.h"
#شامل "hex-common.h"
#شامل "htp-tensor.h"
#شامل "htp-vtcm.h"

#تعریف_کرو htp_act_preamble                                 \
    ثابت ساخت htp_tensor * src0 = actx->octx->src[0]; \
    ثابت ساخت htp_tensor * src1 = actx->octx->src[1]; \
    ثابت ساخت htp_tensor * dst  = actx->octx->dst;    \
                                                         \
    ثابت uint32_t ne00 = src0->ne[0];                   \
    ثابت uint32_t ne01 = src0->ne[1];                   \
    ثابت uint32_t ne02 = src0->ne[2];                   \
    ثابت uint32_t ne03 = src0->ne[3];                   \
                                                         \
    ثابت uint32_t nb00 = src0->nb[0];                   \
    ثابت uint32_t nb01 = src0->nb[1];                   \
    ثابت uint32_t nb02 = src0->nb[2];                   \
    ثابت uint32_t nb03 = src0->nb[3];                   \
                                                         \
    ثابت uint32_t ne10 = src1 ? src1->ne[0] : 0;        \
    ثابت uint32_t ne11 = src1 ? src1->ne[1] : 0;        \
    ثابت uint32_t ne12 = src1 ? src1->ne[2] : 0;        \
    ثابت uint32_t ne13 = src1 ? src1->ne[3] : 0;        \
                                                         \
    ثابت uint32_t nb10 = src1 ? src1->nb[0] : 0;        \
    ثابت uint32_t nb11 = src1 ? src1->nb[1] : 0;        \
    ثابت uint32_t nb12 = src1 ? src1->nb[2] : 0;        \
    ثابت uint32_t nb13 = src1 ? src1->nb[3] : 0;        \
                                                         \
    ثابت uint32_t ne0 = dst->ne[0];                     \
    ثابت uint32_t ne1 = dst->ne[1];                     \
    ثابت uint32_t ne2 = dst->ne[2];                     \
    ثابت uint32_t ne3 = dst->ne[3];                     \
                                                         \
    ثابت uint32_t nb0 = dst->nb[0];                     \
    ثابت uint32_t nb1 = dst->nb[1];                     \
    ثابت uint32_t nb2 = dst->nb[2];                     \
    ثابت uint32_t nb3 = dst->nb[3];

ساخت htp_act_context;

قسمتعریف خالی (*glu_compute_f32_fn_t)(ثابت اعشاری * محدود src0,
                                     ثابت اعشاری * محدود src1,
                                     اعشاری * محدود dst,
                                     ثابت uint32_t عدد_صفیں_2,
                                     ثابت ساخت htp_act_context * actx);

قسمتعریف خالی (*glu_compute_f16_fn_t)(ثابت _Float16 * محدود src0,
                                     ثابت _Float16 * محدود src1,
                                     _Float16 * محدود dst,
                                     ثابت uint32_t عدد_صفیں_2,
                                     ثابت ساخت htp_act_context * actx);

ساخت htp_act_context {
    ساخت htp_ops_context * octx;

    خالی *                   compute;
    ثابت حرف *             op_str;

    // Precomputed values
    dma_addr_t               ڈیٹا_src0;
    dma_addr_t               ڈیٹا_src1;
    dma_addr_t               ڈیٹا_dst;

    size_t                   src0_صف_سائز;
    size_t                   src1_صف_سائز;
    size_t                   dst_صف_سائز;

    size_t                   src0_صف_stride;
    size_t                   src1_صف_stride;

    size_t                   src0_صف_سائز_aligned;
    size_t                   src1_صف_سائز_aligned;
    size_t                   dst_صف_سائز_aligned;

    size_t                   src0_spad_half_سائز;
    size_t                   src1_spad_half_سائز;
    size_t                   dst_spad_half_سائز;

    uint32_t                 بلاک;
    uint32_t                 src0_nrows;
    uint32_t                 src0_nrows_فی_thread;
    uint32_t                 صف_آغاز;
    صحیح                      nc;

    uint8_t *                vtcm_src0;
    uint8_t *                vtcm_src1;
    uint8_t *                vtcm_dst;

    size_t                   vtcm_src0_سائز_فی_thread;
    size_t                   vtcm_src1_سائز_فی_thread;
    size_t                   vtcm_dst_سائز_فی_thread;
};

ساخت htp_act_vtcm_layout {
    size_t کل_بائٹیں;
    size_t off_src0;
    size_t off_src1;
    size_t off_dst;

    size_t src0_بائٹیں_فی_thread;
    size_t src1_بائٹیں_فی_thread;
    size_t dst_بائٹیں_فی_thread;

    uint32_t vtcm_صف_فی_thread;
};

ساکن اندرونی خالی htp_act_vtcm_layout_build(ساخت htp_act_vtcm_layout * L,
                                             size_t                       src0_صف_سائز_aligned,
                                             size_t                       src1_صف_سائز_aligned,
                                             size_t                       dst_صف_سائز_aligned,
                                             uint32_t                     n_دھاگے,
                                             size_t                       vtcm_سائز) {
    ثابت size_t   spad_سائز_فی_صف   = src0_صف_سائز_aligned + src1_صف_سائز_aligned + dst_صف_سائز_aligned;
    ثابت uint32_t vtcm_صف_فی_thread = (uint32_t) (vtcm_سائز / (n_دھاگے * spad_سائز_فی_صف));

    L->vtcm_صف_فی_thread = vtcm_صف_فی_thread;

    L->src0_بائٹیں_فی_thread = src0_صف_سائز_aligned * vtcm_صف_فی_thread;
    L->src1_بائٹیں_فی_thread = src1_صف_سائز_aligned * vtcm_صف_فی_thread;
    L->dst_بائٹیں_فی_thread  = dst_صف_سائز_aligned * vtcm_صف_فی_thread;

    L->off_src0 = 0;
    L->off_src1 = L->off_src0 + L->src0_بائٹیں_فی_thread * n_دھاگے;
    L->off_dst  = L->off_src1 + L->src1_بائٹیں_فی_thread * n_دھاگے;

    L->کل_بائٹیں = L->off_dst + L->dst_بائٹیں_فی_thread * n_دھاگے;
}

#تعریف_کرو htp_glu_op_preamble                                            \
    ثابت size_t src0_صف_سائز_aligned = actx->src0_صف_سائز_aligned;  \
    ثابت size_t src1_صف_سائز_aligned = actx->src1_صف_سائز_aligned;  \
    ثابت size_t dst_صف_سائز_aligned  = actx->dst_صف_سائز_aligned;   \
    ثابت صحیح    nc                    = actx->nc;

// swiglu(x) = x1 * sigmoid(x0)
ساکن خالی swiglu_f32(ثابت اعشاری * محدود src0,
                                                 ثابت اعشاری * محدود src1,
                                                 اعشاری * محدود dst,
                                                 ثابت uint32_t عدد_صفیں_2,
                                                 ثابت ساخت htp_act_context * actx) {
    htp_glu_op_preamble;

    برائے (uint32_t ib = 0; ib < عدد_صفیں_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_صف_سائز_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_صف_سائز_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_صف_سائز_aligned);

        hvx_sigmoid_f32_aa(dst_ptr, src0_ptr, nc);
        hvx_mul_mul_f32_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

// out = x * sigmoid(alpha * x) * (clamp(y, -limit, limit) + 1.f)
ساکن خالی swiglu_oai_f32(ثابت اعشاری * محدود src0,
                                                     ثابت اعشاری * محدود src1,
                                                     اعشاری * محدود dst,
                                                     ثابت uint32_t عدد_صفیں_2,
                                                     ثابت ساخت htp_act_context * actx) {
    htp_glu_op_preamble;
    ثابت اعشاری alpha = ((ثابت اعشاری *) (actx->octx->op_params))[2];
    ثابت اعشاری حد_بندی = ((ثابت اعشاری *) (actx->octx->op_params))[3];

    برائے (uint32_t ib = 0; ib < عدد_صفیں_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_صف_سائز_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_صف_سائز_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_صف_سائز_aligned);

        // x (src0_ptr) = std::min(src0_p[k], limit);
        hvx_min_scalar_f32((uint8_t *) src0_ptr, src0_ptr, حد_بندی, nc);
        // y1 (src1_ptr) = std::clamp(src1_p[k], -limit, limit);
        hvx_clamp_scalar_f32((uint8_t *) src1_ptr, src1_ptr, -حد_بندی, حد_بندی, nc);
        // y (src1_ptr) = y1 + 1.f
        hvx_جوڑو_scalar_f32((uint8_t *) src1_ptr, src1_ptr, 1.0, nc);
        // x1 (dst_ptr) = alpha * x
        hvx_mul_scalar_f32(dst_ptr, src0_ptr, alpha, nc);
        // x2 (dst_ptr) = sigmoid(x1) = 1/(1+exp(-x1))
        hvx_sigmoid_f32_aa(dst_ptr, dst_ptr, nc);
        // out = x * sigmoid(alpha * x) * (y + 1.f)
        hvx_mul_mul_f32_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

ساکن خالی swiglu_clamp_f32(ثابت اعشاری * محدود src0,
                                                       ثابت اعشاری * محدود src1,
                                                       اعشاری * محدود dst,
                                                       ثابت uint32_t                 عدد_صفیں_2,
                                                       ثابت ساخت htp_act_context * actx) {
    htp_glu_op_preamble;
    ثابت اعشاری حد_بندی = ((ثابت اعشاری *) (actx->octx->op_params))[3];

    برائے (uint32_t ib = 0; ib < عدد_صفیں_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_صف_سائز_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_صف_سائز_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_صف_سائز_aligned);

        hvx_min_scalar_f32((uint8_t *) src0_ptr, src0_ptr, حد_بندی, nc);
        hvx_clamp_scalar_f32((uint8_t *) src1_ptr, src1_ptr, -حد_بندی, حد_بندی, nc);
        hvx_sigmoid_f32_aa(dst_ptr, src0_ptr, nc);
        hvx_mul_mul_f32_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

ساکن ثابت اعشاری GELU_COEF_A     = 0.044715f;
ساکن ثابت اعشاری SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;

ساکن اندرونی HVX_سمتیہ hvx_vec_fast_sigmoid_f32_2it(HVX_سمتیہ v) {
    v = Q6_Vqf32_vmpy_VsfVsf(v, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));
    v = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v), Q6_V_vsplat_R(FAST_SIGMOID_C3));

    HVX_سمتیہ اندر_int = hvx_vec_truncate_f32(Q6_Vsf_equals_Vqf32(v));
    HVX_سمتیہ x      = Q6_Vqf32_vsub_Vqf32Vsf(v, Q6_Vsf_equals_Vw(اندر_int));
    HVX_سمتیہ xx     = Q6_Vqf32_vmpy_Vqf32Vqf32(x, x);

    HVX_سمتیہ v1 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(xx), Q6_V_vsplat_R(FAST_SIGMOID_C2));
    v1            = Q6_Vqf32_vadd_Vqf32Vsf(v1, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));

    HVX_سمتیہ v2 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(x), Q6_V_vsplat_R(FAST_SIGMOID_C1));
    v2            = Q6_Vqf32_vmpy_Vqf32Vqf32(v2, xx);
    v2            = Q6_Vqf32_vadd_Vqf32Vqf32(v2, x);

    HVX_سمتیہ v3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v2, v1));
    v3            = Q6_Vw_vaslacc_VwVwR(v3, اندر_int, 24);

    HVX_سمتیہ v4 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(v2, v1));
    HVX_سمتیہ v5 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(v3, v4));

    // Newton-Raphson with 2 iterations
    HVX_سمتیہ two_sf = hvx_vec_splat_f32(2.0f);
    HVX_سمتیہ i_sf   = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x7EEEEBB3), v5);
    HVX_سمتیہ r_qf   = Q6_Vqf32_vmpy_VsfVsf(
        i_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(i_sf, v5)))));
    r_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(
        r_qf, Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(r_qf), v5))));
    HVX_سمتیہ res = Q6_Vsf_equals_Vqf32(r_qf);

    res = Q6_Vqf32_vmpy_VsfVsf(v3, res);

    واپس Q6_Vsf_equals_Vqf32(res);
}

ساکن اندرونی HVX_سمتیہ hvx_vec_fast_sigmoid_f32_guard_2it(HVX_سمتیہ v,
                                                            HVX_سمتیہ one,
                                                            HVX_سمتیہ max_exp,
                                                            HVX_سمتیہ min_exp) {
    ثابت HVX_سمتیہ_Pred pred_max = Q6_Q_vcmp_gt_VsfVsf(max_exp, v);
    ثابت HVX_سمتیہ_Pred pred_min = Q6_Q_vcmp_gt_VsfVsf(v, min_exp);

    HVX_سمتیہ برآمد = hvx_vec_fast_sigmoid_f32_2it(v);
    برآمد            = Q6_V_vmux_QVV(pred_max, برآمد, one);
    واپس Q6_V_vmux_QVV(pred_min, برآمد, Q6_V_vzero());
}

ساکن اندرونی خالی hvx_geglu_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src0, ثابت uint8_t * محدود src1, uint32_t n) {
    أكد((غیرعلامتی طویل) dst  % 128 == 0);
    أكد((غیرعلامتی طویل) src0 % 128 == 0);
    أكد((غیرعلامتی طویل) src1 % 128 == 0);

    HVX_سمتیہ * محدود vdst        = (HVX_سمتیہ *) dst;
    ثابت HVX_سمتیہ * محدود vsrc0 = (ثابت HVX_سمتیہ *) src0;
    ثابت HVX_سمتیہ * محدود vsrc1 = (ثابت HVX_سمتیہ *) src1;

    ثابت uint32_t epv  = 128 / حجم(اعشاری);
    ثابت uint32_t nvec = n / epv;
    ثابت uint32_t nloe = n % epv;

    ثابت اعشاری GELU_COEF_A_TIMES_SQRT = GELU_COEF_A * SQRT_2_OVER_PI;

    ثابت HVX_سمتیہ v_coef_a_times_sqrt = hvx_vec_splat_f32(GELU_COEF_A_TIMES_SQRT);
    ثابت HVX_سمتیہ v_sqrt_2_pi         = hvx_vec_splat_f32(SQRT_2_OVER_PI);
    ثابت HVX_سمتیہ v_one               = hvx_vec_splat_f32(1.0f);
    ثابت HVX_سمتیہ v_max_exp           = hvx_vec_splat_f32(87.0f);
    ثابت HVX_سمتیہ v_min_exp           = hvx_vec_splat_f32(-87.0f);

    uint32_t i = 0;

    _Pragma("unroll(4)")
    برائے (; i < nvec; i++) {
        HVX_سمتیہ x = vsrc0[i];
        HVX_سمتیہ g = vsrc1[i];

        HVX_سمتیہ x2 = hvx_vec_mul_f32_f32(x, x);
        HVX_سمتیہ coef = hvx_vec_mul_f32_f32(x2, v_coef_a_times_sqrt);
        coef = hvx_vec_جوڑو_f32_f32(coef, v_sqrt_2_pi);
        HVX_سمتیہ inner = hvx_vec_mul_f32_f32(x, coef);

        // y2 = 2 * inner = inner + inner
        HVX_سمتیہ y2 = hvx_vec_جوڑو_f32_f32(inner, inner);

        // Fast sigmoid approximation (2 iterations)
        HVX_سمتیہ sig2y = hvx_vec_fast_sigmoid_f32_guard_2it(y2, v_one, v_max_exp, v_min_exp);

        HVX_سمتیہ gelu_x = hvx_vec_mul_f32_f32(x, sig2y);
        vdst[i] = hvx_vec_mul_f32_f32(gelu_x, g);
    }

    اگر (nloe) {
        HVX_سمتیہ x = vsrc0[i];
        HVX_سمتیہ g = vsrc1[i];

        HVX_سمتیہ x2 = hvx_vec_mul_f32_f32(x, x);
        HVX_سمتیہ coef = hvx_vec_mul_f32_f32(x2, v_coef_a_times_sqrt);
        coef = hvx_vec_جوڑو_f32_f32(coef, v_sqrt_2_pi);
        HVX_سمتیہ inner = hvx_vec_mul_f32_f32(x, coef);

        HVX_سمتیہ y2 = hvx_vec_جوڑو_f32_f32(inner, inner);

        HVX_سمتیہ sig2y = hvx_vec_fast_sigmoid_f32_guard_2it(y2, v_one, v_max_exp, v_min_exp);

        HVX_سمتیہ gelu_x = hvx_vec_mul_f32_f32(x, sig2y);
        HVX_سمتیہ res = hvx_vec_mul_f32_f32(gelu_x, g);
        hvx_vec_store_a((خالی *) &vdst[i], nloe * حجم(اعشاری), res);
    }
}

ساکن اندرونی خالی hvx_geglu_تیز_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src0, ثابت uint8_t * محدود src1, uint32_t n) {
    أكد((غیرعلامتی طویل) dst  % 128 == 0);
    أكد((غیرعلامتی طویل) src0 % 128 == 0);
    أكد((غیرعلامتی طویل) src1 % 128 == 0);

    HVX_سمتیہ * محدود vdst        = (HVX_سمتیہ *) dst;
    ثابت HVX_سمتیہ * محدود vsrc0 = (ثابت HVX_سمتیہ *) src0;
    ثابت HVX_سمتیہ * محدود vsrc1 = (ثابت HVX_سمتیہ *) src1;

    ثابت uint32_t epv  = 128 / حجم(اعشاری);
    ثابت uint32_t nvec = n / epv;
    ثابت uint32_t nloe = n % epv;

    ثابت HVX_سمتیہ v_scale    = hvx_vec_splat_f32(1.702f);
    ثابت HVX_سمتیہ v_one      = hvx_vec_splat_f32(1.0f);
    ثابت HVX_سمتیہ v_max_exp  = hvx_vec_splat_f32(87.0f);
    ثابت HVX_سمتیہ v_min_exp  = hvx_vec_splat_f32(-87.0f);

    uint32_t i = 0;

    _Pragma("unroll(4)")
    برائے (; i < nvec; i++) {
        HVX_سمتیہ x = vsrc0[i];
        HVX_سمتیہ g = vsrc1[i];
        HVX_سمتیہ scaled_x = hvx_vec_mul_f32_f32(x, v_scale);
        HVX_سمتیہ sigmoid_x = hvx_vec_fast_sigmoid_f32_guard_2it(scaled_x, v_one, v_max_exp, v_min_exp);
        vdst[i] = hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(x, sigmoid_x), g);
    }

    اگر (nloe) {
        HVX_سمتیہ x = vsrc0[i];
        HVX_سمتیہ g = vsrc1[i];
        HVX_سمتیہ scaled_x = hvx_vec_mul_f32_f32(x, v_scale);
        HVX_سمتیہ sigmoid_x = hvx_vec_fast_sigmoid_f32_guard_2it(scaled_x, v_one, v_max_exp, v_min_exp);
        HVX_سمتیہ نتیجہ = hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(x, sigmoid_x), g);
        hvx_vec_store_a((خالی *) &vdst[i], nloe * حجم(اعشاری), نتیجہ);
    }
}

ساکن اندرونی خالی hvx_geglu_erf_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src0, ثابت uint8_t * محدود src1, uint32_t n) {
    أكد((غیرعلامتی طویل) dst  % 128 == 0);
    أكد((غیرعلامتی طویل) src0 % 128 == 0);
    أكد((غیرعلامتی طویل) src1 % 128 == 0);

    HVX_سمتیہ * محدود vdst        = (HVX_سمتیہ *) dst;
    ثابت HVX_سمتیہ * محدود vsrc0 = (ثابت HVX_سمتیہ *) src0;
    ثابت HVX_سمتیہ * محدود vsrc1 = (ثابت HVX_سمتیہ *) src1;

    ثابت uint32_t epv  = 128 / حجم(اعشاری);
    ثابت uint32_t nvec = n / epv;
    ثابت uint32_t nloe = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    برائے (; i < nvec; i++) {
        HVX_سمتیہ x = vsrc0[i];
        HVX_سمتیہ g = vsrc1[i];
        vdst[i] = hvx_vec_mul_f32_f32(hvx_vec_gelu_erf_f32(x), g);
    }

    اگر (nloe) {
        HVX_سمتیہ x = vsrc0[i];
        HVX_سمتیہ g = vsrc1[i];
        HVX_سمتیہ نتیجہ = hvx_vec_mul_f32_f32(hvx_vec_gelu_erf_f32(x), g);
        hvx_vec_store_a((خالی *) &vdst[i], nloe * حجم(اعشاری), نتیجہ);
    }
}

ساکن اندرونی خالی hvx_geglu_f16_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src0, ثابت uint8_t * محدود src1, uint32_t n) {
    أكد((غیرعلامتی طویل) dst  % 128 == 0);
    أكد((غیرعلامتی طویل) src0 % 128 == 0);
    أكد((غیرعلامتی طویل) src1 % 128 == 0);

    HVX_سمتیہ * محدود vdst        = (HVX_سمتیہ *) dst;
    ثابت HVX_سمتیہ * محدود vsrc0 = (ثابت HVX_سمتیہ *) src0;
    ثابت HVX_سمتیہ * محدود vsrc1 = (ثابت HVX_سمتیہ *) src1;

    ثابت uint32_t epv  = 128 / حجم(_Float16);
    ثابت uint32_t nvec = n / epv;
    ثابت uint32_t nloe = n % epv;

    ثابت HVX_سمتیہ v_coef_a    = hvx_vec_splat_f16((_Float16) GELU_COEF_A);
    ثابت HVX_سمتیہ v_sqrt_2_pi = hvx_vec_splat_f16((_Float16) SQRT_2_OVER_PI);
    ثابت HVX_سمتیہ v_one       = hvx_vec_splat_f16((_Float16) 1.0f);
    ثابت HVX_سمتیہ v_half      = hvx_vec_splat_f16((_Float16) 0.5f);

    uint32_t i = 0;

    _Pragma("unroll(4)")
    برائے (; i < nvec; i++) {
        HVX_سمتیہ x = vsrc0[i];
        HVX_سمتیہ g = vsrc1[i];

        // gelu(x) = 0.5*x*(1 + tanh(SQRT_2_OVER_PI*x*(1 + GELU_COEF_A*x*x)))
        HVX_سمتیہ x2    = hvx_vec_mul_f16_f16(x, x);
        HVX_سمتیہ coef  = hvx_vec_جوڑو_f16_f16(hvx_vec_mul_f16_f16(x2, v_coef_a), v_one);
        HVX_سمتیہ inner = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(x, coef), v_sqrt_2_pi);

        HVX_سمتیہ t = hvx_vec_جوڑو_f16_f16(hvx_vec_tanh_f16(inner), v_one);

        HVX_سمتیہ gelu_x = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(x, t), v_half);
        vdst[i] = hvx_vec_mul_f16_f16(gelu_x, g);
    }

    اگر (nloe) {
        HVX_سمتیہ x = vsrc0[i];
        HVX_سمتیہ g = vsrc1[i];

        HVX_سمتیہ x2    = hvx_vec_mul_f16_f16(x, x);
        HVX_سمتیہ coef  = hvx_vec_جوڑو_f16_f16(hvx_vec_mul_f16_f16(x2, v_coef_a), v_one);
        HVX_سمتیہ inner = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(x, coef), v_sqrt_2_pi);

        HVX_سمتیہ t = hvx_vec_جوڑو_f16_f16(hvx_vec_tanh_f16(inner), v_one);

        HVX_سمتیہ gelu_x = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(x, t), v_half);
        HVX_سمتیہ res    = hvx_vec_mul_f16_f16(gelu_x, g);
        hvx_vec_store_a((خالی *) &vdst[i], nloe * حجم(_Float16), res);
    }
}

// geglu(x, g) = gelu(x) * g
ساکن خالی geglu_f32(ثابت اعشاری * محدود src0,
                                                ثابت اعشاری * محدود src1,
                                                اعشاری * محدود dst,
                                                ثابت uint32_t عدد_صفیں_2,
                                                ثابت ساخت htp_act_context * actx) {
    htp_glu_op_preamble;

    برائے (uint32_t ib = 0; ib < عدد_صفیں_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_صف_سائز_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_صف_سائز_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_صف_سائز_aligned);

        hvx_geglu_f32_aa(dst_ptr, src0_ptr, src1_ptr, nc);
    }
}

// geglu_quick(x, g) = x * sigmoid(1.702 * x) * g
ساکن خالی geglu_تیز_f32(ثابت اعشاری * محدود src0,
                            ثابت اعشاری * محدود src1,
                            اعشاری * محدود dst,
                            ثابت uint32_t عدد_صفیں_2,
                            ثابت ساخت htp_act_context * actx) {
    htp_glu_op_preamble;

    برائے (uint32_t ib = 0; ib < عدد_صفیں_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_صف_سائز_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_صف_سائز_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_صف_سائز_aligned);

        hvx_geglu_تیز_f32_aa(dst_ptr, src0_ptr, src1_ptr, nc);
    }
}

// geglu_erf(x, g) = gelu_erf(x) * g
ساکن خالی geglu_erf_f32(ثابت اعشاری * محدود src0,
                          ثابت اعشاری * محدود src1,
                          اعشاری * محدود dst,
                          ثابت uint32_t عدد_صفیں_2,
                          ثابت ساخت htp_act_context * actx) {
    htp_glu_op_preamble;

    برائے (uint32_t ib = 0; ib < عدد_صفیں_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_صف_سائز_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_صف_سائز_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_صف_سائز_aligned);

        hvx_geglu_erf_f32_aa(dst_ptr, src0_ptr, src1_ptr, nc);
    }
}

// swiglu(x) = x1 * sigmoid(x0)
ساکن خالی swiglu_f16(ثابت _Float16 * محدود src0,
                       ثابت _Float16 * محدود src1,
                       _Float16 * محدود dst,
                       ثابت uint32_t عدد_صفیں_2,
                       ثابت ساخت htp_act_context * actx) {
    htp_glu_op_preamble;

    برائے (uint32_t ib = 0; ib < عدد_صفیں_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_صف_سائز_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_صف_سائز_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_صف_سائز_aligned);

        hvx_sigmoid_f16_aa(dst_ptr, src0_ptr, nc);
        hvx_mul_mul_f16_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

// out = x * sigmoid(alpha * x) * (clamp(y, -limit, limit) + 1.f)
ساکن خالی swiglu_oai_f16(ثابت _Float16 * محدود src0,
                           ثابت _Float16 * محدود src1,
                           _Float16 * محدود dst,
                           ثابت uint32_t عدد_صفیں_2,
                           ثابت ساخت htp_act_context * actx) {
    htp_glu_op_preamble;
    ثابت _Float16 alpha = (_Float16) ((ثابت اعشاری *) (actx->octx->op_params))[2];
    ثابت _Float16 حد_بندی = (_Float16) ((ثابت اعشاری *) (actx->octx->op_params))[3];

    برائے (uint32_t ib = 0; ib < عدد_صفیں_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_صف_سائز_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_صف_سائز_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_صف_سائز_aligned);

        // x (src0_ptr) = std::min(src0_p[k], limit);
        hvx_min_scalar_f16((uint8_t *) src0_ptr, src0_ptr, حد_بندی, nc);
        // y1 (src1_ptr) = std::clamp(src1_p[k], -limit, limit);
        hvx_clamp_scalar_f16((uint8_t *) src1_ptr, src1_ptr, -حد_بندی, حد_بندی, nc);
        // y (src1_ptr) = y1 + 1.f
        hvx_جوڑو_scalar_f16((uint8_t *) src1_ptr, src1_ptr, (_Float16) 1.0f, nc);
        // x1 (dst_ptr) = alpha * x
        hvx_mul_scalar_f16(dst_ptr, src0_ptr, alpha, nc);
        // x2 (dst_ptr) = sigmoid(x1) = 1/(1+exp(-x1))
        hvx_sigmoid_f16_aa(dst_ptr, dst_ptr, nc);
        // out = x * sigmoid(alpha * x) * (y + 1.f)
        hvx_mul_mul_f16_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

ساکن خالی swiglu_clamp_f16(ثابت _Float16 * محدود src0,
                             ثابت _Float16 * محدود src1,
                             _Float16 * محدود dst,
                             ثابت uint32_t                 عدد_صفیں_2,
                             ثابت ساخت htp_act_context * actx) {
    htp_glu_op_preamble;
    ثابت _Float16 حد_بندی = (_Float16) ((ثابت اعشاری *) (actx->octx->op_params))[3];

    برائے (uint32_t ib = 0; ib < عدد_صفیں_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_صف_سائز_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_صف_سائز_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_صف_سائز_aligned);

        hvx_min_scalar_f16((uint8_t *) src0_ptr, src0_ptr, حد_بندی, nc);
        hvx_clamp_scalar_f16((uint8_t *) src1_ptr, src1_ptr, -حد_بندی, حد_بندی, nc);
        hvx_sigmoid_f16_aa(dst_ptr, src0_ptr, nc);
        hvx_mul_mul_f16_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

// geglu(x, g) = gelu(x) * g, gelu via tanh approximation
ساکن خالی geglu_f16(ثابت _Float16 * محدود src0,
                      ثابت _Float16 * محدود src1,
                      _Float16 * محدود dst,
                      ثابت uint32_t عدد_صفیں_2,
                      ثابت ساخت htp_act_context * actx) {
    htp_glu_op_preamble;

    برائے (uint32_t ib = 0; ib < عدد_صفیں_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_صف_سائز_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_صف_سائز_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_صف_سائز_aligned);

        hvx_geglu_f16_aa(dst_ptr, src0_ptr, src1_ptr, nc);
    }
}

#تعریف_کرو DEFINE_GLU_فی_THREAD(TYPE, SUFFIX)                                                                            \
    ساکن خالی glu_##SUFFIX##_per_thread(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {                           \
        ساخت htp_act_context * actx = (ساخت htp_act_context *) ڈیٹا;                                               \
        htp_act_preamble;                                                                                              \
                                                                                                                       \
        ساخت htp_thread_trace * tr = actx->octx->ctx ? &actx->octx->ctx->trace[ith] : تہی;                          \
                                                                                                                       \
        size_t src0_صف_سائز = actx->src0_صف_سائز;                                                                    \
        size_t src1_صف_سائز = actx->src1_صف_سائز;                                                                    \
        size_t dst_صف_سائز  = actx->dst_صف_سائز;                                                                     \
                                                                                                                       \
        size_t src0_صف_stride = actx->src0_صف_stride;                                                                \
        size_t src1_صف_stride = actx->src1_صف_stride;                                                                \
                                                                                                                       \
        ثابت uint32_t src0_nrows            = actx->src0_nrows;                                                       \
        ثابت uint32_t src0_nrows_فی_thread = actx->src0_nrows_فی_thread;                                            \
                                                                                                                       \
        ثابت uint32_t src0_آغاز_صف = actx->صف_آغاز + src0_nrows_فی_thread * ith;                                 \
        ثابت uint32_t src0_انتہا_صف   = MIN(src0_آغاز_صف + src0_nrows_فی_thread, actx->صف_آغاز + src0_nrows);     \
                                                                                                                       \
        /* no work for this thread */                                                                                  \
        اگر (src0_آغاز_صف >= src0_انتہا_صف) {                                                                          \
            واپس;                                                                                                    \
        }                                                                                                              \
                                                                                                                       \
        ثابت dma_addr_t ڈیٹا_src0 = actx->ڈیٹا_src0;                                                                  \
        ثابت dma_addr_t ڈیٹا_src1 = actx->ڈیٹا_src1;                                                                  \
        ثابت dma_addr_t ڈیٹا_dst  = actx->ڈیٹا_dst;                                                                   \
                                                                                                                       \
        ثابت size_t src0_صف_سائز_aligned = actx->src0_صف_سائز_aligned;                                              \
        ثابت size_t src1_صف_سائز_aligned = actx->src1_صف_سائز_aligned;                                              \
        ثابت size_t dst_صف_سائز_aligned  = actx->dst_صف_سائز_aligned;                                               \
                                                                                                                       \
        uint8_t * محدود src0_spad_ڈیٹا = actx->vtcm_src0 + (ith * actx->vtcm_src0_سائز_فی_thread);                 \
        uint8_t * محدود src1_spad_ڈیٹا = actx->vtcm_src1 + (ith * actx->vtcm_src1_سائز_فی_thread);                 \
        uint8_t * محدود dst_spad_ڈیٹا  = actx->vtcm_dst  + (ith * actx->vtcm_dst_سائز_فی_thread);                  \
                                                                                                                       \
        size_t src0_spad_half_سائز = actx->src0_spad_half_سائز;                                                        \
        size_t src1_spad_half_سائز = actx->src1_spad_half_سائز;                                                        \
        size_t dst_spad_half_سائز  = actx->dst_spad_half_سائز;                                                         \
                                                                                                                       \
        ثابت صحیح بلاک_2 = actx->بلاک;                                                                                 \
        اگر (بلاک_2 == 0) {                                                                                              \
            FARF(ERROR, "%s : VTCM reservation %zu is too small, needed %zu\n",                                        \
                 actx->op_str, actx->vtcm_src0_سائز_فی_thread, src0_صف_سائز_aligned);                                \
            واپس;                                                                                                    \
        }                                                                                                              \
                                                                                                                       \
        dma_قطار * dma_q = actx->octx->ctx->dma[ith];                                                                 \
        glu_compute_##SUFFIX##_fn_t compute = (glu_compute_##SUFFIX##_fn_t) actx->compute;                             \
                                                                                                                       \
        برائے (uint32_t ir = src0_آغاز_صف, spad_idx = 0; ir < src0_انتہا_صف && spad_idx < 2; ir += بلاک_2, spad_idx++) { \
            ثابت uint32_t بلاک_سائز_3 = MIN(بلاک_2, src0_انتہا_صف - ir);                                                 \
                                                                                                                       \
            /* Dummy DMA transation for sequencing (interleaving dst,src,dst,...) */                                   \
            dma_قطار_ڈالو(dma_q,                                                                                      \
                           dma_make_ڈیٹا(ڈیٹا_dst, dst_spad_ڈیٹا + (spad_idx * dst_spad_half_سائز)),                   \
                           dst_صف_سائز, dst_صف_سائز_aligned, dst_صف_سائز, 0);                                       \
                                                                                                                       \
            dma_قطار_ڈالو(dma_q,                                                                                      \
                dma_make_ڈیٹا(src0_spad_ڈیٹا + (spad_idx * src0_spad_half_سائز), ڈیٹا_src0 + (ir * src0_صف_stride)),  \
                src0_صف_سائز_aligned, src0_صف_stride, src0_صف_سائز, بلاک_سائز_3);                                    \
                                                                                                                       \
            dma_قطار_ڈالو(dma_q,                                                                                      \
                dma_make_ڈیٹا(src1_spad_ڈیٹا + (spad_idx * src1_spad_half_سائز), ڈیٹا_src1 + (ir * src1_صف_stride)),  \
                src1_صف_سائز_aligned, src1_صف_stride, src1_صف_سائز, بلاک_سائز_3);                                    \
        }                                                                                                              \
                                                                                                                       \
        برائے (uint32_t ir = src0_آغاز_صف; ir < src0_انتہا_صف; ir += بلاک_2) {                                           \
            ثابت uint32_t بلاک_سائز_3 = MIN(بلاک_2, src0_انتہا_صف - ir);                                                 \
                                                                                                                       \
            TYPE * dst_spad  = (TYPE *) dma_قطار_نکالو(dma_q).src;                                                      \
            TYPE * src0_spad = (TYPE *) dma_قطار_نکالو(dma_q).dst;                                                      \
            TYPE * src1_spad = (TYPE *) dma_قطار_نکالو(dma_q).dst;                                                      \
                                                                                                                       \
            htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, ir);                                                     \
            compute(src0_spad, src1_spad, dst_spad, بلاک_سائز_3, actx);                                                 \
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir);                                                      \
                                                                                                                       \
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(ڈیٹا_dst + (ir * dst_صف_سائز), dst_spad),                             \
                           dst_صف_سائز, dst_صف_سائز_aligned, dst_صف_سائز, بلاک_سائز_3);                              \
                                                                                                                       \
            /* prefetch N+2 loop iteration if any */                                                                   \
            ثابت uint32_t pref_بلاک = (ir + بلاک_2 * 2);                                                              \
            اگر (pref_بلاک < src0_انتہا_صف) {                                                                           \
                ثابت uint32_t pref_بلاک_سائز = MIN(بلاک_2, src0_انتہا_صف - pref_بلاک);                                \
                dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(src0_spad, ڈیٹا_src0 + (pref_بلاک * src0_صف_stride)),            \
                               src0_صف_سائز_aligned, src0_صف_stride, src0_صف_سائز, pref_بلاک_سائز);                \
                dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(src1_spad, ڈیٹا_src1 + (pref_بلاک * src1_صف_stride)),            \
                               src1_صف_سائز_aligned, src1_صف_stride, src1_صف_سائز, pref_بلاک_سائز);                \
            }                                                                                                          \
        }                                                                                                              \
                                                                                                                       \
        dma_قطار_flush(dma_q);                                                                                        \
    }

DEFINE_GLU_فی_THREAD(اعشاری, f32)
DEFINE_GLU_فی_THREAD(_Float16, f16)

ساکن صحیح execute_op_activations_f32(ساخت htp_ops_context * octx) {
    ثابت ساخت htp_tensor * src0 = octx->src[0];
    ثابت ساخت htp_tensor * src1 = octx->src[1];
    ثابت ساخت htp_tensor * dst  = octx->dst;

    اگر ((dst->ne[0] * SIZEOF_FP32) != dst->nb[1]) {
        FARF(ERROR, "Non-contiguous dst is not supported at this time \n");
        واپس HTP_حالت_NO_SUPPORT;
    }

    glu_compute_f32_fn_t compute_fn = تہی;
    ثابت حرف *         op_type    = تہی;

    انتخاب (octx->op) {
        حالت HTP_OP_GLU_SWIGLU:
            compute_fn = swiglu_f32;
            op_type    = "swiglu-f32";
            توڑو;

        حالت HTP_OP_GLU_SWIGLU_OAI:
            compute_fn = swiglu_oai_f32;
            op_type    = "swiglu-oai-f32";
            توڑو;

        حالت HTP_OP_GLU_SWIGLU_CLAMP:
            compute_fn = swiglu_clamp_f32;
            op_type    = "swiglu-clamp-f32";
            توڑو;

        حالت HTP_OP_GLU_GEGLU:
            compute_fn = geglu_f32;
            op_type    = "geglu-f32";
            توڑو;

        حالت HTP_OP_GLU_GEGLU_تیز:
            compute_fn = geglu_تیز_f32;
            op_type    = "geglu-quick-f32";
            توڑو;

        حالت HTP_OP_GLU_GEGLU_ERF:
            compute_fn = geglu_erf_f32;
            op_type    = "geglu-erf-f32";
            توڑو;

        طےشدہ:
            FARF(ERROR, "Unsupported activations Op %u\n", octx->op);
            واپس HTP_حالت_NO_SUPPORT;
    }

    ثابت uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت size_t dst_صف_سائز = dst->ne[0] * SIZEOF_FP32;

    uint32_t صف_آغاز = 0;
    uint32_t nrows     = src0_nrows;

    اگر (octx->ctx->mdev.گنتی > 1) {
        uint32_t صفیں_فی_chunk = 0;
        htp_tensor_mdev_صفیں_فی_chunk(dst, حجم(اعشاری), (uint32_t) dst_صف_سائز, &صفیں_فی_chunk);
        ثابت ساخت htp_tensor_mdev_range range = htp_tensor_mdev_تقسیم(src0_nrows, صفیں_فی_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
        صف_آغاز = range.آغاز;
        nrows     = range.گنتی;
    }

    اگر (nrows == 0) {
        واپس HTP_حالت_OK;
    }

    ثابت uint32_t n_دھاگے = octx->n_دھاگے;

    // row_size   = bytes of useful data per row (what the kernel touches / what DMA copies).
    // row_stride = bytes between successive rows in DDR (may exceed row_size for non-contig src).
    ثابت size_t nc_بائٹیں        = dst_صف_سائز;
    ثابت size_t src0_صف_سائز   = nc_بائٹیں;
    ثابت size_t src1_صف_سائز   = nc_بائٹیں;
    ثابت size_t src0_صف_stride = src0->nb[1];
    ثابت size_t src1_صف_stride = src1 ? src1->nb[1] : src0->nb[1];

    ثابت size_t src0_صف_سائز_aligned = hex_باری_اوپر(src0_صف_سائز, VLEN);
    ثابت size_t src1_صف_سائز_aligned = hex_باری_اوپر(src1_صف_سائز, VLEN);
    ثابت size_t dst_صف_سائز_aligned  = hex_باری_اوپر(dst_صف_سائز, VLEN);

    ساخت htp_act_vtcm_layout L;
    htp_act_vtcm_layout_build(&L, src0_صف_سائز_aligned, src1_صف_سائز_aligned, dst_صف_سائز_aligned, n_دھاگے,
                              octx->ctx->vtcm_سائز);

    // Make sure the reserved vtcm size is sufficient
    اگر (L.vtcm_صف_فی_thread == 0) {
        FARF(ERROR, "act-%s : current VTCM reservation %zu is too small for even 1 row per thread, needed at least %zu\n", op_type, octx->ctx->vtcm_سائز,
             (src0_صف_سائز_aligned + src1_صف_سائز_aligned + dst_صف_سائز_aligned) * n_دھاگے);
        واپس HTP_حالت_VTCM_TOO_SMALL;
    }

    اگر (src1) {
        FARF(بلند_2, "%s: %ux%ux%ux%u x %ux%ux%ux%u -> %ux%ux%ux%u : src0-vtcm-size %zu src1-vtcm-size %zu dst-vtcm-size %zu\n",
             op_type, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2],
             src1->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], L.src0_بائٹیں_فی_thread * n_دھاگے,
             L.src1_بائٹیں_فی_thread * n_دھاگے, L.dst_بائٹیں_فی_thread * n_دھاگے);
    } ورنہ {
        FARF(بلند_2, "%s: %ux%ux%ux%u -> %ux%ux%ux%u : src0-vtcm-size %zu src1-vtcm-size %zu dst-vtcm-size %zu\n", op_type,
             src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
             L.src0_بائٹیں_فی_thread * n_دھاگے, L.src1_بائٹیں_فی_thread * n_دھاگے, L.dst_بائٹیں_فی_thread * n_دھاگے);
    }

    // Prepare context
    ساخت htp_act_context actx;
    actx.octx    = octx;
    actx.compute = (خالی *) compute_fn;
    actx.op_str  = op_type;

    actx.src0_nrows_فی_thread = fastdiv(nrows + n_دھاگے - 1, &octx->n_دھاگے_div);

    actx.src0_صف_سائز = src0_صف_سائز;
    actx.src1_صف_سائز = src1_صف_سائز;
    actx.dst_صف_سائز  = dst_صف_سائز;

    actx.src0_صف_سائز_aligned = src0_صف_سائز_aligned;
    actx.src1_صف_سائز_aligned = src1_صف_سائز_aligned;
    actx.dst_صف_سائز_aligned  = dst_صف_سائز_aligned;

    actx.src0_صف_stride = src0_صف_stride;
    actx.src1_صف_stride = src1_صف_stride;

    uint8_t * ثابت بنیاد = (uint8_t *) octx->ctx->vtcm_بنیاد;
    actx.vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, بنیاد, L.off_src0);
    actx.vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, بنیاد, L.off_src1);
    actx.vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, بنیاد, L.off_dst);

    actx.vtcm_src0_سائز_فی_thread = L.src0_بائٹیں_فی_thread;
    actx.vtcm_src1_سائز_فی_thread = L.src1_بائٹیں_فی_thread;
    actx.vtcm_dst_سائز_فی_thread  = L.dst_بائٹیں_فی_thread;

    actx.src0_spad_half_سائز = L.src0_بائٹیں_فی_thread / 2;
    actx.src1_spad_half_سائز = L.src1_بائٹیں_فی_thread / 2;
    actx.dst_spad_half_سائز  = L.dst_بائٹیں_فی_thread / 2;

    actx.بلاک = actx.src0_spad_half_سائز / actx.src0_صف_سائز_aligned;
    actx.src0_nrows = nrows;
    actx.صف_آغاز  = صف_آغاز;

    actx.nc = dst->ne[0];

    // Addresses and GLU logic
    dma_addr_t ڈیٹا_src0 = src0->ڈیٹا;
    dma_addr_t ڈیٹا_src1 = src1 ? src1->ڈیٹا : 0;

    اگر (!src1 && (octx->op == HTP_OP_GLU_SWIGLU ||
                  octx->op == HTP_OP_GLU_SWIGLU_OAI ||
                  octx->op == HTP_OP_GLU_SWIGLU_CLAMP ||
                  octx->op == HTP_OP_GLU_GEGLU ||
                  octx->op == HTP_OP_GLU_GEGLU_تیز ||
                  octx->op == HTP_OP_GLU_GEGLU_ERF)) {
         ثابت int32_t swapped = octx->op_params[1];
         ڈیٹا_src1 = ڈیٹا_src0;
         actx.src1_صف_سائز = actx.src0_صف_سائز;

         size_t nc_اندر_بائٹیں = actx.nc * SIZEOF_FP32;
         اگر (swapped) {
             ڈیٹا_src0 += nc_اندر_بائٹیں;
         } ورنہ {
             ڈیٹا_src1 += nc_اندر_بائٹیں;
         }
    }

    actx.ڈیٹا_src0 = ڈیٹا_src0;
    actx.ڈیٹا_src1 = ڈیٹا_src1;
    actx.ڈیٹا_dst  = dst->ڈیٹا;

    work_قطار_تسلسل(octx->ctx->work_قطار, (کارکن_callback_t)glu_f32_فی_thread, &actx, n_دھاگے);
    واپس HTP_حالت_OK;
}

ساکن صحیح execute_op_activations_f16(ساخت htp_ops_context * octx) {
    ثابت ساخت htp_tensor * src0 = octx->src[0];
    ثابت ساخت htp_tensor * src1 = octx->src[1];
    ثابت ساخت htp_tensor * dst  = octx->dst;

    اگر ((dst->ne[0] * SIZEOF_FP16) != dst->nb[1]) {
        FARF(ERROR, "Non-contiguous dst is not supported at this time \n");
        واپس HTP_حالت_NO_SUPPORT;
    }

    glu_compute_f16_fn_t compute_fn = تہی;
    ثابت حرف *         op_type    = تہی;

    انتخاب (octx->op) {
        حالت HTP_OP_GLU_SWIGLU:
            compute_fn = swiglu_f16;
            op_type    = "swiglu-f16";
            توڑو;

        حالت HTP_OP_GLU_SWIGLU_OAI:
            compute_fn = swiglu_oai_f16;
            op_type    = "swiglu-oai-f16";
            توڑو;

        حالت HTP_OP_GLU_SWIGLU_CLAMP:
            compute_fn = swiglu_clamp_f16;
            op_type    = "swiglu-clamp-f16";
            توڑو;

        حالت HTP_OP_GLU_GEGLU:
            compute_fn = geglu_f16;
            op_type    = "geglu-f16";
            توڑو;

        طےشدہ:
            FARF(ERROR, "Unsupported activations Op %u\n", octx->op);
            واپس HTP_حالت_NO_SUPPORT;
    }

    ثابت uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت size_t dst_صف_سائز = dst->ne[0] * SIZEOF_FP16;

    uint32_t صف_آغاز = 0;
    uint32_t nrows     = src0_nrows;

    اگر (octx->ctx->mdev.گنتی > 1) {
        uint32_t صفیں_فی_chunk = 0;
        htp_tensor_mdev_صفیں_فی_chunk(dst, حجم(_Float16), (uint32_t) dst_صف_سائز, &صفیں_فی_chunk);
        ثابت ساخت htp_tensor_mdev_range range = htp_tensor_mdev_تقسیم(src0_nrows, صفیں_فی_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
        صف_آغاز = range.آغاز;
        nrows     = range.گنتی;
    }

    اگر (nrows == 0) {
        واپس HTP_حالت_OK;
    }

    ثابت uint32_t n_دھاگے = octx->n_دھاگے;

    // row_size   = bytes of useful data per row (what the kernel touches / what DMA copies).
    // row_stride = bytes between successive rows in DDR (may exceed row_size for non-contig src).
    ثابت size_t nc_بائٹیں        = dst_صف_سائز;
    ثابت size_t src0_صف_سائز   = nc_بائٹیں;
    ثابت size_t src1_صف_سائز   = nc_بائٹیں;
    ثابت size_t src0_صف_stride = src0->nb[1];
    ثابت size_t src1_صف_stride = src1 ? src1->nb[1] : src0->nb[1];

    ثابت size_t src0_صف_سائز_aligned = hex_باری_اوپر(src0_صف_سائز, VLEN);
    ثابت size_t src1_صف_سائز_aligned = hex_باری_اوپر(src1_صف_سائز, VLEN);
    ثابت size_t dst_صف_سائز_aligned  = hex_باری_اوپر(dst_صف_سائز, VLEN);

    ساخت htp_act_vtcm_layout L;
    htp_act_vtcm_layout_build(&L, src0_صف_سائز_aligned, src1_صف_سائز_aligned, dst_صف_سائز_aligned, n_دھاگے,
                              octx->ctx->vtcm_سائز);

    // Make sure the reserved vtcm size is sufficient
    اگر (L.vtcm_صف_فی_thread == 0) {
        FARF(ERROR, "act-%s : current VTCM reservation %zu is too small for even 1 row per thread, needed at least %zu\n", op_type, octx->ctx->vtcm_سائز,
             (src0_صف_سائز_aligned + src1_صف_سائز_aligned + dst_صف_سائز_aligned) * n_دھاگے);
        واپس HTP_حالت_VTCM_TOO_SMALL;
    }

    اگر (src1) {
        FARF(بلند_2, "%s: %ux%ux%ux%u x %ux%ux%ux%u -> %ux%ux%ux%u : src0-vtcm-size %zu src1-vtcm-size %zu dst-vtcm-size %zu\n",
             op_type, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2],
             src1->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], L.src0_بائٹیں_فی_thread * n_دھاگے,
             L.src1_بائٹیں_فی_thread * n_دھاگے, L.dst_بائٹیں_فی_thread * n_دھاگے);
    } ورنہ {
        FARF(بلند_2, "%s: %ux%ux%ux%u -> %ux%ux%ux%u : src0-vtcm-size %zu src1-vtcm-size %zu dst-vtcm-size %zu\n", op_type,
             src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
             L.src0_بائٹیں_فی_thread * n_دھاگے, L.src1_بائٹیں_فی_thread * n_دھاگے, L.dst_بائٹیں_فی_thread * n_دھاگے);
    }

    // Prepare context
    ساخت htp_act_context actx;
    actx.octx    = octx;
    actx.compute = (خالی *) compute_fn;
    actx.op_str  = op_type;

    actx.src0_nrows_فی_thread = fastdiv(nrows + n_دھاگے - 1, &octx->n_دھاگے_div);

    actx.src0_صف_سائز = src0_صف_سائز;
    actx.src1_صف_سائز = src1_صف_سائز;
    actx.dst_صف_سائز  = dst_صف_سائز;

    actx.src0_صف_سائز_aligned = src0_صف_سائز_aligned;
    actx.src1_صف_سائز_aligned = src1_صف_سائز_aligned;
    actx.dst_صف_سائز_aligned  = dst_صف_سائز_aligned;

    actx.src0_صف_stride = src0_صف_stride;
    actx.src1_صف_stride = src1_صف_stride;

    uint8_t * ثابت بنیاد = (uint8_t *) octx->ctx->vtcm_بنیاد;
    actx.vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, بنیاد, L.off_src0);
    actx.vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, بنیاد, L.off_src1);
    actx.vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, بنیاد, L.off_dst);

    actx.vtcm_src0_سائز_فی_thread = L.src0_بائٹیں_فی_thread;
    actx.vtcm_src1_سائز_فی_thread = L.src1_بائٹیں_فی_thread;
    actx.vtcm_dst_سائز_فی_thread  = L.dst_بائٹیں_فی_thread;

    actx.src0_spad_half_سائز = L.src0_بائٹیں_فی_thread / 2;
    actx.src1_spad_half_سائز = L.src1_بائٹیں_فی_thread / 2;
    actx.dst_spad_half_سائز  = L.dst_بائٹیں_فی_thread / 2;

    actx.بلاک = actx.src0_spad_half_سائز / actx.src0_صف_سائز_aligned;
    actx.src0_nrows = nrows;
    actx.صف_آغاز  = صف_آغاز;

    actx.nc = dst->ne[0];

    // Addresses and GLU logic
    dma_addr_t ڈیٹا_src0 = src0->ڈیٹا;
    dma_addr_t ڈیٹا_src1 = src1 ? src1->ڈیٹا : 0;

    اگر (!src1 && (octx->op == HTP_OP_GLU_SWIGLU ||
                  octx->op == HTP_OP_GLU_SWIGLU_OAI ||
                  octx->op == HTP_OP_GLU_SWIGLU_CLAMP ||
                  octx->op == HTP_OP_GLU_GEGLU)) {
         ثابت int32_t swapped = octx->op_params[1];
         ڈیٹا_src1 = ڈیٹا_src0;
         actx.src1_صف_سائز = actx.src0_صف_سائز;

         size_t nc_اندر_بائٹیں = actx.nc * SIZEOF_FP16;
         اگر (swapped) {
             ڈیٹا_src0 += nc_اندر_بائٹیں;
         } ورنہ {
             ڈیٹا_src1 += nc_اندر_بائٹیں;
         }
    }

    actx.ڈیٹا_src0 = ڈیٹا_src0;
    actx.ڈیٹا_src1 = ڈیٹا_src1;
    actx.ڈیٹا_dst  = dst->ڈیٹا;

    work_قطار_تسلسل(octx->ctx->work_قطار, (کارکن_callback_t)glu_f16_فی_thread, &actx, n_دھاگے);
    واپس HTP_حالت_OK;
}


صحیح op_activations(ساخت htp_ops_context * octx) {
    انتخاب (octx->src[0]->type) {
        حالت HTP_TYPE_F32:
            واپس execute_op_activations_f32(octx);

        حالت HTP_TYPE_F16:
            واپس execute_op_activations_f16(octx);

        طےشدہ:
            واپس HTP_حالت_NO_SUPPORT;
    }
}
