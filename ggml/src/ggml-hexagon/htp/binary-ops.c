#సూచన clang diagnostic ignored "-Wunused-variable"
#సూచన clang diagnostic ignored "-Wunused-function"
#సూచన clang diagnostic ignored "-Wunused-but-set-variable"

#అన్తర్భావ <HAP_farf.h>
#అన్తర్భావ <HAP_perf.h>

#అన్తర్భావ <math.h>
#అన్తర్భావ <string.h>

#అన్తర్భావ "dma-queue.h"
#అన్తర్భావ "hvx-utils.h"

#నిర్వచించు GGML_COMMON_DECL_C
#అన్తర్భావ "ggml-common.h"
#అన్తర్భావ "hex-common.h"
#అన్తర్భావ "hex-profile.h"
#అన్తర్భావ "binary-ops.h"
#అన్తర్భావ "htp-ctx.h"
#అన్తర్భావ "htp-ops.h"
#అన్తర్భావ "htp-tensor.h"

#నిర్వచితం_కాకపోతే MIN
#నిర్వచించు MIN(a, b) ((a) < (b) ? (a) : (b))
#అయితే_ముగింపు

// Context for binary operations
నిర్మాణం htp_ద్విమాన_context {
    నిర్మాణం htp_ops_context * octx;
    నిర్మాణం htp_ద్విమాన_vtcm_layout vtcm_layout;
    uint8_t * vtcm_పునాది;

    నిర్మాణం fastdiv_విలువలు src0_dim1_div; // ne01
    నిర్మాణం fastdiv_విలువలు src0_dim2_div; // ne02
    నిర్మాణం fastdiv_విలువలు src0_dim12_div;// ne03

    నిర్మాణం fastdiv_విలువలు src1_dim1_div; // ne11
    నిర్మాణం fastdiv_విలువలు src1_dim2_div; // ne12
    నిర్మాణం fastdiv_విలువలు src1_dim3_div; // ne13

    uint32_t ఖండం_max;
    uint32_t nrows_చొప్పున_thread;
    uint32_t మొత్తం_అడ్డువరుసలు;
    uint32_t అడ్డువరుస_ప్రారంభం;
    size_t   src0_అడ్డువరుస_కొలత_aligned;
    size_t   src1_అడ్డువరుస_కొలత_aligned;
    size_t   dst_అడ్డువరుస_కొలత_aligned;
    size_t   అడ్డువరుస_కొలత_బైట్లు;

    bool విభజించు_at_ne01;
    bool విభజించు_at_ne02;

    శూన్య * compute;
};

#నిర్వచించు htp_ద్విమాన_preamble                        \
    స్థిరరాశి నిర్మాణం htp_tensor * src0 = octx->src[0]; \
    స్థిరరాశి నిర్మాణం htp_tensor * src1 = octx->src[1]; \
    స్థిరరాశి నిర్మాణం htp_tensor * dst  = octx->dst;    \
                                                   \
    స్థిరరాశి uint32_t ne00 = src0->ne[0];             \
    స్థిరరాశి uint32_t ne01 = src0->ne[1];             \
    స్థిరరాశి uint32_t ne02 = src0->ne[2];             \
    స్థిరరాశి uint32_t ne03 = src0->ne[3];             \
                                                   \
    స్థిరరాశి uint32_t ne10 = src1->ne[0];             \
    స్థిరరాశి uint32_t ne11 = src1->ne[1];             \
    స్థిరరాశి uint32_t ne12 = src1->ne[2];             \
    స్థిరరాశి uint32_t ne13 = src1->ne[3];             \
                                                   \
    స్థిరరాశి uint32_t nb01 = src0->nb[1];             \
    స్థిరరాశి uint32_t nb02 = src0->nb[2];             \
    స్థిరరాశి uint32_t nb03 = src0->nb[3];             \
                                                   \
    స్థిరరాశి uint32_t nb11 = src1->nb[1];             \
    స్థిరరాశి uint32_t nb12 = src1->nb[2];             \
    స్థిరరాశి uint32_t nb13 = src1->nb[3];             \
                                                   \
    స్థిరరాశి uint32_t nb1 = dst->nb[1];               \
    స్థిరరాశి uint32_t nb2 = dst->nb[2];               \
    స్థిరరాశి uint32_t nb3 = dst->nb[3];

స్థిర అంతఃవరుస uint32_t calc_ఖండం_కొలత(నిర్మాణం htp_ద్విమాన_context * bctx, uint32_t ir, uint32_t చివర_అడ్డువరుస, uint32_t ne01, uint32_t ne02) {
    uint32_t i03, i02, i01, rem;
    i03 = fastdiv(ir, &bctx->src0_dim12_div);
    rem = ir - i03 * (ne02 * ne01);
    i02 = fastdiv(rem, &bctx->src0_dim1_div);
    i01 = rem - i02 * ne01;

    uint32_t అడ్డువరుసలు_ఎడమ = చివర_అడ్డువరుస - ir;
    uint32_t ఖండం_హద్దు = అడ్డువరుసలు_ఎడమ;

    ఒకవేళ (bctx->విభజించు_at_ne01) {
        ఖండం_హద్దు = MIN(ఖండం_హద్దు, ne01 - i01);
    }
    ఒకవేళ (bctx->విభజించు_at_ne02) {
         uint32_t అడ్డువరుసలు_లోపల_plane = (ne02 * ne01) - rem;
         ఖండం_హద్దు = MIN(ఖండం_హద్దు, అడ్డువరుసలు_లోపల_plane);
    }

    తిరిగి MIN(bctx->ఖండం_max, ఖండం_హద్దు);
}

// Out-of-line compute micro-kernels

రకనిర్వచనం శూన్య (*compute_scalar_t)(
    uint8_t * dst, స్థిరరాశి uint8_t * src0, స్థిరరాశి శూన్య * s1_పట్టిక,
    uint32_t cur_i11, uint32_t ne11, uint32_t n_అడ్డువరుసలు,
    size_t dst_stride, size_t src0_stride, uint32_t ne00);

#నిర్వచించు DEFINE_COMPUTE_SCALAR(NAME, TYPE, HVX_STMT)             \
స్థిర శూన్య compute_scalar_##NAME(                              \
    uint8_t * dst, స్థిరరాశి uint8_t * src0, స్థిరరాశి శూన్య * s1_పట్టిక, \
    uint32_t cur_i11, uint32_t ne11, uint32_t n_అడ్డువరుసలు,           \
    size_t dst_stride, size_t src0_stride, uint32_t ne00) {     \
    స్థిరరాశి TYPE * పట్టిక = (స్థిరరాశి TYPE *) s1_పట్టిక;               \
    కొరకు (uint32_t r = 0; r < n_అడ్డువరుసలు; r++) {                     \
        uint8_t * r_dst = dst + r * dst_stride;                 \
        స్థిరరాశి uint8_t * r_src0 = src0 + r * src0_stride;        \
        TYPE val = పట్టిక[cur_i11];                              \
        HVX_STMT;                                               \
        ఒకవేళ (ne11 > 1 && ++cur_i11 == ne11) {                    \
            cur_i11 = 0;                                        \
        }                                                       \
    }                                                           \
}

DEFINE_COMPUTE_SCALAR(కూడు_f32, దశాంశం,    hvx_కూడు_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(కూడు_f16, _Float16, hvx_కూడు_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(sub_f32, దశాంశం,    hvx_sub_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(sub_f16, _Float16, hvx_sub_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(mul_f32, దశాంశం,    hvx_mul_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(mul_f16, _Float16, hvx_mul_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(div_f32, దశాంశం,    hvx_mul_scalar_f32_aa(r_dst, r_src0, 1.0f / (val), ne00))
DEFINE_COMPUTE_SCALAR(div_f16, _Float16, hvx_div_scalar_f16_aa(r_dst, r_src0, val, ne00))

రకనిర్వచనం శూన్య (*compute_same_shape_t)(
    uint8_t * dst, స్థిరరాశి uint8_t * src0, స్థిరరాశి uint8_t * src1,
    uint32_t n_అడ్డువరుసలు, size_t dst_stride, size_t src0_stride, size_t src1_stride, uint32_t ne00);

#నిర్వచించు DEFINE_COMPUTE_SAME_SHAPE(NAME, HVX_FN)                                                  \
స్థిర శూన్య compute_same_shape_##NAME(                                                           \
    uint8_t * dst, స్థిరరాశి uint8_t * src0, స్థిరరాశి uint8_t * src1,                                   \
    uint32_t n_అడ్డువరుసలు, size_t dst_stride, size_t src0_stride, size_t src1_stride, uint32_t ne00) { \
    కొరకు (uint32_t r = 0; r < n_అడ్డువరుసలు; r++) {                                                      \
        HVX_FN(dst + r * dst_stride, src0 + r * src0_stride, src1 + r * src1_stride, ne00);      \
    }                                                                                            \
}

DEFINE_COMPUTE_SAME_SHAPE(కూడు_f32, hvx_కూడు_f32_aaa)
DEFINE_COMPUTE_SAME_SHAPE(కూడు_f16, hvx_కూడు_f16_aaa)
DEFINE_COMPUTE_SAME_SHAPE(sub_f32, hvx_sub_f32_aaa)
DEFINE_COMPUTE_SAME_SHAPE(sub_f16, hvx_sub_f16_aaa)
DEFINE_COMPUTE_SAME_SHAPE(mul_f32, hvx_mul_f32_aaa)
DEFINE_COMPUTE_SAME_SHAPE(mul_f16, hvx_mul_f16_aaa)
DEFINE_COMPUTE_SAME_SHAPE(div_f32, hvx_div_f32_aaa)
DEFINE_COMPUTE_SAME_SHAPE(div_f16, hvx_div_f16_aaa)

రకనిర్వచనం శూన్య (*compute_అడ్డువరుస_bcast_t)(
    uint8_t * dst, స్థిరరాశి uint8_t * src0, స్థిరరాశి uint8_t * src1,
    uint32_t n_అడ్డువరుసలు, size_t dst_stride, size_t src0_stride, uint32_t ne00);

#నిర్వచించు DEFINE_COMPUTE_అడ్డువరుస_BCAST(NAME, HVX_FN)                                               \
స్థిర శూన్య compute_అడ్డువరుస_bcast##NAME(                                                        \
    uint8_t * dst, స్థిరరాశి uint8_t * src0, స్థిరరాశి uint8_t * src1,                               \
    uint32_t n_అడ్డువరుసలు, size_t dst_stride, size_t src0_stride, uint32_t ne00) {                 \
    కొరకు (uint32_t r = 0; r < n_అడ్డువరుసలు; r++) {                                                  \
        HVX_FN(dst + r * dst_stride, src0 + r * src0_stride, src1, ne00);                    \
    }                                                                                        \
}

DEFINE_COMPUTE_అడ్డువరుస_BCAST(కూడు_f32, hvx_కూడు_f32_aaa)
DEFINE_COMPUTE_అడ్డువరుస_BCAST(కూడు_f16, hvx_కూడు_f16_aaa)
DEFINE_COMPUTE_అడ్డువరుస_BCAST(sub_f32, hvx_sub_f32_aaa)
DEFINE_COMPUTE_అడ్డువరుస_BCAST(sub_f16, hvx_sub_f16_aaa)
DEFINE_COMPUTE_అడ్డువరుస_BCAST(mul_f32, hvx_mul_f32_aaa)
DEFINE_COMPUTE_అడ్డువరుస_BCAST(mul_f16, hvx_mul_f16_aaa)
DEFINE_COMPUTE_అడ్డువరుస_BCAST(div_f32, hvx_div_f32_aaa)
DEFINE_COMPUTE_అడ్డువరుస_BCAST(div_f16, hvx_div_f16_aaa)

రకనిర్వచనం శూన్య (*compute_complex_t)(
    uint8_t * dst, స్థిరరాశి uint8_t * src0, స్థిరరాశి uint8_t * src1_plane,
    uint32_t i01, uint32_t ne11, స్థిరరాశి నిర్మాణం fastdiv_విలువలు * div11, uint32_t src1_stride,
    uint32_t n_అడ్డువరుసలు, size_t dst_stride, size_t src0_stride, uint32_t ne00);

#నిర్వచించు DEFINE_COMPUTE_COMPLEX(NAME, HVX_FN)                                                 \
స్థిర శూన్య compute_complex_##NAME(                                                          \
    uint8_t * dst, స్థిరరాశి uint8_t * src0, స్థిరరాశి uint8_t * src1_plane,                         \
    uint32_t i01, uint32_t ne11, స్థిరరాశి నిర్మాణం fastdiv_విలువలు * div11, uint32_t src1_stride,  \
    uint32_t n_అడ్డువరుసలు, size_t dst_stride, size_t src0_stride, uint32_t ne00) {                 \
    కొరకు (uint32_t r = 0; r < n_అడ్డువరుసలు; r++) {                                                  \
        uint32_t i11 = fastmodulo(i01 + r, ne11, div11);                                     \
        స్థిరరాశి uint8_t * r_src1 = src1_plane + i11 * src1_stride;                             \
        HVX_FN(dst + r * dst_stride, src0 + r * src0_stride, r_src1, ne00);                  \
    }                                                                                        \
}

DEFINE_COMPUTE_COMPLEX(కూడు_f32, hvx_కూడు_f32_aau)
DEFINE_COMPUTE_COMPLEX(కూడు_f16, hvx_కూడు_f16_aau)
DEFINE_COMPUTE_COMPLEX(sub_f32, hvx_sub_f32_aau)
DEFINE_COMPUTE_COMPLEX(sub_f16, hvx_sub_f16_aau)
DEFINE_COMPUTE_COMPLEX(mul_f32, hvx_mul_f32_aau)
DEFINE_COMPUTE_COMPLEX(mul_f16, hvx_mul_f16_aau)
DEFINE_COMPUTE_COMPLEX(div_f32, hvx_div_f32_aau)
DEFINE_COMPUTE_COMPLEX(div_f16, hvx_div_f16_aau)

రకనిర్వచనం శూన్య (*compute_repeat_t)(
    uint8_t * dst, స్థిరరాశి uint8_t * src0, స్థిరరాశి uint8_t * src1_plane,
    uint32_t i01, uint32_t ne11, స్థిరరాశి నిర్మాణం fastdiv_విలువలు * div11, uint32_t src1_stride,
    uint32_t n_అడ్డువరుసలు, size_t dst_stride, size_t src0_stride, uint32_t ne00, uint32_t ne10);

#నిర్వచించు DEFINE_COMPUTE_REPEAT(NAME, TYPE, HVX_FN)                                            \
స్థిర శూన్య compute_repeat_##NAME(                                                           \
    uint8_t * dst, స్థిరరాశి uint8_t * src0, స్థిరరాశి uint8_t * src1_plane,                         \
    uint32_t i01, uint32_t ne11, స్థిరరాశి నిర్మాణం fastdiv_విలువలు * div11, uint32_t src1_stride,  \
    uint32_t n_అడ్డువరుసలు, size_t dst_stride, size_t src0_stride, uint32_t ne00, uint32_t ne10) {  \
    కొరకు (uint32_t r = 0; r < n_అడ్డువరుసలు; r++) {                                                  \
        uint32_t i11 = fastmodulo(i01 + r, ne11, div11);                                     \
        స్థిరరాశి uint8_t * r_src1_అడ్డువరుస = src1_plane + i11 * src1_stride;                         \
        uint8_t * r_dst = dst + r * dst_stride;                                              \
        స్థిరరాశి uint8_t * r_src0 = src0 + r * src0_stride;                                     \
        కొరకు (uint32_t c = 0; c < ne00; c += ne10) {                                          \
            uint32_t len = MIN(ne10, ne00 - c);                                              \
            HVX_FN(r_dst + c * పరిమాణం(TYPE), r_src0 + c * పరిమాణం(TYPE), r_src1_అడ్డువరుస, len);    \
        }                                                                                    \
    }                                                                                        \
}

DEFINE_COMPUTE_REPEAT(కూడు_f32, దశాంశం,    hvx_కూడు_f32_uuu)
DEFINE_COMPUTE_REPEAT(కూడు_f16, _Float16, hvx_కూడు_f16_uuu)
DEFINE_COMPUTE_REPEAT(sub_f32, దశాంశం,    hvx_sub_f32_uuu)
DEFINE_COMPUTE_REPEAT(sub_f16, _Float16, hvx_sub_f16_uuu)
DEFINE_COMPUTE_REPEAT(mul_f32, దశాంశం,    hvx_mul_f32_uuu)
DEFINE_COMPUTE_REPEAT(mul_f16, _Float16, hvx_mul_f16_uuu)
DEFINE_COMPUTE_REPEAT(div_f32, దశాంశం,    hvx_div_f32_uuu)
DEFINE_COMPUTE_REPEAT(div_f16, _Float16, hvx_div_f16_uuu)

రకనిర్వచనం శూన్య (*compute_కూడు_id_t)(
    uint8_t * dst, స్థిరరాశి uint8_t * src0, స్థిరరాశి uint8_t * src1_దత్తాంశం, స్థిరరాశి int32_t * src2_ids,
    uint32_t i01, uint32_t i02, uint32_t ne01, uint32_t src1_stride,
    uint32_t n_అడ్డువరుసలు, size_t dst_stride, size_t src0_stride, uint32_t ne00);

స్థిర శూన్య compute_కూడు_id_f32(
    uint8_t * dst, స్థిరరాశి uint8_t * src0, స్థిరరాశి uint8_t * src1_దత్తాంశం, స్థిరరాశి int32_t * src2_ids,
    uint32_t i01, uint32_t i02, uint32_t ne01, uint32_t src1_stride,
    uint32_t n_అడ్డువరుసలు, size_t dst_stride, size_t src0_stride, uint32_t ne00) {
    కొరకు (uint32_t r = 0; r < n_అడ్డువరుసలు; r++) {
        uint32_t r_i01 = i01 + r;
        స్థిరరాశి int32_t idx = src2_ids[i02 * ne01 + r_i01];
        ఒకవేళ (idx < 0) {
            జ్ఞాపకం_నకలు(dst + r * dst_stride, src0 + r * src0_stride, ne00 * పరిమాణం(దశాంశం));
            కొనసాగు;
        }
        స్థిరరాశి uint8_t * r_src1 = src1_దత్తాంశం + idx * src1_stride;
        స్థిరరాశి uint8_t * r_src0 = src0 + r * src0_stride;
        uint8_t * r_dst = dst + r * dst_stride;
        hvx_కూడు_f32_aaa(r_dst, r_src0, r_src1, ne00);
    }
}

// 1. Scalar src1 in VTCM via DMA (ne10 == 1)
స్థిర శూన్య ద్విమాన_thread_scalar(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    నిర్మాణం htp_ద్విమాన_context * bctx = (నిర్మాణం htp_ద్విమాన_context *) దత్తాంశం;
    నిర్మాణం htp_ops_context * octx = bctx->octx;
    htp_ద్విమాన_preamble;

    స్థిరరాశి uint32_t elem_కొలత = (src0->type == HTP_TYPE_F32) ? పరిమాణం(దశాంశం) : పరిమాణం(_Float16);
    స్థిరరాశి size_t plane_stride = (size_t) ne12 * ne11 * elem_కొలత;
    స్థిరరాశి size_t అడ్డువరుస_stride   = (size_t) ne11 * elem_కొలత;

    స్థిరరాశి uint32_t అడ్డువరుస_కొలత_బైట్లు = bctx->అడ్డువరుస_కొలత_బైట్లు;
    స్థిరరాశి uint32_t ప్రారంభం_అడ్డువరుస = bctx->అడ్డువరుస_ప్రారంభం + bctx->nrows_చొప్పున_thread * ith;
    స్థిరరాశి uint32_t చివర_అడ్డువరుస   = MIN(ప్రారంభం_అడ్డువరుస + bctx->nrows_చొప్పున_thread, bctx->అడ్డువరుస_ప్రారంభం + bctx->మొత్తం_అడ్డువరుసలు);
    ఒకవేళ (ప్రారంభం_అడ్డువరుస >= చివర_అడ్డువరుస) తిరిగి;

    FARF(ఎగువ_2, "binary-scalar: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, ప్రారంభం_అడ్డువరుస, చివర_అడ్డువరుస, nb01, bctx->dst_అడ్డువరుస_కొలత_aligned);

    స్థిరరాశి నిర్మాణం htp_ద్విమాన_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_పునాది = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_పునాది, layout->off_src0) + (ith * layout->src0_బైట్లు_చొప్పున_thread);
    uint8_t * dst_spad_పునాది  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_పునాది, layout->off_dst)  + (ith * layout->dst_బైట్లు_చొప్పున_thread);
    size_t src0_spad_half    = layout->src0_spad_half_కొలత;
    size_t dst_spad_half     = layout->dst_spad_half_కొలత;
    స్థిరరాశి శూన్య * s1_పట్టిక    = VTCM_LAYOUT_PTR(స్థిరరాశి శూన్య, bctx->vtcm_పునాది, layout->off_src1);

    dma_వరుస * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = ప్రారంభం_అడ్డువరుస;
    పూర్ణాంకం spad_idx = 0;

    కొరకు (పూర్ణాంకం k = 0; k < 2 && ir_prefetch < చివర_అడ్డువరుస; k++) {
        uint32_t current_ఖండం_కొలత = calc_ఖండం_కొలత(bctx, ir_prefetch, చివర_అడ్డువరుస, ne01, ne02);
        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        rem = ir_prefetch - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->దత్తాంశం + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->దత్తాంశం  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_పునాది + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_పునాది  + spad_idx * dst_spad_half;

        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(dst_curr, d_spad), nb1, bctx->dst_అడ్డువరుస_కొలత_aligned, అడ్డువరుస_కొలత_బైట్లు, 0);
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(s0_spad, src0_curr), bctx->src0_అడ్డువరుస_కొలత_aligned, nb01, అడ్డువరుస_కొలత_బైట్లు,
                                                             current_ఖండం_కొలత);
        ir_prefetch += current_ఖండం_కొలత;
        spad_idx ^= 1;
    }

    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_scalar_t compute = (compute_scalar_t) bctx->compute;

    కొరకు (uint32_t ir = ప్రారంభం_అడ్డువరుస; ir < చివర_అడ్డువరుస; ) {
        uint32_t current_ఖండం_కొలత = calc_ఖండం_కొలత(bctx, ir, చివర_అడ్డువరుస, ne01, ne02);

        uint8_t * d_spad  = (uint8_t *) dma_వరుస_తీయి(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_వరుస_తీయి(dma_q).dst;

        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir, &bctx->src0_dim12_div);
        rem = ir - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;

        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
        uint32_t i11 = fastmodulo(i01, ne11, &bctx->src1_dim1_div);
        స్థిరరాశి uint8_t * scalar_పట్టిక = (స్థిరరాశి uint8_t *) s1_పట్టిక +
            i13 * plane_stride + i12 * అడ్డువరుస_stride;

        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, scalar_పట్టిక, i11, ne11, current_ఖండం_కొలత,
                bctx->dst_అడ్డువరుస_కొలత_aligned, bctx->src0_అడ్డువరుస_కొలత_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->దత్తాంశం + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(dst_curr, d_spad), nb1, bctx->dst_అడ్డువరుస_కొలత_aligned, అడ్డువరుస_కొలత_బైట్లు,
                                                                current_ఖండం_కొలత);

        ఒకవేళ (ir_prefetch < చివర_అడ్డువరుస) {
            uint32_t తదుపరి_ఖండం_కొలత = calc_ఖండం_కొలత(bctx, ir_prefetch, చివర_అడ్డువరుస, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02 = fastdiv(prem, &bctx->src0_dim1_div);
            p01 = prem - p02 * ne01;
            dma_addr_t s0_తదుపరి = src0->దత్తాంశం + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(s0_spad, s0_తదుపరి), bctx->src0_అడ్డువరుస_కొలత_aligned, nb01, అడ్డువరుస_కొలత_బైట్లు,
                                                               తదుపరి_ఖండం_కొలత);
            ir_prefetch += తదుపరి_ఖండం_కొలత;
        }
        ir += current_ఖండం_కొలత;
    }

    dma_వరుస_flush(dma_q);
}

// 2. Vector Same Shape (ne1x == ne0x) or Simple Broadcast
స్థిర శూన్య ద్విమాన_thread_సదిశ_same_shape(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    నిర్మాణం htp_ద్విమాన_context * bctx = (నిర్మాణం htp_ద్విమాన_context *) దత్తాంశం;
    నిర్మాణం htp_ops_context * octx = bctx->octx;
    htp_ద్విమాన_preamble;

    స్థిరరాశి uint32_t అడ్డువరుస_కొలత_బైట్లు = bctx->అడ్డువరుస_కొలత_బైట్లు;
    స్థిరరాశి uint32_t ప్రారంభం_అడ్డువరుస = bctx->అడ్డువరుస_ప్రారంభం + bctx->nrows_చొప్పున_thread * ith;
    స్థిరరాశి uint32_t చివర_అడ్డువరుస   = MIN(ప్రారంభం_అడ్డువరుస + bctx->nrows_చొప్పున_thread, bctx->అడ్డువరుస_ప్రారంభం + bctx->మొత్తం_అడ్డువరుసలు);
    ఒకవేళ (ప్రారంభం_అడ్డువరుస >= చివర_అడ్డువరుస) తిరిగి;

    FARF(ఎగువ_2, "binary-same-shape: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, ప్రారంభం_అడ్డువరుస, చివర_అడ్డువరుస, nb01, bctx->dst_అడ్డువరుస_కొలత_aligned);

    స్థిరరాశి నిర్మాణం htp_ద్విమాన_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_పునాది = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_పునాది, layout->off_src0) + (ith * layout->src0_బైట్లు_చొప్పున_thread);
    uint8_t * src1_spad_పునాది = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_పునాది, layout->off_src1) + (ith * layout->src1_బైట్లు_చొప్పున_thread);
    uint8_t * dst_spad_పునాది  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_పునాది, layout->off_dst)  + (ith * layout->dst_బైట్లు_చొప్పున_thread);

    size_t src0_spad_half = layout->src0_spad_half_కొలత;
    size_t src1_spad_half = layout->src1_spad_half_కొలత;
    size_t dst_spad_half  = layout->dst_spad_half_కొలత;

    dma_వరుస * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = ప్రారంభం_అడ్డువరుస;
    పూర్ణాంకం spad_idx = 0;

    కొరకు (పూర్ణాంకం k = 0; k < 2 && ir_prefetch < చివర_అడ్డువరుస; k++) {
        uint32_t current_ఖండం_కొలత = calc_ఖండం_కొలత(bctx, ir_prefetch, చివర_అడ్డువరుస, ne01, ne02);
        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        rem = ir_prefetch - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;

        uint32_t i13 = (ne13 == 1) ? 0 : i03;
        uint32_t i12 = (ne12 == 1) ? 0 : i02;
        uint32_t i11 = (ne11 == 1) ? 0 : i01;

        dma_addr_t src0_curr = src0->దత్తాంశం + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t src1_curr = src1->దత్తాంశం + i13 * nb13 + i12 * nb12 + i11 * nb11;
        dma_addr_t dst_curr  = dst->దత్తాంశం  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_పునాది + spad_idx * src0_spad_half;
        uint8_t * s1_spad = src1_spad_పునాది + spad_idx * src1_spad_half;
        uint8_t * d_spad  = dst_spad_పునాది  + spad_idx * dst_spad_half;

        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(dst_curr, d_spad), nb1, bctx->dst_అడ్డువరుస_కొలత_aligned, అడ్డువరుస_కొలత_బైట్లు, 0);
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(s0_spad, src0_curr), bctx->src0_అడ్డువరుస_కొలత_aligned, nb01, అడ్డువరుస_కొలత_బైట్లు, current_ఖండం_కొలత);
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(s1_spad, src1_curr), bctx->src1_అడ్డువరుస_కొలత_aligned, nb11, అడ్డువరుస_కొలత_బైట్లు, current_ఖండం_కొలత);
        ir_prefetch += current_ఖండం_కొలత;
        spad_idx ^= 1;
    }

    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_same_shape_t compute = (compute_same_shape_t) bctx->compute;

    కొరకు (uint32_t ir = ప్రారంభం_అడ్డువరుస; ir < చివర_అడ్డువరుస; ) {
        uint32_t current_ఖండం_కొలత = calc_ఖండం_కొలత(bctx, ir, చివర_అడ్డువరుస, ne01, ne02);
        uint8_t * d_spad  = (uint8_t *) dma_వరుస_తీయి(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_వరుస_తీయి(dma_q).dst;
        uint8_t * s1_spad = (uint8_t *) dma_వరుస_తీయి(dma_q).dst;

        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, s1_spad, current_ఖండం_కొలత,
                bctx->dst_అడ్డువరుస_కొలత_aligned, bctx->src0_అడ్డువరుస_కొలత_aligned, bctx->src1_అడ్డువరుస_కొలత_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir, &bctx->src0_dim12_div);
        rem = ir - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;
        dma_addr_t dst_curr = dst->దత్తాంశం + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(dst_curr, d_spad), nb1, bctx->dst_అడ్డువరుస_కొలత_aligned, అడ్డువరుస_కొలత_బైట్లు, current_ఖండం_కొలత);

        ఒకవేళ (ir_prefetch < చివర_అడ్డువరుస) {
            uint32_t తదుపరి_ఖండం_కొలత = calc_ఖండం_కొలత(bctx, ir_prefetch, చివర_అడ్డువరుస, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02 = fastdiv(prem, &bctx->src0_dim1_div);
            p01 = prem - p02 * ne01;

            uint32_t p13 = (ne13 == 1) ? 0 : p03;
            uint32_t p12 = (ne12 == 1) ? 0 : p02;
            uint32_t p11 = (ne11 == 1) ? 0 : p01;

            dma_addr_t s0_తదుపరి = src0->దత్తాంశం + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_addr_t s1_తదుపరి = src1->దత్తాంశం + p13 * nb13 + p12 * nb12 + p11 * nb11;

            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(s0_spad, s0_తదుపరి), bctx->src0_అడ్డువరుస_కొలత_aligned, nb01, అడ్డువరుస_కొలత_బైట్లు, తదుపరి_ఖండం_కొలత);
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(s1_spad, s1_తదుపరి), bctx->src1_అడ్డువరుస_కొలత_aligned, nb11, అడ్డువరుస_కొలత_బైట్లు, తదుపరి_ఖండం_కొలత);

            ir_prefetch += తదుపరి_ఖండం_కొలత;
        }
        ir += current_ఖండం_కొలత;
    }

    dma_వరుస_flush(dma_q);
}

// 3. Row Broadcast (ne11 == 1, ne12 == 1, single row src1)
స్థిర శూన్య ద్విమాన_thread_సదిశ_అడ్డువరుస_ప్రసారం(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    నిర్మాణం htp_ద్విమాన_context * bctx = (నిర్మాణం htp_ద్విమాన_context *) దత్తాంశం;
    నిర్మాణం htp_ops_context * octx = bctx->octx;
    htp_ద్విమాన_preamble;

    స్థిరరాశి uint32_t అడ్డువరుస_కొలత_బైట్లు = bctx->అడ్డువరుస_కొలత_బైట్లు;
    స్థిరరాశి uint32_t ప్రారంభం_అడ్డువరుస  = bctx->అడ్డువరుస_ప్రారంభం + bctx->nrows_చొప్పున_thread * ith;
    స్థిరరాశి uint32_t చివర_అడ్డువరుస    = MIN(ప్రారంభం_అడ్డువరుస + bctx->nrows_చొప్పున_thread, bctx->అడ్డువరుస_ప్రారంభం + bctx->మొత్తం_అడ్డువరుసలు);
    ఒకవేళ (ప్రారంభం_అడ్డువరుస >= చివర_అడ్డువరుస) తిరిగి;

    FARF(ఎగువ_2, "binary-row-bcast: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, ప్రారంభం_అడ్డువరుస, చివర_అడ్డువరుస, nb01, bctx->dst_అడ్డువరుస_కొలత_aligned);

    స్థిరరాశి నిర్మాణం htp_ద్విమాన_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_పునాది = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_పునాది, layout->off_src0) + (ith * layout->src0_బైట్లు_చొప్పున_thread);
    uint8_t * dst_spad_పునాది  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_పునాది, layout->off_dst)  + (ith * layout->dst_బైట్లు_చొప్పున_thread);

    size_t src0_spad_half = layout->src0_spad_half_కొలత;
    size_t dst_spad_half  = layout->dst_spad_half_కొలత;

    dma_వరుస * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = ప్రారంభం_అడ్డువరుస;
    పూర్ణాంకం spad_idx = 0;

    శూన్య * s1_ptr = VTCM_LAYOUT_PTR(శూన్య, bctx->vtcm_పునాది, layout->off_src1);

    కొరకు (పూర్ణాంకం k = 0; k < 2 && ir_prefetch < చివర_అడ్డువరుస; k++) {
        uint32_t current_ఖండం_కొలత = calc_ఖండం_కొలత(bctx, ir_prefetch, చివర_అడ్డువరుస, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->దత్తాంశం + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->దత్తాంశం  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_పునాది + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_పునాది  + spad_idx * dst_spad_half;

        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(dst_curr, d_spad), nb1, bctx->dst_అడ్డువరుస_కొలత_aligned, అడ్డువరుస_కొలత_బైట్లు, 0);
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(s0_spad, src0_curr), bctx->src0_అడ్డువరుస_కొలత_aligned, nb01, అడ్డువరుస_కొలత_బైట్లు, current_ఖండం_కొలత);
        ir_prefetch += current_ఖండం_కొలత;
        spad_idx ^= 1;
    }

    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_అడ్డువరుస_bcast_t compute = (compute_అడ్డువరుస_bcast_t) bctx->compute;

    కొరకు (uint32_t ir = ప్రారంభం_అడ్డువరుస; ir < చివర_అడ్డువరుస; ) {
        uint32_t current_ఖండం_కొలత = calc_ఖండం_కొలత(bctx, ir, చివర_అడ్డువరుస, ne01, ne02);
        uint8_t * d_spad  = (uint8_t *) dma_వరుస_తీయి(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_వరుస_తీయి(dma_q).dst;

        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, (స్థిరరాశి uint8_t *)s1_ptr, current_ఖండం_కొలత,
                bctx->dst_అడ్డువరుస_కొలత_aligned, bctx->src0_అడ్డువరుస_కొలత_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;
        dma_addr_t dst_curr = dst->దత్తాంశం + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(dst_curr, d_spad), nb1, bctx->dst_అడ్డువరుస_కొలత_aligned, అడ్డువరుస_కొలత_బైట్లు, current_ఖండం_కొలత);

        ఒకవేళ (ir_prefetch < చివర_అడ్డువరుస) {
            uint32_t తదుపరి_ఖండం_కొలత = calc_ఖండం_కొలత(bctx, ir_prefetch, చివర_అడ్డువరుస, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02  = fastdiv(prem, &bctx->src0_dim1_div);
            p01  = prem - p02 * ne01;
            dma_addr_t s0_తదుపరి = src0->దత్తాంశం + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(s0_spad, s0_తదుపరి), bctx->src0_అడ్డువరుస_కొలత_aligned, nb01, అడ్డువరుస_కొలత_బైట్లు, తదుపరి_ఖండం_కొలత);
            ir_prefetch += తదుపరి_ఖండం_కొలత;
        }
        ir += current_ఖండం_కొలత;
    }

    dma_వరుస_flush(dma_q);
}

// 4. Vector Complex (ne10 == ne00, complex broadcast)
స్థిర శూన్య ద్విమాన_thread_సదిశ_complex(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    నిర్మాణం htp_ద్విమాన_context * bctx = (నిర్మాణం htp_ద్విమాన_context *) దత్తాంశం;
    నిర్మాణం htp_ops_context * octx = bctx->octx;
    htp_ద్విమాన_preamble;

    స్థిరరాశి uint32_t అడ్డువరుస_కొలత_బైట్లు = bctx->అడ్డువరుస_కొలత_బైట్లు;
    స్థిరరాశి uint32_t ప్రారంభం_అడ్డువరుస  = bctx->అడ్డువరుస_ప్రారంభం + bctx->nrows_చొప్పున_thread * ith;
    స్థిరరాశి uint32_t చివర_అడ్డువరుస    = MIN(ప్రారంభం_అడ్డువరుస + bctx->nrows_చొప్పున_thread, bctx->అడ్డువరుస_ప్రారంభం + bctx->మొత్తం_అడ్డువరుసలు);
    ఒకవేళ (ప్రారంభం_అడ్డువరుస >= చివర_అడ్డువరుస) తిరిగి;

    FARF(ఎగువ_2, "binary-complex: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, ప్రారంభం_అడ్డువరుస, చివర_అడ్డువరుస, nb01, bctx->dst_అడ్డువరుస_కొలత_aligned);

    స్థిరరాశి నిర్మాణం htp_ద్విమాన_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_పునాది = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_పునాది, layout->off_src0) + (ith * layout->src0_బైట్లు_చొప్పున_thread);
    uint8_t * dst_spad_పునాది  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_పునాది, layout->off_dst)  + (ith * layout->dst_బైట్లు_చొప్పున_thread);
    size_t src0_spad_half    = layout->src0_spad_half_కొలత;
    size_t dst_spad_half     = layout->dst_spad_half_కొలత;
    స్థిరరాశి uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(స్థిరరాశి uint8_t, bctx->vtcm_పునాది, layout->off_src1);

    స్థిరరాశి uint32_t elem_కొలత = (src0->type == HTP_TYPE_F32) ? పరిమాణం(దశాంశం) : పరిమాణం(_Float16);
    స్థిరరాశి size_t vtcm_అడ్డువరుస_stride   = (size_t) ne10 * elem_కొలత;
    స్థిరరాశి size_t vtcm_plane_stride = (size_t) ne11 * vtcm_అడ్డువరుస_stride;
    స్థిరరాశి size_t vtcm_vol_stride   = (size_t) ne12 * vtcm_plane_stride;

    dma_వరుస * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = ప్రారంభం_అడ్డువరుస;
    పూర్ణాంకం spad_idx = 0;

    కొరకు (పూర్ణాంకం k = 0; k < 2 && ir_prefetch < చివర_అడ్డువరుస; k++) {
        uint32_t current_ఖండం_కొలత = calc_ఖండం_కొలత(bctx, ir_prefetch, చివర_అడ్డువరుస, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->దత్తాంశం + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->దత్తాంశం  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_పునాది + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_పునాది  + spad_idx * dst_spad_half;

        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(dst_curr, d_spad), nb1, bctx->dst_అడ్డువరుస_కొలత_aligned, అడ్డువరుస_కొలత_బైట్లు, 0);
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(s0_spad, src0_curr), bctx->src0_అడ్డువరుస_కొలత_aligned, nb01, అడ్డువరుస_కొలత_బైట్లు, current_ఖండం_కొలత);
        ir_prefetch += current_ఖండం_కొలత;
        spad_idx ^= 1;
    }

    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_complex_t compute = (compute_complex_t) bctx->compute;

    కొరకు (uint32_t ir = ప్రారంభం_అడ్డువరుస; ir < చివర_అడ్డువరుస; ) {
        uint32_t current_ఖండం_కొలత = calc_ఖండం_కొలత(bctx, ir, చివర_అడ్డువరుస, ne01, ne02);
        uint8_t * d_spad = (uint8_t *) dma_వరుస_తీయి(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_వరుస_తీయి(dma_q).dst;

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
        స్థిరరాశి uint8_t * src1_plane = vtcm_src1 + i13 * vtcm_vol_stride + i12 * vtcm_plane_stride;

        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, src1_plane, i01, ne11, &bctx->src1_dim1_div, vtcm_అడ్డువరుస_stride,
                current_ఖండం_కొలత, bctx->dst_అడ్డువరుస_కొలత_aligned, bctx->src0_అడ్డువరుస_కొలత_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->దత్తాంశం + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(dst_curr, d_spad), nb1, bctx->dst_అడ్డువరుస_కొలత_aligned, అడ్డువరుస_కొలత_బైట్లు, current_ఖండం_కొలత);

        ఒకవేళ (ir_prefetch < చివర_అడ్డువరుస) {
            uint32_t తదుపరి_ఖండం_కొలత = calc_ఖండం_కొలత(bctx, ir_prefetch, చివర_అడ్డువరుస, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02  = fastdiv(prem, &bctx->src0_dim1_div);
            p01  = prem - p02 * ne01;
            dma_addr_t s0_తదుపరి = src0->దత్తాంశం + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(s0_spad, s0_తదుపరి), bctx->src0_అడ్డువరుస_కొలత_aligned, nb01, అడ్డువరుస_కొలత_బైట్లు, తదుపరి_ఖండం_కొలత);
            ir_prefetch += తదుపరి_ఖండం_కొలత;
        }
        ir += current_ఖండం_కొలత;
    }

    dma_వరుస_flush(dma_q);
}

// 5. Element Repeat (ne10 != ne00)
స్థిర శూన్య ద్విమాన_thread_element_repeat(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    నిర్మాణం htp_ద్విమాన_context * bctx = (నిర్మాణం htp_ద్విమాన_context *) దత్తాంశం;
    నిర్మాణం htp_ops_context * octx = bctx->octx;
    htp_ద్విమాన_preamble;

    స్థిరరాశి uint32_t అడ్డువరుస_కొలత_బైట్లు = bctx->అడ్డువరుస_కొలత_బైట్లు;
    స్థిరరాశి uint32_t ప్రారంభం_అడ్డువరుస  = bctx->అడ్డువరుస_ప్రారంభం + bctx->nrows_చొప్పున_thread * ith;
    స్థిరరాశి uint32_t చివర_అడ్డువరుస    = MIN(ప్రారంభం_అడ్డువరుస + bctx->nrows_చొప్పున_thread, bctx->అడ్డువరుస_ప్రారంభం + bctx->మొత్తం_అడ్డువరుసలు);
    ఒకవేళ (ప్రారంభం_అడ్డువరుస >= చివర_అడ్డువరుస) తిరిగి;

    స్థిరరాశి నిర్మాణం htp_ద్విమాన_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_పునాది = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_పునాది, layout->off_src0) + (ith * layout->src0_బైట్లు_చొప్పున_thread);
    uint8_t * dst_spad_పునాది  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_పునాది, layout->off_dst)  + (ith * layout->dst_బైట్లు_చొప్పున_thread);
    size_t src0_spad_half    = layout->src0_spad_half_కొలత;
    size_t dst_spad_half     = layout->dst_spad_half_కొలత;
    స్థిరరాశి uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(స్థిరరాశి uint8_t, bctx->vtcm_పునాది, layout->off_src1);

    స్థిరరాశి uint32_t elem_కొలత = (src0->type == HTP_TYPE_F32) ? పరిమాణం(దశాంశం) : పరిమాణం(_Float16);
    స్థిరరాశి size_t vtcm_అడ్డువరుస_stride   = (size_t) ne10 * elem_కొలత;
    స్థిరరాశి size_t vtcm_plane_stride = (size_t) ne11 * vtcm_అడ్డువరుస_stride;
    స్థిరరాశి size_t vtcm_vol_stride   = (size_t) ne12 * vtcm_plane_stride;

    FARF(ఎగువ_2, "binary-repeat: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, ప్రారంభం_అడ్డువరుస, చివర_అడ్డువరుస, nb01, bctx->dst_అడ్డువరుస_కొలత_aligned);

    dma_వరుస * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = ప్రారంభం_అడ్డువరుస;
    పూర్ణాంకం spad_idx = 0;

    కొరకు (పూర్ణాంకం k = 0; k < 2 && ir_prefetch < చివర_అడ్డువరుస; k++) {
        uint32_t current_ఖండం_కొలత = calc_ఖండం_కొలత(bctx, ir_prefetch, చివర_అడ్డువరుస, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->దత్తాంశం + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->దత్తాంశం  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_పునాది + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_పునాది  + spad_idx * dst_spad_half;

        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(dst_curr, d_spad), nb1, bctx->dst_అడ్డువరుస_కొలత_aligned, అడ్డువరుస_కొలత_బైట్లు, 0);
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(s0_spad, src0_curr), bctx->src0_అడ్డువరుస_కొలత_aligned, nb01, అడ్డువరుస_కొలత_బైట్లు, current_ఖండం_కొలత);
        ir_prefetch += current_ఖండం_కొలత;
        spad_idx ^= 1;
    }

    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_repeat_t compute = (compute_repeat_t) bctx->compute;

    కొరకు (uint32_t ir = ప్రారంభం_అడ్డువరుస; ir < చివర_అడ్డువరుస; ) {
        uint32_t current_ఖండం_కొలత = calc_ఖండం_కొలత(bctx, ir, చివర_అడ్డువరుస, ne01, ne02);
        uint8_t * d_spad  = (uint8_t *) dma_వరుస_తీయి(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_వరుస_తీయి(dma_q).dst;

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
        స్థిరరాశి uint8_t * src1_plane = vtcm_src1 + i13 * vtcm_vol_stride + i12 * vtcm_plane_stride;

        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, src1_plane, i01, ne11, &bctx->src1_dim1_div, vtcm_అడ్డువరుస_stride,
                current_ఖండం_కొలత, bctx->dst_అడ్డువరుస_కొలత_aligned, bctx->src0_అడ్డువరుస_కొలత_aligned, ne00, ne10);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->దత్తాంశం + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(dst_curr, d_spad), nb1, bctx->dst_అడ్డువరుస_కొలత_aligned, అడ్డువరుస_కొలత_బైట్లు, current_ఖండం_కొలత);

        ఒకవేళ (ir_prefetch < చివర_అడ్డువరుస) {
            uint32_t తదుపరి_ఖండం_కొలత = calc_ఖండం_కొలత(bctx, ir_prefetch, చివర_అడ్డువరుస, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02  = fastdiv(prem, &bctx->src0_dim1_div);
            p01  = prem - p02 * ne01;
            dma_addr_t s0_తదుపరి = src0->దత్తాంశం + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(s0_spad, s0_తదుపరి), bctx->src0_అడ్డువరుస_కొలత_aligned, nb01, అడ్డువరుస_కొలత_బైట్లు, తదుపరి_ఖండం_కొలత);
            ir_prefetch += తదుపరి_ఖండం_కొలత;
        }
        ir += current_ఖండం_కొలత;
    }

    dma_వరుస_flush(dma_q);
}

// 6. ADD_ID (src1 gathered via src2 indices)
స్థిర శూన్య ద్విమాన_thread_కూడు_id_f32(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    నిర్మాణం htp_ద్విమాన_context * bctx = (నిర్మాణం htp_ద్విమాన_context *) దత్తాంశం;
    నిర్మాణం htp_ops_context * octx = bctx->octx;

    స్థిరరాశి నిర్మాణం htp_tensor * src0 = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * src1 = octx->src[1];
    స్థిరరాశి నిర్మాణం htp_tensor * dst  = octx->dst;

    స్థిరరాశి uint32_t ne00 = src0->ne[0];
    స్థిరరాశి uint32_t ne01 = src0->ne[1];
    స్థిరరాశి uint32_t ne02 = src0->ne[2];
    స్థిరరాశి uint32_t ne03 = src0->ne[3];

    స్థిరరాశి uint32_t nb01 = src0->nb[1];
    స్థిరరాశి uint32_t nb02 = src0->nb[2];
    స్థిరరాశి uint32_t nb03 = src0->nb[3];
    స్థిరరాశి uint32_t src1_stride = bctx->src1_అడ్డువరుస_కొలత_aligned;

    స్థిరరాశి uint32_t nb1 = dst->nb[1];
    స్థిరరాశి uint32_t nb2 = dst->nb[2];
    స్థిరరాశి uint32_t nb3 = dst->nb[3];

    స్థిరరాశి uint32_t అడ్డువరుస_కొలత_బైట్లు = bctx->అడ్డువరుస_కొలత_బైట్లు;
    స్థిరరాశి uint32_t ప్రారంభం_అడ్డువరుస = bctx->అడ్డువరుస_ప్రారంభం + bctx->nrows_చొప్పున_thread * ith;
    స్థిరరాశి uint32_t చివర_అడ్డువరుస   = MIN(ప్రారంభం_అడ్డువరుస + bctx->nrows_చొప్పున_thread, bctx->అడ్డువరుస_ప్రారంభం + bctx->మొత్తం_అడ్డువరుసలు);
    ఒకవేళ (ప్రారంభం_అడ్డువరుస >= చివర_అడ్డువరుస) తిరిగి;

    స్థిరరాశి నిర్మాణం htp_ద్విమాన_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_పునాది = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_పునాది, layout->off_src0) + (ith * layout->src0_బైట్లు_చొప్పున_thread);
    uint8_t * dst_spad_పునాది  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_పునాది, layout->off_dst)  + (ith * layout->dst_బైట్లు_చొప్పున_thread);
    స్థిరరాశి uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(స్థిరరాశి uint8_t, bctx->vtcm_పునాది, layout->off_src1);
    స్థిరరాశి int32_t * vtcm_src2 = VTCM_LAYOUT_PTR(స్థిరరాశి int32_t, bctx->vtcm_పునాది, layout->off_src2);
    size_t src0_spad_half    = layout->src0_spad_half_కొలత;
    size_t dst_spad_half     = layout->dst_spad_half_కొలత;

    dma_వరుస * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = ప్రారంభం_అడ్డువరుస;
    పూర్ణాంకం spad_idx = 0;

    కొరకు (పూర్ణాంకం k = 0; k < 2 && ir_prefetch < చివర_అడ్డువరుస; k++) {
        uint32_t current_ఖండం_కొలత = calc_ఖండం_కొలత(bctx, ir_prefetch, చివర_అడ్డువరుస, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->దత్తాంశం + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->దత్తాంశం  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_పునాది + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_పునాది  + spad_idx * dst_spad_half;

        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(dst_curr, d_spad), nb1, bctx->dst_అడ్డువరుస_కొలత_aligned, అడ్డువరుస_కొలత_బైట్లు, 0);
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(s0_spad, src0_curr), bctx->src0_అడ్డువరుస_కొలత_aligned, nb01, అడ్డువరుస_కొలత_బైట్లు, current_ఖండం_కొలత);
        ir_prefetch += current_ఖండం_కొలత;
        spad_idx ^= 1;
    }

    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];

    కొరకు (uint32_t ir = ప్రారంభం_అడ్డువరుస; ir < చివర_అడ్డువరుస; ) {
        uint32_t current_ఖండం_కొలత = calc_ఖండం_కొలత(bctx, ir, చివర_అడ్డువరుస, ne01, ne02);
        uint8_t * d_spad = (uint8_t *) dma_వరుస_తీయి(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_వరుస_తీయి(dma_q).dst;

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute_కూడు_id_t compute = (compute_కూడు_id_t) bctx->compute;
        compute(d_spad, s0_spad, vtcm_src1, vtcm_src2,
                i01, i02, ne01, src1_stride,
                current_ఖండం_కొలత, bctx->dst_అడ్డువరుస_కొలత_aligned, bctx->src0_అడ్డువరుస_కొలత_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->దత్తాంశం + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(dst_curr, d_spad), nb1, bctx->dst_అడ్డువరుస_కొలత_aligned, అడ్డువరుస_కొలత_బైట్లు, current_ఖండం_కొలత);

        ఒకవేళ (ir_prefetch < చివర_అడ్డువరుస) {
             uint32_t తదుపరి_ఖండం_కొలత = calc_ఖండం_కొలత(bctx, ir_prefetch, చివర_అడ్డువరుస, ne01, ne02);
             uint32_t p03, p02, p01, prem;
             p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
             prem = ir_prefetch - p03 * (ne02 * ne01);
             p02  = fastdiv(prem, &bctx->src0_dim1_div);
             p01  = prem - p02 * ne01;
             dma_addr_t s0_తదుపరి = src0->దత్తాంశం + p03 * nb03 + p02 * nb02 + p01 * nb01;
             dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(s0_spad, s0_తదుపరి), bctx->src0_అడ్డువరుస_కొలత_aligned, nb01, అడ్డువరుస_కొలత_బైట్లు, తదుపరి_ఖండం_కొలత);
             ir_prefetch += తదుపరి_ఖండం_కొలత;
        }
        ir += current_ఖండం_కొలత;
    }

    dma_వరుస_flush(dma_q);
}

స్థిర అంతఃవరుస శూన్య hvx_div_scalar_f32(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి దశాంశం val, స్థిరరాశి uint32_t అంకె_elems) {
    hvx_mul_scalar_f32(dst, src, 1.0f / val, అంకె_elems);
}

స్థిర అంతఃవరుస శూన్య hvx_div_scalar_f16(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి _Float16 val, స్థిరరాశి uint32_t అంకె_elems) {
    hvx_div_scalar_f16_aa(dst, src, val, అంకె_elems);
}

రకనిర్వచనం శూన్య (*compute_ద్విమాన_chunked_t)(
    uint8_t * పరిమిత dst,
    స్థిరరాశి uint8_t * పరిమిత src0,
    స్థిరరాశి uint8_t * పరిమిత src1,
    స్థిరరాశి uint32_t అంకె_elems
);

రకనిర్వచనం శూన్య (*compute_ద్విమాన_scalar_chunked_f32_t)(
    uint8_t * పరిమిత dst,
    స్థిరరాశి uint8_t * పరిమిత src0,
    స్థిరరాశి దశాంశం val,
    స్థిరరాశి uint32_t అంకె_elems
);

రకనిర్వచనం శూన్య (*compute_ద్విమాన_scalar_chunked_f16_t)(
    uint8_t * పరిమిత dst,
    స్థిరరాశి uint8_t * పరిమిత src0,
    స్థిరరాశి _Float16 val,
    స్థిరరాశి uint32_t అంకె_elems
);

నిర్మాణం ద్విమాన_chunked_context {
    నిర్మాణం htp_ops_context * octx;
    నిర్మాణం htp_ద్విమాన_vtcm_layout vtcm_layout;
    uint8_t *                vtcm_పునాది;
    uint32_t                 elem_ప్రారంభం;
    uint32_t                 nelem;
    uint32_t                 chunk_కొలత;
    uint32_t                 chunks_చొప్పున_thread;
    uint32_t                 మొత్తం_chunks;
    bool                     is_scalar;
    దశాంశం                    scalar_f32;
    _Float16                 scalar_f16;
    compute_ద్విమాన_chunked_t compute;
    compute_ద్విమాన_scalar_chunked_f32_t compute_scalar_f32;
    compute_ద్విమాన_scalar_chunked_f16_t compute_scalar_f16;
};

స్థిర శూన్య ద్విమాన_thread_chunked(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    (శూన్య) nth;
    నిర్మాణం ద్విమాన_chunked_context * ctx = (నిర్మాణం ద్విమాన_chunked_context *) దత్తాంశం;
    నిర్మాణం htp_ops_context * octx = ctx->octx;
    స్థిరరాశి నిర్మాణం htp_tensor * src0 = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * src1 = octx->src[1];
    స్థిరరాశి నిర్మాణం htp_tensor * dst  = octx->dst;

    స్థిరరాశి uint32_t ప్రారంభం_chunk = ctx->chunks_చొప్పున_thread * ith;
    స్థిరరాశి uint32_t చివర_chunk   = MIN(ప్రారంభం_chunk + ctx->chunks_చొప్పున_thread, ctx->మొత్తం_chunks);
    ఒకవేళ (ప్రారంభం_chunk >= చివర_chunk) {
        తిరిగి;
    }

    స్థిరరాశి uint32_t src0_type   = src0->type;
    స్థిరరాశి size_t   elem_కొలత   = (src0_type == HTP_TYPE_F32) ? పరిమాణం(దశాంశం) : పరిమాణం(_Float16);
    స్థిరరాశి uint32_t chunk_కొలత  = ctx->chunk_కొలత;
    స్థిరరాశి size_t   chunk_బైట్లు = ctx->vtcm_layout.src0_spad_half_కొలత;

    FARF(ఎగువ_2, "binary-chunked: %d/%d (%u:%u) chunks %u elems %u",
         ith, nth, ప్రారంభం_chunk, చివర_chunk, ctx->మొత్తం_chunks, ctx->nelem);

    స్థిరరాశి నిర్మాణం htp_ద్విమాన_vtcm_layout * layout = &ctx->vtcm_layout;
    uint8_t * src0_spad_పునాది = VTCM_LAYOUT_PTR(uint8_t, ctx->vtcm_పునాది, layout->off_src0) + (ith * layout->src0_బైట్లు_చొప్పున_thread);
    uint8_t * src1_spad_పునాది = ctx->is_scalar ? ఖాళీ : (VTCM_LAYOUT_PTR(uint8_t, ctx->vtcm_పునాది, layout->off_src1) + (ith * layout->src1_బైట్లు_చొప్పున_thread));
    uint8_t * dst_spad_పునాది  = VTCM_LAYOUT_PTR(uint8_t, ctx->vtcm_పునాది, layout->off_dst)  + (ith * layout->dst_బైట్లు_చొప్పున_thread);

    dma_వరుస * dma_q = octx->ctx->dma[ith];
    uint32_t prefetch_chunk = ప్రారంభం_chunk;
    పూర్ణాంకం spad_idx = 0;

    కొరకు (పూర్ణాంకం k = 0; k < 2 && prefetch_chunk < చివర_chunk; k++) {
        స్థిరరాశి uint32_t c_ప్రారంభం   = ctx->elem_ప్రారంభం + prefetch_chunk * chunk_కొలత;
        స్థిరరాశి uint32_t c_చివర     = MIN(c_ప్రారంభం + chunk_కొలత, ctx->elem_ప్రారంభం + ctx->nelem);
        స్థిరరాశి uint32_t cur_elems = c_చివర - c_ప్రారంభం;
        స్థిరరాశి uint32_t cur_బైట్లు = cur_elems * elem_కొలత;

        dma_addr_t s0_curr = src0->దత్తాంశం + (size_t) c_ప్రారంభం * elem_కొలత;
        dma_addr_t d_curr  = dst->దత్తాంశం  + (size_t) c_ప్రారంభం * elem_కొలత;

        uint8_t * s0_spad = src0_spad_పునాది + spad_idx * chunk_బైట్లు;
        uint8_t * d_spad  = dst_spad_పునాది  + spad_idx * chunk_బైట్లు;

        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(d_curr, d_spad), chunk_బైట్లు, chunk_బైట్లు, cur_బైట్లు, 0);
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(s0_spad, s0_curr), chunk_బైట్లు, chunk_బైట్లు, cur_బైట్లు, 1);
        ఒకవేళ (!ctx->is_scalar) {
            dma_addr_t s1_curr = src1->దత్తాంశం + (size_t) c_ప్రారంభం * elem_కొలత;
            uint8_t * s1_spad = src1_spad_పునాది + spad_idx * chunk_బైట్లు;
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(s1_spad, s1_curr), chunk_బైట్లు, chunk_బైట్లు, cur_బైట్లు, 1);
        }

        prefetch_chunk++;
        spad_idx ^= 1;
    }

    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];

    కొరకు (uint32_t c = ప్రారంభం_chunk; c < చివర_chunk; c++) {
        స్థిరరాశి uint32_t c_ప్రారంభం   = ctx->elem_ప్రారంభం + c * chunk_కొలత;
        స్థిరరాశి uint32_t c_చివర     = MIN(c_ప్రారంభం + chunk_కొలత, ctx->elem_ప్రారంభం + ctx->nelem);
        స్థిరరాశి uint32_t cur_elems = c_చివర - c_ప్రారంభం;
        స్థిరరాశి uint32_t cur_బైట్లు = cur_elems * elem_కొలత;

        uint8_t * d_spad  = (uint8_t *) dma_వరుస_తీయి(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_వరుస_తీయి(dma_q).dst;
        uint8_t * s1_spad = ctx->is_scalar ? ఖాళీ : (uint8_t *) dma_వరుస_తీయి(dma_q).dst;

        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) c);
        ఒకవేళ (ctx->is_scalar) {
            ఒకవేళ (src0_type == HTP_TYPE_F32) {
                ctx->compute_scalar_f32(d_spad, s0_spad, ctx->scalar_f32, cur_elems);
            } లేకపోతే {
                ctx->compute_scalar_f16(d_spad, s0_spad, ctx->scalar_f16, cur_elems);
            }
        } లేకపోతే {
            ctx->compute(d_spad, s0_spad, s1_spad, cur_elems);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) c);

        dma_addr_t dst_curr = dst->దత్తాంశం + (size_t) c_ప్రారంభం * elem_కొలత;
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(dst_curr, d_spad), chunk_బైట్లు, chunk_బైట్లు, cur_బైట్లు, 1);

        ఒకవేళ (prefetch_chunk < చివర_chunk) {
            స్థిరరాశి uint32_t pc_ప్రారంభం  = ctx->elem_ప్రారంభం + prefetch_chunk * chunk_కొలత;
            స్థిరరాశి uint32_t pc_చివర    = MIN(pc_ప్రారంభం + chunk_కొలత, ctx->elem_ప్రారంభం + ctx->nelem);
            స్థిరరాశి uint32_t p_elems   = pc_చివర - pc_ప్రారంభం;
            స్థిరరాశి uint32_t p_బైట్లు   = p_elems * elem_కొలత;

            dma_addr_t s0_తదుపరి = src0->దత్తాంశం + (size_t) pc_ప్రారంభం * elem_కొలత;
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(s0_spad, s0_తదుపరి), chunk_బైట్లు, chunk_బైట్లు, p_బైట్లు, 1);
            ఒకవేళ (!ctx->is_scalar) {
                dma_addr_t s1_తదుపరి = src1->దత్తాంశం + (size_t) pc_ప్రారంభం * elem_కొలత;
                dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(s1_spad, s1_తదుపరి), chunk_బైట్లు, chunk_బైట్లు, p_బైట్లు, 1);
            }

            prefetch_chunk++;
        }
    }

    dma_వరుస_flush(dma_q);
}

స్థిర పూర్ణాంకం execute_op_ద్విమాన(నిర్మాణం htp_ops_context * octx) {
    స్థిరరాశి నిర్మాణం htp_tensor * src0 = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * src1 = octx->src[1];
    స్థిరరాశి నిర్మాణం htp_tensor * dst  = octx->dst;
    స్థిరరాశి నిర్మాణం htp_ద్విమాన_kernel_params * kparams = (స్థిరరాశి నిర్మాణం htp_ద్విమాన_kernel_params *) octx->kernel_params;

    స్థిరరాశి uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    // Use packed row sizes for VTCM allocation and alignment
    స్థిరరాశి uint32_t src0_type = octx->src[0]->type;
    స్థిరరాశి size_t elem_కొలత = (src0_type == HTP_TYPE_F32) ? పరిమాణం(దశాంశం) : పరిమాణం(_Float16);
    స్థిరరాశి size_t src0_అడ్డువరుస_కొలత = src0->ne[0] * elem_కొలత;
    స్థిరరాశి size_t src1_అడ్డువరుస_కొలత = src1->ne[0] * elem_కొలత;
    స్థిరరాశి size_t dst_అడ్డువరుస_కొలత  = dst->ne[0]  * elem_కొలత;

    ఒకవేళ (kparams->kernel_type == HTP_ద్విమాన_KERNEL_CHUNKED) {
        స్థిరరాశి bool is_scalar = kparams->is_scalar || (src1->ne[0] == 1 && src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1);

        compute_ద్విమాన_chunked_t compute = ఖాళీ;
        compute_ద్విమాన_scalar_chunked_f32_t compute_scalar_f32 = ఖాళీ;
        compute_ద్విమాన_scalar_chunked_f16_t compute_scalar_f16 = ఖాళీ;

        ఒకవేళ (is_scalar) {
            ఒకవేళ (src0_type == HTP_TYPE_F32) {
                ఎంపిక (octx->op) {
                    స్థితి HTP_OP_కూడు: compute_scalar_f32 = hvx_కూడు_scalar_f32; ఆపు;
                    స్థితి HTP_OP_SUB: compute_scalar_f32 = hvx_sub_scalar_f32; ఆపు;
                    స్థితి HTP_OP_MUL: compute_scalar_f32 = hvx_mul_scalar_f32; ఆపు;
                    స్థితి HTP_OP_DIV: compute_scalar_f32 = hvx_div_scalar_f32; ఆపు;
                    అప్రమేయ: ఆపు;
                }
            } లేకపోతే ఒకవేళ (src0_type == HTP_TYPE_F16) {
                ఎంపిక (octx->op) {
                    స్థితి HTP_OP_కూడు: compute_scalar_f16 = hvx_కూడు_scalar_f16; ఆపు;
                    స్థితి HTP_OP_SUB: compute_scalar_f16 = hvx_sub_scalar_f16; ఆపు;
                    స్థితి HTP_OP_MUL: compute_scalar_f16 = hvx_mul_scalar_f16; ఆపు;
                    స్థితి HTP_OP_DIV: compute_scalar_f16 = hvx_div_scalar_f16; ఆపు;
                    అప్రమేయ: ఆపు;
                }
            }
            ఒకవేళ (!compute_scalar_f32 && !compute_scalar_f16) {
                తిరిగి HTP_స్థితి_NO_SUPPORT;
            }
        } లేకపోతే {
            ఒకవేళ (src0_type == HTP_TYPE_F32) {
                ఎంపిక (octx->op) {
                    స్థితి HTP_OP_కూడు: compute = hvx_కూడు_f32; ఆపు;
                    స్థితి HTP_OP_SUB: compute = hvx_sub_f32; ఆపు;
                    స్థితి HTP_OP_MUL: compute = hvx_mul_f32; ఆపు;
                    స్థితి HTP_OP_DIV: compute = hvx_div_f32; ఆపు;
                    అప్రమేయ: ఆపు;
                }
            } లేకపోతే ఒకవేళ (src0_type == HTP_TYPE_F16) {
                ఎంపిక (octx->op) {
                    స్థితి HTP_OP_కూడు: compute = hvx_కూడు_f16; ఆపు;
                    స్థితి HTP_OP_SUB: compute = hvx_sub_f16; ఆపు;
                    స్థితి HTP_OP_MUL: compute = hvx_mul_f16; ఆపు;
                    స్థితి HTP_OP_DIV: compute = hvx_div_f16; ఆపు;
                    అప్రమేయ: ఆపు;
                }
            }
            ఒకవేళ (!compute) {
                తిరిగి HTP_స్థితి_NO_SUPPORT;
            }
        }

        స్థిరరాశి uint32_t మొత్తం_elems = (uint32_t) (src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3]);
        ఒకవేళ (మొత్తం_elems == 0) {
            తిరిగి HTP_స్థితి_OK;
        }

        uint32_t elem_ప్రారంభం = 0;
        uint32_t nelem      = మొత్తం_elems;
        స్థిరరాశి uint32_t elems_చొప్పున_పంక్తి = (src0_type == HTP_TYPE_F32) ? 32 : 64;

        ఒకవేళ (octx->ctx->mdev.లెక్క > 1) {
            స్థిరరాశి bool can_విభజించు = htp_tensor_mdev_దత్తాంశం_aligned(dst) &&
                                   htp_tensor_mdev_దత్తాంశం_aligned(src0) &&
                                   (is_scalar || htp_tensor_mdev_దత్తాంశం_aligned(src1)) &&
                                   htp_tensor_is_contiguous(dst, elem_కొలత) &&
                                   htp_tensor_is_contiguous(src0, elem_కొలత) &&
                                   (is_scalar || htp_tensor_is_contiguous(src1, elem_కొలత));
            స్థిరరాశి నిర్మాణం htp_tensor_mdev_range range = htp_tensor_mdev_విభజన(
                మొత్తం_elems, can_విభజించు ? elems_చొప్పున_పంక్తి : 0,
                octx->ctx->mdev.idx, octx->ctx->mdev.లెక్క, &octx->ctx->mdev.లెక్క_div);
            elem_ప్రారంభం = range.ప్రారంభం;
            nelem      = range.లెక్క;
        }

        ఒకవేళ (nelem == 0) {
            తిరిగి HTP_స్థితి_OK;
        }

        ఒకవేళ (!htp_ops_context_సమితి_n_దారాలు(octx, kparams->n_దారాలు)) {
            తిరిగి HTP_స్థితి_INVAL_PARAMS;
        }

        నిర్మాణం htp_ద్విమాన_vtcm_layout vtcm_layout;
        htp_ద్విమాన_vtcm_layout_build(&vtcm_layout, kparams, octx->ctx->vtcm_కొలత);
        ఒకవేళ (vtcm_layout.మొత్తం_బైట్లు == 0 || vtcm_layout.మొత్తం_బైట్లు > octx->ctx->vtcm_కొలత) {
            తిరిగి HTP_స్థితి_VTCM_TOO_SMALL;
        }

        స్థిరరాశి uint32_t chunk_కొలత = kparams->chunk_కొలత > 0 ? kparams->chunk_కొలత : (32768 / elem_కొలత);
        స్థిరరాశి uint32_t మొత్తం_chunks = (nelem + chunk_కొలత - 1) / chunk_కొలత;
        స్థిరరాశి uint32_t n_దారాలు = (మొత్తం_chunks >= 2) ? MIN(octx->n_దారాలు, మొత్తం_chunks) : 1;
        స్థిరరాశి uint32_t chunks_చొప్పున_thread = (మొత్తం_chunks + n_దారాలు - 1) / n_దారాలు;

        దశాంశం scalar_f32 = 0.0f;
        _Float16 scalar_f16 = 0;
        ఒకవేళ (is_scalar) {
            uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, octx->ctx->vtcm_పునాది, vtcm_layout.off_src1);
            dma_వరుస * dma_q = octx->ctx->dma[0];
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(vtcm_src1, src1->దత్తాంశం), 128, 0, elem_కొలత, 1);
            dma_వరుస_తీయి(dma_q);

            ఒకవేళ (src0_type == HTP_TYPE_F32) {
                scalar_f32 = ((స్థిరరాశి దశాంశం *) vtcm_src1)[0];
            } లేకపోతే {
                scalar_f16 = ((స్థిరరాశి _Float16 *) vtcm_src1)[0];
            }
        }

        నిర్మాణం ద్విమాన_chunked_context cctx = {
            .octx                = octx,
            .vtcm_layout         = vtcm_layout,
            .vtcm_పునాది           = (uint8_t *) octx->ctx->vtcm_పునాది,
            .elem_ప్రారంభం          = elem_ప్రారంభం,
            .nelem               = nelem,
            .chunk_కొలత          = chunk_కొలత,
            .chunks_చొప్పున_thread   = chunks_చొప్పున_thread,
            .మొత్తం_chunks        = మొత్తం_chunks,
            .is_scalar           = is_scalar,
            .scalar_f32          = scalar_f32,
            .scalar_f16          = scalar_f16,
            .compute             = compute,
            .compute_scalar_f32  = compute_scalar_f32,
            .compute_scalar_f16  = compute_scalar_f16,
        };

        work_వరుస_నిరంతరం(octx->ctx->work_వరుస, ద్విమాన_thread_chunked, &cctx, n_దారాలు);
        తిరిగి HTP_స్థితి_OK;
    }

    uint32_t అడ్డువరుస_ప్రారంభం = 0;
    uint32_t nrows     = src0_nrows;

    ఒకవేళ (octx->ctx->mdev.లెక్క > 1) {
        uint32_t అడ్డువరుసలు_చొప్పున_chunk = 0;
        htp_tensor_mdev_అడ్డువరుసలు_చొప్పున_chunk(dst, (uint32_t) elem_కొలత, (uint32_t) dst_అడ్డువరుస_కొలత, &అడ్డువరుసలు_చొప్పున_chunk);
        స్థిరరాశి నిర్మాణం htp_tensor_mdev_range range = htp_tensor_mdev_విభజన(src0_nrows, అడ్డువరుసలు_చొప్పున_chunk,
                                                       octx->ctx->mdev.idx, octx->ctx->mdev.లెక్క, &octx->ctx->mdev.లెక్క_div);
        అడ్డువరుస_ప్రారంభం = range.ప్రారంభం;
        nrows     = range.లెక్క;
    }

    ఒకవేళ (nrows == 0) {
        తిరిగి HTP_స్థితి_OK;
    }

    ఒకవేళ (!htp_ops_context_సమితి_n_దారాలు(octx, kparams->n_దారాలు)) {
        తిరిగి HTP_స్థితి_INVAL_PARAMS;
    }

    స్థిరరాశి uint32_t n_దారాలు = octx->n_దారాలు;
    స్థిరరాశి size_t src0_అడ్డువరుస_కొలత_aligned = kparams->src0_అడ్డువరుస_కొలత_aligned;
    స్థిరరాశి size_t src1_అడ్డువరుస_కొలత_aligned = kparams->src1_అడ్డువరుస_కొలత_aligned;
    స్థిరరాశి size_t dst_అడ్డువరుస_కొలత_aligned  = kparams->dst_అడ్డువరుస_కొలత_aligned;

    నిర్మాణం htp_ద్విమాన_context bctx;
    bctx.vtcm_పునాది = (uint8_t *) octx->ctx->vtcm_పునాది;
    htp_ద్విమాన_vtcm_layout_build(&bctx.vtcm_layout, kparams, octx->ctx->vtcm_కొలత);

    ఒకవేళ (bctx.vtcm_layout.అడ్డువరుసలు_చొప్పున_బఫరు == 0 || bctx.vtcm_layout.మొత్తం_బైట్లు > octx->ctx->vtcm_కొలత) {
        తిరిగి HTP_స్థితి_VTCM_TOO_SMALL;
    }

    dma_వరుస * dma_q = octx->ctx->dma[0];
    uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, bctx.vtcm_పునాది, bctx.vtcm_layout.off_src1);
    ఒకవేళ (kparams->kernel_type == HTP_ద్విమాన_KERNEL_అడ్డువరుస_BCAST) {
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(vtcm_src1, src1->దత్తాంశం), bctx.vtcm_layout.src1_కొలత, 0, src1->ne[0] * elem_కొలత, 1);
    } లేకపోతే ఒకవేళ (kparams->kernel_type == HTP_ద్విమాన_KERNEL_SCALAR ||
               kparams->kernel_type == HTP_ద్విమాన_KERNEL_REPEAT ||
               kparams->kernel_type == HTP_ద్విమాన_KERNEL_COMPLEX) {
        ఒకవేళ (!kparams->src1_strided) {
            స్థిరరాశి size_t src1_బైట్లు = (size_t) src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3] * elem_కొలత;
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(vtcm_src1, src1->దత్తాంశం), bctx.vtcm_layout.src1_కొలత, 0, src1_బైట్లు, 1);
        } లేకపోతే {
            స్థిరరాశి uint32_t ne11 = src1->ne[1];
            స్థిరరాశి uint32_t ne12 = src1->ne[2];
            స్థిరరాశి uint32_t ne13 = src1->ne[3];
            స్థిరరాశి size_t అడ్డువరుస_బైట్లు = src1->ne[0] * elem_కొలత;
            కొరకు (uint32_t i13 = 0; i13 < ne13; i13++) {
                కొరకు (uint32_t i12 = 0; i12 < ne12; i12++) {
                    dma_addr_t src_addr = src1->దత్తాంశం + i13 * src1->nb[3] + i12 * src1->nb[2];
                    uint8_t * dst_ptr = vtcm_src1 + (i13 * ne12 * ne11 + i12 * ne11) * అడ్డువరుస_బైట్లు;
                    అయితే (!dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(dst_ptr, src_addr),
                                           అడ్డువరుస_బైట్లు, src1->nb[1], అడ్డువరుస_బైట్లు, ne11)) {
                        dma_వరుస_తీయి(dma_q);
                    }
                }
            }
        }
    } లేకపోతే ఒకవేళ (kparams->kernel_type == HTP_ద్విమాన_KERNEL_కూడు_ID) {
        స్థిరరాశి నిర్మాణం htp_tensor * src2 = octx->src[2];
        uint8_t * vtcm_src2 = VTCM_LAYOUT_PTR(uint8_t, bctx.vtcm_పునాది, bctx.vtcm_layout.off_src2);
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(vtcm_src1, src1->దత్తాంశం),
                       kparams->src1_అడ్డువరుస_కొలత_aligned, src1->nb[1],
                       src1->ne[0] * elem_కొలత, src1->ne[1]);
        స్థిరరాశి size_t src2_అడ్డువరుస_బైట్లు = src2->ne[0] * పరిమాణం(int32_t);
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(vtcm_src2, src2->దత్తాంశం),
                       src2_అడ్డువరుస_బైట్లు, src2->nb[1],
                       src2_అడ్డువరుస_బైట్లు, src2->ne[1]);
    }

    bctx.octx                  = octx;
    bctx.nrows_చొప్పున_thread      = fastdiv(nrows + n_దారాలు - 1, &octx->n_దారాలు_div);
    bctx.మొత్తం_అడ్డువరుసలు            = nrows;
    bctx.అడ్డువరుస_ప్రారంభం             = అడ్డువరుస_ప్రారంభం;
    bctx.ఖండం_max             = bctx.vtcm_layout.అడ్డువరుసలు_చొప్పున_బఫరు;
    bctx.src0_అడ్డువరుస_కొలత_aligned = src0_అడ్డువరుస_కొలత_aligned;
    bctx.src1_అడ్డువరుస_కొలత_aligned = src1_అడ్డువరుస_కొలత_aligned;
    bctx.dst_అడ్డువరుస_కొలత_aligned  = dst_అడ్డువరుస_కొలత_aligned;
    bctx.అడ్డువరుస_కొలత_బైట్లు        = src0_అడ్డువరుస_కొలత;

    bctx.src0_dim1_div  = init_fastdiv_విలువలు(src0->ne[1]);
    bctx.src0_dim2_div  = init_fastdiv_విలువలు(src0->ne[2]);
    bctx.src0_dim12_div = init_fastdiv_విలువలు(src0->ne[1] * src0->ne[2]);

    bctx.src1_dim1_div  = init_fastdiv_విలువలు(src1->ne[1]);
    bctx.src1_dim2_div  = init_fastdiv_విలువలు(src1->ne[2]);
    bctx.src1_dim3_div  = init_fastdiv_విలువలు(src1->ne[3]);

    bctx.విభజించు_at_ne01  = kparams->విభజించు_at_ne01;
    bctx.విభజించు_at_ne02  = kparams->విభజించు_at_ne02;

    కార్మికుడు_callback_t కార్మికుడు_func = ఖాళీ;
    శూన్య * compute_func = ఖాళీ;

    ఎంపిక (kparams->kernel_type) {
        స్థితి HTP_ద్విమాన_KERNEL_SAME_SHAPE:
            కార్మికుడు_func = ద్విమాన_thread_సదిశ_same_shape;
            ఒకవేళ (src0_type == HTP_TYPE_F32) {
                ఎంపిక (octx->op) {
                    స్థితి HTP_OP_కూడు: compute_func = compute_same_shape_కూడు_f32; ఆపు;
                    స్థితి HTP_OP_SUB: compute_func = compute_same_shape_sub_f32; ఆపు;
                    స్థితి HTP_OP_MUL: compute_func = compute_same_shape_mul_f32; ఆపు;
                    స్థితి HTP_OP_DIV: compute_func = compute_same_shape_div_f32; ఆపు;
                    అప్రమేయ: ఆపు;
                }
            } లేకపోతే ఒకవేళ (src0_type == HTP_TYPE_F16) {
                ఎంపిక (octx->op) {
                    స్థితి HTP_OP_కూడు: compute_func = compute_same_shape_కూడు_f16; ఆపు;
                    స్థితి HTP_OP_SUB: compute_func = compute_same_shape_sub_f16; ఆపు;
                    స్థితి HTP_OP_MUL: compute_func = compute_same_shape_mul_f16; ఆపు;
                    స్థితి HTP_OP_DIV: compute_func = compute_same_shape_div_f16; ఆపు;
                    అప్రమేయ: ఆపు;
                }
            }
            ఆపు;
        స్థితి HTP_ద్విమాన_KERNEL_అడ్డువరుస_BCAST:
            కార్మికుడు_func = ద్విమాన_thread_సదిశ_అడ్డువరుస_ప్రసారం;
            ఒకవేళ (src0_type == HTP_TYPE_F32) {
                ఎంపిక (octx->op) {
                    స్థితి HTP_OP_కూడు: compute_func = compute_అడ్డువరుస_bcast_కూడు_f32; ఆపు;
                    స్థితి HTP_OP_SUB: compute_func = compute_అడ్డువరుస_bcast_sub_f32; ఆపు;
                    స్థితి HTP_OP_MUL: compute_func = compute_అడ్డువరుస_bcast_mul_f32; ఆపు;
                    స్థితి HTP_OP_DIV: compute_func = compute_అడ్డువరుస_bcast_div_f32; ఆపు;
                    అప్రమేయ: ఆపు;
                }
            } లేకపోతే ఒకవేళ (src0_type == HTP_TYPE_F16) {
                ఎంపిక (octx->op) {
                    స్థితి HTP_OP_కూడు: compute_func = compute_అడ్డువరుస_bcast_కూడు_f16; ఆపు;
                    స్థితి HTP_OP_SUB: compute_func = compute_అడ్డువరుస_bcast_sub_f16; ఆపు;
                    స్థితి HTP_OP_MUL: compute_func = compute_అడ్డువరుస_bcast_mul_f16; ఆపు;
                    స్థితి HTP_OP_DIV: compute_func = compute_అడ్డువరుస_bcast_div_f16; ఆపు;
                    అప్రమేయ: ఆపు;
                }
            }
            ఆపు;
        స్థితి HTP_ద్విమాన_KERNEL_SCALAR:
            కార్మికుడు_func = ద్విమాన_thread_scalar;
            ఒకవేళ (src0_type == HTP_TYPE_F32) {
                ఎంపిక (octx->op) {
                    స్థితి HTP_OP_కూడు: compute_func = compute_scalar_కూడు_f32; ఆపు;
                    స్థితి HTP_OP_SUB: compute_func = compute_scalar_sub_f32; ఆపు;
                    స్థితి HTP_OP_MUL: compute_func = compute_scalar_mul_f32; ఆపు;
                    స్థితి HTP_OP_DIV: compute_func = compute_scalar_div_f32; ఆపు;
                    అప్రమేయ: ఆపు;
                }
            } లేకపోతే ఒకవేళ (src0_type == HTP_TYPE_F16) {
                ఎంపిక (octx->op) {
                    స్థితి HTP_OP_కూడు: compute_func = compute_scalar_కూడు_f16; ఆపు;
                    స్థితి HTP_OP_SUB: compute_func = compute_scalar_sub_f16; ఆపు;
                    స్థితి HTP_OP_MUL: compute_func = compute_scalar_mul_f16; ఆపు;
                    స్థితి HTP_OP_DIV: compute_func = compute_scalar_div_f16; ఆపు;
                    అప్రమేయ: ఆపు;
                }
            }
            ఆపు;
        స్థితి HTP_ద్విమాన_KERNEL_COMPLEX:
            కార్మికుడు_func = ద్విమాన_thread_సదిశ_complex;
            ఒకవేళ (src0_type == HTP_TYPE_F32) {
                ఎంపిక (octx->op) {
                    స్థితి HTP_OP_కూడు: compute_func = compute_complex_కూడు_f32; ఆపు;
                    స్థితి HTP_OP_SUB: compute_func = compute_complex_sub_f32; ఆపు;
                    స్థితి HTP_OP_MUL: compute_func = compute_complex_mul_f32; ఆపు;
                    స్థితి HTP_OP_DIV: compute_func = compute_complex_div_f32; ఆపు;
                    అప్రమేయ: ఆపు;
                }
            } లేకపోతే ఒకవేళ (src0_type == HTP_TYPE_F16) {
                ఎంపిక (octx->op) {
                    స్థితి HTP_OP_కూడు: compute_func = compute_complex_కూడు_f16; ఆపు;
                    స్థితి HTP_OP_SUB: compute_func = compute_complex_sub_f16; ఆపు;
                    స్థితి HTP_OP_MUL: compute_func = compute_complex_mul_f16; ఆపు;
                    స్థితి HTP_OP_DIV: compute_func = compute_complex_div_f16; ఆపు;
                    అప్రమేయ: ఆపు;
                }
            }
            ఆపు;
        స్థితి HTP_ద్విమాన_KERNEL_REPEAT:
            కార్మికుడు_func = ద్విమాన_thread_element_repeat;
            ఒకవేళ (src0_type == HTP_TYPE_F32) {
                ఎంపిక (octx->op) {
                    స్థితి HTP_OP_కూడు: compute_func = compute_repeat_కూడు_f32; ఆపు;
                    స్థితి HTP_OP_SUB: compute_func = compute_repeat_sub_f32; ఆపు;
                    స్థితి HTP_OP_MUL: compute_func = compute_repeat_mul_f32; ఆపు;
                    స్థితి HTP_OP_DIV: compute_func = compute_repeat_div_f32; ఆపు;
                    అప్రమేయ: ఆపు;
                }
            } లేకపోతే ఒకవేళ (src0_type == HTP_TYPE_F16) {
                ఎంపిక (octx->op) {
                    స్థితి HTP_OP_కూడు: compute_func = compute_repeat_కూడు_f16; ఆపు;
                    స్థితి HTP_OP_SUB: compute_func = compute_repeat_sub_f16; ఆపు;
                    స్థితి HTP_OP_MUL: compute_func = compute_repeat_mul_f16; ఆపు;
                    స్థితి HTP_OP_DIV: compute_func = compute_repeat_div_f16; ఆపు;
                    అప్రమేయ: ఆపు;
                }
            }
            ఆపు;
        స్థితి HTP_ద్విమాన_KERNEL_కూడు_ID:
            ఒకవేళ (octx->op == HTP_OP_కూడు_ID && src0_type == HTP_TYPE_F32) {
                కార్మికుడు_func = ద్విమాన_thread_కూడు_id_f32;
                compute_func = (శూన్య *) compute_కూడు_id_f32;
            }
            ఆపు;
        అప్రమేయ: ఆపు;
    }

    ఒకవేళ (!కార్మికుడు_func || !compute_func) {
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    bctx.compute = compute_func;

    ఒకవేళ (kparams->kernel_type == HTP_ద్విమాన_KERNEL_అడ్డువరుస_BCAST ||
        kparams->kernel_type == HTP_ద్విమాన_KERNEL_SCALAR ||
        kparams->kernel_type == HTP_ద్విమాన_KERNEL_REPEAT ||
        kparams->kernel_type == HTP_ద్విమాన_KERNEL_COMPLEX ||
        kparams->kernel_type == HTP_ద్విమాన_KERNEL_కూడు_ID) {
        dma_వరుస_flush(dma_q);
    }

    work_వరుస_నిరంతరం(octx->ctx->work_వరుస, కార్మికుడు_func, &bctx, n_దారాలు);

    తిరిగి HTP_స్థితి_OK;
}

పూర్ణాంకం op_ద్విమాన(నిర్మాణం htp_ops_context * octx) {
    స్థిరరాశి uint32_t src0_type = octx->src[0]->type;
    ఒకవేళ ((src0_type == HTP_TYPE_F32) || (src0_type == HTP_TYPE_F16)) {
        తిరిగి execute_op_ద్విమాన(octx);
    }

    తిరిగి HTP_స్థితి_NO_SUPPORT;
}
