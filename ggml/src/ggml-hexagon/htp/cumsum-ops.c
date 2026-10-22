#निर्देशः clang diagnostic ignored "-Wunused-variable"
#निर्देशः clang diagnostic ignored "-Wunused-function"
#निर्देशः clang diagnostic ignored "-Wunused-but-set-variable"

#अन्तर्भाव <HAP_farf.h>
#अन्तर्भाव <HAP_perf.h>

#परिभाषय GGML_COMMON_DECL_C
#अन्तर्भाव "ggml-common.h"
#अन्तर्भाव "hex-common.h"
#अन्तर्भाव "hex-profile.h"
#अन्तर्भाव "htp-ctx.h"
#अन्तर्भाव "htp-ops.h"
#अन्तर्भाव "htp-tensor.h"
#अन्तर्भाव "hvx-types.h"
#अन्तर्भाव "hvx-utils.h"
#अन्तर्भाव "dma-queue.h"

#परिभाषय htp_cumsum_tensors_preamble                         \
    ध्रुव संरचना htp_tensor * सीमित src0 = octx->src[0]; \
    ध्रुव संरचना htp_tensor * सीमित dst  = octx->dst;    \
                                                            \
    ध्रुव uint32_t ne00 = src0->ne[0];                      \
    ध्रुव uint32_t ne01 = src0->ne[1];                      \
    ध्रुव uint32_t ne02 = src0->ne[2];                      \
    ध्रुव uint32_t ne03 = src0->ne[3];                      \
                                                            \
    ध्रुव uint32_t ne0 = dst->ne[0];                        \
    ध्रुव uint32_t ne1 = dst->ne[1];                        \
    ध्रुव uint32_t ne2 = dst->ne[2];                        \
    ध्रुव uint32_t ne3 = dst->ne[3];                        \
                                                            \
    ध्रुव uint32_t nb00 = src0->nb[0];                      \
    ध्रुव uint32_t nb01 = src0->nb[1];                      \
    ध्रुव uint32_t nb02 = src0->nb[2];                      \
    ध्रुव uint32_t nb03 = src0->nb[3];                      \
                                                            \
    ध्रुव uint32_t nb0 = dst->nb[0];                        \
    ध्रुव uint32_t nb1 = dst->nb[1];                        \
    ध्रुव uint32_t nb2 = dst->nb[2];                        \
    ध्रुव uint32_t nb3 = dst->nb[3];

संरचना htp_cumsum_context {
    संरचना htp_ops_context * octx;
    size_t          src_row_आकार;
    size_t          dst_row_आकार;
    size_t          src_row_आकार_aligned;
    size_t          dst_row_आकार_aligned;
    uint32_t        पङ्क्तिकाः_प्रत्येकम्_thread;
    uint32_t        योग_पङ्क्तिकाः;
    uint32_t        row_start;
};

#परिभाषय htp_cumsum_preamble                                                \
    संरचना htp_cumsum_context * cctx = (संरचना htp_cumsum_context *) दत्त; \
    संरचना htp_ops_context *    octx = cctx->octx;                         \
    htp_cumsum_tensors_preamble;                                           \
    dma_queue * dma_q = octx->ctx->dma[ith];

// ---------------------------------------------------------------------------
// HVX prefix scan helpers
// ---------------------------------------------------------------------------

#यदि __HVX_ARCH__ > 75
स्थिर अन्तःस्थ HVX_सदिशम् hvx_cumsum_vadd(HVX_सदिशम् a, HVX_सदिशम् b) {
    प्रत्यागम Q6_Vsf_vadd_VsfVsf(a, b);
}
#अन्यथा
स्थिर अन्तःस्थ HVX_सदिशम् hvx_cumsum_vadd(HVX_सदिशम् a, HVX_सदिशम् b) {
    प्रत्यागम Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
}
#यदि_अन्तः  // __HVX_ARCH__ > 75

स्थिर अन्तःस्थ HVX_सदिशम् hvx_उपसर्गः_अवलोकनम्_f32(HVX_सदिशम् v, HVX_सदिशम् carry_अन्तर्) {
    ध्रुव HVX_सदिशम् zero = Q6_V_vsplat_R(0);

    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero,  4));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero,  8));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 16));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 32));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 64));
    v = hvx_cumsum_vadd(v, carry_अन्तर्);

    प्रत्यागम v;
}

स्थिर अन्तःस्थ HVX_सदिशम् hvx_splat_चरमः_f32(HVX_सदिशम् v) {
    प्रत्यागम hvx_vec_repl4(Q6_V_vror_VR(v, 124));
}

स्थिर अन्तःस्थ शून्य hvx_cumsum_row_f32(ध्रुव भिन्न * सीमित src, भिन्न * सीमित dst, uint32_t n) {
    ध्रुव uint32_t nvec = n / VLEN_FP32;
    ध्रुव uint32_t nloe = n % VLEN_FP32;

    HVX_सदिशम् carry = Q6_V_vsplat_R(0);

    प्रति (uint32_t i = 0; i < nvec; i++) {
        HVX_सदिशम् v = *((ध्रुव HVX_UVector *) (src + i * VLEN_FP32));
        v = hvx_उपसर्गः_अवलोकनम्_f32(v, carry);
        hvx_vec_store_u(dst + i * VLEN_FP32, VLEN, v);
        carry = hvx_splat_चरमः_f32(v);
    }

    यदि (nloe) {
        भिन्न acc = hvx_vec_get_f32(carry);
        ध्रुव भिन्न * src_पुच्छ = src + nvec * VLEN_FP32;
        भिन्न       * dst_पुच्छ = dst + nvec * VLEN_FP32;
        प्रति (uint32_t i = 0; i < nloe; i++) {
            acc        += src_पुच्छ[i];
            dst_पुच्छ[i] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Per thread worker: Double-buffered DMA
// ---------------------------------------------------------------------------

स्थिर शून्य cumsum_thread_f32_dma(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    htp_cumsum_preamble;

    ध्रुव uint32_t ir0 = cctx->row_start + cctx->पङ्क्तिकाः_प्रत्येकम्_thread * ith;
    ध्रुव uint32_t ir1 = MIN(ir0 + cctx->पङ्क्तिकाः_प्रत्येकम्_thread, cctx->row_start + cctx->योग_पङ्क्तिकाः);

    यदि (ir0 >= ir1) {
        प्रत्यागम;
    }

    ध्रुव size_t src_row_आकार         = cctx->src_row_आकार;
    ध्रुव size_t dst_row_आकार         = cctx->dst_row_आकार;
    ध्रुव size_t src_row_आकार_aligned = cctx->src_row_आकार_aligned;
    ध्रुव size_t dst_row_आकार_aligned = cctx->dst_row_आकार_aligned;

    ध्रुव dma_addr_t src_दत्त = src0->दत्त;
    ध्रुव dma_addr_t dst_दत्त = dst->दत्त;

    uint8_t * src_spad = octx->src0_spad.दत्त + (ith * src_row_आकार_aligned * 2);
    uint8_t * dst_spad = octx->dst_spad.दत्त  + (ith * dst_row_आकार_aligned * 2);

    प्रति (uint32_t ir = ir0, spad_idx = 0; ir < ir1 && spad_idx < 2; ir++, spad_idx++) {
        // Dummy dst writeback to establish queue ordering
        dma_queue_प्रक्षिप(dma_q,
                       dma_make_दत्त(dst_दत्त, dst_spad + (spad_idx * dst_row_आकार_aligned)),
                       dst_row_आकार, dst_row_आकार_aligned, dst_row_आकार, 0);

        dma_queue_प्रक्षिप(dma_q,
                       dma_make_दत्त(src_spad + (spad_idx * src_row_आकार_aligned),
                                     src_दत्त + (ir * src_row_आकार)),
                       src_row_आकार_aligned, src_row_आकार, src_row_आकार, 1);
    }

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];

    प्रति (uint32_t ir = ir0; ir < ir1; ir++) {
        भिन्न * dst_spad_row = (भिन्न *) dma_queue_निष्कासय(dma_q).src;
        भिन्न * src_spad_row = (भिन्न *) dma_queue_निष्कासय(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        hvx_cumsum_row_f32(src_spad_row, dst_spad_row, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_queue_प्रक्षिप(dma_q,
                       dma_make_दत्त(dst_दत्त + (ir * dst_row_आकार), dst_spad_row),
                       dst_row_आकार, dst_row_आकार_aligned, dst_row_आकार, 1);

        ध्रुव uint32_t अग्र_row = ir + 2;
        यदि (अग्र_row < ir1) {
            dma_queue_प्रक्षिप(dma_q,
                           dma_make_दत्त(src_spad_row, src_दत्त + (अग्र_row * src_row_आकार)),
                           src_row_आकार_aligned, src_row_आकार, src_row_आकार, 1);
        }
    }

    dma_queue_flush(dma_q);

    FARF(उच्चः_2, "cumsum-f32-dma %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

// ---------------------------------------------------------------------------
// Per thread worker: Direct HVX (no DMA)
// ---------------------------------------------------------------------------

स्थिर शून्य cumsum_thread_f32(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    htp_cumsum_preamble;

    ध्रुव uint8_t * src_दत्त = (ध्रुव uint8_t *) src0->दत्त;
    uint8_t *       dst_दत्त = (uint8_t *) dst->दत्त;

    ध्रुव uint32_t ir0 = cctx->row_start + cctx->पङ्क्तिकाः_प्रत्येकम्_thread * ith;
    ध्रुव uint32_t ir1 = MIN(ir0 + cctx->पङ्क्तिकाः_प्रत्येकम्_thread, cctx->row_start + cctx->योग_पङ्क्तिकाः);

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    प्रति (uint32_t ir = ir0; ir < ir1; ir++) {
        ध्रुव भिन्न * सीमित src_row = (ध्रुव भिन्न *) (src_दत्त + ir * cctx->src_row_आकार);
        भिन्न       * सीमित dst_row = (भिन्न *)       (dst_दत्त + ir * cctx->dst_row_आकार);
        hvx_cumsum_row_f32(src_row, dst_row, ne00);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    FARF(उच्चः_2, "cumsum-f32 %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

पूर्णाङ्क op_cumsum_f32(संरचना htp_ops_context * octx) {
    ध्रुव संरचना htp_tensor * src0 = octx->src[0];
    ध्रुव संरचना htp_tensor * dst  = octx->dst;

    ध्रुव uint32_t योग_पङ्क्तिकाः      = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ध्रुव size_t dst_दत्त_row_आकार = dst->ne[0] * परिमाण(भिन्न);

    uint32_t row_start = 0;
    uint32_t nrows     = योग_पङ्क्तिकाः;

    यदि (octx->ctx->mdev.गणनम् > 1) {
        uint32_t पङ्क्तिकाः_प्रत्येकम्_chunk = 0;
        htp_tensor_mdev_पङ्क्तिकाः_प्रत्येकम्_chunk(dst, परिमाण(भिन्न), (uint32_t) dst_दत्त_row_आकार, &पङ्क्तिकाः_प्रत्येकम्_chunk);
        ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(योग_पङ्क्तिकाः, पङ्क्तिकाः_प्रत्येकम्_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.गणनम्, &octx->ctx->mdev.गणनम्_div);
        row_start = range.start;
        nrows     = range.गणनम्;
    }

    यदि (nrows == 0) {
        प्रत्यागम HTP_स्थितिः_OK;
    }

    ध्रुव uint32_t n_तन्तवः = octx->n_तन्तवः;

    ध्रुव size_t src_row_आकार         = src0->nb[1];
    ध्रुव size_t dst_row_आकार         = dst->nb[1];
    ध्रुव size_t src_row_आकार_aligned = hex_आवर्तनम्_ऊर्ध्वम्(src_row_आकार, VLEN);
    ध्रुव size_t dst_row_आकार_aligned = hex_आवर्तनम्_ऊर्ध्वम्(dst_row_आकार, VLEN);

    // 2 ping-pong buffers per thread for src and dst
    ध्रुव size_t spad_प्रत्येकम्_thread = 2 * (src_row_आकार_aligned + dst_row_आकार_aligned);

    octx->src0_spad.आकार_प्रत्येकम्_thread = src_row_आकार_aligned * 2;
    octx->dst_spad.आकार_प्रत्येकम्_thread  = dst_row_आकार_aligned * 2;

    octx->src0_spad.आकार  = n_तन्तवः * octx->src0_spad.आकार_प्रत्येकम्_thread;
    octx->dst_spad.आकार   = n_तन्तवः * octx->dst_spad.आकार_प्रत्येकम्_thread;

    octx->src0_spad.दत्त  = octx->ctx->vtcm_आधारः;                        octx->src0_spad.src = रिक्त;
    octx->dst_spad.दत्त   = octx->src0_spad.दत्त + octx->src0_spad.आकार; octx->dst_spad.src  = रिक्त;

    संरचना htp_cumsum_context cctx = {
        .octx                 = octx,
        .src_row_आकार         = src_row_आकार,
        .dst_row_आकार         = dst_row_आकार,
        .src_row_आकार_aligned = src_row_आकार_aligned,
        .dst_row_आकार_aligned = dst_row_आकार_aligned,
        .पङ्क्तिकाः_प्रत्येकम्_thread      = fastdiv(nrows + n_तन्तवः - 1, &octx->n_तन्तवः_div),
        .योग_पङ्क्तिकाः           = nrows,
        .row_start            = row_start,
    };

    यदि (octx->ctx->vtcm_आकार < spad_प्रत्येकम्_thread * n_तन्तवः) {
        यदि (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
            प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
        }
        work_queue_सततम्(octx->ctx->work_queue, cumsum_thread_f32, &cctx, n_तन्तवः);
    } अन्यथा {
        work_queue_सततम्(octx->ctx->work_queue, cumsum_thread_f32_dma, &cctx, n_तन्तवः);
    }

    प्रत्यागम HTP_स्थितिः_OK;
}

पूर्णाङ्क op_cumsum(संरचना htp_ops_context * octx) {
    ध्रुव संरचना htp_tensor * dst = octx->dst;

    विकल्प (dst->type) {
        पक्ष HTP_TYPE_F32:
            प्रत्यागम op_cumsum_f32(octx);
        स्वभाव:
            प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
    }
}
