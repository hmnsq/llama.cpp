#अन्तर्भाव "concat-ops.h"
#अन्तर्भाव "dma-queue.h"
#अन्तर्भाव "hex-common.h"
#अन्तर्भाव "dma-copy.h"
#अन्तर्भाव "hex-fastdiv.h"
#अन्तर्भाव "hex-profile.h"
#अन्तर्भाव "hexagon_protos.h"
#अन्तर्भाव "hexagon_types.h"
#अन्तर्भाव "htp-ctx.h"
#अन्तर्भाव "htp-fence.h"
#अन्तर्भाव "htp-ops.h"
#अन्तर्भाव "htp-tensor.h"
#अन्तर्भाव "htp-vtcm.h"
#अन्तर्भाव "hvx-utils.h"
#अन्तर्भाव "hvx_hexagon_protos.h"
#अन्तर्भाव "gather-rows.h"

#अन्तर्भाव <string.h>

संरचना htp_concat_context {
    संरचना htp_ops_context * octx;
    uint8_t * spad0_आधारः;
    uint8_t * spad1_आधारः;
    uint32_t  spad0_आकार_प्रत्येकम्_thread;
    uint32_t  spad1_आकार_प्रत्येकम्_thread;
    uint32_t  row_start;
    uint32_t  nrows;
    uint32_t  nrows_प्रत्येकम्_thread;
    uint32_t  nplanes;
    संरचना fastdiv_मानानि div_ne2;
};

स्थिर अन्तःस्थ dma_addr_t concat_plane_addr(ध्रुव संरचना htp_tensor * t, uint32_t p, ध्रुव संरचना fastdiv_मानानि * div_ne2, uint32_t ne2) {
    ध्रुव uint32_t i3 = fastdiv(p, div_ne2);
    ध्रुव uint32_t i2 = p - i3 * ne2;
    प्रत्यागम t->दत्त + i2 * t->nb[2] + i3 * t->nb[3];
}

स्थिर शून्य concat_2d_f32_transposed(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    संरचना htp_concat_context * cctx = (संरचना htp_concat_context *) दत्त;
    संरचना htp_ops_context * octx = cctx->octx;

    ध्रुव संरचना htp_tensor * src0 = octx->src[0];
    ध्रुव संरचना htp_tensor * src1 = octx->src[1];
    ध्रुव संरचना htp_tensor * dst  = octx->dst;

    ध्रुव uint32_t src0_ne0 = src0->ne[0];
    ध्रुव uint32_t src1_ne0 = src1->ne[0];

    ध्रुव uint32_t row_अन्तः = cctx->row_start + cctx->nrows;
    ध्रुव uint32_t start_i = cctx->row_start + ith * cctx->nrows_प्रत्येकम्_thread;
    ध्रुव uint32_t अन्तः_i   = (start_i + cctx->nrows_प्रत्येकम्_thread < row_अन्तः) ? (start_i + cctx->nrows_प्रत्येकम्_thread) : row_अन्तः;
    यदि (start_i >= अन्तः_i) प्रत्यागम;

    dma_queue * dma_q = octx->ctx->dma[ith];

    uint8_t * spad0_आधारः = cctx->spad0_आधारः + ith * cctx->spad0_आकार_प्रत्येकम्_thread;
    uint8_t * spad1_आधारः = cctx->spad1_आधारः + ith * cctx->spad1_आकार_प्रत्येकम्_thread;

    ध्रुव uint32_t खण्डम्_i = 32;
    ध्रुव uint32_t spad1_stride = खण्डम्_i * परिमाण(भिन्न);

    ध्रुव HVX_सदिशम् offsets = hvx_vec_gather_offsets_w(spad1_stride);
    ध्रुव uint32_t src1_ne0_padded = hex_आवर्तनम्_ऊर्ध्वम्(src1_ne0, 32);
    ध्रुव uint32_t src0_row_अष्टकानि  = src0_ne0 * परिमाण(भिन्न);
    ध्रुव uint32_t src0_row_padded = hex_आवर्तनम्_ऊर्ध्वम्(src0_row_अष्टकानि, VLEN);
    ध्रुव uint32_t src0_pre        = src0_row_padded - src0_row_अष्टकानि;
    ध्रुव uint32_t spad0_row_अष्टकानि = src0_row_padded + src1_ne0_padded * परिमाण(भिन्न);

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];

    ध्रुव संरचना fastdiv_मानानि * div_ne2 = &cctx->div_ne2;
    ध्रुव uint32_t ne2 = dst->ne[2];

    uint32_t p = 0;
    uint32_t i = start_i;

    ध्रुव dma_addr_t src1_addr = concat_plane_addr(src1, p, div_ne2, ne2) + i * src1->nb[1];
    dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(spad1_आधारः, src1_addr), spad1_stride, src1->nb[0], MIN(अन्तः_i - i, खण्डम्_i) * परिमाण(भिन्न), src1_ne0);

    ध्रुव dma_addr_t src0_addr = concat_plane_addr(src0, p, div_ne2, ne2) + i * src0->nb[1];
    dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(spad0_आधारः + src0_pre, src0_addr), spad0_row_अष्टकानि, src0->nb[1], src0_row_अष्टकानि, MIN(अन्तः_i - i, खण्डम्_i));

    dma_queue_निष्कासय(dma_q); // src1

    यावत् (p < cctx->nplanes) {
        ध्रुव uint32_t current_खण्डम्_i = MIN(अन्तः_i - i, खण्डम्_i);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        प्रति (uint32_t j = 0; j < src1_ne0; j += 32) {
            ध्रुव uint8_t * src_ptr = spad1_आधारः + j * spad1_stride;
            uint8_t * dst_ptr = spad0_आधारः + src0_row_padded + j * परिमाण(भिन्न);
            hvx_transpose_32x32_w_gather(dst_ptr, spad0_row_अष्टकानि, src_ptr, spad1_stride, offsets, current_खण्डम्_i, MIN(src1_ne0 - j, 32));
        }
        hvx_gather_sync(spad0_आधारः + src0_row_padded);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        uint32_t np = p;
        uint32_t ni = i + खण्डम्_i;
        यदि (ni >= अन्तः_i) {
            ni = start_i;
            np++;
        }
        ध्रुव bool has_अग्र = np < cctx->nplanes;
        ध्रुव uint32_t अग्र_खण्डम्_i = MIN(अन्तः_i - ni, खण्डम्_i);

        // spad1 is free after the gather sync, prefetch next src1 ahead of the dst write
        यदि (has_अग्र) {
            ध्रुव dma_addr_t nsrc1_addr = concat_plane_addr(src1, np, div_ne2, ne2) + ni * src1->nb[1];
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(spad1_आधारः, nsrc1_addr), spad1_stride, src1->nb[0], अग्र_खण्डम्_i * परिमाण(भिन्न), src1_ne0);
        }

        dma_queue_निष्कासय(dma_q); // src0

        ध्रुव dma_addr_t dst_addr = concat_plane_addr(dst, p, div_ne2, ne2) + i * dst->nb[1];
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(dst_addr, spad0_आधारः + src0_pre), dst->nb[1], spad0_row_अष्टकानि, (src0_ne0 + src1_ne0) * परिमाण(भिन्न), current_खण्डम्_i);

        यदि (has_अग्र) {
            dma_queue_निष्कासय(dma_q); // next src1
        }
        dma_queue_निष्कासय(dma_q); // dst

        // spad0 is free after the dst write
        यदि (has_अग्र) {
            ध्रुव dma_addr_t nsrc0_addr = concat_plane_addr(src0, np, div_ne2, ne2) + ni * src0->nb[1];
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(spad0_आधारः + src0_pre, nsrc0_addr), spad0_row_अष्टकानि, src0->nb[1], src0_row_अष्टकानि, अग्र_खण्डम्_i);
        }

        p = np;
        i = ni;
    }
    dma_queue_flush(dma_q);
}

स्थिर शून्य concat_2d_f16_transposed(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    संरचना htp_concat_context * cctx = (संरचना htp_concat_context *) दत्त;
    संरचना htp_ops_context * octx = cctx->octx;

    ध्रुव संरचना htp_tensor * src0 = octx->src[0];
    ध्रुव संरचना htp_tensor * src1 = octx->src[1];
    ध्रुव संरचना htp_tensor * dst  = octx->dst;

    ध्रुव uint32_t src0_ne0 = src0->ne[0];
    ध्रुव uint32_t src1_ne0 = src1->ne[0];

    ध्रुव uint32_t row_अन्तः = cctx->row_start + cctx->nrows;
    ध्रुव uint32_t start_i = cctx->row_start + ith * cctx->nrows_प्रत्येकम्_thread;
    ध्रुव uint32_t अन्तः_i   = (start_i + cctx->nrows_प्रत्येकम्_thread < row_अन्तः) ? (start_i + cctx->nrows_प्रत्येकम्_thread) : row_अन्तः;
    यदि (start_i >= अन्तः_i) प्रत्यागम;

    dma_queue * dma_q = octx->ctx->dma[ith];

    uint8_t * spad0_आधारः = cctx->spad0_आधारः + ith * cctx->spad0_आकार_प्रत्येकम्_thread;
    uint8_t * spad1_आधारः = cctx->spad1_आधारः + ith * cctx->spad1_आकार_प्रत्येकम्_thread;

    ध्रुव uint32_t खण्डम्_i = 64;
    ध्रुव uint32_t spad1_stride = खण्डम्_i * परिमाण(__fp16);

    ध्रुव HVX_सदिशम् offsets = hvx_vec_gather_offsets_h(spad1_stride);
    ध्रुव uint32_t src1_ne0_padded = hex_आवर्तनम्_ऊर्ध्वम्(src1_ne0, 64);
    ध्रुव uint32_t src0_row_अष्टकानि  = src0_ne0 * परिमाण(__fp16);
    ध्रुव uint32_t src0_row_padded = hex_आवर्तनम्_ऊर्ध्वम्(src0_row_अष्टकानि, VLEN);
    ध्रुव uint32_t src0_pre        = src0_row_padded - src0_row_अष्टकानि;
    ध्रुव uint32_t spad0_row_अष्टकानि = src0_row_padded + src1_ne0_padded * परिमाण(__fp16);

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];

    ध्रुव संरचना fastdiv_मानानि * div_ne2 = &cctx->div_ne2;
    ध्रुव uint32_t ne2 = dst->ne[2];

    uint32_t p = 0;
    uint32_t i = start_i;

    ध्रुव dma_addr_t src1_addr = concat_plane_addr(src1, p, div_ne2, ne2) + i * src1->nb[1];
    dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(spad1_आधारः, src1_addr), spad1_stride, src1->nb[0], MIN(अन्तः_i - i, खण्डम्_i) * परिमाण(__fp16), src1_ne0);

    ध्रुव dma_addr_t src0_addr = concat_plane_addr(src0, p, div_ne2, ne2) + i * src0->nb[1];
    dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(spad0_आधारः + src0_pre, src0_addr), spad0_row_अष्टकानि, src0->nb[1], src0_row_अष्टकानि, MIN(अन्तः_i - i, खण्डम्_i));

    dma_queue_निष्कासय(dma_q); // src1

    यावत् (p < cctx->nplanes) {
        ध्रुव uint32_t current_खण्डम्_i = MIN(अन्तः_i - i, खण्डम्_i);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        प्रति (uint32_t j = 0; j < src1_ne0; j += 64) {
            ध्रुव uint8_t * src_ptr = spad1_आधारः + j * spad1_stride;
            uint8_t * dst_ptr = spad0_आधारः + src0_row_padded + j * परिमाण(__fp16);
            hvx_transpose_64x64_h_gather(dst_ptr, spad0_row_अष्टकानि, src_ptr, spad1_stride, offsets, current_खण्डम्_i, MIN(src1_ne0 - j, 64));
        }
        hvx_gather_sync(spad0_आधारः + src0_row_padded);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        uint32_t np = p;
        uint32_t ni = i + खण्डम्_i;
        यदि (ni >= अन्तः_i) {
            ni = start_i;
            np++;
        }
        ध्रुव bool has_अग्र = np < cctx->nplanes;
        ध्रुव uint32_t अग्र_खण्डम्_i = MIN(अन्तः_i - ni, खण्डम्_i);

        // spad1 is free after the gather sync, prefetch next src1 ahead of the dst write
        यदि (has_अग्र) {
            ध्रुव dma_addr_t nsrc1_addr = concat_plane_addr(src1, np, div_ne2, ne2) + ni * src1->nb[1];
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(spad1_आधारः, nsrc1_addr), spad1_stride, src1->nb[0], अग्र_खण्डम्_i * परिमाण(__fp16), src1_ne0);
        }

        dma_queue_निष्कासय(dma_q); // src0

        ध्रुव dma_addr_t dst_addr = concat_plane_addr(dst, p, div_ne2, ne2) + i * dst->nb[1];
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(dst_addr, spad0_आधारः + src0_pre), dst->nb[1], spad0_row_अष्टकानि, (src0_ne0 + src1_ne0) * परिमाण(__fp16), current_खण्डम्_i);

        यदि (has_अग्र) {
            dma_queue_निष्कासय(dma_q); // next src1
        }
        dma_queue_निष्कासय(dma_q); // dst

        // spad0 is free after the dst write
        यदि (has_अग्र) {
            ध्रुव dma_addr_t nsrc0_addr = concat_plane_addr(src0, np, div_ne2, ne2) + ni * src0->nb[1];
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(spad0_आधारः + src0_pre, nsrc0_addr), spad0_row_अष्टकानि, src0->nb[1], src0_row_अष्टकानि, अग्र_खण्डम्_i);
        }

        p = np;
        i = ni;
    }
    dma_queue_flush(dma_q);
}

स्थिर पूर्णाङ्क concat_regular(संरचना htp_ops_context * octx, पूर्णाङ्क dim, uint32_t type_आकार) {
    ध्रुव संरचना htp_tensor * src0 = octx->src[0];
    ध्रुव संरचना htp_tensor * src1 = octx->src[1];
    ध्रुव संरचना htp_tensor * dst  = octx->dst;

    संरचना htp_tensor view0 = *dst;
    संरचना htp_tensor view1 = *dst;
    प्रति (पूर्णाङ्क d = 0; d < HTP_OP_MAX_DIMS; d++) {
        view0.ne[d] = src0->ne[d];
        view1.ne[d] = src1->ne[d];
    }
    view1.दत्त += src0->ne[dim] * dst->nb[dim];

    ध्रुव uint32_t योग_पङ्क्तिकाः_0 = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ध्रुव uint32_t योग_पङ्क्तिकाः_1 = src1->ne[1] * src1->ne[2] * src1->ne[3];

    uint32_t rstart0 = 0, nrows0 = योग_पङ्क्तिकाः_0;
    uint32_t rstart1 = 0, nrows1 = योग_पङ्क्तिकाः_1;

    यदि (octx->ctx->mdev.गणनम् > 1) {
        ध्रुव संरचना htp_tensor_mdev_range range0 = htp_tensor_mdev_partition(
            योग_पङ्क्तिकाः_0, 1, octx->ctx->mdev.idx, octx->ctx->mdev.गणनम्, &octx->ctx->mdev.गणनम्_div);
        rstart0 = range0.start;
        nrows0  = range0.गणनम्;

        ध्रुव संरचना htp_tensor_mdev_range range1 = htp_tensor_mdev_partition(
            योग_पङ्क्तिकाः_1, 1, octx->ctx->mdev.idx, octx->ctx->mdev.गणनम्, &octx->ctx->mdev.गणनम्_div);
        rstart1 = range1.start;
        nrows1  = range1.गणनम्;
    }

    dma_queue * q = octx->ctx->dma[0];

    dma_cpy_sametype_sameshape_range(q, &view0, src0, type_आकार, rstart0, nrows0);
    dma_cpy_sametype_sameshape_range(q, &view1, src1, type_आकार, rstart1, nrows1);
    dma_queue_flush(q);

    प्रत्यागम HTP_स्थितिः_OK;
}

स्थिर पूर्णाङ्क concat_transposed(संरचना htp_ops_context * octx, ध्रुव संरचना htp_concat_kernel_params * kparams, uint32_t type_आकार) {
    यदि (!htp_ops_context_समुच्चयः_n_तन्तवः(octx, kparams->n_तन्तवः)) {
        प्रत्यागम HTP_स्थितिः_INVAL_PARAMS;
    }

    ध्रुव संरचना htp_tensor * dst = octx->dst;

    ध्रुव uint32_t योग_पङ्क्तिकाः = dst->ne[1];
    uint32_t row_start = 0;
    uint32_t nrows     = योग_पङ्क्तिकाः;
    यदि (octx->ctx->mdev.गणनम् > 1) {
        ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(योग_पङ्क्तिकाः, 1, octx->ctx->mdev.idx, octx->ctx->mdev.गणनम्, &octx->ctx->mdev.गणनम्_div);
        row_start = range.start;
        nrows     = range.गणनम्;
    }

    यदि (nrows == 0 || dst->ne[2] == 0 || dst->ne[3] == 0) {
        प्रत्यागम HTP_स्थितिः_OK;
    }

    यदि (kparams->vtcm_आकार > octx->ctx->vtcm_आकार) {
        प्रत्यागम HTP_स्थितिः_VTCM_TOO_SMALL;
    }

    ध्रुव uint32_t n_तन्तवः = octx->n_तन्तवः;

    // layout precomputed on host; kept for reference:
    // struct htp_concat_transposed_vtcm_layout layout;
    // htp_concat_transposed_vtcm_layout_build(&layout, octx->src[0]->ne[0], octx->src[1]->ne[0], type_size, n_threads);

    uint8_t * vtcm_आधारः = (uint8_t *) octx->ctx->vtcm_आधारः;

    संरचना htp_concat_context cctx;
    cctx.octx                  = octx;
    cctx.spad0_आधारः            = vtcm_आधारः;
    cctx.spad1_आधारः            = vtcm_आधारः + n_तन्तवः * kparams->u.transposed.spad0_आकार_प्रत्येकम्_thread;
    cctx.spad0_आकार_प्रत्येकम्_thread = kparams->u.transposed.spad0_आकार_प्रत्येकम्_thread;
    cctx.spad1_आकार_प्रत्येकम्_thread = kparams->u.transposed.spad1_आकार_प्रत्येकम्_thread;
    cctx.row_start             = row_start;
    cctx.nrows                 = nrows;
    cctx.nplanes               = dst->ne[2] * dst->ne[3];
    cctx.div_ne2               = init_fastdiv_मानानि(dst->ne[2]);
    cctx.nrows_प्रत्येकम्_thread      = fastdiv(nrows + n_तन्तवः - 1, &octx->n_तन्तवः_div);

    work_queue_func_t कार्यकर्ता_func = (type_आकार == 4) ? concat_2d_f32_transposed : concat_2d_f16_transposed;
    work_queue_सततम्(octx->ctx->work_queue, कार्यकर्ता_func, &cctx, n_तन्तवः);
    प्रत्यागम HTP_स्थितिः_OK;
}

पूर्णाङ्क op_concat(संरचना htp_ops_context * octx) {
    ध्रुव संरचना htp_concat_kernel_params * kparams = (ध्रुव संरचना htp_concat_kernel_params *) octx->kernel_params;
    ध्रुव संरचना htp_tensor * src0 = octx->src[0];
    ध्रुव संरचना htp_tensor * src1 = octx->src[1];
    ध्रुव संरचना htp_tensor * dst  = octx->dst;

    पूर्णाङ्क स्थितिः = HTP_स्थितिः_OK;
    ध्रुव uint32_t type_आकार = (dst->type == HTP_TYPE_F32 || dst->type == HTP_TYPE_I32) ? 4 : 2;

    विकल्प (kparams->kernel_type) {
        पक्ष HTP_CONCAT_KERNEL_REGULAR:
            स्थितिः = concat_regular(octx, kparams->dim, type_आकार);
            भङ्ग;

        पक्ष HTP_CONCAT_KERNEL_TRANSPOSED:
            स्थितिः = concat_transposed(octx, kparams, type_आकार);
            भङ्ग;

        पक्ष HTP_CONCAT_KERNEL_GATHER_पङ्क्तिकाः:
            स्थितिः = htp_gather_पङ्क्तिकाः(octx, src0, src1, &kparams->u.gather, kparams->n_तन्तवः);
            भङ्ग;

        स्वभाव:
            स्थितिः = HTP_स्थितिः_NO_SUPPORT;
            भङ्ग;
    }

    htp_ops_context_समुच्चयः_स्थितिः(octx, स्थितिः);

    प्रत्यागम octx->स्थितिः;
}
