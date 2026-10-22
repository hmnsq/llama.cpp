#निर्देशः clang diagnostic ignored "-Wunused-variable"
#निर्देशः clang diagnostic ignored "-Wunused-function"
#निर्देशः clang diagnostic ignored "-Wunused-but-set-variable"

#अन्तर्भाव <HAP_farf.h>
#अन्तर्भाव <HAP_perf.h>

#अन्तर्भाव <string.h>

#अन्तर्भाव "hvx-utils.h"

#परिभाषय GGML_COMMON_DECL_C
#अन्तर्भाव "ggml-common.h"
#अन्तर्भाव "htp-ctx.h"
#अन्तर्भाव "hex-common.h"
#अन्तर्भाव "hex-profile.h"
#अन्तर्भाव "htp-ops.h"
#अन्तर्भाव "htp-tensor.h"

संरचना htp_repeat_context {
    संरचना htp_ops_context * octx;

    uint32_t nr0;
    uint32_t nr1;
    uint32_t nr2;
    uint32_t nr3;

    uint32_t nrows_प्रत्येकम्_thread;
    uint32_t योग_dst_पङ्क्तिकाः;  // ne1 * ne2 * ne3
    uint32_t row_start;

    size_t   type_आकार;
};

स्थिर शून्य repeat_job_प्रत्येकम्_thread(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    ध्रुव संरचना htp_repeat_context * rctx = (ध्रुव संरचना htp_repeat_context *) दत्त;
    संरचना htp_ops_context * octx = rctx->octx;
    ध्रुव संरचना htp_tensor * src = octx->src[0];
    ध्रुव संरचना htp_tensor * dst = octx->dst;

    ध्रुव uint32_t ne00 = src->ne[0];
    ध्रुव uint32_t ne01 = src->ne[1];
    ध्रुव uint32_t ne02 = src->ne[2];
    ध्रुव uint32_t ne03 = src->ne[3];

    ध्रुव uint32_t nb00 = src->nb[0];
    ध्रुव uint32_t nb01 = src->nb[1];
    ध्रुव uint32_t nb02 = src->nb[2];
    ध्रुव uint32_t nb03 = src->nb[3];

    ध्रुव uint32_t ne0 = dst->ne[0];
    ध्रुव uint32_t ne1 = dst->ne[1];
    ध्रुव uint32_t ne2 = dst->ne[2];
    ध्रुव uint32_t ne3 = dst->ne[3];

    ध्रुव uint32_t nb0 = dst->nb[0];
    ध्रुव uint32_t nb1 = dst->nb[1];
    ध्रुव uint32_t nb2 = dst->nb[2];
    ध्रुव uint32_t nb3 = dst->nb[3];

    ध्रुव uint32_t nr0 = rctx->nr0;
    ध्रुव uint32_t nr1 = rctx->nr1;
    ध्रुव uint32_t nr2 = rctx->nr2;
    ध्रुव uint32_t nr3 = rctx->nr3;

    ध्रुव size_t row_अष्टकानि = ne00 * rctx->type_आकार;

    ध्रुव uint32_t row_start = rctx->row_start + rctx->nrows_प्रत्येकम्_thread * ith;
    ध्रुव uint32_t row_अन्तः   = MIN(row_start + rctx->nrows_प्रत्येकम्_thread, rctx->row_start + rctx->योग_dst_पङ्क्तिकाः);

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    प्रति (uint32_t dst_row = row_start; dst_row < row_अन्तः; dst_row++) {
        // Decompose flat dst row index into (i1, i2, i3)
        ध्रुव uint32_t i1 = dst_row % ne1;
        ध्रुव uint32_t i2 = (dst_row / ne1) % ne2;
        ध्रुव uint32_t i3 = dst_row / (ne1 * ne2);

        // Map to source indices (tiling)
        ध्रुव uint32_t k1 = i1 % ne01;
        ध्रुव uint32_t k2 = i2 % ne02;
        ध्रुव uint32_t k3 = i3 % ne03;

        ध्रुव uint8_t * src_row = (ध्रुव uint8_t *) src->दत्त + k1 * nb01 + k2 * nb02 + k3 * nb03;
        uint8_t * dst_आधारः      = (uint8_t *) dst->दत्त + i1 * nb1 + i2 * nb2 + i3 * nb3;

        // Tile along dimension 0
        प्रति (uint32_t i0 = 0; i0 < nr0; i0++) {
            uint8_t * dst_ptr = dst_आधारः + i0 * ne00 * nb0;
            स्मृतिं_प्रतिलिख(dst_ptr, src_row, row_अष्टकानि);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    FARF(उच्चः_2, "repeat %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth, src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         row_start, row_अन्तः);
}

पूर्णाङ्क op_repeat(संरचना htp_ops_context * octx) {
    ध्रुव संरचना htp_tensor * src0 = octx->src[0];
    ध्रुव संरचना htp_tensor * dst  = octx->dst;

    // Validate that dst dims are multiples of src dims
    यदि (dst->ne[0] % src0->ne[0] != 0 ||
        dst->ne[1] % src0->ne[1] != 0 ||
        dst->ne[2] % src0->ne[2] != 0 ||
        dst->ne[3] % src0->ne[3] != 0) {
        FARF(ERROR, "repeat: dst dims must be multiples of src dims\n");
        प्रत्यागम HTP_स्थितिः_INVAL_PARAMS;
    }

    size_t type_आकार;
    विकल्प (src0->type) {
        पक्ष HTP_TYPE_F32: type_आकार = 4; भङ्ग;
        पक्ष HTP_TYPE_F16: type_आकार = 2; भङ्ग;
        स्वभाव:
            FARF(ERROR, "repeat: unsupported type %u\n", src0->type);
            प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
    }

    यदि (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
    }

    ध्रुव uint32_t योग_dst_पङ्क्तिकाः  = dst->ne[1] * dst->ne[2] * dst->ne[3];
    ध्रुव size_t dst_row_आकार = dst->ne[0] * type_आकार;

    uint32_t row_start = 0;
    uint32_t nrows     = योग_dst_पङ्क्तिकाः;

    यदि (octx->ctx->mdev.गणनम् > 1) {
        uint32_t पङ्क्तिकाः_प्रत्येकम्_chunk = 0;
        htp_tensor_mdev_पङ्क्तिकाः_प्रत्येकम्_chunk(dst, type_आकार, (uint32_t) dst_row_आकार, &पङ्क्तिकाः_प्रत्येकम्_chunk);
        ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(योग_dst_पङ्क्तिकाः, पङ्क्तिकाः_प्रत्येकम्_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.गणनम्, &octx->ctx->mdev.गणनम्_div);
        row_start = range.start;
        nrows     = range.गणनम्;
    }

    यदि (nrows == 0) {
        प्रत्यागम HTP_स्थितिः_OK;
    }

    ध्रुव uint32_t n_तन्तवः = octx->n_तन्तवः;

    संरचना htp_repeat_context rctx = {
        .octx             = octx,
        .nr0              = dst->ne[0] / src0->ne[0],
        .nr1              = dst->ne[1] / src0->ne[1],
        .nr2              = dst->ne[2] / src0->ne[2],
        .nr3              = dst->ne[3] / src0->ne[3],
        .nrows_प्रत्येकम्_thread = fastdiv(nrows + n_तन्तवः - 1, &octx->n_तन्तवः_div),
        .योग_dst_पङ्क्तिकाः   = nrows,
        .row_start        = row_start,
        .type_आकार        = type_आकार,
    };

    FARF(उच्चः_2, "repeat: (%ux%ux%ux%u) -> (%ux%ux%ux%u) nr=(%u,%u,%u,%u)\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         rctx.nr0, rctx.nr1, rctx.nr2, rctx.nr3);

    work_queue_सततम्(octx->ctx->work_queue, repeat_job_प्रत्येकम्_thread, &rctx, n_तन्तवः);

    प्रत्यागम HTP_स्थितिः_OK;
}
