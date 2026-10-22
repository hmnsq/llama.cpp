#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#ଅନ୍ତର୍ଭାଵ <HAP_farf.h>
#ଅନ୍ତର୍ଭାଵ <string.h>

#ଅନ୍ତର୍ଭାଵ "hex-common.h"
#ଅନ୍ତର୍ଭାଵ "hex-profile.h"

#ଅନ୍ତର୍ଭାଵ "hvx-copy.h"
#ଅନ୍ତର୍ଭାଵ "hvx-utils.h"

#define GGML_COMMON_DECL_C
#ଅନ୍ତର୍ଭାଵ "ggml-common.h"
#ଅନ୍ତର୍ଭାଵ "htp-ctx.h"
#ଅନ୍ତର୍ଭାଵ "htp-ops.h"
#ଅନ୍ତର୍ଭାଵ "htp-tensor.h"

// ggml op_params layout for FILL:
//   op_params[0] (as float) - the scalar fill value

#define fill_preamble                          \
    ଅଚଳ ସଂରଚନା htp_tensor * dst = octx->dst; \
                                               \
    ଅଚଳ uint32_t ne0 = dst->ne[0];           \
    ଅଚଳ uint32_t ne1 = dst->ne[1];           \
    ଅଚଳ uint32_t ne2 = dst->ne[2];           \
    ଅଚଳ uint32_t ne3 = dst->ne[3];           \
                                               \
    ଅଚଳ uint32_t nb1 = dst->nb[1];           \
    ଅଚଳ uint32_t nb2 = dst->nb[2];           \
    ଅଚଳ uint32_t nb3 = dst->nb[3];           \
                                               \
    ଅଚଳ uint32_t nr = ne1 * ne2 * ne3;

ସଂରଚନା htp_fill_context {
    ସଂରଚନା htp_ops_context * octx;
    uint32_t nrows_per_thread;
    uint32_t total_rows;  // ne1 * ne2 * ne3
    uint32_t row_start;
    bool     opt_path;
    HVX_Vector splat_vec;
    uint32_t   elem_size;
};

ସ୍ଥିର ଶୂନ୍ୟ fill_thread(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith, ଶୂନ୍ୟ * data) {
    ଅଚଳ ସଂରଚନା htp_fill_context * fctx = (ଅଚଳ ସଂରଚନା htp_fill_context *) data;
    ସଂରଚନା htp_ops_context        * octx = fctx->octx;
    fill_preamble;

    // Parallelise over the flat row index spanning ne1*ne2*ne3
    ଅଚଳ uint32_t ir0 = fctx->row_start + fctx->nrows_per_thread * ith;
    ଅଚଳ uint32_t ir1 = MIN(ir0 + fctx->nrows_per_thread, fctx->row_start + fctx->total_rows);

    ଯଦି (ir0 >= ir1) {
        ଫେରନ୍ତୁ;
    }

    ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    ଯଦି (fctx->opt_path) {
        // Opt path: tensor is fully contiguous, treat as flat array
        ଅଚଳ uint32_t elem_start = ir0 * ne0;
        ଅଚଳ uint32_t elem_end = ir1 * ne0;
        uint8_t * dst_ptr = (uint8_t *) dst->data + elem_start * fctx->elem_size;
        hvx_splat_u(dst_ptr, fctx->splat_vec, elem_end - elem_start, fctx->elem_size);
    } ଅନ୍ୟଥା {
        // Non-contiguous path: must respect strides
        ପାଇଁ (uint32_t ir = ir0; ir < ir1; ++ir) {
            ଅଚଳ uint32_t i1 = ir % ne1;
            ଅଚଳ uint32_t i2 = (ir / ne1) % ne2;
            ଅଚଳ uint32_t i3 = ir / (ne1 * ne2);
            uint8_t * dst_ptr = (uint8_t *) dst->data + i1*nb1 + i2*nb2 + i3*nb3;
            hvx_splat_u(dst_ptr, fctx->splat_vec, ne0, fctx->elem_size);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir1);
    FARF(ଉପର_2, "fill %u/%u: rows %u:%u\n", ith, nth, ir0, ir1);
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_fill(ସଂରଚନା htp_ops_context * octx) {
    fill_preamble;

    ଯଦି (dst->type != HTP_TYPE_F32 && dst->type != HTP_TYPE_F16) {
        ଫେରନ୍ତୁ HTP_STATUS_NO_SUPPORT;
    }

    ଯଦି (htp_tensor_is_extended(dst)) {
        ଫେରନ୍ତୁ HTP_STATUS_NO_SUPPORT;
    }

    uint32_t row_start = 0;
    uint32_t nrows     = nr;

    ଯଦି (octx->ctx->mdev.ଗଣନା_2 > 1) {
        ଅଚଳ uint32_t row_size = nb1;
        ଅଚଳ uint32_t rows_per_chunk = (row_size > 0) ? (HEX_L2_ଧାଡ଼ି_SIZE / hex_gcd_u32(row_size, HEX_L2_ଧାଡ଼ି_SIZE)) : 1;
        ଅଚଳ ସଂରଚନା htp_tensor_mdev_range range = htp_tensor_mdev_partition(nr, htp_tensor_mdev_data_aligned(dst) ? rows_per_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ଗଣନା_2, &octx->ctx->mdev.ଗଣନା_div);
        row_start = range.start;
        nrows     = range.ଗଣନା_2;
    }

    ଯଦି (nrows == 0) {
        ଫେରନ୍ତୁ HTP_STATUS_OK;
    }

    // nr = ne1*ne2*ne3 (flat row count across all outer dims); parallelise over it.
    ଅଚଳ uint32_t n_threads = octx->n_threads;

    // Optimize if fully contiguous: skip stride arithmetic, treat as flat array
    ଅଚଳ bool opt_path = (nb2 == nb1 * ne1) && (nb3 == nb2 * ne2);

    FARF(ଉପର_2, "fill: (%ux%ux%ux%u) type=%u opt=%d\n",
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], dst->type, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) opt_path);

    ଦଶମିକ val_f32 = 0.f;
    memcpy(&val_f32, &octx->op_params[0], ଆକାର(ଦଶମିକ));

    ସଂରଚନା htp_fill_context fctx = {
        .octx             = octx,
        .nrows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div),
        .total_rows       = nrows,
        .row_start        = row_start,
        .opt_path         = opt_path,
    };

    ବାଛ (dst->type) {
    ସ୍ଥିତି HTP_TYPE_F32:
        fctx.splat_vec = hvx_vec_splat_f32(val_f32);
        fctx.elem_size = ଆକାର(ଦଶମିକ);
        ଭାଙ୍ଗ;
    ସ୍ଥିତି HTP_TYPE_F16:
        fctx.splat_vec = hvx_vec_splat_f16((_Float16) val_f32);
        fctx.elem_size = ଆକାର(_Float16);
        ଭାଙ୍ଗ;
    ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
        ଫେରନ୍ତୁ HTP_STATUS_NO_SUPPORT;
    }

    work_queue_run(octx->ctx->work_queue, fill_thread, &fctx, n_threads);

    ଫେରନ୍ତୁ HTP_STATUS_OK;
}
