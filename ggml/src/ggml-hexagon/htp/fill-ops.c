#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#અન્તર્ભાવ <HAP_farf.h>
#અન્તર્ભાવ <string.h>

#અન્તર્ભાવ "hex-common.h"
#અન્તર્ભાવ "hex-profile.h"

#અન્તર્ભાવ "hvx-copy.h"
#અન્તર્ભાવ "hvx-utils.h"

#define GGML_COMMON_DECL_C
#અન્તર્ભાવ "ggml-common.h"
#અન્તર્ભાવ "htp-ctx.h"
#અન્તર્ભાવ "htp-ops.h"
#અન્તર્ભાવ "htp-tensor.h"

// ggml op_params layout for FILL:
//   op_params[0] (as float) - the scalar fill value

#define fill_preamble                          \
    અચલ સંરચના htp_tensor * dst = octx->dst; \
                                               \
    અચલ uint32_t ne0 = dst->ne[0];           \
    અચલ uint32_t ne1 = dst->ne[1];           \
    અચલ uint32_t ne2 = dst->ne[2];           \
    અચલ uint32_t ne3 = dst->ne[3];           \
                                               \
    અચલ uint32_t nb1 = dst->nb[1];           \
    અચલ uint32_t nb2 = dst->nb[2];           \
    અચલ uint32_t nb3 = dst->nb[3];           \
                                               \
    અચલ uint32_t nr = ne1 * ne2 * ne3;

સંરચના htp_fill_context {
    સંરચના htp_ops_context * octx;
    uint32_t nrows_per_thread;
    uint32_t total_rows;  // ne1 * ne2 * ne3
    uint32_t row_start;
    bool     opt_path;
    HVX_Vector splat_vec;
    uint32_t   elem_size;
};

સ્થિર શૂન્ય fill_thread(unsigned પૂર્ણાંક nth, unsigned પૂર્ણાંક ith, શૂન્ય * data) {
    અચલ સંરચના htp_fill_context * fctx = (અચલ સંરચના htp_fill_context *) data;
    સંરચના htp_ops_context        * octx = fctx->octx;
    fill_preamble;

    // Parallelise over the flat row index spanning ne1*ne2*ne3
    અચલ uint32_t ir0 = fctx->row_start + fctx->nrows_per_thread * ith;
    અચલ uint32_t ir1 = MIN(ir0 + fctx->nrows_per_thread, fctx->row_start + fctx->total_rows);

    જો (ir0 >= ir1) {
        પાછા;
    }

    સંરચના htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    જો (fctx->opt_path) {
        // Opt path: tensor is fully contiguous, treat as flat array
        અચલ uint32_t elem_start = ir0 * ne0;
        અચલ uint32_t elem_end = ir1 * ne0;
        uint8_t * dst_ptr = (uint8_t *) dst->data + elem_start * fctx->elem_size;
        hvx_splat_u(dst_ptr, fctx->splat_vec, elem_end - elem_start, fctx->elem_size);
    } નહીંતર {
        // Non-contiguous path: must respect strides
        માટે (uint32_t ir = ir0; ir < ir1; ++ir) {
            અચલ uint32_t i1 = ir % ne1;
            અચલ uint32_t i2 = (ir / ne1) % ne2;
            અચલ uint32_t i3 = ir / (ne1 * ne2);
            uint8_t * dst_ptr = (uint8_t *) dst->data + i1*nb1 + i2*nb2 + i3*nb3;
            hvx_splat_u(dst_ptr, fctx->splat_vec, ne0, fctx->elem_size);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir1);
    FARF(ઉપલું_2, "fill %u/%u: rows %u:%u\n", ith, nth, ir0, ir1);
}

પૂર્ણાંક op_fill(સંરચના htp_ops_context * octx) {
    fill_preamble;

    જો (dst->type != HTP_TYPE_F32 && dst->type != HTP_TYPE_F16) {
        પાછા HTP_STATUS_NO_SUPPORT;
    }

    જો (htp_tensor_is_extended(dst)) {
        પાછા HTP_STATUS_NO_SUPPORT;
    }

    uint32_t row_start = 0;
    uint32_t nrows     = nr;

    જો (octx->ctx->mdev.ગણતરી > 1) {
        અચલ uint32_t row_size = nb1;
        અચલ uint32_t rows_per_chunk = (row_size > 0) ? (HEX_L2_લીટી_SIZE / hex_gcd_u32(row_size, HEX_L2_લીટી_SIZE)) : 1;
        અચલ સંરચના htp_tensor_mdev_range range = htp_tensor_mdev_partition(nr, htp_tensor_mdev_data_aligned(dst) ? rows_per_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ગણતરી, &octx->ctx->mdev.ગણતરી_div);
        row_start = range.start;
        nrows     = range.ગણતરી;
    }

    જો (nrows == 0) {
        પાછા HTP_STATUS_OK;
    }

    // nr = ne1*ne2*ne3 (flat row count across all outer dims); parallelise over it.
    અચલ uint32_t n_threads = octx->n_threads;

    // Optimize if fully contiguous: skip stride arithmetic, treat as flat array
    અચલ bool opt_path = (nb2 == nb1 * ne1) && (nb3 == nb2 * ne2);

    FARF(ઉપલું_2, "fill: (%ux%ux%ux%u) type=%u opt=%d\n",
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], dst->type, (પૂર્ણાંક) opt_path);

    દશાંશ val_f32 = 0.f;
    memcpy(&val_f32, &octx->op_params[0], કદ(દશાંશ));

    સંરચના htp_fill_context fctx = {
        .octx             = octx,
        .nrows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div),
        .total_rows       = nrows,
        .row_start        = row_start,
        .opt_path         = opt_path,
    };

    પસંદગી (dst->type) {
    સ્થિતિ HTP_TYPE_F32:
        fctx.splat_vec = hvx_vec_splat_f32(val_f32);
        fctx.elem_size = કદ(દશાંશ);
        તોડો;
    સ્થિતિ HTP_TYPE_F16:
        fctx.splat_vec = hvx_vec_splat_f16((_Float16) val_f32);
        fctx.elem_size = કદ(_Float16);
        તોડો;
    મૂળભૂત:
        પાછા HTP_STATUS_NO_SUPPORT;
    }

    work_queue_run(octx->ctx->work_queue, fill_thread, &fctx, n_threads);

    પાછા HTP_STATUS_OK;
}
