#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#அந்தர்பாவ <HAP_farf.h>
#அந்தர்பாவ <string.h>

#அந்தர்பாவ "hex-common.h"
#அந்தர்பாவ "hex-profile.h"

#அந்தர்பாவ "hvx-copy.h"
#அந்தர்பாவ "hvx-utils.h"

#define GGML_COMMON_DECL_C
#அந்தர்பாவ "ggml-common.h"
#அந்தர்பாவ "htp-ctx.h"
#அந்தர்பாவ "htp-ops.h"
#அந்தர்பாவ "htp-tensor.h"

// ggml op_params layout for FILL:
//   op_params[0] (as float) - the scalar fill value

#define fill_preamble                          \
    மாறிலி கட்டமைப்பு htp_tensor * dst = octx->dst; \
                                               \
    மாறிலி uint32_t ne0 = dst->ne[0];           \
    மாறிலி uint32_t ne1 = dst->ne[1];           \
    மாறிலி uint32_t ne2 = dst->ne[2];           \
    மாறிலி uint32_t ne3 = dst->ne[3];           \
                                               \
    மாறிலி uint32_t nb1 = dst->nb[1];           \
    மாறிலி uint32_t nb2 = dst->nb[2];           \
    மாறிலி uint32_t nb3 = dst->nb[3];           \
                                               \
    மாறிலி uint32_t nr = ne1 * ne2 * ne3;

கட்டமைப்பு htp_fill_context {
    கட்டமைப்பு htp_ops_context * octx;
    uint32_t nrows_per_thread;
    uint32_t total_rows;  // ne1 * ne2 * ne3
    uint32_t row_start;
    bool     opt_path;
    HVX_Vector splat_vec;
    uint32_t   elem_size;
};

நிலையான வெறுமை fill_thread(unsigned முழு nth, unsigned முழு ith, வெறுமை * data) {
    மாறிலி கட்டமைப்பு htp_fill_context * fctx = (மாறிலி கட்டமைப்பு htp_fill_context *) data;
    கட்டமைப்பு htp_ops_context        * octx = fctx->octx;
    fill_preamble;

    // Parallelise over the flat row index spanning ne1*ne2*ne3
    மாறிலி uint32_t ir0 = fctx->row_start + fctx->nrows_per_thread * ith;
    மாறிலி uint32_t ir1 = MIN(ir0 + fctx->nrows_per_thread, fctx->row_start + fctx->total_rows);

    எனில் (ir0 >= ir1) {
        திரும்பு;
    }

    கட்டமைப்பு htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    எனில் (fctx->opt_path) {
        // Opt path: tensor is fully contiguous, treat as flat array
        மாறிலி uint32_t elem_start = ir0 * ne0;
        மாறிலி uint32_t elem_end = ir1 * ne0;
        uint8_t * dst_ptr = (uint8_t *) dst->data + elem_start * fctx->elem_size;
        hvx_splat_u(dst_ptr, fctx->splat_vec, elem_end - elem_start, fctx->elem_size);
    } இல்லையெனில் {
        // Non-contiguous path: must respect strides
        க்கு (uint32_t ir = ir0; ir < ir1; ++ir) {
            மாறிலி uint32_t i1 = ir % ne1;
            மாறிலி uint32_t i2 = (ir / ne1) % ne2;
            மாறிலி uint32_t i3 = ir / (ne1 * ne2);
            uint8_t * dst_ptr = (uint8_t *) dst->data + i1*nb1 + i2*nb2 + i3*nb3;
            hvx_splat_u(dst_ptr, fctx->splat_vec, ne0, fctx->elem_size);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir1);
    FARF(உயர்வு_2, "fill %u/%u: rows %u:%u\n", ith, nth, ir0, ir1);
}

முழு op_fill(கட்டமைப்பு htp_ops_context * octx) {
    fill_preamble;

    எனில் (dst->type != HTP_TYPE_F32 && dst->type != HTP_TYPE_F16) {
        திரும்பு HTP_STATUS_NO_SUPPORT;
    }

    எனில் (htp_tensor_is_extended(dst)) {
        திரும்பு HTP_STATUS_NO_SUPPORT;
    }

    uint32_t row_start = 0;
    uint32_t nrows     = nr;

    எனில் (octx->ctx->mdev.மொத்தம் > 1) {
        மாறிலி uint32_t row_size = nb1;
        மாறிலி uint32_t rows_per_chunk = (row_size > 0) ? (HEX_L2_வரி_SIZE / hex_gcd_u32(row_size, HEX_L2_வரி_SIZE)) : 1;
        மாறிலி கட்டமைப்பு htp_tensor_mdev_range range = htp_tensor_mdev_partition(nr, htp_tensor_mdev_data_aligned(dst) ? rows_per_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.மொத்தம், &octx->ctx->mdev.மொத்தம்_div);
        row_start = range.start;
        nrows     = range.மொத்தம்;
    }

    எனில் (nrows == 0) {
        திரும்பு HTP_STATUS_OK;
    }

    // nr = ne1*ne2*ne3 (flat row count across all outer dims); parallelise over it.
    மாறிலி uint32_t n_threads = octx->n_threads;

    // Optimize if fully contiguous: skip stride arithmetic, treat as flat array
    மாறிலி bool opt_path = (nb2 == nb1 * ne1) && (nb3 == nb2 * ne2);

    FARF(உயர்வு_2, "fill: (%ux%ux%ux%u) type=%u opt=%d\n",
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], dst->type, (முழு) opt_path);

    தசம val_f32 = 0.f;
    memcpy(&val_f32, &octx->op_params[0], அளவு(தசம));

    கட்டமைப்பு htp_fill_context fctx = {
        .octx             = octx,
        .nrows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div),
        .total_rows       = nrows,
        .row_start        = row_start,
        .opt_path         = opt_path,
    };

    தேர்வு (dst->type) {
    நிலை HTP_TYPE_F32:
        fctx.splat_vec = hvx_vec_splat_f32(val_f32);
        fctx.elem_size = அளவு(தசம);
        நிறுத்து;
    நிலை HTP_TYPE_F16:
        fctx.splat_vec = hvx_vec_splat_f16((_Float16) val_f32);
        fctx.elem_size = அளவு(_Float16);
        நிறுத்து;
    இயல்பு:
        திரும்பு HTP_STATUS_NO_SUPPORT;
    }

    work_queue_run(octx->ctx->work_queue, fill_thread, &fctx, n_threads);

    திரும்பு HTP_STATUS_OK;
}
