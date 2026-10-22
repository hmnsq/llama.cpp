#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#അന്തര്ഭാവ <HAP_farf.h>
#അന്തര്ഭാവ <string.h>

#അന്തര്ഭാവ "hex-common.h"
#അന്തര്ഭാവ "hex-profile.h"

#അന്തര്ഭാവ "hvx-copy.h"
#അന്തര്ഭാവ "hvx-utils.h"

#define GGML_COMMON_DECL_C
#അന്തര്ഭാവ "ggml-common.h"
#അന്തര്ഭാവ "htp-ctx.h"
#അന്തര്ഭാവ "htp-ops.h"
#അന്തര്ഭാവ "htp-tensor.h"

// ggml op_params layout for FILL:
//   op_params[0] (as float) - the scalar fill value

#define fill_preamble                          \
    സ്ഥിരം ഘടന htp_tensor * dst = octx->dst; \
                                               \
    സ്ഥിരം uint32_t ne0 = dst->ne[0];           \
    സ്ഥിരം uint32_t ne1 = dst->ne[1];           \
    സ്ഥിരം uint32_t ne2 = dst->ne[2];           \
    സ്ഥിരം uint32_t ne3 = dst->ne[3];           \
                                               \
    സ്ഥിരം uint32_t nb1 = dst->nb[1];           \
    സ്ഥിരം uint32_t nb2 = dst->nb[2];           \
    സ്ഥിരം uint32_t nb3 = dst->nb[3];           \
                                               \
    സ്ഥിരം uint32_t nr = ne1 * ne2 * ne3;

ഘടന htp_fill_context {
    ഘടന htp_ops_context * octx;
    uint32_t nrows_per_thread;
    uint32_t total_rows;  // ne1 * ne2 * ne3
    uint32_t row_start;
    bool     opt_path;
    HVX_Vector splat_vec;
    uint32_t   elem_size;
};

സ്ഥിതം ശൂന്യം fill_thread(unsigned പൂർണ്ണസംഖ്യ nth, unsigned പൂർണ്ണസംഖ്യ ith, ശൂന്യം * data) {
    സ്ഥിരം ഘടന htp_fill_context * fctx = (സ്ഥിരം ഘടന htp_fill_context *) data;
    ഘടന htp_ops_context        * octx = fctx->octx;
    fill_preamble;

    // Parallelise over the flat row index spanning ne1*ne2*ne3
    സ്ഥിരം uint32_t ir0 = fctx->row_start + fctx->nrows_per_thread * ith;
    സ്ഥിരം uint32_t ir1 = MIN(ir0 + fctx->nrows_per_thread, fctx->row_start + fctx->total_rows);

    എങ്കിൽ (ir0 >= ir1) {
        തിരികെ;
    }

    ഘടന htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    എങ്കിൽ (fctx->opt_path) {
        // Opt path: tensor is fully contiguous, treat as flat array
        സ്ഥിരം uint32_t elem_start = ir0 * ne0;
        സ്ഥിരം uint32_t elem_end = ir1 * ne0;
        uint8_t * dst_ptr = (uint8_t *) dst->data + elem_start * fctx->elem_size;
        hvx_splat_u(dst_ptr, fctx->splat_vec, elem_end - elem_start, fctx->elem_size);
    } അല്ലെങ്കിൽ {
        // Non-contiguous path: must respect strides
        വേണ്ടി (uint32_t ir = ir0; ir < ir1; ++ir) {
            സ്ഥിരം uint32_t i1 = ir % ne1;
            സ്ഥിരം uint32_t i2 = (ir / ne1) % ne2;
            സ്ഥിരം uint32_t i3 = ir / (ne1 * ne2);
            uint8_t * dst_ptr = (uint8_t *) dst->data + i1*nb1 + i2*nb2 + i3*nb3;
            hvx_splat_u(dst_ptr, fctx->splat_vec, ne0, fctx->elem_size);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir1);
    FARF(മുകളിലെ_2, "fill %u/%u: rows %u:%u\n", ith, nth, ir0, ir1);
}

പൂർണ്ണസംഖ്യ op_fill(ഘടന htp_ops_context * octx) {
    fill_preamble;

    എങ്കിൽ (dst->type != HTP_TYPE_F32 && dst->type != HTP_TYPE_F16) {
        തിരികെ HTP_STATUS_NO_SUPPORT;
    }

    എങ്കിൽ (htp_tensor_is_extended(dst)) {
        തിരികെ HTP_STATUS_NO_SUPPORT;
    }

    uint32_t row_start = 0;
    uint32_t nrows     = nr;

    എങ്കിൽ (octx->ctx->mdev.എണ്ണം > 1) {
        സ്ഥിരം uint32_t row_size = nb1;
        സ്ഥിരം uint32_t rows_per_chunk = (row_size > 0) ? (HEX_L2_വരി_SIZE / hex_gcd_u32(row_size, HEX_L2_വരി_SIZE)) : 1;
        സ്ഥിരം ഘടന htp_tensor_mdev_range range = htp_tensor_mdev_partition(nr, htp_tensor_mdev_data_aligned(dst) ? rows_per_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.എണ്ണം, &octx->ctx->mdev.എണ്ണം_div);
        row_start = range.start;
        nrows     = range.എണ്ണം;
    }

    എങ്കിൽ (nrows == 0) {
        തിരികെ HTP_STATUS_OK;
    }

    // nr = ne1*ne2*ne3 (flat row count across all outer dims); parallelise over it.
    സ്ഥിരം uint32_t n_threads = octx->n_threads;

    // Optimize if fully contiguous: skip stride arithmetic, treat as flat array
    സ്ഥിരം bool opt_path = (nb2 == nb1 * ne1) && (nb3 == nb2 * ne2);

    FARF(മുകളിലെ_2, "fill: (%ux%ux%ux%u) type=%u opt=%d\n",
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], dst->type, (പൂർണ്ണസംഖ്യ) opt_path);

    ദശാംശം val_f32 = 0.f;
    memcpy(&val_f32, &octx->op_params[0], വലുപ്പം(ദശാംശം));

    ഘടന htp_fill_context fctx = {
        .octx             = octx,
        .nrows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div),
        .total_rows       = nrows,
        .row_start        = row_start,
        .opt_path         = opt_path,
    };

    തിരഞ്ഞെടുപ്പ് (dst->type) {
    അവസ്ഥ HTP_TYPE_F32:
        fctx.splat_vec = hvx_vec_splat_f32(val_f32);
        fctx.elem_size = വലുപ്പം(ദശാംശം);
        നിർത്തുക;
    അവസ്ഥ HTP_TYPE_F16:
        fctx.splat_vec = hvx_vec_splat_f16((_Float16) val_f32);
        fctx.elem_size = വലുപ്പം(_Float16);
        നിർത്തുക;
    സ്ഥിരസ്ഥിതി:
        തിരികെ HTP_STATUS_NO_SUPPORT;
    }

    work_queue_run(octx->ctx->work_queue, fill_thread, &fctx, n_threads);

    തിരികെ HTP_STATUS_OK;
}
