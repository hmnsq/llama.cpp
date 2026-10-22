#आशय clang diagnostic ignored "-Wunused-variable"
#आशय clang diagnostic ignored "-Wunused-function"
#आशय clang diagnostic ignored "-Wunused-but-set-variable"

#समावेश <HAP_farf.h>
#समावेश <string.h>

#समावेश "hex-common.h"
#समावेश "hex-profile.h"

#समावेश "hvx-copy.h"
#समावेश "hvx-utils.h"

#घोषणा GGML_COMMON_DECL_C
#समावेश "ggml-common.h"
#समावेश "htp-ctx.h"
#समावेश "htp-ops.h"
#समावेश "htp-tensor.h"

// ggml op_params layout for FILL:
//   op_params[0] (as float) - the scalar fill value

#घोषणा fill_preamble                          \
    स्थिर काष्ठा htp_tensor * dst = octx->dst; \
                                               \
    स्थिर uint32_t ne0 = dst->ne[0];           \
    स्थिर uint32_t ne1 = dst->ne[1];           \
    स्थिर uint32_t ne2 = dst->ne[2];           \
    स्थिर uint32_t ne3 = dst->ne[3];           \
                                               \
    स्थिर uint32_t nb1 = dst->nb[1];           \
    स्थिर uint32_t nb2 = dst->nb[2];           \
    स्थिर uint32_t nb3 = dst->nb[3];           \
                                               \
    स्थिर uint32_t nr = ne1 * ne2 * ne3;

काष्ठा htp_fill_context {
    काष्ठा htp_ops_context * octx;
    uint32_t nrows_per_thread;
    uint32_t योग_rows;  // ne1 * ne2 * ne3
    uint32_t row_start;
    bool     opt_path;
    HVX_Vector splat_vec;
    uint32_t   elem_आकार;
};

अटल व्योम fill_thread(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    स्थिर काष्ठा htp_fill_context * fctx = (स्थिर काष्ठा htp_fill_context *) तथ्य;
    काष्ठा htp_ops_context        * octx = fctx->octx;
    fill_preamble;

    // Parallelise over the flat row index spanning ne1*ne2*ne3
    स्थिर uint32_t ir0 = fctx->row_start + fctx->nrows_per_thread * ith;
    स्थिर uint32_t ir1 = MIN(ir0 + fctx->nrows_per_thread, fctx->row_start + fctx->योग_rows);

    अगर (ir0 >= ir1) {
        वापस;
    }

    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    अगर (fctx->opt_path) {
        // Opt path: tensor is fully contiguous, treat as flat array
        स्थिर uint32_t elem_start = ir0 * ne0;
        स्थिर uint32_t elem_end = ir1 * ne0;
        uint8_t * dst_ptr = (uint8_t *) dst->तथ्य + elem_start * fctx->elem_आकार;
        hvx_splat_u(dst_ptr, fctx->splat_vec, elem_end - elem_start, fctx->elem_आकार);
    } अन्यथा {
        // Non-contiguous path: must respect strides
        क्रम (uint32_t ir = ir0; ir < ir1; ++ir) {
            स्थिर uint32_t i1 = ir % ne1;
            स्थिर uint32_t i2 = (ir / ne1) % ne2;
            स्थिर uint32_t i3 = ir / (ne1 * ne2);
            uint8_t * dst_ptr = (uint8_t *) dst->तथ्य + i1*nb1 + i2*nb2 + i3*nb3;
            hvx_splat_u(dst_ptr, fctx->splat_vec, ne0, fctx->elem_आकार);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir1);
    FARF(ऊपरी_2, "fill %u/%u: rows %u:%u\n", ith, nth, ir0, ir1);
}

पूर्णांक op_fill(काष्ठा htp_ops_context * octx) {
    fill_preamble;

    अगर (dst->type != HTP_TYPE_F32 && dst->type != HTP_TYPE_F16) {
        वापस HTP_STATUS_NO_SUPPORT;
    }

    अगर (htp_tensor_is_extended(dst)) {
        वापस HTP_STATUS_NO_SUPPORT;
    }

    uint32_t row_start = 0;
    uint32_t nrows     = nr;

    अगर (octx->ctx->mdev.गिनती > 1) {
        स्थिर uint32_t row_आकार = nb1;
        स्थिर uint32_t rows_per_chunk = (row_आकार > 0) ? (HEX_L2_पंक्ति_आकार / hex_gcd_u32(row_आकार, HEX_L2_पंक्ति_आकार)) : 1;
        स्थिर काष्ठा htp_tensor_mdev_range range = htp_tensor_mdev_partition(nr, htp_tensor_mdev_तथ्य_aligned(dst) ? rows_per_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
        row_start = range.start;
        nrows     = range.गिनती;
    }

    अगर (nrows == 0) {
        वापस HTP_STATUS_OK;
    }

    // nr = ne1*ne2*ne3 (flat row count across all outer dims); parallelise over it.
    स्थिर uint32_t n_threads = octx->n_threads;

    // Optimize if fully contiguous: skip stride arithmetic, treat as flat array
    स्थिर bool opt_path = (nb2 == nb1 * ne1) && (nb3 == nb2 * ne2);

    FARF(ऊपरी_2, "fill: (%ux%ux%ux%u) type=%u opt=%d\n",
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], dst->type, (पूर्णांक) opt_path);

    भग्न val_f32 = 0.f;
    स_नकल(&val_f32, &octx->op_params[0], माप(भग्न));

    काष्ठा htp_fill_context fctx = {
        .octx             = octx,
        .nrows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div),
        .योग_rows       = nrows,
        .row_start        = row_start,
        .opt_path         = opt_path,
    };

    चयन (dst->type) {
    हाल HTP_TYPE_F32:
        fctx.splat_vec = hvx_vec_splat_f32(val_f32);
        fctx.elem_आकार = माप(भग्न);
        अवरोध;
    हाल HTP_TYPE_F16:
        fctx.splat_vec = hvx_vec_splat_f16((_Float16) val_f32);
        fctx.elem_आकार = माप(_Float16);
        अवरोध;
    शेष:
        वापस HTP_STATUS_NO_SUPPORT;
    }

    work_queue_run(octx->ctx->work_queue, fill_thread, &fctx, n_threads);

    वापस HTP_STATUS_OK;
}
