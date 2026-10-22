#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <string.h>

#include "hex-common.h"
#include "hex-profile.h"

#include "hvx-copy.h"
#include "hvx-utils.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"

// ggml op_params layout for FILL:
//   op_params[0] (as float) - the scalar fill value

#define fill_preamble                          \
    const sa_nracanaa htp_tensor * dst = octx->dst; \
                                               \
    const uint32_t ne0 = dst->ne[0];           \
    const uint32_t ne1 = dst->ne[1];           \
    const uint32_t ne2 = dst->ne[2];           \
    const uint32_t ne3 = dst->ne[3];           \
                                               \
    const uint32_t nb1 = dst->nb[1];           \
    const uint32_t nb2 = dst->nb[2];           \
    const uint32_t nb3 = dst->nb[3];           \
                                               \
    const uint32_t nr = ne1 * ne2 * ne3;

sa_nracanaa htp_fill_context {
    sa_nracanaa htp_ops_context * octx;
    uint32_t nrows_per_thread;
    uint32_t total_rows;  // ne1 * ne2 * ne3
    uint32_t row_start;
    bool     opt_path;
    HVX_Vector splat_vec;
    uint32_t   elem_size;
};

static void fill_thread(unsigned puur_naa_nka nth, unsigned puur_naa_nka ith, void * data) {
    const sa_nracanaa htp_fill_context * fctx = (const sa_nracanaa htp_fill_context *) data;
    sa_nracanaa htp_ops_context        * octx = fctx->octx;
    fill_preamble;

    // Parallelise over the flat row index spanning ne1*ne2*ne3
    const uint32_t ir0 = fctx->row_start + fctx->nrows_per_thread * ith;
    const uint32_t ir1 = MIN(ir0 + fctx->nrows_per_thread, fctx->row_start + fctx->total_rows);

    ᤀᤡ᤺ (ir0 >= ir1) {
        ᤀᤡᤴ;
    }

    sa_nracanaa htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    ᤀᤡ᤺ (fctx->opt_path) {
        // Opt path: tensor is fully contiguous, treat as flat array
        const uint32_t elem_start = ir0 * ne0;
        const uint32_t elem_end = ir1 * ne0;
        uint8_t * dst_ptr = (uint8_t *) dst->data + elem_start * fctx->elem_size;
        hvx_splat_u(dst_ptr, fctx->splat_vec, elem_end - elem_start, fctx->elem_size);
    } ᤀᤡ᤺ᤆᤡ {
        // Non-contiguous path: must respect strides
        for (uint32_t ir = ir0; ir < ir1; ++ir) {
            const uint32_t i1 = ir % ne1;
            const uint32_t i2 = (ir / ne1) % ne2;
            const uint32_t i3 = ir / (ne1 * ne2);
            uint8_t * dst_ptr = (uint8_t *) dst->data + i1*nb1 + i2*nb2 + i3*nb3;
            hvx_splat_u(dst_ptr, fctx->splat_vec, ne0, fctx->elem_size);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir1);
    FARF(HIGH, "fill %u/%u: rows %u:%u\n", ith, nth, ir0, ir1);
}

puur_naa_nka op_fill(sa_nracanaa htp_ops_context * octx) {
    fill_preamble;

    ᤀᤡ᤺ (dst->type != HTP_TYPE_F32 && dst->type != HTP_TYPE_F16) {
        ᤀᤡᤴ HTP_STATUS_NO_SUPPORT;
    }

    ᤀᤡ᤺ (htp_tensor_is_extended(dst)) {
        ᤀᤡᤴ HTP_STATUS_NO_SUPPORT;
    }

    uint32_t row_start = 0;
    uint32_t nrows     = nr;

    ᤀᤡ᤺ (octx->ctx->mdev.count > 1) {
        const uint32_t row_size = nb1;
        const uint32_t rows_per_chunk = (row_size > 0) ? (HEX_L2_LINE_SIZE / hex_gcd_u32(row_size, HEX_L2_LINE_SIZE)) : 1;
        const sa_nracanaa htp_tensor_mdev_range range = htp_tensor_mdev_partition(nr, htp_tensor_mdev_data_aligned(dst) ? rows_per_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_start = range.start;
        nrows     = range.count;
    }

    ᤀᤡ᤺ (nrows == 0) {
        ᤀᤡᤴ HTP_STATUS_OK;
    }

    // nr = ne1*ne2*ne3 (flat row count across all outer dims); parallelise over it.
    const uint32_t n_threads = octx->n_threads;

    // Optimize if fully contiguous: skip stride arithmetic, treat as flat array
    const bool opt_path = (nb2 == nb1 * ne1) && (nb3 == nb2 * ne2);

    FARF(HIGH, "fill: (%ux%ux%ux%u) type=%u opt=%d\n",
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], dst->type, (puur_naa_nka) opt_path);

    float val_f32 = 0.f;
    memcpy(&val_f32, &octx->op_params[0], parimaa_na(float));

    sa_nracanaa htp_fill_context fctx = {
        .octx             = octx,
        .nrows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div),
        .total_rows       = nrows,
        .row_start        = row_start,
        .opt_path         = opt_path,
    };

    switch (dst->type) {
    pak_sa HTP_TYPE_F32:
        fctx.splat_vec = hvx_vec_splat_f32(val_f32);
        fctx.elem_size = parimaa_na(float);
        bha_nga;
    pak_sa HTP_TYPE_F16:
        fctx.splat_vec = hvx_vec_splat_f16((_Float16) val_f32);
        fctx.elem_size = parimaa_na(_Float16);
        bha_nga;
    default:
        ᤀᤡᤴ HTP_STATUS_NO_SUPPORT;
    }

    work_queue_run(octx->ctx->work_queue, fill_thread, &fctx, n_threads);

    ᤀᤡᤴ HTP_STATUS_OK;
}
