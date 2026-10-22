#ضمن "concat-ops.h"
#ضمن "dma-queue.h"
#ضمن "hex-common.h"
#ضمن "dma-copy.h"
#ضمن "hex-fastdiv.h"
#ضمن "hex-profile.h"
#ضمن "hexagon_protos.h"
#ضمن "hexagon_types.h"
#ضمن "htp-ctx.h"
#ضمن "htp-fence.h"
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"
#ضمن "htp-vtcm.h"
#ضمن "hvx-utils.h"
#ضمن "hvx_hexagon_protos.h"
#ضمن "gather-rows.h"

#ضمن <string.h>

بنية htp_concat_context {
    بنية htp_ops_context * octx;
    uint8_t * spad0_بنياد;
    uint8_t * spad1_بنياد;
    uint32_t  spad0_سائيز_في_thread;
    uint32_t  spad1_سائيز_في_thread;
    uint32_t  صف_شروعات;
    uint32_t  nrows;
    uint32_t  nrows_في_thread;
    uint32_t  nplanes;
    بنية fastdiv_قدرون div_ne2;
};

ساكن مضمن dma_addr_t concat_plane_addr(ثابت بنية htp_tensor * t, uint32_t p, ثابت بنية fastdiv_قدرون * div_ne2, uint32_t ne2) {
    ثابت uint32_t i3 = fastdiv(p, div_ne2);
    ثابت uint32_t i2 = p - i3 * ne2;
    أرجع t->معلومات + i2 * t->nb[2] + i3 * t->nb[3];
}

ساكن فراغ concat_2d_f32_transposed(بنانشان صحيح nth, بنانشان صحيح ith, فراغ * معلومات) {
    بنية htp_concat_context * cctx = (بنية htp_concat_context *) معلومات;
    بنية htp_ops_context * octx = cctx->octx;

    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * src1 = octx->src[1];
    ثابت بنية htp_tensor * dst  = octx->dst;

    ثابت uint32_t src0_ne0 = src0->ne[0];
    ثابت uint32_t src1_ne0 = src1->ne[0];

    ثابت uint32_t صف_پڇاڙي = cctx->صف_شروعات + cctx->nrows;
    ثابت uint32_t شروعات_i = cctx->صف_شروعات + ith * cctx->nrows_في_thread;
    ثابت uint32_t پڇاڙي_i   = (شروعات_i + cctx->nrows_في_thread < صف_پڇاڙي) ? (شروعات_i + cctx->nrows_في_thread) : صف_پڇاڙي;
    إذا (شروعات_i >= پڇاڙي_i) أرجع;

    dma_قطار * dma_q = octx->ctx->dma[ith];

    uint8_t * spad0_بنياد = cctx->spad0_بنياد + ith * cctx->spad0_سائيز_في_thread;
    uint8_t * spad1_بنياد = cctx->spad1_بنياد + ith * cctx->spad1_سائيز_في_thread;

    ثابت uint32_t بلاڪ_i = 32;
    ثابت uint32_t spad1_stride = بلاڪ_i * حجم(عائم);

    ثابت HVX_ويڪٽر offsets = hvx_vec_gather_offsets_w(spad1_stride);
    ثابت uint32_t src1_ne0_padded = hex_ڦيرو_مٿي(src1_ne0, 32);
    ثابت uint32_t src0_صف_بائيٽون  = src0_ne0 * حجم(عائم);
    ثابت uint32_t src0_صف_padded = hex_ڦيرو_مٿي(src0_صف_بائيٽون, VLEN);
    ثابت uint32_t src0_pre        = src0_صف_padded - src0_صف_بائيٽون;
    ثابت uint32_t spad0_صف_بائيٽون = src0_صف_padded + src1_ne0_padded * حجم(عائم);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت بنية fastdiv_قدرون * div_ne2 = &cctx->div_ne2;
    ثابت uint32_t ne2 = dst->ne[2];

    uint32_t p = 0;
    uint32_t i = شروعات_i;

    ثابت dma_addr_t src1_addr = concat_plane_addr(src1, p, div_ne2, ne2) + i * src1->nb[1];
    dma_قطار_ڌڪو(dma_q, dma_make_معلومات(spad1_بنياد, src1_addr), spad1_stride, src1->nb[0], MIN(پڇاڙي_i - i, بلاڪ_i) * حجم(عائم), src1_ne0);

    ثابت dma_addr_t src0_addr = concat_plane_addr(src0, p, div_ne2, ne2) + i * src0->nb[1];
    dma_قطار_ڌڪو(dma_q, dma_make_معلومات(spad0_بنياد + src0_pre, src0_addr), spad0_صف_بائيٽون, src0->nb[1], src0_صف_بائيٽون, MIN(پڇاڙي_i - i, بلاڪ_i));

    dma_قطار_ڪڍو(dma_q); // src1

    طالما (p < cctx->nplanes) {
        ثابت uint32_t current_بلاڪ_i = MIN(پڇاڙي_i - i, بلاڪ_i);

        htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        لكل (uint32_t j = 0; j < src1_ne0; j += 32) {
            ثابت uint8_t * src_ptr = spad1_بنياد + j * spad1_stride;
            uint8_t * dst_ptr = spad0_بنياد + src0_صف_padded + j * حجم(عائم);
            hvx_transpose_32x32_w_gather(dst_ptr, spad0_صف_بائيٽون, src_ptr, spad1_stride, offsets, current_بلاڪ_i, MIN(src1_ne0 - j, 32));
        }
        hvx_gather_sync(spad0_بنياد + src0_صف_padded);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        uint32_t np = p;
        uint32_t ni = i + بلاڪ_i;
        إذا (ni >= پڇاڙي_i) {
            ni = شروعات_i;
            np++;
        }
        ثابت bool has_ايندڙ = np < cctx->nplanes;
        ثابت uint32_t ايندڙ_بلاڪ_i = MIN(پڇاڙي_i - ni, بلاڪ_i);

        // spad1 is free after the gather sync, prefetch next src1 ahead of the dst write
        إذا (has_ايندڙ) {
            ثابت dma_addr_t nsrc1_addr = concat_plane_addr(src1, np, div_ne2, ne2) + ni * src1->nb[1];
            dma_قطار_ڌڪو(dma_q, dma_make_معلومات(spad1_بنياد, nsrc1_addr), spad1_stride, src1->nb[0], ايندڙ_بلاڪ_i * حجم(عائم), src1_ne0);
        }

        dma_قطار_ڪڍو(dma_q); // src0

        ثابت dma_addr_t dst_addr = concat_plane_addr(dst, p, div_ne2, ne2) + i * dst->nb[1];
        dma_قطار_ڌڪو(dma_q, dma_make_معلومات(dst_addr, spad0_بنياد + src0_pre), dst->nb[1], spad0_صف_بائيٽون, (src0_ne0 + src1_ne0) * حجم(عائم), current_بلاڪ_i);

        إذا (has_ايندڙ) {
            dma_قطار_ڪڍو(dma_q); // next src1
        }
        dma_قطار_ڪڍو(dma_q); // dst

        // spad0 is free after the dst write
        إذا (has_ايندڙ) {
            ثابت dma_addr_t nsrc0_addr = concat_plane_addr(src0, np, div_ne2, ne2) + ni * src0->nb[1];
            dma_قطار_ڌڪو(dma_q, dma_make_معلومات(spad0_بنياد + src0_pre, nsrc0_addr), spad0_صف_بائيٽون, src0->nb[1], src0_صف_بائيٽون, ايندڙ_بلاڪ_i);
        }

        p = np;
        i = ni;
    }
    dma_قطار_flush(dma_q);
}

ساكن فراغ concat_2d_f16_transposed(بنانشان صحيح nth, بنانشان صحيح ith, فراغ * معلومات) {
    بنية htp_concat_context * cctx = (بنية htp_concat_context *) معلومات;
    بنية htp_ops_context * octx = cctx->octx;

    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * src1 = octx->src[1];
    ثابت بنية htp_tensor * dst  = octx->dst;

    ثابت uint32_t src0_ne0 = src0->ne[0];
    ثابت uint32_t src1_ne0 = src1->ne[0];

    ثابت uint32_t صف_پڇاڙي = cctx->صف_شروعات + cctx->nrows;
    ثابت uint32_t شروعات_i = cctx->صف_شروعات + ith * cctx->nrows_في_thread;
    ثابت uint32_t پڇاڙي_i   = (شروعات_i + cctx->nrows_في_thread < صف_پڇاڙي) ? (شروعات_i + cctx->nrows_في_thread) : صف_پڇاڙي;
    إذا (شروعات_i >= پڇاڙي_i) أرجع;

    dma_قطار * dma_q = octx->ctx->dma[ith];

    uint8_t * spad0_بنياد = cctx->spad0_بنياد + ith * cctx->spad0_سائيز_في_thread;
    uint8_t * spad1_بنياد = cctx->spad1_بنياد + ith * cctx->spad1_سائيز_في_thread;

    ثابت uint32_t بلاڪ_i = 64;
    ثابت uint32_t spad1_stride = بلاڪ_i * حجم(__fp16);

    ثابت HVX_ويڪٽر offsets = hvx_vec_gather_offsets_h(spad1_stride);
    ثابت uint32_t src1_ne0_padded = hex_ڦيرو_مٿي(src1_ne0, 64);
    ثابت uint32_t src0_صف_بائيٽون  = src0_ne0 * حجم(__fp16);
    ثابت uint32_t src0_صف_padded = hex_ڦيرو_مٿي(src0_صف_بائيٽون, VLEN);
    ثابت uint32_t src0_pre        = src0_صف_padded - src0_صف_بائيٽون;
    ثابت uint32_t spad0_صف_بائيٽون = src0_صف_padded + src1_ne0_padded * حجم(__fp16);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت بنية fastdiv_قدرون * div_ne2 = &cctx->div_ne2;
    ثابت uint32_t ne2 = dst->ne[2];

    uint32_t p = 0;
    uint32_t i = شروعات_i;

    ثابت dma_addr_t src1_addr = concat_plane_addr(src1, p, div_ne2, ne2) + i * src1->nb[1];
    dma_قطار_ڌڪو(dma_q, dma_make_معلومات(spad1_بنياد, src1_addr), spad1_stride, src1->nb[0], MIN(پڇاڙي_i - i, بلاڪ_i) * حجم(__fp16), src1_ne0);

    ثابت dma_addr_t src0_addr = concat_plane_addr(src0, p, div_ne2, ne2) + i * src0->nb[1];
    dma_قطار_ڌڪو(dma_q, dma_make_معلومات(spad0_بنياد + src0_pre, src0_addr), spad0_صف_بائيٽون, src0->nb[1], src0_صف_بائيٽون, MIN(پڇاڙي_i - i, بلاڪ_i));

    dma_قطار_ڪڍو(dma_q); // src1

    طالما (p < cctx->nplanes) {
        ثابت uint32_t current_بلاڪ_i = MIN(پڇاڙي_i - i, بلاڪ_i);

        htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        لكل (uint32_t j = 0; j < src1_ne0; j += 64) {
            ثابت uint8_t * src_ptr = spad1_بنياد + j * spad1_stride;
            uint8_t * dst_ptr = spad0_بنياد + src0_صف_padded + j * حجم(__fp16);
            hvx_transpose_64x64_h_gather(dst_ptr, spad0_صف_بائيٽون, src_ptr, spad1_stride, offsets, current_بلاڪ_i, MIN(src1_ne0 - j, 64));
        }
        hvx_gather_sync(spad0_بنياد + src0_صف_padded);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        uint32_t np = p;
        uint32_t ni = i + بلاڪ_i;
        إذا (ni >= پڇاڙي_i) {
            ni = شروعات_i;
            np++;
        }
        ثابت bool has_ايندڙ = np < cctx->nplanes;
        ثابت uint32_t ايندڙ_بلاڪ_i = MIN(پڇاڙي_i - ni, بلاڪ_i);

        // spad1 is free after the gather sync, prefetch next src1 ahead of the dst write
        إذا (has_ايندڙ) {
            ثابت dma_addr_t nsrc1_addr = concat_plane_addr(src1, np, div_ne2, ne2) + ni * src1->nb[1];
            dma_قطار_ڌڪو(dma_q, dma_make_معلومات(spad1_بنياد, nsrc1_addr), spad1_stride, src1->nb[0], ايندڙ_بلاڪ_i * حجم(__fp16), src1_ne0);
        }

        dma_قطار_ڪڍو(dma_q); // src0

        ثابت dma_addr_t dst_addr = concat_plane_addr(dst, p, div_ne2, ne2) + i * dst->nb[1];
        dma_قطار_ڌڪو(dma_q, dma_make_معلومات(dst_addr, spad0_بنياد + src0_pre), dst->nb[1], spad0_صف_بائيٽون, (src0_ne0 + src1_ne0) * حجم(__fp16), current_بلاڪ_i);

        إذا (has_ايندڙ) {
            dma_قطار_ڪڍو(dma_q); // next src1
        }
        dma_قطار_ڪڍو(dma_q); // dst

        // spad0 is free after the dst write
        إذا (has_ايندڙ) {
            ثابت dma_addr_t nsrc0_addr = concat_plane_addr(src0, np, div_ne2, ne2) + ni * src0->nb[1];
            dma_قطار_ڌڪو(dma_q, dma_make_معلومات(spad0_بنياد + src0_pre, nsrc0_addr), spad0_صف_بائيٽون, src0->nb[1], src0_صف_بائيٽون, ايندڙ_بلاڪ_i);
        }

        p = np;
        i = ni;
    }
    dma_قطار_flush(dma_q);
}

ساكن صحيح concat_regular(بنية htp_ops_context * octx, صحيح dim, uint32_t type_سائيز) {
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * src1 = octx->src[1];
    ثابت بنية htp_tensor * dst  = octx->dst;

    بنية htp_tensor view0 = *dst;
    بنية htp_tensor view1 = *dst;
    لكل (صحيح d = 0; d < HTP_OP_MAX_DIMS; d++) {
        view0.ne[d] = src0->ne[d];
        view1.ne[d] = src1->ne[d];
    }
    view1.معلومات += src0->ne[dim] * dst->nb[dim];

    ثابت uint32_t ڪل_صفون_0 = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت uint32_t ڪل_صفون_1 = src1->ne[1] * src1->ne[2] * src1->ne[3];

    uint32_t rstart0 = 0, nrows0 = ڪل_صفون_0;
    uint32_t rstart1 = 0, nrows1 = ڪل_صفون_1;

    إذا (octx->ctx->mdev.ڳڻپ > 1) {
        ثابت بنية htp_tensor_mdev_range range0 = htp_tensor_mdev_حصا_ڪريو(
            ڪل_صفون_0, 1, octx->ctx->mdev.idx, octx->ctx->mdev.ڳڻپ, &octx->ctx->mdev.ڳڻپ_div);
        rstart0 = range0.شروعات;
        nrows0  = range0.ڳڻپ;

        ثابت بنية htp_tensor_mdev_range range1 = htp_tensor_mdev_حصا_ڪريو(
            ڪل_صفون_1, 1, octx->ctx->mdev.idx, octx->ctx->mdev.ڳڻپ, &octx->ctx->mdev.ڳڻپ_div);
        rstart1 = range1.شروعات;
        nrows1  = range1.ڳڻپ;
    }

    dma_قطار * q = octx->ctx->dma[0];

    dma_cpy_sametype_sameshape_range(q, &view0, src0, type_سائيز, rstart0, nrows0);
    dma_cpy_sametype_sameshape_range(q, &view1, src1, type_سائيز, rstart1, nrows1);
    dma_قطار_flush(q);

    أرجع HTP_حالت_OK;
}

ساكن صحيح concat_transposed(بنية htp_ops_context * octx, ثابت بنية htp_concat_kernel_params * kparams, uint32_t type_سائيز) {
    إذا (!htp_ops_context_سيٽ_n_ڌاڳا(octx, kparams->n_ڌاڳا)) {
        أرجع HTP_حالت_INVAL_PARAMS;
    }

    ثابت بنية htp_tensor * dst = octx->dst;

    ثابت uint32_t ڪل_صفون = dst->ne[1];
    uint32_t صف_شروعات = 0;
    uint32_t nrows     = ڪل_صفون;
    إذا (octx->ctx->mdev.ڳڻپ > 1) {
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصا_ڪريو(ڪل_صفون, 1, octx->ctx->mdev.idx, octx->ctx->mdev.ڳڻپ, &octx->ctx->mdev.ڳڻپ_div);
        صف_شروعات = range.شروعات;
        nrows     = range.ڳڻپ;
    }

    إذا (nrows == 0 || dst->ne[2] == 0 || dst->ne[3] == 0) {
        أرجع HTP_حالت_OK;
    }

    إذا (kparams->vtcm_سائيز > octx->ctx->vtcm_سائيز) {
        أرجع HTP_حالت_VTCM_TOO_SMALL;
    }

    ثابت uint32_t n_ڌاڳا = octx->n_ڌاڳا;

    // layout precomputed on host; kept for reference:
    // struct htp_concat_transposed_vtcm_layout layout;
    // htp_concat_transposed_vtcm_layout_build(&layout, octx->src[0]->ne[0], octx->src[1]->ne[0], type_size, n_threads);

    uint8_t * vtcm_بنياد = (uint8_t *) octx->ctx->vtcm_بنياد;

    بنية htp_concat_context cctx;
    cctx.octx                  = octx;
    cctx.spad0_بنياد            = vtcm_بنياد;
    cctx.spad1_بنياد            = vtcm_بنياد + n_ڌاڳا * kparams->u.transposed.spad0_سائيز_في_thread;
    cctx.spad0_سائيز_في_thread = kparams->u.transposed.spad0_سائيز_في_thread;
    cctx.spad1_سائيز_في_thread = kparams->u.transposed.spad1_سائيز_في_thread;
    cctx.صف_شروعات             = صف_شروعات;
    cctx.nrows                 = nrows;
    cctx.nplanes               = dst->ne[2] * dst->ne[3];
    cctx.div_ne2               = init_fastdiv_قدرون(dst->ne[2]);
    cctx.nrows_في_thread      = fastdiv(nrows + n_ڌاڳا - 1, &octx->n_ڌاڳا_div);

    work_قطار_func_t ڪم_ڪندڙ_func = (type_سائيز == 4) ? concat_2d_f32_transposed : concat_2d_f16_transposed;
    work_قطار_لاڳيتو(octx->ctx->work_قطار, ڪم_ڪندڙ_func, &cctx, n_ڌاڳا);
    أرجع HTP_حالت_OK;
}

صحيح op_concat(بنية htp_ops_context * octx) {
    ثابت بنية htp_concat_kernel_params * kparams = (ثابت بنية htp_concat_kernel_params *) octx->kernel_params;
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * src1 = octx->src[1];
    ثابت بنية htp_tensor * dst  = octx->dst;

    صحيح حالت = HTP_حالت_OK;
    ثابت uint32_t type_سائيز = (dst->type == HTP_TYPE_F32 || dst->type == HTP_TYPE_I32) ? 4 : 2;

    بدل (kparams->kernel_type) {
        حالة HTP_CONCAT_KERNEL_REGULAR:
            حالت = concat_regular(octx, kparams->dim, type_سائيز);
            اكسر;

        حالة HTP_CONCAT_KERNEL_TRANSPOSED:
            حالت = concat_transposed(octx, kparams, type_سائيز);
            اكسر;

        حالة HTP_CONCAT_KERNEL_GATHER_صفون:
            حالت = htp_gather_صفون(octx, src0, src1, &kparams->u.gather, kparams->n_ڌاڳا);
            اكسر;

        افتراضي:
            حالت = HTP_حالت_NO_SUPPORT;
            اكسر;
    }

    htp_ops_context_سيٽ_حالت(octx, حالت);

    أرجع octx->حالت;
}
