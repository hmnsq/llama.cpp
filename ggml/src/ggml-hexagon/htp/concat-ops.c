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
    uint8_t * spad0_پایه;
    uint8_t * spad1_پایه;
    uint32_t  spad0_اندازه_در_هر_thread;
    uint32_t  spad1_اندازه_در_هر_thread;
    uint32_t  ردیف_شروع;
    uint32_t  nrows;
    uint32_t  nrows_در_هر_thread;
    uint32_t  nplanes;
    بنية fastdiv_مقادیر div_ne2;
};

ساكن مضمن dma_addr_t concat_plane_addr(ثابت بنية htp_tensor * t, uint32_t p, ثابت بنية fastdiv_مقادیر * div_ne2, uint32_t ne2) {
    ثابت uint32_t i3 = fastdiv(p, div_ne2);
    ثابت uint32_t i2 = p - i3 * ne2;
    أرجع t->داده + i2 * t->nb[2] + i3 * t->nb[3];
}

ساكن فراغ concat_2d_f32_transposed(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    بنية htp_concat_context * cctx = (بنية htp_concat_context *) داده;
    بنية htp_ops_context * octx = cctx->octx;

    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * src1 = octx->src[1];
    ثابت بنية htp_tensor * dst  = octx->dst;

    ثابت uint32_t src0_ne0 = src0->ne[0];
    ثابت uint32_t src1_ne0 = src1->ne[0];

    ثابت uint32_t ردیف_انتها = cctx->ردیف_شروع + cctx->nrows;
    ثابت uint32_t شروع_i = cctx->ردیف_شروع + ith * cctx->nrows_در_هر_thread;
    ثابت uint32_t انتها_i   = (شروع_i + cctx->nrows_در_هر_thread < ردیف_انتها) ? (شروع_i + cctx->nrows_در_هر_thread) : ردیف_انتها;
    إذا (شروع_i >= انتها_i) أرجع;

    dma_صف * dma_q = octx->ctx->dma[ith];

    uint8_t * spad0_پایه = cctx->spad0_پایه + ith * cctx->spad0_اندازه_در_هر_thread;
    uint8_t * spad1_پایه = cctx->spad1_پایه + ith * cctx->spad1_اندازه_در_هر_thread;

    ثابت uint32_t بلوک_i = 32;
    ثابت uint32_t spad1_stride = بلوک_i * حجم(عائم);

    ثابت HVX_وکتور offsets = hvx_vec_gather_offsets_w(spad1_stride);
    ثابت uint32_t src1_ne0_padded = hex_دور_به_بالا(src1_ne0, 32);
    ثابت uint32_t src0_ردیف_بایتها  = src0_ne0 * حجم(عائم);
    ثابت uint32_t src0_ردیف_padded = hex_دور_به_بالا(src0_ردیف_بایتها, VLEN);
    ثابت uint32_t src0_pre        = src0_ردیف_padded - src0_ردیف_بایتها;
    ثابت uint32_t spad0_ردیف_بایتها = src0_ردیف_padded + src1_ne0_padded * حجم(عائم);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت بنية fastdiv_مقادیر * div_ne2 = &cctx->div_ne2;
    ثابت uint32_t ne2 = dst->ne[2];

    uint32_t p = 0;
    uint32_t i = شروع_i;

    ثابت dma_addr_t src1_addr = concat_plane_addr(src1, p, div_ne2, ne2) + i * src1->nb[1];
    dma_صف_افزودن(dma_q, dma_make_داده(spad1_پایه, src1_addr), spad1_stride, src1->nb[0], MIN(انتها_i - i, بلوک_i) * حجم(عائم), src1_ne0);

    ثابت dma_addr_t src0_addr = concat_plane_addr(src0, p, div_ne2, ne2) + i * src0->nb[1];
    dma_صف_افزودن(dma_q, dma_make_داده(spad0_پایه + src0_pre, src0_addr), spad0_ردیف_بایتها, src0->nb[1], src0_ردیف_بایتها, MIN(انتها_i - i, بلوک_i));

    dma_صف_برداشت(dma_q); // src1

    طالما (p < cctx->nplanes) {
        ثابت uint32_t current_بلوک_i = MIN(انتها_i - i, بلوک_i);

        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        لكل (uint32_t j = 0; j < src1_ne0; j += 32) {
            ثابت uint8_t * src_ptr = spad1_پایه + j * spad1_stride;
            uint8_t * dst_ptr = spad0_پایه + src0_ردیف_padded + j * حجم(عائم);
            hvx_transpose_32x32_w_gather(dst_ptr, spad0_ردیف_بایتها, src_ptr, spad1_stride, offsets, current_بلوک_i, MIN(src1_ne0 - j, 32));
        }
        hvx_gather_sync(spad0_پایه + src0_ردیف_padded);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        uint32_t np = p;
        uint32_t ni = i + بلوک_i;
        إذا (ni >= انتها_i) {
            ni = شروع_i;
            np++;
        }
        ثابت bool has_بعدی = np < cctx->nplanes;
        ثابت uint32_t بعدی_بلوک_i = MIN(انتها_i - ni, بلوک_i);

        // spad1 is free after the gather sync, prefetch next src1 ahead of the dst write
        إذا (has_بعدی) {
            ثابت dma_addr_t nsrc1_addr = concat_plane_addr(src1, np, div_ne2, ne2) + ni * src1->nb[1];
            dma_صف_افزودن(dma_q, dma_make_داده(spad1_پایه, nsrc1_addr), spad1_stride, src1->nb[0], بعدی_بلوک_i * حجم(عائم), src1_ne0);
        }

        dma_صف_برداشت(dma_q); // src0

        ثابت dma_addr_t dst_addr = concat_plane_addr(dst, p, div_ne2, ne2) + i * dst->nb[1];
        dma_صف_افزودن(dma_q, dma_make_داده(dst_addr, spad0_پایه + src0_pre), dst->nb[1], spad0_ردیف_بایتها, (src0_ne0 + src1_ne0) * حجم(عائم), current_بلوک_i);

        إذا (has_بعدی) {
            dma_صف_برداشت(dma_q); // next src1
        }
        dma_صف_برداشت(dma_q); // dst

        // spad0 is free after the dst write
        إذا (has_بعدی) {
            ثابت dma_addr_t nsrc0_addr = concat_plane_addr(src0, np, div_ne2, ne2) + ni * src0->nb[1];
            dma_صف_افزودن(dma_q, dma_make_داده(spad0_پایه + src0_pre, nsrc0_addr), spad0_ردیف_بایتها, src0->nb[1], src0_ردیف_بایتها, بعدی_بلوک_i);
        }

        p = np;
        i = ni;
    }
    dma_صف_flush(dma_q);
}

ساكن فراغ concat_2d_f16_transposed(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    بنية htp_concat_context * cctx = (بنية htp_concat_context *) داده;
    بنية htp_ops_context * octx = cctx->octx;

    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * src1 = octx->src[1];
    ثابت بنية htp_tensor * dst  = octx->dst;

    ثابت uint32_t src0_ne0 = src0->ne[0];
    ثابت uint32_t src1_ne0 = src1->ne[0];

    ثابت uint32_t ردیف_انتها = cctx->ردیف_شروع + cctx->nrows;
    ثابت uint32_t شروع_i = cctx->ردیف_شروع + ith * cctx->nrows_در_هر_thread;
    ثابت uint32_t انتها_i   = (شروع_i + cctx->nrows_در_هر_thread < ردیف_انتها) ? (شروع_i + cctx->nrows_در_هر_thread) : ردیف_انتها;
    إذا (شروع_i >= انتها_i) أرجع;

    dma_صف * dma_q = octx->ctx->dma[ith];

    uint8_t * spad0_پایه = cctx->spad0_پایه + ith * cctx->spad0_اندازه_در_هر_thread;
    uint8_t * spad1_پایه = cctx->spad1_پایه + ith * cctx->spad1_اندازه_در_هر_thread;

    ثابت uint32_t بلوک_i = 64;
    ثابت uint32_t spad1_stride = بلوک_i * حجم(__fp16);

    ثابت HVX_وکتور offsets = hvx_vec_gather_offsets_h(spad1_stride);
    ثابت uint32_t src1_ne0_padded = hex_دور_به_بالا(src1_ne0, 64);
    ثابت uint32_t src0_ردیف_بایتها  = src0_ne0 * حجم(__fp16);
    ثابت uint32_t src0_ردیف_padded = hex_دور_به_بالا(src0_ردیف_بایتها, VLEN);
    ثابت uint32_t src0_pre        = src0_ردیف_padded - src0_ردیف_بایتها;
    ثابت uint32_t spad0_ردیف_بایتها = src0_ردیف_padded + src1_ne0_padded * حجم(__fp16);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت بنية fastdiv_مقادیر * div_ne2 = &cctx->div_ne2;
    ثابت uint32_t ne2 = dst->ne[2];

    uint32_t p = 0;
    uint32_t i = شروع_i;

    ثابت dma_addr_t src1_addr = concat_plane_addr(src1, p, div_ne2, ne2) + i * src1->nb[1];
    dma_صف_افزودن(dma_q, dma_make_داده(spad1_پایه, src1_addr), spad1_stride, src1->nb[0], MIN(انتها_i - i, بلوک_i) * حجم(__fp16), src1_ne0);

    ثابت dma_addr_t src0_addr = concat_plane_addr(src0, p, div_ne2, ne2) + i * src0->nb[1];
    dma_صف_افزودن(dma_q, dma_make_داده(spad0_پایه + src0_pre, src0_addr), spad0_ردیف_بایتها, src0->nb[1], src0_ردیف_بایتها, MIN(انتها_i - i, بلوک_i));

    dma_صف_برداشت(dma_q); // src1

    طالما (p < cctx->nplanes) {
        ثابت uint32_t current_بلوک_i = MIN(انتها_i - i, بلوک_i);

        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        لكل (uint32_t j = 0; j < src1_ne0; j += 64) {
            ثابت uint8_t * src_ptr = spad1_پایه + j * spad1_stride;
            uint8_t * dst_ptr = spad0_پایه + src0_ردیف_padded + j * حجم(__fp16);
            hvx_transpose_64x64_h_gather(dst_ptr, spad0_ردیف_بایتها, src_ptr, spad1_stride, offsets, current_بلوک_i, MIN(src1_ne0 - j, 64));
        }
        hvx_gather_sync(spad0_پایه + src0_ردیف_padded);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        uint32_t np = p;
        uint32_t ni = i + بلوک_i;
        إذا (ni >= انتها_i) {
            ni = شروع_i;
            np++;
        }
        ثابت bool has_بعدی = np < cctx->nplanes;
        ثابت uint32_t بعدی_بلوک_i = MIN(انتها_i - ni, بلوک_i);

        // spad1 is free after the gather sync, prefetch next src1 ahead of the dst write
        إذا (has_بعدی) {
            ثابت dma_addr_t nsrc1_addr = concat_plane_addr(src1, np, div_ne2, ne2) + ni * src1->nb[1];
            dma_صف_افزودن(dma_q, dma_make_داده(spad1_پایه, nsrc1_addr), spad1_stride, src1->nb[0], بعدی_بلوک_i * حجم(__fp16), src1_ne0);
        }

        dma_صف_برداشت(dma_q); // src0

        ثابت dma_addr_t dst_addr = concat_plane_addr(dst, p, div_ne2, ne2) + i * dst->nb[1];
        dma_صف_افزودن(dma_q, dma_make_داده(dst_addr, spad0_پایه + src0_pre), dst->nb[1], spad0_ردیف_بایتها, (src0_ne0 + src1_ne0) * حجم(__fp16), current_بلوک_i);

        إذا (has_بعدی) {
            dma_صف_برداشت(dma_q); // next src1
        }
        dma_صف_برداشت(dma_q); // dst

        // spad0 is free after the dst write
        إذا (has_بعدی) {
            ثابت dma_addr_t nsrc0_addr = concat_plane_addr(src0, np, div_ne2, ne2) + ni * src0->nb[1];
            dma_صف_افزودن(dma_q, dma_make_داده(spad0_پایه + src0_pre, nsrc0_addr), spad0_ردیف_بایتها, src0->nb[1], src0_ردیف_بایتها, بعدی_بلوک_i);
        }

        p = np;
        i = ni;
    }
    dma_صف_flush(dma_q);
}

ساكن صحيح concat_regular(بنية htp_ops_context * octx, صحيح dim, uint32_t type_اندازه) {
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * src1 = octx->src[1];
    ثابت بنية htp_tensor * dst  = octx->dst;

    بنية htp_tensor view0 = *dst;
    بنية htp_tensor view1 = *dst;
    لكل (صحيح d = 0; d < HTP_OP_MAX_DIMS; d++) {
        view0.ne[d] = src0->ne[d];
        view1.ne[d] = src1->ne[d];
    }
    view1.داده += src0->ne[dim] * dst->nb[dim];

    ثابت uint32_t مجموع_ردیفها_0 = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت uint32_t مجموع_ردیفها_1 = src1->ne[1] * src1->ne[2] * src1->ne[3];

    uint32_t rstart0 = 0, nrows0 = مجموع_ردیفها_0;
    uint32_t rstart1 = 0, nrows1 = مجموع_ردیفها_1;

    إذا (octx->ctx->mdev.شمار > 1) {
        ثابت بنية htp_tensor_mdev_range range0 = htp_tensor_mdev_افراز(
            مجموع_ردیفها_0, 1, octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
        rstart0 = range0.شروع;
        nrows0  = range0.شمار;

        ثابت بنية htp_tensor_mdev_range range1 = htp_tensor_mdev_افراز(
            مجموع_ردیفها_1, 1, octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
        rstart1 = range1.شروع;
        nrows1  = range1.شمار;
    }

    dma_صف * q = octx->ctx->dma[0];

    dma_cpy_sametype_sameshape_range(q, &view0, src0, type_اندازه, rstart0, nrows0);
    dma_cpy_sametype_sameshape_range(q, &view1, src1, type_اندازه, rstart1, nrows1);
    dma_صف_flush(q);

    أرجع HTP_وضعیت_OK;
}

ساكن صحيح concat_transposed(بنية htp_ops_context * octx, ثابت بنية htp_concat_kernel_params * kparams, uint32_t type_اندازه) {
    إذا (!htp_ops_context_مجموعه_n_نخها(octx, kparams->n_نخها)) {
        أرجع HTP_وضعیت_INVAL_PARAMS;
    }

    ثابت بنية htp_tensor * dst = octx->dst;

    ثابت uint32_t مجموع_ردیفها = dst->ne[1];
    uint32_t ردیف_شروع = 0;
    uint32_t nrows     = مجموع_ردیفها;
    إذا (octx->ctx->mdev.شمار > 1) {
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_افراز(مجموع_ردیفها, 1, octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
        ردیف_شروع = range.شروع;
        nrows     = range.شمار;
    }

    إذا (nrows == 0 || dst->ne[2] == 0 || dst->ne[3] == 0) {
        أرجع HTP_وضعیت_OK;
    }

    إذا (kparams->vtcm_اندازه > octx->ctx->vtcm_اندازه) {
        أرجع HTP_وضعیت_VTCM_TOO_SMALL;
    }

    ثابت uint32_t n_نخها = octx->n_نخها;

    // layout precomputed on host; kept for reference:
    // struct htp_concat_transposed_vtcm_layout layout;
    // htp_concat_transposed_vtcm_layout_build(&layout, octx->src[0]->ne[0], octx->src[1]->ne[0], type_size, n_threads);

    uint8_t * vtcm_پایه = (uint8_t *) octx->ctx->vtcm_پایه;

    بنية htp_concat_context cctx;
    cctx.octx                  = octx;
    cctx.spad0_پایه            = vtcm_پایه;
    cctx.spad1_پایه            = vtcm_پایه + n_نخها * kparams->u.transposed.spad0_اندازه_در_هر_thread;
    cctx.spad0_اندازه_در_هر_thread = kparams->u.transposed.spad0_اندازه_در_هر_thread;
    cctx.spad1_اندازه_در_هر_thread = kparams->u.transposed.spad1_اندازه_در_هر_thread;
    cctx.ردیف_شروع             = ردیف_شروع;
    cctx.nrows                 = nrows;
    cctx.nplanes               = dst->ne[2] * dst->ne[3];
    cctx.div_ne2               = init_fastdiv_مقادیر(dst->ne[2]);
    cctx.nrows_در_هر_thread      = fastdiv(nrows + n_نخها - 1, &octx->n_نخها_div);

    work_صف_func_t کارگر_func = (type_اندازه == 4) ? concat_2d_f32_transposed : concat_2d_f16_transposed;
    work_صف_پیاپی(octx->ctx->work_صف, کارگر_func, &cctx, n_نخها);
    أرجع HTP_وضعیت_OK;
}

صحيح op_concat(بنية htp_ops_context * octx) {
    ثابت بنية htp_concat_kernel_params * kparams = (ثابت بنية htp_concat_kernel_params *) octx->kernel_params;
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * src1 = octx->src[1];
    ثابت بنية htp_tensor * dst  = octx->dst;

    صحيح وضعیت = HTP_وضعیت_OK;
    ثابت uint32_t type_اندازه = (dst->type == HTP_TYPE_F32 || dst->type == HTP_TYPE_I32) ? 4 : 2;

    بدل (kparams->kernel_type) {
        حالة HTP_CONCAT_KERNEL_REGULAR:
            وضعیت = concat_regular(octx, kparams->dim, type_اندازه);
            اكسر;

        حالة HTP_CONCAT_KERNEL_TRANSPOSED:
            وضعیت = concat_transposed(octx, kparams, type_اندازه);
            اكسر;

        حالة HTP_CONCAT_KERNEL_GATHER_ردیفها:
            وضعیت = htp_gather_ردیفها(octx, src0, src1, &kparams->u.gather, kparams->n_نخها);
            اكسر;

        افتراضي:
            وضعیت = HTP_وضعیت_NO_SUPPORT;
            اكسر;
    }

    htp_ops_context_مجموعه_وضعیت(octx, وضعیت);

    أرجع octx->وضعیت;
}
