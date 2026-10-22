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
    uint8_t * spad0_بنسټ;
    uint8_t * spad1_بنسټ;
    uint32_t  spad0_اندازه_فی_thread;
    uint32_t  spad1_اندازه_فی_thread;
    uint32_t  لیکه_پیل;
    uint32_t  nrows;
    uint32_t  nrows_فی_thread;
    uint32_t  nplanes;
    بنية fastdiv_ارزښتونه div_ne2;
};

ساكن مضمن dma_addr_t concat_plane_addr(ثابت بنية htp_tensor * t, uint32_t p, ثابت بنية fastdiv_ارزښتونه * div_ne2, uint32_t ne2) {
    ثابت uint32_t i3 = fastdiv(p, div_ne2);
    ثابت uint32_t i2 = p - i3 * ne2;
    أرجع t->معلومات + i2 * t->nb[2] + i3 * t->nb[3];
}

ساكن فراغ concat_2d_f32_transposed(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    بنية htp_concat_context * cctx = (بنية htp_concat_context *) معلومات;
    بنية htp_ops_context * octx = cctx->octx;

    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * src1 = octx->src[1];
    ثابت بنية htp_tensor * dst  = octx->dst;

    ثابت uint32_t src0_ne0 = src0->ne[0];
    ثابت uint32_t src1_ne0 = src1->ne[0];

    ثابت uint32_t لیکه_پای_ټکی = cctx->لیکه_پیل + cctx->nrows;
    ثابت uint32_t پیل_i = cctx->لیکه_پیل + ith * cctx->nrows_فی_thread;
    ثابت uint32_t پای_ټکی_i   = (پیل_i + cctx->nrows_فی_thread < لیکه_پای_ټکی) ? (پیل_i + cctx->nrows_فی_thread) : لیکه_پای_ټکی;
    إذا (پیل_i >= پای_ټکی_i) أرجع;

    dma_کتار * dma_q = octx->ctx->dma[ith];

    uint8_t * spad0_بنسټ = cctx->spad0_بنسټ + ith * cctx->spad0_اندازه_فی_thread;
    uint8_t * spad1_بنسټ = cctx->spad1_بنسټ + ith * cctx->spad1_اندازه_فی_thread;

    ثابت uint32_t بلاک_i = 32;
    ثابت uint32_t spad1_stride = بلاک_i * حجم(عائم);

    ثابت HVX_ویکټور offsets = hvx_vec_gather_offsets_w(spad1_stride);
    ثابت uint32_t src1_ne0_padded = hex_پړاو_پورته(src1_ne0, 32);
    ثابت uint32_t src0_لیکه_بایټونه  = src0_ne0 * حجم(عائم);
    ثابت uint32_t src0_لیکه_padded = hex_پړاو_پورته(src0_لیکه_بایټونه, VLEN);
    ثابت uint32_t src0_pre        = src0_لیکه_padded - src0_لیکه_بایټونه;
    ثابت uint32_t spad0_لیکه_بایټونه = src0_لیکه_padded + src1_ne0_padded * حجم(عائم);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت بنية fastdiv_ارزښتونه * div_ne2 = &cctx->div_ne2;
    ثابت uint32_t ne2 = dst->ne[2];

    uint32_t p = 0;
    uint32_t i = پیل_i;

    ثابت dma_addr_t src1_addr = concat_plane_addr(src1, p, div_ne2, ne2) + i * src1->nb[1];
    dma_کتار_ټېل(dma_q, dma_make_معلومات(spad1_بنسټ, src1_addr), spad1_stride, src1->nb[0], MIN(پای_ټکی_i - i, بلاک_i) * حجم(عائم), src1_ne0);

    ثابت dma_addr_t src0_addr = concat_plane_addr(src0, p, div_ne2, ne2) + i * src0->nb[1];
    dma_کتار_ټېل(dma_q, dma_make_معلومات(spad0_بنسټ + src0_pre, src0_addr), spad0_لیکه_بایټونه, src0->nb[1], src0_لیکه_بایټونه, MIN(پای_ټکی_i - i, بلاک_i));

    dma_کتار_راوباسه(dma_q); // src1

    طالما (p < cctx->nplanes) {
        ثابت uint32_t current_بلاک_i = MIN(پای_ټکی_i - i, بلاک_i);

        htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        لكل (uint32_t j = 0; j < src1_ne0; j += 32) {
            ثابت uint8_t * src_ptr = spad1_بنسټ + j * spad1_stride;
            uint8_t * dst_ptr = spad0_بنسټ + src0_لیکه_padded + j * حجم(عائم);
            hvx_transpose_32x32_w_gather(dst_ptr, spad0_لیکه_بایټونه, src_ptr, spad1_stride, offsets, current_بلاک_i, MIN(src1_ne0 - j, 32));
        }
        hvx_gather_sync(spad0_بنسټ + src0_لیکه_padded);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        uint32_t np = p;
        uint32_t ni = i + بلاک_i;
        إذا (ni >= پای_ټکی_i) {
            ni = پیل_i;
            np++;
        }
        ثابت bool has_بل = np < cctx->nplanes;
        ثابت uint32_t بل_بلاک_i = MIN(پای_ټکی_i - ni, بلاک_i);

        // spad1 is free after the gather sync, prefetch next src1 ahead of the dst write
        إذا (has_بل) {
            ثابت dma_addr_t nsrc1_addr = concat_plane_addr(src1, np, div_ne2, ne2) + ni * src1->nb[1];
            dma_کتار_ټېل(dma_q, dma_make_معلومات(spad1_بنسټ, nsrc1_addr), spad1_stride, src1->nb[0], بل_بلاک_i * حجم(عائم), src1_ne0);
        }

        dma_کتار_راوباسه(dma_q); // src0

        ثابت dma_addr_t dst_addr = concat_plane_addr(dst, p, div_ne2, ne2) + i * dst->nb[1];
        dma_کتار_ټېل(dma_q, dma_make_معلومات(dst_addr, spad0_بنسټ + src0_pre), dst->nb[1], spad0_لیکه_بایټونه, (src0_ne0 + src1_ne0) * حجم(عائم), current_بلاک_i);

        إذا (has_بل) {
            dma_کتار_راوباسه(dma_q); // next src1
        }
        dma_کتار_راوباسه(dma_q); // dst

        // spad0 is free after the dst write
        إذا (has_بل) {
            ثابت dma_addr_t nsrc0_addr = concat_plane_addr(src0, np, div_ne2, ne2) + ni * src0->nb[1];
            dma_کتار_ټېل(dma_q, dma_make_معلومات(spad0_بنسټ + src0_pre, nsrc0_addr), spad0_لیکه_بایټونه, src0->nb[1], src0_لیکه_بایټونه, بل_بلاک_i);
        }

        p = np;
        i = ni;
    }
    dma_کتار_flush(dma_q);
}

ساكن فراغ concat_2d_f16_transposed(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    بنية htp_concat_context * cctx = (بنية htp_concat_context *) معلومات;
    بنية htp_ops_context * octx = cctx->octx;

    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * src1 = octx->src[1];
    ثابت بنية htp_tensor * dst  = octx->dst;

    ثابت uint32_t src0_ne0 = src0->ne[0];
    ثابت uint32_t src1_ne0 = src1->ne[0];

    ثابت uint32_t لیکه_پای_ټکی = cctx->لیکه_پیل + cctx->nrows;
    ثابت uint32_t پیل_i = cctx->لیکه_پیل + ith * cctx->nrows_فی_thread;
    ثابت uint32_t پای_ټکی_i   = (پیل_i + cctx->nrows_فی_thread < لیکه_پای_ټکی) ? (پیل_i + cctx->nrows_فی_thread) : لیکه_پای_ټکی;
    إذا (پیل_i >= پای_ټکی_i) أرجع;

    dma_کتار * dma_q = octx->ctx->dma[ith];

    uint8_t * spad0_بنسټ = cctx->spad0_بنسټ + ith * cctx->spad0_اندازه_فی_thread;
    uint8_t * spad1_بنسټ = cctx->spad1_بنسټ + ith * cctx->spad1_اندازه_فی_thread;

    ثابت uint32_t بلاک_i = 64;
    ثابت uint32_t spad1_stride = بلاک_i * حجم(__fp16);

    ثابت HVX_ویکټور offsets = hvx_vec_gather_offsets_h(spad1_stride);
    ثابت uint32_t src1_ne0_padded = hex_پړاو_پورته(src1_ne0, 64);
    ثابت uint32_t src0_لیکه_بایټونه  = src0_ne0 * حجم(__fp16);
    ثابت uint32_t src0_لیکه_padded = hex_پړاو_پورته(src0_لیکه_بایټونه, VLEN);
    ثابت uint32_t src0_pre        = src0_لیکه_padded - src0_لیکه_بایټونه;
    ثابت uint32_t spad0_لیکه_بایټونه = src0_لیکه_padded + src1_ne0_padded * حجم(__fp16);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت بنية fastdiv_ارزښتونه * div_ne2 = &cctx->div_ne2;
    ثابت uint32_t ne2 = dst->ne[2];

    uint32_t p = 0;
    uint32_t i = پیل_i;

    ثابت dma_addr_t src1_addr = concat_plane_addr(src1, p, div_ne2, ne2) + i * src1->nb[1];
    dma_کتار_ټېل(dma_q, dma_make_معلومات(spad1_بنسټ, src1_addr), spad1_stride, src1->nb[0], MIN(پای_ټکی_i - i, بلاک_i) * حجم(__fp16), src1_ne0);

    ثابت dma_addr_t src0_addr = concat_plane_addr(src0, p, div_ne2, ne2) + i * src0->nb[1];
    dma_کتار_ټېل(dma_q, dma_make_معلومات(spad0_بنسټ + src0_pre, src0_addr), spad0_لیکه_بایټونه, src0->nb[1], src0_لیکه_بایټونه, MIN(پای_ټکی_i - i, بلاک_i));

    dma_کتار_راوباسه(dma_q); // src1

    طالما (p < cctx->nplanes) {
        ثابت uint32_t current_بلاک_i = MIN(پای_ټکی_i - i, بلاک_i);

        htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        لكل (uint32_t j = 0; j < src1_ne0; j += 64) {
            ثابت uint8_t * src_ptr = spad1_بنسټ + j * spad1_stride;
            uint8_t * dst_ptr = spad0_بنسټ + src0_لیکه_padded + j * حجم(__fp16);
            hvx_transpose_64x64_h_gather(dst_ptr, spad0_لیکه_بایټونه, src_ptr, spad1_stride, offsets, current_بلاک_i, MIN(src1_ne0 - j, 64));
        }
        hvx_gather_sync(spad0_بنسټ + src0_لیکه_padded);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        uint32_t np = p;
        uint32_t ni = i + بلاک_i;
        إذا (ni >= پای_ټکی_i) {
            ni = پیل_i;
            np++;
        }
        ثابت bool has_بل = np < cctx->nplanes;
        ثابت uint32_t بل_بلاک_i = MIN(پای_ټکی_i - ni, بلاک_i);

        // spad1 is free after the gather sync, prefetch next src1 ahead of the dst write
        إذا (has_بل) {
            ثابت dma_addr_t nsrc1_addr = concat_plane_addr(src1, np, div_ne2, ne2) + ni * src1->nb[1];
            dma_کتار_ټېل(dma_q, dma_make_معلومات(spad1_بنسټ, nsrc1_addr), spad1_stride, src1->nb[0], بل_بلاک_i * حجم(__fp16), src1_ne0);
        }

        dma_کتار_راوباسه(dma_q); // src0

        ثابت dma_addr_t dst_addr = concat_plane_addr(dst, p, div_ne2, ne2) + i * dst->nb[1];
        dma_کتار_ټېل(dma_q, dma_make_معلومات(dst_addr, spad0_بنسټ + src0_pre), dst->nb[1], spad0_لیکه_بایټونه, (src0_ne0 + src1_ne0) * حجم(__fp16), current_بلاک_i);

        إذا (has_بل) {
            dma_کتار_راوباسه(dma_q); // next src1
        }
        dma_کتار_راوباسه(dma_q); // dst

        // spad0 is free after the dst write
        إذا (has_بل) {
            ثابت dma_addr_t nsrc0_addr = concat_plane_addr(src0, np, div_ne2, ne2) + ni * src0->nb[1];
            dma_کتار_ټېل(dma_q, dma_make_معلومات(spad0_بنسټ + src0_pre, nsrc0_addr), spad0_لیکه_بایټونه, src0->nb[1], src0_لیکه_بایټونه, بل_بلاک_i);
        }

        p = np;
        i = ni;
    }
    dma_کتار_flush(dma_q);
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
    view1.معلومات += src0->ne[dim] * dst->nb[dim];

    ثابت uint32_t ټول_لیکې_0 = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت uint32_t ټول_لیکې_1 = src1->ne[1] * src1->ne[2] * src1->ne[3];

    uint32_t rstart0 = 0, nrows0 = ټول_لیکې_0;
    uint32_t rstart1 = 0, nrows1 = ټول_لیکې_1;

    إذا (octx->ctx->mdev.شمېر > 1) {
        ثابت بنية htp_tensor_mdev_range range0 = htp_tensor_mdev_برخه_کول(
            ټول_لیکې_0, 1, octx->ctx->mdev.idx, octx->ctx->mdev.شمېر, &octx->ctx->mdev.شمېر_div);
        rstart0 = range0.پیل;
        nrows0  = range0.شمېر;

        ثابت بنية htp_tensor_mdev_range range1 = htp_tensor_mdev_برخه_کول(
            ټول_لیکې_1, 1, octx->ctx->mdev.idx, octx->ctx->mdev.شمېر, &octx->ctx->mdev.شمېر_div);
        rstart1 = range1.پیل;
        nrows1  = range1.شمېر;
    }

    dma_کتار * q = octx->ctx->dma[0];

    dma_cpy_sametype_sameshape_range(q, &view0, src0, type_اندازه, rstart0, nrows0);
    dma_cpy_sametype_sameshape_range(q, &view1, src1, type_اندازه, rstart1, nrows1);
    dma_کتار_flush(q);

    أرجع HTP_حالت_OK;
}

ساكن صحيح concat_transposed(بنية htp_ops_context * octx, ثابت بنية htp_concat_kernel_params * kparams, uint32_t type_اندازه) {
    إذا (!htp_ops_context_سېټ_n_تارونه(octx, kparams->n_تارونه)) {
        أرجع HTP_حالت_INVAL_PARAMS;
    }

    ثابت بنية htp_tensor * dst = octx->dst;

    ثابت uint32_t ټول_لیکې = dst->ne[1];
    uint32_t لیکه_پیل = 0;
    uint32_t nrows     = ټول_لیکې;
    إذا (octx->ctx->mdev.شمېر > 1) {
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_برخه_کول(ټول_لیکې, 1, octx->ctx->mdev.idx, octx->ctx->mdev.شمېر, &octx->ctx->mdev.شمېر_div);
        لیکه_پیل = range.پیل;
        nrows     = range.شمېر;
    }

    إذا (nrows == 0 || dst->ne[2] == 0 || dst->ne[3] == 0) {
        أرجع HTP_حالت_OK;
    }

    إذا (kparams->vtcm_اندازه > octx->ctx->vtcm_اندازه) {
        أرجع HTP_حالت_VTCM_TOO_SMALL;
    }

    ثابت uint32_t n_تارونه = octx->n_تارونه;

    // layout precomputed on host; kept for reference:
    // struct htp_concat_transposed_vtcm_layout layout;
    // htp_concat_transposed_vtcm_layout_build(&layout, octx->src[0]->ne[0], octx->src[1]->ne[0], type_size, n_threads);

    uint8_t * vtcm_بنسټ = (uint8_t *) octx->ctx->vtcm_بنسټ;

    بنية htp_concat_context cctx;
    cctx.octx                  = octx;
    cctx.spad0_بنسټ            = vtcm_بنسټ;
    cctx.spad1_بنسټ            = vtcm_بنسټ + n_تارونه * kparams->u.transposed.spad0_اندازه_فی_thread;
    cctx.spad0_اندازه_فی_thread = kparams->u.transposed.spad0_اندازه_فی_thread;
    cctx.spad1_اندازه_فی_thread = kparams->u.transposed.spad1_اندازه_فی_thread;
    cctx.لیکه_پیل             = لیکه_پیل;
    cctx.nrows                 = nrows;
    cctx.nplanes               = dst->ne[2] * dst->ne[3];
    cctx.div_ne2               = init_fastdiv_ارزښتونه(dst->ne[2]);
    cctx.nrows_فی_thread      = fastdiv(nrows + n_تارونه - 1, &octx->n_تارونه_div);

    work_کتار_func_t کارګر_func = (type_اندازه == 4) ? concat_2d_f32_transposed : concat_2d_f16_transposed;
    work_کتار_پرلپسې(octx->ctx->work_کتار, کارګر_func, &cctx, n_تارونه);
    أرجع HTP_حالت_OK;
}

صحيح op_concat(بنية htp_ops_context * octx) {
    ثابت بنية htp_concat_kernel_params * kparams = (ثابت بنية htp_concat_kernel_params *) octx->kernel_params;
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * src1 = octx->src[1];
    ثابت بنية htp_tensor * dst  = octx->dst;

    صحيح حالت = HTP_حالت_OK;
    ثابت uint32_t type_اندازه = (dst->type == HTP_TYPE_F32 || dst->type == HTP_TYPE_I32) ? 4 : 2;

    بدل (kparams->kernel_type) {
        حالة HTP_CONCAT_KERNEL_REGULAR:
            حالت = concat_regular(octx, kparams->dim, type_اندازه);
            اكسر;

        حالة HTP_CONCAT_KERNEL_TRANSPOSED:
            حالت = concat_transposed(octx, kparams, type_اندازه);
            اكسر;

        حالة HTP_CONCAT_KERNEL_GATHER_لیکې:
            حالت = htp_gather_لیکې(octx, src0, src1, &kparams->u.gather, kparams->n_تارونه);
            اكسر;

        افتراضي:
            حالت = HTP_حالت_NO_SUPPORT;
            اكسر;
    }

    htp_ops_context_سېټ_حالت(octx, حالت);

    أرجع octx->حالت;
}
