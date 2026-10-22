#رهنمود clang diagnostic ignored "-Wunused-variable"
#رهنمود clang diagnostic ignored "-Wunused-function"
#رهنمود clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <HAP_perf.h>

#ضمن <string.h>

#ضمن "dma-queue.h"
#ضمن "hvx-utils.h"

#تعریف_کن GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "htp-ctx.h"
#ضمن "hex-common.h"
#ضمن "hex-profile.h"
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"

بنية htp_roll_context {
    بنية htp_ops_context * octx;

    uint32_t ردیف_شروع;
    uint32_t nrows;
    uint32_t nrows_در_هر_thread;

    بنية fastdiv_مقادیر div_ne1;
    بنية fastdiv_مقادیر div_ne2_ne1;
};

ساكن مضمن uint32_t htp_roll_wrap(int32_t i, uint32_t ne) {
    إذا (i < 0) {
        أرجع (uint32_t) (i + (int32_t) ne);
    }
    إذا ((uint32_t) i >= ne) {
        أرجع (uint32_t) i - ne;
    }
    أرجع (uint32_t) i;
}

#تعریف_کن htp_roll_preamble                             \
    ثابت بنية htp_tensor * src0 = octx->src[0];    \
    ثابت بنية htp_tensor * dst  = octx->dst;       \
                                                      \
    ثابت uint32_t ne0 = dst->ne[0];                  \
    ثابت uint32_t ne1 = dst->ne[1];                  \
    ثابت uint32_t ne2 = dst->ne[2];                  \
    ثابت uint32_t ne3 = dst->ne[3];                  \
                                                      \
    ثابت uint32_t nb01 = src0->nb[1];                \
    ثابت uint32_t nb02 = src0->nb[2];                \
    ثابت uint32_t nb03 = src0->nb[3];                \
                                                      \
    ثابت uint32_t nb1 = dst->nb[1];                  \
    ثابت uint32_t nb2 = dst->nb[2];                  \
    ثابت uint32_t nb3 = dst->nb[3];                  \
                                                      \
    ثابت int32_t s0 = octx->op_params[0];            \
    ثابت int32_t s1 = octx->op_params[1];            \
    ثابت int32_t s2 = octx->op_params[2];            \
    ثابت int32_t s3 = octx->op_params[3];            \
                                                      \
    ثابت uint32_t i0_src0 = htp_roll_wrap(-s0, ne0); \
    ثابت uint32_t n0      = ne0 - i0_src0;

#تعریف_کن htp_roll_dma_preamble dma_صف * q = octx->ctx->dma[0];

ساكن مضمن فراغ roll_dma_افزودن(dma_صف * q,
                                 dma_addr_t  dst,
                                 dma_addr_t  src,
                                 uint32_t    dst_stride,
                                 uint32_t    src_stride,
                                 uint32_t    بایتها,
                                 uint32_t    nrows) {
    إذا (بایتها == 0 || nrows == 0) {
        أرجع;
    }

    إذا (!dma_صف_افزودن(q, dma_make_داده(dst, src), dst_stride, src_stride, بایتها, nrows)) {
        dma_صف_flush(q);
        dma_صف_افزودن(q, dma_make_داده(dst, src),
                       dst_stride, src_stride, بایتها, nrows);
    }
}

ساكن مضمن فراغ roll_dma_افزودن_ردیفها(dma_صف *               q,
                                      ثابت بنية htp_tensor * dst,
                                      ثابت بنية htp_tensor * src0,
                                      uint32_t                  dst_ردیف,
                                      uint32_t                  src_ردیف,
                                      uint32_t                  nrows,
                                      uint32_t                  ردیف_اندازه,
                                      uint32_t                  i0_src0) {
    ثابت dma_addr_t dst_پایه = dst->داده + (size_t) dst_ردیف * ردیف_اندازه;
    ثابت dma_addr_t src_پایه = src0->داده + (size_t) src_ردیف * ردیف_اندازه;
    ثابت uint32_t   n0       = src0->ne[0] - i0_src0;

    roll_dma_افزودن(q, dst_پایه, src_پایه + (size_t) i0_src0 * حجم(عائم),
                  ردیف_اندازه, ردیف_اندازه, n0 * حجم(عائم), nrows);
    roll_dma_افزودن(q, dst_پایه + (size_t) n0 * حجم(عائم), src_پایه,
                  ردیف_اندازه, ردیف_اندازه, i0_src0 * حجم(عائم), nrows);
}

// Same row-wrap split as roll_dma_push_rows, but addressed with explicit byte strides so it
// also works for a src0 that is row-contiguous only (e.g. a permuted view) rather than fully packed.
ساكن مضمن فراغ roll_dma_افزودن_range(dma_صف * q,
                                       dma_addr_t  dst_ردیف,
                                       dma_addr_t  src_ردیف,
                                       uint32_t    dst_stride,
                                       uint32_t    src_stride,
                                       uint32_t    nrows,
                                       uint32_t    i0_src0,
                                       uint32_t    n0) {
    roll_dma_افزودن(q, dst_ردیف, src_ردیف + (size_t) i0_src0 * حجم(عائم),
                  dst_stride, src_stride, n0 * حجم(عائم), nrows);
    roll_dma_افزودن(q, dst_ردیف + (size_t) n0 * حجم(عائم), src_ردیف,
                  dst_stride, src_stride, i0_src0 * حجم(عائم), nrows);
}

ساكن صحيح roll_dma_f32_contiguous(بنية htp_ops_context * octx) {
    htp_roll_preamble;
    htp_roll_dma_preamble;

    ثابت uint32_t ردیف_اندازه = ne0 * حجم(عائم);

    إذا (s1 == 0 && s2 == 0 && s3 == 0) {
        roll_dma_افزودن_ردیفها(q, dst, src0, 0, 0, ne1 * ne2 * ne3, ردیف_اندازه, i0_src0);
        dma_صف_flush(q);
        أرجع HTP_وضعیت_OK;
    }

    إذا (s1 == 0) {
        ثابت uint32_t i2_src0 = htp_roll_wrap(-s2, ne2);
        لكل (uint32_t i3 = 0; i3 < ne3; i3++) {
            ثابت uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
            ثابت uint32_t dst_row0 = i3 * ne2 * ne1;
            ثابت uint32_t src_row0 = (i03 * ne2 + i2_src0) * ne1;
            ثابت uint32_t n2_اولی = ne2 - i2_src0;

            roll_dma_افزودن_ردیفها(q, dst, src0, dst_row0, src_row0, n2_اولی * ne1,
                               ردیف_اندازه, i0_src0);
            roll_dma_افزودن_ردیفها(q, dst, src0, dst_row0 + n2_اولی * ne1, i03 * ne2 * ne1,
                               i2_src0 * ne1, ردیف_اندازه, i0_src0);
        }

        dma_صف_flush(q);
        أرجع HTP_وضعیت_OK;
    }

    ثابت uint32_t i1_src0 = htp_roll_wrap(-s1, ne1);
    ثابت uint32_t n1_اولی = ne1 - i1_src0;

    لكل (uint32_t i3 = 0; i3 < ne3; i3++) {
        ثابت uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
        لكل (uint32_t i2 = 0; i2 < ne2; i2++) {
            ثابت uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);
            ثابت uint32_t dst_row0 = (i3 * ne2 + i2) * ne1;
            ثابت uint32_t src_row0 = (i03 * ne2 + i02) * ne1;

            roll_dma_افزودن_ردیفها(q, dst, src0, dst_row0, src_row0 + i1_src0,
                               n1_اولی, ردیف_اندازه, i0_src0);
            roll_dma_افزودن_ردیفها(q, dst, src0, dst_row0 + n1_اولی, src_row0,
                               i1_src0, ردیف_اندازه, i0_src0);
        }
    }

    dma_صف_flush(q);
    أرجع HTP_وضعیت_OK;
}

// DMA path for a row-contiguous but otherwise arbitrarily strided src0 (e.g. a permuted view).
// Same row-wrap split as above, one DMA push per (i2,i3), addressed via the real nb01/nb02/nb03
// instead of assuming a packed layout.
ساكن صحيح roll_dma_f32_strided(بنية htp_ops_context * octx) {
    htp_roll_preamble;
    htp_roll_dma_preamble;

    ثابت uint32_t i1_src0  = htp_roll_wrap(-s1, ne1);
    ثابت uint32_t n1_اولی = ne1 - i1_src0;

    لكل (uint32_t i3 = 0; i3 < ne3; i3++) {
        ثابت uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
        لكل (uint32_t i2 = 0; i2 < ne2; i2++) {
            ثابت uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);

            ثابت dma_addr_t dst_row0 = dst->داده  + (size_t) i2  * nb2  + (size_t) i3  * nb3;
            ثابت dma_addr_t src_row0 = src0->داده + (size_t) i02 * nb02 + (size_t) i03 * nb03;

            roll_dma_افزودن_range(q, dst_row0, src_row0 + (size_t) i1_src0 * nb01,
                                nb1, nb01, n1_اولی, i0_src0, n0);
            roll_dma_افزودن_range(q, dst_row0 + (size_t) n1_اولی * nb1, src_row0,
                                nb1, nb01, i1_src0, i0_src0, n0);
        }
    }

    dma_صف_flush(q);
    أرجع HTP_وضعیت_OK;
}

ساكن فراغ roll_thread_f32(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    بنية htp_roll_context * rctx = (بنية htp_roll_context *) داده;
    بنية htp_ops_context * octx = rctx->octx;

    htp_roll_preamble;

    ثابت uint32_t ردیف_شروع = rctx->ردیف_شروع + rctx->nrows_در_هر_thread * ith;
    ثابت uint32_t ردیف_انتها   = MIN(ردیف_شروع + rctx->nrows_در_هر_thread, rctx->ردیف_شروع + rctx->nrows);
    إذا (ردیف_شروع >= ردیف_انتها) {
        أرجع;
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, ردیف_شروع);

    لكل (uint32_t ردیف = ردیف_شروع; ردیف < ردیف_انتها; ردیف++) {
        ثابت uint32_t i3  = fastdiv(ردیف, &rctx->div_ne2_ne1);
        ثابت uint32_t rem = ردیف - i3 * ne2 * ne1;
        ثابت uint32_t i2  = fastdiv(rem, &rctx->div_ne1);
        ثابت uint32_t i1  = rem - i2 * ne1;

        ثابت uint32_t i01 = htp_roll_wrap((int32_t) i1 - s1, ne1);
        ثابت uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);
        ثابت uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);

        ثابت uint8_t * src_ردیف = (ثابت uint8_t *) (uintptr_t) src0->داده + i01*nb01 + i02*nb02 + i03*nb03;
        uint8_t * dst_ردیف = (uint8_t *) (uintptr_t) dst->داده + i1*nb1 + i2*nb2 + i3*nb3;

        hex_l2fetch(src_ردیف + i0_src0 * حجم(عائم), n0 * حجم(عائم), ne0 * حجم(عائم), 1);
        hvx_copy_uu(dst_ردیف, src_ردیف + i0_src0 * حجم(عائم), n0, حجم(عائم));

        إذا (i0_src0 != 0) {
            hex_l2fetch(src_ردیف, i0_src0 * حجم(عائم), ne0 * حجم(عائم), 1);
            hvx_copy_uu(dst_ردیف + n0 * حجم(عائم), src_ردیف, i0_src0, حجم(عائم));
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ردیف_شروع);

    FARF(بالا_2, "roll %d/%d: (%ux%ux%ux%u) rows %u:%u shift=(%d,%d,%d,%d)\n",
         ith, nth, ne0, ne1, ne2, ne3,
         ردیف_شروع, ردیف_انتها, s0, s1, s2, s3);
}

صحيح execute_op_roll_f32(بنية htp_ops_context * octx) {
    htp_roll_preamble;

    إذا (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        FARF(ERROR, "roll: unsupported type %u -> %u\n", src0->type, dst->type);
        أرجع HTP_وضعیت_NO_SUPPORT;
    }

    إذا (src0->nb[0] != حجم(عائم) || dst->nb[0] != حجم(عائم)) {
        FARF(ERROR, "roll: unsupported nb0 %u -> %u\n", src0->nb[0], dst->nb[0]);
        أرجع HTP_وضعیت_NO_SUPPORT;
    }

    إذا (src0->ne[0] != ne0 || src0->ne[1] != ne1 ||
        src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        FARF(ERROR, "roll: shape mismatch\n");
        أرجع HTP_وضعیت_INVAL_PARAMS;
    }

    ثابت uint32_t مجموع_ردیفها = ne1 * ne2 * ne3;
    ثابت size_t dst_ردیف_اندازه = ne0 * حجم(عائم);

    uint32_t ردیف_شروع = 0;
    uint32_t nrows     = مجموع_ردیفها;

    إذا (octx->ctx->mdev.شمار > 1) {
        uint32_t ردیفها_در_هر_chunk = 0;
        htp_tensor_mdev_ردیفها_در_هر_chunk(dst, حجم(عائم), (uint32_t) dst_ردیف_اندازه, &ردیفها_در_هر_chunk);
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_افراز(مجموع_ردیفها, ردیفها_در_هر_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
        ردیف_شروع = range.شروع;
        nrows     = range.شمار;
    }

    إذا (nrows == 0) {
        أرجع HTP_وضعیت_OK;
    }

    إذا (octx->ctx->mdev.شمار <= 1) {
        إذا (htp_tensor_is_contiguous(src0, حجم(عائم)) && htp_tensor_is_contiguous(dst, حجم(عائم))) {
            أرجع roll_dma_f32_contiguous(octx);
        }
        أرجع roll_dma_f32_strided(octx);
    }

    إذا (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        أرجع HTP_وضعیت_NO_SUPPORT;
    }

    ثابت uint32_t n_نخها = octx->n_نخها;
    بنية htp_roll_context rctx = {
        .octx             = octx,
        .ردیف_شروع        = ردیف_شروع,
        .nrows            = nrows,
        .nrows_در_هر_thread = fastdiv(nrows + n_نخها - 1, &octx->n_نخها_div),
        .div_ne1          = init_fastdiv_مقادیر(dst->ne[1]),
        .div_ne2_ne1      = init_fastdiv_مقادیر(dst->ne[2] * dst->ne[1]),
    };

    work_صف_پیاپی(octx->ctx->work_صف, roll_thread_f32, &rctx, n_نخها);

    أرجع HTP_وضعیت_OK;
}

صحيح op_roll(بنية htp_ops_context * octx) {
    بدل (octx->src[0]->type) {
        حالة HTP_TYPE_F32:
            أرجع execute_op_roll_f32(octx);

        افتراضي:
            أرجع HTP_وضعیت_NO_SUPPORT;
    }
}
