#ہدایت clang diagnostic ignored "-Wunused-variable"
#ہدایت clang diagnostic ignored "-Wunused-function"
#ہدایت clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <HAP_perf.h>

#ضمن <string.h>

#ضمن "dma-queue.h"
#ضمن "hvx-utils.h"

#تعریف_کرو GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "htp-ctx.h"
#ضمن "hex-common.h"
#ضمن "hex-profile.h"
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"

بنية htp_roll_context {
    بنية htp_ops_context * octx;

    uint32_t صف_شروع;
    uint32_t nrows;
    uint32_t nrows_فی_thread;

    بنية fastdiv_اقدار div_ne1;
    بنية fastdiv_اقدار div_ne2_ne1;
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

#تعریف_کرو htp_roll_preamble                             \
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

#تعریف_کرو htp_roll_dma_preamble dma_قطار * q = octx->ctx->dma[0];

ساكن مضمن فراغ roll_dma_ڈالو(dma_قطار * q,
                                 dma_addr_t  dst,
                                 dma_addr_t  src,
                                 uint32_t    dst_stride,
                                 uint32_t    src_stride,
                                 uint32_t    بائٹاں,
                                 uint32_t    nrows) {
    إذا (بائٹاں == 0 || nrows == 0) {
        أرجع;
    }

    إذا (!dma_قطار_ڈالو(q, dma_make_انکڑے(dst, src), dst_stride, src_stride, بائٹاں, nrows)) {
        dma_قطار_flush(q);
        dma_قطار_ڈالو(q, dma_make_انکڑے(dst, src),
                       dst_stride, src_stride, بائٹاں, nrows);
    }
}

ساكن مضمن فراغ roll_dma_ڈالو_صفاں(dma_قطار *               q,
                                      ثابت بنية htp_tensor * dst,
                                      ثابت بنية htp_tensor * src0,
                                      uint32_t                  dst_صف,
                                      uint32_t                  src_صف,
                                      uint32_t                  nrows,
                                      uint32_t                  صف_ناپ,
                                      uint32_t                  i0_src0) {
    ثابت dma_addr_t dst_ادھار = dst->انکڑے + (size_t) dst_صف * صف_ناپ;
    ثابت dma_addr_t src_ادھار = src0->انکڑے + (size_t) src_صف * صف_ناپ;
    ثابت uint32_t   n0       = src0->ne[0] - i0_src0;

    roll_dma_ڈالو(q, dst_ادھار, src_ادھار + (size_t) i0_src0 * حجم(عائم),
                  صف_ناپ, صف_ناپ, n0 * حجم(عائم), nrows);
    roll_dma_ڈالو(q, dst_ادھار + (size_t) n0 * حجم(عائم), src_ادھار,
                  صف_ناپ, صف_ناپ, i0_src0 * حجم(عائم), nrows);
}

// Same row-wrap split as roll_dma_push_rows, but addressed with explicit byte strides so it
// also works for a src0 that is row-contiguous only (e.g. a permuted view) rather than fully packed.
ساكن مضمن فراغ roll_dma_ڈالو_range(dma_قطار * q,
                                       dma_addr_t  dst_صف,
                                       dma_addr_t  src_صف,
                                       uint32_t    dst_stride,
                                       uint32_t    src_stride,
                                       uint32_t    nrows,
                                       uint32_t    i0_src0,
                                       uint32_t    n0) {
    roll_dma_ڈالو(q, dst_صف, src_صف + (size_t) i0_src0 * حجم(عائم),
                  dst_stride, src_stride, n0 * حجم(عائم), nrows);
    roll_dma_ڈالو(q, dst_صف + (size_t) n0 * حجم(عائم), src_صف,
                  dst_stride, src_stride, i0_src0 * حجم(عائم), nrows);
}

ساكن صحيح roll_dma_f32_contiguous(بنية htp_ops_context * octx) {
    htp_roll_preamble;
    htp_roll_dma_preamble;

    ثابت uint32_t صف_ناپ = ne0 * حجم(عائم);

    إذا (s1 == 0 && s2 == 0 && s3 == 0) {
        roll_dma_ڈالو_صفاں(q, dst, src0, 0, 0, ne1 * ne2 * ne3, صف_ناپ, i0_src0);
        dma_قطار_flush(q);
        أرجع HTP_حالت_OK;
    }

    إذا (s1 == 0) {
        ثابت uint32_t i2_src0 = htp_roll_wrap(-s2, ne2);
        لكل (uint32_t i3 = 0; i3 < ne3; i3++) {
            ثابت uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
            ثابت uint32_t dst_row0 = i3 * ne2 * ne1;
            ثابت uint32_t src_row0 = (i03 * ne2 + i2_src0) * ne1;
            ثابت uint32_t n2_پہلا = ne2 - i2_src0;

            roll_dma_ڈالو_صفاں(q, dst, src0, dst_row0, src_row0, n2_پہلا * ne1,
                               صف_ناپ, i0_src0);
            roll_dma_ڈالو_صفاں(q, dst, src0, dst_row0 + n2_پہلا * ne1, i03 * ne2 * ne1,
                               i2_src0 * ne1, صف_ناپ, i0_src0);
        }

        dma_قطار_flush(q);
        أرجع HTP_حالت_OK;
    }

    ثابت uint32_t i1_src0 = htp_roll_wrap(-s1, ne1);
    ثابت uint32_t n1_پہلا = ne1 - i1_src0;

    لكل (uint32_t i3 = 0; i3 < ne3; i3++) {
        ثابت uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
        لكل (uint32_t i2 = 0; i2 < ne2; i2++) {
            ثابت uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);
            ثابت uint32_t dst_row0 = (i3 * ne2 + i2) * ne1;
            ثابت uint32_t src_row0 = (i03 * ne2 + i02) * ne1;

            roll_dma_ڈالو_صفاں(q, dst, src0, dst_row0, src_row0 + i1_src0,
                               n1_پہلا, صف_ناپ, i0_src0);
            roll_dma_ڈالو_صفاں(q, dst, src0, dst_row0 + n1_پہلا, src_row0,
                               i1_src0, صف_ناپ, i0_src0);
        }
    }

    dma_قطار_flush(q);
    أرجع HTP_حالت_OK;
}

// DMA path for a row-contiguous but otherwise arbitrarily strided src0 (e.g. a permuted view).
// Same row-wrap split as above, one DMA push per (i2,i3), addressed via the real nb01/nb02/nb03
// instead of assuming a packed layout.
ساكن صحيح roll_dma_f32_strided(بنية htp_ops_context * octx) {
    htp_roll_preamble;
    htp_roll_dma_preamble;

    ثابت uint32_t i1_src0  = htp_roll_wrap(-s1, ne1);
    ثابت uint32_t n1_پہلا = ne1 - i1_src0;

    لكل (uint32_t i3 = 0; i3 < ne3; i3++) {
        ثابت uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
        لكل (uint32_t i2 = 0; i2 < ne2; i2++) {
            ثابت uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);

            ثابت dma_addr_t dst_row0 = dst->انکڑے  + (size_t) i2  * nb2  + (size_t) i3  * nb3;
            ثابت dma_addr_t src_row0 = src0->انکڑے + (size_t) i02 * nb02 + (size_t) i03 * nb03;

            roll_dma_ڈالو_range(q, dst_row0, src_row0 + (size_t) i1_src0 * nb01,
                                nb1, nb01, n1_پہلا, i0_src0, n0);
            roll_dma_ڈالو_range(q, dst_row0 + (size_t) n1_پہلا * nb1, src_row0,
                                nb1, nb01, i1_src0, i0_src0, n0);
        }
    }

    dma_قطار_flush(q);
    أرجع HTP_حالت_OK;
}

ساكن فراغ roll_thread_f32(بنانشانی صحيح nth, بنانشانی صحيح ith, فراغ * انکڑے) {
    بنية htp_roll_context * rctx = (بنية htp_roll_context *) انکڑے;
    بنية htp_ops_context * octx = rctx->octx;

    htp_roll_preamble;

    ثابت uint32_t صف_شروع = rctx->صف_شروع + rctx->nrows_فی_thread * ith;
    ثابت uint32_t صف_سرا   = MIN(صف_شروع + rctx->nrows_فی_thread, rctx->صف_شروع + rctx->nrows);
    إذا (صف_شروع >= صف_سرا) {
        أرجع;
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, صف_شروع);

    لكل (uint32_t صف = صف_شروع; صف < صف_سرا; صف++) {
        ثابت uint32_t i3  = fastdiv(صف, &rctx->div_ne2_ne1);
        ثابت uint32_t rem = صف - i3 * ne2 * ne1;
        ثابت uint32_t i2  = fastdiv(rem, &rctx->div_ne1);
        ثابت uint32_t i1  = rem - i2 * ne1;

        ثابت uint32_t i01 = htp_roll_wrap((int32_t) i1 - s1, ne1);
        ثابت uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);
        ثابت uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);

        ثابت uint8_t * src_صف = (ثابت uint8_t *) (uintptr_t) src0->انکڑے + i01*nb01 + i02*nb02 + i03*nb03;
        uint8_t * dst_صف = (uint8_t *) (uintptr_t) dst->انکڑے + i1*nb1 + i2*nb2 + i3*nb3;

        hex_l2fetch(src_صف + i0_src0 * حجم(عائم), n0 * حجم(عائم), ne0 * حجم(عائم), 1);
        hvx_copy_uu(dst_صف, src_صف + i0_src0 * حجم(عائم), n0, حجم(عائم));

        إذا (i0_src0 != 0) {
            hex_l2fetch(src_صف, i0_src0 * حجم(عائم), ne0 * حجم(عائم), 1);
            hvx_copy_uu(dst_صف + n0 * حجم(عائم), src_صف, i0_src0, حجم(عائم));
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, صف_شروع);

    FARF(بلند_2, "roll %d/%d: (%ux%ux%ux%u) rows %u:%u shift=(%d,%d,%d,%d)\n",
         ith, nth, ne0, ne1, ne2, ne3,
         صف_شروع, صف_سرا, s0, s1, s2, s3);
}

صحيح execute_op_roll_f32(بنية htp_ops_context * octx) {
    htp_roll_preamble;

    إذا (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        FARF(ERROR, "roll: unsupported type %u -> %u\n", src0->type, dst->type);
        أرجع HTP_حالت_NO_SUPPORT;
    }

    إذا (src0->nb[0] != حجم(عائم) || dst->nb[0] != حجم(عائم)) {
        FARF(ERROR, "roll: unsupported nb0 %u -> %u\n", src0->nb[0], dst->nb[0]);
        أرجع HTP_حالت_NO_SUPPORT;
    }

    إذا (src0->ne[0] != ne0 || src0->ne[1] != ne1 ||
        src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        FARF(ERROR, "roll: shape mismatch\n");
        أرجع HTP_حالت_INVAL_PARAMS;
    }

    ثابت uint32_t کل_صفاں = ne1 * ne2 * ne3;
    ثابت size_t dst_صف_ناپ = ne0 * حجم(عائم);

    uint32_t صف_شروع = 0;
    uint32_t nrows     = کل_صفاں;

    إذا (octx->ctx->mdev.گنتی > 1) {
        uint32_t صفاں_فی_chunk = 0;
        htp_tensor_mdev_صفاں_فی_chunk(dst, حجم(عائم), (uint32_t) dst_صف_ناپ, &صفاں_فی_chunk);
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصے_کرو(کل_صفاں, صفاں_فی_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
        صف_شروع = range.شروع;
        nrows     = range.گنتی;
    }

    إذا (nrows == 0) {
        أرجع HTP_حالت_OK;
    }

    إذا (octx->ctx->mdev.گنتی <= 1) {
        إذا (htp_tensor_is_contiguous(src0, حجم(عائم)) && htp_tensor_is_contiguous(dst, حجم(عائم))) {
            أرجع roll_dma_f32_contiguous(octx);
        }
        أرجع roll_dma_f32_strided(octx);
    }

    إذا (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    ثابت uint32_t n_دھاگے = octx->n_دھاگے;
    بنية htp_roll_context rctx = {
        .octx             = octx,
        .صف_شروع        = صف_شروع,
        .nrows            = nrows,
        .nrows_فی_thread = fastdiv(nrows + n_دھاگے - 1, &octx->n_دھاگے_div),
        .div_ne1          = init_fastdiv_اقدار(dst->ne[1]),
        .div_ne2_ne1      = init_fastdiv_اقدار(dst->ne[2] * dst->ne[1]),
    };

    work_قطار_لگاتار(octx->ctx->work_قطار, roll_thread_f32, &rctx, n_دھاگے);

    أرجع HTP_حالت_OK;
}

صحيح op_roll(بنية htp_ops_context * octx) {
    بدل (octx->src[0]->type) {
        حالة HTP_TYPE_F32:
            أرجع execute_op_roll_f32(octx);

        افتراضي:
            أرجع HTP_حالت_NO_SUPPORT;
    }
}
