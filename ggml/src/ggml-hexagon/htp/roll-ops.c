#ہدایت clang diagnostic ignored "-Wunused-variable"
#ہدایت clang diagnostic ignored "-Wunused-function"
#ہدایت clang diagnostic ignored "-Wunused-but-set-variable"

#شامل <HAP_farf.h>
#شامل <HAP_perf.h>

#شامل <string.h>

#شامل "dma-queue.h"
#شامل "hvx-utils.h"

#تعریف_کرو GGML_COMMON_DECL_C
#شامل "ggml-common.h"
#شامل "htp-ctx.h"
#شامل "hex-common.h"
#شامل "hex-profile.h"
#شامل "htp-ops.h"
#شامل "htp-tensor.h"

ساخت htp_roll_context {
    ساخت htp_ops_context * octx;

    uint32_t صف_آغاز;
    uint32_t nrows;
    uint32_t nrows_فی_thread;

    ساخت fastdiv_اقدار div_ne1;
    ساخت fastdiv_اقدار div_ne2_ne1;
};

ساکن اندرونی uint32_t htp_roll_wrap(int32_t i, uint32_t ne) {
    اگر (i < 0) {
        واپس (uint32_t) (i + (int32_t) ne);
    }
    اگر ((uint32_t) i >= ne) {
        واپس (uint32_t) i - ne;
    }
    واپس (uint32_t) i;
}

#تعریف_کرو htp_roll_preamble                             \
    ثابت ساخت htp_tensor * src0 = octx->src[0];    \
    ثابت ساخت htp_tensor * dst  = octx->dst;       \
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

ساکن اندرونی خالی roll_dma_ڈالو(dma_قطار * q,
                                 dma_addr_t  dst,
                                 dma_addr_t  src,
                                 uint32_t    dst_stride,
                                 uint32_t    src_stride,
                                 uint32_t    بائٹیں,
                                 uint32_t    nrows) {
    اگر (بائٹیں == 0 || nrows == 0) {
        واپس;
    }

    اگر (!dma_قطار_ڈالو(q, dma_make_ڈیٹا(dst, src), dst_stride, src_stride, بائٹیں, nrows)) {
        dma_قطار_flush(q);
        dma_قطار_ڈالو(q, dma_make_ڈیٹا(dst, src),
                       dst_stride, src_stride, بائٹیں, nrows);
    }
}

ساکن اندرونی خالی roll_dma_ڈالو_صفیں(dma_قطار *               q,
                                      ثابت ساخت htp_tensor * dst,
                                      ثابت ساخت htp_tensor * src0,
                                      uint32_t                  dst_صف,
                                      uint32_t                  src_صف,
                                      uint32_t                  nrows,
                                      uint32_t                  صف_سائز,
                                      uint32_t                  i0_src0) {
    ثابت dma_addr_t dst_بنیاد = dst->ڈیٹا + (size_t) dst_صف * صف_سائز;
    ثابت dma_addr_t src_بنیاد = src0->ڈیٹا + (size_t) src_صف * صف_سائز;
    ثابت uint32_t   n0       = src0->ne[0] - i0_src0;

    roll_dma_ڈالو(q, dst_بنیاد, src_بنیاد + (size_t) i0_src0 * حجم(اعشاری),
                  صف_سائز, صف_سائز, n0 * حجم(اعشاری), nrows);
    roll_dma_ڈالو(q, dst_بنیاد + (size_t) n0 * حجم(اعشاری), src_بنیاد,
                  صف_سائز, صف_سائز, i0_src0 * حجم(اعشاری), nrows);
}

// Same row-wrap split as roll_dma_push_rows, but addressed with explicit byte strides so it
// also works for a src0 that is row-contiguous only (e.g. a permuted view) rather than fully packed.
ساکن اندرونی خالی roll_dma_ڈالو_range(dma_قطار * q,
                                       dma_addr_t  dst_صف,
                                       dma_addr_t  src_صف,
                                       uint32_t    dst_stride,
                                       uint32_t    src_stride,
                                       uint32_t    nrows,
                                       uint32_t    i0_src0,
                                       uint32_t    n0) {
    roll_dma_ڈالو(q, dst_صف, src_صف + (size_t) i0_src0 * حجم(اعشاری),
                  dst_stride, src_stride, n0 * حجم(اعشاری), nrows);
    roll_dma_ڈالو(q, dst_صف + (size_t) n0 * حجم(اعشاری), src_صف,
                  dst_stride, src_stride, i0_src0 * حجم(اعشاری), nrows);
}

ساکن صحیح roll_dma_f32_contiguous(ساخت htp_ops_context * octx) {
    htp_roll_preamble;
    htp_roll_dma_preamble;

    ثابت uint32_t صف_سائز = ne0 * حجم(اعشاری);

    اگر (s1 == 0 && s2 == 0 && s3 == 0) {
        roll_dma_ڈالو_صفیں(q, dst, src0, 0, 0, ne1 * ne2 * ne3, صف_سائز, i0_src0);
        dma_قطار_flush(q);
        واپس HTP_حالت_OK;
    }

    اگر (s1 == 0) {
        ثابت uint32_t i2_src0 = htp_roll_wrap(-s2, ne2);
        برائے (uint32_t i3 = 0; i3 < ne3; i3++) {
            ثابت uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
            ثابت uint32_t dst_row0 = i3 * ne2 * ne1;
            ثابت uint32_t src_row0 = (i03 * ne2 + i2_src0) * ne1;
            ثابت uint32_t n2_پہلا = ne2 - i2_src0;

            roll_dma_ڈالو_صفیں(q, dst, src0, dst_row0, src_row0, n2_پہلا * ne1,
                               صف_سائز, i0_src0);
            roll_dma_ڈالو_صفیں(q, dst, src0, dst_row0 + n2_پہلا * ne1, i03 * ne2 * ne1,
                               i2_src0 * ne1, صف_سائز, i0_src0);
        }

        dma_قطار_flush(q);
        واپس HTP_حالت_OK;
    }

    ثابت uint32_t i1_src0 = htp_roll_wrap(-s1, ne1);
    ثابت uint32_t n1_پہلا = ne1 - i1_src0;

    برائے (uint32_t i3 = 0; i3 < ne3; i3++) {
        ثابت uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
        برائے (uint32_t i2 = 0; i2 < ne2; i2++) {
            ثابت uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);
            ثابت uint32_t dst_row0 = (i3 * ne2 + i2) * ne1;
            ثابت uint32_t src_row0 = (i03 * ne2 + i02) * ne1;

            roll_dma_ڈالو_صفیں(q, dst, src0, dst_row0, src_row0 + i1_src0,
                               n1_پہلا, صف_سائز, i0_src0);
            roll_dma_ڈالو_صفیں(q, dst, src0, dst_row0 + n1_پہلا, src_row0,
                               i1_src0, صف_سائز, i0_src0);
        }
    }

    dma_قطار_flush(q);
    واپس HTP_حالت_OK;
}

// DMA path for a row-contiguous but otherwise arbitrarily strided src0 (e.g. a permuted view).
// Same row-wrap split as above, one DMA push per (i2,i3), addressed via the real nb01/nb02/nb03
// instead of assuming a packed layout.
ساکن صحیح roll_dma_f32_strided(ساخت htp_ops_context * octx) {
    htp_roll_preamble;
    htp_roll_dma_preamble;

    ثابت uint32_t i1_src0  = htp_roll_wrap(-s1, ne1);
    ثابت uint32_t n1_پہلا = ne1 - i1_src0;

    برائے (uint32_t i3 = 0; i3 < ne3; i3++) {
        ثابت uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
        برائے (uint32_t i2 = 0; i2 < ne2; i2++) {
            ثابت uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);

            ثابت dma_addr_t dst_row0 = dst->ڈیٹا  + (size_t) i2  * nb2  + (size_t) i3  * nb3;
            ثابت dma_addr_t src_row0 = src0->ڈیٹا + (size_t) i02 * nb02 + (size_t) i03 * nb03;

            roll_dma_ڈالو_range(q, dst_row0, src_row0 + (size_t) i1_src0 * nb01,
                                nb1, nb01, n1_پہلا, i0_src0, n0);
            roll_dma_ڈالو_range(q, dst_row0 + (size_t) n1_پہلا * nb1, src_row0,
                                nb1, nb01, i1_src0, i0_src0, n0);
        }
    }

    dma_قطار_flush(q);
    واپس HTP_حالت_OK;
}

ساکن خالی roll_thread_f32(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    ساخت htp_roll_context * rctx = (ساخت htp_roll_context *) ڈیٹا;
    ساخت htp_ops_context * octx = rctx->octx;

    htp_roll_preamble;

    ثابت uint32_t صف_آغاز = rctx->صف_آغاز + rctx->nrows_فی_thread * ith;
    ثابت uint32_t صف_انتہا   = MIN(صف_آغاز + rctx->nrows_فی_thread, rctx->صف_آغاز + rctx->nrows);
    اگر (صف_آغاز >= صف_انتہا) {
        واپس;
    }

    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, صف_آغاز);

    برائے (uint32_t صف = صف_آغاز; صف < صف_انتہا; صف++) {
        ثابت uint32_t i3  = fastdiv(صف, &rctx->div_ne2_ne1);
        ثابت uint32_t rem = صف - i3 * ne2 * ne1;
        ثابت uint32_t i2  = fastdiv(rem, &rctx->div_ne1);
        ثابت uint32_t i1  = rem - i2 * ne1;

        ثابت uint32_t i01 = htp_roll_wrap((int32_t) i1 - s1, ne1);
        ثابت uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);
        ثابت uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);

        ثابت uint8_t * src_صف = (ثابت uint8_t *) (uintptr_t) src0->ڈیٹا + i01*nb01 + i02*nb02 + i03*nb03;
        uint8_t * dst_صف = (uint8_t *) (uintptr_t) dst->ڈیٹا + i1*nb1 + i2*nb2 + i3*nb3;

        hex_l2fetch(src_صف + i0_src0 * حجم(اعشاری), n0 * حجم(اعشاری), ne0 * حجم(اعشاری), 1);
        hvx_copy_uu(dst_صف, src_صف + i0_src0 * حجم(اعشاری), n0, حجم(اعشاری));

        اگر (i0_src0 != 0) {
            hex_l2fetch(src_صف, i0_src0 * حجم(اعشاری), ne0 * حجم(اعشاری), 1);
            hvx_copy_uu(dst_صف + n0 * حجم(اعشاری), src_صف, i0_src0, حجم(اعشاری));
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, صف_آغاز);

    FARF(بلند_2, "roll %d/%d: (%ux%ux%ux%u) rows %u:%u shift=(%d,%d,%d,%d)\n",
         ith, nth, ne0, ne1, ne2, ne3,
         صف_آغاز, صف_انتہا, s0, s1, s2, s3);
}

صحیح execute_op_roll_f32(ساخت htp_ops_context * octx) {
    htp_roll_preamble;

    اگر (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        FARF(ERROR, "roll: unsupported type %u -> %u\n", src0->type, dst->type);
        واپس HTP_حالت_NO_SUPPORT;
    }

    اگر (src0->nb[0] != حجم(اعشاری) || dst->nb[0] != حجم(اعشاری)) {
        FARF(ERROR, "roll: unsupported nb0 %u -> %u\n", src0->nb[0], dst->nb[0]);
        واپس HTP_حالت_NO_SUPPORT;
    }

    اگر (src0->ne[0] != ne0 || src0->ne[1] != ne1 ||
        src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        FARF(ERROR, "roll: shape mismatch\n");
        واپس HTP_حالت_INVAL_PARAMS;
    }

    ثابت uint32_t کل_صفیں = ne1 * ne2 * ne3;
    ثابت size_t dst_صف_سائز = ne0 * حجم(اعشاری);

    uint32_t صف_آغاز = 0;
    uint32_t nrows     = کل_صفیں;

    اگر (octx->ctx->mdev.گنتی > 1) {
        uint32_t صفیں_فی_chunk = 0;
        htp_tensor_mdev_صفیں_فی_chunk(dst, حجم(اعشاری), (uint32_t) dst_صف_سائز, &صفیں_فی_chunk);
        ثابت ساخت htp_tensor_mdev_range range = htp_tensor_mdev_تقسیم(کل_صفیں, صفیں_فی_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
        صف_آغاز = range.آغاز;
        nrows     = range.گنتی;
    }

    اگر (nrows == 0) {
        واپس HTP_حالت_OK;
    }

    اگر (octx->ctx->mdev.گنتی <= 1) {
        اگر (htp_tensor_is_contiguous(src0, حجم(اعشاری)) && htp_tensor_is_contiguous(dst, حجم(اعشاری))) {
            واپس roll_dma_f32_contiguous(octx);
        }
        واپس roll_dma_f32_strided(octx);
    }

    اگر (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        واپس HTP_حالت_NO_SUPPORT;
    }

    ثابت uint32_t n_دھاگے = octx->n_دھاگے;
    ساخت htp_roll_context rctx = {
        .octx             = octx,
        .صف_آغاز        = صف_آغاز,
        .nrows            = nrows,
        .nrows_فی_thread = fastdiv(nrows + n_دھاگے - 1, &octx->n_دھاگے_div),
        .div_ne1          = init_fastdiv_اقدار(dst->ne[1]),
        .div_ne2_ne1      = init_fastdiv_اقدار(dst->ne[2] * dst->ne[1]),
    };

    work_قطار_تسلسل(octx->ctx->work_قطار, roll_thread_f32, &rctx, n_دھاگے);

    واپس HTP_حالت_OK;
}

صحیح op_roll(ساخت htp_ops_context * octx) {
    انتخاب (octx->src[0]->type) {
        حالت HTP_TYPE_F32:
            واپس execute_op_roll_f32(octx);

        طےشدہ:
            واپس HTP_حالت_NO_SUPPORT;
    }
}
