#ہدایت clang diagnostic ignored "-Wunused-variable"
#ہدایت clang diagnostic ignored "-Wunused-function"
#ہدایت clang diagnostic ignored "-Wunused-but-set-variable"

#شامل <HAP_farf.h>
#شامل <HAP_perf.h>

#شامل <string.h>
#شامل <math.h>

#شامل "dma-queue.h"
#شامل "hvx-utils.h"

#تعریف_کرو GGML_COMMON_DECL_C
#شامل "ggml-common.h"
#شامل "hex-common.h"
#شامل "hex-profile.h"
#شامل "htp-ctx.h"
#شامل "htp-ops.h"
#شامل "htp-tensor.h"

#تعریف_کرو مجموع_صفیں_preamble                         \
    ثابت ساخت htp_tensor *src0 = octx->src[0]; \
    ثابت ساخت htp_tensor *dst  = octx->dst;    \
                                                  \
    ثابت uint32_t ne00 = src0->ne[0];            \
    ثابت uint32_t ne01 = src0->ne[1];            \
    ثابت uint32_t ne02 = src0->ne[2];            \
    ثابت uint32_t ne03 = src0->ne[3];            \
                                                  \
    ثابت uint32_t nb00 = src0->nb[0];            \
    ثابت uint32_t nb01 = src0->nb[1];            \
    ثابت uint32_t nb02 = src0->nb[2];            \
    ثابت uint32_t nb03 = src0->nb[3];            \
                                                  \
    ثابت uint32_t  ne0 = dst->ne[0];             \
    ثابت uint32_t  ne1 = dst->ne[1];             \
    ثابت uint32_t  ne2 = dst->ne[2];             \
    ثابت uint32_t  ne3 = dst->ne[3];             \
                                                  \
    ثابت uint32_t  nb0 = dst->nb[0];             \
    ثابت uint32_t  nb1 = dst->nb[1];             \
    ثابت uint32_t  nb2 = dst->nb[2];             \
    ثابت uint32_t  nb3 = dst->nb[3];             \

ساخت مجموع_صفیں_context {
    ساخت htp_ops_context * octx;
    ثابت uint8_t * src_ڈیٹا;
    uint8_t       * dst_ڈیٹا;
    uint32_t        ne00;
    size_t          src_stride;
    size_t          dst_stride;
    uint32_t        صفیں_فی_thread;
    uint32_t        کل_صفیں;
    bool            opt_راستہ;
};

ساکن خالی مجموع_صفیں_thread_f32(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی *ڈیٹا) {
    ثابت ساخت مجموع_صفیں_context * smctx = (ثابت ساخت مجموع_صفیں_context *) ڈیٹا;

    ثابت uint32_t صفیں_فی_thread = smctx->صفیں_فی_thread;
    ثابت uint32_t کل_صفیں      = smctx->کل_صفیں;

    ثابت uint32_t آغاز_صف = صفیں_فی_thread * ith;
    ثابت uint32_t انتہا_صف   = MIN(آغاز_صف + صفیں_فی_thread, کل_صفیں);

    اگر (آغاز_صف >= انتہا_صف) {
        واپس;
    }

    ثابت size_t   src_stride = smctx->src_stride;
    ثابت size_t   dst_stride = smctx->dst_stride;
    ثابت uint32_t ne00       = smctx->ne00;
    ثابت bool     opt_راستہ   = smctx->opt_راستہ;

    ثابت اعشاری * محدود src_th = (ثابت اعشاری *) (smctx->src_ڈیٹا + (آغاز_صف * src_stride));
    اعشاری       * محدود dst_th = (اعشاری *)       (smctx->dst_ڈیٹا + (آغاز_صف * dst_stride));

    // Calculate actual number of rows for this thread
    ثابت uint32_t n_صفیں = انتہا_صف - آغاز_صف;

    ساخت htp_thread_trace * tr = &smctx->octx->ctx->trace[ith];
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) آغاز_صف);

    برائے (uint32_t ir = 0; ir < n_صفیں; ir++) {
        ثابت اعشاری * محدود src_local = (ثابت اعشاری *) ((ثابت uint8_t *) src_th + ir * src_stride);
        اعشاری       * محدود dst_local = (اعشاری *)       ((uint8_t *)       dst_th + ir * dst_stride);

        اگر (ir + 1 < n_صفیں) {
            hex_l2fetch((ثابت uint8_t *) src_local + src_stride, src_stride, src_stride, 1);
        }

        اگر (opt_راستہ) {
            *dst_local = hvx_reduce_مجموع_f32_a((ثابت uint8_t *) src_local, ne00);
        } ورنہ {
            *dst_local = hvx_reduce_مجموع_f32((ثابت uint8_t *) src_local, ne00);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) آغاز_صف);
}

صحیح op_مجموع_صفیں(ساخت htp_ops_context * octx) {
    مجموع_صفیں_preamble;

    اگر (octx->src[0]->type != HTP_TYPE_F32) {
        واپس HTP_حالت_NO_SUPPORT;
    }

    اگر (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        واپس HTP_حالت_NO_SUPPORT;
    }

    ثابت uint32_t src0_nrows      = ne01 * ne02 * ne03;
    ثابت size_t dst_ڈیٹا_صف_سائز = dst->ne[0] * حجم(اعشاری);

    uint32_t صف_آغاز = 0;
    uint32_t nrows     = src0_nrows;

    اگر (octx->ctx->mdev.گنتی > 1) {
        uint32_t صفیں_فی_chunk = 0;
        htp_tensor_mdev_صفیں_فی_chunk(dst, حجم(اعشاری), (uint32_t) dst_ڈیٹا_صف_سائز, &صفیں_فی_chunk);
        ثابت ساخت htp_tensor_mdev_range range = htp_tensor_mdev_تقسیم(src0_nrows, صفیں_فی_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
        صف_آغاز = range.آغاز;
        nrows     = range.گنتی;
    }

    اگر (nrows == 0) {
        واپس HTP_حالت_OK;
    }

    ثابت uint32_t n_دھاگے = octx->n_دھاگے;
    ثابت uint32_t صفیں_فی_thread = fastdiv(nrows + n_دھاگے - 1, &octx->n_دھاگے_div);

    bool opt_راستہ = false;
    اگر ((0 == hex_is_aligned((خالی *) src0->ڈیٹا, VLEN)) && !(nb01 & (VLEN - 1))) {
        opt_راستہ = true;
    }

    ساخت مجموع_صفیں_context smctx = {
        .octx            = octx,
        .src_ڈیٹا        = (ثابت uint8_t *) src0->ڈیٹا + صف_آغاز * nb01,
        .dst_ڈیٹا        = (uint8_t *) dst->ڈیٹا + صف_آغاز * nb1,
        .ne00            = ne00,
        .src_stride      = nb01,
        .dst_stride      = nb1,
        .صفیں_فی_thread = صفیں_فی_thread,
        .کل_صفیں      = nrows,
        .opt_راستہ        = opt_راستہ,
    };

    work_قطار_تسلسل(octx->ctx->work_قطار, مجموع_صفیں_thread_f32, &smctx, n_دھاگے);

    واپس HTP_حالت_OK;
}

ساخت مجموع_context {
    ساخت htp_ops_context * octx;
    ثابت اعشاری            * src_ڈیٹا;
    اعشاری                    partial_sums[HTP_MAX_NTHREADS];
    uint32_t                 کل_elems;
    uint32_t                 elems_فی_thread;
};

ساکن خالی مجموع_thread_f32(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    ساخت مجموع_context * sctx = (ساخت مجموع_context *) ڈیٹا;
    ثابت uint32_t آغاز = sctx->elems_فی_thread * ith;
    ثابت uint32_t انتہا   = MIN(آغاز + sctx->elems_فی_thread, sctx->کل_elems);

    اگر (آغاز >= انتہا) {
        sctx->partial_sums[ith] = 0.0f;
        واپس;
    }

    ثابت uint32_t n = انتہا - آغاز;
    ثابت اعشاری * src = sctx->src_ڈیٹا + آغاز;

    ساخت htp_thread_trace * tr = &sctx->octx->ctx->trace[ith];
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) آغاز);

    hex_l2fetch_بلاک((ثابت خالی *) src, n * حجم(اعشاری));

    sctx->partial_sums[ith] = hvx_reduce_مجموع_f32((ثابت uint8_t *) src, n);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) آغاز);
}

صحیح op_مجموع(ساخت htp_ops_context * octx) {
    ثابت ساخت htp_tensor * src0 = octx->src[0];
    ثابت ساخت htp_tensor * dst  = octx->dst;

    اگر (src0->type != HTP_TYPE_F32) {
        واپس HTP_حالت_NO_SUPPORT;
    }

    اگر (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        واپس HTP_حالت_NO_SUPPORT;
    }

    اگر (octx->ctx->mdev.گنتی > 1 && octx->ctx->mdev.idx > 0) {
        واپس HTP_حالت_OK;
    }

    ثابت uint32_t کل_elems = (uint32_t) (src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3]);
    اگر (کل_elems == 0) {
        ((اعشاری *) dst->ڈیٹا)[0] = 0.0f;
        واپس HTP_حالت_OK;
    }

    ثابت uint32_t n_دھاگے = (کل_elems >= 1024) ? MIN(octx->n_دھاگے, HTP_MAX_NTHREADS) : 1;
    ثابت uint32_t raw_chunk = (کل_elems + n_دھاگے - 1) / n_دھاگے;
    ثابت uint32_t elems_فی_thread = hex_باری_اوپر(raw_chunk, 32);

    ساخت مجموع_context sctx = {
        .octx             = octx,
        .src_ڈیٹا         = (ثابت اعشاری *) src0->ڈیٹا,
        .کل_elems      = کل_elems,
        .elems_فی_thread = elems_فی_thread,
    };

    work_قطار_تسلسل(octx->ctx->work_قطار, مجموع_thread_f32, &sctx, n_دھاگے);

    اعشاری مجموع = 0.0f;
    برائے (uint32_t i = 0; i < n_دھاگے; i++) {
        مجموع += sctx.partial_sums[i];
    }
    ((اعشاری *) dst->ڈیٹا)[0] = مجموع;

    واپس HTP_حالت_OK;
}

ساکن اندرونی خالی argmax_قاش_f32(
    ثابت اعشاری * محدود src,
    uint32_t n,
    uint32_t انحراف,
    اعشاری * برآمد_val,
    int32_t * برآمد_idx
) {
    hvx_argmax_f32(src, n, انحراف, برآمد_val, برآمد_idx);
}

ساخت argmax_context {
    ساخت htp_ops_context * octx;
    ثابت اعشاری            * src_ڈیٹا;
    int32_t                * dst_ڈیٹا;
    uint32_t                 ne00;
    uint32_t                 src_stride;
    uint32_t                 dst_stride;
    uint32_t                 صف_آغاز;
    uint32_t                 nrows;
    uint32_t                 صفیں_فی_thread;
    uint32_t                 elems_فی_thread;

    اعشاری                    partial_max[HTP_MAX_NTHREADS];
    int32_t                  partial_idx[HTP_MAX_NTHREADS];
};

ساکن خالی argmax_thread_single_صف(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    ساخت argmax_context * actx = (ساخت argmax_context *) ڈیٹا;
    ثابت uint32_t آغاز = actx->elems_فی_thread * ith;
    ثابت uint32_t انتہا   = MIN(آغاز + actx->elems_فی_thread, actx->ne00);

    اگر (آغاز >= انتہا) {
        actx->partial_max[ith] = -INFINITY;
        actx->partial_idx[ith] = 0;
        واپس;
    }

    ثابت uint32_t n = انتہا - آغاز;
    ثابت اعشاری * src = actx->src_ڈیٹا + آغاز;

    ساخت htp_thread_trace * tr = &actx->octx->ctx->trace[ith];
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) آغاز);

    hex_l2fetch_بلاک((ثابت خالی *) src, n * حجم(اعشاری));

    اعشاری max_val;
    int32_t max_idx;
    argmax_قاش_f32(src, n, آغاز, &max_val, &max_idx);

    actx->partial_max[ith] = max_val;
    actx->partial_idx[ith] = max_idx;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) آغاز);
}

ساکن خالی argmax_thread_multi_صف(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    ساخت argmax_context * actx = (ساخت argmax_context *) ڈیٹا;
    ثابت uint32_t r0 = actx->صف_آغاز + actx->صفیں_فی_thread * ith;
    ثابت uint32_t r1 = MIN(r0 + actx->صفیں_فی_thread, actx->صف_آغاز + actx->nrows);

    اگر (r0 >= r1) {
        واپس;
    }

    ساخت htp_thread_trace * tr = &actx->octx->ctx->trace[ith];
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r0);

    برائے (uint32_t r = r0; r < r1; r++) {
        ثابت اعشاری * src_صف = (ثابت اعشاری *) ((ثابت uint8_t *) actx->src_ڈیٹا + r * actx->src_stride);
        int32_t * dst_val     = (int32_t *) ((uint8_t *) actx->dst_ڈیٹا + r * actx->dst_stride);

        hex_l2fetch_بلاک((ثابت خالی *) src_صف, actx->ne00 * حجم(اعشاری));

        اعشاری max_val;
        int32_t max_idx;
        argmax_قاش_f32(src_صف, actx->ne00, 0, &max_val, &max_idx);

        *dst_val = max_idx;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r0);
}

صحیح op_argmax(ساخت htp_ops_context * octx) {
    ثابت ساخت htp_tensor * src0 = octx->src[0];
    ثابت ساخت htp_tensor * dst  = octx->dst;

    اگر (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_I32) {
        واپس HTP_حالت_NO_SUPPORT;
    }

    اگر (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        واپس HTP_حالت_NO_SUPPORT;
    }

    ثابت uint32_t ne00 = src0->ne[0];
    ثابت uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    اگر (ne00 == 0 || src0_nrows == 0) {
        واپس HTP_حالت_OK;
    }

    اگر (src0_nrows == 1) {
        اگر (octx->ctx->mdev.گنتی > 1 && octx->ctx->mdev.idx > 0) {
            واپس HTP_حالت_OK;
        }

        اگر (ne00 == 1) {
            ((int32_t *) dst->ڈیٹا)[0] = 0;
            واپس HTP_حالت_OK;
        }

        ثابت uint32_t n_دھاگے = (ne00 >= 1024) ? MIN(octx->n_دھاگے, HTP_MAX_NTHREADS) : 1;
        ثابت uint32_t raw_chunk = (ne00 + n_دھاگے - 1) / n_دھاگے;
        ثابت uint32_t elems_فی_thread = hex_باری_اوپر(raw_chunk, 32);

        ساخت argmax_context actx = {
            .octx             = octx,
            .src_ڈیٹا         = (ثابت اعشاری *) src0->ڈیٹا,
            .dst_ڈیٹا         = (int32_t *) dst->ڈیٹا,
            .ne00             = ne00,
            .src_stride       = src0->nb[1] > 0 ? (uint32_t) src0->nb[1] : (uint32_t) (ne00 * حجم(اعشاری)),
            .dst_stride       = dst->nb[0] > 0 ? (uint32_t) dst->nb[0] : (uint32_t) حجم(int32_t),
            .صف_آغاز        = 0,
            .nrows            = 1,
            .صفیں_فی_thread  = 1,
            .elems_فی_thread = elems_فی_thread,
        };

        work_قطار_تسلسل(octx->ctx->work_قطار, argmax_thread_single_صف, &actx, n_دھاگے);

        اعشاری بہترین_val = actx.partial_max[0];
        int32_t بہترین_idx = actx.partial_idx[0];
        برائے (uint32_t i = 1; i < n_دھاگے; i++) {
            اگر (actx.partial_max[i] > بہترین_val) {
                بہترین_val = actx.partial_max[i];
                بہترین_idx = actx.partial_idx[i];
            }
        }
        ((int32_t *) dst->ڈیٹا)[0] = بہترین_idx;
        واپس HTP_حالت_OK;
    }

    uint32_t صف_آغاز = 0;
    uint32_t nrows     = src0_nrows;

    اگر (octx->ctx->mdev.گنتی > 1) {
        ثابت bool can_بانٹو = htp_tensor_mdev_ڈیٹا_aligned(dst) &&
                               htp_tensor_is_contiguous(dst, حجم(int32_t));
        ثابت uint32_t elems_فی_chunk = can_بانٹو ? 32 : 0;
        ثابت ساخت htp_tensor_mdev_range range = htp_tensor_mdev_تقسیم(
            src0_nrows, elems_فی_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
        صف_آغاز = range.آغاز;
        nrows     = range.گنتی;
    }

    اگر (nrows == 0) {
        واپس HTP_حالت_OK;
    }

    ثابت uint32_t n_دھاگے = MIN(octx->n_دھاگے, nrows);
    ثابت uint32_t صفیں_فی_thread = (nrows + n_دھاگے - 1) / n_دھاگے;

    ساخت argmax_context actx = {
        .octx             = octx,
        .src_ڈیٹا         = (ثابت اعشاری *) src0->ڈیٹا,
        .dst_ڈیٹا         = (int32_t *) dst->ڈیٹا,
        .ne00             = ne00,
        .src_stride       = src0->nb[1] > 0 ? (uint32_t) src0->nb[1] : (uint32_t) (ne00 * حجم(اعشاری)),
        .dst_stride       = dst->nb[0] > 0 ? (uint32_t) dst->nb[0] : (uint32_t) حجم(int32_t),
        .صف_آغاز        = صف_آغاز,
        .nrows            = nrows,
        .صفیں_فی_thread  = صفیں_فی_thread,
        .elems_فی_thread = 0,
    };

    work_قطار_تسلسل(octx->ctx->work_قطار, argmax_thread_multi_صف, &actx, n_دھاگے);
    واپس HTP_حالت_OK;
}
