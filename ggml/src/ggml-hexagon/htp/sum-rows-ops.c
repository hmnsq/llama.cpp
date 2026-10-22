#ہدایت clang diagnostic ignored "-Wunused-variable"
#ہدایت clang diagnostic ignored "-Wunused-function"
#ہدایت clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <HAP_perf.h>

#ضمن <string.h>
#ضمن <math.h>

#ضمن "dma-queue.h"
#ضمن "hvx-utils.h"

#تعریف_کرو GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "hex-common.h"
#ضمن "hex-profile.h"
#ضمن "htp-ctx.h"
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"

#تعریف_کرو جوڑ_صفاں_preamble                         \
    ثابت بنية htp_tensor *src0 = octx->src[0]; \
    ثابت بنية htp_tensor *dst  = octx->dst;    \
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

بنية جوڑ_صفاں_context {
    بنية htp_ops_context * octx;
    ثابت uint8_t * src_انکڑے;
    uint8_t       * dst_انکڑے;
    uint32_t        ne00;
    size_t          src_stride;
    size_t          dst_stride;
    uint32_t        صفاں_فی_thread;
    uint32_t        کل_صفاں;
    bool            opt_رستہ;
};

ساكن فراغ جوڑ_صفاں_thread_f32(بنانشانی صحيح nth, بنانشانی صحيح ith, فراغ *انکڑے) {
    ثابت بنية جوڑ_صفاں_context * smctx = (ثابت بنية جوڑ_صفاں_context *) انکڑے;

    ثابت uint32_t صفاں_فی_thread = smctx->صفاں_فی_thread;
    ثابت uint32_t کل_صفاں      = smctx->کل_صفاں;

    ثابت uint32_t شروع_صف = صفاں_فی_thread * ith;
    ثابت uint32_t سرا_صف   = MIN(شروع_صف + صفاں_فی_thread, کل_صفاں);

    إذا (شروع_صف >= سرا_صف) {
        أرجع;
    }

    ثابت size_t   src_stride = smctx->src_stride;
    ثابت size_t   dst_stride = smctx->dst_stride;
    ثابت uint32_t ne00       = smctx->ne00;
    ثابت bool     opt_رستہ   = smctx->opt_رستہ;

    ثابت عائم * محدود src_th = (ثابت عائم *) (smctx->src_انکڑے + (شروع_صف * src_stride));
    عائم       * محدود dst_th = (عائم *)       (smctx->dst_انکڑے + (شروع_صف * dst_stride));

    // Calculate actual number of rows for this thread
    ثابت uint32_t n_صفاں = سرا_صف - شروع_صف;

    بنية htp_thread_trace * tr = &smctx->octx->ctx->trace[ith];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) شروع_صف);

    لكل (uint32_t ir = 0; ir < n_صفاں; ir++) {
        ثابت عائم * محدود src_local = (ثابت عائم *) ((ثابت uint8_t *) src_th + ir * src_stride);
        عائم       * محدود dst_local = (عائم *)       ((uint8_t *)       dst_th + ir * dst_stride);

        إذا (ir + 1 < n_صفاں) {
            hex_l2fetch((ثابت uint8_t *) src_local + src_stride, src_stride, src_stride, 1);
        }

        إذا (opt_رستہ) {
            *dst_local = hvx_reduce_جوڑ_f32_a((ثابت uint8_t *) src_local, ne00);
        } وإلا {
            *dst_local = hvx_reduce_جوڑ_f32((ثابت uint8_t *) src_local, ne00);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) شروع_صف);
}

صحيح op_جوڑ_صفاں(بنية htp_ops_context * octx) {
    جوڑ_صفاں_preamble;

    إذا (octx->src[0]->type != HTP_TYPE_F32) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    إذا (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    ثابت uint32_t src0_nrows      = ne01 * ne02 * ne03;
    ثابت size_t dst_انکڑے_صف_ناپ = dst->ne[0] * حجم(عائم);

    uint32_t صف_شروع = 0;
    uint32_t nrows     = src0_nrows;

    إذا (octx->ctx->mdev.گنتی > 1) {
        uint32_t صفاں_فی_chunk = 0;
        htp_tensor_mdev_صفاں_فی_chunk(dst, حجم(عائم), (uint32_t) dst_انکڑے_صف_ناپ, &صفاں_فی_chunk);
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصے_کرو(src0_nrows, صفاں_فی_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
        صف_شروع = range.شروع;
        nrows     = range.گنتی;
    }

    إذا (nrows == 0) {
        أرجع HTP_حالت_OK;
    }

    ثابت uint32_t n_دھاگے = octx->n_دھاگے;
    ثابت uint32_t صفاں_فی_thread = fastdiv(nrows + n_دھاگے - 1, &octx->n_دھاگے_div);

    bool opt_رستہ = false;
    إذا ((0 == hex_is_aligned((فراغ *) src0->انکڑے, VLEN)) && !(nb01 & (VLEN - 1))) {
        opt_رستہ = true;
    }

    بنية جوڑ_صفاں_context smctx = {
        .octx            = octx,
        .src_انکڑے        = (ثابت uint8_t *) src0->انکڑے + صف_شروع * nb01,
        .dst_انکڑے        = (uint8_t *) dst->انکڑے + صف_شروع * nb1,
        .ne00            = ne00,
        .src_stride      = nb01,
        .dst_stride      = nb1,
        .صفاں_فی_thread = صفاں_فی_thread,
        .کل_صفاں      = nrows,
        .opt_رستہ        = opt_رستہ,
    };

    work_قطار_لگاتار(octx->ctx->work_قطار, جوڑ_صفاں_thread_f32, &smctx, n_دھاگے);

    أرجع HTP_حالت_OK;
}

بنية جوڑ_context {
    بنية htp_ops_context * octx;
    ثابت عائم            * src_انکڑے;
    عائم                    partial_sums[HTP_MAX_NTHREADS];
    uint32_t                 کل_elems;
    uint32_t                 elems_فی_thread;
};

ساكن فراغ جوڑ_thread_f32(بنانشانی صحيح nth, بنانشانی صحيح ith, فراغ * انکڑے) {
    بنية جوڑ_context * sctx = (بنية جوڑ_context *) انکڑے;
    ثابت uint32_t شروع = sctx->elems_فی_thread * ith;
    ثابت uint32_t سرا   = MIN(شروع + sctx->elems_فی_thread, sctx->کل_elems);

    إذا (شروع >= سرا) {
        sctx->partial_sums[ith] = 0.0f;
        أرجع;
    }

    ثابت uint32_t n = سرا - شروع;
    ثابت عائم * src = sctx->src_انکڑے + شروع;

    بنية htp_thread_trace * tr = &sctx->octx->ctx->trace[ith];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) شروع);

    hex_l2fetch_بلاک((ثابت فراغ *) src, n * حجم(عائم));

    sctx->partial_sums[ith] = hvx_reduce_جوڑ_f32((ثابت uint8_t *) src, n);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) شروع);
}

صحيح op_جوڑ(بنية htp_ops_context * octx) {
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * dst  = octx->dst;

    إذا (src0->type != HTP_TYPE_F32) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    إذا (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    إذا (octx->ctx->mdev.گنتی > 1 && octx->ctx->mdev.idx > 0) {
        أرجع HTP_حالت_OK;
    }

    ثابت uint32_t کل_elems = (uint32_t) (src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3]);
    إذا (کل_elems == 0) {
        ((عائم *) dst->انکڑے)[0] = 0.0f;
        أرجع HTP_حالت_OK;
    }

    ثابت uint32_t n_دھاگے = (کل_elems >= 1024) ? MIN(octx->n_دھاگے, HTP_MAX_NTHREADS) : 1;
    ثابت uint32_t raw_chunk = (کل_elems + n_دھاگے - 1) / n_دھاگے;
    ثابت uint32_t elems_فی_thread = hex_گیڑا_اتے(raw_chunk, 32);

    بنية جوڑ_context sctx = {
        .octx             = octx,
        .src_انکڑے         = (ثابت عائم *) src0->انکڑے,
        .کل_elems      = کل_elems,
        .elems_فی_thread = elems_فی_thread,
    };

    work_قطار_لگاتار(octx->ctx->work_قطار, جوڑ_thread_f32, &sctx, n_دھاگے);

    عائم جوڑ = 0.0f;
    لكل (uint32_t i = 0; i < n_دھاگے; i++) {
        جوڑ += sctx.partial_sums[i];
    }
    ((عائم *) dst->انکڑے)[0] = جوڑ;

    أرجع HTP_حالت_OK;
}

ساكن مضمن فراغ argmax_ٹکڑا_f32(
    ثابت عائم * محدود src,
    uint32_t n,
    uint32_t وتھ,
    عائم * باہرلا_val,
    int32_t * باہرلا_idx
) {
    hvx_argmax_f32(src, n, وتھ, باہرلا_val, باہرلا_idx);
}

بنية argmax_context {
    بنية htp_ops_context * octx;
    ثابت عائم            * src_انکڑے;
    int32_t                * dst_انکڑے;
    uint32_t                 ne00;
    uint32_t                 src_stride;
    uint32_t                 dst_stride;
    uint32_t                 صف_شروع;
    uint32_t                 nrows;
    uint32_t                 صفاں_فی_thread;
    uint32_t                 elems_فی_thread;

    عائم                    partial_max[HTP_MAX_NTHREADS];
    int32_t                  partial_idx[HTP_MAX_NTHREADS];
};

ساكن فراغ argmax_thread_single_صف(بنانشانی صحيح nth, بنانشانی صحيح ith, فراغ * انکڑے) {
    بنية argmax_context * actx = (بنية argmax_context *) انکڑے;
    ثابت uint32_t شروع = actx->elems_فی_thread * ith;
    ثابت uint32_t سرا   = MIN(شروع + actx->elems_فی_thread, actx->ne00);

    إذا (شروع >= سرا) {
        actx->partial_max[ith] = -INFINITY;
        actx->partial_idx[ith] = 0;
        أرجع;
    }

    ثابت uint32_t n = سرا - شروع;
    ثابت عائم * src = actx->src_انکڑے + شروع;

    بنية htp_thread_trace * tr = &actx->octx->ctx->trace[ith];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) شروع);

    hex_l2fetch_بلاک((ثابت فراغ *) src, n * حجم(عائم));

    عائم max_val;
    int32_t max_idx;
    argmax_ٹکڑا_f32(src, n, شروع, &max_val, &max_idx);

    actx->partial_max[ith] = max_val;
    actx->partial_idx[ith] = max_idx;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) شروع);
}

ساكن فراغ argmax_thread_multi_صف(بنانشانی صحيح nth, بنانشانی صحيح ith, فراغ * انکڑے) {
    بنية argmax_context * actx = (بنية argmax_context *) انکڑے;
    ثابت uint32_t r0 = actx->صف_شروع + actx->صفاں_فی_thread * ith;
    ثابت uint32_t r1 = MIN(r0 + actx->صفاں_فی_thread, actx->صف_شروع + actx->nrows);

    إذا (r0 >= r1) {
        أرجع;
    }

    بنية htp_thread_trace * tr = &actx->octx->ctx->trace[ith];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r0);

    لكل (uint32_t r = r0; r < r1; r++) {
        ثابت عائم * src_صف = (ثابت عائم *) ((ثابت uint8_t *) actx->src_انکڑے + r * actx->src_stride);
        int32_t * dst_val     = (int32_t *) ((uint8_t *) actx->dst_انکڑے + r * actx->dst_stride);

        hex_l2fetch_بلاک((ثابت فراغ *) src_صف, actx->ne00 * حجم(عائم));

        عائم max_val;
        int32_t max_idx;
        argmax_ٹکڑا_f32(src_صف, actx->ne00, 0, &max_val, &max_idx);

        *dst_val = max_idx;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r0);
}

صحيح op_argmax(بنية htp_ops_context * octx) {
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * dst  = octx->dst;

    إذا (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_I32) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    إذا (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    ثابت uint32_t ne00 = src0->ne[0];
    ثابت uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    إذا (ne00 == 0 || src0_nrows == 0) {
        أرجع HTP_حالت_OK;
    }

    إذا (src0_nrows == 1) {
        إذا (octx->ctx->mdev.گنتی > 1 && octx->ctx->mdev.idx > 0) {
            أرجع HTP_حالت_OK;
        }

        إذا (ne00 == 1) {
            ((int32_t *) dst->انکڑے)[0] = 0;
            أرجع HTP_حالت_OK;
        }

        ثابت uint32_t n_دھاگے = (ne00 >= 1024) ? MIN(octx->n_دھاگے, HTP_MAX_NTHREADS) : 1;
        ثابت uint32_t raw_chunk = (ne00 + n_دھاگے - 1) / n_دھاگے;
        ثابت uint32_t elems_فی_thread = hex_گیڑا_اتے(raw_chunk, 32);

        بنية argmax_context actx = {
            .octx             = octx,
            .src_انکڑے         = (ثابت عائم *) src0->انکڑے,
            .dst_انکڑے         = (int32_t *) dst->انکڑے,
            .ne00             = ne00,
            .src_stride       = src0->nb[1] > 0 ? (uint32_t) src0->nb[1] : (uint32_t) (ne00 * حجم(عائم)),
            .dst_stride       = dst->nb[0] > 0 ? (uint32_t) dst->nb[0] : (uint32_t) حجم(int32_t),
            .صف_شروع        = 0,
            .nrows            = 1,
            .صفاں_فی_thread  = 1,
            .elems_فی_thread = elems_فی_thread,
        };

        work_قطار_لگاتار(octx->ctx->work_قطار, argmax_thread_single_صف, &actx, n_دھاگے);

        عائم بہترین_val = actx.partial_max[0];
        int32_t بہترین_idx = actx.partial_idx[0];
        لكل (uint32_t i = 1; i < n_دھاگے; i++) {
            إذا (actx.partial_max[i] > بہترین_val) {
                بہترین_val = actx.partial_max[i];
                بہترین_idx = actx.partial_idx[i];
            }
        }
        ((int32_t *) dst->انکڑے)[0] = بہترین_idx;
        أرجع HTP_حالت_OK;
    }

    uint32_t صف_شروع = 0;
    uint32_t nrows     = src0_nrows;

    إذا (octx->ctx->mdev.گنتی > 1) {
        ثابت bool can_ونڈو = htp_tensor_mdev_انکڑے_aligned(dst) &&
                               htp_tensor_is_contiguous(dst, حجم(int32_t));
        ثابت uint32_t elems_فی_chunk = can_ونڈو ? 32 : 0;
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصے_کرو(
            src0_nrows, elems_فی_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
        صف_شروع = range.شروع;
        nrows     = range.گنتی;
    }

    إذا (nrows == 0) {
        أرجع HTP_حالت_OK;
    }

    ثابت uint32_t n_دھاگے = MIN(octx->n_دھاگے, nrows);
    ثابت uint32_t صفاں_فی_thread = (nrows + n_دھاگے - 1) / n_دھاگے;

    بنية argmax_context actx = {
        .octx             = octx,
        .src_انکڑے         = (ثابت عائم *) src0->انکڑے,
        .dst_انکڑے         = (int32_t *) dst->انکڑے,
        .ne00             = ne00,
        .src_stride       = src0->nb[1] > 0 ? (uint32_t) src0->nb[1] : (uint32_t) (ne00 * حجم(عائم)),
        .dst_stride       = dst->nb[0] > 0 ? (uint32_t) dst->nb[0] : (uint32_t) حجم(int32_t),
        .صف_شروع        = صف_شروع,
        .nrows            = nrows,
        .صفاں_فی_thread  = صفاں_فی_thread,
        .elems_فی_thread = 0,
    };

    work_قطار_لگاتار(octx->ctx->work_قطار, argmax_thread_multi_صف, &actx, n_دھاگے);
    أرجع HTP_حالت_OK;
}
