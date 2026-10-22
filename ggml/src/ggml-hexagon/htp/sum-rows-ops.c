#هدايت clang diagnostic ignored "-Wunused-variable"
#هدايت clang diagnostic ignored "-Wunused-function"
#هدايت clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <HAP_perf.h>

#ضمن <string.h>
#ضمن <math.h>

#ضمن "dma-queue.h"
#ضمن "hvx-utils.h"

#وصف_ڪريو GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "hex-common.h"
#ضمن "hex-profile.h"
#ضمن "htp-ctx.h"
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"

#وصف_ڪريو جوڙ_صفون_preamble                         \
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

بنية جوڙ_صفون_context {
    بنية htp_ops_context * octx;
    ثابت uint8_t * src_معلومات;
    uint8_t       * dst_معلومات;
    uint32_t        ne00;
    size_t          src_stride;
    size_t          dst_stride;
    uint32_t        صفون_في_thread;
    uint32_t        ڪل_صفون;
    bool            opt_رستو;
};

ساكن فراغ جوڙ_صفون_thread_f32(بنانشان صحيح nth, بنانشان صحيح ith, فراغ *معلومات) {
    ثابت بنية جوڙ_صفون_context * smctx = (ثابت بنية جوڙ_صفون_context *) معلومات;

    ثابت uint32_t صفون_في_thread = smctx->صفون_في_thread;
    ثابت uint32_t ڪل_صفون      = smctx->ڪل_صفون;

    ثابت uint32_t شروعات_صف = صفون_في_thread * ith;
    ثابت uint32_t پڇاڙي_صف   = MIN(شروعات_صف + صفون_في_thread, ڪل_صفون);

    إذا (شروعات_صف >= پڇاڙي_صف) {
        أرجع;
    }

    ثابت size_t   src_stride = smctx->src_stride;
    ثابت size_t   dst_stride = smctx->dst_stride;
    ثابت uint32_t ne00       = smctx->ne00;
    ثابت bool     opt_رستو   = smctx->opt_رستو;

    ثابت عائم * محدود src_th = (ثابت عائم *) (smctx->src_معلومات + (شروعات_صف * src_stride));
    عائم       * محدود dst_th = (عائم *)       (smctx->dst_معلومات + (شروعات_صف * dst_stride));

    // Calculate actual number of rows for this thread
    ثابت uint32_t n_صفون = پڇاڙي_صف - شروعات_صف;

    بنية htp_thread_trace * tr = &smctx->octx->ctx->trace[ith];
    htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) شروعات_صف);

    لكل (uint32_t ir = 0; ir < n_صفون; ir++) {
        ثابت عائم * محدود src_local = (ثابت عائم *) ((ثابت uint8_t *) src_th + ir * src_stride);
        عائم       * محدود dst_local = (عائم *)       ((uint8_t *)       dst_th + ir * dst_stride);

        إذا (ir + 1 < n_صفون) {
            hex_l2fetch((ثابت uint8_t *) src_local + src_stride, src_stride, src_stride, 1);
        }

        إذا (opt_رستو) {
            *dst_local = hvx_reduce_جوڙ_f32_a((ثابت uint8_t *) src_local, ne00);
        } وإلا {
            *dst_local = hvx_reduce_جوڙ_f32((ثابت uint8_t *) src_local, ne00);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) شروعات_صف);
}

صحيح op_جوڙ_صفون(بنية htp_ops_context * octx) {
    جوڙ_صفون_preamble;

    إذا (octx->src[0]->type != HTP_TYPE_F32) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    إذا (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    ثابت uint32_t src0_nrows      = ne01 * ne02 * ne03;
    ثابت size_t dst_معلومات_صف_سائيز = dst->ne[0] * حجم(عائم);

    uint32_t صف_شروعات = 0;
    uint32_t nrows     = src0_nrows;

    إذا (octx->ctx->mdev.ڳڻپ > 1) {
        uint32_t صفون_في_chunk = 0;
        htp_tensor_mdev_صفون_في_chunk(dst, حجم(عائم), (uint32_t) dst_معلومات_صف_سائيز, &صفون_في_chunk);
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصا_ڪريو(src0_nrows, صفون_في_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ڳڻپ, &octx->ctx->mdev.ڳڻپ_div);
        صف_شروعات = range.شروعات;
        nrows     = range.ڳڻپ;
    }

    إذا (nrows == 0) {
        أرجع HTP_حالت_OK;
    }

    ثابت uint32_t n_ڌاڳا = octx->n_ڌاڳا;
    ثابت uint32_t صفون_في_thread = fastdiv(nrows + n_ڌاڳا - 1, &octx->n_ڌاڳا_div);

    bool opt_رستو = false;
    إذا ((0 == hex_is_aligned((فراغ *) src0->معلومات, VLEN)) && !(nb01 & (VLEN - 1))) {
        opt_رستو = true;
    }

    بنية جوڙ_صفون_context smctx = {
        .octx            = octx,
        .src_معلومات        = (ثابت uint8_t *) src0->معلومات + صف_شروعات * nb01,
        .dst_معلومات        = (uint8_t *) dst->معلومات + صف_شروعات * nb1,
        .ne00            = ne00,
        .src_stride      = nb01,
        .dst_stride      = nb1,
        .صفون_في_thread = صفون_في_thread,
        .ڪل_صفون      = nrows,
        .opt_رستو        = opt_رستو,
    };

    work_قطار_لاڳيتو(octx->ctx->work_قطار, جوڙ_صفون_thread_f32, &smctx, n_ڌاڳا);

    أرجع HTP_حالت_OK;
}

بنية جوڙ_context {
    بنية htp_ops_context * octx;
    ثابت عائم            * src_معلومات;
    عائم                    partial_sums[HTP_MAX_NTHREADS];
    uint32_t                 ڪل_elems;
    uint32_t                 elems_في_thread;
};

ساكن فراغ جوڙ_thread_f32(بنانشان صحيح nth, بنانشان صحيح ith, فراغ * معلومات) {
    بنية جوڙ_context * sctx = (بنية جوڙ_context *) معلومات;
    ثابت uint32_t شروعات = sctx->elems_في_thread * ith;
    ثابت uint32_t پڇاڙي   = MIN(شروعات + sctx->elems_في_thread, sctx->ڪل_elems);

    إذا (شروعات >= پڇاڙي) {
        sctx->partial_sums[ith] = 0.0f;
        أرجع;
    }

    ثابت uint32_t n = پڇاڙي - شروعات;
    ثابت عائم * src = sctx->src_معلومات + شروعات;

    بنية htp_thread_trace * tr = &sctx->octx->ctx->trace[ith];
    htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) شروعات);

    hex_l2fetch_بلاڪ((ثابت فراغ *) src, n * حجم(عائم));

    sctx->partial_sums[ith] = hvx_reduce_جوڙ_f32((ثابت uint8_t *) src, n);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) شروعات);
}

صحيح op_جوڙ(بنية htp_ops_context * octx) {
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * dst  = octx->dst;

    إذا (src0->type != HTP_TYPE_F32) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    إذا (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    إذا (octx->ctx->mdev.ڳڻپ > 1 && octx->ctx->mdev.idx > 0) {
        أرجع HTP_حالت_OK;
    }

    ثابت uint32_t ڪل_elems = (uint32_t) (src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3]);
    إذا (ڪل_elems == 0) {
        ((عائم *) dst->معلومات)[0] = 0.0f;
        أرجع HTP_حالت_OK;
    }

    ثابت uint32_t n_ڌاڳا = (ڪل_elems >= 1024) ? MIN(octx->n_ڌاڳا, HTP_MAX_NTHREADS) : 1;
    ثابت uint32_t raw_chunk = (ڪل_elems + n_ڌاڳا - 1) / n_ڌاڳا;
    ثابت uint32_t elems_في_thread = hex_ڦيرو_مٿي(raw_chunk, 32);

    بنية جوڙ_context sctx = {
        .octx             = octx,
        .src_معلومات         = (ثابت عائم *) src0->معلومات,
        .ڪل_elems      = ڪل_elems,
        .elems_في_thread = elems_في_thread,
    };

    work_قطار_لاڳيتو(octx->ctx->work_قطار, جوڙ_thread_f32, &sctx, n_ڌاڳا);

    عائم جوڙ = 0.0f;
    لكل (uint32_t i = 0; i < n_ڌاڳا; i++) {
        جوڙ += sctx.partial_sums[i];
    }
    ((عائم *) dst->معلومات)[0] = جوڙ;

    أرجع HTP_حالت_OK;
}

ساكن مضمن فراغ argmax_ٽڪرو_f32(
    ثابت عائم * محدود src,
    uint32_t n,
    uint32_t وٿي,
    عائم * ٻاهريون_val,
    int32_t * ٻاهريون_idx
) {
    hvx_argmax_f32(src, n, وٿي, ٻاهريون_val, ٻاهريون_idx);
}

بنية argmax_context {
    بنية htp_ops_context * octx;
    ثابت عائم            * src_معلومات;
    int32_t                * dst_معلومات;
    uint32_t                 ne00;
    uint32_t                 src_stride;
    uint32_t                 dst_stride;
    uint32_t                 صف_شروعات;
    uint32_t                 nrows;
    uint32_t                 صفون_في_thread;
    uint32_t                 elems_في_thread;

    عائم                    partial_max[HTP_MAX_NTHREADS];
    int32_t                  partial_idx[HTP_MAX_NTHREADS];
};

ساكن فراغ argmax_thread_single_صف(بنانشان صحيح nth, بنانشان صحيح ith, فراغ * معلومات) {
    بنية argmax_context * actx = (بنية argmax_context *) معلومات;
    ثابت uint32_t شروعات = actx->elems_في_thread * ith;
    ثابت uint32_t پڇاڙي   = MIN(شروعات + actx->elems_في_thread, actx->ne00);

    إذا (شروعات >= پڇاڙي) {
        actx->partial_max[ith] = -INFINITY;
        actx->partial_idx[ith] = 0;
        أرجع;
    }

    ثابت uint32_t n = پڇاڙي - شروعات;
    ثابت عائم * src = actx->src_معلومات + شروعات;

    بنية htp_thread_trace * tr = &actx->octx->ctx->trace[ith];
    htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) شروعات);

    hex_l2fetch_بلاڪ((ثابت فراغ *) src, n * حجم(عائم));

    عائم max_val;
    int32_t max_idx;
    argmax_ٽڪرو_f32(src, n, شروعات, &max_val, &max_idx);

    actx->partial_max[ith] = max_val;
    actx->partial_idx[ith] = max_idx;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) شروعات);
}

ساكن فراغ argmax_thread_multi_صف(بنانشان صحيح nth, بنانشان صحيح ith, فراغ * معلومات) {
    بنية argmax_context * actx = (بنية argmax_context *) معلومات;
    ثابت uint32_t r0 = actx->صف_شروعات + actx->صفون_في_thread * ith;
    ثابت uint32_t r1 = MIN(r0 + actx->صفون_في_thread, actx->صف_شروعات + actx->nrows);

    إذا (r0 >= r1) {
        أرجع;
    }

    بنية htp_thread_trace * tr = &actx->octx->ctx->trace[ith];
    htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r0);

    لكل (uint32_t r = r0; r < r1; r++) {
        ثابت عائم * src_صف = (ثابت عائم *) ((ثابت uint8_t *) actx->src_معلومات + r * actx->src_stride);
        int32_t * dst_val     = (int32_t *) ((uint8_t *) actx->dst_معلومات + r * actx->dst_stride);

        hex_l2fetch_بلاڪ((ثابت فراغ *) src_صف, actx->ne00 * حجم(عائم));

        عائم max_val;
        int32_t max_idx;
        argmax_ٽڪرو_f32(src_صف, actx->ne00, 0, &max_val, &max_idx);

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
        إذا (octx->ctx->mdev.ڳڻپ > 1 && octx->ctx->mdev.idx > 0) {
            أرجع HTP_حالت_OK;
        }

        إذا (ne00 == 1) {
            ((int32_t *) dst->معلومات)[0] = 0;
            أرجع HTP_حالت_OK;
        }

        ثابت uint32_t n_ڌاڳا = (ne00 >= 1024) ? MIN(octx->n_ڌاڳا, HTP_MAX_NTHREADS) : 1;
        ثابت uint32_t raw_chunk = (ne00 + n_ڌاڳا - 1) / n_ڌاڳا;
        ثابت uint32_t elems_في_thread = hex_ڦيرو_مٿي(raw_chunk, 32);

        بنية argmax_context actx = {
            .octx             = octx,
            .src_معلومات         = (ثابت عائم *) src0->معلومات,
            .dst_معلومات         = (int32_t *) dst->معلومات,
            .ne00             = ne00,
            .src_stride       = src0->nb[1] > 0 ? (uint32_t) src0->nb[1] : (uint32_t) (ne00 * حجم(عائم)),
            .dst_stride       = dst->nb[0] > 0 ? (uint32_t) dst->nb[0] : (uint32_t) حجم(int32_t),
            .صف_شروعات        = 0,
            .nrows            = 1,
            .صفون_في_thread  = 1,
            .elems_في_thread = elems_في_thread,
        };

        work_قطار_لاڳيتو(octx->ctx->work_قطار, argmax_thread_single_صف, &actx, n_ڌاڳا);

        عائم بهترين_val = actx.partial_max[0];
        int32_t بهترين_idx = actx.partial_idx[0];
        لكل (uint32_t i = 1; i < n_ڌاڳا; i++) {
            إذا (actx.partial_max[i] > بهترين_val) {
                بهترين_val = actx.partial_max[i];
                بهترين_idx = actx.partial_idx[i];
            }
        }
        ((int32_t *) dst->معلومات)[0] = بهترين_idx;
        أرجع HTP_حالت_OK;
    }

    uint32_t صف_شروعات = 0;
    uint32_t nrows     = src0_nrows;

    إذا (octx->ctx->mdev.ڳڻپ > 1) {
        ثابت bool can_ورهايو = htp_tensor_mdev_معلومات_aligned(dst) &&
                               htp_tensor_is_contiguous(dst, حجم(int32_t));
        ثابت uint32_t elems_في_chunk = can_ورهايو ? 32 : 0;
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصا_ڪريو(
            src0_nrows, elems_في_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ڳڻپ, &octx->ctx->mdev.ڳڻپ_div);
        صف_شروعات = range.شروعات;
        nrows     = range.ڳڻپ;
    }

    إذا (nrows == 0) {
        أرجع HTP_حالت_OK;
    }

    ثابت uint32_t n_ڌاڳا = MIN(octx->n_ڌاڳا, nrows);
    ثابت uint32_t صفون_في_thread = (nrows + n_ڌاڳا - 1) / n_ڌاڳا;

    بنية argmax_context actx = {
        .octx             = octx,
        .src_معلومات         = (ثابت عائم *) src0->معلومات,
        .dst_معلومات         = (int32_t *) dst->معلومات,
        .ne00             = ne00,
        .src_stride       = src0->nb[1] > 0 ? (uint32_t) src0->nb[1] : (uint32_t) (ne00 * حجم(عائم)),
        .dst_stride       = dst->nb[0] > 0 ? (uint32_t) dst->nb[0] : (uint32_t) حجم(int32_t),
        .صف_شروعات        = صف_شروعات,
        .nrows            = nrows,
        .صفون_في_thread  = صفون_في_thread,
        .elems_في_thread = 0,
    };

    work_قطار_لاڳيتو(octx->ctx->work_قطار, argmax_thread_multi_صف, &actx, n_ڌاڳا);
    أرجع HTP_حالت_OK;
}
