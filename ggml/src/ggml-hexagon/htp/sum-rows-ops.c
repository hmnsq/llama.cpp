#رهنمود clang diagnostic ignored "-Wunused-variable"
#رهنمود clang diagnostic ignored "-Wunused-function"
#رهنمود clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <HAP_perf.h>

#ضمن <string.h>
#ضمن <math.h>

#ضمن "dma-queue.h"
#ضمن "hvx-utils.h"

#تعریف_کن GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "hex-common.h"
#ضمن "hex-profile.h"
#ضمن "htp-ctx.h"
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"

#تعریف_کن جمع_ردیفها_preamble                         \
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

بنية جمع_ردیفها_context {
    بنية htp_ops_context * octx;
    ثابت uint8_t * src_داده;
    uint8_t       * dst_داده;
    uint32_t        ne00;
    size_t          src_stride;
    size_t          dst_stride;
    uint32_t        ردیفها_در_هر_thread;
    uint32_t        مجموع_ردیفها;
    bool            opt_مسیر;
};

ساكن فراغ جمع_ردیفها_thread_f32(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ *داده) {
    ثابت بنية جمع_ردیفها_context * smctx = (ثابت بنية جمع_ردیفها_context *) داده;

    ثابت uint32_t ردیفها_در_هر_thread = smctx->ردیفها_در_هر_thread;
    ثابت uint32_t مجموع_ردیفها      = smctx->مجموع_ردیفها;

    ثابت uint32_t شروع_ردیف = ردیفها_در_هر_thread * ith;
    ثابت uint32_t انتها_ردیف   = MIN(شروع_ردیف + ردیفها_در_هر_thread, مجموع_ردیفها);

    إذا (شروع_ردیف >= انتها_ردیف) {
        أرجع;
    }

    ثابت size_t   src_stride = smctx->src_stride;
    ثابت size_t   dst_stride = smctx->dst_stride;
    ثابت uint32_t ne00       = smctx->ne00;
    ثابت bool     opt_مسیر   = smctx->opt_مسیر;

    ثابت عائم * محدود src_th = (ثابت عائم *) (smctx->src_داده + (شروع_ردیف * src_stride));
    عائم       * محدود dst_th = (عائم *)       (smctx->dst_داده + (شروع_ردیف * dst_stride));

    // Calculate actual number of rows for this thread
    ثابت uint32_t n_ردیفها = انتها_ردیف - شروع_ردیف;

    بنية htp_thread_trace * tr = &smctx->octx->ctx->trace[ith];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) شروع_ردیف);

    لكل (uint32_t ir = 0; ir < n_ردیفها; ir++) {
        ثابت عائم * محدود src_local = (ثابت عائم *) ((ثابت uint8_t *) src_th + ir * src_stride);
        عائم       * محدود dst_local = (عائم *)       ((uint8_t *)       dst_th + ir * dst_stride);

        إذا (ir + 1 < n_ردیفها) {
            hex_l2fetch((ثابت uint8_t *) src_local + src_stride, src_stride, src_stride, 1);
        }

        إذا (opt_مسیر) {
            *dst_local = hvx_reduce_جمع_f32_a((ثابت uint8_t *) src_local, ne00);
        } وإلا {
            *dst_local = hvx_reduce_جمع_f32((ثابت uint8_t *) src_local, ne00);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) شروع_ردیف);
}

صحيح op_جمع_ردیفها(بنية htp_ops_context * octx) {
    جمع_ردیفها_preamble;

    إذا (octx->src[0]->type != HTP_TYPE_F32) {
        أرجع HTP_وضعیت_NO_SUPPORT;
    }

    إذا (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        أرجع HTP_وضعیت_NO_SUPPORT;
    }

    ثابت uint32_t src0_nrows      = ne01 * ne02 * ne03;
    ثابت size_t dst_داده_ردیف_اندازه = dst->ne[0] * حجم(عائم);

    uint32_t ردیف_شروع = 0;
    uint32_t nrows     = src0_nrows;

    إذا (octx->ctx->mdev.شمار > 1) {
        uint32_t ردیفها_در_هر_chunk = 0;
        htp_tensor_mdev_ردیفها_در_هر_chunk(dst, حجم(عائم), (uint32_t) dst_داده_ردیف_اندازه, &ردیفها_در_هر_chunk);
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_افراز(src0_nrows, ردیفها_در_هر_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
        ردیف_شروع = range.شروع;
        nrows     = range.شمار;
    }

    إذا (nrows == 0) {
        أرجع HTP_وضعیت_OK;
    }

    ثابت uint32_t n_نخها = octx->n_نخها;
    ثابت uint32_t ردیفها_در_هر_thread = fastdiv(nrows + n_نخها - 1, &octx->n_نخها_div);

    bool opt_مسیر = false;
    إذا ((0 == hex_is_aligned((فراغ *) src0->داده, VLEN)) && !(nb01 & (VLEN - 1))) {
        opt_مسیر = true;
    }

    بنية جمع_ردیفها_context smctx = {
        .octx            = octx,
        .src_داده        = (ثابت uint8_t *) src0->داده + ردیف_شروع * nb01,
        .dst_داده        = (uint8_t *) dst->داده + ردیف_شروع * nb1,
        .ne00            = ne00,
        .src_stride      = nb01,
        .dst_stride      = nb1,
        .ردیفها_در_هر_thread = ردیفها_در_هر_thread,
        .مجموع_ردیفها      = nrows,
        .opt_مسیر        = opt_مسیر,
    };

    work_صف_پیاپی(octx->ctx->work_صف, جمع_ردیفها_thread_f32, &smctx, n_نخها);

    أرجع HTP_وضعیت_OK;
}

بنية جمع_context {
    بنية htp_ops_context * octx;
    ثابت عائم            * src_داده;
    عائم                    partial_sums[HTP_MAX_NTHREADS];
    uint32_t                 مجموع_elems;
    uint32_t                 elems_در_هر_thread;
};

ساكن فراغ جمع_thread_f32(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    بنية جمع_context * sctx = (بنية جمع_context *) داده;
    ثابت uint32_t شروع = sctx->elems_در_هر_thread * ith;
    ثابت uint32_t انتها   = MIN(شروع + sctx->elems_در_هر_thread, sctx->مجموع_elems);

    إذا (شروع >= انتها) {
        sctx->partial_sums[ith] = 0.0f;
        أرجع;
    }

    ثابت uint32_t n = انتها - شروع;
    ثابت عائم * src = sctx->src_داده + شروع;

    بنية htp_thread_trace * tr = &sctx->octx->ctx->trace[ith];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) شروع);

    hex_l2fetch_بلوک((ثابت فراغ *) src, n * حجم(عائم));

    sctx->partial_sums[ith] = hvx_reduce_جمع_f32((ثابت uint8_t *) src, n);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) شروع);
}

صحيح op_جمع(بنية htp_ops_context * octx) {
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * dst  = octx->dst;

    إذا (src0->type != HTP_TYPE_F32) {
        أرجع HTP_وضعیت_NO_SUPPORT;
    }

    إذا (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        أرجع HTP_وضعیت_NO_SUPPORT;
    }

    إذا (octx->ctx->mdev.شمار > 1 && octx->ctx->mdev.idx > 0) {
        أرجع HTP_وضعیت_OK;
    }

    ثابت uint32_t مجموع_elems = (uint32_t) (src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3]);
    إذا (مجموع_elems == 0) {
        ((عائم *) dst->داده)[0] = 0.0f;
        أرجع HTP_وضعیت_OK;
    }

    ثابت uint32_t n_نخها = (مجموع_elems >= 1024) ? MIN(octx->n_نخها, HTP_MAX_NTHREADS) : 1;
    ثابت uint32_t raw_chunk = (مجموع_elems + n_نخها - 1) / n_نخها;
    ثابت uint32_t elems_در_هر_thread = hex_دور_به_بالا(raw_chunk, 32);

    بنية جمع_context sctx = {
        .octx             = octx,
        .src_داده         = (ثابت عائم *) src0->داده,
        .مجموع_elems      = مجموع_elems,
        .elems_در_هر_thread = elems_در_هر_thread,
    };

    work_صف_پیاپی(octx->ctx->work_صف, جمع_thread_f32, &sctx, n_نخها);

    عائم جمع = 0.0f;
    لكل (uint32_t i = 0; i < n_نخها; i++) {
        جمع += sctx.partial_sums[i];
    }
    ((عائم *) dst->داده)[0] = جمع;

    أرجع HTP_وضعیت_OK;
}

ساكن مضمن فراغ argmax_برش_f32(
    ثابت عائم * محدود src,
    uint32_t n,
    uint32_t جابه_جایی,
    عائم * بیرون_val,
    int32_t * بیرون_idx
) {
    hvx_argmax_f32(src, n, جابه_جایی, بیرون_val, بیرون_idx);
}

بنية argmax_context {
    بنية htp_ops_context * octx;
    ثابت عائم            * src_داده;
    int32_t                * dst_داده;
    uint32_t                 ne00;
    uint32_t                 src_stride;
    uint32_t                 dst_stride;
    uint32_t                 ردیف_شروع;
    uint32_t                 nrows;
    uint32_t                 ردیفها_در_هر_thread;
    uint32_t                 elems_در_هر_thread;

    عائم                    partial_max[HTP_MAX_NTHREADS];
    int32_t                  partial_idx[HTP_MAX_NTHREADS];
};

ساكن فراغ argmax_thread_single_ردیف(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    بنية argmax_context * actx = (بنية argmax_context *) داده;
    ثابت uint32_t شروع = actx->elems_در_هر_thread * ith;
    ثابت uint32_t انتها   = MIN(شروع + actx->elems_در_هر_thread, actx->ne00);

    إذا (شروع >= انتها) {
        actx->partial_max[ith] = -INFINITY;
        actx->partial_idx[ith] = 0;
        أرجع;
    }

    ثابت uint32_t n = انتها - شروع;
    ثابت عائم * src = actx->src_داده + شروع;

    بنية htp_thread_trace * tr = &actx->octx->ctx->trace[ith];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) شروع);

    hex_l2fetch_بلوک((ثابت فراغ *) src, n * حجم(عائم));

    عائم max_val;
    int32_t max_idx;
    argmax_برش_f32(src, n, شروع, &max_val, &max_idx);

    actx->partial_max[ith] = max_val;
    actx->partial_idx[ith] = max_idx;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) شروع);
}

ساكن فراغ argmax_thread_multi_ردیف(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    بنية argmax_context * actx = (بنية argmax_context *) داده;
    ثابت uint32_t r0 = actx->ردیف_شروع + actx->ردیفها_در_هر_thread * ith;
    ثابت uint32_t r1 = MIN(r0 + actx->ردیفها_در_هر_thread, actx->ردیف_شروع + actx->nrows);

    إذا (r0 >= r1) {
        أرجع;
    }

    بنية htp_thread_trace * tr = &actx->octx->ctx->trace[ith];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r0);

    لكل (uint32_t r = r0; r < r1; r++) {
        ثابت عائم * src_ردیف = (ثابت عائم *) ((ثابت uint8_t *) actx->src_داده + r * actx->src_stride);
        int32_t * dst_val     = (int32_t *) ((uint8_t *) actx->dst_داده + r * actx->dst_stride);

        hex_l2fetch_بلوک((ثابت فراغ *) src_ردیف, actx->ne00 * حجم(عائم));

        عائم max_val;
        int32_t max_idx;
        argmax_برش_f32(src_ردیف, actx->ne00, 0, &max_val, &max_idx);

        *dst_val = max_idx;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r0);
}

صحيح op_argmax(بنية htp_ops_context * octx) {
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * dst  = octx->dst;

    إذا (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_I32) {
        أرجع HTP_وضعیت_NO_SUPPORT;
    }

    إذا (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        أرجع HTP_وضعیت_NO_SUPPORT;
    }

    ثابت uint32_t ne00 = src0->ne[0];
    ثابت uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    إذا (ne00 == 0 || src0_nrows == 0) {
        أرجع HTP_وضعیت_OK;
    }

    إذا (src0_nrows == 1) {
        إذا (octx->ctx->mdev.شمار > 1 && octx->ctx->mdev.idx > 0) {
            أرجع HTP_وضعیت_OK;
        }

        إذا (ne00 == 1) {
            ((int32_t *) dst->داده)[0] = 0;
            أرجع HTP_وضعیت_OK;
        }

        ثابت uint32_t n_نخها = (ne00 >= 1024) ? MIN(octx->n_نخها, HTP_MAX_NTHREADS) : 1;
        ثابت uint32_t raw_chunk = (ne00 + n_نخها - 1) / n_نخها;
        ثابت uint32_t elems_در_هر_thread = hex_دور_به_بالا(raw_chunk, 32);

        بنية argmax_context actx = {
            .octx             = octx,
            .src_داده         = (ثابت عائم *) src0->داده,
            .dst_داده         = (int32_t *) dst->داده,
            .ne00             = ne00,
            .src_stride       = src0->nb[1] > 0 ? (uint32_t) src0->nb[1] : (uint32_t) (ne00 * حجم(عائم)),
            .dst_stride       = dst->nb[0] > 0 ? (uint32_t) dst->nb[0] : (uint32_t) حجم(int32_t),
            .ردیف_شروع        = 0,
            .nrows            = 1,
            .ردیفها_در_هر_thread  = 1,
            .elems_در_هر_thread = elems_در_هر_thread,
        };

        work_صف_پیاپی(octx->ctx->work_صف, argmax_thread_single_ردیف, &actx, n_نخها);

        عائم بهترین_val = actx.partial_max[0];
        int32_t بهترین_idx = actx.partial_idx[0];
        لكل (uint32_t i = 1; i < n_نخها; i++) {
            إذا (actx.partial_max[i] > بهترین_val) {
                بهترین_val = actx.partial_max[i];
                بهترین_idx = actx.partial_idx[i];
            }
        }
        ((int32_t *) dst->داده)[0] = بهترین_idx;
        أرجع HTP_وضعیت_OK;
    }

    uint32_t ردیف_شروع = 0;
    uint32_t nrows     = src0_nrows;

    إذا (octx->ctx->mdev.شمار > 1) {
        ثابت bool can_تقسیم = htp_tensor_mdev_داده_aligned(dst) &&
                               htp_tensor_is_contiguous(dst, حجم(int32_t));
        ثابت uint32_t elems_در_هر_chunk = can_تقسیم ? 32 : 0;
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_افراز(
            src0_nrows, elems_در_هر_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
        ردیف_شروع = range.شروع;
        nrows     = range.شمار;
    }

    إذا (nrows == 0) {
        أرجع HTP_وضعیت_OK;
    }

    ثابت uint32_t n_نخها = MIN(octx->n_نخها, nrows);
    ثابت uint32_t ردیفها_در_هر_thread = (nrows + n_نخها - 1) / n_نخها;

    بنية argmax_context actx = {
        .octx             = octx,
        .src_داده         = (ثابت عائم *) src0->داده,
        .dst_داده         = (int32_t *) dst->داده,
        .ne00             = ne00,
        .src_stride       = src0->nb[1] > 0 ? (uint32_t) src0->nb[1] : (uint32_t) (ne00 * حجم(عائم)),
        .dst_stride       = dst->nb[0] > 0 ? (uint32_t) dst->nb[0] : (uint32_t) حجم(int32_t),
        .ردیف_شروع        = ردیف_شروع,
        .nrows            = nrows,
        .ردیفها_در_هر_thread  = ردیفها_در_هر_thread,
        .elems_در_هر_thread = 0,
    };

    work_صف_پیاپی(octx->ctx->work_صف, argmax_thread_multi_ردیف, &actx, n_نخها);
    أرجع HTP_وضعیت_OK;
}
