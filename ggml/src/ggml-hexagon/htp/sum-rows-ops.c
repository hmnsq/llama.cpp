#لارښوونه clang diagnostic ignored "-Wunused-variable"
#لارښوونه clang diagnostic ignored "-Wunused-function"
#لارښوونه clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <HAP_perf.h>

#ضمن <string.h>
#ضمن <math.h>

#ضمن "dma-queue.h"
#ضمن "hvx-utils.h"

#تعریف_کړه GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "hex-common.h"
#ضمن "hex-profile.h"
#ضمن "htp-ctx.h"
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"

#تعریف_کړه مجموعه_لیکې_preamble                         \
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

بنية مجموعه_لیکې_context {
    بنية htp_ops_context * octx;
    ثابت uint8_t * src_معلومات;
    uint8_t       * dst_معلومات;
    uint32_t        ne00;
    size_t          src_stride;
    size_t          dst_stride;
    uint32_t        لیکې_فی_thread;
    uint32_t        ټول_لیکې;
    bool            opt_لار;
};

ساكن فراغ مجموعه_لیکې_thread_f32(بینښه صحيح nth, بینښه صحيح ith, فراغ *معلومات) {
    ثابت بنية مجموعه_لیکې_context * smctx = (ثابت بنية مجموعه_لیکې_context *) معلومات;

    ثابت uint32_t لیکې_فی_thread = smctx->لیکې_فی_thread;
    ثابت uint32_t ټول_لیکې      = smctx->ټول_لیکې;

    ثابت uint32_t پیل_لیکه = لیکې_فی_thread * ith;
    ثابت uint32_t پای_ټکی_لیکه   = MIN(پیل_لیکه + لیکې_فی_thread, ټول_لیکې);

    إذا (پیل_لیکه >= پای_ټکی_لیکه) {
        أرجع;
    }

    ثابت size_t   src_stride = smctx->src_stride;
    ثابت size_t   dst_stride = smctx->dst_stride;
    ثابت uint32_t ne00       = smctx->ne00;
    ثابت bool     opt_لار   = smctx->opt_لار;

    ثابت عائم * محدود src_th = (ثابت عائم *) (smctx->src_معلومات + (پیل_لیکه * src_stride));
    عائم       * محدود dst_th = (عائم *)       (smctx->dst_معلومات + (پیل_لیکه * dst_stride));

    // Calculate actual number of rows for this thread
    ثابت uint32_t n_لیکې = پای_ټکی_لیکه - پیل_لیکه;

    بنية htp_thread_trace * tr = &smctx->octx->ctx->trace[ith];
    htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) پیل_لیکه);

    لكل (uint32_t ir = 0; ir < n_لیکې; ir++) {
        ثابت عائم * محدود src_local = (ثابت عائم *) ((ثابت uint8_t *) src_th + ir * src_stride);
        عائم       * محدود dst_local = (عائم *)       ((uint8_t *)       dst_th + ir * dst_stride);

        إذا (ir + 1 < n_لیکې) {
            hex_l2fetch((ثابت uint8_t *) src_local + src_stride, src_stride, src_stride, 1);
        }

        إذا (opt_لار) {
            *dst_local = hvx_reduce_مجموعه_f32_a((ثابت uint8_t *) src_local, ne00);
        } وإلا {
            *dst_local = hvx_reduce_مجموعه_f32((ثابت uint8_t *) src_local, ne00);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) پیل_لیکه);
}

صحيح op_مجموعه_لیکې(بنية htp_ops_context * octx) {
    مجموعه_لیکې_preamble;

    إذا (octx->src[0]->type != HTP_TYPE_F32) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    إذا (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    ثابت uint32_t src0_nrows      = ne01 * ne02 * ne03;
    ثابت size_t dst_معلومات_لیکه_اندازه = dst->ne[0] * حجم(عائم);

    uint32_t لیکه_پیل = 0;
    uint32_t nrows     = src0_nrows;

    إذا (octx->ctx->mdev.شمېر > 1) {
        uint32_t لیکې_فی_chunk = 0;
        htp_tensor_mdev_لیکې_فی_chunk(dst, حجم(عائم), (uint32_t) dst_معلومات_لیکه_اندازه, &لیکې_فی_chunk);
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_برخه_کول(src0_nrows, لیکې_فی_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.شمېر, &octx->ctx->mdev.شمېر_div);
        لیکه_پیل = range.پیل;
        nrows     = range.شمېر;
    }

    إذا (nrows == 0) {
        أرجع HTP_حالت_OK;
    }

    ثابت uint32_t n_تارونه = octx->n_تارونه;
    ثابت uint32_t لیکې_فی_thread = fastdiv(nrows + n_تارونه - 1, &octx->n_تارونه_div);

    bool opt_لار = false;
    إذا ((0 == hex_is_aligned((فراغ *) src0->معلومات, VLEN)) && !(nb01 & (VLEN - 1))) {
        opt_لار = true;
    }

    بنية مجموعه_لیکې_context smctx = {
        .octx            = octx,
        .src_معلومات        = (ثابت uint8_t *) src0->معلومات + لیکه_پیل * nb01,
        .dst_معلومات        = (uint8_t *) dst->معلومات + لیکه_پیل * nb1,
        .ne00            = ne00,
        .src_stride      = nb01,
        .dst_stride      = nb1,
        .لیکې_فی_thread = لیکې_فی_thread,
        .ټول_لیکې      = nrows,
        .opt_لار        = opt_لار,
    };

    work_کتار_پرلپسې(octx->ctx->work_کتار, مجموعه_لیکې_thread_f32, &smctx, n_تارونه);

    أرجع HTP_حالت_OK;
}

بنية مجموعه_context {
    بنية htp_ops_context * octx;
    ثابت عائم            * src_معلومات;
    عائم                    partial_sums[HTP_MAX_NTHREADS];
    uint32_t                 ټول_elems;
    uint32_t                 elems_فی_thread;
};

ساكن فراغ مجموعه_thread_f32(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    بنية مجموعه_context * sctx = (بنية مجموعه_context *) معلومات;
    ثابت uint32_t پیل = sctx->elems_فی_thread * ith;
    ثابت uint32_t پای_ټکی   = MIN(پیل + sctx->elems_فی_thread, sctx->ټول_elems);

    إذا (پیل >= پای_ټکی) {
        sctx->partial_sums[ith] = 0.0f;
        أرجع;
    }

    ثابت uint32_t n = پای_ټکی - پیل;
    ثابت عائم * src = sctx->src_معلومات + پیل;

    بنية htp_thread_trace * tr = &sctx->octx->ctx->trace[ith];
    htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) پیل);

    hex_l2fetch_بلاک((ثابت فراغ *) src, n * حجم(عائم));

    sctx->partial_sums[ith] = hvx_reduce_مجموعه_f32((ثابت uint8_t *) src, n);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) پیل);
}

صحيح op_مجموعه(بنية htp_ops_context * octx) {
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * dst  = octx->dst;

    إذا (src0->type != HTP_TYPE_F32) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    إذا (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    إذا (octx->ctx->mdev.شمېر > 1 && octx->ctx->mdev.idx > 0) {
        أرجع HTP_حالت_OK;
    }

    ثابت uint32_t ټول_elems = (uint32_t) (src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3]);
    إذا (ټول_elems == 0) {
        ((عائم *) dst->معلومات)[0] = 0.0f;
        أرجع HTP_حالت_OK;
    }

    ثابت uint32_t n_تارونه = (ټول_elems >= 1024) ? MIN(octx->n_تارونه, HTP_MAX_NTHREADS) : 1;
    ثابت uint32_t raw_chunk = (ټول_elems + n_تارونه - 1) / n_تارونه;
    ثابت uint32_t elems_فی_thread = hex_پړاو_پورته(raw_chunk, 32);

    بنية مجموعه_context sctx = {
        .octx             = octx,
        .src_معلومات         = (ثابت عائم *) src0->معلومات,
        .ټول_elems      = ټول_elems,
        .elems_فی_thread = elems_فی_thread,
    };

    work_کتار_پرلپسې(octx->ctx->work_کتار, مجموعه_thread_f32, &sctx, n_تارونه);

    عائم مجموعه = 0.0f;
    لكل (uint32_t i = 0; i < n_تارونه; i++) {
        مجموعه += sctx.partial_sums[i];
    }
    ((عائم *) dst->معلومات)[0] = مجموعه;

    أرجع HTP_حالت_OK;
}

ساكن مضمن فراغ argmax_ټوټه_f32(
    ثابت عائم * محدود src,
    uint32_t n,
    uint32_t دننه_واټن,
    عائم * بهر_val,
    int32_t * بهر_idx
) {
    hvx_argmax_f32(src, n, دننه_واټن, بهر_val, بهر_idx);
}

بنية argmax_context {
    بنية htp_ops_context * octx;
    ثابت عائم            * src_معلومات;
    int32_t                * dst_معلومات;
    uint32_t                 ne00;
    uint32_t                 src_stride;
    uint32_t                 dst_stride;
    uint32_t                 لیکه_پیل;
    uint32_t                 nrows;
    uint32_t                 لیکې_فی_thread;
    uint32_t                 elems_فی_thread;

    عائم                    partial_max[HTP_MAX_NTHREADS];
    int32_t                  partial_idx[HTP_MAX_NTHREADS];
};

ساكن فراغ argmax_thread_single_لیکه(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    بنية argmax_context * actx = (بنية argmax_context *) معلومات;
    ثابت uint32_t پیل = actx->elems_فی_thread * ith;
    ثابت uint32_t پای_ټکی   = MIN(پیل + actx->elems_فی_thread, actx->ne00);

    إذا (پیل >= پای_ټکی) {
        actx->partial_max[ith] = -INFINITY;
        actx->partial_idx[ith] = 0;
        أرجع;
    }

    ثابت uint32_t n = پای_ټکی - پیل;
    ثابت عائم * src = actx->src_معلومات + پیل;

    بنية htp_thread_trace * tr = &actx->octx->ctx->trace[ith];
    htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) پیل);

    hex_l2fetch_بلاک((ثابت فراغ *) src, n * حجم(عائم));

    عائم max_val;
    int32_t max_idx;
    argmax_ټوټه_f32(src, n, پیل, &max_val, &max_idx);

    actx->partial_max[ith] = max_val;
    actx->partial_idx[ith] = max_idx;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) پیل);
}

ساكن فراغ argmax_thread_multi_لیکه(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    بنية argmax_context * actx = (بنية argmax_context *) معلومات;
    ثابت uint32_t r0 = actx->لیکه_پیل + actx->لیکې_فی_thread * ith;
    ثابت uint32_t r1 = MIN(r0 + actx->لیکې_فی_thread, actx->لیکه_پیل + actx->nrows);

    إذا (r0 >= r1) {
        أرجع;
    }

    بنية htp_thread_trace * tr = &actx->octx->ctx->trace[ith];
    htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r0);

    لكل (uint32_t r = r0; r < r1; r++) {
        ثابت عائم * src_لیکه = (ثابت عائم *) ((ثابت uint8_t *) actx->src_معلومات + r * actx->src_stride);
        int32_t * dst_val     = (int32_t *) ((uint8_t *) actx->dst_معلومات + r * actx->dst_stride);

        hex_l2fetch_بلاک((ثابت فراغ *) src_لیکه, actx->ne00 * حجم(عائم));

        عائم max_val;
        int32_t max_idx;
        argmax_ټوټه_f32(src_لیکه, actx->ne00, 0, &max_val, &max_idx);

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
        إذا (octx->ctx->mdev.شمېر > 1 && octx->ctx->mdev.idx > 0) {
            أرجع HTP_حالت_OK;
        }

        إذا (ne00 == 1) {
            ((int32_t *) dst->معلومات)[0] = 0;
            أرجع HTP_حالت_OK;
        }

        ثابت uint32_t n_تارونه = (ne00 >= 1024) ? MIN(octx->n_تارونه, HTP_MAX_NTHREADS) : 1;
        ثابت uint32_t raw_chunk = (ne00 + n_تارونه - 1) / n_تارونه;
        ثابت uint32_t elems_فی_thread = hex_پړاو_پورته(raw_chunk, 32);

        بنية argmax_context actx = {
            .octx             = octx,
            .src_معلومات         = (ثابت عائم *) src0->معلومات,
            .dst_معلومات         = (int32_t *) dst->معلومات,
            .ne00             = ne00,
            .src_stride       = src0->nb[1] > 0 ? (uint32_t) src0->nb[1] : (uint32_t) (ne00 * حجم(عائم)),
            .dst_stride       = dst->nb[0] > 0 ? (uint32_t) dst->nb[0] : (uint32_t) حجم(int32_t),
            .لیکه_پیل        = 0,
            .nrows            = 1,
            .لیکې_فی_thread  = 1,
            .elems_فی_thread = elems_فی_thread,
        };

        work_کتار_پرلپسې(octx->ctx->work_کتار, argmax_thread_single_لیکه, &actx, n_تارونه);

        عائم غوره_val = actx.partial_max[0];
        int32_t غوره_idx = actx.partial_idx[0];
        لكل (uint32_t i = 1; i < n_تارونه; i++) {
            إذا (actx.partial_max[i] > غوره_val) {
                غوره_val = actx.partial_max[i];
                غوره_idx = actx.partial_idx[i];
            }
        }
        ((int32_t *) dst->معلومات)[0] = غوره_idx;
        أرجع HTP_حالت_OK;
    }

    uint32_t لیکه_پیل = 0;
    uint32_t nrows     = src0_nrows;

    إذا (octx->ctx->mdev.شمېر > 1) {
        ثابت bool can_وېشل = htp_tensor_mdev_معلومات_aligned(dst) &&
                               htp_tensor_is_contiguous(dst, حجم(int32_t));
        ثابت uint32_t elems_فی_chunk = can_وېشل ? 32 : 0;
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_برخه_کول(
            src0_nrows, elems_فی_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.شمېر, &octx->ctx->mdev.شمېر_div);
        لیکه_پیل = range.پیل;
        nrows     = range.شمېر;
    }

    إذا (nrows == 0) {
        أرجع HTP_حالت_OK;
    }

    ثابت uint32_t n_تارونه = MIN(octx->n_تارونه, nrows);
    ثابت uint32_t لیکې_فی_thread = (nrows + n_تارونه - 1) / n_تارونه;

    بنية argmax_context actx = {
        .octx             = octx,
        .src_معلومات         = (ثابت عائم *) src0->معلومات,
        .dst_معلومات         = (int32_t *) dst->معلومات,
        .ne00             = ne00,
        .src_stride       = src0->nb[1] > 0 ? (uint32_t) src0->nb[1] : (uint32_t) (ne00 * حجم(عائم)),
        .dst_stride       = dst->nb[0] > 0 ? (uint32_t) dst->nb[0] : (uint32_t) حجم(int32_t),
        .لیکه_پیل        = لیکه_پیل,
        .nrows            = nrows,
        .لیکې_فی_thread  = لیکې_فی_thread,
        .elems_فی_thread = 0,
    };

    work_کتار_پرلپسې(octx->ctx->work_کتار, argmax_thread_multi_لیکه, &actx, n_تارونه);
    أرجع HTP_حالت_OK;
}
