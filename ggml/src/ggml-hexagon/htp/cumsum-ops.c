#هدايت clang diagnostic ignored "-Wunused-variable"
#هدايت clang diagnostic ignored "-Wunused-function"
#هدايت clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <HAP_perf.h>

#وصف_ڪريو GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "hex-common.h"
#ضمن "hex-profile.h"
#ضمن "htp-ctx.h"
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"
#ضمن "hvx-types.h"
#ضمن "hvx-utils.h"
#ضمن "dma-queue.h"

#وصف_ڪريو htp_cumsum_tensors_preamble                         \
    ثابت بنية htp_tensor * محدود src0 = octx->src[0]; \
    ثابت بنية htp_tensor * محدود dst  = octx->dst;    \
                                                            \
    ثابت uint32_t ne00 = src0->ne[0];                      \
    ثابت uint32_t ne01 = src0->ne[1];                      \
    ثابت uint32_t ne02 = src0->ne[2];                      \
    ثابت uint32_t ne03 = src0->ne[3];                      \
                                                            \
    ثابت uint32_t ne0 = dst->ne[0];                        \
    ثابت uint32_t ne1 = dst->ne[1];                        \
    ثابت uint32_t ne2 = dst->ne[2];                        \
    ثابت uint32_t ne3 = dst->ne[3];                        \
                                                            \
    ثابت uint32_t nb00 = src0->nb[0];                      \
    ثابت uint32_t nb01 = src0->nb[1];                      \
    ثابت uint32_t nb02 = src0->nb[2];                      \
    ثابت uint32_t nb03 = src0->nb[3];                      \
                                                            \
    ثابت uint32_t nb0 = dst->nb[0];                        \
    ثابت uint32_t nb1 = dst->nb[1];                        \
    ثابت uint32_t nb2 = dst->nb[2];                        \
    ثابت uint32_t nb3 = dst->nb[3];

بنية htp_cumsum_context {
    بنية htp_ops_context * octx;
    size_t          src_صف_سائيز;
    size_t          dst_صف_سائيز;
    size_t          src_صف_سائيز_aligned;
    size_t          dst_صف_سائيز_aligned;
    uint32_t        صفون_في_thread;
    uint32_t        ڪل_صفون;
    uint32_t        صف_شروعات;
};

#وصف_ڪريو htp_cumsum_preamble                                                \
    بنية htp_cumsum_context * cctx = (بنية htp_cumsum_context *) معلومات; \
    بنية htp_ops_context *    octx = cctx->octx;                         \
    htp_cumsum_tensors_preamble;                                           \
    dma_قطار * dma_q = octx->ctx->dma[ith];

// ---------------------------------------------------------------------------
// HVX prefix scan helpers
// ---------------------------------------------------------------------------

#إذا __HVX_ARCH__ > 75
ساكن مضمن HVX_ويڪٽر hvx_cumsum_vadd(HVX_ويڪٽر a, HVX_ويڪٽر b) {
    أرجع Q6_Vsf_vadd_VsfVsf(a, b);
}
#وإلا
ساكن مضمن HVX_ويڪٽر hvx_cumsum_vadd(HVX_ويڪٽر a, HVX_ويڪٽر b) {
    أرجع Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
}
#جيڪڏهن_پورو  // __HVX_ARCH__ > 75

ساكن مضمن HVX_ويڪٽر hvx_اڳياڙي_جاچ_f32(HVX_ويڪٽر v, HVX_ويڪٽر حاصل_اندريون) {
    ثابت HVX_ويڪٽر zero = Q6_V_vsplat_R(0);

    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero,  4));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero,  8));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 16));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 32));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 64));
    v = hvx_cumsum_vadd(v, حاصل_اندريون);

    أرجع v;
}

ساكن مضمن HVX_ويڪٽر hvx_splat_آخري_f32(HVX_ويڪٽر v) {
    أرجع hvx_vec_repl4(Q6_V_vror_VR(v, 124));
}

ساكن مضمن فراغ hvx_cumsum_صف_f32(ثابت عائم * محدود src, عائم * محدود dst, uint32_t n) {
    ثابت uint32_t nvec = n / VLEN_FP32;
    ثابت uint32_t nloe = n % VLEN_FP32;

    HVX_ويڪٽر حاصل = Q6_V_vsplat_R(0);

    لكل (uint32_t i = 0; i < nvec; i++) {
        HVX_ويڪٽر v = *((ثابت HVX_UVector *) (src + i * VLEN_FP32));
        v = hvx_اڳياڙي_جاچ_f32(v, حاصل);
        hvx_vec_store_u(dst + i * VLEN_FP32, VLEN, v);
        حاصل = hvx_splat_آخري_f32(v);
    }

    إذا (nloe) {
        عائم acc = hvx_vec_get_f32(حاصل);
        ثابت عائم * src_پڇ = src + nvec * VLEN_FP32;
        عائم       * dst_پڇ = dst + nvec * VLEN_FP32;
        لكل (uint32_t i = 0; i < nloe; i++) {
            acc        += src_پڇ[i];
            dst_پڇ[i] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Per thread worker: Double-buffered DMA
// ---------------------------------------------------------------------------

ساكن فراغ cumsum_thread_f32_dma(بنانشان صحيح nth, بنانشان صحيح ith, فراغ * معلومات) {
    htp_cumsum_preamble;

    ثابت uint32_t ir0 = cctx->صف_شروعات + cctx->صفون_في_thread * ith;
    ثابت uint32_t ir1 = MIN(ir0 + cctx->صفون_في_thread, cctx->صف_شروعات + cctx->ڪل_صفون);

    إذا (ir0 >= ir1) {
        أرجع;
    }

    ثابت size_t src_صف_سائيز         = cctx->src_صف_سائيز;
    ثابت size_t dst_صف_سائيز         = cctx->dst_صف_سائيز;
    ثابت size_t src_صف_سائيز_aligned = cctx->src_صف_سائيز_aligned;
    ثابت size_t dst_صف_سائيز_aligned = cctx->dst_صف_سائيز_aligned;

    ثابت dma_addr_t src_معلومات = src0->معلومات;
    ثابت dma_addr_t dst_معلومات = dst->معلومات;

    uint8_t * src_spad = octx->src0_spad.معلومات + (ith * src_صف_سائيز_aligned * 2);
    uint8_t * dst_spad = octx->dst_spad.معلومات  + (ith * dst_صف_سائيز_aligned * 2);

    لكل (uint32_t ir = ir0, spad_idx = 0; ir < ir1 && spad_idx < 2; ir++, spad_idx++) {
        // Dummy dst writeback to establish queue ordering
        dma_قطار_ڌڪو(dma_q,
                       dma_make_معلومات(dst_معلومات, dst_spad + (spad_idx * dst_صف_سائيز_aligned)),
                       dst_صف_سائيز, dst_صف_سائيز_aligned, dst_صف_سائيز, 0);

        dma_قطار_ڌڪو(dma_q,
                       dma_make_معلومات(src_spad + (spad_idx * src_صف_سائيز_aligned),
                                     src_معلومات + (ir * src_صف_سائيز)),
                       src_صف_سائيز_aligned, src_صف_سائيز, src_صف_سائيز, 1);
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    لكل (uint32_t ir = ir0; ir < ir1; ir++) {
        عائم * dst_spad_صف = (عائم *) dma_قطار_ڪڍو(dma_q).src;
        عائم * src_spad_صف = (عائم *) dma_قطار_ڪڍو(dma_q).dst;

        htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        hvx_cumsum_صف_f32(src_spad_صف, dst_spad_صف, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_قطار_ڌڪو(dma_q,
                       dma_make_معلومات(dst_معلومات + (ir * dst_صف_سائيز), dst_spad_صف),
                       dst_صف_سائيز, dst_صف_سائيز_aligned, dst_صف_سائيز, 1);

        ثابت uint32_t ايندڙ_صف = ir + 2;
        إذا (ايندڙ_صف < ir1) {
            dma_قطار_ڌڪو(dma_q,
                           dma_make_معلومات(src_spad_صف, src_معلومات + (ايندڙ_صف * src_صف_سائيز)),
                           src_صف_سائيز_aligned, src_صف_سائيز, src_صف_سائيز, 1);
        }
    }

    dma_قطار_flush(dma_q);

    FARF(مٿاهون_2, "cumsum-f32-dma %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

// ---------------------------------------------------------------------------
// Per thread worker: Direct HVX (no DMA)
// ---------------------------------------------------------------------------

ساكن فراغ cumsum_thread_f32(بنانشان صحيح nth, بنانشان صحيح ith, فراغ * معلومات) {
    htp_cumsum_preamble;

    ثابت uint8_t * src_معلومات = (ثابت uint8_t *) src0->معلومات;
    uint8_t *       dst_معلومات = (uint8_t *) dst->معلومات;

    ثابت uint32_t ir0 = cctx->صف_شروعات + cctx->صفون_في_thread * ith;
    ثابت uint32_t ir1 = MIN(ir0 + cctx->صفون_في_thread, cctx->صف_شروعات + cctx->ڪل_صفون);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    لكل (uint32_t ir = ir0; ir < ir1; ir++) {
        ثابت عائم * محدود src_صف = (ثابت عائم *) (src_معلومات + ir * cctx->src_صف_سائيز);
        عائم       * محدود dst_صف = (عائم *)       (dst_معلومات + ir * cctx->dst_صف_سائيز);
        hvx_cumsum_صف_f32(src_صف, dst_صف, ne00);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    FARF(مٿاهون_2, "cumsum-f32 %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

صحيح op_cumsum_f32(بنية htp_ops_context * octx) {
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * dst  = octx->dst;

    ثابت uint32_t ڪل_صفون      = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت size_t dst_معلومات_صف_سائيز = dst->ne[0] * حجم(عائم);

    uint32_t صف_شروعات = 0;
    uint32_t nrows     = ڪل_صفون;

    إذا (octx->ctx->mdev.ڳڻپ > 1) {
        uint32_t صفون_في_chunk = 0;
        htp_tensor_mdev_صفون_في_chunk(dst, حجم(عائم), (uint32_t) dst_معلومات_صف_سائيز, &صفون_في_chunk);
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصا_ڪريو(ڪل_صفون, صفون_في_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ڳڻپ, &octx->ctx->mdev.ڳڻپ_div);
        صف_شروعات = range.شروعات;
        nrows     = range.ڳڻپ;
    }

    إذا (nrows == 0) {
        أرجع HTP_حالت_OK;
    }

    ثابت uint32_t n_ڌاڳا = octx->n_ڌاڳا;

    ثابت size_t src_صف_سائيز         = src0->nb[1];
    ثابت size_t dst_صف_سائيز         = dst->nb[1];
    ثابت size_t src_صف_سائيز_aligned = hex_ڦيرو_مٿي(src_صف_سائيز, VLEN);
    ثابت size_t dst_صف_سائيز_aligned = hex_ڦيرو_مٿي(dst_صف_سائيز, VLEN);

    // 2 ping-pong buffers per thread for src and dst
    ثابت size_t spad_في_thread = 2 * (src_صف_سائيز_aligned + dst_صف_سائيز_aligned);

    octx->src0_spad.سائيز_في_thread = src_صف_سائيز_aligned * 2;
    octx->dst_spad.سائيز_في_thread  = dst_صف_سائيز_aligned * 2;

    octx->src0_spad.سائيز  = n_ڌاڳا * octx->src0_spad.سائيز_في_thread;
    octx->dst_spad.سائيز   = n_ڌاڳا * octx->dst_spad.سائيز_في_thread;

    octx->src0_spad.معلومات  = octx->ctx->vtcm_بنياد;                        octx->src0_spad.src = عدم;
    octx->dst_spad.معلومات   = octx->src0_spad.معلومات + octx->src0_spad.سائيز; octx->dst_spad.src  = عدم;

    بنية htp_cumsum_context cctx = {
        .octx                 = octx,
        .src_صف_سائيز         = src_صف_سائيز,
        .dst_صف_سائيز         = dst_صف_سائيز,
        .src_صف_سائيز_aligned = src_صف_سائيز_aligned,
        .dst_صف_سائيز_aligned = dst_صف_سائيز_aligned,
        .صفون_في_thread      = fastdiv(nrows + n_ڌاڳا - 1, &octx->n_ڌاڳا_div),
        .ڪل_صفون           = nrows,
        .صف_شروعات            = صف_شروعات,
    };

    إذا (octx->ctx->vtcm_سائيز < spad_في_thread * n_ڌاڳا) {
        إذا (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
            أرجع HTP_حالت_NO_SUPPORT;
        }
        work_قطار_لاڳيتو(octx->ctx->work_قطار, cumsum_thread_f32, &cctx, n_ڌاڳا);
    } وإلا {
        work_قطار_لاڳيتو(octx->ctx->work_قطار, cumsum_thread_f32_dma, &cctx, n_ڌاڳا);
    }

    أرجع HTP_حالت_OK;
}

صحيح op_cumsum(بنية htp_ops_context * octx) {
    ثابت بنية htp_tensor * dst = octx->dst;

    بدل (dst->type) {
        حالة HTP_TYPE_F32:
            أرجع op_cumsum_f32(octx);
        افتراضي:
            أرجع HTP_حالت_NO_SUPPORT;
    }
}
