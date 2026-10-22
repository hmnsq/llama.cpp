#ہدایت clang diagnostic ignored "-Wunused-variable"
#ہدایت clang diagnostic ignored "-Wunused-function"
#ہدایت clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <HAP_perf.h>

#تعریف_کرو GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "hex-common.h"
#ضمن "hex-profile.h"
#ضمن "htp-ctx.h"
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"
#ضمن "hvx-types.h"
#ضمن "hvx-utils.h"
#ضمن "dma-queue.h"

#تعریف_کرو htp_cumsum_tensors_preamble                         \
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
    size_t          src_صف_ناپ;
    size_t          dst_صف_ناپ;
    size_t          src_صف_ناپ_aligned;
    size_t          dst_صف_ناپ_aligned;
    uint32_t        صفاں_فی_thread;
    uint32_t        کل_صفاں;
    uint32_t        صف_شروع;
};

#تعریف_کرو htp_cumsum_preamble                                                \
    بنية htp_cumsum_context * cctx = (بنية htp_cumsum_context *) انکڑے; \
    بنية htp_ops_context *    octx = cctx->octx;                         \
    htp_cumsum_tensors_preamble;                                           \
    dma_قطار * dma_q = octx->ctx->dma[ith];

// ---------------------------------------------------------------------------
// HVX prefix scan helpers
// ---------------------------------------------------------------------------

#إذا __HVX_ARCH__ > 75
ساكن مضمن HVX_سمتیہ hvx_cumsum_vadd(HVX_سمتیہ a, HVX_سمتیہ b) {
    أرجع Q6_Vsf_vadd_VsfVsf(a, b);
}
#وإلا
ساكن مضمن HVX_سمتیہ hvx_cumsum_vadd(HVX_سمتیہ a, HVX_سمتیہ b) {
    أرجع Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
}
#جے_انت  // __HVX_ARCH__ > 75

ساكن مضمن HVX_سمتیہ hvx_اگیتر_جانچ_f32(HVX_سمتیہ v, HVX_سمتیہ حاصل_وچ) {
    ثابت HVX_سمتیہ zero = Q6_V_vsplat_R(0);

    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero,  4));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero,  8));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 16));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 32));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 64));
    v = hvx_cumsum_vadd(v, حاصل_وچ);

    أرجع v;
}

ساكن مضمن HVX_سمتیہ hvx_splat_آخری_f32(HVX_سمتیہ v) {
    أرجع hvx_vec_repl4(Q6_V_vror_VR(v, 124));
}

ساكن مضمن فراغ hvx_cumsum_صف_f32(ثابت عائم * محدود src, عائم * محدود dst, uint32_t n) {
    ثابت uint32_t nvec = n / VLEN_FP32;
    ثابت uint32_t nloe = n % VLEN_FP32;

    HVX_سمتیہ حاصل = Q6_V_vsplat_R(0);

    لكل (uint32_t i = 0; i < nvec; i++) {
        HVX_سمتیہ v = *((ثابت HVX_UVector *) (src + i * VLEN_FP32));
        v = hvx_اگیتر_جانچ_f32(v, حاصل);
        hvx_vec_store_u(dst + i * VLEN_FP32, VLEN, v);
        حاصل = hvx_splat_آخری_f32(v);
    }

    إذا (nloe) {
        عائم acc = hvx_vec_get_f32(حاصل);
        ثابت عائم * src_پوچھ = src + nvec * VLEN_FP32;
        عائم       * dst_پوچھ = dst + nvec * VLEN_FP32;
        لكل (uint32_t i = 0; i < nloe; i++) {
            acc        += src_پوچھ[i];
            dst_پوچھ[i] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Per thread worker: Double-buffered DMA
// ---------------------------------------------------------------------------

ساكن فراغ cumsum_thread_f32_dma(بنانشانی صحيح nth, بنانشانی صحيح ith, فراغ * انکڑے) {
    htp_cumsum_preamble;

    ثابت uint32_t ir0 = cctx->صف_شروع + cctx->صفاں_فی_thread * ith;
    ثابت uint32_t ir1 = MIN(ir0 + cctx->صفاں_فی_thread, cctx->صف_شروع + cctx->کل_صفاں);

    إذا (ir0 >= ir1) {
        أرجع;
    }

    ثابت size_t src_صف_ناپ         = cctx->src_صف_ناپ;
    ثابت size_t dst_صف_ناپ         = cctx->dst_صف_ناپ;
    ثابت size_t src_صف_ناپ_aligned = cctx->src_صف_ناپ_aligned;
    ثابت size_t dst_صف_ناپ_aligned = cctx->dst_صف_ناپ_aligned;

    ثابت dma_addr_t src_انکڑے = src0->انکڑے;
    ثابت dma_addr_t dst_انکڑے = dst->انکڑے;

    uint8_t * src_spad = octx->src0_spad.انکڑے + (ith * src_صف_ناپ_aligned * 2);
    uint8_t * dst_spad = octx->dst_spad.انکڑے  + (ith * dst_صف_ناپ_aligned * 2);

    لكل (uint32_t ir = ir0, spad_idx = 0; ir < ir1 && spad_idx < 2; ir++, spad_idx++) {
        // Dummy dst writeback to establish queue ordering
        dma_قطار_ڈالو(dma_q,
                       dma_make_انکڑے(dst_انکڑے, dst_spad + (spad_idx * dst_صف_ناپ_aligned)),
                       dst_صف_ناپ, dst_صف_ناپ_aligned, dst_صف_ناپ, 0);

        dma_قطار_ڈالو(dma_q,
                       dma_make_انکڑے(src_spad + (spad_idx * src_صف_ناپ_aligned),
                                     src_انکڑے + (ir * src_صف_ناپ)),
                       src_صف_ناپ_aligned, src_صف_ناپ, src_صف_ناپ, 1);
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    لكل (uint32_t ir = ir0; ir < ir1; ir++) {
        عائم * dst_spad_صف = (عائم *) dma_قطار_نکالو(dma_q).src;
        عائم * src_spad_صف = (عائم *) dma_قطار_نکالو(dma_q).dst;

        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        hvx_cumsum_صف_f32(src_spad_صف, dst_spad_صف, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_قطار_ڈالو(dma_q,
                       dma_make_انکڑے(dst_انکڑے + (ir * dst_صف_ناپ), dst_spad_صف),
                       dst_صف_ناپ, dst_صف_ناپ_aligned, dst_صف_ناپ, 1);

        ثابت uint32_t اگلا_صف = ir + 2;
        إذا (اگلا_صف < ir1) {
            dma_قطار_ڈالو(dma_q,
                           dma_make_انکڑے(src_spad_صف, src_انکڑے + (اگلا_صف * src_صف_ناپ)),
                           src_صف_ناپ_aligned, src_صف_ناپ, src_صف_ناپ, 1);
        }
    }

    dma_قطار_flush(dma_q);

    FARF(بلند_2, "cumsum-f32-dma %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

// ---------------------------------------------------------------------------
// Per thread worker: Direct HVX (no DMA)
// ---------------------------------------------------------------------------

ساكن فراغ cumsum_thread_f32(بنانشانی صحيح nth, بنانشانی صحيح ith, فراغ * انکڑے) {
    htp_cumsum_preamble;

    ثابت uint8_t * src_انکڑے = (ثابت uint8_t *) src0->انکڑے;
    uint8_t *       dst_انکڑے = (uint8_t *) dst->انکڑے;

    ثابت uint32_t ir0 = cctx->صف_شروع + cctx->صفاں_فی_thread * ith;
    ثابت uint32_t ir1 = MIN(ir0 + cctx->صفاں_فی_thread, cctx->صف_شروع + cctx->کل_صفاں);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    لكل (uint32_t ir = ir0; ir < ir1; ir++) {
        ثابت عائم * محدود src_صف = (ثابت عائم *) (src_انکڑے + ir * cctx->src_صف_ناپ);
        عائم       * محدود dst_صف = (عائم *)       (dst_انکڑے + ir * cctx->dst_صف_ناپ);
        hvx_cumsum_صف_f32(src_صف, dst_صف, ne00);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    FARF(بلند_2, "cumsum-f32 %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

صحيح op_cumsum_f32(بنية htp_ops_context * octx) {
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * dst  = octx->dst;

    ثابت uint32_t کل_صفاں      = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت size_t dst_انکڑے_صف_ناپ = dst->ne[0] * حجم(عائم);

    uint32_t صف_شروع = 0;
    uint32_t nrows     = کل_صفاں;

    إذا (octx->ctx->mdev.گنتی > 1) {
        uint32_t صفاں_فی_chunk = 0;
        htp_tensor_mdev_صفاں_فی_chunk(dst, حجم(عائم), (uint32_t) dst_انکڑے_صف_ناپ, &صفاں_فی_chunk);
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصے_کرو(کل_صفاں, صفاں_فی_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
        صف_شروع = range.شروع;
        nrows     = range.گنتی;
    }

    إذا (nrows == 0) {
        أرجع HTP_حالت_OK;
    }

    ثابت uint32_t n_دھاگے = octx->n_دھاگے;

    ثابت size_t src_صف_ناپ         = src0->nb[1];
    ثابت size_t dst_صف_ناپ         = dst->nb[1];
    ثابت size_t src_صف_ناپ_aligned = hex_گیڑا_اتے(src_صف_ناپ, VLEN);
    ثابت size_t dst_صف_ناپ_aligned = hex_گیڑا_اتے(dst_صف_ناپ, VLEN);

    // 2 ping-pong buffers per thread for src and dst
    ثابت size_t spad_فی_thread = 2 * (src_صف_ناپ_aligned + dst_صف_ناپ_aligned);

    octx->src0_spad.ناپ_فی_thread = src_صف_ناپ_aligned * 2;
    octx->dst_spad.ناپ_فی_thread  = dst_صف_ناپ_aligned * 2;

    octx->src0_spad.ناپ  = n_دھاگے * octx->src0_spad.ناپ_فی_thread;
    octx->dst_spad.ناپ   = n_دھاگے * octx->dst_spad.ناپ_فی_thread;

    octx->src0_spad.انکڑے  = octx->ctx->vtcm_ادھار;                        octx->src0_spad.src = عدم;
    octx->dst_spad.انکڑے   = octx->src0_spad.انکڑے + octx->src0_spad.ناپ; octx->dst_spad.src  = عدم;

    بنية htp_cumsum_context cctx = {
        .octx                 = octx,
        .src_صف_ناپ         = src_صف_ناپ,
        .dst_صف_ناپ         = dst_صف_ناپ,
        .src_صف_ناپ_aligned = src_صف_ناپ_aligned,
        .dst_صف_ناپ_aligned = dst_صف_ناپ_aligned,
        .صفاں_فی_thread      = fastdiv(nrows + n_دھاگے - 1, &octx->n_دھاگے_div),
        .کل_صفاں           = nrows,
        .صف_شروع            = صف_شروع,
    };

    إذا (octx->ctx->vtcm_ناپ < spad_فی_thread * n_دھاگے) {
        إذا (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
            أرجع HTP_حالت_NO_SUPPORT;
        }
        work_قطار_لگاتار(octx->ctx->work_قطار, cumsum_thread_f32, &cctx, n_دھاگے);
    } وإلا {
        work_قطار_لگاتار(octx->ctx->work_قطار, cumsum_thread_f32_dma, &cctx, n_دھاگے);
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
