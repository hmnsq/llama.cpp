#رهنمود clang diagnostic ignored "-Wunused-variable"
#رهنمود clang diagnostic ignored "-Wunused-function"
#رهنمود clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <HAP_perf.h>

#تعریف_کن GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "hex-common.h"
#ضمن "hex-profile.h"
#ضمن "htp-ctx.h"
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"
#ضمن "hvx-types.h"
#ضمن "hvx-utils.h"
#ضمن "dma-queue.h"

#تعریف_کن htp_cumsum_tensors_preamble                         \
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
    size_t          src_ردیف_اندازه;
    size_t          dst_ردیف_اندازه;
    size_t          src_ردیف_اندازه_aligned;
    size_t          dst_ردیف_اندازه_aligned;
    uint32_t        ردیفها_در_هر_thread;
    uint32_t        مجموع_ردیفها;
    uint32_t        ردیف_شروع;
};

#تعریف_کن htp_cumsum_preamble                                                \
    بنية htp_cumsum_context * cctx = (بنية htp_cumsum_context *) داده; \
    بنية htp_ops_context *    octx = cctx->octx;                         \
    htp_cumsum_tensors_preamble;                                           \
    dma_صف * dma_q = octx->ctx->dma[ith];

// ---------------------------------------------------------------------------
// HVX prefix scan helpers
// ---------------------------------------------------------------------------

#إذا __HVX_ARCH__ > 75
ساكن مضمن HVX_وکتور hvx_cumsum_vadd(HVX_وکتور a, HVX_وکتور b) {
    أرجع Q6_Vsf_vadd_VsfVsf(a, b);
}
#وإلا
ساكن مضمن HVX_وکتور hvx_cumsum_vadd(HVX_وکتور a, HVX_وکتور b) {
    أرجع Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
}
#پایان_اگر  // __HVX_ARCH__ > 75

ساكن مضمن HVX_وکتور hvx_پیشوند_پیمایش_f32(HVX_وکتور v, HVX_وکتور انتقال_درون) {
    ثابت HVX_وکتور zero = Q6_V_vsplat_R(0);

    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero,  4));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero,  8));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 16));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 32));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 64));
    v = hvx_cumsum_vadd(v, انتقال_درون);

    أرجع v;
}

ساكن مضمن HVX_وکتور hvx_splat_آخرین_f32(HVX_وکتور v) {
    أرجع hvx_vec_repl4(Q6_V_vror_VR(v, 124));
}

ساكن مضمن فراغ hvx_cumsum_ردیف_f32(ثابت عائم * محدود src, عائم * محدود dst, uint32_t n) {
    ثابت uint32_t nvec = n / VLEN_FP32;
    ثابت uint32_t nloe = n % VLEN_FP32;

    HVX_وکتور انتقال = Q6_V_vsplat_R(0);

    لكل (uint32_t i = 0; i < nvec; i++) {
        HVX_وکتور v = *((ثابت HVX_UVector *) (src + i * VLEN_FP32));
        v = hvx_پیشوند_پیمایش_f32(v, انتقال);
        hvx_vec_store_u(dst + i * VLEN_FP32, VLEN, v);
        انتقال = hvx_splat_آخرین_f32(v);
    }

    إذا (nloe) {
        عائم acc = hvx_vec_get_f32(انتقال);
        ثابت عائم * src_دم = src + nvec * VLEN_FP32;
        عائم       * dst_دم = dst + nvec * VLEN_FP32;
        لكل (uint32_t i = 0; i < nloe; i++) {
            acc        += src_دم[i];
            dst_دم[i] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Per thread worker: Double-buffered DMA
// ---------------------------------------------------------------------------

ساكن فراغ cumsum_thread_f32_dma(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    htp_cumsum_preamble;

    ثابت uint32_t ir0 = cctx->ردیف_شروع + cctx->ردیفها_در_هر_thread * ith;
    ثابت uint32_t ir1 = MIN(ir0 + cctx->ردیفها_در_هر_thread, cctx->ردیف_شروع + cctx->مجموع_ردیفها);

    إذا (ir0 >= ir1) {
        أرجع;
    }

    ثابت size_t src_ردیف_اندازه         = cctx->src_ردیف_اندازه;
    ثابت size_t dst_ردیف_اندازه         = cctx->dst_ردیف_اندازه;
    ثابت size_t src_ردیف_اندازه_aligned = cctx->src_ردیف_اندازه_aligned;
    ثابت size_t dst_ردیف_اندازه_aligned = cctx->dst_ردیف_اندازه_aligned;

    ثابت dma_addr_t src_داده = src0->داده;
    ثابت dma_addr_t dst_داده = dst->داده;

    uint8_t * src_spad = octx->src0_spad.داده + (ith * src_ردیف_اندازه_aligned * 2);
    uint8_t * dst_spad = octx->dst_spad.داده  + (ith * dst_ردیف_اندازه_aligned * 2);

    لكل (uint32_t ir = ir0, spad_idx = 0; ir < ir1 && spad_idx < 2; ir++, spad_idx++) {
        // Dummy dst writeback to establish queue ordering
        dma_صف_افزودن(dma_q,
                       dma_make_داده(dst_داده, dst_spad + (spad_idx * dst_ردیف_اندازه_aligned)),
                       dst_ردیف_اندازه, dst_ردیف_اندازه_aligned, dst_ردیف_اندازه, 0);

        dma_صف_افزودن(dma_q,
                       dma_make_داده(src_spad + (spad_idx * src_ردیف_اندازه_aligned),
                                     src_داده + (ir * src_ردیف_اندازه)),
                       src_ردیف_اندازه_aligned, src_ردیف_اندازه, src_ردیف_اندازه, 1);
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    لكل (uint32_t ir = ir0; ir < ir1; ir++) {
        عائم * dst_spad_ردیف = (عائم *) dma_صف_برداشت(dma_q).src;
        عائم * src_spad_ردیف = (عائم *) dma_صف_برداشت(dma_q).dst;

        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        hvx_cumsum_ردیف_f32(src_spad_ردیف, dst_spad_ردیف, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_صف_افزودن(dma_q,
                       dma_make_داده(dst_داده + (ir * dst_ردیف_اندازه), dst_spad_ردیف),
                       dst_ردیف_اندازه, dst_ردیف_اندازه_aligned, dst_ردیف_اندازه, 1);

        ثابت uint32_t بعدی_ردیف = ir + 2;
        إذا (بعدی_ردیف < ir1) {
            dma_صف_افزودن(dma_q,
                           dma_make_داده(src_spad_ردیف, src_داده + (بعدی_ردیف * src_ردیف_اندازه)),
                           src_ردیف_اندازه_aligned, src_ردیف_اندازه, src_ردیف_اندازه, 1);
        }
    }

    dma_صف_flush(dma_q);

    FARF(بالا_2, "cumsum-f32-dma %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

// ---------------------------------------------------------------------------
// Per thread worker: Direct HVX (no DMA)
// ---------------------------------------------------------------------------

ساكن فراغ cumsum_thread_f32(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    htp_cumsum_preamble;

    ثابت uint8_t * src_داده = (ثابت uint8_t *) src0->داده;
    uint8_t *       dst_داده = (uint8_t *) dst->داده;

    ثابت uint32_t ir0 = cctx->ردیف_شروع + cctx->ردیفها_در_هر_thread * ith;
    ثابت uint32_t ir1 = MIN(ir0 + cctx->ردیفها_در_هر_thread, cctx->ردیف_شروع + cctx->مجموع_ردیفها);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    لكل (uint32_t ir = ir0; ir < ir1; ir++) {
        ثابت عائم * محدود src_ردیف = (ثابت عائم *) (src_داده + ir * cctx->src_ردیف_اندازه);
        عائم       * محدود dst_ردیف = (عائم *)       (dst_داده + ir * cctx->dst_ردیف_اندازه);
        hvx_cumsum_ردیف_f32(src_ردیف, dst_ردیف, ne00);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    FARF(بالا_2, "cumsum-f32 %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

صحيح op_cumsum_f32(بنية htp_ops_context * octx) {
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * dst  = octx->dst;

    ثابت uint32_t مجموع_ردیفها      = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت size_t dst_داده_ردیف_اندازه = dst->ne[0] * حجم(عائم);

    uint32_t ردیف_شروع = 0;
    uint32_t nrows     = مجموع_ردیفها;

    إذا (octx->ctx->mdev.شمار > 1) {
        uint32_t ردیفها_در_هر_chunk = 0;
        htp_tensor_mdev_ردیفها_در_هر_chunk(dst, حجم(عائم), (uint32_t) dst_داده_ردیف_اندازه, &ردیفها_در_هر_chunk);
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_افراز(مجموع_ردیفها, ردیفها_در_هر_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
        ردیف_شروع = range.شروع;
        nrows     = range.شمار;
    }

    إذا (nrows == 0) {
        أرجع HTP_وضعیت_OK;
    }

    ثابت uint32_t n_نخها = octx->n_نخها;

    ثابت size_t src_ردیف_اندازه         = src0->nb[1];
    ثابت size_t dst_ردیف_اندازه         = dst->nb[1];
    ثابت size_t src_ردیف_اندازه_aligned = hex_دور_به_بالا(src_ردیف_اندازه, VLEN);
    ثابت size_t dst_ردیف_اندازه_aligned = hex_دور_به_بالا(dst_ردیف_اندازه, VLEN);

    // 2 ping-pong buffers per thread for src and dst
    ثابت size_t spad_در_هر_thread = 2 * (src_ردیف_اندازه_aligned + dst_ردیف_اندازه_aligned);

    octx->src0_spad.اندازه_در_هر_thread = src_ردیف_اندازه_aligned * 2;
    octx->dst_spad.اندازه_در_هر_thread  = dst_ردیف_اندازه_aligned * 2;

    octx->src0_spad.اندازه  = n_نخها * octx->src0_spad.اندازه_در_هر_thread;
    octx->dst_spad.اندازه   = n_نخها * octx->dst_spad.اندازه_در_هر_thread;

    octx->src0_spad.داده  = octx->ctx->vtcm_پایه;                        octx->src0_spad.src = عدم;
    octx->dst_spad.داده   = octx->src0_spad.داده + octx->src0_spad.اندازه; octx->dst_spad.src  = عدم;

    بنية htp_cumsum_context cctx = {
        .octx                 = octx,
        .src_ردیف_اندازه         = src_ردیف_اندازه,
        .dst_ردیف_اندازه         = dst_ردیف_اندازه,
        .src_ردیف_اندازه_aligned = src_ردیف_اندازه_aligned,
        .dst_ردیف_اندازه_aligned = dst_ردیف_اندازه_aligned,
        .ردیفها_در_هر_thread      = fastdiv(nrows + n_نخها - 1, &octx->n_نخها_div),
        .مجموع_ردیفها           = nrows,
        .ردیف_شروع            = ردیف_شروع,
    };

    إذا (octx->ctx->vtcm_اندازه < spad_در_هر_thread * n_نخها) {
        إذا (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
            أرجع HTP_وضعیت_NO_SUPPORT;
        }
        work_صف_پیاپی(octx->ctx->work_صف, cumsum_thread_f32, &cctx, n_نخها);
    } وإلا {
        work_صف_پیاپی(octx->ctx->work_صف, cumsum_thread_f32_dma, &cctx, n_نخها);
    }

    أرجع HTP_وضعیت_OK;
}

صحيح op_cumsum(بنية htp_ops_context * octx) {
    ثابت بنية htp_tensor * dst = octx->dst;

    بدل (dst->type) {
        حالة HTP_TYPE_F32:
            أرجع op_cumsum_f32(octx);
        افتراضي:
            أرجع HTP_وضعیت_NO_SUPPORT;
    }
}
