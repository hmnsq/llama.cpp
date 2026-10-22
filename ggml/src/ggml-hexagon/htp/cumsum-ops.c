#لارښوونه clang diagnostic ignored "-Wunused-variable"
#لارښوونه clang diagnostic ignored "-Wunused-function"
#لارښوونه clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <HAP_perf.h>

#تعریف_کړه GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "hex-common.h"
#ضمن "hex-profile.h"
#ضمن "htp-ctx.h"
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"
#ضمن "hvx-types.h"
#ضمن "hvx-utils.h"
#ضمن "dma-queue.h"

#تعریف_کړه htp_cumsum_tensors_preamble                         \
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
    size_t          src_لیکه_اندازه;
    size_t          dst_لیکه_اندازه;
    size_t          src_لیکه_اندازه_aligned;
    size_t          dst_لیکه_اندازه_aligned;
    uint32_t        لیکې_فی_thread;
    uint32_t        ټول_لیکې;
    uint32_t        لیکه_پیل;
};

#تعریف_کړه htp_cumsum_preamble                                                \
    بنية htp_cumsum_context * cctx = (بنية htp_cumsum_context *) معلومات; \
    بنية htp_ops_context *    octx = cctx->octx;                         \
    htp_cumsum_tensors_preamble;                                           \
    dma_کتار * dma_q = octx->ctx->dma[ith];

// ---------------------------------------------------------------------------
// HVX prefix scan helpers
// ---------------------------------------------------------------------------

#إذا __HVX_ARCH__ > 75
ساكن مضمن HVX_ویکټور hvx_cumsum_vadd(HVX_ویکټور a, HVX_ویکټور b) {
    أرجع Q6_Vsf_vadd_VsfVsf(a, b);
}
#وإلا
ساكن مضمن HVX_ویکټور hvx_cumsum_vadd(HVX_ویکټور a, HVX_ویکټور b) {
    أرجع Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
}
#د_که_پای  // __HVX_ARCH__ > 75

ساكن مضمن HVX_ویکټور hvx_مختاړی_لټون_f32(HVX_ویکټور v, HVX_ویکټور لېږد_دننه) {
    ثابت HVX_ویکټور zero = Q6_V_vsplat_R(0);

    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero,  4));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero,  8));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 16));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 32));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 64));
    v = hvx_cumsum_vadd(v, لېږد_دننه);

    أرجع v;
}

ساكن مضمن HVX_ویکټور hvx_splat_وروستی_f32(HVX_ویکټور v) {
    أرجع hvx_vec_repl4(Q6_V_vror_VR(v, 124));
}

ساكن مضمن فراغ hvx_cumsum_لیکه_f32(ثابت عائم * محدود src, عائم * محدود dst, uint32_t n) {
    ثابت uint32_t nvec = n / VLEN_FP32;
    ثابت uint32_t nloe = n % VLEN_FP32;

    HVX_ویکټور لېږد = Q6_V_vsplat_R(0);

    لكل (uint32_t i = 0; i < nvec; i++) {
        HVX_ویکټور v = *((ثابت HVX_UVector *) (src + i * VLEN_FP32));
        v = hvx_مختاړی_لټون_f32(v, لېږد);
        hvx_vec_store_u(dst + i * VLEN_FP32, VLEN, v);
        لېږد = hvx_splat_وروستی_f32(v);
    }

    إذا (nloe) {
        عائم acc = hvx_vec_get_f32(لېږد);
        ثابت عائم * src_لکۍ = src + nvec * VLEN_FP32;
        عائم       * dst_لکۍ = dst + nvec * VLEN_FP32;
        لكل (uint32_t i = 0; i < nloe; i++) {
            acc        += src_لکۍ[i];
            dst_لکۍ[i] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Per thread worker: Double-buffered DMA
// ---------------------------------------------------------------------------

ساكن فراغ cumsum_thread_f32_dma(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    htp_cumsum_preamble;

    ثابت uint32_t ir0 = cctx->لیکه_پیل + cctx->لیکې_فی_thread * ith;
    ثابت uint32_t ir1 = MIN(ir0 + cctx->لیکې_فی_thread, cctx->لیکه_پیل + cctx->ټول_لیکې);

    إذا (ir0 >= ir1) {
        أرجع;
    }

    ثابت size_t src_لیکه_اندازه         = cctx->src_لیکه_اندازه;
    ثابت size_t dst_لیکه_اندازه         = cctx->dst_لیکه_اندازه;
    ثابت size_t src_لیکه_اندازه_aligned = cctx->src_لیکه_اندازه_aligned;
    ثابت size_t dst_لیکه_اندازه_aligned = cctx->dst_لیکه_اندازه_aligned;

    ثابت dma_addr_t src_معلومات = src0->معلومات;
    ثابت dma_addr_t dst_معلومات = dst->معلومات;

    uint8_t * src_spad = octx->src0_spad.معلومات + (ith * src_لیکه_اندازه_aligned * 2);
    uint8_t * dst_spad = octx->dst_spad.معلومات  + (ith * dst_لیکه_اندازه_aligned * 2);

    لكل (uint32_t ir = ir0, spad_idx = 0; ir < ir1 && spad_idx < 2; ir++, spad_idx++) {
        // Dummy dst writeback to establish queue ordering
        dma_کتار_ټېل(dma_q,
                       dma_make_معلومات(dst_معلومات, dst_spad + (spad_idx * dst_لیکه_اندازه_aligned)),
                       dst_لیکه_اندازه, dst_لیکه_اندازه_aligned, dst_لیکه_اندازه, 0);

        dma_کتار_ټېل(dma_q,
                       dma_make_معلومات(src_spad + (spad_idx * src_لیکه_اندازه_aligned),
                                     src_معلومات + (ir * src_لیکه_اندازه)),
                       src_لیکه_اندازه_aligned, src_لیکه_اندازه, src_لیکه_اندازه, 1);
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    لكل (uint32_t ir = ir0; ir < ir1; ir++) {
        عائم * dst_spad_لیکه = (عائم *) dma_کتار_راوباسه(dma_q).src;
        عائم * src_spad_لیکه = (عائم *) dma_کتار_راوباسه(dma_q).dst;

        htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        hvx_cumsum_لیکه_f32(src_spad_لیکه, dst_spad_لیکه, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_کتار_ټېل(dma_q,
                       dma_make_معلومات(dst_معلومات + (ir * dst_لیکه_اندازه), dst_spad_لیکه),
                       dst_لیکه_اندازه, dst_لیکه_اندازه_aligned, dst_لیکه_اندازه, 1);

        ثابت uint32_t بل_لیکه = ir + 2;
        إذا (بل_لیکه < ir1) {
            dma_کتار_ټېل(dma_q,
                           dma_make_معلومات(src_spad_لیکه, src_معلومات + (بل_لیکه * src_لیکه_اندازه)),
                           src_لیکه_اندازه_aligned, src_لیکه_اندازه, src_لیکه_اندازه, 1);
        }
    }

    dma_کتار_flush(dma_q);

    FARF(لوړ_2, "cumsum-f32-dma %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

// ---------------------------------------------------------------------------
// Per thread worker: Direct HVX (no DMA)
// ---------------------------------------------------------------------------

ساكن فراغ cumsum_thread_f32(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    htp_cumsum_preamble;

    ثابت uint8_t * src_معلومات = (ثابت uint8_t *) src0->معلومات;
    uint8_t *       dst_معلومات = (uint8_t *) dst->معلومات;

    ثابت uint32_t ir0 = cctx->لیکه_پیل + cctx->لیکې_فی_thread * ith;
    ثابت uint32_t ir1 = MIN(ir0 + cctx->لیکې_فی_thread, cctx->لیکه_پیل + cctx->ټول_لیکې);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    لكل (uint32_t ir = ir0; ir < ir1; ir++) {
        ثابت عائم * محدود src_لیکه = (ثابت عائم *) (src_معلومات + ir * cctx->src_لیکه_اندازه);
        عائم       * محدود dst_لیکه = (عائم *)       (dst_معلومات + ir * cctx->dst_لیکه_اندازه);
        hvx_cumsum_لیکه_f32(src_لیکه, dst_لیکه, ne00);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    FARF(لوړ_2, "cumsum-f32 %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

صحيح op_cumsum_f32(بنية htp_ops_context * octx) {
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * dst  = octx->dst;

    ثابت uint32_t ټول_لیکې      = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت size_t dst_معلومات_لیکه_اندازه = dst->ne[0] * حجم(عائم);

    uint32_t لیکه_پیل = 0;
    uint32_t nrows     = ټول_لیکې;

    إذا (octx->ctx->mdev.شمېر > 1) {
        uint32_t لیکې_فی_chunk = 0;
        htp_tensor_mdev_لیکې_فی_chunk(dst, حجم(عائم), (uint32_t) dst_معلومات_لیکه_اندازه, &لیکې_فی_chunk);
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_برخه_کول(ټول_لیکې, لیکې_فی_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.شمېر, &octx->ctx->mdev.شمېر_div);
        لیکه_پیل = range.پیل;
        nrows     = range.شمېر;
    }

    إذا (nrows == 0) {
        أرجع HTP_حالت_OK;
    }

    ثابت uint32_t n_تارونه = octx->n_تارونه;

    ثابت size_t src_لیکه_اندازه         = src0->nb[1];
    ثابت size_t dst_لیکه_اندازه         = dst->nb[1];
    ثابت size_t src_لیکه_اندازه_aligned = hex_پړاو_پورته(src_لیکه_اندازه, VLEN);
    ثابت size_t dst_لیکه_اندازه_aligned = hex_پړاو_پورته(dst_لیکه_اندازه, VLEN);

    // 2 ping-pong buffers per thread for src and dst
    ثابت size_t spad_فی_thread = 2 * (src_لیکه_اندازه_aligned + dst_لیکه_اندازه_aligned);

    octx->src0_spad.اندازه_فی_thread = src_لیکه_اندازه_aligned * 2;
    octx->dst_spad.اندازه_فی_thread  = dst_لیکه_اندازه_aligned * 2;

    octx->src0_spad.اندازه  = n_تارونه * octx->src0_spad.اندازه_فی_thread;
    octx->dst_spad.اندازه   = n_تارونه * octx->dst_spad.اندازه_فی_thread;

    octx->src0_spad.معلومات  = octx->ctx->vtcm_بنسټ;                        octx->src0_spad.src = عدم;
    octx->dst_spad.معلومات   = octx->src0_spad.معلومات + octx->src0_spad.اندازه; octx->dst_spad.src  = عدم;

    بنية htp_cumsum_context cctx = {
        .octx                 = octx,
        .src_لیکه_اندازه         = src_لیکه_اندازه,
        .dst_لیکه_اندازه         = dst_لیکه_اندازه,
        .src_لیکه_اندازه_aligned = src_لیکه_اندازه_aligned,
        .dst_لیکه_اندازه_aligned = dst_لیکه_اندازه_aligned,
        .لیکې_فی_thread      = fastdiv(nrows + n_تارونه - 1, &octx->n_تارونه_div),
        .ټول_لیکې           = nrows,
        .لیکه_پیل            = لیکه_پیل,
    };

    إذا (octx->ctx->vtcm_اندازه < spad_فی_thread * n_تارونه) {
        إذا (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
            أرجع HTP_حالت_NO_SUPPORT;
        }
        work_کتار_پرلپسې(octx->ctx->work_کتار, cumsum_thread_f32, &cctx, n_تارونه);
    } وإلا {
        work_کتار_پرلپسې(octx->ctx->work_کتار, cumsum_thread_f32_dma, &cctx, n_تارونه);
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
