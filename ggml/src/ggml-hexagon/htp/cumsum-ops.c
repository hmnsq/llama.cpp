#ہدایت clang diagnostic ignored "-Wunused-variable"
#ہدایت clang diagnostic ignored "-Wunused-function"
#ہدایت clang diagnostic ignored "-Wunused-but-set-variable"

#شامل <HAP_farf.h>
#شامل <HAP_perf.h>

#تعریف_کرو GGML_COMMON_DECL_C
#شامل "ggml-common.h"
#شامل "hex-common.h"
#شامل "hex-profile.h"
#شامل "htp-ctx.h"
#شامل "htp-ops.h"
#شامل "htp-tensor.h"
#شامل "hvx-types.h"
#شامل "hvx-utils.h"
#شامل "dma-queue.h"

#تعریف_کرو htp_cumsum_tensors_preamble                         \
    ثابت ساخت htp_tensor * محدود src0 = octx->src[0]; \
    ثابت ساخت htp_tensor * محدود dst  = octx->dst;    \
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

ساخت htp_cumsum_context {
    ساخت htp_ops_context * octx;
    size_t          src_صف_سائز;
    size_t          dst_صف_سائز;
    size_t          src_صف_سائز_aligned;
    size_t          dst_صف_سائز_aligned;
    uint32_t        صفیں_فی_thread;
    uint32_t        کل_صفیں;
    uint32_t        صف_آغاز;
};

#تعریف_کرو htp_cumsum_preamble                                                \
    ساخت htp_cumsum_context * cctx = (ساخت htp_cumsum_context *) ڈیٹا; \
    ساخت htp_ops_context *    octx = cctx->octx;                         \
    htp_cumsum_tensors_preamble;                                           \
    dma_قطار * dma_q = octx->ctx->dma[ith];

// ---------------------------------------------------------------------------
// HVX prefix scan helpers
// ---------------------------------------------------------------------------

#اگر __HVX_ARCH__ > 75
ساکن اندرونی HVX_سمتیہ hvx_cumsum_vadd(HVX_سمتیہ a, HVX_سمتیہ b) {
    واپس Q6_Vsf_vadd_VsfVsf(a, b);
}
#ورنہ
ساکن اندرونی HVX_سمتیہ hvx_cumsum_vadd(HVX_سمتیہ a, HVX_سمتیہ b) {
    واپس Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
}
#اگر_ختم  // __HVX_ARCH__ > 75

ساکن اندرونی HVX_سمتیہ hvx_سابقہ_جائزہ_f32(HVX_سمتیہ v, HVX_سمتیہ حاصل_اندر) {
    ثابت HVX_سمتیہ zero = Q6_V_vsplat_R(0);

    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero,  4));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero,  8));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 16));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 32));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 64));
    v = hvx_cumsum_vadd(v, حاصل_اندر);

    واپس v;
}

ساکن اندرونی HVX_سمتیہ hvx_splat_آخری_f32(HVX_سمتیہ v) {
    واپس hvx_vec_repl4(Q6_V_vror_VR(v, 124));
}

ساکن اندرونی خالی hvx_cumsum_صف_f32(ثابت اعشاری * محدود src, اعشاری * محدود dst, uint32_t n) {
    ثابت uint32_t nvec = n / VLEN_FP32;
    ثابت uint32_t nloe = n % VLEN_FP32;

    HVX_سمتیہ حاصل = Q6_V_vsplat_R(0);

    برائے (uint32_t i = 0; i < nvec; i++) {
        HVX_سمتیہ v = *((ثابت HVX_UVector *) (src + i * VLEN_FP32));
        v = hvx_سابقہ_جائزہ_f32(v, حاصل);
        hvx_vec_store_u(dst + i * VLEN_FP32, VLEN, v);
        حاصل = hvx_splat_آخری_f32(v);
    }

    اگر (nloe) {
        اعشاری acc = hvx_vec_get_f32(حاصل);
        ثابت اعشاری * src_دم = src + nvec * VLEN_FP32;
        اعشاری       * dst_دم = dst + nvec * VLEN_FP32;
        برائے (uint32_t i = 0; i < nloe; i++) {
            acc        += src_دم[i];
            dst_دم[i] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Per thread worker: Double-buffered DMA
// ---------------------------------------------------------------------------

ساکن خالی cumsum_thread_f32_dma(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    htp_cumsum_preamble;

    ثابت uint32_t ir0 = cctx->صف_آغاز + cctx->صفیں_فی_thread * ith;
    ثابت uint32_t ir1 = MIN(ir0 + cctx->صفیں_فی_thread, cctx->صف_آغاز + cctx->کل_صفیں);

    اگر (ir0 >= ir1) {
        واپس;
    }

    ثابت size_t src_صف_سائز         = cctx->src_صف_سائز;
    ثابت size_t dst_صف_سائز         = cctx->dst_صف_سائز;
    ثابت size_t src_صف_سائز_aligned = cctx->src_صف_سائز_aligned;
    ثابت size_t dst_صف_سائز_aligned = cctx->dst_صف_سائز_aligned;

    ثابت dma_addr_t src_ڈیٹا = src0->ڈیٹا;
    ثابت dma_addr_t dst_ڈیٹا = dst->ڈیٹا;

    uint8_t * src_spad = octx->src0_spad.ڈیٹا + (ith * src_صف_سائز_aligned * 2);
    uint8_t * dst_spad = octx->dst_spad.ڈیٹا  + (ith * dst_صف_سائز_aligned * 2);

    برائے (uint32_t ir = ir0, spad_idx = 0; ir < ir1 && spad_idx < 2; ir++, spad_idx++) {
        // Dummy dst writeback to establish queue ordering
        dma_قطار_ڈالو(dma_q,
                       dma_make_ڈیٹا(dst_ڈیٹا, dst_spad + (spad_idx * dst_صف_سائز_aligned)),
                       dst_صف_سائز, dst_صف_سائز_aligned, dst_صف_سائز, 0);

        dma_قطار_ڈالو(dma_q,
                       dma_make_ڈیٹا(src_spad + (spad_idx * src_صف_سائز_aligned),
                                     src_ڈیٹا + (ir * src_صف_سائز)),
                       src_صف_سائز_aligned, src_صف_سائز, src_صف_سائز, 1);
    }

    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];

    برائے (uint32_t ir = ir0; ir < ir1; ir++) {
        اعشاری * dst_spad_صف = (اعشاری *) dma_قطار_نکالو(dma_q).src;
        اعشاری * src_spad_صف = (اعشاری *) dma_قطار_نکالو(dma_q).dst;

        htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        hvx_cumsum_صف_f32(src_spad_صف, dst_spad_صف, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_قطار_ڈالو(dma_q,
                       dma_make_ڈیٹا(dst_ڈیٹا + (ir * dst_صف_سائز), dst_spad_صف),
                       dst_صف_سائز, dst_صف_سائز_aligned, dst_صف_سائز, 1);

        ثابت uint32_t اگلا_صف = ir + 2;
        اگر (اگلا_صف < ir1) {
            dma_قطار_ڈالو(dma_q,
                           dma_make_ڈیٹا(src_spad_صف, src_ڈیٹا + (اگلا_صف * src_صف_سائز)),
                           src_صف_سائز_aligned, src_صف_سائز, src_صف_سائز, 1);
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

ساکن خالی cumsum_thread_f32(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    htp_cumsum_preamble;

    ثابت uint8_t * src_ڈیٹا = (ثابت uint8_t *) src0->ڈیٹا;
    uint8_t *       dst_ڈیٹا = (uint8_t *) dst->ڈیٹا;

    ثابت uint32_t ir0 = cctx->صف_آغاز + cctx->صفیں_فی_thread * ith;
    ثابت uint32_t ir1 = MIN(ir0 + cctx->صفیں_فی_thread, cctx->صف_آغاز + cctx->کل_صفیں);

    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    برائے (uint32_t ir = ir0; ir < ir1; ir++) {
        ثابت اعشاری * محدود src_صف = (ثابت اعشاری *) (src_ڈیٹا + ir * cctx->src_صف_سائز);
        اعشاری       * محدود dst_صف = (اعشاری *)       (dst_ڈیٹا + ir * cctx->dst_صف_سائز);
        hvx_cumsum_صف_f32(src_صف, dst_صف, ne00);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    FARF(بلند_2, "cumsum-f32 %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

صحیح op_cumsum_f32(ساخت htp_ops_context * octx) {
    ثابت ساخت htp_tensor * src0 = octx->src[0];
    ثابت ساخت htp_tensor * dst  = octx->dst;

    ثابت uint32_t کل_صفیں      = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت size_t dst_ڈیٹا_صف_سائز = dst->ne[0] * حجم(اعشاری);

    uint32_t صف_آغاز = 0;
    uint32_t nrows     = کل_صفیں;

    اگر (octx->ctx->mdev.گنتی > 1) {
        uint32_t صفیں_فی_chunk = 0;
        htp_tensor_mdev_صفیں_فی_chunk(dst, حجم(اعشاری), (uint32_t) dst_ڈیٹا_صف_سائز, &صفیں_فی_chunk);
        ثابت ساخت htp_tensor_mdev_range range = htp_tensor_mdev_تقسیم(کل_صفیں, صفیں_فی_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
        صف_آغاز = range.آغاز;
        nrows     = range.گنتی;
    }

    اگر (nrows == 0) {
        واپس HTP_حالت_OK;
    }

    ثابت uint32_t n_دھاگے = octx->n_دھاگے;

    ثابت size_t src_صف_سائز         = src0->nb[1];
    ثابت size_t dst_صف_سائز         = dst->nb[1];
    ثابت size_t src_صف_سائز_aligned = hex_باری_اوپر(src_صف_سائز, VLEN);
    ثابت size_t dst_صف_سائز_aligned = hex_باری_اوپر(dst_صف_سائز, VLEN);

    // 2 ping-pong buffers per thread for src and dst
    ثابت size_t spad_فی_thread = 2 * (src_صف_سائز_aligned + dst_صف_سائز_aligned);

    octx->src0_spad.سائز_فی_thread = src_صف_سائز_aligned * 2;
    octx->dst_spad.سائز_فی_thread  = dst_صف_سائز_aligned * 2;

    octx->src0_spad.سائز  = n_دھاگے * octx->src0_spad.سائز_فی_thread;
    octx->dst_spad.سائز   = n_دھاگے * octx->dst_spad.سائز_فی_thread;

    octx->src0_spad.ڈیٹا  = octx->ctx->vtcm_بنیاد;                        octx->src0_spad.src = تہی;
    octx->dst_spad.ڈیٹا   = octx->src0_spad.ڈیٹا + octx->src0_spad.سائز; octx->dst_spad.src  = تہی;

    ساخت htp_cumsum_context cctx = {
        .octx                 = octx,
        .src_صف_سائز         = src_صف_سائز,
        .dst_صف_سائز         = dst_صف_سائز,
        .src_صف_سائز_aligned = src_صف_سائز_aligned,
        .dst_صف_سائز_aligned = dst_صف_سائز_aligned,
        .صفیں_فی_thread      = fastdiv(nrows + n_دھاگے - 1, &octx->n_دھاگے_div),
        .کل_صفیں           = nrows,
        .صف_آغاز            = صف_آغاز,
    };

    اگر (octx->ctx->vtcm_سائز < spad_فی_thread * n_دھاگے) {
        اگر (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
            واپس HTP_حالت_NO_SUPPORT;
        }
        work_قطار_تسلسل(octx->ctx->work_قطار, cumsum_thread_f32, &cctx, n_دھاگے);
    } ورنہ {
        work_قطار_تسلسل(octx->ctx->work_قطار, cumsum_thread_f32_dma, &cctx, n_دھاگے);
    }

    واپس HTP_حالت_OK;
}

صحیح op_cumsum(ساخت htp_ops_context * octx) {
    ثابت ساخت htp_tensor * dst = octx->dst;

    انتخاب (dst->type) {
        حالت HTP_TYPE_F32:
            واپس op_cumsum_f32(octx);
        طےشدہ:
            واپس HTP_حالت_NO_SUPPORT;
    }
}
