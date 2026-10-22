#సూచన clang diagnostic ignored "-Wunused-variable"
#సూచన clang diagnostic ignored "-Wunused-function"
#సూచన clang diagnostic ignored "-Wunused-but-set-variable"

#అన్తర్భావ <HAP_farf.h>
#అన్తర్భావ <HAP_mem.h>
#అన్తర్భావ <HAP_ps.h>
#అన్తర్భావ <hexagon_protos.h>
#అన్తర్భావ <hexagon_types.h>
#అన్తర్భావ <math.h>
#అన్తర్భావ <qurt_thread.h>
#అన్తర్భావ <string.h>

#నిర్వచించు GGML_COMMON_DECL_C
#అన్తర్భావ "ggml-common.h"
#అన్తర్భావ "htp-ctx.h"
#అన్తర్భావ "dma-queue.h"
#అన్తర్భావ "hex-profile.h"
#అన్తర్భావ "htp-ops.h"
#అన్తర్భావ "htp-tensor.h"
#అన్తర్భావ "hvx-utils.h"
#అన్తర్భావ "ssm-conv.h"

నిర్మాణం htp_ssm_conv_context {
    నిర్మాణం htp_ops_context *                  octx;
    స్థిరరాశి నిర్మాణం htp_ssm_conv_kernel_params * kparams;
    uint32_t                                  nrows_చొప్పున_thread;
    uint32_t                                  d_inner_tile;
    uint32_t                                  అడ్డువరుస_ప్రారంభం;
    uint32_t                                  nrows;
};

// In-register 32x32 fp32 transpose using std 5-stage HVX vshuff butterfly.
స్థిర అంతఃవరుస శూన్య hvx_transpose_32x32_f32(HVX_సదిశ m[32]) {
    HVX_సదిశ tmp[32];

    // Stage 0 (R = -4): pair (2i, 2i+1) for i = 0..15. m -> tmp.
    కొరకు (పూర్ణాంకం i = 0; i < 16; ++i) {
        HVX_సదిశ_Pair p = Q6_W_vshuff_VVR(m[2*i + 1], m[2*i], -4);
        tmp[2*i + 0] = Q6_V_lo_W(p);
        tmp[2*i + 1] = Q6_V_hi_W(p);
    }

    // Stage 1 (R = -8): per block of 4, pair (b+0, b+2) and (b+1, b+3). tmp -> m.
    కొరకు (పూర్ణాంకం b = 0; b < 32; b += 4) {
        HVX_సదిశ_Pair p0 = Q6_W_vshuff_VVR(tmp[b + 2], tmp[b + 0], -8);
        HVX_సదిశ_Pair p1 = Q6_W_vshuff_VVR(tmp[b + 3], tmp[b + 1], -8);
        m[b + 0] = Q6_V_lo_W(p0); m[b + 1] = Q6_V_hi_W(p0);
        m[b + 2] = Q6_V_lo_W(p1); m[b + 3] = Q6_V_hi_W(p1);
    }

    // Stage 2 (R = -16): per block of 8, pair (b+i, b+i+4) for i = 0..3. m -> tmp.
    కొరకు (పూర్ణాంకం b = 0; b < 32; b += 8) {
        కొరకు (పూర్ణాంకం i = 0; i < 4; ++i) {
            HVX_సదిశ_Pair p = Q6_W_vshuff_VVR(m[b + i + 4], m[b + i], -16);
            tmp[b + 2*i + 0] = Q6_V_lo_W(p);
            tmp[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }

    // Stage 3 (R = -32): per block of 16, pair (b+i, b+i+8) for i = 0..7. tmp -> m.
    కొరకు (పూర్ణాంకం b = 0; b < 32; b += 16) {
        కొరకు (పూర్ణాంకం i = 0; i < 8; ++i) {
            HVX_సదిశ_Pair p = Q6_W_vshuff_VVR(tmp[b + i + 8], tmp[b + i], -32);
            m[b + 2*i + 0] = Q6_V_lo_W(p);
            m[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }

    // Stage 4 (R = -64): pair (i, i+16) for i = 0..15. m -> tmp -> m.
    కొరకు (పూర్ణాంకం i = 0; i < 16; ++i) {
        HVX_సదిశ_Pair p = Q6_W_vshuff_VVR(m[i + 16], m[i], -64);
        tmp[2 * i + 0]   = Q6_V_lo_W(p);
        tmp[2 * i + 1]   = Q6_V_hi_W(p);
    }

    కొరకు (పూర్ణాంకం i = 0; i < 32; ++i) {
        m[i] = tmp[i];
    }
}

// HVX deinterleave for d_conv == 4: channel-major raw VTCM -> tap-major T VTCM
స్థిర అంతఃవరుస శూన్య hvx_ssm_conv_unpack_గమ్యం_T_4(స్థిరరాశి దశాంశం * raw, దశాంశం * T, uint32_t d_inner_చొప్పున_thread, uint32_t d_inner_stride) {
    కొరకు (uint32_t cb = 0; cb < d_inner_చొప్పున_thread; cb += VLEN_FP32) {
        HVX_సదిశ v0 = *(స్థిరరాశి HVX_సదిశ *)(raw + (cb + 0) * 4);
        HVX_సదిశ v1 = *(స్థిరరాశి HVX_సదిశ *)(raw + (cb + 8) * 4);
        HVX_సదిశ v2 = *(స్థిరరాశి HVX_సదిశ *)(raw + (cb + 16) * 4);
        HVX_సదిశ v3 = *(స్థిరరాశి HVX_సదిశ *)(raw + (cb + 24) * 4);

        HVX_సదిశ_Pair p01 = Q6_W_vdeal_VVR(v1, v0, -4);
        HVX_సదిశ_Pair p23 = Q6_W_vdeal_VVR(v3, v2, -4);

        HVX_సదిశ_Pair p_w02 = Q6_W_vdeal_VVR(Q6_V_lo_W(p23), Q6_V_lo_W(p01), -4);
        HVX_సదిశ_Pair p_w13 = Q6_W_vdeal_VVR(Q6_V_hi_W(p23), Q6_V_hi_W(p01), -4);

        *(HVX_సదిశ *)(T + 0 * d_inner_stride + cb) = Q6_V_lo_W(p_w02);
        *(HVX_సదిశ *)(T + 1 * d_inner_stride + cb) = Q6_V_lo_W(p_w13);
        *(HVX_సదిశ *)(T + 2 * d_inner_stride + cb) = Q6_V_hi_W(p_w02);
        *(HVX_సదిశ *)(T + 3 * d_inner_stride + cb) = Q6_V_hi_W(p_w13);
    }
}

// HVX transpose for general d_conv <= 32: channel-major raw VTCM -> tap-major T VTCM
స్థిర అంతఃవరుస శూన్య hvx_ssm_conv_unpack_గమ్యం_T_gen(స్థిరరాశి దశాంశం * raw, దశాంశం * T, uint32_t d_inner_చొప్పున_thread, uint32_t d_inner_stride, uint32_t d_conv) {
    uint32_t __attribute__((aligned(VLEN))) ముసుగు_buf[VLEN_FP32] = { 0 };
    కొరకు (uint32_t j = 0; j < d_conv; ++j) {
        ముసుగు_buf[j] = 0xFFFFFFFF;
    }
    స్థిరరాశి HVX_సదిశ ముసుగు = *(స్థిరరాశి HVX_సదిశ *) ముసుగు_buf;

    కొరకు (uint32_t cb = 0; cb < d_inner_చొప్పున_thread; cb += VLEN_FP32) {
        స్థిరరాశి uint32_t cb_n = MIN(VLEN_FP32, d_inner_చొప్పున_thread - cb);
        HVX_సదిశ sub[32];
        కొరకు (uint32_t r = 0; r < cb_n; ++r) {
            స్థిరరాశి దశాంశం * ch_ptr = raw + (cb + r) * d_conv;
            sub[r] = Q6_V_vand_VV(*(స్థిరరాశి HVX_UVector *) ch_ptr, ముసుగు);
        }
        కొరకు (uint32_t r = cb_n; r < 32; ++r) {
            sub[r] = hvx_vec_splat_f32(0.0f);
        }

        hvx_transpose_32x32_f32(sub);

        కొరకు (uint32_t j = 0; j < d_conv; ++j) {
            *(HVX_సదిశ *)(T + j * d_inner_stride + cb) = sub[j];
        }
    }
}

స్థిర అంతఃవరుస శూన్య hvx_ssm_conv_unpack_గమ్యం_T(స్థిరరాశి దశాంశం * raw, దశాంశం * T, uint32_t d_inner_చొప్పున_thread, uint32_t d_inner_stride, uint32_t d_conv) {
    ఒకవేళ (d_conv == 4 && (d_inner_చొప్పున_thread % VLEN_FP32 == 0)) {
        hvx_ssm_conv_unpack_గమ్యం_T_4(raw, T, d_inner_చొప్పున_thread, d_inner_stride);
    } లేకపోతే {
        hvx_ssm_conv_unpack_గమ్యం_T_gen(raw, T, d_inner_చొప్పున_thread, d_inner_stride, d_conv);
    }
}

// Decode dot product specialization for d_conv == 4: multiply in the raw channel-major layout,
// then deinterleave the products so each vector holds one tap of 32 channels, and sum.
// Keeps both operands in DMA layout - no transpose, no scratch.
స్థిర అంతఃవరుస శూన్య hvx_ssm_conv_decode_4(స్థిరరాశి దశాంశం * x, స్థిరరాశి దశాంశం * w, దశాంశం * బహిర్గతం, uint32_t n_ch) {
    కొరకు (uint32_t cb = 0; cb < n_ch; cb += VLEN_FP32) {
        స్థిరరాశి దశాంశం * xp = x + cb * 4;
        స్థిరరాశి దశాంశం * wp = w + cb * 4;

        HVX_సదిశ p0 = Q6_Vqf32_vmpy_VsfVsf(*(స్థిరరాశి HVX_సదిశ *)(xp +  0), *(స్థిరరాశి HVX_సదిశ *)(wp +  0));
        HVX_సదిశ p1 = Q6_Vqf32_vmpy_VsfVsf(*(స్థిరరాశి HVX_సదిశ *)(xp + 32), *(స్థిరరాశి HVX_సదిశ *)(wp + 32));
        HVX_సదిశ p2 = Q6_Vqf32_vmpy_VsfVsf(*(స్థిరరాశి HVX_సదిశ *)(xp + 64), *(స్థిరరాశి HVX_సదిశ *)(wp + 64));
        HVX_సదిశ p3 = Q6_Vqf32_vmpy_VsfVsf(*(స్థిరరాశి HVX_సదిశ *)(xp + 96), *(స్థిరరాశి HVX_సదిశ *)(wp + 96));

        HVX_సదిశ_Pair p01 = Q6_W_vdeal_VVR(p1, p0, -4);
        HVX_సదిశ_Pair p23 = Q6_W_vdeal_VVR(p3, p2, -4);

        HVX_సదిశ_Pair q02 = Q6_W_vdeal_VVR(Q6_V_lo_W(p23), Q6_V_lo_W(p01), -4);
        HVX_సదిశ_Pair q13 = Q6_W_vdeal_VVR(Q6_V_hi_W(p23), Q6_V_hi_W(p01), -4);

        HVX_సదిశ a = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(q02), Q6_V_lo_W(q13));
        HVX_సదిశ b = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_hi_W(q02), Q6_V_hi_W(q13));

        *(HVX_సదిశ *)(బహిర్గతం + cb) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(a, b));
    }
}

// Transpose src0 for prefill: one 32-channel block {32, ncs} (VTCM) -> T {ncs, 32} (VTCM).
// One VTCM gather per output row: lane r picks channel cb+r, the region bound drops
// the lanes past the channel tail.
స్థిర అంతఃవరుస శూన్య hvx_ssm_conv_transpose_ఖండం(స్థిరరాశి దశాంశం * raw_ఖండం,
                                                దశాంశం *       T,
                                                uint32_t      ncs,
                                                uint32_t      cb_n,
                                                HVX_సదిశ    vv) {
    స్థిరరాశి size_t   పునాది = (size_t) raw_ఖండం;
    స్థిరరాశి uint32_t mu   = cb_n * ncs * పరిమాణం(దశాంశం) - 1;

    కొరకు (uint32_t t = 0; t < ncs; ++t) {
        Q6_vgather_ARMVw((HVX_సదిశ *) (T + (size_t) t * VLEN_FP32), పునాది + t * పరిమాణం(దశాంశం), mu, vv);
    }
}

// Single-token decode worker (n_t == 1)
స్థిర శూన్య ssm_conv_thread_f32_decode(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    నిర్మాణం htp_ssm_conv_context *             scctx   = (నిర్మాణం htp_ssm_conv_context *) దత్తాంశం;
    నిర్మాణం htp_ops_context *                  octx    = scctx->octx;
    స్థిరరాశి నిర్మాణం htp_ssm_conv_kernel_params * kparams = scctx->kparams;

    స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత src0 = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత src1 = octx->src[1];
    స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత dst  = octx->dst;

    dma_వరుస * dma_q = octx->ctx->dma[ith];

    స్థిరరాశి uint32_t d_conv  = kparams->d_conv;
    స్థిరరాశి uint32_t d_inner = kparams->d_inner;
    స్థిరరాశి uint32_t n_s     = kparams->n_s;

    స్థిరరాశి uint32_t dr  = scctx->nrows_చొప్పున_thread;
    స్థిరరాశి uint32_t ir0 = scctx->అడ్డువరుస_ప్రారంభం + dr * ith;
    స్థిరరాశి uint32_t ir1 = MIN(ir0 + dr, scctx->అడ్డువరుస_ప్రారంభం + scctx->nrows);

    ఒకవేళ (ir0 >= ir1) {
        తిరిగి;
    }

    స్థిరరాశి uint32_t d_inner_చొప్పున_thread = ir1 - ir0;
    స్థిరరాశి uint32_t d_inner_stride     = hex_విడత_పైకి(d_inner_చొప్పున_thread, VLEN_FP32);
    స్థిరరాశి uint32_t d_inner_tile       = scctx->d_inner_tile;

    స్థిరరాశి size_t src0_stride_inner_బైట్లు = src0->nb[1];
    స్థిరరాశి size_t src0_stride_seq_బైట్లు   = src0->nb[2];
    స్థిరరాశి size_t dst_stride_seq_బైట్లు    = dst->nb[2];

    uint8_t * src1_spad_పునాది = octx->src1_spad.దత్తాంశం + ith * octx->src1_spad.కొలత_చొప్పున_thread;
    uint8_t * src0_spad_పునాది = octx->src0_spad.దత్తాంశం + ith * octx->src0_spad.కొలత_చొప్పున_thread;
    uint8_t * dst_spad_పునాది  = octx->dst_spad.దత్తాంశం  + ith * octx->dst_spad.కొలత_చొప్పున_thread;

    స్థిరరాశి size_t బరువు_బైట్లు    = (size_t) d_inner_చొప్పున_thread * d_conv * పరిమాణం(దశాంశం);
    స్థిరరాశి size_t బరువు_raw_కొలత = hex_విడత_పైకి(బరువు_బైట్లు, 128);

    దశాంశం * src1_raw = (దశాంశం *) src1_spad_పునాది;
    దశాంశం * src1_T   = (దశాంశం *) (src1_spad_పునాది + బరువు_raw_కొలత);

    స్థిరరాశి size_t src0_tile_raw_బైట్లు = hex_విడత_పైకి(d_inner_tile * d_conv * పరిమాణం(దశాంశం), 128);
    స్థిరరాశి size_t dst_tile_బైట్లు      = hex_విడత_పైకి(d_inner_tile * పరిమాణం(దశాంశం), 128);

    దశాంశం * src0_tile_raw[2] = { (దశాంశం *) src0_spad_పునాది, (దశాంశం *) (src0_spad_పునాది + src0_tile_raw_బైట్లు) };
    దశాంశం * src0_T           = (దశాంశం *) (src0_spad_పునాది + 2 * src0_tile_raw_బైట్లు);

    దశాంశం * dst_tile[2] = { (దశాంశం *) dst_spad_పునాది, (దశాంశం *) (dst_spad_పునాది + dst_tile_బైట్లు) };

    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];

    // raw_dot keeps both operands in the DMA layout, so src1 needs no prep pass
    స్థిరరాశి bool raw_dot = (d_conv == 4) && (d_inner_చొప్పున_thread % VLEN_FP32 == 0);

    స్థిరరాశి uint32_t n_tiles   = (d_inner_చొప్పున_thread + d_inner_tile - 1) / d_inner_tile;
    స్థిరరాశి uint32_t n_chunks  = n_s * n_tiles;
    స్థిరరాశి size_t   అడ్డువరుస_బైట్లు = d_conv * పరిమాణం(దశాంశం);

    uint32_t s_fetch        = 0;
    uint32_t tile_off_fetch = 0;

    #నిర్వచించు SSM_CONV_DECODE_నెట్టు_FETCH(c)                                                         \
        చేయి {                                                                                      \
            స్థిరరాశి uint32_t   cur_tile_n = MIN(d_inner_tile, d_inner_చొప్పున_thread - tile_off_fetch); \
            స్థిరరాశి dma_addr_t fetch_ddr  = src0->దత్తాంశం + s_fetch * src0_stride_seq_బైట్లు +          \
                                          (ir0 + tile_off_fetch) * src0_stride_inner_బైట్లు;       \
            dma_వరుస_నెట్టు(dma_q,                                                                 \
                           dma_make_దత్తాంశం((uint8_t *) src0_tile_raw[(c) & 1], fetch_ddr),          \
                           అడ్డువరుస_బైట్లు, src0_stride_inner_బైట్లు, అడ్డువరుస_బైట్లు, cur_tile_n);            \
            tile_off_fetch += d_inner_tile;                                                       \
            ఒకవేళ (tile_off_fetch >= d_inner_చొప్పున_thread) {                                           \
                tile_off_fetch = 0;                                                               \
                s_fetch++;                                                                        \
            }                                                                                     \
        } అయితే (0)

    // Queue weights and initial input tiles together so DDR reads overlap
    స్థిరరాశి dma_addr_t src1_ddr = src1->దత్తాంశం + ir0 * d_conv * పరిమాణం(దశాంశం);
    dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం((uint8_t *) src1_raw, src1_ddr), బరువు_బైట్లు, బరువు_బైట్లు, బరువు_బైట్లు, 1);

    SSM_CONV_DECODE_నెట్టు_FETCH(0);
    ఒకవేళ (n_chunks > 1) {
        SSM_CONV_DECODE_నెట్టు_FETCH(1);
    }

    dma_వరుస_తీయి(dma_q);  // weights

    ఒకవేళ (!raw_dot) {
        // Unpack/transpose src1_raw into src1_T {d_conv, d_inner_stride}
        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);
        hvx_ssm_conv_unpack_గమ్యం_T(src1_raw, src1_T, d_inner_చొప్పున_thread, d_inner_stride, d_conv);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);
    }

    uint32_t i3       = 0;
    uint32_t tile_off = 0;

    కొరకు (uint32_t c = 0; c < n_chunks; ++c) {
        స్థిరరాశి uint32_t tile_n = MIN(d_inner_tile, d_inner_చొప్పున_thread - tile_off);

        ఒకవేళ (c >= 2) {
            dma_వరుస_తీయి(dma_q);  // writeback of chunk c-2, frees dst_tile[c & 1]
        }
        dma_వరుస_తీయి(dma_q);      // fetch chunk c

        దశాంశం * పరిమిత బహిర్గతం = dst_tile[c & 1];

        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i3);
        ఒకవేళ (raw_dot) {
            స్థిరరాశి దశాంశం * xp = src0_tile_raw[c & 1];
            స్థిరరాశి దశాంశం * wp = src1_raw + tile_off * 4;
            hvx_ssm_conv_decode_4(xp, wp, బహిర్గతం, tile_n);
        } లేకపోతే {
            స్థిరరాశి uint32_t tile_stride = hex_విడత_పైకి(tile_n, VLEN_FP32);
            hvx_ssm_conv_unpack_గమ్యం_T(src0_tile_raw[c & 1], src0_T, tile_n, tile_stride, d_conv);

            కొరకు (uint32_t cb = 0; cb < tile_n; cb += VLEN_FP32) {
                స్థిరరాశి uint32_t cb_n = MIN(VLEN_FP32, tile_n - cb);
                HVX_సదిశ acc = hvx_vec_splat_f32(0.0f);
                కొరకు (uint32_t j = 0; j < d_conv; ++j) {
                    HVX_సదిశ x = *(స్థిరరాశి HVX_సదిశ *)(src0_T + j * tile_stride + cb);
                    HVX_సదిశ w = *(స్థిరరాశి HVX_సదిశ *)(src1_T + j * d_inner_stride + tile_off + cb);
                    acc          = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(x, w));
                }
                HVX_సదిశ y = Q6_Vsf_equals_Vqf32(acc);
                ఒకవేళ (cb_n == VLEN_FP32) {
                    *(HVX_సదిశ *)(బహిర్గతం + cb) = y;
                } లేకపోతే {
                    hvx_vec_store_u(బహిర్గతం + cb, cb_n * పరిమాణం(దశాంశం), y);
                }
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i3);

        స్థిరరాశి dma_addr_t dst_ddr        = dst->దత్తాంశం + i3 * dst_stride_seq_బైట్లు + (ir0 + tile_off) * పరిమాణం(దశాంశం);
        స్థిరరాశి size_t     tile_బహిర్గతం_బైట్లు = (size_t) tile_n * పరిమాణం(దశాంశం);
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(dst_ddr, (uint8_t *) బహిర్గతం),
                       tile_బహిర్గతం_బైట్లు, tile_బహిర్గతం_బైట్లు, tile_బహిర్గతం_బైట్లు, 1);

        ఒకవేళ (c + 2 < n_chunks) {
            SSM_CONV_DECODE_నెట్టు_FETCH(c + 2);
        }

        tile_off += d_inner_tile;
        ఒకవేళ (tile_off >= d_inner_చొప్పున_thread) {
            tile_off = 0;
            i3++;
        }
    }

    కొరకు (uint32_t k = MIN(n_chunks, 2); k > 0; --k) {
        dma_వరుస_తీయి(dma_q);  // drain the last writebacks
    }

    #నిర్వచనం_రద్దు SSM_CONV_DECODE_నెట్టు_FETCH

    FARF(ఎగువ_2, "ssm-conv-f32-decode %d/%d: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0], dst->ne[1],
         dst->ne[2], dst->ne[3]);
}

// Multi-token prefill worker (n_t > 1)
స్థిర శూన్య ssm_conv_thread_f32_prefill(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    నిర్మాణం htp_ssm_conv_context *             scctx   = (నిర్మాణం htp_ssm_conv_context *) దత్తాంశం;
    నిర్మాణం htp_ops_context *                  octx    = scctx->octx;
    స్థిరరాశి నిర్మాణం htp_ssm_conv_kernel_params * kparams = scctx->kparams;

    స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత src0 = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత src1 = octx->src[1];
    స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత dst  = octx->dst;

    dma_వరుస * dma_q = octx->ctx->dma[ith];

    స్థిరరాశి uint32_t d_conv  = kparams->d_conv;
    స్థిరరాశి uint32_t d_inner = kparams->d_inner;
    స్థిరరాశి uint32_t n_t     = kparams->n_t;
    స్థిరరాశి uint32_t n_s     = kparams->n_s;
    స్థిరరాశి uint32_t ncs     = src0->ne[0];

    స్థిరరాశి uint32_t dr  = scctx->nrows_చొప్పున_thread;
    స్థిరరాశి uint32_t ir0 = scctx->అడ్డువరుస_ప్రారంభం + dr * ith;
    స్థిరరాశి uint32_t ir1 = MIN(ir0 + dr, scctx->అడ్డువరుస_ప్రారంభం + scctx->nrows);

    ఒకవేళ (ir0 >= ir1) {
        తిరిగి;
    }

    స్థిరరాశి uint32_t d_inner_చొప్పున_thread = ir1 - ir0;
    స్థిరరాశి uint32_t d_inner_stride     = hex_విడత_పైకి(d_inner_చొప్పున_thread, VLEN_FP32);
    స్థిరరాశి uint32_t d_inner_tile       = scctx->d_inner_tile;

    స్థిరరాశి size_t src0_stride_inner_బైట్లు = src0->nb[1];
    స్థిరరాశి size_t src0_stride_seq_బైట్లు   = src0->nb[2];
    స్థిరరాశి size_t dst_stride_token_బైట్లు  = dst->nb[1];
    స్థిరరాశి size_t dst_stride_seq_బైట్లు    = dst->nb[2];

    uint8_t * src1_spad_పునాది = octx->src1_spad.దత్తాంశం + ith * octx->src1_spad.కొలత_చొప్పున_thread;
    uint8_t * src0_spad_పునాది = octx->src0_spad.దత్తాంశం + ith * octx->src0_spad.కొలత_చొప్పున_thread;
    uint8_t * dst_spad_పునాది  = octx->dst_spad.దత్తాంశం  + ith * octx->dst_spad.కొలత_చొప్పున_thread;

    స్థిరరాశి size_t బరువు_బైట్లు    = (size_t) d_inner_చొప్పున_thread * d_conv * పరిమాణం(దశాంశం);
    స్థిరరాశి size_t బరువు_raw_కొలత = hex_విడత_పైకి(బరువు_బైట్లు, 128);

    దశాంశం * src1_raw = (దశాంశం *) src1_spad_పునాది;
    దశాంశం * src1_T   = (దశాంశం *) (src1_spad_పునాది + బరువు_raw_కొలత);

    // src0 spad holds two raw tiles (fetch of tile n+1 overlaps compute of tile n) plus
    // the transposed block. dst spad holds two tiles so a writeback can stay in flight.
    స్థిరరాశి size_t src0_tile_raw_బైట్లు = hex_విడత_పైకి(d_inner_tile * ncs * పరిమాణం(దశాంశం), 128);
    స్థిరరాశి size_t dst_tile_బైట్లు      = hex_విడత_పైకి(d_inner_tile * n_t * పరిమాణం(దశాంశం), 128);

    దశాంశం * src0_tile_raw[2] = { (దశాంశం *) src0_spad_పునాది, (దశాంశం *) (src0_spad_పునాది + src0_tile_raw_బైట్లు) };
    దశాంశం * src0_T           = (దశాంశం *) (src0_spad_పునాది + 2 * src0_tile_raw_బైట్లు);

    దశాంశం * dst_tile[2] = { (దశాంశం *) dst_spad_పునాది, (దశాంశం *) (dst_spad_పునాది + dst_tile_బైట్లు) };

    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];

    స్థిరరాశి uint32_t n_tiles   = (d_inner_చొప్పున_thread + d_inner_tile - 1) / d_inner_tile;
    స్థిరరాశి uint32_t n_chunks  = n_s * n_tiles;
    స్థిరరాశి size_t   అడ్డువరుస_బైట్లు = ncs * పరిమాణం(దశాంశం);

    uint32_t s_fetch        = 0;
    uint32_t tile_off_fetch = 0;

    // Chunk c fetches into src0_tile_raw[c & 1] and writes back from dst_tile[c & 1].
    // Two fetches run ahead, so the queue order is F0 F1 W0 F2 W1 ... and pops follow it.
    #నిర్వచించు SSM_CONV_నెట్టు_FETCH(c)                                                                \
        చేయి {                                                                                      \
            స్థిరరాశి uint32_t   cur_tile_n = MIN(d_inner_tile, d_inner_చొప్పున_thread - tile_off_fetch); \
            స్థిరరాశి dma_addr_t fetch_ddr  = src0->దత్తాంశం + s_fetch * src0_stride_seq_బైట్లు +          \
                                          (ir0 + tile_off_fetch) * src0_stride_inner_బైట్లు;       \
            dma_వరుస_నెట్టు(dma_q,                                                                 \
                           dma_make_దత్తాంశం((uint8_t *) src0_tile_raw[(c) & 1], fetch_ddr),          \
                           అడ్డువరుస_బైట్లు, src0_stride_inner_బైట్లు, అడ్డువరుస_బైట్లు, cur_tile_n);            \
            tile_off_fetch += d_inner_tile;                                                       \
            ఒకవేళ (tile_off_fetch >= d_inner_చొప్పున_thread) {                                           \
                tile_off_fetch = 0;                                                               \
                s_fetch++;                                                                        \
            }                                                                                     \
        } అయితే (0)

    // Queue weights and initial input tiles together so DDR reads overlap
    స్థిరరాశి dma_addr_t src1_ddr = src1->దత్తాంశం + ir0 * d_conv * పరిమాణం(దశాంశం);
    dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం((uint8_t *) src1_raw, src1_ddr), బరువు_బైట్లు, బరువు_బైట్లు, బరువు_బైట్లు, 1);

    SSM_CONV_నెట్టు_FETCH(0);
    ఒకవేళ (n_chunks > 1) {
        SSM_CONV_నెట్టు_FETCH(1);
    }

    dma_వరుస_తీయి(dma_q);  // weights

    // Unpack/transpose src1_raw into src1_T {d_conv, d_inner_stride}
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);
    hvx_ssm_conv_unpack_గమ్యం_T(src1_raw, src1_T, d_inner_చొప్పున_thread, d_inner_stride, d_conv);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);

    స్థిరరాశి uint32_t C_TILE = VLEN_FP32;

    // gather offsets: lane r reads channel r of the raw tile block
    uint32_t __attribute__((aligned(VLEN))) gather_off[VLEN_FP32];
    కొరకు (uint32_t r = 0; r < VLEN_FP32; ++r) {
        gather_off[r] = r * ncs * పరిమాణం(దశాంశం);
    }
    స్థిరరాశి HVX_సదిశ vv = *(స్థిరరాశి HVX_సదిశ *) gather_off;

    uint32_t i3       = 0;
    uint32_t tile_off = 0;

    కొరకు (uint32_t c = 0; c < n_chunks; ++c) {
        స్థిరరాశి uint32_t tile_n = MIN(d_inner_tile, d_inner_చొప్పున_thread - tile_off);

        స్థిరరాశి దశాంశం * పరిమిత raw = src0_tile_raw[c & 1];
        దశాంశం * పరిమిత       బహిర్గతం = dst_tile[c & 1];

        ఒకవేళ (c >= 2) {
            dma_వరుస_తీయి(dma_q);  // writeback of chunk c-2, frees dst_tile[c & 1]
        }
        dma_వరుస_తీయి(dma_q);      // fetch chunk c

        // Channel block outer, token inner: the taps and the sliding window stay in
        // registers, so a new output row costs one src0_T load.
        స్థిరరాశి uint32_t dst_tile_stride = hex_విడత_పైకి(tile_n, C_TILE);

        కొరకు (uint32_t cb = 0; cb < tile_n; cb += C_TILE) {
            స్థిరరాశి uint32_t cb_n = MIN(C_TILE, tile_n - cb);

            htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) tile_off);
            hvx_ssm_conv_transpose_ఖండం(raw + (size_t) cb * ncs, src0_T, ncs, cb_n, vv);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) tile_off);

            స్థిరరాశి దశాంశం * పరిమిత wp = src1_T + tile_off + cb;
            దశాంశం * పరిమిత       op = బహిర్గతం + cb;

            htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) tile_off);
            ఒకవేళ (d_conv == 4) {
                స్థిరరాశి HVX_సదిశ w0 = *(స్థిరరాశి HVX_సదిశ *) (wp);
                స్థిరరాశి HVX_సదిశ w1 = *(స్థిరరాశి HVX_సదిశ *) (wp + d_inner_stride);
                స్థిరరాశి HVX_సదిశ w2 = *(స్థిరరాశి HVX_సదిశ *) (wp + 2 * d_inner_stride);
                స్థిరరాశి HVX_సదిశ w3 = *(స్థిరరాశి HVX_సదిశ *) (wp + 3 * d_inner_stride);

                HVX_సదిశ x0 = *(స్థిరరాశి HVX_సదిశ *) (src0_T);
                HVX_సదిశ x1 = *(స్థిరరాశి HVX_సదిశ *) (src0_T + C_TILE);
                HVX_సదిశ x2 = *(స్థిరరాశి HVX_సదిశ *) (src0_T + 2 * C_TILE);

                కొరకు (uint32_t t = 0; t < n_t; ++t) {
                    స్థిరరాశి HVX_సదిశ x3 = *(స్థిరరాశి HVX_సదిశ *) (src0_T + (t + 3) * C_TILE);

                    HVX_సదిశ a = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(x0, w0),
                                                            Q6_Vqf32_vmpy_VsfVsf(x1, w1));
                    HVX_సదిశ b = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(x2, w2),
                                                            Q6_Vqf32_vmpy_VsfVsf(x3, w3));

                    *(HVX_సదిశ *) (op + t * dst_tile_stride) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(a, b));

                    x0 = x1;
                    x1 = x2;
                    x2 = x3;
                }
            } లేకపోతే {
                కొరకు (uint32_t t = 0; t < n_t; ++t) {
                    HVX_సదిశ acc = hvx_vec_splat_f32(0.0f);
                    కొరకు (uint32_t j = 0; j < d_conv; ++j) {
                        HVX_సదిశ x = *(స్థిరరాశి HVX_సదిశ *) (src0_T + (t + j) * C_TILE);
                        HVX_సదిశ w = *(స్థిరరాశి HVX_సదిశ *) (wp + j * d_inner_stride);
                        acc          = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(x, w));
                    }
                    *(HVX_సదిశ *) (op + t * dst_tile_stride) = Q6_Vsf_equals_Vqf32(acc);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) tile_off);
        }

        // Writeback dst_tile VTCM -> DDR via 2D DMA
        dma_వరుస_నెట్టు(dma_q,
                       dma_make_దత్తాంశం(dst->దత్తాంశం + i3 * dst_stride_seq_బైట్లు + (ir0 + tile_off) * పరిమాణం(దశాంశం),
                                     (uint8_t *) బహిర్గతం),
                       dst_stride_token_బైట్లు, dst_tile_stride * పరిమాణం(దశాంశం), tile_n * పరిమాణం(దశాంశం), n_t);

        ఒకవేళ (c + 2 < n_chunks) {
            SSM_CONV_నెట్టు_FETCH(c + 2);
        }

        tile_off += d_inner_tile;
        ఒకవేళ (tile_off >= d_inner_చొప్పున_thread) {
            tile_off = 0;
            i3++;
        }
    }

    కొరకు (uint32_t k = MIN(n_chunks, 2); k > 0; --k) {
        dma_వరుస_తీయి(dma_q);  // drain the last writebacks
    }

    #నిర్వచనం_రద్దు SSM_CONV_నెట్టు_FETCH

    FARF(ఎగువ_2, "ssm-conv-f32-prefill %d/%d: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0], dst->ne[1],
         dst->ne[2], dst->ne[3]);
}

పూర్ణాంకం op_ssm_conv_f32(నిర్మాణం htp_ops_context * octx) {
    స్థిరరాశి నిర్మాణం htp_tensor * src0 = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * src1 = octx->src[1];
    స్థిరరాశి నిర్మాణం htp_tensor * dst  = octx->dst;

    ఒకవేళ (src0->type != HTP_TYPE_F32 || src1->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    స్థిరరాశి నిర్మాణం htp_ssm_conv_kernel_params * kparams = (స్థిరరాశి నిర్మాణం htp_ssm_conv_kernel_params *) octx->kernel_params;


    ఒకవేళ (!htp_ops_context_సమితి_n_దారాలు(octx, kparams->n_దారాలు)) {
        తిరిగి HTP_స్థితి_INVAL_PARAMS;
    }

    uint32_t అడ్డువరుస_ప్రారంభం = 0;
    uint32_t nrows     = kparams->d_inner;

    ఒకవేళ (octx->ctx->mdev.లెక్క > 1) {
        స్థిరరాశి uint32_t elems_చొప్పున_chunk = VLEN_FP32;
        స్థిరరాశి నిర్మాణం htp_tensor_mdev_range range = htp_tensor_mdev_విభజన(
            kparams->d_inner,
            htp_tensor_mdev_దత్తాంశం_aligned(dst) ? elems_చొప్పున_chunk : 0,
            octx->ctx->mdev.idx,
            octx->ctx->mdev.లెక్క,
            &octx->ctx->mdev.లెక్క_div
        );
        అడ్డువరుస_ప్రారంభం = range.ప్రారంభం;
        nrows     = range.లెక్క;
    }

    ఒకవేళ (nrows == 0) {
        తిరిగి HTP_స్థితి_OK;
    }

    ఒకవేళ (kparams->vtcm_కొలత > octx->ctx->vtcm_కొలత) {
        తిరిగి HTP_స్థితి_VTCM_TOO_SMALL;
    }

    స్థిరరాశి uint32_t n_దారాలు = octx->n_దారాలు;

    octx->src0_spad.కొలత_చొప్పున_thread = kparams->vtcm_src0_కొలత_చొప్పున_thread;
    octx->src1_spad.కొలత_చొప్పున_thread = kparams->vtcm_src1_కొలత_చొప్పున_thread;
    octx->dst_spad.కొలత_చొప్పున_thread  = kparams->vtcm_dst_కొలత_చొప్పున_thread;

    octx->src0_spad.కొలత = kparams->vtcm_src0_కొలత;
    octx->src1_spad.కొలత = kparams->vtcm_src1_కొలత;
    octx->dst_spad.కొలత  = kparams->vtcm_dst_కొలత;

    octx->src0_spad.దత్తాంశం = octx->ctx->vtcm_పునాది;
    octx->src1_spad.దత్తాంశం = octx->src0_spad.దత్తాంశం + octx->src0_spad.కొలత;
    octx->dst_spad.దత్తాంశం  = octx->src1_spad.దత్తాంశం + octx->src1_spad.కొలత;
    octx->src0_spad.src  = ఖాళీ;
    octx->src1_spad.src  = ఖాళీ;
    octx->dst_spad.src   = ఖాళీ;

    స్థిరరాశి uint32_t raw_rpt            = fastdiv(nrows + n_దారాలు - 1, &octx->n_దారాలు_div);
    స్థిరరాశి uint32_t d_inner_చొప్పున_thread = hex_విడత_పైకి(raw_rpt, VLEN_FP32);

    నిర్మాణం htp_ssm_conv_context scctx = {
        .octx             = octx,
        .kparams          = kparams,
        .nrows_చొప్పున_thread = d_inner_చొప్పున_thread,
        .d_inner_tile     = kparams->d_inner_tile,
        .అడ్డువరుస_ప్రారంభం        = అడ్డువరుస_ప్రారంభం,
        .nrows            = nrows,
    };

    FARF(ఎగువ_2, "ssm-conv-f32: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : mode %s\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         kparams->n_t == 1 ? "decode" : "prefill");

    ఒకవేళ (kparams->n_t == 1) {
        work_వరుస_నిరంతరం(octx->ctx->work_వరుస, ssm_conv_thread_f32_decode, &scctx, n_దారాలు);
    } లేకపోతే {
        work_వరుస_నిరంతరం(octx->ctx->work_వరుస, ssm_conv_thread_f32_prefill, &scctx, n_దారాలు);
    }

    తిరిగి HTP_స్థితి_OK;
}

పూర్ణాంకం op_ssm_conv(నిర్మాణం htp_ops_context * octx) {
    స్థిరరాశి నిర్మాణం htp_tensor * dst = octx->dst;

    ఎంపిక (dst->type) {
        స్థితి HTP_TYPE_F32:
            తిరిగి op_ssm_conv_f32(octx);
        అప్రమేయ:
            తిరిగి HTP_స్థితి_NO_SUPPORT;
    }
}
