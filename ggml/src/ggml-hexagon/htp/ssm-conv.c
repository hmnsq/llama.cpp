#ہدایت clang diagnostic ignored "-Wunused-variable"
#ہدایت clang diagnostic ignored "-Wunused-function"
#ہدایت clang diagnostic ignored "-Wunused-but-set-variable"

#شامل <HAP_farf.h>
#شامل <HAP_mem.h>
#شامل <HAP_ps.h>
#شامل <hexagon_protos.h>
#شامل <hexagon_types.h>
#شامل <math.h>
#شامل <qurt_thread.h>
#شامل <string.h>

#تعریف_کرو GGML_COMMON_DECL_C
#شامل "ggml-common.h"
#شامل "htp-ctx.h"
#شامل "dma-queue.h"
#شامل "hex-profile.h"
#شامل "htp-ops.h"
#شامل "htp-tensor.h"
#شامل "hvx-utils.h"
#شامل "ssm-conv.h"

ساخت htp_ssm_conv_context {
    ساخت htp_ops_context *                  octx;
    ثابت ساخت htp_ssm_conv_kernel_params * kparams;
    uint32_t                                  nrows_فی_thread;
    uint32_t                                  d_inner_tile;
    uint32_t                                  صف_آغاز;
    uint32_t                                  nrows;
};

// In-register 32x32 fp32 transpose using std 5-stage HVX vshuff butterfly.
ساکن اندرونی خالی hvx_transpose_32x32_f32(HVX_سمتیہ m[32]) {
    HVX_سمتیہ tmp[32];

    // Stage 0 (R = -4): pair (2i, 2i+1) for i = 0..15. m -> tmp.
    برائے (صحیح i = 0; i < 16; ++i) {
        HVX_سمتیہ_Pair p = Q6_W_vshuff_VVR(m[2*i + 1], m[2*i], -4);
        tmp[2*i + 0] = Q6_V_lo_W(p);
        tmp[2*i + 1] = Q6_V_hi_W(p);
    }

    // Stage 1 (R = -8): per block of 4, pair (b+0, b+2) and (b+1, b+3). tmp -> m.
    برائے (صحیح b = 0; b < 32; b += 4) {
        HVX_سمتیہ_Pair p0 = Q6_W_vshuff_VVR(tmp[b + 2], tmp[b + 0], -8);
        HVX_سمتیہ_Pair p1 = Q6_W_vshuff_VVR(tmp[b + 3], tmp[b + 1], -8);
        m[b + 0] = Q6_V_lo_W(p0); m[b + 1] = Q6_V_hi_W(p0);
        m[b + 2] = Q6_V_lo_W(p1); m[b + 3] = Q6_V_hi_W(p1);
    }

    // Stage 2 (R = -16): per block of 8, pair (b+i, b+i+4) for i = 0..3. m -> tmp.
    برائے (صحیح b = 0; b < 32; b += 8) {
        برائے (صحیح i = 0; i < 4; ++i) {
            HVX_سمتیہ_Pair p = Q6_W_vshuff_VVR(m[b + i + 4], m[b + i], -16);
            tmp[b + 2*i + 0] = Q6_V_lo_W(p);
            tmp[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }

    // Stage 3 (R = -32): per block of 16, pair (b+i, b+i+8) for i = 0..7. tmp -> m.
    برائے (صحیح b = 0; b < 32; b += 16) {
        برائے (صحیح i = 0; i < 8; ++i) {
            HVX_سمتیہ_Pair p = Q6_W_vshuff_VVR(tmp[b + i + 8], tmp[b + i], -32);
            m[b + 2*i + 0] = Q6_V_lo_W(p);
            m[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }

    // Stage 4 (R = -64): pair (i, i+16) for i = 0..15. m -> tmp -> m.
    برائے (صحیح i = 0; i < 16; ++i) {
        HVX_سمتیہ_Pair p = Q6_W_vshuff_VVR(m[i + 16], m[i], -64);
        tmp[2 * i + 0]   = Q6_V_lo_W(p);
        tmp[2 * i + 1]   = Q6_V_hi_W(p);
    }

    برائے (صحیح i = 0; i < 32; ++i) {
        m[i] = tmp[i];
    }
}

// HVX deinterleave for d_conv == 4: channel-major raw VTCM -> tap-major T VTCM
ساکن اندرونی خالی hvx_ssm_conv_unpack_مقصد_T_4(ثابت اعشاری * raw, اعشاری * T, uint32_t d_inner_فی_thread, uint32_t d_inner_stride) {
    برائے (uint32_t cb = 0; cb < d_inner_فی_thread; cb += VLEN_FP32) {
        HVX_سمتیہ v0 = *(ثابت HVX_سمتیہ *)(raw + (cb + 0) * 4);
        HVX_سمتیہ v1 = *(ثابت HVX_سمتیہ *)(raw + (cb + 8) * 4);
        HVX_سمتیہ v2 = *(ثابت HVX_سمتیہ *)(raw + (cb + 16) * 4);
        HVX_سمتیہ v3 = *(ثابت HVX_سمتیہ *)(raw + (cb + 24) * 4);

        HVX_سمتیہ_Pair p01 = Q6_W_vdeal_VVR(v1, v0, -4);
        HVX_سمتیہ_Pair p23 = Q6_W_vdeal_VVR(v3, v2, -4);

        HVX_سمتیہ_Pair p_w02 = Q6_W_vdeal_VVR(Q6_V_lo_W(p23), Q6_V_lo_W(p01), -4);
        HVX_سمتیہ_Pair p_w13 = Q6_W_vdeal_VVR(Q6_V_hi_W(p23), Q6_V_hi_W(p01), -4);

        *(HVX_سمتیہ *)(T + 0 * d_inner_stride + cb) = Q6_V_lo_W(p_w02);
        *(HVX_سمتیہ *)(T + 1 * d_inner_stride + cb) = Q6_V_lo_W(p_w13);
        *(HVX_سمتیہ *)(T + 2 * d_inner_stride + cb) = Q6_V_hi_W(p_w02);
        *(HVX_سمتیہ *)(T + 3 * d_inner_stride + cb) = Q6_V_hi_W(p_w13);
    }
}

// HVX transpose for general d_conv <= 32: channel-major raw VTCM -> tap-major T VTCM
ساکن اندرونی خالی hvx_ssm_conv_unpack_مقصد_T_gen(ثابت اعشاری * raw, اعشاری * T, uint32_t d_inner_فی_thread, uint32_t d_inner_stride, uint32_t d_conv) {
    uint32_t __attribute__((aligned(VLEN))) نقاب_buf[VLEN_FP32] = { 0 };
    برائے (uint32_t j = 0; j < d_conv; ++j) {
        نقاب_buf[j] = 0xFFFFFFFF;
    }
    ثابت HVX_سمتیہ نقاب = *(ثابت HVX_سمتیہ *) نقاب_buf;

    برائے (uint32_t cb = 0; cb < d_inner_فی_thread; cb += VLEN_FP32) {
        ثابت uint32_t cb_n = MIN(VLEN_FP32, d_inner_فی_thread - cb);
        HVX_سمتیہ sub[32];
        برائے (uint32_t r = 0; r < cb_n; ++r) {
            ثابت اعشاری * ch_ptr = raw + (cb + r) * d_conv;
            sub[r] = Q6_V_vand_VV(*(ثابت HVX_UVector *) ch_ptr, نقاب);
        }
        برائے (uint32_t r = cb_n; r < 32; ++r) {
            sub[r] = hvx_vec_splat_f32(0.0f);
        }

        hvx_transpose_32x32_f32(sub);

        برائے (uint32_t j = 0; j < d_conv; ++j) {
            *(HVX_سمتیہ *)(T + j * d_inner_stride + cb) = sub[j];
        }
    }
}

ساکن اندرونی خالی hvx_ssm_conv_unpack_مقصد_T(ثابت اعشاری * raw, اعشاری * T, uint32_t d_inner_فی_thread, uint32_t d_inner_stride, uint32_t d_conv) {
    اگر (d_conv == 4 && (d_inner_فی_thread % VLEN_FP32 == 0)) {
        hvx_ssm_conv_unpack_مقصد_T_4(raw, T, d_inner_فی_thread, d_inner_stride);
    } ورنہ {
        hvx_ssm_conv_unpack_مقصد_T_gen(raw, T, d_inner_فی_thread, d_inner_stride, d_conv);
    }
}

// Decode dot product specialization for d_conv == 4: multiply in the raw channel-major layout,
// then deinterleave the products so each vector holds one tap of 32 channels, and sum.
// Keeps both operands in DMA layout - no transpose, no scratch.
ساکن اندرونی خالی hvx_ssm_conv_decode_4(ثابت اعشاری * x, ثابت اعشاری * w, اعشاری * برآمد, uint32_t n_ch) {
    برائے (uint32_t cb = 0; cb < n_ch; cb += VLEN_FP32) {
        ثابت اعشاری * xp = x + cb * 4;
        ثابت اعشاری * wp = w + cb * 4;

        HVX_سمتیہ p0 = Q6_Vqf32_vmpy_VsfVsf(*(ثابت HVX_سمتیہ *)(xp +  0), *(ثابت HVX_سمتیہ *)(wp +  0));
        HVX_سمتیہ p1 = Q6_Vqf32_vmpy_VsfVsf(*(ثابت HVX_سمتیہ *)(xp + 32), *(ثابت HVX_سمتیہ *)(wp + 32));
        HVX_سمتیہ p2 = Q6_Vqf32_vmpy_VsfVsf(*(ثابت HVX_سمتیہ *)(xp + 64), *(ثابت HVX_سمتیہ *)(wp + 64));
        HVX_سمتیہ p3 = Q6_Vqf32_vmpy_VsfVsf(*(ثابت HVX_سمتیہ *)(xp + 96), *(ثابت HVX_سمتیہ *)(wp + 96));

        HVX_سمتیہ_Pair p01 = Q6_W_vdeal_VVR(p1, p0, -4);
        HVX_سمتیہ_Pair p23 = Q6_W_vdeal_VVR(p3, p2, -4);

        HVX_سمتیہ_Pair q02 = Q6_W_vdeal_VVR(Q6_V_lo_W(p23), Q6_V_lo_W(p01), -4);
        HVX_سمتیہ_Pair q13 = Q6_W_vdeal_VVR(Q6_V_hi_W(p23), Q6_V_hi_W(p01), -4);

        HVX_سمتیہ a = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(q02), Q6_V_lo_W(q13));
        HVX_سمتیہ b = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_hi_W(q02), Q6_V_hi_W(q13));

        *(HVX_سمتیہ *)(برآمد + cb) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(a, b));
    }
}

// Transpose src0 for prefill: one 32-channel block {32, ncs} (VTCM) -> T {ncs, 32} (VTCM).
// One VTCM gather per output row: lane r picks channel cb+r, the region bound drops
// the lanes past the channel tail.
ساکن اندرونی خالی hvx_ssm_conv_transpose_بلاک(ثابت اعشاری * raw_بلاک,
                                                اعشاری *       T,
                                                uint32_t      ncs,
                                                uint32_t      cb_n,
                                                HVX_سمتیہ    vv) {
    ثابت size_t   بنیاد = (size_t) raw_بلاک;
    ثابت uint32_t mu   = cb_n * ncs * حجم(اعشاری) - 1;

    برائے (uint32_t t = 0; t < ncs; ++t) {
        Q6_vgather_ARMVw((HVX_سمتیہ *) (T + (size_t) t * VLEN_FP32), بنیاد + t * حجم(اعشاری), mu, vv);
    }
}

// Single-token decode worker (n_t == 1)
ساکن خالی ssm_conv_thread_f32_decode(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    ساخت htp_ssm_conv_context *             scctx   = (ساخت htp_ssm_conv_context *) ڈیٹا;
    ساخت htp_ops_context *                  octx    = scctx->octx;
    ثابت ساخت htp_ssm_conv_kernel_params * kparams = scctx->kparams;

    ثابت ساخت htp_tensor * محدود src0 = octx->src[0];
    ثابت ساخت htp_tensor * محدود src1 = octx->src[1];
    ثابت ساخت htp_tensor * محدود dst  = octx->dst;

    dma_قطار * dma_q = octx->ctx->dma[ith];

    ثابت uint32_t d_conv  = kparams->d_conv;
    ثابت uint32_t d_inner = kparams->d_inner;
    ثابت uint32_t n_s     = kparams->n_s;

    ثابت uint32_t dr  = scctx->nrows_فی_thread;
    ثابت uint32_t ir0 = scctx->صف_آغاز + dr * ith;
    ثابت uint32_t ir1 = MIN(ir0 + dr, scctx->صف_آغاز + scctx->nrows);

    اگر (ir0 >= ir1) {
        واپس;
    }

    ثابت uint32_t d_inner_فی_thread = ir1 - ir0;
    ثابت uint32_t d_inner_stride     = hex_باری_اوپر(d_inner_فی_thread, VLEN_FP32);
    ثابت uint32_t d_inner_tile       = scctx->d_inner_tile;

    ثابت size_t src0_stride_inner_بائٹیں = src0->nb[1];
    ثابت size_t src0_stride_seq_بائٹیں   = src0->nb[2];
    ثابت size_t dst_stride_seq_بائٹیں    = dst->nb[2];

    uint8_t * src1_spad_بنیاد = octx->src1_spad.ڈیٹا + ith * octx->src1_spad.سائز_فی_thread;
    uint8_t * src0_spad_بنیاد = octx->src0_spad.ڈیٹا + ith * octx->src0_spad.سائز_فی_thread;
    uint8_t * dst_spad_بنیاد  = octx->dst_spad.ڈیٹا  + ith * octx->dst_spad.سائز_فی_thread;

    ثابت size_t وزن_بائٹیں    = (size_t) d_inner_فی_thread * d_conv * حجم(اعشاری);
    ثابت size_t وزن_raw_سائز = hex_باری_اوپر(وزن_بائٹیں, 128);

    اعشاری * src1_raw = (اعشاری *) src1_spad_بنیاد;
    اعشاری * src1_T   = (اعشاری *) (src1_spad_بنیاد + وزن_raw_سائز);

    ثابت size_t src0_tile_raw_بائٹیں = hex_باری_اوپر(d_inner_tile * d_conv * حجم(اعشاری), 128);
    ثابت size_t dst_tile_بائٹیں      = hex_باری_اوپر(d_inner_tile * حجم(اعشاری), 128);

    اعشاری * src0_tile_raw[2] = { (اعشاری *) src0_spad_بنیاد, (اعشاری *) (src0_spad_بنیاد + src0_tile_raw_بائٹیں) };
    اعشاری * src0_T           = (اعشاری *) (src0_spad_بنیاد + 2 * src0_tile_raw_بائٹیں);

    اعشاری * dst_tile[2] = { (اعشاری *) dst_spad_بنیاد, (اعشاری *) (dst_spad_بنیاد + dst_tile_بائٹیں) };

    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];

    // raw_dot keeps both operands in the DMA layout, so src1 needs no prep pass
    ثابت bool raw_dot = (d_conv == 4) && (d_inner_فی_thread % VLEN_FP32 == 0);

    ثابت uint32_t n_tiles   = (d_inner_فی_thread + d_inner_tile - 1) / d_inner_tile;
    ثابت uint32_t n_chunks  = n_s * n_tiles;
    ثابت size_t   صف_بائٹیں = d_conv * حجم(اعشاری);

    uint32_t s_fetch        = 0;
    uint32_t tile_off_fetch = 0;

    #تعریف_کرو SSM_CONV_DECODE_ڈالو_FETCH(c)                                                         \
        کرو {                                                                                      \
            ثابت uint32_t   cur_tile_n = MIN(d_inner_tile, d_inner_فی_thread - tile_off_fetch); \
            ثابت dma_addr_t fetch_ddr  = src0->ڈیٹا + s_fetch * src0_stride_seq_بائٹیں +          \
                                          (ir0 + tile_off_fetch) * src0_stride_inner_بائٹیں;       \
            dma_قطار_ڈالو(dma_q,                                                                 \
                           dma_make_ڈیٹا((uint8_t *) src0_tile_raw[(c) & 1], fetch_ddr),          \
                           صف_بائٹیں, src0_stride_inner_بائٹیں, صف_بائٹیں, cur_tile_n);            \
            tile_off_fetch += d_inner_tile;                                                       \
            اگر (tile_off_fetch >= d_inner_فی_thread) {                                           \
                tile_off_fetch = 0;                                                               \
                s_fetch++;                                                                        \
            }                                                                                     \
        } جب (0)

    // Queue weights and initial input tiles together so DDR reads overlap
    ثابت dma_addr_t src1_ddr = src1->ڈیٹا + ir0 * d_conv * حجم(اعشاری);
    dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا((uint8_t *) src1_raw, src1_ddr), وزن_بائٹیں, وزن_بائٹیں, وزن_بائٹیں, 1);

    SSM_CONV_DECODE_ڈالو_FETCH(0);
    اگر (n_chunks > 1) {
        SSM_CONV_DECODE_ڈالو_FETCH(1);
    }

    dma_قطار_نکالو(dma_q);  // weights

    اگر (!raw_dot) {
        // Unpack/transpose src1_raw into src1_T {d_conv, d_inner_stride}
        htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);
        hvx_ssm_conv_unpack_مقصد_T(src1_raw, src1_T, d_inner_فی_thread, d_inner_stride, d_conv);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);
    }

    uint32_t i3       = 0;
    uint32_t tile_off = 0;

    برائے (uint32_t c = 0; c < n_chunks; ++c) {
        ثابت uint32_t tile_n = MIN(d_inner_tile, d_inner_فی_thread - tile_off);

        اگر (c >= 2) {
            dma_قطار_نکالو(dma_q);  // writeback of chunk c-2, frees dst_tile[c & 1]
        }
        dma_قطار_نکالو(dma_q);      // fetch chunk c

        اعشاری * محدود برآمد = dst_tile[c & 1];

        htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i3);
        اگر (raw_dot) {
            ثابت اعشاری * xp = src0_tile_raw[c & 1];
            ثابت اعشاری * wp = src1_raw + tile_off * 4;
            hvx_ssm_conv_decode_4(xp, wp, برآمد, tile_n);
        } ورنہ {
            ثابت uint32_t tile_stride = hex_باری_اوپر(tile_n, VLEN_FP32);
            hvx_ssm_conv_unpack_مقصد_T(src0_tile_raw[c & 1], src0_T, tile_n, tile_stride, d_conv);

            برائے (uint32_t cb = 0; cb < tile_n; cb += VLEN_FP32) {
                ثابت uint32_t cb_n = MIN(VLEN_FP32, tile_n - cb);
                HVX_سمتیہ acc = hvx_vec_splat_f32(0.0f);
                برائے (uint32_t j = 0; j < d_conv; ++j) {
                    HVX_سمتیہ x = *(ثابت HVX_سمتیہ *)(src0_T + j * tile_stride + cb);
                    HVX_سمتیہ w = *(ثابت HVX_سمتیہ *)(src1_T + j * d_inner_stride + tile_off + cb);
                    acc          = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(x, w));
                }
                HVX_سمتیہ y = Q6_Vsf_equals_Vqf32(acc);
                اگر (cb_n == VLEN_FP32) {
                    *(HVX_سمتیہ *)(برآمد + cb) = y;
                } ورنہ {
                    hvx_vec_store_u(برآمد + cb, cb_n * حجم(اعشاری), y);
                }
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i3);

        ثابت dma_addr_t dst_ddr        = dst->ڈیٹا + i3 * dst_stride_seq_بائٹیں + (ir0 + tile_off) * حجم(اعشاری);
        ثابت size_t     tile_برآمد_بائٹیں = (size_t) tile_n * حجم(اعشاری);
        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(dst_ddr, (uint8_t *) برآمد),
                       tile_برآمد_بائٹیں, tile_برآمد_بائٹیں, tile_برآمد_بائٹیں, 1);

        اگر (c + 2 < n_chunks) {
            SSM_CONV_DECODE_ڈالو_FETCH(c + 2);
        }

        tile_off += d_inner_tile;
        اگر (tile_off >= d_inner_فی_thread) {
            tile_off = 0;
            i3++;
        }
    }

    برائے (uint32_t k = MIN(n_chunks, 2); k > 0; --k) {
        dma_قطار_نکالو(dma_q);  // drain the last writebacks
    }

    #تعریف_منسوخ SSM_CONV_DECODE_ڈالو_FETCH

    FARF(بلند_2, "ssm-conv-f32-decode %d/%d: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0], dst->ne[1],
         dst->ne[2], dst->ne[3]);
}

// Multi-token prefill worker (n_t > 1)
ساکن خالی ssm_conv_thread_f32_prefill(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    ساخت htp_ssm_conv_context *             scctx   = (ساخت htp_ssm_conv_context *) ڈیٹا;
    ساخت htp_ops_context *                  octx    = scctx->octx;
    ثابت ساخت htp_ssm_conv_kernel_params * kparams = scctx->kparams;

    ثابت ساخت htp_tensor * محدود src0 = octx->src[0];
    ثابت ساخت htp_tensor * محدود src1 = octx->src[1];
    ثابت ساخت htp_tensor * محدود dst  = octx->dst;

    dma_قطار * dma_q = octx->ctx->dma[ith];

    ثابت uint32_t d_conv  = kparams->d_conv;
    ثابت uint32_t d_inner = kparams->d_inner;
    ثابت uint32_t n_t     = kparams->n_t;
    ثابت uint32_t n_s     = kparams->n_s;
    ثابت uint32_t ncs     = src0->ne[0];

    ثابت uint32_t dr  = scctx->nrows_فی_thread;
    ثابت uint32_t ir0 = scctx->صف_آغاز + dr * ith;
    ثابت uint32_t ir1 = MIN(ir0 + dr, scctx->صف_آغاز + scctx->nrows);

    اگر (ir0 >= ir1) {
        واپس;
    }

    ثابت uint32_t d_inner_فی_thread = ir1 - ir0;
    ثابت uint32_t d_inner_stride     = hex_باری_اوپر(d_inner_فی_thread, VLEN_FP32);
    ثابت uint32_t d_inner_tile       = scctx->d_inner_tile;

    ثابت size_t src0_stride_inner_بائٹیں = src0->nb[1];
    ثابت size_t src0_stride_seq_بائٹیں   = src0->nb[2];
    ثابت size_t dst_stride_token_بائٹیں  = dst->nb[1];
    ثابت size_t dst_stride_seq_بائٹیں    = dst->nb[2];

    uint8_t * src1_spad_بنیاد = octx->src1_spad.ڈیٹا + ith * octx->src1_spad.سائز_فی_thread;
    uint8_t * src0_spad_بنیاد = octx->src0_spad.ڈیٹا + ith * octx->src0_spad.سائز_فی_thread;
    uint8_t * dst_spad_بنیاد  = octx->dst_spad.ڈیٹا  + ith * octx->dst_spad.سائز_فی_thread;

    ثابت size_t وزن_بائٹیں    = (size_t) d_inner_فی_thread * d_conv * حجم(اعشاری);
    ثابت size_t وزن_raw_سائز = hex_باری_اوپر(وزن_بائٹیں, 128);

    اعشاری * src1_raw = (اعشاری *) src1_spad_بنیاد;
    اعشاری * src1_T   = (اعشاری *) (src1_spad_بنیاد + وزن_raw_سائز);

    // src0 spad holds two raw tiles (fetch of tile n+1 overlaps compute of tile n) plus
    // the transposed block. dst spad holds two tiles so a writeback can stay in flight.
    ثابت size_t src0_tile_raw_بائٹیں = hex_باری_اوپر(d_inner_tile * ncs * حجم(اعشاری), 128);
    ثابت size_t dst_tile_بائٹیں      = hex_باری_اوپر(d_inner_tile * n_t * حجم(اعشاری), 128);

    اعشاری * src0_tile_raw[2] = { (اعشاری *) src0_spad_بنیاد, (اعشاری *) (src0_spad_بنیاد + src0_tile_raw_بائٹیں) };
    اعشاری * src0_T           = (اعشاری *) (src0_spad_بنیاد + 2 * src0_tile_raw_بائٹیں);

    اعشاری * dst_tile[2] = { (اعشاری *) dst_spad_بنیاد, (اعشاری *) (dst_spad_بنیاد + dst_tile_بائٹیں) };

    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت uint32_t n_tiles   = (d_inner_فی_thread + d_inner_tile - 1) / d_inner_tile;
    ثابت uint32_t n_chunks  = n_s * n_tiles;
    ثابت size_t   صف_بائٹیں = ncs * حجم(اعشاری);

    uint32_t s_fetch        = 0;
    uint32_t tile_off_fetch = 0;

    // Chunk c fetches into src0_tile_raw[c & 1] and writes back from dst_tile[c & 1].
    // Two fetches run ahead, so the queue order is F0 F1 W0 F2 W1 ... and pops follow it.
    #تعریف_کرو SSM_CONV_ڈالو_FETCH(c)                                                                \
        کرو {                                                                                      \
            ثابت uint32_t   cur_tile_n = MIN(d_inner_tile, d_inner_فی_thread - tile_off_fetch); \
            ثابت dma_addr_t fetch_ddr  = src0->ڈیٹا + s_fetch * src0_stride_seq_بائٹیں +          \
                                          (ir0 + tile_off_fetch) * src0_stride_inner_بائٹیں;       \
            dma_قطار_ڈالو(dma_q,                                                                 \
                           dma_make_ڈیٹا((uint8_t *) src0_tile_raw[(c) & 1], fetch_ddr),          \
                           صف_بائٹیں, src0_stride_inner_بائٹیں, صف_بائٹیں, cur_tile_n);            \
            tile_off_fetch += d_inner_tile;                                                       \
            اگر (tile_off_fetch >= d_inner_فی_thread) {                                           \
                tile_off_fetch = 0;                                                               \
                s_fetch++;                                                                        \
            }                                                                                     \
        } جب (0)

    // Queue weights and initial input tiles together so DDR reads overlap
    ثابت dma_addr_t src1_ddr = src1->ڈیٹا + ir0 * d_conv * حجم(اعشاری);
    dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا((uint8_t *) src1_raw, src1_ddr), وزن_بائٹیں, وزن_بائٹیں, وزن_بائٹیں, 1);

    SSM_CONV_ڈالو_FETCH(0);
    اگر (n_chunks > 1) {
        SSM_CONV_ڈالو_FETCH(1);
    }

    dma_قطار_نکالو(dma_q);  // weights

    // Unpack/transpose src1_raw into src1_T {d_conv, d_inner_stride}
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);
    hvx_ssm_conv_unpack_مقصد_T(src1_raw, src1_T, d_inner_فی_thread, d_inner_stride, d_conv);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);

    ثابت uint32_t C_TILE = VLEN_FP32;

    // gather offsets: lane r reads channel r of the raw tile block
    uint32_t __attribute__((aligned(VLEN))) gather_off[VLEN_FP32];
    برائے (uint32_t r = 0; r < VLEN_FP32; ++r) {
        gather_off[r] = r * ncs * حجم(اعشاری);
    }
    ثابت HVX_سمتیہ vv = *(ثابت HVX_سمتیہ *) gather_off;

    uint32_t i3       = 0;
    uint32_t tile_off = 0;

    برائے (uint32_t c = 0; c < n_chunks; ++c) {
        ثابت uint32_t tile_n = MIN(d_inner_tile, d_inner_فی_thread - tile_off);

        ثابت اعشاری * محدود raw = src0_tile_raw[c & 1];
        اعشاری * محدود       برآمد = dst_tile[c & 1];

        اگر (c >= 2) {
            dma_قطار_نکالو(dma_q);  // writeback of chunk c-2, frees dst_tile[c & 1]
        }
        dma_قطار_نکالو(dma_q);      // fetch chunk c

        // Channel block outer, token inner: the taps and the sliding window stay in
        // registers, so a new output row costs one src0_T load.
        ثابت uint32_t dst_tile_stride = hex_باری_اوپر(tile_n, C_TILE);

        برائے (uint32_t cb = 0; cb < tile_n; cb += C_TILE) {
            ثابت uint32_t cb_n = MIN(C_TILE, tile_n - cb);

            htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) tile_off);
            hvx_ssm_conv_transpose_بلاک(raw + (size_t) cb * ncs, src0_T, ncs, cb_n, vv);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) tile_off);

            ثابت اعشاری * محدود wp = src1_T + tile_off + cb;
            اعشاری * محدود       op = برآمد + cb;

            htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) tile_off);
            اگر (d_conv == 4) {
                ثابت HVX_سمتیہ w0 = *(ثابت HVX_سمتیہ *) (wp);
                ثابت HVX_سمتیہ w1 = *(ثابت HVX_سمتیہ *) (wp + d_inner_stride);
                ثابت HVX_سمتیہ w2 = *(ثابت HVX_سمتیہ *) (wp + 2 * d_inner_stride);
                ثابت HVX_سمتیہ w3 = *(ثابت HVX_سمتیہ *) (wp + 3 * d_inner_stride);

                HVX_سمتیہ x0 = *(ثابت HVX_سمتیہ *) (src0_T);
                HVX_سمتیہ x1 = *(ثابت HVX_سمتیہ *) (src0_T + C_TILE);
                HVX_سمتیہ x2 = *(ثابت HVX_سمتیہ *) (src0_T + 2 * C_TILE);

                برائے (uint32_t t = 0; t < n_t; ++t) {
                    ثابت HVX_سمتیہ x3 = *(ثابت HVX_سمتیہ *) (src0_T + (t + 3) * C_TILE);

                    HVX_سمتیہ a = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(x0, w0),
                                                            Q6_Vqf32_vmpy_VsfVsf(x1, w1));
                    HVX_سمتیہ b = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(x2, w2),
                                                            Q6_Vqf32_vmpy_VsfVsf(x3, w3));

                    *(HVX_سمتیہ *) (op + t * dst_tile_stride) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(a, b));

                    x0 = x1;
                    x1 = x2;
                    x2 = x3;
                }
            } ورنہ {
                برائے (uint32_t t = 0; t < n_t; ++t) {
                    HVX_سمتیہ acc = hvx_vec_splat_f32(0.0f);
                    برائے (uint32_t j = 0; j < d_conv; ++j) {
                        HVX_سمتیہ x = *(ثابت HVX_سمتیہ *) (src0_T + (t + j) * C_TILE);
                        HVX_سمتیہ w = *(ثابت HVX_سمتیہ *) (wp + j * d_inner_stride);
                        acc          = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(x, w));
                    }
                    *(HVX_سمتیہ *) (op + t * dst_tile_stride) = Q6_Vsf_equals_Vqf32(acc);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) tile_off);
        }

        // Writeback dst_tile VTCM -> DDR via 2D DMA
        dma_قطار_ڈالو(dma_q,
                       dma_make_ڈیٹا(dst->ڈیٹا + i3 * dst_stride_seq_بائٹیں + (ir0 + tile_off) * حجم(اعشاری),
                                     (uint8_t *) برآمد),
                       dst_stride_token_بائٹیں, dst_tile_stride * حجم(اعشاری), tile_n * حجم(اعشاری), n_t);

        اگر (c + 2 < n_chunks) {
            SSM_CONV_ڈالو_FETCH(c + 2);
        }

        tile_off += d_inner_tile;
        اگر (tile_off >= d_inner_فی_thread) {
            tile_off = 0;
            i3++;
        }
    }

    برائے (uint32_t k = MIN(n_chunks, 2); k > 0; --k) {
        dma_قطار_نکالو(dma_q);  // drain the last writebacks
    }

    #تعریف_منسوخ SSM_CONV_ڈالو_FETCH

    FARF(بلند_2, "ssm-conv-f32-prefill %d/%d: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0], dst->ne[1],
         dst->ne[2], dst->ne[3]);
}

صحیح op_ssm_conv_f32(ساخت htp_ops_context * octx) {
    ثابت ساخت htp_tensor * src0 = octx->src[0];
    ثابت ساخت htp_tensor * src1 = octx->src[1];
    ثابت ساخت htp_tensor * dst  = octx->dst;

    اگر (src0->type != HTP_TYPE_F32 || src1->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        واپس HTP_حالت_NO_SUPPORT;
    }

    ثابت ساخت htp_ssm_conv_kernel_params * kparams = (ثابت ساخت htp_ssm_conv_kernel_params *) octx->kernel_params;


    اگر (!htp_ops_context_سیٹ_n_دھاگے(octx, kparams->n_دھاگے)) {
        واپس HTP_حالت_INVAL_PARAMS;
    }

    uint32_t صف_آغاز = 0;
    uint32_t nrows     = kparams->d_inner;

    اگر (octx->ctx->mdev.گنتی > 1) {
        ثابت uint32_t elems_فی_chunk = VLEN_FP32;
        ثابت ساخت htp_tensor_mdev_range range = htp_tensor_mdev_تقسیم(
            kparams->d_inner,
            htp_tensor_mdev_ڈیٹا_aligned(dst) ? elems_فی_chunk : 0,
            octx->ctx->mdev.idx,
            octx->ctx->mdev.گنتی,
            &octx->ctx->mdev.گنتی_div
        );
        صف_آغاز = range.آغاز;
        nrows     = range.گنتی;
    }

    اگر (nrows == 0) {
        واپس HTP_حالت_OK;
    }

    اگر (kparams->vtcm_سائز > octx->ctx->vtcm_سائز) {
        واپس HTP_حالت_VTCM_TOO_SMALL;
    }

    ثابت uint32_t n_دھاگے = octx->n_دھاگے;

    octx->src0_spad.سائز_فی_thread = kparams->vtcm_src0_سائز_فی_thread;
    octx->src1_spad.سائز_فی_thread = kparams->vtcm_src1_سائز_فی_thread;
    octx->dst_spad.سائز_فی_thread  = kparams->vtcm_dst_سائز_فی_thread;

    octx->src0_spad.سائز = kparams->vtcm_src0_سائز;
    octx->src1_spad.سائز = kparams->vtcm_src1_سائز;
    octx->dst_spad.سائز  = kparams->vtcm_dst_سائز;

    octx->src0_spad.ڈیٹا = octx->ctx->vtcm_بنیاد;
    octx->src1_spad.ڈیٹا = octx->src0_spad.ڈیٹا + octx->src0_spad.سائز;
    octx->dst_spad.ڈیٹا  = octx->src1_spad.ڈیٹا + octx->src1_spad.سائز;
    octx->src0_spad.src  = تہی;
    octx->src1_spad.src  = تہی;
    octx->dst_spad.src   = تہی;

    ثابت uint32_t raw_rpt            = fastdiv(nrows + n_دھاگے - 1, &octx->n_دھاگے_div);
    ثابت uint32_t d_inner_فی_thread = hex_باری_اوپر(raw_rpt, VLEN_FP32);

    ساخت htp_ssm_conv_context scctx = {
        .octx             = octx,
        .kparams          = kparams,
        .nrows_فی_thread = d_inner_فی_thread,
        .d_inner_tile     = kparams->d_inner_tile,
        .صف_آغاز        = صف_آغاز,
        .nrows            = nrows,
    };

    FARF(بلند_2, "ssm-conv-f32: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : mode %s\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         kparams->n_t == 1 ? "decode" : "prefill");

    اگر (kparams->n_t == 1) {
        work_قطار_تسلسل(octx->ctx->work_قطار, ssm_conv_thread_f32_decode, &scctx, n_دھاگے);
    } ورنہ {
        work_قطار_تسلسل(octx->ctx->work_قطار, ssm_conv_thread_f32_prefill, &scctx, n_دھاگے);
    }

    واپس HTP_حالت_OK;
}

صحیح op_ssm_conv(ساخت htp_ops_context * octx) {
    ثابت ساخت htp_tensor * dst = octx->dst;

    انتخاب (dst->type) {
        حالت HTP_TYPE_F32:
            واپس op_ssm_conv_f32(octx);
        طےشدہ:
            واپس HTP_حالت_NO_SUPPORT;
    }
}
