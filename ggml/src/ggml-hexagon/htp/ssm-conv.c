#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <HAP_mem.h>
#ضمن <HAP_ps.h>
#ضمن <hexagon_protos.h>
#ضمن <hexagon_types.h>
#ضمن <math.h>
#ضمن <qurt_thread.h>
#ضمن <string.h>

#define GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "htp-ctx.h"
#ضمن "dma-queue.h"
#ضمن "hex-profile.h"
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"
#ضمن "hvx-utils.h"
#ضمن "ssm-conv.h"

بنية htp_ssm_conv_context {
    بنية htp_ops_context *                  octx;
    ثابت بنية htp_ssm_conv_kernel_params * kparams;
    uint32_t                                  nrows_per_thread;
    uint32_t                                  d_inner_tile;
    uint32_t                                  row_start;
    uint32_t                                  nrows;
};

// In-register 32x32 fp32 transpose using std 5-stage HVX vshuff butterfly.
ساكن مضمن فراغ hvx_transpose_32x32_f32(HVX_Vector m[32]) {
    HVX_Vector tmp[32];

    // Stage 0 (R = -4): pair (2i, 2i+1) for i = 0..15. m -> tmp.
    لكل (صحيح i = 0; i < 16; ++i) {
        HVX_VectorPair p = Q6_W_vshuff_VVR(m[2*i + 1], m[2*i], -4);
        tmp[2*i + 0] = Q6_V_lo_W(p);
        tmp[2*i + 1] = Q6_V_hi_W(p);
    }

    // Stage 1 (R = -8): per block of 4, pair (b+0, b+2) and (b+1, b+3). tmp -> m.
    لكل (صحيح b = 0; b < 32; b += 4) {
        HVX_VectorPair p0 = Q6_W_vshuff_VVR(tmp[b + 2], tmp[b + 0], -8);
        HVX_VectorPair p1 = Q6_W_vshuff_VVR(tmp[b + 3], tmp[b + 1], -8);
        m[b + 0] = Q6_V_lo_W(p0); m[b + 1] = Q6_V_hi_W(p0);
        m[b + 2] = Q6_V_lo_W(p1); m[b + 3] = Q6_V_hi_W(p1);
    }

    // Stage 2 (R = -16): per block of 8, pair (b+i, b+i+4) for i = 0..3. m -> tmp.
    لكل (صحيح b = 0; b < 32; b += 8) {
        لكل (صحيح i = 0; i < 4; ++i) {
            HVX_VectorPair p = Q6_W_vshuff_VVR(m[b + i + 4], m[b + i], -16);
            tmp[b + 2*i + 0] = Q6_V_lo_W(p);
            tmp[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }

    // Stage 3 (R = -32): per block of 16, pair (b+i, b+i+8) for i = 0..7. tmp -> m.
    لكل (صحيح b = 0; b < 32; b += 16) {
        لكل (صحيح i = 0; i < 8; ++i) {
            HVX_VectorPair p = Q6_W_vshuff_VVR(tmp[b + i + 8], tmp[b + i], -32);
            m[b + 2*i + 0] = Q6_V_lo_W(p);
            m[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }

    // Stage 4 (R = -64): pair (i, i+16) for i = 0..15. m -> tmp -> m.
    لكل (صحيح i = 0; i < 16; ++i) {
        HVX_VectorPair p = Q6_W_vshuff_VVR(m[i + 16], m[i], -64);
        tmp[2 * i + 0]   = Q6_V_lo_W(p);
        tmp[2 * i + 1]   = Q6_V_hi_W(p);
    }

    لكل (صحيح i = 0; i < 32; ++i) {
        m[i] = tmp[i];
    }
}

// HVX deinterleave for d_conv == 4: channel-major raw VTCM -> tap-major T VTCM
ساكن مضمن فراغ hvx_ssm_conv_unpack_to_T_4(ثابت عائم * raw, عائم * T, uint32_t d_inner_per_thread, uint32_t d_inner_stride) {
    لكل (uint32_t cb = 0; cb < d_inner_per_thread; cb += VLEN_FP32) {
        HVX_Vector v0 = *(ثابت HVX_Vector *)(raw + (cb + 0) * 4);
        HVX_Vector v1 = *(ثابت HVX_Vector *)(raw + (cb + 8) * 4);
        HVX_Vector v2 = *(ثابت HVX_Vector *)(raw + (cb + 16) * 4);
        HVX_Vector v3 = *(ثابت HVX_Vector *)(raw + (cb + 24) * 4);

        HVX_VectorPair p01 = Q6_W_vdeal_VVR(v1, v0, -4);
        HVX_VectorPair p23 = Q6_W_vdeal_VVR(v3, v2, -4);

        HVX_VectorPair p_w02 = Q6_W_vdeal_VVR(Q6_V_lo_W(p23), Q6_V_lo_W(p01), -4);
        HVX_VectorPair p_w13 = Q6_W_vdeal_VVR(Q6_V_hi_W(p23), Q6_V_hi_W(p01), -4);

        *(HVX_Vector *)(T + 0 * d_inner_stride + cb) = Q6_V_lo_W(p_w02);
        *(HVX_Vector *)(T + 1 * d_inner_stride + cb) = Q6_V_lo_W(p_w13);
        *(HVX_Vector *)(T + 2 * d_inner_stride + cb) = Q6_V_hi_W(p_w02);
        *(HVX_Vector *)(T + 3 * d_inner_stride + cb) = Q6_V_hi_W(p_w13);
    }
}

// HVX transpose for general d_conv <= 32: channel-major raw VTCM -> tap-major T VTCM
ساكن مضمن فراغ hvx_ssm_conv_unpack_to_T_gen(ثابت عائم * raw, عائم * T, uint32_t d_inner_per_thread, uint32_t d_inner_stride, uint32_t d_conv) {
    uint32_t __attribute__((aligned(VLEN))) mask_buf[VLEN_FP32] = { 0 };
    لكل (uint32_t j = 0; j < d_conv; ++j) {
        mask_buf[j] = 0xFFFFFFFF;
    }
    ثابت HVX_Vector mask = *(ثابت HVX_Vector *) mask_buf;

    لكل (uint32_t cb = 0; cb < d_inner_per_thread; cb += VLEN_FP32) {
        ثابت uint32_t cb_n = MIN(VLEN_FP32, d_inner_per_thread - cb);
        HVX_Vector sub[32];
        لكل (uint32_t r = 0; r < cb_n; ++r) {
            ثابت عائم * ch_ptr = raw + (cb + r) * d_conv;
            sub[r] = Q6_V_vand_VV(*(ثابت HVX_UVector *) ch_ptr, mask);
        }
        لكل (uint32_t r = cb_n; r < 32; ++r) {
            sub[r] = hvx_vec_splat_f32(0.0f);
        }

        hvx_transpose_32x32_f32(sub);

        لكل (uint32_t j = 0; j < d_conv; ++j) {
            *(HVX_Vector *)(T + j * d_inner_stride + cb) = sub[j];
        }
    }
}

ساكن مضمن فراغ hvx_ssm_conv_unpack_to_T(ثابت عائم * raw, عائم * T, uint32_t d_inner_per_thread, uint32_t d_inner_stride, uint32_t d_conv) {
    إذا (d_conv == 4 && (d_inner_per_thread % VLEN_FP32 == 0)) {
        hvx_ssm_conv_unpack_to_T_4(raw, T, d_inner_per_thread, d_inner_stride);
    } وإلا {
        hvx_ssm_conv_unpack_to_T_gen(raw, T, d_inner_per_thread, d_inner_stride, d_conv);
    }
}

// Decode dot product specialization for d_conv == 4: multiply in the raw channel-major layout,
// then deinterleave the products so each vector holds one tap of 32 channels, and sum.
// Keeps both operands in DMA layout - no transpose, no scratch.
ساكن مضمن فراغ hvx_ssm_conv_decode_4(ثابت عائم * x, ثابت عائم * w, عائم * out, uint32_t n_ch) {
    لكل (uint32_t cb = 0; cb < n_ch; cb += VLEN_FP32) {
        ثابت عائم * xp = x + cb * 4;
        ثابت عائم * wp = w + cb * 4;

        HVX_Vector p0 = Q6_Vqf32_vmpy_VsfVsf(*(ثابت HVX_Vector *)(xp +  0), *(ثابت HVX_Vector *)(wp +  0));
        HVX_Vector p1 = Q6_Vqf32_vmpy_VsfVsf(*(ثابت HVX_Vector *)(xp + 32), *(ثابت HVX_Vector *)(wp + 32));
        HVX_Vector p2 = Q6_Vqf32_vmpy_VsfVsf(*(ثابت HVX_Vector *)(xp + 64), *(ثابت HVX_Vector *)(wp + 64));
        HVX_Vector p3 = Q6_Vqf32_vmpy_VsfVsf(*(ثابت HVX_Vector *)(xp + 96), *(ثابت HVX_Vector *)(wp + 96));

        HVX_VectorPair p01 = Q6_W_vdeal_VVR(p1, p0, -4);
        HVX_VectorPair p23 = Q6_W_vdeal_VVR(p3, p2, -4);

        HVX_VectorPair q02 = Q6_W_vdeal_VVR(Q6_V_lo_W(p23), Q6_V_lo_W(p01), -4);
        HVX_VectorPair q13 = Q6_W_vdeal_VVR(Q6_V_hi_W(p23), Q6_V_hi_W(p01), -4);

        HVX_Vector a = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(q02), Q6_V_lo_W(q13));
        HVX_Vector b = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_hi_W(q02), Q6_V_hi_W(q13));

        *(HVX_Vector *)(out + cb) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(a, b));
    }
}

// Transpose src0 for prefill: one 32-channel block {32, ncs} (VTCM) -> T {ncs, 32} (VTCM).
// One VTCM gather per output row: lane r picks channel cb+r, the region bound drops
// the lanes past the channel tail.
ساكن مضمن فراغ hvx_ssm_conv_transpose_block(ثابت عائم * raw_block,
                                                عائم *       T,
                                                uint32_t      ncs,
                                                uint32_t      cb_n,
                                                HVX_Vector    vv) {
    ثابت size_t   base = (size_t) raw_block;
    ثابت uint32_t mu   = cb_n * ncs * حجم(عائم) - 1;

    لكل (uint32_t t = 0; t < ncs; ++t) {
        Q6_vgather_ARMVw((HVX_Vector *) (T + (size_t) t * VLEN_FP32), base + t * حجم(عائم), mu, vv);
    }
}

// Single-token decode worker (n_t == 1)
ساكن فراغ ssm_conv_thread_f32_decode(unsigned صحيح nth, unsigned صحيح ith, فراغ * data) {
    بنية htp_ssm_conv_context *             scctx   = (بنية htp_ssm_conv_context *) data;
    بنية htp_ops_context *                  octx    = scctx->octx;
    ثابت بنية htp_ssm_conv_kernel_params * kparams = scctx->kparams;

    ثابت بنية htp_tensor * restrict src0 = octx->src[0];
    ثابت بنية htp_tensor * restrict src1 = octx->src[1];
    ثابت بنية htp_tensor * restrict dst  = octx->dst;

    dma_queue * dma_q = octx->ctx->dma[ith];

    ثابت uint32_t d_conv  = kparams->d_conv;
    ثابت uint32_t d_inner = kparams->d_inner;
    ثابت uint32_t n_s     = kparams->n_s;

    ثابت uint32_t dr  = scctx->nrows_per_thread;
    ثابت uint32_t ir0 = scctx->row_start + dr * ith;
    ثابت uint32_t ir1 = MIN(ir0 + dr, scctx->row_start + scctx->nrows);

    إذا (ir0 >= ir1) {
        أرجع;
    }

    ثابت uint32_t d_inner_per_thread = ir1 - ir0;
    ثابت uint32_t d_inner_stride     = hex_round_up(d_inner_per_thread, VLEN_FP32);
    ثابت uint32_t d_inner_tile       = scctx->d_inner_tile;

    ثابت size_t src0_stride_inner_bytes = src0->nb[1];
    ثابت size_t src0_stride_seq_bytes   = src0->nb[2];
    ثابت size_t dst_stride_seq_bytes    = dst->nb[2];

    uint8_t * src1_spad_base = octx->src1_spad.data + ith * octx->src1_spad.size_per_thread;
    uint8_t * src0_spad_base = octx->src0_spad.data + ith * octx->src0_spad.size_per_thread;
    uint8_t * dst_spad_base  = octx->dst_spad.data  + ith * octx->dst_spad.size_per_thread;

    ثابت size_t weight_bytes    = (size_t) d_inner_per_thread * d_conv * حجم(عائم);
    ثابت size_t weight_raw_size = hex_round_up(weight_bytes, 128);

    عائم * src1_raw = (عائم *) src1_spad_base;
    عائم * src1_T   = (عائم *) (src1_spad_base + weight_raw_size);

    ثابت size_t src0_tile_raw_bytes = hex_round_up(d_inner_tile * d_conv * حجم(عائم), 128);
    ثابت size_t dst_tile_bytes      = hex_round_up(d_inner_tile * حجم(عائم), 128);

    عائم * src0_tile_raw[2] = { (عائم *) src0_spad_base, (عائم *) (src0_spad_base + src0_tile_raw_bytes) };
    عائم * src0_T           = (عائم *) (src0_spad_base + 2 * src0_tile_raw_bytes);

    عائم * dst_tile[2] = { (عائم *) dst_spad_base, (عائم *) (dst_spad_base + dst_tile_bytes) };

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    // raw_dot keeps both operands in the DMA layout, so src1 needs no prep pass
    ثابت bool raw_dot = (d_conv == 4) && (d_inner_per_thread % VLEN_FP32 == 0);

    ثابت uint32_t n_tiles   = (d_inner_per_thread + d_inner_tile - 1) / d_inner_tile;
    ثابت uint32_t n_chunks  = n_s * n_tiles;
    ثابت size_t   row_bytes = d_conv * حجم(عائم);

    uint32_t s_fetch        = 0;
    uint32_t tile_off_fetch = 0;

    #define SSM_CONV_DECODE_ادفع_FETCH(c)                                                         \
        افعل {                                                                                      \
            ثابت uint32_t   cur_tile_n = MIN(d_inner_tile, d_inner_per_thread - tile_off_fetch); \
            ثابت dma_addr_t fetch_ddr  = src0->data + s_fetch * src0_stride_seq_bytes +          \
                                          (ir0 + tile_off_fetch) * src0_stride_inner_bytes;       \
            dma_queue_ادفع(dma_q,                                                                 \
                           dma_make_data((uint8_t *) src0_tile_raw[(c) & 1], fetch_ddr),          \
                           row_bytes, src0_stride_inner_bytes, row_bytes, cur_tile_n);            \
            tile_off_fetch += d_inner_tile;                                                       \
            إذا (tile_off_fetch >= d_inner_per_thread) {                                           \
                tile_off_fetch = 0;                                                               \
                s_fetch++;                                                                        \
            }                                                                                     \
        } طالما (0)

    // Queue weights and initial input tiles together so DDR reads overlap
    ثابت dma_addr_t src1_ddr = src1->data + ir0 * d_conv * حجم(عائم);
    dma_queue_ادفع(dma_q, dma_make_data((uint8_t *) src1_raw, src1_ddr), weight_bytes, weight_bytes, weight_bytes, 1);

    SSM_CONV_DECODE_ادفع_FETCH(0);
    إذا (n_chunks > 1) {
        SSM_CONV_DECODE_ادفع_FETCH(1);
    }

    dma_queue_اسحب(dma_q);  // weights

    إذا (!raw_dot) {
        // Unpack/transpose src1_raw into src1_T {d_conv, d_inner_stride}
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);
        hvx_ssm_conv_unpack_to_T(src1_raw, src1_T, d_inner_per_thread, d_inner_stride, d_conv);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);
    }

    uint32_t i3       = 0;
    uint32_t tile_off = 0;

    لكل (uint32_t c = 0; c < n_chunks; ++c) {
        ثابت uint32_t tile_n = MIN(d_inner_tile, d_inner_per_thread - tile_off);

        إذا (c >= 2) {
            dma_queue_اسحب(dma_q);  // writeback of chunk c-2, frees dst_tile[c & 1]
        }
        dma_queue_اسحب(dma_q);      // fetch chunk c

        عائم * restrict out = dst_tile[c & 1];

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i3);
        إذا (raw_dot) {
            ثابت عائم * xp = src0_tile_raw[c & 1];
            ثابت عائم * wp = src1_raw + tile_off * 4;
            hvx_ssm_conv_decode_4(xp, wp, out, tile_n);
        } وإلا {
            ثابت uint32_t tile_stride = hex_round_up(tile_n, VLEN_FP32);
            hvx_ssm_conv_unpack_to_T(src0_tile_raw[c & 1], src0_T, tile_n, tile_stride, d_conv);

            لكل (uint32_t cb = 0; cb < tile_n; cb += VLEN_FP32) {
                ثابت uint32_t cb_n = MIN(VLEN_FP32, tile_n - cb);
                HVX_Vector acc = hvx_vec_splat_f32(0.0f);
                لكل (uint32_t j = 0; j < d_conv; ++j) {
                    HVX_Vector x = *(ثابت HVX_Vector *)(src0_T + j * tile_stride + cb);
                    HVX_Vector w = *(ثابت HVX_Vector *)(src1_T + j * d_inner_stride + tile_off + cb);
                    acc          = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(x, w));
                }
                HVX_Vector y = Q6_Vsf_equals_Vqf32(acc);
                إذا (cb_n == VLEN_FP32) {
                    *(HVX_Vector *)(out + cb) = y;
                } وإلا {
                    hvx_vec_store_u(out + cb, cb_n * حجم(عائم), y);
                }
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i3);

        ثابت dma_addr_t dst_ddr        = dst->data + i3 * dst_stride_seq_bytes + (ir0 + tile_off) * حجم(عائم);
        ثابت size_t     tile_out_bytes = (size_t) tile_n * حجم(عائم);
        dma_queue_ادفع(dma_q, dma_make_data(dst_ddr, (uint8_t *) out),
                       tile_out_bytes, tile_out_bytes, tile_out_bytes, 1);

        إذا (c + 2 < n_chunks) {
            SSM_CONV_DECODE_ادفع_FETCH(c + 2);
        }

        tile_off += d_inner_tile;
        إذا (tile_off >= d_inner_per_thread) {
            tile_off = 0;
            i3++;
        }
    }

    لكل (uint32_t k = MIN(n_chunks, 2); k > 0; --k) {
        dma_queue_اسحب(dma_q);  // drain the last writebacks
    }

    #undef SSM_CONV_DECODE_ادفع_FETCH

    FARF(عالي_2, "ssm-conv-f32-decode %d/%d: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0], dst->ne[1],
         dst->ne[2], dst->ne[3]);
}

// Multi-token prefill worker (n_t > 1)
ساكن فراغ ssm_conv_thread_f32_prefill(unsigned صحيح nth, unsigned صحيح ith, فراغ * data) {
    بنية htp_ssm_conv_context *             scctx   = (بنية htp_ssm_conv_context *) data;
    بنية htp_ops_context *                  octx    = scctx->octx;
    ثابت بنية htp_ssm_conv_kernel_params * kparams = scctx->kparams;

    ثابت بنية htp_tensor * restrict src0 = octx->src[0];
    ثابت بنية htp_tensor * restrict src1 = octx->src[1];
    ثابت بنية htp_tensor * restrict dst  = octx->dst;

    dma_queue * dma_q = octx->ctx->dma[ith];

    ثابت uint32_t d_conv  = kparams->d_conv;
    ثابت uint32_t d_inner = kparams->d_inner;
    ثابت uint32_t n_t     = kparams->n_t;
    ثابت uint32_t n_s     = kparams->n_s;
    ثابت uint32_t ncs     = src0->ne[0];

    ثابت uint32_t dr  = scctx->nrows_per_thread;
    ثابت uint32_t ir0 = scctx->row_start + dr * ith;
    ثابت uint32_t ir1 = MIN(ir0 + dr, scctx->row_start + scctx->nrows);

    إذا (ir0 >= ir1) {
        أرجع;
    }

    ثابت uint32_t d_inner_per_thread = ir1 - ir0;
    ثابت uint32_t d_inner_stride     = hex_round_up(d_inner_per_thread, VLEN_FP32);
    ثابت uint32_t d_inner_tile       = scctx->d_inner_tile;

    ثابت size_t src0_stride_inner_bytes = src0->nb[1];
    ثابت size_t src0_stride_seq_bytes   = src0->nb[2];
    ثابت size_t dst_stride_token_bytes  = dst->nb[1];
    ثابت size_t dst_stride_seq_bytes    = dst->nb[2];

    uint8_t * src1_spad_base = octx->src1_spad.data + ith * octx->src1_spad.size_per_thread;
    uint8_t * src0_spad_base = octx->src0_spad.data + ith * octx->src0_spad.size_per_thread;
    uint8_t * dst_spad_base  = octx->dst_spad.data  + ith * octx->dst_spad.size_per_thread;

    ثابت size_t weight_bytes    = (size_t) d_inner_per_thread * d_conv * حجم(عائم);
    ثابت size_t weight_raw_size = hex_round_up(weight_bytes, 128);

    عائم * src1_raw = (عائم *) src1_spad_base;
    عائم * src1_T   = (عائم *) (src1_spad_base + weight_raw_size);

    // src0 spad holds two raw tiles (fetch of tile n+1 overlaps compute of tile n) plus
    // the transposed block. dst spad holds two tiles so a writeback can stay in flight.
    ثابت size_t src0_tile_raw_bytes = hex_round_up(d_inner_tile * ncs * حجم(عائم), 128);
    ثابت size_t dst_tile_bytes      = hex_round_up(d_inner_tile * n_t * حجم(عائم), 128);

    عائم * src0_tile_raw[2] = { (عائم *) src0_spad_base, (عائم *) (src0_spad_base + src0_tile_raw_bytes) };
    عائم * src0_T           = (عائم *) (src0_spad_base + 2 * src0_tile_raw_bytes);

    عائم * dst_tile[2] = { (عائم *) dst_spad_base, (عائم *) (dst_spad_base + dst_tile_bytes) };

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت uint32_t n_tiles   = (d_inner_per_thread + d_inner_tile - 1) / d_inner_tile;
    ثابت uint32_t n_chunks  = n_s * n_tiles;
    ثابت size_t   row_bytes = ncs * حجم(عائم);

    uint32_t s_fetch        = 0;
    uint32_t tile_off_fetch = 0;

    // Chunk c fetches into src0_tile_raw[c & 1] and writes back from dst_tile[c & 1].
    // Two fetches run ahead, so the queue order is F0 F1 W0 F2 W1 ... and pops follow it.
    #define SSM_CONV_ادفع_FETCH(c)                                                                \
        افعل {                                                                                      \
            ثابت uint32_t   cur_tile_n = MIN(d_inner_tile, d_inner_per_thread - tile_off_fetch); \
            ثابت dma_addr_t fetch_ddr  = src0->data + s_fetch * src0_stride_seq_bytes +          \
                                          (ir0 + tile_off_fetch) * src0_stride_inner_bytes;       \
            dma_queue_ادفع(dma_q,                                                                 \
                           dma_make_data((uint8_t *) src0_tile_raw[(c) & 1], fetch_ddr),          \
                           row_bytes, src0_stride_inner_bytes, row_bytes, cur_tile_n);            \
            tile_off_fetch += d_inner_tile;                                                       \
            إذا (tile_off_fetch >= d_inner_per_thread) {                                           \
                tile_off_fetch = 0;                                                               \
                s_fetch++;                                                                        \
            }                                                                                     \
        } طالما (0)

    // Queue weights and initial input tiles together so DDR reads overlap
    ثابت dma_addr_t src1_ddr = src1->data + ir0 * d_conv * حجم(عائم);
    dma_queue_ادفع(dma_q, dma_make_data((uint8_t *) src1_raw, src1_ddr), weight_bytes, weight_bytes, weight_bytes, 1);

    SSM_CONV_ادفع_FETCH(0);
    إذا (n_chunks > 1) {
        SSM_CONV_ادفع_FETCH(1);
    }

    dma_queue_اسحب(dma_q);  // weights

    // Unpack/transpose src1_raw into src1_T {d_conv, d_inner_stride}
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);
    hvx_ssm_conv_unpack_to_T(src1_raw, src1_T, d_inner_per_thread, d_inner_stride, d_conv);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);

    ثابت uint32_t C_TILE = VLEN_FP32;

    // gather offsets: lane r reads channel r of the raw tile block
    uint32_t __attribute__((aligned(VLEN))) gather_off[VLEN_FP32];
    لكل (uint32_t r = 0; r < VLEN_FP32; ++r) {
        gather_off[r] = r * ncs * حجم(عائم);
    }
    ثابت HVX_Vector vv = *(ثابت HVX_Vector *) gather_off;

    uint32_t i3       = 0;
    uint32_t tile_off = 0;

    لكل (uint32_t c = 0; c < n_chunks; ++c) {
        ثابت uint32_t tile_n = MIN(d_inner_tile, d_inner_per_thread - tile_off);

        ثابت عائم * restrict raw = src0_tile_raw[c & 1];
        عائم * restrict       out = dst_tile[c & 1];

        إذا (c >= 2) {
            dma_queue_اسحب(dma_q);  // writeback of chunk c-2, frees dst_tile[c & 1]
        }
        dma_queue_اسحب(dma_q);      // fetch chunk c

        // Channel block outer, token inner: the taps and the sliding window stay in
        // registers, so a new output row costs one src0_T load.
        ثابت uint32_t dst_tile_stride = hex_round_up(tile_n, C_TILE);

        لكل (uint32_t cb = 0; cb < tile_n; cb += C_TILE) {
            ثابت uint32_t cb_n = MIN(C_TILE, tile_n - cb);

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) tile_off);
            hvx_ssm_conv_transpose_block(raw + (size_t) cb * ncs, src0_T, ncs, cb_n, vv);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) tile_off);

            ثابت عائم * restrict wp = src1_T + tile_off + cb;
            عائم * restrict       op = out + cb;

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) tile_off);
            إذا (d_conv == 4) {
                ثابت HVX_Vector w0 = *(ثابت HVX_Vector *) (wp);
                ثابت HVX_Vector w1 = *(ثابت HVX_Vector *) (wp + d_inner_stride);
                ثابت HVX_Vector w2 = *(ثابت HVX_Vector *) (wp + 2 * d_inner_stride);
                ثابت HVX_Vector w3 = *(ثابت HVX_Vector *) (wp + 3 * d_inner_stride);

                HVX_Vector x0 = *(ثابت HVX_Vector *) (src0_T);
                HVX_Vector x1 = *(ثابت HVX_Vector *) (src0_T + C_TILE);
                HVX_Vector x2 = *(ثابت HVX_Vector *) (src0_T + 2 * C_TILE);

                لكل (uint32_t t = 0; t < n_t; ++t) {
                    ثابت HVX_Vector x3 = *(ثابت HVX_Vector *) (src0_T + (t + 3) * C_TILE);

                    HVX_Vector a = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(x0, w0),
                                                            Q6_Vqf32_vmpy_VsfVsf(x1, w1));
                    HVX_Vector b = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(x2, w2),
                                                            Q6_Vqf32_vmpy_VsfVsf(x3, w3));

                    *(HVX_Vector *) (op + t * dst_tile_stride) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(a, b));

                    x0 = x1;
                    x1 = x2;
                    x2 = x3;
                }
            } وإلا {
                لكل (uint32_t t = 0; t < n_t; ++t) {
                    HVX_Vector acc = hvx_vec_splat_f32(0.0f);
                    لكل (uint32_t j = 0; j < d_conv; ++j) {
                        HVX_Vector x = *(ثابت HVX_Vector *) (src0_T + (t + j) * C_TILE);
                        HVX_Vector w = *(ثابت HVX_Vector *) (wp + j * d_inner_stride);
                        acc          = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(x, w));
                    }
                    *(HVX_Vector *) (op + t * dst_tile_stride) = Q6_Vsf_equals_Vqf32(acc);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) tile_off);
        }

        // Writeback dst_tile VTCM -> DDR via 2D DMA
        dma_queue_ادفع(dma_q,
                       dma_make_data(dst->data + i3 * dst_stride_seq_bytes + (ir0 + tile_off) * حجم(عائم),
                                     (uint8_t *) out),
                       dst_stride_token_bytes, dst_tile_stride * حجم(عائم), tile_n * حجم(عائم), n_t);

        إذا (c + 2 < n_chunks) {
            SSM_CONV_ادفع_FETCH(c + 2);
        }

        tile_off += d_inner_tile;
        إذا (tile_off >= d_inner_per_thread) {
            tile_off = 0;
            i3++;
        }
    }

    لكل (uint32_t k = MIN(n_chunks, 2); k > 0; --k) {
        dma_queue_اسحب(dma_q);  // drain the last writebacks
    }

    #undef SSM_CONV_ادفع_FETCH

    FARF(عالي_2, "ssm-conv-f32-prefill %d/%d: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0], dst->ne[1],
         dst->ne[2], dst->ne[3]);
}

صحيح op_ssm_conv_f32(بنية htp_ops_context * octx) {
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * src1 = octx->src[1];
    ثابت بنية htp_tensor * dst  = octx->dst;

    إذا (src0->type != HTP_TYPE_F32 || src1->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        أرجع HTP_STATUS_NO_SUPPORT;
    }

    ثابت بنية htp_ssm_conv_kernel_params * kparams = (ثابت بنية htp_ssm_conv_kernel_params *) octx->kernel_params;


    إذا (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        أرجع HTP_STATUS_INVAL_PARAMS;
    }

    uint32_t row_start = 0;
    uint32_t nrows     = kparams->d_inner;

    إذا (octx->ctx->mdev.عدد > 1) {
        ثابت uint32_t elems_per_chunk = VLEN_FP32;
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            kparams->d_inner,
            htp_tensor_mdev_data_aligned(dst) ? elems_per_chunk : 0,
            octx->ctx->mdev.idx,
            octx->ctx->mdev.عدد,
            &octx->ctx->mdev.عدد_div
        );
        row_start = range.start;
        nrows     = range.عدد;
    }

    إذا (nrows == 0) {
        أرجع HTP_STATUS_OK;
    }

    إذا (kparams->vtcm_size > octx->ctx->vtcm_size) {
        أرجع HTP_STATUS_VTCM_TOO_SMALL;
    }

    ثابت uint32_t n_threads = octx->n_threads;

    octx->src0_spad.size_per_thread = kparams->vtcm_src0_size_per_thread;
    octx->src1_spad.size_per_thread = kparams->vtcm_src1_size_per_thread;
    octx->dst_spad.size_per_thread  = kparams->vtcm_dst_size_per_thread;

    octx->src0_spad.size = kparams->vtcm_src0_size;
    octx->src1_spad.size = kparams->vtcm_src1_size;
    octx->dst_spad.size  = kparams->vtcm_dst_size;

    octx->src0_spad.data = octx->ctx->vtcm_base;
    octx->src1_spad.data = octx->src0_spad.data + octx->src0_spad.size;
    octx->dst_spad.data  = octx->src1_spad.data + octx->src1_spad.size;
    octx->src0_spad.src  = عدم;
    octx->src1_spad.src  = عدم;
    octx->dst_spad.src   = عدم;

    ثابت uint32_t raw_rpt            = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);
    ثابت uint32_t d_inner_per_thread = hex_round_up(raw_rpt, VLEN_FP32);

    بنية htp_ssm_conv_context scctx = {
        .octx             = octx,
        .kparams          = kparams,
        .nrows_per_thread = d_inner_per_thread,
        .d_inner_tile     = kparams->d_inner_tile,
        .row_start        = row_start,
        .nrows            = nrows,
    };

    FARF(عالي_2, "ssm-conv-f32: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : mode %s\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         kparams->n_t == 1 ? "decode" : "prefill");

    إذا (kparams->n_t == 1) {
        work_queue_run(octx->ctx->work_queue, ssm_conv_thread_f32_decode, &scctx, n_threads);
    } وإلا {
        work_queue_run(octx->ctx->work_queue, ssm_conv_thread_f32_prefill, &scctx, n_threads);
    }

    أرجع HTP_STATUS_OK;
}

صحيح op_ssm_conv(بنية htp_ops_context * octx) {
    ثابت بنية htp_tensor * dst = octx->dst;

    بدل (dst->type) {
        حالة HTP_TYPE_F32:
            أرجع op_ssm_conv_f32(octx);
        افتراضي:
            أرجع HTP_STATUS_NO_SUPPORT;
    }
}
