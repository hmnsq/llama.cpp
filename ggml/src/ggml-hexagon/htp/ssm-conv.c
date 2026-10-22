#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#અન્તર્ભાવ <HAP_farf.h>
#અન્તર્ભાવ <HAP_mem.h>
#અન્તર્ભાવ <HAP_ps.h>
#અન્તર્ભાવ <hexagon_protos.h>
#અન્તર્ભાવ <hexagon_types.h>
#અન્તર્ભાવ <math.h>
#અન્તર્ભાવ <qurt_thread.h>
#અન્તર્ભાવ <string.h>

#define GGML_COMMON_DECL_C
#અન્તર્ભાવ "ggml-common.h"
#અન્તર્ભાવ "htp-ctx.h"
#અન્તર્ભાવ "dma-queue.h"
#અન્તર્ભાવ "hex-profile.h"
#અન્તર્ભાવ "htp-ops.h"
#અન્તર્ભાવ "htp-tensor.h"
#અન્તર્ભાવ "hvx-utils.h"
#અન્તર્ભાવ "ssm-conv.h"

સંરચના htp_ssm_conv_context {
    સંરચના htp_ops_context *                  octx;
    અચલ સંરચના htp_ssm_conv_kernel_params * kparams;
    uint32_t                                  nrows_per_thread;
    uint32_t                                  d_inner_tile;
    uint32_t                                  row_start;
    uint32_t                                  nrows;
};

// In-register 32x32 fp32 transpose using std 5-stage HVX vshuff butterfly.
સ્થિર અંતઃરેખ શૂન્ય hvx_transpose_32x32_f32(HVX_Vector m[32]) {
    HVX_Vector tmp[32];

    // Stage 0 (R = -4): pair (2i, 2i+1) for i = 0..15. m -> tmp.
    માટે (પૂર્ણાંક i = 0; i < 16; ++i) {
        HVX_VectorPair p = Q6_W_vshuff_VVR(m[2*i + 1], m[2*i], -4);
        tmp[2*i + 0] = Q6_V_lo_W(p);
        tmp[2*i + 1] = Q6_V_hi_W(p);
    }

    // Stage 1 (R = -8): per block of 4, pair (b+0, b+2) and (b+1, b+3). tmp -> m.
    માટે (પૂર્ણાંક b = 0; b < 32; b += 4) {
        HVX_VectorPair p0 = Q6_W_vshuff_VVR(tmp[b + 2], tmp[b + 0], -8);
        HVX_VectorPair p1 = Q6_W_vshuff_VVR(tmp[b + 3], tmp[b + 1], -8);
        m[b + 0] = Q6_V_lo_W(p0); m[b + 1] = Q6_V_hi_W(p0);
        m[b + 2] = Q6_V_lo_W(p1); m[b + 3] = Q6_V_hi_W(p1);
    }

    // Stage 2 (R = -16): per block of 8, pair (b+i, b+i+4) for i = 0..3. m -> tmp.
    માટે (પૂર્ણાંક b = 0; b < 32; b += 8) {
        માટે (પૂર્ણાંક i = 0; i < 4; ++i) {
            HVX_VectorPair p = Q6_W_vshuff_VVR(m[b + i + 4], m[b + i], -16);
            tmp[b + 2*i + 0] = Q6_V_lo_W(p);
            tmp[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }

    // Stage 3 (R = -32): per block of 16, pair (b+i, b+i+8) for i = 0..7. tmp -> m.
    માટે (પૂર્ણાંક b = 0; b < 32; b += 16) {
        માટે (પૂર્ણાંક i = 0; i < 8; ++i) {
            HVX_VectorPair p = Q6_W_vshuff_VVR(tmp[b + i + 8], tmp[b + i], -32);
            m[b + 2*i + 0] = Q6_V_lo_W(p);
            m[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }

    // Stage 4 (R = -64): pair (i, i+16) for i = 0..15. m -> tmp -> m.
    માટે (પૂર્ણાંક i = 0; i < 16; ++i) {
        HVX_VectorPair p = Q6_W_vshuff_VVR(m[i + 16], m[i], -64);
        tmp[2 * i + 0]   = Q6_V_lo_W(p);
        tmp[2 * i + 1]   = Q6_V_hi_W(p);
    }

    માટે (પૂર્ણાંક i = 0; i < 32; ++i) {
        m[i] = tmp[i];
    }
}

// HVX deinterleave for d_conv == 4: channel-major raw VTCM -> tap-major T VTCM
સ્થિર અંતઃરેખ શૂન્ય hvx_ssm_conv_unpack_to_T_4(અચલ દશાંશ * raw, દશાંશ * T, uint32_t d_inner_per_thread, uint32_t d_inner_stride) {
    માટે (uint32_t cb = 0; cb < d_inner_per_thread; cb += VLEN_FP32) {
        HVX_Vector v0 = *(અચલ HVX_Vector *)(raw + (cb + 0) * 4);
        HVX_Vector v1 = *(અચલ HVX_Vector *)(raw + (cb + 8) * 4);
        HVX_Vector v2 = *(અચલ HVX_Vector *)(raw + (cb + 16) * 4);
        HVX_Vector v3 = *(અચલ HVX_Vector *)(raw + (cb + 24) * 4);

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
સ્થિર અંતઃરેખ શૂન્ય hvx_ssm_conv_unpack_to_T_gen(અચલ દશાંશ * raw, દશાંશ * T, uint32_t d_inner_per_thread, uint32_t d_inner_stride, uint32_t d_conv) {
    uint32_t __attribute__((aligned(VLEN))) mask_buf[VLEN_FP32] = { 0 };
    માટે (uint32_t j = 0; j < d_conv; ++j) {
        mask_buf[j] = 0xFFFFFFFF;
    }
    અચલ HVX_Vector mask = *(અચલ HVX_Vector *) mask_buf;

    માટે (uint32_t cb = 0; cb < d_inner_per_thread; cb += VLEN_FP32) {
        અચલ uint32_t cb_n = MIN(VLEN_FP32, d_inner_per_thread - cb);
        HVX_Vector sub[32];
        માટે (uint32_t r = 0; r < cb_n; ++r) {
            અચલ દશાંશ * ch_ptr = raw + (cb + r) * d_conv;
            sub[r] = Q6_V_vand_VV(*(અચલ HVX_UVector *) ch_ptr, mask);
        }
        માટે (uint32_t r = cb_n; r < 32; ++r) {
            sub[r] = hvx_vec_splat_f32(0.0f);
        }

        hvx_transpose_32x32_f32(sub);

        માટે (uint32_t j = 0; j < d_conv; ++j) {
            *(HVX_Vector *)(T + j * d_inner_stride + cb) = sub[j];
        }
    }
}

સ્થિર અંતઃરેખ શૂન્ય hvx_ssm_conv_unpack_to_T(અચલ દશાંશ * raw, દશાંશ * T, uint32_t d_inner_per_thread, uint32_t d_inner_stride, uint32_t d_conv) {
    જો (d_conv == 4 && (d_inner_per_thread % VLEN_FP32 == 0)) {
        hvx_ssm_conv_unpack_to_T_4(raw, T, d_inner_per_thread, d_inner_stride);
    } નહીંતર {
        hvx_ssm_conv_unpack_to_T_gen(raw, T, d_inner_per_thread, d_inner_stride, d_conv);
    }
}

// Decode dot product specialization for d_conv == 4: multiply in the raw channel-major layout,
// then deinterleave the products so each vector holds one tap of 32 channels, and sum.
// Keeps both operands in DMA layout - no transpose, no scratch.
સ્થિર અંતઃરેખ શૂન્ય hvx_ssm_conv_decode_4(અચલ દશાંશ * x, અચલ દશાંશ * w, દશાંશ * out, uint32_t n_ch) {
    માટે (uint32_t cb = 0; cb < n_ch; cb += VLEN_FP32) {
        અચલ દશાંશ * xp = x + cb * 4;
        અચલ દશાંશ * wp = w + cb * 4;

        HVX_Vector p0 = Q6_Vqf32_vmpy_VsfVsf(*(અચલ HVX_Vector *)(xp +  0), *(અચલ HVX_Vector *)(wp +  0));
        HVX_Vector p1 = Q6_Vqf32_vmpy_VsfVsf(*(અચલ HVX_Vector *)(xp + 32), *(અચલ HVX_Vector *)(wp + 32));
        HVX_Vector p2 = Q6_Vqf32_vmpy_VsfVsf(*(અચલ HVX_Vector *)(xp + 64), *(અચલ HVX_Vector *)(wp + 64));
        HVX_Vector p3 = Q6_Vqf32_vmpy_VsfVsf(*(અચલ HVX_Vector *)(xp + 96), *(અચલ HVX_Vector *)(wp + 96));

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
સ્થિર અંતઃરેખ શૂન્ય hvx_ssm_conv_transpose_block(અચલ દશાંશ * raw_block,
                                                દશાંશ *       T,
                                                uint32_t      ncs,
                                                uint32_t      cb_n,
                                                HVX_Vector    vv) {
    અચલ size_t   base = (size_t) raw_block;
    અચલ uint32_t mu   = cb_n * ncs * કદ(દશાંશ) - 1;

    માટે (uint32_t t = 0; t < ncs; ++t) {
        Q6_vgather_ARMVw((HVX_Vector *) (T + (size_t) t * VLEN_FP32), base + t * કદ(દશાંશ), mu, vv);
    }
}

// Single-token decode worker (n_t == 1)
સ્થિર શૂન્ય ssm_conv_thread_f32_decode(unsigned પૂર્ણાંક nth, unsigned પૂર્ણાંક ith, શૂન્ય * data) {
    સંરચના htp_ssm_conv_context *             scctx   = (સંરચના htp_ssm_conv_context *) data;
    સંરચના htp_ops_context *                  octx    = scctx->octx;
    અચલ સંરચના htp_ssm_conv_kernel_params * kparams = scctx->kparams;

    અચલ સંરચના htp_tensor * restrict src0 = octx->src[0];
    અચલ સંરચના htp_tensor * restrict src1 = octx->src[1];
    અચલ સંરચના htp_tensor * restrict dst  = octx->dst;

    dma_queue * dma_q = octx->ctx->dma[ith];

    અચલ uint32_t d_conv  = kparams->d_conv;
    અચલ uint32_t d_inner = kparams->d_inner;
    અચલ uint32_t n_s     = kparams->n_s;

    અચલ uint32_t dr  = scctx->nrows_per_thread;
    અચલ uint32_t ir0 = scctx->row_start + dr * ith;
    અચલ uint32_t ir1 = MIN(ir0 + dr, scctx->row_start + scctx->nrows);

    જો (ir0 >= ir1) {
        પાછા;
    }

    અચલ uint32_t d_inner_per_thread = ir1 - ir0;
    અચલ uint32_t d_inner_stride     = hex_round_up(d_inner_per_thread, VLEN_FP32);
    અચલ uint32_t d_inner_tile       = scctx->d_inner_tile;

    અચલ size_t src0_stride_inner_bytes = src0->nb[1];
    અચલ size_t src0_stride_seq_bytes   = src0->nb[2];
    અચલ size_t dst_stride_seq_bytes    = dst->nb[2];

    uint8_t * src1_spad_base = octx->src1_spad.data + ith * octx->src1_spad.size_per_thread;
    uint8_t * src0_spad_base = octx->src0_spad.data + ith * octx->src0_spad.size_per_thread;
    uint8_t * dst_spad_base  = octx->dst_spad.data  + ith * octx->dst_spad.size_per_thread;

    અચલ size_t weight_bytes    = (size_t) d_inner_per_thread * d_conv * કદ(દશાંશ);
    અચલ size_t weight_raw_size = hex_round_up(weight_bytes, 128);

    દશાંશ * src1_raw = (દશાંશ *) src1_spad_base;
    દશાંશ * src1_T   = (દશાંશ *) (src1_spad_base + weight_raw_size);

    અચલ size_t src0_tile_raw_bytes = hex_round_up(d_inner_tile * d_conv * કદ(દશાંશ), 128);
    અચલ size_t dst_tile_bytes      = hex_round_up(d_inner_tile * કદ(દશાંશ), 128);

    દશાંશ * src0_tile_raw[2] = { (દશાંશ *) src0_spad_base, (દશાંશ *) (src0_spad_base + src0_tile_raw_bytes) };
    દશાંશ * src0_T           = (દશાંશ *) (src0_spad_base + 2 * src0_tile_raw_bytes);

    દશાંશ * dst_tile[2] = { (દશાંશ *) dst_spad_base, (દશાંશ *) (dst_spad_base + dst_tile_bytes) };

    સંરચના htp_thread_trace * tr = &octx->ctx->trace[ith];

    // raw_dot keeps both operands in the DMA layout, so src1 needs no prep pass
    અચલ bool raw_dot = (d_conv == 4) && (d_inner_per_thread % VLEN_FP32 == 0);

    અચલ uint32_t n_tiles   = (d_inner_per_thread + d_inner_tile - 1) / d_inner_tile;
    અચલ uint32_t n_chunks  = n_s * n_tiles;
    અચલ size_t   row_bytes = d_conv * કદ(દશાંશ);

    uint32_t s_fetch        = 0;
    uint32_t tile_off_fetch = 0;

    #define SSM_CONV_DECODE_ધકેલો_FETCH(c)                                                         \
        કરો {                                                                                      \
            અચલ uint32_t   cur_tile_n = MIN(d_inner_tile, d_inner_per_thread - tile_off_fetch); \
            અચલ dma_addr_t fetch_ddr  = src0->data + s_fetch * src0_stride_seq_bytes +          \
                                          (ir0 + tile_off_fetch) * src0_stride_inner_bytes;       \
            dma_queue_ધકેલો(dma_q,                                                                 \
                           dma_make_data((uint8_t *) src0_tile_raw[(c) & 1], fetch_ddr),          \
                           row_bytes, src0_stride_inner_bytes, row_bytes, cur_tile_n);            \
            tile_off_fetch += d_inner_tile;                                                       \
            જો (tile_off_fetch >= d_inner_per_thread) {                                           \
                tile_off_fetch = 0;                                                               \
                s_fetch++;                                                                        \
            }                                                                                     \
        } જ્યારે (0)

    // Queue weights and initial input tiles together so DDR reads overlap
    અચલ dma_addr_t src1_ddr = src1->data + ir0 * d_conv * કદ(દશાંશ);
    dma_queue_ધકેલો(dma_q, dma_make_data((uint8_t *) src1_raw, src1_ddr), weight_bytes, weight_bytes, weight_bytes, 1);

    SSM_CONV_DECODE_ધકેલો_FETCH(0);
    જો (n_chunks > 1) {
        SSM_CONV_DECODE_ધકેલો_FETCH(1);
    }

    dma_queue_કાઢો(dma_q);  // weights

    જો (!raw_dot) {
        // Unpack/transpose src1_raw into src1_T {d_conv, d_inner_stride}
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);
        hvx_ssm_conv_unpack_to_T(src1_raw, src1_T, d_inner_per_thread, d_inner_stride, d_conv);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);
    }

    uint32_t i3       = 0;
    uint32_t tile_off = 0;

    માટે (uint32_t c = 0; c < n_chunks; ++c) {
        અચલ uint32_t tile_n = MIN(d_inner_tile, d_inner_per_thread - tile_off);

        જો (c >= 2) {
            dma_queue_કાઢો(dma_q);  // writeback of chunk c-2, frees dst_tile[c & 1]
        }
        dma_queue_કાઢો(dma_q);      // fetch chunk c

        દશાંશ * restrict out = dst_tile[c & 1];

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i3);
        જો (raw_dot) {
            અચલ દશાંશ * xp = src0_tile_raw[c & 1];
            અચલ દશાંશ * wp = src1_raw + tile_off * 4;
            hvx_ssm_conv_decode_4(xp, wp, out, tile_n);
        } નહીંતર {
            અચલ uint32_t tile_stride = hex_round_up(tile_n, VLEN_FP32);
            hvx_ssm_conv_unpack_to_T(src0_tile_raw[c & 1], src0_T, tile_n, tile_stride, d_conv);

            માટે (uint32_t cb = 0; cb < tile_n; cb += VLEN_FP32) {
                અચલ uint32_t cb_n = MIN(VLEN_FP32, tile_n - cb);
                HVX_Vector acc = hvx_vec_splat_f32(0.0f);
                માટે (uint32_t j = 0; j < d_conv; ++j) {
                    HVX_Vector x = *(અચલ HVX_Vector *)(src0_T + j * tile_stride + cb);
                    HVX_Vector w = *(અચલ HVX_Vector *)(src1_T + j * d_inner_stride + tile_off + cb);
                    acc          = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(x, w));
                }
                HVX_Vector y = Q6_Vsf_equals_Vqf32(acc);
                જો (cb_n == VLEN_FP32) {
                    *(HVX_Vector *)(out + cb) = y;
                } નહીંતર {
                    hvx_vec_store_u(out + cb, cb_n * કદ(દશાંશ), y);
                }
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i3);

        અચલ dma_addr_t dst_ddr        = dst->data + i3 * dst_stride_seq_bytes + (ir0 + tile_off) * કદ(દશાંશ);
        અચલ size_t     tile_out_bytes = (size_t) tile_n * કદ(દશાંશ);
        dma_queue_ધકેલો(dma_q, dma_make_data(dst_ddr, (uint8_t *) out),
                       tile_out_bytes, tile_out_bytes, tile_out_bytes, 1);

        જો (c + 2 < n_chunks) {
            SSM_CONV_DECODE_ધકેલો_FETCH(c + 2);
        }

        tile_off += d_inner_tile;
        જો (tile_off >= d_inner_per_thread) {
            tile_off = 0;
            i3++;
        }
    }

    માટે (uint32_t k = MIN(n_chunks, 2); k > 0; --k) {
        dma_queue_કાઢો(dma_q);  // drain the last writebacks
    }

    #undef SSM_CONV_DECODE_ધકેલો_FETCH

    FARF(ઉપલું_2, "ssm-conv-f32-decode %d/%d: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0], dst->ne[1],
         dst->ne[2], dst->ne[3]);
}

// Multi-token prefill worker (n_t > 1)
સ્થિર શૂન્ય ssm_conv_thread_f32_prefill(unsigned પૂર્ણાંક nth, unsigned પૂર્ણાંક ith, શૂન્ય * data) {
    સંરચના htp_ssm_conv_context *             scctx   = (સંરચના htp_ssm_conv_context *) data;
    સંરચના htp_ops_context *                  octx    = scctx->octx;
    અચલ સંરચના htp_ssm_conv_kernel_params * kparams = scctx->kparams;

    અચલ સંરચના htp_tensor * restrict src0 = octx->src[0];
    અચલ સંરચના htp_tensor * restrict src1 = octx->src[1];
    અચલ સંરચના htp_tensor * restrict dst  = octx->dst;

    dma_queue * dma_q = octx->ctx->dma[ith];

    અચલ uint32_t d_conv  = kparams->d_conv;
    અચલ uint32_t d_inner = kparams->d_inner;
    અચલ uint32_t n_t     = kparams->n_t;
    અચલ uint32_t n_s     = kparams->n_s;
    અચલ uint32_t ncs     = src0->ne[0];

    અચલ uint32_t dr  = scctx->nrows_per_thread;
    અચલ uint32_t ir0 = scctx->row_start + dr * ith;
    અચલ uint32_t ir1 = MIN(ir0 + dr, scctx->row_start + scctx->nrows);

    જો (ir0 >= ir1) {
        પાછા;
    }

    અચલ uint32_t d_inner_per_thread = ir1 - ir0;
    અચલ uint32_t d_inner_stride     = hex_round_up(d_inner_per_thread, VLEN_FP32);
    અચલ uint32_t d_inner_tile       = scctx->d_inner_tile;

    અચલ size_t src0_stride_inner_bytes = src0->nb[1];
    અચલ size_t src0_stride_seq_bytes   = src0->nb[2];
    અચલ size_t dst_stride_token_bytes  = dst->nb[1];
    અચલ size_t dst_stride_seq_bytes    = dst->nb[2];

    uint8_t * src1_spad_base = octx->src1_spad.data + ith * octx->src1_spad.size_per_thread;
    uint8_t * src0_spad_base = octx->src0_spad.data + ith * octx->src0_spad.size_per_thread;
    uint8_t * dst_spad_base  = octx->dst_spad.data  + ith * octx->dst_spad.size_per_thread;

    અચલ size_t weight_bytes    = (size_t) d_inner_per_thread * d_conv * કદ(દશાંશ);
    અચલ size_t weight_raw_size = hex_round_up(weight_bytes, 128);

    દશાંશ * src1_raw = (દશાંશ *) src1_spad_base;
    દશાંશ * src1_T   = (દશાંશ *) (src1_spad_base + weight_raw_size);

    // src0 spad holds two raw tiles (fetch of tile n+1 overlaps compute of tile n) plus
    // the transposed block. dst spad holds two tiles so a writeback can stay in flight.
    અચલ size_t src0_tile_raw_bytes = hex_round_up(d_inner_tile * ncs * કદ(દશાંશ), 128);
    અચલ size_t dst_tile_bytes      = hex_round_up(d_inner_tile * n_t * કદ(દશાંશ), 128);

    દશાંશ * src0_tile_raw[2] = { (દશાંશ *) src0_spad_base, (દશાંશ *) (src0_spad_base + src0_tile_raw_bytes) };
    દશાંશ * src0_T           = (દશાંશ *) (src0_spad_base + 2 * src0_tile_raw_bytes);

    દશાંશ * dst_tile[2] = { (દશાંશ *) dst_spad_base, (દશાંશ *) (dst_spad_base + dst_tile_bytes) };

    સંરચના htp_thread_trace * tr = &octx->ctx->trace[ith];

    અચલ uint32_t n_tiles   = (d_inner_per_thread + d_inner_tile - 1) / d_inner_tile;
    અચલ uint32_t n_chunks  = n_s * n_tiles;
    અચલ size_t   row_bytes = ncs * કદ(દશાંશ);

    uint32_t s_fetch        = 0;
    uint32_t tile_off_fetch = 0;

    // Chunk c fetches into src0_tile_raw[c & 1] and writes back from dst_tile[c & 1].
    // Two fetches run ahead, so the queue order is F0 F1 W0 F2 W1 ... and pops follow it.
    #define SSM_CONV_ધકેલો_FETCH(c)                                                                \
        કરો {                                                                                      \
            અચલ uint32_t   cur_tile_n = MIN(d_inner_tile, d_inner_per_thread - tile_off_fetch); \
            અચલ dma_addr_t fetch_ddr  = src0->data + s_fetch * src0_stride_seq_bytes +          \
                                          (ir0 + tile_off_fetch) * src0_stride_inner_bytes;       \
            dma_queue_ધકેલો(dma_q,                                                                 \
                           dma_make_data((uint8_t *) src0_tile_raw[(c) & 1], fetch_ddr),          \
                           row_bytes, src0_stride_inner_bytes, row_bytes, cur_tile_n);            \
            tile_off_fetch += d_inner_tile;                                                       \
            જો (tile_off_fetch >= d_inner_per_thread) {                                           \
                tile_off_fetch = 0;                                                               \
                s_fetch++;                                                                        \
            }                                                                                     \
        } જ્યારે (0)

    // Queue weights and initial input tiles together so DDR reads overlap
    અચલ dma_addr_t src1_ddr = src1->data + ir0 * d_conv * કદ(દશાંશ);
    dma_queue_ધકેલો(dma_q, dma_make_data((uint8_t *) src1_raw, src1_ddr), weight_bytes, weight_bytes, weight_bytes, 1);

    SSM_CONV_ધકેલો_FETCH(0);
    જો (n_chunks > 1) {
        SSM_CONV_ધકેલો_FETCH(1);
    }

    dma_queue_કાઢો(dma_q);  // weights

    // Unpack/transpose src1_raw into src1_T {d_conv, d_inner_stride}
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);
    hvx_ssm_conv_unpack_to_T(src1_raw, src1_T, d_inner_per_thread, d_inner_stride, d_conv);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);

    અચલ uint32_t C_TILE = VLEN_FP32;

    // gather offsets: lane r reads channel r of the raw tile block
    uint32_t __attribute__((aligned(VLEN))) gather_off[VLEN_FP32];
    માટે (uint32_t r = 0; r < VLEN_FP32; ++r) {
        gather_off[r] = r * ncs * કદ(દશાંશ);
    }
    અચલ HVX_Vector vv = *(અચલ HVX_Vector *) gather_off;

    uint32_t i3       = 0;
    uint32_t tile_off = 0;

    માટે (uint32_t c = 0; c < n_chunks; ++c) {
        અચલ uint32_t tile_n = MIN(d_inner_tile, d_inner_per_thread - tile_off);

        અચલ દશાંશ * restrict raw = src0_tile_raw[c & 1];
        દશાંશ * restrict       out = dst_tile[c & 1];

        જો (c >= 2) {
            dma_queue_કાઢો(dma_q);  // writeback of chunk c-2, frees dst_tile[c & 1]
        }
        dma_queue_કાઢો(dma_q);      // fetch chunk c

        // Channel block outer, token inner: the taps and the sliding window stay in
        // registers, so a new output row costs one src0_T load.
        અચલ uint32_t dst_tile_stride = hex_round_up(tile_n, C_TILE);

        માટે (uint32_t cb = 0; cb < tile_n; cb += C_TILE) {
            અચલ uint32_t cb_n = MIN(C_TILE, tile_n - cb);

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) tile_off);
            hvx_ssm_conv_transpose_block(raw + (size_t) cb * ncs, src0_T, ncs, cb_n, vv);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) tile_off);

            અચલ દશાંશ * restrict wp = src1_T + tile_off + cb;
            દશાંશ * restrict       op = out + cb;

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) tile_off);
            જો (d_conv == 4) {
                અચલ HVX_Vector w0 = *(અચલ HVX_Vector *) (wp);
                અચલ HVX_Vector w1 = *(અચલ HVX_Vector *) (wp + d_inner_stride);
                અચલ HVX_Vector w2 = *(અચલ HVX_Vector *) (wp + 2 * d_inner_stride);
                અચલ HVX_Vector w3 = *(અચલ HVX_Vector *) (wp + 3 * d_inner_stride);

                HVX_Vector x0 = *(અચલ HVX_Vector *) (src0_T);
                HVX_Vector x1 = *(અચલ HVX_Vector *) (src0_T + C_TILE);
                HVX_Vector x2 = *(અચલ HVX_Vector *) (src0_T + 2 * C_TILE);

                માટે (uint32_t t = 0; t < n_t; ++t) {
                    અચલ HVX_Vector x3 = *(અચલ HVX_Vector *) (src0_T + (t + 3) * C_TILE);

                    HVX_Vector a = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(x0, w0),
                                                            Q6_Vqf32_vmpy_VsfVsf(x1, w1));
                    HVX_Vector b = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(x2, w2),
                                                            Q6_Vqf32_vmpy_VsfVsf(x3, w3));

                    *(HVX_Vector *) (op + t * dst_tile_stride) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(a, b));

                    x0 = x1;
                    x1 = x2;
                    x2 = x3;
                }
            } નહીંતર {
                માટે (uint32_t t = 0; t < n_t; ++t) {
                    HVX_Vector acc = hvx_vec_splat_f32(0.0f);
                    માટે (uint32_t j = 0; j < d_conv; ++j) {
                        HVX_Vector x = *(અચલ HVX_Vector *) (src0_T + (t + j) * C_TILE);
                        HVX_Vector w = *(અચલ HVX_Vector *) (wp + j * d_inner_stride);
                        acc          = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(x, w));
                    }
                    *(HVX_Vector *) (op + t * dst_tile_stride) = Q6_Vsf_equals_Vqf32(acc);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) tile_off);
        }

        // Writeback dst_tile VTCM -> DDR via 2D DMA
        dma_queue_ધકેલો(dma_q,
                       dma_make_data(dst->data + i3 * dst_stride_seq_bytes + (ir0 + tile_off) * કદ(દશાંશ),
                                     (uint8_t *) out),
                       dst_stride_token_bytes, dst_tile_stride * કદ(દશાંશ), tile_n * કદ(દશાંશ), n_t);

        જો (c + 2 < n_chunks) {
            SSM_CONV_ધકેલો_FETCH(c + 2);
        }

        tile_off += d_inner_tile;
        જો (tile_off >= d_inner_per_thread) {
            tile_off = 0;
            i3++;
        }
    }

    માટે (uint32_t k = MIN(n_chunks, 2); k > 0; --k) {
        dma_queue_કાઢો(dma_q);  // drain the last writebacks
    }

    #undef SSM_CONV_ધકેલો_FETCH

    FARF(ઉપલું_2, "ssm-conv-f32-prefill %d/%d: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0], dst->ne[1],
         dst->ne[2], dst->ne[3]);
}

પૂર્ણાંક op_ssm_conv_f32(સંરચના htp_ops_context * octx) {
    અચલ સંરચના htp_tensor * src0 = octx->src[0];
    અચલ સંરચના htp_tensor * src1 = octx->src[1];
    અચલ સંરચના htp_tensor * dst  = octx->dst;

    જો (src0->type != HTP_TYPE_F32 || src1->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        પાછા HTP_STATUS_NO_SUPPORT;
    }

    અચલ સંરચના htp_ssm_conv_kernel_params * kparams = (અચલ સંરચના htp_ssm_conv_kernel_params *) octx->kernel_params;


    જો (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        પાછા HTP_STATUS_INVAL_PARAMS;
    }

    uint32_t row_start = 0;
    uint32_t nrows     = kparams->d_inner;

    જો (octx->ctx->mdev.ગણતરી > 1) {
        અચલ uint32_t elems_per_chunk = VLEN_FP32;
        અચલ સંરચના htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            kparams->d_inner,
            htp_tensor_mdev_data_aligned(dst) ? elems_per_chunk : 0,
            octx->ctx->mdev.idx,
            octx->ctx->mdev.ગણતરી,
            &octx->ctx->mdev.ગણતરી_div
        );
        row_start = range.start;
        nrows     = range.ગણતરી;
    }

    જો (nrows == 0) {
        પાછા HTP_STATUS_OK;
    }

    જો (kparams->vtcm_size > octx->ctx->vtcm_size) {
        પાછા HTP_STATUS_VTCM_TOO_SMALL;
    }

    અચલ uint32_t n_threads = octx->n_threads;

    octx->src0_spad.size_per_thread = kparams->vtcm_src0_size_per_thread;
    octx->src1_spad.size_per_thread = kparams->vtcm_src1_size_per_thread;
    octx->dst_spad.size_per_thread  = kparams->vtcm_dst_size_per_thread;

    octx->src0_spad.size = kparams->vtcm_src0_size;
    octx->src1_spad.size = kparams->vtcm_src1_size;
    octx->dst_spad.size  = kparams->vtcm_dst_size;

    octx->src0_spad.data = octx->ctx->vtcm_base;
    octx->src1_spad.data = octx->src0_spad.data + octx->src0_spad.size;
    octx->dst_spad.data  = octx->src1_spad.data + octx->src1_spad.size;
    octx->src0_spad.src  = ખાલી;
    octx->src1_spad.src  = ખાલી;
    octx->dst_spad.src   = ખાલી;

    અચલ uint32_t raw_rpt            = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);
    અચલ uint32_t d_inner_per_thread = hex_round_up(raw_rpt, VLEN_FP32);

    સંરચના htp_ssm_conv_context scctx = {
        .octx             = octx,
        .kparams          = kparams,
        .nrows_per_thread = d_inner_per_thread,
        .d_inner_tile     = kparams->d_inner_tile,
        .row_start        = row_start,
        .nrows            = nrows,
    };

    FARF(ઉપલું_2, "ssm-conv-f32: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : mode %s\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         kparams->n_t == 1 ? "decode" : "prefill");

    જો (kparams->n_t == 1) {
        work_queue_run(octx->ctx->work_queue, ssm_conv_thread_f32_decode, &scctx, n_threads);
    } નહીંતર {
        work_queue_run(octx->ctx->work_queue, ssm_conv_thread_f32_prefill, &scctx, n_threads);
    }

    પાછા HTP_STATUS_OK;
}

પૂર્ણાંક op_ssm_conv(સંરચના htp_ops_context * octx) {
    અચલ સંરચના htp_tensor * dst = octx->dst;

    પસંદગી (dst->type) {
        સ્થિતિ HTP_TYPE_F32:
            પાછા op_ssm_conv_f32(octx);
        મૂળભૂત:
            પાછા HTP_STATUS_NO_SUPPORT;
    }
}
