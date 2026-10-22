#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#অন্তর্ভাৱ <HAP_farf.h>
#অন্তর্ভাৱ <HAP_mem.h>
#অন্তর্ভাৱ <HAP_ps.h>
#অন্তর্ভাৱ <hexagon_protos.h>
#অন্তর্ভাৱ <hexagon_types.h>
#অন্তর্ভাৱ <math.h>
#অন্তর্ভাৱ <qurt_thread.h>
#অন্তর্ভাৱ <string.h>

#define GGML_COMMON_DECL_C
#অন্তর্ভাৱ "ggml-common.h"
#অন্তর্ভাৱ "htp-ctx.h"
#অন্তর্ভাৱ "dma-queue.h"
#অন্তর্ভাৱ "hex-profile.h"
#অন্তর্ভাৱ "htp-ops.h"
#অন্তর্ভাৱ "htp-tensor.h"
#অন্তর্ভাৱ "hvx-utils.h"
#অন্তর্ভাৱ "ssm-conv.h"

সংরচনা htp_ssm_conv_context {
    সংরচনা htp_ops_context *                  octx;
    ধ্রুৱ সংরচনা htp_ssm_conv_kernel_params * kparams;
    uint32_t                                  nrows_per_thread;
    uint32_t                                  d_inner_tile;
    uint32_t                                  row_start;
    uint32_t                                  nrows;
};

// In-register 32x32 fp32 transpose using std 5-stage HVX vshuff butterfly.
স্থির অন্তঃস্থ শূন্য hvx_transpose_32x32_f32(HVX_Vector m[32]) {
    HVX_Vector tmp[32];

    // Stage 0 (R = -4): pair (2i, 2i+1) for i = 0..15. m -> tmp.
    বাবে (পূর্ণাঙ্ক i = 0; i < 16; ++i) {
        HVX_VectorPair p = Q6_W_vshuff_VVR(m[2*i + 1], m[2*i], -4);
        tmp[2*i + 0] = Q6_V_lo_W(p);
        tmp[2*i + 1] = Q6_V_hi_W(p);
    }

    // Stage 1 (R = -8): per block of 4, pair (b+0, b+2) and (b+1, b+3). tmp -> m.
    বাবে (পূর্ণাঙ্ক b = 0; b < 32; b += 4) {
        HVX_VectorPair p0 = Q6_W_vshuff_VVR(tmp[b + 2], tmp[b + 0], -8);
        HVX_VectorPair p1 = Q6_W_vshuff_VVR(tmp[b + 3], tmp[b + 1], -8);
        m[b + 0] = Q6_V_lo_W(p0); m[b + 1] = Q6_V_hi_W(p0);
        m[b + 2] = Q6_V_lo_W(p1); m[b + 3] = Q6_V_hi_W(p1);
    }

    // Stage 2 (R = -16): per block of 8, pair (b+i, b+i+4) for i = 0..3. m -> tmp.
    বাবে (পূর্ণাঙ্ক b = 0; b < 32; b += 8) {
        বাবে (পূর্ণাঙ্ক i = 0; i < 4; ++i) {
            HVX_VectorPair p = Q6_W_vshuff_VVR(m[b + i + 4], m[b + i], -16);
            tmp[b + 2*i + 0] = Q6_V_lo_W(p);
            tmp[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }

    // Stage 3 (R = -32): per block of 16, pair (b+i, b+i+8) for i = 0..7. tmp -> m.
    বাবে (পূর্ণাঙ্ক b = 0; b < 32; b += 16) {
        বাবে (পূর্ণাঙ্ক i = 0; i < 8; ++i) {
            HVX_VectorPair p = Q6_W_vshuff_VVR(tmp[b + i + 8], tmp[b + i], -32);
            m[b + 2*i + 0] = Q6_V_lo_W(p);
            m[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }

    // Stage 4 (R = -64): pair (i, i+16) for i = 0..15. m -> tmp -> m.
    বাবে (পূর্ণাঙ্ক i = 0; i < 16; ++i) {
        HVX_VectorPair p = Q6_W_vshuff_VVR(m[i + 16], m[i], -64);
        tmp[2 * i + 0]   = Q6_V_lo_W(p);
        tmp[2 * i + 1]   = Q6_V_hi_W(p);
    }

    বাবে (পূর্ণাঙ্ক i = 0; i < 32; ++i) {
        m[i] = tmp[i];
    }
}

// HVX deinterleave for d_conv == 4: channel-major raw VTCM -> tap-major T VTCM
স্থির অন্তঃস্থ শূন্য hvx_ssm_conv_unpack_to_T_4(ধ্রুৱ ভিন্ন * raw, ভিন্ন * T, uint32_t d_inner_per_thread, uint32_t d_inner_stride) {
    বাবে (uint32_t cb = 0; cb < d_inner_per_thread; cb += VLEN_FP32) {
        HVX_Vector v0 = *(ধ্রুৱ HVX_Vector *)(raw + (cb + 0) * 4);
        HVX_Vector v1 = *(ধ্রুৱ HVX_Vector *)(raw + (cb + 8) * 4);
        HVX_Vector v2 = *(ধ্রুৱ HVX_Vector *)(raw + (cb + 16) * 4);
        HVX_Vector v3 = *(ধ্রুৱ HVX_Vector *)(raw + (cb + 24) * 4);

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
স্থির অন্তঃস্থ শূন্য hvx_ssm_conv_unpack_to_T_gen(ধ্রুৱ ভিন্ন * raw, ভিন্ন * T, uint32_t d_inner_per_thread, uint32_t d_inner_stride, uint32_t d_conv) {
    uint32_t __attribute__((aligned(VLEN))) mask_buf[VLEN_FP32] = { 0 };
    বাবে (uint32_t j = 0; j < d_conv; ++j) {
        mask_buf[j] = 0xFFFFFFFF;
    }
    ধ্রুৱ HVX_Vector mask = *(ধ্রুৱ HVX_Vector *) mask_buf;

    বাবে (uint32_t cb = 0; cb < d_inner_per_thread; cb += VLEN_FP32) {
        ধ্রুৱ uint32_t cb_n = MIN(VLEN_FP32, d_inner_per_thread - cb);
        HVX_Vector sub[32];
        বাবে (uint32_t r = 0; r < cb_n; ++r) {
            ধ্রুৱ ভিন্ন * ch_ptr = raw + (cb + r) * d_conv;
            sub[r] = Q6_V_vand_VV(*(ধ্রুৱ HVX_UVector *) ch_ptr, mask);
        }
        বাবে (uint32_t r = cb_n; r < 32; ++r) {
            sub[r] = hvx_vec_splat_f32(0.0f);
        }

        hvx_transpose_32x32_f32(sub);

        বাবে (uint32_t j = 0; j < d_conv; ++j) {
            *(HVX_Vector *)(T + j * d_inner_stride + cb) = sub[j];
        }
    }
}

স্থির অন্তঃস্থ শূন্য hvx_ssm_conv_unpack_to_T(ধ্রুৱ ভিন্ন * raw, ভিন্ন * T, uint32_t d_inner_per_thread, uint32_t d_inner_stride, uint32_t d_conv) {
    যদি (d_conv == 4 && (d_inner_per_thread % VLEN_FP32 == 0)) {
        hvx_ssm_conv_unpack_to_T_4(raw, T, d_inner_per_thread, d_inner_stride);
    } অন্যথা {
        hvx_ssm_conv_unpack_to_T_gen(raw, T, d_inner_per_thread, d_inner_stride, d_conv);
    }
}

// Decode dot product specialization for d_conv == 4: multiply in the raw channel-major layout,
// then deinterleave the products so each vector holds one tap of 32 channels, and sum.
// Keeps both operands in DMA layout - no transpose, no scratch.
স্থির অন্তঃস্থ শূন্য hvx_ssm_conv_decode_4(ধ্রুৱ ভিন্ন * x, ধ্রুৱ ভিন্ন * w, ভিন্ন * out, uint32_t n_ch) {
    বাবে (uint32_t cb = 0; cb < n_ch; cb += VLEN_FP32) {
        ধ্রুৱ ভিন্ন * xp = x + cb * 4;
        ধ্রুৱ ভিন্ন * wp = w + cb * 4;

        HVX_Vector p0 = Q6_Vqf32_vmpy_VsfVsf(*(ধ্রুৱ HVX_Vector *)(xp +  0), *(ধ্রুৱ HVX_Vector *)(wp +  0));
        HVX_Vector p1 = Q6_Vqf32_vmpy_VsfVsf(*(ধ্রুৱ HVX_Vector *)(xp + 32), *(ধ্রুৱ HVX_Vector *)(wp + 32));
        HVX_Vector p2 = Q6_Vqf32_vmpy_VsfVsf(*(ধ্রুৱ HVX_Vector *)(xp + 64), *(ধ্রুৱ HVX_Vector *)(wp + 64));
        HVX_Vector p3 = Q6_Vqf32_vmpy_VsfVsf(*(ধ্রুৱ HVX_Vector *)(xp + 96), *(ধ্রুৱ HVX_Vector *)(wp + 96));

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
স্থির অন্তঃস্থ শূন্য hvx_ssm_conv_transpose_block(ধ্রুৱ ভিন্ন * raw_block,
                                                ভিন্ন *       T,
                                                uint32_t      ncs,
                                                uint32_t      cb_n,
                                                HVX_Vector    vv) {
    ধ্রুৱ size_t   base = (size_t) raw_block;
    ধ্রুৱ uint32_t mu   = cb_n * ncs * পরিমাণ(ভিন্ন) - 1;

    বাবে (uint32_t t = 0; t < ncs; ++t) {
        Q6_vgather_ARMVw((HVX_Vector *) (T + (size_t) t * VLEN_FP32), base + t * পরিমাণ(ভিন্ন), mu, vv);
    }
}

// Single-token decode worker (n_t == 1)
স্থির শূন্য ssm_conv_thread_f32_decode(unsigned পূর্ণাঙ্ক nth, unsigned পূর্ণাঙ্ক ith, শূন্য * data) {
    সংরচনা htp_ssm_conv_context *             scctx   = (সংরচনা htp_ssm_conv_context *) data;
    সংরচনা htp_ops_context *                  octx    = scctx->octx;
    ধ্রুৱ সংরচনা htp_ssm_conv_kernel_params * kparams = scctx->kparams;

    ধ্রুৱ সংরচনা htp_tensor * restrict src0 = octx->src[0];
    ধ্রুৱ সংরচনা htp_tensor * restrict src1 = octx->src[1];
    ধ্রুৱ সংরচনা htp_tensor * restrict dst  = octx->dst;

    dma_queue * dma_q = octx->ctx->dma[ith];

    ধ্রুৱ uint32_t d_conv  = kparams->d_conv;
    ধ্রুৱ uint32_t d_inner = kparams->d_inner;
    ধ্রুৱ uint32_t n_s     = kparams->n_s;

    ধ্রুৱ uint32_t dr  = scctx->nrows_per_thread;
    ধ্রুৱ uint32_t ir0 = scctx->row_start + dr * ith;
    ধ্রুৱ uint32_t ir1 = MIN(ir0 + dr, scctx->row_start + scctx->nrows);

    যদি (ir0 >= ir1) {
        ঘূৰাই;
    }

    ধ্রুৱ uint32_t d_inner_per_thread = ir1 - ir0;
    ধ্রুৱ uint32_t d_inner_stride     = hex_round_up(d_inner_per_thread, VLEN_FP32);
    ধ্রুৱ uint32_t d_inner_tile       = scctx->d_inner_tile;

    ধ্রুৱ size_t src0_stride_inner_bytes = src0->nb[1];
    ধ্রুৱ size_t src0_stride_seq_bytes   = src0->nb[2];
    ধ্রুৱ size_t dst_stride_seq_bytes    = dst->nb[2];

    uint8_t * src1_spad_base = octx->src1_spad.data + ith * octx->src1_spad.size_per_thread;
    uint8_t * src0_spad_base = octx->src0_spad.data + ith * octx->src0_spad.size_per_thread;
    uint8_t * dst_spad_base  = octx->dst_spad.data  + ith * octx->dst_spad.size_per_thread;

    ধ্রুৱ size_t weight_bytes    = (size_t) d_inner_per_thread * d_conv * পরিমাণ(ভিন্ন);
    ধ্রুৱ size_t weight_raw_size = hex_round_up(weight_bytes, 128);

    ভিন্ন * src1_raw = (ভিন্ন *) src1_spad_base;
    ভিন্ন * src1_T   = (ভিন্ন *) (src1_spad_base + weight_raw_size);

    ধ্রুৱ size_t src0_tile_raw_bytes = hex_round_up(d_inner_tile * d_conv * পরিমাণ(ভিন্ন), 128);
    ধ্রুৱ size_t dst_tile_bytes      = hex_round_up(d_inner_tile * পরিমাণ(ভিন্ন), 128);

    ভিন্ন * src0_tile_raw[2] = { (ভিন্ন *) src0_spad_base, (ভিন্ন *) (src0_spad_base + src0_tile_raw_bytes) };
    ভিন্ন * src0_T           = (ভিন্ন *) (src0_spad_base + 2 * src0_tile_raw_bytes);

    ভিন্ন * dst_tile[2] = { (ভিন্ন *) dst_spad_base, (ভিন্ন *) (dst_spad_base + dst_tile_bytes) };

    সংরচনা htp_thread_trace * tr = &octx->ctx->trace[ith];

    // raw_dot keeps both operands in the DMA layout, so src1 needs no prep pass
    ধ্রুৱ bool raw_dot = (d_conv == 4) && (d_inner_per_thread % VLEN_FP32 == 0);

    ধ্রুৱ uint32_t n_tiles   = (d_inner_per_thread + d_inner_tile - 1) / d_inner_tile;
    ধ্রুৱ uint32_t n_chunks  = n_s * n_tiles;
    ধ্রুৱ size_t   row_bytes = d_conv * পরিমাণ(ভিন্ন);

    uint32_t s_fetch        = 0;
    uint32_t tile_off_fetch = 0;

    #define SSM_CONV_DECODE_ঠেলো_FETCH(c)                                                         \
        কৰা {                                                                                      \
            ধ্রুৱ uint32_t   cur_tile_n = MIN(d_inner_tile, d_inner_per_thread - tile_off_fetch); \
            ধ্রুৱ dma_addr_t fetch_ddr  = src0->data + s_fetch * src0_stride_seq_bytes +          \
                                          (ir0 + tile_off_fetch) * src0_stride_inner_bytes;       \
            dma_queue_ঠেলো(dma_q,                                                                 \
                           dma_make_data((uint8_t *) src0_tile_raw[(c) & 1], fetch_ddr),          \
                           row_bytes, src0_stride_inner_bytes, row_bytes, cur_tile_n);            \
            tile_off_fetch += d_inner_tile;                                                       \
            যদি (tile_off_fetch >= d_inner_per_thread) {                                           \
                tile_off_fetch = 0;                                                               \
                s_fetch++;                                                                        \
            }                                                                                     \
        } যেতিয়ালৈ (0)

    // Queue weights and initial input tiles together so DDR reads overlap
    ধ্রুৱ dma_addr_t src1_ddr = src1->data + ir0 * d_conv * পরিমাণ(ভিন্ন);
    dma_queue_ঠেলো(dma_q, dma_make_data((uint8_t *) src1_raw, src1_ddr), weight_bytes, weight_bytes, weight_bytes, 1);

    SSM_CONV_DECODE_ঠেলো_FETCH(0);
    যদি (n_chunks > 1) {
        SSM_CONV_DECODE_ঠেলো_FETCH(1);
    }

    dma_queue_বেৰ_কৰো(dma_q);  // weights

    যদি (!raw_dot) {
        // Unpack/transpose src1_raw into src1_T {d_conv, d_inner_stride}
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);
        hvx_ssm_conv_unpack_to_T(src1_raw, src1_T, d_inner_per_thread, d_inner_stride, d_conv);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);
    }

    uint32_t i3       = 0;
    uint32_t tile_off = 0;

    বাবে (uint32_t c = 0; c < n_chunks; ++c) {
        ধ্রুৱ uint32_t tile_n = MIN(d_inner_tile, d_inner_per_thread - tile_off);

        যদি (c >= 2) {
            dma_queue_বেৰ_কৰো(dma_q);  // writeback of chunk c-2, frees dst_tile[c & 1]
        }
        dma_queue_বেৰ_কৰো(dma_q);      // fetch chunk c

        ভিন্ন * restrict out = dst_tile[c & 1];

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i3);
        যদি (raw_dot) {
            ধ্রুৱ ভিন্ন * xp = src0_tile_raw[c & 1];
            ধ্রুৱ ভিন্ন * wp = src1_raw + tile_off * 4;
            hvx_ssm_conv_decode_4(xp, wp, out, tile_n);
        } অন্যথা {
            ধ্রুৱ uint32_t tile_stride = hex_round_up(tile_n, VLEN_FP32);
            hvx_ssm_conv_unpack_to_T(src0_tile_raw[c & 1], src0_T, tile_n, tile_stride, d_conv);

            বাবে (uint32_t cb = 0; cb < tile_n; cb += VLEN_FP32) {
                ধ্রুৱ uint32_t cb_n = MIN(VLEN_FP32, tile_n - cb);
                HVX_Vector acc = hvx_vec_splat_f32(0.0f);
                বাবে (uint32_t j = 0; j < d_conv; ++j) {
                    HVX_Vector x = *(ধ্রুৱ HVX_Vector *)(src0_T + j * tile_stride + cb);
                    HVX_Vector w = *(ধ্রুৱ HVX_Vector *)(src1_T + j * d_inner_stride + tile_off + cb);
                    acc          = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(x, w));
                }
                HVX_Vector y = Q6_Vsf_equals_Vqf32(acc);
                যদি (cb_n == VLEN_FP32) {
                    *(HVX_Vector *)(out + cb) = y;
                } অন্যথা {
                    hvx_vec_store_u(out + cb, cb_n * পরিমাণ(ভিন্ন), y);
                }
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i3);

        ধ্রুৱ dma_addr_t dst_ddr        = dst->data + i3 * dst_stride_seq_bytes + (ir0 + tile_off) * পরিমাণ(ভিন্ন);
        ধ্রুৱ size_t     tile_out_bytes = (size_t) tile_n * পরিমাণ(ভিন্ন);
        dma_queue_ঠেলো(dma_q, dma_make_data(dst_ddr, (uint8_t *) out),
                       tile_out_bytes, tile_out_bytes, tile_out_bytes, 1);

        যদি (c + 2 < n_chunks) {
            SSM_CONV_DECODE_ঠেলো_FETCH(c + 2);
        }

        tile_off += d_inner_tile;
        যদি (tile_off >= d_inner_per_thread) {
            tile_off = 0;
            i3++;
        }
    }

    বাবে (uint32_t k = MIN(n_chunks, 2); k > 0; --k) {
        dma_queue_বেৰ_কৰো(dma_q);  // drain the last writebacks
    }

    #undef SSM_CONV_DECODE_ঠেলো_FETCH

    FARF(উপৰেৰ_2, "ssm-conv-f32-decode %d/%d: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0], dst->ne[1],
         dst->ne[2], dst->ne[3]);
}

// Multi-token prefill worker (n_t > 1)
স্থির শূন্য ssm_conv_thread_f32_prefill(unsigned পূর্ণাঙ্ক nth, unsigned পূর্ণাঙ্ক ith, শূন্য * data) {
    সংরচনা htp_ssm_conv_context *             scctx   = (সংরচনা htp_ssm_conv_context *) data;
    সংরচনা htp_ops_context *                  octx    = scctx->octx;
    ধ্রুৱ সংরচনা htp_ssm_conv_kernel_params * kparams = scctx->kparams;

    ধ্রুৱ সংরচনা htp_tensor * restrict src0 = octx->src[0];
    ধ্রুৱ সংরচনা htp_tensor * restrict src1 = octx->src[1];
    ধ্রুৱ সংরচনা htp_tensor * restrict dst  = octx->dst;

    dma_queue * dma_q = octx->ctx->dma[ith];

    ধ্রুৱ uint32_t d_conv  = kparams->d_conv;
    ধ্রুৱ uint32_t d_inner = kparams->d_inner;
    ধ্রুৱ uint32_t n_t     = kparams->n_t;
    ধ্রুৱ uint32_t n_s     = kparams->n_s;
    ধ্রুৱ uint32_t ncs     = src0->ne[0];

    ধ্রুৱ uint32_t dr  = scctx->nrows_per_thread;
    ধ্রুৱ uint32_t ir0 = scctx->row_start + dr * ith;
    ধ্রুৱ uint32_t ir1 = MIN(ir0 + dr, scctx->row_start + scctx->nrows);

    যদি (ir0 >= ir1) {
        ঘূৰাই;
    }

    ধ্রুৱ uint32_t d_inner_per_thread = ir1 - ir0;
    ধ্রুৱ uint32_t d_inner_stride     = hex_round_up(d_inner_per_thread, VLEN_FP32);
    ধ্রুৱ uint32_t d_inner_tile       = scctx->d_inner_tile;

    ধ্রুৱ size_t src0_stride_inner_bytes = src0->nb[1];
    ধ্রুৱ size_t src0_stride_seq_bytes   = src0->nb[2];
    ধ্রুৱ size_t dst_stride_token_bytes  = dst->nb[1];
    ধ্রুৱ size_t dst_stride_seq_bytes    = dst->nb[2];

    uint8_t * src1_spad_base = octx->src1_spad.data + ith * octx->src1_spad.size_per_thread;
    uint8_t * src0_spad_base = octx->src0_spad.data + ith * octx->src0_spad.size_per_thread;
    uint8_t * dst_spad_base  = octx->dst_spad.data  + ith * octx->dst_spad.size_per_thread;

    ধ্রুৱ size_t weight_bytes    = (size_t) d_inner_per_thread * d_conv * পরিমাণ(ভিন্ন);
    ধ্রুৱ size_t weight_raw_size = hex_round_up(weight_bytes, 128);

    ভিন্ন * src1_raw = (ভিন্ন *) src1_spad_base;
    ভিন্ন * src1_T   = (ভিন্ন *) (src1_spad_base + weight_raw_size);

    // src0 spad holds two raw tiles (fetch of tile n+1 overlaps compute of tile n) plus
    // the transposed block. dst spad holds two tiles so a writeback can stay in flight.
    ধ্রুৱ size_t src0_tile_raw_bytes = hex_round_up(d_inner_tile * ncs * পরিমাণ(ভিন্ন), 128);
    ধ্রুৱ size_t dst_tile_bytes      = hex_round_up(d_inner_tile * n_t * পরিমাণ(ভিন্ন), 128);

    ভিন্ন * src0_tile_raw[2] = { (ভিন্ন *) src0_spad_base, (ভিন্ন *) (src0_spad_base + src0_tile_raw_bytes) };
    ভিন্ন * src0_T           = (ভিন্ন *) (src0_spad_base + 2 * src0_tile_raw_bytes);

    ভিন্ন * dst_tile[2] = { (ভিন্ন *) dst_spad_base, (ভিন্ন *) (dst_spad_base + dst_tile_bytes) };

    সংরচনা htp_thread_trace * tr = &octx->ctx->trace[ith];

    ধ্রুৱ uint32_t n_tiles   = (d_inner_per_thread + d_inner_tile - 1) / d_inner_tile;
    ধ্রুৱ uint32_t n_chunks  = n_s * n_tiles;
    ধ্রুৱ size_t   row_bytes = ncs * পরিমাণ(ভিন্ন);

    uint32_t s_fetch        = 0;
    uint32_t tile_off_fetch = 0;

    // Chunk c fetches into src0_tile_raw[c & 1] and writes back from dst_tile[c & 1].
    // Two fetches run ahead, so the queue order is F0 F1 W0 F2 W1 ... and pops follow it.
    #define SSM_CONV_ঠেলো_FETCH(c)                                                                \
        কৰা {                                                                                      \
            ধ্রুৱ uint32_t   cur_tile_n = MIN(d_inner_tile, d_inner_per_thread - tile_off_fetch); \
            ধ্রুৱ dma_addr_t fetch_ddr  = src0->data + s_fetch * src0_stride_seq_bytes +          \
                                          (ir0 + tile_off_fetch) * src0_stride_inner_bytes;       \
            dma_queue_ঠেলো(dma_q,                                                                 \
                           dma_make_data((uint8_t *) src0_tile_raw[(c) & 1], fetch_ddr),          \
                           row_bytes, src0_stride_inner_bytes, row_bytes, cur_tile_n);            \
            tile_off_fetch += d_inner_tile;                                                       \
            যদি (tile_off_fetch >= d_inner_per_thread) {                                           \
                tile_off_fetch = 0;                                                               \
                s_fetch++;                                                                        \
            }                                                                                     \
        } যেতিয়ালৈ (0)

    // Queue weights and initial input tiles together so DDR reads overlap
    ধ্রুৱ dma_addr_t src1_ddr = src1->data + ir0 * d_conv * পরিমাণ(ভিন্ন);
    dma_queue_ঠেলো(dma_q, dma_make_data((uint8_t *) src1_raw, src1_ddr), weight_bytes, weight_bytes, weight_bytes, 1);

    SSM_CONV_ঠেলো_FETCH(0);
    যদি (n_chunks > 1) {
        SSM_CONV_ঠেলো_FETCH(1);
    }

    dma_queue_বেৰ_কৰো(dma_q);  // weights

    // Unpack/transpose src1_raw into src1_T {d_conv, d_inner_stride}
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);
    hvx_ssm_conv_unpack_to_T(src1_raw, src1_T, d_inner_per_thread, d_inner_stride, d_conv);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);

    ধ্রুৱ uint32_t C_TILE = VLEN_FP32;

    // gather offsets: lane r reads channel r of the raw tile block
    uint32_t __attribute__((aligned(VLEN))) gather_off[VLEN_FP32];
    বাবে (uint32_t r = 0; r < VLEN_FP32; ++r) {
        gather_off[r] = r * ncs * পরিমাণ(ভিন্ন);
    }
    ধ্রুৱ HVX_Vector vv = *(ধ্রুৱ HVX_Vector *) gather_off;

    uint32_t i3       = 0;
    uint32_t tile_off = 0;

    বাবে (uint32_t c = 0; c < n_chunks; ++c) {
        ধ্রুৱ uint32_t tile_n = MIN(d_inner_tile, d_inner_per_thread - tile_off);

        ধ্রুৱ ভিন্ন * restrict raw = src0_tile_raw[c & 1];
        ভিন্ন * restrict       out = dst_tile[c & 1];

        যদি (c >= 2) {
            dma_queue_বেৰ_কৰো(dma_q);  // writeback of chunk c-2, frees dst_tile[c & 1]
        }
        dma_queue_বেৰ_কৰো(dma_q);      // fetch chunk c

        // Channel block outer, token inner: the taps and the sliding window stay in
        // registers, so a new output row costs one src0_T load.
        ধ্রুৱ uint32_t dst_tile_stride = hex_round_up(tile_n, C_TILE);

        বাবে (uint32_t cb = 0; cb < tile_n; cb += C_TILE) {
            ধ্রুৱ uint32_t cb_n = MIN(C_TILE, tile_n - cb);

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) tile_off);
            hvx_ssm_conv_transpose_block(raw + (size_t) cb * ncs, src0_T, ncs, cb_n, vv);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) tile_off);

            ধ্রুৱ ভিন্ন * restrict wp = src1_T + tile_off + cb;
            ভিন্ন * restrict       op = out + cb;

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) tile_off);
            যদি (d_conv == 4) {
                ধ্রুৱ HVX_Vector w0 = *(ধ্রুৱ HVX_Vector *) (wp);
                ধ্রুৱ HVX_Vector w1 = *(ধ্রুৱ HVX_Vector *) (wp + d_inner_stride);
                ধ্রুৱ HVX_Vector w2 = *(ধ্রুৱ HVX_Vector *) (wp + 2 * d_inner_stride);
                ধ্রুৱ HVX_Vector w3 = *(ধ্রুৱ HVX_Vector *) (wp + 3 * d_inner_stride);

                HVX_Vector x0 = *(ধ্রুৱ HVX_Vector *) (src0_T);
                HVX_Vector x1 = *(ধ্রুৱ HVX_Vector *) (src0_T + C_TILE);
                HVX_Vector x2 = *(ধ্রুৱ HVX_Vector *) (src0_T + 2 * C_TILE);

                বাবে (uint32_t t = 0; t < n_t; ++t) {
                    ধ্রুৱ HVX_Vector x3 = *(ধ্রুৱ HVX_Vector *) (src0_T + (t + 3) * C_TILE);

                    HVX_Vector a = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(x0, w0),
                                                            Q6_Vqf32_vmpy_VsfVsf(x1, w1));
                    HVX_Vector b = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(x2, w2),
                                                            Q6_Vqf32_vmpy_VsfVsf(x3, w3));

                    *(HVX_Vector *) (op + t * dst_tile_stride) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(a, b));

                    x0 = x1;
                    x1 = x2;
                    x2 = x3;
                }
            } অন্যথা {
                বাবে (uint32_t t = 0; t < n_t; ++t) {
                    HVX_Vector acc = hvx_vec_splat_f32(0.0f);
                    বাবে (uint32_t j = 0; j < d_conv; ++j) {
                        HVX_Vector x = *(ধ্রুৱ HVX_Vector *) (src0_T + (t + j) * C_TILE);
                        HVX_Vector w = *(ধ্রুৱ HVX_Vector *) (wp + j * d_inner_stride);
                        acc          = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(x, w));
                    }
                    *(HVX_Vector *) (op + t * dst_tile_stride) = Q6_Vsf_equals_Vqf32(acc);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) tile_off);
        }

        // Writeback dst_tile VTCM -> DDR via 2D DMA
        dma_queue_ঠেলো(dma_q,
                       dma_make_data(dst->data + i3 * dst_stride_seq_bytes + (ir0 + tile_off) * পরিমাণ(ভিন্ন),
                                     (uint8_t *) out),
                       dst_stride_token_bytes, dst_tile_stride * পরিমাণ(ভিন্ন), tile_n * পরিমাণ(ভিন্ন), n_t);

        যদি (c + 2 < n_chunks) {
            SSM_CONV_ঠেলো_FETCH(c + 2);
        }

        tile_off += d_inner_tile;
        যদি (tile_off >= d_inner_per_thread) {
            tile_off = 0;
            i3++;
        }
    }

    বাবে (uint32_t k = MIN(n_chunks, 2); k > 0; --k) {
        dma_queue_বেৰ_কৰো(dma_q);  // drain the last writebacks
    }

    #undef SSM_CONV_ঠেলো_FETCH

    FARF(উপৰেৰ_2, "ssm-conv-f32-prefill %d/%d: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0], dst->ne[1],
         dst->ne[2], dst->ne[3]);
}

পূর্ণাঙ্ক op_ssm_conv_f32(সংরচনা htp_ops_context * octx) {
    ধ্রুৱ সংরচনা htp_tensor * src0 = octx->src[0];
    ধ্রুৱ সংরচনা htp_tensor * src1 = octx->src[1];
    ধ্রুৱ সংরচনা htp_tensor * dst  = octx->dst;

    যদি (src0->type != HTP_TYPE_F32 || src1->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        ঘূৰাই HTP_STATUS_NO_SUPPORT;
    }

    ধ্রুৱ সংরচনা htp_ssm_conv_kernel_params * kparams = (ধ্রুৱ সংরচনা htp_ssm_conv_kernel_params *) octx->kernel_params;


    যদি (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        ঘূৰাই HTP_STATUS_INVAL_PARAMS;
    }

    uint32_t row_start = 0;
    uint32_t nrows     = kparams->d_inner;

    যদি (octx->ctx->mdev.গুনতি > 1) {
        ধ্রুৱ uint32_t elems_per_chunk = VLEN_FP32;
        ধ্রুৱ সংরচনা htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            kparams->d_inner,
            htp_tensor_mdev_data_aligned(dst) ? elems_per_chunk : 0,
            octx->ctx->mdev.idx,
            octx->ctx->mdev.গুনতি,
            &octx->ctx->mdev.গুনতি_div
        );
        row_start = range.start;
        nrows     = range.গুনতি;
    }

    যদি (nrows == 0) {
        ঘূৰাই HTP_STATUS_OK;
    }

    যদি (kparams->vtcm_size > octx->ctx->vtcm_size) {
        ঘূৰাই HTP_STATUS_VTCM_TOO_SMALL;
    }

    ধ্রুৱ uint32_t n_threads = octx->n_threads;

    octx->src0_spad.size_per_thread = kparams->vtcm_src0_size_per_thread;
    octx->src1_spad.size_per_thread = kparams->vtcm_src1_size_per_thread;
    octx->dst_spad.size_per_thread  = kparams->vtcm_dst_size_per_thread;

    octx->src0_spad.size = kparams->vtcm_src0_size;
    octx->src1_spad.size = kparams->vtcm_src1_size;
    octx->dst_spad.size  = kparams->vtcm_dst_size;

    octx->src0_spad.data = octx->ctx->vtcm_base;
    octx->src1_spad.data = octx->src0_spad.data + octx->src0_spad.size;
    octx->dst_spad.data  = octx->src1_spad.data + octx->src1_spad.size;
    octx->src0_spad.src  = রিক্ত;
    octx->src1_spad.src  = রিক্ত;
    octx->dst_spad.src   = রিক্ত;

    ধ্রুৱ uint32_t raw_rpt            = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);
    ধ্রুৱ uint32_t d_inner_per_thread = hex_round_up(raw_rpt, VLEN_FP32);

    সংরচনা htp_ssm_conv_context scctx = {
        .octx             = octx,
        .kparams          = kparams,
        .nrows_per_thread = d_inner_per_thread,
        .d_inner_tile     = kparams->d_inner_tile,
        .row_start        = row_start,
        .nrows            = nrows,
    };

    FARF(উপৰেৰ_2, "ssm-conv-f32: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : mode %s\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         kparams->n_t == 1 ? "decode" : "prefill");

    যদি (kparams->n_t == 1) {
        work_queue_run(octx->ctx->work_queue, ssm_conv_thread_f32_decode, &scctx, n_threads);
    } অন্যথা {
        work_queue_run(octx->ctx->work_queue, ssm_conv_thread_f32_prefill, &scctx, n_threads);
    }

    ঘূৰাই HTP_STATUS_OK;
}

পূর্ণাঙ্ক op_ssm_conv(সংরচনা htp_ops_context * octx) {
    ধ্রুৱ সংরচনা htp_tensor * dst = octx->dst;

    ৱিকল্প (dst->type) {
        পক্ষ HTP_TYPE_F32:
            ঘূৰাই op_ssm_conv_f32(octx);
        স্ৱভাৱ:
            ঘূৰাই HTP_STATUS_NO_SUPPORT;
    }
}
