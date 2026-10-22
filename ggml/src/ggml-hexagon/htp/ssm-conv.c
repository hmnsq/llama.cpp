#आशय clang diagnostic ignored "-Wunused-variable"
#आशय clang diagnostic ignored "-Wunused-function"
#आशय clang diagnostic ignored "-Wunused-but-set-variable"

#समावेश <HAP_farf.h>
#समावेश <HAP_mem.h>
#समावेश <HAP_ps.h>
#समावेश <hexagon_protos.h>
#समावेश <hexagon_types.h>
#समावेश <math.h>
#समावेश <qurt_thread.h>
#समावेश <string.h>

#घोषणा GGML_COMMON_DECL_C
#समावेश "ggml-common.h"
#समावेश "htp-ctx.h"
#समावेश "dma-queue.h"
#समावेश "hex-profile.h"
#समावेश "htp-ops.h"
#समावेश "htp-tensor.h"
#समावेश "hvx-utils.h"
#समावेश "ssm-conv.h"

काष्ठा htp_ssm_conv_context {
    काष्ठा htp_ops_context *                  octx;
    स्थिर काष्ठा htp_ssm_conv_kernel_params * kparams;
    uint32_t                                  nrows_per_thread;
    uint32_t                                  d_inner_tile;
    uint32_t                                  row_start;
    uint32_t                                  nrows;
};

// In-register 32x32 fp32 transpose using std 5-stage HVX vshuff butterfly.
अटल अंतरभूत व्योम hvx_transpose_32x32_f32(HVX_Vector m[32]) {
    HVX_Vector tmp[32];

    // Stage 0 (R = -4): pair (2i, 2i+1) for i = 0..15. m -> tmp.
    क्रम (पूर्णांक i = 0; i < 16; ++i) {
        HVX_VectorPair p = Q6_W_vshuff_VVR(m[2*i + 1], m[2*i], -4);
        tmp[2*i + 0] = Q6_V_lo_W(p);
        tmp[2*i + 1] = Q6_V_hi_W(p);
    }

    // Stage 1 (R = -8): per block of 4, pair (b+0, b+2) and (b+1, b+3). tmp -> m.
    क्रम (पूर्णांक b = 0; b < 32; b += 4) {
        HVX_VectorPair p0 = Q6_W_vshuff_VVR(tmp[b + 2], tmp[b + 0], -8);
        HVX_VectorPair p1 = Q6_W_vshuff_VVR(tmp[b + 3], tmp[b + 1], -8);
        m[b + 0] = Q6_V_lo_W(p0); m[b + 1] = Q6_V_hi_W(p0);
        m[b + 2] = Q6_V_lo_W(p1); m[b + 3] = Q6_V_hi_W(p1);
    }

    // Stage 2 (R = -16): per block of 8, pair (b+i, b+i+4) for i = 0..3. m -> tmp.
    क्रम (पूर्णांक b = 0; b < 32; b += 8) {
        क्रम (पूर्णांक i = 0; i < 4; ++i) {
            HVX_VectorPair p = Q6_W_vshuff_VVR(m[b + i + 4], m[b + i], -16);
            tmp[b + 2*i + 0] = Q6_V_lo_W(p);
            tmp[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }

    // Stage 3 (R = -32): per block of 16, pair (b+i, b+i+8) for i = 0..7. tmp -> m.
    क्रम (पूर्णांक b = 0; b < 32; b += 16) {
        क्रम (पूर्णांक i = 0; i < 8; ++i) {
            HVX_VectorPair p = Q6_W_vshuff_VVR(tmp[b + i + 8], tmp[b + i], -32);
            m[b + 2*i + 0] = Q6_V_lo_W(p);
            m[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }

    // Stage 4 (R = -64): pair (i, i+16) for i = 0..15. m -> tmp -> m.
    क्रम (पूर्णांक i = 0; i < 16; ++i) {
        HVX_VectorPair p = Q6_W_vshuff_VVR(m[i + 16], m[i], -64);
        tmp[2 * i + 0]   = Q6_V_lo_W(p);
        tmp[2 * i + 1]   = Q6_V_hi_W(p);
    }

    क्रम (पूर्णांक i = 0; i < 32; ++i) {
        m[i] = tmp[i];
    }
}

// HVX deinterleave for d_conv == 4: channel-major raw VTCM -> tap-major T VTCM
अटल अंतरभूत व्योम hvx_ssm_conv_unpack_to_T_4(स्थिर भग्न * raw, भग्न * T, uint32_t d_inner_per_thread, uint32_t d_inner_stride) {
    क्रम (uint32_t cb = 0; cb < d_inner_per_thread; cb += VLEN_FP32) {
        HVX_Vector v0 = *(स्थिर HVX_Vector *)(raw + (cb + 0) * 4);
        HVX_Vector v1 = *(स्थिर HVX_Vector *)(raw + (cb + 8) * 4);
        HVX_Vector v2 = *(स्थिर HVX_Vector *)(raw + (cb + 16) * 4);
        HVX_Vector v3 = *(स्थिर HVX_Vector *)(raw + (cb + 24) * 4);

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
अटल अंतरभूत व्योम hvx_ssm_conv_unpack_to_T_gen(स्थिर भग्न * raw, भग्न * T, uint32_t d_inner_per_thread, uint32_t d_inner_stride, uint32_t d_conv) {
    uint32_t __attribute__((aligned(VLEN))) mask_buf[VLEN_FP32] = { 0 };
    क्रम (uint32_t j = 0; j < d_conv; ++j) {
        mask_buf[j] = 0xFFFFFFFF;
    }
    स्थिर HVX_Vector mask = *(स्थिर HVX_Vector *) mask_buf;

    क्रम (uint32_t cb = 0; cb < d_inner_per_thread; cb += VLEN_FP32) {
        स्थिर uint32_t cb_n = MIN(VLEN_FP32, d_inner_per_thread - cb);
        HVX_Vector sub[32];
        क्रम (uint32_t r = 0; r < cb_n; ++r) {
            स्थिर भग्न * ch_ptr = raw + (cb + r) * d_conv;
            sub[r] = Q6_V_vand_VV(*(स्थिर HVX_UVector *) ch_ptr, mask);
        }
        क्रम (uint32_t r = cb_n; r < 32; ++r) {
            sub[r] = hvx_vec_splat_f32(0.0f);
        }

        hvx_transpose_32x32_f32(sub);

        क्रम (uint32_t j = 0; j < d_conv; ++j) {
            *(HVX_Vector *)(T + j * d_inner_stride + cb) = sub[j];
        }
    }
}

अटल अंतरभूत व्योम hvx_ssm_conv_unpack_to_T(स्थिर भग्न * raw, भग्न * T, uint32_t d_inner_per_thread, uint32_t d_inner_stride, uint32_t d_conv) {
    अगर (d_conv == 4 && (d_inner_per_thread % VLEN_FP32 == 0)) {
        hvx_ssm_conv_unpack_to_T_4(raw, T, d_inner_per_thread, d_inner_stride);
    } अन्यथा {
        hvx_ssm_conv_unpack_to_T_gen(raw, T, d_inner_per_thread, d_inner_stride, d_conv);
    }
}

// Decode dot product specialization for d_conv == 4: multiply in the raw channel-major layout,
// then deinterleave the products so each vector holds one tap of 32 channels, and sum.
// Keeps both operands in DMA layout - no transpose, no scratch.
अटल अंतरभूत व्योम hvx_ssm_conv_decode_4(स्थिर भग्न * x, स्थिर भग्न * w, भग्न * out, uint32_t n_ch) {
    क्रम (uint32_t cb = 0; cb < n_ch; cb += VLEN_FP32) {
        स्थिर भग्न * xp = x + cb * 4;
        स्थिर भग्न * wp = w + cb * 4;

        HVX_Vector p0 = Q6_Vqf32_vmpy_VsfVsf(*(स्थिर HVX_Vector *)(xp +  0), *(स्थिर HVX_Vector *)(wp +  0));
        HVX_Vector p1 = Q6_Vqf32_vmpy_VsfVsf(*(स्थिर HVX_Vector *)(xp + 32), *(स्थिर HVX_Vector *)(wp + 32));
        HVX_Vector p2 = Q6_Vqf32_vmpy_VsfVsf(*(स्थिर HVX_Vector *)(xp + 64), *(स्थिर HVX_Vector *)(wp + 64));
        HVX_Vector p3 = Q6_Vqf32_vmpy_VsfVsf(*(स्थिर HVX_Vector *)(xp + 96), *(स्थिर HVX_Vector *)(wp + 96));

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
अटल अंतरभूत व्योम hvx_ssm_conv_transpose_block(स्थिर भग्न * raw_block,
                                                भग्न *       T,
                                                uint32_t      ncs,
                                                uint32_t      cb_n,
                                                HVX_Vector    vv) {
    स्थिर माप_प्रकार   base = (माप_प्रकार) raw_block;
    स्थिर uint32_t mu   = cb_n * ncs * माप(भग्न) - 1;

    क्रम (uint32_t t = 0; t < ncs; ++t) {
        Q6_vgather_ARMVw((HVX_Vector *) (T + (माप_प्रकार) t * VLEN_FP32), base + t * माप(भग्न), mu, vv);
    }
}

// Single-token decode worker (n_t == 1)
अटल व्योम ssm_conv_thread_f32_decode(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    काष्ठा htp_ssm_conv_context *             scctx   = (काष्ठा htp_ssm_conv_context *) तथ्य;
    काष्ठा htp_ops_context *                  octx    = scctx->octx;
    स्थिर काष्ठा htp_ssm_conv_kernel_params * kparams = scctx->kparams;

    स्थिर काष्ठा htp_tensor * restrict src0 = octx->src[0];
    स्थिर काष्ठा htp_tensor * restrict src1 = octx->src[1];
    स्थिर काष्ठा htp_tensor * restrict dst  = octx->dst;

    dma_queue * dma_q = octx->ctx->dma[ith];

    स्थिर uint32_t d_conv  = kparams->d_conv;
    स्थिर uint32_t d_inner = kparams->d_inner;
    स्थिर uint32_t n_s     = kparams->n_s;

    स्थिर uint32_t dr  = scctx->nrows_per_thread;
    स्थिर uint32_t ir0 = scctx->row_start + dr * ith;
    स्थिर uint32_t ir1 = MIN(ir0 + dr, scctx->row_start + scctx->nrows);

    अगर (ir0 >= ir1) {
        वापस;
    }

    स्थिर uint32_t d_inner_per_thread = ir1 - ir0;
    स्थिर uint32_t d_inner_stride     = hex_round_up(d_inner_per_thread, VLEN_FP32);
    स्थिर uint32_t d_inner_tile       = scctx->d_inner_tile;

    स्थिर माप_प्रकार src0_stride_inner_bytes = src0->nb[1];
    स्थिर माप_प्रकार src0_stride_seq_bytes   = src0->nb[2];
    स्थिर माप_प्रकार dst_stride_seq_bytes    = dst->nb[2];

    uint8_t * src1_spad_base = octx->src1_spad.तथ्य + ith * octx->src1_spad.आकार_per_thread;
    uint8_t * src0_spad_base = octx->src0_spad.तथ्य + ith * octx->src0_spad.आकार_per_thread;
    uint8_t * dst_spad_base  = octx->dst_spad.तथ्य  + ith * octx->dst_spad.आकार_per_thread;

    स्थिर माप_प्रकार weight_bytes    = (माप_प्रकार) d_inner_per_thread * d_conv * माप(भग्न);
    स्थिर माप_प्रकार weight_raw_आकार = hex_round_up(weight_bytes, 128);

    भग्न * src1_raw = (भग्न *) src1_spad_base;
    भग्न * src1_T   = (भग्न *) (src1_spad_base + weight_raw_आकार);

    स्थिर माप_प्रकार src0_tile_raw_bytes = hex_round_up(d_inner_tile * d_conv * माप(भग्न), 128);
    स्थिर माप_प्रकार dst_tile_bytes      = hex_round_up(d_inner_tile * माप(भग्न), 128);

    भग्न * src0_tile_raw[2] = { (भग्न *) src0_spad_base, (भग्न *) (src0_spad_base + src0_tile_raw_bytes) };
    भग्न * src0_T           = (भग्न *) (src0_spad_base + 2 * src0_tile_raw_bytes);

    भग्न * dst_tile[2] = { (भग्न *) dst_spad_base, (भग्न *) (dst_spad_base + dst_tile_bytes) };

    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];

    // raw_dot keeps both operands in the DMA layout, so src1 needs no prep pass
    स्थिर bool raw_dot = (d_conv == 4) && (d_inner_per_thread % VLEN_FP32 == 0);

    स्थिर uint32_t n_tiles   = (d_inner_per_thread + d_inner_tile - 1) / d_inner_tile;
    स्थिर uint32_t n_chunks  = n_s * n_tiles;
    स्थिर माप_प्रकार   row_bytes = d_conv * माप(भग्न);

    uint32_t s_fetch        = 0;
    uint32_t tile_off_fetch = 0;

    #घोषणा SSM_CONV_DECODE_धकेलो_FETCH(c)                                                         \
        करो {                                                                                      \
            स्थिर uint32_t   cur_tile_n = MIN(d_inner_tile, d_inner_per_thread - tile_off_fetch); \
            स्थिर dma_addr_t fetch_ddr  = src0->तथ्य + s_fetch * src0_stride_seq_bytes +          \
                                          (ir0 + tile_off_fetch) * src0_stride_inner_bytes;       \
            dma_queue_धकेलो(dma_q,                                                                 \
                           dma_make_तथ्य((uint8_t *) src0_tile_raw[(c) & 1], fetch_ddr),          \
                           row_bytes, src0_stride_inner_bytes, row_bytes, cur_tile_n);            \
            tile_off_fetch += d_inner_tile;                                                       \
            अगर (tile_off_fetch >= d_inner_per_thread) {                                           \
                tile_off_fetch = 0;                                                               \
                s_fetch++;                                                                        \
            }                                                                                     \
        } जबतक (0)

    // Queue weights and initial input tiles together so DDR reads overlap
    स्थिर dma_addr_t src1_ddr = src1->तथ्य + ir0 * d_conv * माप(भग्न);
    dma_queue_धकेलो(dma_q, dma_make_तथ्य((uint8_t *) src1_raw, src1_ddr), weight_bytes, weight_bytes, weight_bytes, 1);

    SSM_CONV_DECODE_धकेलो_FETCH(0);
    अगर (n_chunks > 1) {
        SSM_CONV_DECODE_धकेलो_FETCH(1);
    }

    dma_queue_निकालो(dma_q);  // weights

    अगर (!raw_dot) {
        // Unpack/transpose src1_raw into src1_T {d_conv, d_inner_stride}
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);
        hvx_ssm_conv_unpack_to_T(src1_raw, src1_T, d_inner_per_thread, d_inner_stride, d_conv);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);
    }

    uint32_t i3       = 0;
    uint32_t tile_off = 0;

    क्रम (uint32_t c = 0; c < n_chunks; ++c) {
        स्थिर uint32_t tile_n = MIN(d_inner_tile, d_inner_per_thread - tile_off);

        अगर (c >= 2) {
            dma_queue_निकालो(dma_q);  // writeback of chunk c-2, frees dst_tile[c & 1]
        }
        dma_queue_निकालो(dma_q);      // fetch chunk c

        भग्न * restrict out = dst_tile[c & 1];

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i3);
        अगर (raw_dot) {
            स्थिर भग्न * xp = src0_tile_raw[c & 1];
            स्थिर भग्न * wp = src1_raw + tile_off * 4;
            hvx_ssm_conv_decode_4(xp, wp, out, tile_n);
        } अन्यथा {
            स्थिर uint32_t tile_stride = hex_round_up(tile_n, VLEN_FP32);
            hvx_ssm_conv_unpack_to_T(src0_tile_raw[c & 1], src0_T, tile_n, tile_stride, d_conv);

            क्रम (uint32_t cb = 0; cb < tile_n; cb += VLEN_FP32) {
                स्थिर uint32_t cb_n = MIN(VLEN_FP32, tile_n - cb);
                HVX_Vector acc = hvx_vec_splat_f32(0.0f);
                क्रम (uint32_t j = 0; j < d_conv; ++j) {
                    HVX_Vector x = *(स्थिर HVX_Vector *)(src0_T + j * tile_stride + cb);
                    HVX_Vector w = *(स्थिर HVX_Vector *)(src1_T + j * d_inner_stride + tile_off + cb);
                    acc          = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(x, w));
                }
                HVX_Vector y = Q6_Vsf_equals_Vqf32(acc);
                अगर (cb_n == VLEN_FP32) {
                    *(HVX_Vector *)(out + cb) = y;
                } अन्यथा {
                    hvx_vec_store_u(out + cb, cb_n * माप(भग्न), y);
                }
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i3);

        स्थिर dma_addr_t dst_ddr        = dst->तथ्य + i3 * dst_stride_seq_bytes + (ir0 + tile_off) * माप(भग्न);
        स्थिर माप_प्रकार     tile_out_bytes = (माप_प्रकार) tile_n * माप(भग्न);
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(dst_ddr, (uint8_t *) out),
                       tile_out_bytes, tile_out_bytes, tile_out_bytes, 1);

        अगर (c + 2 < n_chunks) {
            SSM_CONV_DECODE_धकेलो_FETCH(c + 2);
        }

        tile_off += d_inner_tile;
        अगर (tile_off >= d_inner_per_thread) {
            tile_off = 0;
            i3++;
        }
    }

    क्रम (uint32_t k = MIN(n_chunks, 2); k > 0; --k) {
        dma_queue_निकालो(dma_q);  // drain the last writebacks
    }

    #अघोषित SSM_CONV_DECODE_धकेलो_FETCH

    FARF(ऊपरी_2, "ssm-conv-f32-decode %d/%d: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0], dst->ne[1],
         dst->ne[2], dst->ne[3]);
}

// Multi-token prefill worker (n_t > 1)
अटल व्योम ssm_conv_thread_f32_prefill(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    काष्ठा htp_ssm_conv_context *             scctx   = (काष्ठा htp_ssm_conv_context *) तथ्य;
    काष्ठा htp_ops_context *                  octx    = scctx->octx;
    स्थिर काष्ठा htp_ssm_conv_kernel_params * kparams = scctx->kparams;

    स्थिर काष्ठा htp_tensor * restrict src0 = octx->src[0];
    स्थिर काष्ठा htp_tensor * restrict src1 = octx->src[1];
    स्थिर काष्ठा htp_tensor * restrict dst  = octx->dst;

    dma_queue * dma_q = octx->ctx->dma[ith];

    स्थिर uint32_t d_conv  = kparams->d_conv;
    स्थिर uint32_t d_inner = kparams->d_inner;
    स्थिर uint32_t n_t     = kparams->n_t;
    स्थिर uint32_t n_s     = kparams->n_s;
    स्थिर uint32_t ncs     = src0->ne[0];

    स्थिर uint32_t dr  = scctx->nrows_per_thread;
    स्थिर uint32_t ir0 = scctx->row_start + dr * ith;
    स्थिर uint32_t ir1 = MIN(ir0 + dr, scctx->row_start + scctx->nrows);

    अगर (ir0 >= ir1) {
        वापस;
    }

    स्थिर uint32_t d_inner_per_thread = ir1 - ir0;
    स्थिर uint32_t d_inner_stride     = hex_round_up(d_inner_per_thread, VLEN_FP32);
    स्थिर uint32_t d_inner_tile       = scctx->d_inner_tile;

    स्थिर माप_प्रकार src0_stride_inner_bytes = src0->nb[1];
    स्थिर माप_प्रकार src0_stride_seq_bytes   = src0->nb[2];
    स्थिर माप_प्रकार dst_stride_token_bytes  = dst->nb[1];
    स्थिर माप_प्रकार dst_stride_seq_bytes    = dst->nb[2];

    uint8_t * src1_spad_base = octx->src1_spad.तथ्य + ith * octx->src1_spad.आकार_per_thread;
    uint8_t * src0_spad_base = octx->src0_spad.तथ्य + ith * octx->src0_spad.आकार_per_thread;
    uint8_t * dst_spad_base  = octx->dst_spad.तथ्य  + ith * octx->dst_spad.आकार_per_thread;

    स्थिर माप_प्रकार weight_bytes    = (माप_प्रकार) d_inner_per_thread * d_conv * माप(भग्न);
    स्थिर माप_प्रकार weight_raw_आकार = hex_round_up(weight_bytes, 128);

    भग्न * src1_raw = (भग्न *) src1_spad_base;
    भग्न * src1_T   = (भग्न *) (src1_spad_base + weight_raw_आकार);

    // src0 spad holds two raw tiles (fetch of tile n+1 overlaps compute of tile n) plus
    // the transposed block. dst spad holds two tiles so a writeback can stay in flight.
    स्थिर माप_प्रकार src0_tile_raw_bytes = hex_round_up(d_inner_tile * ncs * माप(भग्न), 128);
    स्थिर माप_प्रकार dst_tile_bytes      = hex_round_up(d_inner_tile * n_t * माप(भग्न), 128);

    भग्न * src0_tile_raw[2] = { (भग्न *) src0_spad_base, (भग्न *) (src0_spad_base + src0_tile_raw_bytes) };
    भग्न * src0_T           = (भग्न *) (src0_spad_base + 2 * src0_tile_raw_bytes);

    भग्न * dst_tile[2] = { (भग्न *) dst_spad_base, (भग्न *) (dst_spad_base + dst_tile_bytes) };

    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];

    स्थिर uint32_t n_tiles   = (d_inner_per_thread + d_inner_tile - 1) / d_inner_tile;
    स्थिर uint32_t n_chunks  = n_s * n_tiles;
    स्थिर माप_प्रकार   row_bytes = ncs * माप(भग्न);

    uint32_t s_fetch        = 0;
    uint32_t tile_off_fetch = 0;

    // Chunk c fetches into src0_tile_raw[c & 1] and writes back from dst_tile[c & 1].
    // Two fetches run ahead, so the queue order is F0 F1 W0 F2 W1 ... and pops follow it.
    #घोषणा SSM_CONV_धकेलो_FETCH(c)                                                                \
        करो {                                                                                      \
            स्थिर uint32_t   cur_tile_n = MIN(d_inner_tile, d_inner_per_thread - tile_off_fetch); \
            स्थिर dma_addr_t fetch_ddr  = src0->तथ्य + s_fetch * src0_stride_seq_bytes +          \
                                          (ir0 + tile_off_fetch) * src0_stride_inner_bytes;       \
            dma_queue_धकेलो(dma_q,                                                                 \
                           dma_make_तथ्य((uint8_t *) src0_tile_raw[(c) & 1], fetch_ddr),          \
                           row_bytes, src0_stride_inner_bytes, row_bytes, cur_tile_n);            \
            tile_off_fetch += d_inner_tile;                                                       \
            अगर (tile_off_fetch >= d_inner_per_thread) {                                           \
                tile_off_fetch = 0;                                                               \
                s_fetch++;                                                                        \
            }                                                                                     \
        } जबतक (0)

    // Queue weights and initial input tiles together so DDR reads overlap
    स्थिर dma_addr_t src1_ddr = src1->तथ्य + ir0 * d_conv * माप(भग्न);
    dma_queue_धकेलो(dma_q, dma_make_तथ्य((uint8_t *) src1_raw, src1_ddr), weight_bytes, weight_bytes, weight_bytes, 1);

    SSM_CONV_धकेलो_FETCH(0);
    अगर (n_chunks > 1) {
        SSM_CONV_धकेलो_FETCH(1);
    }

    dma_queue_निकालो(dma_q);  // weights

    // Unpack/transpose src1_raw into src1_T {d_conv, d_inner_stride}
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);
    hvx_ssm_conv_unpack_to_T(src1_raw, src1_T, d_inner_per_thread, d_inner_stride, d_conv);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);

    स्थिर uint32_t C_TILE = VLEN_FP32;

    // gather offsets: lane r reads channel r of the raw tile block
    uint32_t __attribute__((aligned(VLEN))) gather_off[VLEN_FP32];
    क्रम (uint32_t r = 0; r < VLEN_FP32; ++r) {
        gather_off[r] = r * ncs * माप(भग्न);
    }
    स्थिर HVX_Vector vv = *(स्थिर HVX_Vector *) gather_off;

    uint32_t i3       = 0;
    uint32_t tile_off = 0;

    क्रम (uint32_t c = 0; c < n_chunks; ++c) {
        स्थिर uint32_t tile_n = MIN(d_inner_tile, d_inner_per_thread - tile_off);

        स्थिर भग्न * restrict raw = src0_tile_raw[c & 1];
        भग्न * restrict       out = dst_tile[c & 1];

        अगर (c >= 2) {
            dma_queue_निकालो(dma_q);  // writeback of chunk c-2, frees dst_tile[c & 1]
        }
        dma_queue_निकालो(dma_q);      // fetch chunk c

        // Channel block outer, token inner: the taps and the sliding window stay in
        // registers, so a new output row costs one src0_T load.
        स्थिर uint32_t dst_tile_stride = hex_round_up(tile_n, C_TILE);

        क्रम (uint32_t cb = 0; cb < tile_n; cb += C_TILE) {
            स्थिर uint32_t cb_n = MIN(C_TILE, tile_n - cb);

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) tile_off);
            hvx_ssm_conv_transpose_block(raw + (माप_प्रकार) cb * ncs, src0_T, ncs, cb_n, vv);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) tile_off);

            स्थिर भग्न * restrict wp = src1_T + tile_off + cb;
            भग्न * restrict       op = out + cb;

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) tile_off);
            अगर (d_conv == 4) {
                स्थिर HVX_Vector w0 = *(स्थिर HVX_Vector *) (wp);
                स्थिर HVX_Vector w1 = *(स्थिर HVX_Vector *) (wp + d_inner_stride);
                स्थिर HVX_Vector w2 = *(स्थिर HVX_Vector *) (wp + 2 * d_inner_stride);
                स्थिर HVX_Vector w3 = *(स्थिर HVX_Vector *) (wp + 3 * d_inner_stride);

                HVX_Vector x0 = *(स्थिर HVX_Vector *) (src0_T);
                HVX_Vector x1 = *(स्थिर HVX_Vector *) (src0_T + C_TILE);
                HVX_Vector x2 = *(स्थिर HVX_Vector *) (src0_T + 2 * C_TILE);

                क्रम (uint32_t t = 0; t < n_t; ++t) {
                    स्थिर HVX_Vector x3 = *(स्थिर HVX_Vector *) (src0_T + (t + 3) * C_TILE);

                    HVX_Vector a = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(x0, w0),
                                                            Q6_Vqf32_vmpy_VsfVsf(x1, w1));
                    HVX_Vector b = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(x2, w2),
                                                            Q6_Vqf32_vmpy_VsfVsf(x3, w3));

                    *(HVX_Vector *) (op + t * dst_tile_stride) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(a, b));

                    x0 = x1;
                    x1 = x2;
                    x2 = x3;
                }
            } अन्यथा {
                क्रम (uint32_t t = 0; t < n_t; ++t) {
                    HVX_Vector acc = hvx_vec_splat_f32(0.0f);
                    क्रम (uint32_t j = 0; j < d_conv; ++j) {
                        HVX_Vector x = *(स्थिर HVX_Vector *) (src0_T + (t + j) * C_TILE);
                        HVX_Vector w = *(स्थिर HVX_Vector *) (wp + j * d_inner_stride);
                        acc          = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(x, w));
                    }
                    *(HVX_Vector *) (op + t * dst_tile_stride) = Q6_Vsf_equals_Vqf32(acc);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) tile_off);
        }

        // Writeback dst_tile VTCM -> DDR via 2D DMA
        dma_queue_धकेलो(dma_q,
                       dma_make_तथ्य(dst->तथ्य + i3 * dst_stride_seq_bytes + (ir0 + tile_off) * माप(भग्न),
                                     (uint8_t *) out),
                       dst_stride_token_bytes, dst_tile_stride * माप(भग्न), tile_n * माप(भग्न), n_t);

        अगर (c + 2 < n_chunks) {
            SSM_CONV_धकेलो_FETCH(c + 2);
        }

        tile_off += d_inner_tile;
        अगर (tile_off >= d_inner_per_thread) {
            tile_off = 0;
            i3++;
        }
    }

    क्रम (uint32_t k = MIN(n_chunks, 2); k > 0; --k) {
        dma_queue_निकालो(dma_q);  // drain the last writebacks
    }

    #अघोषित SSM_CONV_धकेलो_FETCH

    FARF(ऊपरी_2, "ssm-conv-f32-prefill %d/%d: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0], dst->ne[1],
         dst->ne[2], dst->ne[3]);
}

पूर्णांक op_ssm_conv_f32(काष्ठा htp_ops_context * octx) {
    स्थिर काष्ठा htp_tensor * src0 = octx->src[0];
    स्थिर काष्ठा htp_tensor * src1 = octx->src[1];
    स्थिर काष्ठा htp_tensor * dst  = octx->dst;

    अगर (src0->type != HTP_TYPE_F32 || src1->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        वापस HTP_STATUS_NO_SUPPORT;
    }

    स्थिर काष्ठा htp_ssm_conv_kernel_params * kparams = (स्थिर काष्ठा htp_ssm_conv_kernel_params *) octx->kernel_params;


    अगर (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        वापस HTP_STATUS_INVAL_PARAMS;
    }

    uint32_t row_start = 0;
    uint32_t nrows     = kparams->d_inner;

    अगर (octx->ctx->mdev.गिनती > 1) {
        स्थिर uint32_t elems_per_chunk = VLEN_FP32;
        स्थिर काष्ठा htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            kparams->d_inner,
            htp_tensor_mdev_तथ्य_aligned(dst) ? elems_per_chunk : 0,
            octx->ctx->mdev.idx,
            octx->ctx->mdev.गिनती,
            &octx->ctx->mdev.गिनती_div
        );
        row_start = range.start;
        nrows     = range.गिनती;
    }

    अगर (nrows == 0) {
        वापस HTP_STATUS_OK;
    }

    अगर (kparams->vtcm_आकार > octx->ctx->vtcm_आकार) {
        वापस HTP_STATUS_VTCM_TOO_SMALL;
    }

    स्थिर uint32_t n_threads = octx->n_threads;

    octx->src0_spad.आकार_per_thread = kparams->vtcm_src0_आकार_per_thread;
    octx->src1_spad.आकार_per_thread = kparams->vtcm_src1_आकार_per_thread;
    octx->dst_spad.आकार_per_thread  = kparams->vtcm_dst_आकार_per_thread;

    octx->src0_spad.आकार = kparams->vtcm_src0_आकार;
    octx->src1_spad.आकार = kparams->vtcm_src1_आकार;
    octx->dst_spad.आकार  = kparams->vtcm_dst_आकार;

    octx->src0_spad.तथ्य = octx->ctx->vtcm_base;
    octx->src1_spad.तथ्य = octx->src0_spad.तथ्य + octx->src0_spad.आकार;
    octx->dst_spad.तथ्य  = octx->src1_spad.तथ्य + octx->src1_spad.आकार;
    octx->src0_spad.src  = शून्य;
    octx->src1_spad.src  = शून्य;
    octx->dst_spad.src   = शून्य;

    स्थिर uint32_t raw_rpt            = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);
    स्थिर uint32_t d_inner_per_thread = hex_round_up(raw_rpt, VLEN_FP32);

    काष्ठा htp_ssm_conv_context scctx = {
        .octx             = octx,
        .kparams          = kparams,
        .nrows_per_thread = d_inner_per_thread,
        .d_inner_tile     = kparams->d_inner_tile,
        .row_start        = row_start,
        .nrows            = nrows,
    };

    FARF(ऊपरी_2, "ssm-conv-f32: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : mode %s\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         kparams->n_t == 1 ? "decode" : "prefill");

    अगर (kparams->n_t == 1) {
        work_queue_run(octx->ctx->work_queue, ssm_conv_thread_f32_decode, &scctx, n_threads);
    } अन्यथा {
        work_queue_run(octx->ctx->work_queue, ssm_conv_thread_f32_prefill, &scctx, n_threads);
    }

    वापस HTP_STATUS_OK;
}

पूर्णांक op_ssm_conv(काष्ठा htp_ops_context * octx) {
    स्थिर काष्ठा htp_tensor * dst = octx->dst;

    चयन (dst->type) {
        हाल HTP_TYPE_F32:
            वापस op_ssm_conv_f32(octx);
        शेष:
            वापस HTP_STATUS_NO_SUPPORT;
    }
}
