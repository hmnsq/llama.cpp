#निर्देशः clang diagnostic ignored "-Wunused-variable"
#निर्देशः clang diagnostic ignored "-Wunused-function"
#निर्देशः clang diagnostic ignored "-Wunused-but-set-variable"

#अन्तर्भाव <HAP_farf.h>
#अन्तर्भाव <HAP_mem.h>
#अन्तर्भाव <HAP_ps.h>
#अन्तर्भाव <hexagon_protos.h>
#अन्तर्भाव <hexagon_types.h>
#अन्तर्भाव <math.h>
#अन्तर्भाव <qurt_thread.h>
#अन्तर्भाव <string.h>

#परिभाषय GGML_COMMON_DECL_C
#अन्तर्भाव "ggml-common.h"
#अन्तर्भाव "htp-ctx.h"
#अन्तर्भाव "dma-queue.h"
#अन्तर्भाव "hex-profile.h"
#अन्तर्भाव "htp-ops.h"
#अन्तर्भाव "htp-tensor.h"
#अन्तर्भाव "hvx-utils.h"
#अन्तर्भाव "ssm-conv.h"

संरचना htp_ssm_conv_context {
    संरचना htp_ops_context *                  octx;
    ध्रुव संरचना htp_ssm_conv_kernel_params * kparams;
    uint32_t                                  nrows_प्रत्येकम्_thread;
    uint32_t                                  d_inner_tile;
    uint32_t                                  row_start;
    uint32_t                                  nrows;
};

// In-register 32x32 fp32 transpose using std 5-stage HVX vshuff butterfly.
स्थिर अन्तःस्थ शून्य hvx_transpose_32x32_f32(HVX_सदिशम् m[32]) {
    HVX_सदिशम् tmp[32];

    // Stage 0 (R = -4): pair (2i, 2i+1) for i = 0..15. m -> tmp.
    प्रति (पूर्णाङ्क i = 0; i < 16; ++i) {
        HVX_सदिशम्_Pair p = Q6_W_vshuff_VVR(m[2*i + 1], m[2*i], -4);
        tmp[2*i + 0] = Q6_V_lo_W(p);
        tmp[2*i + 1] = Q6_V_hi_W(p);
    }

    // Stage 1 (R = -8): per block of 4, pair (b+0, b+2) and (b+1, b+3). tmp -> m.
    प्रति (पूर्णाङ्क b = 0; b < 32; b += 4) {
        HVX_सदिशम्_Pair p0 = Q6_W_vshuff_VVR(tmp[b + 2], tmp[b + 0], -8);
        HVX_सदिशम्_Pair p1 = Q6_W_vshuff_VVR(tmp[b + 3], tmp[b + 1], -8);
        m[b + 0] = Q6_V_lo_W(p0); m[b + 1] = Q6_V_hi_W(p0);
        m[b + 2] = Q6_V_lo_W(p1); m[b + 3] = Q6_V_hi_W(p1);
    }

    // Stage 2 (R = -16): per block of 8, pair (b+i, b+i+4) for i = 0..3. m -> tmp.
    प्रति (पूर्णाङ्क b = 0; b < 32; b += 8) {
        प्रति (पूर्णाङ्क i = 0; i < 4; ++i) {
            HVX_सदिशम्_Pair p = Q6_W_vshuff_VVR(m[b + i + 4], m[b + i], -16);
            tmp[b + 2*i + 0] = Q6_V_lo_W(p);
            tmp[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }

    // Stage 3 (R = -32): per block of 16, pair (b+i, b+i+8) for i = 0..7. tmp -> m.
    प्रति (पूर्णाङ्क b = 0; b < 32; b += 16) {
        प्रति (पूर्णाङ्क i = 0; i < 8; ++i) {
            HVX_सदिशम्_Pair p = Q6_W_vshuff_VVR(tmp[b + i + 8], tmp[b + i], -32);
            m[b + 2*i + 0] = Q6_V_lo_W(p);
            m[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }

    // Stage 4 (R = -64): pair (i, i+16) for i = 0..15. m -> tmp -> m.
    प्रति (पूर्णाङ्क i = 0; i < 16; ++i) {
        HVX_सदिशम्_Pair p = Q6_W_vshuff_VVR(m[i + 16], m[i], -64);
        tmp[2 * i + 0]   = Q6_V_lo_W(p);
        tmp[2 * i + 1]   = Q6_V_hi_W(p);
    }

    प्रति (पूर्णाङ्क i = 0; i < 32; ++i) {
        m[i] = tmp[i];
    }
}

// HVX deinterleave for d_conv == 4: channel-major raw VTCM -> tap-major T VTCM
स्थिर अन्तःस्थ शून्य hvx_ssm_conv_unpack_गन्तव्यम्_T_4(ध्रुव भिन्न * raw, भिन्न * T, uint32_t d_inner_प्रत्येकम्_thread, uint32_t d_inner_stride) {
    प्रति (uint32_t cb = 0; cb < d_inner_प्रत्येकम्_thread; cb += VLEN_FP32) {
        HVX_सदिशम् v0 = *(ध्रुव HVX_सदिशम् *)(raw + (cb + 0) * 4);
        HVX_सदिशम् v1 = *(ध्रुव HVX_सदिशम् *)(raw + (cb + 8) * 4);
        HVX_सदिशम् v2 = *(ध्रुव HVX_सदिशम् *)(raw + (cb + 16) * 4);
        HVX_सदिशम् v3 = *(ध्रुव HVX_सदिशम् *)(raw + (cb + 24) * 4);

        HVX_सदिशम्_Pair p01 = Q6_W_vdeal_VVR(v1, v0, -4);
        HVX_सदिशम्_Pair p23 = Q6_W_vdeal_VVR(v3, v2, -4);

        HVX_सदिशम्_Pair p_w02 = Q6_W_vdeal_VVR(Q6_V_lo_W(p23), Q6_V_lo_W(p01), -4);
        HVX_सदिशम्_Pair p_w13 = Q6_W_vdeal_VVR(Q6_V_hi_W(p23), Q6_V_hi_W(p01), -4);

        *(HVX_सदिशम् *)(T + 0 * d_inner_stride + cb) = Q6_V_lo_W(p_w02);
        *(HVX_सदिशम् *)(T + 1 * d_inner_stride + cb) = Q6_V_lo_W(p_w13);
        *(HVX_सदिशम् *)(T + 2 * d_inner_stride + cb) = Q6_V_hi_W(p_w02);
        *(HVX_सदिशम् *)(T + 3 * d_inner_stride + cb) = Q6_V_hi_W(p_w13);
    }
}

// HVX transpose for general d_conv <= 32: channel-major raw VTCM -> tap-major T VTCM
स्थिर अन्तःस्थ शून्य hvx_ssm_conv_unpack_गन्तव्यम्_T_gen(ध्रुव भिन्न * raw, भिन्न * T, uint32_t d_inner_प्रत्येकम्_thread, uint32_t d_inner_stride, uint32_t d_conv) {
    uint32_t __attribute__((aligned(VLEN))) आवरणम्_buf[VLEN_FP32] = { 0 };
    प्रति (uint32_t j = 0; j < d_conv; ++j) {
        आवरणम्_buf[j] = 0xFFFFFFFF;
    }
    ध्रुव HVX_सदिशम् आवरणम् = *(ध्रुव HVX_सदिशम् *) आवरणम्_buf;

    प्रति (uint32_t cb = 0; cb < d_inner_प्रत्येकम्_thread; cb += VLEN_FP32) {
        ध्रुव uint32_t cb_n = MIN(VLEN_FP32, d_inner_प्रत्येकम्_thread - cb);
        HVX_सदिशम् sub[32];
        प्रति (uint32_t r = 0; r < cb_n; ++r) {
            ध्रुव भिन्न * ch_ptr = raw + (cb + r) * d_conv;
            sub[r] = Q6_V_vand_VV(*(ध्रुव HVX_UVector *) ch_ptr, आवरणम्);
        }
        प्रति (uint32_t r = cb_n; r < 32; ++r) {
            sub[r] = hvx_vec_splat_f32(0.0f);
        }

        hvx_transpose_32x32_f32(sub);

        प्रति (uint32_t j = 0; j < d_conv; ++j) {
            *(HVX_सदिशम् *)(T + j * d_inner_stride + cb) = sub[j];
        }
    }
}

स्थिर अन्तःस्थ शून्य hvx_ssm_conv_unpack_गन्तव्यम्_T(ध्रुव भिन्न * raw, भिन्न * T, uint32_t d_inner_प्रत्येकम्_thread, uint32_t d_inner_stride, uint32_t d_conv) {
    यदि (d_conv == 4 && (d_inner_प्रत्येकम्_thread % VLEN_FP32 == 0)) {
        hvx_ssm_conv_unpack_गन्तव्यम्_T_4(raw, T, d_inner_प्रत्येकम्_thread, d_inner_stride);
    } अन्यथा {
        hvx_ssm_conv_unpack_गन्तव्यम्_T_gen(raw, T, d_inner_प्रत्येकम्_thread, d_inner_stride, d_conv);
    }
}

// Decode dot product specialization for d_conv == 4: multiply in the raw channel-major layout,
// then deinterleave the products so each vector holds one tap of 32 channels, and sum.
// Keeps both operands in DMA layout - no transpose, no scratch.
स्थिर अन्तःस्थ शून्य hvx_ssm_conv_decode_4(ध्रुव भिन्न * x, ध्रुव भिन्न * w, भिन्न * बहिर्गमनम्, uint32_t n_ch) {
    प्रति (uint32_t cb = 0; cb < n_ch; cb += VLEN_FP32) {
        ध्रुव भिन्न * xp = x + cb * 4;
        ध्रुव भिन्न * wp = w + cb * 4;

        HVX_सदिशम् p0 = Q6_Vqf32_vmpy_VsfVsf(*(ध्रुव HVX_सदिशम् *)(xp +  0), *(ध्रुव HVX_सदिशम् *)(wp +  0));
        HVX_सदिशम् p1 = Q6_Vqf32_vmpy_VsfVsf(*(ध्रुव HVX_सदिशम् *)(xp + 32), *(ध्रुव HVX_सदिशम् *)(wp + 32));
        HVX_सदिशम् p2 = Q6_Vqf32_vmpy_VsfVsf(*(ध्रुव HVX_सदिशम् *)(xp + 64), *(ध्रुव HVX_सदिशम् *)(wp + 64));
        HVX_सदिशम् p3 = Q6_Vqf32_vmpy_VsfVsf(*(ध्रुव HVX_सदिशम् *)(xp + 96), *(ध्रुव HVX_सदिशम् *)(wp + 96));

        HVX_सदिशम्_Pair p01 = Q6_W_vdeal_VVR(p1, p0, -4);
        HVX_सदिशम्_Pair p23 = Q6_W_vdeal_VVR(p3, p2, -4);

        HVX_सदिशम्_Pair q02 = Q6_W_vdeal_VVR(Q6_V_lo_W(p23), Q6_V_lo_W(p01), -4);
        HVX_सदिशम्_Pair q13 = Q6_W_vdeal_VVR(Q6_V_hi_W(p23), Q6_V_hi_W(p01), -4);

        HVX_सदिशम् a = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(q02), Q6_V_lo_W(q13));
        HVX_सदिशम् b = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_hi_W(q02), Q6_V_hi_W(q13));

        *(HVX_सदिशम् *)(बहिर्गमनम् + cb) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(a, b));
    }
}

// Transpose src0 for prefill: one 32-channel block {32, ncs} (VTCM) -> T {ncs, 32} (VTCM).
// One VTCM gather per output row: lane r picks channel cb+r, the region bound drops
// the lanes past the channel tail.
स्थिर अन्तःस्थ शून्य hvx_ssm_conv_transpose_खण्डम्(ध्रुव भिन्न * raw_खण्डम्,
                                                भिन्न *       T,
                                                uint32_t      ncs,
                                                uint32_t      cb_n,
                                                HVX_सदिशम्    vv) {
    ध्रुव size_t   आधारः = (size_t) raw_खण्डम्;
    ध्रुव uint32_t mu   = cb_n * ncs * परिमाण(भिन्न) - 1;

    प्रति (uint32_t t = 0; t < ncs; ++t) {
        Q6_vgather_ARMVw((HVX_सदिशम् *) (T + (size_t) t * VLEN_FP32), आधारः + t * परिमाण(भिन्न), mu, vv);
    }
}

// Single-token decode worker (n_t == 1)
स्थिर शून्य ssm_conv_thread_f32_decode(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    संरचना htp_ssm_conv_context *             scctx   = (संरचना htp_ssm_conv_context *) दत्त;
    संरचना htp_ops_context *                  octx    = scctx->octx;
    ध्रुव संरचना htp_ssm_conv_kernel_params * kparams = scctx->kparams;

    ध्रुव संरचना htp_tensor * सीमित src0 = octx->src[0];
    ध्रुव संरचना htp_tensor * सीमित src1 = octx->src[1];
    ध्रुव संरचना htp_tensor * सीमित dst  = octx->dst;

    dma_queue * dma_q = octx->ctx->dma[ith];

    ध्रुव uint32_t d_conv  = kparams->d_conv;
    ध्रुव uint32_t d_inner = kparams->d_inner;
    ध्रुव uint32_t n_s     = kparams->n_s;

    ध्रुव uint32_t dr  = scctx->nrows_प्रत्येकम्_thread;
    ध्रुव uint32_t ir0 = scctx->row_start + dr * ith;
    ध्रुव uint32_t ir1 = MIN(ir0 + dr, scctx->row_start + scctx->nrows);

    यदि (ir0 >= ir1) {
        प्रत्यागम;
    }

    ध्रुव uint32_t d_inner_प्रत्येकम्_thread = ir1 - ir0;
    ध्रुव uint32_t d_inner_stride     = hex_आवर्तनम्_ऊर्ध्वम्(d_inner_प्रत्येकम्_thread, VLEN_FP32);
    ध्रुव uint32_t d_inner_tile       = scctx->d_inner_tile;

    ध्रुव size_t src0_stride_inner_अष्टकानि = src0->nb[1];
    ध्रुव size_t src0_stride_seq_अष्टकानि   = src0->nb[2];
    ध्रुव size_t dst_stride_seq_अष्टकानि    = dst->nb[2];

    uint8_t * src1_spad_आधारः = octx->src1_spad.दत्त + ith * octx->src1_spad.आकार_प्रत्येकम्_thread;
    uint8_t * src0_spad_आधारः = octx->src0_spad.दत्त + ith * octx->src0_spad.आकार_प्रत्येकम्_thread;
    uint8_t * dst_spad_आधारः  = octx->dst_spad.दत्त  + ith * octx->dst_spad.आकार_प्रत्येकम्_thread;

    ध्रुव size_t weight_अष्टकानि    = (size_t) d_inner_प्रत्येकम्_thread * d_conv * परिमाण(भिन्न);
    ध्रुव size_t weight_raw_आकार = hex_आवर्तनम्_ऊर्ध्वम्(weight_अष्टकानि, 128);

    भिन्न * src1_raw = (भिन्न *) src1_spad_आधारः;
    भिन्न * src1_T   = (भिन्न *) (src1_spad_आधारः + weight_raw_आकार);

    ध्रुव size_t src0_tile_raw_अष्टकानि = hex_आवर्तनम्_ऊर्ध्वम्(d_inner_tile * d_conv * परिमाण(भिन्न), 128);
    ध्रुव size_t dst_tile_अष्टकानि      = hex_आवर्तनम्_ऊर्ध्वम्(d_inner_tile * परिमाण(भिन्न), 128);

    भिन्न * src0_tile_raw[2] = { (भिन्न *) src0_spad_आधारः, (भिन्न *) (src0_spad_आधारः + src0_tile_raw_अष्टकानि) };
    भिन्न * src0_T           = (भिन्न *) (src0_spad_आधारः + 2 * src0_tile_raw_अष्टकानि);

    भिन्न * dst_tile[2] = { (भिन्न *) dst_spad_आधारः, (भिन्न *) (dst_spad_आधारः + dst_tile_अष्टकानि) };

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];

    // raw_dot keeps both operands in the DMA layout, so src1 needs no prep pass
    ध्रुव bool raw_dot = (d_conv == 4) && (d_inner_प्रत्येकम्_thread % VLEN_FP32 == 0);

    ध्रुव uint32_t n_tiles   = (d_inner_प्रत्येकम्_thread + d_inner_tile - 1) / d_inner_tile;
    ध्रुव uint32_t n_chunks  = n_s * n_tiles;
    ध्रुव size_t   row_अष्टकानि = d_conv * परिमाण(भिन्न);

    uint32_t s_fetch        = 0;
    uint32_t tile_off_fetch = 0;

    #परिभाषय SSM_CONV_DECODE_प्रक्षिप_FETCH(c)                                                         \
        कुरु {                                                                                      \
            ध्रुव uint32_t   cur_tile_n = MIN(d_inner_tile, d_inner_प्रत्येकम्_thread - tile_off_fetch); \
            ध्रुव dma_addr_t fetch_ddr  = src0->दत्त + s_fetch * src0_stride_seq_अष्टकानि +          \
                                          (ir0 + tile_off_fetch) * src0_stride_inner_अष्टकानि;       \
            dma_queue_प्रक्षिप(dma_q,                                                                 \
                           dma_make_दत्त((uint8_t *) src0_tile_raw[(c) & 1], fetch_ddr),          \
                           row_अष्टकानि, src0_stride_inner_अष्टकानि, row_अष्टकानि, cur_tile_n);            \
            tile_off_fetch += d_inner_tile;                                                       \
            यदि (tile_off_fetch >= d_inner_प्रत्येकम्_thread) {                                           \
                tile_off_fetch = 0;                                                               \
                s_fetch++;                                                                        \
            }                                                                                     \
        } यावत् (0)

    // Queue weights and initial input tiles together so DDR reads overlap
    ध्रुव dma_addr_t src1_ddr = src1->दत्त + ir0 * d_conv * परिमाण(भिन्न);
    dma_queue_प्रक्षिप(dma_q, dma_make_दत्त((uint8_t *) src1_raw, src1_ddr), weight_अष्टकानि, weight_अष्टकानि, weight_अष्टकानि, 1);

    SSM_CONV_DECODE_प्रक्षिप_FETCH(0);
    यदि (n_chunks > 1) {
        SSM_CONV_DECODE_प्रक्षिप_FETCH(1);
    }

    dma_queue_निष्कासय(dma_q);  // weights

    यदि (!raw_dot) {
        // Unpack/transpose src1_raw into src1_T {d_conv, d_inner_stride}
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);
        hvx_ssm_conv_unpack_गन्तव्यम्_T(src1_raw, src1_T, d_inner_प्रत्येकम्_thread, d_inner_stride, d_conv);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);
    }

    uint32_t i3       = 0;
    uint32_t tile_off = 0;

    प्रति (uint32_t c = 0; c < n_chunks; ++c) {
        ध्रुव uint32_t tile_n = MIN(d_inner_tile, d_inner_प्रत्येकम्_thread - tile_off);

        यदि (c >= 2) {
            dma_queue_निष्कासय(dma_q);  // writeback of chunk c-2, frees dst_tile[c & 1]
        }
        dma_queue_निष्कासय(dma_q);      // fetch chunk c

        भिन्न * सीमित बहिर्गमनम् = dst_tile[c & 1];

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i3);
        यदि (raw_dot) {
            ध्रुव भिन्न * xp = src0_tile_raw[c & 1];
            ध्रुव भिन्न * wp = src1_raw + tile_off * 4;
            hvx_ssm_conv_decode_4(xp, wp, बहिर्गमनम्, tile_n);
        } अन्यथा {
            ध्रुव uint32_t tile_stride = hex_आवर्तनम्_ऊर्ध्वम्(tile_n, VLEN_FP32);
            hvx_ssm_conv_unpack_गन्तव्यम्_T(src0_tile_raw[c & 1], src0_T, tile_n, tile_stride, d_conv);

            प्रति (uint32_t cb = 0; cb < tile_n; cb += VLEN_FP32) {
                ध्रुव uint32_t cb_n = MIN(VLEN_FP32, tile_n - cb);
                HVX_सदिशम् acc = hvx_vec_splat_f32(0.0f);
                प्रति (uint32_t j = 0; j < d_conv; ++j) {
                    HVX_सदिशम् x = *(ध्रुव HVX_सदिशम् *)(src0_T + j * tile_stride + cb);
                    HVX_सदिशम् w = *(ध्रुव HVX_सदिशम् *)(src1_T + j * d_inner_stride + tile_off + cb);
                    acc          = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(x, w));
                }
                HVX_सदिशम् y = Q6_Vsf_equals_Vqf32(acc);
                यदि (cb_n == VLEN_FP32) {
                    *(HVX_सदिशम् *)(बहिर्गमनम् + cb) = y;
                } अन्यथा {
                    hvx_vec_store_u(बहिर्गमनम् + cb, cb_n * परिमाण(भिन्न), y);
                }
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i3);

        ध्रुव dma_addr_t dst_ddr        = dst->दत्त + i3 * dst_stride_seq_अष्टकानि + (ir0 + tile_off) * परिमाण(भिन्न);
        ध्रुव size_t     tile_बहिर्गमनम्_अष्टकानि = (size_t) tile_n * परिमाण(भिन्न);
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(dst_ddr, (uint8_t *) बहिर्गमनम्),
                       tile_बहिर्गमनम्_अष्टकानि, tile_बहिर्गमनम्_अष्टकानि, tile_बहिर्गमनम्_अष्टकानि, 1);

        यदि (c + 2 < n_chunks) {
            SSM_CONV_DECODE_प्रक्षिप_FETCH(c + 2);
        }

        tile_off += d_inner_tile;
        यदि (tile_off >= d_inner_प्रत्येकम्_thread) {
            tile_off = 0;
            i3++;
        }
    }

    प्रति (uint32_t k = MIN(n_chunks, 2); k > 0; --k) {
        dma_queue_निष्कासय(dma_q);  // drain the last writebacks
    }

    #अपरिभाषय SSM_CONV_DECODE_प्रक्षिप_FETCH

    FARF(उच्चः_2, "ssm-conv-f32-decode %d/%d: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0], dst->ne[1],
         dst->ne[2], dst->ne[3]);
}

// Multi-token prefill worker (n_t > 1)
स्थिर शून्य ssm_conv_thread_f32_prefill(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    संरचना htp_ssm_conv_context *             scctx   = (संरचना htp_ssm_conv_context *) दत्त;
    संरचना htp_ops_context *                  octx    = scctx->octx;
    ध्रुव संरचना htp_ssm_conv_kernel_params * kparams = scctx->kparams;

    ध्रुव संरचना htp_tensor * सीमित src0 = octx->src[0];
    ध्रुव संरचना htp_tensor * सीमित src1 = octx->src[1];
    ध्रुव संरचना htp_tensor * सीमित dst  = octx->dst;

    dma_queue * dma_q = octx->ctx->dma[ith];

    ध्रुव uint32_t d_conv  = kparams->d_conv;
    ध्रुव uint32_t d_inner = kparams->d_inner;
    ध्रुव uint32_t n_t     = kparams->n_t;
    ध्रुव uint32_t n_s     = kparams->n_s;
    ध्रुव uint32_t ncs     = src0->ne[0];

    ध्रुव uint32_t dr  = scctx->nrows_प्रत्येकम्_thread;
    ध्रुव uint32_t ir0 = scctx->row_start + dr * ith;
    ध्रुव uint32_t ir1 = MIN(ir0 + dr, scctx->row_start + scctx->nrows);

    यदि (ir0 >= ir1) {
        प्रत्यागम;
    }

    ध्रुव uint32_t d_inner_प्रत्येकम्_thread = ir1 - ir0;
    ध्रुव uint32_t d_inner_stride     = hex_आवर्तनम्_ऊर्ध्वम्(d_inner_प्रत्येकम्_thread, VLEN_FP32);
    ध्रुव uint32_t d_inner_tile       = scctx->d_inner_tile;

    ध्रुव size_t src0_stride_inner_अष्टकानि = src0->nb[1];
    ध्रुव size_t src0_stride_seq_अष्टकानि   = src0->nb[2];
    ध्रुव size_t dst_stride_token_अष्टकानि  = dst->nb[1];
    ध्रुव size_t dst_stride_seq_अष्टकानि    = dst->nb[2];

    uint8_t * src1_spad_आधारः = octx->src1_spad.दत्त + ith * octx->src1_spad.आकार_प्रत्येकम्_thread;
    uint8_t * src0_spad_आधारः = octx->src0_spad.दत्त + ith * octx->src0_spad.आकार_प्रत्येकम्_thread;
    uint8_t * dst_spad_आधारः  = octx->dst_spad.दत्त  + ith * octx->dst_spad.आकार_प्रत्येकम्_thread;

    ध्रुव size_t weight_अष्टकानि    = (size_t) d_inner_प्रत्येकम्_thread * d_conv * परिमाण(भिन्न);
    ध्रुव size_t weight_raw_आकार = hex_आवर्तनम्_ऊर्ध्वम्(weight_अष्टकानि, 128);

    भिन्न * src1_raw = (भिन्न *) src1_spad_आधारः;
    भिन्न * src1_T   = (भिन्न *) (src1_spad_आधारः + weight_raw_आकार);

    // src0 spad holds two raw tiles (fetch of tile n+1 overlaps compute of tile n) plus
    // the transposed block. dst spad holds two tiles so a writeback can stay in flight.
    ध्रुव size_t src0_tile_raw_अष्टकानि = hex_आवर्तनम्_ऊर्ध्वम्(d_inner_tile * ncs * परिमाण(भिन्न), 128);
    ध्रुव size_t dst_tile_अष्टकानि      = hex_आवर्तनम्_ऊर्ध्वम्(d_inner_tile * n_t * परिमाण(भिन्न), 128);

    भिन्न * src0_tile_raw[2] = { (भिन्न *) src0_spad_आधारः, (भिन्न *) (src0_spad_आधारः + src0_tile_raw_अष्टकानि) };
    भिन्न * src0_T           = (भिन्न *) (src0_spad_आधारः + 2 * src0_tile_raw_अष्टकानि);

    भिन्न * dst_tile[2] = { (भिन्न *) dst_spad_आधारः, (भिन्न *) (dst_spad_आधारः + dst_tile_अष्टकानि) };

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];

    ध्रुव uint32_t n_tiles   = (d_inner_प्रत्येकम्_thread + d_inner_tile - 1) / d_inner_tile;
    ध्रुव uint32_t n_chunks  = n_s * n_tiles;
    ध्रुव size_t   row_अष्टकानि = ncs * परिमाण(भिन्न);

    uint32_t s_fetch        = 0;
    uint32_t tile_off_fetch = 0;

    // Chunk c fetches into src0_tile_raw[c & 1] and writes back from dst_tile[c & 1].
    // Two fetches run ahead, so the queue order is F0 F1 W0 F2 W1 ... and pops follow it.
    #परिभाषय SSM_CONV_प्रक्षिप_FETCH(c)                                                                \
        कुरु {                                                                                      \
            ध्रुव uint32_t   cur_tile_n = MIN(d_inner_tile, d_inner_प्रत्येकम्_thread - tile_off_fetch); \
            ध्रुव dma_addr_t fetch_ddr  = src0->दत्त + s_fetch * src0_stride_seq_अष्टकानि +          \
                                          (ir0 + tile_off_fetch) * src0_stride_inner_अष्टकानि;       \
            dma_queue_प्रक्षिप(dma_q,                                                                 \
                           dma_make_दत्त((uint8_t *) src0_tile_raw[(c) & 1], fetch_ddr),          \
                           row_अष्टकानि, src0_stride_inner_अष्टकानि, row_अष्टकानि, cur_tile_n);            \
            tile_off_fetch += d_inner_tile;                                                       \
            यदि (tile_off_fetch >= d_inner_प्रत्येकम्_thread) {                                           \
                tile_off_fetch = 0;                                                               \
                s_fetch++;                                                                        \
            }                                                                                     \
        } यावत् (0)

    // Queue weights and initial input tiles together so DDR reads overlap
    ध्रुव dma_addr_t src1_ddr = src1->दत्त + ir0 * d_conv * परिमाण(भिन्न);
    dma_queue_प्रक्षिप(dma_q, dma_make_दत्त((uint8_t *) src1_raw, src1_ddr), weight_अष्टकानि, weight_अष्टकानि, weight_अष्टकानि, 1);

    SSM_CONV_प्रक्षिप_FETCH(0);
    यदि (n_chunks > 1) {
        SSM_CONV_प्रक्षिप_FETCH(1);
    }

    dma_queue_निष्कासय(dma_q);  // weights

    // Unpack/transpose src1_raw into src1_T {d_conv, d_inner_stride}
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);
    hvx_ssm_conv_unpack_गन्तव्यम्_T(src1_raw, src1_T, d_inner_प्रत्येकम्_thread, d_inner_stride, d_conv);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);

    ध्रुव uint32_t C_TILE = VLEN_FP32;

    // gather offsets: lane r reads channel r of the raw tile block
    uint32_t __attribute__((aligned(VLEN))) gather_off[VLEN_FP32];
    प्रति (uint32_t r = 0; r < VLEN_FP32; ++r) {
        gather_off[r] = r * ncs * परिमाण(भिन्न);
    }
    ध्रुव HVX_सदिशम् vv = *(ध्रुव HVX_सदिशम् *) gather_off;

    uint32_t i3       = 0;
    uint32_t tile_off = 0;

    प्रति (uint32_t c = 0; c < n_chunks; ++c) {
        ध्रुव uint32_t tile_n = MIN(d_inner_tile, d_inner_प्रत्येकम्_thread - tile_off);

        ध्रुव भिन्न * सीमित raw = src0_tile_raw[c & 1];
        भिन्न * सीमित       बहिर्गमनम् = dst_tile[c & 1];

        यदि (c >= 2) {
            dma_queue_निष्कासय(dma_q);  // writeback of chunk c-2, frees dst_tile[c & 1]
        }
        dma_queue_निष्कासय(dma_q);      // fetch chunk c

        // Channel block outer, token inner: the taps and the sliding window stay in
        // registers, so a new output row costs one src0_T load.
        ध्रुव uint32_t dst_tile_stride = hex_आवर्तनम्_ऊर्ध्वम्(tile_n, C_TILE);

        प्रति (uint32_t cb = 0; cb < tile_n; cb += C_TILE) {
            ध्रुव uint32_t cb_n = MIN(C_TILE, tile_n - cb);

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) tile_off);
            hvx_ssm_conv_transpose_खण्डम्(raw + (size_t) cb * ncs, src0_T, ncs, cb_n, vv);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) tile_off);

            ध्रुव भिन्न * सीमित wp = src1_T + tile_off + cb;
            भिन्न * सीमित       op = बहिर्गमनम् + cb;

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) tile_off);
            यदि (d_conv == 4) {
                ध्रुव HVX_सदिशम् w0 = *(ध्रुव HVX_सदिशम् *) (wp);
                ध्रुव HVX_सदिशम् w1 = *(ध्रुव HVX_सदिशम् *) (wp + d_inner_stride);
                ध्रुव HVX_सदिशम् w2 = *(ध्रुव HVX_सदिशम् *) (wp + 2 * d_inner_stride);
                ध्रुव HVX_सदिशम् w3 = *(ध्रुव HVX_सदिशम् *) (wp + 3 * d_inner_stride);

                HVX_सदिशम् x0 = *(ध्रुव HVX_सदिशम् *) (src0_T);
                HVX_सदिशम् x1 = *(ध्रुव HVX_सदिशम् *) (src0_T + C_TILE);
                HVX_सदिशम् x2 = *(ध्रुव HVX_सदिशम् *) (src0_T + 2 * C_TILE);

                प्रति (uint32_t t = 0; t < n_t; ++t) {
                    ध्रुव HVX_सदिशम् x3 = *(ध्रुव HVX_सदिशम् *) (src0_T + (t + 3) * C_TILE);

                    HVX_सदिशम् a = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(x0, w0),
                                                            Q6_Vqf32_vmpy_VsfVsf(x1, w1));
                    HVX_सदिशम् b = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(x2, w2),
                                                            Q6_Vqf32_vmpy_VsfVsf(x3, w3));

                    *(HVX_सदिशम् *) (op + t * dst_tile_stride) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(a, b));

                    x0 = x1;
                    x1 = x2;
                    x2 = x3;
                }
            } अन्यथा {
                प्रति (uint32_t t = 0; t < n_t; ++t) {
                    HVX_सदिशम् acc = hvx_vec_splat_f32(0.0f);
                    प्रति (uint32_t j = 0; j < d_conv; ++j) {
                        HVX_सदिशम् x = *(ध्रुव HVX_सदिशम् *) (src0_T + (t + j) * C_TILE);
                        HVX_सदिशम् w = *(ध्रुव HVX_सदिशम् *) (wp + j * d_inner_stride);
                        acc          = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(x, w));
                    }
                    *(HVX_सदिशम् *) (op + t * dst_tile_stride) = Q6_Vsf_equals_Vqf32(acc);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) tile_off);
        }

        // Writeback dst_tile VTCM -> DDR via 2D DMA
        dma_queue_प्रक्षिप(dma_q,
                       dma_make_दत्त(dst->दत्त + i3 * dst_stride_seq_अष्टकानि + (ir0 + tile_off) * परिमाण(भिन्न),
                                     (uint8_t *) बहिर्गमनम्),
                       dst_stride_token_अष्टकानि, dst_tile_stride * परिमाण(भिन्न), tile_n * परिमाण(भिन्न), n_t);

        यदि (c + 2 < n_chunks) {
            SSM_CONV_प्रक्षिप_FETCH(c + 2);
        }

        tile_off += d_inner_tile;
        यदि (tile_off >= d_inner_प्रत्येकम्_thread) {
            tile_off = 0;
            i3++;
        }
    }

    प्रति (uint32_t k = MIN(n_chunks, 2); k > 0; --k) {
        dma_queue_निष्कासय(dma_q);  // drain the last writebacks
    }

    #अपरिभाषय SSM_CONV_प्रक्षिप_FETCH

    FARF(उच्चः_2, "ssm-conv-f32-prefill %d/%d: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0], dst->ne[1],
         dst->ne[2], dst->ne[3]);
}

पूर्णाङ्क op_ssm_conv_f32(संरचना htp_ops_context * octx) {
    ध्रुव संरचना htp_tensor * src0 = octx->src[0];
    ध्रुव संरचना htp_tensor * src1 = octx->src[1];
    ध्रुव संरचना htp_tensor * dst  = octx->dst;

    यदि (src0->type != HTP_TYPE_F32 || src1->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
    }

    ध्रुव संरचना htp_ssm_conv_kernel_params * kparams = (ध्रुव संरचना htp_ssm_conv_kernel_params *) octx->kernel_params;


    यदि (!htp_ops_context_समुच्चयः_n_तन्तवः(octx, kparams->n_तन्तवः)) {
        प्रत्यागम HTP_स्थितिः_INVAL_PARAMS;
    }

    uint32_t row_start = 0;
    uint32_t nrows     = kparams->d_inner;

    यदि (octx->ctx->mdev.गणनम् > 1) {
        ध्रुव uint32_t elems_प्रत्येकम्_chunk = VLEN_FP32;
        ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            kparams->d_inner,
            htp_tensor_mdev_दत्त_aligned(dst) ? elems_प्रत्येकम्_chunk : 0,
            octx->ctx->mdev.idx,
            octx->ctx->mdev.गणनम्,
            &octx->ctx->mdev.गणनम्_div
        );
        row_start = range.start;
        nrows     = range.गणनम्;
    }

    यदि (nrows == 0) {
        प्रत्यागम HTP_स्थितिः_OK;
    }

    यदि (kparams->vtcm_आकार > octx->ctx->vtcm_आकार) {
        प्रत्यागम HTP_स्थितिः_VTCM_TOO_SMALL;
    }

    ध्रुव uint32_t n_तन्तवः = octx->n_तन्तवः;

    octx->src0_spad.आकार_प्रत्येकम्_thread = kparams->vtcm_src0_आकार_प्रत्येकम्_thread;
    octx->src1_spad.आकार_प्रत्येकम्_thread = kparams->vtcm_src1_आकार_प्रत्येकम्_thread;
    octx->dst_spad.आकार_प्रत्येकम्_thread  = kparams->vtcm_dst_आकार_प्रत्येकम्_thread;

    octx->src0_spad.आकार = kparams->vtcm_src0_आकार;
    octx->src1_spad.आकार = kparams->vtcm_src1_आकार;
    octx->dst_spad.आकार  = kparams->vtcm_dst_आकार;

    octx->src0_spad.दत्त = octx->ctx->vtcm_आधारः;
    octx->src1_spad.दत्त = octx->src0_spad.दत्त + octx->src0_spad.आकार;
    octx->dst_spad.दत्त  = octx->src1_spad.दत्त + octx->src1_spad.आकार;
    octx->src0_spad.src  = रिक्त;
    octx->src1_spad.src  = रिक्त;
    octx->dst_spad.src   = रिक्त;

    ध्रुव uint32_t raw_rpt            = fastdiv(nrows + n_तन्तवः - 1, &octx->n_तन्तवः_div);
    ध्रुव uint32_t d_inner_प्रत्येकम्_thread = hex_आवर्तनम्_ऊर्ध्वम्(raw_rpt, VLEN_FP32);

    संरचना htp_ssm_conv_context scctx = {
        .octx             = octx,
        .kparams          = kparams,
        .nrows_प्रत्येकम्_thread = d_inner_प्रत्येकम्_thread,
        .d_inner_tile     = kparams->d_inner_tile,
        .row_start        = row_start,
        .nrows            = nrows,
    };

    FARF(उच्चः_2, "ssm-conv-f32: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : mode %s\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         kparams->n_t == 1 ? "decode" : "prefill");

    यदि (kparams->n_t == 1) {
        work_queue_सततम्(octx->ctx->work_queue, ssm_conv_thread_f32_decode, &scctx, n_तन्तवः);
    } अन्यथा {
        work_queue_सततम्(octx->ctx->work_queue, ssm_conv_thread_f32_prefill, &scctx, n_तन्तवः);
    }

    प्रत्यागम HTP_स्थितिः_OK;
}

पूर्णाङ्क op_ssm_conv(संरचना htp_ops_context * octx) {
    ध्रुव संरचना htp_tensor * dst = octx->dst;

    विकल्प (dst->type) {
        पक्ष HTP_TYPE_F32:
            प्रत्यागम op_ssm_conv_f32(octx);
        स्वभाव:
            प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
    }
}
