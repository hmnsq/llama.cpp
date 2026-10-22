#ಸೂಚನೆ clang diagnostic ignored "-Wunused-variable"
#ಸೂಚನೆ clang diagnostic ignored "-Wunused-function"
#ಸೂಚನೆ clang diagnostic ignored "-Wunused-but-set-variable"

#ಅನ್ತರ್ಭಾವ <HAP_farf.h>
#ಅನ್ತರ್ಭಾವ <HAP_mem.h>
#ಅನ್ತರ್ಭಾವ <HAP_ps.h>
#ಅನ್ತರ್ಭಾವ <hexagon_protos.h>
#ಅನ್ತರ್ಭಾವ <hexagon_types.h>
#ಅನ್ತರ್ಭಾವ <math.h>
#ಅನ್ತರ್ಭಾವ <qurt_thread.h>
#ಅನ್ತರ್ಭಾವ <string.h>

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_COMMON_DECL_C
#ಅನ್ತರ್ಭಾವ "ggml-common.h"
#ಅನ್ತರ್ಭಾವ "htp-ctx.h"
#ಅನ್ತರ್ಭಾವ "dma-queue.h"
#ಅನ್ತರ್ಭಾವ "hex-profile.h"
#ಅನ್ತರ್ಭಾವ "htp-ops.h"
#ಅನ್ತರ್ಭಾವ "htp-tensor.h"
#ಅನ್ತರ್ಭಾವ "hvx-utils.h"
#ಅನ್ತರ್ಭಾವ "ssm-conv.h"

ರಚನೆ htp_ssm_conv_context {
    ರಚನೆ htp_ops_context *                  octx;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_ssm_conv_kernel_params * kparams;
    uint32_t                                  nrows_ಪ್ರತಿ_thread;
    uint32_t                                  d_inner_tile;
    uint32_t                                  ಅಡ್ಡಸಾಲು_ಆರಂಭ;
    uint32_t                                  nrows;
};

// In-register 32x32 fp32 transpose using std 5-stage HVX vshuff butterfly.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_transpose_32x32_f32(HVX_ಸದಿಶ m[32]) {
    HVX_ಸದಿಶ tmp[32];

    // Stage 0 (R = -4): pair (2i, 2i+1) for i = 0..15. m -> tmp.
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 16; ++i) {
        HVX_ಸದಿಶ_Pair p = Q6_W_vshuff_VVR(m[2*i + 1], m[2*i], -4);
        tmp[2*i + 0] = Q6_V_lo_W(p);
        tmp[2*i + 1] = Q6_V_hi_W(p);
    }

    // Stage 1 (R = -8): per block of 4, pair (b+0, b+2) and (b+1, b+3). tmp -> m.
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < 32; b += 4) {
        HVX_ಸದಿಶ_Pair p0 = Q6_W_vshuff_VVR(tmp[b + 2], tmp[b + 0], -8);
        HVX_ಸದಿಶ_Pair p1 = Q6_W_vshuff_VVR(tmp[b + 3], tmp[b + 1], -8);
        m[b + 0] = Q6_V_lo_W(p0); m[b + 1] = Q6_V_hi_W(p0);
        m[b + 2] = Q6_V_lo_W(p1); m[b + 3] = Q6_V_hi_W(p1);
    }

    // Stage 2 (R = -16): per block of 8, pair (b+i, b+i+4) for i = 0..3. m -> tmp.
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < 32; b += 8) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 4; ++i) {
            HVX_ಸದಿಶ_Pair p = Q6_W_vshuff_VVR(m[b + i + 4], m[b + i], -16);
            tmp[b + 2*i + 0] = Q6_V_lo_W(p);
            tmp[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }

    // Stage 3 (R = -32): per block of 16, pair (b+i, b+i+8) for i = 0..7. tmp -> m.
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < 32; b += 16) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 8; ++i) {
            HVX_ಸದಿಶ_Pair p = Q6_W_vshuff_VVR(tmp[b + i + 8], tmp[b + i], -32);
            m[b + 2*i + 0] = Q6_V_lo_W(p);
            m[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }

    // Stage 4 (R = -64): pair (i, i+16) for i = 0..15. m -> tmp -> m.
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 16; ++i) {
        HVX_ಸದಿಶ_Pair p = Q6_W_vshuff_VVR(m[i + 16], m[i], -64);
        tmp[2 * i + 0]   = Q6_V_lo_W(p);
        tmp[2 * i + 1]   = Q6_V_hi_W(p);
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 32; ++i) {
        m[i] = tmp[i];
    }
}

// HVX deinterleave for d_conv == 4: channel-major raw VTCM -> tap-major T VTCM
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_ssm_conv_unpack_ಗಮ್ಯ_T_4(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * raw, ದಶಮಾಂಶ * T, uint32_t d_inner_ಪ್ರತಿ_thread, uint32_t d_inner_stride) {
    ಗಾಗಿ (uint32_t cb = 0; cb < d_inner_ಪ್ರತಿ_thread; cb += VLEN_FP32) {
        HVX_ಸದಿಶ v0 = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *)(raw + (cb + 0) * 4);
        HVX_ಸದಿಶ v1 = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *)(raw + (cb + 8) * 4);
        HVX_ಸದಿಶ v2 = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *)(raw + (cb + 16) * 4);
        HVX_ಸದಿಶ v3 = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *)(raw + (cb + 24) * 4);

        HVX_ಸದಿಶ_Pair p01 = Q6_W_vdeal_VVR(v1, v0, -4);
        HVX_ಸದಿಶ_Pair p23 = Q6_W_vdeal_VVR(v3, v2, -4);

        HVX_ಸದಿಶ_Pair p_w02 = Q6_W_vdeal_VVR(Q6_V_lo_W(p23), Q6_V_lo_W(p01), -4);
        HVX_ಸದಿಶ_Pair p_w13 = Q6_W_vdeal_VVR(Q6_V_hi_W(p23), Q6_V_hi_W(p01), -4);

        *(HVX_ಸದಿಶ *)(T + 0 * d_inner_stride + cb) = Q6_V_lo_W(p_w02);
        *(HVX_ಸದಿಶ *)(T + 1 * d_inner_stride + cb) = Q6_V_lo_W(p_w13);
        *(HVX_ಸದಿಶ *)(T + 2 * d_inner_stride + cb) = Q6_V_hi_W(p_w02);
        *(HVX_ಸದಿಶ *)(T + 3 * d_inner_stride + cb) = Q6_V_hi_W(p_w13);
    }
}

// HVX transpose for general d_conv <= 32: channel-major raw VTCM -> tap-major T VTCM
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_ssm_conv_unpack_ಗಮ್ಯ_T_gen(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * raw, ದಶಮಾಂಶ * T, uint32_t d_inner_ಪ್ರತಿ_thread, uint32_t d_inner_stride, uint32_t d_conv) {
    uint32_t __attribute__((aligned(VLEN))) ಮುಖವಾಡ_buf[VLEN_FP32] = { 0 };
    ಗಾಗಿ (uint32_t j = 0; j < d_conv; ++j) {
        ಮುಖವಾಡ_buf[j] = 0xFFFFFFFF;
    }
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ ಮುಖವಾಡ = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) ಮುಖವಾಡ_buf;

    ಗಾಗಿ (uint32_t cb = 0; cb < d_inner_ಪ್ರತಿ_thread; cb += VLEN_FP32) {
        ಸ್ಥಿರಾಂಕ uint32_t cb_n = MIN(VLEN_FP32, d_inner_ಪ್ರತಿ_thread - cb);
        HVX_ಸದಿಶ sub[32];
        ಗಾಗಿ (uint32_t r = 0; r < cb_n; ++r) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ch_ptr = raw + (cb + r) * d_conv;
            sub[r] = Q6_V_vand_VV(*(ಸ್ಥಿರಾಂಕ HVX_UVector *) ch_ptr, ಮುಖವಾಡ);
        }
        ಗಾಗಿ (uint32_t r = cb_n; r < 32; ++r) {
            sub[r] = hvx_vec_splat_f32(0.0f);
        }

        hvx_transpose_32x32_f32(sub);

        ಗಾಗಿ (uint32_t j = 0; j < d_conv; ++j) {
            *(HVX_ಸದಿಶ *)(T + j * d_inner_stride + cb) = sub[j];
        }
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_ssm_conv_unpack_ಗಮ್ಯ_T(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * raw, ದಶಮಾಂಶ * T, uint32_t d_inner_ಪ್ರತಿ_thread, uint32_t d_inner_stride, uint32_t d_conv) {
    ಆದರೆ (d_conv == 4 && (d_inner_ಪ್ರತಿ_thread % VLEN_FP32 == 0)) {
        hvx_ssm_conv_unpack_ಗಮ್ಯ_T_4(raw, T, d_inner_ಪ್ರತಿ_thread, d_inner_stride);
    } ಇಲ್ಲದಿದ್ದರೆ {
        hvx_ssm_conv_unpack_ಗಮ್ಯ_T_gen(raw, T, d_inner_ಪ್ರತಿ_thread, d_inner_stride, d_conv);
    }
}

// Decode dot product specialization for d_conv == 4: multiply in the raw channel-major layout,
// then deinterleave the products so each vector holds one tap of 32 channels, and sum.
// Keeps both operands in DMA layout - no transpose, no scratch.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_ssm_conv_decode_4(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * w, ದಶಮಾಂಶ * ಹೊರಹೋಗು, uint32_t n_ch) {
    ಗಾಗಿ (uint32_t cb = 0; cb < n_ch; cb += VLEN_FP32) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * xp = x + cb * 4;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * wp = w + cb * 4;

        HVX_ಸದಿಶ p0 = Q6_Vqf32_vmpy_VsfVsf(*(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *)(xp +  0), *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *)(wp +  0));
        HVX_ಸದಿಶ p1 = Q6_Vqf32_vmpy_VsfVsf(*(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *)(xp + 32), *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *)(wp + 32));
        HVX_ಸದಿಶ p2 = Q6_Vqf32_vmpy_VsfVsf(*(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *)(xp + 64), *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *)(wp + 64));
        HVX_ಸದಿಶ p3 = Q6_Vqf32_vmpy_VsfVsf(*(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *)(xp + 96), *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *)(wp + 96));

        HVX_ಸದಿಶ_Pair p01 = Q6_W_vdeal_VVR(p1, p0, -4);
        HVX_ಸದಿಶ_Pair p23 = Q6_W_vdeal_VVR(p3, p2, -4);

        HVX_ಸದಿಶ_Pair q02 = Q6_W_vdeal_VVR(Q6_V_lo_W(p23), Q6_V_lo_W(p01), -4);
        HVX_ಸದಿಶ_Pair q13 = Q6_W_vdeal_VVR(Q6_V_hi_W(p23), Q6_V_hi_W(p01), -4);

        HVX_ಸದಿಶ a = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(q02), Q6_V_lo_W(q13));
        HVX_ಸದಿಶ b = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_hi_W(q02), Q6_V_hi_W(q13));

        *(HVX_ಸದಿಶ *)(ಹೊರಹೋಗು + cb) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(a, b));
    }
}

// Transpose src0 for prefill: one 32-channel block {32, ncs} (VTCM) -> T {ncs, 32} (VTCM).
// One VTCM gather per output row: lane r picks channel cb+r, the region bound drops
// the lanes past the channel tail.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_ssm_conv_transpose_ಖಂಡ(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * raw_ಖಂಡ,
                                                ದಶಮಾಂಶ *       T,
                                                uint32_t      ncs,
                                                uint32_t      cb_n,
                                                HVX_ಸದಿಶ    vv) {
    ಸ್ಥಿರಾಂಕ size_t   ತಳ = (size_t) raw_ಖಂಡ;
    ಸ್ಥಿರಾಂಕ uint32_t mu   = cb_n * ncs * ಗಾತ್ರ(ದಶಮಾಂಶ) - 1;

    ಗಾಗಿ (uint32_t t = 0; t < ncs; ++t) {
        Q6_vgather_ARMVw((HVX_ಸದಿಶ *) (T + (size_t) t * VLEN_FP32), ತಳ + t * ಗಾತ್ರ(ದಶಮಾಂಶ), mu, vv);
    }
}

// Single-token decode worker (n_t == 1)
ಸ್ಥಿರ ಶೂನ್ಯ ssm_conv_thread_f32_decode(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    ರಚನೆ htp_ssm_conv_context *             scctx   = (ರಚನೆ htp_ssm_conv_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_ops_context *                  octx    = scctx->octx;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_ssm_conv_kernel_params * kparams = scctx->kparams;

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ src0 = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ src1 = octx->src[1];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ dst  = octx->dst;

    dma_ಸರತಿ * dma_q = octx->ctx->dma[ith];

    ಸ್ಥಿರಾಂಕ uint32_t d_conv  = kparams->d_conv;
    ಸ್ಥಿರಾಂಕ uint32_t d_inner = kparams->d_inner;
    ಸ್ಥಿರಾಂಕ uint32_t n_s     = kparams->n_s;

    ಸ್ಥಿರಾಂಕ uint32_t dr  = scctx->nrows_ಪ್ರತಿ_thread;
    ಸ್ಥಿರಾಂಕ uint32_t ir0 = scctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + dr * ith;
    ಸ್ಥಿರಾಂಕ uint32_t ir1 = MIN(ir0 + dr, scctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + scctx->nrows);

    ಆದರೆ (ir0 >= ir1) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ uint32_t d_inner_ಪ್ರತಿ_thread = ir1 - ir0;
    ಸ್ಥಿರಾಂಕ uint32_t d_inner_stride     = hex_ಸುತ್ತು_ಮೇಲೆ(d_inner_ಪ್ರತಿ_thread, VLEN_FP32);
    ಸ್ಥಿರಾಂಕ uint32_t d_inner_tile       = scctx->d_inner_tile;

    ಸ್ಥಿರಾಂಕ size_t src0_stride_inner_ಬೈಟುಗಳು = src0->nb[1];
    ಸ್ಥಿರಾಂಕ size_t src0_stride_seq_ಬೈಟುಗಳು   = src0->nb[2];
    ಸ್ಥಿರಾಂಕ size_t dst_stride_seq_ಬೈಟುಗಳು    = dst->nb[2];

    uint8_t * src1_spad_ತಳ = octx->src1_spad.ದತ್ತಾಂಶ + ith * octx->src1_spad.ಗಾತ್ರ_ಪ್ರತಿ_thread;
    uint8_t * src0_spad_ತಳ = octx->src0_spad.ದತ್ತಾಂಶ + ith * octx->src0_spad.ಗಾತ್ರ_ಪ್ರತಿ_thread;
    uint8_t * dst_spad_ತಳ  = octx->dst_spad.ದತ್ತಾಂಶ  + ith * octx->dst_spad.ಗಾತ್ರ_ಪ್ರತಿ_thread;

    ಸ್ಥಿರಾಂಕ size_t ತೂಕ_ಬೈಟುಗಳು    = (size_t) d_inner_ಪ್ರತಿ_thread * d_conv * ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ size_t ತೂಕ_raw_ಗಾತ್ರ = hex_ಸುತ್ತು_ಮೇಲೆ(ತೂಕ_ಬೈಟುಗಳು, 128);

    ದಶಮಾಂಶ * src1_raw = (ದಶಮಾಂಶ *) src1_spad_ತಳ;
    ದಶಮಾಂಶ * src1_T   = (ದಶಮಾಂಶ *) (src1_spad_ತಳ + ತೂಕ_raw_ಗಾತ್ರ);

    ಸ್ಥಿರಾಂಕ size_t src0_tile_raw_ಬೈಟುಗಳು = hex_ಸುತ್ತು_ಮೇಲೆ(d_inner_tile * d_conv * ಗಾತ್ರ(ದಶಮಾಂಶ), 128);
    ಸ್ಥಿರಾಂಕ size_t dst_tile_ಬೈಟುಗಳು      = hex_ಸುತ್ತು_ಮೇಲೆ(d_inner_tile * ಗಾತ್ರ(ದಶಮಾಂಶ), 128);

    ದಶಮಾಂಶ * src0_tile_raw[2] = { (ದಶಮಾಂಶ *) src0_spad_ತಳ, (ದಶಮಾಂಶ *) (src0_spad_ತಳ + src0_tile_raw_ಬೈಟುಗಳು) };
    ದಶಮಾಂಶ * src0_T           = (ದಶಮಾಂಶ *) (src0_spad_ತಳ + 2 * src0_tile_raw_ಬೈಟುಗಳು);

    ದಶಮಾಂಶ * dst_tile[2] = { (ದಶಮಾಂಶ *) dst_spad_ತಳ, (ದಶಮಾಂಶ *) (dst_spad_ತಳ + dst_tile_ಬೈಟುಗಳು) };

    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];

    // raw_dot keeps both operands in the DMA layout, so src1 needs no prep pass
    ಸ್ಥಿರಾಂಕ bool raw_dot = (d_conv == 4) && (d_inner_ಪ್ರತಿ_thread % VLEN_FP32 == 0);

    ಸ್ಥಿರಾಂಕ uint32_t n_tiles   = (d_inner_ಪ್ರತಿ_thread + d_inner_tile - 1) / d_inner_tile;
    ಸ್ಥಿರಾಂಕ uint32_t n_chunks  = n_s * n_tiles;
    ಸ್ಥಿರಾಂಕ size_t   ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು = d_conv * ಗಾತ್ರ(ದಶಮಾಂಶ);

    uint32_t s_fetch        = 0;
    uint32_t tile_off_fetch = 0;

    #ವ್ಯಾಖ್ಯಾನಿಸು SSM_CONV_DECODE_ತಳ್ಳು_FETCH(c)                                                         \
        ಮಾಡು {                                                                                      \
            ಸ್ಥಿರಾಂಕ uint32_t   cur_tile_n = MIN(d_inner_tile, d_inner_ಪ್ರತಿ_thread - tile_off_fetch); \
            ಸ್ಥಿರಾಂಕ dma_addr_t fetch_ddr  = src0->ದತ್ತಾಂಶ + s_fetch * src0_stride_seq_ಬೈಟುಗಳು +          \
                                          (ir0 + tile_off_fetch) * src0_stride_inner_ಬೈಟುಗಳು;       \
            dma_ಸರತಿ_ತಳ್ಳು(dma_q,                                                                 \
                           dma_make_ದತ್ತಾಂಶ((uint8_t *) src0_tile_raw[(c) & 1], fetch_ddr),          \
                           ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, src0_stride_inner_ಬೈಟುಗಳು, ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, cur_tile_n);            \
            tile_off_fetch += d_inner_tile;                                                       \
            ಆದರೆ (tile_off_fetch >= d_inner_ಪ್ರತಿ_thread) {                                           \
                tile_off_fetch = 0;                                                               \
                s_fetch++;                                                                        \
            }                                                                                     \
        } ಆಗ (0)

    // Queue weights and initial input tiles together so DDR reads overlap
    ಸ್ಥಿರಾಂಕ dma_addr_t src1_ddr = src1->ದತ್ತಾಂಶ + ir0 * d_conv * ಗಾತ್ರ(ದಶಮಾಂಶ);
    dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ((uint8_t *) src1_raw, src1_ddr), ತೂಕ_ಬೈಟುಗಳು, ತೂಕ_ಬೈಟುಗಳು, ತೂಕ_ಬೈಟುಗಳು, 1);

    SSM_CONV_DECODE_ತಳ್ಳು_FETCH(0);
    ಆದರೆ (n_chunks > 1) {
        SSM_CONV_DECODE_ತಳ್ಳು_FETCH(1);
    }

    dma_ಸರತಿ_ತೆಗೆ(dma_q);  // weights

    ಆದರೆ (!raw_dot) {
        // Unpack/transpose src1_raw into src1_T {d_conv, d_inner_stride}
        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);
        hvx_ssm_conv_unpack_ಗಮ್ಯ_T(src1_raw, src1_T, d_inner_ಪ್ರತಿ_thread, d_inner_stride, d_conv);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);
    }

    uint32_t i3       = 0;
    uint32_t tile_off = 0;

    ಗಾಗಿ (uint32_t c = 0; c < n_chunks; ++c) {
        ಸ್ಥಿರಾಂಕ uint32_t tile_n = MIN(d_inner_tile, d_inner_ಪ್ರತಿ_thread - tile_off);

        ಆದರೆ (c >= 2) {
            dma_ಸರತಿ_ತೆಗೆ(dma_q);  // writeback of chunk c-2, frees dst_tile[c & 1]
        }
        dma_ಸರತಿ_ತೆಗೆ(dma_q);      // fetch chunk c

        ದಶಮಾಂಶ * ಸೀಮಿತ ಹೊರಹೋಗು = dst_tile[c & 1];

        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i3);
        ಆದರೆ (raw_dot) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * xp = src0_tile_raw[c & 1];
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * wp = src1_raw + tile_off * 4;
            hvx_ssm_conv_decode_4(xp, wp, ಹೊರಹೋಗು, tile_n);
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಸ್ಥಿರಾಂಕ uint32_t tile_stride = hex_ಸುತ್ತು_ಮೇಲೆ(tile_n, VLEN_FP32);
            hvx_ssm_conv_unpack_ಗಮ್ಯ_T(src0_tile_raw[c & 1], src0_T, tile_n, tile_stride, d_conv);

            ಗಾಗಿ (uint32_t cb = 0; cb < tile_n; cb += VLEN_FP32) {
                ಸ್ಥಿರಾಂಕ uint32_t cb_n = MIN(VLEN_FP32, tile_n - cb);
                HVX_ಸದಿಶ acc = hvx_vec_splat_f32(0.0f);
                ಗಾಗಿ (uint32_t j = 0; j < d_conv; ++j) {
                    HVX_ಸದಿಶ x = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *)(src0_T + j * tile_stride + cb);
                    HVX_ಸದಿಶ w = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *)(src1_T + j * d_inner_stride + tile_off + cb);
                    acc          = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(x, w));
                }
                HVX_ಸದಿಶ y = Q6_Vsf_equals_Vqf32(acc);
                ಆದರೆ (cb_n == VLEN_FP32) {
                    *(HVX_ಸದಿಶ *)(ಹೊರಹೋಗು + cb) = y;
                } ಇಲ್ಲದಿದ್ದರೆ {
                    hvx_vec_store_u(ಹೊರಹೋಗು + cb, cb_n * ಗಾತ್ರ(ದಶಮಾಂಶ), y);
                }
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i3);

        ಸ್ಥಿರಾಂಕ dma_addr_t dst_ddr        = dst->ದತ್ತಾಂಶ + i3 * dst_stride_seq_ಬೈಟುಗಳು + (ir0 + tile_off) * ಗಾತ್ರ(ದಶಮಾಂಶ);
        ಸ್ಥಿರಾಂಕ size_t     tile_ಹೊರಹೋಗು_ಬೈಟುಗಳು = (size_t) tile_n * ಗಾತ್ರ(ದಶಮಾಂಶ);
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(dst_ddr, (uint8_t *) ಹೊರಹೋಗು),
                       tile_ಹೊರಹೋಗು_ಬೈಟುಗಳು, tile_ಹೊರಹೋಗು_ಬೈಟುಗಳು, tile_ಹೊರಹೋಗು_ಬೈಟುಗಳು, 1);

        ಆದರೆ (c + 2 < n_chunks) {
            SSM_CONV_DECODE_ತಳ್ಳು_FETCH(c + 2);
        }

        tile_off += d_inner_tile;
        ಆದರೆ (tile_off >= d_inner_ಪ್ರತಿ_thread) {
            tile_off = 0;
            i3++;
        }
    }

    ಗಾಗಿ (uint32_t k = MIN(n_chunks, 2); k > 0; --k) {
        dma_ಸರತಿ_ತೆಗೆ(dma_q);  // drain the last writebacks
    }

    #ವ್ಯಾಖ್ಯಾನ_ರದ್ದು SSM_CONV_DECODE_ತಳ್ಳು_FETCH

    FARF(ಮೇಲಿನ_2, "ssm-conv-f32-decode %d/%d: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0], dst->ne[1],
         dst->ne[2], dst->ne[3]);
}

// Multi-token prefill worker (n_t > 1)
ಸ್ಥಿರ ಶೂನ್ಯ ssm_conv_thread_f32_prefill(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    ರಚನೆ htp_ssm_conv_context *             scctx   = (ರಚನೆ htp_ssm_conv_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_ops_context *                  octx    = scctx->octx;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_ssm_conv_kernel_params * kparams = scctx->kparams;

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ src0 = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ src1 = octx->src[1];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ dst  = octx->dst;

    dma_ಸರತಿ * dma_q = octx->ctx->dma[ith];

    ಸ್ಥಿರಾಂಕ uint32_t d_conv  = kparams->d_conv;
    ಸ್ಥಿರಾಂಕ uint32_t d_inner = kparams->d_inner;
    ಸ್ಥಿರಾಂಕ uint32_t n_t     = kparams->n_t;
    ಸ್ಥಿರಾಂಕ uint32_t n_s     = kparams->n_s;
    ಸ್ಥಿರಾಂಕ uint32_t ncs     = src0->ne[0];

    ಸ್ಥಿರಾಂಕ uint32_t dr  = scctx->nrows_ಪ್ರತಿ_thread;
    ಸ್ಥಿರಾಂಕ uint32_t ir0 = scctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + dr * ith;
    ಸ್ಥಿರಾಂಕ uint32_t ir1 = MIN(ir0 + dr, scctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + scctx->nrows);

    ಆದರೆ (ir0 >= ir1) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ uint32_t d_inner_ಪ್ರತಿ_thread = ir1 - ir0;
    ಸ್ಥಿರಾಂಕ uint32_t d_inner_stride     = hex_ಸುತ್ತು_ಮೇಲೆ(d_inner_ಪ್ರತಿ_thread, VLEN_FP32);
    ಸ್ಥಿರಾಂಕ uint32_t d_inner_tile       = scctx->d_inner_tile;

    ಸ್ಥಿರಾಂಕ size_t src0_stride_inner_ಬೈಟುಗಳು = src0->nb[1];
    ಸ್ಥಿರಾಂಕ size_t src0_stride_seq_ಬೈಟುಗಳು   = src0->nb[2];
    ಸ್ಥಿರಾಂಕ size_t dst_stride_token_ಬೈಟುಗಳು  = dst->nb[1];
    ಸ್ಥಿರಾಂಕ size_t dst_stride_seq_ಬೈಟುಗಳು    = dst->nb[2];

    uint8_t * src1_spad_ತಳ = octx->src1_spad.ದತ್ತಾಂಶ + ith * octx->src1_spad.ಗಾತ್ರ_ಪ್ರತಿ_thread;
    uint8_t * src0_spad_ತಳ = octx->src0_spad.ದತ್ತಾಂಶ + ith * octx->src0_spad.ಗಾತ್ರ_ಪ್ರತಿ_thread;
    uint8_t * dst_spad_ತಳ  = octx->dst_spad.ದತ್ತಾಂಶ  + ith * octx->dst_spad.ಗಾತ್ರ_ಪ್ರತಿ_thread;

    ಸ್ಥಿರಾಂಕ size_t ತೂಕ_ಬೈಟುಗಳು    = (size_t) d_inner_ಪ್ರತಿ_thread * d_conv * ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ size_t ತೂಕ_raw_ಗಾತ್ರ = hex_ಸುತ್ತು_ಮೇಲೆ(ತೂಕ_ಬೈಟುಗಳು, 128);

    ದಶಮಾಂಶ * src1_raw = (ದಶಮಾಂಶ *) src1_spad_ತಳ;
    ದಶಮಾಂಶ * src1_T   = (ದಶಮಾಂಶ *) (src1_spad_ತಳ + ತೂಕ_raw_ಗಾತ್ರ);

    // src0 spad holds two raw tiles (fetch of tile n+1 overlaps compute of tile n) plus
    // the transposed block. dst spad holds two tiles so a writeback can stay in flight.
    ಸ್ಥಿರಾಂಕ size_t src0_tile_raw_ಬೈಟುಗಳು = hex_ಸುತ್ತು_ಮೇಲೆ(d_inner_tile * ncs * ಗಾತ್ರ(ದಶಮಾಂಶ), 128);
    ಸ್ಥಿರಾಂಕ size_t dst_tile_ಬೈಟುಗಳು      = hex_ಸುತ್ತು_ಮೇಲೆ(d_inner_tile * n_t * ಗಾತ್ರ(ದಶಮಾಂಶ), 128);

    ದಶಮಾಂಶ * src0_tile_raw[2] = { (ದಶಮಾಂಶ *) src0_spad_ತಳ, (ದಶಮಾಂಶ *) (src0_spad_ತಳ + src0_tile_raw_ಬೈಟುಗಳು) };
    ದಶಮಾಂಶ * src0_T           = (ದಶಮಾಂಶ *) (src0_spad_ತಳ + 2 * src0_tile_raw_ಬೈಟುಗಳು);

    ದಶಮಾಂಶ * dst_tile[2] = { (ದಶಮಾಂಶ *) dst_spad_ತಳ, (ದಶಮಾಂಶ *) (dst_spad_ತಳ + dst_tile_ಬೈಟುಗಳು) };

    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ಸ್ಥಿರಾಂಕ uint32_t n_tiles   = (d_inner_ಪ್ರತಿ_thread + d_inner_tile - 1) / d_inner_tile;
    ಸ್ಥಿರಾಂಕ uint32_t n_chunks  = n_s * n_tiles;
    ಸ್ಥಿರಾಂಕ size_t   ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು = ncs * ಗಾತ್ರ(ದಶಮಾಂಶ);

    uint32_t s_fetch        = 0;
    uint32_t tile_off_fetch = 0;

    // Chunk c fetches into src0_tile_raw[c & 1] and writes back from dst_tile[c & 1].
    // Two fetches run ahead, so the queue order is F0 F1 W0 F2 W1 ... and pops follow it.
    #ವ್ಯಾಖ್ಯಾನಿಸು SSM_CONV_ತಳ್ಳು_FETCH(c)                                                                \
        ಮಾಡು {                                                                                      \
            ಸ್ಥಿರಾಂಕ uint32_t   cur_tile_n = MIN(d_inner_tile, d_inner_ಪ್ರತಿ_thread - tile_off_fetch); \
            ಸ್ಥಿರಾಂಕ dma_addr_t fetch_ddr  = src0->ದತ್ತಾಂಶ + s_fetch * src0_stride_seq_ಬೈಟುಗಳು +          \
                                          (ir0 + tile_off_fetch) * src0_stride_inner_ಬೈಟುಗಳು;       \
            dma_ಸರತಿ_ತಳ್ಳು(dma_q,                                                                 \
                           dma_make_ದತ್ತಾಂಶ((uint8_t *) src0_tile_raw[(c) & 1], fetch_ddr),          \
                           ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, src0_stride_inner_ಬೈಟುಗಳು, ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, cur_tile_n);            \
            tile_off_fetch += d_inner_tile;                                                       \
            ಆದರೆ (tile_off_fetch >= d_inner_ಪ್ರತಿ_thread) {                                           \
                tile_off_fetch = 0;                                                               \
                s_fetch++;                                                                        \
            }                                                                                     \
        } ಆಗ (0)

    // Queue weights and initial input tiles together so DDR reads overlap
    ಸ್ಥಿರಾಂಕ dma_addr_t src1_ddr = src1->ದತ್ತಾಂಶ + ir0 * d_conv * ಗಾತ್ರ(ದಶಮಾಂಶ);
    dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ((uint8_t *) src1_raw, src1_ddr), ತೂಕ_ಬೈಟುಗಳು, ತೂಕ_ಬೈಟುಗಳು, ತೂಕ_ಬೈಟುಗಳು, 1);

    SSM_CONV_ತಳ್ಳು_FETCH(0);
    ಆದರೆ (n_chunks > 1) {
        SSM_CONV_ತಳ್ಳು_FETCH(1);
    }

    dma_ಸರತಿ_ತೆಗೆ(dma_q);  // weights

    // Unpack/transpose src1_raw into src1_T {d_conv, d_inner_stride}
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);
    hvx_ssm_conv_unpack_ಗಮ್ಯ_T(src1_raw, src1_T, d_inner_ಪ್ರತಿ_thread, d_inner_stride, d_conv);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);

    ಸ್ಥಿರಾಂಕ uint32_t C_TILE = VLEN_FP32;

    // gather offsets: lane r reads channel r of the raw tile block
    uint32_t __attribute__((aligned(VLEN))) gather_off[VLEN_FP32];
    ಗಾಗಿ (uint32_t r = 0; r < VLEN_FP32; ++r) {
        gather_off[r] = r * ncs * ಗಾತ್ರ(ದಶಮಾಂಶ);
    }
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ vv = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) gather_off;

    uint32_t i3       = 0;
    uint32_t tile_off = 0;

    ಗಾಗಿ (uint32_t c = 0; c < n_chunks; ++c) {
        ಸ್ಥಿರಾಂಕ uint32_t tile_n = MIN(d_inner_tile, d_inner_ಪ್ರತಿ_thread - tile_off);

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ raw = src0_tile_raw[c & 1];
        ದಶಮಾಂಶ * ಸೀಮಿತ       ಹೊರಹೋಗು = dst_tile[c & 1];

        ಆದರೆ (c >= 2) {
            dma_ಸರತಿ_ತೆಗೆ(dma_q);  // writeback of chunk c-2, frees dst_tile[c & 1]
        }
        dma_ಸರತಿ_ತೆಗೆ(dma_q);      // fetch chunk c

        // Channel block outer, token inner: the taps and the sliding window stay in
        // registers, so a new output row costs one src0_T load.
        ಸ್ಥಿರಾಂಕ uint32_t dst_tile_stride = hex_ಸುತ್ತು_ಮೇಲೆ(tile_n, C_TILE);

        ಗಾಗಿ (uint32_t cb = 0; cb < tile_n; cb += C_TILE) {
            ಸ್ಥಿರಾಂಕ uint32_t cb_n = MIN(C_TILE, tile_n - cb);

            htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) tile_off);
            hvx_ssm_conv_transpose_ಖಂಡ(raw + (size_t) cb * ncs, src0_T, ncs, cb_n, vv);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) tile_off);

            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ wp = src1_T + tile_off + cb;
            ದಶಮಾಂಶ * ಸೀಮಿತ       op = ಹೊರಹೋಗು + cb;

            htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) tile_off);
            ಆದರೆ (d_conv == 4) {
                ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ w0 = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (wp);
                ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ w1 = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (wp + d_inner_stride);
                ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ w2 = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (wp + 2 * d_inner_stride);
                ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ w3 = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (wp + 3 * d_inner_stride);

                HVX_ಸದಿಶ x0 = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (src0_T);
                HVX_ಸದಿಶ x1 = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (src0_T + C_TILE);
                HVX_ಸದಿಶ x2 = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (src0_T + 2 * C_TILE);

                ಗಾಗಿ (uint32_t t = 0; t < n_t; ++t) {
                    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ x3 = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (src0_T + (t + 3) * C_TILE);

                    HVX_ಸದಿಶ a = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(x0, w0),
                                                            Q6_Vqf32_vmpy_VsfVsf(x1, w1));
                    HVX_ಸದಿಶ b = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(x2, w2),
                                                            Q6_Vqf32_vmpy_VsfVsf(x3, w3));

                    *(HVX_ಸದಿಶ *) (op + t * dst_tile_stride) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(a, b));

                    x0 = x1;
                    x1 = x2;
                    x2 = x3;
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಗಾಗಿ (uint32_t t = 0; t < n_t; ++t) {
                    HVX_ಸದಿಶ acc = hvx_vec_splat_f32(0.0f);
                    ಗಾಗಿ (uint32_t j = 0; j < d_conv; ++j) {
                        HVX_ಸದಿಶ x = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (src0_T + (t + j) * C_TILE);
                        HVX_ಸದಿಶ w = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (wp + j * d_inner_stride);
                        acc          = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(x, w));
                    }
                    *(HVX_ಸದಿಶ *) (op + t * dst_tile_stride) = Q6_Vsf_equals_Vqf32(acc);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) tile_off);
        }

        // Writeback dst_tile VTCM -> DDR via 2D DMA
        dma_ಸರತಿ_ತಳ್ಳು(dma_q,
                       dma_make_ದತ್ತಾಂಶ(dst->ದತ್ತಾಂಶ + i3 * dst_stride_seq_ಬೈಟುಗಳು + (ir0 + tile_off) * ಗಾತ್ರ(ದಶಮಾಂಶ),
                                     (uint8_t *) ಹೊರಹೋಗು),
                       dst_stride_token_ಬೈಟುಗಳು, dst_tile_stride * ಗಾತ್ರ(ದಶಮಾಂಶ), tile_n * ಗಾತ್ರ(ದಶಮಾಂಶ), n_t);

        ಆದರೆ (c + 2 < n_chunks) {
            SSM_CONV_ತಳ್ಳು_FETCH(c + 2);
        }

        tile_off += d_inner_tile;
        ಆದರೆ (tile_off >= d_inner_ಪ್ರತಿ_thread) {
            tile_off = 0;
            i3++;
        }
    }

    ಗಾಗಿ (uint32_t k = MIN(n_chunks, 2); k > 0; --k) {
        dma_ಸರತಿ_ತೆಗೆ(dma_q);  // drain the last writebacks
    }

    #ವ್ಯಾಖ್ಯಾನ_ರದ್ದು SSM_CONV_ತಳ್ಳು_FETCH

    FARF(ಮೇಲಿನ_2, "ssm-conv-f32-prefill %d/%d: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0], dst->ne[1],
         dst->ne[2], dst->ne[3]);
}

ಪೂರ್ಣಾಂಕ op_ssm_conv_f32(ರಚನೆ htp_ops_context * octx) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src0 = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src1 = octx->src[1];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst  = octx->dst;

    ಆದರೆ (src0->type != HTP_TYPE_F32 || src1->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_ssm_conv_kernel_params * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_ssm_conv_kernel_params *) octx->kernel_params;


    ಆದರೆ (!htp_ops_context_ಗಣ_n_ಎಳೆಗಳು(octx, kparams->n_ಎಳೆಗಳು)) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_INVAL_PARAMS;
    }

    uint32_t ಅಡ್ಡಸಾಲು_ಆರಂಭ = 0;
    uint32_t nrows     = kparams->d_inner;

    ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1) {
        ಸ್ಥಿರಾಂಕ uint32_t elems_ಪ್ರತಿ_chunk = VLEN_FP32;
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor_mdev_range range = htp_tensor_mdev_ವಿಭಜನೆ(
            kparams->d_inner,
            htp_tensor_mdev_ದತ್ತಾಂಶ_aligned(dst) ? elems_ಪ್ರತಿ_chunk : 0,
            octx->ctx->mdev.idx,
            octx->ctx->mdev.ಎಣಿಕೆ_2,
            &octx->ctx->mdev.ಎಣಿಕೆ_div
        );
        ಅಡ್ಡಸಾಲು_ಆರಂಭ = range.ಆರಂಭ;
        nrows     = range.ಎಣಿಕೆ_2;
    }

    ಆದರೆ (nrows == 0) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
    }

    ಆದರೆ (kparams->vtcm_ಗಾತ್ರ > octx->ctx->vtcm_ಗಾತ್ರ) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_VTCM_TOO_SMALL;
    }

    ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು = octx->n_ಎಳೆಗಳು;

    octx->src0_spad.ಗಾತ್ರ_ಪ್ರತಿ_thread = kparams->vtcm_src0_ಗಾತ್ರ_ಪ್ರತಿ_thread;
    octx->src1_spad.ಗಾತ್ರ_ಪ್ರತಿ_thread = kparams->vtcm_src1_ಗಾತ್ರ_ಪ್ರತಿ_thread;
    octx->dst_spad.ಗಾತ್ರ_ಪ್ರತಿ_thread  = kparams->vtcm_dst_ಗಾತ್ರ_ಪ್ರತಿ_thread;

    octx->src0_spad.ಗಾತ್ರ_2 = kparams->vtcm_src0_ಗಾತ್ರ;
    octx->src1_spad.ಗಾತ್ರ_2 = kparams->vtcm_src1_ಗಾತ್ರ;
    octx->dst_spad.ಗಾತ್ರ_2  = kparams->vtcm_dst_ಗಾತ್ರ;

    octx->src0_spad.ದತ್ತಾಂಶ = octx->ctx->vtcm_ತಳ;
    octx->src1_spad.ದತ್ತಾಂಶ = octx->src0_spad.ದತ್ತಾಂಶ + octx->src0_spad.ಗಾತ್ರ_2;
    octx->dst_spad.ದತ್ತಾಂಶ  = octx->src1_spad.ದತ್ತಾಂಶ + octx->src1_spad.ಗಾತ್ರ_2;
    octx->src0_spad.src  = ಖಾಲಿ;
    octx->src1_spad.src  = ಖಾಲಿ;
    octx->dst_spad.src   = ಖಾಲಿ;

    ಸ್ಥಿರಾಂಕ uint32_t raw_rpt            = fastdiv(nrows + n_ಎಳೆಗಳು - 1, &octx->n_ಎಳೆಗಳು_div);
    ಸ್ಥಿರಾಂಕ uint32_t d_inner_ಪ್ರತಿ_thread = hex_ಸುತ್ತು_ಮೇಲೆ(raw_rpt, VLEN_FP32);

    ರಚನೆ htp_ssm_conv_context scctx = {
        .octx             = octx,
        .kparams          = kparams,
        .nrows_ಪ್ರತಿ_thread = d_inner_ಪ್ರತಿ_thread,
        .d_inner_tile     = kparams->d_inner_tile,
        .ಅಡ್ಡಸಾಲು_ಆರಂಭ        = ಅಡ್ಡಸಾಲು_ಆರಂಭ,
        .nrows            = nrows,
    };

    FARF(ಮೇಲಿನ_2, "ssm-conv-f32: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : mode %s\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         kparams->n_t == 1 ? "decode" : "prefill");

    ಆದರೆ (kparams->n_t == 1) {
        work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, ssm_conv_thread_f32_decode, &scctx, n_ಎಳೆಗಳು);
    } ಇಲ್ಲದಿದ್ದರೆ {
        work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, ssm_conv_thread_f32_prefill, &scctx, n_ಎಳೆಗಳು);
    }

    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
}

ಪೂರ್ಣಾಂಕ op_ssm_conv(ರಚನೆ htp_ops_context * octx) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst = octx->dst;

    ಆಯ್ಕೆ (dst->type) {
        ಸ್ಥಿತಿ HTP_TYPE_F32:
            ಹಿಂತಿರುಗಿ op_ssm_conv_f32(octx);
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }
}
