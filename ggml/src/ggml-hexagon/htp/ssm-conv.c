#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-variable"
#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-function"
#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-but-set-variable"

#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_farf.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_mem.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_ps.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <hexagon_protos.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <hexagon_types.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <math.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <qurt_thread.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_DECL_C
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ctx.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "dma-queue.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-profile.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-utils.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ssm-conv.h"

ਬਣਤਰ htp_ssm_conv_context {
    ਬਣਤਰ htp_ops_context *                  octx;
    ਅਚਲ ਬਣਤਰ htp_ssm_conv_kernel_params * kparams;
    uint32_t                                  nrows_ਪ੍ਰਤੀ_thread;
    uint32_t                                  d_inner_tile;
    uint32_t                                  ਪੰਕਤੀ_ਸ਼ੁਰੂ;
    uint32_t                                  nrows;
};

// In-register 32x32 fp32 transpose using std 5-stage HVX vshuff butterfly.
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_transpose_32x32_f32(HVX_ਸਦਿਸ਼ m[32]) {
    HVX_ਸਦਿਸ਼ tmp[32];

    // Stage 0 (R = -4): pair (2i, 2i+1) for i = 0..15. m -> tmp.
    ਲਈ (ਪੂਰਨ i = 0; i < 16; ++i) {
        HVX_ਸਦਿਸ਼_Pair p = Q6_W_vshuff_VVR(m[2*i + 1], m[2*i], -4);
        tmp[2*i + 0] = Q6_V_lo_W(p);
        tmp[2*i + 1] = Q6_V_hi_W(p);
    }

    // Stage 1 (R = -8): per block of 4, pair (b+0, b+2) and (b+1, b+3). tmp -> m.
    ਲਈ (ਪੂਰਨ b = 0; b < 32; b += 4) {
        HVX_ਸਦਿਸ਼_Pair p0 = Q6_W_vshuff_VVR(tmp[b + 2], tmp[b + 0], -8);
        HVX_ਸਦਿਸ਼_Pair p1 = Q6_W_vshuff_VVR(tmp[b + 3], tmp[b + 1], -8);
        m[b + 0] = Q6_V_lo_W(p0); m[b + 1] = Q6_V_hi_W(p0);
        m[b + 2] = Q6_V_lo_W(p1); m[b + 3] = Q6_V_hi_W(p1);
    }

    // Stage 2 (R = -16): per block of 8, pair (b+i, b+i+4) for i = 0..3. m -> tmp.
    ਲਈ (ਪੂਰਨ b = 0; b < 32; b += 8) {
        ਲਈ (ਪੂਰਨ i = 0; i < 4; ++i) {
            HVX_ਸਦਿਸ਼_Pair p = Q6_W_vshuff_VVR(m[b + i + 4], m[b + i], -16);
            tmp[b + 2*i + 0] = Q6_V_lo_W(p);
            tmp[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }

    // Stage 3 (R = -32): per block of 16, pair (b+i, b+i+8) for i = 0..7. tmp -> m.
    ਲਈ (ਪੂਰਨ b = 0; b < 32; b += 16) {
        ਲਈ (ਪੂਰਨ i = 0; i < 8; ++i) {
            HVX_ਸਦਿਸ਼_Pair p = Q6_W_vshuff_VVR(tmp[b + i + 8], tmp[b + i], -32);
            m[b + 2*i + 0] = Q6_V_lo_W(p);
            m[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }

    // Stage 4 (R = -64): pair (i, i+16) for i = 0..15. m -> tmp -> m.
    ਲਈ (ਪੂਰਨ i = 0; i < 16; ++i) {
        HVX_ਸਦਿਸ਼_Pair p = Q6_W_vshuff_VVR(m[i + 16], m[i], -64);
        tmp[2 * i + 0]   = Q6_V_lo_W(p);
        tmp[2 * i + 1]   = Q6_V_hi_W(p);
    }

    ਲਈ (ਪੂਰਨ i = 0; i < 32; ++i) {
        m[i] = tmp[i];
    }
}

// HVX deinterleave for d_conv == 4: channel-major raw VTCM -> tap-major T VTCM
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_ssm_conv_unpack_ਮੰਜ਼ਿਲ_T_4(ਅਚਲ ਦਸ਼ਮਲਵ * raw, ਦਸ਼ਮਲਵ * T, uint32_t d_inner_ਪ੍ਰਤੀ_thread, uint32_t d_inner_stride) {
    ਲਈ (uint32_t cb = 0; cb < d_inner_ਪ੍ਰਤੀ_thread; cb += VLEN_FP32) {
        HVX_ਸਦਿਸ਼ v0 = *(ਅਚਲ HVX_ਸਦਿਸ਼ *)(raw + (cb + 0) * 4);
        HVX_ਸਦਿਸ਼ v1 = *(ਅਚਲ HVX_ਸਦਿਸ਼ *)(raw + (cb + 8) * 4);
        HVX_ਸਦਿਸ਼ v2 = *(ਅਚਲ HVX_ਸਦਿਸ਼ *)(raw + (cb + 16) * 4);
        HVX_ਸਦਿਸ਼ v3 = *(ਅਚਲ HVX_ਸਦਿਸ਼ *)(raw + (cb + 24) * 4);

        HVX_ਸਦਿਸ਼_Pair p01 = Q6_W_vdeal_VVR(v1, v0, -4);
        HVX_ਸਦਿਸ਼_Pair p23 = Q6_W_vdeal_VVR(v3, v2, -4);

        HVX_ਸਦਿਸ਼_Pair p_w02 = Q6_W_vdeal_VVR(Q6_V_lo_W(p23), Q6_V_lo_W(p01), -4);
        HVX_ਸਦਿਸ਼_Pair p_w13 = Q6_W_vdeal_VVR(Q6_V_hi_W(p23), Q6_V_hi_W(p01), -4);

        *(HVX_ਸਦਿਸ਼ *)(T + 0 * d_inner_stride + cb) = Q6_V_lo_W(p_w02);
        *(HVX_ਸਦਿਸ਼ *)(T + 1 * d_inner_stride + cb) = Q6_V_lo_W(p_w13);
        *(HVX_ਸਦਿਸ਼ *)(T + 2 * d_inner_stride + cb) = Q6_V_hi_W(p_w02);
        *(HVX_ਸਦਿਸ਼ *)(T + 3 * d_inner_stride + cb) = Q6_V_hi_W(p_w13);
    }
}

// HVX transpose for general d_conv <= 32: channel-major raw VTCM -> tap-major T VTCM
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_ssm_conv_unpack_ਮੰਜ਼ਿਲ_T_gen(ਅਚਲ ਦਸ਼ਮਲਵ * raw, ਦਸ਼ਮਲਵ * T, uint32_t d_inner_ਪ੍ਰਤੀ_thread, uint32_t d_inner_stride, uint32_t d_conv) {
    uint32_t __attribute__((aligned(VLEN))) ਨਕਾਬ_buf[VLEN_FP32] = { 0 };
    ਲਈ (uint32_t j = 0; j < d_conv; ++j) {
        ਨਕਾਬ_buf[j] = 0xFFFFFFFF;
    }
    ਅਚਲ HVX_ਸਦਿਸ਼ ਨਕਾਬ = *(ਅਚਲ HVX_ਸਦਿਸ਼ *) ਨਕਾਬ_buf;

    ਲਈ (uint32_t cb = 0; cb < d_inner_ਪ੍ਰਤੀ_thread; cb += VLEN_FP32) {
        ਅਚਲ uint32_t cb_n = MIN(VLEN_FP32, d_inner_ਪ੍ਰਤੀ_thread - cb);
        HVX_ਸਦਿਸ਼ sub[32];
        ਲਈ (uint32_t r = 0; r < cb_n; ++r) {
            ਅਚਲ ਦਸ਼ਮਲਵ * ch_ptr = raw + (cb + r) * d_conv;
            sub[r] = Q6_V_vand_VV(*(ਅਚਲ HVX_UVector *) ch_ptr, ਨਕਾਬ);
        }
        ਲਈ (uint32_t r = cb_n; r < 32; ++r) {
            sub[r] = hvx_vec_splat_f32(0.0f);
        }

        hvx_transpose_32x32_f32(sub);

        ਲਈ (uint32_t j = 0; j < d_conv; ++j) {
            *(HVX_ਸਦਿਸ਼ *)(T + j * d_inner_stride + cb) = sub[j];
        }
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_ssm_conv_unpack_ਮੰਜ਼ਿਲ_T(ਅਚਲ ਦਸ਼ਮਲਵ * raw, ਦਸ਼ਮਲਵ * T, uint32_t d_inner_ਪ੍ਰਤੀ_thread, uint32_t d_inner_stride, uint32_t d_conv) {
    ਜੇ (d_conv == 4 && (d_inner_ਪ੍ਰਤੀ_thread % VLEN_FP32 == 0)) {
        hvx_ssm_conv_unpack_ਮੰਜ਼ਿਲ_T_4(raw, T, d_inner_ਪ੍ਰਤੀ_thread, d_inner_stride);
    } ਨਹੀਂਤਾਂ {
        hvx_ssm_conv_unpack_ਮੰਜ਼ਿਲ_T_gen(raw, T, d_inner_ਪ੍ਰਤੀ_thread, d_inner_stride, d_conv);
    }
}

// Decode dot product specialization for d_conv == 4: multiply in the raw channel-major layout,
// then deinterleave the products so each vector holds one tap of 32 channels, and sum.
// Keeps both operands in DMA layout - no transpose, no scratch.
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_ssm_conv_decode_4(ਅਚਲ ਦਸ਼ਮਲਵ * x, ਅਚਲ ਦਸ਼ਮਲਵ * w, ਦਸ਼ਮਲਵ * ਬਾਹਰਲਾ, uint32_t n_ch) {
    ਲਈ (uint32_t cb = 0; cb < n_ch; cb += VLEN_FP32) {
        ਅਚਲ ਦਸ਼ਮਲਵ * xp = x + cb * 4;
        ਅਚਲ ਦਸ਼ਮਲਵ * wp = w + cb * 4;

        HVX_ਸਦਿਸ਼ p0 = Q6_Vqf32_vmpy_VsfVsf(*(ਅਚਲ HVX_ਸਦਿਸ਼ *)(xp +  0), *(ਅਚਲ HVX_ਸਦਿਸ਼ *)(wp +  0));
        HVX_ਸਦਿਸ਼ p1 = Q6_Vqf32_vmpy_VsfVsf(*(ਅਚਲ HVX_ਸਦਿਸ਼ *)(xp + 32), *(ਅਚਲ HVX_ਸਦਿਸ਼ *)(wp + 32));
        HVX_ਸਦਿਸ਼ p2 = Q6_Vqf32_vmpy_VsfVsf(*(ਅਚਲ HVX_ਸਦਿਸ਼ *)(xp + 64), *(ਅਚਲ HVX_ਸਦਿਸ਼ *)(wp + 64));
        HVX_ਸਦਿਸ਼ p3 = Q6_Vqf32_vmpy_VsfVsf(*(ਅਚਲ HVX_ਸਦਿਸ਼ *)(xp + 96), *(ਅਚਲ HVX_ਸਦਿਸ਼ *)(wp + 96));

        HVX_ਸਦਿਸ਼_Pair p01 = Q6_W_vdeal_VVR(p1, p0, -4);
        HVX_ਸਦਿਸ਼_Pair p23 = Q6_W_vdeal_VVR(p3, p2, -4);

        HVX_ਸਦਿਸ਼_Pair q02 = Q6_W_vdeal_VVR(Q6_V_lo_W(p23), Q6_V_lo_W(p01), -4);
        HVX_ਸਦਿਸ਼_Pair q13 = Q6_W_vdeal_VVR(Q6_V_hi_W(p23), Q6_V_hi_W(p01), -4);

        HVX_ਸਦਿਸ਼ a = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(q02), Q6_V_lo_W(q13));
        HVX_ਸਦਿਸ਼ b = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_hi_W(q02), Q6_V_hi_W(q13));

        *(HVX_ਸਦਿਸ਼ *)(ਬਾਹਰਲਾ + cb) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(a, b));
    }
}

// Transpose src0 for prefill: one 32-channel block {32, ncs} (VTCM) -> T {ncs, 32} (VTCM).
// One VTCM gather per output row: lane r picks channel cb+r, the region bound drops
// the lanes past the channel tail.
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_ssm_conv_transpose_ਖੰਡ(ਅਚਲ ਦਸ਼ਮਲਵ * raw_ਖੰਡ,
                                                ਦਸ਼ਮਲਵ *       T,
                                                uint32_t      ncs,
                                                uint32_t      cb_n,
                                                HVX_ਸਦਿਸ਼    vv) {
    ਅਚਲ size_t   ਅਧਾਰ = (size_t) raw_ਖੰਡ;
    ਅਚਲ uint32_t mu   = cb_n * ncs * ਆਕਾਰ(ਦਸ਼ਮਲਵ) - 1;

    ਲਈ (uint32_t t = 0; t < ncs; ++t) {
        Q6_vgather_ARMVw((HVX_ਸਦਿਸ਼ *) (T + (size_t) t * VLEN_FP32), ਅਧਾਰ + t * ਆਕਾਰ(ਦਸ਼ਮਲਵ), mu, vv);
    }
}

// Single-token decode worker (n_t == 1)
ਸਥਿਰ ਖਾਲੀ ssm_conv_thread_f32_decode(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    ਬਣਤਰ htp_ssm_conv_context *             scctx   = (ਬਣਤਰ htp_ssm_conv_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_ops_context *                  octx    = scctx->octx;
    ਅਚਲ ਬਣਤਰ htp_ssm_conv_kernel_params * kparams = scctx->kparams;

    ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ src0 = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ src1 = octx->src[1];
    ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ dst  = octx->dst;

    dma_ਕਤਾਰ * dma_q = octx->ctx->dma[ith];

    ਅਚਲ uint32_t d_conv  = kparams->d_conv;
    ਅਚਲ uint32_t d_inner = kparams->d_inner;
    ਅਚਲ uint32_t n_s     = kparams->n_s;

    ਅਚਲ uint32_t dr  = scctx->nrows_ਪ੍ਰਤੀ_thread;
    ਅਚਲ uint32_t ir0 = scctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + dr * ith;
    ਅਚਲ uint32_t ir1 = MIN(ir0 + dr, scctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + scctx->nrows);

    ਜੇ (ir0 >= ir1) {
        ਵਾਪਸ;
    }

    ਅਚਲ uint32_t d_inner_ਪ੍ਰਤੀ_thread = ir1 - ir0;
    ਅਚਲ uint32_t d_inner_stride     = hex_ਗੇੜਾ_ਉੱਪਰ(d_inner_ਪ੍ਰਤੀ_thread, VLEN_FP32);
    ਅਚਲ uint32_t d_inner_tile       = scctx->d_inner_tile;

    ਅਚਲ size_t src0_stride_inner_ਬਾਈਟਾਂ = src0->nb[1];
    ਅਚਲ size_t src0_stride_seq_ਬਾਈਟਾਂ   = src0->nb[2];
    ਅਚਲ size_t dst_stride_seq_ਬਾਈਟਾਂ    = dst->nb[2];

    uint8_t * src1_spad_ਅਧਾਰ = octx->src1_spad.ਅੰਕੜੇ + ith * octx->src1_spad.ਮਾਪ_ਪ੍ਰਤੀ_thread;
    uint8_t * src0_spad_ਅਧਾਰ = octx->src0_spad.ਅੰਕੜੇ + ith * octx->src0_spad.ਮਾਪ_ਪ੍ਰਤੀ_thread;
    uint8_t * dst_spad_ਅਧਾਰ  = octx->dst_spad.ਅੰਕੜੇ  + ith * octx->dst_spad.ਮਾਪ_ਪ੍ਰਤੀ_thread;

    ਅਚਲ size_t ਭਾਰ_ਬਾਈਟਾਂ    = (size_t) d_inner_ਪ੍ਰਤੀ_thread * d_conv * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ size_t ਭਾਰ_raw_ਮਾਪ = hex_ਗੇੜਾ_ਉੱਪਰ(ਭਾਰ_ਬਾਈਟਾਂ, 128);

    ਦਸ਼ਮਲਵ * src1_raw = (ਦਸ਼ਮਲਵ *) src1_spad_ਅਧਾਰ;
    ਦਸ਼ਮਲਵ * src1_T   = (ਦਸ਼ਮਲਵ *) (src1_spad_ਅਧਾਰ + ਭਾਰ_raw_ਮਾਪ);

    ਅਚਲ size_t src0_tile_raw_ਬਾਈਟਾਂ = hex_ਗੇੜਾ_ਉੱਪਰ(d_inner_tile * d_conv * ਆਕਾਰ(ਦਸ਼ਮਲਵ), 128);
    ਅਚਲ size_t dst_tile_ਬਾਈਟਾਂ      = hex_ਗੇੜਾ_ਉੱਪਰ(d_inner_tile * ਆਕਾਰ(ਦਸ਼ਮਲਵ), 128);

    ਦਸ਼ਮਲਵ * src0_tile_raw[2] = { (ਦਸ਼ਮਲਵ *) src0_spad_ਅਧਾਰ, (ਦਸ਼ਮਲਵ *) (src0_spad_ਅਧਾਰ + src0_tile_raw_ਬਾਈਟਾਂ) };
    ਦਸ਼ਮਲਵ * src0_T           = (ਦਸ਼ਮਲਵ *) (src0_spad_ਅਧਾਰ + 2 * src0_tile_raw_ਬਾਈਟਾਂ);

    ਦਸ਼ਮਲਵ * dst_tile[2] = { (ਦਸ਼ਮਲਵ *) dst_spad_ਅਧਾਰ, (ਦਸ਼ਮਲਵ *) (dst_spad_ਅਧਾਰ + dst_tile_ਬਾਈਟਾਂ) };

    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];

    // raw_dot keeps both operands in the DMA layout, so src1 needs no prep pass
    ਅਚਲ bool raw_dot = (d_conv == 4) && (d_inner_ਪ੍ਰਤੀ_thread % VLEN_FP32 == 0);

    ਅਚਲ uint32_t n_tiles   = (d_inner_ਪ੍ਰਤੀ_thread + d_inner_tile - 1) / d_inner_tile;
    ਅਚਲ uint32_t n_chunks  = n_s * n_tiles;
    ਅਚਲ size_t   ਪੰਕਤੀ_ਬਾਈਟਾਂ = d_conv * ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    uint32_t s_fetch        = 0;
    uint32_t tile_off_fetch = 0;

    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SSM_CONV_DECODE_ਧੱਕੋ_FETCH(c)                                                         \
        ਕਰੋ {                                                                                      \
            ਅਚਲ uint32_t   cur_tile_n = MIN(d_inner_tile, d_inner_ਪ੍ਰਤੀ_thread - tile_off_fetch); \
            ਅਚਲ dma_addr_t fetch_ddr  = src0->ਅੰਕੜੇ + s_fetch * src0_stride_seq_ਬਾਈਟਾਂ +          \
                                          (ir0 + tile_off_fetch) * src0_stride_inner_ਬਾਈਟਾਂ;       \
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,                                                                 \
                           dma_make_ਅੰਕੜੇ((uint8_t *) src0_tile_raw[(c) & 1], fetch_ddr),          \
                           ਪੰਕਤੀ_ਬਾਈਟਾਂ, src0_stride_inner_ਬਾਈਟਾਂ, ਪੰਕਤੀ_ਬਾਈਟਾਂ, cur_tile_n);            \
            tile_off_fetch += d_inner_tile;                                                       \
            ਜੇ (tile_off_fetch >= d_inner_ਪ੍ਰਤੀ_thread) {                                           \
                tile_off_fetch = 0;                                                               \
                s_fetch++;                                                                        \
            }                                                                                     \
        } ਜਦੋਂ (0)

    // Queue weights and initial input tiles together so DDR reads overlap
    ਅਚਲ dma_addr_t src1_ddr = src1->ਅੰਕੜੇ + ir0 * d_conv * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ((uint8_t *) src1_raw, src1_ddr), ਭਾਰ_ਬਾਈਟਾਂ, ਭਾਰ_ਬਾਈਟਾਂ, ਭਾਰ_ਬਾਈਟਾਂ, 1);

    SSM_CONV_DECODE_ਧੱਕੋ_FETCH(0);
    ਜੇ (n_chunks > 1) {
        SSM_CONV_DECODE_ਧੱਕੋ_FETCH(1);
    }

    dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);  // weights

    ਜੇ (!raw_dot) {
        // Unpack/transpose src1_raw into src1_T {d_conv, d_inner_stride}
        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);
        hvx_ssm_conv_unpack_ਮੰਜ਼ਿਲ_T(src1_raw, src1_T, d_inner_ਪ੍ਰਤੀ_thread, d_inner_stride, d_conv);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);
    }

    uint32_t i3       = 0;
    uint32_t tile_off = 0;

    ਲਈ (uint32_t c = 0; c < n_chunks; ++c) {
        ਅਚਲ uint32_t tile_n = MIN(d_inner_tile, d_inner_ਪ੍ਰਤੀ_thread - tile_off);

        ਜੇ (c >= 2) {
            dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);  // writeback of chunk c-2, frees dst_tile[c & 1]
        }
        dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);      // fetch chunk c

        ਦਸ਼ਮਲਵ * ਸੀਮਿਤ ਬਾਹਰਲਾ = dst_tile[c & 1];

        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i3);
        ਜੇ (raw_dot) {
            ਅਚਲ ਦਸ਼ਮਲਵ * xp = src0_tile_raw[c & 1];
            ਅਚਲ ਦਸ਼ਮਲਵ * wp = src1_raw + tile_off * 4;
            hvx_ssm_conv_decode_4(xp, wp, ਬਾਹਰਲਾ, tile_n);
        } ਨਹੀਂਤਾਂ {
            ਅਚਲ uint32_t tile_stride = hex_ਗੇੜਾ_ਉੱਪਰ(tile_n, VLEN_FP32);
            hvx_ssm_conv_unpack_ਮੰਜ਼ਿਲ_T(src0_tile_raw[c & 1], src0_T, tile_n, tile_stride, d_conv);

            ਲਈ (uint32_t cb = 0; cb < tile_n; cb += VLEN_FP32) {
                ਅਚਲ uint32_t cb_n = MIN(VLEN_FP32, tile_n - cb);
                HVX_ਸਦਿਸ਼ acc = hvx_vec_splat_f32(0.0f);
                ਲਈ (uint32_t j = 0; j < d_conv; ++j) {
                    HVX_ਸਦਿਸ਼ x = *(ਅਚਲ HVX_ਸਦਿਸ਼ *)(src0_T + j * tile_stride + cb);
                    HVX_ਸਦਿਸ਼ w = *(ਅਚਲ HVX_ਸਦਿਸ਼ *)(src1_T + j * d_inner_stride + tile_off + cb);
                    acc          = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(x, w));
                }
                HVX_ਸਦਿਸ਼ y = Q6_Vsf_equals_Vqf32(acc);
                ਜੇ (cb_n == VLEN_FP32) {
                    *(HVX_ਸਦਿਸ਼ *)(ਬਾਹਰਲਾ + cb) = y;
                } ਨਹੀਂਤਾਂ {
                    hvx_vec_store_u(ਬਾਹਰਲਾ + cb, cb_n * ਆਕਾਰ(ਦਸ਼ਮਲਵ), y);
                }
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i3);

        ਅਚਲ dma_addr_t dst_ddr        = dst->ਅੰਕੜੇ + i3 * dst_stride_seq_ਬਾਈਟਾਂ + (ir0 + tile_off) * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
        ਅਚਲ size_t     tile_ਬਾਹਰਲਾ_ਬਾਈਟਾਂ = (size_t) tile_n * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(dst_ddr, (uint8_t *) ਬਾਹਰਲਾ),
                       tile_ਬਾਹਰਲਾ_ਬਾਈਟਾਂ, tile_ਬਾਹਰਲਾ_ਬਾਈਟਾਂ, tile_ਬਾਹਰਲਾ_ਬਾਈਟਾਂ, 1);

        ਜੇ (c + 2 < n_chunks) {
            SSM_CONV_DECODE_ਧੱਕੋ_FETCH(c + 2);
        }

        tile_off += d_inner_tile;
        ਜੇ (tile_off >= d_inner_ਪ੍ਰਤੀ_thread) {
            tile_off = 0;
            i3++;
        }
    }

    ਲਈ (uint32_t k = MIN(n_chunks, 2); k > 0; --k) {
        dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);  // drain the last writebacks
    }

    #ਪਰਿਭਾਸ਼ਾ_ਰੱਦ SSM_CONV_DECODE_ਧੱਕੋ_FETCH

    FARF(ਉੱਚਾ_2, "ssm-conv-f32-decode %d/%d: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0], dst->ne[1],
         dst->ne[2], dst->ne[3]);
}

// Multi-token prefill worker (n_t > 1)
ਸਥਿਰ ਖਾਲੀ ssm_conv_thread_f32_prefill(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    ਬਣਤਰ htp_ssm_conv_context *             scctx   = (ਬਣਤਰ htp_ssm_conv_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_ops_context *                  octx    = scctx->octx;
    ਅਚਲ ਬਣਤਰ htp_ssm_conv_kernel_params * kparams = scctx->kparams;

    ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ src0 = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ src1 = octx->src[1];
    ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ dst  = octx->dst;

    dma_ਕਤਾਰ * dma_q = octx->ctx->dma[ith];

    ਅਚਲ uint32_t d_conv  = kparams->d_conv;
    ਅਚਲ uint32_t d_inner = kparams->d_inner;
    ਅਚਲ uint32_t n_t     = kparams->n_t;
    ਅਚਲ uint32_t n_s     = kparams->n_s;
    ਅਚਲ uint32_t ncs     = src0->ne[0];

    ਅਚਲ uint32_t dr  = scctx->nrows_ਪ੍ਰਤੀ_thread;
    ਅਚਲ uint32_t ir0 = scctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + dr * ith;
    ਅਚਲ uint32_t ir1 = MIN(ir0 + dr, scctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + scctx->nrows);

    ਜੇ (ir0 >= ir1) {
        ਵਾਪਸ;
    }

    ਅਚਲ uint32_t d_inner_ਪ੍ਰਤੀ_thread = ir1 - ir0;
    ਅਚਲ uint32_t d_inner_stride     = hex_ਗੇੜਾ_ਉੱਪਰ(d_inner_ਪ੍ਰਤੀ_thread, VLEN_FP32);
    ਅਚਲ uint32_t d_inner_tile       = scctx->d_inner_tile;

    ਅਚਲ size_t src0_stride_inner_ਬਾਈਟਾਂ = src0->nb[1];
    ਅਚਲ size_t src0_stride_seq_ਬਾਈਟਾਂ   = src0->nb[2];
    ਅਚਲ size_t dst_stride_token_ਬਾਈਟਾਂ  = dst->nb[1];
    ਅਚਲ size_t dst_stride_seq_ਬਾਈਟਾਂ    = dst->nb[2];

    uint8_t * src1_spad_ਅਧਾਰ = octx->src1_spad.ਅੰਕੜੇ + ith * octx->src1_spad.ਮਾਪ_ਪ੍ਰਤੀ_thread;
    uint8_t * src0_spad_ਅਧਾਰ = octx->src0_spad.ਅੰਕੜੇ + ith * octx->src0_spad.ਮਾਪ_ਪ੍ਰਤੀ_thread;
    uint8_t * dst_spad_ਅਧਾਰ  = octx->dst_spad.ਅੰਕੜੇ  + ith * octx->dst_spad.ਮਾਪ_ਪ੍ਰਤੀ_thread;

    ਅਚਲ size_t ਭਾਰ_ਬਾਈਟਾਂ    = (size_t) d_inner_ਪ੍ਰਤੀ_thread * d_conv * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ size_t ਭਾਰ_raw_ਮਾਪ = hex_ਗੇੜਾ_ਉੱਪਰ(ਭਾਰ_ਬਾਈਟਾਂ, 128);

    ਦਸ਼ਮਲਵ * src1_raw = (ਦਸ਼ਮਲਵ *) src1_spad_ਅਧਾਰ;
    ਦਸ਼ਮਲਵ * src1_T   = (ਦਸ਼ਮਲਵ *) (src1_spad_ਅਧਾਰ + ਭਾਰ_raw_ਮਾਪ);

    // src0 spad holds two raw tiles (fetch of tile n+1 overlaps compute of tile n) plus
    // the transposed block. dst spad holds two tiles so a writeback can stay in flight.
    ਅਚਲ size_t src0_tile_raw_ਬਾਈਟਾਂ = hex_ਗੇੜਾ_ਉੱਪਰ(d_inner_tile * ncs * ਆਕਾਰ(ਦਸ਼ਮਲਵ), 128);
    ਅਚਲ size_t dst_tile_ਬਾਈਟਾਂ      = hex_ਗੇੜਾ_ਉੱਪਰ(d_inner_tile * n_t * ਆਕਾਰ(ਦਸ਼ਮਲਵ), 128);

    ਦਸ਼ਮਲਵ * src0_tile_raw[2] = { (ਦਸ਼ਮਲਵ *) src0_spad_ਅਧਾਰ, (ਦਸ਼ਮਲਵ *) (src0_spad_ਅਧਾਰ + src0_tile_raw_ਬਾਈਟਾਂ) };
    ਦਸ਼ਮਲਵ * src0_T           = (ਦਸ਼ਮਲਵ *) (src0_spad_ਅਧਾਰ + 2 * src0_tile_raw_ਬਾਈਟਾਂ);

    ਦਸ਼ਮਲਵ * dst_tile[2] = { (ਦਸ਼ਮਲਵ *) dst_spad_ਅਧਾਰ, (ਦਸ਼ਮਲਵ *) (dst_spad_ਅਧਾਰ + dst_tile_ਬਾਈਟਾਂ) };

    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ਅਚਲ uint32_t n_tiles   = (d_inner_ਪ੍ਰਤੀ_thread + d_inner_tile - 1) / d_inner_tile;
    ਅਚਲ uint32_t n_chunks  = n_s * n_tiles;
    ਅਚਲ size_t   ਪੰਕਤੀ_ਬਾਈਟਾਂ = ncs * ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    uint32_t s_fetch        = 0;
    uint32_t tile_off_fetch = 0;

    // Chunk c fetches into src0_tile_raw[c & 1] and writes back from dst_tile[c & 1].
    // Two fetches run ahead, so the queue order is F0 F1 W0 F2 W1 ... and pops follow it.
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SSM_CONV_ਧੱਕੋ_FETCH(c)                                                                \
        ਕਰੋ {                                                                                      \
            ਅਚਲ uint32_t   cur_tile_n = MIN(d_inner_tile, d_inner_ਪ੍ਰਤੀ_thread - tile_off_fetch); \
            ਅਚਲ dma_addr_t fetch_ddr  = src0->ਅੰਕੜੇ + s_fetch * src0_stride_seq_ਬਾਈਟਾਂ +          \
                                          (ir0 + tile_off_fetch) * src0_stride_inner_ਬਾਈਟਾਂ;       \
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,                                                                 \
                           dma_make_ਅੰਕੜੇ((uint8_t *) src0_tile_raw[(c) & 1], fetch_ddr),          \
                           ਪੰਕਤੀ_ਬਾਈਟਾਂ, src0_stride_inner_ਬਾਈਟਾਂ, ਪੰਕਤੀ_ਬਾਈਟਾਂ, cur_tile_n);            \
            tile_off_fetch += d_inner_tile;                                                       \
            ਜੇ (tile_off_fetch >= d_inner_ਪ੍ਰਤੀ_thread) {                                           \
                tile_off_fetch = 0;                                                               \
                s_fetch++;                                                                        \
            }                                                                                     \
        } ਜਦੋਂ (0)

    // Queue weights and initial input tiles together so DDR reads overlap
    ਅਚਲ dma_addr_t src1_ddr = src1->ਅੰਕੜੇ + ir0 * d_conv * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ((uint8_t *) src1_raw, src1_ddr), ਭਾਰ_ਬਾਈਟਾਂ, ਭਾਰ_ਬਾਈਟਾਂ, ਭਾਰ_ਬਾਈਟਾਂ, 1);

    SSM_CONV_ਧੱਕੋ_FETCH(0);
    ਜੇ (n_chunks > 1) {
        SSM_CONV_ਧੱਕੋ_FETCH(1);
    }

    dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);  // weights

    // Unpack/transpose src1_raw into src1_T {d_conv, d_inner_stride}
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);
    hvx_ssm_conv_unpack_ਮੰਜ਼ਿਲ_T(src1_raw, src1_T, d_inner_ਪ੍ਰਤੀ_thread, d_inner_stride, d_conv);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);

    ਅਚਲ uint32_t C_TILE = VLEN_FP32;

    // gather offsets: lane r reads channel r of the raw tile block
    uint32_t __attribute__((aligned(VLEN))) gather_off[VLEN_FP32];
    ਲਈ (uint32_t r = 0; r < VLEN_FP32; ++r) {
        gather_off[r] = r * ncs * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    }
    ਅਚਲ HVX_ਸਦਿਸ਼ vv = *(ਅਚਲ HVX_ਸਦਿਸ਼ *) gather_off;

    uint32_t i3       = 0;
    uint32_t tile_off = 0;

    ਲਈ (uint32_t c = 0; c < n_chunks; ++c) {
        ਅਚਲ uint32_t tile_n = MIN(d_inner_tile, d_inner_ਪ੍ਰਤੀ_thread - tile_off);

        ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ raw = src0_tile_raw[c & 1];
        ਦਸ਼ਮਲਵ * ਸੀਮਿਤ       ਬਾਹਰਲਾ = dst_tile[c & 1];

        ਜੇ (c >= 2) {
            dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);  // writeback of chunk c-2, frees dst_tile[c & 1]
        }
        dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);      // fetch chunk c

        // Channel block outer, token inner: the taps and the sliding window stay in
        // registers, so a new output row costs one src0_T load.
        ਅਚਲ uint32_t dst_tile_stride = hex_ਗੇੜਾ_ਉੱਪਰ(tile_n, C_TILE);

        ਲਈ (uint32_t cb = 0; cb < tile_n; cb += C_TILE) {
            ਅਚਲ uint32_t cb_n = MIN(C_TILE, tile_n - cb);

            htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) tile_off);
            hvx_ssm_conv_transpose_ਖੰਡ(raw + (size_t) cb * ncs, src0_T, ncs, cb_n, vv);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) tile_off);

            ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ wp = src1_T + tile_off + cb;
            ਦਸ਼ਮਲਵ * ਸੀਮਿਤ       op = ਬਾਹਰਲਾ + cb;

            htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) tile_off);
            ਜੇ (d_conv == 4) {
                ਅਚਲ HVX_ਸਦਿਸ਼ w0 = *(ਅਚਲ HVX_ਸਦਿਸ਼ *) (wp);
                ਅਚਲ HVX_ਸਦਿਸ਼ w1 = *(ਅਚਲ HVX_ਸਦਿਸ਼ *) (wp + d_inner_stride);
                ਅਚਲ HVX_ਸਦਿਸ਼ w2 = *(ਅਚਲ HVX_ਸਦਿਸ਼ *) (wp + 2 * d_inner_stride);
                ਅਚਲ HVX_ਸਦਿਸ਼ w3 = *(ਅਚਲ HVX_ਸਦਿਸ਼ *) (wp + 3 * d_inner_stride);

                HVX_ਸਦਿਸ਼ x0 = *(ਅਚਲ HVX_ਸਦਿਸ਼ *) (src0_T);
                HVX_ਸਦਿਸ਼ x1 = *(ਅਚਲ HVX_ਸਦਿਸ਼ *) (src0_T + C_TILE);
                HVX_ਸਦਿਸ਼ x2 = *(ਅਚਲ HVX_ਸਦਿਸ਼ *) (src0_T + 2 * C_TILE);

                ਲਈ (uint32_t t = 0; t < n_t; ++t) {
                    ਅਚਲ HVX_ਸਦਿਸ਼ x3 = *(ਅਚਲ HVX_ਸਦਿਸ਼ *) (src0_T + (t + 3) * C_TILE);

                    HVX_ਸਦਿਸ਼ a = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(x0, w0),
                                                            Q6_Vqf32_vmpy_VsfVsf(x1, w1));
                    HVX_ਸਦਿਸ਼ b = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(x2, w2),
                                                            Q6_Vqf32_vmpy_VsfVsf(x3, w3));

                    *(HVX_ਸਦਿਸ਼ *) (op + t * dst_tile_stride) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(a, b));

                    x0 = x1;
                    x1 = x2;
                    x2 = x3;
                }
            } ਨਹੀਂਤਾਂ {
                ਲਈ (uint32_t t = 0; t < n_t; ++t) {
                    HVX_ਸਦਿਸ਼ acc = hvx_vec_splat_f32(0.0f);
                    ਲਈ (uint32_t j = 0; j < d_conv; ++j) {
                        HVX_ਸਦਿਸ਼ x = *(ਅਚਲ HVX_ਸਦਿਸ਼ *) (src0_T + (t + j) * C_TILE);
                        HVX_ਸਦਿਸ਼ w = *(ਅਚਲ HVX_ਸਦਿਸ਼ *) (wp + j * d_inner_stride);
                        acc          = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(x, w));
                    }
                    *(HVX_ਸਦਿਸ਼ *) (op + t * dst_tile_stride) = Q6_Vsf_equals_Vqf32(acc);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) tile_off);
        }

        // Writeback dst_tile VTCM -> DDR via 2D DMA
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,
                       dma_make_ਅੰਕੜੇ(dst->ਅੰਕੜੇ + i3 * dst_stride_seq_ਬਾਈਟਾਂ + (ir0 + tile_off) * ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                                     (uint8_t *) ਬਾਹਰਲਾ),
                       dst_stride_token_ਬਾਈਟਾਂ, dst_tile_stride * ਆਕਾਰ(ਦਸ਼ਮਲਵ), tile_n * ਆਕਾਰ(ਦਸ਼ਮਲਵ), n_t);

        ਜੇ (c + 2 < n_chunks) {
            SSM_CONV_ਧੱਕੋ_FETCH(c + 2);
        }

        tile_off += d_inner_tile;
        ਜੇ (tile_off >= d_inner_ਪ੍ਰਤੀ_thread) {
            tile_off = 0;
            i3++;
        }
    }

    ਲਈ (uint32_t k = MIN(n_chunks, 2); k > 0; --k) {
        dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);  // drain the last writebacks
    }

    #ਪਰਿਭਾਸ਼ਾ_ਰੱਦ SSM_CONV_ਧੱਕੋ_FETCH

    FARF(ਉੱਚਾ_2, "ssm-conv-f32-prefill %d/%d: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0], dst->ne[1],
         dst->ne[2], dst->ne[3]);
}

ਪੂਰਨ op_ssm_conv_f32(ਬਣਤਰ htp_ops_context * octx) {
    ਅਚਲ ਬਣਤਰ htp_tensor * src0 = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * src1 = octx->src[1];
    ਅਚਲ ਬਣਤਰ htp_tensor * dst  = octx->dst;

    ਜੇ (src0->type != HTP_TYPE_F32 || src1->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    ਅਚਲ ਬਣਤਰ htp_ssm_conv_kernel_params * kparams = (ਅਚਲ ਬਣਤਰ htp_ssm_conv_kernel_params *) octx->kernel_params;


    ਜੇ (!htp_ops_context_ਸੈੱਟ_n_ਧਾਗੇ(octx, kparams->n_ਧਾਗੇ)) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_INVAL_PARAMS;
    }

    uint32_t ਪੰਕਤੀ_ਸ਼ੁਰੂ = 0;
    uint32_t nrows     = kparams->d_inner;

    ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1) {
        ਅਚਲ uint32_t elems_ਪ੍ਰਤੀ_chunk = VLEN_FP32;
        ਅਚਲ ਬਣਤਰ htp_tensor_mdev_range range = htp_tensor_mdev_ਹਿੱਸੇ_ਕਰੋ(
            kparams->d_inner,
            htp_tensor_mdev_ਅੰਕੜੇ_aligned(dst) ? elems_ਪ੍ਰਤੀ_chunk : 0,
            octx->ctx->mdev.idx,
            octx->ctx->mdev.ਗਿਣਤੀ_2,
            &octx->ctx->mdev.ਗਿਣਤੀ_div
        );
        ਪੰਕਤੀ_ਸ਼ੁਰੂ = range.ਸ਼ੁਰੂ;
        nrows     = range.ਗਿਣਤੀ_2;
    }

    ਜੇ (nrows == 0) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
    }

    ਜੇ (kparams->vtcm_ਮਾਪ > octx->ctx->vtcm_ਮਾਪ) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_VTCM_TOO_SMALL;
    }

    ਅਚਲ uint32_t n_ਧਾਗੇ = octx->n_ਧਾਗੇ;

    octx->src0_spad.ਮਾਪ_ਪ੍ਰਤੀ_thread = kparams->vtcm_src0_ਮਾਪ_ਪ੍ਰਤੀ_thread;
    octx->src1_spad.ਮਾਪ_ਪ੍ਰਤੀ_thread = kparams->vtcm_src1_ਮਾਪ_ਪ੍ਰਤੀ_thread;
    octx->dst_spad.ਮਾਪ_ਪ੍ਰਤੀ_thread  = kparams->vtcm_dst_ਮਾਪ_ਪ੍ਰਤੀ_thread;

    octx->src0_spad.ਮਾਪ = kparams->vtcm_src0_ਮਾਪ;
    octx->src1_spad.ਮਾਪ = kparams->vtcm_src1_ਮਾਪ;
    octx->dst_spad.ਮਾਪ  = kparams->vtcm_dst_ਮਾਪ;

    octx->src0_spad.ਅੰਕੜੇ = octx->ctx->vtcm_ਅਧਾਰ;
    octx->src1_spad.ਅੰਕੜੇ = octx->src0_spad.ਅੰਕੜੇ + octx->src0_spad.ਮਾਪ;
    octx->dst_spad.ਅੰਕੜੇ  = octx->src1_spad.ਅੰਕੜੇ + octx->src1_spad.ਮਾਪ;
    octx->src0_spad.src  = ਨਿਰਾ;
    octx->src1_spad.src  = ਨਿਰਾ;
    octx->dst_spad.src   = ਨਿਰਾ;

    ਅਚਲ uint32_t raw_rpt            = fastdiv(nrows + n_ਧਾਗੇ - 1, &octx->n_ਧਾਗੇ_div);
    ਅਚਲ uint32_t d_inner_ਪ੍ਰਤੀ_thread = hex_ਗੇੜਾ_ਉੱਪਰ(raw_rpt, VLEN_FP32);

    ਬਣਤਰ htp_ssm_conv_context scctx = {
        .octx             = octx,
        .kparams          = kparams,
        .nrows_ਪ੍ਰਤੀ_thread = d_inner_ਪ੍ਰਤੀ_thread,
        .d_inner_tile     = kparams->d_inner_tile,
        .ਪੰਕਤੀ_ਸ਼ੁਰੂ        = ਪੰਕਤੀ_ਸ਼ੁਰੂ,
        .nrows            = nrows,
    };

    FARF(ਉੱਚਾ_2, "ssm-conv-f32: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : mode %s\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         kparams->n_t == 1 ? "decode" : "prefill");

    ਜੇ (kparams->n_t == 1) {
        work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, ssm_conv_thread_f32_decode, &scctx, n_ਧਾਗੇ);
    } ਨਹੀਂਤਾਂ {
        work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, ssm_conv_thread_f32_prefill, &scctx, n_ਧਾਗੇ);
    }

    ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
}

ਪੂਰਨ op_ssm_conv(ਬਣਤਰ htp_ops_context * octx) {
    ਅਚਲ ਬਣਤਰ htp_tensor * dst = octx->dst;

    ਚੋਣ (dst->type) {
        ਹਾਲਤ HTP_TYPE_F32:
            ਵਾਪਸ op_ssm_conv_f32(octx);
        ਮੂਲ:
            ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }
}
