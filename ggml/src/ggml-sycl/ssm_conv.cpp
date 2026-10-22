#ಅನ್ತರ್ಭಾವ "ssm_conv.hpp"
#ಅನ್ತರ್ಭಾವ "common.hpp"
#ಅನ್ತರ್ಭಾವ "element_wise.hpp"

#ಅನ್ತರ್ಭಾವ <cstdio>

ಬಳಸಿ ನಾಮಾಕಾಶ sycl;

// One output element of the conv. DC is d_conv as a compile-time constant (0 keeps the
// runtime loop); unfused callers pass literal false/nullptr so the epilogue folds away.
ಮಾದರಿ <ಪೂರ್ಣಾಂಕ DC>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ ssm_conv_element(
    size_t idx,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *src_ದತ್ತಾಂಶ,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *weights,
    ದಶಮಾಂಶ *dst_ದತ್ತಾಂಶ,
    ಪೂರ್ಣಾಂಕ d_conv,
    ಪೂರ್ಣಾಂಕ d_inner,
    ಪೂರ್ಣಾಂಕ n_t,
    ಪೂರ್ಣಾಂಕ src_stride_inner,
    ಪೂರ್ಣಾಂಕ src_stride_seq,
    ಪೂರ್ಣಾಂಕ dst_stride_token,
    ಪೂರ್ಣಾಂಕ dst_stride_seq,
    ತಾರ್ಕಿಕ apply_silu,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *bias
) {
    // src is token-contiguous per channel, dst is channel-contiguous per token,
    // so indexing token-fastest coalesces the d_conv loads.
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ token   = static_cast<ಪೂರ್ಣಾಂಕ>(idx % n_t);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ channel = static_cast<ಪೂರ್ಣಾಂಕ>((idx / n_t) % d_inner);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ seq     = static_cast<ಪೂರ್ಣಾಂಕ>(idx / (static_cast<size_t>(n_t) * static_cast<size_t>(d_inner)));

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *s = src_ದತ್ತಾಂಶ
        + static_cast<size_t>(seq) * static_cast<size_t>(src_stride_seq)
        + static_cast<size_t>(channel) * static_cast<size_t>(src_stride_inner)
        + static_cast<size_t>(token);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *c = weights + static_cast<size_t>(channel) * static_cast<size_t>(d_conv);

    ದಶಮಾಂಶ sumf = 0.0f;
    ಆದರೆ constexpr (DC > 0) {
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < DC; ++i0) {
            sumf += s[i0] * c[i0];
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < d_conv; ++i0) {
            sumf += s[i0] * c[i0];
        }
    }

    // fused bias add: the ADD node broadcasts a 1-D channel bias over tokens
    ಆದರೆ (bias != ಶೂನ್ಯಸೂಚಕ) {
        sumf += bias[channel];
    }

    ಸ್ಥಿರಾಂಕ size_t dst_idx =
        static_cast<size_t>(seq) * static_cast<size_t>(dst_stride_seq) +
        static_cast<size_t>(token) * static_cast<size_t>(dst_stride_token) +
        static_cast<size_t>(channel);

    dst_ದತ್ತಾಂಶ[dst_idx] = apply_silu ? op_silu(sumf) : sumf;
}

// FUSED=false keeps apply_silu/bias out of the kernel capture list, so the unfused launch
// takes the pre-fusion argument list; matters at n_t == 1, where the op is launch-bound.
ಮಾದರಿ <ಪೂರ್ಣಾಂಕ DC, ತಾರ್ಕಿಕ FUSED>
ಸ್ಥಿರ ಶೂನ್ಯ kernel_ssm_conv_impl(
    queue &q,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *src_ದತ್ತಾಂಶ,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *weights,
    ದಶಮಾಂಶ *dst_ದತ್ತಾಂಶ,
    ಪೂರ್ಣಾಂಕ d_conv,
    ಪೂರ್ಣಾಂಕ d_inner,
    ಪೂರ್ಣಾಂಕ n_t,
    ಪೂರ್ಣಾಂಕ n_s,
    ಪೂರ್ಣಾಂಕ ncs __attribute__((ಬಳಸದ_2)),
    ಪೂರ್ಣಾಂಕ src_stride_inner,
    ಪೂರ್ಣಾಂಕ src_stride_seq,
    ಪೂರ್ಣಾಂಕ dst_stride_token,
    ಪೂರ್ಣಾಂಕ dst_stride_seq,
    ತಾರ್ಕಿಕ apply_silu,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *bias
) {
    ಸ್ಥಿರಾಂಕ size_t ಒಟ್ಟು_work = static_cast<size_t>(d_inner) * static_cast<size_t>(n_t) * static_cast<size_t>(n_s);
    ಸ್ಥಿರಾಂಕ size_t work_ಗುಂಪು_ಗಾತ್ರ = 256;
    ಸ್ಥಿರಾಂಕ size_t ಅಂಕೆ_work_groups = (ಒಟ್ಟು_work + work_ಗುಂಪು_ಗಾತ್ರ - 1) / work_ಗುಂಪು_ಗಾತ್ರ;

    ಸ್ಥಿರಾಂಕ range<1> global_range(ಅಂಕೆ_work_groups * work_ಗುಂಪು_ಗಾತ್ರ);
    ಸ್ಥಿರಾಂಕ range<1> local_range(work_ಗುಂಪು_ಗಾತ್ರ);

    ಆದರೆ constexpr (FUSED) {
        q.submit([&](handler &h) {
            h.parallel_for(
                nd_range<1>(global_range, local_range),
                [=](nd_ಅಂಶ<1> ಅಂಶ) {
                    ಸ್ಥಿರಾಂಕ size_t idx = ಅಂಶ.get_global_id(0);
                    ಆದರೆ (idx >= ಒಟ್ಟು_work) {
                        ಹಿಂತಿರುಗಿ;
                    }

                    ssm_conv_element<DC>(idx, src_ದತ್ತಾಂಶ, weights, dst_ದತ್ತಾಂಶ, d_conv, d_inner, n_t,
                                         src_stride_inner, src_stride_seq, dst_stride_token,
                                         dst_stride_seq, apply_silu, bias);
                }
            );
        });
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_ಬಳಸದ(apply_silu);
        GGML_ಬಳಸದ(bias);

        q.submit([&](handler &h) {
            h.parallel_for(
                nd_range<1>(global_range, local_range),
                [=](nd_ಅಂಶ<1> ಅಂಶ) {
                    ಸ್ಥಿರಾಂಕ size_t idx = ಅಂಶ.get_global_id(0);
                    ಆದರೆ (idx >= ಒಟ್ಟು_work) {
                        ಹಿಂತಿರುಗಿ;
                    }

                    ssm_conv_element<DC>(idx, src_ದತ್ತಾಂಶ, weights, dst_ದತ್ತಾಂಶ, d_conv, d_inner, n_t,
                                         src_stride_inner, src_stride_seq, dst_stride_token,
                                         dst_stride_seq, ಸುಳ್ಳು, ಶೂನ್ಯಸೂಚಕ);
                }
            );
        });
    }
}

// SLM transpose tile: coalesces both the loads and the stores. The +1 pad makes the row
// stride 33, coprime with 32 banks, so both phases are bank-conflict-free.
ಮಾದರಿ <ಪೂರ್ಣಾಂಕ DC, ಪೂರ್ಣಾಂಕ TT, ಪೂರ್ಣಾಂಕ TC, ಪೂರ್ಣಾಂಕ WG>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ ssm_conv_tile(
    nd_ಅಂಶ<1> it, local_accessor<ದಶಮಾಂಶ, 1> tile, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *src_ದತ್ತಾಂಶ, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *weights,
    ದಶಮಾಂಶ *dst_ದತ್ತಾಂಶ, ಪೂರ್ಣಾಂಕ n_t, ಪೂರ್ಣಾಂಕ nt_tiles, ಪೂರ್ಣಾಂಕ nc_tiles, ಪೂರ್ಣಾಂಕ src_stride_inner,
    ಪೂರ್ಣಾಂಕ src_stride_seq, ಪೂರ್ಣಾಂಕ dst_stride_token, ಪೂರ್ಣಾಂಕ dst_stride_seq, ತಾರ್ಕಿಕ apply_silu,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *bias
) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ    lid = static_cast<ಪೂರ್ಣಾಂಕ>(it.get_local_id(0));
    ಸ್ಥಿರಾಂಕ size_t g   = it.get_ಗುಂಪು(0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ    tt  = static_cast<ಪೂರ್ಣಾಂಕ>(g % nt_tiles);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ    ct  = static_cast<ಪೂರ್ಣಾಂಕ>((g / nt_tiles) % nc_tiles);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ    seq = static_cast<ಪೂರ್ಣಾಂಕ>(g / (static_cast<size_t>(nt_tiles) * nc_tiles));
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ    t0 = tt * TT, c0 = ct * TC;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ti = lid % TT;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ cj = lid / TT;
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ r = 0; r < TC / (WG / TT); ++r) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ c   = cj + r * (WG / TT);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tok = t0 + ti;
        ದಶಮಾಂಶ sumf = 0.0f;
        ಆದರೆ (tok < n_t) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *s = src_ದತ್ತಾಂಶ + static_cast<size_t>(seq) * src_stride_seq
                           + static_cast<size_t>(c0 + c) * src_stride_inner + tok;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *cw = weights + static_cast<size_t>(c0 + c) * DC;
#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < DC; ++i) sumf += s[i] * cw[i];
            ಆದರೆ (bias != ಶೂನ್ಯಸೂಚಕ) sumf += bias[c0 + c];
            ಆದರೆ (apply_silu) sumf = op_silu(sumf);
        }
        tile[c * (TT + 1) + ti] = sumf;
    }
    it.barrier(access::fence_space::local_space);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ cc = lid % TC;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tj = lid / TC;
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ r = 0; r < TT / (WG / TC); ++r) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ t   = tj + r * (WG / TC);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tok = t0 + t;
        ಆದರೆ (tok < n_t) {
            dst_ದತ್ತಾಂಶ[static_cast<size_t>(seq) * dst_stride_seq
                     + static_cast<size_t>(tok) * dst_stride_token + c0 + cc]
                = tile[cc * (TT + 1) + t];
        }
    }
}

// Same FUSED split as kernel_ssm_conv_impl. The fused instantiation keeps the runtime
// apply_silu/bias branches: at n_t >= 32 they are amortized over the whole tile.
ಮಾದರಿ <ಪೂರ್ಣಾಂಕ DC, ತಾರ್ಕಿಕ FUSED>
ಸ್ಥಿರ ಶೂನ್ಯ kernel_ssm_conv_tiled(
    queue &q, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *src_ದತ್ತಾಂಶ, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *weights, ದಶಮಾಂಶ *dst_ದತ್ತಾಂಶ,
    ಪೂರ್ಣಾಂಕ d_inner, ಪೂರ್ಣಾಂಕ n_t, ಪೂರ್ಣಾಂಕ n_s, ಪೂರ್ಣಾಂಕ src_stride_inner, ಪೂರ್ಣಾಂಕ src_stride_seq,
    ಪೂರ್ಣಾಂಕ dst_stride_token, ಪೂರ್ಣಾಂಕ dst_stride_seq, ತಾರ್ಕಿಕ apply_silu, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *bias
) {
    constexpr ಪೂರ್ಣಾಂಕ TT = 32, TC = 32, WG = 256;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nt_tiles = (n_t + TT - 1) / TT;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nc_tiles = d_inner / TC;
    ಸ್ಥಿರಾಂಕ size_t groups = static_cast<size_t>(nt_tiles) * nc_tiles * n_s;

    ಆದರೆ constexpr (FUSED) {
        q.submit([&](handler &h) {
            local_accessor<ದಶಮಾಂಶ, 1> tile(range<1>(TC * (TT + 1)), h);
            h.parallel_for(nd_range<1>(range<1>(groups * WG), range<1>(WG)), [=](nd_ಅಂಶ<1> it) {
                ssm_conv_tile<DC, TT, TC, WG>(it, tile, src_ದತ್ತಾಂಶ, weights, dst_ದತ್ತಾಂಶ, n_t, nt_tiles,
                                              nc_tiles, src_stride_inner, src_stride_seq,
                                              dst_stride_token, dst_stride_seq, apply_silu, bias);
            });
        });
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_ಬಳಸದ(apply_silu);
        GGML_ಬಳಸದ(bias);

        q.submit([&](handler &h) {
            local_accessor<ದಶಮಾಂಶ, 1> tile(range<1>(TC * (TT + 1)), h);
            h.parallel_for(nd_range<1>(range<1>(groups * WG), range<1>(WG)), [=](nd_ಅಂಶ<1> it) {
                ssm_conv_tile<DC, TT, TC, WG>(it, tile, src_ದತ್ತಾಂಶ, weights, dst_ದತ್ತಾಂಶ, n_t, nt_tiles,
                                              nc_tiles, src_stride_inner, src_stride_seq,
                                              dst_stride_token, dst_stride_seq, ಸುಳ್ಳು, ಶೂನ್ಯಸೂಚಕ);
            });
        });
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ kernel_ssm_conv(
    queue &q,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *src_ದತ್ತಾಂಶ,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *weights,
    ದಶಮಾಂಶ *dst_ದತ್ತಾಂಶ,
    ಪೂರ್ಣಾಂಕ d_conv,
    ಪೂರ್ಣಾಂಕ d_inner,
    ಪೂರ್ಣಾಂಕ n_t,
    ಪೂರ್ಣಾಂಕ n_s,
    ಪೂರ್ಣಾಂಕ ncs,
    ಪೂರ್ಣಾಂಕ src_stride_inner,
    ಪೂರ್ಣಾಂಕ src_stride_seq,
    ಪೂರ್ಣಾಂಕ dst_stride_token,
    ಪೂರ್ಣಾಂಕ dst_stride_seq,
    ತಾರ್ಕಿಕ apply_silu,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *bias
) {
    // Only the fused instantiations carry apply_silu/bias as kernel arguments; the plain
    // ssm_conv launch keeps the argument list it had before the fusion landed.
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ fused = apply_silu || bias != ಶೂನ್ಯಸೂಚಕ;

    // d_inner must be a multiple of 32 so the channel tiles are exact; the transpose is only
    // worth it for n_t >= 32. d_conv == 4 is the only window with a DC-specialized kernel.
    ಆದರೆ (d_conv == 4 && n_t >= 32 && (d_inner % 32) == 0) {
        ಆದರೆ (fused) {
            kernel_ssm_conv_tiled<4, ನಿಜ>(q, src_ದತ್ತಾಂಶ, weights, dst_ದತ್ತಾಂಶ, d_inner, n_t, n_s,
                                           src_stride_inner, src_stride_seq, dst_stride_token,
                                           dst_stride_seq, apply_silu, bias);
        } ಇಲ್ಲದಿದ್ದರೆ {
            kernel_ssm_conv_tiled<4, ಸುಳ್ಳು>(q, src_ದತ್ತಾಂಶ, weights, dst_ದತ್ತಾಂಶ, d_inner, n_t, n_s,
                                            src_stride_inner, src_stride_seq, dst_stride_token,
                                            dst_stride_seq, apply_silu, bias);
        }
        ಹಿಂತಿರುಗಿ;
    }

    ಆದರೆ (d_conv == 4) {
        ಆದರೆ (fused) {
            kernel_ssm_conv_impl<4, ನಿಜ>(q, src_ದತ್ತಾಂಶ, weights, dst_ದತ್ತಾಂಶ, d_conv, d_inner, n_t, n_s,
                                          ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                          dst_stride_seq, apply_silu, bias);
        } ಇಲ್ಲದಿದ್ದರೆ {
            kernel_ssm_conv_impl<4, ಸುಳ್ಳು>(q, src_ದತ್ತಾಂಶ, weights, dst_ದತ್ತಾಂಶ, d_conv, d_inner, n_t, n_s,
                                           ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                           dst_stride_seq, apply_silu, bias);
        }
        ಹಿಂತಿರುಗಿ;
    }

    ಆದರೆ (fused) {
        kernel_ssm_conv_impl<0, ನಿಜ>(q, src_ದತ್ತಾಂಶ, weights, dst_ದತ್ತಾಂಶ, d_conv, d_inner, n_t, n_s,
                                      ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                      dst_stride_seq, apply_silu, bias);
    } ಇಲ್ಲದಿದ್ದರೆ {
        kernel_ssm_conv_impl<0, ಸುಳ್ಳು>(q, src_ದತ್ತಾಂಶ, weights, dst_ದತ್ತಾಂಶ, d_conv, d_inner, n_t, n_s,
                                       ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                       dst_stride_seq, apply_silu, bias);
    }
}

ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * silu_dst = ಶೂನ್ಯಸೂಚಕ, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * bias = ಶೂನ್ಯಸೂಚಕ) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(bias == ಶೂನ್ಯಸೂಚಕ || silu_dst != ಶೂನ್ಯಸೂಚಕ);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ d_conv   = src1->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncs      = src0->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ d_inner  = src0->ne[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_t      = dst->ne[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_s      = dst->ne[2];

    GGML_ASSERT(src0->ne[0] == d_conv - 1 + n_t);
    GGML_ASSERT(src0->ne[1] == d_inner);
    GGML_ASSERT(src1->ne[1] == d_inner);

    GGML_ASSERT(dst->ne[0] == d_inner);
    GGML_ASSERT(dst->ne[1] == n_t);
    GGML_ASSERT(dst->ne[2] == n_s);

    GGML_ASSERT(src0->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ));
    GGML_ASSERT(src1->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ));

    GGML_ASSERT(src0->nb[1] == src0->ne[0] * ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ src_stride_inner = ncs;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ src_stride_seq   = ncs * d_inner;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dst_stride_token = d_inner;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dst_stride_seq   = d_inner * n_t;

    ಪ್ರಯತ್ನಿಸು {
        queue *q = ctx.stream();

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *src_ದತ್ತಾಂಶ = static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(src0->ದತ್ತಾಂಶ);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *weights  = static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(src1->ದತ್ತಾಂಶ);
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ apply_silu = silu_dst != ಶೂನ್ಯಸೂಚಕ;
        ದಶಮಾಂಶ *dst_ದತ್ತಾಂಶ       = static_cast<ದಶಮಾಂಶ *>((silu_dst ? silu_dst : dst)->ದತ್ತಾಂಶ);

        GGML_ASSERT(src_ದತ್ತಾಂಶ && weights && dst_ದತ್ತಾಂಶ);

        kernel_ssm_conv(
            *q,
            src_ದತ್ತಾಂಶ,
            weights,
            dst_ದತ್ತಾಂಶ,
            d_conv,
            d_inner,
            n_t,
            n_s,
            ncs,
            src_stride_inner,
            src_stride_seq,
            dst_stride_token,
            dst_stride_seq,
            apply_silu,
            bias
        );

    } ಹಿಡಿ (ಸ್ಥಿರಾಂಕ std::exception &e) {
        std::fprintf(stderr, "[SYCL-SSM_CONV] ERROR: %s\n", e.what());
        ಎಸೆ;
    }
}

ಶೂನ್ಯ ggml_sycl_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_ssm_conv(ctx, dst);
}

// Fused ssm_conv + ADD + SiLU: write silu(conv(x) + b) straight into silu_dst, eliding the
// standalone SiLU launch and its HBM round-trip of the conv output.
ಶೂನ್ಯ ggml_sycl_ssm_conv_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * ಕೂಡಿಸು_2, ggml_tensor * silu_dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    GGML_ASSERT(silu_dst && ggml_are_same_shape(dst, silu_dst) && silu_dst->type == GGML_TYPE_F32);
    // the fused kernel reads only the ADD's bias operand; the ADD result is never written
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * bias = ಶೂನ್ಯಸೂಚಕ;
    ಆದರೆ (ಕೂಡಿಸು_2 != ಶೂನ್ಯಸೂಚಕ) {
        ಸ್ಥಿರಾಂಕ ggml_tensor * bias_t = (ಕೂಡಿಸು_2->src[0] == dst) ? ಕೂಡಿಸು_2->src[1] : ಕೂಡಿಸು_2->src[0];
        bias = static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(bias_t->ದತ್ತಾಂಶ);
    }
    ggml_sycl_op_ssm_conv(ctx, dst, silu_dst, bias);
}
