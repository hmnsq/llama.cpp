#ಅನ್ತರ್ಭಾವ "ssm_conv.hpp"
#ಅನ್ತರ್ಭಾವ "common.hpp"
#ಅನ್ತರ್ಭಾವ "element_wise.hpp"

#ಅನ್ತರ್ಭಾವ <cstdio>

using ನಾಮಾಕಾಶ sycl;

// One output element of the conv. DC is d_conv as a compile-time constant (0 keeps the
// runtime loop); unfused callers pass literal false/nullptr so the epilogue folds away.
template <ಪೂರ್ಣಾಙ್ಕ DC>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ ssm_conv_element(
    size_t idx,
    ಧ್ರುವ ಭಿನ್ನ *src_data,
    ಧ್ರುವ ಭಿನ್ನ *weights,
    ಭಿನ್ನ *dst_data,
    ಪೂರ್ಣಾಙ್ಕ d_conv,
    ಪೂರ್ಣಾಙ್ಕ d_inner,
    ಪೂರ್ಣಾಙ್ಕ n_t,
    ಪೂರ್ಣಾಙ್ಕ src_stride_inner,
    ಪೂರ್ಣಾಙ್ಕ src_stride_seq,
    ಪೂರ್ಣಾಙ್ಕ dst_stride_token,
    ಪೂರ್ಣಾಙ್ಕ dst_stride_seq,
    ತಾರ್ಕಿಕ apply_silu,
    ಧ್ರುವ ಭಿನ್ನ *bias
) {
    // src is token-contiguous per channel, dst is channel-contiguous per token,
    // so indexing token-fastest coalesces the d_conv loads.
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ token   = static_cast<ಪೂರ್ಣಾಙ್ಕ>(idx % n_t);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ channel = static_cast<ಪೂರ್ಣಾಙ್ಕ>((idx / n_t) % d_inner);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ seq     = static_cast<ಪೂರ್ಣಾಙ್ಕ>(idx / (static_cast<size_t>(n_t) * static_cast<size_t>(d_inner)));

    ಧ್ರುವ ಭಿನ್ನ *s = src_data
        + static_cast<size_t>(seq) * static_cast<size_t>(src_stride_seq)
        + static_cast<size_t>(channel) * static_cast<size_t>(src_stride_inner)
        + static_cast<size_t>(token);

    ಧ್ರುವ ಭಿನ್ನ *c = weights + static_cast<size_t>(channel) * static_cast<size_t>(d_conv);

    ಭಿನ್ನ sumf = 0.0f;
    ಯದಿ constexpr (DC > 0) {
#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i0 = 0; i0 < DC; ++i0) {
            sumf += s[i0] * c[i0];
        }
    } ಅನ್ಯಥಾ {
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i0 = 0; i0 < d_conv; ++i0) {
            sumf += s[i0] * c[i0];
        }
    }

    // fused bias add: the ADD node broadcasts a 1-D channel bias over tokens
    ಯದಿ (bias != nullptr) {
        sumf += bias[channel];
    }

    ಧ್ರುವ size_t dst_idx =
        static_cast<size_t>(seq) * static_cast<size_t>(dst_stride_seq) +
        static_cast<size_t>(token) * static_cast<size_t>(dst_stride_token) +
        static_cast<size_t>(channel);

    dst_data[dst_idx] = apply_silu ? op_silu(sumf) : sumf;
}

// FUSED=false keeps apply_silu/bias out of the kernel capture list, so the unfused launch
// takes the pre-fusion argument list; matters at n_t == 1, where the op is launch-bound.
template <ಪೂರ್ಣಾಙ್ಕ DC, ತಾರ್ಕಿಕ FUSED>
ಸ್ಥಿರ ಶೂನ್ಯ kernel_ssm_conv_impl(
    queue &q,
    ಧ್ರುವ ಭಿನ್ನ *src_data,
    ಧ್ರುವ ಭಿನ್ನ *weights,
    ಭಿನ್ನ *dst_data,
    ಪೂರ್ಣಾಙ್ಕ d_conv,
    ಪೂರ್ಣಾಙ್ಕ d_inner,
    ಪೂರ್ಣಾಙ್ಕ n_t,
    ಪೂರ್ಣಾಙ್ಕ n_s,
    ಪೂರ್ಣಾಙ್ಕ ncs __attribute__((unused)),
    ಪೂರ್ಣಾಙ್ಕ src_stride_inner,
    ಪೂರ್ಣಾಙ್ಕ src_stride_seq,
    ಪೂರ್ಣಾಙ್ಕ dst_stride_token,
    ಪೂರ್ಣಾಙ್ಕ dst_stride_seq,
    ತಾರ್ಕಿಕ apply_silu,
    ಧ್ರುವ ಭಿನ್ನ *bias
) {
    ಧ್ರುವ size_t total_work = static_cast<size_t>(d_inner) * static_cast<size_t>(n_t) * static_cast<size_t>(n_s);
    ಧ್ರುವ size_t work_group_size = 256;
    ಧ್ರುವ size_t ಅಂಕೆ_work_groups = (total_work + work_group_size - 1) / work_group_size;

    ಧ್ರುವ range<1> global_range(ಅಂಕೆ_work_groups * work_group_size);
    ಧ್ರುವ range<1> local_range(work_group_size);

    ಯದಿ constexpr (FUSED) {
        q.submit([&](handler &h) {
            h.parallel_for(
                nd_range<1>(global_range, local_range),
                [=](nd_item<1> item) {
                    ಧ್ರುವ size_t idx = item.get_global_id(0);
                    ಯದಿ (idx >= total_work) {
                        ಪ್ರತ್ಯಾಗಮ;
                    }

                    ssm_conv_element<DC>(idx, src_data, weights, dst_data, d_conv, d_inner, n_t,
                                         src_stride_inner, src_stride_seq, dst_stride_token,
                                         dst_stride_seq, apply_silu, bias);
                }
            );
        });
    } ಅನ್ಯಥಾ {
        GGML_UNUSED(apply_silu);
        GGML_UNUSED(bias);

        q.submit([&](handler &h) {
            h.parallel_for(
                nd_range<1>(global_range, local_range),
                [=](nd_item<1> item) {
                    ಧ್ರುವ size_t idx = item.get_global_id(0);
                    ಯದಿ (idx >= total_work) {
                        ಪ್ರತ್ಯಾಗಮ;
                    }

                    ssm_conv_element<DC>(idx, src_data, weights, dst_data, d_conv, d_inner, n_t,
                                         src_stride_inner, src_stride_seq, dst_stride_token,
                                         dst_stride_seq, ಅಸತ್ಯ, nullptr);
                }
            );
        });
    }
}

// SLM transpose tile: coalesces both the loads and the stores. The +1 pad makes the row
// stride 33, coprime with 32 banks, so both phases are bank-conflict-free.
template <ಪೂರ್ಣಾಙ್ಕ DC, ಪೂರ್ಣಾಙ್ಕ TT, ಪೂರ್ಣಾಙ್ಕ TC, ಪೂರ್ಣಾಙ್ಕ WG>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ ssm_conv_tile(
    nd_item<1> it, local_accessor<ಭಿನ್ನ, 1> tile, ಧ್ರುವ ಭಿನ್ನ *src_data, ಧ್ರುವ ಭಿನ್ನ *weights,
    ಭಿನ್ನ *dst_data, ಪೂರ್ಣಾಙ್ಕ n_t, ಪೂರ್ಣಾಙ್ಕ nt_tiles, ಪೂರ್ಣಾಙ್ಕ nc_tiles, ಪೂರ್ಣಾಙ್ಕ src_stride_inner,
    ಪೂರ್ಣಾಙ್ಕ src_stride_seq, ಪೂರ್ಣಾಙ್ಕ dst_stride_token, ಪೂರ್ಣಾಙ್ಕ dst_stride_seq, ತಾರ್ಕಿಕ apply_silu,
    ಧ್ರುವ ಭಿನ್ನ *bias
) {
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ    lid = static_cast<ಪೂರ್ಣಾಙ್ಕ>(it.get_local_id(0));
    ಧ್ರುವ size_t g   = it.get_group(0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ    tt  = static_cast<ಪೂರ್ಣಾಙ್ಕ>(g % nt_tiles);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ    ct  = static_cast<ಪೂರ್ಣಾಙ್ಕ>((g / nt_tiles) % nc_tiles);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ    seq = static_cast<ಪೂರ್ಣಾಙ್ಕ>(g / (static_cast<size_t>(nt_tiles) * nc_tiles));
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ    t0 = tt * TT, c0 = ct * TC;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ti = lid % TT;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ cj = lid / TT;
#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ r = 0; r < TC / (WG / TT); ++r) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ c   = cj + r * (WG / TT);
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ tok = t0 + ti;
        ಭಿನ್ನ sumf = 0.0f;
        ಯದಿ (tok < n_t) {
            ಧ್ರುವ ಭಿನ್ನ *s = src_data + static_cast<size_t>(seq) * src_stride_seq
                           + static_cast<size_t>(c0 + c) * src_stride_inner + tok;
            ಧ್ರುವ ಭಿನ್ನ *cw = weights + static_cast<size_t>(c0 + c) * DC;
#pragma unroll
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < DC; ++i) sumf += s[i] * cw[i];
            ಯದಿ (bias != nullptr) sumf += bias[c0 + c];
            ಯದಿ (apply_silu) sumf = op_silu(sumf);
        }
        tile[c * (TT + 1) + ti] = sumf;
    }
    it.barrier(access::fence_space::local_space);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ cc = lid % TC;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ tj = lid / TC;
#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ r = 0; r < TT / (WG / TC); ++r) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ t   = tj + r * (WG / TC);
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ tok = t0 + t;
        ಯದಿ (tok < n_t) {
            dst_data[static_cast<size_t>(seq) * dst_stride_seq
                     + static_cast<size_t>(tok) * dst_stride_token + c0 + cc]
                = tile[cc * (TT + 1) + t];
        }
    }
}

// Same FUSED split as kernel_ssm_conv_impl. The fused instantiation keeps the runtime
// apply_silu/bias branches: at n_t >= 32 they are amortized over the whole tile.
template <ಪೂರ್ಣಾಙ್ಕ DC, ತಾರ್ಕಿಕ FUSED>
ಸ್ಥಿರ ಶೂನ್ಯ kernel_ssm_conv_tiled(
    queue &q, ಧ್ರುವ ಭಿನ್ನ *src_data, ಧ್ರುವ ಭಿನ್ನ *weights, ಭಿನ್ನ *dst_data,
    ಪೂರ್ಣಾಙ್ಕ d_inner, ಪೂರ್ಣಾಙ್ಕ n_t, ಪೂರ್ಣಾಙ್ಕ n_s, ಪೂರ್ಣಾಙ್ಕ src_stride_inner, ಪೂರ್ಣಾಙ್ಕ src_stride_seq,
    ಪೂರ್ಣಾಙ್ಕ dst_stride_token, ಪೂರ್ಣಾಙ್ಕ dst_stride_seq, ತಾರ್ಕಿಕ apply_silu, ಧ್ರುವ ಭಿನ್ನ *bias
) {
    constexpr ಪೂರ್ಣಾಙ್ಕ TT = 32, TC = 32, WG = 256;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nt_tiles = (n_t + TT - 1) / TT;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nc_tiles = d_inner / TC;
    ಧ್ರುವ size_t groups = static_cast<size_t>(nt_tiles) * nc_tiles * n_s;

    ಯದಿ constexpr (FUSED) {
        q.submit([&](handler &h) {
            local_accessor<ಭಿನ್ನ, 1> tile(range<1>(TC * (TT + 1)), h);
            h.parallel_for(nd_range<1>(range<1>(groups * WG), range<1>(WG)), [=](nd_item<1> it) {
                ssm_conv_tile<DC, TT, TC, WG>(it, tile, src_data, weights, dst_data, n_t, nt_tiles,
                                              nc_tiles, src_stride_inner, src_stride_seq,
                                              dst_stride_token, dst_stride_seq, apply_silu, bias);
            });
        });
    } ಅನ್ಯಥಾ {
        GGML_UNUSED(apply_silu);
        GGML_UNUSED(bias);

        q.submit([&](handler &h) {
            local_accessor<ಭಿನ್ನ, 1> tile(range<1>(TC * (TT + 1)), h);
            h.parallel_for(nd_range<1>(range<1>(groups * WG), range<1>(WG)), [=](nd_item<1> it) {
                ssm_conv_tile<DC, TT, TC, WG>(it, tile, src_data, weights, dst_data, n_t, nt_tiles,
                                              nc_tiles, src_stride_inner, src_stride_seq,
                                              dst_stride_token, dst_stride_seq, ಅಸತ್ಯ, nullptr);
            });
        });
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ kernel_ssm_conv(
    queue &q,
    ಧ್ರುವ ಭಿನ್ನ *src_data,
    ಧ್ರುವ ಭಿನ್ನ *weights,
    ಭಿನ್ನ *dst_data,
    ಪೂರ್ಣಾಙ್ಕ d_conv,
    ಪೂರ್ಣಾಙ್ಕ d_inner,
    ಪೂರ್ಣಾಙ್ಕ n_t,
    ಪೂರ್ಣಾಙ್ಕ n_s,
    ಪೂರ್ಣಾಙ್ಕ ncs,
    ಪೂರ್ಣಾಙ್ಕ src_stride_inner,
    ಪೂರ್ಣಾಙ್ಕ src_stride_seq,
    ಪೂರ್ಣಾಙ್ಕ dst_stride_token,
    ಪೂರ್ಣಾಙ್ಕ dst_stride_seq,
    ತಾರ್ಕಿಕ apply_silu,
    ಧ್ರುವ ಭಿನ್ನ *bias
) {
    // Only the fused instantiations carry apply_silu/bias as kernel arguments; the plain
    // ssm_conv launch keeps the argument list it had before the fusion landed.
    ಧ್ರುವ ತಾರ್ಕಿಕ fused = apply_silu || bias != nullptr;

    // d_inner must be a multiple of 32 so the channel tiles are exact; the transpose is only
    // worth it for n_t >= 32. d_conv == 4 is the only window with a DC-specialized kernel.
    ಯದಿ (d_conv == 4 && n_t >= 32 && (d_inner % 32) == 0) {
        ಯದಿ (fused) {
            kernel_ssm_conv_tiled<4, ಸತ್ಯ>(q, src_data, weights, dst_data, d_inner, n_t, n_s,
                                           src_stride_inner, src_stride_seq, dst_stride_token,
                                           dst_stride_seq, apply_silu, bias);
        } ಅನ್ಯಥಾ {
            kernel_ssm_conv_tiled<4, ಅಸತ್ಯ>(q, src_data, weights, dst_data, d_inner, n_t, n_s,
                                            src_stride_inner, src_stride_seq, dst_stride_token,
                                            dst_stride_seq, apply_silu, bias);
        }
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಯದಿ (d_conv == 4) {
        ಯದಿ (fused) {
            kernel_ssm_conv_impl<4, ಸತ್ಯ>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                          ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                          dst_stride_seq, apply_silu, bias);
        } ಅನ್ಯಥಾ {
            kernel_ssm_conv_impl<4, ಅಸತ್ಯ>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                           ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                           dst_stride_seq, apply_silu, bias);
        }
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಯದಿ (fused) {
        kernel_ssm_conv_impl<0, ಸತ್ಯ>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                      ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                      dst_stride_seq, apply_silu, bias);
    } ಅನ್ಯಥಾ {
        kernel_ssm_conv_impl<0, ಅಸತ್ಯ>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                       ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                       dst_stride_seq, apply_silu, bias);
    }
}

ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_sycl_op_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * silu_dst = nullptr, ಧ್ರುವ ಭಿನ್ನ * bias = nullptr) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(bias == nullptr || silu_dst != nullptr);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ d_conv   = src1->ne[0];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ncs      = src0->ne[0];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ d_inner  = src0->ne[1];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_t      = dst->ne[1];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_s      = dst->ne[2];

    GGML_ASSERT(src0->ne[0] == d_conv - 1 + n_t);
    GGML_ASSERT(src0->ne[1] == d_inner);
    GGML_ASSERT(src1->ne[1] == d_inner);

    GGML_ASSERT(dst->ne[0] == d_inner);
    GGML_ASSERT(dst->ne[1] == n_t);
    GGML_ASSERT(dst->ne[2] == n_s);

    GGML_ASSERT(src0->nb[0] == ಪರಿಮಾಣ(ಭಿನ್ನ));
    GGML_ASSERT(src1->nb[0] == ಪರಿಮಾಣ(ಭಿನ್ನ));

    GGML_ASSERT(src0->nb[1] == src0->ne[0] * ಪರಿಮಾಣ(ಭಿನ್ನ));

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ src_stride_inner = ncs;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ src_stride_seq   = ncs * d_inner;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ dst_stride_token = d_inner;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ dst_stride_seq   = d_inner * n_t;

    ಯತ್ನ {
        queue *q = ctx.stream();

        ಧ್ರುವ ಭಿನ್ನ *src_data = static_cast<ಧ್ರುವ ಭಿನ್ನ *>(src0->data);
        ಧ್ರುವ ಭಿನ್ನ *weights  = static_cast<ಧ್ರುವ ಭಿನ್ನ *>(src1->data);
        ಧ್ರುವ ತಾರ್ಕಿಕ apply_silu = silu_dst != nullptr;
        ಭಿನ್ನ *dst_data       = static_cast<ಭಿನ್ನ *>((silu_dst ? silu_dst : dst)->data);

        GGML_ASSERT(src_data && weights && dst_data);

        kernel_ssm_conv(
            *q,
            src_data,
            weights,
            dst_data,
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

    } ಗೃಹಾಣ (ಧ್ರುವ std::exception &e) {
        std::fprintf(stderr, "[SYCL-SSM_CONV] ERROR: %s\n", e.what());
        ಕ್ಷಿಪ;
    }
}

ಶೂನ್ಯ ggml_sycl_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_ssm_conv(ctx, dst);
}

// Fused ssm_conv + ADD + SiLU: write silu(conv(x) + b) straight into silu_dst, eliding the
// standalone SiLU launch and its HBM round-trip of the conv output.
ಶೂನ್ಯ ggml_sycl_ssm_conv_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * add, ggml_tensor * silu_dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    GGML_ASSERT(silu_dst && ggml_are_same_shape(dst, silu_dst) && silu_dst->type == GGML_TYPE_F32);
    // the fused kernel reads only the ADD's bias operand; the ADD result is never written
    ಧ್ರುವ ಭಿನ್ನ * bias = nullptr;
    ಯದಿ (add != nullptr) {
        ಧ್ರುವ ggml_tensor * bias_t = (add->src[0] == dst) ? add->src[1] : add->src[0];
        bias = static_cast<ಧ್ರುವ ಭಿನ್ನ *>(bias_t->data);
    }
    ggml_sycl_op_ssm_conv(ctx, dst, silu_dst, bias);
}
