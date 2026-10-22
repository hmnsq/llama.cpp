#അന്തര്ഭാവ "ssm_conv.hpp"
#അന്തര്ഭാവ "common.hpp"
#അന്തര്ഭാവ "element_wise.hpp"

#അന്തര്ഭാവ <cstdio>

using നാമാകാശ sycl;

// One output element of the conv. DC is d_conv as a compile-time constant (0 keeps the
// runtime loop); unfused callers pass literal false/nullptr so the epilogue folds away.
template <പൂർണ്ണസംഖ്യ DC>
സ്ഥിതം __dpct_inline__ ശൂന്യം ssm_conv_element(
    size_t idx,
    സ്ഥിരം ദശാംശം *src_data,
    സ്ഥിരം ദശാംശം *weights,
    ദശാംശം *dst_data,
    പൂർണ്ണസംഖ്യ d_conv,
    പൂർണ്ണസംഖ്യ d_inner,
    പൂർണ്ണസംഖ്യ n_t,
    പൂർണ്ണസംഖ്യ src_stride_inner,
    പൂർണ്ണസംഖ്യ src_stride_seq,
    പൂർണ്ണസംഖ്യ dst_stride_token,
    പൂർണ്ണസംഖ്യ dst_stride_seq,
    തർക്കം apply_silu,
    സ്ഥിരം ദശാംശം *bias
) {
    // src is token-contiguous per channel, dst is channel-contiguous per token,
    // so indexing token-fastest coalesces the d_conv loads.
    സ്ഥിരം പൂർണ്ണസംഖ്യ token   = static_cast<പൂർണ്ണസംഖ്യ>(idx % n_t);
    സ്ഥിരം പൂർണ്ണസംഖ്യ channel = static_cast<പൂർണ്ണസംഖ്യ>((idx / n_t) % d_inner);
    സ്ഥിരം പൂർണ്ണസംഖ്യ seq     = static_cast<പൂർണ്ണസംഖ്യ>(idx / (static_cast<size_t>(n_t) * static_cast<size_t>(d_inner)));

    സ്ഥിരം ദശാംശം *s = src_data
        + static_cast<size_t>(seq) * static_cast<size_t>(src_stride_seq)
        + static_cast<size_t>(channel) * static_cast<size_t>(src_stride_inner)
        + static_cast<size_t>(token);

    സ്ഥിരം ദശാംശം *c = weights + static_cast<size_t>(channel) * static_cast<size_t>(d_conv);

    ദശാംശം sumf = 0.0f;
    എങ്കിൽ constexpr (DC > 0) {
#pragma unroll
        വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = 0; i0 < DC; ++i0) {
            sumf += s[i0] * c[i0];
        }
    } അല്ലെങ്കിൽ {
        വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = 0; i0 < d_conv; ++i0) {
            sumf += s[i0] * c[i0];
        }
    }

    // fused bias add: the ADD node broadcasts a 1-D channel bias over tokens
    എങ്കിൽ (bias != nullptr) {
        sumf += bias[channel];
    }

    സ്ഥിരം size_t dst_idx =
        static_cast<size_t>(seq) * static_cast<size_t>(dst_stride_seq) +
        static_cast<size_t>(token) * static_cast<size_t>(dst_stride_token) +
        static_cast<size_t>(channel);

    dst_data[dst_idx] = apply_silu ? op_silu(sumf) : sumf;
}

// FUSED=false keeps apply_silu/bias out of the kernel capture list, so the unfused launch
// takes the pre-fusion argument list; matters at n_t == 1, where the op is launch-bound.
template <പൂർണ്ണസംഖ്യ DC, തർക്കം FUSED>
സ്ഥിതം ശൂന്യം kernel_ssm_conv_impl(
    queue &q,
    സ്ഥിരം ദശാംശം *src_data,
    സ്ഥിരം ദശാംശം *weights,
    ദശാംശം *dst_data,
    പൂർണ്ണസംഖ്യ d_conv,
    പൂർണ്ണസംഖ്യ d_inner,
    പൂർണ്ണസംഖ്യ n_t,
    പൂർണ്ണസംഖ്യ n_s,
    പൂർണ്ണസംഖ്യ ncs __attribute__((unused)),
    പൂർണ്ണസംഖ്യ src_stride_inner,
    പൂർണ്ണസംഖ്യ src_stride_seq,
    പൂർണ്ണസംഖ്യ dst_stride_token,
    പൂർണ്ണസംഖ്യ dst_stride_seq,
    തർക്കം apply_silu,
    സ്ഥിരം ദശാംശം *bias
) {
    സ്ഥിരം size_t total_work = static_cast<size_t>(d_inner) * static_cast<size_t>(n_t) * static_cast<size_t>(n_s);
    സ്ഥിരം size_t work_group_size = 256;
    സ്ഥിരം size_t അക്കം_work_groups = (total_work + work_group_size - 1) / work_group_size;

    സ്ഥിരം range<1> global_range(അക്കം_work_groups * work_group_size);
    സ്ഥിരം range<1> local_range(work_group_size);

    എങ്കിൽ constexpr (FUSED) {
        q.submit([&](handler &h) {
            h.parallel_for(
                nd_range<1>(global_range, local_range),
                [=](nd_item<1> item) {
                    സ്ഥിരം size_t idx = item.get_global_id(0);
                    എങ്കിൽ (idx >= total_work) {
                        തിരികെ;
                    }

                    ssm_conv_element<DC>(idx, src_data, weights, dst_data, d_conv, d_inner, n_t,
                                         src_stride_inner, src_stride_seq, dst_stride_token,
                                         dst_stride_seq, apply_silu, bias);
                }
            );
        });
    } അല്ലെങ്കിൽ {
        GGML_UNUSED(apply_silu);
        GGML_UNUSED(bias);

        q.submit([&](handler &h) {
            h.parallel_for(
                nd_range<1>(global_range, local_range),
                [=](nd_item<1> item) {
                    സ്ഥിരം size_t idx = item.get_global_id(0);
                    എങ്കിൽ (idx >= total_work) {
                        തിരികെ;
                    }

                    ssm_conv_element<DC>(idx, src_data, weights, dst_data, d_conv, d_inner, n_t,
                                         src_stride_inner, src_stride_seq, dst_stride_token,
                                         dst_stride_seq, അസത്യം, nullptr);
                }
            );
        });
    }
}

// SLM transpose tile: coalesces both the loads and the stores. The +1 pad makes the row
// stride 33, coprime with 32 banks, so both phases are bank-conflict-free.
template <പൂർണ്ണസംഖ്യ DC, പൂർണ്ണസംഖ്യ TT, പൂർണ്ണസംഖ്യ TC, പൂർണ്ണസംഖ്യ WG>
സ്ഥിതം __dpct_inline__ ശൂന്യം ssm_conv_tile(
    nd_item<1> it, local_accessor<ദശാംശം, 1> tile, സ്ഥിരം ദശാംശം *src_data, സ്ഥിരം ദശാംശം *weights,
    ദശാംശം *dst_data, പൂർണ്ണസംഖ്യ n_t, പൂർണ്ണസംഖ്യ nt_tiles, പൂർണ്ണസംഖ്യ nc_tiles, പൂർണ്ണസംഖ്യ src_stride_inner,
    പൂർണ്ണസംഖ്യ src_stride_seq, പൂർണ്ണസംഖ്യ dst_stride_token, പൂർണ്ണസംഖ്യ dst_stride_seq, തർക്കം apply_silu,
    സ്ഥിരം ദശാംശം *bias
) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ    lid = static_cast<പൂർണ്ണസംഖ്യ>(it.get_local_id(0));
    സ്ഥിരം size_t g   = it.get_group(0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ    tt  = static_cast<പൂർണ്ണസംഖ്യ>(g % nt_tiles);
    സ്ഥിരം പൂർണ്ണസംഖ്യ    ct  = static_cast<പൂർണ്ണസംഖ്യ>((g / nt_tiles) % nc_tiles);
    സ്ഥിരം പൂർണ്ണസംഖ്യ    seq = static_cast<പൂർണ്ണസംഖ്യ>(g / (static_cast<size_t>(nt_tiles) * nc_tiles));
    സ്ഥിരം പൂർണ്ണസംഖ്യ    t0 = tt * TT, c0 = ct * TC;

    സ്ഥിരം പൂർണ്ണസംഖ്യ ti = lid % TT;
    സ്ഥിരം പൂർണ്ണസംഖ്യ cj = lid / TT;
#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ r = 0; r < TC / (WG / TT); ++r) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ c   = cj + r * (WG / TT);
        സ്ഥിരം പൂർണ്ണസംഖ്യ tok = t0 + ti;
        ദശാംശം sumf = 0.0f;
        എങ്കിൽ (tok < n_t) {
            സ്ഥിരം ദശാംശം *s = src_data + static_cast<size_t>(seq) * src_stride_seq
                           + static_cast<size_t>(c0 + c) * src_stride_inner + tok;
            സ്ഥിരം ദശാംശം *cw = weights + static_cast<size_t>(c0 + c) * DC;
#pragma unroll
            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < DC; ++i) sumf += s[i] * cw[i];
            എങ്കിൽ (bias != nullptr) sumf += bias[c0 + c];
            എങ്കിൽ (apply_silu) sumf = op_silu(sumf);
        }
        tile[c * (TT + 1) + ti] = sumf;
    }
    it.barrier(access::fence_space::local_space);

    സ്ഥിരം പൂർണ്ണസംഖ്യ cc = lid % TC;
    സ്ഥിരം പൂർണ്ണസംഖ്യ tj = lid / TC;
#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ r = 0; r < TT / (WG / TC); ++r) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ t   = tj + r * (WG / TC);
        സ്ഥിരം പൂർണ്ണസംഖ്യ tok = t0 + t;
        എങ്കിൽ (tok < n_t) {
            dst_data[static_cast<size_t>(seq) * dst_stride_seq
                     + static_cast<size_t>(tok) * dst_stride_token + c0 + cc]
                = tile[cc * (TT + 1) + t];
        }
    }
}

// Same FUSED split as kernel_ssm_conv_impl. The fused instantiation keeps the runtime
// apply_silu/bias branches: at n_t >= 32 they are amortized over the whole tile.
template <പൂർണ്ണസംഖ്യ DC, തർക്കം FUSED>
സ്ഥിതം ശൂന്യം kernel_ssm_conv_tiled(
    queue &q, സ്ഥിരം ദശാംശം *src_data, സ്ഥിരം ദശാംശം *weights, ദശാംശം *dst_data,
    പൂർണ്ണസംഖ്യ d_inner, പൂർണ്ണസംഖ്യ n_t, പൂർണ്ണസംഖ്യ n_s, പൂർണ്ണസംഖ്യ src_stride_inner, പൂർണ്ണസംഖ്യ src_stride_seq,
    പൂർണ്ണസംഖ്യ dst_stride_token, പൂർണ്ണസംഖ്യ dst_stride_seq, തർക്കം apply_silu, സ്ഥിരം ദശാംശം *bias
) {
    constexpr പൂർണ്ണസംഖ്യ TT = 32, TC = 32, WG = 256;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nt_tiles = (n_t + TT - 1) / TT;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nc_tiles = d_inner / TC;
    സ്ഥിരം size_t groups = static_cast<size_t>(nt_tiles) * nc_tiles * n_s;

    എങ്കിൽ constexpr (FUSED) {
        q.submit([&](handler &h) {
            local_accessor<ദശാംശം, 1> tile(range<1>(TC * (TT + 1)), h);
            h.parallel_for(nd_range<1>(range<1>(groups * WG), range<1>(WG)), [=](nd_item<1> it) {
                ssm_conv_tile<DC, TT, TC, WG>(it, tile, src_data, weights, dst_data, n_t, nt_tiles,
                                              nc_tiles, src_stride_inner, src_stride_seq,
                                              dst_stride_token, dst_stride_seq, apply_silu, bias);
            });
        });
    } അല്ലെങ്കിൽ {
        GGML_UNUSED(apply_silu);
        GGML_UNUSED(bias);

        q.submit([&](handler &h) {
            local_accessor<ദശാംശം, 1> tile(range<1>(TC * (TT + 1)), h);
            h.parallel_for(nd_range<1>(range<1>(groups * WG), range<1>(WG)), [=](nd_item<1> it) {
                ssm_conv_tile<DC, TT, TC, WG>(it, tile, src_data, weights, dst_data, n_t, nt_tiles,
                                              nc_tiles, src_stride_inner, src_stride_seq,
                                              dst_stride_token, dst_stride_seq, അസത്യം, nullptr);
            });
        });
    }
}

സ്ഥിതം ശൂന്യം kernel_ssm_conv(
    queue &q,
    സ്ഥിരം ദശാംശം *src_data,
    സ്ഥിരം ദശാംശം *weights,
    ദശാംശം *dst_data,
    പൂർണ്ണസംഖ്യ d_conv,
    പൂർണ്ണസംഖ്യ d_inner,
    പൂർണ്ണസംഖ്യ n_t,
    പൂർണ്ണസംഖ്യ n_s,
    പൂർണ്ണസംഖ്യ ncs,
    പൂർണ്ണസംഖ്യ src_stride_inner,
    പൂർണ്ണസംഖ്യ src_stride_seq,
    പൂർണ്ണസംഖ്യ dst_stride_token,
    പൂർണ്ണസംഖ്യ dst_stride_seq,
    തർക്കം apply_silu,
    സ്ഥിരം ദശാംശം *bias
) {
    // Only the fused instantiations carry apply_silu/bias as kernel arguments; the plain
    // ssm_conv launch keeps the argument list it had before the fusion landed.
    സ്ഥിരം തർക്കം fused = apply_silu || bias != nullptr;

    // d_inner must be a multiple of 32 so the channel tiles are exact; the transpose is only
    // worth it for n_t >= 32. d_conv == 4 is the only window with a DC-specialized kernel.
    എങ്കിൽ (d_conv == 4 && n_t >= 32 && (d_inner % 32) == 0) {
        എങ്കിൽ (fused) {
            kernel_ssm_conv_tiled<4, സത്യം>(q, src_data, weights, dst_data, d_inner, n_t, n_s,
                                           src_stride_inner, src_stride_seq, dst_stride_token,
                                           dst_stride_seq, apply_silu, bias);
        } അല്ലെങ്കിൽ {
            kernel_ssm_conv_tiled<4, അസത്യം>(q, src_data, weights, dst_data, d_inner, n_t, n_s,
                                            src_stride_inner, src_stride_seq, dst_stride_token,
                                            dst_stride_seq, apply_silu, bias);
        }
        തിരികെ;
    }

    എങ്കിൽ (d_conv == 4) {
        എങ്കിൽ (fused) {
            kernel_ssm_conv_impl<4, സത്യം>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                          ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                          dst_stride_seq, apply_silu, bias);
        } അല്ലെങ്കിൽ {
            kernel_ssm_conv_impl<4, അസത്യം>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                           ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                           dst_stride_seq, apply_silu, bias);
        }
        തിരികെ;
    }

    എങ്കിൽ (fused) {
        kernel_ssm_conv_impl<0, സത്യം>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                      ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                      dst_stride_seq, apply_silu, bias);
    } അല്ലെങ്കിൽ {
        kernel_ssm_conv_impl<0, അസത്യം>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                       ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                       dst_stride_seq, apply_silu, bias);
    }
}

വരിയിൽ ശൂന്യം ggml_sycl_op_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * silu_dst = nullptr, സ്ഥിരം ദശാംശം * bias = nullptr) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(bias == nullptr || silu_dst != nullptr);

    സ്ഥിരം പൂർണ്ണസംഖ്യ d_conv   = src1->ne[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncs      = src0->ne[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ d_inner  = src0->ne[1];
    സ്ഥിരം പൂർണ്ണസംഖ്യ n_t      = dst->ne[1];
    സ്ഥിരം പൂർണ്ണസംഖ്യ n_s      = dst->ne[2];

    GGML_ASSERT(src0->ne[0] == d_conv - 1 + n_t);
    GGML_ASSERT(src0->ne[1] == d_inner);
    GGML_ASSERT(src1->ne[1] == d_inner);

    GGML_ASSERT(dst->ne[0] == d_inner);
    GGML_ASSERT(dst->ne[1] == n_t);
    GGML_ASSERT(dst->ne[2] == n_s);

    GGML_ASSERT(src0->nb[0] == വലുപ്പം(ദശാംശം));
    GGML_ASSERT(src1->nb[0] == വലുപ്പം(ദശാംശം));

    GGML_ASSERT(src0->nb[1] == src0->ne[0] * വലുപ്പം(ദശാംശം));

    സ്ഥിരം പൂർണ്ണസംഖ്യ src_stride_inner = ncs;
    സ്ഥിരം പൂർണ്ണസംഖ്യ src_stride_seq   = ncs * d_inner;
    സ്ഥിരം പൂർണ്ണസംഖ്യ dst_stride_token = d_inner;
    സ്ഥിരം പൂർണ്ണസംഖ്യ dst_stride_seq   = d_inner * n_t;

    ശ്രമിക്കുക {
        queue *q = ctx.stream();

        സ്ഥിരം ദശാംശം *src_data = static_cast<സ്ഥിരം ദശാംശം *>(src0->data);
        സ്ഥിരം ദശാംശം *weights  = static_cast<സ്ഥിരം ദശാംശം *>(src1->data);
        സ്ഥിരം തർക്കം apply_silu = silu_dst != nullptr;
        ദശാംശം *dst_data       = static_cast<ദശാംശം *>((silu_dst ? silu_dst : dst)->data);

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

    } പിടിക്കുക (സ്ഥിരം std::exception &e) {
        std::fprintf(stderr, "[SYCL-SSM_CONV] ERROR: %s\n", e.what());
        എറിയുക;
    }
}

ശൂന്യം ggml_sycl_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_ssm_conv(ctx, dst);
}

// Fused ssm_conv + ADD + SiLU: write silu(conv(x) + b) straight into silu_dst, eliding the
// standalone SiLU launch and its HBM round-trip of the conv output.
ശൂന്യം ggml_sycl_ssm_conv_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * add, ggml_tensor * silu_dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    GGML_ASSERT(silu_dst && ggml_are_same_shape(dst, silu_dst) && silu_dst->type == GGML_TYPE_F32);
    // the fused kernel reads only the ADD's bias operand; the ADD result is never written
    സ്ഥിരം ദശാംശം * bias = nullptr;
    എങ്കിൽ (add != nullptr) {
        സ്ഥിരം ggml_tensor * bias_t = (add->src[0] == dst) ? add->src[1] : add->src[0];
        bias = static_cast<സ്ഥിരം ദശാംശം *>(bias_t->data);
    }
    ggml_sycl_op_ssm_conv(ctx, dst, silu_dst, bias);
}
