#ਅਨ੍ਤਰ੍ਭਾਵ "ssm_conv.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "common.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "element_wise.hpp"

#ਅਨ੍ਤਰ੍ਭਾਵ <cstdio>

ਵਰਤੋਂ ਨਾਮਾਕਾਸ਼ sycl;

// One output element of the conv. DC is d_conv as a compile-time constant (0 keeps the
// runtime loop); unfused callers pass literal false/nullptr so the epilogue folds away.
ਸਾਂਚਾ <ਪੂਰਨ DC>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ ssm_conv_element(
    size_t idx,
    ਅਚਲ ਦਸ਼ਮਲਵ *src_ਅੰਕੜੇ,
    ਅਚਲ ਦਸ਼ਮਲਵ *weights,
    ਦਸ਼ਮਲਵ *dst_ਅੰਕੜੇ,
    ਪੂਰਨ d_conv,
    ਪੂਰਨ d_inner,
    ਪੂਰਨ n_t,
    ਪੂਰਨ src_stride_inner,
    ਪੂਰਨ src_stride_seq,
    ਪੂਰਨ dst_stride_token,
    ਪੂਰਨ dst_stride_seq,
    ਤਾਰਕਿਕ apply_silu,
    ਅਚਲ ਦਸ਼ਮਲਵ *bias
) {
    // src is token-contiguous per channel, dst is channel-contiguous per token,
    // so indexing token-fastest coalesces the d_conv loads.
    ਅਚਲ ਪੂਰਨ token   = static_cast<ਪੂਰਨ>(idx % n_t);
    ਅਚਲ ਪੂਰਨ channel = static_cast<ਪੂਰਨ>((idx / n_t) % d_inner);
    ਅਚਲ ਪੂਰਨ seq     = static_cast<ਪੂਰਨ>(idx / (static_cast<size_t>(n_t) * static_cast<size_t>(d_inner)));

    ਅਚਲ ਦਸ਼ਮਲਵ *s = src_ਅੰਕੜੇ
        + static_cast<size_t>(seq) * static_cast<size_t>(src_stride_seq)
        + static_cast<size_t>(channel) * static_cast<size_t>(src_stride_inner)
        + static_cast<size_t>(token);

    ਅਚਲ ਦਸ਼ਮਲਵ *c = weights + static_cast<size_t>(channel) * static_cast<size_t>(d_conv);

    ਦਸ਼ਮਲਵ sumf = 0.0f;
    ਜੇ constexpr (DC > 0) {
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ i0 = 0; i0 < DC; ++i0) {
            sumf += s[i0] * c[i0];
        }
    } ਨਹੀਂਤਾਂ {
        ਲਈ (ਪੂਰਨ i0 = 0; i0 < d_conv; ++i0) {
            sumf += s[i0] * c[i0];
        }
    }

    // fused bias add: the ADD node broadcasts a 1-D channel bias over tokens
    ਜੇ (bias != ਖਾਲੀ_ਸੂਚਕ) {
        sumf += bias[channel];
    }

    ਅਚਲ size_t dst_idx =
        static_cast<size_t>(seq) * static_cast<size_t>(dst_stride_seq) +
        static_cast<size_t>(token) * static_cast<size_t>(dst_stride_token) +
        static_cast<size_t>(channel);

    dst_ਅੰਕੜੇ[dst_idx] = apply_silu ? op_silu(sumf) : sumf;
}

// FUSED=false keeps apply_silu/bias out of the kernel capture list, so the unfused launch
// takes the pre-fusion argument list; matters at n_t == 1, where the op is launch-bound.
ਸਾਂਚਾ <ਪੂਰਨ DC, ਤਾਰਕਿਕ FUSED>
ਸਥਿਰ ਖਾਲੀ kernel_ssm_conv_impl(
    queue &q,
    ਅਚਲ ਦਸ਼ਮਲਵ *src_ਅੰਕੜੇ,
    ਅਚਲ ਦਸ਼ਮਲਵ *weights,
    ਦਸ਼ਮਲਵ *dst_ਅੰਕੜੇ,
    ਪੂਰਨ d_conv,
    ਪੂਰਨ d_inner,
    ਪੂਰਨ n_t,
    ਪੂਰਨ n_s,
    ਪੂਰਨ ncs __attribute__((ਅਣਵਰਤਿਆ_2)),
    ਪੂਰਨ src_stride_inner,
    ਪੂਰਨ src_stride_seq,
    ਪੂਰਨ dst_stride_token,
    ਪੂਰਨ dst_stride_seq,
    ਤਾਰਕਿਕ apply_silu,
    ਅਚਲ ਦਸ਼ਮਲਵ *bias
) {
    ਅਚਲ size_t ਕੁੱਲ_work = static_cast<size_t>(d_inner) * static_cast<size_t>(n_t) * static_cast<size_t>(n_s);
    ਅਚਲ size_t work_ਸਮੂਹ_ਮਾਪ = 256;
    ਅਚਲ size_t ਅੰਕ_work_groups = (ਕੁੱਲ_work + work_ਸਮੂਹ_ਮਾਪ - 1) / work_ਸਮੂਹ_ਮਾਪ;

    ਅਚਲ range<1> global_range(ਅੰਕ_work_groups * work_ਸਮੂਹ_ਮਾਪ);
    ਅਚਲ range<1> local_range(work_ਸਮੂਹ_ਮਾਪ);

    ਜੇ constexpr (FUSED) {
        q.submit([&](handler &h) {
            h.parallel_for(
                nd_range<1>(global_range, local_range),
                [=](nd_ਵਸਤੂ<1> ਵਸਤੂ) {
                    ਅਚਲ size_t idx = ਵਸਤੂ.get_global_id(0);
                    ਜੇ (idx >= ਕੁੱਲ_work) {
                        ਵਾਪਸ;
                    }

                    ssm_conv_element<DC>(idx, src_ਅੰਕੜੇ, weights, dst_ਅੰਕੜੇ, d_conv, d_inner, n_t,
                                         src_stride_inner, src_stride_seq, dst_stride_token,
                                         dst_stride_seq, apply_silu, bias);
                }
            );
        });
    } ਨਹੀਂਤਾਂ {
        GGML_ਅਣਵਰਤਿਆ(apply_silu);
        GGML_ਅਣਵਰਤਿਆ(bias);

        q.submit([&](handler &h) {
            h.parallel_for(
                nd_range<1>(global_range, local_range),
                [=](nd_ਵਸਤੂ<1> ਵਸਤੂ) {
                    ਅਚਲ size_t idx = ਵਸਤੂ.get_global_id(0);
                    ਜੇ (idx >= ਕੁੱਲ_work) {
                        ਵਾਪਸ;
                    }

                    ssm_conv_element<DC>(idx, src_ਅੰਕੜੇ, weights, dst_ਅੰਕੜੇ, d_conv, d_inner, n_t,
                                         src_stride_inner, src_stride_seq, dst_stride_token,
                                         dst_stride_seq, ਝੂਠ, ਖਾਲੀ_ਸੂਚਕ);
                }
            );
        });
    }
}

// SLM transpose tile: coalesces both the loads and the stores. The +1 pad makes the row
// stride 33, coprime with 32 banks, so both phases are bank-conflict-free.
ਸਾਂਚਾ <ਪੂਰਨ DC, ਪੂਰਨ TT, ਪੂਰਨ TC, ਪੂਰਨ WG>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ ssm_conv_tile(
    nd_ਵਸਤੂ<1> it, local_accessor<ਦਸ਼ਮਲਵ, 1> tile, ਅਚਲ ਦਸ਼ਮਲਵ *src_ਅੰਕੜੇ, ਅਚਲ ਦਸ਼ਮਲਵ *weights,
    ਦਸ਼ਮਲਵ *dst_ਅੰਕੜੇ, ਪੂਰਨ n_t, ਪੂਰਨ nt_tiles, ਪੂਰਨ nc_tiles, ਪੂਰਨ src_stride_inner,
    ਪੂਰਨ src_stride_seq, ਪੂਰਨ dst_stride_token, ਪੂਰਨ dst_stride_seq, ਤਾਰਕਿਕ apply_silu,
    ਅਚਲ ਦਸ਼ਮਲਵ *bias
) {
    ਅਚਲ ਪੂਰਨ    lid = static_cast<ਪੂਰਨ>(it.get_local_id(0));
    ਅਚਲ size_t g   = it.get_ਸਮੂਹ(0);
    ਅਚਲ ਪੂਰਨ    tt  = static_cast<ਪੂਰਨ>(g % nt_tiles);
    ਅਚਲ ਪੂਰਨ    ct  = static_cast<ਪੂਰਨ>((g / nt_tiles) % nc_tiles);
    ਅਚਲ ਪੂਰਨ    seq = static_cast<ਪੂਰਨ>(g / (static_cast<size_t>(nt_tiles) * nc_tiles));
    ਅਚਲ ਪੂਰਨ    t0 = tt * TT, c0 = ct * TC;

    ਅਚਲ ਪੂਰਨ ti = lid % TT;
    ਅਚਲ ਪੂਰਨ cj = lid / TT;
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ r = 0; r < TC / (WG / TT); ++r) {
        ਅਚਲ ਪੂਰਨ c   = cj + r * (WG / TT);
        ਅਚਲ ਪੂਰਨ tok = t0 + ti;
        ਦਸ਼ਮਲਵ sumf = 0.0f;
        ਜੇ (tok < n_t) {
            ਅਚਲ ਦਸ਼ਮਲਵ *s = src_ਅੰਕੜੇ + static_cast<size_t>(seq) * src_stride_seq
                           + static_cast<size_t>(c0 + c) * src_stride_inner + tok;
            ਅਚਲ ਦਸ਼ਮਲਵ *cw = weights + static_cast<size_t>(c0 + c) * DC;
#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ i = 0; i < DC; ++i) sumf += s[i] * cw[i];
            ਜੇ (bias != ਖਾਲੀ_ਸੂਚਕ) sumf += bias[c0 + c];
            ਜੇ (apply_silu) sumf = op_silu(sumf);
        }
        tile[c * (TT + 1) + ti] = sumf;
    }
    it.barrier(access::fence_space::local_space);

    ਅਚਲ ਪੂਰਨ cc = lid % TC;
    ਅਚਲ ਪੂਰਨ tj = lid / TC;
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ r = 0; r < TT / (WG / TC); ++r) {
        ਅਚਲ ਪੂਰਨ t   = tj + r * (WG / TC);
        ਅਚਲ ਪੂਰਨ tok = t0 + t;
        ਜੇ (tok < n_t) {
            dst_ਅੰਕੜੇ[static_cast<size_t>(seq) * dst_stride_seq
                     + static_cast<size_t>(tok) * dst_stride_token + c0 + cc]
                = tile[cc * (TT + 1) + t];
        }
    }
}

// Same FUSED split as kernel_ssm_conv_impl. The fused instantiation keeps the runtime
// apply_silu/bias branches: at n_t >= 32 they are amortized over the whole tile.
ਸਾਂਚਾ <ਪੂਰਨ DC, ਤਾਰਕਿਕ FUSED>
ਸਥਿਰ ਖਾਲੀ kernel_ssm_conv_tiled(
    queue &q, ਅਚਲ ਦਸ਼ਮਲਵ *src_ਅੰਕੜੇ, ਅਚਲ ਦਸ਼ਮਲਵ *weights, ਦਸ਼ਮਲਵ *dst_ਅੰਕੜੇ,
    ਪੂਰਨ d_inner, ਪੂਰਨ n_t, ਪੂਰਨ n_s, ਪੂਰਨ src_stride_inner, ਪੂਰਨ src_stride_seq,
    ਪੂਰਨ dst_stride_token, ਪੂਰਨ dst_stride_seq, ਤਾਰਕਿਕ apply_silu, ਅਚਲ ਦਸ਼ਮਲਵ *bias
) {
    constexpr ਪੂਰਨ TT = 32, TC = 32, WG = 256;
    ਅਚਲ ਪੂਰਨ nt_tiles = (n_t + TT - 1) / TT;
    ਅਚਲ ਪੂਰਨ nc_tiles = d_inner / TC;
    ਅਚਲ size_t groups = static_cast<size_t>(nt_tiles) * nc_tiles * n_s;

    ਜੇ constexpr (FUSED) {
        q.submit([&](handler &h) {
            local_accessor<ਦਸ਼ਮਲਵ, 1> tile(range<1>(TC * (TT + 1)), h);
            h.parallel_for(nd_range<1>(range<1>(groups * WG), range<1>(WG)), [=](nd_ਵਸਤੂ<1> it) {
                ssm_conv_tile<DC, TT, TC, WG>(it, tile, src_ਅੰਕੜੇ, weights, dst_ਅੰਕੜੇ, n_t, nt_tiles,
                                              nc_tiles, src_stride_inner, src_stride_seq,
                                              dst_stride_token, dst_stride_seq, apply_silu, bias);
            });
        });
    } ਨਹੀਂਤਾਂ {
        GGML_ਅਣਵਰਤਿਆ(apply_silu);
        GGML_ਅਣਵਰਤਿਆ(bias);

        q.submit([&](handler &h) {
            local_accessor<ਦਸ਼ਮਲਵ, 1> tile(range<1>(TC * (TT + 1)), h);
            h.parallel_for(nd_range<1>(range<1>(groups * WG), range<1>(WG)), [=](nd_ਵਸਤੂ<1> it) {
                ssm_conv_tile<DC, TT, TC, WG>(it, tile, src_ਅੰਕੜੇ, weights, dst_ਅੰਕੜੇ, n_t, nt_tiles,
                                              nc_tiles, src_stride_inner, src_stride_seq,
                                              dst_stride_token, dst_stride_seq, ਝੂਠ, ਖਾਲੀ_ਸੂਚਕ);
            });
        });
    }
}

ਸਥਿਰ ਖਾਲੀ kernel_ssm_conv(
    queue &q,
    ਅਚਲ ਦਸ਼ਮਲਵ *src_ਅੰਕੜੇ,
    ਅਚਲ ਦਸ਼ਮਲਵ *weights,
    ਦਸ਼ਮਲਵ *dst_ਅੰਕੜੇ,
    ਪੂਰਨ d_conv,
    ਪੂਰਨ d_inner,
    ਪੂਰਨ n_t,
    ਪੂਰਨ n_s,
    ਪੂਰਨ ncs,
    ਪੂਰਨ src_stride_inner,
    ਪੂਰਨ src_stride_seq,
    ਪੂਰਨ dst_stride_token,
    ਪੂਰਨ dst_stride_seq,
    ਤਾਰਕਿਕ apply_silu,
    ਅਚਲ ਦਸ਼ਮਲਵ *bias
) {
    // Only the fused instantiations carry apply_silu/bias as kernel arguments; the plain
    // ssm_conv launch keeps the argument list it had before the fusion landed.
    ਅਚਲ ਤਾਰਕਿਕ fused = apply_silu || bias != ਖਾਲੀ_ਸੂਚਕ;

    // d_inner must be a multiple of 32 so the channel tiles are exact; the transpose is only
    // worth it for n_t >= 32. d_conv == 4 is the only window with a DC-specialized kernel.
    ਜੇ (d_conv == 4 && n_t >= 32 && (d_inner % 32) == 0) {
        ਜੇ (fused) {
            kernel_ssm_conv_tiled<4, ਸੱਚ>(q, src_ਅੰਕੜੇ, weights, dst_ਅੰਕੜੇ, d_inner, n_t, n_s,
                                           src_stride_inner, src_stride_seq, dst_stride_token,
                                           dst_stride_seq, apply_silu, bias);
        } ਨਹੀਂਤਾਂ {
            kernel_ssm_conv_tiled<4, ਝੂਠ>(q, src_ਅੰਕੜੇ, weights, dst_ਅੰਕੜੇ, d_inner, n_t, n_s,
                                            src_stride_inner, src_stride_seq, dst_stride_token,
                                            dst_stride_seq, apply_silu, bias);
        }
        ਵਾਪਸ;
    }

    ਜੇ (d_conv == 4) {
        ਜੇ (fused) {
            kernel_ssm_conv_impl<4, ਸੱਚ>(q, src_ਅੰਕੜੇ, weights, dst_ਅੰਕੜੇ, d_conv, d_inner, n_t, n_s,
                                          ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                          dst_stride_seq, apply_silu, bias);
        } ਨਹੀਂਤਾਂ {
            kernel_ssm_conv_impl<4, ਝੂਠ>(q, src_ਅੰਕੜੇ, weights, dst_ਅੰਕੜੇ, d_conv, d_inner, n_t, n_s,
                                           ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                           dst_stride_seq, apply_silu, bias);
        }
        ਵਾਪਸ;
    }

    ਜੇ (fused) {
        kernel_ssm_conv_impl<0, ਸੱਚ>(q, src_ਅੰਕੜੇ, weights, dst_ਅੰਕੜੇ, d_conv, d_inner, n_t, n_s,
                                      ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                      dst_stride_seq, apply_silu, bias);
    } ਨਹੀਂਤਾਂ {
        kernel_ssm_conv_impl<0, ਝੂਠ>(q, src_ਅੰਕੜੇ, weights, dst_ਅੰਕੜੇ, d_conv, d_inner, n_t, n_s,
                                       ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                       dst_stride_seq, apply_silu, bias);
    }
}

ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * silu_dst = ਖਾਲੀ_ਸੂਚਕ, ਅਚਲ ਦਸ਼ਮਲਵ * bias = ਖਾਲੀ_ਸੂਚਕ) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(bias == ਖਾਲੀ_ਸੂਚਕ || silu_dst != ਖਾਲੀ_ਸੂਚਕ);

    ਅਚਲ ਪੂਰਨ d_conv   = src1->ne[0];
    ਅਚਲ ਪੂਰਨ ncs      = src0->ne[0];
    ਅਚਲ ਪੂਰਨ d_inner  = src0->ne[1];
    ਅਚਲ ਪੂਰਨ n_t      = dst->ne[1];
    ਅਚਲ ਪੂਰਨ n_s      = dst->ne[2];

    GGML_ASSERT(src0->ne[0] == d_conv - 1 + n_t);
    GGML_ASSERT(src0->ne[1] == d_inner);
    GGML_ASSERT(src1->ne[1] == d_inner);

    GGML_ASSERT(dst->ne[0] == d_inner);
    GGML_ASSERT(dst->ne[1] == n_t);
    GGML_ASSERT(dst->ne[2] == n_s);

    GGML_ASSERT(src0->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    GGML_ASSERT(src1->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    GGML_ASSERT(src0->nb[1] == src0->ne[0] * ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਅਚਲ ਪੂਰਨ src_stride_inner = ncs;
    ਅਚਲ ਪੂਰਨ src_stride_seq   = ncs * d_inner;
    ਅਚਲ ਪੂਰਨ dst_stride_token = d_inner;
    ਅਚਲ ਪੂਰਨ dst_stride_seq   = d_inner * n_t;

    ਕੋਸ਼ਿਸ਼ {
        queue *q = ctx.stream();

        ਅਚਲ ਦਸ਼ਮਲਵ *src_ਅੰਕੜੇ = static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(src0->ਅੰਕੜੇ);
        ਅਚਲ ਦਸ਼ਮਲਵ *weights  = static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(src1->ਅੰਕੜੇ);
        ਅਚਲ ਤਾਰਕਿਕ apply_silu = silu_dst != ਖਾਲੀ_ਸੂਚਕ;
        ਦਸ਼ਮਲਵ *dst_ਅੰਕੜੇ       = static_cast<ਦਸ਼ਮਲਵ *>((silu_dst ? silu_dst : dst)->ਅੰਕੜੇ);

        GGML_ASSERT(src_ਅੰਕੜੇ && weights && dst_ਅੰਕੜੇ);

        kernel_ssm_conv(
            *q,
            src_ਅੰਕੜੇ,
            weights,
            dst_ਅੰਕੜੇ,
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

    } ਫੜੋ (ਅਚਲ std::exception &e) {
        std::fprintf(stderr, "[SYCL-SSM_CONV] ERROR: %s\n", e.what());
        ਸੁੱਟੋ;
    }
}

ਖਾਲੀ ggml_sycl_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_ssm_conv(ctx, dst);
}

// Fused ssm_conv + ADD + SiLU: write silu(conv(x) + b) straight into silu_dst, eliding the
// standalone SiLU launch and its HBM round-trip of the conv output.
ਖਾਲੀ ggml_sycl_ssm_conv_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * ਜੋੜੋ_2, ggml_tensor * silu_dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    GGML_ASSERT(silu_dst && ggml_are_same_shape(dst, silu_dst) && silu_dst->type == GGML_TYPE_F32);
    // the fused kernel reads only the ADD's bias operand; the ADD result is never written
    ਅਚਲ ਦਸ਼ਮਲਵ * bias = ਖਾਲੀ_ਸੂਚਕ;
    ਜੇ (ਜੋੜੋ_2 != ਖਾਲੀ_ਸੂਚਕ) {
        ਅਚਲ ggml_tensor * bias_t = (ਜੋੜੋ_2->src[0] == dst) ? ਜੋੜੋ_2->src[1] : ਜੋੜੋ_2->src[0];
        bias = static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(bias_t->ਅੰਕੜੇ);
    }
    ggml_sycl_op_ssm_conv(ctx, dst, silu_dst, bias);
}
