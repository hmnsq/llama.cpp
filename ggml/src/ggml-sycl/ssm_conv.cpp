#અન્તર્ભાવ "ssm_conv.hpp"
#અન્તર્ભાવ "common.hpp"
#અન્તર્ભાવ "element_wise.hpp"

#અન્તર્ભાવ <cstdio>

using નામાકાશ sycl;

// One output element of the conv. DC is d_conv as a compile-time constant (0 keeps the
// runtime loop); unfused callers pass literal false/nullptr so the epilogue folds away.
template <પૂર્ણાંક DC>
સ્થિર __dpct_inline__ શૂન્ય ssm_conv_element(
    size_t idx,
    અચલ દશાંશ *src_data,
    અચલ દશાંશ *weights,
    દશાંશ *dst_data,
    પૂર્ણાંક d_conv,
    પૂર્ણાંક d_inner,
    પૂર્ણાંક n_t,
    પૂર્ણાંક src_stride_inner,
    પૂર્ણાંક src_stride_seq,
    પૂર્ણાંક dst_stride_token,
    પૂર્ણાંક dst_stride_seq,
    તાર્કિક apply_silu,
    અચલ દશાંશ *bias
) {
    // src is token-contiguous per channel, dst is channel-contiguous per token,
    // so indexing token-fastest coalesces the d_conv loads.
    અચલ પૂર્ણાંક token   = static_cast<પૂર્ણાંક>(idx % n_t);
    અચલ પૂર્ણાંક channel = static_cast<પૂર્ણાંક>((idx / n_t) % d_inner);
    અચલ પૂર્ણાંક seq     = static_cast<પૂર્ણાંક>(idx / (static_cast<size_t>(n_t) * static_cast<size_t>(d_inner)));

    અચલ દશાંશ *s = src_data
        + static_cast<size_t>(seq) * static_cast<size_t>(src_stride_seq)
        + static_cast<size_t>(channel) * static_cast<size_t>(src_stride_inner)
        + static_cast<size_t>(token);

    અચલ દશાંશ *c = weights + static_cast<size_t>(channel) * static_cast<size_t>(d_conv);

    દશાંશ sumf = 0.0f;
    જો constexpr (DC > 0) {
#pragma unroll
        માટે (પૂર્ણાંક i0 = 0; i0 < DC; ++i0) {
            sumf += s[i0] * c[i0];
        }
    } નહીંતર {
        માટે (પૂર્ણાંક i0 = 0; i0 < d_conv; ++i0) {
            sumf += s[i0] * c[i0];
        }
    }

    // fused bias add: the ADD node broadcasts a 1-D channel bias over tokens
    જો (bias != nullptr) {
        sumf += bias[channel];
    }

    અચલ size_t dst_idx =
        static_cast<size_t>(seq) * static_cast<size_t>(dst_stride_seq) +
        static_cast<size_t>(token) * static_cast<size_t>(dst_stride_token) +
        static_cast<size_t>(channel);

    dst_data[dst_idx] = apply_silu ? op_silu(sumf) : sumf;
}

// FUSED=false keeps apply_silu/bias out of the kernel capture list, so the unfused launch
// takes the pre-fusion argument list; matters at n_t == 1, where the op is launch-bound.
template <પૂર્ણાંક DC, તાર્કિક FUSED>
સ્થિર શૂન્ય kernel_ssm_conv_impl(
    queue &q,
    અચલ દશાંશ *src_data,
    અચલ દશાંશ *weights,
    દશાંશ *dst_data,
    પૂર્ણાંક d_conv,
    પૂર્ણાંક d_inner,
    પૂર્ણાંક n_t,
    પૂર્ણાંક n_s,
    પૂર્ણાંક ncs __attribute__((unused)),
    પૂર્ણાંક src_stride_inner,
    પૂર્ણાંક src_stride_seq,
    પૂર્ણાંક dst_stride_token,
    પૂર્ણાંક dst_stride_seq,
    તાર્કિક apply_silu,
    અચલ દશાંશ *bias
) {
    અચલ size_t total_work = static_cast<size_t>(d_inner) * static_cast<size_t>(n_t) * static_cast<size_t>(n_s);
    અચલ size_t work_group_size = 256;
    અચલ size_t અંક_work_groups = (total_work + work_group_size - 1) / work_group_size;

    અચલ range<1> global_range(અંક_work_groups * work_group_size);
    અચલ range<1> local_range(work_group_size);

    જો constexpr (FUSED) {
        q.submit([&](handler &h) {
            h.parallel_for(
                nd_range<1>(global_range, local_range),
                [=](nd_item<1> item) {
                    અચલ size_t idx = item.get_global_id(0);
                    જો (idx >= total_work) {
                        પાછા;
                    }

                    ssm_conv_element<DC>(idx, src_data, weights, dst_data, d_conv, d_inner, n_t,
                                         src_stride_inner, src_stride_seq, dst_stride_token,
                                         dst_stride_seq, apply_silu, bias);
                }
            );
        });
    } નહીંતર {
        GGML_UNUSED(apply_silu);
        GGML_UNUSED(bias);

        q.submit([&](handler &h) {
            h.parallel_for(
                nd_range<1>(global_range, local_range),
                [=](nd_item<1> item) {
                    અચલ size_t idx = item.get_global_id(0);
                    જો (idx >= total_work) {
                        પાછા;
                    }

                    ssm_conv_element<DC>(idx, src_data, weights, dst_data, d_conv, d_inner, n_t,
                                         src_stride_inner, src_stride_seq, dst_stride_token,
                                         dst_stride_seq, ખોટું, nullptr);
                }
            );
        });
    }
}

// SLM transpose tile: coalesces both the loads and the stores. The +1 pad makes the row
// stride 33, coprime with 32 banks, so both phases are bank-conflict-free.
template <પૂર્ણાંક DC, પૂર્ણાંક TT, પૂર્ણાંક TC, પૂર્ણાંક WG>
સ્થિર __dpct_inline__ શૂન્ય ssm_conv_tile(
    nd_item<1> it, local_accessor<દશાંશ, 1> tile, અચલ દશાંશ *src_data, અચલ દશાંશ *weights,
    દશાંશ *dst_data, પૂર્ણાંક n_t, પૂર્ણાંક nt_tiles, પૂર્ણાંક nc_tiles, પૂર્ણાંક src_stride_inner,
    પૂર્ણાંક src_stride_seq, પૂર્ણાંક dst_stride_token, પૂર્ણાંક dst_stride_seq, તાર્કિક apply_silu,
    અચલ દશાંશ *bias
) {
    અચલ પૂર્ણાંક    lid = static_cast<પૂર્ણાંક>(it.get_local_id(0));
    અચલ size_t g   = it.get_group(0);
    અચલ પૂર્ણાંક    tt  = static_cast<પૂર્ણાંક>(g % nt_tiles);
    અચલ પૂર્ણાંક    ct  = static_cast<પૂર્ણાંક>((g / nt_tiles) % nc_tiles);
    અચલ પૂર્ણાંક    seq = static_cast<પૂર્ણાંક>(g / (static_cast<size_t>(nt_tiles) * nc_tiles));
    અચલ પૂર્ણાંક    t0 = tt * TT, c0 = ct * TC;

    અચલ પૂર્ણાંક ti = lid % TT;
    અચલ પૂર્ણાંક cj = lid / TT;
#pragma unroll
    માટે (પૂર્ણાંક r = 0; r < TC / (WG / TT); ++r) {
        અચલ પૂર્ણાંક c   = cj + r * (WG / TT);
        અચલ પૂર્ણાંક tok = t0 + ti;
        દશાંશ sumf = 0.0f;
        જો (tok < n_t) {
            અચલ દશાંશ *s = src_data + static_cast<size_t>(seq) * src_stride_seq
                           + static_cast<size_t>(c0 + c) * src_stride_inner + tok;
            અચલ દશાંશ *cw = weights + static_cast<size_t>(c0 + c) * DC;
#pragma unroll
            માટે (પૂર્ણાંક i = 0; i < DC; ++i) sumf += s[i] * cw[i];
            જો (bias != nullptr) sumf += bias[c0 + c];
            જો (apply_silu) sumf = op_silu(sumf);
        }
        tile[c * (TT + 1) + ti] = sumf;
    }
    it.barrier(access::fence_space::local_space);

    અચલ પૂર્ણાંક cc = lid % TC;
    અચલ પૂર્ણાંક tj = lid / TC;
#pragma unroll
    માટે (પૂર્ણાંક r = 0; r < TT / (WG / TC); ++r) {
        અચલ પૂર્ણાંક t   = tj + r * (WG / TC);
        અચલ પૂર્ણાંક tok = t0 + t;
        જો (tok < n_t) {
            dst_data[static_cast<size_t>(seq) * dst_stride_seq
                     + static_cast<size_t>(tok) * dst_stride_token + c0 + cc]
                = tile[cc * (TT + 1) + t];
        }
    }
}

// Same FUSED split as kernel_ssm_conv_impl. The fused instantiation keeps the runtime
// apply_silu/bias branches: at n_t >= 32 they are amortized over the whole tile.
template <પૂર્ણાંક DC, તાર્કિક FUSED>
સ્થિર શૂન્ય kernel_ssm_conv_tiled(
    queue &q, અચલ દશાંશ *src_data, અચલ દશાંશ *weights, દશાંશ *dst_data,
    પૂર્ણાંક d_inner, પૂર્ણાંક n_t, પૂર્ણાંક n_s, પૂર્ણાંક src_stride_inner, પૂર્ણાંક src_stride_seq,
    પૂર્ણાંક dst_stride_token, પૂર્ણાંક dst_stride_seq, તાર્કિક apply_silu, અચલ દશાંશ *bias
) {
    constexpr પૂર્ણાંક TT = 32, TC = 32, WG = 256;
    અચલ પૂર્ણાંક nt_tiles = (n_t + TT - 1) / TT;
    અચલ પૂર્ણાંક nc_tiles = d_inner / TC;
    અચલ size_t groups = static_cast<size_t>(nt_tiles) * nc_tiles * n_s;

    જો constexpr (FUSED) {
        q.submit([&](handler &h) {
            local_accessor<દશાંશ, 1> tile(range<1>(TC * (TT + 1)), h);
            h.parallel_for(nd_range<1>(range<1>(groups * WG), range<1>(WG)), [=](nd_item<1> it) {
                ssm_conv_tile<DC, TT, TC, WG>(it, tile, src_data, weights, dst_data, n_t, nt_tiles,
                                              nc_tiles, src_stride_inner, src_stride_seq,
                                              dst_stride_token, dst_stride_seq, apply_silu, bias);
            });
        });
    } નહીંતર {
        GGML_UNUSED(apply_silu);
        GGML_UNUSED(bias);

        q.submit([&](handler &h) {
            local_accessor<દશાંશ, 1> tile(range<1>(TC * (TT + 1)), h);
            h.parallel_for(nd_range<1>(range<1>(groups * WG), range<1>(WG)), [=](nd_item<1> it) {
                ssm_conv_tile<DC, TT, TC, WG>(it, tile, src_data, weights, dst_data, n_t, nt_tiles,
                                              nc_tiles, src_stride_inner, src_stride_seq,
                                              dst_stride_token, dst_stride_seq, ખોટું, nullptr);
            });
        });
    }
}

સ્થિર શૂન્ય kernel_ssm_conv(
    queue &q,
    અચલ દશાંશ *src_data,
    અચલ દશાંશ *weights,
    દશાંશ *dst_data,
    પૂર્ણાંક d_conv,
    પૂર્ણાંક d_inner,
    પૂર્ણાંક n_t,
    પૂર્ણાંક n_s,
    પૂર્ણાંક ncs,
    પૂર્ણાંક src_stride_inner,
    પૂર્ણાંક src_stride_seq,
    પૂર્ણાંક dst_stride_token,
    પૂર્ણાંક dst_stride_seq,
    તાર્કિક apply_silu,
    અચલ દશાંશ *bias
) {
    // Only the fused instantiations carry apply_silu/bias as kernel arguments; the plain
    // ssm_conv launch keeps the argument list it had before the fusion landed.
    અચલ તાર્કિક fused = apply_silu || bias != nullptr;

    // d_inner must be a multiple of 32 so the channel tiles are exact; the transpose is only
    // worth it for n_t >= 32. d_conv == 4 is the only window with a DC-specialized kernel.
    જો (d_conv == 4 && n_t >= 32 && (d_inner % 32) == 0) {
        જો (fused) {
            kernel_ssm_conv_tiled<4, સાચું>(q, src_data, weights, dst_data, d_inner, n_t, n_s,
                                           src_stride_inner, src_stride_seq, dst_stride_token,
                                           dst_stride_seq, apply_silu, bias);
        } નહીંતર {
            kernel_ssm_conv_tiled<4, ખોટું>(q, src_data, weights, dst_data, d_inner, n_t, n_s,
                                            src_stride_inner, src_stride_seq, dst_stride_token,
                                            dst_stride_seq, apply_silu, bias);
        }
        પાછા;
    }

    જો (d_conv == 4) {
        જો (fused) {
            kernel_ssm_conv_impl<4, સાચું>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                          ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                          dst_stride_seq, apply_silu, bias);
        } નહીંતર {
            kernel_ssm_conv_impl<4, ખોટું>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                           ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                           dst_stride_seq, apply_silu, bias);
        }
        પાછા;
    }

    જો (fused) {
        kernel_ssm_conv_impl<0, સાચું>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                      ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                      dst_stride_seq, apply_silu, bias);
    } નહીંતર {
        kernel_ssm_conv_impl<0, ખોટું>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                       ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                       dst_stride_seq, apply_silu, bias);
    }
}

અંતઃરેખ શૂન્ય ggml_sycl_op_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * silu_dst = nullptr, અચલ દશાંશ * bias = nullptr) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(bias == nullptr || silu_dst != nullptr);

    અચલ પૂર્ણાંક d_conv   = src1->ne[0];
    અચલ પૂર્ણાંક ncs      = src0->ne[0];
    અચલ પૂર્ણાંક d_inner  = src0->ne[1];
    અચલ પૂર્ણાંક n_t      = dst->ne[1];
    અચલ પૂર્ણાંક n_s      = dst->ne[2];

    GGML_ASSERT(src0->ne[0] == d_conv - 1 + n_t);
    GGML_ASSERT(src0->ne[1] == d_inner);
    GGML_ASSERT(src1->ne[1] == d_inner);

    GGML_ASSERT(dst->ne[0] == d_inner);
    GGML_ASSERT(dst->ne[1] == n_t);
    GGML_ASSERT(dst->ne[2] == n_s);

    GGML_ASSERT(src0->nb[0] == કદ(દશાંશ));
    GGML_ASSERT(src1->nb[0] == કદ(દશાંશ));

    GGML_ASSERT(src0->nb[1] == src0->ne[0] * કદ(દશાંશ));

    અચલ પૂર્ણાંક src_stride_inner = ncs;
    અચલ પૂર્ણાંક src_stride_seq   = ncs * d_inner;
    અચલ પૂર્ણાંક dst_stride_token = d_inner;
    અચલ પૂર્ણાંક dst_stride_seq   = d_inner * n_t;

    પ્રયાસ {
        queue *q = ctx.stream();

        અચલ દશાંશ *src_data = static_cast<અચલ દશાંશ *>(src0->data);
        અચલ દશાંશ *weights  = static_cast<અચલ દશાંશ *>(src1->data);
        અચલ તાર્કિક apply_silu = silu_dst != nullptr;
        દશાંશ *dst_data       = static_cast<દશાંશ *>((silu_dst ? silu_dst : dst)->data);

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

    } પકડો (અચલ std::exception &e) {
        std::fprintf(stderr, "[SYCL-SSM_CONV] ERROR: %s\n", e.what());
        ફેંકો;
    }
}

શૂન્ય ggml_sycl_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_ssm_conv(ctx, dst);
}

// Fused ssm_conv + ADD + SiLU: write silu(conv(x) + b) straight into silu_dst, eliding the
// standalone SiLU launch and its HBM round-trip of the conv output.
શૂન્ય ggml_sycl_ssm_conv_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * add, ggml_tensor * silu_dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    GGML_ASSERT(silu_dst && ggml_are_same_shape(dst, silu_dst) && silu_dst->type == GGML_TYPE_F32);
    // the fused kernel reads only the ADD's bias operand; the ADD result is never written
    અચલ દશાંશ * bias = nullptr;
    જો (add != nullptr) {
        અચલ ggml_tensor * bias_t = (add->src[0] == dst) ? add->src[1] : add->src[0];
        bias = static_cast<અચલ દશાંશ *>(bias_t->data);
    }
    ggml_sycl_op_ssm_conv(ctx, dst, silu_dst, bias);
}
