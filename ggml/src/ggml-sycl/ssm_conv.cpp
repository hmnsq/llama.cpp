#ଅନ୍ତର୍ଭାଵ "ssm_conv.hpp"
#ଅନ୍ତର୍ଭାଵ "common.hpp"
#ଅନ୍ତର୍ଭାଵ "element_wise.hpp"

#ଅନ୍ତର୍ଭାଵ <cstdio>

using ନାମାକାଶ sycl;

// One output element of the conv. DC is d_conv as a compile-time constant (0 keeps the
// runtime loop); unfused callers pass literal false/nullptr so the epilogue folds away.
template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା DC>
ସ୍ଥିର __dpct_inline__ ଶୂନ୍ୟ ssm_conv_element(
    size_t idx,
    ଅଚଳ ଦଶମିକ *src_data,
    ଅଚଳ ଦଶମିକ *weights,
    ଦଶମିକ *dst_data,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା d_conv,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା d_inner,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_t,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା src_stride_inner,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା src_stride_seq,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା dst_stride_token,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା dst_stride_seq,
    ତାର୍କିକ apply_silu,
    ଅଚଳ ଦଶମିକ *bias
) {
    // src is token-contiguous per channel, dst is channel-contiguous per token,
    // so indexing token-fastest coalesces the d_conv loads.
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା token   = static_cast<ପୂର୍ଣ୍ଣସଂଖ୍ୟା>(idx % n_t);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା channel = static_cast<ପୂର୍ଣ୍ଣସଂଖ୍ୟା>((idx / n_t) % d_inner);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା seq     = static_cast<ପୂର୍ଣ୍ଣସଂଖ୍ୟା>(idx / (static_cast<size_t>(n_t) * static_cast<size_t>(d_inner)));

    ଅଚଳ ଦଶମିକ *s = src_data
        + static_cast<size_t>(seq) * static_cast<size_t>(src_stride_seq)
        + static_cast<size_t>(channel) * static_cast<size_t>(src_stride_inner)
        + static_cast<size_t>(token);

    ଅଚଳ ଦଶମିକ *c = weights + static_cast<size_t>(channel) * static_cast<size_t>(d_conv);

    ଦଶମିକ sumf = 0.0f;
    ଯଦି constexpr (DC > 0) {
#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0 = 0; i0 < DC; ++i0) {
            sumf += s[i0] * c[i0];
        }
    } ଅନ୍ୟଥା {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0 = 0; i0 < d_conv; ++i0) {
            sumf += s[i0] * c[i0];
        }
    }

    // fused bias add: the ADD node broadcasts a 1-D channel bias over tokens
    ଯଦି (bias != nullptr) {
        sumf += bias[channel];
    }

    ଅଚଳ size_t dst_idx =
        static_cast<size_t>(seq) * static_cast<size_t>(dst_stride_seq) +
        static_cast<size_t>(token) * static_cast<size_t>(dst_stride_token) +
        static_cast<size_t>(channel);

    dst_data[dst_idx] = apply_silu ? op_silu(sumf) : sumf;
}

// FUSED=false keeps apply_silu/bias out of the kernel capture list, so the unfused launch
// takes the pre-fusion argument list; matters at n_t == 1, where the op is launch-bound.
template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା DC, ତାର୍କିକ FUSED>
ସ୍ଥିର ଶୂନ୍ୟ kernel_ssm_conv_impl(
    queue &q,
    ଅଚଳ ଦଶମିକ *src_data,
    ଅଚଳ ଦଶମିକ *weights,
    ଦଶମିକ *dst_data,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା d_conv,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା d_inner,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_t,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_s,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncs __attribute__((unused)),
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା src_stride_inner,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା src_stride_seq,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା dst_stride_token,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା dst_stride_seq,
    ତାର୍କିକ apply_silu,
    ଅଚଳ ଦଶମିକ *bias
) {
    ଅଚଳ size_t total_work = static_cast<size_t>(d_inner) * static_cast<size_t>(n_t) * static_cast<size_t>(n_s);
    ଅଚଳ size_t work_group_size = 256;
    ଅଚଳ size_t ଅଙ୍କ_work_groups = (total_work + work_group_size - 1) / work_group_size;

    ଅଚଳ range<1> global_range(ଅଙ୍କ_work_groups * work_group_size);
    ଅଚଳ range<1> local_range(work_group_size);

    ଯଦି constexpr (FUSED) {
        q.submit([&](handler &h) {
            h.parallel_for(
                nd_range<1>(global_range, local_range),
                [=](nd_item<1> item) {
                    ଅଚଳ size_t idx = item.get_global_id(0);
                    ଯଦି (idx >= total_work) {
                        ଫେରନ୍ତୁ;
                    }

                    ssm_conv_element<DC>(idx, src_data, weights, dst_data, d_conv, d_inner, n_t,
                                         src_stride_inner, src_stride_seq, dst_stride_token,
                                         dst_stride_seq, apply_silu, bias);
                }
            );
        });
    } ଅନ୍ୟଥା {
        GGML_UNUSED(apply_silu);
        GGML_UNUSED(bias);

        q.submit([&](handler &h) {
            h.parallel_for(
                nd_range<1>(global_range, local_range),
                [=](nd_item<1> item) {
                    ଅଚଳ size_t idx = item.get_global_id(0);
                    ଯଦି (idx >= total_work) {
                        ଫେରନ୍ତୁ;
                    }

                    ssm_conv_element<DC>(idx, src_data, weights, dst_data, d_conv, d_inner, n_t,
                                         src_stride_inner, src_stride_seq, dst_stride_token,
                                         dst_stride_seq, ମିଥ୍ୟା, nullptr);
                }
            );
        });
    }
}

// SLM transpose tile: coalesces both the loads and the stores. The +1 pad makes the row
// stride 33, coprime with 32 banks, so both phases are bank-conflict-free.
template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା DC, ପୂର୍ଣ୍ଣସଂଖ୍ୟା TT, ପୂର୍ଣ୍ଣସଂଖ୍ୟା TC, ପୂର୍ଣ୍ଣସଂଖ୍ୟା WG>
ସ୍ଥିର __dpct_inline__ ଶୂନ୍ୟ ssm_conv_tile(
    nd_item<1> it, local_accessor<ଦଶମିକ, 1> tile, ଅଚଳ ଦଶମିକ *src_data, ଅଚଳ ଦଶମିକ *weights,
    ଦଶମିକ *dst_data, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_t, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nt_tiles, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nc_tiles, ପୂର୍ଣ୍ଣସଂଖ୍ୟା src_stride_inner,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା src_stride_seq, ପୂର୍ଣ୍ଣସଂଖ୍ୟା dst_stride_token, ପୂର୍ଣ୍ଣସଂଖ୍ୟା dst_stride_seq, ତାର୍କିକ apply_silu,
    ଅଚଳ ଦଶମିକ *bias
) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା    lid = static_cast<ପୂର୍ଣ୍ଣସଂଖ୍ୟା>(it.get_local_id(0));
    ଅଚଳ size_t g   = it.get_group(0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା    tt  = static_cast<ପୂର୍ଣ୍ଣସଂଖ୍ୟା>(g % nt_tiles);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା    ct  = static_cast<ପୂର୍ଣ୍ଣସଂଖ୍ୟା>((g / nt_tiles) % nc_tiles);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା    seq = static_cast<ପୂର୍ଣ୍ଣସଂଖ୍ୟା>(g / (static_cast<size_t>(nt_tiles) * nc_tiles));
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା    t0 = tt * TT, c0 = ct * TC;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ti = lid % TT;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା cj = lid / TT;
#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା r = 0; r < TC / (WG / TT); ++r) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା c   = cj + r * (WG / TT);
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା tok = t0 + ti;
        ଦଶମିକ sumf = 0.0f;
        ଯଦି (tok < n_t) {
            ଅଚଳ ଦଶମିକ *s = src_data + static_cast<size_t>(seq) * src_stride_seq
                           + static_cast<size_t>(c0 + c) * src_stride_inner + tok;
            ଅଚଳ ଦଶମିକ *cw = weights + static_cast<size_t>(c0 + c) * DC;
#pragma unroll
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < DC; ++i) sumf += s[i] * cw[i];
            ଯଦି (bias != nullptr) sumf += bias[c0 + c];
            ଯଦି (apply_silu) sumf = op_silu(sumf);
        }
        tile[c * (TT + 1) + ti] = sumf;
    }
    it.barrier(access::fence_space::local_space);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା cc = lid % TC;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା tj = lid / TC;
#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା r = 0; r < TT / (WG / TC); ++r) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା t   = tj + r * (WG / TC);
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା tok = t0 + t;
        ଯଦି (tok < n_t) {
            dst_data[static_cast<size_t>(seq) * dst_stride_seq
                     + static_cast<size_t>(tok) * dst_stride_token + c0 + cc]
                = tile[cc * (TT + 1) + t];
        }
    }
}

// Same FUSED split as kernel_ssm_conv_impl. The fused instantiation keeps the runtime
// apply_silu/bias branches: at n_t >= 32 they are amortized over the whole tile.
template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା DC, ତାର୍କିକ FUSED>
ସ୍ଥିର ଶୂନ୍ୟ kernel_ssm_conv_tiled(
    queue &q, ଅଚଳ ଦଶମିକ *src_data, ଅଚଳ ଦଶମିକ *weights, ଦଶମିକ *dst_data,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା d_inner, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_t, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_s, ପୂର୍ଣ୍ଣସଂଖ୍ୟା src_stride_inner, ପୂର୍ଣ୍ଣସଂଖ୍ୟା src_stride_seq,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା dst_stride_token, ପୂର୍ଣ୍ଣସଂଖ୍ୟା dst_stride_seq, ତାର୍କିକ apply_silu, ଅଚଳ ଦଶମିକ *bias
) {
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା TT = 32, TC = 32, WG = 256;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nt_tiles = (n_t + TT - 1) / TT;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nc_tiles = d_inner / TC;
    ଅଚଳ size_t groups = static_cast<size_t>(nt_tiles) * nc_tiles * n_s;

    ଯଦି constexpr (FUSED) {
        q.submit([&](handler &h) {
            local_accessor<ଦଶମିକ, 1> tile(range<1>(TC * (TT + 1)), h);
            h.parallel_for(nd_range<1>(range<1>(groups * WG), range<1>(WG)), [=](nd_item<1> it) {
                ssm_conv_tile<DC, TT, TC, WG>(it, tile, src_data, weights, dst_data, n_t, nt_tiles,
                                              nc_tiles, src_stride_inner, src_stride_seq,
                                              dst_stride_token, dst_stride_seq, apply_silu, bias);
            });
        });
    } ଅନ୍ୟଥା {
        GGML_UNUSED(apply_silu);
        GGML_UNUSED(bias);

        q.submit([&](handler &h) {
            local_accessor<ଦଶମିକ, 1> tile(range<1>(TC * (TT + 1)), h);
            h.parallel_for(nd_range<1>(range<1>(groups * WG), range<1>(WG)), [=](nd_item<1> it) {
                ssm_conv_tile<DC, TT, TC, WG>(it, tile, src_data, weights, dst_data, n_t, nt_tiles,
                                              nc_tiles, src_stride_inner, src_stride_seq,
                                              dst_stride_token, dst_stride_seq, ମିଥ୍ୟା, nullptr);
            });
        });
    }
}

ସ୍ଥିର ଶୂନ୍ୟ kernel_ssm_conv(
    queue &q,
    ଅଚଳ ଦଶମିକ *src_data,
    ଅଚଳ ଦଶମିକ *weights,
    ଦଶମିକ *dst_data,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା d_conv,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା d_inner,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_t,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_s,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncs,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା src_stride_inner,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା src_stride_seq,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା dst_stride_token,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା dst_stride_seq,
    ତାର୍କିକ apply_silu,
    ଅଚଳ ଦଶମିକ *bias
) {
    // Only the fused instantiations carry apply_silu/bias as kernel arguments; the plain
    // ssm_conv launch keeps the argument list it had before the fusion landed.
    ଅଚଳ ତାର୍କିକ fused = apply_silu || bias != nullptr;

    // d_inner must be a multiple of 32 so the channel tiles are exact; the transpose is only
    // worth it for n_t >= 32. d_conv == 4 is the only window with a DC-specialized kernel.
    ଯଦି (d_conv == 4 && n_t >= 32 && (d_inner % 32) == 0) {
        ଯଦି (fused) {
            kernel_ssm_conv_tiled<4, ସତ୍ୟ>(q, src_data, weights, dst_data, d_inner, n_t, n_s,
                                           src_stride_inner, src_stride_seq, dst_stride_token,
                                           dst_stride_seq, apply_silu, bias);
        } ଅନ୍ୟଥା {
            kernel_ssm_conv_tiled<4, ମିଥ୍ୟା>(q, src_data, weights, dst_data, d_inner, n_t, n_s,
                                            src_stride_inner, src_stride_seq, dst_stride_token,
                                            dst_stride_seq, apply_silu, bias);
        }
        ଫେରନ୍ତୁ;
    }

    ଯଦି (d_conv == 4) {
        ଯଦି (fused) {
            kernel_ssm_conv_impl<4, ସତ୍ୟ>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                          ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                          dst_stride_seq, apply_silu, bias);
        } ଅନ୍ୟଥା {
            kernel_ssm_conv_impl<4, ମିଥ୍ୟା>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                           ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                           dst_stride_seq, apply_silu, bias);
        }
        ଫେରନ୍ତୁ;
    }

    ଯଦି (fused) {
        kernel_ssm_conv_impl<0, ସତ୍ୟ>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                      ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                      dst_stride_seq, apply_silu, bias);
    } ଅନ୍ୟଥା {
        kernel_ssm_conv_impl<0, ମିଥ୍ୟା>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                       ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                       dst_stride_seq, apply_silu, bias);
    }
}

ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * silu_dst = nullptr, ଅଚଳ ଦଶମିକ * bias = nullptr) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(bias == nullptr || silu_dst != nullptr);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା d_conv   = src1->ne[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncs      = src0->ne[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା d_inner  = src0->ne[1];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_t      = dst->ne[1];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_s      = dst->ne[2];

    GGML_ASSERT(src0->ne[0] == d_conv - 1 + n_t);
    GGML_ASSERT(src0->ne[1] == d_inner);
    GGML_ASSERT(src1->ne[1] == d_inner);

    GGML_ASSERT(dst->ne[0] == d_inner);
    GGML_ASSERT(dst->ne[1] == n_t);
    GGML_ASSERT(dst->ne[2] == n_s);

    GGML_ASSERT(src0->nb[0] == ଆକାର(ଦଶମିକ));
    GGML_ASSERT(src1->nb[0] == ଆକାର(ଦଶମିକ));

    GGML_ASSERT(src0->nb[1] == src0->ne[0] * ଆକାର(ଦଶମିକ));

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା src_stride_inner = ncs;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା src_stride_seq   = ncs * d_inner;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dst_stride_token = d_inner;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dst_stride_seq   = d_inner * n_t;

    ଚେଷ୍ଟା {
        queue *q = ctx.stream();

        ଅଚଳ ଦଶମିକ *src_data = static_cast<ଅଚଳ ଦଶମିକ *>(src0->data);
        ଅଚଳ ଦଶମିକ *weights  = static_cast<ଅଚଳ ଦଶମିକ *>(src1->data);
        ଅଚଳ ତାର୍କିକ apply_silu = silu_dst != nullptr;
        ଦଶମିକ *dst_data       = static_cast<ଦଶମିକ *>((silu_dst ? silu_dst : dst)->data);

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

    } ଧର (ଅଚଳ std::exception &e) {
        std::fprintf(stderr, "[SYCL-SSM_CONV] ERROR: %s\n", e.what());
        ଫିଙ୍ଗ;
    }
}

ଶୂନ୍ୟ ggml_sycl_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_ssm_conv(ctx, dst);
}

// Fused ssm_conv + ADD + SiLU: write silu(conv(x) + b) straight into silu_dst, eliding the
// standalone SiLU launch and its HBM round-trip of the conv output.
ଶୂନ୍ୟ ggml_sycl_ssm_conv_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * add, ggml_tensor * silu_dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    GGML_ASSERT(silu_dst && ggml_are_same_shape(dst, silu_dst) && silu_dst->type == GGML_TYPE_F32);
    // the fused kernel reads only the ADD's bias operand; the ADD result is never written
    ଅଚଳ ଦଶମିକ * bias = nullptr;
    ଯଦି (add != nullptr) {
        ଅଚଳ ggml_tensor * bias_t = (add->src[0] == dst) ? add->src[1] : add->src[0];
        bias = static_cast<ଅଚଳ ଦଶମିକ *>(bias_t->data);
    }
    ggml_sycl_op_ssm_conv(ctx, dst, silu_dst, bias);
}
