#অন্তর্ভাব "ssm_conv.hpp"
#অন্তর্ভাব "common.hpp"
#অন্তর্ভাব "element_wise.hpp"

#অন্তর্ভাব <cstdio>

using নামাকাশ sycl;

// One output element of the conv. DC is d_conv as a compile-time constant (0 keeps the
// runtime loop); unfused callers pass literal false/nullptr so the epilogue folds away.
template <পূর্ণাঙ্ক DC>
স্থির __dpct_inline__ শূন্য ssm_conv_element(
    size_t idx,
    ধ্রুব ভিন্ন *src_data,
    ধ্রুব ভিন্ন *weights,
    ভিন্ন *dst_data,
    পূর্ণাঙ্ক d_conv,
    পূর্ণাঙ্ক d_inner,
    পূর্ণাঙ্ক n_t,
    পূর্ণাঙ্ক src_stride_inner,
    পূর্ণাঙ্ক src_stride_seq,
    পূর্ণাঙ্ক dst_stride_token,
    পূর্ণাঙ্ক dst_stride_seq,
    তার্কিক apply_silu,
    ধ্রুব ভিন্ন *bias
) {
    // src is token-contiguous per channel, dst is channel-contiguous per token,
    // so indexing token-fastest coalesces the d_conv loads.
    ধ্রুব পূর্ণাঙ্ক token   = static_cast<পূর্ণাঙ্ক>(idx % n_t);
    ধ্রুব পূর্ণাঙ্ক channel = static_cast<পূর্ণাঙ্ক>((idx / n_t) % d_inner);
    ধ্রুব পূর্ণাঙ্ক seq     = static_cast<পূর্ণাঙ্ক>(idx / (static_cast<size_t>(n_t) * static_cast<size_t>(d_inner)));

    ধ্রুব ভিন্ন *s = src_data
        + static_cast<size_t>(seq) * static_cast<size_t>(src_stride_seq)
        + static_cast<size_t>(channel) * static_cast<size_t>(src_stride_inner)
        + static_cast<size_t>(token);

    ধ্রুব ভিন্ন *c = weights + static_cast<size_t>(channel) * static_cast<size_t>(d_conv);

    ভিন্ন sumf = 0.0f;
    যদি constexpr (DC > 0) {
#pragma unroll
        প্রতি (পূর্ণাঙ্ক i0 = 0; i0 < DC; ++i0) {
            sumf += s[i0] * c[i0];
        }
    } অন্যথা {
        প্রতি (পূর্ণাঙ্ক i0 = 0; i0 < d_conv; ++i0) {
            sumf += s[i0] * c[i0];
        }
    }

    // fused bias add: the ADD node broadcasts a 1-D channel bias over tokens
    যদি (bias != nullptr) {
        sumf += bias[channel];
    }

    ধ্রুব size_t dst_idx =
        static_cast<size_t>(seq) * static_cast<size_t>(dst_stride_seq) +
        static_cast<size_t>(token) * static_cast<size_t>(dst_stride_token) +
        static_cast<size_t>(channel);

    dst_data[dst_idx] = apply_silu ? op_silu(sumf) : sumf;
}

// FUSED=false keeps apply_silu/bias out of the kernel capture list, so the unfused launch
// takes the pre-fusion argument list; matters at n_t == 1, where the op is launch-bound.
template <পূর্ণাঙ্ক DC, তার্কিক FUSED>
স্থির শূন্য kernel_ssm_conv_impl(
    queue &q,
    ধ্রুব ভিন্ন *src_data,
    ধ্রুব ভিন্ন *weights,
    ভিন্ন *dst_data,
    পূর্ণাঙ্ক d_conv,
    পূর্ণাঙ্ক d_inner,
    পূর্ণাঙ্ক n_t,
    পূর্ণাঙ্ক n_s,
    পূর্ণাঙ্ক ncs __attribute__((unused)),
    পূর্ণাঙ্ক src_stride_inner,
    পূর্ণাঙ্ক src_stride_seq,
    পূর্ণাঙ্ক dst_stride_token,
    পূর্ণাঙ্ক dst_stride_seq,
    তার্কিক apply_silu,
    ধ্রুব ভিন্ন *bias
) {
    ধ্রুব size_t total_work = static_cast<size_t>(d_inner) * static_cast<size_t>(n_t) * static_cast<size_t>(n_s);
    ধ্রুব size_t work_group_size = 256;
    ধ্রুব size_t অঙ্ক_work_groups = (total_work + work_group_size - 1) / work_group_size;

    ধ্রুব range<1> global_range(অঙ্ক_work_groups * work_group_size);
    ধ্রুব range<1> local_range(work_group_size);

    যদি constexpr (FUSED) {
        q.submit([&](handler &h) {
            h.parallel_for(
                nd_range<1>(global_range, local_range),
                [=](nd_item<1> item) {
                    ধ্রুব size_t idx = item.get_global_id(0);
                    যদি (idx >= total_work) {
                        প্রত্যাগম;
                    }

                    ssm_conv_element<DC>(idx, src_data, weights, dst_data, d_conv, d_inner, n_t,
                                         src_stride_inner, src_stride_seq, dst_stride_token,
                                         dst_stride_seq, apply_silu, bias);
                }
            );
        });
    } অন্যথা {
        GGML_UNUSED(apply_silu);
        GGML_UNUSED(bias);

        q.submit([&](handler &h) {
            h.parallel_for(
                nd_range<1>(global_range, local_range),
                [=](nd_item<1> item) {
                    ধ্রুব size_t idx = item.get_global_id(0);
                    যদি (idx >= total_work) {
                        প্রত্যাগম;
                    }

                    ssm_conv_element<DC>(idx, src_data, weights, dst_data, d_conv, d_inner, n_t,
                                         src_stride_inner, src_stride_seq, dst_stride_token,
                                         dst_stride_seq, অসত্য, nullptr);
                }
            );
        });
    }
}

// SLM transpose tile: coalesces both the loads and the stores. The +1 pad makes the row
// stride 33, coprime with 32 banks, so both phases are bank-conflict-free.
template <পূর্ণাঙ্ক DC, পূর্ণাঙ্ক TT, পূর্ণাঙ্ক TC, পূর্ণাঙ্ক WG>
স্থির __dpct_inline__ শূন্য ssm_conv_tile(
    nd_item<1> it, local_accessor<ভিন্ন, 1> tile, ধ্রুব ভিন্ন *src_data, ধ্রুব ভিন্ন *weights,
    ভিন্ন *dst_data, পূর্ণাঙ্ক n_t, পূর্ণাঙ্ক nt_tiles, পূর্ণাঙ্ক nc_tiles, পূর্ণাঙ্ক src_stride_inner,
    পূর্ণাঙ্ক src_stride_seq, পূর্ণাঙ্ক dst_stride_token, পূর্ণাঙ্ক dst_stride_seq, তার্কিক apply_silu,
    ধ্রুব ভিন্ন *bias
) {
    ধ্রুব পূর্ণাঙ্ক    lid = static_cast<পূর্ণাঙ্ক>(it.get_local_id(0));
    ধ্রুব size_t g   = it.get_group(0);
    ধ্রুব পূর্ণাঙ্ক    tt  = static_cast<পূর্ণাঙ্ক>(g % nt_tiles);
    ধ্রুব পূর্ণাঙ্ক    ct  = static_cast<পূর্ণাঙ্ক>((g / nt_tiles) % nc_tiles);
    ধ্রুব পূর্ণাঙ্ক    seq = static_cast<পূর্ণাঙ্ক>(g / (static_cast<size_t>(nt_tiles) * nc_tiles));
    ধ্রুব পূর্ণাঙ্ক    t0 = tt * TT, c0 = ct * TC;

    ধ্রুব পূর্ণাঙ্ক ti = lid % TT;
    ধ্রুব পূর্ণাঙ্ক cj = lid / TT;
#pragma unroll
    প্রতি (পূর্ণাঙ্ক r = 0; r < TC / (WG / TT); ++r) {
        ধ্রুব পূর্ণাঙ্ক c   = cj + r * (WG / TT);
        ধ্রুব পূর্ণাঙ্ক tok = t0 + ti;
        ভিন্ন sumf = 0.0f;
        যদি (tok < n_t) {
            ধ্রুব ভিন্ন *s = src_data + static_cast<size_t>(seq) * src_stride_seq
                           + static_cast<size_t>(c0 + c) * src_stride_inner + tok;
            ধ্রুব ভিন্ন *cw = weights + static_cast<size_t>(c0 + c) * DC;
#pragma unroll
            প্রতি (পূর্ণাঙ্ক i = 0; i < DC; ++i) sumf += s[i] * cw[i];
            যদি (bias != nullptr) sumf += bias[c0 + c];
            যদি (apply_silu) sumf = op_silu(sumf);
        }
        tile[c * (TT + 1) + ti] = sumf;
    }
    it.barrier(access::fence_space::local_space);

    ধ্রুব পূর্ণাঙ্ক cc = lid % TC;
    ধ্রুব পূর্ণাঙ্ক tj = lid / TC;
#pragma unroll
    প্রতি (পূর্ণাঙ্ক r = 0; r < TT / (WG / TC); ++r) {
        ধ্রুব পূর্ণাঙ্ক t   = tj + r * (WG / TC);
        ধ্রুব পূর্ণাঙ্ক tok = t0 + t;
        যদি (tok < n_t) {
            dst_data[static_cast<size_t>(seq) * dst_stride_seq
                     + static_cast<size_t>(tok) * dst_stride_token + c0 + cc]
                = tile[cc * (TT + 1) + t];
        }
    }
}

// Same FUSED split as kernel_ssm_conv_impl. The fused instantiation keeps the runtime
// apply_silu/bias branches: at n_t >= 32 they are amortized over the whole tile.
template <পূর্ণাঙ্ক DC, তার্কিক FUSED>
স্থির শূন্য kernel_ssm_conv_tiled(
    queue &q, ধ্রুব ভিন্ন *src_data, ধ্রুব ভিন্ন *weights, ভিন্ন *dst_data,
    পূর্ণাঙ্ক d_inner, পূর্ণাঙ্ক n_t, পূর্ণাঙ্ক n_s, পূর্ণাঙ্ক src_stride_inner, পূর্ণাঙ্ক src_stride_seq,
    পূর্ণাঙ্ক dst_stride_token, পূর্ণাঙ্ক dst_stride_seq, তার্কিক apply_silu, ধ্রুব ভিন্ন *bias
) {
    constexpr পূর্ণাঙ্ক TT = 32, TC = 32, WG = 256;
    ধ্রুব পূর্ণাঙ্ক nt_tiles = (n_t + TT - 1) / TT;
    ধ্রুব পূর্ণাঙ্ক nc_tiles = d_inner / TC;
    ধ্রুব size_t groups = static_cast<size_t>(nt_tiles) * nc_tiles * n_s;

    যদি constexpr (FUSED) {
        q.submit([&](handler &h) {
            local_accessor<ভিন্ন, 1> tile(range<1>(TC * (TT + 1)), h);
            h.parallel_for(nd_range<1>(range<1>(groups * WG), range<1>(WG)), [=](nd_item<1> it) {
                ssm_conv_tile<DC, TT, TC, WG>(it, tile, src_data, weights, dst_data, n_t, nt_tiles,
                                              nc_tiles, src_stride_inner, src_stride_seq,
                                              dst_stride_token, dst_stride_seq, apply_silu, bias);
            });
        });
    } অন্যথা {
        GGML_UNUSED(apply_silu);
        GGML_UNUSED(bias);

        q.submit([&](handler &h) {
            local_accessor<ভিন্ন, 1> tile(range<1>(TC * (TT + 1)), h);
            h.parallel_for(nd_range<1>(range<1>(groups * WG), range<1>(WG)), [=](nd_item<1> it) {
                ssm_conv_tile<DC, TT, TC, WG>(it, tile, src_data, weights, dst_data, n_t, nt_tiles,
                                              nc_tiles, src_stride_inner, src_stride_seq,
                                              dst_stride_token, dst_stride_seq, অসত্য, nullptr);
            });
        });
    }
}

স্থির শূন্য kernel_ssm_conv(
    queue &q,
    ধ্রুব ভিন্ন *src_data,
    ধ্রুব ভিন্ন *weights,
    ভিন্ন *dst_data,
    পূর্ণাঙ্ক d_conv,
    পূর্ণাঙ্ক d_inner,
    পূর্ণাঙ্ক n_t,
    পূর্ণাঙ্ক n_s,
    পূর্ণাঙ্ক ncs,
    পূর্ণাঙ্ক src_stride_inner,
    পূর্ণাঙ্ক src_stride_seq,
    পূর্ণাঙ্ক dst_stride_token,
    পূর্ণাঙ্ক dst_stride_seq,
    তার্কিক apply_silu,
    ধ্রুব ভিন্ন *bias
) {
    // Only the fused instantiations carry apply_silu/bias as kernel arguments; the plain
    // ssm_conv launch keeps the argument list it had before the fusion landed.
    ধ্রুব তার্কিক fused = apply_silu || bias != nullptr;

    // d_inner must be a multiple of 32 so the channel tiles are exact; the transpose is only
    // worth it for n_t >= 32. d_conv == 4 is the only window with a DC-specialized kernel.
    যদি (d_conv == 4 && n_t >= 32 && (d_inner % 32) == 0) {
        যদি (fused) {
            kernel_ssm_conv_tiled<4, সত্য>(q, src_data, weights, dst_data, d_inner, n_t, n_s,
                                           src_stride_inner, src_stride_seq, dst_stride_token,
                                           dst_stride_seq, apply_silu, bias);
        } অন্যথা {
            kernel_ssm_conv_tiled<4, অসত্য>(q, src_data, weights, dst_data, d_inner, n_t, n_s,
                                            src_stride_inner, src_stride_seq, dst_stride_token,
                                            dst_stride_seq, apply_silu, bias);
        }
        প্রত্যাগম;
    }

    যদি (d_conv == 4) {
        যদি (fused) {
            kernel_ssm_conv_impl<4, সত্য>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                          ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                          dst_stride_seq, apply_silu, bias);
        } অন্যথা {
            kernel_ssm_conv_impl<4, অসত্য>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                           ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                           dst_stride_seq, apply_silu, bias);
        }
        প্রত্যাগম;
    }

    যদি (fused) {
        kernel_ssm_conv_impl<0, সত্য>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                      ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                      dst_stride_seq, apply_silu, bias);
    } অন্যথা {
        kernel_ssm_conv_impl<0, অসত্য>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                       ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                       dst_stride_seq, apply_silu, bias);
    }
}

অন্তঃস্থ শূন্য ggml_sycl_op_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * silu_dst = nullptr, ধ্রুব ভিন্ন * bias = nullptr) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(bias == nullptr || silu_dst != nullptr);

    ধ্রুব পূর্ণাঙ্ক d_conv   = src1->ne[0];
    ধ্রুব পূর্ণাঙ্ক ncs      = src0->ne[0];
    ধ্রুব পূর্ণাঙ্ক d_inner  = src0->ne[1];
    ধ্রুব পূর্ণাঙ্ক n_t      = dst->ne[1];
    ধ্রুব পূর্ণাঙ্ক n_s      = dst->ne[2];

    GGML_ASSERT(src0->ne[0] == d_conv - 1 + n_t);
    GGML_ASSERT(src0->ne[1] == d_inner);
    GGML_ASSERT(src1->ne[1] == d_inner);

    GGML_ASSERT(dst->ne[0] == d_inner);
    GGML_ASSERT(dst->ne[1] == n_t);
    GGML_ASSERT(dst->ne[2] == n_s);

    GGML_ASSERT(src0->nb[0] == পরিমাণ(ভিন্ন));
    GGML_ASSERT(src1->nb[0] == পরিমাণ(ভিন্ন));

    GGML_ASSERT(src0->nb[1] == src0->ne[0] * পরিমাণ(ভিন্ন));

    ধ্রুব পূর্ণাঙ্ক src_stride_inner = ncs;
    ধ্রুব পূর্ণাঙ্ক src_stride_seq   = ncs * d_inner;
    ধ্রুব পূর্ণাঙ্ক dst_stride_token = d_inner;
    ধ্রুব পূর্ণাঙ্ক dst_stride_seq   = d_inner * n_t;

    যত্ন {
        queue *q = ctx.stream();

        ধ্রুব ভিন্ন *src_data = static_cast<ধ্রুব ভিন্ন *>(src0->data);
        ধ্রুব ভিন্ন *weights  = static_cast<ধ্রুব ভিন্ন *>(src1->data);
        ধ্রুব তার্কিক apply_silu = silu_dst != nullptr;
        ভিন্ন *dst_data       = static_cast<ভিন্ন *>((silu_dst ? silu_dst : dst)->data);

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

    } গৃহাণ (ধ্রুব std::exception &e) {
        std::fprintf(stderr, "[SYCL-SSM_CONV] ERROR: %s\n", e.what());
        ক্ষিপ;
    }
}

শূন্য ggml_sycl_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_ssm_conv(ctx, dst);
}

// Fused ssm_conv + ADD + SiLU: write silu(conv(x) + b) straight into silu_dst, eliding the
// standalone SiLU launch and its HBM round-trip of the conv output.
শূন্য ggml_sycl_ssm_conv_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * add, ggml_tensor * silu_dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    GGML_ASSERT(silu_dst && ggml_are_same_shape(dst, silu_dst) && silu_dst->type == GGML_TYPE_F32);
    // the fused kernel reads only the ADD's bias operand; the ADD result is never written
    ধ্রুব ভিন্ন * bias = nullptr;
    যদি (add != nullptr) {
        ধ্রুব ggml_tensor * bias_t = (add->src[0] == dst) ? add->src[1] : add->src[0];
        bias = static_cast<ধ্রুব ভিন্ন *>(bias_t->data);
    }
    ggml_sycl_op_ssm_conv(ctx, dst, silu_dst, bias);
}
