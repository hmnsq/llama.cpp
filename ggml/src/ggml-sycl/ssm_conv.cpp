#অন্তর্ভাব "ssm_conv.hpp"
#অন্তর্ভাব "common.hpp"
#অন্তর্ভাব "element_wise.hpp"

#অন্তর্ভাব <cstdio>

using নামাকাশ sycl;

// One output element of the conv. DC is d_conv as a compile-time constant (0 keeps the
// runtime loop); unfused callers pass literal false/nullptr so the epilogue folds away.
template <পূর্ণসংখ্যা DC>
স্থির __dpct_inline__ শূন্য ssm_conv_element(
    size_t idx,
    ধ্রুবক দশমিক *src_তথ্য,
    ধ্রুবক দশমিক *weights,
    দশমিক *dst_তথ্য,
    পূর্ণসংখ্যা d_conv,
    পূর্ণসংখ্যা d_inner,
    পূর্ণসংখ্যা n_t,
    পূর্ণসংখ্যা src_stride_inner,
    পূর্ণসংখ্যা src_stride_seq,
    পূর্ণসংখ্যা dst_stride_token,
    পূর্ণসংখ্যা dst_stride_seq,
    যৌক্তিক apply_silu,
    ধ্রুবক দশমিক *bias
) {
    // src is token-contiguous per channel, dst is channel-contiguous per token,
    // so indexing token-fastest coalesces the d_conv loads.
    ধ্রুবক পূর্ণসংখ্যা token   = static_cast<পূর্ণসংখ্যা>(idx % n_t);
    ধ্রুবক পূর্ণসংখ্যা channel = static_cast<পূর্ণসংখ্যা>((idx / n_t) % d_inner);
    ধ্রুবক পূর্ণসংখ্যা seq     = static_cast<পূর্ণসংখ্যা>(idx / (static_cast<size_t>(n_t) * static_cast<size_t>(d_inner)));

    ধ্রুবক দশমিক *s = src_তথ্য
        + static_cast<size_t>(seq) * static_cast<size_t>(src_stride_seq)
        + static_cast<size_t>(channel) * static_cast<size_t>(src_stride_inner)
        + static_cast<size_t>(token);

    ধ্রুবক দশমিক *c = weights + static_cast<size_t>(channel) * static_cast<size_t>(d_conv);

    দশমিক sumf = 0.0f;
    যদি constexpr (DC > 0) {
#pragma unroll
        জন্য (পূর্ণসংখ্যা i0 = 0; i0 < DC; ++i0) {
            sumf += s[i0] * c[i0];
        }
    } অন্যথা {
        জন্য (পূর্ণসংখ্যা i0 = 0; i0 < d_conv; ++i0) {
            sumf += s[i0] * c[i0];
        }
    }

    // fused bias add: the ADD node broadcasts a 1-D channel bias over tokens
    যদি (bias != nullptr) {
        sumf += bias[channel];
    }

    ধ্রুবক size_t dst_idx =
        static_cast<size_t>(seq) * static_cast<size_t>(dst_stride_seq) +
        static_cast<size_t>(token) * static_cast<size_t>(dst_stride_token) +
        static_cast<size_t>(channel);

    dst_তথ্য[dst_idx] = apply_silu ? op_silu(sumf) : sumf;
}

// FUSED=false keeps apply_silu/bias out of the kernel capture list, so the unfused launch
// takes the pre-fusion argument list; matters at n_t == 1, where the op is launch-bound.
template <পূর্ণসংখ্যা DC, যৌক্তিক FUSED>
স্থির শূন্য kernel_ssm_conv_impl(
    queue &q,
    ধ্রুবক দশমিক *src_তথ্য,
    ধ্রুবক দশমিক *weights,
    দশমিক *dst_তথ্য,
    পূর্ণসংখ্যা d_conv,
    পূর্ণসংখ্যা d_inner,
    পূর্ণসংখ্যা n_t,
    পূর্ণসংখ্যা n_s,
    পূর্ণসংখ্যা ncs __attribute__((unused)),
    পূর্ণসংখ্যা src_stride_inner,
    পূর্ণসংখ্যা src_stride_seq,
    পূর্ণসংখ্যা dst_stride_token,
    পূর্ণসংখ্যা dst_stride_seq,
    যৌক্তিক apply_silu,
    ধ্রুবক দশমিক *bias
) {
    ধ্রুবক size_t যোগ_work = static_cast<size_t>(d_inner) * static_cast<size_t>(n_t) * static_cast<size_t>(n_s);
    ধ্রুবক size_t work_group_আকার = 256;
    ধ্রুবক size_t অঙ্ক_work_groups = (যোগ_work + work_group_আকার - 1) / work_group_আকার;

    ধ্রুবক range<1> global_range(অঙ্ক_work_groups * work_group_আকার);
    ধ্রুবক range<1> local_range(work_group_আকার);

    যদি constexpr (FUSED) {
        q.submit([&](handler &h) {
            h.parallel_for(
                nd_range<1>(global_range, local_range),
                [=](nd_item<1> item) {
                    ধ্রুবক size_t idx = item.get_global_id(0);
                    যদি (idx >= যোগ_work) {
                        ফেরত;
                    }

                    ssm_conv_element<DC>(idx, src_তথ্য, weights, dst_তথ্য, d_conv, d_inner, n_t,
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
                    ধ্রুবক size_t idx = item.get_global_id(0);
                    যদি (idx >= যোগ_work) {
                        ফেরত;
                    }

                    ssm_conv_element<DC>(idx, src_তথ্য, weights, dst_তথ্য, d_conv, d_inner, n_t,
                                         src_stride_inner, src_stride_seq, dst_stride_token,
                                         dst_stride_seq, মিথ্যা, nullptr);
                }
            );
        });
    }
}

// SLM transpose tile: coalesces both the loads and the stores. The +1 pad makes the row
// stride 33, coprime with 32 banks, so both phases are bank-conflict-free.
template <পূর্ণসংখ্যা DC, পূর্ণসংখ্যা TT, পূর্ণসংখ্যা TC, পূর্ণসংখ্যা WG>
স্থির __dpct_inline__ শূন্য ssm_conv_tile(
    nd_item<1> it, local_accessor<দশমিক, 1> tile, ধ্রুবক দশমিক *src_তথ্য, ধ্রুবক দশমিক *weights,
    দশমিক *dst_তথ্য, পূর্ণসংখ্যা n_t, পূর্ণসংখ্যা nt_tiles, পূর্ণসংখ্যা nc_tiles, পূর্ণসংখ্যা src_stride_inner,
    পূর্ণসংখ্যা src_stride_seq, পূর্ণসংখ্যা dst_stride_token, পূর্ণসংখ্যা dst_stride_seq, যৌক্তিক apply_silu,
    ধ্রুবক দশমিক *bias
) {
    ধ্রুবক পূর্ণসংখ্যা    lid = static_cast<পূর্ণসংখ্যা>(it.get_local_id(0));
    ধ্রুবক size_t g   = it.get_group(0);
    ধ্রুবক পূর্ণসংখ্যা    tt  = static_cast<পূর্ণসংখ্যা>(g % nt_tiles);
    ধ্রুবক পূর্ণসংখ্যা    ct  = static_cast<পূর্ণসংখ্যা>((g / nt_tiles) % nc_tiles);
    ধ্রুবক পূর্ণসংখ্যা    seq = static_cast<পূর্ণসংখ্যা>(g / (static_cast<size_t>(nt_tiles) * nc_tiles));
    ধ্রুবক পূর্ণসংখ্যা    t0 = tt * TT, c0 = ct * TC;

    ধ্রুবক পূর্ণসংখ্যা ti = lid % TT;
    ধ্রুবক পূর্ণসংখ্যা cj = lid / TT;
#pragma unroll
    জন্য (পূর্ণসংখ্যা r = 0; r < TC / (WG / TT); ++r) {
        ধ্রুবক পূর্ণসংখ্যা c   = cj + r * (WG / TT);
        ধ্রুবক পূর্ণসংখ্যা tok = t0 + ti;
        দশমিক sumf = 0.0f;
        যদি (tok < n_t) {
            ধ্রুবক দশমিক *s = src_তথ্য + static_cast<size_t>(seq) * src_stride_seq
                           + static_cast<size_t>(c0 + c) * src_stride_inner + tok;
            ধ্রুবক দশমিক *cw = weights + static_cast<size_t>(c0 + c) * DC;
#pragma unroll
            জন্য (পূর্ণসংখ্যা i = 0; i < DC; ++i) sumf += s[i] * cw[i];
            যদি (bias != nullptr) sumf += bias[c0 + c];
            যদি (apply_silu) sumf = op_silu(sumf);
        }
        tile[c * (TT + 1) + ti] = sumf;
    }
    it.barrier(access::fence_space::local_space);

    ধ্রুবক পূর্ণসংখ্যা cc = lid % TC;
    ধ্রুবক পূর্ণসংখ্যা tj = lid / TC;
#pragma unroll
    জন্য (পূর্ণসংখ্যা r = 0; r < TT / (WG / TC); ++r) {
        ধ্রুবক পূর্ণসংখ্যা t   = tj + r * (WG / TC);
        ধ্রুবক পূর্ণসংখ্যা tok = t0 + t;
        যদি (tok < n_t) {
            dst_তথ্য[static_cast<size_t>(seq) * dst_stride_seq
                     + static_cast<size_t>(tok) * dst_stride_token + c0 + cc]
                = tile[cc * (TT + 1) + t];
        }
    }
}

// Same FUSED split as kernel_ssm_conv_impl. The fused instantiation keeps the runtime
// apply_silu/bias branches: at n_t >= 32 they are amortized over the whole tile.
template <পূর্ণসংখ্যা DC, যৌক্তিক FUSED>
স্থির শূন্য kernel_ssm_conv_tiled(
    queue &q, ধ্রুবক দশমিক *src_তথ্য, ধ্রুবক দশমিক *weights, দশমিক *dst_তথ্য,
    পূর্ণসংখ্যা d_inner, পূর্ণসংখ্যা n_t, পূর্ণসংখ্যা n_s, পূর্ণসংখ্যা src_stride_inner, পূর্ণসংখ্যা src_stride_seq,
    পূর্ণসংখ্যা dst_stride_token, পূর্ণসংখ্যা dst_stride_seq, যৌক্তিক apply_silu, ধ্রুবক দশমিক *bias
) {
    constexpr পূর্ণসংখ্যা TT = 32, TC = 32, WG = 256;
    ধ্রুবক পূর্ণসংখ্যা nt_tiles = (n_t + TT - 1) / TT;
    ধ্রুবক পূর্ণসংখ্যা nc_tiles = d_inner / TC;
    ধ্রুবক size_t groups = static_cast<size_t>(nt_tiles) * nc_tiles * n_s;

    যদি constexpr (FUSED) {
        q.submit([&](handler &h) {
            local_accessor<দশমিক, 1> tile(range<1>(TC * (TT + 1)), h);
            h.parallel_for(nd_range<1>(range<1>(groups * WG), range<1>(WG)), [=](nd_item<1> it) {
                ssm_conv_tile<DC, TT, TC, WG>(it, tile, src_তথ্য, weights, dst_তথ্য, n_t, nt_tiles,
                                              nc_tiles, src_stride_inner, src_stride_seq,
                                              dst_stride_token, dst_stride_seq, apply_silu, bias);
            });
        });
    } অন্যথা {
        GGML_UNUSED(apply_silu);
        GGML_UNUSED(bias);

        q.submit([&](handler &h) {
            local_accessor<দশমিক, 1> tile(range<1>(TC * (TT + 1)), h);
            h.parallel_for(nd_range<1>(range<1>(groups * WG), range<1>(WG)), [=](nd_item<1> it) {
                ssm_conv_tile<DC, TT, TC, WG>(it, tile, src_তথ্য, weights, dst_তথ্য, n_t, nt_tiles,
                                              nc_tiles, src_stride_inner, src_stride_seq,
                                              dst_stride_token, dst_stride_seq, মিথ্যা, nullptr);
            });
        });
    }
}

স্থির শূন্য kernel_ssm_conv(
    queue &q,
    ধ্রুবক দশমিক *src_তথ্য,
    ধ্রুবক দশমিক *weights,
    দশমিক *dst_তথ্য,
    পূর্ণসংখ্যা d_conv,
    পূর্ণসংখ্যা d_inner,
    পূর্ণসংখ্যা n_t,
    পূর্ণসংখ্যা n_s,
    পূর্ণসংখ্যা ncs,
    পূর্ণসংখ্যা src_stride_inner,
    পূর্ণসংখ্যা src_stride_seq,
    পূর্ণসংখ্যা dst_stride_token,
    পূর্ণসংখ্যা dst_stride_seq,
    যৌক্তিক apply_silu,
    ধ্রুবক দশমিক *bias
) {
    // Only the fused instantiations carry apply_silu/bias as kernel arguments; the plain
    // ssm_conv launch keeps the argument list it had before the fusion landed.
    ধ্রুবক যৌক্তিক fused = apply_silu || bias != nullptr;

    // d_inner must be a multiple of 32 so the channel tiles are exact; the transpose is only
    // worth it for n_t >= 32. d_conv == 4 is the only window with a DC-specialized kernel.
    যদি (d_conv == 4 && n_t >= 32 && (d_inner % 32) == 0) {
        যদি (fused) {
            kernel_ssm_conv_tiled<4, সত্য>(q, src_তথ্য, weights, dst_তথ্য, d_inner, n_t, n_s,
                                           src_stride_inner, src_stride_seq, dst_stride_token,
                                           dst_stride_seq, apply_silu, bias);
        } অন্যথা {
            kernel_ssm_conv_tiled<4, মিথ্যা>(q, src_তথ্য, weights, dst_তথ্য, d_inner, n_t, n_s,
                                            src_stride_inner, src_stride_seq, dst_stride_token,
                                            dst_stride_seq, apply_silu, bias);
        }
        ফেরত;
    }

    যদি (d_conv == 4) {
        যদি (fused) {
            kernel_ssm_conv_impl<4, সত্য>(q, src_তথ্য, weights, dst_তথ্য, d_conv, d_inner, n_t, n_s,
                                          ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                          dst_stride_seq, apply_silu, bias);
        } অন্যথা {
            kernel_ssm_conv_impl<4, মিথ্যা>(q, src_তথ্য, weights, dst_তথ্য, d_conv, d_inner, n_t, n_s,
                                           ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                           dst_stride_seq, apply_silu, bias);
        }
        ফেরত;
    }

    যদি (fused) {
        kernel_ssm_conv_impl<0, সত্য>(q, src_তথ্য, weights, dst_তথ্য, d_conv, d_inner, n_t, n_s,
                                      ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                      dst_stride_seq, apply_silu, bias);
    } অন্যথা {
        kernel_ssm_conv_impl<0, মিথ্যা>(q, src_তথ্য, weights, dst_তথ্য, d_conv, d_inner, n_t, n_s,
                                       ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                       dst_stride_seq, apply_silu, bias);
    }
}

অন্তঃরেখ শূন্য ggml_sycl_op_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * silu_dst = nullptr, ধ্রুবক দশমিক * bias = nullptr) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(bias == nullptr || silu_dst != nullptr);

    ধ্রুবক পূর্ণসংখ্যা d_conv   = src1->ne[0];
    ধ্রুবক পূর্ণসংখ্যা ncs      = src0->ne[0];
    ধ্রুবক পূর্ণসংখ্যা d_inner  = src0->ne[1];
    ধ্রুবক পূর্ণসংখ্যা n_t      = dst->ne[1];
    ধ্রুবক পূর্ণসংখ্যা n_s      = dst->ne[2];

    GGML_ASSERT(src0->ne[0] == d_conv - 1 + n_t);
    GGML_ASSERT(src0->ne[1] == d_inner);
    GGML_ASSERT(src1->ne[1] == d_inner);

    GGML_ASSERT(dst->ne[0] == d_inner);
    GGML_ASSERT(dst->ne[1] == n_t);
    GGML_ASSERT(dst->ne[2] == n_s);

    GGML_ASSERT(src0->nb[0] == আকার(দশমিক));
    GGML_ASSERT(src1->nb[0] == আকার(দশমিক));

    GGML_ASSERT(src0->nb[1] == src0->ne[0] * আকার(দশমিক));

    ধ্রুবক পূর্ণসংখ্যা src_stride_inner = ncs;
    ধ্রুবক পূর্ণসংখ্যা src_stride_seq   = ncs * d_inner;
    ধ্রুবক পূর্ণসংখ্যা dst_stride_token = d_inner;
    ধ্রুবক পূর্ণসংখ্যা dst_stride_seq   = d_inner * n_t;

    চেষ্টা {
        queue *q = ctx.stream();

        ধ্রুবক দশমিক *src_তথ্য = static_cast<ধ্রুবক দশমিক *>(src0->তথ্য);
        ধ্রুবক দশমিক *weights  = static_cast<ধ্রুবক দশমিক *>(src1->তথ্য);
        ধ্রুবক যৌক্তিক apply_silu = silu_dst != nullptr;
        দশমিক *dst_তথ্য       = static_cast<দশমিক *>((silu_dst ? silu_dst : dst)->তথ্য);

        GGML_ASSERT(src_তথ্য && weights && dst_তথ্য);

        kernel_ssm_conv(
            *q,
            src_তথ্য,
            weights,
            dst_তথ্য,
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

    } ধরো (ধ্রুবক std::exception &e) {
        std::fprintf(stderr, "[SYCL-SSM_CONV] ERROR: %s\n", e.what());
        ছোড়ো;
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
    ধ্রুবক দশমিক * bias = nullptr;
    যদি (add != nullptr) {
        ধ্রুবক ggml_tensor * bias_t = (add->src[0] == dst) ? add->src[1] : add->src[0];
        bias = static_cast<ধ্রুবক দশমিক *>(bias_t->তথ্য);
    }
    ggml_sycl_op_ssm_conv(ctx, dst, silu_dst, bias);
}
