#ضمن "ssm_conv.hpp"
#ضمن "common.hpp"
#ضمن "element_wise.hpp"

#ضمن <cstdio>

کارول نطاق sycl;

// One output element of the conv. DC is d_conv as a compile-time constant (0 keeps the
// runtime loop); unfused callers pass literal false/nullptr so the epilogue folds away.
کينډۍ <صحيح DC>
ساكن __dpct_inline__ فراغ ssm_conv_element(
    size_t idx,
    ثابت عائم *src_معلومات,
    ثابت عائم *weights,
    عائم *dst_معلومات,
    صحيح d_conv,
    صحيح d_inner,
    صحيح n_t,
    صحيح src_stride_inner,
    صحيح src_stride_seq,
    صحيح dst_stride_token,
    صحيح dst_stride_seq,
    منطقي apply_silu,
    ثابت عائم *bias
) {
    // src is token-contiguous per channel, dst is channel-contiguous per token,
    // so indexing token-fastest coalesces the d_conv loads.
    ثابت صحيح token   = static_cast<صحيح>(idx % n_t);
    ثابت صحيح channel = static_cast<صحيح>((idx / n_t) % d_inner);
    ثابت صحيح seq     = static_cast<صحيح>(idx / (static_cast<size_t>(n_t) * static_cast<size_t>(d_inner)));

    ثابت عائم *s = src_معلومات
        + static_cast<size_t>(seq) * static_cast<size_t>(src_stride_seq)
        + static_cast<size_t>(channel) * static_cast<size_t>(src_stride_inner)
        + static_cast<size_t>(token);

    ثابت عائم *c = weights + static_cast<size_t>(channel) * static_cast<size_t>(d_conv);

    عائم sumf = 0.0f;
    إذا constexpr (DC > 0) {
#لارښوونه unroll
        لكل (صحيح i0 = 0; i0 < DC; ++i0) {
            sumf += s[i0] * c[i0];
        }
    } وإلا {
        لكل (صحيح i0 = 0; i0 < d_conv; ++i0) {
            sumf += s[i0] * c[i0];
        }
    }

    // fused bias add: the ADD node broadcasts a 1-D channel bias over tokens
    إذا (bias != تش_اشاره) {
        sumf += bias[channel];
    }

    ثابت size_t dst_idx =
        static_cast<size_t>(seq) * static_cast<size_t>(dst_stride_seq) +
        static_cast<size_t>(token) * static_cast<size_t>(dst_stride_token) +
        static_cast<size_t>(channel);

    dst_معلومات[dst_idx] = apply_silu ? op_silu(sumf) : sumf;
}

// FUSED=false keeps apply_silu/bias out of the kernel capture list, so the unfused launch
// takes the pre-fusion argument list; matters at n_t == 1, where the op is launch-bound.
کينډۍ <صحيح DC, منطقي FUSED>
ساكن فراغ kernel_ssm_conv_impl(
    queue &q,
    ثابت عائم *src_معلومات,
    ثابت عائم *weights,
    عائم *dst_معلومات,
    صحيح d_conv,
    صحيح d_inner,
    صحيح n_t,
    صحيح n_s,
    صحيح ncs __attribute__((ناکارول_شوی_2)),
    صحيح src_stride_inner,
    صحيح src_stride_seq,
    صحيح dst_stride_token,
    صحيح dst_stride_seq,
    منطقي apply_silu,
    ثابت عائم *bias
) {
    ثابت size_t ټول_work = static_cast<size_t>(d_inner) * static_cast<size_t>(n_t) * static_cast<size_t>(n_s);
    ثابت size_t work_ډله_اندازه = 256;
    ثابت size_t عدد_work_groups = (ټول_work + work_ډله_اندازه - 1) / work_ډله_اندازه;

    ثابت range<1> global_range(عدد_work_groups * work_ډله_اندازه);
    ثابت range<1> local_range(work_ډله_اندازه);

    إذا constexpr (FUSED) {
        q.submit([&](handler &h) {
            h.parallel_for(
                nd_range<1>(global_range, local_range),
                [=](nd_توکی<1> توکی) {
                    ثابت size_t idx = توکی.get_global_id(0);
                    إذا (idx >= ټول_work) {
                        أرجع;
                    }

                    ssm_conv_element<DC>(idx, src_معلومات, weights, dst_معلومات, d_conv, d_inner, n_t,
                                         src_stride_inner, src_stride_seq, dst_stride_token,
                                         dst_stride_seq, apply_silu, bias);
                }
            );
        });
    } وإلا {
        GGML_ناکارول_شوی(apply_silu);
        GGML_ناکارول_شوی(bias);

        q.submit([&](handler &h) {
            h.parallel_for(
                nd_range<1>(global_range, local_range),
                [=](nd_توکی<1> توکی) {
                    ثابت size_t idx = توکی.get_global_id(0);
                    إذا (idx >= ټول_work) {
                        أرجع;
                    }

                    ssm_conv_element<DC>(idx, src_معلومات, weights, dst_معلومات, d_conv, d_inner, n_t,
                                         src_stride_inner, src_stride_seq, dst_stride_token,
                                         dst_stride_seq, خطأ, تش_اشاره);
                }
            );
        });
    }
}

// SLM transpose tile: coalesces both the loads and the stores. The +1 pad makes the row
// stride 33, coprime with 32 banks, so both phases are bank-conflict-free.
کينډۍ <صحيح DC, صحيح TT, صحيح TC, صحيح WG>
ساكن __dpct_inline__ فراغ ssm_conv_tile(
    nd_توکی<1> it, local_accessor<عائم, 1> tile, ثابت عائم *src_معلومات, ثابت عائم *weights,
    عائم *dst_معلومات, صحيح n_t, صحيح nt_tiles, صحيح nc_tiles, صحيح src_stride_inner,
    صحيح src_stride_seq, صحيح dst_stride_token, صحيح dst_stride_seq, منطقي apply_silu,
    ثابت عائم *bias
) {
    ثابت صحيح    lid = static_cast<صحيح>(it.get_local_id(0));
    ثابت size_t g   = it.get_ډله(0);
    ثابت صحيح    tt  = static_cast<صحيح>(g % nt_tiles);
    ثابت صحيح    ct  = static_cast<صحيح>((g / nt_tiles) % nc_tiles);
    ثابت صحيح    seq = static_cast<صحيح>(g / (static_cast<size_t>(nt_tiles) * nc_tiles));
    ثابت صحيح    t0 = tt * TT, c0 = ct * TC;

    ثابت صحيح ti = lid % TT;
    ثابت صحيح cj = lid / TT;
#لارښوونه unroll
    لكل (صحيح r = 0; r < TC / (WG / TT); ++r) {
        ثابت صحيح c   = cj + r * (WG / TT);
        ثابت صحيح tok = t0 + ti;
        عائم sumf = 0.0f;
        إذا (tok < n_t) {
            ثابت عائم *s = src_معلومات + static_cast<size_t>(seq) * src_stride_seq
                           + static_cast<size_t>(c0 + c) * src_stride_inner + tok;
            ثابت عائم *cw = weights + static_cast<size_t>(c0 + c) * DC;
#لارښوونه unroll
            لكل (صحيح i = 0; i < DC; ++i) sumf += s[i] * cw[i];
            إذا (bias != تش_اشاره) sumf += bias[c0 + c];
            إذا (apply_silu) sumf = op_silu(sumf);
        }
        tile[c * (TT + 1) + ti] = sumf;
    }
    it.barrier(access::fence_space::local_space);

    ثابت صحيح cc = lid % TC;
    ثابت صحيح tj = lid / TC;
#لارښوونه unroll
    لكل (صحيح r = 0; r < TT / (WG / TC); ++r) {
        ثابت صحيح t   = tj + r * (WG / TC);
        ثابت صحيح tok = t0 + t;
        إذا (tok < n_t) {
            dst_معلومات[static_cast<size_t>(seq) * dst_stride_seq
                     + static_cast<size_t>(tok) * dst_stride_token + c0 + cc]
                = tile[cc * (TT + 1) + t];
        }
    }
}

// Same FUSED split as kernel_ssm_conv_impl. The fused instantiation keeps the runtime
// apply_silu/bias branches: at n_t >= 32 they are amortized over the whole tile.
کينډۍ <صحيح DC, منطقي FUSED>
ساكن فراغ kernel_ssm_conv_tiled(
    queue &q, ثابت عائم *src_معلومات, ثابت عائم *weights, عائم *dst_معلومات,
    صحيح d_inner, صحيح n_t, صحيح n_s, صحيح src_stride_inner, صحيح src_stride_seq,
    صحيح dst_stride_token, صحيح dst_stride_seq, منطقي apply_silu, ثابت عائم *bias
) {
    constexpr صحيح TT = 32, TC = 32, WG = 256;
    ثابت صحيح nt_tiles = (n_t + TT - 1) / TT;
    ثابت صحيح nc_tiles = d_inner / TC;
    ثابت size_t groups = static_cast<size_t>(nt_tiles) * nc_tiles * n_s;

    إذا constexpr (FUSED) {
        q.submit([&](handler &h) {
            local_accessor<عائم, 1> tile(range<1>(TC * (TT + 1)), h);
            h.parallel_for(nd_range<1>(range<1>(groups * WG), range<1>(WG)), [=](nd_توکی<1> it) {
                ssm_conv_tile<DC, TT, TC, WG>(it, tile, src_معلومات, weights, dst_معلومات, n_t, nt_tiles,
                                              nc_tiles, src_stride_inner, src_stride_seq,
                                              dst_stride_token, dst_stride_seq, apply_silu, bias);
            });
        });
    } وإلا {
        GGML_ناکارول_شوی(apply_silu);
        GGML_ناکارول_شوی(bias);

        q.submit([&](handler &h) {
            local_accessor<عائم, 1> tile(range<1>(TC * (TT + 1)), h);
            h.parallel_for(nd_range<1>(range<1>(groups * WG), range<1>(WG)), [=](nd_توکی<1> it) {
                ssm_conv_tile<DC, TT, TC, WG>(it, tile, src_معلومات, weights, dst_معلومات, n_t, nt_tiles,
                                              nc_tiles, src_stride_inner, src_stride_seq,
                                              dst_stride_token, dst_stride_seq, خطأ, تش_اشاره);
            });
        });
    }
}

ساكن فراغ kernel_ssm_conv(
    queue &q,
    ثابت عائم *src_معلومات,
    ثابت عائم *weights,
    عائم *dst_معلومات,
    صحيح d_conv,
    صحيح d_inner,
    صحيح n_t,
    صحيح n_s,
    صحيح ncs,
    صحيح src_stride_inner,
    صحيح src_stride_seq,
    صحيح dst_stride_token,
    صحيح dst_stride_seq,
    منطقي apply_silu,
    ثابت عائم *bias
) {
    // Only the fused instantiations carry apply_silu/bias as kernel arguments; the plain
    // ssm_conv launch keeps the argument list it had before the fusion landed.
    ثابت منطقي fused = apply_silu || bias != تش_اشاره;

    // d_inner must be a multiple of 32 so the channel tiles are exact; the transpose is only
    // worth it for n_t >= 32. d_conv == 4 is the only window with a DC-specialized kernel.
    إذا (d_conv == 4 && n_t >= 32 && (d_inner % 32) == 0) {
        إذا (fused) {
            kernel_ssm_conv_tiled<4, ریښتیا>(q, src_معلومات, weights, dst_معلومات, d_inner, n_t, n_s,
                                           src_stride_inner, src_stride_seq, dst_stride_token,
                                           dst_stride_seq, apply_silu, bias);
        } وإلا {
            kernel_ssm_conv_tiled<4, خطأ>(q, src_معلومات, weights, dst_معلومات, d_inner, n_t, n_s,
                                            src_stride_inner, src_stride_seq, dst_stride_token,
                                            dst_stride_seq, apply_silu, bias);
        }
        أرجع;
    }

    إذا (d_conv == 4) {
        إذا (fused) {
            kernel_ssm_conv_impl<4, ریښتیا>(q, src_معلومات, weights, dst_معلومات, d_conv, d_inner, n_t, n_s,
                                          ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                          dst_stride_seq, apply_silu, bias);
        } وإلا {
            kernel_ssm_conv_impl<4, خطأ>(q, src_معلومات, weights, dst_معلومات, d_conv, d_inner, n_t, n_s,
                                           ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                           dst_stride_seq, apply_silu, bias);
        }
        أرجع;
    }

    إذا (fused) {
        kernel_ssm_conv_impl<0, ریښتیا>(q, src_معلومات, weights, dst_معلومات, d_conv, d_inner, n_t, n_s,
                                      ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                      dst_stride_seq, apply_silu, bias);
    } وإلا {
        kernel_ssm_conv_impl<0, خطأ>(q, src_معلومات, weights, dst_معلومات, d_conv, d_inner, n_t, n_s,
                                       ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                       dst_stride_seq, apply_silu, bias);
    }
}

مضمن فراغ ggml_sycl_op_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * silu_dst = تش_اشاره, ثابت عائم * bias = تش_اشاره) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(bias == تش_اشاره || silu_dst != تش_اشاره);

    ثابت صحيح d_conv   = src1->ne[0];
    ثابت صحيح ncs      = src0->ne[0];
    ثابت صحيح d_inner  = src0->ne[1];
    ثابت صحيح n_t      = dst->ne[1];
    ثابت صحيح n_s      = dst->ne[2];

    GGML_ASSERT(src0->ne[0] == d_conv - 1 + n_t);
    GGML_ASSERT(src0->ne[1] == d_inner);
    GGML_ASSERT(src1->ne[1] == d_inner);

    GGML_ASSERT(dst->ne[0] == d_inner);
    GGML_ASSERT(dst->ne[1] == n_t);
    GGML_ASSERT(dst->ne[2] == n_s);

    GGML_ASSERT(src0->nb[0] == حجم(عائم));
    GGML_ASSERT(src1->nb[0] == حجم(عائم));

    GGML_ASSERT(src0->nb[1] == src0->ne[0] * حجم(عائم));

    ثابت صحيح src_stride_inner = ncs;
    ثابت صحيح src_stride_seq   = ncs * d_inner;
    ثابت صحيح dst_stride_token = d_inner;
    ثابت صحيح dst_stride_seq   = d_inner * n_t;

    حاول {
        queue *q = ctx.stream();

        ثابت عائم *src_معلومات = static_cast<ثابت عائم *>(src0->معلومات);
        ثابت عائم *weights  = static_cast<ثابت عائم *>(src1->معلومات);
        ثابت منطقي apply_silu = silu_dst != تش_اشاره;
        عائم *dst_معلومات       = static_cast<عائم *>((silu_dst ? silu_dst : dst)->معلومات);

        GGML_ASSERT(src_معلومات && weights && dst_معلومات);

        kernel_ssm_conv(
            *q,
            src_معلومات,
            weights,
            dst_معلومات,
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

    } امسك (ثابت std::exception &e) {
        std::fprintf(stderr, "[SYCL-SSM_CONV] ERROR: %s\n", e.what());
        ارم;
    }
}

فراغ ggml_sycl_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_ssm_conv(ctx, dst);
}

// Fused ssm_conv + ADD + SiLU: write silu(conv(x) + b) straight into silu_dst, eliding the
// standalone SiLU launch and its HBM round-trip of the conv output.
فراغ ggml_sycl_ssm_conv_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * جمع_کړه_2, ggml_tensor * silu_dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    GGML_ASSERT(silu_dst && ggml_are_same_shape(dst, silu_dst) && silu_dst->type == GGML_TYPE_F32);
    // the fused kernel reads only the ADD's bias operand; the ADD result is never written
    ثابت عائم * bias = تش_اشاره;
    إذا (جمع_کړه_2 != تش_اشاره) {
        ثابت ggml_tensor * bias_t = (جمع_کړه_2->src[0] == dst) ? جمع_کړه_2->src[1] : جمع_کړه_2->src[0];
        bias = static_cast<ثابت عائم *>(bias_t->معلومات);
    }
    ggml_sycl_op_ssm_conv(ctx, dst, silu_dst, bias);
}
