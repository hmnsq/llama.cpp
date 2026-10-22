#అన్తర్భావ "ssm_conv.hpp"
#అన్తర్భావ "common.hpp"
#అన్తర్భావ "element_wise.hpp"

#అన్తర్భావ <cstdio>

ఉపయోగించి నామాకాశ sycl;

// One output element of the conv. DC is d_conv as a compile-time constant (0 keeps the
// runtime loop); unfused callers pass literal false/nullptr so the epilogue folds away.
మూస <పూర్ణాంకం DC>
స్థిర __dpct_inline__ శూన్య ssm_conv_element(
    size_t idx,
    స్థిరరాశి దశాంశం *src_దత్తాంశం,
    స్థిరరాశి దశాంశం *weights,
    దశాంశం *dst_దత్తాంశం,
    పూర్ణాంకం d_conv,
    పూర్ణాంకం d_inner,
    పూర్ణాంకం n_t,
    పూర్ణాంకం src_stride_inner,
    పూర్ణాంకం src_stride_seq,
    పూర్ణాంకం dst_stride_token,
    పూర్ణాంకం dst_stride_seq,
    తార్కిక apply_silu,
    స్థిరరాశి దశాంశం *bias
) {
    // src is token-contiguous per channel, dst is channel-contiguous per token,
    // so indexing token-fastest coalesces the d_conv loads.
    స్థిరరాశి పూర్ణాంకం token   = static_cast<పూర్ణాంకం>(idx % n_t);
    స్థిరరాశి పూర్ణాంకం channel = static_cast<పూర్ణాంకం>((idx / n_t) % d_inner);
    స్థిరరాశి పూర్ణాంకం seq     = static_cast<పూర్ణాంకం>(idx / (static_cast<size_t>(n_t) * static_cast<size_t>(d_inner)));

    స్థిరరాశి దశాంశం *s = src_దత్తాంశం
        + static_cast<size_t>(seq) * static_cast<size_t>(src_stride_seq)
        + static_cast<size_t>(channel) * static_cast<size_t>(src_stride_inner)
        + static_cast<size_t>(token);

    స్థిరరాశి దశాంశం *c = weights + static_cast<size_t>(channel) * static_cast<size_t>(d_conv);

    దశాంశం sumf = 0.0f;
    ఒకవేళ constexpr (DC > 0) {
#సూచన unroll
        కొరకు (పూర్ణాంకం i0 = 0; i0 < DC; ++i0) {
            sumf += s[i0] * c[i0];
        }
    } లేకపోతే {
        కొరకు (పూర్ణాంకం i0 = 0; i0 < d_conv; ++i0) {
            sumf += s[i0] * c[i0];
        }
    }

    // fused bias add: the ADD node broadcasts a 1-D channel bias over tokens
    ఒకవేళ (bias != శూన్యసూచిక) {
        sumf += bias[channel];
    }

    స్థిరరాశి size_t dst_idx =
        static_cast<size_t>(seq) * static_cast<size_t>(dst_stride_seq) +
        static_cast<size_t>(token) * static_cast<size_t>(dst_stride_token) +
        static_cast<size_t>(channel);

    dst_దత్తాంశం[dst_idx] = apply_silu ? op_silu(sumf) : sumf;
}

// FUSED=false keeps apply_silu/bias out of the kernel capture list, so the unfused launch
// takes the pre-fusion argument list; matters at n_t == 1, where the op is launch-bound.
మూస <పూర్ణాంకం DC, తార్కిక FUSED>
స్థిర శూన్య kernel_ssm_conv_impl(
    queue &q,
    స్థిరరాశి దశాంశం *src_దత్తాంశం,
    స్థిరరాశి దశాంశం *weights,
    దశాంశం *dst_దత్తాంశం,
    పూర్ణాంకం d_conv,
    పూర్ణాంకం d_inner,
    పూర్ణాంకం n_t,
    పూర్ణాంకం n_s,
    పూర్ణాంకం ncs __attribute__((వాడని_2)),
    పూర్ణాంకం src_stride_inner,
    పూర్ణాంకం src_stride_seq,
    పూర్ణాంకం dst_stride_token,
    పూర్ణాంకం dst_stride_seq,
    తార్కిక apply_silu,
    స్థిరరాశి దశాంశం *bias
) {
    స్థిరరాశి size_t మొత్తం_work = static_cast<size_t>(d_inner) * static_cast<size_t>(n_t) * static_cast<size_t>(n_s);
    స్థిరరాశి size_t work_సమూహం_కొలత = 256;
    స్థిరరాశి size_t అంకె_work_groups = (మొత్తం_work + work_సమూహం_కొలత - 1) / work_సమూహం_కొలత;

    స్థిరరాశి range<1> global_range(అంకె_work_groups * work_సమూహం_కొలత);
    స్థిరరాశి range<1> local_range(work_సమూహం_కొలత);

    ఒకవేళ constexpr (FUSED) {
        q.submit([&](handler &h) {
            h.parallel_for(
                nd_range<1>(global_range, local_range),
                [=](nd_అంశం<1> అంశం) {
                    స్థిరరాశి size_t idx = అంశం.get_global_id(0);
                    ఒకవేళ (idx >= మొత్తం_work) {
                        తిరిగి;
                    }

                    ssm_conv_element<DC>(idx, src_దత్తాంశం, weights, dst_దత్తాంశం, d_conv, d_inner, n_t,
                                         src_stride_inner, src_stride_seq, dst_stride_token,
                                         dst_stride_seq, apply_silu, bias);
                }
            );
        });
    } లేకపోతే {
        GGML_వాడని(apply_silu);
        GGML_వాడని(bias);

        q.submit([&](handler &h) {
            h.parallel_for(
                nd_range<1>(global_range, local_range),
                [=](nd_అంశం<1> అంశం) {
                    స్థిరరాశి size_t idx = అంశం.get_global_id(0);
                    ఒకవేళ (idx >= మొత్తం_work) {
                        తిరిగి;
                    }

                    ssm_conv_element<DC>(idx, src_దత్తాంశం, weights, dst_దత్తాంశం, d_conv, d_inner, n_t,
                                         src_stride_inner, src_stride_seq, dst_stride_token,
                                         dst_stride_seq, అబద్ధం, శూన్యసూచిక);
                }
            );
        });
    }
}

// SLM transpose tile: coalesces both the loads and the stores. The +1 pad makes the row
// stride 33, coprime with 32 banks, so both phases are bank-conflict-free.
మూస <పూర్ణాంకం DC, పూర్ణాంకం TT, పూర్ణాంకం TC, పూర్ణాంకం WG>
స్థిర __dpct_inline__ శూన్య ssm_conv_tile(
    nd_అంశం<1> it, local_accessor<దశాంశం, 1> tile, స్థిరరాశి దశాంశం *src_దత్తాంశం, స్థిరరాశి దశాంశం *weights,
    దశాంశం *dst_దత్తాంశం, పూర్ణాంకం n_t, పూర్ణాంకం nt_tiles, పూర్ణాంకం nc_tiles, పూర్ణాంకం src_stride_inner,
    పూర్ణాంకం src_stride_seq, పూర్ణాంకం dst_stride_token, పూర్ణాంకం dst_stride_seq, తార్కిక apply_silu,
    స్థిరరాశి దశాంశం *bias
) {
    స్థిరరాశి పూర్ణాంకం    lid = static_cast<పూర్ణాంకం>(it.get_local_id(0));
    స్థిరరాశి size_t g   = it.get_సమూహం(0);
    స్థిరరాశి పూర్ణాంకం    tt  = static_cast<పూర్ణాంకం>(g % nt_tiles);
    స్థిరరాశి పూర్ణాంకం    ct  = static_cast<పూర్ణాంకం>((g / nt_tiles) % nc_tiles);
    స్థిరరాశి పూర్ణాంకం    seq = static_cast<పూర్ణాంకం>(g / (static_cast<size_t>(nt_tiles) * nc_tiles));
    స్థిరరాశి పూర్ణాంకం    t0 = tt * TT, c0 = ct * TC;

    స్థిరరాశి పూర్ణాంకం ti = lid % TT;
    స్థిరరాశి పూర్ణాంకం cj = lid / TT;
#సూచన unroll
    కొరకు (పూర్ణాంకం r = 0; r < TC / (WG / TT); ++r) {
        స్థిరరాశి పూర్ణాంకం c   = cj + r * (WG / TT);
        స్థిరరాశి పూర్ణాంకం tok = t0 + ti;
        దశాంశం sumf = 0.0f;
        ఒకవేళ (tok < n_t) {
            స్థిరరాశి దశాంశం *s = src_దత్తాంశం + static_cast<size_t>(seq) * src_stride_seq
                           + static_cast<size_t>(c0 + c) * src_stride_inner + tok;
            స్థిరరాశి దశాంశం *cw = weights + static_cast<size_t>(c0 + c) * DC;
#సూచన unroll
            కొరకు (పూర్ణాంకం i = 0; i < DC; ++i) sumf += s[i] * cw[i];
            ఒకవేళ (bias != శూన్యసూచిక) sumf += bias[c0 + c];
            ఒకవేళ (apply_silu) sumf = op_silu(sumf);
        }
        tile[c * (TT + 1) + ti] = sumf;
    }
    it.barrier(access::fence_space::local_space);

    స్థిరరాశి పూర్ణాంకం cc = lid % TC;
    స్థిరరాశి పూర్ణాంకం tj = lid / TC;
#సూచన unroll
    కొరకు (పూర్ణాంకం r = 0; r < TT / (WG / TC); ++r) {
        స్థిరరాశి పూర్ణాంకం t   = tj + r * (WG / TC);
        స్థిరరాశి పూర్ణాంకం tok = t0 + t;
        ఒకవేళ (tok < n_t) {
            dst_దత్తాంశం[static_cast<size_t>(seq) * dst_stride_seq
                     + static_cast<size_t>(tok) * dst_stride_token + c0 + cc]
                = tile[cc * (TT + 1) + t];
        }
    }
}

// Same FUSED split as kernel_ssm_conv_impl. The fused instantiation keeps the runtime
// apply_silu/bias branches: at n_t >= 32 they are amortized over the whole tile.
మూస <పూర్ణాంకం DC, తార్కిక FUSED>
స్థిర శూన్య kernel_ssm_conv_tiled(
    queue &q, స్థిరరాశి దశాంశం *src_దత్తాంశం, స్థిరరాశి దశాంశం *weights, దశాంశం *dst_దత్తాంశం,
    పూర్ణాంకం d_inner, పూర్ణాంకం n_t, పూర్ణాంకం n_s, పూర్ణాంకం src_stride_inner, పూర్ణాంకం src_stride_seq,
    పూర్ణాంకం dst_stride_token, పూర్ణాంకం dst_stride_seq, తార్కిక apply_silu, స్థిరరాశి దశాంశం *bias
) {
    constexpr పూర్ణాంకం TT = 32, TC = 32, WG = 256;
    స్థిరరాశి పూర్ణాంకం nt_tiles = (n_t + TT - 1) / TT;
    స్థిరరాశి పూర్ణాంకం nc_tiles = d_inner / TC;
    స్థిరరాశి size_t groups = static_cast<size_t>(nt_tiles) * nc_tiles * n_s;

    ఒకవేళ constexpr (FUSED) {
        q.submit([&](handler &h) {
            local_accessor<దశాంశం, 1> tile(range<1>(TC * (TT + 1)), h);
            h.parallel_for(nd_range<1>(range<1>(groups * WG), range<1>(WG)), [=](nd_అంశం<1> it) {
                ssm_conv_tile<DC, TT, TC, WG>(it, tile, src_దత్తాంశం, weights, dst_దత్తాంశం, n_t, nt_tiles,
                                              nc_tiles, src_stride_inner, src_stride_seq,
                                              dst_stride_token, dst_stride_seq, apply_silu, bias);
            });
        });
    } లేకపోతే {
        GGML_వాడని(apply_silu);
        GGML_వాడని(bias);

        q.submit([&](handler &h) {
            local_accessor<దశాంశం, 1> tile(range<1>(TC * (TT + 1)), h);
            h.parallel_for(nd_range<1>(range<1>(groups * WG), range<1>(WG)), [=](nd_అంశం<1> it) {
                ssm_conv_tile<DC, TT, TC, WG>(it, tile, src_దత్తాంశం, weights, dst_దత్తాంశం, n_t, nt_tiles,
                                              nc_tiles, src_stride_inner, src_stride_seq,
                                              dst_stride_token, dst_stride_seq, అబద్ధం, శూన్యసూచిక);
            });
        });
    }
}

స్థిర శూన్య kernel_ssm_conv(
    queue &q,
    స్థిరరాశి దశాంశం *src_దత్తాంశం,
    స్థిరరాశి దశాంశం *weights,
    దశాంశం *dst_దత్తాంశం,
    పూర్ణాంకం d_conv,
    పూర్ణాంకం d_inner,
    పూర్ణాంకం n_t,
    పూర్ణాంకం n_s,
    పూర్ణాంకం ncs,
    పూర్ణాంకం src_stride_inner,
    పూర్ణాంకం src_stride_seq,
    పూర్ణాంకం dst_stride_token,
    పూర్ణాంకం dst_stride_seq,
    తార్కిక apply_silu,
    స్థిరరాశి దశాంశం *bias
) {
    // Only the fused instantiations carry apply_silu/bias as kernel arguments; the plain
    // ssm_conv launch keeps the argument list it had before the fusion landed.
    స్థిరరాశి తార్కిక fused = apply_silu || bias != శూన్యసూచిక;

    // d_inner must be a multiple of 32 so the channel tiles are exact; the transpose is only
    // worth it for n_t >= 32. d_conv == 4 is the only window with a DC-specialized kernel.
    ఒకవేళ (d_conv == 4 && n_t >= 32 && (d_inner % 32) == 0) {
        ఒకవేళ (fused) {
            kernel_ssm_conv_tiled<4, నిజం>(q, src_దత్తాంశం, weights, dst_దత్తాంశం, d_inner, n_t, n_s,
                                           src_stride_inner, src_stride_seq, dst_stride_token,
                                           dst_stride_seq, apply_silu, bias);
        } లేకపోతే {
            kernel_ssm_conv_tiled<4, అబద్ధం>(q, src_దత్తాంశం, weights, dst_దత్తాంశం, d_inner, n_t, n_s,
                                            src_stride_inner, src_stride_seq, dst_stride_token,
                                            dst_stride_seq, apply_silu, bias);
        }
        తిరిగి;
    }

    ఒకవేళ (d_conv == 4) {
        ఒకవేళ (fused) {
            kernel_ssm_conv_impl<4, నిజం>(q, src_దత్తాంశం, weights, dst_దత్తాంశం, d_conv, d_inner, n_t, n_s,
                                          ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                          dst_stride_seq, apply_silu, bias);
        } లేకపోతే {
            kernel_ssm_conv_impl<4, అబద్ధం>(q, src_దత్తాంశం, weights, dst_దత్తాంశం, d_conv, d_inner, n_t, n_s,
                                           ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                           dst_stride_seq, apply_silu, bias);
        }
        తిరిగి;
    }

    ఒకవేళ (fused) {
        kernel_ssm_conv_impl<0, నిజం>(q, src_దత్తాంశం, weights, dst_దత్తాంశం, d_conv, d_inner, n_t, n_s,
                                      ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                      dst_stride_seq, apply_silu, bias);
    } లేకపోతే {
        kernel_ssm_conv_impl<0, అబద్ధం>(q, src_దత్తాంశం, weights, dst_దత్తాంశం, d_conv, d_inner, n_t, n_s,
                                       ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                       dst_stride_seq, apply_silu, bias);
    }
}

అంతఃవరుస శూన్య ggml_sycl_op_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * silu_dst = శూన్యసూచిక, స్థిరరాశి దశాంశం * bias = శూన్యసూచిక) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(bias == శూన్యసూచిక || silu_dst != శూన్యసూచిక);

    స్థిరరాశి పూర్ణాంకం d_conv   = src1->ne[0];
    స్థిరరాశి పూర్ణాంకం ncs      = src0->ne[0];
    స్థిరరాశి పూర్ణాంకం d_inner  = src0->ne[1];
    స్థిరరాశి పూర్ణాంకం n_t      = dst->ne[1];
    స్థిరరాశి పూర్ణాంకం n_s      = dst->ne[2];

    GGML_ASSERT(src0->ne[0] == d_conv - 1 + n_t);
    GGML_ASSERT(src0->ne[1] == d_inner);
    GGML_ASSERT(src1->ne[1] == d_inner);

    GGML_ASSERT(dst->ne[0] == d_inner);
    GGML_ASSERT(dst->ne[1] == n_t);
    GGML_ASSERT(dst->ne[2] == n_s);

    GGML_ASSERT(src0->nb[0] == పరిమాణం(దశాంశం));
    GGML_ASSERT(src1->nb[0] == పరిమాణం(దశాంశం));

    GGML_ASSERT(src0->nb[1] == src0->ne[0] * పరిమాణం(దశాంశం));

    స్థిరరాశి పూర్ణాంకం src_stride_inner = ncs;
    స్థిరరాశి పూర్ణాంకం src_stride_seq   = ncs * d_inner;
    స్థిరరాశి పూర్ణాంకం dst_stride_token = d_inner;
    స్థిరరాశి పూర్ణాంకం dst_stride_seq   = d_inner * n_t;

    ప్రయత్నించు {
        queue *q = ctx.stream();

        స్థిరరాశి దశాంశం *src_దత్తాంశం = static_cast<స్థిరరాశి దశాంశం *>(src0->దత్తాంశం);
        స్థిరరాశి దశాంశం *weights  = static_cast<స్థిరరాశి దశాంశం *>(src1->దత్తాంశం);
        స్థిరరాశి తార్కిక apply_silu = silu_dst != శూన్యసూచిక;
        దశాంశం *dst_దత్తాంశం       = static_cast<దశాంశం *>((silu_dst ? silu_dst : dst)->దత్తాంశం);

        GGML_ASSERT(src_దత్తాంశం && weights && dst_దత్తాంశం);

        kernel_ssm_conv(
            *q,
            src_దత్తాంశం,
            weights,
            dst_దత్తాంశం,
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

    } పట్టుకో (స్థిరరాశి std::exception &e) {
        std::fprintf(stderr, "[SYCL-SSM_CONV] ERROR: %s\n", e.what());
        విసురు;
    }
}

శూన్య ggml_sycl_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_ssm_conv(ctx, dst);
}

// Fused ssm_conv + ADD + SiLU: write silu(conv(x) + b) straight into silu_dst, eliding the
// standalone SiLU launch and its HBM round-trip of the conv output.
శూన్య ggml_sycl_ssm_conv_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * కూడు_2, ggml_tensor * silu_dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    GGML_ASSERT(silu_dst && ggml_are_same_shape(dst, silu_dst) && silu_dst->type == GGML_TYPE_F32);
    // the fused kernel reads only the ADD's bias operand; the ADD result is never written
    స్థిరరాశి దశాంశం * bias = శూన్యసూచిక;
    ఒకవేళ (కూడు_2 != శూన్యసూచిక) {
        స్థిరరాశి ggml_tensor * bias_t = (కూడు_2->src[0] == dst) ? కూడు_2->src[1] : కూడు_2->src[0];
        bias = static_cast<స్థిరరాశి దశాంశం *>(bias_t->దత్తాంశం);
    }
    ggml_sycl_op_ssm_conv(ctx, dst, silu_dst, bias);
}
