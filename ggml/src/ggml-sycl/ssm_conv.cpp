#அந்தர்பாவ "ssm_conv.hpp"
#அந்தர்பாவ "common.hpp"
#அந்தர்பாவ "element_wise.hpp"

#அந்தர்பாவ <cstdio>

using நாமாகாஶ sycl;

// One output element of the conv. DC is d_conv as a compile-time constant (0 keeps the
// runtime loop); unfused callers pass literal false/nullptr so the epilogue folds away.
template <முழு DC>
நிலையான __dpct_inline__ வெறுமை ssm_conv_element(
    size_t idx,
    மாறிலி தசம *src_data,
    மாறிலி தசம *weights,
    தசம *dst_data,
    முழு d_conv,
    முழு d_inner,
    முழு n_t,
    முழு src_stride_inner,
    முழு src_stride_seq,
    முழு dst_stride_token,
    முழு dst_stride_seq,
    தருக்க apply_silu,
    மாறிலி தசம *bias
) {
    // src is token-contiguous per channel, dst is channel-contiguous per token,
    // so indexing token-fastest coalesces the d_conv loads.
    மாறிலி முழு token   = static_cast<முழு>(idx % n_t);
    மாறிலி முழு channel = static_cast<முழு>((idx / n_t) % d_inner);
    மாறிலி முழு seq     = static_cast<முழு>(idx / (static_cast<size_t>(n_t) * static_cast<size_t>(d_inner)));

    மாறிலி தசம *s = src_data
        + static_cast<size_t>(seq) * static_cast<size_t>(src_stride_seq)
        + static_cast<size_t>(channel) * static_cast<size_t>(src_stride_inner)
        + static_cast<size_t>(token);

    மாறிலி தசம *c = weights + static_cast<size_t>(channel) * static_cast<size_t>(d_conv);

    தசம sumf = 0.0f;
    எனில் constexpr (DC > 0) {
#pragma unroll
        க்கு (முழு i0 = 0; i0 < DC; ++i0) {
            sumf += s[i0] * c[i0];
        }
    } இல்லையெனில் {
        க்கு (முழு i0 = 0; i0 < d_conv; ++i0) {
            sumf += s[i0] * c[i0];
        }
    }

    // fused bias add: the ADD node broadcasts a 1-D channel bias over tokens
    எனில் (bias != nullptr) {
        sumf += bias[channel];
    }

    மாறிலி size_t dst_idx =
        static_cast<size_t>(seq) * static_cast<size_t>(dst_stride_seq) +
        static_cast<size_t>(token) * static_cast<size_t>(dst_stride_token) +
        static_cast<size_t>(channel);

    dst_data[dst_idx] = apply_silu ? op_silu(sumf) : sumf;
}

// FUSED=false keeps apply_silu/bias out of the kernel capture list, so the unfused launch
// takes the pre-fusion argument list; matters at n_t == 1, where the op is launch-bound.
template <முழு DC, தருக்க FUSED>
நிலையான வெறுமை kernel_ssm_conv_impl(
    queue &q,
    மாறிலி தசம *src_data,
    மாறிலி தசம *weights,
    தசம *dst_data,
    முழு d_conv,
    முழு d_inner,
    முழு n_t,
    முழு n_s,
    முழு ncs __attribute__((unused)),
    முழு src_stride_inner,
    முழு src_stride_seq,
    முழு dst_stride_token,
    முழு dst_stride_seq,
    தருக்க apply_silu,
    மாறிலி தசம *bias
) {
    மாறிலி size_t total_work = static_cast<size_t>(d_inner) * static_cast<size_t>(n_t) * static_cast<size_t>(n_s);
    மாறிலி size_t work_group_size = 256;
    மாறிலி size_t இலக்கம்_work_groups = (total_work + work_group_size - 1) / work_group_size;

    மாறிலி range<1> global_range(இலக்கம்_work_groups * work_group_size);
    மாறிலி range<1> local_range(work_group_size);

    எனில் constexpr (FUSED) {
        q.submit([&](handler &h) {
            h.parallel_for(
                nd_range<1>(global_range, local_range),
                [=](nd_item<1> item) {
                    மாறிலி size_t idx = item.get_global_id(0);
                    எனில் (idx >= total_work) {
                        திரும்பு;
                    }

                    ssm_conv_element<DC>(idx, src_data, weights, dst_data, d_conv, d_inner, n_t,
                                         src_stride_inner, src_stride_seq, dst_stride_token,
                                         dst_stride_seq, apply_silu, bias);
                }
            );
        });
    } இல்லையெனில் {
        GGML_UNUSED(apply_silu);
        GGML_UNUSED(bias);

        q.submit([&](handler &h) {
            h.parallel_for(
                nd_range<1>(global_range, local_range),
                [=](nd_item<1> item) {
                    மாறிலி size_t idx = item.get_global_id(0);
                    எனில் (idx >= total_work) {
                        திரும்பு;
                    }

                    ssm_conv_element<DC>(idx, src_data, weights, dst_data, d_conv, d_inner, n_t,
                                         src_stride_inner, src_stride_seq, dst_stride_token,
                                         dst_stride_seq, பொய், nullptr);
                }
            );
        });
    }
}

// SLM transpose tile: coalesces both the loads and the stores. The +1 pad makes the row
// stride 33, coprime with 32 banks, so both phases are bank-conflict-free.
template <முழு DC, முழு TT, முழு TC, முழு WG>
நிலையான __dpct_inline__ வெறுமை ssm_conv_tile(
    nd_item<1> it, local_accessor<தசம, 1> tile, மாறிலி தசம *src_data, மாறிலி தசம *weights,
    தசம *dst_data, முழு n_t, முழு nt_tiles, முழு nc_tiles, முழு src_stride_inner,
    முழு src_stride_seq, முழு dst_stride_token, முழு dst_stride_seq, தருக்க apply_silu,
    மாறிலி தசம *bias
) {
    மாறிலி முழு    lid = static_cast<முழு>(it.get_local_id(0));
    மாறிலி size_t g   = it.get_group(0);
    மாறிலி முழு    tt  = static_cast<முழு>(g % nt_tiles);
    மாறிலி முழு    ct  = static_cast<முழு>((g / nt_tiles) % nc_tiles);
    மாறிலி முழு    seq = static_cast<முழு>(g / (static_cast<size_t>(nt_tiles) * nc_tiles));
    மாறிலி முழு    t0 = tt * TT, c0 = ct * TC;

    மாறிலி முழு ti = lid % TT;
    மாறிலி முழு cj = lid / TT;
#pragma unroll
    க்கு (முழு r = 0; r < TC / (WG / TT); ++r) {
        மாறிலி முழு c   = cj + r * (WG / TT);
        மாறிலி முழு tok = t0 + ti;
        தசம sumf = 0.0f;
        எனில் (tok < n_t) {
            மாறிலி தசம *s = src_data + static_cast<size_t>(seq) * src_stride_seq
                           + static_cast<size_t>(c0 + c) * src_stride_inner + tok;
            மாறிலி தசம *cw = weights + static_cast<size_t>(c0 + c) * DC;
#pragma unroll
            க்கு (முழு i = 0; i < DC; ++i) sumf += s[i] * cw[i];
            எனில் (bias != nullptr) sumf += bias[c0 + c];
            எனில் (apply_silu) sumf = op_silu(sumf);
        }
        tile[c * (TT + 1) + ti] = sumf;
    }
    it.barrier(access::fence_space::local_space);

    மாறிலி முழு cc = lid % TC;
    மாறிலி முழு tj = lid / TC;
#pragma unroll
    க்கு (முழு r = 0; r < TT / (WG / TC); ++r) {
        மாறிலி முழு t   = tj + r * (WG / TC);
        மாறிலி முழு tok = t0 + t;
        எனில் (tok < n_t) {
            dst_data[static_cast<size_t>(seq) * dst_stride_seq
                     + static_cast<size_t>(tok) * dst_stride_token + c0 + cc]
                = tile[cc * (TT + 1) + t];
        }
    }
}

// Same FUSED split as kernel_ssm_conv_impl. The fused instantiation keeps the runtime
// apply_silu/bias branches: at n_t >= 32 they are amortized over the whole tile.
template <முழு DC, தருக்க FUSED>
நிலையான வெறுமை kernel_ssm_conv_tiled(
    queue &q, மாறிலி தசம *src_data, மாறிலி தசம *weights, தசம *dst_data,
    முழு d_inner, முழு n_t, முழு n_s, முழு src_stride_inner, முழு src_stride_seq,
    முழு dst_stride_token, முழு dst_stride_seq, தருக்க apply_silu, மாறிலி தசம *bias
) {
    constexpr முழு TT = 32, TC = 32, WG = 256;
    மாறிலி முழு nt_tiles = (n_t + TT - 1) / TT;
    மாறிலி முழு nc_tiles = d_inner / TC;
    மாறிலி size_t groups = static_cast<size_t>(nt_tiles) * nc_tiles * n_s;

    எனில் constexpr (FUSED) {
        q.submit([&](handler &h) {
            local_accessor<தசம, 1> tile(range<1>(TC * (TT + 1)), h);
            h.parallel_for(nd_range<1>(range<1>(groups * WG), range<1>(WG)), [=](nd_item<1> it) {
                ssm_conv_tile<DC, TT, TC, WG>(it, tile, src_data, weights, dst_data, n_t, nt_tiles,
                                              nc_tiles, src_stride_inner, src_stride_seq,
                                              dst_stride_token, dst_stride_seq, apply_silu, bias);
            });
        });
    } இல்லையெனில் {
        GGML_UNUSED(apply_silu);
        GGML_UNUSED(bias);

        q.submit([&](handler &h) {
            local_accessor<தசம, 1> tile(range<1>(TC * (TT + 1)), h);
            h.parallel_for(nd_range<1>(range<1>(groups * WG), range<1>(WG)), [=](nd_item<1> it) {
                ssm_conv_tile<DC, TT, TC, WG>(it, tile, src_data, weights, dst_data, n_t, nt_tiles,
                                              nc_tiles, src_stride_inner, src_stride_seq,
                                              dst_stride_token, dst_stride_seq, பொய், nullptr);
            });
        });
    }
}

நிலையான வெறுமை kernel_ssm_conv(
    queue &q,
    மாறிலி தசம *src_data,
    மாறிலி தசம *weights,
    தசம *dst_data,
    முழு d_conv,
    முழு d_inner,
    முழு n_t,
    முழு n_s,
    முழு ncs,
    முழு src_stride_inner,
    முழு src_stride_seq,
    முழு dst_stride_token,
    முழு dst_stride_seq,
    தருக்க apply_silu,
    மாறிலி தசம *bias
) {
    // Only the fused instantiations carry apply_silu/bias as kernel arguments; the plain
    // ssm_conv launch keeps the argument list it had before the fusion landed.
    மாறிலி தருக்க fused = apply_silu || bias != nullptr;

    // d_inner must be a multiple of 32 so the channel tiles are exact; the transpose is only
    // worth it for n_t >= 32. d_conv == 4 is the only window with a DC-specialized kernel.
    எனில் (d_conv == 4 && n_t >= 32 && (d_inner % 32) == 0) {
        எனில் (fused) {
            kernel_ssm_conv_tiled<4, உண்மை>(q, src_data, weights, dst_data, d_inner, n_t, n_s,
                                           src_stride_inner, src_stride_seq, dst_stride_token,
                                           dst_stride_seq, apply_silu, bias);
        } இல்லையெனில் {
            kernel_ssm_conv_tiled<4, பொய்>(q, src_data, weights, dst_data, d_inner, n_t, n_s,
                                            src_stride_inner, src_stride_seq, dst_stride_token,
                                            dst_stride_seq, apply_silu, bias);
        }
        திரும்பு;
    }

    எனில் (d_conv == 4) {
        எனில் (fused) {
            kernel_ssm_conv_impl<4, உண்மை>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                          ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                          dst_stride_seq, apply_silu, bias);
        } இல்லையெனில் {
            kernel_ssm_conv_impl<4, பொய்>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                           ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                           dst_stride_seq, apply_silu, bias);
        }
        திரும்பு;
    }

    எனில் (fused) {
        kernel_ssm_conv_impl<0, உண்மை>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                      ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                      dst_stride_seq, apply_silu, bias);
    } இல்லையெனில் {
        kernel_ssm_conv_impl<0, பொய்>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                       ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                       dst_stride_seq, apply_silu, bias);
    }
}

உள்வரி வெறுமை ggml_sycl_op_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * silu_dst = nullptr, மாறிலி தசம * bias = nullptr) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(bias == nullptr || silu_dst != nullptr);

    மாறிலி முழு d_conv   = src1->ne[0];
    மாறிலி முழு ncs      = src0->ne[0];
    மாறிலி முழு d_inner  = src0->ne[1];
    மாறிலி முழு n_t      = dst->ne[1];
    மாறிலி முழு n_s      = dst->ne[2];

    GGML_ASSERT(src0->ne[0] == d_conv - 1 + n_t);
    GGML_ASSERT(src0->ne[1] == d_inner);
    GGML_ASSERT(src1->ne[1] == d_inner);

    GGML_ASSERT(dst->ne[0] == d_inner);
    GGML_ASSERT(dst->ne[1] == n_t);
    GGML_ASSERT(dst->ne[2] == n_s);

    GGML_ASSERT(src0->nb[0] == அளவு(தசம));
    GGML_ASSERT(src1->nb[0] == அளவு(தசம));

    GGML_ASSERT(src0->nb[1] == src0->ne[0] * அளவு(தசம));

    மாறிலி முழு src_stride_inner = ncs;
    மாறிலி முழு src_stride_seq   = ncs * d_inner;
    மாறிலி முழு dst_stride_token = d_inner;
    மாறிலி முழு dst_stride_seq   = d_inner * n_t;

    முயல் {
        queue *q = ctx.stream();

        மாறிலி தசம *src_data = static_cast<மாறிலி தசம *>(src0->data);
        மாறிலி தசம *weights  = static_cast<மாறிலி தசம *>(src1->data);
        மாறிலி தருக்க apply_silu = silu_dst != nullptr;
        தசம *dst_data       = static_cast<தசம *>((silu_dst ? silu_dst : dst)->data);

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

    } பிடி (மாறிலி std::exception &e) {
        std::fprintf(stderr, "[SYCL-SSM_CONV] ERROR: %s\n", e.what());
        எறி;
    }
}

வெறுமை ggml_sycl_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_ssm_conv(ctx, dst);
}

// Fused ssm_conv + ADD + SiLU: write silu(conv(x) + b) straight into silu_dst, eliding the
// standalone SiLU launch and its HBM round-trip of the conv output.
வெறுமை ggml_sycl_ssm_conv_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * add, ggml_tensor * silu_dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    GGML_ASSERT(silu_dst && ggml_are_same_shape(dst, silu_dst) && silu_dst->type == GGML_TYPE_F32);
    // the fused kernel reads only the ADD's bias operand; the ADD result is never written
    மாறிலி தசம * bias = nullptr;
    எனில் (add != nullptr) {
        மாறிலி ggml_tensor * bias_t = (add->src[0] == dst) ? add->src[1] : add->src[0];
        bias = static_cast<மாறிலி தசம *>(bias_t->data);
    }
    ggml_sycl_op_ssm_conv(ctx, dst, silu_dst, bias);
}
