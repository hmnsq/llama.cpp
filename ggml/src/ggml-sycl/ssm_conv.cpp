#شامل "ssm_conv.hpp"
#شامل "common.hpp"
#شامل "element_wise.hpp"

#شامل <cstdio>

using فضاینام sycl;

// One output element of the conv. DC is d_conv as a compile-time constant (0 keeps the
// runtime loop); unfused callers pass literal false/nullptr so the epilogue folds away.
template <صحیح DC>
ایستا __dpct_inline__ تهی ssm_conv_element(
    size_t idx,
    ثابت اعشاری *src_data,
    ثابت اعشاری *weights,
    اعشاری *dst_data,
    صحیح d_conv,
    صحیح d_inner,
    صحیح n_t,
    صحیح src_stride_inner,
    صحیح src_stride_seq,
    صحیح dst_stride_token,
    صحیح dst_stride_seq,
    منطقی apply_silu,
    ثابت اعشاری *bias
) {
    // src is token-contiguous per channel, dst is channel-contiguous per token,
    // so indexing token-fastest coalesces the d_conv loads.
    ثابت صحیح token   = static_cast<صحیح>(idx % n_t);
    ثابت صحیح channel = static_cast<صحیح>((idx / n_t) % d_inner);
    ثابت صحیح seq     = static_cast<صحیح>(idx / (static_cast<size_t>(n_t) * static_cast<size_t>(d_inner)));

    ثابت اعشاری *s = src_data
        + static_cast<size_t>(seq) * static_cast<size_t>(src_stride_seq)
        + static_cast<size_t>(channel) * static_cast<size_t>(src_stride_inner)
        + static_cast<size_t>(token);

    ثابت اعشاری *c = weights + static_cast<size_t>(channel) * static_cast<size_t>(d_conv);

    اعشاری sumf = 0.0f;
    اگر constexpr (DC > 0) {
#pragma unroll
        برای (صحیح i0 = 0; i0 < DC; ++i0) {
            sumf += s[i0] * c[i0];
        }
    } وگرنه {
        برای (صحیح i0 = 0; i0 < d_conv; ++i0) {
            sumf += s[i0] * c[i0];
        }
    }

    // fused bias add: the ADD node broadcasts a 1-D channel bias over tokens
    اگر (bias != nullptr) {
        sumf += bias[channel];
    }

    ثابت size_t dst_idx =
        static_cast<size_t>(seq) * static_cast<size_t>(dst_stride_seq) +
        static_cast<size_t>(token) * static_cast<size_t>(dst_stride_token) +
        static_cast<size_t>(channel);

    dst_data[dst_idx] = apply_silu ? op_silu(sumf) : sumf;
}

// FUSED=false keeps apply_silu/bias out of the kernel capture list, so the unfused launch
// takes the pre-fusion argument list; matters at n_t == 1, where the op is launch-bound.
template <صحیح DC, منطقی FUSED>
ایستا تهی kernel_ssm_conv_impl(
    queue &q,
    ثابت اعشاری *src_data,
    ثابت اعشاری *weights,
    اعشاری *dst_data,
    صحیح d_conv,
    صحیح d_inner,
    صحیح n_t,
    صحیح n_s,
    صحیح ncs __attribute__((unused)),
    صحیح src_stride_inner,
    صحیح src_stride_seq,
    صحیح dst_stride_token,
    صحیح dst_stride_seq,
    منطقی apply_silu,
    ثابت اعشاری *bias
) {
    ثابت size_t total_work = static_cast<size_t>(d_inner) * static_cast<size_t>(n_t) * static_cast<size_t>(n_s);
    ثابت size_t work_group_size = 256;
    ثابت size_t رقم_work_groups = (total_work + work_group_size - 1) / work_group_size;

    ثابت range<1> global_range(رقم_work_groups * work_group_size);
    ثابت range<1> local_range(work_group_size);

    اگر constexpr (FUSED) {
        q.submit([&](handler &h) {
            h.parallel_for(
                nd_range<1>(global_range, local_range),
                [=](nd_item<1> item) {
                    ثابت size_t idx = item.get_global_id(0);
                    اگر (idx >= total_work) {
                        بازگشت;
                    }

                    ssm_conv_element<DC>(idx, src_data, weights, dst_data, d_conv, d_inner, n_t,
                                         src_stride_inner, src_stride_seq, dst_stride_token,
                                         dst_stride_seq, apply_silu, bias);
                }
            );
        });
    } وگرنه {
        GGML_UNUSED(apply_silu);
        GGML_UNUSED(bias);

        q.submit([&](handler &h) {
            h.parallel_for(
                nd_range<1>(global_range, local_range),
                [=](nd_item<1> item) {
                    ثابت size_t idx = item.get_global_id(0);
                    اگر (idx >= total_work) {
                        بازگشت;
                    }

                    ssm_conv_element<DC>(idx, src_data, weights, dst_data, d_conv, d_inner, n_t,
                                         src_stride_inner, src_stride_seq, dst_stride_token,
                                         dst_stride_seq, نادرست, nullptr);
                }
            );
        });
    }
}

// SLM transpose tile: coalesces both the loads and the stores. The +1 pad makes the row
// stride 33, coprime with 32 banks, so both phases are bank-conflict-free.
template <صحیح DC, صحیح TT, صحیح TC, صحیح WG>
ایستا __dpct_inline__ تهی ssm_conv_tile(
    nd_item<1> it, local_accessor<اعشاری, 1> tile, ثابت اعشاری *src_data, ثابت اعشاری *weights,
    اعشاری *dst_data, صحیح n_t, صحیح nt_tiles, صحیح nc_tiles, صحیح src_stride_inner,
    صحیح src_stride_seq, صحیح dst_stride_token, صحیح dst_stride_seq, منطقی apply_silu,
    ثابت اعشاری *bias
) {
    ثابت صحیح    lid = static_cast<صحیح>(it.get_local_id(0));
    ثابت size_t g   = it.get_group(0);
    ثابت صحیح    tt  = static_cast<صحیح>(g % nt_tiles);
    ثابت صحیح    ct  = static_cast<صحیح>((g / nt_tiles) % nc_tiles);
    ثابت صحیح    seq = static_cast<صحیح>(g / (static_cast<size_t>(nt_tiles) * nc_tiles));
    ثابت صحیح    t0 = tt * TT, c0 = ct * TC;

    ثابت صحیح ti = lid % TT;
    ثابت صحیح cj = lid / TT;
#pragma unroll
    برای (صحیح r = 0; r < TC / (WG / TT); ++r) {
        ثابت صحیح c   = cj + r * (WG / TT);
        ثابت صحیح tok = t0 + ti;
        اعشاری sumf = 0.0f;
        اگر (tok < n_t) {
            ثابت اعشاری *s = src_data + static_cast<size_t>(seq) * src_stride_seq
                           + static_cast<size_t>(c0 + c) * src_stride_inner + tok;
            ثابت اعشاری *cw = weights + static_cast<size_t>(c0 + c) * DC;
#pragma unroll
            برای (صحیح i = 0; i < DC; ++i) sumf += s[i] * cw[i];
            اگر (bias != nullptr) sumf += bias[c0 + c];
            اگر (apply_silu) sumf = op_silu(sumf);
        }
        tile[c * (TT + 1) + ti] = sumf;
    }
    it.barrier(access::fence_space::local_space);

    ثابت صحیح cc = lid % TC;
    ثابت صحیح tj = lid / TC;
#pragma unroll
    برای (صحیح r = 0; r < TT / (WG / TC); ++r) {
        ثابت صحیح t   = tj + r * (WG / TC);
        ثابت صحیح tok = t0 + t;
        اگر (tok < n_t) {
            dst_data[static_cast<size_t>(seq) * dst_stride_seq
                     + static_cast<size_t>(tok) * dst_stride_token + c0 + cc]
                = tile[cc * (TT + 1) + t];
        }
    }
}

// Same FUSED split as kernel_ssm_conv_impl. The fused instantiation keeps the runtime
// apply_silu/bias branches: at n_t >= 32 they are amortized over the whole tile.
template <صحیح DC, منطقی FUSED>
ایستا تهی kernel_ssm_conv_tiled(
    queue &q, ثابت اعشاری *src_data, ثابت اعشاری *weights, اعشاری *dst_data,
    صحیح d_inner, صحیح n_t, صحیح n_s, صحیح src_stride_inner, صحیح src_stride_seq,
    صحیح dst_stride_token, صحیح dst_stride_seq, منطقی apply_silu, ثابت اعشاری *bias
) {
    constexpr صحیح TT = 32, TC = 32, WG = 256;
    ثابت صحیح nt_tiles = (n_t + TT - 1) / TT;
    ثابت صحیح nc_tiles = d_inner / TC;
    ثابت size_t groups = static_cast<size_t>(nt_tiles) * nc_tiles * n_s;

    اگر constexpr (FUSED) {
        q.submit([&](handler &h) {
            local_accessor<اعشاری, 1> tile(range<1>(TC * (TT + 1)), h);
            h.parallel_for(nd_range<1>(range<1>(groups * WG), range<1>(WG)), [=](nd_item<1> it) {
                ssm_conv_tile<DC, TT, TC, WG>(it, tile, src_data, weights, dst_data, n_t, nt_tiles,
                                              nc_tiles, src_stride_inner, src_stride_seq,
                                              dst_stride_token, dst_stride_seq, apply_silu, bias);
            });
        });
    } وگرنه {
        GGML_UNUSED(apply_silu);
        GGML_UNUSED(bias);

        q.submit([&](handler &h) {
            local_accessor<اعشاری, 1> tile(range<1>(TC * (TT + 1)), h);
            h.parallel_for(nd_range<1>(range<1>(groups * WG), range<1>(WG)), [=](nd_item<1> it) {
                ssm_conv_tile<DC, TT, TC, WG>(it, tile, src_data, weights, dst_data, n_t, nt_tiles,
                                              nc_tiles, src_stride_inner, src_stride_seq,
                                              dst_stride_token, dst_stride_seq, نادرست, nullptr);
            });
        });
    }
}

ایستا تهی kernel_ssm_conv(
    queue &q,
    ثابت اعشاری *src_data,
    ثابت اعشاری *weights,
    اعشاری *dst_data,
    صحیح d_conv,
    صحیح d_inner,
    صحیح n_t,
    صحیح n_s,
    صحیح ncs,
    صحیح src_stride_inner,
    صحیح src_stride_seq,
    صحیح dst_stride_token,
    صحیح dst_stride_seq,
    منطقی apply_silu,
    ثابت اعشاری *bias
) {
    // Only the fused instantiations carry apply_silu/bias as kernel arguments; the plain
    // ssm_conv launch keeps the argument list it had before the fusion landed.
    ثابت منطقی fused = apply_silu || bias != nullptr;

    // d_inner must be a multiple of 32 so the channel tiles are exact; the transpose is only
    // worth it for n_t >= 32. d_conv == 4 is the only window with a DC-specialized kernel.
    اگر (d_conv == 4 && n_t >= 32 && (d_inner % 32) == 0) {
        اگر (fused) {
            kernel_ssm_conv_tiled<4, درست>(q, src_data, weights, dst_data, d_inner, n_t, n_s,
                                           src_stride_inner, src_stride_seq, dst_stride_token,
                                           dst_stride_seq, apply_silu, bias);
        } وگرنه {
            kernel_ssm_conv_tiled<4, نادرست>(q, src_data, weights, dst_data, d_inner, n_t, n_s,
                                            src_stride_inner, src_stride_seq, dst_stride_token,
                                            dst_stride_seq, apply_silu, bias);
        }
        بازگشت;
    }

    اگر (d_conv == 4) {
        اگر (fused) {
            kernel_ssm_conv_impl<4, درست>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                          ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                          dst_stride_seq, apply_silu, bias);
        } وگرنه {
            kernel_ssm_conv_impl<4, نادرست>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                           ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                           dst_stride_seq, apply_silu, bias);
        }
        بازگشت;
    }

    اگر (fused) {
        kernel_ssm_conv_impl<0, درست>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                      ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                      dst_stride_seq, apply_silu, bias);
    } وگرنه {
        kernel_ssm_conv_impl<0, نادرست>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                       ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                       dst_stride_seq, apply_silu, bias);
    }
}

درونخطی تهی ggml_sycl_op_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * silu_dst = nullptr, ثابت اعشاری * bias = nullptr) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(bias == nullptr || silu_dst != nullptr);

    ثابت صحیح d_conv   = src1->ne[0];
    ثابت صحیح ncs      = src0->ne[0];
    ثابت صحیح d_inner  = src0->ne[1];
    ثابت صحیح n_t      = dst->ne[1];
    ثابت صحیح n_s      = dst->ne[2];

    GGML_ASSERT(src0->ne[0] == d_conv - 1 + n_t);
    GGML_ASSERT(src0->ne[1] == d_inner);
    GGML_ASSERT(src1->ne[1] == d_inner);

    GGML_ASSERT(dst->ne[0] == d_inner);
    GGML_ASSERT(dst->ne[1] == n_t);
    GGML_ASSERT(dst->ne[2] == n_s);

    GGML_ASSERT(src0->nb[0] == اندازه(اعشاری));
    GGML_ASSERT(src1->nb[0] == اندازه(اعشاری));

    GGML_ASSERT(src0->nb[1] == src0->ne[0] * اندازه(اعشاری));

    ثابت صحیح src_stride_inner = ncs;
    ثابت صحیح src_stride_seq   = ncs * d_inner;
    ثابت صحیح dst_stride_token = d_inner;
    ثابت صحیح dst_stride_seq   = d_inner * n_t;

    تلاش {
        queue *q = ctx.stream();

        ثابت اعشاری *src_data = static_cast<ثابت اعشاری *>(src0->data);
        ثابت اعشاری *weights  = static_cast<ثابت اعشاری *>(src1->data);
        ثابت منطقی apply_silu = silu_dst != nullptr;
        اعشاری *dst_data       = static_cast<اعشاری *>((silu_dst ? silu_dst : dst)->data);

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

    } بگیر (ثابت std::exception &e) {
        std::fprintf(stderr, "[SYCL-SSM_CONV] ERROR: %s\n", e.what());
        پرتابکن;
    }
}

تهی ggml_sycl_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_ssm_conv(ctx, dst);
}

// Fused ssm_conv + ADD + SiLU: write silu(conv(x) + b) straight into silu_dst, eliding the
// standalone SiLU launch and its HBM round-trip of the conv output.
تهی ggml_sycl_ssm_conv_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * add, ggml_tensor * silu_dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    GGML_ASSERT(silu_dst && ggml_are_same_shape(dst, silu_dst) && silu_dst->type == GGML_TYPE_F32);
    // the fused kernel reads only the ADD's bias operand; the ADD result is never written
    ثابت اعشاری * bias = nullptr;
    اگر (add != nullptr) {
        ثابت ggml_tensor * bias_t = (add->src[0] == dst) ? add->src[1] : add->src[0];
        bias = static_cast<ثابت اعشاری *>(bias_t->data);
    }
    ggml_sycl_op_ssm_conv(ctx, dst, silu_dst, bias);
}
