#अन्तर्भाव "ssm_conv.hpp"
#अन्तर्भाव "common.hpp"
#अन्तर्भाव "element_wise.hpp"

#अन्तर्भाव <cstdio>

using नामाकाश sycl;

// One output element of the conv. DC is d_conv as a compile-time constant (0 keeps the
// runtime loop); unfused callers pass literal false/nullptr so the epilogue folds away.
ढाँचा <पूर्णाङ्क DC>
स्थिर __dpct_inline__ शून्य ssm_conv_element(
    माप_प्रकार idx,
    अचर भिन्न *src_data,
    अचर भिन्न *weights,
    भिन्न *dst_data,
    पूर्णाङ्क d_conv,
    पूर्णाङ्क d_inner,
    पूर्णाङ्क n_t,
    पूर्णाङ्क src_stride_inner,
    पूर्णाङ्क src_stride_seq,
    पूर्णाङ्क dst_stride_token,
    पूर्णाङ्क dst_stride_seq,
    तार्किक apply_silu,
    अचर भिन्न *bias
) {
    // src is token-contiguous per channel, dst is channel-contiguous per token,
    // so indexing token-fastest coalesces the d_conv loads.
    अचर पूर्णाङ्क token   = static_cast<पूर्णाङ्क>(idx % n_t);
    अचर पूर्णाङ्क channel = static_cast<पूर्णाङ्क>((idx / n_t) % d_inner);
    अचर पूर्णाङ्क seq     = static_cast<पूर्णाङ्क>(idx / (static_cast<माप_प्रकार>(n_t) * static_cast<माप_प्रकार>(d_inner)));

    अचर भिन्न *s = src_data
        + static_cast<माप_प्रकार>(seq) * static_cast<माप_प्रकार>(src_stride_seq)
        + static_cast<माप_प्रकार>(channel) * static_cast<माप_प्रकार>(src_stride_inner)
        + static_cast<माप_प्रकार>(token);

    अचर भिन्न *c = weights + static_cast<माप_प्रकार>(channel) * static_cast<माप_प्रकार>(d_conv);

    भिन्न sumf = 0.0f;
    यदि constexpr (DC > 0) {
#pragma unroll
        लागि (पूर्णाङ्क i0 = 0; i0 < DC; ++i0) {
            sumf += s[i0] * c[i0];
        }
    } अन्यथा {
        लागि (पूर्णाङ्क i0 = 0; i0 < d_conv; ++i0) {
            sumf += s[i0] * c[i0];
        }
    }

    // fused bias add: the ADD node broadcasts a 1-D channel bias over tokens
    यदि (bias != nullptr) {
        sumf += bias[channel];
    }

    अचर माप_प्रकार dst_idx =
        static_cast<माप_प्रकार>(seq) * static_cast<माप_प्रकार>(dst_stride_seq) +
        static_cast<माप_प्रकार>(token) * static_cast<माप_प्रकार>(dst_stride_token) +
        static_cast<माप_प्रकार>(channel);

    dst_data[dst_idx] = apply_silu ? op_silu(sumf) : sumf;
}

// FUSED=false keeps apply_silu/bias out of the kernel capture list, so the unfused launch
// takes the pre-fusion argument list; matters at n_t == 1, where the op is launch-bound.
ढाँचा <पूर्णाङ्क DC, तार्किक FUSED>
स्थिर शून्य kernel_ssm_conv_impl(
    queue &q,
    अचर भिन्न *src_data,
    अचर भिन्न *weights,
    भिन्न *dst_data,
    पूर्णाङ्क d_conv,
    पूर्णाङ्क d_inner,
    पूर्णाङ्क n_t,
    पूर्णाङ्क n_s,
    पूर्णाङ्क ncs __attribute__((unused)),
    पूर्णाङ्क src_stride_inner,
    पूर्णाङ्क src_stride_seq,
    पूर्णाङ्क dst_stride_token,
    पूर्णाङ्क dst_stride_seq,
    तार्किक apply_silu,
    अचर भिन्न *bias
) {
    अचर माप_प्रकार total_work = static_cast<माप_प्रकार>(d_inner) * static_cast<माप_प्रकार>(n_t) * static_cast<माप_प्रकार>(n_s);
    अचर माप_प्रकार work_group_size = 256;
    अचर माप_प्रकार अङ्क_work_groups = (total_work + work_group_size - 1) / work_group_size;

    अचर range<1> global_range(अङ्क_work_groups * work_group_size);
    अचर range<1> local_range(work_group_size);

    यदि constexpr (FUSED) {
        q.submit([&](handler &h) {
            h.parallel_for(
                nd_range<1>(global_range, local_range),
                [=](nd_item<1> item) {
                    अचर माप_प्रकार idx = item.get_global_id(0);
                    यदि (idx >= total_work) {
                        फर्काऊ;
                    }

                    ssm_conv_element<DC>(idx, src_data, weights, dst_data, d_conv, d_inner, n_t,
                                         src_stride_inner, src_stride_seq, dst_stride_token,
                                         dst_stride_seq, apply_silu, bias);
                }
            );
        });
    } अन्यथा {
        GGML_UNUSED(apply_silu);
        GGML_UNUSED(bias);

        q.submit([&](handler &h) {
            h.parallel_for(
                nd_range<1>(global_range, local_range),
                [=](nd_item<1> item) {
                    अचर माप_प्रकार idx = item.get_global_id(0);
                    यदि (idx >= total_work) {
                        फर्काऊ;
                    }

                    ssm_conv_element<DC>(idx, src_data, weights, dst_data, d_conv, d_inner, n_t,
                                         src_stride_inner, src_stride_seq, dst_stride_token,
                                         dst_stride_seq, असत्य, nullptr);
                }
            );
        });
    }
}

// SLM transpose tile: coalesces both the loads and the stores. The +1 pad makes the row
// stride 33, coprime with 32 banks, so both phases are bank-conflict-free.
ढाँचा <पूर्णाङ्क DC, पूर्णाङ्क TT, पूर्णाङ्क TC, पूर्णाङ्क WG>
स्थिर __dpct_inline__ शून्य ssm_conv_tile(
    nd_item<1> it, local_accessor<भिन्न, 1> tile, अचर भिन्न *src_data, अचर भिन्न *weights,
    भिन्न *dst_data, पूर्णाङ्क n_t, पूर्णाङ्क nt_tiles, पूर्णाङ्क nc_tiles, पूर्णाङ्क src_stride_inner,
    पूर्णाङ्क src_stride_seq, पूर्णाङ्क dst_stride_token, पूर्णाङ्क dst_stride_seq, तार्किक apply_silu,
    अचर भिन्न *bias
) {
    अचर पूर्णाङ्क    lid = static_cast<पूर्णाङ्क>(it.get_local_id(0));
    अचर माप_प्रकार g   = it.get_group(0);
    अचर पूर्णाङ्क    tt  = static_cast<पूर्णाङ्क>(g % nt_tiles);
    अचर पूर्णाङ्क    ct  = static_cast<पूर्णाङ्क>((g / nt_tiles) % nc_tiles);
    अचर पूर्णाङ्क    seq = static_cast<पूर्णाङ्क>(g / (static_cast<माप_प्रकार>(nt_tiles) * nc_tiles));
    अचर पूर्णाङ्क    t0 = tt * TT, c0 = ct * TC;

    अचर पूर्णाङ्क ti = lid % TT;
    अचर पूर्णाङ्क cj = lid / TT;
#pragma unroll
    लागि (पूर्णाङ्क r = 0; r < TC / (WG / TT); ++r) {
        अचर पूर्णाङ्क c   = cj + r * (WG / TT);
        अचर पूर्णाङ्क tok = t0 + ti;
        भिन्न sumf = 0.0f;
        यदि (tok < n_t) {
            अचर भिन्न *s = src_data + static_cast<माप_प्रकार>(seq) * src_stride_seq
                           + static_cast<माप_प्रकार>(c0 + c) * src_stride_inner + tok;
            अचर भिन्न *cw = weights + static_cast<माप_प्रकार>(c0 + c) * DC;
#pragma unroll
            लागि (पूर्णाङ्क i = 0; i < DC; ++i) sumf += s[i] * cw[i];
            यदि (bias != nullptr) sumf += bias[c0 + c];
            यदि (apply_silu) sumf = op_silu(sumf);
        }
        tile[c * (TT + 1) + ti] = sumf;
    }
    it.barrier(access::fence_space::local_space);

    अचर पूर्णाङ्क cc = lid % TC;
    अचर पूर्णाङ्क tj = lid / TC;
#pragma unroll
    लागि (पूर्णाङ्क r = 0; r < TT / (WG / TC); ++r) {
        अचर पूर्णाङ्क t   = tj + r * (WG / TC);
        अचर पूर्णाङ्क tok = t0 + t;
        यदि (tok < n_t) {
            dst_data[static_cast<माप_प्रकार>(seq) * dst_stride_seq
                     + static_cast<माप_प्रकार>(tok) * dst_stride_token + c0 + cc]
                = tile[cc * (TT + 1) + t];
        }
    }
}

// Same FUSED split as kernel_ssm_conv_impl. The fused instantiation keeps the runtime
// apply_silu/bias branches: at n_t >= 32 they are amortized over the whole tile.
ढाँचा <पूर्णाङ्क DC, तार्किक FUSED>
स्थिर शून्य kernel_ssm_conv_tiled(
    queue &q, अचर भिन्न *src_data, अचर भिन्न *weights, भिन्न *dst_data,
    पूर्णाङ्क d_inner, पूर्णाङ्क n_t, पूर्णाङ्क n_s, पूर्णाङ्क src_stride_inner, पूर्णाङ्क src_stride_seq,
    पूर्णाङ्क dst_stride_token, पूर्णाङ्क dst_stride_seq, तार्किक apply_silu, अचर भिन्न *bias
) {
    constexpr पूर्णाङ्क TT = 32, TC = 32, WG = 256;
    अचर पूर्णाङ्क nt_tiles = (n_t + TT - 1) / TT;
    अचर पूर्णाङ्क nc_tiles = d_inner / TC;
    अचर माप_प्रकार groups = static_cast<माप_प्रकार>(nt_tiles) * nc_tiles * n_s;

    यदि constexpr (FUSED) {
        q.submit([&](handler &h) {
            local_accessor<भिन्न, 1> tile(range<1>(TC * (TT + 1)), h);
            h.parallel_for(nd_range<1>(range<1>(groups * WG), range<1>(WG)), [=](nd_item<1> it) {
                ssm_conv_tile<DC, TT, TC, WG>(it, tile, src_data, weights, dst_data, n_t, nt_tiles,
                                              nc_tiles, src_stride_inner, src_stride_seq,
                                              dst_stride_token, dst_stride_seq, apply_silu, bias);
            });
        });
    } अन्यथा {
        GGML_UNUSED(apply_silu);
        GGML_UNUSED(bias);

        q.submit([&](handler &h) {
            local_accessor<भिन्न, 1> tile(range<1>(TC * (TT + 1)), h);
            h.parallel_for(nd_range<1>(range<1>(groups * WG), range<1>(WG)), [=](nd_item<1> it) {
                ssm_conv_tile<DC, TT, TC, WG>(it, tile, src_data, weights, dst_data, n_t, nt_tiles,
                                              nc_tiles, src_stride_inner, src_stride_seq,
                                              dst_stride_token, dst_stride_seq, असत्य, nullptr);
            });
        });
    }
}

स्थिर शून्य kernel_ssm_conv(
    queue &q,
    अचर भिन्न *src_data,
    अचर भिन्न *weights,
    भिन्न *dst_data,
    पूर्णाङ्क d_conv,
    पूर्णाङ्क d_inner,
    पूर्णाङ्क n_t,
    पूर्णाङ्क n_s,
    पूर्णाङ्क ncs,
    पूर्णाङ्क src_stride_inner,
    पूर्णाङ्क src_stride_seq,
    पूर्णाङ्क dst_stride_token,
    पूर्णाङ्क dst_stride_seq,
    तार्किक apply_silu,
    अचर भिन्न *bias
) {
    // Only the fused instantiations carry apply_silu/bias as kernel arguments; the plain
    // ssm_conv launch keeps the argument list it had before the fusion landed.
    अचर तार्किक fused = apply_silu || bias != nullptr;

    // d_inner must be a multiple of 32 so the channel tiles are exact; the transpose is only
    // worth it for n_t >= 32. d_conv == 4 is the only window with a DC-specialized kernel.
    यदि (d_conv == 4 && n_t >= 32 && (d_inner % 32) == 0) {
        यदि (fused) {
            kernel_ssm_conv_tiled<4, सत्य>(q, src_data, weights, dst_data, d_inner, n_t, n_s,
                                           src_stride_inner, src_stride_seq, dst_stride_token,
                                           dst_stride_seq, apply_silu, bias);
        } अन्यथा {
            kernel_ssm_conv_tiled<4, असत्य>(q, src_data, weights, dst_data, d_inner, n_t, n_s,
                                            src_stride_inner, src_stride_seq, dst_stride_token,
                                            dst_stride_seq, apply_silu, bias);
        }
        फर्काऊ;
    }

    यदि (d_conv == 4) {
        यदि (fused) {
            kernel_ssm_conv_impl<4, सत्य>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                          ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                          dst_stride_seq, apply_silu, bias);
        } अन्यथा {
            kernel_ssm_conv_impl<4, असत्य>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                           ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                           dst_stride_seq, apply_silu, bias);
        }
        फर्काऊ;
    }

    यदि (fused) {
        kernel_ssm_conv_impl<0, सत्य>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                      ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                      dst_stride_seq, apply_silu, bias);
    } अन्यथा {
        kernel_ssm_conv_impl<0, असत्य>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                       ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                       dst_stride_seq, apply_silu, bias);
    }
}

अन्तःस्थ शून्य ggml_sycl_op_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * silu_dst = nullptr, अचर भिन्न * bias = nullptr) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(bias == nullptr || silu_dst != nullptr);

    अचर पूर्णाङ्क d_conv   = src1->ne[0];
    अचर पूर्णाङ्क ncs      = src0->ne[0];
    अचर पूर्णाङ्क d_inner  = src0->ne[1];
    अचर पूर्णाङ्क n_t      = dst->ne[1];
    अचर पूर्णाङ्क n_s      = dst->ne[2];

    GGML_ASSERT(src0->ne[0] == d_conv - 1 + n_t);
    GGML_ASSERT(src0->ne[1] == d_inner);
    GGML_ASSERT(src1->ne[1] == d_inner);

    GGML_ASSERT(dst->ne[0] == d_inner);
    GGML_ASSERT(dst->ne[1] == n_t);
    GGML_ASSERT(dst->ne[2] == n_s);

    GGML_ASSERT(src0->nb[0] == परिमाण(भिन्न));
    GGML_ASSERT(src1->nb[0] == परिमाण(भिन्न));

    GGML_ASSERT(src0->nb[1] == src0->ne[0] * परिमाण(भिन्न));

    अचर पूर्णाङ्क src_stride_inner = ncs;
    अचर पूर्णाङ्क src_stride_seq   = ncs * d_inner;
    अचर पूर्णाङ्क dst_stride_token = d_inner;
    अचर पूर्णाङ्क dst_stride_seq   = d_inner * n_t;

    यत्न {
        queue *q = ctx.stream();

        अचर भिन्न *src_data = static_cast<अचर भिन्न *>(src0->data);
        अचर भिन्न *weights  = static_cast<अचर भिन्न *>(src1->data);
        अचर तार्किक apply_silu = silu_dst != nullptr;
        भिन्न *dst_data       = static_cast<भिन्न *>((silu_dst ? silu_dst : dst)->data);

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

    } गृहाण (अचर std::exception &e) {
        std::ख_लिखो(मानक_त्रुटि, "[SYCL-SSM_CONV] ERROR: %s\n", e.what());
        क्षिप;
    }
}

शून्य ggml_sycl_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_ssm_conv(ctx, dst);
}

// Fused ssm_conv + ADD + SiLU: write silu(conv(x) + b) straight into silu_dst, eliding the
// standalone SiLU launch and its HBM round-trip of the conv output.
शून्य ggml_sycl_ssm_conv_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * add, ggml_tensor * silu_dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    GGML_ASSERT(silu_dst && ggml_are_same_shape(dst, silu_dst) && silu_dst->type == GGML_TYPE_F32);
    // the fused kernel reads only the ADD's bias operand; the ADD result is never written
    अचर भिन्न * bias = nullptr;
    यदि (add != nullptr) {
        अचर ggml_tensor * bias_t = (add->src[0] == dst) ? add->src[1] : add->src[0];
        bias = static_cast<अचर भिन्न *>(bias_t->data);
    }
    ggml_sycl_op_ssm_conv(ctx, dst, silu_dst, bias);
}
