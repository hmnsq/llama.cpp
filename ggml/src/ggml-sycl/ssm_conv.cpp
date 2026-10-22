#include "ssm_conv.hpp"
#include "common.hpp"
#include "element_wise.hpp"

#include <cstdio>

using namespace sycl;

// One output element of the conv. DC is d_conv as a compile-time constant (0 keeps the
// runtime loop); unfused callers pass literal false/nullptr so the epilogue folds away.
template <ཧྲིལ DC>
static __dpct_inline__ void ssm_conv_element(
    size_t idx,
    const float *src_data,
    const float *weights,
    float *dst_data,
    ཧྲིལ d_conv,
    ཧྲིལ d_inner,
    ཧྲིལ n_t,
    ཧྲིལ src_stride_inner,
    ཧྲིལ src_stride_seq,
    ཧྲིལ dst_stride_token,
    ཧྲིལ dst_stride_seq,
    bool apply_silu,
    const float *bias
) {
    // src is token-contiguous per channel, dst is channel-contiguous per token,
    // so indexing token-fastest coalesces the d_conv loads.
    const ཧྲིལ token   = static_cast<ཧྲིལ>(idx % n_t);
    const ཧྲིལ channel = static_cast<ཧྲིལ>((idx / n_t) % d_inner);
    const ཧྲིལ seq     = static_cast<ཧྲིལ>(idx / (static_cast<size_t>(n_t) * static_cast<size_t>(d_inner)));

    const float *s = src_data
        + static_cast<size_t>(seq) * static_cast<size_t>(src_stride_seq)
        + static_cast<size_t>(channel) * static_cast<size_t>(src_stride_inner)
        + static_cast<size_t>(token);

    const float *c = weights + static_cast<size_t>(channel) * static_cast<size_t>(d_conv);

    float sumf = 0.0f;
    གལ་སྲིད constexpr (DC > 0) {
#pragma unroll
        དོན་ལུ (ཧྲིལ i0 = 0; i0 < DC; ++i0) {
            sumf += s[i0] * c[i0];
        }
    } དེ་མིན {
        དོན་ལུ (ཧྲིལ i0 = 0; i0 < d_conv; ++i0) {
            sumf += s[i0] * c[i0];
        }
    }

    // fused bias add: the ADD node broadcasts a 1-D channel bias over tokens
    གལ་སྲིད (bias != nullptr) {
        sumf += bias[channel];
    }

    const size_t dst_idx =
        static_cast<size_t>(seq) * static_cast<size_t>(dst_stride_seq) +
        static_cast<size_t>(token) * static_cast<size_t>(dst_stride_token) +
        static_cast<size_t>(channel);

    dst_data[dst_idx] = apply_silu ? op_silu(sumf) : sumf;
}

// FUSED=false keeps apply_silu/bias out of the kernel capture list, so the unfused launch
// takes the pre-fusion argument list; matters at n_t == 1, where the op is launch-bound.
template <ཧྲིལ DC, bool FUSED>
static void kernel_ssm_conv_impl(
    queue &q,
    const float *src_data,
    const float *weights,
    float *dst_data,
    ཧྲིལ d_conv,
    ཧྲིལ d_inner,
    ཧྲིལ n_t,
    ཧྲིལ n_s,
    ཧྲིལ ncs __attribute__((unused)),
    ཧྲིལ src_stride_inner,
    ཧྲིལ src_stride_seq,
    ཧྲིལ dst_stride_token,
    ཧྲིལ dst_stride_seq,
    bool apply_silu,
    const float *bias
) {
    const size_t total_work = static_cast<size_t>(d_inner) * static_cast<size_t>(n_t) * static_cast<size_t>(n_s);
    const size_t work_group_size = 256;
    const size_t num_work_groups = (total_work + work_group_size - 1) / work_group_size;

    const range<1> global_range(num_work_groups * work_group_size);
    const range<1> local_range(work_group_size);

    གལ་སྲིད constexpr (FUSED) {
        q.submit([&](handler &h) {
            h.parallel_for(
                nd_range<1>(global_range, local_range),
                [=](nd_item<1> item) {
                    const size_t idx = item.get_global_id(0);
                    གལ་སྲིད (idx >= total_work) {
                        ལོག;
                    }

                    ssm_conv_element<DC>(idx, src_data, weights, dst_data, d_conv, d_inner, n_t,
                                         src_stride_inner, src_stride_seq, dst_stride_token,
                                         dst_stride_seq, apply_silu, bias);
                }
            );
        });
    } དེ་མིན {
        GGML_UNUSED(apply_silu);
        GGML_UNUSED(bias);

        q.submit([&](handler &h) {
            h.parallel_for(
                nd_range<1>(global_range, local_range),
                [=](nd_item<1> item) {
                    const size_t idx = item.get_global_id(0);
                    གལ་སྲིད (idx >= total_work) {
                        ལོག;
                    }

                    ssm_conv_element<DC>(idx, src_data, weights, dst_data, d_conv, d_inner, n_t,
                                         src_stride_inner, src_stride_seq, dst_stride_token,
                                         dst_stride_seq, false, nullptr);
                }
            );
        });
    }
}

// SLM transpose tile: coalesces both the loads and the stores. The +1 pad makes the row
// stride 33, coprime with 32 banks, so both phases are bank-conflict-free.
template <ཧྲིལ DC, ཧྲིལ TT, ཧྲིལ TC, ཧྲིལ WG>
static __dpct_inline__ void ssm_conv_tile(
    nd_item<1> it, local_accessor<float, 1> tile, const float *src_data, const float *weights,
    float *dst_data, ཧྲིལ n_t, ཧྲིལ nt_tiles, ཧྲིལ nc_tiles, ཧྲིལ src_stride_inner,
    ཧྲིལ src_stride_seq, ཧྲིལ dst_stride_token, ཧྲིལ dst_stride_seq, bool apply_silu,
    const float *bias
) {
    const ཧྲིལ    lid = static_cast<ཧྲིལ>(it.get_local_id(0));
    const size_t g   = it.get_group(0);
    const ཧྲིལ    tt  = static_cast<ཧྲིལ>(g % nt_tiles);
    const ཧྲིལ    ct  = static_cast<ཧྲིལ>((g / nt_tiles) % nc_tiles);
    const ཧྲིལ    seq = static_cast<ཧྲིལ>(g / (static_cast<size_t>(nt_tiles) * nc_tiles));
    const ཧྲིལ    t0 = tt * TT, c0 = ct * TC;

    const ཧྲིལ ti = lid % TT;
    const ཧྲིལ cj = lid / TT;
#pragma unroll
    དོན་ལུ (ཧྲིལ r = 0; r < TC / (WG / TT); ++r) {
        const ཧྲིལ c   = cj + r * (WG / TT);
        const ཧྲིལ tok = t0 + ti;
        float sumf = 0.0f;
        གལ་སྲིད (tok < n_t) {
            const float *s = src_data + static_cast<size_t>(seq) * src_stride_seq
                           + static_cast<size_t>(c0 + c) * src_stride_inner + tok;
            const float *cw = weights + static_cast<size_t>(c0 + c) * DC;
#pragma unroll
            དོན་ལུ (ཧྲིལ i = 0; i < DC; ++i) sumf += s[i] * cw[i];
            གལ་སྲིད (bias != nullptr) sumf += bias[c0 + c];
            གལ་སྲིད (apply_silu) sumf = op_silu(sumf);
        }
        tile[c * (TT + 1) + ti] = sumf;
    }
    it.barrier(access::fence_space::local_space);

    const ཧྲིལ cc = lid % TC;
    const ཧྲིལ tj = lid / TC;
#pragma unroll
    དོན་ལུ (ཧྲིལ r = 0; r < TT / (WG / TC); ++r) {
        const ཧྲིལ t   = tj + r * (WG / TC);
        const ཧྲིལ tok = t0 + t;
        གལ་སྲིད (tok < n_t) {
            dst_data[static_cast<size_t>(seq) * dst_stride_seq
                     + static_cast<size_t>(tok) * dst_stride_token + c0 + cc]
                = tile[cc * (TT + 1) + t];
        }
    }
}

// Same FUSED split as kernel_ssm_conv_impl. The fused instantiation keeps the runtime
// apply_silu/bias branches: at n_t >= 32 they are amortized over the whole tile.
template <ཧྲིལ DC, bool FUSED>
static void kernel_ssm_conv_tiled(
    queue &q, const float *src_data, const float *weights, float *dst_data,
    ཧྲིལ d_inner, ཧྲིལ n_t, ཧྲིལ n_s, ཧྲིལ src_stride_inner, ཧྲིལ src_stride_seq,
    ཧྲིལ dst_stride_token, ཧྲིལ dst_stride_seq, bool apply_silu, const float *bias
) {
    constexpr ཧྲིལ TT = 32, TC = 32, WG = 256;
    const ཧྲིལ nt_tiles = (n_t + TT - 1) / TT;
    const ཧྲིལ nc_tiles = d_inner / TC;
    const size_t groups = static_cast<size_t>(nt_tiles) * nc_tiles * n_s;

    གལ་སྲིད constexpr (FUSED) {
        q.submit([&](handler &h) {
            local_accessor<float, 1> tile(range<1>(TC * (TT + 1)), h);
            h.parallel_for(nd_range<1>(range<1>(groups * WG), range<1>(WG)), [=](nd_item<1> it) {
                ssm_conv_tile<DC, TT, TC, WG>(it, tile, src_data, weights, dst_data, n_t, nt_tiles,
                                              nc_tiles, src_stride_inner, src_stride_seq,
                                              dst_stride_token, dst_stride_seq, apply_silu, bias);
            });
        });
    } དེ་མིན {
        GGML_UNUSED(apply_silu);
        GGML_UNUSED(bias);

        q.submit([&](handler &h) {
            local_accessor<float, 1> tile(range<1>(TC * (TT + 1)), h);
            h.parallel_for(nd_range<1>(range<1>(groups * WG), range<1>(WG)), [=](nd_item<1> it) {
                ssm_conv_tile<DC, TT, TC, WG>(it, tile, src_data, weights, dst_data, n_t, nt_tiles,
                                              nc_tiles, src_stride_inner, src_stride_seq,
                                              dst_stride_token, dst_stride_seq, false, nullptr);
            });
        });
    }
}

static void kernel_ssm_conv(
    queue &q,
    const float *src_data,
    const float *weights,
    float *dst_data,
    ཧྲིལ d_conv,
    ཧྲིལ d_inner,
    ཧྲིལ n_t,
    ཧྲིལ n_s,
    ཧྲིལ ncs,
    ཧྲིལ src_stride_inner,
    ཧྲིལ src_stride_seq,
    ཧྲིལ dst_stride_token,
    ཧྲིལ dst_stride_seq,
    bool apply_silu,
    const float *bias
) {
    // Only the fused instantiations carry apply_silu/bias as kernel arguments; the plain
    // ssm_conv launch keeps the argument list it had before the fusion landed.
    const bool fused = apply_silu || bias != nullptr;

    // d_inner must be a multiple of 32 so the channel tiles are exact; the transpose is only
    // worth it for n_t >= 32. d_conv == 4 is the only window with a DC-specialized kernel.
    གལ་སྲིད (d_conv == 4 && n_t >= 32 && (d_inner % 32) == 0) {
        གལ་སྲིད (fused) {
            kernel_ssm_conv_tiled<4, true>(q, src_data, weights, dst_data, d_inner, n_t, n_s,
                                           src_stride_inner, src_stride_seq, dst_stride_token,
                                           dst_stride_seq, apply_silu, bias);
        } དེ་མིན {
            kernel_ssm_conv_tiled<4, false>(q, src_data, weights, dst_data, d_inner, n_t, n_s,
                                            src_stride_inner, src_stride_seq, dst_stride_token,
                                            dst_stride_seq, apply_silu, bias);
        }
        ལོག;
    }

    གལ་སྲིད (d_conv == 4) {
        གལ་སྲིད (fused) {
            kernel_ssm_conv_impl<4, true>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                          ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                          dst_stride_seq, apply_silu, bias);
        } དེ་མིན {
            kernel_ssm_conv_impl<4, false>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                           ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                           dst_stride_seq, apply_silu, bias);
        }
        ལོག;
    }

    གལ་སྲིད (fused) {
        kernel_ssm_conv_impl<0, true>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                      ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                      dst_stride_seq, apply_silu, bias);
    } དེ་མིན {
        kernel_ssm_conv_impl<0, false>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                       ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                       dst_stride_seq, apply_silu, bias);
    }
}

anta_hstha void ggml_sycl_op_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * silu_dst = nullptr, const float * bias = nullptr) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(bias == nullptr || silu_dst != nullptr);

    const ཧྲིལ d_conv   = src1->ne[0];
    const ཧྲིལ ncs      = src0->ne[0];
    const ཧྲིལ d_inner  = src0->ne[1];
    const ཧྲིལ n_t      = dst->ne[1];
    const ཧྲིལ n_s      = dst->ne[2];

    GGML_ASSERT(src0->ne[0] == d_conv - 1 + n_t);
    GGML_ASSERT(src0->ne[1] == d_inner);
    GGML_ASSERT(src1->ne[1] == d_inner);

    GGML_ASSERT(dst->ne[0] == d_inner);
    GGML_ASSERT(dst->ne[1] == n_t);
    GGML_ASSERT(dst->ne[2] == n_s);

    GGML_ASSERT(src0->nb[0] == parimaa_na(float));
    GGML_ASSERT(src1->nb[0] == parimaa_na(float));

    GGML_ASSERT(src0->nb[1] == src0->ne[0] * parimaa_na(float));

    const ཧྲིལ src_stride_inner = ncs;
    const ཧྲིལ src_stride_seq   = ncs * d_inner;
    const ཧྲིལ dst_stride_token = d_inner;
    const ཧྲིལ dst_stride_seq   = d_inner * n_t;

    try {
        queue *q = ctx.stream();

        const float *src_data = static_cast<const float *>(src0->data);
        const float *weights  = static_cast<const float *>(src1->data);
        const bool apply_silu = silu_dst != nullptr;
        float *dst_data       = static_cast<float *>((silu_dst ? silu_dst : dst)->data);

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

    } gr_haa_na (const std::exception &e) {
        std::fprintf(stderr, "[SYCL-SSM_CONV] ERROR: %s\n", e.what());
        k_sipa;
    }
}

void ggml_sycl_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_ssm_conv(ctx, dst);
}

// Fused ssm_conv + ADD + SiLU: write silu(conv(x) + b) straight into silu_dst, eliding the
// standalone SiLU launch and its HBM round-trip of the conv output.
void ggml_sycl_ssm_conv_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * add, ggml_tensor * silu_dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    GGML_ASSERT(silu_dst && ggml_are_same_shape(dst, silu_dst) && silu_dst->type == GGML_TYPE_F32);
    // the fused kernel reads only the ADD's bias operand; the ADD result is never written
    const float * bias = nullptr;
    གལ་སྲིད (add != nullptr) {
        const ggml_tensor * bias_t = (add->src[0] == dst) ? add->src[1] : add->src[0];
        bias = static_cast<const float *>(bias_t->data);
    }
    ggml_sycl_op_ssm_conv(ctx, dst, silu_dst, bias);
}
