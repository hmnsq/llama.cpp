#include "ssm_conv.hpp"
#include "common.hpp"
#include "element_wise.hpp"

#include <cstdio>

using namespace sycl;

// One output element of the conv. DC is d_conv as a compile-time constant (0 keeps the
// runtime loop); unfused callers pass literal false/nullptr so the epilogue folds away.
template <ကိန်းပြည့် DC>
static __dpct_inline__ void ssm_conv_element(
    size_t idx,
    const float *src_data,
    const float *weights,
    float *dst_data,
    ကိန်းပြည့် d_conv,
    ကိန်းပြည့် d_inner,
    ကိန်းပြည့် n_t,
    ကိန်းပြည့် src_stride_inner,
    ကိန်းပြည့် src_stride_seq,
    ကိန်းပြည့် dst_stride_token,
    ကိန်းပြည့် dst_stride_seq,
    bool apply_silu,
    const float *bias
) {
    // src is token-contiguous per channel, dst is channel-contiguous per token,
    // so indexing token-fastest coalesces the d_conv loads.
    const ကိန်းပြည့် token   = static_cast<ကိန်းပြည့်>(idx % n_t);
    const ကိန်းပြည့် channel = static_cast<ကိန်းပြည့်>((idx / n_t) % d_inner);
    const ကိန်းပြည့် seq     = static_cast<ကိန်းပြည့်>(idx / (static_cast<size_t>(n_t) * static_cast<size_t>(d_inner)));

    const float *s = src_data
        + static_cast<size_t>(seq) * static_cast<size_t>(src_stride_seq)
        + static_cast<size_t>(channel) * static_cast<size_t>(src_stride_inner)
        + static_cast<size_t>(token);

    const float *c = weights + static_cast<size_t>(channel) * static_cast<size_t>(d_conv);

    float sumf = 0.0f;
    အကယ်၍ constexpr (DC > 0) {
#pragma unroll
        အတွက် (ကိန်းပြည့် i0 = 0; i0 < DC; ++i0) {
            sumf += s[i0] * c[i0];
        }
    } မဟုတ်ပါက {
        အတွက် (ကိန်းပြည့် i0 = 0; i0 < d_conv; ++i0) {
            sumf += s[i0] * c[i0];
        }
    }

    // fused bias add: the ADD node broadcasts a 1-D channel bias over tokens
    အကယ်၍ (bias != nullptr) {
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
template <ကိန်းပြည့် DC, bool FUSED>
static void kernel_ssm_conv_impl(
    queue &q,
    const float *src_data,
    const float *weights,
    float *dst_data,
    ကိန်းပြည့် d_conv,
    ကိန်းပြည့် d_inner,
    ကိန်းပြည့် n_t,
    ကိန်းပြည့် n_s,
    ကိန်းပြည့် ncs __attribute__((unused)),
    ကိန်းပြည့် src_stride_inner,
    ကိန်းပြည့် src_stride_seq,
    ကိန်းပြည့် dst_stride_token,
    ကိန်းပြည့် dst_stride_seq,
    bool apply_silu,
    const float *bias
) {
    const size_t total_work = static_cast<size_t>(d_inner) * static_cast<size_t>(n_t) * static_cast<size_t>(n_s);
    const size_t work_group_size = 256;
    const size_t num_work_groups = (total_work + work_group_size - 1) / work_group_size;

    const range<1> global_range(num_work_groups * work_group_size);
    const range<1> local_range(work_group_size);

    အကယ်၍ constexpr (FUSED) {
        q.submit([&](handler &h) {
            h.parallel_for(
                nd_range<1>(global_range, local_range),
                [=](nd_item<1> item) {
                    const size_t idx = item.get_global_id(0);
                    အကယ်၍ (idx >= total_work) {
                        ပြန်ပေး;
                    }

                    ssm_conv_element<DC>(idx, src_data, weights, dst_data, d_conv, d_inner, n_t,
                                         src_stride_inner, src_stride_seq, dst_stride_token,
                                         dst_stride_seq, apply_silu, bias);
                }
            );
        });
    } မဟုတ်ပါက {
        GGML_UNUSED(apply_silu);
        GGML_UNUSED(bias);

        q.submit([&](handler &h) {
            h.parallel_for(
                nd_range<1>(global_range, local_range),
                [=](nd_item<1> item) {
                    const size_t idx = item.get_global_id(0);
                    အကယ်၍ (idx >= total_work) {
                        ပြန်ပေး;
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
template <ကိန်းပြည့် DC, ကိန်းပြည့် TT, ကိန်းပြည့် TC, ကိန်းပြည့် WG>
static __dpct_inline__ void ssm_conv_tile(
    nd_item<1> it, local_accessor<float, 1> tile, const float *src_data, const float *weights,
    float *dst_data, ကိန်းပြည့် n_t, ကိန်းပြည့် nt_tiles, ကိန်းပြည့် nc_tiles, ကိန်းပြည့် src_stride_inner,
    ကိန်းပြည့် src_stride_seq, ကိန်းပြည့် dst_stride_token, ကိန်းပြည့် dst_stride_seq, bool apply_silu,
    const float *bias
) {
    const ကိန်းပြည့်    lid = static_cast<ကိန်းပြည့်>(it.get_local_id(0));
    const size_t g   = it.get_group(0);
    const ကိန်းပြည့်    tt  = static_cast<ကိန်းပြည့်>(g % nt_tiles);
    const ကိန်းပြည့်    ct  = static_cast<ကိန်းပြည့်>((g / nt_tiles) % nc_tiles);
    const ကိန်းပြည့်    seq = static_cast<ကိန်းပြည့်>(g / (static_cast<size_t>(nt_tiles) * nc_tiles));
    const ကိန်းပြည့်    t0 = tt * TT, c0 = ct * TC;

    const ကိန်းပြည့် ti = lid % TT;
    const ကိန်းပြည့် cj = lid / TT;
#pragma unroll
    အတွက် (ကိန်းပြည့် r = 0; r < TC / (WG / TT); ++r) {
        const ကိန်းပြည့် c   = cj + r * (WG / TT);
        const ကိန်းပြည့် tok = t0 + ti;
        float sumf = 0.0f;
        အကယ်၍ (tok < n_t) {
            const float *s = src_data + static_cast<size_t>(seq) * src_stride_seq
                           + static_cast<size_t>(c0 + c) * src_stride_inner + tok;
            const float *cw = weights + static_cast<size_t>(c0 + c) * DC;
#pragma unroll
            အတွက် (ကိန်းပြည့် i = 0; i < DC; ++i) sumf += s[i] * cw[i];
            အကယ်၍ (bias != nullptr) sumf += bias[c0 + c];
            အကယ်၍ (apply_silu) sumf = op_silu(sumf);
        }
        tile[c * (TT + 1) + ti] = sumf;
    }
    it.barrier(access::fence_space::local_space);

    const ကိန်းပြည့် cc = lid % TC;
    const ကိန်းပြည့် tj = lid / TC;
#pragma unroll
    အတွက် (ကိန်းပြည့် r = 0; r < TT / (WG / TC); ++r) {
        const ကိန်းပြည့် t   = tj + r * (WG / TC);
        const ကိန်းပြည့် tok = t0 + t;
        အကယ်၍ (tok < n_t) {
            dst_data[static_cast<size_t>(seq) * dst_stride_seq
                     + static_cast<size_t>(tok) * dst_stride_token + c0 + cc]
                = tile[cc * (TT + 1) + t];
        }
    }
}

// Same FUSED split as kernel_ssm_conv_impl. The fused instantiation keeps the runtime
// apply_silu/bias branches: at n_t >= 32 they are amortized over the whole tile.
template <ကိန်းပြည့် DC, bool FUSED>
static void kernel_ssm_conv_tiled(
    queue &q, const float *src_data, const float *weights, float *dst_data,
    ကိန်းပြည့် d_inner, ကိန်းပြည့် n_t, ကိန်းပြည့် n_s, ကိန်းပြည့် src_stride_inner, ကိန်းပြည့် src_stride_seq,
    ကိန်းပြည့် dst_stride_token, ကိန်းပြည့် dst_stride_seq, bool apply_silu, const float *bias
) {
    constexpr ကိန်းပြည့် TT = 32, TC = 32, WG = 256;
    const ကိန်းပြည့် nt_tiles = (n_t + TT - 1) / TT;
    const ကိန်းပြည့် nc_tiles = d_inner / TC;
    const size_t groups = static_cast<size_t>(nt_tiles) * nc_tiles * n_s;

    အကယ်၍ constexpr (FUSED) {
        q.submit([&](handler &h) {
            local_accessor<float, 1> tile(range<1>(TC * (TT + 1)), h);
            h.parallel_for(nd_range<1>(range<1>(groups * WG), range<1>(WG)), [=](nd_item<1> it) {
                ssm_conv_tile<DC, TT, TC, WG>(it, tile, src_data, weights, dst_data, n_t, nt_tiles,
                                              nc_tiles, src_stride_inner, src_stride_seq,
                                              dst_stride_token, dst_stride_seq, apply_silu, bias);
            });
        });
    } မဟုတ်ပါက {
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
    ကိန်းပြည့် d_conv,
    ကိန်းပြည့် d_inner,
    ကိန်းပြည့် n_t,
    ကိန်းပြည့် n_s,
    ကိန်းပြည့် ncs,
    ကိန်းပြည့် src_stride_inner,
    ကိန်းပြည့် src_stride_seq,
    ကိန်းပြည့် dst_stride_token,
    ကိန်းပြည့် dst_stride_seq,
    bool apply_silu,
    const float *bias
) {
    // Only the fused instantiations carry apply_silu/bias as kernel arguments; the plain
    // ssm_conv launch keeps the argument list it had before the fusion landed.
    const bool fused = apply_silu || bias != nullptr;

    // d_inner must be a multiple of 32 so the channel tiles are exact; the transpose is only
    // worth it for n_t >= 32. d_conv == 4 is the only window with a DC-specialized kernel.
    အကယ်၍ (d_conv == 4 && n_t >= 32 && (d_inner % 32) == 0) {
        အကယ်၍ (fused) {
            kernel_ssm_conv_tiled<4, true>(q, src_data, weights, dst_data, d_inner, n_t, n_s,
                                           src_stride_inner, src_stride_seq, dst_stride_token,
                                           dst_stride_seq, apply_silu, bias);
        } မဟုတ်ပါက {
            kernel_ssm_conv_tiled<4, false>(q, src_data, weights, dst_data, d_inner, n_t, n_s,
                                            src_stride_inner, src_stride_seq, dst_stride_token,
                                            dst_stride_seq, apply_silu, bias);
        }
        ပြန်ပေး;
    }

    အကယ်၍ (d_conv == 4) {
        အကယ်၍ (fused) {
            kernel_ssm_conv_impl<4, true>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                          ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                          dst_stride_seq, apply_silu, bias);
        } မဟုတ်ပါက {
            kernel_ssm_conv_impl<4, false>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                           ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                           dst_stride_seq, apply_silu, bias);
        }
        ပြန်ပေး;
    }

    အကယ်၍ (fused) {
        kernel_ssm_conv_impl<0, true>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                      ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                      dst_stride_seq, apply_silu, bias);
    } မဟုတ်ပါက {
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

    const ကိန်းပြည့် d_conv   = src1->ne[0];
    const ကိန်းပြည့် ncs      = src0->ne[0];
    const ကိန်းပြည့် d_inner  = src0->ne[1];
    const ကိန်းပြည့် n_t      = dst->ne[1];
    const ကိန်းပြည့် n_s      = dst->ne[2];

    GGML_ASSERT(src0->ne[0] == d_conv - 1 + n_t);
    GGML_ASSERT(src0->ne[1] == d_inner);
    GGML_ASSERT(src1->ne[1] == d_inner);

    GGML_ASSERT(dst->ne[0] == d_inner);
    GGML_ASSERT(dst->ne[1] == n_t);
    GGML_ASSERT(dst->ne[2] == n_s);

    GGML_ASSERT(src0->nb[0] == parimaa_na(float));
    GGML_ASSERT(src1->nb[0] == parimaa_na(float));

    GGML_ASSERT(src0->nb[1] == src0->ne[0] * parimaa_na(float));

    const ကိန်းပြည့် src_stride_inner = ncs;
    const ကိန်းပြည့် src_stride_seq   = ncs * d_inner;
    const ကိန်းပြည့် dst_stride_token = d_inner;
    const ကိန်းပြည့် dst_stride_seq   = d_inner * n_t;

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
    အကယ်၍ (add != nullptr) {
        const ggml_tensor * bias_t = (add->src[0] == dst) ? add->src[1] : add->src[0];
        bias = static_cast<const float *>(bias_t->data);
    }
    ggml_sycl_op_ssm_conv(ctx, dst, silu_dst, bias);
}
