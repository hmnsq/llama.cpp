#അന്തര്ഭാവ <array>
#അന്തര്ഭാവ <cstdint>
#അന്തര്ഭാവ <cstdio>
#അന്തര്ഭാവ <cstring>
#അന്തര്ഭാവ <string>
#അന്തര്ഭാവ <optional>
#അന്തര്ഭാവ <unordered_map>
#അന്തര്ഭാവ <vector>

#അന്തര്ഭാവ "fattn-onednn.hpp"
#അന്തര്ഭാവ "fattn-tile.hpp"
#അന്തര്ഭാവ "convert.hpp"

// set minimum query length to treat as prefill (32)
#define GGML_SYCL_FA_ONEDNN_MIN_Q 32

തർക്കം ggml_sycl_fattn_onednn_binds_kv(സ്ഥിരം ggml_tensor * K, സ്ഥിരം ggml_tensor * V) {
    എങ്കിൽ (K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16) {
        തിരികെ അസത്യം;
    }
    auto bindable = [](സ്ഥിരം ggml_tensor * t) {
        തിരികെ t->nb[0] == വലുപ്പം(sycl::half) && t->nb[1] % വലുപ്പം(sycl::half) == 0 &&
               t->nb[2] % വലുപ്പം(sycl::half) == 0 && t->nb[3] % വലുപ്പം(sycl::half) == 0;
    };
    തിരികെ bindable(K) && bindable(V);
}

തർക്കം ggml_sycl_flash_attn_ext_onednn_supported(സ്ഥിരം ggml_tensor * dst, തർക്കം use_shape_limit) {
#എങ്കിൽ !GGML_SYCL_DNNL
    GGML_UNUSED(dst);
    GGML_UNUSED(use_shape_limit);
    തിരികെ അസത്യം;
#അല്ലെങ്കിൽ
    എങ്കിൽ (!g_ggml_sycl_fa_onednn) {
        തിരികെ അസത്യം;
    }
    // without a fused SDPA kernel oneDNN falls back to the same matmul, so reuse the check
    എങ്കിൽ (!ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
        തിരികെ അസത്യം;
    }
    സ്ഥിരം ggml_tensor * Q     = dst->src[0];
    സ്ഥിരം ggml_tensor * K     = dst->src[1];
    സ്ഥിരം ggml_tensor * V     = dst->src[2];
    സ്ഥിരം ggml_tensor * mask  = dst->src[3];
    സ്ഥിരം ggml_tensor * sinks = dst->src[4];

    // F16 KV: native SDPA at any KV length.
    // Non-F16: dequant to F16 then SDPA at prefill lengths. Only the
    // standard quantized KV cache types (Q4_0-Q8_0) and F32 are accepted
    // because their to_fp16_sycl conversion is verified. BF16 and IQ*
    // are excluded: BF16 needs a strided conversion kernel that does not
    // exist yet; IQ types are model-weight-only quants with no dequant
    // registration and are never used as KV caches.
    എങ്കിൽ (K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16) {
        auto kt = K->type, vt = V->type;
        തർക്കം k_ok = kt == GGML_TYPE_F32 || kt == GGML_TYPE_Q4_0 || kt == GGML_TYPE_Q4_1 ||
                    kt == GGML_TYPE_Q5_0 || kt == GGML_TYPE_Q5_1 || kt == GGML_TYPE_Q8_0;
        തർക്കം v_ok = vt == GGML_TYPE_F32 || vt == GGML_TYPE_Q4_0 || vt == GGML_TYPE_Q4_1 ||
                    vt == GGML_TYPE_Q5_0 || vt == GGML_TYPE_Q5_1 || vt == GGML_TYPE_Q8_0;
        എങ്കിൽ (!k_ok || !v_ok) {
            തിരികെ അസത്യം;
        }
        എങ്കിൽ (use_shape_limit && (Q->ne[1] < 32 || K->ne[1] < 1024)) {
            തിരികെ അസത്യം;
        }
        വേണ്ടി (സ്ഥിരം ggml_tensor * t : {K, V}) {
            എങ്കിൽ (t->type == GGML_TYPE_F16 && t->nb[1] % (t->ne[0] * 2) != 0) {
                തിരികെ അസത്യം;
            }
        }
    }
    // This is the improved SPDA gate. Rather than gating Alchemist GPUs from all SPDA features, we instead target only the failing shapes.
    // If the GPU being assessed isn't in the grouping below, it has full access to all SPDA shapes. Otherwise, if it's an Alchemist GPU, we block only the shapes with head sizes that fail.
    // It is much easier to compare the device to a small list of failing cases than to define all the passing ones.
    സ്ഥിരം gpu_arch arch = ggml_sycl_info().devices[ggml_sycl_get_device()].hw_info.arch;
    തർക്കം support_spda = !(arch == gpu_arch::intel_gpu_dg2_g10 ||
                         arch == gpu_arch::intel_gpu_dg2_g11 ||
                         arch == gpu_arch::intel_gpu_dg2_g12);

    എങ്കിൽ (!support_spda && K->ne[0] == 64) {
        തിരികെ അസത്യം;
    }
    // Optional KV-length ceiling (GGML_SYCL_FA_ONEDNN_MAX_KV, 0 = unlimited). Escape hatch:
    // very long sequences make the fused SDPA slow enough to risk the xe driver watchdog on
    // some stacks; past the cap we fall back to the native FA kernel instead.
    എങ്കിൽ (g_ggml_sycl_fa_onednn_max_kv > 0 && K->ne[1] > g_ggml_sycl_fa_onednn_max_kv) {
        തിരികെ അസത്യം;
    }
    // gate for the following cases
    // 1. if the oneDNN graph Add node has no input --> skip
    // 2. types other than f16 need different logical_tensor declaration
    // 3. the mask must be shape [1, 1, q, seq]
    // 4. sinks: excludes attention sink (Xiao et al., 2024) that can't be modeled by oneDNN graph
    എങ്കിൽ (!mask || mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || mask->ne[3] != 1 || sinks) {
        തിരികെ അസത്യം;
    }
    ദശാംശം max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&max_bias,      (സ്ഥിരം ദശാംശം *) dst->op_params + 1, വലുപ്പം(ദശാംശം));
    memcpy(&logit_softcap, (സ്ഥിരം ദശാംശം *) dst->op_params + 2, വലുപ്പം(ദശാംശം));
    എങ്കിൽ (max_bias != 0.0f || logit_softcap != 0.0f) {
        തിരികെ അസത്യം;
    }
    // K and V must share head_dim: the SDPA graph uses a single `d` for both.
    സ്ഥിരം int64_t d = K->ne[0];
    എങ്കിൽ (V->ne[0] != d || Q->ne[3] != 1) {
        തിരികെ അസത്യം;
    }
    // GQA must divide evenly.
    എങ്കിൽ (K->ne[2] == 0 || Q->ne[2] % K->ne[2] != 0) {
        തിരികെ അസത്യം;
    }
    // Prefill only.
    എങ്കിൽ (use_shape_limit && Q->ne[1] < GGML_SYCL_FA_ONEDNN_MIN_Q) {
        തിരികെ അസത്യം;
    }
    തിരികെ സത്യം;
#endif
}

#എങ്കിൽ GGML_SYCL_DNNL

#അന്തര്ഭാവ "dnnl.hpp"
#അന്തര്ഭാവ "dnnl_sycl.hpp"
#അന്തര്ഭാവ "oneapi/dnnl/dnnl_graph.hpp"   // graph API lives only under oneapi/dnnl/, not at the include root

using നാമാകാശ dnnl;
using നാമാകാശ dnnl::graph;

// strided src (f16 or f32) -> contiguous f16 [ne0,ne1,ne2,ne3] (ne0 innermost). nb* are BYTE strides.
template <typename src_t>
സ്ഥിതം ശൂന്യം cont_to_f16_sycl(സ്ഥിരം അക്ഷരം * src, sycl::half * dst,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
        size_t nb1, size_t nb2, size_t nb3, dpct::queue_ptr stream) {
    സ്ഥിരം int64_t n = ne0 * ne1 * ne2 * ne3;
    stream->parallel_for(sycl::range<1>(n), [=](sycl::id<1> ix) {
        സ്ഥിരം int64_t gid = ix[0];
        int64_t       i   = gid;
        സ്ഥിരം int64_t i0 = i % ne0; i /= ne0;
        സ്ഥിരം int64_t i1 = i % ne1; i /= ne1;
        സ്ഥിരം int64_t i2 = i % ne2; സ്ഥിരം int64_t i3 = i / ne2;
        സ്ഥിരം src_t * p = (സ്ഥിരം src_t *) (src + i1 * nb1 + i2 * nb2 + i3 * nb3) + i0;
        dst[gid] = (sycl::half) (*p);
    });
}

// oneDNN SDPA out (f16 contiguous [mb,H,q,d]) -> ggml dst (f32 [head_dim,H,n_tok,mb], contiguous).
സ്ഥിതം ശൂന്യം permute_sdpa_out_sycl(സ്ഥിരം sycl::half * out, ദശാംശം * dst,
        int64_t mb, int64_t H, int64_t q, int64_t d, dpct::queue_ptr stream) {
    സ്ഥിരം int64_t n = mb * H * q * d;
    stream->parallel_for(sycl::range<1>(n), [=](sycl::id<1> ix) {
        സ്ഥിരം int64_t gid = ix[0];
        int64_t       i   = gid;
        സ്ഥിരം int64_t e = i % d; i /= d;
        സ്ഥിരം int64_t t = i % q; i /= q;
        സ്ഥിരം int64_t h = i % H; സ്ഥിരം int64_t b = i / H;
        dst[e + h * d + t * d * H + b * d * H * q] = (ദശാംശം) out[gid];
    });
}

ഘടന sdpa_partition {
    compiled_partition          cp;
    std::vector<logical_tensor> ins;
    logical_tensor              out;
    size_t id_q = 0, id_k = 0, id_v = 0, id_scale = 0, id_mask = 0;
    തർക്കം   ok = അസത്യം;
};

// Build + compile the contiguous-input GQA SDPA graph (MatMul->Divide->Add->SoftMax->MatMul), f32 out.
// Mirrors the hardware-verified scratch/onednn_sdpa_probe.cpp build_gqa (partitions=1, sdp_primitive_kernel_t).
സ്ഥിതം sdpa_partition build_sdpa(സ്ഥിരം engine & eng, പൂർണ്ണസംഖ്യ H, പൂർണ്ണസംഖ്യ Hkv, പൂർണ്ണസംഖ്യ q, പൂർണ്ണസംഖ്യ seq, പൂർണ്ണസംഖ്യ d,
                                 സ്ഥിരം std::array<int64_t, 5> & k_str, സ്ഥിരം std::array<int64_t, 5> & v_str) ശ്രമിക്കുക {
    using ltype = logical_tensor::layout_type;
    using dt    = logical_tensor::data_type;
    using ldims = logical_tensor::dims;
    സ്ഥിരം dt    fi = dt::f32, t = dt::f16;
    സ്ഥിരം പൂർണ്ണസംഖ്യ   rep = H / Hkv;
    സ്ഥിരം ldims q_sz = {1, Hkv, rep, q, d}, kv_sz = {1, Hkv, 1, seq, d}, s_sz = {1, Hkv, rep, q, seq},
                sc = {1, 1, 1, 1, 1}, msk = {1, 1, 1, q, seq}, o_sz = {1, Hkv, rep, q, d};
    സ്ഥിരം ldims k_st(k_str.begin(), k_str.end()), v_st(v_str.begin(), v_str.end());
    int64_t        id = 0;
    sdpa_partition E;

    auto query  = logical_tensor(id++, t,  q_sz, ltype::strided);
    auto താക്കോൽ    = logical_tensor(id++, t,  kv_sz, k_st);
    auto score  = logical_tensor(id++, fi, s_sz, ltype::strided);
    auto bmm1   = op(id++, op::kind::MatMul, "bmm1");
    bmm1.set_attr<തർക്കം>(op::attr::transpose_b, സത്യം);          // key is [.., seq, d]
    bmm1.add_inputs({query, താക്കോൽ}); bmm1.add_outputs({score});

    auto scale  = logical_tensor(id++, t,  sc,   ltype::strided);
    auto scaled = logical_tensor(id++, fi, s_sz, ltype::strided);
    auto sdiv   = op(id++, op::kind::Divide, "scale_div");     // score / (1/kq_scale) == score * kq_scale
    sdiv.add_inputs({score, scale}); sdiv.add_outputs({scaled});

    auto mask   = logical_tensor(id++, t,  msk,  ltype::strided);
    auto masked = logical_tensor(id++, fi, s_sz, ltype::strided);
    auto madd   = op(id++, op::kind::Add, "mask_add");
    madd.add_inputs({scaled, mask}); madd.add_outputs({masked});

    auto probs  = logical_tensor(id++, t,  s_sz, ltype::strided);
    auto smax   = op(id++, op::kind::SoftMax, "softmax");
    smax.set_attr<int64_t>(op::attr::axis, -1);
    smax.set_attr<std::string>(op::attr::mode, "inf_as_zero");
    smax.add_inputs({masked}); smax.add_outputs({probs});

    auto value  = logical_tensor(id++, t,  kv_sz, v_st);
    // f16 output is REQUIRED to hit sdp_primitive_kernel_t (the systolic micro-kernel); an f32 output
    // falls to larger_partition_kernel_t which materializes N^2 (confirmed: scratch/onednn_sdpa_kernel_probe.cpp).
    // converted to the f32 ggml dst in the permute below.
    auto output = logical_tensor(id++, t,  o_sz, ltype::strided);   // f16 contiguous [mb,Hkv,rep,q,d]
    auto bmm2   = op(id++, op::kind::MatMul, "bmm2");
    bmm2.add_inputs({probs, value}); bmm2.add_outputs({output});

    dnnl::graph::graph g(eng.get_kind());
    g.add_op(bmm1); g.add_op(sdiv); g.add_op(madd); g.add_op(smax); g.add_op(bmm2);
    g.finalize();

    auto parts = g.get_partitions();
    എങ്കിൽ (parts.size() != 1 || !parts[0].is_supported()) {
        GGML_LOG_WARN("%s: oneDNN did not fuse the SDPA graph; falling back to TILE kernel\n", __func__);
        തിരികെ E;   // ok stays false -> caller falls back to TILE
    }
    E.ins      = parts[0].get_നിവേശം_ports();
    E.out      = parts[0].get_output_ports()[0];
    E.cp       = parts[0].compile(E.ins, {E.out}, eng);
    E.out      = E.cp.query_logical_tensor(E.out.get_id());
    E.id_q     = query.get_id(); E.id_k = താക്കോൽ.get_id(); E.id_v = value.get_id();
    E.id_scale = scale.get_id(); E.id_mask = mask.get_id();
    E.ok       = സത്യം;
    തിരികെ E;
}
പിടിക്കുക (സ്ഥിരം std::exception & e) {
    // compile() can reject a stride set the partitioner never inspects; memoise the failure so the
    // fallback costs one build rather than one per call.
    GGML_LOG_WARN("%s: oneDNN SDPA partition build failed (%s); falling back to TILE kernel\n", __func__, e.what());
    തിരികെ {};
}

ശൂന്യം ggml_sycl_flash_attn_ext_onednn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) ശ്രമിക്കുക {
    സ്ഥിരം ggml_tensor * Q    = dst->src[0];
    സ്ഥിരം ggml_tensor * K    = dst->src[1];
    സ്ഥിരം ggml_tensor * V    = dst->src[2];
    സ്ഥിരം ggml_tensor * mask = dst->src[3];

    സ്ഥിരം int64_t d   = K->ne[0];   // head_dim
    സ്ഥിരം int64_t seq = K->ne[1];   // n_kv
    സ്ഥിരം int64_t Hkv = K->ne[2];   // n_head_kv
    സ്ഥിരം int64_t H   = Q->ne[2];   // n_head
    സ്ഥിരം int64_t q   = Q->ne[1];   // n_tok
    സ്ഥിരം int64_t mb  = Q->ne[3];   // batch (== 1, gated)

    ദശാംശം kq_scale = 1.0f;
    memcpy(&kq_scale, (സ്ഥിരം ദശാംശം *) dst->op_params + 0, വലുപ്പം(ദശാംശം));

    dpct::queue_ptr stream = ctx.stream();
    dnnl::engine    eng    = ctx.engine_dnnl(stream);
    dnnl::stream    strm   = ctx.stream_dnnl(stream);

    സ്ഥിരം ggml_sycl_fattn_extra extra = ggml_sycl_fattn_get_extra(dst);

    // Q: always f32 -- copy to dense f16.
    std::optional<ggml_sycl_pool_alloc<sycl::half>> Qf_pool;
    sycl::half * Qf_ptr = (sycl::half *) extra.Q_buffer_ptr;
    എങ്കിൽ (!Qf_ptr) {
        Qf_pool.emplace(ctx.pool(), (size_t) H * q * d);
        Qf_ptr = Qf_pool->get();
    }
    cont_to_f16_sycl<ദശാംശം>((സ്ഥിരം അക്ഷരം *) Q->data, Qf_ptr, d, q, H, mb, Q->nb[1], Q->nb[2], Q->nb[3], stream);

    // K/V: bind the f16 cache in place. llama.cpp permutes it to [token][head][dim], so its head
    // plane is strided rather than dense, which is what an explicit stride vector expresses.
    // Quantized and f32 KV still stage a dense copy -- the layout the k_str/v_str defaults describe.
    sycl::half * K_ptr = nullptr;
    sycl::half * V_ptr = nullptr;
    std::array<int64_t, 5> k_str{ Hkv * seq * d, seq * d, seq * d, d, 1 };
    std::array<int64_t, 5> v_str = k_str;
    std::optional<ggml_sycl_pool_alloc<sycl::half>> Kf_pool;
    std::optional<ggml_sycl_pool_alloc<sycl::half>> Vf_pool;
    // Helper: hand out reserved space, or fall back to the pool.
    auto stage_k = [&](size_t n) { എങ്കിൽ (extra.K_buffer_ptr) { തിരികെ (sycl::half *) extra.K_buffer_ptr; }
                                  Kf_pool.emplace(ctx.pool(), n); തിരികെ Kf_pool->get(); };
    auto stage_v = [&](size_t n) { എങ്കിൽ (extra.V_buffer_ptr) { തിരികെ (sycl::half *) extra.V_buffer_ptr; }
                                  Vf_pool.emplace(ctx.pool(), n); തിരികെ Vf_pool->get(); };

    auto elem_strides = [](സ്ഥിരം ggml_tensor * t) {
        സ്ഥിരം int64_t s1 = (int64_t) (t->nb[1] / t->nb[0]);
        സ്ഥിരം int64_t s2 = (int64_t) (t->nb[2] / t->nb[0]);
        സ്ഥിരം int64_t s3 = (int64_t) (t->nb[3] / t->nb[0]);
        // dims are {mb=1, Hkv, rep=1, seq, d}; the size-1 dims at 0 and 2 never advance an address.
        തിരികെ std::array<int64_t, 5>{ s3, s2, s2, s1, 1 };
    };

    എങ്കിൽ (ggml_sycl_fattn_onednn_binds_kv(K, V)) {
        K_ptr = (sycl::half *) K->data;
        V_ptr = (sycl::half *) V->data;
        k_str = elem_strides(K);
        v_str = elem_strides(V);
    } അല്ലെങ്കിൽ എങ്കിൽ (K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16) {
        K_ptr = stage_k((size_t) Hkv * seq * d);
        V_ptr = stage_v((size_t) Hkv * seq * d);
        cont_to_f16_sycl<sycl::half>((സ്ഥിരം അക്ഷരം *) K->data, K_ptr, d, seq, Hkv, mb, K->nb[1], K->nb[2], K->nb[3], stream);
        cont_to_f16_sycl<sycl::half>((സ്ഥിരം അക്ഷരം *) V->data, V_ptr, d, seq, Hkv, mb, V->nb[1], V->nb[2], V->nb[3], stream);
    } അല്ലെങ്കിൽ എങ്കിൽ (ggml_is_quantized(K->type)) {
        // Quantized K/V: dequant to dense F16 using pool, same lifetime as F16 path.
        K_ptr = stage_k((size_t) ggml_nelements(K));
        {
            സ്ഥിരം അക്ഷരം * K_data = (സ്ഥിരം അക്ഷരം *)K->data;
            സ്ഥിരം തർക്കം k_non_dense = ((int64_t)K->ne[1] * K->nb[1] != K->nb[2]) && K->ne[2] > 1;
            സ്ഥിരം തർക്കം k_gemma = k_non_dense &&
                ((int64_t)K->nb[2] < (int64_t)K->ne[1] * (int64_t)K->nb[1]);
            എങ്കിൽ (ggml_is_contiguously_allocated(K) && !k_non_dense) {
                to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(K->type, dst);
                to_fp16(K_data, K_ptr, ggml_nelements(K), stream);
            } അല്ലെങ്കിൽ {
                സ്ഥിരം size_t bs = ggml_blck_size(K->type);
                സ്ഥിരം size_t ts = ggml_type_size(K->type);
                to_fp16_nc_sycl_t to_fp16 = ggml_get_to_fp16_nc_sycl(K->type);
                int64_t s01, s02, s03;
                എങ്കിൽ (k_gemma) {
                    സ്ഥിരം int64_t blk_per_row = (int64_t)K->ne[0] / bs;
                    s01 = (int64_t)Hkv * blk_per_row;
                    s02 = blk_per_row;
                    s03 = (int64_t)K->ne[1] * s01;
                } അല്ലെങ്കിൽ {
                    s01 = (int64_t)K->nb[1] / ts;
                    s02 = (int64_t)K->nb[2] / ts;
                    s03 = (int64_t)K->nb[3] / ts;
                }
                to_fp16(K_data, K_ptr,
                        K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                        s01, s02, s03, stream);
            }
        }
        // Quantized V: always dequant separately. Even when K and V share
        // the same underlying allocation (V is a view of K with the same
        // data pointer), their logical values differ because the quantized
        // elements at different positions/offsets represent different K/V
        // data. Master's F16 path also never aliases K and V.
        V_ptr = stage_v((size_t) ggml_nelements(V));
        {
            സ്ഥിരം അക്ഷരം * V_data = (സ്ഥിരം അക്ഷരം *)V->data;
            സ്ഥിരം തർക്കം v_non_dense = ((int64_t)V->ne[1] * V->nb[1] != V->nb[2]) && V->ne[2] > 1;
            സ്ഥിരം തർക്കം v_gemma = v_non_dense &&
                ((int64_t)V->nb[2] < (int64_t)V->ne[1] * (int64_t)V->nb[1]);
            എങ്കിൽ (ggml_is_contiguously_allocated(V) && !v_non_dense) {
                to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(V->type, dst);
                to_fp16(V_data, V_ptr, ggml_nelements(V), stream);
            } അല്ലെങ്കിൽ {
                സ്ഥിരം size_t bs = ggml_blck_size(V->type);
                സ്ഥിരം size_t ts = ggml_type_size(V->type);
                to_fp16_nc_sycl_t to_fp16 = ggml_get_to_fp16_nc_sycl(V->type);
                int64_t s01, s02, s03;
                എങ്കിൽ (v_gemma) {
                    സ്ഥിരം int64_t blk_per_row = (int64_t)V->ne[0] / bs;
                    s01 = (int64_t)V->ne[2] * blk_per_row;
                    s02 = blk_per_row;
                    s03 = (int64_t)V->ne[1] * s01;
                } അല്ലെങ്കിൽ {
                    s01 = (int64_t)V->nb[1] / ts;
                    s02 = (int64_t)V->nb[2] / ts;
                    s03 = (int64_t)V->nb[3] / ts;
                }
                to_fp16(V_data, V_ptr,
                        V->ne[0], V->ne[1], V->ne[2], V->ne[3],
                        s01, s02, s03, stream);
            }
        }
    } അല്ലെങ്കിൽ {
        // F32: strided copy to dense F16 via cont_to_f16_sycl<float>.
        K_ptr = stage_k((size_t) ggml_nelements(K));
        cont_to_f16_sycl<ദശാംശം>((സ്ഥിരം അക്ഷരം *) K->data, K_ptr, K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                                K->nb[1], K->nb[2], K->nb[3], stream);
        V_ptr = stage_v((size_t) ggml_nelements(V));
        cont_to_f16_sycl<ദശാംശം>((സ്ഥിരം അക്ഷരം *) V->data, V_ptr, V->ne[0], V->ne[1], V->ne[2], V->ne[3],
                                V->nb[1], V->nb[2], V->nb[3], stream);
    }

    // divide-by-(1/scale) reproduces ggml's score *= kq_scale on the proven probe graph.
    //
    // The scale must not be uploaded with an async memcpy from a stack local: on the in-order
    // queue that copy waits behind the K/V staging kernels, and once those take long enough
    // (n_kv >= ~26k on B70) the host frame is recycled before the copy runs, feeding the SDPA a
    // garbage scale (output collapses to a repeated token). Write the scalar from a kernel
    // instead -- the value is captured into the command, so no host memory has to outlive the
    // call, and the enqueue stays async.
    സ്ഥിരം sycl::half scale_h = (sycl::half) (1.0f / kq_scale);
    std::optional<ggml_sycl_pool_alloc<sycl::half>> scbuf;
    sycl::half * scale_dev = (sycl::half *) extra.scale_buffer_ptr;
    എങ്കിൽ (!scale_dev) {
        scbuf.emplace(ctx.pool(), 1);
        scale_dev = scbuf->get();
    }
    stream->single_task([=]() { *scale_dev = scale_h; });

    // f16 contiguous SDPA out [mb,H,q,d]
    std::optional<ggml_sycl_pool_alloc<sycl::half>> outf_pool;
    sycl::half * outf_ptr = (sycl::half *) extra.out_buffer_ptr;
    എങ്കിൽ (!outf_ptr) {
        outf_pool.emplace(ctx.pool(), (size_t) H * q * d);
        outf_ptr = outf_pool->get();
    }

    // compile once per (device, shape, KV strides), reuse across layers/calls. Stride 2 always
    // repeats stride 1 and stride 4 is always 1, so the key covers every entry that can differ.
    സ്ഥിതം std::unordered_map<std::string, sdpa_partition> cache;
    അക്ഷരം keyb[256];
    snprintf(keyb, വലുപ്പം(keyb), "%d:%lld:%lld:%lld:%lld:%lld:%lld:%lld:%lld:%lld:%lld:%lld", ggml_sycl_get_device(),
             (long long) H, (long long) Hkv, (long long) q, (long long) seq, (long long) d,
             (long long) k_str[0], (long long) k_str[1], (long long) k_str[3],
             (long long) v_str[0], (long long) v_str[1], (long long) v_str[3]);
    auto it = cache.find(keyb);
    എങ്കിൽ (it == cache.end()) {
        it = cache.emplace(keyb, build_sdpa(eng, (പൂർണ്ണസംഖ്യ) H, (പൂർണ്ണസംഖ്യ) Hkv, (പൂർണ്ണസംഖ്യ) q, (പൂർണ്ണസംഖ്യ) seq, (പൂർണ്ണസംഖ്യ) d, k_str, v_str)).first;
    }
    sdpa_partition & E = it->second;
    എങ്കിൽ (!E.ok) {
        // oneDNN can decline a shape or a stride set that _supported() never sees; build_sdpa warns per key.
        ggml_sycl_flash_attn_ext_tile(ctx, dst);
        തിരികെ;
    }

    auto id2ptr = [&](size_t r) -> ശൂന്യം * {
        എങ്കിൽ (r == E.id_q)     തിരികെ Qf_ptr;
        എങ്കിൽ (r == E.id_k)     തിരികെ K_ptr;
        എങ്കിൽ (r == E.id_v)     തിരികെ V_ptr;
        എങ്കിൽ (r == E.id_scale) തിരികെ scale_dev;
        എങ്കിൽ (r == E.id_mask)  തിരികെ (ശൂന്യം *) mask->data;
        തിരികെ nullptr;
    };
    std::vector<tensor> ti;
    ti.reserve(E.ins.size());
    വേണ്ടി (auto & lt : E.ins) {
        ti.emplace_back(lt, eng, id2ptr(lt.get_id()));
    }
    tensor to(E.out, eng, outf_ptr);
    E.cp.execute(strm, ti, {to});

    permute_sdpa_out_sycl(outf_ptr, (ദശാംശം *) dst->data, mb, H, q, d, stream);
    // Single device needs no sync: the dnnl stream wraps this same in-order queue, so the SDPA
    // serializes with the staging kernels before it and the permute/pool reuse after it. The
    // garbage output formerly blamed on the missing sync here was the scale use-after-return
    // fixed above. Keep the conservative wait for multi-GPU, where other devices' streams can
    // race the pool:
    എങ്കിൽ (ggml_sycl_info().device_എണ്ണം > 1) {
        stream->wait_and_throw();
    }
}
പിടിക്കുക (സ്ഥിരം std::exception & e) {
    // any oneDNN/SYCL failure is non-fatal: fall back to the existing kernel (strictly additive).
    GGML_LOG_WARN("%s: oneDNN SDPA failed (%s); falling back to TILE kernel\n", __func__, e.what());
    ggml_sycl_flash_attn_ext_tile(ctx, dst);
}

#endif // GGML_SYCL_DNNL
