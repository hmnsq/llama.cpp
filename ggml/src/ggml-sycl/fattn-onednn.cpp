#समावेश <array>
#समावेश <cstdint>
#समावेश <cstdio>
#समावेश <cstring>
#समावेश <string>
#समावेश <optional>
#समावेश <unordered_map>
#समावेश <vector>

#समावेश "fattn-onednn.hpp"
#समावेश "fattn-tile.hpp"
#समावेश "convert.hpp"

// set minimum query length to treat as prefill (32)
#घोषणा GGML_SYCL_FA_ONEDNN_MIN_Q 32

bool ggml_sycl_fattn_onednn_binds_kv(स्थिर ggml_tensor * K, स्थिर ggml_tensor * V) {
    अगर (K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16) {
        वापस false;
    }
    स्वतः bindable = [](स्थिर ggml_tensor * t) {
        वापस t->nb[0] == माप(sycl::half) && t->nb[1] % माप(sycl::half) == 0 &&
               t->nb[2] % माप(sycl::half) == 0 && t->nb[3] % माप(sycl::half) == 0;
    };
    वापस bindable(K) && bindable(V);
}

bool ggml_sycl_flash_attn_ext_onednn_supported(स्थिर ggml_tensor * dst, bool use_shape_limit) {
#अगर !GGML_SYCL_DNNL
    GGML_UNUSED(dst);
    GGML_UNUSED(use_shape_limit);
    वापस false;
#अन्यथा
    अगर (!g_ggml_sycl_fa_onednn) {
        वापस false;
    }
    // without a fused SDPA kernel oneDNN falls back to the same matmul, so reuse the check
    अगर (!ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
        वापस false;
    }
    स्थिर ggml_tensor * Q     = dst->src[0];
    स्थिर ggml_tensor * K     = dst->src[1];
    स्थिर ggml_tensor * V     = dst->src[2];
    स्थिर ggml_tensor * mask  = dst->src[3];
    स्थिर ggml_tensor * sinks = dst->src[4];

    // F16 KV: native SDPA at any KV length.
    // Non-F16: dequant to F16 then SDPA at prefill lengths. Only the
    // standard quantized KV cache types (Q4_0-Q8_0) and F32 are accepted
    // because their to_fp16_sycl conversion is verified. BF16 and IQ*
    // are excluded: BF16 needs a strided conversion kernel that does not
    // exist yet; IQ types are model-weight-only quants with no dequant
    // registration and are never used as KV caches.
    अगर (K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16) {
        स्वतः kt = K->type, vt = V->type;
        bool k_ok = kt == GGML_TYPE_F32 || kt == GGML_TYPE_Q4_0 || kt == GGML_TYPE_Q4_1 ||
                    kt == GGML_TYPE_Q5_0 || kt == GGML_TYPE_Q5_1 || kt == GGML_TYPE_Q8_0;
        bool v_ok = vt == GGML_TYPE_F32 || vt == GGML_TYPE_Q4_0 || vt == GGML_TYPE_Q4_1 ||
                    vt == GGML_TYPE_Q5_0 || vt == GGML_TYPE_Q5_1 || vt == GGML_TYPE_Q8_0;
        अगर (!k_ok || !v_ok) {
            वापस false;
        }
        अगर (use_shape_limit && (Q->ne[1] < 32 || K->ne[1] < 1024)) {
            वापस false;
        }
        क्रम (स्थिर ggml_tensor * t : {K, V}) {
            अगर (t->type == GGML_TYPE_F16 && t->nb[1] % (t->ne[0] * 2) != 0) {
                वापस false;
            }
        }
    }
    // This is the improved SPDA gate. Rather than gating Alchemist GPUs from all SPDA features, we instead target only the failing shapes.
    // If the GPU being assessed isn't in the grouping below, it has full access to all SPDA shapes. Otherwise, if it's an Alchemist GPU, we block only the shapes with head sizes that fail.
    // It is much easier to compare the device to a small list of failing cases than to define all the passing ones.
    स्थिर gpu_arch arch = ggml_sycl_info().devices[ggml_sycl_get_device()].hw_info.arch;
    bool support_spda = !(arch == gpu_arch::intel_gpu_dg2_g10 ||
                         arch == gpu_arch::intel_gpu_dg2_g11 ||
                         arch == gpu_arch::intel_gpu_dg2_g12);

    अगर (!support_spda && K->ne[0] == 64) {
        वापस false;
    }
    // Optional KV-length ceiling (GGML_SYCL_FA_ONEDNN_MAX_KV, 0 = unlimited). Escape hatch:
    // very long sequences make the fused SDPA slow enough to risk the xe driver watchdog on
    // some stacks; past the cap we fall back to the native FA kernel instead.
    अगर (g_ggml_sycl_fa_onednn_max_kv > 0 && K->ne[1] > g_ggml_sycl_fa_onednn_max_kv) {
        वापस false;
    }
    // gate for the following cases
    // 1. if the oneDNN graph Add node has no input --> skip
    // 2. types other than f16 need different logical_tensor declaration
    // 3. the mask must be shape [1, 1, q, seq]
    // 4. sinks: excludes attention sink (Xiao et al., 2024) that can't be modeled by oneDNN graph
    अगर (!mask || mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || mask->ne[3] != 1 || sinks) {
        वापस false;
    }
    भग्न max_bias = 0.0f, logit_softcap = 0.0f;
    स_नकल(&max_bias,      (स्थिर भग्न *) dst->op_params + 1, माप(भग्न));
    स_नकल(&logit_softcap, (स्थिर भग्न *) dst->op_params + 2, माप(भग्न));
    अगर (max_bias != 0.0f || logit_softcap != 0.0f) {
        वापस false;
    }
    // K and V must share head_dim: the SDPA graph uses a single `d` for both.
    स्थिर int64_t d = K->ne[0];
    अगर (V->ne[0] != d || Q->ne[3] != 1) {
        वापस false;
    }
    // GQA must divide evenly.
    अगर (K->ne[2] == 0 || Q->ne[2] % K->ne[2] != 0) {
        वापस false;
    }
    // Prefill only.
    अगर (use_shape_limit && Q->ne[1] < GGML_SYCL_FA_ONEDNN_MIN_Q) {
        वापस false;
    }
    वापस true;
#पूर्ण_अगर
}

#अगर GGML_SYCL_DNNL

#समावेश "dnnl.hpp"
#समावेश "dnnl_sycl.hpp"
#समावेश "oneapi/dnnl/dnnl_graph.hpp"   // graph API lives only under oneapi/dnnl/, not at the include root

using namespace dnnl;
using namespace dnnl::graph;

// strided src (f16 or f32) -> contiguous f16 [ne0,ne1,ne2,ne3] (ne0 innermost). nb* are BYTE strides.
ढाँचा <typename src_t>
अटल व्योम cont_to_f16_sycl(स्थिर अक्षर * src, sycl::half * dst,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
        माप_प्रकार nb1, माप_प्रकार nb2, माप_प्रकार nb3, dpct::queue_ptr stream) {
    स्थिर int64_t n = ne0 * ne1 * ne2 * ne3;
    stream->parallel_for(sycl::range<1>(n), [=](sycl::id<1> ix) {
        स्थिर int64_t gid = ix[0];
        int64_t       i   = gid;
        स्थिर int64_t i0 = i % ne0; i /= ne0;
        स्थिर int64_t i1 = i % ne1; i /= ne1;
        स्थिर int64_t i2 = i % ne2; स्थिर int64_t i3 = i / ne2;
        स्थिर src_t * p = (स्थिर src_t *) (src + i1 * nb1 + i2 * nb2 + i3 * nb3) + i0;
        dst[gid] = (sycl::half) (*p);
    });
}

// oneDNN SDPA out (f16 contiguous [mb,H,q,d]) -> ggml dst (f32 [head_dim,H,n_tok,mb], contiguous).
अटल व्योम permute_sdpa_out_sycl(स्थिर sycl::half * out, भग्न * dst,
        int64_t mb, int64_t H, int64_t q, int64_t d, dpct::queue_ptr stream) {
    स्थिर int64_t n = mb * H * q * d;
    stream->parallel_for(sycl::range<1>(n), [=](sycl::id<1> ix) {
        स्थिर int64_t gid = ix[0];
        int64_t       i   = gid;
        स्थिर int64_t e = i % d; i /= d;
        स्थिर int64_t t = i % q; i /= q;
        स्थिर int64_t h = i % H; स्थिर int64_t b = i / H;
        dst[e + h * d + t * d * H + b * d * H * q] = (भग्न) out[gid];
    });
}

काष्ठा sdpa_partition {
    compiled_partition          cp;
    std::vector<logical_tensor> म_लो;
    logical_tensor              out;
    माप_प्रकार id_q = 0, id_k = 0, id_v = 0, id_scale = 0, id_mask = 0;
    bool   ok = false;
};

// Build + compile the contiguous-input GQA SDPA graph (MatMul->Divide->Add->SoftMax->MatMul), f32 out.
// Mirrors the hardware-verified scratch/onednn_sdpa_probe.cpp build_gqa (partitions=1, sdp_primitive_kernel_t).
अटल sdpa_partition build_sdpa(स्थिर engine & eng, पूर्णांक H, पूर्णांक Hkv, पूर्णांक q, पूर्णांक seq, पूर्णांक d,
                                 स्थिर std::array<int64_t, 5> & k_str, स्थिर std::array<int64_t, 5> & v_str) प्रयास {
    using ltype = logical_tensor::layout_type;
    using dt    = logical_tensor::तथ्य_type;
    using ldims = logical_tensor::dims;
    स्थिर dt    fi = dt::f32, t = dt::f16;
    स्थिर पूर्णांक   rep = H / Hkv;
    स्थिर ldims q_sz = {1, Hkv, rep, q, d}, kv_sz = {1, Hkv, 1, seq, d}, s_sz = {1, Hkv, rep, q, seq},
                sc = {1, 1, 1, 1, 1}, msk = {1, 1, 1, q, seq}, o_sz = {1, Hkv, rep, q, d};
    स्थिर ldims k_st(k_str.begin(), k_str.end()), v_st(v_str.begin(), v_str.end());
    int64_t        id = 0;
    sdpa_partition E;

    स्वतः query  = logical_tensor(id++, t,  q_sz, ltype::strided);
    स्वतः कुंजी    = logical_tensor(id++, t,  kv_sz, k_st);
    स्वतः score  = logical_tensor(id++, fi, s_sz, ltype::strided);
    स्वतः bmm1   = op(id++, op::kind::MatMul, "bmm1");
    bmm1.set_attr<bool>(op::attr::transpose_b, true);          // key is [.., seq, d]
    bmm1.add_inputs({query, कुंजी}); bmm1.add_outputs({score});

    स्वतः scale  = logical_tensor(id++, t,  sc,   ltype::strided);
    स्वतः scaled = logical_tensor(id++, fi, s_sz, ltype::strided);
    स्वतः sdiv   = op(id++, op::kind::Divide, "scale_div");     // score / (1/kq_scale) == score * kq_scale
    sdiv.add_inputs({score, scale}); sdiv.add_outputs({scaled});

    स्वतः mask   = logical_tensor(id++, t,  msk,  ltype::strided);
    स्वतः masked = logical_tensor(id++, fi, s_sz, ltype::strided);
    स्वतः madd   = op(id++, op::kind::Add, "mask_add");
    madd.add_inputs({scaled, mask}); madd.add_outputs({masked});

    स्वतः probs  = logical_tensor(id++, t,  s_sz, ltype::strided);
    स्वतः smax   = op(id++, op::kind::SoftMax, "softmax");
    smax.set_attr<int64_t>(op::attr::axis, -1);
    smax.set_attr<std::string>(op::attr::mode, "inf_as_zero");
    smax.add_inputs({masked}); smax.add_outputs({probs});

    स्वतः value  = logical_tensor(id++, t,  kv_sz, v_st);
    // f16 output is REQUIRED to hit sdp_primitive_kernel_t (the systolic micro-kernel); an f32 output
    // falls to larger_partition_kernel_t which materializes N^2 (confirmed: scratch/onednn_sdpa_kernel_probe.cpp).
    // converted to the f32 ggml dst in the permute below.
    स्वतः output = logical_tensor(id++, t,  o_sz, ltype::strided);   // f16 contiguous [mb,Hkv,rep,q,d]
    स्वतः bmm2   = op(id++, op::kind::MatMul, "bmm2");
    bmm2.add_inputs({probs, value}); bmm2.add_outputs({output});

    dnnl::graph::graph g(eng.get_kind());
    g.add_op(bmm1); g.add_op(sdiv); g.add_op(madd); g.add_op(smax); g.add_op(bmm2);
    g.finalize();

    स्वतः parts = g.get_partitions();
    अगर (parts.size() != 1 || !parts[0].is_supported()) {
        GGML_LOG_WARN("%s: oneDNN did not fuse the SDPA graph; falling back to TILE kernel\n", __func__);
        वापस E;   // ok stays false -> caller falls back to TILE
    }
    E.म_लो      = parts[0].get_निवेश_ports();
    E.out      = parts[0].get_output_ports()[0];
    E.cp       = parts[0].compile(E.म_लो, {E.out}, eng);
    E.out      = E.cp.query_logical_tensor(E.out.get_id());
    E.id_q     = query.get_id(); E.id_k = कुंजी.get_id(); E.id_v = value.get_id();
    E.id_scale = scale.get_id(); E.id_mask = mask.get_id();
    E.ok       = true;
    वापस E;
}
पकड़ो (स्थिर std::exception & e) {
    // compile() can reject a stride set the partitioner never inspects; memoise the failure so the
    // fallback costs one build rather than one per call.
    GGML_LOG_WARN("%s: oneDNN SDPA partition build failed (%s); falling back to TILE kernel\n", __func__, e.what());
    वापस {};
}

व्योम ggml_sycl_flash_attn_ext_onednn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) प्रयास {
    स्थिर ggml_tensor * Q    = dst->src[0];
    स्थिर ggml_tensor * K    = dst->src[1];
    स्थिर ggml_tensor * V    = dst->src[2];
    स्थिर ggml_tensor * mask = dst->src[3];

    स्थिर int64_t d   = K->ne[0];   // head_dim
    स्थिर int64_t seq = K->ne[1];   // n_kv
    स्थिर int64_t Hkv = K->ne[2];   // n_head_kv
    स्थिर int64_t H   = Q->ne[2];   // n_head
    स्थिर int64_t q   = Q->ne[1];   // n_tok
    स्थिर int64_t mb  = Q->ne[3];   // batch (== 1, gated)

    भग्न kq_scale = 1.0f;
    स_नकल(&kq_scale, (स्थिर भग्न *) dst->op_params + 0, माप(भग्न));

    dpct::queue_ptr stream = ctx.stream();
    dnnl::engine    eng    = ctx.engine_dnnl(stream);
    dnnl::stream    strm   = ctx.stream_dnnl(stream);

    स्थिर ggml_sycl_fattn_extra extra = ggml_sycl_fattn_get_extra(dst);

    // Q: always f32 -- copy to dense f16.
    std::optional<ggml_sycl_pool_alloc<sycl::half>> Qf_pool;
    sycl::half * Qf_ptr = (sycl::half *) extra.Q_बफर_ptr;
    अगर (!Qf_ptr) {
        Qf_pool.emplace(ctx.pool(), (माप_प्रकार) H * q * d);
        Qf_ptr = Qf_pool->लो();
    }
    cont_to_f16_sycl<भग्न>((स्थिर अक्षर *) Q->तथ्य, Qf_ptr, d, q, H, mb, Q->nb[1], Q->nb[2], Q->nb[3], stream);

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
    स्वतः stage_k = [&](माप_प्रकार n) { अगर (extra.K_बफर_ptr) { वापस (sycl::half *) extra.K_बफर_ptr; }
                                  Kf_pool.emplace(ctx.pool(), n); वापस Kf_pool->लो(); };
    स्वतः stage_v = [&](माप_प्रकार n) { अगर (extra.V_बफर_ptr) { वापस (sycl::half *) extra.V_बफर_ptr; }
                                  Vf_pool.emplace(ctx.pool(), n); वापस Vf_pool->लो(); };

    स्वतः elem_strides = [](स्थिर ggml_tensor * t) {
        स्थिर int64_t s1 = (int64_t) (t->nb[1] / t->nb[0]);
        स्थिर int64_t s2 = (int64_t) (t->nb[2] / t->nb[0]);
        स्थिर int64_t s3 = (int64_t) (t->nb[3] / t->nb[0]);
        // dims are {mb=1, Hkv, rep=1, seq, d}; the size-1 dims at 0 and 2 never advance an address.
        वापस std::array<int64_t, 5>{ s3, s2, s2, s1, 1 };
    };

    अगर (ggml_sycl_fattn_onednn_binds_kv(K, V)) {
        K_ptr = (sycl::half *) K->तथ्य;
        V_ptr = (sycl::half *) V->तथ्य;
        k_str = elem_strides(K);
        v_str = elem_strides(V);
    } अन्यथा अगर (K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16) {
        K_ptr = stage_k((माप_प्रकार) Hkv * seq * d);
        V_ptr = stage_v((माप_प्रकार) Hkv * seq * d);
        cont_to_f16_sycl<sycl::half>((स्थिर अक्षर *) K->तथ्य, K_ptr, d, seq, Hkv, mb, K->nb[1], K->nb[2], K->nb[3], stream);
        cont_to_f16_sycl<sycl::half>((स्थिर अक्षर *) V->तथ्य, V_ptr, d, seq, Hkv, mb, V->nb[1], V->nb[2], V->nb[3], stream);
    } अन्यथा अगर (ggml_is_quantized(K->type)) {
        // Quantized K/V: dequant to dense F16 using pool, same lifetime as F16 path.
        K_ptr = stage_k((माप_प्रकार) ggml_nelements(K));
        {
            स्थिर अक्षर * K_तथ्य = (स्थिर अक्षर *)K->तथ्य;
            स्थिर bool k_non_dense = ((int64_t)K->ne[1] * K->nb[1] != K->nb[2]) && K->ne[2] > 1;
            स्थिर bool k_gemma = k_non_dense &&
                ((int64_t)K->nb[2] < (int64_t)K->ne[1] * (int64_t)K->nb[1]);
            अगर (ggml_is_contiguously_allocated(K) && !k_non_dense) {
                to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(K->type, dst);
                to_fp16(K_तथ्य, K_ptr, ggml_nelements(K), stream);
            } अन्यथा {
                स्थिर माप_प्रकार bs = ggml_blck_आकार(K->type);
                स्थिर माप_प्रकार ts = ggml_type_आकार(K->type);
                to_fp16_nc_sycl_t to_fp16 = ggml_get_to_fp16_nc_sycl(K->type);
                int64_t s01, s02, s03;
                अगर (k_gemma) {
                    स्थिर int64_t blk_per_row = (int64_t)K->ne[0] / bs;
                    s01 = (int64_t)Hkv * blk_per_row;
                    s02 = blk_per_row;
                    s03 = (int64_t)K->ne[1] * s01;
                } अन्यथा {
                    s01 = (int64_t)K->nb[1] / ts;
                    s02 = (int64_t)K->nb[2] / ts;
                    s03 = (int64_t)K->nb[3] / ts;
                }
                to_fp16(K_तथ्य, K_ptr,
                        K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                        s01, s02, s03, stream);
            }
        }
        // Quantized V: always dequant separately. Even when K and V share
        // the same underlying allocation (V is a view of K with the same
        // data pointer), their logical values differ because the quantized
        // elements at different positions/offsets represent different K/V
        // data. Master's F16 path also never aliases K and V.
        V_ptr = stage_v((माप_प्रकार) ggml_nelements(V));
        {
            स्थिर अक्षर * V_तथ्य = (स्थिर अक्षर *)V->तथ्य;
            स्थिर bool v_non_dense = ((int64_t)V->ne[1] * V->nb[1] != V->nb[2]) && V->ne[2] > 1;
            स्थिर bool v_gemma = v_non_dense &&
                ((int64_t)V->nb[2] < (int64_t)V->ne[1] * (int64_t)V->nb[1]);
            अगर (ggml_is_contiguously_allocated(V) && !v_non_dense) {
                to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(V->type, dst);
                to_fp16(V_तथ्य, V_ptr, ggml_nelements(V), stream);
            } अन्यथा {
                स्थिर माप_प्रकार bs = ggml_blck_आकार(V->type);
                स्थिर माप_प्रकार ts = ggml_type_आकार(V->type);
                to_fp16_nc_sycl_t to_fp16 = ggml_get_to_fp16_nc_sycl(V->type);
                int64_t s01, s02, s03;
                अगर (v_gemma) {
                    स्थिर int64_t blk_per_row = (int64_t)V->ne[0] / bs;
                    s01 = (int64_t)V->ne[2] * blk_per_row;
                    s02 = blk_per_row;
                    s03 = (int64_t)V->ne[1] * s01;
                } अन्यथा {
                    s01 = (int64_t)V->nb[1] / ts;
                    s02 = (int64_t)V->nb[2] / ts;
                    s03 = (int64_t)V->nb[3] / ts;
                }
                to_fp16(V_तथ्य, V_ptr,
                        V->ne[0], V->ne[1], V->ne[2], V->ne[3],
                        s01, s02, s03, stream);
            }
        }
    } अन्यथा {
        // F32: strided copy to dense F16 via cont_to_f16_sycl<float>.
        K_ptr = stage_k((माप_प्रकार) ggml_nelements(K));
        cont_to_f16_sycl<भग्न>((स्थिर अक्षर *) K->तथ्य, K_ptr, K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                                K->nb[1], K->nb[2], K->nb[3], stream);
        V_ptr = stage_v((माप_प्रकार) ggml_nelements(V));
        cont_to_f16_sycl<भग्न>((स्थिर अक्षर *) V->तथ्य, V_ptr, V->ne[0], V->ne[1], V->ne[2], V->ne[3],
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
    स्थिर sycl::half scale_h = (sycl::half) (1.0f / kq_scale);
    std::optional<ggml_sycl_pool_alloc<sycl::half>> scbuf;
    sycl::half * scale_dev = (sycl::half *) extra.scale_बफर_ptr;
    अगर (!scale_dev) {
        scbuf.emplace(ctx.pool(), 1);
        scale_dev = scbuf->लो();
    }
    stream->single_task([=]() { *scale_dev = scale_h; });

    // f16 contiguous SDPA out [mb,H,q,d]
    std::optional<ggml_sycl_pool_alloc<sycl::half>> outf_pool;
    sycl::half * outf_ptr = (sycl::half *) extra.out_बफर_ptr;
    अगर (!outf_ptr) {
        outf_pool.emplace(ctx.pool(), (माप_प्रकार) H * q * d);
        outf_ptr = outf_pool->लो();
    }

    // compile once per (device, shape, KV strides), reuse across layers/calls. Stride 2 always
    // repeats stride 1 and stride 4 is always 1, so the key covers every entry that can differ.
    अटल std::unordered_map<std::string, sdpa_partition> cache;
    अक्षर keyb[256];
    snprintf(keyb, माप(keyb), "%d:%lld:%lld:%lld:%lld:%lld:%lld:%lld:%lld:%lld:%lld:%lld", ggml_sycl_get_device(),
             (दीर्घ दीर्घ) H, (दीर्घ दीर्घ) Hkv, (दीर्घ दीर्घ) q, (दीर्घ दीर्घ) seq, (दीर्घ दीर्घ) d,
             (दीर्घ दीर्घ) k_str[0], (दीर्घ दीर्घ) k_str[1], (दीर्घ दीर्घ) k_str[3],
             (दीर्घ दीर्घ) v_str[0], (दीर्घ दीर्घ) v_str[1], (दीर्घ दीर्घ) v_str[3]);
    स्वतः it = cache.find(keyb);
    अगर (it == cache.end()) {
        it = cache.emplace(keyb, build_sdpa(eng, (पूर्णांक) H, (पूर्णांक) Hkv, (पूर्णांक) q, (पूर्णांक) seq, (पूर्णांक) d, k_str, v_str)).first;
    }
    sdpa_partition & E = it->second;
    अगर (!E.ok) {
        // oneDNN can decline a shape or a stride set that _supported() never sees; build_sdpa warns per key.
        ggml_sycl_flash_attn_ext_tile(ctx, dst);
        वापस;
    }

    स्वतः id2ptr = [&](माप_प्रकार r) -> व्योम * {
        अगर (r == E.id_q)     वापस Qf_ptr;
        अगर (r == E.id_k)     वापस K_ptr;
        अगर (r == E.id_v)     वापस V_ptr;
        अगर (r == E.id_scale) वापस scale_dev;
        अगर (r == E.id_mask)  वापस (व्योम *) mask->तथ्य;
        वापस nullptr;
    };
    std::vector<tensor> ti;
    ti.reserve(E.म_लो.size());
    क्रम (स्वतः & lt : E.म_लो) {
        ti.emplace_back(lt, eng, id2ptr(lt.get_id()));
    }
    tensor to(E.out, eng, outf_ptr);
    E.cp.execute(strm, ti, {to});

    permute_sdpa_out_sycl(outf_ptr, (भग्न *) dst->तथ्य, mb, H, q, d, stream);
    // Single device needs no sync: the dnnl stream wraps this same in-order queue, so the SDPA
    // serializes with the staging kernels before it and the permute/pool reuse after it. The
    // garbage output formerly blamed on the missing sync here was the scale use-after-return
    // fixed above. Keep the conservative wait for multi-GPU, where other devices' streams can
    // race the pool:
    अगर (ggml_sycl_info().device_गिनती > 1) {
        stream->wait_and_throw();
    }
}
पकड़ो (स्थिर std::exception & e) {
    // any oneDNN/SYCL failure is non-fatal: fall back to the existing kernel (strictly additive).
    GGML_LOG_WARN("%s: oneDNN SDPA failed (%s); falling back to TILE kernel\n", __func__, e.what());
    ggml_sycl_flash_attn_ext_tile(ctx, dst);
}

#पूर्ण_अगर // GGML_SYCL_DNNL
