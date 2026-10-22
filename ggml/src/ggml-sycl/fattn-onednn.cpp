#अन्तर्भाव <array>
#अन्तर्भाव <cstdint>
#अन्तर्भाव <cstdio>
#अन्तर्भाव <cstring>
#अन्तर्भाव <string>
#अन्तर्भाव <optional>
#अन्तर्भाव <unordered_map>
#अन्तर्भाव <vector>

#अन्तर्भाव "fattn-onednn.hpp"
#अन्तर्भाव "fattn-tile.hpp"
#अन्तर्भाव "convert.hpp"

// set minimum query length to treat as prefill (32)
#परिभाषय GGML_SYCL_FA_ONEDNN_MIN_Q 32

तार्किक ggml_sycl_fattn_onednn_binds_kv(ध्रुव ggml_tensor * K, ध्रुव ggml_tensor * V) {
    यदि (K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16) {
        प्रत्यागम असत्य;
    }
    स्वचालित bindable = [](ध्रुव ggml_tensor * t) {
        प्रत्यागम t->nb[0] == परिमाण(sycl::half) && t->nb[1] % परिमाण(sycl::half) == 0 &&
               t->nb[2] % परिमाण(sycl::half) == 0 && t->nb[3] % परिमाण(sycl::half) == 0;
    };
    प्रत्यागम bindable(K) && bindable(V);
}

तार्किक ggml_sycl_flash_attn_ext_onednn_supported(ध्रुव ggml_tensor * dst, तार्किक use_shape_सीमा) {
#यदि !GGML_SYCL_DNNL
    GGML_अप्रयुक्तम्(dst);
    GGML_अप्रयुक्तम्(use_shape_सीमा);
    प्रत्यागम असत्य;
#अन्यथा
    यदि (!g_ggml_sycl_fa_onednn) {
        प्रत्यागम असत्य;
    }
    // without a fused SDPA kernel oneDNN falls back to the same matmul, so reuse the check
    यदि (!ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
        प्रत्यागम असत्य;
    }
    ध्रुव ggml_tensor * Q     = dst->src[0];
    ध्रुव ggml_tensor * K     = dst->src[1];
    ध्रुव ggml_tensor * V     = dst->src[2];
    ध्रुव ggml_tensor * आवरणम्  = dst->src[3];
    ध्रुव ggml_tensor * sinks = dst->src[4];

    // F16 KV: native SDPA at any KV length.
    // Non-F16: dequant to F16 then SDPA at prefill lengths. Only the
    // standard quantized KV cache types (Q4_0-Q8_0) and F32 are accepted
    // because their to_fp16_sycl conversion is verified. BF16 and IQ*
    // are excluded: BF16 needs a strided conversion kernel that does not
    // exist yet; IQ types are model-weight-only quants with no dequant
    // registration and are never used as KV caches.
    यदि (K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16) {
        स्वचालित kt = K->type, vt = V->type;
        तार्किक k_ok = kt == GGML_TYPE_F32 || kt == GGML_TYPE_Q4_0 || kt == GGML_TYPE_Q4_1 ||
                    kt == GGML_TYPE_Q5_0 || kt == GGML_TYPE_Q5_1 || kt == GGML_TYPE_Q8_0;
        तार्किक v_ok = vt == GGML_TYPE_F32 || vt == GGML_TYPE_Q4_0 || vt == GGML_TYPE_Q4_1 ||
                    vt == GGML_TYPE_Q5_0 || vt == GGML_TYPE_Q5_1 || vt == GGML_TYPE_Q8_0;
        यदि (!k_ok || !v_ok) {
            प्रत्यागम असत्य;
        }
        यदि (use_shape_सीमा && (Q->ne[1] < 32 || K->ne[1] < 1024)) {
            प्रत्यागम असत्य;
        }
        प्रति (ध्रुव ggml_tensor * t : {K, V}) {
            यदि (t->type == GGML_TYPE_F16 && t->nb[1] % (t->ne[0] * 2) != 0) {
                प्रत्यागम असत्य;
            }
        }
    }
    // This is the improved SPDA gate. Rather than gating Alchemist GPUs from all SPDA features, we instead target only the failing shapes.
    // If the GPU being assessed isn't in the grouping below, it has full access to all SPDA shapes. Otherwise, if it's an Alchemist GPU, we block only the shapes with head sizes that fail.
    // It is much easier to compare the device to a small list of failing cases than to define all the passing ones.
    ध्रुव gpu_arch arch = ggml_sycl_info().devices[ggml_sycl_get_device()].hw_info.arch;
    तार्किक support_spda = !(arch == gpu_arch::intel_gpu_dg2_g10 ||
                         arch == gpu_arch::intel_gpu_dg2_g11 ||
                         arch == gpu_arch::intel_gpu_dg2_g12);

    यदि (!support_spda && K->ne[0] == 64) {
        प्रत्यागम असत्य;
    }
    // Optional KV-length ceiling (GGML_SYCL_FA_ONEDNN_MAX_KV, 0 = unlimited). Escape hatch:
    // very long sequences make the fused SDPA slow enough to risk the xe driver watchdog on
    // some stacks; past the cap we fall back to the native FA kernel instead.
    यदि (g_ggml_sycl_fa_onednn_max_kv > 0 && K->ne[1] > g_ggml_sycl_fa_onednn_max_kv) {
        प्रत्यागम असत्य;
    }
    // gate for the following cases
    // 1. if the oneDNN graph Add node has no input --> skip
    // 2. types other than f16 need different logical_tensor declaration
    // 3. the mask must be shape [1, 1, q, seq]
    // 4. sinks: excludes attention sink (Xiao et al., 2024) that can't be modeled by oneDNN graph
    यदि (!आवरणम् || आवरणम्->type != GGML_TYPE_F16 || आवरणम्->ne[2] != 1 || आवरणम्->ne[3] != 1 || sinks) {
        प्रत्यागम असत्य;
    }
    भिन्न max_bias = 0.0f, logit_softcap = 0.0f;
    स्मृतिं_प्रतिलिख(&max_bias,      (ध्रुव भिन्न *) dst->op_params + 1, परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&logit_softcap, (ध्रुव भिन्न *) dst->op_params + 2, परिमाण(भिन्न));
    यदि (max_bias != 0.0f || logit_softcap != 0.0f) {
        प्रत्यागम असत्य;
    }
    // K and V must share head_dim: the SDPA graph uses a single `d` for both.
    ध्रुव int64_t d = K->ne[0];
    यदि (V->ne[0] != d || Q->ne[3] != 1) {
        प्रत्यागम असत्य;
    }
    // GQA must divide evenly.
    यदि (K->ne[2] == 0 || Q->ne[2] % K->ne[2] != 0) {
        प्रत्यागम असत्य;
    }
    // Prefill only.
    यदि (use_shape_सीमा && Q->ne[1] < GGML_SYCL_FA_ONEDNN_MIN_Q) {
        प्रत्यागम असत्य;
    }
    प्रत्यागम सत्य;
#यदि_अन्तः
}

#यदि GGML_SYCL_DNNL

#अन्तर्भाव "dnnl.hpp"
#अन्तर्भाव "dnnl_sycl.hpp"
#अन्तर्भाव "oneapi/dnnl/dnnl_graph.hpp"   // graph API lives only under oneapi/dnnl/, not at the include root

उपयुज्य नामाकाश dnnl;
उपयुज्य नामाकाश dnnl::graph;

// strided src (f16 or f32) -> contiguous f16 [ne0,ne1,ne2,ne3] (ne0 innermost). nb* are BYTE strides.
प्रतिरूपम् <प्रकारनाम src_t>
स्थिर शून्य cont_गन्तव्यम्_f16_sycl(ध्रुव वर्ण * src, sycl::half * dst,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
        size_t nb1, size_t nb2, size_t nb3, dpct::queue_ptr stream) {
    ध्रुव int64_t n = ne0 * ne1 * ne2 * ne3;
    stream->parallel_for(sycl::range<1>(n), [=](sycl::id<1> ix) {
        ध्रुव int64_t gid = ix[0];
        int64_t       i   = gid;
        ध्रुव int64_t i0 = i % ne0; i /= ne0;
        ध्रुव int64_t i1 = i % ne1; i /= ne1;
        ध्रुव int64_t i2 = i % ne2; ध्रुव int64_t i3 = i / ne2;
        ध्रुव src_t * p = (ध्रुव src_t *) (src + i1 * nb1 + i2 * nb2 + i3 * nb3) + i0;
        dst[gid] = (sycl::half) (*p);
    });
}

// oneDNN SDPA out (f16 contiguous [mb,H,q,d]) -> ggml dst (f32 [head_dim,H,n_tok,mb], contiguous).
स्थिर शून्य permute_sdpa_बहिर्गमनम्_sycl(ध्रुव sycl::half * बहिर्गमनम्, भिन्न * dst,
        int64_t mb, int64_t H, int64_t q, int64_t d, dpct::queue_ptr stream) {
    ध्रुव int64_t n = mb * H * q * d;
    stream->parallel_for(sycl::range<1>(n), [=](sycl::id<1> ix) {
        ध्रुव int64_t gid = ix[0];
        int64_t       i   = gid;
        ध्रुव int64_t e = i % d; i /= d;
        ध्रुव int64_t t = i % q; i /= q;
        ध्रुव int64_t h = i % H; ध्रुव int64_t b = i / H;
        dst[e + h * d + t * d * H + b * d * H * q] = (भिन्न) बहिर्गमनम्[gid];
    });
}

संरचना sdpa_partition {
    compiled_partition          cp;
    std::vector<logical_tensor> ins;
    logical_tensor              बहिर्गमनम्;
    size_t id_q = 0, id_k = 0, id_v = 0, id_scale = 0, id_आवरणम् = 0;
    तार्किक   ok = असत्य;
};

// Build + compile the contiguous-input GQA SDPA graph (MatMul->Divide->Add->SoftMax->MatMul), f32 out.
// Mirrors the hardware-verified scratch/onednn_sdpa_probe.cpp build_gqa (partitions=1, sdp_primitive_kernel_t).
स्थिर sdpa_partition build_sdpa(ध्रुव engine & eng, पूर्णाङ्क H, पूर्णाङ्क Hkv, पूर्णाङ्क q, पूर्णाङ्क seq, पूर्णाङ्क d,
                                 ध्रुव std::array<int64_t, 5> & k_str, ध्रुव std::array<int64_t, 5> & v_str) यत्न {
    उपयुज्य ltype = logical_tensor::layout_type;
    उपयुज्य dt    = logical_tensor::दत्त_type;
    उपयुज्य ldims = logical_tensor::dims;
    ध्रुव dt    fi = dt::f32, t = dt::f16;
    ध्रुव पूर्णाङ्क   rep = H / Hkv;
    ध्रुव ldims q_sz = {1, Hkv, rep, q, d}, kv_sz = {1, Hkv, 1, seq, d}, s_sz = {1, Hkv, rep, q, seq},
                sc = {1, 1, 1, 1, 1}, msk = {1, 1, 1, q, seq}, o_sz = {1, Hkv, rep, q, d};
    ध्रुव ldims k_st(k_str.begin(), k_str.end()), v_st(v_str.begin(), v_str.end());
    int64_t        id = 0;
    sdpa_partition E;

    स्वचालित query  = logical_tensor(id++, t,  q_sz, ltype::strided);
    स्वचालित कुञ्चिका    = logical_tensor(id++, t,  kv_sz, k_st);
    स्वचालित score  = logical_tensor(id++, fi, s_sz, ltype::strided);
    स्वचालित bmm1   = op(id++, op::विधा::MatMul, "bmm1");
    bmm1.समुच्चयः_attr<तार्किक>(op::attr::transpose_b, सत्य);          // key is [.., seq, d]
    bmm1.सङ्कलय_inputs({query, कुञ्चिका}); bmm1.सङ्कलय_outputs({score});

    स्वचालित scale  = logical_tensor(id++, t,  sc,   ltype::strided);
    स्वचालित scaled = logical_tensor(id++, fi, s_sz, ltype::strided);
    स्वचालित sdiv   = op(id++, op::विधा::Divide, "scale_div");     // score / (1/kq_scale) == score * kq_scale
    sdiv.सङ्कलय_inputs({score, scale}); sdiv.सङ्कलय_outputs({scaled});

    स्वचालित आवरणम्   = logical_tensor(id++, t,  msk,  ltype::strided);
    स्वचालित masked = logical_tensor(id++, fi, s_sz, ltype::strided);
    स्वचालित madd   = op(id++, op::विधा::सङ्कलय, "mask_add");
    madd.सङ्कलय_inputs({scaled, आवरणम्}); madd.सङ्कलय_outputs({masked});

    स्वचालित probs  = logical_tensor(id++, t,  s_sz, ltype::strided);
    स्वचालित smax   = op(id++, op::विधा::SoftMax, "softmax");
    smax.समुच्चयः_attr<int64_t>(op::attr::axis, -1);
    smax.समुच्चयः_attr<std::string>(op::attr::mode, "inf_as_zero");
    smax.सङ्कलय_inputs({masked}); smax.सङ्कलय_outputs({probs});

    स्वचालित value  = logical_tensor(id++, t,  kv_sz, v_st);
    // f16 output is REQUIRED to hit sdp_primitive_kernel_t (the systolic micro-kernel); an f32 output
    // falls to larger_partition_kernel_t which materializes N^2 (confirmed: scratch/onednn_sdpa_kernel_probe.cpp).
    // converted to the f32 ggml dst in the permute below.
    स्वचालित output = logical_tensor(id++, t,  o_sz, ltype::strided);   // f16 contiguous [mb,Hkv,rep,q,d]
    स्वचालित bmm2   = op(id++, op::विधा::MatMul, "bmm2");
    bmm2.सङ्कलय_inputs({probs, value}); bmm2.सङ्कलय_outputs({output});

    dnnl::graph::graph g(eng.get_विधा());
    g.सङ्कलय_op(bmm1); g.सङ्कलय_op(sdiv); g.सङ्कलय_op(madd); g.सङ्कलय_op(smax); g.सङ्कलय_op(bmm2);
    g.finalize();

    स्वचालित parts = g.get_partitions();
    यदि (parts.size() != 1 || !parts[0].is_supported()) {
        GGML_LOG_WARN("%s: oneDNN did not fuse the SDPA graph; falling back to TILE kernel\n", __func__);
        प्रत्यागम E;   // ok stays false -> caller falls back to TILE
    }
    E.ins      = parts[0].get_निवेशः_ports();
    E.बहिर्गमनम्      = parts[0].get_output_ports()[0];
    E.cp       = parts[0].compile(E.ins, {E.बहिर्गमनम्}, eng);
    E.बहिर्गमनम्      = E.cp.query_logical_tensor(E.बहिर्गमनम्.get_id());
    E.id_q     = query.get_id(); E.id_k = कुञ्चिका.get_id(); E.id_v = value.get_id();
    E.id_scale = scale.get_id(); E.id_आवरणम् = आवरणम्.get_id();
    E.ok       = सत्य;
    प्रत्यागम E;
}
गृहाण (ध्रुव std::exception & e) {
    // compile() can reject a stride set the partitioner never inspects; memoise the failure so the
    // fallback costs one build rather than one per call.
    GGML_LOG_WARN("%s: oneDNN SDPA partition build failed (%s); falling back to TILE kernel\n", __func__, e.what());
    प्रत्यागम {};
}

शून्य ggml_sycl_flash_attn_ext_onednn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) यत्न {
    ध्रुव ggml_tensor * Q    = dst->src[0];
    ध्रुव ggml_tensor * K    = dst->src[1];
    ध्रुव ggml_tensor * V    = dst->src[2];
    ध्रुव ggml_tensor * आवरणम् = dst->src[3];

    ध्रुव int64_t d   = K->ne[0];   // head_dim
    ध्रुव int64_t seq = K->ne[1];   // n_kv
    ध्रुव int64_t Hkv = K->ne[2];   // n_head_kv
    ध्रुव int64_t H   = Q->ne[2];   // n_head
    ध्रुव int64_t q   = Q->ne[1];   // n_tok
    ध्रुव int64_t mb  = Q->ne[3];   // batch (== 1, gated)

    भिन्न kq_scale = 1.0f;
    स्मृतिं_प्रतिलिख(&kq_scale, (ध्रुव भिन्न *) dst->op_params + 0, परिमाण(भिन्न));

    dpct::queue_ptr stream = ctx.stream();
    dnnl::engine    eng    = ctx.engine_dnnl(stream);
    dnnl::stream    strm   = ctx.प्रवाहः_dnnl(stream);

    ध्रुव ggml_sycl_fattn_extra extra = ggml_sycl_fattn_get_extra(dst);

    // Q: always f32 -- copy to dense f16.
    std::optional<ggml_sycl_pool_alloc<sycl::half>> Qf_pool;
    sycl::half * Qf_ptr = (sycl::half *) extra.Q_बफर_ptr;
    यदि (!Qf_ptr) {
        Qf_pool.emplace(ctx.pool(), (size_t) H * q * d);
        Qf_ptr = Qf_pool->get();
    }
    cont_गन्तव्यम्_f16_sycl<भिन्न>((ध्रुव वर्ण *) Q->दत्त, Qf_ptr, d, q, H, mb, Q->nb[1], Q->nb[2], Q->nb[3], stream);

    // K/V: bind the f16 cache in place. llama.cpp permutes it to [token][head][dim], so its head
    // plane is strided rather than dense, which is what an explicit stride vector expresses.
    // Quantized and f32 KV still stage a dense copy -- the layout the k_str/v_str defaults describe.
    sycl::half * K_ptr = शून्यसूचकः;
    sycl::half * V_ptr = शून्यसूचकः;
    std::array<int64_t, 5> k_str{ Hkv * seq * d, seq * d, seq * d, d, 1 };
    std::array<int64_t, 5> v_str = k_str;
    std::optional<ggml_sycl_pool_alloc<sycl::half>> Kf_pool;
    std::optional<ggml_sycl_pool_alloc<sycl::half>> Vf_pool;
    // Helper: hand out reserved space, or fall back to the pool.
    स्वचालित stage_k = [&](size_t n) { यदि (extra.K_बफर_ptr) { प्रत्यागम (sycl::half *) extra.K_बफर_ptr; }
                                  Kf_pool.emplace(ctx.pool(), n); प्रत्यागम Kf_pool->get(); };
    स्वचालित stage_v = [&](size_t n) { यदि (extra.V_बफर_ptr) { प्रत्यागम (sycl::half *) extra.V_बफर_ptr; }
                                  Vf_pool.emplace(ctx.pool(), n); प्रत्यागम Vf_pool->get(); };

    स्वचालित elem_strides = [](ध्रुव ggml_tensor * t) {
        ध्रुव int64_t s1 = (int64_t) (t->nb[1] / t->nb[0]);
        ध्रुव int64_t s2 = (int64_t) (t->nb[2] / t->nb[0]);
        ध्रुव int64_t s3 = (int64_t) (t->nb[3] / t->nb[0]);
        // dims are {mb=1, Hkv, rep=1, seq, d}; the size-1 dims at 0 and 2 never advance an address.
        प्रत्यागम std::array<int64_t, 5>{ s3, s2, s2, s1, 1 };
    };

    यदि (ggml_sycl_fattn_onednn_binds_kv(K, V)) {
        K_ptr = (sycl::half *) K->दत्त;
        V_ptr = (sycl::half *) V->दत्त;
        k_str = elem_strides(K);
        v_str = elem_strides(V);
    } अन्यथा यदि (K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16) {
        K_ptr = stage_k((size_t) Hkv * seq * d);
        V_ptr = stage_v((size_t) Hkv * seq * d);
        cont_गन्तव्यम्_f16_sycl<sycl::half>((ध्रुव वर्ण *) K->दत्त, K_ptr, d, seq, Hkv, mb, K->nb[1], K->nb[2], K->nb[3], stream);
        cont_गन्तव्यम्_f16_sycl<sycl::half>((ध्रुव वर्ण *) V->दत्त, V_ptr, d, seq, Hkv, mb, V->nb[1], V->nb[2], V->nb[3], stream);
    } अन्यथा यदि (ggml_is_quantized(K->type)) {
        // Quantized K/V: dequant to dense F16 using pool, same lifetime as F16 path.
        K_ptr = stage_k((size_t) ggml_nelements(K));
        {
            ध्रुव वर्ण * K_दत्त = (ध्रुव वर्ण *)K->दत्त;
            ध्रुव तार्किक k_non_dense = ((int64_t)K->ne[1] * K->nb[1] != K->nb[2]) && K->ne[2] > 1;
            ध्रुव तार्किक k_gemma = k_non_dense &&
                ((int64_t)K->nb[2] < (int64_t)K->ne[1] * (int64_t)K->nb[1]);
            यदि (ggml_is_contiguously_allocated(K) && !k_non_dense) {
                गन्तव्यम्_fp16_sycl_t गन्तव्यम्_fp16 = ggml_get_गन्तव्यम्_fp16_sycl(K->type, dst);
                गन्तव्यम्_fp16(K_दत्त, K_ptr, ggml_nelements(K), stream);
            } अन्यथा {
                ध्रुव size_t bs = ggml_blck_आकार(K->type);
                ध्रुव size_t ts = ggml_type_आकार(K->type);
                गन्तव्यम्_fp16_nc_sycl_t गन्तव्यम्_fp16 = ggml_get_गन्तव्यम्_fp16_nc_sycl(K->type);
                int64_t s01, s02, s03;
                यदि (k_gemma) {
                    ध्रुव int64_t blk_प्रत्येकम्_row = (int64_t)K->ne[0] / bs;
                    s01 = (int64_t)Hkv * blk_प्रत्येकम्_row;
                    s02 = blk_प्रत्येकम्_row;
                    s03 = (int64_t)K->ne[1] * s01;
                } अन्यथा {
                    s01 = (int64_t)K->nb[1] / ts;
                    s02 = (int64_t)K->nb[2] / ts;
                    s03 = (int64_t)K->nb[3] / ts;
                }
                गन्तव्यम्_fp16(K_दत्त, K_ptr,
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
            ध्रुव वर्ण * V_दत्त = (ध्रुव वर्ण *)V->दत्त;
            ध्रुव तार्किक v_non_dense = ((int64_t)V->ne[1] * V->nb[1] != V->nb[2]) && V->ne[2] > 1;
            ध्रुव तार्किक v_gemma = v_non_dense &&
                ((int64_t)V->nb[2] < (int64_t)V->ne[1] * (int64_t)V->nb[1]);
            यदि (ggml_is_contiguously_allocated(V) && !v_non_dense) {
                गन्तव्यम्_fp16_sycl_t गन्तव्यम्_fp16 = ggml_get_गन्तव्यम्_fp16_sycl(V->type, dst);
                गन्तव्यम्_fp16(V_दत्त, V_ptr, ggml_nelements(V), stream);
            } अन्यथा {
                ध्रुव size_t bs = ggml_blck_आकार(V->type);
                ध्रुव size_t ts = ggml_type_आकार(V->type);
                गन्तव्यम्_fp16_nc_sycl_t गन्तव्यम्_fp16 = ggml_get_गन्तव्यम्_fp16_nc_sycl(V->type);
                int64_t s01, s02, s03;
                यदि (v_gemma) {
                    ध्रुव int64_t blk_प्रत्येकम्_row = (int64_t)V->ne[0] / bs;
                    s01 = (int64_t)V->ne[2] * blk_प्रत्येकम्_row;
                    s02 = blk_प्रत्येकम्_row;
                    s03 = (int64_t)V->ne[1] * s01;
                } अन्यथा {
                    s01 = (int64_t)V->nb[1] / ts;
                    s02 = (int64_t)V->nb[2] / ts;
                    s03 = (int64_t)V->nb[3] / ts;
                }
                गन्तव्यम्_fp16(V_दत्त, V_ptr,
                        V->ne[0], V->ne[1], V->ne[2], V->ne[3],
                        s01, s02, s03, stream);
            }
        }
    } अन्यथा {
        // F32: strided copy to dense F16 via cont_to_f16_sycl<float>.
        K_ptr = stage_k((size_t) ggml_nelements(K));
        cont_गन्तव्यम्_f16_sycl<भिन्न>((ध्रुव वर्ण *) K->दत्त, K_ptr, K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                                K->nb[1], K->nb[2], K->nb[3], stream);
        V_ptr = stage_v((size_t) ggml_nelements(V));
        cont_गन्तव्यम्_f16_sycl<भिन्न>((ध्रुव वर्ण *) V->दत्त, V_ptr, V->ne[0], V->ne[1], V->ne[2], V->ne[3],
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
    ध्रुव sycl::half scale_h = (sycl::half) (1.0f / kq_scale);
    std::optional<ggml_sycl_pool_alloc<sycl::half>> scbuf;
    sycl::half * scale_dev = (sycl::half *) extra.scale_बफर_ptr;
    यदि (!scale_dev) {
        scbuf.emplace(ctx.pool(), 1);
        scale_dev = scbuf->get();
    }
    stream->single_task([=]() { *scale_dev = scale_h; });

    // f16 contiguous SDPA out [mb,H,q,d]
    std::optional<ggml_sycl_pool_alloc<sycl::half>> outf_pool;
    sycl::half * outf_ptr = (sycl::half *) extra.बहिर्गमनम्_बफर_ptr;
    यदि (!outf_ptr) {
        outf_pool.emplace(ctx.pool(), (size_t) H * q * d);
        outf_ptr = outf_pool->get();
    }

    // compile once per (device, shape, KV strides), reuse across layers/calls. Stride 2 always
    // repeats stride 1 and stride 4 is always 1, so the key covers every entry that can differ.
    स्थिर std::unordered_map<std::string, sdpa_partition> cache;
    वर्ण keyb[256];
    snprintf(keyb, परिमाण(keyb), "%d:%lld:%lld:%lld:%lld:%lld:%lld:%lld:%lld:%lld:%lld:%lld", ggml_sycl_get_device(),
             (दीर्घ दीर्घ) H, (दीर्घ दीर्घ) Hkv, (दीर्घ दीर्घ) q, (दीर्घ दीर्घ) seq, (दीर्घ दीर्घ) d,
             (दीर्घ दीर्घ) k_str[0], (दीर्घ दीर्घ) k_str[1], (दीर्घ दीर्घ) k_str[3],
             (दीर्घ दीर्घ) v_str[0], (दीर्घ दीर्घ) v_str[1], (दीर्घ दीर्घ) v_str[3]);
    स्वचालित it = cache.find(keyb);
    यदि (it == cache.end()) {
        it = cache.emplace(keyb, build_sdpa(eng, (पूर्णाङ्क) H, (पूर्णाङ्क) Hkv, (पूर्णाङ्क) q, (पूर्णाङ्क) seq, (पूर्णाङ्क) d, k_str, v_str)).first;
    }
    sdpa_partition & E = it->second;
    यदि (!E.ok) {
        // oneDNN can decline a shape or a stride set that _supported() never sees; build_sdpa warns per key.
        ggml_sycl_flash_attn_ext_tile(ctx, dst);
        प्रत्यागम;
    }

    स्वचालित id2ptr = [&](size_t r) -> शून्य * {
        यदि (r == E.id_q)     प्रत्यागम Qf_ptr;
        यदि (r == E.id_k)     प्रत्यागम K_ptr;
        यदि (r == E.id_v)     प्रत्यागम V_ptr;
        यदि (r == E.id_scale) प्रत्यागम scale_dev;
        यदि (r == E.id_आवरणम्)  प्रत्यागम (शून्य *) आवरणम्->दत्त;
        प्रत्यागम शून्यसूचकः;
    };
    std::vector<tensor> ti;
    ti.reserve(E.ins.size());
    प्रति (स्वचालित & lt : E.ins) {
        ti.emplace_back(lt, eng, id2ptr(lt.get_id()));
    }
    tensor गन्तव्यम्(E.बहिर्गमनम्, eng, outf_ptr);
    E.cp.execute(strm, ti, {गन्तव्यम्});

    permute_sdpa_बहिर्गमनम्_sycl(outf_ptr, (भिन्न *) dst->दत्त, mb, H, q, d, stream);
    // Single device needs no sync: the dnnl stream wraps this same in-order queue, so the SDPA
    // serializes with the staging kernels before it and the permute/pool reuse after it. The
    // garbage output formerly blamed on the missing sync here was the scale use-after-return
    // fixed above. Keep the conservative wait for multi-GPU, where other devices' streams can
    // race the pool:
    यदि (ggml_sycl_info().device_गणनम् > 1) {
        stream->wait_and_throw();
    }
}
गृहाण (ध्रुव std::exception & e) {
    // any oneDNN/SYCL failure is non-fatal: fall back to the existing kernel (strictly additive).
    GGML_LOG_WARN("%s: oneDNN SDPA failed (%s); falling back to TILE kernel\n", __func__, e.what());
    ggml_sycl_flash_attn_ext_tile(ctx, dst);
}

#यदि_अन्तः // GGML_SYCL_DNNL
