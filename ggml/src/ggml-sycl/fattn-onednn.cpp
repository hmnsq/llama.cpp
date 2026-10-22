#అన్తర్భావ <array>
#అన్తర్భావ <cstdint>
#అన్తర్భావ <cstdio>
#అన్తర్భావ <cstring>
#అన్తర్భావ <string>
#అన్తర్భావ <optional>
#అన్తర్భావ <unordered_map>
#అన్తర్భావ <vector>

#అన్తర్భావ "fattn-onednn.hpp"
#అన్తర్భావ "fattn-tile.hpp"
#అన్తర్భావ "convert.hpp"

// set minimum query length to treat as prefill (32)
#నిర్వచించు GGML_SYCL_FA_ONEDNN_MIN_Q 32

తార్కిక ggml_sycl_fattn_onednn_binds_kv(స్థిరరాశి ggml_tensor * K, స్థిరరాశి ggml_tensor * V) {
    ఒకవేళ (K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16) {
        తిరిగి అబద్ధం;
    }
    స్వయంచాలక bindable = [](స్థిరరాశి ggml_tensor * t) {
        తిరిగి t->nb[0] == పరిమాణం(sycl::half) && t->nb[1] % పరిమాణం(sycl::half) == 0 &&
               t->nb[2] % పరిమాణం(sycl::half) == 0 && t->nb[3] % పరిమాణం(sycl::half) == 0;
    };
    తిరిగి bindable(K) && bindable(V);
}

తార్కిక ggml_sycl_flash_attn_ext_onednn_supported(స్థిరరాశి ggml_tensor * dst, తార్కిక use_shape_హద్దు) {
#ఒకవేళ !GGML_SYCL_DNNL
    GGML_వాడని(dst);
    GGML_వాడని(use_shape_హద్దు);
    తిరిగి అబద్ధం;
#లేకపోతే
    ఒకవేళ (!g_ggml_sycl_fa_onednn) {
        తిరిగి అబద్ధం;
    }
    // without a fused SDPA kernel oneDNN falls back to the same matmul, so reuse the check
    ఒకవేళ (!ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
        తిరిగి అబద్ధం;
    }
    స్థిరరాశి ggml_tensor * Q     = dst->src[0];
    స్థిరరాశి ggml_tensor * K     = dst->src[1];
    స్థిరరాశి ggml_tensor * V     = dst->src[2];
    స్థిరరాశి ggml_tensor * ముసుగు  = dst->src[3];
    స్థిరరాశి ggml_tensor * sinks = dst->src[4];

    // F16 KV: native SDPA at any KV length.
    // Non-F16: dequant to F16 then SDPA at prefill lengths. Only the
    // standard quantized KV cache types (Q4_0-Q8_0) and F32 are accepted
    // because their to_fp16_sycl conversion is verified. BF16 and IQ*
    // are excluded: BF16 needs a strided conversion kernel that does not
    // exist yet; IQ types are model-weight-only quants with no dequant
    // registration and are never used as KV caches.
    ఒకవేళ (K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16) {
        స్వయంచాలక kt = K->type, vt = V->type;
        తార్కిక k_ok = kt == GGML_TYPE_F32 || kt == GGML_TYPE_Q4_0 || kt == GGML_TYPE_Q4_1 ||
                    kt == GGML_TYPE_Q5_0 || kt == GGML_TYPE_Q5_1 || kt == GGML_TYPE_Q8_0;
        తార్కిక v_ok = vt == GGML_TYPE_F32 || vt == GGML_TYPE_Q4_0 || vt == GGML_TYPE_Q4_1 ||
                    vt == GGML_TYPE_Q5_0 || vt == GGML_TYPE_Q5_1 || vt == GGML_TYPE_Q8_0;
        ఒకవేళ (!k_ok || !v_ok) {
            తిరిగి అబద్ధం;
        }
        ఒకవేళ (use_shape_హద్దు && (Q->ne[1] < 32 || K->ne[1] < 1024)) {
            తిరిగి అబద్ధం;
        }
        కొరకు (స్థిరరాశి ggml_tensor * t : {K, V}) {
            ఒకవేళ (t->type == GGML_TYPE_F16 && t->nb[1] % (t->ne[0] * 2) != 0) {
                తిరిగి అబద్ధం;
            }
        }
    }
    // This is the improved SPDA gate. Rather than gating Alchemist GPUs from all SPDA features, we instead target only the failing shapes.
    // If the GPU being assessed isn't in the grouping below, it has full access to all SPDA shapes. Otherwise, if it's an Alchemist GPU, we block only the shapes with head sizes that fail.
    // It is much easier to compare the device to a small list of failing cases than to define all the passing ones.
    స్థిరరాశి gpu_arch arch = ggml_sycl_info().devices[ggml_sycl_get_device()].hw_info.arch;
    తార్కిక support_spda = !(arch == gpu_arch::intel_gpu_dg2_g10 ||
                         arch == gpu_arch::intel_gpu_dg2_g11 ||
                         arch == gpu_arch::intel_gpu_dg2_g12);

    ఒకవేళ (!support_spda && K->ne[0] == 64) {
        తిరిగి అబద్ధం;
    }
    // Optional KV-length ceiling (GGML_SYCL_FA_ONEDNN_MAX_KV, 0 = unlimited). Escape hatch:
    // very long sequences make the fused SDPA slow enough to risk the xe driver watchdog on
    // some stacks; past the cap we fall back to the native FA kernel instead.
    ఒకవేళ (g_ggml_sycl_fa_onednn_max_kv > 0 && K->ne[1] > g_ggml_sycl_fa_onednn_max_kv) {
        తిరిగి అబద్ధం;
    }
    // gate for the following cases
    // 1. if the oneDNN graph Add node has no input --> skip
    // 2. types other than f16 need different logical_tensor declaration
    // 3. the mask must be shape [1, 1, q, seq]
    // 4. sinks: excludes attention sink (Xiao et al., 2024) that can't be modeled by oneDNN graph
    ఒకవేళ (!ముసుగు || ముసుగు->type != GGML_TYPE_F16 || ముసుగు->ne[2] != 1 || ముసుగు->ne[3] != 1 || sinks) {
        తిరిగి అబద్ధం;
    }
    దశాంశం max_bias = 0.0f, logit_softcap = 0.0f;
    జ్ఞాపకం_నకలు(&max_bias,      (స్థిరరాశి దశాంశం *) dst->op_params + 1, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&logit_softcap, (స్థిరరాశి దశాంశం *) dst->op_params + 2, పరిమాణం(దశాంశం));
    ఒకవేళ (max_bias != 0.0f || logit_softcap != 0.0f) {
        తిరిగి అబద్ధం;
    }
    // K and V must share head_dim: the SDPA graph uses a single `d` for both.
    స్థిరరాశి int64_t d = K->ne[0];
    ఒకవేళ (V->ne[0] != d || Q->ne[3] != 1) {
        తిరిగి అబద్ధం;
    }
    // GQA must divide evenly.
    ఒకవేళ (K->ne[2] == 0 || Q->ne[2] % K->ne[2] != 0) {
        తిరిగి అబద్ధం;
    }
    // Prefill only.
    ఒకవేళ (use_shape_హద్దు && Q->ne[1] < GGML_SYCL_FA_ONEDNN_MIN_Q) {
        తిరిగి అబద్ధం;
    }
    తిరిగి నిజం;
#అయితే_ముగింపు
}

#ఒకవేళ GGML_SYCL_DNNL

#అన్తర్భావ "dnnl.hpp"
#అన్తర్భావ "dnnl_sycl.hpp"
#అన్తర్భావ "oneapi/dnnl/dnnl_graph.hpp"   // graph API lives only under oneapi/dnnl/, not at the include root

ఉపయోగించి నామాకాశ dnnl;
ఉపయోగించి నామాకాశ dnnl::graph;

// strided src (f16 or f32) -> contiguous f16 [ne0,ne1,ne2,ne3] (ne0 innermost). nb* are BYTE strides.
మూస <రకం_పేరు src_t>
స్థిర శూన్య cont_గమ్యం_f16_sycl(స్థిరరాశి అక్షరం * src, sycl::half * dst,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
        size_t nb1, size_t nb2, size_t nb3, dpct::వరుస_ptr stream) {
    స్థిరరాశి int64_t n = ne0 * ne1 * ne2 * ne3;
    stream->parallel_for(sycl::range<1>(n), [=](sycl::id<1> ix) {
        స్థిరరాశి int64_t gid = ix[0];
        int64_t       i   = gid;
        స్థిరరాశి int64_t i0 = i % ne0; i /= ne0;
        స్థిరరాశి int64_t i1 = i % ne1; i /= ne1;
        స్థిరరాశి int64_t i2 = i % ne2; స్థిరరాశి int64_t i3 = i / ne2;
        స్థిరరాశి src_t * p = (స్థిరరాశి src_t *) (src + i1 * nb1 + i2 * nb2 + i3 * nb3) + i0;
        dst[gid] = (sycl::half) (*p);
    });
}

// oneDNN SDPA out (f16 contiguous [mb,H,q,d]) -> ggml dst (f32 [head_dim,H,n_tok,mb], contiguous).
స్థిర శూన్య క్రమమార్పు_sdpa_బహిర్గతం_sycl(స్థిరరాశి sycl::half * బహిర్గతం, దశాంశం * dst,
        int64_t mb, int64_t H, int64_t q, int64_t d, dpct::వరుస_ptr stream) {
    స్థిరరాశి int64_t n = mb * H * q * d;
    stream->parallel_for(sycl::range<1>(n), [=](sycl::id<1> ix) {
        స్థిరరాశి int64_t gid = ix[0];
        int64_t       i   = gid;
        స్థిరరాశి int64_t e = i % d; i /= d;
        స్థిరరాశి int64_t t = i % q; i /= q;
        స్థిరరాశి int64_t h = i % H; స్థిరరాశి int64_t b = i / H;
        dst[e + h * d + t * d * H + b * d * H * q] = (దశాంశం) బహిర్గతం[gid];
    });
}

నిర్మాణం sdpa_విభజన {
    compiled_విభజన          cp;
    std::vector<logical_tensor> ins;
    logical_tensor              బహిర్గతం;
    size_t id_q = 0, id_k = 0, id_v = 0, id_scale = 0, id_ముసుగు = 0;
    తార్కిక   ok = అబద్ధం;
};

// Build + compile the contiguous-input GQA SDPA graph (MatMul->Divide->Add->SoftMax->MatMul), f32 out.
// Mirrors the hardware-verified scratch/onednn_sdpa_probe.cpp build_gqa (partitions=1, sdp_primitive_kernel_t).
స్థిర sdpa_విభజన build_sdpa(స్థిరరాశి engine & eng, పూర్ణాంకం H, పూర్ణాంకం Hkv, పూర్ణాంకం q, పూర్ణాంకం seq, పూర్ణాంకం d,
                                 స్థిరరాశి std::array<int64_t, 5> & k_str, స్థిరరాశి std::array<int64_t, 5> & v_str) ప్రయత్నించు {
    ఉపయోగించి ltype = logical_tensor::layout_type;
    ఉపయోగించి dt    = logical_tensor::దత్తాంశం_type;
    ఉపయోగించి ldims = logical_tensor::dims;
    స్థిరరాశి dt    fi = dt::f32, t = dt::f16;
    స్థిరరాశి పూర్ణాంకం   rep = H / Hkv;
    స్థిరరాశి ldims q_sz = {1, Hkv, rep, q, d}, kv_sz = {1, Hkv, 1, seq, d}, s_sz = {1, Hkv, rep, q, seq},
                sc = {1, 1, 1, 1, 1}, msk = {1, 1, 1, q, seq}, o_sz = {1, Hkv, rep, q, d};
    స్థిరరాశి ldims k_st(k_str.begin(), k_str.end()), v_st(v_str.begin(), v_str.end());
    int64_t        id = 0;
    sdpa_విభజన E;

    స్వయంచాలక query  = logical_tensor(id++, t,  q_sz, ltype::strided);
    స్వయంచాలక కీలకం    = logical_tensor(id++, t,  kv_sz, k_st);
    స్వయంచాలక score  = logical_tensor(id++, fi, s_sz, ltype::strided);
    స్వయంచాలక bmm1   = op(id++, op::రకం_2::MatMul, "bmm1");
    bmm1.సమితి_attr<తార్కిక>(op::attr::transpose_b, నిజం);          // key is [.., seq, d]
    bmm1.కూడు_inputs({query, కీలకం}); bmm1.కూడు_outputs({score});

    స్వయంచాలక scale  = logical_tensor(id++, t,  sc,   ltype::strided);
    స్వయంచాలక scaled = logical_tensor(id++, fi, s_sz, ltype::strided);
    స్వయంచాలక sdiv   = op(id++, op::రకం_2::Divide, "scale_div");     // score / (1/kq_scale) == score * kq_scale
    sdiv.కూడు_inputs({score, scale}); sdiv.కూడు_outputs({scaled});

    స్వయంచాలక ముసుగు   = logical_tensor(id++, t,  msk,  ltype::strided);
    స్వయంచాలక masked = logical_tensor(id++, fi, s_sz, ltype::strided);
    స్వయంచాలక madd   = op(id++, op::రకం_2::కూడు, "mask_add");
    madd.కూడు_inputs({scaled, ముసుగు}); madd.కూడు_outputs({masked});

    స్వయంచాలక probs  = logical_tensor(id++, t,  s_sz, ltype::strided);
    స్వయంచాలక smax   = op(id++, op::రకం_2::SoftMax, "softmax");
    smax.సమితి_attr<int64_t>(op::attr::axis, -1);
    smax.సమితి_attr<std::string>(op::attr::mode, "inf_as_zero");
    smax.కూడు_inputs({masked}); smax.కూడు_outputs({probs});

    స్వయంచాలక value  = logical_tensor(id++, t,  kv_sz, v_st);
    // f16 output is REQUIRED to hit sdp_primitive_kernel_t (the systolic micro-kernel); an f32 output
    // falls to larger_partition_kernel_t which materializes N^2 (confirmed: scratch/onednn_sdpa_kernel_probe.cpp).
    // converted to the f32 ggml dst in the permute below.
    స్వయంచాలక output = logical_tensor(id++, t,  o_sz, ltype::strided);   // f16 contiguous [mb,Hkv,rep,q,d]
    స్వయంచాలక bmm2   = op(id++, op::రకం_2::MatMul, "bmm2");
    bmm2.కూడు_inputs({probs, value}); bmm2.కూడు_outputs({output});

    dnnl::graph::graph g(eng.get_రకం());
    g.కూడు_op(bmm1); g.కూడు_op(sdiv); g.కూడు_op(madd); g.కూడు_op(smax); g.కూడు_op(bmm2);
    g.finalize();

    స్వయంచాలక parts = g.get_partitions();
    ఒకవేళ (parts.size() != 1 || !parts[0].is_supported()) {
        GGML_LOG_WARN("%s: oneDNN did not fuse the SDPA graph; falling back to TILE kernel\n", __func__);
        తిరిగి E;   // ok stays false -> caller falls back to TILE
    }
    E.ins      = parts[0].get_ఆదానం_ports();
    E.బహిర్గతం      = parts[0].get_output_ports()[0];
    E.cp       = parts[0].compile(E.ins, {E.బహిర్గతం}, eng);
    E.బహిర్గతం      = E.cp.query_logical_tensor(E.బహిర్గతం.get_id());
    E.id_q     = query.get_id(); E.id_k = కీలకం.get_id(); E.id_v = value.get_id();
    E.id_scale = scale.get_id(); E.id_ముసుగు = ముసుగు.get_id();
    E.ok       = నిజం;
    తిరిగి E;
}
పట్టుకో (స్థిరరాశి std::exception & e) {
    // compile() can reject a stride set the partitioner never inspects; memoise the failure so the
    // fallback costs one build rather than one per call.
    GGML_LOG_WARN("%s: oneDNN SDPA partition build failed (%s); falling back to TILE kernel\n", __func__, e.what());
    తిరిగి {};
}

శూన్య ggml_sycl_flash_attn_ext_onednn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) ప్రయత్నించు {
    స్థిరరాశి ggml_tensor * Q    = dst->src[0];
    స్థిరరాశి ggml_tensor * K    = dst->src[1];
    స్థిరరాశి ggml_tensor * V    = dst->src[2];
    స్థిరరాశి ggml_tensor * ముసుగు = dst->src[3];

    స్థిరరాశి int64_t d   = K->ne[0];   // head_dim
    స్థిరరాశి int64_t seq = K->ne[1];   // n_kv
    స్థిరరాశి int64_t Hkv = K->ne[2];   // n_head_kv
    స్థిరరాశి int64_t H   = Q->ne[2];   // n_head
    స్థిరరాశి int64_t q   = Q->ne[1];   // n_tok
    స్థిరరాశి int64_t mb  = Q->ne[3];   // batch (== 1, gated)

    దశాంశం kq_scale = 1.0f;
    జ్ఞాపకం_నకలు(&kq_scale, (స్థిరరాశి దశాంశం *) dst->op_params + 0, పరిమాణం(దశాంశం));

    dpct::వరుస_ptr stream = ctx.stream();
    dnnl::engine    eng    = ctx.engine_dnnl(stream);
    dnnl::stream    strm   = ctx.ప్రవాహం_dnnl(stream);

    స్థిరరాశి ggml_sycl_fattn_extra extra = ggml_sycl_fattn_get_extra(dst);

    // Q: always f32 -- copy to dense f16.
    std::optional<ggml_sycl_pool_alloc<sycl::half>> Qf_pool;
    sycl::half * Qf_ptr = (sycl::half *) extra.Q_బఫరు_ptr;
    ఒకవేళ (!Qf_ptr) {
        Qf_pool.emplace(ctx.pool(), (size_t) H * q * d);
        Qf_ptr = Qf_pool->get();
    }
    cont_గమ్యం_f16_sycl<దశాంశం>((స్థిరరాశి అక్షరం *) Q->దత్తాంశం, Qf_ptr, d, q, H, mb, Q->nb[1], Q->nb[2], Q->nb[3], stream);

    // K/V: bind the f16 cache in place. llama.cpp permutes it to [token][head][dim], so its head
    // plane is strided rather than dense, which is what an explicit stride vector expresses.
    // Quantized and f32 KV still stage a dense copy -- the layout the k_str/v_str defaults describe.
    sycl::half * K_ptr = శూన్యసూచిక;
    sycl::half * V_ptr = శూన్యసూచిక;
    std::array<int64_t, 5> k_str{ Hkv * seq * d, seq * d, seq * d, d, 1 };
    std::array<int64_t, 5> v_str = k_str;
    std::optional<ggml_sycl_pool_alloc<sycl::half>> Kf_pool;
    std::optional<ggml_sycl_pool_alloc<sycl::half>> Vf_pool;
    // Helper: hand out reserved space, or fall back to the pool.
    స్వయంచాలక stage_k = [&](size_t n) { ఒకవేళ (extra.K_బఫరు_ptr) { తిరిగి (sycl::half *) extra.K_బఫరు_ptr; }
                                  Kf_pool.emplace(ctx.pool(), n); తిరిగి Kf_pool->get(); };
    స్వయంచాలక stage_v = [&](size_t n) { ఒకవేళ (extra.V_బఫరు_ptr) { తిరిగి (sycl::half *) extra.V_బఫరు_ptr; }
                                  Vf_pool.emplace(ctx.pool(), n); తిరిగి Vf_pool->get(); };

    స్వయంచాలక elem_strides = [](స్థిరరాశి ggml_tensor * t) {
        స్థిరరాశి int64_t s1 = (int64_t) (t->nb[1] / t->nb[0]);
        స్థిరరాశి int64_t s2 = (int64_t) (t->nb[2] / t->nb[0]);
        స్థిరరాశి int64_t s3 = (int64_t) (t->nb[3] / t->nb[0]);
        // dims are {mb=1, Hkv, rep=1, seq, d}; the size-1 dims at 0 and 2 never advance an address.
        తిరిగి std::array<int64_t, 5>{ s3, s2, s2, s1, 1 };
    };

    ఒకవేళ (ggml_sycl_fattn_onednn_binds_kv(K, V)) {
        K_ptr = (sycl::half *) K->దత్తాంశం;
        V_ptr = (sycl::half *) V->దత్తాంశం;
        k_str = elem_strides(K);
        v_str = elem_strides(V);
    } లేకపోతే ఒకవేళ (K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16) {
        K_ptr = stage_k((size_t) Hkv * seq * d);
        V_ptr = stage_v((size_t) Hkv * seq * d);
        cont_గమ్యం_f16_sycl<sycl::half>((స్థిరరాశి అక్షరం *) K->దత్తాంశం, K_ptr, d, seq, Hkv, mb, K->nb[1], K->nb[2], K->nb[3], stream);
        cont_గమ్యం_f16_sycl<sycl::half>((స్థిరరాశి అక్షరం *) V->దత్తాంశం, V_ptr, d, seq, Hkv, mb, V->nb[1], V->nb[2], V->nb[3], stream);
    } లేకపోతే ఒకవేళ (ggml_is_quantized(K->type)) {
        // Quantized K/V: dequant to dense F16 using pool, same lifetime as F16 path.
        K_ptr = stage_k((size_t) ggml_nelements(K));
        {
            స్థిరరాశి అక్షరం * K_దత్తాంశం = (స్థిరరాశి అక్షరం *)K->దత్తాంశం;
            స్థిరరాశి తార్కిక k_non_dense = ((int64_t)K->ne[1] * K->nb[1] != K->nb[2]) && K->ne[2] > 1;
            స్థిరరాశి తార్కిక k_gemma = k_non_dense &&
                ((int64_t)K->nb[2] < (int64_t)K->ne[1] * (int64_t)K->nb[1]);
            ఒకవేళ (ggml_is_contiguously_allocated(K) && !k_non_dense) {
                గమ్యం_fp16_sycl_t గమ్యం_fp16 = ggml_get_గమ్యం_fp16_sycl(K->type, dst);
                గమ్యం_fp16(K_దత్తాంశం, K_ptr, ggml_nelements(K), stream);
            } లేకపోతే {
                స్థిరరాశి size_t bs = ggml_blck_కొలత(K->type);
                స్థిరరాశి size_t ts = ggml_type_కొలత(K->type);
                గమ్యం_fp16_nc_sycl_t గమ్యం_fp16 = ggml_get_గమ్యం_fp16_nc_sycl(K->type);
                int64_t s01, s02, s03;
                ఒకవేళ (k_gemma) {
                    స్థిరరాశి int64_t blk_చొప్పున_అడ్డువరుస = (int64_t)K->ne[0] / bs;
                    s01 = (int64_t)Hkv * blk_చొప్పున_అడ్డువరుస;
                    s02 = blk_చొప్పున_అడ్డువరుస;
                    s03 = (int64_t)K->ne[1] * s01;
                } లేకపోతే {
                    s01 = (int64_t)K->nb[1] / ts;
                    s02 = (int64_t)K->nb[2] / ts;
                    s03 = (int64_t)K->nb[3] / ts;
                }
                గమ్యం_fp16(K_దత్తాంశం, K_ptr,
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
            స్థిరరాశి అక్షరం * V_దత్తాంశం = (స్థిరరాశి అక్షరం *)V->దత్తాంశం;
            స్థిరరాశి తార్కిక v_non_dense = ((int64_t)V->ne[1] * V->nb[1] != V->nb[2]) && V->ne[2] > 1;
            స్థిరరాశి తార్కిక v_gemma = v_non_dense &&
                ((int64_t)V->nb[2] < (int64_t)V->ne[1] * (int64_t)V->nb[1]);
            ఒకవేళ (ggml_is_contiguously_allocated(V) && !v_non_dense) {
                గమ్యం_fp16_sycl_t గమ్యం_fp16 = ggml_get_గమ్యం_fp16_sycl(V->type, dst);
                గమ్యం_fp16(V_దత్తాంశం, V_ptr, ggml_nelements(V), stream);
            } లేకపోతే {
                స్థిరరాశి size_t bs = ggml_blck_కొలత(V->type);
                స్థిరరాశి size_t ts = ggml_type_కొలత(V->type);
                గమ్యం_fp16_nc_sycl_t గమ్యం_fp16 = ggml_get_గమ్యం_fp16_nc_sycl(V->type);
                int64_t s01, s02, s03;
                ఒకవేళ (v_gemma) {
                    స్థిరరాశి int64_t blk_చొప్పున_అడ్డువరుస = (int64_t)V->ne[0] / bs;
                    s01 = (int64_t)V->ne[2] * blk_చొప్పున_అడ్డువరుస;
                    s02 = blk_చొప్పున_అడ్డువరుస;
                    s03 = (int64_t)V->ne[1] * s01;
                } లేకపోతే {
                    s01 = (int64_t)V->nb[1] / ts;
                    s02 = (int64_t)V->nb[2] / ts;
                    s03 = (int64_t)V->nb[3] / ts;
                }
                గమ్యం_fp16(V_దత్తాంశం, V_ptr,
                        V->ne[0], V->ne[1], V->ne[2], V->ne[3],
                        s01, s02, s03, stream);
            }
        }
    } లేకపోతే {
        // F32: strided copy to dense F16 via cont_to_f16_sycl<float>.
        K_ptr = stage_k((size_t) ggml_nelements(K));
        cont_గమ్యం_f16_sycl<దశాంశం>((స్థిరరాశి అక్షరం *) K->దత్తాంశం, K_ptr, K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                                K->nb[1], K->nb[2], K->nb[3], stream);
        V_ptr = stage_v((size_t) ggml_nelements(V));
        cont_గమ్యం_f16_sycl<దశాంశం>((స్థిరరాశి అక్షరం *) V->దత్తాంశం, V_ptr, V->ne[0], V->ne[1], V->ne[2], V->ne[3],
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
    స్థిరరాశి sycl::half scale_h = (sycl::half) (1.0f / kq_scale);
    std::optional<ggml_sycl_pool_alloc<sycl::half>> scbuf;
    sycl::half * scale_dev = (sycl::half *) extra.scale_బఫరు_ptr;
    ఒకవేళ (!scale_dev) {
        scbuf.emplace(ctx.pool(), 1);
        scale_dev = scbuf->get();
    }
    stream->single_task([=]() { *scale_dev = scale_h; });

    // f16 contiguous SDPA out [mb,H,q,d]
    std::optional<ggml_sycl_pool_alloc<sycl::half>> outf_pool;
    sycl::half * outf_ptr = (sycl::half *) extra.బహిర్గతం_బఫరు_ptr;
    ఒకవేళ (!outf_ptr) {
        outf_pool.emplace(ctx.pool(), (size_t) H * q * d);
        outf_ptr = outf_pool->get();
    }

    // compile once per (device, shape, KV strides), reuse across layers/calls. Stride 2 always
    // repeats stride 1 and stride 4 is always 1, so the key covers every entry that can differ.
    స్థిర std::unordered_map<std::string, sdpa_విభజన> cache;
    అక్షరం keyb[256];
    snprintf(keyb, పరిమాణం(keyb), "%d:%lld:%lld:%lld:%lld:%lld:%lld:%lld:%lld:%lld:%lld:%lld", ggml_sycl_get_device(),
             (దీర్ఘ దీర్ఘ) H, (దీర్ఘ దీర్ఘ) Hkv, (దీర్ఘ దీర్ఘ) q, (దీర్ఘ దీర్ఘ) seq, (దీర్ఘ దీర్ఘ) d,
             (దీర్ఘ దీర్ఘ) k_str[0], (దీర్ఘ దీర్ఘ) k_str[1], (దీర్ఘ దీర్ఘ) k_str[3],
             (దీర్ఘ దీర్ఘ) v_str[0], (దీర్ఘ దీర్ఘ) v_str[1], (దీర్ఘ దీర్ఘ) v_str[3]);
    స్వయంచాలక it = cache.find(keyb);
    ఒకవేళ (it == cache.end()) {
        it = cache.emplace(keyb, build_sdpa(eng, (పూర్ణాంకం) H, (పూర్ణాంకం) Hkv, (పూర్ణాంకం) q, (పూర్ణాంకం) seq, (పూర్ణాంకం) d, k_str, v_str)).మొదటి;
    }
    sdpa_విభజన & E = it->second;
    ఒకవేళ (!E.ok) {
        // oneDNN can decline a shape or a stride set that _supported() never sees; build_sdpa warns per key.
        ggml_sycl_flash_attn_ext_tile(ctx, dst);
        తిరిగి;
    }

    స్వయంచాలక id2ptr = [&](size_t r) -> శూన్య * {
        ఒకవేళ (r == E.id_q)     తిరిగి Qf_ptr;
        ఒకవేళ (r == E.id_k)     తిరిగి K_ptr;
        ఒకవేళ (r == E.id_v)     తిరిగి V_ptr;
        ఒకవేళ (r == E.id_scale) తిరిగి scale_dev;
        ఒకవేళ (r == E.id_ముసుగు)  తిరిగి (శూన్య *) ముసుగు->దత్తాంశం;
        తిరిగి శూన్యసూచిక;
    };
    std::vector<tensor> ti;
    ti.reserve(E.ins.size());
    కొరకు (స్వయంచాలక & lt : E.ins) {
        ti.emplace_back(lt, eng, id2ptr(lt.get_id()));
    }
    tensor గమ్యం(E.బహిర్గతం, eng, outf_ptr);
    E.cp.execute(strm, ti, {గమ్యం});

    క్రమమార్పు_sdpa_బహిర్గతం_sycl(outf_ptr, (దశాంశం *) dst->దత్తాంశం, mb, H, q, d, stream);
    // Single device needs no sync: the dnnl stream wraps this same in-order queue, so the SDPA
    // serializes with the staging kernels before it and the permute/pool reuse after it. The
    // garbage output formerly blamed on the missing sync here was the scale use-after-return
    // fixed above. Keep the conservative wait for multi-GPU, where other devices' streams can
    // race the pool:
    ఒకవేళ (ggml_sycl_info().device_లెక్క > 1) {
        stream->wait_and_throw();
    }
}
పట్టుకో (స్థిరరాశి std::exception & e) {
    // any oneDNN/SYCL failure is non-fatal: fall back to the existing kernel (strictly additive).
    GGML_LOG_WARN("%s: oneDNN SDPA failed (%s); falling back to TILE kernel\n", __func__, e.what());
    ggml_sycl_flash_attn_ext_tile(ctx, dst);
}

#అయితే_ముగింపు // GGML_SYCL_DNNL
