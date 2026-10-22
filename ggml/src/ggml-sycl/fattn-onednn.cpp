#ಅನ್ತರ್ಭಾವ <array>
#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <cstdio>
#ಅನ್ತರ್ಭಾವ <cstring>
#ಅನ್ತರ್ಭಾವ <string>
#ಅನ್ತರ್ಭಾವ <optional>
#ಅನ್ತರ್ಭಾವ <unordered_map>
#ಅನ್ತರ್ಭಾವ <vector>

#ಅನ್ತರ್ಭಾವ "fattn-onednn.hpp"
#ಅನ್ತರ್ಭಾವ "fattn-tile.hpp"
#ಅನ್ತರ್ಭಾವ "convert.hpp"

// set minimum query length to treat as prefill (32)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SYCL_FA_ONEDNN_MIN_Q 32

ತಾರ್ಕಿಕ ggml_sycl_fattn_onednn_binds_kv(ಸ್ಥಿರಾಂಕ ggml_tensor * K, ಸ್ಥಿರಾಂಕ ggml_tensor * V) {
    ಆದರೆ (K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಸ್ವಯಂಚಾಲಿತ bindable = [](ಸ್ಥಿರಾಂಕ ggml_tensor * t) {
        ಹಿಂತಿರುಗಿ t->nb[0] == ಗಾತ್ರ(sycl::half) && t->nb[1] % ಗಾತ್ರ(sycl::half) == 0 &&
               t->nb[2] % ಗಾತ್ರ(sycl::half) == 0 && t->nb[3] % ಗಾತ್ರ(sycl::half) == 0;
    };
    ಹಿಂತಿರುಗಿ bindable(K) && bindable(V);
}

ತಾರ್ಕಿಕ ggml_sycl_flash_attn_ext_onednn_supported(ಸ್ಥಿರಾಂಕ ggml_tensor * dst, ತಾರ್ಕಿಕ use_shape_ಎಲ್ಲೆ) {
#ಆದರೆ !GGML_SYCL_DNNL
    GGML_ಬಳಸದ(dst);
    GGML_ಬಳಸದ(use_shape_ಎಲ್ಲೆ);
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
#ಇಲ್ಲದಿದ್ದರೆ
    ಆದರೆ (!g_ggml_sycl_fa_onednn) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    // without a fused SDPA kernel oneDNN falls back to the same matmul, so reuse the check
    ಆದರೆ (!ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಸ್ಥಿರಾಂಕ ggml_tensor * Q     = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * K     = dst->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * V     = dst->src[2];
    ಸ್ಥಿರಾಂಕ ggml_tensor * ಮುಖವಾಡ  = dst->src[3];
    ಸ್ಥಿರಾಂಕ ggml_tensor * sinks = dst->src[4];

    // F16 KV: native SDPA at any KV length.
    // Non-F16: dequant to F16 then SDPA at prefill lengths. Only the
    // standard quantized KV cache types (Q4_0-Q8_0) and F32 are accepted
    // because their to_fp16_sycl conversion is verified. BF16 and IQ*
    // are excluded: BF16 needs a strided conversion kernel that does not
    // exist yet; IQ types are model-weight-only quants with no dequant
    // registration and are never used as KV caches.
    ಆದರೆ (K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16) {
        ಸ್ವಯಂಚಾಲಿತ kt = K->type, vt = V->type;
        ತಾರ್ಕಿಕ k_ok = kt == GGML_TYPE_F32 || kt == GGML_TYPE_Q4_0 || kt == GGML_TYPE_Q4_1 ||
                    kt == GGML_TYPE_Q5_0 || kt == GGML_TYPE_Q5_1 || kt == GGML_TYPE_Q8_0;
        ತಾರ್ಕಿಕ v_ok = vt == GGML_TYPE_F32 || vt == GGML_TYPE_Q4_0 || vt == GGML_TYPE_Q4_1 ||
                    vt == GGML_TYPE_Q5_0 || vt == GGML_TYPE_Q5_1 || vt == GGML_TYPE_Q8_0;
        ಆದರೆ (!k_ok || !v_ok) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಆದರೆ (use_shape_ಎಲ್ಲೆ && (Q->ne[1] < 32 || K->ne[1] < 1024)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ggml_tensor * t : {K, V}) {
            ಆದರೆ (t->type == GGML_TYPE_F16 && t->nb[1] % (t->ne[0] * 2) != 0) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
        }
    }
    // This is the improved SPDA gate. Rather than gating Alchemist GPUs from all SPDA features, we instead target only the failing shapes.
    // If the GPU being assessed isn't in the grouping below, it has full access to all SPDA shapes. Otherwise, if it's an Alchemist GPU, we block only the shapes with head sizes that fail.
    // It is much easier to compare the device to a small list of failing cases than to define all the passing ones.
    ಸ್ಥಿರಾಂಕ gpu_arch arch = ggml_sycl_info().devices[ggml_sycl_get_device()].hw_info.arch;
    ತಾರ್ಕಿಕ support_spda = !(arch == gpu_arch::intel_gpu_dg2_g10 ||
                         arch == gpu_arch::intel_gpu_dg2_g11 ||
                         arch == gpu_arch::intel_gpu_dg2_g12);

    ಆದರೆ (!support_spda && K->ne[0] == 64) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    // Optional KV-length ceiling (GGML_SYCL_FA_ONEDNN_MAX_KV, 0 = unlimited). Escape hatch:
    // very long sequences make the fused SDPA slow enough to risk the xe driver watchdog on
    // some stacks; past the cap we fall back to the native FA kernel instead.
    ಆದರೆ (g_ggml_sycl_fa_onednn_max_kv > 0 && K->ne[1] > g_ggml_sycl_fa_onednn_max_kv) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    // gate for the following cases
    // 1. if the oneDNN graph Add node has no input --> skip
    // 2. types other than f16 need different logical_tensor declaration
    // 3. the mask must be shape [1, 1, q, seq]
    // 4. sinks: excludes attention sink (Xiao et al., 2024) that can't be modeled by oneDNN graph
    ಆದರೆ (!ಮುಖವಾಡ || ಮುಖವಾಡ->type != GGML_TYPE_F16 || ಮುಖವಾಡ->ne[2] != 1 || ಮುಖವಾಡ->ne[3] != 1 || sinks) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ದಶಮಾಂಶ max_bias = 0.0f, logit_softcap = 0.0f;
    ಸ್ಮರಣೆ_ನಕಲು(&max_bias,      (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) dst->op_params + 1, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&logit_softcap, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) dst->op_params + 2, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಆದರೆ (max_bias != 0.0f || logit_softcap != 0.0f) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    // K and V must share head_dim: the SDPA graph uses a single `d` for both.
    ಸ್ಥಿರಾಂಕ int64_t d = K->ne[0];
    ಆದರೆ (V->ne[0] != d || Q->ne[3] != 1) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    // GQA must divide evenly.
    ಆದರೆ (K->ne[2] == 0 || Q->ne[2] % K->ne[2] != 0) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    // Prefill only.
    ಆದರೆ (use_shape_ಎಲ್ಲೆ && Q->ne[1] < GGML_SYCL_FA_ONEDNN_MIN_Q) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
#ಆದರೆ_ಅಂತ್ಯ
}

#ಆದರೆ GGML_SYCL_DNNL

#ಅನ್ತರ್ಭಾವ "dnnl.hpp"
#ಅನ್ತರ್ಭಾವ "dnnl_sycl.hpp"
#ಅನ್ತರ್ಭಾವ "oneapi/dnnl/dnnl_graph.hpp"   // graph API lives only under oneapi/dnnl/, not at the include root

ಬಳಸಿ ನಾಮಾಕಾಶ dnnl;
ಬಳಸಿ ನಾಮಾಕಾಶ dnnl::graph;

// strided src (f16 or f32) -> contiguous f16 [ne0,ne1,ne2,ne3] (ne0 innermost). nb* are BYTE strides.
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು src_t>
ಸ್ಥಿರ ಶೂನ್ಯ cont_ಗಮ್ಯ_f16_sycl(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * src, sycl::half * dst,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
        size_t nb1, size_t nb2, size_t nb3, dpct::ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ int64_t n = ne0 * ne1 * ne2 * ne3;
    stream->parallel_for(sycl::range<1>(n), [=](sycl::id<1> ix) {
        ಸ್ಥಿರಾಂಕ int64_t gid = ix[0];
        int64_t       i   = gid;
        ಸ್ಥಿರಾಂಕ int64_t i0 = i % ne0; i /= ne0;
        ಸ್ಥಿರಾಂಕ int64_t i1 = i % ne1; i /= ne1;
        ಸ್ಥಿರಾಂಕ int64_t i2 = i % ne2; ಸ್ಥಿರಾಂಕ int64_t i3 = i / ne2;
        ಸ್ಥಿರಾಂಕ src_t * p = (ಸ್ಥಿರಾಂಕ src_t *) (src + i1 * nb1 + i2 * nb2 + i3 * nb3) + i0;
        dst[gid] = (sycl::half) (*p);
    });
}

// oneDNN SDPA out (f16 contiguous [mb,H,q,d]) -> ggml dst (f32 [head_dim,H,n_tok,mb], contiguous).
ಸ್ಥಿರ ಶೂನ್ಯ ಕ್ರಮಬದಲಾವಣೆ_sdpa_ಹೊರಹೋಗು_sycl(ಸ್ಥಿರಾಂಕ sycl::half * ಹೊರಹೋಗು, ದಶಮಾಂಶ * dst,
        int64_t mb, int64_t H, int64_t q, int64_t d, dpct::ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ int64_t n = mb * H * q * d;
    stream->parallel_for(sycl::range<1>(n), [=](sycl::id<1> ix) {
        ಸ್ಥಿರಾಂಕ int64_t gid = ix[0];
        int64_t       i   = gid;
        ಸ್ಥಿರಾಂಕ int64_t e = i % d; i /= d;
        ಸ್ಥಿರಾಂಕ int64_t t = i % q; i /= q;
        ಸ್ಥಿರಾಂಕ int64_t h = i % H; ಸ್ಥಿರಾಂಕ int64_t b = i / H;
        dst[e + h * d + t * d * H + b * d * H * q] = (ದಶಮಾಂಶ) ಹೊರಹೋಗು[gid];
    });
}

ರಚನೆ sdpa_ವಿಭಜನೆ {
    compiled_ವಿಭಜನೆ          cp;
    std::vector<logical_tensor> ins;
    logical_tensor              ಹೊರಹೋಗು;
    size_t id_q = 0, id_k = 0, id_v = 0, id_scale = 0, id_ಮುಖವಾಡ = 0;
    ತಾರ್ಕಿಕ   ok = ಸುಳ್ಳು;
};

// Build + compile the contiguous-input GQA SDPA graph (MatMul->Divide->Add->SoftMax->MatMul), f32 out.
// Mirrors the hardware-verified scratch/onednn_sdpa_probe.cpp build_gqa (partitions=1, sdp_primitive_kernel_t).
ಸ್ಥಿರ sdpa_ವಿಭಜನೆ build_sdpa(ಸ್ಥಿರಾಂಕ engine & eng, ಪೂರ್ಣಾಂಕ H, ಪೂರ್ಣಾಂಕ Hkv, ಪೂರ್ಣಾಂಕ q, ಪೂರ್ಣಾಂಕ seq, ಪೂರ್ಣಾಂಕ d,
                                 ಸ್ಥಿರಾಂಕ std::array<int64_t, 5> & k_str, ಸ್ಥಿರಾಂಕ std::array<int64_t, 5> & v_str) ಪ್ರಯತ್ನಿಸು {
    ಬಳಸಿ ltype = logical_tensor::layout_type;
    ಬಳಸಿ dt    = logical_tensor::ದತ್ತಾಂಶ_type;
    ಬಳಸಿ ldims = logical_tensor::dims;
    ಸ್ಥಿರಾಂಕ dt    fi = dt::f32, t = dt::f16;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ   rep = H / Hkv;
    ಸ್ಥಿರಾಂಕ ldims q_sz = {1, Hkv, rep, q, d}, kv_sz = {1, Hkv, 1, seq, d}, s_sz = {1, Hkv, rep, q, seq},
                sc = {1, 1, 1, 1, 1}, msk = {1, 1, 1, q, seq}, o_sz = {1, Hkv, rep, q, d};
    ಸ್ಥಿರಾಂಕ ldims k_st(k_str.begin(), k_str.end()), v_st(v_str.begin(), v_str.end());
    int64_t        id = 0;
    sdpa_ವಿಭಜನೆ E;

    ಸ್ವಯಂಚಾಲಿತ query  = logical_tensor(id++, t,  q_sz, ltype::strided);
    ಸ್ವಯಂಚಾಲಿತ ಕೀಲಿ    = logical_tensor(id++, t,  kv_sz, k_st);
    ಸ್ವಯಂಚಾಲಿತ score  = logical_tensor(id++, fi, s_sz, ltype::strided);
    ಸ್ವಯಂಚಾಲಿತ bmm1   = op(id++, op::ಬಗೆ::MatMul, "bmm1");
    bmm1.ಗಣ_attr<ತಾರ್ಕಿಕ>(op::attr::transpose_b, ನಿಜ);          // key is [.., seq, d]
    bmm1.ಕೂಡಿಸು_inputs({query, ಕೀಲಿ}); bmm1.ಕೂಡಿಸು_outputs({score});

    ಸ್ವಯಂಚಾಲಿತ scale  = logical_tensor(id++, t,  sc,   ltype::strided);
    ಸ್ವಯಂಚಾಲಿತ scaled = logical_tensor(id++, fi, s_sz, ltype::strided);
    ಸ್ವಯಂಚಾಲಿತ sdiv   = op(id++, op::ಬಗೆ::Divide, "scale_div");     // score / (1/kq_scale) == score * kq_scale
    sdiv.ಕೂಡಿಸು_inputs({score, scale}); sdiv.ಕೂಡಿಸು_outputs({scaled});

    ಸ್ವಯಂಚಾಲಿತ ಮುಖವಾಡ   = logical_tensor(id++, t,  msk,  ltype::strided);
    ಸ್ವಯಂಚಾಲಿತ masked = logical_tensor(id++, fi, s_sz, ltype::strided);
    ಸ್ವಯಂಚಾಲಿತ madd   = op(id++, op::ಬಗೆ::ಕೂಡಿಸು, "mask_add");
    madd.ಕೂಡಿಸು_inputs({scaled, ಮುಖವಾಡ}); madd.ಕೂಡಿಸು_outputs({masked});

    ಸ್ವಯಂಚಾಲಿತ probs  = logical_tensor(id++, t,  s_sz, ltype::strided);
    ಸ್ವಯಂಚಾಲಿತ smax   = op(id++, op::ಬಗೆ::SoftMax, "softmax");
    smax.ಗಣ_attr<int64_t>(op::attr::axis, -1);
    smax.ಗಣ_attr<std::string>(op::attr::mode, "inf_as_zero");
    smax.ಕೂಡಿಸು_inputs({masked}); smax.ಕೂಡಿಸು_outputs({probs});

    ಸ್ವಯಂಚಾಲಿತ value  = logical_tensor(id++, t,  kv_sz, v_st);
    // f16 output is REQUIRED to hit sdp_primitive_kernel_t (the systolic micro-kernel); an f32 output
    // falls to larger_partition_kernel_t which materializes N^2 (confirmed: scratch/onednn_sdpa_kernel_probe.cpp).
    // converted to the f32 ggml dst in the permute below.
    ಸ್ವಯಂಚಾಲಿತ output = logical_tensor(id++, t,  o_sz, ltype::strided);   // f16 contiguous [mb,Hkv,rep,q,d]
    ಸ್ವಯಂಚಾಲಿತ bmm2   = op(id++, op::ಬಗೆ::MatMul, "bmm2");
    bmm2.ಕೂಡಿಸು_inputs({probs, value}); bmm2.ಕೂಡಿಸು_outputs({output});

    dnnl::graph::graph g(eng.get_ಬಗೆ());
    g.ಕೂಡಿಸು_op(bmm1); g.ಕೂಡಿಸು_op(sdiv); g.ಕೂಡಿಸು_op(madd); g.ಕೂಡಿಸು_op(smax); g.ಕೂಡಿಸು_op(bmm2);
    g.finalize();

    ಸ್ವಯಂಚಾಲಿತ parts = g.get_partitions();
    ಆದರೆ (parts.size() != 1 || !parts[0].is_supported()) {
        GGML_LOG_WARN("%s: oneDNN did not fuse the SDPA graph; falling back to TILE kernel\n", __func__);
        ಹಿಂತಿರುಗಿ E;   // ok stays false -> caller falls back to TILE
    }
    E.ins      = parts[0].get_ಒಳಹರಿವು_ports();
    E.ಹೊರಹೋಗು      = parts[0].get_output_ports()[0];
    E.cp       = parts[0].compile(E.ins, {E.ಹೊರಹೋಗು}, eng);
    E.ಹೊರಹೋಗು      = E.cp.query_logical_tensor(E.ಹೊರಹೋಗು.get_id());
    E.id_q     = query.get_id(); E.id_k = ಕೀಲಿ.get_id(); E.id_v = value.get_id();
    E.id_scale = scale.get_id(); E.id_ಮುಖವಾಡ = ಮುಖವಾಡ.get_id();
    E.ok       = ನಿಜ;
    ಹಿಂತಿರುಗಿ E;
}
ಹಿಡಿ (ಸ್ಥಿರಾಂಕ std::exception & e) {
    // compile() can reject a stride set the partitioner never inspects; memoise the failure so the
    // fallback costs one build rather than one per call.
    GGML_LOG_WARN("%s: oneDNN SDPA partition build failed (%s); falling back to TILE kernel\n", __func__, e.what());
    ಹಿಂತಿರುಗಿ {};
}

ಶೂನ್ಯ ggml_sycl_flash_attn_ext_onednn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) ಪ್ರಯತ್ನಿಸು {
    ಸ್ಥಿರಾಂಕ ggml_tensor * Q    = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * K    = dst->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * V    = dst->src[2];
    ಸ್ಥಿರಾಂಕ ggml_tensor * ಮುಖವಾಡ = dst->src[3];

    ಸ್ಥಿರಾಂಕ int64_t d   = K->ne[0];   // head_dim
    ಸ್ಥಿರಾಂಕ int64_t seq = K->ne[1];   // n_kv
    ಸ್ಥಿರಾಂಕ int64_t Hkv = K->ne[2];   // n_head_kv
    ಸ್ಥಿರಾಂಕ int64_t H   = Q->ne[2];   // n_head
    ಸ್ಥಿರಾಂಕ int64_t q   = Q->ne[1];   // n_tok
    ಸ್ಥಿರಾಂಕ int64_t mb  = Q->ne[3];   // batch (== 1, gated)

    ದಶಮಾಂಶ kq_scale = 1.0f;
    ಸ್ಮರಣೆ_ನಕಲು(&kq_scale, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) dst->op_params + 0, ಗಾತ್ರ(ದಶಮಾಂಶ));

    dpct::ಸರತಿ_ptr stream = ctx.stream();
    dnnl::engine    eng    = ctx.engine_dnnl(stream);
    dnnl::stream    strm   = ctx.ಹರಿವು_dnnl(stream);

    ಸ್ಥಿರಾಂಕ ggml_sycl_fattn_extra extra = ggml_sycl_fattn_get_extra(dst);

    // Q: always f32 -- copy to dense f16.
    std::optional<ggml_sycl_pool_alloc<sycl::half>> Qf_pool;
    sycl::half * Qf_ptr = (sycl::half *) extra.Q_ಬಫರ್_ptr;
    ಆದರೆ (!Qf_ptr) {
        Qf_pool.emplace(ctx.pool(), (size_t) H * q * d);
        Qf_ptr = Qf_pool->get();
    }
    cont_ಗಮ್ಯ_f16_sycl<ದಶಮಾಂಶ>((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) Q->ದತ್ತಾಂಶ, Qf_ptr, d, q, H, mb, Q->nb[1], Q->nb[2], Q->nb[3], stream);

    // K/V: bind the f16 cache in place. llama.cpp permutes it to [token][head][dim], so its head
    // plane is strided rather than dense, which is what an explicit stride vector expresses.
    // Quantized and f32 KV still stage a dense copy -- the layout the k_str/v_str defaults describe.
    sycl::half * K_ptr = ಶೂನ್ಯಸೂಚಕ;
    sycl::half * V_ptr = ಶೂನ್ಯಸೂಚಕ;
    std::array<int64_t, 5> k_str{ Hkv * seq * d, seq * d, seq * d, d, 1 };
    std::array<int64_t, 5> v_str = k_str;
    std::optional<ggml_sycl_pool_alloc<sycl::half>> Kf_pool;
    std::optional<ggml_sycl_pool_alloc<sycl::half>> Vf_pool;
    // Helper: hand out reserved space, or fall back to the pool.
    ಸ್ವಯಂಚಾಲಿತ stage_k = [&](size_t n) { ಆದರೆ (extra.K_ಬಫರ್_ptr) { ಹಿಂತಿರುಗಿ (sycl::half *) extra.K_ಬಫರ್_ptr; }
                                  Kf_pool.emplace(ctx.pool(), n); ಹಿಂತಿರುಗಿ Kf_pool->get(); };
    ಸ್ವಯಂಚಾಲಿತ stage_v = [&](size_t n) { ಆದರೆ (extra.V_ಬಫರ್_ptr) { ಹಿಂತಿರುಗಿ (sycl::half *) extra.V_ಬಫರ್_ptr; }
                                  Vf_pool.emplace(ctx.pool(), n); ಹಿಂತಿರುಗಿ Vf_pool->get(); };

    ಸ್ವಯಂಚಾಲಿತ elem_strides = [](ಸ್ಥಿರಾಂಕ ggml_tensor * t) {
        ಸ್ಥಿರಾಂಕ int64_t s1 = (int64_t) (t->nb[1] / t->nb[0]);
        ಸ್ಥಿರಾಂಕ int64_t s2 = (int64_t) (t->nb[2] / t->nb[0]);
        ಸ್ಥಿರಾಂಕ int64_t s3 = (int64_t) (t->nb[3] / t->nb[0]);
        // dims are {mb=1, Hkv, rep=1, seq, d}; the size-1 dims at 0 and 2 never advance an address.
        ಹಿಂತಿರುಗಿ std::array<int64_t, 5>{ s3, s2, s2, s1, 1 };
    };

    ಆದರೆ (ggml_sycl_fattn_onednn_binds_kv(K, V)) {
        K_ptr = (sycl::half *) K->ದತ್ತಾಂಶ;
        V_ptr = (sycl::half *) V->ದತ್ತಾಂಶ;
        k_str = elem_strides(K);
        v_str = elem_strides(V);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16) {
        K_ptr = stage_k((size_t) Hkv * seq * d);
        V_ptr = stage_v((size_t) Hkv * seq * d);
        cont_ಗಮ್ಯ_f16_sycl<sycl::half>((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) K->ದತ್ತಾಂಶ, K_ptr, d, seq, Hkv, mb, K->nb[1], K->nb[2], K->nb[3], stream);
        cont_ಗಮ್ಯ_f16_sycl<sycl::half>((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) V->ದತ್ತಾಂಶ, V_ptr, d, seq, Hkv, mb, V->nb[1], V->nb[2], V->nb[3], stream);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ggml_is_quantized(K->type)) {
        // Quantized K/V: dequant to dense F16 using pool, same lifetime as F16 path.
        K_ptr = stage_k((size_t) ggml_nelements(K));
        {
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * K_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)K->ದತ್ತಾಂಶ;
            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ k_non_dense = ((int64_t)K->ne[1] * K->nb[1] != K->nb[2]) && K->ne[2] > 1;
            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ k_gemma = k_non_dense &&
                ((int64_t)K->nb[2] < (int64_t)K->ne[1] * (int64_t)K->nb[1]);
            ಆದರೆ (ggml_is_contiguously_allocated(K) && !k_non_dense) {
                ಗಮ್ಯ_fp16_sycl_t ಗಮ್ಯ_fp16 = ggml_get_ಗಮ್ಯ_fp16_sycl(K->type, dst);
                ಗಮ್ಯ_fp16(K_ದತ್ತಾಂಶ, K_ptr, ggml_nelements(K), stream);
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಸ್ಥಿರಾಂಕ size_t bs = ggml_blck_ಗಾತ್ರ(K->type);
                ಸ್ಥಿರಾಂಕ size_t ts = ggml_type_ಗಾತ್ರ(K->type);
                ಗಮ್ಯ_fp16_nc_sycl_t ಗಮ್ಯ_fp16 = ggml_get_ಗಮ್ಯ_fp16_nc_sycl(K->type);
                int64_t s01, s02, s03;
                ಆದರೆ (k_gemma) {
                    ಸ್ಥಿರಾಂಕ int64_t blk_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು = (int64_t)K->ne[0] / bs;
                    s01 = (int64_t)Hkv * blk_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
                    s02 = blk_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
                    s03 = (int64_t)K->ne[1] * s01;
                } ಇಲ್ಲದಿದ್ದರೆ {
                    s01 = (int64_t)K->nb[1] / ts;
                    s02 = (int64_t)K->nb[2] / ts;
                    s03 = (int64_t)K->nb[3] / ts;
                }
                ಗಮ್ಯ_fp16(K_ದತ್ತಾಂಶ, K_ptr,
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
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * V_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)V->ದತ್ತಾಂಶ;
            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ v_non_dense = ((int64_t)V->ne[1] * V->nb[1] != V->nb[2]) && V->ne[2] > 1;
            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ v_gemma = v_non_dense &&
                ((int64_t)V->nb[2] < (int64_t)V->ne[1] * (int64_t)V->nb[1]);
            ಆದರೆ (ggml_is_contiguously_allocated(V) && !v_non_dense) {
                ಗಮ್ಯ_fp16_sycl_t ಗಮ್ಯ_fp16 = ggml_get_ಗಮ್ಯ_fp16_sycl(V->type, dst);
                ಗಮ್ಯ_fp16(V_ದತ್ತಾಂಶ, V_ptr, ggml_nelements(V), stream);
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಸ್ಥಿರಾಂಕ size_t bs = ggml_blck_ಗಾತ್ರ(V->type);
                ಸ್ಥಿರಾಂಕ size_t ts = ggml_type_ಗಾತ್ರ(V->type);
                ಗಮ್ಯ_fp16_nc_sycl_t ಗಮ್ಯ_fp16 = ggml_get_ಗಮ್ಯ_fp16_nc_sycl(V->type);
                int64_t s01, s02, s03;
                ಆದರೆ (v_gemma) {
                    ಸ್ಥಿರಾಂಕ int64_t blk_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು = (int64_t)V->ne[0] / bs;
                    s01 = (int64_t)V->ne[2] * blk_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
                    s02 = blk_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
                    s03 = (int64_t)V->ne[1] * s01;
                } ಇಲ್ಲದಿದ್ದರೆ {
                    s01 = (int64_t)V->nb[1] / ts;
                    s02 = (int64_t)V->nb[2] / ts;
                    s03 = (int64_t)V->nb[3] / ts;
                }
                ಗಮ್ಯ_fp16(V_ದತ್ತಾಂಶ, V_ptr,
                        V->ne[0], V->ne[1], V->ne[2], V->ne[3],
                        s01, s02, s03, stream);
            }
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        // F32: strided copy to dense F16 via cont_to_f16_sycl<float>.
        K_ptr = stage_k((size_t) ggml_nelements(K));
        cont_ಗಮ್ಯ_f16_sycl<ದಶಮಾಂಶ>((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) K->ದತ್ತಾಂಶ, K_ptr, K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                                K->nb[1], K->nb[2], K->nb[3], stream);
        V_ptr = stage_v((size_t) ggml_nelements(V));
        cont_ಗಮ್ಯ_f16_sycl<ದಶಮಾಂಶ>((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) V->ದತ್ತಾಂಶ, V_ptr, V->ne[0], V->ne[1], V->ne[2], V->ne[3],
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
    ಸ್ಥಿರಾಂಕ sycl::half scale_h = (sycl::half) (1.0f / kq_scale);
    std::optional<ggml_sycl_pool_alloc<sycl::half>> scbuf;
    sycl::half * scale_dev = (sycl::half *) extra.scale_ಬಫರ್_ptr;
    ಆದರೆ (!scale_dev) {
        scbuf.emplace(ctx.pool(), 1);
        scale_dev = scbuf->get();
    }
    stream->single_task([=]() { *scale_dev = scale_h; });

    // f16 contiguous SDPA out [mb,H,q,d]
    std::optional<ggml_sycl_pool_alloc<sycl::half>> outf_pool;
    sycl::half * outf_ptr = (sycl::half *) extra.ಹೊರಹೋಗು_ಬಫರ್_ptr;
    ಆದರೆ (!outf_ptr) {
        outf_pool.emplace(ctx.pool(), (size_t) H * q * d);
        outf_ptr = outf_pool->get();
    }

    // compile once per (device, shape, KV strides), reuse across layers/calls. Stride 2 always
    // repeats stride 1 and stride 4 is always 1, so the key covers every entry that can differ.
    ಸ್ಥಿರ std::unordered_map<std::string, sdpa_ವಿಭಜನೆ> cache;
    ಅಕ್ಷರ keyb[256];
    snprintf(keyb, ಗಾತ್ರ(keyb), "%d:%lld:%lld:%lld:%lld:%lld:%lld:%lld:%lld:%lld:%lld:%lld", ggml_sycl_get_device(),
             (ದೀರ್ಘ ದೀರ್ಘ) H, (ದೀರ್ಘ ದೀರ್ಘ) Hkv, (ದೀರ್ಘ ದೀರ್ಘ) q, (ದೀರ್ಘ ದೀರ್ಘ) seq, (ದೀರ್ಘ ದೀರ್ಘ) d,
             (ದೀರ್ಘ ದೀರ್ಘ) k_str[0], (ದೀರ್ಘ ದೀರ್ಘ) k_str[1], (ದೀರ್ಘ ದೀರ್ಘ) k_str[3],
             (ದೀರ್ಘ ದೀರ್ಘ) v_str[0], (ದೀರ್ಘ ದೀರ್ಘ) v_str[1], (ದೀರ್ಘ ದೀರ್ಘ) v_str[3]);
    ಸ್ವಯಂಚಾಲಿತ it = cache.find(keyb);
    ಆದರೆ (it == cache.end()) {
        it = cache.emplace(keyb, build_sdpa(eng, (ಪೂರ್ಣಾಂಕ) H, (ಪೂರ್ಣಾಂಕ) Hkv, (ಪೂರ್ಣಾಂಕ) q, (ಪೂರ್ಣಾಂಕ) seq, (ಪೂರ್ಣಾಂಕ) d, k_str, v_str)).ಮೊದಲ;
    }
    sdpa_ವಿಭಜನೆ & E = it->second;
    ಆದರೆ (!E.ok) {
        // oneDNN can decline a shape or a stride set that _supported() never sees; build_sdpa warns per key.
        ggml_sycl_flash_attn_ext_tile(ctx, dst);
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ವಯಂಚಾಲಿತ id2ptr = [&](size_t r) -> ಶೂನ್ಯ * {
        ಆದರೆ (r == E.id_q)     ಹಿಂತಿರುಗಿ Qf_ptr;
        ಆದರೆ (r == E.id_k)     ಹಿಂತಿರುಗಿ K_ptr;
        ಆದರೆ (r == E.id_v)     ಹಿಂತಿರುಗಿ V_ptr;
        ಆದರೆ (r == E.id_scale) ಹಿಂತಿರುಗಿ scale_dev;
        ಆದರೆ (r == E.id_ಮುಖವಾಡ)  ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *) ಮುಖವಾಡ->ದತ್ತಾಂಶ;
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    };
    std::vector<tensor> ti;
    ti.reserve(E.ins.size());
    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ & lt : E.ins) {
        ti.emplace_back(lt, eng, id2ptr(lt.get_id()));
    }
    tensor ಗಮ್ಯ(E.ಹೊರಹೋಗು, eng, outf_ptr);
    E.cp.execute(strm, ti, {ಗಮ್ಯ});

    ಕ್ರಮಬದಲಾವಣೆ_sdpa_ಹೊರಹೋಗು_sycl(outf_ptr, (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ, mb, H, q, d, stream);
    // Single device needs no sync: the dnnl stream wraps this same in-order queue, so the SDPA
    // serializes with the staging kernels before it and the permute/pool reuse after it. The
    // garbage output formerly blamed on the missing sync here was the scale use-after-return
    // fixed above. Keep the conservative wait for multi-GPU, where other devices' streams can
    // race the pool:
    ಆದರೆ (ggml_sycl_info().device_ಎಣಿಕೆ > 1) {
        stream->wait_and_throw();
    }
}
ಹಿಡಿ (ಸ್ಥಿರಾಂಕ std::exception & e) {
    // any oneDNN/SYCL failure is non-fatal: fall back to the existing kernel (strictly additive).
    GGML_LOG_WARN("%s: oneDNN SDPA failed (%s); falling back to TILE kernel\n", __func__, e.what());
    ggml_sycl_flash_attn_ext_tile(ctx, dst);
}

#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_DNNL
