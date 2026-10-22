#অন্তর্ভাব <array>
#অন্তর্ভাব <cstdint>
#অন্তর্ভাব <cstdio>
#অন্তর্ভাব <cstring>
#অন্তর্ভাব <string>
#অন্তর্ভাব <optional>
#অন্তর্ভাব <unordered_map>
#অন্তর্ভাব <vector>

#অন্তর্ভাব "fattn-onednn.hpp"
#অন্তর্ভাব "fattn-tile.hpp"
#অন্তর্ভাব "convert.hpp"

// set minimum query length to treat as prefill (32)
#define GGML_SYCL_FA_ONEDNN_MIN_Q 32

যৌক্তিক ggml_sycl_fattn_onednn_binds_kv(ধ্রুবক ggml_tensor * K, ধ্রুবক ggml_tensor * V) {
    যদি (K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16) {
        ফেরত মিথ্যা;
    }
    auto bindable = [](ধ্রুবক ggml_tensor * t) {
        ফেরত t->nb[0] == আকার(sycl::half) && t->nb[1] % আকার(sycl::half) == 0 &&
               t->nb[2] % আকার(sycl::half) == 0 && t->nb[3] % আকার(sycl::half) == 0;
    };
    ফেরত bindable(K) && bindable(V);
}

যৌক্তিক ggml_sycl_flash_attn_ext_onednn_supported(ধ্রুবক ggml_tensor * dst, যৌক্তিক use_shape_limit) {
#যদি !GGML_SYCL_DNNL
    GGML_UNUSED(dst);
    GGML_UNUSED(use_shape_limit);
    ফেরত মিথ্যা;
#অন্যথা
    যদি (!g_ggml_sycl_fa_onednn) {
        ফেরত মিথ্যা;
    }
    // without a fused SDPA kernel oneDNN falls back to the same matmul, so reuse the check
    যদি (!ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
        ফেরত মিথ্যা;
    }
    ধ্রুবক ggml_tensor * Q     = dst->src[0];
    ধ্রুবক ggml_tensor * K     = dst->src[1];
    ধ্রুবক ggml_tensor * V     = dst->src[2];
    ধ্রুবক ggml_tensor * mask  = dst->src[3];
    ধ্রুবক ggml_tensor * sinks = dst->src[4];

    // F16 KV: native SDPA at any KV length.
    // Non-F16: dequant to F16 then SDPA at prefill lengths. Only the
    // standard quantized KV cache types (Q4_0-Q8_0) and F32 are accepted
    // because their to_fp16_sycl conversion is verified. BF16 and IQ*
    // are excluded: BF16 needs a strided conversion kernel that does not
    // exist yet; IQ types are model-weight-only quants with no dequant
    // registration and are never used as KV caches.
    যদি (K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16) {
        auto kt = K->type, vt = V->type;
        যৌক্তিক k_ok = kt == GGML_TYPE_F32 || kt == GGML_TYPE_Q4_0 || kt == GGML_TYPE_Q4_1 ||
                    kt == GGML_TYPE_Q5_0 || kt == GGML_TYPE_Q5_1 || kt == GGML_TYPE_Q8_0;
        যৌক্তিক v_ok = vt == GGML_TYPE_F32 || vt == GGML_TYPE_Q4_0 || vt == GGML_TYPE_Q4_1 ||
                    vt == GGML_TYPE_Q5_0 || vt == GGML_TYPE_Q5_1 || vt == GGML_TYPE_Q8_0;
        যদি (!k_ok || !v_ok) {
            ফেরত মিথ্যা;
        }
        যদি (use_shape_limit && (Q->ne[1] < 32 || K->ne[1] < 1024)) {
            ফেরত মিথ্যা;
        }
        জন্য (ধ্রুবক ggml_tensor * t : {K, V}) {
            যদি (t->type == GGML_TYPE_F16 && t->nb[1] % (t->ne[0] * 2) != 0) {
                ফেরত মিথ্যা;
            }
        }
    }
    // This is the improved SPDA gate. Rather than gating Alchemist GPUs from all SPDA features, we instead target only the failing shapes.
    // If the GPU being assessed isn't in the grouping below, it has full access to all SPDA shapes. Otherwise, if it's an Alchemist GPU, we block only the shapes with head sizes that fail.
    // It is much easier to compare the device to a small list of failing cases than to define all the passing ones.
    ধ্রুবক gpu_arch arch = ggml_sycl_info().devices[ggml_sycl_get_device()].hw_info.arch;
    যৌক্তিক support_spda = !(arch == gpu_arch::intel_gpu_dg2_g10 ||
                         arch == gpu_arch::intel_gpu_dg2_g11 ||
                         arch == gpu_arch::intel_gpu_dg2_g12);

    যদি (!support_spda && K->ne[0] == 64) {
        ফেরত মিথ্যা;
    }
    // Optional KV-length ceiling (GGML_SYCL_FA_ONEDNN_MAX_KV, 0 = unlimited). Escape hatch:
    // very long sequences make the fused SDPA slow enough to risk the xe driver watchdog on
    // some stacks; past the cap we fall back to the native FA kernel instead.
    যদি (g_ggml_sycl_fa_onednn_max_kv > 0 && K->ne[1] > g_ggml_sycl_fa_onednn_max_kv) {
        ফেরত মিথ্যা;
    }
    // gate for the following cases
    // 1. if the oneDNN graph Add node has no input --> skip
    // 2. types other than f16 need different logical_tensor declaration
    // 3. the mask must be shape [1, 1, q, seq]
    // 4. sinks: excludes attention sink (Xiao et al., 2024) that can't be modeled by oneDNN graph
    যদি (!mask || mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || mask->ne[3] != 1 || sinks) {
        ফেরত মিথ্যা;
    }
    দশমিক max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&max_bias,      (ধ্রুবক দশমিক *) dst->op_params + 1, আকার(দশমিক));
    memcpy(&logit_softcap, (ধ্রুবক দশমিক *) dst->op_params + 2, আকার(দশমিক));
    যদি (max_bias != 0.0f || logit_softcap != 0.0f) {
        ফেরত মিথ্যা;
    }
    // K and V must share head_dim: the SDPA graph uses a single `d` for both.
    ধ্রুবক int64_t d = K->ne[0];
    যদি (V->ne[0] != d || Q->ne[3] != 1) {
        ফেরত মিথ্যা;
    }
    // GQA must divide evenly.
    যদি (K->ne[2] == 0 || Q->ne[2] % K->ne[2] != 0) {
        ফেরত মিথ্যা;
    }
    // Prefill only.
    যদি (use_shape_limit && Q->ne[1] < GGML_SYCL_FA_ONEDNN_MIN_Q) {
        ফেরত মিথ্যা;
    }
    ফেরত সত্য;
#endif
}

#যদি GGML_SYCL_DNNL

#অন্তর্ভাব "dnnl.hpp"
#অন্তর্ভাব "dnnl_sycl.hpp"
#অন্তর্ভাব "oneapi/dnnl/dnnl_graph.hpp"   // graph API lives only under oneapi/dnnl/, not at the include root

using নামাকাশ dnnl;
using নামাকাশ dnnl::graph;

// strided src (f16 or f32) -> contiguous f16 [ne0,ne1,ne2,ne3] (ne0 innermost). nb* are BYTE strides.
template <typename src_t>
স্থির শূন্য cont_to_f16_sycl(ধ্রুবক অক্ষর * src, sycl::half * dst,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
        size_t nb1, size_t nb2, size_t nb3, dpct::queue_ptr stream) {
    ধ্রুবক int64_t n = ne0 * ne1 * ne2 * ne3;
    stream->parallel_for(sycl::range<1>(n), [=](sycl::id<1> ix) {
        ধ্রুবক int64_t gid = ix[0];
        int64_t       i   = gid;
        ধ্রুবক int64_t i0 = i % ne0; i /= ne0;
        ধ্রুবক int64_t i1 = i % ne1; i /= ne1;
        ধ্রুবক int64_t i2 = i % ne2; ধ্রুবক int64_t i3 = i / ne2;
        ধ্রুবক src_t * p = (ধ্রুবক src_t *) (src + i1 * nb1 + i2 * nb2 + i3 * nb3) + i0;
        dst[gid] = (sycl::half) (*p);
    });
}

// oneDNN SDPA out (f16 contiguous [mb,H,q,d]) -> ggml dst (f32 [head_dim,H,n_tok,mb], contiguous).
স্থির শূন্য permute_sdpa_out_sycl(ধ্রুবক sycl::half * out, দশমিক * dst,
        int64_t mb, int64_t H, int64_t q, int64_t d, dpct::queue_ptr stream) {
    ধ্রুবক int64_t n = mb * H * q * d;
    stream->parallel_for(sycl::range<1>(n), [=](sycl::id<1> ix) {
        ধ্রুবক int64_t gid = ix[0];
        int64_t       i   = gid;
        ধ্রুবক int64_t e = i % d; i /= d;
        ধ্রুবক int64_t t = i % q; i /= q;
        ধ্রুবক int64_t h = i % H; ধ্রুবক int64_t b = i / H;
        dst[e + h * d + t * d * H + b * d * H * q] = (দশমিক) out[gid];
    });
}

কাঠামো sdpa_partition {
    compiled_partition          cp;
    std::vector<logical_tensor> ins;
    logical_tensor              out;
    size_t id_q = 0, id_k = 0, id_v = 0, id_scale = 0, id_mask = 0;
    যৌক্তিক   ok = মিথ্যা;
};

// Build + compile the contiguous-input GQA SDPA graph (MatMul->Divide->Add->SoftMax->MatMul), f32 out.
// Mirrors the hardware-verified scratch/onednn_sdpa_probe.cpp build_gqa (partitions=1, sdp_primitive_kernel_t).
স্থির sdpa_partition build_sdpa(ধ্রুবক engine & eng, পূর্ণসংখ্যা H, পূর্ণসংখ্যা Hkv, পূর্ণসংখ্যা q, পূর্ণসংখ্যা seq, পূর্ণসংখ্যা d,
                                 ধ্রুবক std::array<int64_t, 5> & k_str, ধ্রুবক std::array<int64_t, 5> & v_str) চেষ্টা {
    using ltype = logical_tensor::layout_type;
    using dt    = logical_tensor::তথ্য_type;
    using ldims = logical_tensor::dims;
    ধ্রুবক dt    fi = dt::f32, t = dt::f16;
    ধ্রুবক পূর্ণসংখ্যা   rep = H / Hkv;
    ধ্রুবক ldims q_sz = {1, Hkv, rep, q, d}, kv_sz = {1, Hkv, 1, seq, d}, s_sz = {1, Hkv, rep, q, seq},
                sc = {1, 1, 1, 1, 1}, msk = {1, 1, 1, q, seq}, o_sz = {1, Hkv, rep, q, d};
    ধ্রুবক ldims k_st(k_str.begin(), k_str.end()), v_st(v_str.begin(), v_str.end());
    int64_t        id = 0;
    sdpa_partition E;

    auto query  = logical_tensor(id++, t,  q_sz, ltype::strided);
    auto চাবি    = logical_tensor(id++, t,  kv_sz, k_st);
    auto score  = logical_tensor(id++, fi, s_sz, ltype::strided);
    auto bmm1   = op(id++, op::kind::MatMul, "bmm1");
    bmm1.set_attr<যৌক্তিক>(op::attr::transpose_b, সত্য);          // key is [.., seq, d]
    bmm1.add_inputs({query, চাবি}); bmm1.add_outputs({score});

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
    যদি (parts.size() != 1 || !parts[0].is_supported()) {
        GGML_LOG_WARN("%s: oneDNN did not fuse the SDPA graph; falling back to TILE kernel\n", __func__);
        ফেরত E;   // ok stays false -> caller falls back to TILE
    }
    E.ins      = parts[0].get_প্রবেশ_ports();
    E.out      = parts[0].get_output_ports()[0];
    E.cp       = parts[0].compile(E.ins, {E.out}, eng);
    E.out      = E.cp.query_logical_tensor(E.out.get_id());
    E.id_q     = query.get_id(); E.id_k = চাবি.get_id(); E.id_v = value.get_id();
    E.id_scale = scale.get_id(); E.id_mask = mask.get_id();
    E.ok       = সত্য;
    ফেরত E;
}
ধরো (ধ্রুবক std::exception & e) {
    // compile() can reject a stride set the partitioner never inspects; memoise the failure so the
    // fallback costs one build rather than one per call.
    GGML_LOG_WARN("%s: oneDNN SDPA partition build failed (%s); falling back to TILE kernel\n", __func__, e.what());
    ফেরত {};
}

শূন্য ggml_sycl_flash_attn_ext_onednn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) চেষ্টা {
    ধ্রুবক ggml_tensor * Q    = dst->src[0];
    ধ্রুবক ggml_tensor * K    = dst->src[1];
    ধ্রুবক ggml_tensor * V    = dst->src[2];
    ধ্রুবক ggml_tensor * mask = dst->src[3];

    ধ্রুবক int64_t d   = K->ne[0];   // head_dim
    ধ্রুবক int64_t seq = K->ne[1];   // n_kv
    ধ্রুবক int64_t Hkv = K->ne[2];   // n_head_kv
    ধ্রুবক int64_t H   = Q->ne[2];   // n_head
    ধ্রুবক int64_t q   = Q->ne[1];   // n_tok
    ধ্রুবক int64_t mb  = Q->ne[3];   // batch (== 1, gated)

    দশমিক kq_scale = 1.0f;
    memcpy(&kq_scale, (ধ্রুবক দশমিক *) dst->op_params + 0, আকার(দশমিক));

    dpct::queue_ptr stream = ctx.stream();
    dnnl::engine    eng    = ctx.engine_dnnl(stream);
    dnnl::stream    strm   = ctx.stream_dnnl(stream);

    ধ্রুবক ggml_sycl_fattn_extra extra = ggml_sycl_fattn_get_extra(dst);

    // Q: always f32 -- copy to dense f16.
    std::optional<ggml_sycl_pool_alloc<sycl::half>> Qf_pool;
    sycl::half * Qf_ptr = (sycl::half *) extra.Q_বাফার_ptr;
    যদি (!Qf_ptr) {
        Qf_pool.emplace(ctx.pool(), (size_t) H * q * d);
        Qf_ptr = Qf_pool->get();
    }
    cont_to_f16_sycl<দশমিক>((ধ্রুবক অক্ষর *) Q->তথ্য, Qf_ptr, d, q, H, mb, Q->nb[1], Q->nb[2], Q->nb[3], stream);

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
    auto stage_k = [&](size_t n) { যদি (extra.K_বাফার_ptr) { ফেরত (sycl::half *) extra.K_বাফার_ptr; }
                                  Kf_pool.emplace(ctx.pool(), n); ফেরত Kf_pool->get(); };
    auto stage_v = [&](size_t n) { যদি (extra.V_বাফার_ptr) { ফেরত (sycl::half *) extra.V_বাফার_ptr; }
                                  Vf_pool.emplace(ctx.pool(), n); ফেরত Vf_pool->get(); };

    auto elem_strides = [](ধ্রুবক ggml_tensor * t) {
        ধ্রুবক int64_t s1 = (int64_t) (t->nb[1] / t->nb[0]);
        ধ্রুবক int64_t s2 = (int64_t) (t->nb[2] / t->nb[0]);
        ধ্রুবক int64_t s3 = (int64_t) (t->nb[3] / t->nb[0]);
        // dims are {mb=1, Hkv, rep=1, seq, d}; the size-1 dims at 0 and 2 never advance an address.
        ফেরত std::array<int64_t, 5>{ s3, s2, s2, s1, 1 };
    };

    যদি (ggml_sycl_fattn_onednn_binds_kv(K, V)) {
        K_ptr = (sycl::half *) K->তথ্য;
        V_ptr = (sycl::half *) V->তথ্য;
        k_str = elem_strides(K);
        v_str = elem_strides(V);
    } অন্যথা যদি (K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16) {
        K_ptr = stage_k((size_t) Hkv * seq * d);
        V_ptr = stage_v((size_t) Hkv * seq * d);
        cont_to_f16_sycl<sycl::half>((ধ্রুবক অক্ষর *) K->তথ্য, K_ptr, d, seq, Hkv, mb, K->nb[1], K->nb[2], K->nb[3], stream);
        cont_to_f16_sycl<sycl::half>((ধ্রুবক অক্ষর *) V->তথ্য, V_ptr, d, seq, Hkv, mb, V->nb[1], V->nb[2], V->nb[3], stream);
    } অন্যথা যদি (ggml_is_quantized(K->type)) {
        // Quantized K/V: dequant to dense F16 using pool, same lifetime as F16 path.
        K_ptr = stage_k((size_t) ggml_nelements(K));
        {
            ধ্রুবক অক্ষর * K_তথ্য = (ধ্রুবক অক্ষর *)K->তথ্য;
            ধ্রুবক যৌক্তিক k_non_dense = ((int64_t)K->ne[1] * K->nb[1] != K->nb[2]) && K->ne[2] > 1;
            ধ্রুবক যৌক্তিক k_gemma = k_non_dense &&
                ((int64_t)K->nb[2] < (int64_t)K->ne[1] * (int64_t)K->nb[1]);
            যদি (ggml_is_contiguously_allocated(K) && !k_non_dense) {
                to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(K->type, dst);
                to_fp16(K_তথ্য, K_ptr, ggml_nelements(K), stream);
            } অন্যথা {
                ধ্রুবক size_t bs = ggml_blck_আকার(K->type);
                ধ্রুবক size_t ts = ggml_type_আকার(K->type);
                to_fp16_nc_sycl_t to_fp16 = ggml_get_to_fp16_nc_sycl(K->type);
                int64_t s01, s02, s03;
                যদি (k_gemma) {
                    ধ্রুবক int64_t blk_per_row = (int64_t)K->ne[0] / bs;
                    s01 = (int64_t)Hkv * blk_per_row;
                    s02 = blk_per_row;
                    s03 = (int64_t)K->ne[1] * s01;
                } অন্যথা {
                    s01 = (int64_t)K->nb[1] / ts;
                    s02 = (int64_t)K->nb[2] / ts;
                    s03 = (int64_t)K->nb[3] / ts;
                }
                to_fp16(K_তথ্য, K_ptr,
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
            ধ্রুবক অক্ষর * V_তথ্য = (ধ্রুবক অক্ষর *)V->তথ্য;
            ধ্রুবক যৌক্তিক v_non_dense = ((int64_t)V->ne[1] * V->nb[1] != V->nb[2]) && V->ne[2] > 1;
            ধ্রুবক যৌক্তিক v_gemma = v_non_dense &&
                ((int64_t)V->nb[2] < (int64_t)V->ne[1] * (int64_t)V->nb[1]);
            যদি (ggml_is_contiguously_allocated(V) && !v_non_dense) {
                to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(V->type, dst);
                to_fp16(V_তথ্য, V_ptr, ggml_nelements(V), stream);
            } অন্যথা {
                ধ্রুবক size_t bs = ggml_blck_আকার(V->type);
                ধ্রুবক size_t ts = ggml_type_আকার(V->type);
                to_fp16_nc_sycl_t to_fp16 = ggml_get_to_fp16_nc_sycl(V->type);
                int64_t s01, s02, s03;
                যদি (v_gemma) {
                    ধ্রুবক int64_t blk_per_row = (int64_t)V->ne[0] / bs;
                    s01 = (int64_t)V->ne[2] * blk_per_row;
                    s02 = blk_per_row;
                    s03 = (int64_t)V->ne[1] * s01;
                } অন্যথা {
                    s01 = (int64_t)V->nb[1] / ts;
                    s02 = (int64_t)V->nb[2] / ts;
                    s03 = (int64_t)V->nb[3] / ts;
                }
                to_fp16(V_তথ্য, V_ptr,
                        V->ne[0], V->ne[1], V->ne[2], V->ne[3],
                        s01, s02, s03, stream);
            }
        }
    } অন্যথা {
        // F32: strided copy to dense F16 via cont_to_f16_sycl<float>.
        K_ptr = stage_k((size_t) ggml_nelements(K));
        cont_to_f16_sycl<দশমিক>((ধ্রুবক অক্ষর *) K->তথ্য, K_ptr, K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                                K->nb[1], K->nb[2], K->nb[3], stream);
        V_ptr = stage_v((size_t) ggml_nelements(V));
        cont_to_f16_sycl<দশমিক>((ধ্রুবক অক্ষর *) V->তথ্য, V_ptr, V->ne[0], V->ne[1], V->ne[2], V->ne[3],
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
    ধ্রুবক sycl::half scale_h = (sycl::half) (1.0f / kq_scale);
    std::optional<ggml_sycl_pool_alloc<sycl::half>> scbuf;
    sycl::half * scale_dev = (sycl::half *) extra.scale_বাফার_ptr;
    যদি (!scale_dev) {
        scbuf.emplace(ctx.pool(), 1);
        scale_dev = scbuf->get();
    }
    stream->single_task([=]() { *scale_dev = scale_h; });

    // f16 contiguous SDPA out [mb,H,q,d]
    std::optional<ggml_sycl_pool_alloc<sycl::half>> outf_pool;
    sycl::half * outf_ptr = (sycl::half *) extra.out_বাফার_ptr;
    যদি (!outf_ptr) {
        outf_pool.emplace(ctx.pool(), (size_t) H * q * d);
        outf_ptr = outf_pool->get();
    }

    // compile once per (device, shape, KV strides), reuse across layers/calls. Stride 2 always
    // repeats stride 1 and stride 4 is always 1, so the key covers every entry that can differ.
    স্থির std::unordered_map<std::string, sdpa_partition> cache;
    অক্ষর keyb[256];
    snprintf(keyb, আকার(keyb), "%d:%lld:%lld:%lld:%lld:%lld:%lld:%lld:%lld:%lld:%lld:%lld", ggml_sycl_get_device(),
             (long long) H, (long long) Hkv, (long long) q, (long long) seq, (long long) d,
             (long long) k_str[0], (long long) k_str[1], (long long) k_str[3],
             (long long) v_str[0], (long long) v_str[1], (long long) v_str[3]);
    auto it = cache.find(keyb);
    যদি (it == cache.end()) {
        it = cache.emplace(keyb, build_sdpa(eng, (পূর্ণসংখ্যা) H, (পূর্ণসংখ্যা) Hkv, (পূর্ণসংখ্যা) q, (পূর্ণসংখ্যা) seq, (পূর্ণসংখ্যা) d, k_str, v_str)).first;
    }
    sdpa_partition & E = it->second;
    যদি (!E.ok) {
        // oneDNN can decline a shape or a stride set that _supported() never sees; build_sdpa warns per key.
        ggml_sycl_flash_attn_ext_tile(ctx, dst);
        ফেরত;
    }

    auto id2ptr = [&](size_t r) -> শূন্য * {
        যদি (r == E.id_q)     ফেরত Qf_ptr;
        যদি (r == E.id_k)     ফেরত K_ptr;
        যদি (r == E.id_v)     ফেরত V_ptr;
        যদি (r == E.id_scale) ফেরত scale_dev;
        যদি (r == E.id_mask)  ফেরত (শূন্য *) mask->তথ্য;
        ফেরত nullptr;
    };
    std::vector<tensor> ti;
    ti.reserve(E.ins.size());
    জন্য (auto & lt : E.ins) {
        ti.emplace_back(lt, eng, id2ptr(lt.get_id()));
    }
    tensor to(E.out, eng, outf_ptr);
    E.cp.execute(strm, ti, {to});

    permute_sdpa_out_sycl(outf_ptr, (দশমিক *) dst->তথ্য, mb, H, q, d, stream);
    // Single device needs no sync: the dnnl stream wraps this same in-order queue, so the SDPA
    // serializes with the staging kernels before it and the permute/pool reuse after it. The
    // garbage output formerly blamed on the missing sync here was the scale use-after-return
    // fixed above. Keep the conservative wait for multi-GPU, where other devices' streams can
    // race the pool:
    যদি (ggml_sycl_info().device_গুনতি > 1) {
        stream->wait_and_throw();
    }
}
ধরো (ধ্রুবক std::exception & e) {
    // any oneDNN/SYCL failure is non-fatal: fall back to the existing kernel (strictly additive).
    GGML_LOG_WARN("%s: oneDNN SDPA failed (%s); falling back to TILE kernel\n", __func__, e.what());
    ggml_sycl_flash_attn_ext_tile(ctx, dst);
}

#endif // GGML_SYCL_DNNL
