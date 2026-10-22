#ਅਨ੍ਤਰ੍ਭਾਵ "fused-gemm.hpp"

#ਅਨ੍ਤਰ੍ਭਾਵ <sycl/ext/oneapi/matrix/matrix.hpp>

#ਅਨ੍ਤਰ੍ਭਾਵ <algorithm>
#ਅਨ੍ਤਰ੍ਭਾਵ <string>
#ਅਨ੍ਤਰ੍ਭਾਵ <tuple>
#ਅਨ੍ਤਰ੍ਭਾਵ <mutex>
#ਅਨ੍ਤਰ੍ਭਾਵ <set>
#ਅਨ੍ਤਰ੍ਭਾਵ <unordered_map>

ਨਾਮਾਕਾਸ਼ mx = sycl::ext::oneapi::experimental::ਮੈਟ੍ਰਿਕਸ;

// FG_ / fg_ is short for fused GEMM: the weights are dequantized inside the GEMM, into the XMX tiles.

// A k step is one 32-value weight sub-block; iq3_s and the other superblock formats split their
// superblock into steps of this width. The sub-groups of a work-group each walk their own K range
// and are summed at the end.
ਸਥਿਰ constexpr ਪੂਰਨ FG_BK     = QK4_NL;
ਸਥਿਰ constexpr ਪੂਰਨ FG_KSPLIT = 4;

// Element traits of one joint_matrix operand type. The A stage and the B pack compute in f32 and
// convert once, in registers, when they write the element, so any type costs the same one pass.
//   store: storage in SLM (A) and in the packed B buffer
//   mtype: matrix_type in matrix_combinations
//   mode:  GGML_SYCL_DYNAMIC_PRECISION value that selects this type
//   src:   ggml type that needs no conversion into this type (GGML_TYPE_COUNT: none)
//   slow:  XMX throughput class, 0 is fastest. f16 and bf16 share the DPAS rate; tf32 does half the
//          K per instruction. B60, Qwen3-30B-A3B pp512: f16 1108, bf16 1000, tf32 751 t/s
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T> ਬਣਤਰ fg_elem;

ਸਾਂਚਾ <> ਬਣਤਰ fg_elem<sycl::half> {
    ਵਰਤੋਂ store = sycl::half;
    ਵਰਤੋਂ pair  = sycl::half2;
    ਸਥਿਰ constexpr mx::ਮੈਟ੍ਰਿਕਸ_type mtype = mx::ਮੈਟ੍ਰਿਕਸ_type::fp16;
    ਸਥਿਰ constexpr ਪੂਰਨ             mode  = GGML_SYCL_DYNAMIC_PRECISION_F16;
    ਸਥਿਰ constexpr ggml_type       src   = GGML_TYPE_F16;
    ਸਥਿਰ constexpr ਪੂਰਨ             mant  = 10;
    ਸਥਿਰ constexpr ਪੂਰਨ             slow  = 0;
    ਸਥਿਰ store cvt(ਦਸ਼ਮਲਵ x) { ਵਾਪਸ (store) x; }
    ਸਥਿਰ pair make(ਦਸ਼ਮਲਵ x, ਦਸ਼ਮਲਵ y) { ਵਾਪਸ pair((store) x, (store) y); }
};

// tf32 rounds to nearest even with plain bit ops: round_to_tf32 needs a SPIR-V extension that the
// DG2 AOT target rejects
ਸਥਿਰ ਸਤਰਅੰਦਰ uint32_t fg_ਗੇੜਾ_ਬਿੱਟ(ਦਸ਼ਮਲਵ x, ਪੂਰਨ drop) {
    ਅਚਲ uint32_t u = sycl::ਦੋਆਧਾਰੀ_ਅੰਕ_cast<uint32_t>(x);
    ਜੇ ((u & 0x7f800000u) == 0x7f800000u) {
        ਵਾਪਸ (u & 0x7fffffu) ? u | (1u << drop) : u; // nan stays nan
    }
    ਵਾਪਸ u + ((1u << (drop - 1)) - 1) + ((u >> drop) & 1);
}

ਬਣਤਰ alignas(4) fg_bf16x2 {
    sycl::ext::oneapi::bfloat16 x, y;
};

ਸਾਂਚਾ <> ਬਣਤਰ fg_elem<sycl::ext::oneapi::bfloat16> {
    ਵਰਤੋਂ store = sycl::ext::oneapi::bfloat16;
    ਵਰਤੋਂ pair  = fg_bf16x2;
    ਸਥਿਰ constexpr mx::ਮੈਟ੍ਰਿਕਸ_type mtype = mx::ਮੈਟ੍ਰਿਕਸ_type::bf16;
    ਸਥਿਰ constexpr ਪੂਰਨ             mode  = GGML_SYCL_DYNAMIC_PRECISION_BF16;
    ਸਥਿਰ constexpr ggml_type       src   = GGML_TYPE_BF16;
    ਸਥਿਰ constexpr ਪੂਰਨ             mant  = 7;
    ਸਥਿਰ constexpr ਪੂਰਨ             slow  = 0;
    ਸਥਿਰ store cvt(ਦਸ਼ਮਲਵ x) { ਵਾਪਸ store(x); }
    ਸਥਿਰ pair make(ਦਸ਼ਮਲਵ x, ਦਸ਼ਮਲਵ y) { ਵਾਪਸ { cvt(x), cvt(y) }; }
};

// tf32 keeps f32 range and f16 mantissa, in f32 storage
ਸਾਂਚਾ <> ਬਣਤਰ fg_elem<mx::precision::tf32> {
    ਵਰਤੋਂ store = ਦਸ਼ਮਲਵ;
    ਵਰਤੋਂ pair  = sycl::float2;
    ਸਥਿਰ constexpr mx::ਮੈਟ੍ਰਿਕਸ_type mtype = mx::ਮੈਟ੍ਰਿਕਸ_type::tf32;
    ਸਥਿਰ constexpr ਪੂਰਨ             mode  = GGML_SYCL_DYNAMIC_PRECISION_TF32;
    ਸਥਿਰ constexpr ggml_type       src   = GGML_TYPE_ਗਿਣਤੀ;
    ਸਥਿਰ constexpr ਪੂਰਨ             mant  = 10;
    ਸਥਿਰ constexpr ਪੂਰਨ             slow  = 1;
    ਸਥਿਰ store cvt(ਦਸ਼ਮਲਵ x) { ਵਾਪਸ sycl::ਦੋਆਧਾਰੀ_ਅੰਕ_cast<ਦਸ਼ਮਲਵ>(fg_ਗੇੜਾ_ਬਿੱਟ(x, 13) & ~0x1fffu); }
    ਸਥਿਰ pair make(ਦਸ਼ਮਲਵ x, ਦਸ਼ਮਲਵ y) { ਵਾਪਸ pair(cvt(x), cvt(y)); }
};

// One joint_matrix combination (A type, B type, TM x TN x TK, sub-group size; C and D are f32) and
// the tiling built on it. A sub-group owns SG_ROWS rows of A (at least 16) and BN (at least 32)
// columns of B. A and B may differ: the device lists the pairs it supports.
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ TA, ਕਿਸਮ_ਨਾਂ TB, ਪੂਰਨ TM_, ਪੂਰਨ TN_, ਪੂਰਨ TK_, ਪੂਰਨ SG_> ਬਣਤਰ fg_combo {
    ਵਰਤੋਂ ta  = TA;
    ਵਰਤੋਂ tb  = TB;
    ਵਰਤੋਂ EA  = fg_elem<TA>;
    ਵਰਤੋਂ EB  = fg_elem<TB>;
    ਵਰਤੋਂ tsa = ਕਿਸਮ_ਨਾਂ EA::store;
    ਵਰਤੋਂ tsb = ਕਿਸਮ_ਨਾਂ EB::store;
    ਸਥਿਰ constexpr ਪੂਰਨ TM = TM_;
    ਸਥਿਰ constexpr ਪੂਰਨ TN = TN_;
    ਸਥਿਰ constexpr ਪੂਰਨ TK = TK_;
    ਸਥਿਰ constexpr ਪੂਰਨ SG = SG_;
    ਸਥਿਰ constexpr ਪੂਰਨ VNNI    = 4 / ਆਕਾਰ(tsb);  // K rows of B packed in one 32-bit word
    ਸਥਿਰ constexpr ਪੂਰਨ SG_ਪੰਕਤੀਆਂ = TM > 16 ? TM : 16;
    ਸਥਿਰ constexpr ਪੂਰਨ RPL     = SG_ਪੰਕਤੀਆਂ / SG;     // A rows one lane decodes per k step
    ਸਥਿਰ constexpr ਪੂਰਨ MT      = SG_ਪੰਕਤੀਆਂ / TM;
    ਸਥਿਰ constexpr ਪੂਰਨ BN      = TN > 32 ? TN : 32;
    ਸਥਿਰ constexpr ਪੂਰਨ NT      = BN / TN;
    ਸਥਿਰ constexpr ਪੂਰਨ WG_ਮਾਪ = FG_KSPLIT * SG;
    ਸਥਿਰ constexpr mx::layout b_layout = VNNI == 1 ? mx::layout::ਪੰਕਤੀ_major : mx::layout::ext_intel_packed;
    // a 64-wide N is mostly padding here and a 32x64 f32 accumulator needs 128 registers per lane,
    // so it spills: 13x slower on B60
    ਸਥਿਰ constexpr ਤਾਰਕਿਕ efficient = TN <= 32;
    static_assert(SG_ਪੰਕਤੀਆਂ % SG == 0 && SG_ਪੰਕਤੀਆਂ % TM == 0 && BN % TN == 0 && FG_BK % TK == 0, "bad tile");
    static_assert(BN <= GGML_SYCL_FG_MAX_N, "header gate must cover the tile width");
};

ਵਰਤੋਂ fg_half = sycl::half;
ਵਰਤੋਂ fg_bf16 = sycl::ext::oneapi::bfloat16;
ਵਰਤੋਂ fg_tf32 = mx::precision::tf32;

// One bit of GGML_SYCL_XMX_GATHER_SHAPES per combination. Only combinations some device lists in
// matrix_combinations are built (appendix of sycl_ext_oneapi_matrix and the runtime's own list).
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ F> ਸਥਿਰ ਖਾਲੀ fg_ਦੇਖੋ_combo(ਪੂਰਨ idx, F && f);
ਸਥਿਰ constexpr ਪੂਰਨ FG_N_COMBOS = 8;

// A spir64_gen AOT build (GGML_SYCL_XMX_AOT_SG) drops the combinations of the other sub-group size
// entirely: ocloc rejects even an empty kernel that asks for a sub-group size it lacks.
ਸਾਂਚਾ <ਪੂਰਨ SG> ਸਥਿਰ constexpr ਤਾਰਕਿਕ fg_listed() {
#ਜੇ defined(GGML_SYCL_XMX_AOT_SG)
    ਵਾਪਸ SG == GGML_SYCL_XMX_AOT_SG;
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ ਸੱਚ;
#ਜੇ_ਅੰਤ
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ S, ਕਿਸਮ_ਨਾਂ F> ਸਥਿਰ ਖਾਲੀ fg_call_combo(F && f) {
    ਜੇ constexpr (fg_listed<S::SG>()) {
        f(S{});
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ F> ਸਥਿਰ ਖਾਲੀ fg_ਦੇਖੋ_combo(ਪੂਰਨ idx, F && f) {
    ਚੋਣ (idx) {
        ਹਾਲਤ 0: fg_call_combo<fg_combo<fg_half, fg_half, 8, 16, 16, 16>>(f);  ਰੋਕੋ; // Xe2, Xe3, Xe-HPC
        ਹਾਲਤ 1: fg_call_combo<fg_combo<fg_half, fg_half, 16, 16, 16, 16>>(f); ਰੋਕੋ; // Xe2, Xe3, Xe-HPC
        ਹਾਲਤ 2: fg_call_combo<fg_combo<fg_half, fg_half, 32, 64, 16, 16>>(f); ਰੋਕੋ; // Xe2, Xe3, Xe-HPC
        ਹਾਲਤ 3: fg_call_combo<fg_combo<fg_half, fg_half, 32, 64, 32, 16>>(f); ਰੋਕੋ; // Xe2, Xe3, Xe-HPC
        ਹਾਲਤ 4: fg_call_combo<fg_combo<fg_half, fg_half, 8, 8, 16, 8>>(f);    ਰੋਕੋ; // Xe-HPG (Arc A), ARL-H
        ਹਾਲਤ 5: fg_call_combo<fg_combo<fg_tf32, fg_tf32, 8, 16, 8, 16>>(f);   ਰੋਕੋ; // Xe2, Xe3, Xe-HPC
        ਹਾਲਤ 6: fg_call_combo<fg_combo<fg_bf16, fg_bf16, 8, 16, 16, 16>>(f);  ਰੋਕੋ; // Xe2, Xe3, Xe-HPC
        ਹਾਲਤ 7: fg_call_combo<fg_combo<fg_bf16, fg_bf16, 8, 8, 16, 8>>(f);    ਰੋਕੋ; // Xe-HPG (Arc A), ARL-H
        ਮੂਲ: GGML_ABORT("bad XMX combination %d", idx);
    }
}

// AOT with -fsycl-targets=intel_gpu_*: compile each tile body only for targets with its sub-group
// size, since IGC fails on the other ones. A JIT build keeps them all, but each combination lands in
// its own device image (joint_matrix is an optional kernel feature) and only a combination the
// device reports is launched, so the runtime never asks IGC for the others.
#ਜੇ defined(__SYCL_DEVICE_ONLY__)
#    ਜੇ __SYCL_TARGET_INTEL_GPU_ACM_G10__ || __SYCL_TARGET_INTEL_GPU_ACM_G11__ || __SYCL_TARGET_INTEL_GPU_ACM_G12__ || \
        __SYCL_TARGET_INTEL_GPU_ARL_H__
#        define FG_AOT_SG 8
#    elif __SYCL_TARGET_INTEL_GPU_PVC__ || __SYCL_TARGET_INTEL_GPU_PVC_VG__ || __SYCL_TARGET_INTEL_GPU_BMG_G21__ || \
        __SYCL_TARGET_INTEL_GPU_BMG_G31__ || __SYCL_TARGET_INTEL_GPU_LNL_M__ || __SYCL_TARGET_INTEL_GPU_PTL_H__ ||   \
        __SYCL_TARGET_INTEL_GPU_PTL_U__ || __SYCL_TARGET_INTEL_GPU_WCL__ || __SYCL_TARGET_INTEL_GPU_NVL_S__ ||       \
        __SYCL_TARGET_INTEL_GPU_NVL_U__ || __SYCL_TARGET_INTEL_GPU_NVL_P__
#        define FG_AOT_SG 16
#    elif __SYCL_TARGET_INTEL_GPU_TGLLP__ || __SYCL_TARGET_INTEL_GPU_RKL__ || __SYCL_TARGET_INTEL_GPU_ADL_S__ || \
        __SYCL_TARGET_INTEL_GPU_ADL_P__ || __SYCL_TARGET_INTEL_GPU_ADL_N__ || __SYCL_TARGET_INTEL_GPU_DG1__ ||   \
        __SYCL_TARGET_INTEL_GPU_MTL_U__ || __SYCL_TARGET_INTEL_GPU_MTL_H__
#        define FG_AOT_SG 0 // no XMX
#    endif
#ਜੇ_ਅੰਤ

ਸਾਂਚਾ <ਪੂਰਨ SG> ਸਥਿਰ constexpr ਤਾਰਕਿਕ fg_built() {
#ਜੇ defined(FG_AOT_SG)
    ਵਾਪਸ SG == FG_AOT_SG;
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ ਸੱਚ;
#ਜੇ_ਅੰਤ
}

// Upper bound on the tile count when total_rows rows are routed to n_as experts: the worst case gives
// each expert one row and fills whole tiles with the rest. The bound depends only on the shape, not
// on the routing, so the pool reuses one buffer every ubatch instead of keeping one per size seen.
ਸਥਿਰ constexpr int64_t grouped_gemm_max_tiles(int64_t ਕੁੱਲ_ਪੰਕਤੀਆਂ, int64_t n_as, int64_t BN) {
    ਵਾਪਸ ਕੁੱਲ_ਪੰਕਤੀਆਂ <= n_as ? ਕੁੱਲ_ਪੰਕਤੀਆਂ : n_as + (ਕੁੱਲ_ਪੰਕਤੀਆਂ - n_as) / BN;
}
// Tiles do not cross experts, so the bound is not ceil(total_rows / BN): 34 rows over 2 experts with
// BN = 16 split 17 + 17 need 2 + 2 tiles, where the ceil gives 3.
static_assert(grouped_gemm_max_tiles(34, 2, 16) == 4);

// the device lists S with an f32 accumulator and output
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ S> ਸਥਿਰ ਤਾਰਕਿਕ fg_device_has_combo(ਅਚਲ std::vector<mx::combination> & combinations) {
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & c : combinations) {
        ਜੇ (c.atype == S::EA::mtype && c.btype == S::EB::mtype && c.ctype == mx::ਮੈਟ੍ਰਿਕਸ_type::fp32 &&
            c.dtype == mx::ਮੈਟ੍ਰਿਕਸ_type::fp32 &&
            (c.max_msize >= (size_t) S::TM || c.msize == (size_t) S::TM) &&
            (c.max_nsize >= (size_t) S::TN || c.nsize == (size_t) S::TN) &&
            (c.max_ksize >= (size_t) S::TK || c.ksize == (size_t) S::TK)) {
            ਵਾਪਸ ਸੱਚ;
        }
    }
    ਵਾਪਸ ਝੂਠ;
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T> ਸਥਿਰ ਅਚਲ ਅੱਖਰ * fg_type_name() {
    ਵਾਪਸ std::is_same_v<T, fg_half> ? "f16" : std::is_same_v<T, fg_bf16> ? "bf16" : "tf32";
}

ਸਥਿਰ std::string fg_combo_name(ਪੂਰਨ idx) {
    std::string name;
    fg_ਦੇਖੋ_combo(idx, [&](ਸਵੈਚਾਲਿਤ s) {
        ਵਰਤੋਂ S = decltype(s);
        name = std::string(fg_type_name<ਕਿਸਮ_ਨਾਂ S::ta>()) + "x" + fg_type_name<ਕਿਸਮ_ਨਾਂ S::tb>() + " " +
               std::to_string(S::TM) + "x" + std::to_string(S::TN) + "x" + std::to_string(S::TK) + " sg" +
               std::to_string(S::SG);
    });
    ਵਾਪਸ name;
}

// Combinations this build has kernels for and the device lists, one bit each. Cached per device:
// on a mixed box the first caller's verdict is not the others'.
ਸਥਿਰ ਪੂਰਨ fg_device_combos(ਅਚਲ sycl::device & dev) {
    ਸਥਿਰ std::mutex                            mtx;
    ਸਥਿਰ std::unordered_map<sycl::device, ਪੂਰਨ> ਗਿਆਤ;
    std::lock_guard<std::mutex>                  ਤਾਲਾ(mtx);
    ਅਚਲ ਸਵੈਚਾਲਿਤ                                   it = ਗਿਆਤ.find(dev);
    ਜੇ (it != ਗਿਆਤ.end()) {
        ਵਾਪਸ it->second;
    }
    ਪੂਰਨ ਉਪਲਬਧ = 0;
    ਕੋਸ਼ਿਸ਼ {
        ਅਚਲ ਸਵੈਚਾਲਿਤ combinations = dev.get_info<sycl::ext::oneapi::experimental::info::device::ਮੈਟ੍ਰਿਕਸ_combinations>();
        ਅਚਲ ਸਵੈਚਾਲਿਤ sg_sizes     = dev.get_info<sycl::info::device::sub_ਸਮੂਹ_sizes>();
        ਲਈ (ਪੂਰਨ idx = 0; idx < FG_N_COMBOS; ++idx) {
            fg_ਦੇਖੋ_combo(idx, [&](ਸਵੈਚਾਲਿਤ s) {
                ਵਰਤੋਂ S = decltype(s);
                ਅਚਲ ਤਾਰਕਿਕ sg_ok = std::find(sg_sizes.begin(), sg_sizes.end(), (size_t) S::SG) != sg_sizes.end();
                ਜੇ (sg_ok && fg_device_has_combo<S>(combinations)) {
                    ਉਪਲਬਧ |= 1 << idx;
                }
            });
        }
    } ਫੜੋ (ਅਚਲ sycl::exception &) {
        ਉਪਲਬਧ = 0;
    }
    GGML_LOG_INFO("%s: %s: XMX dequant-GEMM combinations available 0x%x, allowed 0x%x\n", __func__,
                  dev.get_info<sycl::info::device::name>().c_str(), ਉਪਲਬਧ, g_ggml_sycl_xmx_gather_shapes);
    ਗਿਆਤ.emplace(dev, ਉਪਲਬਧ);
    ਵਾਪਸ ਉਪਲਬਧ;
}

// Rank of combination S for a src1 of type src1_type, lower is better. Order:
//  1. throughput: a tile that does not spill, then the fastest type class of A and B
//  2. B type equal to the src1 type, so the pack is a plain copy
//  3. B at least as precise as f16
//  4. the device's native DPAS tile (8 x SG x 32 bytes of K), then the largest M x K
// A costs nothing to convert: the A stage emits any type at the same cost.
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ S> ਸਥਿਰ int64_t fg_rank(ggml_type src1_type) {
    ਅਚਲ int64_t spills  = !S::efficient;
    ਅਚਲ int64_t slow    = std::max(S::EA::slow, S::EB::slow);
    ਅਚਲ int64_t convert = S::EB::src != src1_type;
    ਅਚਲ int64_t lossy   = S::EB::mant < 10;
    ਅਚਲ int64_t foreign = !(S::TM == 8 && S::TN == S::SG);
    ਅਚਲ int64_t mk      = 1024 - S::TM * S::TK;
    ਵਾਪਸ ((((spills * 2 + slow) * 2 + convert) * 2 + lossy) * 2 + foreign) * 2048 + mk;
}

// whether XMX operands of type mode (a GGML_SYCL_DYNAMIC_PRECISION value) meet the src1 request
// [TAG_GGML_PREC]. f16 lacks the f32 range that BF16 and F32 ask for; an F32 request goes only as far
// down as GGML_SYCL_DYNAMIC_REQUIRED_PRECISION allows.
ਸਥਿਰ ਤਾਰਕਿਕ fg_ਬਹੁਲਕ_meets(ਪੂਰਨ mode, int32_t src1_prec) {
    ਜੇ (src1_prec == GGML_PREC_UNDEFINED || src1_prec >= GGML_PREC_F16) {
        ਵਾਪਸ ਸੱਚ;
    }
    ਜੇ (src1_prec >= GGML_PREC_BF16) {
        ਵਾਪਸ mode != GGML_SYCL_DYNAMIC_PRECISION_F16;
    }
    ਚੋਣ (g_ggml_sycl_dynamic_required_precision) {
        ਹਾਲਤ GGML_SYCL_DYNAMIC_PRECISION_TF32: ਵਾਪਸ mode == GGML_SYCL_DYNAMIC_PRECISION_TF32;
        ਹਾਲਤ GGML_SYCL_DYNAMIC_PRECISION_BF16: ਵਾਪਸ mode != GGML_SYCL_DYNAMIC_PRECISION_F16;
        ਮੂਲ:                               ਵਾਪਸ ਝੂਠ;
    }
}

// Best allowed combination for this call, or -1 if none. The type is GGML_SYCL_DYNAMIC_PRECISION if it
// meets the src1 request; if not, bf16 then tf32 for a BF16 request (fastest first) and tf32 then bf16
// for an F32 request (most mantissa first).
ਸਥਿਰ ਪੂਰਨ fg_pick_combo(dpct::ਕਤਾਰ_ptr stream, ggml_type src1_type, int32_t src1_prec) {
    ਅਚਲ sycl::device dev     = stream->get_device();
    ਅਚਲ ਪੂਰਨ          allowed = fg_device_combos(dev) & g_ggml_sycl_xmx_gather_shapes;
    ਅਚਲ ਤਾਰਕਿਕ         f32_req = src1_prec != GGML_PREC_UNDEFINED && src1_prec < GGML_PREC_BF16;
    ਅਚਲ ਪੂਰਨ          modes[] = {
        g_ggml_sycl_dynamic_precision,
        f32_req ? GGML_SYCL_DYNAMIC_PRECISION_TF32 : GGML_SYCL_DYNAMIC_PRECISION_BF16,
        f32_req ? GGML_SYCL_DYNAMIC_PRECISION_BF16 : GGML_SYCL_DYNAMIC_PRECISION_TF32,
    };
    ਪੂਰਨ     ਉੱਤਮ      = -1;
    int64_t ਉੱਤਮ_rank = 0;
    ਲਈ (ਪੂਰਨ i = 0; i < 3 && ਉੱਤਮ < 0; ++i) {
        ਅਚਲ ਪੂਰਨ mode = modes[i];
        ਜੇ (!fg_ਬਹੁਲਕ_meets(mode, src1_prec)) {
            ਜਾਰੀ;
        }
        ਲਈ (ਪੂਰਨ idx = 0; idx < FG_N_COMBOS; ++idx) {
            ਜੇ (!(allowed & (1 << idx))) {
                ਜਾਰੀ;
            }
            fg_ਦੇਖੋ_combo(idx, [&](ਸਵੈਚਾਲਿਤ s) {
                ਵਰਤੋਂ S = decltype(s);
                ਜੇ (S::EA::mode != mode || S::EB::mode != mode) {
                    ਵਾਪਸ;
                }
                ਅਚਲ int64_t rank = fg_rank<S>(src1_type);
                ਜੇ (ਉੱਤਮ < 0 || rank < ਉੱਤਮ_rank) {
                    ਉੱਤਮ      = idx;
                    ਉੱਤਮ_rank = rank;
                }
            });
        }
    }
    // log each distinct decision once
    ਸਥਿਰ std::mutex                                      mtx;
    ਸਥਿਰ std::set<std::tuple<size_t, ਪੂਰਨ, int32_t, ਪੂਰਨ>> ਵੇਖਿਆ;
    std::lock_guard<std::mutex>                            ਤਾਲਾ(mtx);
    ਜੇ (ਵੇਖਿਆ.emplace(std::hash<sycl::device>{}(dev), (ਪੂਰਨ) src1_type, src1_prec, ਉੱਤਮ).second) {
        GGML_LOG_INFO("%s: src1 %s, src1 prec %d -> %s\n", __func__, ggml_type_name(src1_type), src1_prec,
                      ਉੱਤਮ >= 0 ? fg_combo_name(ਉੱਤਮ).c_str() : "none (library GEMM)");
    }
    ਵਾਪਸ ਉੱਤਮ;
}

// src1 [N][K] -> packed [K/V][Npad][V] so B tiles load straight from global memory
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ E, ਕਿਸਮ_ਨਾਂ T_src>
ਸਥਿਰ ਖਾਲੀ fused_gemm_pack_b(ਅਚਲ T_src * y, ਕਿਸਮ_ਨਾਂ E::store * packed, ਪੂਰਨ N, ਪੂਰਨ Npad, ਪੂਰਨ K,
                              dpct::ਕਤਾਰ_ptr stream) {
    constexpr ਪੂਰਨ V   = 4 / ਆਕਾਰ(ਕਿਸਮ_ਨਾਂ E::store);
    ਅਚਲ ਪੂਰਨ     kqs = K / V;
    stream->parallel_for(sycl::range<1>((size_t) Npad * kqs), [=](sycl::id<1> id) {
        ਅਚਲ ਪੂਰਨ idx = id[0];
        ਅਚਲ ਪੂਰਨ n   = idx / kqs;
        ਅਚਲ ਪੂਰਨ kq  = idx - n * kqs;
        ਕਿਸਮ_ਨਾਂ E::store vals[V] = {};
        ਜੇ (n < N) {
            ਅਚਲ T_src * src = y + (size_t) n * K + V * kq;
#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ v = 0; v < V; ++v) {
                vals[v] = E::cvt((ਦਸ਼ਮਲਵ) src[v]);
            }
        }
        ਕਿਸਮ_ਨਾਂ E::store * ਬਾਹਰਲਾ = packed + ((size_t) kq * Npad + n) * V;
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ v = 0; v < V; ++v) {
            ਬਾਹਰਲਾ[v] = vals[v];
        }
    });
}

// A stage: one lane owns one row and decodes FG_BK values of it per k step, with every scale
// folded into the value so the mad below sees plain A elements. One overload per weight format.
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ E>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ fg_stage_a(ਅਚਲ ਖੰਡ_iq4_nl * __restrict__ xrow, ਅਚਲ ਪੂਰਨ kb, ਕਿਸਮ_ਨਾਂ E::pair * a) {
    ਅਚਲ ਖੰਡ_iq4_nl blk = xrow[kb];
    ਅਚਲ ਦਸ਼ਮਲਵ        d   = (ਦਸ਼ਮਲਵ) blk.d;
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ j = 0; j < QK4_NL / 2; j += 2) {
        ਅਚਲ uint8_t q0 = blk.qs[j];
        ਅਚਲ uint8_t q1 = blk.qs[j + 1];
        a[j / 2]     = E::make(d * kvalues_iq4nl[q0 & 0xf], d * kvalues_iq4nl[q1 & 0xf]);
        a[j / 2 + 8] = E::make(d * kvalues_iq4nl[q0 >> 4], d * kvalues_iq4nl[q1 >> 4]);
    }
}

// iq3_s: k step kb is sub-block kb % 8 of superblock kb / 8. The superblock is 110 bytes, so read
// only the fields of that sub-block instead of copying the block. Same decode as
// dequantize_block_iq3_s: grid entries are taken as dwords and the sign bit is a plain shift.
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ E>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ fg_stage_a(ਅਚਲ ਖੰਡ_iq3_s * __restrict__ xrow, ਅਚਲ ਪੂਰਨ kb, ਕਿਸਮ_ਨਾਂ E::pair * a) {
    static_assert(QK_K == 256, "the iq3_s A stage assumes 8 sub-blocks per superblock");
    ਅਚਲ ਖੰਡ_iq3_s * blk = xrow + kb / (QK_K / 32);
    ਅਚਲ ਪੂਰਨ           ib8 = kb % (QK_K / 32);
    ਅਚਲ uint8_t *     qs  = blk->qs + 8 * ib8;
    ਅਚਲ ਪੂਰਨ           qh  = blk->qh[ib8];
    ਅਚਲ ਦਸ਼ਮਲਵ         d   = (ਦਸ਼ਮਲਵ) blk->d * (1 + 2 * ((blk->scales[ib8 / 2] >> (4 * (ib8 % 2))) & 0xf));
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ il = 0; il < 4; ++il) {
        ਅਚਲ uint32_t grid1 = iq3s_grid[qs[2 * il + 0] | ((qh << (8 - 2 * il)) & 256)];
        ਅਚਲ uint32_t grid2 = iq3s_grid[qs[2 * il + 1] | ((qh << (7 - 2 * il)) & 256)];
        ਅਚਲ ਪੂਰਨ      signs = blk->signs[4 * ib8 + il];
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ j = 0; j < 2; ++j) {
            ਅਚਲ ਦਸ਼ਮਲਵ g1a = (ਦਸ਼ਮਲਵ) ((grid1 >> (16 * j + 0)) & 0xff);
            ਅਚਲ ਦਸ਼ਮਲਵ g1b = (ਦਸ਼ਮਲਵ) ((grid1 >> (16 * j + 8)) & 0xff);
            ਅਚਲ ਦਸ਼ਮਲਵ g2a = (ਦਸ਼ਮਲਵ) ((grid2 >> (16 * j + 0)) & 0xff);
            ਅਚਲ ਦਸ਼ਮਲਵ g2b = (ਦਸ਼ਮਲਵ) ((grid2 >> (16 * j + 8)) & 0xff);
            ਅਚਲ ਪੂਰਨ   s   = 2 * j;
            a[4 * il + j]     = E::make(d * ((signs & (1 << (s + 0))) ? -g1a : g1a),
                                        d * ((signs & (1 << (s + 1))) ? -g1b : g1b));
            a[4 * il + j + 2] = E::make(d * ((signs & (1 << (s + 4))) ? -g2a : g2a),
                                        d * ((signs & (1 << (s + 5))) ? -g2b : g2b));
        }
    }
}

// values per stored block, so a row of K values is K/qk blocks
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ ਖੰਡ_q_t> ਬਣਤਰ fg_ਖੰਡ_traits;
ਸਾਂਚਾ <> ਬਣਤਰ fg_ਖੰਡ_traits<ਖੰਡ_iq4_nl> { ਸਥਿਰ constexpr ਪੂਰਨ qk = QK4_NL; };
ਸਾਂਚਾ <> ਬਣਤਰ fg_ਖੰਡ_traits<ਖੰਡ_iq3_s>  { ਸਥਿਰ constexpr ਪੂਰਨ qk = QK_K; };
ਸਾਂਚਾ <> ਬਣਤਰ fg_ਖੰਡ_traits<ਖੰਡ_iq3_xxs> { ਸਥਿਰ constexpr ਪੂਰਨ qk = QK_K; };
ਸਾਂਚਾ <> ਬਣਤਰ fg_ਖੰਡ_traits<ਖੰਡ_iq4_xs>  { ਸਥਿਰ constexpr ਪੂਰਨ qk = QK_K; };
ਸਾਂਚਾ <> ਬਣਤਰ fg_ਖੰਡ_traits<ਖੰਡ_iq2_xxs> { ਸਥਿਰ constexpr ਪੂਰਨ qk = QK_K; };
ਸਾਂਚਾ <> ਬਣਤਰ fg_ਖੰਡ_traits<ਖੰਡ_iq2_xs>  { ਸਥਿਰ constexpr ਪੂਰਨ qk = QK_K; };
ਸਾਂਚਾ <> ਬਣਤਰ fg_ਖੰਡ_traits<ਖੰਡ_iq2_s>   { ਸਥਿਰ constexpr ਪੂਰਨ qk = QK_K; };
ਸਾਂਚਾ <> ਬਣਤਰ fg_ਖੰਡ_traits<ਖੰਡ_iq1_s>   { ਸਥਿਰ constexpr ਪੂਰਨ qk = QK_K; };
ਸਾਂਚਾ <> ਬਣਤਰ fg_ਖੰਡ_traits<ਖੰਡ_iq1_m>   { ਸਥਿਰ constexpr ਪੂਰਨ qk = QK_K; };
ਸਾਂਚਾ <> ਬਣਤਰ fg_ਖੰਡ_traits<ਖੰਡ_q8_0>    { ਸਥਿਰ constexpr ਪੂਰਨ qk = QK8_0; };
ਸਾਂਚਾ <> ਬਣਤਰ fg_ਖੰਡ_traits<ਖੰਡ_q4_K>    { ਸਥਿਰ constexpr ਪੂਰਨ qk = QK_K; };
ਸਾਂਚਾ <> ਬਣਤਰ fg_ਖੰਡ_traits<ਖੰਡ_q5_K>    { ਸਥਿਰ constexpr ਪੂਰਨ qk = QK_K; };
ਸਾਂਚਾ <> ਬਣਤਰ fg_ਖੰਡ_traits<ਖੰਡ_q6_K>    { ਸਥਿਰ constexpr ਪੂਰਨ qk = QK_K; };

// The A stages below are the dequantize_block_iq* kernels rewritten for one k step. There a
// work-item handled one quarter (il) of one 32-wide sub-block (ib); here one lane produces the
// whole step, so il becomes a loop and ib is kb inside the superblock. Each quarter yields 8
// consecutive values, i.e. 4 pairs at a[4*il], so nothing larger than 8 floats is ever live.
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FG_SUPERBLOCK(T)                                                         \
    static_assert(QK_K == 256, "the " #T " A stage assumes 8 sub-blocks per superblock"); \
    ਅਚਲ T * blk = xrow + kb / (QK_K / 32);                                     \
    ਅਚਲ ਪੂਰਨ ib  = kb % (QK_K / 32)

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ E>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ fg_pack_quarter(ਅਚਲ ਦਸ਼ਮਲਵ * __restrict__ t, ਕਿਸਮ_ਨਾਂ E::pair * a, ਪੂਰਨ il) {
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ j = 0; j < 4; ++j) {
        a[4 * il + j] = E::make(t[2 * j], t[2 * j + 1]);
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ E>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ fg_stage_a(ਅਚਲ ਖੰਡ_iq4_xs * __restrict__ xrow, ਅਚਲ ਪੂਰਨ kb, ਕਿਸਮ_ਨਾਂ E::pair * a) {
    FG_SUPERBLOCK(ਖੰਡ_iq4_xs);
    // low nibbles fill the first half of the step, high nibbles the second, so the two halves
    // land at a[0..7] and a[8..15] and no quarter loop is needed
    ਅਚਲ ਦਸ਼ਮਲਵ d = (ਦਸ਼ਮਲਵ) blk->d *
        ((((blk->scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) | (((blk->scales_h >> (2 * ib)) & 3) << 4)) - 32);
    ਅਚਲ uint8_t * q4 = blk->qs + 16 * ib;
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ j = 0; j < 8; ++j) {
        a[j]     = E::make(d * kvalues_iq4nl[q4[2 * j] & 0xf], d * kvalues_iq4nl[q4[2 * j + 1] & 0xf]);
        a[8 + j] = E::make(d * kvalues_iq4nl[q4[2 * j] >> 4],  d * kvalues_iq4nl[q4[2 * j + 1] >> 4]);
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ E>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ fg_stage_a(ਅਚਲ ਖੰਡ_iq3_xxs * __restrict__ xrow, ਅਚਲ ਪੂਰਨ kb, ਕਿਸਮ_ਨਾਂ E::pair * a) {
    FG_SUPERBLOCK(ਖੰਡ_iq3_xxs);
    ਅਚਲ uint8_t *  q3    = blk->qs + 8 * ib;
    ਅਚਲ uint16_t * gas   = (ਅਚਲ uint16_t *) (blk->qs + QK_K / 4) + 2 * ib;
    ਅਚਲ uint32_t   aux32 = gas[0] | (gas[1] << 16);
    ਅਚਲ ਦਸ਼ਮਲਵ      d     = (ਦਸ਼ਮਲਵ) blk->d * (0.5f + (aux32 >> 28)) * 0.5f;
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ il = 0; il < 4; ++il) {
        ਅਚਲ uint8_t * grid1 = (ਅਚਲ uint8_t *) (iq3xxs_grid + q3[2 * il + 0]);
        ਅਚਲ uint8_t * grid2 = (ਅਚਲ uint8_t *) (iq3xxs_grid + q3[2 * il + 1]);
        ਅਚਲ uint8_t   signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];
        ਦਸ਼ਮਲਵ t[8];
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ j = 0; j < 4; ++j) {
            t[j + 0] = d * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f);
            t[j + 4] = d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ E>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ fg_stage_a(ਅਚਲ ਖੰਡ_iq2_xxs * __restrict__ xrow, ਅਚਲ ਪੂਰਨ kb, ਕਿਸਮ_ਨਾਂ E::pair * a) {
    FG_SUPERBLOCK(ਖੰਡ_iq2_xxs);
    ਅਚਲ uint16_t * q2    = blk->qs + 4 * ib;
    ਅਚਲ uint8_t *  aux8  = (ਅਚਲ uint8_t *) q2;
    ਅਚਲ uint32_t   aux32 = q2[2] | (q2[3] << 16);
    ਅਚਲ ਦਸ਼ਮਲਵ      d     = (ਦਸ਼ਮਲਵ) blk->d * (0.5f + (aux32 >> 28)) * 0.25f;
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ il = 0; il < 4; ++il) {
        ਅਚਲ uint8_t * grid  = (ਅਚਲ uint8_t *) (iq2xxs_grid + aux8[il]);
        ਅਚਲ uint8_t   signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];
        ਦਸ਼ਮਲਵ t[8];
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ j = 0; j < 8; ++j) {
            t[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ E>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ fg_stage_a(ਅਚਲ ਖੰਡ_iq2_xs * __restrict__ xrow, ਅਚਲ ਪੂਰਨ kb, ਕਿਸਮ_ਨਾਂ E::pair * a) {
    FG_SUPERBLOCK(ਖੰਡ_iq2_xs);
    ਅਚਲ uint16_t * q2 = blk->qs + 4 * ib;
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ il = 0; il < 4; ++il) {
        ਅਚਲ uint8_t * grid  = (ਅਚਲ uint8_t *) (iq2xs_grid + (q2[il] & 511));
        ਅਚਲ ਦਸ਼ਮਲਵ     d     = (ਦਸ਼ਮਲਵ) blk->d * (0.5f + ((blk->scales[ib] >> (4 * (il / 2))) & 0xf)) * 0.25f;
        ਅਚਲ uint8_t   signs = ksigns_iq2xs[q2[il] >> 9];
        ਦਸ਼ਮਲਵ t[8];
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ j = 0; j < 8; ++j) {
            t[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ E>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ fg_stage_a(ਅਚਲ ਖੰਡ_iq2_s * __restrict__ xrow, ਅਚਲ ਪੂਰਨ kb, ਕਿਸਮ_ਨਾਂ E::pair * a) {
    FG_SUPERBLOCK(ਖੰਡ_iq2_s);
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ il = 0; il < 4; ++il) {
        ਅਚਲ uint8_t * grid =
            (ਅਚਲ uint8_t *) (iq2s_grid + (blk->qs[4 * ib + il] | ((blk->qh[ib] << (8 - 2 * il)) & 0x300)));
        ਅਚਲ ਦਸ਼ਮਲਵ   d     = (ਦਸ਼ਮਲਵ) blk->d * (0.5f + ((blk->scales[ib] >> (4 * (il / 2))) & 0xf)) * 0.25f;
        ਅਚਲ uint8_t signs = blk->qs[QK_K / 8 + 4 * ib + il];
        ਦਸ਼ਮਲਵ t[8];
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ j = 0; j < 8; ++j) {
            t[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ E>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ fg_stage_a(ਅਚਲ ਖੰਡ_iq1_s * __restrict__ xrow, ਅਚਲ ਪੂਰਨ kb, ਕਿਸਮ_ਨਾਂ E::pair * a) {
    FG_SUPERBLOCK(ਖੰਡ_iq1_s);
    ਅਚਲ ਦਸ਼ਮਲਵ delta = blk->qh[ib] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
    ਅਚਲ ਦਸ਼ਮਲਵ d     = (ਦਸ਼ਮਲਵ) blk->d * (2 * ((blk->qh[ib] >> 12) & 7) + 1);
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ il = 0; il < 4; ++il) {
        uint32_t       grid32[2];
        ਅਚਲ int8_t * q = (ਅਚਲ int8_t *) grid32;
        grid32[0] = iq1s_grid_gpu[blk->qs[4 * ib + il] | (((blk->qh[ib] >> (3 * il)) & 7) << 8)];
        grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
        grid32[0] &= 0x0f0f0f0f;
        ਦਸ਼ਮਲਵ t[8];
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ j = 0; j < 8; ++j) {
            t[j] = d * (q[j] + delta);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ E>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ fg_stage_a(ਅਚਲ ਖੰਡ_iq1_m * __restrict__ xrow, ਅਚਲ ਪੂਰਨ kb, ਕਿਸਮ_ਨਾਂ E::pair * a) {
    FG_SUPERBLOCK(ਖੰਡ_iq1_m);
    ਅਚਲ uint16_t * sc = (ਅਚਲ uint16_t *) blk->scales;
    iq1m_scale_t     scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ il = 0; il < 4; ++il) {
        ਅਚਲ ਪੂਰਨ   ib16  = 2 * ib + il / 2;
        ਅਚਲ ਦਸ਼ਮਲਵ d     = (ਦਸ਼ਮਲਵ) scale.f16 * (2 * ((sc[ib16 / 4] >> (3 * (ib16 % 4))) & 0x7) + 1);
        ਅਚਲ ਦਸ਼ਮਲਵ delta = blk->qh[2 * ib + il / 2] & (0x08 << (4 * (il % 2))) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
        uint32_t       grid32[2];
        ਅਚਲ int8_t * q = (ਅਚਲ int8_t *) grid32;
        grid32[0] = iq1s_grid_gpu[blk->qs[4 * ib + il] |
                                  (((blk->qh[2 * ib + il / 2] >> (4 * (il % 2))) & 7) << 8)];
        grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
        grid32[0] &= 0x0f0f0f0f;
        ਦਸ਼ਮਲਵ t[8];
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ j = 0; j < 8; ++j) {
            t[j] = d * (q[j] + delta);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}


// q8_0 and the k-quants. Each decode takes the fields of one 32-value sub-block, so the canonical
// layout and the reorder (SoA) layout of reorder_qw() share it and differ only in where the fields
// live, as in dequantize.hpp.
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ E>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ fg_decode_q8_0(ਅਚਲ int8_t * __restrict__ qs, ਅਚਲ ਦਸ਼ਮਲਵ d, ਕਿਸਮ_ਨਾਂ E::pair * a) {
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ j = 0; j < QK8_0 / 2; ++j) {
        a[j] = E::make(d * qs[2 * j], d * qs[2 * j + 1]);
    }
}

// same unpack as get_scale_min_k4() in dequantize.hpp
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ fg_scale_min_k4(ਅਚਲ ਪੂਰਨ j, ਅਚਲ uint8_t * __restrict__ q, uint8_t & d, uint8_t & m) {
    ਜੇ (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } ਨਹੀਂਤਾਂ {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

// q4_K sub-block ib (0..7) uses scale/min pair ib and the low (even ib) or high (odd ib) nibbles of
// qs[32 * (ib / 2) ...], as in dequantize_row_q4_K
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ E>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ fg_decode_q4_K(ਅਚਲ uint8_t * __restrict__ qs, ਅਚਲ uint8_t * __restrict__ scales,
                                           ਅਚਲ sycl::half2 dm, ਅਚਲ ਪੂਰਨ ib, ਕਿਸਮ_ਨਾਂ E::pair * a) {
    uint8_t sc, mb;
    fg_scale_min_k4(ib, scales, sc, mb);
    ਅਚਲ ਦਸ਼ਮਲਵ     d     = (ਦਸ਼ਮਲਵ) dm[0] * sc;
    ਅਚਲ ਦਸ਼ਮਲਵ     m     = (ਦਸ਼ਮਲਵ) dm[1] * mb;
    ਅਚਲ uint8_t * q     = qs + 32 * (ib / 2);
    ਅਚਲ ਪੂਰਨ       ਖਿਸਕਾਓ = 4 * (ib % 2);
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ j = 0; j < 16; ++j) {
        a[j] = E::make(d * ((q[2 * j] >> ਖਿਸਕਾਓ) & 0xF) - m, d * ((q[2 * j + 1] >> ਖਿਸਕਾਓ) & 0xF) - m);
    }
}

// q5_K: q4_K plus one high bit per value, bit ib of qh[l]
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ E>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ fg_decode_q5_K(ਅਚਲ uint8_t * __restrict__ qs, ਅਚਲ uint8_t * __restrict__ qh,
                                           ਅਚਲ uint8_t * __restrict__ scales, ਅਚਲ sycl::half2 dm, ਅਚਲ ਪੂਰਨ ib,
                                           ਕਿਸਮ_ਨਾਂ E::pair * a) {
    uint8_t sc, mb;
    fg_scale_min_k4(ib, scales, sc, mb);
    ਅਚਲ ਦਸ਼ਮਲਵ     d     = (ਦਸ਼ਮਲਵ) dm[0] * sc;
    ਅਚਲ ਦਸ਼ਮਲਵ     m     = (ਦਸ਼ਮਲਵ) dm[1] * mb;
    ਅਚਲ uint8_t * q     = qs + 32 * (ib / 2);
    ਅਚਲ ਪੂਰਨ       ਖਿਸਕਾਓ = 4 * (ib % 2);
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ j = 0; j < 16; ++j) {
        ਅਚਲ ਪੂਰਨ l = 2 * j;
        a[j] = E::make(d * (((q[l] >> ਖਿਸਕਾਓ) & 0xF) | (((qh[l] >> ib) & 1) << 4)) - m,
                       d * (((q[l + 1] >> ਖਿਸਕਾਓ) & 0xF) | (((qh[l + 1] >> ib) & 1) << 4)) - m);
    }
}

// q6_K: sub-block ib is quarter r = ib % 4 of half h = ib / 4. The half selects ql + 64h, qh + 32h
// and scales + 8h; the quarter selects ql + 32(r & 1), the ql nibble r / 2, the qh bit pair r and
// scales + 2r, as in dequantize_row_q6_K. The scale changes at value 16 of the sub-block.
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ E>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ fg_decode_q6_K(ਅਚਲ uint8_t * __restrict__ ql, ਅਚਲ uint8_t * __restrict__ qh,
                                           ਅਚਲ int8_t * __restrict__ scales, ਅਚਲ ਦਸ਼ਮਲਵ d, ਅਚਲ ਪੂਰਨ ib,
                                           ਕਿਸਮ_ਨਾਂ E::pair * a) {
    ਅਚਲ ਪੂਰਨ       h  = ib / 4;
    ਅਚਲ ਪੂਰਨ       r  = ib % 4;
    ਅਚਲ uint8_t * q  = ql + 64 * h + 32 * (r & 1);
    ਅਚਲ uint8_t * hb = qh + 32 * h;
    ਅਚਲ int8_t *  sc = scales + 8 * h + 2 * r;
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ j = 0; j < 16; ++j) {
        ਅਚਲ ਪੂਰਨ   l  = 2 * j;
        ਅਚਲ ਦਸ਼ਮਲਵ dl = d * sc[j / 8];
        ਅਚਲ ਪੂਰਨ   q0 = (((q[l] >> (4 * (r / 2))) & 0xF) | (((hb[l] >> (2 * r)) & 3) << 4)) - 32;
        ਅਚਲ ਪੂਰਨ   q1 = (((q[l + 1] >> (4 * (r / 2))) & 0xF) | (((hb[l + 1] >> (2 * r)) & 3) << 4)) - 32;
        a[j] = E::make(dl * q0, dl * q1);
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ E>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ fg_stage_a(ਅਚਲ ਖੰਡ_q8_0 * __restrict__ xrow, ਅਚਲ ਪੂਰਨ kb, ਕਿਸਮ_ਨਾਂ E::pair * a) {
    fg_decode_q8_0<E>(xrow[kb].qs, (ਦਸ਼ਮਲਵ) xrow[kb].d, a);
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ E>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ fg_stage_a(ਅਚਲ ਖੰਡ_q4_K * __restrict__ xrow, ਅਚਲ ਪੂਰਨ kb, ਕਿਸਮ_ਨਾਂ E::pair * a) {
    FG_SUPERBLOCK(ਖੰਡ_q4_K);
    fg_decode_q4_K<E>(blk->qs, blk->scales, blk->dm, ib, a);
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ E>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ fg_stage_a(ਅਚਲ ਖੰਡ_q5_K * __restrict__ xrow, ਅਚਲ ਪੂਰਨ kb, ਕਿਸਮ_ਨਾਂ E::pair * a) {
    FG_SUPERBLOCK(ਖੰਡ_q5_K);
    fg_decode_q5_K<E>(blk->qs, blk->qh, blk->scales, blk->dm, ib, a);
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ E>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ fg_stage_a(ਅਚਲ ਖੰਡ_q6_K * __restrict__ xrow, ਅਚਲ ਪੂਰਨ kb, ਕਿਸਮ_ਨਾਂ E::pair * a) {
    FG_SUPERBLOCK(ਖੰਡ_q6_K);
    fg_decode_q6_K<E>(blk->ql, blk->qh, blk->scales, (ਦਸ਼ਮਲਵ) blk->d, ib, a);
}

// Reorder (SoA) layout: each block field is one stream over the nblocks of the matrix (of the expert
// slice for MUL_MAT_ID), in the order reorder_qw() writes them. Block ib holds k step kb.
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ ਖੰਡ_q_t> ਬਣਤਰ fg_soa {
    ਸਥਿਰ constexpr ਤਾਰਕਿਕ supported = ਝੂਠ;
};

ਸਾਂਚਾ <> ਬਣਤਰ fg_soa<ਖੰਡ_q8_0> {
    ਸਥਿਰ constexpr ਤਾਰਕਿਕ supported = ਸੱਚ;
    // [qs][d]
    ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ E>
    ਸਥਿਰ __dpct_inline__ ਖਾਲੀ stage(ਅਚਲ uint8_t * x, ਅਚਲ size_t nblocks, ਅਚਲ size_t ib, ਅਚਲ ਪੂਰਨ,
                                      ਕਿਸਮ_ਨਾਂ E::pair * a) {
        ਅਚਲ ਦਸ਼ਮਲਵ d = (ਦਸ਼ਮਲਵ) ((ਅਚਲ sycl::half *) (x + nblocks * QK8_0))[ib];
        fg_decode_q8_0<E>((ਅਚਲ int8_t *) x + ib * QK8_0, d, a);
    }
};

ਸਾਂਚਾ <> ਬਣਤਰ fg_soa<ਖੰਡ_q4_K> {
    ਸਥਿਰ constexpr ਤਾਰਕਿਕ supported = ਸੱਚ;
    // [qs][scales][dm]
    ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ E>
    ਸਥਿਰ __dpct_inline__ ਖਾਲੀ stage(ਅਚਲ uint8_t * x, ਅਚਲ size_t nblocks, ਅਚਲ size_t ib, ਅਚਲ ਪੂਰਨ kb,
                                      ਕਿਸਮ_ਨਾਂ E::pair * a) {
        ਅਚਲ uint8_t *   scales = x + nblocks * (QK_K / 2);
        ਅਚਲ sycl::half2 dm     = ((ਅਚਲ sycl::half2 *) (scales + nblocks * K_SCALE_ਮਾਪ))[ib];
        fg_decode_q4_K<E>(x + ib * (QK_K / 2), scales + ib * K_SCALE_ਮਾਪ, dm, kb % (QK_K / 32), a);
    }
};

ਸਾਂਚਾ <> ਬਣਤਰ fg_soa<ਖੰਡ_q5_K> {
    ਸਥਿਰ constexpr ਤਾਰਕਿਕ supported = ਸੱਚ;
    // [qs][qh][scales][dm]
    ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ E>
    ਸਥਿਰ __dpct_inline__ ਖਾਲੀ stage(ਅਚਲ uint8_t * x, ਅਚਲ size_t nblocks, ਅਚਲ size_t ib, ਅਚਲ ਪੂਰਨ kb,
                                      ਕਿਸਮ_ਨਾਂ E::pair * a) {
        ਅਚਲ uint8_t *   qh     = x + nblocks * (QK_K / 2);
        ਅਚਲ uint8_t *   scales = qh + nblocks * (QK_K / 8);
        ਅਚਲ sycl::half2 dm     = ((ਅਚਲ sycl::half2 *) (scales + nblocks * K_SCALE_ਮਾਪ))[ib];
        fg_decode_q5_K<E>(x + ib * (QK_K / 2), qh + ib * (QK_K / 8), scales + ib * K_SCALE_ਮਾਪ, dm,
                          kb % (QK_K / 32), a);
    }
};

ਸਾਂਚਾ <> ਬਣਤਰ fg_soa<ਖੰਡ_q6_K> {
    ਸਥਿਰ constexpr ਤਾਰਕਿਕ supported = ਸੱਚ;
    // [ql][qh][scales][d]
    ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ E>
    ਸਥਿਰ __dpct_inline__ ਖਾਲੀ stage(ਅਚਲ uint8_t * x, ਅਚਲ size_t nblocks, ਅਚਲ size_t ib, ਅਚਲ ਪੂਰਨ kb,
                                      ਕਿਸਮ_ਨਾਂ E::pair * a) {
        ਅਚਲ uint8_t * qh     = x + nblocks * (QK_K / 2);
        ਅਚਲ uint8_t * scales = qh + nblocks * (QK_K / 4);
        ਅਚਲ ਦਸ਼ਮਲਵ     d      = (ਦਸ਼ਮਲਵ) ((ਅਚਲ sycl::half *) (scales + nblocks * (QK_K / 16)))[ib];
        fg_decode_q6_K<E>(x + ib * (QK_K / 2), qh + ib * (QK_K / 4), (ਅਚਲ int8_t *) scales + ib * (QK_K / 16), d,
                          kb % (QK_K / 32), a);
    }
};


// one SG_ROWS x BN output tile: B columns [b0, b0 + BN) of packed_b go to dst columns [n0, n1),
// n1 - n0 <= BN
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ S, ਕਿਸਮ_ਨਾਂ ਖੰਡ_q_t, ਤਾਰਕਿਕ reordered>
ਸਥਿਰ ਖਾਲੀ fused_dequant_gemm_tile(
    ਅਚਲ ਖੰਡ_q_t * __restrict__ x,
    ਅਚਲ ਕਿਸਮ_ਨਾਂ S::tsb * __restrict__ packed_b,
    ਦਸ਼ਮਲਵ * __restrict__ dst,
    ਅਚਲ ਪੂਰਨ M, ਅਚਲ ਪੂਰਨ Npad, ਅਚਲ ਪੂਰਨ K, ਅਚਲ ਪੂਰਨ ldd,
    ਅਚਲ ਪੂਰਨ b0, ਅਚਲ ਪੂਰਨ n0, ਅਚਲ ਪੂਰਨ n1,
    sycl::local_accessor<ਕਿਸਮ_ਨਾਂ S::tsa, 1> tile_a,
    sycl::local_accessor<ਦਸ਼ਮਲਵ, 1> tile_c,
    ਅਚਲ sycl::nd_ਵਸਤੂ<2> & ਵਸਤੂ) {
    ਜੇ constexpr (fg_built<S::SG>()) {
        ਵਰਤੋਂ EA = ਕਿਸਮ_ਨਾਂ S::EA;
        ਵਰਤੋਂ TA = ਕਿਸਮ_ਨਾਂ S::ta;
        ਵਰਤੋਂ TB = ਕਿਸਮ_ਨਾਂ S::tb;
        ਅਚਲ ਸਵੈਚਾਲਿਤ sg     = ਵਸਤੂ.get_sub_ਸਮੂਹ();
        ਅਚਲ ਪੂਰਨ  sg_id  = sg.get_ਸਮੂਹ_id()[0];
        ਅਚਲ ਪੂਰਨ  lane   = sg.get_local_id()[0];
        ਅਚਲ ਪੂਰਨ  m0     = ਵਸਤੂ.get_ਸਮੂਹ(1) * S::SG_ਪੰਕਤੀਆਂ;
        ਅਚਲ ਪੂਰਨ  nstep  = K / FG_BK;
        ਅਚਲ ਪੂਰਨ  a_ਅਧਾਰ = sg_id * S::SG_ਪੰਕਤੀਆਂ * FG_BK;
        ਅਚਲ ਪੂਰਨ  c_ਅਧਾਰ = sg_id * S::SG_ਪੰਕਤੀਆਂ * S::BN;

        mx::joint_ਮੈਟ੍ਰਿਕਸ<sycl::sub_ਸਮੂਹ, ਦਸ਼ਮਲਵ, mx::use::ਸੰਚਾਇਕ, S::TM, S::TN> acc[S::MT][S::NT];
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ mt = 0; mt < S::MT; ++mt) {
#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ nt = 0; nt < S::NT; ++nt) {
                mx::joint_ਮੈਟ੍ਰਿਕਸ_fill(sg, acc[mt][nt], 0.0f);
            }
        }

        // lane decodes rows lane, lane + SG, ... of the sub-group's SG_ROWS
        constexpr ਪੂਰਨ         KPB     = fg_ਖੰਡ_traits<ਖੰਡ_q_t>::qk / FG_BK; // k steps per block
        ਅਚਲ size_t          bpr     = K / fg_ਖੰਡ_traits<ਖੰਡ_q_t>::qk;
        ਅਚਲ size_t          nblocks = (size_t) M * bpr;
        size_t                ਪੰਕਤੀ_blk[S::RPL];
        ਤਾਰਕਿਕ                  ਪੰਕਤੀ_ok[S::RPL];
        ਕਿਸਮ_ਨਾਂ EA::pair *   a[S::RPL];
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ r = 0; r < S::RPL; ++r) {
            ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = m0 + r * S::SG + lane;
            ਪੰਕਤੀ_ok[r]  = ਪੰਕਤੀ < M;
            ਪੰਕਤੀ_blk[r] = (size_t) (ਪੰਕਤੀ_ok[r] ? ਪੰਕਤੀ : 0) * bpr;
            a[r]       = (ਕਿਸਮ_ਨਾਂ EA::pair *) &tile_a[a_ਅਧਾਰ + (r * S::SG + lane) * FG_BK];
        }

        ਅਚਲ ਸਵੈਚਾਲਿਤ b_ptr = sycl::ਪਤਾ_space_cast<sycl::access::ਪਤਾ_space::global_space,
                                                    sycl::access::decorated::no>(packed_b);
        ਅਚਲ ਪੂਰਨ b_stride = Npad * S::VNNI;

        ਅਚਲ ਪੂਰਨ kb_begin = (sg_id * nstep) / FG_KSPLIT;
        ਅਚਲ ਪੂਰਨ kb_ਸਿਰਾ   = ((sg_id + 1) * nstep) / FG_KSPLIT;
        ਲਈ (ਪੂਰਨ kb = kb_begin; kb < kb_ਸਿਰਾ; ++kb) {
#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ r = 0; r < S::RPL; ++r) {
                ਜੇ (ਪੰਕਤੀ_ok[r]) {
                    ਜੇ constexpr (reordered) {
                        fg_soa<ਖੰਡ_q_t>::ਸਾਂਚਾ stage<EA>((ਅਚਲ uint8_t *) x, nblocks, ਪੰਕਤੀ_blk[r] + kb / KPB, kb,
                                                              a[r]);
                    } ਨਹੀਂਤਾਂ {
                        fg_stage_a<EA>(x + ਪੰਕਤੀ_blk[r], kb, a[r]);
                    }
                } ਨਹੀਂਤਾਂ {
#ਹਦਾਇਤ unroll
                    ਲਈ (ਪੂਰਨ j = 0; j < FG_BK / 2; ++j) {
                        a[r][j] = EA::make(0.0f, 0.0f);
                    }
                }
            }
            sycl::ਸਮੂਹ_barrier(sg);

#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ kt = 0; kt < FG_BK / S::TK; ++kt) {
                ਅਚਲ ਪੂਰਨ kq0 = (kb * FG_BK + kt * S::TK) / S::VNNI;
                mx::joint_ਮੈਟ੍ਰਿਕਸ<sycl::sub_ਸਮੂਹ, TB, mx::use::b, S::TK, S::TN, S::b_layout> sub_b[S::NT];
#ਹਦਾਇਤ unroll
                ਲਈ (ਪੂਰਨ nt = 0; nt < S::NT; ++nt) {
                    mx::joint_ਮੈਟ੍ਰਿਕਸ_ਲੱਦੋ(sg, sub_b[nt], b_ptr + (size_t) kq0 * b_stride + (b0 + nt * S::TN) * S::VNNI, b_stride);
                }
#ਹਦਾਇਤ unroll
                ਲਈ (ਪੂਰਨ mt = 0; mt < S::MT; ++mt) {
                    mx::joint_ਮੈਟ੍ਰਿਕਸ<sycl::sub_ਸਮੂਹ, TA, mx::use::a, S::TM, S::TK, mx::layout::ਪੰਕਤੀ_major> sub_a;
                    mx::joint_ਮੈਟ੍ਰਿਕਸ_ਲੱਦੋ(sg, sub_a,
                        tile_a.ਸਾਂਚਾ get_multi_ptr<sycl::access::decorated::no>() + a_ਅਧਾਰ + (mt * S::TM) * FG_BK + kt * S::TK,
                        FG_BK);
#ਹਦਾਇਤ unroll
                    ਲਈ (ਪੂਰਨ nt = 0; nt < S::NT; ++nt) {
                        mx::joint_ਮੈਟ੍ਰਿਕਸ_mad(sg, acc[mt][nt], sub_a, sub_b[nt], acc[mt][nt]);
                    }
                }
            }
            // the next step overwrites tile_a
            sycl::ਸਮੂਹ_barrier(sg);
        }

#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ mt = 0; mt < S::MT; ++mt) {
#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ nt = 0; nt < S::NT; ++nt) {
                mx::joint_ਮੈਟ੍ਰਿਕਸ_store(sg, acc[mt][nt],
                    tile_c.ਸਾਂਚਾ get_multi_ptr<sycl::access::decorated::no>() + c_ਅਧਾਰ + (mt * S::TM) * S::BN + nt * S::TN,
                    S::BN, mx::layout::ਪੰਕਤੀ_major);
            }
        }
        sycl::ਸਮੂਹ_barrier(ਵਸਤੂ.get_ਸਮੂਹ());

        // sum the K splits; consecutive lanes write consecutive rows of one dst column
        ਲਈ (ਪੂਰਨ idx = ਵਸਤੂ.get_local_ਰੇਖਿਕ_id(); idx < S::SG_ਪੰਕਤੀਆਂ * S::BN; idx += S::WG_ਮਾਪ) {
            ਅਚਲ ਪੂਰਨ r = idx % S::SG_ਪੰਕਤੀਆਂ;
            ਅਚਲ ਪੂਰਨ c = idx / S::SG_ਪੰਕਤੀਆਂ;
            ਅਚਲ ਪੂਰਨ m = m0 + r;
            ਅਚਲ ਪੂਰਨ n = n0 + c;
            ਜੇ (m < M && n < n1) {
                ਦਸ਼ਮਲਵ ਜੋੜ = 0.0f;
#ਹਦਾਇਤ unroll
                ਲਈ (ਪੂਰਨ s = 0; s < FG_KSPLIT; ++s) {
                    ਜੋੜ += tile_c[s * S::SG_ਪੰਕਤੀਆਂ * S::BN + r * S::BN + c];
                }
                dst[(size_t) n * ldd + m] = ਜੋੜ;
            }
        }
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ S, ਕਿਸਮ_ਨਾਂ ਖੰਡ_q_t, ਤਾਰਕਿਕ reordered = ਝੂਠ>
ਸਥਿਰ ਖਾਲੀ fused_dequant_gemm_launch(ਅਚਲ ਖਾਲੀ * src0, ਅਚਲ ਕਿਸਮ_ਨਾਂ S::tsb * packed, ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਪੂਰਨ M,
                                      ਅਚਲ ਪੂਰਨ N, ਅਚਲ ਪੂਰਨ Npad, ਅਚਲ ਪੂਰਨ K, ਅਚਲ ਪੂਰਨ ldd,
                                      ਅਚਲ int64_t groups_n, ਅਚਲ int64_t groups_m, dpct::ਕਤਾਰ_ptr stream) {
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<ਕਿਸਮ_ਨਾਂ S::tsa, 1> tile_a(FG_KSPLIT * S::SG_ਪੰਕਤੀਆਂ * FG_BK, cgh);
        sycl::local_accessor<ਦਸ਼ਮਲਵ, 1>          tile_c(FG_KSPLIT * S::SG_ਪੰਕਤੀਆਂ * S::BN, cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(groups_n, groups_m * S::WG_ਮਾਪ), sycl::range<2>(1, S::WG_ਮਾਪ)),
            [=](sycl::nd_ਵਸਤੂ<2> ਵਸਤੂ) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(S::SG)]] {
                ਅਚਲ ਪੂਰਨ n0 = ਵਸਤੂ.get_ਸਮੂਹ(0) * S::BN;
                fused_dequant_gemm_tile<S, ਖੰਡ_q_t, reordered>((ਅਚਲ ਖੰਡ_q_t *) src0, packed, dst, M, Npad, K,
                                                      ldd, n0, n0, N, tile_a, tile_c, ਵਸਤੂ);
            });
    });
}

// grouped: work-group (t, mt) is tile t of the schedule; its B columns sit at t * BN
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ S, ਕਿਸਮ_ਨਾਂ ਖੰਡ_q_t, ਤਾਰਕਿਕ reordered>
ਸਥਿਰ ਖਾਲੀ grouped_dequant_gemm_launch(ਅਚਲ ਅੱਖਰ * src0_dd, ਅਚਲ size_t expert_stride,
                                        ਅਚਲ ggml_sycl_gg_tile * tiles_ptr, ਅਚਲ ਕਿਸਮ_ਨਾਂ S::tsb * packed, ਦਸ਼ਮਲਵ * dst,
                                        ਅਚਲ ਪੂਰਨ M, ਅਚਲ ਪੂਰਨ Npad, ਅਚਲ ਪੂਰਨ K, ਅਚਲ int64_t n_tiles,
                                        ਅਚਲ int64_t groups_m, dpct::ਕਤਾਰ_ptr stream) {
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<ਕਿਸਮ_ਨਾਂ S::tsa, 1> tile_a(FG_KSPLIT * S::SG_ਪੰਕਤੀਆਂ * FG_BK, cgh);
        sycl::local_accessor<ਦਸ਼ਮਲਵ, 1>          tile_c(FG_KSPLIT * S::SG_ਪੰਕਤੀਆਂ * S::BN, cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(n_tiles, groups_m * S::WG_ਮਾਪ), sycl::range<2>(1, S::WG_ਮਾਪ)),
            [=](sycl::nd_ਵਸਤੂ<2> ਵਸਤੂ) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(S::SG)]] {
                ਅਚਲ ਪੂਰਨ               t    = ਵਸਤੂ.get_ਸਮੂਹ(0);
                ਅਚਲ ggml_sycl_gg_tile tile = tiles_ptr[t];
                ਅਚਲ ਖੰਡ_q_t *       x    = (ਅਚਲ ਖੰਡ_q_t *) (src0_dd + (size_t) tile.expert * expert_stride);
                fused_dequant_gemm_tile<S, ਖੰਡ_q_t, reordered>(x, packed, dst, M, Npad, K, M, t * S::BN, tile.n0,
                                                                 tile.n1, tile_a, tile_c, ਵਸਤੂ);
            });
    });
}

// src1 f32 rows -> packed [K/V][n_tiles*BN][V], tile t holds its rows [n0, n1) at columns t*BN..,
// zero past n1. The column runs fastest so a sub-group writes one contiguous run.
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ S>
ਸਥਿਰ ਖਾਲੀ grouped_gemm_pack_b(ਅਚਲ ਦਸ਼ਮਲਵ * y, ਕਿਸਮ_ਨਾਂ S::tsb * packed, ਅਚਲ ggml_sycl_gg_tile * tiles, ਪੂਰਨ Npad,
                                ਪੂਰਨ K, dpct::ਕਤਾਰ_ptr stream) {
    ਵਰਤੋਂ E        = ਕਿਸਮ_ਨਾਂ S::EB;
    constexpr ਪੂਰਨ V = S::VNNI;
    ਅਚਲ ਪੂਰਨ kqs   = K / V;
    stream->parallel_for(sycl::range<1>((size_t) Npad * kqs), [=](sycl::id<1> id) {
        ਅਚਲ size_t idx = id[0];
        ਅਚਲ ਪੂਰਨ    kq  = idx / Npad;
        ਅਚਲ ਪੂਰਨ    n   = idx - (size_t) kq * Npad;
        ਅਚਲ ggml_sycl_gg_tile tile = tiles[n / S::BN];
        ਅਚਲ ਪੂਰਨ    ਪੰਕਤੀ = tile.n0 + n % S::BN;
        // one guarded load run per work-item, as a per-element select costs ~1.5% prefill
        ਕਿਸਮ_ਨਾਂ S::tsb vals[V] = {};
        ਜੇ (ਪੰਕਤੀ < tile.n1) {
            ਅਚਲ ਦਸ਼ਮਲਵ * src = y + (size_t) ਪੰਕਤੀ * K + V * kq;
#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ v = 0; v < V; ++v) {
                vals[v] = E::cvt(src[v]);
            }
        }
        ਕਿਸਮ_ਨਾਂ S::tsb * ਬਾਹਰਲਾ = packed + ((size_t) kq * Npad + n) * V;
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ v = 0; v < V; ++v) {
            ਬਾਹਰਲਾ[v] = vals[v];
        }
    });
}

// q8_0 and the k-quants take only the grouped path. The plain kernel decodes A again for every BN columns
// of a dense batch, and for these formats that costs more than the one dequantization of the library GEMM.
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T> ਸਥਿਰ constexpr ਤਾਰਕਿਕ fg_plain_ok() {
    ਵਾਪਸ !std::is_same_v<T, ਖੰਡ_q8_0> && !std::is_same_v<T, ਖੰਡ_q4_K> && !std::is_same_v<T, ਖੰਡ_q5_K> &&
           !std::is_same_v<T, ਖੰਡ_q6_K>;
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T, ਤਾਰਕਿਕ R> ਬਣਤਰ fg_ਟੈਗ {
    ਵਰਤੋਂ type                     = T;
    ਸਥਿਰ constexpr ਤਾਰਕਿਕ reordered = R;
};

// calls f(fg_tag<block_q_t, reordered>{}) for the weight format and layout; false if it has no A stage
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T, ਕਿਸਮ_ਨਾਂ F> ਸਥਿਰ ਤਾਰਕਿਕ fg_ਦੇਖੋ_layout(ਤਾਰਕਿਕ reordered, F && f) {
    ਜੇ (!reordered) {
        f(fg_ਟੈਗ<T, ਝੂਠ>{});
        ਵਾਪਸ ਸੱਚ;
    }
    ਜੇ constexpr (fg_soa<T>::supported) {
        f(fg_ਟੈਗ<T, ਸੱਚ>{});
        ਵਾਪਸ ਸੱਚ;
    }
    ਵਾਪਸ ਝੂਠ;
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ F> ਸਥਿਰ ਤਾਰਕਿਕ fg_ਦੇਖੋ_type(ggml_type type, ਤਾਰਕਿਕ reordered, F && f) {
    ਚੋਣ (type) {
        ਹਾਲਤ GGML_TYPE_IQ4_NL:  ਵਾਪਸ fg_ਦੇਖੋ_layout<ਖੰਡ_iq4_nl>(reordered, f);
        ਹਾਲਤ GGML_TYPE_IQ3_S:   ਵਾਪਸ fg_ਦੇਖੋ_layout<ਖੰਡ_iq3_s>(reordered, f);
        ਹਾਲਤ GGML_TYPE_IQ4_XS:  ਵਾਪਸ fg_ਦੇਖੋ_layout<ਖੰਡ_iq4_xs>(reordered, f);
        ਹਾਲਤ GGML_TYPE_IQ3_XXS: ਵਾਪਸ fg_ਦੇਖੋ_layout<ਖੰਡ_iq3_xxs>(reordered, f);
        ਹਾਲਤ GGML_TYPE_IQ2_XXS: ਵਾਪਸ fg_ਦੇਖੋ_layout<ਖੰਡ_iq2_xxs>(reordered, f);
        ਹਾਲਤ GGML_TYPE_IQ2_XS:  ਵਾਪਸ fg_ਦੇਖੋ_layout<ਖੰਡ_iq2_xs>(reordered, f);
        ਹਾਲਤ GGML_TYPE_IQ2_S:   ਵਾਪਸ fg_ਦੇਖੋ_layout<ਖੰਡ_iq2_s>(reordered, f);
        ਹਾਲਤ GGML_TYPE_IQ1_S:   ਵਾਪਸ fg_ਦੇਖੋ_layout<ਖੰਡ_iq1_s>(reordered, f);
        ਹਾਲਤ GGML_TYPE_IQ1_M:   ਵਾਪਸ fg_ਦੇਖੋ_layout<ਖੰਡ_iq1_m>(reordered, f);
        ਹਾਲਤ GGML_TYPE_Q8_0:    ਵਾਪਸ fg_ਦੇਖੋ_layout<ਖੰਡ_q8_0>(reordered, f);
        ਹਾਲਤ GGML_TYPE_Q4_K:    ਵਾਪਸ fg_ਦੇਖੋ_layout<ਖੰਡ_q4_K>(reordered, f);
        ਹਾਲਤ GGML_TYPE_Q5_K:    ਵਾਪਸ fg_ਦੇਖੋ_layout<ਖੰਡ_q5_K>(reordered, f);
        ਹਾਲਤ GGML_TYPE_Q6_K:    ਵਾਪਸ fg_ਦੇਖੋ_layout<ਖੰਡ_q6_K>(reordered, f);
        ਮੂਲ:                ਵਾਪਸ ਝੂਠ;
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ S>
ਸਥਿਰ ਤਾਰਕਿਕ fg_fused_ਲੜੀਵਾਰ(ggml_type src0_type, ਤਾਰਕਿਕ src0_reordered, ਅਚਲ ਖਾਲੀ * src0, ਅਚਲ ਖਾਲੀ * src1, ggml_type src1_type,
                         ਦਸ਼ਮਲਵ * dst, int64_t M, int64_t N, int64_t K, int64_t ldd, ggml_sycl_pool & pool,
                         dpct::ਕਤਾਰ_ptr stream) {
    ਅਚਲ int64_t groups_n = (N + S::BN - 1) / S::BN;
    ਅਚਲ int64_t groups_m = (M + S::SG_ਪੰਕਤੀਆਂ - 1) / S::SG_ਪੰਕਤੀਆਂ;
    ਅਚਲ ਪੂਰਨ     Npad     = (ਪੂਰਨ) (groups_n * S::BN);

    // src1 is read in its own type: one pass, converted in registers only if B differs
    ggml_sycl_pool_alloc<ਕਿਸਮ_ਨਾਂ S::tsb> packed_b(pool, (size_t) K * Npad);
    ਜੇ (src1_type == GGML_TYPE_F16) {
        fused_gemm_pack_b<ਕਿਸਮ_ਨਾਂ S::EB>((ਅਚਲ sycl::half *) src1, packed_b.get(), (ਪੂਰਨ) N, Npad, (ਪੂਰਨ) K, stream);
    } ਨਹੀਂਤਾਂ ਜੇ (src1_type == GGML_TYPE_BF16) {
        fused_gemm_pack_b<ਕਿਸਮ_ਨਾਂ S::EB>((ਅਚਲ fg_bf16 *) src1, packed_b.get(), (ਪੂਰਨ) N, Npad, (ਪੂਰਨ) K, stream);
    } ਨਹੀਂਤਾਂ {
        fused_gemm_pack_b<ਕਿਸਮ_ਨਾਂ S::EB>((ਅਚਲ ਦਸ਼ਮਲਵ *) src1, packed_b.get(), (ਪੂਰਨ) N, Npad, (ਪੂਰਨ) K, stream);
    }

    ਅਚਲ ਕਿਸਮ_ਨਾਂ S::tsb * packed = packed_b.get();
    ਵਾਪਸ fg_ਦੇਖੋ_type(src0_type, src0_reordered, [&](ਸਵੈਚਾਲਿਤ ਟੈਗ) {
        ਵਰਤੋਂ ਖੰਡ_q_t = ਕਿਸਮ_ਨਾਂ decltype(ਟੈਗ)::type;
        ਜੇ constexpr (fg_plain_ok<ਖੰਡ_q_t>()) {
            fused_dequant_gemm_launch<S, ਖੰਡ_q_t, decltype(ਟੈਗ)::reordered>(src0, packed, dst, (ਪੂਰਨ) M, (ਪੂਰਨ) N,
                                                                              Npad, (ਪੂਰਨ) K, (ਪੂਰਨ) ldd, groups_n,
                                                                              groups_m, stream);
        }
    });
}

ਤਾਰਕਿਕ ggml_sycl_fused_dequant_gemm(ggml_type src0_type, ਤਾਰਕਿਕ src0_reordered, ਅਚਲ ਖਾਲੀ * src0, ਅਚਲ ਖਾਲੀ * src1, ggml_type src1_type,
                                  int32_t src1_prec, ਦਸ਼ਮਲਵ * dst, int64_t M, int64_t N, int64_t K, int64_t ldd,
                                  ggml_sycl_pool & pool, dpct::ਕਤਾਰ_ptr stream) {
    // every BN columns dequantize A again, so wide N is left to the library GEMM.
    // src0_reordered: the persistent weight reorder may have been installed on this
    // tensor by an earlier small-batch mul_mat; the A stage then has to decode the
    // SoA layout. Types without an SoA stage (fg_soa<T>::supported == false)
    // decline via fg_visit_layout and the caller falls through to the
    // reorder-aware conversion route.
    ਤਾਰਕਿਕ plain_ok = ਝੂਠ;
    fg_ਦੇਖੋ_type(src0_type, src0_reordered, [&](ਸਵੈਚਾਲਿਤ ਟੈਗ) { plain_ok = fg_plain_ok<ਕਿਸਮ_ਨਾਂ decltype(ਟੈਗ)::type>(); });
    ਜੇ (g_ggml_sycl_dynamic_precision == GGML_SYCL_DYNAMIC_PRECISION_F32 ||
        !ggml_sycl_xmx_gather_type_enabled(src0_type) || !plain_ok) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (src1_type != GGML_TYPE_F32 && src1_type != GGML_TYPE_F16 && src1_type != GGML_TYPE_BF16) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (!ggml_sycl_fused_dequant_gemm_shape_ok(src0_type, M, N, K, ldd)) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਅਚਲ ਪੂਰਨ combo = fg_pick_combo(stream, src1_type, src1_prec);
    ਜੇ (combo < 0) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਤਾਰਕਿਕ launched = ਝੂਠ;
    fg_ਦੇਖੋ_combo(combo, [&](ਸਵੈਚਾਲਿਤ s) {
        launched = fg_fused_ਲੜੀਵਾਰ<decltype(s)>(src0_type, src0_reordered, src0, src1, src1_type, dst, M, N, K, ldd, pool, stream);
    });
    ਵਾਪਸ launched;
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ S>
ਸਥਿਰ ਤਾਰਕਿਕ fg_grouped_ਲੜੀਵਾਰ(ggml_type src0_type, ਤਾਰਕਿਕ reordered, ਅਚਲ ਖਾਲੀ * src0_ਅਧਾਰ, size_t expert_stride,
                           ਅਚਲ ਦਸ਼ਮਲਵ * src1, ਦਸ਼ਮਲਵ * dst, ਅਚਲ int64_t * expert_ਪੰਕਤੀ_offsets, int64_t n_as, int64_t M,
                           int64_t K, std::vector<ggml_sycl_gg_tile> & tiles, ggml_sycl_pool & pool,
                           dpct::ਕਤਾਰ_ptr stream) {
    // the host knows every slice, so it lays out the work-groups: no search on the device
    tiles.clear();
    ਲਈ (int64_t e = 0; e < n_as; ++e) {
        ਅਚਲ int64_t end = expert_ਪੰਕਤੀ_offsets[e + 1];
        ਲਈ (int64_t n0 = expert_ਪੰਕਤੀ_offsets[e]; n0 < end; n0 += S::BN) {
            tiles.push_back({ (int32_t) e, (int32_t) n0, (int32_t) std::min<int64_t>(n0 + S::BN, end) });
        }
    }
    ਅਚਲ int64_t n_tiles  = tiles.size();
    ਅਚਲ int64_t groups_m = (M + S::SG_ਪੰਕਤੀਆਂ - 1) / S::SG_ਪੰਕਤੀਆਂ;
    ਅਚਲ ਪੂਰਨ     Npad     = (ਪੂਰਨ) (n_tiles * S::BN);

    ਅਚਲ int64_t max_tiles = grouped_gemm_max_tiles(expert_ਪੰਕਤੀ_offsets[n_as], n_as, S::BN);
    GGML_ASSERT(n_tiles <= max_tiles);
    ggml_sycl_pool_alloc<ggml_sycl_gg_tile> tiles_dev(pool, max_tiles);
    SYCL_CHECK(CHECK_TRY_ERROR(stream->ਸਮ੍ਰਿਤੀ_ਨਕਲ(tiles_dev.get(), tiles.ਅੰਕੜੇ(), n_tiles * ਆਕਾਰ(ggml_sycl_gg_tile))));

    ggml_sycl_pool_alloc<ਕਿਸਮ_ਨਾਂ S::tsb> packed_b(pool, (size_t) K * max_tiles * S::BN);
    grouped_gemm_pack_b<S>(src1, packed_b.get(), tiles_dev.get(), Npad, (ਪੂਰਨ) K, stream);

    ਅਚਲ ਕਿਸਮ_ਨਾਂ S::tsb *   packed    = packed_b.get();
    ਅਚਲ ggml_sycl_gg_tile * tiles_ptr = tiles_dev.get();
    ਅਚਲ ਅੱਖਰ *              src0_dd   = (ਅਚਲ ਅੱਖਰ *) src0_ਅਧਾਰ;
    ਵਾਪਸ fg_ਦੇਖੋ_type(src0_type, reordered, [&](ਸਵੈਚਾਲਿਤ ਟੈਗ) {
        ਵਰਤੋਂ T = decltype(ਟੈਗ);
        grouped_dequant_gemm_launch<S, ਕਿਸਮ_ਨਾਂ T::type, T::reordered>(src0_dd, expert_stride, tiles_ptr, packed, dst,
                                                                       (ਪੂਰਨ) M, Npad, (ਪੂਰਨ) K, n_tiles, groups_m,
                                                                       stream);
    });
}

ਤਾਰਕਿਕ ggml_sycl_grouped_dequant_gemm(ggml_type src0_type, ਤਾਰਕਿਕ reordered, ਅਚਲ ਖਾਲੀ * src0_ਅਧਾਰ, size_t expert_stride,
                                    ਅਚਲ ਦਸ਼ਮਲਵ * src1, int32_t src1_prec, ਦਸ਼ਮਲਵ * dst,
                                    ਅਚਲ int64_t * expert_ਪੰਕਤੀ_offsets, int64_t n_as, int64_t M, int64_t K,
                                    int64_t ਕੁੱਲ_ਪੰਕਤੀਆਂ, std::vector<ggml_sycl_gg_tile> & tiles,
                                    ggml_sycl_pool & pool, dpct::ਕਤਾਰ_ptr stream) {
    int64_t n_ਸਰਗਰਮ = 0;
    ਲਈ (int64_t e = 0; e < n_as; ++e) {
        n_ਸਰਗਰਮ += expert_ਪੰਕਤੀ_offsets[e + 1] > expert_ਪੰਕਤੀ_offsets[e];
    }
    ਜੇ (g_ggml_sycl_dynamic_precision == GGML_SYCL_DYNAMIC_PRECISION_F32 ||
        !ggml_sycl_xmx_gather_type_enabled(src0_type) || !fg_ਦੇਖੋ_type(src0_type, reordered, [](ਸਵੈਚਾਲਿਤ) {})) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (!ggml_sycl_grouped_dequant_gemm_shape_ok(src0_type, M, K, ਕੁੱਲ_ਪੰਕਤੀਆਂ, n_ਸਰਗਰਮ)) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਅਚਲ ਪੂਰਨ combo = fg_pick_combo(stream, GGML_TYPE_F32, src1_prec);
    ਜੇ (combo < 0) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਤਾਰਕਿਕ launched = ਝੂਠ;
    fg_ਦੇਖੋ_combo(combo, [&](ਸਵੈਚਾਲਿਤ s) {
        launched = fg_grouped_ਲੜੀਵਾਰ<decltype(s)>(src0_type, reordered, src0_ਅਧਾਰ, expert_stride, src1, dst,
                                               expert_ਪੰਕਤੀ_offsets, n_as, M, K, tiles, pool, stream);
    });
    ਵਾਪਸ launched;
}
