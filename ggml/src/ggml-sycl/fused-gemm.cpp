#ಅನ್ತರ್ಭಾವ "fused-gemm.hpp"

#ಅನ್ತರ್ಭಾವ <sycl/ext/oneapi/matrix/matrix.hpp>

#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <string>
#ಅನ್ತರ್ಭಾವ <tuple>
#ಅನ್ತರ್ಭಾವ <mutex>
#ಅನ್ತರ್ಭಾವ <set>
#ಅನ್ತರ್ಭಾವ <unordered_map>

ನಾಮಾಕಾಶ mx = sycl::ext::oneapi::experimental::ಮಾತೃಕೆ;

// FG_ / fg_ is short for fused GEMM: the weights are dequantized inside the GEMM, into the XMX tiles.

// A k step is one 32-value weight sub-block; iq3_s and the other superblock formats split their
// superblock into steps of this width. The sub-groups of a work-group each walk their own K range
// and are summed at the end.
ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ FG_BK     = QK4_NL;
ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ FG_KSPLIT = 4;

// Element traits of one joint_matrix operand type. The A stage and the B pack compute in f32 and
// convert once, in registers, when they write the element, so any type costs the same one pass.
//   store: storage in SLM (A) and in the packed B buffer
//   mtype: matrix_type in matrix_combinations
//   mode:  GGML_SYCL_DYNAMIC_PRECISION value that selects this type
//   src:   ggml type that needs no conversion into this type (GGML_TYPE_COUNT: none)
//   slow:  XMX throughput class, 0 is fastest. f16 and bf16 share the DPAS rate; tf32 does half the
//          K per instruction. B60, Qwen3-30B-A3B pp512: f16 1108, bf16 1000, tf32 751 t/s
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T> ರಚನೆ fg_elem;

ಮಾದರಿ <> ರಚನೆ fg_elem<sycl::half> {
    ಬಳಸಿ store = sycl::half;
    ಬಳಸಿ pair  = sycl::half2;
    ಸ್ಥಿರ constexpr mx::ಮಾತೃಕೆ_type mtype = mx::ಮಾತೃಕೆ_type::fp16;
    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ             mode  = GGML_SYCL_DYNAMIC_PRECISION_F16;
    ಸ್ಥಿರ constexpr ggml_type       src   = GGML_TYPE_F16;
    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ             mant  = 10;
    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ             slow  = 0;
    ಸ್ಥಿರ store cvt(ದಶಮಾಂಶ x) { ಹಿಂತಿರುಗಿ (store) x; }
    ಸ್ಥಿರ pair make(ದಶಮಾಂಶ x, ದಶಮಾಂಶ y) { ಹಿಂತಿರುಗಿ pair((store) x, (store) y); }
};

// tf32 rounds to nearest even with plain bit ops: round_to_tf32 needs a SPIR-V extension that the
// DG2 AOT target rejects
ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint32_t fg_ಸುತ್ತು_ಬಿಟ್ಟುಗಳು(ದಶಮಾಂಶ x, ಪೂರ್ಣಾಂಕ drop) {
    ಸ್ಥಿರಾಂಕ uint32_t u = sycl::ಬಿಟ್_cast<uint32_t>(x);
    ಆದರೆ ((u & 0x7f800000u) == 0x7f800000u) {
        ಹಿಂತಿರುಗಿ (u & 0x7fffffu) ? u | (1u << drop) : u; // nan stays nan
    }
    ಹಿಂತಿರುಗಿ u + ((1u << (drop - 1)) - 1) + ((u >> drop) & 1);
}

ರಚನೆ alignas(4) fg_bf16x2 {
    sycl::ext::oneapi::bfloat16 x, y;
};

ಮಾದರಿ <> ರಚನೆ fg_elem<sycl::ext::oneapi::bfloat16> {
    ಬಳಸಿ store = sycl::ext::oneapi::bfloat16;
    ಬಳಸಿ pair  = fg_bf16x2;
    ಸ್ಥಿರ constexpr mx::ಮಾತೃಕೆ_type mtype = mx::ಮಾತೃಕೆ_type::bf16;
    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ             mode  = GGML_SYCL_DYNAMIC_PRECISION_BF16;
    ಸ್ಥಿರ constexpr ggml_type       src   = GGML_TYPE_BF16;
    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ             mant  = 7;
    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ             slow  = 0;
    ಸ್ಥಿರ store cvt(ದಶಮಾಂಶ x) { ಹಿಂತಿರುಗಿ store(x); }
    ಸ್ಥಿರ pair make(ದಶಮಾಂಶ x, ದಶಮಾಂಶ y) { ಹಿಂತಿರುಗಿ { cvt(x), cvt(y) }; }
};

// tf32 keeps f32 range and f16 mantissa, in f32 storage
ಮಾದರಿ <> ರಚನೆ fg_elem<mx::precision::tf32> {
    ಬಳಸಿ store = ದಶಮಾಂಶ;
    ಬಳಸಿ pair  = sycl::float2;
    ಸ್ಥಿರ constexpr mx::ಮಾತೃಕೆ_type mtype = mx::ಮಾತೃಕೆ_type::tf32;
    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ             mode  = GGML_SYCL_DYNAMIC_PRECISION_TF32;
    ಸ್ಥಿರ constexpr ggml_type       src   = GGML_TYPE_ಎಣಿಕೆ;
    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ             mant  = 10;
    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ             slow  = 1;
    ಸ್ಥಿರ store cvt(ದಶಮಾಂಶ x) { ಹಿಂತಿರುಗಿ sycl::ಬಿಟ್_cast<ದಶಮಾಂಶ>(fg_ಸುತ್ತು_ಬಿಟ್ಟುಗಳು(x, 13) & ~0x1fffu); }
    ಸ್ಥಿರ pair make(ದಶಮಾಂಶ x, ದಶಮಾಂಶ y) { ಹಿಂತಿರುಗಿ pair(cvt(x), cvt(y)); }
};

// One joint_matrix combination (A type, B type, TM x TN x TK, sub-group size; C and D are f32) and
// the tiling built on it. A sub-group owns SG_ROWS rows of A (at least 16) and BN (at least 32)
// columns of B. A and B may differ: the device lists the pairs it supports.
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು TA, ಬಗೆ_ಹೆಸರು TB, ಪೂರ್ಣಾಂಕ TM_, ಪೂರ್ಣಾಂಕ TN_, ಪೂರ್ಣಾಂಕ TK_, ಪೂರ್ಣಾಂಕ SG_> ರಚನೆ fg_combo {
    ಬಳಸಿ ta  = TA;
    ಬಳಸಿ tb  = TB;
    ಬಳಸಿ EA  = fg_elem<TA>;
    ಬಳಸಿ EB  = fg_elem<TB>;
    ಬಳಸಿ tsa = ಬಗೆ_ಹೆಸರು EA::store;
    ಬಳಸಿ tsb = ಬಗೆ_ಹೆಸರು EB::store;
    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ TM = TM_;
    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ TN = TN_;
    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ TK = TK_;
    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ SG = SG_;
    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ VNNI    = 4 / ಗಾತ್ರ(tsb);  // K rows of B packed in one 32-bit word
    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ SG_ಅಡ್ಡಸಾಲುಗಳು = TM > 16 ? TM : 16;
    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ RPL     = SG_ಅಡ್ಡಸಾಲುಗಳು / SG;     // A rows one lane decodes per k step
    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ MT      = SG_ಅಡ್ಡಸಾಲುಗಳು / TM;
    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ BN      = TN > 32 ? TN : 32;
    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ NT      = BN / TN;
    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ WG_ಗಾತ್ರ = FG_KSPLIT * SG;
    ಸ್ಥಿರ constexpr mx::layout b_layout = VNNI == 1 ? mx::layout::ಅಡ್ಡಸಾಲು_major : mx::layout::ext_intel_packed;
    // a 64-wide N is mostly padding here and a 32x64 f32 accumulator needs 128 registers per lane,
    // so it spills: 13x slower on B60
    ಸ್ಥಿರ constexpr ತಾರ್ಕಿಕ efficient = TN <= 32;
    static_assert(SG_ಅಡ್ಡಸಾಲುಗಳು % SG == 0 && SG_ಅಡ್ಡಸಾಲುಗಳು % TM == 0 && BN % TN == 0 && FG_BK % TK == 0, "bad tile");
    static_assert(BN <= GGML_SYCL_FG_MAX_N, "header gate must cover the tile width");
};

ಬಳಸಿ fg_half = sycl::half;
ಬಳಸಿ fg_bf16 = sycl::ext::oneapi::bfloat16;
ಬಳಸಿ fg_tf32 = mx::precision::tf32;

// One bit of GGML_SYCL_XMX_GATHER_SHAPES per combination. Only combinations some device lists in
// matrix_combinations are built (appendix of sycl_ext_oneapi_matrix and the runtime's own list).
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು F> ಸ್ಥಿರ ಶೂನ್ಯ fg_ಭೇಟಿ_combo(ಪೂರ್ಣಾಂಕ idx, F && f);
ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ FG_N_COMBOS = 8;

// A spir64_gen AOT build (GGML_SYCL_XMX_AOT_SG) drops the combinations of the other sub-group size
// entirely: ocloc rejects even an empty kernel that asks for a sub-group size it lacks.
ಮಾದರಿ <ಪೂರ್ಣಾಂಕ SG> ಸ್ಥಿರ constexpr ತಾರ್ಕಿಕ fg_listed() {
#ಆದರೆ defined(GGML_SYCL_XMX_AOT_SG)
    ಹಿಂತಿರುಗಿ SG == GGML_SYCL_XMX_AOT_SG;
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ ನಿಜ;
#ಆದರೆ_ಅಂತ್ಯ
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು S, ಬಗೆ_ಹೆಸರು F> ಸ್ಥಿರ ಶೂನ್ಯ fg_call_combo(F && f) {
    ಆದರೆ constexpr (fg_listed<S::SG>()) {
        f(S{});
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು F> ಸ್ಥಿರ ಶೂನ್ಯ fg_ಭೇಟಿ_combo(ಪೂರ್ಣಾಂಕ idx, F && f) {
    ಆಯ್ಕೆ (idx) {
        ಸ್ಥಿತಿ 0: fg_call_combo<fg_combo<fg_half, fg_half, 8, 16, 16, 16>>(f);  ನಿಲ್ಲಿಸು; // Xe2, Xe3, Xe-HPC
        ಸ್ಥಿತಿ 1: fg_call_combo<fg_combo<fg_half, fg_half, 16, 16, 16, 16>>(f); ನಿಲ್ಲಿಸು; // Xe2, Xe3, Xe-HPC
        ಸ್ಥಿತಿ 2: fg_call_combo<fg_combo<fg_half, fg_half, 32, 64, 16, 16>>(f); ನಿಲ್ಲಿಸು; // Xe2, Xe3, Xe-HPC
        ಸ್ಥಿತಿ 3: fg_call_combo<fg_combo<fg_half, fg_half, 32, 64, 32, 16>>(f); ನಿಲ್ಲಿಸು; // Xe2, Xe3, Xe-HPC
        ಸ್ಥಿತಿ 4: fg_call_combo<fg_combo<fg_half, fg_half, 8, 8, 16, 8>>(f);    ನಿಲ್ಲಿಸು; // Xe-HPG (Arc A), ARL-H
        ಸ್ಥಿತಿ 5: fg_call_combo<fg_combo<fg_tf32, fg_tf32, 8, 16, 8, 16>>(f);   ನಿಲ್ಲಿಸು; // Xe2, Xe3, Xe-HPC
        ಸ್ಥಿತಿ 6: fg_call_combo<fg_combo<fg_bf16, fg_bf16, 8, 16, 16, 16>>(f);  ನಿಲ್ಲಿಸು; // Xe2, Xe3, Xe-HPC
        ಸ್ಥಿತಿ 7: fg_call_combo<fg_combo<fg_bf16, fg_bf16, 8, 8, 16, 8>>(f);    ನಿಲ್ಲಿಸು; // Xe-HPG (Arc A), ARL-H
        ಪೂರ್ವನಿಯೋಜಿತ: GGML_ABORT("bad XMX combination %d", idx);
    }
}

// AOT with -fsycl-targets=intel_gpu_*: compile each tile body only for targets with its sub-group
// size, since IGC fails on the other ones. A JIT build keeps them all, but each combination lands in
// its own device image (joint_matrix is an optional kernel feature) and only a combination the
// device reports is launched, so the runtime never asks IGC for the others.
#ಆದರೆ defined(__SYCL_DEVICE_ONLY__)
#    ಆದರೆ __SYCL_TARGET_INTEL_GPU_ACM_G10__ || __SYCL_TARGET_INTEL_GPU_ACM_G11__ || __SYCL_TARGET_INTEL_GPU_ACM_G12__ || \
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
#ಆದರೆ_ಅಂತ್ಯ

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ SG> ಸ್ಥಿರ constexpr ತಾರ್ಕಿಕ fg_built() {
#ಆದರೆ defined(FG_AOT_SG)
    ಹಿಂತಿರುಗಿ SG == FG_AOT_SG;
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ ನಿಜ;
#ಆದರೆ_ಅಂತ್ಯ
}

// Upper bound on the tile count when total_rows rows are routed to n_as experts: the worst case gives
// each expert one row and fills whole tiles with the rest. The bound depends only on the shape, not
// on the routing, so the pool reuses one buffer every ubatch instead of keeping one per size seen.
ಸ್ಥಿರ constexpr int64_t grouped_gemm_max_tiles(int64_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು, int64_t n_as, int64_t BN) {
    ಹಿಂತಿರುಗಿ ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು <= n_as ? ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು : n_as + (ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು - n_as) / BN;
}
// Tiles do not cross experts, so the bound is not ceil(total_rows / BN): 34 rows over 2 experts with
// BN = 16 split 17 + 17 need 2 + 2 tiles, where the ceil gives 3.
static_assert(grouped_gemm_max_tiles(34, 2, 16) == 4);

// the device lists S with an f32 accumulator and output
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು S> ಸ್ಥಿರ ತಾರ್ಕಿಕ fg_device_has_combo(ಸ್ಥಿರಾಂಕ std::vector<mx::combination> & combinations) {
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & c : combinations) {
        ಆದರೆ (c.atype == S::EA::mtype && c.btype == S::EB::mtype && c.ctype == mx::ಮಾತೃಕೆ_type::fp32 &&
            c.dtype == mx::ಮಾತೃಕೆ_type::fp32 &&
            (c.max_msize >= (size_t) S::TM || c.msize == (size_t) S::TM) &&
            (c.max_nsize >= (size_t) S::TN || c.nsize == (size_t) S::TN) &&
            (c.max_ksize >= (size_t) S::TK || c.ksize == (size_t) S::TK)) {
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
    }
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T> ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * fg_type_name() {
    ಹಿಂತಿರುಗಿ std::is_same_v<T, fg_half> ? "f16" : std::is_same_v<T, fg_bf16> ? "bf16" : "tf32";
}

ಸ್ಥಿರ std::string fg_combo_name(ಪೂರ್ಣಾಂಕ idx) {
    std::string name;
    fg_ಭೇಟಿ_combo(idx, [&](ಸ್ವಯಂಚಾಲಿತ s) {
        ಬಳಸಿ S = decltype(s);
        name = std::string(fg_type_name<ಬಗೆ_ಹೆಸರು S::ta>()) + "x" + fg_type_name<ಬಗೆ_ಹೆಸರು S::tb>() + " " +
               std::to_string(S::TM) + "x" + std::to_string(S::TN) + "x" + std::to_string(S::TK) + " sg" +
               std::to_string(S::SG);
    });
    ಹಿಂತಿರುಗಿ name;
}

// Combinations this build has kernels for and the device lists, one bit each. Cached per device:
// on a mixed box the first caller's verdict is not the others'.
ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ fg_device_combos(ಸ್ಥಿರಾಂಕ sycl::device & dev) {
    ಸ್ಥಿರ std::mutex                            mtx;
    ಸ್ಥಿರ std::unordered_map<sycl::device, ಪೂರ್ಣಾಂಕ> ತಿಳಿದ;
    std::lock_guard<std::mutex>                  ಬೀಗ(mtx);
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ                                   it = ತಿಳಿದ.find(dev);
    ಆದರೆ (it != ತಿಳಿದ.end()) {
        ಹಿಂತಿರುಗಿ it->second;
    }
    ಪೂರ್ಣಾಂಕ ಲಭ್ಯ = 0;
    ಪ್ರಯತ್ನಿಸು {
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ combinations = dev.get_info<sycl::ext::oneapi::experimental::info::device::ಮಾತೃಕೆ_combinations>();
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ sg_sizes     = dev.get_info<sycl::info::device::sub_ಗುಂಪು_sizes>();
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ idx = 0; idx < FG_N_COMBOS; ++idx) {
            fg_ಭೇಟಿ_combo(idx, [&](ಸ್ವಯಂಚಾಲಿತ s) {
                ಬಳಸಿ S = decltype(s);
                ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ sg_ok = std::find(sg_sizes.begin(), sg_sizes.end(), (size_t) S::SG) != sg_sizes.end();
                ಆದರೆ (sg_ok && fg_device_has_combo<S>(combinations)) {
                    ಲಭ್ಯ |= 1 << idx;
                }
            });
        }
    } ಹಿಡಿ (ಸ್ಥಿರಾಂಕ sycl::exception &) {
        ಲಭ್ಯ = 0;
    }
    GGML_LOG_INFO("%s: %s: XMX dequant-GEMM combinations available 0x%x, allowed 0x%x\n", __func__,
                  dev.get_info<sycl::info::device::name>().c_str(), ಲಭ್ಯ, g_ggml_sycl_xmx_gather_shapes);
    ತಿಳಿದ.emplace(dev, ಲಭ್ಯ);
    ಹಿಂತಿರುಗಿ ಲಭ್ಯ;
}

// Rank of combination S for a src1 of type src1_type, lower is better. Order:
//  1. throughput: a tile that does not spill, then the fastest type class of A and B
//  2. B type equal to the src1 type, so the pack is a plain copy
//  3. B at least as precise as f16
//  4. the device's native DPAS tile (8 x SG x 32 bytes of K), then the largest M x K
// A costs nothing to convert: the A stage emits any type at the same cost.
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು S> ಸ್ಥಿರ int64_t fg_rank(ggml_type src1_type) {
    ಸ್ಥಿರಾಂಕ int64_t spills  = !S::efficient;
    ಸ್ಥಿರಾಂಕ int64_t slow    = std::max(S::EA::slow, S::EB::slow);
    ಸ್ಥಿರಾಂಕ int64_t convert = S::EB::src != src1_type;
    ಸ್ಥಿರಾಂಕ int64_t lossy   = S::EB::mant < 10;
    ಸ್ಥಿರಾಂಕ int64_t foreign = !(S::TM == 8 && S::TN == S::SG);
    ಸ್ಥಿರಾಂಕ int64_t mk      = 1024 - S::TM * S::TK;
    ಹಿಂತಿರುಗಿ ((((spills * 2 + slow) * 2 + convert) * 2 + lossy) * 2 + foreign) * 2048 + mk;
}

// whether XMX operands of type mode (a GGML_SYCL_DYNAMIC_PRECISION value) meet the src1 request
// [TAG_GGML_PREC]. f16 lacks the f32 range that BF16 and F32 ask for; an F32 request goes only as far
// down as GGML_SYCL_DYNAMIC_REQUIRED_PRECISION allows.
ಸ್ಥಿರ ತಾರ್ಕಿಕ fg_ಬಹುಲಕ_meets(ಪೂರ್ಣಾಂಕ mode, int32_t src1_prec) {
    ಆದರೆ (src1_prec == GGML_PREC_UNDEFINED || src1_prec >= GGML_PREC_F16) {
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }
    ಆದರೆ (src1_prec >= GGML_PREC_BF16) {
        ಹಿಂತಿರುಗಿ mode != GGML_SYCL_DYNAMIC_PRECISION_F16;
    }
    ಆಯ್ಕೆ (g_ggml_sycl_dynamic_required_precision) {
        ಸ್ಥಿತಿ GGML_SYCL_DYNAMIC_PRECISION_TF32: ಹಿಂತಿರುಗಿ mode == GGML_SYCL_DYNAMIC_PRECISION_TF32;
        ಸ್ಥಿತಿ GGML_SYCL_DYNAMIC_PRECISION_BF16: ಹಿಂತಿರುಗಿ mode != GGML_SYCL_DYNAMIC_PRECISION_F16;
        ಪೂರ್ವನಿಯೋಜಿತ:                               ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
}

// Best allowed combination for this call, or -1 if none. The type is GGML_SYCL_DYNAMIC_PRECISION if it
// meets the src1 request; if not, bf16 then tf32 for a BF16 request (fastest first) and tf32 then bf16
// for an F32 request (most mantissa first).
ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ fg_pick_combo(dpct::ಸರತಿ_ptr stream, ggml_type src1_type, int32_t src1_prec) {
    ಸ್ಥಿರಾಂಕ sycl::device dev     = stream->get_device();
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ          allowed = fg_device_combos(dev) & g_ggml_sycl_xmx_gather_shapes;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ         f32_req = src1_prec != GGML_PREC_UNDEFINED && src1_prec < GGML_PREC_BF16;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ          modes[] = {
        g_ggml_sycl_dynamic_precision,
        f32_req ? GGML_SYCL_DYNAMIC_PRECISION_TF32 : GGML_SYCL_DYNAMIC_PRECISION_BF16,
        f32_req ? GGML_SYCL_DYNAMIC_PRECISION_BF16 : GGML_SYCL_DYNAMIC_PRECISION_TF32,
    };
    ಪೂರ್ಣಾಂಕ     ಉತ್ತಮ      = -1;
    int64_t ಉತ್ತಮ_rank = 0;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 3 && ಉತ್ತಮ < 0; ++i) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mode = modes[i];
        ಆದರೆ (!fg_ಬಹುಲಕ_meets(mode, src1_prec)) {
            ಮುಂದುವರಿ;
        }
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ idx = 0; idx < FG_N_COMBOS; ++idx) {
            ಆದರೆ (!(allowed & (1 << idx))) {
                ಮುಂದುವರಿ;
            }
            fg_ಭೇಟಿ_combo(idx, [&](ಸ್ವಯಂಚಾಲಿತ s) {
                ಬಳಸಿ S = decltype(s);
                ಆದರೆ (S::EA::mode != mode || S::EB::mode != mode) {
                    ಹಿಂತಿರುಗಿ;
                }
                ಸ್ಥಿರಾಂಕ int64_t rank = fg_rank<S>(src1_type);
                ಆದರೆ (ಉತ್ತಮ < 0 || rank < ಉತ್ತಮ_rank) {
                    ಉತ್ತಮ      = idx;
                    ಉತ್ತಮ_rank = rank;
                }
            });
        }
    }
    // log each distinct decision once
    ಸ್ಥಿರ std::mutex                                      mtx;
    ಸ್ಥಿರ std::set<std::tuple<size_t, ಪೂರ್ಣಾಂಕ, int32_t, ಪೂರ್ಣಾಂಕ>> ನೋಡಿದ;
    std::lock_guard<std::mutex>                            ಬೀಗ(mtx);
    ಆದರೆ (ನೋಡಿದ.emplace(std::hash<sycl::device>{}(dev), (ಪೂರ್ಣಾಂಕ) src1_type, src1_prec, ಉತ್ತಮ).second) {
        GGML_LOG_INFO("%s: src1 %s, src1 prec %d -> %s\n", __func__, ggml_type_name(src1_type), src1_prec,
                      ಉತ್ತಮ >= 0 ? fg_combo_name(ಉತ್ತಮ).c_str() : "none (library GEMM)");
    }
    ಹಿಂತಿರುಗಿ ಉತ್ತಮ;
}

// src1 [N][K] -> packed [K/V][Npad][V] so B tiles load straight from global memory
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು E, ಬಗೆ_ಹೆಸರು T_src>
ಸ್ಥಿರ ಶೂನ್ಯ fused_gemm_pack_b(ಸ್ಥಿರಾಂಕ T_src * y, ಬಗೆ_ಹೆಸರು E::store * packed, ಪೂರ್ಣಾಂಕ N, ಪೂರ್ಣಾಂಕ Npad, ಪೂರ್ಣಾಂಕ K,
                              dpct::ಸರತಿ_ptr stream) {
    constexpr ಪೂರ್ಣಾಂಕ V   = 4 / ಗಾತ್ರ(ಬಗೆ_ಹೆಸರು E::store);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     kqs = K / V;
    stream->parallel_for(sycl::range<1>((size_t) Npad * kqs), [=](sycl::id<1> id) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ idx = id[0];
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n   = idx / kqs;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kq  = idx - n * kqs;
        ಬಗೆ_ಹೆಸರು E::store vals[V] = {};
        ಆದರೆ (n < N) {
            ಸ್ಥಿರಾಂಕ T_src * src = y + (size_t) n * K + V * kq;
#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ v = 0; v < V; ++v) {
                vals[v] = E::cvt((ದಶಮಾಂಶ) src[v]);
            }
        }
        ಬಗೆ_ಹೆಸರು E::store * ಹೊರಹೋಗು = packed + ((size_t) kq * Npad + n) * V;
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ v = 0; v < V; ++v) {
            ಹೊರಹೋಗು[v] = vals[v];
        }
    });
}

// A stage: one lane owns one row and decodes FG_BK values of it per k step, with every scale
// folded into the value so the mad below sees plain A elements. One overload per weight format.
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು E>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ fg_stage_a(ಸ್ಥಿರಾಂಕ ಖಂಡ_iq4_nl * __restrict__ xrow, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kb, ಬಗೆ_ಹೆಸರು E::pair * a) {
    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq4_nl blk = xrow[kb];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ        d   = (ದಶಮಾಂಶ) blk.d;
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK4_NL / 2; j += 2) {
        ಸ್ಥಿರಾಂಕ uint8_t q0 = blk.qs[j];
        ಸ್ಥಿರಾಂಕ uint8_t q1 = blk.qs[j + 1];
        a[j / 2]     = E::make(d * kvalues_iq4nl[q0 & 0xf], d * kvalues_iq4nl[q1 & 0xf]);
        a[j / 2 + 8] = E::make(d * kvalues_iq4nl[q0 >> 4], d * kvalues_iq4nl[q1 >> 4]);
    }
}

// iq3_s: k step kb is sub-block kb % 8 of superblock kb / 8. The superblock is 110 bytes, so read
// only the fields of that sub-block instead of copying the block. Same decode as
// dequantize_block_iq3_s: grid entries are taken as dwords and the sign bit is a plain shift.
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು E>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ fg_stage_a(ಸ್ಥಿರಾಂಕ ಖಂಡ_iq3_s * __restrict__ xrow, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kb, ಬಗೆ_ಹೆಸರು E::pair * a) {
    static_assert(QK_K == 256, "the iq3_s A stage assumes 8 sub-blocks per superblock");
    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq3_s * blk = xrow + kb / (QK_K / 32);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ           ib8 = kb % (QK_K / 32);
    ಸ್ಥಿರಾಂಕ uint8_t *     qs  = blk->qs + 8 * ib8;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ           qh  = blk->qh[ib8];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ         d   = (ದಶಮಾಂಶ) blk->d * (1 + 2 * ((blk->scales[ib8 / 2] >> (4 * (ib8 % 2))) & 0xf));
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ il = 0; il < 4; ++il) {
        ಸ್ಥಿರಾಂಕ uint32_t grid1 = iq3s_grid[qs[2 * il + 0] | ((qh << (8 - 2 * il)) & 256)];
        ಸ್ಥಿರಾಂಕ uint32_t grid2 = iq3s_grid[qs[2 * il + 1] | ((qh << (7 - 2 * il)) & 256)];
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ      signs = blk->signs[4 * ib8 + il];
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 2; ++j) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ g1a = (ದಶಮಾಂಶ) ((grid1 >> (16 * j + 0)) & 0xff);
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ g1b = (ದಶಮಾಂಶ) ((grid1 >> (16 * j + 8)) & 0xff);
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ g2a = (ದಶಮಾಂಶ) ((grid2 >> (16 * j + 0)) & 0xff);
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ g2b = (ದಶಮಾಂಶ) ((grid2 >> (16 * j + 8)) & 0xff);
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ   s   = 2 * j;
            a[4 * il + j]     = E::make(d * ((signs & (1 << (s + 0))) ? -g1a : g1a),
                                        d * ((signs & (1 << (s + 1))) ? -g1b : g1b));
            a[4 * il + j + 2] = E::make(d * ((signs & (1 << (s + 4))) ? -g2a : g2a),
                                        d * ((signs & (1 << (s + 5))) ? -g2b : g2b));
        }
    }
}

// values per stored block, so a row of K values is K/qk blocks
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು ಖಂಡ_q_t> ರಚನೆ fg_ಖಂಡ_traits;
ಮಾದರಿ <> ರಚನೆ fg_ಖಂಡ_traits<ಖಂಡ_iq4_nl> { ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ qk = QK4_NL; };
ಮಾದರಿ <> ರಚನೆ fg_ಖಂಡ_traits<ಖಂಡ_iq3_s>  { ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ qk = QK_K; };
ಮಾದರಿ <> ರಚನೆ fg_ಖಂಡ_traits<ಖಂಡ_iq3_xxs> { ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ qk = QK_K; };
ಮಾದರಿ <> ರಚನೆ fg_ಖಂಡ_traits<ಖಂಡ_iq4_xs>  { ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ qk = QK_K; };
ಮಾದರಿ <> ರಚನೆ fg_ಖಂಡ_traits<ಖಂಡ_iq2_xxs> { ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ qk = QK_K; };
ಮಾದರಿ <> ರಚನೆ fg_ಖಂಡ_traits<ಖಂಡ_iq2_xs>  { ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ qk = QK_K; };
ಮಾದರಿ <> ರಚನೆ fg_ಖಂಡ_traits<ಖಂಡ_iq2_s>   { ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ qk = QK_K; };
ಮಾದರಿ <> ರಚನೆ fg_ಖಂಡ_traits<ಖಂಡ_iq1_s>   { ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ qk = QK_K; };
ಮಾದರಿ <> ರಚನೆ fg_ಖಂಡ_traits<ಖಂಡ_iq1_m>   { ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ qk = QK_K; };
ಮಾದರಿ <> ರಚನೆ fg_ಖಂಡ_traits<ಖಂಡ_q8_0>    { ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ qk = QK8_0; };
ಮಾದರಿ <> ರಚನೆ fg_ಖಂಡ_traits<ಖಂಡ_q4_K>    { ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ qk = QK_K; };
ಮಾದರಿ <> ರಚನೆ fg_ಖಂಡ_traits<ಖಂಡ_q5_K>    { ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ qk = QK_K; };
ಮಾದರಿ <> ರಚನೆ fg_ಖಂಡ_traits<ಖಂಡ_q6_K>    { ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ qk = QK_K; };

// The A stages below are the dequantize_block_iq* kernels rewritten for one k step. There a
// work-item handled one quarter (il) of one 32-wide sub-block (ib); here one lane produces the
// whole step, so il becomes a loop and ib is kb inside the superblock. Each quarter yields 8
// consecutive values, i.e. 4 pairs at a[4*il], so nothing larger than 8 floats is ever live.
#ವ್ಯಾಖ್ಯಾನಿಸು FG_SUPERBLOCK(T)                                                         \
    static_assert(QK_K == 256, "the " #T " A stage assumes 8 sub-blocks per superblock"); \
    ಸ್ಥಿರಾಂಕ T * blk = xrow + kb / (QK_K / 32);                                     \
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib  = kb % (QK_K / 32)

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು E>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ fg_pack_quarter(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * __restrict__ t, ಬಗೆ_ಹೆಸರು E::pair * a, ಪೂರ್ಣಾಂಕ il) {
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 4; ++j) {
        a[4 * il + j] = E::make(t[2 * j], t[2 * j + 1]);
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು E>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ fg_stage_a(ಸ್ಥಿರಾಂಕ ಖಂಡ_iq4_xs * __restrict__ xrow, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kb, ಬಗೆ_ಹೆಸರು E::pair * a) {
    FG_SUPERBLOCK(ಖಂಡ_iq4_xs);
    // low nibbles fill the first half of the step, high nibbles the second, so the two halves
    // land at a[0..7] and a[8..15] and no quarter loop is needed
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = (ದಶಮಾಂಶ) blk->d *
        ((((blk->scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) | (((blk->scales_h >> (2 * ib)) & 3) << 4)) - 32);
    ಸ್ಥಿರಾಂಕ uint8_t * q4 = blk->qs + 16 * ib;
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 8; ++j) {
        a[j]     = E::make(d * kvalues_iq4nl[q4[2 * j] & 0xf], d * kvalues_iq4nl[q4[2 * j + 1] & 0xf]);
        a[8 + j] = E::make(d * kvalues_iq4nl[q4[2 * j] >> 4],  d * kvalues_iq4nl[q4[2 * j + 1] >> 4]);
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು E>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ fg_stage_a(ಸ್ಥಿರಾಂಕ ಖಂಡ_iq3_xxs * __restrict__ xrow, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kb, ಬಗೆ_ಹೆಸರು E::pair * a) {
    FG_SUPERBLOCK(ಖಂಡ_iq3_xxs);
    ಸ್ಥಿರಾಂಕ uint8_t *  q3    = blk->qs + 8 * ib;
    ಸ್ಥಿರಾಂಕ uint16_t * gas   = (ಸ್ಥಿರಾಂಕ uint16_t *) (blk->qs + QK_K / 4) + 2 * ib;
    ಸ್ಥಿರಾಂಕ uint32_t   aux32 = gas[0] | (gas[1] << 16);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ      d     = (ದಶಮಾಂಶ) blk->d * (0.5f + (aux32 >> 28)) * 0.5f;
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ il = 0; il < 4; ++il) {
        ಸ್ಥಿರಾಂಕ uint8_t * grid1 = (ಸ್ಥಿರಾಂಕ uint8_t *) (iq3xxs_grid + q3[2 * il + 0]);
        ಸ್ಥಿರಾಂಕ uint8_t * grid2 = (ಸ್ಥಿರಾಂಕ uint8_t *) (iq3xxs_grid + q3[2 * il + 1]);
        ಸ್ಥಿರಾಂಕ uint8_t   signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];
        ದಶಮಾಂಶ t[8];
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 4; ++j) {
            t[j + 0] = d * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f);
            t[j + 4] = d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು E>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ fg_stage_a(ಸ್ಥಿರಾಂಕ ಖಂಡ_iq2_xxs * __restrict__ xrow, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kb, ಬಗೆ_ಹೆಸರು E::pair * a) {
    FG_SUPERBLOCK(ಖಂಡ_iq2_xxs);
    ಸ್ಥಿರಾಂಕ uint16_t * q2    = blk->qs + 4 * ib;
    ಸ್ಥಿರಾಂಕ uint8_t *  aux8  = (ಸ್ಥಿರಾಂಕ uint8_t *) q2;
    ಸ್ಥಿರಾಂಕ uint32_t   aux32 = q2[2] | (q2[3] << 16);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ      d     = (ದಶಮಾಂಶ) blk->d * (0.5f + (aux32 >> 28)) * 0.25f;
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ il = 0; il < 4; ++il) {
        ಸ್ಥಿರಾಂಕ uint8_t * grid  = (ಸ್ಥಿರಾಂಕ uint8_t *) (iq2xxs_grid + aux8[il]);
        ಸ್ಥಿರಾಂಕ uint8_t   signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];
        ದಶಮಾಂಶ t[8];
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 8; ++j) {
            t[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು E>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ fg_stage_a(ಸ್ಥಿರಾಂಕ ಖಂಡ_iq2_xs * __restrict__ xrow, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kb, ಬಗೆ_ಹೆಸರು E::pair * a) {
    FG_SUPERBLOCK(ಖಂಡ_iq2_xs);
    ಸ್ಥಿರಾಂಕ uint16_t * q2 = blk->qs + 4 * ib;
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ il = 0; il < 4; ++il) {
        ಸ್ಥಿರಾಂಕ uint8_t * grid  = (ಸ್ಥಿರಾಂಕ uint8_t *) (iq2xs_grid + (q2[il] & 511));
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ     d     = (ದಶಮಾಂಶ) blk->d * (0.5f + ((blk->scales[ib] >> (4 * (il / 2))) & 0xf)) * 0.25f;
        ಸ್ಥಿರಾಂಕ uint8_t   signs = ksigns_iq2xs[q2[il] >> 9];
        ದಶಮಾಂಶ t[8];
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 8; ++j) {
            t[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು E>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ fg_stage_a(ಸ್ಥಿರಾಂಕ ಖಂಡ_iq2_s * __restrict__ xrow, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kb, ಬಗೆ_ಹೆಸರು E::pair * a) {
    FG_SUPERBLOCK(ಖಂಡ_iq2_s);
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ il = 0; il < 4; ++il) {
        ಸ್ಥಿರಾಂಕ uint8_t * grid =
            (ಸ್ಥಿರಾಂಕ uint8_t *) (iq2s_grid + (blk->qs[4 * ib + il] | ((blk->qh[ib] << (8 - 2 * il)) & 0x300)));
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   d     = (ದಶಮಾಂಶ) blk->d * (0.5f + ((blk->scales[ib] >> (4 * (il / 2))) & 0xf)) * 0.25f;
        ಸ್ಥಿರಾಂಕ uint8_t signs = blk->qs[QK_K / 8 + 4 * ib + il];
        ದಶಮಾಂಶ t[8];
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 8; ++j) {
            t[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು E>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ fg_stage_a(ಸ್ಥಿರಾಂಕ ಖಂಡ_iq1_s * __restrict__ xrow, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kb, ಬಗೆ_ಹೆಸರು E::pair * a) {
    FG_SUPERBLOCK(ಖಂಡ_iq1_s);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ delta = blk->qh[ib] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d     = (ದಶಮಾಂಶ) blk->d * (2 * ((blk->qh[ib] >> 12) & 7) + 1);
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ il = 0; il < 4; ++il) {
        uint32_t       grid32[2];
        ಸ್ಥಿರಾಂಕ int8_t * q = (ಸ್ಥಿರಾಂಕ int8_t *) grid32;
        grid32[0] = iq1s_grid_gpu[blk->qs[4 * ib + il] | (((blk->qh[ib] >> (3 * il)) & 7) << 8)];
        grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
        grid32[0] &= 0x0f0f0f0f;
        ದಶಮಾಂಶ t[8];
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 8; ++j) {
            t[j] = d * (q[j] + delta);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು E>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ fg_stage_a(ಸ್ಥಿರಾಂಕ ಖಂಡ_iq1_m * __restrict__ xrow, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kb, ಬಗೆ_ಹೆಸರು E::pair * a) {
    FG_SUPERBLOCK(ಖಂಡ_iq1_m);
    ಸ್ಥಿರಾಂಕ uint16_t * sc = (ಸ್ಥಿರಾಂಕ uint16_t *) blk->scales;
    iq1m_scale_t     scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ il = 0; il < 4; ++il) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ   ib16  = 2 * ib + il / 2;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d     = (ದಶಮಾಂಶ) scale.f16 * (2 * ((sc[ib16 / 4] >> (3 * (ib16 % 4))) & 0x7) + 1);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ delta = blk->qh[2 * ib + il / 2] & (0x08 << (4 * (il % 2))) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
        uint32_t       grid32[2];
        ಸ್ಥಿರಾಂಕ int8_t * q = (ಸ್ಥಿರಾಂಕ int8_t *) grid32;
        grid32[0] = iq1s_grid_gpu[blk->qs[4 * ib + il] |
                                  (((blk->qh[2 * ib + il / 2] >> (4 * (il % 2))) & 7) << 8)];
        grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
        grid32[0] &= 0x0f0f0f0f;
        ದಶಮಾಂಶ t[8];
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 8; ++j) {
            t[j] = d * (q[j] + delta);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}


// q8_0 and the k-quants. Each decode takes the fields of one 32-value sub-block, so the canonical
// layout and the reorder (SoA) layout of reorder_qw() share it and differ only in where the fields
// live, as in dequantize.hpp.
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು E>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ fg_decode_q8_0(ಸ್ಥಿರಾಂಕ int8_t * __restrict__ qs, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d, ಬಗೆ_ಹೆಸರು E::pair * a) {
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK8_0 / 2; ++j) {
        a[j] = E::make(d * qs[2 * j], d * qs[2 * j + 1]);
    }
}

// same unpack as get_scale_min_k4() in dequantize.hpp
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ fg_scale_min_k4(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ j, ಸ್ಥಿರಾಂಕ uint8_t * __restrict__ q, uint8_t & d, uint8_t & m) {
    ಆದರೆ (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } ಇಲ್ಲದಿದ್ದರೆ {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

// q4_K sub-block ib (0..7) uses scale/min pair ib and the low (even ib) or high (odd ib) nibbles of
// qs[32 * (ib / 2) ...], as in dequantize_row_q4_K
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು E>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ fg_decode_q4_K(ಸ್ಥಿರಾಂಕ uint8_t * __restrict__ qs, ಸ್ಥಿರಾಂಕ uint8_t * __restrict__ scales,
                                           ಸ್ಥಿರಾಂಕ sycl::half2 dm, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib, ಬಗೆ_ಹೆಸರು E::pair * a) {
    uint8_t sc, mb;
    fg_scale_min_k4(ib, scales, sc, mb);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ     d     = (ದಶಮಾಂಶ) dm[0] * sc;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ     m     = (ದಶಮಾಂಶ) dm[1] * mb;
    ಸ್ಥಿರಾಂಕ uint8_t * q     = qs + 32 * (ib / 2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       ಸರಿಸು = 4 * (ib % 2);
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 16; ++j) {
        a[j] = E::make(d * ((q[2 * j] >> ಸರಿಸು) & 0xF) - m, d * ((q[2 * j + 1] >> ಸರಿಸು) & 0xF) - m);
    }
}

// q5_K: q4_K plus one high bit per value, bit ib of qh[l]
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು E>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ fg_decode_q5_K(ಸ್ಥಿರಾಂಕ uint8_t * __restrict__ qs, ಸ್ಥಿರಾಂಕ uint8_t * __restrict__ qh,
                                           ಸ್ಥಿರಾಂಕ uint8_t * __restrict__ scales, ಸ್ಥಿರಾಂಕ sycl::half2 dm, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib,
                                           ಬಗೆ_ಹೆಸರು E::pair * a) {
    uint8_t sc, mb;
    fg_scale_min_k4(ib, scales, sc, mb);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ     d     = (ದಶಮಾಂಶ) dm[0] * sc;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ     m     = (ದಶಮಾಂಶ) dm[1] * mb;
    ಸ್ಥಿರಾಂಕ uint8_t * q     = qs + 32 * (ib / 2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       ಸರಿಸು = 4 * (ib % 2);
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 16; ++j) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ l = 2 * j;
        a[j] = E::make(d * (((q[l] >> ಸರಿಸು) & 0xF) | (((qh[l] >> ib) & 1) << 4)) - m,
                       d * (((q[l + 1] >> ಸರಿಸು) & 0xF) | (((qh[l + 1] >> ib) & 1) << 4)) - m);
    }
}

// q6_K: sub-block ib is quarter r = ib % 4 of half h = ib / 4. The half selects ql + 64h, qh + 32h
// and scales + 8h; the quarter selects ql + 32(r & 1), the ql nibble r / 2, the qh bit pair r and
// scales + 2r, as in dequantize_row_q6_K. The scale changes at value 16 of the sub-block.
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು E>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ fg_decode_q6_K(ಸ್ಥಿರಾಂಕ uint8_t * __restrict__ ql, ಸ್ಥಿರಾಂಕ uint8_t * __restrict__ qh,
                                           ಸ್ಥಿರಾಂಕ int8_t * __restrict__ scales, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib,
                                           ಬಗೆ_ಹೆಸರು E::pair * a) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       h  = ib / 4;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       r  = ib % 4;
    ಸ್ಥಿರಾಂಕ uint8_t * q  = ql + 64 * h + 32 * (r & 1);
    ಸ್ಥಿರಾಂಕ uint8_t * hb = qh + 32 * h;
    ಸ್ಥಿರಾಂಕ int8_t *  sc = scales + 8 * h + 2 * r;
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 16; ++j) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ   l  = 2 * j;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dl = d * sc[j / 8];
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ   q0 = (((q[l] >> (4 * (r / 2))) & 0xF) | (((hb[l] >> (2 * r)) & 3) << 4)) - 32;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ   q1 = (((q[l + 1] >> (4 * (r / 2))) & 0xF) | (((hb[l + 1] >> (2 * r)) & 3) << 4)) - 32;
        a[j] = E::make(dl * q0, dl * q1);
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು E>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ fg_stage_a(ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * __restrict__ xrow, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kb, ಬಗೆ_ಹೆಸರು E::pair * a) {
    fg_decode_q8_0<E>(xrow[kb].qs, (ದಶಮಾಂಶ) xrow[kb].d, a);
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು E>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ fg_stage_a(ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_K * __restrict__ xrow, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kb, ಬಗೆ_ಹೆಸರು E::pair * a) {
    FG_SUPERBLOCK(ಖಂಡ_q4_K);
    fg_decode_q4_K<E>(blk->qs, blk->scales, blk->dm, ib, a);
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು E>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ fg_stage_a(ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_K * __restrict__ xrow, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kb, ಬಗೆ_ಹೆಸರು E::pair * a) {
    FG_SUPERBLOCK(ಖಂಡ_q5_K);
    fg_decode_q5_K<E>(blk->qs, blk->qh, blk->scales, blk->dm, ib, a);
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು E>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ fg_stage_a(ಸ್ಥಿರಾಂಕ ಖಂಡ_q6_K * __restrict__ xrow, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kb, ಬಗೆ_ಹೆಸರು E::pair * a) {
    FG_SUPERBLOCK(ಖಂಡ_q6_K);
    fg_decode_q6_K<E>(blk->ql, blk->qh, blk->scales, (ದಶಮಾಂಶ) blk->d, ib, a);
}

// Reorder (SoA) layout: each block field is one stream over the nblocks of the matrix (of the expert
// slice for MUL_MAT_ID), in the order reorder_qw() writes them. Block ib holds k step kb.
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು ಖಂಡ_q_t> ರಚನೆ fg_soa {
    ಸ್ಥಿರ constexpr ತಾರ್ಕಿಕ supported = ಸುಳ್ಳು;
};

ಮಾದರಿ <> ರಚನೆ fg_soa<ಖಂಡ_q8_0> {
    ಸ್ಥಿರ constexpr ತಾರ್ಕಿಕ supported = ನಿಜ;
    // [qs][d]
    ಮಾದರಿ <ಬಗೆ_ಹೆಸರು E>
    ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ stage(ಸ್ಥಿರಾಂಕ uint8_t * x, ಸ್ಥಿರಾಂಕ size_t nblocks, ಸ್ಥಿರಾಂಕ size_t ib, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ,
                                      ಬಗೆ_ಹೆಸರು E::pair * a) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = (ದಶಮಾಂಶ) ((ಸ್ಥಿರಾಂಕ sycl::half *) (x + nblocks * QK8_0))[ib];
        fg_decode_q8_0<E>((ಸ್ಥಿರಾಂಕ int8_t *) x + ib * QK8_0, d, a);
    }
};

ಮಾದರಿ <> ರಚನೆ fg_soa<ಖಂಡ_q4_K> {
    ಸ್ಥಿರ constexpr ತಾರ್ಕಿಕ supported = ನಿಜ;
    // [qs][scales][dm]
    ಮಾದರಿ <ಬಗೆ_ಹೆಸರು E>
    ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ stage(ಸ್ಥಿರಾಂಕ uint8_t * x, ಸ್ಥಿರಾಂಕ size_t nblocks, ಸ್ಥಿರಾಂಕ size_t ib, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kb,
                                      ಬಗೆ_ಹೆಸರು E::pair * a) {
        ಸ್ಥಿರಾಂಕ uint8_t *   scales = x + nblocks * (QK_K / 2);
        ಸ್ಥಿರಾಂಕ sycl::half2 dm     = ((ಸ್ಥಿರಾಂಕ sycl::half2 *) (scales + nblocks * K_SCALE_ಗಾತ್ರ))[ib];
        fg_decode_q4_K<E>(x + ib * (QK_K / 2), scales + ib * K_SCALE_ಗಾತ್ರ, dm, kb % (QK_K / 32), a);
    }
};

ಮಾದರಿ <> ರಚನೆ fg_soa<ಖಂಡ_q5_K> {
    ಸ್ಥಿರ constexpr ತಾರ್ಕಿಕ supported = ನಿಜ;
    // [qs][qh][scales][dm]
    ಮಾದರಿ <ಬಗೆ_ಹೆಸರು E>
    ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ stage(ಸ್ಥಿರಾಂಕ uint8_t * x, ಸ್ಥಿರಾಂಕ size_t nblocks, ಸ್ಥಿರಾಂಕ size_t ib, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kb,
                                      ಬಗೆ_ಹೆಸರು E::pair * a) {
        ಸ್ಥಿರಾಂಕ uint8_t *   qh     = x + nblocks * (QK_K / 2);
        ಸ್ಥಿರಾಂಕ uint8_t *   scales = qh + nblocks * (QK_K / 8);
        ಸ್ಥಿರಾಂಕ sycl::half2 dm     = ((ಸ್ಥಿರಾಂಕ sycl::half2 *) (scales + nblocks * K_SCALE_ಗಾತ್ರ))[ib];
        fg_decode_q5_K<E>(x + ib * (QK_K / 2), qh + ib * (QK_K / 8), scales + ib * K_SCALE_ಗಾತ್ರ, dm,
                          kb % (QK_K / 32), a);
    }
};

ಮಾದರಿ <> ರಚನೆ fg_soa<ಖಂಡ_q6_K> {
    ಸ್ಥಿರ constexpr ತಾರ್ಕಿಕ supported = ನಿಜ;
    // [ql][qh][scales][d]
    ಮಾದರಿ <ಬಗೆ_ಹೆಸರು E>
    ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ stage(ಸ್ಥಿರಾಂಕ uint8_t * x, ಸ್ಥಿರಾಂಕ size_t nblocks, ಸ್ಥಿರಾಂಕ size_t ib, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kb,
                                      ಬಗೆ_ಹೆಸರು E::pair * a) {
        ಸ್ಥಿರಾಂಕ uint8_t * qh     = x + nblocks * (QK_K / 2);
        ಸ್ಥಿರಾಂಕ uint8_t * scales = qh + nblocks * (QK_K / 4);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ     d      = (ದಶಮಾಂಶ) ((ಸ್ಥಿರಾಂಕ sycl::half *) (scales + nblocks * (QK_K / 16)))[ib];
        fg_decode_q6_K<E>(x + ib * (QK_K / 2), qh + ib * (QK_K / 4), (ಸ್ಥಿರಾಂಕ int8_t *) scales + ib * (QK_K / 16), d,
                          kb % (QK_K / 32), a);
    }
};


// one SG_ROWS x BN output tile: B columns [b0, b0 + BN) of packed_b go to dst columns [n0, n1),
// n1 - n0 <= BN
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು S, ಬಗೆ_ಹೆಸರು ಖಂಡ_q_t, ತಾರ್ಕಿಕ reordered>
ಸ್ಥಿರ ಶೂನ್ಯ fused_dequant_gemm_tile(
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q_t * __restrict__ x,
    ಸ್ಥಿರಾಂಕ ಬಗೆ_ಹೆಸರು S::tsb * __restrict__ packed_b,
    ದಶಮಾಂಶ * __restrict__ dst,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ M, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ Npad, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ K, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ldd,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ b0, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n0, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n1,
    sycl::local_accessor<ಬಗೆ_ಹೆಸರು S::tsa, 1> tile_a,
    sycl::local_accessor<ದಶಮಾಂಶ, 1> tile_c,
    ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<2> & ಅಂಶ) {
    ಆದರೆ constexpr (fg_built<S::SG>()) {
        ಬಳಸಿ EA = ಬಗೆ_ಹೆಸರು S::EA;
        ಬಳಸಿ TA = ಬಗೆ_ಹೆಸರು S::ta;
        ಬಳಸಿ TB = ಬಗೆ_ಹೆಸರು S::tb;
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ sg     = ಅಂಶ.get_sub_ಗುಂಪು();
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ  sg_id  = sg.get_ಗುಂಪು_id()[0];
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ  lane   = sg.get_local_id()[0];
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ  m0     = ಅಂಶ.get_ಗುಂಪು(1) * S::SG_ಅಡ್ಡಸಾಲುಗಳು;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ  nstep  = K / FG_BK;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ  a_ತಳ = sg_id * S::SG_ಅಡ್ಡಸಾಲುಗಳು * FG_BK;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ  c_ತಳ = sg_id * S::SG_ಅಡ್ಡಸಾಲುಗಳು * S::BN;

        mx::joint_ಮಾತೃಕೆ<sycl::sub_ಗುಂಪು, ದಶಮಾಂಶ, mx::use::ಸಂಚಯಕ, S::TM, S::TN> acc[S::MT][S::NT];
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ mt = 0; mt < S::MT; ++mt) {
#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ nt = 0; nt < S::NT; ++nt) {
                mx::joint_ಮಾತೃಕೆ_fill(sg, acc[mt][nt], 0.0f);
            }
        }

        // lane decodes rows lane, lane + SG, ... of the sub-group's SG_ROWS
        constexpr ಪೂರ್ಣಾಂಕ         KPB     = fg_ಖಂಡ_traits<ಖಂಡ_q_t>::qk / FG_BK; // k steps per block
        ಸ್ಥಿರಾಂಕ size_t          bpr     = K / fg_ಖಂಡ_traits<ಖಂಡ_q_t>::qk;
        ಸ್ಥಿರಾಂಕ size_t          nblocks = (size_t) M * bpr;
        size_t                ಅಡ್ಡಸಾಲು_blk[S::RPL];
        ತಾರ್ಕಿಕ                  ಅಡ್ಡಸಾಲು_ok[S::RPL];
        ಬಗೆ_ಹೆಸರು EA::pair *   a[S::RPL];
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ r = 0; r < S::RPL; ++r) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = m0 + r * S::SG + lane;
            ಅಡ್ಡಸಾಲು_ok[r]  = ಅಡ್ಡಸಾಲು < M;
            ಅಡ್ಡಸಾಲು_blk[r] = (size_t) (ಅಡ್ಡಸಾಲು_ok[r] ? ಅಡ್ಡಸಾಲು : 0) * bpr;
            a[r]       = (ಬಗೆ_ಹೆಸರು EA::pair *) &tile_a[a_ತಳ + (r * S::SG + lane) * FG_BK];
        }

        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ b_ptr = sycl::ವಿಳಾಸ_space_cast<sycl::access::ವಿಳಾಸ_space::global_space,
                                                    sycl::access::decorated::no>(packed_b);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ b_stride = Npad * S::VNNI;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kb_begin = (sg_id * nstep) / FG_KSPLIT;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kb_ಕೊನೆ   = ((sg_id + 1) * nstep) / FG_KSPLIT;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ kb = kb_begin; kb < kb_ಕೊನೆ; ++kb) {
#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ r = 0; r < S::RPL; ++r) {
                ಆದರೆ (ಅಡ್ಡಸಾಲು_ok[r]) {
                    ಆದರೆ constexpr (reordered) {
                        fg_soa<ಖಂಡ_q_t>::ಮಾದರಿ stage<EA>((ಸ್ಥಿರಾಂಕ uint8_t *) x, nblocks, ಅಡ್ಡಸಾಲು_blk[r] + kb / KPB, kb,
                                                              a[r]);
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        fg_stage_a<EA>(x + ಅಡ್ಡಸಾಲು_blk[r], kb, a[r]);
                    }
                } ಇಲ್ಲದಿದ್ದರೆ {
#ಸೂಚನೆ unroll
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < FG_BK / 2; ++j) {
                        a[r][j] = EA::make(0.0f, 0.0f);
                    }
                }
            }
            sycl::ಗುಂಪು_barrier(sg);

#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ kt = 0; kt < FG_BK / S::TK; ++kt) {
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kq0 = (kb * FG_BK + kt * S::TK) / S::VNNI;
                mx::joint_ಮಾತೃಕೆ<sycl::sub_ಗುಂಪು, TB, mx::use::b, S::TK, S::TN, S::b_layout> sub_b[S::NT];
#ಸೂಚನೆ unroll
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ nt = 0; nt < S::NT; ++nt) {
                    mx::joint_ಮಾತೃಕೆ_ತುಂಬು(sg, sub_b[nt], b_ptr + (size_t) kq0 * b_stride + (b0 + nt * S::TN) * S::VNNI, b_stride);
                }
#ಸೂಚನೆ unroll
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ mt = 0; mt < S::MT; ++mt) {
                    mx::joint_ಮಾತೃಕೆ<sycl::sub_ಗುಂಪು, TA, mx::use::a, S::TM, S::TK, mx::layout::ಅಡ್ಡಸಾಲು_major> sub_a;
                    mx::joint_ಮಾತೃಕೆ_ತುಂಬು(sg, sub_a,
                        tile_a.ಮಾದರಿ get_multi_ptr<sycl::access::decorated::no>() + a_ತಳ + (mt * S::TM) * FG_BK + kt * S::TK,
                        FG_BK);
#ಸೂಚನೆ unroll
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ nt = 0; nt < S::NT; ++nt) {
                        mx::joint_ಮಾತೃಕೆ_mad(sg, acc[mt][nt], sub_a, sub_b[nt], acc[mt][nt]);
                    }
                }
            }
            // the next step overwrites tile_a
            sycl::ಗುಂಪು_barrier(sg);
        }

#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ mt = 0; mt < S::MT; ++mt) {
#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ nt = 0; nt < S::NT; ++nt) {
                mx::joint_ಮಾತೃಕೆ_store(sg, acc[mt][nt],
                    tile_c.ಮಾದರಿ get_multi_ptr<sycl::access::decorated::no>() + c_ತಳ + (mt * S::TM) * S::BN + nt * S::TN,
                    S::BN, mx::layout::ಅಡ್ಡಸಾಲು_major);
            }
        }
        sycl::ಗುಂಪು_barrier(ಅಂಶ.get_ಗುಂಪು());

        // sum the K splits; consecutive lanes write consecutive rows of one dst column
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ idx = ಅಂಶ.get_local_ರೇಖೀಯ_id(); idx < S::SG_ಅಡ್ಡಸಾಲುಗಳು * S::BN; idx += S::WG_ಗಾತ್ರ) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ r = idx % S::SG_ಅಡ್ಡಸಾಲುಗಳು;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ c = idx / S::SG_ಅಡ್ಡಸಾಲುಗಳು;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ m = m0 + r;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n = n0 + c;
            ಆದರೆ (m < M && n < n1) {
                ದಶಮಾಂಶ ಮೊತ್ತ = 0.0f;
#ಸೂಚನೆ unroll
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ s = 0; s < FG_KSPLIT; ++s) {
                    ಮೊತ್ತ += tile_c[s * S::SG_ಅಡ್ಡಸಾಲುಗಳು * S::BN + r * S::BN + c];
                }
                dst[(size_t) n * ldd + m] = ಮೊತ್ತ;
            }
        }
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು S, ಬಗೆ_ಹೆಸರು ಖಂಡ_q_t, ತಾರ್ಕಿಕ reordered = ಸುಳ್ಳು>
ಸ್ಥಿರ ಶೂನ್ಯ fused_dequant_gemm_launch(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src0, ಸ್ಥಿರಾಂಕ ಬಗೆ_ಹೆಸರು S::tsb * packed, ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ M,
                                      ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ N, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ Npad, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ K, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ldd,
                                      ಸ್ಥಿರಾಂಕ int64_t groups_n, ಸ್ಥಿರಾಂಕ int64_t groups_m, dpct::ಸರತಿ_ptr stream) {
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<ಬಗೆ_ಹೆಸರು S::tsa, 1> tile_a(FG_KSPLIT * S::SG_ಅಡ್ಡಸಾಲುಗಳು * FG_BK, cgh);
        sycl::local_accessor<ದಶಮಾಂಶ, 1>          tile_c(FG_KSPLIT * S::SG_ಅಡ್ಡಸಾಲುಗಳು * S::BN, cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(groups_n, groups_m * S::WG_ಗಾತ್ರ), sycl::range<2>(1, S::WG_ಗಾತ್ರ)),
            [=](sycl::nd_ಅಂಶ<2> ಅಂಶ) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(S::SG)]] {
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n0 = ಅಂಶ.get_ಗುಂಪು(0) * S::BN;
                fused_dequant_gemm_tile<S, ಖಂಡ_q_t, reordered>((ಸ್ಥಿರಾಂಕ ಖಂಡ_q_t *) src0, packed, dst, M, Npad, K,
                                                      ldd, n0, n0, N, tile_a, tile_c, ಅಂಶ);
            });
    });
}

// grouped: work-group (t, mt) is tile t of the schedule; its B columns sit at t * BN
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು S, ಬಗೆ_ಹೆಸರು ಖಂಡ_q_t, ತಾರ್ಕಿಕ reordered>
ಸ್ಥಿರ ಶೂನ್ಯ grouped_dequant_gemm_launch(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * src0_dd, ಸ್ಥಿರಾಂಕ size_t expert_stride,
                                        ಸ್ಥಿರಾಂಕ ggml_sycl_gg_tile * tiles_ptr, ಸ್ಥಿರಾಂಕ ಬಗೆ_ಹೆಸರು S::tsb * packed, ದಶಮಾಂಶ * dst,
                                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ M, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ Npad, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ K, ಸ್ಥಿರಾಂಕ int64_t n_tiles,
                                        ಸ್ಥಿರಾಂಕ int64_t groups_m, dpct::ಸರತಿ_ptr stream) {
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<ಬಗೆ_ಹೆಸರು S::tsa, 1> tile_a(FG_KSPLIT * S::SG_ಅಡ್ಡಸಾಲುಗಳು * FG_BK, cgh);
        sycl::local_accessor<ದಶಮಾಂಶ, 1>          tile_c(FG_KSPLIT * S::SG_ಅಡ್ಡಸಾಲುಗಳು * S::BN, cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(n_tiles, groups_m * S::WG_ಗಾತ್ರ), sycl::range<2>(1, S::WG_ಗಾತ್ರ)),
            [=](sycl::nd_ಅಂಶ<2> ಅಂಶ) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(S::SG)]] {
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ               t    = ಅಂಶ.get_ಗುಂಪು(0);
                ಸ್ಥಿರಾಂಕ ggml_sycl_gg_tile tile = tiles_ptr[t];
                ಸ್ಥಿರಾಂಕ ಖಂಡ_q_t *       x    = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q_t *) (src0_dd + (size_t) tile.expert * expert_stride);
                fused_dequant_gemm_tile<S, ಖಂಡ_q_t, reordered>(x, packed, dst, M, Npad, K, M, t * S::BN, tile.n0,
                                                                 tile.n1, tile_a, tile_c, ಅಂಶ);
            });
    });
}

// src1 f32 rows -> packed [K/V][n_tiles*BN][V], tile t holds its rows [n0, n1) at columns t*BN..,
// zero past n1. The column runs fastest so a sub-group writes one contiguous run.
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು S>
ಸ್ಥಿರ ಶೂನ್ಯ grouped_gemm_pack_b(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * y, ಬಗೆ_ಹೆಸರು S::tsb * packed, ಸ್ಥಿರಾಂಕ ggml_sycl_gg_tile * tiles, ಪೂರ್ಣಾಂಕ Npad,
                                ಪೂರ್ಣಾಂಕ K, dpct::ಸರತಿ_ptr stream) {
    ಬಳಸಿ E        = ಬಗೆ_ಹೆಸರು S::EB;
    constexpr ಪೂರ್ಣಾಂಕ V = S::VNNI;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kqs   = K / V;
    stream->parallel_for(sycl::range<1>((size_t) Npad * kqs), [=](sycl::id<1> id) {
        ಸ್ಥಿರಾಂಕ size_t idx = id[0];
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ    kq  = idx / Npad;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ    n   = idx - (size_t) kq * Npad;
        ಸ್ಥಿರಾಂಕ ggml_sycl_gg_tile tile = tiles[n / S::BN];
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ    ಅಡ್ಡಸಾಲು = tile.n0 + n % S::BN;
        // one guarded load run per work-item, as a per-element select costs ~1.5% prefill
        ಬಗೆ_ಹೆಸರು S::tsb vals[V] = {};
        ಆದರೆ (ಅಡ್ಡಸಾಲು < tile.n1) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src = y + (size_t) ಅಡ್ಡಸಾಲು * K + V * kq;
#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ v = 0; v < V; ++v) {
                vals[v] = E::cvt(src[v]);
            }
        }
        ಬಗೆ_ಹೆಸರು S::tsb * ಹೊರಹೋಗು = packed + ((size_t) kq * Npad + n) * V;
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ v = 0; v < V; ++v) {
            ಹೊರಹೋಗು[v] = vals[v];
        }
    });
}

// q8_0 and the k-quants take only the grouped path. The plain kernel decodes A again for every BN columns
// of a dense batch, and for these formats that costs more than the one dequantization of the library GEMM.
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T> ಸ್ಥಿರ constexpr ತಾರ್ಕಿಕ fg_plain_ok() {
    ಹಿಂತಿರುಗಿ !std::is_same_v<T, ಖಂಡ_q8_0> && !std::is_same_v<T, ಖಂಡ_q4_K> && !std::is_same_v<T, ಖಂಡ_q5_K> &&
           !std::is_same_v<T, ಖಂಡ_q6_K>;
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T, ತಾರ್ಕಿಕ R> ರಚನೆ fg_ಟ್ಯಾಗ್ {
    ಬಳಸಿ type                     = T;
    ಸ್ಥಿರ constexpr ತಾರ್ಕಿಕ reordered = R;
};

// calls f(fg_tag<block_q_t, reordered>{}) for the weight format and layout; false if it has no A stage
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T, ಬಗೆ_ಹೆಸರು F> ಸ್ಥಿರ ತಾರ್ಕಿಕ fg_ಭೇಟಿ_layout(ತಾರ್ಕಿಕ reordered, F && f) {
    ಆದರೆ (!reordered) {
        f(fg_ಟ್ಯಾಗ್<T, ಸುಳ್ಳು>{});
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }
    ಆದರೆ constexpr (fg_soa<T>::supported) {
        f(fg_ಟ್ಯಾಗ್<T, ನಿಜ>{});
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು F> ಸ್ಥಿರ ತಾರ್ಕಿಕ fg_ಭೇಟಿ_type(ggml_type type, ತಾರ್ಕಿಕ reordered, F && f) {
    ಆಯ್ಕೆ (type) {
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL:  ಹಿಂತಿರುಗಿ fg_ಭೇಟಿ_layout<ಖಂಡ_iq4_nl>(reordered, f);
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_S:   ಹಿಂತಿರುಗಿ fg_ಭೇಟಿ_layout<ಖಂಡ_iq3_s>(reordered, f);
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_XS:  ಹಿಂತಿರುಗಿ fg_ಭೇಟಿ_layout<ಖಂಡ_iq4_xs>(reordered, f);
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_XXS: ಹಿಂತಿರುಗಿ fg_ಭೇಟಿ_layout<ಖಂಡ_iq3_xxs>(reordered, f);
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XXS: ಹಿಂತಿರುಗಿ fg_ಭೇಟಿ_layout<ಖಂಡ_iq2_xxs>(reordered, f);
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XS:  ಹಿಂತಿರುಗಿ fg_ಭೇಟಿ_layout<ಖಂಡ_iq2_xs>(reordered, f);
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_S:   ಹಿಂತಿರುಗಿ fg_ಭೇಟಿ_layout<ಖಂಡ_iq2_s>(reordered, f);
        ಸ್ಥಿತಿ GGML_TYPE_IQ1_S:   ಹಿಂತಿರುಗಿ fg_ಭೇಟಿ_layout<ಖಂಡ_iq1_s>(reordered, f);
        ಸ್ಥಿತಿ GGML_TYPE_IQ1_M:   ಹಿಂತಿರುಗಿ fg_ಭೇಟಿ_layout<ಖಂಡ_iq1_m>(reordered, f);
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:    ಹಿಂತಿರುಗಿ fg_ಭೇಟಿ_layout<ಖಂಡ_q8_0>(reordered, f);
        ಸ್ಥಿತಿ GGML_TYPE_Q4_K:    ಹಿಂತಿರುಗಿ fg_ಭೇಟಿ_layout<ಖಂಡ_q4_K>(reordered, f);
        ಸ್ಥಿತಿ GGML_TYPE_Q5_K:    ಹಿಂತಿರುಗಿ fg_ಭೇಟಿ_layout<ಖಂಡ_q5_K>(reordered, f);
        ಸ್ಥಿತಿ GGML_TYPE_Q6_K:    ಹಿಂತಿರುಗಿ fg_ಭೇಟಿ_layout<ಖಂಡ_q6_K>(reordered, f);
        ಪೂರ್ವನಿಯೋಜಿತ:                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು S>
ಸ್ಥಿರ ತಾರ್ಕಿಕ fg_fused_ಸತತ(ggml_type src0_type, ತಾರ್ಕಿಕ src0_reordered, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src0, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src1, ggml_type src1_type,
                         ದಶಮಾಂಶ * dst, int64_t M, int64_t N, int64_t K, int64_t ldd, ggml_sycl_pool & pool,
                         dpct::ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ int64_t groups_n = (N + S::BN - 1) / S::BN;
    ಸ್ಥಿರಾಂಕ int64_t groups_m = (M + S::SG_ಅಡ್ಡಸಾಲುಗಳು - 1) / S::SG_ಅಡ್ಡಸಾಲುಗಳು;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     Npad     = (ಪೂರ್ಣಾಂಕ) (groups_n * S::BN);

    // src1 is read in its own type: one pass, converted in registers only if B differs
    ggml_sycl_pool_alloc<ಬಗೆ_ಹೆಸರು S::tsb> packed_b(pool, (size_t) K * Npad);
    ಆದರೆ (src1_type == GGML_TYPE_F16) {
        fused_gemm_pack_b<ಬಗೆ_ಹೆಸರು S::EB>((ಸ್ಥಿರಾಂಕ sycl::half *) src1, packed_b.get(), (ಪೂರ್ಣಾಂಕ) N, Npad, (ಪೂರ್ಣಾಂಕ) K, stream);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src1_type == GGML_TYPE_BF16) {
        fused_gemm_pack_b<ಬಗೆ_ಹೆಸರು S::EB>((ಸ್ಥಿರಾಂಕ fg_bf16 *) src1, packed_b.get(), (ಪೂರ್ಣಾಂಕ) N, Npad, (ಪೂರ್ಣಾಂಕ) K, stream);
    } ಇಲ್ಲದಿದ್ದರೆ {
        fused_gemm_pack_b<ಬಗೆ_ಹೆಸರು S::EB>((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src1, packed_b.get(), (ಪೂರ್ಣಾಂಕ) N, Npad, (ಪೂರ್ಣಾಂಕ) K, stream);
    }

    ಸ್ಥಿರಾಂಕ ಬಗೆ_ಹೆಸರು S::tsb * packed = packed_b.get();
    ಹಿಂತಿರುಗಿ fg_ಭೇಟಿ_type(src0_type, src0_reordered, [&](ಸ್ವಯಂಚಾಲಿತ ಟ್ಯಾಗ್) {
        ಬಳಸಿ ಖಂಡ_q_t = ಬಗೆ_ಹೆಸರು decltype(ಟ್ಯಾಗ್)::type;
        ಆದರೆ constexpr (fg_plain_ok<ಖಂಡ_q_t>()) {
            fused_dequant_gemm_launch<S, ಖಂಡ_q_t, decltype(ಟ್ಯಾಗ್)::reordered>(src0, packed, dst, (ಪೂರ್ಣಾಂಕ) M, (ಪೂರ್ಣಾಂಕ) N,
                                                                              Npad, (ಪೂರ್ಣಾಂಕ) K, (ಪೂರ್ಣಾಂಕ) ldd, groups_n,
                                                                              groups_m, stream);
        }
    });
}

ತಾರ್ಕಿಕ ggml_sycl_fused_dequant_gemm(ggml_type src0_type, ತಾರ್ಕಿಕ src0_reordered, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src0, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src1, ggml_type src1_type,
                                  int32_t src1_prec, ದಶಮಾಂಶ * dst, int64_t M, int64_t N, int64_t K, int64_t ldd,
                                  ggml_sycl_pool & pool, dpct::ಸರತಿ_ptr stream) {
    // every BN columns dequantize A again, so wide N is left to the library GEMM.
    // src0_reordered: the persistent weight reorder may have been installed on this
    // tensor by an earlier small-batch mul_mat; the A stage then has to decode the
    // SoA layout. Types without an SoA stage (fg_soa<T>::supported == false)
    // decline via fg_visit_layout and the caller falls through to the
    // reorder-aware conversion route.
    ತಾರ್ಕಿಕ plain_ok = ಸುಳ್ಳು;
    fg_ಭೇಟಿ_type(src0_type, src0_reordered, [&](ಸ್ವಯಂಚಾಲಿತ ಟ್ಯಾಗ್) { plain_ok = fg_plain_ok<ಬಗೆ_ಹೆಸರು decltype(ಟ್ಯಾಗ್)::type>(); });
    ಆದರೆ (g_ggml_sycl_dynamic_precision == GGML_SYCL_DYNAMIC_PRECISION_F32 ||
        !ggml_sycl_xmx_gather_type_enabled(src0_type) || !plain_ok) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (src1_type != GGML_TYPE_F32 && src1_type != GGML_TYPE_F16 && src1_type != GGML_TYPE_BF16) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (!ggml_sycl_fused_dequant_gemm_shape_ok(src0_type, M, N, K, ldd)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ combo = fg_pick_combo(stream, src1_type, src1_prec);
    ಆದರೆ (combo < 0) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ತಾರ್ಕಿಕ launched = ಸುಳ್ಳು;
    fg_ಭೇಟಿ_combo(combo, [&](ಸ್ವಯಂಚಾಲಿತ s) {
        launched = fg_fused_ಸತತ<decltype(s)>(src0_type, src0_reordered, src0, src1, src1_type, dst, M, N, K, ldd, pool, stream);
    });
    ಹಿಂತಿರುಗಿ launched;
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು S>
ಸ್ಥಿರ ತಾರ್ಕಿಕ fg_grouped_ಸತತ(ggml_type src0_type, ತಾರ್ಕಿಕ reordered, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src0_ತಳ, size_t expert_stride,
                           ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src1, ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ int64_t * expert_ಅಡ್ಡಸಾಲು_offsets, int64_t n_as, int64_t M,
                           int64_t K, std::vector<ggml_sycl_gg_tile> & tiles, ggml_sycl_pool & pool,
                           dpct::ಸರತಿ_ptr stream) {
    // the host knows every slice, so it lays out the work-groups: no search on the device
    tiles.clear();
    ಗಾಗಿ (int64_t e = 0; e < n_as; ++e) {
        ಸ್ಥಿರಾಂಕ int64_t end = expert_ಅಡ್ಡಸಾಲು_offsets[e + 1];
        ಗಾಗಿ (int64_t n0 = expert_ಅಡ್ಡಸಾಲು_offsets[e]; n0 < end; n0 += S::BN) {
            tiles.push_back({ (int32_t) e, (int32_t) n0, (int32_t) std::min<int64_t>(n0 + S::BN, end) });
        }
    }
    ಸ್ಥಿರಾಂಕ int64_t n_tiles  = tiles.size();
    ಸ್ಥಿರಾಂಕ int64_t groups_m = (M + S::SG_ಅಡ್ಡಸಾಲುಗಳು - 1) / S::SG_ಅಡ್ಡಸಾಲುಗಳು;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     Npad     = (ಪೂರ್ಣಾಂಕ) (n_tiles * S::BN);

    ಸ್ಥಿರಾಂಕ int64_t max_tiles = grouped_gemm_max_tiles(expert_ಅಡ್ಡಸಾಲು_offsets[n_as], n_as, S::BN);
    GGML_ASSERT(n_tiles <= max_tiles);
    ggml_sycl_pool_alloc<ggml_sycl_gg_tile> tiles_dev(pool, max_tiles);
    SYCL_CHECK(CHECK_TRY_ERROR(stream->ಸ್ಮರಣೆ_ನಕಲು(tiles_dev.get(), tiles.ದತ್ತಾಂಶ(), n_tiles * ಗಾತ್ರ(ggml_sycl_gg_tile))));

    ggml_sycl_pool_alloc<ಬಗೆ_ಹೆಸರು S::tsb> packed_b(pool, (size_t) K * max_tiles * S::BN);
    grouped_gemm_pack_b<S>(src1, packed_b.get(), tiles_dev.get(), Npad, (ಪೂರ್ಣಾಂಕ) K, stream);

    ಸ್ಥಿರಾಂಕ ಬಗೆ_ಹೆಸರು S::tsb *   packed    = packed_b.get();
    ಸ್ಥಿರಾಂಕ ggml_sycl_gg_tile * tiles_ptr = tiles_dev.get();
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *              src0_dd   = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0_ತಳ;
    ಹಿಂತಿರುಗಿ fg_ಭೇಟಿ_type(src0_type, reordered, [&](ಸ್ವಯಂಚಾಲಿತ ಟ್ಯಾಗ್) {
        ಬಳಸಿ T = decltype(ಟ್ಯಾಗ್);
        grouped_dequant_gemm_launch<S, ಬಗೆ_ಹೆಸರು T::type, T::reordered>(src0_dd, expert_stride, tiles_ptr, packed, dst,
                                                                       (ಪೂರ್ಣಾಂಕ) M, Npad, (ಪೂರ್ಣಾಂಕ) K, n_tiles, groups_m,
                                                                       stream);
    });
}

ತಾರ್ಕಿಕ ggml_sycl_grouped_dequant_gemm(ggml_type src0_type, ತಾರ್ಕಿಕ reordered, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src0_ತಳ, size_t expert_stride,
                                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src1, int32_t src1_prec, ದಶಮಾಂಶ * dst,
                                    ಸ್ಥಿರಾಂಕ int64_t * expert_ಅಡ್ಡಸಾಲು_offsets, int64_t n_as, int64_t M, int64_t K,
                                    int64_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು, std::vector<ggml_sycl_gg_tile> & tiles,
                                    ggml_sycl_pool & pool, dpct::ಸರತಿ_ptr stream) {
    int64_t n_ಸಕ್ರಿಯ = 0;
    ಗಾಗಿ (int64_t e = 0; e < n_as; ++e) {
        n_ಸಕ್ರಿಯ += expert_ಅಡ್ಡಸಾಲು_offsets[e + 1] > expert_ಅಡ್ಡಸಾಲು_offsets[e];
    }
    ಆದರೆ (g_ggml_sycl_dynamic_precision == GGML_SYCL_DYNAMIC_PRECISION_F32 ||
        !ggml_sycl_xmx_gather_type_enabled(src0_type) || !fg_ಭೇಟಿ_type(src0_type, reordered, [](ಸ್ವಯಂಚಾಲಿತ) {})) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (!ggml_sycl_grouped_dequant_gemm_shape_ok(src0_type, M, K, ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು, n_ಸಕ್ರಿಯ)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ combo = fg_pick_combo(stream, GGML_TYPE_F32, src1_prec);
    ಆದರೆ (combo < 0) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ತಾರ್ಕಿಕ launched = ಸುಳ್ಳು;
    fg_ಭೇಟಿ_combo(combo, [&](ಸ್ವಯಂಚಾಲಿತ s) {
        launched = fg_grouped_ಸತತ<decltype(s)>(src0_type, reordered, src0_ತಳ, expert_stride, src1, dst,
                                               expert_ಅಡ್ಡಸಾಲು_offsets, n_as, M, K, tiles, pool, stream);
    });
    ಹಿಂತಿರುಗಿ launched;
}
