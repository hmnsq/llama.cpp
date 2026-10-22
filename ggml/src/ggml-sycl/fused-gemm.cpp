#অন্তর্ভাব "fused-gemm.hpp"

#অন্তর্ভাব <sycl/ext/oneapi/matrix/matrix.hpp>

#অন্তর্ভাব <algorithm>
#অন্তর্ভাব <string>
#অন্তর্ভাব <tuple>
#অন্তর্ভাব <mutex>
#অন্তর্ভাব <set>
#অন্তর্ভাব <unordered_map>

নামাকাশ mx = sycl::ext::oneapi::experimental::matrix;

// FG_ / fg_ is short for fused GEMM: the weights are dequantized inside the GEMM, into the XMX tiles.

// A k step is one 32-value weight sub-block; iq3_s and the other superblock formats split their
// superblock into steps of this width. The sub-groups of a work-group each walk their own K range
// and are summed at the end.
স্থির constexpr পূর্ণসংখ্যা FG_BK     = QK4_NL;
স্থির constexpr পূর্ণসংখ্যা FG_KSPLIT = 4;

// Element traits of one joint_matrix operand type. The A stage and the B pack compute in f32 and
// convert once, in registers, when they write the element, so any type costs the same one pass.
//   store: storage in SLM (A) and in the packed B buffer
//   mtype: matrix_type in matrix_combinations
//   mode:  GGML_SYCL_DYNAMIC_PRECISION value that selects this type
//   src:   ggml type that needs no conversion into this type (GGML_TYPE_COUNT: none)
//   slow:  XMX throughput class, 0 is fastest. f16 and bf16 share the DPAS rate; tf32 does half the
//          K per instruction. B60, Qwen3-30B-A3B pp512: f16 1108, bf16 1000, tf32 751 t/s
template <typename T> কাঠামো fg_elem;

template <> কাঠামো fg_elem<sycl::half> {
    using store = sycl::half;
    using pair  = sycl::half2;
    স্থির constexpr mx::matrix_type mtype = mx::matrix_type::fp16;
    স্থির constexpr পূর্ণসংখ্যা             mode  = GGML_SYCL_DYNAMIC_PRECISION_F16;
    স্থির constexpr ggml_type       src   = GGML_TYPE_F16;
    স্থির constexpr পূর্ণসংখ্যা             mant  = 10;
    স্থির constexpr পূর্ণসংখ্যা             slow  = 0;
    স্থির store cvt(দশমিক x) { ফেরত (store) x; }
    স্থির pair make(দশমিক x, দশমিক y) { ফেরত pair((store) x, (store) y); }
};

// tf32 rounds to nearest even with plain bit ops: round_to_tf32 needs a SPIR-V extension that the
// DG2 AOT target rejects
স্থির অন্তঃরেখ uint32_t fg_round_bits(দশমিক x, পূর্ণসংখ্যা drop) {
    ধ্রুবক uint32_t u = sycl::bit_cast<uint32_t>(x);
    যদি ((u & 0x7f800000u) == 0x7f800000u) {
        ফেরত (u & 0x7fffffu) ? u | (1u << drop) : u; // nan stays nan
    }
    ফেরত u + ((1u << (drop - 1)) - 1) + ((u >> drop) & 1);
}

কাঠামো alignas(4) fg_bf16x2 {
    sycl::ext::oneapi::bfloat16 x, y;
};

template <> কাঠামো fg_elem<sycl::ext::oneapi::bfloat16> {
    using store = sycl::ext::oneapi::bfloat16;
    using pair  = fg_bf16x2;
    স্থির constexpr mx::matrix_type mtype = mx::matrix_type::bf16;
    স্থির constexpr পূর্ণসংখ্যা             mode  = GGML_SYCL_DYNAMIC_PRECISION_BF16;
    স্থির constexpr ggml_type       src   = GGML_TYPE_BF16;
    স্থির constexpr পূর্ণসংখ্যা             mant  = 7;
    স্থির constexpr পূর্ণসংখ্যা             slow  = 0;
    স্থির store cvt(দশমিক x) { ফেরত store(x); }
    স্থির pair make(দশমিক x, দশমিক y) { ফেরত { cvt(x), cvt(y) }; }
};

// tf32 keeps f32 range and f16 mantissa, in f32 storage
template <> কাঠামো fg_elem<mx::precision::tf32> {
    using store = দশমিক;
    using pair  = sycl::float2;
    স্থির constexpr mx::matrix_type mtype = mx::matrix_type::tf32;
    স্থির constexpr পূর্ণসংখ্যা             mode  = GGML_SYCL_DYNAMIC_PRECISION_TF32;
    স্থির constexpr ggml_type       src   = GGML_TYPE_গুনতি;
    স্থির constexpr পূর্ণসংখ্যা             mant  = 10;
    স্থির constexpr পূর্ণসংখ্যা             slow  = 1;
    স্থির store cvt(দশমিক x) { ফেরত sycl::bit_cast<দশমিক>(fg_round_bits(x, 13) & ~0x1fffu); }
    স্থির pair make(দশমিক x, দশমিক y) { ফেরত pair(cvt(x), cvt(y)); }
};

// One joint_matrix combination (A type, B type, TM x TN x TK, sub-group size; C and D are f32) and
// the tiling built on it. A sub-group owns SG_ROWS rows of A (at least 16) and BN (at least 32)
// columns of B. A and B may differ: the device lists the pairs it supports.
template <typename TA, typename TB, পূর্ণসংখ্যা TM_, পূর্ণসংখ্যা TN_, পূর্ণসংখ্যা TK_, পূর্ণসংখ্যা SG_> কাঠামো fg_combo {
    using ta  = TA;
    using tb  = TB;
    using EA  = fg_elem<TA>;
    using EB  = fg_elem<TB>;
    using tsa = typename EA::store;
    using tsb = typename EB::store;
    স্থির constexpr পূর্ণসংখ্যা TM = TM_;
    স্থির constexpr পূর্ণসংখ্যা TN = TN_;
    স্থির constexpr পূর্ণসংখ্যা TK = TK_;
    স্থির constexpr পূর্ণসংখ্যা SG = SG_;
    স্থির constexpr পূর্ণসংখ্যা VNNI    = 4 / আকার(tsb);  // K rows of B packed in one 32-bit word
    স্থির constexpr পূর্ণসংখ্যা SG_ROWS = TM > 16 ? TM : 16;
    স্থির constexpr পূর্ণসংখ্যা RPL     = SG_ROWS / SG;     // A rows one lane decodes per k step
    স্থির constexpr পূর্ণসংখ্যা MT      = SG_ROWS / TM;
    স্থির constexpr পূর্ণসংখ্যা BN      = TN > 32 ? TN : 32;
    স্থির constexpr পূর্ণসংখ্যা NT      = BN / TN;
    স্থির constexpr পূর্ণসংখ্যা WG_আকার = FG_KSPLIT * SG;
    স্থির constexpr mx::layout b_layout = VNNI == 1 ? mx::layout::row_major : mx::layout::ext_intel_packed;
    // a 64-wide N is mostly padding here and a 32x64 f32 accumulator needs 128 registers per lane,
    // so it spills: 13x slower on B60
    স্থির constexpr যৌক্তিক efficient = TN <= 32;
    static_assert(SG_ROWS % SG == 0 && SG_ROWS % TM == 0 && BN % TN == 0 && FG_BK % TK == 0, "bad tile");
    static_assert(BN <= GGML_SYCL_FG_MAX_N, "header gate must cover the tile width");
};

using fg_half = sycl::half;
using fg_bf16 = sycl::ext::oneapi::bfloat16;
using fg_tf32 = mx::precision::tf32;

// One bit of GGML_SYCL_XMX_GATHER_SHAPES per combination. Only combinations some device lists in
// matrix_combinations are built (appendix of sycl_ext_oneapi_matrix and the runtime's own list).
template <typename F> স্থির শূন্য fg_visit_combo(পূর্ণসংখ্যা idx, F && f);
স্থির constexpr পূর্ণসংখ্যা FG_N_COMBOS = 8;

// A spir64_gen AOT build (GGML_SYCL_XMX_AOT_SG) drops the combinations of the other sub-group size
// entirely: ocloc rejects even an empty kernel that asks for a sub-group size it lacks.
template <পূর্ণসংখ্যা SG> স্থির constexpr যৌক্তিক fg_listed() {
#যদি defined(GGML_SYCL_XMX_AOT_SG)
    ফেরত SG == GGML_SYCL_XMX_AOT_SG;
#অন্যথা
    ফেরত সত্য;
#endif
}

template <typename S, typename F> স্থির শূন্য fg_call_combo(F && f) {
    যদি constexpr (fg_listed<S::SG>()) {
        f(S{});
    }
}

template <typename F> স্থির শূন্য fg_visit_combo(পূর্ণসংখ্যা idx, F && f) {
    নির্বাচন (idx) {
        ক্ষেত্র 0: fg_call_combo<fg_combo<fg_half, fg_half, 8, 16, 16, 16>>(f);  বিরতি; // Xe2, Xe3, Xe-HPC
        ক্ষেত্র 1: fg_call_combo<fg_combo<fg_half, fg_half, 16, 16, 16, 16>>(f); বিরতি; // Xe2, Xe3, Xe-HPC
        ক্ষেত্র 2: fg_call_combo<fg_combo<fg_half, fg_half, 32, 64, 16, 16>>(f); বিরতি; // Xe2, Xe3, Xe-HPC
        ক্ষেত্র 3: fg_call_combo<fg_combo<fg_half, fg_half, 32, 64, 32, 16>>(f); বিরতি; // Xe2, Xe3, Xe-HPC
        ক্ষেত্র 4: fg_call_combo<fg_combo<fg_half, fg_half, 8, 8, 16, 8>>(f);    বিরতি; // Xe-HPG (Arc A), ARL-H
        ক্ষেত্র 5: fg_call_combo<fg_combo<fg_tf32, fg_tf32, 8, 16, 8, 16>>(f);   বিরতি; // Xe2, Xe3, Xe-HPC
        ক্ষেত্র 6: fg_call_combo<fg_combo<fg_bf16, fg_bf16, 8, 16, 16, 16>>(f);  বিরতি; // Xe2, Xe3, Xe-HPC
        ক্ষেত্র 7: fg_call_combo<fg_combo<fg_bf16, fg_bf16, 8, 8, 16, 8>>(f);    বিরতি; // Xe-HPG (Arc A), ARL-H
        পূর্বনির্ধারিত: GGML_ABORT("bad XMX combination %d", idx);
    }
}

// AOT with -fsycl-targets=intel_gpu_*: compile each tile body only for targets with its sub-group
// size, since IGC fails on the other ones. A JIT build keeps them all, but each combination lands in
// its own device image (joint_matrix is an optional kernel feature) and only a combination the
// device reports is launched, so the runtime never asks IGC for the others.
#যদি defined(__SYCL_DEVICE_ONLY__)
#    যদি __SYCL_TARGET_INTEL_GPU_ACM_G10__ || __SYCL_TARGET_INTEL_GPU_ACM_G11__ || __SYCL_TARGET_INTEL_GPU_ACM_G12__ || \
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
#endif

template <পূর্ণসংখ্যা SG> স্থির constexpr যৌক্তিক fg_built() {
#যদি defined(FG_AOT_SG)
    ফেরত SG == FG_AOT_SG;
#অন্যথা
    ফেরত সত্য;
#endif
}

// Upper bound on the tile count when total_rows rows are routed to n_as experts: the worst case gives
// each expert one row and fills whole tiles with the rest. The bound depends only on the shape, not
// on the routing, so the pool reuses one buffer every ubatch instead of keeping one per size seen.
স্থির constexpr int64_t grouped_gemm_max_tiles(int64_t যোগ_rows, int64_t n_as, int64_t BN) {
    ফেরত যোগ_rows <= n_as ? যোগ_rows : n_as + (যোগ_rows - n_as) / BN;
}
// Tiles do not cross experts, so the bound is not ceil(total_rows / BN): 34 rows over 2 experts with
// BN = 16 split 17 + 17 need 2 + 2 tiles, where the ceil gives 3.
static_assert(grouped_gemm_max_tiles(34, 2, 16) == 4);

// the device lists S with an f32 accumulator and output
template <typename S> স্থির যৌক্তিক fg_device_has_combo(ধ্রুবক std::vector<mx::combination> & combinations) {
    জন্য (ধ্রুবক auto & c : combinations) {
        যদি (c.atype == S::EA::mtype && c.btype == S::EB::mtype && c.ctype == mx::matrix_type::fp32 &&
            c.dtype == mx::matrix_type::fp32 &&
            (c.max_msize >= (size_t) S::TM || c.msize == (size_t) S::TM) &&
            (c.max_nsize >= (size_t) S::TN || c.nsize == (size_t) S::TN) &&
            (c.max_ksize >= (size_t) S::TK || c.ksize == (size_t) S::TK)) {
            ফেরত সত্য;
        }
    }
    ফেরত মিথ্যা;
}

template <typename T> স্থির ধ্রুবক অক্ষর * fg_type_name() {
    ফেরত std::is_same_v<T, fg_half> ? "f16" : std::is_same_v<T, fg_bf16> ? "bf16" : "tf32";
}

স্থির std::string fg_combo_name(পূর্ণসংখ্যা idx) {
    std::string name;
    fg_visit_combo(idx, [&](auto s) {
        using S = decltype(s);
        name = std::string(fg_type_name<typename S::ta>()) + "x" + fg_type_name<typename S::tb>() + " " +
               std::to_string(S::TM) + "x" + std::to_string(S::TN) + "x" + std::to_string(S::TK) + " sg" +
               std::to_string(S::SG);
    });
    ফেরত name;
}

// Combinations this build has kernels for and the device lists, one bit each. Cached per device:
// on a mixed box the first caller's verdict is not the others'.
স্থির পূর্ণসংখ্যা fg_device_combos(ধ্রুবক sycl::device & dev) {
    স্থির std::mutex                            mtx;
    স্থির std::unordered_map<sycl::device, পূর্ণসংখ্যা> known;
    std::lock_guard<std::mutex>                  lock(mtx);
    ধ্রুবক auto                                   it = known.find(dev);
    যদি (it != known.end()) {
        ফেরত it->second;
    }
    পূর্ণসংখ্যা available = 0;
    চেষ্টা {
        ধ্রুবক auto combinations = dev.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>();
        ধ্রুবক auto sg_sizes     = dev.get_info<sycl::info::device::sub_group_sizes>();
        জন্য (পূর্ণসংখ্যা idx = 0; idx < FG_N_COMBOS; ++idx) {
            fg_visit_combo(idx, [&](auto s) {
                using S = decltype(s);
                ধ্রুবক যৌক্তিক sg_ok = std::find(sg_sizes.begin(), sg_sizes.end(), (size_t) S::SG) != sg_sizes.end();
                যদি (sg_ok && fg_device_has_combo<S>(combinations)) {
                    available |= 1 << idx;
                }
            });
        }
    } ধরো (ধ্রুবক sycl::exception &) {
        available = 0;
    }
    GGML_LOG_INFO("%s: %s: XMX dequant-GEMM combinations available 0x%x, allowed 0x%x\n", __func__,
                  dev.get_info<sycl::info::device::name>().c_str(), available, g_ggml_sycl_xmx_gather_shapes);
    known.emplace(dev, available);
    ফেরত available;
}

// Rank of combination S for a src1 of type src1_type, lower is better. Order:
//  1. throughput: a tile that does not spill, then the fastest type class of A and B
//  2. B type equal to the src1 type, so the pack is a plain copy
//  3. B at least as precise as f16
//  4. the device's native DPAS tile (8 x SG x 32 bytes of K), then the largest M x K
// A costs nothing to convert: the A stage emits any type at the same cost.
template <typename S> স্থির int64_t fg_rank(ggml_type src1_type) {
    ধ্রুবক int64_t spills  = !S::efficient;
    ধ্রুবক int64_t slow    = std::max(S::EA::slow, S::EB::slow);
    ধ্রুবক int64_t convert = S::EB::src != src1_type;
    ধ্রুবক int64_t lossy   = S::EB::mant < 10;
    ধ্রুবক int64_t foreign = !(S::TM == 8 && S::TN == S::SG);
    ধ্রুবক int64_t mk      = 1024 - S::TM * S::TK;
    ফেরত ((((spills * 2 + slow) * 2 + convert) * 2 + lossy) * 2 + foreign) * 2048 + mk;
}

// whether XMX operands of type mode (a GGML_SYCL_DYNAMIC_PRECISION value) meet the src1 request
// [TAG_GGML_PREC]. f16 lacks the f32 range that BF16 and F32 ask for; an F32 request goes only as far
// down as GGML_SYCL_DYNAMIC_REQUIRED_PRECISION allows.
স্থির যৌক্তিক fg_mode_meets(পূর্ণসংখ্যা mode, int32_t src1_prec) {
    যদি (src1_prec == GGML_PREC_UNDEFINED || src1_prec >= GGML_PREC_F16) {
        ফেরত সত্য;
    }
    যদি (src1_prec >= GGML_PREC_BF16) {
        ফেরত mode != GGML_SYCL_DYNAMIC_PRECISION_F16;
    }
    নির্বাচন (g_ggml_sycl_dynamic_required_precision) {
        ক্ষেত্র GGML_SYCL_DYNAMIC_PRECISION_TF32: ফেরত mode == GGML_SYCL_DYNAMIC_PRECISION_TF32;
        ক্ষেত্র GGML_SYCL_DYNAMIC_PRECISION_BF16: ফেরত mode != GGML_SYCL_DYNAMIC_PRECISION_F16;
        পূর্বনির্ধারিত:                               ফেরত মিথ্যা;
    }
}

// Best allowed combination for this call, or -1 if none. The type is GGML_SYCL_DYNAMIC_PRECISION if it
// meets the src1 request; if not, bf16 then tf32 for a BF16 request (fastest first) and tf32 then bf16
// for an F32 request (most mantissa first).
স্থির পূর্ণসংখ্যা fg_pick_combo(dpct::queue_ptr stream, ggml_type src1_type, int32_t src1_prec) {
    ধ্রুবক sycl::device dev     = stream->get_device();
    ধ্রুবক পূর্ণসংখ্যা          allowed = fg_device_combos(dev) & g_ggml_sycl_xmx_gather_shapes;
    ধ্রুবক যৌক্তিক         f32_req = src1_prec != GGML_PREC_UNDEFINED && src1_prec < GGML_PREC_BF16;
    ধ্রুবক পূর্ণসংখ্যা          modes[] = {
        g_ggml_sycl_dynamic_precision,
        f32_req ? GGML_SYCL_DYNAMIC_PRECISION_TF32 : GGML_SYCL_DYNAMIC_PRECISION_BF16,
        f32_req ? GGML_SYCL_DYNAMIC_PRECISION_BF16 : GGML_SYCL_DYNAMIC_PRECISION_TF32,
    };
    পূর্ণসংখ্যা     best      = -1;
    int64_t best_rank = 0;
    জন্য (পূর্ণসংখ্যা i = 0; i < 3 && best < 0; ++i) {
        ধ্রুবক পূর্ণসংখ্যা mode = modes[i];
        যদি (!fg_mode_meets(mode, src1_prec)) {
            চালু;
        }
        জন্য (পূর্ণসংখ্যা idx = 0; idx < FG_N_COMBOS; ++idx) {
            যদি (!(allowed & (1 << idx))) {
                চালু;
            }
            fg_visit_combo(idx, [&](auto s) {
                using S = decltype(s);
                যদি (S::EA::mode != mode || S::EB::mode != mode) {
                    ফেরত;
                }
                ধ্রুবক int64_t rank = fg_rank<S>(src1_type);
                যদি (best < 0 || rank < best_rank) {
                    best      = idx;
                    best_rank = rank;
                }
            });
        }
    }
    // log each distinct decision once
    স্থির std::mutex                                      mtx;
    স্থির std::set<std::tuple<size_t, পূর্ণসংখ্যা, int32_t, পূর্ণসংখ্যা>> seen;
    std::lock_guard<std::mutex>                            lock(mtx);
    যদি (seen.emplace(std::hash<sycl::device>{}(dev), (পূর্ণসংখ্যা) src1_type, src1_prec, best).second) {
        GGML_LOG_INFO("%s: src1 %s, src1 prec %d -> %s\n", __func__, ggml_type_name(src1_type), src1_prec,
                      best >= 0 ? fg_combo_name(best).c_str() : "none (library GEMM)");
    }
    ফেরত best;
}

// src1 [N][K] -> packed [K/V][Npad][V] so B tiles load straight from global memory
template <typename E, typename T_src>
স্থির শূন্য fused_gemm_pack_b(ধ্রুবক T_src * y, typename E::store * packed, পূর্ণসংখ্যা N, পূর্ণসংখ্যা Npad, পূর্ণসংখ্যা K,
                              dpct::queue_ptr stream) {
    constexpr পূর্ণসংখ্যা V   = 4 / আকার(typename E::store);
    ধ্রুবক পূর্ণসংখ্যা     kqs = K / V;
    stream->parallel_for(sycl::range<1>((size_t) Npad * kqs), [=](sycl::id<1> id) {
        ধ্রুবক পূর্ণসংখ্যা idx = id[0];
        ধ্রুবক পূর্ণসংখ্যা n   = idx / kqs;
        ধ্রুবক পূর্ণসংখ্যা kq  = idx - n * kqs;
        typename E::store vals[V] = {};
        যদি (n < N) {
            ধ্রুবক T_src * src = y + (size_t) n * K + V * kq;
#pragma unroll
            জন্য (পূর্ণসংখ্যা v = 0; v < V; ++v) {
                vals[v] = E::cvt((দশমিক) src[v]);
            }
        }
        typename E::store * out = packed + ((size_t) kq * Npad + n) * V;
#pragma unroll
        জন্য (পূর্ণসংখ্যা v = 0; v < V; ++v) {
            out[v] = vals[v];
        }
    });
}

// A stage: one lane owns one row and decodes FG_BK values of it per k step, with every scale
// folded into the value so the mad below sees plain A elements. One overload per weight format.
template <typename E>
স্থির __dpct_inline__ শূন্য fg_stage_a(ধ্রুবক block_iq4_nl * __restrict__ xrow, ধ্রুবক পূর্ণসংখ্যা kb, typename E::pair * a) {
    ধ্রুবক block_iq4_nl blk = xrow[kb];
    ধ্রুবক দশমিক        d   = (দশমিক) blk.d;
#pragma unroll
    জন্য (পূর্ণসংখ্যা j = 0; j < QK4_NL / 2; j += 2) {
        ধ্রুবক uint8_t q0 = blk.qs[j];
        ধ্রুবক uint8_t q1 = blk.qs[j + 1];
        a[j / 2]     = E::make(d * kvalues_iq4nl[q0 & 0xf], d * kvalues_iq4nl[q1 & 0xf]);
        a[j / 2 + 8] = E::make(d * kvalues_iq4nl[q0 >> 4], d * kvalues_iq4nl[q1 >> 4]);
    }
}

// iq3_s: k step kb is sub-block kb % 8 of superblock kb / 8. The superblock is 110 bytes, so read
// only the fields of that sub-block instead of copying the block. Same decode as
// dequantize_block_iq3_s: grid entries are taken as dwords and the sign bit is a plain shift.
template <typename E>
স্থির __dpct_inline__ শূন্য fg_stage_a(ধ্রুবক block_iq3_s * __restrict__ xrow, ধ্রুবক পূর্ণসংখ্যা kb, typename E::pair * a) {
    static_assert(QK_K == 256, "the iq3_s A stage assumes 8 sub-blocks per superblock");
    ধ্রুবক block_iq3_s * blk = xrow + kb / (QK_K / 32);
    ধ্রুবক পূর্ণসংখ্যা           ib8 = kb % (QK_K / 32);
    ধ্রুবক uint8_t *     qs  = blk->qs + 8 * ib8;
    ধ্রুবক পূর্ণসংখ্যা           qh  = blk->qh[ib8];
    ধ্রুবক দশমিক         d   = (দশমিক) blk->d * (1 + 2 * ((blk->scales[ib8 / 2] >> (4 * (ib8 % 2))) & 0xf));
#pragma unroll
    জন্য (পূর্ণসংখ্যা il = 0; il < 4; ++il) {
        ধ্রুবক uint32_t grid1 = iq3s_grid[qs[2 * il + 0] | ((qh << (8 - 2 * il)) & 256)];
        ধ্রুবক uint32_t grid2 = iq3s_grid[qs[2 * il + 1] | ((qh << (7 - 2 * il)) & 256)];
        ধ্রুবক পূর্ণসংখ্যা      signs = blk->signs[4 * ib8 + il];
#pragma unroll
        জন্য (পূর্ণসংখ্যা j = 0; j < 2; ++j) {
            ধ্রুবক দশমিক g1a = (দশমিক) ((grid1 >> (16 * j + 0)) & 0xff);
            ধ্রুবক দশমিক g1b = (দশমিক) ((grid1 >> (16 * j + 8)) & 0xff);
            ধ্রুবক দশমিক g2a = (দশমিক) ((grid2 >> (16 * j + 0)) & 0xff);
            ধ্রুবক দশমিক g2b = (দশমিক) ((grid2 >> (16 * j + 8)) & 0xff);
            ধ্রুবক পূর্ণসংখ্যা   s   = 2 * j;
            a[4 * il + j]     = E::make(d * ((signs & (1 << (s + 0))) ? -g1a : g1a),
                                        d * ((signs & (1 << (s + 1))) ? -g1b : g1b));
            a[4 * il + j + 2] = E::make(d * ((signs & (1 << (s + 4))) ? -g2a : g2a),
                                        d * ((signs & (1 << (s + 5))) ? -g2b : g2b));
        }
    }
}

// values per stored block, so a row of K values is K/qk blocks
template <typename block_q_t> কাঠামো fg_block_traits;
template <> কাঠামো fg_block_traits<block_iq4_nl> { স্থির constexpr পূর্ণসংখ্যা qk = QK4_NL; };
template <> কাঠামো fg_block_traits<block_iq3_s>  { স্থির constexpr পূর্ণসংখ্যা qk = QK_K; };
template <> কাঠামো fg_block_traits<block_iq3_xxs> { স্থির constexpr পূর্ণসংখ্যা qk = QK_K; };
template <> কাঠামো fg_block_traits<block_iq4_xs>  { স্থির constexpr পূর্ণসংখ্যা qk = QK_K; };
template <> কাঠামো fg_block_traits<block_iq2_xxs> { স্থির constexpr পূর্ণসংখ্যা qk = QK_K; };
template <> কাঠামো fg_block_traits<block_iq2_xs>  { স্থির constexpr পূর্ণসংখ্যা qk = QK_K; };
template <> কাঠামো fg_block_traits<block_iq2_s>   { স্থির constexpr পূর্ণসংখ্যা qk = QK_K; };
template <> কাঠামো fg_block_traits<block_iq1_s>   { স্থির constexpr পূর্ণসংখ্যা qk = QK_K; };
template <> কাঠামো fg_block_traits<block_iq1_m>   { স্থির constexpr পূর্ণসংখ্যা qk = QK_K; };
template <> কাঠামো fg_block_traits<block_q8_0>    { স্থির constexpr পূর্ণসংখ্যা qk = QK8_0; };
template <> কাঠামো fg_block_traits<block_q4_K>    { স্থির constexpr পূর্ণসংখ্যা qk = QK_K; };
template <> কাঠামো fg_block_traits<block_q5_K>    { স্থির constexpr পূর্ণসংখ্যা qk = QK_K; };
template <> কাঠামো fg_block_traits<block_q6_K>    { স্থির constexpr পূর্ণসংখ্যা qk = QK_K; };

// The A stages below are the dequantize_block_iq* kernels rewritten for one k step. There a
// work-item handled one quarter (il) of one 32-wide sub-block (ib); here one lane produces the
// whole step, so il becomes a loop and ib is kb inside the superblock. Each quarter yields 8
// consecutive values, i.e. 4 pairs at a[4*il], so nothing larger than 8 floats is ever live.
#define FG_SUPERBLOCK(T)                                                         \
    static_assert(QK_K == 256, "the " #T " A stage assumes 8 sub-blocks per superblock"); \
    ধ্রুবক T * blk = xrow + kb / (QK_K / 32);                                     \
    ধ্রুবক পূর্ণসংখ্যা ib  = kb % (QK_K / 32)

template <typename E>
স্থির __dpct_inline__ শূন্য fg_pack_quarter(ধ্রুবক দশমিক * __restrict__ t, typename E::pair * a, পূর্ণসংখ্যা il) {
#pragma unroll
    জন্য (পূর্ণসংখ্যা j = 0; j < 4; ++j) {
        a[4 * il + j] = E::make(t[2 * j], t[2 * j + 1]);
    }
}

template <typename E>
স্থির __dpct_inline__ শূন্য fg_stage_a(ধ্রুবক block_iq4_xs * __restrict__ xrow, ধ্রুবক পূর্ণসংখ্যা kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_iq4_xs);
    // low nibbles fill the first half of the step, high nibbles the second, so the two halves
    // land at a[0..7] and a[8..15] and no quarter loop is needed
    ধ্রুবক দশমিক d = (দশমিক) blk->d *
        ((((blk->scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) | (((blk->scales_h >> (2 * ib)) & 3) << 4)) - 32);
    ধ্রুবক uint8_t * q4 = blk->qs + 16 * ib;
#pragma unroll
    জন্য (পূর্ণসংখ্যা j = 0; j < 8; ++j) {
        a[j]     = E::make(d * kvalues_iq4nl[q4[2 * j] & 0xf], d * kvalues_iq4nl[q4[2 * j + 1] & 0xf]);
        a[8 + j] = E::make(d * kvalues_iq4nl[q4[2 * j] >> 4],  d * kvalues_iq4nl[q4[2 * j + 1] >> 4]);
    }
}

template <typename E>
স্থির __dpct_inline__ শূন্য fg_stage_a(ধ্রুবক block_iq3_xxs * __restrict__ xrow, ধ্রুবক পূর্ণসংখ্যা kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_iq3_xxs);
    ধ্রুবক uint8_t *  q3    = blk->qs + 8 * ib;
    ধ্রুবক uint16_t * gas   = (ধ্রুবক uint16_t *) (blk->qs + QK_K / 4) + 2 * ib;
    ধ্রুবক uint32_t   aux32 = gas[0] | (gas[1] << 16);
    ধ্রুবক দশমিক      d     = (দশমিক) blk->d * (0.5f + (aux32 >> 28)) * 0.5f;
#pragma unroll
    জন্য (পূর্ণসংখ্যা il = 0; il < 4; ++il) {
        ধ্রুবক uint8_t * grid1 = (ধ্রুবক uint8_t *) (iq3xxs_grid + q3[2 * il + 0]);
        ধ্রুবক uint8_t * grid2 = (ধ্রুবক uint8_t *) (iq3xxs_grid + q3[2 * il + 1]);
        ধ্রুবক uint8_t   signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];
        দশমিক t[8];
#pragma unroll
        জন্য (পূর্ণসংখ্যা j = 0; j < 4; ++j) {
            t[j + 0] = d * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f);
            t[j + 4] = d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

template <typename E>
স্থির __dpct_inline__ শূন্য fg_stage_a(ধ্রুবক block_iq2_xxs * __restrict__ xrow, ধ্রুবক পূর্ণসংখ্যা kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_iq2_xxs);
    ধ্রুবক uint16_t * q2    = blk->qs + 4 * ib;
    ধ্রুবক uint8_t *  aux8  = (ধ্রুবক uint8_t *) q2;
    ধ্রুবক uint32_t   aux32 = q2[2] | (q2[3] << 16);
    ধ্রুবক দশমিক      d     = (দশমিক) blk->d * (0.5f + (aux32 >> 28)) * 0.25f;
#pragma unroll
    জন্য (পূর্ণসংখ্যা il = 0; il < 4; ++il) {
        ধ্রুবক uint8_t * grid  = (ধ্রুবক uint8_t *) (iq2xxs_grid + aux8[il]);
        ধ্রুবক uint8_t   signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];
        দশমিক t[8];
#pragma unroll
        জন্য (পূর্ণসংখ্যা j = 0; j < 8; ++j) {
            t[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

template <typename E>
স্থির __dpct_inline__ শূন্য fg_stage_a(ধ্রুবক block_iq2_xs * __restrict__ xrow, ধ্রুবক পূর্ণসংখ্যা kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_iq2_xs);
    ধ্রুবক uint16_t * q2 = blk->qs + 4 * ib;
#pragma unroll
    জন্য (পূর্ণসংখ্যা il = 0; il < 4; ++il) {
        ধ্রুবক uint8_t * grid  = (ধ্রুবক uint8_t *) (iq2xs_grid + (q2[il] & 511));
        ধ্রুবক দশমিক     d     = (দশমিক) blk->d * (0.5f + ((blk->scales[ib] >> (4 * (il / 2))) & 0xf)) * 0.25f;
        ধ্রুবক uint8_t   signs = ksigns_iq2xs[q2[il] >> 9];
        দশমিক t[8];
#pragma unroll
        জন্য (পূর্ণসংখ্যা j = 0; j < 8; ++j) {
            t[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

template <typename E>
স্থির __dpct_inline__ শূন্য fg_stage_a(ধ্রুবক block_iq2_s * __restrict__ xrow, ধ্রুবক পূর্ণসংখ্যা kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_iq2_s);
#pragma unroll
    জন্য (পূর্ণসংখ্যা il = 0; il < 4; ++il) {
        ধ্রুবক uint8_t * grid =
            (ধ্রুবক uint8_t *) (iq2s_grid + (blk->qs[4 * ib + il] | ((blk->qh[ib] << (8 - 2 * il)) & 0x300)));
        ধ্রুবক দশমিক   d     = (দশমিক) blk->d * (0.5f + ((blk->scales[ib] >> (4 * (il / 2))) & 0xf)) * 0.25f;
        ধ্রুবক uint8_t signs = blk->qs[QK_K / 8 + 4 * ib + il];
        দশমিক t[8];
#pragma unroll
        জন্য (পূর্ণসংখ্যা j = 0; j < 8; ++j) {
            t[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

template <typename E>
স্থির __dpct_inline__ শূন্য fg_stage_a(ধ্রুবক block_iq1_s * __restrict__ xrow, ধ্রুবক পূর্ণসংখ্যা kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_iq1_s);
    ধ্রুবক দশমিক delta = blk->qh[ib] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
    ধ্রুবক দশমিক d     = (দশমিক) blk->d * (2 * ((blk->qh[ib] >> 12) & 7) + 1);
#pragma unroll
    জন্য (পূর্ণসংখ্যা il = 0; il < 4; ++il) {
        uint32_t       grid32[2];
        ধ্রুবক int8_t * q = (ধ্রুবক int8_t *) grid32;
        grid32[0] = iq1s_grid_gpu[blk->qs[4 * ib + il] | (((blk->qh[ib] >> (3 * il)) & 7) << 8)];
        grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
        grid32[0] &= 0x0f0f0f0f;
        দশমিক t[8];
#pragma unroll
        জন্য (পূর্ণসংখ্যা j = 0; j < 8; ++j) {
            t[j] = d * (q[j] + delta);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

template <typename E>
স্থির __dpct_inline__ শূন্য fg_stage_a(ধ্রুবক block_iq1_m * __restrict__ xrow, ধ্রুবক পূর্ণসংখ্যা kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_iq1_m);
    ধ্রুবক uint16_t * sc = (ধ্রুবক uint16_t *) blk->scales;
    iq1m_scale_t     scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
#pragma unroll
    জন্য (পূর্ণসংখ্যা il = 0; il < 4; ++il) {
        ধ্রুবক পূর্ণসংখ্যা   ib16  = 2 * ib + il / 2;
        ধ্রুবক দশমিক d     = (দশমিক) scale.f16 * (2 * ((sc[ib16 / 4] >> (3 * (ib16 % 4))) & 0x7) + 1);
        ধ্রুবক দশমিক delta = blk->qh[2 * ib + il / 2] & (0x08 << (4 * (il % 2))) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
        uint32_t       grid32[2];
        ধ্রুবক int8_t * q = (ধ্রুবক int8_t *) grid32;
        grid32[0] = iq1s_grid_gpu[blk->qs[4 * ib + il] |
                                  (((blk->qh[2 * ib + il / 2] >> (4 * (il % 2))) & 7) << 8)];
        grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
        grid32[0] &= 0x0f0f0f0f;
        দশমিক t[8];
#pragma unroll
        জন্য (পূর্ণসংখ্যা j = 0; j < 8; ++j) {
            t[j] = d * (q[j] + delta);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}


// q8_0 and the k-quants. Each decode takes the fields of one 32-value sub-block, so the canonical
// layout and the reorder (SoA) layout of reorder_qw() share it and differ only in where the fields
// live, as in dequantize.hpp.
template <typename E>
স্থির __dpct_inline__ শূন্য fg_decode_q8_0(ধ্রুবক int8_t * __restrict__ qs, ধ্রুবক দশমিক d, typename E::pair * a) {
#pragma unroll
    জন্য (পূর্ণসংখ্যা j = 0; j < QK8_0 / 2; ++j) {
        a[j] = E::make(d * qs[2 * j], d * qs[2 * j + 1]);
    }
}

// same unpack as get_scale_min_k4() in dequantize.hpp
স্থির __dpct_inline__ শূন্য fg_scale_min_k4(ধ্রুবক পূর্ণসংখ্যা j, ধ্রুবক uint8_t * __restrict__ q, uint8_t & d, uint8_t & m) {
    যদি (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } অন্যথা {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

// q4_K sub-block ib (0..7) uses scale/min pair ib and the low (even ib) or high (odd ib) nibbles of
// qs[32 * (ib / 2) ...], as in dequantize_row_q4_K
template <typename E>
স্থির __dpct_inline__ শূন্য fg_decode_q4_K(ধ্রুবক uint8_t * __restrict__ qs, ধ্রুবক uint8_t * __restrict__ scales,
                                           ধ্রুবক sycl::half2 dm, ধ্রুবক পূর্ণসংখ্যা ib, typename E::pair * a) {
    uint8_t sc, mb;
    fg_scale_min_k4(ib, scales, sc, mb);
    ধ্রুবক দশমিক     d     = (দশমিক) dm[0] * sc;
    ধ্রুবক দশমিক     m     = (দশমিক) dm[1] * mb;
    ধ্রুবক uint8_t * q     = qs + 32 * (ib / 2);
    ধ্রুবক পূর্ণসংখ্যা       shift = 4 * (ib % 2);
#pragma unroll
    জন্য (পূর্ণসংখ্যা j = 0; j < 16; ++j) {
        a[j] = E::make(d * ((q[2 * j] >> shift) & 0xF) - m, d * ((q[2 * j + 1] >> shift) & 0xF) - m);
    }
}

// q5_K: q4_K plus one high bit per value, bit ib of qh[l]
template <typename E>
স্থির __dpct_inline__ শূন্য fg_decode_q5_K(ধ্রুবক uint8_t * __restrict__ qs, ধ্রুবক uint8_t * __restrict__ qh,
                                           ধ্রুবক uint8_t * __restrict__ scales, ধ্রুবক sycl::half2 dm, ধ্রুবক পূর্ণসংখ্যা ib,
                                           typename E::pair * a) {
    uint8_t sc, mb;
    fg_scale_min_k4(ib, scales, sc, mb);
    ধ্রুবক দশমিক     d     = (দশমিক) dm[0] * sc;
    ধ্রুবক দশমিক     m     = (দশমিক) dm[1] * mb;
    ধ্রুবক uint8_t * q     = qs + 32 * (ib / 2);
    ধ্রুবক পূর্ণসংখ্যা       shift = 4 * (ib % 2);
#pragma unroll
    জন্য (পূর্ণসংখ্যা j = 0; j < 16; ++j) {
        ধ্রুবক পূর্ণসংখ্যা l = 2 * j;
        a[j] = E::make(d * (((q[l] >> shift) & 0xF) | (((qh[l] >> ib) & 1) << 4)) - m,
                       d * (((q[l + 1] >> shift) & 0xF) | (((qh[l + 1] >> ib) & 1) << 4)) - m);
    }
}

// q6_K: sub-block ib is quarter r = ib % 4 of half h = ib / 4. The half selects ql + 64h, qh + 32h
// and scales + 8h; the quarter selects ql + 32(r & 1), the ql nibble r / 2, the qh bit pair r and
// scales + 2r, as in dequantize_row_q6_K. The scale changes at value 16 of the sub-block.
template <typename E>
স্থির __dpct_inline__ শূন্য fg_decode_q6_K(ধ্রুবক uint8_t * __restrict__ ql, ধ্রুবক uint8_t * __restrict__ qh,
                                           ধ্রুবক int8_t * __restrict__ scales, ধ্রুবক দশমিক d, ধ্রুবক পূর্ণসংখ্যা ib,
                                           typename E::pair * a) {
    ধ্রুবক পূর্ণসংখ্যা       h  = ib / 4;
    ধ্রুবক পূর্ণসংখ্যা       r  = ib % 4;
    ধ্রুবক uint8_t * q  = ql + 64 * h + 32 * (r & 1);
    ধ্রুবক uint8_t * hb = qh + 32 * h;
    ধ্রুবক int8_t *  sc = scales + 8 * h + 2 * r;
#pragma unroll
    জন্য (পূর্ণসংখ্যা j = 0; j < 16; ++j) {
        ধ্রুবক পূর্ণসংখ্যা   l  = 2 * j;
        ধ্রুবক দশমিক dl = d * sc[j / 8];
        ধ্রুবক পূর্ণসংখ্যা   q0 = (((q[l] >> (4 * (r / 2))) & 0xF) | (((hb[l] >> (2 * r)) & 3) << 4)) - 32;
        ধ্রুবক পূর্ণসংখ্যা   q1 = (((q[l + 1] >> (4 * (r / 2))) & 0xF) | (((hb[l + 1] >> (2 * r)) & 3) << 4)) - 32;
        a[j] = E::make(dl * q0, dl * q1);
    }
}

template <typename E>
স্থির __dpct_inline__ শূন্য fg_stage_a(ধ্রুবক block_q8_0 * __restrict__ xrow, ধ্রুবক পূর্ণসংখ্যা kb, typename E::pair * a) {
    fg_decode_q8_0<E>(xrow[kb].qs, (দশমিক) xrow[kb].d, a);
}

template <typename E>
স্থির __dpct_inline__ শূন্য fg_stage_a(ধ্রুবক block_q4_K * __restrict__ xrow, ধ্রুবক পূর্ণসংখ্যা kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_q4_K);
    fg_decode_q4_K<E>(blk->qs, blk->scales, blk->dm, ib, a);
}

template <typename E>
স্থির __dpct_inline__ শূন্য fg_stage_a(ধ্রুবক block_q5_K * __restrict__ xrow, ধ্রুবক পূর্ণসংখ্যা kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_q5_K);
    fg_decode_q5_K<E>(blk->qs, blk->qh, blk->scales, blk->dm, ib, a);
}

template <typename E>
স্থির __dpct_inline__ শূন্য fg_stage_a(ধ্রুবক block_q6_K * __restrict__ xrow, ধ্রুবক পূর্ণসংখ্যা kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_q6_K);
    fg_decode_q6_K<E>(blk->ql, blk->qh, blk->scales, (দশমিক) blk->d, ib, a);
}

// Reorder (SoA) layout: each block field is one stream over the nblocks of the matrix (of the expert
// slice for MUL_MAT_ID), in the order reorder_qw() writes them. Block ib holds k step kb.
template <typename block_q_t> কাঠামো fg_soa {
    স্থির constexpr যৌক্তিক supported = মিথ্যা;
};

template <> কাঠামো fg_soa<block_q8_0> {
    স্থির constexpr যৌক্তিক supported = সত্য;
    // [qs][d]
    template <typename E>
    স্থির __dpct_inline__ শূন্য stage(ধ্রুবক uint8_t * x, ধ্রুবক size_t nblocks, ধ্রুবক size_t ib, ধ্রুবক পূর্ণসংখ্যা,
                                      typename E::pair * a) {
        ধ্রুবক দশমিক d = (দশমিক) ((ধ্রুবক sycl::half *) (x + nblocks * QK8_0))[ib];
        fg_decode_q8_0<E>((ধ্রুবক int8_t *) x + ib * QK8_0, d, a);
    }
};

template <> কাঠামো fg_soa<block_q4_K> {
    স্থির constexpr যৌক্তিক supported = সত্য;
    // [qs][scales][dm]
    template <typename E>
    স্থির __dpct_inline__ শূন্য stage(ধ্রুবক uint8_t * x, ধ্রুবক size_t nblocks, ধ্রুবক size_t ib, ধ্রুবক পূর্ণসংখ্যা kb,
                                      typename E::pair * a) {
        ধ্রুবক uint8_t *   scales = x + nblocks * (QK_K / 2);
        ধ্রুবক sycl::half2 dm     = ((ধ্রুবক sycl::half2 *) (scales + nblocks * K_SCALE_আকার))[ib];
        fg_decode_q4_K<E>(x + ib * (QK_K / 2), scales + ib * K_SCALE_আকার, dm, kb % (QK_K / 32), a);
    }
};

template <> কাঠামো fg_soa<block_q5_K> {
    স্থির constexpr যৌক্তিক supported = সত্য;
    // [qs][qh][scales][dm]
    template <typename E>
    স্থির __dpct_inline__ শূন্য stage(ধ্রুবক uint8_t * x, ধ্রুবক size_t nblocks, ধ্রুবক size_t ib, ধ্রুবক পূর্ণসংখ্যা kb,
                                      typename E::pair * a) {
        ধ্রুবক uint8_t *   qh     = x + nblocks * (QK_K / 2);
        ধ্রুবক uint8_t *   scales = qh + nblocks * (QK_K / 8);
        ধ্রুবক sycl::half2 dm     = ((ধ্রুবক sycl::half2 *) (scales + nblocks * K_SCALE_আকার))[ib];
        fg_decode_q5_K<E>(x + ib * (QK_K / 2), qh + ib * (QK_K / 8), scales + ib * K_SCALE_আকার, dm,
                          kb % (QK_K / 32), a);
    }
};

template <> কাঠামো fg_soa<block_q6_K> {
    স্থির constexpr যৌক্তিক supported = সত্য;
    // [ql][qh][scales][d]
    template <typename E>
    স্থির __dpct_inline__ শূন্য stage(ধ্রুবক uint8_t * x, ধ্রুবক size_t nblocks, ধ্রুবক size_t ib, ধ্রুবক পূর্ণসংখ্যা kb,
                                      typename E::pair * a) {
        ধ্রুবক uint8_t * qh     = x + nblocks * (QK_K / 2);
        ধ্রুবক uint8_t * scales = qh + nblocks * (QK_K / 4);
        ধ্রুবক দশমিক     d      = (দশমিক) ((ধ্রুবক sycl::half *) (scales + nblocks * (QK_K / 16)))[ib];
        fg_decode_q6_K<E>(x + ib * (QK_K / 2), qh + ib * (QK_K / 4), (ধ্রুবক int8_t *) scales + ib * (QK_K / 16), d,
                          kb % (QK_K / 32), a);
    }
};


// one SG_ROWS x BN output tile: B columns [b0, b0 + BN) of packed_b go to dst columns [n0, n1),
// n1 - n0 <= BN
template <typename S, typename block_q_t, যৌক্তিক reordered>
স্থির শূন্য fused_dequant_gemm_tile(
    ধ্রুবক block_q_t * __restrict__ x,
    ধ্রুবক typename S::tsb * __restrict__ packed_b,
    দশমিক * __restrict__ dst,
    ধ্রুবক পূর্ণসংখ্যা M, ধ্রুবক পূর্ণসংখ্যা Npad, ধ্রুবক পূর্ণসংখ্যা K, ধ্রুবক পূর্ণসংখ্যা ldd,
    ধ্রুবক পূর্ণসংখ্যা b0, ধ্রুবক পূর্ণসংখ্যা n0, ধ্রুবক পূর্ণসংখ্যা n1,
    sycl::local_accessor<typename S::tsa, 1> tile_a,
    sycl::local_accessor<দশমিক, 1> tile_c,
    ধ্রুবক sycl::nd_item<2> & item) {
    যদি constexpr (fg_built<S::SG>()) {
        using EA = typename S::EA;
        using TA = typename S::ta;
        using TB = typename S::tb;
        ধ্রুবক auto sg     = item.get_sub_group();
        ধ্রুবক পূর্ণসংখ্যা  sg_id  = sg.get_group_id()[0];
        ধ্রুবক পূর্ণসংখ্যা  lane   = sg.get_local_id()[0];
        ধ্রুবক পূর্ণসংখ্যা  m0     = item.get_group(1) * S::SG_ROWS;
        ধ্রুবক পূর্ণসংখ্যা  nstep  = K / FG_BK;
        ধ্রুবক পূর্ণসংখ্যা  a_base = sg_id * S::SG_ROWS * FG_BK;
        ধ্রুবক পূর্ণসংখ্যা  c_base = sg_id * S::SG_ROWS * S::BN;

        mx::joint_matrix<sycl::sub_group, দশমিক, mx::use::accumulator, S::TM, S::TN> acc[S::MT][S::NT];
#pragma unroll
        জন্য (পূর্ণসংখ্যা mt = 0; mt < S::MT; ++mt) {
#pragma unroll
            জন্য (পূর্ণসংখ্যা nt = 0; nt < S::NT; ++nt) {
                mx::joint_matrix_fill(sg, acc[mt][nt], 0.0f);
            }
        }

        // lane decodes rows lane, lane + SG, ... of the sub-group's SG_ROWS
        constexpr পূর্ণসংখ্যা         KPB     = fg_block_traits<block_q_t>::qk / FG_BK; // k steps per block
        ধ্রুবক size_t          bpr     = K / fg_block_traits<block_q_t>::qk;
        ধ্রুবক size_t          nblocks = (size_t) M * bpr;
        size_t                row_blk[S::RPL];
        যৌক্তিক                  row_ok[S::RPL];
        typename EA::pair *   a[S::RPL];
#pragma unroll
        জন্য (পূর্ণসংখ্যা r = 0; r < S::RPL; ++r) {
            ধ্রুবক পূর্ণসংখ্যা row = m0 + r * S::SG + lane;
            row_ok[r]  = row < M;
            row_blk[r] = (size_t) (row_ok[r] ? row : 0) * bpr;
            a[r]       = (typename EA::pair *) &tile_a[a_base + (r * S::SG + lane) * FG_BK];
        }

        ধ্রুবক auto b_ptr = sycl::address_space_cast<sycl::access::address_space::global_space,
                                                    sycl::access::decorated::no>(packed_b);
        ধ্রুবক পূর্ণসংখ্যা b_stride = Npad * S::VNNI;

        ধ্রুবক পূর্ণসংখ্যা kb_begin = (sg_id * nstep) / FG_KSPLIT;
        ধ্রুবক পূর্ণসংখ্যা kb_end   = ((sg_id + 1) * nstep) / FG_KSPLIT;
        জন্য (পূর্ণসংখ্যা kb = kb_begin; kb < kb_end; ++kb) {
#pragma unroll
            জন্য (পূর্ণসংখ্যা r = 0; r < S::RPL; ++r) {
                যদি (row_ok[r]) {
                    যদি constexpr (reordered) {
                        fg_soa<block_q_t>::template stage<EA>((ধ্রুবক uint8_t *) x, nblocks, row_blk[r] + kb / KPB, kb,
                                                              a[r]);
                    } অন্যথা {
                        fg_stage_a<EA>(x + row_blk[r], kb, a[r]);
                    }
                } অন্যথা {
#pragma unroll
                    জন্য (পূর্ণসংখ্যা j = 0; j < FG_BK / 2; ++j) {
                        a[r][j] = EA::make(0.0f, 0.0f);
                    }
                }
            }
            sycl::group_barrier(sg);

#pragma unroll
            জন্য (পূর্ণসংখ্যা kt = 0; kt < FG_BK / S::TK; ++kt) {
                ধ্রুবক পূর্ণসংখ্যা kq0 = (kb * FG_BK + kt * S::TK) / S::VNNI;
                mx::joint_matrix<sycl::sub_group, TB, mx::use::b, S::TK, S::TN, S::b_layout> sub_b[S::NT];
#pragma unroll
                জন্য (পূর্ণসংখ্যা nt = 0; nt < S::NT; ++nt) {
                    mx::joint_matrix_load(sg, sub_b[nt], b_ptr + (size_t) kq0 * b_stride + (b0 + nt * S::TN) * S::VNNI, b_stride);
                }
#pragma unroll
                জন্য (পূর্ণসংখ্যা mt = 0; mt < S::MT; ++mt) {
                    mx::joint_matrix<sycl::sub_group, TA, mx::use::a, S::TM, S::TK, mx::layout::row_major> sub_a;
                    mx::joint_matrix_load(sg, sub_a,
                        tile_a.template get_multi_ptr<sycl::access::decorated::no>() + a_base + (mt * S::TM) * FG_BK + kt * S::TK,
                        FG_BK);
#pragma unroll
                    জন্য (পূর্ণসংখ্যা nt = 0; nt < S::NT; ++nt) {
                        mx::joint_matrix_mad(sg, acc[mt][nt], sub_a, sub_b[nt], acc[mt][nt]);
                    }
                }
            }
            // the next step overwrites tile_a
            sycl::group_barrier(sg);
        }

#pragma unroll
        জন্য (পূর্ণসংখ্যা mt = 0; mt < S::MT; ++mt) {
#pragma unroll
            জন্য (পূর্ণসংখ্যা nt = 0; nt < S::NT; ++nt) {
                mx::joint_matrix_store(sg, acc[mt][nt],
                    tile_c.template get_multi_ptr<sycl::access::decorated::no>() + c_base + (mt * S::TM) * S::BN + nt * S::TN,
                    S::BN, mx::layout::row_major);
            }
        }
        sycl::group_barrier(item.get_group());

        // sum the K splits; consecutive lanes write consecutive rows of one dst column
        জন্য (পূর্ণসংখ্যা idx = item.get_local_linear_id(); idx < S::SG_ROWS * S::BN; idx += S::WG_আকার) {
            ধ্রুবক পূর্ণসংখ্যা r = idx % S::SG_ROWS;
            ধ্রুবক পূর্ণসংখ্যা c = idx / S::SG_ROWS;
            ধ্রুবক পূর্ণসংখ্যা m = m0 + r;
            ধ্রুবক পূর্ণসংখ্যা n = n0 + c;
            যদি (m < M && n < n1) {
                দশমিক যোগ_2 = 0.0f;
#pragma unroll
                জন্য (পূর্ণসংখ্যা s = 0; s < FG_KSPLIT; ++s) {
                    যোগ_2 += tile_c[s * S::SG_ROWS * S::BN + r * S::BN + c];
                }
                dst[(size_t) n * ldd + m] = যোগ_2;
            }
        }
    }
}

template <typename S, typename block_q_t, যৌক্তিক reordered = মিথ্যা>
স্থির শূন্য fused_dequant_gemm_launch(ধ্রুবক শূন্য * src0, ধ্রুবক typename S::tsb * packed, দশমিক * dst, ধ্রুবক পূর্ণসংখ্যা M,
                                      ধ্রুবক পূর্ণসংখ্যা N, ধ্রুবক পূর্ণসংখ্যা Npad, ধ্রুবক পূর্ণসংখ্যা K, ধ্রুবক পূর্ণসংখ্যা ldd,
                                      ধ্রুবক int64_t groups_n, ধ্রুবক int64_t groups_m, dpct::queue_ptr stream) {
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<typename S::tsa, 1> tile_a(FG_KSPLIT * S::SG_ROWS * FG_BK, cgh);
        sycl::local_accessor<দশমিক, 1>          tile_c(FG_KSPLIT * S::SG_ROWS * S::BN, cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(groups_n, groups_m * S::WG_আকার), sycl::range<2>(1, S::WG_আকার)),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_আকার(S::SG)]] {
                ধ্রুবক পূর্ণসংখ্যা n0 = item.get_group(0) * S::BN;
                fused_dequant_gemm_tile<S, block_q_t, reordered>((ধ্রুবক block_q_t *) src0, packed, dst, M, Npad, K,
                                                      ldd, n0, n0, N, tile_a, tile_c, item);
            });
    });
}

// grouped: work-group (t, mt) is tile t of the schedule; its B columns sit at t * BN
template <typename S, typename block_q_t, যৌক্তিক reordered>
স্থির শূন্য grouped_dequant_gemm_launch(ধ্রুবক অক্ষর * src0_dd, ধ্রুবক size_t expert_stride,
                                        ধ্রুবক ggml_sycl_gg_tile * tiles_ptr, ধ্রুবক typename S::tsb * packed, দশমিক * dst,
                                        ধ্রুবক পূর্ণসংখ্যা M, ধ্রুবক পূর্ণসংখ্যা Npad, ধ্রুবক পূর্ণসংখ্যা K, ধ্রুবক int64_t n_tiles,
                                        ধ্রুবক int64_t groups_m, dpct::queue_ptr stream) {
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<typename S::tsa, 1> tile_a(FG_KSPLIT * S::SG_ROWS * FG_BK, cgh);
        sycl::local_accessor<দশমিক, 1>          tile_c(FG_KSPLIT * S::SG_ROWS * S::BN, cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(n_tiles, groups_m * S::WG_আকার), sycl::range<2>(1, S::WG_আকার)),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_আকার(S::SG)]] {
                ধ্রুবক পূর্ণসংখ্যা               t    = item.get_group(0);
                ধ্রুবক ggml_sycl_gg_tile tile = tiles_ptr[t];
                ধ্রুবক block_q_t *       x    = (ধ্রুবক block_q_t *) (src0_dd + (size_t) tile.expert * expert_stride);
                fused_dequant_gemm_tile<S, block_q_t, reordered>(x, packed, dst, M, Npad, K, M, t * S::BN, tile.n0,
                                                                 tile.n1, tile_a, tile_c, item);
            });
    });
}

// src1 f32 rows -> packed [K/V][n_tiles*BN][V], tile t holds its rows [n0, n1) at columns t*BN..,
// zero past n1. The column runs fastest so a sub-group writes one contiguous run.
template <typename S>
স্থির শূন্য grouped_gemm_pack_b(ধ্রুবক দশমিক * y, typename S::tsb * packed, ধ্রুবক ggml_sycl_gg_tile * tiles, পূর্ণসংখ্যা Npad,
                                পূর্ণসংখ্যা K, dpct::queue_ptr stream) {
    using E        = typename S::EB;
    constexpr পূর্ণসংখ্যা V = S::VNNI;
    ধ্রুবক পূর্ণসংখ্যা kqs   = K / V;
    stream->parallel_for(sycl::range<1>((size_t) Npad * kqs), [=](sycl::id<1> id) {
        ধ্রুবক size_t idx = id[0];
        ধ্রুবক পূর্ণসংখ্যা    kq  = idx / Npad;
        ধ্রুবক পূর্ণসংখ্যা    n   = idx - (size_t) kq * Npad;
        ধ্রুবক ggml_sycl_gg_tile tile = tiles[n / S::BN];
        ধ্রুবক পূর্ণসংখ্যা    row = tile.n0 + n % S::BN;
        // one guarded load run per work-item, as a per-element select costs ~1.5% prefill
        typename S::tsb vals[V] = {};
        যদি (row < tile.n1) {
            ধ্রুবক দশমিক * src = y + (size_t) row * K + V * kq;
#pragma unroll
            জন্য (পূর্ণসংখ্যা v = 0; v < V; ++v) {
                vals[v] = E::cvt(src[v]);
            }
        }
        typename S::tsb * out = packed + ((size_t) kq * Npad + n) * V;
#pragma unroll
        জন্য (পূর্ণসংখ্যা v = 0; v < V; ++v) {
            out[v] = vals[v];
        }
    });
}

// q8_0 and the k-quants take only the grouped path. The plain kernel decodes A again for every BN columns
// of a dense batch, and for these formats that costs more than the one dequantization of the library GEMM.
template <typename T> স্থির constexpr যৌক্তিক fg_plain_ok() {
    ফেরত !std::is_same_v<T, block_q8_0> && !std::is_same_v<T, block_q4_K> && !std::is_same_v<T, block_q5_K> &&
           !std::is_same_v<T, block_q6_K>;
}

template <typename T, যৌক্তিক R> কাঠামো fg_tag {
    using type                     = T;
    স্থির constexpr যৌক্তিক reordered = R;
};

// calls f(fg_tag<block_q_t, reordered>{}) for the weight format and layout; false if it has no A stage
template <typename T, typename F> স্থির যৌক্তিক fg_visit_layout(যৌক্তিক reordered, F && f) {
    যদি (!reordered) {
        f(fg_tag<T, মিথ্যা>{});
        ফেরত সত্য;
    }
    যদি constexpr (fg_soa<T>::supported) {
        f(fg_tag<T, সত্য>{});
        ফেরত সত্য;
    }
    ফেরত মিথ্যা;
}

template <typename F> স্থির যৌক্তিক fg_visit_type(ggml_type type, যৌক্তিক reordered, F && f) {
    নির্বাচন (type) {
        ক্ষেত্র GGML_TYPE_IQ4_NL:  ফেরত fg_visit_layout<block_iq4_nl>(reordered, f);
        ক্ষেত্র GGML_TYPE_IQ3_S:   ফেরত fg_visit_layout<block_iq3_s>(reordered, f);
        ক্ষেত্র GGML_TYPE_IQ4_XS:  ফেরত fg_visit_layout<block_iq4_xs>(reordered, f);
        ক্ষেত্র GGML_TYPE_IQ3_XXS: ফেরত fg_visit_layout<block_iq3_xxs>(reordered, f);
        ক্ষেত্র GGML_TYPE_IQ2_XXS: ফেরত fg_visit_layout<block_iq2_xxs>(reordered, f);
        ক্ষেত্র GGML_TYPE_IQ2_XS:  ফেরত fg_visit_layout<block_iq2_xs>(reordered, f);
        ক্ষেত্র GGML_TYPE_IQ2_S:   ফেরত fg_visit_layout<block_iq2_s>(reordered, f);
        ক্ষেত্র GGML_TYPE_IQ1_S:   ফেরত fg_visit_layout<block_iq1_s>(reordered, f);
        ক্ষেত্র GGML_TYPE_IQ1_M:   ফেরত fg_visit_layout<block_iq1_m>(reordered, f);
        ক্ষেত্র GGML_TYPE_Q8_0:    ফেরত fg_visit_layout<block_q8_0>(reordered, f);
        ক্ষেত্র GGML_TYPE_Q4_K:    ফেরত fg_visit_layout<block_q4_K>(reordered, f);
        ক্ষেত্র GGML_TYPE_Q5_K:    ফেরত fg_visit_layout<block_q5_K>(reordered, f);
        ক্ষেত্র GGML_TYPE_Q6_K:    ফেরত fg_visit_layout<block_q6_K>(reordered, f);
        পূর্বনির্ধারিত:                ফেরত মিথ্যা;
    }
}

template <typename S>
স্থির যৌক্তিক fg_fused_run(ggml_type src0_type, যৌক্তিক src0_reordered, ধ্রুবক শূন্য * src0, ধ্রুবক শূন্য * src1, ggml_type src1_type,
                         দশমিক * dst, int64_t M, int64_t N, int64_t K, int64_t ldd, ggml_sycl_pool & pool,
                         dpct::queue_ptr stream) {
    ধ্রুবক int64_t groups_n = (N + S::BN - 1) / S::BN;
    ধ্রুবক int64_t groups_m = (M + S::SG_ROWS - 1) / S::SG_ROWS;
    ধ্রুবক পূর্ণসংখ্যা     Npad     = (পূর্ণসংখ্যা) (groups_n * S::BN);

    // src1 is read in its own type: one pass, converted in registers only if B differs
    ggml_sycl_pool_alloc<typename S::tsb> packed_b(pool, (size_t) K * Npad);
    যদি (src1_type == GGML_TYPE_F16) {
        fused_gemm_pack_b<typename S::EB>((ধ্রুবক sycl::half *) src1, packed_b.get(), (পূর্ণসংখ্যা) N, Npad, (পূর্ণসংখ্যা) K, stream);
    } অন্যথা যদি (src1_type == GGML_TYPE_BF16) {
        fused_gemm_pack_b<typename S::EB>((ধ্রুবক fg_bf16 *) src1, packed_b.get(), (পূর্ণসংখ্যা) N, Npad, (পূর্ণসংখ্যা) K, stream);
    } অন্যথা {
        fused_gemm_pack_b<typename S::EB>((ধ্রুবক দশমিক *) src1, packed_b.get(), (পূর্ণসংখ্যা) N, Npad, (পূর্ণসংখ্যা) K, stream);
    }

    ধ্রুবক typename S::tsb * packed = packed_b.get();
    ফেরত fg_visit_type(src0_type, src0_reordered, [&](auto tag) {
        using block_q_t = typename decltype(tag)::type;
        যদি constexpr (fg_plain_ok<block_q_t>()) {
            fused_dequant_gemm_launch<S, block_q_t, decltype(tag)::reordered>(src0, packed, dst, (পূর্ণসংখ্যা) M, (পূর্ণসংখ্যা) N,
                                                                              Npad, (পূর্ণসংখ্যা) K, (পূর্ণসংখ্যা) ldd, groups_n,
                                                                              groups_m, stream);
        }
    });
}

যৌক্তিক ggml_sycl_fused_dequant_gemm(ggml_type src0_type, যৌক্তিক src0_reordered, ধ্রুবক শূন্য * src0, ধ্রুবক শূন্য * src1, ggml_type src1_type,
                                  int32_t src1_prec, দশমিক * dst, int64_t M, int64_t N, int64_t K, int64_t ldd,
                                  ggml_sycl_pool & pool, dpct::queue_ptr stream) {
    // every BN columns dequantize A again, so wide N is left to the library GEMM.
    // src0_reordered: the persistent weight reorder may have been installed on this
    // tensor by an earlier small-batch mul_mat; the A stage then has to decode the
    // SoA layout. Types without an SoA stage (fg_soa<T>::supported == false)
    // decline via fg_visit_layout and the caller falls through to the
    // reorder-aware conversion route.
    যৌক্তিক plain_ok = মিথ্যা;
    fg_visit_type(src0_type, src0_reordered, [&](auto tag) { plain_ok = fg_plain_ok<typename decltype(tag)::type>(); });
    যদি (g_ggml_sycl_dynamic_precision == GGML_SYCL_DYNAMIC_PRECISION_F32 ||
        !ggml_sycl_xmx_gather_type_enabled(src0_type) || !plain_ok) {
        ফেরত মিথ্যা;
    }
    যদি (src1_type != GGML_TYPE_F32 && src1_type != GGML_TYPE_F16 && src1_type != GGML_TYPE_BF16) {
        ফেরত মিথ্যা;
    }
    যদি (!ggml_sycl_fused_dequant_gemm_shape_ok(src0_type, M, N, K, ldd)) {
        ফেরত মিথ্যা;
    }
    ধ্রুবক পূর্ণসংখ্যা combo = fg_pick_combo(stream, src1_type, src1_prec);
    যদি (combo < 0) {
        ফেরত মিথ্যা;
    }
    যৌক্তিক launched = মিথ্যা;
    fg_visit_combo(combo, [&](auto s) {
        launched = fg_fused_run<decltype(s)>(src0_type, src0_reordered, src0, src1, src1_type, dst, M, N, K, ldd, pool, stream);
    });
    ফেরত launched;
}

template <typename S>
স্থির যৌক্তিক fg_grouped_run(ggml_type src0_type, যৌক্তিক reordered, ধ্রুবক শূন্য * src0_base, size_t expert_stride,
                           ধ্রুবক দশমিক * src1, দশমিক * dst, ধ্রুবক int64_t * expert_row_offsets, int64_t n_as, int64_t M,
                           int64_t K, std::vector<ggml_sycl_gg_tile> & tiles, ggml_sycl_pool & pool,
                           dpct::queue_ptr stream) {
    // the host knows every slice, so it lays out the work-groups: no search on the device
    tiles.clear();
    জন্য (int64_t e = 0; e < n_as; ++e) {
        ধ্রুবক int64_t end = expert_row_offsets[e + 1];
        জন্য (int64_t n0 = expert_row_offsets[e]; n0 < end; n0 += S::BN) {
            tiles.push_back({ (int32_t) e, (int32_t) n0, (int32_t) std::min<int64_t>(n0 + S::BN, end) });
        }
    }
    ধ্রুবক int64_t n_tiles  = tiles.size();
    ধ্রুবক int64_t groups_m = (M + S::SG_ROWS - 1) / S::SG_ROWS;
    ধ্রুবক পূর্ণসংখ্যা     Npad     = (পূর্ণসংখ্যা) (n_tiles * S::BN);

    ধ্রুবক int64_t max_tiles = grouped_gemm_max_tiles(expert_row_offsets[n_as], n_as, S::BN);
    GGML_ASSERT(n_tiles <= max_tiles);
    ggml_sycl_pool_alloc<ggml_sycl_gg_tile> tiles_dev(pool, max_tiles);
    SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(tiles_dev.get(), tiles.তথ্য(), n_tiles * আকার(ggml_sycl_gg_tile))));

    ggml_sycl_pool_alloc<typename S::tsb> packed_b(pool, (size_t) K * max_tiles * S::BN);
    grouped_gemm_pack_b<S>(src1, packed_b.get(), tiles_dev.get(), Npad, (পূর্ণসংখ্যা) K, stream);

    ধ্রুবক typename S::tsb *   packed    = packed_b.get();
    ধ্রুবক ggml_sycl_gg_tile * tiles_ptr = tiles_dev.get();
    ধ্রুবক অক্ষর *              src0_dd   = (ধ্রুবক অক্ষর *) src0_base;
    ফেরত fg_visit_type(src0_type, reordered, [&](auto tag) {
        using T = decltype(tag);
        grouped_dequant_gemm_launch<S, typename T::type, T::reordered>(src0_dd, expert_stride, tiles_ptr, packed, dst,
                                                                       (পূর্ণসংখ্যা) M, Npad, (পূর্ণসংখ্যা) K, n_tiles, groups_m,
                                                                       stream);
    });
}

যৌক্তিক ggml_sycl_grouped_dequant_gemm(ggml_type src0_type, যৌক্তিক reordered, ধ্রুবক শূন্য * src0_base, size_t expert_stride,
                                    ধ্রুবক দশমিক * src1, int32_t src1_prec, দশমিক * dst,
                                    ধ্রুবক int64_t * expert_row_offsets, int64_t n_as, int64_t M, int64_t K,
                                    int64_t যোগ_rows, std::vector<ggml_sycl_gg_tile> & tiles,
                                    ggml_sycl_pool & pool, dpct::queue_ptr stream) {
    int64_t n_active = 0;
    জন্য (int64_t e = 0; e < n_as; ++e) {
        n_active += expert_row_offsets[e + 1] > expert_row_offsets[e];
    }
    যদি (g_ggml_sycl_dynamic_precision == GGML_SYCL_DYNAMIC_PRECISION_F32 ||
        !ggml_sycl_xmx_gather_type_enabled(src0_type) || !fg_visit_type(src0_type, reordered, [](auto) {})) {
        ফেরত মিথ্যা;
    }
    যদি (!ggml_sycl_grouped_dequant_gemm_shape_ok(src0_type, M, K, যোগ_rows, n_active)) {
        ফেরত মিথ্যা;
    }
    ধ্রুবক পূর্ণসংখ্যা combo = fg_pick_combo(stream, GGML_TYPE_F32, src1_prec);
    যদি (combo < 0) {
        ফেরত মিথ্যা;
    }
    যৌক্তিক launched = মিথ্যা;
    fg_visit_combo(combo, [&](auto s) {
        launched = fg_grouped_run<decltype(s)>(src0_type, reordered, src0_base, expert_stride, src1, dst,
                                               expert_row_offsets, n_as, M, K, tiles, pool, stream);
    });
    ফেরত launched;
}
