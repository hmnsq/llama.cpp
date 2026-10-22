#అన్తర్భావ "fused-gemm.hpp"

#అన్తర్భావ <sycl/ext/oneapi/matrix/matrix.hpp>

#అన్తర్భావ <algorithm>
#అన్తర్భావ <string>
#అన్తర్భావ <tuple>
#అన్తర్భావ <mutex>
#అన్తర్భావ <set>
#అన్తర్భావ <unordered_map>

నామాకాశ mx = sycl::ext::oneapi::experimental::మాతృక;

// FG_ / fg_ is short for fused GEMM: the weights are dequantized inside the GEMM, into the XMX tiles.

// A k step is one 32-value weight sub-block; iq3_s and the other superblock formats split their
// superblock into steps of this width. The sub-groups of a work-group each walk their own K range
// and are summed at the end.
స్థిర constexpr పూర్ణాంకం FG_BK     = QK4_NL;
స్థిర constexpr పూర్ణాంకం FG_KSPLIT = 4;

// Element traits of one joint_matrix operand type. The A stage and the B pack compute in f32 and
// convert once, in registers, when they write the element, so any type costs the same one pass.
//   store: storage in SLM (A) and in the packed B buffer
//   mtype: matrix_type in matrix_combinations
//   mode:  GGML_SYCL_DYNAMIC_PRECISION value that selects this type
//   src:   ggml type that needs no conversion into this type (GGML_TYPE_COUNT: none)
//   slow:  XMX throughput class, 0 is fastest. f16 and bf16 share the DPAS rate; tf32 does half the
//          K per instruction. B60, Qwen3-30B-A3B pp512: f16 1108, bf16 1000, tf32 751 t/s
మూస <రకం_పేరు T> నిర్మాణం fg_elem;

మూస <> నిర్మాణం fg_elem<sycl::half> {
    ఉపయోగించి store = sycl::half;
    ఉపయోగించి pair  = sycl::half2;
    స్థిర constexpr mx::మాతృక_type mtype = mx::మాతృక_type::fp16;
    స్థిర constexpr పూర్ణాంకం             mode  = GGML_SYCL_DYNAMIC_PRECISION_F16;
    స్థిర constexpr ggml_type       src   = GGML_TYPE_F16;
    స్థిర constexpr పూర్ణాంకం             mant  = 10;
    స్థిర constexpr పూర్ణాంకం             slow  = 0;
    స్థిర store cvt(దశాంశం x) { తిరిగి (store) x; }
    స్థిర pair make(దశాంశం x, దశాంశం y) { తిరిగి pair((store) x, (store) y); }
};

// tf32 rounds to nearest even with plain bit ops: round_to_tf32 needs a SPIR-V extension that the
// DG2 AOT target rejects
స్థిర అంతఃవరుస uint32_t fg_విడత_బిట్లు(దశాంశం x, పూర్ణాంకం drop) {
    స్థిరరాశి uint32_t u = sycl::బిట్టు_cast<uint32_t>(x);
    ఒకవేళ ((u & 0x7f800000u) == 0x7f800000u) {
        తిరిగి (u & 0x7fffffu) ? u | (1u << drop) : u; // nan stays nan
    }
    తిరిగి u + ((1u << (drop - 1)) - 1) + ((u >> drop) & 1);
}

నిర్మాణం alignas(4) fg_bf16x2 {
    sycl::ext::oneapi::bfloat16 x, y;
};

మూస <> నిర్మాణం fg_elem<sycl::ext::oneapi::bfloat16> {
    ఉపయోగించి store = sycl::ext::oneapi::bfloat16;
    ఉపయోగించి pair  = fg_bf16x2;
    స్థిర constexpr mx::మాతృక_type mtype = mx::మాతృక_type::bf16;
    స్థిర constexpr పూర్ణాంకం             mode  = GGML_SYCL_DYNAMIC_PRECISION_BF16;
    స్థిర constexpr ggml_type       src   = GGML_TYPE_BF16;
    స్థిర constexpr పూర్ణాంకం             mant  = 7;
    స్థిర constexpr పూర్ణాంకం             slow  = 0;
    స్థిర store cvt(దశాంశం x) { తిరిగి store(x); }
    స్థిర pair make(దశాంశం x, దశాంశం y) { తిరిగి { cvt(x), cvt(y) }; }
};

// tf32 keeps f32 range and f16 mantissa, in f32 storage
మూస <> నిర్మాణం fg_elem<mx::precision::tf32> {
    ఉపయోగించి store = దశాంశం;
    ఉపయోగించి pair  = sycl::float2;
    స్థిర constexpr mx::మాతృక_type mtype = mx::మాతృక_type::tf32;
    స్థిర constexpr పూర్ణాంకం             mode  = GGML_SYCL_DYNAMIC_PRECISION_TF32;
    స్థిర constexpr ggml_type       src   = GGML_TYPE_లెక్క;
    స్థిర constexpr పూర్ణాంకం             mant  = 10;
    స్థిర constexpr పూర్ణాంకం             slow  = 1;
    స్థిర store cvt(దశాంశం x) { తిరిగి sycl::బిట్టు_cast<దశాంశం>(fg_విడత_బిట్లు(x, 13) & ~0x1fffu); }
    స్థిర pair make(దశాంశం x, దశాంశం y) { తిరిగి pair(cvt(x), cvt(y)); }
};

// One joint_matrix combination (A type, B type, TM x TN x TK, sub-group size; C and D are f32) and
// the tiling built on it. A sub-group owns SG_ROWS rows of A (at least 16) and BN (at least 32)
// columns of B. A and B may differ: the device lists the pairs it supports.
మూస <రకం_పేరు TA, రకం_పేరు TB, పూర్ణాంకం TM_, పూర్ణాంకం TN_, పూర్ణాంకం TK_, పూర్ణాంకం SG_> నిర్మాణం fg_combo {
    ఉపయోగించి ta  = TA;
    ఉపయోగించి tb  = TB;
    ఉపయోగించి EA  = fg_elem<TA>;
    ఉపయోగించి EB  = fg_elem<TB>;
    ఉపయోగించి tsa = రకం_పేరు EA::store;
    ఉపయోగించి tsb = రకం_పేరు EB::store;
    స్థిర constexpr పూర్ణాంకం TM = TM_;
    స్థిర constexpr పూర్ణాంకం TN = TN_;
    స్థిర constexpr పూర్ణాంకం TK = TK_;
    స్థిర constexpr పూర్ణాంకం SG = SG_;
    స్థిర constexpr పూర్ణాంకం VNNI    = 4 / పరిమాణం(tsb);  // K rows of B packed in one 32-bit word
    స్థిర constexpr పూర్ణాంకం SG_అడ్డువరుసలు = TM > 16 ? TM : 16;
    స్థిర constexpr పూర్ణాంకం RPL     = SG_అడ్డువరుసలు / SG;     // A rows one lane decodes per k step
    స్థిర constexpr పూర్ణాంకం MT      = SG_అడ్డువరుసలు / TM;
    స్థిర constexpr పూర్ణాంకం BN      = TN > 32 ? TN : 32;
    స్థిర constexpr పూర్ణాంకం NT      = BN / TN;
    స్థిర constexpr పూర్ణాంకం WG_కొలత = FG_KSPLIT * SG;
    స్థిర constexpr mx::layout b_layout = VNNI == 1 ? mx::layout::అడ్డువరుస_major : mx::layout::ext_intel_packed;
    // a 64-wide N is mostly padding here and a 32x64 f32 accumulator needs 128 registers per lane,
    // so it spills: 13x slower on B60
    స్థిర constexpr తార్కిక efficient = TN <= 32;
    static_assert(SG_అడ్డువరుసలు % SG == 0 && SG_అడ్డువరుసలు % TM == 0 && BN % TN == 0 && FG_BK % TK == 0, "bad tile");
    static_assert(BN <= GGML_SYCL_FG_MAX_N, "header gate must cover the tile width");
};

ఉపయోగించి fg_half = sycl::half;
ఉపయోగించి fg_bf16 = sycl::ext::oneapi::bfloat16;
ఉపయోగించి fg_tf32 = mx::precision::tf32;

// One bit of GGML_SYCL_XMX_GATHER_SHAPES per combination. Only combinations some device lists in
// matrix_combinations are built (appendix of sycl_ext_oneapi_matrix and the runtime's own list).
మూస <రకం_పేరు F> స్థిర శూన్య fg_సందర్శించు_combo(పూర్ణాంకం idx, F && f);
స్థిర constexpr పూర్ణాంకం FG_N_COMBOS = 8;

// A spir64_gen AOT build (GGML_SYCL_XMX_AOT_SG) drops the combinations of the other sub-group size
// entirely: ocloc rejects even an empty kernel that asks for a sub-group size it lacks.
మూస <పూర్ణాంకం SG> స్థిర constexpr తార్కిక fg_listed() {
#ఒకవేళ defined(GGML_SYCL_XMX_AOT_SG)
    తిరిగి SG == GGML_SYCL_XMX_AOT_SG;
#లేకపోతే
    తిరిగి నిజం;
#అయితే_ముగింపు
}

మూస <రకం_పేరు S, రకం_పేరు F> స్థిర శూన్య fg_call_combo(F && f) {
    ఒకవేళ constexpr (fg_listed<S::SG>()) {
        f(S{});
    }
}

మూస <రకం_పేరు F> స్థిర శూన్య fg_సందర్శించు_combo(పూర్ణాంకం idx, F && f) {
    ఎంపిక (idx) {
        స్థితి 0: fg_call_combo<fg_combo<fg_half, fg_half, 8, 16, 16, 16>>(f);  ఆపు; // Xe2, Xe3, Xe-HPC
        స్థితి 1: fg_call_combo<fg_combo<fg_half, fg_half, 16, 16, 16, 16>>(f); ఆపు; // Xe2, Xe3, Xe-HPC
        స్థితి 2: fg_call_combo<fg_combo<fg_half, fg_half, 32, 64, 16, 16>>(f); ఆపు; // Xe2, Xe3, Xe-HPC
        స్థితి 3: fg_call_combo<fg_combo<fg_half, fg_half, 32, 64, 32, 16>>(f); ఆపు; // Xe2, Xe3, Xe-HPC
        స్థితి 4: fg_call_combo<fg_combo<fg_half, fg_half, 8, 8, 16, 8>>(f);    ఆపు; // Xe-HPG (Arc A), ARL-H
        స్థితి 5: fg_call_combo<fg_combo<fg_tf32, fg_tf32, 8, 16, 8, 16>>(f);   ఆపు; // Xe2, Xe3, Xe-HPC
        స్థితి 6: fg_call_combo<fg_combo<fg_bf16, fg_bf16, 8, 16, 16, 16>>(f);  ఆపు; // Xe2, Xe3, Xe-HPC
        స్థితి 7: fg_call_combo<fg_combo<fg_bf16, fg_bf16, 8, 8, 16, 8>>(f);    ఆపు; // Xe-HPG (Arc A), ARL-H
        అప్రమేయ: GGML_ABORT("bad XMX combination %d", idx);
    }
}

// AOT with -fsycl-targets=intel_gpu_*: compile each tile body only for targets with its sub-group
// size, since IGC fails on the other ones. A JIT build keeps them all, but each combination lands in
// its own device image (joint_matrix is an optional kernel feature) and only a combination the
// device reports is launched, so the runtime never asks IGC for the others.
#ఒకవేళ defined(__SYCL_DEVICE_ONLY__)
#    ఒకవేళ __SYCL_TARGET_INTEL_GPU_ACM_G10__ || __SYCL_TARGET_INTEL_GPU_ACM_G11__ || __SYCL_TARGET_INTEL_GPU_ACM_G12__ || \
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
#అయితే_ముగింపు

మూస <పూర్ణాంకం SG> స్థిర constexpr తార్కిక fg_built() {
#ఒకవేళ defined(FG_AOT_SG)
    తిరిగి SG == FG_AOT_SG;
#లేకపోతే
    తిరిగి నిజం;
#అయితే_ముగింపు
}

// Upper bound on the tile count when total_rows rows are routed to n_as experts: the worst case gives
// each expert one row and fills whole tiles with the rest. The bound depends only on the shape, not
// on the routing, so the pool reuses one buffer every ubatch instead of keeping one per size seen.
స్థిర constexpr int64_t grouped_gemm_max_tiles(int64_t మొత్తం_అడ్డువరుసలు, int64_t n_as, int64_t BN) {
    తిరిగి మొత్తం_అడ్డువరుసలు <= n_as ? మొత్తం_అడ్డువరుసలు : n_as + (మొత్తం_అడ్డువరుసలు - n_as) / BN;
}
// Tiles do not cross experts, so the bound is not ceil(total_rows / BN): 34 rows over 2 experts with
// BN = 16 split 17 + 17 need 2 + 2 tiles, where the ceil gives 3.
static_assert(grouped_gemm_max_tiles(34, 2, 16) == 4);

// the device lists S with an f32 accumulator and output
మూస <రకం_పేరు S> స్థిర తార్కిక fg_device_has_combo(స్థిరరాశి std::vector<mx::combination> & combinations) {
    కొరకు (స్థిరరాశి స్వయంచాలక & c : combinations) {
        ఒకవేళ (c.atype == S::EA::mtype && c.btype == S::EB::mtype && c.ctype == mx::మాతృక_type::fp32 &&
            c.dtype == mx::మాతృక_type::fp32 &&
            (c.max_msize >= (size_t) S::TM || c.msize == (size_t) S::TM) &&
            (c.max_nsize >= (size_t) S::TN || c.nsize == (size_t) S::TN) &&
            (c.max_ksize >= (size_t) S::TK || c.ksize == (size_t) S::TK)) {
            తిరిగి నిజం;
        }
    }
    తిరిగి అబద్ధం;
}

మూస <రకం_పేరు T> స్థిర స్థిరరాశి అక్షరం * fg_type_name() {
    తిరిగి std::is_same_v<T, fg_half> ? "f16" : std::is_same_v<T, fg_bf16> ? "bf16" : "tf32";
}

స్థిర std::string fg_combo_name(పూర్ణాంకం idx) {
    std::string name;
    fg_సందర్శించు_combo(idx, [&](స్వయంచాలక s) {
        ఉపయోగించి S = decltype(s);
        name = std::string(fg_type_name<రకం_పేరు S::ta>()) + "x" + fg_type_name<రకం_పేరు S::tb>() + " " +
               std::to_string(S::TM) + "x" + std::to_string(S::TN) + "x" + std::to_string(S::TK) + " sg" +
               std::to_string(S::SG);
    });
    తిరిగి name;
}

// Combinations this build has kernels for and the device lists, one bit each. Cached per device:
// on a mixed box the first caller's verdict is not the others'.
స్థిర పూర్ణాంకం fg_device_combos(స్థిరరాశి sycl::device & dev) {
    స్థిర std::mutex                            mtx;
    స్థిర std::unordered_map<sycl::device, పూర్ణాంకం> తెలిసిన;
    std::lock_guard<std::mutex>                  తాళం(mtx);
    స్థిరరాశి స్వయంచాలక                                   it = తెలిసిన.find(dev);
    ఒకవేళ (it != తెలిసిన.end()) {
        తిరిగి it->second;
    }
    పూర్ణాంకం అందుబాటు = 0;
    ప్రయత్నించు {
        స్థిరరాశి స్వయంచాలక combinations = dev.get_info<sycl::ext::oneapi::experimental::info::device::మాతృక_combinations>();
        స్థిరరాశి స్వయంచాలక sg_sizes     = dev.get_info<sycl::info::device::sub_సమూహం_sizes>();
        కొరకు (పూర్ణాంకం idx = 0; idx < FG_N_COMBOS; ++idx) {
            fg_సందర్శించు_combo(idx, [&](స్వయంచాలక s) {
                ఉపయోగించి S = decltype(s);
                స్థిరరాశి తార్కిక sg_ok = std::find(sg_sizes.begin(), sg_sizes.end(), (size_t) S::SG) != sg_sizes.end();
                ఒకవేళ (sg_ok && fg_device_has_combo<S>(combinations)) {
                    అందుబాటు |= 1 << idx;
                }
            });
        }
    } పట్టుకో (స్థిరరాశి sycl::exception &) {
        అందుబాటు = 0;
    }
    GGML_LOG_INFO("%s: %s: XMX dequant-GEMM combinations available 0x%x, allowed 0x%x\n", __func__,
                  dev.get_info<sycl::info::device::name>().c_str(), అందుబాటు, g_ggml_sycl_xmx_gather_shapes);
    తెలిసిన.emplace(dev, అందుబాటు);
    తిరిగి అందుబాటు;
}

// Rank of combination S for a src1 of type src1_type, lower is better. Order:
//  1. throughput: a tile that does not spill, then the fastest type class of A and B
//  2. B type equal to the src1 type, so the pack is a plain copy
//  3. B at least as precise as f16
//  4. the device's native DPAS tile (8 x SG x 32 bytes of K), then the largest M x K
// A costs nothing to convert: the A stage emits any type at the same cost.
మూస <రకం_పేరు S> స్థిర int64_t fg_rank(ggml_type src1_type) {
    స్థిరరాశి int64_t spills  = !S::efficient;
    స్థిరరాశి int64_t slow    = std::max(S::EA::slow, S::EB::slow);
    స్థిరరాశి int64_t convert = S::EB::src != src1_type;
    స్థిరరాశి int64_t lossy   = S::EB::mant < 10;
    స్థిరరాశి int64_t foreign = !(S::TM == 8 && S::TN == S::SG);
    స్థిరరాశి int64_t mk      = 1024 - S::TM * S::TK;
    తిరిగి ((((spills * 2 + slow) * 2 + convert) * 2 + lossy) * 2 + foreign) * 2048 + mk;
}

// whether XMX operands of type mode (a GGML_SYCL_DYNAMIC_PRECISION value) meet the src1 request
// [TAG_GGML_PREC]. f16 lacks the f32 range that BF16 and F32 ask for; an F32 request goes only as far
// down as GGML_SYCL_DYNAMIC_REQUIRED_PRECISION allows.
స్థిర తార్కిక fg_బహుళకం_meets(పూర్ణాంకం mode, int32_t src1_prec) {
    ఒకవేళ (src1_prec == GGML_PREC_UNDEFINED || src1_prec >= GGML_PREC_F16) {
        తిరిగి నిజం;
    }
    ఒకవేళ (src1_prec >= GGML_PREC_BF16) {
        తిరిగి mode != GGML_SYCL_DYNAMIC_PRECISION_F16;
    }
    ఎంపిక (g_ggml_sycl_dynamic_required_precision) {
        స్థితి GGML_SYCL_DYNAMIC_PRECISION_TF32: తిరిగి mode == GGML_SYCL_DYNAMIC_PRECISION_TF32;
        స్థితి GGML_SYCL_DYNAMIC_PRECISION_BF16: తిరిగి mode != GGML_SYCL_DYNAMIC_PRECISION_F16;
        అప్రమేయ:                               తిరిగి అబద్ధం;
    }
}

// Best allowed combination for this call, or -1 if none. The type is GGML_SYCL_DYNAMIC_PRECISION if it
// meets the src1 request; if not, bf16 then tf32 for a BF16 request (fastest first) and tf32 then bf16
// for an F32 request (most mantissa first).
స్థిర పూర్ణాంకం fg_pick_combo(dpct::వరుస_ptr stream, ggml_type src1_type, int32_t src1_prec) {
    స్థిరరాశి sycl::device dev     = stream->get_device();
    స్థిరరాశి పూర్ణాంకం          allowed = fg_device_combos(dev) & g_ggml_sycl_xmx_gather_shapes;
    స్థిరరాశి తార్కిక         f32_req = src1_prec != GGML_PREC_UNDEFINED && src1_prec < GGML_PREC_BF16;
    స్థిరరాశి పూర్ణాంకం          modes[] = {
        g_ggml_sycl_dynamic_precision,
        f32_req ? GGML_SYCL_DYNAMIC_PRECISION_TF32 : GGML_SYCL_DYNAMIC_PRECISION_BF16,
        f32_req ? GGML_SYCL_DYNAMIC_PRECISION_BF16 : GGML_SYCL_DYNAMIC_PRECISION_TF32,
    };
    పూర్ణాంకం     ఉత్తమ      = -1;
    int64_t ఉత్తమ_rank = 0;
    కొరకు (పూర్ణాంకం i = 0; i < 3 && ఉత్తమ < 0; ++i) {
        స్థిరరాశి పూర్ణాంకం mode = modes[i];
        ఒకవేళ (!fg_బహుళకం_meets(mode, src1_prec)) {
            కొనసాగు;
        }
        కొరకు (పూర్ణాంకం idx = 0; idx < FG_N_COMBOS; ++idx) {
            ఒకవేళ (!(allowed & (1 << idx))) {
                కొనసాగు;
            }
            fg_సందర్శించు_combo(idx, [&](స్వయంచాలక s) {
                ఉపయోగించి S = decltype(s);
                ఒకవేళ (S::EA::mode != mode || S::EB::mode != mode) {
                    తిరిగి;
                }
                స్థిరరాశి int64_t rank = fg_rank<S>(src1_type);
                ఒకవేళ (ఉత్తమ < 0 || rank < ఉత్తమ_rank) {
                    ఉత్తమ      = idx;
                    ఉత్తమ_rank = rank;
                }
            });
        }
    }
    // log each distinct decision once
    స్థిర std::mutex                                      mtx;
    స్థిర std::set<std::tuple<size_t, పూర్ణాంకం, int32_t, పూర్ణాంకం>> చూసిన;
    std::lock_guard<std::mutex>                            తాళం(mtx);
    ఒకవేళ (చూసిన.emplace(std::hash<sycl::device>{}(dev), (పూర్ణాంకం) src1_type, src1_prec, ఉత్తమ).second) {
        GGML_LOG_INFO("%s: src1 %s, src1 prec %d -> %s\n", __func__, ggml_type_name(src1_type), src1_prec,
                      ఉత్తమ >= 0 ? fg_combo_name(ఉత్తమ).c_str() : "none (library GEMM)");
    }
    తిరిగి ఉత్తమ;
}

// src1 [N][K] -> packed [K/V][Npad][V] so B tiles load straight from global memory
మూస <రకం_పేరు E, రకం_పేరు T_src>
స్థిర శూన్య fused_gemm_pack_b(స్థిరరాశి T_src * y, రకం_పేరు E::store * packed, పూర్ణాంకం N, పూర్ణాంకం Npad, పూర్ణాంకం K,
                              dpct::వరుస_ptr stream) {
    constexpr పూర్ణాంకం V   = 4 / పరిమాణం(రకం_పేరు E::store);
    స్థిరరాశి పూర్ణాంకం     kqs = K / V;
    stream->parallel_for(sycl::range<1>((size_t) Npad * kqs), [=](sycl::id<1> id) {
        స్థిరరాశి పూర్ణాంకం idx = id[0];
        స్థిరరాశి పూర్ణాంకం n   = idx / kqs;
        స్థిరరాశి పూర్ణాంకం kq  = idx - n * kqs;
        రకం_పేరు E::store vals[V] = {};
        ఒకవేళ (n < N) {
            స్థిరరాశి T_src * src = y + (size_t) n * K + V * kq;
#సూచన unroll
            కొరకు (పూర్ణాంకం v = 0; v < V; ++v) {
                vals[v] = E::cvt((దశాంశం) src[v]);
            }
        }
        రకం_పేరు E::store * బహిర్గతం = packed + ((size_t) kq * Npad + n) * V;
#సూచన unroll
        కొరకు (పూర్ణాంకం v = 0; v < V; ++v) {
            బహిర్గతం[v] = vals[v];
        }
    });
}

// A stage: one lane owns one row and decodes FG_BK values of it per k step, with every scale
// folded into the value so the mad below sees plain A elements. One overload per weight format.
మూస <రకం_పేరు E>
స్థిర __dpct_inline__ శూన్య fg_stage_a(స్థిరరాశి ఖండం_iq4_nl * __restrict__ xrow, స్థిరరాశి పూర్ణాంకం kb, రకం_పేరు E::pair * a) {
    స్థిరరాశి ఖండం_iq4_nl blk = xrow[kb];
    స్థిరరాశి దశాంశం        d   = (దశాంశం) blk.d;
#సూచన unroll
    కొరకు (పూర్ణాంకం j = 0; j < QK4_NL / 2; j += 2) {
        స్థిరరాశి uint8_t q0 = blk.qs[j];
        స్థిరరాశి uint8_t q1 = blk.qs[j + 1];
        a[j / 2]     = E::make(d * kvalues_iq4nl[q0 & 0xf], d * kvalues_iq4nl[q1 & 0xf]);
        a[j / 2 + 8] = E::make(d * kvalues_iq4nl[q0 >> 4], d * kvalues_iq4nl[q1 >> 4]);
    }
}

// iq3_s: k step kb is sub-block kb % 8 of superblock kb / 8. The superblock is 110 bytes, so read
// only the fields of that sub-block instead of copying the block. Same decode as
// dequantize_block_iq3_s: grid entries are taken as dwords and the sign bit is a plain shift.
మూస <రకం_పేరు E>
స్థిర __dpct_inline__ శూన్య fg_stage_a(స్థిరరాశి ఖండం_iq3_s * __restrict__ xrow, స్థిరరాశి పూర్ణాంకం kb, రకం_పేరు E::pair * a) {
    static_assert(QK_K == 256, "the iq3_s A stage assumes 8 sub-blocks per superblock");
    స్థిరరాశి ఖండం_iq3_s * blk = xrow + kb / (QK_K / 32);
    స్థిరరాశి పూర్ణాంకం           ib8 = kb % (QK_K / 32);
    స్థిరరాశి uint8_t *     qs  = blk->qs + 8 * ib8;
    స్థిరరాశి పూర్ణాంకం           qh  = blk->qh[ib8];
    స్థిరరాశి దశాంశం         d   = (దశాంశం) blk->d * (1 + 2 * ((blk->scales[ib8 / 2] >> (4 * (ib8 % 2))) & 0xf));
#సూచన unroll
    కొరకు (పూర్ణాంకం il = 0; il < 4; ++il) {
        స్థిరరాశి uint32_t grid1 = iq3s_grid[qs[2 * il + 0] | ((qh << (8 - 2 * il)) & 256)];
        స్థిరరాశి uint32_t grid2 = iq3s_grid[qs[2 * il + 1] | ((qh << (7 - 2 * il)) & 256)];
        స్థిరరాశి పూర్ణాంకం      signs = blk->signs[4 * ib8 + il];
#సూచన unroll
        కొరకు (పూర్ణాంకం j = 0; j < 2; ++j) {
            స్థిరరాశి దశాంశం g1a = (దశాంశం) ((grid1 >> (16 * j + 0)) & 0xff);
            స్థిరరాశి దశాంశం g1b = (దశాంశం) ((grid1 >> (16 * j + 8)) & 0xff);
            స్థిరరాశి దశాంశం g2a = (దశాంశం) ((grid2 >> (16 * j + 0)) & 0xff);
            స్థిరరాశి దశాంశం g2b = (దశాంశం) ((grid2 >> (16 * j + 8)) & 0xff);
            స్థిరరాశి పూర్ణాంకం   s   = 2 * j;
            a[4 * il + j]     = E::make(d * ((signs & (1 << (s + 0))) ? -g1a : g1a),
                                        d * ((signs & (1 << (s + 1))) ? -g1b : g1b));
            a[4 * il + j + 2] = E::make(d * ((signs & (1 << (s + 4))) ? -g2a : g2a),
                                        d * ((signs & (1 << (s + 5))) ? -g2b : g2b));
        }
    }
}

// values per stored block, so a row of K values is K/qk blocks
మూస <రకం_పేరు ఖండం_q_t> నిర్మాణం fg_ఖండం_traits;
మూస <> నిర్మాణం fg_ఖండం_traits<ఖండం_iq4_nl> { స్థిర constexpr పూర్ణాంకం qk = QK4_NL; };
మూస <> నిర్మాణం fg_ఖండం_traits<ఖండం_iq3_s>  { స్థిర constexpr పూర్ణాంకం qk = QK_K; };
మూస <> నిర్మాణం fg_ఖండం_traits<ఖండం_iq3_xxs> { స్థిర constexpr పూర్ణాంకం qk = QK_K; };
మూస <> నిర్మాణం fg_ఖండం_traits<ఖండం_iq4_xs>  { స్థిర constexpr పూర్ణాంకం qk = QK_K; };
మూస <> నిర్మాణం fg_ఖండం_traits<ఖండం_iq2_xxs> { స్థిర constexpr పూర్ణాంకం qk = QK_K; };
మూస <> నిర్మాణం fg_ఖండం_traits<ఖండం_iq2_xs>  { స్థిర constexpr పూర్ణాంకం qk = QK_K; };
మూస <> నిర్మాణం fg_ఖండం_traits<ఖండం_iq2_s>   { స్థిర constexpr పూర్ణాంకం qk = QK_K; };
మూస <> నిర్మాణం fg_ఖండం_traits<ఖండం_iq1_s>   { స్థిర constexpr పూర్ణాంకం qk = QK_K; };
మూస <> నిర్మాణం fg_ఖండం_traits<ఖండం_iq1_m>   { స్థిర constexpr పూర్ణాంకం qk = QK_K; };
మూస <> నిర్మాణం fg_ఖండం_traits<ఖండం_q8_0>    { స్థిర constexpr పూర్ణాంకం qk = QK8_0; };
మూస <> నిర్మాణం fg_ఖండం_traits<ఖండం_q4_K>    { స్థిర constexpr పూర్ణాంకం qk = QK_K; };
మూస <> నిర్మాణం fg_ఖండం_traits<ఖండం_q5_K>    { స్థిర constexpr పూర్ణాంకం qk = QK_K; };
మూస <> నిర్మాణం fg_ఖండం_traits<ఖండం_q6_K>    { స్థిర constexpr పూర్ణాంకం qk = QK_K; };

// The A stages below are the dequantize_block_iq* kernels rewritten for one k step. There a
// work-item handled one quarter (il) of one 32-wide sub-block (ib); here one lane produces the
// whole step, so il becomes a loop and ib is kb inside the superblock. Each quarter yields 8
// consecutive values, i.e. 4 pairs at a[4*il], so nothing larger than 8 floats is ever live.
#నిర్వచించు FG_SUPERBLOCK(T)                                                         \
    static_assert(QK_K == 256, "the " #T " A stage assumes 8 sub-blocks per superblock"); \
    స్థిరరాశి T * blk = xrow + kb / (QK_K / 32);                                     \
    స్థిరరాశి పూర్ణాంకం ib  = kb % (QK_K / 32)

మూస <రకం_పేరు E>
స్థిర __dpct_inline__ శూన్య fg_pack_quarter(స్థిరరాశి దశాంశం * __restrict__ t, రకం_పేరు E::pair * a, పూర్ణాంకం il) {
#సూచన unroll
    కొరకు (పూర్ణాంకం j = 0; j < 4; ++j) {
        a[4 * il + j] = E::make(t[2 * j], t[2 * j + 1]);
    }
}

మూస <రకం_పేరు E>
స్థిర __dpct_inline__ శూన్య fg_stage_a(స్థిరరాశి ఖండం_iq4_xs * __restrict__ xrow, స్థిరరాశి పూర్ణాంకం kb, రకం_పేరు E::pair * a) {
    FG_SUPERBLOCK(ఖండం_iq4_xs);
    // low nibbles fill the first half of the step, high nibbles the second, so the two halves
    // land at a[0..7] and a[8..15] and no quarter loop is needed
    స్థిరరాశి దశాంశం d = (దశాంశం) blk->d *
        ((((blk->scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) | (((blk->scales_h >> (2 * ib)) & 3) << 4)) - 32);
    స్థిరరాశి uint8_t * q4 = blk->qs + 16 * ib;
#సూచన unroll
    కొరకు (పూర్ణాంకం j = 0; j < 8; ++j) {
        a[j]     = E::make(d * kvalues_iq4nl[q4[2 * j] & 0xf], d * kvalues_iq4nl[q4[2 * j + 1] & 0xf]);
        a[8 + j] = E::make(d * kvalues_iq4nl[q4[2 * j] >> 4],  d * kvalues_iq4nl[q4[2 * j + 1] >> 4]);
    }
}

మూస <రకం_పేరు E>
స్థిర __dpct_inline__ శూన్య fg_stage_a(స్థిరరాశి ఖండం_iq3_xxs * __restrict__ xrow, స్థిరరాశి పూర్ణాంకం kb, రకం_పేరు E::pair * a) {
    FG_SUPERBLOCK(ఖండం_iq3_xxs);
    స్థిరరాశి uint8_t *  q3    = blk->qs + 8 * ib;
    స్థిరరాశి uint16_t * gas   = (స్థిరరాశి uint16_t *) (blk->qs + QK_K / 4) + 2 * ib;
    స్థిరరాశి uint32_t   aux32 = gas[0] | (gas[1] << 16);
    స్థిరరాశి దశాంశం      d     = (దశాంశం) blk->d * (0.5f + (aux32 >> 28)) * 0.5f;
#సూచన unroll
    కొరకు (పూర్ణాంకం il = 0; il < 4; ++il) {
        స్థిరరాశి uint8_t * grid1 = (స్థిరరాశి uint8_t *) (iq3xxs_grid + q3[2 * il + 0]);
        స్థిరరాశి uint8_t * grid2 = (స్థిరరాశి uint8_t *) (iq3xxs_grid + q3[2 * il + 1]);
        స్థిరరాశి uint8_t   signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];
        దశాంశం t[8];
#సూచన unroll
        కొరకు (పూర్ణాంకం j = 0; j < 4; ++j) {
            t[j + 0] = d * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f);
            t[j + 4] = d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

మూస <రకం_పేరు E>
స్థిర __dpct_inline__ శూన్య fg_stage_a(స్థిరరాశి ఖండం_iq2_xxs * __restrict__ xrow, స్థిరరాశి పూర్ణాంకం kb, రకం_పేరు E::pair * a) {
    FG_SUPERBLOCK(ఖండం_iq2_xxs);
    స్థిరరాశి uint16_t * q2    = blk->qs + 4 * ib;
    స్థిరరాశి uint8_t *  aux8  = (స్థిరరాశి uint8_t *) q2;
    స్థిరరాశి uint32_t   aux32 = q2[2] | (q2[3] << 16);
    స్థిరరాశి దశాంశం      d     = (దశాంశం) blk->d * (0.5f + (aux32 >> 28)) * 0.25f;
#సూచన unroll
    కొరకు (పూర్ణాంకం il = 0; il < 4; ++il) {
        స్థిరరాశి uint8_t * grid  = (స్థిరరాశి uint8_t *) (iq2xxs_grid + aux8[il]);
        స్థిరరాశి uint8_t   signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];
        దశాంశం t[8];
#సూచన unroll
        కొరకు (పూర్ణాంకం j = 0; j < 8; ++j) {
            t[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

మూస <రకం_పేరు E>
స్థిర __dpct_inline__ శూన్య fg_stage_a(స్థిరరాశి ఖండం_iq2_xs * __restrict__ xrow, స్థిరరాశి పూర్ణాంకం kb, రకం_పేరు E::pair * a) {
    FG_SUPERBLOCK(ఖండం_iq2_xs);
    స్థిరరాశి uint16_t * q2 = blk->qs + 4 * ib;
#సూచన unroll
    కొరకు (పూర్ణాంకం il = 0; il < 4; ++il) {
        స్థిరరాశి uint8_t * grid  = (స్థిరరాశి uint8_t *) (iq2xs_grid + (q2[il] & 511));
        స్థిరరాశి దశాంశం     d     = (దశాంశం) blk->d * (0.5f + ((blk->scales[ib] >> (4 * (il / 2))) & 0xf)) * 0.25f;
        స్థిరరాశి uint8_t   signs = ksigns_iq2xs[q2[il] >> 9];
        దశాంశం t[8];
#సూచన unroll
        కొరకు (పూర్ణాంకం j = 0; j < 8; ++j) {
            t[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

మూస <రకం_పేరు E>
స్థిర __dpct_inline__ శూన్య fg_stage_a(స్థిరరాశి ఖండం_iq2_s * __restrict__ xrow, స్థిరరాశి పూర్ణాంకం kb, రకం_పేరు E::pair * a) {
    FG_SUPERBLOCK(ఖండం_iq2_s);
#సూచన unroll
    కొరకు (పూర్ణాంకం il = 0; il < 4; ++il) {
        స్థిరరాశి uint8_t * grid =
            (స్థిరరాశి uint8_t *) (iq2s_grid + (blk->qs[4 * ib + il] | ((blk->qh[ib] << (8 - 2 * il)) & 0x300)));
        స్థిరరాశి దశాంశం   d     = (దశాంశం) blk->d * (0.5f + ((blk->scales[ib] >> (4 * (il / 2))) & 0xf)) * 0.25f;
        స్థిరరాశి uint8_t signs = blk->qs[QK_K / 8 + 4 * ib + il];
        దశాంశం t[8];
#సూచన unroll
        కొరకు (పూర్ణాంకం j = 0; j < 8; ++j) {
            t[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

మూస <రకం_పేరు E>
స్థిర __dpct_inline__ శూన్య fg_stage_a(స్థిరరాశి ఖండం_iq1_s * __restrict__ xrow, స్థిరరాశి పూర్ణాంకం kb, రకం_పేరు E::pair * a) {
    FG_SUPERBLOCK(ఖండం_iq1_s);
    స్థిరరాశి దశాంశం delta = blk->qh[ib] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
    స్థిరరాశి దశాంశం d     = (దశాంశం) blk->d * (2 * ((blk->qh[ib] >> 12) & 7) + 1);
#సూచన unroll
    కొరకు (పూర్ణాంకం il = 0; il < 4; ++il) {
        uint32_t       grid32[2];
        స్థిరరాశి int8_t * q = (స్థిరరాశి int8_t *) grid32;
        grid32[0] = iq1s_grid_gpu[blk->qs[4 * ib + il] | (((blk->qh[ib] >> (3 * il)) & 7) << 8)];
        grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
        grid32[0] &= 0x0f0f0f0f;
        దశాంశం t[8];
#సూచన unroll
        కొరకు (పూర్ణాంకం j = 0; j < 8; ++j) {
            t[j] = d * (q[j] + delta);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

మూస <రకం_పేరు E>
స్థిర __dpct_inline__ శూన్య fg_stage_a(స్థిరరాశి ఖండం_iq1_m * __restrict__ xrow, స్థిరరాశి పూర్ణాంకం kb, రకం_పేరు E::pair * a) {
    FG_SUPERBLOCK(ఖండం_iq1_m);
    స్థిరరాశి uint16_t * sc = (స్థిరరాశి uint16_t *) blk->scales;
    iq1m_scale_t     scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
#సూచన unroll
    కొరకు (పూర్ణాంకం il = 0; il < 4; ++il) {
        స్థిరరాశి పూర్ణాంకం   ib16  = 2 * ib + il / 2;
        స్థిరరాశి దశాంశం d     = (దశాంశం) scale.f16 * (2 * ((sc[ib16 / 4] >> (3 * (ib16 % 4))) & 0x7) + 1);
        స్థిరరాశి దశాంశం delta = blk->qh[2 * ib + il / 2] & (0x08 << (4 * (il % 2))) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
        uint32_t       grid32[2];
        స్థిరరాశి int8_t * q = (స్థిరరాశి int8_t *) grid32;
        grid32[0] = iq1s_grid_gpu[blk->qs[4 * ib + il] |
                                  (((blk->qh[2 * ib + il / 2] >> (4 * (il % 2))) & 7) << 8)];
        grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
        grid32[0] &= 0x0f0f0f0f;
        దశాంశం t[8];
#సూచన unroll
        కొరకు (పూర్ణాంకం j = 0; j < 8; ++j) {
            t[j] = d * (q[j] + delta);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}


// q8_0 and the k-quants. Each decode takes the fields of one 32-value sub-block, so the canonical
// layout and the reorder (SoA) layout of reorder_qw() share it and differ only in where the fields
// live, as in dequantize.hpp.
మూస <రకం_పేరు E>
స్థిర __dpct_inline__ శూన్య fg_decode_q8_0(స్థిరరాశి int8_t * __restrict__ qs, స్థిరరాశి దశాంశం d, రకం_పేరు E::pair * a) {
#సూచన unroll
    కొరకు (పూర్ణాంకం j = 0; j < QK8_0 / 2; ++j) {
        a[j] = E::make(d * qs[2 * j], d * qs[2 * j + 1]);
    }
}

// same unpack as get_scale_min_k4() in dequantize.hpp
స్థిర __dpct_inline__ శూన్య fg_scale_min_k4(స్థిరరాశి పూర్ణాంకం j, స్థిరరాశి uint8_t * __restrict__ q, uint8_t & d, uint8_t & m) {
    ఒకవేళ (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } లేకపోతే {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

// q4_K sub-block ib (0..7) uses scale/min pair ib and the low (even ib) or high (odd ib) nibbles of
// qs[32 * (ib / 2) ...], as in dequantize_row_q4_K
మూస <రకం_పేరు E>
స్థిర __dpct_inline__ శూన్య fg_decode_q4_K(స్థిరరాశి uint8_t * __restrict__ qs, స్థిరరాశి uint8_t * __restrict__ scales,
                                           స్థిరరాశి sycl::half2 dm, స్థిరరాశి పూర్ణాంకం ib, రకం_పేరు E::pair * a) {
    uint8_t sc, mb;
    fg_scale_min_k4(ib, scales, sc, mb);
    స్థిరరాశి దశాంశం     d     = (దశాంశం) dm[0] * sc;
    స్థిరరాశి దశాంశం     m     = (దశాంశం) dm[1] * mb;
    స్థిరరాశి uint8_t * q     = qs + 32 * (ib / 2);
    స్థిరరాశి పూర్ణాంకం       జరుపు = 4 * (ib % 2);
#సూచన unroll
    కొరకు (పూర్ణాంకం j = 0; j < 16; ++j) {
        a[j] = E::make(d * ((q[2 * j] >> జరుపు) & 0xF) - m, d * ((q[2 * j + 1] >> జరుపు) & 0xF) - m);
    }
}

// q5_K: q4_K plus one high bit per value, bit ib of qh[l]
మూస <రకం_పేరు E>
స్థిర __dpct_inline__ శూన్య fg_decode_q5_K(స్థిరరాశి uint8_t * __restrict__ qs, స్థిరరాశి uint8_t * __restrict__ qh,
                                           స్థిరరాశి uint8_t * __restrict__ scales, స్థిరరాశి sycl::half2 dm, స్థిరరాశి పూర్ణాంకం ib,
                                           రకం_పేరు E::pair * a) {
    uint8_t sc, mb;
    fg_scale_min_k4(ib, scales, sc, mb);
    స్థిరరాశి దశాంశం     d     = (దశాంశం) dm[0] * sc;
    స్థిరరాశి దశాంశం     m     = (దశాంశం) dm[1] * mb;
    స్థిరరాశి uint8_t * q     = qs + 32 * (ib / 2);
    స్థిరరాశి పూర్ణాంకం       జరుపు = 4 * (ib % 2);
#సూచన unroll
    కొరకు (పూర్ణాంకం j = 0; j < 16; ++j) {
        స్థిరరాశి పూర్ణాంకం l = 2 * j;
        a[j] = E::make(d * (((q[l] >> జరుపు) & 0xF) | (((qh[l] >> ib) & 1) << 4)) - m,
                       d * (((q[l + 1] >> జరుపు) & 0xF) | (((qh[l + 1] >> ib) & 1) << 4)) - m);
    }
}

// q6_K: sub-block ib is quarter r = ib % 4 of half h = ib / 4. The half selects ql + 64h, qh + 32h
// and scales + 8h; the quarter selects ql + 32(r & 1), the ql nibble r / 2, the qh bit pair r and
// scales + 2r, as in dequantize_row_q6_K. The scale changes at value 16 of the sub-block.
మూస <రకం_పేరు E>
స్థిర __dpct_inline__ శూన్య fg_decode_q6_K(స్థిరరాశి uint8_t * __restrict__ ql, స్థిరరాశి uint8_t * __restrict__ qh,
                                           స్థిరరాశి int8_t * __restrict__ scales, స్థిరరాశి దశాంశం d, స్థిరరాశి పూర్ణాంకం ib,
                                           రకం_పేరు E::pair * a) {
    స్థిరరాశి పూర్ణాంకం       h  = ib / 4;
    స్థిరరాశి పూర్ణాంకం       r  = ib % 4;
    స్థిరరాశి uint8_t * q  = ql + 64 * h + 32 * (r & 1);
    స్థిరరాశి uint8_t * hb = qh + 32 * h;
    స్థిరరాశి int8_t *  sc = scales + 8 * h + 2 * r;
#సూచన unroll
    కొరకు (పూర్ణాంకం j = 0; j < 16; ++j) {
        స్థిరరాశి పూర్ణాంకం   l  = 2 * j;
        స్థిరరాశి దశాంశం dl = d * sc[j / 8];
        స్థిరరాశి పూర్ణాంకం   q0 = (((q[l] >> (4 * (r / 2))) & 0xF) | (((hb[l] >> (2 * r)) & 3) << 4)) - 32;
        స్థిరరాశి పూర్ణాంకం   q1 = (((q[l + 1] >> (4 * (r / 2))) & 0xF) | (((hb[l + 1] >> (2 * r)) & 3) << 4)) - 32;
        a[j] = E::make(dl * q0, dl * q1);
    }
}

మూస <రకం_పేరు E>
స్థిర __dpct_inline__ శూన్య fg_stage_a(స్థిరరాశి ఖండం_q8_0 * __restrict__ xrow, స్థిరరాశి పూర్ణాంకం kb, రకం_పేరు E::pair * a) {
    fg_decode_q8_0<E>(xrow[kb].qs, (దశాంశం) xrow[kb].d, a);
}

మూస <రకం_పేరు E>
స్థిర __dpct_inline__ శూన్య fg_stage_a(స్థిరరాశి ఖండం_q4_K * __restrict__ xrow, స్థిరరాశి పూర్ణాంకం kb, రకం_పేరు E::pair * a) {
    FG_SUPERBLOCK(ఖండం_q4_K);
    fg_decode_q4_K<E>(blk->qs, blk->scales, blk->dm, ib, a);
}

మూస <రకం_పేరు E>
స్థిర __dpct_inline__ శూన్య fg_stage_a(స్థిరరాశి ఖండం_q5_K * __restrict__ xrow, స్థిరరాశి పూర్ణాంకం kb, రకం_పేరు E::pair * a) {
    FG_SUPERBLOCK(ఖండం_q5_K);
    fg_decode_q5_K<E>(blk->qs, blk->qh, blk->scales, blk->dm, ib, a);
}

మూస <రకం_పేరు E>
స్థిర __dpct_inline__ శూన్య fg_stage_a(స్థిరరాశి ఖండం_q6_K * __restrict__ xrow, స్థిరరాశి పూర్ణాంకం kb, రకం_పేరు E::pair * a) {
    FG_SUPERBLOCK(ఖండం_q6_K);
    fg_decode_q6_K<E>(blk->ql, blk->qh, blk->scales, (దశాంశం) blk->d, ib, a);
}

// Reorder (SoA) layout: each block field is one stream over the nblocks of the matrix (of the expert
// slice for MUL_MAT_ID), in the order reorder_qw() writes them. Block ib holds k step kb.
మూస <రకం_పేరు ఖండం_q_t> నిర్మాణం fg_soa {
    స్థిర constexpr తార్కిక supported = అబద్ధం;
};

మూస <> నిర్మాణం fg_soa<ఖండం_q8_0> {
    స్థిర constexpr తార్కిక supported = నిజం;
    // [qs][d]
    మూస <రకం_పేరు E>
    స్థిర __dpct_inline__ శూన్య stage(స్థిరరాశి uint8_t * x, స్థిరరాశి size_t nblocks, స్థిరరాశి size_t ib, స్థిరరాశి పూర్ణాంకం,
                                      రకం_పేరు E::pair * a) {
        స్థిరరాశి దశాంశం d = (దశాంశం) ((స్థిరరాశి sycl::half *) (x + nblocks * QK8_0))[ib];
        fg_decode_q8_0<E>((స్థిరరాశి int8_t *) x + ib * QK8_0, d, a);
    }
};

మూస <> నిర్మాణం fg_soa<ఖండం_q4_K> {
    స్థిర constexpr తార్కిక supported = నిజం;
    // [qs][scales][dm]
    మూస <రకం_పేరు E>
    స్థిర __dpct_inline__ శూన్య stage(స్థిరరాశి uint8_t * x, స్థిరరాశి size_t nblocks, స్థిరరాశి size_t ib, స్థిరరాశి పూర్ణాంకం kb,
                                      రకం_పేరు E::pair * a) {
        స్థిరరాశి uint8_t *   scales = x + nblocks * (QK_K / 2);
        స్థిరరాశి sycl::half2 dm     = ((స్థిరరాశి sycl::half2 *) (scales + nblocks * K_SCALE_కొలత))[ib];
        fg_decode_q4_K<E>(x + ib * (QK_K / 2), scales + ib * K_SCALE_కొలత, dm, kb % (QK_K / 32), a);
    }
};

మూస <> నిర్మాణం fg_soa<ఖండం_q5_K> {
    స్థిర constexpr తార్కిక supported = నిజం;
    // [qs][qh][scales][dm]
    మూస <రకం_పేరు E>
    స్థిర __dpct_inline__ శూన్య stage(స్థిరరాశి uint8_t * x, స్థిరరాశి size_t nblocks, స్థిరరాశి size_t ib, స్థిరరాశి పూర్ణాంకం kb,
                                      రకం_పేరు E::pair * a) {
        స్థిరరాశి uint8_t *   qh     = x + nblocks * (QK_K / 2);
        స్థిరరాశి uint8_t *   scales = qh + nblocks * (QK_K / 8);
        స్థిరరాశి sycl::half2 dm     = ((స్థిరరాశి sycl::half2 *) (scales + nblocks * K_SCALE_కొలత))[ib];
        fg_decode_q5_K<E>(x + ib * (QK_K / 2), qh + ib * (QK_K / 8), scales + ib * K_SCALE_కొలత, dm,
                          kb % (QK_K / 32), a);
    }
};

మూస <> నిర్మాణం fg_soa<ఖండం_q6_K> {
    స్థిర constexpr తార్కిక supported = నిజం;
    // [ql][qh][scales][d]
    మూస <రకం_పేరు E>
    స్థిర __dpct_inline__ శూన్య stage(స్థిరరాశి uint8_t * x, స్థిరరాశి size_t nblocks, స్థిరరాశి size_t ib, స్థిరరాశి పూర్ణాంకం kb,
                                      రకం_పేరు E::pair * a) {
        స్థిరరాశి uint8_t * qh     = x + nblocks * (QK_K / 2);
        స్థిరరాశి uint8_t * scales = qh + nblocks * (QK_K / 4);
        స్థిరరాశి దశాంశం     d      = (దశాంశం) ((స్థిరరాశి sycl::half *) (scales + nblocks * (QK_K / 16)))[ib];
        fg_decode_q6_K<E>(x + ib * (QK_K / 2), qh + ib * (QK_K / 4), (స్థిరరాశి int8_t *) scales + ib * (QK_K / 16), d,
                          kb % (QK_K / 32), a);
    }
};


// one SG_ROWS x BN output tile: B columns [b0, b0 + BN) of packed_b go to dst columns [n0, n1),
// n1 - n0 <= BN
మూస <రకం_పేరు S, రకం_పేరు ఖండం_q_t, తార్కిక reordered>
స్థిర శూన్య fused_dequant_gemm_tile(
    స్థిరరాశి ఖండం_q_t * __restrict__ x,
    స్థిరరాశి రకం_పేరు S::tsb * __restrict__ packed_b,
    దశాంశం * __restrict__ dst,
    స్థిరరాశి పూర్ణాంకం M, స్థిరరాశి పూర్ణాంకం Npad, స్థిరరాశి పూర్ణాంకం K, స్థిరరాశి పూర్ణాంకం ldd,
    స్థిరరాశి పూర్ణాంకం b0, స్థిరరాశి పూర్ణాంకం n0, స్థిరరాశి పూర్ణాంకం n1,
    sycl::local_accessor<రకం_పేరు S::tsa, 1> tile_a,
    sycl::local_accessor<దశాంశం, 1> tile_c,
    స్థిరరాశి sycl::nd_అంశం<2> & అంశం) {
    ఒకవేళ constexpr (fg_built<S::SG>()) {
        ఉపయోగించి EA = రకం_పేరు S::EA;
        ఉపయోగించి TA = రకం_పేరు S::ta;
        ఉపయోగించి TB = రకం_పేరు S::tb;
        స్థిరరాశి స్వయంచాలక sg     = అంశం.get_sub_సమూహం();
        స్థిరరాశి పూర్ణాంకం  sg_id  = sg.get_సమూహం_id()[0];
        స్థిరరాశి పూర్ణాంకం  lane   = sg.get_local_id()[0];
        స్థిరరాశి పూర్ణాంకం  m0     = అంశం.get_సమూహం(1) * S::SG_అడ్డువరుసలు;
        స్థిరరాశి పూర్ణాంకం  nstep  = K / FG_BK;
        స్థిరరాశి పూర్ణాంకం  a_పునాది = sg_id * S::SG_అడ్డువరుసలు * FG_BK;
        స్థిరరాశి పూర్ణాంకం  c_పునాది = sg_id * S::SG_అడ్డువరుసలు * S::BN;

        mx::joint_మాతృక<sycl::sub_సమూహం, దశాంశం, mx::use::సంచాయకం, S::TM, S::TN> acc[S::MT][S::NT];
#సూచన unroll
        కొరకు (పూర్ణాంకం mt = 0; mt < S::MT; ++mt) {
#సూచన unroll
            కొరకు (పూర్ణాంకం nt = 0; nt < S::NT; ++nt) {
                mx::joint_మాతృక_fill(sg, acc[mt][nt], 0.0f);
            }
        }

        // lane decodes rows lane, lane + SG, ... of the sub-group's SG_ROWS
        constexpr పూర్ణాంకం         KPB     = fg_ఖండం_traits<ఖండం_q_t>::qk / FG_BK; // k steps per block
        స్థిరరాశి size_t          bpr     = K / fg_ఖండం_traits<ఖండం_q_t>::qk;
        స్థిరరాశి size_t          nblocks = (size_t) M * bpr;
        size_t                అడ్డువరుస_blk[S::RPL];
        తార్కిక                  అడ్డువరుస_ok[S::RPL];
        రకం_పేరు EA::pair *   a[S::RPL];
#సూచన unroll
        కొరకు (పూర్ణాంకం r = 0; r < S::RPL; ++r) {
            స్థిరరాశి పూర్ణాంకం అడ్డువరుస = m0 + r * S::SG + lane;
            అడ్డువరుస_ok[r]  = అడ్డువరుస < M;
            అడ్డువరుస_blk[r] = (size_t) (అడ్డువరుస_ok[r] ? అడ్డువరుస : 0) * bpr;
            a[r]       = (రకం_పేరు EA::pair *) &tile_a[a_పునాది + (r * S::SG + lane) * FG_BK];
        }

        స్థిరరాశి స్వయంచాలక b_ptr = sycl::చిరునామా_space_cast<sycl::access::చిరునామా_space::global_space,
                                                    sycl::access::decorated::no>(packed_b);
        స్థిరరాశి పూర్ణాంకం b_stride = Npad * S::VNNI;

        స్థిరరాశి పూర్ణాంకం kb_begin = (sg_id * nstep) / FG_KSPLIT;
        స్థిరరాశి పూర్ణాంకం kb_చివర   = ((sg_id + 1) * nstep) / FG_KSPLIT;
        కొరకు (పూర్ణాంకం kb = kb_begin; kb < kb_చివర; ++kb) {
#సూచన unroll
            కొరకు (పూర్ణాంకం r = 0; r < S::RPL; ++r) {
                ఒకవేళ (అడ్డువరుస_ok[r]) {
                    ఒకవేళ constexpr (reordered) {
                        fg_soa<ఖండం_q_t>::మూస stage<EA>((స్థిరరాశి uint8_t *) x, nblocks, అడ్డువరుస_blk[r] + kb / KPB, kb,
                                                              a[r]);
                    } లేకపోతే {
                        fg_stage_a<EA>(x + అడ్డువరుస_blk[r], kb, a[r]);
                    }
                } లేకపోతే {
#సూచన unroll
                    కొరకు (పూర్ణాంకం j = 0; j < FG_BK / 2; ++j) {
                        a[r][j] = EA::make(0.0f, 0.0f);
                    }
                }
            }
            sycl::సమూహం_barrier(sg);

#సూచన unroll
            కొరకు (పూర్ణాంకం kt = 0; kt < FG_BK / S::TK; ++kt) {
                స్థిరరాశి పూర్ణాంకం kq0 = (kb * FG_BK + kt * S::TK) / S::VNNI;
                mx::joint_మాతృక<sycl::sub_సమూహం, TB, mx::use::b, S::TK, S::TN, S::b_layout> sub_b[S::NT];
#సూచన unroll
                కొరకు (పూర్ణాంకం nt = 0; nt < S::NT; ++nt) {
                    mx::joint_మాతృక_ఎక్కించు(sg, sub_b[nt], b_ptr + (size_t) kq0 * b_stride + (b0 + nt * S::TN) * S::VNNI, b_stride);
                }
#సూచన unroll
                కొరకు (పూర్ణాంకం mt = 0; mt < S::MT; ++mt) {
                    mx::joint_మాతృక<sycl::sub_సమూహం, TA, mx::use::a, S::TM, S::TK, mx::layout::అడ్డువరుస_major> sub_a;
                    mx::joint_మాతృక_ఎక్కించు(sg, sub_a,
                        tile_a.మూస get_multi_ptr<sycl::access::decorated::no>() + a_పునాది + (mt * S::TM) * FG_BK + kt * S::TK,
                        FG_BK);
#సూచన unroll
                    కొరకు (పూర్ణాంకం nt = 0; nt < S::NT; ++nt) {
                        mx::joint_మాతృక_mad(sg, acc[mt][nt], sub_a, sub_b[nt], acc[mt][nt]);
                    }
                }
            }
            // the next step overwrites tile_a
            sycl::సమూహం_barrier(sg);
        }

#సూచన unroll
        కొరకు (పూర్ణాంకం mt = 0; mt < S::MT; ++mt) {
#సూచన unroll
            కొరకు (పూర్ణాంకం nt = 0; nt < S::NT; ++nt) {
                mx::joint_మాతృక_store(sg, acc[mt][nt],
                    tile_c.మూస get_multi_ptr<sycl::access::decorated::no>() + c_పునాది + (mt * S::TM) * S::BN + nt * S::TN,
                    S::BN, mx::layout::అడ్డువరుస_major);
            }
        }
        sycl::సమూహం_barrier(అంశం.get_సమూహం());

        // sum the K splits; consecutive lanes write consecutive rows of one dst column
        కొరకు (పూర్ణాంకం idx = అంశం.get_local_రేఖీయ_id(); idx < S::SG_అడ్డువరుసలు * S::BN; idx += S::WG_కొలత) {
            స్థిరరాశి పూర్ణాంకం r = idx % S::SG_అడ్డువరుసలు;
            స్థిరరాశి పూర్ణాంకం c = idx / S::SG_అడ్డువరుసలు;
            స్థిరరాశి పూర్ణాంకం m = m0 + r;
            స్థిరరాశి పూర్ణాంకం n = n0 + c;
            ఒకవేళ (m < M && n < n1) {
                దశాంశం కూడిక = 0.0f;
#సూచన unroll
                కొరకు (పూర్ణాంకం s = 0; s < FG_KSPLIT; ++s) {
                    కూడిక += tile_c[s * S::SG_అడ్డువరుసలు * S::BN + r * S::BN + c];
                }
                dst[(size_t) n * ldd + m] = కూడిక;
            }
        }
    }
}

మూస <రకం_పేరు S, రకం_పేరు ఖండం_q_t, తార్కిక reordered = అబద్ధం>
స్థిర శూన్య fused_dequant_gemm_launch(స్థిరరాశి శూన్య * src0, స్థిరరాశి రకం_పేరు S::tsb * packed, దశాంశం * dst, స్థిరరాశి పూర్ణాంకం M,
                                      స్థిరరాశి పూర్ణాంకం N, స్థిరరాశి పూర్ణాంకం Npad, స్థిరరాశి పూర్ణాంకం K, స్థిరరాశి పూర్ణాంకం ldd,
                                      స్థిరరాశి int64_t groups_n, స్థిరరాశి int64_t groups_m, dpct::వరుస_ptr stream) {
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<రకం_పేరు S::tsa, 1> tile_a(FG_KSPLIT * S::SG_అడ్డువరుసలు * FG_BK, cgh);
        sycl::local_accessor<దశాంశం, 1>          tile_c(FG_KSPLIT * S::SG_అడ్డువరుసలు * S::BN, cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(groups_n, groups_m * S::WG_కొలత), sycl::range<2>(1, S::WG_కొలత)),
            [=](sycl::nd_అంశం<2> అంశం) [[sycl::reqd_sub_సమూహం_కొలత(S::SG)]] {
                స్థిరరాశి పూర్ణాంకం n0 = అంశం.get_సమూహం(0) * S::BN;
                fused_dequant_gemm_tile<S, ఖండం_q_t, reordered>((స్థిరరాశి ఖండం_q_t *) src0, packed, dst, M, Npad, K,
                                                      ldd, n0, n0, N, tile_a, tile_c, అంశం);
            });
    });
}

// grouped: work-group (t, mt) is tile t of the schedule; its B columns sit at t * BN
మూస <రకం_పేరు S, రకం_పేరు ఖండం_q_t, తార్కిక reordered>
స్థిర శూన్య grouped_dequant_gemm_launch(స్థిరరాశి అక్షరం * src0_dd, స్థిరరాశి size_t expert_stride,
                                        స్థిరరాశి ggml_sycl_gg_tile * tiles_ptr, స్థిరరాశి రకం_పేరు S::tsb * packed, దశాంశం * dst,
                                        స్థిరరాశి పూర్ణాంకం M, స్థిరరాశి పూర్ణాంకం Npad, స్థిరరాశి పూర్ణాంకం K, స్థిరరాశి int64_t n_tiles,
                                        స్థిరరాశి int64_t groups_m, dpct::వరుస_ptr stream) {
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<రకం_పేరు S::tsa, 1> tile_a(FG_KSPLIT * S::SG_అడ్డువరుసలు * FG_BK, cgh);
        sycl::local_accessor<దశాంశం, 1>          tile_c(FG_KSPLIT * S::SG_అడ్డువరుసలు * S::BN, cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(n_tiles, groups_m * S::WG_కొలత), sycl::range<2>(1, S::WG_కొలత)),
            [=](sycl::nd_అంశం<2> అంశం) [[sycl::reqd_sub_సమూహం_కొలత(S::SG)]] {
                స్థిరరాశి పూర్ణాంకం               t    = అంశం.get_సమూహం(0);
                స్థిరరాశి ggml_sycl_gg_tile tile = tiles_ptr[t];
                స్థిరరాశి ఖండం_q_t *       x    = (స్థిరరాశి ఖండం_q_t *) (src0_dd + (size_t) tile.expert * expert_stride);
                fused_dequant_gemm_tile<S, ఖండం_q_t, reordered>(x, packed, dst, M, Npad, K, M, t * S::BN, tile.n0,
                                                                 tile.n1, tile_a, tile_c, అంశం);
            });
    });
}

// src1 f32 rows -> packed [K/V][n_tiles*BN][V], tile t holds its rows [n0, n1) at columns t*BN..,
// zero past n1. The column runs fastest so a sub-group writes one contiguous run.
మూస <రకం_పేరు S>
స్థిర శూన్య grouped_gemm_pack_b(స్థిరరాశి దశాంశం * y, రకం_పేరు S::tsb * packed, స్థిరరాశి ggml_sycl_gg_tile * tiles, పూర్ణాంకం Npad,
                                పూర్ణాంకం K, dpct::వరుస_ptr stream) {
    ఉపయోగించి E        = రకం_పేరు S::EB;
    constexpr పూర్ణాంకం V = S::VNNI;
    స్థిరరాశి పూర్ణాంకం kqs   = K / V;
    stream->parallel_for(sycl::range<1>((size_t) Npad * kqs), [=](sycl::id<1> id) {
        స్థిరరాశి size_t idx = id[0];
        స్థిరరాశి పూర్ణాంకం    kq  = idx / Npad;
        స్థిరరాశి పూర్ణాంకం    n   = idx - (size_t) kq * Npad;
        స్థిరరాశి ggml_sycl_gg_tile tile = tiles[n / S::BN];
        స్థిరరాశి పూర్ణాంకం    అడ్డువరుస = tile.n0 + n % S::BN;
        // one guarded load run per work-item, as a per-element select costs ~1.5% prefill
        రకం_పేరు S::tsb vals[V] = {};
        ఒకవేళ (అడ్డువరుస < tile.n1) {
            స్థిరరాశి దశాంశం * src = y + (size_t) అడ్డువరుస * K + V * kq;
#సూచన unroll
            కొరకు (పూర్ణాంకం v = 0; v < V; ++v) {
                vals[v] = E::cvt(src[v]);
            }
        }
        రకం_పేరు S::tsb * బహిర్గతం = packed + ((size_t) kq * Npad + n) * V;
#సూచన unroll
        కొరకు (పూర్ణాంకం v = 0; v < V; ++v) {
            బహిర్గతం[v] = vals[v];
        }
    });
}

// q8_0 and the k-quants take only the grouped path. The plain kernel decodes A again for every BN columns
// of a dense batch, and for these formats that costs more than the one dequantization of the library GEMM.
మూస <రకం_పేరు T> స్థిర constexpr తార్కిక fg_plain_ok() {
    తిరిగి !std::is_same_v<T, ఖండం_q8_0> && !std::is_same_v<T, ఖండం_q4_K> && !std::is_same_v<T, ఖండం_q5_K> &&
           !std::is_same_v<T, ఖండం_q6_K>;
}

మూస <రకం_పేరు T, తార్కిక R> నిర్మాణం fg_ట్యాగ్ {
    ఉపయోగించి type                     = T;
    స్థిర constexpr తార్కిక reordered = R;
};

// calls f(fg_tag<block_q_t, reordered>{}) for the weight format and layout; false if it has no A stage
మూస <రకం_పేరు T, రకం_పేరు F> స్థిర తార్కిక fg_సందర్శించు_layout(తార్కిక reordered, F && f) {
    ఒకవేళ (!reordered) {
        f(fg_ట్యాగ్<T, అబద్ధం>{});
        తిరిగి నిజం;
    }
    ఒకవేళ constexpr (fg_soa<T>::supported) {
        f(fg_ట్యాగ్<T, నిజం>{});
        తిరిగి నిజం;
    }
    తిరిగి అబద్ధం;
}

మూస <రకం_పేరు F> స్థిర తార్కిక fg_సందర్శించు_type(ggml_type type, తార్కిక reordered, F && f) {
    ఎంపిక (type) {
        స్థితి GGML_TYPE_IQ4_NL:  తిరిగి fg_సందర్శించు_layout<ఖండం_iq4_nl>(reordered, f);
        స్థితి GGML_TYPE_IQ3_S:   తిరిగి fg_సందర్శించు_layout<ఖండం_iq3_s>(reordered, f);
        స్థితి GGML_TYPE_IQ4_XS:  తిరిగి fg_సందర్శించు_layout<ఖండం_iq4_xs>(reordered, f);
        స్థితి GGML_TYPE_IQ3_XXS: తిరిగి fg_సందర్శించు_layout<ఖండం_iq3_xxs>(reordered, f);
        స్థితి GGML_TYPE_IQ2_XXS: తిరిగి fg_సందర్శించు_layout<ఖండం_iq2_xxs>(reordered, f);
        స్థితి GGML_TYPE_IQ2_XS:  తిరిగి fg_సందర్శించు_layout<ఖండం_iq2_xs>(reordered, f);
        స్థితి GGML_TYPE_IQ2_S:   తిరిగి fg_సందర్శించు_layout<ఖండం_iq2_s>(reordered, f);
        స్థితి GGML_TYPE_IQ1_S:   తిరిగి fg_సందర్శించు_layout<ఖండం_iq1_s>(reordered, f);
        స్థితి GGML_TYPE_IQ1_M:   తిరిగి fg_సందర్శించు_layout<ఖండం_iq1_m>(reordered, f);
        స్థితి GGML_TYPE_Q8_0:    తిరిగి fg_సందర్శించు_layout<ఖండం_q8_0>(reordered, f);
        స్థితి GGML_TYPE_Q4_K:    తిరిగి fg_సందర్శించు_layout<ఖండం_q4_K>(reordered, f);
        స్థితి GGML_TYPE_Q5_K:    తిరిగి fg_సందర్శించు_layout<ఖండం_q5_K>(reordered, f);
        స్థితి GGML_TYPE_Q6_K:    తిరిగి fg_సందర్శించు_layout<ఖండం_q6_K>(reordered, f);
        అప్రమేయ:                తిరిగి అబద్ధం;
    }
}

మూస <రకం_పేరు S>
స్థిర తార్కిక fg_fused_నిరంతరం(ggml_type src0_type, తార్కిక src0_reordered, స్థిరరాశి శూన్య * src0, స్థిరరాశి శూన్య * src1, ggml_type src1_type,
                         దశాంశం * dst, int64_t M, int64_t N, int64_t K, int64_t ldd, ggml_sycl_pool & pool,
                         dpct::వరుస_ptr stream) {
    స్థిరరాశి int64_t groups_n = (N + S::BN - 1) / S::BN;
    స్థిరరాశి int64_t groups_m = (M + S::SG_అడ్డువరుసలు - 1) / S::SG_అడ్డువరుసలు;
    స్థిరరాశి పూర్ణాంకం     Npad     = (పూర్ణాంకం) (groups_n * S::BN);

    // src1 is read in its own type: one pass, converted in registers only if B differs
    ggml_sycl_pool_alloc<రకం_పేరు S::tsb> packed_b(pool, (size_t) K * Npad);
    ఒకవేళ (src1_type == GGML_TYPE_F16) {
        fused_gemm_pack_b<రకం_పేరు S::EB>((స్థిరరాశి sycl::half *) src1, packed_b.get(), (పూర్ణాంకం) N, Npad, (పూర్ణాంకం) K, stream);
    } లేకపోతే ఒకవేళ (src1_type == GGML_TYPE_BF16) {
        fused_gemm_pack_b<రకం_పేరు S::EB>((స్థిరరాశి fg_bf16 *) src1, packed_b.get(), (పూర్ణాంకం) N, Npad, (పూర్ణాంకం) K, stream);
    } లేకపోతే {
        fused_gemm_pack_b<రకం_పేరు S::EB>((స్థిరరాశి దశాంశం *) src1, packed_b.get(), (పూర్ణాంకం) N, Npad, (పూర్ణాంకం) K, stream);
    }

    స్థిరరాశి రకం_పేరు S::tsb * packed = packed_b.get();
    తిరిగి fg_సందర్శించు_type(src0_type, src0_reordered, [&](స్వయంచాలక ట్యాగ్) {
        ఉపయోగించి ఖండం_q_t = రకం_పేరు decltype(ట్యాగ్)::type;
        ఒకవేళ constexpr (fg_plain_ok<ఖండం_q_t>()) {
            fused_dequant_gemm_launch<S, ఖండం_q_t, decltype(ట్యాగ్)::reordered>(src0, packed, dst, (పూర్ణాంకం) M, (పూర్ణాంకం) N,
                                                                              Npad, (పూర్ణాంకం) K, (పూర్ణాంకం) ldd, groups_n,
                                                                              groups_m, stream);
        }
    });
}

తార్కిక ggml_sycl_fused_dequant_gemm(ggml_type src0_type, తార్కిక src0_reordered, స్థిరరాశి శూన్య * src0, స్థిరరాశి శూన్య * src1, ggml_type src1_type,
                                  int32_t src1_prec, దశాంశం * dst, int64_t M, int64_t N, int64_t K, int64_t ldd,
                                  ggml_sycl_pool & pool, dpct::వరుస_ptr stream) {
    // every BN columns dequantize A again, so wide N is left to the library GEMM.
    // src0_reordered: the persistent weight reorder may have been installed on this
    // tensor by an earlier small-batch mul_mat; the A stage then has to decode the
    // SoA layout. Types without an SoA stage (fg_soa<T>::supported == false)
    // decline via fg_visit_layout and the caller falls through to the
    // reorder-aware conversion route.
    తార్కిక plain_ok = అబద్ధం;
    fg_సందర్శించు_type(src0_type, src0_reordered, [&](స్వయంచాలక ట్యాగ్) { plain_ok = fg_plain_ok<రకం_పేరు decltype(ట్యాగ్)::type>(); });
    ఒకవేళ (g_ggml_sycl_dynamic_precision == GGML_SYCL_DYNAMIC_PRECISION_F32 ||
        !ggml_sycl_xmx_gather_type_enabled(src0_type) || !plain_ok) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (src1_type != GGML_TYPE_F32 && src1_type != GGML_TYPE_F16 && src1_type != GGML_TYPE_BF16) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (!ggml_sycl_fused_dequant_gemm_shape_ok(src0_type, M, N, K, ldd)) {
        తిరిగి అబద్ధం;
    }
    స్థిరరాశి పూర్ణాంకం combo = fg_pick_combo(stream, src1_type, src1_prec);
    ఒకవేళ (combo < 0) {
        తిరిగి అబద్ధం;
    }
    తార్కిక launched = అబద్ధం;
    fg_సందర్శించు_combo(combo, [&](స్వయంచాలక s) {
        launched = fg_fused_నిరంతరం<decltype(s)>(src0_type, src0_reordered, src0, src1, src1_type, dst, M, N, K, ldd, pool, stream);
    });
    తిరిగి launched;
}

మూస <రకం_పేరు S>
స్థిర తార్కిక fg_grouped_నిరంతరం(ggml_type src0_type, తార్కిక reordered, స్థిరరాశి శూన్య * src0_పునాది, size_t expert_stride,
                           స్థిరరాశి దశాంశం * src1, దశాంశం * dst, స్థిరరాశి int64_t * expert_అడ్డువరుస_offsets, int64_t n_as, int64_t M,
                           int64_t K, std::vector<ggml_sycl_gg_tile> & tiles, ggml_sycl_pool & pool,
                           dpct::వరుస_ptr stream) {
    // the host knows every slice, so it lays out the work-groups: no search on the device
    tiles.clear();
    కొరకు (int64_t e = 0; e < n_as; ++e) {
        స్థిరరాశి int64_t end = expert_అడ్డువరుస_offsets[e + 1];
        కొరకు (int64_t n0 = expert_అడ్డువరుస_offsets[e]; n0 < end; n0 += S::BN) {
            tiles.push_back({ (int32_t) e, (int32_t) n0, (int32_t) std::min<int64_t>(n0 + S::BN, end) });
        }
    }
    స్థిరరాశి int64_t n_tiles  = tiles.size();
    స్థిరరాశి int64_t groups_m = (M + S::SG_అడ్డువరుసలు - 1) / S::SG_అడ్డువరుసలు;
    స్థిరరాశి పూర్ణాంకం     Npad     = (పూర్ణాంకం) (n_tiles * S::BN);

    స్థిరరాశి int64_t max_tiles = grouped_gemm_max_tiles(expert_అడ్డువరుస_offsets[n_as], n_as, S::BN);
    GGML_ASSERT(n_tiles <= max_tiles);
    ggml_sycl_pool_alloc<ggml_sycl_gg_tile> tiles_dev(pool, max_tiles);
    SYCL_CHECK(CHECK_TRY_ERROR(stream->జ్ఞాపకం_నకలు(tiles_dev.get(), tiles.దత్తాంశం(), n_tiles * పరిమాణం(ggml_sycl_gg_tile))));

    ggml_sycl_pool_alloc<రకం_పేరు S::tsb> packed_b(pool, (size_t) K * max_tiles * S::BN);
    grouped_gemm_pack_b<S>(src1, packed_b.get(), tiles_dev.get(), Npad, (పూర్ణాంకం) K, stream);

    స్థిరరాశి రకం_పేరు S::tsb *   packed    = packed_b.get();
    స్థిరరాశి ggml_sycl_gg_tile * tiles_ptr = tiles_dev.get();
    స్థిరరాశి అక్షరం *              src0_dd   = (స్థిరరాశి అక్షరం *) src0_పునాది;
    తిరిగి fg_సందర్శించు_type(src0_type, reordered, [&](స్వయంచాలక ట్యాగ్) {
        ఉపయోగించి T = decltype(ట్యాగ్);
        grouped_dequant_gemm_launch<S, రకం_పేరు T::type, T::reordered>(src0_dd, expert_stride, tiles_ptr, packed, dst,
                                                                       (పూర్ణాంకం) M, Npad, (పూర్ణాంకం) K, n_tiles, groups_m,
                                                                       stream);
    });
}

తార్కిక ggml_sycl_grouped_dequant_gemm(ggml_type src0_type, తార్కిక reordered, స్థిరరాశి శూన్య * src0_పునాది, size_t expert_stride,
                                    స్థిరరాశి దశాంశం * src1, int32_t src1_prec, దశాంశం * dst,
                                    స్థిరరాశి int64_t * expert_అడ్డువరుస_offsets, int64_t n_as, int64_t M, int64_t K,
                                    int64_t మొత్తం_అడ్డువరుసలు, std::vector<ggml_sycl_gg_tile> & tiles,
                                    ggml_sycl_pool & pool, dpct::వరుస_ptr stream) {
    int64_t n_చురుకైన = 0;
    కొరకు (int64_t e = 0; e < n_as; ++e) {
        n_చురుకైన += expert_అడ్డువరుస_offsets[e + 1] > expert_అడ్డువరుస_offsets[e];
    }
    ఒకవేళ (g_ggml_sycl_dynamic_precision == GGML_SYCL_DYNAMIC_PRECISION_F32 ||
        !ggml_sycl_xmx_gather_type_enabled(src0_type) || !fg_సందర్శించు_type(src0_type, reordered, [](స్వయంచాలక) {})) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (!ggml_sycl_grouped_dequant_gemm_shape_ok(src0_type, M, K, మొత్తం_అడ్డువరుసలు, n_చురుకైన)) {
        తిరిగి అబద్ధం;
    }
    స్థిరరాశి పూర్ణాంకం combo = fg_pick_combo(stream, GGML_TYPE_F32, src1_prec);
    ఒకవేళ (combo < 0) {
        తిరిగి అబద్ధం;
    }
    తార్కిక launched = అబద్ధం;
    fg_సందర్శించు_combo(combo, [&](స్వయంచాలక s) {
        launched = fg_grouped_నిరంతరం<decltype(s)>(src0_type, reordered, src0_పునాది, expert_stride, src1, dst,
                                               expert_అడ్డువరుస_offsets, n_as, M, K, tiles, pool, stream);
    });
    తిరిగి launched;
}
