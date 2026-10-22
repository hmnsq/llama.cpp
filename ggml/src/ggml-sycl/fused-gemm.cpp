#समावेश "fused-gemm.hpp"

#समावेश <sycl/ext/oneapi/matrix/matrix.hpp>

#समावेश <algorithm>
#समावेश <string>
#समावेश <tuple>
#समावेश <mutex>
#समावेश <set>
#समावेश <unordered_map>

namespace mx = sycl::ext::oneapi::experimental::matrix;

// FG_ / fg_ is short for fused GEMM: the weights are dequantized inside the GEMM, into the XMX tiles.

// A k step is one 32-value weight sub-block; iq3_s and the other superblock formats split their
// superblock into steps of this width. The sub-groups of a work-group each walk their own K range
// and are summed at the end.
अटल constexpr पूर्णांक FG_BK     = QK4_NL;
अटल constexpr पूर्णांक FG_KSPLIT = 4;

// Element traits of one joint_matrix operand type. The A stage and the B pack compute in f32 and
// convert once, in registers, when they write the element, so any type costs the same one pass.
//   store: storage in SLM (A) and in the packed B buffer
//   mtype: matrix_type in matrix_combinations
//   mode:  GGML_SYCL_DYNAMIC_PRECISION value that selects this type
//   src:   ggml type that needs no conversion into this type (GGML_TYPE_COUNT: none)
//   slow:  XMX throughput class, 0 is fastest. f16 and bf16 share the DPAS rate; tf32 does half the
//          K per instruction. B60, Qwen3-30B-A3B pp512: f16 1108, bf16 1000, tf32 751 t/s
ढाँचा <typename T> काष्ठा fg_elem;

ढाँचा <> काष्ठा fg_elem<sycl::half> {
    using store = sycl::half;
    using pair  = sycl::half2;
    अटल constexpr mx::matrix_type mtype = mx::matrix_type::fp16;
    अटल constexpr पूर्णांक             mode  = GGML_SYCL_DYNAMIC_PRECISION_F16;
    अटल constexpr ggml_type       src   = GGML_TYPE_F16;
    अटल constexpr पूर्णांक             mant  = 10;
    अटल constexpr पूर्णांक             slow  = 0;
    अटल store cvt(भग्न x) { वापस (store) x; }
    अटल pair make(भग्न x, भग्न y) { वापस pair((store) x, (store) y); }
};

// tf32 rounds to nearest even with plain bit ops: round_to_tf32 needs a SPIR-V extension that the
// DG2 AOT target rejects
अटल अंतरभूत uint32_t fg_round_bits(भग्न x, पूर्णांक drop) {
    स्थिर uint32_t u = sycl::bit_cast<uint32_t>(x);
    अगर ((u & 0x7f800000u) == 0x7f800000u) {
        वापस (u & 0x7fffffu) ? u | (1u << drop) : u; // nan stays nan
    }
    वापस u + ((1u << (drop - 1)) - 1) + ((u >> drop) & 1);
}

काष्ठा alignas(4) fg_bf16x2 {
    sycl::ext::oneapi::bfloat16 x, y;
};

ढाँचा <> काष्ठा fg_elem<sycl::ext::oneapi::bfloat16> {
    using store = sycl::ext::oneapi::bfloat16;
    using pair  = fg_bf16x2;
    अटल constexpr mx::matrix_type mtype = mx::matrix_type::bf16;
    अटल constexpr पूर्णांक             mode  = GGML_SYCL_DYNAMIC_PRECISION_BF16;
    अटल constexpr ggml_type       src   = GGML_TYPE_BF16;
    अटल constexpr पूर्णांक             mant  = 7;
    अटल constexpr पूर्णांक             slow  = 0;
    अटल store cvt(भग्न x) { वापस store(x); }
    अटल pair make(भग्न x, भग्न y) { वापस { cvt(x), cvt(y) }; }
};

// tf32 keeps f32 range and f16 mantissa, in f32 storage
ढाँचा <> काष्ठा fg_elem<mx::precision::tf32> {
    using store = भग्न;
    using pair  = sycl::float2;
    अटल constexpr mx::matrix_type mtype = mx::matrix_type::tf32;
    अटल constexpr पूर्णांक             mode  = GGML_SYCL_DYNAMIC_PRECISION_TF32;
    अटल constexpr ggml_type       src   = GGML_TYPE_गिनती;
    अटल constexpr पूर्णांक             mant  = 10;
    अटल constexpr पूर्णांक             slow  = 1;
    अटल store cvt(भग्न x) { वापस sycl::bit_cast<भग्न>(fg_round_bits(x, 13) & ~0x1fffu); }
    अटल pair make(भग्न x, भग्न y) { वापस pair(cvt(x), cvt(y)); }
};

// One joint_matrix combination (A type, B type, TM x TN x TK, sub-group size; C and D are f32) and
// the tiling built on it. A sub-group owns SG_ROWS rows of A (at least 16) and BN (at least 32)
// columns of B. A and B may differ: the device lists the pairs it supports.
ढाँचा <typename TA, typename TB, पूर्णांक TM_, पूर्णांक TN_, पूर्णांक TK_, पूर्णांक SG_> काष्ठा fg_combo {
    using ta  = TA;
    using tb  = TB;
    using EA  = fg_elem<TA>;
    using EB  = fg_elem<TB>;
    using tsa = typename EA::store;
    using tsb = typename EB::store;
    अटल constexpr पूर्णांक TM = TM_;
    अटल constexpr पूर्णांक TN = TN_;
    अटल constexpr पूर्णांक TK = TK_;
    अटल constexpr पूर्णांक SG = SG_;
    अटल constexpr पूर्णांक VNNI    = 4 / माप(tsb);  // K rows of B packed in one 32-bit word
    अटल constexpr पूर्णांक SG_ROWS = TM > 16 ? TM : 16;
    अटल constexpr पूर्णांक RPL     = SG_ROWS / SG;     // A rows one lane decodes per k step
    अटल constexpr पूर्णांक MT      = SG_ROWS / TM;
    अटल constexpr पूर्णांक BN      = TN > 32 ? TN : 32;
    अटल constexpr पूर्णांक NT      = BN / TN;
    अटल constexpr पूर्णांक WG_आकार = FG_KSPLIT * SG;
    अटल constexpr mx::layout b_layout = VNNI == 1 ? mx::layout::row_major : mx::layout::ext_intel_packed;
    // a 64-wide N is mostly padding here and a 32x64 f32 accumulator needs 128 registers per lane,
    // so it spills: 13x slower on B60
    अटल constexpr bool efficient = TN <= 32;
    static_assert(SG_ROWS % SG == 0 && SG_ROWS % TM == 0 && BN % TN == 0 && FG_BK % TK == 0, "bad tile");
    static_assert(BN <= GGML_SYCL_FG_MAX_N, "header gate must cover the tile width");
};

using fg_half = sycl::half;
using fg_bf16 = sycl::ext::oneapi::bfloat16;
using fg_tf32 = mx::precision::tf32;

// One bit of GGML_SYCL_XMX_GATHER_SHAPES per combination. Only combinations some device lists in
// matrix_combinations are built (appendix of sycl_ext_oneapi_matrix and the runtime's own list).
ढाँचा <typename F> अटल व्योम fg_visit_combo(पूर्णांक idx, F && f);
अटल constexpr पूर्णांक FG_N_COMBOS = 8;

// A spir64_gen AOT build (GGML_SYCL_XMX_AOT_SG) drops the combinations of the other sub-group size
// entirely: ocloc rejects even an empty kernel that asks for a sub-group size it lacks.
ढाँचा <पूर्णांक SG> अटल constexpr bool fg_listed() {
#अगर defined(GGML_SYCL_XMX_AOT_SG)
    वापस SG == GGML_SYCL_XMX_AOT_SG;
#अन्यथा
    वापस true;
#पूर्ण_अगर
}

ढाँचा <typename S, typename F> अटल व्योम fg_call_combo(F && f) {
    अगर constexpr (fg_listed<S::SG>()) {
        f(S{});
    }
}

ढाँचा <typename F> अटल व्योम fg_visit_combo(पूर्णांक idx, F && f) {
    चयन (idx) {
        हाल 0: fg_call_combo<fg_combo<fg_half, fg_half, 8, 16, 16, 16>>(f);  अवरोध; // Xe2, Xe3, Xe-HPC
        हाल 1: fg_call_combo<fg_combo<fg_half, fg_half, 16, 16, 16, 16>>(f); अवरोध; // Xe2, Xe3, Xe-HPC
        हाल 2: fg_call_combo<fg_combo<fg_half, fg_half, 32, 64, 16, 16>>(f); अवरोध; // Xe2, Xe3, Xe-HPC
        हाल 3: fg_call_combo<fg_combo<fg_half, fg_half, 32, 64, 32, 16>>(f); अवरोध; // Xe2, Xe3, Xe-HPC
        हाल 4: fg_call_combo<fg_combo<fg_half, fg_half, 8, 8, 16, 8>>(f);    अवरोध; // Xe-HPG (Arc A), ARL-H
        हाल 5: fg_call_combo<fg_combo<fg_tf32, fg_tf32, 8, 16, 8, 16>>(f);   अवरोध; // Xe2, Xe3, Xe-HPC
        हाल 6: fg_call_combo<fg_combo<fg_bf16, fg_bf16, 8, 16, 16, 16>>(f);  अवरोध; // Xe2, Xe3, Xe-HPC
        हाल 7: fg_call_combo<fg_combo<fg_bf16, fg_bf16, 8, 8, 16, 8>>(f);    अवरोध; // Xe-HPG (Arc A), ARL-H
        शेष: GGML_ABORT("bad XMX combination %d", idx);
    }
}

// AOT with -fsycl-targets=intel_gpu_*: compile each tile body only for targets with its sub-group
// size, since IGC fails on the other ones. A JIT build keeps them all, but each combination lands in
// its own device image (joint_matrix is an optional kernel feature) and only a combination the
// device reports is launched, so the runtime never asks IGC for the others.
#अगर defined(__SYCL_DEVICE_ONLY__)
#    अगर __SYCL_TARGET_INTEL_GPU_ACM_G10__ || __SYCL_TARGET_INTEL_GPU_ACM_G11__ || __SYCL_TARGET_INTEL_GPU_ACM_G12__ || \
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
#पूर्ण_अगर

ढाँचा <पूर्णांक SG> अटल constexpr bool fg_built() {
#अगर defined(FG_AOT_SG)
    वापस SG == FG_AOT_SG;
#अन्यथा
    वापस true;
#पूर्ण_अगर
}

// Upper bound on the tile count when total_rows rows are routed to n_as experts: the worst case gives
// each expert one row and fills whole tiles with the rest. The bound depends only on the shape, not
// on the routing, so the pool reuses one buffer every ubatch instead of keeping one per size seen.
अटल constexpr int64_t grouped_gemm_max_tiles(int64_t योग_rows, int64_t n_as, int64_t BN) {
    वापस योग_rows <= n_as ? योग_rows : n_as + (योग_rows - n_as) / BN;
}
// Tiles do not cross experts, so the bound is not ceil(total_rows / BN): 34 rows over 2 experts with
// BN = 16 split 17 + 17 need 2 + 2 tiles, where the ceil gives 3.
static_assert(grouped_gemm_max_tiles(34, 2, 16) == 4);

// the device lists S with an f32 accumulator and output
ढाँचा <typename S> अटल bool fg_device_has_combo(स्थिर std::vector<mx::combination> & combinations) {
    क्रम (स्थिर स्वतः & c : combinations) {
        अगर (c.atype == S::EA::mtype && c.btype == S::EB::mtype && c.ctype == mx::matrix_type::fp32 &&
            c.dtype == mx::matrix_type::fp32 &&
            (c.max_msize >= (माप_प्रकार) S::TM || c.msize == (माप_प्रकार) S::TM) &&
            (c.max_nsize >= (माप_प्रकार) S::TN || c.nsize == (माप_प्रकार) S::TN) &&
            (c.max_ksize >= (माप_प्रकार) S::TK || c.ksize == (माप_प्रकार) S::TK)) {
            वापस true;
        }
    }
    वापस false;
}

ढाँचा <typename T> अटल स्थिर अक्षर * fg_type_name() {
    वापस std::is_same_v<T, fg_half> ? "f16" : std::is_same_v<T, fg_bf16> ? "bf16" : "tf32";
}

अटल std::string fg_combo_name(पूर्णांक idx) {
    std::string name;
    fg_visit_combo(idx, [&](स्वतः s) {
        using S = decltype(s);
        name = std::string(fg_type_name<typename S::ta>()) + "x" + fg_type_name<typename S::tb>() + " " +
               std::to_string(S::TM) + "x" + std::to_string(S::TN) + "x" + std::to_string(S::TK) + " sg" +
               std::to_string(S::SG);
    });
    वापस name;
}

// Combinations this build has kernels for and the device lists, one bit each. Cached per device:
// on a mixed box the first caller's verdict is not the others'.
अटल पूर्णांक fg_device_combos(स्थिर sycl::device & dev) {
    अटल std::mutex                            mtx;
    अटल std::unordered_map<sycl::device, पूर्णांक> known;
    std::lock_guard<std::mutex>                  lock(mtx);
    स्थिर स्वतः                                   it = known.find(dev);
    अगर (it != known.end()) {
        वापस it->second;
    }
    पूर्णांक available = 0;
    प्रयास {
        स्थिर स्वतः combinations = dev.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>();
        स्थिर स्वतः sg_sizes     = dev.get_info<sycl::info::device::sub_group_sizes>();
        क्रम (पूर्णांक idx = 0; idx < FG_N_COMBOS; ++idx) {
            fg_visit_combo(idx, [&](स्वतः s) {
                using S = decltype(s);
                स्थिर bool sg_ok = std::find(sg_sizes.begin(), sg_sizes.end(), (माप_प्रकार) S::SG) != sg_sizes.end();
                अगर (sg_ok && fg_device_has_combo<S>(combinations)) {
                    available |= 1 << idx;
                }
            });
        }
    } पकड़ो (स्थिर sycl::exception &) {
        available = 0;
    }
    GGML_LOG_INFO("%s: %s: XMX dequant-GEMM combinations available 0x%x, allowed 0x%x\n", __func__,
                  dev.get_info<sycl::info::device::name>().c_str(), available, g_ggml_sycl_xmx_gather_shapes);
    known.emplace(dev, available);
    वापस available;
}

// Rank of combination S for a src1 of type src1_type, lower is better. Order:
//  1. throughput: a tile that does not spill, then the fastest type class of A and B
//  2. B type equal to the src1 type, so the pack is a plain copy
//  3. B at least as precise as f16
//  4. the device's native DPAS tile (8 x SG x 32 bytes of K), then the largest M x K
// A costs nothing to convert: the A stage emits any type at the same cost.
ढाँचा <typename S> अटल int64_t fg_rank(ggml_type src1_type) {
    स्थिर int64_t spills  = !S::efficient;
    स्थिर int64_t slow    = std::max(S::EA::slow, S::EB::slow);
    स्थिर int64_t convert = S::EB::src != src1_type;
    स्थिर int64_t lossy   = S::EB::mant < 10;
    स्थिर int64_t foreign = !(S::TM == 8 && S::TN == S::SG);
    स्थिर int64_t mk      = 1024 - S::TM * S::TK;
    वापस ((((spills * 2 + slow) * 2 + convert) * 2 + lossy) * 2 + foreign) * 2048 + mk;
}

// whether XMX operands of type mode (a GGML_SYCL_DYNAMIC_PRECISION value) meet the src1 request
// [TAG_GGML_PREC]. f16 lacks the f32 range that BF16 and F32 ask for; an F32 request goes only as far
// down as GGML_SYCL_DYNAMIC_REQUIRED_PRECISION allows.
अटल bool fg_mode_meets(पूर्णांक mode, int32_t src1_prec) {
    अगर (src1_prec == GGML_PREC_UNDEFINED || src1_prec >= GGML_PREC_F16) {
        वापस true;
    }
    अगर (src1_prec >= GGML_PREC_BF16) {
        वापस mode != GGML_SYCL_DYNAMIC_PRECISION_F16;
    }
    चयन (g_ggml_sycl_dynamic_required_precision) {
        हाल GGML_SYCL_DYNAMIC_PRECISION_TF32: वापस mode == GGML_SYCL_DYNAMIC_PRECISION_TF32;
        हाल GGML_SYCL_DYNAMIC_PRECISION_BF16: वापस mode != GGML_SYCL_DYNAMIC_PRECISION_F16;
        शेष:                               वापस false;
    }
}

// Best allowed combination for this call, or -1 if none. The type is GGML_SYCL_DYNAMIC_PRECISION if it
// meets the src1 request; if not, bf16 then tf32 for a BF16 request (fastest first) and tf32 then bf16
// for an F32 request (most mantissa first).
अटल पूर्णांक fg_pick_combo(dpct::queue_ptr stream, ggml_type src1_type, int32_t src1_prec) {
    स्थिर sycl::device dev     = stream->get_device();
    स्थिर पूर्णांक          allowed = fg_device_combos(dev) & g_ggml_sycl_xmx_gather_shapes;
    स्थिर bool         f32_req = src1_prec != GGML_PREC_UNDEFINED && src1_prec < GGML_PREC_BF16;
    स्थिर पूर्णांक          modes[] = {
        g_ggml_sycl_dynamic_precision,
        f32_req ? GGML_SYCL_DYNAMIC_PRECISION_TF32 : GGML_SYCL_DYNAMIC_PRECISION_BF16,
        f32_req ? GGML_SYCL_DYNAMIC_PRECISION_BF16 : GGML_SYCL_DYNAMIC_PRECISION_TF32,
    };
    पूर्णांक     best      = -1;
    int64_t best_rank = 0;
    क्रम (पूर्णांक i = 0; i < 3 && best < 0; ++i) {
        स्थिर पूर्णांक mode = modes[i];
        अगर (!fg_mode_meets(mode, src1_prec)) {
            जारी;
        }
        क्रम (पूर्णांक idx = 0; idx < FG_N_COMBOS; ++idx) {
            अगर (!(allowed & (1 << idx))) {
                जारी;
            }
            fg_visit_combo(idx, [&](स्वतः s) {
                using S = decltype(s);
                अगर (S::EA::mode != mode || S::EB::mode != mode) {
                    वापस;
                }
                स्थिर int64_t rank = fg_rank<S>(src1_type);
                अगर (best < 0 || rank < best_rank) {
                    best      = idx;
                    best_rank = rank;
                }
            });
        }
    }
    // log each distinct decision once
    अटल std::mutex                                      mtx;
    अटल std::set<std::tuple<माप_प्रकार, पूर्णांक, int32_t, पूर्णांक>> seen;
    std::lock_guard<std::mutex>                            lock(mtx);
    अगर (seen.emplace(std::hash<sycl::device>{}(dev), (पूर्णांक) src1_type, src1_prec, best).second) {
        GGML_LOG_INFO("%s: src1 %s, src1 prec %d -> %s\n", __func__, ggml_type_name(src1_type), src1_prec,
                      best >= 0 ? fg_combo_name(best).c_str() : "none (library GEMM)");
    }
    वापस best;
}

// src1 [N][K] -> packed [K/V][Npad][V] so B tiles load straight from global memory
ढाँचा <typename E, typename T_src>
अटल व्योम fused_gemm_pack_b(स्थिर T_src * y, typename E::store * packed, पूर्णांक N, पूर्णांक Npad, पूर्णांक K,
                              dpct::queue_ptr stream) {
    constexpr पूर्णांक V   = 4 / माप(typename E::store);
    स्थिर पूर्णांक     kqs = K / V;
    stream->parallel_for(sycl::range<1>((माप_प्रकार) Npad * kqs), [=](sycl::id<1> id) {
        स्थिर पूर्णांक idx = id[0];
        स्थिर पूर्णांक n   = idx / kqs;
        स्थिर पूर्णांक kq  = idx - n * kqs;
        typename E::store vals[V] = {};
        अगर (n < N) {
            स्थिर T_src * src = y + (माप_प्रकार) n * K + V * kq;
#आशय unroll
            क्रम (पूर्णांक v = 0; v < V; ++v) {
                vals[v] = E::cvt((भग्न) src[v]);
            }
        }
        typename E::store * out = packed + ((माप_प्रकार) kq * Npad + n) * V;
#आशय unroll
        क्रम (पूर्णांक v = 0; v < V; ++v) {
            out[v] = vals[v];
        }
    });
}

// A stage: one lane owns one row and decodes FG_BK values of it per k step, with every scale
// folded into the value so the mad below sees plain A elements. One overload per weight format.
ढाँचा <typename E>
अटल __dpct_inline__ व्योम fg_stage_a(स्थिर block_iq4_nl * __restrict__ xrow, स्थिर पूर्णांक kb, typename E::pair * a) {
    स्थिर block_iq4_nl blk = xrow[kb];
    स्थिर भग्न        d   = (भग्न) blk.d;
#आशय unroll
    क्रम (पूर्णांक j = 0; j < QK4_NL / 2; j += 2) {
        स्थिर uint8_t q0 = blk.qs[j];
        स्थिर uint8_t q1 = blk.qs[j + 1];
        a[j / 2]     = E::make(d * kvalues_iq4nl[q0 & 0xf], d * kvalues_iq4nl[q1 & 0xf]);
        a[j / 2 + 8] = E::make(d * kvalues_iq4nl[q0 >> 4], d * kvalues_iq4nl[q1 >> 4]);
    }
}

// iq3_s: k step kb is sub-block kb % 8 of superblock kb / 8. The superblock is 110 bytes, so read
// only the fields of that sub-block instead of copying the block. Same decode as
// dequantize_block_iq3_s: grid entries are taken as dwords and the sign bit is a plain shift.
ढाँचा <typename E>
अटल __dpct_inline__ व्योम fg_stage_a(स्थिर block_iq3_s * __restrict__ xrow, स्थिर पूर्णांक kb, typename E::pair * a) {
    static_assert(QK_K == 256, "the iq3_s A stage assumes 8 sub-blocks per superblock");
    स्थिर block_iq3_s * blk = xrow + kb / (QK_K / 32);
    स्थिर पूर्णांक           ib8 = kb % (QK_K / 32);
    स्थिर uint8_t *     qs  = blk->qs + 8 * ib8;
    स्थिर पूर्णांक           qh  = blk->qh[ib8];
    स्थिर भग्न         d   = (भग्न) blk->d * (1 + 2 * ((blk->scales[ib8 / 2] >> (4 * (ib8 % 2))) & 0xf));
#आशय unroll
    क्रम (पूर्णांक il = 0; il < 4; ++il) {
        स्थिर uint32_t grid1 = iq3s_grid[qs[2 * il + 0] | ((qh << (8 - 2 * il)) & 256)];
        स्थिर uint32_t grid2 = iq3s_grid[qs[2 * il + 1] | ((qh << (7 - 2 * il)) & 256)];
        स्थिर पूर्णांक      signs = blk->signs[4 * ib8 + il];
#आशय unroll
        क्रम (पूर्णांक j = 0; j < 2; ++j) {
            स्थिर भग्न g1a = (भग्न) ((grid1 >> (16 * j + 0)) & 0xff);
            स्थिर भग्न g1b = (भग्न) ((grid1 >> (16 * j + 8)) & 0xff);
            स्थिर भग्न g2a = (भग्न) ((grid2 >> (16 * j + 0)) & 0xff);
            स्थिर भग्न g2b = (भग्न) ((grid2 >> (16 * j + 8)) & 0xff);
            स्थिर पूर्णांक   s   = 2 * j;
            a[4 * il + j]     = E::make(d * ((signs & (1 << (s + 0))) ? -g1a : g1a),
                                        d * ((signs & (1 << (s + 1))) ? -g1b : g1b));
            a[4 * il + j + 2] = E::make(d * ((signs & (1 << (s + 4))) ? -g2a : g2a),
                                        d * ((signs & (1 << (s + 5))) ? -g2b : g2b));
        }
    }
}

// values per stored block, so a row of K values is K/qk blocks
ढाँचा <typename block_q_t> काष्ठा fg_block_traits;
ढाँचा <> काष्ठा fg_block_traits<block_iq4_nl> { अटल constexpr पूर्णांक qk = QK4_NL; };
ढाँचा <> काष्ठा fg_block_traits<block_iq3_s>  { अटल constexpr पूर्णांक qk = QK_K; };
ढाँचा <> काष्ठा fg_block_traits<block_iq3_xxs> { अटल constexpr पूर्णांक qk = QK_K; };
ढाँचा <> काष्ठा fg_block_traits<block_iq4_xs>  { अटल constexpr पूर्णांक qk = QK_K; };
ढाँचा <> काष्ठा fg_block_traits<block_iq2_xxs> { अटल constexpr पूर्णांक qk = QK_K; };
ढाँचा <> काष्ठा fg_block_traits<block_iq2_xs>  { अटल constexpr पूर्णांक qk = QK_K; };
ढाँचा <> काष्ठा fg_block_traits<block_iq2_s>   { अटल constexpr पूर्णांक qk = QK_K; };
ढाँचा <> काष्ठा fg_block_traits<block_iq1_s>   { अटल constexpr पूर्णांक qk = QK_K; };
ढाँचा <> काष्ठा fg_block_traits<block_iq1_m>   { अटल constexpr पूर्णांक qk = QK_K; };
ढाँचा <> काष्ठा fg_block_traits<block_q8_0>    { अटल constexpr पूर्णांक qk = QK8_0; };
ढाँचा <> काष्ठा fg_block_traits<block_q4_K>    { अटल constexpr पूर्णांक qk = QK_K; };
ढाँचा <> काष्ठा fg_block_traits<block_q5_K>    { अटल constexpr पूर्णांक qk = QK_K; };
ढाँचा <> काष्ठा fg_block_traits<block_q6_K>    { अटल constexpr पूर्णांक qk = QK_K; };

// The A stages below are the dequantize_block_iq* kernels rewritten for one k step. There a
// work-item handled one quarter (il) of one 32-wide sub-block (ib); here one lane produces the
// whole step, so il becomes a loop and ib is kb inside the superblock. Each quarter yields 8
// consecutive values, i.e. 4 pairs at a[4*il], so nothing larger than 8 floats is ever live.
#घोषणा FG_SUPERBLOCK(T)                                                         \
    static_assert(QK_K == 256, "the " #T " A stage assumes 8 sub-blocks per superblock"); \
    स्थिर T * blk = xrow + kb / (QK_K / 32);                                     \
    स्थिर पूर्णांक ib  = kb % (QK_K / 32)

ढाँचा <typename E>
अटल __dpct_inline__ व्योम fg_pack_quarter(स्थिर भग्न * __restrict__ t, typename E::pair * a, पूर्णांक il) {
#आशय unroll
    क्रम (पूर्णांक j = 0; j < 4; ++j) {
        a[4 * il + j] = E::make(t[2 * j], t[2 * j + 1]);
    }
}

ढाँचा <typename E>
अटल __dpct_inline__ व्योम fg_stage_a(स्थिर block_iq4_xs * __restrict__ xrow, स्थिर पूर्णांक kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_iq4_xs);
    // low nibbles fill the first half of the step, high nibbles the second, so the two halves
    // land at a[0..7] and a[8..15] and no quarter loop is needed
    स्थिर भग्न d = (भग्न) blk->d *
        ((((blk->scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) | (((blk->scales_h >> (2 * ib)) & 3) << 4)) - 32);
    स्थिर uint8_t * q4 = blk->qs + 16 * ib;
#आशय unroll
    क्रम (पूर्णांक j = 0; j < 8; ++j) {
        a[j]     = E::make(d * kvalues_iq4nl[q4[2 * j] & 0xf], d * kvalues_iq4nl[q4[2 * j + 1] & 0xf]);
        a[8 + j] = E::make(d * kvalues_iq4nl[q4[2 * j] >> 4],  d * kvalues_iq4nl[q4[2 * j + 1] >> 4]);
    }
}

ढाँचा <typename E>
अटल __dpct_inline__ व्योम fg_stage_a(स्थिर block_iq3_xxs * __restrict__ xrow, स्थिर पूर्णांक kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_iq3_xxs);
    स्थिर uint8_t *  q3    = blk->qs + 8 * ib;
    स्थिर uint16_t * gas   = (स्थिर uint16_t *) (blk->qs + QK_K / 4) + 2 * ib;
    स्थिर uint32_t   aux32 = gas[0] | (gas[1] << 16);
    स्थिर भग्न      d     = (भग्न) blk->d * (0.5f + (aux32 >> 28)) * 0.5f;
#आशय unroll
    क्रम (पूर्णांक il = 0; il < 4; ++il) {
        स्थिर uint8_t * grid1 = (स्थिर uint8_t *) (iq3xxs_grid + q3[2 * il + 0]);
        स्थिर uint8_t * grid2 = (स्थिर uint8_t *) (iq3xxs_grid + q3[2 * il + 1]);
        स्थिर uint8_t   signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];
        भग्न t[8];
#आशय unroll
        क्रम (पूर्णांक j = 0; j < 4; ++j) {
            t[j + 0] = d * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f);
            t[j + 4] = d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

ढाँचा <typename E>
अटल __dpct_inline__ व्योम fg_stage_a(स्थिर block_iq2_xxs * __restrict__ xrow, स्थिर पूर्णांक kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_iq2_xxs);
    स्थिर uint16_t * q2    = blk->qs + 4 * ib;
    स्थिर uint8_t *  aux8  = (स्थिर uint8_t *) q2;
    स्थिर uint32_t   aux32 = q2[2] | (q2[3] << 16);
    स्थिर भग्न      d     = (भग्न) blk->d * (0.5f + (aux32 >> 28)) * 0.25f;
#आशय unroll
    क्रम (पूर्णांक il = 0; il < 4; ++il) {
        स्थिर uint8_t * grid  = (स्थिर uint8_t *) (iq2xxs_grid + aux8[il]);
        स्थिर uint8_t   signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];
        भग्न t[8];
#आशय unroll
        क्रम (पूर्णांक j = 0; j < 8; ++j) {
            t[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

ढाँचा <typename E>
अटल __dpct_inline__ व्योम fg_stage_a(स्थिर block_iq2_xs * __restrict__ xrow, स्थिर पूर्णांक kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_iq2_xs);
    स्थिर uint16_t * q2 = blk->qs + 4 * ib;
#आशय unroll
    क्रम (पूर्णांक il = 0; il < 4; ++il) {
        स्थिर uint8_t * grid  = (स्थिर uint8_t *) (iq2xs_grid + (q2[il] & 511));
        स्थिर भग्न     d     = (भग्न) blk->d * (0.5f + ((blk->scales[ib] >> (4 * (il / 2))) & 0xf)) * 0.25f;
        स्थिर uint8_t   signs = ksigns_iq2xs[q2[il] >> 9];
        भग्न t[8];
#आशय unroll
        क्रम (पूर्णांक j = 0; j < 8; ++j) {
            t[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

ढाँचा <typename E>
अटल __dpct_inline__ व्योम fg_stage_a(स्थिर block_iq2_s * __restrict__ xrow, स्थिर पूर्णांक kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_iq2_s);
#आशय unroll
    क्रम (पूर्णांक il = 0; il < 4; ++il) {
        स्थिर uint8_t * grid =
            (स्थिर uint8_t *) (iq2s_grid + (blk->qs[4 * ib + il] | ((blk->qh[ib] << (8 - 2 * il)) & 0x300)));
        स्थिर भग्न   d     = (भग्न) blk->d * (0.5f + ((blk->scales[ib] >> (4 * (il / 2))) & 0xf)) * 0.25f;
        स्थिर uint8_t signs = blk->qs[QK_K / 8 + 4 * ib + il];
        भग्न t[8];
#आशय unroll
        क्रम (पूर्णांक j = 0; j < 8; ++j) {
            t[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

ढाँचा <typename E>
अटल __dpct_inline__ व्योम fg_stage_a(स्थिर block_iq1_s * __restrict__ xrow, स्थिर पूर्णांक kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_iq1_s);
    स्थिर भग्न delta = blk->qh[ib] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
    स्थिर भग्न d     = (भग्न) blk->d * (2 * ((blk->qh[ib] >> 12) & 7) + 1);
#आशय unroll
    क्रम (पूर्णांक il = 0; il < 4; ++il) {
        uint32_t       grid32[2];
        स्थिर int8_t * q = (स्थिर int8_t *) grid32;
        grid32[0] = iq1s_grid_gpu[blk->qs[4 * ib + il] | (((blk->qh[ib] >> (3 * il)) & 7) << 8)];
        grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
        grid32[0] &= 0x0f0f0f0f;
        भग्न t[8];
#आशय unroll
        क्रम (पूर्णांक j = 0; j < 8; ++j) {
            t[j] = d * (q[j] + delta);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

ढाँचा <typename E>
अटल __dpct_inline__ व्योम fg_stage_a(स्थिर block_iq1_m * __restrict__ xrow, स्थिर पूर्णांक kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_iq1_m);
    स्थिर uint16_t * sc = (स्थिर uint16_t *) blk->scales;
    iq1m_scale_t     scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
#आशय unroll
    क्रम (पूर्णांक il = 0; il < 4; ++il) {
        स्थिर पूर्णांक   ib16  = 2 * ib + il / 2;
        स्थिर भग्न d     = (भग्न) scale.f16 * (2 * ((sc[ib16 / 4] >> (3 * (ib16 % 4))) & 0x7) + 1);
        स्थिर भग्न delta = blk->qh[2 * ib + il / 2] & (0x08 << (4 * (il % 2))) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
        uint32_t       grid32[2];
        स्थिर int8_t * q = (स्थिर int8_t *) grid32;
        grid32[0] = iq1s_grid_gpu[blk->qs[4 * ib + il] |
                                  (((blk->qh[2 * ib + il / 2] >> (4 * (il % 2))) & 7) << 8)];
        grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
        grid32[0] &= 0x0f0f0f0f;
        भग्न t[8];
#आशय unroll
        क्रम (पूर्णांक j = 0; j < 8; ++j) {
            t[j] = d * (q[j] + delta);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}


// q8_0 and the k-quants. Each decode takes the fields of one 32-value sub-block, so the canonical
// layout and the reorder (SoA) layout of reorder_qw() share it and differ only in where the fields
// live, as in dequantize.hpp.
ढाँचा <typename E>
अटल __dpct_inline__ व्योम fg_decode_q8_0(स्थिर int8_t * __restrict__ qs, स्थिर भग्न d, typename E::pair * a) {
#आशय unroll
    क्रम (पूर्णांक j = 0; j < QK8_0 / 2; ++j) {
        a[j] = E::make(d * qs[2 * j], d * qs[2 * j + 1]);
    }
}

// same unpack as get_scale_min_k4() in dequantize.hpp
अटल __dpct_inline__ व्योम fg_scale_min_k4(स्थिर पूर्णांक j, स्थिर uint8_t * __restrict__ q, uint8_t & d, uint8_t & m) {
    अगर (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } अन्यथा {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

// q4_K sub-block ib (0..7) uses scale/min pair ib and the low (even ib) or high (odd ib) nibbles of
// qs[32 * (ib / 2) ...], as in dequantize_row_q4_K
ढाँचा <typename E>
अटल __dpct_inline__ व्योम fg_decode_q4_K(स्थिर uint8_t * __restrict__ qs, स्थिर uint8_t * __restrict__ scales,
                                           स्थिर sycl::half2 dm, स्थिर पूर्णांक ib, typename E::pair * a) {
    uint8_t sc, mb;
    fg_scale_min_k4(ib, scales, sc, mb);
    स्थिर भग्न     d     = (भग्न) dm[0] * sc;
    स्थिर भग्न     m     = (भग्न) dm[1] * mb;
    स्थिर uint8_t * q     = qs + 32 * (ib / 2);
    स्थिर पूर्णांक       shift = 4 * (ib % 2);
#आशय unroll
    क्रम (पूर्णांक j = 0; j < 16; ++j) {
        a[j] = E::make(d * ((q[2 * j] >> shift) & 0xF) - m, d * ((q[2 * j + 1] >> shift) & 0xF) - m);
    }
}

// q5_K: q4_K plus one high bit per value, bit ib of qh[l]
ढाँचा <typename E>
अटल __dpct_inline__ व्योम fg_decode_q5_K(स्थिर uint8_t * __restrict__ qs, स्थिर uint8_t * __restrict__ qh,
                                           स्थिर uint8_t * __restrict__ scales, स्थिर sycl::half2 dm, स्थिर पूर्णांक ib,
                                           typename E::pair * a) {
    uint8_t sc, mb;
    fg_scale_min_k4(ib, scales, sc, mb);
    स्थिर भग्न     d     = (भग्न) dm[0] * sc;
    स्थिर भग्न     m     = (भग्न) dm[1] * mb;
    स्थिर uint8_t * q     = qs + 32 * (ib / 2);
    स्थिर पूर्णांक       shift = 4 * (ib % 2);
#आशय unroll
    क्रम (पूर्णांक j = 0; j < 16; ++j) {
        स्थिर पूर्णांक l = 2 * j;
        a[j] = E::make(d * (((q[l] >> shift) & 0xF) | (((qh[l] >> ib) & 1) << 4)) - m,
                       d * (((q[l + 1] >> shift) & 0xF) | (((qh[l + 1] >> ib) & 1) << 4)) - m);
    }
}

// q6_K: sub-block ib is quarter r = ib % 4 of half h = ib / 4. The half selects ql + 64h, qh + 32h
// and scales + 8h; the quarter selects ql + 32(r & 1), the ql nibble r / 2, the qh bit pair r and
// scales + 2r, as in dequantize_row_q6_K. The scale changes at value 16 of the sub-block.
ढाँचा <typename E>
अटल __dpct_inline__ व्योम fg_decode_q6_K(स्थिर uint8_t * __restrict__ ql, स्थिर uint8_t * __restrict__ qh,
                                           स्थिर int8_t * __restrict__ scales, स्थिर भग्न d, स्थिर पूर्णांक ib,
                                           typename E::pair * a) {
    स्थिर पूर्णांक       h  = ib / 4;
    स्थिर पूर्णांक       r  = ib % 4;
    स्थिर uint8_t * q  = ql + 64 * h + 32 * (r & 1);
    स्थिर uint8_t * hb = qh + 32 * h;
    स्थिर int8_t *  sc = scales + 8 * h + 2 * r;
#आशय unroll
    क्रम (पूर्णांक j = 0; j < 16; ++j) {
        स्थिर पूर्णांक   l  = 2 * j;
        स्थिर भग्न dl = d * sc[j / 8];
        स्थिर पूर्णांक   q0 = (((q[l] >> (4 * (r / 2))) & 0xF) | (((hb[l] >> (2 * r)) & 3) << 4)) - 32;
        स्थिर पूर्णांक   q1 = (((q[l + 1] >> (4 * (r / 2))) & 0xF) | (((hb[l + 1] >> (2 * r)) & 3) << 4)) - 32;
        a[j] = E::make(dl * q0, dl * q1);
    }
}

ढाँचा <typename E>
अटल __dpct_inline__ व्योम fg_stage_a(स्थिर block_q8_0 * __restrict__ xrow, स्थिर पूर्णांक kb, typename E::pair * a) {
    fg_decode_q8_0<E>(xrow[kb].qs, (भग्न) xrow[kb].d, a);
}

ढाँचा <typename E>
अटल __dpct_inline__ व्योम fg_stage_a(स्थिर block_q4_K * __restrict__ xrow, स्थिर पूर्णांक kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_q4_K);
    fg_decode_q4_K<E>(blk->qs, blk->scales, blk->dm, ib, a);
}

ढाँचा <typename E>
अटल __dpct_inline__ व्योम fg_stage_a(स्थिर block_q5_K * __restrict__ xrow, स्थिर पूर्णांक kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_q5_K);
    fg_decode_q5_K<E>(blk->qs, blk->qh, blk->scales, blk->dm, ib, a);
}

ढाँचा <typename E>
अटल __dpct_inline__ व्योम fg_stage_a(स्थिर block_q6_K * __restrict__ xrow, स्थिर पूर्णांक kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_q6_K);
    fg_decode_q6_K<E>(blk->ql, blk->qh, blk->scales, (भग्न) blk->d, ib, a);
}

// Reorder (SoA) layout: each block field is one stream over the nblocks of the matrix (of the expert
// slice for MUL_MAT_ID), in the order reorder_qw() writes them. Block ib holds k step kb.
ढाँचा <typename block_q_t> काष्ठा fg_soa {
    अटल constexpr bool supported = false;
};

ढाँचा <> काष्ठा fg_soa<block_q8_0> {
    अटल constexpr bool supported = true;
    // [qs][d]
    ढाँचा <typename E>
    अटल __dpct_inline__ व्योम stage(स्थिर uint8_t * x, स्थिर माप_प्रकार nblocks, स्थिर माप_प्रकार ib, स्थिर पूर्णांक,
                                      typename E::pair * a) {
        स्थिर भग्न d = (भग्न) ((स्थिर sycl::half *) (x + nblocks * QK8_0))[ib];
        fg_decode_q8_0<E>((स्थिर int8_t *) x + ib * QK8_0, d, a);
    }
};

ढाँचा <> काष्ठा fg_soa<block_q4_K> {
    अटल constexpr bool supported = true;
    // [qs][scales][dm]
    ढाँचा <typename E>
    अटल __dpct_inline__ व्योम stage(स्थिर uint8_t * x, स्थिर माप_प्रकार nblocks, स्थिर माप_प्रकार ib, स्थिर पूर्णांक kb,
                                      typename E::pair * a) {
        स्थिर uint8_t *   scales = x + nblocks * (QK_K / 2);
        स्थिर sycl::half2 dm     = ((स्थिर sycl::half2 *) (scales + nblocks * K_SCALE_आकार))[ib];
        fg_decode_q4_K<E>(x + ib * (QK_K / 2), scales + ib * K_SCALE_आकार, dm, kb % (QK_K / 32), a);
    }
};

ढाँचा <> काष्ठा fg_soa<block_q5_K> {
    अटल constexpr bool supported = true;
    // [qs][qh][scales][dm]
    ढाँचा <typename E>
    अटल __dpct_inline__ व्योम stage(स्थिर uint8_t * x, स्थिर माप_प्रकार nblocks, स्थिर माप_प्रकार ib, स्थिर पूर्णांक kb,
                                      typename E::pair * a) {
        स्थिर uint8_t *   qh     = x + nblocks * (QK_K / 2);
        स्थिर uint8_t *   scales = qh + nblocks * (QK_K / 8);
        स्थिर sycl::half2 dm     = ((स्थिर sycl::half2 *) (scales + nblocks * K_SCALE_आकार))[ib];
        fg_decode_q5_K<E>(x + ib * (QK_K / 2), qh + ib * (QK_K / 8), scales + ib * K_SCALE_आकार, dm,
                          kb % (QK_K / 32), a);
    }
};

ढाँचा <> काष्ठा fg_soa<block_q6_K> {
    अटल constexpr bool supported = true;
    // [ql][qh][scales][d]
    ढाँचा <typename E>
    अटल __dpct_inline__ व्योम stage(स्थिर uint8_t * x, स्थिर माप_प्रकार nblocks, स्थिर माप_प्रकार ib, स्थिर पूर्णांक kb,
                                      typename E::pair * a) {
        स्थिर uint8_t * qh     = x + nblocks * (QK_K / 2);
        स्थिर uint8_t * scales = qh + nblocks * (QK_K / 4);
        स्थिर भग्न     d      = (भग्न) ((स्थिर sycl::half *) (scales + nblocks * (QK_K / 16)))[ib];
        fg_decode_q6_K<E>(x + ib * (QK_K / 2), qh + ib * (QK_K / 4), (स्थिर int8_t *) scales + ib * (QK_K / 16), d,
                          kb % (QK_K / 32), a);
    }
};


// one SG_ROWS x BN output tile: B columns [b0, b0 + BN) of packed_b go to dst columns [n0, n1),
// n1 - n0 <= BN
ढाँचा <typename S, typename block_q_t, bool reordered>
अटल व्योम fused_dequant_gemm_tile(
    स्थिर block_q_t * __restrict__ x,
    स्थिर typename S::tsb * __restrict__ packed_b,
    भग्न * __restrict__ dst,
    स्थिर पूर्णांक M, स्थिर पूर्णांक Npad, स्थिर पूर्णांक K, स्थिर पूर्णांक ldd,
    स्थिर पूर्णांक b0, स्थिर पूर्णांक n0, स्थिर पूर्णांक n1,
    sycl::local_accessor<typename S::tsa, 1> tile_a,
    sycl::local_accessor<भग्न, 1> tile_c,
    स्थिर sycl::nd_item<2> & item) {
    अगर constexpr (fg_built<S::SG>()) {
        using EA = typename S::EA;
        using TA = typename S::ta;
        using TB = typename S::tb;
        स्थिर स्वतः sg     = item.get_sub_group();
        स्थिर पूर्णांक  sg_id  = sg.get_group_id()[0];
        स्थिर पूर्णांक  lane   = sg.get_local_id()[0];
        स्थिर पूर्णांक  m0     = item.get_group(1) * S::SG_ROWS;
        स्थिर पूर्णांक  nstep  = K / FG_BK;
        स्थिर पूर्णांक  a_base = sg_id * S::SG_ROWS * FG_BK;
        स्थिर पूर्णांक  c_base = sg_id * S::SG_ROWS * S::BN;

        mx::joint_matrix<sycl::sub_group, भग्न, mx::use::accumulator, S::TM, S::TN> acc[S::MT][S::NT];
#आशय unroll
        क्रम (पूर्णांक mt = 0; mt < S::MT; ++mt) {
#आशय unroll
            क्रम (पूर्णांक nt = 0; nt < S::NT; ++nt) {
                mx::joint_matrix_fill(sg, acc[mt][nt], 0.0f);
            }
        }

        // lane decodes rows lane, lane + SG, ... of the sub-group's SG_ROWS
        constexpr पूर्णांक         KPB     = fg_block_traits<block_q_t>::qk / FG_BK; // k steps per block
        स्थिर माप_प्रकार          bpr     = K / fg_block_traits<block_q_t>::qk;
        स्थिर माप_प्रकार          nblocks = (माप_प्रकार) M * bpr;
        माप_प्रकार                row_blk[S::RPL];
        bool                  row_ok[S::RPL];
        typename EA::pair *   a[S::RPL];
#आशय unroll
        क्रम (पूर्णांक r = 0; r < S::RPL; ++r) {
            स्थिर पूर्णांक row = m0 + r * S::SG + lane;
            row_ok[r]  = row < M;
            row_blk[r] = (माप_प्रकार) (row_ok[r] ? row : 0) * bpr;
            a[r]       = (typename EA::pair *) &tile_a[a_base + (r * S::SG + lane) * FG_BK];
        }

        स्थिर स्वतः b_ptr = sycl::address_space_cast<sycl::access::address_space::global_space,
                                                    sycl::access::decorated::no>(packed_b);
        स्थिर पूर्णांक b_stride = Npad * S::VNNI;

        स्थिर पूर्णांक kb_begin = (sg_id * nstep) / FG_KSPLIT;
        स्थिर पूर्णांक kb_end   = ((sg_id + 1) * nstep) / FG_KSPLIT;
        क्रम (पूर्णांक kb = kb_begin; kb < kb_end; ++kb) {
#आशय unroll
            क्रम (पूर्णांक r = 0; r < S::RPL; ++r) {
                अगर (row_ok[r]) {
                    अगर constexpr (reordered) {
                        fg_soa<block_q_t>::ढाँचा stage<EA>((स्थिर uint8_t *) x, nblocks, row_blk[r] + kb / KPB, kb,
                                                              a[r]);
                    } अन्यथा {
                        fg_stage_a<EA>(x + row_blk[r], kb, a[r]);
                    }
                } अन्यथा {
#आशय unroll
                    क्रम (पूर्णांक j = 0; j < FG_BK / 2; ++j) {
                        a[r][j] = EA::make(0.0f, 0.0f);
                    }
                }
            }
            sycl::group_barrier(sg);

#आशय unroll
            क्रम (पूर्णांक kt = 0; kt < FG_BK / S::TK; ++kt) {
                स्थिर पूर्णांक kq0 = (kb * FG_BK + kt * S::TK) / S::VNNI;
                mx::joint_matrix<sycl::sub_group, TB, mx::use::b, S::TK, S::TN, S::b_layout> sub_b[S::NT];
#आशय unroll
                क्रम (पूर्णांक nt = 0; nt < S::NT; ++nt) {
                    mx::joint_matrix_load(sg, sub_b[nt], b_ptr + (माप_प्रकार) kq0 * b_stride + (b0 + nt * S::TN) * S::VNNI, b_stride);
                }
#आशय unroll
                क्रम (पूर्णांक mt = 0; mt < S::MT; ++mt) {
                    mx::joint_matrix<sycl::sub_group, TA, mx::use::a, S::TM, S::TK, mx::layout::row_major> sub_a;
                    mx::joint_matrix_load(sg, sub_a,
                        tile_a.ढाँचा get_multi_ptr<sycl::access::decorated::no>() + a_base + (mt * S::TM) * FG_BK + kt * S::TK,
                        FG_BK);
#आशय unroll
                    क्रम (पूर्णांक nt = 0; nt < S::NT; ++nt) {
                        mx::joint_matrix_mad(sg, acc[mt][nt], sub_a, sub_b[nt], acc[mt][nt]);
                    }
                }
            }
            // the next step overwrites tile_a
            sycl::group_barrier(sg);
        }

#आशय unroll
        क्रम (पूर्णांक mt = 0; mt < S::MT; ++mt) {
#आशय unroll
            क्रम (पूर्णांक nt = 0; nt < S::NT; ++nt) {
                mx::joint_matrix_store(sg, acc[mt][nt],
                    tile_c.ढाँचा get_multi_ptr<sycl::access::decorated::no>() + c_base + (mt * S::TM) * S::BN + nt * S::TN,
                    S::BN, mx::layout::row_major);
            }
        }
        sycl::group_barrier(item.get_group());

        // sum the K splits; consecutive lanes write consecutive rows of one dst column
        क्रम (पूर्णांक idx = item.get_local_linear_id(); idx < S::SG_ROWS * S::BN; idx += S::WG_आकार) {
            स्थिर पूर्णांक r = idx % S::SG_ROWS;
            स्थिर पूर्णांक c = idx / S::SG_ROWS;
            स्थिर पूर्णांक m = m0 + r;
            स्थिर पूर्णांक n = n0 + c;
            अगर (m < M && n < n1) {
                भग्न योग_2 = 0.0f;
#आशय unroll
                क्रम (पूर्णांक s = 0; s < FG_KSPLIT; ++s) {
                    योग_2 += tile_c[s * S::SG_ROWS * S::BN + r * S::BN + c];
                }
                dst[(माप_प्रकार) n * ldd + m] = योग_2;
            }
        }
    }
}

ढाँचा <typename S, typename block_q_t, bool reordered = false>
अटल व्योम fused_dequant_gemm_launch(स्थिर व्योम * src0, स्थिर typename S::tsb * packed, भग्न * dst, स्थिर पूर्णांक M,
                                      स्थिर पूर्णांक N, स्थिर पूर्णांक Npad, स्थिर पूर्णांक K, स्थिर पूर्णांक ldd,
                                      स्थिर int64_t groups_n, स्थिर int64_t groups_m, dpct::queue_ptr stream) {
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<typename S::tsa, 1> tile_a(FG_KSPLIT * S::SG_ROWS * FG_BK, cgh);
        sycl::local_accessor<भग्न, 1>          tile_c(FG_KSPLIT * S::SG_ROWS * S::BN, cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(groups_n, groups_m * S::WG_आकार), sycl::range<2>(1, S::WG_आकार)),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_आकार(S::SG)]] {
                स्थिर पूर्णांक n0 = item.get_group(0) * S::BN;
                fused_dequant_gemm_tile<S, block_q_t, reordered>((स्थिर block_q_t *) src0, packed, dst, M, Npad, K,
                                                      ldd, n0, n0, N, tile_a, tile_c, item);
            });
    });
}

// grouped: work-group (t, mt) is tile t of the schedule; its B columns sit at t * BN
ढाँचा <typename S, typename block_q_t, bool reordered>
अटल व्योम grouped_dequant_gemm_launch(स्थिर अक्षर * src0_dd, स्थिर माप_प्रकार expert_stride,
                                        स्थिर ggml_sycl_gg_tile * tiles_ptr, स्थिर typename S::tsb * packed, भग्न * dst,
                                        स्थिर पूर्णांक M, स्थिर पूर्णांक Npad, स्थिर पूर्णांक K, स्थिर int64_t n_tiles,
                                        स्थिर int64_t groups_m, dpct::queue_ptr stream) {
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<typename S::tsa, 1> tile_a(FG_KSPLIT * S::SG_ROWS * FG_BK, cgh);
        sycl::local_accessor<भग्न, 1>          tile_c(FG_KSPLIT * S::SG_ROWS * S::BN, cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(n_tiles, groups_m * S::WG_आकार), sycl::range<2>(1, S::WG_आकार)),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_आकार(S::SG)]] {
                स्थिर पूर्णांक               t    = item.get_group(0);
                स्थिर ggml_sycl_gg_tile tile = tiles_ptr[t];
                स्थिर block_q_t *       x    = (स्थिर block_q_t *) (src0_dd + (माप_प्रकार) tile.expert * expert_stride);
                fused_dequant_gemm_tile<S, block_q_t, reordered>(x, packed, dst, M, Npad, K, M, t * S::BN, tile.n0,
                                                                 tile.n1, tile_a, tile_c, item);
            });
    });
}

// src1 f32 rows -> packed [K/V][n_tiles*BN][V], tile t holds its rows [n0, n1) at columns t*BN..,
// zero past n1. The column runs fastest so a sub-group writes one contiguous run.
ढाँचा <typename S>
अटल व्योम grouped_gemm_pack_b(स्थिर भग्न * y, typename S::tsb * packed, स्थिर ggml_sycl_gg_tile * tiles, पूर्णांक Npad,
                                पूर्णांक K, dpct::queue_ptr stream) {
    using E        = typename S::EB;
    constexpr पूर्णांक V = S::VNNI;
    स्थिर पूर्णांक kqs   = K / V;
    stream->parallel_for(sycl::range<1>((माप_प्रकार) Npad * kqs), [=](sycl::id<1> id) {
        स्थिर माप_प्रकार idx = id[0];
        स्थिर पूर्णांक    kq  = idx / Npad;
        स्थिर पूर्णांक    n   = idx - (माप_प्रकार) kq * Npad;
        स्थिर ggml_sycl_gg_tile tile = tiles[n / S::BN];
        स्थिर पूर्णांक    row = tile.n0 + n % S::BN;
        // one guarded load run per work-item, as a per-element select costs ~1.5% prefill
        typename S::tsb vals[V] = {};
        अगर (row < tile.n1) {
            स्थिर भग्न * src = y + (माप_प्रकार) row * K + V * kq;
#आशय unroll
            क्रम (पूर्णांक v = 0; v < V; ++v) {
                vals[v] = E::cvt(src[v]);
            }
        }
        typename S::tsb * out = packed + ((माप_प्रकार) kq * Npad + n) * V;
#आशय unroll
        क्रम (पूर्णांक v = 0; v < V; ++v) {
            out[v] = vals[v];
        }
    });
}

// q8_0 and the k-quants take only the grouped path. The plain kernel decodes A again for every BN columns
// of a dense batch, and for these formats that costs more than the one dequantization of the library GEMM.
ढाँचा <typename T> अटल constexpr bool fg_plain_ok() {
    वापस !std::is_same_v<T, block_q8_0> && !std::is_same_v<T, block_q4_K> && !std::is_same_v<T, block_q5_K> &&
           !std::is_same_v<T, block_q6_K>;
}

ढाँचा <typename T, bool R> काष्ठा fg_tag {
    using type                     = T;
    अटल constexpr bool reordered = R;
};

// calls f(fg_tag<block_q_t, reordered>{}) for the weight format and layout; false if it has no A stage
ढाँचा <typename T, typename F> अटल bool fg_visit_layout(bool reordered, F && f) {
    अगर (!reordered) {
        f(fg_tag<T, false>{});
        वापस true;
    }
    अगर constexpr (fg_soa<T>::supported) {
        f(fg_tag<T, true>{});
        वापस true;
    }
    वापस false;
}

ढाँचा <typename F> अटल bool fg_visit_type(ggml_type type, bool reordered, F && f) {
    चयन (type) {
        हाल GGML_TYPE_IQ4_NL:  वापस fg_visit_layout<block_iq4_nl>(reordered, f);
        हाल GGML_TYPE_IQ3_S:   वापस fg_visit_layout<block_iq3_s>(reordered, f);
        हाल GGML_TYPE_IQ4_XS:  वापस fg_visit_layout<block_iq4_xs>(reordered, f);
        हाल GGML_TYPE_IQ3_XXS: वापस fg_visit_layout<block_iq3_xxs>(reordered, f);
        हाल GGML_TYPE_IQ2_XXS: वापस fg_visit_layout<block_iq2_xxs>(reordered, f);
        हाल GGML_TYPE_IQ2_XS:  वापस fg_visit_layout<block_iq2_xs>(reordered, f);
        हाल GGML_TYPE_IQ2_S:   वापस fg_visit_layout<block_iq2_s>(reordered, f);
        हाल GGML_TYPE_IQ1_S:   वापस fg_visit_layout<block_iq1_s>(reordered, f);
        हाल GGML_TYPE_IQ1_M:   वापस fg_visit_layout<block_iq1_m>(reordered, f);
        हाल GGML_TYPE_Q8_0:    वापस fg_visit_layout<block_q8_0>(reordered, f);
        हाल GGML_TYPE_Q4_K:    वापस fg_visit_layout<block_q4_K>(reordered, f);
        हाल GGML_TYPE_Q5_K:    वापस fg_visit_layout<block_q5_K>(reordered, f);
        हाल GGML_TYPE_Q6_K:    वापस fg_visit_layout<block_q6_K>(reordered, f);
        शेष:                वापस false;
    }
}

ढाँचा <typename S>
अटल bool fg_fused_run(ggml_type src0_type, bool src0_reordered, स्थिर व्योम * src0, स्थिर व्योम * src1, ggml_type src1_type,
                         भग्न * dst, int64_t M, int64_t N, int64_t K, int64_t ldd, ggml_sycl_pool & pool,
                         dpct::queue_ptr stream) {
    स्थिर int64_t groups_n = (N + S::BN - 1) / S::BN;
    स्थिर int64_t groups_m = (M + S::SG_ROWS - 1) / S::SG_ROWS;
    स्थिर पूर्णांक     Npad     = (पूर्णांक) (groups_n * S::BN);

    // src1 is read in its own type: one pass, converted in registers only if B differs
    ggml_sycl_pool_alloc<typename S::tsb> packed_b(pool, (माप_प्रकार) K * Npad);
    अगर (src1_type == GGML_TYPE_F16) {
        fused_gemm_pack_b<typename S::EB>((स्थिर sycl::half *) src1, packed_b.लो(), (पूर्णांक) N, Npad, (पूर्णांक) K, stream);
    } अन्यथा अगर (src1_type == GGML_TYPE_BF16) {
        fused_gemm_pack_b<typename S::EB>((स्थिर fg_bf16 *) src1, packed_b.लो(), (पूर्णांक) N, Npad, (पूर्णांक) K, stream);
    } अन्यथा {
        fused_gemm_pack_b<typename S::EB>((स्थिर भग्न *) src1, packed_b.लो(), (पूर्णांक) N, Npad, (पूर्णांक) K, stream);
    }

    स्थिर typename S::tsb * packed = packed_b.लो();
    वापस fg_visit_type(src0_type, src0_reordered, [&](स्वतः tag) {
        using block_q_t = typename decltype(tag)::type;
        अगर constexpr (fg_plain_ok<block_q_t>()) {
            fused_dequant_gemm_launch<S, block_q_t, decltype(tag)::reordered>(src0, packed, dst, (पूर्णांक) M, (पूर्णांक) N,
                                                                              Npad, (पूर्णांक) K, (पूर्णांक) ldd, groups_n,
                                                                              groups_m, stream);
        }
    });
}

bool ggml_sycl_fused_dequant_gemm(ggml_type src0_type, bool src0_reordered, स्थिर व्योम * src0, स्थिर व्योम * src1, ggml_type src1_type,
                                  int32_t src1_prec, भग्न * dst, int64_t M, int64_t N, int64_t K, int64_t ldd,
                                  ggml_sycl_pool & pool, dpct::queue_ptr stream) {
    // every BN columns dequantize A again, so wide N is left to the library GEMM.
    // src0_reordered: the persistent weight reorder may have been installed on this
    // tensor by an earlier small-batch mul_mat; the A stage then has to decode the
    // SoA layout. Types without an SoA stage (fg_soa<T>::supported == false)
    // decline via fg_visit_layout and the caller falls through to the
    // reorder-aware conversion route.
    bool plain_ok = false;
    fg_visit_type(src0_type, src0_reordered, [&](स्वतः tag) { plain_ok = fg_plain_ok<typename decltype(tag)::type>(); });
    अगर (g_ggml_sycl_dynamic_precision == GGML_SYCL_DYNAMIC_PRECISION_F32 ||
        !ggml_sycl_xmx_gather_type_enabled(src0_type) || !plain_ok) {
        वापस false;
    }
    अगर (src1_type != GGML_TYPE_F32 && src1_type != GGML_TYPE_F16 && src1_type != GGML_TYPE_BF16) {
        वापस false;
    }
    अगर (!ggml_sycl_fused_dequant_gemm_shape_ok(src0_type, M, N, K, ldd)) {
        वापस false;
    }
    स्थिर पूर्णांक combo = fg_pick_combo(stream, src1_type, src1_prec);
    अगर (combo < 0) {
        वापस false;
    }
    bool launched = false;
    fg_visit_combo(combo, [&](स्वतः s) {
        launched = fg_fused_run<decltype(s)>(src0_type, src0_reordered, src0, src1, src1_type, dst, M, N, K, ldd, pool, stream);
    });
    वापस launched;
}

ढाँचा <typename S>
अटल bool fg_grouped_run(ggml_type src0_type, bool reordered, स्थिर व्योम * src0_base, माप_प्रकार expert_stride,
                           स्थिर भग्न * src1, भग्न * dst, स्थिर int64_t * expert_row_offsets, int64_t n_as, int64_t M,
                           int64_t K, std::vector<ggml_sycl_gg_tile> & tiles, ggml_sycl_pool & pool,
                           dpct::queue_ptr stream) {
    // the host knows every slice, so it lays out the work-groups: no search on the device
    tiles.सफा();
    क्रम (int64_t e = 0; e < n_as; ++e) {
        स्थिर int64_t end = expert_row_offsets[e + 1];
        क्रम (int64_t n0 = expert_row_offsets[e]; n0 < end; n0 += S::BN) {
            tiles.push_back({ (int32_t) e, (int32_t) n0, (int32_t) std::min<int64_t>(n0 + S::BN, end) });
        }
    }
    स्थिर int64_t n_tiles  = tiles.size();
    स्थिर int64_t groups_m = (M + S::SG_ROWS - 1) / S::SG_ROWS;
    स्थिर पूर्णांक     Npad     = (पूर्णांक) (n_tiles * S::BN);

    स्थिर int64_t max_tiles = grouped_gemm_max_tiles(expert_row_offsets[n_as], n_as, S::BN);
    GGML_ASSERT(n_tiles <= max_tiles);
    ggml_sycl_pool_alloc<ggml_sycl_gg_tile> tiles_dev(pool, max_tiles);
    SYCL_CHECK(CHECK_TRY_ERROR(stream->स_नकल(tiles_dev.लो(), tiles.तथ्य(), n_tiles * माप(ggml_sycl_gg_tile))));

    ggml_sycl_pool_alloc<typename S::tsb> packed_b(pool, (माप_प्रकार) K * max_tiles * S::BN);
    grouped_gemm_pack_b<S>(src1, packed_b.लो(), tiles_dev.लो(), Npad, (पूर्णांक) K, stream);

    स्थिर typename S::tsb *   packed    = packed_b.लो();
    स्थिर ggml_sycl_gg_tile * tiles_ptr = tiles_dev.लो();
    स्थिर अक्षर *              src0_dd   = (स्थिर अक्षर *) src0_base;
    वापस fg_visit_type(src0_type, reordered, [&](स्वतः tag) {
        using T = decltype(tag);
        grouped_dequant_gemm_launch<S, typename T::type, T::reordered>(src0_dd, expert_stride, tiles_ptr, packed, dst,
                                                                       (पूर्णांक) M, Npad, (पूर्णांक) K, n_tiles, groups_m,
                                                                       stream);
    });
}

bool ggml_sycl_grouped_dequant_gemm(ggml_type src0_type, bool reordered, स्थिर व्योम * src0_base, माप_प्रकार expert_stride,
                                    स्थिर भग्न * src1, int32_t src1_prec, भग्न * dst,
                                    स्थिर int64_t * expert_row_offsets, int64_t n_as, int64_t M, int64_t K,
                                    int64_t योग_rows, std::vector<ggml_sycl_gg_tile> & tiles,
                                    ggml_sycl_pool & pool, dpct::queue_ptr stream) {
    int64_t n_active = 0;
    क्रम (int64_t e = 0; e < n_as; ++e) {
        n_active += expert_row_offsets[e + 1] > expert_row_offsets[e];
    }
    अगर (g_ggml_sycl_dynamic_precision == GGML_SYCL_DYNAMIC_PRECISION_F32 ||
        !ggml_sycl_xmx_gather_type_enabled(src0_type) || !fg_visit_type(src0_type, reordered, [](स्वतः) {})) {
        वापस false;
    }
    अगर (!ggml_sycl_grouped_dequant_gemm_shape_ok(src0_type, M, K, योग_rows, n_active)) {
        वापस false;
    }
    स्थिर पूर्णांक combo = fg_pick_combo(stream, GGML_TYPE_F32, src1_prec);
    अगर (combo < 0) {
        वापस false;
    }
    bool launched = false;
    fg_visit_combo(combo, [&](स्वतः s) {
        launched = fg_grouped_run<decltype(s)>(src0_type, reordered, src0_base, expert_stride, src1, dst,
                                               expert_row_offsets, n_as, M, K, tiles, pool, stream);
    });
    वापस launched;
}
