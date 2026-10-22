#अन्तर्भाव "fused-gemm.hpp"

#अन्तर्भाव <sycl/ext/oneapi/matrix/matrix.hpp>

#अन्तर्भाव <algorithm>
#अन्तर्भाव <string>
#अन्तर्भाव <tuple>
#अन्तर्भाव <mutex>
#अन्तर्भाव <set>
#अन्तर्भाव <unordered_map>

नामाकाश mx = sycl::ext::oneapi::experimental::matrix;

// FG_ / fg_ is short for fused GEMM: the weights are dequantized inside the GEMM, into the XMX tiles.

// A k step is one 32-value weight sub-block; iq3_s and the other superblock formats split their
// superblock into steps of this width. The sub-groups of a work-group each walk their own K range
// and are summed at the end.
स्थिर constexpr पूर्णांक FG_BK     = QK4_NL;
स्थिर constexpr पूर्णांक FG_KSPLIT = 4;

// Element traits of one joint_matrix operand type. The A stage and the B pack compute in f32 and
// convert once, in registers, when they write the element, so any type costs the same one pass.
//   store: storage in SLM (A) and in the packed B buffer
//   mtype: matrix_type in matrix_combinations
//   mode:  GGML_SYCL_DYNAMIC_PRECISION value that selects this type
//   src:   ggml type that needs no conversion into this type (GGML_TYPE_COUNT: none)
//   slow:  XMX throughput class, 0 is fastest. f16 and bf16 share the DPAS rate; tf32 does half the
//          K per instruction. B60, Qwen3-30B-A3B pp512: f16 1108, bf16 1000, tf32 751 t/s
ढाँचा <typename T> संरचना fg_elem;

ढाँचा <> संरचना fg_elem<sycl::half> {
    using store = sycl::half;
    using pair  = sycl::half2;
    स्थिर constexpr mx::matrix_type mtype = mx::matrix_type::fp16;
    स्थिर constexpr पूर्णांक             mode  = GGML_SYCL_DYNAMIC_PRECISION_F16;
    स्थिर constexpr ggml_type       src   = GGML_TYPE_F16;
    स्थिर constexpr पूर्णांक             mant  = 10;
    स्थिर constexpr पूर्णांक             slow  = 0;
    स्थिर store cvt(दशांश x) { परत (store) x; }
    स्थिर pair make(दशांश x, दशांश y) { परत pair((store) x, (store) y); }
};

// tf32 rounds to nearest even with plain bit ops: round_to_tf32 needs a SPIR-V extension that the
// DG2 AOT target rejects
स्थिर ओळीत uint32_t fg_round_bits(दशांश x, पूर्णांक drop) {
    स्थिरांक uint32_t u = sycl::bit_cast<uint32_t>(x);
    जर ((u & 0x7f800000u) == 0x7f800000u) {
        परत (u & 0x7fffffu) ? u | (1u << drop) : u; // nan stays nan
    }
    परत u + ((1u << (drop - 1)) - 1) + ((u >> drop) & 1);
}

संरचना alignas(4) fg_bf16x2 {
    sycl::ext::oneapi::bfloat16 x, y;
};

ढाँचा <> संरचना fg_elem<sycl::ext::oneapi::bfloat16> {
    using store = sycl::ext::oneapi::bfloat16;
    using pair  = fg_bf16x2;
    स्थिर constexpr mx::matrix_type mtype = mx::matrix_type::bf16;
    स्थिर constexpr पूर्णांक             mode  = GGML_SYCL_DYNAMIC_PRECISION_BF16;
    स्थिर constexpr ggml_type       src   = GGML_TYPE_BF16;
    स्थिर constexpr पूर्णांक             mant  = 7;
    स्थिर constexpr पूर्णांक             slow  = 0;
    स्थिर store cvt(दशांश x) { परत store(x); }
    स्थिर pair make(दशांश x, दशांश y) { परत { cvt(x), cvt(y) }; }
};

// tf32 keeps f32 range and f16 mantissa, in f32 storage
ढाँचा <> संरचना fg_elem<mx::precision::tf32> {
    using store = दशांश;
    using pair  = sycl::float2;
    स्थिर constexpr mx::matrix_type mtype = mx::matrix_type::tf32;
    स्थिर constexpr पूर्णांक             mode  = GGML_SYCL_DYNAMIC_PRECISION_TF32;
    स्थिर constexpr ggml_type       src   = GGML_TYPE_मोजणी;
    स्थिर constexpr पूर्णांक             mant  = 10;
    स्थिर constexpr पूर्णांक             slow  = 1;
    स्थिर store cvt(दशांश x) { परत sycl::bit_cast<दशांश>(fg_round_bits(x, 13) & ~0x1fffu); }
    स्थिर pair make(दशांश x, दशांश y) { परत pair(cvt(x), cvt(y)); }
};

// One joint_matrix combination (A type, B type, TM x TN x TK, sub-group size; C and D are f32) and
// the tiling built on it. A sub-group owns SG_ROWS rows of A (at least 16) and BN (at least 32)
// columns of B. A and B may differ: the device lists the pairs it supports.
ढाँचा <typename TA, typename TB, पूर्णांक TM_, पूर्णांक TN_, पूर्णांक TK_, पूर्णांक SG_> संरचना fg_combo {
    using ta  = TA;
    using tb  = TB;
    using EA  = fg_elem<TA>;
    using EB  = fg_elem<TB>;
    using tsa = typename EA::store;
    using tsb = typename EB::store;
    स्थिर constexpr पूर्णांक TM = TM_;
    स्थिर constexpr पूर्णांक TN = TN_;
    स्थिर constexpr पूर्णांक TK = TK_;
    स्थिर constexpr पूर्णांक SG = SG_;
    स्थिर constexpr पूर्णांक VNNI    = 4 / आकार(tsb);  // K rows of B packed in one 32-bit word
    स्थिर constexpr पूर्णांक SG_ROWS = TM > 16 ? TM : 16;
    स्थिर constexpr पूर्णांक RPL     = SG_ROWS / SG;     // A rows one lane decodes per k step
    स्थिर constexpr पूर्णांक MT      = SG_ROWS / TM;
    स्थिर constexpr पूर्णांक BN      = TN > 32 ? TN : 32;
    स्थिर constexpr पूर्णांक NT      = BN / TN;
    स्थिर constexpr पूर्णांक WG_SIZE = FG_KSPLIT * SG;
    स्थिर constexpr mx::layout b_layout = VNNI == 1 ? mx::layout::row_major : mx::layout::ext_intel_packed;
    // a 64-wide N is mostly padding here and a 32x64 f32 accumulator needs 128 registers per lane,
    // so it spills: 13x slower on B60
    स्थिर constexpr तार्किक efficient = TN <= 32;
    static_assert(SG_ROWS % SG == 0 && SG_ROWS % TM == 0 && BN % TN == 0 && FG_BK % TK == 0, "bad tile");
    static_assert(BN <= GGML_SYCL_FG_MAX_N, "header gate must cover the tile width");
};

using fg_half = sycl::half;
using fg_bf16 = sycl::ext::oneapi::bfloat16;
using fg_tf32 = mx::precision::tf32;

// One bit of GGML_SYCL_XMX_GATHER_SHAPES per combination. Only combinations some device lists in
// matrix_combinations are built (appendix of sycl_ext_oneapi_matrix and the runtime's own list).
ढाँचा <typename F> स्थिर शून्य fg_visit_combo(पूर्णांक idx, F && f);
स्थिर constexpr पूर्णांक FG_N_COMBOS = 8;

// A spir64_gen AOT build (GGML_SYCL_XMX_AOT_SG) drops the combinations of the other sub-group size
// entirely: ocloc rejects even an empty kernel that asks for a sub-group size it lacks.
ढाँचा <पूर्णांक SG> स्थिर constexpr तार्किक fg_listed() {
#जर defined(GGML_SYCL_XMX_AOT_SG)
    परत SG == GGML_SYCL_XMX_AOT_SG;
#अन्यथा
    परत सत्य;
#endif
}

ढाँचा <typename S, typename F> स्थिर शून्य fg_call_combo(F && f) {
    जर constexpr (fg_listed<S::SG>()) {
        f(S{});
    }
}

ढाँचा <typename F> स्थिर शून्य fg_visit_combo(पूर्णांक idx, F && f) {
    निवड (idx) {
        स्थिती 0: fg_call_combo<fg_combo<fg_half, fg_half, 8, 16, 16, 16>>(f);  थांबा; // Xe2, Xe3, Xe-HPC
        स्थिती 1: fg_call_combo<fg_combo<fg_half, fg_half, 16, 16, 16, 16>>(f); थांबा; // Xe2, Xe3, Xe-HPC
        स्थिती 2: fg_call_combo<fg_combo<fg_half, fg_half, 32, 64, 16, 16>>(f); थांबा; // Xe2, Xe3, Xe-HPC
        स्थिती 3: fg_call_combo<fg_combo<fg_half, fg_half, 32, 64, 32, 16>>(f); थांबा; // Xe2, Xe3, Xe-HPC
        स्थिती 4: fg_call_combo<fg_combo<fg_half, fg_half, 8, 8, 16, 8>>(f);    थांबा; // Xe-HPG (Arc A), ARL-H
        स्थिती 5: fg_call_combo<fg_combo<fg_tf32, fg_tf32, 8, 16, 8, 16>>(f);   थांबा; // Xe2, Xe3, Xe-HPC
        स्थिती 6: fg_call_combo<fg_combo<fg_bf16, fg_bf16, 8, 16, 16, 16>>(f);  थांबा; // Xe2, Xe3, Xe-HPC
        स्थिती 7: fg_call_combo<fg_combo<fg_bf16, fg_bf16, 8, 8, 16, 8>>(f);    थांबा; // Xe-HPG (Arc A), ARL-H
        पूर्वनिर्धारित: GGML_ABORT("bad XMX combination %d", idx);
    }
}

// AOT with -fsycl-targets=intel_gpu_*: compile each tile body only for targets with its sub-group
// size, since IGC fails on the other ones. A JIT build keeps them all, but each combination lands in
// its own device image (joint_matrix is an optional kernel feature) and only a combination the
// device reports is launched, so the runtime never asks IGC for the others.
#जर defined(__SYCL_DEVICE_ONLY__)
#    जर __SYCL_TARGET_INTEL_GPU_ACM_G10__ || __SYCL_TARGET_INTEL_GPU_ACM_G11__ || __SYCL_TARGET_INTEL_GPU_ACM_G12__ || \
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

ढाँचा <पूर्णांक SG> स्थिर constexpr तार्किक fg_built() {
#जर defined(FG_AOT_SG)
    परत SG == FG_AOT_SG;
#अन्यथा
    परत सत्य;
#endif
}

// Upper bound on the tile count when total_rows rows are routed to n_as experts: the worst case gives
// each expert one row and fills whole tiles with the rest. The bound depends only on the shape, not
// on the routing, so the pool reuses one buffer every ubatch instead of keeping one per size seen.
स्थिर constexpr int64_t grouped_gemm_max_tiles(int64_t total_rows, int64_t n_as, int64_t BN) {
    परत total_rows <= n_as ? total_rows : n_as + (total_rows - n_as) / BN;
}
// Tiles do not cross experts, so the bound is not ceil(total_rows / BN): 34 rows over 2 experts with
// BN = 16 split 17 + 17 need 2 + 2 tiles, where the ceil gives 3.
static_assert(grouped_gemm_max_tiles(34, 2, 16) == 4);

// the device lists S with an f32 accumulator and output
ढाँचा <typename S> स्थिर तार्किक fg_device_has_combo(स्थिरांक std::vector<mx::combination> & combinations) {
    साठी (स्थिरांक स्वतः & c : combinations) {
        जर (c.atype == S::EA::mtype && c.btype == S::EB::mtype && c.ctype == mx::matrix_type::fp32 &&
            c.dtype == mx::matrix_type::fp32 &&
            (c.max_msize >= (माप_प्रकार) S::TM || c.msize == (माप_प्रकार) S::TM) &&
            (c.max_nsize >= (माप_प्रकार) S::TN || c.nsize == (माप_प्रकार) S::TN) &&
            (c.max_ksize >= (माप_प्रकार) S::TK || c.ksize == (माप_प्रकार) S::TK)) {
            परत सत्य;
        }
    }
    परत असत्य;
}

ढाँचा <typename T> स्थिर स्थिरांक अक्षर * fg_type_name() {
    परत std::is_same_v<T, fg_half> ? "f16" : std::is_same_v<T, fg_bf16> ? "bf16" : "tf32";
}

स्थिर std::string fg_combo_name(पूर्णांक idx) {
    std::string name;
    fg_visit_combo(idx, [&](स्वतः s) {
        using S = decltype(s);
        name = std::string(fg_type_name<typename S::ta>()) + "x" + fg_type_name<typename S::tb>() + " " +
               std::to_string(S::TM) + "x" + std::to_string(S::TN) + "x" + std::to_string(S::TK) + " sg" +
               std::to_string(S::SG);
    });
    परत name;
}

// Combinations this build has kernels for and the device lists, one bit each. Cached per device:
// on a mixed box the first caller's verdict is not the others'.
स्थिर पूर्णांक fg_device_combos(स्थिरांक sycl::device & dev) {
    स्थिर std::mutex                            mtx;
    स्थिर std::unordered_map<sycl::device, पूर्णांक> known;
    std::lock_guard<std::mutex>                  lock(mtx);
    स्थिरांक स्वतः                                   it = known.find(dev);
    जर (it != known.end()) {
        परत it->second;
    }
    पूर्णांक available = 0;
    प्रयत्न {
        स्थिरांक स्वतः combinations = dev.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>();
        स्थिरांक स्वतः sg_sizes     = dev.get_info<sycl::info::device::sub_group_sizes>();
        साठी (पूर्णांक idx = 0; idx < FG_N_COMBOS; ++idx) {
            fg_visit_combo(idx, [&](स्वतः s) {
                using S = decltype(s);
                स्थिरांक तार्किक sg_ok = std::find(sg_sizes.begin(), sg_sizes.end(), (माप_प्रकार) S::SG) != sg_sizes.end();
                जर (sg_ok && fg_device_has_combo<S>(combinations)) {
                    available |= 1 << idx;
                }
            });
        }
    } पकडा (स्थिरांक sycl::exception &) {
        available = 0;
    }
    GGML_LOG_INFO("%s: %s: XMX dequant-GEMM combinations available 0x%x, allowed 0x%x\n", __func__,
                  dev.get_info<sycl::info::device::name>().c_str(), available, g_ggml_sycl_xmx_gather_shapes);
    known.emplace(dev, available);
    परत available;
}

// Rank of combination S for a src1 of type src1_type, lower is better. Order:
//  1. throughput: a tile that does not spill, then the fastest type class of A and B
//  2. B type equal to the src1 type, so the pack is a plain copy
//  3. B at least as precise as f16
//  4. the device's native DPAS tile (8 x SG x 32 bytes of K), then the largest M x K
// A costs nothing to convert: the A stage emits any type at the same cost.
ढाँचा <typename S> स्थिर int64_t fg_rank(ggml_type src1_type) {
    स्थिरांक int64_t spills  = !S::efficient;
    स्थिरांक int64_t slow    = std::max(S::EA::slow, S::EB::slow);
    स्थिरांक int64_t convert = S::EB::src != src1_type;
    स्थिरांक int64_t lossy   = S::EB::mant < 10;
    स्थिरांक int64_t foreign = !(S::TM == 8 && S::TN == S::SG);
    स्थिरांक int64_t mk      = 1024 - S::TM * S::TK;
    परत ((((spills * 2 + slow) * 2 + convert) * 2 + lossy) * 2 + foreign) * 2048 + mk;
}

// whether XMX operands of type mode (a GGML_SYCL_DYNAMIC_PRECISION value) meet the src1 request
// [TAG_GGML_PREC]. f16 lacks the f32 range that BF16 and F32 ask for; an F32 request goes only as far
// down as GGML_SYCL_DYNAMIC_REQUIRED_PRECISION allows.
स्थिर तार्किक fg_mode_meets(पूर्णांक mode, int32_t src1_prec) {
    जर (src1_prec == GGML_PREC_UNDEFINED || src1_prec >= GGML_PREC_F16) {
        परत सत्य;
    }
    जर (src1_prec >= GGML_PREC_BF16) {
        परत mode != GGML_SYCL_DYNAMIC_PRECISION_F16;
    }
    निवड (g_ggml_sycl_dynamic_required_precision) {
        स्थिती GGML_SYCL_DYNAMIC_PRECISION_TF32: परत mode == GGML_SYCL_DYNAMIC_PRECISION_TF32;
        स्थिती GGML_SYCL_DYNAMIC_PRECISION_BF16: परत mode != GGML_SYCL_DYNAMIC_PRECISION_F16;
        पूर्वनिर्धारित:                               परत असत्य;
    }
}

// Best allowed combination for this call, or -1 if none. The type is GGML_SYCL_DYNAMIC_PRECISION if it
// meets the src1 request; if not, bf16 then tf32 for a BF16 request (fastest first) and tf32 then bf16
// for an F32 request (most mantissa first).
स्थिर पूर्णांक fg_pick_combo(dpct::queue_ptr stream, ggml_type src1_type, int32_t src1_prec) {
    स्थिरांक sycl::device dev     = stream->get_device();
    स्थिरांक पूर्णांक          allowed = fg_device_combos(dev) & g_ggml_sycl_xmx_gather_shapes;
    स्थिरांक तार्किक         f32_req = src1_prec != GGML_PREC_UNDEFINED && src1_prec < GGML_PREC_BF16;
    स्थिरांक पूर्णांक          modes[] = {
        g_ggml_sycl_dynamic_precision,
        f32_req ? GGML_SYCL_DYNAMIC_PRECISION_TF32 : GGML_SYCL_DYNAMIC_PRECISION_BF16,
        f32_req ? GGML_SYCL_DYNAMIC_PRECISION_BF16 : GGML_SYCL_DYNAMIC_PRECISION_TF32,
    };
    पूर्णांक     best      = -1;
    int64_t best_rank = 0;
    साठी (पूर्णांक i = 0; i < 3 && best < 0; ++i) {
        स्थिरांक पूर्णांक mode = modes[i];
        जर (!fg_mode_meets(mode, src1_prec)) {
            सुरूठेवा;
        }
        साठी (पूर्णांक idx = 0; idx < FG_N_COMBOS; ++idx) {
            जर (!(allowed & (1 << idx))) {
                सुरूठेवा;
            }
            fg_visit_combo(idx, [&](स्वतः s) {
                using S = decltype(s);
                जर (S::EA::mode != mode || S::EB::mode != mode) {
                    परत;
                }
                स्थिरांक int64_t rank = fg_rank<S>(src1_type);
                जर (best < 0 || rank < best_rank) {
                    best      = idx;
                    best_rank = rank;
                }
            });
        }
    }
    // log each distinct decision once
    स्थिर std::mutex                                      mtx;
    स्थिर std::set<std::tuple<माप_प्रकार, पूर्णांक, int32_t, पूर्णांक>> seen;
    std::lock_guard<std::mutex>                            lock(mtx);
    जर (seen.emplace(std::hash<sycl::device>{}(dev), (पूर्णांक) src1_type, src1_prec, best).second) {
        GGML_LOG_INFO("%s: src1 %s, src1 prec %d -> %s\n", __func__, ggml_type_name(src1_type), src1_prec,
                      best >= 0 ? fg_combo_name(best).c_str() : "none (library GEMM)");
    }
    परत best;
}

// src1 [N][K] -> packed [K/V][Npad][V] so B tiles load straight from global memory
ढाँचा <typename E, typename T_src>
स्थिर शून्य fused_gemm_pack_b(स्थिरांक T_src * y, typename E::store * packed, पूर्णांक N, पूर्णांक Npad, पूर्णांक K,
                              dpct::queue_ptr stream) {
    constexpr पूर्णांक V   = 4 / आकार(typename E::store);
    स्थिरांक पूर्णांक     kqs = K / V;
    stream->parallel_for(sycl::range<1>((माप_प्रकार) Npad * kqs), [=](sycl::id<1> id) {
        स्थिरांक पूर्णांक idx = id[0];
        स्थिरांक पूर्णांक n   = idx / kqs;
        स्थिरांक पूर्णांक kq  = idx - n * kqs;
        typename E::store vals[V] = {};
        जर (n < N) {
            स्थिरांक T_src * src = y + (माप_प्रकार) n * K + V * kq;
#pragma unroll
            साठी (पूर्णांक v = 0; v < V; ++v) {
                vals[v] = E::cvt((दशांश) src[v]);
            }
        }
        typename E::store * out = packed + ((माप_प्रकार) kq * Npad + n) * V;
#pragma unroll
        साठी (पूर्णांक v = 0; v < V; ++v) {
            out[v] = vals[v];
        }
    });
}

// A stage: one lane owns one row and decodes FG_BK values of it per k step, with every scale
// folded into the value so the mad below sees plain A elements. One overload per weight format.
ढाँचा <typename E>
स्थिर __dpct_inline__ शून्य fg_stage_a(स्थिरांक block_iq4_nl * __restrict__ xrow, स्थिरांक पूर्णांक kb, typename E::pair * a) {
    स्थिरांक block_iq4_nl blk = xrow[kb];
    स्थिरांक दशांश        d   = (दशांश) blk.d;
#pragma unroll
    साठी (पूर्णांक j = 0; j < QK4_NL / 2; j += 2) {
        स्थिरांक uint8_t q0 = blk.qs[j];
        स्थिरांक uint8_t q1 = blk.qs[j + 1];
        a[j / 2]     = E::make(d * kvalues_iq4nl[q0 & 0xf], d * kvalues_iq4nl[q1 & 0xf]);
        a[j / 2 + 8] = E::make(d * kvalues_iq4nl[q0 >> 4], d * kvalues_iq4nl[q1 >> 4]);
    }
}

// iq3_s: k step kb is sub-block kb % 8 of superblock kb / 8. The superblock is 110 bytes, so read
// only the fields of that sub-block instead of copying the block. Same decode as
// dequantize_block_iq3_s: grid entries are taken as dwords and the sign bit is a plain shift.
ढाँचा <typename E>
स्थिर __dpct_inline__ शून्य fg_stage_a(स्थिरांक block_iq3_s * __restrict__ xrow, स्थिरांक पूर्णांक kb, typename E::pair * a) {
    static_assert(QK_K == 256, "the iq3_s A stage assumes 8 sub-blocks per superblock");
    स्थिरांक block_iq3_s * blk = xrow + kb / (QK_K / 32);
    स्थिरांक पूर्णांक           ib8 = kb % (QK_K / 32);
    स्थिरांक uint8_t *     qs  = blk->qs + 8 * ib8;
    स्थिरांक पूर्णांक           qh  = blk->qh[ib8];
    स्थिरांक दशांश         d   = (दशांश) blk->d * (1 + 2 * ((blk->scales[ib8 / 2] >> (4 * (ib8 % 2))) & 0xf));
#pragma unroll
    साठी (पूर्णांक il = 0; il < 4; ++il) {
        स्थिरांक uint32_t grid1 = iq3s_grid[qs[2 * il + 0] | ((qh << (8 - 2 * il)) & 256)];
        स्थिरांक uint32_t grid2 = iq3s_grid[qs[2 * il + 1] | ((qh << (7 - 2 * il)) & 256)];
        स्थिरांक पूर्णांक      signs = blk->signs[4 * ib8 + il];
#pragma unroll
        साठी (पूर्णांक j = 0; j < 2; ++j) {
            स्थिरांक दशांश g1a = (दशांश) ((grid1 >> (16 * j + 0)) & 0xff);
            स्थिरांक दशांश g1b = (दशांश) ((grid1 >> (16 * j + 8)) & 0xff);
            स्थिरांक दशांश g2a = (दशांश) ((grid2 >> (16 * j + 0)) & 0xff);
            स्थिरांक दशांश g2b = (दशांश) ((grid2 >> (16 * j + 8)) & 0xff);
            स्थिरांक पूर्णांक   s   = 2 * j;
            a[4 * il + j]     = E::make(d * ((signs & (1 << (s + 0))) ? -g1a : g1a),
                                        d * ((signs & (1 << (s + 1))) ? -g1b : g1b));
            a[4 * il + j + 2] = E::make(d * ((signs & (1 << (s + 4))) ? -g2a : g2a),
                                        d * ((signs & (1 << (s + 5))) ? -g2b : g2b));
        }
    }
}

// values per stored block, so a row of K values is K/qk blocks
ढाँचा <typename block_q_t> संरचना fg_block_traits;
ढाँचा <> संरचना fg_block_traits<block_iq4_nl> { स्थिर constexpr पूर्णांक qk = QK4_NL; };
ढाँचा <> संरचना fg_block_traits<block_iq3_s>  { स्थिर constexpr पूर्णांक qk = QK_K; };
ढाँचा <> संरचना fg_block_traits<block_iq3_xxs> { स्थिर constexpr पूर्णांक qk = QK_K; };
ढाँचा <> संरचना fg_block_traits<block_iq4_xs>  { स्थिर constexpr पूर्णांक qk = QK_K; };
ढाँचा <> संरचना fg_block_traits<block_iq2_xxs> { स्थिर constexpr पूर्णांक qk = QK_K; };
ढाँचा <> संरचना fg_block_traits<block_iq2_xs>  { स्थिर constexpr पूर्णांक qk = QK_K; };
ढाँचा <> संरचना fg_block_traits<block_iq2_s>   { स्थिर constexpr पूर्णांक qk = QK_K; };
ढाँचा <> संरचना fg_block_traits<block_iq1_s>   { स्थिर constexpr पूर्णांक qk = QK_K; };
ढाँचा <> संरचना fg_block_traits<block_iq1_m>   { स्थिर constexpr पूर्णांक qk = QK_K; };
ढाँचा <> संरचना fg_block_traits<block_q8_0>    { स्थिर constexpr पूर्णांक qk = QK8_0; };
ढाँचा <> संरचना fg_block_traits<block_q4_K>    { स्थिर constexpr पूर्णांक qk = QK_K; };
ढाँचा <> संरचना fg_block_traits<block_q5_K>    { स्थिर constexpr पूर्णांक qk = QK_K; };
ढाँचा <> संरचना fg_block_traits<block_q6_K>    { स्थिर constexpr पूर्णांक qk = QK_K; };

// The A stages below are the dequantize_block_iq* kernels rewritten for one k step. There a
// work-item handled one quarter (il) of one 32-wide sub-block (ib); here one lane produces the
// whole step, so il becomes a loop and ib is kb inside the superblock. Each quarter yields 8
// consecutive values, i.e. 4 pairs at a[4*il], so nothing larger than 8 floats is ever live.
#define FG_SUPERBLOCK(T)                                                         \
    static_assert(QK_K == 256, "the " #T " A stage assumes 8 sub-blocks per superblock"); \
    स्थिरांक T * blk = xrow + kb / (QK_K / 32);                                     \
    स्थिरांक पूर्णांक ib  = kb % (QK_K / 32)

ढाँचा <typename E>
स्थिर __dpct_inline__ शून्य fg_pack_quarter(स्थिरांक दशांश * __restrict__ t, typename E::pair * a, पूर्णांक il) {
#pragma unroll
    साठी (पूर्णांक j = 0; j < 4; ++j) {
        a[4 * il + j] = E::make(t[2 * j], t[2 * j + 1]);
    }
}

ढाँचा <typename E>
स्थिर __dpct_inline__ शून्य fg_stage_a(स्थिरांक block_iq4_xs * __restrict__ xrow, स्थिरांक पूर्णांक kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_iq4_xs);
    // low nibbles fill the first half of the step, high nibbles the second, so the two halves
    // land at a[0..7] and a[8..15] and no quarter loop is needed
    स्थिरांक दशांश d = (दशांश) blk->d *
        ((((blk->scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) | (((blk->scales_h >> (2 * ib)) & 3) << 4)) - 32);
    स्थिरांक uint8_t * q4 = blk->qs + 16 * ib;
#pragma unroll
    साठी (पूर्णांक j = 0; j < 8; ++j) {
        a[j]     = E::make(d * kvalues_iq4nl[q4[2 * j] & 0xf], d * kvalues_iq4nl[q4[2 * j + 1] & 0xf]);
        a[8 + j] = E::make(d * kvalues_iq4nl[q4[2 * j] >> 4],  d * kvalues_iq4nl[q4[2 * j + 1] >> 4]);
    }
}

ढाँचा <typename E>
स्थिर __dpct_inline__ शून्य fg_stage_a(स्थिरांक block_iq3_xxs * __restrict__ xrow, स्थिरांक पूर्णांक kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_iq3_xxs);
    स्थिरांक uint8_t *  q3    = blk->qs + 8 * ib;
    स्थिरांक uint16_t * gas   = (स्थिरांक uint16_t *) (blk->qs + QK_K / 4) + 2 * ib;
    स्थिरांक uint32_t   aux32 = gas[0] | (gas[1] << 16);
    स्थिरांक दशांश      d     = (दशांश) blk->d * (0.5f + (aux32 >> 28)) * 0.5f;
#pragma unroll
    साठी (पूर्णांक il = 0; il < 4; ++il) {
        स्थिरांक uint8_t * grid1 = (स्थिरांक uint8_t *) (iq3xxs_grid + q3[2 * il + 0]);
        स्थिरांक uint8_t * grid2 = (स्थिरांक uint8_t *) (iq3xxs_grid + q3[2 * il + 1]);
        स्थिरांक uint8_t   signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];
        दशांश t[8];
#pragma unroll
        साठी (पूर्णांक j = 0; j < 4; ++j) {
            t[j + 0] = d * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f);
            t[j + 4] = d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

ढाँचा <typename E>
स्थिर __dpct_inline__ शून्य fg_stage_a(स्थिरांक block_iq2_xxs * __restrict__ xrow, स्थिरांक पूर्णांक kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_iq2_xxs);
    स्थिरांक uint16_t * q2    = blk->qs + 4 * ib;
    स्थिरांक uint8_t *  aux8  = (स्थिरांक uint8_t *) q2;
    स्थिरांक uint32_t   aux32 = q2[2] | (q2[3] << 16);
    स्थिरांक दशांश      d     = (दशांश) blk->d * (0.5f + (aux32 >> 28)) * 0.25f;
#pragma unroll
    साठी (पूर्णांक il = 0; il < 4; ++il) {
        स्थिरांक uint8_t * grid  = (स्थिरांक uint8_t *) (iq2xxs_grid + aux8[il]);
        स्थिरांक uint8_t   signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];
        दशांश t[8];
#pragma unroll
        साठी (पूर्णांक j = 0; j < 8; ++j) {
            t[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

ढाँचा <typename E>
स्थिर __dpct_inline__ शून्य fg_stage_a(स्थिरांक block_iq2_xs * __restrict__ xrow, स्थिरांक पूर्णांक kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_iq2_xs);
    स्थिरांक uint16_t * q2 = blk->qs + 4 * ib;
#pragma unroll
    साठी (पूर्णांक il = 0; il < 4; ++il) {
        स्थिरांक uint8_t * grid  = (स्थिरांक uint8_t *) (iq2xs_grid + (q2[il] & 511));
        स्थिरांक दशांश     d     = (दशांश) blk->d * (0.5f + ((blk->scales[ib] >> (4 * (il / 2))) & 0xf)) * 0.25f;
        स्थिरांक uint8_t   signs = ksigns_iq2xs[q2[il] >> 9];
        दशांश t[8];
#pragma unroll
        साठी (पूर्णांक j = 0; j < 8; ++j) {
            t[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

ढाँचा <typename E>
स्थिर __dpct_inline__ शून्य fg_stage_a(स्थिरांक block_iq2_s * __restrict__ xrow, स्थिरांक पूर्णांक kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_iq2_s);
#pragma unroll
    साठी (पूर्णांक il = 0; il < 4; ++il) {
        स्थिरांक uint8_t * grid =
            (स्थिरांक uint8_t *) (iq2s_grid + (blk->qs[4 * ib + il] | ((blk->qh[ib] << (8 - 2 * il)) & 0x300)));
        स्थिरांक दशांश   d     = (दशांश) blk->d * (0.5f + ((blk->scales[ib] >> (4 * (il / 2))) & 0xf)) * 0.25f;
        स्थिरांक uint8_t signs = blk->qs[QK_K / 8 + 4 * ib + il];
        दशांश t[8];
#pragma unroll
        साठी (पूर्णांक j = 0; j < 8; ++j) {
            t[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

ढाँचा <typename E>
स्थिर __dpct_inline__ शून्य fg_stage_a(स्थिरांक block_iq1_s * __restrict__ xrow, स्थिरांक पूर्णांक kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_iq1_s);
    स्थिरांक दशांश delta = blk->qh[ib] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
    स्थिरांक दशांश d     = (दशांश) blk->d * (2 * ((blk->qh[ib] >> 12) & 7) + 1);
#pragma unroll
    साठी (पूर्णांक il = 0; il < 4; ++il) {
        uint32_t       grid32[2];
        स्थिरांक int8_t * q = (स्थिरांक int8_t *) grid32;
        grid32[0] = iq1s_grid_gpu[blk->qs[4 * ib + il] | (((blk->qh[ib] >> (3 * il)) & 7) << 8)];
        grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
        grid32[0] &= 0x0f0f0f0f;
        दशांश t[8];
#pragma unroll
        साठी (पूर्णांक j = 0; j < 8; ++j) {
            t[j] = d * (q[j] + delta);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

ढाँचा <typename E>
स्थिर __dpct_inline__ शून्य fg_stage_a(स्थिरांक block_iq1_m * __restrict__ xrow, स्थिरांक पूर्णांक kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_iq1_m);
    स्थिरांक uint16_t * sc = (स्थिरांक uint16_t *) blk->scales;
    iq1m_scale_t     scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
#pragma unroll
    साठी (पूर्णांक il = 0; il < 4; ++il) {
        स्थिरांक पूर्णांक   ib16  = 2 * ib + il / 2;
        स्थिरांक दशांश d     = (दशांश) scale.f16 * (2 * ((sc[ib16 / 4] >> (3 * (ib16 % 4))) & 0x7) + 1);
        स्थिरांक दशांश delta = blk->qh[2 * ib + il / 2] & (0x08 << (4 * (il % 2))) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
        uint32_t       grid32[2];
        स्थिरांक int8_t * q = (स्थिरांक int8_t *) grid32;
        grid32[0] = iq1s_grid_gpu[blk->qs[4 * ib + il] |
                                  (((blk->qh[2 * ib + il / 2] >> (4 * (il % 2))) & 7) << 8)];
        grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
        grid32[0] &= 0x0f0f0f0f;
        दशांश t[8];
#pragma unroll
        साठी (पूर्णांक j = 0; j < 8; ++j) {
            t[j] = d * (q[j] + delta);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}


// q8_0 and the k-quants. Each decode takes the fields of one 32-value sub-block, so the canonical
// layout and the reorder (SoA) layout of reorder_qw() share it and differ only in where the fields
// live, as in dequantize.hpp.
ढाँचा <typename E>
स्थिर __dpct_inline__ शून्य fg_decode_q8_0(स्थिरांक int8_t * __restrict__ qs, स्थिरांक दशांश d, typename E::pair * a) {
#pragma unroll
    साठी (पूर्णांक j = 0; j < QK8_0 / 2; ++j) {
        a[j] = E::make(d * qs[2 * j], d * qs[2 * j + 1]);
    }
}

// same unpack as get_scale_min_k4() in dequantize.hpp
स्थिर __dpct_inline__ शून्य fg_scale_min_k4(स्थिरांक पूर्णांक j, स्थिरांक uint8_t * __restrict__ q, uint8_t & d, uint8_t & m) {
    जर (j < 4) {
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
स्थिर __dpct_inline__ शून्य fg_decode_q4_K(स्थिरांक uint8_t * __restrict__ qs, स्थिरांक uint8_t * __restrict__ scales,
                                           स्थिरांक sycl::half2 dm, स्थिरांक पूर्णांक ib, typename E::pair * a) {
    uint8_t sc, mb;
    fg_scale_min_k4(ib, scales, sc, mb);
    स्थिरांक दशांश     d     = (दशांश) dm[0] * sc;
    स्थिरांक दशांश     m     = (दशांश) dm[1] * mb;
    स्थिरांक uint8_t * q     = qs + 32 * (ib / 2);
    स्थिरांक पूर्णांक       shift = 4 * (ib % 2);
#pragma unroll
    साठी (पूर्णांक j = 0; j < 16; ++j) {
        a[j] = E::make(d * ((q[2 * j] >> shift) & 0xF) - m, d * ((q[2 * j + 1] >> shift) & 0xF) - m);
    }
}

// q5_K: q4_K plus one high bit per value, bit ib of qh[l]
ढाँचा <typename E>
स्थिर __dpct_inline__ शून्य fg_decode_q5_K(स्थिरांक uint8_t * __restrict__ qs, स्थिरांक uint8_t * __restrict__ qh,
                                           स्थिरांक uint8_t * __restrict__ scales, स्थिरांक sycl::half2 dm, स्थिरांक पूर्णांक ib,
                                           typename E::pair * a) {
    uint8_t sc, mb;
    fg_scale_min_k4(ib, scales, sc, mb);
    स्थिरांक दशांश     d     = (दशांश) dm[0] * sc;
    स्थिरांक दशांश     m     = (दशांश) dm[1] * mb;
    स्थिरांक uint8_t * q     = qs + 32 * (ib / 2);
    स्थिरांक पूर्णांक       shift = 4 * (ib % 2);
#pragma unroll
    साठी (पूर्णांक j = 0; j < 16; ++j) {
        स्थिरांक पूर्णांक l = 2 * j;
        a[j] = E::make(d * (((q[l] >> shift) & 0xF) | (((qh[l] >> ib) & 1) << 4)) - m,
                       d * (((q[l + 1] >> shift) & 0xF) | (((qh[l + 1] >> ib) & 1) << 4)) - m);
    }
}

// q6_K: sub-block ib is quarter r = ib % 4 of half h = ib / 4. The half selects ql + 64h, qh + 32h
// and scales + 8h; the quarter selects ql + 32(r & 1), the ql nibble r / 2, the qh bit pair r and
// scales + 2r, as in dequantize_row_q6_K. The scale changes at value 16 of the sub-block.
ढाँचा <typename E>
स्थिर __dpct_inline__ शून्य fg_decode_q6_K(स्थिरांक uint8_t * __restrict__ ql, स्थिरांक uint8_t * __restrict__ qh,
                                           स्थिरांक int8_t * __restrict__ scales, स्थिरांक दशांश d, स्थिरांक पूर्णांक ib,
                                           typename E::pair * a) {
    स्थिरांक पूर्णांक       h  = ib / 4;
    स्थिरांक पूर्णांक       r  = ib % 4;
    स्थिरांक uint8_t * q  = ql + 64 * h + 32 * (r & 1);
    स्थिरांक uint8_t * hb = qh + 32 * h;
    स्थिरांक int8_t *  sc = scales + 8 * h + 2 * r;
#pragma unroll
    साठी (पूर्णांक j = 0; j < 16; ++j) {
        स्थिरांक पूर्णांक   l  = 2 * j;
        स्थिरांक दशांश dl = d * sc[j / 8];
        स्थिरांक पूर्णांक   q0 = (((q[l] >> (4 * (r / 2))) & 0xF) | (((hb[l] >> (2 * r)) & 3) << 4)) - 32;
        स्थिरांक पूर्णांक   q1 = (((q[l + 1] >> (4 * (r / 2))) & 0xF) | (((hb[l + 1] >> (2 * r)) & 3) << 4)) - 32;
        a[j] = E::make(dl * q0, dl * q1);
    }
}

ढाँचा <typename E>
स्थिर __dpct_inline__ शून्य fg_stage_a(स्थिरांक block_q8_0 * __restrict__ xrow, स्थिरांक पूर्णांक kb, typename E::pair * a) {
    fg_decode_q8_0<E>(xrow[kb].qs, (दशांश) xrow[kb].d, a);
}

ढाँचा <typename E>
स्थिर __dpct_inline__ शून्य fg_stage_a(स्थिरांक block_q4_K * __restrict__ xrow, स्थिरांक पूर्णांक kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_q4_K);
    fg_decode_q4_K<E>(blk->qs, blk->scales, blk->dm, ib, a);
}

ढाँचा <typename E>
स्थिर __dpct_inline__ शून्य fg_stage_a(स्थिरांक block_q5_K * __restrict__ xrow, स्थिरांक पूर्णांक kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_q5_K);
    fg_decode_q5_K<E>(blk->qs, blk->qh, blk->scales, blk->dm, ib, a);
}

ढाँचा <typename E>
स्थिर __dpct_inline__ शून्य fg_stage_a(स्थिरांक block_q6_K * __restrict__ xrow, स्थिरांक पूर्णांक kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_q6_K);
    fg_decode_q6_K<E>(blk->ql, blk->qh, blk->scales, (दशांश) blk->d, ib, a);
}

// Reorder (SoA) layout: each block field is one stream over the nblocks of the matrix (of the expert
// slice for MUL_MAT_ID), in the order reorder_qw() writes them. Block ib holds k step kb.
ढाँचा <typename block_q_t> संरचना fg_soa {
    स्थिर constexpr तार्किक supported = असत्य;
};

ढाँचा <> संरचना fg_soa<block_q8_0> {
    स्थिर constexpr तार्किक supported = सत्य;
    // [qs][d]
    ढाँचा <typename E>
    स्थिर __dpct_inline__ शून्य stage(स्थिरांक uint8_t * x, स्थिरांक माप_प्रकार nblocks, स्थिरांक माप_प्रकार ib, स्थिरांक पूर्णांक,
                                      typename E::pair * a) {
        स्थिरांक दशांश d = (दशांश) ((स्थिरांक sycl::half *) (x + nblocks * QK8_0))[ib];
        fg_decode_q8_0<E>((स्थिरांक int8_t *) x + ib * QK8_0, d, a);
    }
};

ढाँचा <> संरचना fg_soa<block_q4_K> {
    स्थिर constexpr तार्किक supported = सत्य;
    // [qs][scales][dm]
    ढाँचा <typename E>
    स्थिर __dpct_inline__ शून्य stage(स्थिरांक uint8_t * x, स्थिरांक माप_प्रकार nblocks, स्थिरांक माप_प्रकार ib, स्थिरांक पूर्णांक kb,
                                      typename E::pair * a) {
        स्थिरांक uint8_t *   scales = x + nblocks * (QK_K / 2);
        स्थिरांक sycl::half2 dm     = ((स्थिरांक sycl::half2 *) (scales + nblocks * K_SCALE_SIZE))[ib];
        fg_decode_q4_K<E>(x + ib * (QK_K / 2), scales + ib * K_SCALE_SIZE, dm, kb % (QK_K / 32), a);
    }
};

ढाँचा <> संरचना fg_soa<block_q5_K> {
    स्थिर constexpr तार्किक supported = सत्य;
    // [qs][qh][scales][dm]
    ढाँचा <typename E>
    स्थिर __dpct_inline__ शून्य stage(स्थिरांक uint8_t * x, स्थिरांक माप_प्रकार nblocks, स्थिरांक माप_प्रकार ib, स्थिरांक पूर्णांक kb,
                                      typename E::pair * a) {
        स्थिरांक uint8_t *   qh     = x + nblocks * (QK_K / 2);
        स्थिरांक uint8_t *   scales = qh + nblocks * (QK_K / 8);
        स्थिरांक sycl::half2 dm     = ((स्थिरांक sycl::half2 *) (scales + nblocks * K_SCALE_SIZE))[ib];
        fg_decode_q5_K<E>(x + ib * (QK_K / 2), qh + ib * (QK_K / 8), scales + ib * K_SCALE_SIZE, dm,
                          kb % (QK_K / 32), a);
    }
};

ढाँचा <> संरचना fg_soa<block_q6_K> {
    स्थिर constexpr तार्किक supported = सत्य;
    // [ql][qh][scales][d]
    ढाँचा <typename E>
    स्थिर __dpct_inline__ शून्य stage(स्थिरांक uint8_t * x, स्थिरांक माप_प्रकार nblocks, स्थिरांक माप_प्रकार ib, स्थिरांक पूर्णांक kb,
                                      typename E::pair * a) {
        स्थिरांक uint8_t * qh     = x + nblocks * (QK_K / 2);
        स्थिरांक uint8_t * scales = qh + nblocks * (QK_K / 4);
        स्थिरांक दशांश     d      = (दशांश) ((स्थिरांक sycl::half *) (scales + nblocks * (QK_K / 16)))[ib];
        fg_decode_q6_K<E>(x + ib * (QK_K / 2), qh + ib * (QK_K / 4), (स्थिरांक int8_t *) scales + ib * (QK_K / 16), d,
                          kb % (QK_K / 32), a);
    }
};


// one SG_ROWS x BN output tile: B columns [b0, b0 + BN) of packed_b go to dst columns [n0, n1),
// n1 - n0 <= BN
ढाँचा <typename S, typename block_q_t, तार्किक reordered>
स्थिर शून्य fused_dequant_gemm_tile(
    स्थिरांक block_q_t * __restrict__ x,
    स्थिरांक typename S::tsb * __restrict__ packed_b,
    दशांश * __restrict__ dst,
    स्थिरांक पूर्णांक M, स्थिरांक पूर्णांक Npad, स्थिरांक पूर्णांक K, स्थिरांक पूर्णांक ldd,
    स्थिरांक पूर्णांक b0, स्थिरांक पूर्णांक n0, स्थिरांक पूर्णांक n1,
    sycl::local_accessor<typename S::tsa, 1> tile_a,
    sycl::local_accessor<दशांश, 1> tile_c,
    स्थिरांक sycl::nd_item<2> & item) {
    जर constexpr (fg_built<S::SG>()) {
        using EA = typename S::EA;
        using TA = typename S::ta;
        using TB = typename S::tb;
        स्थिरांक स्वतः sg     = item.get_sub_group();
        स्थिरांक पूर्णांक  sg_id  = sg.get_group_id()[0];
        स्थिरांक पूर्णांक  lane   = sg.get_local_id()[0];
        स्थिरांक पूर्णांक  m0     = item.get_group(1) * S::SG_ROWS;
        स्थिरांक पूर्णांक  nstep  = K / FG_BK;
        स्थिरांक पूर्णांक  a_base = sg_id * S::SG_ROWS * FG_BK;
        स्थिरांक पूर्णांक  c_base = sg_id * S::SG_ROWS * S::BN;

        mx::joint_matrix<sycl::sub_group, दशांश, mx::use::accumulator, S::TM, S::TN> acc[S::MT][S::NT];
#pragma unroll
        साठी (पूर्णांक mt = 0; mt < S::MT; ++mt) {
#pragma unroll
            साठी (पूर्णांक nt = 0; nt < S::NT; ++nt) {
                mx::joint_matrix_fill(sg, acc[mt][nt], 0.0f);
            }
        }

        // lane decodes rows lane, lane + SG, ... of the sub-group's SG_ROWS
        constexpr पूर्णांक         KPB     = fg_block_traits<block_q_t>::qk / FG_BK; // k steps per block
        स्थिरांक माप_प्रकार          bpr     = K / fg_block_traits<block_q_t>::qk;
        स्थिरांक माप_प्रकार          nblocks = (माप_प्रकार) M * bpr;
        माप_प्रकार                row_blk[S::RPL];
        तार्किक                  row_ok[S::RPL];
        typename EA::pair *   a[S::RPL];
#pragma unroll
        साठी (पूर्णांक r = 0; r < S::RPL; ++r) {
            स्थिरांक पूर्णांक row = m0 + r * S::SG + lane;
            row_ok[r]  = row < M;
            row_blk[r] = (माप_प्रकार) (row_ok[r] ? row : 0) * bpr;
            a[r]       = (typename EA::pair *) &tile_a[a_base + (r * S::SG + lane) * FG_BK];
        }

        स्थिरांक स्वतः b_ptr = sycl::address_space_cast<sycl::access::address_space::global_space,
                                                    sycl::access::decorated::no>(packed_b);
        स्थिरांक पूर्णांक b_stride = Npad * S::VNNI;

        स्थिरांक पूर्णांक kb_begin = (sg_id * nstep) / FG_KSPLIT;
        स्थिरांक पूर्णांक kb_end   = ((sg_id + 1) * nstep) / FG_KSPLIT;
        साठी (पूर्णांक kb = kb_begin; kb < kb_end; ++kb) {
#pragma unroll
            साठी (पूर्णांक r = 0; r < S::RPL; ++r) {
                जर (row_ok[r]) {
                    जर constexpr (reordered) {
                        fg_soa<block_q_t>::ढाँचा stage<EA>((स्थिरांक uint8_t *) x, nblocks, row_blk[r] + kb / KPB, kb,
                                                              a[r]);
                    } अन्यथा {
                        fg_stage_a<EA>(x + row_blk[r], kb, a[r]);
                    }
                } अन्यथा {
#pragma unroll
                    साठी (पूर्णांक j = 0; j < FG_BK / 2; ++j) {
                        a[r][j] = EA::make(0.0f, 0.0f);
                    }
                }
            }
            sycl::group_barrier(sg);

#pragma unroll
            साठी (पूर्णांक kt = 0; kt < FG_BK / S::TK; ++kt) {
                स्थिरांक पूर्णांक kq0 = (kb * FG_BK + kt * S::TK) / S::VNNI;
                mx::joint_matrix<sycl::sub_group, TB, mx::use::b, S::TK, S::TN, S::b_layout> sub_b[S::NT];
#pragma unroll
                साठी (पूर्णांक nt = 0; nt < S::NT; ++nt) {
                    mx::joint_matrix_load(sg, sub_b[nt], b_ptr + (माप_प्रकार) kq0 * b_stride + (b0 + nt * S::TN) * S::VNNI, b_stride);
                }
#pragma unroll
                साठी (पूर्णांक mt = 0; mt < S::MT; ++mt) {
                    mx::joint_matrix<sycl::sub_group, TA, mx::use::a, S::TM, S::TK, mx::layout::row_major> sub_a;
                    mx::joint_matrix_load(sg, sub_a,
                        tile_a.ढाँचा get_multi_ptr<sycl::access::decorated::no>() + a_base + (mt * S::TM) * FG_BK + kt * S::TK,
                        FG_BK);
#pragma unroll
                    साठी (पूर्णांक nt = 0; nt < S::NT; ++nt) {
                        mx::joint_matrix_mad(sg, acc[mt][nt], sub_a, sub_b[nt], acc[mt][nt]);
                    }
                }
            }
            // the next step overwrites tile_a
            sycl::group_barrier(sg);
        }

#pragma unroll
        साठी (पूर्णांक mt = 0; mt < S::MT; ++mt) {
#pragma unroll
            साठी (पूर्णांक nt = 0; nt < S::NT; ++nt) {
                mx::joint_matrix_store(sg, acc[mt][nt],
                    tile_c.ढाँचा get_multi_ptr<sycl::access::decorated::no>() + c_base + (mt * S::TM) * S::BN + nt * S::TN,
                    S::BN, mx::layout::row_major);
            }
        }
        sycl::group_barrier(item.get_group());

        // sum the K splits; consecutive lanes write consecutive rows of one dst column
        साठी (पूर्णांक idx = item.get_local_linear_id(); idx < S::SG_ROWS * S::BN; idx += S::WG_SIZE) {
            स्थिरांक पूर्णांक r = idx % S::SG_ROWS;
            स्थिरांक पूर्णांक c = idx / S::SG_ROWS;
            स्थिरांक पूर्णांक m = m0 + r;
            स्थिरांक पूर्णांक n = n0 + c;
            जर (m < M && n < n1) {
                दशांश sum = 0.0f;
#pragma unroll
                साठी (पूर्णांक s = 0; s < FG_KSPLIT; ++s) {
                    sum += tile_c[s * S::SG_ROWS * S::BN + r * S::BN + c];
                }
                dst[(माप_प्रकार) n * ldd + m] = sum;
            }
        }
    }
}

ढाँचा <typename S, typename block_q_t, तार्किक reordered = असत्य>
स्थिर शून्य fused_dequant_gemm_launch(स्थिरांक शून्य * src0, स्थिरांक typename S::tsb * packed, दशांश * dst, स्थिरांक पूर्णांक M,
                                      स्थिरांक पूर्णांक N, स्थिरांक पूर्णांक Npad, स्थिरांक पूर्णांक K, स्थिरांक पूर्णांक ldd,
                                      स्थिरांक int64_t groups_n, स्थिरांक int64_t groups_m, dpct::queue_ptr stream) {
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<typename S::tsa, 1> tile_a(FG_KSPLIT * S::SG_ROWS * FG_BK, cgh);
        sycl::local_accessor<दशांश, 1>          tile_c(FG_KSPLIT * S::SG_ROWS * S::BN, cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(groups_n, groups_m * S::WG_SIZE), sycl::range<2>(1, S::WG_SIZE)),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(S::SG)]] {
                स्थिरांक पूर्णांक n0 = item.get_group(0) * S::BN;
                fused_dequant_gemm_tile<S, block_q_t, reordered>((स्थिरांक block_q_t *) src0, packed, dst, M, Npad, K,
                                                      ldd, n0, n0, N, tile_a, tile_c, item);
            });
    });
}

// grouped: work-group (t, mt) is tile t of the schedule; its B columns sit at t * BN
ढाँचा <typename S, typename block_q_t, तार्किक reordered>
स्थिर शून्य grouped_dequant_gemm_launch(स्थिरांक अक्षर * src0_dd, स्थिरांक माप_प्रकार expert_stride,
                                        स्थिरांक ggml_sycl_gg_tile * tiles_ptr, स्थिरांक typename S::tsb * packed, दशांश * dst,
                                        स्थिरांक पूर्णांक M, स्थिरांक पूर्णांक Npad, स्थिरांक पूर्णांक K, स्थिरांक int64_t n_tiles,
                                        स्थिरांक int64_t groups_m, dpct::queue_ptr stream) {
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<typename S::tsa, 1> tile_a(FG_KSPLIT * S::SG_ROWS * FG_BK, cgh);
        sycl::local_accessor<दशांश, 1>          tile_c(FG_KSPLIT * S::SG_ROWS * S::BN, cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(n_tiles, groups_m * S::WG_SIZE), sycl::range<2>(1, S::WG_SIZE)),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(S::SG)]] {
                स्थिरांक पूर्णांक               t    = item.get_group(0);
                स्थिरांक ggml_sycl_gg_tile tile = tiles_ptr[t];
                स्थिरांक block_q_t *       x    = (स्थिरांक block_q_t *) (src0_dd + (माप_प्रकार) tile.expert * expert_stride);
                fused_dequant_gemm_tile<S, block_q_t, reordered>(x, packed, dst, M, Npad, K, M, t * S::BN, tile.n0,
                                                                 tile.n1, tile_a, tile_c, item);
            });
    });
}

// src1 f32 rows -> packed [K/V][n_tiles*BN][V], tile t holds its rows [n0, n1) at columns t*BN..,
// zero past n1. The column runs fastest so a sub-group writes one contiguous run.
ढाँचा <typename S>
स्थिर शून्य grouped_gemm_pack_b(स्थिरांक दशांश * y, typename S::tsb * packed, स्थिरांक ggml_sycl_gg_tile * tiles, पूर्णांक Npad,
                                पूर्णांक K, dpct::queue_ptr stream) {
    using E        = typename S::EB;
    constexpr पूर्णांक V = S::VNNI;
    स्थिरांक पूर्णांक kqs   = K / V;
    stream->parallel_for(sycl::range<1>((माप_प्रकार) Npad * kqs), [=](sycl::id<1> id) {
        स्थिरांक माप_प्रकार idx = id[0];
        स्थिरांक पूर्णांक    kq  = idx / Npad;
        स्थिरांक पूर्णांक    n   = idx - (माप_प्रकार) kq * Npad;
        स्थिरांक ggml_sycl_gg_tile tile = tiles[n / S::BN];
        स्थिरांक पूर्णांक    row = tile.n0 + n % S::BN;
        // one guarded load run per work-item, as a per-element select costs ~1.5% prefill
        typename S::tsb vals[V] = {};
        जर (row < tile.n1) {
            स्थिरांक दशांश * src = y + (माप_प्रकार) row * K + V * kq;
#pragma unroll
            साठी (पूर्णांक v = 0; v < V; ++v) {
                vals[v] = E::cvt(src[v]);
            }
        }
        typename S::tsb * out = packed + ((माप_प्रकार) kq * Npad + n) * V;
#pragma unroll
        साठी (पूर्णांक v = 0; v < V; ++v) {
            out[v] = vals[v];
        }
    });
}

// q8_0 and the k-quants take only the grouped path. The plain kernel decodes A again for every BN columns
// of a dense batch, and for these formats that costs more than the one dequantization of the library GEMM.
ढाँचा <typename T> स्थिर constexpr तार्किक fg_plain_ok() {
    परत !std::is_same_v<T, block_q8_0> && !std::is_same_v<T, block_q4_K> && !std::is_same_v<T, block_q5_K> &&
           !std::is_same_v<T, block_q6_K>;
}

ढाँचा <typename T, तार्किक R> संरचना fg_tag {
    using type                     = T;
    स्थिर constexpr तार्किक reordered = R;
};

// calls f(fg_tag<block_q_t, reordered>{}) for the weight format and layout; false if it has no A stage
ढाँचा <typename T, typename F> स्थिर तार्किक fg_visit_layout(तार्किक reordered, F && f) {
    जर (!reordered) {
        f(fg_tag<T, असत्य>{});
        परत सत्य;
    }
    जर constexpr (fg_soa<T>::supported) {
        f(fg_tag<T, सत्य>{});
        परत सत्य;
    }
    परत असत्य;
}

ढाँचा <typename F> स्थिर तार्किक fg_visit_type(ggml_type type, तार्किक reordered, F && f) {
    निवड (type) {
        स्थिती GGML_TYPE_IQ4_NL:  परत fg_visit_layout<block_iq4_nl>(reordered, f);
        स्थिती GGML_TYPE_IQ3_S:   परत fg_visit_layout<block_iq3_s>(reordered, f);
        स्थिती GGML_TYPE_IQ4_XS:  परत fg_visit_layout<block_iq4_xs>(reordered, f);
        स्थिती GGML_TYPE_IQ3_XXS: परत fg_visit_layout<block_iq3_xxs>(reordered, f);
        स्थिती GGML_TYPE_IQ2_XXS: परत fg_visit_layout<block_iq2_xxs>(reordered, f);
        स्थिती GGML_TYPE_IQ2_XS:  परत fg_visit_layout<block_iq2_xs>(reordered, f);
        स्थिती GGML_TYPE_IQ2_S:   परत fg_visit_layout<block_iq2_s>(reordered, f);
        स्थिती GGML_TYPE_IQ1_S:   परत fg_visit_layout<block_iq1_s>(reordered, f);
        स्थिती GGML_TYPE_IQ1_M:   परत fg_visit_layout<block_iq1_m>(reordered, f);
        स्थिती GGML_TYPE_Q8_0:    परत fg_visit_layout<block_q8_0>(reordered, f);
        स्थिती GGML_TYPE_Q4_K:    परत fg_visit_layout<block_q4_K>(reordered, f);
        स्थिती GGML_TYPE_Q5_K:    परत fg_visit_layout<block_q5_K>(reordered, f);
        स्थिती GGML_TYPE_Q6_K:    परत fg_visit_layout<block_q6_K>(reordered, f);
        पूर्वनिर्धारित:                परत असत्य;
    }
}

ढाँचा <typename S>
स्थिर तार्किक fg_fused_run(ggml_type src0_type, तार्किक src0_reordered, स्थिरांक शून्य * src0, स्थिरांक शून्य * src1, ggml_type src1_type,
                         दशांश * dst, int64_t M, int64_t N, int64_t K, int64_t ldd, ggml_sycl_pool & pool,
                         dpct::queue_ptr stream) {
    स्थिरांक int64_t groups_n = (N + S::BN - 1) / S::BN;
    स्थिरांक int64_t groups_m = (M + S::SG_ROWS - 1) / S::SG_ROWS;
    स्थिरांक पूर्णांक     Npad     = (पूर्णांक) (groups_n * S::BN);

    // src1 is read in its own type: one pass, converted in registers only if B differs
    ggml_sycl_pool_alloc<typename S::tsb> packed_b(pool, (माप_प्रकार) K * Npad);
    जर (src1_type == GGML_TYPE_F16) {
        fused_gemm_pack_b<typename S::EB>((स्थिरांक sycl::half *) src1, packed_b.get(), (पूर्णांक) N, Npad, (पूर्णांक) K, stream);
    } अन्यथा जर (src1_type == GGML_TYPE_BF16) {
        fused_gemm_pack_b<typename S::EB>((स्थिरांक fg_bf16 *) src1, packed_b.get(), (पूर्णांक) N, Npad, (पूर्णांक) K, stream);
    } अन्यथा {
        fused_gemm_pack_b<typename S::EB>((स्थिरांक दशांश *) src1, packed_b.get(), (पूर्णांक) N, Npad, (पूर्णांक) K, stream);
    }

    स्थिरांक typename S::tsb * packed = packed_b.get();
    परत fg_visit_type(src0_type, src0_reordered, [&](स्वतः tag) {
        using block_q_t = typename decltype(tag)::type;
        जर constexpr (fg_plain_ok<block_q_t>()) {
            fused_dequant_gemm_launch<S, block_q_t, decltype(tag)::reordered>(src0, packed, dst, (पूर्णांक) M, (पूर्णांक) N,
                                                                              Npad, (पूर्णांक) K, (पूर्णांक) ldd, groups_n,
                                                                              groups_m, stream);
        }
    });
}

तार्किक ggml_sycl_fused_dequant_gemm(ggml_type src0_type, तार्किक src0_reordered, स्थिरांक शून्य * src0, स्थिरांक शून्य * src1, ggml_type src1_type,
                                  int32_t src1_prec, दशांश * dst, int64_t M, int64_t N, int64_t K, int64_t ldd,
                                  ggml_sycl_pool & pool, dpct::queue_ptr stream) {
    // every BN columns dequantize A again, so wide N is left to the library GEMM.
    // src0_reordered: the persistent weight reorder may have been installed on this
    // tensor by an earlier small-batch mul_mat; the A stage then has to decode the
    // SoA layout. Types without an SoA stage (fg_soa<T>::supported == false)
    // decline via fg_visit_layout and the caller falls through to the
    // reorder-aware conversion route.
    तार्किक plain_ok = असत्य;
    fg_visit_type(src0_type, src0_reordered, [&](स्वतः tag) { plain_ok = fg_plain_ok<typename decltype(tag)::type>(); });
    जर (g_ggml_sycl_dynamic_precision == GGML_SYCL_DYNAMIC_PRECISION_F32 ||
        !ggml_sycl_xmx_gather_type_enabled(src0_type) || !plain_ok) {
        परत असत्य;
    }
    जर (src1_type != GGML_TYPE_F32 && src1_type != GGML_TYPE_F16 && src1_type != GGML_TYPE_BF16) {
        परत असत्य;
    }
    जर (!ggml_sycl_fused_dequant_gemm_shape_ok(src0_type, M, N, K, ldd)) {
        परत असत्य;
    }
    स्थिरांक पूर्णांक combo = fg_pick_combo(stream, src1_type, src1_prec);
    जर (combo < 0) {
        परत असत्य;
    }
    तार्किक launched = असत्य;
    fg_visit_combo(combo, [&](स्वतः s) {
        launched = fg_fused_run<decltype(s)>(src0_type, src0_reordered, src0, src1, src1_type, dst, M, N, K, ldd, pool, stream);
    });
    परत launched;
}

ढाँचा <typename S>
स्थिर तार्किक fg_grouped_run(ggml_type src0_type, तार्किक reordered, स्थिरांक शून्य * src0_base, माप_प्रकार expert_stride,
                           स्थिरांक दशांश * src1, दशांश * dst, स्थिरांक int64_t * expert_row_offsets, int64_t n_as, int64_t M,
                           int64_t K, std::vector<ggml_sycl_gg_tile> & tiles, ggml_sycl_pool & pool,
                           dpct::queue_ptr stream) {
    // the host knows every slice, so it lays out the work-groups: no search on the device
    tiles.clear();
    साठी (int64_t e = 0; e < n_as; ++e) {
        स्थिरांक int64_t end = expert_row_offsets[e + 1];
        साठी (int64_t n0 = expert_row_offsets[e]; n0 < end; n0 += S::BN) {
            tiles.push_back({ (int32_t) e, (int32_t) n0, (int32_t) std::min<int64_t>(n0 + S::BN, end) });
        }
    }
    स्थिरांक int64_t n_tiles  = tiles.size();
    स्थिरांक int64_t groups_m = (M + S::SG_ROWS - 1) / S::SG_ROWS;
    स्थिरांक पूर्णांक     Npad     = (पूर्णांक) (n_tiles * S::BN);

    स्थिरांक int64_t max_tiles = grouped_gemm_max_tiles(expert_row_offsets[n_as], n_as, S::BN);
    GGML_ASSERT(n_tiles <= max_tiles);
    ggml_sycl_pool_alloc<ggml_sycl_gg_tile> tiles_dev(pool, max_tiles);
    SYCL_CHECK(CHECK_TRY_ERROR(stream->स_नकल(tiles_dev.get(), tiles.data(), n_tiles * आकार(ggml_sycl_gg_tile))));

    ggml_sycl_pool_alloc<typename S::tsb> packed_b(pool, (माप_प्रकार) K * max_tiles * S::BN);
    grouped_gemm_pack_b<S>(src1, packed_b.get(), tiles_dev.get(), Npad, (पूर्णांक) K, stream);

    स्थिरांक typename S::tsb *   packed    = packed_b.get();
    स्थिरांक ggml_sycl_gg_tile * tiles_ptr = tiles_dev.get();
    स्थिरांक अक्षर *              src0_dd   = (स्थिरांक अक्षर *) src0_base;
    परत fg_visit_type(src0_type, reordered, [&](स्वतः tag) {
        using T = decltype(tag);
        grouped_dequant_gemm_launch<S, typename T::type, T::reordered>(src0_dd, expert_stride, tiles_ptr, packed, dst,
                                                                       (पूर्णांक) M, Npad, (पूर्णांक) K, n_tiles, groups_m,
                                                                       stream);
    });
}

तार्किक ggml_sycl_grouped_dequant_gemm(ggml_type src0_type, तार्किक reordered, स्थिरांक शून्य * src0_base, माप_प्रकार expert_stride,
                                    स्थिरांक दशांश * src1, int32_t src1_prec, दशांश * dst,
                                    स्थिरांक int64_t * expert_row_offsets, int64_t n_as, int64_t M, int64_t K,
                                    int64_t total_rows, std::vector<ggml_sycl_gg_tile> & tiles,
                                    ggml_sycl_pool & pool, dpct::queue_ptr stream) {
    int64_t n_active = 0;
    साठी (int64_t e = 0; e < n_as; ++e) {
        n_active += expert_row_offsets[e + 1] > expert_row_offsets[e];
    }
    जर (g_ggml_sycl_dynamic_precision == GGML_SYCL_DYNAMIC_PRECISION_F32 ||
        !ggml_sycl_xmx_gather_type_enabled(src0_type) || !fg_visit_type(src0_type, reordered, [](स्वतः) {})) {
        परत असत्य;
    }
    जर (!ggml_sycl_grouped_dequant_gemm_shape_ok(src0_type, M, K, total_rows, n_active)) {
        परत असत्य;
    }
    स्थिरांक पूर्णांक combo = fg_pick_combo(stream, GGML_TYPE_F32, src1_prec);
    जर (combo < 0) {
        परत असत्य;
    }
    तार्किक launched = असत्य;
    fg_visit_combo(combo, [&](स्वतः s) {
        launched = fg_grouped_run<decltype(s)>(src0_type, reordered, src0_base, expert_stride, src1, dst,
                                               expert_row_offsets, n_as, M, K, tiles, pool, stream);
    });
    परत launched;
}
