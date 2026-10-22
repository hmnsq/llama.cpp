#ضمن "fused-gemm.hpp"

#ضمن <sycl/ext/oneapi/matrix/matrix.hpp>

#ضمن <algorithm>
#ضمن <string>
#ضمن <tuple>
#ضمن <mutex>
#ضمن <set>
#ضمن <unordered_map>

نطاق mx = sycl::ext::oneapi::experimental::ميٽرڪس;

// FG_ / fg_ is short for fused GEMM: the weights are dequantized inside the GEMM, into the XMX tiles.

// A k step is one 32-value weight sub-block; iq3_s and the other superblock formats split their
// superblock into steps of this width. The sub-groups of a work-group each walk their own K range
// and are summed at the end.
ساكن constexpr صحيح FG_BK     = QK4_NL;
ساكن constexpr صحيح FG_KSPLIT = 4;

// Element traits of one joint_matrix operand type. The A stage and the B pack compute in f32 and
// convert once, in registers, when they write the element, so any type costs the same one pass.
//   store: storage in SLM (A) and in the packed B buffer
//   mtype: matrix_type in matrix_combinations
//   mode:  GGML_SYCL_DYNAMIC_PRECISION value that selects this type
//   src:   ggml type that needs no conversion into this type (GGML_TYPE_COUNT: none)
//   slow:  XMX throughput class, 0 is fastest. f16 and bf16 share the DPAS rate; tf32 does half the
//          K per instruction. B60, Qwen3-30B-A3B pp512: f16 1108, bf16 1000, tf32 751 t/s
سانچو <قسم_نالو T> بنية fg_elem;

سانچو <> بنية fg_elem<sycl::half> {
    استعمال store = sycl::half;
    استعمال pair  = sycl::half2;
    ساكن constexpr mx::ميٽرڪس_type mtype = mx::ميٽرڪس_type::fp16;
    ساكن constexpr صحيح             mode  = GGML_SYCL_DYNAMIC_PRECISION_F16;
    ساكن constexpr ggml_type       src   = GGML_TYPE_F16;
    ساكن constexpr صحيح             mant  = 10;
    ساكن constexpr صحيح             slow  = 0;
    ساكن store cvt(عائم x) { أرجع (store) x; }
    ساكن pair make(عائم x, عائم y) { أرجع pair((store) x, (store) y); }
};

// tf32 rounds to nearest even with plain bit ops: round_to_tf32 needs a SPIR-V extension that the
// DG2 AOT target rejects
ساكن مضمن uint32_t fg_ڦيرو_بٽ(عائم x, صحيح drop) {
    ثابت uint32_t u = sycl::ٻٽو_انگ_cast<uint32_t>(x);
    إذا ((u & 0x7f800000u) == 0x7f800000u) {
        أرجع (u & 0x7fffffu) ? u | (1u << drop) : u; // nan stays nan
    }
    أرجع u + ((1u << (drop - 1)) - 1) + ((u >> drop) & 1);
}

بنية alignas(4) fg_bf16x2 {
    sycl::ext::oneapi::bfloat16 x, y;
};

سانچو <> بنية fg_elem<sycl::ext::oneapi::bfloat16> {
    استعمال store = sycl::ext::oneapi::bfloat16;
    استعمال pair  = fg_bf16x2;
    ساكن constexpr mx::ميٽرڪس_type mtype = mx::ميٽرڪس_type::bf16;
    ساكن constexpr صحيح             mode  = GGML_SYCL_DYNAMIC_PRECISION_BF16;
    ساكن constexpr ggml_type       src   = GGML_TYPE_BF16;
    ساكن constexpr صحيح             mant  = 7;
    ساكن constexpr صحيح             slow  = 0;
    ساكن store cvt(عائم x) { أرجع store(x); }
    ساكن pair make(عائم x, عائم y) { أرجع { cvt(x), cvt(y) }; }
};

// tf32 keeps f32 range and f16 mantissa, in f32 storage
سانچو <> بنية fg_elem<mx::precision::tf32> {
    استعمال store = عائم;
    استعمال pair  = sycl::float2;
    ساكن constexpr mx::ميٽرڪس_type mtype = mx::ميٽرڪس_type::tf32;
    ساكن constexpr صحيح             mode  = GGML_SYCL_DYNAMIC_PRECISION_TF32;
    ساكن constexpr ggml_type       src   = GGML_TYPE_ڳڻپ;
    ساكن constexpr صحيح             mant  = 10;
    ساكن constexpr صحيح             slow  = 1;
    ساكن store cvt(عائم x) { أرجع sycl::ٻٽو_انگ_cast<عائم>(fg_ڦيرو_بٽ(x, 13) & ~0x1fffu); }
    ساكن pair make(عائم x, عائم y) { أرجع pair(cvt(x), cvt(y)); }
};

// One joint_matrix combination (A type, B type, TM x TN x TK, sub-group size; C and D are f32) and
// the tiling built on it. A sub-group owns SG_ROWS rows of A (at least 16) and BN (at least 32)
// columns of B. A and B may differ: the device lists the pairs it supports.
سانچو <قسم_نالو TA, قسم_نالو TB, صحيح TM_, صحيح TN_, صحيح TK_, صحيح SG_> بنية fg_combo {
    استعمال ta  = TA;
    استعمال tb  = TB;
    استعمال EA  = fg_elem<TA>;
    استعمال EB  = fg_elem<TB>;
    استعمال tsa = قسم_نالو EA::store;
    استعمال tsb = قسم_نالو EB::store;
    ساكن constexpr صحيح TM = TM_;
    ساكن constexpr صحيح TN = TN_;
    ساكن constexpr صحيح TK = TK_;
    ساكن constexpr صحيح SG = SG_;
    ساكن constexpr صحيح VNNI    = 4 / حجم(tsb);  // K rows of B packed in one 32-bit word
    ساكن constexpr صحيح SG_صفون = TM > 16 ? TM : 16;
    ساكن constexpr صحيح RPL     = SG_صفون / SG;     // A rows one lane decodes per k step
    ساكن constexpr صحيح MT      = SG_صفون / TM;
    ساكن constexpr صحيح BN      = TN > 32 ? TN : 32;
    ساكن constexpr صحيح NT      = BN / TN;
    ساكن constexpr صحيح WG_سائيز = FG_KSPLIT * SG;
    ساكن constexpr mx::layout b_layout = VNNI == 1 ? mx::layout::صف_major : mx::layout::ext_intel_packed;
    // a 64-wide N is mostly padding here and a 32x64 f32 accumulator needs 128 registers per lane,
    // so it spills: 13x slower on B60
    ساكن constexpr منطقي efficient = TN <= 32;
    static_assert(SG_صفون % SG == 0 && SG_صفون % TM == 0 && BN % TN == 0 && FG_BK % TK == 0, "bad tile");
    static_assert(BN <= GGML_SYCL_FG_MAX_N, "header gate must cover the tile width");
};

استعمال fg_half = sycl::half;
استعمال fg_bf16 = sycl::ext::oneapi::bfloat16;
استعمال fg_tf32 = mx::precision::tf32;

// One bit of GGML_SYCL_XMX_GATHER_SHAPES per combination. Only combinations some device lists in
// matrix_combinations are built (appendix of sycl_ext_oneapi_matrix and the runtime's own list).
سانچو <قسم_نالو F> ساكن فراغ fg_گهمو_combo(صحيح idx, F && f);
ساكن constexpr صحيح FG_N_COMBOS = 8;

// A spir64_gen AOT build (GGML_SYCL_XMX_AOT_SG) drops the combinations of the other sub-group size
// entirely: ocloc rejects even an empty kernel that asks for a sub-group size it lacks.
سانچو <صحيح SG> ساكن constexpr منطقي fg_listed() {
#إذا defined(GGML_SYCL_XMX_AOT_SG)
    أرجع SG == GGML_SYCL_XMX_AOT_SG;
#وإلا
    أرجع سچ;
#جيڪڏهن_پورو
}

سانچو <قسم_نالو S, قسم_نالو F> ساكن فراغ fg_call_combo(F && f) {
    إذا constexpr (fg_listed<S::SG>()) {
        f(S{});
    }
}

سانچو <قسم_نالو F> ساكن فراغ fg_گهمو_combo(صحيح idx, F && f) {
    بدل (idx) {
        حالة 0: fg_call_combo<fg_combo<fg_half, fg_half, 8, 16, 16, 16>>(f);  اكسر; // Xe2, Xe3, Xe-HPC
        حالة 1: fg_call_combo<fg_combo<fg_half, fg_half, 16, 16, 16, 16>>(f); اكسر; // Xe2, Xe3, Xe-HPC
        حالة 2: fg_call_combo<fg_combo<fg_half, fg_half, 32, 64, 16, 16>>(f); اكسر; // Xe2, Xe3, Xe-HPC
        حالة 3: fg_call_combo<fg_combo<fg_half, fg_half, 32, 64, 32, 16>>(f); اكسر; // Xe2, Xe3, Xe-HPC
        حالة 4: fg_call_combo<fg_combo<fg_half, fg_half, 8, 8, 16, 8>>(f);    اكسر; // Xe-HPG (Arc A), ARL-H
        حالة 5: fg_call_combo<fg_combo<fg_tf32, fg_tf32, 8, 16, 8, 16>>(f);   اكسر; // Xe2, Xe3, Xe-HPC
        حالة 6: fg_call_combo<fg_combo<fg_bf16, fg_bf16, 8, 16, 16, 16>>(f);  اكسر; // Xe2, Xe3, Xe-HPC
        حالة 7: fg_call_combo<fg_combo<fg_bf16, fg_bf16, 8, 8, 16, 8>>(f);    اكسر; // Xe-HPG (Arc A), ARL-H
        افتراضي: GGML_ABORT("bad XMX combination %d", idx);
    }
}

// AOT with -fsycl-targets=intel_gpu_*: compile each tile body only for targets with its sub-group
// size, since IGC fails on the other ones. A JIT build keeps them all, but each combination lands in
// its own device image (joint_matrix is an optional kernel feature) and only a combination the
// device reports is launched, so the runtime never asks IGC for the others.
#إذا defined(__SYCL_DEVICE_ONLY__)
#    إذا __SYCL_TARGET_INTEL_GPU_ACM_G10__ || __SYCL_TARGET_INTEL_GPU_ACM_G11__ || __SYCL_TARGET_INTEL_GPU_ACM_G12__ || \
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
#جيڪڏهن_پورو

سانچو <صحيح SG> ساكن constexpr منطقي fg_built() {
#إذا defined(FG_AOT_SG)
    أرجع SG == FG_AOT_SG;
#وإلا
    أرجع سچ;
#جيڪڏهن_پورو
}

// Upper bound on the tile count when total_rows rows are routed to n_as experts: the worst case gives
// each expert one row and fills whole tiles with the rest. The bound depends only on the shape, not
// on the routing, so the pool reuses one buffer every ubatch instead of keeping one per size seen.
ساكن constexpr int64_t grouped_gemm_max_tiles(int64_t ڪل_صفون, int64_t n_as, int64_t BN) {
    أرجع ڪل_صفون <= n_as ? ڪل_صفون : n_as + (ڪل_صفون - n_as) / BN;
}
// Tiles do not cross experts, so the bound is not ceil(total_rows / BN): 34 rows over 2 experts with
// BN = 16 split 17 + 17 need 2 + 2 tiles, where the ceil gives 3.
static_assert(grouped_gemm_max_tiles(34, 2, 16) == 4);

// the device lists S with an f32 accumulator and output
سانچو <قسم_نالو S> ساكن منطقي fg_device_has_combo(ثابت std::vector<mx::combination> & combinations) {
    لكل (ثابت خودڪار & c : combinations) {
        إذا (c.atype == S::EA::mtype && c.btype == S::EB::mtype && c.ctype == mx::ميٽرڪس_type::fp32 &&
            c.dtype == mx::ميٽرڪس_type::fp32 &&
            (c.max_msize >= (size_t) S::TM || c.msize == (size_t) S::TM) &&
            (c.max_nsize >= (size_t) S::TN || c.nsize == (size_t) S::TN) &&
            (c.max_ksize >= (size_t) S::TK || c.ksize == (size_t) S::TK)) {
            أرجع سچ;
        }
    }
    أرجع خطأ;
}

سانچو <قسم_نالو T> ساكن ثابت حرف * fg_type_name() {
    أرجع std::is_same_v<T, fg_half> ? "f16" : std::is_same_v<T, fg_bf16> ? "bf16" : "tf32";
}

ساكن std::string fg_combo_name(صحيح idx) {
    std::string name;
    fg_گهمو_combo(idx, [&](خودڪار s) {
        استعمال S = decltype(s);
        name = std::string(fg_type_name<قسم_نالو S::ta>()) + "x" + fg_type_name<قسم_نالو S::tb>() + " " +
               std::to_string(S::TM) + "x" + std::to_string(S::TN) + "x" + std::to_string(S::TK) + " sg" +
               std::to_string(S::SG);
    });
    أرجع name;
}

// Combinations this build has kernels for and the device lists, one bit each. Cached per device:
// on a mixed box the first caller's verdict is not the others'.
ساكن صحيح fg_device_combos(ثابت sycl::device & dev) {
    ساكن std::mutex                            mtx;
    ساكن std::unordered_map<sycl::device, صحيح> معلوم;
    std::lock_guard<std::mutex>                  تالو(mtx);
    ثابت خودڪار                                   it = معلوم.find(dev);
    إذا (it != معلوم.end()) {
        أرجع it->second;
    }
    صحيح موجود = 0;
    حاول {
        ثابت خودڪار combinations = dev.get_info<sycl::ext::oneapi::experimental::info::device::ميٽرڪس_combinations>();
        ثابت خودڪار sg_sizes     = dev.get_info<sycl::info::device::sub_گروهه_sizes>();
        لكل (صحيح idx = 0; idx < FG_N_COMBOS; ++idx) {
            fg_گهمو_combo(idx, [&](خودڪار s) {
                استعمال S = decltype(s);
                ثابت منطقي sg_ok = std::find(sg_sizes.begin(), sg_sizes.end(), (size_t) S::SG) != sg_sizes.end();
                إذا (sg_ok && fg_device_has_combo<S>(combinations)) {
                    موجود |= 1 << idx;
                }
            });
        }
    } امسك (ثابت sycl::exception &) {
        موجود = 0;
    }
    GGML_LOG_INFO("%s: %s: XMX dequant-GEMM combinations available 0x%x, allowed 0x%x\n", __func__,
                  dev.get_info<sycl::info::device::name>().c_str(), موجود, g_ggml_sycl_xmx_gather_shapes);
    معلوم.emplace(dev, موجود);
    أرجع موجود;
}

// Rank of combination S for a src1 of type src1_type, lower is better. Order:
//  1. throughput: a tile that does not spill, then the fastest type class of A and B
//  2. B type equal to the src1 type, so the pack is a plain copy
//  3. B at least as precise as f16
//  4. the device's native DPAS tile (8 x SG x 32 bytes of K), then the largest M x K
// A costs nothing to convert: the A stage emits any type at the same cost.
سانچو <قسم_نالو S> ساكن int64_t fg_rank(ggml_type src1_type) {
    ثابت int64_t spills  = !S::efficient;
    ثابت int64_t slow    = std::max(S::EA::slow, S::EB::slow);
    ثابت int64_t convert = S::EB::src != src1_type;
    ثابت int64_t lossy   = S::EB::mant < 10;
    ثابت int64_t foreign = !(S::TM == 8 && S::TN == S::SG);
    ثابت int64_t mk      = 1024 - S::TM * S::TK;
    أرجع ((((spills * 2 + slow) * 2 + convert) * 2 + lossy) * 2 + foreign) * 2048 + mk;
}

// whether XMX operands of type mode (a GGML_SYCL_DYNAMIC_PRECISION value) meet the src1 request
// [TAG_GGML_PREC]. f16 lacks the f32 range that BF16 and F32 ask for; an F32 request goes only as far
// down as GGML_SYCL_DYNAMIC_REQUIRED_PRECISION allows.
ساكن منطقي fg_گهڻائي_meets(صحيح mode, int32_t src1_prec) {
    إذا (src1_prec == GGML_PREC_UNDEFINED || src1_prec >= GGML_PREC_F16) {
        أرجع سچ;
    }
    إذا (src1_prec >= GGML_PREC_BF16) {
        أرجع mode != GGML_SYCL_DYNAMIC_PRECISION_F16;
    }
    بدل (g_ggml_sycl_dynamic_required_precision) {
        حالة GGML_SYCL_DYNAMIC_PRECISION_TF32: أرجع mode == GGML_SYCL_DYNAMIC_PRECISION_TF32;
        حالة GGML_SYCL_DYNAMIC_PRECISION_BF16: أرجع mode != GGML_SYCL_DYNAMIC_PRECISION_F16;
        افتراضي:                               أرجع خطأ;
    }
}

// Best allowed combination for this call, or -1 if none. The type is GGML_SYCL_DYNAMIC_PRECISION if it
// meets the src1 request; if not, bf16 then tf32 for a BF16 request (fastest first) and tf32 then bf16
// for an F32 request (most mantissa first).
ساكن صحيح fg_pick_combo(dpct::قطار_ptr stream, ggml_type src1_type, int32_t src1_prec) {
    ثابت sycl::device dev     = stream->get_device();
    ثابت صحيح          allowed = fg_device_combos(dev) & g_ggml_sycl_xmx_gather_shapes;
    ثابت منطقي         f32_req = src1_prec != GGML_PREC_UNDEFINED && src1_prec < GGML_PREC_BF16;
    ثابت صحيح          modes[] = {
        g_ggml_sycl_dynamic_precision,
        f32_req ? GGML_SYCL_DYNAMIC_PRECISION_TF32 : GGML_SYCL_DYNAMIC_PRECISION_BF16,
        f32_req ? GGML_SYCL_DYNAMIC_PRECISION_BF16 : GGML_SYCL_DYNAMIC_PRECISION_TF32,
    };
    صحيح     بهترين      = -1;
    int64_t بهترين_rank = 0;
    لكل (صحيح i = 0; i < 3 && بهترين < 0; ++i) {
        ثابت صحيح mode = modes[i];
        إذا (!fg_گهڻائي_meets(mode, src1_prec)) {
            استمر;
        }
        لكل (صحيح idx = 0; idx < FG_N_COMBOS; ++idx) {
            إذا (!(allowed & (1 << idx))) {
                استمر;
            }
            fg_گهمو_combo(idx, [&](خودڪار s) {
                استعمال S = decltype(s);
                إذا (S::EA::mode != mode || S::EB::mode != mode) {
                    أرجع;
                }
                ثابت int64_t rank = fg_rank<S>(src1_type);
                إذا (بهترين < 0 || rank < بهترين_rank) {
                    بهترين      = idx;
                    بهترين_rank = rank;
                }
            });
        }
    }
    // log each distinct decision once
    ساكن std::mutex                                      mtx;
    ساكن std::set<std::tuple<size_t, صحيح, int32_t, صحيح>> ڏٺل;
    std::lock_guard<std::mutex>                            تالو(mtx);
    إذا (ڏٺل.emplace(std::hash<sycl::device>{}(dev), (صحيح) src1_type, src1_prec, بهترين).second) {
        GGML_LOG_INFO("%s: src1 %s, src1 prec %d -> %s\n", __func__, ggml_type_name(src1_type), src1_prec,
                      بهترين >= 0 ? fg_combo_name(بهترين).c_str() : "none (library GEMM)");
    }
    أرجع بهترين;
}

// src1 [N][K] -> packed [K/V][Npad][V] so B tiles load straight from global memory
سانچو <قسم_نالو E, قسم_نالو T_src>
ساكن فراغ fused_gemm_pack_b(ثابت T_src * y, قسم_نالو E::store * packed, صحيح N, صحيح Npad, صحيح K,
                              dpct::قطار_ptr stream) {
    constexpr صحيح V   = 4 / حجم(قسم_نالو E::store);
    ثابت صحيح     kqs = K / V;
    stream->parallel_for(sycl::range<1>((size_t) Npad * kqs), [=](sycl::id<1> id) {
        ثابت صحيح idx = id[0];
        ثابت صحيح n   = idx / kqs;
        ثابت صحيح kq  = idx - n * kqs;
        قسم_نالو E::store vals[V] = {};
        إذا (n < N) {
            ثابت T_src * src = y + (size_t) n * K + V * kq;
#هدايت unroll
            لكل (صحيح v = 0; v < V; ++v) {
                vals[v] = E::cvt((عائم) src[v]);
            }
        }
        قسم_نالو E::store * ٻاهريون = packed + ((size_t) kq * Npad + n) * V;
#هدايت unroll
        لكل (صحيح v = 0; v < V; ++v) {
            ٻاهريون[v] = vals[v];
        }
    });
}

// A stage: one lane owns one row and decodes FG_BK values of it per k step, with every scale
// folded into the value so the mad below sees plain A elements. One overload per weight format.
سانچو <قسم_نالو E>
ساكن __dpct_inline__ فراغ fg_stage_a(ثابت بلاڪ_iq4_nl * __restrict__ xrow, ثابت صحيح kb, قسم_نالو E::pair * a) {
    ثابت بلاڪ_iq4_nl blk = xrow[kb];
    ثابت عائم        d   = (عائم) blk.d;
#هدايت unroll
    لكل (صحيح j = 0; j < QK4_NL / 2; j += 2) {
        ثابت uint8_t q0 = blk.qs[j];
        ثابت uint8_t q1 = blk.qs[j + 1];
        a[j / 2]     = E::make(d * kvalues_iq4nl[q0 & 0xf], d * kvalues_iq4nl[q1 & 0xf]);
        a[j / 2 + 8] = E::make(d * kvalues_iq4nl[q0 >> 4], d * kvalues_iq4nl[q1 >> 4]);
    }
}

// iq3_s: k step kb is sub-block kb % 8 of superblock kb / 8. The superblock is 110 bytes, so read
// only the fields of that sub-block instead of copying the block. Same decode as
// dequantize_block_iq3_s: grid entries are taken as dwords and the sign bit is a plain shift.
سانچو <قسم_نالو E>
ساكن __dpct_inline__ فراغ fg_stage_a(ثابت بلاڪ_iq3_s * __restrict__ xrow, ثابت صحيح kb, قسم_نالو E::pair * a) {
    static_assert(QK_K == 256, "the iq3_s A stage assumes 8 sub-blocks per superblock");
    ثابت بلاڪ_iq3_s * blk = xrow + kb / (QK_K / 32);
    ثابت صحيح           ib8 = kb % (QK_K / 32);
    ثابت uint8_t *     qs  = blk->qs + 8 * ib8;
    ثابت صحيح           qh  = blk->qh[ib8];
    ثابت عائم         d   = (عائم) blk->d * (1 + 2 * ((blk->scales[ib8 / 2] >> (4 * (ib8 % 2))) & 0xf));
#هدايت unroll
    لكل (صحيح il = 0; il < 4; ++il) {
        ثابت uint32_t grid1 = iq3s_grid[qs[2 * il + 0] | ((qh << (8 - 2 * il)) & 256)];
        ثابت uint32_t grid2 = iq3s_grid[qs[2 * il + 1] | ((qh << (7 - 2 * il)) & 256)];
        ثابت صحيح      signs = blk->signs[4 * ib8 + il];
#هدايت unroll
        لكل (صحيح j = 0; j < 2; ++j) {
            ثابت عائم g1a = (عائم) ((grid1 >> (16 * j + 0)) & 0xff);
            ثابت عائم g1b = (عائم) ((grid1 >> (16 * j + 8)) & 0xff);
            ثابت عائم g2a = (عائم) ((grid2 >> (16 * j + 0)) & 0xff);
            ثابت عائم g2b = (عائم) ((grid2 >> (16 * j + 8)) & 0xff);
            ثابت صحيح   s   = 2 * j;
            a[4 * il + j]     = E::make(d * ((signs & (1 << (s + 0))) ? -g1a : g1a),
                                        d * ((signs & (1 << (s + 1))) ? -g1b : g1b));
            a[4 * il + j + 2] = E::make(d * ((signs & (1 << (s + 4))) ? -g2a : g2a),
                                        d * ((signs & (1 << (s + 5))) ? -g2b : g2b));
        }
    }
}

// values per stored block, so a row of K values is K/qk blocks
سانچو <قسم_نالو بلاڪ_q_t> بنية fg_بلاڪ_traits;
سانچو <> بنية fg_بلاڪ_traits<بلاڪ_iq4_nl> { ساكن constexpr صحيح qk = QK4_NL; };
سانچو <> بنية fg_بلاڪ_traits<بلاڪ_iq3_s>  { ساكن constexpr صحيح qk = QK_K; };
سانچو <> بنية fg_بلاڪ_traits<بلاڪ_iq3_xxs> { ساكن constexpr صحيح qk = QK_K; };
سانچو <> بنية fg_بلاڪ_traits<بلاڪ_iq4_xs>  { ساكن constexpr صحيح qk = QK_K; };
سانچو <> بنية fg_بلاڪ_traits<بلاڪ_iq2_xxs> { ساكن constexpr صحيح qk = QK_K; };
سانچو <> بنية fg_بلاڪ_traits<بلاڪ_iq2_xs>  { ساكن constexpr صحيح qk = QK_K; };
سانچو <> بنية fg_بلاڪ_traits<بلاڪ_iq2_s>   { ساكن constexpr صحيح qk = QK_K; };
سانچو <> بنية fg_بلاڪ_traits<بلاڪ_iq1_s>   { ساكن constexpr صحيح qk = QK_K; };
سانچو <> بنية fg_بلاڪ_traits<بلاڪ_iq1_m>   { ساكن constexpr صحيح qk = QK_K; };
سانچو <> بنية fg_بلاڪ_traits<بلاڪ_q8_0>    { ساكن constexpr صحيح qk = QK8_0; };
سانچو <> بنية fg_بلاڪ_traits<بلاڪ_q4_K>    { ساكن constexpr صحيح qk = QK_K; };
سانچو <> بنية fg_بلاڪ_traits<بلاڪ_q5_K>    { ساكن constexpr صحيح qk = QK_K; };
سانچو <> بنية fg_بلاڪ_traits<بلاڪ_q6_K>    { ساكن constexpr صحيح qk = QK_K; };

// The A stages below are the dequantize_block_iq* kernels rewritten for one k step. There a
// work-item handled one quarter (il) of one 32-wide sub-block (ib); here one lane produces the
// whole step, so il becomes a loop and ib is kb inside the superblock. Each quarter yields 8
// consecutive values, i.e. 4 pairs at a[4*il], so nothing larger than 8 floats is ever live.
#وصف_ڪريو FG_SUPERBLOCK(T)                                                         \
    static_assert(QK_K == 256, "the " #T " A stage assumes 8 sub-blocks per superblock"); \
    ثابت T * blk = xrow + kb / (QK_K / 32);                                     \
    ثابت صحيح ib  = kb % (QK_K / 32)

سانچو <قسم_نالو E>
ساكن __dpct_inline__ فراغ fg_pack_quarter(ثابت عائم * __restrict__ t, قسم_نالو E::pair * a, صحيح il) {
#هدايت unroll
    لكل (صحيح j = 0; j < 4; ++j) {
        a[4 * il + j] = E::make(t[2 * j], t[2 * j + 1]);
    }
}

سانچو <قسم_نالو E>
ساكن __dpct_inline__ فراغ fg_stage_a(ثابت بلاڪ_iq4_xs * __restrict__ xrow, ثابت صحيح kb, قسم_نالو E::pair * a) {
    FG_SUPERBLOCK(بلاڪ_iq4_xs);
    // low nibbles fill the first half of the step, high nibbles the second, so the two halves
    // land at a[0..7] and a[8..15] and no quarter loop is needed
    ثابت عائم d = (عائم) blk->d *
        ((((blk->scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) | (((blk->scales_h >> (2 * ib)) & 3) << 4)) - 32);
    ثابت uint8_t * q4 = blk->qs + 16 * ib;
#هدايت unroll
    لكل (صحيح j = 0; j < 8; ++j) {
        a[j]     = E::make(d * kvalues_iq4nl[q4[2 * j] & 0xf], d * kvalues_iq4nl[q4[2 * j + 1] & 0xf]);
        a[8 + j] = E::make(d * kvalues_iq4nl[q4[2 * j] >> 4],  d * kvalues_iq4nl[q4[2 * j + 1] >> 4]);
    }
}

سانچو <قسم_نالو E>
ساكن __dpct_inline__ فراغ fg_stage_a(ثابت بلاڪ_iq3_xxs * __restrict__ xrow, ثابت صحيح kb, قسم_نالو E::pair * a) {
    FG_SUPERBLOCK(بلاڪ_iq3_xxs);
    ثابت uint8_t *  q3    = blk->qs + 8 * ib;
    ثابت uint16_t * gas   = (ثابت uint16_t *) (blk->qs + QK_K / 4) + 2 * ib;
    ثابت uint32_t   aux32 = gas[0] | (gas[1] << 16);
    ثابت عائم      d     = (عائم) blk->d * (0.5f + (aux32 >> 28)) * 0.5f;
#هدايت unroll
    لكل (صحيح il = 0; il < 4; ++il) {
        ثابت uint8_t * grid1 = (ثابت uint8_t *) (iq3xxs_grid + q3[2 * il + 0]);
        ثابت uint8_t * grid2 = (ثابت uint8_t *) (iq3xxs_grid + q3[2 * il + 1]);
        ثابت uint8_t   signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];
        عائم t[8];
#هدايت unroll
        لكل (صحيح j = 0; j < 4; ++j) {
            t[j + 0] = d * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f);
            t[j + 4] = d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

سانچو <قسم_نالو E>
ساكن __dpct_inline__ فراغ fg_stage_a(ثابت بلاڪ_iq2_xxs * __restrict__ xrow, ثابت صحيح kb, قسم_نالو E::pair * a) {
    FG_SUPERBLOCK(بلاڪ_iq2_xxs);
    ثابت uint16_t * q2    = blk->qs + 4 * ib;
    ثابت uint8_t *  aux8  = (ثابت uint8_t *) q2;
    ثابت uint32_t   aux32 = q2[2] | (q2[3] << 16);
    ثابت عائم      d     = (عائم) blk->d * (0.5f + (aux32 >> 28)) * 0.25f;
#هدايت unroll
    لكل (صحيح il = 0; il < 4; ++il) {
        ثابت uint8_t * grid  = (ثابت uint8_t *) (iq2xxs_grid + aux8[il]);
        ثابت uint8_t   signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];
        عائم t[8];
#هدايت unroll
        لكل (صحيح j = 0; j < 8; ++j) {
            t[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

سانچو <قسم_نالو E>
ساكن __dpct_inline__ فراغ fg_stage_a(ثابت بلاڪ_iq2_xs * __restrict__ xrow, ثابت صحيح kb, قسم_نالو E::pair * a) {
    FG_SUPERBLOCK(بلاڪ_iq2_xs);
    ثابت uint16_t * q2 = blk->qs + 4 * ib;
#هدايت unroll
    لكل (صحيح il = 0; il < 4; ++il) {
        ثابت uint8_t * grid  = (ثابت uint8_t *) (iq2xs_grid + (q2[il] & 511));
        ثابت عائم     d     = (عائم) blk->d * (0.5f + ((blk->scales[ib] >> (4 * (il / 2))) & 0xf)) * 0.25f;
        ثابت uint8_t   signs = ksigns_iq2xs[q2[il] >> 9];
        عائم t[8];
#هدايت unroll
        لكل (صحيح j = 0; j < 8; ++j) {
            t[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

سانچو <قسم_نالو E>
ساكن __dpct_inline__ فراغ fg_stage_a(ثابت بلاڪ_iq2_s * __restrict__ xrow, ثابت صحيح kb, قسم_نالو E::pair * a) {
    FG_SUPERBLOCK(بلاڪ_iq2_s);
#هدايت unroll
    لكل (صحيح il = 0; il < 4; ++il) {
        ثابت uint8_t * grid =
            (ثابت uint8_t *) (iq2s_grid + (blk->qs[4 * ib + il] | ((blk->qh[ib] << (8 - 2 * il)) & 0x300)));
        ثابت عائم   d     = (عائم) blk->d * (0.5f + ((blk->scales[ib] >> (4 * (il / 2))) & 0xf)) * 0.25f;
        ثابت uint8_t signs = blk->qs[QK_K / 8 + 4 * ib + il];
        عائم t[8];
#هدايت unroll
        لكل (صحيح j = 0; j < 8; ++j) {
            t[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

سانچو <قسم_نالو E>
ساكن __dpct_inline__ فراغ fg_stage_a(ثابت بلاڪ_iq1_s * __restrict__ xrow, ثابت صحيح kb, قسم_نالو E::pair * a) {
    FG_SUPERBLOCK(بلاڪ_iq1_s);
    ثابت عائم delta = blk->qh[ib] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
    ثابت عائم d     = (عائم) blk->d * (2 * ((blk->qh[ib] >> 12) & 7) + 1);
#هدايت unroll
    لكل (صحيح il = 0; il < 4; ++il) {
        uint32_t       grid32[2];
        ثابت int8_t * q = (ثابت int8_t *) grid32;
        grid32[0] = iq1s_grid_gpu[blk->qs[4 * ib + il] | (((blk->qh[ib] >> (3 * il)) & 7) << 8)];
        grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
        grid32[0] &= 0x0f0f0f0f;
        عائم t[8];
#هدايت unroll
        لكل (صحيح j = 0; j < 8; ++j) {
            t[j] = d * (q[j] + delta);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

سانچو <قسم_نالو E>
ساكن __dpct_inline__ فراغ fg_stage_a(ثابت بلاڪ_iq1_m * __restrict__ xrow, ثابت صحيح kb, قسم_نالو E::pair * a) {
    FG_SUPERBLOCK(بلاڪ_iq1_m);
    ثابت uint16_t * sc = (ثابت uint16_t *) blk->scales;
    iq1m_scale_t     scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
#هدايت unroll
    لكل (صحيح il = 0; il < 4; ++il) {
        ثابت صحيح   ib16  = 2 * ib + il / 2;
        ثابت عائم d     = (عائم) scale.f16 * (2 * ((sc[ib16 / 4] >> (3 * (ib16 % 4))) & 0x7) + 1);
        ثابت عائم delta = blk->qh[2 * ib + il / 2] & (0x08 << (4 * (il % 2))) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
        uint32_t       grid32[2];
        ثابت int8_t * q = (ثابت int8_t *) grid32;
        grid32[0] = iq1s_grid_gpu[blk->qs[4 * ib + il] |
                                  (((blk->qh[2 * ib + il / 2] >> (4 * (il % 2))) & 7) << 8)];
        grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
        grid32[0] &= 0x0f0f0f0f;
        عائم t[8];
#هدايت unroll
        لكل (صحيح j = 0; j < 8; ++j) {
            t[j] = d * (q[j] + delta);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}


// q8_0 and the k-quants. Each decode takes the fields of one 32-value sub-block, so the canonical
// layout and the reorder (SoA) layout of reorder_qw() share it and differ only in where the fields
// live, as in dequantize.hpp.
سانچو <قسم_نالو E>
ساكن __dpct_inline__ فراغ fg_decode_q8_0(ثابت int8_t * __restrict__ qs, ثابت عائم d, قسم_نالو E::pair * a) {
#هدايت unroll
    لكل (صحيح j = 0; j < QK8_0 / 2; ++j) {
        a[j] = E::make(d * qs[2 * j], d * qs[2 * j + 1]);
    }
}

// same unpack as get_scale_min_k4() in dequantize.hpp
ساكن __dpct_inline__ فراغ fg_scale_min_k4(ثابت صحيح j, ثابت uint8_t * __restrict__ q, uint8_t & d, uint8_t & m) {
    إذا (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } وإلا {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

// q4_K sub-block ib (0..7) uses scale/min pair ib and the low (even ib) or high (odd ib) nibbles of
// qs[32 * (ib / 2) ...], as in dequantize_row_q4_K
سانچو <قسم_نالو E>
ساكن __dpct_inline__ فراغ fg_decode_q4_K(ثابت uint8_t * __restrict__ qs, ثابت uint8_t * __restrict__ scales,
                                           ثابت sycl::half2 dm, ثابت صحيح ib, قسم_نالو E::pair * a) {
    uint8_t sc, mb;
    fg_scale_min_k4(ib, scales, sc, mb);
    ثابت عائم     d     = (عائم) dm[0] * sc;
    ثابت عائم     m     = (عائم) dm[1] * mb;
    ثابت uint8_t * q     = qs + 32 * (ib / 2);
    ثابت صحيح       سرڪايو = 4 * (ib % 2);
#هدايت unroll
    لكل (صحيح j = 0; j < 16; ++j) {
        a[j] = E::make(d * ((q[2 * j] >> سرڪايو) & 0xF) - m, d * ((q[2 * j + 1] >> سرڪايو) & 0xF) - m);
    }
}

// q5_K: q4_K plus one high bit per value, bit ib of qh[l]
سانچو <قسم_نالو E>
ساكن __dpct_inline__ فراغ fg_decode_q5_K(ثابت uint8_t * __restrict__ qs, ثابت uint8_t * __restrict__ qh,
                                           ثابت uint8_t * __restrict__ scales, ثابت sycl::half2 dm, ثابت صحيح ib,
                                           قسم_نالو E::pair * a) {
    uint8_t sc, mb;
    fg_scale_min_k4(ib, scales, sc, mb);
    ثابت عائم     d     = (عائم) dm[0] * sc;
    ثابت عائم     m     = (عائم) dm[1] * mb;
    ثابت uint8_t * q     = qs + 32 * (ib / 2);
    ثابت صحيح       سرڪايو = 4 * (ib % 2);
#هدايت unroll
    لكل (صحيح j = 0; j < 16; ++j) {
        ثابت صحيح l = 2 * j;
        a[j] = E::make(d * (((q[l] >> سرڪايو) & 0xF) | (((qh[l] >> ib) & 1) << 4)) - m,
                       d * (((q[l + 1] >> سرڪايو) & 0xF) | (((qh[l + 1] >> ib) & 1) << 4)) - m);
    }
}

// q6_K: sub-block ib is quarter r = ib % 4 of half h = ib / 4. The half selects ql + 64h, qh + 32h
// and scales + 8h; the quarter selects ql + 32(r & 1), the ql nibble r / 2, the qh bit pair r and
// scales + 2r, as in dequantize_row_q6_K. The scale changes at value 16 of the sub-block.
سانچو <قسم_نالو E>
ساكن __dpct_inline__ فراغ fg_decode_q6_K(ثابت uint8_t * __restrict__ ql, ثابت uint8_t * __restrict__ qh,
                                           ثابت int8_t * __restrict__ scales, ثابت عائم d, ثابت صحيح ib,
                                           قسم_نالو E::pair * a) {
    ثابت صحيح       h  = ib / 4;
    ثابت صحيح       r  = ib % 4;
    ثابت uint8_t * q  = ql + 64 * h + 32 * (r & 1);
    ثابت uint8_t * hb = qh + 32 * h;
    ثابت int8_t *  sc = scales + 8 * h + 2 * r;
#هدايت unroll
    لكل (صحيح j = 0; j < 16; ++j) {
        ثابت صحيح   l  = 2 * j;
        ثابت عائم dl = d * sc[j / 8];
        ثابت صحيح   q0 = (((q[l] >> (4 * (r / 2))) & 0xF) | (((hb[l] >> (2 * r)) & 3) << 4)) - 32;
        ثابت صحيح   q1 = (((q[l + 1] >> (4 * (r / 2))) & 0xF) | (((hb[l + 1] >> (2 * r)) & 3) << 4)) - 32;
        a[j] = E::make(dl * q0, dl * q1);
    }
}

سانچو <قسم_نالو E>
ساكن __dpct_inline__ فراغ fg_stage_a(ثابت بلاڪ_q8_0 * __restrict__ xrow, ثابت صحيح kb, قسم_نالو E::pair * a) {
    fg_decode_q8_0<E>(xrow[kb].qs, (عائم) xrow[kb].d, a);
}

سانچو <قسم_نالو E>
ساكن __dpct_inline__ فراغ fg_stage_a(ثابت بلاڪ_q4_K * __restrict__ xrow, ثابت صحيح kb, قسم_نالو E::pair * a) {
    FG_SUPERBLOCK(بلاڪ_q4_K);
    fg_decode_q4_K<E>(blk->qs, blk->scales, blk->dm, ib, a);
}

سانچو <قسم_نالو E>
ساكن __dpct_inline__ فراغ fg_stage_a(ثابت بلاڪ_q5_K * __restrict__ xrow, ثابت صحيح kb, قسم_نالو E::pair * a) {
    FG_SUPERBLOCK(بلاڪ_q5_K);
    fg_decode_q5_K<E>(blk->qs, blk->qh, blk->scales, blk->dm, ib, a);
}

سانچو <قسم_نالو E>
ساكن __dpct_inline__ فراغ fg_stage_a(ثابت بلاڪ_q6_K * __restrict__ xrow, ثابت صحيح kb, قسم_نالو E::pair * a) {
    FG_SUPERBLOCK(بلاڪ_q6_K);
    fg_decode_q6_K<E>(blk->ql, blk->qh, blk->scales, (عائم) blk->d, ib, a);
}

// Reorder (SoA) layout: each block field is one stream over the nblocks of the matrix (of the expert
// slice for MUL_MAT_ID), in the order reorder_qw() writes them. Block ib holds k step kb.
سانچو <قسم_نالو بلاڪ_q_t> بنية fg_soa {
    ساكن constexpr منطقي supported = خطأ;
};

سانچو <> بنية fg_soa<بلاڪ_q8_0> {
    ساكن constexpr منطقي supported = سچ;
    // [qs][d]
    سانچو <قسم_نالو E>
    ساكن __dpct_inline__ فراغ stage(ثابت uint8_t * x, ثابت size_t nblocks, ثابت size_t ib, ثابت صحيح,
                                      قسم_نالو E::pair * a) {
        ثابت عائم d = (عائم) ((ثابت sycl::half *) (x + nblocks * QK8_0))[ib];
        fg_decode_q8_0<E>((ثابت int8_t *) x + ib * QK8_0, d, a);
    }
};

سانچو <> بنية fg_soa<بلاڪ_q4_K> {
    ساكن constexpr منطقي supported = سچ;
    // [qs][scales][dm]
    سانچو <قسم_نالو E>
    ساكن __dpct_inline__ فراغ stage(ثابت uint8_t * x, ثابت size_t nblocks, ثابت size_t ib, ثابت صحيح kb,
                                      قسم_نالو E::pair * a) {
        ثابت uint8_t *   scales = x + nblocks * (QK_K / 2);
        ثابت sycl::half2 dm     = ((ثابت sycl::half2 *) (scales + nblocks * K_SCALE_سائيز))[ib];
        fg_decode_q4_K<E>(x + ib * (QK_K / 2), scales + ib * K_SCALE_سائيز, dm, kb % (QK_K / 32), a);
    }
};

سانچو <> بنية fg_soa<بلاڪ_q5_K> {
    ساكن constexpr منطقي supported = سچ;
    // [qs][qh][scales][dm]
    سانچو <قسم_نالو E>
    ساكن __dpct_inline__ فراغ stage(ثابت uint8_t * x, ثابت size_t nblocks, ثابت size_t ib, ثابت صحيح kb,
                                      قسم_نالو E::pair * a) {
        ثابت uint8_t *   qh     = x + nblocks * (QK_K / 2);
        ثابت uint8_t *   scales = qh + nblocks * (QK_K / 8);
        ثابت sycl::half2 dm     = ((ثابت sycl::half2 *) (scales + nblocks * K_SCALE_سائيز))[ib];
        fg_decode_q5_K<E>(x + ib * (QK_K / 2), qh + ib * (QK_K / 8), scales + ib * K_SCALE_سائيز, dm,
                          kb % (QK_K / 32), a);
    }
};

سانچو <> بنية fg_soa<بلاڪ_q6_K> {
    ساكن constexpr منطقي supported = سچ;
    // [ql][qh][scales][d]
    سانچو <قسم_نالو E>
    ساكن __dpct_inline__ فراغ stage(ثابت uint8_t * x, ثابت size_t nblocks, ثابت size_t ib, ثابت صحيح kb,
                                      قسم_نالو E::pair * a) {
        ثابت uint8_t * qh     = x + nblocks * (QK_K / 2);
        ثابت uint8_t * scales = qh + nblocks * (QK_K / 4);
        ثابت عائم     d      = (عائم) ((ثابت sycl::half *) (scales + nblocks * (QK_K / 16)))[ib];
        fg_decode_q6_K<E>(x + ib * (QK_K / 2), qh + ib * (QK_K / 4), (ثابت int8_t *) scales + ib * (QK_K / 16), d,
                          kb % (QK_K / 32), a);
    }
};


// one SG_ROWS x BN output tile: B columns [b0, b0 + BN) of packed_b go to dst columns [n0, n1),
// n1 - n0 <= BN
سانچو <قسم_نالو S, قسم_نالو بلاڪ_q_t, منطقي reordered>
ساكن فراغ fused_dequant_gemm_tile(
    ثابت بلاڪ_q_t * __restrict__ x,
    ثابت قسم_نالو S::tsb * __restrict__ packed_b,
    عائم * __restrict__ dst,
    ثابت صحيح M, ثابت صحيح Npad, ثابت صحيح K, ثابت صحيح ldd,
    ثابت صحيح b0, ثابت صحيح n0, ثابت صحيح n1,
    sycl::local_accessor<قسم_نالو S::tsa, 1> tile_a,
    sycl::local_accessor<عائم, 1> tile_c,
    ثابت sycl::nd_شي<2> & شي) {
    إذا constexpr (fg_built<S::SG>()) {
        استعمال EA = قسم_نالو S::EA;
        استعمال TA = قسم_نالو S::ta;
        استعمال TB = قسم_نالو S::tb;
        ثابت خودڪار sg     = شي.get_sub_گروهه();
        ثابت صحيح  sg_id  = sg.get_گروهه_id()[0];
        ثابت صحيح  lane   = sg.get_local_id()[0];
        ثابت صحيح  m0     = شي.get_گروهه(1) * S::SG_صفون;
        ثابت صحيح  nstep  = K / FG_BK;
        ثابت صحيح  a_بنياد = sg_id * S::SG_صفون * FG_BK;
        ثابت صحيح  c_بنياد = sg_id * S::SG_صفون * S::BN;

        mx::joint_ميٽرڪس<sycl::sub_گروهه, عائم, mx::use::گڏ_ڪندڙ, S::TM, S::TN> acc[S::MT][S::NT];
#هدايت unroll
        لكل (صحيح mt = 0; mt < S::MT; ++mt) {
#هدايت unroll
            لكل (صحيح nt = 0; nt < S::NT; ++nt) {
                mx::joint_ميٽرڪس_fill(sg, acc[mt][nt], 0.0f);
            }
        }

        // lane decodes rows lane, lane + SG, ... of the sub-group's SG_ROWS
        constexpr صحيح         KPB     = fg_بلاڪ_traits<بلاڪ_q_t>::qk / FG_BK; // k steps per block
        ثابت size_t          bpr     = K / fg_بلاڪ_traits<بلاڪ_q_t>::qk;
        ثابت size_t          nblocks = (size_t) M * bpr;
        size_t                صف_blk[S::RPL];
        منطقي                  صف_ok[S::RPL];
        قسم_نالو EA::pair *   a[S::RPL];
#هدايت unroll
        لكل (صحيح r = 0; r < S::RPL; ++r) {
            ثابت صحيح صف = m0 + r * S::SG + lane;
            صف_ok[r]  = صف < M;
            صف_blk[r] = (size_t) (صف_ok[r] ? صف : 0) * bpr;
            a[r]       = (قسم_نالو EA::pair *) &tile_a[a_بنياد + (r * S::SG + lane) * FG_BK];
        }

        ثابت خودڪار b_ptr = sycl::پتو_space_cast<sycl::access::پتو_space::global_space,
                                                    sycl::access::decorated::no>(packed_b);
        ثابت صحيح b_stride = Npad * S::VNNI;

        ثابت صحيح kb_begin = (sg_id * nstep) / FG_KSPLIT;
        ثابت صحيح kb_پڇاڙي   = ((sg_id + 1) * nstep) / FG_KSPLIT;
        لكل (صحيح kb = kb_begin; kb < kb_پڇاڙي; ++kb) {
#هدايت unroll
            لكل (صحيح r = 0; r < S::RPL; ++r) {
                إذا (صف_ok[r]) {
                    إذا constexpr (reordered) {
                        fg_soa<بلاڪ_q_t>::سانچو stage<EA>((ثابت uint8_t *) x, nblocks, صف_blk[r] + kb / KPB, kb,
                                                              a[r]);
                    } وإلا {
                        fg_stage_a<EA>(x + صف_blk[r], kb, a[r]);
                    }
                } وإلا {
#هدايت unroll
                    لكل (صحيح j = 0; j < FG_BK / 2; ++j) {
                        a[r][j] = EA::make(0.0f, 0.0f);
                    }
                }
            }
            sycl::گروهه_barrier(sg);

#هدايت unroll
            لكل (صحيح kt = 0; kt < FG_BK / S::TK; ++kt) {
                ثابت صحيح kq0 = (kb * FG_BK + kt * S::TK) / S::VNNI;
                mx::joint_ميٽرڪس<sycl::sub_گروهه, TB, mx::use::b, S::TK, S::TN, S::b_layout> sub_b[S::NT];
#هدايت unroll
                لكل (صحيح nt = 0; nt < S::NT; ++nt) {
                    mx::joint_ميٽرڪس_ڀريو(sg, sub_b[nt], b_ptr + (size_t) kq0 * b_stride + (b0 + nt * S::TN) * S::VNNI, b_stride);
                }
#هدايت unroll
                لكل (صحيح mt = 0; mt < S::MT; ++mt) {
                    mx::joint_ميٽرڪس<sycl::sub_گروهه, TA, mx::use::a, S::TM, S::TK, mx::layout::صف_major> sub_a;
                    mx::joint_ميٽرڪس_ڀريو(sg, sub_a,
                        tile_a.سانچو get_multi_ptr<sycl::access::decorated::no>() + a_بنياد + (mt * S::TM) * FG_BK + kt * S::TK,
                        FG_BK);
#هدايت unroll
                    لكل (صحيح nt = 0; nt < S::NT; ++nt) {
                        mx::joint_ميٽرڪس_mad(sg, acc[mt][nt], sub_a, sub_b[nt], acc[mt][nt]);
                    }
                }
            }
            // the next step overwrites tile_a
            sycl::گروهه_barrier(sg);
        }

#هدايت unroll
        لكل (صحيح mt = 0; mt < S::MT; ++mt) {
#هدايت unroll
            لكل (صحيح nt = 0; nt < S::NT; ++nt) {
                mx::joint_ميٽرڪس_store(sg, acc[mt][nt],
                    tile_c.سانچو get_multi_ptr<sycl::access::decorated::no>() + c_بنياد + (mt * S::TM) * S::BN + nt * S::TN,
                    S::BN, mx::layout::صف_major);
            }
        }
        sycl::گروهه_barrier(شي.get_گروهه());

        // sum the K splits; consecutive lanes write consecutive rows of one dst column
        لكل (صحيح idx = شي.get_local_لڪيري_id(); idx < S::SG_صفون * S::BN; idx += S::WG_سائيز) {
            ثابت صحيح r = idx % S::SG_صفون;
            ثابت صحيح c = idx / S::SG_صفون;
            ثابت صحيح m = m0 + r;
            ثابت صحيح n = n0 + c;
            إذا (m < M && n < n1) {
                عائم جوڙ = 0.0f;
#هدايت unroll
                لكل (صحيح s = 0; s < FG_KSPLIT; ++s) {
                    جوڙ += tile_c[s * S::SG_صفون * S::BN + r * S::BN + c];
                }
                dst[(size_t) n * ldd + m] = جوڙ;
            }
        }
    }
}

سانچو <قسم_نالو S, قسم_نالو بلاڪ_q_t, منطقي reordered = خطأ>
ساكن فراغ fused_dequant_gemm_launch(ثابت فراغ * src0, ثابت قسم_نالو S::tsb * packed, عائم * dst, ثابت صحيح M,
                                      ثابت صحيح N, ثابت صحيح Npad, ثابت صحيح K, ثابت صحيح ldd,
                                      ثابت int64_t groups_n, ثابت int64_t groups_m, dpct::قطار_ptr stream) {
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<قسم_نالو S::tsa, 1> tile_a(FG_KSPLIT * S::SG_صفون * FG_BK, cgh);
        sycl::local_accessor<عائم, 1>          tile_c(FG_KSPLIT * S::SG_صفون * S::BN, cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(groups_n, groups_m * S::WG_سائيز), sycl::range<2>(1, S::WG_سائيز)),
            [=](sycl::nd_شي<2> شي) [[sycl::reqd_sub_گروهه_سائيز(S::SG)]] {
                ثابت صحيح n0 = شي.get_گروهه(0) * S::BN;
                fused_dequant_gemm_tile<S, بلاڪ_q_t, reordered>((ثابت بلاڪ_q_t *) src0, packed, dst, M, Npad, K,
                                                      ldd, n0, n0, N, tile_a, tile_c, شي);
            });
    });
}

// grouped: work-group (t, mt) is tile t of the schedule; its B columns sit at t * BN
سانچو <قسم_نالو S, قسم_نالو بلاڪ_q_t, منطقي reordered>
ساكن فراغ grouped_dequant_gemm_launch(ثابت حرف * src0_dd, ثابت size_t expert_stride,
                                        ثابت ggml_sycl_gg_tile * tiles_ptr, ثابت قسم_نالو S::tsb * packed, عائم * dst,
                                        ثابت صحيح M, ثابت صحيح Npad, ثابت صحيح K, ثابت int64_t n_tiles,
                                        ثابت int64_t groups_m, dpct::قطار_ptr stream) {
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<قسم_نالو S::tsa, 1> tile_a(FG_KSPLIT * S::SG_صفون * FG_BK, cgh);
        sycl::local_accessor<عائم, 1>          tile_c(FG_KSPLIT * S::SG_صفون * S::BN, cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(n_tiles, groups_m * S::WG_سائيز), sycl::range<2>(1, S::WG_سائيز)),
            [=](sycl::nd_شي<2> شي) [[sycl::reqd_sub_گروهه_سائيز(S::SG)]] {
                ثابت صحيح               t    = شي.get_گروهه(0);
                ثابت ggml_sycl_gg_tile tile = tiles_ptr[t];
                ثابت بلاڪ_q_t *       x    = (ثابت بلاڪ_q_t *) (src0_dd + (size_t) tile.expert * expert_stride);
                fused_dequant_gemm_tile<S, بلاڪ_q_t, reordered>(x, packed, dst, M, Npad, K, M, t * S::BN, tile.n0,
                                                                 tile.n1, tile_a, tile_c, شي);
            });
    });
}

// src1 f32 rows -> packed [K/V][n_tiles*BN][V], tile t holds its rows [n0, n1) at columns t*BN..,
// zero past n1. The column runs fastest so a sub-group writes one contiguous run.
سانچو <قسم_نالو S>
ساكن فراغ grouped_gemm_pack_b(ثابت عائم * y, قسم_نالو S::tsb * packed, ثابت ggml_sycl_gg_tile * tiles, صحيح Npad,
                                صحيح K, dpct::قطار_ptr stream) {
    استعمال E        = قسم_نالو S::EB;
    constexpr صحيح V = S::VNNI;
    ثابت صحيح kqs   = K / V;
    stream->parallel_for(sycl::range<1>((size_t) Npad * kqs), [=](sycl::id<1> id) {
        ثابت size_t idx = id[0];
        ثابت صحيح    kq  = idx / Npad;
        ثابت صحيح    n   = idx - (size_t) kq * Npad;
        ثابت ggml_sycl_gg_tile tile = tiles[n / S::BN];
        ثابت صحيح    صف = tile.n0 + n % S::BN;
        // one guarded load run per work-item, as a per-element select costs ~1.5% prefill
        قسم_نالو S::tsb vals[V] = {};
        إذا (صف < tile.n1) {
            ثابت عائم * src = y + (size_t) صف * K + V * kq;
#هدايت unroll
            لكل (صحيح v = 0; v < V; ++v) {
                vals[v] = E::cvt(src[v]);
            }
        }
        قسم_نالو S::tsb * ٻاهريون = packed + ((size_t) kq * Npad + n) * V;
#هدايت unroll
        لكل (صحيح v = 0; v < V; ++v) {
            ٻاهريون[v] = vals[v];
        }
    });
}

// q8_0 and the k-quants take only the grouped path. The plain kernel decodes A again for every BN columns
// of a dense batch, and for these formats that costs more than the one dequantization of the library GEMM.
سانچو <قسم_نالو T> ساكن constexpr منطقي fg_plain_ok() {
    أرجع !std::is_same_v<T, بلاڪ_q8_0> && !std::is_same_v<T, بلاڪ_q4_K> && !std::is_same_v<T, بلاڪ_q5_K> &&
           !std::is_same_v<T, بلاڪ_q6_K>;
}

سانچو <قسم_نالو T, منطقي R> بنية fg_ٽيگ {
    استعمال type                     = T;
    ساكن constexpr منطقي reordered = R;
};

// calls f(fg_tag<block_q_t, reordered>{}) for the weight format and layout; false if it has no A stage
سانچو <قسم_نالو T, قسم_نالو F> ساكن منطقي fg_گهمو_layout(منطقي reordered, F && f) {
    إذا (!reordered) {
        f(fg_ٽيگ<T, خطأ>{});
        أرجع سچ;
    }
    إذا constexpr (fg_soa<T>::supported) {
        f(fg_ٽيگ<T, سچ>{});
        أرجع سچ;
    }
    أرجع خطأ;
}

سانچو <قسم_نالو F> ساكن منطقي fg_گهمو_type(ggml_type type, منطقي reordered, F && f) {
    بدل (type) {
        حالة GGML_TYPE_IQ4_NL:  أرجع fg_گهمو_layout<بلاڪ_iq4_nl>(reordered, f);
        حالة GGML_TYPE_IQ3_S:   أرجع fg_گهمو_layout<بلاڪ_iq3_s>(reordered, f);
        حالة GGML_TYPE_IQ4_XS:  أرجع fg_گهمو_layout<بلاڪ_iq4_xs>(reordered, f);
        حالة GGML_TYPE_IQ3_XXS: أرجع fg_گهمو_layout<بلاڪ_iq3_xxs>(reordered, f);
        حالة GGML_TYPE_IQ2_XXS: أرجع fg_گهمو_layout<بلاڪ_iq2_xxs>(reordered, f);
        حالة GGML_TYPE_IQ2_XS:  أرجع fg_گهمو_layout<بلاڪ_iq2_xs>(reordered, f);
        حالة GGML_TYPE_IQ2_S:   أرجع fg_گهمو_layout<بلاڪ_iq2_s>(reordered, f);
        حالة GGML_TYPE_IQ1_S:   أرجع fg_گهمو_layout<بلاڪ_iq1_s>(reordered, f);
        حالة GGML_TYPE_IQ1_M:   أرجع fg_گهمو_layout<بلاڪ_iq1_m>(reordered, f);
        حالة GGML_TYPE_Q8_0:    أرجع fg_گهمو_layout<بلاڪ_q8_0>(reordered, f);
        حالة GGML_TYPE_Q4_K:    أرجع fg_گهمو_layout<بلاڪ_q4_K>(reordered, f);
        حالة GGML_TYPE_Q5_K:    أرجع fg_گهمو_layout<بلاڪ_q5_K>(reordered, f);
        حالة GGML_TYPE_Q6_K:    أرجع fg_گهمو_layout<بلاڪ_q6_K>(reordered, f);
        افتراضي:                أرجع خطأ;
    }
}

سانچو <قسم_نالو S>
ساكن منطقي fg_fused_لاڳيتو(ggml_type src0_type, منطقي src0_reordered, ثابت فراغ * src0, ثابت فراغ * src1, ggml_type src1_type,
                         عائم * dst, int64_t M, int64_t N, int64_t K, int64_t ldd, ggml_sycl_pool & pool,
                         dpct::قطار_ptr stream) {
    ثابت int64_t groups_n = (N + S::BN - 1) / S::BN;
    ثابت int64_t groups_m = (M + S::SG_صفون - 1) / S::SG_صفون;
    ثابت صحيح     Npad     = (صحيح) (groups_n * S::BN);

    // src1 is read in its own type: one pass, converted in registers only if B differs
    ggml_sycl_pool_alloc<قسم_نالو S::tsb> packed_b(pool, (size_t) K * Npad);
    إذا (src1_type == GGML_TYPE_F16) {
        fused_gemm_pack_b<قسم_نالو S::EB>((ثابت sycl::half *) src1, packed_b.get(), (صحيح) N, Npad, (صحيح) K, stream);
    } وإلا إذا (src1_type == GGML_TYPE_BF16) {
        fused_gemm_pack_b<قسم_نالو S::EB>((ثابت fg_bf16 *) src1, packed_b.get(), (صحيح) N, Npad, (صحيح) K, stream);
    } وإلا {
        fused_gemm_pack_b<قسم_نالو S::EB>((ثابت عائم *) src1, packed_b.get(), (صحيح) N, Npad, (صحيح) K, stream);
    }

    ثابت قسم_نالو S::tsb * packed = packed_b.get();
    أرجع fg_گهمو_type(src0_type, src0_reordered, [&](خودڪار ٽيگ) {
        استعمال بلاڪ_q_t = قسم_نالو decltype(ٽيگ)::type;
        إذا constexpr (fg_plain_ok<بلاڪ_q_t>()) {
            fused_dequant_gemm_launch<S, بلاڪ_q_t, decltype(ٽيگ)::reordered>(src0, packed, dst, (صحيح) M, (صحيح) N,
                                                                              Npad, (صحيح) K, (صحيح) ldd, groups_n,
                                                                              groups_m, stream);
        }
    });
}

منطقي ggml_sycl_fused_dequant_gemm(ggml_type src0_type, منطقي src0_reordered, ثابت فراغ * src0, ثابت فراغ * src1, ggml_type src1_type,
                                  int32_t src1_prec, عائم * dst, int64_t M, int64_t N, int64_t K, int64_t ldd,
                                  ggml_sycl_pool & pool, dpct::قطار_ptr stream) {
    // every BN columns dequantize A again, so wide N is left to the library GEMM.
    // src0_reordered: the persistent weight reorder may have been installed on this
    // tensor by an earlier small-batch mul_mat; the A stage then has to decode the
    // SoA layout. Types without an SoA stage (fg_soa<T>::supported == false)
    // decline via fg_visit_layout and the caller falls through to the
    // reorder-aware conversion route.
    منطقي plain_ok = خطأ;
    fg_گهمو_type(src0_type, src0_reordered, [&](خودڪار ٽيگ) { plain_ok = fg_plain_ok<قسم_نالو decltype(ٽيگ)::type>(); });
    إذا (g_ggml_sycl_dynamic_precision == GGML_SYCL_DYNAMIC_PRECISION_F32 ||
        !ggml_sycl_xmx_gather_type_enabled(src0_type) || !plain_ok) {
        أرجع خطأ;
    }
    إذا (src1_type != GGML_TYPE_F32 && src1_type != GGML_TYPE_F16 && src1_type != GGML_TYPE_BF16) {
        أرجع خطأ;
    }
    إذا (!ggml_sycl_fused_dequant_gemm_shape_ok(src0_type, M, N, K, ldd)) {
        أرجع خطأ;
    }
    ثابت صحيح combo = fg_pick_combo(stream, src1_type, src1_prec);
    إذا (combo < 0) {
        أرجع خطأ;
    }
    منطقي launched = خطأ;
    fg_گهمو_combo(combo, [&](خودڪار s) {
        launched = fg_fused_لاڳيتو<decltype(s)>(src0_type, src0_reordered, src0, src1, src1_type, dst, M, N, K, ldd, pool, stream);
    });
    أرجع launched;
}

سانچو <قسم_نالو S>
ساكن منطقي fg_grouped_لاڳيتو(ggml_type src0_type, منطقي reordered, ثابت فراغ * src0_بنياد, size_t expert_stride,
                           ثابت عائم * src1, عائم * dst, ثابت int64_t * expert_صف_offsets, int64_t n_as, int64_t M,
                           int64_t K, std::vector<ggml_sycl_gg_tile> & tiles, ggml_sycl_pool & pool,
                           dpct::قطار_ptr stream) {
    // the host knows every slice, so it lays out the work-groups: no search on the device
    tiles.clear();
    لكل (int64_t e = 0; e < n_as; ++e) {
        ثابت int64_t end = expert_صف_offsets[e + 1];
        لكل (int64_t n0 = expert_صف_offsets[e]; n0 < end; n0 += S::BN) {
            tiles.push_back({ (int32_t) e, (int32_t) n0, (int32_t) std::min<int64_t>(n0 + S::BN, end) });
        }
    }
    ثابت int64_t n_tiles  = tiles.size();
    ثابت int64_t groups_m = (M + S::SG_صفون - 1) / S::SG_صفون;
    ثابت صحيح     Npad     = (صحيح) (n_tiles * S::BN);

    ثابت int64_t max_tiles = grouped_gemm_max_tiles(expert_صف_offsets[n_as], n_as, S::BN);
    GGML_ASSERT(n_tiles <= max_tiles);
    ggml_sycl_pool_alloc<ggml_sycl_gg_tile> tiles_dev(pool, max_tiles);
    SYCL_CHECK(CHECK_TRY_ERROR(stream->يادگيري_نقل(tiles_dev.get(), tiles.معلومات(), n_tiles * حجم(ggml_sycl_gg_tile))));

    ggml_sycl_pool_alloc<قسم_نالو S::tsb> packed_b(pool, (size_t) K * max_tiles * S::BN);
    grouped_gemm_pack_b<S>(src1, packed_b.get(), tiles_dev.get(), Npad, (صحيح) K, stream);

    ثابت قسم_نالو S::tsb *   packed    = packed_b.get();
    ثابت ggml_sycl_gg_tile * tiles_ptr = tiles_dev.get();
    ثابت حرف *              src0_dd   = (ثابت حرف *) src0_بنياد;
    أرجع fg_گهمو_type(src0_type, reordered, [&](خودڪار ٽيگ) {
        استعمال T = decltype(ٽيگ);
        grouped_dequant_gemm_launch<S, قسم_نالو T::type, T::reordered>(src0_dd, expert_stride, tiles_ptr, packed, dst,
                                                                       (صحيح) M, Npad, (صحيح) K, n_tiles, groups_m,
                                                                       stream);
    });
}

منطقي ggml_sycl_grouped_dequant_gemm(ggml_type src0_type, منطقي reordered, ثابت فراغ * src0_بنياد, size_t expert_stride,
                                    ثابت عائم * src1, int32_t src1_prec, عائم * dst,
                                    ثابت int64_t * expert_صف_offsets, int64_t n_as, int64_t M, int64_t K,
                                    int64_t ڪل_صفون, std::vector<ggml_sycl_gg_tile> & tiles,
                                    ggml_sycl_pool & pool, dpct::قطار_ptr stream) {
    int64_t n_سرگرم = 0;
    لكل (int64_t e = 0; e < n_as; ++e) {
        n_سرگرم += expert_صف_offsets[e + 1] > expert_صف_offsets[e];
    }
    إذا (g_ggml_sycl_dynamic_precision == GGML_SYCL_DYNAMIC_PRECISION_F32 ||
        !ggml_sycl_xmx_gather_type_enabled(src0_type) || !fg_گهمو_type(src0_type, reordered, [](خودڪار) {})) {
        أرجع خطأ;
    }
    إذا (!ggml_sycl_grouped_dequant_gemm_shape_ok(src0_type, M, K, ڪل_صفون, n_سرگرم)) {
        أرجع خطأ;
    }
    ثابت صحيح combo = fg_pick_combo(stream, GGML_TYPE_F32, src1_prec);
    إذا (combo < 0) {
        أرجع خطأ;
    }
    منطقي launched = خطأ;
    fg_گهمو_combo(combo, [&](خودڪار s) {
        launched = fg_grouped_لاڳيتو<decltype(s)>(src0_type, reordered, src0_بنياد, expert_stride, src1, dst,
                                               expert_صف_offsets, n_as, M, K, tiles, pool, stream);
    });
    أرجع launched;
}
