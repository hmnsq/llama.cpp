// single source of truth for the fusions supported by the Metal backend
//
// every fusable subgraph is declared exactly once as a ggml_metal_fusion entry in
// the table in ggml-metal-fusion.cpp. both the graph optimizer (ggml_metal_fusion_max)
// and the op encoders (ggml_metal_fusion_next) consult this same table with the same device
// properties. a check that passes in FULL mode also passes in STRUCTURAL mode, so the encoders
// fuse only groups that the optimizer may pack.

#رهنمود once

#ضمن "ggml-impl.h"

#ضمن <stdbool.h>

#اگر_تعریف_شده __cplusplus
خارجي "C" {
#پایان_اگر

بنية ggml_metal_device_props;

// the maximum number of nodes that can be fused in a single kernel
// (also the maximum length of a packed fusion group during graph optimization)
#تعریف_کن GGML_METAL_FUSION_MAX 16

تعريفنوع تعداد ggml_metal_fusion_نما {
    // structural checks for the graph optimizer, which runs before the graph tensors are allocated (weights
    // already are); a check may skip conditions that differ between batch sizes, and so accept more than FULL
    GGML_METAL_FUSION_STRUCTURAL = 0,
    // full checks, including buffer placement; used by the op encoders
    GGML_METAL_FUSION_پر,
} ggml_metal_fusion_نما;

// identifier of each fusion pattern so the op encoders know which kernel to use
تعريفنوع تعداد ggml_metal_fusion_id {
    GGML_METAL_FUSION_NONE = 0,
    GGML_METAL_FUSION_NORM_MUL,     // NORM/RMS_NORM + MUL
    GGML_METAL_FUSION_NORM_MUL_جمع_کن, // NORM/RMS_NORM + MUL + ADD
    GGML_METAL_FUSION_NORM_SCALE,   // NORM/RMS_NORM + SCALE
    GGML_METAL_FUSION_جمع_کن_CHAIN,    // ADD x N (N in [2, 7])
    GGML_METAL_FUSION_SNAKE,        // MUL + SIN + SQR + MUL + ADD
    GGML_METAL_FUSION_GDN_CACHE,    // GATED_DELTA_NET + CPY (write snapshots into the recurrent cache)
    GGML_METAL_FUSION_TOPK_MOE,     // SOFT_MAX + ARGSORT + GET_ROWS + norm/scale (MoE routing)
    GGML_METAL_FUSION_MOE_REDUCE,   // MUL + expert VIEWs + ADD chain (MoE output reduction)
    GGML_METAL_FUSION_SSM_CONV_SILU, // SSM_CONV + UNARY (silu)
    GGML_METAL_FUSION_MUL_MAT_جمع_کن,  // MUL_MAT + ADD (residual added in the few-row MMA store)
} ggml_metal_fusion_id;

بنية ggml_metal_fusion; // defined in ggml-metal-fusion.cpp

تعريفنوع بنية ggml_metal_fusion ggml_metal_fusion;

// access the fusion identifier without exposing the full pattern definition
ggml_metal_fusion_id ggml_metal_fusion_get_id(ثابت بنية ggml_metal_fusion * fusion);

// apply any alloc-dependencies required by the fused kernels during graph optimize
فراغ ggml_metal_fusion_جمع_کن_alloc_deps(
        فراغ * user_داده,
        فراغ (*جمع_کن_alloc_dep)(فراغ *, بنية ggml_tensor *, بنية ggml_tensor *),
        بنية ggml_cgraph * gf);

// ---- shared fusion info ---------------------------------------------------

// shared fusion debugging context, owned by the device; newly created backend contexts for that
// device register with it so the fusion counters are race-free and accumulate across contexts.
بنية ggml_metal_fusion_info; // defined in ggml-metal-fusion.cpp

بنية ggml_metal_fusion_info * ggml_metal_fusion_info_init(bool enabled, صحيح debug);
فراغ ggml_metal_fusion_info_free(بنية ggml_metal_fusion_info * finfo);

bool ggml_metal_fusion_info_enabled(ثابت بنية ggml_metal_fusion_info * finfo);
bool ggml_metal_fusion_info_stats  (ثابت بنية ggml_metal_fusion_info * finfo);
صحيح  ggml_metal_fusion_info_debug  (ثابت بنية ggml_metal_fusion_info * finfo);

صحيح          ggml_metal_fusion_info_n_fusions(ثابت بنية ggml_metal_fusion_info * finfo);
ثابت حرف * ggml_metal_fusion_info_label    (ثابت بنية ggml_metal_fusion_info * finfo, صحيح idx);
uint64_t     ggml_metal_fusion_info_شمار    (ثابت بنية ggml_metal_fusion_info * finfo, صحيح idx);

فراغ ggml_metal_fusion_info_شمار_fusion(بنية ggml_metal_fusion_info * finfo, ثابت بنية ggml_metal_fusion * fusion);
فراغ ggml_metal_fusion_info_مجموعه_enabled (بنية ggml_metal_fusion_info * finfo, bool enabled);

فراغ ggml_metal_fusion_info_stats_init (      بنية ggml_metal_fusion_info * finfo);
فراغ ggml_metal_fusion_info_stats_reset(      بنية ggml_metal_fusion_info * finfo);
صحيح  ggml_metal_fusion_info_stats_get  (ثابت بنية ggml_metal_fusion_info * finfo, ثابت حرف ** labels, uint64_t * counts, صحيح n);
فراغ ggml_metal_fusion_info_labels_init(      بنية ggml_metal_fusion_info * finfo);

// compute phase: longest fusion starting at idx (a position in node_idxs) that matches in `mode` on a device with
// props. returns the matching pattern (nullptr if no fusion) and sets *n_out to the number of nodes consumed.
ثابت ggml_metal_fusion * ggml_metal_fusion_بعدی(
        ثابت بنية ggml_cgraph * gf,
        ثابت صحيح * گره_idxs,
        صحيح n_idxs,
        صحيح idx,
        ggml_metal_fusion_نما نما,
        صحيح * n_بیرون);

// optimize phase: maximum number of nodes starting at idx (a raw sequential graph index) that
// could be fused on a device with props, chaining patterns back-to-back. returns at least 1.
صحيح ggml_metal_fusion_max(ثابت بنية ggml_cgraph * gf, صحيح idx);

#اگر_تعریف_شده __cplusplus
}
#پایان_اگر
