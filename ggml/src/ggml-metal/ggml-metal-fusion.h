// single source of truth for the fusions supported by the Metal backend
//
// every fusable subgraph is declared exactly once as a ggml_metal_fusion entry in
// the table in ggml-metal-fusion.cpp. both the graph optimizer (ggml_metal_fusion_max)
// and the op encoders (ggml_metal_fusion_next) consult this same table with the same device
// properties. a check that passes in FULL mode also passes in STRUCTURAL mode, so the encoders
// fuse only groups that the optimizer may pack.

#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <stdbool.h>

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __cplusplus
ਬਾਹਰੀ "C" {
#ਜੇ_ਅੰਤ

ਬਣਤਰ ggml_metal_device_props;

// the maximum number of nodes that can be fused in a single kernel
// (also the maximum length of a packed fusion group during graph optimization)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_METAL_FUSION_MAX 16

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਗਿਣਤੀ ggml_metal_fusion_ਬਹੁਲਕ {
    // structural checks for the graph optimizer, which runs before the graph tensors are allocated (weights
    // already are); a check may skip conditions that differ between batch sizes, and so accept more than FULL
    GGML_METAL_FUSION_STRUCTURAL = 0,
    // full checks, including buffer placement; used by the op encoders
    GGML_METAL_FUSION_ਭਰਿਆ,
} ggml_metal_fusion_ਬਹੁਲਕ;

// identifier of each fusion pattern so the op encoders know which kernel to use
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਗਿਣਤੀ ggml_metal_fusion_id {
    GGML_METAL_FUSION_NONE = 0,
    GGML_METAL_FUSION_NORM_MUL,     // NORM/RMS_NORM + MUL
    GGML_METAL_FUSION_NORM_MUL_ਜੋੜੋ, // NORM/RMS_NORM + MUL + ADD
    GGML_METAL_FUSION_NORM_SCALE,   // NORM/RMS_NORM + SCALE
    GGML_METAL_FUSION_ਜੋੜੋ_CHAIN,    // ADD x N (N in [2, 7])
    GGML_METAL_FUSION_SNAKE,        // MUL + SIN + SQR + MUL + ADD
    GGML_METAL_FUSION_GDN_CACHE,    // GATED_DELTA_NET + CPY (write snapshots into the recurrent cache)
    GGML_METAL_FUSION_TOPK_MOE,     // SOFT_MAX + ARGSORT + GET_ROWS + norm/scale (MoE routing)
    GGML_METAL_FUSION_MOE_REDUCE,   // MUL + expert VIEWs + ADD chain (MoE output reduction)
    GGML_METAL_FUSION_SSM_CONV_SILU, // SSM_CONV + UNARY (silu)
    GGML_METAL_FUSION_MUL_MAT_ਜੋੜੋ,  // MUL_MAT + ADD (residual added in the few-row MMA store)
} ggml_metal_fusion_id;

ਬਣਤਰ ggml_metal_fusion; // defined in ggml-metal-fusion.cpp

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ ggml_metal_fusion ggml_metal_fusion;

// access the fusion identifier without exposing the full pattern definition
ggml_metal_fusion_id ggml_metal_fusion_get_id(ਅਚਲ ਬਣਤਰ ggml_metal_fusion * fusion);

// apply any alloc-dependencies required by the fused kernels during graph optimize
ਖਾਲੀ ggml_metal_fusion_ਜੋੜੋ_alloc_deps(
        ਖਾਲੀ * user_ਅੰਕੜੇ,
        ਖਾਲੀ (*ਜੋੜੋ_alloc_dep)(ਖਾਲੀ *, ਬਣਤਰ ggml_tensor *, ਬਣਤਰ ggml_tensor *),
        ਬਣਤਰ ggml_cgraph * gf);

// ---- shared fusion info ---------------------------------------------------

// shared fusion debugging context, owned by the device; newly created backend contexts for that
// device register with it so the fusion counters are race-free and accumulate across contexts.
ਬਣਤਰ ggml_metal_fusion_info; // defined in ggml-metal-fusion.cpp

ਬਣਤਰ ggml_metal_fusion_info * ggml_metal_fusion_info_init(bool enabled, ਪੂਰਨ debug);
ਖਾਲੀ ggml_metal_fusion_info_free(ਬਣਤਰ ggml_metal_fusion_info * finfo);

bool ggml_metal_fusion_info_enabled(ਅਚਲ ਬਣਤਰ ggml_metal_fusion_info * finfo);
bool ggml_metal_fusion_info_stats  (ਅਚਲ ਬਣਤਰ ggml_metal_fusion_info * finfo);
ਪੂਰਨ  ggml_metal_fusion_info_debug  (ਅਚਲ ਬਣਤਰ ggml_metal_fusion_info * finfo);

ਪੂਰਨ          ggml_metal_fusion_info_n_fusions(ਅਚਲ ਬਣਤਰ ggml_metal_fusion_info * finfo);
ਅਚਲ ਅੱਖਰ * ggml_metal_fusion_info_label    (ਅਚਲ ਬਣਤਰ ggml_metal_fusion_info * finfo, ਪੂਰਨ idx);
uint64_t     ggml_metal_fusion_info_ਗਿਣਤੀ    (ਅਚਲ ਬਣਤਰ ggml_metal_fusion_info * finfo, ਪੂਰਨ idx);

ਖਾਲੀ ggml_metal_fusion_info_ਗਿਣਤੀ_fusion(ਬਣਤਰ ggml_metal_fusion_info * finfo, ਅਚਲ ਬਣਤਰ ggml_metal_fusion * fusion);
ਖਾਲੀ ggml_metal_fusion_info_ਸੈੱਟ_enabled (ਬਣਤਰ ggml_metal_fusion_info * finfo, bool enabled);

ਖਾਲੀ ggml_metal_fusion_info_stats_init (      ਬਣਤਰ ggml_metal_fusion_info * finfo);
ਖਾਲੀ ggml_metal_fusion_info_stats_reset(      ਬਣਤਰ ggml_metal_fusion_info * finfo);
ਪੂਰਨ  ggml_metal_fusion_info_stats_get  (ਅਚਲ ਬਣਤਰ ggml_metal_fusion_info * finfo, ਅਚਲ ਅੱਖਰ ** labels, uint64_t * counts, ਪੂਰਨ n);
ਖਾਲੀ ggml_metal_fusion_info_labels_init(      ਬਣਤਰ ggml_metal_fusion_info * finfo);

// compute phase: longest fusion starting at idx (a position in node_idxs) that matches in `mode` on a device with
// props. returns the matching pattern (nullptr if no fusion) and sets *n_out to the number of nodes consumed.
ਅਚਲ ggml_metal_fusion * ggml_metal_fusion_ਅਗਲਾ(
        ਅਚਲ ਬਣਤਰ ggml_cgraph * gf,
        ਅਚਲ ਪੂਰਨ * ਗੰਢ_idxs,
        ਪੂਰਨ n_idxs,
        ਪੂਰਨ idx,
        ggml_metal_fusion_ਬਹੁਲਕ ਬਹੁਲਕ,
        ਪੂਰਨ * n_ਬਾਹਰਲਾ);

// optimize phase: maximum number of nodes starting at idx (a raw sequential graph index) that
// could be fused on a device with props, chaining patterns back-to-back. returns at least 1.
ਪੂਰਨ ggml_metal_fusion_max(ਅਚਲ ਬਣਤਰ ggml_cgraph * gf, ਪੂਰਨ idx);

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __cplusplus
}
#ਜੇ_ਅੰਤ
