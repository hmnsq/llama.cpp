// single source of truth for the fusions supported by the Metal backend
//
// every fusable subgraph is declared exactly once as a ggml_metal_fusion entry in
// the table in ggml-metal-fusion.cpp. both the graph optimizer (ggml_metal_fusion_max)
// and the op encoders (ggml_metal_fusion_next) consult this same table with the same device
// properties. a check that passes in FULL mode also passes in STRUCTURAL mode, so the encoders
// fuse only groups that the optimizer may pack.

#ಸೂಚನೆ once

#ಅನ್ತರ್ಭಾವ "ggml-impl.h"

#ಅನ್ತರ್ಭಾವ <stdbool.h>

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __cplusplus
ಬಾಹ್ಯ "C" {
#ಆದರೆ_ಅಂತ್ಯ

ರಚನೆ ggml_metal_device_props;

// the maximum number of nodes that can be fused in a single kernel
// (also the maximum length of a packed fusion group during graph optimization)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_METAL_FUSION_MAX 16

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಎಣಿಕೆ ggml_metal_fusion_ಬಹುಲಕ {
    // structural checks for the graph optimizer, which runs before the graph tensors are allocated (weights
    // already are); a check may skip conditions that differ between batch sizes, and so accept more than FULL
    GGML_METAL_FUSION_STRUCTURAL = 0,
    // full checks, including buffer placement; used by the op encoders
    GGML_METAL_FUSION_ತುಂಬಿದ,
} ggml_metal_fusion_ಬಹುಲಕ;

// identifier of each fusion pattern so the op encoders know which kernel to use
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಎಣಿಕೆ ggml_metal_fusion_id {
    GGML_METAL_FUSION_NONE = 0,
    GGML_METAL_FUSION_NORM_MUL,     // NORM/RMS_NORM + MUL
    GGML_METAL_FUSION_NORM_MUL_ಕೂಡಿಸು, // NORM/RMS_NORM + MUL + ADD
    GGML_METAL_FUSION_NORM_SCALE,   // NORM/RMS_NORM + SCALE
    GGML_METAL_FUSION_ಕೂಡಿಸು_CHAIN,    // ADD x N (N in [2, 7])
    GGML_METAL_FUSION_SNAKE,        // MUL + SIN + SQR + MUL + ADD
    GGML_METAL_FUSION_GDN_CACHE,    // GATED_DELTA_NET + CPY (write snapshots into the recurrent cache)
    GGML_METAL_FUSION_TOPK_MOE,     // SOFT_MAX + ARGSORT + GET_ROWS + norm/scale (MoE routing)
    GGML_METAL_FUSION_MOE_REDUCE,   // MUL + expert VIEWs + ADD chain (MoE output reduction)
    GGML_METAL_FUSION_SSM_CONV_SILU, // SSM_CONV + UNARY (silu)
    GGML_METAL_FUSION_MUL_MAT_ಕೂಡಿಸು,  // MUL_MAT + ADD (residual added in the few-row MMA store)
} ggml_metal_fusion_id;

ರಚನೆ ggml_metal_fusion; // defined in ggml-metal-fusion.cpp

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ ggml_metal_fusion ggml_metal_fusion;

// access the fusion identifier without exposing the full pattern definition
ggml_metal_fusion_id ggml_metal_fusion_get_id(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_metal_fusion * fusion);

// apply any alloc-dependencies required by the fused kernels during graph optimize
ಶೂನ್ಯ ggml_metal_fusion_ಕೂಡಿಸು_alloc_deps(
        ಶೂನ್ಯ * user_ದತ್ತಾಂಶ,
        ಶೂನ್ಯ (*ಕೂಡಿಸು_alloc_dep)(ಶೂನ್ಯ *, ರಚನೆ ggml_tensor *, ರಚನೆ ggml_tensor *),
        ರಚನೆ ggml_cgraph * gf);

// ---- shared fusion info ---------------------------------------------------

// shared fusion debugging context, owned by the device; newly created backend contexts for that
// device register with it so the fusion counters are race-free and accumulate across contexts.
ರಚನೆ ggml_metal_fusion_info; // defined in ggml-metal-fusion.cpp

ರಚನೆ ggml_metal_fusion_info * ggml_metal_fusion_info_init(bool enabled, ಪೂರ್ಣಾಂಕ debug);
ಶೂನ್ಯ ggml_metal_fusion_info_free(ರಚನೆ ggml_metal_fusion_info * finfo);

bool ggml_metal_fusion_info_enabled(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_metal_fusion_info * finfo);
bool ggml_metal_fusion_info_stats  (ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_metal_fusion_info * finfo);
ಪೂರ್ಣಾಂಕ  ggml_metal_fusion_info_debug  (ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_metal_fusion_info * finfo);

ಪೂರ್ಣಾಂಕ          ggml_metal_fusion_info_n_fusions(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_metal_fusion_info * finfo);
ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_metal_fusion_info_label    (ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_metal_fusion_info * finfo, ಪೂರ್ಣಾಂಕ idx);
uint64_t     ggml_metal_fusion_info_ಎಣಿಕೆ    (ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_metal_fusion_info * finfo, ಪೂರ್ಣಾಂಕ idx);

ಶೂನ್ಯ ggml_metal_fusion_info_ಎಣಿಕೆ_fusion(ರಚನೆ ggml_metal_fusion_info * finfo, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_metal_fusion * fusion);
ಶೂನ್ಯ ggml_metal_fusion_info_ಗಣ_enabled (ರಚನೆ ggml_metal_fusion_info * finfo, bool enabled);

ಶೂನ್ಯ ggml_metal_fusion_info_stats_init (      ರಚನೆ ggml_metal_fusion_info * finfo);
ಶೂನ್ಯ ggml_metal_fusion_info_stats_reset(      ರಚನೆ ggml_metal_fusion_info * finfo);
ಪೂರ್ಣಾಂಕ  ggml_metal_fusion_info_stats_get  (ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_metal_fusion_info * finfo, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ ** labels, uint64_t * counts, ಪೂರ್ಣಾಂಕ n);
ಶೂನ್ಯ ggml_metal_fusion_info_labels_init(      ರಚನೆ ggml_metal_fusion_info * finfo);

// compute phase: longest fusion starting at idx (a position in node_idxs) that matches in `mode` on a device with
// props. returns the matching pattern (nullptr if no fusion) and sets *n_out to the number of nodes consumed.
ಸ್ಥಿರಾಂಕ ggml_metal_fusion * ggml_metal_fusion_ಮುಂದಿನ(
        ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * gf,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * ಗಂಟು_idxs,
        ಪೂರ್ಣಾಂಕ n_idxs,
        ಪೂರ್ಣಾಂಕ idx,
        ggml_metal_fusion_ಬಹುಲಕ ಬಹುಲಕ,
        ಪೂರ್ಣಾಂಕ * n_ಹೊರಹೋಗು);

// optimize phase: maximum number of nodes starting at idx (a raw sequential graph index) that
// could be fused on a device with props, chaining patterns back-to-back. returns at least 1.
ಪೂರ್ಣಾಂಕ ggml_metal_fusion_max(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * gf, ಪೂರ್ಣಾಂಕ idx);

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __cplusplus
}
#ಆದರೆ_ಅಂತ್ಯ
