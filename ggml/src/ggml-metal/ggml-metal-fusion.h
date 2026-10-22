// single source of truth for the fusions supported by the Metal backend
//
// every fusable subgraph is declared exactly once as a ggml_metal_fusion entry in
// the table in ggml-metal-fusion.cpp. both the graph optimizer (ggml_metal_fusion_max)
// and the op encoders (ggml_metal_fusion_next) consult this same table with the same device
// properties. a check that passes in FULL mode also passes in STRUCTURAL mode, so the encoders
// fuse only groups that the optimizer may pack.

#సూచన once

#అన్తర్భావ "ggml-impl.h"

#అన్తర్భావ <stdbool.h>

#నిర్వచితమైతే __cplusplus
బాహ్య "C" {
#అయితే_ముగింపు

నిర్మాణం ggml_metal_device_props;

// the maximum number of nodes that can be fused in a single kernel
// (also the maximum length of a packed fusion group during graph optimization)
#నిర్వచించు GGML_METAL_FUSION_MAX 16

రకనిర్వచనం గణన ggml_metal_fusion_బహుళకం {
    // structural checks for the graph optimizer, which runs before the graph tensors are allocated (weights
    // already are); a check may skip conditions that differ between batch sizes, and so accept more than FULL
    GGML_METAL_FUSION_STRUCTURAL = 0,
    // full checks, including buffer placement; used by the op encoders
    GGML_METAL_FUSION_నిండిన,
} ggml_metal_fusion_బహుళకం;

// identifier of each fusion pattern so the op encoders know which kernel to use
రకనిర్వచనం గణన ggml_metal_fusion_id {
    GGML_METAL_FUSION_NONE = 0,
    GGML_METAL_FUSION_NORM_MUL,     // NORM/RMS_NORM + MUL
    GGML_METAL_FUSION_NORM_MUL_కూడు, // NORM/RMS_NORM + MUL + ADD
    GGML_METAL_FUSION_NORM_SCALE,   // NORM/RMS_NORM + SCALE
    GGML_METAL_FUSION_కూడు_CHAIN,    // ADD x N (N in [2, 7])
    GGML_METAL_FUSION_SNAKE,        // MUL + SIN + SQR + MUL + ADD
    GGML_METAL_FUSION_GDN_CACHE,    // GATED_DELTA_NET + CPY (write snapshots into the recurrent cache)
    GGML_METAL_FUSION_TOPK_MOE,     // SOFT_MAX + ARGSORT + GET_ROWS + norm/scale (MoE routing)
    GGML_METAL_FUSION_MOE_REDUCE,   // MUL + expert VIEWs + ADD chain (MoE output reduction)
    GGML_METAL_FUSION_SSM_CONV_SILU, // SSM_CONV + UNARY (silu)
    GGML_METAL_FUSION_MUL_MAT_కూడు,  // MUL_MAT + ADD (residual added in the few-row MMA store)
} ggml_metal_fusion_id;

నిర్మాణం ggml_metal_fusion; // defined in ggml-metal-fusion.cpp

రకనిర్వచనం నిర్మాణం ggml_metal_fusion ggml_metal_fusion;

// access the fusion identifier without exposing the full pattern definition
ggml_metal_fusion_id ggml_metal_fusion_get_id(స్థిరరాశి నిర్మాణం ggml_metal_fusion * fusion);

// apply any alloc-dependencies required by the fused kernels during graph optimize
శూన్య ggml_metal_fusion_కూడు_alloc_deps(
        శూన్య * user_దత్తాంశం,
        శూన్య (*కూడు_alloc_dep)(శూన్య *, నిర్మాణం ggml_tensor *, నిర్మాణం ggml_tensor *),
        నిర్మాణం ggml_cgraph * gf);

// ---- shared fusion info ---------------------------------------------------

// shared fusion debugging context, owned by the device; newly created backend contexts for that
// device register with it so the fusion counters are race-free and accumulate across contexts.
నిర్మాణం ggml_metal_fusion_info; // defined in ggml-metal-fusion.cpp

నిర్మాణం ggml_metal_fusion_info * ggml_metal_fusion_info_init(bool enabled, పూర్ణాంకం debug);
శూన్య ggml_metal_fusion_info_free(నిర్మాణం ggml_metal_fusion_info * finfo);

bool ggml_metal_fusion_info_enabled(స్థిరరాశి నిర్మాణం ggml_metal_fusion_info * finfo);
bool ggml_metal_fusion_info_stats  (స్థిరరాశి నిర్మాణం ggml_metal_fusion_info * finfo);
పూర్ణాంకం  ggml_metal_fusion_info_debug  (స్థిరరాశి నిర్మాణం ggml_metal_fusion_info * finfo);

పూర్ణాంకం          ggml_metal_fusion_info_n_fusions(స్థిరరాశి నిర్మాణం ggml_metal_fusion_info * finfo);
స్థిరరాశి అక్షరం * ggml_metal_fusion_info_label    (స్థిరరాశి నిర్మాణం ggml_metal_fusion_info * finfo, పూర్ణాంకం idx);
uint64_t     ggml_metal_fusion_info_లెక్క    (స్థిరరాశి నిర్మాణం ggml_metal_fusion_info * finfo, పూర్ణాంకం idx);

శూన్య ggml_metal_fusion_info_లెక్క_fusion(నిర్మాణం ggml_metal_fusion_info * finfo, స్థిరరాశి నిర్మాణం ggml_metal_fusion * fusion);
శూన్య ggml_metal_fusion_info_సమితి_enabled (నిర్మాణం ggml_metal_fusion_info * finfo, bool enabled);

శూన్య ggml_metal_fusion_info_stats_init (      నిర్మాణం ggml_metal_fusion_info * finfo);
శూన్య ggml_metal_fusion_info_stats_reset(      నిర్మాణం ggml_metal_fusion_info * finfo);
పూర్ణాంకం  ggml_metal_fusion_info_stats_get  (స్థిరరాశి నిర్మాణం ggml_metal_fusion_info * finfo, స్థిరరాశి అక్షరం ** labels, uint64_t * counts, పూర్ణాంకం n);
శూన్య ggml_metal_fusion_info_labels_init(      నిర్మాణం ggml_metal_fusion_info * finfo);

// compute phase: longest fusion starting at idx (a position in node_idxs) that matches in `mode` on a device with
// props. returns the matching pattern (nullptr if no fusion) and sets *n_out to the number of nodes consumed.
స్థిరరాశి ggml_metal_fusion * ggml_metal_fusion_తదుపరి(
        స్థిరరాశి నిర్మాణం ggml_cgraph * gf,
        స్థిరరాశి పూర్ణాంకం * కణుపు_idxs,
        పూర్ణాంకం n_idxs,
        పూర్ణాంకం idx,
        ggml_metal_fusion_బహుళకం బహుళకం,
        పూర్ణాంకం * n_బహిర్గతం);

// optimize phase: maximum number of nodes starting at idx (a raw sequential graph index) that
// could be fused on a device with props, chaining patterns back-to-back. returns at least 1.
పూర్ణాంకం ggml_metal_fusion_max(స్థిరరాశి నిర్మాణం ggml_cgraph * gf, పూర్ణాంకం idx);

#నిర్వచితమైతే __cplusplus
}
#అయితే_ముగింపు
