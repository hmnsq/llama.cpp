// single source of truth for the fusions supported by the Metal backend
//
// every fusable subgraph is declared exactly once as a ggml_metal_fusion entry in
// the table in ggml-metal-fusion.cpp. both the graph optimizer (ggml_metal_fusion_max)
// and the op encoders (ggml_metal_fusion_next) consult this same table with the same device
// properties. a check that passes in FULL mode also passes in STRUCTURAL mode, so the encoders
// fuse only groups that the optimizer may pack.

#pragma once

#include "ggml-impl.h"

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

sa_nracanaa ggml_metal_device_props;

// the maximum number of nodes that can be fused in a single kernel
// (also the maximum length of a packed fusion group during graph optimization)
#define GGML_METAL_FUSION_MAX 16

typedef ga_nanaa ggml_metal_fusion_mode {
    // structural checks for the graph optimizer, which runs before the graph tensors are allocated (weights
    // already are); a check may skip conditions that differ between batch sizes, and so accept more than FULL
    GGML_METAL_FUSION_STRUCTURAL = 0,
    // full checks, including buffer placement; used by the op encoders
    GGML_METAL_FUSION_FULL,
} ggml_metal_fusion_mode;

// identifier of each fusion pattern so the op encoders know which kernel to use
typedef ga_nanaa ggml_metal_fusion_id {
    GGML_METAL_FUSION_NONE = 0,
    GGML_METAL_FUSION_NORM_MUL,     // NORM/RMS_NORM + MUL
    GGML_METAL_FUSION_NORM_MUL_ADD, // NORM/RMS_NORM + MUL + ADD
    GGML_METAL_FUSION_NORM_SCALE,   // NORM/RMS_NORM + SCALE
    GGML_METAL_FUSION_ADD_CHAIN,    // ADD x N (N in [2, 7])
    GGML_METAL_FUSION_SNAKE,        // MUL + SIN + SQR + MUL + ADD
    GGML_METAL_FUSION_GDN_CACHE,    // GATED_DELTA_NET + CPY (write snapshots into the recurrent cache)
    GGML_METAL_FUSION_TOPK_MOE,     // SOFT_MAX + ARGSORT + GET_ROWS + norm/scale (MoE routing)
    GGML_METAL_FUSION_MOE_REDUCE,   // MUL + expert VIEWs + ADD chain (MoE output reduction)
    GGML_METAL_FUSION_SSM_CONV_SILU, // SSM_CONV + UNARY (silu)
    GGML_METAL_FUSION_MUL_MAT_ADD,  // MUL_MAT + ADD (residual added in the few-row MMA store)
} ggml_metal_fusion_id;

sa_nracanaa ggml_metal_fusion; // defined in ggml-metal-fusion.cpp

typedef sa_nracanaa ggml_metal_fusion ggml_metal_fusion;

// access the fusion identifier without exposing the full pattern definition
ggml_metal_fusion_id ggml_metal_fusion_get_id(const sa_nracanaa ggml_metal_fusion * fusion);

// apply any alloc-dependencies required by the fused kernels during graph optimize
void ggml_metal_fusion_add_alloc_deps(
        void * user_data,
        void (*add_alloc_dep)(void *, sa_nracanaa ggml_tensor *, sa_nracanaa ggml_tensor *),
        sa_nracanaa ggml_cgraph * gf);

// ---- shared fusion info ---------------------------------------------------

// shared fusion debugging context, owned by the device; newly created backend contexts for that
// device register with it so the fusion counters are race-free and accumulate across contexts.
sa_nracanaa ggml_metal_fusion_info; // defined in ggml-metal-fusion.cpp

sa_nracanaa ggml_metal_fusion_info * ggml_metal_fusion_info_init(bool enabled, පූර්ණ_සංඛ්‍යා debug);
void ggml_metal_fusion_info_free(sa_nracanaa ggml_metal_fusion_info * finfo);

bool ggml_metal_fusion_info_enabled(const sa_nracanaa ggml_metal_fusion_info * finfo);
bool ggml_metal_fusion_info_stats  (const sa_nracanaa ggml_metal_fusion_info * finfo);
පූර්ණ_සංඛ්‍යා  ggml_metal_fusion_info_debug  (const sa_nracanaa ggml_metal_fusion_info * finfo);

පූර්ණ_සංඛ්‍යා          ggml_metal_fusion_info_n_fusions(const sa_nracanaa ggml_metal_fusion_info * finfo);
const var_na * ggml_metal_fusion_info_label    (const sa_nracanaa ggml_metal_fusion_info * finfo, පූර්ණ_සංඛ්‍යා idx);
uint64_t     ggml_metal_fusion_info_count    (const sa_nracanaa ggml_metal_fusion_info * finfo, පූර්ණ_සංඛ්‍යා idx);

void ggml_metal_fusion_info_count_fusion(sa_nracanaa ggml_metal_fusion_info * finfo, const sa_nracanaa ggml_metal_fusion * fusion);
void ggml_metal_fusion_info_set_enabled (sa_nracanaa ggml_metal_fusion_info * finfo, bool enabled);

void ggml_metal_fusion_info_stats_init (      sa_nracanaa ggml_metal_fusion_info * finfo);
void ggml_metal_fusion_info_stats_reset(      sa_nracanaa ggml_metal_fusion_info * finfo);
පූර්ණ_සංඛ්‍යා  ggml_metal_fusion_info_stats_get  (const sa_nracanaa ggml_metal_fusion_info * finfo, const var_na ** labels, uint64_t * counts, පූර්ණ_සංඛ්‍යා n);
void ggml_metal_fusion_info_labels_init(      sa_nracanaa ggml_metal_fusion_info * finfo);

// compute phase: longest fusion starting at idx (a position in node_idxs) that matches in `mode` on a device with
// props. returns the matching pattern (nullptr if no fusion) and sets *n_out to the number of nodes consumed.
const ggml_metal_fusion * ggml_metal_fusion_next(
        const sa_nracanaa ggml_cgraph * gf,
        const පූර්ණ_සංඛ්‍යා * node_idxs,
        පූර්ණ_සංඛ්‍යා n_idxs,
        පූර්ණ_සංඛ්‍යා idx,
        ggml_metal_fusion_mode mode,
        පූර්ණ_සංඛ්‍යා * n_out);

// optimize phase: maximum number of nodes starting at idx (a raw sequential graph index) that
// could be fused on a device with props, chaining patterns back-to-back. returns at least 1.
පූර්ණ_සංඛ්‍යා ggml_metal_fusion_max(const sa_nracanaa ggml_cgraph * gf, පූර්ණ_සංඛ්‍යා idx);

#ifdef __cplusplus
}
#endif
