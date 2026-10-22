// single source of truth for the fusions supported by the Metal backend
//
// every fusable subgraph is declared exactly once as a ggml_metal_fusion entry in
// the table in ggml-metal-fusion.cpp. both the graph optimizer (ggml_metal_fusion_max)
// and the op encoders (ggml_metal_fusion_next) consult this same table with the same device
// properties. a check that passes in FULL mode also passes in STRUCTURAL mode, so the encoders
// fuse only groups that the optimizer may pack.

#pragma once

#അന്തര്ഭാവ "ggml-impl.h"

#അന്തര്ഭാവ <stdbool.h>

#ifdef __cplusplus
ബാഹ്യ "C" {
#endif

ഘടന ggml_metal_device_props;

// the maximum number of nodes that can be fused in a single kernel
// (also the maximum length of a packed fusion group during graph optimization)
#define GGML_METAL_FUSION_MAX 16

തരനിർവചനം എണ്ണൽ ggml_metal_fusion_mode {
    // structural checks for the graph optimizer, which runs before the graph tensors are allocated (weights
    // already are); a check may skip conditions that differ between batch sizes, and so accept more than FULL
    GGML_METAL_FUSION_STRUCTURAL = 0,
    // full checks, including buffer placement; used by the op encoders
    GGML_METAL_FUSION_FULL,
} ggml_metal_fusion_mode;

// identifier of each fusion pattern so the op encoders know which kernel to use
തരനിർവചനം എണ്ണൽ ggml_metal_fusion_id {
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

ഘടന ggml_metal_fusion; // defined in ggml-metal-fusion.cpp

തരനിർവചനം ഘടന ggml_metal_fusion ggml_metal_fusion;

// access the fusion identifier without exposing the full pattern definition
ggml_metal_fusion_id ggml_metal_fusion_get_id(സ്ഥിരം ഘടന ggml_metal_fusion * fusion);

// apply any alloc-dependencies required by the fused kernels during graph optimize
ശൂന്യം ggml_metal_fusion_add_alloc_deps(
        ശൂന്യം * user_data,
        ശൂന്യം (*add_alloc_dep)(ശൂന്യം *, ഘടന ggml_tensor *, ഘടന ggml_tensor *),
        ഘടന ggml_cgraph * gf);

// ---- shared fusion info ---------------------------------------------------

// shared fusion debugging context, owned by the device; newly created backend contexts for that
// device register with it so the fusion counters are race-free and accumulate across contexts.
ഘടന ggml_metal_fusion_info; // defined in ggml-metal-fusion.cpp

ഘടന ggml_metal_fusion_info * ggml_metal_fusion_info_init(bool enabled, പൂർണ്ണസംഖ്യ debug);
ശൂന്യം ggml_metal_fusion_info_free(ഘടന ggml_metal_fusion_info * finfo);

bool ggml_metal_fusion_info_enabled(സ്ഥിരം ഘടന ggml_metal_fusion_info * finfo);
bool ggml_metal_fusion_info_stats  (സ്ഥിരം ഘടന ggml_metal_fusion_info * finfo);
പൂർണ്ണസംഖ്യ  ggml_metal_fusion_info_debug  (സ്ഥിരം ഘടന ggml_metal_fusion_info * finfo);

പൂർണ്ണസംഖ്യ          ggml_metal_fusion_info_n_fusions(സ്ഥിരം ഘടന ggml_metal_fusion_info * finfo);
സ്ഥിരം അക്ഷരം * ggml_metal_fusion_info_label    (സ്ഥിരം ഘടന ggml_metal_fusion_info * finfo, പൂർണ്ണസംഖ്യ idx);
uint64_t     ggml_metal_fusion_info_എണ്ണം    (സ്ഥിരം ഘടന ggml_metal_fusion_info * finfo, പൂർണ്ണസംഖ്യ idx);

ശൂന്യം ggml_metal_fusion_info_എണ്ണം_fusion(ഘടന ggml_metal_fusion_info * finfo, സ്ഥിരം ഘടന ggml_metal_fusion * fusion);
ശൂന്യം ggml_metal_fusion_info_set_enabled (ഘടന ggml_metal_fusion_info * finfo, bool enabled);

ശൂന്യം ggml_metal_fusion_info_stats_init (      ഘടന ggml_metal_fusion_info * finfo);
ശൂന്യം ggml_metal_fusion_info_stats_reset(      ഘടന ggml_metal_fusion_info * finfo);
പൂർണ്ണസംഖ്യ  ggml_metal_fusion_info_stats_get  (സ്ഥിരം ഘടന ggml_metal_fusion_info * finfo, സ്ഥിരം അക്ഷരം ** labels, uint64_t * counts, പൂർണ്ണസംഖ്യ n);
ശൂന്യം ggml_metal_fusion_info_labels_init(      ഘടന ggml_metal_fusion_info * finfo);

// compute phase: longest fusion starting at idx (a position in node_idxs) that matches in `mode` on a device with
// props. returns the matching pattern (nullptr if no fusion) and sets *n_out to the number of nodes consumed.
സ്ഥിരം ggml_metal_fusion * ggml_metal_fusion_next(
        സ്ഥിരം ഘടന ggml_cgraph * gf,
        സ്ഥിരം പൂർണ്ണസംഖ്യ * കണ്ണി_idxs,
        പൂർണ്ണസംഖ്യ n_idxs,
        പൂർണ്ണസംഖ്യ idx,
        ggml_metal_fusion_mode mode,
        പൂർണ്ണസംഖ്യ * n_out);

// optimize phase: maximum number of nodes starting at idx (a raw sequential graph index) that
// could be fused on a device with props, chaining patterns back-to-back. returns at least 1.
പൂർണ്ണസംഖ്യ ggml_metal_fusion_max(സ്ഥിരം ഘടന ggml_cgraph * gf, പൂർണ്ണസംഖ്യ idx);

#ifdef __cplusplus
}
#endif
