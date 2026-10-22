#ifndef GGML_SYCL_TOPK_MOE_HPP
#define GGML_SYCL_TOPK_MOE_HPP

#അന്തര്ഭാവ "common.hpp"

// Detect a fusable op subgraph starting at cgraph node `i` and, if found, dispatch the fused
// kernel. Returns the number of *following* nodes consumed (0 = no fusion applies at i).
പൂർണ്ണസംഖ്യ ggml_sycl_fuse(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, പൂർണ്ണസംഖ്യ i);

പൂർണ്ണസംഖ്യ ggml_sycl_fuse_topk_moe(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, പൂർണ്ണസംഖ്യ i);

#endif  // GGML_SYCL_TOPK_MOE_HPP
