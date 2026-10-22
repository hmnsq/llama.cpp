#ifndef GGML_SYCL_TOPK_MOE_HPP
#define GGML_SYCL_TOPK_MOE_HPP

#ಅನ್ತರ್ಭಾವ "common.hpp"

// Detect a fusable op subgraph starting at cgraph node `i` and, if found, dispatch the fused
// kernel. Returns the number of *following* nodes consumed (0 = no fusion applies at i).
ಪೂರ್ಣಾಙ್ಕ ggml_sycl_fuse(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, ಪೂರ್ಣಾಙ್ಕ i);

ಪೂರ್ಣಾಙ್ಕ ggml_sycl_fuse_topk_moe(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, ಪೂರ್ಣಾಙ್ಕ i);

#endif  // GGML_SYCL_TOPK_MOE_HPP
