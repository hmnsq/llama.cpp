#ifndef GGML_SYCL_TOPK_MOE_HPP
#define GGML_SYCL_TOPK_MOE_HPP

#અન્તર્ભાવ "common.hpp"

// Detect a fusable op subgraph starting at cgraph node `i` and, if found, dispatch the fused
// kernel. Returns the number of *following* nodes consumed (0 = no fusion applies at i).
પૂર્ણાંક ggml_sycl_fuse(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, પૂર્ણાંક i);

પૂર્ણાંક ggml_sycl_fuse_topk_moe(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, પૂર્ણાંક i);

#endif  // GGML_SYCL_TOPK_MOE_HPP
