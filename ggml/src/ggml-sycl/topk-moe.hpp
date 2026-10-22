#ifndef GGML_SYCL_TOPK_MOE_HPP
#define GGML_SYCL_TOPK_MOE_HPP

#include "common.hpp"

// Detect a fusable op subgraph starting at cgraph node `i` and, if found, dispatch the fused
// kernel. Returns the number of *following* nodes consumed (0 = no fusion applies at i).
ကိန်းပြည့် ggml_sycl_fuse(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, ကိန်းပြည့် i);

ကိန်းပြည့် ggml_sycl_fuse_topk_moe(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, ကိန်းပြည့် i);

#endif  // GGML_SYCL_TOPK_MOE_HPP
