#ifndef GGML_SYCL_TOPK_MOE_HPP
#define GGML_SYCL_TOPK_MOE_HPP

#অন্তর্ভাব "common.hpp"

// Detect a fusable op subgraph starting at cgraph node `i` and, if found, dispatch the fused
// kernel. Returns the number of *following* nodes consumed (0 = no fusion applies at i).
পূর্ণাঙ্ক ggml_sycl_fuse(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, পূর্ণাঙ্ক i);

পূর্ণাঙ্ক ggml_sycl_fuse_topk_moe(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, পূর্ণাঙ্ক i);

#endif  // GGML_SYCL_TOPK_MOE_HPP
