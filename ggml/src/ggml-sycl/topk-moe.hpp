#اگر_غیر_تعریف_شدہ GGML_SYCL_TOPK_MOE_HPP
#تعریف_کرو GGML_SYCL_TOPK_MOE_HPP

#شامل "common.hpp"

// Detect a fusable op subgraph starting at cgraph node `i` and, if found, dispatch the fused
// kernel. Returns the number of *following* nodes consumed (0 = no fusion applies at i).
صحیح ggml_sycl_fuse(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, صحیح i);

صحیح ggml_sycl_fuse_topk_moe(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, صحیح i);

#اگر_ختم  // GGML_SYCL_TOPK_MOE_HPP
