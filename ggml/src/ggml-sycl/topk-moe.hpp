#جيڪڏهن_وصف_نه_ٿيل GGML_SYCL_TOPK_MOE_HPP
#وصف_ڪريو GGML_SYCL_TOPK_MOE_HPP

#ضمن "common.hpp"

// Detect a fusable op subgraph starting at cgraph node `i` and, if found, dispatch the fused
// kernel. Returns the number of *following* nodes consumed (0 = no fusion applies at i).
صحيح ggml_sycl_fuse(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, صحيح i);

صحيح ggml_sycl_fuse_topk_moe(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, صحيح i);

#جيڪڏهن_پورو  // GGML_SYCL_TOPK_MOE_HPP
