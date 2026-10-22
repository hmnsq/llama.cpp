#جے_بنا_تعریف GGML_SYCL_TOPK_MOE_HPP
#تعریف_کرو GGML_SYCL_TOPK_MOE_HPP

#ضمن "common.hpp"

// Detect a fusable op subgraph starting at cgraph node `i` and, if found, dispatch the fused
// kernel. Returns the number of *following* nodes consumed (0 = no fusion applies at i).
صحيح ggml_sycl_fuse(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, صحيح i);

صحيح ggml_sycl_fuse_topk_moe(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, صحيح i);

#جے_انت  // GGML_SYCL_TOPK_MOE_HPP
