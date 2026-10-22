#यदि_अपरिभाषितम् GGML_SYCL_TOPK_MOE_HPP
#परिभाषय GGML_SYCL_TOPK_MOE_HPP

#अन्तर्भाव "common.hpp"

// Detect a fusable op subgraph starting at cgraph node `i` and, if found, dispatch the fused
// kernel. Returns the number of *following* nodes consumed (0 = no fusion applies at i).
पूर्णाङ्क ggml_sycl_fuse(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, पूर्णाङ्क i);

पूर्णाङ्क ggml_sycl_fuse_topk_moe(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, पूर्णाङ्क i);

#यदि_अन्तः  // GGML_SYCL_TOPK_MOE_HPP
