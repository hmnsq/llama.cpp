#నిర్వచితం_కాకపోతే GGML_SYCL_TOPK_MOE_HPP
#నిర్వచించు GGML_SYCL_TOPK_MOE_HPP

#అన్తర్భావ "common.hpp"

// Detect a fusable op subgraph starting at cgraph node `i` and, if found, dispatch the fused
// kernel. Returns the number of *following* nodes consumed (0 = no fusion applies at i).
పూర్ణాంకం ggml_sycl_fuse(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, పూర్ణాంకం i);

పూర్ణాంకం ggml_sycl_fuse_topk_moe(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, పూర్ణాంకం i);

#అయితే_ముగింపు  // GGML_SYCL_TOPK_MOE_HPP
