#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_TOPK_MOE_HPP
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SYCL_TOPK_MOE_HPP

#ਅਨ੍ਤਰ੍ਭਾਵ "common.hpp"

// Detect a fusable op subgraph starting at cgraph node `i` and, if found, dispatch the fused
// kernel. Returns the number of *following* nodes consumed (0 = no fusion applies at i).
ਪੂਰਨ ggml_sycl_fuse(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, ਪੂਰਨ i);

ਪੂਰਨ ggml_sycl_fuse_topk_moe(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, ਪੂਰਨ i);

#ਜੇ_ਅੰਤ  // GGML_SYCL_TOPK_MOE_HPP
