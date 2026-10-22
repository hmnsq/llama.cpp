#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_SYCL_TOPK_MOE_HPP
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SYCL_TOPK_MOE_HPP

#ಅನ್ತರ್ಭಾವ "common.hpp"

// Detect a fusable op subgraph starting at cgraph node `i` and, if found, dispatch the fused
// kernel. Returns the number of *following* nodes consumed (0 = no fusion applies at i).
ಪೂರ್ಣಾಂಕ ggml_sycl_fuse(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ i);

ಪೂರ್ಣಾಂಕ ggml_sycl_fuse_topk_moe(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ i);

#ಆದರೆ_ಅಂತ್ಯ  // GGML_SYCL_TOPK_MOE_HPP
