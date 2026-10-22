#अगर_अघोषित GGML_SYCL_TOPK_MOE_HPP
#घोषणा GGML_SYCL_TOPK_MOE_HPP

#समावेश "common.hpp"

// Detect a fusable op subgraph starting at cgraph node `i` and, if found, dispatch the fused
// kernel. Returns the number of *following* nodes consumed (0 = no fusion applies at i).
पूर्णांक ggml_sycl_fuse(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, पूर्णांक i);

पूर्णांक ggml_sycl_fuse_topk_moe(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, पूर्णांक i);

#पूर्ण_अगर  // GGML_SYCL_TOPK_MOE_HPP
