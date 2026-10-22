#pragma once

#അന്തര്ഭാവ <sycl/sycl.hpp>
#അന്തര്ഭാവ "dpct/helper.hpp"
#അന്തര്ഭാവ "common.hpp"
#അന്തര്ഭാവ "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
ഘടന ggml_sycl_gated_delta_net_fused_cache {
    ദശാംശം * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

ശൂന്യം ggml_sycl_op_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
ശൂന്യം ggml_sycl_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_sycl_try_gdn_cache_fusion)
ശൂന്യം ggml_sycl_op_gated_delta_net_fused_cache(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              ggml_sycl_gated_delta_net_fused_cache cache);
