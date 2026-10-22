#pragma once

#અન્તર્ભાવ <sycl/sycl.hpp>
#અન્તર્ભાવ "dpct/helper.hpp"
#અન્તર્ભાવ "common.hpp"
#અન્તર્ભાવ "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
સંરચના ggml_sycl_gated_delta_net_fused_cache {
    દશાંશ * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

શૂન્ય ggml_sycl_op_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
શૂન્ય ggml_sycl_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_sycl_try_gdn_cache_fusion)
શૂન્ય ggml_sycl_op_gated_delta_net_fused_cache(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              ggml_sycl_gated_delta_net_fused_cache cache);
