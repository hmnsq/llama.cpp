#pragma once

#অন্তর্ভাব <sycl/sycl.hpp>
#অন্তর্ভাব "dpct/helper.hpp"
#অন্তর্ভাব "common.hpp"
#অন্তর্ভাব "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
সংরচনা ggml_sycl_gated_delta_net_fused_cache {
    ভিন্ন * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

শূন্য ggml_sycl_op_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
শূন্য ggml_sycl_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_sycl_try_gdn_cache_fusion)
শূন্য ggml_sycl_op_gated_delta_net_fused_cache(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              ggml_sycl_gated_delta_net_fused_cache cache);
