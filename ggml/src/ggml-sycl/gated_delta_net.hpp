#ہدایت once

#شامل <sycl/sycl.hpp>
#شامل "dpct/helper.hpp"
#شامل "common.hpp"
#شامل "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
ساخت ggml_sycl_gated_delta_net_fused_cache {
    اعشاری * ڈیٹا;        // rollback slot 0
    int64_t خانہ_stride; // between rollback slots (0 when K==1)
};

خالی ggml_sycl_op_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
خالی ggml_sycl_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_sycl_try_gdn_cache_fusion)
خالی ggml_sycl_op_gated_delta_net_fused_cache(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              ggml_sycl_gated_delta_net_fused_cache cache);
