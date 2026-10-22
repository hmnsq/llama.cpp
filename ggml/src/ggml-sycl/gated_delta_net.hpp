#هدايت once

#ضمن <sycl/sycl.hpp>
#ضمن "dpct/helper.hpp"
#ضمن "common.hpp"
#ضمن "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
بنية ggml_sycl_gated_delta_net_fused_cache {
    عائم * معلومات;        // rollback slot 0
    int64_t جڳهه_stride; // between rollback slots (0 when K==1)
};

فراغ ggml_sycl_op_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
فراغ ggml_sycl_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_sycl_try_gdn_cache_fusion)
فراغ ggml_sycl_op_gated_delta_net_fused_cache(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              ggml_sycl_gated_delta_net_fused_cache cache);
