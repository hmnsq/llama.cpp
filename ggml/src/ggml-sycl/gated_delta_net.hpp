#సూచన once

#అన్తర్భావ <sycl/sycl.hpp>
#అన్తర్భావ "dpct/helper.hpp"
#అన్తర్భావ "common.hpp"
#అన్తర్భావ "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
నిర్మాణం ggml_sycl_gated_delta_net_fused_cache {
    దశాంశం * దత్తాంశం;        // rollback slot 0
    int64_t స్థానం_stride; // between rollback slots (0 when K==1)
};

శూన్య ggml_sycl_op_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
శూన్య ggml_sycl_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_sycl_try_gdn_cache_fusion)
శూన్య ggml_sycl_op_gated_delta_net_fused_cache(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              ggml_sycl_gated_delta_net_fused_cache cache);
