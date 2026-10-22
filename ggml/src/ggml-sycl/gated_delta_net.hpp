#pragma once

#अन्तर्भाव <sycl/sycl.hpp>
#अन्तर्भाव "dpct/helper.hpp"
#अन्तर्भाव "common.hpp"
#अन्तर्भाव "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
संरचना ggml_sycl_gated_delta_net_fused_cache {
    दशांश * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

शून्य ggml_sycl_op_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
शून्य ggml_sycl_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_sycl_try_gdn_cache_fusion)
शून्य ggml_sycl_op_gated_delta_net_fused_cache(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              ggml_sycl_gated_delta_net_fused_cache cache);
