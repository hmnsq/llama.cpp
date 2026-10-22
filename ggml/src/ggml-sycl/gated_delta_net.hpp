#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ <sycl/sycl.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ "dpct/helper.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "common.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
ਬਣਤਰ ggml_sycl_gated_delta_net_fused_cache {
    ਦਸ਼ਮਲਵ * ਅੰਕੜੇ;        // rollback slot 0
    int64_t ਥਾਂ_stride; // between rollback slots (0 when K==1)
};

ਖਾਲੀ ggml_sycl_op_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
ਖਾਲੀ ggml_sycl_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_sycl_try_gdn_cache_fusion)
ਖਾਲੀ ggml_sycl_op_gated_delta_net_fused_cache(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              ggml_sycl_gated_delta_net_fused_cache cache);
