#अगर_अघोषित GGML_SYCL_FATTN_ONEDNN_HPP
#घोषणा GGML_SYCL_FATTN_ONEDNN_HPP

#समावेश "common.hpp"

// Static-only check: fused-XMX oneDNN Graph SDPA path==flash-attn op
// (f16 KV, no softcap/ALiBi, single stream, tuned head_dim, prefill-sized q.)
bool ggml_sycl_flash_attn_ext_onednn_supported(स्थिर ggml_tensor * dst, bool use_shape_limit = true);

// True when the oneDNN path binds an F16 KV cache in place instead of staging a dense copy of
// it. Depends only on the types and strides of K and V, so the answer holds for every call.
bool ggml_sycl_fattn_onednn_binds_kv(स्थिर ggml_tensor * K, स्थिर ggml_tensor * V);

// Run flash attention through oneDNN's fused xmx SDPA
// execute the cached SDPA partition, write the f32 dst. Falls back to the TILE kernel on any failure.
व्योम ggml_sycl_flash_attn_ext_onednn(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#पूर्ण_अगर // GGML_SYCL_FATTN_ONEDNN_HPP
