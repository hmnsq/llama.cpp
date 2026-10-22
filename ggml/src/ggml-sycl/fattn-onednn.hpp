#यदि_अपरिभाषितम् GGML_SYCL_FATTN_ONEDNN_HPP
#परिभाषय GGML_SYCL_FATTN_ONEDNN_HPP

#अन्तर्भाव "common.hpp"

// Static-only check: fused-XMX oneDNN Graph SDPA path==flash-attn op
// (f16 KV, no softcap/ALiBi, single stream, tuned head_dim, prefill-sized q.)
तार्किक ggml_sycl_flash_attn_ext_onednn_supported(ध्रुव ggml_tensor * dst, तार्किक use_shape_सीमा = सत्य);

// True when the oneDNN path binds an F16 KV cache in place instead of staging a dense copy of
// it. Depends only on the types and strides of K and V, so the answer holds for every call.
तार्किक ggml_sycl_fattn_onednn_binds_kv(ध्रुव ggml_tensor * K, ध्रुव ggml_tensor * V);

// Run flash attention through oneDNN's fused xmx SDPA
// execute the cached SDPA partition, write the f32 dst. Falls back to the TILE kernel on any failure.
शून्य ggml_sycl_flash_attn_ext_onednn(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#यदि_अन्तः // GGML_SYCL_FATTN_ONEDNN_HPP
