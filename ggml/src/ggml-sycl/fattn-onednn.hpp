#ifndef GGML_SYCL_FATTN_ONEDNN_HPP
#define GGML_SYCL_FATTN_ONEDNN_HPP

#অন্তর্ভাব "common.hpp"

// Static-only check: fused-XMX oneDNN Graph SDPA path==flash-attn op
// (f16 KV, no softcap/ALiBi, single stream, tuned head_dim, prefill-sized q.)
তার্কিক ggml_sycl_flash_attn_ext_onednn_supported(ধ্রুব ggml_tensor * dst, তার্কিক use_shape_limit = সত্য);

// True when the oneDNN path binds an F16 KV cache in place instead of staging a dense copy of
// it. Depends only on the types and strides of K and V, so the answer holds for every call.
তার্কিক ggml_sycl_fattn_onednn_binds_kv(ধ্রুব ggml_tensor * K, ধ্রুব ggml_tensor * V);

// Run flash attention through oneDNN's fused xmx SDPA
// execute the cached SDPA partition, write the f32 dst. Falls back to the TILE kernel on any failure.
শূন্য ggml_sycl_flash_attn_ext_onednn(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_FATTN_ONEDNN_HPP
