#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_FATTN_ONEDNN_HPP
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SYCL_FATTN_ONEDNN_HPP

#ਅਨ੍ਤਰ੍ਭਾਵ "common.hpp"

// Static-only check: fused-XMX oneDNN Graph SDPA path==flash-attn op
// (f16 KV, no softcap/ALiBi, single stream, tuned head_dim, prefill-sized q.)
ਤਾਰਕਿਕ ggml_sycl_flash_attn_ext_onednn_supported(ਅਚਲ ggml_tensor * dst, ਤਾਰਕਿਕ use_shape_ਹੱਦਬੰਦੀ = ਸੱਚ);

// True when the oneDNN path binds an F16 KV cache in place instead of staging a dense copy of
// it. Depends only on the types and strides of K and V, so the answer holds for every call.
ਤਾਰਕਿਕ ggml_sycl_fattn_onednn_binds_kv(ਅਚਲ ggml_tensor * K, ਅਚਲ ggml_tensor * V);

// Run flash attention through oneDNN's fused xmx SDPA
// execute the cached SDPA partition, write the f32 dst. Falls back to the TILE kernel on any failure.
ਖਾਲੀ ggml_sycl_flash_attn_ext_onednn(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#ਜੇ_ਅੰਤ // GGML_SYCL_FATTN_ONEDNN_HPP
