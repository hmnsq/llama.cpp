#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_SYCL_FATTN_ONEDNN_HPP
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SYCL_FATTN_ONEDNN_HPP

#ಅನ್ತರ್ಭಾವ "common.hpp"

// Static-only check: fused-XMX oneDNN Graph SDPA path==flash-attn op
// (f16 KV, no softcap/ALiBi, single stream, tuned head_dim, prefill-sized q.)
ತಾರ್ಕಿಕ ggml_sycl_flash_attn_ext_onednn_supported(ಸ್ಥಿರಾಂಕ ggml_tensor * dst, ತಾರ್ಕಿಕ use_shape_ಎಲ್ಲೆ = ನಿಜ);

// True when the oneDNN path binds an F16 KV cache in place instead of staging a dense copy of
// it. Depends only on the types and strides of K and V, so the answer holds for every call.
ತಾರ್ಕಿಕ ggml_sycl_fattn_onednn_binds_kv(ಸ್ಥಿರಾಂಕ ggml_tensor * K, ಸ್ಥಿರಾಂಕ ggml_tensor * V);

// Run flash attention through oneDNN's fused xmx SDPA
// execute the cached SDPA partition, write the f32 dst. Falls back to the TILE kernel on any failure.
ಶೂನ್ಯ ggml_sycl_flash_attn_ext_onednn(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_FATTN_ONEDNN_HPP
