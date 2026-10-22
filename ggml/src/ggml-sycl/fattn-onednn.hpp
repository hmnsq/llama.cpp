#ifndef GGML_SYCL_FATTN_ONEDNN_HPP
#define GGML_SYCL_FATTN_ONEDNN_HPP

#અન્તર્ભાવ "common.hpp"

// Static-only check: fused-XMX oneDNN Graph SDPA path==flash-attn op
// (f16 KV, no softcap/ALiBi, single stream, tuned head_dim, prefill-sized q.)
તાર્કિક ggml_sycl_flash_attn_ext_onednn_supported(અચલ ggml_tensor * dst, તાર્કિક use_shape_limit = સાચું);

// True when the oneDNN path binds an F16 KV cache in place instead of staging a dense copy of
// it. Depends only on the types and strides of K and V, so the answer holds for every call.
તાર્કિક ggml_sycl_fattn_onednn_binds_kv(અચલ ggml_tensor * K, અચલ ggml_tensor * V);

// Run flash attention through oneDNN's fused xmx SDPA
// execute the cached SDPA partition, write the f32 dst. Falls back to the TILE kernel on any failure.
શૂન્ય ggml_sycl_flash_attn_ext_onednn(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_FATTN_ONEDNN_HPP
