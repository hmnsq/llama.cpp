#که_نه_تعریف_شوی GGML_SYCL_FATTN_ONEDNN_HPP
#تعریف_کړه GGML_SYCL_FATTN_ONEDNN_HPP

#ضمن "common.hpp"

// Static-only check: fused-XMX oneDNN Graph SDPA path==flash-attn op
// (f16 KV, no softcap/ALiBi, single stream, tuned head_dim, prefill-sized q.)
منطقي ggml_sycl_flash_attn_ext_onednn_supported(ثابت ggml_tensor * dst, منطقي use_shape_حد = ریښتیا);

// True when the oneDNN path binds an F16 KV cache in place instead of staging a dense copy of
// it. Depends only on the types and strides of K and V, so the answer holds for every call.
منطقي ggml_sycl_fattn_onednn_binds_kv(ثابت ggml_tensor * K, ثابت ggml_tensor * V);

// Run flash attention through oneDNN's fused xmx SDPA
// execute the cached SDPA partition, write the f32 dst. Falls back to the TILE kernel on any failure.
فراغ ggml_sycl_flash_attn_ext_onednn(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#د_که_پای // GGML_SYCL_FATTN_ONEDNN_HPP
