#నిర్వచితం_కాకపోతే GGML_SYCL_FATTN_ONEDNN_HPP
#నిర్వచించు GGML_SYCL_FATTN_ONEDNN_HPP

#అన్తర్భావ "common.hpp"

// Static-only check: fused-XMX oneDNN Graph SDPA path==flash-attn op
// (f16 KV, no softcap/ALiBi, single stream, tuned head_dim, prefill-sized q.)
తార్కిక ggml_sycl_flash_attn_ext_onednn_supported(స్థిరరాశి ggml_tensor * dst, తార్కిక use_shape_హద్దు = నిజం);

// True when the oneDNN path binds an F16 KV cache in place instead of staging a dense copy of
// it. Depends only on the types and strides of K and V, so the answer holds for every call.
తార్కిక ggml_sycl_fattn_onednn_binds_kv(స్థిరరాశి ggml_tensor * K, స్థిరరాశి ggml_tensor * V);

// Run flash attention through oneDNN's fused xmx SDPA
// execute the cached SDPA partition, write the f32 dst. Falls back to the TILE kernel on any failure.
శూన్య ggml_sycl_flash_attn_ext_onednn(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#అయితే_ముగింపు // GGML_SYCL_FATTN_ONEDNN_HPP
