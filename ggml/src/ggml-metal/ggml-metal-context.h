#సూచన once

#అన్తర్భావ "ggml-metal-device.h"

#నిర్వచితమైతే __cplusplus
బాహ్య "C" {
#అయితే_ముగింపు

//
// backend context
//

రకనిర్వచనం నిర్మాణం ggml_metal * ggml_metal_t;

ggml_metal_t ggml_metal_init(ggml_metal_device_t dev);
శూన్య ggml_metal_free(ggml_metal_t ctx);

స్థిరరాశి అక్షరం * ggml_metal_get_name(ggml_metal_t ctx);

శూన్య ggml_metal_synchronize(ggml_metal_t ctx);

శూన్య ggml_metal_సమితి_tensor_async(ggml_metal_t ctx, నిర్మాణం ggml_tensor * tensor, స్థిరరాశి శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t కొలత);
శూన్య ggml_metal_get_tensor_async(ggml_metal_t ctx, స్థిరరాశి నిర్మాణం ggml_tensor * tensor, శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t కొలత);
bool ggml_metal_cpy_tensor_async(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, స్థిరరాశి నిర్మాణం ggml_tensor * src, నిర్మాణం ggml_tensor * dst);

గణన ggml_స్థితి ggml_metal_లేఖాచిత్రం_compute (ggml_metal_t ctx, నిర్మాణం ggml_cgraph * gf);
శూన్య             ggml_metal_లేఖాచిత్రం_optimize(ggml_metal_t ctx, నిర్మాణం ggml_cgraph * gf);

శూన్య ggml_metal_event_రికార్డు(ggml_metal_t ctx, ggml_metal_event_t ev);
శూన్య ggml_metal_event_wait  (ggml_metal_t ctx, ggml_metal_event_t ev);

ggml_metal_event_t ggml_metal_get_ev_cpy(ggml_metal_t ctx);

శూన్య ggml_metal_సమితి_n_cb            (ggml_metal_t ctx, పూర్ణాంకం n_cb);
శూన్య ggml_metal_సమితి_abort_callback  (ggml_metal_t ctx, ggml_abort_callback abort_callback, శూన్య * user_దత్తాంశం);

bool ggml_metal_supports_family     (ggml_metal_t ctx, పూర్ణాంకం family);
శూన్య ggml_metal_capture_తదుపరి_compute(ggml_metal_t ctx);

#నిర్వచితమైతే __cplusplus
}
#అయితే_ముగింపు
