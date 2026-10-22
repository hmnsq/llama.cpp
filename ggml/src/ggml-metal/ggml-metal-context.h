#निर्देशः once

#अन्तर्भाव "ggml-metal-device.h"

#यदि_परिभाषितम् __cplusplus
बाह्य "C" {
#यदि_अन्तः

//
// backend context
//

प्रकारनिर्देश संरचना ggml_metal * ggml_metal_t;

ggml_metal_t ggml_metal_init(ggml_metal_device_t dev);
शून्य ggml_metal_free(ggml_metal_t ctx);

ध्रुव वर्ण * ggml_metal_get_name(ggml_metal_t ctx);

शून्य ggml_metal_synchronize(ggml_metal_t ctx);

शून्य ggml_metal_समुच्चयः_tensor_async(ggml_metal_t ctx, संरचना ggml_tensor * tensor, ध्रुव शून्य * दत्त, size_t विस्थापनम्, size_t आकार);
शून्य ggml_metal_get_tensor_async(ggml_metal_t ctx, ध्रुव संरचना ggml_tensor * tensor, शून्य * दत्त, size_t विस्थापनम्, size_t आकार);
bool ggml_metal_cpy_tensor_async(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, ध्रुव संरचना ggml_tensor * src, संरचना ggml_tensor * dst);

गणना ggml_स्थितिः ggml_metal_graph_compute (ggml_metal_t ctx, संरचना ggml_cgraph * gf);
शून्य             ggml_metal_graph_optimize(ggml_metal_t ctx, संरचना ggml_cgraph * gf);

शून्य ggml_metal_event_लेखः(ggml_metal_t ctx, ggml_metal_event_t ev);
शून्य ggml_metal_event_wait  (ggml_metal_t ctx, ggml_metal_event_t ev);

ggml_metal_event_t ggml_metal_get_ev_cpy(ggml_metal_t ctx);

शून्य ggml_metal_समुच्चयः_n_cb            (ggml_metal_t ctx, पूर्णाङ्क n_cb);
शून्य ggml_metal_समुच्चयः_abort_callback  (ggml_metal_t ctx, ggml_abort_callback abort_callback, शून्य * user_दत्त);

bool ggml_metal_supports_family     (ggml_metal_t ctx, पूर्णाङ्क family);
शून्य ggml_metal_capture_अग्र_compute(ggml_metal_t ctx);

#यदि_परिभाषितम् __cplusplus
}
#यदि_अन्तः
