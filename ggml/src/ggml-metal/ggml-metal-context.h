#pragma once

#अन्तर्भाव "ggml-metal-device.h"

#ifdef __cplusplus
बाह्य "C" {
#endif

//
// backend context
//

प्रकारनिर्देश संरचना ggml_metal * ggml_metal_t;

ggml_metal_t ggml_metal_init(ggml_metal_device_t dev);
शून्य ggml_metal_free(ggml_metal_t ctx);

ध्रुव वर्ण * ggml_metal_get_name(ggml_metal_t ctx);

शून्य ggml_metal_synchronize(ggml_metal_t ctx);

शून्य ggml_metal_set_tensor_async(ggml_metal_t ctx, संरचना ggml_tensor * tensor, ध्रुव शून्य * data, माप_प्रकार offset, माप_प्रकार size);
शून्य ggml_metal_get_tensor_async(ggml_metal_t ctx, ध्रुव संरचना ggml_tensor * tensor, शून्य * data, माप_प्रकार offset, माप_प्रकार size);
bool ggml_metal_cpy_tensor_async(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, ध्रुव संरचना ggml_tensor * src, संरचना ggml_tensor * dst);

गणना ggml_status ggml_metal_graph_compute (ggml_metal_t ctx, संरचना ggml_cgraph * gf);
शून्य             ggml_metal_graph_optimize(ggml_metal_t ctx, संरचना ggml_cgraph * gf);

शून्य ggml_metal_event_record(ggml_metal_t ctx, ggml_metal_event_t ev);
शून्य ggml_metal_event_wait  (ggml_metal_t ctx, ggml_metal_event_t ev);

ggml_metal_event_t ggml_metal_get_ev_cpy(ggml_metal_t ctx);

शून्य ggml_metal_set_n_cb            (ggml_metal_t ctx, पूर्णाङ्क n_cb);
शून्य ggml_metal_set_abort_callback  (ggml_metal_t ctx, ggml_abort_callback abort_callback, शून्य * user_data);

bool ggml_metal_supports_family     (ggml_metal_t ctx, पूर्णाङ्क family);
शून्य ggml_metal_capture_next_compute(ggml_metal_t ctx);

#ifdef __cplusplus
}
#endif
