#आशय once

#समावेश "ggml-metal-device.h"

#अगर_घोषित __cplusplus
बाह्य "C" {
#पूर्ण_अगर

//
// backend context
//

प्रकार काष्ठा ggml_metal * ggml_metal_t;

ggml_metal_t ggml_metal_init(ggml_metal_device_t dev);
व्योम ggml_metal_free(ggml_metal_t ctx);

स्थिर अक्षर * ggml_metal_get_name(ggml_metal_t ctx);

व्योम ggml_metal_synchronize(ggml_metal_t ctx);

व्योम ggml_metal_set_tensor_async(ggml_metal_t ctx, काष्ठा ggml_tensor * tensor, स्थिर व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार आकार);
व्योम ggml_metal_get_tensor_async(ggml_metal_t ctx, स्थिर काष्ठा ggml_tensor * tensor, व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार आकार);
bool ggml_metal_cpy_tensor_async(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, स्थिर काष्ठा ggml_tensor * src, काष्ठा ggml_tensor * dst);

क्रमागत ggml_status ggml_metal_graph_compute (ggml_metal_t ctx, काष्ठा ggml_cgraph * gf);
व्योम             ggml_metal_graph_optimize(ggml_metal_t ctx, काष्ठा ggml_cgraph * gf);

व्योम ggml_metal_event_record(ggml_metal_t ctx, ggml_metal_event_t ev);
व्योम ggml_metal_event_wait  (ggml_metal_t ctx, ggml_metal_event_t ev);

ggml_metal_event_t ggml_metal_get_ev_cpy(ggml_metal_t ctx);

व्योम ggml_metal_set_n_cb            (ggml_metal_t ctx, पूर्णांक n_cb);
व्योम ggml_metal_set_abort_callback  (ggml_metal_t ctx, ggml_abort_callback abort_callback, व्योम * user_तथ्य);

bool ggml_metal_supports_family     (ggml_metal_t ctx, पूर्णांक family);
व्योम ggml_metal_capture_अग्र_compute(ggml_metal_t ctx);

#अगर_घोषित __cplusplus
}
#पूर्ण_अगर
