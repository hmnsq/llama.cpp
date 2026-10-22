#pragma once

#અન્તર્ભાવ "ggml-metal-device.h"

#ifdef __cplusplus
બાહ્ય "C" {
#endif

//
// backend context
//

પ્રકારવ્યાખ્યા સંરચના ggml_metal * ggml_metal_t;

ggml_metal_t ggml_metal_init(ggml_metal_device_t dev);
શૂન્ય ggml_metal_free(ggml_metal_t ctx);

અચલ અક્ષર * ggml_metal_get_name(ggml_metal_t ctx);

શૂન્ય ggml_metal_synchronize(ggml_metal_t ctx);

શૂન્ય ggml_metal_set_tensor_async(ggml_metal_t ctx, સંરચના ggml_tensor * tensor, અચલ શૂન્ય * data, size_t offset, size_t size);
શૂન્ય ggml_metal_get_tensor_async(ggml_metal_t ctx, અચલ સંરચના ggml_tensor * tensor, શૂન્ય * data, size_t offset, size_t size);
bool ggml_metal_cpy_tensor_async(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, અચલ સંરચના ggml_tensor * src, સંરચના ggml_tensor * dst);

ગણના ggml_status ggml_metal_graph_compute (ggml_metal_t ctx, સંરચના ggml_cgraph * gf);
શૂન્ય             ggml_metal_graph_optimize(ggml_metal_t ctx, સંરચના ggml_cgraph * gf);

શૂન્ય ggml_metal_event_record(ggml_metal_t ctx, ggml_metal_event_t ev);
શૂન્ય ggml_metal_event_wait  (ggml_metal_t ctx, ggml_metal_event_t ev);

ggml_metal_event_t ggml_metal_get_ev_cpy(ggml_metal_t ctx);

શૂન્ય ggml_metal_set_n_cb            (ggml_metal_t ctx, પૂર્ણાંક n_cb);
શૂન્ય ggml_metal_set_abort_callback  (ggml_metal_t ctx, ggml_abort_callback abort_callback, શૂન્ય * user_data);

bool ggml_metal_supports_family     (ggml_metal_t ctx, પૂર્ણાંક family);
શૂન્ય ggml_metal_capture_next_compute(ggml_metal_t ctx);

#ifdef __cplusplus
}
#endif
