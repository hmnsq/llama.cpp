#pragma once

#അന്തര്ഭാവ "ggml-metal-device.h"

#ifdef __cplusplus
ബാഹ്യ "C" {
#endif

//
// backend context
//

തരനിർവചനം ഘടന ggml_metal * ggml_metal_t;

ggml_metal_t ggml_metal_init(ggml_metal_device_t dev);
ശൂന്യം ggml_metal_free(ggml_metal_t ctx);

സ്ഥിരം അക്ഷരം * ggml_metal_get_name(ggml_metal_t ctx);

ശൂന്യം ggml_metal_synchronize(ggml_metal_t ctx);

ശൂന്യം ggml_metal_set_tensor_async(ggml_metal_t ctx, ഘടന ggml_tensor * tensor, സ്ഥിരം ശൂന്യം * data, size_t offset, size_t size);
ശൂന്യം ggml_metal_get_tensor_async(ggml_metal_t ctx, സ്ഥിരം ഘടന ggml_tensor * tensor, ശൂന്യം * data, size_t offset, size_t size);
bool ggml_metal_cpy_tensor_async(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, സ്ഥിരം ഘടന ggml_tensor * src, ഘടന ggml_tensor * dst);

എണ്ണൽ ggml_status ggml_metal_graph_compute (ggml_metal_t ctx, ഘടന ggml_cgraph * gf);
ശൂന്യം             ggml_metal_graph_optimize(ggml_metal_t ctx, ഘടന ggml_cgraph * gf);

ശൂന്യം ggml_metal_event_record(ggml_metal_t ctx, ggml_metal_event_t ev);
ശൂന്യം ggml_metal_event_wait  (ggml_metal_t ctx, ggml_metal_event_t ev);

ggml_metal_event_t ggml_metal_get_ev_cpy(ggml_metal_t ctx);

ശൂന്യം ggml_metal_set_n_cb            (ggml_metal_t ctx, പൂർണ്ണസംഖ്യ n_cb);
ശൂന്യം ggml_metal_set_abort_callback  (ggml_metal_t ctx, ggml_abort_callback abort_callback, ശൂന്യം * user_data);

bool ggml_metal_supports_family     (ggml_metal_t ctx, പൂർണ്ണസംഖ്യ family);
ശൂന്യം ggml_metal_capture_next_compute(ggml_metal_t ctx);

#ifdef __cplusplus
}
#endif
