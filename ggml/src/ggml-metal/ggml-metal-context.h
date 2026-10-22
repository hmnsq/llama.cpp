#pragma once

#அந்தர்பாவ "ggml-metal-device.h"

#ifdef __cplusplus
வெளி "C" {
#endif

//
// backend context
//

வகைவரை கட்டமைப்பு ggml_metal * ggml_metal_t;

ggml_metal_t ggml_metal_init(ggml_metal_device_t dev);
வெறுமை ggml_metal_free(ggml_metal_t ctx);

மாறிலி எழுத்து * ggml_metal_get_name(ggml_metal_t ctx);

வெறுமை ggml_metal_synchronize(ggml_metal_t ctx);

வெறுமை ggml_metal_set_tensor_async(ggml_metal_t ctx, கட்டமைப்பு ggml_tensor * tensor, மாறிலி வெறுமை * data, size_t offset, size_t size);
வெறுமை ggml_metal_get_tensor_async(ggml_metal_t ctx, மாறிலி கட்டமைப்பு ggml_tensor * tensor, வெறுமை * data, size_t offset, size_t size);
bool ggml_metal_cpy_tensor_async(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, மாறிலி கட்டமைப்பு ggml_tensor * src, கட்டமைப்பு ggml_tensor * dst);

எண்ணிக்கை ggml_status ggml_metal_graph_compute (ggml_metal_t ctx, கட்டமைப்பு ggml_cgraph * gf);
வெறுமை             ggml_metal_graph_optimize(ggml_metal_t ctx, கட்டமைப்பு ggml_cgraph * gf);

வெறுமை ggml_metal_event_record(ggml_metal_t ctx, ggml_metal_event_t ev);
வெறுமை ggml_metal_event_wait  (ggml_metal_t ctx, ggml_metal_event_t ev);

ggml_metal_event_t ggml_metal_get_ev_cpy(ggml_metal_t ctx);

வெறுமை ggml_metal_set_n_cb            (ggml_metal_t ctx, முழு n_cb);
வெறுமை ggml_metal_set_abort_callback  (ggml_metal_t ctx, ggml_abort_callback abort_callback, வெறுமை * user_data);

bool ggml_metal_supports_family     (ggml_metal_t ctx, முழு family);
வெறுமை ggml_metal_capture_next_compute(ggml_metal_t ctx);

#ifdef __cplusplus
}
#endif
