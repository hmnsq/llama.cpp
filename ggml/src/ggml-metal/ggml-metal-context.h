#pragma once

#অন্তর্ভাৱ "ggml-metal-device.h"

#ifdef __cplusplus
বাহ্য "C" {
#endif

//
// backend context
//

প্রকারনির্দেশ সংরচনা ggml_metal * ggml_metal_t;

ggml_metal_t ggml_metal_init(ggml_metal_device_t dev);
শূন্য ggml_metal_free(ggml_metal_t ctx);

ধ্রুৱ ৱর্ণ * ggml_metal_get_name(ggml_metal_t ctx);

শূন্য ggml_metal_synchronize(ggml_metal_t ctx);

শূন্য ggml_metal_set_tensor_async(ggml_metal_t ctx, সংরচনা ggml_tensor * tensor, ধ্রুৱ শূন্য * data, size_t offset, size_t size);
শূন্য ggml_metal_get_tensor_async(ggml_metal_t ctx, ধ্রুৱ সংরচনা ggml_tensor * tensor, শূন্য * data, size_t offset, size_t size);
bool ggml_metal_cpy_tensor_async(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, ধ্রুৱ সংরচনা ggml_tensor * src, সংরচনা ggml_tensor * dst);

গণনা ggml_status ggml_metal_graph_compute (ggml_metal_t ctx, সংরচনা ggml_cgraph * gf);
শূন্য             ggml_metal_graph_optimize(ggml_metal_t ctx, সংরচনা ggml_cgraph * gf);

শূন্য ggml_metal_event_record(ggml_metal_t ctx, ggml_metal_event_t ev);
শূন্য ggml_metal_event_wait  (ggml_metal_t ctx, ggml_metal_event_t ev);

ggml_metal_event_t ggml_metal_get_ev_cpy(ggml_metal_t ctx);

শূন্য ggml_metal_set_n_cb            (ggml_metal_t ctx, পূর্ণাঙ্ক n_cb);
শূন্য ggml_metal_set_abort_callback  (ggml_metal_t ctx, ggml_abort_callback abort_callback, শূন্য * user_data);

bool ggml_metal_supports_family     (ggml_metal_t ctx, পূর্ণাঙ্ক family);
শূন্য ggml_metal_capture_next_compute(ggml_metal_t ctx);

#ifdef __cplusplus
}
#endif
