#pragma once

#شامل "ggml-metal-device.h"

#ifdef __cplusplus
بیرونی "C" {
#endif

//
// backend context
//

تعریفنوع ساختار ggml_metal * ggml_metal_t;

ggml_metal_t ggml_metal_init(ggml_metal_device_t dev);
تهی ggml_metal_free(ggml_metal_t ctx);

ثابت نویسه * ggml_metal_get_name(ggml_metal_t ctx);

تهی ggml_metal_synchronize(ggml_metal_t ctx);

تهی ggml_metal_set_tensor_async(ggml_metal_t ctx, ساختار ggml_tensor * tensor, ثابت تهی * data, size_t offset, size_t size);
تهی ggml_metal_get_tensor_async(ggml_metal_t ctx, ثابت ساختار ggml_tensor * tensor, تهی * data, size_t offset, size_t size);
bool ggml_metal_cpy_tensor_async(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, ثابت ساختار ggml_tensor * src, ساختار ggml_tensor * dst);

شمارش ggml_status ggml_metal_graph_compute (ggml_metal_t ctx, ساختار ggml_cgraph * gf);
تهی             ggml_metal_graph_optimize(ggml_metal_t ctx, ساختار ggml_cgraph * gf);

تهی ggml_metal_event_record(ggml_metal_t ctx, ggml_metal_event_t ev);
تهی ggml_metal_event_wait  (ggml_metal_t ctx, ggml_metal_event_t ev);

ggml_metal_event_t ggml_metal_get_ev_cpy(ggml_metal_t ctx);

تهی ggml_metal_set_n_cb            (ggml_metal_t ctx, صحیح n_cb);
تهی ggml_metal_set_abort_callback  (ggml_metal_t ctx, ggml_abort_callback abort_callback, تهی * user_data);

bool ggml_metal_supports_family     (ggml_metal_t ctx, صحیح family);
تهی ggml_metal_capture_next_compute(ggml_metal_t ctx);

#ifdef __cplusplus
}
#endif
