#pragma once

#ضمن "ggml-metal-device.h"

#ifdef __cplusplus
خارجي "C" {
#endif

//
// backend context
//

تعريفنوع بنية ggml_metal * ggml_metal_t;

ggml_metal_t ggml_metal_init(ggml_metal_device_t dev);
فراغ ggml_metal_free(ggml_metal_t ctx);

ثابت حرف * ggml_metal_get_name(ggml_metal_t ctx);

فراغ ggml_metal_synchronize(ggml_metal_t ctx);

فراغ ggml_metal_set_tensor_async(ggml_metal_t ctx, بنية ggml_tensor * tensor, ثابت فراغ * data, size_t offset, size_t size);
فراغ ggml_metal_get_tensor_async(ggml_metal_t ctx, ثابت بنية ggml_tensor * tensor, فراغ * data, size_t offset, size_t size);
bool ggml_metal_cpy_tensor_async(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, ثابت بنية ggml_tensor * src, بنية ggml_tensor * dst);

تعداد ggml_status ggml_metal_graph_compute (ggml_metal_t ctx, بنية ggml_cgraph * gf);
فراغ             ggml_metal_graph_optimize(ggml_metal_t ctx, بنية ggml_cgraph * gf);

فراغ ggml_metal_event_record(ggml_metal_t ctx, ggml_metal_event_t ev);
فراغ ggml_metal_event_wait  (ggml_metal_t ctx, ggml_metal_event_t ev);

ggml_metal_event_t ggml_metal_get_ev_cpy(ggml_metal_t ctx);

فراغ ggml_metal_set_n_cb            (ggml_metal_t ctx, صحيح n_cb);
فراغ ggml_metal_set_abort_callback  (ggml_metal_t ctx, ggml_abort_callback abort_callback, فراغ * user_data);

bool ggml_metal_supports_family     (ggml_metal_t ctx, صحيح family);
فراغ ggml_metal_capture_next_compute(ggml_metal_t ctx);

#ifdef __cplusplus
}
#endif
