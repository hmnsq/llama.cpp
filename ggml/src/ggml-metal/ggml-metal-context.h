#pragma once

#ಅನ್ತರ್ಭಾವ "ggml-metal-device.h"

#ifdef __cplusplus
ಬಾಹ್ಯ "C" {
#endif

//
// backend context
//

ಪ್ರಕಾರನಿರ್ದೇಶ ಸಂರಚನಾ ggml_metal * ggml_metal_t;

ggml_metal_t ggml_metal_init(ggml_metal_device_t dev);
ಶೂನ್ಯ ggml_metal_free(ggml_metal_t ctx);

ಧ್ರುವ ವರ್ಣ * ggml_metal_get_name(ggml_metal_t ctx);

ಶೂನ್ಯ ggml_metal_synchronize(ggml_metal_t ctx);

ಶೂನ್ಯ ggml_metal_set_tensor_async(ggml_metal_t ctx, ಸಂರಚನಾ ggml_tensor * tensor, ಧ್ರುವ ಶೂನ್ಯ * data, size_t offset, size_t size);
ಶೂನ್ಯ ggml_metal_get_tensor_async(ggml_metal_t ctx, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor, ಶೂನ್ಯ * data, size_t offset, size_t size);
bool ggml_metal_cpy_tensor_async(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src, ಸಂರಚನಾ ggml_tensor * dst);

ಗಣನಾ ggml_status ggml_metal_graph_compute (ggml_metal_t ctx, ಸಂರಚನಾ ggml_cgraph * gf);
ಶೂನ್ಯ             ggml_metal_graph_optimize(ggml_metal_t ctx, ಸಂರಚನಾ ggml_cgraph * gf);

ಶೂನ್ಯ ggml_metal_event_record(ggml_metal_t ctx, ggml_metal_event_t ev);
ಶೂನ್ಯ ggml_metal_event_wait  (ggml_metal_t ctx, ggml_metal_event_t ev);

ggml_metal_event_t ggml_metal_get_ev_cpy(ggml_metal_t ctx);

ಶೂನ್ಯ ggml_metal_set_n_cb            (ggml_metal_t ctx, ಪೂರ್ಣಾಙ್ಕ n_cb);
ಶೂನ್ಯ ggml_metal_set_abort_callback  (ggml_metal_t ctx, ggml_abort_callback abort_callback, ಶೂನ್ಯ * user_data);

bool ggml_metal_supports_family     (ggml_metal_t ctx, ಪೂರ್ಣಾಙ್ಕ family);
ಶೂನ್ಯ ggml_metal_capture_next_compute(ggml_metal_t ctx);

#ifdef __cplusplus
}
#endif
