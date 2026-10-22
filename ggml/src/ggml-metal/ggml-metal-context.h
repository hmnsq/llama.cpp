#pragma once

#ଅନ୍ତର୍ଭାଵ "ggml-metal-device.h"

#ifdef __cplusplus
ବାହ୍ୟ "C" {
#endif

//
// backend context
//

ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା ggml_metal * ggml_metal_t;

ggml_metal_t ggml_metal_init(ggml_metal_device_t dev);
ଶୂନ୍ୟ ggml_metal_free(ggml_metal_t ctx);

ଅଚଳ ଅକ୍ଷର * ggml_metal_get_name(ggml_metal_t ctx);

ଶୂନ୍ୟ ggml_metal_synchronize(ggml_metal_t ctx);

ଶୂନ୍ୟ ggml_metal_set_tensor_async(ggml_metal_t ctx, ସଂରଚନା ggml_tensor * tensor, ଅଚଳ ଶୂନ୍ୟ * data, size_t offset, size_t size);
ଶୂନ୍ୟ ggml_metal_get_tensor_async(ggml_metal_t ctx, ଅଚଳ ସଂରଚନା ggml_tensor * tensor, ଶୂନ୍ୟ * data, size_t offset, size_t size);
bool ggml_metal_cpy_tensor_async(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, ଅଚଳ ସଂରଚନା ggml_tensor * src, ସଂରଚନା ggml_tensor * dst);

ଗଣନା ggml_status ggml_metal_graph_compute (ggml_metal_t ctx, ସଂରଚନା ggml_cgraph * gf);
ଶୂନ୍ୟ             ggml_metal_graph_optimize(ggml_metal_t ctx, ସଂରଚନା ggml_cgraph * gf);

ଶୂନ୍ୟ ggml_metal_event_record(ggml_metal_t ctx, ggml_metal_event_t ev);
ଶୂନ୍ୟ ggml_metal_event_wait  (ggml_metal_t ctx, ggml_metal_event_t ev);

ggml_metal_event_t ggml_metal_get_ev_cpy(ggml_metal_t ctx);

ଶୂନ୍ୟ ggml_metal_set_n_cb            (ggml_metal_t ctx, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_cb);
ଶୂନ୍ୟ ggml_metal_set_abort_callback  (ggml_metal_t ctx, ggml_abort_callback abort_callback, ଶୂନ୍ୟ * user_data);

bool ggml_metal_supports_family     (ggml_metal_t ctx, ପୂର୍ଣ୍ଣସଂଖ୍ୟା family);
ଶୂନ୍ୟ ggml_metal_capture_next_compute(ggml_metal_t ctx);

#ifdef __cplusplus
}
#endif
