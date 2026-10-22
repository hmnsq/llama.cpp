#رهنمود once

#ضمن "ggml-metal-device.h"

#اگر_تعریف_شده __cplusplus
خارجي "C" {
#پایان_اگر

//
// backend context
//

تعريفنوع بنية ggml_metal * ggml_metal_t;

ggml_metal_t ggml_metal_init(ggml_metal_device_t dev);
فراغ ggml_metal_free(ggml_metal_t ctx);

ثابت حرف * ggml_metal_get_name(ggml_metal_t ctx);

فراغ ggml_metal_synchronize(ggml_metal_t ctx);

فراغ ggml_metal_مجموعه_tensor_async(ggml_metal_t ctx, بنية ggml_tensor * tensor, ثابت فراغ * داده, size_t جابه_جایی, size_t اندازه);
فراغ ggml_metal_get_tensor_async(ggml_metal_t ctx, ثابت بنية ggml_tensor * tensor, فراغ * داده, size_t جابه_جایی, size_t اندازه);
bool ggml_metal_cpy_tensor_async(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, ثابت بنية ggml_tensor * src, بنية ggml_tensor * dst);

تعداد ggml_وضعیت ggml_metal_گراف_compute (ggml_metal_t ctx, بنية ggml_cgraph * gf);
فراغ             ggml_metal_گراف_optimize(ggml_metal_t ctx, بنية ggml_cgraph * gf);

فراغ ggml_metal_event_رکورد(ggml_metal_t ctx, ggml_metal_event_t ev);
فراغ ggml_metal_event_wait  (ggml_metal_t ctx, ggml_metal_event_t ev);

ggml_metal_event_t ggml_metal_get_ev_cpy(ggml_metal_t ctx);

فراغ ggml_metal_مجموعه_n_cb            (ggml_metal_t ctx, صحيح n_cb);
فراغ ggml_metal_مجموعه_abort_callback  (ggml_metal_t ctx, ggml_abort_callback abort_callback, فراغ * user_داده);

bool ggml_metal_supports_family     (ggml_metal_t ctx, صحيح family);
فراغ ggml_metal_capture_بعدی_compute(ggml_metal_t ctx);

#اگر_تعریف_شده __cplusplus
}
#پایان_اگر
