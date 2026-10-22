#ہدایت once

#شامل "ggml-metal-device.h"

#اگر_تعریف_شدہ __cplusplus
خارجی "C" {
#اگر_ختم

//
// backend context
//

قسمتعریف ساخت ggml_metal * ggml_metal_t;

ggml_metal_t ggml_metal_init(ggml_metal_device_t dev);
خالی ggml_metal_free(ggml_metal_t ctx);

ثابت حرف * ggml_metal_get_name(ggml_metal_t ctx);

خالی ggml_metal_synchronize(ggml_metal_t ctx);

خالی ggml_metal_سیٹ_tensor_async(ggml_metal_t ctx, ساخت ggml_tensor * tensor, ثابت خالی * ڈیٹا, size_t انحراف, size_t سائز);
خالی ggml_metal_get_tensor_async(ggml_metal_t ctx, ثابت ساخت ggml_tensor * tensor, خالی * ڈیٹا, size_t انحراف, size_t سائز);
bool ggml_metal_cpy_tensor_async(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, ثابت ساخت ggml_tensor * src, ساخت ggml_tensor * dst);

شمار ggml_حالت ggml_metal_گراف_compute (ggml_metal_t ctx, ساخت ggml_cgraph * gf);
خالی             ggml_metal_گراف_optimize(ggml_metal_t ctx, ساخت ggml_cgraph * gf);

خالی ggml_metal_event_ریکارڈ(ggml_metal_t ctx, ggml_metal_event_t ev);
خالی ggml_metal_event_wait  (ggml_metal_t ctx, ggml_metal_event_t ev);

ggml_metal_event_t ggml_metal_get_ev_cpy(ggml_metal_t ctx);

خالی ggml_metal_سیٹ_n_cb            (ggml_metal_t ctx, صحیح n_cb);
خالی ggml_metal_سیٹ_abort_callback  (ggml_metal_t ctx, ggml_abort_callback abort_callback, خالی * user_ڈیٹا);

bool ggml_metal_supports_family     (ggml_metal_t ctx, صحیح family);
خالی ggml_metal_capture_اگلا_compute(ggml_metal_t ctx);

#اگر_تعریف_شدہ __cplusplus
}
#اگر_ختم
