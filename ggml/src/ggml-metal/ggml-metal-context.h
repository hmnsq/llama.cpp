#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-metal-device.h"

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __cplusplus
ਬਾਹਰੀ "C" {
#ਜੇ_ਅੰਤ

//
// backend context
//

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ ggml_metal * ggml_metal_t;

ggml_metal_t ggml_metal_init(ggml_metal_device_t dev);
ਖਾਲੀ ggml_metal_free(ggml_metal_t ctx);

ਅਚਲ ਅੱਖਰ * ggml_metal_get_name(ggml_metal_t ctx);

ਖਾਲੀ ggml_metal_synchronize(ggml_metal_t ctx);

ਖਾਲੀ ggml_metal_ਸੈੱਟ_tensor_async(ggml_metal_t ctx, ਬਣਤਰ ggml_tensor * tensor, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t ਮਾਪ);
ਖਾਲੀ ggml_metal_get_tensor_async(ggml_metal_t ctx, ਅਚਲ ਬਣਤਰ ggml_tensor * tensor, ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t ਮਾਪ);
bool ggml_metal_cpy_tensor_async(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, ਅਚਲ ਬਣਤਰ ggml_tensor * src, ਬਣਤਰ ggml_tensor * dst);

ਗਿਣਤੀ ggml_ਸਥਿਤੀ ggml_metal_ਗ੍ਰਾਫ_compute (ggml_metal_t ctx, ਬਣਤਰ ggml_cgraph * gf);
ਖਾਲੀ             ggml_metal_ਗ੍ਰਾਫ_optimize(ggml_metal_t ctx, ਬਣਤਰ ggml_cgraph * gf);

ਖਾਲੀ ggml_metal_event_ਰਿਕਾਰਡ(ggml_metal_t ctx, ggml_metal_event_t ev);
ਖਾਲੀ ggml_metal_event_wait  (ggml_metal_t ctx, ggml_metal_event_t ev);

ggml_metal_event_t ggml_metal_get_ev_cpy(ggml_metal_t ctx);

ਖਾਲੀ ggml_metal_ਸੈੱਟ_n_cb            (ggml_metal_t ctx, ਪੂਰਨ n_cb);
ਖਾਲੀ ggml_metal_ਸੈੱਟ_abort_callback  (ggml_metal_t ctx, ggml_abort_callback abort_callback, ਖਾਲੀ * user_ਅੰਕੜੇ);

bool ggml_metal_supports_family     (ggml_metal_t ctx, ਪੂਰਨ family);
ਖਾਲੀ ggml_metal_capture_ਅਗਲਾ_compute(ggml_metal_t ctx);

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __cplusplus
}
#ਜੇ_ਅੰਤ
