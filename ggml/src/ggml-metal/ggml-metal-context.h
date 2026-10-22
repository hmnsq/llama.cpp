#ಸೂಚನೆ once

#ಅನ್ತರ್ಭಾವ "ggml-metal-device.h"

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __cplusplus
ಬಾಹ್ಯ "C" {
#ಆದರೆ_ಅಂತ್ಯ

//
// backend context
//

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ ggml_metal * ggml_metal_t;

ggml_metal_t ggml_metal_init(ggml_metal_device_t dev);
ಶೂನ್ಯ ggml_metal_free(ggml_metal_t ctx);

ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_metal_get_name(ggml_metal_t ctx);

ಶೂನ್ಯ ggml_metal_synchronize(ggml_metal_t ctx);

ಶೂನ್ಯ ggml_metal_ಗಣ_tensor_async(ggml_metal_t ctx, ರಚನೆ ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t ಗಾತ್ರ_2);
ಶೂನ್ಯ ggml_metal_get_tensor_async(ggml_metal_t ctx, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor, ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t ಗಾತ್ರ_2);
bool ggml_metal_cpy_tensor_async(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src, ರಚನೆ ggml_tensor * dst);

ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ ggml_metal_ನಕ್ಷೆ_compute (ggml_metal_t ctx, ರಚನೆ ggml_cgraph * gf);
ಶೂನ್ಯ             ggml_metal_ನಕ್ಷೆ_optimize(ggml_metal_t ctx, ರಚನೆ ggml_cgraph * gf);

ಶೂನ್ಯ ggml_metal_event_ದಾಖಲೆ(ggml_metal_t ctx, ggml_metal_event_t ev);
ಶೂನ್ಯ ggml_metal_event_wait  (ggml_metal_t ctx, ggml_metal_event_t ev);

ggml_metal_event_t ggml_metal_get_ev_cpy(ggml_metal_t ctx);

ಶೂನ್ಯ ggml_metal_ಗಣ_n_cb            (ggml_metal_t ctx, ಪೂರ್ಣಾಂಕ n_cb);
ಶೂನ್ಯ ggml_metal_ಗಣ_abort_callback  (ggml_metal_t ctx, ggml_abort_callback abort_callback, ಶೂನ್ಯ * user_ದತ್ತಾಂಶ);

bool ggml_metal_supports_family     (ggml_metal_t ctx, ಪೂರ್ಣಾಂಕ family);
ಶೂನ್ಯ ggml_metal_capture_ಮುಂದಿನ_compute(ggml_metal_t ctx);

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __cplusplus
}
#ಆದರೆ_ಅಂತ್ಯ
