#ಸೂಚನೆ once

/* device */
ಶೂನ್ಯ                           apir_device_get_device_ಎಣಿಕೆ(ರಚನೆ virtgpu * gpu);
ಪೂರ್ಣಾಂಕ                            apir_device_get_ಎಣಿಕೆ(ರಚನೆ virtgpu * gpu);
ಅಕ್ಷರ *                         apir_device_get_name(ರಚನೆ virtgpu * gpu);
ಅಕ್ಷರ *                         apir_device_get_description(ರಚನೆ virtgpu * gpu);
uint32_t                       apir_device_get_type(ರಚನೆ virtgpu * gpu);
ಶೂನ್ಯ                           apir_device_get_ಸ್ಮರಣೆ(ರಚನೆ virtgpu * gpu, size_t * ಬಿಡುಗಡೆ, size_t * ಒಟ್ಟು);
bool                           apir_device_supports_op(ರಚನೆ virtgpu * gpu, ಸ್ಥಿರಾಂಕ ggml_tensor * op);
apir_ಬಫರ್_type_host_handle_t apir_device_get_ಬಫರ್_type(ರಚನೆ virtgpu * gpu);
ಶೂನ್ಯ                           apir_device_get_props(ರಚನೆ virtgpu * gpu,
                                                     bool *           async,
                                                     bool *           host_ಬಫರ್,
                                                     bool *           ಬಫರ್_from_host_ptr,
                                                     bool *           events,
                                                     bool *           mmap_support);
apir_ಬಫರ್_context_t          apir_device_ಬಫರ್_from_ptr(ರಚನೆ virtgpu * gpu, size_t ಗಾತ್ರ_2, size_t max_tensor_ಗಾತ್ರ);

/* buffer-type */
ಅಕ್ಷರ *                apir_ಬಫರ್_type_get_name(ರಚನೆ virtgpu * gpu, apir_ಬಫರ್_type_host_handle_t host_handle);
size_t                apir_ಬಫರ್_type_get_alignment(ರಚನೆ virtgpu * gpu, apir_ಬಫರ್_type_host_handle_t host_handle);
size_t                apir_ಬಫರ್_type_get_max_ಗಾತ್ರ(ರಚನೆ virtgpu * gpu, apir_ಬಫರ್_type_host_handle_t host_handle);
/* apir_buffer_type_is_host is deprecated. */
apir_ಬಫರ್_context_t apir_ಬಫರ್_type_alloc_ಬಫರ್(ರಚನೆ virtgpu *               gpu,
                                                    apir_ಬಫರ್_type_host_handle_t host_handle,
                                                    size_t                         ಗಾತ್ರ_2);
size_t                apir_ಬಫರ್_type_get_alloc_ಗಾತ್ರ(ರಚನೆ virtgpu *               gpu,
                                                      apir_ಬಫರ್_type_host_handle_t host_handle,
                                                      ಸ್ಥಿರಾಂಕ ggml_tensor *            op);

/*ಬಫರ್*/
ಶೂನ್ಯ * apir_ಬಫರ್_get_ತಳ(ರಚನೆ virtgpu * gpu, apir_ಬಫರ್_context_t * ಬಫರ್_context);
ಶೂನ್ಯ   apir_ಬಫರ್_ಗಣ_tensor(ರಚನೆ virtgpu *        gpu,
                              apir_ಬಫರ್_context_t * ಬಫರ್_context,
                              ggml_tensor *           tensor,
                              ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *            ದತ್ತಾಂಶ,
                              size_t                  ಸ್ಥಾನಾಂತರ,
                              size_t                  ಗಾತ್ರ_2);
ಶೂನ್ಯ   apir_ಬಫರ್_get_tensor(ರಚನೆ virtgpu *        gpu,
                              apir_ಬಫರ್_context_t * ಬಫರ್_context,
                              ಸ್ಥಿರಾಂಕ ggml_tensor *     tensor,
                              ಶೂನ್ಯ *                  ದತ್ತಾಂಶ,
                              size_t                  ಸ್ಥಾನಾಂತರ,
                              size_t                  ಗಾತ್ರ_2);
bool   apir_ಬಫರ್_cpy_tensor(ರಚನೆ virtgpu *        gpu,
                              apir_ಬಫರ್_context_t * ಬಫರ್_context,
                              ಸ್ಥಿರಾಂಕ ggml_tensor *     src,
                              ಸ್ಥಿರಾಂಕ ggml_tensor *     dst);
ಶೂನ್ಯ   apir_ಬಫರ್_clear(ರಚನೆ virtgpu * gpu, apir_ಬಫರ್_context_t * ಬಫರ್_context, uint8_t ಮೌಲ್ಯ);
ಶೂನ್ಯ   apir_ಬಫರ್_free_ಬಫರ್(ರಚನೆ virtgpu * gpu, apir_ಬಫರ್_context_t * ಬಫರ್_context);

/* backend */
ggml_ಸ್ಥಿತಿ apir_backend_ನಕ್ಷೆ_compute(ರಚನೆ virtgpu * gpu, ggml_cgraph * cgraph);
