#ਹਦਾਇਤ once

/* device */
ਖਾਲੀ                           apir_device_get_device_ਗਿਣਤੀ(ਬਣਤਰ virtgpu * gpu);
ਪੂਰਨ                            apir_device_get_ਗਿਣਤੀ(ਬਣਤਰ virtgpu * gpu);
ਅੱਖਰ *                         apir_device_get_name(ਬਣਤਰ virtgpu * gpu);
ਅੱਖਰ *                         apir_device_get_description(ਬਣਤਰ virtgpu * gpu);
uint32_t                       apir_device_get_type(ਬਣਤਰ virtgpu * gpu);
ਖਾਲੀ                           apir_device_get_ਸਮ੍ਰਿਤੀ(ਬਣਤਰ virtgpu * gpu, size_t * ਮੁਕਤ, size_t * ਕੁੱਲ);
bool                           apir_device_supports_op(ਬਣਤਰ virtgpu * gpu, ਅਚਲ ggml_tensor * op);
apir_ਬਫਰ_type_host_handle_t apir_device_get_ਬਫਰ_type(ਬਣਤਰ virtgpu * gpu);
ਖਾਲੀ                           apir_device_get_props(ਬਣਤਰ virtgpu * gpu,
                                                     bool *           async,
                                                     bool *           host_ਬਫਰ,
                                                     bool *           ਬਫਰ_from_host_ptr,
                                                     bool *           events,
                                                     bool *           mmap_support);
apir_ਬਫਰ_context_t          apir_device_ਬਫਰ_from_ptr(ਬਣਤਰ virtgpu * gpu, size_t ਮਾਪ, size_t max_tensor_ਮਾਪ);

/* buffer-type */
ਅੱਖਰ *                apir_ਬਫਰ_type_get_name(ਬਣਤਰ virtgpu * gpu, apir_ਬਫਰ_type_host_handle_t host_handle);
size_t                apir_ਬਫਰ_type_get_alignment(ਬਣਤਰ virtgpu * gpu, apir_ਬਫਰ_type_host_handle_t host_handle);
size_t                apir_ਬਫਰ_type_get_max_ਮਾਪ(ਬਣਤਰ virtgpu * gpu, apir_ਬਫਰ_type_host_handle_t host_handle);
/* apir_buffer_type_is_host is deprecated. */
apir_ਬਫਰ_context_t apir_ਬਫਰ_type_alloc_ਬਫਰ(ਬਣਤਰ virtgpu *               gpu,
                                                    apir_ਬਫਰ_type_host_handle_t host_handle,
                                                    size_t                         ਮਾਪ);
size_t                apir_ਬਫਰ_type_get_alloc_ਮਾਪ(ਬਣਤਰ virtgpu *               gpu,
                                                      apir_ਬਫਰ_type_host_handle_t host_handle,
                                                      ਅਚਲ ggml_tensor *            op);

/*ਬਫਰ*/
ਖਾਲੀ * apir_ਬਫਰ_get_ਅਧਾਰ(ਬਣਤਰ virtgpu * gpu, apir_ਬਫਰ_context_t * ਬਫਰ_context);
ਖਾਲੀ   apir_ਬਫਰ_ਸੈੱਟ_tensor(ਬਣਤਰ virtgpu *        gpu,
                              apir_ਬਫਰ_context_t * ਬਫਰ_context,
                              ggml_tensor *           tensor,
                              ਅਚਲ ਖਾਲੀ *            ਅੰਕੜੇ,
                              size_t                  ਵਿੱਥ,
                              size_t                  ਮਾਪ);
ਖਾਲੀ   apir_ਬਫਰ_get_tensor(ਬਣਤਰ virtgpu *        gpu,
                              apir_ਬਫਰ_context_t * ਬਫਰ_context,
                              ਅਚਲ ggml_tensor *     tensor,
                              ਖਾਲੀ *                  ਅੰਕੜੇ,
                              size_t                  ਵਿੱਥ,
                              size_t                  ਮਾਪ);
bool   apir_ਬਫਰ_cpy_tensor(ਬਣਤਰ virtgpu *        gpu,
                              apir_ਬਫਰ_context_t * ਬਫਰ_context,
                              ਅਚਲ ggml_tensor *     src,
                              ਅਚਲ ggml_tensor *     dst);
ਖਾਲੀ   apir_ਬਫਰ_clear(ਬਣਤਰ virtgpu * gpu, apir_ਬਫਰ_context_t * ਬਫਰ_context, uint8_t ਮੁੱਲ);
ਖਾਲੀ   apir_ਬਫਰ_free_ਬਫਰ(ਬਣਤਰ virtgpu * gpu, apir_ਬਫਰ_context_t * ਬਫਰ_context);

/* backend */
ggml_ਸਥਿਤੀ apir_backend_ਗ੍ਰਾਫ_compute(ਬਣਤਰ virtgpu * gpu, ggml_cgraph * cgraph);
