#ہدایت once

/* device */
خالی                           apir_device_get_device_گنتی(ساخت virtgpu * gpu);
صحیح                            apir_device_get_گنتی(ساخت virtgpu * gpu);
حرف *                         apir_device_get_name(ساخت virtgpu * gpu);
حرف *                         apir_device_get_description(ساخت virtgpu * gpu);
uint32_t                       apir_device_get_type(ساخت virtgpu * gpu);
خالی                           apir_device_get_حافظہ(ساخت virtgpu * gpu, size_t * آزاد, size_t * کل);
bool                           apir_device_supports_op(ساخت virtgpu * gpu, ثابت ggml_tensor * op);
apir_بفر_type_host_handle_t apir_device_get_بفر_type(ساخت virtgpu * gpu);
خالی                           apir_device_get_props(ساخت virtgpu * gpu,
                                                     bool *           async,
                                                     bool *           host_بفر,
                                                     bool *           بفر_from_host_ptr,
                                                     bool *           events,
                                                     bool *           mmap_support);
apir_بفر_context_t          apir_device_بفر_from_ptr(ساخت virtgpu * gpu, size_t سائز, size_t max_tensor_سائز);

/* buffer-type */
حرف *                apir_بفر_type_get_name(ساخت virtgpu * gpu, apir_بفر_type_host_handle_t host_handle);
size_t                apir_بفر_type_get_alignment(ساخت virtgpu * gpu, apir_بفر_type_host_handle_t host_handle);
size_t                apir_بفر_type_get_max_سائز(ساخت virtgpu * gpu, apir_بفر_type_host_handle_t host_handle);
/* apir_buffer_type_is_host is deprecated. */
apir_بفر_context_t apir_بفر_type_alloc_بفر(ساخت virtgpu *               gpu,
                                                    apir_بفر_type_host_handle_t host_handle,
                                                    size_t                         سائز);
size_t                apir_بفر_type_get_alloc_سائز(ساخت virtgpu *               gpu,
                                                      apir_بفر_type_host_handle_t host_handle,
                                                      ثابت ggml_tensor *            op);

/*بفر*/
خالی * apir_بفر_get_بنیاد(ساخت virtgpu * gpu, apir_بفر_context_t * بفر_context);
خالی   apir_بفر_سیٹ_tensor(ساخت virtgpu *        gpu,
                              apir_بفر_context_t * بفر_context,
                              ggml_tensor *           tensor,
                              ثابت خالی *            ڈیٹا,
                              size_t                  انحراف,
                              size_t                  سائز);
خالی   apir_بفر_get_tensor(ساخت virtgpu *        gpu,
                              apir_بفر_context_t * بفر_context,
                              ثابت ggml_tensor *     tensor,
                              خالی *                  ڈیٹا,
                              size_t                  انحراف,
                              size_t                  سائز);
bool   apir_بفر_cpy_tensor(ساخت virtgpu *        gpu,
                              apir_بفر_context_t * بفر_context,
                              ثابت ggml_tensor *     src,
                              ثابت ggml_tensor *     dst);
خالی   apir_بفر_clear(ساخت virtgpu * gpu, apir_بفر_context_t * بفر_context, uint8_t قدر);
خالی   apir_بفر_free_بفر(ساخت virtgpu * gpu, apir_بفر_context_t * بفر_context);

/* backend */
ggml_حالت apir_backend_گراف_compute(ساخت virtgpu * gpu, ggml_cgraph * cgraph);
