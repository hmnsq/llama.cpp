#ہدایت once

/* device */
فراغ                           apir_device_get_device_گنتی(بنية virtgpu * gpu);
صحيح                            apir_device_get_گنتی(بنية virtgpu * gpu);
حرف *                         apir_device_get_name(بنية virtgpu * gpu);
حرف *                         apir_device_get_description(بنية virtgpu * gpu);
uint32_t                       apir_device_get_type(بنية virtgpu * gpu);
فراغ                           apir_device_get_حافظہ(بنية virtgpu * gpu, size_t * حرر, size_t * کل);
bool                           apir_device_supports_op(بنية virtgpu * gpu, ثابت ggml_tensor * op);
apir_بفر_type_host_handle_t apir_device_get_بفر_type(بنية virtgpu * gpu);
فراغ                           apir_device_get_props(بنية virtgpu * gpu,
                                                     bool *           async,
                                                     bool *           host_بفر,
                                                     bool *           بفر_from_host_ptr,
                                                     bool *           events,
                                                     bool *           mmap_support);
apir_بفر_context_t          apir_device_بفر_from_ptr(بنية virtgpu * gpu, size_t ناپ, size_t max_tensor_ناپ);

/* buffer-type */
حرف *                apir_بفر_type_get_name(بنية virtgpu * gpu, apir_بفر_type_host_handle_t host_handle);
size_t                apir_بفر_type_get_alignment(بنية virtgpu * gpu, apir_بفر_type_host_handle_t host_handle);
size_t                apir_بفر_type_get_max_ناپ(بنية virtgpu * gpu, apir_بفر_type_host_handle_t host_handle);
/* apir_buffer_type_is_host is deprecated. */
apir_بفر_context_t apir_بفر_type_alloc_بفر(بنية virtgpu *               gpu,
                                                    apir_بفر_type_host_handle_t host_handle,
                                                    size_t                         ناپ);
size_t                apir_بفر_type_get_alloc_ناپ(بنية virtgpu *               gpu,
                                                      apir_بفر_type_host_handle_t host_handle,
                                                      ثابت ggml_tensor *            op);

/*بفر*/
فراغ * apir_بفر_get_ادھار(بنية virtgpu * gpu, apir_بفر_context_t * بفر_context);
فراغ   apir_بفر_سیٹ_tensor(بنية virtgpu *        gpu,
                              apir_بفر_context_t * بفر_context,
                              ggml_tensor *           tensor,
                              ثابت فراغ *            انکڑے,
                              size_t                  وتھ,
                              size_t                  ناپ);
فراغ   apir_بفر_get_tensor(بنية virtgpu *        gpu,
                              apir_بفر_context_t * بفر_context,
                              ثابت ggml_tensor *     tensor,
                              فراغ *                  انکڑے,
                              size_t                  وتھ,
                              size_t                  ناپ);
bool   apir_بفر_cpy_tensor(بنية virtgpu *        gpu,
                              apir_بفر_context_t * بفر_context,
                              ثابت ggml_tensor *     src,
                              ثابت ggml_tensor *     dst);
فراغ   apir_بفر_clear(بنية virtgpu * gpu, apir_بفر_context_t * بفر_context, uint8_t قدر);
فراغ   apir_بفر_free_بفر(بنية virtgpu * gpu, apir_بفر_context_t * بفر_context);

/* backend */
ggml_حالت apir_backend_گراف_compute(بنية virtgpu * gpu, ggml_cgraph * cgraph);
