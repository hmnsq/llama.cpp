#هدايت once

/* device */
فراغ                           apir_device_get_device_ڳڻپ(بنية virtgpu * gpu);
صحيح                            apir_device_get_ڳڻپ(بنية virtgpu * gpu);
حرف *                         apir_device_get_name(بنية virtgpu * gpu);
حرف *                         apir_device_get_description(بنية virtgpu * gpu);
uint32_t                       apir_device_get_type(بنية virtgpu * gpu);
فراغ                           apir_device_get_يادگيري(بنية virtgpu * gpu, size_t * حرر, size_t * ڪل);
bool                           apir_device_supports_op(بنية virtgpu * gpu, ثابت ggml_tensor * op);
apir_بفر_type_host_handle_t apir_device_get_بفر_type(بنية virtgpu * gpu);
فراغ                           apir_device_get_props(بنية virtgpu * gpu,
                                                     bool *           async,
                                                     bool *           host_بفر,
                                                     bool *           بفر_from_host_ptr,
                                                     bool *           events,
                                                     bool *           mmap_support);
apir_بفر_context_t          apir_device_بفر_from_ptr(بنية virtgpu * gpu, size_t سائيز, size_t max_tensor_سائيز);

/* buffer-type */
حرف *                apir_بفر_type_get_name(بنية virtgpu * gpu, apir_بفر_type_host_handle_t host_handle);
size_t                apir_بفر_type_get_alignment(بنية virtgpu * gpu, apir_بفر_type_host_handle_t host_handle);
size_t                apir_بفر_type_get_max_سائيز(بنية virtgpu * gpu, apir_بفر_type_host_handle_t host_handle);
/* apir_buffer_type_is_host is deprecated. */
apir_بفر_context_t apir_بفر_type_alloc_بفر(بنية virtgpu *               gpu,
                                                    apir_بفر_type_host_handle_t host_handle,
                                                    size_t                         سائيز);
size_t                apir_بفر_type_get_alloc_سائيز(بنية virtgpu *               gpu,
                                                      apir_بفر_type_host_handle_t host_handle,
                                                      ثابت ggml_tensor *            op);

/*بفر*/
فراغ * apir_بفر_get_بنياد(بنية virtgpu * gpu, apir_بفر_context_t * بفر_context);
فراغ   apir_بفر_سيٽ_tensor(بنية virtgpu *        gpu,
                              apir_بفر_context_t * بفر_context,
                              ggml_tensor *           tensor,
                              ثابت فراغ *            معلومات,
                              size_t                  وٿي,
                              size_t                  سائيز);
فراغ   apir_بفر_get_tensor(بنية virtgpu *        gpu,
                              apir_بفر_context_t * بفر_context,
                              ثابت ggml_tensor *     tensor,
                              فراغ *                  معلومات,
                              size_t                  وٿي,
                              size_t                  سائيز);
bool   apir_بفر_cpy_tensor(بنية virtgpu *        gpu,
                              apir_بفر_context_t * بفر_context,
                              ثابت ggml_tensor *     src,
                              ثابت ggml_tensor *     dst);
فراغ   apir_بفر_clear(بنية virtgpu * gpu, apir_بفر_context_t * بفر_context, uint8_t قدر);
فراغ   apir_بفر_free_بفر(بنية virtgpu * gpu, apir_بفر_context_t * بفر_context);

/* backend */
ggml_حالت apir_backend_گراف_compute(بنية virtgpu * gpu, ggml_cgraph * cgraph);
