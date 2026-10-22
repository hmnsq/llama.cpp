#لارښوونه once

/* device */
فراغ                           apir_device_get_device_شمېر(بنية virtgpu * gpu);
صحيح                            apir_device_get_شمېر(بنية virtgpu * gpu);
حرف *                         apir_device_get_name(بنية virtgpu * gpu);
حرف *                         apir_device_get_description(بنية virtgpu * gpu);
uint32_t                       apir_device_get_type(بنية virtgpu * gpu);
فراغ                           apir_device_get_حافظه(بنية virtgpu * gpu, size_t * حرر, size_t * ټول);
bool                           apir_device_supports_op(بنية virtgpu * gpu, ثابت ggml_tensor * op);
apir_بفر_type_host_handle_t apir_device_get_بفر_type(بنية virtgpu * gpu);
فراغ                           apir_device_get_props(بنية virtgpu * gpu,
                                                     bool *           async,
                                                     bool *           host_بفر,
                                                     bool *           بفر_from_host_ptr,
                                                     bool *           events,
                                                     bool *           mmap_support);
apir_بفر_context_t          apir_device_بفر_from_ptr(بنية virtgpu * gpu, size_t اندازه, size_t max_tensor_اندازه);

/* buffer-type */
حرف *                apir_بفر_type_get_name(بنية virtgpu * gpu, apir_بفر_type_host_handle_t host_handle);
size_t                apir_بفر_type_get_alignment(بنية virtgpu * gpu, apir_بفر_type_host_handle_t host_handle);
size_t                apir_بفر_type_get_max_اندازه(بنية virtgpu * gpu, apir_بفر_type_host_handle_t host_handle);
/* apir_buffer_type_is_host is deprecated. */
apir_بفر_context_t apir_بفر_type_alloc_بفر(بنية virtgpu *               gpu,
                                                    apir_بفر_type_host_handle_t host_handle,
                                                    size_t                         اندازه);
size_t                apir_بفر_type_get_alloc_اندازه(بنية virtgpu *               gpu,
                                                      apir_بفر_type_host_handle_t host_handle,
                                                      ثابت ggml_tensor *            op);

/*بفر*/
فراغ * apir_بفر_get_بنسټ(بنية virtgpu * gpu, apir_بفر_context_t * بفر_context);
فراغ   apir_بفر_سېټ_tensor(بنية virtgpu *        gpu,
                              apir_بفر_context_t * بفر_context,
                              ggml_tensor *           tensor,
                              ثابت فراغ *            معلومات,
                              size_t                  دننه_واټن,
                              size_t                  اندازه);
فراغ   apir_بفر_get_tensor(بنية virtgpu *        gpu,
                              apir_بفر_context_t * بفر_context,
                              ثابت ggml_tensor *     tensor,
                              فراغ *                  معلومات,
                              size_t                  دننه_واټن,
                              size_t                  اندازه);
bool   apir_بفر_cpy_tensor(بنية virtgpu *        gpu,
                              apir_بفر_context_t * بفر_context,
                              ثابت ggml_tensor *     src,
                              ثابت ggml_tensor *     dst);
فراغ   apir_بفر_clear(بنية virtgpu * gpu, apir_بفر_context_t * بفر_context, uint8_t ارزښت);
فراغ   apir_بفر_free_بفر(بنية virtgpu * gpu, apir_بفر_context_t * بفر_context);

/* backend */
ggml_حالت apir_backend_ګراف_compute(بنية virtgpu * gpu, ggml_cgraph * cgraph);
