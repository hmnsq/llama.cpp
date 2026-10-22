#رهنمود once

/* device */
فراغ                           apir_device_get_device_شمار(بنية virtgpu * gpu);
صحيح                            apir_device_get_شمار(بنية virtgpu * gpu);
حرف *                         apir_device_get_name(بنية virtgpu * gpu);
حرف *                         apir_device_get_description(بنية virtgpu * gpu);
uint32_t                       apir_device_get_type(بنية virtgpu * gpu);
فراغ                           apir_device_get_حافظه(بنية virtgpu * gpu, size_t * حرر, size_t * مجموع);
bool                           apir_device_supports_op(بنية virtgpu * gpu, ثابت ggml_tensor * op);
apir_میانگیر_type_host_handle_t apir_device_get_میانگیر_type(بنية virtgpu * gpu);
فراغ                           apir_device_get_props(بنية virtgpu * gpu,
                                                     bool *           async,
                                                     bool *           host_میانگیر,
                                                     bool *           میانگیر_from_host_ptr,
                                                     bool *           events,
                                                     bool *           mmap_support);
apir_میانگیر_context_t          apir_device_میانگیر_from_ptr(بنية virtgpu * gpu, size_t اندازه, size_t max_tensor_اندازه);

/* buffer-type */
حرف *                apir_میانگیر_type_get_name(بنية virtgpu * gpu, apir_میانگیر_type_host_handle_t host_handle);
size_t                apir_میانگیر_type_get_alignment(بنية virtgpu * gpu, apir_میانگیر_type_host_handle_t host_handle);
size_t                apir_میانگیر_type_get_max_اندازه(بنية virtgpu * gpu, apir_میانگیر_type_host_handle_t host_handle);
/* apir_buffer_type_is_host is deprecated. */
apir_میانگیر_context_t apir_میانگیر_type_alloc_میانگیر(بنية virtgpu *               gpu,
                                                    apir_میانگیر_type_host_handle_t host_handle,
                                                    size_t                         اندازه);
size_t                apir_میانگیر_type_get_alloc_اندازه(بنية virtgpu *               gpu,
                                                      apir_میانگیر_type_host_handle_t host_handle,
                                                      ثابت ggml_tensor *            op);

/*میانگیر*/
فراغ * apir_میانگیر_get_پایه(بنية virtgpu * gpu, apir_میانگیر_context_t * میانگیر_context);
فراغ   apir_میانگیر_مجموعه_tensor(بنية virtgpu *        gpu,
                              apir_میانگیر_context_t * میانگیر_context,
                              ggml_tensor *           tensor,
                              ثابت فراغ *            داده,
                              size_t                  جابه_جایی,
                              size_t                  اندازه);
فراغ   apir_میانگیر_get_tensor(بنية virtgpu *        gpu,
                              apir_میانگیر_context_t * میانگیر_context,
                              ثابت ggml_tensor *     tensor,
                              فراغ *                  داده,
                              size_t                  جابه_جایی,
                              size_t                  اندازه);
bool   apir_میانگیر_cpy_tensor(بنية virtgpu *        gpu,
                              apir_میانگیر_context_t * میانگیر_context,
                              ثابت ggml_tensor *     src,
                              ثابت ggml_tensor *     dst);
فراغ   apir_میانگیر_clear(بنية virtgpu * gpu, apir_میانگیر_context_t * میانگیر_context, uint8_t مقدار);
فراغ   apir_میانگیر_free_میانگیر(بنية virtgpu * gpu, apir_میانگیر_context_t * میانگیر_context);

/* backend */
ggml_وضعیت apir_backend_گراف_compute(بنية virtgpu * gpu, ggml_cgraph * cgraph);
