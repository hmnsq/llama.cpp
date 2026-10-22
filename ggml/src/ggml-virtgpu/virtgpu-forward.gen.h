#निर्देशः once

/* device */
शून्य                           apir_device_get_device_गणनम्(संरचना virtgpu * gpu);
पूर्णाङ्क                            apir_device_get_गणनम्(संरचना virtgpu * gpu);
वर्ण *                         apir_device_get_name(संरचना virtgpu * gpu);
वर्ण *                         apir_device_get_description(संरचना virtgpu * gpu);
uint32_t                       apir_device_get_type(संरचना virtgpu * gpu);
शून्य                           apir_device_get_स्मृतिः(संरचना virtgpu * gpu, size_t * मुक्त, size_t * योग);
bool                           apir_device_supports_op(संरचना virtgpu * gpu, ध्रुव ggml_tensor * op);
apir_बफर_type_host_handle_t apir_device_get_बफर_type(संरचना virtgpu * gpu);
शून्य                           apir_device_get_props(संरचना virtgpu * gpu,
                                                     bool *           async,
                                                     bool *           host_बफर,
                                                     bool *           बफर_from_host_ptr,
                                                     bool *           events,
                                                     bool *           mmap_support);
apir_बफर_context_t          apir_device_बफर_from_ptr(संरचना virtgpu * gpu, size_t आकार, size_t max_tensor_आकार);

/* buffer-type */
वर्ण *                apir_बफर_type_get_name(संरचना virtgpu * gpu, apir_बफर_type_host_handle_t host_handle);
size_t                apir_बफर_type_get_alignment(संरचना virtgpu * gpu, apir_बफर_type_host_handle_t host_handle);
size_t                apir_बफर_type_get_max_आकार(संरचना virtgpu * gpu, apir_बफर_type_host_handle_t host_handle);
/* apir_buffer_type_is_host is deprecated. */
apir_बफर_context_t apir_बफर_type_alloc_बफर(संरचना virtgpu *               gpu,
                                                    apir_बफर_type_host_handle_t host_handle,
                                                    size_t                         आकार);
size_t                apir_बफर_type_get_alloc_आकार(संरचना virtgpu *               gpu,
                                                      apir_बफर_type_host_handle_t host_handle,
                                                      ध्रुव ggml_tensor *            op);

/*बफर*/
शून्य * apir_बफर_get_आधारः(संरचना virtgpu * gpu, apir_बफर_context_t * बफर_context);
शून्य   apir_बफर_समुच्चयः_tensor(संरचना virtgpu *        gpu,
                              apir_बफर_context_t * बफर_context,
                              ggml_tensor *           tensor,
                              ध्रुव शून्य *            दत्त,
                              size_t                  विस्थापनम्,
                              size_t                  आकार);
शून्य   apir_बफर_get_tensor(संरचना virtgpu *        gpu,
                              apir_बफर_context_t * बफर_context,
                              ध्रुव ggml_tensor *     tensor,
                              शून्य *                  दत्त,
                              size_t                  विस्थापनम्,
                              size_t                  आकार);
bool   apir_बफर_cpy_tensor(संरचना virtgpu *        gpu,
                              apir_बफर_context_t * बफर_context,
                              ध्रुव ggml_tensor *     src,
                              ध्रुव ggml_tensor *     dst);
शून्य   apir_बफर_clear(संरचना virtgpu * gpu, apir_बफर_context_t * बफर_context, uint8_t मानम्);
शून्य   apir_बफर_free_बफर(संरचना virtgpu * gpu, apir_बफर_context_t * बफर_context);

/* backend */
ggml_स्थितिः apir_backend_graph_compute(संरचना virtgpu * gpu, ggml_cgraph * cgraph);
