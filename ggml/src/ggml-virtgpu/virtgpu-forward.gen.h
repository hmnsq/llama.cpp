#आशय once

/* device */
व्योम                           apir_device_get_device_गिनती(काष्ठा virtgpu * gpu);
पूर्णांक                            apir_device_get_गिनती(काष्ठा virtgpu * gpu);
अक्षर *                         apir_device_get_name(काष्ठा virtgpu * gpu);
अक्षर *                         apir_device_get_description(काष्ठा virtgpu * gpu);
uint32_t                       apir_device_get_type(काष्ठा virtgpu * gpu);
व्योम                           apir_device_get_memory(काष्ठा virtgpu * gpu, माप_प्रकार * मुक्त, माप_प्रकार * योग);
bool                           apir_device_supports_op(काष्ठा virtgpu * gpu, स्थिर ggml_tensor * op);
apir_बफर_type_host_handle_t apir_device_get_बफर_type(काष्ठा virtgpu * gpu);
व्योम                           apir_device_get_props(काष्ठा virtgpu * gpu,
                                                     bool *           async,
                                                     bool *           host_बफर,
                                                     bool *           बफर_from_host_ptr,
                                                     bool *           events,
                                                     bool *           mmap_support);
apir_बफर_context_t          apir_device_बफर_from_ptr(काष्ठा virtgpu * gpu, माप_प्रकार आकार, माप_प्रकार max_tensor_आकार);

/* buffer-type */
अक्षर *                apir_बफर_type_get_name(काष्ठा virtgpu * gpu, apir_बफर_type_host_handle_t host_handle);
माप_प्रकार                apir_बफर_type_get_alignment(काष्ठा virtgpu * gpu, apir_बफर_type_host_handle_t host_handle);
माप_प्रकार                apir_बफर_type_get_max_आकार(काष्ठा virtgpu * gpu, apir_बफर_type_host_handle_t host_handle);
/* apir_buffer_type_is_host is deprecated. */
apir_बफर_context_t apir_बफर_type_alloc_बफर(काष्ठा virtgpu *               gpu,
                                                    apir_बफर_type_host_handle_t host_handle,
                                                    माप_प्रकार                         आकार);
माप_प्रकार                apir_बफर_type_get_alloc_आकार(काष्ठा virtgpu *               gpu,
                                                      apir_बफर_type_host_handle_t host_handle,
                                                      स्थिर ggml_tensor *            op);

/*बफर*/
व्योम * apir_बफर_get_base(काष्ठा virtgpu * gpu, apir_बफर_context_t * बफर_context);
व्योम   apir_बफर_set_tensor(काष्ठा virtgpu *        gpu,
                              apir_बफर_context_t * बफर_context,
                              ggml_tensor *           tensor,
                              स्थिर व्योम *            तथ्य,
                              माप_प्रकार                  offset,
                              माप_प्रकार                  आकार);
व्योम   apir_बफर_get_tensor(काष्ठा virtgpu *        gpu,
                              apir_बफर_context_t * बफर_context,
                              स्थिर ggml_tensor *     tensor,
                              व्योम *                  तथ्य,
                              माप_प्रकार                  offset,
                              माप_प्रकार                  आकार);
bool   apir_बफर_cpy_tensor(काष्ठा virtgpu *        gpu,
                              apir_बफर_context_t * बफर_context,
                              स्थिर ggml_tensor *     src,
                              स्थिर ggml_tensor *     dst);
व्योम   apir_बफर_clear(काष्ठा virtgpu * gpu, apir_बफर_context_t * बफर_context, uint8_t मूल्य);
व्योम   apir_बफर_free_बफर(काष्ठा virtgpu * gpu, apir_बफर_context_t * बफर_context);

/* backend */
ggml_status apir_backend_graph_compute(काष्ठा virtgpu * gpu, ggml_cgraph * cgraph);
