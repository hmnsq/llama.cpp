#pragma once

/* device */
शून्य                           apir_device_get_device_गिनती(संरचना virtgpu * gpu);
पूर्णाङ्क                            apir_device_get_गिनती(संरचना virtgpu * gpu);
वर्ण *                         apir_device_get_name(संरचना virtgpu * gpu);
वर्ण *                         apir_device_get_description(संरचना virtgpu * gpu);
uint32_t                       apir_device_get_type(संरचना virtgpu * gpu);
शून्य                           apir_device_get_memory(संरचना virtgpu * gpu, size_t * मुक्त, size_t * total);
bool                           apir_device_supports_op(संरचना virtgpu * gpu, ध्रुव ggml_tensor * op);
apir_buffer_type_host_handle_t apir_device_get_buffer_type(संरचना virtgpu * gpu);
शून्य                           apir_device_get_props(संरचना virtgpu * gpu,
                                                     bool *           async,
                                                     bool *           host_buffer,
                                                     bool *           buffer_from_host_ptr,
                                                     bool *           events,
                                                     bool *           mmap_support);
apir_buffer_context_t          apir_device_buffer_from_ptr(संरचना virtgpu * gpu, size_t size, size_t max_tensor_size);

/* buffer-type */
वर्ण *                apir_buffer_type_get_name(संरचना virtgpu * gpu, apir_buffer_type_host_handle_t host_handle);
size_t                apir_buffer_type_get_alignment(संरचना virtgpu * gpu, apir_buffer_type_host_handle_t host_handle);
size_t                apir_buffer_type_get_max_size(संरचना virtgpu * gpu, apir_buffer_type_host_handle_t host_handle);
/* apir_buffer_type_is_host is deprecated. */
apir_buffer_context_t apir_buffer_type_alloc_buffer(संरचना virtgpu *               gpu,
                                                    apir_buffer_type_host_handle_t host_handle,
                                                    size_t                         size);
size_t                apir_buffer_type_get_alloc_size(संरचना virtgpu *               gpu,
                                                      apir_buffer_type_host_handle_t host_handle,
                                                      ध्रुव ggml_tensor *            op);

/* buffer */
शून्य * apir_buffer_get_base(संरचना virtgpu * gpu, apir_buffer_context_t * buffer_context);
शून्य   apir_buffer_set_tensor(संरचना virtgpu *        gpu,
                              apir_buffer_context_t * buffer_context,
                              ggml_tensor *           tensor,
                              ध्रुव शून्य *            data,
                              size_t                  offset,
                              size_t                  size);
शून्य   apir_buffer_get_tensor(संरचना virtgpu *        gpu,
                              apir_buffer_context_t * buffer_context,
                              ध्रुव ggml_tensor *     tensor,
                              शून्य *                  data,
                              size_t                  offset,
                              size_t                  size);
bool   apir_buffer_cpy_tensor(संरचना virtgpu *        gpu,
                              apir_buffer_context_t * buffer_context,
                              ध्रुव ggml_tensor *     src,
                              ध्रुव ggml_tensor *     dst);
शून्य   apir_buffer_clear(संरचना virtgpu * gpu, apir_buffer_context_t * buffer_context, uint8_t मूल्य);
शून्य   apir_buffer_free_buffer(संरचना virtgpu * gpu, apir_buffer_context_t * buffer_context);

/* backend */
ggml_status apir_backend_graph_compute(संरचना virtgpu * gpu, ggml_cgraph * cgraph);
