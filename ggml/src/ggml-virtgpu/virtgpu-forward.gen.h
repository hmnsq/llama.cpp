#pragma once

/* device */
शून्य                           apir_device_get_device_मोजणी(संरचना virtgpu * gpu);
पूर्णांक                            apir_device_get_मोजणी(संरचना virtgpu * gpu);
अक्षर *                         apir_device_get_name(संरचना virtgpu * gpu);
अक्षर *                         apir_device_get_description(संरचना virtgpu * gpu);
uint32_t                       apir_device_get_type(संरचना virtgpu * gpu);
शून्य                           apir_device_get_memory(संरचना virtgpu * gpu, माप_प्रकार * मुक्त, माप_प्रकार * total);
bool                           apir_device_supports_op(संरचना virtgpu * gpu, स्थिरांक ggml_tensor * op);
apir_buffer_type_host_handle_t apir_device_get_buffer_type(संरचना virtgpu * gpu);
शून्य                           apir_device_get_props(संरचना virtgpu * gpu,
                                                     bool *           async,
                                                     bool *           host_buffer,
                                                     bool *           buffer_from_host_ptr,
                                                     bool *           events,
                                                     bool *           mmap_support);
apir_buffer_context_t          apir_device_buffer_from_ptr(संरचना virtgpu * gpu, माप_प्रकार size, माप_प्रकार max_tensor_size);

/* buffer-type */
अक्षर *                apir_buffer_type_get_name(संरचना virtgpu * gpu, apir_buffer_type_host_handle_t host_handle);
माप_प्रकार                apir_buffer_type_get_alignment(संरचना virtgpu * gpu, apir_buffer_type_host_handle_t host_handle);
माप_प्रकार                apir_buffer_type_get_max_size(संरचना virtgpu * gpu, apir_buffer_type_host_handle_t host_handle);
/* apir_buffer_type_is_host is deprecated. */
apir_buffer_context_t apir_buffer_type_alloc_buffer(संरचना virtgpu *               gpu,
                                                    apir_buffer_type_host_handle_t host_handle,
                                                    माप_प्रकार                         size);
माप_प्रकार                apir_buffer_type_get_alloc_size(संरचना virtgpu *               gpu,
                                                      apir_buffer_type_host_handle_t host_handle,
                                                      स्थिरांक ggml_tensor *            op);

/* buffer */
शून्य * apir_buffer_get_base(संरचना virtgpu * gpu, apir_buffer_context_t * buffer_context);
शून्य   apir_buffer_set_tensor(संरचना virtgpu *        gpu,
                              apir_buffer_context_t * buffer_context,
                              ggml_tensor *           tensor,
                              स्थिरांक शून्य *            data,
                              माप_प्रकार                  offset,
                              माप_प्रकार                  size);
शून्य   apir_buffer_get_tensor(संरचना virtgpu *        gpu,
                              apir_buffer_context_t * buffer_context,
                              स्थिरांक ggml_tensor *     tensor,
                              शून्य *                  data,
                              माप_प्रकार                  offset,
                              माप_प्रकार                  size);
bool   apir_buffer_cpy_tensor(संरचना virtgpu *        gpu,
                              apir_buffer_context_t * buffer_context,
                              स्थिरांक ggml_tensor *     src,
                              स्थिरांक ggml_tensor *     dst);
शून्य   apir_buffer_clear(संरचना virtgpu * gpu, apir_buffer_context_t * buffer_context, uint8_t मूल्य);
शून्य   apir_buffer_free_buffer(संरचना virtgpu * gpu, apir_buffer_context_t * buffer_context);

/* backend */
ggml_status apir_backend_graph_compute(संरचना virtgpu * gpu, ggml_cgraph * cgraph);
