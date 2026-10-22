#pragma once

/* device */
ശൂന്യം                           apir_device_get_device_എണ്ണം(ഘടന virtgpu * gpu);
പൂർണ്ണസംഖ്യ                            apir_device_get_എണ്ണം(ഘടന virtgpu * gpu);
അക്ഷരം *                         apir_device_get_name(ഘടന virtgpu * gpu);
അക്ഷരം *                         apir_device_get_description(ഘടന virtgpu * gpu);
uint32_t                       apir_device_get_type(ഘടന virtgpu * gpu);
ശൂന്യം                           apir_device_get_memory(ഘടന virtgpu * gpu, size_t * സ്വതന്ത്രം, size_t * total);
bool                           apir_device_supports_op(ഘടന virtgpu * gpu, സ്ഥിരം ggml_tensor * op);
apir_buffer_type_host_handle_t apir_device_get_buffer_type(ഘടന virtgpu * gpu);
ശൂന്യം                           apir_device_get_props(ഘടന virtgpu * gpu,
                                                     bool *           async,
                                                     bool *           host_buffer,
                                                     bool *           buffer_from_host_ptr,
                                                     bool *           events,
                                                     bool *           mmap_support);
apir_buffer_context_t          apir_device_buffer_from_ptr(ഘടന virtgpu * gpu, size_t size, size_t max_tensor_size);

/* buffer-type */
അക്ഷരം *                apir_buffer_type_get_name(ഘടന virtgpu * gpu, apir_buffer_type_host_handle_t host_handle);
size_t                apir_buffer_type_get_alignment(ഘടന virtgpu * gpu, apir_buffer_type_host_handle_t host_handle);
size_t                apir_buffer_type_get_max_size(ഘടന virtgpu * gpu, apir_buffer_type_host_handle_t host_handle);
/* apir_buffer_type_is_host is deprecated. */
apir_buffer_context_t apir_buffer_type_alloc_buffer(ഘടന virtgpu *               gpu,
                                                    apir_buffer_type_host_handle_t host_handle,
                                                    size_t                         size);
size_t                apir_buffer_type_get_alloc_size(ഘടന virtgpu *               gpu,
                                                      apir_buffer_type_host_handle_t host_handle,
                                                      സ്ഥിരം ggml_tensor *            op);

/* buffer */
ശൂന്യം * apir_buffer_get_base(ഘടന virtgpu * gpu, apir_buffer_context_t * buffer_context);
ശൂന്യം   apir_buffer_set_tensor(ഘടന virtgpu *        gpu,
                              apir_buffer_context_t * buffer_context,
                              ggml_tensor *           tensor,
                              സ്ഥിരം ശൂന്യം *            data,
                              size_t                  offset,
                              size_t                  size);
ശൂന്യം   apir_buffer_get_tensor(ഘടന virtgpu *        gpu,
                              apir_buffer_context_t * buffer_context,
                              സ്ഥിരം ggml_tensor *     tensor,
                              ശൂന്യം *                  data,
                              size_t                  offset,
                              size_t                  size);
bool   apir_buffer_cpy_tensor(ഘടന virtgpu *        gpu,
                              apir_buffer_context_t * buffer_context,
                              സ്ഥിരം ggml_tensor *     src,
                              സ്ഥിരം ggml_tensor *     dst);
ശൂന്യം   apir_buffer_clear(ഘടന virtgpu * gpu, apir_buffer_context_t * buffer_context, uint8_t മൂല്യം);
ശൂന്യം   apir_buffer_free_buffer(ഘടന virtgpu * gpu, apir_buffer_context_t * buffer_context);

/* backend */
ggml_status apir_backend_graph_compute(ഘടന virtgpu * gpu, ggml_cgraph * cgraph);
