#pragma once

/* device */
வெறுமை                           apir_device_get_device_மொத்தம்(கட்டமைப்பு virtgpu * gpu);
முழு                            apir_device_get_மொத்தம்(கட்டமைப்பு virtgpu * gpu);
எழுத்து *                         apir_device_get_name(கட்டமைப்பு virtgpu * gpu);
எழுத்து *                         apir_device_get_description(கட்டமைப்பு virtgpu * gpu);
uint32_t                       apir_device_get_type(கட்டமைப்பு virtgpu * gpu);
வெறுமை                           apir_device_get_memory(கட்டமைப்பு virtgpu * gpu, size_t * விடுவி, size_t * total);
bool                           apir_device_supports_op(கட்டமைப்பு virtgpu * gpu, மாறிலி ggml_tensor * op);
apir_buffer_type_host_handle_t apir_device_get_buffer_type(கட்டமைப்பு virtgpu * gpu);
வெறுமை                           apir_device_get_props(கட்டமைப்பு virtgpu * gpu,
                                                     bool *           async,
                                                     bool *           host_buffer,
                                                     bool *           buffer_from_host_ptr,
                                                     bool *           events,
                                                     bool *           mmap_support);
apir_buffer_context_t          apir_device_buffer_from_ptr(கட்டமைப்பு virtgpu * gpu, size_t size, size_t max_tensor_size);

/* buffer-type */
எழுத்து *                apir_buffer_type_get_name(கட்டமைப்பு virtgpu * gpu, apir_buffer_type_host_handle_t host_handle);
size_t                apir_buffer_type_get_alignment(கட்டமைப்பு virtgpu * gpu, apir_buffer_type_host_handle_t host_handle);
size_t                apir_buffer_type_get_max_size(கட்டமைப்பு virtgpu * gpu, apir_buffer_type_host_handle_t host_handle);
/* apir_buffer_type_is_host is deprecated. */
apir_buffer_context_t apir_buffer_type_alloc_buffer(கட்டமைப்பு virtgpu *               gpu,
                                                    apir_buffer_type_host_handle_t host_handle,
                                                    size_t                         size);
size_t                apir_buffer_type_get_alloc_size(கட்டமைப்பு virtgpu *               gpu,
                                                      apir_buffer_type_host_handle_t host_handle,
                                                      மாறிலி ggml_tensor *            op);

/* buffer */
வெறுமை * apir_buffer_get_base(கட்டமைப்பு virtgpu * gpu, apir_buffer_context_t * buffer_context);
வெறுமை   apir_buffer_set_tensor(கட்டமைப்பு virtgpu *        gpu,
                              apir_buffer_context_t * buffer_context,
                              ggml_tensor *           tensor,
                              மாறிலி வெறுமை *            data,
                              size_t                  offset,
                              size_t                  size);
வெறுமை   apir_buffer_get_tensor(கட்டமைப்பு virtgpu *        gpu,
                              apir_buffer_context_t * buffer_context,
                              மாறிலி ggml_tensor *     tensor,
                              வெறுமை *                  data,
                              size_t                  offset,
                              size_t                  size);
bool   apir_buffer_cpy_tensor(கட்டமைப்பு virtgpu *        gpu,
                              apir_buffer_context_t * buffer_context,
                              மாறிலி ggml_tensor *     src,
                              மாறிலி ggml_tensor *     dst);
வெறுமை   apir_buffer_clear(கட்டமைப்பு virtgpu * gpu, apir_buffer_context_t * buffer_context, uint8_t மதிப்பு);
வெறுமை   apir_buffer_free_buffer(கட்டமைப்பு virtgpu * gpu, apir_buffer_context_t * buffer_context);

/* backend */
ggml_status apir_backend_graph_compute(கட்டமைப்பு virtgpu * gpu, ggml_cgraph * cgraph);
