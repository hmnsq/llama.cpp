#pragma once

/* device */
শূন্য                           apir_device_get_device_গুনতি(সংরচনা virtgpu * gpu);
পূর্ণাঙ্ক                            apir_device_get_গুনতি(সংরচনা virtgpu * gpu);
বর্ণ *                         apir_device_get_name(সংরচনা virtgpu * gpu);
বর্ণ *                         apir_device_get_description(সংরচনা virtgpu * gpu);
uint32_t                       apir_device_get_type(সংরচনা virtgpu * gpu);
শূন্য                           apir_device_get_memory(সংরচনা virtgpu * gpu, size_t * মুক্ত, size_t * total);
bool                           apir_device_supports_op(সংরচনা virtgpu * gpu, ধ্রুব ggml_tensor * op);
apir_buffer_type_host_handle_t apir_device_get_buffer_type(সংরচনা virtgpu * gpu);
শূন্য                           apir_device_get_props(সংরচনা virtgpu * gpu,
                                                     bool *           async,
                                                     bool *           host_buffer,
                                                     bool *           buffer_from_host_ptr,
                                                     bool *           events,
                                                     bool *           mmap_support);
apir_buffer_context_t          apir_device_buffer_from_ptr(সংরচনা virtgpu * gpu, size_t size, size_t max_tensor_size);

/* buffer-type */
বর্ণ *                apir_buffer_type_get_name(সংরচনা virtgpu * gpu, apir_buffer_type_host_handle_t host_handle);
size_t                apir_buffer_type_get_alignment(সংরচনা virtgpu * gpu, apir_buffer_type_host_handle_t host_handle);
size_t                apir_buffer_type_get_max_size(সংরচনা virtgpu * gpu, apir_buffer_type_host_handle_t host_handle);
/* apir_buffer_type_is_host is deprecated. */
apir_buffer_context_t apir_buffer_type_alloc_buffer(সংরচনা virtgpu *               gpu,
                                                    apir_buffer_type_host_handle_t host_handle,
                                                    size_t                         size);
size_t                apir_buffer_type_get_alloc_size(সংরচনা virtgpu *               gpu,
                                                      apir_buffer_type_host_handle_t host_handle,
                                                      ধ্রুব ggml_tensor *            op);

/* buffer */
শূন্য * apir_buffer_get_base(সংরচনা virtgpu * gpu, apir_buffer_context_t * buffer_context);
শূন্য   apir_buffer_set_tensor(সংরচনা virtgpu *        gpu,
                              apir_buffer_context_t * buffer_context,
                              ggml_tensor *           tensor,
                              ধ্রুব শূন্য *            data,
                              size_t                  offset,
                              size_t                  size);
শূন্য   apir_buffer_get_tensor(সংরচনা virtgpu *        gpu,
                              apir_buffer_context_t * buffer_context,
                              ধ্রুব ggml_tensor *     tensor,
                              শূন্য *                  data,
                              size_t                  offset,
                              size_t                  size);
bool   apir_buffer_cpy_tensor(সংরচনা virtgpu *        gpu,
                              apir_buffer_context_t * buffer_context,
                              ধ্রুব ggml_tensor *     src,
                              ধ্রুব ggml_tensor *     dst);
শূন্য   apir_buffer_clear(সংরচনা virtgpu * gpu, apir_buffer_context_t * buffer_context, uint8_t মান);
শূন্য   apir_buffer_free_buffer(সংরচনা virtgpu * gpu, apir_buffer_context_t * buffer_context);

/* backend */
ggml_status apir_backend_graph_compute(সংরচনা virtgpu * gpu, ggml_cgraph * cgraph);
