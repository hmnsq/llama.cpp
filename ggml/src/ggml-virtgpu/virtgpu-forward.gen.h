#pragma once

/* device */
শূন্য                           apir_device_get_device_গুনতি(কাঠামো virtgpu * gpu);
পূর্ণসংখ্যা                            apir_device_get_গুনতি(কাঠামো virtgpu * gpu);
অক্ষর *                         apir_device_get_name(কাঠামো virtgpu * gpu);
অক্ষর *                         apir_device_get_description(কাঠামো virtgpu * gpu);
uint32_t                       apir_device_get_type(কাঠামো virtgpu * gpu);
শূন্য                           apir_device_get_memory(কাঠামো virtgpu * gpu, size_t * মুক্ত, size_t * যোগ);
bool                           apir_device_supports_op(কাঠামো virtgpu * gpu, ধ্রুবক ggml_tensor * op);
apir_বাফার_type_host_handle_t apir_device_get_বাফার_type(কাঠামো virtgpu * gpu);
শূন্য                           apir_device_get_props(কাঠামো virtgpu * gpu,
                                                     bool *           async,
                                                     bool *           host_বাফার,
                                                     bool *           বাফার_from_host_ptr,
                                                     bool *           events,
                                                     bool *           mmap_support);
apir_বাফার_context_t          apir_device_বাফার_from_ptr(কাঠামো virtgpu * gpu, size_t আকার_2, size_t max_tensor_আকার);

/* buffer-type */
অক্ষর *                apir_বাফার_type_get_name(কাঠামো virtgpu * gpu, apir_বাফার_type_host_handle_t host_handle);
size_t                apir_বাফার_type_get_alignment(কাঠামো virtgpu * gpu, apir_বাফার_type_host_handle_t host_handle);
size_t                apir_বাফার_type_get_max_আকার(কাঠামো virtgpu * gpu, apir_বাফার_type_host_handle_t host_handle);
/* apir_buffer_type_is_host is deprecated. */
apir_বাফার_context_t apir_বাফার_type_alloc_বাফার(কাঠামো virtgpu *               gpu,
                                                    apir_বাফার_type_host_handle_t host_handle,
                                                    size_t                         আকার_2);
size_t                apir_বাফার_type_get_alloc_আকার(কাঠামো virtgpu *               gpu,
                                                      apir_বাফার_type_host_handle_t host_handle,
                                                      ধ্রুবক ggml_tensor *            op);

/*বাফার*/
শূন্য * apir_বাফার_get_base(কাঠামো virtgpu * gpu, apir_বাফার_context_t * বাফার_context);
শূন্য   apir_বাফার_set_tensor(কাঠামো virtgpu *        gpu,
                              apir_বাফার_context_t * বাফার_context,
                              ggml_tensor *           tensor,
                              ধ্রুবক শূন্য *            তথ্য,
                              size_t                  offset,
                              size_t                  আকার_2);
শূন্য   apir_বাফার_get_tensor(কাঠামো virtgpu *        gpu,
                              apir_বাফার_context_t * বাফার_context,
                              ধ্রুবক ggml_tensor *     tensor,
                              শূন্য *                  তথ্য,
                              size_t                  offset,
                              size_t                  আকার_2);
bool   apir_বাফার_cpy_tensor(কাঠামো virtgpu *        gpu,
                              apir_বাফার_context_t * বাফার_context,
                              ধ্রুবক ggml_tensor *     src,
                              ধ্রুবক ggml_tensor *     dst);
শূন্য   apir_বাফার_clear(কাঠামো virtgpu * gpu, apir_বাফার_context_t * বাফার_context, uint8_t মান);
শূন্য   apir_বাফার_free_বাফার(কাঠামো virtgpu * gpu, apir_বাফার_context_t * বাফার_context);

/* backend */
ggml_status apir_backend_graph_compute(কাঠামো virtgpu * gpu, ggml_cgraph * cgraph);
