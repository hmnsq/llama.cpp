#pragma once

/* device */
تهی                           apir_device_get_device_شمار(ساختار virtgpu * gpu);
صحیح                            apir_device_get_شمار(ساختار virtgpu * gpu);
نویسه *                         apir_device_get_name(ساختار virtgpu * gpu);
نویسه *                         apir_device_get_description(ساختار virtgpu * gpu);
uint32_t                       apir_device_get_type(ساختار virtgpu * gpu);
تهی                           apir_device_get_memory(ساختار virtgpu * gpu, size_t * آزاد, size_t * total);
bool                           apir_device_supports_op(ساختار virtgpu * gpu, ثابت ggml_tensor * op);
apir_buffer_type_host_handle_t apir_device_get_buffer_type(ساختار virtgpu * gpu);
تهی                           apir_device_get_props(ساختار virtgpu * gpu,
                                                     bool *           async,
                                                     bool *           host_buffer,
                                                     bool *           buffer_from_host_ptr,
                                                     bool *           events,
                                                     bool *           mmap_support);
apir_buffer_context_t          apir_device_buffer_from_ptr(ساختار virtgpu * gpu, size_t size, size_t max_tensor_size);

/* buffer-type */
نویسه *                apir_buffer_type_get_name(ساختار virtgpu * gpu, apir_buffer_type_host_handle_t host_handle);
size_t                apir_buffer_type_get_alignment(ساختار virtgpu * gpu, apir_buffer_type_host_handle_t host_handle);
size_t                apir_buffer_type_get_max_size(ساختار virtgpu * gpu, apir_buffer_type_host_handle_t host_handle);
/* apir_buffer_type_is_host is deprecated. */
apir_buffer_context_t apir_buffer_type_alloc_buffer(ساختار virtgpu *               gpu,
                                                    apir_buffer_type_host_handle_t host_handle,
                                                    size_t                         size);
size_t                apir_buffer_type_get_alloc_size(ساختار virtgpu *               gpu,
                                                      apir_buffer_type_host_handle_t host_handle,
                                                      ثابت ggml_tensor *            op);

/* buffer */
تهی * apir_buffer_get_base(ساختار virtgpu * gpu, apir_buffer_context_t * buffer_context);
تهی   apir_buffer_set_tensor(ساختار virtgpu *        gpu,
                              apir_buffer_context_t * buffer_context,
                              ggml_tensor *           tensor,
                              ثابت تهی *            data,
                              size_t                  offset,
                              size_t                  size);
تهی   apir_buffer_get_tensor(ساختار virtgpu *        gpu,
                              apir_buffer_context_t * buffer_context,
                              ثابت ggml_tensor *     tensor,
                              تهی *                  data,
                              size_t                  offset,
                              size_t                  size);
bool   apir_buffer_cpy_tensor(ساختار virtgpu *        gpu,
                              apir_buffer_context_t * buffer_context,
                              ثابت ggml_tensor *     src,
                              ثابت ggml_tensor *     dst);
تهی   apir_buffer_clear(ساختار virtgpu * gpu, apir_buffer_context_t * buffer_context, uint8_t مقدار);
تهی   apir_buffer_free_buffer(ساختار virtgpu * gpu, apir_buffer_context_t * buffer_context);

/* backend */
ggml_status apir_backend_graph_compute(ساختار virtgpu * gpu, ggml_cgraph * cgraph);
