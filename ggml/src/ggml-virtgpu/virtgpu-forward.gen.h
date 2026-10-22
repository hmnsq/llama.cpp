#pragma once

/* device */
فراغ                           apir_device_get_device_count(بنية virtgpu * gpu);
صحيح                            apir_device_get_count(بنية virtgpu * gpu);
حرف *                         apir_device_get_name(بنية virtgpu * gpu);
حرف *                         apir_device_get_description(بنية virtgpu * gpu);
uint32_t                       apir_device_get_type(بنية virtgpu * gpu);
فراغ                           apir_device_get_memory(بنية virtgpu * gpu, size_t * حرر, size_t * total);
bool                           apir_device_supports_op(بنية virtgpu * gpu, ثابت ggml_tensor * op);
apir_buffer_type_host_handle_t apir_device_get_buffer_type(بنية virtgpu * gpu);
فراغ                           apir_device_get_props(بنية virtgpu * gpu,
                                                     bool *           async,
                                                     bool *           host_buffer,
                                                     bool *           buffer_from_host_ptr,
                                                     bool *           events,
                                                     bool *           mmap_support);
apir_buffer_context_t          apir_device_buffer_from_ptr(بنية virtgpu * gpu, size_t size, size_t max_tensor_size);

/* buffer-type */
حرف *                apir_buffer_type_get_name(بنية virtgpu * gpu, apir_buffer_type_host_handle_t host_handle);
size_t                apir_buffer_type_get_alignment(بنية virtgpu * gpu, apir_buffer_type_host_handle_t host_handle);
size_t                apir_buffer_type_get_max_size(بنية virtgpu * gpu, apir_buffer_type_host_handle_t host_handle);
/* apir_buffer_type_is_host is deprecated. */
apir_buffer_context_t apir_buffer_type_alloc_buffer(بنية virtgpu *               gpu,
                                                    apir_buffer_type_host_handle_t host_handle,
                                                    size_t                         size);
size_t                apir_buffer_type_get_alloc_size(بنية virtgpu *               gpu,
                                                      apir_buffer_type_host_handle_t host_handle,
                                                      ثابت ggml_tensor *            op);

/* buffer */
فراغ * apir_buffer_get_base(بنية virtgpu * gpu, apir_buffer_context_t * buffer_context);
فراغ   apir_buffer_set_tensor(بنية virtgpu *        gpu,
                              apir_buffer_context_t * buffer_context,
                              ggml_tensor *           tensor,
                              ثابت فراغ *            data,
                              size_t                  offset,
                              size_t                  size);
فراغ   apir_buffer_get_tensor(بنية virtgpu *        gpu,
                              apir_buffer_context_t * buffer_context,
                              ثابت ggml_tensor *     tensor,
                              فراغ *                  data,
                              size_t                  offset,
                              size_t                  size);
bool   apir_buffer_cpy_tensor(بنية virtgpu *        gpu,
                              apir_buffer_context_t * buffer_context,
                              ثابت ggml_tensor *     src,
                              ثابت ggml_tensor *     dst);
فراغ   apir_buffer_clear(بنية virtgpu * gpu, apir_buffer_context_t * buffer_context, uint8_t value);
فراغ   apir_buffer_free_buffer(بنية virtgpu * gpu, apir_buffer_context_t * buffer_context);

/* backend */
ggml_status apir_backend_graph_compute(بنية virtgpu * gpu, ggml_cgraph * cgraph);
