#pragma once

/* device */
શૂન્ય                           apir_device_get_device_ગણતરી(સંરચના virtgpu * gpu);
પૂર્ણાંક                            apir_device_get_ગણતરી(સંરચના virtgpu * gpu);
અક્ષર *                         apir_device_get_name(સંરચના virtgpu * gpu);
અક્ષર *                         apir_device_get_description(સંરચના virtgpu * gpu);
uint32_t                       apir_device_get_type(સંરચના virtgpu * gpu);
શૂન્ય                           apir_device_get_memory(સંરચના virtgpu * gpu, size_t * મુક્ત, size_t * total);
bool                           apir_device_supports_op(સંરચના virtgpu * gpu, અચલ ggml_tensor * op);
apir_buffer_type_host_handle_t apir_device_get_buffer_type(સંરચના virtgpu * gpu);
શૂન્ય                           apir_device_get_props(સંરચના virtgpu * gpu,
                                                     bool *           async,
                                                     bool *           host_buffer,
                                                     bool *           buffer_from_host_ptr,
                                                     bool *           events,
                                                     bool *           mmap_support);
apir_buffer_context_t          apir_device_buffer_from_ptr(સંરચના virtgpu * gpu, size_t size, size_t max_tensor_size);

/* buffer-type */
અક્ષર *                apir_buffer_type_get_name(સંરચના virtgpu * gpu, apir_buffer_type_host_handle_t host_handle);
size_t                apir_buffer_type_get_alignment(સંરચના virtgpu * gpu, apir_buffer_type_host_handle_t host_handle);
size_t                apir_buffer_type_get_max_size(સંરચના virtgpu * gpu, apir_buffer_type_host_handle_t host_handle);
/* apir_buffer_type_is_host is deprecated. */
apir_buffer_context_t apir_buffer_type_alloc_buffer(સંરચના virtgpu *               gpu,
                                                    apir_buffer_type_host_handle_t host_handle,
                                                    size_t                         size);
size_t                apir_buffer_type_get_alloc_size(સંરચના virtgpu *               gpu,
                                                      apir_buffer_type_host_handle_t host_handle,
                                                      અચલ ggml_tensor *            op);

/* buffer */
શૂન્ય * apir_buffer_get_base(સંરચના virtgpu * gpu, apir_buffer_context_t * buffer_context);
શૂન્ય   apir_buffer_set_tensor(સંરચના virtgpu *        gpu,
                              apir_buffer_context_t * buffer_context,
                              ggml_tensor *           tensor,
                              અચલ શૂન્ય *            data,
                              size_t                  offset,
                              size_t                  size);
શૂન્ય   apir_buffer_get_tensor(સંરચના virtgpu *        gpu,
                              apir_buffer_context_t * buffer_context,
                              અચલ ggml_tensor *     tensor,
                              શૂન્ય *                  data,
                              size_t                  offset,
                              size_t                  size);
bool   apir_buffer_cpy_tensor(સંરચના virtgpu *        gpu,
                              apir_buffer_context_t * buffer_context,
                              અચલ ggml_tensor *     src,
                              અચલ ggml_tensor *     dst);
શૂન્ય   apir_buffer_clear(સંરચના virtgpu * gpu, apir_buffer_context_t * buffer_context, uint8_t કિંમત);
શૂન્ય   apir_buffer_free_buffer(સંરચના virtgpu * gpu, apir_buffer_context_t * buffer_context);

/* backend */
ggml_status apir_backend_graph_compute(સંરચના virtgpu * gpu, ggml_cgraph * cgraph);
