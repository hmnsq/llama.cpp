#pragma once

/* device */
ଶୂନ୍ୟ                           apir_device_get_device_ଗଣନା(ସଂରଚନା virtgpu * gpu);
ପୂର୍ଣ୍ଣସଂଖ୍ୟା                            apir_device_get_ଗଣନା(ସଂରଚନା virtgpu * gpu);
ଅକ୍ଷର *                         apir_device_get_name(ସଂରଚନା virtgpu * gpu);
ଅକ୍ଷର *                         apir_device_get_description(ସଂରଚନା virtgpu * gpu);
uint32_t                       apir_device_get_type(ସଂରଚନା virtgpu * gpu);
ଶୂନ୍ୟ                           apir_device_get_memory(ସଂରଚନା virtgpu * gpu, size_t * ମୁକ୍ତ, size_t * total);
bool                           apir_device_supports_op(ସଂରଚନା virtgpu * gpu, ଅଚଳ ggml_tensor * op);
apir_buffer_type_host_handle_t apir_device_get_buffer_type(ସଂରଚନା virtgpu * gpu);
ଶୂନ୍ୟ                           apir_device_get_props(ସଂରଚନା virtgpu * gpu,
                                                     bool *           async,
                                                     bool *           host_buffer,
                                                     bool *           buffer_from_host_ptr,
                                                     bool *           events,
                                                     bool *           mmap_support);
apir_buffer_context_t          apir_device_buffer_from_ptr(ସଂରଚନା virtgpu * gpu, size_t size, size_t max_tensor_size);

/* buffer-type */
ଅକ୍ଷର *                apir_buffer_type_get_name(ସଂରଚନା virtgpu * gpu, apir_buffer_type_host_handle_t host_handle);
size_t                apir_buffer_type_get_alignment(ସଂରଚନା virtgpu * gpu, apir_buffer_type_host_handle_t host_handle);
size_t                apir_buffer_type_get_max_size(ସଂରଚନା virtgpu * gpu, apir_buffer_type_host_handle_t host_handle);
/* apir_buffer_type_is_host is deprecated. */
apir_buffer_context_t apir_buffer_type_alloc_buffer(ସଂରଚନା virtgpu *               gpu,
                                                    apir_buffer_type_host_handle_t host_handle,
                                                    size_t                         size);
size_t                apir_buffer_type_get_alloc_size(ସଂରଚନା virtgpu *               gpu,
                                                      apir_buffer_type_host_handle_t host_handle,
                                                      ଅଚଳ ggml_tensor *            op);

/* buffer */
ଶୂନ୍ୟ * apir_buffer_get_base(ସଂରଚନା virtgpu * gpu, apir_buffer_context_t * buffer_context);
ଶୂନ୍ୟ   apir_buffer_set_tensor(ସଂରଚନା virtgpu *        gpu,
                              apir_buffer_context_t * buffer_context,
                              ggml_tensor *           tensor,
                              ଅଚଳ ଶୂନ୍ୟ *            data,
                              size_t                  offset,
                              size_t                  size);
ଶୂନ୍ୟ   apir_buffer_get_tensor(ସଂରଚନା virtgpu *        gpu,
                              apir_buffer_context_t * buffer_context,
                              ଅଚଳ ggml_tensor *     tensor,
                              ଶୂନ୍ୟ *                  data,
                              size_t                  offset,
                              size_t                  size);
bool   apir_buffer_cpy_tensor(ସଂରଚନା virtgpu *        gpu,
                              apir_buffer_context_t * buffer_context,
                              ଅଚଳ ggml_tensor *     src,
                              ଅଚଳ ggml_tensor *     dst);
ଶୂନ୍ୟ   apir_buffer_clear(ସଂରଚନା virtgpu * gpu, apir_buffer_context_t * buffer_context, uint8_t ମୂଲ୍ୟ);
ଶୂନ୍ୟ   apir_buffer_free_buffer(ସଂରଚନା virtgpu * gpu, apir_buffer_context_t * buffer_context);

/* backend */
ggml_status apir_backend_graph_compute(ସଂରଚନା virtgpu * gpu, ggml_cgraph * cgraph);
