#pragma once

/* device */
ಶೂನ್ಯ                           apir_device_get_device_ಎಣಿಕೆ(ಸಂರಚನಾ virtgpu * gpu);
ಪೂರ್ಣಾಙ್ಕ                            apir_device_get_ಎಣಿಕೆ(ಸಂರಚನಾ virtgpu * gpu);
ವರ್ಣ *                         apir_device_get_name(ಸಂರಚನಾ virtgpu * gpu);
ವರ್ಣ *                         apir_device_get_description(ಸಂರಚನಾ virtgpu * gpu);
uint32_t                       apir_device_get_type(ಸಂರಚನಾ virtgpu * gpu);
ಶೂನ್ಯ                           apir_device_get_memory(ಸಂರಚನಾ virtgpu * gpu, size_t * ಮುಕ್ತ, size_t * total);
bool                           apir_device_supports_op(ಸಂರಚನಾ virtgpu * gpu, ಧ್ರುವ ggml_tensor * op);
apir_buffer_type_host_handle_t apir_device_get_buffer_type(ಸಂರಚನಾ virtgpu * gpu);
ಶೂನ್ಯ                           apir_device_get_props(ಸಂರಚನಾ virtgpu * gpu,
                                                     bool *           async,
                                                     bool *           host_buffer,
                                                     bool *           buffer_from_host_ptr,
                                                     bool *           events,
                                                     bool *           mmap_support);
apir_buffer_context_t          apir_device_buffer_from_ptr(ಸಂರಚನಾ virtgpu * gpu, size_t size, size_t max_tensor_size);

/* buffer-type */
ವರ್ಣ *                apir_buffer_type_get_name(ಸಂರಚನಾ virtgpu * gpu, apir_buffer_type_host_handle_t host_handle);
size_t                apir_buffer_type_get_alignment(ಸಂರಚನಾ virtgpu * gpu, apir_buffer_type_host_handle_t host_handle);
size_t                apir_buffer_type_get_max_size(ಸಂರಚನಾ virtgpu * gpu, apir_buffer_type_host_handle_t host_handle);
/* apir_buffer_type_is_host is deprecated. */
apir_buffer_context_t apir_buffer_type_alloc_buffer(ಸಂರಚನಾ virtgpu *               gpu,
                                                    apir_buffer_type_host_handle_t host_handle,
                                                    size_t                         size);
size_t                apir_buffer_type_get_alloc_size(ಸಂರಚನಾ virtgpu *               gpu,
                                                      apir_buffer_type_host_handle_t host_handle,
                                                      ಧ್ರುವ ggml_tensor *            op);

/* buffer */
ಶೂನ್ಯ * apir_buffer_get_base(ಸಂರಚನಾ virtgpu * gpu, apir_buffer_context_t * buffer_context);
ಶೂನ್ಯ   apir_buffer_set_tensor(ಸಂರಚನಾ virtgpu *        gpu,
                              apir_buffer_context_t * buffer_context,
                              ggml_tensor *           tensor,
                              ಧ್ರುವ ಶೂನ್ಯ *            data,
                              size_t                  offset,
                              size_t                  size);
ಶೂನ್ಯ   apir_buffer_get_tensor(ಸಂರಚನಾ virtgpu *        gpu,
                              apir_buffer_context_t * buffer_context,
                              ಧ್ರುವ ggml_tensor *     tensor,
                              ಶೂನ್ಯ *                  data,
                              size_t                  offset,
                              size_t                  size);
bool   apir_buffer_cpy_tensor(ಸಂರಚನಾ virtgpu *        gpu,
                              apir_buffer_context_t * buffer_context,
                              ಧ್ರುವ ggml_tensor *     src,
                              ಧ್ರುವ ggml_tensor *     dst);
ಶೂನ್ಯ   apir_buffer_clear(ಸಂರಚನಾ virtgpu * gpu, apir_buffer_context_t * buffer_context, uint8_t ಮೌಲ್ಯ);
ಶೂನ್ಯ   apir_buffer_free_buffer(ಸಂರಚನಾ virtgpu * gpu, apir_buffer_context_t * buffer_context);

/* backend */
ggml_status apir_backend_graph_compute(ಸಂರಚನಾ virtgpu * gpu, ggml_cgraph * cgraph);
