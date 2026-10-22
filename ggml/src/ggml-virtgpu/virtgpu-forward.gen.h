#సూచన once

/* device */
శూన్య                           apir_device_get_device_లెక్క(నిర్మాణం virtgpu * gpu);
పూర్ణాంకం                            apir_device_get_లెక్క(నిర్మాణం virtgpu * gpu);
అక్షరం *                         apir_device_get_name(నిర్మాణం virtgpu * gpu);
అక్షరం *                         apir_device_get_description(నిర్మాణం virtgpu * gpu);
uint32_t                       apir_device_get_type(నిర్మాణం virtgpu * gpu);
శూన్య                           apir_device_get_జ్ఞాపకం(నిర్మాణం virtgpu * gpu, size_t * విడుదల, size_t * మొత్తం);
bool                           apir_device_supports_op(నిర్మాణం virtgpu * gpu, స్థిరరాశి ggml_tensor * op);
apir_బఫరు_type_host_handle_t apir_device_get_బఫరు_type(నిర్మాణం virtgpu * gpu);
శూన్య                           apir_device_get_props(నిర్మాణం virtgpu * gpu,
                                                     bool *           async,
                                                     bool *           host_బఫరు,
                                                     bool *           బఫరు_from_host_ptr,
                                                     bool *           events,
                                                     bool *           mmap_support);
apir_బఫరు_context_t          apir_device_బఫరు_from_ptr(నిర్మాణం virtgpu * gpu, size_t కొలత, size_t max_tensor_కొలత);

/* buffer-type */
అక్షరం *                apir_బఫరు_type_get_name(నిర్మాణం virtgpu * gpu, apir_బఫరు_type_host_handle_t host_handle);
size_t                apir_బఫరు_type_get_alignment(నిర్మాణం virtgpu * gpu, apir_బఫరు_type_host_handle_t host_handle);
size_t                apir_బఫరు_type_get_max_కొలత(నిర్మాణం virtgpu * gpu, apir_బఫరు_type_host_handle_t host_handle);
/* apir_buffer_type_is_host is deprecated. */
apir_బఫరు_context_t apir_బఫరు_type_alloc_బఫరు(నిర్మాణం virtgpu *               gpu,
                                                    apir_బఫరు_type_host_handle_t host_handle,
                                                    size_t                         కొలత);
size_t                apir_బఫరు_type_get_alloc_కొలత(నిర్మాణం virtgpu *               gpu,
                                                      apir_బఫరు_type_host_handle_t host_handle,
                                                      స్థిరరాశి ggml_tensor *            op);

/*బఫరు*/
శూన్య * apir_బఫరు_get_పునాది(నిర్మాణం virtgpu * gpu, apir_బఫరు_context_t * బఫరు_context);
శూన్య   apir_బఫరు_సమితి_tensor(నిర్మాణం virtgpu *        gpu,
                              apir_బఫరు_context_t * బఫరు_context,
                              ggml_tensor *           tensor,
                              స్థిరరాశి శూన్య *            దత్తాంశం,
                              size_t                  స్థానభ్రంశం,
                              size_t                  కొలత);
శూన్య   apir_బఫరు_get_tensor(నిర్మాణం virtgpu *        gpu,
                              apir_బఫరు_context_t * బఫరు_context,
                              స్థిరరాశి ggml_tensor *     tensor,
                              శూన్య *                  దత్తాంశం,
                              size_t                  స్థానభ్రంశం,
                              size_t                  కొలత);
bool   apir_బఫరు_cpy_tensor(నిర్మాణం virtgpu *        gpu,
                              apir_బఫరు_context_t * బఫరు_context,
                              స్థిరరాశి ggml_tensor *     src,
                              స్థిరరాశి ggml_tensor *     dst);
శూన్య   apir_బఫరు_clear(నిర్మాణం virtgpu * gpu, apir_బఫరు_context_t * బఫరు_context, uint8_t విలువ);
శూన్య   apir_బఫరు_free_బఫరు(నిర్మాణం virtgpu * gpu, apir_బఫరు_context_t * బఫరు_context);

/* backend */
ggml_స్థితి apir_backend_లేఖాచిత్రం_compute(నిర్మాణం virtgpu * gpu, ggml_cgraph * cgraph);
