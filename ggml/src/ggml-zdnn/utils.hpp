#నిర్వచితం_కాకపోతే GGML_ZDNN_UTILITIES_HPP
#నిర్వచించు GGML_ZDNN_UTILITIES_HPP

#అన్తర్భావ "common.hpp"

zdnn_దత్తాంశం_types ggml_zdnn_type_mapping(ggml_type type);

శూన్య ggml_zdnn_create_tensor(zdnn_tensor_desc & pre_tfm_desc,
                             zdnn_tensor_desc & tfm_desc,
                             zdnn_ztensor     & ztensor,
                      స్థిరరాశి ggml_tensor       * src,
                      స్థిరరాశి int64_t           * ne,
                      స్థిరరాశి zdnn_దత్తాంశం_layouts   layout);

శూన్య ggml_zdnn_ఎక్కించు_tensor(zdnn_ztensor & ztensor, శూన్య * బఫరు);

int64_t ggml_zdnn_q_సమూహం_కొలత(స్థిరరాశి ggml_tensor * tensor);

శూన్య ggml_zdnn_ఎక్కించు_tensor_q(ggml_backend_zdnn_బఫరు * బఫరు, స్థిరరాశి ggml_tensor * tensor);

శూన్య ggml_zdnn_init_tensor(ggml_backend_zdnn_బఫరు * బఫరు, స్థిరరాశి ggml_tensor * tensor);

#అయితే_ముగింపు  // GGML_ZDNN_UTILITIES_HPP
