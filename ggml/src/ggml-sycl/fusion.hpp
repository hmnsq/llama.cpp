#నిర్వచితం_కాకపోతే GGML_SYCL_FUSION_HPP
#నిర్వచించు GGML_SYCL_FUSION_HPP

#అన్తర్భావ <initializer_list>

#అన్తర్భావ "common.hpp"

// Backend-side fusability test. `ops` names a candidate op sequence starting at cgraph node
// `node_idx`, and `unary_ops` the GGML_UNARY_OP each GGML_OP_UNARY in `ops` must carry, in
// order; the result is true only if ggml considers that subgraph fusable *and* the SYCL
// kernel which would service it accepts the tensors involved (types, shapes, contiguity).
//
// Lives in its own translation unit because it grows a branch per supported op sequence.
తార్కిక ggml_sycl_can_fuse(స్థిరరాశి ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx, std::initializer_list<గణన ggml_op> ops,
                        std::initializer_list<గణన ggml_unary_op> unary_ops);

#అయితే_ముగింపు  // GGML_SYCL_FUSION_HPP
