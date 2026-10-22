#اگر_غیر_تعریف_شدہ GGML_SYCL_FUSION_HPP
#تعریف_کرو GGML_SYCL_FUSION_HPP

#شامل <initializer_list>

#شامل "common.hpp"

// Backend-side fusability test. `ops` names a candidate op sequence starting at cgraph node
// `node_idx`, and `unary_ops` the GGML_UNARY_OP each GGML_OP_UNARY in `ops` must carry, in
// order; the result is true only if ggml considers that subgraph fusable *and* the SYCL
// kernel which would service it accepts the tensors involved (types, shapes, contiguity).
//
// Lives in its own translation unit because it grows a branch per supported op sequence.
منطقی ggml_sycl_can_fuse(ثابت ggml_cgraph * cgraph, صحیح گرہ_idx, std::initializer_list<شمار ggml_op> ops,
                        std::initializer_list<شمار ggml_unary_op> unary_ops);

#اگر_ختم  // GGML_SYCL_FUSION_HPP
