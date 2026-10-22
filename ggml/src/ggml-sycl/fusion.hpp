#جے_بنا_تعریف GGML_SYCL_FUSION_HPP
#تعریف_کرو GGML_SYCL_FUSION_HPP

#ضمن <initializer_list>

#ضمن "common.hpp"

// Backend-side fusability test. `ops` names a candidate op sequence starting at cgraph node
// `node_idx`, and `unary_ops` the GGML_UNARY_OP each GGML_OP_UNARY in `ops` must carry, in
// order; the result is true only if ggml considers that subgraph fusable *and* the SYCL
// kernel which would service it accepts the tensors involved (types, shapes, contiguity).
//
// Lives in its own translation unit because it grows a branch per supported op sequence.
منطقي ggml_sycl_can_fuse(ثابت ggml_cgraph * cgraph, صحيح گرہ_idx, std::initializer_list<تعداد ggml_op> ops,
                        std::initializer_list<تعداد ggml_unary_op> unary_ops);

#جے_انت  // GGML_SYCL_FUSION_HPP
