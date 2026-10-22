#अगर_अघोषित GGML_SYCL_FUSION_HPP
#घोषणा GGML_SYCL_FUSION_HPP

#समावेश <initializer_list>

#समावेश "common.hpp"

// Backend-side fusability test. `ops` names a candidate op sequence starting at cgraph node
// `node_idx`, and `unary_ops` the GGML_UNARY_OP each GGML_OP_UNARY in `ops` must carry, in
// order; the result is true only if ggml considers that subgraph fusable *and* the SYCL
// kernel which would service it accepts the tensors involved (types, shapes, contiguity).
//
// Lives in its own translation unit because it grows a branch per supported op sequence.
bool ggml_sycl_can_fuse(स्थिर ggml_cgraph * cgraph, पूर्णांक गाँठ_idx, std::initializer_list<क्रमागत ggml_op> ops,
                        std::initializer_list<क्रमागत ggml_unary_op> unary_ops);

#पूर्ण_अगर  // GGML_SYCL_FUSION_HPP
