#ifndef GGML_SYCL_FUSION_HPP
#define GGML_SYCL_FUSION_HPP

#অন্তর্ভাব <initializer_list>

#অন্তর্ভাব "common.hpp"

// Backend-side fusability test. `ops` names a candidate op sequence starting at cgraph node
// `node_idx`, and `unary_ops` the GGML_UNARY_OP each GGML_OP_UNARY in `ops` must carry, in
// order; the result is true only if ggml considers that subgraph fusable *and* the SYCL
// kernel which would service it accepts the tensors involved (types, shapes, contiguity).
//
// Lives in its own translation unit because it grows a branch per supported op sequence.
যৌক্তিক ggml_sycl_can_fuse(ধ্রুবক ggml_cgraph * cgraph, পূর্ণসংখ্যা গ্রন্থি_idx, std::initializer_list<গণনা ggml_op> ops,
                        std::initializer_list<গণনা ggml_unary_op> unary_ops);

#endif  // GGML_SYCL_FUSION_HPP
