#ifndef GGML_SYCL_FUSION_HPP
#define GGML_SYCL_FUSION_HPP

#അന്തര്ഭാവ <initializer_list>

#അന്തര്ഭാവ "common.hpp"

// Backend-side fusability test. `ops` names a candidate op sequence starting at cgraph node
// `node_idx`, and `unary_ops` the GGML_UNARY_OP each GGML_OP_UNARY in `ops` must carry, in
// order; the result is true only if ggml considers that subgraph fusable *and* the SYCL
// kernel which would service it accepts the tensors involved (types, shapes, contiguity).
//
// Lives in its own translation unit because it grows a branch per supported op sequence.
തർക്കം ggml_sycl_can_fuse(സ്ഥിരം ggml_cgraph * cgraph, പൂർണ്ണസംഖ്യ കണ്ണി_idx, std::initializer_list<എണ്ണൽ ggml_op> ops,
                        std::initializer_list<എണ്ണൽ ggml_unary_op> unary_ops);

#endif  // GGML_SYCL_FUSION_HPP
