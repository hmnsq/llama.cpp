#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_FUSION_HPP
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SYCL_FUSION_HPP

#ਅਨ੍ਤਰ੍ਭਾਵ <initializer_list>

#ਅਨ੍ਤਰ੍ਭਾਵ "common.hpp"

// Backend-side fusability test. `ops` names a candidate op sequence starting at cgraph node
// `node_idx`, and `unary_ops` the GGML_UNARY_OP each GGML_OP_UNARY in `ops` must carry, in
// order; the result is true only if ggml considers that subgraph fusable *and* the SYCL
// kernel which would service it accepts the tensors involved (types, shapes, contiguity).
//
// Lives in its own translation unit because it grows a branch per supported op sequence.
ਤਾਰਕਿਕ ggml_sycl_can_fuse(ਅਚਲ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx, std::initializer_list<ਗਿਣਤੀ ggml_op> ops,
                        std::initializer_list<ਗਿਣਤੀ ggml_unary_op> unary_ops);

#ਜੇ_ਅੰਤ  // GGML_SYCL_FUSION_HPP
