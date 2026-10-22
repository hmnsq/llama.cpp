#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_SYCL_FUSION_HPP
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SYCL_FUSION_HPP

#ಅನ್ತರ್ಭಾವ <initializer_list>

#ಅನ್ತರ್ಭಾವ "common.hpp"

// Backend-side fusability test. `ops` names a candidate op sequence starting at cgraph node
// `node_idx`, and `unary_ops` the GGML_UNARY_OP each GGML_OP_UNARY in `ops` must carry, in
// order; the result is true only if ggml considers that subgraph fusable *and* the SYCL
// kernel which would service it accepts the tensors involved (types, shapes, contiguity).
//
// Lives in its own translation unit because it grows a branch per supported op sequence.
ತಾರ್ಕಿಕ ggml_sycl_can_fuse(ಸ್ಥಿರಾಂಕ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, std::initializer_list<ಎಣಿಕೆ ggml_op> ops,
                        std::initializer_list<ಎಣಿಕೆ ggml_unary_op> unary_ops);

#ಆದರೆ_ಅಂತ್ಯ  // GGML_SYCL_FUSION_HPP
