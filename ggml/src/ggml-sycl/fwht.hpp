#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_SYCL_FWHT_HPP
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SYCL_FWHT_HPP

#ಅನ್ತರ್ಭಾವ "common.hpp"

// Fast Walsh-Hadamard transform, the fast path for a MUL_MAT whose src0 ggml has
// tagged GGML_HINT_SRC0_IS_HADAMARD. src0 is not read at all. Returns false if the
// shape is not one this can serve, in which case the caller must fall through to the
// ordinary mat-mul dispatch.
ತಾರ್ಕಿಕ ggml_sycl_op_fwht(ggml_backend_sycl_context & ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src, ggml_tensor * dst);

#ಆದರೆ_ಅಂತ್ಯ  // GGML_SYCL_FWHT_HPP
