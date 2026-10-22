#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_FWHT_HPP
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SYCL_FWHT_HPP

#ਅਨ੍ਤਰ੍ਭਾਵ "common.hpp"

// Fast Walsh-Hadamard transform, the fast path for a MUL_MAT whose src0 ggml has
// tagged GGML_HINT_SRC0_IS_HADAMARD. src0 is not read at all. Returns false if the
// shape is not one this can serve, in which case the caller must fall through to the
// ordinary mat-mul dispatch.
ਤਾਰਕਿਕ ggml_sycl_op_fwht(ggml_backend_sycl_context & ctx, ਅਚਲ ggml_tensor * src, ggml_tensor * dst);

#ਜੇ_ਅੰਤ  // GGML_SYCL_FWHT_HPP
