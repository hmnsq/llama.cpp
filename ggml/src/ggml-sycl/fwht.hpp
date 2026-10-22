#که_نه_تعریف_شوی GGML_SYCL_FWHT_HPP
#تعریف_کړه GGML_SYCL_FWHT_HPP

#ضمن "common.hpp"

// Fast Walsh-Hadamard transform, the fast path for a MUL_MAT whose src0 ggml has
// tagged GGML_HINT_SRC0_IS_HADAMARD. src0 is not read at all. Returns false if the
// shape is not one this can serve, in which case the caller must fall through to the
// ordinary mat-mul dispatch.
منطقي ggml_sycl_op_fwht(ggml_backend_sycl_context & ctx, ثابت ggml_tensor * src, ggml_tensor * dst);

#د_که_پای  // GGML_SYCL_FWHT_HPP
