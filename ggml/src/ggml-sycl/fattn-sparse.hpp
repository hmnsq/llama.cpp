#که_نه_تعریف_شوی GGML_SYCL_FATTN_SPARSE_HPP
#تعریف_کړه GGML_SYCL_FATTN_SPARSE_HPP

#ضمن "common.hpp"

// Gather the K/V rows selected by a sparse mask and re-dispatch the dense
// kernels onto them. Returns false if the caller should use the dense path.
منطقي ggml_sycl_flash_attn_ext_sparse(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#د_که_پای // GGML_SYCL_FATTN_SPARSE_HPP
