#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_FATTN_SPARSE_HPP
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SYCL_FATTN_SPARSE_HPP

#ਅਨ੍ਤਰ੍ਭਾਵ "common.hpp"

// Gather the K/V rows selected by a sparse mask and re-dispatch the dense
// kernels onto them. Returns false if the caller should use the dense path.
ਤਾਰਕਿਕ ggml_sycl_flash_attn_ext_sparse(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#ਜੇ_ਅੰਤ // GGML_SYCL_FATTN_SPARSE_HPP
