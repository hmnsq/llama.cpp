#अगर_अघोषित GGML_SYCL_FATTN_SPARSE_HPP
#घोषणा GGML_SYCL_FATTN_SPARSE_HPP

#समावेश "common.hpp"

// Gather the K/V rows selected by a sparse mask and re-dispatch the dense
// kernels onto them. Returns false if the caller should use the dense path.
bool ggml_sycl_flash_attn_ext_sparse(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#पूर्ण_अगर // GGML_SYCL_FATTN_SPARSE_HPP
