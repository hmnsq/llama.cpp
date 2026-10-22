#నిర్వచితం_కాకపోతే GGML_SYCL_FATTN_SPARSE_HPP
#నిర్వచించు GGML_SYCL_FATTN_SPARSE_HPP

#అన్తర్భావ "common.hpp"

// Gather the K/V rows selected by a sparse mask and re-dispatch the dense
// kernels onto them. Returns false if the caller should use the dense path.
తార్కిక ggml_sycl_flash_attn_ext_sparse(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#అయితే_ముగింపు // GGML_SYCL_FATTN_SPARSE_HPP
