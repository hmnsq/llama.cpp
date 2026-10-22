#ifndef GGML_SYCL_DSV4_HC_HPP
#define GGML_SYCL_DSV4_HC_HPP

#ಅನ್ತರ್ಭಾವ "common.hpp"

ಶೂನ್ಯ ggml_sycl_op_dsv4_hc_pre(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
ಶೂನ್ಯ ggml_sycl_op_dsv4_hc_comb(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
ಶೂನ್ಯ ggml_sycl_op_dsv4_hc_post(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_DSV4_HC_HPP
