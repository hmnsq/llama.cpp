#pragma once

#ಅನ್ತರ್ಭಾವ "common.hpp"

ಶೂನ್ಯ ggml_sycl_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
ಶೂನ್ಯ ggml_sycl_ssm_conv_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * add, ggml_tensor * silu_dst);
