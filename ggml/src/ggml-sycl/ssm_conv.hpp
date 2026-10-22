#pragma once

#شامل "common.hpp"

تهی ggml_sycl_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
تهی ggml_sycl_ssm_conv_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * add, ggml_tensor * silu_dst);
