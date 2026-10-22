#pragma once

#অন্তর্ভাৱ "common.hpp"

শূন্য ggml_sycl_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
শূন্য ggml_sycl_ssm_conv_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * add, ggml_tensor * silu_dst);
