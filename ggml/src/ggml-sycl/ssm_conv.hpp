#pragma once

#અન્તર્ભાવ "common.hpp"

શૂન્ય ggml_sycl_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
શૂન્ય ggml_sycl_ssm_conv_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * add, ggml_tensor * silu_dst);
