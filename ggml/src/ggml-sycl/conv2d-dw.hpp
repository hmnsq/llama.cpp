#ifndef GGML_SYCL_CONV2D_DW_HPP
#define GGML_SYCL_CONV2D_DW_HPP

#અન્તર્ભાવ "common.hpp"

#define SYCL_CONV2D_DW_BLOCK_SIZE 256

શૂન્ય ggml_sycl_op_conv2d_dw(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_CONV2D_DW_HPP
