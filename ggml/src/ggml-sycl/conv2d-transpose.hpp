#ifndef GGML_SYCL_CONV2D_TRANSPOSE_HPP
#define GGML_SYCL_CONV2D_TRANSPOSE_HPP

#شامل "common.hpp"

#define SYCL_CONV2D_TRANSPOSE_BLOCK_SIZE 256

تهی ggml_sycl_op_conv2d_transpose(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_CONV2D_TRANSPOSE_HPP
