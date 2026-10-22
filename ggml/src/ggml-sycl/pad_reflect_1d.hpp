#ifndef GGML_SYCL_PAD_REFLECT_1D_HPP
#define GGML_SYCL_PAD_REFLECT_1D_HPP

#અન્તર્ભાવ "common.hpp"

#define SYCL_PAD_REFLECT_1D_BLOCK_SIZE 256

શૂન્ય ggml_sycl_op_pad_reflect_1d(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

#endif // GGML_SYCL_PAD_REFLECT_1D_HPP
