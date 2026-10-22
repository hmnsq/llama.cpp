#ifndef GGML_SYCL_PAD_REFLECT_1D_HPP
#define GGML_SYCL_PAD_REFLECT_1D_HPP

#অন্তর্ভাৱ "common.hpp"

#define SYCL_PAD_REFLECT_1D_BLOCK_SIZE 256

শূন্য ggml_sycl_op_pad_reflect_1d(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

#endif // GGML_SYCL_PAD_REFLECT_1D_HPP
