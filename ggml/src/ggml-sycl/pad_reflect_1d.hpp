#ifndef GGML_SYCL_PAD_REFLECT_1D_HPP
#define GGML_SYCL_PAD_REFLECT_1D_HPP

#अन्तर्भाव "common.hpp"

#define SYCL_PAD_REFLECT_1D_BLOCK_SIZE 256

शून्य ggml_sycl_op_pad_reflect_1d(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

#endif // GGML_SYCL_PAD_REFLECT_1D_HPP
