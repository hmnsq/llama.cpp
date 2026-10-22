//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_SYCL_MMQ_HPP
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SYCL_MMQ_HPP

#ಅನ್ತರ್ಭಾವ "common.hpp"

ಶೂನ್ಯ ggml_sycl_op_mul_mat_q(
    ggml_backend_sycl_context & ctx,
    ಸ್ಥಿರಾಂಕ ggml_tensor* src0,
    ಸ್ಥಿರಾಂಕ ggml_tensor* src1,
    ggml_tensor* dst,
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* src0_dd_i,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* src1_ddf_i,
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* src1_ddq_i,
    ದಶಮಾಂಶ* dst_dd_i,
    ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_ಕೆಳಗಿನ,
    ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_ಮೇಲಿನ,
    ಸ್ಥಿರಾಂಕ int64_t src1_ncols,
    ಸ್ಥಿರಾಂಕ int64_t src1_padded_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ,
    ಸ್ಥಿರಾಂಕ dpct::ಸರತಿ_ptr& stream);

#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_MMQ_HPP
