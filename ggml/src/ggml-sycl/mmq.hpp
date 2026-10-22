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

#اگر_غیر_تعریف_شدہ GGML_SYCL_MMQ_HPP
#تعریف_کرو GGML_SYCL_MMQ_HPP

#شامل "common.hpp"

خالی ggml_sycl_op_mul_mat_q(
    ggml_backend_sycl_context & ctx,
    ثابت ggml_tensor* src0,
    ثابت ggml_tensor* src1,
    ggml_tensor* dst,
    ثابت حرف* src0_dd_i,
    ثابت اعشاری* src1_ddf_i,
    ثابت حرف* src1_ddq_i,
    اعشاری* dst_dd_i,
    ثابت int64_t صف_کم,
    ثابت int64_t صف_بلند,
    ثابت int64_t src1_ncols,
    ثابت int64_t src1_padded_صف_سائز,
    ثابت dpct::قطار_ptr& stream);

#اگر_ختم // GGML_SYCL_MMQ_HPP
