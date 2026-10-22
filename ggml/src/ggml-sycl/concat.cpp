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

#ضمن "ggml.h"

#ضمن "concat.hpp"

ساكن مضمن size_t elem_اندازه(ggml_type t) {
    أرجع ggml_type_اندازه(t) / ggml_blck_اندازه(t);
}

کينډۍ <ډول_نوم T>
ساكن فراغ concat_T_dim0(ثابت T *x, ثابت T *y, T *dst,
                            ثابت صحيح ne0, ثابت صحيح ne00,
                            ثابت sycl::nd_توکی<3> &توکی_ct1) {
  صحيح nidx = توکی_ct1.get_local_id(2) +
             توکی_ct1.get_ډله(2) * توکی_ct1.get_local_range(2);
  إذا (nidx >= ne0) {
    أرجع;
  }
  // operation
  صحيح دننه_واټن_dst = nidx + توکی_ct1.get_ډله(1) * ne0 +
                   توکی_ct1.get_ډله(0) * ne0 * توکی_ct1.get_ډله_range(1);
  إذا (nidx < ne00) { // src0
    صحيح دننه_واټن_src = nidx + توکی_ct1.get_ډله(1) * ne00 +
                     توکی_ct1.get_ډله(0) * ne00 * توکی_ct1.get_ډله_range(1);
    dst[دننه_واټن_dst] = x[دننه_واټن_src];
  } وإلا {
    صحيح دننه_واټن_src =
        nidx - ne00 + توکی_ct1.get_ډله(1) * (ne0 - ne00) +
        توکی_ct1.get_ډله(0) * (ne0 - ne00) * توکی_ct1.get_ډله_range(1);
    dst[دننه_واټن_dst] = y[دننه_واټن_src];
  }
}

کينډۍ <ډول_نوم T>
ساكن فراغ concat_T_dim1(ثابت T *x, ثابت T *y, T *dst,
                            ثابت صحيح ne0, ثابت صحيح ne01,
                            ثابت sycl::nd_توکی<3> &توکی_ct1) {
  صحيح nidx = توکی_ct1.get_local_id(2) +
             توکی_ct1.get_ډله(2) * توکی_ct1.get_local_range(2);
  إذا (nidx >= ne0) {
    أرجع;
  }
  // operation
  صحيح دننه_واټن_dst = nidx + توکی_ct1.get_ډله(1) * ne0 +
                   توکی_ct1.get_ډله(0) * ne0 * توکی_ct1.get_ډله_range(1);
  إذا (توکی_ct1.get_ډله(1) < (size_t) ne01) { // src0
    صحيح دننه_واټن_src =
        nidx + توکی_ct1.get_ډله(1) * ne0 + توکی_ct1.get_ډله(0) * ne0 * ne01;
    dst[دننه_واټن_dst] = x[دننه_واټن_src];
  } وإلا {
    صحيح دننه_واټن_src =
        nidx + (توکی_ct1.get_ډله(1) - ne01) * ne0 +
        توکی_ct1.get_ډله(0) * ne0 * (توکی_ct1.get_ډله_range(1) - ne01);
    dst[دننه_واټن_dst] = y[دننه_واټن_src];
  }
}

کينډۍ <ډول_نوم T>
ساكن فراغ concat_T_dim2(ثابت T *x, ثابت T *y, T *dst,
                            ثابت صحيح ne0, ثابت صحيح ne02,
                            ثابت sycl::nd_توکی<3> &توکی_ct1) {
  صحيح nidx = توکی_ct1.get_local_id(2) +
             توکی_ct1.get_ډله(2) * توکی_ct1.get_local_range(2);
  إذا (nidx >= ne0) {
    أرجع;
  }
  // operation
  صحيح دننه_واټن_dst = nidx + توکی_ct1.get_ډله(1) * ne0 +
                   توکی_ct1.get_ډله(0) * ne0 * توکی_ct1.get_ډله_range(1);
  إذا (توکی_ct1.get_ډله(0) < (size_t) ne02) { // src0
    صحيح دننه_واټن_src = nidx + توکی_ct1.get_ډله(1) * ne0 +
                     توکی_ct1.get_ډله(0) * ne0 * توکی_ct1.get_ډله_range(1);
    dst[دننه_واټن_dst] = x[دننه_واټن_src];
  } وإلا {
    صحيح دننه_واټن_src =
        nidx + توکی_ct1.get_ډله(1) * ne0 +
        (توکی_ct1.get_ډله(0) - ne02) * ne0 * توکی_ct1.get_ډله_range(1);
    dst[دننه_واټن_dst] = y[دننه_واټن_src];
  }
}

کينډۍ <ډول_نوم T>
ساكن فراغ concat_T_sycl(ثابت T *x, ثابت T *y, T *dst,
                            صحيح ne00, صحيح ne01, صحيح ne02, صحيح ne0, صحيح ne1,
                            صحيح ne2, صحيح dim, کتار_ptr stream) {
  صحيح عدد_بلاکونه = (ne0 + SYCL_CONCAT_بلاک_اندازه - 1) / SYCL_CONCAT_بلاک_اندازه;
  sycl::range<3> gridDim(ne2, ne1, عدد_بلاکونه);
  بدل (dim) {
  حالة 0:
      stream->parallel_for(sycl::nd_range<3>(gridDim * sycl::range<3>(1, 1, SYCL_CONCAT_بلاک_اندازه),
                                          sycl::range<3>(1, 1, SYCL_CONCAT_بلاک_اندازه)),
                        [=](sycl::nd_توکی<3> توکی_ct1) { concat_T_dim0<T>(x, y, dst, ne0, ne00, توکی_ct1); });
      اكسر;
  حالة 1:
      stream->parallel_for(sycl::nd_range<3>(gridDim * sycl::range<3>(1, 1, SYCL_CONCAT_بلاک_اندازه),
                                          sycl::range<3>(1, 1, SYCL_CONCAT_بلاک_اندازه)),
                        [=](sycl::nd_توکی<3> توکی_ct1) { concat_T_dim1<T>(x, y, dst, ne0, ne01, توکی_ct1); });
      اكسر;
  // dim >=2 will be dispatched to the default path
  افتراضي:
      stream->parallel_for(sycl::nd_range<3>(gridDim * sycl::range<3>(1, 1, SYCL_CONCAT_بلاک_اندازه),
                                          sycl::range<3>(1, 1, SYCL_CONCAT_بلاک_اندازه)),
                        [=](sycl::nd_توکی<3> توکی_ct1) { concat_T_dim2<T>(x, y, dst, ne0, ne02, توکی_ct1); });
      اكسر;
  }
}

// non-contiguous kernel (slow)
کينډۍ<ډول_نوم T>
ساكن فراغ concat_T_sycl_non_cont(
    کتار_ptr stream, ثابت حرف *src0, ثابت حرف *src1, حرف *dst,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03, uint64_t nb00,
    uint64_t nb01, uint64_t nb02, uint64_t nb03, int64_t /*ne10*/,
    int64_t /*ne11*/, int64_t /*ne12*/, int64_t /*ne13*/, uint64_t nb10,
    uint64_t nb11, uint64_t nb12, uint64_t nb13, int64_t ne0, int64_t ne1,
    int64_t ne2, int64_t ne3, uint64_t nb0, uint64_t nb1, uint64_t nb2,
    uint64_t nb3, int32_t dim) {
  sycl::range<3> gridDim(ne3, ne2, ne1);

  // Avoid oversubscribing device when there is not enough elements along the innermost dim to
  // fill a full SYCL_CONCAT_BLOCK_SIZE. For larger # of elements, the full SYCL_CONCAT_BLOCK_SIZE
  // is used.
  ثابت int64_t ne0_pad   = GGML_PAD(ne0, WARP_اندازه);
  ثابت int64_t بلاک_ne0 = ne0_pad < SYCL_CONCAT_بلاک_اندازه ? ne0_pad : (int64_t) SYCL_CONCAT_بلاک_اندازه;
  sycl::range<3> بلاک_Dim(1, 1, بلاک_ne0);

  stream->parallel_for(sycl::nd_range<3>(gridDim * بلاک_Dim, بلاک_Dim), [=](sycl::nd_توکی<3> توکی_ct1) {
      int64_t i3 = توکی_ct1.get_ډله(0);
      int64_t i2 = توکی_ct1.get_ډله(1);
      int64_t i1 = توکی_ct1.get_ډله(2);

      int64_t o[4] = { 0, 0, 0, 0 };
      o[dim]       = dim == 0 ? ne00 : (dim == 1 ? ne01 : (dim == 2 ? ne02 : ne03));

      ثابت T * x;

      لكل (صحيح i0 = توکی_ct1.get_local_id(2); i0 < ne0; i0 += توکی_ct1.get_local_range(2)) {
          إذا (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
              x = (ثابت T *) (src0 + (i3) *nb03 + (i2) *nb02 + (i1) *nb01 + (i0) *nb00);
          } وإلا {
              x = (ثابت T *) (src1 + (i3 - o[3]) * nb13 + (i2 - o[2]) * nb12 + (i1 - o[1]) * nb11 +
                                   (i0 - o[0]) * nb10);
          }

          T *y = (T *)(dst + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

          *y = *x;
      }
  });
}

کينډۍ <ډول_نوم T>
فراغ concat_impl_sycl(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ثابت ggml_tensor *  src0   = dst->src[0];
    ثابت ggml_tensor *  src1   = dst->src[1];
    کتار_ptr            stream = ctx.stream();

    ثابت int32_t dim = ((int32_t *) dst->op_params)[0];

    إذا (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        ثابت T * src0_d = (ثابت T *) src0->معلومات;
        ثابت T * src1_d = (ثابت T *) src1->معلومات;
        T * dst_d = (T *) dst->معلومات;
        size_t type_اندازه = elem_اندازه(dst->type);
        إذا (dim != 3) {
            لكل (صحيح i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<T>(src0_d + i3 * (src0->nb[3] / type_اندازه), src1_d + i3 * (src1->nb[3] / type_اندازه),
                                dst_d + i3 * (dst->nb[3] / type_اندازه), src0->ne[0], src0->ne[1], src0->ne[2], dst->ne[0],
                                dst->ne[1], dst->ne[2], dim, stream);
            }
        } وإلا {
            ثابت size_t size0 = ggml_nbytes(src0);
            ثابت size_t size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->حافظه_نقل(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->حافظه_نقل(dst_d + size0 / type_اندازه, src1_d, size1)));
        }
    } وإلا {
        concat_T_sycl_non_cont<T>(stream, (ثابت حرف *) src0->معلومات, (ثابت حرف *) src1->معلومات, (حرف *) dst->معلومات,
                                 src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src0->nb[0], src0->nb[1],
                                 src0->nb[2], src0->nb[3], src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                                 src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3], dst->ne[0], dst->ne[1], dst->ne[2],
                                 dst->ne[3], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

ساكن فراغ concat_impl_q4_0_sycl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ثابت ggml_tensor * src0   = dst->src[0];
    ثابت ggml_tensor * src1   = dst->src[1];
    کتار_ptr           stream = ctx.stream();

    ثابت int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(src1->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(dst->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(src0->ne[0] % QK4_0 == 0);
    GGML_ASSERT(src1->ne[0] % QK4_0 == 0);
    GGML_ASSERT(dst->ne[0] % QK4_0 == 0);

    ثابت صحيح ne00_blk = src0->ne[0] / QK4_0;
    ثابت صحيح ne0_blk  = dst->ne[0] / QK4_0;

    إذا (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        ثابت بلاک_q4_0 * src0_d = (ثابت بلاک_q4_0 *) src0->معلومات;
        ثابت بلاک_q4_0 * src1_d = (ثابت بلاک_q4_0 *) src1->معلومات;
        بلاک_q4_0 * dst_d        = (بلاک_q4_0 *) dst->معلومات;
        ثابت size_t type_اندازه    = حجم(بلاک_q4_0);

        إذا (dim != 3) {
            لكل (صحيح i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<بلاک_q4_0>(
                    src0_d + i3 * (src0->nb[3] / type_اندازه),
                    src1_d + i3 * (src1->nb[3] / type_اندازه),
                    dst_d + i3 * (dst->nb[3] / type_اندازه),
                    ne00_blk, src0->ne[1], src0->ne[2], ne0_blk,
                    dst->ne[1], dst->ne[2], dim, stream);
            }
        } وإلا {
            ثابت size_t size0 = ggml_nbytes(src0);
            ثابت size_t size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->حافظه_نقل(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->حافظه_نقل((حرف *) dst_d + size0, src1_d, size1)));
        }
    } وإلا {
        concat_T_sycl_non_cont<بلاک_q4_0>(
            stream, (ثابت حرف *) src0->معلومات, (ثابت حرف *) src1->معلومات,
            (حرف *) dst->معلومات,
            ne00_blk, src0->ne[1], src0->ne[2], src0->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->ne[0] / QK4_0, src1->ne[1], src1->ne[2], src1->ne[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            ne0_blk, dst->ne[1], dst->ne[2], dst->ne[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

ساكن فراغ concat_impl_q4_1_sycl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ثابت ggml_tensor * src0   = dst->src[0];
    ثابت ggml_tensor * src1   = dst->src[1];
    کتار_ptr           stream = ctx.stream();

    ثابت int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(src1->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(dst->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(src0->ne[0] % QK4_1 == 0);
    GGML_ASSERT(src1->ne[0] % QK4_1 == 0);
    GGML_ASSERT(dst->ne[0] % QK4_1 == 0);

    ثابت صحيح ne00_blk = src0->ne[0] / QK4_1;
    ثابت صحيح ne0_blk  = dst->ne[0] / QK4_1;

    إذا (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        ثابت بلاک_q4_1 * src0_d = (ثابت بلاک_q4_1 *) src0->معلومات;
        ثابت بلاک_q4_1 * src1_d = (ثابت بلاک_q4_1 *) src1->معلومات;
        بلاک_q4_1 * dst_d        = (بلاک_q4_1 *) dst->معلومات;
        ثابت size_t type_اندازه    = حجم(بلاک_q4_1);

        إذا (dim != 3) {
            لكل (صحيح i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<بلاک_q4_1>(
                    src0_d + i3 * (src0->nb[3] / type_اندازه),
                    src1_d + i3 * (src1->nb[3] / type_اندازه),
                    dst_d + i3 * (dst->nb[3] / type_اندازه),
                    ne00_blk, src0->ne[1], src0->ne[2], ne0_blk,
                    dst->ne[1], dst->ne[2], dim, stream);
            }
        } وإلا {
            ثابت size_t size0 = ggml_nbytes(src0);
            ثابت size_t size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->حافظه_نقل(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->حافظه_نقل((حرف *) dst_d + size0, src1_d, size1)));
        }
    } وإلا {
        concat_T_sycl_non_cont<بلاک_q4_1>(
            stream, (ثابت حرف *) src0->معلومات, (ثابت حرف *) src1->معلومات,
            (حرف *) dst->معلومات,
            ne00_blk, src0->ne[1], src0->ne[2], src0->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->ne[0] / QK4_1, src1->ne[1], src1->ne[2], src1->ne[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            ne0_blk, dst->ne[1], dst->ne[2], dst->ne[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

ساكن فراغ concat_impl_q5_0_sycl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ثابت ggml_tensor * src0   = dst->src[0];
    ثابت ggml_tensor * src1   = dst->src[1];
    کتار_ptr           stream = ctx.stream();

    ثابت int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_Q5_0);
    GGML_ASSERT(src1->type == GGML_TYPE_Q5_0);
    GGML_ASSERT(dst->type == GGML_TYPE_Q5_0);
    GGML_ASSERT(src0->ne[0] % QK5_0 == 0);
    GGML_ASSERT(src1->ne[0] % QK5_0 == 0);
    GGML_ASSERT(dst->ne[0] % QK5_0 == 0);

    ثابت صحيح ne00_blk = src0->ne[0] / QK5_0;
    ثابت صحيح ne0_blk  = dst->ne[0] / QK5_0;

    إذا (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        ثابت بلاک_q5_0 * src0_d = (ثابت بلاک_q5_0 *) src0->معلومات;
        ثابت بلاک_q5_0 * src1_d = (ثابت بلاک_q5_0 *) src1->معلومات;
        بلاک_q5_0 * dst_d        = (بلاک_q5_0 *) dst->معلومات;
        ثابت size_t type_اندازه    = حجم(بلاک_q5_0);

        إذا (dim != 3) {
            لكل (صحيح i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<بلاک_q5_0>(
                    src0_d + i3 * (src0->nb[3] / type_اندازه),
                    src1_d + i3 * (src1->nb[3] / type_اندازه),
                    dst_d + i3 * (dst->nb[3] / type_اندازه),
                    ne00_blk, src0->ne[1], src0->ne[2], ne0_blk,
                    dst->ne[1], dst->ne[2], dim, stream);
            }
        } وإلا {
            ثابت size_t size0 = ggml_nbytes(src0);
            ثابت size_t size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->حافظه_نقل(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->حافظه_نقل((حرف *) dst_d + size0, src1_d, size1)));
        }
    } وإلا {
        concat_T_sycl_non_cont<بلاک_q5_0>(
            stream, (ثابت حرف *) src0->معلومات, (ثابت حرف *) src1->معلومات,
            (حرف *) dst->معلومات,
            ne00_blk, src0->ne[1], src0->ne[2], src0->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->ne[0] / QK5_0, src1->ne[1], src1->ne[2], src1->ne[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            ne0_blk, dst->ne[1], dst->ne[2], dst->ne[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

ساكن فراغ concat_impl_q5_1_sycl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ثابت ggml_tensor * src0   = dst->src[0];
    ثابت ggml_tensor * src1   = dst->src[1];
    کتار_ptr           stream = ctx.stream();

    ثابت int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_Q5_1);
    GGML_ASSERT(src1->type == GGML_TYPE_Q5_1);
    GGML_ASSERT(dst->type == GGML_TYPE_Q5_1);
    GGML_ASSERT(src0->ne[0] % QK5_1 == 0);
    GGML_ASSERT(src1->ne[0] % QK5_1 == 0);
    GGML_ASSERT(dst->ne[0] % QK5_1 == 0);

    ثابت صحيح ne00_blk = src0->ne[0] / QK5_1;
    ثابت صحيح ne0_blk  = dst->ne[0] / QK5_1;

    إذا (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        ثابت بلاک_q5_1 * src0_d = (ثابت بلاک_q5_1 *) src0->معلومات;
        ثابت بلاک_q5_1 * src1_d = (ثابت بلاک_q5_1 *) src1->معلومات;
        بلاک_q5_1 * dst_d        = (بلاک_q5_1 *) dst->معلومات;
        ثابت size_t type_اندازه    = حجم(بلاک_q5_1);

        إذا (dim != 3) {
            لكل (صحيح i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<بلاک_q5_1>(
                    src0_d + i3 * (src0->nb[3] / type_اندازه),
                    src1_d + i3 * (src1->nb[3] / type_اندازه),
                    dst_d + i3 * (dst->nb[3] / type_اندازه),
                    ne00_blk, src0->ne[1], src0->ne[2], ne0_blk,
                    dst->ne[1], dst->ne[2], dim, stream);
            }
        } وإلا {
            ثابت size_t size0 = ggml_nbytes(src0);
            ثابت size_t size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->حافظه_نقل(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->حافظه_نقل((حرف *) dst_d + size0, src1_d, size1)));
        }
    } وإلا {
        concat_T_sycl_non_cont<بلاک_q5_1>(
            stream, (ثابت حرف *) src0->معلومات, (ثابت حرف *) src1->معلومات,
            (حرف *) dst->معلومات,
            ne00_blk, src0->ne[1], src0->ne[2], src0->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->ne[0] / QK5_1, src1->ne[1], src1->ne[2], src1->ne[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            ne0_blk, dst->ne[1], dst->ne[2], dst->ne[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

ساكن فراغ concat_impl_q8_0_sycl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ثابت ggml_tensor * src0   = dst->src[0];
    ثابت ggml_tensor * src1   = dst->src[1];
    کتار_ptr           stream = ctx.stream();

    ثابت int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(src1->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(dst->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(src0->ne[0] % QK8_0 == 0);
    GGML_ASSERT(src1->ne[0] % QK8_0 == 0);
    GGML_ASSERT(dst->ne[0] % QK8_0 == 0);

    ثابت صحيح ne00_blk = src0->ne[0] / QK8_0;
    ثابت صحيح ne0_blk  = dst->ne[0] / QK8_0;

    إذا (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        ثابت بلاک_q8_0 * src0_d = (ثابت بلاک_q8_0 *) src0->معلومات;
        ثابت بلاک_q8_0 * src1_d = (ثابت بلاک_q8_0 *) src1->معلومات;
        بلاک_q8_0 * dst_d        = (بلاک_q8_0 *) dst->معلومات;
        ثابت size_t type_اندازه    = حجم(بلاک_q8_0);

        إذا (dim != 3) {
            لكل (صحيح i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<بلاک_q8_0>(
                    src0_d + i3 * (src0->nb[3] / type_اندازه),
                    src1_d + i3 * (src1->nb[3] / type_اندازه),
                    dst_d + i3 * (dst->nb[3] / type_اندازه),
                    ne00_blk, src0->ne[1], src0->ne[2], ne0_blk,
                    dst->ne[1], dst->ne[2], dim, stream);
            }
        } وإلا {
            ثابت size_t size0 = ggml_nbytes(src0);
            ثابت size_t size1 = ggml_nbytes(src1);
            SYCL_CHECK(CHECK_TRY_ERROR(stream->حافظه_نقل(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->حافظه_نقل((حرف *) dst_d + size0, src1_d, size1)));
        }
    } وإلا {
        concat_T_sycl_non_cont<بلاک_q8_0>(
            stream, (ثابت حرف *) src0->معلومات, (ثابت حرف *) src1->معلومات,
            (حرف *) dst->معلومات,
            ne00_blk, src0->ne[1], src0->ne[2], src0->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->ne[0] / QK8_0, src1->ne[1], src1->ne[2], src1->ne[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            ne0_blk, dst->ne[1], dst->ne[2], dst->ne[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

فراغ ggml_sycl_op_concat(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    بدل (dst->type) {
    حالة GGML_TYPE_F32:
        concat_impl_sycl<عائم>(ctx, dst);
        اكسر;
    حالة GGML_TYPE_F16:
        concat_impl_sycl<sycl::half>(ctx, dst);
        اكسر;
#که_تعریف_شوی GGML_SYCL_HAS_BF16
    حالة GGML_TYPE_BF16:
        concat_impl_sycl<sycl::ext::oneapi::bfloat16>(ctx, dst);
        اكسر;
#د_که_پای
    حالة GGML_TYPE_I32:
        concat_impl_sycl<int32_t>(ctx, dst);
        اكسر;
    حالة GGML_TYPE_I16:
        concat_impl_sycl<int16_t>(ctx, dst);
        اكسر;
    حالة GGML_TYPE_I64:
        concat_impl_sycl<int64_t>(ctx, dst);
        اكسر;
    حالة GGML_TYPE_I8:
        concat_impl_sycl<int8_t>(ctx, dst);
        اكسر;
    حالة GGML_TYPE_Q4_0:
        concat_impl_q4_0_sycl(ctx, dst);
        اكسر;
    حالة GGML_TYPE_Q4_1:
        concat_impl_q4_1_sycl(ctx, dst);
        اكسر;
    حالة GGML_TYPE_Q5_0:
        concat_impl_q5_0_sycl(ctx, dst);
        اكسر;
    حالة GGML_TYPE_Q5_1:
        concat_impl_q5_1_sycl(ctx, dst);
        اكسر;
    حالة GGML_TYPE_Q8_0:
        concat_impl_q8_0_sycl(ctx, dst);
        اكسر;
    افتراضي:
        fprintf(stderr, "%s: unsupported types: dst: %s\n", __func__, ggml_type_name(dst->type));
        GGML_ASSERT(خطأ);
    }
}
