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

#अन्तर्भाव "ggml.h"

#अन्तर्भाव "concat.hpp"

स्थिर ओळीत माप_प्रकार elem_size(ggml_type t) {
    परत ggml_type_size(t) / ggml_blck_size(t);
}

ढाँचा <typename T>
स्थिर शून्य concat_T_dim0(स्थिरांक T *x, स्थिरांक T *y, T *dst,
                            स्थिरांक पूर्णांक ne0, स्थिरांक पूर्णांक ne00,
                            स्थिरांक sycl::nd_item<3> &item_ct1) {
  पूर्णांक nidx = item_ct1.get_local_id(2) +
             item_ct1.get_group(2) * item_ct1.get_local_range(2);
  जर (nidx >= ne0) {
    परत;
  }
  // operation
  पूर्णांक offset_dst = nidx + item_ct1.get_group(1) * ne0 +
                   item_ct1.get_group(0) * ne0 * item_ct1.get_group_range(1);
  जर (nidx < ne00) { // src0
    पूर्णांक offset_src = nidx + item_ct1.get_group(1) * ne00 +
                     item_ct1.get_group(0) * ne00 * item_ct1.get_group_range(1);
    dst[offset_dst] = x[offset_src];
  } अन्यथा {
    पूर्णांक offset_src =
        nidx - ne00 + item_ct1.get_group(1) * (ne0 - ne00) +
        item_ct1.get_group(0) * (ne0 - ne00) * item_ct1.get_group_range(1);
    dst[offset_dst] = y[offset_src];
  }
}

ढाँचा <typename T>
स्थिर शून्य concat_T_dim1(स्थिरांक T *x, स्थिरांक T *y, T *dst,
                            स्थिरांक पूर्णांक ne0, स्थिरांक पूर्णांक ne01,
                            स्थिरांक sycl::nd_item<3> &item_ct1) {
  पूर्णांक nidx = item_ct1.get_local_id(2) +
             item_ct1.get_group(2) * item_ct1.get_local_range(2);
  जर (nidx >= ne0) {
    परत;
  }
  // operation
  पूर्णांक offset_dst = nidx + item_ct1.get_group(1) * ne0 +
                   item_ct1.get_group(0) * ne0 * item_ct1.get_group_range(1);
  जर (item_ct1.get_group(1) < (माप_प्रकार) ne01) { // src0
    पूर्णांक offset_src =
        nidx + item_ct1.get_group(1) * ne0 + item_ct1.get_group(0) * ne0 * ne01;
    dst[offset_dst] = x[offset_src];
  } अन्यथा {
    पूर्णांक offset_src =
        nidx + (item_ct1.get_group(1) - ne01) * ne0 +
        item_ct1.get_group(0) * ne0 * (item_ct1.get_group_range(1) - ne01);
    dst[offset_dst] = y[offset_src];
  }
}

ढाँचा <typename T>
स्थिर शून्य concat_T_dim2(स्थिरांक T *x, स्थिरांक T *y, T *dst,
                            स्थिरांक पूर्णांक ne0, स्थिरांक पूर्णांक ne02,
                            स्थिरांक sycl::nd_item<3> &item_ct1) {
  पूर्णांक nidx = item_ct1.get_local_id(2) +
             item_ct1.get_group(2) * item_ct1.get_local_range(2);
  जर (nidx >= ne0) {
    परत;
  }
  // operation
  पूर्णांक offset_dst = nidx + item_ct1.get_group(1) * ne0 +
                   item_ct1.get_group(0) * ne0 * item_ct1.get_group_range(1);
  जर (item_ct1.get_group(0) < (माप_प्रकार) ne02) { // src0
    पूर्णांक offset_src = nidx + item_ct1.get_group(1) * ne0 +
                     item_ct1.get_group(0) * ne0 * item_ct1.get_group_range(1);
    dst[offset_dst] = x[offset_src];
  } अन्यथा {
    पूर्णांक offset_src =
        nidx + item_ct1.get_group(1) * ne0 +
        (item_ct1.get_group(0) - ne02) * ne0 * item_ct1.get_group_range(1);
    dst[offset_dst] = y[offset_src];
  }
}

ढाँचा <typename T>
स्थिर शून्य concat_T_sycl(स्थिरांक T *x, स्थिरांक T *y, T *dst,
                            पूर्णांक ne00, पूर्णांक ne01, पूर्णांक ne02, पूर्णांक ne0, पूर्णांक ne1,
                            पूर्णांक ne2, पूर्णांक dim, queue_ptr stream) {
  पूर्णांक अंक_blocks = (ne0 + SYCL_CONCAT_BLOCK_SIZE - 1) / SYCL_CONCAT_BLOCK_SIZE;
  sycl::range<3> gridDim(ne2, ne1, अंक_blocks);
  निवड (dim) {
  स्थिती 0:
      stream->parallel_for(sycl::nd_range<3>(gridDim * sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE),
                                          sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE)),
                        [=](sycl::nd_item<3> item_ct1) { concat_T_dim0<T>(x, y, dst, ne0, ne00, item_ct1); });
      थांबा;
  स्थिती 1:
      stream->parallel_for(sycl::nd_range<3>(gridDim * sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE),
                                          sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE)),
                        [=](sycl::nd_item<3> item_ct1) { concat_T_dim1<T>(x, y, dst, ne0, ne01, item_ct1); });
      थांबा;
  // dim >=2 will be dispatched to the default path
  पूर्वनिर्धारित:
      stream->parallel_for(sycl::nd_range<3>(gridDim * sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE),
                                          sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE)),
                        [=](sycl::nd_item<3> item_ct1) { concat_T_dim2<T>(x, y, dst, ne0, ne02, item_ct1); });
      थांबा;
  }
}

// non-contiguous kernel (slow)
ढाँचा<typename T>
स्थिर शून्य concat_T_sycl_non_cont(
    queue_ptr stream, स्थिरांक अक्षर *src0, स्थिरांक अक्षर *src1, अक्षर *dst,
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
  स्थिरांक int64_t ne0_pad   = GGML_PAD(ne0, WARP_SIZE);
  स्थिरांक int64_t block_ne0 = ne0_pad < SYCL_CONCAT_BLOCK_SIZE ? ne0_pad : (int64_t) SYCL_CONCAT_BLOCK_SIZE;
  sycl::range<3> blockDim(1, 1, block_ne0);

  stream->parallel_for(sycl::nd_range<3>(gridDim * blockDim, blockDim), [=](sycl::nd_item<3> item_ct1) {
      int64_t i3 = item_ct1.get_group(0);
      int64_t i2 = item_ct1.get_group(1);
      int64_t i1 = item_ct1.get_group(2);

      int64_t o[4] = { 0, 0, 0, 0 };
      o[dim]       = dim == 0 ? ne00 : (dim == 1 ? ne01 : (dim == 2 ? ne02 : ne03));

      स्थिरांक T * x;

      साठी (पूर्णांक i0 = item_ct1.get_local_id(2); i0 < ne0; i0 += item_ct1.get_local_range(2)) {
          जर (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
              x = (स्थिरांक T *) (src0 + (i3) *nb03 + (i2) *nb02 + (i1) *nb01 + (i0) *nb00);
          } अन्यथा {
              x = (स्थिरांक T *) (src1 + (i3 - o[3]) * nb13 + (i2 - o[2]) * nb12 + (i1 - o[1]) * nb11 +
                                   (i0 - o[0]) * nb10);
          }

          T *y = (T *)(dst + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

          *y = *x;
      }
  });
}

ढाँचा <typename T>
शून्य concat_impl_sycl(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    स्थिरांक ggml_tensor *  src0   = dst->src[0];
    स्थिरांक ggml_tensor *  src1   = dst->src[1];
    queue_ptr            stream = ctx.stream();

    स्थिरांक int32_t dim = ((int32_t *) dst->op_params)[0];

    जर (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        स्थिरांक T * src0_d = (स्थिरांक T *) src0->data;
        स्थिरांक T * src1_d = (स्थिरांक T *) src1->data;
        T * dst_d = (T *) dst->data;
        माप_प्रकार type_size = elem_size(dst->type);
        जर (dim != 3) {
            साठी (पूर्णांक i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<T>(src0_d + i3 * (src0->nb[3] / type_size), src1_d + i3 * (src1->nb[3] / type_size),
                                dst_d + i3 * (dst->nb[3] / type_size), src0->ne[0], src0->ne[1], src0->ne[2], dst->ne[0],
                                dst->ne[1], dst->ne[2], dim, stream);
            }
        } अन्यथा {
            स्थिरांक माप_प्रकार size0 = ggml_nbytes(src0);
            स्थिरांक माप_प्रकार size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->स_नकल(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->स_नकल(dst_d + size0 / type_size, src1_d, size1)));
        }
    } अन्यथा {
        concat_T_sycl_non_cont<T>(stream, (स्थिरांक अक्षर *) src0->data, (स्थिरांक अक्षर *) src1->data, (अक्षर *) dst->data,
                                 src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src0->nb[0], src0->nb[1],
                                 src0->nb[2], src0->nb[3], src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                                 src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3], dst->ne[0], dst->ne[1], dst->ne[2],
                                 dst->ne[3], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

स्थिर शून्य concat_impl_q4_0_sycl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    स्थिरांक ggml_tensor * src0   = dst->src[0];
    स्थिरांक ggml_tensor * src1   = dst->src[1];
    queue_ptr           stream = ctx.stream();

    स्थिरांक int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(src1->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(dst->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(src0->ne[0] % QK4_0 == 0);
    GGML_ASSERT(src1->ne[0] % QK4_0 == 0);
    GGML_ASSERT(dst->ne[0] % QK4_0 == 0);

    स्थिरांक पूर्णांक ne00_blk = src0->ne[0] / QK4_0;
    स्थिरांक पूर्णांक ne0_blk  = dst->ne[0] / QK4_0;

    जर (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        स्थिरांक block_q4_0 * src0_d = (स्थिरांक block_q4_0 *) src0->data;
        स्थिरांक block_q4_0 * src1_d = (स्थिरांक block_q4_0 *) src1->data;
        block_q4_0 * dst_d        = (block_q4_0 *) dst->data;
        स्थिरांक माप_प्रकार type_size    = आकार(block_q4_0);

        जर (dim != 3) {
            साठी (पूर्णांक i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<block_q4_0>(
                    src0_d + i3 * (src0->nb[3] / type_size),
                    src1_d + i3 * (src1->nb[3] / type_size),
                    dst_d + i3 * (dst->nb[3] / type_size),
                    ne00_blk, src0->ne[1], src0->ne[2], ne0_blk,
                    dst->ne[1], dst->ne[2], dim, stream);
            }
        } अन्यथा {
            स्थिरांक माप_प्रकार size0 = ggml_nbytes(src0);
            स्थिरांक माप_प्रकार size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->स_नकल(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->स_नकल((अक्षर *) dst_d + size0, src1_d, size1)));
        }
    } अन्यथा {
        concat_T_sycl_non_cont<block_q4_0>(
            stream, (स्थिरांक अक्षर *) src0->data, (स्थिरांक अक्षर *) src1->data,
            (अक्षर *) dst->data,
            ne00_blk, src0->ne[1], src0->ne[2], src0->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->ne[0] / QK4_0, src1->ne[1], src1->ne[2], src1->ne[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            ne0_blk, dst->ne[1], dst->ne[2], dst->ne[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

स्थिर शून्य concat_impl_q4_1_sycl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    स्थिरांक ggml_tensor * src0   = dst->src[0];
    स्थिरांक ggml_tensor * src1   = dst->src[1];
    queue_ptr           stream = ctx.stream();

    स्थिरांक int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(src1->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(dst->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(src0->ne[0] % QK4_1 == 0);
    GGML_ASSERT(src1->ne[0] % QK4_1 == 0);
    GGML_ASSERT(dst->ne[0] % QK4_1 == 0);

    स्थिरांक पूर्णांक ne00_blk = src0->ne[0] / QK4_1;
    स्थिरांक पूर्णांक ne0_blk  = dst->ne[0] / QK4_1;

    जर (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        स्थिरांक block_q4_1 * src0_d = (स्थिरांक block_q4_1 *) src0->data;
        स्थिरांक block_q4_1 * src1_d = (स्थिरांक block_q4_1 *) src1->data;
        block_q4_1 * dst_d        = (block_q4_1 *) dst->data;
        स्थिरांक माप_प्रकार type_size    = आकार(block_q4_1);

        जर (dim != 3) {
            साठी (पूर्णांक i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<block_q4_1>(
                    src0_d + i3 * (src0->nb[3] / type_size),
                    src1_d + i3 * (src1->nb[3] / type_size),
                    dst_d + i3 * (dst->nb[3] / type_size),
                    ne00_blk, src0->ne[1], src0->ne[2], ne0_blk,
                    dst->ne[1], dst->ne[2], dim, stream);
            }
        } अन्यथा {
            स्थिरांक माप_प्रकार size0 = ggml_nbytes(src0);
            स्थिरांक माप_प्रकार size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->स_नकल(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->स_नकल((अक्षर *) dst_d + size0, src1_d, size1)));
        }
    } अन्यथा {
        concat_T_sycl_non_cont<block_q4_1>(
            stream, (स्थिरांक अक्षर *) src0->data, (स्थिरांक अक्षर *) src1->data,
            (अक्षर *) dst->data,
            ne00_blk, src0->ne[1], src0->ne[2], src0->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->ne[0] / QK4_1, src1->ne[1], src1->ne[2], src1->ne[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            ne0_blk, dst->ne[1], dst->ne[2], dst->ne[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

स्थिर शून्य concat_impl_q5_0_sycl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    स्थिरांक ggml_tensor * src0   = dst->src[0];
    स्थिरांक ggml_tensor * src1   = dst->src[1];
    queue_ptr           stream = ctx.stream();

    स्थिरांक int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_Q5_0);
    GGML_ASSERT(src1->type == GGML_TYPE_Q5_0);
    GGML_ASSERT(dst->type == GGML_TYPE_Q5_0);
    GGML_ASSERT(src0->ne[0] % QK5_0 == 0);
    GGML_ASSERT(src1->ne[0] % QK5_0 == 0);
    GGML_ASSERT(dst->ne[0] % QK5_0 == 0);

    स्थिरांक पूर्णांक ne00_blk = src0->ne[0] / QK5_0;
    स्थिरांक पूर्णांक ne0_blk  = dst->ne[0] / QK5_0;

    जर (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        स्थिरांक block_q5_0 * src0_d = (स्थिरांक block_q5_0 *) src0->data;
        स्थिरांक block_q5_0 * src1_d = (स्थिरांक block_q5_0 *) src1->data;
        block_q5_0 * dst_d        = (block_q5_0 *) dst->data;
        स्थिरांक माप_प्रकार type_size    = आकार(block_q5_0);

        जर (dim != 3) {
            साठी (पूर्णांक i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<block_q5_0>(
                    src0_d + i3 * (src0->nb[3] / type_size),
                    src1_d + i3 * (src1->nb[3] / type_size),
                    dst_d + i3 * (dst->nb[3] / type_size),
                    ne00_blk, src0->ne[1], src0->ne[2], ne0_blk,
                    dst->ne[1], dst->ne[2], dim, stream);
            }
        } अन्यथा {
            स्थिरांक माप_प्रकार size0 = ggml_nbytes(src0);
            स्थिरांक माप_प्रकार size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->स_नकल(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->स_नकल((अक्षर *) dst_d + size0, src1_d, size1)));
        }
    } अन्यथा {
        concat_T_sycl_non_cont<block_q5_0>(
            stream, (स्थिरांक अक्षर *) src0->data, (स्थिरांक अक्षर *) src1->data,
            (अक्षर *) dst->data,
            ne00_blk, src0->ne[1], src0->ne[2], src0->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->ne[0] / QK5_0, src1->ne[1], src1->ne[2], src1->ne[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            ne0_blk, dst->ne[1], dst->ne[2], dst->ne[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

स्थिर शून्य concat_impl_q5_1_sycl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    स्थिरांक ggml_tensor * src0   = dst->src[0];
    स्थिरांक ggml_tensor * src1   = dst->src[1];
    queue_ptr           stream = ctx.stream();

    स्थिरांक int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_Q5_1);
    GGML_ASSERT(src1->type == GGML_TYPE_Q5_1);
    GGML_ASSERT(dst->type == GGML_TYPE_Q5_1);
    GGML_ASSERT(src0->ne[0] % QK5_1 == 0);
    GGML_ASSERT(src1->ne[0] % QK5_1 == 0);
    GGML_ASSERT(dst->ne[0] % QK5_1 == 0);

    स्थिरांक पूर्णांक ne00_blk = src0->ne[0] / QK5_1;
    स्थिरांक पूर्णांक ne0_blk  = dst->ne[0] / QK5_1;

    जर (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        स्थिरांक block_q5_1 * src0_d = (स्थिरांक block_q5_1 *) src0->data;
        स्थिरांक block_q5_1 * src1_d = (स्थिरांक block_q5_1 *) src1->data;
        block_q5_1 * dst_d        = (block_q5_1 *) dst->data;
        स्थिरांक माप_प्रकार type_size    = आकार(block_q5_1);

        जर (dim != 3) {
            साठी (पूर्णांक i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<block_q5_1>(
                    src0_d + i3 * (src0->nb[3] / type_size),
                    src1_d + i3 * (src1->nb[3] / type_size),
                    dst_d + i3 * (dst->nb[3] / type_size),
                    ne00_blk, src0->ne[1], src0->ne[2], ne0_blk,
                    dst->ne[1], dst->ne[2], dim, stream);
            }
        } अन्यथा {
            स्थिरांक माप_प्रकार size0 = ggml_nbytes(src0);
            स्थिरांक माप_प्रकार size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->स_नकल(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->स_नकल((अक्षर *) dst_d + size0, src1_d, size1)));
        }
    } अन्यथा {
        concat_T_sycl_non_cont<block_q5_1>(
            stream, (स्थिरांक अक्षर *) src0->data, (स्थिरांक अक्षर *) src1->data,
            (अक्षर *) dst->data,
            ne00_blk, src0->ne[1], src0->ne[2], src0->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->ne[0] / QK5_1, src1->ne[1], src1->ne[2], src1->ne[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            ne0_blk, dst->ne[1], dst->ne[2], dst->ne[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

स्थिर शून्य concat_impl_q8_0_sycl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    स्थिरांक ggml_tensor * src0   = dst->src[0];
    स्थिरांक ggml_tensor * src1   = dst->src[1];
    queue_ptr           stream = ctx.stream();

    स्थिरांक int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(src1->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(dst->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(src0->ne[0] % QK8_0 == 0);
    GGML_ASSERT(src1->ne[0] % QK8_0 == 0);
    GGML_ASSERT(dst->ne[0] % QK8_0 == 0);

    स्थिरांक पूर्णांक ne00_blk = src0->ne[0] / QK8_0;
    स्थिरांक पूर्णांक ne0_blk  = dst->ne[0] / QK8_0;

    जर (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        स्थिरांक block_q8_0 * src0_d = (स्थिरांक block_q8_0 *) src0->data;
        स्थिरांक block_q8_0 * src1_d = (स्थिरांक block_q8_0 *) src1->data;
        block_q8_0 * dst_d        = (block_q8_0 *) dst->data;
        स्थिरांक माप_प्रकार type_size    = आकार(block_q8_0);

        जर (dim != 3) {
            साठी (पूर्णांक i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<block_q8_0>(
                    src0_d + i3 * (src0->nb[3] / type_size),
                    src1_d + i3 * (src1->nb[3] / type_size),
                    dst_d + i3 * (dst->nb[3] / type_size),
                    ne00_blk, src0->ne[1], src0->ne[2], ne0_blk,
                    dst->ne[1], dst->ne[2], dim, stream);
            }
        } अन्यथा {
            स्थिरांक माप_प्रकार size0 = ggml_nbytes(src0);
            स्थिरांक माप_प्रकार size1 = ggml_nbytes(src1);
            SYCL_CHECK(CHECK_TRY_ERROR(stream->स_नकल(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->स_नकल((अक्षर *) dst_d + size0, src1_d, size1)));
        }
    } अन्यथा {
        concat_T_sycl_non_cont<block_q8_0>(
            stream, (स्थिरांक अक्षर *) src0->data, (स्थिरांक अक्षर *) src1->data,
            (अक्षर *) dst->data,
            ne00_blk, src0->ne[1], src0->ne[2], src0->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->ne[0] / QK8_0, src1->ne[1], src1->ne[2], src1->ne[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            ne0_blk, dst->ne[1], dst->ne[2], dst->ne[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

शून्य ggml_sycl_op_concat(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    निवड (dst->type) {
    स्थिती GGML_TYPE_F32:
        concat_impl_sycl<दशांश>(ctx, dst);
        थांबा;
    स्थिती GGML_TYPE_F16:
        concat_impl_sycl<sycl::half>(ctx, dst);
        थांबा;
#ifdef GGML_SYCL_HAS_BF16
    स्थिती GGML_TYPE_BF16:
        concat_impl_sycl<sycl::ext::oneapi::bfloat16>(ctx, dst);
        थांबा;
#endif
    स्थिती GGML_TYPE_I32:
        concat_impl_sycl<int32_t>(ctx, dst);
        थांबा;
    स्थिती GGML_TYPE_I16:
        concat_impl_sycl<int16_t>(ctx, dst);
        थांबा;
    स्थिती GGML_TYPE_I64:
        concat_impl_sycl<int64_t>(ctx, dst);
        थांबा;
    स्थिती GGML_TYPE_I8:
        concat_impl_sycl<int8_t>(ctx, dst);
        थांबा;
    स्थिती GGML_TYPE_Q4_0:
        concat_impl_q4_0_sycl(ctx, dst);
        थांबा;
    स्थिती GGML_TYPE_Q4_1:
        concat_impl_q4_1_sycl(ctx, dst);
        थांबा;
    स्थिती GGML_TYPE_Q5_0:
        concat_impl_q5_0_sycl(ctx, dst);
        थांबा;
    स्थिती GGML_TYPE_Q5_1:
        concat_impl_q5_1_sycl(ctx, dst);
        थांबा;
    स्थिती GGML_TYPE_Q8_0:
        concat_impl_q8_0_sycl(ctx, dst);
        थांबा;
    पूर्वनिर्धारित:
        ख_लिखो(मानक_त्रुटि, "%s: unsupported types: dst: %s\n", __func__, ggml_type_name(dst->type));
        GGML_ASSERT(असत्य);
    }
}
