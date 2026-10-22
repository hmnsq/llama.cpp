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

स्थिर अन्तःस्थ size_t elem_आकार(ggml_type t) {
    प्रत्यागम ggml_type_आकार(t) / ggml_blck_आकार(t);
}

प्रतिरूपम् <प्रकारनाम T>
स्थिर शून्य concat_T_dim0(ध्रुव T *x, ध्रुव T *y, T *dst,
                            ध्रुव पूर्णाङ्क ne0, ध्रुव पूर्णाङ्क ne00,
                            ध्रुव sycl::nd_item<3> &item_ct1) {
  पूर्णाङ्क nidx = item_ct1.get_local_id(2) +
             item_ct1.get_गणः(2) * item_ct1.get_local_range(2);
  यदि (nidx >= ne0) {
    प्रत्यागम;
  }
  // operation
  पूर्णाङ्क विस्थापनम्_dst = nidx + item_ct1.get_गणः(1) * ne0 +
                   item_ct1.get_गणः(0) * ne0 * item_ct1.get_गणः_range(1);
  यदि (nidx < ne00) { // src0
    पूर्णाङ्क विस्थापनम्_src = nidx + item_ct1.get_गणः(1) * ne00 +
                     item_ct1.get_गणः(0) * ne00 * item_ct1.get_गणः_range(1);
    dst[विस्थापनम्_dst] = x[विस्थापनम्_src];
  } अन्यथा {
    पूर्णाङ्क विस्थापनम्_src =
        nidx - ne00 + item_ct1.get_गणः(1) * (ne0 - ne00) +
        item_ct1.get_गणः(0) * (ne0 - ne00) * item_ct1.get_गणः_range(1);
    dst[विस्थापनम्_dst] = y[विस्थापनम्_src];
  }
}

प्रतिरूपम् <प्रकारनाम T>
स्थिर शून्य concat_T_dim1(ध्रुव T *x, ध्रुव T *y, T *dst,
                            ध्रुव पूर्णाङ्क ne0, ध्रुव पूर्णाङ्क ne01,
                            ध्रुव sycl::nd_item<3> &item_ct1) {
  पूर्णाङ्क nidx = item_ct1.get_local_id(2) +
             item_ct1.get_गणः(2) * item_ct1.get_local_range(2);
  यदि (nidx >= ne0) {
    प्रत्यागम;
  }
  // operation
  पूर्णाङ्क विस्थापनम्_dst = nidx + item_ct1.get_गणः(1) * ne0 +
                   item_ct1.get_गणः(0) * ne0 * item_ct1.get_गणः_range(1);
  यदि (item_ct1.get_गणः(1) < (size_t) ne01) { // src0
    पूर्णाङ्क विस्थापनम्_src =
        nidx + item_ct1.get_गणः(1) * ne0 + item_ct1.get_गणः(0) * ne0 * ne01;
    dst[विस्थापनम्_dst] = x[विस्थापनम्_src];
  } अन्यथा {
    पूर्णाङ्क विस्थापनम्_src =
        nidx + (item_ct1.get_गणः(1) - ne01) * ne0 +
        item_ct1.get_गणः(0) * ne0 * (item_ct1.get_गणः_range(1) - ne01);
    dst[विस्थापनम्_dst] = y[विस्थापनम्_src];
  }
}

प्रतिरूपम् <प्रकारनाम T>
स्थिर शून्य concat_T_dim2(ध्रुव T *x, ध्रुव T *y, T *dst,
                            ध्रुव पूर्णाङ्क ne0, ध्रुव पूर्णाङ्क ne02,
                            ध्रुव sycl::nd_item<3> &item_ct1) {
  पूर्णाङ्क nidx = item_ct1.get_local_id(2) +
             item_ct1.get_गणः(2) * item_ct1.get_local_range(2);
  यदि (nidx >= ne0) {
    प्रत्यागम;
  }
  // operation
  पूर्णाङ्क विस्थापनम्_dst = nidx + item_ct1.get_गणः(1) * ne0 +
                   item_ct1.get_गणः(0) * ne0 * item_ct1.get_गणः_range(1);
  यदि (item_ct1.get_गणः(0) < (size_t) ne02) { // src0
    पूर्णाङ्क विस्थापनम्_src = nidx + item_ct1.get_गणः(1) * ne0 +
                     item_ct1.get_गणः(0) * ne0 * item_ct1.get_गणः_range(1);
    dst[विस्थापनम्_dst] = x[विस्थापनम्_src];
  } अन्यथा {
    पूर्णाङ्क विस्थापनम्_src =
        nidx + item_ct1.get_गणः(1) * ne0 +
        (item_ct1.get_गणः(0) - ne02) * ne0 * item_ct1.get_गणः_range(1);
    dst[विस्थापनम्_dst] = y[विस्थापनम्_src];
  }
}

प्रतिरूपम् <प्रकारनाम T>
स्थिर शून्य concat_T_sycl(ध्रुव T *x, ध्रुव T *y, T *dst,
                            पूर्णाङ्क ne00, पूर्णाङ्क ne01, पूर्णाङ्क ne02, पूर्णाङ्क ne0, पूर्णाङ्क ne1,
                            पूर्णाङ्क ne2, पूर्णाङ्क dim, queue_ptr stream) {
  पूर्णाङ्क अङ्कः_खण्डानि = (ne0 + SYCL_CONCAT_खण्डम्_आकार - 1) / SYCL_CONCAT_खण्डम्_आकार;
  sycl::range<3> gridDim(ne2, ne1, अङ्कः_खण्डानि);
  विकल्प (dim) {
  पक्ष 0:
      stream->parallel_for(sycl::nd_range<3>(gridDim * sycl::range<3>(1, 1, SYCL_CONCAT_खण्डम्_आकार),
                                          sycl::range<3>(1, 1, SYCL_CONCAT_खण्डम्_आकार)),
                        [=](sycl::nd_item<3> item_ct1) { concat_T_dim0<T>(x, y, dst, ne0, ne00, item_ct1); });
      भङ्ग;
  पक्ष 1:
      stream->parallel_for(sycl::nd_range<3>(gridDim * sycl::range<3>(1, 1, SYCL_CONCAT_खण्डम्_आकार),
                                          sycl::range<3>(1, 1, SYCL_CONCAT_खण्डम्_आकार)),
                        [=](sycl::nd_item<3> item_ct1) { concat_T_dim1<T>(x, y, dst, ne0, ne01, item_ct1); });
      भङ्ग;
  // dim >=2 will be dispatched to the default path
  स्वभाव:
      stream->parallel_for(sycl::nd_range<3>(gridDim * sycl::range<3>(1, 1, SYCL_CONCAT_खण्डम्_आकार),
                                          sycl::range<3>(1, 1, SYCL_CONCAT_खण्डम्_आकार)),
                        [=](sycl::nd_item<3> item_ct1) { concat_T_dim2<T>(x, y, dst, ne0, ne02, item_ct1); });
      भङ्ग;
  }
}

// non-contiguous kernel (slow)
प्रतिरूपम्<प्रकारनाम T>
स्थिर शून्य concat_T_sycl_non_cont(
    queue_ptr stream, ध्रुव वर्ण *src0, ध्रुव वर्ण *src1, वर्ण *dst,
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
  ध्रुव int64_t ne0_pad   = GGML_PAD(ne0, WARP_आकार);
  ध्रुव int64_t खण्डम्_ne0 = ne0_pad < SYCL_CONCAT_खण्डम्_आकार ? ne0_pad : (int64_t) SYCL_CONCAT_खण्डम्_आकार;
  sycl::range<3> खण्डम्_Dim(1, 1, खण्डम्_ne0);

  stream->parallel_for(sycl::nd_range<3>(gridDim * खण्डम्_Dim, खण्डम्_Dim), [=](sycl::nd_item<3> item_ct1) {
      int64_t i3 = item_ct1.get_गणः(0);
      int64_t i2 = item_ct1.get_गणः(1);
      int64_t i1 = item_ct1.get_गणः(2);

      int64_t o[4] = { 0, 0, 0, 0 };
      o[dim]       = dim == 0 ? ne00 : (dim == 1 ? ne01 : (dim == 2 ? ne02 : ne03));

      ध्रुव T * x;

      प्रति (पूर्णाङ्क i0 = item_ct1.get_local_id(2); i0 < ne0; i0 += item_ct1.get_local_range(2)) {
          यदि (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
              x = (ध्रुव T *) (src0 + (i3) *nb03 + (i2) *nb02 + (i1) *nb01 + (i0) *nb00);
          } अन्यथा {
              x = (ध्रुव T *) (src1 + (i3 - o[3]) * nb13 + (i2 - o[2]) * nb12 + (i1 - o[1]) * nb11 +
                                   (i0 - o[0]) * nb10);
          }

          T *y = (T *)(dst + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

          *y = *x;
      }
  });
}

प्रतिरूपम् <प्रकारनाम T>
शून्य concat_impl_sycl(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ध्रुव ggml_tensor *  src0   = dst->src[0];
    ध्रुव ggml_tensor *  src1   = dst->src[1];
    queue_ptr            stream = ctx.stream();

    ध्रुव int32_t dim = ((int32_t *) dst->op_params)[0];

    यदि (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        ध्रुव T * src0_d = (ध्रुव T *) src0->दत्त;
        ध्रुव T * src1_d = (ध्रुव T *) src1->दत्त;
        T * dst_d = (T *) dst->दत्त;
        size_t type_आकार = elem_आकार(dst->type);
        यदि (dim != 3) {
            प्रति (पूर्णाङ्क i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<T>(src0_d + i3 * (src0->nb[3] / type_आकार), src1_d + i3 * (src1->nb[3] / type_आकार),
                                dst_d + i3 * (dst->nb[3] / type_आकार), src0->ne[0], src0->ne[1], src0->ne[2], dst->ne[0],
                                dst->ne[1], dst->ne[2], dim, stream);
            }
        } अन्यथा {
            ध्रुव size_t size0 = ggml_nbytes(src0);
            ध्रुव size_t size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->स्मृतिं_प्रतिलिख(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->स्मृतिं_प्रतिलिख(dst_d + size0 / type_आकार, src1_d, size1)));
        }
    } अन्यथा {
        concat_T_sycl_non_cont<T>(stream, (ध्रुव वर्ण *) src0->दत्त, (ध्रुव वर्ण *) src1->दत्त, (वर्ण *) dst->दत्त,
                                 src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src0->nb[0], src0->nb[1],
                                 src0->nb[2], src0->nb[3], src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                                 src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3], dst->ne[0], dst->ne[1], dst->ne[2],
                                 dst->ne[3], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

स्थिर शून्य concat_impl_q4_0_sycl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ध्रुव ggml_tensor * src0   = dst->src[0];
    ध्रुव ggml_tensor * src1   = dst->src[1];
    queue_ptr           stream = ctx.stream();

    ध्रुव int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(src1->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(dst->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(src0->ne[0] % QK4_0 == 0);
    GGML_ASSERT(src1->ne[0] % QK4_0 == 0);
    GGML_ASSERT(dst->ne[0] % QK4_0 == 0);

    ध्रुव पूर्णाङ्क ne00_blk = src0->ne[0] / QK4_0;
    ध्रुव पूर्णाङ्क ne0_blk  = dst->ne[0] / QK4_0;

    यदि (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        ध्रुव खण्डम्_q4_0 * src0_d = (ध्रुव खण्डम्_q4_0 *) src0->दत्त;
        ध्रुव खण्डम्_q4_0 * src1_d = (ध्रुव खण्डम्_q4_0 *) src1->दत्त;
        खण्डम्_q4_0 * dst_d        = (खण्डम्_q4_0 *) dst->दत्त;
        ध्रुव size_t type_आकार    = परिमाण(खण्डम्_q4_0);

        यदि (dim != 3) {
            प्रति (पूर्णाङ्क i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<खण्डम्_q4_0>(
                    src0_d + i3 * (src0->nb[3] / type_आकार),
                    src1_d + i3 * (src1->nb[3] / type_आकार),
                    dst_d + i3 * (dst->nb[3] / type_आकार),
                    ne00_blk, src0->ne[1], src0->ne[2], ne0_blk,
                    dst->ne[1], dst->ne[2], dim, stream);
            }
        } अन्यथा {
            ध्रुव size_t size0 = ggml_nbytes(src0);
            ध्रुव size_t size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->स्मृतिं_प्रतिलिख(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->स्मृतिं_प्रतिलिख((वर्ण *) dst_d + size0, src1_d, size1)));
        }
    } अन्यथा {
        concat_T_sycl_non_cont<खण्डम्_q4_0>(
            stream, (ध्रुव वर्ण *) src0->दत्त, (ध्रुव वर्ण *) src1->दत्त,
            (वर्ण *) dst->दत्त,
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
    ध्रुव ggml_tensor * src0   = dst->src[0];
    ध्रुव ggml_tensor * src1   = dst->src[1];
    queue_ptr           stream = ctx.stream();

    ध्रुव int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(src1->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(dst->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(src0->ne[0] % QK4_1 == 0);
    GGML_ASSERT(src1->ne[0] % QK4_1 == 0);
    GGML_ASSERT(dst->ne[0] % QK4_1 == 0);

    ध्रुव पूर्णाङ्क ne00_blk = src0->ne[0] / QK4_1;
    ध्रुव पूर्णाङ्क ne0_blk  = dst->ne[0] / QK4_1;

    यदि (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        ध्रुव खण्डम्_q4_1 * src0_d = (ध्रुव खण्डम्_q4_1 *) src0->दत्त;
        ध्रुव खण्डम्_q4_1 * src1_d = (ध्रुव खण्डम्_q4_1 *) src1->दत्त;
        खण्डम्_q4_1 * dst_d        = (खण्डम्_q4_1 *) dst->दत्त;
        ध्रुव size_t type_आकार    = परिमाण(खण्डम्_q4_1);

        यदि (dim != 3) {
            प्रति (पूर्णाङ्क i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<खण्डम्_q4_1>(
                    src0_d + i3 * (src0->nb[3] / type_आकार),
                    src1_d + i3 * (src1->nb[3] / type_आकार),
                    dst_d + i3 * (dst->nb[3] / type_आकार),
                    ne00_blk, src0->ne[1], src0->ne[2], ne0_blk,
                    dst->ne[1], dst->ne[2], dim, stream);
            }
        } अन्यथा {
            ध्रुव size_t size0 = ggml_nbytes(src0);
            ध्रुव size_t size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->स्मृतिं_प्रतिलिख(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->स्मृतिं_प्रतिलिख((वर्ण *) dst_d + size0, src1_d, size1)));
        }
    } अन्यथा {
        concat_T_sycl_non_cont<खण्डम्_q4_1>(
            stream, (ध्रुव वर्ण *) src0->दत्त, (ध्रुव वर्ण *) src1->दत्त,
            (वर्ण *) dst->दत्त,
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
    ध्रुव ggml_tensor * src0   = dst->src[0];
    ध्रुव ggml_tensor * src1   = dst->src[1];
    queue_ptr           stream = ctx.stream();

    ध्रुव int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_Q5_0);
    GGML_ASSERT(src1->type == GGML_TYPE_Q5_0);
    GGML_ASSERT(dst->type == GGML_TYPE_Q5_0);
    GGML_ASSERT(src0->ne[0] % QK5_0 == 0);
    GGML_ASSERT(src1->ne[0] % QK5_0 == 0);
    GGML_ASSERT(dst->ne[0] % QK5_0 == 0);

    ध्रुव पूर्णाङ्क ne00_blk = src0->ne[0] / QK5_0;
    ध्रुव पूर्णाङ्क ne0_blk  = dst->ne[0] / QK5_0;

    यदि (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        ध्रुव खण्डम्_q5_0 * src0_d = (ध्रुव खण्डम्_q5_0 *) src0->दत्त;
        ध्रुव खण्डम्_q5_0 * src1_d = (ध्रुव खण्डम्_q5_0 *) src1->दत्त;
        खण्डम्_q5_0 * dst_d        = (खण्डम्_q5_0 *) dst->दत्त;
        ध्रुव size_t type_आकार    = परिमाण(खण्डम्_q5_0);

        यदि (dim != 3) {
            प्रति (पूर्णाङ्क i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<खण्डम्_q5_0>(
                    src0_d + i3 * (src0->nb[3] / type_आकार),
                    src1_d + i3 * (src1->nb[3] / type_आकार),
                    dst_d + i3 * (dst->nb[3] / type_आकार),
                    ne00_blk, src0->ne[1], src0->ne[2], ne0_blk,
                    dst->ne[1], dst->ne[2], dim, stream);
            }
        } अन्यथा {
            ध्रुव size_t size0 = ggml_nbytes(src0);
            ध्रुव size_t size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->स्मृतिं_प्रतिलिख(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->स्मृतिं_प्रतिलिख((वर्ण *) dst_d + size0, src1_d, size1)));
        }
    } अन्यथा {
        concat_T_sycl_non_cont<खण्डम्_q5_0>(
            stream, (ध्रुव वर्ण *) src0->दत्त, (ध्रुव वर्ण *) src1->दत्त,
            (वर्ण *) dst->दत्त,
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
    ध्रुव ggml_tensor * src0   = dst->src[0];
    ध्रुव ggml_tensor * src1   = dst->src[1];
    queue_ptr           stream = ctx.stream();

    ध्रुव int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_Q5_1);
    GGML_ASSERT(src1->type == GGML_TYPE_Q5_1);
    GGML_ASSERT(dst->type == GGML_TYPE_Q5_1);
    GGML_ASSERT(src0->ne[0] % QK5_1 == 0);
    GGML_ASSERT(src1->ne[0] % QK5_1 == 0);
    GGML_ASSERT(dst->ne[0] % QK5_1 == 0);

    ध्रुव पूर्णाङ्क ne00_blk = src0->ne[0] / QK5_1;
    ध्रुव पूर्णाङ्क ne0_blk  = dst->ne[0] / QK5_1;

    यदि (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        ध्रुव खण्डम्_q5_1 * src0_d = (ध्रुव खण्डम्_q5_1 *) src0->दत्त;
        ध्रुव खण्डम्_q5_1 * src1_d = (ध्रुव खण्डम्_q5_1 *) src1->दत्त;
        खण्डम्_q5_1 * dst_d        = (खण्डम्_q5_1 *) dst->दत्त;
        ध्रुव size_t type_आकार    = परिमाण(खण्डम्_q5_1);

        यदि (dim != 3) {
            प्रति (पूर्णाङ्क i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<खण्डम्_q5_1>(
                    src0_d + i3 * (src0->nb[3] / type_आकार),
                    src1_d + i3 * (src1->nb[3] / type_आकार),
                    dst_d + i3 * (dst->nb[3] / type_आकार),
                    ne00_blk, src0->ne[1], src0->ne[2], ne0_blk,
                    dst->ne[1], dst->ne[2], dim, stream);
            }
        } अन्यथा {
            ध्रुव size_t size0 = ggml_nbytes(src0);
            ध्रुव size_t size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->स्मृतिं_प्रतिलिख(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->स्मृतिं_प्रतिलिख((वर्ण *) dst_d + size0, src1_d, size1)));
        }
    } अन्यथा {
        concat_T_sycl_non_cont<खण्डम्_q5_1>(
            stream, (ध्रुव वर्ण *) src0->दत्त, (ध्रुव वर्ण *) src1->दत्त,
            (वर्ण *) dst->दत्त,
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
    ध्रुव ggml_tensor * src0   = dst->src[0];
    ध्रुव ggml_tensor * src1   = dst->src[1];
    queue_ptr           stream = ctx.stream();

    ध्रुव int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(src1->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(dst->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(src0->ne[0] % QK8_0 == 0);
    GGML_ASSERT(src1->ne[0] % QK8_0 == 0);
    GGML_ASSERT(dst->ne[0] % QK8_0 == 0);

    ध्रुव पूर्णाङ्क ne00_blk = src0->ne[0] / QK8_0;
    ध्रुव पूर्णाङ्क ne0_blk  = dst->ne[0] / QK8_0;

    यदि (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        ध्रुव खण्डम्_q8_0 * src0_d = (ध्रुव खण्डम्_q8_0 *) src0->दत्त;
        ध्रुव खण्डम्_q8_0 * src1_d = (ध्रुव खण्डम्_q8_0 *) src1->दत्त;
        खण्डम्_q8_0 * dst_d        = (खण्डम्_q8_0 *) dst->दत्त;
        ध्रुव size_t type_आकार    = परिमाण(खण्डम्_q8_0);

        यदि (dim != 3) {
            प्रति (पूर्णाङ्क i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<खण्डम्_q8_0>(
                    src0_d + i3 * (src0->nb[3] / type_आकार),
                    src1_d + i3 * (src1->nb[3] / type_आकार),
                    dst_d + i3 * (dst->nb[3] / type_आकार),
                    ne00_blk, src0->ne[1], src0->ne[2], ne0_blk,
                    dst->ne[1], dst->ne[2], dim, stream);
            }
        } अन्यथा {
            ध्रुव size_t size0 = ggml_nbytes(src0);
            ध्रुव size_t size1 = ggml_nbytes(src1);
            SYCL_CHECK(CHECK_TRY_ERROR(stream->स्मृतिं_प्रतिलिख(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->स्मृतिं_प्रतिलिख((वर्ण *) dst_d + size0, src1_d, size1)));
        }
    } अन्यथा {
        concat_T_sycl_non_cont<खण्डम्_q8_0>(
            stream, (ध्रुव वर्ण *) src0->दत्त, (ध्रुव वर्ण *) src1->दत्त,
            (वर्ण *) dst->दत्त,
            ne00_blk, src0->ne[1], src0->ne[2], src0->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->ne[0] / QK8_0, src1->ne[1], src1->ne[2], src1->ne[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            ne0_blk, dst->ne[1], dst->ne[2], dst->ne[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

शून्य ggml_sycl_op_concat(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    विकल्प (dst->type) {
    पक्ष GGML_TYPE_F32:
        concat_impl_sycl<भिन्न>(ctx, dst);
        भङ्ग;
    पक्ष GGML_TYPE_F16:
        concat_impl_sycl<sycl::half>(ctx, dst);
        भङ्ग;
#यदि_परिभाषितम् GGML_SYCL_HAS_BF16
    पक्ष GGML_TYPE_BF16:
        concat_impl_sycl<sycl::ext::oneapi::bfloat16>(ctx, dst);
        भङ्ग;
#यदि_अन्तः
    पक्ष GGML_TYPE_I32:
        concat_impl_sycl<int32_t>(ctx, dst);
        भङ्ग;
    पक्ष GGML_TYPE_I16:
        concat_impl_sycl<int16_t>(ctx, dst);
        भङ्ग;
    पक्ष GGML_TYPE_I64:
        concat_impl_sycl<int64_t>(ctx, dst);
        भङ्ग;
    पक्ष GGML_TYPE_I8:
        concat_impl_sycl<int8_t>(ctx, dst);
        भङ्ग;
    पक्ष GGML_TYPE_Q4_0:
        concat_impl_q4_0_sycl(ctx, dst);
        भङ्ग;
    पक्ष GGML_TYPE_Q4_1:
        concat_impl_q4_1_sycl(ctx, dst);
        भङ्ग;
    पक्ष GGML_TYPE_Q5_0:
        concat_impl_q5_0_sycl(ctx, dst);
        भङ्ग;
    पक्ष GGML_TYPE_Q5_1:
        concat_impl_q5_1_sycl(ctx, dst);
        भङ्ग;
    पक्ष GGML_TYPE_Q8_0:
        concat_impl_q8_0_sycl(ctx, dst);
        भङ्ग;
    स्वभाव:
        fprintf(stderr, "%s: unsupported types: dst: %s\n", __func__, ggml_type_name(dst->type));
        GGML_ASSERT(असत्य);
    }
}
