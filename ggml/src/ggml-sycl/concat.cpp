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

#অন্তর্ভাৱ "ggml.h"

#অন্তর্ভাৱ "concat.hpp"

স্থির অন্তঃস্থ size_t elem_size(ggml_type t) {
    ঘূৰাই ggml_type_size(t) / ggml_blck_size(t);
}

template <typename T>
স্থির শূন্য concat_T_dim0(ধ্রুৱ T *x, ধ্রুৱ T *y, T *dst,
                            ধ্রুৱ পূর্ণাঙ্ক ne0, ধ্রুৱ পূর্ণাঙ্ক ne00,
                            ধ্রুৱ sycl::nd_item<3> &item_ct1) {
  পূর্ণাঙ্ক nidx = item_ct1.get_local_id(2) +
             item_ct1.get_group(2) * item_ct1.get_local_range(2);
  যদি (nidx >= ne0) {
    ঘূৰাই;
  }
  // operation
  পূর্ণাঙ্ক offset_dst = nidx + item_ct1.get_group(1) * ne0 +
                   item_ct1.get_group(0) * ne0 * item_ct1.get_group_range(1);
  যদি (nidx < ne00) { // src0
    পূর্ণাঙ্ক offset_src = nidx + item_ct1.get_group(1) * ne00 +
                     item_ct1.get_group(0) * ne00 * item_ct1.get_group_range(1);
    dst[offset_dst] = x[offset_src];
  } অন্যথা {
    পূর্ণাঙ্ক offset_src =
        nidx - ne00 + item_ct1.get_group(1) * (ne0 - ne00) +
        item_ct1.get_group(0) * (ne0 - ne00) * item_ct1.get_group_range(1);
    dst[offset_dst] = y[offset_src];
  }
}

template <typename T>
স্থির শূন্য concat_T_dim1(ধ্রুৱ T *x, ধ্রুৱ T *y, T *dst,
                            ধ্রুৱ পূর্ণাঙ্ক ne0, ধ্রুৱ পূর্ণাঙ্ক ne01,
                            ধ্রুৱ sycl::nd_item<3> &item_ct1) {
  পূর্ণাঙ্ক nidx = item_ct1.get_local_id(2) +
             item_ct1.get_group(2) * item_ct1.get_local_range(2);
  যদি (nidx >= ne0) {
    ঘূৰাই;
  }
  // operation
  পূর্ণাঙ্ক offset_dst = nidx + item_ct1.get_group(1) * ne0 +
                   item_ct1.get_group(0) * ne0 * item_ct1.get_group_range(1);
  যদি (item_ct1.get_group(1) < (size_t) ne01) { // src0
    পূর্ণাঙ্ক offset_src =
        nidx + item_ct1.get_group(1) * ne0 + item_ct1.get_group(0) * ne0 * ne01;
    dst[offset_dst] = x[offset_src];
  } অন্যথা {
    পূর্ণাঙ্ক offset_src =
        nidx + (item_ct1.get_group(1) - ne01) * ne0 +
        item_ct1.get_group(0) * ne0 * (item_ct1.get_group_range(1) - ne01);
    dst[offset_dst] = y[offset_src];
  }
}

template <typename T>
স্থির শূন্য concat_T_dim2(ধ্রুৱ T *x, ধ্রুৱ T *y, T *dst,
                            ধ্রুৱ পূর্ণাঙ্ক ne0, ধ্রুৱ পূর্ণাঙ্ক ne02,
                            ধ্রুৱ sycl::nd_item<3> &item_ct1) {
  পূর্ণাঙ্ক nidx = item_ct1.get_local_id(2) +
             item_ct1.get_group(2) * item_ct1.get_local_range(2);
  যদি (nidx >= ne0) {
    ঘূৰাই;
  }
  // operation
  পূর্ণাঙ্ক offset_dst = nidx + item_ct1.get_group(1) * ne0 +
                   item_ct1.get_group(0) * ne0 * item_ct1.get_group_range(1);
  যদি (item_ct1.get_group(0) < (size_t) ne02) { // src0
    পূর্ণাঙ্ক offset_src = nidx + item_ct1.get_group(1) * ne0 +
                     item_ct1.get_group(0) * ne0 * item_ct1.get_group_range(1);
    dst[offset_dst] = x[offset_src];
  } অন্যথা {
    পূর্ণাঙ্ক offset_src =
        nidx + item_ct1.get_group(1) * ne0 +
        (item_ct1.get_group(0) - ne02) * ne0 * item_ct1.get_group_range(1);
    dst[offset_dst] = y[offset_src];
  }
}

template <typename T>
স্থির শূন্য concat_T_sycl(ধ্রুৱ T *x, ধ্রুৱ T *y, T *dst,
                            পূর্ণাঙ্ক ne00, পূর্ণাঙ্ক ne01, পূর্ণাঙ্ক ne02, পূর্ণাঙ্ক ne0, পূর্ণাঙ্ক ne1,
                            পূর্ণাঙ্ক ne2, পূর্ণাঙ্ক dim, queue_ptr stream) {
  পূর্ণাঙ্ক অঙ্ক_blocks = (ne0 + SYCL_CONCAT_BLOCK_SIZE - 1) / SYCL_CONCAT_BLOCK_SIZE;
  sycl::range<3> gridDim(ne2, ne1, অঙ্ক_blocks);
  ৱিকল্প (dim) {
  পক্ষ 0:
      stream->parallel_for(sycl::nd_range<3>(gridDim * sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE),
                                          sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE)),
                        [=](sycl::nd_item<3> item_ct1) { concat_T_dim0<T>(x, y, dst, ne0, ne00, item_ct1); });
      ভঙ্গ;
  পক্ষ 1:
      stream->parallel_for(sycl::nd_range<3>(gridDim * sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE),
                                          sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE)),
                        [=](sycl::nd_item<3> item_ct1) { concat_T_dim1<T>(x, y, dst, ne0, ne01, item_ct1); });
      ভঙ্গ;
  // dim >=2 will be dispatched to the default path
  স্ৱভাৱ:
      stream->parallel_for(sycl::nd_range<3>(gridDim * sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE),
                                          sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE)),
                        [=](sycl::nd_item<3> item_ct1) { concat_T_dim2<T>(x, y, dst, ne0, ne02, item_ct1); });
      ভঙ্গ;
  }
}

// non-contiguous kernel (slow)
template<typename T>
স্থির শূন্য concat_T_sycl_non_cont(
    queue_ptr stream, ধ্রুৱ ৱর্ণ *src0, ধ্রুৱ ৱর্ণ *src1, ৱর্ণ *dst,
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
  ধ্রুৱ int64_t ne0_pad   = GGML_PAD(ne0, WARP_SIZE);
  ধ্রুৱ int64_t block_ne0 = ne0_pad < SYCL_CONCAT_BLOCK_SIZE ? ne0_pad : (int64_t) SYCL_CONCAT_BLOCK_SIZE;
  sycl::range<3> blockDim(1, 1, block_ne0);

  stream->parallel_for(sycl::nd_range<3>(gridDim * blockDim, blockDim), [=](sycl::nd_item<3> item_ct1) {
      int64_t i3 = item_ct1.get_group(0);
      int64_t i2 = item_ct1.get_group(1);
      int64_t i1 = item_ct1.get_group(2);

      int64_t o[4] = { 0, 0, 0, 0 };
      o[dim]       = dim == 0 ? ne00 : (dim == 1 ? ne01 : (dim == 2 ? ne02 : ne03));

      ধ্রুৱ T * x;

      বাবে (পূর্ণাঙ্ক i0 = item_ct1.get_local_id(2); i0 < ne0; i0 += item_ct1.get_local_range(2)) {
          যদি (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
              x = (ধ্রুৱ T *) (src0 + (i3) *nb03 + (i2) *nb02 + (i1) *nb01 + (i0) *nb00);
          } অন্যথা {
              x = (ধ্রুৱ T *) (src1 + (i3 - o[3]) * nb13 + (i2 - o[2]) * nb12 + (i1 - o[1]) * nb11 +
                                   (i0 - o[0]) * nb10);
          }

          T *y = (T *)(dst + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

          *y = *x;
      }
  });
}

template <typename T>
শূন্য concat_impl_sycl(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ধ্রুৱ ggml_tensor *  src0   = dst->src[0];
    ধ্রুৱ ggml_tensor *  src1   = dst->src[1];
    queue_ptr            stream = ctx.stream();

    ধ্রুৱ int32_t dim = ((int32_t *) dst->op_params)[0];

    যদি (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        ধ্রুৱ T * src0_d = (ধ্রুৱ T *) src0->data;
        ধ্রুৱ T * src1_d = (ধ্রুৱ T *) src1->data;
        T * dst_d = (T *) dst->data;
        size_t type_size = elem_size(dst->type);
        যদি (dim != 3) {
            বাবে (পূর্ণাঙ্ক i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<T>(src0_d + i3 * (src0->nb[3] / type_size), src1_d + i3 * (src1->nb[3] / type_size),
                                dst_d + i3 * (dst->nb[3] / type_size), src0->ne[0], src0->ne[1], src0->ne[2], dst->ne[0],
                                dst->ne[1], dst->ne[2], dim, stream);
            }
        } অন্যথা {
            ধ্রুৱ size_t size0 = ggml_nbytes(src0);
            ধ্রুৱ size_t size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(dst_d + size0 / type_size, src1_d, size1)));
        }
    } অন্যথা {
        concat_T_sycl_non_cont<T>(stream, (ধ্রুৱ ৱর্ণ *) src0->data, (ধ্রুৱ ৱর্ণ *) src1->data, (ৱর্ণ *) dst->data,
                                 src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src0->nb[0], src0->nb[1],
                                 src0->nb[2], src0->nb[3], src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                                 src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3], dst->ne[0], dst->ne[1], dst->ne[2],
                                 dst->ne[3], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

স্থির শূন্য concat_impl_q4_0_sycl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ধ্রুৱ ggml_tensor * src0   = dst->src[0];
    ধ্রুৱ ggml_tensor * src1   = dst->src[1];
    queue_ptr           stream = ctx.stream();

    ধ্রুৱ int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(src1->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(dst->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(src0->ne[0] % QK4_0 == 0);
    GGML_ASSERT(src1->ne[0] % QK4_0 == 0);
    GGML_ASSERT(dst->ne[0] % QK4_0 == 0);

    ধ্রুৱ পূর্ণাঙ্ক ne00_blk = src0->ne[0] / QK4_0;
    ধ্রুৱ পূর্ণাঙ্ক ne0_blk  = dst->ne[0] / QK4_0;

    যদি (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        ধ্রুৱ block_q4_0 * src0_d = (ধ্রুৱ block_q4_0 *) src0->data;
        ধ্রুৱ block_q4_0 * src1_d = (ধ্রুৱ block_q4_0 *) src1->data;
        block_q4_0 * dst_d        = (block_q4_0 *) dst->data;
        ধ্রুৱ size_t type_size    = পরিমাণ(block_q4_0);

        যদি (dim != 3) {
            বাবে (পূর্ণাঙ্ক i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<block_q4_0>(
                    src0_d + i3 * (src0->nb[3] / type_size),
                    src1_d + i3 * (src1->nb[3] / type_size),
                    dst_d + i3 * (dst->nb[3] / type_size),
                    ne00_blk, src0->ne[1], src0->ne[2], ne0_blk,
                    dst->ne[1], dst->ne[2], dim, stream);
            }
        } অন্যথা {
            ধ্রুৱ size_t size0 = ggml_nbytes(src0);
            ধ্রুৱ size_t size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy((ৱর্ণ *) dst_d + size0, src1_d, size1)));
        }
    } অন্যথা {
        concat_T_sycl_non_cont<block_q4_0>(
            stream, (ধ্রুৱ ৱর্ণ *) src0->data, (ধ্রুৱ ৱর্ণ *) src1->data,
            (ৱর্ণ *) dst->data,
            ne00_blk, src0->ne[1], src0->ne[2], src0->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->ne[0] / QK4_0, src1->ne[1], src1->ne[2], src1->ne[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            ne0_blk, dst->ne[1], dst->ne[2], dst->ne[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

স্থির শূন্য concat_impl_q4_1_sycl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ধ্রুৱ ggml_tensor * src0   = dst->src[0];
    ধ্রুৱ ggml_tensor * src1   = dst->src[1];
    queue_ptr           stream = ctx.stream();

    ধ্রুৱ int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(src1->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(dst->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(src0->ne[0] % QK4_1 == 0);
    GGML_ASSERT(src1->ne[0] % QK4_1 == 0);
    GGML_ASSERT(dst->ne[0] % QK4_1 == 0);

    ধ্রুৱ পূর্ণাঙ্ক ne00_blk = src0->ne[0] / QK4_1;
    ধ্রুৱ পূর্ণাঙ্ক ne0_blk  = dst->ne[0] / QK4_1;

    যদি (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        ধ্রুৱ block_q4_1 * src0_d = (ধ্রুৱ block_q4_1 *) src0->data;
        ধ্রুৱ block_q4_1 * src1_d = (ধ্রুৱ block_q4_1 *) src1->data;
        block_q4_1 * dst_d        = (block_q4_1 *) dst->data;
        ধ্রুৱ size_t type_size    = পরিমাণ(block_q4_1);

        যদি (dim != 3) {
            বাবে (পূর্ণাঙ্ক i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<block_q4_1>(
                    src0_d + i3 * (src0->nb[3] / type_size),
                    src1_d + i3 * (src1->nb[3] / type_size),
                    dst_d + i3 * (dst->nb[3] / type_size),
                    ne00_blk, src0->ne[1], src0->ne[2], ne0_blk,
                    dst->ne[1], dst->ne[2], dim, stream);
            }
        } অন্যথা {
            ধ্রুৱ size_t size0 = ggml_nbytes(src0);
            ধ্রুৱ size_t size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy((ৱর্ণ *) dst_d + size0, src1_d, size1)));
        }
    } অন্যথা {
        concat_T_sycl_non_cont<block_q4_1>(
            stream, (ধ্রুৱ ৱর্ণ *) src0->data, (ধ্রুৱ ৱর্ণ *) src1->data,
            (ৱর্ণ *) dst->data,
            ne00_blk, src0->ne[1], src0->ne[2], src0->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->ne[0] / QK4_1, src1->ne[1], src1->ne[2], src1->ne[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            ne0_blk, dst->ne[1], dst->ne[2], dst->ne[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

স্থির শূন্য concat_impl_q5_0_sycl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ধ্রুৱ ggml_tensor * src0   = dst->src[0];
    ধ্রুৱ ggml_tensor * src1   = dst->src[1];
    queue_ptr           stream = ctx.stream();

    ধ্রুৱ int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_Q5_0);
    GGML_ASSERT(src1->type == GGML_TYPE_Q5_0);
    GGML_ASSERT(dst->type == GGML_TYPE_Q5_0);
    GGML_ASSERT(src0->ne[0] % QK5_0 == 0);
    GGML_ASSERT(src1->ne[0] % QK5_0 == 0);
    GGML_ASSERT(dst->ne[0] % QK5_0 == 0);

    ধ্রুৱ পূর্ণাঙ্ক ne00_blk = src0->ne[0] / QK5_0;
    ধ্রুৱ পূর্ণাঙ্ক ne0_blk  = dst->ne[0] / QK5_0;

    যদি (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        ধ্রুৱ block_q5_0 * src0_d = (ধ্রুৱ block_q5_0 *) src0->data;
        ধ্রুৱ block_q5_0 * src1_d = (ধ্রুৱ block_q5_0 *) src1->data;
        block_q5_0 * dst_d        = (block_q5_0 *) dst->data;
        ধ্রুৱ size_t type_size    = পরিমাণ(block_q5_0);

        যদি (dim != 3) {
            বাবে (পূর্ণাঙ্ক i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<block_q5_0>(
                    src0_d + i3 * (src0->nb[3] / type_size),
                    src1_d + i3 * (src1->nb[3] / type_size),
                    dst_d + i3 * (dst->nb[3] / type_size),
                    ne00_blk, src0->ne[1], src0->ne[2], ne0_blk,
                    dst->ne[1], dst->ne[2], dim, stream);
            }
        } অন্যথা {
            ধ্রুৱ size_t size0 = ggml_nbytes(src0);
            ধ্রুৱ size_t size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy((ৱর্ণ *) dst_d + size0, src1_d, size1)));
        }
    } অন্যথা {
        concat_T_sycl_non_cont<block_q5_0>(
            stream, (ধ্রুৱ ৱর্ণ *) src0->data, (ধ্রুৱ ৱর্ণ *) src1->data,
            (ৱর্ণ *) dst->data,
            ne00_blk, src0->ne[1], src0->ne[2], src0->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->ne[0] / QK5_0, src1->ne[1], src1->ne[2], src1->ne[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            ne0_blk, dst->ne[1], dst->ne[2], dst->ne[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

স্থির শূন্য concat_impl_q5_1_sycl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ধ্রুৱ ggml_tensor * src0   = dst->src[0];
    ধ্রুৱ ggml_tensor * src1   = dst->src[1];
    queue_ptr           stream = ctx.stream();

    ধ্রুৱ int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_Q5_1);
    GGML_ASSERT(src1->type == GGML_TYPE_Q5_1);
    GGML_ASSERT(dst->type == GGML_TYPE_Q5_1);
    GGML_ASSERT(src0->ne[0] % QK5_1 == 0);
    GGML_ASSERT(src1->ne[0] % QK5_1 == 0);
    GGML_ASSERT(dst->ne[0] % QK5_1 == 0);

    ধ্রুৱ পূর্ণাঙ্ক ne00_blk = src0->ne[0] / QK5_1;
    ধ্রুৱ পূর্ণাঙ্ক ne0_blk  = dst->ne[0] / QK5_1;

    যদি (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        ধ্রুৱ block_q5_1 * src0_d = (ধ্রুৱ block_q5_1 *) src0->data;
        ধ্রুৱ block_q5_1 * src1_d = (ধ্রুৱ block_q5_1 *) src1->data;
        block_q5_1 * dst_d        = (block_q5_1 *) dst->data;
        ধ্রুৱ size_t type_size    = পরিমাণ(block_q5_1);

        যদি (dim != 3) {
            বাবে (পূর্ণাঙ্ক i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<block_q5_1>(
                    src0_d + i3 * (src0->nb[3] / type_size),
                    src1_d + i3 * (src1->nb[3] / type_size),
                    dst_d + i3 * (dst->nb[3] / type_size),
                    ne00_blk, src0->ne[1], src0->ne[2], ne0_blk,
                    dst->ne[1], dst->ne[2], dim, stream);
            }
        } অন্যথা {
            ধ্রুৱ size_t size0 = ggml_nbytes(src0);
            ধ্রুৱ size_t size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy((ৱর্ণ *) dst_d + size0, src1_d, size1)));
        }
    } অন্যথা {
        concat_T_sycl_non_cont<block_q5_1>(
            stream, (ধ্রুৱ ৱর্ণ *) src0->data, (ধ্রুৱ ৱর্ণ *) src1->data,
            (ৱর্ণ *) dst->data,
            ne00_blk, src0->ne[1], src0->ne[2], src0->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->ne[0] / QK5_1, src1->ne[1], src1->ne[2], src1->ne[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            ne0_blk, dst->ne[1], dst->ne[2], dst->ne[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

স্থির শূন্য concat_impl_q8_0_sycl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ধ্রুৱ ggml_tensor * src0   = dst->src[0];
    ধ্রুৱ ggml_tensor * src1   = dst->src[1];
    queue_ptr           stream = ctx.stream();

    ধ্রুৱ int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(src1->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(dst->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(src0->ne[0] % QK8_0 == 0);
    GGML_ASSERT(src1->ne[0] % QK8_0 == 0);
    GGML_ASSERT(dst->ne[0] % QK8_0 == 0);

    ধ্রুৱ পূর্ণাঙ্ক ne00_blk = src0->ne[0] / QK8_0;
    ধ্রুৱ পূর্ণাঙ্ক ne0_blk  = dst->ne[0] / QK8_0;

    যদি (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        ধ্রুৱ block_q8_0 * src0_d = (ধ্রুৱ block_q8_0 *) src0->data;
        ধ্রুৱ block_q8_0 * src1_d = (ধ্রুৱ block_q8_0 *) src1->data;
        block_q8_0 * dst_d        = (block_q8_0 *) dst->data;
        ধ্রুৱ size_t type_size    = পরিমাণ(block_q8_0);

        যদি (dim != 3) {
            বাবে (পূর্ণাঙ্ক i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<block_q8_0>(
                    src0_d + i3 * (src0->nb[3] / type_size),
                    src1_d + i3 * (src1->nb[3] / type_size),
                    dst_d + i3 * (dst->nb[3] / type_size),
                    ne00_blk, src0->ne[1], src0->ne[2], ne0_blk,
                    dst->ne[1], dst->ne[2], dim, stream);
            }
        } অন্যথা {
            ধ্রুৱ size_t size0 = ggml_nbytes(src0);
            ধ্রুৱ size_t size1 = ggml_nbytes(src1);
            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy((ৱর্ণ *) dst_d + size0, src1_d, size1)));
        }
    } অন্যথা {
        concat_T_sycl_non_cont<block_q8_0>(
            stream, (ধ্রুৱ ৱর্ণ *) src0->data, (ধ্রুৱ ৱর্ণ *) src1->data,
            (ৱর্ণ *) dst->data,
            ne00_blk, src0->ne[1], src0->ne[2], src0->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->ne[0] / QK8_0, src1->ne[1], src1->ne[2], src1->ne[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            ne0_blk, dst->ne[1], dst->ne[2], dst->ne[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

শূন্য ggml_sycl_op_concat(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ৱিকল্প (dst->type) {
    পক্ষ GGML_TYPE_F32:
        concat_impl_sycl<ভিন্ন>(ctx, dst);
        ভঙ্গ;
    পক্ষ GGML_TYPE_F16:
        concat_impl_sycl<sycl::half>(ctx, dst);
        ভঙ্গ;
#ifdef GGML_SYCL_HAS_BF16
    পক্ষ GGML_TYPE_BF16:
        concat_impl_sycl<sycl::ext::oneapi::bfloat16>(ctx, dst);
        ভঙ্গ;
#endif
    পক্ষ GGML_TYPE_I32:
        concat_impl_sycl<int32_t>(ctx, dst);
        ভঙ্গ;
    পক্ষ GGML_TYPE_I16:
        concat_impl_sycl<int16_t>(ctx, dst);
        ভঙ্গ;
    পক্ষ GGML_TYPE_I64:
        concat_impl_sycl<int64_t>(ctx, dst);
        ভঙ্গ;
    পক্ষ GGML_TYPE_I8:
        concat_impl_sycl<int8_t>(ctx, dst);
        ভঙ্গ;
    পক্ষ GGML_TYPE_Q4_0:
        concat_impl_q4_0_sycl(ctx, dst);
        ভঙ্গ;
    পক্ষ GGML_TYPE_Q4_1:
        concat_impl_q4_1_sycl(ctx, dst);
        ভঙ্গ;
    পক্ষ GGML_TYPE_Q5_0:
        concat_impl_q5_0_sycl(ctx, dst);
        ভঙ্গ;
    পক্ষ GGML_TYPE_Q5_1:
        concat_impl_q5_1_sycl(ctx, dst);
        ভঙ্গ;
    পক্ষ GGML_TYPE_Q8_0:
        concat_impl_q8_0_sycl(ctx, dst);
        ভঙ্গ;
    স্ৱভাৱ:
        fprintf(stderr, "%s: unsupported types: dst: %s\n", __func__, ggml_type_name(dst->type));
        GGML_ASSERT(মিছা);
    }
}
