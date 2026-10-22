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

#അന്തര്ഭാവ "ggml.h"

#അന്തര്ഭാവ "concat.hpp"

സ്ഥിതം വരിയിൽ size_t elem_size(ggml_type t) {
    തിരികെ ggml_type_size(t) / ggml_blck_size(t);
}

template <typename T>
സ്ഥിതം ശൂന്യം concat_T_dim0(സ്ഥിരം T *x, സ്ഥിരം T *y, T *dst,
                            സ്ഥിരം പൂർണ്ണസംഖ്യ ne0, സ്ഥിരം പൂർണ്ണസംഖ്യ ne00,
                            സ്ഥിരം sycl::nd_item<3> &item_ct1) {
  പൂർണ്ണസംഖ്യ nidx = item_ct1.get_local_id(2) +
             item_ct1.get_group(2) * item_ct1.get_local_range(2);
  എങ്കിൽ (nidx >= ne0) {
    തിരികെ;
  }
  // operation
  പൂർണ്ണസംഖ്യ offset_dst = nidx + item_ct1.get_group(1) * ne0 +
                   item_ct1.get_group(0) * ne0 * item_ct1.get_group_range(1);
  എങ്കിൽ (nidx < ne00) { // src0
    പൂർണ്ണസംഖ്യ offset_src = nidx + item_ct1.get_group(1) * ne00 +
                     item_ct1.get_group(0) * ne00 * item_ct1.get_group_range(1);
    dst[offset_dst] = x[offset_src];
  } അല്ലെങ്കിൽ {
    പൂർണ്ണസംഖ്യ offset_src =
        nidx - ne00 + item_ct1.get_group(1) * (ne0 - ne00) +
        item_ct1.get_group(0) * (ne0 - ne00) * item_ct1.get_group_range(1);
    dst[offset_dst] = y[offset_src];
  }
}

template <typename T>
സ്ഥിതം ശൂന്യം concat_T_dim1(സ്ഥിരം T *x, സ്ഥിരം T *y, T *dst,
                            സ്ഥിരം പൂർണ്ണസംഖ്യ ne0, സ്ഥിരം പൂർണ്ണസംഖ്യ ne01,
                            സ്ഥിരം sycl::nd_item<3> &item_ct1) {
  പൂർണ്ണസംഖ്യ nidx = item_ct1.get_local_id(2) +
             item_ct1.get_group(2) * item_ct1.get_local_range(2);
  എങ്കിൽ (nidx >= ne0) {
    തിരികെ;
  }
  // operation
  പൂർണ്ണസംഖ്യ offset_dst = nidx + item_ct1.get_group(1) * ne0 +
                   item_ct1.get_group(0) * ne0 * item_ct1.get_group_range(1);
  എങ്കിൽ (item_ct1.get_group(1) < (size_t) ne01) { // src0
    പൂർണ്ണസംഖ്യ offset_src =
        nidx + item_ct1.get_group(1) * ne0 + item_ct1.get_group(0) * ne0 * ne01;
    dst[offset_dst] = x[offset_src];
  } അല്ലെങ്കിൽ {
    പൂർണ്ണസംഖ്യ offset_src =
        nidx + (item_ct1.get_group(1) - ne01) * ne0 +
        item_ct1.get_group(0) * ne0 * (item_ct1.get_group_range(1) - ne01);
    dst[offset_dst] = y[offset_src];
  }
}

template <typename T>
സ്ഥിതം ശൂന്യം concat_T_dim2(സ്ഥിരം T *x, സ്ഥിരം T *y, T *dst,
                            സ്ഥിരം പൂർണ്ണസംഖ്യ ne0, സ്ഥിരം പൂർണ്ണസംഖ്യ ne02,
                            സ്ഥിരം sycl::nd_item<3> &item_ct1) {
  പൂർണ്ണസംഖ്യ nidx = item_ct1.get_local_id(2) +
             item_ct1.get_group(2) * item_ct1.get_local_range(2);
  എങ്കിൽ (nidx >= ne0) {
    തിരികെ;
  }
  // operation
  പൂർണ്ണസംഖ്യ offset_dst = nidx + item_ct1.get_group(1) * ne0 +
                   item_ct1.get_group(0) * ne0 * item_ct1.get_group_range(1);
  എങ്കിൽ (item_ct1.get_group(0) < (size_t) ne02) { // src0
    പൂർണ്ണസംഖ്യ offset_src = nidx + item_ct1.get_group(1) * ne0 +
                     item_ct1.get_group(0) * ne0 * item_ct1.get_group_range(1);
    dst[offset_dst] = x[offset_src];
  } അല്ലെങ്കിൽ {
    പൂർണ്ണസംഖ്യ offset_src =
        nidx + item_ct1.get_group(1) * ne0 +
        (item_ct1.get_group(0) - ne02) * ne0 * item_ct1.get_group_range(1);
    dst[offset_dst] = y[offset_src];
  }
}

template <typename T>
സ്ഥിതം ശൂന്യം concat_T_sycl(സ്ഥിരം T *x, സ്ഥിരം T *y, T *dst,
                            പൂർണ്ണസംഖ്യ ne00, പൂർണ്ണസംഖ്യ ne01, പൂർണ്ണസംഖ്യ ne02, പൂർണ്ണസംഖ്യ ne0, പൂർണ്ണസംഖ്യ ne1,
                            പൂർണ്ണസംഖ്യ ne2, പൂർണ്ണസംഖ്യ dim, queue_ptr stream) {
  പൂർണ്ണസംഖ്യ അക്കം_blocks = (ne0 + SYCL_CONCAT_BLOCK_SIZE - 1) / SYCL_CONCAT_BLOCK_SIZE;
  sycl::range<3> gridDim(ne2, ne1, അക്കം_blocks);
  തിരഞ്ഞെടുപ്പ് (dim) {
  അവസ്ഥ 0:
      stream->parallel_for(sycl::nd_range<3>(gridDim * sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE),
                                          sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE)),
                        [=](sycl::nd_item<3> item_ct1) { concat_T_dim0<T>(x, y, dst, ne0, ne00, item_ct1); });
      നിർത്തുക;
  അവസ്ഥ 1:
      stream->parallel_for(sycl::nd_range<3>(gridDim * sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE),
                                          sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE)),
                        [=](sycl::nd_item<3> item_ct1) { concat_T_dim1<T>(x, y, dst, ne0, ne01, item_ct1); });
      നിർത്തുക;
  // dim >=2 will be dispatched to the default path
  സ്ഥിരസ്ഥിതി:
      stream->parallel_for(sycl::nd_range<3>(gridDim * sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE),
                                          sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE)),
                        [=](sycl::nd_item<3> item_ct1) { concat_T_dim2<T>(x, y, dst, ne0, ne02, item_ct1); });
      നിർത്തുക;
  }
}

// non-contiguous kernel (slow)
template<typename T>
സ്ഥിതം ശൂന്യം concat_T_sycl_non_cont(
    queue_ptr stream, സ്ഥിരം അക്ഷരം *src0, സ്ഥിരം അക്ഷരം *src1, അക്ഷരം *dst,
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
  സ്ഥിരം int64_t ne0_pad   = GGML_PAD(ne0, WARP_SIZE);
  സ്ഥിരം int64_t block_ne0 = ne0_pad < SYCL_CONCAT_BLOCK_SIZE ? ne0_pad : (int64_t) SYCL_CONCAT_BLOCK_SIZE;
  sycl::range<3> blockDim(1, 1, block_ne0);

  stream->parallel_for(sycl::nd_range<3>(gridDim * blockDim, blockDim), [=](sycl::nd_item<3> item_ct1) {
      int64_t i3 = item_ct1.get_group(0);
      int64_t i2 = item_ct1.get_group(1);
      int64_t i1 = item_ct1.get_group(2);

      int64_t o[4] = { 0, 0, 0, 0 };
      o[dim]       = dim == 0 ? ne00 : (dim == 1 ? ne01 : (dim == 2 ? ne02 : ne03));

      സ്ഥിരം T * x;

      വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = item_ct1.get_local_id(2); i0 < ne0; i0 += item_ct1.get_local_range(2)) {
          എങ്കിൽ (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
              x = (സ്ഥിരം T *) (src0 + (i3) *nb03 + (i2) *nb02 + (i1) *nb01 + (i0) *nb00);
          } അല്ലെങ്കിൽ {
              x = (സ്ഥിരം T *) (src1 + (i3 - o[3]) * nb13 + (i2 - o[2]) * nb12 + (i1 - o[1]) * nb11 +
                                   (i0 - o[0]) * nb10);
          }

          T *y = (T *)(dst + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

          *y = *x;
      }
  });
}

template <typename T>
ശൂന്യം concat_impl_sycl(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    സ്ഥിരം ggml_tensor *  src0   = dst->src[0];
    സ്ഥിരം ggml_tensor *  src1   = dst->src[1];
    queue_ptr            stream = ctx.stream();

    സ്ഥിരം int32_t dim = ((int32_t *) dst->op_params)[0];

    എങ്കിൽ (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        സ്ഥിരം T * src0_d = (സ്ഥിരം T *) src0->data;
        സ്ഥിരം T * src1_d = (സ്ഥിരം T *) src1->data;
        T * dst_d = (T *) dst->data;
        size_t type_size = elem_size(dst->type);
        എങ്കിൽ (dim != 3) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<T>(src0_d + i3 * (src0->nb[3] / type_size), src1_d + i3 * (src1->nb[3] / type_size),
                                dst_d + i3 * (dst->nb[3] / type_size), src0->ne[0], src0->ne[1], src0->ne[2], dst->ne[0],
                                dst->ne[1], dst->ne[2], dim, stream);
            }
        } അല്ലെങ്കിൽ {
            സ്ഥിരം size_t size0 = ggml_nbytes(src0);
            സ്ഥിരം size_t size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(dst_d + size0 / type_size, src1_d, size1)));
        }
    } അല്ലെങ്കിൽ {
        concat_T_sycl_non_cont<T>(stream, (സ്ഥിരം അക്ഷരം *) src0->data, (സ്ഥിരം അക്ഷരം *) src1->data, (അക്ഷരം *) dst->data,
                                 src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src0->nb[0], src0->nb[1],
                                 src0->nb[2], src0->nb[3], src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                                 src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3], dst->ne[0], dst->ne[1], dst->ne[2],
                                 dst->ne[3], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

സ്ഥിതം ശൂന്യം concat_impl_q4_0_sycl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    സ്ഥിരം ggml_tensor * src0   = dst->src[0];
    സ്ഥിരം ggml_tensor * src1   = dst->src[1];
    queue_ptr           stream = ctx.stream();

    സ്ഥിരം int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(src1->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(dst->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(src0->ne[0] % QK4_0 == 0);
    GGML_ASSERT(src1->ne[0] % QK4_0 == 0);
    GGML_ASSERT(dst->ne[0] % QK4_0 == 0);

    സ്ഥിരം പൂർണ്ണസംഖ്യ ne00_blk = src0->ne[0] / QK4_0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ne0_blk  = dst->ne[0] / QK4_0;

    എങ്കിൽ (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        സ്ഥിരം block_q4_0 * src0_d = (സ്ഥിരം block_q4_0 *) src0->data;
        സ്ഥിരം block_q4_0 * src1_d = (സ്ഥിരം block_q4_0 *) src1->data;
        block_q4_0 * dst_d        = (block_q4_0 *) dst->data;
        സ്ഥിരം size_t type_size    = വലുപ്പം(block_q4_0);

        എങ്കിൽ (dim != 3) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<block_q4_0>(
                    src0_d + i3 * (src0->nb[3] / type_size),
                    src1_d + i3 * (src1->nb[3] / type_size),
                    dst_d + i3 * (dst->nb[3] / type_size),
                    ne00_blk, src0->ne[1], src0->ne[2], ne0_blk,
                    dst->ne[1], dst->ne[2], dim, stream);
            }
        } അല്ലെങ്കിൽ {
            സ്ഥിരം size_t size0 = ggml_nbytes(src0);
            സ്ഥിരം size_t size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy((അക്ഷരം *) dst_d + size0, src1_d, size1)));
        }
    } അല്ലെങ്കിൽ {
        concat_T_sycl_non_cont<block_q4_0>(
            stream, (സ്ഥിരം അക്ഷരം *) src0->data, (സ്ഥിരം അക്ഷരം *) src1->data,
            (അക്ഷരം *) dst->data,
            ne00_blk, src0->ne[1], src0->ne[2], src0->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->ne[0] / QK4_0, src1->ne[1], src1->ne[2], src1->ne[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            ne0_blk, dst->ne[1], dst->ne[2], dst->ne[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

സ്ഥിതം ശൂന്യം concat_impl_q4_1_sycl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    സ്ഥിരം ggml_tensor * src0   = dst->src[0];
    സ്ഥിരം ggml_tensor * src1   = dst->src[1];
    queue_ptr           stream = ctx.stream();

    സ്ഥിരം int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(src1->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(dst->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(src0->ne[0] % QK4_1 == 0);
    GGML_ASSERT(src1->ne[0] % QK4_1 == 0);
    GGML_ASSERT(dst->ne[0] % QK4_1 == 0);

    സ്ഥിരം പൂർണ്ണസംഖ്യ ne00_blk = src0->ne[0] / QK4_1;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ne0_blk  = dst->ne[0] / QK4_1;

    എങ്കിൽ (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        സ്ഥിരം block_q4_1 * src0_d = (സ്ഥിരം block_q4_1 *) src0->data;
        സ്ഥിരം block_q4_1 * src1_d = (സ്ഥിരം block_q4_1 *) src1->data;
        block_q4_1 * dst_d        = (block_q4_1 *) dst->data;
        സ്ഥിരം size_t type_size    = വലുപ്പം(block_q4_1);

        എങ്കിൽ (dim != 3) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<block_q4_1>(
                    src0_d + i3 * (src0->nb[3] / type_size),
                    src1_d + i3 * (src1->nb[3] / type_size),
                    dst_d + i3 * (dst->nb[3] / type_size),
                    ne00_blk, src0->ne[1], src0->ne[2], ne0_blk,
                    dst->ne[1], dst->ne[2], dim, stream);
            }
        } അല്ലെങ്കിൽ {
            സ്ഥിരം size_t size0 = ggml_nbytes(src0);
            സ്ഥിരം size_t size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy((അക്ഷരം *) dst_d + size0, src1_d, size1)));
        }
    } അല്ലെങ്കിൽ {
        concat_T_sycl_non_cont<block_q4_1>(
            stream, (സ്ഥിരം അക്ഷരം *) src0->data, (സ്ഥിരം അക്ഷരം *) src1->data,
            (അക്ഷരം *) dst->data,
            ne00_blk, src0->ne[1], src0->ne[2], src0->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->ne[0] / QK4_1, src1->ne[1], src1->ne[2], src1->ne[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            ne0_blk, dst->ne[1], dst->ne[2], dst->ne[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

സ്ഥിതം ശൂന്യം concat_impl_q5_0_sycl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    സ്ഥിരം ggml_tensor * src0   = dst->src[0];
    സ്ഥിരം ggml_tensor * src1   = dst->src[1];
    queue_ptr           stream = ctx.stream();

    സ്ഥിരം int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_Q5_0);
    GGML_ASSERT(src1->type == GGML_TYPE_Q5_0);
    GGML_ASSERT(dst->type == GGML_TYPE_Q5_0);
    GGML_ASSERT(src0->ne[0] % QK5_0 == 0);
    GGML_ASSERT(src1->ne[0] % QK5_0 == 0);
    GGML_ASSERT(dst->ne[0] % QK5_0 == 0);

    സ്ഥിരം പൂർണ്ണസംഖ്യ ne00_blk = src0->ne[0] / QK5_0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ne0_blk  = dst->ne[0] / QK5_0;

    എങ്കിൽ (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        സ്ഥിരം block_q5_0 * src0_d = (സ്ഥിരം block_q5_0 *) src0->data;
        സ്ഥിരം block_q5_0 * src1_d = (സ്ഥിരം block_q5_0 *) src1->data;
        block_q5_0 * dst_d        = (block_q5_0 *) dst->data;
        സ്ഥിരം size_t type_size    = വലുപ്പം(block_q5_0);

        എങ്കിൽ (dim != 3) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<block_q5_0>(
                    src0_d + i3 * (src0->nb[3] / type_size),
                    src1_d + i3 * (src1->nb[3] / type_size),
                    dst_d + i3 * (dst->nb[3] / type_size),
                    ne00_blk, src0->ne[1], src0->ne[2], ne0_blk,
                    dst->ne[1], dst->ne[2], dim, stream);
            }
        } അല്ലെങ്കിൽ {
            സ്ഥിരം size_t size0 = ggml_nbytes(src0);
            സ്ഥിരം size_t size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy((അക്ഷരം *) dst_d + size0, src1_d, size1)));
        }
    } അല്ലെങ്കിൽ {
        concat_T_sycl_non_cont<block_q5_0>(
            stream, (സ്ഥിരം അക്ഷരം *) src0->data, (സ്ഥിരം അക്ഷരം *) src1->data,
            (അക്ഷരം *) dst->data,
            ne00_blk, src0->ne[1], src0->ne[2], src0->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->ne[0] / QK5_0, src1->ne[1], src1->ne[2], src1->ne[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            ne0_blk, dst->ne[1], dst->ne[2], dst->ne[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

സ്ഥിതം ശൂന്യം concat_impl_q5_1_sycl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    സ്ഥിരം ggml_tensor * src0   = dst->src[0];
    സ്ഥിരം ggml_tensor * src1   = dst->src[1];
    queue_ptr           stream = ctx.stream();

    സ്ഥിരം int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_Q5_1);
    GGML_ASSERT(src1->type == GGML_TYPE_Q5_1);
    GGML_ASSERT(dst->type == GGML_TYPE_Q5_1);
    GGML_ASSERT(src0->ne[0] % QK5_1 == 0);
    GGML_ASSERT(src1->ne[0] % QK5_1 == 0);
    GGML_ASSERT(dst->ne[0] % QK5_1 == 0);

    സ്ഥിരം പൂർണ്ണസംഖ്യ ne00_blk = src0->ne[0] / QK5_1;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ne0_blk  = dst->ne[0] / QK5_1;

    എങ്കിൽ (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        സ്ഥിരം block_q5_1 * src0_d = (സ്ഥിരം block_q5_1 *) src0->data;
        സ്ഥിരം block_q5_1 * src1_d = (സ്ഥിരം block_q5_1 *) src1->data;
        block_q5_1 * dst_d        = (block_q5_1 *) dst->data;
        സ്ഥിരം size_t type_size    = വലുപ്പം(block_q5_1);

        എങ്കിൽ (dim != 3) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<block_q5_1>(
                    src0_d + i3 * (src0->nb[3] / type_size),
                    src1_d + i3 * (src1->nb[3] / type_size),
                    dst_d + i3 * (dst->nb[3] / type_size),
                    ne00_blk, src0->ne[1], src0->ne[2], ne0_blk,
                    dst->ne[1], dst->ne[2], dim, stream);
            }
        } അല്ലെങ്കിൽ {
            സ്ഥിരം size_t size0 = ggml_nbytes(src0);
            സ്ഥിരം size_t size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy((അക്ഷരം *) dst_d + size0, src1_d, size1)));
        }
    } അല്ലെങ്കിൽ {
        concat_T_sycl_non_cont<block_q5_1>(
            stream, (സ്ഥിരം അക്ഷരം *) src0->data, (സ്ഥിരം അക്ഷരം *) src1->data,
            (അക്ഷരം *) dst->data,
            ne00_blk, src0->ne[1], src0->ne[2], src0->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->ne[0] / QK5_1, src1->ne[1], src1->ne[2], src1->ne[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            ne0_blk, dst->ne[1], dst->ne[2], dst->ne[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

സ്ഥിതം ശൂന്യം concat_impl_q8_0_sycl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    സ്ഥിരം ggml_tensor * src0   = dst->src[0];
    സ്ഥിരം ggml_tensor * src1   = dst->src[1];
    queue_ptr           stream = ctx.stream();

    സ്ഥിരം int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(src1->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(dst->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(src0->ne[0] % QK8_0 == 0);
    GGML_ASSERT(src1->ne[0] % QK8_0 == 0);
    GGML_ASSERT(dst->ne[0] % QK8_0 == 0);

    സ്ഥിരം പൂർണ്ണസംഖ്യ ne00_blk = src0->ne[0] / QK8_0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ne0_blk  = dst->ne[0] / QK8_0;

    എങ്കിൽ (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        സ്ഥിരം block_q8_0 * src0_d = (സ്ഥിരം block_q8_0 *) src0->data;
        സ്ഥിരം block_q8_0 * src1_d = (സ്ഥിരം block_q8_0 *) src1->data;
        block_q8_0 * dst_d        = (block_q8_0 *) dst->data;
        സ്ഥിരം size_t type_size    = വലുപ്പം(block_q8_0);

        എങ്കിൽ (dim != 3) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<block_q8_0>(
                    src0_d + i3 * (src0->nb[3] / type_size),
                    src1_d + i3 * (src1->nb[3] / type_size),
                    dst_d + i3 * (dst->nb[3] / type_size),
                    ne00_blk, src0->ne[1], src0->ne[2], ne0_blk,
                    dst->ne[1], dst->ne[2], dim, stream);
            }
        } അല്ലെങ്കിൽ {
            സ്ഥിരം size_t size0 = ggml_nbytes(src0);
            സ്ഥിരം size_t size1 = ggml_nbytes(src1);
            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy((അക്ഷരം *) dst_d + size0, src1_d, size1)));
        }
    } അല്ലെങ്കിൽ {
        concat_T_sycl_non_cont<block_q8_0>(
            stream, (സ്ഥിരം അക്ഷരം *) src0->data, (സ്ഥിരം അക്ഷരം *) src1->data,
            (അക്ഷരം *) dst->data,
            ne00_blk, src0->ne[1], src0->ne[2], src0->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->ne[0] / QK8_0, src1->ne[1], src1->ne[2], src1->ne[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            ne0_blk, dst->ne[1], dst->ne[2], dst->ne[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

ശൂന്യം ggml_sycl_op_concat(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    തിരഞ്ഞെടുപ്പ് (dst->type) {
    അവസ്ഥ GGML_TYPE_F32:
        concat_impl_sycl<ദശാംശം>(ctx, dst);
        നിർത്തുക;
    അവസ്ഥ GGML_TYPE_F16:
        concat_impl_sycl<sycl::half>(ctx, dst);
        നിർത്തുക;
#ifdef GGML_SYCL_HAS_BF16
    അവസ്ഥ GGML_TYPE_BF16:
        concat_impl_sycl<sycl::ext::oneapi::bfloat16>(ctx, dst);
        നിർത്തുക;
#endif
    അവസ്ഥ GGML_TYPE_I32:
        concat_impl_sycl<int32_t>(ctx, dst);
        നിർത്തുക;
    അവസ്ഥ GGML_TYPE_I16:
        concat_impl_sycl<int16_t>(ctx, dst);
        നിർത്തുക;
    അവസ്ഥ GGML_TYPE_I64:
        concat_impl_sycl<int64_t>(ctx, dst);
        നിർത്തുക;
    അവസ്ഥ GGML_TYPE_I8:
        concat_impl_sycl<int8_t>(ctx, dst);
        നിർത്തുക;
    അവസ്ഥ GGML_TYPE_Q4_0:
        concat_impl_q4_0_sycl(ctx, dst);
        നിർത്തുക;
    അവസ്ഥ GGML_TYPE_Q4_1:
        concat_impl_q4_1_sycl(ctx, dst);
        നിർത്തുക;
    അവസ്ഥ GGML_TYPE_Q5_0:
        concat_impl_q5_0_sycl(ctx, dst);
        നിർത്തുക;
    അവസ്ഥ GGML_TYPE_Q5_1:
        concat_impl_q5_1_sycl(ctx, dst);
        നിർത്തുക;
    അവസ്ഥ GGML_TYPE_Q8_0:
        concat_impl_q8_0_sycl(ctx, dst);
        നിർത്തുക;
    സ്ഥിരസ്ഥിതി:
        fprintf(stderr, "%s: unsupported types: dst: %s\n", __func__, ggml_type_name(dst->type));
        GGML_ASSERT(അസത്യം);
    }
}
