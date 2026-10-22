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

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "concat.hpp"

ਸਥਿਰ ਸਤਰਅੰਦਰ size_t elem_ਮਾਪ(ggml_type t) {
    ਵਾਪਸ ggml_type_ਮਾਪ(t) / ggml_blck_ਮਾਪ(t);
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਖਾਲੀ concat_T_dim0(ਅਚਲ T *x, ਅਚਲ T *y, T *dst,
                            ਅਚਲ ਪੂਰਨ ne0, ਅਚਲ ਪੂਰਨ ne00,
                            ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {
  ਪੂਰਨ nidx = ਵਸਤੂ_ct1.get_local_id(2) +
             ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(2);
  ਜੇ (nidx >= ne0) {
    ਵਾਪਸ;
  }
  // operation
  ਪੂਰਨ ਵਿੱਥ_dst = nidx + ਵਸਤੂ_ct1.get_ਸਮੂਹ(1) * ne0 +
                   ਵਸਤੂ_ct1.get_ਸਮੂਹ(0) * ne0 * ਵਸਤੂ_ct1.get_ਸਮੂਹ_range(1);
  ਜੇ (nidx < ne00) { // src0
    ਪੂਰਨ ਵਿੱਥ_src = nidx + ਵਸਤੂ_ct1.get_ਸਮੂਹ(1) * ne00 +
                     ਵਸਤੂ_ct1.get_ਸਮੂਹ(0) * ne00 * ਵਸਤੂ_ct1.get_ਸਮੂਹ_range(1);
    dst[ਵਿੱਥ_dst] = x[ਵਿੱਥ_src];
  } ਨਹੀਂਤਾਂ {
    ਪੂਰਨ ਵਿੱਥ_src =
        nidx - ne00 + ਵਸਤੂ_ct1.get_ਸਮੂਹ(1) * (ne0 - ne00) +
        ਵਸਤੂ_ct1.get_ਸਮੂਹ(0) * (ne0 - ne00) * ਵਸਤੂ_ct1.get_ਸਮੂਹ_range(1);
    dst[ਵਿੱਥ_dst] = y[ਵਿੱਥ_src];
  }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਖਾਲੀ concat_T_dim1(ਅਚਲ T *x, ਅਚਲ T *y, T *dst,
                            ਅਚਲ ਪੂਰਨ ne0, ਅਚਲ ਪੂਰਨ ne01,
                            ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {
  ਪੂਰਨ nidx = ਵਸਤੂ_ct1.get_local_id(2) +
             ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(2);
  ਜੇ (nidx >= ne0) {
    ਵਾਪਸ;
  }
  // operation
  ਪੂਰਨ ਵਿੱਥ_dst = nidx + ਵਸਤੂ_ct1.get_ਸਮੂਹ(1) * ne0 +
                   ਵਸਤੂ_ct1.get_ਸਮੂਹ(0) * ne0 * ਵਸਤੂ_ct1.get_ਸਮੂਹ_range(1);
  ਜੇ (ਵਸਤੂ_ct1.get_ਸਮੂਹ(1) < (size_t) ne01) { // src0
    ਪੂਰਨ ਵਿੱਥ_src =
        nidx + ਵਸਤੂ_ct1.get_ਸਮੂਹ(1) * ne0 + ਵਸਤੂ_ct1.get_ਸਮੂਹ(0) * ne0 * ne01;
    dst[ਵਿੱਥ_dst] = x[ਵਿੱਥ_src];
  } ਨਹੀਂਤਾਂ {
    ਪੂਰਨ ਵਿੱਥ_src =
        nidx + (ਵਸਤੂ_ct1.get_ਸਮੂਹ(1) - ne01) * ne0 +
        ਵਸਤੂ_ct1.get_ਸਮੂਹ(0) * ne0 * (ਵਸਤੂ_ct1.get_ਸਮੂਹ_range(1) - ne01);
    dst[ਵਿੱਥ_dst] = y[ਵਿੱਥ_src];
  }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਖਾਲੀ concat_T_dim2(ਅਚਲ T *x, ਅਚਲ T *y, T *dst,
                            ਅਚਲ ਪੂਰਨ ne0, ਅਚਲ ਪੂਰਨ ne02,
                            ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {
  ਪੂਰਨ nidx = ਵਸਤੂ_ct1.get_local_id(2) +
             ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(2);
  ਜੇ (nidx >= ne0) {
    ਵਾਪਸ;
  }
  // operation
  ਪੂਰਨ ਵਿੱਥ_dst = nidx + ਵਸਤੂ_ct1.get_ਸਮੂਹ(1) * ne0 +
                   ਵਸਤੂ_ct1.get_ਸਮੂਹ(0) * ne0 * ਵਸਤੂ_ct1.get_ਸਮੂਹ_range(1);
  ਜੇ (ਵਸਤੂ_ct1.get_ਸਮੂਹ(0) < (size_t) ne02) { // src0
    ਪੂਰਨ ਵਿੱਥ_src = nidx + ਵਸਤੂ_ct1.get_ਸਮੂਹ(1) * ne0 +
                     ਵਸਤੂ_ct1.get_ਸਮੂਹ(0) * ne0 * ਵਸਤੂ_ct1.get_ਸਮੂਹ_range(1);
    dst[ਵਿੱਥ_dst] = x[ਵਿੱਥ_src];
  } ਨਹੀਂਤਾਂ {
    ਪੂਰਨ ਵਿੱਥ_src =
        nidx + ਵਸਤੂ_ct1.get_ਸਮੂਹ(1) * ne0 +
        (ਵਸਤੂ_ct1.get_ਸਮੂਹ(0) - ne02) * ne0 * ਵਸਤੂ_ct1.get_ਸਮੂਹ_range(1);
    dst[ਵਿੱਥ_dst] = y[ਵਿੱਥ_src];
  }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਖਾਲੀ concat_T_sycl(ਅਚਲ T *x, ਅਚਲ T *y, T *dst,
                            ਪੂਰਨ ne00, ਪੂਰਨ ne01, ਪੂਰਨ ne02, ਪੂਰਨ ne0, ਪੂਰਨ ne1,
                            ਪੂਰਨ ne2, ਪੂਰਨ dim, ਕਤਾਰ_ptr stream) {
  ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = (ne0 + SYCL_CONCAT_ਖੰਡ_ਮਾਪ - 1) / SYCL_CONCAT_ਖੰਡ_ਮਾਪ;
  sycl::range<3> gridDim(ne2, ne1, ਅੰਕ_ਖੰਡਾਂ);
  ਚੋਣ (dim) {
  ਹਾਲਤ 0:
      stream->parallel_for(sycl::nd_range<3>(gridDim * sycl::range<3>(1, 1, SYCL_CONCAT_ਖੰਡ_ਮਾਪ),
                                          sycl::range<3>(1, 1, SYCL_CONCAT_ਖੰਡ_ਮਾਪ)),
                        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) { concat_T_dim0<T>(x, y, dst, ne0, ne00, ਵਸਤੂ_ct1); });
      ਰੋਕੋ;
  ਹਾਲਤ 1:
      stream->parallel_for(sycl::nd_range<3>(gridDim * sycl::range<3>(1, 1, SYCL_CONCAT_ਖੰਡ_ਮਾਪ),
                                          sycl::range<3>(1, 1, SYCL_CONCAT_ਖੰਡ_ਮਾਪ)),
                        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) { concat_T_dim1<T>(x, y, dst, ne0, ne01, ਵਸਤੂ_ct1); });
      ਰੋਕੋ;
  // dim >=2 will be dispatched to the default path
  ਮੂਲ:
      stream->parallel_for(sycl::nd_range<3>(gridDim * sycl::range<3>(1, 1, SYCL_CONCAT_ਖੰਡ_ਮਾਪ),
                                          sycl::range<3>(1, 1, SYCL_CONCAT_ਖੰਡ_ਮਾਪ)),
                        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) { concat_T_dim2<T>(x, y, dst, ne0, ne02, ਵਸਤੂ_ct1); });
      ਰੋਕੋ;
  }
}

// non-contiguous kernel (slow)
ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਖਾਲੀ concat_T_sycl_non_cont(
    ਕਤਾਰ_ptr stream, ਅਚਲ ਅੱਖਰ *src0, ਅਚਲ ਅੱਖਰ *src1, ਅੱਖਰ *dst,
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
  ਅਚਲ int64_t ne0_pad   = GGML_PAD(ne0, WARP_ਮਾਪ);
  ਅਚਲ int64_t ਖੰਡ_ne0 = ne0_pad < SYCL_CONCAT_ਖੰਡ_ਮਾਪ ? ne0_pad : (int64_t) SYCL_CONCAT_ਖੰਡ_ਮਾਪ;
  sycl::range<3> ਖੰਡ_Dim(1, 1, ਖੰਡ_ne0);

  stream->parallel_for(sycl::nd_range<3>(gridDim * ਖੰਡ_Dim, ਖੰਡ_Dim), [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
      int64_t i3 = ਵਸਤੂ_ct1.get_ਸਮੂਹ(0);
      int64_t i2 = ਵਸਤੂ_ct1.get_ਸਮੂਹ(1);
      int64_t i1 = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2);

      int64_t o[4] = { 0, 0, 0, 0 };
      o[dim]       = dim == 0 ? ne00 : (dim == 1 ? ne01 : (dim == 2 ? ne02 : ne03));

      ਅਚਲ T * x;

      ਲਈ (ਪੂਰਨ i0 = ਵਸਤੂ_ct1.get_local_id(2); i0 < ne0; i0 += ਵਸਤੂ_ct1.get_local_range(2)) {
          ਜੇ (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
              x = (ਅਚਲ T *) (src0 + (i3) *nb03 + (i2) *nb02 + (i1) *nb01 + (i0) *nb00);
          } ਨਹੀਂਤਾਂ {
              x = (ਅਚਲ T *) (src1 + (i3 - o[3]) * nb13 + (i2 - o[2]) * nb12 + (i1 - o[1]) * nb11 +
                                   (i0 - o[0]) * nb10);
          }

          T *y = (T *)(dst + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

          *y = *x;
      }
  });
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
ਖਾਲੀ concat_impl_sycl(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ਅਚਲ ggml_tensor *  src0   = dst->src[0];
    ਅਚਲ ggml_tensor *  src1   = dst->src[1];
    ਕਤਾਰ_ptr            stream = ctx.stream();

    ਅਚਲ int32_t dim = ((int32_t *) dst->op_params)[0];

    ਜੇ (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        ਅਚਲ T * src0_d = (ਅਚਲ T *) src0->ਅੰਕੜੇ;
        ਅਚਲ T * src1_d = (ਅਚਲ T *) src1->ਅੰਕੜੇ;
        T * dst_d = (T *) dst->ਅੰਕੜੇ;
        size_t type_ਮਾਪ = elem_ਮਾਪ(dst->type);
        ਜੇ (dim != 3) {
            ਲਈ (ਪੂਰਨ i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<T>(src0_d + i3 * (src0->nb[3] / type_ਮਾਪ), src1_d + i3 * (src1->nb[3] / type_ਮਾਪ),
                                dst_d + i3 * (dst->nb[3] / type_ਮਾਪ), src0->ne[0], src0->ne[1], src0->ne[2], dst->ne[0],
                                dst->ne[1], dst->ne[2], dim, stream);
            }
        } ਨਹੀਂਤਾਂ {
            ਅਚਲ size_t size0 = ggml_nbytes(src0);
            ਅਚਲ size_t size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->ਸਮ੍ਰਿਤੀ_ਨਕਲ(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->ਸਮ੍ਰਿਤੀ_ਨਕਲ(dst_d + size0 / type_ਮਾਪ, src1_d, size1)));
        }
    } ਨਹੀਂਤਾਂ {
        concat_T_sycl_non_cont<T>(stream, (ਅਚਲ ਅੱਖਰ *) src0->ਅੰਕੜੇ, (ਅਚਲ ਅੱਖਰ *) src1->ਅੰਕੜੇ, (ਅੱਖਰ *) dst->ਅੰਕੜੇ,
                                 src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src0->nb[0], src0->nb[1],
                                 src0->nb[2], src0->nb[3], src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                                 src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3], dst->ne[0], dst->ne[1], dst->ne[2],
                                 dst->ne[3], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

ਸਥਿਰ ਖਾਲੀ concat_impl_q4_0_sycl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ਅਚਲ ggml_tensor * src0   = dst->src[0];
    ਅਚਲ ggml_tensor * src1   = dst->src[1];
    ਕਤਾਰ_ptr           stream = ctx.stream();

    ਅਚਲ int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(src1->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(dst->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(src0->ne[0] % QK4_0 == 0);
    GGML_ASSERT(src1->ne[0] % QK4_0 == 0);
    GGML_ASSERT(dst->ne[0] % QK4_0 == 0);

    ਅਚਲ ਪੂਰਨ ne00_blk = src0->ne[0] / QK4_0;
    ਅਚਲ ਪੂਰਨ ne0_blk  = dst->ne[0] / QK4_0;

    ਜੇ (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        ਅਚਲ ਖੰਡ_q4_0 * src0_d = (ਅਚਲ ਖੰਡ_q4_0 *) src0->ਅੰਕੜੇ;
        ਅਚਲ ਖੰਡ_q4_0 * src1_d = (ਅਚਲ ਖੰਡ_q4_0 *) src1->ਅੰਕੜੇ;
        ਖੰਡ_q4_0 * dst_d        = (ਖੰਡ_q4_0 *) dst->ਅੰਕੜੇ;
        ਅਚਲ size_t type_ਮਾਪ    = ਆਕਾਰ(ਖੰਡ_q4_0);

        ਜੇ (dim != 3) {
            ਲਈ (ਪੂਰਨ i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<ਖੰਡ_q4_0>(
                    src0_d + i3 * (src0->nb[3] / type_ਮਾਪ),
                    src1_d + i3 * (src1->nb[3] / type_ਮਾਪ),
                    dst_d + i3 * (dst->nb[3] / type_ਮਾਪ),
                    ne00_blk, src0->ne[1], src0->ne[2], ne0_blk,
                    dst->ne[1], dst->ne[2], dim, stream);
            }
        } ਨਹੀਂਤਾਂ {
            ਅਚਲ size_t size0 = ggml_nbytes(src0);
            ਅਚਲ size_t size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->ਸਮ੍ਰਿਤੀ_ਨਕਲ(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->ਸਮ੍ਰਿਤੀ_ਨਕਲ((ਅੱਖਰ *) dst_d + size0, src1_d, size1)));
        }
    } ਨਹੀਂਤਾਂ {
        concat_T_sycl_non_cont<ਖੰਡ_q4_0>(
            stream, (ਅਚਲ ਅੱਖਰ *) src0->ਅੰਕੜੇ, (ਅਚਲ ਅੱਖਰ *) src1->ਅੰਕੜੇ,
            (ਅੱਖਰ *) dst->ਅੰਕੜੇ,
            ne00_blk, src0->ne[1], src0->ne[2], src0->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->ne[0] / QK4_0, src1->ne[1], src1->ne[2], src1->ne[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            ne0_blk, dst->ne[1], dst->ne[2], dst->ne[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

ਸਥਿਰ ਖਾਲੀ concat_impl_q4_1_sycl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ਅਚਲ ggml_tensor * src0   = dst->src[0];
    ਅਚਲ ggml_tensor * src1   = dst->src[1];
    ਕਤਾਰ_ptr           stream = ctx.stream();

    ਅਚਲ int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(src1->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(dst->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(src0->ne[0] % QK4_1 == 0);
    GGML_ASSERT(src1->ne[0] % QK4_1 == 0);
    GGML_ASSERT(dst->ne[0] % QK4_1 == 0);

    ਅਚਲ ਪੂਰਨ ne00_blk = src0->ne[0] / QK4_1;
    ਅਚਲ ਪੂਰਨ ne0_blk  = dst->ne[0] / QK4_1;

    ਜੇ (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        ਅਚਲ ਖੰਡ_q4_1 * src0_d = (ਅਚਲ ਖੰਡ_q4_1 *) src0->ਅੰਕੜੇ;
        ਅਚਲ ਖੰਡ_q4_1 * src1_d = (ਅਚਲ ਖੰਡ_q4_1 *) src1->ਅੰਕੜੇ;
        ਖੰਡ_q4_1 * dst_d        = (ਖੰਡ_q4_1 *) dst->ਅੰਕੜੇ;
        ਅਚਲ size_t type_ਮਾਪ    = ਆਕਾਰ(ਖੰਡ_q4_1);

        ਜੇ (dim != 3) {
            ਲਈ (ਪੂਰਨ i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<ਖੰਡ_q4_1>(
                    src0_d + i3 * (src0->nb[3] / type_ਮਾਪ),
                    src1_d + i3 * (src1->nb[3] / type_ਮਾਪ),
                    dst_d + i3 * (dst->nb[3] / type_ਮਾਪ),
                    ne00_blk, src0->ne[1], src0->ne[2], ne0_blk,
                    dst->ne[1], dst->ne[2], dim, stream);
            }
        } ਨਹੀਂਤਾਂ {
            ਅਚਲ size_t size0 = ggml_nbytes(src0);
            ਅਚਲ size_t size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->ਸਮ੍ਰਿਤੀ_ਨਕਲ(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->ਸਮ੍ਰਿਤੀ_ਨਕਲ((ਅੱਖਰ *) dst_d + size0, src1_d, size1)));
        }
    } ਨਹੀਂਤਾਂ {
        concat_T_sycl_non_cont<ਖੰਡ_q4_1>(
            stream, (ਅਚਲ ਅੱਖਰ *) src0->ਅੰਕੜੇ, (ਅਚਲ ਅੱਖਰ *) src1->ਅੰਕੜੇ,
            (ਅੱਖਰ *) dst->ਅੰਕੜੇ,
            ne00_blk, src0->ne[1], src0->ne[2], src0->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->ne[0] / QK4_1, src1->ne[1], src1->ne[2], src1->ne[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            ne0_blk, dst->ne[1], dst->ne[2], dst->ne[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

ਸਥਿਰ ਖਾਲੀ concat_impl_q5_0_sycl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ਅਚਲ ggml_tensor * src0   = dst->src[0];
    ਅਚਲ ggml_tensor * src1   = dst->src[1];
    ਕਤਾਰ_ptr           stream = ctx.stream();

    ਅਚਲ int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_Q5_0);
    GGML_ASSERT(src1->type == GGML_TYPE_Q5_0);
    GGML_ASSERT(dst->type == GGML_TYPE_Q5_0);
    GGML_ASSERT(src0->ne[0] % QK5_0 == 0);
    GGML_ASSERT(src1->ne[0] % QK5_0 == 0);
    GGML_ASSERT(dst->ne[0] % QK5_0 == 0);

    ਅਚਲ ਪੂਰਨ ne00_blk = src0->ne[0] / QK5_0;
    ਅਚਲ ਪੂਰਨ ne0_blk  = dst->ne[0] / QK5_0;

    ਜੇ (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        ਅਚਲ ਖੰਡ_q5_0 * src0_d = (ਅਚਲ ਖੰਡ_q5_0 *) src0->ਅੰਕੜੇ;
        ਅਚਲ ਖੰਡ_q5_0 * src1_d = (ਅਚਲ ਖੰਡ_q5_0 *) src1->ਅੰਕੜੇ;
        ਖੰਡ_q5_0 * dst_d        = (ਖੰਡ_q5_0 *) dst->ਅੰਕੜੇ;
        ਅਚਲ size_t type_ਮਾਪ    = ਆਕਾਰ(ਖੰਡ_q5_0);

        ਜੇ (dim != 3) {
            ਲਈ (ਪੂਰਨ i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<ਖੰਡ_q5_0>(
                    src0_d + i3 * (src0->nb[3] / type_ਮਾਪ),
                    src1_d + i3 * (src1->nb[3] / type_ਮਾਪ),
                    dst_d + i3 * (dst->nb[3] / type_ਮਾਪ),
                    ne00_blk, src0->ne[1], src0->ne[2], ne0_blk,
                    dst->ne[1], dst->ne[2], dim, stream);
            }
        } ਨਹੀਂਤਾਂ {
            ਅਚਲ size_t size0 = ggml_nbytes(src0);
            ਅਚਲ size_t size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->ਸਮ੍ਰਿਤੀ_ਨਕਲ(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->ਸਮ੍ਰਿਤੀ_ਨਕਲ((ਅੱਖਰ *) dst_d + size0, src1_d, size1)));
        }
    } ਨਹੀਂਤਾਂ {
        concat_T_sycl_non_cont<ਖੰਡ_q5_0>(
            stream, (ਅਚਲ ਅੱਖਰ *) src0->ਅੰਕੜੇ, (ਅਚਲ ਅੱਖਰ *) src1->ਅੰਕੜੇ,
            (ਅੱਖਰ *) dst->ਅੰਕੜੇ,
            ne00_blk, src0->ne[1], src0->ne[2], src0->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->ne[0] / QK5_0, src1->ne[1], src1->ne[2], src1->ne[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            ne0_blk, dst->ne[1], dst->ne[2], dst->ne[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

ਸਥਿਰ ਖਾਲੀ concat_impl_q5_1_sycl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ਅਚਲ ggml_tensor * src0   = dst->src[0];
    ਅਚਲ ggml_tensor * src1   = dst->src[1];
    ਕਤਾਰ_ptr           stream = ctx.stream();

    ਅਚਲ int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_Q5_1);
    GGML_ASSERT(src1->type == GGML_TYPE_Q5_1);
    GGML_ASSERT(dst->type == GGML_TYPE_Q5_1);
    GGML_ASSERT(src0->ne[0] % QK5_1 == 0);
    GGML_ASSERT(src1->ne[0] % QK5_1 == 0);
    GGML_ASSERT(dst->ne[0] % QK5_1 == 0);

    ਅਚਲ ਪੂਰਨ ne00_blk = src0->ne[0] / QK5_1;
    ਅਚਲ ਪੂਰਨ ne0_blk  = dst->ne[0] / QK5_1;

    ਜੇ (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        ਅਚਲ ਖੰਡ_q5_1 * src0_d = (ਅਚਲ ਖੰਡ_q5_1 *) src0->ਅੰਕੜੇ;
        ਅਚਲ ਖੰਡ_q5_1 * src1_d = (ਅਚਲ ਖੰਡ_q5_1 *) src1->ਅੰਕੜੇ;
        ਖੰਡ_q5_1 * dst_d        = (ਖੰਡ_q5_1 *) dst->ਅੰਕੜੇ;
        ਅਚਲ size_t type_ਮਾਪ    = ਆਕਾਰ(ਖੰਡ_q5_1);

        ਜੇ (dim != 3) {
            ਲਈ (ਪੂਰਨ i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<ਖੰਡ_q5_1>(
                    src0_d + i3 * (src0->nb[3] / type_ਮਾਪ),
                    src1_d + i3 * (src1->nb[3] / type_ਮਾਪ),
                    dst_d + i3 * (dst->nb[3] / type_ਮਾਪ),
                    ne00_blk, src0->ne[1], src0->ne[2], ne0_blk,
                    dst->ne[1], dst->ne[2], dim, stream);
            }
        } ਨਹੀਂਤਾਂ {
            ਅਚਲ size_t size0 = ggml_nbytes(src0);
            ਅਚਲ size_t size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->ਸਮ੍ਰਿਤੀ_ਨਕਲ(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->ਸਮ੍ਰਿਤੀ_ਨਕਲ((ਅੱਖਰ *) dst_d + size0, src1_d, size1)));
        }
    } ਨਹੀਂਤਾਂ {
        concat_T_sycl_non_cont<ਖੰਡ_q5_1>(
            stream, (ਅਚਲ ਅੱਖਰ *) src0->ਅੰਕੜੇ, (ਅਚਲ ਅੱਖਰ *) src1->ਅੰਕੜੇ,
            (ਅੱਖਰ *) dst->ਅੰਕੜੇ,
            ne00_blk, src0->ne[1], src0->ne[2], src0->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->ne[0] / QK5_1, src1->ne[1], src1->ne[2], src1->ne[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            ne0_blk, dst->ne[1], dst->ne[2], dst->ne[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

ਸਥਿਰ ਖਾਲੀ concat_impl_q8_0_sycl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ਅਚਲ ggml_tensor * src0   = dst->src[0];
    ਅਚਲ ggml_tensor * src1   = dst->src[1];
    ਕਤਾਰ_ptr           stream = ctx.stream();

    ਅਚਲ int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(src1->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(dst->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(src0->ne[0] % QK8_0 == 0);
    GGML_ASSERT(src1->ne[0] % QK8_0 == 0);
    GGML_ASSERT(dst->ne[0] % QK8_0 == 0);

    ਅਚਲ ਪੂਰਨ ne00_blk = src0->ne[0] / QK8_0;
    ਅਚਲ ਪੂਰਨ ne0_blk  = dst->ne[0] / QK8_0;

    ਜੇ (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        ਅਚਲ ਖੰਡ_q8_0 * src0_d = (ਅਚਲ ਖੰਡ_q8_0 *) src0->ਅੰਕੜੇ;
        ਅਚਲ ਖੰਡ_q8_0 * src1_d = (ਅਚਲ ਖੰਡ_q8_0 *) src1->ਅੰਕੜੇ;
        ਖੰਡ_q8_0 * dst_d        = (ਖੰਡ_q8_0 *) dst->ਅੰਕੜੇ;
        ਅਚਲ size_t type_ਮਾਪ    = ਆਕਾਰ(ਖੰਡ_q8_0);

        ਜੇ (dim != 3) {
            ਲਈ (ਪੂਰਨ i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<ਖੰਡ_q8_0>(
                    src0_d + i3 * (src0->nb[3] / type_ਮਾਪ),
                    src1_d + i3 * (src1->nb[3] / type_ਮਾਪ),
                    dst_d + i3 * (dst->nb[3] / type_ਮਾਪ),
                    ne00_blk, src0->ne[1], src0->ne[2], ne0_blk,
                    dst->ne[1], dst->ne[2], dim, stream);
            }
        } ਨਹੀਂਤਾਂ {
            ਅਚਲ size_t size0 = ggml_nbytes(src0);
            ਅਚਲ size_t size1 = ggml_nbytes(src1);
            SYCL_CHECK(CHECK_TRY_ERROR(stream->ਸਮ੍ਰਿਤੀ_ਨਕਲ(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->ਸਮ੍ਰਿਤੀ_ਨਕਲ((ਅੱਖਰ *) dst_d + size0, src1_d, size1)));
        }
    } ਨਹੀਂਤਾਂ {
        concat_T_sycl_non_cont<ਖੰਡ_q8_0>(
            stream, (ਅਚਲ ਅੱਖਰ *) src0->ਅੰਕੜੇ, (ਅਚਲ ਅੱਖਰ *) src1->ਅੰਕੜੇ,
            (ਅੱਖਰ *) dst->ਅੰਕੜੇ,
            ne00_blk, src0->ne[1], src0->ne[2], src0->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->ne[0] / QK8_0, src1->ne[1], src1->ne[2], src1->ne[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            ne0_blk, dst->ne[1], dst->ne[2], dst->ne[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

ਖਾਲੀ ggml_sycl_op_concat(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ਚੋਣ (dst->type) {
    ਹਾਲਤ GGML_TYPE_F32:
        concat_impl_sycl<ਦਸ਼ਮਲਵ>(ctx, dst);
        ਰੋਕੋ;
    ਹਾਲਤ GGML_TYPE_F16:
        concat_impl_sycl<sycl::half>(ctx, dst);
        ਰੋਕੋ;
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_HAS_BF16
    ਹਾਲਤ GGML_TYPE_BF16:
        concat_impl_sycl<sycl::ext::oneapi::bfloat16>(ctx, dst);
        ਰੋਕੋ;
#ਜੇ_ਅੰਤ
    ਹਾਲਤ GGML_TYPE_I32:
        concat_impl_sycl<int32_t>(ctx, dst);
        ਰੋਕੋ;
    ਹਾਲਤ GGML_TYPE_I16:
        concat_impl_sycl<int16_t>(ctx, dst);
        ਰੋਕੋ;
    ਹਾਲਤ GGML_TYPE_I64:
        concat_impl_sycl<int64_t>(ctx, dst);
        ਰੋਕੋ;
    ਹਾਲਤ GGML_TYPE_I8:
        concat_impl_sycl<int8_t>(ctx, dst);
        ਰੋਕੋ;
    ਹਾਲਤ GGML_TYPE_Q4_0:
        concat_impl_q4_0_sycl(ctx, dst);
        ਰੋਕੋ;
    ਹਾਲਤ GGML_TYPE_Q4_1:
        concat_impl_q4_1_sycl(ctx, dst);
        ਰੋਕੋ;
    ਹਾਲਤ GGML_TYPE_Q5_0:
        concat_impl_q5_0_sycl(ctx, dst);
        ਰੋਕੋ;
    ਹਾਲਤ GGML_TYPE_Q5_1:
        concat_impl_q5_1_sycl(ctx, dst);
        ਰੋਕੋ;
    ਹਾਲਤ GGML_TYPE_Q8_0:
        concat_impl_q8_0_sycl(ctx, dst);
        ਰੋਕੋ;
    ਮੂਲ:
        fprintf(stderr, "%s: unsupported types: dst: %s\n", __func__, ggml_type_name(dst->type));
        GGML_ASSERT(ਝੂਠ);
    }
}
