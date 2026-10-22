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

#ಅನ್ತರ್ಭಾವ "ggml.h"

#ಅನ್ತರ್ಭಾವ "concat.hpp"

ಸ್ಥಿರ ಸಾಲೊಳಗೆ size_t elem_ಗಾತ್ರ(ggml_type t) {
    ಹಿಂತಿರುಗಿ ggml_type_ಗಾತ್ರ(t) / ggml_blck_ಗಾತ್ರ(t);
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ ಶೂನ್ಯ concat_T_dim0(ಸ್ಥಿರಾಂಕ T *x, ಸ್ಥಿರಾಂಕ T *y, T *dst,
                            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne0, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne00,
                            ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {
  ಪೂರ್ಣಾಂಕ nidx = ಅಂಶ_ct1.get_local_id(2) +
             ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(2);
  ಆದರೆ (nidx >= ne0) {
    ಹಿಂತಿರುಗಿ;
  }
  // operation
  ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ_dst = nidx + ಅಂಶ_ct1.get_ಗುಂಪು(1) * ne0 +
                   ಅಂಶ_ct1.get_ಗುಂಪು(0) * ne0 * ಅಂಶ_ct1.get_ಗುಂಪು_range(1);
  ಆದರೆ (nidx < ne00) { // src0
    ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ_src = nidx + ಅಂಶ_ct1.get_ಗುಂಪು(1) * ne00 +
                     ಅಂಶ_ct1.get_ಗುಂಪು(0) * ne00 * ಅಂಶ_ct1.get_ಗುಂಪು_range(1);
    dst[ಸ್ಥಾನಾಂತರ_dst] = x[ಸ್ಥಾನಾಂತರ_src];
  } ಇಲ್ಲದಿದ್ದರೆ {
    ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ_src =
        nidx - ne00 + ಅಂಶ_ct1.get_ಗುಂಪು(1) * (ne0 - ne00) +
        ಅಂಶ_ct1.get_ಗುಂಪು(0) * (ne0 - ne00) * ಅಂಶ_ct1.get_ಗುಂಪು_range(1);
    dst[ಸ್ಥಾನಾಂತರ_dst] = y[ಸ್ಥಾನಾಂತರ_src];
  }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ ಶೂನ್ಯ concat_T_dim1(ಸ್ಥಿರಾಂಕ T *x, ಸ್ಥಿರಾಂಕ T *y, T *dst,
                            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne0, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne01,
                            ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {
  ಪೂರ್ಣಾಂಕ nidx = ಅಂಶ_ct1.get_local_id(2) +
             ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(2);
  ಆದರೆ (nidx >= ne0) {
    ಹಿಂತಿರುಗಿ;
  }
  // operation
  ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ_dst = nidx + ಅಂಶ_ct1.get_ಗುಂಪು(1) * ne0 +
                   ಅಂಶ_ct1.get_ಗುಂಪು(0) * ne0 * ಅಂಶ_ct1.get_ಗುಂಪು_range(1);
  ಆದರೆ (ಅಂಶ_ct1.get_ಗುಂಪು(1) < (size_t) ne01) { // src0
    ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ_src =
        nidx + ಅಂಶ_ct1.get_ಗುಂಪು(1) * ne0 + ಅಂಶ_ct1.get_ಗುಂಪು(0) * ne0 * ne01;
    dst[ಸ್ಥಾನಾಂತರ_dst] = x[ಸ್ಥಾನಾಂತರ_src];
  } ಇಲ್ಲದಿದ್ದರೆ {
    ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ_src =
        nidx + (ಅಂಶ_ct1.get_ಗುಂಪು(1) - ne01) * ne0 +
        ಅಂಶ_ct1.get_ಗುಂಪು(0) * ne0 * (ಅಂಶ_ct1.get_ಗುಂಪು_range(1) - ne01);
    dst[ಸ್ಥಾನಾಂತರ_dst] = y[ಸ್ಥಾನಾಂತರ_src];
  }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ ಶೂನ್ಯ concat_T_dim2(ಸ್ಥಿರಾಂಕ T *x, ಸ್ಥಿರಾಂಕ T *y, T *dst,
                            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne0, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne02,
                            ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {
  ಪೂರ್ಣಾಂಕ nidx = ಅಂಶ_ct1.get_local_id(2) +
             ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(2);
  ಆದರೆ (nidx >= ne0) {
    ಹಿಂತಿರುಗಿ;
  }
  // operation
  ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ_dst = nidx + ಅಂಶ_ct1.get_ಗುಂಪು(1) * ne0 +
                   ಅಂಶ_ct1.get_ಗುಂಪು(0) * ne0 * ಅಂಶ_ct1.get_ಗುಂಪು_range(1);
  ಆದರೆ (ಅಂಶ_ct1.get_ಗುಂಪು(0) < (size_t) ne02) { // src0
    ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ_src = nidx + ಅಂಶ_ct1.get_ಗುಂಪು(1) * ne0 +
                     ಅಂಶ_ct1.get_ಗುಂಪು(0) * ne0 * ಅಂಶ_ct1.get_ಗುಂಪು_range(1);
    dst[ಸ್ಥಾನಾಂತರ_dst] = x[ಸ್ಥಾನಾಂತರ_src];
  } ಇಲ್ಲದಿದ್ದರೆ {
    ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ_src =
        nidx + ಅಂಶ_ct1.get_ಗುಂಪು(1) * ne0 +
        (ಅಂಶ_ct1.get_ಗುಂಪು(0) - ne02) * ne0 * ಅಂಶ_ct1.get_ಗುಂಪು_range(1);
    dst[ಸ್ಥಾನಾಂತರ_dst] = y[ಸ್ಥಾನಾಂತರ_src];
  }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ ಶೂನ್ಯ concat_T_sycl(ಸ್ಥಿರಾಂಕ T *x, ಸ್ಥಿರಾಂಕ T *y, T *dst,
                            ಪೂರ್ಣಾಂಕ ne00, ಪೂರ್ಣಾಂಕ ne01, ಪೂರ್ಣಾಂಕ ne02, ಪೂರ್ಣಾಂಕ ne0, ಪೂರ್ಣಾಂಕ ne1,
                            ಪೂರ್ಣಾಂಕ ne2, ಪೂರ್ಣಾಂಕ dim, ಸರತಿ_ptr stream) {
  ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಖಂಡಗಳು = (ne0 + SYCL_CONCAT_ಖಂಡ_ಗಾತ್ರ - 1) / SYCL_CONCAT_ಖಂಡ_ಗಾತ್ರ;
  sycl::range<3> gridDim(ne2, ne1, ಅಂಕೆ_ಖಂಡಗಳು);
  ಆಯ್ಕೆ (dim) {
  ಸ್ಥಿತಿ 0:
      stream->parallel_for(sycl::nd_range<3>(gridDim * sycl::range<3>(1, 1, SYCL_CONCAT_ಖಂಡ_ಗಾತ್ರ),
                                          sycl::range<3>(1, 1, SYCL_CONCAT_ಖಂಡ_ಗಾತ್ರ)),
                        [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) { concat_T_dim0<T>(x, y, dst, ne0, ne00, ಅಂಶ_ct1); });
      ನಿಲ್ಲಿಸು;
  ಸ್ಥಿತಿ 1:
      stream->parallel_for(sycl::nd_range<3>(gridDim * sycl::range<3>(1, 1, SYCL_CONCAT_ಖಂಡ_ಗಾತ್ರ),
                                          sycl::range<3>(1, 1, SYCL_CONCAT_ಖಂಡ_ಗಾತ್ರ)),
                        [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) { concat_T_dim1<T>(x, y, dst, ne0, ne01, ಅಂಶ_ct1); });
      ನಿಲ್ಲಿಸು;
  // dim >=2 will be dispatched to the default path
  ಪೂರ್ವನಿಯೋಜಿತ:
      stream->parallel_for(sycl::nd_range<3>(gridDim * sycl::range<3>(1, 1, SYCL_CONCAT_ಖಂಡ_ಗಾತ್ರ),
                                          sycl::range<3>(1, 1, SYCL_CONCAT_ಖಂಡ_ಗಾತ್ರ)),
                        [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) { concat_T_dim2<T>(x, y, dst, ne0, ne02, ಅಂಶ_ct1); });
      ನಿಲ್ಲಿಸು;
  }
}

// non-contiguous kernel (slow)
ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ ಶೂನ್ಯ concat_T_sycl_non_cont(
    ಸರತಿ_ptr stream, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *src0, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *src1, ಅಕ್ಷರ *dst,
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
  ಸ್ಥಿರಾಂಕ int64_t ne0_pad   = GGML_PAD(ne0, WARP_ಗಾತ್ರ);
  ಸ್ಥಿರಾಂಕ int64_t ಖಂಡ_ne0 = ne0_pad < SYCL_CONCAT_ಖಂಡ_ಗಾತ್ರ ? ne0_pad : (int64_t) SYCL_CONCAT_ಖಂಡ_ಗಾತ್ರ;
  sycl::range<3> ಖಂಡ_Dim(1, 1, ಖಂಡ_ne0);

  stream->parallel_for(sycl::nd_range<3>(gridDim * ಖಂಡ_Dim, ಖಂಡ_Dim), [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
      int64_t i3 = ಅಂಶ_ct1.get_ಗುಂಪು(0);
      int64_t i2 = ಅಂಶ_ct1.get_ಗುಂಪು(1);
      int64_t i1 = ಅಂಶ_ct1.get_ಗುಂಪು(2);

      int64_t o[4] = { 0, 0, 0, 0 };
      o[dim]       = dim == 0 ? ne00 : (dim == 1 ? ne01 : (dim == 2 ? ne02 : ne03));

      ಸ್ಥಿರಾಂಕ T * x;

      ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = ಅಂಶ_ct1.get_local_id(2); i0 < ne0; i0 += ಅಂಶ_ct1.get_local_range(2)) {
          ಆದರೆ (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
              x = (ಸ್ಥಿರಾಂಕ T *) (src0 + (i3) *nb03 + (i2) *nb02 + (i1) *nb01 + (i0) *nb00);
          } ಇಲ್ಲದಿದ್ದರೆ {
              x = (ಸ್ಥಿರಾಂಕ T *) (src1 + (i3 - o[3]) * nb13 + (i2 - o[2]) * nb12 + (i1 - o[1]) * nb11 +
                                   (i0 - o[0]) * nb10);
          }

          T *y = (T *)(dst + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

          *y = *x;
      }
  });
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
ಶೂನ್ಯ concat_impl_sycl(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ಸ್ಥಿರಾಂಕ ggml_tensor *  src0   = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor *  src1   = dst->src[1];
    ಸರತಿ_ptr            stream = ctx.stream();

    ಸ್ಥಿರಾಂಕ int32_t dim = ((int32_t *) dst->op_params)[0];

    ಆದರೆ (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        ಸ್ಥಿರಾಂಕ T * src0_d = (ಸ್ಥಿರಾಂಕ T *) src0->ದತ್ತಾಂಶ;
        ಸ್ಥಿರಾಂಕ T * src1_d = (ಸ್ಥಿರಾಂಕ T *) src1->ದತ್ತಾಂಶ;
        T * dst_d = (T *) dst->ದತ್ತಾಂಶ;
        size_t type_ಗಾತ್ರ = elem_ಗಾತ್ರ(dst->type);
        ಆದರೆ (dim != 3) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<T>(src0_d + i3 * (src0->nb[3] / type_ಗಾತ್ರ), src1_d + i3 * (src1->nb[3] / type_ಗಾತ್ರ),
                                dst_d + i3 * (dst->nb[3] / type_ಗಾತ್ರ), src0->ne[0], src0->ne[1], src0->ne[2], dst->ne[0],
                                dst->ne[1], dst->ne[2], dim, stream);
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಸ್ಥಿರಾಂಕ size_t size0 = ggml_nbytes(src0);
            ಸ್ಥಿರಾಂಕ size_t size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->ಸ್ಮರಣೆ_ನಕಲು(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->ಸ್ಮರಣೆ_ನಕಲು(dst_d + size0 / type_ಗಾತ್ರ, src1_d, size1)));
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        concat_T_sycl_non_cont<T>(stream, (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ, (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ, (ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ,
                                 src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src0->nb[0], src0->nb[1],
                                 src0->nb[2], src0->nb[3], src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                                 src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3], dst->ne[0], dst->ne[1], dst->ne[2],
                                 dst->ne[3], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ concat_impl_q4_0_sycl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0   = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1   = dst->src[1];
    ಸರತಿ_ptr           stream = ctx.stream();

    ಸ್ಥಿರಾಂಕ int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(src1->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(dst->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(src0->ne[0] % QK4_0 == 0);
    GGML_ASSERT(src1->ne[0] % QK4_0 == 0);
    GGML_ASSERT(dst->ne[0] % QK4_0 == 0);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne00_blk = src0->ne[0] / QK4_0;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne0_blk  = dst->ne[0] / QK4_0;

    ಆದರೆ (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 * src0_d = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 *) src0->ದತ್ತಾಂಶ;
        ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 * src1_d = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 *) src1->ದತ್ತಾಂಶ;
        ಖಂಡ_q4_0 * dst_d        = (ಖಂಡ_q4_0 *) dst->ದತ್ತಾಂಶ;
        ಸ್ಥಿರಾಂಕ size_t type_ಗಾತ್ರ    = ಗಾತ್ರ(ಖಂಡ_q4_0);

        ಆದರೆ (dim != 3) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<ಖಂಡ_q4_0>(
                    src0_d + i3 * (src0->nb[3] / type_ಗಾತ್ರ),
                    src1_d + i3 * (src1->nb[3] / type_ಗಾತ್ರ),
                    dst_d + i3 * (dst->nb[3] / type_ಗಾತ್ರ),
                    ne00_blk, src0->ne[1], src0->ne[2], ne0_blk,
                    dst->ne[1], dst->ne[2], dim, stream);
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಸ್ಥಿರಾಂಕ size_t size0 = ggml_nbytes(src0);
            ಸ್ಥಿರಾಂಕ size_t size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->ಸ್ಮರಣೆ_ನಕಲು(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->ಸ್ಮರಣೆ_ನಕಲು((ಅಕ್ಷರ *) dst_d + size0, src1_d, size1)));
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        concat_T_sycl_non_cont<ಖಂಡ_q4_0>(
            stream, (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ, (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ,
            (ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ,
            ne00_blk, src0->ne[1], src0->ne[2], src0->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->ne[0] / QK4_0, src1->ne[1], src1->ne[2], src1->ne[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            ne0_blk, dst->ne[1], dst->ne[2], dst->ne[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ concat_impl_q4_1_sycl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0   = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1   = dst->src[1];
    ಸರತಿ_ptr           stream = ctx.stream();

    ಸ್ಥಿರಾಂಕ int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(src1->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(dst->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(src0->ne[0] % QK4_1 == 0);
    GGML_ASSERT(src1->ne[0] % QK4_1 == 0);
    GGML_ASSERT(dst->ne[0] % QK4_1 == 0);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne00_blk = src0->ne[0] / QK4_1;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne0_blk  = dst->ne[0] / QK4_1;

    ಆದರೆ (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_1 * src0_d = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_1 *) src0->ದತ್ತಾಂಶ;
        ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_1 * src1_d = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_1 *) src1->ದತ್ತಾಂಶ;
        ಖಂಡ_q4_1 * dst_d        = (ಖಂಡ_q4_1 *) dst->ದತ್ತಾಂಶ;
        ಸ್ಥಿರಾಂಕ size_t type_ಗಾತ್ರ    = ಗಾತ್ರ(ಖಂಡ_q4_1);

        ಆದರೆ (dim != 3) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<ಖಂಡ_q4_1>(
                    src0_d + i3 * (src0->nb[3] / type_ಗಾತ್ರ),
                    src1_d + i3 * (src1->nb[3] / type_ಗಾತ್ರ),
                    dst_d + i3 * (dst->nb[3] / type_ಗಾತ್ರ),
                    ne00_blk, src0->ne[1], src0->ne[2], ne0_blk,
                    dst->ne[1], dst->ne[2], dim, stream);
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಸ್ಥಿರಾಂಕ size_t size0 = ggml_nbytes(src0);
            ಸ್ಥಿರಾಂಕ size_t size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->ಸ್ಮರಣೆ_ನಕಲು(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->ಸ್ಮರಣೆ_ನಕಲು((ಅಕ್ಷರ *) dst_d + size0, src1_d, size1)));
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        concat_T_sycl_non_cont<ಖಂಡ_q4_1>(
            stream, (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ, (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ,
            (ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ,
            ne00_blk, src0->ne[1], src0->ne[2], src0->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->ne[0] / QK4_1, src1->ne[1], src1->ne[2], src1->ne[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            ne0_blk, dst->ne[1], dst->ne[2], dst->ne[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ concat_impl_q5_0_sycl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0   = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1   = dst->src[1];
    ಸರತಿ_ptr           stream = ctx.stream();

    ಸ್ಥಿರಾಂಕ int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_Q5_0);
    GGML_ASSERT(src1->type == GGML_TYPE_Q5_0);
    GGML_ASSERT(dst->type == GGML_TYPE_Q5_0);
    GGML_ASSERT(src0->ne[0] % QK5_0 == 0);
    GGML_ASSERT(src1->ne[0] % QK5_0 == 0);
    GGML_ASSERT(dst->ne[0] % QK5_0 == 0);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne00_blk = src0->ne[0] / QK5_0;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne0_blk  = dst->ne[0] / QK5_0;

    ಆದರೆ (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_0 * src0_d = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_0 *) src0->ದತ್ತಾಂಶ;
        ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_0 * src1_d = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_0 *) src1->ದತ್ತಾಂಶ;
        ಖಂಡ_q5_0 * dst_d        = (ಖಂಡ_q5_0 *) dst->ದತ್ತಾಂಶ;
        ಸ್ಥಿರಾಂಕ size_t type_ಗಾತ್ರ    = ಗಾತ್ರ(ಖಂಡ_q5_0);

        ಆದರೆ (dim != 3) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<ಖಂಡ_q5_0>(
                    src0_d + i3 * (src0->nb[3] / type_ಗಾತ್ರ),
                    src1_d + i3 * (src1->nb[3] / type_ಗಾತ್ರ),
                    dst_d + i3 * (dst->nb[3] / type_ಗಾತ್ರ),
                    ne00_blk, src0->ne[1], src0->ne[2], ne0_blk,
                    dst->ne[1], dst->ne[2], dim, stream);
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಸ್ಥಿರಾಂಕ size_t size0 = ggml_nbytes(src0);
            ಸ್ಥಿರಾಂಕ size_t size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->ಸ್ಮರಣೆ_ನಕಲು(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->ಸ್ಮರಣೆ_ನಕಲು((ಅಕ್ಷರ *) dst_d + size0, src1_d, size1)));
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        concat_T_sycl_non_cont<ಖಂಡ_q5_0>(
            stream, (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ, (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ,
            (ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ,
            ne00_blk, src0->ne[1], src0->ne[2], src0->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->ne[0] / QK5_0, src1->ne[1], src1->ne[2], src1->ne[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            ne0_blk, dst->ne[1], dst->ne[2], dst->ne[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ concat_impl_q5_1_sycl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0   = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1   = dst->src[1];
    ಸರತಿ_ptr           stream = ctx.stream();

    ಸ್ಥಿರಾಂಕ int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_Q5_1);
    GGML_ASSERT(src1->type == GGML_TYPE_Q5_1);
    GGML_ASSERT(dst->type == GGML_TYPE_Q5_1);
    GGML_ASSERT(src0->ne[0] % QK5_1 == 0);
    GGML_ASSERT(src1->ne[0] % QK5_1 == 0);
    GGML_ASSERT(dst->ne[0] % QK5_1 == 0);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne00_blk = src0->ne[0] / QK5_1;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne0_blk  = dst->ne[0] / QK5_1;

    ಆದರೆ (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_1 * src0_d = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_1 *) src0->ದತ್ತಾಂಶ;
        ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_1 * src1_d = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_1 *) src1->ದತ್ತಾಂಶ;
        ಖಂಡ_q5_1 * dst_d        = (ಖಂಡ_q5_1 *) dst->ದತ್ತಾಂಶ;
        ಸ್ಥಿರಾಂಕ size_t type_ಗಾತ್ರ    = ಗಾತ್ರ(ಖಂಡ_q5_1);

        ಆದರೆ (dim != 3) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<ಖಂಡ_q5_1>(
                    src0_d + i3 * (src0->nb[3] / type_ಗಾತ್ರ),
                    src1_d + i3 * (src1->nb[3] / type_ಗಾತ್ರ),
                    dst_d + i3 * (dst->nb[3] / type_ಗಾತ್ರ),
                    ne00_blk, src0->ne[1], src0->ne[2], ne0_blk,
                    dst->ne[1], dst->ne[2], dim, stream);
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಸ್ಥಿರಾಂಕ size_t size0 = ggml_nbytes(src0);
            ಸ್ಥಿರಾಂಕ size_t size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->ಸ್ಮರಣೆ_ನಕಲು(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->ಸ್ಮರಣೆ_ನಕಲು((ಅಕ್ಷರ *) dst_d + size0, src1_d, size1)));
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        concat_T_sycl_non_cont<ಖಂಡ_q5_1>(
            stream, (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ, (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ,
            (ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ,
            ne00_blk, src0->ne[1], src0->ne[2], src0->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->ne[0] / QK5_1, src1->ne[1], src1->ne[2], src1->ne[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            ne0_blk, dst->ne[1], dst->ne[2], dst->ne[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ concat_impl_q8_0_sycl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0   = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1   = dst->src[1];
    ಸರತಿ_ptr           stream = ctx.stream();

    ಸ್ಥಿರಾಂಕ int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(src1->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(dst->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(src0->ne[0] % QK8_0 == 0);
    GGML_ASSERT(src1->ne[0] % QK8_0 == 0);
    GGML_ASSERT(dst->ne[0] % QK8_0 == 0);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne00_blk = src0->ne[0] / QK8_0;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne0_blk  = dst->ne[0] / QK8_0;

    ಆದರೆ (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * src0_d = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *) src0->ದತ್ತಾಂಶ;
        ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * src1_d = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *) src1->ದತ್ತಾಂಶ;
        ಖಂಡ_q8_0 * dst_d        = (ಖಂಡ_q8_0 *) dst->ದತ್ತಾಂಶ;
        ಸ್ಥಿರಾಂಕ size_t type_ಗಾತ್ರ    = ಗಾತ್ರ(ಖಂಡ_q8_0);

        ಆದರೆ (dim != 3) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<ಖಂಡ_q8_0>(
                    src0_d + i3 * (src0->nb[3] / type_ಗಾತ್ರ),
                    src1_d + i3 * (src1->nb[3] / type_ಗಾತ್ರ),
                    dst_d + i3 * (dst->nb[3] / type_ಗಾತ್ರ),
                    ne00_blk, src0->ne[1], src0->ne[2], ne0_blk,
                    dst->ne[1], dst->ne[2], dim, stream);
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಸ್ಥಿರಾಂಕ size_t size0 = ggml_nbytes(src0);
            ಸ್ಥಿರಾಂಕ size_t size1 = ggml_nbytes(src1);
            SYCL_CHECK(CHECK_TRY_ERROR(stream->ಸ್ಮರಣೆ_ನಕಲು(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->ಸ್ಮರಣೆ_ನಕಲು((ಅಕ್ಷರ *) dst_d + size0, src1_d, size1)));
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        concat_T_sycl_non_cont<ಖಂಡ_q8_0>(
            stream, (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ, (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ,
            (ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ,
            ne00_blk, src0->ne[1], src0->ne[2], src0->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->ne[0] / QK8_0, src1->ne[1], src1->ne[2], src1->ne[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            ne0_blk, dst->ne[1], dst->ne[2], dst->ne[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

ಶೂನ್ಯ ggml_sycl_op_concat(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ಆಯ್ಕೆ (dst->type) {
    ಸ್ಥಿತಿ GGML_TYPE_F32:
        concat_impl_sycl<ದಶಮಾಂಶ>(ctx, dst);
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_TYPE_F16:
        concat_impl_sycl<sycl::half>(ctx, dst);
        ನಿಲ್ಲಿಸು;
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_HAS_BF16
    ಸ್ಥಿತಿ GGML_TYPE_BF16:
        concat_impl_sycl<sycl::ext::oneapi::bfloat16>(ctx, dst);
        ನಿಲ್ಲಿಸು;
#ಆದರೆ_ಅಂತ್ಯ
    ಸ್ಥಿತಿ GGML_TYPE_I32:
        concat_impl_sycl<int32_t>(ctx, dst);
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_TYPE_I16:
        concat_impl_sycl<int16_t>(ctx, dst);
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_TYPE_I64:
        concat_impl_sycl<int64_t>(ctx, dst);
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_TYPE_I8:
        concat_impl_sycl<int8_t>(ctx, dst);
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
        concat_impl_q4_0_sycl(ctx, dst);
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
        concat_impl_q4_1_sycl(ctx, dst);
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
        concat_impl_q5_0_sycl(ctx, dst);
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
        concat_impl_q5_1_sycl(ctx, dst);
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
        concat_impl_q8_0_sycl(ctx, dst);
        ನಿಲ್ಲಿಸು;
    ಪೂರ್ವನಿಯೋಜಿತ:
        fprintf(stderr, "%s: unsupported types: dst: %s\n", __func__, ggml_type_name(dst->type));
        GGML_ASSERT(ಸುಳ್ಳು);
    }
}
