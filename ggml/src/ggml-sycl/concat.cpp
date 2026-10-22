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

#అన్తర్భావ "ggml.h"

#అన్తర్భావ "concat.hpp"

స్థిర అంతఃవరుస size_t elem_కొలత(ggml_type t) {
    తిరిగి ggml_type_కొలత(t) / ggml_blck_కొలత(t);
}

మూస <రకం_పేరు T>
స్థిర శూన్య concat_T_dim0(స్థిరరాశి T *x, స్థిరరాశి T *y, T *dst,
                            స్థిరరాశి పూర్ణాంకం ne0, స్థిరరాశి పూర్ణాంకం ne00,
                            స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {
  పూర్ణాంకం nidx = అంశం_ct1.get_local_id(2) +
             అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(2);
  ఒకవేళ (nidx >= ne0) {
    తిరిగి;
  }
  // operation
  పూర్ణాంకం స్థానభ్రంశం_dst = nidx + అంశం_ct1.get_సమూహం(1) * ne0 +
                   అంశం_ct1.get_సమూహం(0) * ne0 * అంశం_ct1.get_సమూహం_range(1);
  ఒకవేళ (nidx < ne00) { // src0
    పూర్ణాంకం స్థానభ్రంశం_src = nidx + అంశం_ct1.get_సమూహం(1) * ne00 +
                     అంశం_ct1.get_సమూహం(0) * ne00 * అంశం_ct1.get_సమూహం_range(1);
    dst[స్థానభ్రంశం_dst] = x[స్థానభ్రంశం_src];
  } లేకపోతే {
    పూర్ణాంకం స్థానభ్రంశం_src =
        nidx - ne00 + అంశం_ct1.get_సమూహం(1) * (ne0 - ne00) +
        అంశం_ct1.get_సమూహం(0) * (ne0 - ne00) * అంశం_ct1.get_సమూహం_range(1);
    dst[స్థానభ్రంశం_dst] = y[స్థానభ్రంశం_src];
  }
}

మూస <రకం_పేరు T>
స్థిర శూన్య concat_T_dim1(స్థిరరాశి T *x, స్థిరరాశి T *y, T *dst,
                            స్థిరరాశి పూర్ణాంకం ne0, స్థిరరాశి పూర్ణాంకం ne01,
                            స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {
  పూర్ణాంకం nidx = అంశం_ct1.get_local_id(2) +
             అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(2);
  ఒకవేళ (nidx >= ne0) {
    తిరిగి;
  }
  // operation
  పూర్ణాంకం స్థానభ్రంశం_dst = nidx + అంశం_ct1.get_సమూహం(1) * ne0 +
                   అంశం_ct1.get_సమూహం(0) * ne0 * అంశం_ct1.get_సమూహం_range(1);
  ఒకవేళ (అంశం_ct1.get_సమూహం(1) < (size_t) ne01) { // src0
    పూర్ణాంకం స్థానభ్రంశం_src =
        nidx + అంశం_ct1.get_సమూహం(1) * ne0 + అంశం_ct1.get_సమూహం(0) * ne0 * ne01;
    dst[స్థానభ్రంశం_dst] = x[స్థానభ్రంశం_src];
  } లేకపోతే {
    పూర్ణాంకం స్థానభ్రంశం_src =
        nidx + (అంశం_ct1.get_సమూహం(1) - ne01) * ne0 +
        అంశం_ct1.get_సమూహం(0) * ne0 * (అంశం_ct1.get_సమూహం_range(1) - ne01);
    dst[స్థానభ్రంశం_dst] = y[స్థానభ్రంశం_src];
  }
}

మూస <రకం_పేరు T>
స్థిర శూన్య concat_T_dim2(స్థిరరాశి T *x, స్థిరరాశి T *y, T *dst,
                            స్థిరరాశి పూర్ణాంకం ne0, స్థిరరాశి పూర్ణాంకం ne02,
                            స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {
  పూర్ణాంకం nidx = అంశం_ct1.get_local_id(2) +
             అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(2);
  ఒకవేళ (nidx >= ne0) {
    తిరిగి;
  }
  // operation
  పూర్ణాంకం స్థానభ్రంశం_dst = nidx + అంశం_ct1.get_సమూహం(1) * ne0 +
                   అంశం_ct1.get_సమూహం(0) * ne0 * అంశం_ct1.get_సమూహం_range(1);
  ఒకవేళ (అంశం_ct1.get_సమూహం(0) < (size_t) ne02) { // src0
    పూర్ణాంకం స్థానభ్రంశం_src = nidx + అంశం_ct1.get_సమూహం(1) * ne0 +
                     అంశం_ct1.get_సమూహం(0) * ne0 * అంశం_ct1.get_సమూహం_range(1);
    dst[స్థానభ్రంశం_dst] = x[స్థానభ్రంశం_src];
  } లేకపోతే {
    పూర్ణాంకం స్థానభ్రంశం_src =
        nidx + అంశం_ct1.get_సమూహం(1) * ne0 +
        (అంశం_ct1.get_సమూహం(0) - ne02) * ne0 * అంశం_ct1.get_సమూహం_range(1);
    dst[స్థానభ్రంశం_dst] = y[స్థానభ్రంశం_src];
  }
}

మూస <రకం_పేరు T>
స్థిర శూన్య concat_T_sycl(స్థిరరాశి T *x, స్థిరరాశి T *y, T *dst,
                            పూర్ణాంకం ne00, పూర్ణాంకం ne01, పూర్ణాంకం ne02, పూర్ణాంకం ne0, పూర్ణాంకం ne1,
                            పూర్ణాంకం ne2, పూర్ణాంకం dim, వరుస_ptr stream) {
  పూర్ణాంకం అంకె_ఖండాలు = (ne0 + SYCL_CONCAT_ఖండం_కొలత - 1) / SYCL_CONCAT_ఖండం_కొలత;
  sycl::range<3> gridDim(ne2, ne1, అంకె_ఖండాలు);
  ఎంపిక (dim) {
  స్థితి 0:
      stream->parallel_for(sycl::nd_range<3>(gridDim * sycl::range<3>(1, 1, SYCL_CONCAT_ఖండం_కొలత),
                                          sycl::range<3>(1, 1, SYCL_CONCAT_ఖండం_కొలత)),
                        [=](sycl::nd_అంశం<3> అంశం_ct1) { concat_T_dim0<T>(x, y, dst, ne0, ne00, అంశం_ct1); });
      ఆపు;
  స్థితి 1:
      stream->parallel_for(sycl::nd_range<3>(gridDim * sycl::range<3>(1, 1, SYCL_CONCAT_ఖండం_కొలత),
                                          sycl::range<3>(1, 1, SYCL_CONCAT_ఖండం_కొలత)),
                        [=](sycl::nd_అంశం<3> అంశం_ct1) { concat_T_dim1<T>(x, y, dst, ne0, ne01, అంశం_ct1); });
      ఆపు;
  // dim >=2 will be dispatched to the default path
  అప్రమేయ:
      stream->parallel_for(sycl::nd_range<3>(gridDim * sycl::range<3>(1, 1, SYCL_CONCAT_ఖండం_కొలత),
                                          sycl::range<3>(1, 1, SYCL_CONCAT_ఖండం_కొలత)),
                        [=](sycl::nd_అంశం<3> అంశం_ct1) { concat_T_dim2<T>(x, y, dst, ne0, ne02, అంశం_ct1); });
      ఆపు;
  }
}

// non-contiguous kernel (slow)
మూస<రకం_పేరు T>
స్థిర శూన్య concat_T_sycl_non_cont(
    వరుస_ptr stream, స్థిరరాశి అక్షరం *src0, స్థిరరాశి అక్షరం *src1, అక్షరం *dst,
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
  స్థిరరాశి int64_t ne0_pad   = GGML_PAD(ne0, WARP_కొలత);
  స్థిరరాశి int64_t ఖండం_ne0 = ne0_pad < SYCL_CONCAT_ఖండం_కొలత ? ne0_pad : (int64_t) SYCL_CONCAT_ఖండం_కొలత;
  sycl::range<3> ఖండం_Dim(1, 1, ఖండం_ne0);

  stream->parallel_for(sycl::nd_range<3>(gridDim * ఖండం_Dim, ఖండం_Dim), [=](sycl::nd_అంశం<3> అంశం_ct1) {
      int64_t i3 = అంశం_ct1.get_సమూహం(0);
      int64_t i2 = అంశం_ct1.get_సమూహం(1);
      int64_t i1 = అంశం_ct1.get_సమూహం(2);

      int64_t o[4] = { 0, 0, 0, 0 };
      o[dim]       = dim == 0 ? ne00 : (dim == 1 ? ne01 : (dim == 2 ? ne02 : ne03));

      స్థిరరాశి T * x;

      కొరకు (పూర్ణాంకం i0 = అంశం_ct1.get_local_id(2); i0 < ne0; i0 += అంశం_ct1.get_local_range(2)) {
          ఒకవేళ (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
              x = (స్థిరరాశి T *) (src0 + (i3) *nb03 + (i2) *nb02 + (i1) *nb01 + (i0) *nb00);
          } లేకపోతే {
              x = (స్థిరరాశి T *) (src1 + (i3 - o[3]) * nb13 + (i2 - o[2]) * nb12 + (i1 - o[1]) * nb11 +
                                   (i0 - o[0]) * nb10);
          }

          T *y = (T *)(dst + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

          *y = *x;
      }
  });
}

మూస <రకం_పేరు T>
శూన్య concat_impl_sycl(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    స్థిరరాశి ggml_tensor *  src0   = dst->src[0];
    స్థిరరాశి ggml_tensor *  src1   = dst->src[1];
    వరుస_ptr            stream = ctx.stream();

    స్థిరరాశి int32_t dim = ((int32_t *) dst->op_params)[0];

    ఒకవేళ (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        స్థిరరాశి T * src0_d = (స్థిరరాశి T *) src0->దత్తాంశం;
        స్థిరరాశి T * src1_d = (స్థిరరాశి T *) src1->దత్తాంశం;
        T * dst_d = (T *) dst->దత్తాంశం;
        size_t type_కొలత = elem_కొలత(dst->type);
        ఒకవేళ (dim != 3) {
            కొరకు (పూర్ణాంకం i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<T>(src0_d + i3 * (src0->nb[3] / type_కొలత), src1_d + i3 * (src1->nb[3] / type_కొలత),
                                dst_d + i3 * (dst->nb[3] / type_కొలత), src0->ne[0], src0->ne[1], src0->ne[2], dst->ne[0],
                                dst->ne[1], dst->ne[2], dim, stream);
            }
        } లేకపోతే {
            స్థిరరాశి size_t size0 = ggml_nbytes(src0);
            స్థిరరాశి size_t size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->జ్ఞాపకం_నకలు(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->జ్ఞాపకం_నకలు(dst_d + size0 / type_కొలత, src1_d, size1)));
        }
    } లేకపోతే {
        concat_T_sycl_non_cont<T>(stream, (స్థిరరాశి అక్షరం *) src0->దత్తాంశం, (స్థిరరాశి అక్షరం *) src1->దత్తాంశం, (అక్షరం *) dst->దత్తాంశం,
                                 src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src0->nb[0], src0->nb[1],
                                 src0->nb[2], src0->nb[3], src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                                 src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3], dst->ne[0], dst->ne[1], dst->ne[2],
                                 dst->ne[3], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

స్థిర శూన్య concat_impl_q4_0_sycl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    స్థిరరాశి ggml_tensor * src0   = dst->src[0];
    స్థిరరాశి ggml_tensor * src1   = dst->src[1];
    వరుస_ptr           stream = ctx.stream();

    స్థిరరాశి int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(src1->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(dst->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(src0->ne[0] % QK4_0 == 0);
    GGML_ASSERT(src1->ne[0] % QK4_0 == 0);
    GGML_ASSERT(dst->ne[0] % QK4_0 == 0);

    స్థిరరాశి పూర్ణాంకం ne00_blk = src0->ne[0] / QK4_0;
    స్థిరరాశి పూర్ణాంకం ne0_blk  = dst->ne[0] / QK4_0;

    ఒకవేళ (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        స్థిరరాశి ఖండం_q4_0 * src0_d = (స్థిరరాశి ఖండం_q4_0 *) src0->దత్తాంశం;
        స్థిరరాశి ఖండం_q4_0 * src1_d = (స్థిరరాశి ఖండం_q4_0 *) src1->దత్తాంశం;
        ఖండం_q4_0 * dst_d        = (ఖండం_q4_0 *) dst->దత్తాంశం;
        స్థిరరాశి size_t type_కొలత    = పరిమాణం(ఖండం_q4_0);

        ఒకవేళ (dim != 3) {
            కొరకు (పూర్ణాంకం i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<ఖండం_q4_0>(
                    src0_d + i3 * (src0->nb[3] / type_కొలత),
                    src1_d + i3 * (src1->nb[3] / type_కొలత),
                    dst_d + i3 * (dst->nb[3] / type_కొలత),
                    ne00_blk, src0->ne[1], src0->ne[2], ne0_blk,
                    dst->ne[1], dst->ne[2], dim, stream);
            }
        } లేకపోతే {
            స్థిరరాశి size_t size0 = ggml_nbytes(src0);
            స్థిరరాశి size_t size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->జ్ఞాపకం_నకలు(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->జ్ఞాపకం_నకలు((అక్షరం *) dst_d + size0, src1_d, size1)));
        }
    } లేకపోతే {
        concat_T_sycl_non_cont<ఖండం_q4_0>(
            stream, (స్థిరరాశి అక్షరం *) src0->దత్తాంశం, (స్థిరరాశి అక్షరం *) src1->దత్తాంశం,
            (అక్షరం *) dst->దత్తాంశం,
            ne00_blk, src0->ne[1], src0->ne[2], src0->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->ne[0] / QK4_0, src1->ne[1], src1->ne[2], src1->ne[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            ne0_blk, dst->ne[1], dst->ne[2], dst->ne[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

స్థిర శూన్య concat_impl_q4_1_sycl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    స్థిరరాశి ggml_tensor * src0   = dst->src[0];
    స్థిరరాశి ggml_tensor * src1   = dst->src[1];
    వరుస_ptr           stream = ctx.stream();

    స్థిరరాశి int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(src1->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(dst->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(src0->ne[0] % QK4_1 == 0);
    GGML_ASSERT(src1->ne[0] % QK4_1 == 0);
    GGML_ASSERT(dst->ne[0] % QK4_1 == 0);

    స్థిరరాశి పూర్ణాంకం ne00_blk = src0->ne[0] / QK4_1;
    స్థిరరాశి పూర్ణాంకం ne0_blk  = dst->ne[0] / QK4_1;

    ఒకవేళ (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        స్థిరరాశి ఖండం_q4_1 * src0_d = (స్థిరరాశి ఖండం_q4_1 *) src0->దత్తాంశం;
        స్థిరరాశి ఖండం_q4_1 * src1_d = (స్థిరరాశి ఖండం_q4_1 *) src1->దత్తాంశం;
        ఖండం_q4_1 * dst_d        = (ఖండం_q4_1 *) dst->దత్తాంశం;
        స్థిరరాశి size_t type_కొలత    = పరిమాణం(ఖండం_q4_1);

        ఒకవేళ (dim != 3) {
            కొరకు (పూర్ణాంకం i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<ఖండం_q4_1>(
                    src0_d + i3 * (src0->nb[3] / type_కొలత),
                    src1_d + i3 * (src1->nb[3] / type_కొలత),
                    dst_d + i3 * (dst->nb[3] / type_కొలత),
                    ne00_blk, src0->ne[1], src0->ne[2], ne0_blk,
                    dst->ne[1], dst->ne[2], dim, stream);
            }
        } లేకపోతే {
            స్థిరరాశి size_t size0 = ggml_nbytes(src0);
            స్థిరరాశి size_t size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->జ్ఞాపకం_నకలు(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->జ్ఞాపకం_నకలు((అక్షరం *) dst_d + size0, src1_d, size1)));
        }
    } లేకపోతే {
        concat_T_sycl_non_cont<ఖండం_q4_1>(
            stream, (స్థిరరాశి అక్షరం *) src0->దత్తాంశం, (స్థిరరాశి అక్షరం *) src1->దత్తాంశం,
            (అక్షరం *) dst->దత్తాంశం,
            ne00_blk, src0->ne[1], src0->ne[2], src0->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->ne[0] / QK4_1, src1->ne[1], src1->ne[2], src1->ne[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            ne0_blk, dst->ne[1], dst->ne[2], dst->ne[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

స్థిర శూన్య concat_impl_q5_0_sycl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    స్థిరరాశి ggml_tensor * src0   = dst->src[0];
    స్థిరరాశి ggml_tensor * src1   = dst->src[1];
    వరుస_ptr           stream = ctx.stream();

    స్థిరరాశి int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_Q5_0);
    GGML_ASSERT(src1->type == GGML_TYPE_Q5_0);
    GGML_ASSERT(dst->type == GGML_TYPE_Q5_0);
    GGML_ASSERT(src0->ne[0] % QK5_0 == 0);
    GGML_ASSERT(src1->ne[0] % QK5_0 == 0);
    GGML_ASSERT(dst->ne[0] % QK5_0 == 0);

    స్థిరరాశి పూర్ణాంకం ne00_blk = src0->ne[0] / QK5_0;
    స్థిరరాశి పూర్ణాంకం ne0_blk  = dst->ne[0] / QK5_0;

    ఒకవేళ (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        స్థిరరాశి ఖండం_q5_0 * src0_d = (స్థిరరాశి ఖండం_q5_0 *) src0->దత్తాంశం;
        స్థిరరాశి ఖండం_q5_0 * src1_d = (స్థిరరాశి ఖండం_q5_0 *) src1->దత్తాంశం;
        ఖండం_q5_0 * dst_d        = (ఖండం_q5_0 *) dst->దత్తాంశం;
        స్థిరరాశి size_t type_కొలత    = పరిమాణం(ఖండం_q5_0);

        ఒకవేళ (dim != 3) {
            కొరకు (పూర్ణాంకం i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<ఖండం_q5_0>(
                    src0_d + i3 * (src0->nb[3] / type_కొలత),
                    src1_d + i3 * (src1->nb[3] / type_కొలత),
                    dst_d + i3 * (dst->nb[3] / type_కొలత),
                    ne00_blk, src0->ne[1], src0->ne[2], ne0_blk,
                    dst->ne[1], dst->ne[2], dim, stream);
            }
        } లేకపోతే {
            స్థిరరాశి size_t size0 = ggml_nbytes(src0);
            స్థిరరాశి size_t size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->జ్ఞాపకం_నకలు(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->జ్ఞాపకం_నకలు((అక్షరం *) dst_d + size0, src1_d, size1)));
        }
    } లేకపోతే {
        concat_T_sycl_non_cont<ఖండం_q5_0>(
            stream, (స్థిరరాశి అక్షరం *) src0->దత్తాంశం, (స్థిరరాశి అక్షరం *) src1->దత్తాంశం,
            (అక్షరం *) dst->దత్తాంశం,
            ne00_blk, src0->ne[1], src0->ne[2], src0->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->ne[0] / QK5_0, src1->ne[1], src1->ne[2], src1->ne[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            ne0_blk, dst->ne[1], dst->ne[2], dst->ne[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

స్థిర శూన్య concat_impl_q5_1_sycl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    స్థిరరాశి ggml_tensor * src0   = dst->src[0];
    స్థిరరాశి ggml_tensor * src1   = dst->src[1];
    వరుస_ptr           stream = ctx.stream();

    స్థిరరాశి int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_Q5_1);
    GGML_ASSERT(src1->type == GGML_TYPE_Q5_1);
    GGML_ASSERT(dst->type == GGML_TYPE_Q5_1);
    GGML_ASSERT(src0->ne[0] % QK5_1 == 0);
    GGML_ASSERT(src1->ne[0] % QK5_1 == 0);
    GGML_ASSERT(dst->ne[0] % QK5_1 == 0);

    స్థిరరాశి పూర్ణాంకం ne00_blk = src0->ne[0] / QK5_1;
    స్థిరరాశి పూర్ణాంకం ne0_blk  = dst->ne[0] / QK5_1;

    ఒకవేళ (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        స్థిరరాశి ఖండం_q5_1 * src0_d = (స్థిరరాశి ఖండం_q5_1 *) src0->దత్తాంశం;
        స్థిరరాశి ఖండం_q5_1 * src1_d = (స్థిరరాశి ఖండం_q5_1 *) src1->దత్తాంశం;
        ఖండం_q5_1 * dst_d        = (ఖండం_q5_1 *) dst->దత్తాంశం;
        స్థిరరాశి size_t type_కొలత    = పరిమాణం(ఖండం_q5_1);

        ఒకవేళ (dim != 3) {
            కొరకు (పూర్ణాంకం i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<ఖండం_q5_1>(
                    src0_d + i3 * (src0->nb[3] / type_కొలత),
                    src1_d + i3 * (src1->nb[3] / type_కొలత),
                    dst_d + i3 * (dst->nb[3] / type_కొలత),
                    ne00_blk, src0->ne[1], src0->ne[2], ne0_blk,
                    dst->ne[1], dst->ne[2], dim, stream);
            }
        } లేకపోతే {
            స్థిరరాశి size_t size0 = ggml_nbytes(src0);
            స్థిరరాశి size_t size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->జ్ఞాపకం_నకలు(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->జ్ఞాపకం_నకలు((అక్షరం *) dst_d + size0, src1_d, size1)));
        }
    } లేకపోతే {
        concat_T_sycl_non_cont<ఖండం_q5_1>(
            stream, (స్థిరరాశి అక్షరం *) src0->దత్తాంశం, (స్థిరరాశి అక్షరం *) src1->దత్తాంశం,
            (అక్షరం *) dst->దత్తాంశం,
            ne00_blk, src0->ne[1], src0->ne[2], src0->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->ne[0] / QK5_1, src1->ne[1], src1->ne[2], src1->ne[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            ne0_blk, dst->ne[1], dst->ne[2], dst->ne[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

స్థిర శూన్య concat_impl_q8_0_sycl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    స్థిరరాశి ggml_tensor * src0   = dst->src[0];
    స్థిరరాశి ggml_tensor * src1   = dst->src[1];
    వరుస_ptr           stream = ctx.stream();

    స్థిరరాశి int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(src1->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(dst->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(src0->ne[0] % QK8_0 == 0);
    GGML_ASSERT(src1->ne[0] % QK8_0 == 0);
    GGML_ASSERT(dst->ne[0] % QK8_0 == 0);

    స్థిరరాశి పూర్ణాంకం ne00_blk = src0->ne[0] / QK8_0;
    స్థిరరాశి పూర్ణాంకం ne0_blk  = dst->ne[0] / QK8_0;

    ఒకవేళ (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        స్థిరరాశి ఖండం_q8_0 * src0_d = (స్థిరరాశి ఖండం_q8_0 *) src0->దత్తాంశం;
        స్థిరరాశి ఖండం_q8_0 * src1_d = (స్థిరరాశి ఖండం_q8_0 *) src1->దత్తాంశం;
        ఖండం_q8_0 * dst_d        = (ఖండం_q8_0 *) dst->దత్తాంశం;
        స్థిరరాశి size_t type_కొలత    = పరిమాణం(ఖండం_q8_0);

        ఒకవేళ (dim != 3) {
            కొరకు (పూర్ణాంకం i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<ఖండం_q8_0>(
                    src0_d + i3 * (src0->nb[3] / type_కొలత),
                    src1_d + i3 * (src1->nb[3] / type_కొలత),
                    dst_d + i3 * (dst->nb[3] / type_కొలత),
                    ne00_blk, src0->ne[1], src0->ne[2], ne0_blk,
                    dst->ne[1], dst->ne[2], dim, stream);
            }
        } లేకపోతే {
            స్థిరరాశి size_t size0 = ggml_nbytes(src0);
            స్థిరరాశి size_t size1 = ggml_nbytes(src1);
            SYCL_CHECK(CHECK_TRY_ERROR(stream->జ్ఞాపకం_నకలు(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->జ్ఞాపకం_నకలు((అక్షరం *) dst_d + size0, src1_d, size1)));
        }
    } లేకపోతే {
        concat_T_sycl_non_cont<ఖండం_q8_0>(
            stream, (స్థిరరాశి అక్షరం *) src0->దత్తాంశం, (స్థిరరాశి అక్షరం *) src1->దత్తాంశం,
            (అక్షరం *) dst->దత్తాంశం,
            ne00_blk, src0->ne[1], src0->ne[2], src0->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->ne[0] / QK8_0, src1->ne[1], src1->ne[2], src1->ne[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            ne0_blk, dst->ne[1], dst->ne[2], dst->ne[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

శూన్య ggml_sycl_op_concat(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ఎంపిక (dst->type) {
    స్థితి GGML_TYPE_F32:
        concat_impl_sycl<దశాంశం>(ctx, dst);
        ఆపు;
    స్థితి GGML_TYPE_F16:
        concat_impl_sycl<sycl::half>(ctx, dst);
        ఆపు;
#నిర్వచితమైతే GGML_SYCL_HAS_BF16
    స్థితి GGML_TYPE_BF16:
        concat_impl_sycl<sycl::ext::oneapi::bfloat16>(ctx, dst);
        ఆపు;
#అయితే_ముగింపు
    స్థితి GGML_TYPE_I32:
        concat_impl_sycl<int32_t>(ctx, dst);
        ఆపు;
    స్థితి GGML_TYPE_I16:
        concat_impl_sycl<int16_t>(ctx, dst);
        ఆపు;
    స్థితి GGML_TYPE_I64:
        concat_impl_sycl<int64_t>(ctx, dst);
        ఆపు;
    స్థితి GGML_TYPE_I8:
        concat_impl_sycl<int8_t>(ctx, dst);
        ఆపు;
    స్థితి GGML_TYPE_Q4_0:
        concat_impl_q4_0_sycl(ctx, dst);
        ఆపు;
    స్థితి GGML_TYPE_Q4_1:
        concat_impl_q4_1_sycl(ctx, dst);
        ఆపు;
    స్థితి GGML_TYPE_Q5_0:
        concat_impl_q5_0_sycl(ctx, dst);
        ఆపు;
    స్థితి GGML_TYPE_Q5_1:
        concat_impl_q5_1_sycl(ctx, dst);
        ఆపు;
    స్థితి GGML_TYPE_Q8_0:
        concat_impl_q8_0_sycl(ctx, dst);
        ఆపు;
    అప్రమేయ:
        fprintf(stderr, "%s: unsupported types: dst: %s\n", __func__, ggml_type_name(dst->type));
        GGML_ASSERT(అబద్ధం);
    }
}
