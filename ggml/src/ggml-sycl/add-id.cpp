#అన్తర్భావ <sycl/sycl.hpp>
#అన్తర్భావ "common.hpp"
#అన్తర్భావ "add-id.hpp"

స్థిర శూన్య కూడు_id_kernel(
    స్థిరరాశి దశాంశం* src0,
    స్థిరరాశి దశాంశం* src1,
    స్థిరరాశి int32_t* src2,
    దశాంశం* dst,
    int64_t ne0,
    int64_t ne1,
    size_t nb01,
    size_t nb02,
    size_t nb11,
    size_t nb21,
    sycl::nd_అంశం<3> అంశం_ct1) {
  స్థిరరాశి int64_t i1 = అంశం_ct1.get_సమూహం(2);
  స్థిరరాశి int64_t i2 = అంశం_ct1.get_సమూహం(1);

  స్థిరరాశి పూర్ణాంకం i11 =
      *(స్థిరరాశి int32_t*)((స్థిరరాశి అక్షరం*)src2 + i1 * పరిమాణం(int32_t) + i2 * nb21);

  స్థిరరాశి size_t nb1 = ne0 * పరిమాణం(దశాంశం);
  స్థిరరాశి size_t nb2 = ne1 * nb1;

  దశాంశం* dst_అడ్డువరుస = (దశాంశం*)((అక్షరం*)dst + i1 * nb1 + i2 * nb2);
  స్థిరరాశి దశాంశం* src0_అడ్డువరుస =
      (స్థిరరాశి దశాంశం*)((స్థిరరాశి అక్షరం*)src0 + i1 * nb01 + i2 * nb02);
  స్థిరరాశి దశాంశం* src1_అడ్డువరుస = (స్థిరరాశి దశాంశం*)((స్థిరరాశి అక్షరం*)src1 + i11 * nb11);

  కొరకు (int64_t i0 = అంశం_ct1.get_local_id(2); i0 < ne0;
       i0 += అంశం_ct1.get_local_range(2)) {
    dst_అడ్డువరుస[i0] = src0_అడ్డువరుస[i0] + src1_అడ్డువరుస[i0];
  }
}

శూన్య ggml_sycl_కూడు_id(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
  స్థిరరాశి ggml_tensor* src0 = dst->src[0];
  స్థిరరాశి ggml_tensor* src1 = dst->src[1];
  స్థిరరాశి ggml_tensor* src2 = dst->src[2];

  GGML_TENSOR_TERNARY_OP_LOCALS

  GGML_ASSERT(dst->type == GGML_TYPE_F32);
  GGML_ASSERT(src0->type == GGML_TYPE_F32);
  GGML_ASSERT(src1->type == GGML_TYPE_F32);
  GGML_ASSERT(src2->type == GGML_TYPE_I32);

  GGML_ASSERT(nb00 == పరిమాణం(దశాంశం));
  GGML_ASSERT(nb10 == పరిమాణం(దశాంశం));
  GGML_ASSERT(nb20 == పరిమాణం(int32_t));

  స్థిరరాశి దశాంశం* src0_d = (స్థిరరాశి దశాంశం*)src0->దత్తాంశం;
  స్థిరరాశి దశాంశం* src1_d = (స్థిరరాశి దశాంశం*)src1->దత్తాంశం;
  స్థిరరాశి int32_t* src2_d = (స్థిరరాశి int32_t*)src2->దత్తాంశం;
  దశాంశం* dst_d = (దశాంశం*)dst->దత్తాంశం;

  స్థిరరాశి గుర్తులేని పూర్ణాంకం max_work_సమూహం_కొలత = ggml_sycl_info().max_work_సమూహం_sizes[ctx.device];
  GGML_ASSERT(max_work_సమూహం_కొలత % (WARP_కొలత * WARP_కొలత) == 0);

  పూర్ణాంకం దారాలు = std::min((గుర్తులేని పూర్ణాంకం)ne00, max_work_సమూహం_కొలత);  //నిలువుల_సంఖ్య

  ctx.stream()->parallel_for(
      sycl::nd_range<3>(
          sycl::range<3>(1, ne02, ne01) * sycl::range<3>(1, 1, దారాలు),
          sycl::range<3>(1, 1, దారాలు)),
      [=](sycl::nd_అంశం<3> అంశం_ct1) {
        కూడు_id_kernel(
            src0_d,
            src1_d,
            src2_d,
            dst_d,
            ne0,
            ne1,
            nb01,
            nb02,
            nb11,
            nb21,
            అంశం_ct1);
      });
}
