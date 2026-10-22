#ضمن <sycl/sycl.hpp>
#ضمن "common.hpp"
#ضمن "add-id.hpp"

ساكن فراغ جمع_کن_id_kernel(
    ثابت عائم* src0,
    ثابت عائم* src1,
    ثابت int32_t* src2,
    عائم* dst,
    int64_t ne0,
    int64_t ne1,
    size_t nb01,
    size_t nb02,
    size_t nb11,
    size_t nb21,
    sycl::nd_قلم<3> قلم_ct1) {
  ثابت int64_t i1 = قلم_ct1.get_گروه(2);
  ثابت int64_t i2 = قلم_ct1.get_گروه(1);

  ثابت صحيح i11 =
      *(ثابت int32_t*)((ثابت حرف*)src2 + i1 * حجم(int32_t) + i2 * nb21);

  ثابت size_t nb1 = ne0 * حجم(عائم);
  ثابت size_t nb2 = ne1 * nb1;

  عائم* dst_ردیف = (عائم*)((حرف*)dst + i1 * nb1 + i2 * nb2);
  ثابت عائم* src0_ردیف =
      (ثابت عائم*)((ثابت حرف*)src0 + i1 * nb01 + i2 * nb02);
  ثابت عائم* src1_ردیف = (ثابت عائم*)((ثابت حرف*)src1 + i11 * nb11);

  لكل (int64_t i0 = قلم_ct1.get_local_id(2); i0 < ne0;
       i0 += قلم_ct1.get_local_range(2)) {
    dst_ردیف[i0] = src0_ردیف[i0] + src1_ردیف[i0];
  }
}

فراغ ggml_sycl_جمع_کن_id(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
  ثابت ggml_tensor* src0 = dst->src[0];
  ثابت ggml_tensor* src1 = dst->src[1];
  ثابت ggml_tensor* src2 = dst->src[2];

  GGML_TENSOR_TERNARY_OP_LOCALS

  GGML_ASSERT(dst->type == GGML_TYPE_F32);
  GGML_ASSERT(src0->type == GGML_TYPE_F32);
  GGML_ASSERT(src1->type == GGML_TYPE_F32);
  GGML_ASSERT(src2->type == GGML_TYPE_I32);

  GGML_ASSERT(nb00 == حجم(عائم));
  GGML_ASSERT(nb10 == حجم(عائم));
  GGML_ASSERT(nb20 == حجم(int32_t));

  ثابت عائم* src0_d = (ثابت عائم*)src0->داده;
  ثابت عائم* src1_d = (ثابت عائم*)src1->داده;
  ثابت int32_t* src2_d = (ثابت int32_t*)src2->داده;
  عائم* dst_d = (عائم*)dst->داده;

  ثابت بیعلامت صحيح max_work_گروه_اندازه = ggml_sycl_info().max_work_گروه_sizes[ctx.device];
  GGML_ASSERT(max_work_گروه_اندازه % (WARP_اندازه * WARP_اندازه) == 0);

  صحيح نخها = std::min((بیعلامت صحيح)ne00, max_work_گروه_اندازه);  //تعداد_ستون

  ctx.stream()->parallel_for(
      sycl::nd_range<3>(
          sycl::range<3>(1, ne02, ne01) * sycl::range<3>(1, 1, نخها),
          sycl::range<3>(1, 1, نخها)),
      [=](sycl::nd_قلم<3> قلم_ct1) {
        جمع_کن_id_kernel(
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
            قلم_ct1);
      });
}
