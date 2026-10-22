#ضمن <sycl/sycl.hpp>
#ضمن "common.hpp"
#ضمن "add-id.hpp"

ساكن فراغ جمع_کړه_id_kernel(
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
    sycl::nd_توکی<3> توکی_ct1) {
  ثابت int64_t i1 = توکی_ct1.get_ډله(2);
  ثابت int64_t i2 = توکی_ct1.get_ډله(1);

  ثابت صحيح i11 =
      *(ثابت int32_t*)((ثابت حرف*)src2 + i1 * حجم(int32_t) + i2 * nb21);

  ثابت size_t nb1 = ne0 * حجم(عائم);
  ثابت size_t nb2 = ne1 * nb1;

  عائم* dst_لیکه = (عائم*)((حرف*)dst + i1 * nb1 + i2 * nb2);
  ثابت عائم* src0_لیکه =
      (ثابت عائم*)((ثابت حرف*)src0 + i1 * nb01 + i2 * nb02);
  ثابت عائم* src1_لیکه = (ثابت عائم*)((ثابت حرف*)src1 + i11 * nb11);

  لكل (int64_t i0 = توکی_ct1.get_local_id(2); i0 < ne0;
       i0 += توکی_ct1.get_local_range(2)) {
    dst_لیکه[i0] = src0_لیکه[i0] + src1_لیکه[i0];
  }
}

فراغ ggml_sycl_جمع_کړه_id(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
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

  ثابت عائم* src0_d = (ثابت عائم*)src0->معلومات;
  ثابت عائم* src1_d = (ثابت عائم*)src1->معلومات;
  ثابت int32_t* src2_d = (ثابت int32_t*)src2->معلومات;
  عائم* dst_d = (عائم*)dst->معلومات;

  ثابت بینښه صحيح max_work_ډله_اندازه = ggml_sycl_info().max_work_ډله_sizes[ctx.device];
  GGML_ASSERT(max_work_ډله_اندازه % (WARP_اندازه * WARP_اندازه) == 0);

  صحيح تارونه = std::min((بینښه صحيح)ne00, max_work_ډله_اندازه);  //ستون_شمېر

  ctx.stream()->parallel_for(
      sycl::nd_range<3>(
          sycl::range<3>(1, ne02, ne01) * sycl::range<3>(1, 1, تارونه),
          sycl::range<3>(1, 1, تارونه)),
      [=](sycl::nd_توکی<3> توکی_ct1) {
        جمع_کړه_id_kernel(
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
            توکی_ct1);
      });
}
