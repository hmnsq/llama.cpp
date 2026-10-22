#समावेश <sycl/sycl.hpp>
#समावेश "common.hpp"
#समावेश "add-id.hpp"

अटल व्योम add_id_kernel(
    स्थिर भग्न* src0,
    स्थिर भग्न* src1,
    स्थिर int32_t* src2,
    भग्न* dst,
    int64_t ne0,
    int64_t ne1,
    माप_प्रकार nb01,
    माप_प्रकार nb02,
    माप_प्रकार nb11,
    माप_प्रकार nb21,
    sycl::nd_item<3> item_ct1) {
  स्थिर int64_t i1 = item_ct1.get_group(2);
  स्थिर int64_t i2 = item_ct1.get_group(1);

  स्थिर पूर्णांक i11 =
      *(स्थिर int32_t*)((स्थिर अक्षर*)src2 + i1 * माप(int32_t) + i2 * nb21);

  स्थिर माप_प्रकार nb1 = ne0 * माप(भग्न);
  स्थिर माप_प्रकार nb2 = ne1 * nb1;

  भग्न* dst_row = (भग्न*)((अक्षर*)dst + i1 * nb1 + i2 * nb2);
  स्थिर भग्न* src0_row =
      (स्थिर भग्न*)((स्थिर अक्षर*)src0 + i1 * nb01 + i2 * nb02);
  स्थिर भग्न* src1_row = (स्थिर भग्न*)((स्थिर अक्षर*)src1 + i11 * nb11);

  क्रम (int64_t i0 = item_ct1.get_local_id(2); i0 < ne0;
       i0 += item_ct1.get_local_range(2)) {
    dst_row[i0] = src0_row[i0] + src1_row[i0];
  }
}

व्योम ggml_sycl_add_id(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
  स्थिर ggml_tensor* src0 = dst->src[0];
  स्थिर ggml_tensor* src1 = dst->src[1];
  स्थिर ggml_tensor* src2 = dst->src[2];

  GGML_TENSOR_TERNARY_OP_LOCALS

  GGML_ASSERT(dst->type == GGML_TYPE_F32);
  GGML_ASSERT(src0->type == GGML_TYPE_F32);
  GGML_ASSERT(src1->type == GGML_TYPE_F32);
  GGML_ASSERT(src2->type == GGML_TYPE_I32);

  GGML_ASSERT(nb00 == माप(भग्न));
  GGML_ASSERT(nb10 == माप(भग्न));
  GGML_ASSERT(nb20 == माप(int32_t));

  स्थिर भग्न* src0_d = (स्थिर भग्न*)src0->तथ्य;
  स्थिर भग्न* src1_d = (स्थिर भग्न*)src1->तथ्य;
  स्थिर int32_t* src2_d = (स्थिर int32_t*)src2->तथ्य;
  भग्न* dst_d = (भग्न*)dst->तथ्य;

  स्थिर अचिन्हित पूर्णांक max_work_group_आकार = ggml_sycl_info().max_work_group_sizes[ctx.device];
  GGML_ASSERT(max_work_group_आकार % (WARP_आकार * WARP_आकार) == 0);

  पूर्णांक threads = std::min((अचिन्हित पूर्णांक)ne00, max_work_group_आकार);  // cols

  ctx.stream()->parallel_for(
      sycl::nd_range<3>(
          sycl::range<3>(1, ne02, ne01) * sycl::range<3>(1, 1, threads),
          sycl::range<3>(1, 1, threads)),
      [=](sycl::nd_item<3> item_ct1) {
        add_id_kernel(
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
            item_ct1);
      });
}
