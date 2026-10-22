#अन्तर्भाव <sycl/sycl.hpp>
#अन्तर्भाव "common.hpp"
#अन्तर्भाव "add-id.hpp"

स्थिर शून्य add_id_kernel(
    स्थिरांक दशांश* src0,
    स्थिरांक दशांश* src1,
    स्थिरांक int32_t* src2,
    दशांश* dst,
    int64_t ne0,
    int64_t ne1,
    माप_प्रकार nb01,
    माप_प्रकार nb02,
    माप_प्रकार nb11,
    माप_प्रकार nb21,
    sycl::nd_item<3> item_ct1) {
  स्थिरांक int64_t i1 = item_ct1.get_group(2);
  स्थिरांक int64_t i2 = item_ct1.get_group(1);

  स्थिरांक पूर्णांक i11 =
      *(स्थिरांक int32_t*)((स्थिरांक अक्षर*)src2 + i1 * आकार(int32_t) + i2 * nb21);

  स्थिरांक माप_प्रकार nb1 = ne0 * आकार(दशांश);
  स्थिरांक माप_प्रकार nb2 = ne1 * nb1;

  दशांश* dst_row = (दशांश*)((अक्षर*)dst + i1 * nb1 + i2 * nb2);
  स्थिरांक दशांश* src0_row =
      (स्थिरांक दशांश*)((स्थिरांक अक्षर*)src0 + i1 * nb01 + i2 * nb02);
  स्थिरांक दशांश* src1_row = (स्थिरांक दशांश*)((स्थिरांक अक्षर*)src1 + i11 * nb11);

  साठी (int64_t i0 = item_ct1.get_local_id(2); i0 < ne0;
       i0 += item_ct1.get_local_range(2)) {
    dst_row[i0] = src0_row[i0] + src1_row[i0];
  }
}

शून्य ggml_sycl_add_id(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
  स्थिरांक ggml_tensor* src0 = dst->src[0];
  स्थिरांक ggml_tensor* src1 = dst->src[1];
  स्थिरांक ggml_tensor* src2 = dst->src[2];

  GGML_TENSOR_TERNARY_OP_LOCALS

  GGML_ASSERT(dst->type == GGML_TYPE_F32);
  GGML_ASSERT(src0->type == GGML_TYPE_F32);
  GGML_ASSERT(src1->type == GGML_TYPE_F32);
  GGML_ASSERT(src2->type == GGML_TYPE_I32);

  GGML_ASSERT(nb00 == आकार(दशांश));
  GGML_ASSERT(nb10 == आकार(दशांश));
  GGML_ASSERT(nb20 == आकार(int32_t));

  स्थिरांक दशांश* src0_d = (स्थिरांक दशांश*)src0->data;
  स्थिरांक दशांश* src1_d = (स्थिरांक दशांश*)src1->data;
  स्थिरांक int32_t* src2_d = (स्थिरांक int32_t*)src2->data;
  दशांश* dst_d = (दशांश*)dst->data;

  स्थिरांक अचिन्हित पूर्णांक max_work_group_size = ggml_sycl_info().max_work_group_sizes[ctx.device];
  GGML_ASSERT(max_work_group_size % (WARP_SIZE * WARP_SIZE) == 0);

  पूर्णांक threads = std::min((अचिन्हित पूर्णांक)ne00, max_work_group_size);  // cols

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
