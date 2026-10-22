#अन्तर्भाव <sycl/sycl.hpp>
#अन्तर्भाव "common.hpp"
#अन्तर्भाव "add-id.hpp"

स्थिर शून्य add_id_kernel(
    ध्रुव भिन्न* src0,
    ध्रुव भिन्न* src1,
    ध्रुव int32_t* src2,
    भिन्न* dst,
    int64_t ne0,
    int64_t ne1,
    size_t nb01,
    size_t nb02,
    size_t nb11,
    size_t nb21,
    sycl::nd_item<3> item_ct1) {
  ध्रुव int64_t i1 = item_ct1.get_group(2);
  ध्रुव int64_t i2 = item_ct1.get_group(1);

  ध्रुव पूर्णाङ्क i11 =
      *(ध्रुव int32_t*)((ध्रुव वर्ण*)src2 + i1 * परिमाण(int32_t) + i2 * nb21);

  ध्रुव size_t nb1 = ne0 * परिमाण(भिन्न);
  ध्रुव size_t nb2 = ne1 * nb1;

  भिन्न* dst_row = (भिन्न*)((वर्ण*)dst + i1 * nb1 + i2 * nb2);
  ध्रुव भिन्न* src0_row =
      (ध्रुव भिन्न*)((ध्रुव वर्ण*)src0 + i1 * nb01 + i2 * nb02);
  ध्रुव भिन्न* src1_row = (ध्रुव भिन्न*)((ध्रुव वर्ण*)src1 + i11 * nb11);

  प्रति (int64_t i0 = item_ct1.get_local_id(2); i0 < ne0;
       i0 += item_ct1.get_local_range(2)) {
    dst_row[i0] = src0_row[i0] + src1_row[i0];
  }
}

शून्य ggml_sycl_add_id(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
  ध्रुव ggml_tensor* src0 = dst->src[0];
  ध्रुव ggml_tensor* src1 = dst->src[1];
  ध्रुव ggml_tensor* src2 = dst->src[2];

  GGML_TENSOR_TERNARY_OP_LOCALS

  GGML_ASSERT(dst->type == GGML_TYPE_F32);
  GGML_ASSERT(src0->type == GGML_TYPE_F32);
  GGML_ASSERT(src1->type == GGML_TYPE_F32);
  GGML_ASSERT(src2->type == GGML_TYPE_I32);

  GGML_ASSERT(nb00 == परिमाण(भिन्न));
  GGML_ASSERT(nb10 == परिमाण(भिन्न));
  GGML_ASSERT(nb20 == परिमाण(int32_t));

  ध्रुव भिन्न* src0_d = (ध्रुव भिन्न*)src0->data;
  ध्रुव भिन्न* src1_d = (ध्रुव भिन्न*)src1->data;
  ध्रुव int32_t* src2_d = (ध्रुव int32_t*)src2->data;
  भिन्न* dst_d = (भिन्न*)dst->data;

  ध्रुव unsigned पूर्णाङ्क max_work_group_size = ggml_sycl_info().max_work_group_sizes[ctx.device];
  GGML_ASSERT(max_work_group_size % (WARP_SIZE * WARP_SIZE) == 0);

  पूर्णाङ्क threads = std::min((unsigned पूर्णाङ्क)ne00, max_work_group_size);  // cols

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
