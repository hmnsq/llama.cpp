#ଅନ୍ତର୍ଭାଵ <sycl/sycl.hpp>
#ଅନ୍ତର୍ଭାଵ "common.hpp"
#ଅନ୍ତର୍ଭାଵ "add-id.hpp"

ସ୍ଥିର ଶୂନ୍ୟ add_id_kernel(
    ଅଚଳ ଦଶମିକ* src0,
    ଅଚଳ ଦଶମିକ* src1,
    ଅଚଳ int32_t* src2,
    ଦଶମିକ* dst,
    int64_t ne0,
    int64_t ne1,
    size_t nb01,
    size_t nb02,
    size_t nb11,
    size_t nb21,
    sycl::nd_item<3> item_ct1) {
  ଅଚଳ int64_t i1 = item_ct1.get_group(2);
  ଅଚଳ int64_t i2 = item_ct1.get_group(1);

  ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i11 =
      *(ଅଚଳ int32_t*)((ଅଚଳ ଅକ୍ଷର*)src2 + i1 * ଆକାର(int32_t) + i2 * nb21);

  ଅଚଳ size_t nb1 = ne0 * ଆକାର(ଦଶମିକ);
  ଅଚଳ size_t nb2 = ne1 * nb1;

  ଦଶମିକ* dst_row = (ଦଶମିକ*)((ଅକ୍ଷର*)dst + i1 * nb1 + i2 * nb2);
  ଅଚଳ ଦଶମିକ* src0_row =
      (ଅଚଳ ଦଶମିକ*)((ଅଚଳ ଅକ୍ଷର*)src0 + i1 * nb01 + i2 * nb02);
  ଅଚଳ ଦଶମିକ* src1_row = (ଅଚଳ ଦଶମିକ*)((ଅଚଳ ଅକ୍ଷର*)src1 + i11 * nb11);

  ପାଇଁ (int64_t i0 = item_ct1.get_local_id(2); i0 < ne0;
       i0 += item_ct1.get_local_range(2)) {
    dst_row[i0] = src0_row[i0] + src1_row[i0];
  }
}

ଶୂନ୍ୟ ggml_sycl_add_id(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
  ଅଚଳ ggml_tensor* src0 = dst->src[0];
  ଅଚଳ ggml_tensor* src1 = dst->src[1];
  ଅଚଳ ggml_tensor* src2 = dst->src[2];

  GGML_TENSOR_TERNARY_OP_LOCALS

  GGML_ASSERT(dst->type == GGML_TYPE_F32);
  GGML_ASSERT(src0->type == GGML_TYPE_F32);
  GGML_ASSERT(src1->type == GGML_TYPE_F32);
  GGML_ASSERT(src2->type == GGML_TYPE_I32);

  GGML_ASSERT(nb00 == ଆକାର(ଦଶମିକ));
  GGML_ASSERT(nb10 == ଆକାର(ଦଶମିକ));
  GGML_ASSERT(nb20 == ଆକାର(int32_t));

  ଅଚଳ ଦଶମିକ* src0_d = (ଅଚଳ ଦଶମିକ*)src0->data;
  ଅଚଳ ଦଶମିକ* src1_d = (ଅଚଳ ଦଶମିକ*)src1->data;
  ଅଚଳ int32_t* src2_d = (ଅଚଳ int32_t*)src2->data;
  ଦଶମିକ* dst_d = (ଦଶମିକ*)dst->data;

  ଅଚଳ unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା max_work_group_size = ggml_sycl_info().max_work_group_sizes[ctx.device];
  GGML_ASSERT(max_work_group_size % (WARP_SIZE * WARP_SIZE) == 0);

  ପୂର୍ଣ୍ଣସଂଖ୍ୟା threads = std::min((unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା)ne00, max_work_group_size);  // cols

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
