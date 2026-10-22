#অন্তর্ভাব <sycl/sycl.hpp>
#অন্তর্ভাব "common.hpp"
#অন্তর্ভাব "add-id.hpp"

স্থির শূন্য add_id_kernel(
    ধ্রুব ভিন্ন* src0,
    ধ্রুব ভিন্ন* src1,
    ধ্রুব int32_t* src2,
    ভিন্ন* dst,
    int64_t ne0,
    int64_t ne1,
    size_t nb01,
    size_t nb02,
    size_t nb11,
    size_t nb21,
    sycl::nd_item<3> item_ct1) {
  ধ্রুব int64_t i1 = item_ct1.get_group(2);
  ধ্রুব int64_t i2 = item_ct1.get_group(1);

  ধ্রুব পূর্ণাঙ্ক i11 =
      *(ধ্রুব int32_t*)((ধ্রুব বর্ণ*)src2 + i1 * পরিমাণ(int32_t) + i2 * nb21);

  ধ্রুব size_t nb1 = ne0 * পরিমাণ(ভিন্ন);
  ধ্রুব size_t nb2 = ne1 * nb1;

  ভিন্ন* dst_row = (ভিন্ন*)((বর্ণ*)dst + i1 * nb1 + i2 * nb2);
  ধ্রুব ভিন্ন* src0_row =
      (ধ্রুব ভিন্ন*)((ধ্রুব বর্ণ*)src0 + i1 * nb01 + i2 * nb02);
  ধ্রুব ভিন্ন* src1_row = (ধ্রুব ভিন্ন*)((ধ্রুব বর্ণ*)src1 + i11 * nb11);

  প্রতি (int64_t i0 = item_ct1.get_local_id(2); i0 < ne0;
       i0 += item_ct1.get_local_range(2)) {
    dst_row[i0] = src0_row[i0] + src1_row[i0];
  }
}

শূন্য ggml_sycl_add_id(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
  ধ্রুব ggml_tensor* src0 = dst->src[0];
  ধ্রুব ggml_tensor* src1 = dst->src[1];
  ধ্রুব ggml_tensor* src2 = dst->src[2];

  GGML_TENSOR_TERNARY_OP_LOCALS

  GGML_ASSERT(dst->type == GGML_TYPE_F32);
  GGML_ASSERT(src0->type == GGML_TYPE_F32);
  GGML_ASSERT(src1->type == GGML_TYPE_F32);
  GGML_ASSERT(src2->type == GGML_TYPE_I32);

  GGML_ASSERT(nb00 == পরিমাণ(ভিন্ন));
  GGML_ASSERT(nb10 == পরিমাণ(ভিন্ন));
  GGML_ASSERT(nb20 == পরিমাণ(int32_t));

  ধ্রুব ভিন্ন* src0_d = (ধ্রুব ভিন্ন*)src0->data;
  ধ্রুব ভিন্ন* src1_d = (ধ্রুব ভিন্ন*)src1->data;
  ধ্রুব int32_t* src2_d = (ধ্রুব int32_t*)src2->data;
  ভিন্ন* dst_d = (ভিন্ন*)dst->data;

  ধ্রুব unsigned পূর্ণাঙ্ক max_work_group_size = ggml_sycl_info().max_work_group_sizes[ctx.device];
  GGML_ASSERT(max_work_group_size % (WARP_SIZE * WARP_SIZE) == 0);

  পূর্ণাঙ্ক threads = std::min((unsigned পূর্ণাঙ্ক)ne00, max_work_group_size);  // cols

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
