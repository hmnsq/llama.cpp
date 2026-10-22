#અન્તર્ભાવ <sycl/sycl.hpp>
#અન્તર્ભાવ "common.hpp"
#અન્તર્ભાવ "add-id.hpp"

સ્થિર શૂન્ય add_id_kernel(
    અચલ દશાંશ* src0,
    અચલ દશાંશ* src1,
    અચલ int32_t* src2,
    દશાંશ* dst,
    int64_t ne0,
    int64_t ne1,
    size_t nb01,
    size_t nb02,
    size_t nb11,
    size_t nb21,
    sycl::nd_item<3> item_ct1) {
  અચલ int64_t i1 = item_ct1.get_group(2);
  અચલ int64_t i2 = item_ct1.get_group(1);

  અચલ પૂર્ણાંક i11 =
      *(અચલ int32_t*)((અચલ અક્ષર*)src2 + i1 * કદ(int32_t) + i2 * nb21);

  અચલ size_t nb1 = ne0 * કદ(દશાંશ);
  અચલ size_t nb2 = ne1 * nb1;

  દશાંશ* dst_row = (દશાંશ*)((અક્ષર*)dst + i1 * nb1 + i2 * nb2);
  અચલ દશાંશ* src0_row =
      (અચલ દશાંશ*)((અચલ અક્ષર*)src0 + i1 * nb01 + i2 * nb02);
  અચલ દશાંશ* src1_row = (અચલ દશાંશ*)((અચલ અક્ષર*)src1 + i11 * nb11);

  માટે (int64_t i0 = item_ct1.get_local_id(2); i0 < ne0;
       i0 += item_ct1.get_local_range(2)) {
    dst_row[i0] = src0_row[i0] + src1_row[i0];
  }
}

શૂન્ય ggml_sycl_add_id(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
  અચલ ggml_tensor* src0 = dst->src[0];
  અચલ ggml_tensor* src1 = dst->src[1];
  અચલ ggml_tensor* src2 = dst->src[2];

  GGML_TENSOR_TERNARY_OP_LOCALS

  GGML_ASSERT(dst->type == GGML_TYPE_F32);
  GGML_ASSERT(src0->type == GGML_TYPE_F32);
  GGML_ASSERT(src1->type == GGML_TYPE_F32);
  GGML_ASSERT(src2->type == GGML_TYPE_I32);

  GGML_ASSERT(nb00 == કદ(દશાંશ));
  GGML_ASSERT(nb10 == કદ(દશાંશ));
  GGML_ASSERT(nb20 == કદ(int32_t));

  અચલ દશાંશ* src0_d = (અચલ દશાંશ*)src0->data;
  અચલ દશાંશ* src1_d = (અચલ દશાંશ*)src1->data;
  અચલ int32_t* src2_d = (અચલ int32_t*)src2->data;
  દશાંશ* dst_d = (દશાંશ*)dst->data;

  અચલ unsigned પૂર્ણાંક max_work_group_size = ggml_sycl_info().max_work_group_sizes[ctx.device];
  GGML_ASSERT(max_work_group_size % (WARP_SIZE * WARP_SIZE) == 0);

  પૂર્ણાંક threads = std::min((unsigned પૂર્ણાંક)ne00, max_work_group_size);  // cols

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
