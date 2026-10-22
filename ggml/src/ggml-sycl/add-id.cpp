#شامل <sycl/sycl.hpp>
#شامل "common.hpp"
#شامل "add-id.hpp"

ایستا تهی add_id_kernel(
    ثابت اعشاری* src0,
    ثابت اعشاری* src1,
    ثابت int32_t* src2,
    اعشاری* dst,
    int64_t ne0,
    int64_t ne1,
    size_t nb01,
    size_t nb02,
    size_t nb11,
    size_t nb21,
    sycl::nd_item<3> item_ct1) {
  ثابت int64_t i1 = item_ct1.get_group(2);
  ثابت int64_t i2 = item_ct1.get_group(1);

  ثابت صحیح i11 =
      *(ثابت int32_t*)((ثابت نویسه*)src2 + i1 * اندازه(int32_t) + i2 * nb21);

  ثابت size_t nb1 = ne0 * اندازه(اعشاری);
  ثابت size_t nb2 = ne1 * nb1;

  اعشاری* dst_row = (اعشاری*)((نویسه*)dst + i1 * nb1 + i2 * nb2);
  ثابت اعشاری* src0_row =
      (ثابت اعشاری*)((ثابت نویسه*)src0 + i1 * nb01 + i2 * nb02);
  ثابت اعشاری* src1_row = (ثابت اعشاری*)((ثابت نویسه*)src1 + i11 * nb11);

  برای (int64_t i0 = item_ct1.get_local_id(2); i0 < ne0;
       i0 += item_ct1.get_local_range(2)) {
    dst_row[i0] = src0_row[i0] + src1_row[i0];
  }
}

تهی ggml_sycl_add_id(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
  ثابت ggml_tensor* src0 = dst->src[0];
  ثابت ggml_tensor* src1 = dst->src[1];
  ثابت ggml_tensor* src2 = dst->src[2];

  GGML_TENSOR_TERNARY_OP_LOCALS

  GGML_ASSERT(dst->type == GGML_TYPE_F32);
  GGML_ASSERT(src0->type == GGML_TYPE_F32);
  GGML_ASSERT(src1->type == GGML_TYPE_F32);
  GGML_ASSERT(src2->type == GGML_TYPE_I32);

  GGML_ASSERT(nb00 == اندازه(اعشاری));
  GGML_ASSERT(nb10 == اندازه(اعشاری));
  GGML_ASSERT(nb20 == اندازه(int32_t));

  ثابت اعشاری* src0_d = (ثابت اعشاری*)src0->data;
  ثابت اعشاری* src1_d = (ثابت اعشاری*)src1->data;
  ثابت int32_t* src2_d = (ثابت int32_t*)src2->data;
  اعشاری* dst_d = (اعشاری*)dst->data;

  ثابت unsigned صحیح max_work_group_size = ggml_sycl_info().max_work_group_sizes[ctx.device];
  GGML_ASSERT(max_work_group_size % (WARP_SIZE * WARP_SIZE) == 0);

  صحیح threads = std::min((unsigned صحیح)ne00, max_work_group_size);  // cols

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
