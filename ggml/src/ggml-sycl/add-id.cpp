#ಅನ್ತರ್ಭಾವ <sycl/sycl.hpp>
#ಅನ್ತರ್ಭಾವ "common.hpp"
#ಅನ್ತರ್ಭಾವ "add-id.hpp"

ಸ್ಥಿರ ಶೂನ್ಯ add_id_kernel(
    ಧ್ರುವ ಭಿನ್ನ* src0,
    ಧ್ರುವ ಭಿನ್ನ* src1,
    ಧ್ರುವ int32_t* src2,
    ಭಿನ್ನ* dst,
    int64_t ne0,
    int64_t ne1,
    size_t nb01,
    size_t nb02,
    size_t nb11,
    size_t nb21,
    sycl::nd_item<3> item_ct1) {
  ಧ್ರುವ int64_t i1 = item_ct1.get_group(2);
  ಧ್ರುವ int64_t i2 = item_ct1.get_group(1);

  ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i11 =
      *(ಧ್ರುವ int32_t*)((ಧ್ರುವ ವರ್ಣ*)src2 + i1 * ಪರಿಮಾಣ(int32_t) + i2 * nb21);

  ಧ್ರುವ size_t nb1 = ne0 * ಪರಿಮಾಣ(ಭಿನ್ನ);
  ಧ್ರುವ size_t nb2 = ne1 * nb1;

  ಭಿನ್ನ* dst_row = (ಭಿನ್ನ*)((ವರ್ಣ*)dst + i1 * nb1 + i2 * nb2);
  ಧ್ರುವ ಭಿನ್ನ* src0_row =
      (ಧ್ರುವ ಭಿನ್ನ*)((ಧ್ರುವ ವರ್ಣ*)src0 + i1 * nb01 + i2 * nb02);
  ಧ್ರುವ ಭಿನ್ನ* src1_row = (ಧ್ರುವ ಭಿನ್ನ*)((ಧ್ರುವ ವರ್ಣ*)src1 + i11 * nb11);

  ಪ್ರತಿ (int64_t i0 = item_ct1.get_local_id(2); i0 < ne0;
       i0 += item_ct1.get_local_range(2)) {
    dst_row[i0] = src0_row[i0] + src1_row[i0];
  }
}

ಶೂನ್ಯ ggml_sycl_add_id(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
  ಧ್ರುವ ggml_tensor* src0 = dst->src[0];
  ಧ್ರುವ ggml_tensor* src1 = dst->src[1];
  ಧ್ರುವ ggml_tensor* src2 = dst->src[2];

  GGML_TENSOR_TERNARY_OP_LOCALS

  GGML_ASSERT(dst->type == GGML_TYPE_F32);
  GGML_ASSERT(src0->type == GGML_TYPE_F32);
  GGML_ASSERT(src1->type == GGML_TYPE_F32);
  GGML_ASSERT(src2->type == GGML_TYPE_I32);

  GGML_ASSERT(nb00 == ಪರಿಮಾಣ(ಭಿನ್ನ));
  GGML_ASSERT(nb10 == ಪರಿಮಾಣ(ಭಿನ್ನ));
  GGML_ASSERT(nb20 == ಪರಿಮಾಣ(int32_t));

  ಧ್ರುವ ಭಿನ್ನ* src0_d = (ಧ್ರುವ ಭಿನ್ನ*)src0->data;
  ಧ್ರುವ ಭಿನ್ನ* src1_d = (ಧ್ರುವ ಭಿನ್ನ*)src1->data;
  ಧ್ರುವ int32_t* src2_d = (ಧ್ರುವ int32_t*)src2->data;
  ಭಿನ್ನ* dst_d = (ಭಿನ್ನ*)dst->data;

  ಧ್ರುವ unsigned ಪೂರ್ಣಾಙ್ಕ max_work_group_size = ggml_sycl_info().max_work_group_sizes[ctx.device];
  GGML_ASSERT(max_work_group_size % (WARP_SIZE * WARP_SIZE) == 0);

  ಪೂರ್ಣಾಙ್ಕ threads = std::min((unsigned ಪೂರ್ಣಾಙ್ಕ)ne00, max_work_group_size);  // cols

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
