#அந்தர்பாவ <sycl/sycl.hpp>
#அந்தர்பாவ "common.hpp"
#அந்தர்பாவ "add-id.hpp"

நிலையான வெறுமை add_id_kernel(
    மாறிலி தசம* src0,
    மாறிலி தசம* src1,
    மாறிலி int32_t* src2,
    தசம* dst,
    int64_t ne0,
    int64_t ne1,
    size_t nb01,
    size_t nb02,
    size_t nb11,
    size_t nb21,
    sycl::nd_item<3> item_ct1) {
  மாறிலி int64_t i1 = item_ct1.get_group(2);
  மாறிலி int64_t i2 = item_ct1.get_group(1);

  மாறிலி முழு i11 =
      *(மாறிலி int32_t*)((மாறிலி எழுத்து*)src2 + i1 * அளவு(int32_t) + i2 * nb21);

  மாறிலி size_t nb1 = ne0 * அளவு(தசம);
  மாறிலி size_t nb2 = ne1 * nb1;

  தசம* dst_row = (தசம*)((எழுத்து*)dst + i1 * nb1 + i2 * nb2);
  மாறிலி தசம* src0_row =
      (மாறிலி தசம*)((மாறிலி எழுத்து*)src0 + i1 * nb01 + i2 * nb02);
  மாறிலி தசம* src1_row = (மாறிலி தசம*)((மாறிலி எழுத்து*)src1 + i11 * nb11);

  க்கு (int64_t i0 = item_ct1.get_local_id(2); i0 < ne0;
       i0 += item_ct1.get_local_range(2)) {
    dst_row[i0] = src0_row[i0] + src1_row[i0];
  }
}

வெறுமை ggml_sycl_add_id(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
  மாறிலி ggml_tensor* src0 = dst->src[0];
  மாறிலி ggml_tensor* src1 = dst->src[1];
  மாறிலி ggml_tensor* src2 = dst->src[2];

  GGML_TENSOR_TERNARY_OP_LOCALS

  GGML_ASSERT(dst->type == GGML_TYPE_F32);
  GGML_ASSERT(src0->type == GGML_TYPE_F32);
  GGML_ASSERT(src1->type == GGML_TYPE_F32);
  GGML_ASSERT(src2->type == GGML_TYPE_I32);

  GGML_ASSERT(nb00 == அளவு(தசம));
  GGML_ASSERT(nb10 == அளவு(தசம));
  GGML_ASSERT(nb20 == அளவு(int32_t));

  மாறிலி தசம* src0_d = (மாறிலி தசம*)src0->data;
  மாறிலி தசம* src1_d = (மாறிலி தசம*)src1->data;
  மாறிலி int32_t* src2_d = (மாறிலி int32_t*)src2->data;
  தசம* dst_d = (தசம*)dst->data;

  மாறிலி unsigned முழு max_work_group_size = ggml_sycl_info().max_work_group_sizes[ctx.device];
  GGML_ASSERT(max_work_group_size % (WARP_SIZE * WARP_SIZE) == 0);

  முழு threads = std::min((unsigned முழு)ne00, max_work_group_size);  // cols

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
