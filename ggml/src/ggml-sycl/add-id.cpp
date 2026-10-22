#ਅਨ੍ਤਰ੍ਭਾਵ <sycl/sycl.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ "common.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "add-id.hpp"

ਸਥਿਰ ਖਾਲੀ ਜੋੜੋ_id_kernel(
    ਅਚਲ ਦਸ਼ਮਲਵ* src0,
    ਅਚਲ ਦਸ਼ਮਲਵ* src1,
    ਅਚਲ int32_t* src2,
    ਦਸ਼ਮਲਵ* dst,
    int64_t ne0,
    int64_t ne1,
    size_t nb01,
    size_t nb02,
    size_t nb11,
    size_t nb21,
    sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
  ਅਚਲ int64_t i1 = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2);
  ਅਚਲ int64_t i2 = ਵਸਤੂ_ct1.get_ਸਮੂਹ(1);

  ਅਚਲ ਪੂਰਨ i11 =
      *(ਅਚਲ int32_t*)((ਅਚਲ ਅੱਖਰ*)src2 + i1 * ਆਕਾਰ(int32_t) + i2 * nb21);

  ਅਚਲ size_t nb1 = ne0 * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
  ਅਚਲ size_t nb2 = ne1 * nb1;

  ਦਸ਼ਮਲਵ* dst_ਪੰਕਤੀ = (ਦਸ਼ਮਲਵ*)((ਅੱਖਰ*)dst + i1 * nb1 + i2 * nb2);
  ਅਚਲ ਦਸ਼ਮਲਵ* src0_ਪੰਕਤੀ =
      (ਅਚਲ ਦਸ਼ਮਲਵ*)((ਅਚਲ ਅੱਖਰ*)src0 + i1 * nb01 + i2 * nb02);
  ਅਚਲ ਦਸ਼ਮਲਵ* src1_ਪੰਕਤੀ = (ਅਚਲ ਦਸ਼ਮਲਵ*)((ਅਚਲ ਅੱਖਰ*)src1 + i11 * nb11);

  ਲਈ (int64_t i0 = ਵਸਤੂ_ct1.get_local_id(2); i0 < ne0;
       i0 += ਵਸਤੂ_ct1.get_local_range(2)) {
    dst_ਪੰਕਤੀ[i0] = src0_ਪੰਕਤੀ[i0] + src1_ਪੰਕਤੀ[i0];
  }
}

ਖਾਲੀ ggml_sycl_ਜੋੜੋ_id(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
  ਅਚਲ ggml_tensor* src0 = dst->src[0];
  ਅਚਲ ggml_tensor* src1 = dst->src[1];
  ਅਚਲ ggml_tensor* src2 = dst->src[2];

  GGML_TENSOR_TERNARY_OP_LOCALS

  GGML_ASSERT(dst->type == GGML_TYPE_F32);
  GGML_ASSERT(src0->type == GGML_TYPE_F32);
  GGML_ASSERT(src1->type == GGML_TYPE_F32);
  GGML_ASSERT(src2->type == GGML_TYPE_I32);

  GGML_ASSERT(nb00 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
  GGML_ASSERT(nb10 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
  GGML_ASSERT(nb20 == ਆਕਾਰ(int32_t));

  ਅਚਲ ਦਸ਼ਮਲਵ* src0_d = (ਅਚਲ ਦਸ਼ਮਲਵ*)src0->ਅੰਕੜੇ;
  ਅਚਲ ਦਸ਼ਮਲਵ* src1_d = (ਅਚਲ ਦਸ਼ਮਲਵ*)src1->ਅੰਕੜੇ;
  ਅਚਲ int32_t* src2_d = (ਅਚਲ int32_t*)src2->ਅੰਕੜੇ;
  ਦਸ਼ਮਲਵ* dst_d = (ਦਸ਼ਮਲਵ*)dst->ਅੰਕੜੇ;

  ਅਚਲ ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ max_work_ਸਮੂਹ_ਮਾਪ = ggml_sycl_info().max_work_ਸਮੂਹ_sizes[ctx.device];
  GGML_ASSERT(max_work_ਸਮੂਹ_ਮਾਪ % (WARP_ਮਾਪ * WARP_ਮਾਪ) == 0);

  ਪੂਰਨ ਧਾਗੇ = std::min((ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ)ne00, max_work_ਸਮੂਹ_ਮਾਪ);  //ਖਾਨੇ_ਗਿਣਤੀ

  ctx.stream()->parallel_for(
      sycl::nd_range<3>(
          sycl::range<3>(1, ne02, ne01) * sycl::range<3>(1, 1, ਧਾਗੇ),
          sycl::range<3>(1, 1, ਧਾਗੇ)),
      [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
        ਜੋੜੋ_id_kernel(
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
            ਵਸਤੂ_ct1);
      });
}
