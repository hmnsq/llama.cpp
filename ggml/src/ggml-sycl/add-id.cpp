#ಅನ್ತರ್ಭಾವ <sycl/sycl.hpp>
#ಅನ್ತರ್ಭಾವ "common.hpp"
#ಅನ್ತರ್ಭಾವ "add-id.hpp"

ಸ್ಥಿರ ಶೂನ್ಯ ಕೂಡಿಸು_id_kernel(
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* src0,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* src1,
    ಸ್ಥಿರಾಂಕ int32_t* src2,
    ದಶಮಾಂಶ* dst,
    int64_t ne0,
    int64_t ne1,
    size_t nb01,
    size_t nb02,
    size_t nb11,
    size_t nb21,
    sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
  ಸ್ಥಿರಾಂಕ int64_t i1 = ಅಂಶ_ct1.get_ಗುಂಪು(2);
  ಸ್ಥಿರಾಂಕ int64_t i2 = ಅಂಶ_ct1.get_ಗುಂಪು(1);

  ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i11 =
      *(ಸ್ಥಿರಾಂಕ int32_t*)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ*)src2 + i1 * ಗಾತ್ರ(int32_t) + i2 * nb21);

  ಸ್ಥಿರಾಂಕ size_t nb1 = ne0 * ಗಾತ್ರ(ದಶಮಾಂಶ);
  ಸ್ಥಿರಾಂಕ size_t nb2 = ne1 * nb1;

  ದಶಮಾಂಶ* dst_ಅಡ್ಡಸಾಲು = (ದಶಮಾಂಶ*)((ಅಕ್ಷರ*)dst + i1 * nb1 + i2 * nb2);
  ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* src0_ಅಡ್ಡಸಾಲು =
      (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ*)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ*)src0 + i1 * nb01 + i2 * nb02);
  ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* src1_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ*)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ*)src1 + i11 * nb11);

  ಗಾಗಿ (int64_t i0 = ಅಂಶ_ct1.get_local_id(2); i0 < ne0;
       i0 += ಅಂಶ_ct1.get_local_range(2)) {
    dst_ಅಡ್ಡಸಾಲು[i0] = src0_ಅಡ್ಡಸಾಲು[i0] + src1_ಅಡ್ಡಸಾಲು[i0];
  }
}

ಶೂನ್ಯ ggml_sycl_ಕೂಡಿಸು_id(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
  ಸ್ಥಿರಾಂಕ ggml_tensor* src0 = dst->src[0];
  ಸ್ಥಿರಾಂಕ ggml_tensor* src1 = dst->src[1];
  ಸ್ಥಿರಾಂಕ ggml_tensor* src2 = dst->src[2];

  GGML_TENSOR_TERNARY_OP_LOCALS

  GGML_ASSERT(dst->type == GGML_TYPE_F32);
  GGML_ASSERT(src0->type == GGML_TYPE_F32);
  GGML_ASSERT(src1->type == GGML_TYPE_F32);
  GGML_ASSERT(src2->type == GGML_TYPE_I32);

  GGML_ASSERT(nb00 == ಗಾತ್ರ(ದಶಮಾಂಶ));
  GGML_ASSERT(nb10 == ಗಾತ್ರ(ದಶಮಾಂಶ));
  GGML_ASSERT(nb20 == ಗಾತ್ರ(int32_t));

  ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* src0_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ*)src0->ದತ್ತಾಂಶ;
  ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* src1_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ*)src1->ದತ್ತಾಂಶ;
  ಸ್ಥಿರಾಂಕ int32_t* src2_d = (ಸ್ಥಿರಾಂಕ int32_t*)src2->ದತ್ತಾಂಶ;
  ದಶಮಾಂಶ* dst_d = (ದಶಮಾಂಶ*)dst->ದತ್ತಾಂಶ;

  ಸ್ಥಿರಾಂಕ ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ max_work_ಗುಂಪು_ಗಾತ್ರ = ggml_sycl_info().max_work_ಗುಂಪು_sizes[ctx.device];
  GGML_ASSERT(max_work_ಗುಂಪು_ಗಾತ್ರ % (WARP_ಗಾತ್ರ * WARP_ಗಾತ್ರ) == 0);

  ಪೂರ್ಣಾಂಕ ಎಳೆಗಳು = std::min((ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ)ne00, max_work_ಗುಂಪು_ಗಾತ್ರ);  //ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ

  ctx.stream()->parallel_for(
      sycl::nd_range<3>(
          sycl::range<3>(1, ne02, ne01) * sycl::range<3>(1, 1, ಎಳೆಗಳು),
          sycl::range<3>(1, 1, ಎಳೆಗಳು)),
      [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
        ಕೂಡಿಸು_id_kernel(
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
            ಅಂಶ_ct1);
      });
}
