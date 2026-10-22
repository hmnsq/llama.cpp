#അന്തര്ഭാവ <sycl/sycl.hpp>
#അന്തര്ഭാവ "common.hpp"
#അന്തര്ഭാവ "add-id.hpp"

സ്ഥിതം ശൂന്യം add_id_kernel(
    സ്ഥിരം ദശാംശം* src0,
    സ്ഥിരം ദശാംശം* src1,
    സ്ഥിരം int32_t* src2,
    ദശാംശം* dst,
    int64_t ne0,
    int64_t ne1,
    size_t nb01,
    size_t nb02,
    size_t nb11,
    size_t nb21,
    sycl::nd_item<3> item_ct1) {
  സ്ഥിരം int64_t i1 = item_ct1.get_group(2);
  സ്ഥിരം int64_t i2 = item_ct1.get_group(1);

  സ്ഥിരം പൂർണ്ണസംഖ്യ i11 =
      *(സ്ഥിരം int32_t*)((സ്ഥിരം അക്ഷരം*)src2 + i1 * വലുപ്പം(int32_t) + i2 * nb21);

  സ്ഥിരം size_t nb1 = ne0 * വലുപ്പം(ദശാംശം);
  സ്ഥിരം size_t nb2 = ne1 * nb1;

  ദശാംശം* dst_row = (ദശാംശം*)((അക്ഷരം*)dst + i1 * nb1 + i2 * nb2);
  സ്ഥിരം ദശാംശം* src0_row =
      (സ്ഥിരം ദശാംശം*)((സ്ഥിരം അക്ഷരം*)src0 + i1 * nb01 + i2 * nb02);
  സ്ഥിരം ദശാംശം* src1_row = (സ്ഥിരം ദശാംശം*)((സ്ഥിരം അക്ഷരം*)src1 + i11 * nb11);

  വേണ്ടി (int64_t i0 = item_ct1.get_local_id(2); i0 < ne0;
       i0 += item_ct1.get_local_range(2)) {
    dst_row[i0] = src0_row[i0] + src1_row[i0];
  }
}

ശൂന്യം ggml_sycl_add_id(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
  സ്ഥിരം ggml_tensor* src0 = dst->src[0];
  സ്ഥിരം ggml_tensor* src1 = dst->src[1];
  സ്ഥിരം ggml_tensor* src2 = dst->src[2];

  GGML_TENSOR_TERNARY_OP_LOCALS

  GGML_ASSERT(dst->type == GGML_TYPE_F32);
  GGML_ASSERT(src0->type == GGML_TYPE_F32);
  GGML_ASSERT(src1->type == GGML_TYPE_F32);
  GGML_ASSERT(src2->type == GGML_TYPE_I32);

  GGML_ASSERT(nb00 == വലുപ്പം(ദശാംശം));
  GGML_ASSERT(nb10 == വലുപ്പം(ദശാംശം));
  GGML_ASSERT(nb20 == വലുപ്പം(int32_t));

  സ്ഥിരം ദശാംശം* src0_d = (സ്ഥിരം ദശാംശം*)src0->data;
  സ്ഥിരം ദശാംശം* src1_d = (സ്ഥിരം ദശാംശം*)src1->data;
  സ്ഥിരം int32_t* src2_d = (സ്ഥിരം int32_t*)src2->data;
  ദശാംശം* dst_d = (ദശാംശം*)dst->data;

  സ്ഥിരം unsigned പൂർണ്ണസംഖ്യ max_work_group_size = ggml_sycl_info().max_work_group_sizes[ctx.device];
  GGML_ASSERT(max_work_group_size % (WARP_SIZE * WARP_SIZE) == 0);

  പൂർണ്ണസംഖ്യ threads = std::min((unsigned പൂർണ്ണസംഖ്യ)ne00, max_work_group_size);  // cols

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
