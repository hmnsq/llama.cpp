#شامل <sycl/sycl.hpp>
#شامل "common.hpp"
#شامل "add-id.hpp"

ساکن خالی جوڑو_id_kernel(
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
    sycl::nd_شے<3> شے_ct1) {
  ثابت int64_t i1 = شے_ct1.get_گروہ(2);
  ثابت int64_t i2 = شے_ct1.get_گروہ(1);

  ثابت صحیح i11 =
      *(ثابت int32_t*)((ثابت حرف*)src2 + i1 * حجم(int32_t) + i2 * nb21);

  ثابت size_t nb1 = ne0 * حجم(اعشاری);
  ثابت size_t nb2 = ne1 * nb1;

  اعشاری* dst_صف = (اعشاری*)((حرف*)dst + i1 * nb1 + i2 * nb2);
  ثابت اعشاری* src0_صف =
      (ثابت اعشاری*)((ثابت حرف*)src0 + i1 * nb01 + i2 * nb02);
  ثابت اعشاری* src1_صف = (ثابت اعشاری*)((ثابت حرف*)src1 + i11 * nb11);

  برائے (int64_t i0 = شے_ct1.get_local_id(2); i0 < ne0;
       i0 += شے_ct1.get_local_range(2)) {
    dst_صف[i0] = src0_صف[i0] + src1_صف[i0];
  }
}

خالی ggml_sycl_جوڑو_id(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
  ثابت ggml_tensor* src0 = dst->src[0];
  ثابت ggml_tensor* src1 = dst->src[1];
  ثابت ggml_tensor* src2 = dst->src[2];

  GGML_TENSOR_TERNARY_OP_LOCALS

  GGML_ASSERT(dst->type == GGML_TYPE_F32);
  GGML_ASSERT(src0->type == GGML_TYPE_F32);
  GGML_ASSERT(src1->type == GGML_TYPE_F32);
  GGML_ASSERT(src2->type == GGML_TYPE_I32);

  GGML_ASSERT(nb00 == حجم(اعشاری));
  GGML_ASSERT(nb10 == حجم(اعشاری));
  GGML_ASSERT(nb20 == حجم(int32_t));

  ثابت اعشاری* src0_d = (ثابت اعشاری*)src0->ڈیٹا;
  ثابت اعشاری* src1_d = (ثابت اعشاری*)src1->ڈیٹا;
  ثابت int32_t* src2_d = (ثابت int32_t*)src2->ڈیٹا;
  اعشاری* dst_d = (اعشاری*)dst->ڈیٹا;

  ثابت غیرعلامتی صحیح max_work_گروہ_سائز = ggml_sycl_info().max_work_گروہ_sizes[ctx.device];
  GGML_ASSERT(max_work_گروہ_سائز % (WARP_سائز * WARP_سائز) == 0);

  صحیح دھاگے = std::min((غیرعلامتی صحیح)ne00, max_work_گروہ_سائز);  //ستون_تعداد

  ctx.stream()->parallel_for(
      sycl::nd_range<3>(
          sycl::range<3>(1, ne02, ne01) * sycl::range<3>(1, 1, دھاگے),
          sycl::range<3>(1, 1, دھاگے)),
      [=](sycl::nd_شے<3> شے_ct1) {
        جوڑو_id_kernel(
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
            شے_ct1);
      });
}
