#ضمن "pad_reflect_1d.hpp"

ساكن فراغ pad_reflect_1d_kernel_f32(
    ثابت فراغ *__restrict__ src0, فراغ *__restrict__ dst, ثابت int64_t ne0,
    ثابت int64_t ne00, ثابت sycl::uint3 ne01, ثابت int64_t ne02,
    ثابت int64_t ne03, ثابت int64_t nb00, ثابت int64_t nb01,
    ثابت int64_t nb02, ثابت int64_t nb03, ثابت int64_t nb0,
    ثابت int64_t nb1, ثابت int64_t nb2, ثابت int64_t nb3, ثابت صحيح p0,
    ثابت صحيح p1, sycl::nd_شے<3> شے_ct1) {

    ثابت int64_t i3 = شے_ct1.get_گروہ(0);
    ثابت int64_t i2 = شے_ct1.get_گروہ(1);

    ثابت sycl::uint2 div_mod_packed =
        fast_div_modulo(شے_ct1.get_گروہ(2), ne01);
    ثابت int64_t tile1 = div_mod_packed.y();
    ثابت int64_t tile0 = div_mod_packed.x();
    ثابت int64_t i1 = tile1;
    ثابت int64_t i0 =
        شے_ct1.get_local_id(2) + tile0 * شے_ct1.get_local_range(2);

    إذا (i0 >= ne0 || i1 >= ne01.z() || i2 >= ne02 || i3 >= ne03) {
        أرجع;
    }

    ثابت حرف *src0_ptr =
        (ثابت حرف *)src0 + i3 * nb03 + i2 * nb02 + i1 * nb01;
    حرف *dst_ptr = (حرف *)dst + i3 * nb3 + i2 * nb2 + i1 * nb1;

    ثابت int64_t rel_i0 = i0 - p0; // relative i0 in src0
    int64_t src_idx;

    إذا (rel_i0 < 0) {
        // Left padding - reflect
        src_idx = -rel_i0;
    } وإلا إذا (rel_i0 < ne00) {
        // Middle - copy
        src_idx = rel_i0;
    } وإلا {
        // Right padding - reflect
        src_idx = 2 * ne00 - 2 - rel_i0;
    }
    ثابت عائم value = *(ثابت عائم *)(src0_ptr + src_idx * nb00);
    *(عائم *)(dst_ptr + i0 * nb0) = value;

    GGML_بن_ورتیا(p1);
}

فراغ ggml_sycl_op_pad_reflect_1d(ggml_backend_sycl_context &ctx,
                                 ggml_tensor *dst) {

    ثابت ggml_tensor *src0 = dst->src[0];
    dpct::قطار_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ثابت int32_t *opts = (ثابت int32_t *)dst->op_params;
    ثابت صحيح p0 = opts[0];
    ثابت صحيح p1 = opts[1];

    ثابت int64_t ne00 = src0->ne[0];
    ثابت int64_t ne01 = src0->ne[1];
    ثابت sycl::uint3 ne01_packed = init_fastdiv_اقدار(ne01);
    ثابت int64_t ne02 = src0->ne[2];
    ثابت int64_t ne03 = src0->ne[3];

    ثابت int64_t ne0 = dst->ne[0];

    GGML_ASSERT(ne0 == ne00 + p0 + p1);

    constexpr int64_t bx = SYCL_PAD_REFLECT_1_D_بلاک_ناپ;
    ثابت int64_t tiles0 = (ne0 + bx - 1) / bx;
    ثابت dpct::dim3 grid_dims((بنانشانی)(ne01 * tiles0), (بنانشانی)ne02,
                               (بنانشانی)ne03);
    ثابت dpct::dim3 بلاک_dims((بنانشانی)bx, 1, 1);

    stream->submit([&](sycl::handler &cgh) {
        خودکار src0_انکڑے_ct0 = src0->انکڑے;
        خودکار dst_انکڑے_ct1 = dst->انکڑے;
        خودکار src0_nb_ct7 = src0->nb[0];
        خودکار src0_nb_ct8 = src0->nb[1];
        خودکار src0_nb_ct9 = src0->nb[2];
        خودکار src0_nb_ct10 = src0->nb[3];
        خودکار dst_nb_ct11 = dst->nb[0];
        خودکار dst_nb_ct12 = dst->nb[1];
        خودکار dst_nb_ct13 = dst->nb[2];
        خودکار dst_nb_ct14 = dst->nb[3];

        cgh.parallel_for(sycl::nd_range<3>(grid_dims * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> شے_ct1) {
                             pad_reflect_1d_kernel_f32(
                                 src0_انکڑے_ct0, dst_انکڑے_ct1, ne0, ne00,
                                 ne01_packed, ne02, ne03, src0_nb_ct7,
                                 src0_nb_ct8, src0_nb_ct9, src0_nb_ct10,
                                 dst_nb_ct11, dst_nb_ct12, dst_nb_ct13,
                                 dst_nb_ct14, p0, p1, شے_ct1);
                         });
    });
}
