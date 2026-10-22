#ਅਨ੍ਤਰ੍ਭਾਵ "pad_reflect_1d.hpp"

ਸਥਿਰ ਖਾਲੀ pad_reflect_1d_kernel_f32(
    ਅਚਲ ਖਾਲੀ *__restrict__ src0, ਖਾਲੀ *__restrict__ dst, ਅਚਲ int64_t ne0,
    ਅਚਲ int64_t ne00, ਅਚਲ sycl::uint3 ne01, ਅਚਲ int64_t ne02,
    ਅਚਲ int64_t ne03, ਅਚਲ int64_t nb00, ਅਚਲ int64_t nb01,
    ਅਚਲ int64_t nb02, ਅਚਲ int64_t nb03, ਅਚਲ int64_t nb0,
    ਅਚਲ int64_t nb1, ਅਚਲ int64_t nb2, ਅਚਲ int64_t nb3, ਅਚਲ ਪੂਰਨ p0,
    ਅਚਲ ਪੂਰਨ p1, sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {

    ਅਚਲ int64_t i3 = ਵਸਤੂ_ct1.get_ਸਮੂਹ(0);
    ਅਚਲ int64_t i2 = ਵਸਤੂ_ct1.get_ਸਮੂਹ(1);

    ਅਚਲ sycl::uint2 div_mod_packed =
        fast_div_modulo(ਵਸਤੂ_ct1.get_ਸਮੂਹ(2), ne01);
    ਅਚਲ int64_t tile1 = div_mod_packed.y();
    ਅਚਲ int64_t tile0 = div_mod_packed.x();
    ਅਚਲ int64_t i1 = tile1;
    ਅਚਲ int64_t i0 =
        ਵਸਤੂ_ct1.get_local_id(2) + tile0 * ਵਸਤੂ_ct1.get_local_range(2);

    ਜੇ (i0 >= ne0 || i1 >= ne01.z() || i2 >= ne02 || i3 >= ne03) {
        ਵਾਪਸ;
    }

    ਅਚਲ ਅੱਖਰ *src0_ptr =
        (ਅਚਲ ਅੱਖਰ *)src0 + i3 * nb03 + i2 * nb02 + i1 * nb01;
    ਅੱਖਰ *dst_ptr = (ਅੱਖਰ *)dst + i3 * nb3 + i2 * nb2 + i1 * nb1;

    ਅਚਲ int64_t rel_i0 = i0 - p0; // relative i0 in src0
    int64_t src_idx;

    ਜੇ (rel_i0 < 0) {
        // Left padding - reflect
        src_idx = -rel_i0;
    } ਨਹੀਂਤਾਂ ਜੇ (rel_i0 < ne00) {
        // Middle - copy
        src_idx = rel_i0;
    } ਨਹੀਂਤਾਂ {
        // Right padding - reflect
        src_idx = 2 * ne00 - 2 - rel_i0;
    }
    ਅਚਲ ਦਸ਼ਮਲਵ value = *(ਅਚਲ ਦਸ਼ਮਲਵ *)(src0_ptr + src_idx * nb00);
    *(ਦਸ਼ਮਲਵ *)(dst_ptr + i0 * nb0) = value;

    GGML_ਅਣਵਰਤਿਆ(p1);
}

ਖਾਲੀ ggml_sycl_op_pad_reflect_1d(ggml_backend_sycl_context &ctx,
                                 ggml_tensor *dst) {

    ਅਚਲ ggml_tensor *src0 = dst->src[0];
    dpct::ਕਤਾਰ_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ਅਚਲ int32_t *opts = (ਅਚਲ int32_t *)dst->op_params;
    ਅਚਲ ਪੂਰਨ p0 = opts[0];
    ਅਚਲ ਪੂਰਨ p1 = opts[1];

    ਅਚਲ int64_t ne00 = src0->ne[0];
    ਅਚਲ int64_t ne01 = src0->ne[1];
    ਅਚਲ sycl::uint3 ne01_packed = init_fastdiv_ਮੁੱਲਾਂ(ne01);
    ਅਚਲ int64_t ne02 = src0->ne[2];
    ਅਚਲ int64_t ne03 = src0->ne[3];

    ਅਚਲ int64_t ne0 = dst->ne[0];

    GGML_ASSERT(ne0 == ne00 + p0 + p1);

    constexpr int64_t bx = SYCL_PAD_REFLECT_1_D_ਖੰਡ_ਮਾਪ;
    ਅਚਲ int64_t tiles0 = (ne0 + bx - 1) / bx;
    ਅਚਲ dpct::dim3 grid_dims((ਬਿਨਾਨਿਸ਼ਾਨੀ)(ne01 * tiles0), (ਬਿਨਾਨਿਸ਼ਾਨੀ)ne02,
                               (ਬਿਨਾਨਿਸ਼ਾਨੀ)ne03);
    ਅਚਲ dpct::dim3 ਖੰਡ_dims((ਬਿਨਾਨਿਸ਼ਾਨੀ)bx, 1, 1);

    stream->submit([&](sycl::handler &cgh) {
        ਸਵੈਚਾਲਿਤ src0_ਅੰਕੜੇ_ct0 = src0->ਅੰਕੜੇ;
        ਸਵੈਚਾਲਿਤ dst_ਅੰਕੜੇ_ct1 = dst->ਅੰਕੜੇ;
        ਸਵੈਚਾਲਿਤ src0_nb_ct7 = src0->nb[0];
        ਸਵੈਚਾਲਿਤ src0_nb_ct8 = src0->nb[1];
        ਸਵੈਚਾਲਿਤ src0_nb_ct9 = src0->nb[2];
        ਸਵੈਚਾਲਿਤ src0_nb_ct10 = src0->nb[3];
        ਸਵੈਚਾਲਿਤ dst_nb_ct11 = dst->nb[0];
        ਸਵੈਚਾਲਿਤ dst_nb_ct12 = dst->nb[1];
        ਸਵੈਚਾਲਿਤ dst_nb_ct13 = dst->nb[2];
        ਸਵੈਚਾਲਿਤ dst_nb_ct14 = dst->nb[3];

        cgh.parallel_for(sycl::nd_range<3>(grid_dims * ਖੰਡ_dims, ਖੰਡ_dims),
                         [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                             pad_reflect_1d_kernel_f32(
                                 src0_ਅੰਕੜੇ_ct0, dst_ਅੰਕੜੇ_ct1, ne0, ne00,
                                 ne01_packed, ne02, ne03, src0_nb_ct7,
                                 src0_nb_ct8, src0_nb_ct9, src0_nb_ct10,
                                 dst_nb_ct11, dst_nb_ct12, dst_nb_ct13,
                                 dst_nb_ct14, p0, p1, ਵਸਤੂ_ct1);
                         });
    });
}
