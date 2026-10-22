#ಅನ್ತರ್ಭಾವ "pad_reflect_1d.hpp"

ಸ್ಥಿರ ಶೂನ್ಯ pad_reflect_1d_kernel_f32(
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ src0, ಶೂನ್ಯ *__restrict__ dst, ಸ್ಥಿರಾಂಕ int64_t ne0,
    ಸ್ಥಿರಾಂಕ int64_t ne00, ಸ್ಥಿರಾಂಕ sycl::uint3 ne01, ಸ್ಥಿರಾಂಕ int64_t ne02,
    ಸ್ಥಿರಾಂಕ int64_t ne03, ಸ್ಥಿರಾಂಕ int64_t nb00, ಸ್ಥಿರಾಂಕ int64_t nb01,
    ಸ್ಥಿರಾಂಕ int64_t nb02, ಸ್ಥಿರಾಂಕ int64_t nb03, ಸ್ಥಿರಾಂಕ int64_t nb0,
    ಸ್ಥಿರಾಂಕ int64_t nb1, ಸ್ಥಿರಾಂಕ int64_t nb2, ಸ್ಥಿರಾಂಕ int64_t nb3, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ p0,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ p1, sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {

    ಸ್ಥಿರಾಂಕ int64_t i3 = ಅಂಶ_ct1.get_ಗುಂಪು(0);
    ಸ್ಥಿರಾಂಕ int64_t i2 = ಅಂಶ_ct1.get_ಗುಂಪು(1);

    ಸ್ಥಿರಾಂಕ sycl::uint2 div_mod_packed =
        fast_div_modulo(ಅಂಶ_ct1.get_ಗುಂಪು(2), ne01);
    ಸ್ಥಿರಾಂಕ int64_t tile1 = div_mod_packed.y();
    ಸ್ಥಿರಾಂಕ int64_t tile0 = div_mod_packed.x();
    ಸ್ಥಿರಾಂಕ int64_t i1 = tile1;
    ಸ್ಥಿರಾಂಕ int64_t i0 =
        ಅಂಶ_ct1.get_local_id(2) + tile0 * ಅಂಶ_ct1.get_local_range(2);

    ಆದರೆ (i0 >= ne0 || i1 >= ne01.z() || i2 >= ne02 || i3 >= ne03) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *src0_ptr =
        (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)src0 + i3 * nb03 + i2 * nb02 + i1 * nb01;
    ಅಕ್ಷರ *dst_ptr = (ಅಕ್ಷರ *)dst + i3 * nb3 + i2 * nb2 + i1 * nb1;

    ಸ್ಥಿರಾಂಕ int64_t rel_i0 = i0 - p0; // relative i0 in src0
    int64_t src_idx;

    ಆದರೆ (rel_i0 < 0) {
        // Left padding - reflect
        src_idx = -rel_i0;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (rel_i0 < ne00) {
        // Middle - copy
        src_idx = rel_i0;
    } ಇಲ್ಲದಿದ್ದರೆ {
        // Right padding - reflect
        src_idx = 2 * ne00 - 2 - rel_i0;
    }
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ value = *(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)(src0_ptr + src_idx * nb00);
    *(ದಶಮಾಂಶ *)(dst_ptr + i0 * nb0) = value;

    GGML_ಬಳಸದ(p1);
}

ಶೂನ್ಯ ggml_sycl_op_pad_reflect_1d(ggml_backend_sycl_context &ctx,
                                 ggml_tensor *dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor *src0 = dst->src[0];
    dpct::ಸರತಿ_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ಸ್ಥಿರಾಂಕ int32_t *opts = (ಸ್ಥಿರಾಂಕ int32_t *)dst->op_params;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ p0 = opts[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ p1 = opts[1];

    ಸ್ಥಿರಾಂಕ int64_t ne00 = src0->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ne01 = src0->ne[1];
    ಸ್ಥಿರಾಂಕ sycl::uint3 ne01_packed = init_fastdiv_ಮೌಲ್ಯಗಳು(ne01);
    ಸ್ಥಿರಾಂಕ int64_t ne02 = src0->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t ne03 = src0->ne[3];

    ಸ್ಥಿರಾಂಕ int64_t ne0 = dst->ne[0];

    GGML_ASSERT(ne0 == ne00 + p0 + p1);

    constexpr int64_t bx = SYCL_PAD_REFLECT_1_D_ಖಂಡ_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ int64_t tiles0 = (ne0 + bx - 1) / bx;
    ಸ್ಥಿರಾಂಕ dpct::dim3 grid_dims((ಚಿಹ್ನೆರಹಿತ)(ne01 * tiles0), (ಚಿಹ್ನೆರಹಿತ)ne02,
                               (ಚಿಹ್ನೆರಹಿತ)ne03);
    ಸ್ಥಿರಾಂಕ dpct::dim3 ಖಂಡ_dims((ಚಿಹ್ನೆರಹಿತ)bx, 1, 1);

    stream->submit([&](sycl::handler &cgh) {
        ಸ್ವಯಂಚಾಲಿತ src0_ದತ್ತಾಂಶ_ct0 = src0->ದತ್ತಾಂಶ;
        ಸ್ವಯಂಚಾಲಿತ dst_ದತ್ತಾಂಶ_ct1 = dst->ದತ್ತಾಂಶ;
        ಸ್ವಯಂಚಾಲಿತ src0_nb_ct7 = src0->nb[0];
        ಸ್ವಯಂಚಾಲಿತ src0_nb_ct8 = src0->nb[1];
        ಸ್ವಯಂಚಾಲಿತ src0_nb_ct9 = src0->nb[2];
        ಸ್ವಯಂಚಾಲಿತ src0_nb_ct10 = src0->nb[3];
        ಸ್ವಯಂಚಾಲಿತ dst_nb_ct11 = dst->nb[0];
        ಸ್ವಯಂಚಾಲಿತ dst_nb_ct12 = dst->nb[1];
        ಸ್ವಯಂಚಾಲಿತ dst_nb_ct13 = dst->nb[2];
        ಸ್ವಯಂಚಾಲಿತ dst_nb_ct14 = dst->nb[3];

        cgh.parallel_for(sycl::nd_range<3>(grid_dims * ಖಂಡ_dims, ಖಂಡ_dims),
                         [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                             pad_reflect_1d_kernel_f32(
                                 src0_ದತ್ತಾಂಶ_ct0, dst_ದತ್ತಾಂಶ_ct1, ne0, ne00,
                                 ne01_packed, ne02, ne03, src0_nb_ct7,
                                 src0_nb_ct8, src0_nb_ct9, src0_nb_ct10,
                                 dst_nb_ct11, dst_nb_ct12, dst_nb_ct13,
                                 dst_nb_ct14, p0, p1, ಅಂಶ_ct1);
                         });
    });
}
