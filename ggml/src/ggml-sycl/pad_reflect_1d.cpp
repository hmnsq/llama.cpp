#ಅನ್ತರ್ಭಾವ "pad_reflect_1d.hpp"

ಸ್ಥಿರ ಶೂನ್ಯ pad_reflect_1d_kernel_f32(
    ಧ್ರುವ ಶೂನ್ಯ *__restrict__ src0, ಶೂನ್ಯ *__restrict__ dst, ಧ್ರುವ int64_t ne0,
    ಧ್ರುವ int64_t ne00, ಧ್ರುವ sycl::uint3 ne01, ಧ್ರುವ int64_t ne02,
    ಧ್ರುವ int64_t ne03, ಧ್ರುವ int64_t nb00, ಧ್ರುವ int64_t nb01,
    ಧ್ರುವ int64_t nb02, ಧ್ರುವ int64_t nb03, ಧ್ರುವ int64_t nb0,
    ಧ್ರುವ int64_t nb1, ಧ್ರುವ int64_t nb2, ಧ್ರುವ int64_t nb3, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ p0,
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ p1, sycl::nd_item<3> item_ct1) {

    ಧ್ರುವ int64_t i3 = item_ct1.get_group(0);
    ಧ್ರುವ int64_t i2 = item_ct1.get_group(1);

    ಧ್ರುವ sycl::uint2 div_mod_packed =
        fast_div_modulo(item_ct1.get_group(2), ne01);
    ಧ್ರುವ int64_t tile1 = div_mod_packed.y();
    ಧ್ರುವ int64_t tile0 = div_mod_packed.x();
    ಧ್ರುವ int64_t i1 = tile1;
    ಧ್ರುವ int64_t i0 =
        item_ct1.get_local_id(2) + tile0 * item_ct1.get_local_range(2);

    ಯದಿ (i0 >= ne0 || i1 >= ne01.z() || i2 >= ne02 || i3 >= ne03) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಧ್ರುವ ವರ್ಣ *src0_ptr =
        (ಧ್ರುವ ವರ್ಣ *)src0 + i3 * nb03 + i2 * nb02 + i1 * nb01;
    ವರ್ಣ *dst_ptr = (ವರ್ಣ *)dst + i3 * nb3 + i2 * nb2 + i1 * nb1;

    ಧ್ರುವ int64_t rel_i0 = i0 - p0; // relative i0 in src0
    int64_t src_idx;

    ಯದಿ (rel_i0 < 0) {
        // Left padding - reflect
        src_idx = -rel_i0;
    } ಅನ್ಯಥಾ ಯದಿ (rel_i0 < ne00) {
        // Middle - copy
        src_idx = rel_i0;
    } ಅನ್ಯಥಾ {
        // Right padding - reflect
        src_idx = 2 * ne00 - 2 - rel_i0;
    }
    ಧ್ರುವ ಭಿನ್ನ value = *(ಧ್ರುವ ಭಿನ್ನ *)(src0_ptr + src_idx * nb00);
    *(ಭಿನ್ನ *)(dst_ptr + i0 * nb0) = value;

    GGML_UNUSED(p1);
}

ಶೂನ್ಯ ggml_sycl_op_pad_reflect_1d(ggml_backend_sycl_context &ctx,
                                 ggml_tensor *dst) {

    ಧ್ರುವ ggml_tensor *src0 = dst->src[0];
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ಧ್ರುವ int32_t *opts = (ಧ್ರುವ int32_t *)dst->op_params;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ p0 = opts[0];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ p1 = opts[1];

    ಧ್ರುವ int64_t ne00 = src0->ne[0];
    ಧ್ರುವ int64_t ne01 = src0->ne[1];
    ಧ್ರುವ sycl::uint3 ne01_packed = init_fastdiv_ಮೌಲ್ಯಗಳು(ne01);
    ಧ್ರುವ int64_t ne02 = src0->ne[2];
    ಧ್ರುವ int64_t ne03 = src0->ne[3];

    ಧ್ರುವ int64_t ne0 = dst->ne[0];

    GGML_ASSERT(ne0 == ne00 + p0 + p1);

    constexpr int64_t bx = SYCL_PAD_REFLECT_1D_BLOCK_SIZE;
    ಧ್ರುವ int64_t tiles0 = (ne0 + bx - 1) / bx;
    ಧ್ರುವ dpct::dim3 grid_dims((unsigned)(ne01 * tiles0), (unsigned)ne02,
                               (unsigned)ne03);
    ಧ್ರುವ dpct::dim3 block_dims((unsigned)bx, 1, 1);

    stream->submit([&](sycl::handler &cgh) {
        auto src0_data_ct0 = src0->data;
        auto dst_data_ct1 = dst->data;
        auto src0_nb_ct7 = src0->nb[0];
        auto src0_nb_ct8 = src0->nb[1];
        auto src0_nb_ct9 = src0->nb[2];
        auto src0_nb_ct10 = src0->nb[3];
        auto dst_nb_ct11 = dst->nb[0];
        auto dst_nb_ct12 = dst->nb[1];
        auto dst_nb_ct13 = dst->nb[2];
        auto dst_nb_ct14 = dst->nb[3];

        cgh.parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             pad_reflect_1d_kernel_f32(
                                 src0_data_ct0, dst_data_ct1, ne0, ne00,
                                 ne01_packed, ne02, ne03, src0_nb_ct7,
                                 src0_nb_ct8, src0_nb_ct9, src0_nb_ct10,
                                 dst_nb_ct11, dst_nb_ct12, dst_nb_ct13,
                                 dst_nb_ct14, p0, p1, item_ct1);
                         });
    });
}
