#अन्तर्भाव "pad_reflect_1d.hpp"

स्थिर शून्य pad_reflect_1d_kernel_f32(
    स्थिरांक शून्य *__restrict__ src0, शून्य *__restrict__ dst, स्थिरांक int64_t ne0,
    स्थिरांक int64_t ne00, स्थिरांक sycl::uint3 ne01, स्थिरांक int64_t ne02,
    स्थिरांक int64_t ne03, स्थिरांक int64_t nb00, स्थिरांक int64_t nb01,
    स्थिरांक int64_t nb02, स्थिरांक int64_t nb03, स्थिरांक int64_t nb0,
    स्थिरांक int64_t nb1, स्थिरांक int64_t nb2, स्थिरांक int64_t nb3, स्थिरांक पूर्णांक p0,
    स्थिरांक पूर्णांक p1, sycl::nd_item<3> item_ct1) {

    स्थिरांक int64_t i3 = item_ct1.get_group(0);
    स्थिरांक int64_t i2 = item_ct1.get_group(1);

    स्थिरांक sycl::uint2 div_mod_packed =
        fast_div_modulo(item_ct1.get_group(2), ne01);
    स्थिरांक int64_t tile1 = div_mod_packed.y();
    स्थिरांक int64_t tile0 = div_mod_packed.x();
    स्थिरांक int64_t i1 = tile1;
    स्थिरांक int64_t i0 =
        item_ct1.get_local_id(2) + tile0 * item_ct1.get_local_range(2);

    जर (i0 >= ne0 || i1 >= ne01.z() || i2 >= ne02 || i3 >= ne03) {
        परत;
    }

    स्थिरांक अक्षर *src0_ptr =
        (स्थिरांक अक्षर *)src0 + i3 * nb03 + i2 * nb02 + i1 * nb01;
    अक्षर *dst_ptr = (अक्षर *)dst + i3 * nb3 + i2 * nb2 + i1 * nb1;

    स्थिरांक int64_t rel_i0 = i0 - p0; // relative i0 in src0
    int64_t src_idx;

    जर (rel_i0 < 0) {
        // Left padding - reflect
        src_idx = -rel_i0;
    } अन्यथा जर (rel_i0 < ne00) {
        // Middle - copy
        src_idx = rel_i0;
    } अन्यथा {
        // Right padding - reflect
        src_idx = 2 * ne00 - 2 - rel_i0;
    }
    स्थिरांक दशांश value = *(स्थिरांक दशांश *)(src0_ptr + src_idx * nb00);
    *(दशांश *)(dst_ptr + i0 * nb0) = value;

    GGML_UNUSED(p1);
}

शून्य ggml_sycl_op_pad_reflect_1d(ggml_backend_sycl_context &ctx,
                                 ggml_tensor *dst) {

    स्थिरांक ggml_tensor *src0 = dst->src[0];
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    स्थिरांक int32_t *opts = (स्थिरांक int32_t *)dst->op_params;
    स्थिरांक पूर्णांक p0 = opts[0];
    स्थिरांक पूर्णांक p1 = opts[1];

    स्थिरांक int64_t ne00 = src0->ne[0];
    स्थिरांक int64_t ne01 = src0->ne[1];
    स्थिरांक sycl::uint3 ne01_packed = init_fastdiv_मूल्ये(ne01);
    स्थिरांक int64_t ne02 = src0->ne[2];
    स्थिरांक int64_t ne03 = src0->ne[3];

    स्थिरांक int64_t ne0 = dst->ne[0];

    GGML_ASSERT(ne0 == ne00 + p0 + p1);

    constexpr int64_t bx = SYCL_PAD_REFLECT_1D_BLOCK_SIZE;
    स्थिरांक int64_t tiles0 = (ne0 + bx - 1) / bx;
    स्थिरांक dpct::dim3 grid_dims((अचिन्हित)(ne01 * tiles0), (अचिन्हित)ne02,
                               (अचिन्हित)ne03);
    स्थिरांक dpct::dim3 block_dims((अचिन्हित)bx, 1, 1);

    stream->submit([&](sycl::handler &cgh) {
        स्वतः src0_data_ct0 = src0->data;
        स्वतः dst_data_ct1 = dst->data;
        स्वतः src0_nb_ct7 = src0->nb[0];
        स्वतः src0_nb_ct8 = src0->nb[1];
        स्वतः src0_nb_ct9 = src0->nb[2];
        स्वतः src0_nb_ct10 = src0->nb[3];
        स्वतः dst_nb_ct11 = dst->nb[0];
        स्वतः dst_nb_ct12 = dst->nb[1];
        स्वतः dst_nb_ct13 = dst->nb[2];
        स्वतः dst_nb_ct14 = dst->nb[3];

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
