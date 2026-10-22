#अन्तर्भाव "convert.hpp"
#अन्तर्भाव "dequantize.hpp"
#अन्तर्भाव "presets.hpp"

प्रतिरूपम् <पूर्णाङ्क qk, पूर्णाङ्क qr, dequantize_kernel_t dequantize_kernel, प्रकारनाम dst_t>
स्थिर शून्य dequantize_खण्डम्(ध्रुव शून्य * __restrict__ vx, dst_t * __restrict__ y, ध्रुव int64_t k,
                             ध्रुव sycl::nd_item<3> &item_ct1) {
    ध्रुव int64_t i = 2 * (item_ct1.get_local_range(2) * item_ct1.get_गणः(2) +
                       item_ct1.get_local_id(2));

    यदि (i >= k) {
        प्रत्यागम;
    }

    ध्रुव int64_t ib = i/qk; // block index
    ध्रुव int64_t iqs = (i%qk)/qr; // quant index
    ध्रुव int64_t iybs = i - i%qk; // y block start index
    ध्रुव int64_t y_विस्थापनम् = qr == 1 ? 1 : qk/2;

    // dequantize
    dfloat2 v;
    dequantize_kernel(vx, ib, iqs, v);

    y[iybs + iqs + 0] = v.x();
    y[iybs + iqs + y_विस्थापनम्] = v.y();
}

प्रतिरूपम् <पूर्णाङ्क qk, पूर्णाङ्क qr, dequantize_kernel_t dequantize_kernel, प्रकारनाम dst_t>
स्थिर शून्य dequantize_खण्डम्_sycl(ध्रुव शून्य *__restrict__ vx,
                                  dst_t *__restrict__ y, ध्रुव int64_t k,
                                  dpct::queue_ptr stream) {
    ध्रुव int64_t अङ्कः_खण्डानि = (k + 2*SYCL_DEQUANTIZE_खण्डम्_आकार - 1) / (2*SYCL_DEQUANTIZE_खण्डम्_आकार);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, अङ्कः_खण्डानि) *
                    sycl::range<3>(1, 1, SYCL_DEQUANTIZE_खण्डम्_आकार),
                sycl::range<3>(1, 1, SYCL_DEQUANTIZE_खण्डम्_आकार)),
            [=](sycl::nd_item<3> item_ct1) {
                dequantize_खण्डम्<qk, qr, dequantize_kernel>(vx, y, k, item_ct1);
            });
    }
}

प्रतिरूपम् <प्रकारनाम dst_t>
स्थिर शून्य dequantize_row_q2_K_sycl(ध्रुव शून्य *vx, dst_t *y, ध्रुव int64_t k,
                                     dpct::queue_ptr stream) {
    ध्रुव int64_t nb = k / QK_K;
#यदि QK_K == 256
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 64),
                                               sycl::range<3>(1, 1, 64)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_खण्डम्_q2_K(vx, y, item_ct1);
                             });
    }
#अन्यथा
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_खण्डम्_q2_K(vx, y, item_ct1);
                             });
    }

#यदि_अन्तः
}

प्रतिरूपम् <प्रकारनाम dst_t>
स्थिर शून्य dequantize_row_q2_K_sycl_reorder(ध्रुव शून्य *vx, dst_t *y, ध्रुव int64_t k,
                                             dpct::queue_ptr stream) {
    ध्रुव int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
        [=](sycl::nd_item<3> item_ct1) {
            dequantize_खण्डम्_q2_K_reorder(vx, y + item_ct1.get_गणः(2) * QK_K, item_ct1.get_गणः(2), item_ct1, nb);
        });
}

प्रतिरूपम् <प्रकारनाम dst_t>
स्थिर शून्य dequantize_row_q3_K_sycl(ध्रुव शून्य *vx, dst_t *y, ध्रुव int64_t k,
                                     dpct::queue_ptr stream) {
    ध्रुव int64_t nb = k / QK_K;
#यदि QK_K == 256
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 64),
                                               sycl::range<3>(1, 1, 64)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_खण्डम्_q3_K(vx, y, item_ct1);
                             });
    }
#अन्यथा
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_खण्डम्_q3_K(vx, y, item_ct1);
                             });
    }
#यदि_अन्तः
}

प्रतिरूपम् <प्रकारनाम dst_t>
स्थिर शून्य dequantize_row_q3_K_sycl_reorder(ध्रुव शून्य *vx, dst_t *y, ध्रुव int64_t k,
                                             dpct::queue_ptr stream) {
    ध्रुव int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
        [=](sycl::nd_item<3> item_ct1) {
            dequantize_खण्डम्_q3_K_reorder(vx, y + item_ct1.get_गणः(2) * QK_K, item_ct1.get_गणः(2), item_ct1, nb);
        });
}

प्रतिरूपम् <प्रकारनाम dst_t>
स्थिर शून्य dequantize_row_q4_0_sycl(ध्रुव शून्य *vx, dst_t *y, ध्रुव int64_t k,
                                     dpct::queue_ptr stream) {
    ध्रुव int64_t nb32 = k / 32;
    ध्रुव int64_t nb = (k + 255) / 256;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_खण्डम्_q4_0(vx, y, nb32, item_ct1);
                             });
    }
}

प्रतिरूपम् <प्रकारनाम dst_t>
स्थिर शून्य dequantize_row_q4_0_sycl_reorder(ध्रुव शून्य *vx, dst_t *y, ध्रुव int64_t k,
                                     dpct::queue_ptr stream) {

    dpct::has_capability_or_fail(stream->get_device(),
                                    {sycl::aspect::fp16});

    GGML_ASSERT(k % QK4_0 == 0);
    ध्रुव int64_t n_groups = ceil_div(k / 2, SYCL_DEQUANTIZE_खण्डम्_आकार);
    // decrease global range when it exceeds the max int; the kernel loops to cover the rest
    ध्रुव int64_t wg_आकार  = downsample_sycl_global_range(n_groups, SYCL_DEQUANTIZE_खण्डम्_आकार);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, n_groups) *
        sycl::range<3>(1, 1, wg_आकार),
        sycl::range<3>(1, 1, wg_आकार)),
        [=](sycl::nd_item<3> item_ct1) {
            dequantize_खण्डम्_q4_0_reorder(vx, y, k, item_ct1);
        });
}

प्रतिरूपम् <प्रकारनाम dst_t>
स्थिर शून्य dequantize_row_q8_0_sycl_reorder(ध्रुव शून्य *vx, dst_t *y, ध्रुव int64_t k,
                                     dpct::queue_ptr stream) {

    dpct::has_capability_or_fail(stream->get_device(),
                                    {sycl::aspect::fp16});

    // flat range<1>, two pairs (4 elements) per item: no nd_range/group math, one vectorized store.
    // k % QK8_0 == 0 implies k % 4 == 0, so the wide launch covers the row exactly (no tail).
    GGML_ASSERT(k % QK8_0 == 0);
    stream->parallel_for(sycl::range<1>((size_t) (k / 4)),
        [=](sycl::id<1> idx) {
            dequantize_खण्डम्_q8_0_reorder_wide(vx, y, k, idx);
        });
}

प्रतिरूपम् <प्रकारनाम dst_t>
स्थिर शून्य dequantize_row_q4_1_sycl(ध्रुव शून्य *vx, dst_t *y, ध्रुव int64_t k,
                                     dpct::queue_ptr stream) {
    ध्रुव int64_t nb32 = k / 32;
    ध्रुव int64_t nb = (k + 255) / 256;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_खण्डम्_q4_1(vx, y, nb32, item_ct1);
                             });
    }
}


प्रतिरूपम् <प्रकारनाम dst_t>
स्थिर शून्य dequantize_row_q4_K_sycl(ध्रुव शून्य *vx, dst_t *y, ध्रुव int64_t k,
                                     dpct::queue_ptr stream) {
    ध्रुव int64_t nb = k / QK_K;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) nb * SYCL_DEQK_WI_प्रत्येकम्_खण्डम्),
                                           sycl::range<1>((size_t) SYCL_DEQK_WI_प्रत्येकम्_खण्डम्)),
                         [=](sycl::nd_item<1> it) { dequantize_खण्डम्_q4_K_wide(vx, y, nb, it); });
}

प्रतिरूपम् <प्रकारनाम dst_t>
स्थिर शून्य dequantize_row_q4_K_sycl_reorder(ध्रुव शून्य * vx, dst_t * y, ध्रुव int64_t k, dpct::queue_ptr stream) {
    ध्रुव int64_t nb = k / QK_K;
    ध्रुव size_t  local_आकार  = 32;
    ध्रुव size_t  global_आकार = nb * local_आकार;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint8_t, 1> scale_local_acc(sycl::range<1>(12), cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_आकार), sycl::range<1>(local_आकार)),
                         [=](sycl::nd_item<1> item_ct1) {
                             dequantize_खण्डम्_q4_K_reorder(vx, y + item_ct1.get_गणः(0) * QK_K, item_ct1.get_गणः(0), get_pointer(scale_local_acc), item_ct1, nb);
                         });
    });
}

प्रतिरूपम् <प्रकारनाम dst_t>
स्थिर शून्य dequantize_row_q5_K_sycl(ध्रुव शून्य *vx, dst_t *y, ध्रुव int64_t k,
                                     dpct::queue_ptr stream) {
    ध्रुव int64_t nb = k / QK_K;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) nb * SYCL_DEQK_WI_प्रत्येकम्_खण्डम्),
                                           sycl::range<1>((size_t) SYCL_DEQK_WI_प्रत्येकम्_खण्डम्)),
                         [=](sycl::nd_item<1> it) { dequantize_खण्डम्_q5_K_wide(vx, y, nb, it); });
}

प्रतिरूपम् <प्रकारनाम dst_t>
स्थिर शून्य dequantize_row_q5_K_sycl_reorder(ध्रुव शून्य * vx, dst_t * y, ध्रुव int64_t k, dpct::queue_ptr stream) {
    ध्रुव int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint8_t, 1> scale_local_acc(sycl::range<1>(K_SCALE_आकार), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
            [=](sycl::nd_item<3> item_ct1) {
                dequantize_खण्डम्_q5_K_reorder(vx, y + item_ct1.get_गणः(2) * QK_K, item_ct1.get_गणः(2), get_pointer(scale_local_acc), item_ct1, nb);
            });
    });
}

प्रतिरूपम् <प्रकारनाम dst_t>
स्थिर शून्य dequantize_row_q6_K_sycl(ध्रुव शून्य *vx, dst_t *y, ध्रुव int64_t k,
                                     dpct::queue_ptr stream) {
    ध्रुव int64_t nb = k / QK_K;
#यदि QK_K == 256
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 64),
                                               sycl::range<3>(1, 1, 64)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_खण्डम्_q6_K(vx, y, item_ct1);
                             });
    }
#अन्यथा
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_खण्डम्_q6_K(vx, y, item_ct1);
                             });
    }

#यदि_अन्तः
}

प्रतिरूपम् <प्रकारनाम dst_t>
स्थिर शून्य dequantize_row_q6_K_sycl_reorder(ध्रुव शून्य * vx, dst_t * y, ध्रुव int64_t k, dpct::queue_ptr stream) {
    ध्रुव int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
        [=](sycl::nd_item<3> item_ct1) { dequantize_खण्डम्_q6_K_reorder(vx, y + item_ct1.get_गणः(2) * QK_K, item_ct1.get_गणः(2), item_ct1, nb); });
}

प्रतिरूपम् <प्रकारनाम dst_t>
स्थिर शून्य dequantize_row_iq1_s_sycl(ध्रुव शून्य *vx, dst_t *y, ध्रुव int64_t k,
                                        dpct::queue_ptr stream) {
    ध्रुव int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_खण्डम्_iq1_s(
                                     vx, y, item_ct1, iq1s_grid_gpu
                                     );
                             });
        });
    }
}

प्रतिरूपम् <प्रकारनाम dst_t>
स्थिर शून्य dequantize_row_iq1_m_sycl(ध्रुव शून्य *vx, dst_t *y, ध्रुव int64_t k,
                                        dpct::queue_ptr stream) {
    ध्रुव int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_खण्डम्_iq1_m(
                                     vx, y, item_ct1, iq1s_grid_gpu
                                     );
                             });
        });
    }
}

प्रतिरूपम् <प्रकारनाम dst_t>
स्थिर शून्य dequantize_row_iq2_xxs_sycl(ध्रुव शून्य *vx, dst_t *y, ध्रुव int64_t k,
                                        dpct::queue_ptr stream) {
    ध्रुव int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_खण्डम्_iq2_xxs(
                                     vx, y, item_ct1, iq2xxs_grid,
                                     ksigns_iq2xs, kmask_iq2xs);
                             });
        });
    }
}

प्रतिरूपम् <प्रकारनाम dst_t>
स्थिर शून्य dequantize_row_iq2_xs_sycl(ध्रुव शून्य *vx, dst_t *y, ध्रुव int64_t k,
                                       dpct::queue_ptr stream) {
    ध्रुव int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_खण्डम्_iq2_xs(
                                     vx, y, item_ct1, iq2xs_grid,
                                     ksigns_iq2xs, kmask_iq2xs);
                             });
        });
    }
}

प्रतिरूपम् <प्रकारनाम dst_t>
स्थिर शून्य dequantize_row_iq2_s_sycl(ध्रुव शून्य *vx, dst_t *y, ध्रुव int64_t k,
                                      dpct::queue_ptr stream) {
    ध्रुव int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_खण्डम्_iq2_s(vx, y, item_ct1);
                             });
        });
    }
}


प्रतिरूपम् <प्रकारनाम dst_t>
स्थिर शून्य dequantize_row_iq3_xxs_sycl(ध्रुव शून्य *vx, dst_t *y, ध्रुव int64_t k,
                                        dpct::queue_ptr stream) {
    ध्रुव int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_खण्डम्_iq3_xxs(
                                     vx, y, item_ct1, iq3xxs_grid,
                                     ksigns_iq2xs, kmask_iq2xs);
                             });
        });
    }
}

प्रतिरूपम् <प्रकारनाम dst_t>
स्थिर शून्य dequantize_row_iq3_xxs_sycl_reorder(ध्रुव शून्य * vx, dst_t * y, ध्रुव int64_t k,
                                                dpct::queue_ptr stream) {
    ध्रुव int64_t nb = k / QK_K;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_item<3> item_ct1) {
            dequantize_खण्डम्_iq3_xxs_reorder(
                vx, y + item_ct1.get_गणः(2) * QK_K, item_ct1.get_गणः(2), item_ct1, nb,
                iq3xxs_grid, ksigns_iq2xs, kmask_iq2xs);
        });
}

प्रतिरूपम् <प्रकारनाम dst_t>
स्थिर शून्य dequantize_row_iq3_s_sycl(ध्रुव शून्य *vx, dst_t *y, ध्रुव int64_t k,
                                        dpct::queue_ptr stream) {
    ध्रुव int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_खण्डम्_iq3_s(
                                     vx, y, item_ct1, kmask_iq2xs, iq3s_grid);
                             });
        });
    }
}

प्रतिरूपम् <प्रकारनाम dst_t>
स्थिर शून्य dequantize_row_iq3_s_sycl_reorder(ध्रुव शून्य * vx, dst_t * y, ध्रुव int64_t k,
                                              dpct::queue_ptr stream) {
    ध्रुव int64_t nb = k / QK_K;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_item<3> item_ct1) {
            dequantize_खण्डम्_iq3_s_reorder(vx, y + item_ct1.get_गणः(2) * QK_K, item_ct1.get_गणः(2),
                                           item_ct1, nb, kmask_iq2xs, iq3s_grid);
        });
}

प्रतिरूपम् <प्रकारनाम dst_t>
स्थिर शून्य dequantize_row_iq4_xs_sycl(ध्रुव शून्य *vx, dst_t *y, ध्रुव int64_t k,
                                       dpct::queue_ptr stream) {
    ध्रुव int64_t nb = (k + QK_K - 1) / QK_K;
#यदि QK_K == 64
    dequantize_row_iq4_nl_sycl(vx, y, k, stream);
#अन्यथा
      {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                  cgh.parallel_for(
                      sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                            sycl::range<3>(1, 1, 32),
                                        sycl::range<3>(1, 1, 32)),
                      [=](sycl::nd_item<3> item_ct1) {
                            dequantize_खण्डम्_iq4_xs(vx, y, item_ct1);
                      });
            });
      }
#यदि_अन्तः
}

प्रतिरूपम् <प्रकारनाम dst_t>
स्थिर शून्य dequantize_row_iq4_nl_sycl(ध्रुव शून्य *vx, dst_t *y, ध्रुव int64_t k,
                                       dpct::queue_ptr stream) {
    ध्रुव int64_t nb = (k + QK_K - 1) / QK_K;
      {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                  cgh.parallel_for(
                      sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                            sycl::range<3>(1, 1, 32),
                                        sycl::range<3>(1, 1, 32)),
                      [=](sycl::nd_item<3> item_ct1) {
                            dequantize_खण्डम्_iq4_nl(vx, y, item_ct1);
                      });
            });
      }
}

प्रतिरूपम् <प्रकारनाम dst_t>
स्थिर शून्य dequantize_row_mxfp4_sycl(ध्रुव शून्य * vx, dst_t * y, ध्रुव int64_t k, dpct::queue_ptr stream) {
    ध्रुव पूर्णाङ्क nb = (k + QK_K - 1) / QK_K;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_item<3> item_ct1) {
            dequantize_खण्डम्_mxfp4(vx, y, item_ct1);
        });
}

प्रतिरूपम् <प्रकारनाम dst_t>
स्थिर शून्य dequantize_row_mxfp4_sycl_reorder(ध्रुव शून्य * vx, dst_t * y, ध्रुव int64_t k, dpct::queue_ptr stream) {
    GGML_ASSERT(k % QK_MXFP4 == 0);
    ध्रुव पूर्णाङ्क n_warp = (k / QK_MXFP4 + WARP_आकार - 1) / WARP_आकार;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, n_warp) * sycl::range<3>(1, 1, WARP_आकार),
                          sycl::range<3>(1, 1, WARP_आकार)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
            dequantize_खण्डम्_mxfp4_reorder(vx, y, k, item_ct1);
        });
}

प्रतिरूपम् <प्रकारनाम dst_t>
स्थिर शून्य dequantize_row_nvfp4_sycl(ध्रुव शून्य * vx, dst_t * y, ध्रुव int64_t k, dpct::queue_ptr stream) {
    GGML_ASSERT(k % QK_NVFP4 == 0);
    ध्रुव पूर्णाङ्क nb = k / QK_NVFP4;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_item<3> /*item_ct1*/) {
            dequantize_खण्डम्_nvfp4(vx, y, k);
        });
}


प्रतिरूपम् <पूर्णाङ्क qk, पूर्णाङ्क qr, dequantize_kernel_t dequantize_kernel, प्रकारनाम dst_t>
स्थिर शून्य dequantize_खण्डम्_nc(ध्रुव शून्य * __restrict__ vx, dst_t * __restrict__ y,
        ध्रुव int64_t ne00, ध्रुव int64_t ne01, ध्रुव int64_t ne02,
        ध्रुव int64_t s01, ध्रुव int64_t s02, ध्रुव int64_t s03) {
    स्वचालित          item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ध्रुव int64_t i00 = 2 * (int64_t(item_ct1.get_local_range(2)) * item_ct1.get_गणः(2) + item_ct1.get_local_id(2));

    यदि (i00 >= ne00) {
        प्रत्यागम;
    }

    ध्रुव int64_t i01 = item_ct1.get_गणः(1);
    ध्रुव int64_t i02 = item_ct1.get_गणः(0) % ne02;
    ध्रुव int64_t i03 = item_ct1.get_गणः(0) / ne02;

    ध्रुव int64_t ibx0 = i03*s03 + i02*s02 + i01*s01;

    ध्रुव int64_t ib = ibx0 + i00/qk; // block index
    ध्रुव int64_t iqs = (i00%qk)/qr; // quant index
    ध्रुव int64_t iybs = i00 - i00%qk; // y block start index
    ध्रुव int64_t y_विस्थापनम् = qr == 1 ? 1 : qk/2;

    // dequantize
    #यदि_परिभाषितम् GGML_SYCL_F16
        sycl::half2 v;
    #अन्यथा
        sycl::float2 v;
    #यदि_अन्तः

    dequantize_kernel(vx, ib, iqs, v);

    ध्रुव int64_t iy0 = ((i03*ne02 + i02)*ne01 + i01)*ne00 + iybs + iqs;
    y[iy0 + 0]        = ggml_sycl_cast<dst_t>(v.x());
    y[iy0 + y_विस्थापनम्] = ggml_sycl_cast<dst_t>(v.y());
}


प्रतिरूपम् <पूर्णाङ्क qk, पूर्णाङ्क qr, dequantize_kernel_t dequantize_kernel, प्रकारनाम dst_t>
स्थिर शून्य dequantize_खण्डम्_nc_sycl(ध्रुव शून्य *    vx,
                                  dst_t *         y,
                                  ध्रुव int64_t   ne00,
                                  ध्रुव int64_t   ne01,
                                  ध्रुव int64_t   ne02,
                                  ध्रुव int64_t   ne03,
                                  ध्रुव int64_t   s01,
                                  ध्रुव int64_t   s02,
                                  ध्रुव int64_t   s03,
                                  dpct::queue_ptr stream) {
    ध्रुव dpct::dim3 अङ्कः_खण्डानि((ne00 + 2 * SYCL_DEQUANTIZE_खण्डम्_आकार - 1) / (2 * SYCL_DEQUANTIZE_खण्डम्_आकार), ne01,
                                ne02 * ne03);
    stream->parallel_for(sycl::nd_range<3>(अङ्कः_खण्डानि * sycl::range<3>(1, 1, SYCL_DEQUANTIZE_खण्डम्_आकार),
                                           sycl::range<3>(1, 1, SYCL_DEQUANTIZE_खण्डम्_आकार)),
                         [=](sycl::nd_item<3> item_ct1) {
                             GGML_अप्रयुक्तम्(item_ct1);
                             dequantize_खण्डम्_nc<qk, qr, dequantize_kernel>(vx, y, ne00, ne01, ne02, s01, s02, s03);
                         });
}
प्रतिरूपम् <प्रकारनाम src_t, प्रकारनाम dst_t>
स्थिर शून्य convert_unary_nc(ध्रुव शून्य * __restrict__ vx, dst_t * __restrict__ y, ध्रुव int64_t ne00, ध्रुव int64_t ne01,
                          ध्रुव int64_t ne02, ध्रुव int64_t s01, ध्रुव int64_t s02, ध्रुव int64_t s03,
                          ध्रुव sycl::nd_item<3> & item_ct1) {

    ध्रुव int64_t work_गणः_आकार = item_ct1.get_local_range(2);
    ध्रुव int64_t global_id       = item_ct1.get_local_id(2) + work_गणः_आकार * item_ct1.get_गणः(2);

    ध्रुव int64_t i01 = item_ct1.get_गणः(1);
    ध्रुव int64_t i02 = item_ct1.get_गणः(0) % ne02;
    ध्रुव int64_t i03 = item_ct1.get_गणः(0) / ne02;

    // make each work-item deal with more elements since sycl global range can not exceed max int
    ध्रुव src_t * x = static_cast<ध्रुव src_t *>(vx);
    ध्रुव int64_t ix = i03 * s03 + i02 * s02 + i01 * s01;
    ध्रुव int64_t iy = ((i03 * ne02 + i02) * ne01 + i01) * ne00;

#निर्देशः unroll
    प्रति (int64_t i00 = global_id; i00 < ne00; i00 += work_गणः_आकार * item_ct1.get_गणः_range(2)) {
        y[iy + i00] = static_cast<dst_t>(x[ix + i00]);
    }
}

प्रतिरूपम् <प्रकारनाम src_t, प्रकारनाम dst_t>
स्थिर शून्य convert_unary_nc_sycl(ध्रुव शून्य * __restrict__ vx, dst_t * __restrict__ y,
                                  ध्रुव int64_t ne00, ध्रुव int64_t ne01, ध्रुव int64_t ne02, ध्रुव int64_t ne03,
                                  ध्रुव int64_t s01, ध्रुव int64_t s02, ध्रुव int64_t s03, dpct::queue_ptr queue) {
    dpct::has_capability_or_fail(queue->get_device(), { sycl::aspect::fp16 });

    sycl::range<3> global_आकार(ne02 * ne03, ne01, ceil_div(ne00, SYCL_DEQUANTIZE_खण्डम्_आकार));

    // decrease global range when it exceeds the max int
    // TODO: Downsample logic is separated from the kernel, a rewrite is desirable
    int64_t        downsized_workgroup = downsample_sycl_global_range(global_आकार[0], SYCL_DEQUANTIZE_खण्डम्_आकार);
    sycl::range<3> workgroup_आकार(1, 1, downsized_workgroup);

    queue->parallel_for(sycl::nd_range<3>(global_आकार * workgroup_आकार, workgroup_आकार), [=](sycl::nd_item<3> item_ct1) {
        convert_unary_nc<src_t>(vx, y, ne00, ne01, ne02, s01, s02, s03, item_ct1);
    });
}

प्रतिरूपम् <प्रकारनाम src_t, प्रकारनाम dst_t>
स्थिर शून्य convert_unary_sycl(ध्रुव शून्य * vx, dst_t * y, ध्रुव int64_t k, dpct::queue_ptr queue) {
    convert_unary_nc_sycl<src_t>(vx, y, k, 1, 1, 1, k, k, k, queue);
}


स्थिर तार्किक is_reordered_src0(ध्रुव ggml_tensor * dst) {
    प्रत्यागम dst->src[0]->extra &&
           ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder;
}

प्रतिरूपम् <प्रकारनाम dst_t>
स्थिर गन्तव्यम्_t_sycl_t<dst_t> ggml_get_dequant_sycl(ggml_type type, ggml_tensor * dst) {
    विकल्प (type) {
        पक्ष GGML_TYPE_Q1_0:
            प्रत्यागम dequantize_खण्डम्_sycl<QK1_0, QR1_0, dequantize_q1_0, dst_t>;
        पक्ष GGML_TYPE_Q2_0:
            प्रत्यागम dequantize_खण्डम्_sycl<QK2_0, QR2_0, dequantize_q2_0, dst_t>;
        पक्ष GGML_TYPE_Q4_0:
            यदि (is_reordered_src0(dst)) {
                प्रत्यागम dequantize_row_q4_0_sycl_reorder;
            } अन्यथा यदि constexpr (std::is_same_v<dst_t, भिन्न>) {
                प्रत्यागम dequantize_row_q4_0_sycl;
            } अन्यथा {
                प्रत्यागम dequantize_खण्डम्_sycl<QK4_0, QR4_0, dequantize_q4_0, dst_t>;
            }
        पक्ष GGML_TYPE_Q4_1:
            यदि constexpr (std::is_same_v<dst_t, भिन्न>) {
                प्रत्यागम dequantize_row_q4_1_sycl;
            } अन्यथा {
                प्रत्यागम dequantize_खण्डम्_sycl<QK4_1, QR4_1, dequantize_q4_1, dst_t>;
            }
        पक्ष GGML_TYPE_Q5_0:
            प्रत्यागम dequantize_खण्डम्_sycl<QK5_0, QR5_0, dequantize_q5_0, dst_t>;
        पक्ष GGML_TYPE_Q5_1:
            प्रत्यागम dequantize_खण्डम्_sycl<QK5_1, QR5_1, dequantize_q5_1, dst_t>;
        पक्ष GGML_TYPE_Q8_0:
            यदि (is_reordered_src0(dst)) {
                प्रत्यागम dequantize_row_q8_0_sycl_reorder;
            } अन्यथा {
                प्रत्यागम dequantize_खण्डम्_sycl<QK8_0, QR8_0, dequantize_q8_0, dst_t>;
            }
        पक्ष GGML_TYPE_Q2_K:
            यदि (is_reordered_src0(dst)) {
                प्रत्यागम dequantize_row_q2_K_sycl_reorder;
            } अन्यथा {
                प्रत्यागम dequantize_row_q2_K_sycl;
            }
        पक्ष GGML_TYPE_Q3_K:
            यदि (is_reordered_src0(dst)) {
                प्रत्यागम dequantize_row_q3_K_sycl_reorder;
            } अन्यथा {
                प्रत्यागम dequantize_row_q3_K_sycl;
            }
        पक्ष GGML_TYPE_Q4_K:
            यदि (is_reordered_src0(dst)) {
                प्रत्यागम dequantize_row_q4_K_sycl_reorder;
            } अन्यथा {
                प्रत्यागम dequantize_row_q4_K_sycl;
            }
        पक्ष GGML_TYPE_Q5_K:
            यदि (is_reordered_src0(dst)) {
                प्रत्यागम dequantize_row_q5_K_sycl_reorder;
            } अन्यथा {
                प्रत्यागम dequantize_row_q5_K_sycl;
            }
        पक्ष GGML_TYPE_Q6_K:
            यदि (is_reordered_src0(dst)) {
                प्रत्यागम dequantize_row_q6_K_sycl_reorder;
            } अन्यथा {
                प्रत्यागम dequantize_row_q6_K_sycl;
            }
        पक्ष GGML_TYPE_IQ1_S:
            प्रत्यागम dequantize_row_iq1_s_sycl;
        पक्ष GGML_TYPE_IQ1_M:
            प्रत्यागम dequantize_row_iq1_m_sycl;
        पक्ष GGML_TYPE_IQ2_XXS:
            प्रत्यागम dequantize_row_iq2_xxs_sycl;
        पक्ष GGML_TYPE_IQ2_XS:
            प्रत्यागम dequantize_row_iq2_xs_sycl;
        पक्ष GGML_TYPE_IQ2_S:
            प्रत्यागम dequantize_row_iq2_s_sycl;
        पक्ष GGML_TYPE_IQ3_XXS:
            यदि (is_reordered_src0(dst)) {
                प्रत्यागम dequantize_row_iq3_xxs_sycl_reorder;
            } अन्यथा {
                प्रत्यागम dequantize_row_iq3_xxs_sycl;
            }
        पक्ष GGML_TYPE_IQ3_S:
            यदि (is_reordered_src0(dst)) {
                प्रत्यागम dequantize_row_iq3_s_sycl_reorder;
            } अन्यथा {
                प्रत्यागम dequantize_row_iq3_s_sycl;
            }
        पक्ष GGML_TYPE_IQ4_XS:
            प्रत्यागम dequantize_row_iq4_xs_sycl;
        पक्ष GGML_TYPE_IQ4_NL:
            प्रत्यागम dequantize_row_iq4_nl_sycl;
        पक्ष GGML_TYPE_MXFP4:
            यदि (is_reordered_src0(dst)) {
                प्रत्यागम dequantize_row_mxfp4_sycl_reorder;
            }
            प्रत्यागम dequantize_row_mxfp4_sycl;
        पक्ष GGML_TYPE_NVFP4:
            प्रत्यागम dequantize_row_nvfp4_sycl;
        पक्ष GGML_TYPE_F32:
            यदि constexpr (std::is_same_v<dst_t, sycl::half>) {
                प्रत्यागम convert_unary_sycl<भिन्न>;
            }
            भङ्ग;
        पक्ष GGML_TYPE_F16:
            यदि constexpr (std::is_same_v<dst_t, भिन्न>) {
                प्रत्यागम convert_unary_sycl<sycl::half>;
            }
            भङ्ग;
#यदि_परिभाषितम् GGML_SYCL_HAS_BF16
        पक्ष GGML_TYPE_BF16:
            प्रत्यागम convert_unary_sycl<sycl::ext::oneapi::bfloat16>;
#यदि_अन्तः
        स्वभाव:
            भङ्ग;
    }
    GGML_ABORT("fatal error: unsupport data type=%s\n", ggml_type_name(type));
}

गन्तव्यम्_fp16_sycl_t ggml_get_गन्तव्यम्_fp16_sycl(ggml_type type, ggml_tensor * dst) {
    प्रत्यागम ggml_get_dequant_sycl<sycl::half>(type, dst);
}

गन्तव्यम्_fp32_sycl_t ggml_get_गन्तव्यम्_fp32_sycl(ggml_type type, ggml_tensor *dst) {
    प्रत्यागम ggml_get_dequant_sycl<भिन्न>(type, dst);
}


#यदि_परिभाषितम् GGML_SYCL_HAS_BF16
गन्तव्यम्_bf16_sycl_t ggml_get_गन्तव्यम्_bf16_sycl(ggml_type type, ggml_tensor * /*dst*/) {
    विकल्प (type) {
        पक्ष GGML_TYPE_F32:
            प्रत्यागम convert_unary_sycl<भिन्न>;
        पक्ष GGML_TYPE_F16:
            प्रत्यागम convert_unary_sycl<sycl::half>;
        पक्ष GGML_TYPE_BF16:
            प्रत्यागम convert_unary_sycl<sycl::ext::oneapi::bfloat16>;
        स्वभाव:
            GGML_ABORT("fatal error: unsupport data type=%s\n", ggml_type_name(type));
    }
}
#यदि_अन्तः

गन्तव्यम्_fp16_nc_sycl_t ggml_get_गन्तव्यम्_fp16_nc_sycl(ggml_type type) {
    विकल्प (type) {
        पक्ष GGML_TYPE_F32:
            प्रत्यागम convert_unary_nc_sycl<भिन्न>;
#यदि_परिभाषितम् GGML_SYCL_HAS_BF16
        पक्ष GGML_TYPE_BF16:
            प्रत्यागम convert_unary_nc_sycl<sycl::ext::oneapi::bfloat16>;
#यदि_अन्तः
        पक्ष GGML_TYPE_Q1_0:
            प्रत्यागम dequantize_खण्डम्_nc_sycl<QK1_0, QR1_0, dequantize_q1_0>;
        पक्ष GGML_TYPE_Q4_0:
            प्रत्यागम dequantize_खण्डम्_nc_sycl<QK4_0, QR4_0, dequantize_q4_0>;
        पक्ष GGML_TYPE_Q4_1:
            प्रत्यागम dequantize_खण्डम्_nc_sycl<QK4_1, QR4_1, dequantize_q4_1>;
        पक्ष GGML_TYPE_Q5_0:
            प्रत्यागम dequantize_खण्डम्_nc_sycl<QK5_0, QR5_0, dequantize_q5_0>;
        पक्ष GGML_TYPE_Q5_1:
            प्रत्यागम dequantize_खण्डम्_nc_sycl<QK5_1, QR5_1, dequantize_q5_1>;
        पक्ष GGML_TYPE_Q8_0:
            प्रत्यागम dequantize_खण्डम्_nc_sycl<QK8_0, QR8_0, dequantize_q8_0>;
        स्वभाव:
            प्रत्यागम शून्यसूचकः;
    }
}
