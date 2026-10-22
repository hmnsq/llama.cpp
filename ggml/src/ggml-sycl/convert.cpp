#अन्तर्भाव "convert.hpp"
#अन्तर्भाव "dequantize.hpp"
#अन्तर्भाव "presets.hpp"

ढाँचा <पूर्णांक qk, पूर्णांक qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
स्थिर शून्य dequantize_block(स्थिरांक शून्य * __restrict__ vx, dst_t * __restrict__ y, स्थिरांक int64_t k,
                             स्थिरांक sycl::nd_item<3> &item_ct1) {
    स्थिरांक int64_t i = 2 * (item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                       item_ct1.get_local_id(2));

    जर (i >= k) {
        परत;
    }

    स्थिरांक int64_t ib = i/qk; // block index
    स्थिरांक int64_t iqs = (i%qk)/qr; // quant index
    स्थिरांक int64_t iybs = i - i%qk; // y block start index
    स्थिरांक int64_t y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    dfloat2 v;
    dequantize_kernel(vx, ib, iqs, v);

    y[iybs + iqs + 0] = v.x();
    y[iybs + iqs + y_offset] = v.y();
}

ढाँचा <पूर्णांक qk, पूर्णांक qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
स्थिर शून्य dequantize_block_sycl(स्थिरांक शून्य *__restrict__ vx,
                                  dst_t *__restrict__ y, स्थिरांक int64_t k,
                                  dpct::queue_ptr stream) {
    स्थिरांक int64_t अंक_blocks = (k + 2*SYCL_DEQUANTIZE_BLOCK_SIZE - 1) / (2*SYCL_DEQUANTIZE_BLOCK_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, अंक_blocks) *
                    sycl::range<3>(1, 1, SYCL_DEQUANTIZE_BLOCK_SIZE),
                sycl::range<3>(1, 1, SYCL_DEQUANTIZE_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) {
                dequantize_block<qk, qr, dequantize_kernel>(vx, y, k, item_ct1);
            });
    }
}

ढाँचा <typename dst_t>
स्थिर शून्य dequantize_row_q2_K_sycl(स्थिरांक शून्य *vx, dst_t *y, स्थिरांक int64_t k,
                                     dpct::queue_ptr stream) {
    स्थिरांक int64_t nb = k / QK_K;
#जर QK_K == 256
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 64),
                                               sycl::range<3>(1, 1, 64)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q2_K(vx, y, item_ct1);
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
                                 dequantize_block_q2_K(vx, y, item_ct1);
                             });
    }

#endif
}

ढाँचा <typename dst_t>
स्थिर शून्य dequantize_row_q2_K_sycl_reorder(स्थिरांक शून्य *vx, dst_t *y, स्थिरांक int64_t k,
                                             dpct::queue_ptr stream) {
    स्थिरांक int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
        [=](sycl::nd_item<3> item_ct1) {
            dequantize_block_q2_K_reorder(vx, y + item_ct1.get_group(2) * QK_K, item_ct1.get_group(2), item_ct1, nb);
        });
}

ढाँचा <typename dst_t>
स्थिर शून्य dequantize_row_q3_K_sycl(स्थिरांक शून्य *vx, dst_t *y, स्थिरांक int64_t k,
                                     dpct::queue_ptr stream) {
    स्थिरांक int64_t nb = k / QK_K;
#जर QK_K == 256
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 64),
                                               sycl::range<3>(1, 1, 64)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q3_K(vx, y, item_ct1);
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
                                 dequantize_block_q3_K(vx, y, item_ct1);
                             });
    }
#endif
}

ढाँचा <typename dst_t>
स्थिर शून्य dequantize_row_q3_K_sycl_reorder(स्थिरांक शून्य *vx, dst_t *y, स्थिरांक int64_t k,
                                             dpct::queue_ptr stream) {
    स्थिरांक int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
        [=](sycl::nd_item<3> item_ct1) {
            dequantize_block_q3_K_reorder(vx, y + item_ct1.get_group(2) * QK_K, item_ct1.get_group(2), item_ct1, nb);
        });
}

ढाँचा <typename dst_t>
स्थिर शून्य dequantize_row_q4_0_sycl(स्थिरांक शून्य *vx, dst_t *y, स्थिरांक int64_t k,
                                     dpct::queue_ptr stream) {
    स्थिरांक int64_t nb32 = k / 32;
    स्थिरांक int64_t nb = (k + 255) / 256;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q4_0(vx, y, nb32, item_ct1);
                             });
    }
}

ढाँचा <typename dst_t>
स्थिर शून्य dequantize_row_q4_0_sycl_reorder(स्थिरांक शून्य *vx, dst_t *y, स्थिरांक int64_t k,
                                     dpct::queue_ptr stream) {

    dpct::has_capability_or_fail(stream->get_device(),
                                    {sycl::aspect::fp16});

    GGML_ASSERT(k % QK4_0 == 0);
    स्थिरांक int64_t n_groups = ceil_div(k / 2, SYCL_DEQUANTIZE_BLOCK_SIZE);
    // decrease global range when it exceeds the max int; the kernel loops to cover the rest
    स्थिरांक int64_t wg_size  = downsample_sycl_global_range(n_groups, SYCL_DEQUANTIZE_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, n_groups) *
        sycl::range<3>(1, 1, wg_size),
        sycl::range<3>(1, 1, wg_size)),
        [=](sycl::nd_item<3> item_ct1) {
            dequantize_block_q4_0_reorder(vx, y, k, item_ct1);
        });
}

ढाँचा <typename dst_t>
स्थिर शून्य dequantize_row_q8_0_sycl_reorder(स्थिरांक शून्य *vx, dst_t *y, स्थिरांक int64_t k,
                                     dpct::queue_ptr stream) {

    dpct::has_capability_or_fail(stream->get_device(),
                                    {sycl::aspect::fp16});

    // flat range<1>, two pairs (4 elements) per item: no nd_range/group math, one vectorized store.
    // k % QK8_0 == 0 implies k % 4 == 0, so the wide launch covers the row exactly (no tail).
    GGML_ASSERT(k % QK8_0 == 0);
    stream->parallel_for(sycl::range<1>((माप_प्रकार) (k / 4)),
        [=](sycl::id<1> idx) {
            dequantize_block_q8_0_reorder_wide(vx, y, k, idx);
        });
}

ढाँचा <typename dst_t>
स्थिर शून्य dequantize_row_q4_1_sycl(स्थिरांक शून्य *vx, dst_t *y, स्थिरांक int64_t k,
                                     dpct::queue_ptr stream) {
    स्थिरांक int64_t nb32 = k / 32;
    स्थिरांक int64_t nb = (k + 255) / 256;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q4_1(vx, y, nb32, item_ct1);
                             });
    }
}


ढाँचा <typename dst_t>
स्थिर शून्य dequantize_row_q4_K_sycl(स्थिरांक शून्य *vx, dst_t *y, स्थिरांक int64_t k,
                                     dpct::queue_ptr stream) {
    स्थिरांक int64_t nb = k / QK_K;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(sycl::nd_range<1>(sycl::range<1>((माप_प्रकार) nb * SYCL_DEQK_WI_PER_BLOCK),
                                           sycl::range<1>((माप_प्रकार) SYCL_DEQK_WI_PER_BLOCK)),
                         [=](sycl::nd_item<1> it) { dequantize_block_q4_K_wide(vx, y, nb, it); });
}

ढाँचा <typename dst_t>
स्थिर शून्य dequantize_row_q4_K_sycl_reorder(स्थिरांक शून्य * vx, dst_t * y, स्थिरांक int64_t k, dpct::queue_ptr stream) {
    स्थिरांक int64_t nb = k / QK_K;
    स्थिरांक माप_प्रकार  local_size  = 32;
    स्थिरांक माप_प्रकार  global_size = nb * local_size;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint8_t, 1> scale_local_acc(sycl::range<1>(12), cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(local_size)),
                         [=](sycl::nd_item<1> item_ct1) {
                             dequantize_block_q4_K_reorder(vx, y + item_ct1.get_group(0) * QK_K, item_ct1.get_group(0), get_pointer(scale_local_acc), item_ct1, nb);
                         });
    });
}

ढाँचा <typename dst_t>
स्थिर शून्य dequantize_row_q5_K_sycl(स्थिरांक शून्य *vx, dst_t *y, स्थिरांक int64_t k,
                                     dpct::queue_ptr stream) {
    स्थिरांक int64_t nb = k / QK_K;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(sycl::nd_range<1>(sycl::range<1>((माप_प्रकार) nb * SYCL_DEQK_WI_PER_BLOCK),
                                           sycl::range<1>((माप_प्रकार) SYCL_DEQK_WI_PER_BLOCK)),
                         [=](sycl::nd_item<1> it) { dequantize_block_q5_K_wide(vx, y, nb, it); });
}

ढाँचा <typename dst_t>
स्थिर शून्य dequantize_row_q5_K_sycl_reorder(स्थिरांक शून्य * vx, dst_t * y, स्थिरांक int64_t k, dpct::queue_ptr stream) {
    स्थिरांक int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint8_t, 1> scale_local_acc(sycl::range<1>(K_SCALE_SIZE), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
            [=](sycl::nd_item<3> item_ct1) {
                dequantize_block_q5_K_reorder(vx, y + item_ct1.get_group(2) * QK_K, item_ct1.get_group(2), get_pointer(scale_local_acc), item_ct1, nb);
            });
    });
}

ढाँचा <typename dst_t>
स्थिर शून्य dequantize_row_q6_K_sycl(स्थिरांक शून्य *vx, dst_t *y, स्थिरांक int64_t k,
                                     dpct::queue_ptr stream) {
    स्थिरांक int64_t nb = k / QK_K;
#जर QK_K == 256
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 64),
                                               sycl::range<3>(1, 1, 64)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q6_K(vx, y, item_ct1);
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
                                 dequantize_block_q6_K(vx, y, item_ct1);
                             });
    }

#endif
}

ढाँचा <typename dst_t>
स्थिर शून्य dequantize_row_q6_K_sycl_reorder(स्थिरांक शून्य * vx, dst_t * y, स्थिरांक int64_t k, dpct::queue_ptr stream) {
    स्थिरांक int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
        [=](sycl::nd_item<3> item_ct1) { dequantize_block_q6_K_reorder(vx, y + item_ct1.get_group(2) * QK_K, item_ct1.get_group(2), item_ct1, nb); });
}

ढाँचा <typename dst_t>
स्थिर शून्य dequantize_row_iq1_s_sycl(स्थिरांक शून्य *vx, dst_t *y, स्थिरांक int64_t k,
                                        dpct::queue_ptr stream) {
    स्थिरांक int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq1_s(
                                     vx, y, item_ct1, iq1s_grid_gpu
                                     );
                             });
        });
    }
}

ढाँचा <typename dst_t>
स्थिर शून्य dequantize_row_iq1_m_sycl(स्थिरांक शून्य *vx, dst_t *y, स्थिरांक int64_t k,
                                        dpct::queue_ptr stream) {
    स्थिरांक int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq1_m(
                                     vx, y, item_ct1, iq1s_grid_gpu
                                     );
                             });
        });
    }
}

ढाँचा <typename dst_t>
स्थिर शून्य dequantize_row_iq2_xxs_sycl(स्थिरांक शून्य *vx, dst_t *y, स्थिरांक int64_t k,
                                        dpct::queue_ptr stream) {
    स्थिरांक int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq2_xxs(
                                     vx, y, item_ct1, iq2xxs_grid,
                                     ksigns_iq2xs, kmask_iq2xs);
                             });
        });
    }
}

ढाँचा <typename dst_t>
स्थिर शून्य dequantize_row_iq2_xs_sycl(स्थिरांक शून्य *vx, dst_t *y, स्थिरांक int64_t k,
                                       dpct::queue_ptr stream) {
    स्थिरांक int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq2_xs(
                                     vx, y, item_ct1, iq2xs_grid,
                                     ksigns_iq2xs, kmask_iq2xs);
                             });
        });
    }
}

ढाँचा <typename dst_t>
स्थिर शून्य dequantize_row_iq2_s_sycl(स्थिरांक शून्य *vx, dst_t *y, स्थिरांक int64_t k,
                                      dpct::queue_ptr stream) {
    स्थिरांक int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq2_s(vx, y, item_ct1);
                             });
        });
    }
}


ढाँचा <typename dst_t>
स्थिर शून्य dequantize_row_iq3_xxs_sycl(स्थिरांक शून्य *vx, dst_t *y, स्थिरांक int64_t k,
                                        dpct::queue_ptr stream) {
    स्थिरांक int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq3_xxs(
                                     vx, y, item_ct1, iq3xxs_grid,
                                     ksigns_iq2xs, kmask_iq2xs);
                             });
        });
    }
}

ढाँचा <typename dst_t>
स्थिर शून्य dequantize_row_iq3_xxs_sycl_reorder(स्थिरांक शून्य * vx, dst_t * y, स्थिरांक int64_t k,
                                                dpct::queue_ptr stream) {
    स्थिरांक int64_t nb = k / QK_K;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_item<3> item_ct1) {
            dequantize_block_iq3_xxs_reorder(
                vx, y + item_ct1.get_group(2) * QK_K, item_ct1.get_group(2), item_ct1, nb,
                iq3xxs_grid, ksigns_iq2xs, kmask_iq2xs);
        });
}

ढाँचा <typename dst_t>
स्थिर शून्य dequantize_row_iq3_s_sycl(स्थिरांक शून्य *vx, dst_t *y, स्थिरांक int64_t k,
                                        dpct::queue_ptr stream) {
    स्थिरांक int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq3_s(
                                     vx, y, item_ct1, kmask_iq2xs, iq3s_grid);
                             });
        });
    }
}

ढाँचा <typename dst_t>
स्थिर शून्य dequantize_row_iq3_s_sycl_reorder(स्थिरांक शून्य * vx, dst_t * y, स्थिरांक int64_t k,
                                              dpct::queue_ptr stream) {
    स्थिरांक int64_t nb = k / QK_K;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_item<3> item_ct1) {
            dequantize_block_iq3_s_reorder(vx, y + item_ct1.get_group(2) * QK_K, item_ct1.get_group(2),
                                           item_ct1, nb, kmask_iq2xs, iq3s_grid);
        });
}

ढाँचा <typename dst_t>
स्थिर शून्य dequantize_row_iq4_xs_sycl(स्थिरांक शून्य *vx, dst_t *y, स्थिरांक int64_t k,
                                       dpct::queue_ptr stream) {
    स्थिरांक int64_t nb = (k + QK_K - 1) / QK_K;
#जर QK_K == 64
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
                            dequantize_block_iq4_xs(vx, y, item_ct1);
                      });
            });
      }
#endif
}

ढाँचा <typename dst_t>
स्थिर शून्य dequantize_row_iq4_nl_sycl(स्थिरांक शून्य *vx, dst_t *y, स्थिरांक int64_t k,
                                       dpct::queue_ptr stream) {
    स्थिरांक int64_t nb = (k + QK_K - 1) / QK_K;
      {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                  cgh.parallel_for(
                      sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                            sycl::range<3>(1, 1, 32),
                                        sycl::range<3>(1, 1, 32)),
                      [=](sycl::nd_item<3> item_ct1) {
                            dequantize_block_iq4_nl(vx, y, item_ct1);
                      });
            });
      }
}

ढाँचा <typename dst_t>
स्थिर शून्य dequantize_row_mxfp4_sycl(स्थिरांक शून्य * vx, dst_t * y, स्थिरांक int64_t k, dpct::queue_ptr stream) {
    स्थिरांक पूर्णांक nb = (k + QK_K - 1) / QK_K;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_item<3> item_ct1) {
            dequantize_block_mxfp4(vx, y, item_ct1);
        });
}

ढाँचा <typename dst_t>
स्थिर शून्य dequantize_row_mxfp4_sycl_reorder(स्थिरांक शून्य * vx, dst_t * y, स्थिरांक int64_t k, dpct::queue_ptr stream) {
    GGML_ASSERT(k % QK_MXFP4 == 0);
    स्थिरांक पूर्णांक n_warp = (k / QK_MXFP4 + WARP_SIZE - 1) / WARP_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, n_warp) * sycl::range<3>(1, 1, WARP_SIZE),
                          sycl::range<3>(1, 1, WARP_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            dequantize_block_mxfp4_reorder(vx, y, k, item_ct1);
        });
}

ढाँचा <typename dst_t>
स्थिर शून्य dequantize_row_nvfp4_sycl(स्थिरांक शून्य * vx, dst_t * y, स्थिरांक int64_t k, dpct::queue_ptr stream) {
    GGML_ASSERT(k % QK_NVFP4 == 0);
    स्थिरांक पूर्णांक nb = k / QK_NVFP4;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_item<3> /*item_ct1*/) {
            dequantize_block_nvfp4(vx, y, k);
        });
}


ढाँचा <पूर्णांक qk, पूर्णांक qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
स्थिर शून्य dequantize_block_nc(स्थिरांक शून्य * __restrict__ vx, dst_t * __restrict__ y,
        स्थिरांक int64_t ne00, स्थिरांक int64_t ne01, स्थिरांक int64_t ne02,
        स्थिरांक int64_t s01, स्थिरांक int64_t s02, स्थिरांक int64_t s03) {
    स्वतः          item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    स्थिरांक int64_t i00 = 2 * (int64_t(item_ct1.get_local_range(2)) * item_ct1.get_group(2) + item_ct1.get_local_id(2));

    जर (i00 >= ne00) {
        परत;
    }

    स्थिरांक int64_t i01 = item_ct1.get_group(1);
    स्थिरांक int64_t i02 = item_ct1.get_group(0) % ne02;
    स्थिरांक int64_t i03 = item_ct1.get_group(0) / ne02;

    स्थिरांक int64_t ibx0 = i03*s03 + i02*s02 + i01*s01;

    स्थिरांक int64_t ib = ibx0 + i00/qk; // block index
    स्थिरांक int64_t iqs = (i00%qk)/qr; // quant index
    स्थिरांक int64_t iybs = i00 - i00%qk; // y block start index
    स्थिरांक int64_t y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    #ifdef GGML_SYCL_F16
        sycl::half2 v;
    #अन्यथा
        sycl::float2 v;
    #endif

    dequantize_kernel(vx, ib, iqs, v);

    स्थिरांक int64_t iy0 = ((i03*ne02 + i02)*ne01 + i01)*ne00 + iybs + iqs;
    y[iy0 + 0]        = ggml_sycl_cast<dst_t>(v.x());
    y[iy0 + y_offset] = ggml_sycl_cast<dst_t>(v.y());
}


ढाँचा <पूर्णांक qk, पूर्णांक qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
स्थिर शून्य dequantize_block_nc_sycl(स्थिरांक शून्य *    vx,
                                  dst_t *         y,
                                  स्थिरांक int64_t   ne00,
                                  स्थिरांक int64_t   ne01,
                                  स्थिरांक int64_t   ne02,
                                  स्थिरांक int64_t   ne03,
                                  स्थिरांक int64_t   s01,
                                  स्थिरांक int64_t   s02,
                                  स्थिरांक int64_t   s03,
                                  dpct::queue_ptr stream) {
    स्थिरांक dpct::dim3 अंक_blocks((ne00 + 2 * SYCL_DEQUANTIZE_BLOCK_SIZE - 1) / (2 * SYCL_DEQUANTIZE_BLOCK_SIZE), ne01,
                                ne02 * ne03);
    stream->parallel_for(sycl::nd_range<3>(अंक_blocks * sycl::range<3>(1, 1, SYCL_DEQUANTIZE_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_DEQUANTIZE_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) {
                             GGML_UNUSED(item_ct1);
                             dequantize_block_nc<qk, qr, dequantize_kernel>(vx, y, ne00, ne01, ne02, s01, s02, s03);
                         });
}
ढाँचा <typename src_t, typename dst_t>
स्थिर शून्य convert_unary_nc(स्थिरांक शून्य * __restrict__ vx, dst_t * __restrict__ y, स्थिरांक int64_t ne00, स्थिरांक int64_t ne01,
                          स्थिरांक int64_t ne02, स्थिरांक int64_t s01, स्थिरांक int64_t s02, स्थिरांक int64_t s03,
                          स्थिरांक sycl::nd_item<3> & item_ct1) {

    स्थिरांक int64_t work_group_size = item_ct1.get_local_range(2);
    स्थिरांक int64_t global_id       = item_ct1.get_local_id(2) + work_group_size * item_ct1.get_group(2);

    स्थिरांक int64_t i01 = item_ct1.get_group(1);
    स्थिरांक int64_t i02 = item_ct1.get_group(0) % ne02;
    स्थिरांक int64_t i03 = item_ct1.get_group(0) / ne02;

    // make each work-item deal with more elements since sycl global range can not exceed max int
    स्थिरांक src_t * x = static_cast<स्थिरांक src_t *>(vx);
    स्थिरांक int64_t ix = i03 * s03 + i02 * s02 + i01 * s01;
    स्थिरांक int64_t iy = ((i03 * ne02 + i02) * ne01 + i01) * ne00;

#pragma unroll
    साठी (int64_t i00 = global_id; i00 < ne00; i00 += work_group_size * item_ct1.get_group_range(2)) {
        y[iy + i00] = static_cast<dst_t>(x[ix + i00]);
    }
}

ढाँचा <typename src_t, typename dst_t>
स्थिर शून्य convert_unary_nc_sycl(स्थिरांक शून्य * __restrict__ vx, dst_t * __restrict__ y,
                                  स्थिरांक int64_t ne00, स्थिरांक int64_t ne01, स्थिरांक int64_t ne02, स्थिरांक int64_t ne03,
                                  स्थिरांक int64_t s01, स्थिरांक int64_t s02, स्थिरांक int64_t s03, dpct::queue_ptr queue) {
    dpct::has_capability_or_fail(queue->get_device(), { sycl::aspect::fp16 });

    sycl::range<3> global_size(ne02 * ne03, ne01, ceil_div(ne00, SYCL_DEQUANTIZE_BLOCK_SIZE));

    // decrease global range when it exceeds the max int
    // TODO: Downsample logic is separated from the kernel, a rewrite is desirable
    int64_t        downsized_workgroup = downsample_sycl_global_range(global_size[0], SYCL_DEQUANTIZE_BLOCK_SIZE);
    sycl::range<3> workgroup_size(1, 1, downsized_workgroup);

    queue->parallel_for(sycl::nd_range<3>(global_size * workgroup_size, workgroup_size), [=](sycl::nd_item<3> item_ct1) {
        convert_unary_nc<src_t>(vx, y, ne00, ne01, ne02, s01, s02, s03, item_ct1);
    });
}

ढाँचा <typename src_t, typename dst_t>
स्थिर शून्य convert_unary_sycl(स्थिरांक शून्य * vx, dst_t * y, स्थिरांक int64_t k, dpct::queue_ptr queue) {
    convert_unary_nc_sycl<src_t>(vx, y, k, 1, 1, 1, k, k, k, queue);
}


स्थिर तार्किक is_reordered_src0(स्थिरांक ggml_tensor * dst) {
    परत dst->src[0]->extra &&
           ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder;
}

ढाँचा <typename dst_t>
स्थिर to_t_sycl_t<dst_t> ggml_get_dequant_sycl(ggml_type type, ggml_tensor * dst) {
    निवड (type) {
        स्थिती GGML_TYPE_Q1_0:
            परत dequantize_block_sycl<QK1_0, QR1_0, dequantize_q1_0, dst_t>;
        स्थिती GGML_TYPE_Q2_0:
            परत dequantize_block_sycl<QK2_0, QR2_0, dequantize_q2_0, dst_t>;
        स्थिती GGML_TYPE_Q4_0:
            जर (is_reordered_src0(dst)) {
                परत dequantize_row_q4_0_sycl_reorder;
            } अन्यथा जर constexpr (std::is_same_v<dst_t, दशांश>) {
                परत dequantize_row_q4_0_sycl;
            } अन्यथा {
                परत dequantize_block_sycl<QK4_0, QR4_0, dequantize_q4_0, dst_t>;
            }
        स्थिती GGML_TYPE_Q4_1:
            जर constexpr (std::is_same_v<dst_t, दशांश>) {
                परत dequantize_row_q4_1_sycl;
            } अन्यथा {
                परत dequantize_block_sycl<QK4_1, QR4_1, dequantize_q4_1, dst_t>;
            }
        स्थिती GGML_TYPE_Q5_0:
            परत dequantize_block_sycl<QK5_0, QR5_0, dequantize_q5_0, dst_t>;
        स्थिती GGML_TYPE_Q5_1:
            परत dequantize_block_sycl<QK5_1, QR5_1, dequantize_q5_1, dst_t>;
        स्थिती GGML_TYPE_Q8_0:
            जर (is_reordered_src0(dst)) {
                परत dequantize_row_q8_0_sycl_reorder;
            } अन्यथा {
                परत dequantize_block_sycl<QK8_0, QR8_0, dequantize_q8_0, dst_t>;
            }
        स्थिती GGML_TYPE_Q2_K:
            जर (is_reordered_src0(dst)) {
                परत dequantize_row_q2_K_sycl_reorder;
            } अन्यथा {
                परत dequantize_row_q2_K_sycl;
            }
        स्थिती GGML_TYPE_Q3_K:
            जर (is_reordered_src0(dst)) {
                परत dequantize_row_q3_K_sycl_reorder;
            } अन्यथा {
                परत dequantize_row_q3_K_sycl;
            }
        स्थिती GGML_TYPE_Q4_K:
            जर (is_reordered_src0(dst)) {
                परत dequantize_row_q4_K_sycl_reorder;
            } अन्यथा {
                परत dequantize_row_q4_K_sycl;
            }
        स्थिती GGML_TYPE_Q5_K:
            जर (is_reordered_src0(dst)) {
                परत dequantize_row_q5_K_sycl_reorder;
            } अन्यथा {
                परत dequantize_row_q5_K_sycl;
            }
        स्थिती GGML_TYPE_Q6_K:
            जर (is_reordered_src0(dst)) {
                परत dequantize_row_q6_K_sycl_reorder;
            } अन्यथा {
                परत dequantize_row_q6_K_sycl;
            }
        स्थिती GGML_TYPE_IQ1_S:
            परत dequantize_row_iq1_s_sycl;
        स्थिती GGML_TYPE_IQ1_M:
            परत dequantize_row_iq1_m_sycl;
        स्थिती GGML_TYPE_IQ2_XXS:
            परत dequantize_row_iq2_xxs_sycl;
        स्थिती GGML_TYPE_IQ2_XS:
            परत dequantize_row_iq2_xs_sycl;
        स्थिती GGML_TYPE_IQ2_S:
            परत dequantize_row_iq2_s_sycl;
        स्थिती GGML_TYPE_IQ3_XXS:
            जर (is_reordered_src0(dst)) {
                परत dequantize_row_iq3_xxs_sycl_reorder;
            } अन्यथा {
                परत dequantize_row_iq3_xxs_sycl;
            }
        स्थिती GGML_TYPE_IQ3_S:
            जर (is_reordered_src0(dst)) {
                परत dequantize_row_iq3_s_sycl_reorder;
            } अन्यथा {
                परत dequantize_row_iq3_s_sycl;
            }
        स्थिती GGML_TYPE_IQ4_XS:
            परत dequantize_row_iq4_xs_sycl;
        स्थिती GGML_TYPE_IQ4_NL:
            परत dequantize_row_iq4_nl_sycl;
        स्थिती GGML_TYPE_MXFP4:
            जर (is_reordered_src0(dst)) {
                परत dequantize_row_mxfp4_sycl_reorder;
            }
            परत dequantize_row_mxfp4_sycl;
        स्थिती GGML_TYPE_NVFP4:
            परत dequantize_row_nvfp4_sycl;
        स्थिती GGML_TYPE_F32:
            जर constexpr (std::is_same_v<dst_t, sycl::half>) {
                परत convert_unary_sycl<दशांश>;
            }
            थांबा;
        स्थिती GGML_TYPE_F16:
            जर constexpr (std::is_same_v<dst_t, दशांश>) {
                परत convert_unary_sycl<sycl::half>;
            }
            थांबा;
#ifdef GGML_SYCL_HAS_BF16
        स्थिती GGML_TYPE_BF16:
            परत convert_unary_sycl<sycl::ext::oneapi::bfloat16>;
#endif
        पूर्वनिर्धारित:
            थांबा;
    }
    GGML_ABORT("fatal error: unsupport data type=%s\n", ggml_type_name(type));
}

to_fp16_sycl_t ggml_get_to_fp16_sycl(ggml_type type, ggml_tensor * dst) {
    परत ggml_get_dequant_sycl<sycl::half>(type, dst);
}

to_fp32_sycl_t ggml_get_to_fp32_sycl(ggml_type type, ggml_tensor *dst) {
    परत ggml_get_dequant_sycl<दशांश>(type, dst);
}


#ifdef GGML_SYCL_HAS_BF16
to_bf16_sycl_t ggml_get_to_bf16_sycl(ggml_type type, ggml_tensor * /*dst*/) {
    निवड (type) {
        स्थिती GGML_TYPE_F32:
            परत convert_unary_sycl<दशांश>;
        स्थिती GGML_TYPE_F16:
            परत convert_unary_sycl<sycl::half>;
        स्थिती GGML_TYPE_BF16:
            परत convert_unary_sycl<sycl::ext::oneapi::bfloat16>;
        पूर्वनिर्धारित:
            GGML_ABORT("fatal error: unsupport data type=%s\n", ggml_type_name(type));
    }
}
#endif

to_fp16_nc_sycl_t ggml_get_to_fp16_nc_sycl(ggml_type type) {
    निवड (type) {
        स्थिती GGML_TYPE_F32:
            परत convert_unary_nc_sycl<दशांश>;
#ifdef GGML_SYCL_HAS_BF16
        स्थिती GGML_TYPE_BF16:
            परत convert_unary_nc_sycl<sycl::ext::oneapi::bfloat16>;
#endif
        स्थिती GGML_TYPE_Q1_0:
            परत dequantize_block_nc_sycl<QK1_0, QR1_0, dequantize_q1_0>;
        स्थिती GGML_TYPE_Q4_0:
            परत dequantize_block_nc_sycl<QK4_0, QR4_0, dequantize_q4_0>;
        स्थिती GGML_TYPE_Q4_1:
            परत dequantize_block_nc_sycl<QK4_1, QR4_1, dequantize_q4_1>;
        स्थिती GGML_TYPE_Q5_0:
            परत dequantize_block_nc_sycl<QK5_0, QR5_0, dequantize_q5_0>;
        स्थिती GGML_TYPE_Q5_1:
            परत dequantize_block_nc_sycl<QK5_1, QR5_1, dequantize_q5_1>;
        स्थिती GGML_TYPE_Q8_0:
            परत dequantize_block_nc_sycl<QK8_0, QR8_0, dequantize_q8_0>;
        पूर्वनिर्धारित:
            परत nullptr;
    }
}
