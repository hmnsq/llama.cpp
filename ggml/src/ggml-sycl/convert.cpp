#ଅନ୍ତର୍ଭାଵ "convert.hpp"
#ଅନ୍ତର୍ଭାଵ "dequantize.hpp"
#ଅନ୍ତର୍ଭାଵ "presets.hpp"

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା qk, ପୂର୍ଣ୍ଣସଂଖ୍ୟା qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
ସ୍ଥିର ଶୂନ୍ୟ dequantize_block(ଅଚଳ ଶୂନ୍ୟ * __restrict__ vx, dst_t * __restrict__ y, ଅଚଳ int64_t k,
                             ଅଚଳ sycl::nd_item<3> &item_ct1) {
    ଅଚଳ int64_t i = 2 * (item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                       item_ct1.get_local_id(2));

    ଯଦି (i >= k) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ int64_t ib = i/qk; // block index
    ଅଚଳ int64_t iqs = (i%qk)/qr; // quant index
    ଅଚଳ int64_t iybs = i - i%qk; // y block start index
    ଅଚଳ int64_t y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    dfloat2 v;
    dequantize_kernel(vx, ib, iqs, v);

    y[iybs + iqs + 0] = v.x();
    y[iybs + iqs + y_offset] = v.y();
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା qk, ପୂର୍ଣ୍ଣସଂଖ୍ୟା qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
ସ୍ଥିର ଶୂନ୍ୟ dequantize_block_sycl(ଅଚଳ ଶୂନ୍ୟ *__restrict__ vx,
                                  dst_t *__restrict__ y, ଅଚଳ int64_t k,
                                  dpct::queue_ptr stream) {
    ଅଚଳ int64_t ଅଙ୍କ_blocks = (k + 2*SYCL_DEQUANTIZE_BLOCK_SIZE - 1) / (2*SYCL_DEQUANTIZE_BLOCK_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, ଅଙ୍କ_blocks) *
                    sycl::range<3>(1, 1, SYCL_DEQUANTIZE_BLOCK_SIZE),
                sycl::range<3>(1, 1, SYCL_DEQUANTIZE_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) {
                dequantize_block<qk, qr, dequantize_kernel>(vx, y, k, item_ct1);
            });
    }
}

template <typename dst_t>
ସ୍ଥିର ଶୂନ୍ୟ dequantize_row_q2_K_sycl(ଅଚଳ ଶୂନ୍ୟ *vx, dst_t *y, ଅଚଳ int64_t k,
                                     dpct::queue_ptr stream) {
    ଅଚଳ int64_t nb = k / QK_K;
#ଯଦି QK_K == 256
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
#ଅନ୍ୟଥା
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

template <typename dst_t>
ସ୍ଥିର ଶୂନ୍ୟ dequantize_row_q2_K_sycl_reorder(ଅଚଳ ଶୂନ୍ୟ *vx, dst_t *y, ଅଚଳ int64_t k,
                                             dpct::queue_ptr stream) {
    ଅଚଳ int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
        [=](sycl::nd_item<3> item_ct1) {
            dequantize_block_q2_K_reorder(vx, y + item_ct1.get_group(2) * QK_K, item_ct1.get_group(2), item_ct1, nb);
        });
}

template <typename dst_t>
ସ୍ଥିର ଶୂନ୍ୟ dequantize_row_q3_K_sycl(ଅଚଳ ଶୂନ୍ୟ *vx, dst_t *y, ଅଚଳ int64_t k,
                                     dpct::queue_ptr stream) {
    ଅଚଳ int64_t nb = k / QK_K;
#ଯଦି QK_K == 256
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
#ଅନ୍ୟଥା
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

template <typename dst_t>
ସ୍ଥିର ଶୂନ୍ୟ dequantize_row_q3_K_sycl_reorder(ଅଚଳ ଶୂନ୍ୟ *vx, dst_t *y, ଅଚଳ int64_t k,
                                             dpct::queue_ptr stream) {
    ଅଚଳ int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
        [=](sycl::nd_item<3> item_ct1) {
            dequantize_block_q3_K_reorder(vx, y + item_ct1.get_group(2) * QK_K, item_ct1.get_group(2), item_ct1, nb);
        });
}

template <typename dst_t>
ସ୍ଥିର ଶୂନ୍ୟ dequantize_row_q4_0_sycl(ଅଚଳ ଶୂନ୍ୟ *vx, dst_t *y, ଅଚଳ int64_t k,
                                     dpct::queue_ptr stream) {
    ଅଚଳ int64_t nb32 = k / 32;
    ଅଚଳ int64_t nb = (k + 255) / 256;
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

template <typename dst_t>
ସ୍ଥିର ଶୂନ୍ୟ dequantize_row_q4_0_sycl_reorder(ଅଚଳ ଶୂନ୍ୟ *vx, dst_t *y, ଅଚଳ int64_t k,
                                     dpct::queue_ptr stream) {

    dpct::has_capability_or_fail(stream->get_device(),
                                    {sycl::aspect::fp16});

    GGML_ASSERT(k % QK4_0 == 0);
    ଅଚଳ int64_t n_groups = ceil_div(k / 2, SYCL_DEQUANTIZE_BLOCK_SIZE);
    // decrease global range when it exceeds the max int; the kernel loops to cover the rest
    ଅଚଳ int64_t wg_size  = downsample_sycl_global_range(n_groups, SYCL_DEQUANTIZE_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, n_groups) *
        sycl::range<3>(1, 1, wg_size),
        sycl::range<3>(1, 1, wg_size)),
        [=](sycl::nd_item<3> item_ct1) {
            dequantize_block_q4_0_reorder(vx, y, k, item_ct1);
        });
}

template <typename dst_t>
ସ୍ଥିର ଶୂନ୍ୟ dequantize_row_q8_0_sycl_reorder(ଅଚଳ ଶୂନ୍ୟ *vx, dst_t *y, ଅଚଳ int64_t k,
                                     dpct::queue_ptr stream) {

    dpct::has_capability_or_fail(stream->get_device(),
                                    {sycl::aspect::fp16});

    // flat range<1>, two pairs (4 elements) per item: no nd_range/group math, one vectorized store.
    // k % QK8_0 == 0 implies k % 4 == 0, so the wide launch covers the row exactly (no tail).
    GGML_ASSERT(k % QK8_0 == 0);
    stream->parallel_for(sycl::range<1>((size_t) (k / 4)),
        [=](sycl::id<1> idx) {
            dequantize_block_q8_0_reorder_wide(vx, y, k, idx);
        });
}

template <typename dst_t>
ସ୍ଥିର ଶୂନ୍ୟ dequantize_row_q4_1_sycl(ଅଚଳ ଶୂନ୍ୟ *vx, dst_t *y, ଅଚଳ int64_t k,
                                     dpct::queue_ptr stream) {
    ଅଚଳ int64_t nb32 = k / 32;
    ଅଚଳ int64_t nb = (k + 255) / 256;
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


template <typename dst_t>
ସ୍ଥିର ଶୂନ୍ୟ dequantize_row_q4_K_sycl(ଅଚଳ ଶୂନ୍ୟ *vx, dst_t *y, ଅଚଳ int64_t k,
                                     dpct::queue_ptr stream) {
    ଅଚଳ int64_t nb = k / QK_K;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) nb * SYCL_DEQK_WI_PER_BLOCK),
                                           sycl::range<1>((size_t) SYCL_DEQK_WI_PER_BLOCK)),
                         [=](sycl::nd_item<1> it) { dequantize_block_q4_K_wide(vx, y, nb, it); });
}

template <typename dst_t>
ସ୍ଥିର ଶୂନ୍ୟ dequantize_row_q4_K_sycl_reorder(ଅଚଳ ଶୂନ୍ୟ * vx, dst_t * y, ଅଚଳ int64_t k, dpct::queue_ptr stream) {
    ଅଚଳ int64_t nb = k / QK_K;
    ଅଚଳ size_t  local_size  = 32;
    ଅଚଳ size_t  global_size = nb * local_size;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint8_t, 1> scale_local_acc(sycl::range<1>(12), cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(local_size)),
                         [=](sycl::nd_item<1> item_ct1) {
                             dequantize_block_q4_K_reorder(vx, y + item_ct1.get_group(0) * QK_K, item_ct1.get_group(0), get_pointer(scale_local_acc), item_ct1, nb);
                         });
    });
}

template <typename dst_t>
ସ୍ଥିର ଶୂନ୍ୟ dequantize_row_q5_K_sycl(ଅଚଳ ଶୂନ୍ୟ *vx, dst_t *y, ଅଚଳ int64_t k,
                                     dpct::queue_ptr stream) {
    ଅଚଳ int64_t nb = k / QK_K;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) nb * SYCL_DEQK_WI_PER_BLOCK),
                                           sycl::range<1>((size_t) SYCL_DEQK_WI_PER_BLOCK)),
                         [=](sycl::nd_item<1> it) { dequantize_block_q5_K_wide(vx, y, nb, it); });
}

template <typename dst_t>
ସ୍ଥିର ଶୂନ୍ୟ dequantize_row_q5_K_sycl_reorder(ଅଚଳ ଶୂନ୍ୟ * vx, dst_t * y, ଅଚଳ int64_t k, dpct::queue_ptr stream) {
    ଅଚଳ int64_t nb = k / QK_K;

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

template <typename dst_t>
ସ୍ଥିର ଶୂନ୍ୟ dequantize_row_q6_K_sycl(ଅଚଳ ଶୂନ୍ୟ *vx, dst_t *y, ଅଚଳ int64_t k,
                                     dpct::queue_ptr stream) {
    ଅଚଳ int64_t nb = k / QK_K;
#ଯଦି QK_K == 256
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
#ଅନ୍ୟଥା
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

template <typename dst_t>
ସ୍ଥିର ଶୂନ୍ୟ dequantize_row_q6_K_sycl_reorder(ଅଚଳ ଶୂନ୍ୟ * vx, dst_t * y, ଅଚଳ int64_t k, dpct::queue_ptr stream) {
    ଅଚଳ int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
        [=](sycl::nd_item<3> item_ct1) { dequantize_block_q6_K_reorder(vx, y + item_ct1.get_group(2) * QK_K, item_ct1.get_group(2), item_ct1, nb); });
}

template <typename dst_t>
ସ୍ଥିର ଶୂନ୍ୟ dequantize_row_iq1_s_sycl(ଅଚଳ ଶୂନ୍ୟ *vx, dst_t *y, ଅଚଳ int64_t k,
                                        dpct::queue_ptr stream) {
    ଅଚଳ int64_t nb = k / QK_K;
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

template <typename dst_t>
ସ୍ଥିର ଶୂନ୍ୟ dequantize_row_iq1_m_sycl(ଅଚଳ ଶୂନ୍ୟ *vx, dst_t *y, ଅଚଳ int64_t k,
                                        dpct::queue_ptr stream) {
    ଅଚଳ int64_t nb = k / QK_K;
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

template <typename dst_t>
ସ୍ଥିର ଶୂନ୍ୟ dequantize_row_iq2_xxs_sycl(ଅଚଳ ଶୂନ୍ୟ *vx, dst_t *y, ଅଚଳ int64_t k,
                                        dpct::queue_ptr stream) {
    ଅଚଳ int64_t nb = k / QK_K;
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

template <typename dst_t>
ସ୍ଥିର ଶୂନ୍ୟ dequantize_row_iq2_xs_sycl(ଅଚଳ ଶୂନ୍ୟ *vx, dst_t *y, ଅଚଳ int64_t k,
                                       dpct::queue_ptr stream) {
    ଅଚଳ int64_t nb = k / QK_K;
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

template <typename dst_t>
ସ୍ଥିର ଶୂନ୍ୟ dequantize_row_iq2_s_sycl(ଅଚଳ ଶୂନ୍ୟ *vx, dst_t *y, ଅଚଳ int64_t k,
                                      dpct::queue_ptr stream) {
    ଅଚଳ int64_t nb = k / QK_K;
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


template <typename dst_t>
ସ୍ଥିର ଶୂନ୍ୟ dequantize_row_iq3_xxs_sycl(ଅଚଳ ଶୂନ୍ୟ *vx, dst_t *y, ଅଚଳ int64_t k,
                                        dpct::queue_ptr stream) {
    ଅଚଳ int64_t nb = k / QK_K;
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

template <typename dst_t>
ସ୍ଥିର ଶୂନ୍ୟ dequantize_row_iq3_xxs_sycl_reorder(ଅଚଳ ଶୂନ୍ୟ * vx, dst_t * y, ଅଚଳ int64_t k,
                                                dpct::queue_ptr stream) {
    ଅଚଳ int64_t nb = k / QK_K;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_item<3> item_ct1) {
            dequantize_block_iq3_xxs_reorder(
                vx, y + item_ct1.get_group(2) * QK_K, item_ct1.get_group(2), item_ct1, nb,
                iq3xxs_grid, ksigns_iq2xs, kmask_iq2xs);
        });
}

template <typename dst_t>
ସ୍ଥିର ଶୂନ୍ୟ dequantize_row_iq3_s_sycl(ଅଚଳ ଶୂନ୍ୟ *vx, dst_t *y, ଅଚଳ int64_t k,
                                        dpct::queue_ptr stream) {
    ଅଚଳ int64_t nb = k / QK_K;
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

template <typename dst_t>
ସ୍ଥିର ଶୂନ୍ୟ dequantize_row_iq3_s_sycl_reorder(ଅଚଳ ଶୂନ୍ୟ * vx, dst_t * y, ଅଚଳ int64_t k,
                                              dpct::queue_ptr stream) {
    ଅଚଳ int64_t nb = k / QK_K;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_item<3> item_ct1) {
            dequantize_block_iq3_s_reorder(vx, y + item_ct1.get_group(2) * QK_K, item_ct1.get_group(2),
                                           item_ct1, nb, kmask_iq2xs, iq3s_grid);
        });
}

template <typename dst_t>
ସ୍ଥିର ଶୂନ୍ୟ dequantize_row_iq4_xs_sycl(ଅଚଳ ଶୂନ୍ୟ *vx, dst_t *y, ଅଚଳ int64_t k,
                                       dpct::queue_ptr stream) {
    ଅଚଳ int64_t nb = (k + QK_K - 1) / QK_K;
#ଯଦି QK_K == 64
    dequantize_row_iq4_nl_sycl(vx, y, k, stream);
#ଅନ୍ୟଥା
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

template <typename dst_t>
ସ୍ଥିର ଶୂନ୍ୟ dequantize_row_iq4_nl_sycl(ଅଚଳ ଶୂନ୍ୟ *vx, dst_t *y, ଅଚଳ int64_t k,
                                       dpct::queue_ptr stream) {
    ଅଚଳ int64_t nb = (k + QK_K - 1) / QK_K;
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

template <typename dst_t>
ସ୍ଥିର ଶୂନ୍ୟ dequantize_row_mxfp4_sycl(ଅଚଳ ଶୂନ୍ୟ * vx, dst_t * y, ଅଚଳ int64_t k, dpct::queue_ptr stream) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb = (k + QK_K - 1) / QK_K;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_item<3> item_ct1) {
            dequantize_block_mxfp4(vx, y, item_ct1);
        });
}

template <typename dst_t>
ସ୍ଥିର ଶୂନ୍ୟ dequantize_row_mxfp4_sycl_reorder(ଅଚଳ ଶୂନ୍ୟ * vx, dst_t * y, ଅଚଳ int64_t k, dpct::queue_ptr stream) {
    GGML_ASSERT(k % QK_MXFP4 == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_warp = (k / QK_MXFP4 + WARP_SIZE - 1) / WARP_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, n_warp) * sycl::range<3>(1, 1, WARP_SIZE),
                          sycl::range<3>(1, 1, WARP_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            dequantize_block_mxfp4_reorder(vx, y, k, item_ct1);
        });
}

template <typename dst_t>
ସ୍ଥିର ଶୂନ୍ୟ dequantize_row_nvfp4_sycl(ଅଚଳ ଶୂନ୍ୟ * vx, dst_t * y, ଅଚଳ int64_t k, dpct::queue_ptr stream) {
    GGML_ASSERT(k % QK_NVFP4 == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb = k / QK_NVFP4;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_item<3> /*item_ct1*/) {
            dequantize_block_nvfp4(vx, y, k);
        });
}


template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା qk, ପୂର୍ଣ୍ଣସଂଖ୍ୟା qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
ସ୍ଥିର ଶୂନ୍ୟ dequantize_block_nc(ଅଚଳ ଶୂନ୍ୟ * __restrict__ vx, dst_t * __restrict__ y,
        ଅଚଳ int64_t ne00, ଅଚଳ int64_t ne01, ଅଚଳ int64_t ne02,
        ଅଚଳ int64_t s01, ଅଚଳ int64_t s02, ଅଚଳ int64_t s03) {
    auto          item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ଅଚଳ int64_t i00 = 2 * (int64_t(item_ct1.get_local_range(2)) * item_ct1.get_group(2) + item_ct1.get_local_id(2));

    ଯଦି (i00 >= ne00) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ int64_t i01 = item_ct1.get_group(1);
    ଅଚଳ int64_t i02 = item_ct1.get_group(0) % ne02;
    ଅଚଳ int64_t i03 = item_ct1.get_group(0) / ne02;

    ଅଚଳ int64_t ibx0 = i03*s03 + i02*s02 + i01*s01;

    ଅଚଳ int64_t ib = ibx0 + i00/qk; // block index
    ଅଚଳ int64_t iqs = (i00%qk)/qr; // quant index
    ଅଚଳ int64_t iybs = i00 - i00%qk; // y block start index
    ଅଚଳ int64_t y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    #ifdef GGML_SYCL_F16
        sycl::half2 v;
    #ଅନ୍ୟଥା
        sycl::float2 v;
    #endif

    dequantize_kernel(vx, ib, iqs, v);

    ଅଚଳ int64_t iy0 = ((i03*ne02 + i02)*ne01 + i01)*ne00 + iybs + iqs;
    y[iy0 + 0]        = ggml_sycl_cast<dst_t>(v.x());
    y[iy0 + y_offset] = ggml_sycl_cast<dst_t>(v.y());
}


template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା qk, ପୂର୍ଣ୍ଣସଂଖ୍ୟା qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
ସ୍ଥିର ଶୂନ୍ୟ dequantize_block_nc_sycl(ଅଚଳ ଶୂନ୍ୟ *    vx,
                                  dst_t *         y,
                                  ଅଚଳ int64_t   ne00,
                                  ଅଚଳ int64_t   ne01,
                                  ଅଚଳ int64_t   ne02,
                                  ଅଚଳ int64_t   ne03,
                                  ଅଚଳ int64_t   s01,
                                  ଅଚଳ int64_t   s02,
                                  ଅଚଳ int64_t   s03,
                                  dpct::queue_ptr stream) {
    ଅଚଳ dpct::dim3 ଅଙ୍କ_blocks((ne00 + 2 * SYCL_DEQUANTIZE_BLOCK_SIZE - 1) / (2 * SYCL_DEQUANTIZE_BLOCK_SIZE), ne01,
                                ne02 * ne03);
    stream->parallel_for(sycl::nd_range<3>(ଅଙ୍କ_blocks * sycl::range<3>(1, 1, SYCL_DEQUANTIZE_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_DEQUANTIZE_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) {
                             GGML_UNUSED(item_ct1);
                             dequantize_block_nc<qk, qr, dequantize_kernel>(vx, y, ne00, ne01, ne02, s01, s02, s03);
                         });
}
template <typename src_t, typename dst_t>
ସ୍ଥିର ଶୂନ୍ୟ convert_unary_nc(ଅଚଳ ଶୂନ୍ୟ * __restrict__ vx, dst_t * __restrict__ y, ଅଚଳ int64_t ne00, ଅଚଳ int64_t ne01,
                          ଅଚଳ int64_t ne02, ଅଚଳ int64_t s01, ଅଚଳ int64_t s02, ଅଚଳ int64_t s03,
                          ଅଚଳ sycl::nd_item<3> & item_ct1) {

    ଅଚଳ int64_t work_group_size = item_ct1.get_local_range(2);
    ଅଚଳ int64_t global_id       = item_ct1.get_local_id(2) + work_group_size * item_ct1.get_group(2);

    ଅଚଳ int64_t i01 = item_ct1.get_group(1);
    ଅଚଳ int64_t i02 = item_ct1.get_group(0) % ne02;
    ଅଚଳ int64_t i03 = item_ct1.get_group(0) / ne02;

    // make each work-item deal with more elements since sycl global range can not exceed max int
    ଅଚଳ src_t * x = static_cast<ଅଚଳ src_t *>(vx);
    ଅଚଳ int64_t ix = i03 * s03 + i02 * s02 + i01 * s01;
    ଅଚଳ int64_t iy = ((i03 * ne02 + i02) * ne01 + i01) * ne00;

#pragma unroll
    ପାଇଁ (int64_t i00 = global_id; i00 < ne00; i00 += work_group_size * item_ct1.get_group_range(2)) {
        y[iy + i00] = static_cast<dst_t>(x[ix + i00]);
    }
}

template <typename src_t, typename dst_t>
ସ୍ଥିର ଶୂନ୍ୟ convert_unary_nc_sycl(ଅଚଳ ଶୂନ୍ୟ * __restrict__ vx, dst_t * __restrict__ y,
                                  ଅଚଳ int64_t ne00, ଅଚଳ int64_t ne01, ଅଚଳ int64_t ne02, ଅଚଳ int64_t ne03,
                                  ଅଚଳ int64_t s01, ଅଚଳ int64_t s02, ଅଚଳ int64_t s03, dpct::queue_ptr queue) {
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

template <typename src_t, typename dst_t>
ସ୍ଥିର ଶୂନ୍ୟ convert_unary_sycl(ଅଚଳ ଶୂନ୍ୟ * vx, dst_t * y, ଅଚଳ int64_t k, dpct::queue_ptr queue) {
    convert_unary_nc_sycl<src_t>(vx, y, k, 1, 1, 1, k, k, k, queue);
}


ସ୍ଥିର ତାର୍କିକ is_reordered_src0(ଅଚଳ ggml_tensor * dst) {
    ଫେରନ୍ତୁ dst->src[0]->extra &&
           ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder;
}

template <typename dst_t>
ସ୍ଥିର to_t_sycl_t<dst_t> ggml_get_dequant_sycl(ggml_type type, ggml_tensor * dst) {
    ବାଛ (type) {
        ସ୍ଥିତି GGML_TYPE_Q1_0:
            ଫେରନ୍ତୁ dequantize_block_sycl<QK1_0, QR1_0, dequantize_q1_0, dst_t>;
        ସ୍ଥିତି GGML_TYPE_Q2_0:
            ଫେରନ୍ତୁ dequantize_block_sycl<QK2_0, QR2_0, dequantize_q2_0, dst_t>;
        ସ୍ଥିତି GGML_TYPE_Q4_0:
            ଯଦି (is_reordered_src0(dst)) {
                ଫେରନ୍ତୁ dequantize_row_q4_0_sycl_reorder;
            } ଅନ୍ୟଥା ଯଦି constexpr (std::is_same_v<dst_t, ଦଶମିକ>) {
                ଫେରନ୍ତୁ dequantize_row_q4_0_sycl;
            } ଅନ୍ୟଥା {
                ଫେରନ୍ତୁ dequantize_block_sycl<QK4_0, QR4_0, dequantize_q4_0, dst_t>;
            }
        ସ୍ଥିତି GGML_TYPE_Q4_1:
            ଯଦି constexpr (std::is_same_v<dst_t, ଦଶମିକ>) {
                ଫେରନ୍ତୁ dequantize_row_q4_1_sycl;
            } ଅନ୍ୟଥା {
                ଫେରନ୍ତୁ dequantize_block_sycl<QK4_1, QR4_1, dequantize_q4_1, dst_t>;
            }
        ସ୍ଥିତି GGML_TYPE_Q5_0:
            ଫେରନ୍ତୁ dequantize_block_sycl<QK5_0, QR5_0, dequantize_q5_0, dst_t>;
        ସ୍ଥିତି GGML_TYPE_Q5_1:
            ଫେରନ୍ତୁ dequantize_block_sycl<QK5_1, QR5_1, dequantize_q5_1, dst_t>;
        ସ୍ଥିତି GGML_TYPE_Q8_0:
            ଯଦି (is_reordered_src0(dst)) {
                ଫେରନ୍ତୁ dequantize_row_q8_0_sycl_reorder;
            } ଅନ୍ୟଥା {
                ଫେରନ୍ତୁ dequantize_block_sycl<QK8_0, QR8_0, dequantize_q8_0, dst_t>;
            }
        ସ୍ଥିତି GGML_TYPE_Q2_K:
            ଯଦି (is_reordered_src0(dst)) {
                ଫେରନ୍ତୁ dequantize_row_q2_K_sycl_reorder;
            } ଅନ୍ୟଥା {
                ଫେରନ୍ତୁ dequantize_row_q2_K_sycl;
            }
        ସ୍ଥିତି GGML_TYPE_Q3_K:
            ଯଦି (is_reordered_src0(dst)) {
                ଫେରନ୍ତୁ dequantize_row_q3_K_sycl_reorder;
            } ଅନ୍ୟଥା {
                ଫେରନ୍ତୁ dequantize_row_q3_K_sycl;
            }
        ସ୍ଥିତି GGML_TYPE_Q4_K:
            ଯଦି (is_reordered_src0(dst)) {
                ଫେରନ୍ତୁ dequantize_row_q4_K_sycl_reorder;
            } ଅନ୍ୟଥା {
                ଫେରନ୍ତୁ dequantize_row_q4_K_sycl;
            }
        ସ୍ଥିତି GGML_TYPE_Q5_K:
            ଯଦି (is_reordered_src0(dst)) {
                ଫେରନ୍ତୁ dequantize_row_q5_K_sycl_reorder;
            } ଅନ୍ୟଥା {
                ଫେରନ୍ତୁ dequantize_row_q5_K_sycl;
            }
        ସ୍ଥିତି GGML_TYPE_Q6_K:
            ଯଦି (is_reordered_src0(dst)) {
                ଫେରନ୍ତୁ dequantize_row_q6_K_sycl_reorder;
            } ଅନ୍ୟଥା {
                ଫେରନ୍ତୁ dequantize_row_q6_K_sycl;
            }
        ସ୍ଥିତି GGML_TYPE_IQ1_S:
            ଫେରନ୍ତୁ dequantize_row_iq1_s_sycl;
        ସ୍ଥିତି GGML_TYPE_IQ1_M:
            ଫେରନ୍ତୁ dequantize_row_iq1_m_sycl;
        ସ୍ଥିତି GGML_TYPE_IQ2_XXS:
            ଫେରନ୍ତୁ dequantize_row_iq2_xxs_sycl;
        ସ୍ଥିତି GGML_TYPE_IQ2_XS:
            ଫେରନ୍ତୁ dequantize_row_iq2_xs_sycl;
        ସ୍ଥିତି GGML_TYPE_IQ2_S:
            ଫେରନ୍ତୁ dequantize_row_iq2_s_sycl;
        ସ୍ଥିତି GGML_TYPE_IQ3_XXS:
            ଯଦି (is_reordered_src0(dst)) {
                ଫେରନ୍ତୁ dequantize_row_iq3_xxs_sycl_reorder;
            } ଅନ୍ୟଥା {
                ଫେରନ୍ତୁ dequantize_row_iq3_xxs_sycl;
            }
        ସ୍ଥିତି GGML_TYPE_IQ3_S:
            ଯଦି (is_reordered_src0(dst)) {
                ଫେରନ୍ତୁ dequantize_row_iq3_s_sycl_reorder;
            } ଅନ୍ୟଥା {
                ଫେରନ୍ତୁ dequantize_row_iq3_s_sycl;
            }
        ସ୍ଥିତି GGML_TYPE_IQ4_XS:
            ଫେରନ୍ତୁ dequantize_row_iq4_xs_sycl;
        ସ୍ଥିତି GGML_TYPE_IQ4_NL:
            ଫେରନ୍ତୁ dequantize_row_iq4_nl_sycl;
        ସ୍ଥିତି GGML_TYPE_MXFP4:
            ଯଦି (is_reordered_src0(dst)) {
                ଫେରନ୍ତୁ dequantize_row_mxfp4_sycl_reorder;
            }
            ଫେରନ୍ତୁ dequantize_row_mxfp4_sycl;
        ସ୍ଥିତି GGML_TYPE_NVFP4:
            ଫେରନ୍ତୁ dequantize_row_nvfp4_sycl;
        ସ୍ଥିତି GGML_TYPE_F32:
            ଯଦି constexpr (std::is_same_v<dst_t, sycl::half>) {
                ଫେରନ୍ତୁ convert_unary_sycl<ଦଶମିକ>;
            }
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F16:
            ଯଦି constexpr (std::is_same_v<dst_t, ଦଶମିକ>) {
                ଫେରନ୍ତୁ convert_unary_sycl<sycl::half>;
            }
            ଭାଙ୍ଗ;
#ifdef GGML_SYCL_HAS_BF16
        ସ୍ଥିତି GGML_TYPE_BF16:
            ଫେରନ୍ତୁ convert_unary_sycl<sycl::ext::oneapi::bfloat16>;
#endif
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଭାଙ୍ଗ;
    }
    GGML_ABORT("fatal error: unsupport data type=%s\n", ggml_type_name(type));
}

to_fp16_sycl_t ggml_get_to_fp16_sycl(ggml_type type, ggml_tensor * dst) {
    ଫେରନ୍ତୁ ggml_get_dequant_sycl<sycl::half>(type, dst);
}

to_fp32_sycl_t ggml_get_to_fp32_sycl(ggml_type type, ggml_tensor *dst) {
    ଫେରନ୍ତୁ ggml_get_dequant_sycl<ଦଶମିକ>(type, dst);
}


#ifdef GGML_SYCL_HAS_BF16
to_bf16_sycl_t ggml_get_to_bf16_sycl(ggml_type type, ggml_tensor * /*dst*/) {
    ବାଛ (type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            ଫେରନ୍ତୁ convert_unary_sycl<ଦଶମିକ>;
        ସ୍ଥିତି GGML_TYPE_F16:
            ଫେରନ୍ତୁ convert_unary_sycl<sycl::half>;
        ସ୍ଥିତି GGML_TYPE_BF16:
            ଫେରନ୍ତୁ convert_unary_sycl<sycl::ext::oneapi::bfloat16>;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            GGML_ABORT("fatal error: unsupport data type=%s\n", ggml_type_name(type));
    }
}
#endif

to_fp16_nc_sycl_t ggml_get_to_fp16_nc_sycl(ggml_type type) {
    ବାଛ (type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            ଫେରନ୍ତୁ convert_unary_nc_sycl<ଦଶମିକ>;
#ifdef GGML_SYCL_HAS_BF16
        ସ୍ଥିତି GGML_TYPE_BF16:
            ଫେରନ୍ତୁ convert_unary_nc_sycl<sycl::ext::oneapi::bfloat16>;
#endif
        ସ୍ଥିତି GGML_TYPE_Q1_0:
            ଫେରନ୍ତୁ dequantize_block_nc_sycl<QK1_0, QR1_0, dequantize_q1_0>;
        ସ୍ଥିତି GGML_TYPE_Q4_0:
            ଫେରନ୍ତୁ dequantize_block_nc_sycl<QK4_0, QR4_0, dequantize_q4_0>;
        ସ୍ଥିତି GGML_TYPE_Q4_1:
            ଫେରନ୍ତୁ dequantize_block_nc_sycl<QK4_1, QR4_1, dequantize_q4_1>;
        ସ୍ଥିତି GGML_TYPE_Q5_0:
            ଫେରନ୍ତୁ dequantize_block_nc_sycl<QK5_0, QR5_0, dequantize_q5_0>;
        ସ୍ଥିତି GGML_TYPE_Q5_1:
            ଫେରନ୍ତୁ dequantize_block_nc_sycl<QK5_1, QR5_1, dequantize_q5_1>;
        ସ୍ଥିତି GGML_TYPE_Q8_0:
            ଫେରନ୍ତୁ dequantize_block_nc_sycl<QK8_0, QR8_0, dequantize_q8_0>;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ nullptr;
    }
}
