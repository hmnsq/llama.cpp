#ಅನ್ತರ್ಭಾವ "convert.hpp"
#ಅನ್ತರ್ಭಾವ "dequantize.hpp"
#ಅನ್ತರ್ಭಾವ "presets.hpp"

template <ಪೂರ್ಣಾಙ್ಕ qk, ಪೂರ್ಣಾಙ್ಕ qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_block(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ y, ಧ್ರುವ int64_t k,
                             ಧ್ರುವ sycl::nd_item<3> &item_ct1) {
    ಧ್ರುವ int64_t i = 2 * (item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                       item_ct1.get_local_id(2));

    ಯದಿ (i >= k) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಧ್ರುವ int64_t ib = i/qk; // block index
    ಧ್ರುವ int64_t iqs = (i%qk)/qr; // quant index
    ಧ್ರುವ int64_t iybs = i - i%qk; // y block start index
    ಧ್ರುವ int64_t y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    dfloat2 v;
    dequantize_kernel(vx, ib, iqs, v);

    y[iybs + iqs + 0] = v.x();
    y[iybs + iqs + y_offset] = v.y();
}

template <ಪೂರ್ಣಾಙ್ಕ qk, ಪೂರ್ಣಾಙ್ಕ qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_block_sycl(ಧ್ರುವ ಶೂನ್ಯ *__restrict__ vx,
                                  dst_t *__restrict__ y, ಧ್ರುವ int64_t k,
                                  dpct::queue_ptr stream) {
    ಧ್ರುವ int64_t ಅಂಕೆ_blocks = (k + 2*SYCL_DEQUANTIZE_BLOCK_SIZE - 1) / (2*SYCL_DEQUANTIZE_BLOCK_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, ಅಂಕೆ_blocks) *
                    sycl::range<3>(1, 1, SYCL_DEQUANTIZE_BLOCK_SIZE),
                sycl::range<3>(1, 1, SYCL_DEQUANTIZE_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) {
                dequantize_block<qk, qr, dequantize_kernel>(vx, y, k, item_ct1);
            });
    }
}

template <typename dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_row_q2_K_sycl(ಧ್ರುವ ಶೂನ್ಯ *vx, dst_t *y, ಧ್ರುವ int64_t k,
                                     dpct::queue_ptr stream) {
    ಧ್ರುವ int64_t nb = k / QK_K;
#ಯದಿ QK_K == 256
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
#ಅನ್ಯಥಾ
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
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_row_q2_K_sycl_reorder(ಧ್ರುವ ಶೂನ್ಯ *vx, dst_t *y, ಧ್ರುವ int64_t k,
                                             dpct::queue_ptr stream) {
    ಧ್ರುವ int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
        [=](sycl::nd_item<3> item_ct1) {
            dequantize_block_q2_K_reorder(vx, y + item_ct1.get_group(2) * QK_K, item_ct1.get_group(2), item_ct1, nb);
        });
}

template <typename dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_row_q3_K_sycl(ಧ್ರುವ ಶೂನ್ಯ *vx, dst_t *y, ಧ್ರುವ int64_t k,
                                     dpct::queue_ptr stream) {
    ಧ್ರುವ int64_t nb = k / QK_K;
#ಯದಿ QK_K == 256
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
#ಅನ್ಯಥಾ
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
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_row_q3_K_sycl_reorder(ಧ್ರುವ ಶೂನ್ಯ *vx, dst_t *y, ಧ್ರುವ int64_t k,
                                             dpct::queue_ptr stream) {
    ಧ್ರುವ int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
        [=](sycl::nd_item<3> item_ct1) {
            dequantize_block_q3_K_reorder(vx, y + item_ct1.get_group(2) * QK_K, item_ct1.get_group(2), item_ct1, nb);
        });
}

template <typename dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_row_q4_0_sycl(ಧ್ರುವ ಶೂನ್ಯ *vx, dst_t *y, ಧ್ರುವ int64_t k,
                                     dpct::queue_ptr stream) {
    ಧ್ರುವ int64_t nb32 = k / 32;
    ಧ್ರುವ int64_t nb = (k + 255) / 256;
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
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_row_q4_0_sycl_reorder(ಧ್ರುವ ಶೂನ್ಯ *vx, dst_t *y, ಧ್ರುವ int64_t k,
                                     dpct::queue_ptr stream) {

    dpct::has_capability_or_fail(stream->get_device(),
                                    {sycl::aspect::fp16});

    GGML_ASSERT(k % QK4_0 == 0);
    ಧ್ರುವ int64_t n_groups = ceil_div(k / 2, SYCL_DEQUANTIZE_BLOCK_SIZE);
    // decrease global range when it exceeds the max int; the kernel loops to cover the rest
    ಧ್ರುವ int64_t wg_size  = downsample_sycl_global_range(n_groups, SYCL_DEQUANTIZE_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, n_groups) *
        sycl::range<3>(1, 1, wg_size),
        sycl::range<3>(1, 1, wg_size)),
        [=](sycl::nd_item<3> item_ct1) {
            dequantize_block_q4_0_reorder(vx, y, k, item_ct1);
        });
}

template <typename dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_row_q8_0_sycl_reorder(ಧ್ರುವ ಶೂನ್ಯ *vx, dst_t *y, ಧ್ರುವ int64_t k,
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
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_row_q4_1_sycl(ಧ್ರುವ ಶೂನ್ಯ *vx, dst_t *y, ಧ್ರುವ int64_t k,
                                     dpct::queue_ptr stream) {
    ಧ್ರುವ int64_t nb32 = k / 32;
    ಧ್ರುವ int64_t nb = (k + 255) / 256;
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
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_row_q4_K_sycl(ಧ್ರುವ ಶೂನ್ಯ *vx, dst_t *y, ಧ್ರುವ int64_t k,
                                     dpct::queue_ptr stream) {
    ಧ್ರುವ int64_t nb = k / QK_K;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) nb * SYCL_DEQK_WI_PER_BLOCK),
                                           sycl::range<1>((size_t) SYCL_DEQK_WI_PER_BLOCK)),
                         [=](sycl::nd_item<1> it) { dequantize_block_q4_K_wide(vx, y, nb, it); });
}

template <typename dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_row_q4_K_sycl_reorder(ಧ್ರುವ ಶೂನ್ಯ * vx, dst_t * y, ಧ್ರುವ int64_t k, dpct::queue_ptr stream) {
    ಧ್ರುವ int64_t nb = k / QK_K;
    ಧ್ರುವ size_t  local_size  = 32;
    ಧ್ರುವ size_t  global_size = nb * local_size;

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
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_row_q5_K_sycl(ಧ್ರುವ ಶೂನ್ಯ *vx, dst_t *y, ಧ್ರುವ int64_t k,
                                     dpct::queue_ptr stream) {
    ಧ್ರುವ int64_t nb = k / QK_K;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) nb * SYCL_DEQK_WI_PER_BLOCK),
                                           sycl::range<1>((size_t) SYCL_DEQK_WI_PER_BLOCK)),
                         [=](sycl::nd_item<1> it) { dequantize_block_q5_K_wide(vx, y, nb, it); });
}

template <typename dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_row_q5_K_sycl_reorder(ಧ್ರುವ ಶೂನ್ಯ * vx, dst_t * y, ಧ್ರುವ int64_t k, dpct::queue_ptr stream) {
    ಧ್ರುವ int64_t nb = k / QK_K;

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
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_row_q6_K_sycl(ಧ್ರುವ ಶೂನ್ಯ *vx, dst_t *y, ಧ್ರುವ int64_t k,
                                     dpct::queue_ptr stream) {
    ಧ್ರುವ int64_t nb = k / QK_K;
#ಯದಿ QK_K == 256
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
#ಅನ್ಯಥಾ
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
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_row_q6_K_sycl_reorder(ಧ್ರುವ ಶೂನ್ಯ * vx, dst_t * y, ಧ್ರುವ int64_t k, dpct::queue_ptr stream) {
    ಧ್ರುವ int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
        [=](sycl::nd_item<3> item_ct1) { dequantize_block_q6_K_reorder(vx, y + item_ct1.get_group(2) * QK_K, item_ct1.get_group(2), item_ct1, nb); });
}

template <typename dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_row_iq1_s_sycl(ಧ್ರುವ ಶೂನ್ಯ *vx, dst_t *y, ಧ್ರುವ int64_t k,
                                        dpct::queue_ptr stream) {
    ಧ್ರುವ int64_t nb = k / QK_K;
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
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_row_iq1_m_sycl(ಧ್ರುವ ಶೂನ್ಯ *vx, dst_t *y, ಧ್ರುವ int64_t k,
                                        dpct::queue_ptr stream) {
    ಧ್ರುವ int64_t nb = k / QK_K;
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
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_row_iq2_xxs_sycl(ಧ್ರುವ ಶೂನ್ಯ *vx, dst_t *y, ಧ್ರುವ int64_t k,
                                        dpct::queue_ptr stream) {
    ಧ್ರುವ int64_t nb = k / QK_K;
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
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_row_iq2_xs_sycl(ಧ್ರುವ ಶೂನ್ಯ *vx, dst_t *y, ಧ್ರುವ int64_t k,
                                       dpct::queue_ptr stream) {
    ಧ್ರುವ int64_t nb = k / QK_K;
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
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_row_iq2_s_sycl(ಧ್ರುವ ಶೂನ್ಯ *vx, dst_t *y, ಧ್ರುವ int64_t k,
                                      dpct::queue_ptr stream) {
    ಧ್ರುವ int64_t nb = k / QK_K;
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
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_row_iq3_xxs_sycl(ಧ್ರುವ ಶೂನ್ಯ *vx, dst_t *y, ಧ್ರುವ int64_t k,
                                        dpct::queue_ptr stream) {
    ಧ್ರುವ int64_t nb = k / QK_K;
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
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_row_iq3_xxs_sycl_reorder(ಧ್ರುವ ಶೂನ್ಯ * vx, dst_t * y, ಧ್ರುವ int64_t k,
                                                dpct::queue_ptr stream) {
    ಧ್ರುವ int64_t nb = k / QK_K;
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
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_row_iq3_s_sycl(ಧ್ರುವ ಶೂನ್ಯ *vx, dst_t *y, ಧ್ರುವ int64_t k,
                                        dpct::queue_ptr stream) {
    ಧ್ರುವ int64_t nb = k / QK_K;
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
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_row_iq3_s_sycl_reorder(ಧ್ರುವ ಶೂನ್ಯ * vx, dst_t * y, ಧ್ರುವ int64_t k,
                                              dpct::queue_ptr stream) {
    ಧ್ರುವ int64_t nb = k / QK_K;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_item<3> item_ct1) {
            dequantize_block_iq3_s_reorder(vx, y + item_ct1.get_group(2) * QK_K, item_ct1.get_group(2),
                                           item_ct1, nb, kmask_iq2xs, iq3s_grid);
        });
}

template <typename dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_row_iq4_xs_sycl(ಧ್ರುವ ಶೂನ್ಯ *vx, dst_t *y, ಧ್ರುವ int64_t k,
                                       dpct::queue_ptr stream) {
    ಧ್ರುವ int64_t nb = (k + QK_K - 1) / QK_K;
#ಯದಿ QK_K == 64
    dequantize_row_iq4_nl_sycl(vx, y, k, stream);
#ಅನ್ಯಥಾ
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
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_row_iq4_nl_sycl(ಧ್ರುವ ಶೂನ್ಯ *vx, dst_t *y, ಧ್ರುವ int64_t k,
                                       dpct::queue_ptr stream) {
    ಧ್ರುವ int64_t nb = (k + QK_K - 1) / QK_K;
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
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_row_mxfp4_sycl(ಧ್ರುವ ಶೂನ್ಯ * vx, dst_t * y, ಧ್ರುವ int64_t k, dpct::queue_ptr stream) {
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb = (k + QK_K - 1) / QK_K;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_item<3> item_ct1) {
            dequantize_block_mxfp4(vx, y, item_ct1);
        });
}

template <typename dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_row_mxfp4_sycl_reorder(ಧ್ರುವ ಶೂನ್ಯ * vx, dst_t * y, ಧ್ರುವ int64_t k, dpct::queue_ptr stream) {
    GGML_ASSERT(k % QK_MXFP4 == 0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_warp = (k / QK_MXFP4 + WARP_SIZE - 1) / WARP_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, n_warp) * sycl::range<3>(1, 1, WARP_SIZE),
                          sycl::range<3>(1, 1, WARP_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            dequantize_block_mxfp4_reorder(vx, y, k, item_ct1);
        });
}

template <typename dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_row_nvfp4_sycl(ಧ್ರುವ ಶೂನ್ಯ * vx, dst_t * y, ಧ್ರುವ int64_t k, dpct::queue_ptr stream) {
    GGML_ASSERT(k % QK_NVFP4 == 0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb = k / QK_NVFP4;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_item<3> /*item_ct1*/) {
            dequantize_block_nvfp4(vx, y, k);
        });
}


template <ಪೂರ್ಣಾಙ್ಕ qk, ಪೂರ್ಣಾಙ್ಕ qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_block_nc(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ y,
        ಧ್ರುವ int64_t ne00, ಧ್ರುವ int64_t ne01, ಧ್ರುವ int64_t ne02,
        ಧ್ರುವ int64_t s01, ಧ್ರುವ int64_t s02, ಧ್ರುವ int64_t s03) {
    auto          item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ಧ್ರುವ int64_t i00 = 2 * (int64_t(item_ct1.get_local_range(2)) * item_ct1.get_group(2) + item_ct1.get_local_id(2));

    ಯದಿ (i00 >= ne00) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಧ್ರುವ int64_t i01 = item_ct1.get_group(1);
    ಧ್ರುವ int64_t i02 = item_ct1.get_group(0) % ne02;
    ಧ್ರುವ int64_t i03 = item_ct1.get_group(0) / ne02;

    ಧ್ರುವ int64_t ibx0 = i03*s03 + i02*s02 + i01*s01;

    ಧ್ರುವ int64_t ib = ibx0 + i00/qk; // block index
    ಧ್ರುವ int64_t iqs = (i00%qk)/qr; // quant index
    ಧ್ರುವ int64_t iybs = i00 - i00%qk; // y block start index
    ಧ್ರುವ int64_t y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    #ifdef GGML_SYCL_F16
        sycl::half2 v;
    #ಅನ್ಯಥಾ
        sycl::float2 v;
    #endif

    dequantize_kernel(vx, ib, iqs, v);

    ಧ್ರುವ int64_t iy0 = ((i03*ne02 + i02)*ne01 + i01)*ne00 + iybs + iqs;
    y[iy0 + 0]        = ggml_sycl_cast<dst_t>(v.x());
    y[iy0 + y_offset] = ggml_sycl_cast<dst_t>(v.y());
}


template <ಪೂರ್ಣಾಙ್ಕ qk, ಪೂರ್ಣಾಙ್ಕ qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_block_nc_sycl(ಧ್ರುವ ಶೂನ್ಯ *    vx,
                                  dst_t *         y,
                                  ಧ್ರುವ int64_t   ne00,
                                  ಧ್ರುವ int64_t   ne01,
                                  ಧ್ರುವ int64_t   ne02,
                                  ಧ್ರುವ int64_t   ne03,
                                  ಧ್ರುವ int64_t   s01,
                                  ಧ್ರುವ int64_t   s02,
                                  ಧ್ರುವ int64_t   s03,
                                  dpct::queue_ptr stream) {
    ಧ್ರುವ dpct::dim3 ಅಂಕೆ_blocks((ne00 + 2 * SYCL_DEQUANTIZE_BLOCK_SIZE - 1) / (2 * SYCL_DEQUANTIZE_BLOCK_SIZE), ne01,
                                ne02 * ne03);
    stream->parallel_for(sycl::nd_range<3>(ಅಂಕೆ_blocks * sycl::range<3>(1, 1, SYCL_DEQUANTIZE_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_DEQUANTIZE_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) {
                             GGML_UNUSED(item_ct1);
                             dequantize_block_nc<qk, qr, dequantize_kernel>(vx, y, ne00, ne01, ne02, s01, s02, s03);
                         });
}
template <typename src_t, typename dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ convert_unary_nc(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ y, ಧ್ರುವ int64_t ne00, ಧ್ರುವ int64_t ne01,
                          ಧ್ರುವ int64_t ne02, ಧ್ರುವ int64_t s01, ಧ್ರುವ int64_t s02, ಧ್ರುವ int64_t s03,
                          ಧ್ರುವ sycl::nd_item<3> & item_ct1) {

    ಧ್ರುವ int64_t work_group_size = item_ct1.get_local_range(2);
    ಧ್ರುವ int64_t global_id       = item_ct1.get_local_id(2) + work_group_size * item_ct1.get_group(2);

    ಧ್ರುವ int64_t i01 = item_ct1.get_group(1);
    ಧ್ರುವ int64_t i02 = item_ct1.get_group(0) % ne02;
    ಧ್ರುವ int64_t i03 = item_ct1.get_group(0) / ne02;

    // make each work-item deal with more elements since sycl global range can not exceed max int
    ಧ್ರುವ src_t * x = static_cast<ಧ್ರುವ src_t *>(vx);
    ಧ್ರುವ int64_t ix = i03 * s03 + i02 * s02 + i01 * s01;
    ಧ್ರುವ int64_t iy = ((i03 * ne02 + i02) * ne01 + i01) * ne00;

#pragma unroll
    ಪ್ರತಿ (int64_t i00 = global_id; i00 < ne00; i00 += work_group_size * item_ct1.get_group_range(2)) {
        y[iy + i00] = static_cast<dst_t>(x[ix + i00]);
    }
}

template <typename src_t, typename dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ convert_unary_nc_sycl(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ y,
                                  ಧ್ರುವ int64_t ne00, ಧ್ರುವ int64_t ne01, ಧ್ರುವ int64_t ne02, ಧ್ರುವ int64_t ne03,
                                  ಧ್ರುವ int64_t s01, ಧ್ರುವ int64_t s02, ಧ್ರುವ int64_t s03, dpct::queue_ptr queue) {
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
ಸ್ಥಿರ ಶೂನ್ಯ convert_unary_sycl(ಧ್ರುವ ಶೂನ್ಯ * vx, dst_t * y, ಧ್ರುವ int64_t k, dpct::queue_ptr queue) {
    convert_unary_nc_sycl<src_t>(vx, y, k, 1, 1, 1, k, k, k, queue);
}


ಸ್ಥಿರ ತಾರ್ಕಿಕ is_reordered_src0(ಧ್ರುವ ggml_tensor * dst) {
    ಪ್ರತ್ಯಾಗಮ dst->src[0]->extra &&
           ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder;
}

template <typename dst_t>
ಸ್ಥಿರ to_t_sycl_t<dst_t> ggml_get_dequant_sycl(ggml_type type, ggml_tensor * dst) {
    ವಿಕಲ್ಪ (type) {
        ಪಕ್ಷ GGML_TYPE_Q1_0:
            ಪ್ರತ್ಯಾಗಮ dequantize_block_sycl<QK1_0, QR1_0, dequantize_q1_0, dst_t>;
        ಪಕ್ಷ GGML_TYPE_Q2_0:
            ಪ್ರತ್ಯಾಗಮ dequantize_block_sycl<QK2_0, QR2_0, dequantize_q2_0, dst_t>;
        ಪಕ್ಷ GGML_TYPE_Q4_0:
            ಯದಿ (is_reordered_src0(dst)) {
                ಪ್ರತ್ಯಾಗಮ dequantize_row_q4_0_sycl_reorder;
            } ಅನ್ಯಥಾ ಯದಿ constexpr (std::is_same_v<dst_t, ಭಿನ್ನ>) {
                ಪ್ರತ್ಯಾಗಮ dequantize_row_q4_0_sycl;
            } ಅನ್ಯಥಾ {
                ಪ್ರತ್ಯಾಗಮ dequantize_block_sycl<QK4_0, QR4_0, dequantize_q4_0, dst_t>;
            }
        ಪಕ್ಷ GGML_TYPE_Q4_1:
            ಯದಿ constexpr (std::is_same_v<dst_t, ಭಿನ್ನ>) {
                ಪ್ರತ್ಯಾಗಮ dequantize_row_q4_1_sycl;
            } ಅನ್ಯಥಾ {
                ಪ್ರತ್ಯಾಗಮ dequantize_block_sycl<QK4_1, QR4_1, dequantize_q4_1, dst_t>;
            }
        ಪಕ್ಷ GGML_TYPE_Q5_0:
            ಪ್ರತ್ಯಾಗಮ dequantize_block_sycl<QK5_0, QR5_0, dequantize_q5_0, dst_t>;
        ಪಕ್ಷ GGML_TYPE_Q5_1:
            ಪ್ರತ್ಯಾಗಮ dequantize_block_sycl<QK5_1, QR5_1, dequantize_q5_1, dst_t>;
        ಪಕ್ಷ GGML_TYPE_Q8_0:
            ಯದಿ (is_reordered_src0(dst)) {
                ಪ್ರತ್ಯಾಗಮ dequantize_row_q8_0_sycl_reorder;
            } ಅನ್ಯಥಾ {
                ಪ್ರತ್ಯಾಗಮ dequantize_block_sycl<QK8_0, QR8_0, dequantize_q8_0, dst_t>;
            }
        ಪಕ್ಷ GGML_TYPE_Q2_K:
            ಯದಿ (is_reordered_src0(dst)) {
                ಪ್ರತ್ಯಾಗಮ dequantize_row_q2_K_sycl_reorder;
            } ಅನ್ಯಥಾ {
                ಪ್ರತ್ಯಾಗಮ dequantize_row_q2_K_sycl;
            }
        ಪಕ್ಷ GGML_TYPE_Q3_K:
            ಯದಿ (is_reordered_src0(dst)) {
                ಪ್ರತ್ಯಾಗಮ dequantize_row_q3_K_sycl_reorder;
            } ಅನ್ಯಥಾ {
                ಪ್ರತ್ಯಾಗಮ dequantize_row_q3_K_sycl;
            }
        ಪಕ್ಷ GGML_TYPE_Q4_K:
            ಯದಿ (is_reordered_src0(dst)) {
                ಪ್ರತ್ಯಾಗಮ dequantize_row_q4_K_sycl_reorder;
            } ಅನ್ಯಥಾ {
                ಪ್ರತ್ಯಾಗಮ dequantize_row_q4_K_sycl;
            }
        ಪಕ್ಷ GGML_TYPE_Q5_K:
            ಯದಿ (is_reordered_src0(dst)) {
                ಪ್ರತ್ಯಾಗಮ dequantize_row_q5_K_sycl_reorder;
            } ಅನ್ಯಥಾ {
                ಪ್ರತ್ಯಾಗಮ dequantize_row_q5_K_sycl;
            }
        ಪಕ್ಷ GGML_TYPE_Q6_K:
            ಯದಿ (is_reordered_src0(dst)) {
                ಪ್ರತ್ಯಾಗಮ dequantize_row_q6_K_sycl_reorder;
            } ಅನ್ಯಥಾ {
                ಪ್ರತ್ಯಾಗಮ dequantize_row_q6_K_sycl;
            }
        ಪಕ್ಷ GGML_TYPE_IQ1_S:
            ಪ್ರತ್ಯಾಗಮ dequantize_row_iq1_s_sycl;
        ಪಕ್ಷ GGML_TYPE_IQ1_M:
            ಪ್ರತ್ಯಾಗಮ dequantize_row_iq1_m_sycl;
        ಪಕ್ಷ GGML_TYPE_IQ2_XXS:
            ಪ್ರತ್ಯಾಗಮ dequantize_row_iq2_xxs_sycl;
        ಪಕ್ಷ GGML_TYPE_IQ2_XS:
            ಪ್ರತ್ಯಾಗಮ dequantize_row_iq2_xs_sycl;
        ಪಕ್ಷ GGML_TYPE_IQ2_S:
            ಪ್ರತ್ಯಾಗಮ dequantize_row_iq2_s_sycl;
        ಪಕ್ಷ GGML_TYPE_IQ3_XXS:
            ಯದಿ (is_reordered_src0(dst)) {
                ಪ್ರತ್ಯಾಗಮ dequantize_row_iq3_xxs_sycl_reorder;
            } ಅನ್ಯಥಾ {
                ಪ್ರತ್ಯಾಗಮ dequantize_row_iq3_xxs_sycl;
            }
        ಪಕ್ಷ GGML_TYPE_IQ3_S:
            ಯದಿ (is_reordered_src0(dst)) {
                ಪ್ರತ್ಯಾಗಮ dequantize_row_iq3_s_sycl_reorder;
            } ಅನ್ಯಥಾ {
                ಪ್ರತ್ಯಾಗಮ dequantize_row_iq3_s_sycl;
            }
        ಪಕ್ಷ GGML_TYPE_IQ4_XS:
            ಪ್ರತ್ಯಾಗಮ dequantize_row_iq4_xs_sycl;
        ಪಕ್ಷ GGML_TYPE_IQ4_NL:
            ಪ್ರತ್ಯಾಗಮ dequantize_row_iq4_nl_sycl;
        ಪಕ್ಷ GGML_TYPE_MXFP4:
            ಯದಿ (is_reordered_src0(dst)) {
                ಪ್ರತ್ಯಾಗಮ dequantize_row_mxfp4_sycl_reorder;
            }
            ಪ್ರತ್ಯಾಗಮ dequantize_row_mxfp4_sycl;
        ಪಕ್ಷ GGML_TYPE_NVFP4:
            ಪ್ರತ್ಯಾಗಮ dequantize_row_nvfp4_sycl;
        ಪಕ್ಷ GGML_TYPE_F32:
            ಯದಿ constexpr (std::is_same_v<dst_t, sycl::half>) {
                ಪ್ರತ್ಯಾಗಮ convert_unary_sycl<ಭಿನ್ನ>;
            }
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_F16:
            ಯದಿ constexpr (std::is_same_v<dst_t, ಭಿನ್ನ>) {
                ಪ್ರತ್ಯಾಗಮ convert_unary_sycl<sycl::half>;
            }
            ಭಙ್ಗ;
#ifdef GGML_SYCL_HAS_BF16
        ಪಕ್ಷ GGML_TYPE_BF16:
            ಪ್ರತ್ಯಾಗಮ convert_unary_sycl<sycl::ext::oneapi::bfloat16>;
#endif
        ಸ್ವಭಾವ:
            ಭಙ್ಗ;
    }
    GGML_ABORT("fatal error: unsupport data type=%s\n", ggml_type_name(type));
}

to_fp16_sycl_t ggml_get_to_fp16_sycl(ggml_type type, ggml_tensor * dst) {
    ಪ್ರತ್ಯಾಗಮ ggml_get_dequant_sycl<sycl::half>(type, dst);
}

to_fp32_sycl_t ggml_get_to_fp32_sycl(ggml_type type, ggml_tensor *dst) {
    ಪ್ರತ್ಯಾಗಮ ggml_get_dequant_sycl<ಭಿನ್ನ>(type, dst);
}


#ifdef GGML_SYCL_HAS_BF16
to_bf16_sycl_t ggml_get_to_bf16_sycl(ggml_type type, ggml_tensor * /*dst*/) {
    ವಿಕಲ್ಪ (type) {
        ಪಕ್ಷ GGML_TYPE_F32:
            ಪ್ರತ್ಯಾಗಮ convert_unary_sycl<ಭಿನ್ನ>;
        ಪಕ್ಷ GGML_TYPE_F16:
            ಪ್ರತ್ಯಾಗಮ convert_unary_sycl<sycl::half>;
        ಪಕ್ಷ GGML_TYPE_BF16:
            ಪ್ರತ್ಯಾಗಮ convert_unary_sycl<sycl::ext::oneapi::bfloat16>;
        ಸ್ವಭಾವ:
            GGML_ABORT("fatal error: unsupport data type=%s\n", ggml_type_name(type));
    }
}
#endif

to_fp16_nc_sycl_t ggml_get_to_fp16_nc_sycl(ggml_type type) {
    ವಿಕಲ್ಪ (type) {
        ಪಕ್ಷ GGML_TYPE_F32:
            ಪ್ರತ್ಯಾಗಮ convert_unary_nc_sycl<ಭಿನ್ನ>;
#ifdef GGML_SYCL_HAS_BF16
        ಪಕ್ಷ GGML_TYPE_BF16:
            ಪ್ರತ್ಯಾಗಮ convert_unary_nc_sycl<sycl::ext::oneapi::bfloat16>;
#endif
        ಪಕ್ಷ GGML_TYPE_Q1_0:
            ಪ್ರತ್ಯಾಗಮ dequantize_block_nc_sycl<QK1_0, QR1_0, dequantize_q1_0>;
        ಪಕ್ಷ GGML_TYPE_Q4_0:
            ಪ್ರತ್ಯಾಗಮ dequantize_block_nc_sycl<QK4_0, QR4_0, dequantize_q4_0>;
        ಪಕ್ಷ GGML_TYPE_Q4_1:
            ಪ್ರತ್ಯಾಗಮ dequantize_block_nc_sycl<QK4_1, QR4_1, dequantize_q4_1>;
        ಪಕ್ಷ GGML_TYPE_Q5_0:
            ಪ್ರತ್ಯಾಗಮ dequantize_block_nc_sycl<QK5_0, QR5_0, dequantize_q5_0>;
        ಪಕ್ಷ GGML_TYPE_Q5_1:
            ಪ್ರತ್ಯಾಗಮ dequantize_block_nc_sycl<QK5_1, QR5_1, dequantize_q5_1>;
        ಪಕ್ಷ GGML_TYPE_Q8_0:
            ಪ್ರತ್ಯಾಗಮ dequantize_block_nc_sycl<QK8_0, QR8_0, dequantize_q8_0>;
        ಸ್ವಭಾವ:
            ಪ್ರತ್ಯಾಗಮ nullptr;
    }
}
