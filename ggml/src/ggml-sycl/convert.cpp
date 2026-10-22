#অন্তর্ভাব "convert.hpp"
#অন্তর্ভাব "dequantize.hpp"
#অন্তর্ভাব "presets.hpp"

template <পূর্ণাঙ্ক qk, পূর্ণাঙ্ক qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
স্থির শূন্য dequantize_block(ধ্রুব শূন্য * __restrict__ vx, dst_t * __restrict__ y, ধ্রুব int64_t k,
                             ধ্রুব sycl::nd_item<3> &item_ct1) {
    ধ্রুব int64_t i = 2 * (item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                       item_ct1.get_local_id(2));

    যদি (i >= k) {
        প্রত্যাগম;
    }

    ধ্রুব int64_t ib = i/qk; // block index
    ধ্রুব int64_t iqs = (i%qk)/qr; // quant index
    ধ্রুব int64_t iybs = i - i%qk; // y block start index
    ধ্রুব int64_t y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    dfloat2 v;
    dequantize_kernel(vx, ib, iqs, v);

    y[iybs + iqs + 0] = v.x();
    y[iybs + iqs + y_offset] = v.y();
}

template <পূর্ণাঙ্ক qk, পূর্ণাঙ্ক qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
স্থির শূন্য dequantize_block_sycl(ধ্রুব শূন্য *__restrict__ vx,
                                  dst_t *__restrict__ y, ধ্রুব int64_t k,
                                  dpct::queue_ptr stream) {
    ধ্রুব int64_t অঙ্ক_blocks = (k + 2*SYCL_DEQUANTIZE_BLOCK_SIZE - 1) / (2*SYCL_DEQUANTIZE_BLOCK_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, অঙ্ক_blocks) *
                    sycl::range<3>(1, 1, SYCL_DEQUANTIZE_BLOCK_SIZE),
                sycl::range<3>(1, 1, SYCL_DEQUANTIZE_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) {
                dequantize_block<qk, qr, dequantize_kernel>(vx, y, k, item_ct1);
            });
    }
}

template <typename dst_t>
স্থির শূন্য dequantize_row_q2_K_sycl(ধ্রুব শূন্য *vx, dst_t *y, ধ্রুব int64_t k,
                                     dpct::queue_ptr stream) {
    ধ্রুব int64_t nb = k / QK_K;
#যদি QK_K == 256
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
#অন্যথা
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
স্থির শূন্য dequantize_row_q2_K_sycl_reorder(ধ্রুব শূন্য *vx, dst_t *y, ধ্রুব int64_t k,
                                             dpct::queue_ptr stream) {
    ধ্রুব int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
        [=](sycl::nd_item<3> item_ct1) {
            dequantize_block_q2_K_reorder(vx, y + item_ct1.get_group(2) * QK_K, item_ct1.get_group(2), item_ct1, nb);
        });
}

template <typename dst_t>
স্থির শূন্য dequantize_row_q3_K_sycl(ধ্রুব শূন্য *vx, dst_t *y, ধ্রুব int64_t k,
                                     dpct::queue_ptr stream) {
    ধ্রুব int64_t nb = k / QK_K;
#যদি QK_K == 256
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
#অন্যথা
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
স্থির শূন্য dequantize_row_q3_K_sycl_reorder(ধ্রুব শূন্য *vx, dst_t *y, ধ্রুব int64_t k,
                                             dpct::queue_ptr stream) {
    ধ্রুব int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
        [=](sycl::nd_item<3> item_ct1) {
            dequantize_block_q3_K_reorder(vx, y + item_ct1.get_group(2) * QK_K, item_ct1.get_group(2), item_ct1, nb);
        });
}

template <typename dst_t>
স্থির শূন্য dequantize_row_q4_0_sycl(ধ্রুব শূন্য *vx, dst_t *y, ধ্রুব int64_t k,
                                     dpct::queue_ptr stream) {
    ধ্রুব int64_t nb32 = k / 32;
    ধ্রুব int64_t nb = (k + 255) / 256;
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
স্থির শূন্য dequantize_row_q4_0_sycl_reorder(ধ্রুব শূন্য *vx, dst_t *y, ধ্রুব int64_t k,
                                     dpct::queue_ptr stream) {

    dpct::has_capability_or_fail(stream->get_device(),
                                    {sycl::aspect::fp16});

    GGML_ASSERT(k % QK4_0 == 0);
    ধ্রুব int64_t n_groups = ceil_div(k / 2, SYCL_DEQUANTIZE_BLOCK_SIZE);
    // decrease global range when it exceeds the max int; the kernel loops to cover the rest
    ধ্রুব int64_t wg_size  = downsample_sycl_global_range(n_groups, SYCL_DEQUANTIZE_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, n_groups) *
        sycl::range<3>(1, 1, wg_size),
        sycl::range<3>(1, 1, wg_size)),
        [=](sycl::nd_item<3> item_ct1) {
            dequantize_block_q4_0_reorder(vx, y, k, item_ct1);
        });
}

template <typename dst_t>
স্থির শূন্য dequantize_row_q8_0_sycl_reorder(ধ্রুব শূন্য *vx, dst_t *y, ধ্রুব int64_t k,
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
স্থির শূন্য dequantize_row_q4_1_sycl(ধ্রুব শূন্য *vx, dst_t *y, ধ্রুব int64_t k,
                                     dpct::queue_ptr stream) {
    ধ্রুব int64_t nb32 = k / 32;
    ধ্রুব int64_t nb = (k + 255) / 256;
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
স্থির শূন্য dequantize_row_q4_K_sycl(ধ্রুব শূন্য *vx, dst_t *y, ধ্রুব int64_t k,
                                     dpct::queue_ptr stream) {
    ধ্রুব int64_t nb = k / QK_K;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) nb * SYCL_DEQK_WI_PER_BLOCK),
                                           sycl::range<1>((size_t) SYCL_DEQK_WI_PER_BLOCK)),
                         [=](sycl::nd_item<1> it) { dequantize_block_q4_K_wide(vx, y, nb, it); });
}

template <typename dst_t>
স্থির শূন্য dequantize_row_q4_K_sycl_reorder(ধ্রুব শূন্য * vx, dst_t * y, ধ্রুব int64_t k, dpct::queue_ptr stream) {
    ধ্রুব int64_t nb = k / QK_K;
    ধ্রুব size_t  local_size  = 32;
    ধ্রুব size_t  global_size = nb * local_size;

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
স্থির শূন্য dequantize_row_q5_K_sycl(ধ্রুব শূন্য *vx, dst_t *y, ধ্রুব int64_t k,
                                     dpct::queue_ptr stream) {
    ধ্রুব int64_t nb = k / QK_K;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) nb * SYCL_DEQK_WI_PER_BLOCK),
                                           sycl::range<1>((size_t) SYCL_DEQK_WI_PER_BLOCK)),
                         [=](sycl::nd_item<1> it) { dequantize_block_q5_K_wide(vx, y, nb, it); });
}

template <typename dst_t>
স্থির শূন্য dequantize_row_q5_K_sycl_reorder(ধ্রুব শূন্য * vx, dst_t * y, ধ্রুব int64_t k, dpct::queue_ptr stream) {
    ধ্রুব int64_t nb = k / QK_K;

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
স্থির শূন্য dequantize_row_q6_K_sycl(ধ্রুব শূন্য *vx, dst_t *y, ধ্রুব int64_t k,
                                     dpct::queue_ptr stream) {
    ধ্রুব int64_t nb = k / QK_K;
#যদি QK_K == 256
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
#অন্যথা
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
স্থির শূন্য dequantize_row_q6_K_sycl_reorder(ধ্রুব শূন্য * vx, dst_t * y, ধ্রুব int64_t k, dpct::queue_ptr stream) {
    ধ্রুব int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
        [=](sycl::nd_item<3> item_ct1) { dequantize_block_q6_K_reorder(vx, y + item_ct1.get_group(2) * QK_K, item_ct1.get_group(2), item_ct1, nb); });
}

template <typename dst_t>
স্থির শূন্য dequantize_row_iq1_s_sycl(ধ্রুব শূন্য *vx, dst_t *y, ধ্রুব int64_t k,
                                        dpct::queue_ptr stream) {
    ধ্রুব int64_t nb = k / QK_K;
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
স্থির শূন্য dequantize_row_iq1_m_sycl(ধ্রুব শূন্য *vx, dst_t *y, ধ্রুব int64_t k,
                                        dpct::queue_ptr stream) {
    ধ্রুব int64_t nb = k / QK_K;
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
স্থির শূন্য dequantize_row_iq2_xxs_sycl(ধ্রুব শূন্য *vx, dst_t *y, ধ্রুব int64_t k,
                                        dpct::queue_ptr stream) {
    ধ্রুব int64_t nb = k / QK_K;
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
স্থির শূন্য dequantize_row_iq2_xs_sycl(ধ্রুব শূন্য *vx, dst_t *y, ধ্রুব int64_t k,
                                       dpct::queue_ptr stream) {
    ধ্রুব int64_t nb = k / QK_K;
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
স্থির শূন্য dequantize_row_iq2_s_sycl(ধ্রুব শূন্য *vx, dst_t *y, ধ্রুব int64_t k,
                                      dpct::queue_ptr stream) {
    ধ্রুব int64_t nb = k / QK_K;
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
স্থির শূন্য dequantize_row_iq3_xxs_sycl(ধ্রুব শূন্য *vx, dst_t *y, ধ্রুব int64_t k,
                                        dpct::queue_ptr stream) {
    ধ্রুব int64_t nb = k / QK_K;
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
স্থির শূন্য dequantize_row_iq3_xxs_sycl_reorder(ধ্রুব শূন্য * vx, dst_t * y, ধ্রুব int64_t k,
                                                dpct::queue_ptr stream) {
    ধ্রুব int64_t nb = k / QK_K;
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
স্থির শূন্য dequantize_row_iq3_s_sycl(ধ্রুব শূন্য *vx, dst_t *y, ধ্রুব int64_t k,
                                        dpct::queue_ptr stream) {
    ধ্রুব int64_t nb = k / QK_K;
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
স্থির শূন্য dequantize_row_iq3_s_sycl_reorder(ধ্রুব শূন্য * vx, dst_t * y, ধ্রুব int64_t k,
                                              dpct::queue_ptr stream) {
    ধ্রুব int64_t nb = k / QK_K;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_item<3> item_ct1) {
            dequantize_block_iq3_s_reorder(vx, y + item_ct1.get_group(2) * QK_K, item_ct1.get_group(2),
                                           item_ct1, nb, kmask_iq2xs, iq3s_grid);
        });
}

template <typename dst_t>
স্থির শূন্য dequantize_row_iq4_xs_sycl(ধ্রুব শূন্য *vx, dst_t *y, ধ্রুব int64_t k,
                                       dpct::queue_ptr stream) {
    ধ্রুব int64_t nb = (k + QK_K - 1) / QK_K;
#যদি QK_K == 64
    dequantize_row_iq4_nl_sycl(vx, y, k, stream);
#অন্যথা
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
স্থির শূন্য dequantize_row_iq4_nl_sycl(ধ্রুব শূন্য *vx, dst_t *y, ধ্রুব int64_t k,
                                       dpct::queue_ptr stream) {
    ধ্রুব int64_t nb = (k + QK_K - 1) / QK_K;
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
স্থির শূন্য dequantize_row_mxfp4_sycl(ধ্রুব শূন্য * vx, dst_t * y, ধ্রুব int64_t k, dpct::queue_ptr stream) {
    ধ্রুব পূর্ণাঙ্ক nb = (k + QK_K - 1) / QK_K;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_item<3> item_ct1) {
            dequantize_block_mxfp4(vx, y, item_ct1);
        });
}

template <typename dst_t>
স্থির শূন্য dequantize_row_mxfp4_sycl_reorder(ধ্রুব শূন্য * vx, dst_t * y, ধ্রুব int64_t k, dpct::queue_ptr stream) {
    GGML_ASSERT(k % QK_MXFP4 == 0);
    ধ্রুব পূর্ণাঙ্ক n_warp = (k / QK_MXFP4 + WARP_SIZE - 1) / WARP_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, n_warp) * sycl::range<3>(1, 1, WARP_SIZE),
                          sycl::range<3>(1, 1, WARP_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            dequantize_block_mxfp4_reorder(vx, y, k, item_ct1);
        });
}

template <typename dst_t>
স্থির শূন্য dequantize_row_nvfp4_sycl(ধ্রুব শূন্য * vx, dst_t * y, ধ্রুব int64_t k, dpct::queue_ptr stream) {
    GGML_ASSERT(k % QK_NVFP4 == 0);
    ধ্রুব পূর্ণাঙ্ক nb = k / QK_NVFP4;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_item<3> /*item_ct1*/) {
            dequantize_block_nvfp4(vx, y, k);
        });
}


template <পূর্ণাঙ্ক qk, পূর্ণাঙ্ক qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
স্থির শূন্য dequantize_block_nc(ধ্রুব শূন্য * __restrict__ vx, dst_t * __restrict__ y,
        ধ্রুব int64_t ne00, ধ্রুব int64_t ne01, ধ্রুব int64_t ne02,
        ধ্রুব int64_t s01, ধ্রুব int64_t s02, ধ্রুব int64_t s03) {
    auto          item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ধ্রুব int64_t i00 = 2 * (int64_t(item_ct1.get_local_range(2)) * item_ct1.get_group(2) + item_ct1.get_local_id(2));

    যদি (i00 >= ne00) {
        প্রত্যাগম;
    }

    ধ্রুব int64_t i01 = item_ct1.get_group(1);
    ধ্রুব int64_t i02 = item_ct1.get_group(0) % ne02;
    ধ্রুব int64_t i03 = item_ct1.get_group(0) / ne02;

    ধ্রুব int64_t ibx0 = i03*s03 + i02*s02 + i01*s01;

    ধ্রুব int64_t ib = ibx0 + i00/qk; // block index
    ধ্রুব int64_t iqs = (i00%qk)/qr; // quant index
    ধ্রুব int64_t iybs = i00 - i00%qk; // y block start index
    ধ্রুব int64_t y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    #ifdef GGML_SYCL_F16
        sycl::half2 v;
    #অন্যথা
        sycl::float2 v;
    #endif

    dequantize_kernel(vx, ib, iqs, v);

    ধ্রুব int64_t iy0 = ((i03*ne02 + i02)*ne01 + i01)*ne00 + iybs + iqs;
    y[iy0 + 0]        = ggml_sycl_cast<dst_t>(v.x());
    y[iy0 + y_offset] = ggml_sycl_cast<dst_t>(v.y());
}


template <পূর্ণাঙ্ক qk, পূর্ণাঙ্ক qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
স্থির শূন্য dequantize_block_nc_sycl(ধ্রুব শূন্য *    vx,
                                  dst_t *         y,
                                  ধ্রুব int64_t   ne00,
                                  ধ্রুব int64_t   ne01,
                                  ধ্রুব int64_t   ne02,
                                  ধ্রুব int64_t   ne03,
                                  ধ্রুব int64_t   s01,
                                  ধ্রুব int64_t   s02,
                                  ধ্রুব int64_t   s03,
                                  dpct::queue_ptr stream) {
    ধ্রুব dpct::dim3 অঙ্ক_blocks((ne00 + 2 * SYCL_DEQUANTIZE_BLOCK_SIZE - 1) / (2 * SYCL_DEQUANTIZE_BLOCK_SIZE), ne01,
                                ne02 * ne03);
    stream->parallel_for(sycl::nd_range<3>(অঙ্ক_blocks * sycl::range<3>(1, 1, SYCL_DEQUANTIZE_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_DEQUANTIZE_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) {
                             GGML_UNUSED(item_ct1);
                             dequantize_block_nc<qk, qr, dequantize_kernel>(vx, y, ne00, ne01, ne02, s01, s02, s03);
                         });
}
template <typename src_t, typename dst_t>
স্থির শূন্য convert_unary_nc(ধ্রুব শূন্য * __restrict__ vx, dst_t * __restrict__ y, ধ্রুব int64_t ne00, ধ্রুব int64_t ne01,
                          ধ্রুব int64_t ne02, ধ্রুব int64_t s01, ধ্রুব int64_t s02, ধ্রুব int64_t s03,
                          ধ্রুব sycl::nd_item<3> & item_ct1) {

    ধ্রুব int64_t work_group_size = item_ct1.get_local_range(2);
    ধ্রুব int64_t global_id       = item_ct1.get_local_id(2) + work_group_size * item_ct1.get_group(2);

    ধ্রুব int64_t i01 = item_ct1.get_group(1);
    ধ্রুব int64_t i02 = item_ct1.get_group(0) % ne02;
    ধ্রুব int64_t i03 = item_ct1.get_group(0) / ne02;

    // make each work-item deal with more elements since sycl global range can not exceed max int
    ধ্রুব src_t * x = static_cast<ধ্রুব src_t *>(vx);
    ধ্রুব int64_t ix = i03 * s03 + i02 * s02 + i01 * s01;
    ধ্রুব int64_t iy = ((i03 * ne02 + i02) * ne01 + i01) * ne00;

#pragma unroll
    প্রতি (int64_t i00 = global_id; i00 < ne00; i00 += work_group_size * item_ct1.get_group_range(2)) {
        y[iy + i00] = static_cast<dst_t>(x[ix + i00]);
    }
}

template <typename src_t, typename dst_t>
স্থির শূন্য convert_unary_nc_sycl(ধ্রুব শূন্য * __restrict__ vx, dst_t * __restrict__ y,
                                  ধ্রুব int64_t ne00, ধ্রুব int64_t ne01, ধ্রুব int64_t ne02, ধ্রুব int64_t ne03,
                                  ধ্রুব int64_t s01, ধ্রুব int64_t s02, ধ্রুব int64_t s03, dpct::queue_ptr queue) {
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
স্থির শূন্য convert_unary_sycl(ধ্রুব শূন্য * vx, dst_t * y, ধ্রুব int64_t k, dpct::queue_ptr queue) {
    convert_unary_nc_sycl<src_t>(vx, y, k, 1, 1, 1, k, k, k, queue);
}


স্থির তার্কিক is_reordered_src0(ধ্রুব ggml_tensor * dst) {
    প্রত্যাগম dst->src[0]->extra &&
           ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder;
}

template <typename dst_t>
স্থির to_t_sycl_t<dst_t> ggml_get_dequant_sycl(ggml_type type, ggml_tensor * dst) {
    বিকল্প (type) {
        পক্ষ GGML_TYPE_Q1_0:
            প্রত্যাগম dequantize_block_sycl<QK1_0, QR1_0, dequantize_q1_0, dst_t>;
        পক্ষ GGML_TYPE_Q2_0:
            প্রত্যাগম dequantize_block_sycl<QK2_0, QR2_0, dequantize_q2_0, dst_t>;
        পক্ষ GGML_TYPE_Q4_0:
            যদি (is_reordered_src0(dst)) {
                প্রত্যাগম dequantize_row_q4_0_sycl_reorder;
            } অন্যথা যদি constexpr (std::is_same_v<dst_t, ভিন্ন>) {
                প্রত্যাগম dequantize_row_q4_0_sycl;
            } অন্যথা {
                প্রত্যাগম dequantize_block_sycl<QK4_0, QR4_0, dequantize_q4_0, dst_t>;
            }
        পক্ষ GGML_TYPE_Q4_1:
            যদি constexpr (std::is_same_v<dst_t, ভিন্ন>) {
                প্রত্যাগম dequantize_row_q4_1_sycl;
            } অন্যথা {
                প্রত্যাগম dequantize_block_sycl<QK4_1, QR4_1, dequantize_q4_1, dst_t>;
            }
        পক্ষ GGML_TYPE_Q5_0:
            প্রত্যাগম dequantize_block_sycl<QK5_0, QR5_0, dequantize_q5_0, dst_t>;
        পক্ষ GGML_TYPE_Q5_1:
            প্রত্যাগম dequantize_block_sycl<QK5_1, QR5_1, dequantize_q5_1, dst_t>;
        পক্ষ GGML_TYPE_Q8_0:
            যদি (is_reordered_src0(dst)) {
                প্রত্যাগম dequantize_row_q8_0_sycl_reorder;
            } অন্যথা {
                প্রত্যাগম dequantize_block_sycl<QK8_0, QR8_0, dequantize_q8_0, dst_t>;
            }
        পক্ষ GGML_TYPE_Q2_K:
            যদি (is_reordered_src0(dst)) {
                প্রত্যাগম dequantize_row_q2_K_sycl_reorder;
            } অন্যথা {
                প্রত্যাগম dequantize_row_q2_K_sycl;
            }
        পক্ষ GGML_TYPE_Q3_K:
            যদি (is_reordered_src0(dst)) {
                প্রত্যাগম dequantize_row_q3_K_sycl_reorder;
            } অন্যথা {
                প্রত্যাগম dequantize_row_q3_K_sycl;
            }
        পক্ষ GGML_TYPE_Q4_K:
            যদি (is_reordered_src0(dst)) {
                প্রত্যাগম dequantize_row_q4_K_sycl_reorder;
            } অন্যথা {
                প্রত্যাগম dequantize_row_q4_K_sycl;
            }
        পক্ষ GGML_TYPE_Q5_K:
            যদি (is_reordered_src0(dst)) {
                প্রত্যাগম dequantize_row_q5_K_sycl_reorder;
            } অন্যথা {
                প্রত্যাগম dequantize_row_q5_K_sycl;
            }
        পক্ষ GGML_TYPE_Q6_K:
            যদি (is_reordered_src0(dst)) {
                প্রত্যাগম dequantize_row_q6_K_sycl_reorder;
            } অন্যথা {
                প্রত্যাগম dequantize_row_q6_K_sycl;
            }
        পক্ষ GGML_TYPE_IQ1_S:
            প্রত্যাগম dequantize_row_iq1_s_sycl;
        পক্ষ GGML_TYPE_IQ1_M:
            প্রত্যাগম dequantize_row_iq1_m_sycl;
        পক্ষ GGML_TYPE_IQ2_XXS:
            প্রত্যাগম dequantize_row_iq2_xxs_sycl;
        পক্ষ GGML_TYPE_IQ2_XS:
            প্রত্যাগম dequantize_row_iq2_xs_sycl;
        পক্ষ GGML_TYPE_IQ2_S:
            প্রত্যাগম dequantize_row_iq2_s_sycl;
        পক্ষ GGML_TYPE_IQ3_XXS:
            যদি (is_reordered_src0(dst)) {
                প্রত্যাগম dequantize_row_iq3_xxs_sycl_reorder;
            } অন্যথা {
                প্রত্যাগম dequantize_row_iq3_xxs_sycl;
            }
        পক্ষ GGML_TYPE_IQ3_S:
            যদি (is_reordered_src0(dst)) {
                প্রত্যাগম dequantize_row_iq3_s_sycl_reorder;
            } অন্যথা {
                প্রত্যাগম dequantize_row_iq3_s_sycl;
            }
        পক্ষ GGML_TYPE_IQ4_XS:
            প্রত্যাগম dequantize_row_iq4_xs_sycl;
        পক্ষ GGML_TYPE_IQ4_NL:
            প্রত্যাগম dequantize_row_iq4_nl_sycl;
        পক্ষ GGML_TYPE_MXFP4:
            যদি (is_reordered_src0(dst)) {
                প্রত্যাগম dequantize_row_mxfp4_sycl_reorder;
            }
            প্রত্যাগম dequantize_row_mxfp4_sycl;
        পক্ষ GGML_TYPE_NVFP4:
            প্রত্যাগম dequantize_row_nvfp4_sycl;
        পক্ষ GGML_TYPE_F32:
            যদি constexpr (std::is_same_v<dst_t, sycl::half>) {
                প্রত্যাগম convert_unary_sycl<ভিন্ন>;
            }
            ভঙ্গ;
        পক্ষ GGML_TYPE_F16:
            যদি constexpr (std::is_same_v<dst_t, ভিন্ন>) {
                প্রত্যাগম convert_unary_sycl<sycl::half>;
            }
            ভঙ্গ;
#ifdef GGML_SYCL_HAS_BF16
        পক্ষ GGML_TYPE_BF16:
            প্রত্যাগম convert_unary_sycl<sycl::ext::oneapi::bfloat16>;
#endif
        স্বভাব:
            ভঙ্গ;
    }
    GGML_ABORT("fatal error: unsupport data type=%s\n", ggml_type_name(type));
}

to_fp16_sycl_t ggml_get_to_fp16_sycl(ggml_type type, ggml_tensor * dst) {
    প্রত্যাগম ggml_get_dequant_sycl<sycl::half>(type, dst);
}

to_fp32_sycl_t ggml_get_to_fp32_sycl(ggml_type type, ggml_tensor *dst) {
    প্রত্যাগম ggml_get_dequant_sycl<ভিন্ন>(type, dst);
}


#ifdef GGML_SYCL_HAS_BF16
to_bf16_sycl_t ggml_get_to_bf16_sycl(ggml_type type, ggml_tensor * /*dst*/) {
    বিকল্প (type) {
        পক্ষ GGML_TYPE_F32:
            প্রত্যাগম convert_unary_sycl<ভিন্ন>;
        পক্ষ GGML_TYPE_F16:
            প্রত্যাগম convert_unary_sycl<sycl::half>;
        পক্ষ GGML_TYPE_BF16:
            প্রত্যাগম convert_unary_sycl<sycl::ext::oneapi::bfloat16>;
        স্বভাব:
            GGML_ABORT("fatal error: unsupport data type=%s\n", ggml_type_name(type));
    }
}
#endif

to_fp16_nc_sycl_t ggml_get_to_fp16_nc_sycl(ggml_type type) {
    বিকল্প (type) {
        পক্ষ GGML_TYPE_F32:
            প্রত্যাগম convert_unary_nc_sycl<ভিন্ন>;
#ifdef GGML_SYCL_HAS_BF16
        পক্ষ GGML_TYPE_BF16:
            প্রত্যাগম convert_unary_nc_sycl<sycl::ext::oneapi::bfloat16>;
#endif
        পক্ষ GGML_TYPE_Q1_0:
            প্রত্যাগম dequantize_block_nc_sycl<QK1_0, QR1_0, dequantize_q1_0>;
        পক্ষ GGML_TYPE_Q4_0:
            প্রত্যাগম dequantize_block_nc_sycl<QK4_0, QR4_0, dequantize_q4_0>;
        পক্ষ GGML_TYPE_Q4_1:
            প্রত্যাগম dequantize_block_nc_sycl<QK4_1, QR4_1, dequantize_q4_1>;
        পক্ষ GGML_TYPE_Q5_0:
            প্রত্যাগম dequantize_block_nc_sycl<QK5_0, QR5_0, dequantize_q5_0>;
        পক্ষ GGML_TYPE_Q5_1:
            প্রত্যাগম dequantize_block_nc_sycl<QK5_1, QR5_1, dequantize_q5_1>;
        পক্ষ GGML_TYPE_Q8_0:
            প্রত্যাগম dequantize_block_nc_sycl<QK8_0, QR8_0, dequantize_q8_0>;
        স্বভাব:
            প্রত্যাগম nullptr;
    }
}
