#समावेश "convert.hpp"
#समावेश "dequantize.hpp"
#समावेश "presets.hpp"

ढाँचा <पूर्णांक qk, पूर्णांक qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
अटल व्योम dequantize_block(स्थिर व्योम * __restrict__ vx, dst_t * __restrict__ y, स्थिर int64_t k,
                             स्थिर sycl::nd_item<3> &item_ct1) {
    स्थिर int64_t i = 2 * (item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                       item_ct1.get_local_id(2));

    अगर (i >= k) {
        वापस;
    }

    स्थिर int64_t ib = i/qk; // block index
    स्थिर int64_t iqs = (i%qk)/qr; // quant index
    स्थिर int64_t iybs = i - i%qk; // y block start index
    स्थिर int64_t y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    dfloat2 v;
    dequantize_kernel(vx, ib, iqs, v);

    y[iybs + iqs + 0] = v.x();
    y[iybs + iqs + y_offset] = v.y();
}

ढाँचा <पूर्णांक qk, पूर्णांक qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
अटल व्योम dequantize_block_sycl(स्थिर व्योम *__restrict__ vx,
                                  dst_t *__restrict__ y, स्थिर int64_t k,
                                  dpct::queue_ptr stream) {
    स्थिर int64_t अंक_blocks = (k + 2*SYCL_DEQUANTIZE_BLOCK_आकार - 1) / (2*SYCL_DEQUANTIZE_BLOCK_आकार);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, अंक_blocks) *
                    sycl::range<3>(1, 1, SYCL_DEQUANTIZE_BLOCK_आकार),
                sycl::range<3>(1, 1, SYCL_DEQUANTIZE_BLOCK_आकार)),
            [=](sycl::nd_item<3> item_ct1) {
                dequantize_block<qk, qr, dequantize_kernel>(vx, y, k, item_ct1);
            });
    }
}

ढाँचा <typename dst_t>
अटल व्योम dequantize_row_q2_K_sycl(स्थिर व्योम *vx, dst_t *y, स्थिर int64_t k,
                                     dpct::queue_ptr stream) {
    स्थिर int64_t nb = k / QK_K;
#अगर QK_K == 256
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

#पूर्ण_अगर
}

ढाँचा <typename dst_t>
अटल व्योम dequantize_row_q2_K_sycl_reorder(स्थिर व्योम *vx, dst_t *y, स्थिर int64_t k,
                                             dpct::queue_ptr stream) {
    स्थिर int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
        [=](sycl::nd_item<3> item_ct1) {
            dequantize_block_q2_K_reorder(vx, y + item_ct1.get_group(2) * QK_K, item_ct1.get_group(2), item_ct1, nb);
        });
}

ढाँचा <typename dst_t>
अटल व्योम dequantize_row_q3_K_sycl(स्थिर व्योम *vx, dst_t *y, स्थिर int64_t k,
                                     dpct::queue_ptr stream) {
    स्थिर int64_t nb = k / QK_K;
#अगर QK_K == 256
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
#पूर्ण_अगर
}

ढाँचा <typename dst_t>
अटल व्योम dequantize_row_q3_K_sycl_reorder(स्थिर व्योम *vx, dst_t *y, स्थिर int64_t k,
                                             dpct::queue_ptr stream) {
    स्थिर int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
        [=](sycl::nd_item<3> item_ct1) {
            dequantize_block_q3_K_reorder(vx, y + item_ct1.get_group(2) * QK_K, item_ct1.get_group(2), item_ct1, nb);
        });
}

ढाँचा <typename dst_t>
अटल व्योम dequantize_row_q4_0_sycl(स्थिर व्योम *vx, dst_t *y, स्थिर int64_t k,
                                     dpct::queue_ptr stream) {
    स्थिर int64_t nb32 = k / 32;
    स्थिर int64_t nb = (k + 255) / 256;
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
अटल व्योम dequantize_row_q4_0_sycl_reorder(स्थिर व्योम *vx, dst_t *y, स्थिर int64_t k,
                                     dpct::queue_ptr stream) {

    dpct::has_capability_or_fail(stream->get_device(),
                                    {sycl::aspect::fp16});

    GGML_ASSERT(k % QK4_0 == 0);
    स्थिर int64_t n_groups = ceil_div(k / 2, SYCL_DEQUANTIZE_BLOCK_आकार);
    // decrease global range when it exceeds the max int; the kernel loops to cover the rest
    स्थिर int64_t wg_आकार  = downsample_sycl_global_range(n_groups, SYCL_DEQUANTIZE_BLOCK_आकार);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, n_groups) *
        sycl::range<3>(1, 1, wg_आकार),
        sycl::range<3>(1, 1, wg_आकार)),
        [=](sycl::nd_item<3> item_ct1) {
            dequantize_block_q4_0_reorder(vx, y, k, item_ct1);
        });
}

ढाँचा <typename dst_t>
अटल व्योम dequantize_row_q8_0_sycl_reorder(स्थिर व्योम *vx, dst_t *y, स्थिर int64_t k,
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
अटल व्योम dequantize_row_q4_1_sycl(स्थिर व्योम *vx, dst_t *y, स्थिर int64_t k,
                                     dpct::queue_ptr stream) {
    स्थिर int64_t nb32 = k / 32;
    स्थिर int64_t nb = (k + 255) / 256;
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
अटल व्योम dequantize_row_q4_K_sycl(स्थिर व्योम *vx, dst_t *y, स्थिर int64_t k,
                                     dpct::queue_ptr stream) {
    स्थिर int64_t nb = k / QK_K;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(sycl::nd_range<1>(sycl::range<1>((माप_प्रकार) nb * SYCL_DEQK_WI_PER_BLOCK),
                                           sycl::range<1>((माप_प्रकार) SYCL_DEQK_WI_PER_BLOCK)),
                         [=](sycl::nd_item<1> it) { dequantize_block_q4_K_wide(vx, y, nb, it); });
}

ढाँचा <typename dst_t>
अटल व्योम dequantize_row_q4_K_sycl_reorder(स्थिर व्योम * vx, dst_t * y, स्थिर int64_t k, dpct::queue_ptr stream) {
    स्थिर int64_t nb = k / QK_K;
    स्थिर माप_प्रकार  local_आकार  = 32;
    स्थिर माप_प्रकार  global_आकार = nb * local_आकार;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint8_t, 1> scale_local_acc(sycl::range<1>(12), cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_आकार), sycl::range<1>(local_आकार)),
                         [=](sycl::nd_item<1> item_ct1) {
                             dequantize_block_q4_K_reorder(vx, y + item_ct1.get_group(0) * QK_K, item_ct1.get_group(0), get_pointer(scale_local_acc), item_ct1, nb);
                         });
    });
}

ढाँचा <typename dst_t>
अटल व्योम dequantize_row_q5_K_sycl(स्थिर व्योम *vx, dst_t *y, स्थिर int64_t k,
                                     dpct::queue_ptr stream) {
    स्थिर int64_t nb = k / QK_K;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(sycl::nd_range<1>(sycl::range<1>((माप_प्रकार) nb * SYCL_DEQK_WI_PER_BLOCK),
                                           sycl::range<1>((माप_प्रकार) SYCL_DEQK_WI_PER_BLOCK)),
                         [=](sycl::nd_item<1> it) { dequantize_block_q5_K_wide(vx, y, nb, it); });
}

ढाँचा <typename dst_t>
अटल व्योम dequantize_row_q5_K_sycl_reorder(स्थिर व्योम * vx, dst_t * y, स्थिर int64_t k, dpct::queue_ptr stream) {
    स्थिर int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint8_t, 1> scale_local_acc(sycl::range<1>(K_SCALE_आकार), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
            [=](sycl::nd_item<3> item_ct1) {
                dequantize_block_q5_K_reorder(vx, y + item_ct1.get_group(2) * QK_K, item_ct1.get_group(2), get_pointer(scale_local_acc), item_ct1, nb);
            });
    });
}

ढाँचा <typename dst_t>
अटल व्योम dequantize_row_q6_K_sycl(स्थिर व्योम *vx, dst_t *y, स्थिर int64_t k,
                                     dpct::queue_ptr stream) {
    स्थिर int64_t nb = k / QK_K;
#अगर QK_K == 256
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

#पूर्ण_अगर
}

ढाँचा <typename dst_t>
अटल व्योम dequantize_row_q6_K_sycl_reorder(स्थिर व्योम * vx, dst_t * y, स्थिर int64_t k, dpct::queue_ptr stream) {
    स्थिर int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
        [=](sycl::nd_item<3> item_ct1) { dequantize_block_q6_K_reorder(vx, y + item_ct1.get_group(2) * QK_K, item_ct1.get_group(2), item_ct1, nb); });
}

ढाँचा <typename dst_t>
अटल व्योम dequantize_row_iq1_s_sycl(स्थिर व्योम *vx, dst_t *y, स्थिर int64_t k,
                                        dpct::queue_ptr stream) {
    स्थिर int64_t nb = k / QK_K;
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
अटल व्योम dequantize_row_iq1_m_sycl(स्थिर व्योम *vx, dst_t *y, स्थिर int64_t k,
                                        dpct::queue_ptr stream) {
    स्थिर int64_t nb = k / QK_K;
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
अटल व्योम dequantize_row_iq2_xxs_sycl(स्थिर व्योम *vx, dst_t *y, स्थिर int64_t k,
                                        dpct::queue_ptr stream) {
    स्थिर int64_t nb = k / QK_K;
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
अटल व्योम dequantize_row_iq2_xs_sycl(स्थिर व्योम *vx, dst_t *y, स्थिर int64_t k,
                                       dpct::queue_ptr stream) {
    स्थिर int64_t nb = k / QK_K;
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
अटल व्योम dequantize_row_iq2_s_sycl(स्थिर व्योम *vx, dst_t *y, स्थिर int64_t k,
                                      dpct::queue_ptr stream) {
    स्थिर int64_t nb = k / QK_K;
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
अटल व्योम dequantize_row_iq3_xxs_sycl(स्थिर व्योम *vx, dst_t *y, स्थिर int64_t k,
                                        dpct::queue_ptr stream) {
    स्थिर int64_t nb = k / QK_K;
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
अटल व्योम dequantize_row_iq3_xxs_sycl_reorder(स्थिर व्योम * vx, dst_t * y, स्थिर int64_t k,
                                                dpct::queue_ptr stream) {
    स्थिर int64_t nb = k / QK_K;
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
अटल व्योम dequantize_row_iq3_s_sycl(स्थिर व्योम *vx, dst_t *y, स्थिर int64_t k,
                                        dpct::queue_ptr stream) {
    स्थिर int64_t nb = k / QK_K;
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
अटल व्योम dequantize_row_iq3_s_sycl_reorder(स्थिर व्योम * vx, dst_t * y, स्थिर int64_t k,
                                              dpct::queue_ptr stream) {
    स्थिर int64_t nb = k / QK_K;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_item<3> item_ct1) {
            dequantize_block_iq3_s_reorder(vx, y + item_ct1.get_group(2) * QK_K, item_ct1.get_group(2),
                                           item_ct1, nb, kmask_iq2xs, iq3s_grid);
        });
}

ढाँचा <typename dst_t>
अटल व्योम dequantize_row_iq4_xs_sycl(स्थिर व्योम *vx, dst_t *y, स्थिर int64_t k,
                                       dpct::queue_ptr stream) {
    स्थिर int64_t nb = (k + QK_K - 1) / QK_K;
#अगर QK_K == 64
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
#पूर्ण_अगर
}

ढाँचा <typename dst_t>
अटल व्योम dequantize_row_iq4_nl_sycl(स्थिर व्योम *vx, dst_t *y, स्थिर int64_t k,
                                       dpct::queue_ptr stream) {
    स्थिर int64_t nb = (k + QK_K - 1) / QK_K;
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
अटल व्योम dequantize_row_mxfp4_sycl(स्थिर व्योम * vx, dst_t * y, स्थिर int64_t k, dpct::queue_ptr stream) {
    स्थिर पूर्णांक nb = (k + QK_K - 1) / QK_K;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_item<3> item_ct1) {
            dequantize_block_mxfp4(vx, y, item_ct1);
        });
}

ढाँचा <typename dst_t>
अटल व्योम dequantize_row_mxfp4_sycl_reorder(स्थिर व्योम * vx, dst_t * y, स्थिर int64_t k, dpct::queue_ptr stream) {
    GGML_ASSERT(k % QK_MXFP4 == 0);
    स्थिर पूर्णांक n_warp = (k / QK_MXFP4 + WARP_आकार - 1) / WARP_आकार;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, n_warp) * sycl::range<3>(1, 1, WARP_आकार),
                          sycl::range<3>(1, 1, WARP_आकार)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
            dequantize_block_mxfp4_reorder(vx, y, k, item_ct1);
        });
}

ढाँचा <typename dst_t>
अटल व्योम dequantize_row_nvfp4_sycl(स्थिर व्योम * vx, dst_t * y, स्थिर int64_t k, dpct::queue_ptr stream) {
    GGML_ASSERT(k % QK_NVFP4 == 0);
    स्थिर पूर्णांक nb = k / QK_NVFP4;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_item<3> /*item_ct1*/) {
            dequantize_block_nvfp4(vx, y, k);
        });
}


ढाँचा <पूर्णांक qk, पूर्णांक qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
अटल व्योम dequantize_block_nc(स्थिर व्योम * __restrict__ vx, dst_t * __restrict__ y,
        स्थिर int64_t ne00, स्थिर int64_t ne01, स्थिर int64_t ne02,
        स्थिर int64_t s01, स्थिर int64_t s02, स्थिर int64_t s03) {
    स्वतः          item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    स्थिर int64_t i00 = 2 * (int64_t(item_ct1.get_local_range(2)) * item_ct1.get_group(2) + item_ct1.get_local_id(2));

    अगर (i00 >= ne00) {
        वापस;
    }

    स्थिर int64_t i01 = item_ct1.get_group(1);
    स्थिर int64_t i02 = item_ct1.get_group(0) % ne02;
    स्थिर int64_t i03 = item_ct1.get_group(0) / ne02;

    स्थिर int64_t ibx0 = i03*s03 + i02*s02 + i01*s01;

    स्थिर int64_t ib = ibx0 + i00/qk; // block index
    स्थिर int64_t iqs = (i00%qk)/qr; // quant index
    स्थिर int64_t iybs = i00 - i00%qk; // y block start index
    स्थिर int64_t y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    #अगर_घोषित GGML_SYCL_F16
        sycl::half2 v;
    #अन्यथा
        sycl::float2 v;
    #पूर्ण_अगर

    dequantize_kernel(vx, ib, iqs, v);

    स्थिर int64_t iy0 = ((i03*ne02 + i02)*ne01 + i01)*ne00 + iybs + iqs;
    y[iy0 + 0]        = ggml_sycl_cast<dst_t>(v.x());
    y[iy0 + y_offset] = ggml_sycl_cast<dst_t>(v.y());
}


ढाँचा <पूर्णांक qk, पूर्णांक qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
अटल व्योम dequantize_block_nc_sycl(स्थिर व्योम *    vx,
                                  dst_t *         y,
                                  स्थिर int64_t   ne00,
                                  स्थिर int64_t   ne01,
                                  स्थिर int64_t   ne02,
                                  स्थिर int64_t   ne03,
                                  स्थिर int64_t   s01,
                                  स्थिर int64_t   s02,
                                  स्थिर int64_t   s03,
                                  dpct::queue_ptr stream) {
    स्थिर dpct::dim3 अंक_blocks((ne00 + 2 * SYCL_DEQUANTIZE_BLOCK_आकार - 1) / (2 * SYCL_DEQUANTIZE_BLOCK_आकार), ne01,
                                ne02 * ne03);
    stream->parallel_for(sycl::nd_range<3>(अंक_blocks * sycl::range<3>(1, 1, SYCL_DEQUANTIZE_BLOCK_आकार),
                                           sycl::range<3>(1, 1, SYCL_DEQUANTIZE_BLOCK_आकार)),
                         [=](sycl::nd_item<3> item_ct1) {
                             GGML_UNUSED(item_ct1);
                             dequantize_block_nc<qk, qr, dequantize_kernel>(vx, y, ne00, ne01, ne02, s01, s02, s03);
                         });
}
ढाँचा <typename src_t, typename dst_t>
अटल व्योम convert_unary_nc(स्थिर व्योम * __restrict__ vx, dst_t * __restrict__ y, स्थिर int64_t ne00, स्थिर int64_t ne01,
                          स्थिर int64_t ne02, स्थिर int64_t s01, स्थिर int64_t s02, स्थिर int64_t s03,
                          स्थिर sycl::nd_item<3> & item_ct1) {

    स्थिर int64_t work_group_आकार = item_ct1.get_local_range(2);
    स्थिर int64_t global_id       = item_ct1.get_local_id(2) + work_group_आकार * item_ct1.get_group(2);

    स्थिर int64_t i01 = item_ct1.get_group(1);
    स्थिर int64_t i02 = item_ct1.get_group(0) % ne02;
    स्थिर int64_t i03 = item_ct1.get_group(0) / ne02;

    // make each work-item deal with more elements since sycl global range can not exceed max int
    स्थिर src_t * x = static_cast<स्थिर src_t *>(vx);
    स्थिर int64_t ix = i03 * s03 + i02 * s02 + i01 * s01;
    स्थिर int64_t iy = ((i03 * ne02 + i02) * ne01 + i01) * ne00;

#आशय unroll
    क्रम (int64_t i00 = global_id; i00 < ne00; i00 += work_group_आकार * item_ct1.get_group_range(2)) {
        y[iy + i00] = static_cast<dst_t>(x[ix + i00]);
    }
}

ढाँचा <typename src_t, typename dst_t>
अटल व्योम convert_unary_nc_sycl(स्थिर व्योम * __restrict__ vx, dst_t * __restrict__ y,
                                  स्थिर int64_t ne00, स्थिर int64_t ne01, स्थिर int64_t ne02, स्थिर int64_t ne03,
                                  स्थिर int64_t s01, स्थिर int64_t s02, स्थिर int64_t s03, dpct::queue_ptr queue) {
    dpct::has_capability_or_fail(queue->get_device(), { sycl::aspect::fp16 });

    sycl::range<3> global_आकार(ne02 * ne03, ne01, ceil_div(ne00, SYCL_DEQUANTIZE_BLOCK_आकार));

    // decrease global range when it exceeds the max int
    // TODO: Downsample logic is separated from the kernel, a rewrite is desirable
    int64_t        downsized_workgroup = downsample_sycl_global_range(global_आकार[0], SYCL_DEQUANTIZE_BLOCK_आकार);
    sycl::range<3> workgroup_आकार(1, 1, downsized_workgroup);

    queue->parallel_for(sycl::nd_range<3>(global_आकार * workgroup_आकार, workgroup_आकार), [=](sycl::nd_item<3> item_ct1) {
        convert_unary_nc<src_t>(vx, y, ne00, ne01, ne02, s01, s02, s03, item_ct1);
    });
}

ढाँचा <typename src_t, typename dst_t>
अटल व्योम convert_unary_sycl(स्थिर व्योम * vx, dst_t * y, स्थिर int64_t k, dpct::queue_ptr queue) {
    convert_unary_nc_sycl<src_t>(vx, y, k, 1, 1, 1, k, k, k, queue);
}


अटल bool is_reordered_src0(स्थिर ggml_tensor * dst) {
    वापस dst->src[0]->extra &&
           ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder;
}

ढाँचा <typename dst_t>
अटल to_t_sycl_t<dst_t> ggml_get_dequant_sycl(ggml_type type, ggml_tensor * dst) {
    चयन (type) {
        हाल GGML_TYPE_Q1_0:
            वापस dequantize_block_sycl<QK1_0, QR1_0, dequantize_q1_0, dst_t>;
        हाल GGML_TYPE_Q2_0:
            वापस dequantize_block_sycl<QK2_0, QR2_0, dequantize_q2_0, dst_t>;
        हाल GGML_TYPE_Q4_0:
            अगर (is_reordered_src0(dst)) {
                वापस dequantize_row_q4_0_sycl_reorder;
            } अन्यथा अगर constexpr (std::is_same_v<dst_t, भग्न>) {
                वापस dequantize_row_q4_0_sycl;
            } अन्यथा {
                वापस dequantize_block_sycl<QK4_0, QR4_0, dequantize_q4_0, dst_t>;
            }
        हाल GGML_TYPE_Q4_1:
            अगर constexpr (std::is_same_v<dst_t, भग्न>) {
                वापस dequantize_row_q4_1_sycl;
            } अन्यथा {
                वापस dequantize_block_sycl<QK4_1, QR4_1, dequantize_q4_1, dst_t>;
            }
        हाल GGML_TYPE_Q5_0:
            वापस dequantize_block_sycl<QK5_0, QR5_0, dequantize_q5_0, dst_t>;
        हाल GGML_TYPE_Q5_1:
            वापस dequantize_block_sycl<QK5_1, QR5_1, dequantize_q5_1, dst_t>;
        हाल GGML_TYPE_Q8_0:
            अगर (is_reordered_src0(dst)) {
                वापस dequantize_row_q8_0_sycl_reorder;
            } अन्यथा {
                वापस dequantize_block_sycl<QK8_0, QR8_0, dequantize_q8_0, dst_t>;
            }
        हाल GGML_TYPE_Q2_K:
            अगर (is_reordered_src0(dst)) {
                वापस dequantize_row_q2_K_sycl_reorder;
            } अन्यथा {
                वापस dequantize_row_q2_K_sycl;
            }
        हाल GGML_TYPE_Q3_K:
            अगर (is_reordered_src0(dst)) {
                वापस dequantize_row_q3_K_sycl_reorder;
            } अन्यथा {
                वापस dequantize_row_q3_K_sycl;
            }
        हाल GGML_TYPE_Q4_K:
            अगर (is_reordered_src0(dst)) {
                वापस dequantize_row_q4_K_sycl_reorder;
            } अन्यथा {
                वापस dequantize_row_q4_K_sycl;
            }
        हाल GGML_TYPE_Q5_K:
            अगर (is_reordered_src0(dst)) {
                वापस dequantize_row_q5_K_sycl_reorder;
            } अन्यथा {
                वापस dequantize_row_q5_K_sycl;
            }
        हाल GGML_TYPE_Q6_K:
            अगर (is_reordered_src0(dst)) {
                वापस dequantize_row_q6_K_sycl_reorder;
            } अन्यथा {
                वापस dequantize_row_q6_K_sycl;
            }
        हाल GGML_TYPE_IQ1_S:
            वापस dequantize_row_iq1_s_sycl;
        हाल GGML_TYPE_IQ1_M:
            वापस dequantize_row_iq1_m_sycl;
        हाल GGML_TYPE_IQ2_XXS:
            वापस dequantize_row_iq2_xxs_sycl;
        हाल GGML_TYPE_IQ2_XS:
            वापस dequantize_row_iq2_xs_sycl;
        हाल GGML_TYPE_IQ2_S:
            वापस dequantize_row_iq2_s_sycl;
        हाल GGML_TYPE_IQ3_XXS:
            अगर (is_reordered_src0(dst)) {
                वापस dequantize_row_iq3_xxs_sycl_reorder;
            } अन्यथा {
                वापस dequantize_row_iq3_xxs_sycl;
            }
        हाल GGML_TYPE_IQ3_S:
            अगर (is_reordered_src0(dst)) {
                वापस dequantize_row_iq3_s_sycl_reorder;
            } अन्यथा {
                वापस dequantize_row_iq3_s_sycl;
            }
        हाल GGML_TYPE_IQ4_XS:
            वापस dequantize_row_iq4_xs_sycl;
        हाल GGML_TYPE_IQ4_NL:
            वापस dequantize_row_iq4_nl_sycl;
        हाल GGML_TYPE_MXFP4:
            अगर (is_reordered_src0(dst)) {
                वापस dequantize_row_mxfp4_sycl_reorder;
            }
            वापस dequantize_row_mxfp4_sycl;
        हाल GGML_TYPE_NVFP4:
            वापस dequantize_row_nvfp4_sycl;
        हाल GGML_TYPE_F32:
            अगर constexpr (std::is_same_v<dst_t, sycl::half>) {
                वापस convert_unary_sycl<भग्न>;
            }
            अवरोध;
        हाल GGML_TYPE_F16:
            अगर constexpr (std::is_same_v<dst_t, भग्न>) {
                वापस convert_unary_sycl<sycl::half>;
            }
            अवरोध;
#अगर_घोषित GGML_SYCL_HAS_BF16
        हाल GGML_TYPE_BF16:
            वापस convert_unary_sycl<sycl::ext::oneapi::bfloat16>;
#पूर्ण_अगर
        शेष:
            अवरोध;
    }
    GGML_ABORT("fatal error: unsupport data type=%s\n", ggml_type_name(type));
}

to_fp16_sycl_t ggml_get_to_fp16_sycl(ggml_type type, ggml_tensor * dst) {
    वापस ggml_get_dequant_sycl<sycl::half>(type, dst);
}

to_fp32_sycl_t ggml_get_to_fp32_sycl(ggml_type type, ggml_tensor *dst) {
    वापस ggml_get_dequant_sycl<भग्न>(type, dst);
}


#अगर_घोषित GGML_SYCL_HAS_BF16
to_bf16_sycl_t ggml_get_to_bf16_sycl(ggml_type type, ggml_tensor * /*dst*/) {
    चयन (type) {
        हाल GGML_TYPE_F32:
            वापस convert_unary_sycl<भग्न>;
        हाल GGML_TYPE_F16:
            वापस convert_unary_sycl<sycl::half>;
        हाल GGML_TYPE_BF16:
            वापस convert_unary_sycl<sycl::ext::oneapi::bfloat16>;
        शेष:
            GGML_ABORT("fatal error: unsupport data type=%s\n", ggml_type_name(type));
    }
}
#पूर्ण_अगर

to_fp16_nc_sycl_t ggml_get_to_fp16_nc_sycl(ggml_type type) {
    चयन (type) {
        हाल GGML_TYPE_F32:
            वापस convert_unary_nc_sycl<भग्न>;
#अगर_घोषित GGML_SYCL_HAS_BF16
        हाल GGML_TYPE_BF16:
            वापस convert_unary_nc_sycl<sycl::ext::oneapi::bfloat16>;
#पूर्ण_अगर
        हाल GGML_TYPE_Q1_0:
            वापस dequantize_block_nc_sycl<QK1_0, QR1_0, dequantize_q1_0>;
        हाल GGML_TYPE_Q4_0:
            वापस dequantize_block_nc_sycl<QK4_0, QR4_0, dequantize_q4_0>;
        हाल GGML_TYPE_Q4_1:
            वापस dequantize_block_nc_sycl<QK4_1, QR4_1, dequantize_q4_1>;
        हाल GGML_TYPE_Q5_0:
            वापस dequantize_block_nc_sycl<QK5_0, QR5_0, dequantize_q5_0>;
        हाल GGML_TYPE_Q5_1:
            वापस dequantize_block_nc_sycl<QK5_1, QR5_1, dequantize_q5_1>;
        हाल GGML_TYPE_Q8_0:
            वापस dequantize_block_nc_sycl<QK8_0, QR8_0, dequantize_q8_0>;
        शेष:
            वापस nullptr;
    }
}
