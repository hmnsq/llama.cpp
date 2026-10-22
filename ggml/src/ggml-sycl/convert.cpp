#ضمن "convert.hpp"
#ضمن "dequantize.hpp"
#ضمن "presets.hpp"

سانچا <صحيح qk, صحيح qr, dequantize_kernel_t dequantize_kernel, قسم_ناں dst_t>
ساكن فراغ dequantize_بلاک(ثابت فراغ * __restrict__ vx, dst_t * __restrict__ y, ثابت int64_t k,
                             ثابت sycl::nd_شے<3> &شے_ct1) {
    ثابت int64_t i = 2 * (شے_ct1.get_local_range(2) * شے_ct1.get_گروہ(2) +
                       شے_ct1.get_local_id(2));

    إذا (i >= k) {
        أرجع;
    }

    ثابت int64_t ib = i/qk; // block index
    ثابت int64_t iqs = (i%qk)/qr; // quant index
    ثابت int64_t iybs = i - i%qk; // y block start index
    ثابت int64_t y_وتھ = qr == 1 ? 1 : qk/2;

    // dequantize
    dfloat2 v;
    dequantize_kernel(vx, ib, iqs, v);

    y[iybs + iqs + 0] = v.x();
    y[iybs + iqs + y_وتھ] = v.y();
}

سانچا <صحيح qk, صحيح qr, dequantize_kernel_t dequantize_kernel, قسم_ناں dst_t>
ساكن فراغ dequantize_بلاک_sycl(ثابت فراغ *__restrict__ vx,
                                  dst_t *__restrict__ y, ثابت int64_t k,
                                  dpct::قطار_ptr stream) {
    ثابت int64_t عدد_بلاکاں = (k + 2*SYCL_DEQUANTIZE_بلاک_ناپ - 1) / (2*SYCL_DEQUANTIZE_بلاک_ناپ);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, عدد_بلاکاں) *
                    sycl::range<3>(1, 1, SYCL_DEQUANTIZE_بلاک_ناپ),
                sycl::range<3>(1, 1, SYCL_DEQUANTIZE_بلاک_ناپ)),
            [=](sycl::nd_شے<3> شے_ct1) {
                dequantize_بلاک<qk, qr, dequantize_kernel>(vx, y, k, شے_ct1);
            });
    }
}

سانچا <قسم_ناں dst_t>
ساكن فراغ dequantize_صف_q2_K_sycl(ثابت فراغ *vx, dst_t *y, ثابت int64_t k,
                                     dpct::قطار_ptr stream) {
    ثابت int64_t nb = k / QK_K;
#إذا QK_K == 256
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 64),
                                               sycl::range<3>(1, 1, 64)),
                             [=](sycl::nd_شے<3> شے_ct1) {
                                 dequantize_بلاک_q2_K(vx, y, شے_ct1);
                             });
    }
#وإلا
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_شے<3> شے_ct1) {
                                 dequantize_بلاک_q2_K(vx, y, شے_ct1);
                             });
    }

#جے_انت
}

سانچا <قسم_ناں dst_t>
ساكن فراغ dequantize_صف_q2_K_sycl_reorder(ثابت فراغ *vx, dst_t *y, ثابت int64_t k,
                                             dpct::قطار_ptr stream) {
    ثابت int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
        [=](sycl::nd_شے<3> شے_ct1) {
            dequantize_بلاک_q2_K_reorder(vx, y + شے_ct1.get_گروہ(2) * QK_K, شے_ct1.get_گروہ(2), شے_ct1, nb);
        });
}

سانچا <قسم_ناں dst_t>
ساكن فراغ dequantize_صف_q3_K_sycl(ثابت فراغ *vx, dst_t *y, ثابت int64_t k,
                                     dpct::قطار_ptr stream) {
    ثابت int64_t nb = k / QK_K;
#إذا QK_K == 256
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 64),
                                               sycl::range<3>(1, 1, 64)),
                             [=](sycl::nd_شے<3> شے_ct1) {
                                 dequantize_بلاک_q3_K(vx, y, شے_ct1);
                             });
    }
#وإلا
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_شے<3> شے_ct1) {
                                 dequantize_بلاک_q3_K(vx, y, شے_ct1);
                             });
    }
#جے_انت
}

سانچا <قسم_ناں dst_t>
ساكن فراغ dequantize_صف_q3_K_sycl_reorder(ثابت فراغ *vx, dst_t *y, ثابت int64_t k,
                                             dpct::قطار_ptr stream) {
    ثابت int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
        [=](sycl::nd_شے<3> شے_ct1) {
            dequantize_بلاک_q3_K_reorder(vx, y + شے_ct1.get_گروہ(2) * QK_K, شے_ct1.get_گروہ(2), شے_ct1, nb);
        });
}

سانچا <قسم_ناں dst_t>
ساكن فراغ dequantize_صف_q4_0_sycl(ثابت فراغ *vx, dst_t *y, ثابت int64_t k,
                                     dpct::قطار_ptr stream) {
    ثابت int64_t nb32 = k / 32;
    ثابت int64_t nb = (k + 255) / 256;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_شے<3> شے_ct1) {
                                 dequantize_بلاک_q4_0(vx, y, nb32, شے_ct1);
                             });
    }
}

سانچا <قسم_ناں dst_t>
ساكن فراغ dequantize_صف_q4_0_sycl_reorder(ثابت فراغ *vx, dst_t *y, ثابت int64_t k,
                                     dpct::قطار_ptr stream) {

    dpct::has_capability_or_fail(stream->get_device(),
                                    {sycl::aspect::fp16});

    GGML_ASSERT(k % QK4_0 == 0);
    ثابت int64_t n_groups = ceil_div(k / 2, SYCL_DEQUANTIZE_بلاک_ناپ);
    // decrease global range when it exceeds the max int; the kernel loops to cover the rest
    ثابت int64_t wg_ناپ  = downsample_sycl_global_range(n_groups, SYCL_DEQUANTIZE_بلاک_ناپ);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, n_groups) *
        sycl::range<3>(1, 1, wg_ناپ),
        sycl::range<3>(1, 1, wg_ناپ)),
        [=](sycl::nd_شے<3> شے_ct1) {
            dequantize_بلاک_q4_0_reorder(vx, y, k, شے_ct1);
        });
}

سانچا <قسم_ناں dst_t>
ساكن فراغ dequantize_صف_q8_0_sycl_reorder(ثابت فراغ *vx, dst_t *y, ثابت int64_t k,
                                     dpct::قطار_ptr stream) {

    dpct::has_capability_or_fail(stream->get_device(),
                                    {sycl::aspect::fp16});

    // flat range<1>, two pairs (4 elements) per item: no nd_range/group math, one vectorized store.
    // k % QK8_0 == 0 implies k % 4 == 0, so the wide launch covers the row exactly (no tail).
    GGML_ASSERT(k % QK8_0 == 0);
    stream->parallel_for(sycl::range<1>((size_t) (k / 4)),
        [=](sycl::id<1> idx) {
            dequantize_بلاک_q8_0_reorder_wide(vx, y, k, idx);
        });
}

سانچا <قسم_ناں dst_t>
ساكن فراغ dequantize_صف_q4_1_sycl(ثابت فراغ *vx, dst_t *y, ثابت int64_t k,
                                     dpct::قطار_ptr stream) {
    ثابت int64_t nb32 = k / 32;
    ثابت int64_t nb = (k + 255) / 256;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_شے<3> شے_ct1) {
                                 dequantize_بلاک_q4_1(vx, y, nb32, شے_ct1);
                             });
    }
}


سانچا <قسم_ناں dst_t>
ساكن فراغ dequantize_صف_q4_K_sycl(ثابت فراغ *vx, dst_t *y, ثابت int64_t k,
                                     dpct::قطار_ptr stream) {
    ثابت int64_t nb = k / QK_K;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) nb * SYCL_DEQK_WI_فی_بلاک),
                                           sycl::range<1>((size_t) SYCL_DEQK_WI_فی_بلاک)),
                         [=](sycl::nd_شے<1> it) { dequantize_بلاک_q4_K_wide(vx, y, nb, it); });
}

سانچا <قسم_ناں dst_t>
ساكن فراغ dequantize_صف_q4_K_sycl_reorder(ثابت فراغ * vx, dst_t * y, ثابت int64_t k, dpct::قطار_ptr stream) {
    ثابت int64_t nb = k / QK_K;
    ثابت size_t  local_ناپ  = 32;
    ثابت size_t  global_ناپ = nb * local_ناپ;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint8_t, 1> scale_local_acc(sycl::range<1>(12), cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_ناپ), sycl::range<1>(local_ناپ)),
                         [=](sycl::nd_شے<1> شے_ct1) {
                             dequantize_بلاک_q4_K_reorder(vx, y + شے_ct1.get_گروہ(0) * QK_K, شے_ct1.get_گروہ(0), get_اشارہ_گر(scale_local_acc), شے_ct1, nb);
                         });
    });
}

سانچا <قسم_ناں dst_t>
ساكن فراغ dequantize_صف_q5_K_sycl(ثابت فراغ *vx, dst_t *y, ثابت int64_t k,
                                     dpct::قطار_ptr stream) {
    ثابت int64_t nb = k / QK_K;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) nb * SYCL_DEQK_WI_فی_بلاک),
                                           sycl::range<1>((size_t) SYCL_DEQK_WI_فی_بلاک)),
                         [=](sycl::nd_شے<1> it) { dequantize_بلاک_q5_K_wide(vx, y, nb, it); });
}

سانچا <قسم_ناں dst_t>
ساكن فراغ dequantize_صف_q5_K_sycl_reorder(ثابت فراغ * vx, dst_t * y, ثابت int64_t k, dpct::قطار_ptr stream) {
    ثابت int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint8_t, 1> scale_local_acc(sycl::range<1>(K_SCALE_ناپ), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
            [=](sycl::nd_شے<3> شے_ct1) {
                dequantize_بلاک_q5_K_reorder(vx, y + شے_ct1.get_گروہ(2) * QK_K, شے_ct1.get_گروہ(2), get_اشارہ_گر(scale_local_acc), شے_ct1, nb);
            });
    });
}

سانچا <قسم_ناں dst_t>
ساكن فراغ dequantize_صف_q6_K_sycl(ثابت فراغ *vx, dst_t *y, ثابت int64_t k,
                                     dpct::قطار_ptr stream) {
    ثابت int64_t nb = k / QK_K;
#إذا QK_K == 256
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 64),
                                               sycl::range<3>(1, 1, 64)),
                             [=](sycl::nd_شے<3> شے_ct1) {
                                 dequantize_بلاک_q6_K(vx, y, شے_ct1);
                             });
    }
#وإلا
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_شے<3> شے_ct1) {
                                 dequantize_بلاک_q6_K(vx, y, شے_ct1);
                             });
    }

#جے_انت
}

سانچا <قسم_ناں dst_t>
ساكن فراغ dequantize_صف_q6_K_sycl_reorder(ثابت فراغ * vx, dst_t * y, ثابت int64_t k, dpct::قطار_ptr stream) {
    ثابت int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
        [=](sycl::nd_شے<3> شے_ct1) { dequantize_بلاک_q6_K_reorder(vx, y + شے_ct1.get_گروہ(2) * QK_K, شے_ct1.get_گروہ(2), شے_ct1, nb); });
}

سانچا <قسم_ناں dst_t>
ساكن فراغ dequantize_صف_iq1_s_sycl(ثابت فراغ *vx, dst_t *y, ثابت int64_t k,
                                        dpct::قطار_ptr stream) {
    ثابت int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_شے<3> شے_ct1) {
                                 dequantize_بلاک_iq1_s(
                                     vx, y, شے_ct1, iq1s_grid_gpu
                                     );
                             });
        });
    }
}

سانچا <قسم_ناں dst_t>
ساكن فراغ dequantize_صف_iq1_m_sycl(ثابت فراغ *vx, dst_t *y, ثابت int64_t k,
                                        dpct::قطار_ptr stream) {
    ثابت int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_شے<3> شے_ct1) {
                                 dequantize_بلاک_iq1_m(
                                     vx, y, شے_ct1, iq1s_grid_gpu
                                     );
                             });
        });
    }
}

سانچا <قسم_ناں dst_t>
ساكن فراغ dequantize_صف_iq2_xxs_sycl(ثابت فراغ *vx, dst_t *y, ثابت int64_t k,
                                        dpct::قطار_ptr stream) {
    ثابت int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_شے<3> شے_ct1) {
                                 dequantize_بلاک_iq2_xxs(
                                     vx, y, شے_ct1, iq2xxs_grid,
                                     ksigns_iq2xs, kmask_iq2xs);
                             });
        });
    }
}

سانچا <قسم_ناں dst_t>
ساكن فراغ dequantize_صف_iq2_xs_sycl(ثابت فراغ *vx, dst_t *y, ثابت int64_t k,
                                       dpct::قطار_ptr stream) {
    ثابت int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_شے<3> شے_ct1) {
                                 dequantize_بلاک_iq2_xs(
                                     vx, y, شے_ct1, iq2xs_grid,
                                     ksigns_iq2xs, kmask_iq2xs);
                             });
        });
    }
}

سانچا <قسم_ناں dst_t>
ساكن فراغ dequantize_صف_iq2_s_sycl(ثابت فراغ *vx, dst_t *y, ثابت int64_t k,
                                      dpct::قطار_ptr stream) {
    ثابت int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_شے<3> شے_ct1) {
                                 dequantize_بلاک_iq2_s(vx, y, شے_ct1);
                             });
        });
    }
}


سانچا <قسم_ناں dst_t>
ساكن فراغ dequantize_صف_iq3_xxs_sycl(ثابت فراغ *vx, dst_t *y, ثابت int64_t k,
                                        dpct::قطار_ptr stream) {
    ثابت int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_شے<3> شے_ct1) {
                                 dequantize_بلاک_iq3_xxs(
                                     vx, y, شے_ct1, iq3xxs_grid,
                                     ksigns_iq2xs, kmask_iq2xs);
                             });
        });
    }
}

سانچا <قسم_ناں dst_t>
ساكن فراغ dequantize_صف_iq3_xxs_sycl_reorder(ثابت فراغ * vx, dst_t * y, ثابت int64_t k,
                                                dpct::قطار_ptr stream) {
    ثابت int64_t nb = k / QK_K;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_شے<3> شے_ct1) {
            dequantize_بلاک_iq3_xxs_reorder(
                vx, y + شے_ct1.get_گروہ(2) * QK_K, شے_ct1.get_گروہ(2), شے_ct1, nb,
                iq3xxs_grid, ksigns_iq2xs, kmask_iq2xs);
        });
}

سانچا <قسم_ناں dst_t>
ساكن فراغ dequantize_صف_iq3_s_sycl(ثابت فراغ *vx, dst_t *y, ثابت int64_t k,
                                        dpct::قطار_ptr stream) {
    ثابت int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_شے<3> شے_ct1) {
                                 dequantize_بلاک_iq3_s(
                                     vx, y, شے_ct1, kmask_iq2xs, iq3s_grid);
                             });
        });
    }
}

سانچا <قسم_ناں dst_t>
ساكن فراغ dequantize_صف_iq3_s_sycl_reorder(ثابت فراغ * vx, dst_t * y, ثابت int64_t k,
                                              dpct::قطار_ptr stream) {
    ثابت int64_t nb = k / QK_K;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_شے<3> شے_ct1) {
            dequantize_بلاک_iq3_s_reorder(vx, y + شے_ct1.get_گروہ(2) * QK_K, شے_ct1.get_گروہ(2),
                                           شے_ct1, nb, kmask_iq2xs, iq3s_grid);
        });
}

سانچا <قسم_ناں dst_t>
ساكن فراغ dequantize_صف_iq4_xs_sycl(ثابت فراغ *vx, dst_t *y, ثابت int64_t k,
                                       dpct::قطار_ptr stream) {
    ثابت int64_t nb = (k + QK_K - 1) / QK_K;
#إذا QK_K == 64
    dequantize_صف_iq4_nl_sycl(vx, y, k, stream);
#وإلا
      {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                  cgh.parallel_for(
                      sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                            sycl::range<3>(1, 1, 32),
                                        sycl::range<3>(1, 1, 32)),
                      [=](sycl::nd_شے<3> شے_ct1) {
                            dequantize_بلاک_iq4_xs(vx, y, شے_ct1);
                      });
            });
      }
#جے_انت
}

سانچا <قسم_ناں dst_t>
ساكن فراغ dequantize_صف_iq4_nl_sycl(ثابت فراغ *vx, dst_t *y, ثابت int64_t k,
                                       dpct::قطار_ptr stream) {
    ثابت int64_t nb = (k + QK_K - 1) / QK_K;
      {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                  cgh.parallel_for(
                      sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                            sycl::range<3>(1, 1, 32),
                                        sycl::range<3>(1, 1, 32)),
                      [=](sycl::nd_شے<3> شے_ct1) {
                            dequantize_بلاک_iq4_nl(vx, y, شے_ct1);
                      });
            });
      }
}

سانچا <قسم_ناں dst_t>
ساكن فراغ dequantize_صف_mxfp4_sycl(ثابت فراغ * vx, dst_t * y, ثابت int64_t k, dpct::قطار_ptr stream) {
    ثابت صحيح nb = (k + QK_K - 1) / QK_K;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_شے<3> شے_ct1) {
            dequantize_بلاک_mxfp4(vx, y, شے_ct1);
        });
}

سانچا <قسم_ناں dst_t>
ساكن فراغ dequantize_صف_mxfp4_sycl_reorder(ثابت فراغ * vx, dst_t * y, ثابت int64_t k, dpct::قطار_ptr stream) {
    GGML_ASSERT(k % QK_MXFP4 == 0);
    ثابت صحيح n_warp = (k / QK_MXFP4 + WARP_ناپ - 1) / WARP_ناپ;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, n_warp) * sycl::range<3>(1, 1, WARP_ناپ),
                          sycl::range<3>(1, 1, WARP_ناپ)),
        [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
            dequantize_بلاک_mxfp4_reorder(vx, y, k, شے_ct1);
        });
}

سانچا <قسم_ناں dst_t>
ساكن فراغ dequantize_صف_nvfp4_sycl(ثابت فراغ * vx, dst_t * y, ثابت int64_t k, dpct::قطار_ptr stream) {
    GGML_ASSERT(k % QK_NVFP4 == 0);
    ثابت صحيح nb = k / QK_NVFP4;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_شے<3> /*item_ct1*/) {
            dequantize_بلاک_nvfp4(vx, y, k);
        });
}


سانچا <صحيح qk, صحيح qr, dequantize_kernel_t dequantize_kernel, قسم_ناں dst_t>
ساكن فراغ dequantize_بلاک_nc(ثابت فراغ * __restrict__ vx, dst_t * __restrict__ y,
        ثابت int64_t ne00, ثابت int64_t ne01, ثابت int64_t ne02,
        ثابت int64_t s01, ثابت int64_t s02, ثابت int64_t s03) {
    خودکار          شے_ct1 = sycl::ext::oneapi::this_work_شے::get_nd_شے<3>();
    ثابت int64_t i00 = 2 * (int64_t(شے_ct1.get_local_range(2)) * شے_ct1.get_گروہ(2) + شے_ct1.get_local_id(2));

    إذا (i00 >= ne00) {
        أرجع;
    }

    ثابت int64_t i01 = شے_ct1.get_گروہ(1);
    ثابت int64_t i02 = شے_ct1.get_گروہ(0) % ne02;
    ثابت int64_t i03 = شے_ct1.get_گروہ(0) / ne02;

    ثابت int64_t ibx0 = i03*s03 + i02*s02 + i01*s01;

    ثابت int64_t ib = ibx0 + i00/qk; // block index
    ثابت int64_t iqs = (i00%qk)/qr; // quant index
    ثابت int64_t iybs = i00 - i00%qk; // y block start index
    ثابت int64_t y_وتھ = qr == 1 ? 1 : qk/2;

    // dequantize
    #جے_تعریف_شدہ GGML_SYCL_F16
        sycl::half2 v;
    #وإلا
        sycl::float2 v;
    #جے_انت

    dequantize_kernel(vx, ib, iqs, v);

    ثابت int64_t iy0 = ((i03*ne02 + i02)*ne01 + i01)*ne00 + iybs + iqs;
    y[iy0 + 0]        = ggml_sycl_cast<dst_t>(v.x());
    y[iy0 + y_وتھ] = ggml_sycl_cast<dst_t>(v.y());
}


سانچا <صحيح qk, صحيح qr, dequantize_kernel_t dequantize_kernel, قسم_ناں dst_t>
ساكن فراغ dequantize_بلاک_nc_sycl(ثابت فراغ *    vx,
                                  dst_t *         y,
                                  ثابت int64_t   ne00,
                                  ثابت int64_t   ne01,
                                  ثابت int64_t   ne02,
                                  ثابت int64_t   ne03,
                                  ثابت int64_t   s01,
                                  ثابت int64_t   s02,
                                  ثابت int64_t   s03,
                                  dpct::قطار_ptr stream) {
    ثابت dpct::dim3 عدد_بلاکاں((ne00 + 2 * SYCL_DEQUANTIZE_بلاک_ناپ - 1) / (2 * SYCL_DEQUANTIZE_بلاک_ناپ), ne01,
                                ne02 * ne03);
    stream->parallel_for(sycl::nd_range<3>(عدد_بلاکاں * sycl::range<3>(1, 1, SYCL_DEQUANTIZE_بلاک_ناپ),
                                           sycl::range<3>(1, 1, SYCL_DEQUANTIZE_بلاک_ناپ)),
                         [=](sycl::nd_شے<3> شے_ct1) {
                             GGML_بن_ورتیا(شے_ct1);
                             dequantize_بلاک_nc<qk, qr, dequantize_kernel>(vx, y, ne00, ne01, ne02, s01, s02, s03);
                         });
}
سانچا <قسم_ناں src_t, قسم_ناں dst_t>
ساكن فراغ convert_unary_nc(ثابت فراغ * __restrict__ vx, dst_t * __restrict__ y, ثابت int64_t ne00, ثابت int64_t ne01,
                          ثابت int64_t ne02, ثابت int64_t s01, ثابت int64_t s02, ثابت int64_t s03,
                          ثابت sycl::nd_شے<3> & شے_ct1) {

    ثابت int64_t work_گروہ_ناپ = شے_ct1.get_local_range(2);
    ثابت int64_t global_id       = شے_ct1.get_local_id(2) + work_گروہ_ناپ * شے_ct1.get_گروہ(2);

    ثابت int64_t i01 = شے_ct1.get_گروہ(1);
    ثابت int64_t i02 = شے_ct1.get_گروہ(0) % ne02;
    ثابت int64_t i03 = شے_ct1.get_گروہ(0) / ne02;

    // make each work-item deal with more elements since sycl global range can not exceed max int
    ثابت src_t * x = static_cast<ثابت src_t *>(vx);
    ثابت int64_t ix = i03 * s03 + i02 * s02 + i01 * s01;
    ثابت int64_t iy = ((i03 * ne02 + i02) * ne01 + i01) * ne00;

#ہدایت unroll
    لكل (int64_t i00 = global_id; i00 < ne00; i00 += work_گروہ_ناپ * شے_ct1.get_گروہ_range(2)) {
        y[iy + i00] = static_cast<dst_t>(x[ix + i00]);
    }
}

سانچا <قسم_ناں src_t, قسم_ناں dst_t>
ساكن فراغ convert_unary_nc_sycl(ثابت فراغ * __restrict__ vx, dst_t * __restrict__ y,
                                  ثابت int64_t ne00, ثابت int64_t ne01, ثابت int64_t ne02, ثابت int64_t ne03,
                                  ثابت int64_t s01, ثابت int64_t s02, ثابت int64_t s03, dpct::قطار_ptr queue) {
    dpct::has_capability_or_fail(queue->get_device(), { sycl::aspect::fp16 });

    sycl::range<3> global_ناپ(ne02 * ne03, ne01, ceil_div(ne00, SYCL_DEQUANTIZE_بلاک_ناپ));

    // decrease global range when it exceeds the max int
    // TODO: Downsample logic is separated from the kernel, a rewrite is desirable
    int64_t        downsized_workgroup = downsample_sycl_global_range(global_ناپ[0], SYCL_DEQUANTIZE_بلاک_ناپ);
    sycl::range<3> workgroup_ناپ(1, 1, downsized_workgroup);

    queue->parallel_for(sycl::nd_range<3>(global_ناپ * workgroup_ناپ, workgroup_ناپ), [=](sycl::nd_شے<3> شے_ct1) {
        convert_unary_nc<src_t>(vx, y, ne00, ne01, ne02, s01, s02, s03, شے_ct1);
    });
}

سانچا <قسم_ناں src_t, قسم_ناں dst_t>
ساكن فراغ convert_unary_sycl(ثابت فراغ * vx, dst_t * y, ثابت int64_t k, dpct::قطار_ptr queue) {
    convert_unary_nc_sycl<src_t>(vx, y, k, 1, 1, 1, k, k, k, queue);
}


ساكن منطقي is_reordered_src0(ثابت ggml_tensor * dst) {
    أرجع dst->src[0]->extra &&
           ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder;
}

سانچا <قسم_ناں dst_t>
ساكن منزل_t_sycl_t<dst_t> ggml_get_dequant_sycl(ggml_type type, ggml_tensor * dst) {
    بدل (type) {
        حالة GGML_TYPE_Q1_0:
            أرجع dequantize_بلاک_sycl<QK1_0, QR1_0, dequantize_q1_0, dst_t>;
        حالة GGML_TYPE_Q2_0:
            أرجع dequantize_بلاک_sycl<QK2_0, QR2_0, dequantize_q2_0, dst_t>;
        حالة GGML_TYPE_Q4_0:
            إذا (is_reordered_src0(dst)) {
                أرجع dequantize_صف_q4_0_sycl_reorder;
            } وإلا إذا constexpr (std::is_same_v<dst_t, عائم>) {
                أرجع dequantize_صف_q4_0_sycl;
            } وإلا {
                أرجع dequantize_بلاک_sycl<QK4_0, QR4_0, dequantize_q4_0, dst_t>;
            }
        حالة GGML_TYPE_Q4_1:
            إذا constexpr (std::is_same_v<dst_t, عائم>) {
                أرجع dequantize_صف_q4_1_sycl;
            } وإلا {
                أرجع dequantize_بلاک_sycl<QK4_1, QR4_1, dequantize_q4_1, dst_t>;
            }
        حالة GGML_TYPE_Q5_0:
            أرجع dequantize_بلاک_sycl<QK5_0, QR5_0, dequantize_q5_0, dst_t>;
        حالة GGML_TYPE_Q5_1:
            أرجع dequantize_بلاک_sycl<QK5_1, QR5_1, dequantize_q5_1, dst_t>;
        حالة GGML_TYPE_Q8_0:
            إذا (is_reordered_src0(dst)) {
                أرجع dequantize_صف_q8_0_sycl_reorder;
            } وإلا {
                أرجع dequantize_بلاک_sycl<QK8_0, QR8_0, dequantize_q8_0, dst_t>;
            }
        حالة GGML_TYPE_Q2_K:
            إذا (is_reordered_src0(dst)) {
                أرجع dequantize_صف_q2_K_sycl_reorder;
            } وإلا {
                أرجع dequantize_صف_q2_K_sycl;
            }
        حالة GGML_TYPE_Q3_K:
            إذا (is_reordered_src0(dst)) {
                أرجع dequantize_صف_q3_K_sycl_reorder;
            } وإلا {
                أرجع dequantize_صف_q3_K_sycl;
            }
        حالة GGML_TYPE_Q4_K:
            إذا (is_reordered_src0(dst)) {
                أرجع dequantize_صف_q4_K_sycl_reorder;
            } وإلا {
                أرجع dequantize_صف_q4_K_sycl;
            }
        حالة GGML_TYPE_Q5_K:
            إذا (is_reordered_src0(dst)) {
                أرجع dequantize_صف_q5_K_sycl_reorder;
            } وإلا {
                أرجع dequantize_صف_q5_K_sycl;
            }
        حالة GGML_TYPE_Q6_K:
            إذا (is_reordered_src0(dst)) {
                أرجع dequantize_صف_q6_K_sycl_reorder;
            } وإلا {
                أرجع dequantize_صف_q6_K_sycl;
            }
        حالة GGML_TYPE_IQ1_S:
            أرجع dequantize_صف_iq1_s_sycl;
        حالة GGML_TYPE_IQ1_M:
            أرجع dequantize_صف_iq1_m_sycl;
        حالة GGML_TYPE_IQ2_XXS:
            أرجع dequantize_صف_iq2_xxs_sycl;
        حالة GGML_TYPE_IQ2_XS:
            أرجع dequantize_صف_iq2_xs_sycl;
        حالة GGML_TYPE_IQ2_S:
            أرجع dequantize_صف_iq2_s_sycl;
        حالة GGML_TYPE_IQ3_XXS:
            إذا (is_reordered_src0(dst)) {
                أرجع dequantize_صف_iq3_xxs_sycl_reorder;
            } وإلا {
                أرجع dequantize_صف_iq3_xxs_sycl;
            }
        حالة GGML_TYPE_IQ3_S:
            إذا (is_reordered_src0(dst)) {
                أرجع dequantize_صف_iq3_s_sycl_reorder;
            } وإلا {
                أرجع dequantize_صف_iq3_s_sycl;
            }
        حالة GGML_TYPE_IQ4_XS:
            أرجع dequantize_صف_iq4_xs_sycl;
        حالة GGML_TYPE_IQ4_NL:
            أرجع dequantize_صف_iq4_nl_sycl;
        حالة GGML_TYPE_MXFP4:
            إذا (is_reordered_src0(dst)) {
                أرجع dequantize_صف_mxfp4_sycl_reorder;
            }
            أرجع dequantize_صف_mxfp4_sycl;
        حالة GGML_TYPE_NVFP4:
            أرجع dequantize_صف_nvfp4_sycl;
        حالة GGML_TYPE_F32:
            إذا constexpr (std::is_same_v<dst_t, sycl::half>) {
                أرجع convert_unary_sycl<عائم>;
            }
            اكسر;
        حالة GGML_TYPE_F16:
            إذا constexpr (std::is_same_v<dst_t, عائم>) {
                أرجع convert_unary_sycl<sycl::half>;
            }
            اكسر;
#جے_تعریف_شدہ GGML_SYCL_HAS_BF16
        حالة GGML_TYPE_BF16:
            أرجع convert_unary_sycl<sycl::ext::oneapi::bfloat16>;
#جے_انت
        افتراضي:
            اكسر;
    }
    GGML_ABORT("fatal error: unsupport data type=%s\n", ggml_type_name(type));
}

منزل_fp16_sycl_t ggml_get_منزل_fp16_sycl(ggml_type type, ggml_tensor * dst) {
    أرجع ggml_get_dequant_sycl<sycl::half>(type, dst);
}

منزل_fp32_sycl_t ggml_get_منزل_fp32_sycl(ggml_type type, ggml_tensor *dst) {
    أرجع ggml_get_dequant_sycl<عائم>(type, dst);
}


#جے_تعریف_شدہ GGML_SYCL_HAS_BF16
منزل_bf16_sycl_t ggml_get_منزل_bf16_sycl(ggml_type type, ggml_tensor * /*dst*/) {
    بدل (type) {
        حالة GGML_TYPE_F32:
            أرجع convert_unary_sycl<عائم>;
        حالة GGML_TYPE_F16:
            أرجع convert_unary_sycl<sycl::half>;
        حالة GGML_TYPE_BF16:
            أرجع convert_unary_sycl<sycl::ext::oneapi::bfloat16>;
        افتراضي:
            GGML_ABORT("fatal error: unsupport data type=%s\n", ggml_type_name(type));
    }
}
#جے_انت

منزل_fp16_nc_sycl_t ggml_get_منزل_fp16_nc_sycl(ggml_type type) {
    بدل (type) {
        حالة GGML_TYPE_F32:
            أرجع convert_unary_nc_sycl<عائم>;
#جے_تعریف_شدہ GGML_SYCL_HAS_BF16
        حالة GGML_TYPE_BF16:
            أرجع convert_unary_nc_sycl<sycl::ext::oneapi::bfloat16>;
#جے_انت
        حالة GGML_TYPE_Q1_0:
            أرجع dequantize_بلاک_nc_sycl<QK1_0, QR1_0, dequantize_q1_0>;
        حالة GGML_TYPE_Q4_0:
            أرجع dequantize_بلاک_nc_sycl<QK4_0, QR4_0, dequantize_q4_0>;
        حالة GGML_TYPE_Q4_1:
            أرجع dequantize_بلاک_nc_sycl<QK4_1, QR4_1, dequantize_q4_1>;
        حالة GGML_TYPE_Q5_0:
            أرجع dequantize_بلاک_nc_sycl<QK5_0, QR5_0, dequantize_q5_0>;
        حالة GGML_TYPE_Q5_1:
            أرجع dequantize_بلاک_nc_sycl<QK5_1, QR5_1, dequantize_q5_1>;
        حالة GGML_TYPE_Q8_0:
            أرجع dequantize_بلاک_nc_sycl<QK8_0, QR8_0, dequantize_q8_0>;
        افتراضي:
            أرجع خالی_سوچک;
    }
}
