#అన్తర్భావ "convert.hpp"
#అన్తర్భావ "dequantize.hpp"
#అన్తర్భావ "presets.hpp"

మూస <పూర్ణాంకం qk, పూర్ణాంకం qr, dequantize_kernel_t dequantize_kernel, రకం_పేరు dst_t>
స్థిర శూన్య dequantize_ఖండం(స్థిరరాశి శూన్య * __restrict__ vx, dst_t * __restrict__ y, స్థిరరాశి int64_t k,
                             స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {
    స్థిరరాశి int64_t i = 2 * (అంశం_ct1.get_local_range(2) * అంశం_ct1.get_సమూహం(2) +
                       అంశం_ct1.get_local_id(2));

    ఒకవేళ (i >= k) {
        తిరిగి;
    }

    స్థిరరాశి int64_t ib = i/qk; // block index
    స్థిరరాశి int64_t iqs = (i%qk)/qr; // quant index
    స్థిరరాశి int64_t iybs = i - i%qk; // y block start index
    స్థిరరాశి int64_t y_స్థానభ్రంశం = qr == 1 ? 1 : qk/2;

    // dequantize
    dfloat2 v;
    dequantize_kernel(vx, ib, iqs, v);

    y[iybs + iqs + 0] = v.x();
    y[iybs + iqs + y_స్థానభ్రంశం] = v.y();
}

మూస <పూర్ణాంకం qk, పూర్ణాంకం qr, dequantize_kernel_t dequantize_kernel, రకం_పేరు dst_t>
స్థిర శూన్య dequantize_ఖండం_sycl(స్థిరరాశి శూన్య *__restrict__ vx,
                                  dst_t *__restrict__ y, స్థిరరాశి int64_t k,
                                  dpct::వరుస_ptr stream) {
    స్థిరరాశి int64_t అంకె_ఖండాలు = (k + 2*SYCL_DEQUANTIZE_ఖండం_కొలత - 1) / (2*SYCL_DEQUANTIZE_ఖండం_కొలత);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, అంకె_ఖండాలు) *
                    sycl::range<3>(1, 1, SYCL_DEQUANTIZE_ఖండం_కొలత),
                sycl::range<3>(1, 1, SYCL_DEQUANTIZE_ఖండం_కొలత)),
            [=](sycl::nd_అంశం<3> అంశం_ct1) {
                dequantize_ఖండం<qk, qr, dequantize_kernel>(vx, y, k, అంశం_ct1);
            });
    }
}

మూస <రకం_పేరు dst_t>
స్థిర శూన్య dequantize_అడ్డువరుస_q2_K_sycl(స్థిరరాశి శూన్య *vx, dst_t *y, స్థిరరాశి int64_t k,
                                     dpct::వరుస_ptr stream) {
    స్థిరరాశి int64_t nb = k / QK_K;
#ఒకవేళ QK_K == 256
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 64),
                                               sycl::range<3>(1, 1, 64)),
                             [=](sycl::nd_అంశం<3> అంశం_ct1) {
                                 dequantize_ఖండం_q2_K(vx, y, అంశం_ct1);
                             });
    }
#లేకపోతే
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_అంశం<3> అంశం_ct1) {
                                 dequantize_ఖండం_q2_K(vx, y, అంశం_ct1);
                             });
    }

#అయితే_ముగింపు
}

మూస <రకం_పేరు dst_t>
స్థిర శూన్య dequantize_అడ్డువరుస_q2_K_sycl_reorder(స్థిరరాశి శూన్య *vx, dst_t *y, స్థిరరాశి int64_t k,
                                             dpct::వరుస_ptr stream) {
    స్థిరరాశి int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
        [=](sycl::nd_అంశం<3> అంశం_ct1) {
            dequantize_ఖండం_q2_K_reorder(vx, y + అంశం_ct1.get_సమూహం(2) * QK_K, అంశం_ct1.get_సమూహం(2), అంశం_ct1, nb);
        });
}

మూస <రకం_పేరు dst_t>
స్థిర శూన్య dequantize_అడ్డువరుస_q3_K_sycl(స్థిరరాశి శూన్య *vx, dst_t *y, స్థిరరాశి int64_t k,
                                     dpct::వరుస_ptr stream) {
    స్థిరరాశి int64_t nb = k / QK_K;
#ఒకవేళ QK_K == 256
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 64),
                                               sycl::range<3>(1, 1, 64)),
                             [=](sycl::nd_అంశం<3> అంశం_ct1) {
                                 dequantize_ఖండం_q3_K(vx, y, అంశం_ct1);
                             });
    }
#లేకపోతే
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_అంశం<3> అంశం_ct1) {
                                 dequantize_ఖండం_q3_K(vx, y, అంశం_ct1);
                             });
    }
#అయితే_ముగింపు
}

మూస <రకం_పేరు dst_t>
స్థిర శూన్య dequantize_అడ్డువరుస_q3_K_sycl_reorder(స్థిరరాశి శూన్య *vx, dst_t *y, స్థిరరాశి int64_t k,
                                             dpct::వరుస_ptr stream) {
    స్థిరరాశి int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
        [=](sycl::nd_అంశం<3> అంశం_ct1) {
            dequantize_ఖండం_q3_K_reorder(vx, y + అంశం_ct1.get_సమూహం(2) * QK_K, అంశం_ct1.get_సమూహం(2), అంశం_ct1, nb);
        });
}

మూస <రకం_పేరు dst_t>
స్థిర శూన్య dequantize_అడ్డువరుస_q4_0_sycl(స్థిరరాశి శూన్య *vx, dst_t *y, స్థిరరాశి int64_t k,
                                     dpct::వరుస_ptr stream) {
    స్థిరరాశి int64_t nb32 = k / 32;
    స్థిరరాశి int64_t nb = (k + 255) / 256;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_అంశం<3> అంశం_ct1) {
                                 dequantize_ఖండం_q4_0(vx, y, nb32, అంశం_ct1);
                             });
    }
}

మూస <రకం_పేరు dst_t>
స్థిర శూన్య dequantize_అడ్డువరుస_q4_0_sycl_reorder(స్థిరరాశి శూన్య *vx, dst_t *y, స్థిరరాశి int64_t k,
                                     dpct::వరుస_ptr stream) {

    dpct::has_capability_or_fail(stream->get_device(),
                                    {sycl::aspect::fp16});

    GGML_ASSERT(k % QK4_0 == 0);
    స్థిరరాశి int64_t n_groups = ceil_div(k / 2, SYCL_DEQUANTIZE_ఖండం_కొలత);
    // decrease global range when it exceeds the max int; the kernel loops to cover the rest
    స్థిరరాశి int64_t wg_కొలత  = downsample_sycl_global_range(n_groups, SYCL_DEQUANTIZE_ఖండం_కొలత);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, n_groups) *
        sycl::range<3>(1, 1, wg_కొలత),
        sycl::range<3>(1, 1, wg_కొలత)),
        [=](sycl::nd_అంశం<3> అంశం_ct1) {
            dequantize_ఖండం_q4_0_reorder(vx, y, k, అంశం_ct1);
        });
}

మూస <రకం_పేరు dst_t>
స్థిర శూన్య dequantize_అడ్డువరుస_q8_0_sycl_reorder(స్థిరరాశి శూన్య *vx, dst_t *y, స్థిరరాశి int64_t k,
                                     dpct::వరుస_ptr stream) {

    dpct::has_capability_or_fail(stream->get_device(),
                                    {sycl::aspect::fp16});

    // flat range<1>, two pairs (4 elements) per item: no nd_range/group math, one vectorized store.
    // k % QK8_0 == 0 implies k % 4 == 0, so the wide launch covers the row exactly (no tail).
    GGML_ASSERT(k % QK8_0 == 0);
    stream->parallel_for(sycl::range<1>((size_t) (k / 4)),
        [=](sycl::id<1> idx) {
            dequantize_ఖండం_q8_0_reorder_wide(vx, y, k, idx);
        });
}

మూస <రకం_పేరు dst_t>
స్థిర శూన్య dequantize_అడ్డువరుస_q4_1_sycl(స్థిరరాశి శూన్య *vx, dst_t *y, స్థిరరాశి int64_t k,
                                     dpct::వరుస_ptr stream) {
    స్థిరరాశి int64_t nb32 = k / 32;
    స్థిరరాశి int64_t nb = (k + 255) / 256;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_అంశం<3> అంశం_ct1) {
                                 dequantize_ఖండం_q4_1(vx, y, nb32, అంశం_ct1);
                             });
    }
}


మూస <రకం_పేరు dst_t>
స్థిర శూన్య dequantize_అడ్డువరుస_q4_K_sycl(స్థిరరాశి శూన్య *vx, dst_t *y, స్థిరరాశి int64_t k,
                                     dpct::వరుస_ptr stream) {
    స్థిరరాశి int64_t nb = k / QK_K;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) nb * SYCL_DEQK_WI_చొప్పున_ఖండం),
                                           sycl::range<1>((size_t) SYCL_DEQK_WI_చొప్పున_ఖండం)),
                         [=](sycl::nd_అంశం<1> it) { dequantize_ఖండం_q4_K_wide(vx, y, nb, it); });
}

మూస <రకం_పేరు dst_t>
స్థిర శూన్య dequantize_అడ్డువరుస_q4_K_sycl_reorder(స్థిరరాశి శూన్య * vx, dst_t * y, స్థిరరాశి int64_t k, dpct::వరుస_ptr stream) {
    స్థిరరాశి int64_t nb = k / QK_K;
    స్థిరరాశి size_t  local_కొలత  = 32;
    స్థిరరాశి size_t  global_కొలత = nb * local_కొలత;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint8_t, 1> scale_local_acc(sycl::range<1>(12), cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_కొలత), sycl::range<1>(local_కొలత)),
                         [=](sycl::nd_అంశం<1> అంశం_ct1) {
                             dequantize_ఖండం_q4_K_reorder(vx, y + అంశం_ct1.get_సమూహం(0) * QK_K, అంశం_ct1.get_సమూహం(0), get_సూచిక(scale_local_acc), అంశం_ct1, nb);
                         });
    });
}

మూస <రకం_పేరు dst_t>
స్థిర శూన్య dequantize_అడ్డువరుస_q5_K_sycl(స్థిరరాశి శూన్య *vx, dst_t *y, స్థిరరాశి int64_t k,
                                     dpct::వరుస_ptr stream) {
    స్థిరరాశి int64_t nb = k / QK_K;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) nb * SYCL_DEQK_WI_చొప్పున_ఖండం),
                                           sycl::range<1>((size_t) SYCL_DEQK_WI_చొప్పున_ఖండం)),
                         [=](sycl::nd_అంశం<1> it) { dequantize_ఖండం_q5_K_wide(vx, y, nb, it); });
}

మూస <రకం_పేరు dst_t>
స్థిర శూన్య dequantize_అడ్డువరుస_q5_K_sycl_reorder(స్థిరరాశి శూన్య * vx, dst_t * y, స్థిరరాశి int64_t k, dpct::వరుస_ptr stream) {
    స్థిరరాశి int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint8_t, 1> scale_local_acc(sycl::range<1>(K_SCALE_కొలత), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
            [=](sycl::nd_అంశం<3> అంశం_ct1) {
                dequantize_ఖండం_q5_K_reorder(vx, y + అంశం_ct1.get_సమూహం(2) * QK_K, అంశం_ct1.get_సమూహం(2), get_సూచిక(scale_local_acc), అంశం_ct1, nb);
            });
    });
}

మూస <రకం_పేరు dst_t>
స్థిర శూన్య dequantize_అడ్డువరుస_q6_K_sycl(స్థిరరాశి శూన్య *vx, dst_t *y, స్థిరరాశి int64_t k,
                                     dpct::వరుస_ptr stream) {
    స్థిరరాశి int64_t nb = k / QK_K;
#ఒకవేళ QK_K == 256
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 64),
                                               sycl::range<3>(1, 1, 64)),
                             [=](sycl::nd_అంశం<3> అంశం_ct1) {
                                 dequantize_ఖండం_q6_K(vx, y, అంశం_ct1);
                             });
    }
#లేకపోతే
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_అంశం<3> అంశం_ct1) {
                                 dequantize_ఖండం_q6_K(vx, y, అంశం_ct1);
                             });
    }

#అయితే_ముగింపు
}

మూస <రకం_పేరు dst_t>
స్థిర శూన్య dequantize_అడ్డువరుస_q6_K_sycl_reorder(స్థిరరాశి శూన్య * vx, dst_t * y, స్థిరరాశి int64_t k, dpct::వరుస_ptr stream) {
    స్థిరరాశి int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
        [=](sycl::nd_అంశం<3> అంశం_ct1) { dequantize_ఖండం_q6_K_reorder(vx, y + అంశం_ct1.get_సమూహం(2) * QK_K, అంశం_ct1.get_సమూహం(2), అంశం_ct1, nb); });
}

మూస <రకం_పేరు dst_t>
స్థిర శూన్య dequantize_అడ్డువరుస_iq1_s_sycl(స్థిరరాశి శూన్య *vx, dst_t *y, స్థిరరాశి int64_t k,
                                        dpct::వరుస_ptr stream) {
    స్థిరరాశి int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_అంశం<3> అంశం_ct1) {
                                 dequantize_ఖండం_iq1_s(
                                     vx, y, అంశం_ct1, iq1s_grid_gpu
                                     );
                             });
        });
    }
}

మూస <రకం_పేరు dst_t>
స్థిర శూన్య dequantize_అడ్డువరుస_iq1_m_sycl(స్థిరరాశి శూన్య *vx, dst_t *y, స్థిరరాశి int64_t k,
                                        dpct::వరుస_ptr stream) {
    స్థిరరాశి int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_అంశం<3> అంశం_ct1) {
                                 dequantize_ఖండం_iq1_m(
                                     vx, y, అంశం_ct1, iq1s_grid_gpu
                                     );
                             });
        });
    }
}

మూస <రకం_పేరు dst_t>
స్థిర శూన్య dequantize_అడ్డువరుస_iq2_xxs_sycl(స్థిరరాశి శూన్య *vx, dst_t *y, స్థిరరాశి int64_t k,
                                        dpct::వరుస_ptr stream) {
    స్థిరరాశి int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_అంశం<3> అంశం_ct1) {
                                 dequantize_ఖండం_iq2_xxs(
                                     vx, y, అంశం_ct1, iq2xxs_grid,
                                     ksigns_iq2xs, kmask_iq2xs);
                             });
        });
    }
}

మూస <రకం_పేరు dst_t>
స్థిర శూన్య dequantize_అడ్డువరుస_iq2_xs_sycl(స్థిరరాశి శూన్య *vx, dst_t *y, స్థిరరాశి int64_t k,
                                       dpct::వరుస_ptr stream) {
    స్థిరరాశి int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_అంశం<3> అంశం_ct1) {
                                 dequantize_ఖండం_iq2_xs(
                                     vx, y, అంశం_ct1, iq2xs_grid,
                                     ksigns_iq2xs, kmask_iq2xs);
                             });
        });
    }
}

మూస <రకం_పేరు dst_t>
స్థిర శూన్య dequantize_అడ్డువరుస_iq2_s_sycl(స్థిరరాశి శూన్య *vx, dst_t *y, స్థిరరాశి int64_t k,
                                      dpct::వరుస_ptr stream) {
    స్థిరరాశి int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_అంశం<3> అంశం_ct1) {
                                 dequantize_ఖండం_iq2_s(vx, y, అంశం_ct1);
                             });
        });
    }
}


మూస <రకం_పేరు dst_t>
స్థిర శూన్య dequantize_అడ్డువరుస_iq3_xxs_sycl(స్థిరరాశి శూన్య *vx, dst_t *y, స్థిరరాశి int64_t k,
                                        dpct::వరుస_ptr stream) {
    స్థిరరాశి int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_అంశం<3> అంశం_ct1) {
                                 dequantize_ఖండం_iq3_xxs(
                                     vx, y, అంశం_ct1, iq3xxs_grid,
                                     ksigns_iq2xs, kmask_iq2xs);
                             });
        });
    }
}

మూస <రకం_పేరు dst_t>
స్థిర శూన్య dequantize_అడ్డువరుస_iq3_xxs_sycl_reorder(స్థిరరాశి శూన్య * vx, dst_t * y, స్థిరరాశి int64_t k,
                                                dpct::వరుస_ptr stream) {
    స్థిరరాశి int64_t nb = k / QK_K;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_అంశం<3> అంశం_ct1) {
            dequantize_ఖండం_iq3_xxs_reorder(
                vx, y + అంశం_ct1.get_సమూహం(2) * QK_K, అంశం_ct1.get_సమూహం(2), అంశం_ct1, nb,
                iq3xxs_grid, ksigns_iq2xs, kmask_iq2xs);
        });
}

మూస <రకం_పేరు dst_t>
స్థిర శూన్య dequantize_అడ్డువరుస_iq3_s_sycl(స్థిరరాశి శూన్య *vx, dst_t *y, స్థిరరాశి int64_t k,
                                        dpct::వరుస_ptr stream) {
    స్థిరరాశి int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_అంశం<3> అంశం_ct1) {
                                 dequantize_ఖండం_iq3_s(
                                     vx, y, అంశం_ct1, kmask_iq2xs, iq3s_grid);
                             });
        });
    }
}

మూస <రకం_పేరు dst_t>
స్థిర శూన్య dequantize_అడ్డువరుస_iq3_s_sycl_reorder(స్థిరరాశి శూన్య * vx, dst_t * y, స్థిరరాశి int64_t k,
                                              dpct::వరుస_ptr stream) {
    స్థిరరాశి int64_t nb = k / QK_K;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_అంశం<3> అంశం_ct1) {
            dequantize_ఖండం_iq3_s_reorder(vx, y + అంశం_ct1.get_సమూహం(2) * QK_K, అంశం_ct1.get_సమూహం(2),
                                           అంశం_ct1, nb, kmask_iq2xs, iq3s_grid);
        });
}

మూస <రకం_పేరు dst_t>
స్థిర శూన్య dequantize_అడ్డువరుస_iq4_xs_sycl(స్థిరరాశి శూన్య *vx, dst_t *y, స్థిరరాశి int64_t k,
                                       dpct::వరుస_ptr stream) {
    స్థిరరాశి int64_t nb = (k + QK_K - 1) / QK_K;
#ఒకవేళ QK_K == 64
    dequantize_అడ్డువరుస_iq4_nl_sycl(vx, y, k, stream);
#లేకపోతే
      {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                  cgh.parallel_for(
                      sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                            sycl::range<3>(1, 1, 32),
                                        sycl::range<3>(1, 1, 32)),
                      [=](sycl::nd_అంశం<3> అంశం_ct1) {
                            dequantize_ఖండం_iq4_xs(vx, y, అంశం_ct1);
                      });
            });
      }
#అయితే_ముగింపు
}

మూస <రకం_పేరు dst_t>
స్థిర శూన్య dequantize_అడ్డువరుస_iq4_nl_sycl(స్థిరరాశి శూన్య *vx, dst_t *y, స్థిరరాశి int64_t k,
                                       dpct::వరుస_ptr stream) {
    స్థిరరాశి int64_t nb = (k + QK_K - 1) / QK_K;
      {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                  cgh.parallel_for(
                      sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                            sycl::range<3>(1, 1, 32),
                                        sycl::range<3>(1, 1, 32)),
                      [=](sycl::nd_అంశం<3> అంశం_ct1) {
                            dequantize_ఖండం_iq4_nl(vx, y, అంశం_ct1);
                      });
            });
      }
}

మూస <రకం_పేరు dst_t>
స్థిర శూన్య dequantize_అడ్డువరుస_mxfp4_sycl(స్థిరరాశి శూన్య * vx, dst_t * y, స్థిరరాశి int64_t k, dpct::వరుస_ptr stream) {
    స్థిరరాశి పూర్ణాంకం nb = (k + QK_K - 1) / QK_K;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_అంశం<3> అంశం_ct1) {
            dequantize_ఖండం_mxfp4(vx, y, అంశం_ct1);
        });
}

మూస <రకం_పేరు dst_t>
స్థిర శూన్య dequantize_అడ్డువరుస_mxfp4_sycl_reorder(స్థిరరాశి శూన్య * vx, dst_t * y, స్థిరరాశి int64_t k, dpct::వరుస_ptr stream) {
    GGML_ASSERT(k % QK_MXFP4 == 0);
    స్థిరరాశి పూర్ణాంకం n_warp = (k / QK_MXFP4 + WARP_కొలత - 1) / WARP_కొలత;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, n_warp) * sycl::range<3>(1, 1, WARP_కొలత),
                          sycl::range<3>(1, 1, WARP_కొలత)),
        [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
            dequantize_ఖండం_mxfp4_reorder(vx, y, k, అంశం_ct1);
        });
}

మూస <రకం_పేరు dst_t>
స్థిర శూన్య dequantize_అడ్డువరుస_nvfp4_sycl(స్థిరరాశి శూన్య * vx, dst_t * y, స్థిరరాశి int64_t k, dpct::వరుస_ptr stream) {
    GGML_ASSERT(k % QK_NVFP4 == 0);
    స్థిరరాశి పూర్ణాంకం nb = k / QK_NVFP4;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_అంశం<3> /*item_ct1*/) {
            dequantize_ఖండం_nvfp4(vx, y, k);
        });
}


మూస <పూర్ణాంకం qk, పూర్ణాంకం qr, dequantize_kernel_t dequantize_kernel, రకం_పేరు dst_t>
స్థిర శూన్య dequantize_ఖండం_nc(స్థిరరాశి శూన్య * __restrict__ vx, dst_t * __restrict__ y,
        స్థిరరాశి int64_t ne00, స్థిరరాశి int64_t ne01, స్థిరరాశి int64_t ne02,
        స్థిరరాశి int64_t s01, స్థిరరాశి int64_t s02, స్థిరరాశి int64_t s03) {
    స్వయంచాలక          అంశం_ct1 = sycl::ext::oneapi::this_work_అంశం::get_nd_అంశం<3>();
    స్థిరరాశి int64_t i00 = 2 * (int64_t(అంశం_ct1.get_local_range(2)) * అంశం_ct1.get_సమూహం(2) + అంశం_ct1.get_local_id(2));

    ఒకవేళ (i00 >= ne00) {
        తిరిగి;
    }

    స్థిరరాశి int64_t i01 = అంశం_ct1.get_సమూహం(1);
    స్థిరరాశి int64_t i02 = అంశం_ct1.get_సమూహం(0) % ne02;
    స్థిరరాశి int64_t i03 = అంశం_ct1.get_సమూహం(0) / ne02;

    స్థిరరాశి int64_t ibx0 = i03*s03 + i02*s02 + i01*s01;

    స్థిరరాశి int64_t ib = ibx0 + i00/qk; // block index
    స్థిరరాశి int64_t iqs = (i00%qk)/qr; // quant index
    స్థిరరాశి int64_t iybs = i00 - i00%qk; // y block start index
    స్థిరరాశి int64_t y_స్థానభ్రంశం = qr == 1 ? 1 : qk/2;

    // dequantize
    #నిర్వచితమైతే GGML_SYCL_F16
        sycl::half2 v;
    #లేకపోతే
        sycl::float2 v;
    #అయితే_ముగింపు

    dequantize_kernel(vx, ib, iqs, v);

    స్థిరరాశి int64_t iy0 = ((i03*ne02 + i02)*ne01 + i01)*ne00 + iybs + iqs;
    y[iy0 + 0]        = ggml_sycl_cast<dst_t>(v.x());
    y[iy0 + y_స్థానభ్రంశం] = ggml_sycl_cast<dst_t>(v.y());
}


మూస <పూర్ణాంకం qk, పూర్ణాంకం qr, dequantize_kernel_t dequantize_kernel, రకం_పేరు dst_t>
స్థిర శూన్య dequantize_ఖండం_nc_sycl(స్థిరరాశి శూన్య *    vx,
                                  dst_t *         y,
                                  స్థిరరాశి int64_t   ne00,
                                  స్థిరరాశి int64_t   ne01,
                                  స్థిరరాశి int64_t   ne02,
                                  స్థిరరాశి int64_t   ne03,
                                  స్థిరరాశి int64_t   s01,
                                  స్థిరరాశి int64_t   s02,
                                  స్థిరరాశి int64_t   s03,
                                  dpct::వరుస_ptr stream) {
    స్థిరరాశి dpct::dim3 అంకె_ఖండాలు((ne00 + 2 * SYCL_DEQUANTIZE_ఖండం_కొలత - 1) / (2 * SYCL_DEQUANTIZE_ఖండం_కొలత), ne01,
                                ne02 * ne03);
    stream->parallel_for(sycl::nd_range<3>(అంకె_ఖండాలు * sycl::range<3>(1, 1, SYCL_DEQUANTIZE_ఖండం_కొలత),
                                           sycl::range<3>(1, 1, SYCL_DEQUANTIZE_ఖండం_కొలత)),
                         [=](sycl::nd_అంశం<3> అంశం_ct1) {
                             GGML_వాడని(అంశం_ct1);
                             dequantize_ఖండం_nc<qk, qr, dequantize_kernel>(vx, y, ne00, ne01, ne02, s01, s02, s03);
                         });
}
మూస <రకం_పేరు src_t, రకం_పేరు dst_t>
స్థిర శూన్య convert_unary_nc(స్థిరరాశి శూన్య * __restrict__ vx, dst_t * __restrict__ y, స్థిరరాశి int64_t ne00, స్థిరరాశి int64_t ne01,
                          స్థిరరాశి int64_t ne02, స్థిరరాశి int64_t s01, స్థిరరాశి int64_t s02, స్థిరరాశి int64_t s03,
                          స్థిరరాశి sycl::nd_అంశం<3> & అంశం_ct1) {

    స్థిరరాశి int64_t work_సమూహం_కొలత = అంశం_ct1.get_local_range(2);
    స్థిరరాశి int64_t global_id       = అంశం_ct1.get_local_id(2) + work_సమూహం_కొలత * అంశం_ct1.get_సమూహం(2);

    స్థిరరాశి int64_t i01 = అంశం_ct1.get_సమూహం(1);
    స్థిరరాశి int64_t i02 = అంశం_ct1.get_సమూహం(0) % ne02;
    స్థిరరాశి int64_t i03 = అంశం_ct1.get_సమూహం(0) / ne02;

    // make each work-item deal with more elements since sycl global range can not exceed max int
    స్థిరరాశి src_t * x = static_cast<స్థిరరాశి src_t *>(vx);
    స్థిరరాశి int64_t ix = i03 * s03 + i02 * s02 + i01 * s01;
    స్థిరరాశి int64_t iy = ((i03 * ne02 + i02) * ne01 + i01) * ne00;

#సూచన unroll
    కొరకు (int64_t i00 = global_id; i00 < ne00; i00 += work_సమూహం_కొలత * అంశం_ct1.get_సమూహం_range(2)) {
        y[iy + i00] = static_cast<dst_t>(x[ix + i00]);
    }
}

మూస <రకం_పేరు src_t, రకం_పేరు dst_t>
స్థిర శూన్య convert_unary_nc_sycl(స్థిరరాశి శూన్య * __restrict__ vx, dst_t * __restrict__ y,
                                  స్థిరరాశి int64_t ne00, స్థిరరాశి int64_t ne01, స్థిరరాశి int64_t ne02, స్థిరరాశి int64_t ne03,
                                  స్థిరరాశి int64_t s01, స్థిరరాశి int64_t s02, స్థిరరాశి int64_t s03, dpct::వరుస_ptr queue) {
    dpct::has_capability_or_fail(queue->get_device(), { sycl::aspect::fp16 });

    sycl::range<3> global_కొలత(ne02 * ne03, ne01, ceil_div(ne00, SYCL_DEQUANTIZE_ఖండం_కొలత));

    // decrease global range when it exceeds the max int
    // TODO: Downsample logic is separated from the kernel, a rewrite is desirable
    int64_t        downsized_workgroup = downsample_sycl_global_range(global_కొలత[0], SYCL_DEQUANTIZE_ఖండం_కొలత);
    sycl::range<3> workgroup_కొలత(1, 1, downsized_workgroup);

    queue->parallel_for(sycl::nd_range<3>(global_కొలత * workgroup_కొలత, workgroup_కొలత), [=](sycl::nd_అంశం<3> అంశం_ct1) {
        convert_unary_nc<src_t>(vx, y, ne00, ne01, ne02, s01, s02, s03, అంశం_ct1);
    });
}

మూస <రకం_పేరు src_t, రకం_పేరు dst_t>
స్థిర శూన్య convert_unary_sycl(స్థిరరాశి శూన్య * vx, dst_t * y, స్థిరరాశి int64_t k, dpct::వరుస_ptr queue) {
    convert_unary_nc_sycl<src_t>(vx, y, k, 1, 1, 1, k, k, k, queue);
}


స్థిర తార్కిక is_reordered_src0(స్థిరరాశి ggml_tensor * dst) {
    తిరిగి dst->src[0]->extra &&
           ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder;
}

మూస <రకం_పేరు dst_t>
స్థిర గమ్యం_t_sycl_t<dst_t> ggml_get_dequant_sycl(ggml_type type, ggml_tensor * dst) {
    ఎంపిక (type) {
        స్థితి GGML_TYPE_Q1_0:
            తిరిగి dequantize_ఖండం_sycl<QK1_0, QR1_0, dequantize_q1_0, dst_t>;
        స్థితి GGML_TYPE_Q2_0:
            తిరిగి dequantize_ఖండం_sycl<QK2_0, QR2_0, dequantize_q2_0, dst_t>;
        స్థితి GGML_TYPE_Q4_0:
            ఒకవేళ (is_reordered_src0(dst)) {
                తిరిగి dequantize_అడ్డువరుస_q4_0_sycl_reorder;
            } లేకపోతే ఒకవేళ constexpr (std::is_same_v<dst_t, దశాంశం>) {
                తిరిగి dequantize_అడ్డువరుస_q4_0_sycl;
            } లేకపోతే {
                తిరిగి dequantize_ఖండం_sycl<QK4_0, QR4_0, dequantize_q4_0, dst_t>;
            }
        స్థితి GGML_TYPE_Q4_1:
            ఒకవేళ constexpr (std::is_same_v<dst_t, దశాంశం>) {
                తిరిగి dequantize_అడ్డువరుస_q4_1_sycl;
            } లేకపోతే {
                తిరిగి dequantize_ఖండం_sycl<QK4_1, QR4_1, dequantize_q4_1, dst_t>;
            }
        స్థితి GGML_TYPE_Q5_0:
            తిరిగి dequantize_ఖండం_sycl<QK5_0, QR5_0, dequantize_q5_0, dst_t>;
        స్థితి GGML_TYPE_Q5_1:
            తిరిగి dequantize_ఖండం_sycl<QK5_1, QR5_1, dequantize_q5_1, dst_t>;
        స్థితి GGML_TYPE_Q8_0:
            ఒకవేళ (is_reordered_src0(dst)) {
                తిరిగి dequantize_అడ్డువరుస_q8_0_sycl_reorder;
            } లేకపోతే {
                తిరిగి dequantize_ఖండం_sycl<QK8_0, QR8_0, dequantize_q8_0, dst_t>;
            }
        స్థితి GGML_TYPE_Q2_K:
            ఒకవేళ (is_reordered_src0(dst)) {
                తిరిగి dequantize_అడ్డువరుస_q2_K_sycl_reorder;
            } లేకపోతే {
                తిరిగి dequantize_అడ్డువరుస_q2_K_sycl;
            }
        స్థితి GGML_TYPE_Q3_K:
            ఒకవేళ (is_reordered_src0(dst)) {
                తిరిగి dequantize_అడ్డువరుస_q3_K_sycl_reorder;
            } లేకపోతే {
                తిరిగి dequantize_అడ్డువరుస_q3_K_sycl;
            }
        స్థితి GGML_TYPE_Q4_K:
            ఒకవేళ (is_reordered_src0(dst)) {
                తిరిగి dequantize_అడ్డువరుస_q4_K_sycl_reorder;
            } లేకపోతే {
                తిరిగి dequantize_అడ్డువరుస_q4_K_sycl;
            }
        స్థితి GGML_TYPE_Q5_K:
            ఒకవేళ (is_reordered_src0(dst)) {
                తిరిగి dequantize_అడ్డువరుస_q5_K_sycl_reorder;
            } లేకపోతే {
                తిరిగి dequantize_అడ్డువరుస_q5_K_sycl;
            }
        స్థితి GGML_TYPE_Q6_K:
            ఒకవేళ (is_reordered_src0(dst)) {
                తిరిగి dequantize_అడ్డువరుస_q6_K_sycl_reorder;
            } లేకపోతే {
                తిరిగి dequantize_అడ్డువరుస_q6_K_sycl;
            }
        స్థితి GGML_TYPE_IQ1_S:
            తిరిగి dequantize_అడ్డువరుస_iq1_s_sycl;
        స్థితి GGML_TYPE_IQ1_M:
            తిరిగి dequantize_అడ్డువరుస_iq1_m_sycl;
        స్థితి GGML_TYPE_IQ2_XXS:
            తిరిగి dequantize_అడ్డువరుస_iq2_xxs_sycl;
        స్థితి GGML_TYPE_IQ2_XS:
            తిరిగి dequantize_అడ్డువరుస_iq2_xs_sycl;
        స్థితి GGML_TYPE_IQ2_S:
            తిరిగి dequantize_అడ్డువరుస_iq2_s_sycl;
        స్థితి GGML_TYPE_IQ3_XXS:
            ఒకవేళ (is_reordered_src0(dst)) {
                తిరిగి dequantize_అడ్డువరుస_iq3_xxs_sycl_reorder;
            } లేకపోతే {
                తిరిగి dequantize_అడ్డువరుస_iq3_xxs_sycl;
            }
        స్థితి GGML_TYPE_IQ3_S:
            ఒకవేళ (is_reordered_src0(dst)) {
                తిరిగి dequantize_అడ్డువరుస_iq3_s_sycl_reorder;
            } లేకపోతే {
                తిరిగి dequantize_అడ్డువరుస_iq3_s_sycl;
            }
        స్థితి GGML_TYPE_IQ4_XS:
            తిరిగి dequantize_అడ్డువరుస_iq4_xs_sycl;
        స్థితి GGML_TYPE_IQ4_NL:
            తిరిగి dequantize_అడ్డువరుస_iq4_nl_sycl;
        స్థితి GGML_TYPE_MXFP4:
            ఒకవేళ (is_reordered_src0(dst)) {
                తిరిగి dequantize_అడ్డువరుస_mxfp4_sycl_reorder;
            }
            తిరిగి dequantize_అడ్డువరుస_mxfp4_sycl;
        స్థితి GGML_TYPE_NVFP4:
            తిరిగి dequantize_అడ్డువరుస_nvfp4_sycl;
        స్థితి GGML_TYPE_F32:
            ఒకవేళ constexpr (std::is_same_v<dst_t, sycl::half>) {
                తిరిగి convert_unary_sycl<దశాంశం>;
            }
            ఆపు;
        స్థితి GGML_TYPE_F16:
            ఒకవేళ constexpr (std::is_same_v<dst_t, దశాంశం>) {
                తిరిగి convert_unary_sycl<sycl::half>;
            }
            ఆపు;
#నిర్వచితమైతే GGML_SYCL_HAS_BF16
        స్థితి GGML_TYPE_BF16:
            తిరిగి convert_unary_sycl<sycl::ext::oneapi::bfloat16>;
#అయితే_ముగింపు
        అప్రమేయ:
            ఆపు;
    }
    GGML_ABORT("fatal error: unsupport data type=%s\n", ggml_type_name(type));
}

గమ్యం_fp16_sycl_t ggml_get_గమ్యం_fp16_sycl(ggml_type type, ggml_tensor * dst) {
    తిరిగి ggml_get_dequant_sycl<sycl::half>(type, dst);
}

గమ్యం_fp32_sycl_t ggml_get_గమ్యం_fp32_sycl(ggml_type type, ggml_tensor *dst) {
    తిరిగి ggml_get_dequant_sycl<దశాంశం>(type, dst);
}


#నిర్వచితమైతే GGML_SYCL_HAS_BF16
గమ్యం_bf16_sycl_t ggml_get_గమ్యం_bf16_sycl(ggml_type type, ggml_tensor * /*dst*/) {
    ఎంపిక (type) {
        స్థితి GGML_TYPE_F32:
            తిరిగి convert_unary_sycl<దశాంశం>;
        స్థితి GGML_TYPE_F16:
            తిరిగి convert_unary_sycl<sycl::half>;
        స్థితి GGML_TYPE_BF16:
            తిరిగి convert_unary_sycl<sycl::ext::oneapi::bfloat16>;
        అప్రమేయ:
            GGML_ABORT("fatal error: unsupport data type=%s\n", ggml_type_name(type));
    }
}
#అయితే_ముగింపు

గమ్యం_fp16_nc_sycl_t ggml_get_గమ్యం_fp16_nc_sycl(ggml_type type) {
    ఎంపిక (type) {
        స్థితి GGML_TYPE_F32:
            తిరిగి convert_unary_nc_sycl<దశాంశం>;
#నిర్వచితమైతే GGML_SYCL_HAS_BF16
        స్థితి GGML_TYPE_BF16:
            తిరిగి convert_unary_nc_sycl<sycl::ext::oneapi::bfloat16>;
#అయితే_ముగింపు
        స్థితి GGML_TYPE_Q1_0:
            తిరిగి dequantize_ఖండం_nc_sycl<QK1_0, QR1_0, dequantize_q1_0>;
        స్థితి GGML_TYPE_Q4_0:
            తిరిగి dequantize_ఖండం_nc_sycl<QK4_0, QR4_0, dequantize_q4_0>;
        స్థితి GGML_TYPE_Q4_1:
            తిరిగి dequantize_ఖండం_nc_sycl<QK4_1, QR4_1, dequantize_q4_1>;
        స్థితి GGML_TYPE_Q5_0:
            తిరిగి dequantize_ఖండం_nc_sycl<QK5_0, QR5_0, dequantize_q5_0>;
        స్థితి GGML_TYPE_Q5_1:
            తిరిగి dequantize_ఖండం_nc_sycl<QK5_1, QR5_1, dequantize_q5_1>;
        స్థితి GGML_TYPE_Q8_0:
            తిరిగి dequantize_ఖండం_nc_sycl<QK8_0, QR8_0, dequantize_q8_0>;
        అప్రమేయ:
            తిరిగి శూన్యసూచిక;
    }
}
