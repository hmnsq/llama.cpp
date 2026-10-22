#شامل "convert.hpp"
#شامل "dequantize.hpp"
#شامل "presets.hpp"

سانچہ <صحیح qk, صحیح qr, dequantize_kernel_t dequantize_kernel, قسم_نام dst_t>
ساکن خالی dequantize_بلاک(ثابت خالی * __restrict__ vx, dst_t * __restrict__ y, ثابت int64_t k,
                             ثابت sycl::nd_شے<3> &شے_ct1) {
    ثابت int64_t i = 2 * (شے_ct1.get_local_range(2) * شے_ct1.get_گروہ(2) +
                       شے_ct1.get_local_id(2));

    اگر (i >= k) {
        واپس;
    }

    ثابت int64_t ib = i/qk; // block index
    ثابت int64_t iqs = (i%qk)/qr; // quant index
    ثابت int64_t iybs = i - i%qk; // y block start index
    ثابت int64_t y_انحراف = qr == 1 ? 1 : qk/2;

    // dequantize
    dfloat2 v;
    dequantize_kernel(vx, ib, iqs, v);

    y[iybs + iqs + 0] = v.x();
    y[iybs + iqs + y_انحراف] = v.y();
}

سانچہ <صحیح qk, صحیح qr, dequantize_kernel_t dequantize_kernel, قسم_نام dst_t>
ساکن خالی dequantize_بلاک_sycl(ثابت خالی *__restrict__ vx,
                                  dst_t *__restrict__ y, ثابت int64_t k,
                                  dpct::قطار_ptr stream) {
    ثابت int64_t عدد_بلاکیں = (k + 2*SYCL_DEQUANTIZE_بلاک_سائز - 1) / (2*SYCL_DEQUANTIZE_بلاک_سائز);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, عدد_بلاکیں) *
                    sycl::range<3>(1, 1, SYCL_DEQUANTIZE_بلاک_سائز),
                sycl::range<3>(1, 1, SYCL_DEQUANTIZE_بلاک_سائز)),
            [=](sycl::nd_شے<3> شے_ct1) {
                dequantize_بلاک<qk, qr, dequantize_kernel>(vx, y, k, شے_ct1);
            });
    }
}

سانچہ <قسم_نام dst_t>
ساکن خالی dequantize_صف_q2_K_sycl(ثابت خالی *vx, dst_t *y, ثابت int64_t k,
                                     dpct::قطار_ptr stream) {
    ثابت int64_t nb = k / QK_K;
#اگر QK_K == 256
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
#ورنہ
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

#اگر_ختم
}

سانچہ <قسم_نام dst_t>
ساکن خالی dequantize_صف_q2_K_sycl_reorder(ثابت خالی *vx, dst_t *y, ثابت int64_t k,
                                             dpct::قطار_ptr stream) {
    ثابت int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
        [=](sycl::nd_شے<3> شے_ct1) {
            dequantize_بلاک_q2_K_reorder(vx, y + شے_ct1.get_گروہ(2) * QK_K, شے_ct1.get_گروہ(2), شے_ct1, nb);
        });
}

سانچہ <قسم_نام dst_t>
ساکن خالی dequantize_صف_q3_K_sycl(ثابت خالی *vx, dst_t *y, ثابت int64_t k,
                                     dpct::قطار_ptr stream) {
    ثابت int64_t nb = k / QK_K;
#اگر QK_K == 256
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
#ورنہ
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
#اگر_ختم
}

سانچہ <قسم_نام dst_t>
ساکن خالی dequantize_صف_q3_K_sycl_reorder(ثابت خالی *vx, dst_t *y, ثابت int64_t k,
                                             dpct::قطار_ptr stream) {
    ثابت int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
        [=](sycl::nd_شے<3> شے_ct1) {
            dequantize_بلاک_q3_K_reorder(vx, y + شے_ct1.get_گروہ(2) * QK_K, شے_ct1.get_گروہ(2), شے_ct1, nb);
        });
}

سانچہ <قسم_نام dst_t>
ساکن خالی dequantize_صف_q4_0_sycl(ثابت خالی *vx, dst_t *y, ثابت int64_t k,
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

سانچہ <قسم_نام dst_t>
ساکن خالی dequantize_صف_q4_0_sycl_reorder(ثابت خالی *vx, dst_t *y, ثابت int64_t k,
                                     dpct::قطار_ptr stream) {

    dpct::has_capability_or_fail(stream->get_device(),
                                    {sycl::aspect::fp16});

    GGML_ASSERT(k % QK4_0 == 0);
    ثابت int64_t n_groups = ceil_div(k / 2, SYCL_DEQUANTIZE_بلاک_سائز);
    // decrease global range when it exceeds the max int; the kernel loops to cover the rest
    ثابت int64_t wg_سائز  = downsample_sycl_global_range(n_groups, SYCL_DEQUANTIZE_بلاک_سائز);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, n_groups) *
        sycl::range<3>(1, 1, wg_سائز),
        sycl::range<3>(1, 1, wg_سائز)),
        [=](sycl::nd_شے<3> شے_ct1) {
            dequantize_بلاک_q4_0_reorder(vx, y, k, شے_ct1);
        });
}

سانچہ <قسم_نام dst_t>
ساکن خالی dequantize_صف_q8_0_sycl_reorder(ثابت خالی *vx, dst_t *y, ثابت int64_t k,
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

سانچہ <قسم_نام dst_t>
ساکن خالی dequantize_صف_q4_1_sycl(ثابت خالی *vx, dst_t *y, ثابت int64_t k,
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


سانچہ <قسم_نام dst_t>
ساکن خالی dequantize_صف_q4_K_sycl(ثابت خالی *vx, dst_t *y, ثابت int64_t k,
                                     dpct::قطار_ptr stream) {
    ثابت int64_t nb = k / QK_K;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) nb * SYCL_DEQK_WI_فی_بلاک),
                                           sycl::range<1>((size_t) SYCL_DEQK_WI_فی_بلاک)),
                         [=](sycl::nd_شے<1> it) { dequantize_بلاک_q4_K_wide(vx, y, nb, it); });
}

سانچہ <قسم_نام dst_t>
ساکن خالی dequantize_صف_q4_K_sycl_reorder(ثابت خالی * vx, dst_t * y, ثابت int64_t k, dpct::قطار_ptr stream) {
    ثابت int64_t nb = k / QK_K;
    ثابت size_t  local_سائز  = 32;
    ثابت size_t  global_سائز = nb * local_سائز;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint8_t, 1> scale_local_acc(sycl::range<1>(12), cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_سائز), sycl::range<1>(local_سائز)),
                         [=](sycl::nd_شے<1> شے_ct1) {
                             dequantize_بلاک_q4_K_reorder(vx, y + شے_ct1.get_گروہ(0) * QK_K, شے_ct1.get_گروہ(0), get_اشارہ_گر(scale_local_acc), شے_ct1, nb);
                         });
    });
}

سانچہ <قسم_نام dst_t>
ساکن خالی dequantize_صف_q5_K_sycl(ثابت خالی *vx, dst_t *y, ثابت int64_t k,
                                     dpct::قطار_ptr stream) {
    ثابت int64_t nb = k / QK_K;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) nb * SYCL_DEQK_WI_فی_بلاک),
                                           sycl::range<1>((size_t) SYCL_DEQK_WI_فی_بلاک)),
                         [=](sycl::nd_شے<1> it) { dequantize_بلاک_q5_K_wide(vx, y, nb, it); });
}

سانچہ <قسم_نام dst_t>
ساکن خالی dequantize_صف_q5_K_sycl_reorder(ثابت خالی * vx, dst_t * y, ثابت int64_t k, dpct::قطار_ptr stream) {
    ثابت int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint8_t, 1> scale_local_acc(sycl::range<1>(K_SCALE_سائز), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
            [=](sycl::nd_شے<3> شے_ct1) {
                dequantize_بلاک_q5_K_reorder(vx, y + شے_ct1.get_گروہ(2) * QK_K, شے_ct1.get_گروہ(2), get_اشارہ_گر(scale_local_acc), شے_ct1, nb);
            });
    });
}

سانچہ <قسم_نام dst_t>
ساکن خالی dequantize_صف_q6_K_sycl(ثابت خالی *vx, dst_t *y, ثابت int64_t k,
                                     dpct::قطار_ptr stream) {
    ثابت int64_t nb = k / QK_K;
#اگر QK_K == 256
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
#ورنہ
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

#اگر_ختم
}

سانچہ <قسم_نام dst_t>
ساکن خالی dequantize_صف_q6_K_sycl_reorder(ثابت خالی * vx, dst_t * y, ثابت int64_t k, dpct::قطار_ptr stream) {
    ثابت int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
        [=](sycl::nd_شے<3> شے_ct1) { dequantize_بلاک_q6_K_reorder(vx, y + شے_ct1.get_گروہ(2) * QK_K, شے_ct1.get_گروہ(2), شے_ct1, nb); });
}

سانچہ <قسم_نام dst_t>
ساکن خالی dequantize_صف_iq1_s_sycl(ثابت خالی *vx, dst_t *y, ثابت int64_t k,
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

سانچہ <قسم_نام dst_t>
ساکن خالی dequantize_صف_iq1_m_sycl(ثابت خالی *vx, dst_t *y, ثابت int64_t k,
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

سانچہ <قسم_نام dst_t>
ساکن خالی dequantize_صف_iq2_xxs_sycl(ثابت خالی *vx, dst_t *y, ثابت int64_t k,
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

سانچہ <قسم_نام dst_t>
ساکن خالی dequantize_صف_iq2_xs_sycl(ثابت خالی *vx, dst_t *y, ثابت int64_t k,
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

سانچہ <قسم_نام dst_t>
ساکن خالی dequantize_صف_iq2_s_sycl(ثابت خالی *vx, dst_t *y, ثابت int64_t k,
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


سانچہ <قسم_نام dst_t>
ساکن خالی dequantize_صف_iq3_xxs_sycl(ثابت خالی *vx, dst_t *y, ثابت int64_t k,
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

سانچہ <قسم_نام dst_t>
ساکن خالی dequantize_صف_iq3_xxs_sycl_reorder(ثابت خالی * vx, dst_t * y, ثابت int64_t k,
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

سانچہ <قسم_نام dst_t>
ساکن خالی dequantize_صف_iq3_s_sycl(ثابت خالی *vx, dst_t *y, ثابت int64_t k,
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

سانچہ <قسم_نام dst_t>
ساکن خالی dequantize_صف_iq3_s_sycl_reorder(ثابت خالی * vx, dst_t * y, ثابت int64_t k,
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

سانچہ <قسم_نام dst_t>
ساکن خالی dequantize_صف_iq4_xs_sycl(ثابت خالی *vx, dst_t *y, ثابت int64_t k,
                                       dpct::قطار_ptr stream) {
    ثابت int64_t nb = (k + QK_K - 1) / QK_K;
#اگر QK_K == 64
    dequantize_صف_iq4_nl_sycl(vx, y, k, stream);
#ورنہ
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
#اگر_ختم
}

سانچہ <قسم_نام dst_t>
ساکن خالی dequantize_صف_iq4_nl_sycl(ثابت خالی *vx, dst_t *y, ثابت int64_t k,
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

سانچہ <قسم_نام dst_t>
ساکن خالی dequantize_صف_mxfp4_sycl(ثابت خالی * vx, dst_t * y, ثابت int64_t k, dpct::قطار_ptr stream) {
    ثابت صحیح nb = (k + QK_K - 1) / QK_K;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_شے<3> شے_ct1) {
            dequantize_بلاک_mxfp4(vx, y, شے_ct1);
        });
}

سانچہ <قسم_نام dst_t>
ساکن خالی dequantize_صف_mxfp4_sycl_reorder(ثابت خالی * vx, dst_t * y, ثابت int64_t k, dpct::قطار_ptr stream) {
    GGML_ASSERT(k % QK_MXFP4 == 0);
    ثابت صحیح n_warp = (k / QK_MXFP4 + WARP_سائز - 1) / WARP_سائز;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, n_warp) * sycl::range<3>(1, 1, WARP_سائز),
                          sycl::range<3>(1, 1, WARP_سائز)),
        [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
            dequantize_بلاک_mxfp4_reorder(vx, y, k, شے_ct1);
        });
}

سانچہ <قسم_نام dst_t>
ساکن خالی dequantize_صف_nvfp4_sycl(ثابت خالی * vx, dst_t * y, ثابت int64_t k, dpct::قطار_ptr stream) {
    GGML_ASSERT(k % QK_NVFP4 == 0);
    ثابت صحیح nb = k / QK_NVFP4;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_شے<3> /*item_ct1*/) {
            dequantize_بلاک_nvfp4(vx, y, k);
        });
}


سانچہ <صحیح qk, صحیح qr, dequantize_kernel_t dequantize_kernel, قسم_نام dst_t>
ساکن خالی dequantize_بلاک_nc(ثابت خالی * __restrict__ vx, dst_t * __restrict__ y,
        ثابت int64_t ne00, ثابت int64_t ne01, ثابت int64_t ne02,
        ثابت int64_t s01, ثابت int64_t s02, ثابت int64_t s03) {
    خودکار          شے_ct1 = sycl::ext::oneapi::this_work_شے::get_nd_شے<3>();
    ثابت int64_t i00 = 2 * (int64_t(شے_ct1.get_local_range(2)) * شے_ct1.get_گروہ(2) + شے_ct1.get_local_id(2));

    اگر (i00 >= ne00) {
        واپس;
    }

    ثابت int64_t i01 = شے_ct1.get_گروہ(1);
    ثابت int64_t i02 = شے_ct1.get_گروہ(0) % ne02;
    ثابت int64_t i03 = شے_ct1.get_گروہ(0) / ne02;

    ثابت int64_t ibx0 = i03*s03 + i02*s02 + i01*s01;

    ثابت int64_t ib = ibx0 + i00/qk; // block index
    ثابت int64_t iqs = (i00%qk)/qr; // quant index
    ثابت int64_t iybs = i00 - i00%qk; // y block start index
    ثابت int64_t y_انحراف = qr == 1 ? 1 : qk/2;

    // dequantize
    #اگر_تعریف_شدہ GGML_SYCL_F16
        sycl::half2 v;
    #ورنہ
        sycl::float2 v;
    #اگر_ختم

    dequantize_kernel(vx, ib, iqs, v);

    ثابت int64_t iy0 = ((i03*ne02 + i02)*ne01 + i01)*ne00 + iybs + iqs;
    y[iy0 + 0]        = ggml_sycl_cast<dst_t>(v.x());
    y[iy0 + y_انحراف] = ggml_sycl_cast<dst_t>(v.y());
}


سانچہ <صحیح qk, صحیح qr, dequantize_kernel_t dequantize_kernel, قسم_نام dst_t>
ساکن خالی dequantize_بلاک_nc_sycl(ثابت خالی *    vx,
                                  dst_t *         y,
                                  ثابت int64_t   ne00,
                                  ثابت int64_t   ne01,
                                  ثابت int64_t   ne02,
                                  ثابت int64_t   ne03,
                                  ثابت int64_t   s01,
                                  ثابت int64_t   s02,
                                  ثابت int64_t   s03,
                                  dpct::قطار_ptr stream) {
    ثابت dpct::dim3 عدد_بلاکیں((ne00 + 2 * SYCL_DEQUANTIZE_بلاک_سائز - 1) / (2 * SYCL_DEQUANTIZE_بلاک_سائز), ne01,
                                ne02 * ne03);
    stream->parallel_for(sycl::nd_range<3>(عدد_بلاکیں * sycl::range<3>(1, 1, SYCL_DEQUANTIZE_بلاک_سائز),
                                           sycl::range<3>(1, 1, SYCL_DEQUANTIZE_بلاک_سائز)),
                         [=](sycl::nd_شے<3> شے_ct1) {
                             GGML_غیر_مستعمل(شے_ct1);
                             dequantize_بلاک_nc<qk, qr, dequantize_kernel>(vx, y, ne00, ne01, ne02, s01, s02, s03);
                         });
}
سانچہ <قسم_نام src_t, قسم_نام dst_t>
ساکن خالی convert_unary_nc(ثابت خالی * __restrict__ vx, dst_t * __restrict__ y, ثابت int64_t ne00, ثابت int64_t ne01,
                          ثابت int64_t ne02, ثابت int64_t s01, ثابت int64_t s02, ثابت int64_t s03,
                          ثابت sycl::nd_شے<3> & شے_ct1) {

    ثابت int64_t work_گروہ_سائز = شے_ct1.get_local_range(2);
    ثابت int64_t global_id       = شے_ct1.get_local_id(2) + work_گروہ_سائز * شے_ct1.get_گروہ(2);

    ثابت int64_t i01 = شے_ct1.get_گروہ(1);
    ثابت int64_t i02 = شے_ct1.get_گروہ(0) % ne02;
    ثابت int64_t i03 = شے_ct1.get_گروہ(0) / ne02;

    // make each work-item deal with more elements since sycl global range can not exceed max int
    ثابت src_t * x = static_cast<ثابت src_t *>(vx);
    ثابت int64_t ix = i03 * s03 + i02 * s02 + i01 * s01;
    ثابت int64_t iy = ((i03 * ne02 + i02) * ne01 + i01) * ne00;

#ہدایت unroll
    برائے (int64_t i00 = global_id; i00 < ne00; i00 += work_گروہ_سائز * شے_ct1.get_گروہ_range(2)) {
        y[iy + i00] = static_cast<dst_t>(x[ix + i00]);
    }
}

سانچہ <قسم_نام src_t, قسم_نام dst_t>
ساکن خالی convert_unary_nc_sycl(ثابت خالی * __restrict__ vx, dst_t * __restrict__ y,
                                  ثابت int64_t ne00, ثابت int64_t ne01, ثابت int64_t ne02, ثابت int64_t ne03,
                                  ثابت int64_t s01, ثابت int64_t s02, ثابت int64_t s03, dpct::قطار_ptr queue) {
    dpct::has_capability_or_fail(queue->get_device(), { sycl::aspect::fp16 });

    sycl::range<3> global_سائز(ne02 * ne03, ne01, ceil_div(ne00, SYCL_DEQUANTIZE_بلاک_سائز));

    // decrease global range when it exceeds the max int
    // TODO: Downsample logic is separated from the kernel, a rewrite is desirable
    int64_t        downsized_workgroup = downsample_sycl_global_range(global_سائز[0], SYCL_DEQUANTIZE_بلاک_سائز);
    sycl::range<3> workgroup_سائز(1, 1, downsized_workgroup);

    queue->parallel_for(sycl::nd_range<3>(global_سائز * workgroup_سائز, workgroup_سائز), [=](sycl::nd_شے<3> شے_ct1) {
        convert_unary_nc<src_t>(vx, y, ne00, ne01, ne02, s01, s02, s03, شے_ct1);
    });
}

سانچہ <قسم_نام src_t, قسم_نام dst_t>
ساکن خالی convert_unary_sycl(ثابت خالی * vx, dst_t * y, ثابت int64_t k, dpct::قطار_ptr queue) {
    convert_unary_nc_sycl<src_t>(vx, y, k, 1, 1, 1, k, k, k, queue);
}


ساکن منطقی is_reordered_src0(ثابت ggml_tensor * dst) {
    واپس dst->src[0]->extra &&
           ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder;
}

سانچہ <قسم_نام dst_t>
ساکن مقصد_t_sycl_t<dst_t> ggml_get_dequant_sycl(ggml_type type, ggml_tensor * dst) {
    انتخاب (type) {
        حالت GGML_TYPE_Q1_0:
            واپس dequantize_بلاک_sycl<QK1_0, QR1_0, dequantize_q1_0, dst_t>;
        حالت GGML_TYPE_Q2_0:
            واپس dequantize_بلاک_sycl<QK2_0, QR2_0, dequantize_q2_0, dst_t>;
        حالت GGML_TYPE_Q4_0:
            اگر (is_reordered_src0(dst)) {
                واپس dequantize_صف_q4_0_sycl_reorder;
            } ورنہ اگر constexpr (std::is_same_v<dst_t, اعشاری>) {
                واپس dequantize_صف_q4_0_sycl;
            } ورنہ {
                واپس dequantize_بلاک_sycl<QK4_0, QR4_0, dequantize_q4_0, dst_t>;
            }
        حالت GGML_TYPE_Q4_1:
            اگر constexpr (std::is_same_v<dst_t, اعشاری>) {
                واپس dequantize_صف_q4_1_sycl;
            } ورنہ {
                واپس dequantize_بلاک_sycl<QK4_1, QR4_1, dequantize_q4_1, dst_t>;
            }
        حالت GGML_TYPE_Q5_0:
            واپس dequantize_بلاک_sycl<QK5_0, QR5_0, dequantize_q5_0, dst_t>;
        حالت GGML_TYPE_Q5_1:
            واپس dequantize_بلاک_sycl<QK5_1, QR5_1, dequantize_q5_1, dst_t>;
        حالت GGML_TYPE_Q8_0:
            اگر (is_reordered_src0(dst)) {
                واپس dequantize_صف_q8_0_sycl_reorder;
            } ورنہ {
                واپس dequantize_بلاک_sycl<QK8_0, QR8_0, dequantize_q8_0, dst_t>;
            }
        حالت GGML_TYPE_Q2_K:
            اگر (is_reordered_src0(dst)) {
                واپس dequantize_صف_q2_K_sycl_reorder;
            } ورنہ {
                واپس dequantize_صف_q2_K_sycl;
            }
        حالت GGML_TYPE_Q3_K:
            اگر (is_reordered_src0(dst)) {
                واپس dequantize_صف_q3_K_sycl_reorder;
            } ورنہ {
                واپس dequantize_صف_q3_K_sycl;
            }
        حالت GGML_TYPE_Q4_K:
            اگر (is_reordered_src0(dst)) {
                واپس dequantize_صف_q4_K_sycl_reorder;
            } ورنہ {
                واپس dequantize_صف_q4_K_sycl;
            }
        حالت GGML_TYPE_Q5_K:
            اگر (is_reordered_src0(dst)) {
                واپس dequantize_صف_q5_K_sycl_reorder;
            } ورنہ {
                واپس dequantize_صف_q5_K_sycl;
            }
        حالت GGML_TYPE_Q6_K:
            اگر (is_reordered_src0(dst)) {
                واپس dequantize_صف_q6_K_sycl_reorder;
            } ورنہ {
                واپس dequantize_صف_q6_K_sycl;
            }
        حالت GGML_TYPE_IQ1_S:
            واپس dequantize_صف_iq1_s_sycl;
        حالت GGML_TYPE_IQ1_M:
            واپس dequantize_صف_iq1_m_sycl;
        حالت GGML_TYPE_IQ2_XXS:
            واپس dequantize_صف_iq2_xxs_sycl;
        حالت GGML_TYPE_IQ2_XS:
            واپس dequantize_صف_iq2_xs_sycl;
        حالت GGML_TYPE_IQ2_S:
            واپس dequantize_صف_iq2_s_sycl;
        حالت GGML_TYPE_IQ3_XXS:
            اگر (is_reordered_src0(dst)) {
                واپس dequantize_صف_iq3_xxs_sycl_reorder;
            } ورنہ {
                واپس dequantize_صف_iq3_xxs_sycl;
            }
        حالت GGML_TYPE_IQ3_S:
            اگر (is_reordered_src0(dst)) {
                واپس dequantize_صف_iq3_s_sycl_reorder;
            } ورنہ {
                واپس dequantize_صف_iq3_s_sycl;
            }
        حالت GGML_TYPE_IQ4_XS:
            واپس dequantize_صف_iq4_xs_sycl;
        حالت GGML_TYPE_IQ4_NL:
            واپس dequantize_صف_iq4_nl_sycl;
        حالت GGML_TYPE_MXFP4:
            اگر (is_reordered_src0(dst)) {
                واپس dequantize_صف_mxfp4_sycl_reorder;
            }
            واپس dequantize_صف_mxfp4_sycl;
        حالت GGML_TYPE_NVFP4:
            واپس dequantize_صف_nvfp4_sycl;
        حالت GGML_TYPE_F32:
            اگر constexpr (std::is_same_v<dst_t, sycl::half>) {
                واپس convert_unary_sycl<اعشاری>;
            }
            توڑو;
        حالت GGML_TYPE_F16:
            اگر constexpr (std::is_same_v<dst_t, اعشاری>) {
                واپس convert_unary_sycl<sycl::half>;
            }
            توڑو;
#اگر_تعریف_شدہ GGML_SYCL_HAS_BF16
        حالت GGML_TYPE_BF16:
            واپس convert_unary_sycl<sycl::ext::oneapi::bfloat16>;
#اگر_ختم
        طےشدہ:
            توڑو;
    }
    GGML_ABORT("fatal error: unsupport data type=%s\n", ggml_type_name(type));
}

مقصد_fp16_sycl_t ggml_get_مقصد_fp16_sycl(ggml_type type, ggml_tensor * dst) {
    واپس ggml_get_dequant_sycl<sycl::half>(type, dst);
}

مقصد_fp32_sycl_t ggml_get_مقصد_fp32_sycl(ggml_type type, ggml_tensor *dst) {
    واپس ggml_get_dequant_sycl<اعشاری>(type, dst);
}


#اگر_تعریف_شدہ GGML_SYCL_HAS_BF16
مقصد_bf16_sycl_t ggml_get_مقصد_bf16_sycl(ggml_type type, ggml_tensor * /*dst*/) {
    انتخاب (type) {
        حالت GGML_TYPE_F32:
            واپس convert_unary_sycl<اعشاری>;
        حالت GGML_TYPE_F16:
            واپس convert_unary_sycl<sycl::half>;
        حالت GGML_TYPE_BF16:
            واپس convert_unary_sycl<sycl::ext::oneapi::bfloat16>;
        طےشدہ:
            GGML_ABORT("fatal error: unsupport data type=%s\n", ggml_type_name(type));
    }
}
#اگر_ختم

مقصد_fp16_nc_sycl_t ggml_get_مقصد_fp16_nc_sycl(ggml_type type) {
    انتخاب (type) {
        حالت GGML_TYPE_F32:
            واپس convert_unary_nc_sycl<اعشاری>;
#اگر_تعریف_شدہ GGML_SYCL_HAS_BF16
        حالت GGML_TYPE_BF16:
            واپس convert_unary_nc_sycl<sycl::ext::oneapi::bfloat16>;
#اگر_ختم
        حالت GGML_TYPE_Q1_0:
            واپس dequantize_بلاک_nc_sycl<QK1_0, QR1_0, dequantize_q1_0>;
        حالت GGML_TYPE_Q4_0:
            واپس dequantize_بلاک_nc_sycl<QK4_0, QR4_0, dequantize_q4_0>;
        حالت GGML_TYPE_Q4_1:
            واپس dequantize_بلاک_nc_sycl<QK4_1, QR4_1, dequantize_q4_1>;
        حالت GGML_TYPE_Q5_0:
            واپس dequantize_بلاک_nc_sycl<QK5_0, QR5_0, dequantize_q5_0>;
        حالت GGML_TYPE_Q5_1:
            واپس dequantize_بلاک_nc_sycl<QK5_1, QR5_1, dequantize_q5_1>;
        حالت GGML_TYPE_Q8_0:
            واپس dequantize_بلاک_nc_sycl<QK8_0, QR8_0, dequantize_q8_0>;
        طےشدہ:
            واپس خالی_اشاریہ;
    }
}
