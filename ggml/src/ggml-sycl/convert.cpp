#ಅನ್ತರ್ಭಾವ "convert.hpp"
#ಅನ್ತರ್ಭಾವ "dequantize.hpp"
#ಅನ್ತರ್ಭಾವ "presets.hpp"

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ qk, ಪೂರ್ಣಾಂಕ qr, dequantize_kernel_t dequantize_kernel, ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಖಂಡ(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ y, ಸ್ಥಿರಾಂಕ int64_t k,
                             ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {
    ಸ್ಥಿರಾಂಕ int64_t i = 2 * (ಅಂಶ_ct1.get_local_range(2) * ಅಂಶ_ct1.get_ಗುಂಪು(2) +
                       ಅಂಶ_ct1.get_local_id(2));

    ಆದರೆ (i >= k) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ int64_t ib = i/qk; // block index
    ಸ್ಥಿರಾಂಕ int64_t iqs = (i%qk)/qr; // quant index
    ಸ್ಥಿರಾಂಕ int64_t iybs = i - i%qk; // y block start index
    ಸ್ಥಿರಾಂಕ int64_t y_ಸ್ಥಾನಾಂತರ = qr == 1 ? 1 : qk/2;

    // dequantize
    dfloat2 v;
    dequantize_kernel(vx, ib, iqs, v);

    y[iybs + iqs + 0] = v.x();
    y[iybs + iqs + y_ಸ್ಥಾನಾಂತರ] = v.y();
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ qk, ಪೂರ್ಣಾಂಕ qr, dequantize_kernel_t dequantize_kernel, ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಖಂಡ_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vx,
                                  dst_t *__restrict__ y, ಸ್ಥಿರಾಂಕ int64_t k,
                                  dpct::ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ int64_t ಅಂಕೆ_ಖಂಡಗಳು = (k + 2*SYCL_DEQUANTIZE_ಖಂಡ_ಗಾತ್ರ - 1) / (2*SYCL_DEQUANTIZE_ಖಂಡ_ಗಾತ್ರ);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, ಅಂಕೆ_ಖಂಡಗಳು) *
                    sycl::range<3>(1, 1, SYCL_DEQUANTIZE_ಖಂಡ_ಗಾತ್ರ),
                sycl::range<3>(1, 1, SYCL_DEQUANTIZE_ಖಂಡ_ಗಾತ್ರ)),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                dequantize_ಖಂಡ<qk, qr, dequantize_kernel>(vx, y, k, ಅಂಶ_ct1);
            });
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_q2_K_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, dst_t *y, ಸ್ಥಿರಾಂಕ int64_t k,
                                     dpct::ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ int64_t nb = k / QK_K;
#ಆದರೆ QK_K == 256
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 64),
                                               sycl::range<3>(1, 1, 64)),
                             [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                                 dequantize_ಖಂಡ_q2_K(vx, y, ಅಂಶ_ct1);
                             });
    }
#ಇಲ್ಲದಿದ್ದರೆ
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                                 dequantize_ಖಂಡ_q2_K(vx, y, ಅಂಶ_ct1);
                             });
    }

#ಆದರೆ_ಅಂತ್ಯ
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_q2_K_sycl_reorder(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, dst_t *y, ಸ್ಥಿರಾಂಕ int64_t k,
                                             dpct::ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
        [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
            dequantize_ಖಂಡ_q2_K_reorder(vx, y + ಅಂಶ_ct1.get_ಗುಂಪು(2) * QK_K, ಅಂಶ_ct1.get_ಗುಂಪು(2), ಅಂಶ_ct1, nb);
        });
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_q3_K_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, dst_t *y, ಸ್ಥಿರಾಂಕ int64_t k,
                                     dpct::ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ int64_t nb = k / QK_K;
#ಆದರೆ QK_K == 256
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 64),
                                               sycl::range<3>(1, 1, 64)),
                             [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                                 dequantize_ಖಂಡ_q3_K(vx, y, ಅಂಶ_ct1);
                             });
    }
#ಇಲ್ಲದಿದ್ದರೆ
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                                 dequantize_ಖಂಡ_q3_K(vx, y, ಅಂಶ_ct1);
                             });
    }
#ಆದರೆ_ಅಂತ್ಯ
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_q3_K_sycl_reorder(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, dst_t *y, ಸ್ಥಿರಾಂಕ int64_t k,
                                             dpct::ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
        [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
            dequantize_ಖಂಡ_q3_K_reorder(vx, y + ಅಂಶ_ct1.get_ಗುಂಪು(2) * QK_K, ಅಂಶ_ct1.get_ಗುಂಪು(2), ಅಂಶ_ct1, nb);
        });
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_q4_0_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, dst_t *y, ಸ್ಥಿರಾಂಕ int64_t k,
                                     dpct::ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ int64_t nb32 = k / 32;
    ಸ್ಥಿರಾಂಕ int64_t nb = (k + 255) / 256;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                                 dequantize_ಖಂಡ_q4_0(vx, y, nb32, ಅಂಶ_ct1);
                             });
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_q4_0_sycl_reorder(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, dst_t *y, ಸ್ಥಿರಾಂಕ int64_t k,
                                     dpct::ಸರತಿ_ptr stream) {

    dpct::has_capability_or_fail(stream->get_device(),
                                    {sycl::aspect::fp16});

    GGML_ASSERT(k % QK4_0 == 0);
    ಸ್ಥಿರಾಂಕ int64_t n_groups = ceil_div(k / 2, SYCL_DEQUANTIZE_ಖಂಡ_ಗಾತ್ರ);
    // decrease global range when it exceeds the max int; the kernel loops to cover the rest
    ಸ್ಥಿರಾಂಕ int64_t wg_ಗಾತ್ರ  = downsample_sycl_global_range(n_groups, SYCL_DEQUANTIZE_ಖಂಡ_ಗಾತ್ರ);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, n_groups) *
        sycl::range<3>(1, 1, wg_ಗಾತ್ರ),
        sycl::range<3>(1, 1, wg_ಗಾತ್ರ)),
        [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
            dequantize_ಖಂಡ_q4_0_reorder(vx, y, k, ಅಂಶ_ct1);
        });
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_q8_0_sycl_reorder(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, dst_t *y, ಸ್ಥಿರಾಂಕ int64_t k,
                                     dpct::ಸರತಿ_ptr stream) {

    dpct::has_capability_or_fail(stream->get_device(),
                                    {sycl::aspect::fp16});

    // flat range<1>, two pairs (4 elements) per item: no nd_range/group math, one vectorized store.
    // k % QK8_0 == 0 implies k % 4 == 0, so the wide launch covers the row exactly (no tail).
    GGML_ASSERT(k % QK8_0 == 0);
    stream->parallel_for(sycl::range<1>((size_t) (k / 4)),
        [=](sycl::id<1> idx) {
            dequantize_ಖಂಡ_q8_0_reorder_wide(vx, y, k, idx);
        });
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_q4_1_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, dst_t *y, ಸ್ಥಿರಾಂಕ int64_t k,
                                     dpct::ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ int64_t nb32 = k / 32;
    ಸ್ಥಿರಾಂಕ int64_t nb = (k + 255) / 256;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                                 dequantize_ಖಂಡ_q4_1(vx, y, nb32, ಅಂಶ_ct1);
                             });
    }
}


ಮಾದರಿ <ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_q4_K_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, dst_t *y, ಸ್ಥಿರಾಂಕ int64_t k,
                                     dpct::ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ int64_t nb = k / QK_K;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) nb * SYCL_DEQK_WI_ಪ್ರತಿ_ಖಂಡ),
                                           sycl::range<1>((size_t) SYCL_DEQK_WI_ಪ್ರತಿ_ಖಂಡ)),
                         [=](sycl::nd_ಅಂಶ<1> it) { dequantize_ಖಂಡ_q4_K_wide(vx, y, nb, it); });
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_q4_K_sycl_reorder(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, dst_t * y, ಸ್ಥಿರಾಂಕ int64_t k, dpct::ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ int64_t nb = k / QK_K;
    ಸ್ಥಿರಾಂಕ size_t  local_ಗಾತ್ರ  = 32;
    ಸ್ಥಿರಾಂಕ size_t  global_ಗಾತ್ರ = nb * local_ಗಾತ್ರ;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint8_t, 1> scale_local_acc(sycl::range<1>(12), cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_ಗಾತ್ರ), sycl::range<1>(local_ಗಾತ್ರ)),
                         [=](sycl::nd_ಅಂಶ<1> ಅಂಶ_ct1) {
                             dequantize_ಖಂಡ_q4_K_reorder(vx, y + ಅಂಶ_ct1.get_ಗುಂಪು(0) * QK_K, ಅಂಶ_ct1.get_ಗುಂಪು(0), get_ಸೂಚಕ(scale_local_acc), ಅಂಶ_ct1, nb);
                         });
    });
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_q5_K_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, dst_t *y, ಸ್ಥಿರಾಂಕ int64_t k,
                                     dpct::ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ int64_t nb = k / QK_K;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) nb * SYCL_DEQK_WI_ಪ್ರತಿ_ಖಂಡ),
                                           sycl::range<1>((size_t) SYCL_DEQK_WI_ಪ್ರತಿ_ಖಂಡ)),
                         [=](sycl::nd_ಅಂಶ<1> it) { dequantize_ಖಂಡ_q5_K_wide(vx, y, nb, it); });
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_q5_K_sycl_reorder(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, dst_t * y, ಸ್ಥಿರಾಂಕ int64_t k, dpct::ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint8_t, 1> scale_local_acc(sycl::range<1>(K_SCALE_ಗಾತ್ರ), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                dequantize_ಖಂಡ_q5_K_reorder(vx, y + ಅಂಶ_ct1.get_ಗುಂಪು(2) * QK_K, ಅಂಶ_ct1.get_ಗುಂಪು(2), get_ಸೂಚಕ(scale_local_acc), ಅಂಶ_ct1, nb);
            });
    });
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_q6_K_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, dst_t *y, ಸ್ಥಿರಾಂಕ int64_t k,
                                     dpct::ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ int64_t nb = k / QK_K;
#ಆದರೆ QK_K == 256
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 64),
                                               sycl::range<3>(1, 1, 64)),
                             [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                                 dequantize_ಖಂಡ_q6_K(vx, y, ಅಂಶ_ct1);
                             });
    }
#ಇಲ್ಲದಿದ್ದರೆ
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                                 dequantize_ಖಂಡ_q6_K(vx, y, ಅಂಶ_ct1);
                             });
    }

#ಆದರೆ_ಅಂತ್ಯ
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_q6_K_sycl_reorder(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, dst_t * y, ಸ್ಥಿರಾಂಕ int64_t k, dpct::ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
        [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) { dequantize_ಖಂಡ_q6_K_reorder(vx, y + ಅಂಶ_ct1.get_ಗುಂಪು(2) * QK_K, ಅಂಶ_ct1.get_ಗುಂಪು(2), ಅಂಶ_ct1, nb); });
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_iq1_s_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, dst_t *y, ಸ್ಥಿರಾಂಕ int64_t k,
                                        dpct::ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                                 dequantize_ಖಂಡ_iq1_s(
                                     vx, y, ಅಂಶ_ct1, iq1s_grid_gpu
                                     );
                             });
        });
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_iq1_m_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, dst_t *y, ಸ್ಥಿರಾಂಕ int64_t k,
                                        dpct::ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                                 dequantize_ಖಂಡ_iq1_m(
                                     vx, y, ಅಂಶ_ct1, iq1s_grid_gpu
                                     );
                             });
        });
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_iq2_xxs_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, dst_t *y, ಸ್ಥಿರಾಂಕ int64_t k,
                                        dpct::ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                                 dequantize_ಖಂಡ_iq2_xxs(
                                     vx, y, ಅಂಶ_ct1, iq2xxs_grid,
                                     ksigns_iq2xs, kmask_iq2xs);
                             });
        });
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_iq2_xs_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, dst_t *y, ಸ್ಥಿರಾಂಕ int64_t k,
                                       dpct::ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                                 dequantize_ಖಂಡ_iq2_xs(
                                     vx, y, ಅಂಶ_ct1, iq2xs_grid,
                                     ksigns_iq2xs, kmask_iq2xs);
                             });
        });
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_iq2_s_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, dst_t *y, ಸ್ಥಿರಾಂಕ int64_t k,
                                      dpct::ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                                 dequantize_ಖಂಡ_iq2_s(vx, y, ಅಂಶ_ct1);
                             });
        });
    }
}


ಮಾದರಿ <ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_iq3_xxs_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, dst_t *y, ಸ್ಥಿರಾಂಕ int64_t k,
                                        dpct::ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                                 dequantize_ಖಂಡ_iq3_xxs(
                                     vx, y, ಅಂಶ_ct1, iq3xxs_grid,
                                     ksigns_iq2xs, kmask_iq2xs);
                             });
        });
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_iq3_xxs_sycl_reorder(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, dst_t * y, ಸ್ಥಿರಾಂಕ int64_t k,
                                                dpct::ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ int64_t nb = k / QK_K;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
            dequantize_ಖಂಡ_iq3_xxs_reorder(
                vx, y + ಅಂಶ_ct1.get_ಗುಂಪು(2) * QK_K, ಅಂಶ_ct1.get_ಗುಂಪು(2), ಅಂಶ_ct1, nb,
                iq3xxs_grid, ksigns_iq2xs, kmask_iq2xs);
        });
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_iq3_s_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, dst_t *y, ಸ್ಥಿರಾಂಕ int64_t k,
                                        dpct::ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                                 dequantize_ಖಂಡ_iq3_s(
                                     vx, y, ಅಂಶ_ct1, kmask_iq2xs, iq3s_grid);
                             });
        });
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_iq3_s_sycl_reorder(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, dst_t * y, ಸ್ಥಿರಾಂಕ int64_t k,
                                              dpct::ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ int64_t nb = k / QK_K;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
            dequantize_ಖಂಡ_iq3_s_reorder(vx, y + ಅಂಶ_ct1.get_ಗುಂಪು(2) * QK_K, ಅಂಶ_ct1.get_ಗುಂಪು(2),
                                           ಅಂಶ_ct1, nb, kmask_iq2xs, iq3s_grid);
        });
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_iq4_xs_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, dst_t *y, ಸ್ಥಿರಾಂಕ int64_t k,
                                       dpct::ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ int64_t nb = (k + QK_K - 1) / QK_K;
#ಆದರೆ QK_K == 64
    dequantize_ಅಡ್ಡಸಾಲು_iq4_nl_sycl(vx, y, k, stream);
#ಇಲ್ಲದಿದ್ದರೆ
      {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                  cgh.parallel_for(
                      sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                            sycl::range<3>(1, 1, 32),
                                        sycl::range<3>(1, 1, 32)),
                      [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                            dequantize_ಖಂಡ_iq4_xs(vx, y, ಅಂಶ_ct1);
                      });
            });
      }
#ಆದರೆ_ಅಂತ್ಯ
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_iq4_nl_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, dst_t *y, ಸ್ಥಿರಾಂಕ int64_t k,
                                       dpct::ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ int64_t nb = (k + QK_K - 1) / QK_K;
      {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                  cgh.parallel_for(
                      sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                            sycl::range<3>(1, 1, 32),
                                        sycl::range<3>(1, 1, 32)),
                      [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                            dequantize_ಖಂಡ_iq4_nl(vx, y, ಅಂಶ_ct1);
                      });
            });
      }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_mxfp4_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, dst_t * y, ಸ್ಥಿರಾಂಕ int64_t k, dpct::ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = (k + QK_K - 1) / QK_K;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
            dequantize_ಖಂಡ_mxfp4(vx, y, ಅಂಶ_ct1);
        });
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_mxfp4_sycl_reorder(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, dst_t * y, ಸ್ಥಿರಾಂಕ int64_t k, dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(k % QK_MXFP4 == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_warp = (k / QK_MXFP4 + WARP_ಗಾತ್ರ - 1) / WARP_ಗಾತ್ರ;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, n_warp) * sycl::range<3>(1, 1, WARP_ಗಾತ್ರ),
                          sycl::range<3>(1, 1, WARP_ಗಾತ್ರ)),
        [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
            dequantize_ಖಂಡ_mxfp4_reorder(vx, y, k, ಅಂಶ_ct1);
        });
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_nvfp4_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, dst_t * y, ಸ್ಥಿರಾಂಕ int64_t k, dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(k % QK_NVFP4 == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = k / QK_NVFP4;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_ಅಂಶ<3> /*item_ct1*/) {
            dequantize_ಖಂಡ_nvfp4(vx, y, k);
        });
}


ಮಾದರಿ <ಪೂರ್ಣಾಂಕ qk, ಪೂರ್ಣಾಂಕ qr, dequantize_kernel_t dequantize_kernel, ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಖಂಡ_nc(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ y,
        ಸ್ಥಿರಾಂಕ int64_t ne00, ಸ್ಥಿರಾಂಕ int64_t ne01, ಸ್ಥಿರಾಂಕ int64_t ne02,
        ಸ್ಥಿರಾಂಕ int64_t s01, ಸ್ಥಿರಾಂಕ int64_t s02, ಸ್ಥಿರಾಂಕ int64_t s03) {
    ಸ್ವಯಂಚಾಲಿತ          ಅಂಶ_ct1 = sycl::ext::oneapi::this_work_ಅಂಶ::get_nd_ಅಂಶ<3>();
    ಸ್ಥಿರಾಂಕ int64_t i00 = 2 * (int64_t(ಅಂಶ_ct1.get_local_range(2)) * ಅಂಶ_ct1.get_ಗುಂಪು(2) + ಅಂಶ_ct1.get_local_id(2));

    ಆದರೆ (i00 >= ne00) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ int64_t i01 = ಅಂಶ_ct1.get_ಗುಂಪು(1);
    ಸ್ಥಿರಾಂಕ int64_t i02 = ಅಂಶ_ct1.get_ಗುಂಪು(0) % ne02;
    ಸ್ಥಿರಾಂಕ int64_t i03 = ಅಂಶ_ct1.get_ಗುಂಪು(0) / ne02;

    ಸ್ಥಿರಾಂಕ int64_t ibx0 = i03*s03 + i02*s02 + i01*s01;

    ಸ್ಥಿರಾಂಕ int64_t ib = ibx0 + i00/qk; // block index
    ಸ್ಥಿರಾಂಕ int64_t iqs = (i00%qk)/qr; // quant index
    ಸ್ಥಿರಾಂಕ int64_t iybs = i00 - i00%qk; // y block start index
    ಸ್ಥಿರಾಂಕ int64_t y_ಸ್ಥಾನಾಂತರ = qr == 1 ? 1 : qk/2;

    // dequantize
    #ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_F16
        sycl::half2 v;
    #ಇಲ್ಲದಿದ್ದರೆ
        sycl::float2 v;
    #ಆದರೆ_ಅಂತ್ಯ

    dequantize_kernel(vx, ib, iqs, v);

    ಸ್ಥಿರಾಂಕ int64_t iy0 = ((i03*ne02 + i02)*ne01 + i01)*ne00 + iybs + iqs;
    y[iy0 + 0]        = ggml_sycl_cast<dst_t>(v.x());
    y[iy0 + y_ಸ್ಥಾನಾಂತರ] = ggml_sycl_cast<dst_t>(v.y());
}


ಮಾದರಿ <ಪೂರ್ಣಾಂಕ qk, ಪೂರ್ಣಾಂಕ qr, dequantize_kernel_t dequantize_kernel, ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಖಂಡ_nc_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *    vx,
                                  dst_t *         y,
                                  ಸ್ಥಿರಾಂಕ int64_t   ne00,
                                  ಸ್ಥಿರಾಂಕ int64_t   ne01,
                                  ಸ್ಥಿರಾಂಕ int64_t   ne02,
                                  ಸ್ಥಿರಾಂಕ int64_t   ne03,
                                  ಸ್ಥಿರಾಂಕ int64_t   s01,
                                  ಸ್ಥಿರಾಂಕ int64_t   s02,
                                  ಸ್ಥಿರಾಂಕ int64_t   s03,
                                  dpct::ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ dpct::dim3 ಅಂಕೆ_ಖಂಡಗಳು((ne00 + 2 * SYCL_DEQUANTIZE_ಖಂಡ_ಗಾತ್ರ - 1) / (2 * SYCL_DEQUANTIZE_ಖಂಡ_ಗಾತ್ರ), ne01,
                                ne02 * ne03);
    stream->parallel_for(sycl::nd_range<3>(ಅಂಕೆ_ಖಂಡಗಳು * sycl::range<3>(1, 1, SYCL_DEQUANTIZE_ಖಂಡ_ಗಾತ್ರ),
                                           sycl::range<3>(1, 1, SYCL_DEQUANTIZE_ಖಂಡ_ಗಾತ್ರ)),
                         [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                             GGML_ಬಳಸದ(ಅಂಶ_ct1);
                             dequantize_ಖಂಡ_nc<qk, qr, dequantize_kernel>(vx, y, ne00, ne01, ne02, s01, s02, s03);
                         });
}
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು src_t, ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ convert_unary_nc(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ y, ಸ್ಥಿರಾಂಕ int64_t ne00, ಸ್ಥಿರಾಂಕ int64_t ne01,
                          ಸ್ಥಿರಾಂಕ int64_t ne02, ಸ್ಥಿರಾಂಕ int64_t s01, ಸ್ಥಿರಾಂಕ int64_t s02, ಸ್ಥಿರಾಂಕ int64_t s03,
                          ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> & ಅಂಶ_ct1) {

    ಸ್ಥಿರಾಂಕ int64_t work_ಗುಂಪು_ಗಾತ್ರ = ಅಂಶ_ct1.get_local_range(2);
    ಸ್ಥಿರಾಂಕ int64_t global_id       = ಅಂಶ_ct1.get_local_id(2) + work_ಗುಂಪು_ಗಾತ್ರ * ಅಂಶ_ct1.get_ಗುಂಪು(2);

    ಸ್ಥಿರಾಂಕ int64_t i01 = ಅಂಶ_ct1.get_ಗುಂಪು(1);
    ಸ್ಥಿರಾಂಕ int64_t i02 = ಅಂಶ_ct1.get_ಗುಂಪು(0) % ne02;
    ಸ್ಥಿರಾಂಕ int64_t i03 = ಅಂಶ_ct1.get_ಗುಂಪು(0) / ne02;

    // make each work-item deal with more elements since sycl global range can not exceed max int
    ಸ್ಥಿರಾಂಕ src_t * x = static_cast<ಸ್ಥಿರಾಂಕ src_t *>(vx);
    ಸ್ಥಿರಾಂಕ int64_t ix = i03 * s03 + i02 * s02 + i01 * s01;
    ಸ್ಥಿರಾಂಕ int64_t iy = ((i03 * ne02 + i02) * ne01 + i01) * ne00;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (int64_t i00 = global_id; i00 < ne00; i00 += work_ಗುಂಪು_ಗಾತ್ರ * ಅಂಶ_ct1.get_ಗುಂಪು_range(2)) {
        y[iy + i00] = static_cast<dst_t>(x[ix + i00]);
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು src_t, ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ convert_unary_nc_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ y,
                                  ಸ್ಥಿರಾಂಕ int64_t ne00, ಸ್ಥಿರಾಂಕ int64_t ne01, ಸ್ಥಿರಾಂಕ int64_t ne02, ಸ್ಥಿರಾಂಕ int64_t ne03,
                                  ಸ್ಥಿರಾಂಕ int64_t s01, ಸ್ಥಿರಾಂಕ int64_t s02, ಸ್ಥಿರಾಂಕ int64_t s03, dpct::ಸರತಿ_ptr queue) {
    dpct::has_capability_or_fail(queue->get_device(), { sycl::aspect::fp16 });

    sycl::range<3> global_ಗಾತ್ರ(ne02 * ne03, ne01, ceil_div(ne00, SYCL_DEQUANTIZE_ಖಂಡ_ಗಾತ್ರ));

    // decrease global range when it exceeds the max int
    // TODO: Downsample logic is separated from the kernel, a rewrite is desirable
    int64_t        downsized_workgroup = downsample_sycl_global_range(global_ಗಾತ್ರ[0], SYCL_DEQUANTIZE_ಖಂಡ_ಗಾತ್ರ);
    sycl::range<3> workgroup_ಗಾತ್ರ(1, 1, downsized_workgroup);

    queue->parallel_for(sycl::nd_range<3>(global_ಗಾತ್ರ * workgroup_ಗಾತ್ರ, workgroup_ಗಾತ್ರ), [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
        convert_unary_nc<src_t>(vx, y, ne00, ne01, ne02, s01, s02, s03, ಅಂಶ_ct1);
    });
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು src_t, ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ convert_unary_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, dst_t * y, ಸ್ಥಿರಾಂಕ int64_t k, dpct::ಸರತಿ_ptr queue) {
    convert_unary_nc_sycl<src_t>(vx, y, k, 1, 1, 1, k, k, k, queue);
}


ಸ್ಥಿರ ತಾರ್ಕಿಕ is_reordered_src0(ಸ್ಥಿರಾಂಕ ggml_tensor * dst) {
    ಹಿಂತಿರುಗಿ dst->src[0]->extra &&
           ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder;
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಗಮ್ಯ_t_sycl_t<dst_t> ggml_get_dequant_sycl(ggml_type type, ggml_tensor * dst) {
    ಆಯ್ಕೆ (type) {
        ಸ್ಥಿತಿ GGML_TYPE_Q1_0:
            ಹಿಂತಿರುಗಿ dequantize_ಖಂಡ_sycl<QK1_0, QR1_0, dequantize_q1_0, dst_t>;
        ಸ್ಥಿತಿ GGML_TYPE_Q2_0:
            ಹಿಂತಿರುಗಿ dequantize_ಖಂಡ_sycl<QK2_0, QR2_0, dequantize_q2_0, dst_t>;
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
            ಆದರೆ (is_reordered_src0(dst)) {
                ಹಿಂತಿರುಗಿ dequantize_ಅಡ್ಡಸಾಲು_q4_0_sycl_reorder;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<dst_t, ದಶಮಾಂಶ>) {
                ಹಿಂತಿರುಗಿ dequantize_ಅಡ್ಡಸಾಲು_q4_0_sycl;
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಹಿಂತಿರುಗಿ dequantize_ಖಂಡ_sycl<QK4_0, QR4_0, dequantize_q4_0, dst_t>;
            }
        ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
            ಆದರೆ constexpr (std::is_same_v<dst_t, ದಶಮಾಂಶ>) {
                ಹಿಂತಿರುಗಿ dequantize_ಅಡ್ಡಸಾಲು_q4_1_sycl;
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಹಿಂತಿರುಗಿ dequantize_ಖಂಡ_sycl<QK4_1, QR4_1, dequantize_q4_1, dst_t>;
            }
        ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
            ಹಿಂತಿರುಗಿ dequantize_ಖಂಡ_sycl<QK5_0, QR5_0, dequantize_q5_0, dst_t>;
        ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
            ಹಿಂತಿರುಗಿ dequantize_ಖಂಡ_sycl<QK5_1, QR5_1, dequantize_q5_1, dst_t>;
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
            ಆದರೆ (is_reordered_src0(dst)) {
                ಹಿಂತಿರುಗಿ dequantize_ಅಡ್ಡಸಾಲು_q8_0_sycl_reorder;
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಹಿಂತಿರುಗಿ dequantize_ಖಂಡ_sycl<QK8_0, QR8_0, dequantize_q8_0, dst_t>;
            }
        ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
            ಆದರೆ (is_reordered_src0(dst)) {
                ಹಿಂತಿರುಗಿ dequantize_ಅಡ್ಡಸಾಲು_q2_K_sycl_reorder;
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಹಿಂತಿರುಗಿ dequantize_ಅಡ್ಡಸಾಲು_q2_K_sycl;
            }
        ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
            ಆದರೆ (is_reordered_src0(dst)) {
                ಹಿಂತಿರುಗಿ dequantize_ಅಡ್ಡಸಾಲು_q3_K_sycl_reorder;
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಹಿಂತಿರುಗಿ dequantize_ಅಡ್ಡಸಾಲು_q3_K_sycl;
            }
        ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
            ಆದರೆ (is_reordered_src0(dst)) {
                ಹಿಂತಿರುಗಿ dequantize_ಅಡ್ಡಸಾಲು_q4_K_sycl_reorder;
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಹಿಂತಿರುಗಿ dequantize_ಅಡ್ಡಸಾಲು_q4_K_sycl;
            }
        ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
            ಆದರೆ (is_reordered_src0(dst)) {
                ಹಿಂತಿರುಗಿ dequantize_ಅಡ್ಡಸಾಲು_q5_K_sycl_reorder;
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಹಿಂತಿರುಗಿ dequantize_ಅಡ್ಡಸಾಲು_q5_K_sycl;
            }
        ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
            ಆದರೆ (is_reordered_src0(dst)) {
                ಹಿಂತಿರುಗಿ dequantize_ಅಡ್ಡಸಾಲು_q6_K_sycl_reorder;
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಹಿಂತಿರುಗಿ dequantize_ಅಡ್ಡಸಾಲು_q6_K_sycl;
            }
        ಸ್ಥಿತಿ GGML_TYPE_IQ1_S:
            ಹಿಂತಿರುಗಿ dequantize_ಅಡ್ಡಸಾಲು_iq1_s_sycl;
        ಸ್ಥಿತಿ GGML_TYPE_IQ1_M:
            ಹಿಂತಿರುಗಿ dequantize_ಅಡ್ಡಸಾಲು_iq1_m_sycl;
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XXS:
            ಹಿಂತಿರುಗಿ dequantize_ಅಡ್ಡಸಾಲು_iq2_xxs_sycl;
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XS:
            ಹಿಂತಿರುಗಿ dequantize_ಅಡ್ಡಸಾಲು_iq2_xs_sycl;
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_S:
            ಹಿಂತಿರುಗಿ dequantize_ಅಡ್ಡಸಾಲು_iq2_s_sycl;
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_XXS:
            ಆದರೆ (is_reordered_src0(dst)) {
                ಹಿಂತಿರುಗಿ dequantize_ಅಡ್ಡಸಾಲು_iq3_xxs_sycl_reorder;
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಹಿಂತಿರುಗಿ dequantize_ಅಡ್ಡಸಾಲು_iq3_xxs_sycl;
            }
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_S:
            ಆದರೆ (is_reordered_src0(dst)) {
                ಹಿಂತಿರುಗಿ dequantize_ಅಡ್ಡಸಾಲು_iq3_s_sycl_reorder;
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಹಿಂತಿರುಗಿ dequantize_ಅಡ್ಡಸಾಲು_iq3_s_sycl;
            }
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_XS:
            ಹಿಂತಿರುಗಿ dequantize_ಅಡ್ಡಸಾಲು_iq4_xs_sycl;
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL:
            ಹಿಂತಿರುಗಿ dequantize_ಅಡ್ಡಸಾಲು_iq4_nl_sycl;
        ಸ್ಥಿತಿ GGML_TYPE_MXFP4:
            ಆದರೆ (is_reordered_src0(dst)) {
                ಹಿಂತಿರುಗಿ dequantize_ಅಡ್ಡಸಾಲು_mxfp4_sycl_reorder;
            }
            ಹಿಂತಿರುಗಿ dequantize_ಅಡ್ಡಸಾಲು_mxfp4_sycl;
        ಸ್ಥಿತಿ GGML_TYPE_NVFP4:
            ಹಿಂತಿರುಗಿ dequantize_ಅಡ್ಡಸಾಲು_nvfp4_sycl;
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            ಆದರೆ constexpr (std::is_same_v<dst_t, sycl::half>) {
                ಹಿಂತಿರುಗಿ convert_unary_sycl<ದಶಮಾಂಶ>;
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            ಆದರೆ constexpr (std::is_same_v<dst_t, ದಶಮಾಂಶ>) {
                ಹಿಂತಿರುಗಿ convert_unary_sycl<sycl::half>;
            }
            ನಿಲ್ಲಿಸು;
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_HAS_BF16
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            ಹಿಂತಿರುಗಿ convert_unary_sycl<sycl::ext::oneapi::bfloat16>;
#ಆದರೆ_ಅಂತ್ಯ
        ಪೂರ್ವನಿಯೋಜಿತ:
            ನಿಲ್ಲಿಸು;
    }
    GGML_ABORT("fatal error: unsupport data type=%s\n", ggml_type_name(type));
}

ಗಮ್ಯ_fp16_sycl_t ggml_get_ಗಮ್ಯ_fp16_sycl(ggml_type type, ggml_tensor * dst) {
    ಹಿಂತಿರುಗಿ ggml_get_dequant_sycl<sycl::half>(type, dst);
}

ಗಮ್ಯ_fp32_sycl_t ggml_get_ಗಮ್ಯ_fp32_sycl(ggml_type type, ggml_tensor *dst) {
    ಹಿಂತಿರುಗಿ ggml_get_dequant_sycl<ದಶಮಾಂಶ>(type, dst);
}


#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_HAS_BF16
ಗಮ್ಯ_bf16_sycl_t ggml_get_ಗಮ್ಯ_bf16_sycl(ggml_type type, ggml_tensor * /*dst*/) {
    ಆಯ್ಕೆ (type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            ಹಿಂತಿರುಗಿ convert_unary_sycl<ದಶಮಾಂಶ>;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            ಹಿಂತಿರುಗಿ convert_unary_sycl<sycl::half>;
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            ಹಿಂತಿರುಗಿ convert_unary_sycl<sycl::ext::oneapi::bfloat16>;
        ಪೂರ್ವನಿಯೋಜಿತ:
            GGML_ABORT("fatal error: unsupport data type=%s\n", ggml_type_name(type));
    }
}
#ಆದರೆ_ಅಂತ್ಯ

ಗಮ್ಯ_fp16_nc_sycl_t ggml_get_ಗಮ್ಯ_fp16_nc_sycl(ggml_type type) {
    ಆಯ್ಕೆ (type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            ಹಿಂತಿರುಗಿ convert_unary_nc_sycl<ದಶಮಾಂಶ>;
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_HAS_BF16
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            ಹಿಂತಿರುಗಿ convert_unary_nc_sycl<sycl::ext::oneapi::bfloat16>;
#ಆದರೆ_ಅಂತ್ಯ
        ಸ್ಥಿತಿ GGML_TYPE_Q1_0:
            ಹಿಂತಿರುಗಿ dequantize_ಖಂಡ_nc_sycl<QK1_0, QR1_0, dequantize_q1_0>;
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
            ಹಿಂತಿರುಗಿ dequantize_ಖಂಡ_nc_sycl<QK4_0, QR4_0, dequantize_q4_0>;
        ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
            ಹಿಂತಿರುಗಿ dequantize_ಖಂಡ_nc_sycl<QK4_1, QR4_1, dequantize_q4_1>;
        ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
            ಹಿಂತಿರುಗಿ dequantize_ಖಂಡ_nc_sycl<QK5_0, QR5_0, dequantize_q5_0>;
        ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
            ಹಿಂತಿರುಗಿ dequantize_ಖಂಡ_nc_sycl<QK5_1, QR5_1, dequantize_q5_1>;
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
            ಹಿಂತಿರುಗಿ dequantize_ಖಂಡ_nc_sycl<QK8_0, QR8_0, dequantize_q8_0>;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
}
