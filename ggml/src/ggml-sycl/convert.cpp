#ਅਨ੍ਤਰ੍ਭਾਵ "convert.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "dequantize.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "presets.hpp"

ਸਾਂਚਾ <ਪੂਰਨ qk, ਪੂਰਨ qr, dequantize_kernel_t dequantize_kernel, ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਖੰਡ(ਅਚਲ ਖਾਲੀ * __restrict__ vx, dst_t * __restrict__ y, ਅਚਲ int64_t k,
                             ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {
    ਅਚਲ int64_t i = 2 * (ਵਸਤੂ_ct1.get_local_range(2) * ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) +
                       ਵਸਤੂ_ct1.get_local_id(2));

    ਜੇ (i >= k) {
        ਵਾਪਸ;
    }

    ਅਚਲ int64_t ib = i/qk; // block index
    ਅਚਲ int64_t iqs = (i%qk)/qr; // quant index
    ਅਚਲ int64_t iybs = i - i%qk; // y block start index
    ਅਚਲ int64_t y_ਵਿੱਥ = qr == 1 ? 1 : qk/2;

    // dequantize
    dfloat2 v;
    dequantize_kernel(vx, ib, iqs, v);

    y[iybs + iqs + 0] = v.x();
    y[iybs + iqs + y_ਵਿੱਥ] = v.y();
}

ਸਾਂਚਾ <ਪੂਰਨ qk, ਪੂਰਨ qr, dequantize_kernel_t dequantize_kernel, ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਖੰਡ_sycl(ਅਚਲ ਖਾਲੀ *__restrict__ vx,
                                  dst_t *__restrict__ y, ਅਚਲ int64_t k,
                                  dpct::ਕਤਾਰ_ptr stream) {
    ਅਚਲ int64_t ਅੰਕ_ਖੰਡਾਂ = (k + 2*SYCL_DEQUANTIZE_ਖੰਡ_ਮਾਪ - 1) / (2*SYCL_DEQUANTIZE_ਖੰਡ_ਮਾਪ);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) *
                    sycl::range<3>(1, 1, SYCL_DEQUANTIZE_ਖੰਡ_ਮਾਪ),
                sycl::range<3>(1, 1, SYCL_DEQUANTIZE_ਖੰਡ_ਮਾਪ)),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                dequantize_ਖੰਡ<qk, qr, dequantize_kernel>(vx, y, k, ਵਸਤੂ_ct1);
            });
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਪੰਕਤੀ_q2_K_sycl(ਅਚਲ ਖਾਲੀ *vx, dst_t *y, ਅਚਲ int64_t k,
                                     dpct::ਕਤਾਰ_ptr stream) {
    ਅਚਲ int64_t nb = k / QK_K;
#ਜੇ QK_K == 256
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 64),
                                               sycl::range<3>(1, 1, 64)),
                             [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                                 dequantize_ਖੰਡ_q2_K(vx, y, ਵਸਤੂ_ct1);
                             });
    }
#ਨਹੀਂਤਾਂ
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                                 dequantize_ਖੰਡ_q2_K(vx, y, ਵਸਤੂ_ct1);
                             });
    }

#ਜੇ_ਅੰਤ
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਪੰਕਤੀ_q2_K_sycl_reorder(ਅਚਲ ਖਾਲੀ *vx, dst_t *y, ਅਚਲ int64_t k,
                                             dpct::ਕਤਾਰ_ptr stream) {
    ਅਚਲ int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
            dequantize_ਖੰਡ_q2_K_reorder(vx, y + ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * QK_K, ਵਸਤੂ_ct1.get_ਸਮੂਹ(2), ਵਸਤੂ_ct1, nb);
        });
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਪੰਕਤੀ_q3_K_sycl(ਅਚਲ ਖਾਲੀ *vx, dst_t *y, ਅਚਲ int64_t k,
                                     dpct::ਕਤਾਰ_ptr stream) {
    ਅਚਲ int64_t nb = k / QK_K;
#ਜੇ QK_K == 256
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 64),
                                               sycl::range<3>(1, 1, 64)),
                             [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                                 dequantize_ਖੰਡ_q3_K(vx, y, ਵਸਤੂ_ct1);
                             });
    }
#ਨਹੀਂਤਾਂ
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                                 dequantize_ਖੰਡ_q3_K(vx, y, ਵਸਤੂ_ct1);
                             });
    }
#ਜੇ_ਅੰਤ
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਪੰਕਤੀ_q3_K_sycl_reorder(ਅਚਲ ਖਾਲੀ *vx, dst_t *y, ਅਚਲ int64_t k,
                                             dpct::ਕਤਾਰ_ptr stream) {
    ਅਚਲ int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
            dequantize_ਖੰਡ_q3_K_reorder(vx, y + ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * QK_K, ਵਸਤੂ_ct1.get_ਸਮੂਹ(2), ਵਸਤੂ_ct1, nb);
        });
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਪੰਕਤੀ_q4_0_sycl(ਅਚਲ ਖਾਲੀ *vx, dst_t *y, ਅਚਲ int64_t k,
                                     dpct::ਕਤਾਰ_ptr stream) {
    ਅਚਲ int64_t nb32 = k / 32;
    ਅਚਲ int64_t nb = (k + 255) / 256;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                                 dequantize_ਖੰਡ_q4_0(vx, y, nb32, ਵਸਤੂ_ct1);
                             });
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਪੰਕਤੀ_q4_0_sycl_reorder(ਅਚਲ ਖਾਲੀ *vx, dst_t *y, ਅਚਲ int64_t k,
                                     dpct::ਕਤਾਰ_ptr stream) {

    dpct::has_capability_or_fail(stream->get_device(),
                                    {sycl::aspect::fp16});

    GGML_ASSERT(k % QK4_0 == 0);
    ਅਚਲ int64_t n_groups = ceil_div(k / 2, SYCL_DEQUANTIZE_ਖੰਡ_ਮਾਪ);
    // decrease global range when it exceeds the max int; the kernel loops to cover the rest
    ਅਚਲ int64_t wg_ਮਾਪ  = downsample_sycl_global_range(n_groups, SYCL_DEQUANTIZE_ਖੰਡ_ਮਾਪ);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, n_groups) *
        sycl::range<3>(1, 1, wg_ਮਾਪ),
        sycl::range<3>(1, 1, wg_ਮਾਪ)),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
            dequantize_ਖੰਡ_q4_0_reorder(vx, y, k, ਵਸਤੂ_ct1);
        });
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਪੰਕਤੀ_q8_0_sycl_reorder(ਅਚਲ ਖਾਲੀ *vx, dst_t *y, ਅਚਲ int64_t k,
                                     dpct::ਕਤਾਰ_ptr stream) {

    dpct::has_capability_or_fail(stream->get_device(),
                                    {sycl::aspect::fp16});

    // flat range<1>, two pairs (4 elements) per item: no nd_range/group math, one vectorized store.
    // k % QK8_0 == 0 implies k % 4 == 0, so the wide launch covers the row exactly (no tail).
    GGML_ASSERT(k % QK8_0 == 0);
    stream->parallel_for(sycl::range<1>((size_t) (k / 4)),
        [=](sycl::id<1> idx) {
            dequantize_ਖੰਡ_q8_0_reorder_wide(vx, y, k, idx);
        });
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਪੰਕਤੀ_q4_1_sycl(ਅਚਲ ਖਾਲੀ *vx, dst_t *y, ਅਚਲ int64_t k,
                                     dpct::ਕਤਾਰ_ptr stream) {
    ਅਚਲ int64_t nb32 = k / 32;
    ਅਚਲ int64_t nb = (k + 255) / 256;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                                 dequantize_ਖੰਡ_q4_1(vx, y, nb32, ਵਸਤੂ_ct1);
                             });
    }
}


ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਪੰਕਤੀ_q4_K_sycl(ਅਚਲ ਖਾਲੀ *vx, dst_t *y, ਅਚਲ int64_t k,
                                     dpct::ਕਤਾਰ_ptr stream) {
    ਅਚਲ int64_t nb = k / QK_K;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) nb * SYCL_DEQK_WI_ਪ੍ਰਤੀ_ਖੰਡ),
                                           sycl::range<1>((size_t) SYCL_DEQK_WI_ਪ੍ਰਤੀ_ਖੰਡ)),
                         [=](sycl::nd_ਵਸਤੂ<1> it) { dequantize_ਖੰਡ_q4_K_wide(vx, y, nb, it); });
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਪੰਕਤੀ_q4_K_sycl_reorder(ਅਚਲ ਖਾਲੀ * vx, dst_t * y, ਅਚਲ int64_t k, dpct::ਕਤਾਰ_ptr stream) {
    ਅਚਲ int64_t nb = k / QK_K;
    ਅਚਲ size_t  local_ਮਾਪ  = 32;
    ਅਚਲ size_t  global_ਮਾਪ = nb * local_ਮਾਪ;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint8_t, 1> scale_local_acc(sycl::range<1>(12), cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_ਮਾਪ), sycl::range<1>(local_ਮਾਪ)),
                         [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ_ct1) {
                             dequantize_ਖੰਡ_q4_K_reorder(vx, y + ਵਸਤੂ_ct1.get_ਸਮੂਹ(0) * QK_K, ਵਸਤੂ_ct1.get_ਸਮੂਹ(0), get_ਸੂਚਕ(scale_local_acc), ਵਸਤੂ_ct1, nb);
                         });
    });
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਪੰਕਤੀ_q5_K_sycl(ਅਚਲ ਖਾਲੀ *vx, dst_t *y, ਅਚਲ int64_t k,
                                     dpct::ਕਤਾਰ_ptr stream) {
    ਅਚਲ int64_t nb = k / QK_K;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) nb * SYCL_DEQK_WI_ਪ੍ਰਤੀ_ਖੰਡ),
                                           sycl::range<1>((size_t) SYCL_DEQK_WI_ਪ੍ਰਤੀ_ਖੰਡ)),
                         [=](sycl::nd_ਵਸਤੂ<1> it) { dequantize_ਖੰਡ_q5_K_wide(vx, y, nb, it); });
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਪੰਕਤੀ_q5_K_sycl_reorder(ਅਚਲ ਖਾਲੀ * vx, dst_t * y, ਅਚਲ int64_t k, dpct::ਕਤਾਰ_ptr stream) {
    ਅਚਲ int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint8_t, 1> scale_local_acc(sycl::range<1>(K_SCALE_ਮਾਪ), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                dequantize_ਖੰਡ_q5_K_reorder(vx, y + ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * QK_K, ਵਸਤੂ_ct1.get_ਸਮੂਹ(2), get_ਸੂਚਕ(scale_local_acc), ਵਸਤੂ_ct1, nb);
            });
    });
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਪੰਕਤੀ_q6_K_sycl(ਅਚਲ ਖਾਲੀ *vx, dst_t *y, ਅਚਲ int64_t k,
                                     dpct::ਕਤਾਰ_ptr stream) {
    ਅਚਲ int64_t nb = k / QK_K;
#ਜੇ QK_K == 256
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 64),
                                               sycl::range<3>(1, 1, 64)),
                             [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                                 dequantize_ਖੰਡ_q6_K(vx, y, ਵਸਤੂ_ct1);
                             });
    }
#ਨਹੀਂਤਾਂ
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                                 dequantize_ਖੰਡ_q6_K(vx, y, ਵਸਤੂ_ct1);
                             });
    }

#ਜੇ_ਅੰਤ
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਪੰਕਤੀ_q6_K_sycl_reorder(ਅਚਲ ਖਾਲੀ * vx, dst_t * y, ਅਚਲ int64_t k, dpct::ਕਤਾਰ_ptr stream) {
    ਅਚਲ int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) { dequantize_ਖੰਡ_q6_K_reorder(vx, y + ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * QK_K, ਵਸਤੂ_ct1.get_ਸਮੂਹ(2), ਵਸਤੂ_ct1, nb); });
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਪੰਕਤੀ_iq1_s_sycl(ਅਚਲ ਖਾਲੀ *vx, dst_t *y, ਅਚਲ int64_t k,
                                        dpct::ਕਤਾਰ_ptr stream) {
    ਅਚਲ int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                                 dequantize_ਖੰਡ_iq1_s(
                                     vx, y, ਵਸਤੂ_ct1, iq1s_grid_gpu
                                     );
                             });
        });
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਪੰਕਤੀ_iq1_m_sycl(ਅਚਲ ਖਾਲੀ *vx, dst_t *y, ਅਚਲ int64_t k,
                                        dpct::ਕਤਾਰ_ptr stream) {
    ਅਚਲ int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                                 dequantize_ਖੰਡ_iq1_m(
                                     vx, y, ਵਸਤੂ_ct1, iq1s_grid_gpu
                                     );
                             });
        });
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਪੰਕਤੀ_iq2_xxs_sycl(ਅਚਲ ਖਾਲੀ *vx, dst_t *y, ਅਚਲ int64_t k,
                                        dpct::ਕਤਾਰ_ptr stream) {
    ਅਚਲ int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                                 dequantize_ਖੰਡ_iq2_xxs(
                                     vx, y, ਵਸਤੂ_ct1, iq2xxs_grid,
                                     ksigns_iq2xs, kmask_iq2xs);
                             });
        });
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਪੰਕਤੀ_iq2_xs_sycl(ਅਚਲ ਖਾਲੀ *vx, dst_t *y, ਅਚਲ int64_t k,
                                       dpct::ਕਤਾਰ_ptr stream) {
    ਅਚਲ int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                                 dequantize_ਖੰਡ_iq2_xs(
                                     vx, y, ਵਸਤੂ_ct1, iq2xs_grid,
                                     ksigns_iq2xs, kmask_iq2xs);
                             });
        });
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਪੰਕਤੀ_iq2_s_sycl(ਅਚਲ ਖਾਲੀ *vx, dst_t *y, ਅਚਲ int64_t k,
                                      dpct::ਕਤਾਰ_ptr stream) {
    ਅਚਲ int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                                 dequantize_ਖੰਡ_iq2_s(vx, y, ਵਸਤੂ_ct1);
                             });
        });
    }
}


ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਪੰਕਤੀ_iq3_xxs_sycl(ਅਚਲ ਖਾਲੀ *vx, dst_t *y, ਅਚਲ int64_t k,
                                        dpct::ਕਤਾਰ_ptr stream) {
    ਅਚਲ int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                                 dequantize_ਖੰਡ_iq3_xxs(
                                     vx, y, ਵਸਤੂ_ct1, iq3xxs_grid,
                                     ksigns_iq2xs, kmask_iq2xs);
                             });
        });
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਪੰਕਤੀ_iq3_xxs_sycl_reorder(ਅਚਲ ਖਾਲੀ * vx, dst_t * y, ਅਚਲ int64_t k,
                                                dpct::ਕਤਾਰ_ptr stream) {
    ਅਚਲ int64_t nb = k / QK_K;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
            dequantize_ਖੰਡ_iq3_xxs_reorder(
                vx, y + ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * QK_K, ਵਸਤੂ_ct1.get_ਸਮੂਹ(2), ਵਸਤੂ_ct1, nb,
                iq3xxs_grid, ksigns_iq2xs, kmask_iq2xs);
        });
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਪੰਕਤੀ_iq3_s_sycl(ਅਚਲ ਖਾਲੀ *vx, dst_t *y, ਅਚਲ int64_t k,
                                        dpct::ਕਤਾਰ_ptr stream) {
    ਅਚਲ int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                                 dequantize_ਖੰਡ_iq3_s(
                                     vx, y, ਵਸਤੂ_ct1, kmask_iq2xs, iq3s_grid);
                             });
        });
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਪੰਕਤੀ_iq3_s_sycl_reorder(ਅਚਲ ਖਾਲੀ * vx, dst_t * y, ਅਚਲ int64_t k,
                                              dpct::ਕਤਾਰ_ptr stream) {
    ਅਚਲ int64_t nb = k / QK_K;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
            dequantize_ਖੰਡ_iq3_s_reorder(vx, y + ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * QK_K, ਵਸਤੂ_ct1.get_ਸਮੂਹ(2),
                                           ਵਸਤੂ_ct1, nb, kmask_iq2xs, iq3s_grid);
        });
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਪੰਕਤੀ_iq4_xs_sycl(ਅਚਲ ਖਾਲੀ *vx, dst_t *y, ਅਚਲ int64_t k,
                                       dpct::ਕਤਾਰ_ptr stream) {
    ਅਚਲ int64_t nb = (k + QK_K - 1) / QK_K;
#ਜੇ QK_K == 64
    dequantize_ਪੰਕਤੀ_iq4_nl_sycl(vx, y, k, stream);
#ਨਹੀਂਤਾਂ
      {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                  cgh.parallel_for(
                      sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                            sycl::range<3>(1, 1, 32),
                                        sycl::range<3>(1, 1, 32)),
                      [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                            dequantize_ਖੰਡ_iq4_xs(vx, y, ਵਸਤੂ_ct1);
                      });
            });
      }
#ਜੇ_ਅੰਤ
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਪੰਕਤੀ_iq4_nl_sycl(ਅਚਲ ਖਾਲੀ *vx, dst_t *y, ਅਚਲ int64_t k,
                                       dpct::ਕਤਾਰ_ptr stream) {
    ਅਚਲ int64_t nb = (k + QK_K - 1) / QK_K;
      {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                  cgh.parallel_for(
                      sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                            sycl::range<3>(1, 1, 32),
                                        sycl::range<3>(1, 1, 32)),
                      [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                            dequantize_ਖੰਡ_iq4_nl(vx, y, ਵਸਤੂ_ct1);
                      });
            });
      }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਪੰਕਤੀ_mxfp4_sycl(ਅਚਲ ਖਾਲੀ * vx, dst_t * y, ਅਚਲ int64_t k, dpct::ਕਤਾਰ_ptr stream) {
    ਅਚਲ ਪੂਰਨ nb = (k + QK_K - 1) / QK_K;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
            dequantize_ਖੰਡ_mxfp4(vx, y, ਵਸਤੂ_ct1);
        });
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਪੰਕਤੀ_mxfp4_sycl_reorder(ਅਚਲ ਖਾਲੀ * vx, dst_t * y, ਅਚਲ int64_t k, dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(k % QK_MXFP4 == 0);
    ਅਚਲ ਪੂਰਨ n_warp = (k / QK_MXFP4 + WARP_ਮਾਪ - 1) / WARP_ਮਾਪ;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, n_warp) * sycl::range<3>(1, 1, WARP_ਮਾਪ),
                          sycl::range<3>(1, 1, WARP_ਮਾਪ)),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
            dequantize_ਖੰਡ_mxfp4_reorder(vx, y, k, ਵਸਤੂ_ct1);
        });
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਪੰਕਤੀ_nvfp4_sycl(ਅਚਲ ਖਾਲੀ * vx, dst_t * y, ਅਚਲ int64_t k, dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(k % QK_NVFP4 == 0);
    ਅਚਲ ਪੂਰਨ nb = k / QK_NVFP4;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_ਵਸਤੂ<3> /*item_ct1*/) {
            dequantize_ਖੰਡ_nvfp4(vx, y, k);
        });
}


ਸਾਂਚਾ <ਪੂਰਨ qk, ਪੂਰਨ qr, dequantize_kernel_t dequantize_kernel, ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਖੰਡ_nc(ਅਚਲ ਖਾਲੀ * __restrict__ vx, dst_t * __restrict__ y,
        ਅਚਲ int64_t ne00, ਅਚਲ int64_t ne01, ਅਚਲ int64_t ne02,
        ਅਚਲ int64_t s01, ਅਚਲ int64_t s02, ਅਚਲ int64_t s03) {
    ਸਵੈਚਾਲਿਤ          ਵਸਤੂ_ct1 = sycl::ext::oneapi::this_work_ਵਸਤੂ::get_nd_ਵਸਤੂ<3>();
    ਅਚਲ int64_t i00 = 2 * (int64_t(ਵਸਤੂ_ct1.get_local_range(2)) * ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) + ਵਸਤੂ_ct1.get_local_id(2));

    ਜੇ (i00 >= ne00) {
        ਵਾਪਸ;
    }

    ਅਚਲ int64_t i01 = ਵਸਤੂ_ct1.get_ਸਮੂਹ(1);
    ਅਚਲ int64_t i02 = ਵਸਤੂ_ct1.get_ਸਮੂਹ(0) % ne02;
    ਅਚਲ int64_t i03 = ਵਸਤੂ_ct1.get_ਸਮੂਹ(0) / ne02;

    ਅਚਲ int64_t ibx0 = i03*s03 + i02*s02 + i01*s01;

    ਅਚਲ int64_t ib = ibx0 + i00/qk; // block index
    ਅਚਲ int64_t iqs = (i00%qk)/qr; // quant index
    ਅਚਲ int64_t iybs = i00 - i00%qk; // y block start index
    ਅਚਲ int64_t y_ਵਿੱਥ = qr == 1 ? 1 : qk/2;

    // dequantize
    #ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_F16
        sycl::half2 v;
    #ਨਹੀਂਤਾਂ
        sycl::float2 v;
    #ਜੇ_ਅੰਤ

    dequantize_kernel(vx, ib, iqs, v);

    ਅਚਲ int64_t iy0 = ((i03*ne02 + i02)*ne01 + i01)*ne00 + iybs + iqs;
    y[iy0 + 0]        = ggml_sycl_cast<dst_t>(v.x());
    y[iy0 + y_ਵਿੱਥ] = ggml_sycl_cast<dst_t>(v.y());
}


ਸਾਂਚਾ <ਪੂਰਨ qk, ਪੂਰਨ qr, dequantize_kernel_t dequantize_kernel, ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਖੰਡ_nc_sycl(ਅਚਲ ਖਾਲੀ *    vx,
                                  dst_t *         y,
                                  ਅਚਲ int64_t   ne00,
                                  ਅਚਲ int64_t   ne01,
                                  ਅਚਲ int64_t   ne02,
                                  ਅਚਲ int64_t   ne03,
                                  ਅਚਲ int64_t   s01,
                                  ਅਚਲ int64_t   s02,
                                  ਅਚਲ int64_t   s03,
                                  dpct::ਕਤਾਰ_ptr stream) {
    ਅਚਲ dpct::dim3 ਅੰਕ_ਖੰਡਾਂ((ne00 + 2 * SYCL_DEQUANTIZE_ਖੰਡ_ਮਾਪ - 1) / (2 * SYCL_DEQUANTIZE_ਖੰਡ_ਮਾਪ), ne01,
                                ne02 * ne03);
    stream->parallel_for(sycl::nd_range<3>(ਅੰਕ_ਖੰਡਾਂ * sycl::range<3>(1, 1, SYCL_DEQUANTIZE_ਖੰਡ_ਮਾਪ),
                                           sycl::range<3>(1, 1, SYCL_DEQUANTIZE_ਖੰਡ_ਮਾਪ)),
                         [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                             GGML_ਅਣਵਰਤਿਆ(ਵਸਤੂ_ct1);
                             dequantize_ਖੰਡ_nc<qk, qr, dequantize_kernel>(vx, y, ne00, ne01, ne02, s01, s02, s03);
                         });
}
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ src_t, ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ convert_unary_nc(ਅਚਲ ਖਾਲੀ * __restrict__ vx, dst_t * __restrict__ y, ਅਚਲ int64_t ne00, ਅਚਲ int64_t ne01,
                          ਅਚਲ int64_t ne02, ਅਚਲ int64_t s01, ਅਚਲ int64_t s02, ਅਚਲ int64_t s03,
                          ਅਚਲ sycl::nd_ਵਸਤੂ<3> & ਵਸਤੂ_ct1) {

    ਅਚਲ int64_t work_ਸਮੂਹ_ਮਾਪ = ਵਸਤੂ_ct1.get_local_range(2);
    ਅਚਲ int64_t global_id       = ਵਸਤੂ_ct1.get_local_id(2) + work_ਸਮੂਹ_ਮਾਪ * ਵਸਤੂ_ct1.get_ਸਮੂਹ(2);

    ਅਚਲ int64_t i01 = ਵਸਤੂ_ct1.get_ਸਮੂਹ(1);
    ਅਚਲ int64_t i02 = ਵਸਤੂ_ct1.get_ਸਮੂਹ(0) % ne02;
    ਅਚਲ int64_t i03 = ਵਸਤੂ_ct1.get_ਸਮੂਹ(0) / ne02;

    // make each work-item deal with more elements since sycl global range can not exceed max int
    ਅਚਲ src_t * x = static_cast<ਅਚਲ src_t *>(vx);
    ਅਚਲ int64_t ix = i03 * s03 + i02 * s02 + i01 * s01;
    ਅਚਲ int64_t iy = ((i03 * ne02 + i02) * ne01 + i01) * ne00;

#ਹਦਾਇਤ unroll
    ਲਈ (int64_t i00 = global_id; i00 < ne00; i00 += work_ਸਮੂਹ_ਮਾਪ * ਵਸਤੂ_ct1.get_ਸਮੂਹ_range(2)) {
        y[iy + i00] = static_cast<dst_t>(x[ix + i00]);
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ src_t, ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ convert_unary_nc_sycl(ਅਚਲ ਖਾਲੀ * __restrict__ vx, dst_t * __restrict__ y,
                                  ਅਚਲ int64_t ne00, ਅਚਲ int64_t ne01, ਅਚਲ int64_t ne02, ਅਚਲ int64_t ne03,
                                  ਅਚਲ int64_t s01, ਅਚਲ int64_t s02, ਅਚਲ int64_t s03, dpct::ਕਤਾਰ_ptr queue) {
    dpct::has_capability_or_fail(queue->get_device(), { sycl::aspect::fp16 });

    sycl::range<3> global_ਮਾਪ(ne02 * ne03, ne01, ceil_div(ne00, SYCL_DEQUANTIZE_ਖੰਡ_ਮਾਪ));

    // decrease global range when it exceeds the max int
    // TODO: Downsample logic is separated from the kernel, a rewrite is desirable
    int64_t        downsized_workgroup = downsample_sycl_global_range(global_ਮਾਪ[0], SYCL_DEQUANTIZE_ਖੰਡ_ਮਾਪ);
    sycl::range<3> workgroup_ਮਾਪ(1, 1, downsized_workgroup);

    queue->parallel_for(sycl::nd_range<3>(global_ਮਾਪ * workgroup_ਮਾਪ, workgroup_ਮਾਪ), [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
        convert_unary_nc<src_t>(vx, y, ne00, ne01, ne02, s01, s02, s03, ਵਸਤੂ_ct1);
    });
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ src_t, ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ convert_unary_sycl(ਅਚਲ ਖਾਲੀ * vx, dst_t * y, ਅਚਲ int64_t k, dpct::ਕਤਾਰ_ptr queue) {
    convert_unary_nc_sycl<src_t>(vx, y, k, 1, 1, 1, k, k, k, queue);
}


ਸਥਿਰ ਤਾਰਕਿਕ is_reordered_src0(ਅਚਲ ggml_tensor * dst) {
    ਵਾਪਸ dst->src[0]->extra &&
           ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder;
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਮੰਜ਼ਿਲ_t_sycl_t<dst_t> ggml_get_dequant_sycl(ggml_type type, ggml_tensor * dst) {
    ਚੋਣ (type) {
        ਹਾਲਤ GGML_TYPE_Q1_0:
            ਵਾਪਸ dequantize_ਖੰਡ_sycl<QK1_0, QR1_0, dequantize_q1_0, dst_t>;
        ਹਾਲਤ GGML_TYPE_Q2_0:
            ਵਾਪਸ dequantize_ਖੰਡ_sycl<QK2_0, QR2_0, dequantize_q2_0, dst_t>;
        ਹਾਲਤ GGML_TYPE_Q4_0:
            ਜੇ (is_reordered_src0(dst)) {
                ਵਾਪਸ dequantize_ਪੰਕਤੀ_q4_0_sycl_reorder;
            } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<dst_t, ਦਸ਼ਮਲਵ>) {
                ਵਾਪਸ dequantize_ਪੰਕਤੀ_q4_0_sycl;
            } ਨਹੀਂਤਾਂ {
                ਵਾਪਸ dequantize_ਖੰਡ_sycl<QK4_0, QR4_0, dequantize_q4_0, dst_t>;
            }
        ਹਾਲਤ GGML_TYPE_Q4_1:
            ਜੇ constexpr (std::is_same_v<dst_t, ਦਸ਼ਮਲਵ>) {
                ਵਾਪਸ dequantize_ਪੰਕਤੀ_q4_1_sycl;
            } ਨਹੀਂਤਾਂ {
                ਵਾਪਸ dequantize_ਖੰਡ_sycl<QK4_1, QR4_1, dequantize_q4_1, dst_t>;
            }
        ਹਾਲਤ GGML_TYPE_Q5_0:
            ਵਾਪਸ dequantize_ਖੰਡ_sycl<QK5_0, QR5_0, dequantize_q5_0, dst_t>;
        ਹਾਲਤ GGML_TYPE_Q5_1:
            ਵਾਪਸ dequantize_ਖੰਡ_sycl<QK5_1, QR5_1, dequantize_q5_1, dst_t>;
        ਹਾਲਤ GGML_TYPE_Q8_0:
            ਜੇ (is_reordered_src0(dst)) {
                ਵਾਪਸ dequantize_ਪੰਕਤੀ_q8_0_sycl_reorder;
            } ਨਹੀਂਤਾਂ {
                ਵਾਪਸ dequantize_ਖੰਡ_sycl<QK8_0, QR8_0, dequantize_q8_0, dst_t>;
            }
        ਹਾਲਤ GGML_TYPE_Q2_K:
            ਜੇ (is_reordered_src0(dst)) {
                ਵਾਪਸ dequantize_ਪੰਕਤੀ_q2_K_sycl_reorder;
            } ਨਹੀਂਤਾਂ {
                ਵਾਪਸ dequantize_ਪੰਕਤੀ_q2_K_sycl;
            }
        ਹਾਲਤ GGML_TYPE_Q3_K:
            ਜੇ (is_reordered_src0(dst)) {
                ਵਾਪਸ dequantize_ਪੰਕਤੀ_q3_K_sycl_reorder;
            } ਨਹੀਂਤਾਂ {
                ਵਾਪਸ dequantize_ਪੰਕਤੀ_q3_K_sycl;
            }
        ਹਾਲਤ GGML_TYPE_Q4_K:
            ਜੇ (is_reordered_src0(dst)) {
                ਵਾਪਸ dequantize_ਪੰਕਤੀ_q4_K_sycl_reorder;
            } ਨਹੀਂਤਾਂ {
                ਵਾਪਸ dequantize_ਪੰਕਤੀ_q4_K_sycl;
            }
        ਹਾਲਤ GGML_TYPE_Q5_K:
            ਜੇ (is_reordered_src0(dst)) {
                ਵਾਪਸ dequantize_ਪੰਕਤੀ_q5_K_sycl_reorder;
            } ਨਹੀਂਤਾਂ {
                ਵਾਪਸ dequantize_ਪੰਕਤੀ_q5_K_sycl;
            }
        ਹਾਲਤ GGML_TYPE_Q6_K:
            ਜੇ (is_reordered_src0(dst)) {
                ਵਾਪਸ dequantize_ਪੰਕਤੀ_q6_K_sycl_reorder;
            } ਨਹੀਂਤਾਂ {
                ਵਾਪਸ dequantize_ਪੰਕਤੀ_q6_K_sycl;
            }
        ਹਾਲਤ GGML_TYPE_IQ1_S:
            ਵਾਪਸ dequantize_ਪੰਕਤੀ_iq1_s_sycl;
        ਹਾਲਤ GGML_TYPE_IQ1_M:
            ਵਾਪਸ dequantize_ਪੰਕਤੀ_iq1_m_sycl;
        ਹਾਲਤ GGML_TYPE_IQ2_XXS:
            ਵਾਪਸ dequantize_ਪੰਕਤੀ_iq2_xxs_sycl;
        ਹਾਲਤ GGML_TYPE_IQ2_XS:
            ਵਾਪਸ dequantize_ਪੰਕਤੀ_iq2_xs_sycl;
        ਹਾਲਤ GGML_TYPE_IQ2_S:
            ਵਾਪਸ dequantize_ਪੰਕਤੀ_iq2_s_sycl;
        ਹਾਲਤ GGML_TYPE_IQ3_XXS:
            ਜੇ (is_reordered_src0(dst)) {
                ਵਾਪਸ dequantize_ਪੰਕਤੀ_iq3_xxs_sycl_reorder;
            } ਨਹੀਂਤਾਂ {
                ਵਾਪਸ dequantize_ਪੰਕਤੀ_iq3_xxs_sycl;
            }
        ਹਾਲਤ GGML_TYPE_IQ3_S:
            ਜੇ (is_reordered_src0(dst)) {
                ਵਾਪਸ dequantize_ਪੰਕਤੀ_iq3_s_sycl_reorder;
            } ਨਹੀਂਤਾਂ {
                ਵਾਪਸ dequantize_ਪੰਕਤੀ_iq3_s_sycl;
            }
        ਹਾਲਤ GGML_TYPE_IQ4_XS:
            ਵਾਪਸ dequantize_ਪੰਕਤੀ_iq4_xs_sycl;
        ਹਾਲਤ GGML_TYPE_IQ4_NL:
            ਵਾਪਸ dequantize_ਪੰਕਤੀ_iq4_nl_sycl;
        ਹਾਲਤ GGML_TYPE_MXFP4:
            ਜੇ (is_reordered_src0(dst)) {
                ਵਾਪਸ dequantize_ਪੰਕਤੀ_mxfp4_sycl_reorder;
            }
            ਵਾਪਸ dequantize_ਪੰਕਤੀ_mxfp4_sycl;
        ਹਾਲਤ GGML_TYPE_NVFP4:
            ਵਾਪਸ dequantize_ਪੰਕਤੀ_nvfp4_sycl;
        ਹਾਲਤ GGML_TYPE_F32:
            ਜੇ constexpr (std::is_same_v<dst_t, sycl::half>) {
                ਵਾਪਸ convert_unary_sycl<ਦਸ਼ਮਲਵ>;
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F16:
            ਜੇ constexpr (std::is_same_v<dst_t, ਦਸ਼ਮਲਵ>) {
                ਵਾਪਸ convert_unary_sycl<sycl::half>;
            }
            ਰੋਕੋ;
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_HAS_BF16
        ਹਾਲਤ GGML_TYPE_BF16:
            ਵਾਪਸ convert_unary_sycl<sycl::ext::oneapi::bfloat16>;
#ਜੇ_ਅੰਤ
        ਮੂਲ:
            ਰੋਕੋ;
    }
    GGML_ABORT("fatal error: unsupport data type=%s\n", ggml_type_name(type));
}

ਮੰਜ਼ਿਲ_fp16_sycl_t ggml_get_ਮੰਜ਼ਿਲ_fp16_sycl(ggml_type type, ggml_tensor * dst) {
    ਵਾਪਸ ggml_get_dequant_sycl<sycl::half>(type, dst);
}

ਮੰਜ਼ਿਲ_fp32_sycl_t ggml_get_ਮੰਜ਼ਿਲ_fp32_sycl(ggml_type type, ggml_tensor *dst) {
    ਵਾਪਸ ggml_get_dequant_sycl<ਦਸ਼ਮਲਵ>(type, dst);
}


#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_HAS_BF16
ਮੰਜ਼ਿਲ_bf16_sycl_t ggml_get_ਮੰਜ਼ਿਲ_bf16_sycl(ggml_type type, ggml_tensor * /*dst*/) {
    ਚੋਣ (type) {
        ਹਾਲਤ GGML_TYPE_F32:
            ਵਾਪਸ convert_unary_sycl<ਦਸ਼ਮਲਵ>;
        ਹਾਲਤ GGML_TYPE_F16:
            ਵਾਪਸ convert_unary_sycl<sycl::half>;
        ਹਾਲਤ GGML_TYPE_BF16:
            ਵਾਪਸ convert_unary_sycl<sycl::ext::oneapi::bfloat16>;
        ਮੂਲ:
            GGML_ABORT("fatal error: unsupport data type=%s\n", ggml_type_name(type));
    }
}
#ਜੇ_ਅੰਤ

ਮੰਜ਼ਿਲ_fp16_nc_sycl_t ggml_get_ਮੰਜ਼ਿਲ_fp16_nc_sycl(ggml_type type) {
    ਚੋਣ (type) {
        ਹਾਲਤ GGML_TYPE_F32:
            ਵਾਪਸ convert_unary_nc_sycl<ਦਸ਼ਮਲਵ>;
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_HAS_BF16
        ਹਾਲਤ GGML_TYPE_BF16:
            ਵਾਪਸ convert_unary_nc_sycl<sycl::ext::oneapi::bfloat16>;
#ਜੇ_ਅੰਤ
        ਹਾਲਤ GGML_TYPE_Q1_0:
            ਵਾਪਸ dequantize_ਖੰਡ_nc_sycl<QK1_0, QR1_0, dequantize_q1_0>;
        ਹਾਲਤ GGML_TYPE_Q4_0:
            ਵਾਪਸ dequantize_ਖੰਡ_nc_sycl<QK4_0, QR4_0, dequantize_q4_0>;
        ਹਾਲਤ GGML_TYPE_Q4_1:
            ਵਾਪਸ dequantize_ਖੰਡ_nc_sycl<QK4_1, QR4_1, dequantize_q4_1>;
        ਹਾਲਤ GGML_TYPE_Q5_0:
            ਵਾਪਸ dequantize_ਖੰਡ_nc_sycl<QK5_0, QR5_0, dequantize_q5_0>;
        ਹਾਲਤ GGML_TYPE_Q5_1:
            ਵਾਪਸ dequantize_ਖੰਡ_nc_sycl<QK5_1, QR5_1, dequantize_q5_1>;
        ਹਾਲਤ GGML_TYPE_Q8_0:
            ਵਾਪਸ dequantize_ਖੰਡ_nc_sycl<QK8_0, QR8_0, dequantize_q8_0>;
        ਮੂਲ:
            ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
}
