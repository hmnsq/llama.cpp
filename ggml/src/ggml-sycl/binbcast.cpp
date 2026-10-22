#ਅਨ੍ਤਰ੍ਭਾਵ "binbcast.hpp"

#ਅਨ੍ਤਰ੍ਭਾਵ <algorithm>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstddef>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <sycl/sycl.hpp>

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"

ਸਾਂਚਾ<ਦਸ਼ਮਲਵ (*bin_op)(ਅਚਲ ਦਸ਼ਮਲਵ, ਅਚਲ ਦਸ਼ਮਲਵ), ਕਿਸਮ_ਨਾਂ src0_t, ਕਿਸਮ_ਨਾਂ src1_t, ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ k_bin_bcast(ਅਚਲ src0_t * src0, ਅਚਲ src1_t * src1, dst_t * dst,
        ਪੂਰਨ ne0, ਪੂਰਨ ne1, ਪੂਰਨ ne2, ਪੂਰਨ ne3,
        ਪੂਰਨ ne10, ਪੂਰਨ ne11, ਪੂਰਨ ne12, ਪੂਰਨ ne13,
        /*int s0, */ ਪੂਰਨ s1,  ਪੂਰਨ s2,  ਪੂਰਨ s3,
        ਪੂਰਨ s00, ਪੂਰਨ s01, ਪੂਰਨ s02, ਪੂਰਨ s03,
        ਪੂਰਨ s10, ਪੂਰਨ s11, ਪੂਰਨ s12, ਪੂਰਨ s13,
        ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {
    ਅਚਲ ਪੂਰਨ i0s = ਵਸਤੂ_ct1.get_local_range(2) * ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) +
                    ਵਸਤੂ_ct1.get_local_id(2);
    ਅਚਲ ਪੂਰਨ i1 = (ਵਸਤੂ_ct1.get_local_range(1) * ਵਸਤੂ_ct1.get_ਸਮੂਹ(1) +
                    ਵਸਤੂ_ct1.get_local_id(1));
    ਅਚਲ ਪੂਰਨ i2 = (ਵਸਤੂ_ct1.get_local_range(0) * ਵਸਤੂ_ct1.get_ਸਮੂਹ(0) +
                    ਵਸਤੂ_ct1.get_local_id(0)) /
                   ne3;
    ਅਚਲ ਪੂਰਨ i3 = (ਵਸਤੂ_ct1.get_local_range(0) * ਵਸਤੂ_ct1.get_ਸਮੂਹ(0) +
                    ਵਸਤੂ_ct1.get_local_id(0)) %
                   ne3;

    ਜੇ (i0s >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        ਵਾਪਸ;
    }

    ਅਚਲ ਪੂਰਨ i11 = i1 % ne11;
    ਅਚਲ ਪੂਰਨ i12 = i2 % ne12;
    ਅਚਲ ਪੂਰਨ i13 = i3 % ne13;

    ਅਚਲ size_t i_src0 =  i3*s03 +  i2*s02 +  i1*s01;
    ਅਚਲ size_t i_src1 = i13*s13 + i12*s12 + i11*s11;
    ਅਚਲ size_t i_dst  =  i3*s3  +  i2*s2  +  i1*s1;

    ਅਚਲ src0_t * src0_ਪੰਕਤੀ = src0 + i_src0;
    ਅਚਲ src1_t * src1_ਪੰਕਤੀ = src1 + i_src1;
    dst_t * dst_ਪੰਕਤੀ = dst + i_dst;

    ਲਈ (ਪੂਰਨ i0 = i0s; i0 < ne0;
         i0 += ਵਸਤੂ_ct1.get_local_range(2) * ਵਸਤੂ_ct1.get_ਸਮੂਹ_range(2)) {
        ਅਚਲ ਪੂਰਨ i10 = i0 % ne10;
        dst_ਪੰਕਤੀ[i0] = (dst_t)bin_op(src0 ? (ਦਸ਼ਮਲਵ)src0_ਪੰਕਤੀ[i0*s00] : 0.0f, (ਦਸ਼ਮਲਵ)src1_ਪੰਕਤੀ[i10*s10]);
    }
}

ਸਾਂਚਾ<ਦਸ਼ਮਲਵ (*bin_op)(ਅਚਲ ਦਸ਼ਮਲਵ, ਅਚਲ ਦਸ਼ਮਲਵ), ਕਿਸਮ_ਨਾਂ src0_t, ਕਿਸਮ_ਨਾਂ src1_t, ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ k_bin_bcast_unravel(ਅਚਲ src0_t * src0, ਅਚਲ src1_t * src1, dst_t * dst,
        ਪੂਰਨ ne0, ਪੂਰਨ ne1, ਪੂਰਨ ne2, ਪੂਰਨ ne3,
        ਪੂਰਨ ne10, ਪੂਰਨ ne11, ਪੂਰਨ ne12, ਪੂਰਨ ne13,
        /*int s0, */ ਪੂਰਨ s1,  ਪੂਰਨ s2,  ਪੂਰਨ s3,
        ਪੂਰਨ s00, ਪੂਰਨ s01, ਪੂਰਨ s02, ਪੂਰਨ s03,
        ਪੂਰਨ s10, ਪੂਰਨ s11, ਪੂਰਨ s12, ਪੂਰਨ s13,
        ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {

    ਅਚਲ ਪੂਰਨ i = ਵਸਤੂ_ct1.get_local_range(2) * ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) +
                  ਵਸਤੂ_ct1.get_local_id(2);

    ਅਚਲ ਪੂਰਨ i3 = i/(ne2*ne1*ne0);
    ਅਚਲ ਪੂਰਨ i2 = (i/(ne1*ne0)) % ne2;
    ਅਚਲ ਪੂਰਨ i1 = (i/ne0) % ne1;
    ਅਚਲ ਪੂਰਨ i0 = i % ne0;

    ਜੇ (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        ਵਾਪਸ;
    }

    ਅਚਲ ਪੂਰਨ i11 = i1 % ne11;
    ਅਚਲ ਪੂਰਨ i12 = i2 % ne12;
    ਅਚਲ ਪੂਰਨ i13 = i3 % ne13;

    ਅਚਲ size_t i_src0 =  i3*s03 +  i2*s02 +  i1*s01;
    ਅਚਲ size_t i_src1 = i13*s13 + i12*s12 + i11*s11;
    ਅਚਲ size_t i_dst  =  i3*s3  +  i2*s2  +  i1*s1;

    ਅਚਲ src0_t * src0_ਪੰਕਤੀ = src0 + i_src0;
    ਅਚਲ src1_t * src1_ਪੰਕਤੀ = src1 + i_src1;
    dst_t * dst_ਪੰਕਤੀ = dst + i_dst;

    ਅਚਲ ਪੂਰਨ i10 = i0 % ne10;
    dst_ਪੰਕਤੀ[i0] = (dst_t)bin_op(src0 ? (ਦਸ਼ਮਲਵ)src0_ਪੰਕਤੀ[i0*s00] : 0.0f, (ਦਸ਼ਮਲਵ)src1_ਪੰਕਤੀ[i10*s10]);
}


ਸਾਂਚਾ<ਦਸ਼ਮਲਵ (*bin_op)(ਅਚਲ ਦਸ਼ਮਲਵ, ਅਚਲ ਦਸ਼ਮਲਵ)>
ਬਣਤਰ bin_bcast_sycl {
    ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ src0_t, ਕਿਸਮ_ਨਾਂ src1_t, ਕਿਸਮ_ਨਾਂ dst_t>
    ਖਾਲੀ ਸੰਚਾਲਕ()(ਅਚਲ src0_t * src0_dd, ਅਚਲ src1_t * src1_dd, dst_t * dst_dd, ਅਚਲ int64_t ne00,
                    ਅਚਲ int64_t ne01, ਅਚਲ int64_t ne02, ਅਚਲ int64_t ne03, ਅਚਲ int64_t ne10, ਅਚਲ int64_t ne11,
                    ਅਚਲ int64_t ne12, ਅਚਲ int64_t ne13, ਅਚਲ int64_t ne0, ਅਚਲ int64_t ne1, ਅਚਲ int64_t ne2,
                    ਅਚਲ int64_t ne3, ਅਚਲ size_t nb00, ਅਚਲ size_t nb01, ਅਚਲ size_t nb02, ਅਚਲ size_t nb03,
                    ਅਚਲ size_t nb10, ਅਚਲ size_t nb11, ਅਚਲ size_t nb12, ਅਚਲ size_t nb13, ਅਚਲ size_t nb0,
                    ਅਚਲ size_t nb1, ਅਚਲ size_t nb2, ਅਚਲ size_t nb3, ਅਚਲ ਤਾਰਕਿਕ src0_is_contiguous,
                    ਅਚਲ ਤਾਰਕਿਕ src1_is_contiguous, ਅਚਲ ਤਾਰਕਿਕ src0_is_permuted, ਅਚਲ ਤਾਰਕਿਕ src1_is_permuted,
                    ਕਤਾਰ_ptr stream) {
        ਪੂਰਨ nr0 = ne10 / ne0;
        ਪੂਰਨ nr1 = ne11/ne1;
        ਪੂਰਨ nr2 = ne12/ne2;
        ਪੂਰਨ nr3 = ne13/ne3;

        ਪੂਰਨ nr[4] = { nr0, nr1, nr2, nr3 };

        // collapse dimensions until first broadcast dimension
        int64_t cne[] = {ne0, ne1, ne2, ne3};
        int64_t cne0[] = {ne00, ne01, ne02, ne03};
        int64_t cne1[] = {ne10, ne11, ne12, ne13};
        size_t cnb[] = {nb0, nb1, nb2, nb3};
        size_t cnb0[] = {nb00, nb01, nb02, nb03};
        size_t cnb1[] = {nb10, nb11, nb12, nb13};
        ਸਵੈਚਾਲਿਤ collapse = [](int64_t cne[]) {
            cne[0] *= cne[1];
            cne[1] = cne[2];
            cne[2] = cne[3];
            cne[3] = 1;
        };

        ਸਵੈਚਾਲਿਤ collapse_nb = [](size_t cnb[], int64_t cne[]) {
            cnb[1] *= cne[1];
            cnb[2] *= cne[2];
            cnb[3] *= cne[3];
        };

        ਜੇ (src0_is_contiguous && src1_is_contiguous && !src0_is_permuted && !src1_is_permuted) {
            ਲਈ (ਪੂਰਨ i = 0; i < 4; i++) {
                ਜੇ (nr[i] != 1) {
                    ਰੋਕੋ;
                }
                ਜੇ (i > 0) {
                    collapse_nb(cnb, cne);
                    collapse_nb(cnb0, cne0);
                    collapse_nb(cnb1, cne1);
                    collapse(cne);
                    collapse(cne0);
                    collapse(cne1);
                }
            }
        }
        {
            int64_t ne0 = cne[0];
            int64_t ne1 = cne[1];
            int64_t ne2 = cne[2];
            int64_t ne3 = cne[3];

            int64_t ne10 = cne1[0];
            int64_t ne11 = cne1[1];
            int64_t ne12 = cne1[2];
            int64_t ne13 = cne1[3];

            size_t nb0 = cnb[0];
            size_t nb1 = cnb[1];
            size_t nb2 = cnb[2];
            size_t nb3 = cnb[3];

            size_t nb00 = cnb0[0];
            size_t nb01 = cnb0[1];
            size_t nb02 = cnb0[2];
            size_t nb03 = cnb0[3];

            size_t nb10 = cnb1[0];
            size_t nb11 = cnb1[1];
            size_t nb12 = cnb1[2];
            size_t nb13 = cnb1[3];

            // size_t s0 = nb0 / sizeof(dst_t);
            size_t s1 = nb1 / ਆਕਾਰ(dst_t);
            size_t s2 = nb2 / ਆਕਾਰ(dst_t);
            size_t s3 = nb3 / ਆਕਾਰ(dst_t);

            size_t s10 = nb10 / ਆਕਾਰ(src1_t);
            size_t s11 = nb11 / ਆਕਾਰ(src1_t);
            size_t s12 = nb12 / ਆਕਾਰ(src1_t);
            size_t s13 = nb13 / ਆਕਾਰ(src1_t);

            size_t s00 = nb00 / ਆਕਾਰ(src0_t);
            size_t s01 = nb01 / ਆਕਾਰ(src0_t);
            size_t s02 = nb02 / ਆਕਾਰ(src0_t);
            size_t s03 = nb03 / ਆਕਾਰ(src0_t);

            GGML_ਅਣਵਰਤਿਆ(s00);

            GGML_ASSERT(nb0 % ਆਕਾਰ(dst_t) == 0);
            GGML_ASSERT(nb1 % ਆਕਾਰ(dst_t) == 0);
            GGML_ASSERT(nb2 % ਆਕਾਰ(dst_t) == 0);
            GGML_ASSERT(nb3 % ਆਕਾਰ(dst_t) == 0);

            GGML_ASSERT(nb00 % ਆਕਾਰ(src0_t) == 0);
            GGML_ASSERT(nb01 % ਆਕਾਰ(src0_t) == 0);
            GGML_ASSERT(nb02 % ਆਕਾਰ(src0_t) == 0);
            GGML_ASSERT(nb03 % ਆਕਾਰ(src0_t) == 0);

            GGML_ASSERT(nb10 % ਆਕਾਰ(src1_t) == 0);
            GGML_ASSERT(nb11 % ਆਕਾਰ(src1_t) == 0);
            GGML_ASSERT(nb12 % ਆਕਾਰ(src1_t) == 0);
            GGML_ASSERT(nb13 % ਆਕਾਰ(src1_t) == 0);

            ਅਚਲ ਪੂਰਨ ਖੰਡ_ਮਾਪ_3 = 128;

            int64_t hne0 = std::max(ne0/2LL, 1LL);

            sycl::range<3> ਖੰਡ_dims(1, 1, 1);
            ਖੰਡ_dims[2] = std::min<ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ>(hne0, ਖੰਡ_ਮਾਪ_3);
            ਖੰਡ_dims[1] = std::min<ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ>(
                ne1, ਖੰਡ_ਮਾਪ_3 / (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ)ਖੰਡ_dims[2]);
            ਖੰਡ_dims[0] = std::min(
                std::min<ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ>(
                    ne2 * ne3, ਖੰਡ_ਮਾਪ_3 / (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ)ਖੰਡ_dims[2] /
                                   (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ)ਖੰਡ_dims[1]),
                64U);

            sycl::range<3> ਖੰਡ_nums(
                (ne2 * ne3 + ਖੰਡ_dims[0] - 1) / ਖੰਡ_dims[0],
                (ne1 + ਖੰਡ_dims[1] - 1) / ਖੰਡ_dims[1],
                (hne0 + ਖੰਡ_dims[2] - 1) / ਖੰਡ_dims[2]);

            ਜੇ (ਖੰਡ_nums[0] > 65535) {
                // this is the maximum number of blocks in z direction, fallback to 1D grid kernel
                ਪੂਰਨ ਖੰਡ_ਅੰਕ = (ne0*ne1*ne2*ne3 + ਖੰਡ_ਮਾਪ_3 - 1) / ਖੰਡ_ਮਾਪ_3;
                {
                    dpct::has_capability_or_fail(stream->get_device(),
                                                 {sycl::aspect::fp16});

                    stream->parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, ਖੰਡ_ਅੰਕ) *
                                              sycl::range<3>(1, 1, ਖੰਡ_ਮਾਪ_3),
                                          sycl::range<3>(1, 1, ਖੰਡ_ਮਾਪ_3)),
                        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                            k_bin_bcast_unravel<bin_op>(
                                src0_dd, src1_dd, dst_dd, ne0, ne1, ne2, ne3,
                                ne10, ne11, ne12, ne13, s1, s2, s3, s00, s01, s02,
                                s03, s10, s11, s12, s13, ਵਸਤੂ_ct1);
                        });
                }
            } ਨਹੀਂਤਾਂ {
                /*
                DPCT1049:16: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                dpct::has_capability_or_fail(stream->get_device(),
                                             {sycl::aspect::fp16});

                stream->parallel_for(
                    sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                    [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                        k_bin_bcast<bin_op>(src0_dd, src1_dd, dst_dd, ne0, ne1,
                                            ne2, ne3, ne10, ne11, ne12, ne13,
                                            s1, s2, s3, s00, s01, s02, s03, s10, s11, s12, s13,
                                            ਵਸਤੂ_ct1);
                    });
            }
        }
    }
};

ਸਾਂਚਾ <ਵਰਗ op>
ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_bin_bcast(ggml_backend_sycl_context & ctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1,
                                   ggml_tensor * dst) {
    dpct::ਕਤਾਰ_ptr main_ਧਾਰਾ = ctx.stream();
    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

    ਜੇ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        op()((ਅਚਲ ਦਸ਼ਮਲਵ *) src0->ਅੰਕੜੇ, (ਅਚਲ ਦਸ਼ਮਲਵ *) src1->ਅੰਕੜੇ, (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ, ne00, ne01, ne02, ne03, ne10,
             ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2, nb3,
             ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1), main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
        op()((ਅਚਲ sycl::half *) src0->ਅੰਕੜੇ, (ਅਚਲ sycl::half *) src1->ਅੰਕੜੇ, (sycl::half *) dst->ਅੰਕੜੇ, ne00, ne01,
             ne02, ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13,
             nb0, nb1, nb2, nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1),
             main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
        op()((ਅਚਲ sycl::half *) src0->ਅੰਕੜੇ, (ਅਚਲ ਦਸ਼ਮਲਵ *) src1->ਅੰਕੜੇ, (sycl::half *) dst->ਅੰਕੜੇ, ne00, ne01, ne02,
             ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1,
             nb2, nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1),
             main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I32) {
        op()((ਅਚਲ int32_t *) src0->ਅੰਕੜੇ, (ਅਚਲ int32_t *) src1->ਅੰਕੜੇ, (int32_t *) dst->ਅੰਕੜੇ, ne00, ne01, ne02, ne03,
             ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2,
             nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1),
             main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_I16 && src1->type == GGML_TYPE_I16 && dst->type == GGML_TYPE_I16) {
        op()((ਅਚਲ int16_t *) src0->ਅੰਕੜੇ, (ਅਚਲ int16_t *) src1->ਅੰਕੜੇ, (int16_t *) dst->ਅੰਕੜੇ, ne00, ne01, ne02, ne03,
             ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2,
             nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1),
             main_ਧਾਰਾ);
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_HAS_BF16
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) {
        op()((ਅਚਲ sycl::ext::oneapi::bfloat16 *) src0->ਅੰਕੜੇ, (ਅਚਲ sycl::ext::oneapi::bfloat16 *) src1->ਅੰਕੜੇ,
             (sycl::ext::oneapi::bfloat16 *) dst->ਅੰਕੜੇ, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2,
             ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2, nb3, ggml_is_contiguous(src0),
             ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1), main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_BF16) {
        op()((ਅਚਲ sycl::ext::oneapi::bfloat16 *) src0->ਅੰਕੜੇ, (ਅਚਲ ਦਸ਼ਮਲਵ *) src1->ਅੰਕੜੇ,
             (sycl::ext::oneapi::bfloat16 *) dst->ਅੰਕੜੇ, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2,
             ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2, nb3, ggml_is_contiguous(src0),
             ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1), main_ਧਾਰਾ);
#ਜੇ_ਅੰਤ
    } ਨਹੀਂਤਾਂ {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__, ggml_type_name(dst->type),
                ggml_type_name(src0->type), ggml_type_name(src1->type));
        GGML_ABORT("fatal error");
    }
}

ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_ਜੋੜੋ(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_ਜੋੜੋ>>(ctx, dst->src[0], dst->src[1], dst);
}

ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_sub(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_sub>>(ctx, dst->src[0], dst->src[1], dst);
}

ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_mul(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_mul>>(ctx, dst->src[0], dst->src[1], dst);
}

ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_div(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_div>>(ctx, dst->src[0], dst->src[1], dst);
}

ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_repeat(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_repeat>>(ctx, dst, dst->src[0], dst);
}


ਖਾਲੀ ggml_sycl_ਜੋੜੋ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_ਜੋੜੋ(ctx, dst);
}

ਖਾਲੀ ggml_sycl_sub(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_sub(ctx, dst);
}

ਖਾਲੀ ggml_sycl_mul(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_mul(ctx, dst);
}

ਖਾਲੀ ggml_sycl_div(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_div(ctx, dst);
}

ਖਾਲੀ ggml_sycl_repeat(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_repeat(ctx, dst);
}

// fused ADD+ADD: dst = (src0 + src1) + src2. Same indexing as k_bin_bcast, so mixed
// types, broadcast, and non-contiguous layouts that add() already handles also fuse.
ਸਾਂਚਾ<ਦਸ਼ਮਲਵ (*bin_op)(ਅਚਲ ਦਸ਼ਮਲਵ, ਅਚਲ ਦਸ਼ਮਲਵ), ਕਿਸਮ_ਨਾਂ src0_t, ਕਿਸਮ_ਨਾਂ src1_t, ਕਿਸਮ_ਨਾਂ src2_t, ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ k_bin_bcast3(ਅਚਲ src0_t * src0, ਅਚਲ src1_t * src1, ਅਚਲ src2_t * src2, dst_t * dst,
        ਪੂਰਨ ne0, ਪੂਰਨ ne1, ਪੂਰਨ ne2, ਪੂਰਨ ne3,
        ਪੂਰਨ ne10, ਪੂਰਨ ne11, ਪੂਰਨ ne12, ਪੂਰਨ ne13,
        ਪੂਰਨ ne20, ਪੂਰਨ ne21, ਪੂਰਨ ne22, ਪੂਰਨ ne23,
        ਪੂਰਨ s1, ਪੂਰਨ s2, ਪੂਰਨ s3,
        ਪੂਰਨ s00, ਪੂਰਨ s01, ਪੂਰਨ s02, ਪੂਰਨ s03,
        ਪੂਰਨ s10, ਪੂਰਨ s11, ਪੂਰਨ s12, ਪੂਰਨ s13,
        ਪੂਰਨ s20, ਪੂਰਨ s21, ਪੂਰਨ s22, ਪੂਰਨ s23,
        ਅਚਲ sycl::nd_ਵਸਤੂ<3> & ਵਸਤੂ_ct1) {
    ਅਚਲ ਪੂਰਨ i0s = ਵਸਤੂ_ct1.get_local_range(2) * ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) +
                    ਵਸਤੂ_ct1.get_local_id(2);
    ਅਚਲ ਪੂਰਨ i1 = (ਵਸਤੂ_ct1.get_local_range(1) * ਵਸਤੂ_ct1.get_ਸਮੂਹ(1) +
                    ਵਸਤੂ_ct1.get_local_id(1));
    ਅਚਲ ਪੂਰਨ i2 = (ਵਸਤੂ_ct1.get_local_range(0) * ਵਸਤੂ_ct1.get_ਸਮੂਹ(0) +
                    ਵਸਤੂ_ct1.get_local_id(0)) /
                   ne3;
    ਅਚਲ ਪੂਰਨ i3 = (ਵਸਤੂ_ct1.get_local_range(0) * ਵਸਤੂ_ct1.get_ਸਮੂਹ(0) +
                    ਵਸਤੂ_ct1.get_local_id(0)) %
                   ne3;

    ਜੇ (i0s >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        ਵਾਪਸ;
    }

    ਅਚਲ ਪੂਰਨ i11 = i1 % ne11;
    ਅਚਲ ਪੂਰਨ i12 = i2 % ne12;
    ਅਚਲ ਪੂਰਨ i13 = i3 % ne13;
    ਅਚਲ ਪੂਰਨ i21 = i1 % ne21;
    ਅਚਲ ਪੂਰਨ i22 = i2 % ne22;
    ਅਚਲ ਪੂਰਨ i23 = i3 % ne23;

    ਅਚਲ size_t i_src0 = i3 * s03 + i2 * s02 + i1 * s01;
    ਅਚਲ size_t i_src1 = i13 * s13 + i12 * s12 + i11 * s11;
    ਅਚਲ size_t i_src2 = i23 * s23 + i22 * s22 + i21 * s21;
    ਅਚਲ size_t i_dst  = i3 * s3 + i2 * s2 + i1 * s1;

    ਅਚਲ src0_t * src0_ਪੰਕਤੀ = src0 + i_src0;
    ਅਚਲ src1_t * src1_ਪੰਕਤੀ = src1 + i_src1;
    ਅਚਲ src2_t * src2_ਪੰਕਤੀ = src2 + i_src2;
    dst_t *        dst_ਪੰਕਤੀ  = dst + i_dst;

    ਲਈ (ਪੂਰਨ i0 = i0s; i0 < ne0;
         i0 += ਵਸਤੂ_ct1.get_local_range(2) * ਵਸਤੂ_ct1.get_ਸਮੂਹ_range(2)) {
        ਅਚਲ ਪੂਰਨ   i10 = i0 % ne10;
        ਅਚਲ ਪੂਰਨ   i20 = i0 % ne20;
        ਅਚਲ ਦਸ਼ਮਲਵ acc = bin_op((ਦਸ਼ਮਲਵ) src0_ਪੰਕਤੀ[i0 * s00], (ਦਸ਼ਮਲਵ) src1_ਪੰਕਤੀ[i10 * s10]);
        dst_ਪੰਕਤੀ[i0]     = (dst_t) bin_op(acc, (ਦਸ਼ਮਲਵ) src2_ਪੰਕਤੀ[i20 * s20]);
    }
}

ਸਾਂਚਾ<ਦਸ਼ਮਲਵ (*bin_op)(ਅਚਲ ਦਸ਼ਮਲਵ, ਅਚਲ ਦਸ਼ਮਲਵ), ਕਿਸਮ_ਨਾਂ src0_t, ਕਿਸਮ_ਨਾਂ src1_t, ਕਿਸਮ_ਨਾਂ src2_t, ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ k_bin_bcast3_unravel(ਅਚਲ src0_t * src0, ਅਚਲ src1_t * src1, ਅਚਲ src2_t * src2, dst_t * dst,
        ਪੂਰਨ ne0, ਪੂਰਨ ne1, ਪੂਰਨ ne2, ਪੂਰਨ ne3,
        ਪੂਰਨ ne10, ਪੂਰਨ ne11, ਪੂਰਨ ne12, ਪੂਰਨ ne13,
        ਪੂਰਨ ne20, ਪੂਰਨ ne21, ਪੂਰਨ ne22, ਪੂਰਨ ne23,
        ਪੂਰਨ s1, ਪੂਰਨ s2, ਪੂਰਨ s3,
        ਪੂਰਨ s00, ਪੂਰਨ s01, ਪੂਰਨ s02, ਪੂਰਨ s03,
        ਪੂਰਨ s10, ਪੂਰਨ s11, ਪੂਰਨ s12, ਪੂਰਨ s13,
        ਪੂਰਨ s20, ਪੂਰਨ s21, ਪੂਰਨ s22, ਪੂਰਨ s23,
        ਅਚਲ sycl::nd_ਵਸਤੂ<3> & ਵਸਤੂ_ct1) {
    ਅਚਲ ਪੂਰਨ i = ਵਸਤੂ_ct1.get_local_range(2) * ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) +
                  ਵਸਤੂ_ct1.get_local_id(2);

    ਅਚਲ ਪੂਰਨ i3 = i / (ne2 * ne1 * ne0);
    ਅਚਲ ਪੂਰਨ i2 = (i / (ne1 * ne0)) % ne2;
    ਅਚਲ ਪੂਰਨ i1 = (i / ne0) % ne1;
    ਅਚਲ ਪੂਰਨ i0 = i % ne0;

    ਜੇ (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        ਵਾਪਸ;
    }

    ਅਚਲ ਪੂਰਨ i11 = i1 % ne11;
    ਅਚਲ ਪੂਰਨ i12 = i2 % ne12;
    ਅਚਲ ਪੂਰਨ i13 = i3 % ne13;
    ਅਚਲ ਪੂਰਨ i21 = i1 % ne21;
    ਅਚਲ ਪੂਰਨ i22 = i2 % ne22;
    ਅਚਲ ਪੂਰਨ i23 = i3 % ne23;

    ਅਚਲ size_t i_src0 = i3 * s03 + i2 * s02 + i1 * s01;
    ਅਚਲ size_t i_src1 = i13 * s13 + i12 * s12 + i11 * s11;
    ਅਚਲ size_t i_src2 = i23 * s23 + i22 * s22 + i21 * s21;
    ਅਚਲ size_t i_dst  = i3 * s3 + i2 * s2 + i1 * s1;

    ਅਚਲ ਪੂਰਨ   i10 = i0 % ne10;
    ਅਚਲ ਪੂਰਨ   i20 = i0 % ne20;
    ਅਚਲ ਦਸ਼ਮਲਵ acc = bin_op((ਦਸ਼ਮਲਵ) src0[i_src0 + i0 * s00], (ਦਸ਼ਮਲਵ) src1[i_src1 + i10 * s10]);
    dst[i_dst + i0] = (dst_t) bin_op(acc, (ਦਸ਼ਮਲਵ) src2[i_src2 + i20 * s20]);
}

ਸਾਂਚਾ<ਦਸ਼ਮਲਵ (*bin_op)(ਅਚਲ ਦਸ਼ਮਲਵ, ਅਚਲ ਦਸ਼ਮਲਵ), ਕਿਸਮ_ਨਾਂ src0_t, ਕਿਸਮ_ਨਾਂ src1_t, ਕਿਸਮ_ਨਾਂ src2_t, ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ launch_bin_bcast3(ggml_backend_sycl_context & ctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1,
                              ਅਚਲ ggml_tensor * src2, ggml_tensor * dst) {
    dpct::ਕਤਾਰ_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_ਸੈੱਟ_device(ctx.device));

    GGML_TENSOR_TERNARY_OP_LOCALS

    ਪੂਰਨ nr1[4] = { (ਪੂਰਨ) (ne10 / ne0), (ਪੂਰਨ) (ne11 / ne1), (ਪੂਰਨ) (ne12 / ne2), (ਪੂਰਨ) (ne13 / ne3) };
    ਪੂਰਨ nr2[4] = { (ਪੂਰਨ) (ne20 / ne0), (ਪੂਰਨ) (ne21 / ne1), (ਪੂਰਨ) (ne22 / ne2), (ਪੂਰਨ) (ne23 / ne3) };

    int64_t cne[]  = { ne0, ne1, ne2, ne3 };
    int64_t cne0[] = { ne00, ne01, ne02, ne03 };
    int64_t cne1[] = { ne10, ne11, ne12, ne13 };
    int64_t cne2[] = { ne20, ne21, ne22, ne23 };
    size_t  cnb[]  = { nb0, nb1, nb2, nb3 };
    size_t  cnb0[] = { nb00, nb01, nb02, nb03 };
    size_t  cnb1[] = { nb10, nb11, nb12, nb13 };
    size_t  cnb2[] = { nb20, nb21, nb22, nb23 };

    ਸਵੈਚਾਲਿਤ collapse = [](int64_t cne[]) {
        cne[0] *= cne[1];
        cne[1] = cne[2];
        cne[2] = cne[3];
        cne[3] = 1;
    };

    ਸਵੈਚਾਲਿਤ collapse_nb = [](size_t cnb[], int64_t cne[]) {
        cnb[1] *= cne[1];
        cnb[2] *= cne[2];
        cnb[3] *= cne[3];
    };

    ਅਚਲ ਤਾਰਕਿਕ can_collapse = ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(src2) &&
                              !ggml_is_permuted(src0) && !ggml_is_permuted(src1) && !ggml_is_permuted(src2);
    ਜੇ (can_collapse) {
        ਲਈ (ਪੂਰਨ i = 0; i < 4; i++) {
            ਜੇ (nr1[i] != 1 || nr2[i] != 1) {
                ਰੋਕੋ;
            }
            ਜੇ (i > 0) {
                collapse_nb(cnb, cne);
                collapse_nb(cnb0, cne0);
                collapse_nb(cnb1, cne1);
                collapse_nb(cnb2, cne2);
                collapse(cne);
                collapse(cne0);
                collapse(cne1);
                collapse(cne2);
            }
        }
    }

    {
        int64_t ne0 = cne[0];
        int64_t ne1 = cne[1];
        int64_t ne2 = cne[2];
        int64_t ne3 = cne[3];

        int64_t ne10 = cne1[0];
        int64_t ne11 = cne1[1];
        int64_t ne12 = cne1[2];
        int64_t ne13 = cne1[3];

        int64_t ne20 = cne2[0];
        int64_t ne21 = cne2[1];
        int64_t ne22 = cne2[2];
        int64_t ne23 = cne2[3];

        size_t s1 = cnb[1] / ਆਕਾਰ(dst_t);
        size_t s2 = cnb[2] / ਆਕਾਰ(dst_t);
        size_t s3 = cnb[3] / ਆਕਾਰ(dst_t);

        size_t s00 = cnb0[0] / ਆਕਾਰ(src0_t);
        size_t s01 = cnb0[1] / ਆਕਾਰ(src0_t);
        size_t s02 = cnb0[2] / ਆਕਾਰ(src0_t);
        size_t s03 = cnb0[3] / ਆਕਾਰ(src0_t);

        size_t s10 = cnb1[0] / ਆਕਾਰ(src1_t);
        size_t s11 = cnb1[1] / ਆਕਾਰ(src1_t);
        size_t s12 = cnb1[2] / ਆਕਾਰ(src1_t);
        size_t s13 = cnb1[3] / ਆਕਾਰ(src1_t);

        size_t s20 = cnb2[0] / ਆਕਾਰ(src2_t);
        size_t s21 = cnb2[1] / ਆਕਾਰ(src2_t);
        size_t s22 = cnb2[2] / ਆਕਾਰ(src2_t);
        size_t s23 = cnb2[3] / ਆਕਾਰ(src2_t);

        GGML_ASSERT(cnb[0] % ਆਕਾਰ(dst_t) == 0 && cnb[1] % ਆਕਾਰ(dst_t) == 0 && cnb[2] % ਆਕਾਰ(dst_t) == 0 &&
                    cnb[3] % ਆਕਾਰ(dst_t) == 0);
        GGML_ASSERT(cnb0[0] % ਆਕਾਰ(src0_t) == 0 && cnb0[1] % ਆਕਾਰ(src0_t) == 0 && cnb0[2] % ਆਕਾਰ(src0_t) == 0 &&
                    cnb0[3] % ਆਕਾਰ(src0_t) == 0);
        GGML_ASSERT(cnb1[0] % ਆਕਾਰ(src1_t) == 0 && cnb1[1] % ਆਕਾਰ(src1_t) == 0 && cnb1[2] % ਆਕਾਰ(src1_t) == 0 &&
                    cnb1[3] % ਆਕਾਰ(src1_t) == 0);
        GGML_ASSERT(cnb2[0] % ਆਕਾਰ(src2_t) == 0 && cnb2[1] % ਆਕਾਰ(src2_t) == 0 && cnb2[2] % ਆਕਾਰ(src2_t) == 0 &&
                    cnb2[3] % ਆਕਾਰ(src2_t) == 0);

        ਅਚਲ src0_t * src0_dd = (ਅਚਲ src0_t *) src0->ਅੰਕੜੇ;
        ਅਚਲ src1_t * src1_dd = (ਅਚਲ src1_t *) src1->ਅੰਕੜੇ;
        ਅਚਲ src2_t * src2_dd = (ਅਚਲ src2_t *) src2->ਅੰਕੜੇ;
        dst_t *        dst_dd  = (dst_t *) dst->ਅੰਕੜੇ;

        ਅਚਲ ਪੂਰਨ ਖੰਡ_ਮਾਪ_3 = 128;
        int64_t   hne0       = std::max(ne0 / 2LL, 1LL);

        sycl::range<3> ਖੰਡ_dims(1, 1, 1);
        ਖੰਡ_dims[2] = std::min<ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ>(hne0, ਖੰਡ_ਮਾਪ_3);
        ਖੰਡ_dims[1] = std::min<ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ>(ne1, ਖੰਡ_ਮਾਪ_3 / (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ) ਖੰਡ_dims[2]);
        ਖੰਡ_dims[0] = std::min(std::min<ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ>(ne2 * ne3,
                                                        ਖੰਡ_ਮਾਪ_3 / (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ) ਖੰਡ_dims[2] /
                                                            (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ) ਖੰਡ_dims[1]),
                                 64U);

        sycl::range<3> ਖੰਡ_nums((ne2 * ne3 + ਖੰਡ_dims[0] - 1) / ਖੰਡ_dims[0],
                                  (ne1 + ਖੰਡ_dims[1] - 1) / ਖੰਡ_dims[1],
                                  (hne0 + ਖੰਡ_dims[2] - 1) / ਖੰਡ_dims[2]);

        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        ਜੇ (ਖੰਡ_nums[0] > 65535) {
            ਪੂਰਨ ਖੰਡ_ਅੰਕ = (ne0 * ne1 * ne2 * ne3 + ਖੰਡ_ਮਾਪ_3 - 1) / ਖੰਡ_ਮਾਪ_3;
            stream->parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, ਖੰਡ_ਅੰਕ) * sycl::range<3>(1, 1, ਖੰਡ_ਮਾਪ_3),
                                  sycl::range<3>(1, 1, ਖੰਡ_ਮਾਪ_3)),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                    k_bin_bcast3_unravel<bin_op>(src0_dd, src1_dd, src2_dd, dst_dd, ne0, ne1, ne2, ne3, ne10, ne11,
                                                 ne12, ne13, ne20, ne21, ne22, ne23, s1, s2, s3, s00, s01, s02, s03,
                                                 s10, s11, s12, s13, s20, s21, s22, s23, ਵਸਤੂ_ct1);
                });
        } ਨਹੀਂਤਾਂ {
            stream->parallel_for(sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                                 [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                                     k_bin_bcast3<bin_op>(src0_dd, src1_dd, src2_dd, dst_dd, ne0, ne1, ne2, ne3, ne10,
                                                          ne11, ne12, ne13, ne20, ne21, ne22, ne23, s1, s2, s3, s00,
                                                          s01, s02, s03, s10, s11, s12, s13, s20, s21, s22, s23,
                                                          ਵਸਤੂ_ct1);
                                 });
        }
    }
}

ਖਾਲੀ ggml_sycl_op_ਜੋੜੋ_ਜੋੜੋ_fused(ggml_backend_sycl_context & ctx, ggml_tensor * add0, ggml_tensor * add1) {
    ਅਚਲ ggml_tensor * src0 = add0->src[0];
    ਅਚਲ ggml_tensor * src1 = add0->src[1];
    ਅਚਲ ggml_tensor * src2 = add1->src[1];
    ggml_tensor *       dst  = add1;

    GGML_ASSERT(add1->src[0] == add0);
    GGML_ASSERT(ggml_sycl_ਜੋੜੋ_kernel_supports(src0->type, src1->type, add0->type));
    GGML_ASSERT(ggml_sycl_ਜੋੜੋ_kernel_supports(add0->type, src2->type, dst->type));

    ਜੇ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32) {
        launch_bin_bcast3<op_ਜੋੜੋ, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ>(ctx, src0, src1, src2, dst);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && src2->type == GGML_TYPE_F16 &&
               dst->type == GGML_TYPE_F16) {
        launch_bin_bcast3<op_ਜੋੜੋ, sycl::half, sycl::half, sycl::half, sycl::half>(ctx, src0, src1, src2, dst);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_F32 &&
               dst->type == GGML_TYPE_F16) {
        launch_bin_bcast3<op_ਜੋੜੋ, sycl::half, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ, sycl::half>(ctx, src0, src1, src2, dst);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && src2->type == GGML_TYPE_F32 &&
               dst->type == GGML_TYPE_F16) {
        launch_bin_bcast3<op_ਜੋੜੋ, sycl::half, sycl::half, ਦਸ਼ਮਲਵ, sycl::half>(ctx, src0, src1, src2, dst);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_F16 &&
               dst->type == GGML_TYPE_F16) {
        launch_bin_bcast3<op_ਜੋੜੋ, sycl::half, ਦਸ਼ਮਲਵ, sycl::half, sycl::half>(ctx, src0, src1, src2, dst);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32 && src2->type == GGML_TYPE_I32 &&
               dst->type == GGML_TYPE_I32) {
        launch_bin_bcast3<op_ਜੋੜੋ, int32_t, int32_t, int32_t, int32_t>(ctx, src0, src1, src2, dst);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_I16 && src1->type == GGML_TYPE_I16 && src2->type == GGML_TYPE_I16 &&
               dst->type == GGML_TYPE_I16) {
        launch_bin_bcast3<op_ਜੋੜੋ, int16_t, int16_t, int16_t, int16_t>(ctx, src0, src1, src2, dst);
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_HAS_BF16
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && src2->type == GGML_TYPE_BF16 &&
               dst->type == GGML_TYPE_BF16) {
        launch_bin_bcast3<op_ਜੋੜੋ, sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16,
                          sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16>(ctx, src0, src1, src2, dst);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_F32 &&
               dst->type == GGML_TYPE_BF16) {
        launch_bin_bcast3<op_ਜੋੜੋ, sycl::ext::oneapi::bfloat16, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ, sycl::ext::oneapi::bfloat16>(
            ctx, src0, src1, src2, dst);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && src2->type == GGML_TYPE_F32 &&
               dst->type == GGML_TYPE_BF16) {
        launch_bin_bcast3<op_ਜੋੜੋ, sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16, ਦਸ਼ਮਲਵ,
                          sycl::ext::oneapi::bfloat16>(ctx, src0, src1, src2, dst);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_BF16 &&
               dst->type == GGML_TYPE_BF16) {
        launch_bin_bcast3<op_ਜੋੜੋ, sycl::ext::oneapi::bfloat16, ਦਸ਼ਮਲਵ, sycl::ext::oneapi::bfloat16,
                          sycl::ext::oneapi::bfloat16>(ctx, src0, src1, src2, dst);
#ਜੇ_ਅੰਤ
    } ਨਹੀਂਤਾਂ {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s, src1: %s, src2: %s\n", __func__,
                ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type),
                ggml_type_name(src2->type));
        GGML_ABORT("fatal error");
    }
}

