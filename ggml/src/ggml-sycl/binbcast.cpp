#ಅನ್ತರ್ಭಾವ "binbcast.hpp"

#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <cstddef>
#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <sycl/sycl.hpp>

#ಅನ್ತರ್ಭಾವ "ggml.h"

ಮಾದರಿ<ದಶಮಾಂಶ (*bin_op)(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ), ಬಗೆ_ಹೆಸರು src0_t, ಬಗೆ_ಹೆಸರು src1_t, ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ k_bin_bcast(ಸ್ಥಿರಾಂಕ src0_t * src0, ಸ್ಥಿರಾಂಕ src1_t * src1, dst_t * dst,
        ಪೂರ್ಣಾಂಕ ne0, ಪೂರ್ಣಾಂಕ ne1, ಪೂರ್ಣಾಂಕ ne2, ಪೂರ್ಣಾಂಕ ne3,
        ಪೂರ್ಣಾಂಕ ne10, ಪೂರ್ಣಾಂಕ ne11, ಪೂರ್ಣಾಂಕ ne12, ಪೂರ್ಣಾಂಕ ne13,
        /*int s0, */ ಪೂರ್ಣಾಂಕ s1,  ಪೂರ್ಣಾಂಕ s2,  ಪೂರ್ಣಾಂಕ s3,
        ಪೂರ್ಣಾಂಕ s00, ಪೂರ್ಣಾಂಕ s01, ಪೂರ್ಣಾಂಕ s02, ಪೂರ್ಣಾಂಕ s03,
        ಪೂರ್ಣಾಂಕ s10, ಪೂರ್ಣಾಂಕ s11, ಪೂರ್ಣಾಂಕ s12, ಪೂರ್ಣಾಂಕ s13,
        ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i0s = ಅಂಶ_ct1.get_local_range(2) * ಅಂಶ_ct1.get_ಗುಂಪು(2) +
                    ಅಂಶ_ct1.get_local_id(2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i1 = (ಅಂಶ_ct1.get_local_range(1) * ಅಂಶ_ct1.get_ಗುಂಪು(1) +
                    ಅಂಶ_ct1.get_local_id(1));
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i2 = (ಅಂಶ_ct1.get_local_range(0) * ಅಂಶ_ct1.get_ಗುಂಪು(0) +
                    ಅಂಶ_ct1.get_local_id(0)) /
                   ne3;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i3 = (ಅಂಶ_ct1.get_local_range(0) * ಅಂಶ_ct1.get_ಗುಂಪು(0) +
                    ಅಂಶ_ct1.get_local_id(0)) %
                   ne3;

    ಆದರೆ (i0s >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i11 = i1 % ne11;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i12 = i2 % ne12;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i13 = i3 % ne13;

    ಸ್ಥಿರಾಂಕ size_t i_src0 =  i3*s03 +  i2*s02 +  i1*s01;
    ಸ್ಥಿರಾಂಕ size_t i_src1 = i13*s13 + i12*s12 + i11*s11;
    ಸ್ಥಿರಾಂಕ size_t i_dst  =  i3*s3  +  i2*s2  +  i1*s1;

    ಸ್ಥಿರಾಂಕ src0_t * src0_ಅಡ್ಡಸಾಲು = src0 + i_src0;
    ಸ್ಥಿರಾಂಕ src1_t * src1_ಅಡ್ಡಸಾಲು = src1 + i_src1;
    dst_t * dst_ಅಡ್ಡಸಾಲು = dst + i_dst;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = i0s; i0 < ne0;
         i0 += ಅಂಶ_ct1.get_local_range(2) * ಅಂಶ_ct1.get_ಗುಂಪು_range(2)) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i10 = i0 % ne10;
        dst_ಅಡ್ಡಸಾಲು[i0] = (dst_t)bin_op(src0 ? (ದಶಮಾಂಶ)src0_ಅಡ್ಡಸಾಲು[i0*s00] : 0.0f, (ದಶಮಾಂಶ)src1_ಅಡ್ಡಸಾಲು[i10*s10]);
    }
}

ಮಾದರಿ<ದಶಮಾಂಶ (*bin_op)(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ), ಬಗೆ_ಹೆಸರು src0_t, ಬಗೆ_ಹೆಸರು src1_t, ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ k_bin_bcast_unravel(ಸ್ಥಿರಾಂಕ src0_t * src0, ಸ್ಥಿರಾಂಕ src1_t * src1, dst_t * dst,
        ಪೂರ್ಣಾಂಕ ne0, ಪೂರ್ಣಾಂಕ ne1, ಪೂರ್ಣಾಂಕ ne2, ಪೂರ್ಣಾಂಕ ne3,
        ಪೂರ್ಣಾಂಕ ne10, ಪೂರ್ಣಾಂಕ ne11, ಪೂರ್ಣಾಂಕ ne12, ಪೂರ್ಣಾಂಕ ne13,
        /*int s0, */ ಪೂರ್ಣಾಂಕ s1,  ಪೂರ್ಣಾಂಕ s2,  ಪೂರ್ಣಾಂಕ s3,
        ಪೂರ್ಣಾಂಕ s00, ಪೂರ್ಣಾಂಕ s01, ಪೂರ್ಣಾಂಕ s02, ಪೂರ್ಣಾಂಕ s03,
        ಪೂರ್ಣಾಂಕ s10, ಪೂರ್ಣಾಂಕ s11, ಪೂರ್ಣಾಂಕ s12, ಪೂರ್ಣಾಂಕ s13,
        ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i = ಅಂಶ_ct1.get_local_range(2) * ಅಂಶ_ct1.get_ಗುಂಪು(2) +
                  ಅಂಶ_ct1.get_local_id(2);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i3 = i/(ne2*ne1*ne0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i2 = (i/(ne1*ne0)) % ne2;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i1 = (i/ne0) % ne1;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i0 = i % ne0;

    ಆದರೆ (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i11 = i1 % ne11;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i12 = i2 % ne12;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i13 = i3 % ne13;

    ಸ್ಥಿರಾಂಕ size_t i_src0 =  i3*s03 +  i2*s02 +  i1*s01;
    ಸ್ಥಿರಾಂಕ size_t i_src1 = i13*s13 + i12*s12 + i11*s11;
    ಸ್ಥಿರಾಂಕ size_t i_dst  =  i3*s3  +  i2*s2  +  i1*s1;

    ಸ್ಥಿರಾಂಕ src0_t * src0_ಅಡ್ಡಸಾಲು = src0 + i_src0;
    ಸ್ಥಿರಾಂಕ src1_t * src1_ಅಡ್ಡಸಾಲು = src1 + i_src1;
    dst_t * dst_ಅಡ್ಡಸಾಲು = dst + i_dst;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i10 = i0 % ne10;
    dst_ಅಡ್ಡಸಾಲು[i0] = (dst_t)bin_op(src0 ? (ದಶಮಾಂಶ)src0_ಅಡ್ಡಸಾಲು[i0*s00] : 0.0f, (ದಶಮಾಂಶ)src1_ಅಡ್ಡಸಾಲು[i10*s10]);
}


ಮಾದರಿ<ದಶಮಾಂಶ (*bin_op)(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ)>
ರಚನೆ bin_bcast_sycl {
    ಮಾದರಿ <ಬಗೆ_ಹೆಸರು src0_t, ಬಗೆ_ಹೆಸರು src1_t, ಬಗೆ_ಹೆಸರು dst_t>
    ಶೂನ್ಯ ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ src0_t * src0_dd, ಸ್ಥಿರಾಂಕ src1_t * src1_dd, dst_t * dst_dd, ಸ್ಥಿರಾಂಕ int64_t ne00,
                    ಸ್ಥಿರಾಂಕ int64_t ne01, ಸ್ಥಿರಾಂಕ int64_t ne02, ಸ್ಥಿರಾಂಕ int64_t ne03, ಸ್ಥಿರಾಂಕ int64_t ne10, ಸ್ಥಿರಾಂಕ int64_t ne11,
                    ಸ್ಥಿರಾಂಕ int64_t ne12, ಸ್ಥಿರಾಂಕ int64_t ne13, ಸ್ಥಿರಾಂಕ int64_t ne0, ಸ್ಥಿರಾಂಕ int64_t ne1, ಸ್ಥಿರಾಂಕ int64_t ne2,
                    ಸ್ಥಿರಾಂಕ int64_t ne3, ಸ್ಥಿರಾಂಕ size_t nb00, ಸ್ಥಿರಾಂಕ size_t nb01, ಸ್ಥಿರಾಂಕ size_t nb02, ಸ್ಥಿರಾಂಕ size_t nb03,
                    ಸ್ಥಿರಾಂಕ size_t nb10, ಸ್ಥಿರಾಂಕ size_t nb11, ಸ್ಥಿರಾಂಕ size_t nb12, ಸ್ಥಿರಾಂಕ size_t nb13, ಸ್ಥಿರಾಂಕ size_t nb0,
                    ಸ್ಥಿರಾಂಕ size_t nb1, ಸ್ಥಿರಾಂಕ size_t nb2, ಸ್ಥಿರಾಂಕ size_t nb3, ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ src0_is_contiguous,
                    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ src1_is_contiguous, ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ src0_is_permuted, ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ src1_is_permuted,
                    ಸರತಿ_ptr stream) {
        ಪೂರ್ಣಾಂಕ nr0 = ne10 / ne0;
        ಪೂರ್ಣಾಂಕ nr1 = ne11/ne1;
        ಪೂರ್ಣಾಂಕ nr2 = ne12/ne2;
        ಪೂರ್ಣಾಂಕ nr3 = ne13/ne3;

        ಪೂರ್ಣಾಂಕ nr[4] = { nr0, nr1, nr2, nr3 };

        // collapse dimensions until first broadcast dimension
        int64_t cne[] = {ne0, ne1, ne2, ne3};
        int64_t cne0[] = {ne00, ne01, ne02, ne03};
        int64_t cne1[] = {ne10, ne11, ne12, ne13};
        size_t cnb[] = {nb0, nb1, nb2, nb3};
        size_t cnb0[] = {nb00, nb01, nb02, nb03};
        size_t cnb1[] = {nb10, nb11, nb12, nb13};
        ಸ್ವಯಂಚಾಲಿತ collapse = [](int64_t cne[]) {
            cne[0] *= cne[1];
            cne[1] = cne[2];
            cne[2] = cne[3];
            cne[3] = 1;
        };

        ಸ್ವಯಂಚಾಲಿತ collapse_nb = [](size_t cnb[], int64_t cne[]) {
            cnb[1] *= cne[1];
            cnb[2] *= cne[2];
            cnb[3] *= cne[3];
        };

        ಆದರೆ (src0_is_contiguous && src1_is_contiguous && !src0_is_permuted && !src1_is_permuted) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 4; i++) {
                ಆದರೆ (nr[i] != 1) {
                    ನಿಲ್ಲಿಸು;
                }
                ಆದರೆ (i > 0) {
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
            size_t s1 = nb1 / ಗಾತ್ರ(dst_t);
            size_t s2 = nb2 / ಗಾತ್ರ(dst_t);
            size_t s3 = nb3 / ಗಾತ್ರ(dst_t);

            size_t s10 = nb10 / ಗಾತ್ರ(src1_t);
            size_t s11 = nb11 / ಗಾತ್ರ(src1_t);
            size_t s12 = nb12 / ಗಾತ್ರ(src1_t);
            size_t s13 = nb13 / ಗಾತ್ರ(src1_t);

            size_t s00 = nb00 / ಗಾತ್ರ(src0_t);
            size_t s01 = nb01 / ಗಾತ್ರ(src0_t);
            size_t s02 = nb02 / ಗಾತ್ರ(src0_t);
            size_t s03 = nb03 / ಗಾತ್ರ(src0_t);

            GGML_ಬಳಸದ(s00);

            GGML_ASSERT(nb0 % ಗಾತ್ರ(dst_t) == 0);
            GGML_ASSERT(nb1 % ಗಾತ್ರ(dst_t) == 0);
            GGML_ASSERT(nb2 % ಗಾತ್ರ(dst_t) == 0);
            GGML_ASSERT(nb3 % ಗಾತ್ರ(dst_t) == 0);

            GGML_ASSERT(nb00 % ಗಾತ್ರ(src0_t) == 0);
            GGML_ASSERT(nb01 % ಗಾತ್ರ(src0_t) == 0);
            GGML_ASSERT(nb02 % ಗಾತ್ರ(src0_t) == 0);
            GGML_ASSERT(nb03 % ಗಾತ್ರ(src0_t) == 0);

            GGML_ASSERT(nb10 % ಗಾತ್ರ(src1_t) == 0);
            GGML_ASSERT(nb11 % ಗಾತ್ರ(src1_t) == 0);
            GGML_ASSERT(nb12 % ಗಾತ್ರ(src1_t) == 0);
            GGML_ASSERT(nb13 % ಗಾತ್ರ(src1_t) == 0);

            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಗಾತ್ರ_3 = 128;

            int64_t hne0 = std::max(ne0/2LL, 1LL);

            sycl::range<3> ಖಂಡ_dims(1, 1, 1);
            ಖಂಡ_dims[2] = std::min<ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ>(hne0, ಖಂಡ_ಗಾತ್ರ_3);
            ಖಂಡ_dims[1] = std::min<ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ>(
                ne1, ಖಂಡ_ಗಾತ್ರ_3 / (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ)ಖಂಡ_dims[2]);
            ಖಂಡ_dims[0] = std::min(
                std::min<ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ>(
                    ne2 * ne3, ಖಂಡ_ಗಾತ್ರ_3 / (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ)ಖಂಡ_dims[2] /
                                   (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ)ಖಂಡ_dims[1]),
                64U);

            sycl::range<3> ಖಂಡ_nums(
                (ne2 * ne3 + ಖಂಡ_dims[0] - 1) / ಖಂಡ_dims[0],
                (ne1 + ಖಂಡ_dims[1] - 1) / ಖಂಡ_dims[1],
                (hne0 + ಖಂಡ_dims[2] - 1) / ಖಂಡ_dims[2]);

            ಆದರೆ (ಖಂಡ_nums[0] > 65535) {
                // this is the maximum number of blocks in z direction, fallback to 1D grid kernel
                ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ = (ne0*ne1*ne2*ne3 + ಖಂಡ_ಗಾತ್ರ_3 - 1) / ಖಂಡ_ಗಾತ್ರ_3;
                {
                    dpct::has_capability_or_fail(stream->get_device(),
                                                 {sycl::aspect::fp16});

                    stream->parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, ಖಂಡ_ಅಂಕೆ) *
                                              sycl::range<3>(1, 1, ಖಂಡ_ಗಾತ್ರ_3),
                                          sycl::range<3>(1, 1, ಖಂಡ_ಗಾತ್ರ_3)),
                        [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                            k_bin_bcast_unravel<bin_op>(
                                src0_dd, src1_dd, dst_dd, ne0, ne1, ne2, ne3,
                                ne10, ne11, ne12, ne13, s1, s2, s3, s00, s01, s02,
                                s03, s10, s11, s12, s13, ಅಂಶ_ct1);
                        });
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                /*
                DPCT1049:16: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                dpct::has_capability_or_fail(stream->get_device(),
                                             {sycl::aspect::fp16});

                stream->parallel_for(
                    sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                    [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                        k_bin_bcast<bin_op>(src0_dd, src1_dd, dst_dd, ne0, ne1,
                                            ne2, ne3, ne10, ne11, ne12, ne13,
                                            s1, s2, s3, s00, s01, s02, s03, s10, s11, s12, s13,
                                            ಅಂಶ_ct1);
                    });
            }
        }
    }
};

ಮಾದರಿ <ವರ್ಗ op>
ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_bin_bcast(ggml_backend_sycl_context & ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1,
                                   ggml_tensor * dst) {
    dpct::ಸರತಿ_ptr main_ಹರಿವು = ctx.stream();
    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    ಆದರೆ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        op()((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src0->ದತ್ತಾಂಶ, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src1->ದತ್ತಾಂಶ, (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ, ne00, ne01, ne02, ne03, ne10,
             ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2, nb3,
             ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1), main_ಹರಿವು);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
        op()((ಸ್ಥಿರಾಂಕ sycl::half *) src0->ದತ್ತಾಂಶ, (ಸ್ಥಿರಾಂಕ sycl::half *) src1->ದತ್ತಾಂಶ, (sycl::half *) dst->ದತ್ತಾಂಶ, ne00, ne01,
             ne02, ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13,
             nb0, nb1, nb2, nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1),
             main_ಹರಿವು);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
        op()((ಸ್ಥಿರಾಂಕ sycl::half *) src0->ದತ್ತಾಂಶ, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src1->ದತ್ತಾಂಶ, (sycl::half *) dst->ದತ್ತಾಂಶ, ne00, ne01, ne02,
             ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1,
             nb2, nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1),
             main_ಹರಿವು);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I32) {
        op()((ಸ್ಥಿರಾಂಕ int32_t *) src0->ದತ್ತಾಂಶ, (ಸ್ಥಿರಾಂಕ int32_t *) src1->ದತ್ತಾಂಶ, (int32_t *) dst->ದತ್ತಾಂಶ, ne00, ne01, ne02, ne03,
             ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2,
             nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1),
             main_ಹರಿವು);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_I16 && src1->type == GGML_TYPE_I16 && dst->type == GGML_TYPE_I16) {
        op()((ಸ್ಥಿರಾಂಕ int16_t *) src0->ದತ್ತಾಂಶ, (ಸ್ಥಿರಾಂಕ int16_t *) src1->ದತ್ತಾಂಶ, (int16_t *) dst->ದತ್ತಾಂಶ, ne00, ne01, ne02, ne03,
             ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2,
             nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1),
             main_ಹರಿವು);
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_HAS_BF16
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) {
        op()((ಸ್ಥಿರಾಂಕ sycl::ext::oneapi::bfloat16 *) src0->ದತ್ತಾಂಶ, (ಸ್ಥಿರಾಂಕ sycl::ext::oneapi::bfloat16 *) src1->ದತ್ತಾಂಶ,
             (sycl::ext::oneapi::bfloat16 *) dst->ದತ್ತಾಂಶ, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2,
             ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2, nb3, ggml_is_contiguous(src0),
             ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1), main_ಹರಿವು);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_BF16) {
        op()((ಸ್ಥಿರಾಂಕ sycl::ext::oneapi::bfloat16 *) src0->ದತ್ತಾಂಶ, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src1->ದತ್ತಾಂಶ,
             (sycl::ext::oneapi::bfloat16 *) dst->ದತ್ತಾಂಶ, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2,
             ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2, nb3, ggml_is_contiguous(src0),
             ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1), main_ಹರಿವು);
#ಆದರೆ_ಅಂತ್ಯ
    } ಇಲ್ಲದಿದ್ದರೆ {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__, ggml_type_name(dst->type),
                ggml_type_name(src0->type), ggml_type_name(src1->type));
        GGML_ABORT("fatal error");
    }
}

ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_ಕೂಡಿಸು(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_ಕೂಡಿಸು>>(ctx, dst->src[0], dst->src[1], dst);
}

ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_sub(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_sub>>(ctx, dst->src[0], dst->src[1], dst);
}

ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_mul(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_mul>>(ctx, dst->src[0], dst->src[1], dst);
}

ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_div(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_div>>(ctx, dst->src[0], dst->src[1], dst);
}

ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_repeat(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_repeat>>(ctx, dst, dst->src[0], dst);
}


ಶೂನ್ಯ ggml_sycl_ಕೂಡಿಸು(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_ಕೂಡಿಸು(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_sub(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_sub(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_mul(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_mul(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_div(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_div(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_repeat(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_repeat(ctx, dst);
}

// fused ADD+ADD: dst = (src0 + src1) + src2. Same indexing as k_bin_bcast, so mixed
// types, broadcast, and non-contiguous layouts that add() already handles also fuse.
ಮಾದರಿ<ದಶಮಾಂಶ (*bin_op)(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ), ಬಗೆ_ಹೆಸರು src0_t, ಬಗೆ_ಹೆಸರು src1_t, ಬಗೆ_ಹೆಸರು src2_t, ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ k_bin_bcast3(ಸ್ಥಿರಾಂಕ src0_t * src0, ಸ್ಥಿರಾಂಕ src1_t * src1, ಸ್ಥಿರಾಂಕ src2_t * src2, dst_t * dst,
        ಪೂರ್ಣಾಂಕ ne0, ಪೂರ್ಣಾಂಕ ne1, ಪೂರ್ಣಾಂಕ ne2, ಪೂರ್ಣಾಂಕ ne3,
        ಪೂರ್ಣಾಂಕ ne10, ಪೂರ್ಣಾಂಕ ne11, ಪೂರ್ಣಾಂಕ ne12, ಪೂರ್ಣಾಂಕ ne13,
        ಪೂರ್ಣಾಂಕ ne20, ಪೂರ್ಣಾಂಕ ne21, ಪೂರ್ಣಾಂಕ ne22, ಪೂರ್ಣಾಂಕ ne23,
        ಪೂರ್ಣಾಂಕ s1, ಪೂರ್ಣಾಂಕ s2, ಪೂರ್ಣಾಂಕ s3,
        ಪೂರ್ಣಾಂಕ s00, ಪೂರ್ಣಾಂಕ s01, ಪೂರ್ಣಾಂಕ s02, ಪೂರ್ಣಾಂಕ s03,
        ಪೂರ್ಣಾಂಕ s10, ಪೂರ್ಣಾಂಕ s11, ಪೂರ್ಣಾಂಕ s12, ಪೂರ್ಣಾಂಕ s13,
        ಪೂರ್ಣಾಂಕ s20, ಪೂರ್ಣಾಂಕ s21, ಪೂರ್ಣಾಂಕ s22, ಪೂರ್ಣಾಂಕ s23,
        ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> & ಅಂಶ_ct1) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i0s = ಅಂಶ_ct1.get_local_range(2) * ಅಂಶ_ct1.get_ಗುಂಪು(2) +
                    ಅಂಶ_ct1.get_local_id(2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i1 = (ಅಂಶ_ct1.get_local_range(1) * ಅಂಶ_ct1.get_ಗುಂಪು(1) +
                    ಅಂಶ_ct1.get_local_id(1));
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i2 = (ಅಂಶ_ct1.get_local_range(0) * ಅಂಶ_ct1.get_ಗುಂಪು(0) +
                    ಅಂಶ_ct1.get_local_id(0)) /
                   ne3;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i3 = (ಅಂಶ_ct1.get_local_range(0) * ಅಂಶ_ct1.get_ಗುಂಪು(0) +
                    ಅಂಶ_ct1.get_local_id(0)) %
                   ne3;

    ಆದರೆ (i0s >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i11 = i1 % ne11;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i12 = i2 % ne12;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i13 = i3 % ne13;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i21 = i1 % ne21;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i22 = i2 % ne22;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i23 = i3 % ne23;

    ಸ್ಥಿರಾಂಕ size_t i_src0 = i3 * s03 + i2 * s02 + i1 * s01;
    ಸ್ಥಿರಾಂಕ size_t i_src1 = i13 * s13 + i12 * s12 + i11 * s11;
    ಸ್ಥಿರಾಂಕ size_t i_src2 = i23 * s23 + i22 * s22 + i21 * s21;
    ಸ್ಥಿರಾಂಕ size_t i_dst  = i3 * s3 + i2 * s2 + i1 * s1;

    ಸ್ಥಿರಾಂಕ src0_t * src0_ಅಡ್ಡಸಾಲು = src0 + i_src0;
    ಸ್ಥಿರಾಂಕ src1_t * src1_ಅಡ್ಡಸಾಲು = src1 + i_src1;
    ಸ್ಥಿರಾಂಕ src2_t * src2_ಅಡ್ಡಸಾಲು = src2 + i_src2;
    dst_t *        dst_ಅಡ್ಡಸಾಲು  = dst + i_dst;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = i0s; i0 < ne0;
         i0 += ಅಂಶ_ct1.get_local_range(2) * ಅಂಶ_ct1.get_ಗುಂಪು_range(2)) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ   i10 = i0 % ne10;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ   i20 = i0 % ne20;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ acc = bin_op((ದಶಮಾಂಶ) src0_ಅಡ್ಡಸಾಲು[i0 * s00], (ದಶಮಾಂಶ) src1_ಅಡ್ಡಸಾಲು[i10 * s10]);
        dst_ಅಡ್ಡಸಾಲು[i0]     = (dst_t) bin_op(acc, (ದಶಮಾಂಶ) src2_ಅಡ್ಡಸಾಲು[i20 * s20]);
    }
}

ಮಾದರಿ<ದಶಮಾಂಶ (*bin_op)(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ), ಬಗೆ_ಹೆಸರು src0_t, ಬಗೆ_ಹೆಸರು src1_t, ಬಗೆ_ಹೆಸರು src2_t, ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ k_bin_bcast3_unravel(ಸ್ಥಿರಾಂಕ src0_t * src0, ಸ್ಥಿರಾಂಕ src1_t * src1, ಸ್ಥಿರಾಂಕ src2_t * src2, dst_t * dst,
        ಪೂರ್ಣಾಂಕ ne0, ಪೂರ್ಣಾಂಕ ne1, ಪೂರ್ಣಾಂಕ ne2, ಪೂರ್ಣಾಂಕ ne3,
        ಪೂರ್ಣಾಂಕ ne10, ಪೂರ್ಣಾಂಕ ne11, ಪೂರ್ಣಾಂಕ ne12, ಪೂರ್ಣಾಂಕ ne13,
        ಪೂರ್ಣಾಂಕ ne20, ಪೂರ್ಣಾಂಕ ne21, ಪೂರ್ಣಾಂಕ ne22, ಪೂರ್ಣಾಂಕ ne23,
        ಪೂರ್ಣಾಂಕ s1, ಪೂರ್ಣಾಂಕ s2, ಪೂರ್ಣಾಂಕ s3,
        ಪೂರ್ಣಾಂಕ s00, ಪೂರ್ಣಾಂಕ s01, ಪೂರ್ಣಾಂಕ s02, ಪೂರ್ಣಾಂಕ s03,
        ಪೂರ್ಣಾಂಕ s10, ಪೂರ್ಣಾಂಕ s11, ಪೂರ್ಣಾಂಕ s12, ಪೂರ್ಣಾಂಕ s13,
        ಪೂರ್ಣಾಂಕ s20, ಪೂರ್ಣಾಂಕ s21, ಪೂರ್ಣಾಂಕ s22, ಪೂರ್ಣಾಂಕ s23,
        ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> & ಅಂಶ_ct1) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i = ಅಂಶ_ct1.get_local_range(2) * ಅಂಶ_ct1.get_ಗುಂಪು(2) +
                  ಅಂಶ_ct1.get_local_id(2);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i3 = i / (ne2 * ne1 * ne0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i2 = (i / (ne1 * ne0)) % ne2;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i1 = (i / ne0) % ne1;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i0 = i % ne0;

    ಆದರೆ (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i11 = i1 % ne11;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i12 = i2 % ne12;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i13 = i3 % ne13;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i21 = i1 % ne21;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i22 = i2 % ne22;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i23 = i3 % ne23;

    ಸ್ಥಿರಾಂಕ size_t i_src0 = i3 * s03 + i2 * s02 + i1 * s01;
    ಸ್ಥಿರಾಂಕ size_t i_src1 = i13 * s13 + i12 * s12 + i11 * s11;
    ಸ್ಥಿರಾಂಕ size_t i_src2 = i23 * s23 + i22 * s22 + i21 * s21;
    ಸ್ಥಿರಾಂಕ size_t i_dst  = i3 * s3 + i2 * s2 + i1 * s1;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ   i10 = i0 % ne10;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ   i20 = i0 % ne20;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ acc = bin_op((ದಶಮಾಂಶ) src0[i_src0 + i0 * s00], (ದಶಮಾಂಶ) src1[i_src1 + i10 * s10]);
    dst[i_dst + i0] = (dst_t) bin_op(acc, (ದಶಮಾಂಶ) src2[i_src2 + i20 * s20]);
}

ಮಾದರಿ<ದಶಮಾಂಶ (*bin_op)(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ), ಬಗೆ_ಹೆಸರು src0_t, ಬಗೆ_ಹೆಸರು src1_t, ಬಗೆ_ಹೆಸರು src2_t, ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ launch_bin_bcast3(ggml_backend_sycl_context & ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1,
                              ಸ್ಥಿರಾಂಕ ggml_tensor * src2, ggml_tensor * dst) {
    dpct::ಸರತಿ_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_ಗಣ_device(ctx.device));

    GGML_TENSOR_TERNARY_OP_LOCALS

    ಪೂರ್ಣಾಂಕ nr1[4] = { (ಪೂರ್ಣಾಂಕ) (ne10 / ne0), (ಪೂರ್ಣಾಂಕ) (ne11 / ne1), (ಪೂರ್ಣಾಂಕ) (ne12 / ne2), (ಪೂರ್ಣಾಂಕ) (ne13 / ne3) };
    ಪೂರ್ಣಾಂಕ nr2[4] = { (ಪೂರ್ಣಾಂಕ) (ne20 / ne0), (ಪೂರ್ಣಾಂಕ) (ne21 / ne1), (ಪೂರ್ಣಾಂಕ) (ne22 / ne2), (ಪೂರ್ಣಾಂಕ) (ne23 / ne3) };

    int64_t cne[]  = { ne0, ne1, ne2, ne3 };
    int64_t cne0[] = { ne00, ne01, ne02, ne03 };
    int64_t cne1[] = { ne10, ne11, ne12, ne13 };
    int64_t cne2[] = { ne20, ne21, ne22, ne23 };
    size_t  cnb[]  = { nb0, nb1, nb2, nb3 };
    size_t  cnb0[] = { nb00, nb01, nb02, nb03 };
    size_t  cnb1[] = { nb10, nb11, nb12, nb13 };
    size_t  cnb2[] = { nb20, nb21, nb22, nb23 };

    ಸ್ವಯಂಚಾಲಿತ collapse = [](int64_t cne[]) {
        cne[0] *= cne[1];
        cne[1] = cne[2];
        cne[2] = cne[3];
        cne[3] = 1;
    };

    ಸ್ವಯಂಚಾಲಿತ collapse_nb = [](size_t cnb[], int64_t cne[]) {
        cnb[1] *= cne[1];
        cnb[2] *= cne[2];
        cnb[3] *= cne[3];
    };

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ can_collapse = ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(src2) &&
                              !ggml_is_permuted(src0) && !ggml_is_permuted(src1) && !ggml_is_permuted(src2);
    ಆದರೆ (can_collapse) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 4; i++) {
            ಆದರೆ (nr1[i] != 1 || nr2[i] != 1) {
                ನಿಲ್ಲಿಸು;
            }
            ಆದರೆ (i > 0) {
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

        size_t s1 = cnb[1] / ಗಾತ್ರ(dst_t);
        size_t s2 = cnb[2] / ಗಾತ್ರ(dst_t);
        size_t s3 = cnb[3] / ಗಾತ್ರ(dst_t);

        size_t s00 = cnb0[0] / ಗಾತ್ರ(src0_t);
        size_t s01 = cnb0[1] / ಗಾತ್ರ(src0_t);
        size_t s02 = cnb0[2] / ಗಾತ್ರ(src0_t);
        size_t s03 = cnb0[3] / ಗಾತ್ರ(src0_t);

        size_t s10 = cnb1[0] / ಗಾತ್ರ(src1_t);
        size_t s11 = cnb1[1] / ಗಾತ್ರ(src1_t);
        size_t s12 = cnb1[2] / ಗಾತ್ರ(src1_t);
        size_t s13 = cnb1[3] / ಗಾತ್ರ(src1_t);

        size_t s20 = cnb2[0] / ಗಾತ್ರ(src2_t);
        size_t s21 = cnb2[1] / ಗಾತ್ರ(src2_t);
        size_t s22 = cnb2[2] / ಗಾತ್ರ(src2_t);
        size_t s23 = cnb2[3] / ಗಾತ್ರ(src2_t);

        GGML_ASSERT(cnb[0] % ಗಾತ್ರ(dst_t) == 0 && cnb[1] % ಗಾತ್ರ(dst_t) == 0 && cnb[2] % ಗಾತ್ರ(dst_t) == 0 &&
                    cnb[3] % ಗಾತ್ರ(dst_t) == 0);
        GGML_ASSERT(cnb0[0] % ಗಾತ್ರ(src0_t) == 0 && cnb0[1] % ಗಾತ್ರ(src0_t) == 0 && cnb0[2] % ಗಾತ್ರ(src0_t) == 0 &&
                    cnb0[3] % ಗಾತ್ರ(src0_t) == 0);
        GGML_ASSERT(cnb1[0] % ಗಾತ್ರ(src1_t) == 0 && cnb1[1] % ಗಾತ್ರ(src1_t) == 0 && cnb1[2] % ಗಾತ್ರ(src1_t) == 0 &&
                    cnb1[3] % ಗಾತ್ರ(src1_t) == 0);
        GGML_ASSERT(cnb2[0] % ಗಾತ್ರ(src2_t) == 0 && cnb2[1] % ಗಾತ್ರ(src2_t) == 0 && cnb2[2] % ಗಾತ್ರ(src2_t) == 0 &&
                    cnb2[3] % ಗಾತ್ರ(src2_t) == 0);

        ಸ್ಥಿರಾಂಕ src0_t * src0_dd = (ಸ್ಥಿರಾಂಕ src0_t *) src0->ದತ್ತಾಂಶ;
        ಸ್ಥಿರಾಂಕ src1_t * src1_dd = (ಸ್ಥಿರಾಂಕ src1_t *) src1->ದತ್ತಾಂಶ;
        ಸ್ಥಿರಾಂಕ src2_t * src2_dd = (ಸ್ಥಿರಾಂಕ src2_t *) src2->ದತ್ತಾಂಶ;
        dst_t *        dst_dd  = (dst_t *) dst->ದತ್ತಾಂಶ;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಗಾತ್ರ_3 = 128;
        int64_t   hne0       = std::max(ne0 / 2LL, 1LL);

        sycl::range<3> ಖಂಡ_dims(1, 1, 1);
        ಖಂಡ_dims[2] = std::min<ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ>(hne0, ಖಂಡ_ಗಾತ್ರ_3);
        ಖಂಡ_dims[1] = std::min<ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ>(ne1, ಖಂಡ_ಗಾತ್ರ_3 / (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ) ಖಂಡ_dims[2]);
        ಖಂಡ_dims[0] = std::min(std::min<ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ>(ne2 * ne3,
                                                        ಖಂಡ_ಗಾತ್ರ_3 / (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ) ಖಂಡ_dims[2] /
                                                            (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ) ಖಂಡ_dims[1]),
                                 64U);

        sycl::range<3> ಖಂಡ_nums((ne2 * ne3 + ಖಂಡ_dims[0] - 1) / ಖಂಡ_dims[0],
                                  (ne1 + ಖಂಡ_dims[1] - 1) / ಖಂಡ_dims[1],
                                  (hne0 + ಖಂಡ_dims[2] - 1) / ಖಂಡ_dims[2]);

        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        ಆದರೆ (ಖಂಡ_nums[0] > 65535) {
            ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ = (ne0 * ne1 * ne2 * ne3 + ಖಂಡ_ಗಾತ್ರ_3 - 1) / ಖಂಡ_ಗಾತ್ರ_3;
            stream->parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, ಖಂಡ_ಅಂಕೆ) * sycl::range<3>(1, 1, ಖಂಡ_ಗಾತ್ರ_3),
                                  sycl::range<3>(1, 1, ಖಂಡ_ಗಾತ್ರ_3)),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                    k_bin_bcast3_unravel<bin_op>(src0_dd, src1_dd, src2_dd, dst_dd, ne0, ne1, ne2, ne3, ne10, ne11,
                                                 ne12, ne13, ne20, ne21, ne22, ne23, s1, s2, s3, s00, s01, s02, s03,
                                                 s10, s11, s12, s13, s20, s21, s22, s23, ಅಂಶ_ct1);
                });
        } ಇಲ್ಲದಿದ್ದರೆ {
            stream->parallel_for(sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                                 [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                                     k_bin_bcast3<bin_op>(src0_dd, src1_dd, src2_dd, dst_dd, ne0, ne1, ne2, ne3, ne10,
                                                          ne11, ne12, ne13, ne20, ne21, ne22, ne23, s1, s2, s3, s00,
                                                          s01, s02, s03, s10, s11, s12, s13, s20, s21, s22, s23,
                                                          ಅಂಶ_ct1);
                                 });
        }
    }
}

ಶೂನ್ಯ ggml_sycl_op_ಕೂಡಿಸು_ಕೂಡಿಸು_fused(ggml_backend_sycl_context & ctx, ggml_tensor * add0, ggml_tensor * add1) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = add0->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = add0->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src2 = add1->src[1];
    ggml_tensor *       dst  = add1;

    GGML_ASSERT(add1->src[0] == add0);
    GGML_ASSERT(ggml_sycl_ಕೂಡಿಸು_kernel_supports(src0->type, src1->type, add0->type));
    GGML_ASSERT(ggml_sycl_ಕೂಡಿಸು_kernel_supports(add0->type, src2->type, dst->type));

    ಆದರೆ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32) {
        launch_bin_bcast3<op_ಕೂಡಿಸು, ದಶಮಾಂಶ, ದಶಮಾಂಶ, ದಶಮಾಂಶ, ದಶಮಾಂಶ>(ctx, src0, src1, src2, dst);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && src2->type == GGML_TYPE_F16 &&
               dst->type == GGML_TYPE_F16) {
        launch_bin_bcast3<op_ಕೂಡಿಸು, sycl::half, sycl::half, sycl::half, sycl::half>(ctx, src0, src1, src2, dst);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_F32 &&
               dst->type == GGML_TYPE_F16) {
        launch_bin_bcast3<op_ಕೂಡಿಸು, sycl::half, ದಶಮಾಂಶ, ದಶಮಾಂಶ, sycl::half>(ctx, src0, src1, src2, dst);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && src2->type == GGML_TYPE_F32 &&
               dst->type == GGML_TYPE_F16) {
        launch_bin_bcast3<op_ಕೂಡಿಸು, sycl::half, sycl::half, ದಶಮಾಂಶ, sycl::half>(ctx, src0, src1, src2, dst);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_F16 &&
               dst->type == GGML_TYPE_F16) {
        launch_bin_bcast3<op_ಕೂಡಿಸು, sycl::half, ದಶಮಾಂಶ, sycl::half, sycl::half>(ctx, src0, src1, src2, dst);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32 && src2->type == GGML_TYPE_I32 &&
               dst->type == GGML_TYPE_I32) {
        launch_bin_bcast3<op_ಕೂಡಿಸು, int32_t, int32_t, int32_t, int32_t>(ctx, src0, src1, src2, dst);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_I16 && src1->type == GGML_TYPE_I16 && src2->type == GGML_TYPE_I16 &&
               dst->type == GGML_TYPE_I16) {
        launch_bin_bcast3<op_ಕೂಡಿಸು, int16_t, int16_t, int16_t, int16_t>(ctx, src0, src1, src2, dst);
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_HAS_BF16
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && src2->type == GGML_TYPE_BF16 &&
               dst->type == GGML_TYPE_BF16) {
        launch_bin_bcast3<op_ಕೂಡಿಸು, sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16,
                          sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16>(ctx, src0, src1, src2, dst);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_F32 &&
               dst->type == GGML_TYPE_BF16) {
        launch_bin_bcast3<op_ಕೂಡಿಸು, sycl::ext::oneapi::bfloat16, ದಶಮಾಂಶ, ದಶಮಾಂಶ, sycl::ext::oneapi::bfloat16>(
            ctx, src0, src1, src2, dst);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && src2->type == GGML_TYPE_F32 &&
               dst->type == GGML_TYPE_BF16) {
        launch_bin_bcast3<op_ಕೂಡಿಸು, sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16, ದಶಮಾಂಶ,
                          sycl::ext::oneapi::bfloat16>(ctx, src0, src1, src2, dst);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_BF16 &&
               dst->type == GGML_TYPE_BF16) {
        launch_bin_bcast3<op_ಕೂಡಿಸು, sycl::ext::oneapi::bfloat16, ದಶಮಾಂಶ, sycl::ext::oneapi::bfloat16,
                          sycl::ext::oneapi::bfloat16>(ctx, src0, src1, src2, dst);
#ಆದರೆ_ಅಂತ್ಯ
    } ಇಲ್ಲದಿದ್ದರೆ {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s, src1: %s, src2: %s\n", __func__,
                ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type),
                ggml_type_name(src2->type));
        GGML_ABORT("fatal error");
    }
}

