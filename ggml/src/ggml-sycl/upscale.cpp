#ಅನ್ತರ್ಭಾವ "upscale.hpp"

ಸ್ಥಿರ ಶೂನ್ಯ upscale_f32(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb00, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb01, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb02, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb03,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne10, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne11, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne12, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne13,
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ sf0, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ sf1, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ sf2, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ sf3) {
    ಸ್ವಯಂಚಾಲಿತ ಅಂಶ_ct1 = sycl::ext::oneapi::this_work_ಅಂಶ::get_nd_ಅಂಶ<3>();
    ಪೂರ್ಣಾಂಕ  ಸೂಚ್ಯಂಕ    = ಅಂಶ_ct1.get_local_id(2) + ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(2);
    ಆದರೆ (ಸೂಚ್ಯಂಕ >= ne10 * ne11 * ne12 * ne13) {
        ಹಿಂತಿರುಗಿ;
    }

    ಪೂರ್ಣಾಂಕ i10 = ಸೂಚ್ಯಂಕ % ne10;
    ಪೂರ್ಣಾಂಕ i11 = (ಸೂಚ್ಯಂಕ / ne10) % ne11;
    ಪೂರ್ಣಾಂಕ i12 = (ಸೂಚ್ಯಂಕ / (ne10 * ne11)) % ne12;
    ಪೂರ್ಣಾಂಕ i13 = (ಸೂಚ್ಯಂಕ / (ne10 * ne11 * ne12)) % ne13;

    ಪೂರ್ಣಾಂಕ i00 = i10 / sf0;
    ಪೂರ್ಣಾಂಕ i01 = i11 / sf1;
    ಪೂರ್ಣಾಂಕ i02 = i12 / sf2;
    ಪೂರ್ಣಾಂಕ i03 = i13 / sf3;

    dst[ಸೂಚ್ಯಂಕ] = *((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ*)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ*)x + i03 * nb03 + i02 * nb02 +
                                  i01 * nb01 + i00 * nb00));
}

ಸ್ಥಿರ ಶೂನ್ಯ upscale_f32_bilinear(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb00, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb01, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb02, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb03,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne00_src, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne01_src,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne10_dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne11_dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne12_dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne13_dst,
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ sf0, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ sf1, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ sf2, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ sf3,
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ pixel_ಸ್ಥಾನಾಂತರ) {
    ಸ್ವಯಂಚಾಲಿತ ಅಂಶ_ct1 = sycl::ext::oneapi::this_work_ಅಂಶ::get_nd_ಅಂಶ<3>();
    ಸ್ಥಿರಾಂಕ int64_t ಸೂಚ್ಯಂಕ = ಅಂಶ_ct1.get_local_id(2) +
        ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(2);
    ಸ್ಥಿರಾಂಕ int64_t dst_ಒಟ್ಟು_elements = ne10_dst * ne11_dst * ne12_dst * ne13_dst;

  ಆದರೆ (ಸೂಚ್ಯಂಕ >= dst_ಒಟ್ಟು_elements) {
    ಹಿಂತಿರುಗಿ;
  }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i10_dst = ಸೂಚ್ಯಂಕ % ne10_dst;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i11_dst = (ಸೂಚ್ಯಂಕ / ne10_dst) % ne11_dst;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i12_dst = (ಸೂಚ್ಯಂಕ / (ne10_dst * ne11_dst)) % ne12_dst;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i13_dst = ಸೂಚ್ಯಂಕ / (ne10_dst * ne11_dst * ne12_dst);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i02_src = (ಪೂರ್ಣಾಂಕ)(i12_dst / sf2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i03_src = (ಪೂರ್ಣಾಂಕ)(i13_dst / sf3);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ y_src_f = ((ದಶಮಾಂಶ)i11_dst + pixel_ಸ್ಥಾನಾಂತರ) / sf1 - pixel_ಸ್ಥಾನಾಂತರ;
    ಪೂರ್ಣಾಂಕ         y0_src    = (ಪೂರ್ಣಾಂಕ) sycl::floor((ದಶಮಾಂಶ) y_src_f);
    ಪೂರ್ಣಾಂಕ y1_src    = y0_src + 1;

    y0_src = sycl::max(0, sycl::min(y0_src, ne01_src - 1));
    y1_src = sycl::max(0, sycl::min(y1_src, ne01_src - 1));

    ದಶಮಾಂಶ dy = y_src_f - (ದಶಮಾಂಶ)y0_src;
    dy       = sycl::max(0.0f, sycl::min(dy, 1.0f));

    ದಶಮಾಂಶ x_src_f = ((ದಶಮಾಂಶ)i10_dst + pixel_ಸ್ಥಾನಾಂತರ) / sf0 - pixel_ಸ್ಥಾನಾಂತರ;
    ಪೂರ್ಣಾಂಕ   x0_src    = (ಪೂರ್ಣಾಂಕ) sycl::floor(x_src_f);
    ಪೂರ್ಣಾಂಕ x1_src    = x0_src + 1;

    x0_src = sycl::max(0, sycl::min(x0_src, ne00_src - 1));
    x1_src = sycl::max(0, sycl::min(x1_src, ne00_src - 1));

    ದಶಮಾಂಶ dx = x_src_f - (ದಶಮಾಂಶ)x0_src;
    dx       = sycl::max(0.0f, sycl::min(dx, 1.0f));

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* p_a =
        (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ*)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ*)x + (int64_t)x0_src * nb00 +
                       (int64_t)y0_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* p_b =
        (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ*)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ*)x + (int64_t)x1_src * nb00 +
                       (int64_t)y0_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* p_c =
        (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ*)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ*)x + (int64_t)x0_src * nb00 +
                       (int64_t)y1_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* p_d =
        (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ*)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ*)x + (int64_t)x1_src * nb00 +
                       (int64_t)y1_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ val_a = *p_a;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ val_b = *p_b;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ val_c = *p_c;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ val_d = *p_d;

    ದಶಮಾಂಶ ಫಲಿತಾಂಶ = val_a * (1.0f - dx) * (1.0f - dy) +
                   val_b * dx * (1.0f - dy) +
                   val_c * (1.0f - dx) * dy +
                   val_d * dx * dy;

    dst[ಸೂಚ್ಯಂಕ] = ಫಲಿತಾಂಶ;
}

// Similar to F.interpolate(..., mode="bilinear", align_corners=False, antialias=True)
// https://github.com/pytorch/pytorch/blob/8871ff29b743948d1225389d5b7068f37b22750b/aten/src/ATen/native/cpu/UpSampleKernel.cpp
ಸ್ಥಿರ ಶೂನ್ಯ upscale_f32_bilinear_antialias(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src0,
                                           ದಶಮಾಂಶ *       dst,
                                           ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     nb00,
                                           ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     nb01,
                                           ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     nb02,
                                           ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     nb03,
                                           ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     ne00_src,
                                           ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     ne01_src,
                                           ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     ne10_dst,
                                           ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     ne11_dst,
                                           ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     ne12_dst,
                                           ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     ne13_dst,
                                           ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   sf0,
                                           ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   sf1,
                                           ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   sf2,
                                           ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   sf3,
                                           ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   pixel_ಸ್ಥಾನಾಂತರ) {
    ಸ್ವಯಂಚಾಲಿತ ಅಂಶ_ct1 = sycl::ext::oneapi::this_work_ಅಂಶ::get_nd_ಅಂಶ<3>();
    ಸ್ಥಿರಾಂಕ int64_t ಸೂಚ್ಯಂಕ = ಅಂಶ_ct1.get_local_id(2) +
        ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(2);
    ಸ್ಥಿರಾಂಕ int64_t dst_ಒಟ್ಟು_elements = ne10_dst * ne11_dst * ne12_dst * ne13_dst;

    ಆದರೆ (ಸೂಚ್ಯಂಕ >= dst_ಒಟ್ಟು_elements) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i10_dst = ಸೂಚ್ಯಂಕ % ne10_dst;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i11_dst = (ಸೂಚ್ಯಂಕ / ne10_dst) % ne11_dst;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i12_dst = (ಸೂಚ್ಯಂಕ / (ne10_dst * ne11_dst)) % ne12_dst;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i13_dst = ಸೂಚ್ಯಂಕ / (ne10_dst * ne11_dst * ne12_dst);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i02_src = (ಪೂರ್ಣಾಂಕ)(i12_dst / sf2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i03_src = (ಪೂರ್ಣಾಂಕ)(i13_dst / sf3);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ y = ((ದಶಮಾಂಶ)i11_dst + pixel_ಸ್ಥಾನಾಂತರ) / sf1;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x = ((ದಶಮಾಂಶ)i10_dst + pixel_ಸ್ಥಾನಾಂತರ) / sf0;

    // support and invscale, minimum 1 pixel for bilinear
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ support1  = sycl::max(1.0f / sf1, 1.0f);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ invscale1 = 1.0f / support1;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ support0  = sycl::max(1.0f / sf0, 1.0f);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ invscale0 = 1.0f / support0;

    // the range of source pixels that contribute
    ಸ್ಥಿರಾಂಕ int64_t x_min = sycl::max(int64_t(0), int64_t(x - support0 + pixel_ಸ್ಥಾನಾಂತರ));
    ಸ್ಥಿರಾಂಕ int64_t x_max = sycl::min(int64_t(ne00_src), int64_t(x + support0 + pixel_ಸ್ಥಾನಾಂತರ));
    ಸ್ಥಿರಾಂಕ int64_t y_min = sycl::max(int64_t(0), int64_t(y - support1 + pixel_ಸ್ಥಾನಾಂತರ));
    ಸ್ಥಿರಾಂಕ int64_t y_max = sycl::min(int64_t(ne01_src), int64_t(y + support1 + pixel_ಸ್ಥಾನಾಂತರ));

    // bilinear filter with antialiasing
    ದಶಮಾಂಶ val = 0.0f;
    ದಶಮಾಂಶ ಒಟ್ಟು_ತೂಕ = 0.0f;

    ಸ್ವಯಂಚಾಲಿತ triangle_filter = [](ದಶಮಾಂಶ x) -> ದಶಮಾಂಶ {
        ಹಿಂತಿರುಗಿ sycl::max(1.0f - sycl::fabs(x), 0.0f);
    };

    ಗಾಗಿ (int64_t sy = y_min; sy < y_max; sy++) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ತೂಕ_y = triangle_filter((sy - y + pixel_ಸ್ಥಾನಾಂತರ) * invscale1);

        ಗಾಗಿ (int64_t sx = x_min; sx < x_max; sx++) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ತೂಕ_x = triangle_filter((sx - x + pixel_ಸ್ಥಾನಾಂತರ) * invscale0);
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ತೂಕ = ತೂಕ_x * ತೂಕ_y;

            ಆದರೆ (ತೂಕ <= 0.0f) {
                ಮುಂದುವರಿ;
            }

            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ pixel =
                *(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ*)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ*)src0 + sx * nb00 + sy * nb01 +
                                i02_src * nb02 + i03_src * nb03);
            val += pixel * ತೂಕ;
            ಒಟ್ಟು_ತೂಕ += ತೂಕ;
        }
    }

    ಆದರೆ (ಒಟ್ಟು_ತೂಕ > 0.0f) {
        val /= ಒಟ್ಟು_ತೂಕ;
    }

    dst[ಸೂಚ್ಯಂಕ] = val;
}

ನಾಮಾಕಾಶ bicubic_interpolation {
ಸ್ಥಿರ ದಶಮಾಂಶ weight1(ದಶಮಾಂಶ x, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ &a) { ಹಿಂತಿರುಗಿ ((a + 2) * x - (a + 3)) * x * x + 1; }
ಸ್ಥಿರ ದಶಮಾಂಶ weight2(ದಶಮಾಂಶ x, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ &a) { ಹಿಂತಿರುಗಿ ((a * x - 5 * a) * x + 8 * a) * x - 4 * a; }

ಸ್ಥಿರ ದಶಮಾಂಶ bicubic(ದಶಮಾಂಶ p0, ದಶಮಾಂಶ p1, ದಶಮಾಂಶ p2, ದಶಮಾಂಶ p3, ದಶಮಾಂಶ x, ದಶಮಾಂಶ a) {
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ w0 = weight2(x + 1, a);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ w1 = weight1(x + 0, a);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ w2 = weight1(1 - x, a);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ w3 = weight2(2 - x, a);
    ಹಿಂತಿರುಗಿ p0 * w0 + p1 * w1 + p2 * w2 + p3 * w3;
}

}

ಸ್ಥಿರ ಶೂನ್ಯ upscale_f32_bicubic(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb00, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb01, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb02, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb03,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne00_src, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne01_src,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne10_dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne11_dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne12_dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne13_dst,
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ sf0, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ sf1, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ sf2, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ sf3,
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ pixel_ಸ್ಥಾನಾಂತರ) {
    ಸ್ವಯಂಚಾಲಿತ ಅಂಶ_ct1 = sycl::ext::oneapi::this_work_ಅಂಶ::get_nd_ಅಂಶ<3>();
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ a = -0.75f;
    ಬಳಸಿ bicubic_interpolation::bicubic;

    ಸ್ಥಿರಾಂಕ int64_t ಸೂಚ್ಯಂಕ = ಅಂಶ_ct1.get_local_id(2) +
        ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(2);
    ಸ್ಥಿರಾಂಕ int64_t dst_ಒಟ್ಟು_elements =
        ne10_dst * ne11_dst * ne12_dst * ne13_dst;

    ಆದರೆ (ಸೂಚ್ಯಂಕ >= dst_ಒಟ್ಟು_elements) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i10_dst = ಸೂಚ್ಯಂಕ % ne10_dst;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i11_dst = (ಸೂಚ್ಯಂಕ / ne10_dst) % ne11_dst;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i12_dst = (ಸೂಚ್ಯಂಕ / (ne10_dst * ne11_dst)) % ne12_dst;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i13_dst = ಸೂಚ್ಯಂಕ / (ne10_dst * ne11_dst * ne12_dst);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i02_src = (ಪೂರ್ಣಾಂಕ)(i12_dst / sf2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i03_src = (ಪೂರ್ಣಾಂಕ)(i13_dst / sf3);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ y_src_f = ((ದಶಮಾಂಶ)i11_dst + pixel_ಸ್ಥಾನಾಂತರ) / sf1 - pixel_ಸ್ಥಾನಾಂತರ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ   y0_src  = (ಪೂರ್ಣಾಂಕ) sycl::floor((ದಶಮಾಂಶ) y_src_f);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dy      = y_src_f - (ದಶಮಾಂಶ)y0_src;

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x_src_f = ((ದಶಮಾಂಶ)i10_dst + pixel_ಸ್ಥಾನಾಂತರ) / sf0 - pixel_ಸ್ಥಾನಾಂತರ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ   x0_src  = (ಪೂರ್ಣಾಂಕ) sycl::floor((ದಶಮಾಂಶ) x_src_f);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dx      = x_src_f - (ದಶಮಾಂಶ)x0_src;

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * x_ತಳ = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)x + (int64_t)i02_src * nb02 + (int64_t)i03_src * nb03;

    ಸ್ವಯಂಚಾಲಿತ load = [=](ಪೂರ್ಣಾಂಕ x_off, ಪೂರ್ಣಾಂಕ y_off) -> ದಶಮಾಂಶ {
        ಪೂರ್ಣಾಂಕ i00_src = sycl::max(0, sycl::min(x0_src + x_off, ne00_src - 1));
        ಪೂರ್ಣಾಂಕ i01_src = sycl::max(0, sycl::min(y0_src + y_off, ne01_src - 1));
        ಹಿಂತಿರುಗಿ *(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)(x_ತಳ + (int64_t)i00_src * nb00 + (int64_t)i01_src * nb01);
    };

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಫಲಿತಾಂಶ = bicubic(
        bicubic(load(-1, -1), load(0, -1), load(1, -1), load(2, -1), dx, a),
        bicubic(load(-1, 0), load(0, 0), load(1, 0), load(2, 0), dx, a),
        bicubic(load(-1, 1), load(0, 1), load(1, 1), load(2, 1), dx, a),
        bicubic(load(-1, 2), load(0, 2), load(1, 2), load(2, 2), dx, a),
        dy,
        a);

    dst[ಸೂಚ್ಯಂಕ] = ಫಲಿತಾಂಶ;
}

ಸ್ಥಿರ ಶೂನ್ಯ upscale_f32_sycl(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *   x,
                             ದಶಮಾಂಶ *         dst,
                             ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       nb00,
                             ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       nb01,
                             ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       nb02,
                             ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       nb03,
                             ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       ne10,
                             ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       ne11,
                             ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       ne12,
                             ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       ne13,
                             ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ     sf0,
                             ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ     sf1,
                             ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ     sf2,
                             ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ     sf3,
                             dpct::ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ int64_t dst_ಗಾತ್ರ   = ne10 * ne11 * ne12 * ne13;
    ಸ್ಥಿರಾಂಕ int64_t ಅಂಕೆ_ಖಂಡಗಳು = (dst_ಗಾತ್ರ + SYCL_UPSCALE_ಖಂಡ_ಗಾತ್ರ - 1) / SYCL_UPSCALE_ಖಂಡ_ಗಾತ್ರ;

    stream->parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, ಅಂಕೆ_ಖಂಡಗಳು) * sycl::range<3>(1, 1, SYCL_UPSCALE_ಖಂಡ_ಗಾತ್ರ),
             sycl::range<3>(1, 1, SYCL_UPSCALE_ಖಂಡ_ಗಾತ್ರ)),
        [=](sycl::nd_ಅಂಶ<3> /*item_ct1*/) {
            upscale_f32(x, dst, nb00, nb01, nb02, nb03, ne10, ne11, ne12, ne13, sf0, sf1, sf2, sf3);
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ upscale_f32_bilinear_sycl(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *   x,
                                      ದಶಮಾಂಶ *         dst,
                                      ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       nb00,
                                      ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       nb01,
                                      ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       nb02,
                                      ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       nb03,
                                      ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       ne00_src,
                                      ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       ne01_src,
                                      ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       ne10_dst,
                                      ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       ne11_dst,
                                      ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       ne12_dst,
                                      ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       ne13_dst,
                                      ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ     sf0,
                                      ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ     sf1,
                                      ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ     sf2,
                                      ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ     sf3,
                                      ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ     pixel_ಸ್ಥಾನಾಂತರ,
                                      ತಾರ್ಕಿಕ            antialias,
                                      dpct::ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ int64_t dst_ಗಾತ್ರ   = ne10_dst * ne11_dst * ne12_dst * ne13_dst;
    ಸ್ಥಿರಾಂಕ int64_t ಅಂಕೆ_ಖಂಡಗಳು = (dst_ಗಾತ್ರ + SYCL_UPSCALE_ಖಂಡ_ಗಾತ್ರ - 1) / SYCL_UPSCALE_ಖಂಡ_ಗಾತ್ರ;

    ಆದರೆ (antialias) {
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, ಅಂಕೆ_ಖಂಡಗಳು) * sycl::range<3>(1, 1, SYCL_UPSCALE_ಖಂಡ_ಗಾತ್ರ),
                sycl::range<3>(1, 1, SYCL_UPSCALE_ಖಂಡ_ಗಾತ್ರ)),
            [=](sycl::nd_ಅಂಶ<3> /*item_ct1*/) {
                upscale_f32_bilinear_antialias(
                    x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst,
                    ne12_dst, ne13_dst, sf0, sf1, sf2, sf3, pixel_ಸ್ಥಾನಾಂತರ);
            });
    } ಇಲ್ಲದಿದ್ದರೆ {
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, ಅಂಕೆ_ಖಂಡಗಳು) * sycl::range<3>(1, 1, SYCL_UPSCALE_ಖಂಡ_ಗಾತ್ರ),
                sycl::range<3>(1, 1, SYCL_UPSCALE_ಖಂಡ_ಗಾತ್ರ)),
            [=](sycl::nd_ಅಂಶ<3> /*item_ct1*/) {
                upscale_f32_bilinear(
                    x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst, ne12_dst,
                    ne13_dst, sf0, sf1, sf2, sf3, pixel_ಸ್ಥಾನಾಂತರ);
            });
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ upscale_f32_bicubic_sycl(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *   x,
                                     ದಶಮಾಂಶ *         dst,
                                     ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       nb00,
                                     ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       nb01,
                                     ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       nb02,
                                     ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       nb03,
                                     ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       ne00_src,
                                     ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       ne01_src,
                                     ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       ne10_dst,
                                     ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       ne11_dst,
                                     ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       ne12_dst,
                                     ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       ne13_dst,
                                     ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ     sf0,
                                     ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ     sf1,
                                     ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ     sf2,
                                     ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ     sf3,
                                     ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ     pixel_ಸ್ಥಾನಾಂತರ,
                                     dpct::ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ int64_t dst_ಗಾತ್ರ   = ne10_dst * ne11_dst * ne12_dst * ne13_dst;
    ಸ್ಥಿರಾಂಕ int64_t ಅಂಕೆ_ಖಂಡಗಳು = (dst_ಗಾತ್ರ + SYCL_UPSCALE_ಖಂಡ_ಗಾತ್ರ - 1) / SYCL_UPSCALE_ಖಂಡ_ಗಾತ್ರ;

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, 1, ಅಂಕೆ_ಖಂಡಗಳು) * sycl::range<3>(1, 1, SYCL_UPSCALE_ಖಂಡ_ಗಾತ್ರ),
                    sycl::range<3>(1, 1, SYCL_UPSCALE_ಖಂಡ_ಗಾತ್ರ)),
                [=](sycl::nd_ಅಂಶ<3> /*item_ct1*/) {
                    upscale_f32_bicubic(
                        x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst,
                        ne12_dst, ne13_dst, sf0, sf1, sf2, sf3, pixel_ಸ್ಥಾನಾಂತರ);
                });
        });
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_op_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src0_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)src0->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * dst_d = (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ;
    dpct::ಸರತಿ_ptr     stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಬಹುಲಕ_flags = dst->op_params[0];
    ಸ್ಥಿರಾಂಕ ggml_scale_ಬಹುಲಕ mode = (ggml_scale_ಬಹುಲಕ)(ಬಹುಲಕ_flags & 0xFF);

    ದಶಮಾಂಶ sf0 = (ದಶಮಾಂಶ)dst->ne[0]/src0->ne[0];
    ದಶಮಾಂಶ sf1 = (ದಶಮಾಂಶ)dst->ne[1]/src0->ne[1];
    ದಶಮಾಂಶ sf2 = (ದಶಮಾಂಶ)dst->ne[2]/src0->ne[2];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ sf3 = (ದಶಮಾಂಶ)dst->ne[3]/src0->ne[3];

    ದಶಮಾಂಶ pixel_ಸ್ಥಾನಾಂತರ = 0.5f;
    ಆದರೆ (ಬಹುಲಕ_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        sf0 = dst->ne[0] > 1 && src0->ne[0] > 1
            ? (ದಶಮಾಂಶ)(dst->ne[0] - 1) / (src0->ne[0] - 1)
            : sf0;
        sf1 = dst->ne[1] > 1 && src0->ne[1] > 1
            ? (ದಶಮಾಂಶ)(dst->ne[1] - 1) / (src0->ne[1] - 1)
            : sf1;
        pixel_ಸ್ಥಾನಾಂತರ = 0.0f;
    }

    ಆದರೆ (mode == GGML_SCALE_ಬಹುಲಕ_ಹತ್ತಿರದ) {
        upscale_f32_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], sf0, sf1, sf2, sf3, stream);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (mode == GGML_SCALE_ಬಹುಲಕ_BILINEAR) {
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ antialias = (ಬಹುಲಕ_flags & GGML_SCALE_FLAG_ANTIALIAS);
        upscale_f32_bilinear_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src0->ne[0], src0->ne[1], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            sf0, sf1, sf2, sf3, pixel_ಸ್ಥಾನಾಂತರ, antialias, stream);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (mode == GGML_SCALE_ಬಹುಲಕ_BICUBIC) {
        upscale_f32_bicubic_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src0->ne[0], src0->ne[1], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            sf0, sf1, sf2, sf3, pixel_ಸ್ಥಾನಾಂತರ, stream);
    }
}

ಶೂನ್ಯ ggml_sycl_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_upscale(ctx, dst);
}
