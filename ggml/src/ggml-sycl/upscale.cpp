#ਅਨ੍ਤਰ੍ਭਾਵ "upscale.hpp"

ਸਥਿਰ ਖਾਲੀ upscale_f32(ਅਚਲ ਦਸ਼ਮਲਵ * x, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
        ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ ne13,
        ਅਚਲ ਦਸ਼ਮਲਵ sf0, ਅਚਲ ਦਸ਼ਮਲਵ sf1, ਅਚਲ ਦਸ਼ਮਲਵ sf2, ਅਚਲ ਦਸ਼ਮਲਵ sf3) {
    ਸਵੈਚਾਲਿਤ ਵਸਤੂ_ct1 = sycl::ext::oneapi::this_work_ਵਸਤੂ::get_nd_ਵਸਤੂ<3>();
    ਪੂਰਨ  ਸੂਚਕਾਂਕ    = ਵਸਤੂ_ct1.get_local_id(2) + ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(2);
    ਜੇ (ਸੂਚਕਾਂਕ >= ne10 * ne11 * ne12 * ne13) {
        ਵਾਪਸ;
    }

    ਪੂਰਨ i10 = ਸੂਚਕਾਂਕ % ne10;
    ਪੂਰਨ i11 = (ਸੂਚਕਾਂਕ / ne10) % ne11;
    ਪੂਰਨ i12 = (ਸੂਚਕਾਂਕ / (ne10 * ne11)) % ne12;
    ਪੂਰਨ i13 = (ਸੂਚਕਾਂਕ / (ne10 * ne11 * ne12)) % ne13;

    ਪੂਰਨ i00 = i10 / sf0;
    ਪੂਰਨ i01 = i11 / sf1;
    ਪੂਰਨ i02 = i12 / sf2;
    ਪੂਰਨ i03 = i13 / sf3;

    dst[ਸੂਚਕਾਂਕ] = *((ਅਚਲ ਦਸ਼ਮਲਵ*)((ਅਚਲ ਅੱਖਰ*)x + i03 * nb03 + i02 * nb02 +
                                  i01 * nb01 + i00 * nb00));
}

ਸਥਿਰ ਖਾਲੀ upscale_f32_bilinear(ਅਚਲ ਦਸ਼ਮਲਵ * x, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
        ਅਚਲ ਪੂਰਨ ne00_src, ਅਚਲ ਪੂਰਨ ne01_src,
        ਅਚਲ ਪੂਰਨ ne10_dst, ਅਚਲ ਪੂਰਨ ne11_dst, ਅਚਲ ਪੂਰਨ ne12_dst, ਅਚਲ ਪੂਰਨ ne13_dst,
        ਅਚਲ ਦਸ਼ਮਲਵ sf0, ਅਚਲ ਦਸ਼ਮਲਵ sf1, ਅਚਲ ਦਸ਼ਮਲਵ sf2, ਅਚਲ ਦਸ਼ਮਲਵ sf3,
        ਅਚਲ ਦਸ਼ਮਲਵ pixel_ਵਿੱਥ) {
    ਸਵੈਚਾਲਿਤ ਵਸਤੂ_ct1 = sycl::ext::oneapi::this_work_ਵਸਤੂ::get_nd_ਵਸਤੂ<3>();
    ਅਚਲ int64_t ਸੂਚਕਾਂਕ = ਵਸਤੂ_ct1.get_local_id(2) +
        ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(2);
    ਅਚਲ int64_t dst_ਕੁੱਲ_elements = ne10_dst * ne11_dst * ne12_dst * ne13_dst;

  ਜੇ (ਸੂਚਕਾਂਕ >= dst_ਕੁੱਲ_elements) {
    ਵਾਪਸ;
  }

    ਅਚਲ ਪੂਰਨ i10_dst = ਸੂਚਕਾਂਕ % ne10_dst;
    ਅਚਲ ਪੂਰਨ i11_dst = (ਸੂਚਕਾਂਕ / ne10_dst) % ne11_dst;
    ਅਚਲ ਪੂਰਨ i12_dst = (ਸੂਚਕਾਂਕ / (ne10_dst * ne11_dst)) % ne12_dst;
    ਅਚਲ ਪੂਰਨ i13_dst = ਸੂਚਕਾਂਕ / (ne10_dst * ne11_dst * ne12_dst);

    ਅਚਲ ਪੂਰਨ i02_src = (ਪੂਰਨ)(i12_dst / sf2);
    ਅਚਲ ਪੂਰਨ i03_src = (ਪੂਰਨ)(i13_dst / sf3);

    ਅਚਲ ਦਸ਼ਮਲਵ y_src_f = ((ਦਸ਼ਮਲਵ)i11_dst + pixel_ਵਿੱਥ) / sf1 - pixel_ਵਿੱਥ;
    ਪੂਰਨ         y0_src    = (ਪੂਰਨ) sycl::floor((ਦਸ਼ਮਲਵ) y_src_f);
    ਪੂਰਨ y1_src    = y0_src + 1;

    y0_src = sycl::max(0, sycl::min(y0_src, ne01_src - 1));
    y1_src = sycl::max(0, sycl::min(y1_src, ne01_src - 1));

    ਦਸ਼ਮਲਵ dy = y_src_f - (ਦਸ਼ਮਲਵ)y0_src;
    dy       = sycl::max(0.0f, sycl::min(dy, 1.0f));

    ਦਸ਼ਮਲਵ x_src_f = ((ਦਸ਼ਮਲਵ)i10_dst + pixel_ਵਿੱਥ) / sf0 - pixel_ਵਿੱਥ;
    ਪੂਰਨ   x0_src    = (ਪੂਰਨ) sycl::floor(x_src_f);
    ਪੂਰਨ x1_src    = x0_src + 1;

    x0_src = sycl::max(0, sycl::min(x0_src, ne00_src - 1));
    x1_src = sycl::max(0, sycl::min(x1_src, ne00_src - 1));

    ਦਸ਼ਮਲਵ dx = x_src_f - (ਦਸ਼ਮਲਵ)x0_src;
    dx       = sycl::max(0.0f, sycl::min(dx, 1.0f));

    ਅਚਲ ਦਸ਼ਮਲਵ* p_a =
        (ਅਚਲ ਦਸ਼ਮਲਵ*)((ਅਚਲ ਅੱਖਰ*)x + (int64_t)x0_src * nb00 +
                       (int64_t)y0_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    ਅਚਲ ਦਸ਼ਮਲਵ* p_b =
        (ਅਚਲ ਦਸ਼ਮਲਵ*)((ਅਚਲ ਅੱਖਰ*)x + (int64_t)x1_src * nb00 +
                       (int64_t)y0_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    ਅਚਲ ਦਸ਼ਮਲਵ* p_c =
        (ਅਚਲ ਦਸ਼ਮਲਵ*)((ਅਚਲ ਅੱਖਰ*)x + (int64_t)x0_src * nb00 +
                       (int64_t)y1_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    ਅਚਲ ਦਸ਼ਮਲਵ* p_d =
        (ਅਚਲ ਦਸ਼ਮਲਵ*)((ਅਚਲ ਅੱਖਰ*)x + (int64_t)x1_src * nb00 +
                       (int64_t)y1_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);

    ਅਚਲ ਦਸ਼ਮਲਵ val_a = *p_a;
    ਅਚਲ ਦਸ਼ਮਲਵ val_b = *p_b;
    ਅਚਲ ਦਸ਼ਮਲਵ val_c = *p_c;
    ਅਚਲ ਦਸ਼ਮਲਵ val_d = *p_d;

    ਦਸ਼ਮਲਵ ਨਤੀਜਾ = val_a * (1.0f - dx) * (1.0f - dy) +
                   val_b * dx * (1.0f - dy) +
                   val_c * (1.0f - dx) * dy +
                   val_d * dx * dy;

    dst[ਸੂਚਕਾਂਕ] = ਨਤੀਜਾ;
}

// Similar to F.interpolate(..., mode="bilinear", align_corners=False, antialias=True)
// https://github.com/pytorch/pytorch/blob/8871ff29b743948d1225389d5b7068f37b22750b/aten/src/ATen/native/cpu/UpSampleKernel.cpp
ਸਥਿਰ ਖਾਲੀ upscale_f32_bilinear_antialias(ਅਚਲ ਦਸ਼ਮਲਵ * src0,
                                           ਦਸ਼ਮਲਵ *       dst,
                                           ਅਚਲ ਪੂਰਨ     nb00,
                                           ਅਚਲ ਪੂਰਨ     nb01,
                                           ਅਚਲ ਪੂਰਨ     nb02,
                                           ਅਚਲ ਪੂਰਨ     nb03,
                                           ਅਚਲ ਪੂਰਨ     ne00_src,
                                           ਅਚਲ ਪੂਰਨ     ne01_src,
                                           ਅਚਲ ਪੂਰਨ     ne10_dst,
                                           ਅਚਲ ਪੂਰਨ     ne11_dst,
                                           ਅਚਲ ਪੂਰਨ     ne12_dst,
                                           ਅਚਲ ਪੂਰਨ     ne13_dst,
                                           ਅਚਲ ਦਸ਼ਮਲਵ   sf0,
                                           ਅਚਲ ਦਸ਼ਮਲਵ   sf1,
                                           ਅਚਲ ਦਸ਼ਮਲਵ   sf2,
                                           ਅਚਲ ਦਸ਼ਮਲਵ   sf3,
                                           ਅਚਲ ਦਸ਼ਮਲਵ   pixel_ਵਿੱਥ) {
    ਸਵੈਚਾਲਿਤ ਵਸਤੂ_ct1 = sycl::ext::oneapi::this_work_ਵਸਤੂ::get_nd_ਵਸਤੂ<3>();
    ਅਚਲ int64_t ਸੂਚਕਾਂਕ = ਵਸਤੂ_ct1.get_local_id(2) +
        ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(2);
    ਅਚਲ int64_t dst_ਕੁੱਲ_elements = ne10_dst * ne11_dst * ne12_dst * ne13_dst;

    ਜੇ (ਸੂਚਕਾਂਕ >= dst_ਕੁੱਲ_elements) {
        ਵਾਪਸ;
    }

    ਅਚਲ ਪੂਰਨ i10_dst = ਸੂਚਕਾਂਕ % ne10_dst;
    ਅਚਲ ਪੂਰਨ i11_dst = (ਸੂਚਕਾਂਕ / ne10_dst) % ne11_dst;
    ਅਚਲ ਪੂਰਨ i12_dst = (ਸੂਚਕਾਂਕ / (ne10_dst * ne11_dst)) % ne12_dst;
    ਅਚਲ ਪੂਰਨ i13_dst = ਸੂਚਕਾਂਕ / (ne10_dst * ne11_dst * ne12_dst);

    ਅਚਲ ਪੂਰਨ i02_src = (ਪੂਰਨ)(i12_dst / sf2);
    ਅਚਲ ਪੂਰਨ i03_src = (ਪੂਰਨ)(i13_dst / sf3);

    ਅਚਲ ਦਸ਼ਮਲਵ y = ((ਦਸ਼ਮਲਵ)i11_dst + pixel_ਵਿੱਥ) / sf1;
    ਅਚਲ ਦਸ਼ਮਲਵ x = ((ਦਸ਼ਮਲਵ)i10_dst + pixel_ਵਿੱਥ) / sf0;

    // support and invscale, minimum 1 pixel for bilinear
    ਅਚਲ ਦਸ਼ਮਲਵ support1  = sycl::max(1.0f / sf1, 1.0f);
    ਅਚਲ ਦਸ਼ਮਲਵ invscale1 = 1.0f / support1;
    ਅਚਲ ਦਸ਼ਮਲਵ support0  = sycl::max(1.0f / sf0, 1.0f);
    ਅਚਲ ਦਸ਼ਮਲਵ invscale0 = 1.0f / support0;

    // the range of source pixels that contribute
    ਅਚਲ int64_t x_min = sycl::max(int64_t(0), int64_t(x - support0 + pixel_ਵਿੱਥ));
    ਅਚਲ int64_t x_max = sycl::min(int64_t(ne00_src), int64_t(x + support0 + pixel_ਵਿੱਥ));
    ਅਚਲ int64_t y_min = sycl::max(int64_t(0), int64_t(y - support1 + pixel_ਵਿੱਥ));
    ਅਚਲ int64_t y_max = sycl::min(int64_t(ne01_src), int64_t(y + support1 + pixel_ਵਿੱਥ));

    // bilinear filter with antialiasing
    ਦਸ਼ਮਲਵ val = 0.0f;
    ਦਸ਼ਮਲਵ ਕੁੱਲ_ਭਾਰ = 0.0f;

    ਸਵੈਚਾਲਿਤ triangle_filter = [](ਦਸ਼ਮਲਵ x) -> ਦਸ਼ਮਲਵ {
        ਵਾਪਸ sycl::max(1.0f - sycl::fabs(x), 0.0f);
    };

    ਲਈ (int64_t sy = y_min; sy < y_max; sy++) {
        ਅਚਲ ਦਸ਼ਮਲਵ ਭਾਰ_y = triangle_filter((sy - y + pixel_ਵਿੱਥ) * invscale1);

        ਲਈ (int64_t sx = x_min; sx < x_max; sx++) {
            ਅਚਲ ਦਸ਼ਮਲਵ ਭਾਰ_x = triangle_filter((sx - x + pixel_ਵਿੱਥ) * invscale0);
            ਅਚਲ ਦਸ਼ਮਲਵ ਭਾਰ = ਭਾਰ_x * ਭਾਰ_y;

            ਜੇ (ਭਾਰ <= 0.0f) {
                ਜਾਰੀ;
            }

            ਅਚਲ ਦਸ਼ਮਲਵ pixel =
                *(ਅਚਲ ਦਸ਼ਮਲਵ*)((ਅਚਲ ਅੱਖਰ*)src0 + sx * nb00 + sy * nb01 +
                                i02_src * nb02 + i03_src * nb03);
            val += pixel * ਭਾਰ;
            ਕੁੱਲ_ਭਾਰ += ਭਾਰ;
        }
    }

    ਜੇ (ਕੁੱਲ_ਭਾਰ > 0.0f) {
        val /= ਕੁੱਲ_ਭਾਰ;
    }

    dst[ਸੂਚਕਾਂਕ] = val;
}

ਨਾਮਾਕਾਸ਼ bicubic_interpolation {
ਸਥਿਰ ਦਸ਼ਮਲਵ weight1(ਦਸ਼ਮਲਵ x, ਅਚਲ ਦਸ਼ਮਲਵ &a) { ਵਾਪਸ ((a + 2) * x - (a + 3)) * x * x + 1; }
ਸਥਿਰ ਦਸ਼ਮਲਵ weight2(ਦਸ਼ਮਲਵ x, ਅਚਲ ਦਸ਼ਮਲਵ &a) { ਵਾਪਸ ((a * x - 5 * a) * x + 8 * a) * x - 4 * a; }

ਸਥਿਰ ਦਸ਼ਮਲਵ bicubic(ਦਸ਼ਮਲਵ p0, ਦਸ਼ਮਲਵ p1, ਦਸ਼ਮਲਵ p2, ਦਸ਼ਮਲਵ p3, ਦਸ਼ਮਲਵ x, ਦਸ਼ਮਲਵ a) {
    ਅਚਲ ਦਸ਼ਮਲਵ w0 = weight2(x + 1, a);
    ਅਚਲ ਦਸ਼ਮਲਵ w1 = weight1(x + 0, a);
    ਅਚਲ ਦਸ਼ਮਲਵ w2 = weight1(1 - x, a);
    ਅਚਲ ਦਸ਼ਮਲਵ w3 = weight2(2 - x, a);
    ਵਾਪਸ p0 * w0 + p1 * w1 + p2 * w2 + p3 * w3;
}

}

ਸਥਿਰ ਖਾਲੀ upscale_f32_bicubic(ਅਚਲ ਦਸ਼ਮਲਵ * x, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
        ਅਚਲ ਪੂਰਨ ne00_src, ਅਚਲ ਪੂਰਨ ne01_src,
        ਅਚਲ ਪੂਰਨ ne10_dst, ਅਚਲ ਪੂਰਨ ne11_dst, ਅਚਲ ਪੂਰਨ ne12_dst, ਅਚਲ ਪੂਰਨ ne13_dst,
        ਅਚਲ ਦਸ਼ਮਲਵ sf0, ਅਚਲ ਦਸ਼ਮਲਵ sf1, ਅਚਲ ਦਸ਼ਮਲਵ sf2, ਅਚਲ ਦਸ਼ਮਲਵ sf3,
        ਅਚਲ ਦਸ਼ਮਲਵ pixel_ਵਿੱਥ) {
    ਸਵੈਚਾਲਿਤ ਵਸਤੂ_ct1 = sycl::ext::oneapi::this_work_ਵਸਤੂ::get_nd_ਵਸਤੂ<3>();
    ਅਚਲ ਦਸ਼ਮਲਵ a = -0.75f;
    ਵਰਤੋਂ bicubic_interpolation::bicubic;

    ਅਚਲ int64_t ਸੂਚਕਾਂਕ = ਵਸਤੂ_ct1.get_local_id(2) +
        ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(2);
    ਅਚਲ int64_t dst_ਕੁੱਲ_elements =
        ne10_dst * ne11_dst * ne12_dst * ne13_dst;

    ਜੇ (ਸੂਚਕਾਂਕ >= dst_ਕੁੱਲ_elements) {
        ਵਾਪਸ;
    }

    ਅਚਲ ਪੂਰਨ i10_dst = ਸੂਚਕਾਂਕ % ne10_dst;
    ਅਚਲ ਪੂਰਨ i11_dst = (ਸੂਚਕਾਂਕ / ne10_dst) % ne11_dst;
    ਅਚਲ ਪੂਰਨ i12_dst = (ਸੂਚਕਾਂਕ / (ne10_dst * ne11_dst)) % ne12_dst;
    ਅਚਲ ਪੂਰਨ i13_dst = ਸੂਚਕਾਂਕ / (ne10_dst * ne11_dst * ne12_dst);

    ਅਚਲ ਪੂਰਨ i02_src = (ਪੂਰਨ)(i12_dst / sf2);
    ਅਚਲ ਪੂਰਨ i03_src = (ਪੂਰਨ)(i13_dst / sf3);

    ਅਚਲ ਦਸ਼ਮਲਵ y_src_f = ((ਦਸ਼ਮਲਵ)i11_dst + pixel_ਵਿੱਥ) / sf1 - pixel_ਵਿੱਥ;
    ਅਚਲ ਪੂਰਨ   y0_src  = (ਪੂਰਨ) sycl::floor((ਦਸ਼ਮਲਵ) y_src_f);
    ਅਚਲ ਦਸ਼ਮਲਵ dy      = y_src_f - (ਦਸ਼ਮਲਵ)y0_src;

    ਅਚਲ ਦਸ਼ਮਲਵ x_src_f = ((ਦਸ਼ਮਲਵ)i10_dst + pixel_ਵਿੱਥ) / sf0 - pixel_ਵਿੱਥ;
    ਅਚਲ ਪੂਰਨ   x0_src  = (ਪੂਰਨ) sycl::floor((ਦਸ਼ਮਲਵ) x_src_f);
    ਅਚਲ ਦਸ਼ਮਲਵ dx      = x_src_f - (ਦਸ਼ਮਲਵ)x0_src;

    ਅਚਲ ਅੱਖਰ * x_ਅਧਾਰ = (ਅਚਲ ਅੱਖਰ *)x + (int64_t)i02_src * nb02 + (int64_t)i03_src * nb03;

    ਸਵੈਚਾਲਿਤ load = [=](ਪੂਰਨ x_off, ਪੂਰਨ y_off) -> ਦਸ਼ਮਲਵ {
        ਪੂਰਨ i00_src = sycl::max(0, sycl::min(x0_src + x_off, ne00_src - 1));
        ਪੂਰਨ i01_src = sycl::max(0, sycl::min(y0_src + y_off, ne01_src - 1));
        ਵਾਪਸ *(ਅਚਲ ਦਸ਼ਮਲਵ *)(x_ਅਧਾਰ + (int64_t)i00_src * nb00 + (int64_t)i01_src * nb01);
    };

    ਅਚਲ ਦਸ਼ਮਲਵ ਨਤੀਜਾ = bicubic(
        bicubic(load(-1, -1), load(0, -1), load(1, -1), load(2, -1), dx, a),
        bicubic(load(-1, 0), load(0, 0), load(1, 0), load(2, 0), dx, a),
        bicubic(load(-1, 1), load(0, 1), load(1, 1), load(2, 1), dx, a),
        bicubic(load(-1, 2), load(0, 2), load(1, 2), load(2, 2), dx, a),
        dy,
        a);

    dst[ਸੂਚਕਾਂਕ] = ਨਤੀਜਾ;
}

ਸਥਿਰ ਖਾਲੀ upscale_f32_sycl(ਅਚਲ ਦਸ਼ਮਲਵ *   x,
                             ਦਸ਼ਮਲਵ *         dst,
                             ਅਚਲ ਪੂਰਨ       nb00,
                             ਅਚਲ ਪੂਰਨ       nb01,
                             ਅਚਲ ਪੂਰਨ       nb02,
                             ਅਚਲ ਪੂਰਨ       nb03,
                             ਅਚਲ ਪੂਰਨ       ne10,
                             ਅਚਲ ਪੂਰਨ       ne11,
                             ਅਚਲ ਪੂਰਨ       ne12,
                             ਅਚਲ ਪੂਰਨ       ne13,
                             ਅਚਲ ਦਸ਼ਮਲਵ     sf0,
                             ਅਚਲ ਦਸ਼ਮਲਵ     sf1,
                             ਅਚਲ ਦਸ਼ਮਲਵ     sf2,
                             ਅਚਲ ਦਸ਼ਮਲਵ     sf3,
                             dpct::ਕਤਾਰ_ptr stream) {
    ਅਚਲ int64_t dst_ਮਾਪ   = ne10 * ne11 * ne12 * ne13;
    ਅਚਲ int64_t ਅੰਕ_ਖੰਡਾਂ = (dst_ਮਾਪ + SYCL_UPSCALE_ਖੰਡ_ਮਾਪ - 1) / SYCL_UPSCALE_ਖੰਡ_ਮਾਪ;

    stream->parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_UPSCALE_ਖੰਡ_ਮਾਪ),
             sycl::range<3>(1, 1, SYCL_UPSCALE_ਖੰਡ_ਮਾਪ)),
        [=](sycl::nd_ਵਸਤੂ<3> /*item_ct1*/) {
            upscale_f32(x, dst, nb00, nb01, nb02, nb03, ne10, ne11, ne12, ne13, sf0, sf1, sf2, sf3);
        });
}

ਸਥਿਰ ਖਾਲੀ upscale_f32_bilinear_sycl(ਅਚਲ ਦਸ਼ਮਲਵ *   x,
                                      ਦਸ਼ਮਲਵ *         dst,
                                      ਅਚਲ ਪੂਰਨ       nb00,
                                      ਅਚਲ ਪੂਰਨ       nb01,
                                      ਅਚਲ ਪੂਰਨ       nb02,
                                      ਅਚਲ ਪੂਰਨ       nb03,
                                      ਅਚਲ ਪੂਰਨ       ne00_src,
                                      ਅਚਲ ਪੂਰਨ       ne01_src,
                                      ਅਚਲ ਪੂਰਨ       ne10_dst,
                                      ਅਚਲ ਪੂਰਨ       ne11_dst,
                                      ਅਚਲ ਪੂਰਨ       ne12_dst,
                                      ਅਚਲ ਪੂਰਨ       ne13_dst,
                                      ਅਚਲ ਦਸ਼ਮਲਵ     sf0,
                                      ਅਚਲ ਦਸ਼ਮਲਵ     sf1,
                                      ਅਚਲ ਦਸ਼ਮਲਵ     sf2,
                                      ਅਚਲ ਦਸ਼ਮਲਵ     sf3,
                                      ਅਚਲ ਦਸ਼ਮਲਵ     pixel_ਵਿੱਥ,
                                      ਤਾਰਕਿਕ            antialias,
                                      dpct::ਕਤਾਰ_ptr stream) {
    ਅਚਲ int64_t dst_ਮਾਪ   = ne10_dst * ne11_dst * ne12_dst * ne13_dst;
    ਅਚਲ int64_t ਅੰਕ_ਖੰਡਾਂ = (dst_ਮਾਪ + SYCL_UPSCALE_ਖੰਡ_ਮਾਪ - 1) / SYCL_UPSCALE_ਖੰਡ_ਮਾਪ;

    ਜੇ (antialias) {
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_UPSCALE_ਖੰਡ_ਮਾਪ),
                sycl::range<3>(1, 1, SYCL_UPSCALE_ਖੰਡ_ਮਾਪ)),
            [=](sycl::nd_ਵਸਤੂ<3> /*item_ct1*/) {
                upscale_f32_bilinear_antialias(
                    x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst,
                    ne12_dst, ne13_dst, sf0, sf1, sf2, sf3, pixel_ਵਿੱਥ);
            });
    } ਨਹੀਂਤਾਂ {
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_UPSCALE_ਖੰਡ_ਮਾਪ),
                sycl::range<3>(1, 1, SYCL_UPSCALE_ਖੰਡ_ਮਾਪ)),
            [=](sycl::nd_ਵਸਤੂ<3> /*item_ct1*/) {
                upscale_f32_bilinear(
                    x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst, ne12_dst,
                    ne13_dst, sf0, sf1, sf2, sf3, pixel_ਵਿੱਥ);
            });
    }
}

ਸਥਿਰ ਖਾਲੀ upscale_f32_bicubic_sycl(ਅਚਲ ਦਸ਼ਮਲਵ *   x,
                                     ਦਸ਼ਮਲਵ *         dst,
                                     ਅਚਲ ਪੂਰਨ       nb00,
                                     ਅਚਲ ਪੂਰਨ       nb01,
                                     ਅਚਲ ਪੂਰਨ       nb02,
                                     ਅਚਲ ਪੂਰਨ       nb03,
                                     ਅਚਲ ਪੂਰਨ       ne00_src,
                                     ਅਚਲ ਪੂਰਨ       ne01_src,
                                     ਅਚਲ ਪੂਰਨ       ne10_dst,
                                     ਅਚਲ ਪੂਰਨ       ne11_dst,
                                     ਅਚਲ ਪੂਰਨ       ne12_dst,
                                     ਅਚਲ ਪੂਰਨ       ne13_dst,
                                     ਅਚਲ ਦਸ਼ਮਲਵ     sf0,
                                     ਅਚਲ ਦਸ਼ਮਲਵ     sf1,
                                     ਅਚਲ ਦਸ਼ਮਲਵ     sf2,
                                     ਅਚਲ ਦਸ਼ਮਲਵ     sf3,
                                     ਅਚਲ ਦਸ਼ਮਲਵ     pixel_ਵਿੱਥ,
                                     dpct::ਕਤਾਰ_ptr stream) {
    ਅਚਲ int64_t dst_ਮਾਪ   = ne10_dst * ne11_dst * ne12_dst * ne13_dst;
    ਅਚਲ int64_t ਅੰਕ_ਖੰਡਾਂ = (dst_ਮਾਪ + SYCL_UPSCALE_ਖੰਡ_ਮਾਪ - 1) / SYCL_UPSCALE_ਖੰਡ_ਮਾਪ;

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_UPSCALE_ਖੰਡ_ਮਾਪ),
                    sycl::range<3>(1, 1, SYCL_UPSCALE_ਖੰਡ_ਮਾਪ)),
                [=](sycl::nd_ਵਸਤੂ<3> /*item_ct1*/) {
                    upscale_f32_bicubic(
                        x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst,
                        ne12_dst, ne13_dst, sf0, sf1, sf2, sf3, pixel_ਵਿੱਥ);
                });
        });
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_sycl_op_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ਦਸ਼ਮਲਵ * src0_d = (ਅਚਲ ਦਸ਼ਮਲਵ *)src0->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * dst_d = (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ;
    dpct::ਕਤਾਰ_ptr     stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    ਅਚਲ ਪੂਰਨ ਬਹੁਲਕ_flags = dst->op_params[0];
    ਅਚਲ ggml_scale_ਬਹੁਲਕ mode = (ggml_scale_ਬਹੁਲਕ)(ਬਹੁਲਕ_flags & 0xFF);

    ਦਸ਼ਮਲਵ sf0 = (ਦਸ਼ਮਲਵ)dst->ne[0]/src0->ne[0];
    ਦਸ਼ਮਲਵ sf1 = (ਦਸ਼ਮਲਵ)dst->ne[1]/src0->ne[1];
    ਦਸ਼ਮਲਵ sf2 = (ਦਸ਼ਮਲਵ)dst->ne[2]/src0->ne[2];
    ਅਚਲ ਦਸ਼ਮਲਵ sf3 = (ਦਸ਼ਮਲਵ)dst->ne[3]/src0->ne[3];

    ਦਸ਼ਮਲਵ pixel_ਵਿੱਥ = 0.5f;
    ਜੇ (ਬਹੁਲਕ_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        sf0 = dst->ne[0] > 1 && src0->ne[0] > 1
            ? (ਦਸ਼ਮਲਵ)(dst->ne[0] - 1) / (src0->ne[0] - 1)
            : sf0;
        sf1 = dst->ne[1] > 1 && src0->ne[1] > 1
            ? (ਦਸ਼ਮਲਵ)(dst->ne[1] - 1) / (src0->ne[1] - 1)
            : sf1;
        pixel_ਵਿੱਥ = 0.0f;
    }

    ਜੇ (mode == GGML_SCALE_ਬਹੁਲਕ_ਸਭ_ਤੋਂ_ਨੇੜੇ) {
        upscale_f32_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], sf0, sf1, sf2, sf3, stream);
    } ਨਹੀਂਤਾਂ ਜੇ (mode == GGML_SCALE_ਬਹੁਲਕ_BILINEAR) {
        ਅਚਲ ਤਾਰਕਿਕ antialias = (ਬਹੁਲਕ_flags & GGML_SCALE_FLAG_ANTIALIAS);
        upscale_f32_bilinear_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src0->ne[0], src0->ne[1], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            sf0, sf1, sf2, sf3, pixel_ਵਿੱਥ, antialias, stream);
    } ਨਹੀਂਤਾਂ ਜੇ (mode == GGML_SCALE_ਬਹੁਲਕ_BICUBIC) {
        upscale_f32_bicubic_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src0->ne[0], src0->ne[1], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            sf0, sf1, sf2, sf3, pixel_ਵਿੱਥ, stream);
    }
}

ਖਾਲੀ ggml_sycl_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_upscale(ctx, dst);
}
