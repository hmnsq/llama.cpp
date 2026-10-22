#ಅನ್ತರ್ಭಾವ "upscale.hpp"

ಸ್ಥಿರ ಶೂನ್ಯ upscale_f32(ಧ್ರುವ ಭಿನ್ನ * x, ಭಿನ್ನ * dst,
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne13,
        ಧ್ರುವ ಭಿನ್ನ sf0, ಧ್ರುವ ಭಿನ್ನ sf1, ಧ್ರುವ ಭಿನ್ನ sf2, ಧ್ರುವ ಭಿನ್ನ sf3) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ಪೂರ್ಣಾಙ್ಕ  index    = item_ct1.get_local_id(2) + item_ct1.get_group(2) * item_ct1.get_local_range(2);
    ಯದಿ (index >= ne10 * ne11 * ne12 * ne13) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಪೂರ್ಣಾಙ್ಕ i10 = index % ne10;
    ಪೂರ್ಣಾಙ್ಕ i11 = (index / ne10) % ne11;
    ಪೂರ್ಣಾಙ್ಕ i12 = (index / (ne10 * ne11)) % ne12;
    ಪೂರ್ಣಾಙ್ಕ i13 = (index / (ne10 * ne11 * ne12)) % ne13;

    ಪೂರ್ಣಾಙ್ಕ i00 = i10 / sf0;
    ಪೂರ್ಣಾಙ್ಕ i01 = i11 / sf1;
    ಪೂರ್ಣಾಙ್ಕ i02 = i12 / sf2;
    ಪೂರ್ಣಾಙ್ಕ i03 = i13 / sf3;

    dst[index] = *((ಧ್ರುವ ಭಿನ್ನ*)((ಧ್ರುವ ವರ್ಣ*)x + i03 * nb03 + i02 * nb02 +
                                  i01 * nb01 + i00 * nb00));
}

ಸ್ಥಿರ ಶೂನ್ಯ upscale_f32_bilinear(ಧ್ರುವ ಭಿನ್ನ * x, ಭಿನ್ನ * dst,
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00_src, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01_src,
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10_dst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11_dst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12_dst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne13_dst,
        ಧ್ರುವ ಭಿನ್ನ sf0, ಧ್ರುವ ಭಿನ್ನ sf1, ಧ್ರುವ ಭಿನ್ನ sf2, ಧ್ರುವ ಭಿನ್ನ sf3,
        ಧ್ರುವ ಭಿನ್ನ pixel_offset) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ಧ್ರುವ int64_t index = item_ct1.get_local_id(2) +
        item_ct1.get_group(2) * item_ct1.get_local_range(2);
    ಧ್ರುವ int64_t dst_total_elements = ne10_dst * ne11_dst * ne12_dst * ne13_dst;

  ಯದಿ (index >= dst_total_elements) {
    ಪ್ರತ್ಯಾಗಮ;
  }

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i10_dst = index % ne10_dst;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i11_dst = (index / ne10_dst) % ne11_dst;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i12_dst = (index / (ne10_dst * ne11_dst)) % ne12_dst;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i13_dst = index / (ne10_dst * ne11_dst * ne12_dst);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i02_src = (ಪೂರ್ಣಾಙ್ಕ)(i12_dst / sf2);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i03_src = (ಪೂರ್ಣಾಙ್ಕ)(i13_dst / sf3);

    ಧ್ರುವ ಭಿನ್ನ y_src_f = ((ಭಿನ್ನ)i11_dst + pixel_offset) / sf1 - pixel_offset;
    ಪೂರ್ಣಾಙ್ಕ         y0_src    = (ಪೂರ್ಣಾಙ್ಕ) sycl::floor((ಭಿನ್ನ) y_src_f);
    ಪೂರ್ಣಾಙ್ಕ y1_src    = y0_src + 1;

    y0_src = sycl::max(0, sycl::min(y0_src, ne01_src - 1));
    y1_src = sycl::max(0, sycl::min(y1_src, ne01_src - 1));

    ಭಿನ್ನ dy = y_src_f - (ಭಿನ್ನ)y0_src;
    dy       = sycl::max(0.0f, sycl::min(dy, 1.0f));

    ಭಿನ್ನ x_src_f = ((ಭಿನ್ನ)i10_dst + pixel_offset) / sf0 - pixel_offset;
    ಪೂರ್ಣಾಙ್ಕ   x0_src    = (ಪೂರ್ಣಾಙ್ಕ) sycl::floor(x_src_f);
    ಪೂರ್ಣಾಙ್ಕ x1_src    = x0_src + 1;

    x0_src = sycl::max(0, sycl::min(x0_src, ne00_src - 1));
    x1_src = sycl::max(0, sycl::min(x1_src, ne00_src - 1));

    ಭಿನ್ನ dx = x_src_f - (ಭಿನ್ನ)x0_src;
    dx       = sycl::max(0.0f, sycl::min(dx, 1.0f));

    ಧ್ರುವ ಭಿನ್ನ* p_a =
        (ಧ್ರುವ ಭಿನ್ನ*)((ಧ್ರುವ ವರ್ಣ*)x + (int64_t)x0_src * nb00 +
                       (int64_t)y0_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    ಧ್ರುವ ಭಿನ್ನ* p_b =
        (ಧ್ರುವ ಭಿನ್ನ*)((ಧ್ರುವ ವರ್ಣ*)x + (int64_t)x1_src * nb00 +
                       (int64_t)y0_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    ಧ್ರುವ ಭಿನ್ನ* p_c =
        (ಧ್ರುವ ಭಿನ್ನ*)((ಧ್ರುವ ವರ್ಣ*)x + (int64_t)x0_src * nb00 +
                       (int64_t)y1_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    ಧ್ರುವ ಭಿನ್ನ* p_d =
        (ಧ್ರುವ ಭಿನ್ನ*)((ಧ್ರುವ ವರ್ಣ*)x + (int64_t)x1_src * nb00 +
                       (int64_t)y1_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);

    ಧ್ರುವ ಭಿನ್ನ val_a = *p_a;
    ಧ್ರುವ ಭಿನ್ನ val_b = *p_b;
    ಧ್ರುವ ಭಿನ್ನ val_c = *p_c;
    ಧ್ರುವ ಭಿನ್ನ val_d = *p_d;

    ಭಿನ್ನ result = val_a * (1.0f - dx) * (1.0f - dy) +
                   val_b * dx * (1.0f - dy) +
                   val_c * (1.0f - dx) * dy +
                   val_d * dx * dy;

    dst[index] = result;
}

// Similar to F.interpolate(..., mode="bilinear", align_corners=False, antialias=True)
// https://github.com/pytorch/pytorch/blob/8871ff29b743948d1225389d5b7068f37b22750b/aten/src/ATen/native/cpu/UpSampleKernel.cpp
ಸ್ಥಿರ ಶೂನ್ಯ upscale_f32_bilinear_antialias(ಧ್ರುವ ಭಿನ್ನ * src0,
                                           ಭಿನ್ನ *       dst,
                                           ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ     nb00,
                                           ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ     nb01,
                                           ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ     nb02,
                                           ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ     nb03,
                                           ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ     ne00_src,
                                           ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ     ne01_src,
                                           ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ     ne10_dst,
                                           ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ     ne11_dst,
                                           ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ     ne12_dst,
                                           ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ     ne13_dst,
                                           ಧ್ರುವ ಭಿನ್ನ   sf0,
                                           ಧ್ರುವ ಭಿನ್ನ   sf1,
                                           ಧ್ರುವ ಭಿನ್ನ   sf2,
                                           ಧ್ರುವ ಭಿನ್ನ   sf3,
                                           ಧ್ರುವ ಭಿನ್ನ   pixel_offset) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ಧ್ರುವ int64_t index = item_ct1.get_local_id(2) +
        item_ct1.get_group(2) * item_ct1.get_local_range(2);
    ಧ್ರುವ int64_t dst_total_elements = ne10_dst * ne11_dst * ne12_dst * ne13_dst;

    ಯದಿ (index >= dst_total_elements) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i10_dst = index % ne10_dst;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i11_dst = (index / ne10_dst) % ne11_dst;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i12_dst = (index / (ne10_dst * ne11_dst)) % ne12_dst;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i13_dst = index / (ne10_dst * ne11_dst * ne12_dst);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i02_src = (ಪೂರ್ಣಾಙ್ಕ)(i12_dst / sf2);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i03_src = (ಪೂರ್ಣಾಙ್ಕ)(i13_dst / sf3);

    ಧ್ರುವ ಭಿನ್ನ y = ((ಭಿನ್ನ)i11_dst + pixel_offset) / sf1;
    ಧ್ರುವ ಭಿನ್ನ x = ((ಭಿನ್ನ)i10_dst + pixel_offset) / sf0;

    // support and invscale, minimum 1 pixel for bilinear
    ಧ್ರುವ ಭಿನ್ನ support1  = sycl::max(1.0f / sf1, 1.0f);
    ಧ್ರುವ ಭಿನ್ನ invscale1 = 1.0f / support1;
    ಧ್ರುವ ಭಿನ್ನ support0  = sycl::max(1.0f / sf0, 1.0f);
    ಧ್ರುವ ಭಿನ್ನ invscale0 = 1.0f / support0;

    // the range of source pixels that contribute
    ಧ್ರುವ int64_t x_min = sycl::max(int64_t(0), int64_t(x - support0 + pixel_offset));
    ಧ್ರುವ int64_t x_max = sycl::min(int64_t(ne00_src), int64_t(x + support0 + pixel_offset));
    ಧ್ರುವ int64_t y_min = sycl::max(int64_t(0), int64_t(y - support1 + pixel_offset));
    ಧ್ರುವ int64_t y_max = sycl::min(int64_t(ne01_src), int64_t(y + support1 + pixel_offset));

    // bilinear filter with antialiasing
    ಭಿನ್ನ val = 0.0f;
    ಭಿನ್ನ total_weight = 0.0f;

    auto triangle_filter = [](ಭಿನ್ನ x) -> ಭಿನ್ನ {
        ಪ್ರತ್ಯಾಗಮ sycl::max(1.0f - sycl::fabs(x), 0.0f);
    };

    ಪ್ರತಿ (int64_t sy = y_min; sy < y_max; sy++) {
        ಧ್ರುವ ಭಿನ್ನ weight_y = triangle_filter((sy - y + pixel_offset) * invscale1);

        ಪ್ರತಿ (int64_t sx = x_min; sx < x_max; sx++) {
            ಧ್ರುವ ಭಿನ್ನ weight_x = triangle_filter((sx - x + pixel_offset) * invscale0);
            ಧ್ರುವ ಭಿನ್ನ weight = weight_x * weight_y;

            ಯದಿ (weight <= 0.0f) {
                ಅನುವರ್ತ;
            }

            ಧ್ರುವ ಭಿನ್ನ pixel =
                *(ಧ್ರುವ ಭಿನ್ನ*)((ಧ್ರುವ ವರ್ಣ*)src0 + sx * nb00 + sy * nb01 +
                                i02_src * nb02 + i03_src * nb03);
            val += pixel * weight;
            total_weight += weight;
        }
    }

    ಯದಿ (total_weight > 0.0f) {
        val /= total_weight;
    }

    dst[index] = val;
}

ನಾಮಾಕಾಶ bicubic_interpolation {
ಸ್ಥಿರ ಭಿನ್ನ weight1(ಭಿನ್ನ x, ಧ್ರುವ ಭಿನ್ನ &a) { ಪ್ರತ್ಯಾಗಮ ((a + 2) * x - (a + 3)) * x * x + 1; }
ಸ್ಥಿರ ಭಿನ್ನ weight2(ಭಿನ್ನ x, ಧ್ರುವ ಭಿನ್ನ &a) { ಪ್ರತ್ಯಾಗಮ ((a * x - 5 * a) * x + 8 * a) * x - 4 * a; }

ಸ್ಥಿರ ಭಿನ್ನ bicubic(ಭಿನ್ನ p0, ಭಿನ್ನ p1, ಭಿನ್ನ p2, ಭಿನ್ನ p3, ಭಿನ್ನ x, ಭಿನ್ನ a) {
    ಧ್ರುವ ಭಿನ್ನ w0 = weight2(x + 1, a);
    ಧ್ರುವ ಭಿನ್ನ w1 = weight1(x + 0, a);
    ಧ್ರುವ ಭಿನ್ನ w2 = weight1(1 - x, a);
    ಧ್ರುವ ಭಿನ್ನ w3 = weight2(2 - x, a);
    ಪ್ರತ್ಯಾಗಮ p0 * w0 + p1 * w1 + p2 * w2 + p3 * w3;
}

}

ಸ್ಥಿರ ಶೂನ್ಯ upscale_f32_bicubic(ಧ್ರುವ ಭಿನ್ನ * x, ಭಿನ್ನ * dst,
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00_src, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01_src,
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10_dst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11_dst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12_dst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne13_dst,
        ಧ್ರುವ ಭಿನ್ನ sf0, ಧ್ರುವ ಭಿನ್ನ sf1, ಧ್ರುವ ಭಿನ್ನ sf2, ಧ್ರುವ ಭಿನ್ನ sf3,
        ಧ್ರುವ ಭಿನ್ನ pixel_offset) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ಧ್ರುವ ಭಿನ್ನ a = -0.75f;
    using bicubic_interpolation::bicubic;

    ಧ್ರುವ int64_t index = item_ct1.get_local_id(2) +
        item_ct1.get_group(2) * item_ct1.get_local_range(2);
    ಧ್ರುವ int64_t dst_total_elements =
        ne10_dst * ne11_dst * ne12_dst * ne13_dst;

    ಯದಿ (index >= dst_total_elements) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i10_dst = index % ne10_dst;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i11_dst = (index / ne10_dst) % ne11_dst;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i12_dst = (index / (ne10_dst * ne11_dst)) % ne12_dst;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i13_dst = index / (ne10_dst * ne11_dst * ne12_dst);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i02_src = (ಪೂರ್ಣಾಙ್ಕ)(i12_dst / sf2);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i03_src = (ಪೂರ್ಣಾಙ್ಕ)(i13_dst / sf3);

    ಧ್ರುವ ಭಿನ್ನ y_src_f = ((ಭಿನ್ನ)i11_dst + pixel_offset) / sf1 - pixel_offset;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ   y0_src  = (ಪೂರ್ಣಾಙ್ಕ) sycl::floor((ಭಿನ್ನ) y_src_f);
    ಧ್ರುವ ಭಿನ್ನ dy      = y_src_f - (ಭಿನ್ನ)y0_src;

    ಧ್ರುವ ಭಿನ್ನ x_src_f = ((ಭಿನ್ನ)i10_dst + pixel_offset) / sf0 - pixel_offset;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ   x0_src  = (ಪೂರ್ಣಾಙ್ಕ) sycl::floor((ಭಿನ್ನ) x_src_f);
    ಧ್ರುವ ಭಿನ್ನ dx      = x_src_f - (ಭಿನ್ನ)x0_src;

    ಧ್ರುವ ವರ್ಣ * x_base = (ಧ್ರುವ ವರ್ಣ *)x + (int64_t)i02_src * nb02 + (int64_t)i03_src * nb03;

    auto load = [=](ಪೂರ್ಣಾಙ್ಕ x_off, ಪೂರ್ಣಾಙ್ಕ y_off) -> ಭಿನ್ನ {
        ಪೂರ್ಣಾಙ್ಕ i00_src = sycl::max(0, sycl::min(x0_src + x_off, ne00_src - 1));
        ಪೂರ್ಣಾಙ್ಕ i01_src = sycl::max(0, sycl::min(y0_src + y_off, ne01_src - 1));
        ಪ್ರತ್ಯಾಗಮ *(ಧ್ರುವ ಭಿನ್ನ *)(x_base + (int64_t)i00_src * nb00 + (int64_t)i01_src * nb01);
    };

    ಧ್ರುವ ಭಿನ್ನ result = bicubic(
        bicubic(load(-1, -1), load(0, -1), load(1, -1), load(2, -1), dx, a),
        bicubic(load(-1, 0), load(0, 0), load(1, 0), load(2, 0), dx, a),
        bicubic(load(-1, 1), load(0, 1), load(1, 1), load(2, 1), dx, a),
        bicubic(load(-1, 2), load(0, 2), load(1, 2), load(2, 2), dx, a),
        dy,
        a);

    dst[index] = result;
}

ಸ್ಥಿರ ಶೂನ್ಯ upscale_f32_sycl(ಧ್ರುವ ಭಿನ್ನ *   x,
                             ಭಿನ್ನ *         dst,
                             ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ       nb00,
                             ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ       nb01,
                             ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ       nb02,
                             ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ       nb03,
                             ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ       ne10,
                             ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ       ne11,
                             ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ       ne12,
                             ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ       ne13,
                             ಧ್ರುವ ಭಿನ್ನ     sf0,
                             ಧ್ರುವ ಭಿನ್ನ     sf1,
                             ಧ್ರುವ ಭಿನ್ನ     sf2,
                             ಧ್ರುವ ಭಿನ್ನ     sf3,
                             dpct::queue_ptr stream) {
    ಧ್ರುವ int64_t dst_size   = ne10 * ne11 * ne12 * ne13;
    ಧ್ರುವ int64_t ಅಂಕೆ_blocks = (dst_size + SYCL_UPSCALE_BLOCK_SIZE - 1) / SYCL_UPSCALE_BLOCK_SIZE;

    stream->parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE),
             sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE)),
        [=](sycl::nd_item<3> /*item_ct1*/) {
            upscale_f32(x, dst, nb00, nb01, nb02, nb03, ne10, ne11, ne12, ne13, sf0, sf1, sf2, sf3);
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ upscale_f32_bilinear_sycl(ಧ್ರುವ ಭಿನ್ನ *   x,
                                      ಭಿನ್ನ *         dst,
                                      ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ       nb00,
                                      ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ       nb01,
                                      ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ       nb02,
                                      ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ       nb03,
                                      ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ       ne00_src,
                                      ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ       ne01_src,
                                      ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ       ne10_dst,
                                      ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ       ne11_dst,
                                      ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ       ne12_dst,
                                      ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ       ne13_dst,
                                      ಧ್ರುವ ಭಿನ್ನ     sf0,
                                      ಧ್ರುವ ಭಿನ್ನ     sf1,
                                      ಧ್ರುವ ಭಿನ್ನ     sf2,
                                      ಧ್ರುವ ಭಿನ್ನ     sf3,
                                      ಧ್ರುವ ಭಿನ್ನ     pixel_offset,
                                      ತಾರ್ಕಿಕ            antialias,
                                      dpct::queue_ptr stream) {
    ಧ್ರುವ int64_t dst_size   = ne10_dst * ne11_dst * ne12_dst * ne13_dst;
    ಧ್ರುವ int64_t ಅಂಕೆ_blocks = (dst_size + SYCL_UPSCALE_BLOCK_SIZE - 1) / SYCL_UPSCALE_BLOCK_SIZE;

    ಯದಿ (antialias) {
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE),
                sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE)),
            [=](sycl::nd_item<3> /*item_ct1*/) {
                upscale_f32_bilinear_antialias(
                    x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst,
                    ne12_dst, ne13_dst, sf0, sf1, sf2, sf3, pixel_offset);
            });
    } ಅನ್ಯಥಾ {
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE),
                sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE)),
            [=](sycl::nd_item<3> /*item_ct1*/) {
                upscale_f32_bilinear(
                    x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst, ne12_dst,
                    ne13_dst, sf0, sf1, sf2, sf3, pixel_offset);
            });
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ upscale_f32_bicubic_sycl(ಧ್ರುವ ಭಿನ್ನ *   x,
                                     ಭಿನ್ನ *         dst,
                                     ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ       nb00,
                                     ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ       nb01,
                                     ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ       nb02,
                                     ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ       nb03,
                                     ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ       ne00_src,
                                     ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ       ne01_src,
                                     ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ       ne10_dst,
                                     ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ       ne11_dst,
                                     ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ       ne12_dst,
                                     ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ       ne13_dst,
                                     ಧ್ರುವ ಭಿನ್ನ     sf0,
                                     ಧ್ರುವ ಭಿನ್ನ     sf1,
                                     ಧ್ರುವ ಭಿನ್ನ     sf2,
                                     ಧ್ರುವ ಭಿನ್ನ     sf3,
                                     ಧ್ರುವ ಭಿನ್ನ     pixel_offset,
                                     dpct::queue_ptr stream) {
    ಧ್ರುವ int64_t dst_size   = ne10_dst * ne11_dst * ne12_dst * ne13_dst;
    ಧ್ರುವ int64_t ಅಂಕೆ_blocks = (dst_size + SYCL_UPSCALE_BLOCK_SIZE - 1) / SYCL_UPSCALE_BLOCK_SIZE;

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE),
                    sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE)),
                [=](sycl::nd_item<3> /*item_ct1*/) {
                    upscale_f32_bicubic(
                        x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst,
                        ne12_dst, ne13_dst, sf0, sf1, sf2, sf3, pixel_offset);
                });
        });
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_op_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ಧ್ರುವ ggml_tensor * src0 = dst->src[0];
    ಧ್ರುವ ಭಿನ್ನ * src0_d = (ಧ್ರುವ ಭಿನ್ನ *)src0->data;
    ಭಿನ್ನ * dst_d = (ಭಿನ್ನ *)dst->data;
    dpct::queue_ptr     stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ mode_flags = dst->op_params[0];
    ಧ್ರುವ ggml_scale_mode mode = (ggml_scale_mode)(mode_flags & 0xFF);

    ಭಿನ್ನ sf0 = (ಭಿನ್ನ)dst->ne[0]/src0->ne[0];
    ಭಿನ್ನ sf1 = (ಭಿನ್ನ)dst->ne[1]/src0->ne[1];
    ಭಿನ್ನ sf2 = (ಭಿನ್ನ)dst->ne[2]/src0->ne[2];
    ಧ್ರುವ ಭಿನ್ನ sf3 = (ಭಿನ್ನ)dst->ne[3]/src0->ne[3];

    ಭಿನ್ನ pixel_offset = 0.5f;
    ಯದಿ (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        sf0 = dst->ne[0] > 1 && src0->ne[0] > 1
            ? (ಭಿನ್ನ)(dst->ne[0] - 1) / (src0->ne[0] - 1)
            : sf0;
        sf1 = dst->ne[1] > 1 && src0->ne[1] > 1
            ? (ಭಿನ್ನ)(dst->ne[1] - 1) / (src0->ne[1] - 1)
            : sf1;
        pixel_offset = 0.0f;
    }

    ಯದಿ (mode == GGML_SCALE_MODE_NEAREST) {
        upscale_f32_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], sf0, sf1, sf2, sf3, stream);
    } ಅನ್ಯಥಾ ಯದಿ (mode == GGML_SCALE_MODE_BILINEAR) {
        ಧ್ರುವ ತಾರ್ಕಿಕ antialias = (mode_flags & GGML_SCALE_FLAG_ANTIALIAS);
        upscale_f32_bilinear_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src0->ne[0], src0->ne[1], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            sf0, sf1, sf2, sf3, pixel_offset, antialias, stream);
    } ಅನ್ಯಥಾ ಯದಿ (mode == GGML_SCALE_MODE_BICUBIC) {
        upscale_f32_bicubic_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src0->ne[0], src0->ne[1], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            sf0, sf1, sf2, sf3, pixel_offset, stream);
    }
}

ಶೂನ್ಯ ggml_sycl_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_upscale(ctx, dst);
}
