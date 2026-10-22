#ଅନ୍ତର୍ଭାଵ "upscale.hpp"

ସ୍ଥିର ଶୂନ୍ୟ upscale_f32(ଅଚଳ ଦଶମିକ * x, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne13,
        ଅଚଳ ଦଶମିକ sf0, ଅଚଳ ଦଶମିକ sf1, ଅଚଳ ଦଶମିକ sf2, ଅଚଳ ଦଶମିକ sf3) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା  index    = item_ct1.get_local_id(2) + item_ct1.get_group(2) * item_ct1.get_local_range(2);
    ଯଦି (index >= ne10 * ne11 * ne12 * ne13) {
        ଫେରନ୍ତୁ;
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା i10 = index % ne10;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା i11 = (index / ne10) % ne11;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା i12 = (index / (ne10 * ne11)) % ne12;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା i13 = (index / (ne10 * ne11 * ne12)) % ne13;

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା i00 = i10 / sf0;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା i01 = i11 / sf1;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା i02 = i12 / sf2;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା i03 = i13 / sf3;

    dst[index] = *((ଅଚଳ ଦଶମିକ*)((ଅଚଳ ଅକ୍ଷର*)x + i03 * nb03 + i02 * nb02 +
                                  i01 * nb01 + i00 * nb00));
}

ସ୍ଥିର ଶୂନ୍ୟ upscale_f32_bilinear(ଅଚଳ ଦଶମିକ * x, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00_src, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01_src,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10_dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11_dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12_dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne13_dst,
        ଅଚଳ ଦଶମିକ sf0, ଅଚଳ ଦଶମିକ sf1, ଅଚଳ ଦଶମିକ sf2, ଅଚଳ ଦଶମିକ sf3,
        ଅଚଳ ଦଶମିକ pixel_offset) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ଅଚଳ int64_t index = item_ct1.get_local_id(2) +
        item_ct1.get_group(2) * item_ct1.get_local_range(2);
    ଅଚଳ int64_t dst_total_elements = ne10_dst * ne11_dst * ne12_dst * ne13_dst;

  ଯଦି (index >= dst_total_elements) {
    ଫେରନ୍ତୁ;
  }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i10_dst = index % ne10_dst;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i11_dst = (index / ne10_dst) % ne11_dst;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i12_dst = (index / (ne10_dst * ne11_dst)) % ne12_dst;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i13_dst = index / (ne10_dst * ne11_dst * ne12_dst);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i02_src = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)(i12_dst / sf2);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i03_src = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)(i13_dst / sf3);

    ଅଚଳ ଦଶମିକ y_src_f = ((ଦଶମିକ)i11_dst + pixel_offset) / sf1 - pixel_offset;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା         y0_src    = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) sycl::floor((ଦଶମିକ) y_src_f);
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା y1_src    = y0_src + 1;

    y0_src = sycl::max(0, sycl::min(y0_src, ne01_src - 1));
    y1_src = sycl::max(0, sycl::min(y1_src, ne01_src - 1));

    ଦଶମିକ dy = y_src_f - (ଦଶମିକ)y0_src;
    dy       = sycl::max(0.0f, sycl::min(dy, 1.0f));

    ଦଶମିକ x_src_f = ((ଦଶମିକ)i10_dst + pixel_offset) / sf0 - pixel_offset;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା   x0_src    = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) sycl::floor(x_src_f);
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା x1_src    = x0_src + 1;

    x0_src = sycl::max(0, sycl::min(x0_src, ne00_src - 1));
    x1_src = sycl::max(0, sycl::min(x1_src, ne00_src - 1));

    ଦଶମିକ dx = x_src_f - (ଦଶମିକ)x0_src;
    dx       = sycl::max(0.0f, sycl::min(dx, 1.0f));

    ଅଚଳ ଦଶମିକ* p_a =
        (ଅଚଳ ଦଶମିକ*)((ଅଚଳ ଅକ୍ଷର*)x + (int64_t)x0_src * nb00 +
                       (int64_t)y0_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    ଅଚଳ ଦଶମିକ* p_b =
        (ଅଚଳ ଦଶମିକ*)((ଅଚଳ ଅକ୍ଷର*)x + (int64_t)x1_src * nb00 +
                       (int64_t)y0_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    ଅଚଳ ଦଶମିକ* p_c =
        (ଅଚଳ ଦଶମିକ*)((ଅଚଳ ଅକ୍ଷର*)x + (int64_t)x0_src * nb00 +
                       (int64_t)y1_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    ଅଚଳ ଦଶମିକ* p_d =
        (ଅଚଳ ଦଶମିକ*)((ଅଚଳ ଅକ୍ଷର*)x + (int64_t)x1_src * nb00 +
                       (int64_t)y1_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);

    ଅଚଳ ଦଶମିକ val_a = *p_a;
    ଅଚଳ ଦଶମିକ val_b = *p_b;
    ଅଚଳ ଦଶମିକ val_c = *p_c;
    ଅଚଳ ଦଶମିକ val_d = *p_d;

    ଦଶମିକ result = val_a * (1.0f - dx) * (1.0f - dy) +
                   val_b * dx * (1.0f - dy) +
                   val_c * (1.0f - dx) * dy +
                   val_d * dx * dy;

    dst[index] = result;
}

// Similar to F.interpolate(..., mode="bilinear", align_corners=False, antialias=True)
// https://github.com/pytorch/pytorch/blob/8871ff29b743948d1225389d5b7068f37b22750b/aten/src/ATen/native/cpu/UpSampleKernel.cpp
ସ୍ଥିର ଶୂନ୍ୟ upscale_f32_bilinear_antialias(ଅଚଳ ଦଶମିକ * src0,
                                           ଦଶମିକ *       dst,
                                           ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା     nb00,
                                           ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା     nb01,
                                           ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା     nb02,
                                           ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା     nb03,
                                           ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା     ne00_src,
                                           ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା     ne01_src,
                                           ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା     ne10_dst,
                                           ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା     ne11_dst,
                                           ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା     ne12_dst,
                                           ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା     ne13_dst,
                                           ଅଚଳ ଦଶମିକ   sf0,
                                           ଅଚଳ ଦଶମିକ   sf1,
                                           ଅଚଳ ଦଶମିକ   sf2,
                                           ଅଚଳ ଦଶମିକ   sf3,
                                           ଅଚଳ ଦଶମିକ   pixel_offset) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ଅଚଳ int64_t index = item_ct1.get_local_id(2) +
        item_ct1.get_group(2) * item_ct1.get_local_range(2);
    ଅଚଳ int64_t dst_total_elements = ne10_dst * ne11_dst * ne12_dst * ne13_dst;

    ଯଦି (index >= dst_total_elements) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i10_dst = index % ne10_dst;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i11_dst = (index / ne10_dst) % ne11_dst;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i12_dst = (index / (ne10_dst * ne11_dst)) % ne12_dst;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i13_dst = index / (ne10_dst * ne11_dst * ne12_dst);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i02_src = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)(i12_dst / sf2);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i03_src = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)(i13_dst / sf3);

    ଅଚଳ ଦଶମିକ y = ((ଦଶମିକ)i11_dst + pixel_offset) / sf1;
    ଅଚଳ ଦଶମିକ x = ((ଦଶମିକ)i10_dst + pixel_offset) / sf0;

    // support and invscale, minimum 1 pixel for bilinear
    ଅଚଳ ଦଶମିକ support1  = sycl::max(1.0f / sf1, 1.0f);
    ଅଚଳ ଦଶମିକ invscale1 = 1.0f / support1;
    ଅଚଳ ଦଶମିକ support0  = sycl::max(1.0f / sf0, 1.0f);
    ଅଚଳ ଦଶମିକ invscale0 = 1.0f / support0;

    // the range of source pixels that contribute
    ଅଚଳ int64_t x_min = sycl::max(int64_t(0), int64_t(x - support0 + pixel_offset));
    ଅଚଳ int64_t x_max = sycl::min(int64_t(ne00_src), int64_t(x + support0 + pixel_offset));
    ଅଚଳ int64_t y_min = sycl::max(int64_t(0), int64_t(y - support1 + pixel_offset));
    ଅଚଳ int64_t y_max = sycl::min(int64_t(ne01_src), int64_t(y + support1 + pixel_offset));

    // bilinear filter with antialiasing
    ଦଶମିକ val = 0.0f;
    ଦଶମିକ total_weight = 0.0f;

    auto triangle_filter = [](ଦଶମିକ x) -> ଦଶମିକ {
        ଫେରନ୍ତୁ sycl::max(1.0f - sycl::fabs(x), 0.0f);
    };

    ପାଇଁ (int64_t sy = y_min; sy < y_max; sy++) {
        ଅଚଳ ଦଶମିକ weight_y = triangle_filter((sy - y + pixel_offset) * invscale1);

        ପାଇଁ (int64_t sx = x_min; sx < x_max; sx++) {
            ଅଚଳ ଦଶମିକ weight_x = triangle_filter((sx - x + pixel_offset) * invscale0);
            ଅଚଳ ଦଶମିକ weight = weight_x * weight_y;

            ଯଦି (weight <= 0.0f) {
                ଜାରି;
            }

            ଅଚଳ ଦଶମିକ pixel =
                *(ଅଚଳ ଦଶମିକ*)((ଅଚଳ ଅକ୍ଷର*)src0 + sx * nb00 + sy * nb01 +
                                i02_src * nb02 + i03_src * nb03);
            val += pixel * weight;
            total_weight += weight;
        }
    }

    ଯଦି (total_weight > 0.0f) {
        val /= total_weight;
    }

    dst[index] = val;
}

ନାମାକାଶ bicubic_interpolation {
ସ୍ଥିର ଦଶମିକ weight1(ଦଶମିକ x, ଅଚଳ ଦଶମିକ &a) { ଫେରନ୍ତୁ ((a + 2) * x - (a + 3)) * x * x + 1; }
ସ୍ଥିର ଦଶମିକ weight2(ଦଶମିକ x, ଅଚଳ ଦଶମିକ &a) { ଫେରନ୍ତୁ ((a * x - 5 * a) * x + 8 * a) * x - 4 * a; }

ସ୍ଥିର ଦଶମିକ bicubic(ଦଶମିକ p0, ଦଶମିକ p1, ଦଶମିକ p2, ଦଶମିକ p3, ଦଶମିକ x, ଦଶମିକ a) {
    ଅଚଳ ଦଶମିକ w0 = weight2(x + 1, a);
    ଅଚଳ ଦଶମିକ w1 = weight1(x + 0, a);
    ଅଚଳ ଦଶମିକ w2 = weight1(1 - x, a);
    ଅଚଳ ଦଶମିକ w3 = weight2(2 - x, a);
    ଫେରନ୍ତୁ p0 * w0 + p1 * w1 + p2 * w2 + p3 * w3;
}

}

ସ୍ଥିର ଶୂନ୍ୟ upscale_f32_bicubic(ଅଚଳ ଦଶମିକ * x, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00_src, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01_src,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10_dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11_dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12_dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne13_dst,
        ଅଚଳ ଦଶମିକ sf0, ଅଚଳ ଦଶମିକ sf1, ଅଚଳ ଦଶମିକ sf2, ଅଚଳ ଦଶମିକ sf3,
        ଅଚଳ ଦଶମିକ pixel_offset) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ଅଚଳ ଦଶମିକ a = -0.75f;
    using bicubic_interpolation::bicubic;

    ଅଚଳ int64_t index = item_ct1.get_local_id(2) +
        item_ct1.get_group(2) * item_ct1.get_local_range(2);
    ଅଚଳ int64_t dst_total_elements =
        ne10_dst * ne11_dst * ne12_dst * ne13_dst;

    ଯଦି (index >= dst_total_elements) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i10_dst = index % ne10_dst;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i11_dst = (index / ne10_dst) % ne11_dst;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i12_dst = (index / (ne10_dst * ne11_dst)) % ne12_dst;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i13_dst = index / (ne10_dst * ne11_dst * ne12_dst);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i02_src = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)(i12_dst / sf2);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i03_src = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)(i13_dst / sf3);

    ଅଚଳ ଦଶମିକ y_src_f = ((ଦଶମିକ)i11_dst + pixel_offset) / sf1 - pixel_offset;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା   y0_src  = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) sycl::floor((ଦଶମିକ) y_src_f);
    ଅଚଳ ଦଶମିକ dy      = y_src_f - (ଦଶମିକ)y0_src;

    ଅଚଳ ଦଶମିକ x_src_f = ((ଦଶମିକ)i10_dst + pixel_offset) / sf0 - pixel_offset;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା   x0_src  = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) sycl::floor((ଦଶମିକ) x_src_f);
    ଅଚଳ ଦଶମିକ dx      = x_src_f - (ଦଶମିକ)x0_src;

    ଅଚଳ ଅକ୍ଷର * x_base = (ଅଚଳ ଅକ୍ଷର *)x + (int64_t)i02_src * nb02 + (int64_t)i03_src * nb03;

    auto load = [=](ପୂର୍ଣ୍ଣସଂଖ୍ୟା x_off, ପୂର୍ଣ୍ଣସଂଖ୍ୟା y_off) -> ଦଶମିକ {
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା i00_src = sycl::max(0, sycl::min(x0_src + x_off, ne00_src - 1));
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା i01_src = sycl::max(0, sycl::min(y0_src + y_off, ne01_src - 1));
        ଫେରନ୍ତୁ *(ଅଚଳ ଦଶମିକ *)(x_base + (int64_t)i00_src * nb00 + (int64_t)i01_src * nb01);
    };

    ଅଚଳ ଦଶମିକ result = bicubic(
        bicubic(load(-1, -1), load(0, -1), load(1, -1), load(2, -1), dx, a),
        bicubic(load(-1, 0), load(0, 0), load(1, 0), load(2, 0), dx, a),
        bicubic(load(-1, 1), load(0, 1), load(1, 1), load(2, 1), dx, a),
        bicubic(load(-1, 2), load(0, 2), load(1, 2), load(2, 2), dx, a),
        dy,
        a);

    dst[index] = result;
}

ସ୍ଥିର ଶୂନ୍ୟ upscale_f32_sycl(ଅଚଳ ଦଶମିକ *   x,
                             ଦଶମିକ *         dst,
                             ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       nb00,
                             ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       nb01,
                             ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       nb02,
                             ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       nb03,
                             ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       ne10,
                             ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       ne11,
                             ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       ne12,
                             ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       ne13,
                             ଅଚଳ ଦଶମିକ     sf0,
                             ଅଚଳ ଦଶମିକ     sf1,
                             ଅଚଳ ଦଶମିକ     sf2,
                             ଅଚଳ ଦଶମିକ     sf3,
                             dpct::queue_ptr stream) {
    ଅଚଳ int64_t dst_size   = ne10 * ne11 * ne12 * ne13;
    ଅଚଳ int64_t ଅଙ୍କ_blocks = (dst_size + SYCL_UPSCALE_BLOCK_SIZE - 1) / SYCL_UPSCALE_BLOCK_SIZE;

    stream->parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE),
             sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE)),
        [=](sycl::nd_item<3> /*item_ct1*/) {
            upscale_f32(x, dst, nb00, nb01, nb02, nb03, ne10, ne11, ne12, ne13, sf0, sf1, sf2, sf3);
        });
}

ସ୍ଥିର ଶୂନ୍ୟ upscale_f32_bilinear_sycl(ଅଚଳ ଦଶମିକ *   x,
                                      ଦଶମିକ *         dst,
                                      ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       nb00,
                                      ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       nb01,
                                      ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       nb02,
                                      ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       nb03,
                                      ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       ne00_src,
                                      ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       ne01_src,
                                      ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       ne10_dst,
                                      ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       ne11_dst,
                                      ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       ne12_dst,
                                      ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       ne13_dst,
                                      ଅଚଳ ଦଶମିକ     sf0,
                                      ଅଚଳ ଦଶମିକ     sf1,
                                      ଅଚଳ ଦଶମିକ     sf2,
                                      ଅଚଳ ଦଶମିକ     sf3,
                                      ଅଚଳ ଦଶମିକ     pixel_offset,
                                      ତାର୍କିକ            antialias,
                                      dpct::queue_ptr stream) {
    ଅଚଳ int64_t dst_size   = ne10_dst * ne11_dst * ne12_dst * ne13_dst;
    ଅଚଳ int64_t ଅଙ୍କ_blocks = (dst_size + SYCL_UPSCALE_BLOCK_SIZE - 1) / SYCL_UPSCALE_BLOCK_SIZE;

    ଯଦି (antialias) {
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE),
                sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE)),
            [=](sycl::nd_item<3> /*item_ct1*/) {
                upscale_f32_bilinear_antialias(
                    x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst,
                    ne12_dst, ne13_dst, sf0, sf1, sf2, sf3, pixel_offset);
            });
    } ଅନ୍ୟଥା {
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE),
                sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE)),
            [=](sycl::nd_item<3> /*item_ct1*/) {
                upscale_f32_bilinear(
                    x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst, ne12_dst,
                    ne13_dst, sf0, sf1, sf2, sf3, pixel_offset);
            });
    }
}

ସ୍ଥିର ଶୂନ୍ୟ upscale_f32_bicubic_sycl(ଅଚଳ ଦଶମିକ *   x,
                                     ଦଶମିକ *         dst,
                                     ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       nb00,
                                     ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       nb01,
                                     ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       nb02,
                                     ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       nb03,
                                     ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       ne00_src,
                                     ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       ne01_src,
                                     ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       ne10_dst,
                                     ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       ne11_dst,
                                     ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       ne12_dst,
                                     ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       ne13_dst,
                                     ଅଚଳ ଦଶମିକ     sf0,
                                     ଅଚଳ ଦଶମିକ     sf1,
                                     ଅଚଳ ଦଶମିକ     sf2,
                                     ଅଚଳ ଦଶମିକ     sf3,
                                     ଅଚଳ ଦଶମିକ     pixel_offset,
                                     dpct::queue_ptr stream) {
    ଅଚଳ int64_t dst_size   = ne10_dst * ne11_dst * ne12_dst * ne13_dst;
    ଅଚଳ int64_t ଅଙ୍କ_blocks = (dst_size + SYCL_UPSCALE_BLOCK_SIZE - 1) / SYCL_UPSCALE_BLOCK_SIZE;

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE),
                    sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE)),
                [=](sycl::nd_item<3> /*item_ct1*/) {
                    upscale_f32_bicubic(
                        x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst,
                        ne12_dst, ne13_dst, sf0, sf1, sf2, sf3, pixel_offset);
                });
        });
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_sycl_op_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ଦଶମିକ * src0_d = (ଅଚଳ ଦଶମିକ *)src0->data;
    ଦଶମିକ * dst_d = (ଦଶମିକ *)dst->data;
    dpct::queue_ptr     stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା mode_flags = dst->op_params[0];
    ଅଚଳ ggml_scale_mode mode = (ggml_scale_mode)(mode_flags & 0xFF);

    ଦଶମିକ sf0 = (ଦଶମିକ)dst->ne[0]/src0->ne[0];
    ଦଶମିକ sf1 = (ଦଶମିକ)dst->ne[1]/src0->ne[1];
    ଦଶମିକ sf2 = (ଦଶମିକ)dst->ne[2]/src0->ne[2];
    ଅଚଳ ଦଶମିକ sf3 = (ଦଶମିକ)dst->ne[3]/src0->ne[3];

    ଦଶମିକ pixel_offset = 0.5f;
    ଯଦି (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        sf0 = dst->ne[0] > 1 && src0->ne[0] > 1
            ? (ଦଶମିକ)(dst->ne[0] - 1) / (src0->ne[0] - 1)
            : sf0;
        sf1 = dst->ne[1] > 1 && src0->ne[1] > 1
            ? (ଦଶମିକ)(dst->ne[1] - 1) / (src0->ne[1] - 1)
            : sf1;
        pixel_offset = 0.0f;
    }

    ଯଦି (mode == GGML_SCALE_MODE_NEAREST) {
        upscale_f32_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], sf0, sf1, sf2, sf3, stream);
    } ଅନ୍ୟଥା ଯଦି (mode == GGML_SCALE_MODE_BILINEAR) {
        ଅଚଳ ତାର୍କିକ antialias = (mode_flags & GGML_SCALE_FLAG_ANTIALIAS);
        upscale_f32_bilinear_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src0->ne[0], src0->ne[1], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            sf0, sf1, sf2, sf3, pixel_offset, antialias, stream);
    } ଅନ୍ୟଥା ଯଦି (mode == GGML_SCALE_MODE_BICUBIC) {
        upscale_f32_bicubic_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src0->ne[0], src0->ne[1], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            sf0, sf1, sf2, sf3, pixel_offset, stream);
    }
}

ଶୂନ୍ୟ ggml_sycl_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_upscale(ctx, dst);
}
