#অন্তর্ভাৱ "upscale.hpp"

স্থির শূন্য upscale_f32(ধ্রুৱ ভিন্ন * x, ভিন্ন * dst,
        ধ্রুৱ পূর্ণাঙ্ক nb00, ধ্রুৱ পূর্ণাঙ্ক nb01, ধ্রুৱ পূর্ণাঙ্ক nb02, ধ্রুৱ পূর্ণাঙ্ক nb03,
        ধ্রুৱ পূর্ণাঙ্ক ne10, ধ্রুৱ পূর্ণাঙ্ক ne11, ধ্রুৱ পূর্ণাঙ্ক ne12, ধ্রুৱ পূর্ণাঙ্ক ne13,
        ধ্রুৱ ভিন্ন sf0, ধ্রুৱ ভিন্ন sf1, ধ্রুৱ ভিন্ন sf2, ধ্রুৱ ভিন্ন sf3) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    পূর্ণাঙ্ক  index    = item_ct1.get_local_id(2) + item_ct1.get_group(2) * item_ct1.get_local_range(2);
    যদি (index >= ne10 * ne11 * ne12 * ne13) {
        ঘূৰাই;
    }

    পূর্ণাঙ্ক i10 = index % ne10;
    পূর্ণাঙ্ক i11 = (index / ne10) % ne11;
    পূর্ণাঙ্ক i12 = (index / (ne10 * ne11)) % ne12;
    পূর্ণাঙ্ক i13 = (index / (ne10 * ne11 * ne12)) % ne13;

    পূর্ণাঙ্ক i00 = i10 / sf0;
    পূর্ণাঙ্ক i01 = i11 / sf1;
    পূর্ণাঙ্ক i02 = i12 / sf2;
    পূর্ণাঙ্ক i03 = i13 / sf3;

    dst[index] = *((ধ্রুৱ ভিন্ন*)((ধ্রুৱ ৱর্ণ*)x + i03 * nb03 + i02 * nb02 +
                                  i01 * nb01 + i00 * nb00));
}

স্থির শূন্য upscale_f32_bilinear(ধ্রুৱ ভিন্ন * x, ভিন্ন * dst,
        ধ্রুৱ পূর্ণাঙ্ক nb00, ধ্রুৱ পূর্ণাঙ্ক nb01, ধ্রুৱ পূর্ণাঙ্ক nb02, ধ্রুৱ পূর্ণাঙ্ক nb03,
        ধ্রুৱ পূর্ণাঙ্ক ne00_src, ধ্রুৱ পূর্ণাঙ্ক ne01_src,
        ধ্রুৱ পূর্ণাঙ্ক ne10_dst, ধ্রুৱ পূর্ণাঙ্ক ne11_dst, ধ্রুৱ পূর্ণাঙ্ক ne12_dst, ধ্রুৱ পূর্ণাঙ্ক ne13_dst,
        ধ্রুৱ ভিন্ন sf0, ধ্রুৱ ভিন্ন sf1, ধ্রুৱ ভিন্ন sf2, ধ্রুৱ ভিন্ন sf3,
        ধ্রুৱ ভিন্ন pixel_offset) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ধ্রুৱ int64_t index = item_ct1.get_local_id(2) +
        item_ct1.get_group(2) * item_ct1.get_local_range(2);
    ধ্রুৱ int64_t dst_total_elements = ne10_dst * ne11_dst * ne12_dst * ne13_dst;

  যদি (index >= dst_total_elements) {
    ঘূৰাই;
  }

    ধ্রুৱ পূর্ণাঙ্ক i10_dst = index % ne10_dst;
    ধ্রুৱ পূর্ণাঙ্ক i11_dst = (index / ne10_dst) % ne11_dst;
    ধ্রুৱ পূর্ণাঙ্ক i12_dst = (index / (ne10_dst * ne11_dst)) % ne12_dst;
    ধ্রুৱ পূর্ণাঙ্ক i13_dst = index / (ne10_dst * ne11_dst * ne12_dst);

    ধ্রুৱ পূর্ণাঙ্ক i02_src = (পূর্ণাঙ্ক)(i12_dst / sf2);
    ধ্রুৱ পূর্ণাঙ্ক i03_src = (পূর্ণাঙ্ক)(i13_dst / sf3);

    ধ্রুৱ ভিন্ন y_src_f = ((ভিন্ন)i11_dst + pixel_offset) / sf1 - pixel_offset;
    পূর্ণাঙ্ক         y0_src    = (পূর্ণাঙ্ক) sycl::floor((ভিন্ন) y_src_f);
    পূর্ণাঙ্ক y1_src    = y0_src + 1;

    y0_src = sycl::max(0, sycl::min(y0_src, ne01_src - 1));
    y1_src = sycl::max(0, sycl::min(y1_src, ne01_src - 1));

    ভিন্ন dy = y_src_f - (ভিন্ন)y0_src;
    dy       = sycl::max(0.0f, sycl::min(dy, 1.0f));

    ভিন্ন x_src_f = ((ভিন্ন)i10_dst + pixel_offset) / sf0 - pixel_offset;
    পূর্ণাঙ্ক   x0_src    = (পূর্ণাঙ্ক) sycl::floor(x_src_f);
    পূর্ণাঙ্ক x1_src    = x0_src + 1;

    x0_src = sycl::max(0, sycl::min(x0_src, ne00_src - 1));
    x1_src = sycl::max(0, sycl::min(x1_src, ne00_src - 1));

    ভিন্ন dx = x_src_f - (ভিন্ন)x0_src;
    dx       = sycl::max(0.0f, sycl::min(dx, 1.0f));

    ধ্রুৱ ভিন্ন* p_a =
        (ধ্রুৱ ভিন্ন*)((ধ্রুৱ ৱর্ণ*)x + (int64_t)x0_src * nb00 +
                       (int64_t)y0_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    ধ্রুৱ ভিন্ন* p_b =
        (ধ্রুৱ ভিন্ন*)((ধ্রুৱ ৱর্ণ*)x + (int64_t)x1_src * nb00 +
                       (int64_t)y0_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    ধ্রুৱ ভিন্ন* p_c =
        (ধ্রুৱ ভিন্ন*)((ধ্রুৱ ৱর্ণ*)x + (int64_t)x0_src * nb00 +
                       (int64_t)y1_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    ধ্রুৱ ভিন্ন* p_d =
        (ধ্রুৱ ভিন্ন*)((ধ্রুৱ ৱর্ণ*)x + (int64_t)x1_src * nb00 +
                       (int64_t)y1_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);

    ধ্রুৱ ভিন্ন val_a = *p_a;
    ধ্রুৱ ভিন্ন val_b = *p_b;
    ধ্রুৱ ভিন্ন val_c = *p_c;
    ধ্রুৱ ভিন্ন val_d = *p_d;

    ভিন্ন result = val_a * (1.0f - dx) * (1.0f - dy) +
                   val_b * dx * (1.0f - dy) +
                   val_c * (1.0f - dx) * dy +
                   val_d * dx * dy;

    dst[index] = result;
}

// Similar to F.interpolate(..., mode="bilinear", align_corners=False, antialias=True)
// https://github.com/pytorch/pytorch/blob/8871ff29b743948d1225389d5b7068f37b22750b/aten/src/ATen/native/cpu/UpSampleKernel.cpp
স্থির শূন্য upscale_f32_bilinear_antialias(ধ্রুৱ ভিন্ন * src0,
                                           ভিন্ন *       dst,
                                           ধ্রুৱ পূর্ণাঙ্ক     nb00,
                                           ধ্রুৱ পূর্ণাঙ্ক     nb01,
                                           ধ্রুৱ পূর্ণাঙ্ক     nb02,
                                           ধ্রুৱ পূর্ণাঙ্ক     nb03,
                                           ধ্রুৱ পূর্ণাঙ্ক     ne00_src,
                                           ধ্রুৱ পূর্ণাঙ্ক     ne01_src,
                                           ধ্রুৱ পূর্ণাঙ্ক     ne10_dst,
                                           ধ্রুৱ পূর্ণাঙ্ক     ne11_dst,
                                           ধ্রুৱ পূর্ণাঙ্ক     ne12_dst,
                                           ধ্রুৱ পূর্ণাঙ্ক     ne13_dst,
                                           ধ্রুৱ ভিন্ন   sf0,
                                           ধ্রুৱ ভিন্ন   sf1,
                                           ধ্রুৱ ভিন্ন   sf2,
                                           ধ্রুৱ ভিন্ন   sf3,
                                           ধ্রুৱ ভিন্ন   pixel_offset) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ধ্রুৱ int64_t index = item_ct1.get_local_id(2) +
        item_ct1.get_group(2) * item_ct1.get_local_range(2);
    ধ্রুৱ int64_t dst_total_elements = ne10_dst * ne11_dst * ne12_dst * ne13_dst;

    যদি (index >= dst_total_elements) {
        ঘূৰাই;
    }

    ধ্রুৱ পূর্ণাঙ্ক i10_dst = index % ne10_dst;
    ধ্রুৱ পূর্ণাঙ্ক i11_dst = (index / ne10_dst) % ne11_dst;
    ধ্রুৱ পূর্ণাঙ্ক i12_dst = (index / (ne10_dst * ne11_dst)) % ne12_dst;
    ধ্রুৱ পূর্ণাঙ্ক i13_dst = index / (ne10_dst * ne11_dst * ne12_dst);

    ধ্রুৱ পূর্ণাঙ্ক i02_src = (পূর্ণাঙ্ক)(i12_dst / sf2);
    ধ্রুৱ পূর্ণাঙ্ক i03_src = (পূর্ণাঙ্ক)(i13_dst / sf3);

    ধ্রুৱ ভিন্ন y = ((ভিন্ন)i11_dst + pixel_offset) / sf1;
    ধ্রুৱ ভিন্ন x = ((ভিন্ন)i10_dst + pixel_offset) / sf0;

    // support and invscale, minimum 1 pixel for bilinear
    ধ্রুৱ ভিন্ন support1  = sycl::max(1.0f / sf1, 1.0f);
    ধ্রুৱ ভিন্ন invscale1 = 1.0f / support1;
    ধ্রুৱ ভিন্ন support0  = sycl::max(1.0f / sf0, 1.0f);
    ধ্রুৱ ভিন্ন invscale0 = 1.0f / support0;

    // the range of source pixels that contribute
    ধ্রুৱ int64_t x_min = sycl::max(int64_t(0), int64_t(x - support0 + pixel_offset));
    ধ্রুৱ int64_t x_max = sycl::min(int64_t(ne00_src), int64_t(x + support0 + pixel_offset));
    ধ্রুৱ int64_t y_min = sycl::max(int64_t(0), int64_t(y - support1 + pixel_offset));
    ধ্রুৱ int64_t y_max = sycl::min(int64_t(ne01_src), int64_t(y + support1 + pixel_offset));

    // bilinear filter with antialiasing
    ভিন্ন val = 0.0f;
    ভিন্ন total_weight = 0.0f;

    auto triangle_filter = [](ভিন্ন x) -> ভিন্ন {
        ঘূৰাই sycl::max(1.0f - sycl::fabs(x), 0.0f);
    };

    বাবে (int64_t sy = y_min; sy < y_max; sy++) {
        ধ্রুৱ ভিন্ন weight_y = triangle_filter((sy - y + pixel_offset) * invscale1);

        বাবে (int64_t sx = x_min; sx < x_max; sx++) {
            ধ্রুৱ ভিন্ন weight_x = triangle_filter((sx - x + pixel_offset) * invscale0);
            ধ্রুৱ ভিন্ন weight = weight_x * weight_y;

            যদি (weight <= 0.0f) {
                অনুৱর্ত;
            }

            ধ্রুৱ ভিন্ন pixel =
                *(ধ্রুৱ ভিন্ন*)((ধ্রুৱ ৱর্ণ*)src0 + sx * nb00 + sy * nb01 +
                                i02_src * nb02 + i03_src * nb03);
            val += pixel * weight;
            total_weight += weight;
        }
    }

    যদি (total_weight > 0.0f) {
        val /= total_weight;
    }

    dst[index] = val;
}

নামাকাশ bicubic_interpolation {
স্থির ভিন্ন weight1(ভিন্ন x, ধ্রুৱ ভিন্ন &a) { ঘূৰাই ((a + 2) * x - (a + 3)) * x * x + 1; }
স্থির ভিন্ন weight2(ভিন্ন x, ধ্রুৱ ভিন্ন &a) { ঘূৰাই ((a * x - 5 * a) * x + 8 * a) * x - 4 * a; }

স্থির ভিন্ন bicubic(ভিন্ন p0, ভিন্ন p1, ভিন্ন p2, ভিন্ন p3, ভিন্ন x, ভিন্ন a) {
    ধ্রুৱ ভিন্ন w0 = weight2(x + 1, a);
    ধ্রুৱ ভিন্ন w1 = weight1(x + 0, a);
    ধ্রুৱ ভিন্ন w2 = weight1(1 - x, a);
    ধ্রুৱ ভিন্ন w3 = weight2(2 - x, a);
    ঘূৰাই p0 * w0 + p1 * w1 + p2 * w2 + p3 * w3;
}

}

স্থির শূন্য upscale_f32_bicubic(ধ্রুৱ ভিন্ন * x, ভিন্ন * dst,
        ধ্রুৱ পূর্ণাঙ্ক nb00, ধ্রুৱ পূর্ণাঙ্ক nb01, ধ্রুৱ পূর্ণাঙ্ক nb02, ধ্রুৱ পূর্ণাঙ্ক nb03,
        ধ্রুৱ পূর্ণাঙ্ক ne00_src, ধ্রুৱ পূর্ণাঙ্ক ne01_src,
        ধ্রুৱ পূর্ণাঙ্ক ne10_dst, ধ্রুৱ পূর্ণাঙ্ক ne11_dst, ধ্রুৱ পূর্ণাঙ্ক ne12_dst, ধ্রুৱ পূর্ণাঙ্ক ne13_dst,
        ধ্রুৱ ভিন্ন sf0, ধ্রুৱ ভিন্ন sf1, ধ্রুৱ ভিন্ন sf2, ধ্রুৱ ভিন্ন sf3,
        ধ্রুৱ ভিন্ন pixel_offset) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ধ্রুৱ ভিন্ন a = -0.75f;
    using bicubic_interpolation::bicubic;

    ধ্রুৱ int64_t index = item_ct1.get_local_id(2) +
        item_ct1.get_group(2) * item_ct1.get_local_range(2);
    ধ্রুৱ int64_t dst_total_elements =
        ne10_dst * ne11_dst * ne12_dst * ne13_dst;

    যদি (index >= dst_total_elements) {
        ঘূৰাই;
    }

    ধ্রুৱ পূর্ণাঙ্ক i10_dst = index % ne10_dst;
    ধ্রুৱ পূর্ণাঙ্ক i11_dst = (index / ne10_dst) % ne11_dst;
    ধ্রুৱ পূর্ণাঙ্ক i12_dst = (index / (ne10_dst * ne11_dst)) % ne12_dst;
    ধ্রুৱ পূর্ণাঙ্ক i13_dst = index / (ne10_dst * ne11_dst * ne12_dst);

    ধ্রুৱ পূর্ণাঙ্ক i02_src = (পূর্ণাঙ্ক)(i12_dst / sf2);
    ধ্রুৱ পূর্ণাঙ্ক i03_src = (পূর্ণাঙ্ক)(i13_dst / sf3);

    ধ্রুৱ ভিন্ন y_src_f = ((ভিন্ন)i11_dst + pixel_offset) / sf1 - pixel_offset;
    ধ্রুৱ পূর্ণাঙ্ক   y0_src  = (পূর্ণাঙ্ক) sycl::floor((ভিন্ন) y_src_f);
    ধ্রুৱ ভিন্ন dy      = y_src_f - (ভিন্ন)y0_src;

    ধ্রুৱ ভিন্ন x_src_f = ((ভিন্ন)i10_dst + pixel_offset) / sf0 - pixel_offset;
    ধ্রুৱ পূর্ণাঙ্ক   x0_src  = (পূর্ণাঙ্ক) sycl::floor((ভিন্ন) x_src_f);
    ধ্রুৱ ভিন্ন dx      = x_src_f - (ভিন্ন)x0_src;

    ধ্রুৱ ৱর্ণ * x_base = (ধ্রুৱ ৱর্ণ *)x + (int64_t)i02_src * nb02 + (int64_t)i03_src * nb03;

    auto load = [=](পূর্ণাঙ্ক x_off, পূর্ণাঙ্ক y_off) -> ভিন্ন {
        পূর্ণাঙ্ক i00_src = sycl::max(0, sycl::min(x0_src + x_off, ne00_src - 1));
        পূর্ণাঙ্ক i01_src = sycl::max(0, sycl::min(y0_src + y_off, ne01_src - 1));
        ঘূৰাই *(ধ্রুৱ ভিন্ন *)(x_base + (int64_t)i00_src * nb00 + (int64_t)i01_src * nb01);
    };

    ধ্রুৱ ভিন্ন result = bicubic(
        bicubic(load(-1, -1), load(0, -1), load(1, -1), load(2, -1), dx, a),
        bicubic(load(-1, 0), load(0, 0), load(1, 0), load(2, 0), dx, a),
        bicubic(load(-1, 1), load(0, 1), load(1, 1), load(2, 1), dx, a),
        bicubic(load(-1, 2), load(0, 2), load(1, 2), load(2, 2), dx, a),
        dy,
        a);

    dst[index] = result;
}

স্থির শূন্য upscale_f32_sycl(ধ্রুৱ ভিন্ন *   x,
                             ভিন্ন *         dst,
                             ধ্রুৱ পূর্ণাঙ্ক       nb00,
                             ধ্রুৱ পূর্ণাঙ্ক       nb01,
                             ধ্রুৱ পূর্ণাঙ্ক       nb02,
                             ধ্রুৱ পূর্ণাঙ্ক       nb03,
                             ধ্রুৱ পূর্ণাঙ্ক       ne10,
                             ধ্রুৱ পূর্ণাঙ্ক       ne11,
                             ধ্রুৱ পূর্ণাঙ্ক       ne12,
                             ধ্রুৱ পূর্ণাঙ্ক       ne13,
                             ধ্রুৱ ভিন্ন     sf0,
                             ধ্রুৱ ভিন্ন     sf1,
                             ধ্রুৱ ভিন্ন     sf2,
                             ধ্রুৱ ভিন্ন     sf3,
                             dpct::queue_ptr stream) {
    ধ্রুৱ int64_t dst_size   = ne10 * ne11 * ne12 * ne13;
    ধ্রুৱ int64_t অঙ্ক_blocks = (dst_size + SYCL_UPSCALE_BLOCK_SIZE - 1) / SYCL_UPSCALE_BLOCK_SIZE;

    stream->parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE),
             sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE)),
        [=](sycl::nd_item<3> /*item_ct1*/) {
            upscale_f32(x, dst, nb00, nb01, nb02, nb03, ne10, ne11, ne12, ne13, sf0, sf1, sf2, sf3);
        });
}

স্থির শূন্য upscale_f32_bilinear_sycl(ধ্রুৱ ভিন্ন *   x,
                                      ভিন্ন *         dst,
                                      ধ্রুৱ পূর্ণাঙ্ক       nb00,
                                      ধ্রুৱ পূর্ণাঙ্ক       nb01,
                                      ধ্রুৱ পূর্ণাঙ্ক       nb02,
                                      ধ্রুৱ পূর্ণাঙ্ক       nb03,
                                      ধ্রুৱ পূর্ণাঙ্ক       ne00_src,
                                      ধ্রুৱ পূর্ণাঙ্ক       ne01_src,
                                      ধ্রুৱ পূর্ণাঙ্ক       ne10_dst,
                                      ধ্রুৱ পূর্ণাঙ্ক       ne11_dst,
                                      ধ্রুৱ পূর্ণাঙ্ক       ne12_dst,
                                      ধ্রুৱ পূর্ণাঙ্ক       ne13_dst,
                                      ধ্রুৱ ভিন্ন     sf0,
                                      ধ্রুৱ ভিন্ন     sf1,
                                      ধ্রুৱ ভিন্ন     sf2,
                                      ধ্রুৱ ভিন্ন     sf3,
                                      ধ্রুৱ ভিন্ন     pixel_offset,
                                      তার্কিক            antialias,
                                      dpct::queue_ptr stream) {
    ধ্রুৱ int64_t dst_size   = ne10_dst * ne11_dst * ne12_dst * ne13_dst;
    ধ্রুৱ int64_t অঙ্ক_blocks = (dst_size + SYCL_UPSCALE_BLOCK_SIZE - 1) / SYCL_UPSCALE_BLOCK_SIZE;

    যদি (antialias) {
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE),
                sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE)),
            [=](sycl::nd_item<3> /*item_ct1*/) {
                upscale_f32_bilinear_antialias(
                    x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst,
                    ne12_dst, ne13_dst, sf0, sf1, sf2, sf3, pixel_offset);
            });
    } অন্যথা {
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE),
                sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE)),
            [=](sycl::nd_item<3> /*item_ct1*/) {
                upscale_f32_bilinear(
                    x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst, ne12_dst,
                    ne13_dst, sf0, sf1, sf2, sf3, pixel_offset);
            });
    }
}

স্থির শূন্য upscale_f32_bicubic_sycl(ধ্রুৱ ভিন্ন *   x,
                                     ভিন্ন *         dst,
                                     ধ্রুৱ পূর্ণাঙ্ক       nb00,
                                     ধ্রুৱ পূর্ণাঙ্ক       nb01,
                                     ধ্রুৱ পূর্ণাঙ্ক       nb02,
                                     ধ্রুৱ পূর্ণাঙ্ক       nb03,
                                     ধ্রুৱ পূর্ণাঙ্ক       ne00_src,
                                     ধ্রুৱ পূর্ণাঙ্ক       ne01_src,
                                     ধ্রুৱ পূর্ণাঙ্ক       ne10_dst,
                                     ধ্রুৱ পূর্ণাঙ্ক       ne11_dst,
                                     ধ্রুৱ পূর্ণাঙ্ক       ne12_dst,
                                     ধ্রুৱ পূর্ণাঙ্ক       ne13_dst,
                                     ধ্রুৱ ভিন্ন     sf0,
                                     ধ্রুৱ ভিন্ন     sf1,
                                     ধ্রুৱ ভিন্ন     sf2,
                                     ধ্রুৱ ভিন্ন     sf3,
                                     ধ্রুৱ ভিন্ন     pixel_offset,
                                     dpct::queue_ptr stream) {
    ধ্রুৱ int64_t dst_size   = ne10_dst * ne11_dst * ne12_dst * ne13_dst;
    ধ্রুৱ int64_t অঙ্ক_blocks = (dst_size + SYCL_UPSCALE_BLOCK_SIZE - 1) / SYCL_UPSCALE_BLOCK_SIZE;

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE),
                    sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE)),
                [=](sycl::nd_item<3> /*item_ct1*/) {
                    upscale_f32_bicubic(
                        x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst,
                        ne12_dst, ne13_dst, sf0, sf1, sf2, sf3, pixel_offset);
                });
        });
    }
}

স্থির শূন্য ggml_sycl_op_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ধ্রুৱ ggml_tensor * src0 = dst->src[0];
    ধ্রুৱ ভিন্ন * src0_d = (ধ্রুৱ ভিন্ন *)src0->data;
    ভিন্ন * dst_d = (ভিন্ন *)dst->data;
    dpct::queue_ptr     stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    ধ্রুৱ পূর্ণাঙ্ক mode_flags = dst->op_params[0];
    ধ্রুৱ ggml_scale_mode mode = (ggml_scale_mode)(mode_flags & 0xFF);

    ভিন্ন sf0 = (ভিন্ন)dst->ne[0]/src0->ne[0];
    ভিন্ন sf1 = (ভিন্ন)dst->ne[1]/src0->ne[1];
    ভিন্ন sf2 = (ভিন্ন)dst->ne[2]/src0->ne[2];
    ধ্রুৱ ভিন্ন sf3 = (ভিন্ন)dst->ne[3]/src0->ne[3];

    ভিন্ন pixel_offset = 0.5f;
    যদি (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        sf0 = dst->ne[0] > 1 && src0->ne[0] > 1
            ? (ভিন্ন)(dst->ne[0] - 1) / (src0->ne[0] - 1)
            : sf0;
        sf1 = dst->ne[1] > 1 && src0->ne[1] > 1
            ? (ভিন্ন)(dst->ne[1] - 1) / (src0->ne[1] - 1)
            : sf1;
        pixel_offset = 0.0f;
    }

    যদি (mode == GGML_SCALE_MODE_NEAREST) {
        upscale_f32_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], sf0, sf1, sf2, sf3, stream);
    } অন্যথা যদি (mode == GGML_SCALE_MODE_BILINEAR) {
        ধ্রুৱ তার্কিক antialias = (mode_flags & GGML_SCALE_FLAG_ANTIALIAS);
        upscale_f32_bilinear_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src0->ne[0], src0->ne[1], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            sf0, sf1, sf2, sf3, pixel_offset, antialias, stream);
    } অন্যথা যদি (mode == GGML_SCALE_MODE_BICUBIC) {
        upscale_f32_bicubic_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src0->ne[0], src0->ne[1], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            sf0, sf1, sf2, sf3, pixel_offset, stream);
    }
}

শূন্য ggml_sycl_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_upscale(ctx, dst);
}
