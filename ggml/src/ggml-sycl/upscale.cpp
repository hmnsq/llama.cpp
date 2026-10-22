#অন্তর্ভাব "upscale.hpp"

স্থির শূন্য upscale_f32(ধ্রুবক দশমিক * x, দশমিক * dst,
        ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
        ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা ne13,
        ধ্রুবক দশমিক sf0, ধ্রুবক দশমিক sf1, ধ্রুবক দশমিক sf2, ধ্রুবক দশমিক sf3) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    পূর্ণসংখ্যা  সূচক    = item_ct1.get_local_id(2) + item_ct1.get_group(2) * item_ct1.get_local_range(2);
    যদি (সূচক >= ne10 * ne11 * ne12 * ne13) {
        ফেরত;
    }

    পূর্ণসংখ্যা i10 = সূচক % ne10;
    পূর্ণসংখ্যা i11 = (সূচক / ne10) % ne11;
    পূর্ণসংখ্যা i12 = (সূচক / (ne10 * ne11)) % ne12;
    পূর্ণসংখ্যা i13 = (সূচক / (ne10 * ne11 * ne12)) % ne13;

    পূর্ণসংখ্যা i00 = i10 / sf0;
    পূর্ণসংখ্যা i01 = i11 / sf1;
    পূর্ণসংখ্যা i02 = i12 / sf2;
    পূর্ণসংখ্যা i03 = i13 / sf3;

    dst[সূচক] = *((ধ্রুবক দশমিক*)((ধ্রুবক অক্ষর*)x + i03 * nb03 + i02 * nb02 +
                                  i01 * nb01 + i00 * nb00));
}

স্থির শূন্য upscale_f32_bilinear(ধ্রুবক দশমিক * x, দশমিক * dst,
        ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
        ধ্রুবক পূর্ণসংখ্যা ne00_src, ধ্রুবক পূর্ণসংখ্যা ne01_src,
        ধ্রুবক পূর্ণসংখ্যা ne10_dst, ধ্রুবক পূর্ণসংখ্যা ne11_dst, ধ্রুবক পূর্ণসংখ্যা ne12_dst, ধ্রুবক পূর্ণসংখ্যা ne13_dst,
        ধ্রুবক দশমিক sf0, ধ্রুবক দশমিক sf1, ধ্রুবক দশমিক sf2, ধ্রুবক দশমিক sf3,
        ধ্রুবক দশমিক pixel_offset) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ধ্রুবক int64_t সূচক = item_ct1.get_local_id(2) +
        item_ct1.get_group(2) * item_ct1.get_local_range(2);
    ধ্রুবক int64_t dst_যোগ_elements = ne10_dst * ne11_dst * ne12_dst * ne13_dst;

  যদি (সূচক >= dst_যোগ_elements) {
    ফেরত;
  }

    ধ্রুবক পূর্ণসংখ্যা i10_dst = সূচক % ne10_dst;
    ধ্রুবক পূর্ণসংখ্যা i11_dst = (সূচক / ne10_dst) % ne11_dst;
    ধ্রুবক পূর্ণসংখ্যা i12_dst = (সূচক / (ne10_dst * ne11_dst)) % ne12_dst;
    ধ্রুবক পূর্ণসংখ্যা i13_dst = সূচক / (ne10_dst * ne11_dst * ne12_dst);

    ধ্রুবক পূর্ণসংখ্যা i02_src = (পূর্ণসংখ্যা)(i12_dst / sf2);
    ধ্রুবক পূর্ণসংখ্যা i03_src = (পূর্ণসংখ্যা)(i13_dst / sf3);

    ধ্রুবক দশমিক y_src_f = ((দশমিক)i11_dst + pixel_offset) / sf1 - pixel_offset;
    পূর্ণসংখ্যা         y0_src    = (পূর্ণসংখ্যা) sycl::floor((দশমিক) y_src_f);
    পূর্ণসংখ্যা y1_src    = y0_src + 1;

    y0_src = sycl::max(0, sycl::min(y0_src, ne01_src - 1));
    y1_src = sycl::max(0, sycl::min(y1_src, ne01_src - 1));

    দশমিক dy = y_src_f - (দশমিক)y0_src;
    dy       = sycl::max(0.0f, sycl::min(dy, 1.0f));

    দশমিক x_src_f = ((দশমিক)i10_dst + pixel_offset) / sf0 - pixel_offset;
    পূর্ণসংখ্যা   x0_src    = (পূর্ণসংখ্যা) sycl::floor(x_src_f);
    পূর্ণসংখ্যা x1_src    = x0_src + 1;

    x0_src = sycl::max(0, sycl::min(x0_src, ne00_src - 1));
    x1_src = sycl::max(0, sycl::min(x1_src, ne00_src - 1));

    দশমিক dx = x_src_f - (দশমিক)x0_src;
    dx       = sycl::max(0.0f, sycl::min(dx, 1.0f));

    ধ্রুবক দশমিক* p_a =
        (ধ্রুবক দশমিক*)((ধ্রুবক অক্ষর*)x + (int64_t)x0_src * nb00 +
                       (int64_t)y0_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    ধ্রুবক দশমিক* p_b =
        (ধ্রুবক দশমিক*)((ধ্রুবক অক্ষর*)x + (int64_t)x1_src * nb00 +
                       (int64_t)y0_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    ধ্রুবক দশমিক* p_c =
        (ধ্রুবক দশমিক*)((ধ্রুবক অক্ষর*)x + (int64_t)x0_src * nb00 +
                       (int64_t)y1_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    ধ্রুবক দশমিক* p_d =
        (ধ্রুবক দশমিক*)((ধ্রুবক অক্ষর*)x + (int64_t)x1_src * nb00 +
                       (int64_t)y1_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);

    ধ্রুবক দশমিক val_a = *p_a;
    ধ্রুবক দশমিক val_b = *p_b;
    ধ্রুবক দশমিক val_c = *p_c;
    ধ্রুবক দশমিক val_d = *p_d;

    দশমিক ফলাফল = val_a * (1.0f - dx) * (1.0f - dy) +
                   val_b * dx * (1.0f - dy) +
                   val_c * (1.0f - dx) * dy +
                   val_d * dx * dy;

    dst[সূচক] = ফলাফল;
}

// Similar to F.interpolate(..., mode="bilinear", align_corners=False, antialias=True)
// https://github.com/pytorch/pytorch/blob/8871ff29b743948d1225389d5b7068f37b22750b/aten/src/ATen/native/cpu/UpSampleKernel.cpp
স্থির শূন্য upscale_f32_bilinear_antialias(ধ্রুবক দশমিক * src0,
                                           দশমিক *       dst,
                                           ধ্রুবক পূর্ণসংখ্যা     nb00,
                                           ধ্রুবক পূর্ণসংখ্যা     nb01,
                                           ধ্রুবক পূর্ণসংখ্যা     nb02,
                                           ধ্রুবক পূর্ণসংখ্যা     nb03,
                                           ধ্রুবক পূর্ণসংখ্যা     ne00_src,
                                           ধ্রুবক পূর্ণসংখ্যা     ne01_src,
                                           ধ্রুবক পূর্ণসংখ্যা     ne10_dst,
                                           ধ্রুবক পূর্ণসংখ্যা     ne11_dst,
                                           ধ্রুবক পূর্ণসংখ্যা     ne12_dst,
                                           ধ্রুবক পূর্ণসংখ্যা     ne13_dst,
                                           ধ্রুবক দশমিক   sf0,
                                           ধ্রুবক দশমিক   sf1,
                                           ধ্রুবক দশমিক   sf2,
                                           ধ্রুবক দশমিক   sf3,
                                           ধ্রুবক দশমিক   pixel_offset) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ধ্রুবক int64_t সূচক = item_ct1.get_local_id(2) +
        item_ct1.get_group(2) * item_ct1.get_local_range(2);
    ধ্রুবক int64_t dst_যোগ_elements = ne10_dst * ne11_dst * ne12_dst * ne13_dst;

    যদি (সূচক >= dst_যোগ_elements) {
        ফেরত;
    }

    ধ্রুবক পূর্ণসংখ্যা i10_dst = সূচক % ne10_dst;
    ধ্রুবক পূর্ণসংখ্যা i11_dst = (সূচক / ne10_dst) % ne11_dst;
    ধ্রুবক পূর্ণসংখ্যা i12_dst = (সূচক / (ne10_dst * ne11_dst)) % ne12_dst;
    ধ্রুবক পূর্ণসংখ্যা i13_dst = সূচক / (ne10_dst * ne11_dst * ne12_dst);

    ধ্রুবক পূর্ণসংখ্যা i02_src = (পূর্ণসংখ্যা)(i12_dst / sf2);
    ধ্রুবক পূর্ণসংখ্যা i03_src = (পূর্ণসংখ্যা)(i13_dst / sf3);

    ধ্রুবক দশমিক y = ((দশমিক)i11_dst + pixel_offset) / sf1;
    ধ্রুবক দশমিক x = ((দশমিক)i10_dst + pixel_offset) / sf0;

    // support and invscale, minimum 1 pixel for bilinear
    ধ্রুবক দশমিক support1  = sycl::max(1.0f / sf1, 1.0f);
    ধ্রুবক দশমিক invscale1 = 1.0f / support1;
    ধ্রুবক দশমিক support0  = sycl::max(1.0f / sf0, 1.0f);
    ধ্রুবক দশমিক invscale0 = 1.0f / support0;

    // the range of source pixels that contribute
    ধ্রুবক int64_t x_min = sycl::max(int64_t(0), int64_t(x - support0 + pixel_offset));
    ধ্রুবক int64_t x_max = sycl::min(int64_t(ne00_src), int64_t(x + support0 + pixel_offset));
    ধ্রুবক int64_t y_min = sycl::max(int64_t(0), int64_t(y - support1 + pixel_offset));
    ধ্রুবক int64_t y_max = sycl::min(int64_t(ne01_src), int64_t(y + support1 + pixel_offset));

    // bilinear filter with antialiasing
    দশমিক val = 0.0f;
    দশমিক যোগ_weight = 0.0f;

    auto triangle_filter = [](দশমিক x) -> দশমিক {
        ফেরত sycl::max(1.0f - sycl::fabs(x), 0.0f);
    };

    জন্য (int64_t sy = y_min; sy < y_max; sy++) {
        ধ্রুবক দশমিক weight_y = triangle_filter((sy - y + pixel_offset) * invscale1);

        জন্য (int64_t sx = x_min; sx < x_max; sx++) {
            ধ্রুবক দশমিক weight_x = triangle_filter((sx - x + pixel_offset) * invscale0);
            ধ্রুবক দশমিক weight = weight_x * weight_y;

            যদি (weight <= 0.0f) {
                চালু;
            }

            ধ্রুবক দশমিক pixel =
                *(ধ্রুবক দশমিক*)((ধ্রুবক অক্ষর*)src0 + sx * nb00 + sy * nb01 +
                                i02_src * nb02 + i03_src * nb03);
            val += pixel * weight;
            যোগ_weight += weight;
        }
    }

    যদি (যোগ_weight > 0.0f) {
        val /= যোগ_weight;
    }

    dst[সূচক] = val;
}

নামাকাশ bicubic_interpolation {
স্থির দশমিক weight1(দশমিক x, ধ্রুবক দশমিক &a) { ফেরত ((a + 2) * x - (a + 3)) * x * x + 1; }
স্থির দশমিক weight2(দশমিক x, ধ্রুবক দশমিক &a) { ফেরত ((a * x - 5 * a) * x + 8 * a) * x - 4 * a; }

স্থির দশমিক bicubic(দশমিক p0, দশমিক p1, দশমিক p2, দশমিক p3, দশমিক x, দশমিক a) {
    ধ্রুবক দশমিক w0 = weight2(x + 1, a);
    ধ্রুবক দশমিক w1 = weight1(x + 0, a);
    ধ্রুবক দশমিক w2 = weight1(1 - x, a);
    ধ্রুবক দশমিক w3 = weight2(2 - x, a);
    ফেরত p0 * w0 + p1 * w1 + p2 * w2 + p3 * w3;
}

}

স্থির শূন্য upscale_f32_bicubic(ধ্রুবক দশমিক * x, দশমিক * dst,
        ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
        ধ্রুবক পূর্ণসংখ্যা ne00_src, ধ্রুবক পূর্ণসংখ্যা ne01_src,
        ধ্রুবক পূর্ণসংখ্যা ne10_dst, ধ্রুবক পূর্ণসংখ্যা ne11_dst, ধ্রুবক পূর্ণসংখ্যা ne12_dst, ধ্রুবক পূর্ণসংখ্যা ne13_dst,
        ধ্রুবক দশমিক sf0, ধ্রুবক দশমিক sf1, ধ্রুবক দশমিক sf2, ধ্রুবক দশমিক sf3,
        ধ্রুবক দশমিক pixel_offset) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ধ্রুবক দশমিক a = -0.75f;
    using bicubic_interpolation::bicubic;

    ধ্রুবক int64_t সূচক = item_ct1.get_local_id(2) +
        item_ct1.get_group(2) * item_ct1.get_local_range(2);
    ধ্রুবক int64_t dst_যোগ_elements =
        ne10_dst * ne11_dst * ne12_dst * ne13_dst;

    যদি (সূচক >= dst_যোগ_elements) {
        ফেরত;
    }

    ধ্রুবক পূর্ণসংখ্যা i10_dst = সূচক % ne10_dst;
    ধ্রুবক পূর্ণসংখ্যা i11_dst = (সূচক / ne10_dst) % ne11_dst;
    ধ্রুবক পূর্ণসংখ্যা i12_dst = (সূচক / (ne10_dst * ne11_dst)) % ne12_dst;
    ধ্রুবক পূর্ণসংখ্যা i13_dst = সূচক / (ne10_dst * ne11_dst * ne12_dst);

    ধ্রুবক পূর্ণসংখ্যা i02_src = (পূর্ণসংখ্যা)(i12_dst / sf2);
    ধ্রুবক পূর্ণসংখ্যা i03_src = (পূর্ণসংখ্যা)(i13_dst / sf3);

    ধ্রুবক দশমিক y_src_f = ((দশমিক)i11_dst + pixel_offset) / sf1 - pixel_offset;
    ধ্রুবক পূর্ণসংখ্যা   y0_src  = (পূর্ণসংখ্যা) sycl::floor((দশমিক) y_src_f);
    ধ্রুবক দশমিক dy      = y_src_f - (দশমিক)y0_src;

    ধ্রুবক দশমিক x_src_f = ((দশমিক)i10_dst + pixel_offset) / sf0 - pixel_offset;
    ধ্রুবক পূর্ণসংখ্যা   x0_src  = (পূর্ণসংখ্যা) sycl::floor((দশমিক) x_src_f);
    ধ্রুবক দশমিক dx      = x_src_f - (দশমিক)x0_src;

    ধ্রুবক অক্ষর * x_base = (ধ্রুবক অক্ষর *)x + (int64_t)i02_src * nb02 + (int64_t)i03_src * nb03;

    auto load = [=](পূর্ণসংখ্যা x_off, পূর্ণসংখ্যা y_off) -> দশমিক {
        পূর্ণসংখ্যা i00_src = sycl::max(0, sycl::min(x0_src + x_off, ne00_src - 1));
        পূর্ণসংখ্যা i01_src = sycl::max(0, sycl::min(y0_src + y_off, ne01_src - 1));
        ফেরত *(ধ্রুবক দশমিক *)(x_base + (int64_t)i00_src * nb00 + (int64_t)i01_src * nb01);
    };

    ধ্রুবক দশমিক ফলাফল = bicubic(
        bicubic(load(-1, -1), load(0, -1), load(1, -1), load(2, -1), dx, a),
        bicubic(load(-1, 0), load(0, 0), load(1, 0), load(2, 0), dx, a),
        bicubic(load(-1, 1), load(0, 1), load(1, 1), load(2, 1), dx, a),
        bicubic(load(-1, 2), load(0, 2), load(1, 2), load(2, 2), dx, a),
        dy,
        a);

    dst[সূচক] = ফলাফল;
}

স্থির শূন্য upscale_f32_sycl(ধ্রুবক দশমিক *   x,
                             দশমিক *         dst,
                             ধ্রুবক পূর্ণসংখ্যা       nb00,
                             ধ্রুবক পূর্ণসংখ্যা       nb01,
                             ধ্রুবক পূর্ণসংখ্যা       nb02,
                             ধ্রুবক পূর্ণসংখ্যা       nb03,
                             ধ্রুবক পূর্ণসংখ্যা       ne10,
                             ধ্রুবক পূর্ণসংখ্যা       ne11,
                             ধ্রুবক পূর্ণসংখ্যা       ne12,
                             ধ্রুবক পূর্ণসংখ্যা       ne13,
                             ধ্রুবক দশমিক     sf0,
                             ধ্রুবক দশমিক     sf1,
                             ধ্রুবক দশমিক     sf2,
                             ধ্রুবক দশমিক     sf3,
                             dpct::queue_ptr stream) {
    ধ্রুবক int64_t dst_আকার   = ne10 * ne11 * ne12 * ne13;
    ধ্রুবক int64_t অঙ্ক_blocks = (dst_আকার + SYCL_UPSCALE_BLOCK_আকার - 1) / SYCL_UPSCALE_BLOCK_আকার;

    stream->parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_আকার),
             sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_আকার)),
        [=](sycl::nd_item<3> /*item_ct1*/) {
            upscale_f32(x, dst, nb00, nb01, nb02, nb03, ne10, ne11, ne12, ne13, sf0, sf1, sf2, sf3);
        });
}

স্থির শূন্য upscale_f32_bilinear_sycl(ধ্রুবক দশমিক *   x,
                                      দশমিক *         dst,
                                      ধ্রুবক পূর্ণসংখ্যা       nb00,
                                      ধ্রুবক পূর্ণসংখ্যা       nb01,
                                      ধ্রুবক পূর্ণসংখ্যা       nb02,
                                      ধ্রুবক পূর্ণসংখ্যা       nb03,
                                      ধ্রুবক পূর্ণসংখ্যা       ne00_src,
                                      ধ্রুবক পূর্ণসংখ্যা       ne01_src,
                                      ধ্রুবক পূর্ণসংখ্যা       ne10_dst,
                                      ধ্রুবক পূর্ণসংখ্যা       ne11_dst,
                                      ধ্রুবক পূর্ণসংখ্যা       ne12_dst,
                                      ধ্রুবক পূর্ণসংখ্যা       ne13_dst,
                                      ধ্রুবক দশমিক     sf0,
                                      ধ্রুবক দশমিক     sf1,
                                      ধ্রুবক দশমিক     sf2,
                                      ধ্রুবক দশমিক     sf3,
                                      ধ্রুবক দশমিক     pixel_offset,
                                      যৌক্তিক            antialias,
                                      dpct::queue_ptr stream) {
    ধ্রুবক int64_t dst_আকার   = ne10_dst * ne11_dst * ne12_dst * ne13_dst;
    ধ্রুবক int64_t অঙ্ক_blocks = (dst_আকার + SYCL_UPSCALE_BLOCK_আকার - 1) / SYCL_UPSCALE_BLOCK_আকার;

    যদি (antialias) {
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_আকার),
                sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_আকার)),
            [=](sycl::nd_item<3> /*item_ct1*/) {
                upscale_f32_bilinear_antialias(
                    x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst,
                    ne12_dst, ne13_dst, sf0, sf1, sf2, sf3, pixel_offset);
            });
    } অন্যথা {
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_আকার),
                sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_আকার)),
            [=](sycl::nd_item<3> /*item_ct1*/) {
                upscale_f32_bilinear(
                    x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst, ne12_dst,
                    ne13_dst, sf0, sf1, sf2, sf3, pixel_offset);
            });
    }
}

স্থির শূন্য upscale_f32_bicubic_sycl(ধ্রুবক দশমিক *   x,
                                     দশমিক *         dst,
                                     ধ্রুবক পূর্ণসংখ্যা       nb00,
                                     ধ্রুবক পূর্ণসংখ্যা       nb01,
                                     ধ্রুবক পূর্ণসংখ্যা       nb02,
                                     ধ্রুবক পূর্ণসংখ্যা       nb03,
                                     ধ্রুবক পূর্ণসংখ্যা       ne00_src,
                                     ধ্রুবক পূর্ণসংখ্যা       ne01_src,
                                     ধ্রুবক পূর্ণসংখ্যা       ne10_dst,
                                     ধ্রুবক পূর্ণসংখ্যা       ne11_dst,
                                     ধ্রুবক পূর্ণসংখ্যা       ne12_dst,
                                     ধ্রুবক পূর্ণসংখ্যা       ne13_dst,
                                     ধ্রুবক দশমিক     sf0,
                                     ধ্রুবক দশমিক     sf1,
                                     ধ্রুবক দশমিক     sf2,
                                     ধ্রুবক দশমিক     sf3,
                                     ধ্রুবক দশমিক     pixel_offset,
                                     dpct::queue_ptr stream) {
    ধ্রুবক int64_t dst_আকার   = ne10_dst * ne11_dst * ne12_dst * ne13_dst;
    ধ্রুবক int64_t অঙ্ক_blocks = (dst_আকার + SYCL_UPSCALE_BLOCK_আকার - 1) / SYCL_UPSCALE_BLOCK_আকার;

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_আকার),
                    sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_আকার)),
                [=](sycl::nd_item<3> /*item_ct1*/) {
                    upscale_f32_bicubic(
                        x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst,
                        ne12_dst, ne13_dst, sf0, sf1, sf2, sf3, pixel_offset);
                });
        });
    }
}

স্থির শূন্য ggml_sycl_op_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ধ্রুবক ggml_tensor * src0 = dst->src[0];
    ধ্রুবক দশমিক * src0_d = (ধ্রুবক দশমিক *)src0->তথ্য;
    দশমিক * dst_d = (দশমিক *)dst->তথ্য;
    dpct::queue_ptr     stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    ধ্রুবক পূর্ণসংখ্যা mode_flags = dst->op_params[0];
    ধ্রুবক ggml_scale_mode mode = (ggml_scale_mode)(mode_flags & 0xFF);

    দশমিক sf0 = (দশমিক)dst->ne[0]/src0->ne[0];
    দশমিক sf1 = (দশমিক)dst->ne[1]/src0->ne[1];
    দশমিক sf2 = (দশমিক)dst->ne[2]/src0->ne[2];
    ধ্রুবক দশমিক sf3 = (দশমিক)dst->ne[3]/src0->ne[3];

    দশমিক pixel_offset = 0.5f;
    যদি (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        sf0 = dst->ne[0] > 1 && src0->ne[0] > 1
            ? (দশমিক)(dst->ne[0] - 1) / (src0->ne[0] - 1)
            : sf0;
        sf1 = dst->ne[1] > 1 && src0->ne[1] > 1
            ? (দশমিক)(dst->ne[1] - 1) / (src0->ne[1] - 1)
            : sf1;
        pixel_offset = 0.0f;
    }

    যদি (mode == GGML_SCALE_MODE_NEAREST) {
        upscale_f32_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], sf0, sf1, sf2, sf3, stream);
    } অন্যথা যদি (mode == GGML_SCALE_MODE_BILINEAR) {
        ধ্রুবক যৌক্তিক antialias = (mode_flags & GGML_SCALE_FLAG_ANTIALIAS);
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
