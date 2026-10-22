#अन्तर्भाव "upscale.hpp"

स्थिर शून्य upscale_f32(ध्रुव भिन्न * x, भिन्न * dst,
        ध्रुव पूर्णाङ्क nb00, ध्रुव पूर्णाङ्क nb01, ध्रुव पूर्णाङ्क nb02, ध्रुव पूर्णाङ्क nb03,
        ध्रुव पूर्णाङ्क ne10, ध्रुव पूर्णाङ्क ne11, ध्रुव पूर्णाङ्क ne12, ध्रुव पूर्णाङ्क ne13,
        ध्रुव भिन्न sf0, ध्रुव भिन्न sf1, ध्रुव भिन्न sf2, ध्रुव भिन्न sf3) {
    स्वचालित item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    पूर्णाङ्क  सूचकांक    = item_ct1.get_local_id(2) + item_ct1.get_गणः(2) * item_ct1.get_local_range(2);
    यदि (सूचकांक >= ne10 * ne11 * ne12 * ne13) {
        प्रत्यागम;
    }

    पूर्णाङ्क i10 = सूचकांक % ne10;
    पूर्णाङ्क i11 = (सूचकांक / ne10) % ne11;
    पूर्णाङ्क i12 = (सूचकांक / (ne10 * ne11)) % ne12;
    पूर्णाङ्क i13 = (सूचकांक / (ne10 * ne11 * ne12)) % ne13;

    पूर्णाङ्क i00 = i10 / sf0;
    पूर्णाङ्क i01 = i11 / sf1;
    पूर्णाङ्क i02 = i12 / sf2;
    पूर्णाङ्क i03 = i13 / sf3;

    dst[सूचकांक] = *((ध्रुव भिन्न*)((ध्रुव वर्ण*)x + i03 * nb03 + i02 * nb02 +
                                  i01 * nb01 + i00 * nb00));
}

स्थिर शून्य upscale_f32_bilinear(ध्रुव भिन्न * x, भिन्न * dst,
        ध्रुव पूर्णाङ्क nb00, ध्रुव पूर्णाङ्क nb01, ध्रुव पूर्णाङ्क nb02, ध्रुव पूर्णाङ्क nb03,
        ध्रुव पूर्णाङ्क ne00_src, ध्रुव पूर्णाङ्क ne01_src,
        ध्रुव पूर्णाङ्क ne10_dst, ध्रुव पूर्णाङ्क ne11_dst, ध्रुव पूर्णाङ्क ne12_dst, ध्रुव पूर्णाङ्क ne13_dst,
        ध्रुव भिन्न sf0, ध्रुव भिन्न sf1, ध्रुव भिन्न sf2, ध्रुव भिन्न sf3,
        ध्रुव भिन्न pixel_विस्थापनम्) {
    स्वचालित item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ध्रुव int64_t सूचकांक = item_ct1.get_local_id(2) +
        item_ct1.get_गणः(2) * item_ct1.get_local_range(2);
    ध्रुव int64_t dst_योग_elements = ne10_dst * ne11_dst * ne12_dst * ne13_dst;

  यदि (सूचकांक >= dst_योग_elements) {
    प्रत्यागम;
  }

    ध्रुव पूर्णाङ्क i10_dst = सूचकांक % ne10_dst;
    ध्रुव पूर्णाङ्क i11_dst = (सूचकांक / ne10_dst) % ne11_dst;
    ध्रुव पूर्णाङ्क i12_dst = (सूचकांक / (ne10_dst * ne11_dst)) % ne12_dst;
    ध्रुव पूर्णाङ्क i13_dst = सूचकांक / (ne10_dst * ne11_dst * ne12_dst);

    ध्रुव पूर्णाङ्क i02_src = (पूर्णाङ्क)(i12_dst / sf2);
    ध्रुव पूर्णाङ्क i03_src = (पूर्णाङ्क)(i13_dst / sf3);

    ध्रुव भिन्न y_src_f = ((भिन्न)i11_dst + pixel_विस्थापनम्) / sf1 - pixel_विस्थापनम्;
    पूर्णाङ्क         y0_src    = (पूर्णाङ्क) sycl::floor((भिन्न) y_src_f);
    पूर्णाङ्क y1_src    = y0_src + 1;

    y0_src = sycl::max(0, sycl::min(y0_src, ne01_src - 1));
    y1_src = sycl::max(0, sycl::min(y1_src, ne01_src - 1));

    भिन्न dy = y_src_f - (भिन्न)y0_src;
    dy       = sycl::max(0.0f, sycl::min(dy, 1.0f));

    भिन्न x_src_f = ((भिन्न)i10_dst + pixel_विस्थापनम्) / sf0 - pixel_विस्थापनम्;
    पूर्णाङ्क   x0_src    = (पूर्णाङ्क) sycl::floor(x_src_f);
    पूर्णाङ्क x1_src    = x0_src + 1;

    x0_src = sycl::max(0, sycl::min(x0_src, ne00_src - 1));
    x1_src = sycl::max(0, sycl::min(x1_src, ne00_src - 1));

    भिन्न dx = x_src_f - (भिन्न)x0_src;
    dx       = sycl::max(0.0f, sycl::min(dx, 1.0f));

    ध्रुव भिन्न* p_a =
        (ध्रुव भिन्न*)((ध्रुव वर्ण*)x + (int64_t)x0_src * nb00 +
                       (int64_t)y0_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    ध्रुव भिन्न* p_b =
        (ध्रुव भिन्न*)((ध्रुव वर्ण*)x + (int64_t)x1_src * nb00 +
                       (int64_t)y0_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    ध्रुव भिन्न* p_c =
        (ध्रुव भिन्न*)((ध्रुव वर्ण*)x + (int64_t)x0_src * nb00 +
                       (int64_t)y1_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    ध्रुव भिन्न* p_d =
        (ध्रुव भिन्न*)((ध्रुव वर्ण*)x + (int64_t)x1_src * nb00 +
                       (int64_t)y1_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);

    ध्रुव भिन्न val_a = *p_a;
    ध्रुव भिन्न val_b = *p_b;
    ध्रुव भिन्न val_c = *p_c;
    ध्रुव भिन्न val_d = *p_d;

    भिन्न परिणाम = val_a * (1.0f - dx) * (1.0f - dy) +
                   val_b * dx * (1.0f - dy) +
                   val_c * (1.0f - dx) * dy +
                   val_d * dx * dy;

    dst[सूचकांक] = परिणाम;
}

// Similar to F.interpolate(..., mode="bilinear", align_corners=False, antialias=True)
// https://github.com/pytorch/pytorch/blob/8871ff29b743948d1225389d5b7068f37b22750b/aten/src/ATen/native/cpu/UpSampleKernel.cpp
स्थिर शून्य upscale_f32_bilinear_antialias(ध्रुव भिन्न * src0,
                                           भिन्न *       dst,
                                           ध्रुव पूर्णाङ्क     nb00,
                                           ध्रुव पूर्णाङ्क     nb01,
                                           ध्रुव पूर्णाङ्क     nb02,
                                           ध्रुव पूर्णाङ्क     nb03,
                                           ध्रुव पूर्णाङ्क     ne00_src,
                                           ध्रुव पूर्णाङ्क     ne01_src,
                                           ध्रुव पूर्णाङ्क     ne10_dst,
                                           ध्रुव पूर्णाङ्क     ne11_dst,
                                           ध्रुव पूर्णाङ्क     ne12_dst,
                                           ध्रुव पूर्णाङ्क     ne13_dst,
                                           ध्रुव भिन्न   sf0,
                                           ध्रुव भिन्न   sf1,
                                           ध्रुव भिन्न   sf2,
                                           ध्रुव भिन्न   sf3,
                                           ध्रुव भिन्न   pixel_विस्थापनम्) {
    स्वचालित item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ध्रुव int64_t सूचकांक = item_ct1.get_local_id(2) +
        item_ct1.get_गणः(2) * item_ct1.get_local_range(2);
    ध्रुव int64_t dst_योग_elements = ne10_dst * ne11_dst * ne12_dst * ne13_dst;

    यदि (सूचकांक >= dst_योग_elements) {
        प्रत्यागम;
    }

    ध्रुव पूर्णाङ्क i10_dst = सूचकांक % ne10_dst;
    ध्रुव पूर्णाङ्क i11_dst = (सूचकांक / ne10_dst) % ne11_dst;
    ध्रुव पूर्णाङ्क i12_dst = (सूचकांक / (ne10_dst * ne11_dst)) % ne12_dst;
    ध्रुव पूर्णाङ्क i13_dst = सूचकांक / (ne10_dst * ne11_dst * ne12_dst);

    ध्रुव पूर्णाङ्क i02_src = (पूर्णाङ्क)(i12_dst / sf2);
    ध्रुव पूर्णाङ्क i03_src = (पूर्णाङ्क)(i13_dst / sf3);

    ध्रुव भिन्न y = ((भिन्न)i11_dst + pixel_विस्थापनम्) / sf1;
    ध्रुव भिन्न x = ((भिन्न)i10_dst + pixel_विस्थापनम्) / sf0;

    // support and invscale, minimum 1 pixel for bilinear
    ध्रुव भिन्न support1  = sycl::max(1.0f / sf1, 1.0f);
    ध्रुव भिन्न invscale1 = 1.0f / support1;
    ध्रुव भिन्न support0  = sycl::max(1.0f / sf0, 1.0f);
    ध्रुव भिन्न invscale0 = 1.0f / support0;

    // the range of source pixels that contribute
    ध्रुव int64_t x_min = sycl::max(int64_t(0), int64_t(x - support0 + pixel_विस्थापनम्));
    ध्रुव int64_t x_max = sycl::min(int64_t(ne00_src), int64_t(x + support0 + pixel_विस्थापनम्));
    ध्रुव int64_t y_min = sycl::max(int64_t(0), int64_t(y - support1 + pixel_विस्थापनम्));
    ध्रुव int64_t y_max = sycl::min(int64_t(ne01_src), int64_t(y + support1 + pixel_विस्थापनम्));

    // bilinear filter with antialiasing
    भिन्न val = 0.0f;
    भिन्न योग_weight = 0.0f;

    स्वचालित triangle_filter = [](भिन्न x) -> भिन्न {
        प्रत्यागम sycl::max(1.0f - sycl::fabs(x), 0.0f);
    };

    प्रति (int64_t sy = y_min; sy < y_max; sy++) {
        ध्रुव भिन्न weight_y = triangle_filter((sy - y + pixel_विस्थापनम्) * invscale1);

        प्रति (int64_t sx = x_min; sx < x_max; sx++) {
            ध्रुव भिन्न weight_x = triangle_filter((sx - x + pixel_विस्थापनम्) * invscale0);
            ध्रुव भिन्न weight = weight_x * weight_y;

            यदि (weight <= 0.0f) {
                अनुवर्त;
            }

            ध्रुव भिन्न pixel =
                *(ध्रुव भिन्न*)((ध्रुव वर्ण*)src0 + sx * nb00 + sy * nb01 +
                                i02_src * nb02 + i03_src * nb03);
            val += pixel * weight;
            योग_weight += weight;
        }
    }

    यदि (योग_weight > 0.0f) {
        val /= योग_weight;
    }

    dst[सूचकांक] = val;
}

नामाकाश bicubic_interpolation {
स्थिर भिन्न weight1(भिन्न x, ध्रुव भिन्न &a) { प्रत्यागम ((a + 2) * x - (a + 3)) * x * x + 1; }
स्थिर भिन्न weight2(भिन्न x, ध्रुव भिन्न &a) { प्रत्यागम ((a * x - 5 * a) * x + 8 * a) * x - 4 * a; }

स्थिर भिन्न bicubic(भिन्न p0, भिन्न p1, भिन्न p2, भिन्न p3, भिन्न x, भिन्न a) {
    ध्रुव भिन्न w0 = weight2(x + 1, a);
    ध्रुव भिन्न w1 = weight1(x + 0, a);
    ध्रुव भिन्न w2 = weight1(1 - x, a);
    ध्रुव भिन्न w3 = weight2(2 - x, a);
    प्रत्यागम p0 * w0 + p1 * w1 + p2 * w2 + p3 * w3;
}

}

स्थिर शून्य upscale_f32_bicubic(ध्रुव भिन्न * x, भिन्न * dst,
        ध्रुव पूर्णाङ्क nb00, ध्रुव पूर्णाङ्क nb01, ध्रुव पूर्णाङ्क nb02, ध्रुव पूर्णाङ्क nb03,
        ध्रुव पूर्णाङ्क ne00_src, ध्रुव पूर्णाङ्क ne01_src,
        ध्रुव पूर्णाङ्क ne10_dst, ध्रुव पूर्णाङ्क ne11_dst, ध्रुव पूर्णाङ्क ne12_dst, ध्रुव पूर्णाङ्क ne13_dst,
        ध्रुव भिन्न sf0, ध्रुव भिन्न sf1, ध्रुव भिन्न sf2, ध्रुव भिन्न sf3,
        ध्रुव भिन्न pixel_विस्थापनम्) {
    स्वचालित item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ध्रुव भिन्न a = -0.75f;
    उपयुज्य bicubic_interpolation::bicubic;

    ध्रुव int64_t सूचकांक = item_ct1.get_local_id(2) +
        item_ct1.get_गणः(2) * item_ct1.get_local_range(2);
    ध्रुव int64_t dst_योग_elements =
        ne10_dst * ne11_dst * ne12_dst * ne13_dst;

    यदि (सूचकांक >= dst_योग_elements) {
        प्रत्यागम;
    }

    ध्रुव पूर्णाङ्क i10_dst = सूचकांक % ne10_dst;
    ध्रुव पूर्णाङ्क i11_dst = (सूचकांक / ne10_dst) % ne11_dst;
    ध्रुव पूर्णाङ्क i12_dst = (सूचकांक / (ne10_dst * ne11_dst)) % ne12_dst;
    ध्रुव पूर्णाङ्क i13_dst = सूचकांक / (ne10_dst * ne11_dst * ne12_dst);

    ध्रुव पूर्णाङ्क i02_src = (पूर्णाङ्क)(i12_dst / sf2);
    ध्रुव पूर्णाङ्क i03_src = (पूर्णाङ्क)(i13_dst / sf3);

    ध्रुव भिन्न y_src_f = ((भिन्न)i11_dst + pixel_विस्थापनम्) / sf1 - pixel_विस्थापनम्;
    ध्रुव पूर्णाङ्क   y0_src  = (पूर्णाङ्क) sycl::floor((भिन्न) y_src_f);
    ध्रुव भिन्न dy      = y_src_f - (भिन्न)y0_src;

    ध्रुव भिन्न x_src_f = ((भिन्न)i10_dst + pixel_विस्थापनम्) / sf0 - pixel_विस्थापनम्;
    ध्रुव पूर्णाङ्क   x0_src  = (पूर्णाङ्क) sycl::floor((भिन्न) x_src_f);
    ध्रुव भिन्न dx      = x_src_f - (भिन्न)x0_src;

    ध्रुव वर्ण * x_आधारः = (ध्रुव वर्ण *)x + (int64_t)i02_src * nb02 + (int64_t)i03_src * nb03;

    स्वचालित load = [=](पूर्णाङ्क x_off, पूर्णाङ्क y_off) -> भिन्न {
        पूर्णाङ्क i00_src = sycl::max(0, sycl::min(x0_src + x_off, ne00_src - 1));
        पूर्णाङ्क i01_src = sycl::max(0, sycl::min(y0_src + y_off, ne01_src - 1));
        प्रत्यागम *(ध्रुव भिन्न *)(x_आधारः + (int64_t)i00_src * nb00 + (int64_t)i01_src * nb01);
    };

    ध्रुव भिन्न परिणाम = bicubic(
        bicubic(load(-1, -1), load(0, -1), load(1, -1), load(2, -1), dx, a),
        bicubic(load(-1, 0), load(0, 0), load(1, 0), load(2, 0), dx, a),
        bicubic(load(-1, 1), load(0, 1), load(1, 1), load(2, 1), dx, a),
        bicubic(load(-1, 2), load(0, 2), load(1, 2), load(2, 2), dx, a),
        dy,
        a);

    dst[सूचकांक] = परिणाम;
}

स्थिर शून्य upscale_f32_sycl(ध्रुव भिन्न *   x,
                             भिन्न *         dst,
                             ध्रुव पूर्णाङ्क       nb00,
                             ध्रुव पूर्णाङ्क       nb01,
                             ध्रुव पूर्णाङ्क       nb02,
                             ध्रुव पूर्णाङ्क       nb03,
                             ध्रुव पूर्णाङ्क       ne10,
                             ध्रुव पूर्णाङ्क       ne11,
                             ध्रुव पूर्णाङ्क       ne12,
                             ध्रुव पूर्णाङ्क       ne13,
                             ध्रुव भिन्न     sf0,
                             ध्रुव भिन्न     sf1,
                             ध्रुव भिन्न     sf2,
                             ध्रुव भिन्न     sf3,
                             dpct::queue_ptr stream) {
    ध्रुव int64_t dst_आकार   = ne10 * ne11 * ne12 * ne13;
    ध्रुव int64_t अङ्कः_खण्डानि = (dst_आकार + SYCL_UPSCALE_खण्डम्_आकार - 1) / SYCL_UPSCALE_खण्डम्_आकार;

    stream->parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, अङ्कः_खण्डानि) * sycl::range<3>(1, 1, SYCL_UPSCALE_खण्डम्_आकार),
             sycl::range<3>(1, 1, SYCL_UPSCALE_खण्डम्_आकार)),
        [=](sycl::nd_item<3> /*item_ct1*/) {
            upscale_f32(x, dst, nb00, nb01, nb02, nb03, ne10, ne11, ne12, ne13, sf0, sf1, sf2, sf3);
        });
}

स्थिर शून्य upscale_f32_bilinear_sycl(ध्रुव भिन्न *   x,
                                      भिन्न *         dst,
                                      ध्रुव पूर्णाङ्क       nb00,
                                      ध्रुव पूर्णाङ्क       nb01,
                                      ध्रुव पूर्णाङ्क       nb02,
                                      ध्रुव पूर्णाङ्क       nb03,
                                      ध्रुव पूर्णाङ्क       ne00_src,
                                      ध्रुव पूर्णाङ्क       ne01_src,
                                      ध्रुव पूर्णाङ्क       ne10_dst,
                                      ध्रुव पूर्णाङ्क       ne11_dst,
                                      ध्रुव पूर्णाङ्क       ne12_dst,
                                      ध्रुव पूर्णाङ्क       ne13_dst,
                                      ध्रुव भिन्न     sf0,
                                      ध्रुव भिन्न     sf1,
                                      ध्रुव भिन्न     sf2,
                                      ध्रुव भिन्न     sf3,
                                      ध्रुव भिन्न     pixel_विस्थापनम्,
                                      तार्किक            antialias,
                                      dpct::queue_ptr stream) {
    ध्रुव int64_t dst_आकार   = ne10_dst * ne11_dst * ne12_dst * ne13_dst;
    ध्रुव int64_t अङ्कः_खण्डानि = (dst_आकार + SYCL_UPSCALE_खण्डम्_आकार - 1) / SYCL_UPSCALE_खण्डम्_आकार;

    यदि (antialias) {
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, अङ्कः_खण्डानि) * sycl::range<3>(1, 1, SYCL_UPSCALE_खण्डम्_आकार),
                sycl::range<3>(1, 1, SYCL_UPSCALE_खण्डम्_आकार)),
            [=](sycl::nd_item<3> /*item_ct1*/) {
                upscale_f32_bilinear_antialias(
                    x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst,
                    ne12_dst, ne13_dst, sf0, sf1, sf2, sf3, pixel_विस्थापनम्);
            });
    } अन्यथा {
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, अङ्कः_खण्डानि) * sycl::range<3>(1, 1, SYCL_UPSCALE_खण्डम्_आकार),
                sycl::range<3>(1, 1, SYCL_UPSCALE_खण्डम्_आकार)),
            [=](sycl::nd_item<3> /*item_ct1*/) {
                upscale_f32_bilinear(
                    x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst, ne12_dst,
                    ne13_dst, sf0, sf1, sf2, sf3, pixel_विस्थापनम्);
            });
    }
}

स्थिर शून्य upscale_f32_bicubic_sycl(ध्रुव भिन्न *   x,
                                     भिन्न *         dst,
                                     ध्रुव पूर्णाङ्क       nb00,
                                     ध्रुव पूर्णाङ्क       nb01,
                                     ध्रुव पूर्णाङ्क       nb02,
                                     ध्रुव पूर्णाङ्क       nb03,
                                     ध्रुव पूर्णाङ्क       ne00_src,
                                     ध्रुव पूर्णाङ्क       ne01_src,
                                     ध्रुव पूर्णाङ्क       ne10_dst,
                                     ध्रुव पूर्णाङ्क       ne11_dst,
                                     ध्रुव पूर्णाङ्क       ne12_dst,
                                     ध्रुव पूर्णाङ्क       ne13_dst,
                                     ध्रुव भिन्न     sf0,
                                     ध्रुव भिन्न     sf1,
                                     ध्रुव भिन्न     sf2,
                                     ध्रुव भिन्न     sf3,
                                     ध्रुव भिन्न     pixel_विस्थापनम्,
                                     dpct::queue_ptr stream) {
    ध्रुव int64_t dst_आकार   = ne10_dst * ne11_dst * ne12_dst * ne13_dst;
    ध्रुव int64_t अङ्कः_खण्डानि = (dst_आकार + SYCL_UPSCALE_खण्डम्_आकार - 1) / SYCL_UPSCALE_खण्डम्_आकार;

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, 1, अङ्कः_खण्डानि) * sycl::range<3>(1, 1, SYCL_UPSCALE_खण्डम्_आकार),
                    sycl::range<3>(1, 1, SYCL_UPSCALE_खण्डम्_आकार)),
                [=](sycl::nd_item<3> /*item_ct1*/) {
                    upscale_f32_bicubic(
                        x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst,
                        ne12_dst, ne13_dst, sf0, sf1, sf2, sf3, pixel_विस्थापनम्);
                });
        });
    }
}

स्थिर शून्य ggml_sycl_op_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव भिन्न * src0_d = (ध्रुव भिन्न *)src0->दत्त;
    भिन्न * dst_d = (भिन्न *)dst->दत्त;
    dpct::queue_ptr     stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    ध्रुव पूर्णाङ्क बहुलकम्_flags = dst->op_params[0];
    ध्रुव ggml_scale_बहुलकम् mode = (ggml_scale_बहुलकम्)(बहुलकम्_flags & 0xFF);

    भिन्न sf0 = (भिन्न)dst->ne[0]/src0->ne[0];
    भिन्न sf1 = (भिन्न)dst->ne[1]/src0->ne[1];
    भिन्न sf2 = (भिन्न)dst->ne[2]/src0->ne[2];
    ध्रुव भिन्न sf3 = (भिन्न)dst->ne[3]/src0->ne[3];

    भिन्न pixel_विस्थापनम् = 0.5f;
    यदि (बहुलकम्_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        sf0 = dst->ne[0] > 1 && src0->ne[0] > 1
            ? (भिन्न)(dst->ne[0] - 1) / (src0->ne[0] - 1)
            : sf0;
        sf1 = dst->ne[1] > 1 && src0->ne[1] > 1
            ? (भिन्न)(dst->ne[1] - 1) / (src0->ne[1] - 1)
            : sf1;
        pixel_विस्थापनम् = 0.0f;
    }

    यदि (mode == GGML_SCALE_बहुलकम्_निकटतमः) {
        upscale_f32_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], sf0, sf1, sf2, sf3, stream);
    } अन्यथा यदि (mode == GGML_SCALE_बहुलकम्_BILINEAR) {
        ध्रुव तार्किक antialias = (बहुलकम्_flags & GGML_SCALE_FLAG_ANTIALIAS);
        upscale_f32_bilinear_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src0->ne[0], src0->ne[1], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            sf0, sf1, sf2, sf3, pixel_विस्थापनम्, antialias, stream);
    } अन्यथा यदि (mode == GGML_SCALE_बहुलकम्_BICUBIC) {
        upscale_f32_bicubic_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src0->ne[0], src0->ne[1], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            sf0, sf1, sf2, sf3, pixel_विस्थापनम्, stream);
    }
}

शून्य ggml_sycl_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_upscale(ctx, dst);
}
