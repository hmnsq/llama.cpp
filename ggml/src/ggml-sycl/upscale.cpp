#समावेश "upscale.hpp"

अटल व्योम upscale_f32(स्थिर भग्न * x, भग्न * dst,
        स्थिर पूर्णांक nb00, स्थिर पूर्णांक nb01, स्थिर पूर्णांक nb02, स्थिर पूर्णांक nb03,
        स्थिर पूर्णांक ne10, स्थिर पूर्णांक ne11, स्थिर पूर्णांक ne12, स्थिर पूर्णांक ne13,
        स्थिर भग्न sf0, स्थिर भग्न sf1, स्थिर भग्न sf2, स्थिर भग्न sf3) {
    स्वतः item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    पूर्णांक  सूचकांक    = item_ct1.get_local_id(2) + item_ct1.get_group(2) * item_ct1.get_local_range(2);
    अगर (सूचकांक >= ne10 * ne11 * ne12 * ne13) {
        वापस;
    }

    पूर्णांक i10 = सूचकांक % ne10;
    पूर्णांक i11 = (सूचकांक / ne10) % ne11;
    पूर्णांक i12 = (सूचकांक / (ne10 * ne11)) % ne12;
    पूर्णांक i13 = (सूचकांक / (ne10 * ne11 * ne12)) % ne13;

    पूर्णांक i00 = i10 / sf0;
    पूर्णांक i01 = i11 / sf1;
    पूर्णांक i02 = i12 / sf2;
    पूर्णांक i03 = i13 / sf3;

    dst[सूचकांक] = *((स्थिर भग्न*)((स्थिर अक्षर*)x + i03 * nb03 + i02 * nb02 +
                                  i01 * nb01 + i00 * nb00));
}

अटल व्योम upscale_f32_bilinear(स्थिर भग्न * x, भग्न * dst,
        स्थिर पूर्णांक nb00, स्थिर पूर्णांक nb01, स्थिर पूर्णांक nb02, स्थिर पूर्णांक nb03,
        स्थिर पूर्णांक ne00_src, स्थिर पूर्णांक ne01_src,
        स्थिर पूर्णांक ne10_dst, स्थिर पूर्णांक ne11_dst, स्थिर पूर्णांक ne12_dst, स्थिर पूर्णांक ne13_dst,
        स्थिर भग्न sf0, स्थिर भग्न sf1, स्थिर भग्न sf2, स्थिर भग्न sf3,
        स्थिर भग्न pixel_offset) {
    स्वतः item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    स्थिर int64_t सूचकांक = item_ct1.get_local_id(2) +
        item_ct1.get_group(2) * item_ct1.get_local_range(2);
    स्थिर int64_t dst_योग_elements = ne10_dst * ne11_dst * ne12_dst * ne13_dst;

  अगर (सूचकांक >= dst_योग_elements) {
    वापस;
  }

    स्थिर पूर्णांक i10_dst = सूचकांक % ne10_dst;
    स्थिर पूर्णांक i11_dst = (सूचकांक / ne10_dst) % ne11_dst;
    स्थिर पूर्णांक i12_dst = (सूचकांक / (ne10_dst * ne11_dst)) % ne12_dst;
    स्थिर पूर्णांक i13_dst = सूचकांक / (ne10_dst * ne11_dst * ne12_dst);

    स्थिर पूर्णांक i02_src = (पूर्णांक)(i12_dst / sf2);
    स्थिर पूर्णांक i03_src = (पूर्णांक)(i13_dst / sf3);

    स्थिर भग्न y_src_f = ((भग्न)i11_dst + pixel_offset) / sf1 - pixel_offset;
    पूर्णांक         y0_src    = (पूर्णांक) sycl::न्यूनमान((भग्न) y_src_f);
    पूर्णांक y1_src    = y0_src + 1;

    y0_src = sycl::max(0, sycl::min(y0_src, ne01_src - 1));
    y1_src = sycl::max(0, sycl::min(y1_src, ne01_src - 1));

    भग्न dy = y_src_f - (भग्न)y0_src;
    dy       = sycl::max(0.0f, sycl::min(dy, 1.0f));

    भग्न x_src_f = ((भग्न)i10_dst + pixel_offset) / sf0 - pixel_offset;
    पूर्णांक   x0_src    = (पूर्णांक) sycl::न्यूनमान(x_src_f);
    पूर्णांक x1_src    = x0_src + 1;

    x0_src = sycl::max(0, sycl::min(x0_src, ne00_src - 1));
    x1_src = sycl::max(0, sycl::min(x1_src, ne00_src - 1));

    भग्न dx = x_src_f - (भग्न)x0_src;
    dx       = sycl::max(0.0f, sycl::min(dx, 1.0f));

    स्थिर भग्न* p_a =
        (स्थिर भग्न*)((स्थिर अक्षर*)x + (int64_t)x0_src * nb00 +
                       (int64_t)y0_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    स्थिर भग्न* p_b =
        (स्थिर भग्न*)((स्थिर अक्षर*)x + (int64_t)x1_src * nb00 +
                       (int64_t)y0_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    स्थिर भग्न* p_c =
        (स्थिर भग्न*)((स्थिर अक्षर*)x + (int64_t)x0_src * nb00 +
                       (int64_t)y1_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    स्थिर भग्न* p_d =
        (स्थिर भग्न*)((स्थिर अक्षर*)x + (int64_t)x1_src * nb00 +
                       (int64_t)y1_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);

    स्थिर भग्न val_a = *p_a;
    स्थिर भग्न val_b = *p_b;
    स्थिर भग्न val_c = *p_c;
    स्थिर भग्न val_d = *p_d;

    भग्न परिणाम = val_a * (1.0f - dx) * (1.0f - dy) +
                   val_b * dx * (1.0f - dy) +
                   val_c * (1.0f - dx) * dy +
                   val_d * dx * dy;

    dst[सूचकांक] = परिणाम;
}

// Similar to F.interpolate(..., mode="bilinear", align_corners=False, antialias=True)
// https://github.com/pytorch/pytorch/blob/8871ff29b743948d1225389d5b7068f37b22750b/aten/src/ATen/native/cpu/UpSampleKernel.cpp
अटल व्योम upscale_f32_bilinear_antialias(स्थिर भग्न * src0,
                                           भग्न *       dst,
                                           स्थिर पूर्णांक     nb00,
                                           स्थिर पूर्णांक     nb01,
                                           स्थिर पूर्णांक     nb02,
                                           स्थिर पूर्णांक     nb03,
                                           स्थिर पूर्णांक     ne00_src,
                                           स्थिर पूर्णांक     ne01_src,
                                           स्थिर पूर्णांक     ne10_dst,
                                           स्थिर पूर्णांक     ne11_dst,
                                           स्थिर पूर्णांक     ne12_dst,
                                           स्थिर पूर्णांक     ne13_dst,
                                           स्थिर भग्न   sf0,
                                           स्थिर भग्न   sf1,
                                           स्थिर भग्न   sf2,
                                           स्थिर भग्न   sf3,
                                           स्थिर भग्न   pixel_offset) {
    स्वतः item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    स्थिर int64_t सूचकांक = item_ct1.get_local_id(2) +
        item_ct1.get_group(2) * item_ct1.get_local_range(2);
    स्थिर int64_t dst_योग_elements = ne10_dst * ne11_dst * ne12_dst * ne13_dst;

    अगर (सूचकांक >= dst_योग_elements) {
        वापस;
    }

    स्थिर पूर्णांक i10_dst = सूचकांक % ne10_dst;
    स्थिर पूर्णांक i11_dst = (सूचकांक / ne10_dst) % ne11_dst;
    स्थिर पूर्णांक i12_dst = (सूचकांक / (ne10_dst * ne11_dst)) % ne12_dst;
    स्थिर पूर्णांक i13_dst = सूचकांक / (ne10_dst * ne11_dst * ne12_dst);

    स्थिर पूर्णांक i02_src = (पूर्णांक)(i12_dst / sf2);
    स्थिर पूर्णांक i03_src = (पूर्णांक)(i13_dst / sf3);

    स्थिर भग्न y = ((भग्न)i11_dst + pixel_offset) / sf1;
    स्थिर भग्न x = ((भग्न)i10_dst + pixel_offset) / sf0;

    // support and invscale, minimum 1 pixel for bilinear
    स्थिर भग्न support1  = sycl::max(1.0f / sf1, 1.0f);
    स्थिर भग्न invscale1 = 1.0f / support1;
    स्थिर भग्न support0  = sycl::max(1.0f / sf0, 1.0f);
    स्थिर भग्न invscale0 = 1.0f / support0;

    // the range of source pixels that contribute
    स्थिर int64_t x_min = sycl::max(int64_t(0), int64_t(x - support0 + pixel_offset));
    स्थिर int64_t x_max = sycl::min(int64_t(ne00_src), int64_t(x + support0 + pixel_offset));
    स्थिर int64_t y_min = sycl::max(int64_t(0), int64_t(y - support1 + pixel_offset));
    स्थिर int64_t y_max = sycl::min(int64_t(ne01_src), int64_t(y + support1 + pixel_offset));

    // bilinear filter with antialiasing
    भग्न val = 0.0f;
    भग्न योग_weight = 0.0f;

    स्वतः triangle_filter = [](भग्न x) -> भग्न {
        वापस sycl::max(1.0f - sycl::भ_असल(x), 0.0f);
    };

    क्रम (int64_t sy = y_min; sy < y_max; sy++) {
        स्थिर भग्न weight_y = triangle_filter((sy - y + pixel_offset) * invscale1);

        क्रम (int64_t sx = x_min; sx < x_max; sx++) {
            स्थिर भग्न weight_x = triangle_filter((sx - x + pixel_offset) * invscale0);
            स्थिर भग्न weight = weight_x * weight_y;

            अगर (weight <= 0.0f) {
                जारी;
            }

            स्थिर भग्न pixel =
                *(स्थिर भग्न*)((स्थिर अक्षर*)src0 + sx * nb00 + sy * nb01 +
                                i02_src * nb02 + i03_src * nb03);
            val += pixel * weight;
            योग_weight += weight;
        }
    }

    अगर (योग_weight > 0.0f) {
        val /= योग_weight;
    }

    dst[सूचकांक] = val;
}

namespace bicubic_interpolation {
अटल भग्न weight1(भग्न x, स्थिर भग्न &a) { वापस ((a + 2) * x - (a + 3)) * x * x + 1; }
अटल भग्न weight2(भग्न x, स्थिर भग्न &a) { वापस ((a * x - 5 * a) * x + 8 * a) * x - 4 * a; }

अटल भग्न bicubic(भग्न p0, भग्न p1, भग्न p2, भग्न p3, भग्न x, भग्न a) {
    स्थिर भग्न w0 = weight2(x + 1, a);
    स्थिर भग्न w1 = weight1(x + 0, a);
    स्थिर भग्न w2 = weight1(1 - x, a);
    स्थिर भग्न w3 = weight2(2 - x, a);
    वापस p0 * w0 + p1 * w1 + p2 * w2 + p3 * w3;
}

}

अटल व्योम upscale_f32_bicubic(स्थिर भग्न * x, भग्न * dst,
        स्थिर पूर्णांक nb00, स्थिर पूर्णांक nb01, स्थिर पूर्णांक nb02, स्थिर पूर्णांक nb03,
        स्थिर पूर्णांक ne00_src, स्थिर पूर्णांक ne01_src,
        स्थिर पूर्णांक ne10_dst, स्थिर पूर्णांक ne11_dst, स्थिर पूर्णांक ne12_dst, स्थिर पूर्णांक ne13_dst,
        स्थिर भग्न sf0, स्थिर भग्न sf1, स्थिर भग्न sf2, स्थिर भग्न sf3,
        स्थिर भग्न pixel_offset) {
    स्वतः item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    स्थिर भग्न a = -0.75f;
    using bicubic_interpolation::bicubic;

    स्थिर int64_t सूचकांक = item_ct1.get_local_id(2) +
        item_ct1.get_group(2) * item_ct1.get_local_range(2);
    स्थिर int64_t dst_योग_elements =
        ne10_dst * ne11_dst * ne12_dst * ne13_dst;

    अगर (सूचकांक >= dst_योग_elements) {
        वापस;
    }

    स्थिर पूर्णांक i10_dst = सूचकांक % ne10_dst;
    स्थिर पूर्णांक i11_dst = (सूचकांक / ne10_dst) % ne11_dst;
    स्थिर पूर्णांक i12_dst = (सूचकांक / (ne10_dst * ne11_dst)) % ne12_dst;
    स्थिर पूर्णांक i13_dst = सूचकांक / (ne10_dst * ne11_dst * ne12_dst);

    स्थिर पूर्णांक i02_src = (पूर्णांक)(i12_dst / sf2);
    स्थिर पूर्णांक i03_src = (पूर्णांक)(i13_dst / sf3);

    स्थिर भग्न y_src_f = ((भग्न)i11_dst + pixel_offset) / sf1 - pixel_offset;
    स्थिर पूर्णांक   y0_src  = (पूर्णांक) sycl::न्यूनमान((भग्न) y_src_f);
    स्थिर भग्न dy      = y_src_f - (भग्न)y0_src;

    स्थिर भग्न x_src_f = ((भग्न)i10_dst + pixel_offset) / sf0 - pixel_offset;
    स्थिर पूर्णांक   x0_src  = (पूर्णांक) sycl::न्यूनमान((भग्न) x_src_f);
    स्थिर भग्न dx      = x_src_f - (भग्न)x0_src;

    स्थिर अक्षर * x_base = (स्थिर अक्षर *)x + (int64_t)i02_src * nb02 + (int64_t)i03_src * nb03;

    स्वतः load = [=](पूर्णांक x_off, पूर्णांक y_off) -> भग्न {
        पूर्णांक i00_src = sycl::max(0, sycl::min(x0_src + x_off, ne00_src - 1));
        पूर्णांक i01_src = sycl::max(0, sycl::min(y0_src + y_off, ne01_src - 1));
        वापस *(स्थिर भग्न *)(x_base + (int64_t)i00_src * nb00 + (int64_t)i01_src * nb01);
    };

    स्थिर भग्न परिणाम = bicubic(
        bicubic(load(-1, -1), load(0, -1), load(1, -1), load(2, -1), dx, a),
        bicubic(load(-1, 0), load(0, 0), load(1, 0), load(2, 0), dx, a),
        bicubic(load(-1, 1), load(0, 1), load(1, 1), load(2, 1), dx, a),
        bicubic(load(-1, 2), load(0, 2), load(1, 2), load(2, 2), dx, a),
        dy,
        a);

    dst[सूचकांक] = परिणाम;
}

अटल व्योम upscale_f32_sycl(स्थिर भग्न *   x,
                             भग्न *         dst,
                             स्थिर पूर्णांक       nb00,
                             स्थिर पूर्णांक       nb01,
                             स्थिर पूर्णांक       nb02,
                             स्थिर पूर्णांक       nb03,
                             स्थिर पूर्णांक       ne10,
                             स्थिर पूर्णांक       ne11,
                             स्थिर पूर्णांक       ne12,
                             स्थिर पूर्णांक       ne13,
                             स्थिर भग्न     sf0,
                             स्थिर भग्न     sf1,
                             स्थिर भग्न     sf2,
                             स्थिर भग्न     sf3,
                             dpct::queue_ptr stream) {
    स्थिर int64_t dst_आकार   = ne10 * ne11 * ne12 * ne13;
    स्थिर int64_t अंक_blocks = (dst_आकार + SYCL_UPSCALE_BLOCK_आकार - 1) / SYCL_UPSCALE_BLOCK_आकार;

    stream->parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_आकार),
             sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_आकार)),
        [=](sycl::nd_item<3> /*item_ct1*/) {
            upscale_f32(x, dst, nb00, nb01, nb02, nb03, ne10, ne11, ne12, ne13, sf0, sf1, sf2, sf3);
        });
}

अटल व्योम upscale_f32_bilinear_sycl(स्थिर भग्न *   x,
                                      भग्न *         dst,
                                      स्थिर पूर्णांक       nb00,
                                      स्थिर पूर्णांक       nb01,
                                      स्थिर पूर्णांक       nb02,
                                      स्थिर पूर्णांक       nb03,
                                      स्थिर पूर्णांक       ne00_src,
                                      स्थिर पूर्णांक       ne01_src,
                                      स्थिर पूर्णांक       ne10_dst,
                                      स्थिर पूर्णांक       ne11_dst,
                                      स्थिर पूर्णांक       ne12_dst,
                                      स्थिर पूर्णांक       ne13_dst,
                                      स्थिर भग्न     sf0,
                                      स्थिर भग्न     sf1,
                                      स्थिर भग्न     sf2,
                                      स्थिर भग्न     sf3,
                                      स्थिर भग्न     pixel_offset,
                                      bool            antialias,
                                      dpct::queue_ptr stream) {
    स्थिर int64_t dst_आकार   = ne10_dst * ne11_dst * ne12_dst * ne13_dst;
    स्थिर int64_t अंक_blocks = (dst_आकार + SYCL_UPSCALE_BLOCK_आकार - 1) / SYCL_UPSCALE_BLOCK_आकार;

    अगर (antialias) {
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_आकार),
                sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_आकार)),
            [=](sycl::nd_item<3> /*item_ct1*/) {
                upscale_f32_bilinear_antialias(
                    x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst,
                    ne12_dst, ne13_dst, sf0, sf1, sf2, sf3, pixel_offset);
            });
    } अन्यथा {
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_आकार),
                sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_आकार)),
            [=](sycl::nd_item<3> /*item_ct1*/) {
                upscale_f32_bilinear(
                    x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst, ne12_dst,
                    ne13_dst, sf0, sf1, sf2, sf3, pixel_offset);
            });
    }
}

अटल व्योम upscale_f32_bicubic_sycl(स्थिर भग्न *   x,
                                     भग्न *         dst,
                                     स्थिर पूर्णांक       nb00,
                                     स्थिर पूर्णांक       nb01,
                                     स्थिर पूर्णांक       nb02,
                                     स्थिर पूर्णांक       nb03,
                                     स्थिर पूर्णांक       ne00_src,
                                     स्थिर पूर्णांक       ne01_src,
                                     स्थिर पूर्णांक       ne10_dst,
                                     स्थिर पूर्णांक       ne11_dst,
                                     स्थिर पूर्णांक       ne12_dst,
                                     स्थिर पूर्णांक       ne13_dst,
                                     स्थिर भग्न     sf0,
                                     स्थिर भग्न     sf1,
                                     स्थिर भग्न     sf2,
                                     स्थिर भग्न     sf3,
                                     स्थिर भग्न     pixel_offset,
                                     dpct::queue_ptr stream) {
    स्थिर int64_t dst_आकार   = ne10_dst * ne11_dst * ne12_dst * ne13_dst;
    स्थिर int64_t अंक_blocks = (dst_आकार + SYCL_UPSCALE_BLOCK_आकार - 1) / SYCL_UPSCALE_BLOCK_आकार;

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_आकार),
                    sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_आकार)),
                [=](sycl::nd_item<3> /*item_ct1*/) {
                    upscale_f32_bicubic(
                        x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst,
                        ne12_dst, ne13_dst, sf0, sf1, sf2, sf3, pixel_offset);
                });
        });
    }
}

अटल व्योम ggml_sycl_op_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर भग्न * src0_d = (स्थिर भग्न *)src0->तथ्य;
    भग्न * dst_d = (भग्न *)dst->तथ्य;
    dpct::queue_ptr     stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    स्थिर पूर्णांक mode_flags = dst->op_params[0];
    स्थिर ggml_scale_mode mode = (ggml_scale_mode)(mode_flags & 0xFF);

    भग्न sf0 = (भग्न)dst->ne[0]/src0->ne[0];
    भग्न sf1 = (भग्न)dst->ne[1]/src0->ne[1];
    भग्न sf2 = (भग्न)dst->ne[2]/src0->ne[2];
    स्थिर भग्न sf3 = (भग्न)dst->ne[3]/src0->ne[3];

    भग्न pixel_offset = 0.5f;
    अगर (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        sf0 = dst->ne[0] > 1 && src0->ne[0] > 1
            ? (भग्न)(dst->ne[0] - 1) / (src0->ne[0] - 1)
            : sf0;
        sf1 = dst->ne[1] > 1 && src0->ne[1] > 1
            ? (भग्न)(dst->ne[1] - 1) / (src0->ne[1] - 1)
            : sf1;
        pixel_offset = 0.0f;
    }

    अगर (mode == GGML_SCALE_MODE_NEAREST) {
        upscale_f32_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], sf0, sf1, sf2, sf3, stream);
    } अन्यथा अगर (mode == GGML_SCALE_MODE_BILINEAR) {
        स्थिर bool antialias = (mode_flags & GGML_SCALE_FLAG_ANTIALIAS);
        upscale_f32_bilinear_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src0->ne[0], src0->ne[1], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            sf0, sf1, sf2, sf3, pixel_offset, antialias, stream);
    } अन्यथा अगर (mode == GGML_SCALE_MODE_BICUBIC) {
        upscale_f32_bicubic_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src0->ne[0], src0->ne[1], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            sf0, sf1, sf2, sf3, pixel_offset, stream);
    }
}

व्योम ggml_sycl_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_upscale(ctx, dst);
}
