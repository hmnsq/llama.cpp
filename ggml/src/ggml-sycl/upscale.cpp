#अन्तर्भाव "upscale.hpp"

स्थिर शून्य upscale_f32(अचर भिन्न * x, भिन्न * dst,
        अचर पूर्णाङ्क nb00, अचर पूर्णाङ्क nb01, अचर पूर्णाङ्क nb02, अचर पूर्णाङ्क nb03,
        अचर पूर्णाङ्क ne10, अचर पूर्णाङ्क ne11, अचर पूर्णाङ्क ne12, अचर पूर्णाङ्क ne13,
        अचर भिन्न sf0, अचर भिन्न sf1, अचर भिन्न sf2, अचर भिन्न sf3) {
    स्वतः item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    पूर्णाङ्क  index    = item_ct1.get_local_id(2) + item_ct1.get_group(2) * item_ct1.get_local_range(2);
    यदि (index >= ne10 * ne11 * ne12 * ne13) {
        फर्काऊ;
    }

    पूर्णाङ्क i10 = index % ne10;
    पूर्णाङ्क i11 = (index / ne10) % ne11;
    पूर्णाङ्क i12 = (index / (ne10 * ne11)) % ne12;
    पूर्णाङ्क i13 = (index / (ne10 * ne11 * ne12)) % ne13;

    पूर्णाङ्क i00 = i10 / sf0;
    पूर्णाङ्क i01 = i11 / sf1;
    पूर्णाङ्क i02 = i12 / sf2;
    पूर्णाङ्क i03 = i13 / sf3;

    dst[index] = *((अचर भिन्न*)((अचर वर्ण*)x + i03 * nb03 + i02 * nb02 +
                                  i01 * nb01 + i00 * nb00));
}

स्थिर शून्य upscale_f32_bilinear(अचर भिन्न * x, भिन्न * dst,
        अचर पूर्णाङ्क nb00, अचर पूर्णाङ्क nb01, अचर पूर्णाङ्क nb02, अचर पूर्णाङ्क nb03,
        अचर पूर्णाङ्क ne00_src, अचर पूर्णाङ्क ne01_src,
        अचर पूर्णाङ्क ne10_dst, अचर पूर्णाङ्क ne11_dst, अचर पूर्णाङ्क ne12_dst, अचर पूर्णाङ्क ne13_dst,
        अचर भिन्न sf0, अचर भिन्न sf1, अचर भिन्न sf2, अचर भिन्न sf3,
        अचर भिन्न pixel_offset) {
    स्वतः item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    अचर int64_t index = item_ct1.get_local_id(2) +
        item_ct1.get_group(2) * item_ct1.get_local_range(2);
    अचर int64_t dst_total_elements = ne10_dst * ne11_dst * ne12_dst * ne13_dst;

  यदि (index >= dst_total_elements) {
    फर्काऊ;
  }

    अचर पूर्णाङ्क i10_dst = index % ne10_dst;
    अचर पूर्णाङ्क i11_dst = (index / ne10_dst) % ne11_dst;
    अचर पूर्णाङ्क i12_dst = (index / (ne10_dst * ne11_dst)) % ne12_dst;
    अचर पूर्णाङ्क i13_dst = index / (ne10_dst * ne11_dst * ne12_dst);

    अचर पूर्णाङ्क i02_src = (पूर्णाङ्क)(i12_dst / sf2);
    अचर पूर्णाङ्क i03_src = (पूर्णाङ्क)(i13_dst / sf3);

    अचर भिन्न y_src_f = ((भिन्न)i11_dst + pixel_offset) / sf1 - pixel_offset;
    पूर्णाङ्क         y0_src    = (पूर्णाङ्क) sycl::न्यूनमान((भिन्न) y_src_f);
    पूर्णाङ्क y1_src    = y0_src + 1;

    y0_src = sycl::max(0, sycl::min(y0_src, ne01_src - 1));
    y1_src = sycl::max(0, sycl::min(y1_src, ne01_src - 1));

    भिन्न dy = y_src_f - (भिन्न)y0_src;
    dy       = sycl::max(0.0f, sycl::min(dy, 1.0f));

    भिन्न x_src_f = ((भिन्न)i10_dst + pixel_offset) / sf0 - pixel_offset;
    पूर्णाङ्क   x0_src    = (पूर्णाङ्क) sycl::न्यूनमान(x_src_f);
    पूर्णाङ्क x1_src    = x0_src + 1;

    x0_src = sycl::max(0, sycl::min(x0_src, ne00_src - 1));
    x1_src = sycl::max(0, sycl::min(x1_src, ne00_src - 1));

    भिन्न dx = x_src_f - (भिन्न)x0_src;
    dx       = sycl::max(0.0f, sycl::min(dx, 1.0f));

    अचर भिन्न* p_a =
        (अचर भिन्न*)((अचर वर्ण*)x + (int64_t)x0_src * nb00 +
                       (int64_t)y0_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    अचर भिन्न* p_b =
        (अचर भिन्न*)((अचर वर्ण*)x + (int64_t)x1_src * nb00 +
                       (int64_t)y0_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    अचर भिन्न* p_c =
        (अचर भिन्न*)((अचर वर्ण*)x + (int64_t)x0_src * nb00 +
                       (int64_t)y1_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    अचर भिन्न* p_d =
        (अचर भिन्न*)((अचर वर्ण*)x + (int64_t)x1_src * nb00 +
                       (int64_t)y1_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);

    अचर भिन्न val_a = *p_a;
    अचर भिन्न val_b = *p_b;
    अचर भिन्न val_c = *p_c;
    अचर भिन्न val_d = *p_d;

    भिन्न result = val_a * (1.0f - dx) * (1.0f - dy) +
                   val_b * dx * (1.0f - dy) +
                   val_c * (1.0f - dx) * dy +
                   val_d * dx * dy;

    dst[index] = result;
}

// Similar to F.interpolate(..., mode="bilinear", align_corners=False, antialias=True)
// https://github.com/pytorch/pytorch/blob/8871ff29b743948d1225389d5b7068f37b22750b/aten/src/ATen/native/cpu/UpSampleKernel.cpp
स्थिर शून्य upscale_f32_bilinear_antialias(अचर भिन्न * src0,
                                           भिन्न *       dst,
                                           अचर पूर्णाङ्क     nb00,
                                           अचर पूर्णाङ्क     nb01,
                                           अचर पूर्णाङ्क     nb02,
                                           अचर पूर्णाङ्क     nb03,
                                           अचर पूर्णाङ्क     ne00_src,
                                           अचर पूर्णाङ्क     ne01_src,
                                           अचर पूर्णाङ्क     ne10_dst,
                                           अचर पूर्णाङ्क     ne11_dst,
                                           अचर पूर्णाङ्क     ne12_dst,
                                           अचर पूर्णाङ्क     ne13_dst,
                                           अचर भिन्न   sf0,
                                           अचर भिन्न   sf1,
                                           अचर भिन्न   sf2,
                                           अचर भिन्न   sf3,
                                           अचर भिन्न   pixel_offset) {
    स्वतः item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    अचर int64_t index = item_ct1.get_local_id(2) +
        item_ct1.get_group(2) * item_ct1.get_local_range(2);
    अचर int64_t dst_total_elements = ne10_dst * ne11_dst * ne12_dst * ne13_dst;

    यदि (index >= dst_total_elements) {
        फर्काऊ;
    }

    अचर पूर्णाङ्क i10_dst = index % ne10_dst;
    अचर पूर्णाङ्क i11_dst = (index / ne10_dst) % ne11_dst;
    अचर पूर्णाङ्क i12_dst = (index / (ne10_dst * ne11_dst)) % ne12_dst;
    अचर पूर्णाङ्क i13_dst = index / (ne10_dst * ne11_dst * ne12_dst);

    अचर पूर्णाङ्क i02_src = (पूर्णाङ्क)(i12_dst / sf2);
    अचर पूर्णाङ्क i03_src = (पूर्णाङ्क)(i13_dst / sf3);

    अचर भिन्न y = ((भिन्न)i11_dst + pixel_offset) / sf1;
    अचर भिन्न x = ((भिन्न)i10_dst + pixel_offset) / sf0;

    // support and invscale, minimum 1 pixel for bilinear
    अचर भिन्न support1  = sycl::max(1.0f / sf1, 1.0f);
    अचर भिन्न invscale1 = 1.0f / support1;
    अचर भिन्न support0  = sycl::max(1.0f / sf0, 1.0f);
    अचर भिन्न invscale0 = 1.0f / support0;

    // the range of source pixels that contribute
    अचर int64_t x_min = sycl::max(int64_t(0), int64_t(x - support0 + pixel_offset));
    अचर int64_t x_max = sycl::min(int64_t(ne00_src), int64_t(x + support0 + pixel_offset));
    अचर int64_t y_min = sycl::max(int64_t(0), int64_t(y - support1 + pixel_offset));
    अचर int64_t y_max = sycl::min(int64_t(ne01_src), int64_t(y + support1 + pixel_offset));

    // bilinear filter with antialiasing
    भिन्न val = 0.0f;
    भिन्न total_weight = 0.0f;

    स्वतः triangle_filter = [](भिन्न x) -> भिन्न {
        फर्काऊ sycl::max(1.0f - sycl::भ_असल(x), 0.0f);
    };

    लागि (int64_t sy = y_min; sy < y_max; sy++) {
        अचर भिन्न weight_y = triangle_filter((sy - y + pixel_offset) * invscale1);

        लागि (int64_t sx = x_min; sx < x_max; sx++) {
            अचर भिन्न weight_x = triangle_filter((sx - x + pixel_offset) * invscale0);
            अचर भिन्न weight = weight_x * weight_y;

            यदि (weight <= 0.0f) {
                जारी;
            }

            अचर भिन्न pixel =
                *(अचर भिन्न*)((अचर वर्ण*)src0 + sx * nb00 + sy * nb01 +
                                i02_src * nb02 + i03_src * nb03);
            val += pixel * weight;
            total_weight += weight;
        }
    }

    यदि (total_weight > 0.0f) {
        val /= total_weight;
    }

    dst[index] = val;
}

नामाकाश bicubic_interpolation {
स्थिर भिन्न weight1(भिन्न x, अचर भिन्न &a) { फर्काऊ ((a + 2) * x - (a + 3)) * x * x + 1; }
स्थिर भिन्न weight2(भिन्न x, अचर भिन्न &a) { फर्काऊ ((a * x - 5 * a) * x + 8 * a) * x - 4 * a; }

स्थिर भिन्न bicubic(भिन्न p0, भिन्न p1, भिन्न p2, भिन्न p3, भिन्न x, भिन्न a) {
    अचर भिन्न w0 = weight2(x + 1, a);
    अचर भिन्न w1 = weight1(x + 0, a);
    अचर भिन्न w2 = weight1(1 - x, a);
    अचर भिन्न w3 = weight2(2 - x, a);
    फर्काऊ p0 * w0 + p1 * w1 + p2 * w2 + p3 * w3;
}

}

स्थिर शून्य upscale_f32_bicubic(अचर भिन्न * x, भिन्न * dst,
        अचर पूर्णाङ्क nb00, अचर पूर्णाङ्क nb01, अचर पूर्णाङ्क nb02, अचर पूर्णाङ्क nb03,
        अचर पूर्णाङ्क ne00_src, अचर पूर्णाङ्क ne01_src,
        अचर पूर्णाङ्क ne10_dst, अचर पूर्णाङ्क ne11_dst, अचर पूर्णाङ्क ne12_dst, अचर पूर्णाङ्क ne13_dst,
        अचर भिन्न sf0, अचर भिन्न sf1, अचर भिन्न sf2, अचर भिन्न sf3,
        अचर भिन्न pixel_offset) {
    स्वतः item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    अचर भिन्न a = -0.75f;
    using bicubic_interpolation::bicubic;

    अचर int64_t index = item_ct1.get_local_id(2) +
        item_ct1.get_group(2) * item_ct1.get_local_range(2);
    अचर int64_t dst_total_elements =
        ne10_dst * ne11_dst * ne12_dst * ne13_dst;

    यदि (index >= dst_total_elements) {
        फर्काऊ;
    }

    अचर पूर्णाङ्क i10_dst = index % ne10_dst;
    अचर पूर्णाङ्क i11_dst = (index / ne10_dst) % ne11_dst;
    अचर पूर्णाङ्क i12_dst = (index / (ne10_dst * ne11_dst)) % ne12_dst;
    अचर पूर्णाङ्क i13_dst = index / (ne10_dst * ne11_dst * ne12_dst);

    अचर पूर्णाङ्क i02_src = (पूर्णाङ्क)(i12_dst / sf2);
    अचर पूर्णाङ्क i03_src = (पूर्णाङ्क)(i13_dst / sf3);

    अचर भिन्न y_src_f = ((भिन्न)i11_dst + pixel_offset) / sf1 - pixel_offset;
    अचर पूर्णाङ्क   y0_src  = (पूर्णाङ्क) sycl::न्यूनमान((भिन्न) y_src_f);
    अचर भिन्न dy      = y_src_f - (भिन्न)y0_src;

    अचर भिन्न x_src_f = ((भिन्न)i10_dst + pixel_offset) / sf0 - pixel_offset;
    अचर पूर्णाङ्क   x0_src  = (पूर्णाङ्क) sycl::न्यूनमान((भिन्न) x_src_f);
    अचर भिन्न dx      = x_src_f - (भिन्न)x0_src;

    अचर वर्ण * x_base = (अचर वर्ण *)x + (int64_t)i02_src * nb02 + (int64_t)i03_src * nb03;

    स्वतः load = [=](पूर्णाङ्क x_off, पूर्णाङ्क y_off) -> भिन्न {
        पूर्णाङ्क i00_src = sycl::max(0, sycl::min(x0_src + x_off, ne00_src - 1));
        पूर्णाङ्क i01_src = sycl::max(0, sycl::min(y0_src + y_off, ne01_src - 1));
        फर्काऊ *(अचर भिन्न *)(x_base + (int64_t)i00_src * nb00 + (int64_t)i01_src * nb01);
    };

    अचर भिन्न result = bicubic(
        bicubic(load(-1, -1), load(0, -1), load(1, -1), load(2, -1), dx, a),
        bicubic(load(-1, 0), load(0, 0), load(1, 0), load(2, 0), dx, a),
        bicubic(load(-1, 1), load(0, 1), load(1, 1), load(2, 1), dx, a),
        bicubic(load(-1, 2), load(0, 2), load(1, 2), load(2, 2), dx, a),
        dy,
        a);

    dst[index] = result;
}

स्थिर शून्य upscale_f32_sycl(अचर भिन्न *   x,
                             भिन्न *         dst,
                             अचर पूर्णाङ्क       nb00,
                             अचर पूर्णाङ्क       nb01,
                             अचर पूर्णाङ्क       nb02,
                             अचर पूर्णाङ्क       nb03,
                             अचर पूर्णाङ्क       ne10,
                             अचर पूर्णाङ्क       ne11,
                             अचर पूर्णाङ्क       ne12,
                             अचर पूर्णाङ्क       ne13,
                             अचर भिन्न     sf0,
                             अचर भिन्न     sf1,
                             अचर भिन्न     sf2,
                             अचर भिन्न     sf3,
                             dpct::queue_ptr stream) {
    अचर int64_t dst_size   = ne10 * ne11 * ne12 * ne13;
    अचर int64_t अङ्क_blocks = (dst_size + SYCL_UPSCALE_BLOCK_SIZE - 1) / SYCL_UPSCALE_BLOCK_SIZE;

    stream->parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, अङ्क_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE),
             sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE)),
        [=](sycl::nd_item<3> /*item_ct1*/) {
            upscale_f32(x, dst, nb00, nb01, nb02, nb03, ne10, ne11, ne12, ne13, sf0, sf1, sf2, sf3);
        });
}

स्थिर शून्य upscale_f32_bilinear_sycl(अचर भिन्न *   x,
                                      भिन्न *         dst,
                                      अचर पूर्णाङ्क       nb00,
                                      अचर पूर्णाङ्क       nb01,
                                      अचर पूर्णाङ्क       nb02,
                                      अचर पूर्णाङ्क       nb03,
                                      अचर पूर्णाङ्क       ne00_src,
                                      अचर पूर्णाङ्क       ne01_src,
                                      अचर पूर्णाङ्क       ne10_dst,
                                      अचर पूर्णाङ्क       ne11_dst,
                                      अचर पूर्णाङ्क       ne12_dst,
                                      अचर पूर्णाङ्क       ne13_dst,
                                      अचर भिन्न     sf0,
                                      अचर भिन्न     sf1,
                                      अचर भिन्न     sf2,
                                      अचर भिन्न     sf3,
                                      अचर भिन्न     pixel_offset,
                                      तार्किक            antialias,
                                      dpct::queue_ptr stream) {
    अचर int64_t dst_size   = ne10_dst * ne11_dst * ne12_dst * ne13_dst;
    अचर int64_t अङ्क_blocks = (dst_size + SYCL_UPSCALE_BLOCK_SIZE - 1) / SYCL_UPSCALE_BLOCK_SIZE;

    यदि (antialias) {
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, अङ्क_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE),
                sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE)),
            [=](sycl::nd_item<3> /*item_ct1*/) {
                upscale_f32_bilinear_antialias(
                    x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst,
                    ne12_dst, ne13_dst, sf0, sf1, sf2, sf3, pixel_offset);
            });
    } अन्यथा {
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, अङ्क_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE),
                sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE)),
            [=](sycl::nd_item<3> /*item_ct1*/) {
                upscale_f32_bilinear(
                    x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst, ne12_dst,
                    ne13_dst, sf0, sf1, sf2, sf3, pixel_offset);
            });
    }
}

स्थिर शून्य upscale_f32_bicubic_sycl(अचर भिन्न *   x,
                                     भिन्न *         dst,
                                     अचर पूर्णाङ्क       nb00,
                                     अचर पूर्णाङ्क       nb01,
                                     अचर पूर्णाङ्क       nb02,
                                     अचर पूर्णाङ्क       nb03,
                                     अचर पूर्णाङ्क       ne00_src,
                                     अचर पूर्णाङ्क       ne01_src,
                                     अचर पूर्णाङ्क       ne10_dst,
                                     अचर पूर्णाङ्क       ne11_dst,
                                     अचर पूर्णाङ्क       ne12_dst,
                                     अचर पूर्णाङ्क       ne13_dst,
                                     अचर भिन्न     sf0,
                                     अचर भिन्न     sf1,
                                     अचर भिन्न     sf2,
                                     अचर भिन्न     sf3,
                                     अचर भिन्न     pixel_offset,
                                     dpct::queue_ptr stream) {
    अचर int64_t dst_size   = ne10_dst * ne11_dst * ne12_dst * ne13_dst;
    अचर int64_t अङ्क_blocks = (dst_size + SYCL_UPSCALE_BLOCK_SIZE - 1) / SYCL_UPSCALE_BLOCK_SIZE;

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, 1, अङ्क_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE),
                    sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE)),
                [=](sycl::nd_item<3> /*item_ct1*/) {
                    upscale_f32_bicubic(
                        x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst,
                        ne12_dst, ne13_dst, sf0, sf1, sf2, sf3, pixel_offset);
                });
        });
    }
}

स्थिर शून्य ggml_sycl_op_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    अचर ggml_tensor * src0 = dst->src[0];
    अचर भिन्न * src0_d = (अचर भिन्न *)src0->data;
    भिन्न * dst_d = (भिन्न *)dst->data;
    dpct::queue_ptr     stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    अचर पूर्णाङ्क mode_flags = dst->op_params[0];
    अचर ggml_scale_mode mode = (ggml_scale_mode)(mode_flags & 0xFF);

    भिन्न sf0 = (भिन्न)dst->ne[0]/src0->ne[0];
    भिन्न sf1 = (भिन्न)dst->ne[1]/src0->ne[1];
    भिन्न sf2 = (भिन्न)dst->ne[2]/src0->ne[2];
    अचर भिन्न sf3 = (भिन्न)dst->ne[3]/src0->ne[3];

    भिन्न pixel_offset = 0.5f;
    यदि (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        sf0 = dst->ne[0] > 1 && src0->ne[0] > 1
            ? (भिन्न)(dst->ne[0] - 1) / (src0->ne[0] - 1)
            : sf0;
        sf1 = dst->ne[1] > 1 && src0->ne[1] > 1
            ? (भिन्न)(dst->ne[1] - 1) / (src0->ne[1] - 1)
            : sf1;
        pixel_offset = 0.0f;
    }

    यदि (mode == GGML_SCALE_MODE_NEAREST) {
        upscale_f32_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], sf0, sf1, sf2, sf3, stream);
    } अन्यथा यदि (mode == GGML_SCALE_MODE_BILINEAR) {
        अचर तार्किक antialias = (mode_flags & GGML_SCALE_FLAG_ANTIALIAS);
        upscale_f32_bilinear_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src0->ne[0], src0->ne[1], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            sf0, sf1, sf2, sf3, pixel_offset, antialias, stream);
    } अन्यथा यदि (mode == GGML_SCALE_MODE_BICUBIC) {
        upscale_f32_bicubic_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src0->ne[0], src0->ne[1], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            sf0, sf1, sf2, sf3, pixel_offset, stream);
    }
}

शून्य ggml_sycl_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_upscale(ctx, dst);
}
