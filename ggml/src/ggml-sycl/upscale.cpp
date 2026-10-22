#അന്തര്ഭാവ "upscale.hpp"

സ്ഥിതം ശൂന്യം upscale_f32(സ്ഥിരം ദശാംശം * x, ദശാംശം * dst,
        സ്ഥിരം പൂർണ്ണസംഖ്യ nb00, സ്ഥിരം പൂർണ്ണസംഖ്യ nb01, സ്ഥിരം പൂർണ്ണസംഖ്യ nb02, സ്ഥിരം പൂർണ്ണസംഖ്യ nb03,
        സ്ഥിരം പൂർണ്ണസംഖ്യ ne10, സ്ഥിരം പൂർണ്ണസംഖ്യ ne11, സ്ഥിരം പൂർണ്ണസംഖ്യ ne12, സ്ഥിരം പൂർണ്ണസംഖ്യ ne13,
        സ്ഥിരം ദശാംശം sf0, സ്ഥിരം ദശാംശം sf1, സ്ഥിരം ദശാംശം sf2, സ്ഥിരം ദശാംശം sf3) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    പൂർണ്ണസംഖ്യ  index    = item_ct1.get_local_id(2) + item_ct1.get_group(2) * item_ct1.get_local_range(2);
    എങ്കിൽ (index >= ne10 * ne11 * ne12 * ne13) {
        തിരികെ;
    }

    പൂർണ്ണസംഖ്യ i10 = index % ne10;
    പൂർണ്ണസംഖ്യ i11 = (index / ne10) % ne11;
    പൂർണ്ണസംഖ്യ i12 = (index / (ne10 * ne11)) % ne12;
    പൂർണ്ണസംഖ്യ i13 = (index / (ne10 * ne11 * ne12)) % ne13;

    പൂർണ്ണസംഖ്യ i00 = i10 / sf0;
    പൂർണ്ണസംഖ്യ i01 = i11 / sf1;
    പൂർണ്ണസംഖ്യ i02 = i12 / sf2;
    പൂർണ്ണസംഖ്യ i03 = i13 / sf3;

    dst[index] = *((സ്ഥിരം ദശാംശം*)((സ്ഥിരം അക്ഷരം*)x + i03 * nb03 + i02 * nb02 +
                                  i01 * nb01 + i00 * nb00));
}

സ്ഥിതം ശൂന്യം upscale_f32_bilinear(സ്ഥിരം ദശാംശം * x, ദശാംശം * dst,
        സ്ഥിരം പൂർണ്ണസംഖ്യ nb00, സ്ഥിരം പൂർണ്ണസംഖ്യ nb01, സ്ഥിരം പൂർണ്ണസംഖ്യ nb02, സ്ഥിരം പൂർണ്ണസംഖ്യ nb03,
        സ്ഥിരം പൂർണ്ണസംഖ്യ ne00_src, സ്ഥിരം പൂർണ്ണസംഖ്യ ne01_src,
        സ്ഥിരം പൂർണ്ണസംഖ്യ ne10_dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ne11_dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ne12_dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ne13_dst,
        സ്ഥിരം ദശാംശം sf0, സ്ഥിരം ദശാംശം sf1, സ്ഥിരം ദശാംശം sf2, സ്ഥിരം ദശാംശം sf3,
        സ്ഥിരം ദശാംശം pixel_offset) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    സ്ഥിരം int64_t index = item_ct1.get_local_id(2) +
        item_ct1.get_group(2) * item_ct1.get_local_range(2);
    സ്ഥിരം int64_t dst_total_elements = ne10_dst * ne11_dst * ne12_dst * ne13_dst;

  എങ്കിൽ (index >= dst_total_elements) {
    തിരികെ;
  }

    സ്ഥിരം പൂർണ്ണസംഖ്യ i10_dst = index % ne10_dst;
    സ്ഥിരം പൂർണ്ണസംഖ്യ i11_dst = (index / ne10_dst) % ne11_dst;
    സ്ഥിരം പൂർണ്ണസംഖ്യ i12_dst = (index / (ne10_dst * ne11_dst)) % ne12_dst;
    സ്ഥിരം പൂർണ്ണസംഖ്യ i13_dst = index / (ne10_dst * ne11_dst * ne12_dst);

    സ്ഥിരം പൂർണ്ണസംഖ്യ i02_src = (പൂർണ്ണസംഖ്യ)(i12_dst / sf2);
    സ്ഥിരം പൂർണ്ണസംഖ്യ i03_src = (പൂർണ്ണസംഖ്യ)(i13_dst / sf3);

    സ്ഥിരം ദശാംശം y_src_f = ((ദശാംശം)i11_dst + pixel_offset) / sf1 - pixel_offset;
    പൂർണ്ണസംഖ്യ         y0_src    = (പൂർണ്ണസംഖ്യ) sycl::floor((ദശാംശം) y_src_f);
    പൂർണ്ണസംഖ്യ y1_src    = y0_src + 1;

    y0_src = sycl::max(0, sycl::min(y0_src, ne01_src - 1));
    y1_src = sycl::max(0, sycl::min(y1_src, ne01_src - 1));

    ദശാംശം dy = y_src_f - (ദശാംശം)y0_src;
    dy       = sycl::max(0.0f, sycl::min(dy, 1.0f));

    ദശാംശം x_src_f = ((ദശാംശം)i10_dst + pixel_offset) / sf0 - pixel_offset;
    പൂർണ്ണസംഖ്യ   x0_src    = (പൂർണ്ണസംഖ്യ) sycl::floor(x_src_f);
    പൂർണ്ണസംഖ്യ x1_src    = x0_src + 1;

    x0_src = sycl::max(0, sycl::min(x0_src, ne00_src - 1));
    x1_src = sycl::max(0, sycl::min(x1_src, ne00_src - 1));

    ദശാംശം dx = x_src_f - (ദശാംശം)x0_src;
    dx       = sycl::max(0.0f, sycl::min(dx, 1.0f));

    സ്ഥിരം ദശാംശം* p_a =
        (സ്ഥിരം ദശാംശം*)((സ്ഥിരം അക്ഷരം*)x + (int64_t)x0_src * nb00 +
                       (int64_t)y0_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    സ്ഥിരം ദശാംശം* p_b =
        (സ്ഥിരം ദശാംശം*)((സ്ഥിരം അക്ഷരം*)x + (int64_t)x1_src * nb00 +
                       (int64_t)y0_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    സ്ഥിരം ദശാംശം* p_c =
        (സ്ഥിരം ദശാംശം*)((സ്ഥിരം അക്ഷരം*)x + (int64_t)x0_src * nb00 +
                       (int64_t)y1_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    സ്ഥിരം ദശാംശം* p_d =
        (സ്ഥിരം ദശാംശം*)((സ്ഥിരം അക്ഷരം*)x + (int64_t)x1_src * nb00 +
                       (int64_t)y1_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);

    സ്ഥിരം ദശാംശം val_a = *p_a;
    സ്ഥിരം ദശാംശം val_b = *p_b;
    സ്ഥിരം ദശാംശം val_c = *p_c;
    സ്ഥിരം ദശാംശം val_d = *p_d;

    ദശാംശം result = val_a * (1.0f - dx) * (1.0f - dy) +
                   val_b * dx * (1.0f - dy) +
                   val_c * (1.0f - dx) * dy +
                   val_d * dx * dy;

    dst[index] = result;
}

// Similar to F.interpolate(..., mode="bilinear", align_corners=False, antialias=True)
// https://github.com/pytorch/pytorch/blob/8871ff29b743948d1225389d5b7068f37b22750b/aten/src/ATen/native/cpu/UpSampleKernel.cpp
സ്ഥിതം ശൂന്യം upscale_f32_bilinear_antialias(സ്ഥിരം ദശാംശം * src0,
                                           ദശാംശം *       dst,
                                           സ്ഥിരം പൂർണ്ണസംഖ്യ     nb00,
                                           സ്ഥിരം പൂർണ്ണസംഖ്യ     nb01,
                                           സ്ഥിരം പൂർണ്ണസംഖ്യ     nb02,
                                           സ്ഥിരം പൂർണ്ണസംഖ്യ     nb03,
                                           സ്ഥിരം പൂർണ്ണസംഖ്യ     ne00_src,
                                           സ്ഥിരം പൂർണ്ണസംഖ്യ     ne01_src,
                                           സ്ഥിരം പൂർണ്ണസംഖ്യ     ne10_dst,
                                           സ്ഥിരം പൂർണ്ണസംഖ്യ     ne11_dst,
                                           സ്ഥിരം പൂർണ്ണസംഖ്യ     ne12_dst,
                                           സ്ഥിരം പൂർണ്ണസംഖ്യ     ne13_dst,
                                           സ്ഥിരം ദശാംശം   sf0,
                                           സ്ഥിരം ദശാംശം   sf1,
                                           സ്ഥിരം ദശാംശം   sf2,
                                           സ്ഥിരം ദശാംശം   sf3,
                                           സ്ഥിരം ദശാംശം   pixel_offset) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    സ്ഥിരം int64_t index = item_ct1.get_local_id(2) +
        item_ct1.get_group(2) * item_ct1.get_local_range(2);
    സ്ഥിരം int64_t dst_total_elements = ne10_dst * ne11_dst * ne12_dst * ne13_dst;

    എങ്കിൽ (index >= dst_total_elements) {
        തിരികെ;
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ i10_dst = index % ne10_dst;
    സ്ഥിരം പൂർണ്ണസംഖ്യ i11_dst = (index / ne10_dst) % ne11_dst;
    സ്ഥിരം പൂർണ്ണസംഖ്യ i12_dst = (index / (ne10_dst * ne11_dst)) % ne12_dst;
    സ്ഥിരം പൂർണ്ണസംഖ്യ i13_dst = index / (ne10_dst * ne11_dst * ne12_dst);

    സ്ഥിരം പൂർണ്ണസംഖ്യ i02_src = (പൂർണ്ണസംഖ്യ)(i12_dst / sf2);
    സ്ഥിരം പൂർണ്ണസംഖ്യ i03_src = (പൂർണ്ണസംഖ്യ)(i13_dst / sf3);

    സ്ഥിരം ദശാംശം y = ((ദശാംശം)i11_dst + pixel_offset) / sf1;
    സ്ഥിരം ദശാംശം x = ((ദശാംശം)i10_dst + pixel_offset) / sf0;

    // support and invscale, minimum 1 pixel for bilinear
    സ്ഥിരം ദശാംശം support1  = sycl::max(1.0f / sf1, 1.0f);
    സ്ഥിരം ദശാംശം invscale1 = 1.0f / support1;
    സ്ഥിരം ദശാംശം support0  = sycl::max(1.0f / sf0, 1.0f);
    സ്ഥിരം ദശാംശം invscale0 = 1.0f / support0;

    // the range of source pixels that contribute
    സ്ഥിരം int64_t x_min = sycl::max(int64_t(0), int64_t(x - support0 + pixel_offset));
    സ്ഥിരം int64_t x_max = sycl::min(int64_t(ne00_src), int64_t(x + support0 + pixel_offset));
    സ്ഥിരം int64_t y_min = sycl::max(int64_t(0), int64_t(y - support1 + pixel_offset));
    സ്ഥിരം int64_t y_max = sycl::min(int64_t(ne01_src), int64_t(y + support1 + pixel_offset));

    // bilinear filter with antialiasing
    ദശാംശം val = 0.0f;
    ദശാംശം total_weight = 0.0f;

    auto triangle_filter = [](ദശാംശം x) -> ദശാംശം {
        തിരികെ sycl::max(1.0f - sycl::fabs(x), 0.0f);
    };

    വേണ്ടി (int64_t sy = y_min; sy < y_max; sy++) {
        സ്ഥിരം ദശാംശം weight_y = triangle_filter((sy - y + pixel_offset) * invscale1);

        വേണ്ടി (int64_t sx = x_min; sx < x_max; sx++) {
            സ്ഥിരം ദശാംശം weight_x = triangle_filter((sx - x + pixel_offset) * invscale0);
            സ്ഥിരം ദശാംശം weight = weight_x * weight_y;

            എങ്കിൽ (weight <= 0.0f) {
                തുടരുക;
            }

            സ്ഥിരം ദശാംശം pixel =
                *(സ്ഥിരം ദശാംശം*)((സ്ഥിരം അക്ഷരം*)src0 + sx * nb00 + sy * nb01 +
                                i02_src * nb02 + i03_src * nb03);
            val += pixel * weight;
            total_weight += weight;
        }
    }

    എങ്കിൽ (total_weight > 0.0f) {
        val /= total_weight;
    }

    dst[index] = val;
}

നാമാകാശ bicubic_interpolation {
സ്ഥിതം ദശാംശം weight1(ദശാംശം x, സ്ഥിരം ദശാംശം &a) { തിരികെ ((a + 2) * x - (a + 3)) * x * x + 1; }
സ്ഥിതം ദശാംശം weight2(ദശാംശം x, സ്ഥിരം ദശാംശം &a) { തിരികെ ((a * x - 5 * a) * x + 8 * a) * x - 4 * a; }

സ്ഥിതം ദശാംശം bicubic(ദശാംശം p0, ദശാംശം p1, ദശാംശം p2, ദശാംശം p3, ദശാംശം x, ദശാംശം a) {
    സ്ഥിരം ദശാംശം w0 = weight2(x + 1, a);
    സ്ഥിരം ദശാംശം w1 = weight1(x + 0, a);
    സ്ഥിരം ദശാംശം w2 = weight1(1 - x, a);
    സ്ഥിരം ദശാംശം w3 = weight2(2 - x, a);
    തിരികെ p0 * w0 + p1 * w1 + p2 * w2 + p3 * w3;
}

}

സ്ഥിതം ശൂന്യം upscale_f32_bicubic(സ്ഥിരം ദശാംശം * x, ദശാംശം * dst,
        സ്ഥിരം പൂർണ്ണസംഖ്യ nb00, സ്ഥിരം പൂർണ്ണസംഖ്യ nb01, സ്ഥിരം പൂർണ്ണസംഖ്യ nb02, സ്ഥിരം പൂർണ്ണസംഖ്യ nb03,
        സ്ഥിരം പൂർണ്ണസംഖ്യ ne00_src, സ്ഥിരം പൂർണ്ണസംഖ്യ ne01_src,
        സ്ഥിരം പൂർണ്ണസംഖ്യ ne10_dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ne11_dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ne12_dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ne13_dst,
        സ്ഥിരം ദശാംശം sf0, സ്ഥിരം ദശാംശം sf1, സ്ഥിരം ദശാംശം sf2, സ്ഥിരം ദശാംശം sf3,
        സ്ഥിരം ദശാംശം pixel_offset) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    സ്ഥിരം ദശാംശം a = -0.75f;
    using bicubic_interpolation::bicubic;

    സ്ഥിരം int64_t index = item_ct1.get_local_id(2) +
        item_ct1.get_group(2) * item_ct1.get_local_range(2);
    സ്ഥിരം int64_t dst_total_elements =
        ne10_dst * ne11_dst * ne12_dst * ne13_dst;

    എങ്കിൽ (index >= dst_total_elements) {
        തിരികെ;
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ i10_dst = index % ne10_dst;
    സ്ഥിരം പൂർണ്ണസംഖ്യ i11_dst = (index / ne10_dst) % ne11_dst;
    സ്ഥിരം പൂർണ്ണസംഖ്യ i12_dst = (index / (ne10_dst * ne11_dst)) % ne12_dst;
    സ്ഥിരം പൂർണ്ണസംഖ്യ i13_dst = index / (ne10_dst * ne11_dst * ne12_dst);

    സ്ഥിരം പൂർണ്ണസംഖ്യ i02_src = (പൂർണ്ണസംഖ്യ)(i12_dst / sf2);
    സ്ഥിരം പൂർണ്ണസംഖ്യ i03_src = (പൂർണ്ണസംഖ്യ)(i13_dst / sf3);

    സ്ഥിരം ദശാംശം y_src_f = ((ദശാംശം)i11_dst + pixel_offset) / sf1 - pixel_offset;
    സ്ഥിരം പൂർണ്ണസംഖ്യ   y0_src  = (പൂർണ്ണസംഖ്യ) sycl::floor((ദശാംശം) y_src_f);
    സ്ഥിരം ദശാംശം dy      = y_src_f - (ദശാംശം)y0_src;

    സ്ഥിരം ദശാംശം x_src_f = ((ദശാംശം)i10_dst + pixel_offset) / sf0 - pixel_offset;
    സ്ഥിരം പൂർണ്ണസംഖ്യ   x0_src  = (പൂർണ്ണസംഖ്യ) sycl::floor((ദശാംശം) x_src_f);
    സ്ഥിരം ദശാംശം dx      = x_src_f - (ദശാംശം)x0_src;

    സ്ഥിരം അക്ഷരം * x_base = (സ്ഥിരം അക്ഷരം *)x + (int64_t)i02_src * nb02 + (int64_t)i03_src * nb03;

    auto load = [=](പൂർണ്ണസംഖ്യ x_off, പൂർണ്ണസംഖ്യ y_off) -> ദശാംശം {
        പൂർണ്ണസംഖ്യ i00_src = sycl::max(0, sycl::min(x0_src + x_off, ne00_src - 1));
        പൂർണ്ണസംഖ്യ i01_src = sycl::max(0, sycl::min(y0_src + y_off, ne01_src - 1));
        തിരികെ *(സ്ഥിരം ദശാംശം *)(x_base + (int64_t)i00_src * nb00 + (int64_t)i01_src * nb01);
    };

    സ്ഥിരം ദശാംശം result = bicubic(
        bicubic(load(-1, -1), load(0, -1), load(1, -1), load(2, -1), dx, a),
        bicubic(load(-1, 0), load(0, 0), load(1, 0), load(2, 0), dx, a),
        bicubic(load(-1, 1), load(0, 1), load(1, 1), load(2, 1), dx, a),
        bicubic(load(-1, 2), load(0, 2), load(1, 2), load(2, 2), dx, a),
        dy,
        a);

    dst[index] = result;
}

സ്ഥിതം ശൂന്യം upscale_f32_sycl(സ്ഥിരം ദശാംശം *   x,
                             ദശാംശം *         dst,
                             സ്ഥിരം പൂർണ്ണസംഖ്യ       nb00,
                             സ്ഥിരം പൂർണ്ണസംഖ്യ       nb01,
                             സ്ഥിരം പൂർണ്ണസംഖ്യ       nb02,
                             സ്ഥിരം പൂർണ്ണസംഖ്യ       nb03,
                             സ്ഥിരം പൂർണ്ണസംഖ്യ       ne10,
                             സ്ഥിരം പൂർണ്ണസംഖ്യ       ne11,
                             സ്ഥിരം പൂർണ്ണസംഖ്യ       ne12,
                             സ്ഥിരം പൂർണ്ണസംഖ്യ       ne13,
                             സ്ഥിരം ദശാംശം     sf0,
                             സ്ഥിരം ദശാംശം     sf1,
                             സ്ഥിരം ദശാംശം     sf2,
                             സ്ഥിരം ദശാംശം     sf3,
                             dpct::queue_ptr stream) {
    സ്ഥിരം int64_t dst_size   = ne10 * ne11 * ne12 * ne13;
    സ്ഥിരം int64_t അക്കം_blocks = (dst_size + SYCL_UPSCALE_BLOCK_SIZE - 1) / SYCL_UPSCALE_BLOCK_SIZE;

    stream->parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, അക്കം_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE),
             sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE)),
        [=](sycl::nd_item<3> /*item_ct1*/) {
            upscale_f32(x, dst, nb00, nb01, nb02, nb03, ne10, ne11, ne12, ne13, sf0, sf1, sf2, sf3);
        });
}

സ്ഥിതം ശൂന്യം upscale_f32_bilinear_sycl(സ്ഥിരം ദശാംശം *   x,
                                      ദശാംശം *         dst,
                                      സ്ഥിരം പൂർണ്ണസംഖ്യ       nb00,
                                      സ്ഥിരം പൂർണ്ണസംഖ്യ       nb01,
                                      സ്ഥിരം പൂർണ്ണസംഖ്യ       nb02,
                                      സ്ഥിരം പൂർണ്ണസംഖ്യ       nb03,
                                      സ്ഥിരം പൂർണ്ണസംഖ്യ       ne00_src,
                                      സ്ഥിരം പൂർണ്ണസംഖ്യ       ne01_src,
                                      സ്ഥിരം പൂർണ്ണസംഖ്യ       ne10_dst,
                                      സ്ഥിരം പൂർണ്ണസംഖ്യ       ne11_dst,
                                      സ്ഥിരം പൂർണ്ണസംഖ്യ       ne12_dst,
                                      സ്ഥിരം പൂർണ്ണസംഖ്യ       ne13_dst,
                                      സ്ഥിരം ദശാംശം     sf0,
                                      സ്ഥിരം ദശാംശം     sf1,
                                      സ്ഥിരം ദശാംശം     sf2,
                                      സ്ഥിരം ദശാംശം     sf3,
                                      സ്ഥിരം ദശാംശം     pixel_offset,
                                      തർക്കം            antialias,
                                      dpct::queue_ptr stream) {
    സ്ഥിരം int64_t dst_size   = ne10_dst * ne11_dst * ne12_dst * ne13_dst;
    സ്ഥിരം int64_t അക്കം_blocks = (dst_size + SYCL_UPSCALE_BLOCK_SIZE - 1) / SYCL_UPSCALE_BLOCK_SIZE;

    എങ്കിൽ (antialias) {
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, അക്കം_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE),
                sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE)),
            [=](sycl::nd_item<3> /*item_ct1*/) {
                upscale_f32_bilinear_antialias(
                    x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst,
                    ne12_dst, ne13_dst, sf0, sf1, sf2, sf3, pixel_offset);
            });
    } അല്ലെങ്കിൽ {
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, അക്കം_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE),
                sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE)),
            [=](sycl::nd_item<3> /*item_ct1*/) {
                upscale_f32_bilinear(
                    x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst, ne12_dst,
                    ne13_dst, sf0, sf1, sf2, sf3, pixel_offset);
            });
    }
}

സ്ഥിതം ശൂന്യം upscale_f32_bicubic_sycl(സ്ഥിരം ദശാംശം *   x,
                                     ദശാംശം *         dst,
                                     സ്ഥിരം പൂർണ്ണസംഖ്യ       nb00,
                                     സ്ഥിരം പൂർണ്ണസംഖ്യ       nb01,
                                     സ്ഥിരം പൂർണ്ണസംഖ്യ       nb02,
                                     സ്ഥിരം പൂർണ്ണസംഖ്യ       nb03,
                                     സ്ഥിരം പൂർണ്ണസംഖ്യ       ne00_src,
                                     സ്ഥിരം പൂർണ്ണസംഖ്യ       ne01_src,
                                     സ്ഥിരം പൂർണ്ണസംഖ്യ       ne10_dst,
                                     സ്ഥിരം പൂർണ്ണസംഖ്യ       ne11_dst,
                                     സ്ഥിരം പൂർണ്ണസംഖ്യ       ne12_dst,
                                     സ്ഥിരം പൂർണ്ണസംഖ്യ       ne13_dst,
                                     സ്ഥിരം ദശാംശം     sf0,
                                     സ്ഥിരം ദശാംശം     sf1,
                                     സ്ഥിരം ദശാംശം     sf2,
                                     സ്ഥിരം ദശാംശം     sf3,
                                     സ്ഥിരം ദശാംശം     pixel_offset,
                                     dpct::queue_ptr stream) {
    സ്ഥിരം int64_t dst_size   = ne10_dst * ne11_dst * ne12_dst * ne13_dst;
    സ്ഥിരം int64_t അക്കം_blocks = (dst_size + SYCL_UPSCALE_BLOCK_SIZE - 1) / SYCL_UPSCALE_BLOCK_SIZE;

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, 1, അക്കം_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE),
                    sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE)),
                [=](sycl::nd_item<3> /*item_ct1*/) {
                    upscale_f32_bicubic(
                        x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst,
                        ne12_dst, ne13_dst, sf0, sf1, sf2, sf3, pixel_offset);
                });
        });
    }
}

സ്ഥിതം ശൂന്യം ggml_sycl_op_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ദശാംശം * src0_d = (സ്ഥിരം ദശാംശം *)src0->data;
    ദശാംശം * dst_d = (ദശാംശം *)dst->data;
    dpct::queue_ptr     stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    സ്ഥിരം പൂർണ്ണസംഖ്യ mode_flags = dst->op_params[0];
    സ്ഥിരം ggml_scale_mode mode = (ggml_scale_mode)(mode_flags & 0xFF);

    ദശാംശം sf0 = (ദശാംശം)dst->ne[0]/src0->ne[0];
    ദശാംശം sf1 = (ദശാംശം)dst->ne[1]/src0->ne[1];
    ദശാംശം sf2 = (ദശാംശം)dst->ne[2]/src0->ne[2];
    സ്ഥിരം ദശാംശം sf3 = (ദശാംശം)dst->ne[3]/src0->ne[3];

    ദശാംശം pixel_offset = 0.5f;
    എങ്കിൽ (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        sf0 = dst->ne[0] > 1 && src0->ne[0] > 1
            ? (ദശാംശം)(dst->ne[0] - 1) / (src0->ne[0] - 1)
            : sf0;
        sf1 = dst->ne[1] > 1 && src0->ne[1] > 1
            ? (ദശാംശം)(dst->ne[1] - 1) / (src0->ne[1] - 1)
            : sf1;
        pixel_offset = 0.0f;
    }

    എങ്കിൽ (mode == GGML_SCALE_MODE_NEAREST) {
        upscale_f32_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], sf0, sf1, sf2, sf3, stream);
    } അല്ലെങ്കിൽ എങ്കിൽ (mode == GGML_SCALE_MODE_BILINEAR) {
        സ്ഥിരം തർക്കം antialias = (mode_flags & GGML_SCALE_FLAG_ANTIALIAS);
        upscale_f32_bilinear_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src0->ne[0], src0->ne[1], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            sf0, sf1, sf2, sf3, pixel_offset, antialias, stream);
    } അല്ലെങ്കിൽ എങ്കിൽ (mode == GGML_SCALE_MODE_BICUBIC) {
        upscale_f32_bicubic_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src0->ne[0], src0->ne[1], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            sf0, sf1, sf2, sf3, pixel_offset, stream);
    }
}

ശൂന്യം ggml_sycl_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_upscale(ctx, dst);
}
