#அந்தர்பாவ "upscale.hpp"

நிலையான வெறுமை upscale_f32(மாறிலி தசம * x, தசம * dst,
        மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
        மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு ne13,
        மாறிலி தசம sf0, மாறிலி தசம sf1, மாறிலி தசம sf2, மாறிலி தசம sf3) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    முழு  index    = item_ct1.get_local_id(2) + item_ct1.get_group(2) * item_ct1.get_local_range(2);
    எனில் (index >= ne10 * ne11 * ne12 * ne13) {
        திரும்பு;
    }

    முழு i10 = index % ne10;
    முழு i11 = (index / ne10) % ne11;
    முழு i12 = (index / (ne10 * ne11)) % ne12;
    முழு i13 = (index / (ne10 * ne11 * ne12)) % ne13;

    முழு i00 = i10 / sf0;
    முழு i01 = i11 / sf1;
    முழு i02 = i12 / sf2;
    முழு i03 = i13 / sf3;

    dst[index] = *((மாறிலி தசம*)((மாறிலி எழுத்து*)x + i03 * nb03 + i02 * nb02 +
                                  i01 * nb01 + i00 * nb00));
}

நிலையான வெறுமை upscale_f32_bilinear(மாறிலி தசம * x, தசம * dst,
        மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
        மாறிலி முழு ne00_src, மாறிலி முழு ne01_src,
        மாறிலி முழு ne10_dst, மாறிலி முழு ne11_dst, மாறிலி முழு ne12_dst, மாறிலி முழு ne13_dst,
        மாறிலி தசம sf0, மாறிலி தசம sf1, மாறிலி தசம sf2, மாறிலி தசம sf3,
        மாறிலி தசம pixel_offset) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    மாறிலி int64_t index = item_ct1.get_local_id(2) +
        item_ct1.get_group(2) * item_ct1.get_local_range(2);
    மாறிலி int64_t dst_total_elements = ne10_dst * ne11_dst * ne12_dst * ne13_dst;

  எனில் (index >= dst_total_elements) {
    திரும்பு;
  }

    மாறிலி முழு i10_dst = index % ne10_dst;
    மாறிலி முழு i11_dst = (index / ne10_dst) % ne11_dst;
    மாறிலி முழு i12_dst = (index / (ne10_dst * ne11_dst)) % ne12_dst;
    மாறிலி முழு i13_dst = index / (ne10_dst * ne11_dst * ne12_dst);

    மாறிலி முழு i02_src = (முழு)(i12_dst / sf2);
    மாறிலி முழு i03_src = (முழு)(i13_dst / sf3);

    மாறிலி தசம y_src_f = ((தசம)i11_dst + pixel_offset) / sf1 - pixel_offset;
    முழு         y0_src    = (முழு) sycl::floor((தசம) y_src_f);
    முழு y1_src    = y0_src + 1;

    y0_src = sycl::max(0, sycl::min(y0_src, ne01_src - 1));
    y1_src = sycl::max(0, sycl::min(y1_src, ne01_src - 1));

    தசம dy = y_src_f - (தசம)y0_src;
    dy       = sycl::max(0.0f, sycl::min(dy, 1.0f));

    தசம x_src_f = ((தசம)i10_dst + pixel_offset) / sf0 - pixel_offset;
    முழு   x0_src    = (முழு) sycl::floor(x_src_f);
    முழு x1_src    = x0_src + 1;

    x0_src = sycl::max(0, sycl::min(x0_src, ne00_src - 1));
    x1_src = sycl::max(0, sycl::min(x1_src, ne00_src - 1));

    தசம dx = x_src_f - (தசம)x0_src;
    dx       = sycl::max(0.0f, sycl::min(dx, 1.0f));

    மாறிலி தசம* p_a =
        (மாறிலி தசம*)((மாறிலி எழுத்து*)x + (int64_t)x0_src * nb00 +
                       (int64_t)y0_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    மாறிலி தசம* p_b =
        (மாறிலி தசம*)((மாறிலி எழுத்து*)x + (int64_t)x1_src * nb00 +
                       (int64_t)y0_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    மாறிலி தசம* p_c =
        (மாறிலி தசம*)((மாறிலி எழுத்து*)x + (int64_t)x0_src * nb00 +
                       (int64_t)y1_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    மாறிலி தசம* p_d =
        (மாறிலி தசம*)((மாறிலி எழுத்து*)x + (int64_t)x1_src * nb00 +
                       (int64_t)y1_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);

    மாறிலி தசம val_a = *p_a;
    மாறிலி தசம val_b = *p_b;
    மாறிலி தசம val_c = *p_c;
    மாறிலி தசம val_d = *p_d;

    தசம result = val_a * (1.0f - dx) * (1.0f - dy) +
                   val_b * dx * (1.0f - dy) +
                   val_c * (1.0f - dx) * dy +
                   val_d * dx * dy;

    dst[index] = result;
}

// Similar to F.interpolate(..., mode="bilinear", align_corners=False, antialias=True)
// https://github.com/pytorch/pytorch/blob/8871ff29b743948d1225389d5b7068f37b22750b/aten/src/ATen/native/cpu/UpSampleKernel.cpp
நிலையான வெறுமை upscale_f32_bilinear_antialias(மாறிலி தசம * src0,
                                           தசம *       dst,
                                           மாறிலி முழு     nb00,
                                           மாறிலி முழு     nb01,
                                           மாறிலி முழு     nb02,
                                           மாறிலி முழு     nb03,
                                           மாறிலி முழு     ne00_src,
                                           மாறிலி முழு     ne01_src,
                                           மாறிலி முழு     ne10_dst,
                                           மாறிலி முழு     ne11_dst,
                                           மாறிலி முழு     ne12_dst,
                                           மாறிலி முழு     ne13_dst,
                                           மாறிலி தசம   sf0,
                                           மாறிலி தசம   sf1,
                                           மாறிலி தசம   sf2,
                                           மாறிலி தசம   sf3,
                                           மாறிலி தசம   pixel_offset) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    மாறிலி int64_t index = item_ct1.get_local_id(2) +
        item_ct1.get_group(2) * item_ct1.get_local_range(2);
    மாறிலி int64_t dst_total_elements = ne10_dst * ne11_dst * ne12_dst * ne13_dst;

    எனில் (index >= dst_total_elements) {
        திரும்பு;
    }

    மாறிலி முழு i10_dst = index % ne10_dst;
    மாறிலி முழு i11_dst = (index / ne10_dst) % ne11_dst;
    மாறிலி முழு i12_dst = (index / (ne10_dst * ne11_dst)) % ne12_dst;
    மாறிலி முழு i13_dst = index / (ne10_dst * ne11_dst * ne12_dst);

    மாறிலி முழு i02_src = (முழு)(i12_dst / sf2);
    மாறிலி முழு i03_src = (முழு)(i13_dst / sf3);

    மாறிலி தசம y = ((தசம)i11_dst + pixel_offset) / sf1;
    மாறிலி தசம x = ((தசம)i10_dst + pixel_offset) / sf0;

    // support and invscale, minimum 1 pixel for bilinear
    மாறிலி தசம support1  = sycl::max(1.0f / sf1, 1.0f);
    மாறிலி தசம invscale1 = 1.0f / support1;
    மாறிலி தசம support0  = sycl::max(1.0f / sf0, 1.0f);
    மாறிலி தசம invscale0 = 1.0f / support0;

    // the range of source pixels that contribute
    மாறிலி int64_t x_min = sycl::max(int64_t(0), int64_t(x - support0 + pixel_offset));
    மாறிலி int64_t x_max = sycl::min(int64_t(ne00_src), int64_t(x + support0 + pixel_offset));
    மாறிலி int64_t y_min = sycl::max(int64_t(0), int64_t(y - support1 + pixel_offset));
    மாறிலி int64_t y_max = sycl::min(int64_t(ne01_src), int64_t(y + support1 + pixel_offset));

    // bilinear filter with antialiasing
    தசம val = 0.0f;
    தசம total_weight = 0.0f;

    auto triangle_filter = [](தசம x) -> தசம {
        திரும்பு sycl::max(1.0f - sycl::fabs(x), 0.0f);
    };

    க்கு (int64_t sy = y_min; sy < y_max; sy++) {
        மாறிலி தசம weight_y = triangle_filter((sy - y + pixel_offset) * invscale1);

        க்கு (int64_t sx = x_min; sx < x_max; sx++) {
            மாறிலி தசம weight_x = triangle_filter((sx - x + pixel_offset) * invscale0);
            மாறிலி தசம weight = weight_x * weight_y;

            எனில் (weight <= 0.0f) {
                தொடர்;
            }

            மாறிலி தசம pixel =
                *(மாறிலி தசம*)((மாறிலி எழுத்து*)src0 + sx * nb00 + sy * nb01 +
                                i02_src * nb02 + i03_src * nb03);
            val += pixel * weight;
            total_weight += weight;
        }
    }

    எனில் (total_weight > 0.0f) {
        val /= total_weight;
    }

    dst[index] = val;
}

நாமாகாஶ bicubic_interpolation {
நிலையான தசம weight1(தசம x, மாறிலி தசம &a) { திரும்பு ((a + 2) * x - (a + 3)) * x * x + 1; }
நிலையான தசம weight2(தசம x, மாறிலி தசம &a) { திரும்பு ((a * x - 5 * a) * x + 8 * a) * x - 4 * a; }

நிலையான தசம bicubic(தசம p0, தசம p1, தசம p2, தசம p3, தசம x, தசம a) {
    மாறிலி தசம w0 = weight2(x + 1, a);
    மாறிலி தசம w1 = weight1(x + 0, a);
    மாறிலி தசம w2 = weight1(1 - x, a);
    மாறிலி தசம w3 = weight2(2 - x, a);
    திரும்பு p0 * w0 + p1 * w1 + p2 * w2 + p3 * w3;
}

}

நிலையான வெறுமை upscale_f32_bicubic(மாறிலி தசம * x, தசம * dst,
        மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
        மாறிலி முழு ne00_src, மாறிலி முழு ne01_src,
        மாறிலி முழு ne10_dst, மாறிலி முழு ne11_dst, மாறிலி முழு ne12_dst, மாறிலி முழு ne13_dst,
        மாறிலி தசம sf0, மாறிலி தசம sf1, மாறிலி தசம sf2, மாறிலி தசம sf3,
        மாறிலி தசம pixel_offset) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    மாறிலி தசம a = -0.75f;
    using bicubic_interpolation::bicubic;

    மாறிலி int64_t index = item_ct1.get_local_id(2) +
        item_ct1.get_group(2) * item_ct1.get_local_range(2);
    மாறிலி int64_t dst_total_elements =
        ne10_dst * ne11_dst * ne12_dst * ne13_dst;

    எனில் (index >= dst_total_elements) {
        திரும்பு;
    }

    மாறிலி முழு i10_dst = index % ne10_dst;
    மாறிலி முழு i11_dst = (index / ne10_dst) % ne11_dst;
    மாறிலி முழு i12_dst = (index / (ne10_dst * ne11_dst)) % ne12_dst;
    மாறிலி முழு i13_dst = index / (ne10_dst * ne11_dst * ne12_dst);

    மாறிலி முழு i02_src = (முழு)(i12_dst / sf2);
    மாறிலி முழு i03_src = (முழு)(i13_dst / sf3);

    மாறிலி தசம y_src_f = ((தசம)i11_dst + pixel_offset) / sf1 - pixel_offset;
    மாறிலி முழு   y0_src  = (முழு) sycl::floor((தசம) y_src_f);
    மாறிலி தசம dy      = y_src_f - (தசம)y0_src;

    மாறிலி தசம x_src_f = ((தசம)i10_dst + pixel_offset) / sf0 - pixel_offset;
    மாறிலி முழு   x0_src  = (முழு) sycl::floor((தசம) x_src_f);
    மாறிலி தசம dx      = x_src_f - (தசம)x0_src;

    மாறிலி எழுத்து * x_base = (மாறிலி எழுத்து *)x + (int64_t)i02_src * nb02 + (int64_t)i03_src * nb03;

    auto load = [=](முழு x_off, முழு y_off) -> தசம {
        முழு i00_src = sycl::max(0, sycl::min(x0_src + x_off, ne00_src - 1));
        முழு i01_src = sycl::max(0, sycl::min(y0_src + y_off, ne01_src - 1));
        திரும்பு *(மாறிலி தசம *)(x_base + (int64_t)i00_src * nb00 + (int64_t)i01_src * nb01);
    };

    மாறிலி தசம result = bicubic(
        bicubic(load(-1, -1), load(0, -1), load(1, -1), load(2, -1), dx, a),
        bicubic(load(-1, 0), load(0, 0), load(1, 0), load(2, 0), dx, a),
        bicubic(load(-1, 1), load(0, 1), load(1, 1), load(2, 1), dx, a),
        bicubic(load(-1, 2), load(0, 2), load(1, 2), load(2, 2), dx, a),
        dy,
        a);

    dst[index] = result;
}

நிலையான வெறுமை upscale_f32_sycl(மாறிலி தசம *   x,
                             தசம *         dst,
                             மாறிலி முழு       nb00,
                             மாறிலி முழு       nb01,
                             மாறிலி முழு       nb02,
                             மாறிலி முழு       nb03,
                             மாறிலி முழு       ne10,
                             மாறிலி முழு       ne11,
                             மாறிலி முழு       ne12,
                             மாறிலி முழு       ne13,
                             மாறிலி தசம     sf0,
                             மாறிலி தசம     sf1,
                             மாறிலி தசம     sf2,
                             மாறிலி தசம     sf3,
                             dpct::queue_ptr stream) {
    மாறிலி int64_t dst_size   = ne10 * ne11 * ne12 * ne13;
    மாறிலி int64_t இலக்கம்_blocks = (dst_size + SYCL_UPSCALE_BLOCK_SIZE - 1) / SYCL_UPSCALE_BLOCK_SIZE;

    stream->parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE),
             sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE)),
        [=](sycl::nd_item<3> /*item_ct1*/) {
            upscale_f32(x, dst, nb00, nb01, nb02, nb03, ne10, ne11, ne12, ne13, sf0, sf1, sf2, sf3);
        });
}

நிலையான வெறுமை upscale_f32_bilinear_sycl(மாறிலி தசம *   x,
                                      தசம *         dst,
                                      மாறிலி முழு       nb00,
                                      மாறிலி முழு       nb01,
                                      மாறிலி முழு       nb02,
                                      மாறிலி முழு       nb03,
                                      மாறிலி முழு       ne00_src,
                                      மாறிலி முழு       ne01_src,
                                      மாறிலி முழு       ne10_dst,
                                      மாறிலி முழு       ne11_dst,
                                      மாறிலி முழு       ne12_dst,
                                      மாறிலி முழு       ne13_dst,
                                      மாறிலி தசம     sf0,
                                      மாறிலி தசம     sf1,
                                      மாறிலி தசம     sf2,
                                      மாறிலி தசம     sf3,
                                      மாறிலி தசம     pixel_offset,
                                      தருக்க            antialias,
                                      dpct::queue_ptr stream) {
    மாறிலி int64_t dst_size   = ne10_dst * ne11_dst * ne12_dst * ne13_dst;
    மாறிலி int64_t இலக்கம்_blocks = (dst_size + SYCL_UPSCALE_BLOCK_SIZE - 1) / SYCL_UPSCALE_BLOCK_SIZE;

    எனில் (antialias) {
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE),
                sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE)),
            [=](sycl::nd_item<3> /*item_ct1*/) {
                upscale_f32_bilinear_antialias(
                    x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst,
                    ne12_dst, ne13_dst, sf0, sf1, sf2, sf3, pixel_offset);
            });
    } இல்லையெனில் {
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE),
                sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE)),
            [=](sycl::nd_item<3> /*item_ct1*/) {
                upscale_f32_bilinear(
                    x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst, ne12_dst,
                    ne13_dst, sf0, sf1, sf2, sf3, pixel_offset);
            });
    }
}

நிலையான வெறுமை upscale_f32_bicubic_sycl(மாறிலி தசம *   x,
                                     தசம *         dst,
                                     மாறிலி முழு       nb00,
                                     மாறிலி முழு       nb01,
                                     மாறிலி முழு       nb02,
                                     மாறிலி முழு       nb03,
                                     மாறிலி முழு       ne00_src,
                                     மாறிலி முழு       ne01_src,
                                     மாறிலி முழு       ne10_dst,
                                     மாறிலி முழு       ne11_dst,
                                     மாறிலி முழு       ne12_dst,
                                     மாறிலி முழு       ne13_dst,
                                     மாறிலி தசம     sf0,
                                     மாறிலி தசம     sf1,
                                     மாறிலி தசம     sf2,
                                     மாறிலி தசம     sf3,
                                     மாறிலி தசம     pixel_offset,
                                     dpct::queue_ptr stream) {
    மாறிலி int64_t dst_size   = ne10_dst * ne11_dst * ne12_dst * ne13_dst;
    மாறிலி int64_t இலக்கம்_blocks = (dst_size + SYCL_UPSCALE_BLOCK_SIZE - 1) / SYCL_UPSCALE_BLOCK_SIZE;

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE),
                    sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE)),
                [=](sycl::nd_item<3> /*item_ct1*/) {
                    upscale_f32_bicubic(
                        x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst,
                        ne12_dst, ne13_dst, sf0, sf1, sf2, sf3, pixel_offset);
                });
        });
    }
}

நிலையான வெறுமை ggml_sycl_op_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி தசம * src0_d = (மாறிலி தசம *)src0->data;
    தசம * dst_d = (தசம *)dst->data;
    dpct::queue_ptr     stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    மாறிலி முழு mode_flags = dst->op_params[0];
    மாறிலி ggml_scale_mode mode = (ggml_scale_mode)(mode_flags & 0xFF);

    தசம sf0 = (தசம)dst->ne[0]/src0->ne[0];
    தசம sf1 = (தசம)dst->ne[1]/src0->ne[1];
    தசம sf2 = (தசம)dst->ne[2]/src0->ne[2];
    மாறிலி தசம sf3 = (தசம)dst->ne[3]/src0->ne[3];

    தசம pixel_offset = 0.5f;
    எனில் (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        sf0 = dst->ne[0] > 1 && src0->ne[0] > 1
            ? (தசம)(dst->ne[0] - 1) / (src0->ne[0] - 1)
            : sf0;
        sf1 = dst->ne[1] > 1 && src0->ne[1] > 1
            ? (தசம)(dst->ne[1] - 1) / (src0->ne[1] - 1)
            : sf1;
        pixel_offset = 0.0f;
    }

    எனில் (mode == GGML_SCALE_MODE_NEAREST) {
        upscale_f32_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], sf0, sf1, sf2, sf3, stream);
    } இல்லையெனில் எனில் (mode == GGML_SCALE_MODE_BILINEAR) {
        மாறிலி தருக்க antialias = (mode_flags & GGML_SCALE_FLAG_ANTIALIAS);
        upscale_f32_bilinear_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src0->ne[0], src0->ne[1], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            sf0, sf1, sf2, sf3, pixel_offset, antialias, stream);
    } இல்லையெனில் எனில் (mode == GGML_SCALE_MODE_BICUBIC) {
        upscale_f32_bicubic_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src0->ne[0], src0->ne[1], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            sf0, sf1, sf2, sf3, pixel_offset, stream);
    }
}

வெறுமை ggml_sycl_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_upscale(ctx, dst);
}
