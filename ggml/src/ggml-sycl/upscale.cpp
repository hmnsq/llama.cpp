#અન્તર્ભાવ "upscale.hpp"

સ્થિર શૂન્ય upscale_f32(અચલ દશાંશ * x, દશાંશ * dst,
        અચલ પૂર્ણાંક nb00, અચલ પૂર્ણાંક nb01, અચલ પૂર્ણાંક nb02, અચલ પૂર્ણાંક nb03,
        અચલ પૂર્ણાંક ne10, અચલ પૂર્ણાંક ne11, અચલ પૂર્ણાંક ne12, અચલ પૂર્ણાંક ne13,
        અચલ દશાંશ sf0, અચલ દશાંશ sf1, અચલ દશાંશ sf2, અચલ દશાંશ sf3) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    પૂર્ણાંક  index    = item_ct1.get_local_id(2) + item_ct1.get_group(2) * item_ct1.get_local_range(2);
    જો (index >= ne10 * ne11 * ne12 * ne13) {
        પાછા;
    }

    પૂર્ણાંક i10 = index % ne10;
    પૂર્ણાંક i11 = (index / ne10) % ne11;
    પૂર્ણાંક i12 = (index / (ne10 * ne11)) % ne12;
    પૂર્ણાંક i13 = (index / (ne10 * ne11 * ne12)) % ne13;

    પૂર્ણાંક i00 = i10 / sf0;
    પૂર્ણાંક i01 = i11 / sf1;
    પૂર્ણાંક i02 = i12 / sf2;
    પૂર્ણાંક i03 = i13 / sf3;

    dst[index] = *((અચલ દશાંશ*)((અચલ અક્ષર*)x + i03 * nb03 + i02 * nb02 +
                                  i01 * nb01 + i00 * nb00));
}

સ્થિર શૂન્ય upscale_f32_bilinear(અચલ દશાંશ * x, દશાંશ * dst,
        અચલ પૂર્ણાંક nb00, અચલ પૂર્ણાંક nb01, અચલ પૂર્ણાંક nb02, અચલ પૂર્ણાંક nb03,
        અચલ પૂર્ણાંક ne00_src, અચલ પૂર્ણાંક ne01_src,
        અચલ પૂર્ણાંક ne10_dst, અચલ પૂર્ણાંક ne11_dst, અચલ પૂર્ણાંક ne12_dst, અચલ પૂર્ણાંક ne13_dst,
        અચલ દશાંશ sf0, અચલ દશાંશ sf1, અચલ દશાંશ sf2, અચલ દશાંશ sf3,
        અચલ દશાંશ pixel_offset) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    અચલ int64_t index = item_ct1.get_local_id(2) +
        item_ct1.get_group(2) * item_ct1.get_local_range(2);
    અચલ int64_t dst_total_elements = ne10_dst * ne11_dst * ne12_dst * ne13_dst;

  જો (index >= dst_total_elements) {
    પાછા;
  }

    અચલ પૂર્ણાંક i10_dst = index % ne10_dst;
    અચલ પૂર્ણાંક i11_dst = (index / ne10_dst) % ne11_dst;
    અચલ પૂર્ણાંક i12_dst = (index / (ne10_dst * ne11_dst)) % ne12_dst;
    અચલ પૂર્ણાંક i13_dst = index / (ne10_dst * ne11_dst * ne12_dst);

    અચલ પૂર્ણાંક i02_src = (પૂર્ણાંક)(i12_dst / sf2);
    અચલ પૂર્ણાંક i03_src = (પૂર્ણાંક)(i13_dst / sf3);

    અચલ દશાંશ y_src_f = ((દશાંશ)i11_dst + pixel_offset) / sf1 - pixel_offset;
    પૂર્ણાંક         y0_src    = (પૂર્ણાંક) sycl::floor((દશાંશ) y_src_f);
    પૂર્ણાંક y1_src    = y0_src + 1;

    y0_src = sycl::max(0, sycl::min(y0_src, ne01_src - 1));
    y1_src = sycl::max(0, sycl::min(y1_src, ne01_src - 1));

    દશાંશ dy = y_src_f - (દશાંશ)y0_src;
    dy       = sycl::max(0.0f, sycl::min(dy, 1.0f));

    દશાંશ x_src_f = ((દશાંશ)i10_dst + pixel_offset) / sf0 - pixel_offset;
    પૂર્ણાંક   x0_src    = (પૂર્ણાંક) sycl::floor(x_src_f);
    પૂર્ણાંક x1_src    = x0_src + 1;

    x0_src = sycl::max(0, sycl::min(x0_src, ne00_src - 1));
    x1_src = sycl::max(0, sycl::min(x1_src, ne00_src - 1));

    દશાંશ dx = x_src_f - (દશાંશ)x0_src;
    dx       = sycl::max(0.0f, sycl::min(dx, 1.0f));

    અચલ દશાંશ* p_a =
        (અચલ દશાંશ*)((અચલ અક્ષર*)x + (int64_t)x0_src * nb00 +
                       (int64_t)y0_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    અચલ દશાંશ* p_b =
        (અચલ દશાંશ*)((અચલ અક્ષર*)x + (int64_t)x1_src * nb00 +
                       (int64_t)y0_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    અચલ દશાંશ* p_c =
        (અચલ દશાંશ*)((અચલ અક્ષર*)x + (int64_t)x0_src * nb00 +
                       (int64_t)y1_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    અચલ દશાંશ* p_d =
        (અચલ દશાંશ*)((અચલ અક્ષર*)x + (int64_t)x1_src * nb00 +
                       (int64_t)y1_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);

    અચલ દશાંશ val_a = *p_a;
    અચલ દશાંશ val_b = *p_b;
    અચલ દશાંશ val_c = *p_c;
    અચલ દશાંશ val_d = *p_d;

    દશાંશ result = val_a * (1.0f - dx) * (1.0f - dy) +
                   val_b * dx * (1.0f - dy) +
                   val_c * (1.0f - dx) * dy +
                   val_d * dx * dy;

    dst[index] = result;
}

// Similar to F.interpolate(..., mode="bilinear", align_corners=False, antialias=True)
// https://github.com/pytorch/pytorch/blob/8871ff29b743948d1225389d5b7068f37b22750b/aten/src/ATen/native/cpu/UpSampleKernel.cpp
સ્થિર શૂન્ય upscale_f32_bilinear_antialias(અચલ દશાંશ * src0,
                                           દશાંશ *       dst,
                                           અચલ પૂર્ણાંક     nb00,
                                           અચલ પૂર્ણાંક     nb01,
                                           અચલ પૂર્ણાંક     nb02,
                                           અચલ પૂર્ણાંક     nb03,
                                           અચલ પૂર્ણાંક     ne00_src,
                                           અચલ પૂર્ણાંક     ne01_src,
                                           અચલ પૂર્ણાંક     ne10_dst,
                                           અચલ પૂર્ણાંક     ne11_dst,
                                           અચલ પૂર્ણાંક     ne12_dst,
                                           અચલ પૂર્ણાંક     ne13_dst,
                                           અચલ દશાંશ   sf0,
                                           અચલ દશાંશ   sf1,
                                           અચલ દશાંશ   sf2,
                                           અચલ દશાંશ   sf3,
                                           અચલ દશાંશ   pixel_offset) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    અચલ int64_t index = item_ct1.get_local_id(2) +
        item_ct1.get_group(2) * item_ct1.get_local_range(2);
    અચલ int64_t dst_total_elements = ne10_dst * ne11_dst * ne12_dst * ne13_dst;

    જો (index >= dst_total_elements) {
        પાછા;
    }

    અચલ પૂર્ણાંક i10_dst = index % ne10_dst;
    અચલ પૂર્ણાંક i11_dst = (index / ne10_dst) % ne11_dst;
    અચલ પૂર્ણાંક i12_dst = (index / (ne10_dst * ne11_dst)) % ne12_dst;
    અચલ પૂર્ણાંક i13_dst = index / (ne10_dst * ne11_dst * ne12_dst);

    અચલ પૂર્ણાંક i02_src = (પૂર્ણાંક)(i12_dst / sf2);
    અચલ પૂર્ણાંક i03_src = (પૂર્ણાંક)(i13_dst / sf3);

    અચલ દશાંશ y = ((દશાંશ)i11_dst + pixel_offset) / sf1;
    અચલ દશાંશ x = ((દશાંશ)i10_dst + pixel_offset) / sf0;

    // support and invscale, minimum 1 pixel for bilinear
    અચલ દશાંશ support1  = sycl::max(1.0f / sf1, 1.0f);
    અચલ દશાંશ invscale1 = 1.0f / support1;
    અચલ દશાંશ support0  = sycl::max(1.0f / sf0, 1.0f);
    અચલ દશાંશ invscale0 = 1.0f / support0;

    // the range of source pixels that contribute
    અચલ int64_t x_min = sycl::max(int64_t(0), int64_t(x - support0 + pixel_offset));
    અચલ int64_t x_max = sycl::min(int64_t(ne00_src), int64_t(x + support0 + pixel_offset));
    અચલ int64_t y_min = sycl::max(int64_t(0), int64_t(y - support1 + pixel_offset));
    અચલ int64_t y_max = sycl::min(int64_t(ne01_src), int64_t(y + support1 + pixel_offset));

    // bilinear filter with antialiasing
    દશાંશ val = 0.0f;
    દશાંશ total_weight = 0.0f;

    auto triangle_filter = [](દશાંશ x) -> દશાંશ {
        પાછા sycl::max(1.0f - sycl::fabs(x), 0.0f);
    };

    માટે (int64_t sy = y_min; sy < y_max; sy++) {
        અચલ દશાંશ weight_y = triangle_filter((sy - y + pixel_offset) * invscale1);

        માટે (int64_t sx = x_min; sx < x_max; sx++) {
            અચલ દશાંશ weight_x = triangle_filter((sx - x + pixel_offset) * invscale0);
            અચલ દશાંશ weight = weight_x * weight_y;

            જો (weight <= 0.0f) {
                ચાલુ;
            }

            અચલ દશાંશ pixel =
                *(અચલ દશાંશ*)((અચલ અક્ષર*)src0 + sx * nb00 + sy * nb01 +
                                i02_src * nb02 + i03_src * nb03);
            val += pixel * weight;
            total_weight += weight;
        }
    }

    જો (total_weight > 0.0f) {
        val /= total_weight;
    }

    dst[index] = val;
}

નામાકાશ bicubic_interpolation {
સ્થિર દશાંશ weight1(દશાંશ x, અચલ દશાંશ &a) { પાછા ((a + 2) * x - (a + 3)) * x * x + 1; }
સ્થિર દશાંશ weight2(દશાંશ x, અચલ દશાંશ &a) { પાછા ((a * x - 5 * a) * x + 8 * a) * x - 4 * a; }

સ્થિર દશાંશ bicubic(દશાંશ p0, દશાંશ p1, દશાંશ p2, દશાંશ p3, દશાંશ x, દશાંશ a) {
    અચલ દશાંશ w0 = weight2(x + 1, a);
    અચલ દશાંશ w1 = weight1(x + 0, a);
    અચલ દશાંશ w2 = weight1(1 - x, a);
    અચલ દશાંશ w3 = weight2(2 - x, a);
    પાછા p0 * w0 + p1 * w1 + p2 * w2 + p3 * w3;
}

}

સ્થિર શૂન્ય upscale_f32_bicubic(અચલ દશાંશ * x, દશાંશ * dst,
        અચલ પૂર્ણાંક nb00, અચલ પૂર્ણાંક nb01, અચલ પૂર્ણાંક nb02, અચલ પૂર્ણાંક nb03,
        અચલ પૂર્ણાંક ne00_src, અચલ પૂર્ણાંક ne01_src,
        અચલ પૂર્ણાંક ne10_dst, અચલ પૂર્ણાંક ne11_dst, અચલ પૂર્ણાંક ne12_dst, અચલ પૂર્ણાંક ne13_dst,
        અચલ દશાંશ sf0, અચલ દશાંશ sf1, અચલ દશાંશ sf2, અચલ દશાંશ sf3,
        અચલ દશાંશ pixel_offset) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    અચલ દશાંશ a = -0.75f;
    using bicubic_interpolation::bicubic;

    અચલ int64_t index = item_ct1.get_local_id(2) +
        item_ct1.get_group(2) * item_ct1.get_local_range(2);
    અચલ int64_t dst_total_elements =
        ne10_dst * ne11_dst * ne12_dst * ne13_dst;

    જો (index >= dst_total_elements) {
        પાછા;
    }

    અચલ પૂર્ણાંક i10_dst = index % ne10_dst;
    અચલ પૂર્ણાંક i11_dst = (index / ne10_dst) % ne11_dst;
    અચલ પૂર્ણાંક i12_dst = (index / (ne10_dst * ne11_dst)) % ne12_dst;
    અચલ પૂર્ણાંક i13_dst = index / (ne10_dst * ne11_dst * ne12_dst);

    અચલ પૂર્ણાંક i02_src = (પૂર્ણાંક)(i12_dst / sf2);
    અચલ પૂર્ણાંક i03_src = (પૂર્ણાંક)(i13_dst / sf3);

    અચલ દશાંશ y_src_f = ((દશાંશ)i11_dst + pixel_offset) / sf1 - pixel_offset;
    અચલ પૂર્ણાંક   y0_src  = (પૂર્ણાંક) sycl::floor((દશાંશ) y_src_f);
    અચલ દશાંશ dy      = y_src_f - (દશાંશ)y0_src;

    અચલ દશાંશ x_src_f = ((દશાંશ)i10_dst + pixel_offset) / sf0 - pixel_offset;
    અચલ પૂર્ણાંક   x0_src  = (પૂર્ણાંક) sycl::floor((દશાંશ) x_src_f);
    અચલ દશાંશ dx      = x_src_f - (દશાંશ)x0_src;

    અચલ અક્ષર * x_base = (અચલ અક્ષર *)x + (int64_t)i02_src * nb02 + (int64_t)i03_src * nb03;

    auto load = [=](પૂર્ણાંક x_off, પૂર્ણાંક y_off) -> દશાંશ {
        પૂર્ણાંક i00_src = sycl::max(0, sycl::min(x0_src + x_off, ne00_src - 1));
        પૂર્ણાંક i01_src = sycl::max(0, sycl::min(y0_src + y_off, ne01_src - 1));
        પાછા *(અચલ દશાંશ *)(x_base + (int64_t)i00_src * nb00 + (int64_t)i01_src * nb01);
    };

    અચલ દશાંશ result = bicubic(
        bicubic(load(-1, -1), load(0, -1), load(1, -1), load(2, -1), dx, a),
        bicubic(load(-1, 0), load(0, 0), load(1, 0), load(2, 0), dx, a),
        bicubic(load(-1, 1), load(0, 1), load(1, 1), load(2, 1), dx, a),
        bicubic(load(-1, 2), load(0, 2), load(1, 2), load(2, 2), dx, a),
        dy,
        a);

    dst[index] = result;
}

સ્થિર શૂન્ય upscale_f32_sycl(અચલ દશાંશ *   x,
                             દશાંશ *         dst,
                             અચલ પૂર્ણાંક       nb00,
                             અચલ પૂર્ણાંક       nb01,
                             અચલ પૂર્ણાંક       nb02,
                             અચલ પૂર્ણાંક       nb03,
                             અચલ પૂર્ણાંક       ne10,
                             અચલ પૂર્ણાંક       ne11,
                             અચલ પૂર્ણાંક       ne12,
                             અચલ પૂર્ણાંક       ne13,
                             અચલ દશાંશ     sf0,
                             અચલ દશાંશ     sf1,
                             અચલ દશાંશ     sf2,
                             અચલ દશાંશ     sf3,
                             dpct::queue_ptr stream) {
    અચલ int64_t dst_size   = ne10 * ne11 * ne12 * ne13;
    અચલ int64_t અંક_blocks = (dst_size + SYCL_UPSCALE_BLOCK_SIZE - 1) / SYCL_UPSCALE_BLOCK_SIZE;

    stream->parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, અંક_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE),
             sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE)),
        [=](sycl::nd_item<3> /*item_ct1*/) {
            upscale_f32(x, dst, nb00, nb01, nb02, nb03, ne10, ne11, ne12, ne13, sf0, sf1, sf2, sf3);
        });
}

સ્થિર શૂન્ય upscale_f32_bilinear_sycl(અચલ દશાંશ *   x,
                                      દશાંશ *         dst,
                                      અચલ પૂર્ણાંક       nb00,
                                      અચલ પૂર્ણાંક       nb01,
                                      અચલ પૂર્ણાંક       nb02,
                                      અચલ પૂર્ણાંક       nb03,
                                      અચલ પૂર્ણાંક       ne00_src,
                                      અચલ પૂર્ણાંક       ne01_src,
                                      અચલ પૂર્ણાંક       ne10_dst,
                                      અચલ પૂર્ણાંક       ne11_dst,
                                      અચલ પૂર્ણાંક       ne12_dst,
                                      અચલ પૂર્ણાંક       ne13_dst,
                                      અચલ દશાંશ     sf0,
                                      અચલ દશાંશ     sf1,
                                      અચલ દશાંશ     sf2,
                                      અચલ દશાંશ     sf3,
                                      અચલ દશાંશ     pixel_offset,
                                      તાર્કિક            antialias,
                                      dpct::queue_ptr stream) {
    અચલ int64_t dst_size   = ne10_dst * ne11_dst * ne12_dst * ne13_dst;
    અચલ int64_t અંક_blocks = (dst_size + SYCL_UPSCALE_BLOCK_SIZE - 1) / SYCL_UPSCALE_BLOCK_SIZE;

    જો (antialias) {
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, અંક_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE),
                sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE)),
            [=](sycl::nd_item<3> /*item_ct1*/) {
                upscale_f32_bilinear_antialias(
                    x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst,
                    ne12_dst, ne13_dst, sf0, sf1, sf2, sf3, pixel_offset);
            });
    } નહીંતર {
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, અંક_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE),
                sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE)),
            [=](sycl::nd_item<3> /*item_ct1*/) {
                upscale_f32_bilinear(
                    x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst, ne12_dst,
                    ne13_dst, sf0, sf1, sf2, sf3, pixel_offset);
            });
    }
}

સ્થિર શૂન્ય upscale_f32_bicubic_sycl(અચલ દશાંશ *   x,
                                     દશાંશ *         dst,
                                     અચલ પૂર્ણાંક       nb00,
                                     અચલ પૂર્ણાંક       nb01,
                                     અચલ પૂર્ણાંક       nb02,
                                     અચલ પૂર્ણાંક       nb03,
                                     અચલ પૂર્ણાંક       ne00_src,
                                     અચલ પૂર્ણાંક       ne01_src,
                                     અચલ પૂર્ણાંક       ne10_dst,
                                     અચલ પૂર્ણાંક       ne11_dst,
                                     અચલ પૂર્ણાંક       ne12_dst,
                                     અચલ પૂર્ણાંક       ne13_dst,
                                     અચલ દશાંશ     sf0,
                                     અચલ દશાંશ     sf1,
                                     અચલ દશાંશ     sf2,
                                     અચલ દશાંશ     sf3,
                                     અચલ દશાંશ     pixel_offset,
                                     dpct::queue_ptr stream) {
    અચલ int64_t dst_size   = ne10_dst * ne11_dst * ne12_dst * ne13_dst;
    અચલ int64_t અંક_blocks = (dst_size + SYCL_UPSCALE_BLOCK_SIZE - 1) / SYCL_UPSCALE_BLOCK_SIZE;

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, 1, અંક_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE),
                    sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE)),
                [=](sycl::nd_item<3> /*item_ct1*/) {
                    upscale_f32_bicubic(
                        x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst,
                        ne12_dst, ne13_dst, sf0, sf1, sf2, sf3, pixel_offset);
                });
        });
    }
}

સ્થિર શૂન્ય ggml_sycl_op_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ દશાંશ * src0_d = (અચલ દશાંશ *)src0->data;
    દશાંશ * dst_d = (દશાંશ *)dst->data;
    dpct::queue_ptr     stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    અચલ પૂર્ણાંક mode_flags = dst->op_params[0];
    અચલ ggml_scale_mode mode = (ggml_scale_mode)(mode_flags & 0xFF);

    દશાંશ sf0 = (દશાંશ)dst->ne[0]/src0->ne[0];
    દશાંશ sf1 = (દશાંશ)dst->ne[1]/src0->ne[1];
    દશાંશ sf2 = (દશાંશ)dst->ne[2]/src0->ne[2];
    અચલ દશાંશ sf3 = (દશાંશ)dst->ne[3]/src0->ne[3];

    દશાંશ pixel_offset = 0.5f;
    જો (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        sf0 = dst->ne[0] > 1 && src0->ne[0] > 1
            ? (દશાંશ)(dst->ne[0] - 1) / (src0->ne[0] - 1)
            : sf0;
        sf1 = dst->ne[1] > 1 && src0->ne[1] > 1
            ? (દશાંશ)(dst->ne[1] - 1) / (src0->ne[1] - 1)
            : sf1;
        pixel_offset = 0.0f;
    }

    જો (mode == GGML_SCALE_MODE_NEAREST) {
        upscale_f32_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], sf0, sf1, sf2, sf3, stream);
    } નહીંતર જો (mode == GGML_SCALE_MODE_BILINEAR) {
        અચલ તાર્કિક antialias = (mode_flags & GGML_SCALE_FLAG_ANTIALIAS);
        upscale_f32_bilinear_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src0->ne[0], src0->ne[1], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            sf0, sf1, sf2, sf3, pixel_offset, antialias, stream);
    } નહીંતર જો (mode == GGML_SCALE_MODE_BICUBIC) {
        upscale_f32_bicubic_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src0->ne[0], src0->ne[1], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            sf0, sf1, sf2, sf3, pixel_offset, stream);
    }
}

શૂન્ય ggml_sycl_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_upscale(ctx, dst);
}
