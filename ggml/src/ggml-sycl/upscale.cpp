#شامل "upscale.hpp"

ایستا تهی upscale_f32(ثابت اعشاری * x, اعشاری * dst,
        ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
        ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح ne13,
        ثابت اعشاری sf0, ثابت اعشاری sf1, ثابت اعشاری sf2, ثابت اعشاری sf3) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    صحیح  index    = item_ct1.get_local_id(2) + item_ct1.get_group(2) * item_ct1.get_local_range(2);
    اگر (index >= ne10 * ne11 * ne12 * ne13) {
        بازگشت;
    }

    صحیح i10 = index % ne10;
    صحیح i11 = (index / ne10) % ne11;
    صحیح i12 = (index / (ne10 * ne11)) % ne12;
    صحیح i13 = (index / (ne10 * ne11 * ne12)) % ne13;

    صحیح i00 = i10 / sf0;
    صحیح i01 = i11 / sf1;
    صحیح i02 = i12 / sf2;
    صحیح i03 = i13 / sf3;

    dst[index] = *((ثابت اعشاری*)((ثابت نویسه*)x + i03 * nb03 + i02 * nb02 +
                                  i01 * nb01 + i00 * nb00));
}

ایستا تهی upscale_f32_bilinear(ثابت اعشاری * x, اعشاری * dst,
        ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
        ثابت صحیح ne00_src, ثابت صحیح ne01_src,
        ثابت صحیح ne10_dst, ثابت صحیح ne11_dst, ثابت صحیح ne12_dst, ثابت صحیح ne13_dst,
        ثابت اعشاری sf0, ثابت اعشاری sf1, ثابت اعشاری sf2, ثابت اعشاری sf3,
        ثابت اعشاری pixel_offset) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ثابت int64_t index = item_ct1.get_local_id(2) +
        item_ct1.get_group(2) * item_ct1.get_local_range(2);
    ثابت int64_t dst_total_elements = ne10_dst * ne11_dst * ne12_dst * ne13_dst;

  اگر (index >= dst_total_elements) {
    بازگشت;
  }

    ثابت صحیح i10_dst = index % ne10_dst;
    ثابت صحیح i11_dst = (index / ne10_dst) % ne11_dst;
    ثابت صحیح i12_dst = (index / (ne10_dst * ne11_dst)) % ne12_dst;
    ثابت صحیح i13_dst = index / (ne10_dst * ne11_dst * ne12_dst);

    ثابت صحیح i02_src = (صحیح)(i12_dst / sf2);
    ثابت صحیح i03_src = (صحیح)(i13_dst / sf3);

    ثابت اعشاری y_src_f = ((اعشاری)i11_dst + pixel_offset) / sf1 - pixel_offset;
    صحیح         y0_src    = (صحیح) sycl::floor((اعشاری) y_src_f);
    صحیح y1_src    = y0_src + 1;

    y0_src = sycl::max(0, sycl::min(y0_src, ne01_src - 1));
    y1_src = sycl::max(0, sycl::min(y1_src, ne01_src - 1));

    اعشاری dy = y_src_f - (اعشاری)y0_src;
    dy       = sycl::max(0.0f, sycl::min(dy, 1.0f));

    اعشاری x_src_f = ((اعشاری)i10_dst + pixel_offset) / sf0 - pixel_offset;
    صحیح   x0_src    = (صحیح) sycl::floor(x_src_f);
    صحیح x1_src    = x0_src + 1;

    x0_src = sycl::max(0, sycl::min(x0_src, ne00_src - 1));
    x1_src = sycl::max(0, sycl::min(x1_src, ne00_src - 1));

    اعشاری dx = x_src_f - (اعشاری)x0_src;
    dx       = sycl::max(0.0f, sycl::min(dx, 1.0f));

    ثابت اعشاری* p_a =
        (ثابت اعشاری*)((ثابت نویسه*)x + (int64_t)x0_src * nb00 +
                       (int64_t)y0_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    ثابت اعشاری* p_b =
        (ثابت اعشاری*)((ثابت نویسه*)x + (int64_t)x1_src * nb00 +
                       (int64_t)y0_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    ثابت اعشاری* p_c =
        (ثابت اعشاری*)((ثابت نویسه*)x + (int64_t)x0_src * nb00 +
                       (int64_t)y1_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    ثابت اعشاری* p_d =
        (ثابت اعشاری*)((ثابت نویسه*)x + (int64_t)x1_src * nb00 +
                       (int64_t)y1_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);

    ثابت اعشاری val_a = *p_a;
    ثابت اعشاری val_b = *p_b;
    ثابت اعشاری val_c = *p_c;
    ثابت اعشاری val_d = *p_d;

    اعشاری result = val_a * (1.0f - dx) * (1.0f - dy) +
                   val_b * dx * (1.0f - dy) +
                   val_c * (1.0f - dx) * dy +
                   val_d * dx * dy;

    dst[index] = result;
}

// Similar to F.interpolate(..., mode="bilinear", align_corners=False, antialias=True)
// https://github.com/pytorch/pytorch/blob/8871ff29b743948d1225389d5b7068f37b22750b/aten/src/ATen/native/cpu/UpSampleKernel.cpp
ایستا تهی upscale_f32_bilinear_antialias(ثابت اعشاری * src0,
                                           اعشاری *       dst,
                                           ثابت صحیح     nb00,
                                           ثابت صحیح     nb01,
                                           ثابت صحیح     nb02,
                                           ثابت صحیح     nb03,
                                           ثابت صحیح     ne00_src,
                                           ثابت صحیح     ne01_src,
                                           ثابت صحیح     ne10_dst,
                                           ثابت صحیح     ne11_dst,
                                           ثابت صحیح     ne12_dst,
                                           ثابت صحیح     ne13_dst,
                                           ثابت اعشاری   sf0,
                                           ثابت اعشاری   sf1,
                                           ثابت اعشاری   sf2,
                                           ثابت اعشاری   sf3,
                                           ثابت اعشاری   pixel_offset) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ثابت int64_t index = item_ct1.get_local_id(2) +
        item_ct1.get_group(2) * item_ct1.get_local_range(2);
    ثابت int64_t dst_total_elements = ne10_dst * ne11_dst * ne12_dst * ne13_dst;

    اگر (index >= dst_total_elements) {
        بازگشت;
    }

    ثابت صحیح i10_dst = index % ne10_dst;
    ثابت صحیح i11_dst = (index / ne10_dst) % ne11_dst;
    ثابت صحیح i12_dst = (index / (ne10_dst * ne11_dst)) % ne12_dst;
    ثابت صحیح i13_dst = index / (ne10_dst * ne11_dst * ne12_dst);

    ثابت صحیح i02_src = (صحیح)(i12_dst / sf2);
    ثابت صحیح i03_src = (صحیح)(i13_dst / sf3);

    ثابت اعشاری y = ((اعشاری)i11_dst + pixel_offset) / sf1;
    ثابت اعشاری x = ((اعشاری)i10_dst + pixel_offset) / sf0;

    // support and invscale, minimum 1 pixel for bilinear
    ثابت اعشاری support1  = sycl::max(1.0f / sf1, 1.0f);
    ثابت اعشاری invscale1 = 1.0f / support1;
    ثابت اعشاری support0  = sycl::max(1.0f / sf0, 1.0f);
    ثابت اعشاری invscale0 = 1.0f / support0;

    // the range of source pixels that contribute
    ثابت int64_t x_min = sycl::max(int64_t(0), int64_t(x - support0 + pixel_offset));
    ثابت int64_t x_max = sycl::min(int64_t(ne00_src), int64_t(x + support0 + pixel_offset));
    ثابت int64_t y_min = sycl::max(int64_t(0), int64_t(y - support1 + pixel_offset));
    ثابت int64_t y_max = sycl::min(int64_t(ne01_src), int64_t(y + support1 + pixel_offset));

    // bilinear filter with antialiasing
    اعشاری val = 0.0f;
    اعشاری total_weight = 0.0f;

    auto triangle_filter = [](اعشاری x) -> اعشاری {
        بازگشت sycl::max(1.0f - sycl::fabs(x), 0.0f);
    };

    برای (int64_t sy = y_min; sy < y_max; sy++) {
        ثابت اعشاری weight_y = triangle_filter((sy - y + pixel_offset) * invscale1);

        برای (int64_t sx = x_min; sx < x_max; sx++) {
            ثابت اعشاری weight_x = triangle_filter((sx - x + pixel_offset) * invscale0);
            ثابت اعشاری weight = weight_x * weight_y;

            اگر (weight <= 0.0f) {
                ادامه;
            }

            ثابت اعشاری pixel =
                *(ثابت اعشاری*)((ثابت نویسه*)src0 + sx * nb00 + sy * nb01 +
                                i02_src * nb02 + i03_src * nb03);
            val += pixel * weight;
            total_weight += weight;
        }
    }

    اگر (total_weight > 0.0f) {
        val /= total_weight;
    }

    dst[index] = val;
}

فضاینام bicubic_interpolation {
ایستا اعشاری weight1(اعشاری x, ثابت اعشاری &a) { بازگشت ((a + 2) * x - (a + 3)) * x * x + 1; }
ایستا اعشاری weight2(اعشاری x, ثابت اعشاری &a) { بازگشت ((a * x - 5 * a) * x + 8 * a) * x - 4 * a; }

ایستا اعشاری bicubic(اعشاری p0, اعشاری p1, اعشاری p2, اعشاری p3, اعشاری x, اعشاری a) {
    ثابت اعشاری w0 = weight2(x + 1, a);
    ثابت اعشاری w1 = weight1(x + 0, a);
    ثابت اعشاری w2 = weight1(1 - x, a);
    ثابت اعشاری w3 = weight2(2 - x, a);
    بازگشت p0 * w0 + p1 * w1 + p2 * w2 + p3 * w3;
}

}

ایستا تهی upscale_f32_bicubic(ثابت اعشاری * x, اعشاری * dst,
        ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
        ثابت صحیح ne00_src, ثابت صحیح ne01_src,
        ثابت صحیح ne10_dst, ثابت صحیح ne11_dst, ثابت صحیح ne12_dst, ثابت صحیح ne13_dst,
        ثابت اعشاری sf0, ثابت اعشاری sf1, ثابت اعشاری sf2, ثابت اعشاری sf3,
        ثابت اعشاری pixel_offset) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ثابت اعشاری a = -0.75f;
    using bicubic_interpolation::bicubic;

    ثابت int64_t index = item_ct1.get_local_id(2) +
        item_ct1.get_group(2) * item_ct1.get_local_range(2);
    ثابت int64_t dst_total_elements =
        ne10_dst * ne11_dst * ne12_dst * ne13_dst;

    اگر (index >= dst_total_elements) {
        بازگشت;
    }

    ثابت صحیح i10_dst = index % ne10_dst;
    ثابت صحیح i11_dst = (index / ne10_dst) % ne11_dst;
    ثابت صحیح i12_dst = (index / (ne10_dst * ne11_dst)) % ne12_dst;
    ثابت صحیح i13_dst = index / (ne10_dst * ne11_dst * ne12_dst);

    ثابت صحیح i02_src = (صحیح)(i12_dst / sf2);
    ثابت صحیح i03_src = (صحیح)(i13_dst / sf3);

    ثابت اعشاری y_src_f = ((اعشاری)i11_dst + pixel_offset) / sf1 - pixel_offset;
    ثابت صحیح   y0_src  = (صحیح) sycl::floor((اعشاری) y_src_f);
    ثابت اعشاری dy      = y_src_f - (اعشاری)y0_src;

    ثابت اعشاری x_src_f = ((اعشاری)i10_dst + pixel_offset) / sf0 - pixel_offset;
    ثابت صحیح   x0_src  = (صحیح) sycl::floor((اعشاری) x_src_f);
    ثابت اعشاری dx      = x_src_f - (اعشاری)x0_src;

    ثابت نویسه * x_base = (ثابت نویسه *)x + (int64_t)i02_src * nb02 + (int64_t)i03_src * nb03;

    auto load = [=](صحیح x_off, صحیح y_off) -> اعشاری {
        صحیح i00_src = sycl::max(0, sycl::min(x0_src + x_off, ne00_src - 1));
        صحیح i01_src = sycl::max(0, sycl::min(y0_src + y_off, ne01_src - 1));
        بازگشت *(ثابت اعشاری *)(x_base + (int64_t)i00_src * nb00 + (int64_t)i01_src * nb01);
    };

    ثابت اعشاری result = bicubic(
        bicubic(load(-1, -1), load(0, -1), load(1, -1), load(2, -1), dx, a),
        bicubic(load(-1, 0), load(0, 0), load(1, 0), load(2, 0), dx, a),
        bicubic(load(-1, 1), load(0, 1), load(1, 1), load(2, 1), dx, a),
        bicubic(load(-1, 2), load(0, 2), load(1, 2), load(2, 2), dx, a),
        dy,
        a);

    dst[index] = result;
}

ایستا تهی upscale_f32_sycl(ثابت اعشاری *   x,
                             اعشاری *         dst,
                             ثابت صحیح       nb00,
                             ثابت صحیح       nb01,
                             ثابت صحیح       nb02,
                             ثابت صحیح       nb03,
                             ثابت صحیح       ne10,
                             ثابت صحیح       ne11,
                             ثابت صحیح       ne12,
                             ثابت صحیح       ne13,
                             ثابت اعشاری     sf0,
                             ثابت اعشاری     sf1,
                             ثابت اعشاری     sf2,
                             ثابت اعشاری     sf3,
                             dpct::queue_ptr stream) {
    ثابت int64_t dst_size   = ne10 * ne11 * ne12 * ne13;
    ثابت int64_t رقم_blocks = (dst_size + SYCL_UPSCALE_BLOCK_SIZE - 1) / SYCL_UPSCALE_BLOCK_SIZE;

    stream->parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, رقم_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE),
             sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE)),
        [=](sycl::nd_item<3> /*item_ct1*/) {
            upscale_f32(x, dst, nb00, nb01, nb02, nb03, ne10, ne11, ne12, ne13, sf0, sf1, sf2, sf3);
        });
}

ایستا تهی upscale_f32_bilinear_sycl(ثابت اعشاری *   x,
                                      اعشاری *         dst,
                                      ثابت صحیح       nb00,
                                      ثابت صحیح       nb01,
                                      ثابت صحیح       nb02,
                                      ثابت صحیح       nb03,
                                      ثابت صحیح       ne00_src,
                                      ثابت صحیح       ne01_src,
                                      ثابت صحیح       ne10_dst,
                                      ثابت صحیح       ne11_dst,
                                      ثابت صحیح       ne12_dst,
                                      ثابت صحیح       ne13_dst,
                                      ثابت اعشاری     sf0,
                                      ثابت اعشاری     sf1,
                                      ثابت اعشاری     sf2,
                                      ثابت اعشاری     sf3,
                                      ثابت اعشاری     pixel_offset,
                                      منطقی            antialias,
                                      dpct::queue_ptr stream) {
    ثابت int64_t dst_size   = ne10_dst * ne11_dst * ne12_dst * ne13_dst;
    ثابت int64_t رقم_blocks = (dst_size + SYCL_UPSCALE_BLOCK_SIZE - 1) / SYCL_UPSCALE_BLOCK_SIZE;

    اگر (antialias) {
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, رقم_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE),
                sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE)),
            [=](sycl::nd_item<3> /*item_ct1*/) {
                upscale_f32_bilinear_antialias(
                    x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst,
                    ne12_dst, ne13_dst, sf0, sf1, sf2, sf3, pixel_offset);
            });
    } وگرنه {
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, رقم_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE),
                sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE)),
            [=](sycl::nd_item<3> /*item_ct1*/) {
                upscale_f32_bilinear(
                    x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst, ne12_dst,
                    ne13_dst, sf0, sf1, sf2, sf3, pixel_offset);
            });
    }
}

ایستا تهی upscale_f32_bicubic_sycl(ثابت اعشاری *   x,
                                     اعشاری *         dst,
                                     ثابت صحیح       nb00,
                                     ثابت صحیح       nb01,
                                     ثابت صحیح       nb02,
                                     ثابت صحیح       nb03,
                                     ثابت صحیح       ne00_src,
                                     ثابت صحیح       ne01_src,
                                     ثابت صحیح       ne10_dst,
                                     ثابت صحیح       ne11_dst,
                                     ثابت صحیح       ne12_dst,
                                     ثابت صحیح       ne13_dst,
                                     ثابت اعشاری     sf0,
                                     ثابت اعشاری     sf1,
                                     ثابت اعشاری     sf2,
                                     ثابت اعشاری     sf3,
                                     ثابت اعشاری     pixel_offset,
                                     dpct::queue_ptr stream) {
    ثابت int64_t dst_size   = ne10_dst * ne11_dst * ne12_dst * ne13_dst;
    ثابت int64_t رقم_blocks = (dst_size + SYCL_UPSCALE_BLOCK_SIZE - 1) / SYCL_UPSCALE_BLOCK_SIZE;

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, 1, رقم_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE),
                    sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE)),
                [=](sycl::nd_item<3> /*item_ct1*/) {
                    upscale_f32_bicubic(
                        x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst,
                        ne12_dst, ne13_dst, sf0, sf1, sf2, sf3, pixel_offset);
                });
        });
    }
}

ایستا تهی ggml_sycl_op_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت اعشاری * src0_d = (ثابت اعشاری *)src0->data;
    اعشاری * dst_d = (اعشاری *)dst->data;
    dpct::queue_ptr     stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    ثابت صحیح mode_flags = dst->op_params[0];
    ثابت ggml_scale_mode mode = (ggml_scale_mode)(mode_flags & 0xFF);

    اعشاری sf0 = (اعشاری)dst->ne[0]/src0->ne[0];
    اعشاری sf1 = (اعشاری)dst->ne[1]/src0->ne[1];
    اعشاری sf2 = (اعشاری)dst->ne[2]/src0->ne[2];
    ثابت اعشاری sf3 = (اعشاری)dst->ne[3]/src0->ne[3];

    اعشاری pixel_offset = 0.5f;
    اگر (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        sf0 = dst->ne[0] > 1 && src0->ne[0] > 1
            ? (اعشاری)(dst->ne[0] - 1) / (src0->ne[0] - 1)
            : sf0;
        sf1 = dst->ne[1] > 1 && src0->ne[1] > 1
            ? (اعشاری)(dst->ne[1] - 1) / (src0->ne[1] - 1)
            : sf1;
        pixel_offset = 0.0f;
    }

    اگر (mode == GGML_SCALE_MODE_NEAREST) {
        upscale_f32_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], sf0, sf1, sf2, sf3, stream);
    } وگرنه اگر (mode == GGML_SCALE_MODE_BILINEAR) {
        ثابت منطقی antialias = (mode_flags & GGML_SCALE_FLAG_ANTIALIAS);
        upscale_f32_bilinear_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src0->ne[0], src0->ne[1], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            sf0, sf1, sf2, sf3, pixel_offset, antialias, stream);
    } وگرنه اگر (mode == GGML_SCALE_MODE_BICUBIC) {
        upscale_f32_bicubic_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src0->ne[0], src0->ne[1], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            sf0, sf1, sf2, sf3, pixel_offset, stream);
    }
}

تهی ggml_sycl_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_upscale(ctx, dst);
}
