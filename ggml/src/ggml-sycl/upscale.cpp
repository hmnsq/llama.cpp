#ضمن "upscale.hpp"

ساكن فراغ upscale_f32(ثابت عائم * x, عائم * dst,
        ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
        ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح ne13,
        ثابت عائم sf0, ثابت عائم sf1, ثابت عائم sf2, ثابت عائم sf3) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    صحيح  index    = item_ct1.get_local_id(2) + item_ct1.get_group(2) * item_ct1.get_local_range(2);
    إذا (index >= ne10 * ne11 * ne12 * ne13) {
        أرجع;
    }

    صحيح i10 = index % ne10;
    صحيح i11 = (index / ne10) % ne11;
    صحيح i12 = (index / (ne10 * ne11)) % ne12;
    صحيح i13 = (index / (ne10 * ne11 * ne12)) % ne13;

    صحيح i00 = i10 / sf0;
    صحيح i01 = i11 / sf1;
    صحيح i02 = i12 / sf2;
    صحيح i03 = i13 / sf3;

    dst[index] = *((ثابت عائم*)((ثابت حرف*)x + i03 * nb03 + i02 * nb02 +
                                  i01 * nb01 + i00 * nb00));
}

ساكن فراغ upscale_f32_bilinear(ثابت عائم * x, عائم * dst,
        ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
        ثابت صحيح ne00_src, ثابت صحيح ne01_src,
        ثابت صحيح ne10_dst, ثابت صحيح ne11_dst, ثابت صحيح ne12_dst, ثابت صحيح ne13_dst,
        ثابت عائم sf0, ثابت عائم sf1, ثابت عائم sf2, ثابت عائم sf3,
        ثابت عائم pixel_offset) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ثابت int64_t index = item_ct1.get_local_id(2) +
        item_ct1.get_group(2) * item_ct1.get_local_range(2);
    ثابت int64_t dst_total_elements = ne10_dst * ne11_dst * ne12_dst * ne13_dst;

  إذا (index >= dst_total_elements) {
    أرجع;
  }

    ثابت صحيح i10_dst = index % ne10_dst;
    ثابت صحيح i11_dst = (index / ne10_dst) % ne11_dst;
    ثابت صحيح i12_dst = (index / (ne10_dst * ne11_dst)) % ne12_dst;
    ثابت صحيح i13_dst = index / (ne10_dst * ne11_dst * ne12_dst);

    ثابت صحيح i02_src = (صحيح)(i12_dst / sf2);
    ثابت صحيح i03_src = (صحيح)(i13_dst / sf3);

    ثابت عائم y_src_f = ((عائم)i11_dst + pixel_offset) / sf1 - pixel_offset;
    صحيح         y0_src    = (صحيح) sycl::floor((عائم) y_src_f);
    صحيح y1_src    = y0_src + 1;

    y0_src = sycl::max(0, sycl::min(y0_src, ne01_src - 1));
    y1_src = sycl::max(0, sycl::min(y1_src, ne01_src - 1));

    عائم dy = y_src_f - (عائم)y0_src;
    dy       = sycl::max(0.0f, sycl::min(dy, 1.0f));

    عائم x_src_f = ((عائم)i10_dst + pixel_offset) / sf0 - pixel_offset;
    صحيح   x0_src    = (صحيح) sycl::floor(x_src_f);
    صحيح x1_src    = x0_src + 1;

    x0_src = sycl::max(0, sycl::min(x0_src, ne00_src - 1));
    x1_src = sycl::max(0, sycl::min(x1_src, ne00_src - 1));

    عائم dx = x_src_f - (عائم)x0_src;
    dx       = sycl::max(0.0f, sycl::min(dx, 1.0f));

    ثابت عائم* p_a =
        (ثابت عائم*)((ثابت حرف*)x + (int64_t)x0_src * nb00 +
                       (int64_t)y0_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    ثابت عائم* p_b =
        (ثابت عائم*)((ثابت حرف*)x + (int64_t)x1_src * nb00 +
                       (int64_t)y0_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    ثابت عائم* p_c =
        (ثابت عائم*)((ثابت حرف*)x + (int64_t)x0_src * nb00 +
                       (int64_t)y1_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    ثابت عائم* p_d =
        (ثابت عائم*)((ثابت حرف*)x + (int64_t)x1_src * nb00 +
                       (int64_t)y1_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);

    ثابت عائم val_a = *p_a;
    ثابت عائم val_b = *p_b;
    ثابت عائم val_c = *p_c;
    ثابت عائم val_d = *p_d;

    عائم result = val_a * (1.0f - dx) * (1.0f - dy) +
                   val_b * dx * (1.0f - dy) +
                   val_c * (1.0f - dx) * dy +
                   val_d * dx * dy;

    dst[index] = result;
}

// Similar to F.interpolate(..., mode="bilinear", align_corners=False, antialias=True)
// https://github.com/pytorch/pytorch/blob/8871ff29b743948d1225389d5b7068f37b22750b/aten/src/ATen/native/cpu/UpSampleKernel.cpp
ساكن فراغ upscale_f32_bilinear_antialias(ثابت عائم * src0,
                                           عائم *       dst,
                                           ثابت صحيح     nb00,
                                           ثابت صحيح     nb01,
                                           ثابت صحيح     nb02,
                                           ثابت صحيح     nb03,
                                           ثابت صحيح     ne00_src,
                                           ثابت صحيح     ne01_src,
                                           ثابت صحيح     ne10_dst,
                                           ثابت صحيح     ne11_dst,
                                           ثابت صحيح     ne12_dst,
                                           ثابت صحيح     ne13_dst,
                                           ثابت عائم   sf0,
                                           ثابت عائم   sf1,
                                           ثابت عائم   sf2,
                                           ثابت عائم   sf3,
                                           ثابت عائم   pixel_offset) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ثابت int64_t index = item_ct1.get_local_id(2) +
        item_ct1.get_group(2) * item_ct1.get_local_range(2);
    ثابت int64_t dst_total_elements = ne10_dst * ne11_dst * ne12_dst * ne13_dst;

    إذا (index >= dst_total_elements) {
        أرجع;
    }

    ثابت صحيح i10_dst = index % ne10_dst;
    ثابت صحيح i11_dst = (index / ne10_dst) % ne11_dst;
    ثابت صحيح i12_dst = (index / (ne10_dst * ne11_dst)) % ne12_dst;
    ثابت صحيح i13_dst = index / (ne10_dst * ne11_dst * ne12_dst);

    ثابت صحيح i02_src = (صحيح)(i12_dst / sf2);
    ثابت صحيح i03_src = (صحيح)(i13_dst / sf3);

    ثابت عائم y = ((عائم)i11_dst + pixel_offset) / sf1;
    ثابت عائم x = ((عائم)i10_dst + pixel_offset) / sf0;

    // support and invscale, minimum 1 pixel for bilinear
    ثابت عائم support1  = sycl::max(1.0f / sf1, 1.0f);
    ثابت عائم invscale1 = 1.0f / support1;
    ثابت عائم support0  = sycl::max(1.0f / sf0, 1.0f);
    ثابت عائم invscale0 = 1.0f / support0;

    // the range of source pixels that contribute
    ثابت int64_t x_min = sycl::max(int64_t(0), int64_t(x - support0 + pixel_offset));
    ثابت int64_t x_max = sycl::min(int64_t(ne00_src), int64_t(x + support0 + pixel_offset));
    ثابت int64_t y_min = sycl::max(int64_t(0), int64_t(y - support1 + pixel_offset));
    ثابت int64_t y_max = sycl::min(int64_t(ne01_src), int64_t(y + support1 + pixel_offset));

    // bilinear filter with antialiasing
    عائم val = 0.0f;
    عائم total_weight = 0.0f;

    auto triangle_filter = [](عائم x) -> عائم {
        أرجع sycl::max(1.0f - sycl::fabs(x), 0.0f);
    };

    لكل (int64_t sy = y_min; sy < y_max; sy++) {
        ثابت عائم weight_y = triangle_filter((sy - y + pixel_offset) * invscale1);

        لكل (int64_t sx = x_min; sx < x_max; sx++) {
            ثابت عائم weight_x = triangle_filter((sx - x + pixel_offset) * invscale0);
            ثابت عائم weight = weight_x * weight_y;

            إذا (weight <= 0.0f) {
                استمر;
            }

            ثابت عائم pixel =
                *(ثابت عائم*)((ثابت حرف*)src0 + sx * nb00 + sy * nb01 +
                                i02_src * nb02 + i03_src * nb03);
            val += pixel * weight;
            total_weight += weight;
        }
    }

    إذا (total_weight > 0.0f) {
        val /= total_weight;
    }

    dst[index] = val;
}

نطاق bicubic_interpolation {
ساكن عائم weight1(عائم x, ثابت عائم &a) { أرجع ((a + 2) * x - (a + 3)) * x * x + 1; }
ساكن عائم weight2(عائم x, ثابت عائم &a) { أرجع ((a * x - 5 * a) * x + 8 * a) * x - 4 * a; }

ساكن عائم bicubic(عائم p0, عائم p1, عائم p2, عائم p3, عائم x, عائم a) {
    ثابت عائم w0 = weight2(x + 1, a);
    ثابت عائم w1 = weight1(x + 0, a);
    ثابت عائم w2 = weight1(1 - x, a);
    ثابت عائم w3 = weight2(2 - x, a);
    أرجع p0 * w0 + p1 * w1 + p2 * w2 + p3 * w3;
}

}

ساكن فراغ upscale_f32_bicubic(ثابت عائم * x, عائم * dst,
        ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
        ثابت صحيح ne00_src, ثابت صحيح ne01_src,
        ثابت صحيح ne10_dst, ثابت صحيح ne11_dst, ثابت صحيح ne12_dst, ثابت صحيح ne13_dst,
        ثابت عائم sf0, ثابت عائم sf1, ثابت عائم sf2, ثابت عائم sf3,
        ثابت عائم pixel_offset) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ثابت عائم a = -0.75f;
    using bicubic_interpolation::bicubic;

    ثابت int64_t index = item_ct1.get_local_id(2) +
        item_ct1.get_group(2) * item_ct1.get_local_range(2);
    ثابت int64_t dst_total_elements =
        ne10_dst * ne11_dst * ne12_dst * ne13_dst;

    إذا (index >= dst_total_elements) {
        أرجع;
    }

    ثابت صحيح i10_dst = index % ne10_dst;
    ثابت صحيح i11_dst = (index / ne10_dst) % ne11_dst;
    ثابت صحيح i12_dst = (index / (ne10_dst * ne11_dst)) % ne12_dst;
    ثابت صحيح i13_dst = index / (ne10_dst * ne11_dst * ne12_dst);

    ثابت صحيح i02_src = (صحيح)(i12_dst / sf2);
    ثابت صحيح i03_src = (صحيح)(i13_dst / sf3);

    ثابت عائم y_src_f = ((عائم)i11_dst + pixel_offset) / sf1 - pixel_offset;
    ثابت صحيح   y0_src  = (صحيح) sycl::floor((عائم) y_src_f);
    ثابت عائم dy      = y_src_f - (عائم)y0_src;

    ثابت عائم x_src_f = ((عائم)i10_dst + pixel_offset) / sf0 - pixel_offset;
    ثابت صحيح   x0_src  = (صحيح) sycl::floor((عائم) x_src_f);
    ثابت عائم dx      = x_src_f - (عائم)x0_src;

    ثابت حرف * x_base = (ثابت حرف *)x + (int64_t)i02_src * nb02 + (int64_t)i03_src * nb03;

    auto load = [=](صحيح x_off, صحيح y_off) -> عائم {
        صحيح i00_src = sycl::max(0, sycl::min(x0_src + x_off, ne00_src - 1));
        صحيح i01_src = sycl::max(0, sycl::min(y0_src + y_off, ne01_src - 1));
        أرجع *(ثابت عائم *)(x_base + (int64_t)i00_src * nb00 + (int64_t)i01_src * nb01);
    };

    ثابت عائم result = bicubic(
        bicubic(load(-1, -1), load(0, -1), load(1, -1), load(2, -1), dx, a),
        bicubic(load(-1, 0), load(0, 0), load(1, 0), load(2, 0), dx, a),
        bicubic(load(-1, 1), load(0, 1), load(1, 1), load(2, 1), dx, a),
        bicubic(load(-1, 2), load(0, 2), load(1, 2), load(2, 2), dx, a),
        dy,
        a);

    dst[index] = result;
}

ساكن فراغ upscale_f32_sycl(ثابت عائم *   x,
                             عائم *         dst,
                             ثابت صحيح       nb00,
                             ثابت صحيح       nb01,
                             ثابت صحيح       nb02,
                             ثابت صحيح       nb03,
                             ثابت صحيح       ne10,
                             ثابت صحيح       ne11,
                             ثابت صحيح       ne12,
                             ثابت صحيح       ne13,
                             ثابت عائم     sf0,
                             ثابت عائم     sf1,
                             ثابت عائم     sf2,
                             ثابت عائم     sf3,
                             dpct::queue_ptr stream) {
    ثابت int64_t dst_size   = ne10 * ne11 * ne12 * ne13;
    ثابت int64_t اڠک_blocks = (dst_size + SYCL_UPSCALE_BLOCK_SIZE - 1) / SYCL_UPSCALE_BLOCK_SIZE;

    stream->parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE),
             sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE)),
        [=](sycl::nd_item<3> /*item_ct1*/) {
            upscale_f32(x, dst, nb00, nb01, nb02, nb03, ne10, ne11, ne12, ne13, sf0, sf1, sf2, sf3);
        });
}

ساكن فراغ upscale_f32_bilinear_sycl(ثابت عائم *   x,
                                      عائم *         dst,
                                      ثابت صحيح       nb00,
                                      ثابت صحيح       nb01,
                                      ثابت صحيح       nb02,
                                      ثابت صحيح       nb03,
                                      ثابت صحيح       ne00_src,
                                      ثابت صحيح       ne01_src,
                                      ثابت صحيح       ne10_dst,
                                      ثابت صحيح       ne11_dst,
                                      ثابت صحيح       ne12_dst,
                                      ثابت صحيح       ne13_dst,
                                      ثابت عائم     sf0,
                                      ثابت عائم     sf1,
                                      ثابت عائم     sf2,
                                      ثابت عائم     sf3,
                                      ثابت عائم     pixel_offset,
                                      منطقي            antialias,
                                      dpct::queue_ptr stream) {
    ثابت int64_t dst_size   = ne10_dst * ne11_dst * ne12_dst * ne13_dst;
    ثابت int64_t اڠک_blocks = (dst_size + SYCL_UPSCALE_BLOCK_SIZE - 1) / SYCL_UPSCALE_BLOCK_SIZE;

    إذا (antialias) {
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE),
                sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE)),
            [=](sycl::nd_item<3> /*item_ct1*/) {
                upscale_f32_bilinear_antialias(
                    x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst,
                    ne12_dst, ne13_dst, sf0, sf1, sf2, sf3, pixel_offset);
            });
    } وإلا {
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE),
                sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE)),
            [=](sycl::nd_item<3> /*item_ct1*/) {
                upscale_f32_bilinear(
                    x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst, ne12_dst,
                    ne13_dst, sf0, sf1, sf2, sf3, pixel_offset);
            });
    }
}

ساكن فراغ upscale_f32_bicubic_sycl(ثابت عائم *   x,
                                     عائم *         dst,
                                     ثابت صحيح       nb00,
                                     ثابت صحيح       nb01,
                                     ثابت صحيح       nb02,
                                     ثابت صحيح       nb03,
                                     ثابت صحيح       ne00_src,
                                     ثابت صحيح       ne01_src,
                                     ثابت صحيح       ne10_dst,
                                     ثابت صحيح       ne11_dst,
                                     ثابت صحيح       ne12_dst,
                                     ثابت صحيح       ne13_dst,
                                     ثابت عائم     sf0,
                                     ثابت عائم     sf1,
                                     ثابت عائم     sf2,
                                     ثابت عائم     sf3,
                                     ثابت عائم     pixel_offset,
                                     dpct::queue_ptr stream) {
    ثابت int64_t dst_size   = ne10_dst * ne11_dst * ne12_dst * ne13_dst;
    ثابت int64_t اڠک_blocks = (dst_size + SYCL_UPSCALE_BLOCK_SIZE - 1) / SYCL_UPSCALE_BLOCK_SIZE;

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE),
                    sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE)),
                [=](sycl::nd_item<3> /*item_ct1*/) {
                    upscale_f32_bicubic(
                        x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst,
                        ne12_dst, ne13_dst, sf0, sf1, sf2, sf3, pixel_offset);
                });
        });
    }
}

ساكن فراغ ggml_sycl_op_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت عائم * src0_d = (ثابت عائم *)src0->data;
    عائم * dst_d = (عائم *)dst->data;
    dpct::queue_ptr     stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    ثابت صحيح mode_flags = dst->op_params[0];
    ثابت ggml_scale_mode mode = (ggml_scale_mode)(mode_flags & 0xFF);

    عائم sf0 = (عائم)dst->ne[0]/src0->ne[0];
    عائم sf1 = (عائم)dst->ne[1]/src0->ne[1];
    عائم sf2 = (عائم)dst->ne[2]/src0->ne[2];
    ثابت عائم sf3 = (عائم)dst->ne[3]/src0->ne[3];

    عائم pixel_offset = 0.5f;
    إذا (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        sf0 = dst->ne[0] > 1 && src0->ne[0] > 1
            ? (عائم)(dst->ne[0] - 1) / (src0->ne[0] - 1)
            : sf0;
        sf1 = dst->ne[1] > 1 && src0->ne[1] > 1
            ? (عائم)(dst->ne[1] - 1) / (src0->ne[1] - 1)
            : sf1;
        pixel_offset = 0.0f;
    }

    إذا (mode == GGML_SCALE_MODE_NEAREST) {
        upscale_f32_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], sf0, sf1, sf2, sf3, stream);
    } وإلا إذا (mode == GGML_SCALE_MODE_BILINEAR) {
        ثابت منطقي antialias = (mode_flags & GGML_SCALE_FLAG_ANTIALIAS);
        upscale_f32_bilinear_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src0->ne[0], src0->ne[1], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            sf0, sf1, sf2, sf3, pixel_offset, antialias, stream);
    } وإلا إذا (mode == GGML_SCALE_MODE_BICUBIC) {
        upscale_f32_bicubic_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src0->ne[0], src0->ne[1], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            sf0, sf1, sf2, sf3, pixel_offset, stream);
    }
}

فراغ ggml_sycl_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_upscale(ctx, dst);
}
