#شامل "upscale.hpp"

ساکن خالی upscale_f32(ثابت اعشاری * x, اعشاری * dst,
        ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
        ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح ne13,
        ثابت اعشاری sf0, ثابت اعشاری sf1, ثابت اعشاری sf2, ثابت اعشاری sf3) {
    خودکار شے_ct1 = sycl::ext::oneapi::this_work_شے::get_nd_شے<3>();
    صحیح  اشاریہ    = شے_ct1.get_local_id(2) + شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(2);
    اگر (اشاریہ >= ne10 * ne11 * ne12 * ne13) {
        واپس;
    }

    صحیح i10 = اشاریہ % ne10;
    صحیح i11 = (اشاریہ / ne10) % ne11;
    صحیح i12 = (اشاریہ / (ne10 * ne11)) % ne12;
    صحیح i13 = (اشاریہ / (ne10 * ne11 * ne12)) % ne13;

    صحیح i00 = i10 / sf0;
    صحیح i01 = i11 / sf1;
    صحیح i02 = i12 / sf2;
    صحیح i03 = i13 / sf3;

    dst[اشاریہ] = *((ثابت اعشاری*)((ثابت حرف*)x + i03 * nb03 + i02 * nb02 +
                                  i01 * nb01 + i00 * nb00));
}

ساکن خالی upscale_f32_bilinear(ثابت اعشاری * x, اعشاری * dst,
        ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
        ثابت صحیح ne00_src, ثابت صحیح ne01_src,
        ثابت صحیح ne10_dst, ثابت صحیح ne11_dst, ثابت صحیح ne12_dst, ثابت صحیح ne13_dst,
        ثابت اعشاری sf0, ثابت اعشاری sf1, ثابت اعشاری sf2, ثابت اعشاری sf3,
        ثابت اعشاری pixel_انحراف) {
    خودکار شے_ct1 = sycl::ext::oneapi::this_work_شے::get_nd_شے<3>();
    ثابت int64_t اشاریہ = شے_ct1.get_local_id(2) +
        شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(2);
    ثابت int64_t dst_کل_elements = ne10_dst * ne11_dst * ne12_dst * ne13_dst;

  اگر (اشاریہ >= dst_کل_elements) {
    واپس;
  }

    ثابت صحیح i10_dst = اشاریہ % ne10_dst;
    ثابت صحیح i11_dst = (اشاریہ / ne10_dst) % ne11_dst;
    ثابت صحیح i12_dst = (اشاریہ / (ne10_dst * ne11_dst)) % ne12_dst;
    ثابت صحیح i13_dst = اشاریہ / (ne10_dst * ne11_dst * ne12_dst);

    ثابت صحیح i02_src = (صحیح)(i12_dst / sf2);
    ثابت صحیح i03_src = (صحیح)(i13_dst / sf3);

    ثابت اعشاری y_src_f = ((اعشاری)i11_dst + pixel_انحراف) / sf1 - pixel_انحراف;
    صحیح         y0_src    = (صحیح) sycl::floor((اعشاری) y_src_f);
    صحیح y1_src    = y0_src + 1;

    y0_src = sycl::max(0, sycl::min(y0_src, ne01_src - 1));
    y1_src = sycl::max(0, sycl::min(y1_src, ne01_src - 1));

    اعشاری dy = y_src_f - (اعشاری)y0_src;
    dy       = sycl::max(0.0f, sycl::min(dy, 1.0f));

    اعشاری x_src_f = ((اعشاری)i10_dst + pixel_انحراف) / sf0 - pixel_انحراف;
    صحیح   x0_src    = (صحیح) sycl::floor(x_src_f);
    صحیح x1_src    = x0_src + 1;

    x0_src = sycl::max(0, sycl::min(x0_src, ne00_src - 1));
    x1_src = sycl::max(0, sycl::min(x1_src, ne00_src - 1));

    اعشاری dx = x_src_f - (اعشاری)x0_src;
    dx       = sycl::max(0.0f, sycl::min(dx, 1.0f));

    ثابت اعشاری* p_a =
        (ثابت اعشاری*)((ثابت حرف*)x + (int64_t)x0_src * nb00 +
                       (int64_t)y0_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    ثابت اعشاری* p_b =
        (ثابت اعشاری*)((ثابت حرف*)x + (int64_t)x1_src * nb00 +
                       (int64_t)y0_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    ثابت اعشاری* p_c =
        (ثابت اعشاری*)((ثابت حرف*)x + (int64_t)x0_src * nb00 +
                       (int64_t)y1_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    ثابت اعشاری* p_d =
        (ثابت اعشاری*)((ثابت حرف*)x + (int64_t)x1_src * nb00 +
                       (int64_t)y1_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);

    ثابت اعشاری val_a = *p_a;
    ثابت اعشاری val_b = *p_b;
    ثابت اعشاری val_c = *p_c;
    ثابت اعشاری val_d = *p_d;

    اعشاری نتیجہ = val_a * (1.0f - dx) * (1.0f - dy) +
                   val_b * dx * (1.0f - dy) +
                   val_c * (1.0f - dx) * dy +
                   val_d * dx * dy;

    dst[اشاریہ] = نتیجہ;
}

// Similar to F.interpolate(..., mode="bilinear", align_corners=False, antialias=True)
// https://github.com/pytorch/pytorch/blob/8871ff29b743948d1225389d5b7068f37b22750b/aten/src/ATen/native/cpu/UpSampleKernel.cpp
ساکن خالی upscale_f32_bilinear_antialias(ثابت اعشاری * src0,
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
                                           ثابت اعشاری   pixel_انحراف) {
    خودکار شے_ct1 = sycl::ext::oneapi::this_work_شے::get_nd_شے<3>();
    ثابت int64_t اشاریہ = شے_ct1.get_local_id(2) +
        شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(2);
    ثابت int64_t dst_کل_elements = ne10_dst * ne11_dst * ne12_dst * ne13_dst;

    اگر (اشاریہ >= dst_کل_elements) {
        واپس;
    }

    ثابت صحیح i10_dst = اشاریہ % ne10_dst;
    ثابت صحیح i11_dst = (اشاریہ / ne10_dst) % ne11_dst;
    ثابت صحیح i12_dst = (اشاریہ / (ne10_dst * ne11_dst)) % ne12_dst;
    ثابت صحیح i13_dst = اشاریہ / (ne10_dst * ne11_dst * ne12_dst);

    ثابت صحیح i02_src = (صحیح)(i12_dst / sf2);
    ثابت صحیح i03_src = (صحیح)(i13_dst / sf3);

    ثابت اعشاری y = ((اعشاری)i11_dst + pixel_انحراف) / sf1;
    ثابت اعشاری x = ((اعشاری)i10_dst + pixel_انحراف) / sf0;

    // support and invscale, minimum 1 pixel for bilinear
    ثابت اعشاری support1  = sycl::max(1.0f / sf1, 1.0f);
    ثابت اعشاری invscale1 = 1.0f / support1;
    ثابت اعشاری support0  = sycl::max(1.0f / sf0, 1.0f);
    ثابت اعشاری invscale0 = 1.0f / support0;

    // the range of source pixels that contribute
    ثابت int64_t x_min = sycl::max(int64_t(0), int64_t(x - support0 + pixel_انحراف));
    ثابت int64_t x_max = sycl::min(int64_t(ne00_src), int64_t(x + support0 + pixel_انحراف));
    ثابت int64_t y_min = sycl::max(int64_t(0), int64_t(y - support1 + pixel_انحراف));
    ثابت int64_t y_max = sycl::min(int64_t(ne01_src), int64_t(y + support1 + pixel_انحراف));

    // bilinear filter with antialiasing
    اعشاری val = 0.0f;
    اعشاری کل_وزن = 0.0f;

    خودکار triangle_filter = [](اعشاری x) -> اعشاری {
        واپس sycl::max(1.0f - sycl::fabs(x), 0.0f);
    };

    برائے (int64_t sy = y_min; sy < y_max; sy++) {
        ثابت اعشاری وزن_y = triangle_filter((sy - y + pixel_انحراف) * invscale1);

        برائے (int64_t sx = x_min; sx < x_max; sx++) {
            ثابت اعشاری وزن_x = triangle_filter((sx - x + pixel_انحراف) * invscale0);
            ثابت اعشاری وزن = وزن_x * وزن_y;

            اگر (وزن <= 0.0f) {
                جاری;
            }

            ثابت اعشاری pixel =
                *(ثابت اعشاری*)((ثابت حرف*)src0 + sx * nb00 + sy * nb01 +
                                i02_src * nb02 + i03_src * nb03);
            val += pixel * وزن;
            کل_وزن += وزن;
        }
    }

    اگر (کل_وزن > 0.0f) {
        val /= کل_وزن;
    }

    dst[اشاریہ] = val;
}

نامخلا bicubic_interpolation {
ساکن اعشاری weight1(اعشاری x, ثابت اعشاری &a) { واپس ((a + 2) * x - (a + 3)) * x * x + 1; }
ساکن اعشاری weight2(اعشاری x, ثابت اعشاری &a) { واپس ((a * x - 5 * a) * x + 8 * a) * x - 4 * a; }

ساکن اعشاری bicubic(اعشاری p0, اعشاری p1, اعشاری p2, اعشاری p3, اعشاری x, اعشاری a) {
    ثابت اعشاری w0 = weight2(x + 1, a);
    ثابت اعشاری w1 = weight1(x + 0, a);
    ثابت اعشاری w2 = weight1(1 - x, a);
    ثابت اعشاری w3 = weight2(2 - x, a);
    واپس p0 * w0 + p1 * w1 + p2 * w2 + p3 * w3;
}

}

ساکن خالی upscale_f32_bicubic(ثابت اعشاری * x, اعشاری * dst,
        ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
        ثابت صحیح ne00_src, ثابت صحیح ne01_src,
        ثابت صحیح ne10_dst, ثابت صحیح ne11_dst, ثابت صحیح ne12_dst, ثابت صحیح ne13_dst,
        ثابت اعشاری sf0, ثابت اعشاری sf1, ثابت اعشاری sf2, ثابت اعشاری sf3,
        ثابت اعشاری pixel_انحراف) {
    خودکار شے_ct1 = sycl::ext::oneapi::this_work_شے::get_nd_شے<3>();
    ثابت اعشاری a = -0.75f;
    استعمال bicubic_interpolation::bicubic;

    ثابت int64_t اشاریہ = شے_ct1.get_local_id(2) +
        شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(2);
    ثابت int64_t dst_کل_elements =
        ne10_dst * ne11_dst * ne12_dst * ne13_dst;

    اگر (اشاریہ >= dst_کل_elements) {
        واپس;
    }

    ثابت صحیح i10_dst = اشاریہ % ne10_dst;
    ثابت صحیح i11_dst = (اشاریہ / ne10_dst) % ne11_dst;
    ثابت صحیح i12_dst = (اشاریہ / (ne10_dst * ne11_dst)) % ne12_dst;
    ثابت صحیح i13_dst = اشاریہ / (ne10_dst * ne11_dst * ne12_dst);

    ثابت صحیح i02_src = (صحیح)(i12_dst / sf2);
    ثابت صحیح i03_src = (صحیح)(i13_dst / sf3);

    ثابت اعشاری y_src_f = ((اعشاری)i11_dst + pixel_انحراف) / sf1 - pixel_انحراف;
    ثابت صحیح   y0_src  = (صحیح) sycl::floor((اعشاری) y_src_f);
    ثابت اعشاری dy      = y_src_f - (اعشاری)y0_src;

    ثابت اعشاری x_src_f = ((اعشاری)i10_dst + pixel_انحراف) / sf0 - pixel_انحراف;
    ثابت صحیح   x0_src  = (صحیح) sycl::floor((اعشاری) x_src_f);
    ثابت اعشاری dx      = x_src_f - (اعشاری)x0_src;

    ثابت حرف * x_بنیاد = (ثابت حرف *)x + (int64_t)i02_src * nb02 + (int64_t)i03_src * nb03;

    خودکار load = [=](صحیح x_off, صحیح y_off) -> اعشاری {
        صحیح i00_src = sycl::max(0, sycl::min(x0_src + x_off, ne00_src - 1));
        صحیح i01_src = sycl::max(0, sycl::min(y0_src + y_off, ne01_src - 1));
        واپس *(ثابت اعشاری *)(x_بنیاد + (int64_t)i00_src * nb00 + (int64_t)i01_src * nb01);
    };

    ثابت اعشاری نتیجہ = bicubic(
        bicubic(load(-1, -1), load(0, -1), load(1, -1), load(2, -1), dx, a),
        bicubic(load(-1, 0), load(0, 0), load(1, 0), load(2, 0), dx, a),
        bicubic(load(-1, 1), load(0, 1), load(1, 1), load(2, 1), dx, a),
        bicubic(load(-1, 2), load(0, 2), load(1, 2), load(2, 2), dx, a),
        dy,
        a);

    dst[اشاریہ] = نتیجہ;
}

ساکن خالی upscale_f32_sycl(ثابت اعشاری *   x,
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
                             dpct::قطار_ptr stream) {
    ثابت int64_t dst_سائز   = ne10 * ne11 * ne12 * ne13;
    ثابت int64_t عدد_بلاکیں = (dst_سائز + SYCL_UPSCALE_بلاک_سائز - 1) / SYCL_UPSCALE_بلاک_سائز;

    stream->parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_UPSCALE_بلاک_سائز),
             sycl::range<3>(1, 1, SYCL_UPSCALE_بلاک_سائز)),
        [=](sycl::nd_شے<3> /*item_ct1*/) {
            upscale_f32(x, dst, nb00, nb01, nb02, nb03, ne10, ne11, ne12, ne13, sf0, sf1, sf2, sf3);
        });
}

ساکن خالی upscale_f32_bilinear_sycl(ثابت اعشاری *   x,
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
                                      ثابت اعشاری     pixel_انحراف,
                                      منطقی            antialias,
                                      dpct::قطار_ptr stream) {
    ثابت int64_t dst_سائز   = ne10_dst * ne11_dst * ne12_dst * ne13_dst;
    ثابت int64_t عدد_بلاکیں = (dst_سائز + SYCL_UPSCALE_بلاک_سائز - 1) / SYCL_UPSCALE_بلاک_سائز;

    اگر (antialias) {
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_UPSCALE_بلاک_سائز),
                sycl::range<3>(1, 1, SYCL_UPSCALE_بلاک_سائز)),
            [=](sycl::nd_شے<3> /*item_ct1*/) {
                upscale_f32_bilinear_antialias(
                    x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst,
                    ne12_dst, ne13_dst, sf0, sf1, sf2, sf3, pixel_انحراف);
            });
    } ورنہ {
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_UPSCALE_بلاک_سائز),
                sycl::range<3>(1, 1, SYCL_UPSCALE_بلاک_سائز)),
            [=](sycl::nd_شے<3> /*item_ct1*/) {
                upscale_f32_bilinear(
                    x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst, ne12_dst,
                    ne13_dst, sf0, sf1, sf2, sf3, pixel_انحراف);
            });
    }
}

ساکن خالی upscale_f32_bicubic_sycl(ثابت اعشاری *   x,
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
                                     ثابت اعشاری     pixel_انحراف,
                                     dpct::قطار_ptr stream) {
    ثابت int64_t dst_سائز   = ne10_dst * ne11_dst * ne12_dst * ne13_dst;
    ثابت int64_t عدد_بلاکیں = (dst_سائز + SYCL_UPSCALE_بلاک_سائز - 1) / SYCL_UPSCALE_بلاک_سائز;

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_UPSCALE_بلاک_سائز),
                    sycl::range<3>(1, 1, SYCL_UPSCALE_بلاک_سائز)),
                [=](sycl::nd_شے<3> /*item_ct1*/) {
                    upscale_f32_bicubic(
                        x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst,
                        ne12_dst, ne13_dst, sf0, sf1, sf2, sf3, pixel_انحراف);
                });
        });
    }
}

ساکن خالی ggml_sycl_op_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت اعشاری * src0_d = (ثابت اعشاری *)src0->ڈیٹا;
    اعشاری * dst_d = (اعشاری *)dst->ڈیٹا;
    dpct::قطار_ptr     stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    ثابت صحیح منوال_flags = dst->op_params[0];
    ثابت ggml_scale_منوال mode = (ggml_scale_منوال)(منوال_flags & 0xFF);

    اعشاری sf0 = (اعشاری)dst->ne[0]/src0->ne[0];
    اعشاری sf1 = (اعشاری)dst->ne[1]/src0->ne[1];
    اعشاری sf2 = (اعشاری)dst->ne[2]/src0->ne[2];
    ثابت اعشاری sf3 = (اعشاری)dst->ne[3]/src0->ne[3];

    اعشاری pixel_انحراف = 0.5f;
    اگر (منوال_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        sf0 = dst->ne[0] > 1 && src0->ne[0] > 1
            ? (اعشاری)(dst->ne[0] - 1) / (src0->ne[0] - 1)
            : sf0;
        sf1 = dst->ne[1] > 1 && src0->ne[1] > 1
            ? (اعشاری)(dst->ne[1] - 1) / (src0->ne[1] - 1)
            : sf1;
        pixel_انحراف = 0.0f;
    }

    اگر (mode == GGML_SCALE_منوال_قریب_ترین) {
        upscale_f32_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], sf0, sf1, sf2, sf3, stream);
    } ورنہ اگر (mode == GGML_SCALE_منوال_BILINEAR) {
        ثابت منطقی antialias = (منوال_flags & GGML_SCALE_FLAG_ANTIALIAS);
        upscale_f32_bilinear_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src0->ne[0], src0->ne[1], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            sf0, sf1, sf2, sf3, pixel_انحراف, antialias, stream);
    } ورنہ اگر (mode == GGML_SCALE_منوال_BICUBIC) {
        upscale_f32_bicubic_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src0->ne[0], src0->ne[1], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            sf0, sf1, sf2, sf3, pixel_انحراف, stream);
    }
}

خالی ggml_sycl_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_upscale(ctx, dst);
}
