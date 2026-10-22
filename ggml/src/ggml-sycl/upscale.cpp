#అన్తర్భావ "upscale.hpp"

స్థిర శూన్య upscale_f32(స్థిరరాశి దశాంశం * x, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం nb00, స్థిరరాశి పూర్ణాంకం nb01, స్థిరరాశి పూర్ణాంకం nb02, స్థిరరాశి పూర్ణాంకం nb03,
        స్థిరరాశి పూర్ణాంకం ne10, స్థిరరాశి పూర్ణాంకం ne11, స్థిరరాశి పూర్ణాంకం ne12, స్థిరరాశి పూర్ణాంకం ne13,
        స్థిరరాశి దశాంశం sf0, స్థిరరాశి దశాంశం sf1, స్థిరరాశి దశాంశం sf2, స్థిరరాశి దశాంశం sf3) {
    స్వయంచాలక అంశం_ct1 = sycl::ext::oneapi::this_work_అంశం::get_nd_అంశం<3>();
    పూర్ణాంకం  సూచీ    = అంశం_ct1.get_local_id(2) + అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(2);
    ఒకవేళ (సూచీ >= ne10 * ne11 * ne12 * ne13) {
        తిరిగి;
    }

    పూర్ణాంకం i10 = సూచీ % ne10;
    పూర్ణాంకం i11 = (సూచీ / ne10) % ne11;
    పూర్ణాంకం i12 = (సూచీ / (ne10 * ne11)) % ne12;
    పూర్ణాంకం i13 = (సూచీ / (ne10 * ne11 * ne12)) % ne13;

    పూర్ణాంకం i00 = i10 / sf0;
    పూర్ణాంకం i01 = i11 / sf1;
    పూర్ణాంకం i02 = i12 / sf2;
    పూర్ణాంకం i03 = i13 / sf3;

    dst[సూచీ] = *((స్థిరరాశి దశాంశం*)((స్థిరరాశి అక్షరం*)x + i03 * nb03 + i02 * nb02 +
                                  i01 * nb01 + i00 * nb00));
}

స్థిర శూన్య upscale_f32_bilinear(స్థిరరాశి దశాంశం * x, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం nb00, స్థిరరాశి పూర్ణాంకం nb01, స్థిరరాశి పూర్ణాంకం nb02, స్థిరరాశి పూర్ణాంకం nb03,
        స్థిరరాశి పూర్ణాంకం ne00_src, స్థిరరాశి పూర్ణాంకం ne01_src,
        స్థిరరాశి పూర్ణాంకం ne10_dst, స్థిరరాశి పూర్ణాంకం ne11_dst, స్థిరరాశి పూర్ణాంకం ne12_dst, స్థిరరాశి పూర్ణాంకం ne13_dst,
        స్థిరరాశి దశాంశం sf0, స్థిరరాశి దశాంశం sf1, స్థిరరాశి దశాంశం sf2, స్థిరరాశి దశాంశం sf3,
        స్థిరరాశి దశాంశం pixel_స్థానభ్రంశం) {
    స్వయంచాలక అంశం_ct1 = sycl::ext::oneapi::this_work_అంశం::get_nd_అంశం<3>();
    స్థిరరాశి int64_t సూచీ = అంశం_ct1.get_local_id(2) +
        అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(2);
    స్థిరరాశి int64_t dst_మొత్తం_elements = ne10_dst * ne11_dst * ne12_dst * ne13_dst;

  ఒకవేళ (సూచీ >= dst_మొత్తం_elements) {
    తిరిగి;
  }

    స్థిరరాశి పూర్ణాంకం i10_dst = సూచీ % ne10_dst;
    స్థిరరాశి పూర్ణాంకం i11_dst = (సూచీ / ne10_dst) % ne11_dst;
    స్థిరరాశి పూర్ణాంకం i12_dst = (సూచీ / (ne10_dst * ne11_dst)) % ne12_dst;
    స్థిరరాశి పూర్ణాంకం i13_dst = సూచీ / (ne10_dst * ne11_dst * ne12_dst);

    స్థిరరాశి పూర్ణాంకం i02_src = (పూర్ణాంకం)(i12_dst / sf2);
    స్థిరరాశి పూర్ణాంకం i03_src = (పూర్ణాంకం)(i13_dst / sf3);

    స్థిరరాశి దశాంశం y_src_f = ((దశాంశం)i11_dst + pixel_స్థానభ్రంశం) / sf1 - pixel_స్థానభ్రంశం;
    పూర్ణాంకం         y0_src    = (పూర్ణాంకం) sycl::floor((దశాంశం) y_src_f);
    పూర్ణాంకం y1_src    = y0_src + 1;

    y0_src = sycl::max(0, sycl::min(y0_src, ne01_src - 1));
    y1_src = sycl::max(0, sycl::min(y1_src, ne01_src - 1));

    దశాంశం dy = y_src_f - (దశాంశం)y0_src;
    dy       = sycl::max(0.0f, sycl::min(dy, 1.0f));

    దశాంశం x_src_f = ((దశాంశం)i10_dst + pixel_స్థానభ్రంశం) / sf0 - pixel_స్థానభ్రంశం;
    పూర్ణాంకం   x0_src    = (పూర్ణాంకం) sycl::floor(x_src_f);
    పూర్ణాంకం x1_src    = x0_src + 1;

    x0_src = sycl::max(0, sycl::min(x0_src, ne00_src - 1));
    x1_src = sycl::max(0, sycl::min(x1_src, ne00_src - 1));

    దశాంశం dx = x_src_f - (దశాంశం)x0_src;
    dx       = sycl::max(0.0f, sycl::min(dx, 1.0f));

    స్థిరరాశి దశాంశం* p_a =
        (స్థిరరాశి దశాంశం*)((స్థిరరాశి అక్షరం*)x + (int64_t)x0_src * nb00 +
                       (int64_t)y0_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    స్థిరరాశి దశాంశం* p_b =
        (స్థిరరాశి దశాంశం*)((స్థిరరాశి అక్షరం*)x + (int64_t)x1_src * nb00 +
                       (int64_t)y0_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    స్థిరరాశి దశాంశం* p_c =
        (స్థిరరాశి దశాంశం*)((స్థిరరాశి అక్షరం*)x + (int64_t)x0_src * nb00 +
                       (int64_t)y1_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    స్థిరరాశి దశాంశం* p_d =
        (స్థిరరాశి దశాంశం*)((స్థిరరాశి అక్షరం*)x + (int64_t)x1_src * nb00 +
                       (int64_t)y1_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);

    స్థిరరాశి దశాంశం val_a = *p_a;
    స్థిరరాశి దశాంశం val_b = *p_b;
    స్థిరరాశి దశాంశం val_c = *p_c;
    స్థిరరాశి దశాంశం val_d = *p_d;

    దశాంశం ఫలితం = val_a * (1.0f - dx) * (1.0f - dy) +
                   val_b * dx * (1.0f - dy) +
                   val_c * (1.0f - dx) * dy +
                   val_d * dx * dy;

    dst[సూచీ] = ఫలితం;
}

// Similar to F.interpolate(..., mode="bilinear", align_corners=False, antialias=True)
// https://github.com/pytorch/pytorch/blob/8871ff29b743948d1225389d5b7068f37b22750b/aten/src/ATen/native/cpu/UpSampleKernel.cpp
స్థిర శూన్య upscale_f32_bilinear_antialias(స్థిరరాశి దశాంశం * src0,
                                           దశాంశం *       dst,
                                           స్థిరరాశి పూర్ణాంకం     nb00,
                                           స్థిరరాశి పూర్ణాంకం     nb01,
                                           స్థిరరాశి పూర్ణాంకం     nb02,
                                           స్థిరరాశి పూర్ణాంకం     nb03,
                                           స్థిరరాశి పూర్ణాంకం     ne00_src,
                                           స్థిరరాశి పూర్ణాంకం     ne01_src,
                                           స్థిరరాశి పూర్ణాంకం     ne10_dst,
                                           స్థిరరాశి పూర్ణాంకం     ne11_dst,
                                           స్థిరరాశి పూర్ణాంకం     ne12_dst,
                                           స్థిరరాశి పూర్ణాంకం     ne13_dst,
                                           స్థిరరాశి దశాంశం   sf0,
                                           స్థిరరాశి దశాంశం   sf1,
                                           స్థిరరాశి దశాంశం   sf2,
                                           స్థిరరాశి దశాంశం   sf3,
                                           స్థిరరాశి దశాంశం   pixel_స్థానభ్రంశం) {
    స్వయంచాలక అంశం_ct1 = sycl::ext::oneapi::this_work_అంశం::get_nd_అంశం<3>();
    స్థిరరాశి int64_t సూచీ = అంశం_ct1.get_local_id(2) +
        అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(2);
    స్థిరరాశి int64_t dst_మొత్తం_elements = ne10_dst * ne11_dst * ne12_dst * ne13_dst;

    ఒకవేళ (సూచీ >= dst_మొత్తం_elements) {
        తిరిగి;
    }

    స్థిరరాశి పూర్ణాంకం i10_dst = సూచీ % ne10_dst;
    స్థిరరాశి పూర్ణాంకం i11_dst = (సూచీ / ne10_dst) % ne11_dst;
    స్థిరరాశి పూర్ణాంకం i12_dst = (సూచీ / (ne10_dst * ne11_dst)) % ne12_dst;
    స్థిరరాశి పూర్ణాంకం i13_dst = సూచీ / (ne10_dst * ne11_dst * ne12_dst);

    స్థిరరాశి పూర్ణాంకం i02_src = (పూర్ణాంకం)(i12_dst / sf2);
    స్థిరరాశి పూర్ణాంకం i03_src = (పూర్ణాంకం)(i13_dst / sf3);

    స్థిరరాశి దశాంశం y = ((దశాంశం)i11_dst + pixel_స్థానభ్రంశం) / sf1;
    స్థిరరాశి దశాంశం x = ((దశాంశం)i10_dst + pixel_స్థానభ్రంశం) / sf0;

    // support and invscale, minimum 1 pixel for bilinear
    స్థిరరాశి దశాంశం support1  = sycl::max(1.0f / sf1, 1.0f);
    స్థిరరాశి దశాంశం invscale1 = 1.0f / support1;
    స్థిరరాశి దశాంశం support0  = sycl::max(1.0f / sf0, 1.0f);
    స్థిరరాశి దశాంశం invscale0 = 1.0f / support0;

    // the range of source pixels that contribute
    స్థిరరాశి int64_t x_min = sycl::max(int64_t(0), int64_t(x - support0 + pixel_స్థానభ్రంశం));
    స్థిరరాశి int64_t x_max = sycl::min(int64_t(ne00_src), int64_t(x + support0 + pixel_స్థానభ్రంశం));
    స్థిరరాశి int64_t y_min = sycl::max(int64_t(0), int64_t(y - support1 + pixel_స్థానభ్రంశం));
    స్థిరరాశి int64_t y_max = sycl::min(int64_t(ne01_src), int64_t(y + support1 + pixel_స్థానభ్రంశం));

    // bilinear filter with antialiasing
    దశాంశం val = 0.0f;
    దశాంశం మొత్తం_బరువు = 0.0f;

    స్వయంచాలక triangle_filter = [](దశాంశం x) -> దశాంశం {
        తిరిగి sycl::max(1.0f - sycl::fabs(x), 0.0f);
    };

    కొరకు (int64_t sy = y_min; sy < y_max; sy++) {
        స్థిరరాశి దశాంశం బరువు_y = triangle_filter((sy - y + pixel_స్థానభ్రంశం) * invscale1);

        కొరకు (int64_t sx = x_min; sx < x_max; sx++) {
            స్థిరరాశి దశాంశం బరువు_x = triangle_filter((sx - x + pixel_స్థానభ్రంశం) * invscale0);
            స్థిరరాశి దశాంశం బరువు = బరువు_x * బరువు_y;

            ఒకవేళ (బరువు <= 0.0f) {
                కొనసాగు;
            }

            స్థిరరాశి దశాంశం pixel =
                *(స్థిరరాశి దశాంశం*)((స్థిరరాశి అక్షరం*)src0 + sx * nb00 + sy * nb01 +
                                i02_src * nb02 + i03_src * nb03);
            val += pixel * బరువు;
            మొత్తం_బరువు += బరువు;
        }
    }

    ఒకవేళ (మొత్తం_బరువు > 0.0f) {
        val /= మొత్తం_బరువు;
    }

    dst[సూచీ] = val;
}

నామాకాశ bicubic_interpolation {
స్థిర దశాంశం weight1(దశాంశం x, స్థిరరాశి దశాంశం &a) { తిరిగి ((a + 2) * x - (a + 3)) * x * x + 1; }
స్థిర దశాంశం weight2(దశాంశం x, స్థిరరాశి దశాంశం &a) { తిరిగి ((a * x - 5 * a) * x + 8 * a) * x - 4 * a; }

స్థిర దశాంశం bicubic(దశాంశం p0, దశాంశం p1, దశాంశం p2, దశాంశం p3, దశాంశం x, దశాంశం a) {
    స్థిరరాశి దశాంశం w0 = weight2(x + 1, a);
    స్థిరరాశి దశాంశం w1 = weight1(x + 0, a);
    స్థిరరాశి దశాంశం w2 = weight1(1 - x, a);
    స్థిరరాశి దశాంశం w3 = weight2(2 - x, a);
    తిరిగి p0 * w0 + p1 * w1 + p2 * w2 + p3 * w3;
}

}

స్థిర శూన్య upscale_f32_bicubic(స్థిరరాశి దశాంశం * x, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం nb00, స్థిరరాశి పూర్ణాంకం nb01, స్థిరరాశి పూర్ణాంకం nb02, స్థిరరాశి పూర్ణాంకం nb03,
        స్థిరరాశి పూర్ణాంకం ne00_src, స్థిరరాశి పూర్ణాంకం ne01_src,
        స్థిరరాశి పూర్ణాంకం ne10_dst, స్థిరరాశి పూర్ణాంకం ne11_dst, స్థిరరాశి పూర్ణాంకం ne12_dst, స్థిరరాశి పూర్ణాంకం ne13_dst,
        స్థిరరాశి దశాంశం sf0, స్థిరరాశి దశాంశం sf1, స్థిరరాశి దశాంశం sf2, స్థిరరాశి దశాంశం sf3,
        స్థిరరాశి దశాంశం pixel_స్థానభ్రంశం) {
    స్వయంచాలక అంశం_ct1 = sycl::ext::oneapi::this_work_అంశం::get_nd_అంశం<3>();
    స్థిరరాశి దశాంశం a = -0.75f;
    ఉపయోగించి bicubic_interpolation::bicubic;

    స్థిరరాశి int64_t సూచీ = అంశం_ct1.get_local_id(2) +
        అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(2);
    స్థిరరాశి int64_t dst_మొత్తం_elements =
        ne10_dst * ne11_dst * ne12_dst * ne13_dst;

    ఒకవేళ (సూచీ >= dst_మొత్తం_elements) {
        తిరిగి;
    }

    స్థిరరాశి పూర్ణాంకం i10_dst = సూచీ % ne10_dst;
    స్థిరరాశి పూర్ణాంకం i11_dst = (సూచీ / ne10_dst) % ne11_dst;
    స్థిరరాశి పూర్ణాంకం i12_dst = (సూచీ / (ne10_dst * ne11_dst)) % ne12_dst;
    స్థిరరాశి పూర్ణాంకం i13_dst = సూచీ / (ne10_dst * ne11_dst * ne12_dst);

    స్థిరరాశి పూర్ణాంకం i02_src = (పూర్ణాంకం)(i12_dst / sf2);
    స్థిరరాశి పూర్ణాంకం i03_src = (పూర్ణాంకం)(i13_dst / sf3);

    స్థిరరాశి దశాంశం y_src_f = ((దశాంశం)i11_dst + pixel_స్థానభ్రంశం) / sf1 - pixel_స్థానభ్రంశం;
    స్థిరరాశి పూర్ణాంకం   y0_src  = (పూర్ణాంకం) sycl::floor((దశాంశం) y_src_f);
    స్థిరరాశి దశాంశం dy      = y_src_f - (దశాంశం)y0_src;

    స్థిరరాశి దశాంశం x_src_f = ((దశాంశం)i10_dst + pixel_స్థానభ్రంశం) / sf0 - pixel_స్థానభ్రంశం;
    స్థిరరాశి పూర్ణాంకం   x0_src  = (పూర్ణాంకం) sycl::floor((దశాంశం) x_src_f);
    స్థిరరాశి దశాంశం dx      = x_src_f - (దశాంశం)x0_src;

    స్థిరరాశి అక్షరం * x_పునాది = (స్థిరరాశి అక్షరం *)x + (int64_t)i02_src * nb02 + (int64_t)i03_src * nb03;

    స్వయంచాలక load = [=](పూర్ణాంకం x_off, పూర్ణాంకం y_off) -> దశాంశం {
        పూర్ణాంకం i00_src = sycl::max(0, sycl::min(x0_src + x_off, ne00_src - 1));
        పూర్ణాంకం i01_src = sycl::max(0, sycl::min(y0_src + y_off, ne01_src - 1));
        తిరిగి *(స్థిరరాశి దశాంశం *)(x_పునాది + (int64_t)i00_src * nb00 + (int64_t)i01_src * nb01);
    };

    స్థిరరాశి దశాంశం ఫలితం = bicubic(
        bicubic(load(-1, -1), load(0, -1), load(1, -1), load(2, -1), dx, a),
        bicubic(load(-1, 0), load(0, 0), load(1, 0), load(2, 0), dx, a),
        bicubic(load(-1, 1), load(0, 1), load(1, 1), load(2, 1), dx, a),
        bicubic(load(-1, 2), load(0, 2), load(1, 2), load(2, 2), dx, a),
        dy,
        a);

    dst[సూచీ] = ఫలితం;
}

స్థిర శూన్య upscale_f32_sycl(స్థిరరాశి దశాంశం *   x,
                             దశాంశం *         dst,
                             స్థిరరాశి పూర్ణాంకం       nb00,
                             స్థిరరాశి పూర్ణాంకం       nb01,
                             స్థిరరాశి పూర్ణాంకం       nb02,
                             స్థిరరాశి పూర్ణాంకం       nb03,
                             స్థిరరాశి పూర్ణాంకం       ne10,
                             స్థిరరాశి పూర్ణాంకం       ne11,
                             స్థిరరాశి పూర్ణాంకం       ne12,
                             స్థిరరాశి పూర్ణాంకం       ne13,
                             స్థిరరాశి దశాంశం     sf0,
                             స్థిరరాశి దశాంశం     sf1,
                             స్థిరరాశి దశాంశం     sf2,
                             స్థిరరాశి దశాంశం     sf3,
                             dpct::వరుస_ptr stream) {
    స్థిరరాశి int64_t dst_కొలత   = ne10 * ne11 * ne12 * ne13;
    స్థిరరాశి int64_t అంకె_ఖండాలు = (dst_కొలత + SYCL_UPSCALE_ఖండం_కొలత - 1) / SYCL_UPSCALE_ఖండం_కొలత;

    stream->parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, అంకె_ఖండాలు) * sycl::range<3>(1, 1, SYCL_UPSCALE_ఖండం_కొలత),
             sycl::range<3>(1, 1, SYCL_UPSCALE_ఖండం_కొలత)),
        [=](sycl::nd_అంశం<3> /*item_ct1*/) {
            upscale_f32(x, dst, nb00, nb01, nb02, nb03, ne10, ne11, ne12, ne13, sf0, sf1, sf2, sf3);
        });
}

స్థిర శూన్య upscale_f32_bilinear_sycl(స్థిరరాశి దశాంశం *   x,
                                      దశాంశం *         dst,
                                      స్థిరరాశి పూర్ణాంకం       nb00,
                                      స్థిరరాశి పూర్ణాంకం       nb01,
                                      స్థిరరాశి పూర్ణాంకం       nb02,
                                      స్థిరరాశి పూర్ణాంకం       nb03,
                                      స్థిరరాశి పూర్ణాంకం       ne00_src,
                                      స్థిరరాశి పూర్ణాంకం       ne01_src,
                                      స్థిరరాశి పూర్ణాంకం       ne10_dst,
                                      స్థిరరాశి పూర్ణాంకం       ne11_dst,
                                      స్థిరరాశి పూర్ణాంకం       ne12_dst,
                                      స్థిరరాశి పూర్ణాంకం       ne13_dst,
                                      స్థిరరాశి దశాంశం     sf0,
                                      స్థిరరాశి దశాంశం     sf1,
                                      స్థిరరాశి దశాంశం     sf2,
                                      స్థిరరాశి దశాంశం     sf3,
                                      స్థిరరాశి దశాంశం     pixel_స్థానభ్రంశం,
                                      తార్కిక            antialias,
                                      dpct::వరుస_ptr stream) {
    స్థిరరాశి int64_t dst_కొలత   = ne10_dst * ne11_dst * ne12_dst * ne13_dst;
    స్థిరరాశి int64_t అంకె_ఖండాలు = (dst_కొలత + SYCL_UPSCALE_ఖండం_కొలత - 1) / SYCL_UPSCALE_ఖండం_కొలత;

    ఒకవేళ (antialias) {
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, అంకె_ఖండాలు) * sycl::range<3>(1, 1, SYCL_UPSCALE_ఖండం_కొలత),
                sycl::range<3>(1, 1, SYCL_UPSCALE_ఖండం_కొలత)),
            [=](sycl::nd_అంశం<3> /*item_ct1*/) {
                upscale_f32_bilinear_antialias(
                    x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst,
                    ne12_dst, ne13_dst, sf0, sf1, sf2, sf3, pixel_స్థానభ్రంశం);
            });
    } లేకపోతే {
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, అంకె_ఖండాలు) * sycl::range<3>(1, 1, SYCL_UPSCALE_ఖండం_కొలత),
                sycl::range<3>(1, 1, SYCL_UPSCALE_ఖండం_కొలత)),
            [=](sycl::nd_అంశం<3> /*item_ct1*/) {
                upscale_f32_bilinear(
                    x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst, ne12_dst,
                    ne13_dst, sf0, sf1, sf2, sf3, pixel_స్థానభ్రంశం);
            });
    }
}

స్థిర శూన్య upscale_f32_bicubic_sycl(స్థిరరాశి దశాంశం *   x,
                                     దశాంశం *         dst,
                                     స్థిరరాశి పూర్ణాంకం       nb00,
                                     స్థిరరాశి పూర్ణాంకం       nb01,
                                     స్థిరరాశి పూర్ణాంకం       nb02,
                                     స్థిరరాశి పూర్ణాంకం       nb03,
                                     స్థిరరాశి పూర్ణాంకం       ne00_src,
                                     స్థిరరాశి పూర్ణాంకం       ne01_src,
                                     స్థిరరాశి పూర్ణాంకం       ne10_dst,
                                     స్థిరరాశి పూర్ణాంకం       ne11_dst,
                                     స్థిరరాశి పూర్ణాంకం       ne12_dst,
                                     స్థిరరాశి పూర్ణాంకం       ne13_dst,
                                     స్థిరరాశి దశాంశం     sf0,
                                     స్థిరరాశి దశాంశం     sf1,
                                     స్థిరరాశి దశాంశం     sf2,
                                     స్థిరరాశి దశాంశం     sf3,
                                     స్థిరరాశి దశాంశం     pixel_స్థానభ్రంశం,
                                     dpct::వరుస_ptr stream) {
    స్థిరరాశి int64_t dst_కొలత   = ne10_dst * ne11_dst * ne12_dst * ne13_dst;
    స్థిరరాశి int64_t అంకె_ఖండాలు = (dst_కొలత + SYCL_UPSCALE_ఖండం_కొలత - 1) / SYCL_UPSCALE_ఖండం_కొలత;

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, 1, అంకె_ఖండాలు) * sycl::range<3>(1, 1, SYCL_UPSCALE_ఖండం_కొలత),
                    sycl::range<3>(1, 1, SYCL_UPSCALE_ఖండం_కొలత)),
                [=](sycl::nd_అంశం<3> /*item_ct1*/) {
                    upscale_f32_bicubic(
                        x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst,
                        ne12_dst, ne13_dst, sf0, sf1, sf2, sf3, pixel_స్థానభ్రంశం);
                });
        });
    }
}

స్థిర శూన్య ggml_sycl_op_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి దశాంశం * src0_d = (స్థిరరాశి దశాంశం *)src0->దత్తాంశం;
    దశాంశం * dst_d = (దశాంశం *)dst->దత్తాంశం;
    dpct::వరుస_ptr     stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    స్థిరరాశి పూర్ణాంకం బహుళకం_flags = dst->op_params[0];
    స్థిరరాశి ggml_scale_బహుళకం mode = (ggml_scale_బహుళకం)(బహుళకం_flags & 0xFF);

    దశాంశం sf0 = (దశాంశం)dst->ne[0]/src0->ne[0];
    దశాంశం sf1 = (దశాంశం)dst->ne[1]/src0->ne[1];
    దశాంశం sf2 = (దశాంశం)dst->ne[2]/src0->ne[2];
    స్థిరరాశి దశాంశం sf3 = (దశాంశం)dst->ne[3]/src0->ne[3];

    దశాంశం pixel_స్థానభ్రంశం = 0.5f;
    ఒకవేళ (బహుళకం_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        sf0 = dst->ne[0] > 1 && src0->ne[0] > 1
            ? (దశాంశం)(dst->ne[0] - 1) / (src0->ne[0] - 1)
            : sf0;
        sf1 = dst->ne[1] > 1 && src0->ne[1] > 1
            ? (దశాంశం)(dst->ne[1] - 1) / (src0->ne[1] - 1)
            : sf1;
        pixel_స్థానభ్రంశం = 0.0f;
    }

    ఒకవేళ (mode == GGML_SCALE_బహుళకం_సమీప) {
        upscale_f32_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], sf0, sf1, sf2, sf3, stream);
    } లేకపోతే ఒకవేళ (mode == GGML_SCALE_బహుళకం_BILINEAR) {
        స్థిరరాశి తార్కిక antialias = (బహుళకం_flags & GGML_SCALE_FLAG_ANTIALIAS);
        upscale_f32_bilinear_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src0->ne[0], src0->ne[1], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            sf0, sf1, sf2, sf3, pixel_స్థానభ్రంశం, antialias, stream);
    } లేకపోతే ఒకవేళ (mode == GGML_SCALE_బహుళకం_BICUBIC) {
        upscale_f32_bicubic_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src0->ne[0], src0->ne[1], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            sf0, sf1, sf2, sf3, pixel_స్థానభ్రంశం, stream);
    }
}

శూన్య ggml_sycl_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_upscale(ctx, dst);
}
