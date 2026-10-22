#include "upscale.hpp"

static void upscale_f32(const float * x, float * dst,
        const цео nb00, const цео nb01, const цео nb02, const цео nb03,
        const цео ne10, const цео ne11, const цео ne12, const цео ne13,
        const float sf0, const float sf1, const float sf2, const float sf3) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    цео  index    = item_ct1.get_local_id(2) + item_ct1.get_group(2) * item_ct1.get_local_range(2);
    ако (index >= ne10 * ne11 * ne12 * ne13) {
        врати;
    }

    цео i10 = index % ne10;
    цео i11 = (index / ne10) % ne11;
    цео i12 = (index / (ne10 * ne11)) % ne12;
    цео i13 = (index / (ne10 * ne11 * ne12)) % ne13;

    цео i00 = i10 / sf0;
    цео i01 = i11 / sf1;
    цео i02 = i12 / sf2;
    цео i03 = i13 / sf3;

    dst[index] = *((const float*)((const char*)x + i03 * nb03 + i02 * nb02 +
                                  i01 * nb01 + i00 * nb00));
}

static void upscale_f32_bilinear(const float * x, float * dst,
        const цео nb00, const цео nb01, const цео nb02, const цео nb03,
        const цео ne00_src, const цео ne01_src,
        const цео ne10_dst, const цео ne11_dst, const цео ne12_dst, const цео ne13_dst,
        const float sf0, const float sf1, const float sf2, const float sf3,
        const float pixel_offset) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t index = item_ct1.get_local_id(2) +
        item_ct1.get_group(2) * item_ct1.get_local_range(2);
    const int64_t dst_total_elements = ne10_dst * ne11_dst * ne12_dst * ne13_dst;

  ако (index >= dst_total_elements) {
    врати;
  }

    const цео i10_dst = index % ne10_dst;
    const цео i11_dst = (index / ne10_dst) % ne11_dst;
    const цео i12_dst = (index / (ne10_dst * ne11_dst)) % ne12_dst;
    const цео i13_dst = index / (ne10_dst * ne11_dst * ne12_dst);

    const цео i02_src = (цео)(i12_dst / sf2);
    const цео i03_src = (цео)(i13_dst / sf3);

    const float y_src_f = ((float)i11_dst + pixel_offset) / sf1 - pixel_offset;
    цео         y0_src    = (цео) sycl::floor((float) y_src_f);
    цео y1_src    = y0_src + 1;

    y0_src = sycl::max(0, sycl::min(y0_src, ne01_src - 1));
    y1_src = sycl::max(0, sycl::min(y1_src, ne01_src - 1));

    float dy = y_src_f - (float)y0_src;
    dy       = sycl::max(0.0f, sycl::min(dy, 1.0f));

    float x_src_f = ((float)i10_dst + pixel_offset) / sf0 - pixel_offset;
    цео   x0_src    = (цео) sycl::floor(x_src_f);
    цео x1_src    = x0_src + 1;

    x0_src = sycl::max(0, sycl::min(x0_src, ne00_src - 1));
    x1_src = sycl::max(0, sycl::min(x1_src, ne00_src - 1));

    float dx = x_src_f - (float)x0_src;
    dx       = sycl::max(0.0f, sycl::min(dx, 1.0f));

    const float* p_a =
        (const float*)((const char*)x + (int64_t)x0_src * nb00 +
                       (int64_t)y0_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    const float* p_b =
        (const float*)((const char*)x + (int64_t)x1_src * nb00 +
                       (int64_t)y0_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    const float* p_c =
        (const float*)((const char*)x + (int64_t)x0_src * nb00 +
                       (int64_t)y1_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);
    const float* p_d =
        (const float*)((const char*)x + (int64_t)x1_src * nb00 +
                       (int64_t)y1_src * nb01 + (int64_t)i02_src * nb02 +
                       (int64_t)i03_src * nb03);

    const float val_a = *p_a;
    const float val_b = *p_b;
    const float val_c = *p_c;
    const float val_d = *p_d;

    float result = val_a * (1.0f - dx) * (1.0f - dy) +
                   val_b * dx * (1.0f - dy) +
                   val_c * (1.0f - dx) * dy +
                   val_d * dx * dy;

    dst[index] = result;
}

// Similar to F.interpolate(..., mode="bilinear", align_corners=False, antialias=True)
// https://github.com/pytorch/pytorch/blob/8871ff29b743948d1225389d5b7068f37b22750b/aten/src/ATen/native/cpu/UpSampleKernel.cpp
static void upscale_f32_bilinear_antialias(const float * src0,
                                           float *       dst,
                                           const цео     nb00,
                                           const цео     nb01,
                                           const цео     nb02,
                                           const цео     nb03,
                                           const цео     ne00_src,
                                           const цео     ne01_src,
                                           const цео     ne10_dst,
                                           const цео     ne11_dst,
                                           const цео     ne12_dst,
                                           const цео     ne13_dst,
                                           const float   sf0,
                                           const float   sf1,
                                           const float   sf2,
                                           const float   sf3,
                                           const float   pixel_offset) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t index = item_ct1.get_local_id(2) +
        item_ct1.get_group(2) * item_ct1.get_local_range(2);
    const int64_t dst_total_elements = ne10_dst * ne11_dst * ne12_dst * ne13_dst;

    ако (index >= dst_total_elements) {
        врати;
    }

    const цео i10_dst = index % ne10_dst;
    const цео i11_dst = (index / ne10_dst) % ne11_dst;
    const цео i12_dst = (index / (ne10_dst * ne11_dst)) % ne12_dst;
    const цео i13_dst = index / (ne10_dst * ne11_dst * ne12_dst);

    const цео i02_src = (цео)(i12_dst / sf2);
    const цео i03_src = (цео)(i13_dst / sf3);

    const float y = ((float)i11_dst + pixel_offset) / sf1;
    const float x = ((float)i10_dst + pixel_offset) / sf0;

    // support and invscale, minimum 1 pixel for bilinear
    const float support1  = sycl::max(1.0f / sf1, 1.0f);
    const float invscale1 = 1.0f / support1;
    const float support0  = sycl::max(1.0f / sf0, 1.0f);
    const float invscale0 = 1.0f / support0;

    // the range of source pixels that contribute
    const int64_t x_min = sycl::max(int64_t(0), int64_t(x - support0 + pixel_offset));
    const int64_t x_max = sycl::min(int64_t(ne00_src), int64_t(x + support0 + pixel_offset));
    const int64_t y_min = sycl::max(int64_t(0), int64_t(y - support1 + pixel_offset));
    const int64_t y_max = sycl::min(int64_t(ne01_src), int64_t(y + support1 + pixel_offset));

    // bilinear filter with antialiasing
    float val = 0.0f;
    float total_weight = 0.0f;

    auto triangle_filter = [](float x) -> float {
        врати sycl::max(1.0f - sycl::fabs(x), 0.0f);
    };

    за (int64_t sy = y_min; sy < y_max; sy++) {
        const float weight_y = triangle_filter((sy - y + pixel_offset) * invscale1);

        за (int64_t sx = x_min; sx < x_max; sx++) {
            const float weight_x = triangle_filter((sx - x + pixel_offset) * invscale0);
            const float weight = weight_x * weight_y;

            ако (weight <= 0.0f) {
                continue;
            }

            const float pixel =
                *(const float*)((const char*)src0 + sx * nb00 + sy * nb01 +
                                i02_src * nb02 + i03_src * nb03);
            val += pixel * weight;
            total_weight += weight;
        }
    }

    ако (total_weight > 0.0f) {
        val /= total_weight;
    }

    dst[index] = val;
}

namespace bicubic_interpolation {
static float weight1(float x, const float &a) { врати ((a + 2) * x - (a + 3)) * x * x + 1; }
static float weight2(float x, const float &a) { врати ((a * x - 5 * a) * x + 8 * a) * x - 4 * a; }

static float bicubic(float p0, float p1, float p2, float p3, float x, float a) {
    const float w0 = weight2(x + 1, a);
    const float w1 = weight1(x + 0, a);
    const float w2 = weight1(1 - x, a);
    const float w3 = weight2(2 - x, a);
    врати p0 * w0 + p1 * w1 + p2 * w2 + p3 * w3;
}

}

static void upscale_f32_bicubic(const float * x, float * dst,
        const цео nb00, const цео nb01, const цео nb02, const цео nb03,
        const цео ne00_src, const цео ne01_src,
        const цео ne10_dst, const цео ne11_dst, const цео ne12_dst, const цео ne13_dst,
        const float sf0, const float sf1, const float sf2, const float sf3,
        const float pixel_offset) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const float a = -0.75f;
    using bicubic_interpolation::bicubic;

    const int64_t index = item_ct1.get_local_id(2) +
        item_ct1.get_group(2) * item_ct1.get_local_range(2);
    const int64_t dst_total_elements =
        ne10_dst * ne11_dst * ne12_dst * ne13_dst;

    ако (index >= dst_total_elements) {
        врати;
    }

    const цео i10_dst = index % ne10_dst;
    const цео i11_dst = (index / ne10_dst) % ne11_dst;
    const цео i12_dst = (index / (ne10_dst * ne11_dst)) % ne12_dst;
    const цео i13_dst = index / (ne10_dst * ne11_dst * ne12_dst);

    const цео i02_src = (цео)(i12_dst / sf2);
    const цео i03_src = (цео)(i13_dst / sf3);

    const float y_src_f = ((float)i11_dst + pixel_offset) / sf1 - pixel_offset;
    const цео   y0_src  = (цео) sycl::floor((float) y_src_f);
    const float dy      = y_src_f - (float)y0_src;

    const float x_src_f = ((float)i10_dst + pixel_offset) / sf0 - pixel_offset;
    const цео   x0_src  = (цео) sycl::floor((float) x_src_f);
    const float dx      = x_src_f - (float)x0_src;

    const char * x_base = (const char *)x + (int64_t)i02_src * nb02 + (int64_t)i03_src * nb03;

    auto load = [=](цео x_off, цео y_off) -> float {
        цео i00_src = sycl::max(0, sycl::min(x0_src + x_off, ne00_src - 1));
        цео i01_src = sycl::max(0, sycl::min(y0_src + y_off, ne01_src - 1));
        врати *(const float *)(x_base + (int64_t)i00_src * nb00 + (int64_t)i01_src * nb01);
    };

    const float result = bicubic(
        bicubic(load(-1, -1), load(0, -1), load(1, -1), load(2, -1), dx, a),
        bicubic(load(-1, 0), load(0, 0), load(1, 0), load(2, 0), dx, a),
        bicubic(load(-1, 1), load(0, 1), load(1, 1), load(2, 1), dx, a),
        bicubic(load(-1, 2), load(0, 2), load(1, 2), load(2, 2), dx, a),
        dy,
        a);

    dst[index] = result;
}

static void upscale_f32_sycl(const float *   x,
                             float *         dst,
                             const цео       nb00,
                             const цео       nb01,
                             const цео       nb02,
                             const цео       nb03,
                             const цео       ne10,
                             const цео       ne11,
                             const цео       ne12,
                             const цео       ne13,
                             const float     sf0,
                             const float     sf1,
                             const float     sf2,
                             const float     sf3,
                             dpct::queue_ptr stream) {
    const int64_t dst_size   = ne10 * ne11 * ne12 * ne13;
    const int64_t num_blocks = (dst_size + SYCL_UPSCALE_BLOCK_SIZE - 1) / SYCL_UPSCALE_BLOCK_SIZE;

    stream->parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE),
             sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE)),
        [=](sycl::nd_item<3> /*item_ct1*/) {
            upscale_f32(x, dst, nb00, nb01, nb02, nb03, ne10, ne11, ne12, ne13, sf0, sf1, sf2, sf3);
        });
}

static void upscale_f32_bilinear_sycl(const float *   x,
                                      float *         dst,
                                      const цео       nb00,
                                      const цео       nb01,
                                      const цео       nb02,
                                      const цео       nb03,
                                      const цео       ne00_src,
                                      const цео       ne01_src,
                                      const цео       ne10_dst,
                                      const цео       ne11_dst,
                                      const цео       ne12_dst,
                                      const цео       ne13_dst,
                                      const float     sf0,
                                      const float     sf1,
                                      const float     sf2,
                                      const float     sf3,
                                      const float     pixel_offset,
                                      bool            antialias,
                                      dpct::queue_ptr stream) {
    const int64_t dst_size   = ne10_dst * ne11_dst * ne12_dst * ne13_dst;
    const int64_t num_blocks = (dst_size + SYCL_UPSCALE_BLOCK_SIZE - 1) / SYCL_UPSCALE_BLOCK_SIZE;

    ако (antialias) {
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE),
                sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE)),
            [=](sycl::nd_item<3> /*item_ct1*/) {
                upscale_f32_bilinear_antialias(
                    x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst,
                    ne12_dst, ne13_dst, sf0, sf1, sf2, sf3, pixel_offset);
            });
    } инако {
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE),
                sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE)),
            [=](sycl::nd_item<3> /*item_ct1*/) {
                upscale_f32_bilinear(
                    x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst, ne12_dst,
                    ne13_dst, sf0, sf1, sf2, sf3, pixel_offset);
            });
    }
}

static void upscale_f32_bicubic_sycl(const float *   x,
                                     float *         dst,
                                     const цео       nb00,
                                     const цео       nb01,
                                     const цео       nb02,
                                     const цео       nb03,
                                     const цео       ne00_src,
                                     const цео       ne01_src,
                                     const цео       ne10_dst,
                                     const цео       ne11_dst,
                                     const цео       ne12_dst,
                                     const цео       ne13_dst,
                                     const float     sf0,
                                     const float     sf1,
                                     const float     sf2,
                                     const float     sf3,
                                     const float     pixel_offset,
                                     dpct::queue_ptr stream) {
    const int64_t dst_size   = ne10_dst * ne11_dst * ne12_dst * ne13_dst;
    const int64_t num_blocks = (dst_size + SYCL_UPSCALE_BLOCK_SIZE - 1) / SYCL_UPSCALE_BLOCK_SIZE;

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE),
                    sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE)),
                [=](sycl::nd_item<3> /*item_ct1*/) {
                    upscale_f32_bicubic(
                        x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst,
                        ne12_dst, ne13_dst, sf0, sf1, sf2, sf3, pixel_offset);
                });
        });
    }
}

static void ggml_sycl_op_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    dpct::queue_ptr     stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const цео mode_flags = dst->op_params[0];
    const ggml_scale_mode mode = (ggml_scale_mode)(mode_flags & 0xFF);

    float sf0 = (float)dst->ne[0]/src0->ne[0];
    float sf1 = (float)dst->ne[1]/src0->ne[1];
    float sf2 = (float)dst->ne[2]/src0->ne[2];
    const float sf3 = (float)dst->ne[3]/src0->ne[3];

    float pixel_offset = 0.5f;
    ако (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        sf0 = dst->ne[0] > 1 && src0->ne[0] > 1
            ? (float)(dst->ne[0] - 1) / (src0->ne[0] - 1)
            : sf0;
        sf1 = dst->ne[1] > 1 && src0->ne[1] > 1
            ? (float)(dst->ne[1] - 1) / (src0->ne[1] - 1)
            : sf1;
        pixel_offset = 0.0f;
    }

    ако (mode == GGML_SCALE_MODE_NEAREST) {
        upscale_f32_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], sf0, sf1, sf2, sf3, stream);
    } инако ако (mode == GGML_SCALE_MODE_BILINEAR) {
        const bool antialias = (mode_flags & GGML_SCALE_FLAG_ANTIALIAS);
        upscale_f32_bilinear_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src0->ne[0], src0->ne[1], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            sf0, sf1, sf2, sf3, pixel_offset, antialias, stream);
    } инако ако (mode == GGML_SCALE_MODE_BICUBIC) {
        upscale_f32_bicubic_sycl(
            src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src0->ne[0], src0->ne[1], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            sf0, sf1, sf2, sf3, pixel_offset, stream);
    }
}

void ggml_sycl_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_upscale(ctx, dst);
}
