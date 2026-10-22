#ଅନ୍ତର୍ଭାଵ "rvv_kernels.h"

#ଅନ୍ତର୍ଭାଵ "common.h"
#ଅନ୍ତର୍ଭାଵ "ggml.h"
#ଅନ୍ତର୍ଭାଵ "ops.h"
#ଅନ୍ତର୍ଭାଵ "string.h"

#ଅନ୍ତର୍ଭାଵ <algorithm>
#ଅନ୍ତର୍ଭାଵ <cmath>
#ଅନ୍ତର୍ଭାଵ <cstdint>
#ଅନ୍ତର୍ଭାଵ <stdexcept>

#ଯଦି !defined(__riscv_v) || !defined(__riscv_v_intrinsic)
#    error "riscv v extension or v_intrinsic not enabled"
#ଅନ୍ୟଥା
#    include <riscv_vector.h>
#endif

#ଯଦି !defined(__riscv_zfh)
#    error "riscv zfh extension not enabled"
#endif

#ଯଦି defined(__GNUC__)
#    pragma GCC diagnostic ignored "-Woverlength-strings"
#    pragma GCC diagnostic ignored "-Wcast-qual"
#    pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

ନାମାକାଶ spacemit_kernels::rvv {

ନାମାକାଶ {

auto align_up(size_t value, size_t alignment) {
    ଫେରନ୍ତୁ (value + alignment - 1) / alignment * alignment;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ତାର୍କିକ flash_attn_ext_supported_d_vlen1024_vf16(int64_t d) {
    ଫେରନ୍ତୁ d > 0 && d <= 128;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ତାର୍କିକ flash_attn_ext_supported_shape_vlen1024_vf16(int64_t DK, int64_t DV) {
    ଫେରନ୍ତୁ flash_attn_ext_supported_d_vlen1024_vf16(DK) && flash_attn_ext_supported_d_vlen1024_vf16(DV);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ reduce_sum_f32m4_vlen1024(vfloat32m4_t v, size_t vl) {
    vfloat32m1_t s_v = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    s_v              = __riscv_vfredusum_vs_f32m4_f32m1(v, s_v, vl);
    ଫେରନ୍ତୁ __riscv_vfmv_f_s_f32m1_f32(s_v);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ reduce_sum_f32m2_vlen1024(vfloat32m2_t v, size_t vl) {
    vfloat32m1_t s_v = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    s_v              = __riscv_vfredusum_vs_f32m2_f32m1(v, s_v, vl);
    ଫେରନ୍ତୁ __riscv_vfmv_f_s_f32m1_f32(s_v);
}

// Adapted from ggml_v_expf_m2 in vec.h. This is accurate enough for softmax.
ସ୍ଥିର ଧାଡ଼ିଭିତରେ vfloat32m2_t rvv_expf_approx_f32m2(vfloat32m2_t x, size_t vl) {
    ଅଚଳ vfloat32m2_t r = __riscv_vfmv_v_f_f32m2(0x1.8p23f, vl);
    ଅଚଳ vfloat32m2_t z = __riscv_vfmacc_vf_f32m2(r, 0x1.715476p+0f, x, vl);
    ଅଚଳ vfloat32m2_t n = __riscv_vfsub_vv_f32m2(z, r, vl);
    ଅଚଳ vfloat32m2_t b =
        __riscv_vfnmsac_vf_f32m2(__riscv_vfnmsac_vf_f32m2(x, 0x1.62e4p-1f, n, vl), 0x1.7f7d1cp-20f, n, vl);
    ଅଚଳ vuint32m2_t  e = __riscv_vsll_vx_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(z), 23, vl);
    ଅଚଳ vfloat32m2_t k = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vadd_vx_u32m2(e, 0x3f800000, vl));
    ଅଚଳ vbool16_t    c = __riscv_vmfgt_vf_f32m2_b16(__riscv_vfabs_v_f32m2(n, vl), 126.0f, vl);
    ଅଚଳ vfloat32m2_t u = __riscv_vfmul_vv_f32m2(b, b, vl);
    ଅଚଳ vfloat32m2_t j = __riscv_vfmacc_vv_f32m2(
        __riscv_vfmul_vf_f32m2(b, 0x1.ffffecp-1f, vl),
        __riscv_vfmacc_vv_f32m2(
            __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(0x1.fffdb6p-2f, vl), 0x1.555e66p-3f, b, vl),
            __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(0x1.573e2ep-5f, vl), 0x1.0e4020p-7f, b, vl), u, vl),
        u, vl);

    ଯଦି (!__riscv_vcpop_m_b16(c, vl)) {
        ଫେରନ୍ତୁ __riscv_vfmacc_vv_f32m2(k, j, k, vl);
    }

    ଅଚଳ vbool16_t    dm = __riscv_vmfle_vf_f32m2_b16(n, 0.0f, vl);
    ଅଚଳ vuint32m2_t  d  = __riscv_vmerge_vxm_u32m2(__riscv_vmv_v_x_u32m2(0, vl), 0x82000000, dm, vl);
    ଅଚଳ vfloat32m2_t s1 = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vadd_vx_u32m2(d, 0x7f000000, vl));
    ଅଚଳ vfloat32m2_t s2 = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vsub_vv_u32m2(e, d, vl));
    ଅଚଳ vfloat32m2_t r1 =
        __riscv_vmerge_vvm_f32m2(__riscv_vfmacc_vv_f32m2(k, k, j, vl),
                                 __riscv_vfmul_vv_f32m2(__riscv_vfmacc_vv_f32m2(s2, s2, j, vl), s1, vl), c, vl);
    ଫେରନ୍ତୁ __riscv_vmerge_vvm_f32m2(r1, __riscv_vfmul_vv_f32m2(s1, s1, vl),
                                    __riscv_vmfgt_vf_f32m2_b16(__riscv_vfabs_v_f32m2(n, vl), 192.0f, vl), vl);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ vfloat32m2_t rvv_tanh_approx_f32m2(vfloat32m2_t x, size_t vl) {
    ଅଚଳ vfloat32m2_t abs_x       = __riscv_vfabs_v_f32m2(x, vl);
    ଅଚଳ vfloat32m2_t neg_2_abs   = __riscv_vfmul_vf_f32m2(abs_x, -2.0f, vl);
    ଅଚଳ vfloat32m2_t exp_term    = rvv_expf_approx_f32m2(neg_2_abs, vl);
    ଅଚଳ vfloat32m2_t numerator   = __riscv_vfsub_vf_f32m2(exp_term, 1.0f, vl);
    ଅଚଳ vfloat32m2_t denominator = __riscv_vfadd_vf_f32m2(exp_term, 1.0f, vl);
    ଅଚଳ vfloat32m2_t tanh_abs    = __riscv_vfneg_v_f32m2(__riscv_vfdiv_vv_f32m2(numerator, denominator, vl), vl);
    ଅଚଳ vbool16_t    neg_mask    = __riscv_vmflt_vf_f32m2_b16(x, 0.0f, vl);
    ଅଚଳ vfloat32m2_t tanh_neg    = __riscv_vfneg_v_f32m2(tanh_abs, vl);
    ଫେରନ୍ତୁ __riscv_vmerge_vvm_f32m2(tanh_abs, tanh_neg, neg_mask, vl);
}

ସ୍ଥିର ଶୂନ୍ୟ rvv_softcap_tanh_inplace_f32(ଦଶମିକ * dst, int64_t dst_stride, int64_t tile_rows, int64_t n, ଦଶମିକ softcap) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା tq = 0; tq < tile_rows; ++tq, dst += dst_stride) {
        ଦଶମିକ * dst_row   = dst;
        int64_t remaining = n;
        ଯେତେବେଳେ (remaining > 0) {
            ଅଚଳ size_t vl = __riscv_vsetvl_e32m2(remaining);
            vfloat32m2_t v  = __riscv_vle32_v_f32m2(dst_row, vl);
            v               = rvv_tanh_approx_f32m2(v, vl);
            v               = __riscv_vfmul_vf_f32m2(v, softcap, vl);
            __riscv_vse32_v_f32m2(dst_row, v, vl);
            dst_row += vl;
            remaining -= vl;
        }
    }
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ rvv_softmax_exp_inplace_f32(ଦଶମିକ * dst, int64_t n, ଦଶମିକ max_ମୂଲ୍ୟ) {
    ଦଶମିକ row_sum = 0.0f;
    ଯେତେବେଳେ (n > 0) {
        ଅଚଳ size_t vl = __riscv_vsetvl_e32m2(n);
        vfloat32m2_t v  = __riscv_vle32_v_f32m2(dst, vl);
        v               = __riscv_vfsub_vf_f32m2(v, max_ମୂଲ୍ୟ, vl);
        v               = rvv_expf_approx_f32m2(v, vl);
        __riscv_vse32_v_f32m2(dst, v, vl);
        row_sum += reduce_sum_f32m2_vlen1024(v, vl);
        dst += vl;
        n -= vl;
    }
    ଫେରନ୍ତୁ row_sum;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ rvv_add_max_inplace_f32(ଦଶମିକ * dst, ଅଚଳ ଦଶମିକ * src, int64_t n) {
    ଦଶମିକ max_val = -INFINITY;
    ଯେତେବେଳେ (n > 0) {
        ଅଚଳ size_t vl   = __riscv_vsetvl_e32m4(n);
        vfloat32m4_t vdst = __riscv_vle32_v_f32m4(dst, vl);
        vfloat32m4_t vsrc = __riscv_vle32_v_f32m4(src, vl);
        vdst              = __riscv_vfadd_vv_f32m4(vdst, vsrc, vl);
        __riscv_vse32_v_f32m4(dst, vdst, vl);

        vfloat32m1_t seed = __riscv_vfmv_v_f_f32m1(max_val, 1);
        seed              = __riscv_vfredmax_vs_f32m4_f32m1(vdst, seed, vl);
        max_val           = __riscv_vfmv_f_s_f32m1_f32(seed);

        dst += vl;
        src += vl;
        n -= vl;
    }
    ଫେରନ୍ତୁ max_val;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ rvv_softcap_add_max_inplace_f32(ଦଶମିକ * dst, ଅଚଳ ଦଶମିକ * src, int64_t n, ଦଶମିକ softcap) {
    ଯଦି (softcap == 0.0f) {
        ଫେରନ୍ତୁ rvv_add_max_inplace_f32(dst, src, n);
    }

    ଦଶମିକ max_val = -INFINITY;
    ଯେତେବେଳେ (n > 0) {
        ଅଚଳ size_t vl   = __riscv_vsetvl_e32m2(n);
        vfloat32m2_t vdst = __riscv_vle32_v_f32m2(dst, vl);
        vfloat32m2_t vsrc = __riscv_vle32_v_f32m2(src, vl);
        vdst              = rvv_tanh_approx_f32m2(vdst, vl);
        vdst              = __riscv_vfmul_vf_f32m2(vdst, softcap, vl);
        vdst              = __riscv_vfadd_vv_f32m2(vdst, vsrc, vl);
        __riscv_vse32_v_f32m2(dst, vdst, vl);

        vfloat32m1_t seed = __riscv_vfmv_v_f_f32m1(max_val, 1);
        seed              = __riscv_vfredmax_vs_f32m2_f32m1(vdst, seed, vl);
        max_val           = __riscv_vfmv_f_s_f32m1_f32(seed);

        dst += vl;
        src += vl;
        n -= vl;
    }
    ଫେରନ୍ତୁ max_val;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ rvv_zero_f32(ଦଶମିକ * dst, int64_t n) {
    ଯେତେବେଳେ (n > 0) {
        ଅଚଳ size_t       vl = __riscv_vsetvl_e32m4(n);
        ଅଚଳ vfloat32m4_t z  = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        __riscv_vse32_v_f32m4(dst, z, vl);
        dst += vl;
        n -= vl;
    }
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ rvv_scale_f32(ଦଶମିକ * dst, ଦଶମିକ scale, int64_t n) {
    ଯେତେବେଳେ (n > 0) {
        ଅଚଳ size_t vl = __riscv_vsetvl_e32m4(n);
        vfloat32m4_t v  = __riscv_vle32_v_f32m4(dst, vl);
        v               = __riscv_vfmul_vf_f32m4(v, scale, vl);
        __riscv_vse32_v_f32m4(dst, v, vl);
        dst += vl;
        n -= vl;
    }
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ rvv_add_inplace_f32(ଦଶମିକ *       dst,
                                       int64_t       dst_stride,
                                       ଅଚଳ ଦଶମିକ * src,
                                       int64_t       src_stride,
                                       int64_t       tile_rows,
                                       int64_t       n) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା tq = 0; tq < tile_rows; ++tq, dst += dst_stride, src += src_stride) {
        int64_t       remaining = n;
        ଦଶମିକ *       dst_row   = dst;
        ଅଚଳ ଦଶମିକ * src_row   = src;
        ଯେତେବେଳେ (remaining > 0) {
            ଅଚଳ size_t vl   = __riscv_vsetvl_e32m4(remaining);
            vfloat32m4_t vdst = __riscv_vle32_v_f32m4(dst_row, vl);
            vfloat32m4_t vsrc = __riscv_vle32_v_f32m4(src_row, vl);
            vdst              = __riscv_vfadd_vv_f32m4(vdst, vsrc, vl);
            __riscv_vse32_v_f32m4(dst_row, vdst, vl);
            dst_row += vl;
            src_row += vl;
            remaining -= vl;
        }
    }
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ rvv_max_f32(ଅଚଳ ଦଶମିକ * src, int64_t n) {
    ଦଶମିକ max_val = -INFINITY;
    ଯେତେବେଳେ (n > 0) {
        ଅଚଳ size_t       vl   = __riscv_vsetvl_e32m4(n);
        ଅଚଳ vfloat32m4_t v    = __riscv_vle32_v_f32m4(src, vl);
        vfloat32m1_t       seed = __riscv_vfmv_v_f_f32m1(max_val, 1);
        seed                    = __riscv_vfredmax_vs_f32m4_f32m1(v, seed, vl);
        max_val                 = __riscv_vfmv_f_s_f32m1_f32(seed);
        src += vl;
        n -= vl;
    }
    ଫେରନ୍ତୁ max_val;
}

ସ୍ଥିର ଶୂନ୍ୟ rvv_pack_f32_as_scaled_f16(ଶୂନ୍ୟ *       dst,
                                       int64_t      dst_row_stride,
                                       ଅଚଳ ଶୂନ୍ୟ * src,
                                       int64_t      src_row_stride,
                                       int64_t      tile_rows,
                                       int64_t      n,
                                       ଦଶମିକ        scale) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା tq = 0; tq < tile_rows; ++tq) {
        ଅଚଳ ଦଶମିକ * row_ptr     = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src + tq * src_row_stride);
        _Float16 *    dst_row_ptr = (_Float16 *) ((ଅକ୍ଷର *) dst + tq * dst_row_stride);
        int64_t       remaining   = n;
        ଯେତେବେଳେ (remaining > 0) {
            ଅଚଳ size_t vl        = __riscv_vsetvl_e32m4(remaining);
            vfloat32m4_t v32       = __riscv_vle32_v_f32m4(row_ptr, vl);
            v32                    = __riscv_vfmul_vf_f32m4(v32, scale, vl);
            ଅଚଳ vfloat16m2_t v16 = __riscv_vfncvt_f_f_w_f16m2(v32, vl);
            __riscv_vse16_v_f16m2(dst_row_ptr, v16, vl);
            dst_row_ptr += vl;
            row_ptr += vl;
            remaining -= vl;
        }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ rvv_pack_scaled_f16_as_f32(ଶୂନ୍ୟ *       dst,
                                       int64_t      dst_row_stride,
                                       ଅଚଳ ଶୂନ୍ୟ * src,
                                       int64_t      src_row_stride,
                                       int64_t      tile_rows,
                                       int64_t      n,
                                       ଦଶମିକ        scale) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା tq = 0; tq < tile_rows; ++tq) {
        ଅଚଳ _Float16 * row_ptr     = (ଅଚଳ _Float16 *) ((ଅଚଳ ଅକ୍ଷର *) src + tq * src_row_stride);
        ଦଶମିକ *          dst_row_ptr = (ଦଶମିକ *) ((ଅକ୍ଷର *) dst + tq * dst_row_stride);
        int64_t          remaining   = n;
        ଯେତେବେଳେ (remaining > 0) {
            ଅଚଳ size_t       vl  = __riscv_vsetvl_e16m2(remaining);
            ଅଚଳ vfloat16m2_t v16 = __riscv_vle16_v_f16m2(row_ptr, vl);
            vfloat32m4_t       v32 = __riscv_vfwcvt_f_f_v_f32m4(v16, vl);
            v32                    = __riscv_vfmul_vf_f32m4(v32, scale, vl);
            __riscv_vse32_v_f32m4(dst_row_ptr, v32, vl);
            dst_row_ptr += vl;
            row_ptr += vl;
            remaining -= vl;
        }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ rvv_pack_scaled_f32_as_f32(ଶୂନ୍ୟ *       dst,
                                       int64_t      dst_row_stride,
                                       ଅଚଳ ଶୂନ୍ୟ * src,
                                       int64_t      src_row_stride,
                                       int64_t      tile_rows,
                                       int64_t      n,
                                       ଦଶମିକ *      scale) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା tq = 0; tq < tile_rows; ++tq) {
        ଅଚଳ ଦଶମିକ * row_ptr     = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src + tq * src_row_stride);
        ଦଶମିକ *       dst_row_ptr = (ଦଶମିକ *) ((ଅକ୍ଷର *) dst + tq * dst_row_stride);
        int64_t       remaining   = n;
        ଯେତେବେଳେ (remaining > 0) {
            ଅଚଳ size_t vl  = __riscv_vsetvl_e32m4(remaining);
            vfloat32m4_t v32 = __riscv_vle32_v_f32m4(row_ptr, vl);
            v32              = __riscv_vfmul_vf_f32m4(v32, scale[tq], vl);
            __riscv_vse32_v_f32m4(dst_row_ptr, v32, vl);
            dst_row_ptr += vl;
            row_ptr += vl;
            remaining -= vl;
        }
    }
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ rvv_transposed_s32_mn_to_nm(int8_t * dst,
                                               int64_t  n_dst_stride,
                                               int8_t * src,
                                               int64_t  m_src_stride,
                                               int64_t  m,
                                               int64_t  n) {
    int8_t * in  = src;
    int8_t * out = dst;

    __asm__ ଅସ୍ଥିର(
        "vsetvli                t0, zero, e32, m1, tu, mu     \n\t"
        "mul                    t3, t0, %[os0]                \n\t"
        "srli                   t2, %[isz0], 3                \n\t"
        "blez                   t2, M1%=                      \n\t"

        "LOOP_M8%=:                                           \n\t"
        "addi                   a1, %[dst], 0                 \n\t"
        "addi                   s1, %[src], 0                 \n\t"
        "add                    s2, %[src], %[is0]            \n\t"
        "add                    s3, s2, %[is0]                \n\t"
        "add                    s4, s3, %[is0]                \n\t"
        "add                    s5, s4, %[is0]                \n\t"
        "add                    s6, s5, %[is0]                \n\t"
        "add                    s7, s6, %[is0]                \n\t"
        "add                    s8, s7, %[is0]                \n\t"
        "addi                   t1, %[isz1], 0                \n\t"

        "LOOP_M8N%=:                                          \n\t"
        "vsetvli                t0, t1, e32, m1, tu, mu       \n\t"
        "sub                    t1, t1, t0                    \n\t"
        "vle32.v                v0, (s1)                      \n\t"
        "sh2add                 s1, t0, s1                    \n\t"
        "vle32.v                v1, (s2)                      \n\t"
        "sh2add                 s2, t0, s2                    \n\t"
        "vle32.v                v2, (s3)                      \n\t"
        "sh2add                 s3, t0, s3                    \n\t"
        "vle32.v                v3, (s4)                      \n\t"
        "sh2add                 s4, t0, s4                    \n\t"
        "vle32.v                v4, (s5)                      \n\t"
        "sh2add                 s5, t0, s5                    \n\t"
        "vle32.v                v5, (s6)                      \n\t"
        "sh2add                 s6, t0, s6                    \n\t"
        "vle32.v                v6, (s7)                      \n\t"
        "sh2add                 s7, t0, s7                    \n\t"
        "vle32.v                v7, (s8)                      \n\t"
        "sh2add                 s8, t0, s8                    \n\t"
        "vssseg8e32.v           v0, (a1), %[os0]              \n\t"
        "add                    a1, a1, t3                    \n\t"
        "bnez                   t1, LOOP_M8N%=                \n\t"
        "sh3add                 %[src], %[is0], %[src]        \n\t"
        "addi                   %[dst], %[dst], 32            \n\t"
        "addi                   t2, t2, -1                    \n\t"
        "bnez                   t2, LOOP_M8%=                 \n\t"

        "M1%=:                                                \n\t"
        "andi                   t2, %[isz0], 7                \n\t"
        "blez                   t2, END%=                     \n\t"

        "LOOP_M1%=:                                           \n\t"
        "addi                   a1, %[dst], 0                 \n\t"
        "addi                   s1, %[src], 0                 \n\t"
        "addi                   t1, %[isz1], 0                \n\t"

        "LOOP_M1N%=:                                          \n\t"
        "vsetvli                t0, t1, e32, m1, tu, mu       \n\t"
        "sub                    t1, t1, t0                    \n\t"
        "vle32.v                v0, (s1)                      \n\t"
        "sh2add                 s1, t0, s1                    \n\t"
        "vsse32.v               v0, (a1), %[os0]              \n\t"
        "add                    a1, a1, t3                    \n\t"
        "bnez                   t1, LOOP_M1N%=                \n\t"
        "add                    %[src], %[is0], %[src]        \n\t"
        "addi                   %[dst], %[dst], 4             \n\t"
        "addi                   t2, t2, -1                    \n\t"
        "bnez                   t2, LOOP_M1%=                 \n\t"
        "END%=:                                               \n\t"

        : [src] "+r"(in), [dst] "+r"(out), [isz0] "+r"(m)
        : [isz1] "r"(n), [is0] "r"(m_src_stride), [os0] "r"(n_dst_stride)
        : "cc", "t0", "t1", "t2", "t3", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "a1");
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ rvv_transposed_s16_mn_to_nm(int8_t * dst,
                                               int64_t  n_dst_stride,
                                               int8_t * src,
                                               int64_t  m_src_stride,
                                               int64_t  m,
                                               int64_t  n) {
    int8_t * in  = src;
    int8_t * out = dst;

    __asm__ ଅସ୍ଥିର(
        "vsetvli                t0, zero, e16, m1, tu, mu     \n\t"
        "mul                    t3, t0, %[os0]                \n\t"
        "srli                   t2, %[isz0], 3                \n\t"
        "blez                   t2, M1%=                      \n\t"

        "LOOP_M8%=:                                           \n\t"
        "addi                   a1, %[dst], 0                 \n\t"
        "addi                   s1, %[src], 0                 \n\t"
        "add                    s2, %[src], %[is0]            \n\t"
        "add                    s3, s2, %[is0]                \n\t"
        "add                    s4, s3, %[is0]                \n\t"
        "add                    s5, s4, %[is0]                \n\t"
        "add                    s6, s5, %[is0]                \n\t"
        "add                    s7, s6, %[is0]                \n\t"
        "add                    s8, s7, %[is0]                \n\t"
        "addi                   t1, %[isz1], 0                \n\t"

        "LOOP_M8N%=:                                          \n\t"
        "vsetvli                t0, t1, e16, m1, tu, mu       \n\t"
        "sub                    t1, t1, t0                    \n\t"
        "vle16.v                v0, (s1)                      \n\t"
        "sh1add                 s1, t0, s1                    \n\t"
        "vle16.v                v1, (s2)                      \n\t"
        "sh1add                 s2, t0, s2                    \n\t"
        "vle16.v                v2, (s3)                      \n\t"
        "sh1add                 s3, t0, s3                    \n\t"
        "vle16.v                v3, (s4)                      \n\t"
        "sh1add                 s4, t0, s4                    \n\t"
        "vle16.v                v4, (s5)                      \n\t"
        "sh1add                 s5, t0, s5                    \n\t"
        "vle16.v                v5, (s6)                      \n\t"
        "sh1add                 s6, t0, s6                    \n\t"
        "vle16.v                v6, (s7)                      \n\t"
        "sh1add                 s7, t0, s7                    \n\t"
        "vle16.v                v7, (s8)                      \n\t"
        "sh1add                 s8, t0, s8                    \n\t"
        "vssseg8e16.v           v0, (a1), %[os0]              \n\t"
        "add                    a1, a1, t3                    \n\t"
        "bnez                   t1, LOOP_M8N%=                \n\t"
        "sh3add                 %[src], %[is0], %[src]        \n\t"
        "addi                   %[dst], %[dst], 16            \n\t"
        "addi                   t2, t2, -1                    \n\t"
        "bnez                   t2, LOOP_M8%=                 \n\t"

        "M1%=:                                                \n\t"
        "andi                   t2, %[isz0], 7                \n\t"
        "blez                   t2, END%=                     \n\t"

        "LOOP_M1%=:                                           \n\t"
        "addi                   a1, %[dst], 0                 \n\t"
        "addi                   s1, %[src], 0                 \n\t"
        "addi                   t1, %[isz1], 0                \n\t"

        "LOOP_M1N%=:                                          \n\t"
        "vsetvli                t0, t1, e16, m1, tu, mu       \n\t"
        "sub                    t1, t1, t0                    \n\t"
        "vle16.v                v0, (s1)                      \n\t"
        "sh1add                 s1, t0, s1                    \n\t"
        "vsse16.v               v0, (a1), %[os0]              \n\t"
        "add                    a1, a1, t3                    \n\t"
        "bnez                   t1, LOOP_M1N%=                \n\t"
        "add                    %[src], %[is0], %[src]        \n\t"
        "addi                   %[dst], %[dst], 2             \n\t"
        "addi                   t2, t2, -1                    \n\t"
        "bnez                   t2, LOOP_M1%=                 \n\t"
        "END%=:                                               \n\t"

        : [src] "+r"(in), [dst] "+r"(out), [isz0] "+r"(m)
        : [isz1] "r"(n), [is0] "r"(m_src_stride), [os0] "r"(n_dst_stride)
        : "cc", "t0", "t1", "t2", "t3", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "a1");
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ rvv_qk_dot_tile_f16_x1(ଦଶମିକ *          dst,
                                          ଅଚଳ _Float16 * q_row,
                                          ଅଚଳ _Float16 * k_pack,
                                          int64_t          dk,
                                          int64_t          kv_tile) {
    ଅଚଳ size_t vl  = __riscv_vsetvl_e16m1(kv_tile);
    vfloat32m2_t acc = __riscv_vfmv_v_f_f32m2(0.0f, vl);

    ପାଇଁ (int64_t d = 0; d < dk; ++d) {
        ଅଚଳ vfloat16m1_t k_vec = __riscv_vle16_v_f16m1(k_pack + d * ggml_fa_tile_config::KV, vl);
        acc                      = __riscv_vfwmacc_vf_f32m2(acc, q_row[d], k_vec, vl);
    }

    __riscv_vse32_v_f32m2(dst, acc, vl);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ rvv_qk_dot_tile_f16_x4(ଦଶମିକ *          dst0,
                                          ଦଶମିକ *          dst1,
                                          ଦଶମିକ *          dst2,
                                          ଦଶମିକ *          dst3,
                                          ଅଚଳ _Float16 * q0,
                                          ଅଚଳ _Float16 * q1,
                                          ଅଚଳ _Float16 * q2,
                                          ଅଚଳ _Float16 * q3,
                                          ଅଚଳ _Float16 * k_pack,
                                          int64_t          dk,
                                          int64_t          kv_tile) {
    ଅଚଳ size_t vl   = __riscv_vsetvl_e16m1(kv_tile);
    vfloat32m2_t acc0 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    vfloat32m2_t acc1 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    vfloat32m2_t acc2 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    vfloat32m2_t acc3 = __riscv_vfmv_v_f_f32m2(0.0f, vl);

    ପାଇଁ (int64_t d = 0; d < dk; ++d) {
        ଅଚଳ vfloat16m1_t k_vec = __riscv_vle16_v_f16m1(k_pack + d * ggml_fa_tile_config::KV, vl);
        acc0                     = __riscv_vfwmacc_vf_f32m2(acc0, q0[d], k_vec, vl);
        acc1                     = __riscv_vfwmacc_vf_f32m2(acc1, q1[d], k_vec, vl);
        acc2                     = __riscv_vfwmacc_vf_f32m2(acc2, q2[d], k_vec, vl);
        acc3                     = __riscv_vfwmacc_vf_f32m2(acc3, q3[d], k_vec, vl);
    }

    __riscv_vse32_v_f32m2(dst0, acc0, vl);
    __riscv_vse32_v_f32m2(dst1, acc1, vl);
    __riscv_vse32_v_f32m2(dst2, acc2, vl);
    __riscv_vse32_v_f32m2(dst3, acc3, vl);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ rvv_pv_accumulate_f16_x1(ଦଶମିକ *          dst,
                                            ଅଚଳ ଦଶମିକ *    prob,
                                            ଅଚଳ _Float16 * v_pack,
                                            int64_t          kv_tile,
                                            int64_t          dv) {
    int64_t d_ବାମ = dv;
    int64_t d_off  = 0;

    ଯେତେବେଳେ (d_ବାମ > 0) {
        ଅଚଳ size_t vl  = __riscv_vsetvl_e16m2(d_ବାମ);
        vfloat32m4_t acc = __riscv_vle32_v_f32m4(dst + d_off, vl);

        ପାଇଁ (int64_t tk = 0; tk < kv_tile; ++tk) {
            ଅଚଳ vfloat16m2_t v16 = __riscv_vle16_v_f16m2(v_pack + tk * dv + d_off, vl);
            ଅଚଳ vfloat32m4_t v32 = __riscv_vfwcvt_f_f_v_f32m4(v16, vl);
            acc                    = __riscv_vfmacc_vf_f32m4(acc, prob[tk], v32, vl);
        }

        __riscv_vse32_v_f32m4(dst + d_off, acc, vl);
        d_ବାମ -= vl;
        d_off += vl;
    }
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ rvv_pv_accumulate_f16_x4(ଦଶମିକ *          dst0,
                                            ଦଶମିକ *          dst1,
                                            ଦଶମିକ *          dst2,
                                            ଦଶମିକ *          dst3,
                                            ଅଚଳ ଦଶମିକ *    prob0,
                                            ଅଚଳ ଦଶମିକ *    prob1,
                                            ଅଚଳ ଦଶମିକ *    prob2,
                                            ଅଚଳ ଦଶମିକ *    prob3,
                                            ଅଚଳ _Float16 * v_pack,
                                            int64_t          kv_tile,
                                            int64_t          dv) {
    int64_t d_ବାମ = dv;
    int64_t d_off  = 0;

    ଯେତେବେଳେ (d_ବାମ > 0) {
        ଅଚଳ size_t vl   = __riscv_vsetvl_e16m2(d_ବାମ);
        vfloat32m4_t acc0 = __riscv_vle32_v_f32m4(dst0 + d_off, vl);
        vfloat32m4_t acc1 = __riscv_vle32_v_f32m4(dst1 + d_off, vl);
        vfloat32m4_t acc2 = __riscv_vle32_v_f32m4(dst2 + d_off, vl);
        vfloat32m4_t acc3 = __riscv_vle32_v_f32m4(dst3 + d_off, vl);

        ପାଇଁ (int64_t tk = 0; tk < kv_tile; ++tk) {
            ଅଚଳ vfloat16m2_t v16 = __riscv_vle16_v_f16m2(v_pack + tk * dv + d_off, vl);
            ଅଚଳ vfloat32m4_t v32 = __riscv_vfwcvt_f_f_v_f32m4(v16, vl);
            acc0                   = __riscv_vfmacc_vf_f32m4(acc0, prob0[tk], v32, vl);
            acc1                   = __riscv_vfmacc_vf_f32m4(acc1, prob1[tk], v32, vl);
            acc2                   = __riscv_vfmacc_vf_f32m4(acc2, prob2[tk], v32, vl);
            acc3                   = __riscv_vfmacc_vf_f32m4(acc3, prob3[tk], v32, vl);
        }

        __riscv_vse32_v_f32m4(dst0 + d_off, acc0, vl);
        __riscv_vse32_v_f32m4(dst1 + d_off, acc1, vl);
        __riscv_vse32_v_f32m4(dst2 + d_off, acc2, vl);
        __riscv_vse32_v_f32m4(dst3 + d_off, acc3, vl);
        d_ବାମ -= vl;
        d_off += vl;
    }
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ rvv_qk_dot_tile(ଦଶମିକ *       dst,
                                   ଅଚଳ ଦଶମିକ * q_row,
                                   ଅଚଳ ଦଶମିକ * k_pack,
                                   int64_t       dk,
                                   int64_t       kv_tile,
                                   ଦଶମିକ         scale) {
    ଅଚଳ size_t vl  = __riscv_vsetvl_e32m4(kv_tile);
    vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);

    ପାଇଁ (int64_t d = 0; d < dk; ++d) {
        ଅଚଳ vfloat32m4_t k_vec = __riscv_vle32_v_f32m4(k_pack + d * kv_tile, vl);
        acc                      = __riscv_vfmacc_vf_f32m4(acc, q_row[d] * scale, k_vec, vl);
    }

    __riscv_vse32_v_f32m4(dst, acc, vl);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ rvv_pv_accumulate(ଦଶମିକ *       dst,
                                     ଅଚଳ ଦଶମିକ * prob,
                                     ଅଚଳ ଦଶମିକ * v_pack,
                                     int64_t       kv_tile,
                                     int64_t       dv) {
    int64_t d_ବାମ = dv;
    int64_t d_off  = 0;

    ଯେତେବେଳେ (d_ବାମ > 0) {
        ଅଚଳ size_t vl  = __riscv_vsetvl_e32m4(d_ବାମ);
        vfloat32m4_t acc = __riscv_vle32_v_f32m4(dst + d_off, vl);

        ପାଇଁ (int64_t tk = 0; tk < kv_tile; ++tk) {
            ଅଚଳ vfloat32m4_t v_vec = __riscv_vle32_v_f32m4(v_pack + tk * dv + d_off, vl);
            acc                      = __riscv_vfmacc_vf_f32m4(acc, prob[tk], v_vec, vl);
        }

        __riscv_vse32_v_f32m4(dst + d_off, acc, vl);
        d_ବାମ -= vl;
        d_off += vl;
    }
}

ସ୍ଥିର ଶୂନ୍ୟ permute_transpose_impl(ଅଚଳ ggml_tensor * src0,
                                   ggml_tensor *       dst,
                                   int64_t             batch,
                                   int64_t             m,
                                   int64_t             n,
                                   int64_t             batch_stride,
                                   int64_t             m_src_stride,
                                   int64_t             n_src_stride,
                                   int64_t             n_dst_stride,
                                   ପୂର୍ଣ୍ଣସଂଖ୍ୟା                 ith,
                                   ପୂର୍ଣ୍ଣସଂଖ୍ୟା                 nth) {
    GGML_ASSERT(n_src_stride == ଆକାର(int32_t) || n_src_stride == ଆକାର(int16_t));

    ଯଦି (n_src_stride == ଆକାର(int32_t)) {
        ପାଇଁ (int64_t bi = ith; bi < batch; bi += nth) {
            rvv_transposed_s32_mn_to_nm((int8_t *) ((ଅକ୍ଷର *) dst->data + bi * batch_stride), n_dst_stride,
                                        (int8_t *) ((ଅକ୍ଷର *) src0->data + bi * batch_stride), m_src_stride, m, n);
        }
    } ଅନ୍ୟଥା ଯଦି (n_src_stride == ଆକାର(int16_t)) {
        ପାଇଁ (int64_t bi = ith; bi < batch; bi += nth) {
            rvv_transposed_s16_mn_to_nm((int8_t *) ((ଅକ୍ଷର *) dst->data + bi * batch_stride), n_dst_stride,
                                        (int8_t *) ((ଅକ୍ଷର *) src0->data + bi * batch_stride), m_src_stride, m, n);
        }
    } ଅନ୍ୟଥା {
        GGML_ABORT("not implemented");
    }
}

template <size_t QLEN>
ସ୍ଥିର ଶୂନ୍ୟ flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_mrow(ଦଶମିକ **            pq,
                                                                  ଅଚଳ ଅକ୍ଷର *        k_data_row,
                                                                  ଅଚଳ ଅକ୍ଷର *        v_data_row,
                                                                  ଅଚଳ ggml_fp16_t * mp,
                                                                  ଦଶମିକ **            sinks,
                                                                  ଦଶମିକ **            dst,
                                                                  ଦଶମିକ               scale,
                                                                  ଦଶମିକ               logit_softcap,
                                                                  ଦଶମିକ               slope,
                                                                  int64_t             nek1,
                                                                  int64_t             nbk1,
                                                                  int64_t             nbv1,
                                                                  int64_t             DV,
                                                                  int64_t             DK,
                                                                  ଶୂନ୍ୟ *              tcm_buffer,
                                                                  size_t              tcm_buffer_size) {
    GGML_ASSERT(flash_attn_ext_supported_shape_vlen1024_vf16(DK, DV));
    ଦଶମିକ S[QLEN] = { 0.0f };       // sum
    ଦଶମିକ M[QLEN] = { -INFINITY };  // maximum KQ value

    _Float16 *   kq16_buffer          = (_Float16 *) tcm_buffer;
    _Float16 *   qv_buffer            = kq16_buffer + QLEN * DV;
    ଅଚଳ size_t qkv_temp_buffer_size = (QLEN * DV + QLEN * DK) * ଆକାର(_Float16);
    ଅକ୍ଷର *       kv_tile_buffer       = (ଅକ୍ଷର *) (qv_buffer + QLEN * DK);

    {
        vfloat16m2_t VKQ16_v = __riscv_vfmv_v_f_f16m2(0.0f, DV);
        ପାଇଁ (int64_t i = 0; i < QLEN; ++i) {
            __riscv_vse16_v_f16m2(kq16_buffer + i * DV, VKQ16_v, DV);
            vfloat16m2_t Q_q_v = __riscv_vfncvt_f_f_w_f16m2(__riscv_vle32_v_f32m4(pq[i], DK), DK);
            __riscv_vse16_v_f16m2(qv_buffer + i * DK, Q_q_v, DK);
        }
    }

    ଅଚଳ uintptr_t scratch_addr = reinterpret_cast<uintptr_t>(kv_tile_buffer);
    ଅଚଳ size_t    scratch_size = tcm_buffer_size > qkv_temp_buffer_size ? tcm_buffer_size - qkv_temp_buffer_size : 0;
    ଅଚଳ uintptr_t kq_tile_addr = align_up(scratch_addr, alignof(ଦଶମିକ));
    ଅଚଳ size_t    scratch_prefix = kq_tile_addr - scratch_addr;
    ଅଚଳ size_t    packed_tile_size =
        QLEN * ଆକାର(ଦଶମିକ) + DK * ଆକାର(_Float16) + DV * ଆକାର(_Float16) + ଆକାର(ଦଶମିକ);
    ଅଚଳ int64_t max_ic_tile_step = ((int64_t) __riscv_vsetvlmax_e16m1()) & ~((int64_t) 7);
    ଅଚଳ int64_t max_fit_by_tcm =
        scratch_size > scratch_prefix ? (int64_t) ((scratch_size - scratch_prefix) / packed_tile_size) : 0;
    ଅଚଳ int64_t ic_tile_step = std::min(max_ic_tile_step, max_fit_by_tcm) & ~((int64_t) 7);

    ଅଚଳ uintptr_t k_tile_addr  = kq_tile_addr + QLEN * ic_tile_step * ଆକାର(ଦଶମିକ);
    ଅଚଳ uintptr_t v_tile_addr  = k_tile_addr + DK * ic_tile_step * ଆକାର(_Float16);
    ଅଚଳ uintptr_t mv_tile_addr = v_tile_addr + ic_tile_step * DV * ଆକାର(_Float16);

    ଯଦି (ic_tile_step >= 8) {
        ଦଶମିକ *    kq_tile_buffer = reinterpret_cast<ଦଶମିକ *>(kq_tile_addr);
        _Float16 * k_tile_pack    = reinterpret_cast<_Float16 *>(k_tile_addr);
        _Float16 * v_tile_pack    = reinterpret_cast<_Float16 *>(v_tile_addr);
        ଦଶମିକ *    mv_tile_pack   = reinterpret_cast<ଦଶମିକ *>(mv_tile_addr);

        ଅଚଳ int64_t k_tile_byte_stride = ic_tile_step * (int64_t) ଆକାର(_Float16);

        int64_t ic_step = 0;
        ପାଇଁ (int64_t ic = 0; ic < nek1; ++ic) {
            ଅଚଳ ଦଶମିକ mv = mp ? slope * ((_Float16 *) mp)[ic] : 0.0f;

            ଯଦି (mv != -INFINITY) {
                ଅଚଳ _Float16 * k_data = (ଅଚଳ _Float16 *) (k_data_row + ic * nbk1);
                ଅଚଳ _Float16 * v_data = (ଅଚଳ _Float16 *) (v_data_row + ic * nbv1);

                ଅଚଳ vfloat16m2_t k_data_v = __riscv_vle16_v_f16m2(k_data, DK);
                ଅଚଳ vfloat16m2_t v_data_v = __riscv_vle16_v_f16m2(v_data, DV);
                __riscv_vsse16_v_f16m2(k_tile_pack + ic_step, k_tile_byte_stride, k_data_v, DK);
                __riscv_vse16_v_f16m2(v_tile_pack + ic_step * DV, v_data_v, DV);
                mv_tile_pack[ic_step] = mv;
                ic_step++;
            }

            ଯଦି (ic_step > 0 && (ic_step == ic_tile_step || ic == (nek1 - 1))) {
                ଯଦି constexpr (QLEN == 4) {
                    ଅଚଳ size_t qk_vl   = __riscv_vsetvl_e16m1(ic_step);
                    vfloat32m2_t qk_acc0 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);
                    vfloat32m2_t qk_acc1 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);
                    vfloat32m2_t qk_acc2 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);
                    vfloat32m2_t qk_acc3 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);

                    ପାଇଁ (int64_t d = 0; d < DK; ++d) {
                        ଅଚଳ vfloat16m1_t k_vec = __riscv_vle16_v_f16m1(k_tile_pack + d * ic_tile_step, qk_vl);
                        qk_acc0 = __riscv_vfwmacc_vf_f32m2(qk_acc0, qv_buffer[0 * DK + d], k_vec, qk_vl);
                        qk_acc1 = __riscv_vfwmacc_vf_f32m2(qk_acc1, qv_buffer[1 * DK + d], k_vec, qk_vl);
                        qk_acc2 = __riscv_vfwmacc_vf_f32m2(qk_acc2, qv_buffer[2 * DK + d], k_vec, qk_vl);
                        qk_acc3 = __riscv_vfwmacc_vf_f32m2(qk_acc3, qv_buffer[3 * DK + d], k_vec, qk_vl);
                    }

                    qk_acc0 = __riscv_vfmul_vf_f32m2(qk_acc0, scale, qk_vl);
                    qk_acc1 = __riscv_vfmul_vf_f32m2(qk_acc1, scale, qk_vl);
                    qk_acc2 = __riscv_vfmul_vf_f32m2(qk_acc2, scale, qk_vl);
                    qk_acc3 = __riscv_vfmul_vf_f32m2(qk_acc3, scale, qk_vl);

                    __riscv_vse32_v_f32m2(kq_tile_buffer + 0 * ic_tile_step, qk_acc0, qk_vl);
                    __riscv_vse32_v_f32m2(kq_tile_buffer + 1 * ic_tile_step, qk_acc1, qk_vl);
                    __riscv_vse32_v_f32m2(kq_tile_buffer + 2 * ic_tile_step, qk_acc2, qk_vl);
                    __riscv_vse32_v_f32m2(kq_tile_buffer + 3 * ic_tile_step, qk_acc3, qk_vl);
                } ଅନ୍ୟଥା {
                    static_assert(QLEN == 2, "unsupported QLEN");

                    ଅଚଳ size_t qk_vl   = __riscv_vsetvl_e16m1(ic_step);
                    vfloat32m2_t qk_acc0 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);
                    vfloat32m2_t qk_acc1 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);

                    ପାଇଁ (int64_t d = 0; d < DK; ++d) {
                        ଅଚଳ vfloat16m1_t k_vec = __riscv_vle16_v_f16m1(k_tile_pack + d * ic_tile_step, qk_vl);
                        qk_acc0 = __riscv_vfwmacc_vf_f32m2(qk_acc0, qv_buffer[0 * DK + d], k_vec, qk_vl);
                        qk_acc1 = __riscv_vfwmacc_vf_f32m2(qk_acc1, qv_buffer[1 * DK + d], k_vec, qk_vl);
                    }

                    qk_acc0 = __riscv_vfmul_vf_f32m2(qk_acc0, scale, qk_vl);
                    qk_acc1 = __riscv_vfmul_vf_f32m2(qk_acc1, scale, qk_vl);

                    __riscv_vse32_v_f32m2(kq_tile_buffer + 0 * ic_tile_step, qk_acc0, qk_vl);
                    __riscv_vse32_v_f32m2(kq_tile_buffer + 1 * ic_tile_step, qk_acc1, qk_vl);
                }

                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < QLEN; ++i) {
                    ଦଶମିକ *     row_ptr = kq_tile_buffer + i * ic_tile_step;
                    ଅଚଳ ଦଶମିକ tile_max =
                        rvv_softcap_add_max_inplace_f32(row_ptr, mv_tile_pack, ic_step, logit_softcap);

                    ଅଚଳ ଦଶମିକ Mold = M[i];

                    ଯଦି (tile_max > Mold) {
                        ଅଚଳ ଦଶମିକ ms = expf(Mold - tile_max);
                        M[i]           = tile_max;
                        S[i] *= ms;

                        vfloat16m2_t VKQ16_v = __riscv_vle16_v_f16m2(kq16_buffer + i * DV, DV);
                        VKQ16_v              = __riscv_vfmul_vf_f16m2(VKQ16_v, (_Float16) ms, DV);
                        __riscv_vse16_v_f16m2(kq16_buffer + i * DV, VKQ16_v, DV);
                    }

                    S[i] += rvv_softmax_exp_inplace_f32(row_ptr, ic_step, M[i]);
                }

                ଯଦି constexpr (QLEN == 4) {
                    vfloat16m2_t pv_acc0 = __riscv_vle16_v_f16m2(kq16_buffer + 0 * DV, DV);
                    vfloat16m2_t pv_acc1 = __riscv_vle16_v_f16m2(kq16_buffer + 1 * DV, DV);
                    vfloat16m2_t pv_acc2 = __riscv_vle16_v_f16m2(kq16_buffer + 2 * DV, DV);
                    vfloat16m2_t pv_acc3 = __riscv_vle16_v_f16m2(kq16_buffer + 3 * DV, DV);

                    ପାଇଁ (int64_t tk = 0; tk < ic_step; ++tk) {
                        ଅଚଳ vfloat16m2_t v16 = __riscv_vle16_v_f16m2(v_tile_pack + tk * DV, DV);
                        pv_acc0 =
                            __riscv_vfmacc_vf_f16m2(pv_acc0, (_Float16) kq_tile_buffer[0 * ic_tile_step + tk], v16, DV);
                        pv_acc1 =
                            __riscv_vfmacc_vf_f16m2(pv_acc1, (_Float16) kq_tile_buffer[1 * ic_tile_step + tk], v16, DV);
                        pv_acc2 =
                            __riscv_vfmacc_vf_f16m2(pv_acc2, (_Float16) kq_tile_buffer[2 * ic_tile_step + tk], v16, DV);
                        pv_acc3 =
                            __riscv_vfmacc_vf_f16m2(pv_acc3, (_Float16) kq_tile_buffer[3 * ic_tile_step + tk], v16, DV);
                    }

                    __riscv_vse16_v_f16m2(kq16_buffer + 0 * DV, pv_acc0, DV);
                    __riscv_vse16_v_f16m2(kq16_buffer + 1 * DV, pv_acc1, DV);
                    __riscv_vse16_v_f16m2(kq16_buffer + 2 * DV, pv_acc2, DV);
                    __riscv_vse16_v_f16m2(kq16_buffer + 3 * DV, pv_acc3, DV);
                } ଅନ୍ୟଥା {
                    static_assert(QLEN == 2, "unsupported QLEN");
                    vfloat16m2_t pv_acc0 = __riscv_vle16_v_f16m2(kq16_buffer + 0 * DV, DV);
                    vfloat16m2_t pv_acc1 = __riscv_vle16_v_f16m2(kq16_buffer + 1 * DV, DV);

                    ପାଇଁ (int64_t tk = 0; tk < ic_step; ++tk) {
                        ଅଚଳ vfloat16m2_t v16 = __riscv_vle16_v_f16m2(v_tile_pack + tk * DV, DV);
                        pv_acc0 =
                            __riscv_vfmacc_vf_f16m2(pv_acc0, (_Float16) kq_tile_buffer[0 * ic_tile_step + tk], v16, DV);
                        pv_acc1 =
                            __riscv_vfmacc_vf_f16m2(pv_acc1, (_Float16) kq_tile_buffer[1 * ic_tile_step + tk], v16, DV);
                    }

                    __riscv_vse16_v_f16m2(kq16_buffer + 0 * DV, pv_acc0, DV);
                    __riscv_vse16_v_f16m2(kq16_buffer + 1 * DV, pv_acc1, DV);
                }

                ic_step = 0;
            }
        }
    } ଅନ୍ୟଥା {
        ପାଇଁ (int64_t ic = 0; ic < nek1; ++ic) {
            ଅଚଳ ଦଶମିକ mv = mp ? slope * ((_Float16 *) mp)[ic] : 0.0f;

            ଅଚଳ ଅକ୍ଷର * k_data = k_data_row + ic * nbk1;
            ଅଚଳ ଅକ୍ଷର * v_data = v_data_row + ic * nbv1;

            vfloat16m2_t k_data_v;
            vfloat16m2_t v_data_v;

            ଯଦି (mv != -INFINITY) {
                k_data_v = __riscv_vle16_v_f16m2((_Float16 *) k_data, DK);
                v_data_v = __riscv_vle16_v_f16m2((_Float16 *) v_data, DV);
            } ଅନ୍ୟଥା {
                ଜାରି;
            }

            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < QLEN; ++i) {
                vfloat16m2_t Q_q_v    = __riscv_vle16_v_f16m2(qv_buffer + i * DK, DK);
                vfloat32m4_t qk_acc_v = __riscv_vfwmul_vv_f32m4(k_data_v, Q_q_v, DK);
                ଦଶମିକ        s        = reduce_sum_f32m4_vlen1024(qk_acc_v, DK);
                s                     = s * scale;
                ଯଦି (logit_softcap != 0.0f) {
                    s = logit_softcap * tanhf(s);
                }
                s += mv;

                ଅଚଳ ଦଶମିକ Mold = M[i];

                ଦଶମିକ ms = 1.0f;  // upon new higher max val, scale VKQ and KQ sum with this value
                ଦଶମିକ vs = 1.0f;  // post-softmax KQ value, expf(s - M)

                vfloat16m2_t VKQ16_v = __riscv_vle16_v_f16m2(kq16_buffer + i * DV, DV);
                ଯଦି (s > M[i]) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M[i] = s;
                    ms   = expf(Mold - M[i]);

                    // V = V*expf(Mold - M)
                    VKQ16_v = __riscv_vfmul_vf_f16m2(VKQ16_v, ms, DV);
                } ଅନ୍ୟଥା {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M[i]);
                }
                VKQ16_v = __riscv_vfmacc_vf_f16m2(VKQ16_v, vs, v_data_v, DV);
                __riscv_vse16_v_f16m2(kq16_buffer + i * DV, VKQ16_v, DV);
                S[i] = S[i] * ms + vs;  // scale and increment sum with partial sum
            }
        }
    }

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < QLEN; ++i) {
        vfloat16m2_t VKQ16_v = __riscv_vle16_v_f16m2(kq16_buffer + i * DV, DV);
        vfloat32m4_t VKQ32_v = __riscv_vfwcvt_f_f_v_f32m4(VKQ16_v, DV);

        // sinks
        ଯଦି (sinks[i]) {
            ଅଚଳ ଦଶମିକ s = *(sinks[i]);

            ଦଶମିକ ms = 1.0f;
            ଦଶମିକ vs = 1.0f;

            ଯଦି (s > M[i]) {
                ms      = expf(M[i] - s);
                M[i]    = s;
                VKQ32_v = __riscv_vfmul_vf_f32m4(VKQ32_v, ms, DV);
            } ଅନ୍ୟଥା {
                vs = expf(s - M[i]);
            }

            S[i] = S[i] * ms + vs;
        }

        // V /= S
        ଅଚଳ ଦଶମିକ S_inv = S[i] == 0.0f ? 0.0f : 1.0f / S[i];

        VKQ32_v = __riscv_vfmul_vf_f32m4(VKQ32_v, S_inv, DV);

        __riscv_vse32_v_f32m4(dst[i], VKQ32_v, DV);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_m1(ଅଚଳ ଦଶମିକ *       pq,
                                                                ଅଚଳ ଅକ୍ଷର *        k_data_row,
                                                                ଅଚଳ ଅକ୍ଷର *        v_data_row,
                                                                ଅଚଳ ggml_fp16_t * mp,
                                                                ଅଚଳ ଦଶମିକ *       sinks,
                                                                ଦଶମିକ *             dst,
                                                                ଦଶମିକ               scale,
                                                                ଦଶମିକ               logit_softcap,
                                                                ଦଶମିକ               slope,
                                                                int64_t             nek1,
                                                                int64_t             nbk1,
                                                                int64_t             nbv1,
                                                                int64_t             DV,
                                                                int64_t             DK) {
    GGML_ASSERT(flash_attn_ext_supported_shape_vlen1024_vf16(DK, DV));

    ଦଶମିକ S = 0.0f;       // sum
    ଦଶମିକ M = -INFINITY;  // maximum KQ value

    vfloat16m2_t VKQ16_v = __riscv_vfmv_v_f_f16m2(0.0f, DV);

    vfloat16m2_t Q_q_v = __riscv_vfncvt_f_f_w_f16m2(__riscv_vle32_v_f32m4(pq, DK), DK);

    ପାଇଁ (int64_t ic = 0; ic < nek1; ++ic) {
        ଅଚଳ ଦଶମିକ mv = mp ? slope * ((_Float16 *) mp)[ic] : 0.0f;
        ଯଦି (mv == -INFINITY) {
            ଜାରି;
        }

        ଅଚଳ ଅକ୍ଷର * k_data = k_data_row + ic * nbk1;

        vfloat16m2_t k_data_v = __riscv_vle16_v_f16m2((_Float16 *) k_data, DK);

        vfloat32m4_t qk_acc_v = __riscv_vfwmul_vv_f32m4(k_data_v, Q_q_v, DK);
        ଦଶମିକ        s        = reduce_sum_f32m4_vlen1024(qk_acc_v, DK);

        s = s * scale;  // scale KQ value

        ଯଦି (logit_softcap != 0.0f) {
            s = logit_softcap * tanhf(s);
        }

        s += mv;  // apply mask

        ଅଚଳ ଦଶମିକ Mold = M;

        ଦଶମିକ ms = 1.0f;  // upon new higher max val, scale VKQ and KQ sum with this value
        ଦଶମିକ vs = 1.0f;  // post-softmax KQ value, expf(s - M)

        ଅଚଳ ଅକ୍ଷର * v_data = v_data_row + ic * nbv1;

        vfloat16m2_t v_data_v = __riscv_vle16_v_f16m2((_Float16 *) v_data, DV);

        ଯଦି (s > M) {
            // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
            M  = s;
            ms = expf(Mold - M);

            // V = V*expf(Mold - M)
            VKQ16_v = __riscv_vfmul_vf_f16m2(VKQ16_v, ms, DV);
        } ଅନ୍ୟଥା {
            // no new maximum, ms == 1.0f, vs != 1.0f
            vs = expf(s - M);
        }

        VKQ16_v = __riscv_vfmacc_vf_f16m2(VKQ16_v, vs, v_data_v, DV);

        S = S * ms + vs;  // scale and increment sum with partial sum
    }

    vfloat32m4_t VKQ32_v = __riscv_vfwcvt_f_f_v_f32m4(VKQ16_v, DV);

    // sinks
    ଯଦି (sinks) {
        ଅଚଳ ଦଶମିକ s = *sinks;

        ଦଶମିକ ms = 1.0f;
        ଦଶମିକ vs = 1.0f;

        ଯଦି (s > M) {
            ms      = expf(M - s);
            M       = s;
            VKQ32_v = __riscv_vfmul_vf_f32m4(VKQ32_v, ms, DV);
        } ଅନ୍ୟଥା {
            vs = expf(s - M);
        }

        S = S * ms + vs;
    }

    // V /= S
    ଅଚଳ ଦଶମିକ S_inv = S == 0.0f ? 0.0f : 1.0f / S;

    VKQ32_v = __riscv_vfmul_vf_f32m4(VKQ32_v, S_inv, DV);

    __riscv_vse32_v_f32m4(dst, VKQ32_v, DV);
}

}  // namespace

ଶୂନ୍ୟ memcpy1d(ଶୂନ୍ୟ * dst, ଅଚଳ ଶୂନ୍ୟ * src, int64_t size) {
    size_t byte_size_all = size;
    size_t vlen          = __riscv_vlenb() * 8;
    ଯଦି (vlen == 256) {
        // 1024 bytes
        __asm__ ଅସ୍ଥିର(
            //
            "srli           t0, %[size], 10             \n\t"
            "blez           t0, memcpy_tail%=           \n\t"
            "vsetvli        t1, x0, e8, m8, tu, mu      \n\t"
            "memcpy_main_loop%=:                        \n\t"
            "addi           t0, t0, -1                  \n\t"
            "vle8.v         v0, (%[s])                  \n\t"
            "addi           %[s], %[s], 256             \n\t"
            "vle8.v         v8, (%[s])                  \n\t"
            "addi           %[s], %[s], 256             \n\t"
            "vle8.v         v16, (%[s])                 \n\t"
            "addi           %[s], %[s], 256             \n\t"
            "vle8.v         v24, (%[s])                 \n\t"
            "addi           %[s], %[s], 256             \n\t"
            //
            "vse8.v         v0, (%[d])                  \n\t"
            "addi           %[d], %[d], 256             \n\t"
            "vse8.v         v8, (%[d])                  \n\t"
            "addi           %[d], %[d], 256             \n\t"
            "vse8.v         v16, (%[d])                 \n\t"
            "addi           %[d], %[d], 256             \n\t"
            "vse8.v         v24, (%[d])                 \n\t"
            "addi           %[d], %[d], 256             \n\t"
            //
            "bnez           t0, memcpy_main_loop%=      \n\t"
            "memcpy_tail%=:                             \n\t"
            "andi           t1, %[size], 1023           \n\t"
            "blez           t1, out%=                   \n\t"
            "memcpy_tail_loop%=:                        \n\t"
            "vsetvli        t0, t1, e8, m8, tu, mu      \n\t"
            "sub            t1, t1, t0                  \n\t"
            "vle8.v         v0, (%[s])                  \n\t"
            "add            %[s], %[s], t0              \n\t"
            "vse8.v         v0, (%[d])                  \n\t"
            "add            %[d], %[d], t0              \n\t"
            "bnez           t1, memcpy_tail_loop%=      \n\t"
            "out%=:                                     \n\t"
            : [s] "+r"(src), [d] "+r"(dst)
            : [size] "r"(byte_size_all)
            : "cc", "t0", "t1");
    } ଅନ୍ୟଥା ଯଦି (vlen == 1024) {
        // 2048 bytes
        __asm__ ଅସ୍ଥିର(
            //
            "srli           t0, %[size], 11             \n\t"
            "blez           t0, memcpy_tail%=           \n\t"
            "vsetvli        t1, x0, e8, m8, tu, mu      \n\t"
            "addi           t2, %[s], 1024              \n\t"
            "addi           t3, %[d], 1024              \n\t"
            "li             t5, 2048                    \n\t"
            "memcpy_main_loop%=:                        \n\t"
            "addi           t0, t0, -1                  \n\t"
            "vle8.v         v0, (%[s])                  \n\t"
            "add            %[s], %[s], t5              \n\t"
            "vle8.v         v8, (t2)                    \n\t"
            "add            t2, t2, t5                  \n\t"
            //
            "vse8.v         v0, (%[d])                  \n\t"
            "add            %[d], %[d], t5              \n\t"
            "vse8.v         v8, (t3)                    \n\t"
            "add            t3, t3, t5                  \n\t"
            //
            "bnez           t0, memcpy_main_loop%=      \n\t"
            "memcpy_tail%=:                             \n\t"
            "andi           t1, %[size], 2047           \n\t"
            "blez           t1, out%=                   \n\t"
            "memcpy_tail_loop%=:                        \n\t"
            "vsetvli        t0, t1, e8, m2, tu, mu      \n\t"
            "sub            t1, t1, t0                  \n\t"
            "vle8.v         v0, (%[s])                  \n\t"
            "add            %[s], %[s], t0              \n\t"
            "vse8.v         v0, (%[d])                  \n\t"
            "add            %[d], %[d], t0              \n\t"
            "bnez           t1, memcpy_tail_loop%=      \n\t"
            "out%=:                                     \n\t"
            : [s] "+r"(src), [d] "+r"(dst)
            : [size] "r"(byte_size_all)
            : "cc", "t0", "t1", "t2", "t3", "t5");
    } ଅନ୍ୟଥା {
        __asm__ ଅସ୍ଥିର(
            //
            "add            t1, %[size], zero           \n\t"
            "memcpy_tail_loop%=:                        \n\t"
            "vsetvli        t0, t1, e8, m8, tu, mu      \n\t"
            "sub            t1, t1, t0                  \n\t"
            "vle8.v         v0, (%[s])                  \n\t"
            "add            %[s], %[s], t0              \n\t"
            "vse8.v         v0, (%[d])                  \n\t"
            "add            %[d], %[d], t0              \n\t"
            "bnez           t1, memcpy_tail_loop%=      \n\t"
            : [s] "+r"(src), [d] "+r"(dst)
            : [size] "r"(byte_size_all)
            : "cc", "t0", "t1", "t2", "t4", "t3");
    }
}

ଶୂନ୍ୟ memcpy2d(ଶୂନ୍ୟ * dst, int64_t dst_stride, ଅଚଳ ଶୂନ୍ୟ * src, int64_t src_stride, int64_t tile_rows, int64_t size) {
    ପାଇଁ (int64_t i = 0; i < tile_rows; ++i) {
        memcpy1d((ଅକ୍ଷର *) dst + i * dst_stride, (ଅଚଳ ଅକ୍ଷର *) src + i * src_stride, size);
    }
}

ଶୂନ୍ୟ forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(ଅଚଳ ggml_compute_params * params,
                                                        ggml_tensor *               dst,
                                                        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                         ir0,
                                                        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                         ir1,
                                                        ଶୂନ୍ୟ *                      tcm_buffer,
                                                        size_t                      tcm_buffer_size) {
    ଅଚଳ ggml_tensor * q     = dst->src[0];
    ଅଚଳ ggml_tensor * k     = dst->src[1];
    ଅଚଳ ggml_tensor * v     = dst->src[2];
    ଅଚଳ ggml_tensor * mask  = dst->src[3];
    ଅଚଳ ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q, ne)
    GGML_TENSOR_LOCALS(size_t, nbq, q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k, ne)
    GGML_TENSOR_LOCALS(size_t, nbk, k, nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v, ne)
    GGML_TENSOR_LOCALS(size_t, nbv, v, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t, nb, dst, nb)

    ଅଚଳ int64_t DK = nek0;
    ଅଚଳ int64_t DV = nev0;
    ଅଚଳ int64_t N  = neq1;

    GGML_ASSERT(flash_attn_ext_supported_shape_vlen1024_vf16(DK, DV));

    // broadcast factors
    ଅଚଳ int64_t rk2 = neq2 / nek2;
    ଅଚଳ int64_t rk3 = neq3 / nek3;

    ଅଚଳ int64_t rv2 = neq2 / nev2;
    ଅଚଳ int64_t rv3 = neq3 / nev3;

    // parallelize by q rows using ggml_vec_dot_f32

    ଦଶମିକ scale         = *((ଦଶମିକ *) dst->op_params + 0);
    ଦଶମିକ max_bias      = *((ଦଶମିକ *) dst->op_params + 1);
    ଦଶମିକ logit_softcap = *((ଦଶମିକ *) dst->op_params + 2);

    ଯଦି (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    ଅଚଳ uint32_t n_head      = neq2;
    ଅଚଳ uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    ଅଚଳ ଦଶମିକ m0 = powf(2.0f, -(max_bias) / n_head_log2);
    ଅଚଳ ଦଶମିକ m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା KV_row_size = DK * ଆକାର(_Float16) + DV * ଆକାର(_Float16);

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith     = params->ith;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir_step = 1;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir = ir0; ir < ir1; ir += ir_step) {
        // q indices
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iq3 = ir / (neq2 * neq1);
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iq2 = (ir - iq3 * neq2 * neq1) / neq1;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iq1 = (ir - iq3 * neq2 * neq1 - iq2 * neq1);

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iq3_1 = (ir + 1) / (neq2 * neq1);
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iq2_1 = (ir + 1 - iq3_1 * neq2 * neq1) / neq1;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iq1_1 = (ir + 1 - iq3_1 * neq2 * neq1 - iq2_1 * neq1);

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iq3_2 = (ir + 2) / (neq2 * neq1);
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iq2_2 = (ir + 2 - iq3_2 * neq2 * neq1) / neq1;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iq1_2 = (ir + 2 - iq3_2 * neq2 * neq1 - iq2_2 * neq1);

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iq3_3 = (ir + 3) / (neq2 * neq1);
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iq2_3 = (ir + 3 - iq3_3 * neq2 * neq1) / neq1;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iq1_3 = (ir + 3 - iq3_3 * neq2 * neq1 - iq2_3 * neq1);

        ଅଚଳ uint32_t h = iq2;  // head index
        ଅଚଳ ଦଶମିକ    slope =
            (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2 * (h - n_head_log2) + 1) : 1.0f;

        ଅଚଳ ggml_fp16_t * mp =
            mask ? (ggml_fp16_t *) ((ଅକ୍ଷର *) mask->data + iq1 * mask->nb[1] + (iq2 % mask->ne[2]) * mask->nb[2] +
                                    (iq3 % mask->ne[3]) * mask->nb[3]) :
                   ଖାଲି;

        ଅଚଳ ତାର୍କିକ mp_equal_2 = iq1_1 == iq1 && (iq2 % mask->ne[2]) == (iq2_1 % mask->ne[2]) &&
                                (iq3 % mask->ne[3]) == (iq3_1 % mask->ne[3]);

        ଅଚଳ ତାର୍କିକ mp_equal_4 = mp_equal_2 && iq1_2 == iq1 && (iq2 % mask->ne[2]) == (iq2_2 % mask->ne[2]) &&
                                (iq3 % mask->ne[3]) == (iq3_2 % mask->ne[3]) && iq1_3 == iq1 &&
                                (iq2 % mask->ne[2]) == (iq2_3 % mask->ne[2]) &&
                                (iq3 % mask->ne[3]) == (iq3_3 % mask->ne[3]);

        // k indices
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ik3 = iq3 / rk3;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ik2 = iq2 / rk2;

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ik3_1 = iq3_1 / rk3;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ik2_1 = iq2_1 / rk2;

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ik3_2 = iq3_2 / rk3;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ik2_2 = iq2_2 / rk2;

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ik3_3 = iq3_3 / rk3;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ik2_3 = iq2_3 / rk2;

        // v indices
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iv3 = iq3 / rv3;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iv2 = iq2 / rv2;

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iv3_1 = iq3_1 / rv3;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iv2_1 = iq2_1 / rv2;

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iv3_2 = iq3_2 / rv3;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iv2_2 = iq2_2 / rv2;

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iv3_3 = iq3_3 / rv3;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iv2_3 = iq2_3 / rv2;

        ଅଚଳ ଦଶମିକ * pq = (ଅଚଳ ଦଶମିକ *) ((ଅକ୍ଷର *) q->data + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3));

        std::array<ଦଶମିକ *, 4> pq_buffer;
        std::array<ଦଶମିକ *, 4> sinks_buffer;
        std::array<ଦଶମିକ *, 4> dst_buffer;

        ଯଦି (tcm_buffer != nullptr && 4 * KV_row_size < tcm_buffer_size && ir < (ir1 - 3) && mp_equal_4 &&
            ik3_3 == ik3 && ik2_3 == ik2 && iv3_3 == iv3 && iv2_3 == iv2 && ik3_2 == ik3 && ik2_2 == ik2 &&
            iv3_2 == iv3 && iv2_2 == iv2 && ik3_1 == ik3 && ik2_1 == ik2 && iv3_1 == iv3 && iv2_1 == iv2) {
            ir_step = 4;

            pq_buffer[0] = (ଦଶମିକ *) ((ଅକ୍ଷର *) q->data + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3));
            pq_buffer[1] = (ଦଶମିକ *) ((ଅକ୍ଷର *) q->data + (iq1_1 * nbq1 + iq2_1 * nbq2 + iq3_1 * nbq3));
            pq_buffer[2] = (ଦଶମିକ *) ((ଅକ୍ଷର *) q->data + (iq1_2 * nbq1 + iq2_2 * nbq2 + iq3_2 * nbq3));
            pq_buffer[3] = (ଦଶମିକ *) ((ଅକ୍ଷର *) q->data + (iq1_3 * nbq1 + iq2_3 * nbq2 + iq3_3 * nbq3));

            sinks_buffer[0] = sinks ? ((ଦଶମିକ *) ((ଅକ୍ଷର *) sinks->data)) + iq2 : nullptr;
            sinks_buffer[1] = sinks ? ((ଦଶମିକ *) ((ଅକ୍ଷର *) sinks->data)) + iq2_1 : nullptr;
            sinks_buffer[2] = sinks ? ((ଦଶମିକ *) ((ଅକ୍ଷର *) sinks->data)) + iq2_2 : nullptr;
            sinks_buffer[3] = sinks ? ((ଦଶମିକ *) ((ଅକ୍ଷର *) sinks->data)) + iq2_3 : nullptr;

            dst_buffer[0] = (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + (iq3 * ne2 * ne1 + iq2 + iq1 * ne1) * nb1);
            dst_buffer[1] = (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + (iq3_1 * ne2 * ne1 + iq2_1 + iq1_1 * ne1) * nb1);
            dst_buffer[2] = (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + (iq3_2 * ne2 * ne1 + iq2_2 + iq1_2 * ne1) * nb1);
            dst_buffer[3] = (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + (iq3_3 * ne2 * ne1 + iq2_3 + iq1_3 * ne1) * nb1);

            flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_mrow<4>(  //
                pq_buffer.data(),                                      //
                (ଅଚଳ ଅକ୍ଷର *) k->data + (ik2 * nbk2 + ik3 * nbk3),    //
                (ଅଚଳ ଅକ୍ଷର *) v->data + (iv2 * nbv2 + iv3 * nbv3),    //
                mp,                                                    //
                sinks_buffer.data(),                                   //
                dst_buffer.data(),                                     //
                scale, logit_softcap, slope, nek1, nbk1, nbv1, DV, DK, tcm_buffer, tcm_buffer_size);
        } ଅନ୍ୟଥା ଯଦି (tcm_buffer != nullptr && 2 * KV_row_size < tcm_buffer_size && ir < (ir1 - 1) && mp_equal_2 &&
                   ik3_1 == ik3 && ik2_1 == ik2 && iv3_1 == iv3 && iv2_1 == iv2) {
            ir_step = 2;

            pq_buffer[0] = (ଦଶମିକ *) ((ଅକ୍ଷର *) q->data + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3));
            pq_buffer[1] = (ଦଶମିକ *) ((ଅକ୍ଷର *) q->data + (iq1_1 * nbq1 + iq2_1 * nbq2 + iq3_1 * nbq3));

            sinks_buffer[0] = sinks ? ((ଦଶମିକ *) ((ଅକ୍ଷର *) sinks->data)) + iq2 : nullptr;
            sinks_buffer[1] = sinks ? ((ଦଶମିକ *) ((ଅକ୍ଷର *) sinks->data)) + iq2_1 : nullptr;

            dst_buffer[0] = (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + (iq3 * ne2 * ne1 + iq2 + iq1 * ne1) * nb1);
            dst_buffer[1] = (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + (iq3_1 * ne2 * ne1 + iq2_1 + iq1_1 * ne1) * nb1);

            flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_mrow<2>(  //
                pq_buffer.data(),                                      //
                (ଅଚଳ ଅକ୍ଷର *) k->data + (ik2 * nbk2 + ik3 * nbk3),    //
                (ଅଚଳ ଅକ୍ଷର *) v->data + (iv2 * nbv2 + iv3 * nbv3),    //
                mp,                                                    //
                sinks_buffer.data(),                                   //
                dst_buffer.data(),                                     //
                scale, logit_softcap, slope, nek1, nbk1, nbv1, DV, DK, tcm_buffer, tcm_buffer_size);
        } ଅନ୍ୟଥା {
            ir_step = 1;
            flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_m1(                             //
                pq,                                                                          //
                (ଅଚଳ ଅକ୍ଷର *) k->data + (ik2 * nbk2 + ik3 * nbk3),                          //
                (ଅଚଳ ଅକ୍ଷର *) v->data + (iv2 * nbv2 + iv3 * nbv3),                          //
                mp,                                                                          //
                sinks ? ((ଦଶମିକ *) ((ଅକ୍ଷର *) sinks->data)) + h : nullptr,                    //
                (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + (iq3 * ne2 * ne1 + iq2 + iq1 * ne1) * nb1),  //
                scale, logit_softcap, slope, nek1, nbk1, nbv1, DV, DK);
        }
    }
}

ଶୂନ୍ୟ forward_flash_attn_ext_f16_tiled_vlen1024_vf16(ଅଚଳ ggml_compute_params * params,
                                                    ggml_tensor *               dst,
                                                    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                         ir0,
                                                    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                         ir1,
                                                    ଶୂନ୍ୟ *                      tcm_buffer,
                                                    size_t                      tcm_buffer_size) {
    ଅଚଳ ggml_tensor * q     = dst->src[0];
    ଅଚଳ ggml_tensor * k     = dst->src[1];
    ଅଚଳ ggml_tensor * v     = dst->src[2];
    ଅଚଳ ggml_tensor * mask  = dst->src[3];
    ଅଚଳ ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q, ne)
    GGML_TENSOR_LOCALS(size_t, nbq, q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k, ne)
    GGML_TENSOR_LOCALS(size_t, nbk, k, nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v, ne)
    GGML_TENSOR_LOCALS(size_t, nbv, v, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t, nb, dst, nb)

    ଅଚଳ int64_t DK = nek0;
    ଅଚଳ int64_t DV = nev0;
    ଅଚଳ int64_t N  = neq1;

    GGML_ASSERT(flash_attn_ext_supported_shape_vlen1024_vf16(DK, DV));

    GGML_ASSERT(ne0 == DV);
    GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_size(q->type));
    GGML_ASSERT(nbk0 == ggml_type_size(k->type));
    GGML_ASSERT(nbv0 == ggml_type_size(v->type));

    GGML_ASSERT(neq0 == DK);
    GGML_ASSERT(nek0 == DK);
    GGML_ASSERT(nev0 == DV);

    GGML_ASSERT(neq1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == ଆକାର(ଦଶମିକ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(k->type == v->type);
    ଅଚଳ ggml_type kv_type = k->type;

    // broadcast factors
    ଅଚଳ int64_t rk2 = neq2 / nek2;
    ଅଚଳ int64_t rk3 = neq3 / nek3;

    ଅଚଳ int64_t rv2 = neq2 / nev2;
    ଅଚଳ int64_t rv3 = neq3 / nev3;

    ଦଶମିକ * param_list    = (ଦଶମିକ *) dst->op_params;
    ଦଶମିକ   scale         = param_list[0];
    ଦଶମିକ   max_bias      = param_list[1];
    ଦଶମିକ   logit_softcap = param_list[2];

    ଯଦି (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    ଅଚଳ uint32_t n_head      = neq2;
    ଅଚଳ uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    ଅଚଳ ଦଶମିକ m0 = powf(2.0f, -(max_bias) / n_head_log2);
    ଅଚଳ ଦଶମିକ m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;

    ସ୍ଥିର constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା Q_TILE_SZ  = ggml_fa_tile_config::Q;
    ସ୍ଥିର constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା KV_TILE_SZ = ggml_fa_tile_config::KV;

    // Per-thread scratch layout:
    // Q_f32:   Q_TILE_SZ * DK
    // KQ:      Q_TILE_SZ * KV_TILE_SZ
    // mask32:  Q_TILE_SZ * KV_TILE_SZ
    // VKQ32:   Q_TILE_SZ * DV
    // V32:     KV_TILE_SZ * DV
    // K_f32:   DK * KV_TILE_SZ (transposed K tile)
    ଦଶମିକ *      base = (ଦଶମିକ *) params->wdata + ith * (Q_TILE_SZ * DK + 2 * Q_TILE_SZ * KV_TILE_SZ + Q_TILE_SZ * DV +
                                                    KV_TILE_SZ * DV + KV_TILE_SZ * DK + CACHE_ଧାଡ଼ି_SIZE_F32);
    ଅଚଳ size_t base_size =
        (Q_TILE_SZ * DK + 2 * Q_TILE_SZ * KV_TILE_SZ + Q_TILE_SZ * DV + KV_TILE_SZ * DV + KV_TILE_SZ * DK) *
            ଆକାର(ଦଶମିକ) +
        CACHE_ଧାଡ଼ି_SIZE_F32;

    ଯଦି (base_size <= tcm_buffer_size && tcm_buffer != nullptr) {
        base = (ଦଶମିକ *) tcm_buffer;
    }

    ଦଶମିକ   S_M_Buf[Q_TILE_SZ * 2];  // buffer to hold S, M, bias for one tile to reduce register pressure in main loop
    ଦଶମିକ * S = S_M_Buf;
    ଦଶମିକ * M = S_M_Buf + Q_TILE_SZ;

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir = ir0;
    ଯେତେବେଳେ (ir < ir1) {
        // q indices for the start of this tile
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iq3 = ir / (neq2 * neq1);
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iq2 = (ir - iq3 * neq2 * neq1) / neq1;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iq1 = (ir - iq3 * neq2 * neq1 - iq2 * neq1);

        // Number of valid rows in this tile:
        // - limited by tile size (Q_TILE_SZ)
        // - limited by chunk boundary (ir1 - ir)
        // - limited by head boundary (neq1 - iq1) to avoid crossing into next head
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା tile_rows = MIN(Q_TILE_SZ, MIN((ପୂର୍ଣ୍ଣସଂଖ୍ୟା) (ir1 - ir), (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) (neq1 - iq1)));
        GGML_ASSERT(tile_rows > 0);

        ଅଚଳ uint32_t h = iq2;  // head index
        ଅଚଳ ଦଶମିକ    slope =
            (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2 * (h - n_head_log2) + 1) : 1.0f;

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < Q_TILE_SZ; ++i) {
            S[i] = 0.;
            M[i] = -INFINITY;
        }

        ଦଶମିକ *    Q_f32  = base;
        ଦଶମିକ *    KQ     = (ଦଶମିକ *) ((ଅକ୍ଷର *) base + Q_TILE_SZ * DK * ଆକାର(ଦଶମିକ));
        ଦଶମିକ *    mask32 = KQ + Q_TILE_SZ * KV_TILE_SZ;
        ଦଶମିକ *    VKQ32  = mask32 + Q_TILE_SZ * KV_TILE_SZ;
        ଦଶମିକ *    V32    = VKQ32 + Q_TILE_SZ * DV;
        ଦଶମିକ *    K_f32  = V32 + KV_TILE_SZ * DV;
        _Float16 * Q_f16  = (_Float16 *) Q_f32;
        _Float16 * V_f16  = (_Float16 *) V32;
        _Float16 * K_f16  = (_Float16 *) K_f32;

        rvv_zero_f32(VKQ32, Q_TILE_SZ * DV);

        // k indices
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ik3 = iq3 / rk3;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ik2 = iq2 / rk2;

        // v indices
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iv3 = iq3 / rv3;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iv2 = iq2 / rv2;

        ଅଚଳ ଦଶମିକ * pq = (ଅଚଳ ଦଶମିକ *) ((ଅକ୍ଷର *) q->data + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3));
        ଯଦି (kv_type == GGML_TYPE_F16) {
            rvv_pack_f32_as_scaled_f16((uint8_t *) Q_f16, DK * ଆକାର(_Float16), (uint8_t *) pq, nbq1, tile_rows, DK,
                                       scale);
        } ଅନ୍ୟଥା {
            memcpy2d(Q_f32, DK * ଆକାର(ଦଶମିକ), pq, nbq1, tile_rows, DK * ଆକାର(ଦଶମିକ));
        }

        ପାଇଁ (int64_t ic = 0; ic < nek1; ic += KV_TILE_SZ) {
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା kv_tile = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) std::min((int64_t) KV_TILE_SZ, nek1 - ic);

            rvv_zero_f32(K_f32, DK * KV_TILE_SZ);
            rvv_zero_f32(V32, KV_TILE_SZ * DV);

            // skip the tile entirely if all the masks are -inf
            ଯଦି (mask) {
                ତାର୍କିକ                can_skip = ସତ୍ୟ;
                ଅଚଳ ggml_fp16_t * mp_row =
                    (ଅଚଳ ggml_fp16_t *) ((ଅଚଳ ଅକ୍ଷର *) mask->data + iq1 * mask->nb[1] +
                                           (iq2 % mask->ne[2]) * mask->nb[2] + (iq3 % mask->ne[3]) * mask->nb[3]);
                rvv_pack_scaled_f16_as_f32(mask32, KV_TILE_SZ * ଆକାର(ଦଶମିକ), mp_row + ic, mask->nb[1], tile_rows,
                                           kv_tile, slope);

                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା tq = 0; tq < tile_rows; tq++) {
                    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା tk = 0; tk < kv_tile; tk++) {
                        ଯଦି (mask32[tq * KV_TILE_SZ + tk] != -INFINITY) {
                            can_skip = ମିଥ୍ୟା;
                        }
                    }
                    // Pad remaining mask entries with -inf
                    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା tk = kv_tile; tk < KV_TILE_SZ; tk++) {
                        mask32[tq * KV_TILE_SZ + tk] = -INFINITY;
                    }
                }

                ଯଦି (can_skip) {
                    ଜାରି;
                }
            }

            ଯଦି (kv_type == GGML_TYPE_F16) {
                rvv_transposed_s16_mn_to_nm((int8_t *) K_f16, KV_TILE_SZ * ଆକାର(_Float16),
                                            (int8_t *) k->data + ic * nbk1 + ik2 * nbk2 + ik3 * nbk3, nbk1, kv_tile,
                                            DK);

                ପୂର୍ଣ୍ଣସଂଖ୍ୟା tq = 0;
                ପାଇଁ (; tq + 3 < tile_rows; tq += 4) {
                    rvv_qk_dot_tile_f16_x4(KQ + (tq + 0) * KV_TILE_SZ, KQ + (tq + 1) * KV_TILE_SZ,
                                           KQ + (tq + 2) * KV_TILE_SZ, KQ + (tq + 3) * KV_TILE_SZ,
                                           Q_f16 + (tq + 0) * DK, Q_f16 + (tq + 1) * DK, Q_f16 + (tq + 2) * DK,
                                           Q_f16 + (tq + 3) * DK, K_f16, DK, kv_tile);
                }
                ପାଇଁ (; tq < tile_rows; ++tq) {
                    rvv_qk_dot_tile_f16_x1(KQ + tq * KV_TILE_SZ, Q_f16 + tq * DK, K_f16, DK, kv_tile);
                }
            } ଅନ୍ୟଥା {
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା tk = 0; tk < kv_tile; tk++) {
                    ଅଚଳ ଅକ୍ଷର *  k_data = (ଅଚଳ ଅକ୍ଷର *) k->data + (ic + tk) * nbk1 + ik2 * nbk2 + ik3 * nbk3;
                    ଦଶମିକ *       k_col  = K_f32 + tk;
                    ଅଚଳ ଦଶମିକ * k_src  = (ଅଚଳ ଦଶମିକ *) k_data;
                    ପାଇଁ (int64_t dk = 0; dk < DK; ++dk) {
                        k_col[dk * KV_TILE_SZ] = k_src[dk];
                    }
                }

                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା tq = 0; tq < tile_rows; ++tq) {
                    rvv_qk_dot_tile(KQ + tq * KV_TILE_SZ, Q_f32 + tq * DK, K_f32, DK, KV_TILE_SZ, scale);
                }
            }

            // Set padded KQ entries to -inf so softmax gives them zero weight
            ଯଦି (kv_tile < KV_TILE_SZ) {
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା tq = 0; tq < tile_rows; tq++) {
                    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା tk = kv_tile; tk < KV_TILE_SZ; tk++) {
                        KQ[tq * KV_TILE_SZ + tk] = -INFINITY;
                    }
                }
            }

            ଯଦି (logit_softcap != 0.0f) {
                rvv_softcap_tanh_inplace_f32(KQ, KV_TILE_SZ, tile_rows, KV_TILE_SZ, logit_softcap);
            }

            ଯଦି (mask) {
                rvv_add_inplace_f32(KQ, KV_TILE_SZ, mask32, KV_TILE_SZ, tile_rows, KV_TILE_SZ);
            }

            ତାର୍କିକ skip[Q_TILE_SZ] = {};

            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା tq = 0; tq < tile_rows; tq++) {
                ଦଶମିକ * kq_row = KQ + tq * KV_TILE_SZ;

                ଅଚଳ ଦଶମିକ tile_max = rvv_max_f32(kq_row, KV_TILE_SZ);

                ଯଦି (tile_max == -INFINITY) {
                    skip[tq] = ସତ୍ୟ;
                    ଜାରି;
                }

                ଅଚଳ ଦଶମିକ Mold = M[tq];
                ଅଚଳ ଦଶମିକ Mnew = fmaxf(Mold, tile_max);

                ଯଦି (Mnew > Mold) {
                    ଅଚଳ ଦଶମିକ ms = expf(Mold - Mnew);
                    rvv_scale_f32(VKQ32 + tq * DV, ms, DV);
                    S[tq] *= ms;
                }
                M[tq] = Mnew;

                S[tq] += rvv_softmax_exp_inplace_f32(kq_row, KV_TILE_SZ, Mnew);
            }

            // Pack V as contiguous [KV_TILE_SZ][DV].
            ଯଦି (kv_type == GGML_TYPE_F16) {
                ଅଚଳ ଅକ୍ଷର * v_data = (ଅଚଳ ଅକ୍ଷର *) v->data + ic * nbv1 + iv2 * nbv2 + iv3 * nbv3;
                memcpy2d(V_f16, DV * ଆକାର(_Float16), v_data, nbv1, kv_tile, DV * ଆକାର(_Float16));

                ପୂର୍ଣ୍ଣସଂଖ୍ୟା tq = 0;
                ପାଇଁ (; tq + 3 < tile_rows; tq += 4) {
                    ଯଦି (skip[tq + 0] || skip[tq + 1] || skip[tq + 2] || skip[tq + 3]) {
                        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < 4; ++i) {
                            ଯଦି (!skip[tq + i]) {
                                rvv_pv_accumulate_f16_x1(VKQ32 + (tq + i) * DV, KQ + (tq + i) * KV_TILE_SZ, V_f16,
                                                         KV_TILE_SZ, DV);
                            }
                        }
                        ଜାରି;
                    }

                    rvv_pv_accumulate_f16_x4(VKQ32 + (tq + 0) * DV, VKQ32 + (tq + 1) * DV, VKQ32 + (tq + 2) * DV,
                                             VKQ32 + (tq + 3) * DV, KQ + (tq + 0) * KV_TILE_SZ,
                                             KQ + (tq + 1) * KV_TILE_SZ, KQ + (tq + 2) * KV_TILE_SZ,
                                             KQ + (tq + 3) * KV_TILE_SZ, V_f16, KV_TILE_SZ, DV);
                }
                ପାଇଁ (; tq < tile_rows; ++tq) {
                    ଯଦି (!skip[tq]) {
                        rvv_pv_accumulate_f16_x1(VKQ32 + tq * DV, KQ + tq * KV_TILE_SZ, V_f16, KV_TILE_SZ, DV);
                    }
                }
            } ଅନ୍ୟଥା {
                ଅଚଳ ଅକ୍ଷର * v_data = (ଅଚଳ ଅକ୍ଷର *) v->data + ic * nbv1 + iv2 * nbv2 + iv3 * nbv3;
                memcpy2d(V32, DV * ଆକାର(ଦଶମିକ), v_data, nbv1, kv_tile, DV * ଆକାର(ଦଶମିକ));

                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା tq = 0; tq < tile_rows; ++tq) {
                    ଯଦି (!skip[tq]) {
                        rvv_pv_accumulate(VKQ32 + tq * DV, KQ + tq * KV_TILE_SZ, V32, KV_TILE_SZ, DV);
                    }
                }
            }
        }

        // sinks (apply only to valid rows in the tile)
        ଯଦି (sinks) {
            ଅଚଳ ଦଶମିକ s = ((ଦଶମିକ *) ((ଅକ୍ଷର *) sinks->data))[h];

            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା tq = 0; tq < tile_rows; tq++) {
                ଦଶମିକ ms = 1.0f;
                ଦଶମିକ vs = 1.0f;

                ଯଦି (s > M[tq]) {
                    ms = expf(M[tq] - s);
                    rvv_scale_f32(VKQ32 + tq * DV, ms, DV);
                } ଅନ୍ୟଥା {
                    vs = expf(s - M[tq]);
                }

                ଦଶମିକ S_temp = S[tq] * ms + vs;
                S[tq]        = S_temp == 0.0f ? 0.0f : 1.0f / S_temp;
            }
        } ଅନ୍ୟଥା {
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା tq = 0; tq < tile_rows; tq++) {
                ଅଚଳ ଦଶମିକ S_inv = S[tq] == 0.0f ? 0.0f : 1.0f / S[tq];
                S[tq]             = S_inv;
            }
        }

        ଦଶମିକ * dst_ptr = (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + (iq3 * ne2 * ne1 + iq2 + (iq1) *ne1) * nb1);
        rvv_pack_scaled_f32_as_f32(dst_ptr, nb1 * ne1, VKQ32, DV * ଆକାର(ଦଶମିକ), tile_rows, DV, S);

        ir += tile_rows;
    }
}

ଶୂନ୍ୟ forward_rms_norm_f32(ggml_compute_params * params, ggml_tensor * op) {
    ଅଚଳ ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(src0->nb[0] == ଆକାର(ଦଶମିକ));

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    ଦଶମିକ epsilon = *((ଦଶମିକ *) dst->op_params);

    GGML_ASSERT(epsilon > 0.0f);

    auto * ନିବେଶ  = (ଅକ୍ଷର *) src0->data;
    auto * output = (ଅକ୍ଷର *) dst->data;

    ଅଚଳ auto hidden_size     = ne00;
    ଅଚଳ auto task_ଗଣନା      = ne01 * ne02 * ne03;
    ଅଚଳ auto task_per_thread = (task_ଗଣନା + nth - 1) / nth;

    ଅଚଳ auto task_begin = ith * task_per_thread;
    ଅଚଳ auto task_end   = std::min((ith + 1) * task_per_thread, task_ଗଣନା);

    ପାଇଁ (auto task_idx = task_begin; task_idx < task_end; task_idx++) {
        int64_t i03 = task_idx / (ne02 * ne01);
        int64_t i02 = (task_idx - i03 * ne02 * ne01) / ne01;
        int64_t i01 = (task_idx - i03 * ne02 * ne01 - i02 * ne01);

        auto * p_ନିବେଶ       = (ଦଶମିକ *) (ନିବେଶ + i01 * nb01 + i02 * nb02 + i03 * nb03);
        auto * p_output      = (ଦଶମିକ *) (output + i01 * nb1 + i02 * nb2 + i03 * nb3);
        auto * p_temp_output = p_output;

        size_t       gvl    = __riscv_vsetvlmax_e32m4();
        vfloat32m4_t sum_sq = __riscv_vfmv_v_f_f32m4(0.f, gvl);
        int64_t      length = hidden_size;
        ଯେତେବେଳେ (length > 0) {
            gvl                   = __riscv_vsetvl_e32m4(length);
            vfloat32m4_t src_data = __riscv_vle32_v_f32m4(p_ନିବେଶ, gvl);
            sum_sq                = __riscv_vfmacc_vv_f32m4(sum_sq, src_data, src_data, gvl);
            __riscv_vse32_v_f32m4(p_temp_output, src_data, gvl);

            p_ନିବେଶ += gvl;
            p_temp_output += gvl;
            length -= gvl;
        }

        gvl                 = __riscv_vsetvlmax_e32m1();
        vfloat32m1_t zero_v = __riscv_vfmv_v_f_f32m1(0.f, gvl);
        vfloat32m1_t mean_square_v =
            __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(sum_sq, 0), __riscv_vget_v_f32m4_f32m1(sum_sq, 1), gvl);

        mean_square_v = __riscv_vfadd_vv_f32m1(mean_square_v, __riscv_vget_v_f32m4_f32m1(sum_sq, 2), gvl);
        mean_square_v = __riscv_vfadd_vv_f32m1(mean_square_v, __riscv_vget_v_f32m4_f32m1(sum_sq, 3), gvl);
        mean_square_v = __riscv_vfredusum_vs_f32m1_f32m1(mean_square_v, zero_v, gvl);

        ଦଶମିକ mean_square = __riscv_vfmv_f_s_f32m1_f32(mean_square_v);
        mean_square /= hidden_size;

        mean_square = sqrt(mean_square + epsilon);

        mean_square   = 1.0f / mean_square;
        length        = hidden_size;
        p_temp_output = p_output;

        ଯେତେବେଳେ (length > 0) {
            gvl                   = __riscv_vsetvl_e32m4(length);
            vfloat32m4_t src_data = __riscv_vle32_v_f32m4(p_temp_output, gvl);
            src_data              = __riscv_vfmul_vf_f32m4(src_data, mean_square, gvl);
            __riscv_vse32_v_f32m4(p_output, src_data, gvl);
            p_temp_output += gvl;
            p_output += gvl;
            length -= gvl;
        }
    }
}

template <size_t MB_ROWS>
ଶୂନ୍ୟ quantize_a_nrow_i8_ref(size_t blk_len, ଅଚଳ ଦଶମିକ * a_ptr, size_t ଗଣନା_k, uint8_t * quant_a_ptr) {
    int64_t a_blk_stride        = q8_blk_size(blk_len, ସତ୍ୟ);
    int64_t a_nrow_block_stride = a_blk_stride * MB_ROWS;
    ପାଇଁ (size_t k = 0; k < ଗଣନା_k; k += blk_len, quant_a_ptr += a_nrow_block_stride) {
        ଦଶମିକ *   scale_a_ptr = reinterpret_cast<ଦଶମିକ *>(quant_a_ptr);
        int16_t * a_sum_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + ଆକାର(ଦଶମିକ) * MB_ROWS);
        int8_t *  quant_a_blk =
            reinterpret_cast<int8_t *>(quant_a_ptr + ଆକାର(ଦଶମିକ) * MB_ROWS + ଆକାର(int16_t) * MB_ROWS);

        ପାଇଁ (size_t row = 0; row < MB_ROWS; row++) {
            ଦଶମିକ max_abs_a = 0.0f;
            ପାଇଁ (size_t bk = 0; bk < blk_len; bk++) {
                max_abs_a = std::max(max_abs_a, std::abs(a_ptr[row * ଗଣନା_k + k + bk]));
            }

            ଦଶମିକ rep_scale_a = ((1 << 7) - 1) / max_abs_a;
            scale_a_ptr[row]  = 1 / rep_scale_a;

            int16_t a_sum = 0;
            ପାଇଁ (size_t bk = 0; bk < blk_len; bk++) {
                ଅଚଳ int8_t quantized = static_cast<int8_t>(
                    std::clamp(std::nearbyintf(a_ptr[row * ଗଣନା_k + k + bk] * rep_scale_a), -128.0f, 127.0f));
                quant_a_blk[row * blk_len + bk] = quantized;
                a_sum += quantized;
            }
            a_sum_ptr[row] = -a_sum;
        }
    }
}

template <size_t MB_ROWS>
ଶୂନ୍ୟ quantize_a_nrow_i8_hp_ref(size_t blk_len, ଅଚଳ ଦଶମିକ * a_ptr, size_t ଗଣନା_k, uint8_t * quant_a_ptr) {
    constexpr size_t k_subblk_len = 32;
    ଅଚଳ size_t     subblk_ଗଣନା = blk_len / k_subblk_len;

    GGML_ASSERT(blk_len == 256);

    ଦଶମିକ   scale_temp[8]       = { 0.0f };
    int64_t a_blk_stride        = q8_hp_blk_size(blk_len, ସତ୍ୟ, ସତ୍ୟ);
    int64_t a_nrow_block_stride = a_blk_stride * MB_ROWS;
    int64_t a_subblk_stride     = q8_hp_blk_size(k_subblk_len, ମିଥ୍ୟା, ମିଥ୍ୟା) * MB_ROWS;

    ପାଇଁ (size_t k = 0; k < ଗଣନା_k; k += blk_len, quant_a_ptr += a_nrow_block_stride) {
        _Float16 * a_sum_ptr = reinterpret_cast<_Float16 *>(quant_a_ptr + a_subblk_stride * subblk_ଗଣନା);

        ଦଶମିକ scale_avg = 0.0f;
        ପାଇଁ (size_t kk = 0; kk < subblk_ଗଣନା; kk++) {
            ଦଶମିକ max_abs_a = 0.0f;
            ପାଇଁ (size_t row = 0; row < MB_ROWS; row++) {
                ପାଇଁ (size_t bk = 0; bk < k_subblk_len; bk++) {
                    max_abs_a = std::max(max_abs_a, std::abs(a_ptr[row * ଗଣନା_k + k + bk + kk * k_subblk_len]));
                }
            }
            scale_temp[kk] = max_abs_a / ((1 << 7) - 1);
            scale_avg += scale_temp[kk];
        }

        scale_avg /= subblk_ଗଣନା;
        ଦଶମିକ scale_factor = 1.0f / scale_avg;

        _Float16 * scale_avg_ptr =
            reinterpret_cast<_Float16 *>(quant_a_ptr + a_nrow_block_stride - ଆକାର(_Float16) * MB_ROWS);
        scale_avg_ptr[0] = scale_avg;

        ପାଇଁ (size_t kk = 0; kk < subblk_ଗଣନା; kk++) {
            uint8_t *  a_subblk_base = quant_a_ptr + kk * a_subblk_stride;
            _Float16 * scale_a_ptr   = reinterpret_cast<_Float16 *>(a_subblk_base);
            int8_t *   quant_a_blk   = reinterpret_cast<int8_t *>(a_subblk_base + ଆକାର(_Float16) * MB_ROWS);

            scale_a_ptr[0] = static_cast<_Float16>(scale_temp[kk] * scale_factor);

            ଅଚଳ ଦଶମିକ rep_scale_a = 1.0f / scale_temp[kk];

            ପାଇଁ (size_t row = 0; row < MB_ROWS; row++) {
                int16_t a_sum = 0;
                ପାଇଁ (size_t bk = 0; bk < k_subblk_len; bk++) {
                    ଅଚଳ int8_t quantized = static_cast<int8_t>(
                        std::clamp(std::nearbyintf(a_ptr[row * ଗଣନା_k + k + bk + kk * k_subblk_len] * rep_scale_a),
                                   -128.0f, 127.0f));
                    quant_a_blk[row * k_subblk_len + bk] = quantized;
                    a_sum += quantized;
                }
                a_sum_ptr[row * subblk_ଗଣନା + kk] = static_cast<_Float16>(-a_sum) * static_cast<_Float16>(8.0f);
            }
        }
    }
}

template <size_t MB_ROWS>
ଶୂନ୍ୟ quantize_a_nrow_i8k_ref(size_t blk_len, ଅଚଳ ଦଶମିକ * a_ptr, size_t ଗଣନା_k, uint8_t * quant_a_ptr) {
    int64_t a_blk_stride        = q8k_blk_size(256);
    int64_t a_nrow_block_stride = a_blk_stride * MB_ROWS;
    int64_t a_sum_size          = 256 / 16;

    ପାଇଁ (size_t k = 0; k < ଗଣନା_k; k += blk_len, quant_a_ptr += a_nrow_block_stride) {
        ଦଶମିକ *   scale_a_ptr = reinterpret_cast<ଦଶମିକ *>(quant_a_ptr);
        int16_t * a_sum_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + ଆକାର(ଦଶମିକ) * MB_ROWS);
        int8_t *  quant_a_blk =
            reinterpret_cast<int8_t *>(quant_a_ptr + ଆକାର(ଦଶମିକ) * MB_ROWS + ଆକାର(int16_t) * a_sum_size * MB_ROWS);

        ପାଇଁ (size_t row = 0; row < MB_ROWS; row++) {
            ଦଶମିକ max_a     = 0.0f;
            ଦଶମିକ max_abs_a = 0.0f;
            ପାଇଁ (size_t bk = 0; bk < blk_len; bk++) {
                ଦଶମିକ ax = std::abs(a_ptr[row * ଗଣନା_k + k + bk]);
                ଯଦି (ax > max_abs_a) {
                    max_abs_a = ax;
                    max_a     = a_ptr[row * ଗଣନା_k + k + bk];
                }
            }

            ଯଦି (!max_abs_a) {
                scale_a_ptr[row] = 0;
                ପାଇଁ (size_t bki = 0; bki < a_sum_size; bki++) {
                    ପାଇଁ (size_t bk = bki * 16; bk < (bki + 1) * 16; bk++) {
                        quant_a_blk[row * blk_len + bk] = 0;
                    }
                    a_sum_ptr[row * a_sum_size + bki] = 0;
                }
                ଜାରି;
            }

            ଦଶମିକ rep_scale_a = ((1 << 7) - 1) / max_abs_a;
            scale_a_ptr[row]  = 1 / rep_scale_a;

            ପାଇଁ (size_t bki = 0; bki < a_sum_size; bki++) {
                int16_t a_sum = 0;
                ପାଇଁ (size_t bk = bki * 16; bk < (bki + 1) * 16; bk++) {
                    ଅଚଳ int8_t quantized = static_cast<int8_t>(
                        std::clamp(std::nearbyintf(a_ptr[row * ଗଣନା_k + k + bk] * rep_scale_a), -128.0f, 127.0f));
                    quant_a_blk[row * blk_len + bk] = quantized;
                    a_sum += quantized;
                }
                a_sum_ptr[row * a_sum_size + bki] = -a_sum;
            }
        }
    }
}

ଶୂନ୍ୟ quantize_a_row_i8(size_t blk_len, ଅଚଳ ଦଶମିକ * a_ptr, size_t ଗଣନା_k, uint8_t * quant_a_ptr) {
    GGML_ASSERT(blk_len == 32);
    int64_t a_blk_stride = q8_blk_size(blk_len, ସତ୍ୟ);
    size_t  vlenb        = __riscv_vlenb();

    ଯଦି (vlenb == 128) {
        ପାଇଁ (size_t k = 0; k < ଗଣନା_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            ଦଶମିକ *   scale_a_ptr = reinterpret_cast<ଦଶମିକ *>(quant_a_ptr);
            int16_t * a_sum_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + ଆକାର(ଦଶମିକ));
            int8_t *  quant_a_blk = reinterpret_cast<int8_t *>(quant_a_ptr + ଆକାର(ଦଶମିକ) + ଆକାର(int16_t));

            size_t       vl      = __riscv_vsetvl_e32m1(blk_len);
            vfloat32m1_t v_a     = __riscv_vle32_v_f32m1(a_ptr + k, vl);
            vfloat32m1_t v_a_abs = __riscv_vfabs_v_f32m1(v_a, vl);

            vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m1_f32m1(v_a_abs, tmp, vl);
            ଦଶମିକ        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

            ଦଶମିକ scale_a     = max_abs_a / ((1 << 7) - 1);
            ଦଶମିକ rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
            scale_a_ptr[0]    = scale_a;

            vfloat32m1_t v_a_scale    = __riscv_vfmul_vf_f32m1(v_a, rep_scale_a, vl);
            vint16mf2_t  v_a_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a_scale, vl);
            vint8mf4_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a_quant, vl);

            vint16m1_t tmp_sum = __riscv_vmv_v_x_i16m1(0, vl);
            vint16m1_t v_a_sum = __riscv_vwredsum_vs_i8mf4_i16m1(v_a_quant_i8, tmp_sum, vl);
            int16_t    a_sum   = __riscv_vmv_x_s_i16m1_i16(v_a_sum);
            a_sum_ptr[0]       = -a_sum;

            __riscv_vse8_v_i8mf4(quant_a_blk, v_a_quant_i8, vl);
        }
    } ଅନ୍ୟଥା ଯଦି (vlenb == 32) {
        ପାଇଁ (size_t k = 0; k < ଗଣନା_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            ଦଶମିକ *   scale_a_ptr = reinterpret_cast<ଦଶମିକ *>(quant_a_ptr);
            int16_t * a_sum_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + ଆକାର(ଦଶମିକ));
            int8_t *  quant_a_blk = reinterpret_cast<int8_t *>(quant_a_ptr + ଆକାର(ଦଶମିକ) + ଆକାର(int16_t));

            size_t       vl      = __riscv_vsetvl_e32m4(blk_len);
            vfloat32m4_t v_a     = __riscv_vle32_v_f32m4(a_ptr + k, vl);
            vfloat32m4_t v_a_abs = __riscv_vfabs_v_f32m4(v_a, vl);

            vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m4_f32m1(v_a_abs, tmp, vl);
            ଦଶମିକ        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

            ଦଶମିକ scale_a     = max_abs_a / ((1 << 7) - 1);
            ଦଶମିକ rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
            scale_a_ptr[0]    = scale_a;

            vfloat32m4_t v_a_scale    = __riscv_vfmul_vf_f32m4(v_a, rep_scale_a, vl);
            vint16m2_t   v_a_quant    = __riscv_vfncvt_x_f_w_i16m2(v_a_scale, vl);
            vint8m1_t    v_a_quant_i8 = __riscv_vncvt_x_x_w_i8m1(v_a_quant, vl);

            vint16m1_t tmp_sum = __riscv_vmv_v_x_i16m1(0, vl);
            vint16m1_t v_a_sum = __riscv_vwredsum_vs_i8m1_i16m1(v_a_quant_i8, tmp_sum, vl);
            int16_t    a_sum   = __riscv_vmv_x_s_i16m1_i16(v_a_sum);
            a_sum_ptr[0]       = -a_sum;

            __riscv_vse8_v_i8m1(quant_a_blk, v_a_quant_i8, vl);
        }
    } ଅନ୍ୟଥା {
        quantize_a_nrow_i8_ref<1>(blk_len, a_ptr, ଗଣନା_k, quant_a_ptr);
    }
}

ଶୂନ୍ୟ quantize_a_4row_i8(size_t blk_len, ଅଚଳ ଦଶମିକ * a_ptr, size_t ଗଣନା_k, uint8_t * quant_a_ptr) {
    GGML_ASSERT(blk_len == 32);
    int64_t a_blk_stride        = q8_blk_size(blk_len, ସତ୍ୟ);
    int64_t a_nrow_block_stride = a_blk_stride * 4;
    size_t  vlenb               = __riscv_vlenb();

    ଯଦି (vlenb == 128) {
        ପାଇଁ (size_t k = 0; k < ଗଣନା_k; k += blk_len, quant_a_ptr += a_nrow_block_stride) {
            ଦଶମିକ *   scale_a_ptr = reinterpret_cast<ଦଶମିକ *>(quant_a_ptr);
            int16_t * a_sum_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + ଆକାର(ଦଶମିକ) * 4);
            int8_t *  quant_a_blk = reinterpret_cast<int8_t *>(quant_a_ptr + ଆକାର(ଦଶମିକ) * 4 + ଆକାର(int16_t) * 4);

            ପାଇଁ (size_t mi = 0; mi < 4; mi++) {
                size_t       vl      = __riscv_vsetvl_e32m1(blk_len);
                vfloat32m1_t v_a     = __riscv_vle32_v_f32m1(a_ptr + mi * ଗଣନା_k + k, vl);
                vfloat32m1_t v_a_abs = __riscv_vfabs_v_f32m1(v_a, vl);

                vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m1_f32m1(v_a_abs, tmp, vl);
                ଦଶମିକ        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                ଦଶମିକ scale_a     = max_abs_a / ((1 << 7) - 1);
                ଦଶମିକ rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
                scale_a_ptr[mi]   = scale_a;

                vfloat32m1_t v_a_scale    = __riscv_vfmul_vf_f32m1(v_a, rep_scale_a, vl);
                vint16mf2_t  v_a_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a_scale, vl);
                vint8mf4_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a_quant, vl);

                vint16m1_t tmp_sum = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a_sum = __riscv_vwredsum_vs_i8mf4_i16m1(v_a_quant_i8, tmp_sum, vl);
                int16_t    a_sum   = __riscv_vmv_x_s_i16m1_i16(v_a_sum);
                a_sum_ptr[mi]      = -a_sum;

                __riscv_vse8_v_i8mf4(quant_a_blk + mi * blk_len, v_a_quant_i8, vl);
            }
        }
    } ଅନ୍ୟଥା ଯଦି (vlenb == 32) {
        ପାଇଁ (size_t k = 0; k < ଗଣନା_k; k += blk_len, quant_a_ptr += a_nrow_block_stride) {
            ଦଶମିକ *   scale_a_ptr = reinterpret_cast<ଦଶମିକ *>(quant_a_ptr);
            int16_t * a_sum_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + ଆକାର(ଦଶମିକ) * 4);
            int8_t *  quant_a_blk = reinterpret_cast<int8_t *>(quant_a_ptr + ଆକାର(ଦଶମିକ) * 4 + ଆକାର(int16_t) * 4);

            ପାଇଁ (size_t mi = 0; mi < 4; mi++) {
                size_t       vl      = __riscv_vsetvl_e32m4(blk_len);
                vfloat32m4_t v_a     = __riscv_vle32_v_f32m4(a_ptr + mi * ଗଣନା_k + k, vl);
                vfloat32m4_t v_a_abs = __riscv_vfabs_v_f32m4(v_a, vl);

                vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m4_f32m1(v_a_abs, tmp, vl);
                ଦଶମିକ        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                ଦଶମିକ scale_a     = max_abs_a / ((1 << 7) - 1);
                ଦଶମିକ rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
                scale_a_ptr[mi]   = scale_a;

                vfloat32m4_t v_a_scale    = __riscv_vfmul_vf_f32m4(v_a, rep_scale_a, vl);
                vint16m2_t   v_a_quant    = __riscv_vfncvt_x_f_w_i16m2(v_a_scale, vl);
                vint8m1_t    v_a_quant_i8 = __riscv_vncvt_x_x_w_i8m1(v_a_quant, vl);

                vint16m1_t tmp_sum = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a_sum = __riscv_vwredsum_vs_i8m1_i16m1(v_a_quant_i8, tmp_sum, vl);
                int16_t    a_sum   = __riscv_vmv_x_s_i16m1_i16(v_a_sum);
                a_sum_ptr[mi]      = -a_sum;

                __riscv_vse8_v_i8m1(quant_a_blk + mi * blk_len, v_a_quant_i8, vl);
            }
        }
    } ଅନ୍ୟଥା {
        quantize_a_nrow_i8_ref<4>(blk_len, a_ptr, ଗଣନା_k, quant_a_ptr);
    }
}

ଶୂନ୍ୟ quantize_a_row_i8_hp(size_t blk_len, ଅଚଳ ଦଶମିକ * a_ptr, size_t ଗଣନା_k, uint8_t * quant_a_ptr) {
    constexpr size_t k_subblk_len = 32;
    GGML_ASSERT(blk_len == 256);

    constexpr size_t subblk_ଗଣନା             = 256 / k_subblk_len;
    int64_t          a_blk_stride             = q8_hp_blk_size(blk_len, ସତ୍ୟ, ସତ୍ୟ);
    int64_t          a_subblk_stride          = q8_hp_blk_size(k_subblk_len, ମିଥ୍ୟା, ମିଥ୍ୟା);
    size_t           vlenb                    = __riscv_vlenb();
    ଦଶମିକ            scale_temp[subblk_ଗଣନା] = { 0.0f };

    ଯଦି (vlenb == 128) {
        ପାଇଁ (size_t k = 0; k < ଗଣନା_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            _Float16 * a_sum_ptr     = reinterpret_cast<_Float16 *>(quant_a_ptr + a_subblk_stride * subblk_ଗଣନା);
            _Float16 * scale_avg_ptr = reinterpret_cast<_Float16 *>(quant_a_ptr + a_blk_stride - ଆକାର(_Float16));
            ଦଶମିକ      scale_avg     = 0.0f;

            ପାଇଁ (size_t kk = 0; kk < subblk_ଗଣନା; ++kk) {
                ଅଚଳ ଦଶମିକ * a_src_ptr = a_ptr + k + kk * k_subblk_len;

                size_t       vl      = __riscv_vsetvl_e32m1(k_subblk_len);
                vfloat32m1_t v_a     = __riscv_vle32_v_f32m1(a_src_ptr, vl);
                vfloat32m1_t v_a_abs = __riscv_vfabs_v_f32m1(v_a, vl);

                vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m1_f32m1(v_a_abs, tmp, vl);
                ଦଶମିକ        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                scale_temp[kk] = max_abs_a / ((1 << 7) - 1);
                scale_avg += scale_temp[kk];
            }

            scale_avg /= subblk_ଗଣନା;
            ଅଚଳ ଦଶମିକ scale_factor = scale_avg ? 1.0f / scale_avg : 0.0f;
            scale_avg_ptr[0]         = static_cast<_Float16>(scale_avg);

            ପାଇଁ (size_t kk = 0; kk < subblk_ଗଣନା; ++kk) {
                uint8_t *     a_subblk_base = quant_a_ptr + kk * a_subblk_stride;
                _Float16 *    scale_a_ptr   = reinterpret_cast<_Float16 *>(a_subblk_base);
                int8_t *      quant_a_blk   = reinterpret_cast<int8_t *>(a_subblk_base + ଆକାର(_Float16));
                ଅଚଳ ଦଶମିକ * a_src_ptr     = a_ptr + k + kk * k_subblk_len;

                size_t       vl          = __riscv_vsetvl_e32m1(k_subblk_len);
                vfloat32m1_t v_a         = __riscv_vle32_v_f32m1(a_src_ptr, vl);
                ଦଶମିକ        rep_scale_a = scale_temp[kk] ? 1.0f / scale_temp[kk] : 0.0f;
                scale_a_ptr[0]           = static_cast<_Float16>(scale_temp[kk] * scale_factor);

                vfloat32m1_t v_a_scale    = __riscv_vfmul_vf_f32m1(v_a, rep_scale_a, vl);
                vint16mf2_t  v_a_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a_scale, vl);
                vint8mf4_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a_quant, vl);

                vint16m1_t tmp_sum = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a_sum = __riscv_vwredsum_vs_i8mf4_i16m1(v_a_quant_i8, tmp_sum, vl);
                int16_t    a_sum   = __riscv_vmv_x_s_i16m1_i16(v_a_sum);
                a_sum_ptr[kk]      = static_cast<_Float16>(-a_sum) * static_cast<_Float16>(8.0f);

                __riscv_vse8_v_i8mf4(quant_a_blk, v_a_quant_i8, vl);
            }
        }
    } ଅନ୍ୟଥା ଯଦି (vlenb == 32) {
        ପାଇଁ (size_t k = 0; k < ଗଣନା_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            _Float16 * a_sum_ptr     = reinterpret_cast<_Float16 *>(quant_a_ptr + a_subblk_stride * subblk_ଗଣନା);
            _Float16 * scale_avg_ptr = reinterpret_cast<_Float16 *>(quant_a_ptr + a_blk_stride - ଆକାର(_Float16));
            ଦଶମିକ      scale_avg     = 0.0f;

            ପାଇଁ (size_t kk = 0; kk < subblk_ଗଣନା; ++kk) {
                ଅଚଳ ଦଶମିକ * a_src_ptr = a_ptr + k + kk * k_subblk_len;

                size_t       vl      = __riscv_vsetvl_e32m4(k_subblk_len);
                vfloat32m4_t v_a     = __riscv_vle32_v_f32m4(a_src_ptr, vl);
                vfloat32m4_t v_a_abs = __riscv_vfabs_v_f32m4(v_a, vl);

                vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m4_f32m1(v_a_abs, tmp, vl);
                ଦଶମିକ        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                scale_temp[kk] = max_abs_a / ((1 << 7) - 1);
                scale_avg += scale_temp[kk];
            }

            scale_avg /= subblk_ଗଣନା;
            ଅଚଳ ଦଶମିକ scale_factor = scale_avg ? 1.0f / scale_avg : 0.0f;
            scale_avg_ptr[0]         = static_cast<_Float16>(scale_avg);

            ପାଇଁ (size_t kk = 0; kk < subblk_ଗଣନା; ++kk) {
                uint8_t *     a_subblk_base = quant_a_ptr + kk * a_subblk_stride;
                _Float16 *    scale_a_ptr   = reinterpret_cast<_Float16 *>(a_subblk_base);
                int8_t *      quant_a_blk   = reinterpret_cast<int8_t *>(a_subblk_base + ଆକାର(_Float16));
                ଅଚଳ ଦଶମିକ * a_src_ptr     = a_ptr + k + kk * k_subblk_len;

                size_t       vl          = __riscv_vsetvl_e32m4(k_subblk_len);
                vfloat32m4_t v_a         = __riscv_vle32_v_f32m4(a_src_ptr, vl);
                ଦଶମିକ        rep_scale_a = scale_temp[kk] ? 1.0f / scale_temp[kk] : 0.0f;
                scale_a_ptr[0]           = static_cast<_Float16>(scale_temp[kk] * scale_factor);

                vfloat32m4_t v_a_scale    = __riscv_vfmul_vf_f32m4(v_a, rep_scale_a, vl);
                vint16m2_t   v_a_quant    = __riscv_vfncvt_x_f_w_i16m2(v_a_scale, vl);
                vint8m1_t    v_a_quant_i8 = __riscv_vncvt_x_x_w_i8m1(v_a_quant, vl);

                vint16m1_t tmp_sum = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a_sum = __riscv_vwredsum_vs_i8m1_i16m1(v_a_quant_i8, tmp_sum, vl);
                int16_t    a_sum   = __riscv_vmv_x_s_i16m1_i16(v_a_sum);
                a_sum_ptr[kk]      = static_cast<_Float16>(-a_sum) * static_cast<_Float16>(8.0f);

                __riscv_vse8_v_i8m1(quant_a_blk, v_a_quant_i8, vl);
            }
        }
    } ଅନ୍ୟଥା {
        quantize_a_nrow_i8_hp_ref<1>(blk_len, a_ptr, ଗଣନା_k, quant_a_ptr);
    }
}

ଶୂନ୍ୟ quantize_a_4row_i8_hp(size_t blk_len, ଅଚଳ ଦଶମିକ * a_ptr, size_t ଗଣନା_k, uint8_t * quant_a_ptr) {
    constexpr size_t k_subblk_len = 32;
    GGML_ASSERT(blk_len == 256);

    constexpr size_t subblk_ଗଣନା             = 256 / k_subblk_len;
    int64_t          a_blk_stride             = q8_hp_blk_size(blk_len, ସତ୍ୟ, ସତ୍ୟ);
    int64_t          a_nrow_block_stride      = a_blk_stride * 4;
    int64_t          a_subblk_stride          = q8_hp_blk_size(k_subblk_len, ମିଥ୍ୟା, ମିଥ୍ୟା) * 4;
    size_t           vlenb                    = __riscv_vlenb();
    ଦଶମିକ            scale_temp[subblk_ଗଣନା] = { 0.0f };

    ଯଦି (vlenb == 128) {
        ପାଇଁ (size_t k = 0; k < ଗଣନା_k; k += blk_len, quant_a_ptr += a_nrow_block_stride) {
            _Float16 * a_sum_ptr = reinterpret_cast<_Float16 *>(quant_a_ptr + a_subblk_stride * subblk_ଗଣନା);
            _Float16 * scale_avg_ptr =
                reinterpret_cast<_Float16 *>(quant_a_ptr + a_nrow_block_stride - ଆକାର(_Float16) * 4);
            ଦଶମିକ scale_avg = 0.0f;

            ପାଇଁ (size_t kk = 0; kk < subblk_ଗଣନା; ++kk) {
                ଅଚଳ ଦଶମିକ * a_src_ptr0 = a_ptr + 0 * ଗଣନା_k + k + kk * k_subblk_len;
                ଅଚଳ ଦଶମିକ * a_src_ptr1 = a_ptr + 1 * ଗଣନା_k + k + kk * k_subblk_len;
                ଅଚଳ ଦଶମିକ * a_src_ptr2 = a_ptr + 2 * ଗଣନା_k + k + kk * k_subblk_len;
                ଅଚଳ ଦଶମିକ * a_src_ptr3 = a_ptr + 3 * ଗଣନା_k + k + kk * k_subblk_len;

                size_t       vl       = __riscv_vsetvl_e32m1(k_subblk_len);
                vfloat32m1_t v_a0     = __riscv_vle32_v_f32m1(a_src_ptr0, vl);
                vfloat32m1_t v_a1     = __riscv_vle32_v_f32m1(a_src_ptr1, vl);
                vfloat32m1_t v_a2     = __riscv_vle32_v_f32m1(a_src_ptr2, vl);
                vfloat32m1_t v_a3     = __riscv_vle32_v_f32m1(a_src_ptr3, vl);
                vfloat32m1_t v_a0_abs = __riscv_vfabs_v_f32m1(v_a0, vl);
                vfloat32m1_t v_a1_abs = __riscv_vfabs_v_f32m1(v_a1, vl);
                vfloat32m1_t v_a2_abs = __riscv_vfabs_v_f32m1(v_a2, vl);
                vfloat32m1_t v_a3_abs = __riscv_vfabs_v_f32m1(v_a3, vl);

                vfloat32m1_t v_max_abs = __riscv_vfmax_vv_f32m1(v_a0_abs, v_a1_abs, vl);
                v_max_abs              = __riscv_vfmax_vv_f32m1(v_max_abs, v_a2_abs, vl);
                v_max_abs              = __riscv_vfmax_vv_f32m1(v_max_abs, v_a3_abs, vl);

                vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m1_f32m1(v_max_abs, tmp, vl);
                ଦଶମିକ        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                scale_temp[kk] = max_abs_a / ((1 << 7) - 1);
                scale_avg += scale_temp[kk];
            }

            scale_avg /= subblk_ଗଣନା;
            ଅଚଳ ଦଶମିକ scale_factor = scale_avg ? 1.0f / scale_avg : 0.0f;
            scale_avg_ptr[0]         = static_cast<_Float16>(scale_avg);

            ପାଇଁ (size_t kk = 0; kk < subblk_ଗଣନା; ++kk) {
                uint8_t *     a_subblk_base = quant_a_ptr + kk * a_subblk_stride;
                _Float16 *    scale_a_ptr   = reinterpret_cast<_Float16 *>(a_subblk_base);
                int8_t *      quant_a_blk   = reinterpret_cast<int8_t *>(a_subblk_base + ଆକାର(_Float16) * 4);
                ଅଚଳ ଦଶମିକ * a_src_ptr0    = a_ptr + 0 * ଗଣନା_k + k + kk * k_subblk_len;
                ଅଚଳ ଦଶମିକ * a_src_ptr1    = a_ptr + 1 * ଗଣନା_k + k + kk * k_subblk_len;
                ଅଚଳ ଦଶମିକ * a_src_ptr2    = a_ptr + 2 * ଗଣନା_k + k + kk * k_subblk_len;
                ଅଚଳ ଦଶମିକ * a_src_ptr3    = a_ptr + 3 * ଗଣନା_k + k + kk * k_subblk_len;

                size_t       vl   = __riscv_vsetvl_e32m1(k_subblk_len);
                vfloat32m1_t v_a0 = __riscv_vle32_v_f32m1(a_src_ptr0, vl);
                vfloat32m1_t v_a1 = __riscv_vle32_v_f32m1(a_src_ptr1, vl);
                vfloat32m1_t v_a2 = __riscv_vle32_v_f32m1(a_src_ptr2, vl);
                vfloat32m1_t v_a3 = __riscv_vle32_v_f32m1(a_src_ptr3, vl);

                ଦଶମିକ rep_scale_a = scale_temp[kk] ? 1.0f / scale_temp[kk] : 0.0f;
                scale_a_ptr[0]    = static_cast<_Float16>(scale_temp[kk] * scale_factor);

                vfloat32m1_t v_a0_scale    = __riscv_vfmul_vf_f32m1(v_a0, rep_scale_a, vl);
                vfloat32m1_t v_a1_scale    = __riscv_vfmul_vf_f32m1(v_a1, rep_scale_a, vl);
                vfloat32m1_t v_a2_scale    = __riscv_vfmul_vf_f32m1(v_a2, rep_scale_a, vl);
                vfloat32m1_t v_a3_scale    = __riscv_vfmul_vf_f32m1(v_a3, rep_scale_a, vl);
                vint16mf2_t  v_a0_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a0_scale, vl);
                vint16mf2_t  v_a1_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a1_scale, vl);
                vint16mf2_t  v_a2_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a2_scale, vl);
                vint16mf2_t  v_a3_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a3_scale, vl);
                vint8mf4_t   v_a0_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a0_quant, vl);
                vint8mf4_t   v_a1_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a1_quant, vl);
                vint8mf4_t   v_a2_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a2_quant, vl);
                vint8mf4_t   v_a3_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a3_quant, vl);

                vint16m1_t tmp_sum0 = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t tmp_sum1 = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t tmp_sum2 = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t tmp_sum3 = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a0_sum = __riscv_vwredsum_vs_i8mf4_i16m1(v_a0_quant_i8, tmp_sum0, vl);
                vint16m1_t v_a1_sum = __riscv_vwredsum_vs_i8mf4_i16m1(v_a1_quant_i8, tmp_sum1, vl);
                vint16m1_t v_a2_sum = __riscv_vwredsum_vs_i8mf4_i16m1(v_a2_quant_i8, tmp_sum2, vl);
                vint16m1_t v_a3_sum = __riscv_vwredsum_vs_i8mf4_i16m1(v_a3_quant_i8, tmp_sum3, vl);

                a_sum_ptr[0 * subblk_ଗଣନା + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a0_sum)) * static_cast<_Float16>(8.0f);
                a_sum_ptr[1 * subblk_ଗଣନା + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a1_sum)) * static_cast<_Float16>(8.0f);
                a_sum_ptr[2 * subblk_ଗଣନା + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a2_sum)) * static_cast<_Float16>(8.0f);
                a_sum_ptr[3 * subblk_ଗଣନା + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a3_sum)) * static_cast<_Float16>(8.0f);

                __riscv_vse8_v_i8mf4(quant_a_blk + 0 * k_subblk_len, v_a0_quant_i8, vl);
                __riscv_vse8_v_i8mf4(quant_a_blk + 1 * k_subblk_len, v_a1_quant_i8, vl);
                __riscv_vse8_v_i8mf4(quant_a_blk + 2 * k_subblk_len, v_a2_quant_i8, vl);
                __riscv_vse8_v_i8mf4(quant_a_blk + 3 * k_subblk_len, v_a3_quant_i8, vl);
            }
        }
    } ଅନ୍ୟଥା ଯଦି (vlenb == 32) {
        ପାଇଁ (size_t k = 0; k < ଗଣନା_k; k += blk_len, quant_a_ptr += a_nrow_block_stride) {
            _Float16 * a_sum_ptr = reinterpret_cast<_Float16 *>(quant_a_ptr + a_subblk_stride * subblk_ଗଣନା);
            _Float16 * scale_avg_ptr =
                reinterpret_cast<_Float16 *>(quant_a_ptr + a_nrow_block_stride - ଆକାର(_Float16) * 4);
            ଦଶମିକ scale_avg = 0.0f;

            ପାଇଁ (size_t kk = 0; kk < subblk_ଗଣନା; ++kk) {
                ଅଚଳ ଦଶମିକ * a_src_ptr0 = a_ptr + 0 * ଗଣନା_k + k + kk * k_subblk_len;
                ଅଚଳ ଦଶମିକ * a_src_ptr1 = a_ptr + 1 * ଗଣନା_k + k + kk * k_subblk_len;
                ଅଚଳ ଦଶମିକ * a_src_ptr2 = a_ptr + 2 * ଗଣନା_k + k + kk * k_subblk_len;
                ଅଚଳ ଦଶମିକ * a_src_ptr3 = a_ptr + 3 * ଗଣନା_k + k + kk * k_subblk_len;

                size_t       vl   = __riscv_vsetvl_e32m4(k_subblk_len);
                vfloat32m4_t v_a0 = __riscv_vle32_v_f32m4(a_src_ptr0, vl);
                vfloat32m4_t v_a1 = __riscv_vle32_v_f32m4(a_src_ptr1, vl);
                vfloat32m4_t v_a2 = __riscv_vle32_v_f32m4(a_src_ptr2, vl);
                vfloat32m4_t v_a3 = __riscv_vle32_v_f32m4(a_src_ptr3, vl);

                vfloat32m4_t v_a0_abs = __riscv_vfabs_v_f32m4(v_a0, vl);
                vfloat32m4_t v_a1_abs = __riscv_vfabs_v_f32m4(v_a1, vl);
                vfloat32m4_t v_a2_abs = __riscv_vfabs_v_f32m4(v_a2, vl);
                vfloat32m4_t v_a3_abs = __riscv_vfabs_v_f32m4(v_a3, vl);

                vfloat32m4_t v_max_abs = __riscv_vfmax_vv_f32m4(v_a0_abs, v_a1_abs, vl);
                v_max_abs              = __riscv_vfmax_vv_f32m4(v_max_abs, v_a2_abs, vl);
                v_max_abs              = __riscv_vfmax_vv_f32m4(v_max_abs, v_a3_abs, vl);

                vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m4_f32m1(v_max_abs, tmp, vl);
                ଦଶମିକ        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                scale_temp[kk] = max_abs_a / ((1 << 7) - 1);
                scale_avg += scale_temp[kk];
            }

            scale_avg /= subblk_ଗଣନା;
            ଅଚଳ ଦଶମିକ scale_factor = scale_avg ? 1.0f / scale_avg : 0.0f;
            scale_avg_ptr[0]         = static_cast<_Float16>(scale_avg);

            ପାଇଁ (size_t kk = 0; kk < subblk_ଗଣନା; ++kk) {
                uint8_t *     a_subblk_base = quant_a_ptr + kk * a_subblk_stride;
                _Float16 *    scale_a_ptr   = reinterpret_cast<_Float16 *>(a_subblk_base);
                int8_t *      quant_a_blk   = reinterpret_cast<int8_t *>(a_subblk_base + ଆକାର(_Float16) * 4);
                ଅଚଳ ଦଶମିକ * a_src_ptr0    = a_ptr + 0 * ଗଣନା_k + k + kk * k_subblk_len;
                ଅଚଳ ଦଶମିକ * a_src_ptr1    = a_ptr + 1 * ଗଣନା_k + k + kk * k_subblk_len;
                ଅଚଳ ଦଶମିକ * a_src_ptr2    = a_ptr + 2 * ଗଣନା_k + k + kk * k_subblk_len;
                ଅଚଳ ଦଶମିକ * a_src_ptr3    = a_ptr + 3 * ଗଣନା_k + k + kk * k_subblk_len;

                size_t       vl   = __riscv_vsetvl_e32m4(k_subblk_len);
                vfloat32m4_t v_a0 = __riscv_vle32_v_f32m4(a_src_ptr0, vl);
                vfloat32m4_t v_a1 = __riscv_vle32_v_f32m4(a_src_ptr1, vl);
                vfloat32m4_t v_a2 = __riscv_vle32_v_f32m4(a_src_ptr2, vl);
                vfloat32m4_t v_a3 = __riscv_vle32_v_f32m4(a_src_ptr3, vl);

                ଦଶମିକ rep_scale_a = scale_temp[kk] ? 1.0f / scale_temp[kk] : 0.0f;
                scale_a_ptr[0]    = static_cast<_Float16>(scale_temp[kk] * scale_factor);

                vfloat32m4_t v_a0_scale    = __riscv_vfmul_vf_f32m4(v_a0, rep_scale_a, vl);
                vfloat32m4_t v_a1_scale    = __riscv_vfmul_vf_f32m4(v_a1, rep_scale_a, vl);
                vfloat32m4_t v_a2_scale    = __riscv_vfmul_vf_f32m4(v_a2, rep_scale_a, vl);
                vfloat32m4_t v_a3_scale    = __riscv_vfmul_vf_f32m4(v_a3, rep_scale_a, vl);
                vint16m2_t   v_a0_quant    = __riscv_vfncvt_x_f_w_i16m2(v_a0_scale, vl);
                vint16m2_t   v_a1_quant    = __riscv_vfncvt_x_f_w_i16m2(v_a1_scale, vl);
                vint16m2_t   v_a2_quant    = __riscv_vfncvt_x_f_w_i16m2(v_a2_scale, vl);
                vint16m2_t   v_a3_quant    = __riscv_vfncvt_x_f_w_i16m2(v_a3_scale, vl);
                vint8m1_t    v_a0_quant_i8 = __riscv_vncvt_x_x_w_i8m1(v_a0_quant, vl);
                vint8m1_t    v_a1_quant_i8 = __riscv_vncvt_x_x_w_i8m1(v_a1_quant, vl);
                vint8m1_t    v_a2_quant_i8 = __riscv_vncvt_x_x_w_i8m1(v_a2_quant, vl);
                vint8m1_t    v_a3_quant_i8 = __riscv_vncvt_x_x_w_i8m1(v_a3_quant, vl);

                vint16m1_t tmp_sum0 = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t tmp_sum1 = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t tmp_sum2 = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t tmp_sum3 = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a0_sum = __riscv_vwredsum_vs_i8m1_i16m1(v_a0_quant_i8, tmp_sum0, vl);
                vint16m1_t v_a1_sum = __riscv_vwredsum_vs_i8m1_i16m1(v_a1_quant_i8, tmp_sum1, vl);
                vint16m1_t v_a2_sum = __riscv_vwredsum_vs_i8m1_i16m1(v_a2_quant_i8, tmp_sum2, vl);
                vint16m1_t v_a3_sum = __riscv_vwredsum_vs_i8m1_i16m1(v_a3_quant_i8, tmp_sum3, vl);

                a_sum_ptr[0 * subblk_ଗଣନା + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a0_sum)) * static_cast<_Float16>(8.0f);
                a_sum_ptr[1 * subblk_ଗଣନା + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a1_sum)) * static_cast<_Float16>(8.0f);
                a_sum_ptr[2 * subblk_ଗଣନା + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a2_sum)) * static_cast<_Float16>(8.0f);
                a_sum_ptr[3 * subblk_ଗଣନା + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a3_sum)) * static_cast<_Float16>(8.0f);

                __riscv_vse8_v_i8m1(quant_a_blk + 0 * k_subblk_len, v_a0_quant_i8, vl);
                __riscv_vse8_v_i8m1(quant_a_blk + 1 * k_subblk_len, v_a1_quant_i8, vl);
                __riscv_vse8_v_i8m1(quant_a_blk + 2 * k_subblk_len, v_a2_quant_i8, vl);
                __riscv_vse8_v_i8m1(quant_a_blk + 3 * k_subblk_len, v_a3_quant_i8, vl);
            }
        }
    } ଅନ୍ୟଥା {
        quantize_a_nrow_i8_hp_ref<4>(blk_len, a_ptr, ଗଣନା_k, quant_a_ptr);
    }
}

ଶୂନ୍ୟ quantize_a_row_i8k(size_t blk_len, ଅଚଳ ଦଶମିକ * a_ptr, size_t ଗଣନା_k, uint8_t * quant_a_ptr) {
    GGML_ASSERT(blk_len == 256);
    constexpr int64_t a_blk_stride = q8k_blk_size(256);
    constexpr int64_t a_sum_size   = 256 / 16;
    size_t            vlenb        = __riscv_vlenb();

    ଯଦି (vlenb == 128) {
        // vlen = 1024 bits, can process 32 float32 elements with m1
        ପାଇଁ (size_t k = 0; k < ଗଣନା_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            ଦଶମିକ *   scale_a_ptr = reinterpret_cast<ଦଶମିକ *>(quant_a_ptr);
            int16_t * a_sum_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + ଆକାର(ଦଶମିକ));
            int8_t *  quant_a_blk =
                reinterpret_cast<int8_t *>(quant_a_ptr + ଆକାର(ଦଶମିକ) + ଆକାର(int16_t) * a_sum_size);

            // Find max absolute value across all 256 elements
            size_t       vl        = __riscv_vsetvl_e32m1(16);
            vfloat32m1_t v_max_abs = __riscv_vfmv_v_f_f32m1(0.0f, vl);

            ପାଇଁ (size_t bki = 0; bki < a_sum_size; bki++) {
                vfloat32m1_t v_a     = __riscv_vle32_v_f32m1(a_ptr + k + bki * 16, vl);
                vfloat32m1_t v_a_abs = __riscv_vfabs_v_f32m1(v_a, vl);
                v_max_abs            = __riscv_vfmax_vv_f32m1(v_a_abs, v_max_abs, vl);
            }
            vfloat32m1_t tmp         = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t v_local_max = __riscv_vfredmax_vs_f32m1_f32m1(v_max_abs, tmp, vl);
            ଦଶମିକ        max_abs_a   = __riscv_vfmv_f_s_f32m1_f32(v_local_max);

            ଦଶମିକ scale_a     = max_abs_a / ((1 << 7) - 1);
            ଦଶମିକ rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
            scale_a_ptr[0]    = scale_a;

            // Quantize and compute sums for each 16-element group
            ପାଇଁ (size_t bki = 0; bki < a_sum_size; bki++) {
                vfloat32m1_t v_a          = __riscv_vle32_v_f32m1(a_ptr + k + bki * 16, vl);
                vfloat32m1_t v_a_scale    = __riscv_vfmul_vf_f32m1(v_a, rep_scale_a, vl);
                vint16mf2_t  v_a_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a_scale, vl);
                vint8mf4_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a_quant, vl);

                vint16m1_t tmp_sum = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a_sum = __riscv_vwredsum_vs_i8mf4_i16m1(v_a_quant_i8, tmp_sum, vl);
                int16_t    a_sum   = __riscv_vmv_x_s_i16m1_i16(v_a_sum);
                a_sum_ptr[bki]     = -a_sum;

                __riscv_vse8_v_i8mf4(quant_a_blk + bki * 16, v_a_quant_i8, vl);
            }
        }
    } ଅନ୍ୟଥା ଯଦି (vlenb == 32) {
        // vlen = 256 bits, can process 8 float32 elements with m1
        ପାଇଁ (size_t k = 0; k < ଗଣନା_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            ଦଶମିକ *   scale_a_ptr = reinterpret_cast<ଦଶମିକ *>(quant_a_ptr);
            int16_t * a_sum_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + ଆକାର(ଦଶମିକ));
            int8_t *  quant_a_blk =
                reinterpret_cast<int8_t *>(quant_a_ptr + ଆକାର(ଦଶମିକ) + ଆକାର(int16_t) * a_sum_size);

            // Find max absolute value across all 256 elements
            size_t       vl        = __riscv_vsetvl_e32m2(16);
            vfloat32m2_t v_max_abs = __riscv_vfmv_v_f_f32m2(0.0f, vl);

            ପାଇଁ (size_t bki = 0; bki < a_sum_size; bki++) {
                vfloat32m2_t v_a     = __riscv_vle32_v_f32m2(a_ptr + k + bki * 16, vl);
                vfloat32m2_t v_a_abs = __riscv_vfabs_v_f32m2(v_a, vl);
                v_max_abs            = __riscv_vfmax_vv_f32m2(v_a_abs, v_max_abs, vl);
            }
            vfloat32m1_t tmp         = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t v_local_max = __riscv_vfredmax_vs_f32m2_f32m1(v_max_abs, tmp, vl);
            ଦଶମିକ        max_abs_a   = __riscv_vfmv_f_s_f32m1_f32(v_local_max);

            ଦଶମିକ scale_a     = max_abs_a / ((1 << 7) - 1);
            ଦଶମିକ rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
            scale_a_ptr[0]    = scale_a;

            // Quantize and compute sums for each 16-element group
            ପାଇଁ (size_t bki = 0; bki < a_sum_size; bki++) {
                vfloat32m2_t v_a          = __riscv_vle32_v_f32m2(a_ptr + k + bki * 16, vl);
                vfloat32m2_t v_a_scale    = __riscv_vfmul_vf_f32m2(v_a, rep_scale_a, vl);
                vint16m1_t   v_a_quant    = __riscv_vfncvt_x_f_w_i16m1(v_a_scale, vl);
                vint8mf2_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf2(v_a_quant, vl);

                vint16m1_t tmp_sum = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a_sum = __riscv_vwredsum_vs_i8mf2_i16m1(v_a_quant_i8, tmp_sum, vl);
                int16_t    a_sum   = __riscv_vmv_x_s_i16m1_i16(v_a_sum);
                a_sum_ptr[bki]     = -a_sum;

                __riscv_vse8_v_i8mf2(quant_a_blk + bki * 16, v_a_quant_i8, vl);
            }
        }
    } ଅନ୍ୟଥା {
        quantize_a_nrow_i8k_ref<1>(blk_len, a_ptr, ଗଣନା_k, quant_a_ptr);
    }
}

ଶୂନ୍ୟ quantize_a_4row_i8k(size_t blk_len, ଅଚଳ ଦଶମିକ * a_ptr, size_t ଗଣନା_k, uint8_t * quant_a_ptr) {
    GGML_ASSERT(blk_len == 256);
    constexpr int64_t a_blk_stride        = q8k_blk_size(256);
    constexpr int64_t a_nrow_block_stride = a_blk_stride * 4;
    constexpr int64_t a_sum_size          = 256 / 16;
    size_t            vlenb               = __riscv_vlenb();

    ଯଦି (vlenb == 128) {
        // vlen = 1024 bits
        ପାଇଁ (size_t k = 0; k < ଗଣନା_k; k += blk_len, quant_a_ptr += a_nrow_block_stride) {
            ଦଶମିକ *   scale_a_ptr = reinterpret_cast<ଦଶମିକ *>(quant_a_ptr);
            int16_t * a_sum_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + ଆକାର(ଦଶମିକ) * 4);
            int8_t *  quant_a_blk =
                reinterpret_cast<int8_t *>(quant_a_ptr + ଆକାର(ଦଶମିକ) * 4 + ଆକାର(int16_t) * a_sum_size * 4);

            ପାଇଁ (size_t mi = 0; mi < 4; mi++) {
                // Find max absolute value across all 256 elements for this row
                size_t       vl        = __riscv_vsetvl_e32m1(16);
                vfloat32m1_t v_max_abs = __riscv_vfmv_v_f_f32m1(0.0f, vl);

                ପାଇଁ (size_t bki = 0; bki < a_sum_size; bki++) {
                    vfloat32m1_t v_a     = __riscv_vle32_v_f32m1(a_ptr + mi * ଗଣନା_k + k + bki * 16, vl);
                    vfloat32m1_t v_a_abs = __riscv_vfabs_v_f32m1(v_a, vl);
                    v_max_abs            = __riscv_vfmax_vv_f32m1(v_a_abs, v_max_abs, vl);
                }
                vfloat32m1_t tmp         = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_local_max = __riscv_vfredmax_vs_f32m1_f32m1(v_max_abs, tmp, vl);
                ଦଶମିକ        max_abs_a   = __riscv_vfmv_f_s_f32m1_f32(v_local_max);

                ଦଶମିକ scale_a     = max_abs_a / ((1 << 7) - 1);
                ଦଶମିକ rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
                scale_a_ptr[mi]   = scale_a;

                // Quantize and compute sums for each 16-element group
                ପାଇଁ (size_t bki = 0; bki < a_sum_size; bki++) {
                    vfloat32m1_t v_a          = __riscv_vle32_v_f32m1(a_ptr + mi * ଗଣନା_k + k + bki * 16, vl);
                    vfloat32m1_t v_a_scale    = __riscv_vfmul_vf_f32m1(v_a, rep_scale_a, vl);
                    vint16mf2_t  v_a_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a_scale, vl);
                    vint8mf4_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a_quant, vl);

                    vint16m1_t tmp_sum               = __riscv_vmv_v_x_i16m1(0, vl);
                    vint16m1_t v_a_sum               = __riscv_vwredsum_vs_i8mf4_i16m1(v_a_quant_i8, tmp_sum, vl);
                    int16_t    a_sum                 = __riscv_vmv_x_s_i16m1_i16(v_a_sum);
                    a_sum_ptr[mi * a_sum_size + bki] = -a_sum;

                    __riscv_vse8_v_i8mf4(quant_a_blk + mi * blk_len + bki * 16, v_a_quant_i8, vl);
                }
            }
        }
    } ଅନ୍ୟଥା ଯଦି (vlenb == 32) {
        // vlen = 256 bits
        ପାଇଁ (size_t k = 0; k < ଗଣନା_k; k += blk_len, quant_a_ptr += a_nrow_block_stride) {
            ଦଶମିକ *   scale_a_ptr = reinterpret_cast<ଦଶମିକ *>(quant_a_ptr);
            int16_t * a_sum_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + ଆକାର(ଦଶମିକ) * 4);
            int8_t *  quant_a_blk =
                reinterpret_cast<int8_t *>(quant_a_ptr + ଆକାର(ଦଶମିକ) * 4 + ଆକାର(int16_t) * a_sum_size * 4);

            ପାଇଁ (size_t mi = 0; mi < 4; mi++) {
                // Find max absolute value across all 256 elements for this row
                size_t       vl        = __riscv_vsetvl_e32m2(16);
                vfloat32m2_t v_max_abs = __riscv_vfmv_v_f_f32m2(0.0f, vl);

                ପାଇଁ (size_t bki = 0; bki < a_sum_size; bki++) {
                    vfloat32m2_t v_a     = __riscv_vle32_v_f32m2(a_ptr + mi * ଗଣନା_k + k + bki * 16, vl);
                    vfloat32m2_t v_a_abs = __riscv_vfabs_v_f32m2(v_a, vl);
                    v_max_abs            = __riscv_vfmax_vv_f32m2(v_a_abs, v_max_abs, vl);
                }
                vfloat32m1_t tmp         = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_local_max = __riscv_vfredmax_vs_f32m2_f32m1(v_max_abs, tmp, vl);
                ଦଶମିକ        max_abs_a   = __riscv_vfmv_f_s_f32m1_f32(v_local_max);

                ଦଶମିକ scale_a     = max_abs_a / ((1 << 7) - 1);
                ଦଶମିକ rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
                scale_a_ptr[mi]   = scale_a;

                // Quantize and compute sums for each 16-element group
                ପାଇଁ (size_t bki = 0; bki < a_sum_size; bki++) {
                    vfloat32m2_t v_a          = __riscv_vle32_v_f32m2(a_ptr + mi * ଗଣନା_k + k + bki * 16, vl);
                    vfloat32m2_t v_a_scale    = __riscv_vfmul_vf_f32m2(v_a, rep_scale_a, vl);
                    vint16m1_t   v_a_quant    = __riscv_vfncvt_x_f_w_i16m1(v_a_scale, vl);
                    vint8mf2_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf2(v_a_quant, vl);

                    vint16m1_t tmp_sum               = __riscv_vmv_v_x_i16m1(0, vl);
                    vint16m1_t v_a_sum               = __riscv_vwredsum_vs_i8mf2_i16m1(v_a_quant_i8, tmp_sum, vl);
                    int16_t    a_sum                 = __riscv_vmv_x_s_i16m1_i16(v_a_sum);
                    a_sum_ptr[mi * a_sum_size + bki] = -a_sum;

                    __riscv_vse8_v_i8mf2(quant_a_blk + mi * blk_len + bki * 16, v_a_quant_i8, vl);
                }
            }
        }
    } ଅନ୍ୟଥା {
        quantize_a_nrow_i8k_ref<4>(blk_len, a_ptr, ଗଣନା_k, quant_a_ptr);
    }
}

ଶୂନ୍ୟ forward_cpy_with_permute(ggml_compute_params * params, ggml_tensor * op) {
    ଅଚଳ ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା           ith  = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା           nth  = params->nth;

    // [batch, m, n] -> [batch, n, m]
    int64_t batch = src0->ne[2] * src0->ne[3];
    int64_t m     = src0->ne[1];
    int64_t n     = src0->ne[0];

    int64_t batch_stride = src0->nb[2];
    int64_t m_src_stride = src0->nb[0];
    int64_t n_src_stride = src0->nb[1];
    int64_t n_dst_stride = n_src_stride * m;

    permute_transpose_impl(src0, dst, batch, m, n, batch_stride, m_src_stride, n_src_stride, n_dst_stride, ith, nth);
}

ଶୂନ୍ୟ forward_cont_with_permute(ggml_compute_params * params, ggml_tensor * op) {
    ଅଚଳ ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା           ith  = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା           nth  = params->nth;

    // [batch, m, n] -> [batch, n, m]
    int64_t batch = dst->ne[2] * dst->ne[3];
    int64_t n     = dst->ne[1];
    int64_t m     = dst->ne[0];

    int64_t batch_stride = dst->nb[2];
    int64_t m_src_stride = src0->nb[0];
    int64_t n_src_stride = src0->nb[1];
    int64_t n_dst_stride = dst->nb[1];

    permute_transpose_impl(src0, dst, batch, m, n, batch_stride, m_src_stride, n_src_stride, n_dst_stride, ith, nth);
}

ଶୂନ୍ୟ forward_norm_f32(ggml_compute_params * params, ggml_tensor * op) {
    ଅଚଳ ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(src0->nb[0] == ଆକାର(ଦଶମିକ));

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    ଦଶମିକ epsilon = *((ଦଶମିକ *) dst->op_params);

    GGML_ASSERT(epsilon > 0.0f);

    auto * ନିବେଶ  = (ଅକ୍ଷର *) src0->data;
    auto * output = (ଅକ୍ଷର *) dst->data;

    ଅଚଳ auto hidden_size     = ne00;
    ଅଚଳ auto task_ଗଣନା      = ne01 * ne02 * ne03;
    ଅଚଳ auto task_per_thread = (task_ଗଣନା + nth - 1) / nth;

    ଅଚଳ auto task_begin = ith * task_per_thread;
    ଅଚଳ auto task_end   = std::min((ith + 1) * task_per_thread, task_ଗଣନା);

    ପାଇଁ (auto task_idx = task_begin; task_idx < task_end; task_idx++) {
        int64_t i03 = task_idx / (ne02 * ne01);
        int64_t i02 = (task_idx - i03 * ne02 * ne01) / ne01;
        int64_t i01 = (task_idx - i03 * ne02 * ne01 - i02 * ne01);

        auto * p_ନିବେଶ       = (ଦଶମିକ *) (ନିବେଶ + i01 * nb01 + i02 * nb02 + i03 * nb03);
        auto * p_output      = (ଦଶମିକ *) (output + i01 * nb1 + i02 * nb2 + i03 * nb3);
        auto * p_temp_output = p_output;

        size_t       gvl    = __riscv_vsetvlmax_e32m4();
        vfloat32m4_t sum    = __riscv_vfmv_v_f_f32m4(0.f, gvl);
        vfloat32m4_t sum_sq = __riscv_vfmv_v_f_f32m4(0.f, gvl);
        int64_t      length = hidden_size;
        ଯେତେବେଳେ (length > 0) {
            gvl                   = __riscv_vsetvl_e32m4(length);
            // load data
            vfloat32m4_t src_data = __riscv_vle32_v_f32m4(p_ନିବେଶ, gvl);

            sum    = __riscv_vfadd_vv_f32m4(sum, src_data, gvl);
            sum_sq = __riscv_vfmacc_vv_f32m4(sum_sq, src_data, src_data, gvl);

            __riscv_vse32_v_f32m4(p_temp_output, src_data, gvl);

            p_ନିବେଶ += gvl;
            p_temp_output += gvl;
            length -= gvl;
        }

        gvl = __riscv_vsetvlmax_e32m1();

        ଦଶମିକ        mean   = 0.f;
        vfloat32m1_t zero_v = __riscv_vfmv_v_f_f32m1(0.f, gvl);
        vfloat32m1_t mean_v =
            __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(sum, 0), __riscv_vget_v_f32m4_f32m1(sum, 1), gvl);
        mean_v = __riscv_vfadd_vv_f32m1(mean_v, __riscv_vget_v_f32m4_f32m1(sum, 2), gvl);
        mean_v = __riscv_vfadd_vv_f32m1(mean_v, __riscv_vget_v_f32m4_f32m1(sum, 3), gvl);
        mean_v = __riscv_vfredusum_vs_f32m1_f32m1(mean_v, zero_v, gvl);
        mean   = __riscv_vfmv_f_s_f32m1_f32(mean_v);
        mean /= hidden_size;

        vfloat32m1_t mean_square_v =
            __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(sum_sq, 0), __riscv_vget_v_f32m4_f32m1(sum_sq, 1), gvl);
        mean_square_v = __riscv_vfadd_vv_f32m1(mean_square_v, __riscv_vget_v_f32m4_f32m1(sum_sq, 2), gvl);
        mean_square_v = __riscv_vfadd_vv_f32m1(mean_square_v, __riscv_vget_v_f32m4_f32m1(sum_sq, 3), gvl);
        mean_square_v = __riscv_vfredusum_vs_f32m1_f32m1(mean_square_v, zero_v, gvl);

        ଦଶମିକ mean_square = __riscv_vfmv_f_s_f32m1_f32(mean_square_v);
        mean_square /= hidden_size;
        mean_square = sqrt(mean_square - mean * mean + epsilon);

        mean_square   = 1.0f / mean_square;
        length        = hidden_size;
        p_temp_output = p_output;

        ଯେତେବେଳେ (length > 0) {
            gvl                   = __riscv_vsetvl_e32m4(length);
            vfloat32m4_t src_data = __riscv_vle32_v_f32m4(p_temp_output, gvl);
            src_data              = __riscv_vfsub_vf_f32m4(src_data, mean, gvl);
            src_data              = __riscv_vfmul_vf_f32m4(src_data, mean_square, gvl);
            __riscv_vse32_v_f32m4(p_output, src_data, gvl);
            p_temp_output += gvl;
            p_output += gvl;
            length -= gvl;
        }
    }
}

template <ggml_op op_type, typename T> ଶୂନ୍ୟ forward_ଦ୍ୱିମିକ(ggml_compute_params * params, ggml_tensor * op) {
    ଅଚଳ ggml_tensor * src0 = op->src[0];
    ଅଚଳ ggml_tensor * src1 = op->src[1];
    ggml_tensor *       dst  = op;
    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    auto src0_rows = ggml_nrows(src0);
    auto src1_rows = ggml_nrows(src1);

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS

    GGML_ASSERT(nb0 == ଆକାର(T));
    GGML_ASSERT(nb00 == ଆକାର(T));

    ଅଚଳ auto [ir0, ir1] = get_thread_range(params, src0);

    auto compute_func_vv = [&](int64_t blk_len, int64_t r, T * src0_ptr, T * src1_ptr, T * dst_ptr) {
        int64_t idx = 0;
        ଯଦି constexpr (op_type == GGML_OP_ADD) {
            ଯଦି constexpr (std::is_same_v<T, ଦଶମିକ>) {
                ପାଇଁ (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e32m4(blk_len);
                    vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + idx + r, vl);
                    vfloat32m4_t rhs = __riscv_vle32_v_f32m4(src1_ptr + idx, vl);
                    vfloat32m4_t res = __riscv_vfadd_vv_f32m4(lhs, rhs, vl);
                    __riscv_vse32_v_f32m4(dst_ptr + idx + r, res, vl);
                }
            } ଅନ୍ୟଥା ଯଦି constexpr (std::is_same_v<T, _Float16>) {
                ପାଇଁ (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e16m4(blk_len);
                    vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + idx + r), vl);
                    vfloat16m4_t rhs = __riscv_vle16_v_f16m4((src1_ptr + idx), vl);
                    vfloat16m4_t res = __riscv_vfadd_vv_f16m4(lhs, rhs, vl);
                    __riscv_vse16_v_f16m4((dst_ptr + idx + r), res, vl);
                }
            } ଅନ୍ୟଥା {
                GGML_ABORT("fatal error");
            }
        } ଅନ୍ୟଥା ଯଦି constexpr (op_type == GGML_OP_SUB) {
            ଯଦି constexpr (std::is_same_v<T, ଦଶମିକ>) {
                ପାଇଁ (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e32m4(blk_len);
                    vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + idx + r, vl);
                    vfloat32m4_t rhs = __riscv_vle32_v_f32m4(src1_ptr + idx, vl);
                    vfloat32m4_t res = __riscv_vfsub_vv_f32m4(lhs, rhs, vl);
                    __riscv_vse32_v_f32m4(dst_ptr + idx + r, res, vl);
                }
            } ଅନ୍ୟଥା ଯଦି constexpr (std::is_same_v<T, _Float16>) {
                ପାଇଁ (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e16m4(blk_len);
                    vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + idx + r), vl);
                    vfloat16m4_t rhs = __riscv_vle16_v_f16m4((src1_ptr + idx), vl);
                    vfloat16m4_t res = __riscv_vfsub_vv_f16m4(lhs, rhs, vl);
                    __riscv_vse16_v_f16m4((dst_ptr + idx + r), res, vl);
                }
            } ଅନ୍ୟଥା {
                GGML_ABORT("fatal error");
            }
        } ଅନ୍ୟଥା ଯଦି constexpr (op_type == GGML_OP_MUL) {
            ଯଦି constexpr (std::is_same_v<T, ଦଶମିକ>) {
                ପାଇଁ (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e32m4(blk_len);
                    vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + idx + r, vl);
                    vfloat32m4_t rhs = __riscv_vle32_v_f32m4(src1_ptr + idx, vl);
                    vfloat32m4_t res = __riscv_vfmul_vv_f32m4(lhs, rhs, vl);
                    __riscv_vse32_v_f32m4(dst_ptr + idx + r, res, vl);
                }
            } ଅନ୍ୟଥା ଯଦି constexpr (std::is_same_v<T, _Float16>) {
                ପାଇଁ (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e16m4(blk_len);
                    vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + idx + r), vl);
                    vfloat16m4_t rhs = __riscv_vle16_v_f16m4((src1_ptr + idx), vl);
                    vfloat16m4_t res = __riscv_vfmul_vv_f16m4(lhs, rhs, vl);
                    __riscv_vse16_v_f16m4((dst_ptr + idx + r), res, vl);
                }
            } ଅନ୍ୟଥା {
                GGML_ABORT("fatal error");
            }
        } ଅନ୍ୟଥା ଯଦି constexpr (op_type == GGML_OP_DIV) {
            ଯଦି constexpr (std::is_same_v<T, ଦଶମିକ>) {
                ପାଇଁ (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e32m4(blk_len);
                    vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + idx + r, vl);
                    vfloat32m4_t rhs = __riscv_vle32_v_f32m4(src1_ptr + idx, vl);
                    vfloat32m4_t res = __riscv_vfdiv_vv_f32m4(lhs, rhs, vl);
                    __riscv_vse32_v_f32m4(dst_ptr + idx + r, res, vl);
                }
            } ଅନ୍ୟଥା ଯଦି constexpr (std::is_same_v<T, _Float16>) {
                ପାଇଁ (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e16m4(blk_len);
                    vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + idx + r), vl);
                    vfloat16m4_t rhs = __riscv_vle16_v_f16m4((src1_ptr + idx), vl);
                    vfloat16m4_t res = __riscv_vfdiv_vv_f16m4(lhs, rhs, vl);
                    __riscv_vse16_v_f16m4((dst_ptr + idx + r), res, vl);
                }
            } ଅନ୍ୟଥା {
                GGML_ABORT("fatal error");
            }
        } ଅନ୍ୟଥା {
            GGML_ABORT("fatal error");
        }
    };

    ଯଦି (src0_rows == src1_rows && src0_rows == 1 && ne00 == ne10) {
        int64_t task_per_thread = (ne00 + nth - 1) / nth;
        int64_t task_begin      = ith * task_per_thread;
        int64_t task_end        = std::min((ith + 1) * task_per_thread, ne00);

        T * dst_ptr  = ((T *) dst->data) + task_begin;
        T * src0_ptr = ((T *) src0->data) + task_begin;
        T * src1_ptr = ((T *) src1->data) + task_begin;

        compute_func_vv(task_end - task_begin, 0, src0_ptr, src1_ptr, dst_ptr);
    } ଅନ୍ୟଥା ଯଦି (ne10 > 1) {
        ପାଇଁ (int64_t ir = ir0; ir < ir1; ++ir) {
            ଅଚଳ int64_t i03 = ir / (ne02 * ne01);
            ଅଚଳ int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            ଅଚଳ int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

            ଅଚଳ int64_t i13 = i03 % ne13;
            ଅଚଳ int64_t i12 = i02 % ne12;
            ଅଚଳ int64_t i11 = i01 % ne11;

            T * dst_ptr  = (T *) ((ଅକ୍ଷର *) dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1);
            T * src0_ptr = (T *) ((ଅକ୍ଷର *) src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01);
            T * src1_ptr = (T *) ((ଅକ୍ଷର *) src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11);

            // src1 is broadcastable across src0 and dst in i1, i2, i3
            ପାଇଁ (int64_t r = 0; r < ne00; r += ne10) {
                compute_func_vv(ne10, r, src0_ptr, src1_ptr, dst_ptr);
            }
        }
    } ଅନ୍ୟଥା {
        ପାଇଁ (int64_t ir = ir0; ir < ir1; ++ir) {
            ଅଚଳ int64_t i03 = ir / (ne02 * ne01);
            ଅଚଳ int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            ଅଚଳ int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

            ଅଚଳ int64_t i13 = i03 % ne13;
            ଅଚଳ int64_t i12 = i02 % ne12;
            ଅଚଳ int64_t i11 = i01 % ne11;

            T * dst_ptr  = (T *) ((ଅକ୍ଷର *) dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1);
            T * src0_ptr = (T *) ((ଅକ୍ଷର *) src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01);
            T * src1_ptr = (T *) ((ଅକ୍ଷର *) src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11);

            T       rhs_scalar = src1_ptr[0];
            int64_t blk_len    = ne00;
            int64_t r          = 0;

            ପାଇଁ (size_t vl; blk_len > 0; blk_len -= vl, r += vl) {
                ଯଦି constexpr (op_type == GGML_OP_ADD) {
                    ଯଦି constexpr (std::is_same_v<T, ଦଶମିକ>) {
                        vl               = __riscv_vsetvl_e32m4(blk_len);
                        vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + r, vl);
                        vfloat32m4_t res = __riscv_vfadd_vf_f32m4(lhs, rhs_scalar, vl);
                        __riscv_vse32_v_f32m4(dst_ptr + r, res, vl);
                    } ଅନ୍ୟଥା ଯଦି constexpr (std::is_same_v<T, _Float16>) {
                        vl               = __riscv_vsetvl_e16m4(blk_len);
                        vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + r), vl);
                        vfloat16m4_t res = __riscv_vfadd_vf_f16m4(lhs, rhs_scalar, vl);
                        __riscv_vse16_v_f16m4((dst_ptr + r), res, vl);
                    } ଅନ୍ୟଥା {
                        GGML_ABORT("fatal error");
                    }
                } ଅନ୍ୟଥା ଯଦି constexpr (op_type == GGML_OP_SUB) {
                    ଯଦି constexpr (std::is_same_v<T, ଦଶମିକ>) {
                        vl               = __riscv_vsetvl_e32m4(blk_len);
                        vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + r, vl);
                        vfloat32m4_t res = __riscv_vfsub_vf_f32m4(lhs, rhs_scalar, vl);
                        __riscv_vse32_v_f32m4(dst_ptr + r, res, vl);
                    } ଅନ୍ୟଥା ଯଦି constexpr (std::is_same_v<T, _Float16>) {
                        vl               = __riscv_vsetvl_e16m4(blk_len);
                        vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + r), vl);
                        vfloat16m4_t res = __riscv_vfsub_vf_f16m4(lhs, rhs_scalar, vl);
                        __riscv_vse16_v_f16m4((dst_ptr + r), res, vl);
                    } ଅନ୍ୟଥା {
                        GGML_ABORT("fatal error");
                    }
                } ଅନ୍ୟଥା ଯଦି constexpr (op_type == GGML_OP_MUL) {
                    ଯଦି constexpr (std::is_same_v<T, ଦଶମିକ>) {
                        vl               = __riscv_vsetvl_e32m4(blk_len);
                        vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + r, vl);
                        vfloat32m4_t res = __riscv_vfmul_vf_f32m4(lhs, rhs_scalar, vl);
                        __riscv_vse32_v_f32m4(dst_ptr + r, res, vl);
                    } ଅନ୍ୟଥା ଯଦି constexpr (std::is_same_v<T, _Float16>) {
                        vl               = __riscv_vsetvl_e16m4(blk_len);
                        vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + r), vl);
                        vfloat16m4_t res = __riscv_vfmul_vf_f16m4(lhs, rhs_scalar, vl);
                        __riscv_vse16_v_f16m4((dst_ptr + r), res, vl);
                    } ଅନ୍ୟଥା {
                        GGML_ABORT("fatal error");
                    }
                } ଅନ୍ୟଥା ଯଦି constexpr (op_type == GGML_OP_DIV) {
                    ଯଦି constexpr (std::is_same_v<T, ଦଶମିକ>) {
                        vl               = __riscv_vsetvl_e32m4(blk_len);
                        vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + r, vl);
                        vfloat32m4_t res = __riscv_vfdiv_vf_f32m4(lhs, rhs_scalar, vl);
                        __riscv_vse32_v_f32m4(dst_ptr + r, res, vl);
                    } ଅନ୍ୟଥା ଯଦି constexpr (std::is_same_v<T, _Float16>) {
                        vl               = __riscv_vsetvl_e16m4(blk_len);
                        vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + r), vl);
                        vfloat16m4_t res = __riscv_vfdiv_vf_f16m4(lhs, rhs_scalar, vl);
                        __riscv_vse16_v_f16m4((dst_ptr + r), res, vl);
                    } ଅନ୍ୟଥା {
                        GGML_ABORT("fatal error");
                    }
                } ଅନ୍ୟଥା {
                    GGML_ABORT("fatal error");
                }
            }
        }
    }
}

template <typename T> ଶୂନ୍ୟ forward_sum_rows(ଅଚଳ ggml_compute_params * params, ggml_tensor * op) {
    ଅଚଳ ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne0 == 1);
    GGML_ASSERT(ne1 == ne01);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    int64_t n_task          = ne01 * ne02 * ne03;
    int64_t task_per_thread = (n_task + nth - 1) / nth;
    int64_t ir_start        = ith * task_per_thread;
    int64_t ir_end          = std::min(ir_start + task_per_thread, n_task);

    ପାଇଁ (int64_t ir = ir_start; ir < ir_end; ir++) {
        ଅଚଳ int64_t i3 = ir / (ne02 * ne01);
        ଅଚଳ int64_t i2 = (ir - i3 * ne02 * ne01) / ne01;
        ଅଚଳ int64_t i1 = (ir - i3 * ne02 * ne01 - i2 * ne01);

        T * src_row = (T *) ((ଅକ୍ଷର *) src0->data + i1 * nb01 + i2 * nb02 + i3 * nb03);
        T * dst_row = (T *) ((ଅକ୍ଷର *) op->data + i1 * nb1 + i2 * nb2 + i3 * nb3);

        ଦଶମିକ row_sum = 0;

        ଯଦି constexpr (std::is_same_v<T, ଦଶମିକ>) {
            size_t        gvl     = __riscv_vsetvlmax_e32m4();
            vfloat32m4_t  acc_vec = __riscv_vfmv_v_f_f32m4(0.0f, gvl);
            int64_t       length  = ne00;
            ଅଚଳ ଦଶମିକ * p_data  = src_row;

            ଯେତେବେଳେ (length > 0) {
                size_t       vl  = __riscv_vsetvl_e32m4(length);
                vfloat32m4_t vec = __riscv_vle32_v_f32m4(p_data, vl);
                acc_vec          = __riscv_vfadd_vv_f32m4(acc_vec, vec, vl);
                p_data += vl;
                length -= vl;
            }

            gvl                 = __riscv_vsetvlmax_e32m1();
            vfloat32m1_t zero_v = __riscv_vfmv_v_f_f32m1(0.0f, gvl);
            vfloat32m1_t sum_v  = __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(acc_vec, 0),
                                                         __riscv_vget_v_f32m4_f32m1(acc_vec, 1), gvl);
            sum_v               = __riscv_vfadd_vv_f32m1(sum_v, __riscv_vget_v_f32m4_f32m1(acc_vec, 2), gvl);
            sum_v               = __riscv_vfadd_vv_f32m1(sum_v, __riscv_vget_v_f32m4_f32m1(acc_vec, 3), gvl);
            sum_v               = __riscv_vfredusum_vs_f32m1_f32m1(sum_v, zero_v, gvl);
            row_sum             = __riscv_vfmv_f_s_f32m1_f32(sum_v);
        } ଅନ୍ୟଥା ଯଦି constexpr (std::is_same_v<T, _Float16>) {
            size_t           gvl     = __riscv_vsetvlmax_e16m2();
            vfloat32m4_t     acc_vec = __riscv_vfmv_v_f_f32m4(0.0f, gvl);
            int64_t          length  = ne00;
            ଅଚଳ _Float16 * p_data  = src_row;

            ଯେତେବେଳେ (length > 0) {
                size_t       vl      = __riscv_vsetvl_e16m2(length);
                vfloat16m2_t vec_f16 = __riscv_vle16_v_f16m2(p_data, vl);
                vfloat32m4_t vec_f32 = __riscv_vfwcvt_f_f_v_f32m4(vec_f16, vl);
                acc_vec              = __riscv_vfadd_vv_f32m4(acc_vec, vec_f32, vl);
                p_data += vl;
                length -= vl;
            }

            gvl                 = __riscv_vsetvlmax_e32m1();
            vfloat32m1_t zero_v = __riscv_vfmv_v_f_f32m1(0.0f, gvl);
            vfloat32m1_t sum_v  = __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(acc_vec, 0),
                                                         __riscv_vget_v_f32m4_f32m1(acc_vec, 1), gvl);
            sum_v               = __riscv_vfadd_vv_f32m1(sum_v, __riscv_vget_v_f32m4_f32m1(acc_vec, 2), gvl);
            sum_v               = __riscv_vfadd_vv_f32m1(sum_v, __riscv_vget_v_f32m4_f32m1(acc_vec, 3), gvl);
            sum_v               = __riscv_vfredusum_vs_f32m1_f32m1(sum_v, zero_v, gvl);
            row_sum             = __riscv_vfmv_f_s_f32m1_f32(sum_v);
        } ଅନ୍ୟଥା {
            GGML_ABORT("fatal error");
        }

        dst_row[0] = row_sum;
    }
}

template <typename T> ଶୂନ୍ୟ forward_repeat_nrows(ggml_compute_params * params, ggml_tensor * op) {
    ଅଚଳ ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    int64_t nrows            = ggml_nrows(src0);
    int64_t nrows_per_thread = (nrows + nth - 1) / nth;
    int64_t ir_start         = ith * nrows_per_thread;
    int64_t ir_end           = std::min(ir_start + nrows_per_thread, nrows);

    ଯଦି (src0->ne[0] == 1) {
        ପାଇଁ (int64_t ir = ir_start; ir < ir_end; ir++) {
            T * src_row = (T *) ((ଅକ୍ଷର *) src0->data + ir * src0->nb[1]);
            T * dst_row = (T *) ((ଅକ୍ଷର *) dst->data + ir * dst->nb[1]);

            T src_scalar = src_row[0];

            int64_t length = dst->ne[0];
            int64_t idx    = 0;
            size_t  vl     = 0;

            ଯେତେବେଳେ (length > 0) {
                ଯଦି constexpr (std::is_same_v<T, int32_t>) {
                    vl             = __riscv_vsetvl_e32m4(length);
                    vint32m4_t vec = __riscv_vmv_v_x_i32m4(src_scalar, vl);
                    __riscv_vse32_v_i32m4(dst_row + idx, vec, vl);
                } ଅନ୍ୟଥା ଯଦି constexpr (std::is_same_v<T, int16_t>) {
                    vl             = __riscv_vsetvl_e16m4(length);
                    vint16m4_t vec = __riscv_vmv_v_x_i16m4(src_scalar, vl);
                    __riscv_vse16_v_i16m4((dst_row + idx), vec, vl);
                } ଅନ୍ୟଥା {
                    GGML_ABORT("fatal error");
                }
                idx += vl;
                length -= vl;
            }
        }
    } ଅନ୍ୟଥା ଯଦି (src0->ne[0] == dst->ne[0]) {
        ପାଇଁ (int64_t ir = ir_start; ir < ir_end; ir++) {
            T * src_row = (T *) ((ଅକ୍ଷର *) src0->data + ir * src0->nb[1]);
            T * dst_row = (T *) ((ଅକ୍ଷର *) dst->data + ir * dst->nb[1]);

            int64_t length = dst->ne[0];
            int64_t idx    = 0;
            size_t  vl     = 0;

            ଯେତେବେଳେ (length > 0) {
                ଯଦି constexpr (std::is_same_v<T, int32_t>) {
                    vl             = __riscv_vsetvl_e32m4(length);
                    vint32m4_t vec = __riscv_vle32_v_i32m4(src_row + idx, vl);
                    __riscv_vse32_v_i32m4(dst_row + idx, vec, vl);
                } ଅନ୍ୟଥା ଯଦି constexpr (std::is_same_v<T, int16_t>) {
                    vl             = __riscv_vsetvl_e16m4(length);
                    vint16m4_t vec = __riscv_vle16_v_i16m4((src_row + idx), vl);
                    __riscv_vse16_v_i16m4((dst_row + idx), vec, vl);
                } ଅନ୍ୟଥା {
                    GGML_ABORT("fatal error");
                }
                idx += vl;
                length -= vl;
            }
        }
    } ଅନ୍ୟଥା {
        GGML_ABORT("fatal error");
    }
}

template <typename T> ଶୂନ୍ୟ forward_repeat_dim1(ggml_compute_params * params, ggml_tensor * op) {
    ଅଚଳ ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ int64_t ne0 = dst->ne[0];
    ଅଚଳ int64_t ne1 = dst->ne[1];
    ଅଚଳ int64_t ne2 = dst->ne[2];
    ଅଚଳ int64_t ne3 = dst->ne[3];

    ଅଚଳ int64_t total_batches      = ne2 * ne3;
    ଅଚଳ int64_t batches_per_thread = (total_batches + nth - 1) / nth;
    ଅଚଳ int64_t batch_start        = ith * batches_per_thread;
    ଅଚଳ int64_t batch_end          = std::min(batch_start + batches_per_thread, total_batches);

    ପାଇଁ (int64_t b = batch_start; b < batch_end; b++) {
        ଅଚଳ int64_t i3 = b / ne2;
        ଅଚଳ int64_t i2 = b % ne2;

        T * src_base  = (T *) ((ଅକ୍ଷର *) src0->data + i2 * src0->nb[2] + i3 * src0->nb[3]);
        T * dst_batch = (T *) ((ଅକ୍ଷର *) dst->data + i2 * dst->nb[2] + i3 * dst->nb[3]);

        ପାଇଁ (int64_t i1 = 0; i1 < ne1; i1++) {
            T *     dst_ptr = (T *) ((ଅକ୍ଷର *) dst_batch + i1 * dst->nb[1]);
            int64_t length  = ne0;
            int64_t idx     = 0;

            ଯେତେବେଳେ (length > 0) {
                ଯଦି constexpr (std::is_same_v<T, int32_t>) {
                    size_t     vl  = __riscv_vsetvl_e32m4(length);
                    vint32m4_t vec = __riscv_vle32_v_i32m4(src_base + idx, vl);
                    __riscv_vse32_v_i32m4(dst_ptr + idx, vec, vl);
                    idx += vl;
                    length -= vl;
                } ଅନ୍ୟଥା ଯଦି constexpr (std::is_same_v<T, int16_t>) {
                    size_t     vl  = __riscv_vsetvl_e16m4(length);
                    vint16m4_t vec = __riscv_vle16_v_i16m4((src_base + idx), vl);
                    __riscv_vse16_v_i16m4((dst_ptr + idx), vec, vl);
                    idx += vl;
                    length -= vl;
                } ଅନ୍ୟଥା {
                    GGML_ABORT("fatal error");
                }
            }
        }
    }
}

template <typename T> ଶୂନ୍ୟ forward_get_rows(ggml_compute_params * params, ggml_tensor * op) {
    ଅଚଳ ggml_tensor * src0 = op->src[0];
    ଅଚଳ ggml_tensor * src1 = op->src[1];
    ggml_tensor *       dst  = op;

    GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS

    ଅଚଳ int64_t nc = ne00;
    ଅଚଳ int64_t nr = ggml_nelements(src1);

    ପ୍ରତିଜ୍ଞା(ne0 == nc);
    ପ୍ରତିଜ୍ଞା(ne02 == ne11);
    ପ୍ରତିଜ୍ଞା(nb00 == ଆକାର(ଦଶମିକ));
    ପ୍ରତିଜ୍ଞା(ggml_nrows(op) == nr);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା rows_nth = nth;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା cols_nth = 1;

    ଯଦି (nr == 1) {
        rows_nth = 1;
        cols_nth = nth;
    }

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + rows_nth - 1) / rows_nth;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dc = (nc + cols_nth - 1) / cols_nth;

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା rows_ith = ith % rows_nth;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା cols_ith = ith % cols_nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr * rows_ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା cr0 = dc * cols_ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା cr1 = MIN(cr0 + dc, nc);

    ପାଇଁ (int64_t i = ir0; i < ir1; ++i) {
        ଅଚଳ int64_t i12 = i / (ne11 * ne10);
        ଅଚଳ int64_t i11 = (i - i12 * ne11 * ne10) / ne10;
        ଅଚଳ int64_t i10 = (i - i12 * ne11 * ne10 - i11 * ne10);
        ଅଚଳ int64_t i01 = *(int32_t *) ((ଅକ୍ଷର *) src1->data + i10 * nb10 + i11 * nb11 + i12 * nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        memcpy1d(((ଅକ୍ଷର *) dst->data + i10 * nb1 + i11 * nb2 + i12 * nb3) + cr0 * ଆକାର(T),
                 ((ଅକ୍ଷର *) src0->data + i01 * nb01 + i11 * nb02 + i12 * nb03) + cr0 * ଆକାର(T),
                 (cr1 - cr0) * ଆକାର(T));
    }
}

template <typename T> ଶୂନ୍ୟ forward_concat(ggml_compute_params * params, ggml_tensor * op) {
    ଅଚଳ ggml_tensor * src0 = op->src[0];
    ଅଚଳ ggml_tensor * src1 = op->src[1];
    ggml_tensor *       dst  = op;

    GGML_ASSERT(ggml_type_size(src0->type) == ଆକାର(ଦଶମିକ));

    GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS

    ଅଚଳ int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim == 0 && nb0 == ଆକାର(ଦଶମିକ) && nb1 == ଆକାର(ଦଶମିକ) * (ne00 + ne10));

    ଅଚଳ int64_t nr = ggml_nrows(dst);
    ଅଚଳ int64_t nc = ne0;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା rows_nth = nth;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା cols_nth = 1;

    ଯଦି (nr == 1) {
        rows_nth = 1;
        cols_nth = nth;
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + rows_nth - 1) / rows_nth;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dc = (nc + cols_nth - 1) / cols_nth;

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା rows_ith = ith % rows_nth;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା cols_ith = ith % cols_nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr * rows_ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା cr0 = dc * cols_ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା cr1 = MIN(cr0 + dc, nc);

    int64_t o[4] = { 0, 0, 0, 0 };
    o[dim]       = src0->ne[dim];
    ଅଚଳ ଦଶମିକ * x;

    ପାଇଁ (int64_t i = ir0; i < ir1; ++i) {
        ଅଚଳ int64_t i3 = i / (ne02 * ne01);
        ଅଚଳ int64_t i2 = (i - i3 * ne02 * ne01) / ne01;
        ଅଚଳ int64_t i1 = (i - i3 * ne02 * ne01 - i2 * ne01);

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0 = cr0; i0 < cr1; i0++) {
            ଯଦି (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                x = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src0->data + (i0) *nb00 + (i1) *nb01 + (i2) *nb02 + (i3) *nb03);
            } ଅନ୍ୟଥା {
                x = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src1->data + (i0 - o[0]) * nb10 + (i1 - o[1]) * nb11 +
                                     (i2 - o[2]) * nb12 + (i3 - o[3]) * nb13);
            }

            ଦଶମିକ * y = (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + i0 * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3);

            *y = *x;
        }
    }
}

template ଶୂନ୍ୟ forward_ଦ୍ୱିମିକ<GGML_OP_ADD, ଦଶମିକ>(ggml_compute_params * params, ggml_tensor * op);
template ଶୂନ୍ୟ forward_ଦ୍ୱିମିକ<GGML_OP_SUB, ଦଶମିକ>(ggml_compute_params * params, ggml_tensor * op);
template ଶୂନ୍ୟ forward_ଦ୍ୱିମିକ<GGML_OP_MUL, ଦଶମିକ>(ggml_compute_params * params, ggml_tensor * op);
template ଶୂନ୍ୟ forward_ଦ୍ୱିମିକ<GGML_OP_DIV, ଦଶମିକ>(ggml_compute_params * params, ggml_tensor * op);
template ଶୂନ୍ୟ forward_ଦ୍ୱିମିକ<GGML_OP_ADD, _Float16>(ggml_compute_params * params, ggml_tensor * op);
template ଶୂନ୍ୟ forward_ଦ୍ୱିମିକ<GGML_OP_SUB, _Float16>(ggml_compute_params * params, ggml_tensor * op);
template ଶୂନ୍ୟ forward_ଦ୍ୱିମିକ<GGML_OP_MUL, _Float16>(ggml_compute_params * params, ggml_tensor * op);
template ଶୂନ୍ୟ forward_ଦ୍ୱିମିକ<GGML_OP_DIV, _Float16>(ggml_compute_params * params, ggml_tensor * op);
template ଶୂନ୍ୟ forward_sum_rows<ଦଶମିକ>(ଅଚଳ ggml_compute_params * params, ggml_tensor * op);
template ଶୂନ୍ୟ forward_sum_rows<_Float16>(ଅଚଳ ggml_compute_params * params, ggml_tensor * op);
template ଶୂନ୍ୟ forward_repeat_nrows<int32_t>(ggml_compute_params * params, ggml_tensor * op);
template ଶୂନ୍ୟ forward_repeat_nrows<int16_t>(ggml_compute_params * params, ggml_tensor * op);
template ଶୂନ୍ୟ forward_repeat_dim1<int32_t>(ggml_compute_params * params, ggml_tensor * op);
template ଶୂନ୍ୟ forward_repeat_dim1<int16_t>(ggml_compute_params * params, ggml_tensor * op);
template ଶୂନ୍ୟ forward_get_rows<int32_t>(ggml_compute_params * params, ggml_tensor * op);
template ଶୂନ୍ୟ forward_get_rows<int16_t>(ggml_compute_params * params, ggml_tensor * op);
template ଶୂନ୍ୟ forward_concat<int32_t>(ggml_compute_params * params, ggml_tensor * op);
template ଶୂନ୍ୟ forward_concat<int16_t>(ggml_compute_params * params, ggml_tensor * op);

}  // namespace spacemit_kernels::rvv
