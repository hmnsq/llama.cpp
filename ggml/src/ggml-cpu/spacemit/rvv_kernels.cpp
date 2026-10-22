#অন্তর্ভাব "rvv_kernels.h"

#অন্তর্ভাব "common.h"
#অন্তর্ভাব "ggml.h"
#অন্তর্ভাব "ops.h"
#অন্তর্ভাব "string.h"

#অন্তর্ভাব <algorithm>
#অন্তর্ভাব <cmath>
#অন্তর্ভাব <cstdint>
#অন্তর্ভাব <stdexcept>

#যদি !defined(__riscv_v) || !defined(__riscv_v_intrinsic)
#    error "riscv v extension or v_intrinsic not enabled"
#অন্যথা
#    include <riscv_vector.h>
#endif

#যদি !defined(__riscv_zfh)
#    error "riscv zfh extension not enabled"
#endif

#যদি defined(__GNUC__)
#    pragma GCC diagnostic ignored "-Woverlength-strings"
#    pragma GCC diagnostic ignored "-Wcast-qual"
#    pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

নামাকাশ spacemit_kernels::rvv {

নামাকাশ {

auto align_up(size_t value, size_t alignment) {
    ফেরত (value + alignment - 1) / alignment * alignment;
}

স্থির অন্তঃরেখ যৌক্তিক flash_attn_ext_supported_d_vlen1024_vf16(int64_t d) {
    ফেরত d > 0 && d <= 128;
}

স্থির অন্তঃরেখ যৌক্তিক flash_attn_ext_supported_shape_vlen1024_vf16(int64_t DK, int64_t DV) {
    ফেরত flash_attn_ext_supported_d_vlen1024_vf16(DK) && flash_attn_ext_supported_d_vlen1024_vf16(DV);
}

স্থির অন্তঃরেখ দশমিক reduce_যোগ_f32m4_vlen1024(vfloat32m4_t v, size_t vl) {
    vfloat32m1_t s_v = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    s_v              = __riscv_vfredusum_vs_f32m4_f32m1(v, s_v, vl);
    ফেরত __riscv_vfmv_f_s_f32m1_f32(s_v);
}

স্থির অন্তঃরেখ দশমিক reduce_যোগ_f32m2_vlen1024(vfloat32m2_t v, size_t vl) {
    vfloat32m1_t s_v = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    s_v              = __riscv_vfredusum_vs_f32m2_f32m1(v, s_v, vl);
    ফেরত __riscv_vfmv_f_s_f32m1_f32(s_v);
}

// Adapted from ggml_v_expf_m2 in vec.h. This is accurate enough for softmax.
স্থির অন্তঃরেখ vfloat32m2_t rvv_expf_approx_f32m2(vfloat32m2_t x, size_t vl) {
    ধ্রুবক vfloat32m2_t r = __riscv_vfmv_v_f_f32m2(0x1.8p23f, vl);
    ধ্রুবক vfloat32m2_t z = __riscv_vfmacc_vf_f32m2(r, 0x1.715476p+0f, x, vl);
    ধ্রুবক vfloat32m2_t n = __riscv_vfsub_vv_f32m2(z, r, vl);
    ধ্রুবক vfloat32m2_t b =
        __riscv_vfnmsac_vf_f32m2(__riscv_vfnmsac_vf_f32m2(x, 0x1.62e4p-1f, n, vl), 0x1.7f7d1cp-20f, n, vl);
    ধ্রুবক vuint32m2_t  e = __riscv_vsll_vx_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(z), 23, vl);
    ধ্রুবক vfloat32m2_t k = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vadd_vx_u32m2(e, 0x3f800000, vl));
    ধ্রুবক vbool16_t    c = __riscv_vmfgt_vf_f32m2_b16(__riscv_vfabs_v_f32m2(n, vl), 126.0f, vl);
    ধ্রুবক vfloat32m2_t u = __riscv_vfmul_vv_f32m2(b, b, vl);
    ধ্রুবক vfloat32m2_t j = __riscv_vfmacc_vv_f32m2(
        __riscv_vfmul_vf_f32m2(b, 0x1.ffffecp-1f, vl),
        __riscv_vfmacc_vv_f32m2(
            __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(0x1.fffdb6p-2f, vl), 0x1.555e66p-3f, b, vl),
            __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(0x1.573e2ep-5f, vl), 0x1.0e4020p-7f, b, vl), u, vl),
        u, vl);

    যদি (!__riscv_vcpop_m_b16(c, vl)) {
        ফেরত __riscv_vfmacc_vv_f32m2(k, j, k, vl);
    }

    ধ্রুবক vbool16_t    dm = __riscv_vmfle_vf_f32m2_b16(n, 0.0f, vl);
    ধ্রুবক vuint32m2_t  d  = __riscv_vmerge_vxm_u32m2(__riscv_vmv_v_x_u32m2(0, vl), 0x82000000, dm, vl);
    ধ্রুবক vfloat32m2_t s1 = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vadd_vx_u32m2(d, 0x7f000000, vl));
    ধ্রুবক vfloat32m2_t s2 = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vsub_vv_u32m2(e, d, vl));
    ধ্রুবক vfloat32m2_t r1 =
        __riscv_vmerge_vvm_f32m2(__riscv_vfmacc_vv_f32m2(k, k, j, vl),
                                 __riscv_vfmul_vv_f32m2(__riscv_vfmacc_vv_f32m2(s2, s2, j, vl), s1, vl), c, vl);
    ফেরত __riscv_vmerge_vvm_f32m2(r1, __riscv_vfmul_vv_f32m2(s1, s1, vl),
                                    __riscv_vmfgt_vf_f32m2_b16(__riscv_vfabs_v_f32m2(n, vl), 192.0f, vl), vl);
}

স্থির অন্তঃরেখ vfloat32m2_t rvv_tanh_approx_f32m2(vfloat32m2_t x, size_t vl) {
    ধ্রুবক vfloat32m2_t abs_x       = __riscv_vfabs_v_f32m2(x, vl);
    ধ্রুবক vfloat32m2_t neg_2_abs   = __riscv_vfmul_vf_f32m2(abs_x, -2.0f, vl);
    ধ্রুবক vfloat32m2_t exp_term    = rvv_expf_approx_f32m2(neg_2_abs, vl);
    ধ্রুবক vfloat32m2_t numerator   = __riscv_vfsub_vf_f32m2(exp_term, 1.0f, vl);
    ধ্রুবক vfloat32m2_t denominator = __riscv_vfadd_vf_f32m2(exp_term, 1.0f, vl);
    ধ্রুবক vfloat32m2_t tanh_abs    = __riscv_vfneg_v_f32m2(__riscv_vfdiv_vv_f32m2(numerator, denominator, vl), vl);
    ধ্রুবক vbool16_t    neg_mask    = __riscv_vmflt_vf_f32m2_b16(x, 0.0f, vl);
    ধ্রুবক vfloat32m2_t tanh_neg    = __riscv_vfneg_v_f32m2(tanh_abs, vl);
    ফেরত __riscv_vmerge_vvm_f32m2(tanh_abs, tanh_neg, neg_mask, vl);
}

স্থির শূন্য rvv_softcap_tanh_inplace_f32(দশমিক * dst, int64_t dst_stride, int64_t tile_rows, int64_t n, দশমিক softcap) {
    জন্য (পূর্ণসংখ্যা tq = 0; tq < tile_rows; ++tq, dst += dst_stride) {
        দশমিক * dst_row   = dst;
        int64_t remaining = n;
        যখন (remaining > 0) {
            ধ্রুবক size_t vl = __riscv_vsetvl_e32m2(remaining);
            vfloat32m2_t v  = __riscv_vle32_v_f32m2(dst_row, vl);
            v               = rvv_tanh_approx_f32m2(v, vl);
            v               = __riscv_vfmul_vf_f32m2(v, softcap, vl);
            __riscv_vse32_v_f32m2(dst_row, v, vl);
            dst_row += vl;
            remaining -= vl;
        }
    }
}

স্থির অন্তঃরেখ দশমিক rvv_softmax_exp_inplace_f32(দশমিক * dst, int64_t n, দশমিক max_মান) {
    দশমিক row_যোগ = 0.0f;
    যখন (n > 0) {
        ধ্রুবক size_t vl = __riscv_vsetvl_e32m2(n);
        vfloat32m2_t v  = __riscv_vle32_v_f32m2(dst, vl);
        v               = __riscv_vfsub_vf_f32m2(v, max_মান, vl);
        v               = rvv_expf_approx_f32m2(v, vl);
        __riscv_vse32_v_f32m2(dst, v, vl);
        row_যোগ += reduce_যোগ_f32m2_vlen1024(v, vl);
        dst += vl;
        n -= vl;
    }
    ফেরত row_যোগ;
}

স্থির অন্তঃরেখ দশমিক rvv_add_max_inplace_f32(দশমিক * dst, ধ্রুবক দশমিক * src, int64_t n) {
    দশমিক max_val = -INFINITY;
    যখন (n > 0) {
        ধ্রুবক size_t vl   = __riscv_vsetvl_e32m4(n);
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
    ফেরত max_val;
}

স্থির অন্তঃরেখ দশমিক rvv_softcap_add_max_inplace_f32(দশমিক * dst, ধ্রুবক দশমিক * src, int64_t n, দশমিক softcap) {
    যদি (softcap == 0.0f) {
        ফেরত rvv_add_max_inplace_f32(dst, src, n);
    }

    দশমিক max_val = -INFINITY;
    যখন (n > 0) {
        ধ্রুবক size_t vl   = __riscv_vsetvl_e32m2(n);
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
    ফেরত max_val;
}

স্থির অন্তঃরেখ শূন্য rvv_zero_f32(দশমিক * dst, int64_t n) {
    যখন (n > 0) {
        ধ্রুবক size_t       vl = __riscv_vsetvl_e32m4(n);
        ধ্রুবক vfloat32m4_t z  = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        __riscv_vse32_v_f32m4(dst, z, vl);
        dst += vl;
        n -= vl;
    }
}

স্থির অন্তঃরেখ শূন্য rvv_scale_f32(দশমিক * dst, দশমিক scale, int64_t n) {
    যখন (n > 0) {
        ধ্রুবক size_t vl = __riscv_vsetvl_e32m4(n);
        vfloat32m4_t v  = __riscv_vle32_v_f32m4(dst, vl);
        v               = __riscv_vfmul_vf_f32m4(v, scale, vl);
        __riscv_vse32_v_f32m4(dst, v, vl);
        dst += vl;
        n -= vl;
    }
}

স্থির অন্তঃরেখ শূন্য rvv_add_inplace_f32(দশমিক *       dst,
                                       int64_t       dst_stride,
                                       ধ্রুবক দশমিক * src,
                                       int64_t       src_stride,
                                       int64_t       tile_rows,
                                       int64_t       n) {
    জন্য (পূর্ণসংখ্যা tq = 0; tq < tile_rows; ++tq, dst += dst_stride, src += src_stride) {
        int64_t       remaining = n;
        দশমিক *       dst_row   = dst;
        ধ্রুবক দশমিক * src_row   = src;
        যখন (remaining > 0) {
            ধ্রুবক size_t vl   = __riscv_vsetvl_e32m4(remaining);
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

স্থির অন্তঃরেখ দশমিক rvv_max_f32(ধ্রুবক দশমিক * src, int64_t n) {
    দশমিক max_val = -INFINITY;
    যখন (n > 0) {
        ধ্রুবক size_t       vl   = __riscv_vsetvl_e32m4(n);
        ধ্রুবক vfloat32m4_t v    = __riscv_vle32_v_f32m4(src, vl);
        vfloat32m1_t       seed = __riscv_vfmv_v_f_f32m1(max_val, 1);
        seed                    = __riscv_vfredmax_vs_f32m4_f32m1(v, seed, vl);
        max_val                 = __riscv_vfmv_f_s_f32m1_f32(seed);
        src += vl;
        n -= vl;
    }
    ফেরত max_val;
}

স্থির শূন্য rvv_pack_f32_as_scaled_f16(শূন্য *       dst,
                                       int64_t      dst_row_stride,
                                       ধ্রুবক শূন্য * src,
                                       int64_t      src_row_stride,
                                       int64_t      tile_rows,
                                       int64_t      n,
                                       দশমিক        scale) {
    জন্য (পূর্ণসংখ্যা tq = 0; tq < tile_rows; ++tq) {
        ধ্রুবক দশমিক * row_ptr     = (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) src + tq * src_row_stride);
        _Float16 *    dst_row_ptr = (_Float16 *) ((অক্ষর *) dst + tq * dst_row_stride);
        int64_t       remaining   = n;
        যখন (remaining > 0) {
            ধ্রুবক size_t vl        = __riscv_vsetvl_e32m4(remaining);
            vfloat32m4_t v32       = __riscv_vle32_v_f32m4(row_ptr, vl);
            v32                    = __riscv_vfmul_vf_f32m4(v32, scale, vl);
            ধ্রুবক vfloat16m2_t v16 = __riscv_vfncvt_f_f_w_f16m2(v32, vl);
            __riscv_vse16_v_f16m2(dst_row_ptr, v16, vl);
            dst_row_ptr += vl;
            row_ptr += vl;
            remaining -= vl;
        }
    }
}

স্থির শূন্য rvv_pack_scaled_f16_as_f32(শূন্য *       dst,
                                       int64_t      dst_row_stride,
                                       ধ্রুবক শূন্য * src,
                                       int64_t      src_row_stride,
                                       int64_t      tile_rows,
                                       int64_t      n,
                                       দশমিক        scale) {
    জন্য (পূর্ণসংখ্যা tq = 0; tq < tile_rows; ++tq) {
        ধ্রুবক _Float16 * row_ptr     = (ধ্রুবক _Float16 *) ((ধ্রুবক অক্ষর *) src + tq * src_row_stride);
        দশমিক *          dst_row_ptr = (দশমিক *) ((অক্ষর *) dst + tq * dst_row_stride);
        int64_t          remaining   = n;
        যখন (remaining > 0) {
            ধ্রুবক size_t       vl  = __riscv_vsetvl_e16m2(remaining);
            ধ্রুবক vfloat16m2_t v16 = __riscv_vle16_v_f16m2(row_ptr, vl);
            vfloat32m4_t       v32 = __riscv_vfwcvt_f_f_v_f32m4(v16, vl);
            v32                    = __riscv_vfmul_vf_f32m4(v32, scale, vl);
            __riscv_vse32_v_f32m4(dst_row_ptr, v32, vl);
            dst_row_ptr += vl;
            row_ptr += vl;
            remaining -= vl;
        }
    }
}

স্থির শূন্য rvv_pack_scaled_f32_as_f32(শূন্য *       dst,
                                       int64_t      dst_row_stride,
                                       ধ্রুবক শূন্য * src,
                                       int64_t      src_row_stride,
                                       int64_t      tile_rows,
                                       int64_t      n,
                                       দশমিক *      scale) {
    জন্য (পূর্ণসংখ্যা tq = 0; tq < tile_rows; ++tq) {
        ধ্রুবক দশমিক * row_ptr     = (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) src + tq * src_row_stride);
        দশমিক *       dst_row_ptr = (দশমিক *) ((অক্ষর *) dst + tq * dst_row_stride);
        int64_t       remaining   = n;
        যখন (remaining > 0) {
            ধ্রুবক size_t vl  = __riscv_vsetvl_e32m4(remaining);
            vfloat32m4_t v32 = __riscv_vle32_v_f32m4(row_ptr, vl);
            v32              = __riscv_vfmul_vf_f32m4(v32, scale[tq], vl);
            __riscv_vse32_v_f32m4(dst_row_ptr, v32, vl);
            dst_row_ptr += vl;
            row_ptr += vl;
            remaining -= vl;
        }
    }
}

স্থির অন্তঃরেখ শূন্য rvv_transposed_s32_mn_to_nm(int8_t * dst,
                                               int64_t  n_dst_stride,
                                               int8_t * src,
                                               int64_t  m_src_stride,
                                               int64_t  m,
                                               int64_t  n) {
    int8_t * in  = src;
    int8_t * out = dst;

    __asm__ অস্থির(
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

স্থির অন্তঃরেখ শূন্য rvv_transposed_s16_mn_to_nm(int8_t * dst,
                                               int64_t  n_dst_stride,
                                               int8_t * src,
                                               int64_t  m_src_stride,
                                               int64_t  m,
                                               int64_t  n) {
    int8_t * in  = src;
    int8_t * out = dst;

    __asm__ অস্থির(
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

স্থির অন্তঃরেখ শূন্য rvv_qk_dot_tile_f16_x1(দশমিক *          dst,
                                          ধ্রুবক _Float16 * q_row,
                                          ধ্রুবক _Float16 * k_pack,
                                          int64_t          dk,
                                          int64_t          kv_tile) {
    ধ্রুবক size_t vl  = __riscv_vsetvl_e16m1(kv_tile);
    vfloat32m2_t acc = __riscv_vfmv_v_f_f32m2(0.0f, vl);

    জন্য (int64_t d = 0; d < dk; ++d) {
        ধ্রুবক vfloat16m1_t k_vec = __riscv_vle16_v_f16m1(k_pack + d * ggml_fa_tile_config::KV, vl);
        acc                      = __riscv_vfwmacc_vf_f32m2(acc, q_row[d], k_vec, vl);
    }

    __riscv_vse32_v_f32m2(dst, acc, vl);
}

স্থির অন্তঃরেখ শূন্য rvv_qk_dot_tile_f16_x4(দশমিক *          dst0,
                                          দশমিক *          dst1,
                                          দশমিক *          dst2,
                                          দশমিক *          dst3,
                                          ধ্রুবক _Float16 * q0,
                                          ধ্রুবক _Float16 * q1,
                                          ধ্রুবক _Float16 * q2,
                                          ধ্রুবক _Float16 * q3,
                                          ধ্রুবক _Float16 * k_pack,
                                          int64_t          dk,
                                          int64_t          kv_tile) {
    ধ্রুবক size_t vl   = __riscv_vsetvl_e16m1(kv_tile);
    vfloat32m2_t acc0 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    vfloat32m2_t acc1 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    vfloat32m2_t acc2 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    vfloat32m2_t acc3 = __riscv_vfmv_v_f_f32m2(0.0f, vl);

    জন্য (int64_t d = 0; d < dk; ++d) {
        ধ্রুবক vfloat16m1_t k_vec = __riscv_vle16_v_f16m1(k_pack + d * ggml_fa_tile_config::KV, vl);
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

স্থির অন্তঃরেখ শূন্য rvv_pv_accumulate_f16_x1(দশমিক *          dst,
                                            ধ্রুবক দশমিক *    prob,
                                            ধ্রুবক _Float16 * v_pack,
                                            int64_t          kv_tile,
                                            int64_t          dv) {
    int64_t d_বাম = dv;
    int64_t d_off  = 0;

    যখন (d_বাম > 0) {
        ধ্রুবক size_t vl  = __riscv_vsetvl_e16m2(d_বাম);
        vfloat32m4_t acc = __riscv_vle32_v_f32m4(dst + d_off, vl);

        জন্য (int64_t tk = 0; tk < kv_tile; ++tk) {
            ধ্রুবক vfloat16m2_t v16 = __riscv_vle16_v_f16m2(v_pack + tk * dv + d_off, vl);
            ধ্রুবক vfloat32m4_t v32 = __riscv_vfwcvt_f_f_v_f32m4(v16, vl);
            acc                    = __riscv_vfmacc_vf_f32m4(acc, prob[tk], v32, vl);
        }

        __riscv_vse32_v_f32m4(dst + d_off, acc, vl);
        d_বাম -= vl;
        d_off += vl;
    }
}

স্থির অন্তঃরেখ শূন্য rvv_pv_accumulate_f16_x4(দশমিক *          dst0,
                                            দশমিক *          dst1,
                                            দশমিক *          dst2,
                                            দশমিক *          dst3,
                                            ধ্রুবক দশমিক *    prob0,
                                            ধ্রুবক দশমিক *    prob1,
                                            ধ্রুবক দশমিক *    prob2,
                                            ধ্রুবক দশমিক *    prob3,
                                            ধ্রুবক _Float16 * v_pack,
                                            int64_t          kv_tile,
                                            int64_t          dv) {
    int64_t d_বাম = dv;
    int64_t d_off  = 0;

    যখন (d_বাম > 0) {
        ধ্রুবক size_t vl   = __riscv_vsetvl_e16m2(d_বাম);
        vfloat32m4_t acc0 = __riscv_vle32_v_f32m4(dst0 + d_off, vl);
        vfloat32m4_t acc1 = __riscv_vle32_v_f32m4(dst1 + d_off, vl);
        vfloat32m4_t acc2 = __riscv_vle32_v_f32m4(dst2 + d_off, vl);
        vfloat32m4_t acc3 = __riscv_vle32_v_f32m4(dst3 + d_off, vl);

        জন্য (int64_t tk = 0; tk < kv_tile; ++tk) {
            ধ্রুবক vfloat16m2_t v16 = __riscv_vle16_v_f16m2(v_pack + tk * dv + d_off, vl);
            ধ্রুবক vfloat32m4_t v32 = __riscv_vfwcvt_f_f_v_f32m4(v16, vl);
            acc0                   = __riscv_vfmacc_vf_f32m4(acc0, prob0[tk], v32, vl);
            acc1                   = __riscv_vfmacc_vf_f32m4(acc1, prob1[tk], v32, vl);
            acc2                   = __riscv_vfmacc_vf_f32m4(acc2, prob2[tk], v32, vl);
            acc3                   = __riscv_vfmacc_vf_f32m4(acc3, prob3[tk], v32, vl);
        }

        __riscv_vse32_v_f32m4(dst0 + d_off, acc0, vl);
        __riscv_vse32_v_f32m4(dst1 + d_off, acc1, vl);
        __riscv_vse32_v_f32m4(dst2 + d_off, acc2, vl);
        __riscv_vse32_v_f32m4(dst3 + d_off, acc3, vl);
        d_বাম -= vl;
        d_off += vl;
    }
}

স্থির অন্তঃরেখ শূন্য rvv_qk_dot_tile(দশমিক *       dst,
                                   ধ্রুবক দশমিক * q_row,
                                   ধ্রুবক দশমিক * k_pack,
                                   int64_t       dk,
                                   int64_t       kv_tile,
                                   দশমিক         scale) {
    ধ্রুবক size_t vl  = __riscv_vsetvl_e32m4(kv_tile);
    vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);

    জন্য (int64_t d = 0; d < dk; ++d) {
        ধ্রুবক vfloat32m4_t k_vec = __riscv_vle32_v_f32m4(k_pack + d * kv_tile, vl);
        acc                      = __riscv_vfmacc_vf_f32m4(acc, q_row[d] * scale, k_vec, vl);
    }

    __riscv_vse32_v_f32m4(dst, acc, vl);
}

স্থির অন্তঃরেখ শূন্য rvv_pv_accumulate(দশমিক *       dst,
                                     ধ্রুবক দশমিক * prob,
                                     ধ্রুবক দশমিক * v_pack,
                                     int64_t       kv_tile,
                                     int64_t       dv) {
    int64_t d_বাম = dv;
    int64_t d_off  = 0;

    যখন (d_বাম > 0) {
        ধ্রুবক size_t vl  = __riscv_vsetvl_e32m4(d_বাম);
        vfloat32m4_t acc = __riscv_vle32_v_f32m4(dst + d_off, vl);

        জন্য (int64_t tk = 0; tk < kv_tile; ++tk) {
            ধ্রুবক vfloat32m4_t v_vec = __riscv_vle32_v_f32m4(v_pack + tk * dv + d_off, vl);
            acc                      = __riscv_vfmacc_vf_f32m4(acc, prob[tk], v_vec, vl);
        }

        __riscv_vse32_v_f32m4(dst + d_off, acc, vl);
        d_বাম -= vl;
        d_off += vl;
    }
}

স্থির শূন্য permute_transpose_impl(ধ্রুবক ggml_tensor * src0,
                                   ggml_tensor *       dst,
                                   int64_t             batch,
                                   int64_t             m,
                                   int64_t             n,
                                   int64_t             batch_stride,
                                   int64_t             m_src_stride,
                                   int64_t             n_src_stride,
                                   int64_t             n_dst_stride,
                                   পূর্ণসংখ্যা                 ith,
                                   পূর্ণসংখ্যা                 nth) {
    GGML_ASSERT(n_src_stride == আকার(int32_t) || n_src_stride == আকার(int16_t));

    যদি (n_src_stride == আকার(int32_t)) {
        জন্য (int64_t bi = ith; bi < batch; bi += nth) {
            rvv_transposed_s32_mn_to_nm((int8_t *) ((অক্ষর *) dst->তথ্য + bi * batch_stride), n_dst_stride,
                                        (int8_t *) ((অক্ষর *) src0->তথ্য + bi * batch_stride), m_src_stride, m, n);
        }
    } অন্যথা যদি (n_src_stride == আকার(int16_t)) {
        জন্য (int64_t bi = ith; bi < batch; bi += nth) {
            rvv_transposed_s16_mn_to_nm((int8_t *) ((অক্ষর *) dst->তথ্য + bi * batch_stride), n_dst_stride,
                                        (int8_t *) ((অক্ষর *) src0->তথ্য + bi * batch_stride), m_src_stride, m, n);
        }
    } অন্যথা {
        GGML_ABORT("not implemented");
    }
}

template <size_t QLEN>
স্থির শূন্য flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_mrow(দশমিক **            pq,
                                                                  ধ্রুবক অক্ষর *        k_তথ্য_row,
                                                                  ধ্রুবক অক্ষর *        v_তথ্য_row,
                                                                  ধ্রুবক ggml_fp16_t * mp,
                                                                  দশমিক **            sinks,
                                                                  দশমিক **            dst,
                                                                  দশমিক               scale,
                                                                  দশমিক               logit_softcap,
                                                                  দশমিক               slope,
                                                                  int64_t             nek1,
                                                                  int64_t             nbk1,
                                                                  int64_t             nbv1,
                                                                  int64_t             DV,
                                                                  int64_t             DK,
                                                                  শূন্য *              tcm_বাফার,
                                                                  size_t              tcm_বাফার_আকার) {
    GGML_ASSERT(flash_attn_ext_supported_shape_vlen1024_vf16(DK, DV));
    দশমিক S[QLEN] = { 0.0f };       //যোগ
    দশমিক M[QLEN] = { -INFINITY };  // maximum KQ value

    _Float16 *   kq16_বাফার          = (_Float16 *) tcm_বাফার;
    _Float16 *   qv_বাফার            = kq16_বাফার + QLEN * DV;
    ধ্রুবক size_t qkv_অস্থায়ী_বাফার_আকার = (QLEN * DV + QLEN * DK) * আকার(_Float16);
    অক্ষর *       kv_tile_বাফার       = (অক্ষর *) (qv_বাফার + QLEN * DK);

    {
        vfloat16m2_t VKQ16_v = __riscv_vfmv_v_f_f16m2(0.0f, DV);
        জন্য (int64_t i = 0; i < QLEN; ++i) {
            __riscv_vse16_v_f16m2(kq16_বাফার + i * DV, VKQ16_v, DV);
            vfloat16m2_t Q_q_v = __riscv_vfncvt_f_f_w_f16m2(__riscv_vle32_v_f32m4(pq[i], DK), DK);
            __riscv_vse16_v_f16m2(qv_বাফার + i * DK, Q_q_v, DK);
        }
    }

    ধ্রুবক uintptr_t scratch_addr = reinterpret_cast<uintptr_t>(kv_tile_বাফার);
    ধ্রুবক size_t    scratch_আকার = tcm_বাফার_আকার > qkv_অস্থায়ী_বাফার_আকার ? tcm_বাফার_আকার - qkv_অস্থায়ী_বাফার_আকার : 0;
    ধ্রুবক uintptr_t kq_tile_addr = align_up(scratch_addr, alignof(দশমিক));
    ধ্রুবক size_t    scratch_prefix = kq_tile_addr - scratch_addr;
    ধ্রুবক size_t    packed_tile_আকার =
        QLEN * আকার(দশমিক) + DK * আকার(_Float16) + DV * আকার(_Float16) + আকার(দশমিক);
    ধ্রুবক int64_t max_ic_tile_step = ((int64_t) __riscv_vsetvlmax_e16m1()) & ~((int64_t) 7);
    ধ্রুবক int64_t max_fit_by_tcm =
        scratch_আকার > scratch_prefix ? (int64_t) ((scratch_আকার - scratch_prefix) / packed_tile_আকার) : 0;
    ধ্রুবক int64_t ic_tile_step = std::min(max_ic_tile_step, max_fit_by_tcm) & ~((int64_t) 7);

    ধ্রুবক uintptr_t k_tile_addr  = kq_tile_addr + QLEN * ic_tile_step * আকার(দশমিক);
    ধ্রুবক uintptr_t v_tile_addr  = k_tile_addr + DK * ic_tile_step * আকার(_Float16);
    ধ্রুবক uintptr_t mv_tile_addr = v_tile_addr + ic_tile_step * DV * আকার(_Float16);

    যদি (ic_tile_step >= 8) {
        দশমিক *    kq_tile_বাফার = reinterpret_cast<দশমিক *>(kq_tile_addr);
        _Float16 * k_tile_pack    = reinterpret_cast<_Float16 *>(k_tile_addr);
        _Float16 * v_tile_pack    = reinterpret_cast<_Float16 *>(v_tile_addr);
        দশমিক *    mv_tile_pack   = reinterpret_cast<দশমিক *>(mv_tile_addr);

        ধ্রুবক int64_t k_tile_byte_stride = ic_tile_step * (int64_t) আকার(_Float16);

        int64_t ic_step = 0;
        জন্য (int64_t ic = 0; ic < nek1; ++ic) {
            ধ্রুবক দশমিক mv = mp ? slope * ((_Float16 *) mp)[ic] : 0.0f;

            যদি (mv != -INFINITY) {
                ধ্রুবক _Float16 * k_তথ্য = (ধ্রুবক _Float16 *) (k_তথ্য_row + ic * nbk1);
                ধ্রুবক _Float16 * v_তথ্য = (ধ্রুবক _Float16 *) (v_তথ্য_row + ic * nbv1);

                ধ্রুবক vfloat16m2_t k_তথ্য_v = __riscv_vle16_v_f16m2(k_তথ্য, DK);
                ধ্রুবক vfloat16m2_t v_তথ্য_v = __riscv_vle16_v_f16m2(v_তথ্য, DV);
                __riscv_vsse16_v_f16m2(k_tile_pack + ic_step, k_tile_byte_stride, k_তথ্য_v, DK);
                __riscv_vse16_v_f16m2(v_tile_pack + ic_step * DV, v_তথ্য_v, DV);
                mv_tile_pack[ic_step] = mv;
                ic_step++;
            }

            যদি (ic_step > 0 && (ic_step == ic_tile_step || ic == (nek1 - 1))) {
                যদি constexpr (QLEN == 4) {
                    ধ্রুবক size_t qk_vl   = __riscv_vsetvl_e16m1(ic_step);
                    vfloat32m2_t qk_acc0 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);
                    vfloat32m2_t qk_acc1 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);
                    vfloat32m2_t qk_acc2 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);
                    vfloat32m2_t qk_acc3 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);

                    জন্য (int64_t d = 0; d < DK; ++d) {
                        ধ্রুবক vfloat16m1_t k_vec = __riscv_vle16_v_f16m1(k_tile_pack + d * ic_tile_step, qk_vl);
                        qk_acc0 = __riscv_vfwmacc_vf_f32m2(qk_acc0, qv_বাফার[0 * DK + d], k_vec, qk_vl);
                        qk_acc1 = __riscv_vfwmacc_vf_f32m2(qk_acc1, qv_বাফার[1 * DK + d], k_vec, qk_vl);
                        qk_acc2 = __riscv_vfwmacc_vf_f32m2(qk_acc2, qv_বাফার[2 * DK + d], k_vec, qk_vl);
                        qk_acc3 = __riscv_vfwmacc_vf_f32m2(qk_acc3, qv_বাফার[3 * DK + d], k_vec, qk_vl);
                    }

                    qk_acc0 = __riscv_vfmul_vf_f32m2(qk_acc0, scale, qk_vl);
                    qk_acc1 = __riscv_vfmul_vf_f32m2(qk_acc1, scale, qk_vl);
                    qk_acc2 = __riscv_vfmul_vf_f32m2(qk_acc2, scale, qk_vl);
                    qk_acc3 = __riscv_vfmul_vf_f32m2(qk_acc3, scale, qk_vl);

                    __riscv_vse32_v_f32m2(kq_tile_বাফার + 0 * ic_tile_step, qk_acc0, qk_vl);
                    __riscv_vse32_v_f32m2(kq_tile_বাফার + 1 * ic_tile_step, qk_acc1, qk_vl);
                    __riscv_vse32_v_f32m2(kq_tile_বাফার + 2 * ic_tile_step, qk_acc2, qk_vl);
                    __riscv_vse32_v_f32m2(kq_tile_বাফার + 3 * ic_tile_step, qk_acc3, qk_vl);
                } অন্যথা {
                    static_assert(QLEN == 2, "unsupported QLEN");

                    ধ্রুবক size_t qk_vl   = __riscv_vsetvl_e16m1(ic_step);
                    vfloat32m2_t qk_acc0 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);
                    vfloat32m2_t qk_acc1 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);

                    জন্য (int64_t d = 0; d < DK; ++d) {
                        ধ্রুবক vfloat16m1_t k_vec = __riscv_vle16_v_f16m1(k_tile_pack + d * ic_tile_step, qk_vl);
                        qk_acc0 = __riscv_vfwmacc_vf_f32m2(qk_acc0, qv_বাফার[0 * DK + d], k_vec, qk_vl);
                        qk_acc1 = __riscv_vfwmacc_vf_f32m2(qk_acc1, qv_বাফার[1 * DK + d], k_vec, qk_vl);
                    }

                    qk_acc0 = __riscv_vfmul_vf_f32m2(qk_acc0, scale, qk_vl);
                    qk_acc1 = __riscv_vfmul_vf_f32m2(qk_acc1, scale, qk_vl);

                    __riscv_vse32_v_f32m2(kq_tile_বাফার + 0 * ic_tile_step, qk_acc0, qk_vl);
                    __riscv_vse32_v_f32m2(kq_tile_বাফার + 1 * ic_tile_step, qk_acc1, qk_vl);
                }

                জন্য (পূর্ণসংখ্যা i = 0; i < QLEN; ++i) {
                    দশমিক *     row_ptr = kq_tile_বাফার + i * ic_tile_step;
                    ধ্রুবক দশমিক tile_max =
                        rvv_softcap_add_max_inplace_f32(row_ptr, mv_tile_pack, ic_step, logit_softcap);

                    ধ্রুবক দশমিক Mold = M[i];

                    যদি (tile_max > Mold) {
                        ধ্রুবক দশমিক ms = expf(Mold - tile_max);
                        M[i]           = tile_max;
                        S[i] *= ms;

                        vfloat16m2_t VKQ16_v = __riscv_vle16_v_f16m2(kq16_বাফার + i * DV, DV);
                        VKQ16_v              = __riscv_vfmul_vf_f16m2(VKQ16_v, (_Float16) ms, DV);
                        __riscv_vse16_v_f16m2(kq16_বাফার + i * DV, VKQ16_v, DV);
                    }

                    S[i] += rvv_softmax_exp_inplace_f32(row_ptr, ic_step, M[i]);
                }

                যদি constexpr (QLEN == 4) {
                    vfloat16m2_t pv_acc0 = __riscv_vle16_v_f16m2(kq16_বাফার + 0 * DV, DV);
                    vfloat16m2_t pv_acc1 = __riscv_vle16_v_f16m2(kq16_বাফার + 1 * DV, DV);
                    vfloat16m2_t pv_acc2 = __riscv_vle16_v_f16m2(kq16_বাফার + 2 * DV, DV);
                    vfloat16m2_t pv_acc3 = __riscv_vle16_v_f16m2(kq16_বাফার + 3 * DV, DV);

                    জন্য (int64_t tk = 0; tk < ic_step; ++tk) {
                        ধ্রুবক vfloat16m2_t v16 = __riscv_vle16_v_f16m2(v_tile_pack + tk * DV, DV);
                        pv_acc0 =
                            __riscv_vfmacc_vf_f16m2(pv_acc0, (_Float16) kq_tile_বাফার[0 * ic_tile_step + tk], v16, DV);
                        pv_acc1 =
                            __riscv_vfmacc_vf_f16m2(pv_acc1, (_Float16) kq_tile_বাফার[1 * ic_tile_step + tk], v16, DV);
                        pv_acc2 =
                            __riscv_vfmacc_vf_f16m2(pv_acc2, (_Float16) kq_tile_বাফার[2 * ic_tile_step + tk], v16, DV);
                        pv_acc3 =
                            __riscv_vfmacc_vf_f16m2(pv_acc3, (_Float16) kq_tile_বাফার[3 * ic_tile_step + tk], v16, DV);
                    }

                    __riscv_vse16_v_f16m2(kq16_বাফার + 0 * DV, pv_acc0, DV);
                    __riscv_vse16_v_f16m2(kq16_বাফার + 1 * DV, pv_acc1, DV);
                    __riscv_vse16_v_f16m2(kq16_বাফার + 2 * DV, pv_acc2, DV);
                    __riscv_vse16_v_f16m2(kq16_বাফার + 3 * DV, pv_acc3, DV);
                } অন্যথা {
                    static_assert(QLEN == 2, "unsupported QLEN");
                    vfloat16m2_t pv_acc0 = __riscv_vle16_v_f16m2(kq16_বাফার + 0 * DV, DV);
                    vfloat16m2_t pv_acc1 = __riscv_vle16_v_f16m2(kq16_বাফার + 1 * DV, DV);

                    জন্য (int64_t tk = 0; tk < ic_step; ++tk) {
                        ধ্রুবক vfloat16m2_t v16 = __riscv_vle16_v_f16m2(v_tile_pack + tk * DV, DV);
                        pv_acc0 =
                            __riscv_vfmacc_vf_f16m2(pv_acc0, (_Float16) kq_tile_বাফার[0 * ic_tile_step + tk], v16, DV);
                        pv_acc1 =
                            __riscv_vfmacc_vf_f16m2(pv_acc1, (_Float16) kq_tile_বাফার[1 * ic_tile_step + tk], v16, DV);
                    }

                    __riscv_vse16_v_f16m2(kq16_বাফার + 0 * DV, pv_acc0, DV);
                    __riscv_vse16_v_f16m2(kq16_বাফার + 1 * DV, pv_acc1, DV);
                }

                ic_step = 0;
            }
        }
    } অন্যথা {
        জন্য (int64_t ic = 0; ic < nek1; ++ic) {
            ধ্রুবক দশমিক mv = mp ? slope * ((_Float16 *) mp)[ic] : 0.0f;

            ধ্রুবক অক্ষর * k_তথ্য = k_তথ্য_row + ic * nbk1;
            ধ্রুবক অক্ষর * v_তথ্য = v_তথ্য_row + ic * nbv1;

            vfloat16m2_t k_তথ্য_v;
            vfloat16m2_t v_তথ্য_v;

            যদি (mv != -INFINITY) {
                k_তথ্য_v = __riscv_vle16_v_f16m2((_Float16 *) k_তথ্য, DK);
                v_তথ্য_v = __riscv_vle16_v_f16m2((_Float16 *) v_তথ্য, DV);
            } অন্যথা {
                চালু;
            }

            জন্য (পূর্ণসংখ্যা i = 0; i < QLEN; ++i) {
                vfloat16m2_t Q_q_v    = __riscv_vle16_v_f16m2(qv_বাফার + i * DK, DK);
                vfloat32m4_t qk_acc_v = __riscv_vfwmul_vv_f32m4(k_তথ্য_v, Q_q_v, DK);
                দশমিক        s        = reduce_যোগ_f32m4_vlen1024(qk_acc_v, DK);
                s                     = s * scale;
                যদি (logit_softcap != 0.0f) {
                    s = logit_softcap * tanhf(s);
                }
                s += mv;

                ধ্রুবক দশমিক Mold = M[i];

                দশমিক ms = 1.0f;  // upon new higher max val, scale VKQ and KQ sum with this value
                দশমিক vs = 1.0f;  // post-softmax KQ value, expf(s - M)

                vfloat16m2_t VKQ16_v = __riscv_vle16_v_f16m2(kq16_বাফার + i * DV, DV);
                যদি (s > M[i]) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M[i] = s;
                    ms   = expf(Mold - M[i]);

                    // V = V*expf(Mold - M)
                    VKQ16_v = __riscv_vfmul_vf_f16m2(VKQ16_v, ms, DV);
                } অন্যথা {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M[i]);
                }
                VKQ16_v = __riscv_vfmacc_vf_f16m2(VKQ16_v, vs, v_তথ্য_v, DV);
                __riscv_vse16_v_f16m2(kq16_বাফার + i * DV, VKQ16_v, DV);
                S[i] = S[i] * ms + vs;  // scale and increment sum with partial sum
            }
        }
    }

    জন্য (পূর্ণসংখ্যা i = 0; i < QLEN; ++i) {
        vfloat16m2_t VKQ16_v = __riscv_vle16_v_f16m2(kq16_বাফার + i * DV, DV);
        vfloat32m4_t VKQ32_v = __riscv_vfwcvt_f_f_v_f32m4(VKQ16_v, DV);

        // sinks
        যদি (sinks[i]) {
            ধ্রুবক দশমিক s = *(sinks[i]);

            দশমিক ms = 1.0f;
            দশমিক vs = 1.0f;

            যদি (s > M[i]) {
                ms      = expf(M[i] - s);
                M[i]    = s;
                VKQ32_v = __riscv_vfmul_vf_f32m4(VKQ32_v, ms, DV);
            } অন্যথা {
                vs = expf(s - M[i]);
            }

            S[i] = S[i] * ms + vs;
        }

        // V /= S
        ধ্রুবক দশমিক S_inv = S[i] == 0.0f ? 0.0f : 1.0f / S[i];

        VKQ32_v = __riscv_vfmul_vf_f32m4(VKQ32_v, S_inv, DV);

        __riscv_vse32_v_f32m4(dst[i], VKQ32_v, DV);
    }
}

স্থির শূন্য flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_m1(ধ্রুবক দশমিক *       pq,
                                                                ধ্রুবক অক্ষর *        k_তথ্য_row,
                                                                ধ্রুবক অক্ষর *        v_তথ্য_row,
                                                                ধ্রুবক ggml_fp16_t * mp,
                                                                ধ্রুবক দশমিক *       sinks,
                                                                দশমিক *             dst,
                                                                দশমিক               scale,
                                                                দশমিক               logit_softcap,
                                                                দশমিক               slope,
                                                                int64_t             nek1,
                                                                int64_t             nbk1,
                                                                int64_t             nbv1,
                                                                int64_t             DV,
                                                                int64_t             DK) {
    GGML_ASSERT(flash_attn_ext_supported_shape_vlen1024_vf16(DK, DV));

    দশমিক S = 0.0f;       //যোগ
    দশমিক M = -INFINITY;  // maximum KQ value

    vfloat16m2_t VKQ16_v = __riscv_vfmv_v_f_f16m2(0.0f, DV);

    vfloat16m2_t Q_q_v = __riscv_vfncvt_f_f_w_f16m2(__riscv_vle32_v_f32m4(pq, DK), DK);

    জন্য (int64_t ic = 0; ic < nek1; ++ic) {
        ধ্রুবক দশমিক mv = mp ? slope * ((_Float16 *) mp)[ic] : 0.0f;
        যদি (mv == -INFINITY) {
            চালু;
        }

        ধ্রুবক অক্ষর * k_তথ্য = k_তথ্য_row + ic * nbk1;

        vfloat16m2_t k_তথ্য_v = __riscv_vle16_v_f16m2((_Float16 *) k_তথ্য, DK);

        vfloat32m4_t qk_acc_v = __riscv_vfwmul_vv_f32m4(k_তথ্য_v, Q_q_v, DK);
        দশমিক        s        = reduce_যোগ_f32m4_vlen1024(qk_acc_v, DK);

        s = s * scale;  // scale KQ value

        যদি (logit_softcap != 0.0f) {
            s = logit_softcap * tanhf(s);
        }

        s += mv;  // apply mask

        ধ্রুবক দশমিক Mold = M;

        দশমিক ms = 1.0f;  // upon new higher max val, scale VKQ and KQ sum with this value
        দশমিক vs = 1.0f;  // post-softmax KQ value, expf(s - M)

        ধ্রুবক অক্ষর * v_তথ্য = v_তথ্য_row + ic * nbv1;

        vfloat16m2_t v_তথ্য_v = __riscv_vle16_v_f16m2((_Float16 *) v_তথ্য, DV);

        যদি (s > M) {
            // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
            M  = s;
            ms = expf(Mold - M);

            // V = V*expf(Mold - M)
            VKQ16_v = __riscv_vfmul_vf_f16m2(VKQ16_v, ms, DV);
        } অন্যথা {
            // no new maximum, ms == 1.0f, vs != 1.0f
            vs = expf(s - M);
        }

        VKQ16_v = __riscv_vfmacc_vf_f16m2(VKQ16_v, vs, v_তথ্য_v, DV);

        S = S * ms + vs;  // scale and increment sum with partial sum
    }

    vfloat32m4_t VKQ32_v = __riscv_vfwcvt_f_f_v_f32m4(VKQ16_v, DV);

    // sinks
    যদি (sinks) {
        ধ্রুবক দশমিক s = *sinks;

        দশমিক ms = 1.0f;
        দশমিক vs = 1.0f;

        যদি (s > M) {
            ms      = expf(M - s);
            M       = s;
            VKQ32_v = __riscv_vfmul_vf_f32m4(VKQ32_v, ms, DV);
        } অন্যথা {
            vs = expf(s - M);
        }

        S = S * ms + vs;
    }

    // V /= S
    ধ্রুবক দশমিক S_inv = S == 0.0f ? 0.0f : 1.0f / S;

    VKQ32_v = __riscv_vfmul_vf_f32m4(VKQ32_v, S_inv, DV);

    __riscv_vse32_v_f32m4(dst, VKQ32_v, DV);
}

}  // namespace

শূন্য memcpy1d(শূন্য * dst, ধ্রুবক শূন্য * src, int64_t size) {
    size_t byte_আকার_all = size;
    size_t vlen          = __riscv_vlenb() * 8;
    যদি (vlen == 256) {
        // 1024 bytes
        __asm__ অস্থির(
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
            : [size] "r"(byte_আকার_all)
            : "cc", "t0", "t1");
    } অন্যথা যদি (vlen == 1024) {
        // 2048 bytes
        __asm__ অস্থির(
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
            : [size] "r"(byte_আকার_all)
            : "cc", "t0", "t1", "t2", "t3", "t5");
    } অন্যথা {
        __asm__ অস্থির(
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
            : [size] "r"(byte_আকার_all)
            : "cc", "t0", "t1", "t2", "t4", "t3");
    }
}

শূন্য memcpy2d(শূন্য * dst, int64_t dst_stride, ধ্রুবক শূন্য * src, int64_t src_stride, int64_t tile_rows, int64_t size) {
    জন্য (int64_t i = 0; i < tile_rows; ++i) {
        memcpy1d((অক্ষর *) dst + i * dst_stride, (ধ্রুবক অক্ষর *) src + i * src_stride, size);
    }
}

শূন্য forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(ধ্রুবক ggml_compute_params * params,
                                                        ggml_tensor *               dst,
                                                        পূর্ণসংখ্যা                         ir0,
                                                        পূর্ণসংখ্যা                         ir1,
                                                        শূন্য *                      tcm_বাফার,
                                                        size_t                      tcm_বাফার_আকার) {
    ধ্রুবক ggml_tensor * q     = dst->src[0];
    ধ্রুবক ggml_tensor * k     = dst->src[1];
    ধ্রুবক ggml_tensor * v     = dst->src[2];
    ধ্রুবক ggml_tensor * mask  = dst->src[3];
    ধ্রুবক ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q, ne)
    GGML_TENSOR_LOCALS(size_t, nbq, q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k, ne)
    GGML_TENSOR_LOCALS(size_t, nbk, k, nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v, ne)
    GGML_TENSOR_LOCALS(size_t, nbv, v, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t, nb, dst, nb)

    ধ্রুবক int64_t DK = nek0;
    ধ্রুবক int64_t DV = nev0;
    ধ্রুবক int64_t N  = neq1;

    GGML_ASSERT(flash_attn_ext_supported_shape_vlen1024_vf16(DK, DV));

    // broadcast factors
    ধ্রুবক int64_t rk2 = neq2 / nek2;
    ধ্রুবক int64_t rk3 = neq3 / nek3;

    ধ্রুবক int64_t rv2 = neq2 / nev2;
    ধ্রুবক int64_t rv3 = neq3 / nev3;

    // parallelize by q rows using ggml_vec_dot_f32

    দশমিক scale         = *((দশমিক *) dst->op_params + 0);
    দশমিক max_bias      = *((দশমিক *) dst->op_params + 1);
    দশমিক logit_softcap = *((দশমিক *) dst->op_params + 2);

    যদি (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    ধ্রুবক uint32_t n_শীর্ষ      = neq2;
    ধ্রুবক uint32_t n_শীর্ষ_log2 = 1u << (uint32_t) floor(log2(n_শীর্ষ));

    ধ্রুবক দশমিক m0 = powf(2.0f, -(max_bias) / n_শীর্ষ_log2);
    ধ্রুবক দশমিক m1 = powf(2.0f, -(max_bias / 2.0f) / n_শীর্ষ_log2);

    ধ্রুবক পূর্ণসংখ্যা KV_row_আকার = DK * আকার(_Float16) + DV * আকার(_Float16);

    পূর্ণসংখ্যা ith     = params->ith;
    পূর্ণসংখ্যা ir_step = 1;
    জন্য (পূর্ণসংখ্যা ir = ir0; ir < ir1; ir += ir_step) {
        // q indices
        ধ্রুবক পূর্ণসংখ্যা iq3 = ir / (neq2 * neq1);
        ধ্রুবক পূর্ণসংখ্যা iq2 = (ir - iq3 * neq2 * neq1) / neq1;
        ধ্রুবক পূর্ণসংখ্যা iq1 = (ir - iq3 * neq2 * neq1 - iq2 * neq1);

        ধ্রুবক পূর্ণসংখ্যা iq3_1 = (ir + 1) / (neq2 * neq1);
        ধ্রুবক পূর্ণসংখ্যা iq2_1 = (ir + 1 - iq3_1 * neq2 * neq1) / neq1;
        ধ্রুবক পূর্ণসংখ্যা iq1_1 = (ir + 1 - iq3_1 * neq2 * neq1 - iq2_1 * neq1);

        ধ্রুবক পূর্ণসংখ্যা iq3_2 = (ir + 2) / (neq2 * neq1);
        ধ্রুবক পূর্ণসংখ্যা iq2_2 = (ir + 2 - iq3_2 * neq2 * neq1) / neq1;
        ধ্রুবক পূর্ণসংখ্যা iq1_2 = (ir + 2 - iq3_2 * neq2 * neq1 - iq2_2 * neq1);

        ধ্রুবক পূর্ণসংখ্যা iq3_3 = (ir + 3) / (neq2 * neq1);
        ধ্রুবক পূর্ণসংখ্যা iq2_3 = (ir + 3 - iq3_3 * neq2 * neq1) / neq1;
        ধ্রুবক পূর্ণসংখ্যা iq1_3 = (ir + 3 - iq3_3 * neq2 * neq1 - iq2_3 * neq1);

        ধ্রুবক uint32_t h = iq2;  // head index
        ধ্রুবক দশমিক    slope =
            (max_bias > 0.0f) ? h < n_শীর্ষ_log2 ? powf(m0, h + 1) : powf(m1, 2 * (h - n_শীর্ষ_log2) + 1) : 1.0f;

        ধ্রুবক ggml_fp16_t * mp =
            mask ? (ggml_fp16_t *) ((অক্ষর *) mask->তথ্য + iq1 * mask->nb[1] + (iq2 % mask->ne[2]) * mask->nb[2] +
                                    (iq3 % mask->ne[3]) * mask->nb[3]) :
                   ফাঁকা;

        ধ্রুবক যৌক্তিক mp_equal_2 = iq1_1 == iq1 && (iq2 % mask->ne[2]) == (iq2_1 % mask->ne[2]) &&
                                (iq3 % mask->ne[3]) == (iq3_1 % mask->ne[3]);

        ধ্রুবক যৌক্তিক mp_equal_4 = mp_equal_2 && iq1_2 == iq1 && (iq2 % mask->ne[2]) == (iq2_2 % mask->ne[2]) &&
                                (iq3 % mask->ne[3]) == (iq3_2 % mask->ne[3]) && iq1_3 == iq1 &&
                                (iq2 % mask->ne[2]) == (iq2_3 % mask->ne[2]) &&
                                (iq3 % mask->ne[3]) == (iq3_3 % mask->ne[3]);

        // k indices
        ধ্রুবক পূর্ণসংখ্যা ik3 = iq3 / rk3;
        ধ্রুবক পূর্ণসংখ্যা ik2 = iq2 / rk2;

        ধ্রুবক পূর্ণসংখ্যা ik3_1 = iq3_1 / rk3;
        ধ্রুবক পূর্ণসংখ্যা ik2_1 = iq2_1 / rk2;

        ধ্রুবক পূর্ণসংখ্যা ik3_2 = iq3_2 / rk3;
        ধ্রুবক পূর্ণসংখ্যা ik2_2 = iq2_2 / rk2;

        ধ্রুবক পূর্ণসংখ্যা ik3_3 = iq3_3 / rk3;
        ধ্রুবক পূর্ণসংখ্যা ik2_3 = iq2_3 / rk2;

        // v indices
        ধ্রুবক পূর্ণসংখ্যা iv3 = iq3 / rv3;
        ধ্রুবক পূর্ণসংখ্যা iv2 = iq2 / rv2;

        ধ্রুবক পূর্ণসংখ্যা iv3_1 = iq3_1 / rv3;
        ধ্রুবক পূর্ণসংখ্যা iv2_1 = iq2_1 / rv2;

        ধ্রুবক পূর্ণসংখ্যা iv3_2 = iq3_2 / rv3;
        ধ্রুবক পূর্ণসংখ্যা iv2_2 = iq2_2 / rv2;

        ধ্রুবক পূর্ণসংখ্যা iv3_3 = iq3_3 / rv3;
        ধ্রুবক পূর্ণসংখ্যা iv2_3 = iq2_3 / rv2;

        ধ্রুবক দশমিক * pq = (ধ্রুবক দশমিক *) ((অক্ষর *) q->তথ্য + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3));

        std::array<দশমিক *, 4> pq_বাফার;
        std::array<দশমিক *, 4> sinks_বাফার;
        std::array<দশমিক *, 4> dst_বাফার;

        যদি (tcm_বাফার != nullptr && 4 * KV_row_আকার < tcm_বাফার_আকার && ir < (ir1 - 3) && mp_equal_4 &&
            ik3_3 == ik3 && ik2_3 == ik2 && iv3_3 == iv3 && iv2_3 == iv2 && ik3_2 == ik3 && ik2_2 == ik2 &&
            iv3_2 == iv3 && iv2_2 == iv2 && ik3_1 == ik3 && ik2_1 == ik2 && iv3_1 == iv3 && iv2_1 == iv2) {
            ir_step = 4;

            pq_বাফার[0] = (দশমিক *) ((অক্ষর *) q->তথ্য + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3));
            pq_বাফার[1] = (দশমিক *) ((অক্ষর *) q->তথ্য + (iq1_1 * nbq1 + iq2_1 * nbq2 + iq3_1 * nbq3));
            pq_বাফার[2] = (দশমিক *) ((অক্ষর *) q->তথ্য + (iq1_2 * nbq1 + iq2_2 * nbq2 + iq3_2 * nbq3));
            pq_বাফার[3] = (দশমিক *) ((অক্ষর *) q->তথ্য + (iq1_3 * nbq1 + iq2_3 * nbq2 + iq3_3 * nbq3));

            sinks_বাফার[0] = sinks ? ((দশমিক *) ((অক্ষর *) sinks->তথ্য)) + iq2 : nullptr;
            sinks_বাফার[1] = sinks ? ((দশমিক *) ((অক্ষর *) sinks->তথ্য)) + iq2_1 : nullptr;
            sinks_বাফার[2] = sinks ? ((দশমিক *) ((অক্ষর *) sinks->তথ্য)) + iq2_2 : nullptr;
            sinks_বাফার[3] = sinks ? ((দশমিক *) ((অক্ষর *) sinks->তথ্য)) + iq2_3 : nullptr;

            dst_বাফার[0] = (দশমিক *) ((অক্ষর *) dst->তথ্য + (iq3 * ne2 * ne1 + iq2 + iq1 * ne1) * nb1);
            dst_বাফার[1] = (দশমিক *) ((অক্ষর *) dst->তথ্য + (iq3_1 * ne2 * ne1 + iq2_1 + iq1_1 * ne1) * nb1);
            dst_বাফার[2] = (দশমিক *) ((অক্ষর *) dst->তথ্য + (iq3_2 * ne2 * ne1 + iq2_2 + iq1_2 * ne1) * nb1);
            dst_বাফার[3] = (দশমিক *) ((অক্ষর *) dst->তথ্য + (iq3_3 * ne2 * ne1 + iq2_3 + iq1_3 * ne1) * nb1);

            flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_mrow<4>(  //
                pq_বাফার.তথ্য(),                                      //
                (ধ্রুবক অক্ষর *) k->তথ্য + (ik2 * nbk2 + ik3 * nbk3),    //
                (ধ্রুবক অক্ষর *) v->তথ্য + (iv2 * nbv2 + iv3 * nbv3),    //
                mp,                                                    //
                sinks_বাফার.তথ্য(),                                   //
                dst_বাফার.তথ্য(),                                     //
                scale, logit_softcap, slope, nek1, nbk1, nbv1, DV, DK, tcm_বাফার, tcm_বাফার_আকার);
        } অন্যথা যদি (tcm_বাফার != nullptr && 2 * KV_row_আকার < tcm_বাফার_আকার && ir < (ir1 - 1) && mp_equal_2 &&
                   ik3_1 == ik3 && ik2_1 == ik2 && iv3_1 == iv3 && iv2_1 == iv2) {
            ir_step = 2;

            pq_বাফার[0] = (দশমিক *) ((অক্ষর *) q->তথ্য + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3));
            pq_বাফার[1] = (দশমিক *) ((অক্ষর *) q->তথ্য + (iq1_1 * nbq1 + iq2_1 * nbq2 + iq3_1 * nbq3));

            sinks_বাফার[0] = sinks ? ((দশমিক *) ((অক্ষর *) sinks->তথ্য)) + iq2 : nullptr;
            sinks_বাফার[1] = sinks ? ((দশমিক *) ((অক্ষর *) sinks->তথ্য)) + iq2_1 : nullptr;

            dst_বাফার[0] = (দশমিক *) ((অক্ষর *) dst->তথ্য + (iq3 * ne2 * ne1 + iq2 + iq1 * ne1) * nb1);
            dst_বাফার[1] = (দশমিক *) ((অক্ষর *) dst->তথ্য + (iq3_1 * ne2 * ne1 + iq2_1 + iq1_1 * ne1) * nb1);

            flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_mrow<2>(  //
                pq_বাফার.তথ্য(),                                      //
                (ধ্রুবক অক্ষর *) k->তথ্য + (ik2 * nbk2 + ik3 * nbk3),    //
                (ধ্রুবক অক্ষর *) v->তথ্য + (iv2 * nbv2 + iv3 * nbv3),    //
                mp,                                                    //
                sinks_বাফার.তথ্য(),                                   //
                dst_বাফার.তথ্য(),                                     //
                scale, logit_softcap, slope, nek1, nbk1, nbv1, DV, DK, tcm_বাফার, tcm_বাফার_আকার);
        } অন্যথা {
            ir_step = 1;
            flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_m1(                             //
                pq,                                                                          //
                (ধ্রুবক অক্ষর *) k->তথ্য + (ik2 * nbk2 + ik3 * nbk3),                          //
                (ধ্রুবক অক্ষর *) v->তথ্য + (iv2 * nbv2 + iv3 * nbv3),                          //
                mp,                                                                          //
                sinks ? ((দশমিক *) ((অক্ষর *) sinks->তথ্য)) + h : nullptr,                    //
                (দশমিক *) ((অক্ষর *) dst->তথ্য + (iq3 * ne2 * ne1 + iq2 + iq1 * ne1) * nb1),  //
                scale, logit_softcap, slope, nek1, nbk1, nbv1, DV, DK);
        }
    }
}

শূন্য forward_flash_attn_ext_f16_tiled_vlen1024_vf16(ধ্রুবক ggml_compute_params * params,
                                                    ggml_tensor *               dst,
                                                    পূর্ণসংখ্যা                         ir0,
                                                    পূর্ণসংখ্যা                         ir1,
                                                    শূন্য *                      tcm_বাফার,
                                                    size_t                      tcm_বাফার_আকার) {
    ধ্রুবক ggml_tensor * q     = dst->src[0];
    ধ্রুবক ggml_tensor * k     = dst->src[1];
    ধ্রুবক ggml_tensor * v     = dst->src[2];
    ধ্রুবক ggml_tensor * mask  = dst->src[3];
    ধ্রুবক ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q, ne)
    GGML_TENSOR_LOCALS(size_t, nbq, q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k, ne)
    GGML_TENSOR_LOCALS(size_t, nbk, k, nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v, ne)
    GGML_TENSOR_LOCALS(size_t, nbv, v, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t, nb, dst, nb)

    ধ্রুবক int64_t DK = nek0;
    ধ্রুবক int64_t DV = nev0;
    ধ্রুবক int64_t N  = neq1;

    GGML_ASSERT(flash_attn_ext_supported_shape_vlen1024_vf16(DK, DV));

    GGML_ASSERT(ne0 == DV);
    GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_আকার(q->type));
    GGML_ASSERT(nbk0 == ggml_type_আকার(k->type));
    GGML_ASSERT(nbv0 == ggml_type_আকার(v->type));

    GGML_ASSERT(neq0 == DK);
    GGML_ASSERT(nek0 == DK);
    GGML_ASSERT(nev0 == DV);

    GGML_ASSERT(neq1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == আকার(দশমিক));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(k->type == v->type);
    ধ্রুবক ggml_type kv_type = k->type;

    // broadcast factors
    ধ্রুবক int64_t rk2 = neq2 / nek2;
    ধ্রুবক int64_t rk3 = neq3 / nek3;

    ধ্রুবক int64_t rv2 = neq2 / nev2;
    ধ্রুবক int64_t rv3 = neq3 / nev3;

    দশমিক * param_list    = (দশমিক *) dst->op_params;
    দশমিক   scale         = param_list[0];
    দশমিক   max_bias      = param_list[1];
    দশমিক   logit_softcap = param_list[2];

    যদি (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    ধ্রুবক uint32_t n_শীর্ষ      = neq2;
    ধ্রুবক uint32_t n_শীর্ষ_log2 = 1u << (uint32_t) floor(log2(n_শীর্ষ));

    ধ্রুবক দশমিক m0 = powf(2.0f, -(max_bias) / n_শীর্ষ_log2);
    ধ্রুবক দশমিক m1 = powf(2.0f, -(max_bias / 2.0f) / n_শীর্ষ_log2);

    পূর্ণসংখ্যা ith = params->ith;

    স্থির constexpr পূর্ণসংখ্যা Q_TILE_SZ  = ggml_fa_tile_config::Q;
    স্থির constexpr পূর্ণসংখ্যা KV_TILE_SZ = ggml_fa_tile_config::KV;

    // Per-thread scratch layout:
    // Q_f32:   Q_TILE_SZ * DK
    // KQ:      Q_TILE_SZ * KV_TILE_SZ
    // mask32:  Q_TILE_SZ * KV_TILE_SZ
    // VKQ32:   Q_TILE_SZ * DV
    // V32:     KV_TILE_SZ * DV
    // K_f32:   DK * KV_TILE_SZ (transposed K tile)
    দশমিক *      base = (দশমিক *) params->wdata + ith * (Q_TILE_SZ * DK + 2 * Q_TILE_SZ * KV_TILE_SZ + Q_TILE_SZ * DV +
                                                    KV_TILE_SZ * DV + KV_TILE_SZ * DK + CACHE_সারি_আকার_F32);
    ধ্রুবক size_t base_আকার =
        (Q_TILE_SZ * DK + 2 * Q_TILE_SZ * KV_TILE_SZ + Q_TILE_SZ * DV + KV_TILE_SZ * DV + KV_TILE_SZ * DK) *
            আকার(দশমিক) +
        CACHE_সারি_আকার_F32;

    যদি (base_আকার <= tcm_বাফার_আকার && tcm_বাফার != nullptr) {
        base = (দশমিক *) tcm_বাফার;
    }

    দশমিক   S_M_Buf[Q_TILE_SZ * 2];  // buffer to hold S, M, bias for one tile to reduce register pressure in main loop
    দশমিক * S = S_M_Buf;
    দশমিক * M = S_M_Buf + Q_TILE_SZ;

    পূর্ণসংখ্যা ir = ir0;
    যখন (ir < ir1) {
        // q indices for the start of this tile
        ধ্রুবক পূর্ণসংখ্যা iq3 = ir / (neq2 * neq1);
        ধ্রুবক পূর্ণসংখ্যা iq2 = (ir - iq3 * neq2 * neq1) / neq1;
        ধ্রুবক পূর্ণসংখ্যা iq1 = (ir - iq3 * neq2 * neq1 - iq2 * neq1);

        // Number of valid rows in this tile:
        // - limited by tile size (Q_TILE_SZ)
        // - limited by chunk boundary (ir1 - ir)
        // - limited by head boundary (neq1 - iq1) to avoid crossing into next head
        ধ্রুবক পূর্ণসংখ্যা tile_rows = MIN(Q_TILE_SZ, MIN((পূর্ণসংখ্যা) (ir1 - ir), (পূর্ণসংখ্যা) (neq1 - iq1)));
        GGML_ASSERT(tile_rows > 0);

        ধ্রুবক uint32_t h = iq2;  // head index
        ধ্রুবক দশমিক    slope =
            (max_bias > 0.0f) ? h < n_শীর্ষ_log2 ? powf(m0, h + 1) : powf(m1, 2 * (h - n_শীর্ষ_log2) + 1) : 1.0f;

        জন্য (পূর্ণসংখ্যা i = 0; i < Q_TILE_SZ; ++i) {
            S[i] = 0.;
            M[i] = -INFINITY;
        }

        দশমিক *    Q_f32  = base;
        দশমিক *    KQ     = (দশমিক *) ((অক্ষর *) base + Q_TILE_SZ * DK * আকার(দশমিক));
        দশমিক *    mask32 = KQ + Q_TILE_SZ * KV_TILE_SZ;
        দশমিক *    VKQ32  = mask32 + Q_TILE_SZ * KV_TILE_SZ;
        দশমিক *    V32    = VKQ32 + Q_TILE_SZ * DV;
        দশমিক *    K_f32  = V32 + KV_TILE_SZ * DV;
        _Float16 * Q_f16  = (_Float16 *) Q_f32;
        _Float16 * V_f16  = (_Float16 *) V32;
        _Float16 * K_f16  = (_Float16 *) K_f32;

        rvv_zero_f32(VKQ32, Q_TILE_SZ * DV);

        // k indices
        ধ্রুবক পূর্ণসংখ্যা ik3 = iq3 / rk3;
        ধ্রুবক পূর্ণসংখ্যা ik2 = iq2 / rk2;

        // v indices
        ধ্রুবক পূর্ণসংখ্যা iv3 = iq3 / rv3;
        ধ্রুবক পূর্ণসংখ্যা iv2 = iq2 / rv2;

        ধ্রুবক দশমিক * pq = (ধ্রুবক দশমিক *) ((অক্ষর *) q->তথ্য + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3));
        যদি (kv_type == GGML_TYPE_F16) {
            rvv_pack_f32_as_scaled_f16((uint8_t *) Q_f16, DK * আকার(_Float16), (uint8_t *) pq, nbq1, tile_rows, DK,
                                       scale);
        } অন্যথা {
            memcpy2d(Q_f32, DK * আকার(দশমিক), pq, nbq1, tile_rows, DK * আকার(দশমিক));
        }

        জন্য (int64_t ic = 0; ic < nek1; ic += KV_TILE_SZ) {
            ধ্রুবক পূর্ণসংখ্যা kv_tile = (পূর্ণসংখ্যা) std::min((int64_t) KV_TILE_SZ, nek1 - ic);

            rvv_zero_f32(K_f32, DK * KV_TILE_SZ);
            rvv_zero_f32(V32, KV_TILE_SZ * DV);

            // skip the tile entirely if all the masks are -inf
            যদি (mask) {
                যৌক্তিক                can_skip = সত্য;
                ধ্রুবক ggml_fp16_t * mp_row =
                    (ধ্রুবক ggml_fp16_t *) ((ধ্রুবক অক্ষর *) mask->তথ্য + iq1 * mask->nb[1] +
                                           (iq2 % mask->ne[2]) * mask->nb[2] + (iq3 % mask->ne[3]) * mask->nb[3]);
                rvv_pack_scaled_f16_as_f32(mask32, KV_TILE_SZ * আকার(দশমিক), mp_row + ic, mask->nb[1], tile_rows,
                                           kv_tile, slope);

                জন্য (পূর্ণসংখ্যা tq = 0; tq < tile_rows; tq++) {
                    জন্য (পূর্ণসংখ্যা tk = 0; tk < kv_tile; tk++) {
                        যদি (mask32[tq * KV_TILE_SZ + tk] != -INFINITY) {
                            can_skip = মিথ্যা;
                        }
                    }
                    // Pad remaining mask entries with -inf
                    জন্য (পূর্ণসংখ্যা tk = kv_tile; tk < KV_TILE_SZ; tk++) {
                        mask32[tq * KV_TILE_SZ + tk] = -INFINITY;
                    }
                }

                যদি (can_skip) {
                    চালু;
                }
            }

            যদি (kv_type == GGML_TYPE_F16) {
                rvv_transposed_s16_mn_to_nm((int8_t *) K_f16, KV_TILE_SZ * আকার(_Float16),
                                            (int8_t *) k->তথ্য + ic * nbk1 + ik2 * nbk2 + ik3 * nbk3, nbk1, kv_tile,
                                            DK);

                পূর্ণসংখ্যা tq = 0;
                জন্য (; tq + 3 < tile_rows; tq += 4) {
                    rvv_qk_dot_tile_f16_x4(KQ + (tq + 0) * KV_TILE_SZ, KQ + (tq + 1) * KV_TILE_SZ,
                                           KQ + (tq + 2) * KV_TILE_SZ, KQ + (tq + 3) * KV_TILE_SZ,
                                           Q_f16 + (tq + 0) * DK, Q_f16 + (tq + 1) * DK, Q_f16 + (tq + 2) * DK,
                                           Q_f16 + (tq + 3) * DK, K_f16, DK, kv_tile);
                }
                জন্য (; tq < tile_rows; ++tq) {
                    rvv_qk_dot_tile_f16_x1(KQ + tq * KV_TILE_SZ, Q_f16 + tq * DK, K_f16, DK, kv_tile);
                }
            } অন্যথা {
                জন্য (পূর্ণসংখ্যা tk = 0; tk < kv_tile; tk++) {
                    ধ্রুবক অক্ষর *  k_তথ্য = (ধ্রুবক অক্ষর *) k->তথ্য + (ic + tk) * nbk1 + ik2 * nbk2 + ik3 * nbk3;
                    দশমিক *       k_col  = K_f32 + tk;
                    ধ্রুবক দশমিক * k_src  = (ধ্রুবক দশমিক *) k_তথ্য;
                    জন্য (int64_t dk = 0; dk < DK; ++dk) {
                        k_col[dk * KV_TILE_SZ] = k_src[dk];
                    }
                }

                জন্য (পূর্ণসংখ্যা tq = 0; tq < tile_rows; ++tq) {
                    rvv_qk_dot_tile(KQ + tq * KV_TILE_SZ, Q_f32 + tq * DK, K_f32, DK, KV_TILE_SZ, scale);
                }
            }

            // Set padded KQ entries to -inf so softmax gives them zero weight
            যদি (kv_tile < KV_TILE_SZ) {
                জন্য (পূর্ণসংখ্যা tq = 0; tq < tile_rows; tq++) {
                    জন্য (পূর্ণসংখ্যা tk = kv_tile; tk < KV_TILE_SZ; tk++) {
                        KQ[tq * KV_TILE_SZ + tk] = -INFINITY;
                    }
                }
            }

            যদি (logit_softcap != 0.0f) {
                rvv_softcap_tanh_inplace_f32(KQ, KV_TILE_SZ, tile_rows, KV_TILE_SZ, logit_softcap);
            }

            যদি (mask) {
                rvv_add_inplace_f32(KQ, KV_TILE_SZ, mask32, KV_TILE_SZ, tile_rows, KV_TILE_SZ);
            }

            যৌক্তিক skip[Q_TILE_SZ] = {};

            জন্য (পূর্ণসংখ্যা tq = 0; tq < tile_rows; tq++) {
                দশমিক * kq_row = KQ + tq * KV_TILE_SZ;

                ধ্রুবক দশমিক tile_max = rvv_max_f32(kq_row, KV_TILE_SZ);

                যদি (tile_max == -INFINITY) {
                    skip[tq] = সত্য;
                    চালু;
                }

                ধ্রুবক দশমিক Mold = M[tq];
                ধ্রুবক দশমিক Mnew = fmaxf(Mold, tile_max);

                যদি (Mnew > Mold) {
                    ধ্রুবক দশমিক ms = expf(Mold - Mnew);
                    rvv_scale_f32(VKQ32 + tq * DV, ms, DV);
                    S[tq] *= ms;
                }
                M[tq] = Mnew;

                S[tq] += rvv_softmax_exp_inplace_f32(kq_row, KV_TILE_SZ, Mnew);
            }

            // Pack V as contiguous [KV_TILE_SZ][DV].
            যদি (kv_type == GGML_TYPE_F16) {
                ধ্রুবক অক্ষর * v_তথ্য = (ধ্রুবক অক্ষর *) v->তথ্য + ic * nbv1 + iv2 * nbv2 + iv3 * nbv3;
                memcpy2d(V_f16, DV * আকার(_Float16), v_তথ্য, nbv1, kv_tile, DV * আকার(_Float16));

                পূর্ণসংখ্যা tq = 0;
                জন্য (; tq + 3 < tile_rows; tq += 4) {
                    যদি (skip[tq + 0] || skip[tq + 1] || skip[tq + 2] || skip[tq + 3]) {
                        জন্য (পূর্ণসংখ্যা i = 0; i < 4; ++i) {
                            যদি (!skip[tq + i]) {
                                rvv_pv_accumulate_f16_x1(VKQ32 + (tq + i) * DV, KQ + (tq + i) * KV_TILE_SZ, V_f16,
                                                         KV_TILE_SZ, DV);
                            }
                        }
                        চালু;
                    }

                    rvv_pv_accumulate_f16_x4(VKQ32 + (tq + 0) * DV, VKQ32 + (tq + 1) * DV, VKQ32 + (tq + 2) * DV,
                                             VKQ32 + (tq + 3) * DV, KQ + (tq + 0) * KV_TILE_SZ,
                                             KQ + (tq + 1) * KV_TILE_SZ, KQ + (tq + 2) * KV_TILE_SZ,
                                             KQ + (tq + 3) * KV_TILE_SZ, V_f16, KV_TILE_SZ, DV);
                }
                জন্য (; tq < tile_rows; ++tq) {
                    যদি (!skip[tq]) {
                        rvv_pv_accumulate_f16_x1(VKQ32 + tq * DV, KQ + tq * KV_TILE_SZ, V_f16, KV_TILE_SZ, DV);
                    }
                }
            } অন্যথা {
                ধ্রুবক অক্ষর * v_তথ্য = (ধ্রুবক অক্ষর *) v->তথ্য + ic * nbv1 + iv2 * nbv2 + iv3 * nbv3;
                memcpy2d(V32, DV * আকার(দশমিক), v_তথ্য, nbv1, kv_tile, DV * আকার(দশমিক));

                জন্য (পূর্ণসংখ্যা tq = 0; tq < tile_rows; ++tq) {
                    যদি (!skip[tq]) {
                        rvv_pv_accumulate(VKQ32 + tq * DV, KQ + tq * KV_TILE_SZ, V32, KV_TILE_SZ, DV);
                    }
                }
            }
        }

        // sinks (apply only to valid rows in the tile)
        যদি (sinks) {
            ধ্রুবক দশমিক s = ((দশমিক *) ((অক্ষর *) sinks->তথ্য))[h];

            জন্য (পূর্ণসংখ্যা tq = 0; tq < tile_rows; tq++) {
                দশমিক ms = 1.0f;
                দশমিক vs = 1.0f;

                যদি (s > M[tq]) {
                    ms = expf(M[tq] - s);
                    rvv_scale_f32(VKQ32 + tq * DV, ms, DV);
                } অন্যথা {
                    vs = expf(s - M[tq]);
                }

                দশমিক S_অস্থায়ী = S[tq] * ms + vs;
                S[tq]        = S_অস্থায়ী == 0.0f ? 0.0f : 1.0f / S_অস্থায়ী;
            }
        } অন্যথা {
            জন্য (পূর্ণসংখ্যা tq = 0; tq < tile_rows; tq++) {
                ধ্রুবক দশমিক S_inv = S[tq] == 0.0f ? 0.0f : 1.0f / S[tq];
                S[tq]             = S_inv;
            }
        }

        দশমিক * dst_ptr = (দশমিক *) ((অক্ষর *) dst->তথ্য + (iq3 * ne2 * ne1 + iq2 + (iq1) *ne1) * nb1);
        rvv_pack_scaled_f32_as_f32(dst_ptr, nb1 * ne1, VKQ32, DV * আকার(দশমিক), tile_rows, DV, S);

        ir += tile_rows;
    }
}

শূন্য forward_rms_norm_f32(ggml_compute_params * params, ggml_tensor * op) {
    ধ্রুবক ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(src0->nb[0] == আকার(দশমিক));

    পূর্ণসংখ্যা ith = params->ith;
    পূর্ণসংখ্যা nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    দশমিক epsilon = *((দশমিক *) dst->op_params);

    GGML_ASSERT(epsilon > 0.0f);

    auto * প্রবেশ  = (অক্ষর *) src0->তথ্য;
    auto * output = (অক্ষর *) dst->তথ্য;

    ধ্রুবক auto hidden_আকার     = ne00;
    ধ্রুবক auto task_গুনতি      = ne01 * ne02 * ne03;
    ধ্রুবক auto task_per_thread = (task_গুনতি + nth - 1) / nth;

    ধ্রুবক auto task_begin = ith * task_per_thread;
    ধ্রুবক auto task_end   = std::min((ith + 1) * task_per_thread, task_গুনতি);

    জন্য (auto task_idx = task_begin; task_idx < task_end; task_idx++) {
        int64_t i03 = task_idx / (ne02 * ne01);
        int64_t i02 = (task_idx - i03 * ne02 * ne01) / ne01;
        int64_t i01 = (task_idx - i03 * ne02 * ne01 - i02 * ne01);

        auto * p_প্রবেশ       = (দশমিক *) (প্রবেশ + i01 * nb01 + i02 * nb02 + i03 * nb03);
        auto * p_output      = (দশমিক *) (output + i01 * nb1 + i02 * nb2 + i03 * nb3);
        auto * p_অস্থায়ী_output = p_output;

        size_t       gvl    = __riscv_vsetvlmax_e32m4();
        vfloat32m4_t যোগ_sq = __riscv_vfmv_v_f_f32m4(0.f, gvl);
        int64_t      দৈর্ঘ্য = hidden_আকার;
        যখন (দৈর্ঘ্য > 0) {
            gvl                   = __riscv_vsetvl_e32m4(দৈর্ঘ্য);
            vfloat32m4_t src_তথ্য = __riscv_vle32_v_f32m4(p_প্রবেশ, gvl);
            যোগ_sq                = __riscv_vfmacc_vv_f32m4(যোগ_sq, src_তথ্য, src_তথ্য, gvl);
            __riscv_vse32_v_f32m4(p_অস্থায়ী_output, src_তথ্য, gvl);

            p_প্রবেশ += gvl;
            p_অস্থায়ী_output += gvl;
            দৈর্ঘ্য -= gvl;
        }

        gvl                 = __riscv_vsetvlmax_e32m1();
        vfloat32m1_t zero_v = __riscv_vfmv_v_f_f32m1(0.f, gvl);
        vfloat32m1_t mean_square_v =
            __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(যোগ_sq, 0), __riscv_vget_v_f32m4_f32m1(যোগ_sq, 1), gvl);

        mean_square_v = __riscv_vfadd_vv_f32m1(mean_square_v, __riscv_vget_v_f32m4_f32m1(যোগ_sq, 2), gvl);
        mean_square_v = __riscv_vfadd_vv_f32m1(mean_square_v, __riscv_vget_v_f32m4_f32m1(যোগ_sq, 3), gvl);
        mean_square_v = __riscv_vfredusum_vs_f32m1_f32m1(mean_square_v, zero_v, gvl);

        দশমিক mean_square = __riscv_vfmv_f_s_f32m1_f32(mean_square_v);
        mean_square /= hidden_আকার;

        mean_square = sqrt(mean_square + epsilon);

        mean_square   = 1.0f / mean_square;
        দৈর্ঘ্য        = hidden_আকার;
        p_অস্থায়ী_output = p_output;

        যখন (দৈর্ঘ্য > 0) {
            gvl                   = __riscv_vsetvl_e32m4(দৈর্ঘ্য);
            vfloat32m4_t src_তথ্য = __riscv_vle32_v_f32m4(p_অস্থায়ী_output, gvl);
            src_তথ্য              = __riscv_vfmul_vf_f32m4(src_তথ্য, mean_square, gvl);
            __riscv_vse32_v_f32m4(p_output, src_তথ্য, gvl);
            p_অস্থায়ী_output += gvl;
            p_output += gvl;
            দৈর্ঘ্য -= gvl;
        }
    }
}

template <size_t MB_ROWS>
শূন্য quantize_a_nrow_i8_ref(size_t blk_len, ধ্রুবক দশমিক * a_ptr, size_t গুনতি_k, uint8_t * quant_a_ptr) {
    int64_t a_blk_stride        = q8_blk_আকার(blk_len, সত্য);
    int64_t a_nrow_block_stride = a_blk_stride * MB_ROWS;
    জন্য (size_t k = 0; k < গুনতি_k; k += blk_len, quant_a_ptr += a_nrow_block_stride) {
        দশমিক *   scale_a_ptr = reinterpret_cast<দশমিক *>(quant_a_ptr);
        int16_t * a_যোগ_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + আকার(দশমিক) * MB_ROWS);
        int8_t *  quant_a_blk =
            reinterpret_cast<int8_t *>(quant_a_ptr + আকার(দশমিক) * MB_ROWS + আকার(int16_t) * MB_ROWS);

        জন্য (size_t row = 0; row < MB_ROWS; row++) {
            দশমিক max_abs_a = 0.0f;
            জন্য (size_t bk = 0; bk < blk_len; bk++) {
                max_abs_a = std::max(max_abs_a, std::abs(a_ptr[row * গুনতি_k + k + bk]));
            }

            দশমিক rep_scale_a = ((1 << 7) - 1) / max_abs_a;
            scale_a_ptr[row]  = 1 / rep_scale_a;

            int16_t a_যোগ = 0;
            জন্য (size_t bk = 0; bk < blk_len; bk++) {
                ধ্রুবক int8_t quantized = static_cast<int8_t>(
                    std::clamp(std::nearbyintf(a_ptr[row * গুনতি_k + k + bk] * rep_scale_a), -128.0f, 127.0f));
                quant_a_blk[row * blk_len + bk] = quantized;
                a_যোগ += quantized;
            }
            a_যোগ_ptr[row] = -a_যোগ;
        }
    }
}

template <size_t MB_ROWS>
শূন্য quantize_a_nrow_i8_hp_ref(size_t blk_len, ধ্রুবক দশমিক * a_ptr, size_t গুনতি_k, uint8_t * quant_a_ptr) {
    constexpr size_t k_subblk_len = 32;
    ধ্রুবক size_t     subblk_গুনতি = blk_len / k_subblk_len;

    GGML_ASSERT(blk_len == 256);

    দশমিক   scale_অস্থায়ী[8]       = { 0.0f };
    int64_t a_blk_stride        = q8_hp_blk_আকার(blk_len, সত্য, সত্য);
    int64_t a_nrow_block_stride = a_blk_stride * MB_ROWS;
    int64_t a_subblk_stride     = q8_hp_blk_আকার(k_subblk_len, মিথ্যা, মিথ্যা) * MB_ROWS;

    জন্য (size_t k = 0; k < গুনতি_k; k += blk_len, quant_a_ptr += a_nrow_block_stride) {
        _Float16 * a_যোগ_ptr = reinterpret_cast<_Float16 *>(quant_a_ptr + a_subblk_stride * subblk_গুনতি);

        দশমিক scale_avg = 0.0f;
        জন্য (size_t kk = 0; kk < subblk_গুনতি; kk++) {
            দশমিক max_abs_a = 0.0f;
            জন্য (size_t row = 0; row < MB_ROWS; row++) {
                জন্য (size_t bk = 0; bk < k_subblk_len; bk++) {
                    max_abs_a = std::max(max_abs_a, std::abs(a_ptr[row * গুনতি_k + k + bk + kk * k_subblk_len]));
                }
            }
            scale_অস্থায়ী[kk] = max_abs_a / ((1 << 7) - 1);
            scale_avg += scale_অস্থায়ী[kk];
        }

        scale_avg /= subblk_গুনতি;
        দশমিক scale_factor = 1.0f / scale_avg;

        _Float16 * scale_avg_ptr =
            reinterpret_cast<_Float16 *>(quant_a_ptr + a_nrow_block_stride - আকার(_Float16) * MB_ROWS);
        scale_avg_ptr[0] = scale_avg;

        জন্য (size_t kk = 0; kk < subblk_গুনতি; kk++) {
            uint8_t *  a_subblk_base = quant_a_ptr + kk * a_subblk_stride;
            _Float16 * scale_a_ptr   = reinterpret_cast<_Float16 *>(a_subblk_base);
            int8_t *   quant_a_blk   = reinterpret_cast<int8_t *>(a_subblk_base + আকার(_Float16) * MB_ROWS);

            scale_a_ptr[0] = static_cast<_Float16>(scale_অস্থায়ী[kk] * scale_factor);

            ধ্রুবক দশমিক rep_scale_a = 1.0f / scale_অস্থায়ী[kk];

            জন্য (size_t row = 0; row < MB_ROWS; row++) {
                int16_t a_যোগ = 0;
                জন্য (size_t bk = 0; bk < k_subblk_len; bk++) {
                    ধ্রুবক int8_t quantized = static_cast<int8_t>(
                        std::clamp(std::nearbyintf(a_ptr[row * গুনতি_k + k + bk + kk * k_subblk_len] * rep_scale_a),
                                   -128.0f, 127.0f));
                    quant_a_blk[row * k_subblk_len + bk] = quantized;
                    a_যোগ += quantized;
                }
                a_যোগ_ptr[row * subblk_গুনতি + kk] = static_cast<_Float16>(-a_যোগ) * static_cast<_Float16>(8.0f);
            }
        }
    }
}

template <size_t MB_ROWS>
শূন্য quantize_a_nrow_i8k_ref(size_t blk_len, ধ্রুবক দশমিক * a_ptr, size_t গুনতি_k, uint8_t * quant_a_ptr) {
    int64_t a_blk_stride        = q8k_blk_আকার(256);
    int64_t a_nrow_block_stride = a_blk_stride * MB_ROWS;
    int64_t a_যোগ_আকার          = 256 / 16;

    জন্য (size_t k = 0; k < গুনতি_k; k += blk_len, quant_a_ptr += a_nrow_block_stride) {
        দশমিক *   scale_a_ptr = reinterpret_cast<দশমিক *>(quant_a_ptr);
        int16_t * a_যোগ_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + আকার(দশমিক) * MB_ROWS);
        int8_t *  quant_a_blk =
            reinterpret_cast<int8_t *>(quant_a_ptr + আকার(দশমিক) * MB_ROWS + আকার(int16_t) * a_যোগ_আকার * MB_ROWS);

        জন্য (size_t row = 0; row < MB_ROWS; row++) {
            দশমিক max_a     = 0.0f;
            দশমিক max_abs_a = 0.0f;
            জন্য (size_t bk = 0; bk < blk_len; bk++) {
                দশমিক ax = std::abs(a_ptr[row * গুনতি_k + k + bk]);
                যদি (ax > max_abs_a) {
                    max_abs_a = ax;
                    max_a     = a_ptr[row * গুনতি_k + k + bk];
                }
            }

            যদি (!max_abs_a) {
                scale_a_ptr[row] = 0;
                জন্য (size_t bki = 0; bki < a_যোগ_আকার; bki++) {
                    জন্য (size_t bk = bki * 16; bk < (bki + 1) * 16; bk++) {
                        quant_a_blk[row * blk_len + bk] = 0;
                    }
                    a_যোগ_ptr[row * a_যোগ_আকার + bki] = 0;
                }
                চালু;
            }

            দশমিক rep_scale_a = ((1 << 7) - 1) / max_abs_a;
            scale_a_ptr[row]  = 1 / rep_scale_a;

            জন্য (size_t bki = 0; bki < a_যোগ_আকার; bki++) {
                int16_t a_যোগ = 0;
                জন্য (size_t bk = bki * 16; bk < (bki + 1) * 16; bk++) {
                    ধ্রুবক int8_t quantized = static_cast<int8_t>(
                        std::clamp(std::nearbyintf(a_ptr[row * গুনতি_k + k + bk] * rep_scale_a), -128.0f, 127.0f));
                    quant_a_blk[row * blk_len + bk] = quantized;
                    a_যোগ += quantized;
                }
                a_যোগ_ptr[row * a_যোগ_আকার + bki] = -a_যোগ;
            }
        }
    }
}

শূন্য quantize_a_row_i8(size_t blk_len, ধ্রুবক দশমিক * a_ptr, size_t গুনতি_k, uint8_t * quant_a_ptr) {
    GGML_ASSERT(blk_len == 32);
    int64_t a_blk_stride = q8_blk_আকার(blk_len, সত্য);
    size_t  vlenb        = __riscv_vlenb();

    যদি (vlenb == 128) {
        জন্য (size_t k = 0; k < গুনতি_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            দশমিক *   scale_a_ptr = reinterpret_cast<দশমিক *>(quant_a_ptr);
            int16_t * a_যোগ_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + আকার(দশমিক));
            int8_t *  quant_a_blk = reinterpret_cast<int8_t *>(quant_a_ptr + আকার(দশমিক) + আকার(int16_t));

            size_t       vl      = __riscv_vsetvl_e32m1(blk_len);
            vfloat32m1_t v_a     = __riscv_vle32_v_f32m1(a_ptr + k, vl);
            vfloat32m1_t v_a_abs = __riscv_vfabs_v_f32m1(v_a, vl);

            vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m1_f32m1(v_a_abs, tmp, vl);
            দশমিক        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

            দশমিক scale_a     = max_abs_a / ((1 << 7) - 1);
            দশমিক rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
            scale_a_ptr[0]    = scale_a;

            vfloat32m1_t v_a_scale    = __riscv_vfmul_vf_f32m1(v_a, rep_scale_a, vl);
            vint16mf2_t  v_a_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a_scale, vl);
            vint8mf4_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a_quant, vl);

            vint16m1_t tmp_যোগ_2 = __riscv_vmv_v_x_i16m1(0, vl);
            vint16m1_t v_a_যোগ = __riscv_vwredsum_vs_i8mf4_i16m1(v_a_quant_i8, tmp_যোগ_2, vl);
            int16_t    a_যোগ   = __riscv_vmv_x_s_i16m1_i16(v_a_যোগ);
            a_যোগ_ptr[0]       = -a_যোগ;

            __riscv_vse8_v_i8mf4(quant_a_blk, v_a_quant_i8, vl);
        }
    } অন্যথা যদি (vlenb == 32) {
        জন্য (size_t k = 0; k < গুনতি_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            দশমিক *   scale_a_ptr = reinterpret_cast<দশমিক *>(quant_a_ptr);
            int16_t * a_যোগ_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + আকার(দশমিক));
            int8_t *  quant_a_blk = reinterpret_cast<int8_t *>(quant_a_ptr + আকার(দশমিক) + আকার(int16_t));

            size_t       vl      = __riscv_vsetvl_e32m4(blk_len);
            vfloat32m4_t v_a     = __riscv_vle32_v_f32m4(a_ptr + k, vl);
            vfloat32m4_t v_a_abs = __riscv_vfabs_v_f32m4(v_a, vl);

            vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m4_f32m1(v_a_abs, tmp, vl);
            দশমিক        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

            দশমিক scale_a     = max_abs_a / ((1 << 7) - 1);
            দশমিক rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
            scale_a_ptr[0]    = scale_a;

            vfloat32m4_t v_a_scale    = __riscv_vfmul_vf_f32m4(v_a, rep_scale_a, vl);
            vint16m2_t   v_a_quant    = __riscv_vfncvt_x_f_w_i16m2(v_a_scale, vl);
            vint8m1_t    v_a_quant_i8 = __riscv_vncvt_x_x_w_i8m1(v_a_quant, vl);

            vint16m1_t tmp_যোগ_2 = __riscv_vmv_v_x_i16m1(0, vl);
            vint16m1_t v_a_যোগ = __riscv_vwredsum_vs_i8m1_i16m1(v_a_quant_i8, tmp_যোগ_2, vl);
            int16_t    a_যোগ   = __riscv_vmv_x_s_i16m1_i16(v_a_যোগ);
            a_যোগ_ptr[0]       = -a_যোগ;

            __riscv_vse8_v_i8m1(quant_a_blk, v_a_quant_i8, vl);
        }
    } অন্যথা {
        quantize_a_nrow_i8_ref<1>(blk_len, a_ptr, গুনতি_k, quant_a_ptr);
    }
}

শূন্য quantize_a_4row_i8(size_t blk_len, ধ্রুবক দশমিক * a_ptr, size_t গুনতি_k, uint8_t * quant_a_ptr) {
    GGML_ASSERT(blk_len == 32);
    int64_t a_blk_stride        = q8_blk_আকার(blk_len, সত্য);
    int64_t a_nrow_block_stride = a_blk_stride * 4;
    size_t  vlenb               = __riscv_vlenb();

    যদি (vlenb == 128) {
        জন্য (size_t k = 0; k < গুনতি_k; k += blk_len, quant_a_ptr += a_nrow_block_stride) {
            দশমিক *   scale_a_ptr = reinterpret_cast<দশমিক *>(quant_a_ptr);
            int16_t * a_যোগ_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + আকার(দশমিক) * 4);
            int8_t *  quant_a_blk = reinterpret_cast<int8_t *>(quant_a_ptr + আকার(দশমিক) * 4 + আকার(int16_t) * 4);

            জন্য (size_t mi = 0; mi < 4; mi++) {
                size_t       vl      = __riscv_vsetvl_e32m1(blk_len);
                vfloat32m1_t v_a     = __riscv_vle32_v_f32m1(a_ptr + mi * গুনতি_k + k, vl);
                vfloat32m1_t v_a_abs = __riscv_vfabs_v_f32m1(v_a, vl);

                vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m1_f32m1(v_a_abs, tmp, vl);
                দশমিক        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                দশমিক scale_a     = max_abs_a / ((1 << 7) - 1);
                দশমিক rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
                scale_a_ptr[mi]   = scale_a;

                vfloat32m1_t v_a_scale    = __riscv_vfmul_vf_f32m1(v_a, rep_scale_a, vl);
                vint16mf2_t  v_a_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a_scale, vl);
                vint8mf4_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a_quant, vl);

                vint16m1_t tmp_যোগ_2 = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a_যোগ = __riscv_vwredsum_vs_i8mf4_i16m1(v_a_quant_i8, tmp_যোগ_2, vl);
                int16_t    a_যোগ   = __riscv_vmv_x_s_i16m1_i16(v_a_যোগ);
                a_যোগ_ptr[mi]      = -a_যোগ;

                __riscv_vse8_v_i8mf4(quant_a_blk + mi * blk_len, v_a_quant_i8, vl);
            }
        }
    } অন্যথা যদি (vlenb == 32) {
        জন্য (size_t k = 0; k < গুনতি_k; k += blk_len, quant_a_ptr += a_nrow_block_stride) {
            দশমিক *   scale_a_ptr = reinterpret_cast<দশমিক *>(quant_a_ptr);
            int16_t * a_যোগ_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + আকার(দশমিক) * 4);
            int8_t *  quant_a_blk = reinterpret_cast<int8_t *>(quant_a_ptr + আকার(দশমিক) * 4 + আকার(int16_t) * 4);

            জন্য (size_t mi = 0; mi < 4; mi++) {
                size_t       vl      = __riscv_vsetvl_e32m4(blk_len);
                vfloat32m4_t v_a     = __riscv_vle32_v_f32m4(a_ptr + mi * গুনতি_k + k, vl);
                vfloat32m4_t v_a_abs = __riscv_vfabs_v_f32m4(v_a, vl);

                vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m4_f32m1(v_a_abs, tmp, vl);
                দশমিক        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                দশমিক scale_a     = max_abs_a / ((1 << 7) - 1);
                দশমিক rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
                scale_a_ptr[mi]   = scale_a;

                vfloat32m4_t v_a_scale    = __riscv_vfmul_vf_f32m4(v_a, rep_scale_a, vl);
                vint16m2_t   v_a_quant    = __riscv_vfncvt_x_f_w_i16m2(v_a_scale, vl);
                vint8m1_t    v_a_quant_i8 = __riscv_vncvt_x_x_w_i8m1(v_a_quant, vl);

                vint16m1_t tmp_যোগ_2 = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a_যোগ = __riscv_vwredsum_vs_i8m1_i16m1(v_a_quant_i8, tmp_যোগ_2, vl);
                int16_t    a_যোগ   = __riscv_vmv_x_s_i16m1_i16(v_a_যোগ);
                a_যোগ_ptr[mi]      = -a_যোগ;

                __riscv_vse8_v_i8m1(quant_a_blk + mi * blk_len, v_a_quant_i8, vl);
            }
        }
    } অন্যথা {
        quantize_a_nrow_i8_ref<4>(blk_len, a_ptr, গুনতি_k, quant_a_ptr);
    }
}

শূন্য quantize_a_row_i8_hp(size_t blk_len, ধ্রুবক দশমিক * a_ptr, size_t গুনতি_k, uint8_t * quant_a_ptr) {
    constexpr size_t k_subblk_len = 32;
    GGML_ASSERT(blk_len == 256);

    constexpr size_t subblk_গুনতি             = 256 / k_subblk_len;
    int64_t          a_blk_stride             = q8_hp_blk_আকার(blk_len, সত্য, সত্য);
    int64_t          a_subblk_stride          = q8_hp_blk_আকার(k_subblk_len, মিথ্যা, মিথ্যা);
    size_t           vlenb                    = __riscv_vlenb();
    দশমিক            scale_অস্থায়ী[subblk_গুনতি] = { 0.0f };

    যদি (vlenb == 128) {
        জন্য (size_t k = 0; k < গুনতি_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            _Float16 * a_যোগ_ptr     = reinterpret_cast<_Float16 *>(quant_a_ptr + a_subblk_stride * subblk_গুনতি);
            _Float16 * scale_avg_ptr = reinterpret_cast<_Float16 *>(quant_a_ptr + a_blk_stride - আকার(_Float16));
            দশমিক      scale_avg     = 0.0f;

            জন্য (size_t kk = 0; kk < subblk_গুনতি; ++kk) {
                ধ্রুবক দশমিক * a_src_ptr = a_ptr + k + kk * k_subblk_len;

                size_t       vl      = __riscv_vsetvl_e32m1(k_subblk_len);
                vfloat32m1_t v_a     = __riscv_vle32_v_f32m1(a_src_ptr, vl);
                vfloat32m1_t v_a_abs = __riscv_vfabs_v_f32m1(v_a, vl);

                vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m1_f32m1(v_a_abs, tmp, vl);
                দশমিক        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                scale_অস্থায়ী[kk] = max_abs_a / ((1 << 7) - 1);
                scale_avg += scale_অস্থায়ী[kk];
            }

            scale_avg /= subblk_গুনতি;
            ধ্রুবক দশমিক scale_factor = scale_avg ? 1.0f / scale_avg : 0.0f;
            scale_avg_ptr[0]         = static_cast<_Float16>(scale_avg);

            জন্য (size_t kk = 0; kk < subblk_গুনতি; ++kk) {
                uint8_t *     a_subblk_base = quant_a_ptr + kk * a_subblk_stride;
                _Float16 *    scale_a_ptr   = reinterpret_cast<_Float16 *>(a_subblk_base);
                int8_t *      quant_a_blk   = reinterpret_cast<int8_t *>(a_subblk_base + আকার(_Float16));
                ধ্রুবক দশমিক * a_src_ptr     = a_ptr + k + kk * k_subblk_len;

                size_t       vl          = __riscv_vsetvl_e32m1(k_subblk_len);
                vfloat32m1_t v_a         = __riscv_vle32_v_f32m1(a_src_ptr, vl);
                দশমিক        rep_scale_a = scale_অস্থায়ী[kk] ? 1.0f / scale_অস্থায়ী[kk] : 0.0f;
                scale_a_ptr[0]           = static_cast<_Float16>(scale_অস্থায়ী[kk] * scale_factor);

                vfloat32m1_t v_a_scale    = __riscv_vfmul_vf_f32m1(v_a, rep_scale_a, vl);
                vint16mf2_t  v_a_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a_scale, vl);
                vint8mf4_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a_quant, vl);

                vint16m1_t tmp_যোগ_2 = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a_যোগ = __riscv_vwredsum_vs_i8mf4_i16m1(v_a_quant_i8, tmp_যোগ_2, vl);
                int16_t    a_যোগ   = __riscv_vmv_x_s_i16m1_i16(v_a_যোগ);
                a_যোগ_ptr[kk]      = static_cast<_Float16>(-a_যোগ) * static_cast<_Float16>(8.0f);

                __riscv_vse8_v_i8mf4(quant_a_blk, v_a_quant_i8, vl);
            }
        }
    } অন্যথা যদি (vlenb == 32) {
        জন্য (size_t k = 0; k < গুনতি_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            _Float16 * a_যোগ_ptr     = reinterpret_cast<_Float16 *>(quant_a_ptr + a_subblk_stride * subblk_গুনতি);
            _Float16 * scale_avg_ptr = reinterpret_cast<_Float16 *>(quant_a_ptr + a_blk_stride - আকার(_Float16));
            দশমিক      scale_avg     = 0.0f;

            জন্য (size_t kk = 0; kk < subblk_গুনতি; ++kk) {
                ধ্রুবক দশমিক * a_src_ptr = a_ptr + k + kk * k_subblk_len;

                size_t       vl      = __riscv_vsetvl_e32m4(k_subblk_len);
                vfloat32m4_t v_a     = __riscv_vle32_v_f32m4(a_src_ptr, vl);
                vfloat32m4_t v_a_abs = __riscv_vfabs_v_f32m4(v_a, vl);

                vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m4_f32m1(v_a_abs, tmp, vl);
                দশমিক        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                scale_অস্থায়ী[kk] = max_abs_a / ((1 << 7) - 1);
                scale_avg += scale_অস্থায়ী[kk];
            }

            scale_avg /= subblk_গুনতি;
            ধ্রুবক দশমিক scale_factor = scale_avg ? 1.0f / scale_avg : 0.0f;
            scale_avg_ptr[0]         = static_cast<_Float16>(scale_avg);

            জন্য (size_t kk = 0; kk < subblk_গুনতি; ++kk) {
                uint8_t *     a_subblk_base = quant_a_ptr + kk * a_subblk_stride;
                _Float16 *    scale_a_ptr   = reinterpret_cast<_Float16 *>(a_subblk_base);
                int8_t *      quant_a_blk   = reinterpret_cast<int8_t *>(a_subblk_base + আকার(_Float16));
                ধ্রুবক দশমিক * a_src_ptr     = a_ptr + k + kk * k_subblk_len;

                size_t       vl          = __riscv_vsetvl_e32m4(k_subblk_len);
                vfloat32m4_t v_a         = __riscv_vle32_v_f32m4(a_src_ptr, vl);
                দশমিক        rep_scale_a = scale_অস্থায়ী[kk] ? 1.0f / scale_অস্থায়ী[kk] : 0.0f;
                scale_a_ptr[0]           = static_cast<_Float16>(scale_অস্থায়ী[kk] * scale_factor);

                vfloat32m4_t v_a_scale    = __riscv_vfmul_vf_f32m4(v_a, rep_scale_a, vl);
                vint16m2_t   v_a_quant    = __riscv_vfncvt_x_f_w_i16m2(v_a_scale, vl);
                vint8m1_t    v_a_quant_i8 = __riscv_vncvt_x_x_w_i8m1(v_a_quant, vl);

                vint16m1_t tmp_যোগ_2 = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a_যোগ = __riscv_vwredsum_vs_i8m1_i16m1(v_a_quant_i8, tmp_যোগ_2, vl);
                int16_t    a_যোগ   = __riscv_vmv_x_s_i16m1_i16(v_a_যোগ);
                a_যোগ_ptr[kk]      = static_cast<_Float16>(-a_যোগ) * static_cast<_Float16>(8.0f);

                __riscv_vse8_v_i8m1(quant_a_blk, v_a_quant_i8, vl);
            }
        }
    } অন্যথা {
        quantize_a_nrow_i8_hp_ref<1>(blk_len, a_ptr, গুনতি_k, quant_a_ptr);
    }
}

শূন্য quantize_a_4row_i8_hp(size_t blk_len, ধ্রুবক দশমিক * a_ptr, size_t গুনতি_k, uint8_t * quant_a_ptr) {
    constexpr size_t k_subblk_len = 32;
    GGML_ASSERT(blk_len == 256);

    constexpr size_t subblk_গুনতি             = 256 / k_subblk_len;
    int64_t          a_blk_stride             = q8_hp_blk_আকার(blk_len, সত্য, সত্য);
    int64_t          a_nrow_block_stride      = a_blk_stride * 4;
    int64_t          a_subblk_stride          = q8_hp_blk_আকার(k_subblk_len, মিথ্যা, মিথ্যা) * 4;
    size_t           vlenb                    = __riscv_vlenb();
    দশমিক            scale_অস্থায়ী[subblk_গুনতি] = { 0.0f };

    যদি (vlenb == 128) {
        জন্য (size_t k = 0; k < গুনতি_k; k += blk_len, quant_a_ptr += a_nrow_block_stride) {
            _Float16 * a_যোগ_ptr = reinterpret_cast<_Float16 *>(quant_a_ptr + a_subblk_stride * subblk_গুনতি);
            _Float16 * scale_avg_ptr =
                reinterpret_cast<_Float16 *>(quant_a_ptr + a_nrow_block_stride - আকার(_Float16) * 4);
            দশমিক scale_avg = 0.0f;

            জন্য (size_t kk = 0; kk < subblk_গুনতি; ++kk) {
                ধ্রুবক দশমিক * a_src_ptr0 = a_ptr + 0 * গুনতি_k + k + kk * k_subblk_len;
                ধ্রুবক দশমিক * a_src_ptr1 = a_ptr + 1 * গুনতি_k + k + kk * k_subblk_len;
                ধ্রুবক দশমিক * a_src_ptr2 = a_ptr + 2 * গুনতি_k + k + kk * k_subblk_len;
                ধ্রুবক দশমিক * a_src_ptr3 = a_ptr + 3 * গুনতি_k + k + kk * k_subblk_len;

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
                দশমিক        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                scale_অস্থায়ী[kk] = max_abs_a / ((1 << 7) - 1);
                scale_avg += scale_অস্থায়ী[kk];
            }

            scale_avg /= subblk_গুনতি;
            ধ্রুবক দশমিক scale_factor = scale_avg ? 1.0f / scale_avg : 0.0f;
            scale_avg_ptr[0]         = static_cast<_Float16>(scale_avg);

            জন্য (size_t kk = 0; kk < subblk_গুনতি; ++kk) {
                uint8_t *     a_subblk_base = quant_a_ptr + kk * a_subblk_stride;
                _Float16 *    scale_a_ptr   = reinterpret_cast<_Float16 *>(a_subblk_base);
                int8_t *      quant_a_blk   = reinterpret_cast<int8_t *>(a_subblk_base + আকার(_Float16) * 4);
                ধ্রুবক দশমিক * a_src_ptr0    = a_ptr + 0 * গুনতি_k + k + kk * k_subblk_len;
                ধ্রুবক দশমিক * a_src_ptr1    = a_ptr + 1 * গুনতি_k + k + kk * k_subblk_len;
                ধ্রুবক দশমিক * a_src_ptr2    = a_ptr + 2 * গুনতি_k + k + kk * k_subblk_len;
                ধ্রুবক দশমিক * a_src_ptr3    = a_ptr + 3 * গুনতি_k + k + kk * k_subblk_len;

                size_t       vl   = __riscv_vsetvl_e32m1(k_subblk_len);
                vfloat32m1_t v_a0 = __riscv_vle32_v_f32m1(a_src_ptr0, vl);
                vfloat32m1_t v_a1 = __riscv_vle32_v_f32m1(a_src_ptr1, vl);
                vfloat32m1_t v_a2 = __riscv_vle32_v_f32m1(a_src_ptr2, vl);
                vfloat32m1_t v_a3 = __riscv_vle32_v_f32m1(a_src_ptr3, vl);

                দশমিক rep_scale_a = scale_অস্থায়ী[kk] ? 1.0f / scale_অস্থায়ী[kk] : 0.0f;
                scale_a_ptr[0]    = static_cast<_Float16>(scale_অস্থায়ী[kk] * scale_factor);

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
                vint16m1_t v_a0_যোগ = __riscv_vwredsum_vs_i8mf4_i16m1(v_a0_quant_i8, tmp_sum0, vl);
                vint16m1_t v_a1_যোগ = __riscv_vwredsum_vs_i8mf4_i16m1(v_a1_quant_i8, tmp_sum1, vl);
                vint16m1_t v_a2_যোগ = __riscv_vwredsum_vs_i8mf4_i16m1(v_a2_quant_i8, tmp_sum2, vl);
                vint16m1_t v_a3_যোগ = __riscv_vwredsum_vs_i8mf4_i16m1(v_a3_quant_i8, tmp_sum3, vl);

                a_যোগ_ptr[0 * subblk_গুনতি + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a0_যোগ)) * static_cast<_Float16>(8.0f);
                a_যোগ_ptr[1 * subblk_গুনতি + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a1_যোগ)) * static_cast<_Float16>(8.0f);
                a_যোগ_ptr[2 * subblk_গুনতি + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a2_যোগ)) * static_cast<_Float16>(8.0f);
                a_যোগ_ptr[3 * subblk_গুনতি + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a3_যোগ)) * static_cast<_Float16>(8.0f);

                __riscv_vse8_v_i8mf4(quant_a_blk + 0 * k_subblk_len, v_a0_quant_i8, vl);
                __riscv_vse8_v_i8mf4(quant_a_blk + 1 * k_subblk_len, v_a1_quant_i8, vl);
                __riscv_vse8_v_i8mf4(quant_a_blk + 2 * k_subblk_len, v_a2_quant_i8, vl);
                __riscv_vse8_v_i8mf4(quant_a_blk + 3 * k_subblk_len, v_a3_quant_i8, vl);
            }
        }
    } অন্যথা যদি (vlenb == 32) {
        জন্য (size_t k = 0; k < গুনতি_k; k += blk_len, quant_a_ptr += a_nrow_block_stride) {
            _Float16 * a_যোগ_ptr = reinterpret_cast<_Float16 *>(quant_a_ptr + a_subblk_stride * subblk_গুনতি);
            _Float16 * scale_avg_ptr =
                reinterpret_cast<_Float16 *>(quant_a_ptr + a_nrow_block_stride - আকার(_Float16) * 4);
            দশমিক scale_avg = 0.0f;

            জন্য (size_t kk = 0; kk < subblk_গুনতি; ++kk) {
                ধ্রুবক দশমিক * a_src_ptr0 = a_ptr + 0 * গুনতি_k + k + kk * k_subblk_len;
                ধ্রুবক দশমিক * a_src_ptr1 = a_ptr + 1 * গুনতি_k + k + kk * k_subblk_len;
                ধ্রুবক দশমিক * a_src_ptr2 = a_ptr + 2 * গুনতি_k + k + kk * k_subblk_len;
                ধ্রুবক দশমিক * a_src_ptr3 = a_ptr + 3 * গুনতি_k + k + kk * k_subblk_len;

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
                দশমিক        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                scale_অস্থায়ী[kk] = max_abs_a / ((1 << 7) - 1);
                scale_avg += scale_অস্থায়ী[kk];
            }

            scale_avg /= subblk_গুনতি;
            ধ্রুবক দশমিক scale_factor = scale_avg ? 1.0f / scale_avg : 0.0f;
            scale_avg_ptr[0]         = static_cast<_Float16>(scale_avg);

            জন্য (size_t kk = 0; kk < subblk_গুনতি; ++kk) {
                uint8_t *     a_subblk_base = quant_a_ptr + kk * a_subblk_stride;
                _Float16 *    scale_a_ptr   = reinterpret_cast<_Float16 *>(a_subblk_base);
                int8_t *      quant_a_blk   = reinterpret_cast<int8_t *>(a_subblk_base + আকার(_Float16) * 4);
                ধ্রুবক দশমিক * a_src_ptr0    = a_ptr + 0 * গুনতি_k + k + kk * k_subblk_len;
                ধ্রুবক দশমিক * a_src_ptr1    = a_ptr + 1 * গুনতি_k + k + kk * k_subblk_len;
                ধ্রুবক দশমিক * a_src_ptr2    = a_ptr + 2 * গুনতি_k + k + kk * k_subblk_len;
                ধ্রুবক দশমিক * a_src_ptr3    = a_ptr + 3 * গুনতি_k + k + kk * k_subblk_len;

                size_t       vl   = __riscv_vsetvl_e32m4(k_subblk_len);
                vfloat32m4_t v_a0 = __riscv_vle32_v_f32m4(a_src_ptr0, vl);
                vfloat32m4_t v_a1 = __riscv_vle32_v_f32m4(a_src_ptr1, vl);
                vfloat32m4_t v_a2 = __riscv_vle32_v_f32m4(a_src_ptr2, vl);
                vfloat32m4_t v_a3 = __riscv_vle32_v_f32m4(a_src_ptr3, vl);

                দশমিক rep_scale_a = scale_অস্থায়ী[kk] ? 1.0f / scale_অস্থায়ী[kk] : 0.0f;
                scale_a_ptr[0]    = static_cast<_Float16>(scale_অস্থায়ী[kk] * scale_factor);

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
                vint16m1_t v_a0_যোগ = __riscv_vwredsum_vs_i8m1_i16m1(v_a0_quant_i8, tmp_sum0, vl);
                vint16m1_t v_a1_যোগ = __riscv_vwredsum_vs_i8m1_i16m1(v_a1_quant_i8, tmp_sum1, vl);
                vint16m1_t v_a2_যোগ = __riscv_vwredsum_vs_i8m1_i16m1(v_a2_quant_i8, tmp_sum2, vl);
                vint16m1_t v_a3_যোগ = __riscv_vwredsum_vs_i8m1_i16m1(v_a3_quant_i8, tmp_sum3, vl);

                a_যোগ_ptr[0 * subblk_গুনতি + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a0_যোগ)) * static_cast<_Float16>(8.0f);
                a_যোগ_ptr[1 * subblk_গুনতি + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a1_যোগ)) * static_cast<_Float16>(8.0f);
                a_যোগ_ptr[2 * subblk_গুনতি + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a2_যোগ)) * static_cast<_Float16>(8.0f);
                a_যোগ_ptr[3 * subblk_গুনতি + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a3_যোগ)) * static_cast<_Float16>(8.0f);

                __riscv_vse8_v_i8m1(quant_a_blk + 0 * k_subblk_len, v_a0_quant_i8, vl);
                __riscv_vse8_v_i8m1(quant_a_blk + 1 * k_subblk_len, v_a1_quant_i8, vl);
                __riscv_vse8_v_i8m1(quant_a_blk + 2 * k_subblk_len, v_a2_quant_i8, vl);
                __riscv_vse8_v_i8m1(quant_a_blk + 3 * k_subblk_len, v_a3_quant_i8, vl);
            }
        }
    } অন্যথা {
        quantize_a_nrow_i8_hp_ref<4>(blk_len, a_ptr, গুনতি_k, quant_a_ptr);
    }
}

শূন্য quantize_a_row_i8k(size_t blk_len, ধ্রুবক দশমিক * a_ptr, size_t গুনতি_k, uint8_t * quant_a_ptr) {
    GGML_ASSERT(blk_len == 256);
    constexpr int64_t a_blk_stride = q8k_blk_আকার(256);
    constexpr int64_t a_যোগ_আকার   = 256 / 16;
    size_t            vlenb        = __riscv_vlenb();

    যদি (vlenb == 128) {
        // vlen = 1024 bits, can process 32 float32 elements with m1
        জন্য (size_t k = 0; k < গুনতি_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            দশমিক *   scale_a_ptr = reinterpret_cast<দশমিক *>(quant_a_ptr);
            int16_t * a_যোগ_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + আকার(দশমিক));
            int8_t *  quant_a_blk =
                reinterpret_cast<int8_t *>(quant_a_ptr + আকার(দশমিক) + আকার(int16_t) * a_যোগ_আকার);

            // Find max absolute value across all 256 elements
            size_t       vl        = __riscv_vsetvl_e32m1(16);
            vfloat32m1_t v_max_abs = __riscv_vfmv_v_f_f32m1(0.0f, vl);

            জন্য (size_t bki = 0; bki < a_যোগ_আকার; bki++) {
                vfloat32m1_t v_a     = __riscv_vle32_v_f32m1(a_ptr + k + bki * 16, vl);
                vfloat32m1_t v_a_abs = __riscv_vfabs_v_f32m1(v_a, vl);
                v_max_abs            = __riscv_vfmax_vv_f32m1(v_a_abs, v_max_abs, vl);
            }
            vfloat32m1_t tmp         = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t v_local_max = __riscv_vfredmax_vs_f32m1_f32m1(v_max_abs, tmp, vl);
            দশমিক        max_abs_a   = __riscv_vfmv_f_s_f32m1_f32(v_local_max);

            দশমিক scale_a     = max_abs_a / ((1 << 7) - 1);
            দশমিক rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
            scale_a_ptr[0]    = scale_a;

            // Quantize and compute sums for each 16-element group
            জন্য (size_t bki = 0; bki < a_যোগ_আকার; bki++) {
                vfloat32m1_t v_a          = __riscv_vle32_v_f32m1(a_ptr + k + bki * 16, vl);
                vfloat32m1_t v_a_scale    = __riscv_vfmul_vf_f32m1(v_a, rep_scale_a, vl);
                vint16mf2_t  v_a_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a_scale, vl);
                vint8mf4_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a_quant, vl);

                vint16m1_t tmp_যোগ_2 = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a_যোগ = __riscv_vwredsum_vs_i8mf4_i16m1(v_a_quant_i8, tmp_যোগ_2, vl);
                int16_t    a_যোগ   = __riscv_vmv_x_s_i16m1_i16(v_a_যোগ);
                a_যোগ_ptr[bki]     = -a_যোগ;

                __riscv_vse8_v_i8mf4(quant_a_blk + bki * 16, v_a_quant_i8, vl);
            }
        }
    } অন্যথা যদি (vlenb == 32) {
        // vlen = 256 bits, can process 8 float32 elements with m1
        জন্য (size_t k = 0; k < গুনতি_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            দশমিক *   scale_a_ptr = reinterpret_cast<দশমিক *>(quant_a_ptr);
            int16_t * a_যোগ_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + আকার(দশমিক));
            int8_t *  quant_a_blk =
                reinterpret_cast<int8_t *>(quant_a_ptr + আকার(দশমিক) + আকার(int16_t) * a_যোগ_আকার);

            // Find max absolute value across all 256 elements
            size_t       vl        = __riscv_vsetvl_e32m2(16);
            vfloat32m2_t v_max_abs = __riscv_vfmv_v_f_f32m2(0.0f, vl);

            জন্য (size_t bki = 0; bki < a_যোগ_আকার; bki++) {
                vfloat32m2_t v_a     = __riscv_vle32_v_f32m2(a_ptr + k + bki * 16, vl);
                vfloat32m2_t v_a_abs = __riscv_vfabs_v_f32m2(v_a, vl);
                v_max_abs            = __riscv_vfmax_vv_f32m2(v_a_abs, v_max_abs, vl);
            }
            vfloat32m1_t tmp         = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t v_local_max = __riscv_vfredmax_vs_f32m2_f32m1(v_max_abs, tmp, vl);
            দশমিক        max_abs_a   = __riscv_vfmv_f_s_f32m1_f32(v_local_max);

            দশমিক scale_a     = max_abs_a / ((1 << 7) - 1);
            দশমিক rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
            scale_a_ptr[0]    = scale_a;

            // Quantize and compute sums for each 16-element group
            জন্য (size_t bki = 0; bki < a_যোগ_আকার; bki++) {
                vfloat32m2_t v_a          = __riscv_vle32_v_f32m2(a_ptr + k + bki * 16, vl);
                vfloat32m2_t v_a_scale    = __riscv_vfmul_vf_f32m2(v_a, rep_scale_a, vl);
                vint16m1_t   v_a_quant    = __riscv_vfncvt_x_f_w_i16m1(v_a_scale, vl);
                vint8mf2_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf2(v_a_quant, vl);

                vint16m1_t tmp_যোগ_2 = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a_যোগ = __riscv_vwredsum_vs_i8mf2_i16m1(v_a_quant_i8, tmp_যোগ_2, vl);
                int16_t    a_যোগ   = __riscv_vmv_x_s_i16m1_i16(v_a_যোগ);
                a_যোগ_ptr[bki]     = -a_যোগ;

                __riscv_vse8_v_i8mf2(quant_a_blk + bki * 16, v_a_quant_i8, vl);
            }
        }
    } অন্যথা {
        quantize_a_nrow_i8k_ref<1>(blk_len, a_ptr, গুনতি_k, quant_a_ptr);
    }
}

শূন্য quantize_a_4row_i8k(size_t blk_len, ধ্রুবক দশমিক * a_ptr, size_t গুনতি_k, uint8_t * quant_a_ptr) {
    GGML_ASSERT(blk_len == 256);
    constexpr int64_t a_blk_stride        = q8k_blk_আকার(256);
    constexpr int64_t a_nrow_block_stride = a_blk_stride * 4;
    constexpr int64_t a_যোগ_আকার          = 256 / 16;
    size_t            vlenb               = __riscv_vlenb();

    যদি (vlenb == 128) {
        // vlen = 1024 bits
        জন্য (size_t k = 0; k < গুনতি_k; k += blk_len, quant_a_ptr += a_nrow_block_stride) {
            দশমিক *   scale_a_ptr = reinterpret_cast<দশমিক *>(quant_a_ptr);
            int16_t * a_যোগ_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + আকার(দশমিক) * 4);
            int8_t *  quant_a_blk =
                reinterpret_cast<int8_t *>(quant_a_ptr + আকার(দশমিক) * 4 + আকার(int16_t) * a_যোগ_আকার * 4);

            জন্য (size_t mi = 0; mi < 4; mi++) {
                // Find max absolute value across all 256 elements for this row
                size_t       vl        = __riscv_vsetvl_e32m1(16);
                vfloat32m1_t v_max_abs = __riscv_vfmv_v_f_f32m1(0.0f, vl);

                জন্য (size_t bki = 0; bki < a_যোগ_আকার; bki++) {
                    vfloat32m1_t v_a     = __riscv_vle32_v_f32m1(a_ptr + mi * গুনতি_k + k + bki * 16, vl);
                    vfloat32m1_t v_a_abs = __riscv_vfabs_v_f32m1(v_a, vl);
                    v_max_abs            = __riscv_vfmax_vv_f32m1(v_a_abs, v_max_abs, vl);
                }
                vfloat32m1_t tmp         = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_local_max = __riscv_vfredmax_vs_f32m1_f32m1(v_max_abs, tmp, vl);
                দশমিক        max_abs_a   = __riscv_vfmv_f_s_f32m1_f32(v_local_max);

                দশমিক scale_a     = max_abs_a / ((1 << 7) - 1);
                দশমিক rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
                scale_a_ptr[mi]   = scale_a;

                // Quantize and compute sums for each 16-element group
                জন্য (size_t bki = 0; bki < a_যোগ_আকার; bki++) {
                    vfloat32m1_t v_a          = __riscv_vle32_v_f32m1(a_ptr + mi * গুনতি_k + k + bki * 16, vl);
                    vfloat32m1_t v_a_scale    = __riscv_vfmul_vf_f32m1(v_a, rep_scale_a, vl);
                    vint16mf2_t  v_a_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a_scale, vl);
                    vint8mf4_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a_quant, vl);

                    vint16m1_t tmp_যোগ_2               = __riscv_vmv_v_x_i16m1(0, vl);
                    vint16m1_t v_a_যোগ               = __riscv_vwredsum_vs_i8mf4_i16m1(v_a_quant_i8, tmp_যোগ_2, vl);
                    int16_t    a_যোগ                 = __riscv_vmv_x_s_i16m1_i16(v_a_যোগ);
                    a_যোগ_ptr[mi * a_যোগ_আকার + bki] = -a_যোগ;

                    __riscv_vse8_v_i8mf4(quant_a_blk + mi * blk_len + bki * 16, v_a_quant_i8, vl);
                }
            }
        }
    } অন্যথা যদি (vlenb == 32) {
        // vlen = 256 bits
        জন্য (size_t k = 0; k < গুনতি_k; k += blk_len, quant_a_ptr += a_nrow_block_stride) {
            দশমিক *   scale_a_ptr = reinterpret_cast<দশমিক *>(quant_a_ptr);
            int16_t * a_যোগ_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + আকার(দশমিক) * 4);
            int8_t *  quant_a_blk =
                reinterpret_cast<int8_t *>(quant_a_ptr + আকার(দশমিক) * 4 + আকার(int16_t) * a_যোগ_আকার * 4);

            জন্য (size_t mi = 0; mi < 4; mi++) {
                // Find max absolute value across all 256 elements for this row
                size_t       vl        = __riscv_vsetvl_e32m2(16);
                vfloat32m2_t v_max_abs = __riscv_vfmv_v_f_f32m2(0.0f, vl);

                জন্য (size_t bki = 0; bki < a_যোগ_আকার; bki++) {
                    vfloat32m2_t v_a     = __riscv_vle32_v_f32m2(a_ptr + mi * গুনতি_k + k + bki * 16, vl);
                    vfloat32m2_t v_a_abs = __riscv_vfabs_v_f32m2(v_a, vl);
                    v_max_abs            = __riscv_vfmax_vv_f32m2(v_a_abs, v_max_abs, vl);
                }
                vfloat32m1_t tmp         = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_local_max = __riscv_vfredmax_vs_f32m2_f32m1(v_max_abs, tmp, vl);
                দশমিক        max_abs_a   = __riscv_vfmv_f_s_f32m1_f32(v_local_max);

                দশমিক scale_a     = max_abs_a / ((1 << 7) - 1);
                দশমিক rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
                scale_a_ptr[mi]   = scale_a;

                // Quantize and compute sums for each 16-element group
                জন্য (size_t bki = 0; bki < a_যোগ_আকার; bki++) {
                    vfloat32m2_t v_a          = __riscv_vle32_v_f32m2(a_ptr + mi * গুনতি_k + k + bki * 16, vl);
                    vfloat32m2_t v_a_scale    = __riscv_vfmul_vf_f32m2(v_a, rep_scale_a, vl);
                    vint16m1_t   v_a_quant    = __riscv_vfncvt_x_f_w_i16m1(v_a_scale, vl);
                    vint8mf2_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf2(v_a_quant, vl);

                    vint16m1_t tmp_যোগ_2               = __riscv_vmv_v_x_i16m1(0, vl);
                    vint16m1_t v_a_যোগ               = __riscv_vwredsum_vs_i8mf2_i16m1(v_a_quant_i8, tmp_যোগ_2, vl);
                    int16_t    a_যোগ                 = __riscv_vmv_x_s_i16m1_i16(v_a_যোগ);
                    a_যোগ_ptr[mi * a_যোগ_আকার + bki] = -a_যোগ;

                    __riscv_vse8_v_i8mf2(quant_a_blk + mi * blk_len + bki * 16, v_a_quant_i8, vl);
                }
            }
        }
    } অন্যথা {
        quantize_a_nrow_i8k_ref<4>(blk_len, a_ptr, গুনতি_k, quant_a_ptr);
    }
}

শূন্য forward_cpy_with_permute(ggml_compute_params * params, ggml_tensor * op) {
    ধ্রুবক ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;
    ধ্রুবক পূর্ণসংখ্যা           ith  = params->ith;
    ধ্রুবক পূর্ণসংখ্যা           nth  = params->nth;

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

শূন্য forward_cont_with_permute(ggml_compute_params * params, ggml_tensor * op) {
    ধ্রুবক ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;
    ধ্রুবক পূর্ণসংখ্যা           ith  = params->ith;
    ধ্রুবক পূর্ণসংখ্যা           nth  = params->nth;

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

শূন্য forward_norm_f32(ggml_compute_params * params, ggml_tensor * op) {
    ধ্রুবক ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(src0->nb[0] == আকার(দশমিক));

    পূর্ণসংখ্যা ith = params->ith;
    পূর্ণসংখ্যা nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    দশমিক epsilon = *((দশমিক *) dst->op_params);

    GGML_ASSERT(epsilon > 0.0f);

    auto * প্রবেশ  = (অক্ষর *) src0->তথ্য;
    auto * output = (অক্ষর *) dst->তথ্য;

    ধ্রুবক auto hidden_আকার     = ne00;
    ধ্রুবক auto task_গুনতি      = ne01 * ne02 * ne03;
    ধ্রুবক auto task_per_thread = (task_গুনতি + nth - 1) / nth;

    ধ্রুবক auto task_begin = ith * task_per_thread;
    ধ্রুবক auto task_end   = std::min((ith + 1) * task_per_thread, task_গুনতি);

    জন্য (auto task_idx = task_begin; task_idx < task_end; task_idx++) {
        int64_t i03 = task_idx / (ne02 * ne01);
        int64_t i02 = (task_idx - i03 * ne02 * ne01) / ne01;
        int64_t i01 = (task_idx - i03 * ne02 * ne01 - i02 * ne01);

        auto * p_প্রবেশ       = (দশমিক *) (প্রবেশ + i01 * nb01 + i02 * nb02 + i03 * nb03);
        auto * p_output      = (দশমিক *) (output + i01 * nb1 + i02 * nb2 + i03 * nb3);
        auto * p_অস্থায়ী_output = p_output;

        size_t       gvl    = __riscv_vsetvlmax_e32m4();
        vfloat32m4_t যোগ_2    = __riscv_vfmv_v_f_f32m4(0.f, gvl);
        vfloat32m4_t যোগ_sq = __riscv_vfmv_v_f_f32m4(0.f, gvl);
        int64_t      দৈর্ঘ্য = hidden_আকার;
        যখন (দৈর্ঘ্য > 0) {
            gvl                   = __riscv_vsetvl_e32m4(দৈর্ঘ্য);
            // load data
            vfloat32m4_t src_তথ্য = __riscv_vle32_v_f32m4(p_প্রবেশ, gvl);

            যোগ_2    = __riscv_vfadd_vv_f32m4(যোগ_2, src_তথ্য, gvl);
            যোগ_sq = __riscv_vfmacc_vv_f32m4(যোগ_sq, src_তথ্য, src_তথ্য, gvl);

            __riscv_vse32_v_f32m4(p_অস্থায়ী_output, src_তথ্য, gvl);

            p_প্রবেশ += gvl;
            p_অস্থায়ী_output += gvl;
            দৈর্ঘ্য -= gvl;
        }

        gvl = __riscv_vsetvlmax_e32m1();

        দশমিক        mean   = 0.f;
        vfloat32m1_t zero_v = __riscv_vfmv_v_f_f32m1(0.f, gvl);
        vfloat32m1_t mean_v =
            __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(যোগ_2, 0), __riscv_vget_v_f32m4_f32m1(যোগ_2, 1), gvl);
        mean_v = __riscv_vfadd_vv_f32m1(mean_v, __riscv_vget_v_f32m4_f32m1(যোগ_2, 2), gvl);
        mean_v = __riscv_vfadd_vv_f32m1(mean_v, __riscv_vget_v_f32m4_f32m1(যোগ_2, 3), gvl);
        mean_v = __riscv_vfredusum_vs_f32m1_f32m1(mean_v, zero_v, gvl);
        mean   = __riscv_vfmv_f_s_f32m1_f32(mean_v);
        mean /= hidden_আকার;

        vfloat32m1_t mean_square_v =
            __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(যোগ_sq, 0), __riscv_vget_v_f32m4_f32m1(যোগ_sq, 1), gvl);
        mean_square_v = __riscv_vfadd_vv_f32m1(mean_square_v, __riscv_vget_v_f32m4_f32m1(যোগ_sq, 2), gvl);
        mean_square_v = __riscv_vfadd_vv_f32m1(mean_square_v, __riscv_vget_v_f32m4_f32m1(যোগ_sq, 3), gvl);
        mean_square_v = __riscv_vfredusum_vs_f32m1_f32m1(mean_square_v, zero_v, gvl);

        দশমিক mean_square = __riscv_vfmv_f_s_f32m1_f32(mean_square_v);
        mean_square /= hidden_আকার;
        mean_square = sqrt(mean_square - mean * mean + epsilon);

        mean_square   = 1.0f / mean_square;
        দৈর্ঘ্য        = hidden_আকার;
        p_অস্থায়ী_output = p_output;

        যখন (দৈর্ঘ্য > 0) {
            gvl                   = __riscv_vsetvl_e32m4(দৈর্ঘ্য);
            vfloat32m4_t src_তথ্য = __riscv_vle32_v_f32m4(p_অস্থায়ী_output, gvl);
            src_তথ্য              = __riscv_vfsub_vf_f32m4(src_তথ্য, mean, gvl);
            src_তথ্য              = __riscv_vfmul_vf_f32m4(src_তথ্য, mean_square, gvl);
            __riscv_vse32_v_f32m4(p_output, src_তথ্য, gvl);
            p_অস্থায়ী_output += gvl;
            p_output += gvl;
            দৈর্ঘ্য -= gvl;
        }
    }
}

template <ggml_op op_type, typename T> শূন্য forward_দ্বিমিক(ggml_compute_params * params, ggml_tensor * op) {
    ধ্রুবক ggml_tensor * src0 = op->src[0];
    ধ্রুবক ggml_tensor * src1 = op->src[1];
    ggml_tensor *       dst  = op;
    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    auto src0_rows = ggml_nrows(src0);
    auto src1_rows = ggml_nrows(src1);

    পূর্ণসংখ্যা ith = params->ith;
    পূর্ণসংখ্যা nth = params->nth;

    GGML_TENSOR_দ্বিমিক_OP_LOCALS

    GGML_ASSERT(nb0 == আকার(T));
    GGML_ASSERT(nb00 == আকার(T));

    ধ্রুবক auto [ir0, ir1] = get_thread_range(params, src0);

    auto compute_func_vv = [&](int64_t blk_len, int64_t r, T * src0_ptr, T * src1_ptr, T * dst_ptr) {
        int64_t idx = 0;
        যদি constexpr (op_type == GGML_OP_ADD) {
            যদি constexpr (std::is_same_v<T, দশমিক>) {
                জন্য (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e32m4(blk_len);
                    vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + idx + r, vl);
                    vfloat32m4_t rhs = __riscv_vle32_v_f32m4(src1_ptr + idx, vl);
                    vfloat32m4_t res = __riscv_vfadd_vv_f32m4(lhs, rhs, vl);
                    __riscv_vse32_v_f32m4(dst_ptr + idx + r, res, vl);
                }
            } অন্যথা যদি constexpr (std::is_same_v<T, _Float16>) {
                জন্য (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e16m4(blk_len);
                    vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + idx + r), vl);
                    vfloat16m4_t rhs = __riscv_vle16_v_f16m4((src1_ptr + idx), vl);
                    vfloat16m4_t res = __riscv_vfadd_vv_f16m4(lhs, rhs, vl);
                    __riscv_vse16_v_f16m4((dst_ptr + idx + r), res, vl);
                }
            } অন্যথা {
                GGML_ABORT("fatal error");
            }
        } অন্যথা যদি constexpr (op_type == GGML_OP_SUB) {
            যদি constexpr (std::is_same_v<T, দশমিক>) {
                জন্য (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e32m4(blk_len);
                    vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + idx + r, vl);
                    vfloat32m4_t rhs = __riscv_vle32_v_f32m4(src1_ptr + idx, vl);
                    vfloat32m4_t res = __riscv_vfsub_vv_f32m4(lhs, rhs, vl);
                    __riscv_vse32_v_f32m4(dst_ptr + idx + r, res, vl);
                }
            } অন্যথা যদি constexpr (std::is_same_v<T, _Float16>) {
                জন্য (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e16m4(blk_len);
                    vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + idx + r), vl);
                    vfloat16m4_t rhs = __riscv_vle16_v_f16m4((src1_ptr + idx), vl);
                    vfloat16m4_t res = __riscv_vfsub_vv_f16m4(lhs, rhs, vl);
                    __riscv_vse16_v_f16m4((dst_ptr + idx + r), res, vl);
                }
            } অন্যথা {
                GGML_ABORT("fatal error");
            }
        } অন্যথা যদি constexpr (op_type == GGML_OP_MUL) {
            যদি constexpr (std::is_same_v<T, দশমিক>) {
                জন্য (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e32m4(blk_len);
                    vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + idx + r, vl);
                    vfloat32m4_t rhs = __riscv_vle32_v_f32m4(src1_ptr + idx, vl);
                    vfloat32m4_t res = __riscv_vfmul_vv_f32m4(lhs, rhs, vl);
                    __riscv_vse32_v_f32m4(dst_ptr + idx + r, res, vl);
                }
            } অন্যথা যদি constexpr (std::is_same_v<T, _Float16>) {
                জন্য (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e16m4(blk_len);
                    vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + idx + r), vl);
                    vfloat16m4_t rhs = __riscv_vle16_v_f16m4((src1_ptr + idx), vl);
                    vfloat16m4_t res = __riscv_vfmul_vv_f16m4(lhs, rhs, vl);
                    __riscv_vse16_v_f16m4((dst_ptr + idx + r), res, vl);
                }
            } অন্যথা {
                GGML_ABORT("fatal error");
            }
        } অন্যথা যদি constexpr (op_type == GGML_OP_DIV) {
            যদি constexpr (std::is_same_v<T, দশমিক>) {
                জন্য (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e32m4(blk_len);
                    vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + idx + r, vl);
                    vfloat32m4_t rhs = __riscv_vle32_v_f32m4(src1_ptr + idx, vl);
                    vfloat32m4_t res = __riscv_vfdiv_vv_f32m4(lhs, rhs, vl);
                    __riscv_vse32_v_f32m4(dst_ptr + idx + r, res, vl);
                }
            } অন্যথা যদি constexpr (std::is_same_v<T, _Float16>) {
                জন্য (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e16m4(blk_len);
                    vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + idx + r), vl);
                    vfloat16m4_t rhs = __riscv_vle16_v_f16m4((src1_ptr + idx), vl);
                    vfloat16m4_t res = __riscv_vfdiv_vv_f16m4(lhs, rhs, vl);
                    __riscv_vse16_v_f16m4((dst_ptr + idx + r), res, vl);
                }
            } অন্যথা {
                GGML_ABORT("fatal error");
            }
        } অন্যথা {
            GGML_ABORT("fatal error");
        }
    };

    যদি (src0_rows == src1_rows && src0_rows == 1 && ne00 == ne10) {
        int64_t task_per_thread = (ne00 + nth - 1) / nth;
        int64_t task_begin      = ith * task_per_thread;
        int64_t task_end        = std::min((ith + 1) * task_per_thread, ne00);

        T * dst_ptr  = ((T *) dst->তথ্য) + task_begin;
        T * src0_ptr = ((T *) src0->তথ্য) + task_begin;
        T * src1_ptr = ((T *) src1->তথ্য) + task_begin;

        compute_func_vv(task_end - task_begin, 0, src0_ptr, src1_ptr, dst_ptr);
    } অন্যথা যদি (ne10 > 1) {
        জন্য (int64_t ir = ir0; ir < ir1; ++ir) {
            ধ্রুবক int64_t i03 = ir / (ne02 * ne01);
            ধ্রুবক int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            ধ্রুবক int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

            ধ্রুবক int64_t i13 = i03 % ne13;
            ধ্রুবক int64_t i12 = i02 % ne12;
            ধ্রুবক int64_t i11 = i01 % ne11;

            T * dst_ptr  = (T *) ((অক্ষর *) dst->তথ্য + i03 * nb3 + i02 * nb2 + i01 * nb1);
            T * src0_ptr = (T *) ((অক্ষর *) src0->তথ্য + i03 * nb03 + i02 * nb02 + i01 * nb01);
            T * src1_ptr = (T *) ((অক্ষর *) src1->তথ্য + i13 * nb13 + i12 * nb12 + i11 * nb11);

            // src1 is broadcastable across src0 and dst in i1, i2, i3
            জন্য (int64_t r = 0; r < ne00; r += ne10) {
                compute_func_vv(ne10, r, src0_ptr, src1_ptr, dst_ptr);
            }
        }
    } অন্যথা {
        জন্য (int64_t ir = ir0; ir < ir1; ++ir) {
            ধ্রুবক int64_t i03 = ir / (ne02 * ne01);
            ধ্রুবক int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            ধ্রুবক int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

            ধ্রুবক int64_t i13 = i03 % ne13;
            ধ্রুবক int64_t i12 = i02 % ne12;
            ধ্রুবক int64_t i11 = i01 % ne11;

            T * dst_ptr  = (T *) ((অক্ষর *) dst->তথ্য + i03 * nb3 + i02 * nb2 + i01 * nb1);
            T * src0_ptr = (T *) ((অক্ষর *) src0->তথ্য + i03 * nb03 + i02 * nb02 + i01 * nb01);
            T * src1_ptr = (T *) ((অক্ষর *) src1->তথ্য + i13 * nb13 + i12 * nb12 + i11 * nb11);

            T       rhs_scalar = src1_ptr[0];
            int64_t blk_len    = ne00;
            int64_t r          = 0;

            জন্য (size_t vl; blk_len > 0; blk_len -= vl, r += vl) {
                যদি constexpr (op_type == GGML_OP_ADD) {
                    যদি constexpr (std::is_same_v<T, দশমিক>) {
                        vl               = __riscv_vsetvl_e32m4(blk_len);
                        vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + r, vl);
                        vfloat32m4_t res = __riscv_vfadd_vf_f32m4(lhs, rhs_scalar, vl);
                        __riscv_vse32_v_f32m4(dst_ptr + r, res, vl);
                    } অন্যথা যদি constexpr (std::is_same_v<T, _Float16>) {
                        vl               = __riscv_vsetvl_e16m4(blk_len);
                        vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + r), vl);
                        vfloat16m4_t res = __riscv_vfadd_vf_f16m4(lhs, rhs_scalar, vl);
                        __riscv_vse16_v_f16m4((dst_ptr + r), res, vl);
                    } অন্যথা {
                        GGML_ABORT("fatal error");
                    }
                } অন্যথা যদি constexpr (op_type == GGML_OP_SUB) {
                    যদি constexpr (std::is_same_v<T, দশমিক>) {
                        vl               = __riscv_vsetvl_e32m4(blk_len);
                        vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + r, vl);
                        vfloat32m4_t res = __riscv_vfsub_vf_f32m4(lhs, rhs_scalar, vl);
                        __riscv_vse32_v_f32m4(dst_ptr + r, res, vl);
                    } অন্যথা যদি constexpr (std::is_same_v<T, _Float16>) {
                        vl               = __riscv_vsetvl_e16m4(blk_len);
                        vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + r), vl);
                        vfloat16m4_t res = __riscv_vfsub_vf_f16m4(lhs, rhs_scalar, vl);
                        __riscv_vse16_v_f16m4((dst_ptr + r), res, vl);
                    } অন্যথা {
                        GGML_ABORT("fatal error");
                    }
                } অন্যথা যদি constexpr (op_type == GGML_OP_MUL) {
                    যদি constexpr (std::is_same_v<T, দশমিক>) {
                        vl               = __riscv_vsetvl_e32m4(blk_len);
                        vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + r, vl);
                        vfloat32m4_t res = __riscv_vfmul_vf_f32m4(lhs, rhs_scalar, vl);
                        __riscv_vse32_v_f32m4(dst_ptr + r, res, vl);
                    } অন্যথা যদি constexpr (std::is_same_v<T, _Float16>) {
                        vl               = __riscv_vsetvl_e16m4(blk_len);
                        vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + r), vl);
                        vfloat16m4_t res = __riscv_vfmul_vf_f16m4(lhs, rhs_scalar, vl);
                        __riscv_vse16_v_f16m4((dst_ptr + r), res, vl);
                    } অন্যথা {
                        GGML_ABORT("fatal error");
                    }
                } অন্যথা যদি constexpr (op_type == GGML_OP_DIV) {
                    যদি constexpr (std::is_same_v<T, দশমিক>) {
                        vl               = __riscv_vsetvl_e32m4(blk_len);
                        vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + r, vl);
                        vfloat32m4_t res = __riscv_vfdiv_vf_f32m4(lhs, rhs_scalar, vl);
                        __riscv_vse32_v_f32m4(dst_ptr + r, res, vl);
                    } অন্যথা যদি constexpr (std::is_same_v<T, _Float16>) {
                        vl               = __riscv_vsetvl_e16m4(blk_len);
                        vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + r), vl);
                        vfloat16m4_t res = __riscv_vfdiv_vf_f16m4(lhs, rhs_scalar, vl);
                        __riscv_vse16_v_f16m4((dst_ptr + r), res, vl);
                    } অন্যথা {
                        GGML_ABORT("fatal error");
                    }
                } অন্যথা {
                    GGML_ABORT("fatal error");
                }
            }
        }
    }
}

template <typename T> শূন্য forward_যোগ_rows(ধ্রুবক ggml_compute_params * params, ggml_tensor * op) {
    ধ্রুবক ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;

    ধ্রুবক পূর্ণসংখ্যা ith = params->ith;
    ধ্রুবক পূর্ণসংখ্যা nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne0 == 1);
    GGML_ASSERT(ne1 == ne01);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    int64_t n_task          = ne01 * ne02 * ne03;
    int64_t task_per_thread = (n_task + nth - 1) / nth;
    int64_t ir_start        = ith * task_per_thread;
    int64_t ir_end          = std::min(ir_start + task_per_thread, n_task);

    জন্য (int64_t ir = ir_start; ir < ir_end; ir++) {
        ধ্রুবক int64_t i3 = ir / (ne02 * ne01);
        ধ্রুবক int64_t i2 = (ir - i3 * ne02 * ne01) / ne01;
        ধ্রুবক int64_t i1 = (ir - i3 * ne02 * ne01 - i2 * ne01);

        T * src_row = (T *) ((অক্ষর *) src0->তথ্য + i1 * nb01 + i2 * nb02 + i3 * nb03);
        T * dst_row = (T *) ((অক্ষর *) op->তথ্য + i1 * nb1 + i2 * nb2 + i3 * nb3);

        দশমিক row_যোগ = 0;

        যদি constexpr (std::is_same_v<T, দশমিক>) {
            size_t        gvl     = __riscv_vsetvlmax_e32m4();
            vfloat32m4_t  acc_vec = __riscv_vfmv_v_f_f32m4(0.0f, gvl);
            int64_t       দৈর্ঘ্য  = ne00;
            ধ্রুবক দশমিক * p_তথ্য  = src_row;

            যখন (দৈর্ঘ্য > 0) {
                size_t       vl  = __riscv_vsetvl_e32m4(দৈর্ঘ্য);
                vfloat32m4_t vec = __riscv_vle32_v_f32m4(p_তথ্য, vl);
                acc_vec          = __riscv_vfadd_vv_f32m4(acc_vec, vec, vl);
                p_তথ্য += vl;
                দৈর্ঘ্য -= vl;
            }

            gvl                 = __riscv_vsetvlmax_e32m1();
            vfloat32m1_t zero_v = __riscv_vfmv_v_f_f32m1(0.0f, gvl);
            vfloat32m1_t যোগ_v  = __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(acc_vec, 0),
                                                         __riscv_vget_v_f32m4_f32m1(acc_vec, 1), gvl);
            যোগ_v               = __riscv_vfadd_vv_f32m1(যোগ_v, __riscv_vget_v_f32m4_f32m1(acc_vec, 2), gvl);
            যোগ_v               = __riscv_vfadd_vv_f32m1(যোগ_v, __riscv_vget_v_f32m4_f32m1(acc_vec, 3), gvl);
            যোগ_v               = __riscv_vfredusum_vs_f32m1_f32m1(যোগ_v, zero_v, gvl);
            row_যোগ             = __riscv_vfmv_f_s_f32m1_f32(যোগ_v);
        } অন্যথা যদি constexpr (std::is_same_v<T, _Float16>) {
            size_t           gvl     = __riscv_vsetvlmax_e16m2();
            vfloat32m4_t     acc_vec = __riscv_vfmv_v_f_f32m4(0.0f, gvl);
            int64_t          দৈর্ঘ্য  = ne00;
            ধ্রুবক _Float16 * p_তথ্য  = src_row;

            যখন (দৈর্ঘ্য > 0) {
                size_t       vl      = __riscv_vsetvl_e16m2(দৈর্ঘ্য);
                vfloat16m2_t vec_f16 = __riscv_vle16_v_f16m2(p_তথ্য, vl);
                vfloat32m4_t vec_f32 = __riscv_vfwcvt_f_f_v_f32m4(vec_f16, vl);
                acc_vec              = __riscv_vfadd_vv_f32m4(acc_vec, vec_f32, vl);
                p_তথ্য += vl;
                দৈর্ঘ্য -= vl;
            }

            gvl                 = __riscv_vsetvlmax_e32m1();
            vfloat32m1_t zero_v = __riscv_vfmv_v_f_f32m1(0.0f, gvl);
            vfloat32m1_t যোগ_v  = __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(acc_vec, 0),
                                                         __riscv_vget_v_f32m4_f32m1(acc_vec, 1), gvl);
            যোগ_v               = __riscv_vfadd_vv_f32m1(যোগ_v, __riscv_vget_v_f32m4_f32m1(acc_vec, 2), gvl);
            যোগ_v               = __riscv_vfadd_vv_f32m1(যোগ_v, __riscv_vget_v_f32m4_f32m1(acc_vec, 3), gvl);
            যোগ_v               = __riscv_vfredusum_vs_f32m1_f32m1(যোগ_v, zero_v, gvl);
            row_যোগ             = __riscv_vfmv_f_s_f32m1_f32(যোগ_v);
        } অন্যথা {
            GGML_ABORT("fatal error");
        }

        dst_row[0] = row_যোগ;
    }
}

template <typename T> শূন্য forward_repeat_nrows(ggml_compute_params * params, ggml_tensor * op) {
    ধ্রুবক ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;

    ধ্রুবক পূর্ণসংখ্যা ith = params->ith;
    ধ্রুবক পূর্ণসংখ্যা nth = params->nth;

    int64_t nrows            = ggml_nrows(src0);
    int64_t nrows_per_thread = (nrows + nth - 1) / nth;
    int64_t ir_start         = ith * nrows_per_thread;
    int64_t ir_end           = std::min(ir_start + nrows_per_thread, nrows);

    যদি (src0->ne[0] == 1) {
        জন্য (int64_t ir = ir_start; ir < ir_end; ir++) {
            T * src_row = (T *) ((অক্ষর *) src0->তথ্য + ir * src0->nb[1]);
            T * dst_row = (T *) ((অক্ষর *) dst->তথ্য + ir * dst->nb[1]);

            T src_scalar = src_row[0];

            int64_t দৈর্ঘ্য = dst->ne[0];
            int64_t idx    = 0;
            size_t  vl     = 0;

            যখন (দৈর্ঘ্য > 0) {
                যদি constexpr (std::is_same_v<T, int32_t>) {
                    vl             = __riscv_vsetvl_e32m4(দৈর্ঘ্য);
                    vint32m4_t vec = __riscv_vmv_v_x_i32m4(src_scalar, vl);
                    __riscv_vse32_v_i32m4(dst_row + idx, vec, vl);
                } অন্যথা যদি constexpr (std::is_same_v<T, int16_t>) {
                    vl             = __riscv_vsetvl_e16m4(দৈর্ঘ্য);
                    vint16m4_t vec = __riscv_vmv_v_x_i16m4(src_scalar, vl);
                    __riscv_vse16_v_i16m4((dst_row + idx), vec, vl);
                } অন্যথা {
                    GGML_ABORT("fatal error");
                }
                idx += vl;
                দৈর্ঘ্য -= vl;
            }
        }
    } অন্যথা যদি (src0->ne[0] == dst->ne[0]) {
        জন্য (int64_t ir = ir_start; ir < ir_end; ir++) {
            T * src_row = (T *) ((অক্ষর *) src0->তথ্য + ir * src0->nb[1]);
            T * dst_row = (T *) ((অক্ষর *) dst->তথ্য + ir * dst->nb[1]);

            int64_t দৈর্ঘ্য = dst->ne[0];
            int64_t idx    = 0;
            size_t  vl     = 0;

            যখন (দৈর্ঘ্য > 0) {
                যদি constexpr (std::is_same_v<T, int32_t>) {
                    vl             = __riscv_vsetvl_e32m4(দৈর্ঘ্য);
                    vint32m4_t vec = __riscv_vle32_v_i32m4(src_row + idx, vl);
                    __riscv_vse32_v_i32m4(dst_row + idx, vec, vl);
                } অন্যথা যদি constexpr (std::is_same_v<T, int16_t>) {
                    vl             = __riscv_vsetvl_e16m4(দৈর্ঘ্য);
                    vint16m4_t vec = __riscv_vle16_v_i16m4((src_row + idx), vl);
                    __riscv_vse16_v_i16m4((dst_row + idx), vec, vl);
                } অন্যথা {
                    GGML_ABORT("fatal error");
                }
                idx += vl;
                দৈর্ঘ্য -= vl;
            }
        }
    } অন্যথা {
        GGML_ABORT("fatal error");
    }
}

template <typename T> শূন্য forward_repeat_dim1(ggml_compute_params * params, ggml_tensor * op) {
    ধ্রুবক ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;

    ধ্রুবক পূর্ণসংখ্যা ith = params->ith;
    ধ্রুবক পূর্ণসংখ্যা nth = params->nth;

    ধ্রুবক int64_t ne0 = dst->ne[0];
    ধ্রুবক int64_t ne1 = dst->ne[1];
    ধ্রুবক int64_t ne2 = dst->ne[2];
    ধ্রুবক int64_t ne3 = dst->ne[3];

    ধ্রুবক int64_t যোগ_batches      = ne2 * ne3;
    ধ্রুবক int64_t batches_per_thread = (যোগ_batches + nth - 1) / nth;
    ধ্রুবক int64_t batch_start        = ith * batches_per_thread;
    ধ্রুবক int64_t batch_end          = std::min(batch_start + batches_per_thread, যোগ_batches);

    জন্য (int64_t b = batch_start; b < batch_end; b++) {
        ধ্রুবক int64_t i3 = b / ne2;
        ধ্রুবক int64_t i2 = b % ne2;

        T * src_base  = (T *) ((অক্ষর *) src0->তথ্য + i2 * src0->nb[2] + i3 * src0->nb[3]);
        T * dst_batch = (T *) ((অক্ষর *) dst->তথ্য + i2 * dst->nb[2] + i3 * dst->nb[3]);

        জন্য (int64_t i1 = 0; i1 < ne1; i1++) {
            T *     dst_ptr = (T *) ((অক্ষর *) dst_batch + i1 * dst->nb[1]);
            int64_t দৈর্ঘ্য  = ne0;
            int64_t idx     = 0;

            যখন (দৈর্ঘ্য > 0) {
                যদি constexpr (std::is_same_v<T, int32_t>) {
                    size_t     vl  = __riscv_vsetvl_e32m4(দৈর্ঘ্য);
                    vint32m4_t vec = __riscv_vle32_v_i32m4(src_base + idx, vl);
                    __riscv_vse32_v_i32m4(dst_ptr + idx, vec, vl);
                    idx += vl;
                    দৈর্ঘ্য -= vl;
                } অন্যথা যদি constexpr (std::is_same_v<T, int16_t>) {
                    size_t     vl  = __riscv_vsetvl_e16m4(দৈর্ঘ্য);
                    vint16m4_t vec = __riscv_vle16_v_i16m4((src_base + idx), vl);
                    __riscv_vse16_v_i16m4((dst_ptr + idx), vec, vl);
                    idx += vl;
                    দৈর্ঘ্য -= vl;
                } অন্যথা {
                    GGML_ABORT("fatal error");
                }
            }
        }
    }
}

template <typename T> শূন্য forward_get_rows(ggml_compute_params * params, ggml_tensor * op) {
    ধ্রুবক ggml_tensor * src0 = op->src[0];
    ধ্রুবক ggml_tensor * src1 = op->src[1];
    ggml_tensor *       dst  = op;

    GGML_TENSOR_দ্বিমিক_OP_LOCALS

    ধ্রুবক int64_t nc = ne00;
    ধ্রুবক int64_t nr = ggml_nelements(src1);

    প্রতিজ্ঞা(ne0 == nc);
    প্রতিজ্ঞা(ne02 == ne11);
    প্রতিজ্ঞা(nb00 == আকার(দশমিক));
    প্রতিজ্ঞা(ggml_nrows(op) == nr);

    ধ্রুবক পূর্ণসংখ্যা ith = params->ith;
    ধ্রুবক পূর্ণসংখ্যা nth = params->nth;

    পূর্ণসংখ্যা rows_nth = nth;
    পূর্ণসংখ্যা cols_nth = 1;

    যদি (nr == 1) {
        rows_nth = 1;
        cols_nth = nth;
    }

    // rows per thread
    ধ্রুবক পূর্ণসংখ্যা dr = (nr + rows_nth - 1) / rows_nth;
    ধ্রুবক পূর্ণসংখ্যা dc = (nc + cols_nth - 1) / cols_nth;

    পূর্ণসংখ্যা rows_ith = ith % rows_nth;
    পূর্ণসংখ্যা cols_ith = ith % cols_nth;

    // row range for this thread
    ধ্রুবক পূর্ণসংখ্যা ir0 = dr * rows_ith;
    ধ্রুবক পূর্ণসংখ্যা ir1 = MIN(ir0 + dr, nr);

    ধ্রুবক পূর্ণসংখ্যা cr0 = dc * cols_ith;
    ধ্রুবক পূর্ণসংখ্যা cr1 = MIN(cr0 + dc, nc);

    জন্য (int64_t i = ir0; i < ir1; ++i) {
        ধ্রুবক int64_t i12 = i / (ne11 * ne10);
        ধ্রুবক int64_t i11 = (i - i12 * ne11 * ne10) / ne10;
        ধ্রুবক int64_t i10 = (i - i12 * ne11 * ne10 - i11 * ne10);
        ধ্রুবক int64_t i01 = *(int32_t *) ((অক্ষর *) src1->তথ্য + i10 * nb10 + i11 * nb11 + i12 * nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        memcpy1d(((অক্ষর *) dst->তথ্য + i10 * nb1 + i11 * nb2 + i12 * nb3) + cr0 * আকার(T),
                 ((অক্ষর *) src0->তথ্য + i01 * nb01 + i11 * nb02 + i12 * nb03) + cr0 * আকার(T),
                 (cr1 - cr0) * আকার(T));
    }
}

template <typename T> শূন্য forward_concat(ggml_compute_params * params, ggml_tensor * op) {
    ধ্রুবক ggml_tensor * src0 = op->src[0];
    ধ্রুবক ggml_tensor * src1 = op->src[1];
    ggml_tensor *       dst  = op;

    GGML_ASSERT(ggml_type_আকার(src0->type) == আকার(দশমিক));

    GGML_TENSOR_দ্বিমিক_OP_LOCALS

    ধ্রুবক int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim == 0 && nb0 == আকার(দশমিক) && nb1 == আকার(দশমিক) * (ne00 + ne10));

    ধ্রুবক int64_t nr = ggml_nrows(dst);
    ধ্রুবক int64_t nc = ne0;

    ধ্রুবক পূর্ণসংখ্যা ith = params->ith;
    ধ্রুবক পূর্ণসংখ্যা nth = params->nth;

    পূর্ণসংখ্যা rows_nth = nth;
    পূর্ণসংখ্যা cols_nth = 1;

    যদি (nr == 1) {
        rows_nth = 1;
        cols_nth = nth;
    }

    ধ্রুবক পূর্ণসংখ্যা dr = (nr + rows_nth - 1) / rows_nth;
    ধ্রুবক পূর্ণসংখ্যা dc = (nc + cols_nth - 1) / cols_nth;

    পূর্ণসংখ্যা rows_ith = ith % rows_nth;
    পূর্ণসংখ্যা cols_ith = ith % cols_nth;

    // row range for this thread
    ধ্রুবক পূর্ণসংখ্যা ir0 = dr * rows_ith;
    ধ্রুবক পূর্ণসংখ্যা ir1 = MIN(ir0 + dr, nr);

    ধ্রুবক পূর্ণসংখ্যা cr0 = dc * cols_ith;
    ধ্রুবক পূর্ণসংখ্যা cr1 = MIN(cr0 + dc, nc);

    int64_t o[4] = { 0, 0, 0, 0 };
    o[dim]       = src0->ne[dim];
    ধ্রুবক দশমিক * x;

    জন্য (int64_t i = ir0; i < ir1; ++i) {
        ধ্রুবক int64_t i3 = i / (ne02 * ne01);
        ধ্রুবক int64_t i2 = (i - i3 * ne02 * ne01) / ne01;
        ধ্রুবক int64_t i1 = (i - i3 * ne02 * ne01 - i2 * ne01);

        জন্য (পূর্ণসংখ্যা i0 = cr0; i0 < cr1; i0++) {
            যদি (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                x = (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) src0->তথ্য + (i0) *nb00 + (i1) *nb01 + (i2) *nb02 + (i3) *nb03);
            } অন্যথা {
                x = (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) src1->তথ্য + (i0 - o[0]) * nb10 + (i1 - o[1]) * nb11 +
                                     (i2 - o[2]) * nb12 + (i3 - o[3]) * nb13);
            }

            দশমিক * y = (দশমিক *) ((অক্ষর *) dst->তথ্য + i0 * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3);

            *y = *x;
        }
    }
}

template শূন্য forward_দ্বিমিক<GGML_OP_ADD, দশমিক>(ggml_compute_params * params, ggml_tensor * op);
template শূন্য forward_দ্বিমিক<GGML_OP_SUB, দশমিক>(ggml_compute_params * params, ggml_tensor * op);
template শূন্য forward_দ্বিমিক<GGML_OP_MUL, দশমিক>(ggml_compute_params * params, ggml_tensor * op);
template শূন্য forward_দ্বিমিক<GGML_OP_DIV, দশমিক>(ggml_compute_params * params, ggml_tensor * op);
template শূন্য forward_দ্বিমিক<GGML_OP_ADD, _Float16>(ggml_compute_params * params, ggml_tensor * op);
template শূন্য forward_দ্বিমিক<GGML_OP_SUB, _Float16>(ggml_compute_params * params, ggml_tensor * op);
template শূন্য forward_দ্বিমিক<GGML_OP_MUL, _Float16>(ggml_compute_params * params, ggml_tensor * op);
template শূন্য forward_দ্বিমিক<GGML_OP_DIV, _Float16>(ggml_compute_params * params, ggml_tensor * op);
template শূন্য forward_যোগ_rows<দশমিক>(ধ্রুবক ggml_compute_params * params, ggml_tensor * op);
template শূন্য forward_যোগ_rows<_Float16>(ধ্রুবক ggml_compute_params * params, ggml_tensor * op);
template শূন্য forward_repeat_nrows<int32_t>(ggml_compute_params * params, ggml_tensor * op);
template শূন্য forward_repeat_nrows<int16_t>(ggml_compute_params * params, ggml_tensor * op);
template শূন্য forward_repeat_dim1<int32_t>(ggml_compute_params * params, ggml_tensor * op);
template শূন্য forward_repeat_dim1<int16_t>(ggml_compute_params * params, ggml_tensor * op);
template শূন্য forward_get_rows<int32_t>(ggml_compute_params * params, ggml_tensor * op);
template শূন্য forward_get_rows<int16_t>(ggml_compute_params * params, ggml_tensor * op);
template শূন্য forward_concat<int32_t>(ggml_compute_params * params, ggml_tensor * op);
template শূন্য forward_concat<int16_t>(ggml_compute_params * params, ggml_tensor * op);

}  // namespace spacemit_kernels::rvv
