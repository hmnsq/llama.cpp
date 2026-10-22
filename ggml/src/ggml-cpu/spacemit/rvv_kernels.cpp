#شامل "rvv_kernels.h"

#شامل "common.h"
#شامل "ggml.h"
#شامل "ops.h"
#شامل "string.h"

#شامل <algorithm>
#شامل <cmath>
#شامل <cstdint>
#شامل <stdexcept>

#اگر !defined(__riscv_v) || !defined(__riscv_v_intrinsic)
#    error "riscv v extension or v_intrinsic not enabled"
#ورنہ
#    include <riscv_vector.h>
#اگر_ختم

#اگر !defined(__riscv_zfh)
#    error "riscv zfh extension not enabled"
#اگر_ختم

#اگر defined(__GNUC__)
#    pragma GCC diagnostic ignored "-Woverlength-strings"
#    pragma GCC diagnostic ignored "-Wcast-qual"
#    pragma GCC diagnostic ignored "-Wunused-parameter"
#اگر_ختم

نامخلا spacemit_kernels::rvv {

نامخلا {

خودکار align_اوپر(size_t value, size_t alignment) {
    واپس (value + alignment - 1) / alignment * alignment;
}

ساکن اندرونی منطقی flash_attn_ext_supported_d_vlen1024_vf16(int64_t d) {
    واپس d > 0 && d <= 128;
}

ساکن اندرونی منطقی flash_attn_ext_supported_shape_vlen1024_vf16(int64_t DK, int64_t DV) {
    واپس flash_attn_ext_supported_d_vlen1024_vf16(DK) && flash_attn_ext_supported_d_vlen1024_vf16(DV);
}

ساکن اندرونی اعشاری reduce_مجموع_f32m4_vlen1024(vfloat32m4_t v, size_t vl) {
    vfloat32m1_t s_v = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    s_v              = __riscv_vfredusum_vs_f32m4_f32m1(v, s_v, vl);
    واپس __riscv_vfmv_f_s_f32m1_f32(s_v);
}

ساکن اندرونی اعشاری reduce_مجموع_f32m2_vlen1024(vfloat32m2_t v, size_t vl) {
    vfloat32m1_t s_v = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    s_v              = __riscv_vfredusum_vs_f32m2_f32m1(v, s_v, vl);
    واپس __riscv_vfmv_f_s_f32m1_f32(s_v);
}

// Adapted from ggml_v_expf_m2 in vec.h. This is accurate enough for softmax.
ساکن اندرونی vfloat32m2_t rvv_expf_approx_f32m2(vfloat32m2_t x, size_t vl) {
    ثابت vfloat32m2_t r = __riscv_vfmv_v_f_f32m2(0x1.8p23f, vl);
    ثابت vfloat32m2_t z = __riscv_vfmacc_vf_f32m2(r, 0x1.715476p+0f, x, vl);
    ثابت vfloat32m2_t n = __riscv_vfsub_vv_f32m2(z, r, vl);
    ثابت vfloat32m2_t b =
        __riscv_vfnmsac_vf_f32m2(__riscv_vfnmsac_vf_f32m2(x, 0x1.62e4p-1f, n, vl), 0x1.7f7d1cp-20f, n, vl);
    ثابت vuint32m2_t  e = __riscv_vsll_vx_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(z), 23, vl);
    ثابت vfloat32m2_t k = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vadd_vx_u32m2(e, 0x3f800000, vl));
    ثابت vbool16_t    c = __riscv_vmfgt_vf_f32m2_b16(__riscv_vfabs_v_f32m2(n, vl), 126.0f, vl);
    ثابت vfloat32m2_t u = __riscv_vfmul_vv_f32m2(b, b, vl);
    ثابت vfloat32m2_t j = __riscv_vfmacc_vv_f32m2(
        __riscv_vfmul_vf_f32m2(b, 0x1.ffffecp-1f, vl),
        __riscv_vfmacc_vv_f32m2(
            __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(0x1.fffdb6p-2f, vl), 0x1.555e66p-3f, b, vl),
            __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(0x1.573e2ep-5f, vl), 0x1.0e4020p-7f, b, vl), u, vl),
        u, vl);

    اگر (!__riscv_vcpop_m_b16(c, vl)) {
        واپس __riscv_vfmacc_vv_f32m2(k, j, k, vl);
    }

    ثابت vbool16_t    dm = __riscv_vmfle_vf_f32m2_b16(n, 0.0f, vl);
    ثابت vuint32m2_t  d  = __riscv_vmerge_vxm_u32m2(__riscv_vmv_v_x_u32m2(0, vl), 0x82000000, dm, vl);
    ثابت vfloat32m2_t s1 = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vadd_vx_u32m2(d, 0x7f000000, vl));
    ثابت vfloat32m2_t s2 = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vsub_vv_u32m2(e, d, vl));
    ثابت vfloat32m2_t r1 =
        __riscv_vmerge_vvm_f32m2(__riscv_vfmacc_vv_f32m2(k, k, j, vl),
                                 __riscv_vfmul_vv_f32m2(__riscv_vfmacc_vv_f32m2(s2, s2, j, vl), s1, vl), c, vl);
    واپس __riscv_vmerge_vvm_f32m2(r1, __riscv_vfmul_vv_f32m2(s1, s1, vl),
                                    __riscv_vmfgt_vf_f32m2_b16(__riscv_vfabs_v_f32m2(n, vl), 192.0f, vl), vl);
}

ساکن اندرونی vfloat32m2_t rvv_tanh_approx_f32m2(vfloat32m2_t x, size_t vl) {
    ثابت vfloat32m2_t abs_x       = __riscv_vfabs_v_f32m2(x, vl);
    ثابت vfloat32m2_t neg_2_abs   = __riscv_vfmul_vf_f32m2(abs_x, -2.0f, vl);
    ثابت vfloat32m2_t exp_رکن    = rvv_expf_approx_f32m2(neg_2_abs, vl);
    ثابت vfloat32m2_t numerator   = __riscv_vfsub_vf_f32m2(exp_رکن, 1.0f, vl);
    ثابت vfloat32m2_t denominator = __riscv_vfadd_vf_f32m2(exp_رکن, 1.0f, vl);
    ثابت vfloat32m2_t tanh_abs    = __riscv_vfneg_v_f32m2(__riscv_vfdiv_vv_f32m2(numerator, denominator, vl), vl);
    ثابت vbool16_t    neg_نقاب    = __riscv_vmflt_vf_f32m2_b16(x, 0.0f, vl);
    ثابت vfloat32m2_t tanh_neg    = __riscv_vfneg_v_f32m2(tanh_abs, vl);
    واپس __riscv_vmerge_vvm_f32m2(tanh_abs, tanh_neg, neg_نقاب, vl);
}

ساکن خالی rvv_softcap_tanh_inplace_f32(اعشاری * dst, int64_t dst_stride, int64_t tile_صفیں, int64_t n, اعشاری softcap) {
    برائے (صحیح tq = 0; tq < tile_صفیں; ++tq, dst += dst_stride) {
        اعشاری * dst_صف   = dst;
        int64_t بقایا = n;
        جب (بقایا > 0) {
            ثابت size_t vl = __riscv_vsetvl_e32m2(بقایا);
            vfloat32m2_t v  = __riscv_vle32_v_f32m2(dst_صف, vl);
            v               = rvv_tanh_approx_f32m2(v, vl);
            v               = __riscv_vfmul_vf_f32m2(v, softcap, vl);
            __riscv_vse32_v_f32m2(dst_صف, v, vl);
            dst_صف += vl;
            بقایا -= vl;
        }
    }
}

ساکن اندرونی اعشاری rvv_softmax_exp_inplace_f32(اعشاری * dst, int64_t n, اعشاری max_قدر) {
    اعشاری صف_مجموع = 0.0f;
    جب (n > 0) {
        ثابت size_t vl = __riscv_vsetvl_e32m2(n);
        vfloat32m2_t v  = __riscv_vle32_v_f32m2(dst, vl);
        v               = __riscv_vfsub_vf_f32m2(v, max_قدر, vl);
        v               = rvv_expf_approx_f32m2(v, vl);
        __riscv_vse32_v_f32m2(dst, v, vl);
        صف_مجموع += reduce_مجموع_f32m2_vlen1024(v, vl);
        dst += vl;
        n -= vl;
    }
    واپس صف_مجموع;
}

ساکن اندرونی اعشاری rvv_جوڑو_max_inplace_f32(اعشاری * dst, ثابت اعشاری * src, int64_t n) {
    اعشاری max_val = -INFINITY;
    جب (n > 0) {
        ثابت size_t vl   = __riscv_vsetvl_e32m4(n);
        vfloat32m4_t vdst = __riscv_vle32_v_f32m4(dst, vl);
        vfloat32m4_t vsrc = __riscv_vle32_v_f32m4(src, vl);
        vdst              = __riscv_vfadd_vv_f32m4(vdst, vsrc, vl);
        __riscv_vse32_v_f32m4(dst, vdst, vl);

        vfloat32m1_t بیج = __riscv_vfmv_v_f_f32m1(max_val, 1);
        بیج              = __riscv_vfredmax_vs_f32m4_f32m1(vdst, بیج, vl);
        max_val           = __riscv_vfmv_f_s_f32m1_f32(بیج);

        dst += vl;
        src += vl;
        n -= vl;
    }
    واپس max_val;
}

ساکن اندرونی اعشاری rvv_softcap_جوڑو_max_inplace_f32(اعشاری * dst, ثابت اعشاری * src, int64_t n, اعشاری softcap) {
    اگر (softcap == 0.0f) {
        واپس rvv_جوڑو_max_inplace_f32(dst, src, n);
    }

    اعشاری max_val = -INFINITY;
    جب (n > 0) {
        ثابت size_t vl   = __riscv_vsetvl_e32m2(n);
        vfloat32m2_t vdst = __riscv_vle32_v_f32m2(dst, vl);
        vfloat32m2_t vsrc = __riscv_vle32_v_f32m2(src, vl);
        vdst              = rvv_tanh_approx_f32m2(vdst, vl);
        vdst              = __riscv_vfmul_vf_f32m2(vdst, softcap, vl);
        vdst              = __riscv_vfadd_vv_f32m2(vdst, vsrc, vl);
        __riscv_vse32_v_f32m2(dst, vdst, vl);

        vfloat32m1_t بیج = __riscv_vfmv_v_f_f32m1(max_val, 1);
        بیج              = __riscv_vfredmax_vs_f32m2_f32m1(vdst, بیج, vl);
        max_val           = __riscv_vfmv_f_s_f32m1_f32(بیج);

        dst += vl;
        src += vl;
        n -= vl;
    }
    واپس max_val;
}

ساکن اندرونی خالی rvv_zero_f32(اعشاری * dst, int64_t n) {
    جب (n > 0) {
        ثابت size_t       vl = __riscv_vsetvl_e32m4(n);
        ثابت vfloat32m4_t z  = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        __riscv_vse32_v_f32m4(dst, z, vl);
        dst += vl;
        n -= vl;
    }
}

ساکن اندرونی خالی rvv_scale_f32(اعشاری * dst, اعشاری scale, int64_t n) {
    جب (n > 0) {
        ثابت size_t vl = __riscv_vsetvl_e32m4(n);
        vfloat32m4_t v  = __riscv_vle32_v_f32m4(dst, vl);
        v               = __riscv_vfmul_vf_f32m4(v, scale, vl);
        __riscv_vse32_v_f32m4(dst, v, vl);
        dst += vl;
        n -= vl;
    }
}

ساکن اندرونی خالی rvv_جوڑو_inplace_f32(اعشاری *       dst,
                                       int64_t       dst_stride,
                                       ثابت اعشاری * src,
                                       int64_t       src_stride,
                                       int64_t       tile_صفیں,
                                       int64_t       n) {
    برائے (صحیح tq = 0; tq < tile_صفیں; ++tq, dst += dst_stride, src += src_stride) {
        int64_t       بقایا = n;
        اعشاری *       dst_صف   = dst;
        ثابت اعشاری * src_صف   = src;
        جب (بقایا > 0) {
            ثابت size_t vl   = __riscv_vsetvl_e32m4(بقایا);
            vfloat32m4_t vdst = __riscv_vle32_v_f32m4(dst_صف, vl);
            vfloat32m4_t vsrc = __riscv_vle32_v_f32m4(src_صف, vl);
            vdst              = __riscv_vfadd_vv_f32m4(vdst, vsrc, vl);
            __riscv_vse32_v_f32m4(dst_صف, vdst, vl);
            dst_صف += vl;
            src_صف += vl;
            بقایا -= vl;
        }
    }
}

ساکن اندرونی اعشاری rvv_max_f32(ثابت اعشاری * src, int64_t n) {
    اعشاری max_val = -INFINITY;
    جب (n > 0) {
        ثابت size_t       vl   = __riscv_vsetvl_e32m4(n);
        ثابت vfloat32m4_t v    = __riscv_vle32_v_f32m4(src, vl);
        vfloat32m1_t       بیج = __riscv_vfmv_v_f_f32m1(max_val, 1);
        بیج                    = __riscv_vfredmax_vs_f32m4_f32m1(v, بیج, vl);
        max_val                 = __riscv_vfmv_f_s_f32m1_f32(بیج);
        src += vl;
        n -= vl;
    }
    واپس max_val;
}

ساکن خالی rvv_pack_f32_as_scaled_f16(خالی *       dst,
                                       int64_t      dst_صف_stride,
                                       ثابت خالی * src,
                                       int64_t      src_صف_stride,
                                       int64_t      tile_صفیں,
                                       int64_t      n,
                                       اعشاری        scale) {
    برائے (صحیح tq = 0; tq < tile_صفیں; ++tq) {
        ثابت اعشاری * صف_ptr     = (ثابت اعشاری *) ((ثابت حرف *) src + tq * src_صف_stride);
        _Float16 *    dst_صف_ptr = (_Float16 *) ((حرف *) dst + tq * dst_صف_stride);
        int64_t       بقایا   = n;
        جب (بقایا > 0) {
            ثابت size_t vl        = __riscv_vsetvl_e32m4(بقایا);
            vfloat32m4_t v32       = __riscv_vle32_v_f32m4(صف_ptr, vl);
            v32                    = __riscv_vfmul_vf_f32m4(v32, scale, vl);
            ثابت vfloat16m2_t v16 = __riscv_vfncvt_f_f_w_f16m2(v32, vl);
            __riscv_vse16_v_f16m2(dst_صف_ptr, v16, vl);
            dst_صف_ptr += vl;
            صف_ptr += vl;
            بقایا -= vl;
        }
    }
}

ساکن خالی rvv_pack_scaled_f16_as_f32(خالی *       dst,
                                       int64_t      dst_صف_stride,
                                       ثابت خالی * src,
                                       int64_t      src_صف_stride,
                                       int64_t      tile_صفیں,
                                       int64_t      n,
                                       اعشاری        scale) {
    برائے (صحیح tq = 0; tq < tile_صفیں; ++tq) {
        ثابت _Float16 * صف_ptr     = (ثابت _Float16 *) ((ثابت حرف *) src + tq * src_صف_stride);
        اعشاری *          dst_صف_ptr = (اعشاری *) ((حرف *) dst + tq * dst_صف_stride);
        int64_t          بقایا   = n;
        جب (بقایا > 0) {
            ثابت size_t       vl  = __riscv_vsetvl_e16m2(بقایا);
            ثابت vfloat16m2_t v16 = __riscv_vle16_v_f16m2(صف_ptr, vl);
            vfloat32m4_t       v32 = __riscv_vfwcvt_f_f_v_f32m4(v16, vl);
            v32                    = __riscv_vfmul_vf_f32m4(v32, scale, vl);
            __riscv_vse32_v_f32m4(dst_صف_ptr, v32, vl);
            dst_صف_ptr += vl;
            صف_ptr += vl;
            بقایا -= vl;
        }
    }
}

ساکن خالی rvv_pack_scaled_f32_as_f32(خالی *       dst,
                                       int64_t      dst_صف_stride,
                                       ثابت خالی * src,
                                       int64_t      src_صف_stride,
                                       int64_t      tile_صفیں,
                                       int64_t      n,
                                       اعشاری *      scale) {
    برائے (صحیح tq = 0; tq < tile_صفیں; ++tq) {
        ثابت اعشاری * صف_ptr     = (ثابت اعشاری *) ((ثابت حرف *) src + tq * src_صف_stride);
        اعشاری *       dst_صف_ptr = (اعشاری *) ((حرف *) dst + tq * dst_صف_stride);
        int64_t       بقایا   = n;
        جب (بقایا > 0) {
            ثابت size_t vl  = __riscv_vsetvl_e32m4(بقایا);
            vfloat32m4_t v32 = __riscv_vle32_v_f32m4(صف_ptr, vl);
            v32              = __riscv_vfmul_vf_f32m4(v32, scale[tq], vl);
            __riscv_vse32_v_f32m4(dst_صف_ptr, v32, vl);
            dst_صف_ptr += vl;
            صف_ptr += vl;
            بقایا -= vl;
        }
    }
}

ساکن اندرونی خالی rvv_transposed_s32_mn_مقصد_nm(int8_t * dst,
                                               int64_t  n_dst_stride,
                                               int8_t * src,
                                               int64_t  m_src_stride,
                                               int64_t  m,
                                               int64_t  n) {
    int8_t * اندر  = src;
    int8_t * برآمد = dst;

    __asm__ متلون(
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

        : [src] "+r"(اندر), [dst] "+r"(برآمد), [isz0] "+r"(m)
        : [isz1] "r"(n), [is0] "r"(m_src_stride), [os0] "r"(n_dst_stride)
        : "cc", "t0", "t1", "t2", "t3", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "a1");
}

ساکن اندرونی خالی rvv_transposed_s16_mn_مقصد_nm(int8_t * dst,
                                               int64_t  n_dst_stride,
                                               int8_t * src,
                                               int64_t  m_src_stride,
                                               int64_t  m,
                                               int64_t  n) {
    int8_t * اندر  = src;
    int8_t * برآمد = dst;

    __asm__ متلون(
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

        : [src] "+r"(اندر), [dst] "+r"(برآمد), [isz0] "+r"(m)
        : [isz1] "r"(n), [is0] "r"(m_src_stride), [os0] "r"(n_dst_stride)
        : "cc", "t0", "t1", "t2", "t3", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "a1");
}

ساکن اندرونی خالی rvv_qk_dot_tile_f16_x1(اعشاری *          dst,
                                          ثابت _Float16 * q_صف,
                                          ثابت _Float16 * k_pack,
                                          int64_t          dk,
                                          int64_t          kv_tile) {
    ثابت size_t vl  = __riscv_vsetvl_e16m1(kv_tile);
    vfloat32m2_t acc = __riscv_vfmv_v_f_f32m2(0.0f, vl);

    برائے (int64_t d = 0; d < dk; ++d) {
        ثابت vfloat16m1_t k_vec = __riscv_vle16_v_f16m1(k_pack + d * ggml_fa_tile_config::KV, vl);
        acc                      = __riscv_vfwmacc_vf_f32m2(acc, q_صف[d], k_vec, vl);
    }

    __riscv_vse32_v_f32m2(dst, acc, vl);
}

ساکن اندرونی خالی rvv_qk_dot_tile_f16_x4(اعشاری *          dst0,
                                          اعشاری *          dst1,
                                          اعشاری *          dst2,
                                          اعشاری *          dst3,
                                          ثابت _Float16 * q0,
                                          ثابت _Float16 * q1,
                                          ثابت _Float16 * q2,
                                          ثابت _Float16 * q3,
                                          ثابت _Float16 * k_pack,
                                          int64_t          dk,
                                          int64_t          kv_tile) {
    ثابت size_t vl   = __riscv_vsetvl_e16m1(kv_tile);
    vfloat32m2_t acc0 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    vfloat32m2_t acc1 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    vfloat32m2_t acc2 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    vfloat32m2_t acc3 = __riscv_vfmv_v_f_f32m2(0.0f, vl);

    برائے (int64_t d = 0; d < dk; ++d) {
        ثابت vfloat16m1_t k_vec = __riscv_vle16_v_f16m1(k_pack + d * ggml_fa_tile_config::KV, vl);
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

ساکن اندرونی خالی rvv_pv_accumulate_f16_x1(اعشاری *          dst,
                                            ثابت اعشاری *    prob,
                                            ثابت _Float16 * v_pack,
                                            int64_t          kv_tile,
                                            int64_t          dv) {
    int64_t d_بایاں = dv;
    int64_t d_off  = 0;

    جب (d_بایاں > 0) {
        ثابت size_t vl  = __riscv_vsetvl_e16m2(d_بایاں);
        vfloat32m4_t acc = __riscv_vle32_v_f32m4(dst + d_off, vl);

        برائے (int64_t tk = 0; tk < kv_tile; ++tk) {
            ثابت vfloat16m2_t v16 = __riscv_vle16_v_f16m2(v_pack + tk * dv + d_off, vl);
            ثابت vfloat32m4_t v32 = __riscv_vfwcvt_f_f_v_f32m4(v16, vl);
            acc                    = __riscv_vfmacc_vf_f32m4(acc, prob[tk], v32, vl);
        }

        __riscv_vse32_v_f32m4(dst + d_off, acc, vl);
        d_بایاں -= vl;
        d_off += vl;
    }
}

ساکن اندرونی خالی rvv_pv_accumulate_f16_x4(اعشاری *          dst0,
                                            اعشاری *          dst1,
                                            اعشاری *          dst2,
                                            اعشاری *          dst3,
                                            ثابت اعشاری *    prob0,
                                            ثابت اعشاری *    prob1,
                                            ثابت اعشاری *    prob2,
                                            ثابت اعشاری *    prob3,
                                            ثابت _Float16 * v_pack,
                                            int64_t          kv_tile,
                                            int64_t          dv) {
    int64_t d_بایاں = dv;
    int64_t d_off  = 0;

    جب (d_بایاں > 0) {
        ثابت size_t vl   = __riscv_vsetvl_e16m2(d_بایاں);
        vfloat32m4_t acc0 = __riscv_vle32_v_f32m4(dst0 + d_off, vl);
        vfloat32m4_t acc1 = __riscv_vle32_v_f32m4(dst1 + d_off, vl);
        vfloat32m4_t acc2 = __riscv_vle32_v_f32m4(dst2 + d_off, vl);
        vfloat32m4_t acc3 = __riscv_vle32_v_f32m4(dst3 + d_off, vl);

        برائے (int64_t tk = 0; tk < kv_tile; ++tk) {
            ثابت vfloat16m2_t v16 = __riscv_vle16_v_f16m2(v_pack + tk * dv + d_off, vl);
            ثابت vfloat32m4_t v32 = __riscv_vfwcvt_f_f_v_f32m4(v16, vl);
            acc0                   = __riscv_vfmacc_vf_f32m4(acc0, prob0[tk], v32, vl);
            acc1                   = __riscv_vfmacc_vf_f32m4(acc1, prob1[tk], v32, vl);
            acc2                   = __riscv_vfmacc_vf_f32m4(acc2, prob2[tk], v32, vl);
            acc3                   = __riscv_vfmacc_vf_f32m4(acc3, prob3[tk], v32, vl);
        }

        __riscv_vse32_v_f32m4(dst0 + d_off, acc0, vl);
        __riscv_vse32_v_f32m4(dst1 + d_off, acc1, vl);
        __riscv_vse32_v_f32m4(dst2 + d_off, acc2, vl);
        __riscv_vse32_v_f32m4(dst3 + d_off, acc3, vl);
        d_بایاں -= vl;
        d_off += vl;
    }
}

ساکن اندرونی خالی rvv_qk_dot_tile(اعشاری *       dst,
                                   ثابت اعشاری * q_صف,
                                   ثابت اعشاری * k_pack,
                                   int64_t       dk,
                                   int64_t       kv_tile,
                                   اعشاری         scale) {
    ثابت size_t vl  = __riscv_vsetvl_e32m4(kv_tile);
    vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);

    برائے (int64_t d = 0; d < dk; ++d) {
        ثابت vfloat32m4_t k_vec = __riscv_vle32_v_f32m4(k_pack + d * kv_tile, vl);
        acc                      = __riscv_vfmacc_vf_f32m4(acc, q_صف[d] * scale, k_vec, vl);
    }

    __riscv_vse32_v_f32m4(dst, acc, vl);
}

ساکن اندرونی خالی rvv_pv_accumulate(اعشاری *       dst,
                                     ثابت اعشاری * prob,
                                     ثابت اعشاری * v_pack,
                                     int64_t       kv_tile,
                                     int64_t       dv) {
    int64_t d_بایاں = dv;
    int64_t d_off  = 0;

    جب (d_بایاں > 0) {
        ثابت size_t vl  = __riscv_vsetvl_e32m4(d_بایاں);
        vfloat32m4_t acc = __riscv_vle32_v_f32m4(dst + d_off, vl);

        برائے (int64_t tk = 0; tk < kv_tile; ++tk) {
            ثابت vfloat32m4_t v_vec = __riscv_vle32_v_f32m4(v_pack + tk * dv + d_off, vl);
            acc                      = __riscv_vfmacc_vf_f32m4(acc, prob[tk], v_vec, vl);
        }

        __riscv_vse32_v_f32m4(dst + d_off, acc, vl);
        d_بایاں -= vl;
        d_off += vl;
    }
}

ساکن خالی ترتیب_بدلو_transpose_impl(ثابت ggml_tensor * src0,
                                   ggml_tensor *       dst,
                                   int64_t             batch,
                                   int64_t             m,
                                   int64_t             n,
                                   int64_t             batch_stride,
                                   int64_t             m_src_stride,
                                   int64_t             n_src_stride,
                                   int64_t             n_dst_stride,
                                   صحیح                 ith,
                                   صحیح                 nth) {
    GGML_ASSERT(n_src_stride == حجم(int32_t) || n_src_stride == حجم(int16_t));

    اگر (n_src_stride == حجم(int32_t)) {
        برائے (int64_t bi = ith; bi < batch; bi += nth) {
            rvv_transposed_s32_mn_مقصد_nm((int8_t *) ((حرف *) dst->ڈیٹا + bi * batch_stride), n_dst_stride,
                                        (int8_t *) ((حرف *) src0->ڈیٹا + bi * batch_stride), m_src_stride, m, n);
        }
    } ورنہ اگر (n_src_stride == حجم(int16_t)) {
        برائے (int64_t bi = ith; bi < batch; bi += nth) {
            rvv_transposed_s16_mn_مقصد_nm((int8_t *) ((حرف *) dst->ڈیٹا + bi * batch_stride), n_dst_stride,
                                        (int8_t *) ((حرف *) src0->ڈیٹا + bi * batch_stride), m_src_stride, m, n);
        }
    } ورنہ {
        GGML_ABORT("not implemented");
    }
}

سانچہ <size_t QLEN>
ساکن خالی flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_mrow(اعشاری **            pq,
                                                                  ثابت حرف *        k_ڈیٹا_صف,
                                                                  ثابت حرف *        v_ڈیٹا_صف,
                                                                  ثابت ggml_fp16_t * mp,
                                                                  اعشاری **            sinks,
                                                                  اعشاری **            dst,
                                                                  اعشاری               scale,
                                                                  اعشاری               logit_softcap,
                                                                  اعشاری               ڈھلان,
                                                                  int64_t             nek1,
                                                                  int64_t             nbk1,
                                                                  int64_t             nbv1,
                                                                  int64_t             DV,
                                                                  int64_t             DK,
                                                                  خالی *              tcm_بفر,
                                                                  size_t              tcm_بفر_سائز) {
    GGML_ASSERT(flash_attn_ext_supported_shape_vlen1024_vf16(DK, DV));
    اعشاری S[QLEN] = { 0.0f };       //مجموع
    اعشاری M[QLEN] = { -INFINITY };  // maximum KQ value

    _Float16 *   kq16_بفر          = (_Float16 *) tcm_بفر;
    _Float16 *   qv_بفر            = kq16_بفر + QLEN * DV;
    ثابت size_t qkv_عارضی_بفر_سائز = (QLEN * DV + QLEN * DK) * حجم(_Float16);
    حرف *       kv_tile_بفر       = (حرف *) (qv_بفر + QLEN * DK);

    {
        vfloat16m2_t VKQ16_v = __riscv_vfmv_v_f_f16m2(0.0f, DV);
        برائے (int64_t i = 0; i < QLEN; ++i) {
            __riscv_vse16_v_f16m2(kq16_بفر + i * DV, VKQ16_v, DV);
            vfloat16m2_t Q_q_v = __riscv_vfncvt_f_f_w_f16m2(__riscv_vle32_v_f32m4(pq[i], DK), DK);
            __riscv_vse16_v_f16m2(qv_بفر + i * DK, Q_q_v, DK);
        }
    }

    ثابت uintptr_t scratch_addr = reinterpret_cast<uintptr_t>(kv_tile_بفر);
    ثابت size_t    scratch_سائز = tcm_بفر_سائز > qkv_عارضی_بفر_سائز ? tcm_بفر_سائز - qkv_عارضی_بفر_سائز : 0;
    ثابت uintptr_t kq_tile_addr = align_اوپر(scratch_addr, alignof(اعشاری));
    ثابت size_t    scratch_سابقہ = kq_tile_addr - scratch_addr;
    ثابت size_t    packed_tile_سائز =
        QLEN * حجم(اعشاری) + DK * حجم(_Float16) + DV * حجم(_Float16) + حجم(اعشاری);
    ثابت int64_t max_ic_tile_قدم = ((int64_t) __riscv_vsetvlmax_e16m1()) & ~((int64_t) 7);
    ثابت int64_t max_fit_by_tcm =
        scratch_سائز > scratch_سابقہ ? (int64_t) ((scratch_سائز - scratch_سابقہ) / packed_tile_سائز) : 0;
    ثابت int64_t ic_tile_قدم = std::min(max_ic_tile_قدم, max_fit_by_tcm) & ~((int64_t) 7);

    ثابت uintptr_t k_tile_addr  = kq_tile_addr + QLEN * ic_tile_قدم * حجم(اعشاری);
    ثابت uintptr_t v_tile_addr  = k_tile_addr + DK * ic_tile_قدم * حجم(_Float16);
    ثابت uintptr_t mv_tile_addr = v_tile_addr + ic_tile_قدم * DV * حجم(_Float16);

    اگر (ic_tile_قدم >= 8) {
        اعشاری *    kq_tile_بفر = reinterpret_cast<اعشاری *>(kq_tile_addr);
        _Float16 * k_tile_pack    = reinterpret_cast<_Float16 *>(k_tile_addr);
        _Float16 * v_tile_pack    = reinterpret_cast<_Float16 *>(v_tile_addr);
        اعشاری *    mv_tile_pack   = reinterpret_cast<اعشاری *>(mv_tile_addr);

        ثابت int64_t k_tile_byte_stride = ic_tile_قدم * (int64_t) حجم(_Float16);

        int64_t ic_قدم = 0;
        برائے (int64_t ic = 0; ic < nek1; ++ic) {
            ثابت اعشاری mv = mp ? ڈھلان * ((_Float16 *) mp)[ic] : 0.0f;

            اگر (mv != -INFINITY) {
                ثابت _Float16 * k_ڈیٹا = (ثابت _Float16 *) (k_ڈیٹا_صف + ic * nbk1);
                ثابت _Float16 * v_ڈیٹا = (ثابت _Float16 *) (v_ڈیٹا_صف + ic * nbv1);

                ثابت vfloat16m2_t k_ڈیٹا_v = __riscv_vle16_v_f16m2(k_ڈیٹا, DK);
                ثابت vfloat16m2_t v_ڈیٹا_v = __riscv_vle16_v_f16m2(v_ڈیٹا, DV);
                __riscv_vsse16_v_f16m2(k_tile_pack + ic_قدم, k_tile_byte_stride, k_ڈیٹا_v, DK);
                __riscv_vse16_v_f16m2(v_tile_pack + ic_قدم * DV, v_ڈیٹا_v, DV);
                mv_tile_pack[ic_قدم] = mv;
                ic_قدم++;
            }

            اگر (ic_قدم > 0 && (ic_قدم == ic_tile_قدم || ic == (nek1 - 1))) {
                اگر constexpr (QLEN == 4) {
                    ثابت size_t qk_vl   = __riscv_vsetvl_e16m1(ic_قدم);
                    vfloat32m2_t qk_acc0 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);
                    vfloat32m2_t qk_acc1 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);
                    vfloat32m2_t qk_acc2 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);
                    vfloat32m2_t qk_acc3 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);

                    برائے (int64_t d = 0; d < DK; ++d) {
                        ثابت vfloat16m1_t k_vec = __riscv_vle16_v_f16m1(k_tile_pack + d * ic_tile_قدم, qk_vl);
                        qk_acc0 = __riscv_vfwmacc_vf_f32m2(qk_acc0, qv_بفر[0 * DK + d], k_vec, qk_vl);
                        qk_acc1 = __riscv_vfwmacc_vf_f32m2(qk_acc1, qv_بفر[1 * DK + d], k_vec, qk_vl);
                        qk_acc2 = __riscv_vfwmacc_vf_f32m2(qk_acc2, qv_بفر[2 * DK + d], k_vec, qk_vl);
                        qk_acc3 = __riscv_vfwmacc_vf_f32m2(qk_acc3, qv_بفر[3 * DK + d], k_vec, qk_vl);
                    }

                    qk_acc0 = __riscv_vfmul_vf_f32m2(qk_acc0, scale, qk_vl);
                    qk_acc1 = __riscv_vfmul_vf_f32m2(qk_acc1, scale, qk_vl);
                    qk_acc2 = __riscv_vfmul_vf_f32m2(qk_acc2, scale, qk_vl);
                    qk_acc3 = __riscv_vfmul_vf_f32m2(qk_acc3, scale, qk_vl);

                    __riscv_vse32_v_f32m2(kq_tile_بفر + 0 * ic_tile_قدم, qk_acc0, qk_vl);
                    __riscv_vse32_v_f32m2(kq_tile_بفر + 1 * ic_tile_قدم, qk_acc1, qk_vl);
                    __riscv_vse32_v_f32m2(kq_tile_بفر + 2 * ic_tile_قدم, qk_acc2, qk_vl);
                    __riscv_vse32_v_f32m2(kq_tile_بفر + 3 * ic_tile_قدم, qk_acc3, qk_vl);
                } ورنہ {
                    static_assert(QLEN == 2, "unsupported QLEN");

                    ثابت size_t qk_vl   = __riscv_vsetvl_e16m1(ic_قدم);
                    vfloat32m2_t qk_acc0 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);
                    vfloat32m2_t qk_acc1 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);

                    برائے (int64_t d = 0; d < DK; ++d) {
                        ثابت vfloat16m1_t k_vec = __riscv_vle16_v_f16m1(k_tile_pack + d * ic_tile_قدم, qk_vl);
                        qk_acc0 = __riscv_vfwmacc_vf_f32m2(qk_acc0, qv_بفر[0 * DK + d], k_vec, qk_vl);
                        qk_acc1 = __riscv_vfwmacc_vf_f32m2(qk_acc1, qv_بفر[1 * DK + d], k_vec, qk_vl);
                    }

                    qk_acc0 = __riscv_vfmul_vf_f32m2(qk_acc0, scale, qk_vl);
                    qk_acc1 = __riscv_vfmul_vf_f32m2(qk_acc1, scale, qk_vl);

                    __riscv_vse32_v_f32m2(kq_tile_بفر + 0 * ic_tile_قدم, qk_acc0, qk_vl);
                    __riscv_vse32_v_f32m2(kq_tile_بفر + 1 * ic_tile_قدم, qk_acc1, qk_vl);
                }

                برائے (صحیح i = 0; i < QLEN; ++i) {
                    اعشاری *     صف_ptr = kq_tile_بفر + i * ic_tile_قدم;
                    ثابت اعشاری tile_max =
                        rvv_softcap_جوڑو_max_inplace_f32(صف_ptr, mv_tile_pack, ic_قدم, logit_softcap);

                    ثابت اعشاری Mold = M[i];

                    اگر (tile_max > Mold) {
                        ثابت اعشاری ms = expf(Mold - tile_max);
                        M[i]           = tile_max;
                        S[i] *= ms;

                        vfloat16m2_t VKQ16_v = __riscv_vle16_v_f16m2(kq16_بفر + i * DV, DV);
                        VKQ16_v              = __riscv_vfmul_vf_f16m2(VKQ16_v, (_Float16) ms, DV);
                        __riscv_vse16_v_f16m2(kq16_بفر + i * DV, VKQ16_v, DV);
                    }

                    S[i] += rvv_softmax_exp_inplace_f32(صف_ptr, ic_قدم, M[i]);
                }

                اگر constexpr (QLEN == 4) {
                    vfloat16m2_t pv_acc0 = __riscv_vle16_v_f16m2(kq16_بفر + 0 * DV, DV);
                    vfloat16m2_t pv_acc1 = __riscv_vle16_v_f16m2(kq16_بفر + 1 * DV, DV);
                    vfloat16m2_t pv_acc2 = __riscv_vle16_v_f16m2(kq16_بفر + 2 * DV, DV);
                    vfloat16m2_t pv_acc3 = __riscv_vle16_v_f16m2(kq16_بفر + 3 * DV, DV);

                    برائے (int64_t tk = 0; tk < ic_قدم; ++tk) {
                        ثابت vfloat16m2_t v16 = __riscv_vle16_v_f16m2(v_tile_pack + tk * DV, DV);
                        pv_acc0 =
                            __riscv_vfmacc_vf_f16m2(pv_acc0, (_Float16) kq_tile_بفر[0 * ic_tile_قدم + tk], v16, DV);
                        pv_acc1 =
                            __riscv_vfmacc_vf_f16m2(pv_acc1, (_Float16) kq_tile_بفر[1 * ic_tile_قدم + tk], v16, DV);
                        pv_acc2 =
                            __riscv_vfmacc_vf_f16m2(pv_acc2, (_Float16) kq_tile_بفر[2 * ic_tile_قدم + tk], v16, DV);
                        pv_acc3 =
                            __riscv_vfmacc_vf_f16m2(pv_acc3, (_Float16) kq_tile_بفر[3 * ic_tile_قدم + tk], v16, DV);
                    }

                    __riscv_vse16_v_f16m2(kq16_بفر + 0 * DV, pv_acc0, DV);
                    __riscv_vse16_v_f16m2(kq16_بفر + 1 * DV, pv_acc1, DV);
                    __riscv_vse16_v_f16m2(kq16_بفر + 2 * DV, pv_acc2, DV);
                    __riscv_vse16_v_f16m2(kq16_بفر + 3 * DV, pv_acc3, DV);
                } ورنہ {
                    static_assert(QLEN == 2, "unsupported QLEN");
                    vfloat16m2_t pv_acc0 = __riscv_vle16_v_f16m2(kq16_بفر + 0 * DV, DV);
                    vfloat16m2_t pv_acc1 = __riscv_vle16_v_f16m2(kq16_بفر + 1 * DV, DV);

                    برائے (int64_t tk = 0; tk < ic_قدم; ++tk) {
                        ثابت vfloat16m2_t v16 = __riscv_vle16_v_f16m2(v_tile_pack + tk * DV, DV);
                        pv_acc0 =
                            __riscv_vfmacc_vf_f16m2(pv_acc0, (_Float16) kq_tile_بفر[0 * ic_tile_قدم + tk], v16, DV);
                        pv_acc1 =
                            __riscv_vfmacc_vf_f16m2(pv_acc1, (_Float16) kq_tile_بفر[1 * ic_tile_قدم + tk], v16, DV);
                    }

                    __riscv_vse16_v_f16m2(kq16_بفر + 0 * DV, pv_acc0, DV);
                    __riscv_vse16_v_f16m2(kq16_بفر + 1 * DV, pv_acc1, DV);
                }

                ic_قدم = 0;
            }
        }
    } ورنہ {
        برائے (int64_t ic = 0; ic < nek1; ++ic) {
            ثابت اعشاری mv = mp ? ڈھلان * ((_Float16 *) mp)[ic] : 0.0f;

            ثابت حرف * k_ڈیٹا = k_ڈیٹا_صف + ic * nbk1;
            ثابت حرف * v_ڈیٹا = v_ڈیٹا_صف + ic * nbv1;

            vfloat16m2_t k_ڈیٹا_v;
            vfloat16m2_t v_ڈیٹا_v;

            اگر (mv != -INFINITY) {
                k_ڈیٹا_v = __riscv_vle16_v_f16m2((_Float16 *) k_ڈیٹا, DK);
                v_ڈیٹا_v = __riscv_vle16_v_f16m2((_Float16 *) v_ڈیٹا, DV);
            } ورنہ {
                جاری;
            }

            برائے (صحیح i = 0; i < QLEN; ++i) {
                vfloat16m2_t Q_q_v    = __riscv_vle16_v_f16m2(qv_بفر + i * DK, DK);
                vfloat32m4_t qk_acc_v = __riscv_vfwmul_vv_f32m4(k_ڈیٹا_v, Q_q_v, DK);
                اعشاری        s        = reduce_مجموع_f32m4_vlen1024(qk_acc_v, DK);
                s                     = s * scale;
                اگر (logit_softcap != 0.0f) {
                    s = logit_softcap * tanhf(s);
                }
                s += mv;

                ثابت اعشاری Mold = M[i];

                اعشاری ms = 1.0f;  // upon new higher max val, scale VKQ and KQ sum with this value
                اعشاری vs = 1.0f;  // post-softmax KQ value, expf(s - M)

                vfloat16m2_t VKQ16_v = __riscv_vle16_v_f16m2(kq16_بفر + i * DV, DV);
                اگر (s > M[i]) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M[i] = s;
                    ms   = expf(Mold - M[i]);

                    // V = V*expf(Mold - M)
                    VKQ16_v = __riscv_vfmul_vf_f16m2(VKQ16_v, ms, DV);
                } ورنہ {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M[i]);
                }
                VKQ16_v = __riscv_vfmacc_vf_f16m2(VKQ16_v, vs, v_ڈیٹا_v, DV);
                __riscv_vse16_v_f16m2(kq16_بفر + i * DV, VKQ16_v, DV);
                S[i] = S[i] * ms + vs;  // scale and increment sum with partial sum
            }
        }
    }

    برائے (صحیح i = 0; i < QLEN; ++i) {
        vfloat16m2_t VKQ16_v = __riscv_vle16_v_f16m2(kq16_بفر + i * DV, DV);
        vfloat32m4_t VKQ32_v = __riscv_vfwcvt_f_f_v_f32m4(VKQ16_v, DV);

        // sinks
        اگر (sinks[i]) {
            ثابت اعشاری s = *(sinks[i]);

            اعشاری ms = 1.0f;
            اعشاری vs = 1.0f;

            اگر (s > M[i]) {
                ms      = expf(M[i] - s);
                M[i]    = s;
                VKQ32_v = __riscv_vfmul_vf_f32m4(VKQ32_v, ms, DV);
            } ورنہ {
                vs = expf(s - M[i]);
            }

            S[i] = S[i] * ms + vs;
        }

        // V /= S
        ثابت اعشاری S_inv = S[i] == 0.0f ? 0.0f : 1.0f / S[i];

        VKQ32_v = __riscv_vfmul_vf_f32m4(VKQ32_v, S_inv, DV);

        __riscv_vse32_v_f32m4(dst[i], VKQ32_v, DV);
    }
}

ساکن خالی flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_m1(ثابت اعشاری *       pq,
                                                                ثابت حرف *        k_ڈیٹا_صف,
                                                                ثابت حرف *        v_ڈیٹا_صف,
                                                                ثابت ggml_fp16_t * mp,
                                                                ثابت اعشاری *       sinks,
                                                                اعشاری *             dst,
                                                                اعشاری               scale,
                                                                اعشاری               logit_softcap,
                                                                اعشاری               ڈھلان,
                                                                int64_t             nek1,
                                                                int64_t             nbk1,
                                                                int64_t             nbv1,
                                                                int64_t             DV,
                                                                int64_t             DK) {
    GGML_ASSERT(flash_attn_ext_supported_shape_vlen1024_vf16(DK, DV));

    اعشاری S = 0.0f;       //مجموع
    اعشاری M = -INFINITY;  // maximum KQ value

    vfloat16m2_t VKQ16_v = __riscv_vfmv_v_f_f16m2(0.0f, DV);

    vfloat16m2_t Q_q_v = __riscv_vfncvt_f_f_w_f16m2(__riscv_vle32_v_f32m4(pq, DK), DK);

    برائے (int64_t ic = 0; ic < nek1; ++ic) {
        ثابت اعشاری mv = mp ? ڈھلان * ((_Float16 *) mp)[ic] : 0.0f;
        اگر (mv == -INFINITY) {
            جاری;
        }

        ثابت حرف * k_ڈیٹا = k_ڈیٹا_صف + ic * nbk1;

        vfloat16m2_t k_ڈیٹا_v = __riscv_vle16_v_f16m2((_Float16 *) k_ڈیٹا, DK);

        vfloat32m4_t qk_acc_v = __riscv_vfwmul_vv_f32m4(k_ڈیٹا_v, Q_q_v, DK);
        اعشاری        s        = reduce_مجموع_f32m4_vlen1024(qk_acc_v, DK);

        s = s * scale;  // scale KQ value

        اگر (logit_softcap != 0.0f) {
            s = logit_softcap * tanhf(s);
        }

        s += mv;  // apply mask

        ثابت اعشاری Mold = M;

        اعشاری ms = 1.0f;  // upon new higher max val, scale VKQ and KQ sum with this value
        اعشاری vs = 1.0f;  // post-softmax KQ value, expf(s - M)

        ثابت حرف * v_ڈیٹا = v_ڈیٹا_صف + ic * nbv1;

        vfloat16m2_t v_ڈیٹا_v = __riscv_vle16_v_f16m2((_Float16 *) v_ڈیٹا, DV);

        اگر (s > M) {
            // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
            M  = s;
            ms = expf(Mold - M);

            // V = V*expf(Mold - M)
            VKQ16_v = __riscv_vfmul_vf_f16m2(VKQ16_v, ms, DV);
        } ورنہ {
            // no new maximum, ms == 1.0f, vs != 1.0f
            vs = expf(s - M);
        }

        VKQ16_v = __riscv_vfmacc_vf_f16m2(VKQ16_v, vs, v_ڈیٹا_v, DV);

        S = S * ms + vs;  // scale and increment sum with partial sum
    }

    vfloat32m4_t VKQ32_v = __riscv_vfwcvt_f_f_v_f32m4(VKQ16_v, DV);

    // sinks
    اگر (sinks) {
        ثابت اعشاری s = *sinks;

        اعشاری ms = 1.0f;
        اعشاری vs = 1.0f;

        اگر (s > M) {
            ms      = expf(M - s);
            M       = s;
            VKQ32_v = __riscv_vfmul_vf_f32m4(VKQ32_v, ms, DV);
        } ورنہ {
            vs = expf(s - M);
        }

        S = S * ms + vs;
    }

    // V /= S
    ثابت اعشاری S_inv = S == 0.0f ? 0.0f : 1.0f / S;

    VKQ32_v = __riscv_vfmul_vf_f32m4(VKQ32_v, S_inv, DV);

    __riscv_vse32_v_f32m4(dst, VKQ32_v, DV);
}

}  // namespace

خالی memcpy1d(خالی * dst, ثابت خالی * src, int64_t size) {
    size_t byte_سائز_all = size;
    size_t vlen          = __riscv_vlenb() * 8;
    اگر (vlen == 256) {
        // 1024 bytes
        __asm__ متلون(
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
            : [size] "r"(byte_سائز_all)
            : "cc", "t0", "t1");
    } ورنہ اگر (vlen == 1024) {
        // 2048 bytes
        __asm__ متلون(
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
            : [size] "r"(byte_سائز_all)
            : "cc", "t0", "t1", "t2", "t3", "t5");
    } ورنہ {
        __asm__ متلون(
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
            : [size] "r"(byte_سائز_all)
            : "cc", "t0", "t1", "t2", "t4", "t3");
    }
}

خالی memcpy2d(خالی * dst, int64_t dst_stride, ثابت خالی * src, int64_t src_stride, int64_t tile_صفیں, int64_t size) {
    برائے (int64_t i = 0; i < tile_صفیں; ++i) {
        memcpy1d((حرف *) dst + i * dst_stride, (ثابت حرف *) src + i * src_stride, size);
    }
}

خالی forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(ثابت ggml_compute_params * params,
                                                        ggml_tensor *               dst,
                                                        صحیح                         ir0,
                                                        صحیح                         ir1,
                                                        خالی *                      tcm_بفر,
                                                        size_t                      tcm_بفر_سائز) {
    ثابت ggml_tensor * q     = dst->src[0];
    ثابت ggml_tensor * k     = dst->src[1];
    ثابت ggml_tensor * v     = dst->src[2];
    ثابت ggml_tensor * نقاب  = dst->src[3];
    ثابت ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q, ne)
    GGML_TENSOR_LOCALS(size_t, nbq, q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k, ne)
    GGML_TENSOR_LOCALS(size_t, nbk, k, nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v, ne)
    GGML_TENSOR_LOCALS(size_t, nbv, v, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t, nb, dst, nb)

    ثابت int64_t DK = nek0;
    ثابت int64_t DV = nev0;
    ثابت int64_t N  = neq1;

    GGML_ASSERT(flash_attn_ext_supported_shape_vlen1024_vf16(DK, DV));

    // broadcast factors
    ثابت int64_t rk2 = neq2 / nek2;
    ثابت int64_t rk3 = neq3 / nek3;

    ثابت int64_t rv2 = neq2 / nev2;
    ثابت int64_t rv3 = neq3 / nev3;

    // parallelize by q rows using ggml_vec_dot_f32

    اعشاری scale         = *((اعشاری *) dst->op_params + 0);
    اعشاری max_bias      = *((اعشاری *) dst->op_params + 1);
    اعشاری logit_softcap = *((اعشاری *) dst->op_params + 2);

    اگر (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    ثابت uint32_t n_سر      = neq2;
    ثابت uint32_t n_سر_log2 = 1u << (uint32_t) floor(log2(n_سر));

    ثابت اعشاری m0 = powf(2.0f, -(max_bias) / n_سر_log2);
    ثابت اعشاری m1 = powf(2.0f, -(max_bias / 2.0f) / n_سر_log2);

    ثابت صحیح KV_صف_سائز = DK * حجم(_Float16) + DV * حجم(_Float16);

    صحیح ith     = params->ith;
    صحیح ir_قدم = 1;
    برائے (صحیح ir = ir0; ir < ir1; ir += ir_قدم) {
        // q indices
        ثابت صحیح iq3 = ir / (neq2 * neq1);
        ثابت صحیح iq2 = (ir - iq3 * neq2 * neq1) / neq1;
        ثابت صحیح iq1 = (ir - iq3 * neq2 * neq1 - iq2 * neq1);

        ثابت صحیح iq3_1 = (ir + 1) / (neq2 * neq1);
        ثابت صحیح iq2_1 = (ir + 1 - iq3_1 * neq2 * neq1) / neq1;
        ثابت صحیح iq1_1 = (ir + 1 - iq3_1 * neq2 * neq1 - iq2_1 * neq1);

        ثابت صحیح iq3_2 = (ir + 2) / (neq2 * neq1);
        ثابت صحیح iq2_2 = (ir + 2 - iq3_2 * neq2 * neq1) / neq1;
        ثابت صحیح iq1_2 = (ir + 2 - iq3_2 * neq2 * neq1 - iq2_2 * neq1);

        ثابت صحیح iq3_3 = (ir + 3) / (neq2 * neq1);
        ثابت صحیح iq2_3 = (ir + 3 - iq3_3 * neq2 * neq1) / neq1;
        ثابت صحیح iq1_3 = (ir + 3 - iq3_3 * neq2 * neq1 - iq2_3 * neq1);

        ثابت uint32_t h = iq2;  // head index
        ثابت اعشاری    ڈھلان =
            (max_bias > 0.0f) ? h < n_سر_log2 ? powf(m0, h + 1) : powf(m1, 2 * (h - n_سر_log2) + 1) : 1.0f;

        ثابت ggml_fp16_t * mp =
            نقاب ? (ggml_fp16_t *) ((حرف *) نقاب->ڈیٹا + iq1 * نقاب->nb[1] + (iq2 % نقاب->ne[2]) * نقاب->nb[2] +
                                    (iq3 % نقاب->ne[3]) * نقاب->nb[3]) :
                   تہی;

        ثابت منطقی mp_equal_2 = iq1_1 == iq1 && (iq2 % نقاب->ne[2]) == (iq2_1 % نقاب->ne[2]) &&
                                (iq3 % نقاب->ne[3]) == (iq3_1 % نقاب->ne[3]);

        ثابت منطقی mp_equal_4 = mp_equal_2 && iq1_2 == iq1 && (iq2 % نقاب->ne[2]) == (iq2_2 % نقاب->ne[2]) &&
                                (iq3 % نقاب->ne[3]) == (iq3_2 % نقاب->ne[3]) && iq1_3 == iq1 &&
                                (iq2 % نقاب->ne[2]) == (iq2_3 % نقاب->ne[2]) &&
                                (iq3 % نقاب->ne[3]) == (iq3_3 % نقاب->ne[3]);

        // k indices
        ثابت صحیح ik3 = iq3 / rk3;
        ثابت صحیح ik2 = iq2 / rk2;

        ثابت صحیح ik3_1 = iq3_1 / rk3;
        ثابت صحیح ik2_1 = iq2_1 / rk2;

        ثابت صحیح ik3_2 = iq3_2 / rk3;
        ثابت صحیح ik2_2 = iq2_2 / rk2;

        ثابت صحیح ik3_3 = iq3_3 / rk3;
        ثابت صحیح ik2_3 = iq2_3 / rk2;

        // v indices
        ثابت صحیح iv3 = iq3 / rv3;
        ثابت صحیح iv2 = iq2 / rv2;

        ثابت صحیح iv3_1 = iq3_1 / rv3;
        ثابت صحیح iv2_1 = iq2_1 / rv2;

        ثابت صحیح iv3_2 = iq3_2 / rv3;
        ثابت صحیح iv2_2 = iq2_2 / rv2;

        ثابت صحیح iv3_3 = iq3_3 / rv3;
        ثابت صحیح iv2_3 = iq2_3 / rv2;

        ثابت اعشاری * pq = (ثابت اعشاری *) ((حرف *) q->ڈیٹا + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3));

        std::array<اعشاری *, 4> pq_بفر;
        std::array<اعشاری *, 4> sinks_بفر;
        std::array<اعشاری *, 4> dst_بفر;

        اگر (tcm_بفر != خالی_اشاریہ && 4 * KV_صف_سائز < tcm_بفر_سائز && ir < (ir1 - 3) && mp_equal_4 &&
            ik3_3 == ik3 && ik2_3 == ik2 && iv3_3 == iv3 && iv2_3 == iv2 && ik3_2 == ik3 && ik2_2 == ik2 &&
            iv3_2 == iv3 && iv2_2 == iv2 && ik3_1 == ik3 && ik2_1 == ik2 && iv3_1 == iv3 && iv2_1 == iv2) {
            ir_قدم = 4;

            pq_بفر[0] = (اعشاری *) ((حرف *) q->ڈیٹا + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3));
            pq_بفر[1] = (اعشاری *) ((حرف *) q->ڈیٹا + (iq1_1 * nbq1 + iq2_1 * nbq2 + iq3_1 * nbq3));
            pq_بفر[2] = (اعشاری *) ((حرف *) q->ڈیٹا + (iq1_2 * nbq1 + iq2_2 * nbq2 + iq3_2 * nbq3));
            pq_بفر[3] = (اعشاری *) ((حرف *) q->ڈیٹا + (iq1_3 * nbq1 + iq2_3 * nbq2 + iq3_3 * nbq3));

            sinks_بفر[0] = sinks ? ((اعشاری *) ((حرف *) sinks->ڈیٹا)) + iq2 : خالی_اشاریہ;
            sinks_بفر[1] = sinks ? ((اعشاری *) ((حرف *) sinks->ڈیٹا)) + iq2_1 : خالی_اشاریہ;
            sinks_بفر[2] = sinks ? ((اعشاری *) ((حرف *) sinks->ڈیٹا)) + iq2_2 : خالی_اشاریہ;
            sinks_بفر[3] = sinks ? ((اعشاری *) ((حرف *) sinks->ڈیٹا)) + iq2_3 : خالی_اشاریہ;

            dst_بفر[0] = (اعشاری *) ((حرف *) dst->ڈیٹا + (iq3 * ne2 * ne1 + iq2 + iq1 * ne1) * nb1);
            dst_بفر[1] = (اعشاری *) ((حرف *) dst->ڈیٹا + (iq3_1 * ne2 * ne1 + iq2_1 + iq1_1 * ne1) * nb1);
            dst_بفر[2] = (اعشاری *) ((حرف *) dst->ڈیٹا + (iq3_2 * ne2 * ne1 + iq2_2 + iq1_2 * ne1) * nb1);
            dst_بفر[3] = (اعشاری *) ((حرف *) dst->ڈیٹا + (iq3_3 * ne2 * ne1 + iq2_3 + iq1_3 * ne1) * nb1);

            flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_mrow<4>(  //
                pq_بفر.ڈیٹا(),                                      //
                (ثابت حرف *) k->ڈیٹا + (ik2 * nbk2 + ik3 * nbk3),    //
                (ثابت حرف *) v->ڈیٹا + (iv2 * nbv2 + iv3 * nbv3),    //
                mp,                                                    //
                sinks_بفر.ڈیٹا(),                                   //
                dst_بفر.ڈیٹا(),                                     //
                scale, logit_softcap, ڈھلان, nek1, nbk1, nbv1, DV, DK, tcm_بفر, tcm_بفر_سائز);
        } ورنہ اگر (tcm_بفر != خالی_اشاریہ && 2 * KV_صف_سائز < tcm_بفر_سائز && ir < (ir1 - 1) && mp_equal_2 &&
                   ik3_1 == ik3 && ik2_1 == ik2 && iv3_1 == iv3 && iv2_1 == iv2) {
            ir_قدم = 2;

            pq_بفر[0] = (اعشاری *) ((حرف *) q->ڈیٹا + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3));
            pq_بفر[1] = (اعشاری *) ((حرف *) q->ڈیٹا + (iq1_1 * nbq1 + iq2_1 * nbq2 + iq3_1 * nbq3));

            sinks_بفر[0] = sinks ? ((اعشاری *) ((حرف *) sinks->ڈیٹا)) + iq2 : خالی_اشاریہ;
            sinks_بفر[1] = sinks ? ((اعشاری *) ((حرف *) sinks->ڈیٹا)) + iq2_1 : خالی_اشاریہ;

            dst_بفر[0] = (اعشاری *) ((حرف *) dst->ڈیٹا + (iq3 * ne2 * ne1 + iq2 + iq1 * ne1) * nb1);
            dst_بفر[1] = (اعشاری *) ((حرف *) dst->ڈیٹا + (iq3_1 * ne2 * ne1 + iq2_1 + iq1_1 * ne1) * nb1);

            flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_mrow<2>(  //
                pq_بفر.ڈیٹا(),                                      //
                (ثابت حرف *) k->ڈیٹا + (ik2 * nbk2 + ik3 * nbk3),    //
                (ثابت حرف *) v->ڈیٹا + (iv2 * nbv2 + iv3 * nbv3),    //
                mp,                                                    //
                sinks_بفر.ڈیٹا(),                                   //
                dst_بفر.ڈیٹا(),                                     //
                scale, logit_softcap, ڈھلان, nek1, nbk1, nbv1, DV, DK, tcm_بفر, tcm_بفر_سائز);
        } ورنہ {
            ir_قدم = 1;
            flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_m1(                             //
                pq,                                                                          //
                (ثابت حرف *) k->ڈیٹا + (ik2 * nbk2 + ik3 * nbk3),                          //
                (ثابت حرف *) v->ڈیٹا + (iv2 * nbv2 + iv3 * nbv3),                          //
                mp,                                                                          //
                sinks ? ((اعشاری *) ((حرف *) sinks->ڈیٹا)) + h : خالی_اشاریہ,                    //
                (اعشاری *) ((حرف *) dst->ڈیٹا + (iq3 * ne2 * ne1 + iq2 + iq1 * ne1) * nb1),  //
                scale, logit_softcap, ڈھلان, nek1, nbk1, nbv1, DV, DK);
        }
    }
}

خالی forward_flash_attn_ext_f16_tiled_vlen1024_vf16(ثابت ggml_compute_params * params,
                                                    ggml_tensor *               dst,
                                                    صحیح                         ir0,
                                                    صحیح                         ir1,
                                                    خالی *                      tcm_بفر,
                                                    size_t                      tcm_بفر_سائز) {
    ثابت ggml_tensor * q     = dst->src[0];
    ثابت ggml_tensor * k     = dst->src[1];
    ثابت ggml_tensor * v     = dst->src[2];
    ثابت ggml_tensor * نقاب  = dst->src[3];
    ثابت ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q, ne)
    GGML_TENSOR_LOCALS(size_t, nbq, q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k, ne)
    GGML_TENSOR_LOCALS(size_t, nbk, k, nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v, ne)
    GGML_TENSOR_LOCALS(size_t, nbv, v, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t, nb, dst, nb)

    ثابت int64_t DK = nek0;
    ثابت int64_t DV = nev0;
    ثابت int64_t N  = neq1;

    GGML_ASSERT(flash_attn_ext_supported_shape_vlen1024_vf16(DK, DV));

    GGML_ASSERT(ne0 == DV);
    GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_سائز(q->type));
    GGML_ASSERT(nbk0 == ggml_type_سائز(k->type));
    GGML_ASSERT(nbv0 == ggml_type_سائز(v->type));

    GGML_ASSERT(neq0 == DK);
    GGML_ASSERT(nek0 == DK);
    GGML_ASSERT(nev0 == DV);

    GGML_ASSERT(neq1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == حجم(اعشاری));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(k->type == v->type);
    ثابت ggml_type kv_type = k->type;

    // broadcast factors
    ثابت int64_t rk2 = neq2 / nek2;
    ثابت int64_t rk3 = neq3 / nek3;

    ثابت int64_t rv2 = neq2 / nev2;
    ثابت int64_t rv3 = neq3 / nev3;

    اعشاری * param_list    = (اعشاری *) dst->op_params;
    اعشاری   scale         = param_list[0];
    اعشاری   max_bias      = param_list[1];
    اعشاری   logit_softcap = param_list[2];

    اگر (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    ثابت uint32_t n_سر      = neq2;
    ثابت uint32_t n_سر_log2 = 1u << (uint32_t) floor(log2(n_سر));

    ثابت اعشاری m0 = powf(2.0f, -(max_bias) / n_سر_log2);
    ثابت اعشاری m1 = powf(2.0f, -(max_bias / 2.0f) / n_سر_log2);

    صحیح ith = params->ith;

    ساکن constexpr صحیح Q_TILE_SZ  = ggml_fa_tile_config::Q;
    ساکن constexpr صحیح KV_TILE_SZ = ggml_fa_tile_config::KV;

    // Per-thread scratch layout:
    // Q_f32:   Q_TILE_SZ * DK
    // KQ:      Q_TILE_SZ * KV_TILE_SZ
    // mask32:  Q_TILE_SZ * KV_TILE_SZ
    // VKQ32:   Q_TILE_SZ * DV
    // V32:     KV_TILE_SZ * DV
    // K_f32:   DK * KV_TILE_SZ (transposed K tile)
    اعشاری *      بنیاد = (اعشاری *) params->wdata + ith * (Q_TILE_SZ * DK + 2 * Q_TILE_SZ * KV_TILE_SZ + Q_TILE_SZ * DV +
                                                    KV_TILE_SZ * DV + KV_TILE_SZ * DK + CACHE_سطر_سائز_F32);
    ثابت size_t بنیاد_سائز =
        (Q_TILE_SZ * DK + 2 * Q_TILE_SZ * KV_TILE_SZ + Q_TILE_SZ * DV + KV_TILE_SZ * DV + KV_TILE_SZ * DK) *
            حجم(اعشاری) +
        CACHE_سطر_سائز_F32;

    اگر (بنیاد_سائز <= tcm_بفر_سائز && tcm_بفر != خالی_اشاریہ) {
        بنیاد = (اعشاری *) tcm_بفر;
    }

    اعشاری   S_M_Buf[Q_TILE_SZ * 2];  // buffer to hold S, M, bias for one tile to reduce register pressure in main loop
    اعشاری * S = S_M_Buf;
    اعشاری * M = S_M_Buf + Q_TILE_SZ;

    صحیح ir = ir0;
    جب (ir < ir1) {
        // q indices for the start of this tile
        ثابت صحیح iq3 = ir / (neq2 * neq1);
        ثابت صحیح iq2 = (ir - iq3 * neq2 * neq1) / neq1;
        ثابت صحیح iq1 = (ir - iq3 * neq2 * neq1 - iq2 * neq1);

        // Number of valid rows in this tile:
        // - limited by tile size (Q_TILE_SZ)
        // - limited by chunk boundary (ir1 - ir)
        // - limited by head boundary (neq1 - iq1) to avoid crossing into next head
        ثابت صحیح tile_صفیں = MIN(Q_TILE_SZ, MIN((صحیح) (ir1 - ir), (صحیح) (neq1 - iq1)));
        GGML_ASSERT(tile_صفیں > 0);

        ثابت uint32_t h = iq2;  // head index
        ثابت اعشاری    ڈھلان =
            (max_bias > 0.0f) ? h < n_سر_log2 ? powf(m0, h + 1) : powf(m1, 2 * (h - n_سر_log2) + 1) : 1.0f;

        برائے (صحیح i = 0; i < Q_TILE_SZ; ++i) {
            S[i] = 0.;
            M[i] = -INFINITY;
        }

        اعشاری *    Q_f32  = بنیاد;
        اعشاری *    KQ     = (اعشاری *) ((حرف *) بنیاد + Q_TILE_SZ * DK * حجم(اعشاری));
        اعشاری *    mask32 = KQ + Q_TILE_SZ * KV_TILE_SZ;
        اعشاری *    VKQ32  = mask32 + Q_TILE_SZ * KV_TILE_SZ;
        اعشاری *    V32    = VKQ32 + Q_TILE_SZ * DV;
        اعشاری *    K_f32  = V32 + KV_TILE_SZ * DV;
        _Float16 * Q_f16  = (_Float16 *) Q_f32;
        _Float16 * V_f16  = (_Float16 *) V32;
        _Float16 * K_f16  = (_Float16 *) K_f32;

        rvv_zero_f32(VKQ32, Q_TILE_SZ * DV);

        // k indices
        ثابت صحیح ik3 = iq3 / rk3;
        ثابت صحیح ik2 = iq2 / rk2;

        // v indices
        ثابت صحیح iv3 = iq3 / rv3;
        ثابت صحیح iv2 = iq2 / rv2;

        ثابت اعشاری * pq = (ثابت اعشاری *) ((حرف *) q->ڈیٹا + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3));
        اگر (kv_type == GGML_TYPE_F16) {
            rvv_pack_f32_as_scaled_f16((uint8_t *) Q_f16, DK * حجم(_Float16), (uint8_t *) pq, nbq1, tile_صفیں, DK,
                                       scale);
        } ورنہ {
            memcpy2d(Q_f32, DK * حجم(اعشاری), pq, nbq1, tile_صفیں, DK * حجم(اعشاری));
        }

        برائے (int64_t ic = 0; ic < nek1; ic += KV_TILE_SZ) {
            ثابت صحیح kv_tile = (صحیح) std::min((int64_t) KV_TILE_SZ, nek1 - ic);

            rvv_zero_f32(K_f32, DK * KV_TILE_SZ);
            rvv_zero_f32(V32, KV_TILE_SZ * DV);

            // skip the tile entirely if all the masks are -inf
            اگر (نقاب) {
                منطقی                can_skip = سچ;
                ثابت ggml_fp16_t * mp_صف =
                    (ثابت ggml_fp16_t *) ((ثابت حرف *) نقاب->ڈیٹا + iq1 * نقاب->nb[1] +
                                           (iq2 % نقاب->ne[2]) * نقاب->nb[2] + (iq3 % نقاب->ne[3]) * نقاب->nb[3]);
                rvv_pack_scaled_f16_as_f32(mask32, KV_TILE_SZ * حجم(اعشاری), mp_صف + ic, نقاب->nb[1], tile_صفیں,
                                           kv_tile, ڈھلان);

                برائے (صحیح tq = 0; tq < tile_صفیں; tq++) {
                    برائے (صحیح tk = 0; tk < kv_tile; tk++) {
                        اگر (mask32[tq * KV_TILE_SZ + tk] != -INFINITY) {
                            can_skip = جھوٹ;
                        }
                    }
                    // Pad remaining mask entries with -inf
                    برائے (صحیح tk = kv_tile; tk < KV_TILE_SZ; tk++) {
                        mask32[tq * KV_TILE_SZ + tk] = -INFINITY;
                    }
                }

                اگر (can_skip) {
                    جاری;
                }
            }

            اگر (kv_type == GGML_TYPE_F16) {
                rvv_transposed_s16_mn_مقصد_nm((int8_t *) K_f16, KV_TILE_SZ * حجم(_Float16),
                                            (int8_t *) k->ڈیٹا + ic * nbk1 + ik2 * nbk2 + ik3 * nbk3, nbk1, kv_tile,
                                            DK);

                صحیح tq = 0;
                برائے (; tq + 3 < tile_صفیں; tq += 4) {
                    rvv_qk_dot_tile_f16_x4(KQ + (tq + 0) * KV_TILE_SZ, KQ + (tq + 1) * KV_TILE_SZ,
                                           KQ + (tq + 2) * KV_TILE_SZ, KQ + (tq + 3) * KV_TILE_SZ,
                                           Q_f16 + (tq + 0) * DK, Q_f16 + (tq + 1) * DK, Q_f16 + (tq + 2) * DK,
                                           Q_f16 + (tq + 3) * DK, K_f16, DK, kv_tile);
                }
                برائے (; tq < tile_صفیں; ++tq) {
                    rvv_qk_dot_tile_f16_x1(KQ + tq * KV_TILE_SZ, Q_f16 + tq * DK, K_f16, DK, kv_tile);
                }
            } ورنہ {
                برائے (صحیح tk = 0; tk < kv_tile; tk++) {
                    ثابت حرف *  k_ڈیٹا = (ثابت حرف *) k->ڈیٹا + (ic + tk) * nbk1 + ik2 * nbk2 + ik3 * nbk3;
                    اعشاری *       k_ستون  = K_f32 + tk;
                    ثابت اعشاری * k_src  = (ثابت اعشاری *) k_ڈیٹا;
                    برائے (int64_t dk = 0; dk < DK; ++dk) {
                        k_ستون[dk * KV_TILE_SZ] = k_src[dk];
                    }
                }

                برائے (صحیح tq = 0; tq < tile_صفیں; ++tq) {
                    rvv_qk_dot_tile(KQ + tq * KV_TILE_SZ, Q_f32 + tq * DK, K_f32, DK, KV_TILE_SZ, scale);
                }
            }

            // Set padded KQ entries to -inf so softmax gives them zero weight
            اگر (kv_tile < KV_TILE_SZ) {
                برائے (صحیح tq = 0; tq < tile_صفیں; tq++) {
                    برائے (صحیح tk = kv_tile; tk < KV_TILE_SZ; tk++) {
                        KQ[tq * KV_TILE_SZ + tk] = -INFINITY;
                    }
                }
            }

            اگر (logit_softcap != 0.0f) {
                rvv_softcap_tanh_inplace_f32(KQ, KV_TILE_SZ, tile_صفیں, KV_TILE_SZ, logit_softcap);
            }

            اگر (نقاب) {
                rvv_جوڑو_inplace_f32(KQ, KV_TILE_SZ, mask32, KV_TILE_SZ, tile_صفیں, KV_TILE_SZ);
            }

            منطقی skip[Q_TILE_SZ] = {};

            برائے (صحیح tq = 0; tq < tile_صفیں; tq++) {
                اعشاری * kq_صف = KQ + tq * KV_TILE_SZ;

                ثابت اعشاری tile_max = rvv_max_f32(kq_صف, KV_TILE_SZ);

                اگر (tile_max == -INFINITY) {
                    skip[tq] = سچ;
                    جاری;
                }

                ثابت اعشاری Mold = M[tq];
                ثابت اعشاری Mnew = fmaxf(Mold, tile_max);

                اگر (Mnew > Mold) {
                    ثابت اعشاری ms = expf(Mold - Mnew);
                    rvv_scale_f32(VKQ32 + tq * DV, ms, DV);
                    S[tq] *= ms;
                }
                M[tq] = Mnew;

                S[tq] += rvv_softmax_exp_inplace_f32(kq_صف, KV_TILE_SZ, Mnew);
            }

            // Pack V as contiguous [KV_TILE_SZ][DV].
            اگر (kv_type == GGML_TYPE_F16) {
                ثابت حرف * v_ڈیٹا = (ثابت حرف *) v->ڈیٹا + ic * nbv1 + iv2 * nbv2 + iv3 * nbv3;
                memcpy2d(V_f16, DV * حجم(_Float16), v_ڈیٹا, nbv1, kv_tile, DV * حجم(_Float16));

                صحیح tq = 0;
                برائے (; tq + 3 < tile_صفیں; tq += 4) {
                    اگر (skip[tq + 0] || skip[tq + 1] || skip[tq + 2] || skip[tq + 3]) {
                        برائے (صحیح i = 0; i < 4; ++i) {
                            اگر (!skip[tq + i]) {
                                rvv_pv_accumulate_f16_x1(VKQ32 + (tq + i) * DV, KQ + (tq + i) * KV_TILE_SZ, V_f16,
                                                         KV_TILE_SZ, DV);
                            }
                        }
                        جاری;
                    }

                    rvv_pv_accumulate_f16_x4(VKQ32 + (tq + 0) * DV, VKQ32 + (tq + 1) * DV, VKQ32 + (tq + 2) * DV,
                                             VKQ32 + (tq + 3) * DV, KQ + (tq + 0) * KV_TILE_SZ,
                                             KQ + (tq + 1) * KV_TILE_SZ, KQ + (tq + 2) * KV_TILE_SZ,
                                             KQ + (tq + 3) * KV_TILE_SZ, V_f16, KV_TILE_SZ, DV);
                }
                برائے (; tq < tile_صفیں; ++tq) {
                    اگر (!skip[tq]) {
                        rvv_pv_accumulate_f16_x1(VKQ32 + tq * DV, KQ + tq * KV_TILE_SZ, V_f16, KV_TILE_SZ, DV);
                    }
                }
            } ورنہ {
                ثابت حرف * v_ڈیٹا = (ثابت حرف *) v->ڈیٹا + ic * nbv1 + iv2 * nbv2 + iv3 * nbv3;
                memcpy2d(V32, DV * حجم(اعشاری), v_ڈیٹا, nbv1, kv_tile, DV * حجم(اعشاری));

                برائے (صحیح tq = 0; tq < tile_صفیں; ++tq) {
                    اگر (!skip[tq]) {
                        rvv_pv_accumulate(VKQ32 + tq * DV, KQ + tq * KV_TILE_SZ, V32, KV_TILE_SZ, DV);
                    }
                }
            }
        }

        // sinks (apply only to valid rows in the tile)
        اگر (sinks) {
            ثابت اعشاری s = ((اعشاری *) ((حرف *) sinks->ڈیٹا))[h];

            برائے (صحیح tq = 0; tq < tile_صفیں; tq++) {
                اعشاری ms = 1.0f;
                اعشاری vs = 1.0f;

                اگر (s > M[tq]) {
                    ms = expf(M[tq] - s);
                    rvv_scale_f32(VKQ32 + tq * DV, ms, DV);
                } ورنہ {
                    vs = expf(s - M[tq]);
                }

                اعشاری S_عارضی = S[tq] * ms + vs;
                S[tq]        = S_عارضی == 0.0f ? 0.0f : 1.0f / S_عارضی;
            }
        } ورنہ {
            برائے (صحیح tq = 0; tq < tile_صفیں; tq++) {
                ثابت اعشاری S_inv = S[tq] == 0.0f ? 0.0f : 1.0f / S[tq];
                S[tq]             = S_inv;
            }
        }

        اعشاری * dst_ptr = (اعشاری *) ((حرف *) dst->ڈیٹا + (iq3 * ne2 * ne1 + iq2 + (iq1) *ne1) * nb1);
        rvv_pack_scaled_f32_as_f32(dst_ptr, nb1 * ne1, VKQ32, DV * حجم(اعشاری), tile_صفیں, DV, S);

        ir += tile_صفیں;
    }
}

خالی forward_rms_norm_f32(ggml_compute_params * params, ggml_tensor * op) {
    ثابت ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(src0->nb[0] == حجم(اعشاری));

    صحیح ith = params->ith;
    صحیح nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    اعشاری epsilon = *((اعشاری *) dst->op_params);

    GGML_ASSERT(epsilon > 0.0f);

    خودکار * داخلہ  = (حرف *) src0->ڈیٹا;
    خودکار * output = (حرف *) dst->ڈیٹا;

    ثابت خودکار hidden_سائز     = ne00;
    ثابت خودکار task_گنتی      = ne01 * ne02 * ne03;
    ثابت خودکار task_فی_thread = (task_گنتی + nth - 1) / nth;

    ثابت خودکار task_begin = ith * task_فی_thread;
    ثابت خودکار task_انتہا   = std::min((ith + 1) * task_فی_thread, task_گنتی);

    برائے (خودکار task_idx = task_begin; task_idx < task_انتہا; task_idx++) {
        int64_t i03 = task_idx / (ne02 * ne01);
        int64_t i02 = (task_idx - i03 * ne02 * ne01) / ne01;
        int64_t i01 = (task_idx - i03 * ne02 * ne01 - i02 * ne01);

        خودکار * p_داخلہ       = (اعشاری *) (داخلہ + i01 * nb01 + i02 * nb02 + i03 * nb03);
        خودکار * p_output      = (اعشاری *) (output + i01 * nb1 + i02 * nb2 + i03 * nb3);
        خودکار * p_عارضی_output = p_output;

        size_t       gvl    = __riscv_vsetvlmax_e32m4();
        vfloat32m4_t مجموع_sq = __riscv_vfmv_v_f_f32m4(0.f, gvl);
        int64_t      لمبائی = hidden_سائز;
        جب (لمبائی > 0) {
            gvl                   = __riscv_vsetvl_e32m4(لمبائی);
            vfloat32m4_t src_ڈیٹا = __riscv_vle32_v_f32m4(p_داخلہ, gvl);
            مجموع_sq                = __riscv_vfmacc_vv_f32m4(مجموع_sq, src_ڈیٹا, src_ڈیٹا, gvl);
            __riscv_vse32_v_f32m4(p_عارضی_output, src_ڈیٹا, gvl);

            p_داخلہ += gvl;
            p_عارضی_output += gvl;
            لمبائی -= gvl;
        }

        gvl                 = __riscv_vsetvlmax_e32m1();
        vfloat32m1_t zero_v = __riscv_vfmv_v_f_f32m1(0.f, gvl);
        vfloat32m1_t اوسط_مربع_v =
            __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(مجموع_sq, 0), __riscv_vget_v_f32m4_f32m1(مجموع_sq, 1), gvl);

        اوسط_مربع_v = __riscv_vfadd_vv_f32m1(اوسط_مربع_v, __riscv_vget_v_f32m4_f32m1(مجموع_sq, 2), gvl);
        اوسط_مربع_v = __riscv_vfadd_vv_f32m1(اوسط_مربع_v, __riscv_vget_v_f32m4_f32m1(مجموع_sq, 3), gvl);
        اوسط_مربع_v = __riscv_vfredusum_vs_f32m1_f32m1(اوسط_مربع_v, zero_v, gvl);

        اعشاری اوسط_مربع = __riscv_vfmv_f_s_f32m1_f32(اوسط_مربع_v);
        اوسط_مربع /= hidden_سائز;

        اوسط_مربع = جذر(اوسط_مربع + epsilon);

        اوسط_مربع   = 1.0f / اوسط_مربع;
        لمبائی        = hidden_سائز;
        p_عارضی_output = p_output;

        جب (لمبائی > 0) {
            gvl                   = __riscv_vsetvl_e32m4(لمبائی);
            vfloat32m4_t src_ڈیٹا = __riscv_vle32_v_f32m4(p_عارضی_output, gvl);
            src_ڈیٹا              = __riscv_vfmul_vf_f32m4(src_ڈیٹا, اوسط_مربع, gvl);
            __riscv_vse32_v_f32m4(p_output, src_ڈیٹا, gvl);
            p_عارضی_output += gvl;
            p_output += gvl;
            لمبائی -= gvl;
        }
    }
}

سانچہ <size_t MB_صفیں>
خالی quantize_a_nrow_i8_ref(size_t blk_len, ثابت اعشاری * a_ptr, size_t گنتی_k, uint8_t * quant_a_ptr) {
    int64_t a_blk_stride        = q8_blk_سائز(blk_len, سچ);
    int64_t a_nrow_بلاک_stride = a_blk_stride * MB_صفیں;
    برائے (size_t k = 0; k < گنتی_k; k += blk_len, quant_a_ptr += a_nrow_بلاک_stride) {
        اعشاری *   scale_a_ptr = reinterpret_cast<اعشاری *>(quant_a_ptr);
        int16_t * a_مجموع_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + حجم(اعشاری) * MB_صفیں);
        int8_t *  quant_a_blk =
            reinterpret_cast<int8_t *>(quant_a_ptr + حجم(اعشاری) * MB_صفیں + حجم(int16_t) * MB_صفیں);

        برائے (size_t صف = 0; صف < MB_صفیں; صف++) {
            اعشاری max_abs_a = 0.0f;
            برائے (size_t bk = 0; bk < blk_len; bk++) {
                max_abs_a = std::max(max_abs_a, std::مطلق_قدر(a_ptr[صف * گنتی_k + k + bk]));
            }

            اعشاری rep_scale_a = ((1 << 7) - 1) / max_abs_a;
            scale_a_ptr[صف]  = 1 / rep_scale_a;

            int16_t a_مجموع = 0;
            برائے (size_t bk = 0; bk < blk_len; bk++) {
                ثابت int8_t quantized = static_cast<int8_t>(
                    std::clamp(std::nearbyintf(a_ptr[صف * گنتی_k + k + bk] * rep_scale_a), -128.0f, 127.0f));
                quant_a_blk[صف * blk_len + bk] = quantized;
                a_مجموع += quantized;
            }
            a_مجموع_ptr[صف] = -a_مجموع;
        }
    }
}

سانچہ <size_t MB_صفیں>
خالی quantize_a_nrow_i8_hp_ref(size_t blk_len, ثابت اعشاری * a_ptr, size_t گنتی_k, uint8_t * quant_a_ptr) {
    constexpr size_t k_subblk_len = 32;
    ثابت size_t     subblk_گنتی = blk_len / k_subblk_len;

    GGML_ASSERT(blk_len == 256);

    اعشاری   scale_عارضی[8]       = { 0.0f };
    int64_t a_blk_stride        = q8_hp_blk_سائز(blk_len, سچ, سچ);
    int64_t a_nrow_بلاک_stride = a_blk_stride * MB_صفیں;
    int64_t a_subblk_stride     = q8_hp_blk_سائز(k_subblk_len, جھوٹ, جھوٹ) * MB_صفیں;

    برائے (size_t k = 0; k < گنتی_k; k += blk_len, quant_a_ptr += a_nrow_بلاک_stride) {
        _Float16 * a_مجموع_ptr = reinterpret_cast<_Float16 *>(quant_a_ptr + a_subblk_stride * subblk_گنتی);

        اعشاری scale_avg = 0.0f;
        برائے (size_t kk = 0; kk < subblk_گنتی; kk++) {
            اعشاری max_abs_a = 0.0f;
            برائے (size_t صف = 0; صف < MB_صفیں; صف++) {
                برائے (size_t bk = 0; bk < k_subblk_len; bk++) {
                    max_abs_a = std::max(max_abs_a, std::مطلق_قدر(a_ptr[صف * گنتی_k + k + bk + kk * k_subblk_len]));
                }
            }
            scale_عارضی[kk] = max_abs_a / ((1 << 7) - 1);
            scale_avg += scale_عارضی[kk];
        }

        scale_avg /= subblk_گنتی;
        اعشاری scale_ضربی_عدد = 1.0f / scale_avg;

        _Float16 * scale_avg_ptr =
            reinterpret_cast<_Float16 *>(quant_a_ptr + a_nrow_بلاک_stride - حجم(_Float16) * MB_صفیں);
        scale_avg_ptr[0] = scale_avg;

        برائے (size_t kk = 0; kk < subblk_گنتی; kk++) {
            uint8_t *  a_subblk_بنیاد = quant_a_ptr + kk * a_subblk_stride;
            _Float16 * scale_a_ptr   = reinterpret_cast<_Float16 *>(a_subblk_بنیاد);
            int8_t *   quant_a_blk   = reinterpret_cast<int8_t *>(a_subblk_بنیاد + حجم(_Float16) * MB_صفیں);

            scale_a_ptr[0] = static_cast<_Float16>(scale_عارضی[kk] * scale_ضربی_عدد);

            ثابت اعشاری rep_scale_a = 1.0f / scale_عارضی[kk];

            برائے (size_t صف = 0; صف < MB_صفیں; صف++) {
                int16_t a_مجموع = 0;
                برائے (size_t bk = 0; bk < k_subblk_len; bk++) {
                    ثابت int8_t quantized = static_cast<int8_t>(
                        std::clamp(std::nearbyintf(a_ptr[صف * گنتی_k + k + bk + kk * k_subblk_len] * rep_scale_a),
                                   -128.0f, 127.0f));
                    quant_a_blk[صف * k_subblk_len + bk] = quantized;
                    a_مجموع += quantized;
                }
                a_مجموع_ptr[صف * subblk_گنتی + kk] = static_cast<_Float16>(-a_مجموع) * static_cast<_Float16>(8.0f);
            }
        }
    }
}

سانچہ <size_t MB_صفیں>
خالی quantize_a_nrow_i8k_ref(size_t blk_len, ثابت اعشاری * a_ptr, size_t گنتی_k, uint8_t * quant_a_ptr) {
    int64_t a_blk_stride        = q8k_blk_سائز(256);
    int64_t a_nrow_بلاک_stride = a_blk_stride * MB_صفیں;
    int64_t a_مجموع_سائز          = 256 / 16;

    برائے (size_t k = 0; k < گنتی_k; k += blk_len, quant_a_ptr += a_nrow_بلاک_stride) {
        اعشاری *   scale_a_ptr = reinterpret_cast<اعشاری *>(quant_a_ptr);
        int16_t * a_مجموع_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + حجم(اعشاری) * MB_صفیں);
        int8_t *  quant_a_blk =
            reinterpret_cast<int8_t *>(quant_a_ptr + حجم(اعشاری) * MB_صفیں + حجم(int16_t) * a_مجموع_سائز * MB_صفیں);

        برائے (size_t صف = 0; صف < MB_صفیں; صف++) {
            اعشاری max_a     = 0.0f;
            اعشاری max_abs_a = 0.0f;
            برائے (size_t bk = 0; bk < blk_len; bk++) {
                اعشاری ax = std::مطلق_قدر(a_ptr[صف * گنتی_k + k + bk]);
                اگر (ax > max_abs_a) {
                    max_abs_a = ax;
                    max_a     = a_ptr[صف * گنتی_k + k + bk];
                }
            }

            اگر (!max_abs_a) {
                scale_a_ptr[صف] = 0;
                برائے (size_t bki = 0; bki < a_مجموع_سائز; bki++) {
                    برائے (size_t bk = bki * 16; bk < (bki + 1) * 16; bk++) {
                        quant_a_blk[صف * blk_len + bk] = 0;
                    }
                    a_مجموع_ptr[صف * a_مجموع_سائز + bki] = 0;
                }
                جاری;
            }

            اعشاری rep_scale_a = ((1 << 7) - 1) / max_abs_a;
            scale_a_ptr[صف]  = 1 / rep_scale_a;

            برائے (size_t bki = 0; bki < a_مجموع_سائز; bki++) {
                int16_t a_مجموع = 0;
                برائے (size_t bk = bki * 16; bk < (bki + 1) * 16; bk++) {
                    ثابت int8_t quantized = static_cast<int8_t>(
                        std::clamp(std::nearbyintf(a_ptr[صف * گنتی_k + k + bk] * rep_scale_a), -128.0f, 127.0f));
                    quant_a_blk[صف * blk_len + bk] = quantized;
                    a_مجموع += quantized;
                }
                a_مجموع_ptr[صف * a_مجموع_سائز + bki] = -a_مجموع;
            }
        }
    }
}

خالی quantize_a_صف_i8(size_t blk_len, ثابت اعشاری * a_ptr, size_t گنتی_k, uint8_t * quant_a_ptr) {
    GGML_ASSERT(blk_len == 32);
    int64_t a_blk_stride = q8_blk_سائز(blk_len, سچ);
    size_t  vlenb        = __riscv_vlenb();

    اگر (vlenb == 128) {
        برائے (size_t k = 0; k < گنتی_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            اعشاری *   scale_a_ptr = reinterpret_cast<اعشاری *>(quant_a_ptr);
            int16_t * a_مجموع_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + حجم(اعشاری));
            int8_t *  quant_a_blk = reinterpret_cast<int8_t *>(quant_a_ptr + حجم(اعشاری) + حجم(int16_t));

            size_t       vl      = __riscv_vsetvl_e32m1(blk_len);
            vfloat32m1_t v_a     = __riscv_vle32_v_f32m1(a_ptr + k, vl);
            vfloat32m1_t v_a_abs = __riscv_vfabs_v_f32m1(v_a, vl);

            vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m1_f32m1(v_a_abs, tmp, vl);
            اعشاری        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

            اعشاری scale_a     = max_abs_a / ((1 << 7) - 1);
            اعشاری rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
            scale_a_ptr[0]    = scale_a;

            vfloat32m1_t v_a_scale    = __riscv_vfmul_vf_f32m1(v_a, rep_scale_a, vl);
            vint16mf2_t  v_a_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a_scale, vl);
            vint8mf4_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a_quant, vl);

            vint16m1_t tmp_مجموع = __riscv_vmv_v_x_i16m1(0, vl);
            vint16m1_t v_a_مجموع = __riscv_vwredsum_vs_i8mf4_i16m1(v_a_quant_i8, tmp_مجموع, vl);
            int16_t    a_مجموع   = __riscv_vmv_x_s_i16m1_i16(v_a_مجموع);
            a_مجموع_ptr[0]       = -a_مجموع;

            __riscv_vse8_v_i8mf4(quant_a_blk, v_a_quant_i8, vl);
        }
    } ورنہ اگر (vlenb == 32) {
        برائے (size_t k = 0; k < گنتی_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            اعشاری *   scale_a_ptr = reinterpret_cast<اعشاری *>(quant_a_ptr);
            int16_t * a_مجموع_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + حجم(اعشاری));
            int8_t *  quant_a_blk = reinterpret_cast<int8_t *>(quant_a_ptr + حجم(اعشاری) + حجم(int16_t));

            size_t       vl      = __riscv_vsetvl_e32m4(blk_len);
            vfloat32m4_t v_a     = __riscv_vle32_v_f32m4(a_ptr + k, vl);
            vfloat32m4_t v_a_abs = __riscv_vfabs_v_f32m4(v_a, vl);

            vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m4_f32m1(v_a_abs, tmp, vl);
            اعشاری        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

            اعشاری scale_a     = max_abs_a / ((1 << 7) - 1);
            اعشاری rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
            scale_a_ptr[0]    = scale_a;

            vfloat32m4_t v_a_scale    = __riscv_vfmul_vf_f32m4(v_a, rep_scale_a, vl);
            vint16m2_t   v_a_quant    = __riscv_vfncvt_x_f_w_i16m2(v_a_scale, vl);
            vint8m1_t    v_a_quant_i8 = __riscv_vncvt_x_x_w_i8m1(v_a_quant, vl);

            vint16m1_t tmp_مجموع = __riscv_vmv_v_x_i16m1(0, vl);
            vint16m1_t v_a_مجموع = __riscv_vwredsum_vs_i8m1_i16m1(v_a_quant_i8, tmp_مجموع, vl);
            int16_t    a_مجموع   = __riscv_vmv_x_s_i16m1_i16(v_a_مجموع);
            a_مجموع_ptr[0]       = -a_مجموع;

            __riscv_vse8_v_i8m1(quant_a_blk, v_a_quant_i8, vl);
        }
    } ورنہ {
        quantize_a_nrow_i8_ref<1>(blk_len, a_ptr, گنتی_k, quant_a_ptr);
    }
}

خالی quantize_a_4row_i8(size_t blk_len, ثابت اعشاری * a_ptr, size_t گنتی_k, uint8_t * quant_a_ptr) {
    GGML_ASSERT(blk_len == 32);
    int64_t a_blk_stride        = q8_blk_سائز(blk_len, سچ);
    int64_t a_nrow_بلاک_stride = a_blk_stride * 4;
    size_t  vlenb               = __riscv_vlenb();

    اگر (vlenb == 128) {
        برائے (size_t k = 0; k < گنتی_k; k += blk_len, quant_a_ptr += a_nrow_بلاک_stride) {
            اعشاری *   scale_a_ptr = reinterpret_cast<اعشاری *>(quant_a_ptr);
            int16_t * a_مجموع_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + حجم(اعشاری) * 4);
            int8_t *  quant_a_blk = reinterpret_cast<int8_t *>(quant_a_ptr + حجم(اعشاری) * 4 + حجم(int16_t) * 4);

            برائے (size_t mi = 0; mi < 4; mi++) {
                size_t       vl      = __riscv_vsetvl_e32m1(blk_len);
                vfloat32m1_t v_a     = __riscv_vle32_v_f32m1(a_ptr + mi * گنتی_k + k, vl);
                vfloat32m1_t v_a_abs = __riscv_vfabs_v_f32m1(v_a, vl);

                vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m1_f32m1(v_a_abs, tmp, vl);
                اعشاری        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                اعشاری scale_a     = max_abs_a / ((1 << 7) - 1);
                اعشاری rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
                scale_a_ptr[mi]   = scale_a;

                vfloat32m1_t v_a_scale    = __riscv_vfmul_vf_f32m1(v_a, rep_scale_a, vl);
                vint16mf2_t  v_a_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a_scale, vl);
                vint8mf4_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a_quant, vl);

                vint16m1_t tmp_مجموع = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a_مجموع = __riscv_vwredsum_vs_i8mf4_i16m1(v_a_quant_i8, tmp_مجموع, vl);
                int16_t    a_مجموع   = __riscv_vmv_x_s_i16m1_i16(v_a_مجموع);
                a_مجموع_ptr[mi]      = -a_مجموع;

                __riscv_vse8_v_i8mf4(quant_a_blk + mi * blk_len, v_a_quant_i8, vl);
            }
        }
    } ورنہ اگر (vlenb == 32) {
        برائے (size_t k = 0; k < گنتی_k; k += blk_len, quant_a_ptr += a_nrow_بلاک_stride) {
            اعشاری *   scale_a_ptr = reinterpret_cast<اعشاری *>(quant_a_ptr);
            int16_t * a_مجموع_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + حجم(اعشاری) * 4);
            int8_t *  quant_a_blk = reinterpret_cast<int8_t *>(quant_a_ptr + حجم(اعشاری) * 4 + حجم(int16_t) * 4);

            برائے (size_t mi = 0; mi < 4; mi++) {
                size_t       vl      = __riscv_vsetvl_e32m4(blk_len);
                vfloat32m4_t v_a     = __riscv_vle32_v_f32m4(a_ptr + mi * گنتی_k + k, vl);
                vfloat32m4_t v_a_abs = __riscv_vfabs_v_f32m4(v_a, vl);

                vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m4_f32m1(v_a_abs, tmp, vl);
                اعشاری        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                اعشاری scale_a     = max_abs_a / ((1 << 7) - 1);
                اعشاری rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
                scale_a_ptr[mi]   = scale_a;

                vfloat32m4_t v_a_scale    = __riscv_vfmul_vf_f32m4(v_a, rep_scale_a, vl);
                vint16m2_t   v_a_quant    = __riscv_vfncvt_x_f_w_i16m2(v_a_scale, vl);
                vint8m1_t    v_a_quant_i8 = __riscv_vncvt_x_x_w_i8m1(v_a_quant, vl);

                vint16m1_t tmp_مجموع = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a_مجموع = __riscv_vwredsum_vs_i8m1_i16m1(v_a_quant_i8, tmp_مجموع, vl);
                int16_t    a_مجموع   = __riscv_vmv_x_s_i16m1_i16(v_a_مجموع);
                a_مجموع_ptr[mi]      = -a_مجموع;

                __riscv_vse8_v_i8m1(quant_a_blk + mi * blk_len, v_a_quant_i8, vl);
            }
        }
    } ورنہ {
        quantize_a_nrow_i8_ref<4>(blk_len, a_ptr, گنتی_k, quant_a_ptr);
    }
}

خالی quantize_a_صف_i8_hp(size_t blk_len, ثابت اعشاری * a_ptr, size_t گنتی_k, uint8_t * quant_a_ptr) {
    constexpr size_t k_subblk_len = 32;
    GGML_ASSERT(blk_len == 256);

    constexpr size_t subblk_گنتی             = 256 / k_subblk_len;
    int64_t          a_blk_stride             = q8_hp_blk_سائز(blk_len, سچ, سچ);
    int64_t          a_subblk_stride          = q8_hp_blk_سائز(k_subblk_len, جھوٹ, جھوٹ);
    size_t           vlenb                    = __riscv_vlenb();
    اعشاری            scale_عارضی[subblk_گنتی] = { 0.0f };

    اگر (vlenb == 128) {
        برائے (size_t k = 0; k < گنتی_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            _Float16 * a_مجموع_ptr     = reinterpret_cast<_Float16 *>(quant_a_ptr + a_subblk_stride * subblk_گنتی);
            _Float16 * scale_avg_ptr = reinterpret_cast<_Float16 *>(quant_a_ptr + a_blk_stride - حجم(_Float16));
            اعشاری      scale_avg     = 0.0f;

            برائے (size_t kk = 0; kk < subblk_گنتی; ++kk) {
                ثابت اعشاری * a_src_ptr = a_ptr + k + kk * k_subblk_len;

                size_t       vl      = __riscv_vsetvl_e32m1(k_subblk_len);
                vfloat32m1_t v_a     = __riscv_vle32_v_f32m1(a_src_ptr, vl);
                vfloat32m1_t v_a_abs = __riscv_vfabs_v_f32m1(v_a, vl);

                vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m1_f32m1(v_a_abs, tmp, vl);
                اعشاری        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                scale_عارضی[kk] = max_abs_a / ((1 << 7) - 1);
                scale_avg += scale_عارضی[kk];
            }

            scale_avg /= subblk_گنتی;
            ثابت اعشاری scale_ضربی_عدد = scale_avg ? 1.0f / scale_avg : 0.0f;
            scale_avg_ptr[0]         = static_cast<_Float16>(scale_avg);

            برائے (size_t kk = 0; kk < subblk_گنتی; ++kk) {
                uint8_t *     a_subblk_بنیاد = quant_a_ptr + kk * a_subblk_stride;
                _Float16 *    scale_a_ptr   = reinterpret_cast<_Float16 *>(a_subblk_بنیاد);
                int8_t *      quant_a_blk   = reinterpret_cast<int8_t *>(a_subblk_بنیاد + حجم(_Float16));
                ثابت اعشاری * a_src_ptr     = a_ptr + k + kk * k_subblk_len;

                size_t       vl          = __riscv_vsetvl_e32m1(k_subblk_len);
                vfloat32m1_t v_a         = __riscv_vle32_v_f32m1(a_src_ptr, vl);
                اعشاری        rep_scale_a = scale_عارضی[kk] ? 1.0f / scale_عارضی[kk] : 0.0f;
                scale_a_ptr[0]           = static_cast<_Float16>(scale_عارضی[kk] * scale_ضربی_عدد);

                vfloat32m1_t v_a_scale    = __riscv_vfmul_vf_f32m1(v_a, rep_scale_a, vl);
                vint16mf2_t  v_a_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a_scale, vl);
                vint8mf4_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a_quant, vl);

                vint16m1_t tmp_مجموع = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a_مجموع = __riscv_vwredsum_vs_i8mf4_i16m1(v_a_quant_i8, tmp_مجموع, vl);
                int16_t    a_مجموع   = __riscv_vmv_x_s_i16m1_i16(v_a_مجموع);
                a_مجموع_ptr[kk]      = static_cast<_Float16>(-a_مجموع) * static_cast<_Float16>(8.0f);

                __riscv_vse8_v_i8mf4(quant_a_blk, v_a_quant_i8, vl);
            }
        }
    } ورنہ اگر (vlenb == 32) {
        برائے (size_t k = 0; k < گنتی_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            _Float16 * a_مجموع_ptr     = reinterpret_cast<_Float16 *>(quant_a_ptr + a_subblk_stride * subblk_گنتی);
            _Float16 * scale_avg_ptr = reinterpret_cast<_Float16 *>(quant_a_ptr + a_blk_stride - حجم(_Float16));
            اعشاری      scale_avg     = 0.0f;

            برائے (size_t kk = 0; kk < subblk_گنتی; ++kk) {
                ثابت اعشاری * a_src_ptr = a_ptr + k + kk * k_subblk_len;

                size_t       vl      = __riscv_vsetvl_e32m4(k_subblk_len);
                vfloat32m4_t v_a     = __riscv_vle32_v_f32m4(a_src_ptr, vl);
                vfloat32m4_t v_a_abs = __riscv_vfabs_v_f32m4(v_a, vl);

                vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m4_f32m1(v_a_abs, tmp, vl);
                اعشاری        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                scale_عارضی[kk] = max_abs_a / ((1 << 7) - 1);
                scale_avg += scale_عارضی[kk];
            }

            scale_avg /= subblk_گنتی;
            ثابت اعشاری scale_ضربی_عدد = scale_avg ? 1.0f / scale_avg : 0.0f;
            scale_avg_ptr[0]         = static_cast<_Float16>(scale_avg);

            برائے (size_t kk = 0; kk < subblk_گنتی; ++kk) {
                uint8_t *     a_subblk_بنیاد = quant_a_ptr + kk * a_subblk_stride;
                _Float16 *    scale_a_ptr   = reinterpret_cast<_Float16 *>(a_subblk_بنیاد);
                int8_t *      quant_a_blk   = reinterpret_cast<int8_t *>(a_subblk_بنیاد + حجم(_Float16));
                ثابت اعشاری * a_src_ptr     = a_ptr + k + kk * k_subblk_len;

                size_t       vl          = __riscv_vsetvl_e32m4(k_subblk_len);
                vfloat32m4_t v_a         = __riscv_vle32_v_f32m4(a_src_ptr, vl);
                اعشاری        rep_scale_a = scale_عارضی[kk] ? 1.0f / scale_عارضی[kk] : 0.0f;
                scale_a_ptr[0]           = static_cast<_Float16>(scale_عارضی[kk] * scale_ضربی_عدد);

                vfloat32m4_t v_a_scale    = __riscv_vfmul_vf_f32m4(v_a, rep_scale_a, vl);
                vint16m2_t   v_a_quant    = __riscv_vfncvt_x_f_w_i16m2(v_a_scale, vl);
                vint8m1_t    v_a_quant_i8 = __riscv_vncvt_x_x_w_i8m1(v_a_quant, vl);

                vint16m1_t tmp_مجموع = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a_مجموع = __riscv_vwredsum_vs_i8m1_i16m1(v_a_quant_i8, tmp_مجموع, vl);
                int16_t    a_مجموع   = __riscv_vmv_x_s_i16m1_i16(v_a_مجموع);
                a_مجموع_ptr[kk]      = static_cast<_Float16>(-a_مجموع) * static_cast<_Float16>(8.0f);

                __riscv_vse8_v_i8m1(quant_a_blk, v_a_quant_i8, vl);
            }
        }
    } ورنہ {
        quantize_a_nrow_i8_hp_ref<1>(blk_len, a_ptr, گنتی_k, quant_a_ptr);
    }
}

خالی quantize_a_4row_i8_hp(size_t blk_len, ثابت اعشاری * a_ptr, size_t گنتی_k, uint8_t * quant_a_ptr) {
    constexpr size_t k_subblk_len = 32;
    GGML_ASSERT(blk_len == 256);

    constexpr size_t subblk_گنتی             = 256 / k_subblk_len;
    int64_t          a_blk_stride             = q8_hp_blk_سائز(blk_len, سچ, سچ);
    int64_t          a_nrow_بلاک_stride      = a_blk_stride * 4;
    int64_t          a_subblk_stride          = q8_hp_blk_سائز(k_subblk_len, جھوٹ, جھوٹ) * 4;
    size_t           vlenb                    = __riscv_vlenb();
    اعشاری            scale_عارضی[subblk_گنتی] = { 0.0f };

    اگر (vlenb == 128) {
        برائے (size_t k = 0; k < گنتی_k; k += blk_len, quant_a_ptr += a_nrow_بلاک_stride) {
            _Float16 * a_مجموع_ptr = reinterpret_cast<_Float16 *>(quant_a_ptr + a_subblk_stride * subblk_گنتی);
            _Float16 * scale_avg_ptr =
                reinterpret_cast<_Float16 *>(quant_a_ptr + a_nrow_بلاک_stride - حجم(_Float16) * 4);
            اعشاری scale_avg = 0.0f;

            برائے (size_t kk = 0; kk < subblk_گنتی; ++kk) {
                ثابت اعشاری * a_src_ptr0 = a_ptr + 0 * گنتی_k + k + kk * k_subblk_len;
                ثابت اعشاری * a_src_ptr1 = a_ptr + 1 * گنتی_k + k + kk * k_subblk_len;
                ثابت اعشاری * a_src_ptr2 = a_ptr + 2 * گنتی_k + k + kk * k_subblk_len;
                ثابت اعشاری * a_src_ptr3 = a_ptr + 3 * گنتی_k + k + kk * k_subblk_len;

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
                اعشاری        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                scale_عارضی[kk] = max_abs_a / ((1 << 7) - 1);
                scale_avg += scale_عارضی[kk];
            }

            scale_avg /= subblk_گنتی;
            ثابت اعشاری scale_ضربی_عدد = scale_avg ? 1.0f / scale_avg : 0.0f;
            scale_avg_ptr[0]         = static_cast<_Float16>(scale_avg);

            برائے (size_t kk = 0; kk < subblk_گنتی; ++kk) {
                uint8_t *     a_subblk_بنیاد = quant_a_ptr + kk * a_subblk_stride;
                _Float16 *    scale_a_ptr   = reinterpret_cast<_Float16 *>(a_subblk_بنیاد);
                int8_t *      quant_a_blk   = reinterpret_cast<int8_t *>(a_subblk_بنیاد + حجم(_Float16) * 4);
                ثابت اعشاری * a_src_ptr0    = a_ptr + 0 * گنتی_k + k + kk * k_subblk_len;
                ثابت اعشاری * a_src_ptr1    = a_ptr + 1 * گنتی_k + k + kk * k_subblk_len;
                ثابت اعشاری * a_src_ptr2    = a_ptr + 2 * گنتی_k + k + kk * k_subblk_len;
                ثابت اعشاری * a_src_ptr3    = a_ptr + 3 * گنتی_k + k + kk * k_subblk_len;

                size_t       vl   = __riscv_vsetvl_e32m1(k_subblk_len);
                vfloat32m1_t v_a0 = __riscv_vle32_v_f32m1(a_src_ptr0, vl);
                vfloat32m1_t v_a1 = __riscv_vle32_v_f32m1(a_src_ptr1, vl);
                vfloat32m1_t v_a2 = __riscv_vle32_v_f32m1(a_src_ptr2, vl);
                vfloat32m1_t v_a3 = __riscv_vle32_v_f32m1(a_src_ptr3, vl);

                اعشاری rep_scale_a = scale_عارضی[kk] ? 1.0f / scale_عارضی[kk] : 0.0f;
                scale_a_ptr[0]    = static_cast<_Float16>(scale_عارضی[kk] * scale_ضربی_عدد);

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
                vint16m1_t v_a0_مجموع = __riscv_vwredsum_vs_i8mf4_i16m1(v_a0_quant_i8, tmp_sum0, vl);
                vint16m1_t v_a1_مجموع = __riscv_vwredsum_vs_i8mf4_i16m1(v_a1_quant_i8, tmp_sum1, vl);
                vint16m1_t v_a2_مجموع = __riscv_vwredsum_vs_i8mf4_i16m1(v_a2_quant_i8, tmp_sum2, vl);
                vint16m1_t v_a3_مجموع = __riscv_vwredsum_vs_i8mf4_i16m1(v_a3_quant_i8, tmp_sum3, vl);

                a_مجموع_ptr[0 * subblk_گنتی + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a0_مجموع)) * static_cast<_Float16>(8.0f);
                a_مجموع_ptr[1 * subblk_گنتی + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a1_مجموع)) * static_cast<_Float16>(8.0f);
                a_مجموع_ptr[2 * subblk_گنتی + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a2_مجموع)) * static_cast<_Float16>(8.0f);
                a_مجموع_ptr[3 * subblk_گنتی + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a3_مجموع)) * static_cast<_Float16>(8.0f);

                __riscv_vse8_v_i8mf4(quant_a_blk + 0 * k_subblk_len, v_a0_quant_i8, vl);
                __riscv_vse8_v_i8mf4(quant_a_blk + 1 * k_subblk_len, v_a1_quant_i8, vl);
                __riscv_vse8_v_i8mf4(quant_a_blk + 2 * k_subblk_len, v_a2_quant_i8, vl);
                __riscv_vse8_v_i8mf4(quant_a_blk + 3 * k_subblk_len, v_a3_quant_i8, vl);
            }
        }
    } ورنہ اگر (vlenb == 32) {
        برائے (size_t k = 0; k < گنتی_k; k += blk_len, quant_a_ptr += a_nrow_بلاک_stride) {
            _Float16 * a_مجموع_ptr = reinterpret_cast<_Float16 *>(quant_a_ptr + a_subblk_stride * subblk_گنتی);
            _Float16 * scale_avg_ptr =
                reinterpret_cast<_Float16 *>(quant_a_ptr + a_nrow_بلاک_stride - حجم(_Float16) * 4);
            اعشاری scale_avg = 0.0f;

            برائے (size_t kk = 0; kk < subblk_گنتی; ++kk) {
                ثابت اعشاری * a_src_ptr0 = a_ptr + 0 * گنتی_k + k + kk * k_subblk_len;
                ثابت اعشاری * a_src_ptr1 = a_ptr + 1 * گنتی_k + k + kk * k_subblk_len;
                ثابت اعشاری * a_src_ptr2 = a_ptr + 2 * گنتی_k + k + kk * k_subblk_len;
                ثابت اعشاری * a_src_ptr3 = a_ptr + 3 * گنتی_k + k + kk * k_subblk_len;

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
                اعشاری        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                scale_عارضی[kk] = max_abs_a / ((1 << 7) - 1);
                scale_avg += scale_عارضی[kk];
            }

            scale_avg /= subblk_گنتی;
            ثابت اعشاری scale_ضربی_عدد = scale_avg ? 1.0f / scale_avg : 0.0f;
            scale_avg_ptr[0]         = static_cast<_Float16>(scale_avg);

            برائے (size_t kk = 0; kk < subblk_گنتی; ++kk) {
                uint8_t *     a_subblk_بنیاد = quant_a_ptr + kk * a_subblk_stride;
                _Float16 *    scale_a_ptr   = reinterpret_cast<_Float16 *>(a_subblk_بنیاد);
                int8_t *      quant_a_blk   = reinterpret_cast<int8_t *>(a_subblk_بنیاد + حجم(_Float16) * 4);
                ثابت اعشاری * a_src_ptr0    = a_ptr + 0 * گنتی_k + k + kk * k_subblk_len;
                ثابت اعشاری * a_src_ptr1    = a_ptr + 1 * گنتی_k + k + kk * k_subblk_len;
                ثابت اعشاری * a_src_ptr2    = a_ptr + 2 * گنتی_k + k + kk * k_subblk_len;
                ثابت اعشاری * a_src_ptr3    = a_ptr + 3 * گنتی_k + k + kk * k_subblk_len;

                size_t       vl   = __riscv_vsetvl_e32m4(k_subblk_len);
                vfloat32m4_t v_a0 = __riscv_vle32_v_f32m4(a_src_ptr0, vl);
                vfloat32m4_t v_a1 = __riscv_vle32_v_f32m4(a_src_ptr1, vl);
                vfloat32m4_t v_a2 = __riscv_vle32_v_f32m4(a_src_ptr2, vl);
                vfloat32m4_t v_a3 = __riscv_vle32_v_f32m4(a_src_ptr3, vl);

                اعشاری rep_scale_a = scale_عارضی[kk] ? 1.0f / scale_عارضی[kk] : 0.0f;
                scale_a_ptr[0]    = static_cast<_Float16>(scale_عارضی[kk] * scale_ضربی_عدد);

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
                vint16m1_t v_a0_مجموع = __riscv_vwredsum_vs_i8m1_i16m1(v_a0_quant_i8, tmp_sum0, vl);
                vint16m1_t v_a1_مجموع = __riscv_vwredsum_vs_i8m1_i16m1(v_a1_quant_i8, tmp_sum1, vl);
                vint16m1_t v_a2_مجموع = __riscv_vwredsum_vs_i8m1_i16m1(v_a2_quant_i8, tmp_sum2, vl);
                vint16m1_t v_a3_مجموع = __riscv_vwredsum_vs_i8m1_i16m1(v_a3_quant_i8, tmp_sum3, vl);

                a_مجموع_ptr[0 * subblk_گنتی + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a0_مجموع)) * static_cast<_Float16>(8.0f);
                a_مجموع_ptr[1 * subblk_گنتی + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a1_مجموع)) * static_cast<_Float16>(8.0f);
                a_مجموع_ptr[2 * subblk_گنتی + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a2_مجموع)) * static_cast<_Float16>(8.0f);
                a_مجموع_ptr[3 * subblk_گنتی + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a3_مجموع)) * static_cast<_Float16>(8.0f);

                __riscv_vse8_v_i8m1(quant_a_blk + 0 * k_subblk_len, v_a0_quant_i8, vl);
                __riscv_vse8_v_i8m1(quant_a_blk + 1 * k_subblk_len, v_a1_quant_i8, vl);
                __riscv_vse8_v_i8m1(quant_a_blk + 2 * k_subblk_len, v_a2_quant_i8, vl);
                __riscv_vse8_v_i8m1(quant_a_blk + 3 * k_subblk_len, v_a3_quant_i8, vl);
            }
        }
    } ورنہ {
        quantize_a_nrow_i8_hp_ref<4>(blk_len, a_ptr, گنتی_k, quant_a_ptr);
    }
}

خالی quantize_a_صف_i8k(size_t blk_len, ثابت اعشاری * a_ptr, size_t گنتی_k, uint8_t * quant_a_ptr) {
    GGML_ASSERT(blk_len == 256);
    constexpr int64_t a_blk_stride = q8k_blk_سائز(256);
    constexpr int64_t a_مجموع_سائز   = 256 / 16;
    size_t            vlenb        = __riscv_vlenb();

    اگر (vlenb == 128) {
        // vlen = 1024 bits, can process 32 float32 elements with m1
        برائے (size_t k = 0; k < گنتی_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            اعشاری *   scale_a_ptr = reinterpret_cast<اعشاری *>(quant_a_ptr);
            int16_t * a_مجموع_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + حجم(اعشاری));
            int8_t *  quant_a_blk =
                reinterpret_cast<int8_t *>(quant_a_ptr + حجم(اعشاری) + حجم(int16_t) * a_مجموع_سائز);

            // Find max absolute value across all 256 elements
            size_t       vl        = __riscv_vsetvl_e32m1(16);
            vfloat32m1_t v_max_abs = __riscv_vfmv_v_f_f32m1(0.0f, vl);

            برائے (size_t bki = 0; bki < a_مجموع_سائز; bki++) {
                vfloat32m1_t v_a     = __riscv_vle32_v_f32m1(a_ptr + k + bki * 16, vl);
                vfloat32m1_t v_a_abs = __riscv_vfabs_v_f32m1(v_a, vl);
                v_max_abs            = __riscv_vfmax_vv_f32m1(v_a_abs, v_max_abs, vl);
            }
            vfloat32m1_t tmp         = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t v_local_max = __riscv_vfredmax_vs_f32m1_f32m1(v_max_abs, tmp, vl);
            اعشاری        max_abs_a   = __riscv_vfmv_f_s_f32m1_f32(v_local_max);

            اعشاری scale_a     = max_abs_a / ((1 << 7) - 1);
            اعشاری rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
            scale_a_ptr[0]    = scale_a;

            // Quantize and compute sums for each 16-element group
            برائے (size_t bki = 0; bki < a_مجموع_سائز; bki++) {
                vfloat32m1_t v_a          = __riscv_vle32_v_f32m1(a_ptr + k + bki * 16, vl);
                vfloat32m1_t v_a_scale    = __riscv_vfmul_vf_f32m1(v_a, rep_scale_a, vl);
                vint16mf2_t  v_a_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a_scale, vl);
                vint8mf4_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a_quant, vl);

                vint16m1_t tmp_مجموع = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a_مجموع = __riscv_vwredsum_vs_i8mf4_i16m1(v_a_quant_i8, tmp_مجموع, vl);
                int16_t    a_مجموع   = __riscv_vmv_x_s_i16m1_i16(v_a_مجموع);
                a_مجموع_ptr[bki]     = -a_مجموع;

                __riscv_vse8_v_i8mf4(quant_a_blk + bki * 16, v_a_quant_i8, vl);
            }
        }
    } ورنہ اگر (vlenb == 32) {
        // vlen = 256 bits, can process 8 float32 elements with m1
        برائے (size_t k = 0; k < گنتی_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            اعشاری *   scale_a_ptr = reinterpret_cast<اعشاری *>(quant_a_ptr);
            int16_t * a_مجموع_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + حجم(اعشاری));
            int8_t *  quant_a_blk =
                reinterpret_cast<int8_t *>(quant_a_ptr + حجم(اعشاری) + حجم(int16_t) * a_مجموع_سائز);

            // Find max absolute value across all 256 elements
            size_t       vl        = __riscv_vsetvl_e32m2(16);
            vfloat32m2_t v_max_abs = __riscv_vfmv_v_f_f32m2(0.0f, vl);

            برائے (size_t bki = 0; bki < a_مجموع_سائز; bki++) {
                vfloat32m2_t v_a     = __riscv_vle32_v_f32m2(a_ptr + k + bki * 16, vl);
                vfloat32m2_t v_a_abs = __riscv_vfabs_v_f32m2(v_a, vl);
                v_max_abs            = __riscv_vfmax_vv_f32m2(v_a_abs, v_max_abs, vl);
            }
            vfloat32m1_t tmp         = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t v_local_max = __riscv_vfredmax_vs_f32m2_f32m1(v_max_abs, tmp, vl);
            اعشاری        max_abs_a   = __riscv_vfmv_f_s_f32m1_f32(v_local_max);

            اعشاری scale_a     = max_abs_a / ((1 << 7) - 1);
            اعشاری rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
            scale_a_ptr[0]    = scale_a;

            // Quantize and compute sums for each 16-element group
            برائے (size_t bki = 0; bki < a_مجموع_سائز; bki++) {
                vfloat32m2_t v_a          = __riscv_vle32_v_f32m2(a_ptr + k + bki * 16, vl);
                vfloat32m2_t v_a_scale    = __riscv_vfmul_vf_f32m2(v_a, rep_scale_a, vl);
                vint16m1_t   v_a_quant    = __riscv_vfncvt_x_f_w_i16m1(v_a_scale, vl);
                vint8mf2_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf2(v_a_quant, vl);

                vint16m1_t tmp_مجموع = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a_مجموع = __riscv_vwredsum_vs_i8mf2_i16m1(v_a_quant_i8, tmp_مجموع, vl);
                int16_t    a_مجموع   = __riscv_vmv_x_s_i16m1_i16(v_a_مجموع);
                a_مجموع_ptr[bki]     = -a_مجموع;

                __riscv_vse8_v_i8mf2(quant_a_blk + bki * 16, v_a_quant_i8, vl);
            }
        }
    } ورنہ {
        quantize_a_nrow_i8k_ref<1>(blk_len, a_ptr, گنتی_k, quant_a_ptr);
    }
}

خالی quantize_a_4row_i8k(size_t blk_len, ثابت اعشاری * a_ptr, size_t گنتی_k, uint8_t * quant_a_ptr) {
    GGML_ASSERT(blk_len == 256);
    constexpr int64_t a_blk_stride        = q8k_blk_سائز(256);
    constexpr int64_t a_nrow_بلاک_stride = a_blk_stride * 4;
    constexpr int64_t a_مجموع_سائز          = 256 / 16;
    size_t            vlenb               = __riscv_vlenb();

    اگر (vlenb == 128) {
        // vlen = 1024 bits
        برائے (size_t k = 0; k < گنتی_k; k += blk_len, quant_a_ptr += a_nrow_بلاک_stride) {
            اعشاری *   scale_a_ptr = reinterpret_cast<اعشاری *>(quant_a_ptr);
            int16_t * a_مجموع_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + حجم(اعشاری) * 4);
            int8_t *  quant_a_blk =
                reinterpret_cast<int8_t *>(quant_a_ptr + حجم(اعشاری) * 4 + حجم(int16_t) * a_مجموع_سائز * 4);

            برائے (size_t mi = 0; mi < 4; mi++) {
                // Find max absolute value across all 256 elements for this row
                size_t       vl        = __riscv_vsetvl_e32m1(16);
                vfloat32m1_t v_max_abs = __riscv_vfmv_v_f_f32m1(0.0f, vl);

                برائے (size_t bki = 0; bki < a_مجموع_سائز; bki++) {
                    vfloat32m1_t v_a     = __riscv_vle32_v_f32m1(a_ptr + mi * گنتی_k + k + bki * 16, vl);
                    vfloat32m1_t v_a_abs = __riscv_vfabs_v_f32m1(v_a, vl);
                    v_max_abs            = __riscv_vfmax_vv_f32m1(v_a_abs, v_max_abs, vl);
                }
                vfloat32m1_t tmp         = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_local_max = __riscv_vfredmax_vs_f32m1_f32m1(v_max_abs, tmp, vl);
                اعشاری        max_abs_a   = __riscv_vfmv_f_s_f32m1_f32(v_local_max);

                اعشاری scale_a     = max_abs_a / ((1 << 7) - 1);
                اعشاری rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
                scale_a_ptr[mi]   = scale_a;

                // Quantize and compute sums for each 16-element group
                برائے (size_t bki = 0; bki < a_مجموع_سائز; bki++) {
                    vfloat32m1_t v_a          = __riscv_vle32_v_f32m1(a_ptr + mi * گنتی_k + k + bki * 16, vl);
                    vfloat32m1_t v_a_scale    = __riscv_vfmul_vf_f32m1(v_a, rep_scale_a, vl);
                    vint16mf2_t  v_a_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a_scale, vl);
                    vint8mf4_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a_quant, vl);

                    vint16m1_t tmp_مجموع               = __riscv_vmv_v_x_i16m1(0, vl);
                    vint16m1_t v_a_مجموع               = __riscv_vwredsum_vs_i8mf4_i16m1(v_a_quant_i8, tmp_مجموع, vl);
                    int16_t    a_مجموع                 = __riscv_vmv_x_s_i16m1_i16(v_a_مجموع);
                    a_مجموع_ptr[mi * a_مجموع_سائز + bki] = -a_مجموع;

                    __riscv_vse8_v_i8mf4(quant_a_blk + mi * blk_len + bki * 16, v_a_quant_i8, vl);
                }
            }
        }
    } ورنہ اگر (vlenb == 32) {
        // vlen = 256 bits
        برائے (size_t k = 0; k < گنتی_k; k += blk_len, quant_a_ptr += a_nrow_بلاک_stride) {
            اعشاری *   scale_a_ptr = reinterpret_cast<اعشاری *>(quant_a_ptr);
            int16_t * a_مجموع_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + حجم(اعشاری) * 4);
            int8_t *  quant_a_blk =
                reinterpret_cast<int8_t *>(quant_a_ptr + حجم(اعشاری) * 4 + حجم(int16_t) * a_مجموع_سائز * 4);

            برائے (size_t mi = 0; mi < 4; mi++) {
                // Find max absolute value across all 256 elements for this row
                size_t       vl        = __riscv_vsetvl_e32m2(16);
                vfloat32m2_t v_max_abs = __riscv_vfmv_v_f_f32m2(0.0f, vl);

                برائے (size_t bki = 0; bki < a_مجموع_سائز; bki++) {
                    vfloat32m2_t v_a     = __riscv_vle32_v_f32m2(a_ptr + mi * گنتی_k + k + bki * 16, vl);
                    vfloat32m2_t v_a_abs = __riscv_vfabs_v_f32m2(v_a, vl);
                    v_max_abs            = __riscv_vfmax_vv_f32m2(v_a_abs, v_max_abs, vl);
                }
                vfloat32m1_t tmp         = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_local_max = __riscv_vfredmax_vs_f32m2_f32m1(v_max_abs, tmp, vl);
                اعشاری        max_abs_a   = __riscv_vfmv_f_s_f32m1_f32(v_local_max);

                اعشاری scale_a     = max_abs_a / ((1 << 7) - 1);
                اعشاری rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
                scale_a_ptr[mi]   = scale_a;

                // Quantize and compute sums for each 16-element group
                برائے (size_t bki = 0; bki < a_مجموع_سائز; bki++) {
                    vfloat32m2_t v_a          = __riscv_vle32_v_f32m2(a_ptr + mi * گنتی_k + k + bki * 16, vl);
                    vfloat32m2_t v_a_scale    = __riscv_vfmul_vf_f32m2(v_a, rep_scale_a, vl);
                    vint16m1_t   v_a_quant    = __riscv_vfncvt_x_f_w_i16m1(v_a_scale, vl);
                    vint8mf2_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf2(v_a_quant, vl);

                    vint16m1_t tmp_مجموع               = __riscv_vmv_v_x_i16m1(0, vl);
                    vint16m1_t v_a_مجموع               = __riscv_vwredsum_vs_i8mf2_i16m1(v_a_quant_i8, tmp_مجموع, vl);
                    int16_t    a_مجموع                 = __riscv_vmv_x_s_i16m1_i16(v_a_مجموع);
                    a_مجموع_ptr[mi * a_مجموع_سائز + bki] = -a_مجموع;

                    __riscv_vse8_v_i8mf2(quant_a_blk + mi * blk_len + bki * 16, v_a_quant_i8, vl);
                }
            }
        }
    } ورنہ {
        quantize_a_nrow_i8k_ref<4>(blk_len, a_ptr, گنتی_k, quant_a_ptr);
    }
}

خالی forward_cpy_with_ترتیب_بدلو(ggml_compute_params * params, ggml_tensor * op) {
    ثابت ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;
    ثابت صحیح           ith  = params->ith;
    ثابت صحیح           nth  = params->nth;

    // [batch, m, n] -> [batch, n, m]
    int64_t batch = src0->ne[2] * src0->ne[3];
    int64_t m     = src0->ne[1];
    int64_t n     = src0->ne[0];

    int64_t batch_stride = src0->nb[2];
    int64_t m_src_stride = src0->nb[0];
    int64_t n_src_stride = src0->nb[1];
    int64_t n_dst_stride = n_src_stride * m;

    ترتیب_بدلو_transpose_impl(src0, dst, batch, m, n, batch_stride, m_src_stride, n_src_stride, n_dst_stride, ith, nth);
}

خالی forward_cont_with_ترتیب_بدلو(ggml_compute_params * params, ggml_tensor * op) {
    ثابت ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;
    ثابت صحیح           ith  = params->ith;
    ثابت صحیح           nth  = params->nth;

    // [batch, m, n] -> [batch, n, m]
    int64_t batch = dst->ne[2] * dst->ne[3];
    int64_t n     = dst->ne[1];
    int64_t m     = dst->ne[0];

    int64_t batch_stride = dst->nb[2];
    int64_t m_src_stride = src0->nb[0];
    int64_t n_src_stride = src0->nb[1];
    int64_t n_dst_stride = dst->nb[1];

    ترتیب_بدلو_transpose_impl(src0, dst, batch, m, n, batch_stride, m_src_stride, n_src_stride, n_dst_stride, ith, nth);
}

خالی forward_norm_f32(ggml_compute_params * params, ggml_tensor * op) {
    ثابت ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(src0->nb[0] == حجم(اعشاری));

    صحیح ith = params->ith;
    صحیح nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    اعشاری epsilon = *((اعشاری *) dst->op_params);

    GGML_ASSERT(epsilon > 0.0f);

    خودکار * داخلہ  = (حرف *) src0->ڈیٹا;
    خودکار * output = (حرف *) dst->ڈیٹا;

    ثابت خودکار hidden_سائز     = ne00;
    ثابت خودکار task_گنتی      = ne01 * ne02 * ne03;
    ثابت خودکار task_فی_thread = (task_گنتی + nth - 1) / nth;

    ثابت خودکار task_begin = ith * task_فی_thread;
    ثابت خودکار task_انتہا   = std::min((ith + 1) * task_فی_thread, task_گنتی);

    برائے (خودکار task_idx = task_begin; task_idx < task_انتہا; task_idx++) {
        int64_t i03 = task_idx / (ne02 * ne01);
        int64_t i02 = (task_idx - i03 * ne02 * ne01) / ne01;
        int64_t i01 = (task_idx - i03 * ne02 * ne01 - i02 * ne01);

        خودکار * p_داخلہ       = (اعشاری *) (داخلہ + i01 * nb01 + i02 * nb02 + i03 * nb03);
        خودکار * p_output      = (اعشاری *) (output + i01 * nb1 + i02 * nb2 + i03 * nb3);
        خودکار * p_عارضی_output = p_output;

        size_t       gvl    = __riscv_vsetvlmax_e32m4();
        vfloat32m4_t مجموع    = __riscv_vfmv_v_f_f32m4(0.f, gvl);
        vfloat32m4_t مجموع_sq = __riscv_vfmv_v_f_f32m4(0.f, gvl);
        int64_t      لمبائی = hidden_سائز;
        جب (لمبائی > 0) {
            gvl                   = __riscv_vsetvl_e32m4(لمبائی);
            // load data
            vfloat32m4_t src_ڈیٹا = __riscv_vle32_v_f32m4(p_داخلہ, gvl);

            مجموع    = __riscv_vfadd_vv_f32m4(مجموع, src_ڈیٹا, gvl);
            مجموع_sq = __riscv_vfmacc_vv_f32m4(مجموع_sq, src_ڈیٹا, src_ڈیٹا, gvl);

            __riscv_vse32_v_f32m4(p_عارضی_output, src_ڈیٹا, gvl);

            p_داخلہ += gvl;
            p_عارضی_output += gvl;
            لمبائی -= gvl;
        }

        gvl = __riscv_vsetvlmax_e32m1();

        اعشاری        اوسط_2   = 0.f;
        vfloat32m1_t zero_v = __riscv_vfmv_v_f_f32m1(0.f, gvl);
        vfloat32m1_t اوسط_v =
            __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(مجموع, 0), __riscv_vget_v_f32m4_f32m1(مجموع, 1), gvl);
        اوسط_v = __riscv_vfadd_vv_f32m1(اوسط_v, __riscv_vget_v_f32m4_f32m1(مجموع, 2), gvl);
        اوسط_v = __riscv_vfadd_vv_f32m1(اوسط_v, __riscv_vget_v_f32m4_f32m1(مجموع, 3), gvl);
        اوسط_v = __riscv_vfredusum_vs_f32m1_f32m1(اوسط_v, zero_v, gvl);
        اوسط_2   = __riscv_vfmv_f_s_f32m1_f32(اوسط_v);
        اوسط_2 /= hidden_سائز;

        vfloat32m1_t اوسط_مربع_v =
            __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(مجموع_sq, 0), __riscv_vget_v_f32m4_f32m1(مجموع_sq, 1), gvl);
        اوسط_مربع_v = __riscv_vfadd_vv_f32m1(اوسط_مربع_v, __riscv_vget_v_f32m4_f32m1(مجموع_sq, 2), gvl);
        اوسط_مربع_v = __riscv_vfadd_vv_f32m1(اوسط_مربع_v, __riscv_vget_v_f32m4_f32m1(مجموع_sq, 3), gvl);
        اوسط_مربع_v = __riscv_vfredusum_vs_f32m1_f32m1(اوسط_مربع_v, zero_v, gvl);

        اعشاری اوسط_مربع = __riscv_vfmv_f_s_f32m1_f32(اوسط_مربع_v);
        اوسط_مربع /= hidden_سائز;
        اوسط_مربع = جذر(اوسط_مربع - اوسط_2 * اوسط_2 + epsilon);

        اوسط_مربع   = 1.0f / اوسط_مربع;
        لمبائی        = hidden_سائز;
        p_عارضی_output = p_output;

        جب (لمبائی > 0) {
            gvl                   = __riscv_vsetvl_e32m4(لمبائی);
            vfloat32m4_t src_ڈیٹا = __riscv_vle32_v_f32m4(p_عارضی_output, gvl);
            src_ڈیٹا              = __riscv_vfsub_vf_f32m4(src_ڈیٹا, اوسط_2, gvl);
            src_ڈیٹا              = __riscv_vfmul_vf_f32m4(src_ڈیٹا, اوسط_مربع, gvl);
            __riscv_vse32_v_f32m4(p_output, src_ڈیٹا, gvl);
            p_عارضی_output += gvl;
            p_output += gvl;
            لمبائی -= gvl;
        }
    }
}

سانچہ <ggml_op op_type, قسم_نام T> خالی forward_ثنائی(ggml_compute_params * params, ggml_tensor * op) {
    ثابت ggml_tensor * src0 = op->src[0];
    ثابت ggml_tensor * src1 = op->src[1];
    ggml_tensor *       dst  = op;
    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    خودکار src0_صفیں = ggml_nrows(src0);
    خودکار src1_صفیں = ggml_nrows(src1);

    صحیح ith = params->ith;
    صحیح nth = params->nth;

    GGML_TENSOR_ثنائی_OP_LOCALS

    GGML_ASSERT(nb0 == حجم(T));
    GGML_ASSERT(nb00 == حجم(T));

    ثابت خودکار [ir0, ir1] = get_thread_range(params, src0);

    خودکار compute_func_vv = [&](int64_t blk_len, int64_t r, T * src0_ptr, T * src1_ptr, T * dst_ptr) {
        int64_t idx = 0;
        اگر constexpr (op_type == GGML_OP_جوڑو) {
            اگر constexpr (std::is_same_v<T, اعشاری>) {
                برائے (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e32m4(blk_len);
                    vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + idx + r, vl);
                    vfloat32m4_t rhs = __riscv_vle32_v_f32m4(src1_ptr + idx, vl);
                    vfloat32m4_t res = __riscv_vfadd_vv_f32m4(lhs, rhs, vl);
                    __riscv_vse32_v_f32m4(dst_ptr + idx + r, res, vl);
                }
            } ورنہ اگر constexpr (std::is_same_v<T, _Float16>) {
                برائے (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e16m4(blk_len);
                    vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + idx + r), vl);
                    vfloat16m4_t rhs = __riscv_vle16_v_f16m4((src1_ptr + idx), vl);
                    vfloat16m4_t res = __riscv_vfadd_vv_f16m4(lhs, rhs, vl);
                    __riscv_vse16_v_f16m4((dst_ptr + idx + r), res, vl);
                }
            } ورنہ {
                GGML_ABORT("fatal error");
            }
        } ورنہ اگر constexpr (op_type == GGML_OP_SUB) {
            اگر constexpr (std::is_same_v<T, اعشاری>) {
                برائے (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e32m4(blk_len);
                    vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + idx + r, vl);
                    vfloat32m4_t rhs = __riscv_vle32_v_f32m4(src1_ptr + idx, vl);
                    vfloat32m4_t res = __riscv_vfsub_vv_f32m4(lhs, rhs, vl);
                    __riscv_vse32_v_f32m4(dst_ptr + idx + r, res, vl);
                }
            } ورنہ اگر constexpr (std::is_same_v<T, _Float16>) {
                برائے (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e16m4(blk_len);
                    vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + idx + r), vl);
                    vfloat16m4_t rhs = __riscv_vle16_v_f16m4((src1_ptr + idx), vl);
                    vfloat16m4_t res = __riscv_vfsub_vv_f16m4(lhs, rhs, vl);
                    __riscv_vse16_v_f16m4((dst_ptr + idx + r), res, vl);
                }
            } ورنہ {
                GGML_ABORT("fatal error");
            }
        } ورنہ اگر constexpr (op_type == GGML_OP_MUL) {
            اگر constexpr (std::is_same_v<T, اعشاری>) {
                برائے (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e32m4(blk_len);
                    vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + idx + r, vl);
                    vfloat32m4_t rhs = __riscv_vle32_v_f32m4(src1_ptr + idx, vl);
                    vfloat32m4_t res = __riscv_vfmul_vv_f32m4(lhs, rhs, vl);
                    __riscv_vse32_v_f32m4(dst_ptr + idx + r, res, vl);
                }
            } ورنہ اگر constexpr (std::is_same_v<T, _Float16>) {
                برائے (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e16m4(blk_len);
                    vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + idx + r), vl);
                    vfloat16m4_t rhs = __riscv_vle16_v_f16m4((src1_ptr + idx), vl);
                    vfloat16m4_t res = __riscv_vfmul_vv_f16m4(lhs, rhs, vl);
                    __riscv_vse16_v_f16m4((dst_ptr + idx + r), res, vl);
                }
            } ورنہ {
                GGML_ABORT("fatal error");
            }
        } ورنہ اگر constexpr (op_type == GGML_OP_DIV) {
            اگر constexpr (std::is_same_v<T, اعشاری>) {
                برائے (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e32m4(blk_len);
                    vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + idx + r, vl);
                    vfloat32m4_t rhs = __riscv_vle32_v_f32m4(src1_ptr + idx, vl);
                    vfloat32m4_t res = __riscv_vfdiv_vv_f32m4(lhs, rhs, vl);
                    __riscv_vse32_v_f32m4(dst_ptr + idx + r, res, vl);
                }
            } ورنہ اگر constexpr (std::is_same_v<T, _Float16>) {
                برائے (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e16m4(blk_len);
                    vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + idx + r), vl);
                    vfloat16m4_t rhs = __riscv_vle16_v_f16m4((src1_ptr + idx), vl);
                    vfloat16m4_t res = __riscv_vfdiv_vv_f16m4(lhs, rhs, vl);
                    __riscv_vse16_v_f16m4((dst_ptr + idx + r), res, vl);
                }
            } ورنہ {
                GGML_ABORT("fatal error");
            }
        } ورنہ {
            GGML_ABORT("fatal error");
        }
    };

    اگر (src0_صفیں == src1_صفیں && src0_صفیں == 1 && ne00 == ne10) {
        int64_t task_فی_thread = (ne00 + nth - 1) / nth;
        int64_t task_begin      = ith * task_فی_thread;
        int64_t task_انتہا        = std::min((ith + 1) * task_فی_thread, ne00);

        T * dst_ptr  = ((T *) dst->ڈیٹا) + task_begin;
        T * src0_ptr = ((T *) src0->ڈیٹا) + task_begin;
        T * src1_ptr = ((T *) src1->ڈیٹا) + task_begin;

        compute_func_vv(task_انتہا - task_begin, 0, src0_ptr, src1_ptr, dst_ptr);
    } ورنہ اگر (ne10 > 1) {
        برائے (int64_t ir = ir0; ir < ir1; ++ir) {
            ثابت int64_t i03 = ir / (ne02 * ne01);
            ثابت int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            ثابت int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

            ثابت int64_t i13 = i03 % ne13;
            ثابت int64_t i12 = i02 % ne12;
            ثابت int64_t i11 = i01 % ne11;

            T * dst_ptr  = (T *) ((حرف *) dst->ڈیٹا + i03 * nb3 + i02 * nb2 + i01 * nb1);
            T * src0_ptr = (T *) ((حرف *) src0->ڈیٹا + i03 * nb03 + i02 * nb02 + i01 * nb01);
            T * src1_ptr = (T *) ((حرف *) src1->ڈیٹا + i13 * nb13 + i12 * nb12 + i11 * nb11);

            // src1 is broadcastable across src0 and dst in i1, i2, i3
            برائے (int64_t r = 0; r < ne00; r += ne10) {
                compute_func_vv(ne10, r, src0_ptr, src1_ptr, dst_ptr);
            }
        }
    } ورنہ {
        برائے (int64_t ir = ir0; ir < ir1; ++ir) {
            ثابت int64_t i03 = ir / (ne02 * ne01);
            ثابت int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            ثابت int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

            ثابت int64_t i13 = i03 % ne13;
            ثابت int64_t i12 = i02 % ne12;
            ثابت int64_t i11 = i01 % ne11;

            T * dst_ptr  = (T *) ((حرف *) dst->ڈیٹا + i03 * nb3 + i02 * nb2 + i01 * nb1);
            T * src0_ptr = (T *) ((حرف *) src0->ڈیٹا + i03 * nb03 + i02 * nb02 + i01 * nb01);
            T * src1_ptr = (T *) ((حرف *) src1->ڈیٹا + i13 * nb13 + i12 * nb12 + i11 * nb11);

            T       rhs_scalar = src1_ptr[0];
            int64_t blk_len    = ne00;
            int64_t r          = 0;

            برائے (size_t vl; blk_len > 0; blk_len -= vl, r += vl) {
                اگر constexpr (op_type == GGML_OP_جوڑو) {
                    اگر constexpr (std::is_same_v<T, اعشاری>) {
                        vl               = __riscv_vsetvl_e32m4(blk_len);
                        vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + r, vl);
                        vfloat32m4_t res = __riscv_vfadd_vf_f32m4(lhs, rhs_scalar, vl);
                        __riscv_vse32_v_f32m4(dst_ptr + r, res, vl);
                    } ورنہ اگر constexpr (std::is_same_v<T, _Float16>) {
                        vl               = __riscv_vsetvl_e16m4(blk_len);
                        vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + r), vl);
                        vfloat16m4_t res = __riscv_vfadd_vf_f16m4(lhs, rhs_scalar, vl);
                        __riscv_vse16_v_f16m4((dst_ptr + r), res, vl);
                    } ورنہ {
                        GGML_ABORT("fatal error");
                    }
                } ورنہ اگر constexpr (op_type == GGML_OP_SUB) {
                    اگر constexpr (std::is_same_v<T, اعشاری>) {
                        vl               = __riscv_vsetvl_e32m4(blk_len);
                        vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + r, vl);
                        vfloat32m4_t res = __riscv_vfsub_vf_f32m4(lhs, rhs_scalar, vl);
                        __riscv_vse32_v_f32m4(dst_ptr + r, res, vl);
                    } ورنہ اگر constexpr (std::is_same_v<T, _Float16>) {
                        vl               = __riscv_vsetvl_e16m4(blk_len);
                        vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + r), vl);
                        vfloat16m4_t res = __riscv_vfsub_vf_f16m4(lhs, rhs_scalar, vl);
                        __riscv_vse16_v_f16m4((dst_ptr + r), res, vl);
                    } ورنہ {
                        GGML_ABORT("fatal error");
                    }
                } ورنہ اگر constexpr (op_type == GGML_OP_MUL) {
                    اگر constexpr (std::is_same_v<T, اعشاری>) {
                        vl               = __riscv_vsetvl_e32m4(blk_len);
                        vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + r, vl);
                        vfloat32m4_t res = __riscv_vfmul_vf_f32m4(lhs, rhs_scalar, vl);
                        __riscv_vse32_v_f32m4(dst_ptr + r, res, vl);
                    } ورنہ اگر constexpr (std::is_same_v<T, _Float16>) {
                        vl               = __riscv_vsetvl_e16m4(blk_len);
                        vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + r), vl);
                        vfloat16m4_t res = __riscv_vfmul_vf_f16m4(lhs, rhs_scalar, vl);
                        __riscv_vse16_v_f16m4((dst_ptr + r), res, vl);
                    } ورنہ {
                        GGML_ABORT("fatal error");
                    }
                } ورنہ اگر constexpr (op_type == GGML_OP_DIV) {
                    اگر constexpr (std::is_same_v<T, اعشاری>) {
                        vl               = __riscv_vsetvl_e32m4(blk_len);
                        vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + r, vl);
                        vfloat32m4_t res = __riscv_vfdiv_vf_f32m4(lhs, rhs_scalar, vl);
                        __riscv_vse32_v_f32m4(dst_ptr + r, res, vl);
                    } ورنہ اگر constexpr (std::is_same_v<T, _Float16>) {
                        vl               = __riscv_vsetvl_e16m4(blk_len);
                        vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + r), vl);
                        vfloat16m4_t res = __riscv_vfdiv_vf_f16m4(lhs, rhs_scalar, vl);
                        __riscv_vse16_v_f16m4((dst_ptr + r), res, vl);
                    } ورنہ {
                        GGML_ABORT("fatal error");
                    }
                } ورنہ {
                    GGML_ABORT("fatal error");
                }
            }
        }
    }
}

سانچہ <قسم_نام T> خالی forward_مجموع_صفیں(ثابت ggml_compute_params * params, ggml_tensor * op) {
    ثابت ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne0 == 1);
    GGML_ASSERT(ne1 == ne01);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    int64_t n_task          = ne01 * ne02 * ne03;
    int64_t task_فی_thread = (n_task + nth - 1) / nth;
    int64_t ir_آغاز        = ith * task_فی_thread;
    int64_t ir_انتہا          = std::min(ir_آغاز + task_فی_thread, n_task);

    برائے (int64_t ir = ir_آغاز; ir < ir_انتہا; ir++) {
        ثابت int64_t i3 = ir / (ne02 * ne01);
        ثابت int64_t i2 = (ir - i3 * ne02 * ne01) / ne01;
        ثابت int64_t i1 = (ir - i3 * ne02 * ne01 - i2 * ne01);

        T * src_صف = (T *) ((حرف *) src0->ڈیٹا + i1 * nb01 + i2 * nb02 + i3 * nb03);
        T * dst_صف = (T *) ((حرف *) op->ڈیٹا + i1 * nb1 + i2 * nb2 + i3 * nb3);

        اعشاری صف_مجموع = 0;

        اگر constexpr (std::is_same_v<T, اعشاری>) {
            size_t        gvl     = __riscv_vsetvlmax_e32m4();
            vfloat32m4_t  acc_vec = __riscv_vfmv_v_f_f32m4(0.0f, gvl);
            int64_t       لمبائی  = ne00;
            ثابت اعشاری * p_ڈیٹا  = src_صف;

            جب (لمبائی > 0) {
                size_t       vl  = __riscv_vsetvl_e32m4(لمبائی);
                vfloat32m4_t vec = __riscv_vle32_v_f32m4(p_ڈیٹا, vl);
                acc_vec          = __riscv_vfadd_vv_f32m4(acc_vec, vec, vl);
                p_ڈیٹا += vl;
                لمبائی -= vl;
            }

            gvl                 = __riscv_vsetvlmax_e32m1();
            vfloat32m1_t zero_v = __riscv_vfmv_v_f_f32m1(0.0f, gvl);
            vfloat32m1_t مجموع_v  = __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(acc_vec, 0),
                                                         __riscv_vget_v_f32m4_f32m1(acc_vec, 1), gvl);
            مجموع_v               = __riscv_vfadd_vv_f32m1(مجموع_v, __riscv_vget_v_f32m4_f32m1(acc_vec, 2), gvl);
            مجموع_v               = __riscv_vfadd_vv_f32m1(مجموع_v, __riscv_vget_v_f32m4_f32m1(acc_vec, 3), gvl);
            مجموع_v               = __riscv_vfredusum_vs_f32m1_f32m1(مجموع_v, zero_v, gvl);
            صف_مجموع             = __riscv_vfmv_f_s_f32m1_f32(مجموع_v);
        } ورنہ اگر constexpr (std::is_same_v<T, _Float16>) {
            size_t           gvl     = __riscv_vsetvlmax_e16m2();
            vfloat32m4_t     acc_vec = __riscv_vfmv_v_f_f32m4(0.0f, gvl);
            int64_t          لمبائی  = ne00;
            ثابت _Float16 * p_ڈیٹا  = src_صف;

            جب (لمبائی > 0) {
                size_t       vl      = __riscv_vsetvl_e16m2(لمبائی);
                vfloat16m2_t vec_f16 = __riscv_vle16_v_f16m2(p_ڈیٹا, vl);
                vfloat32m4_t vec_f32 = __riscv_vfwcvt_f_f_v_f32m4(vec_f16, vl);
                acc_vec              = __riscv_vfadd_vv_f32m4(acc_vec, vec_f32, vl);
                p_ڈیٹا += vl;
                لمبائی -= vl;
            }

            gvl                 = __riscv_vsetvlmax_e32m1();
            vfloat32m1_t zero_v = __riscv_vfmv_v_f_f32m1(0.0f, gvl);
            vfloat32m1_t مجموع_v  = __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(acc_vec, 0),
                                                         __riscv_vget_v_f32m4_f32m1(acc_vec, 1), gvl);
            مجموع_v               = __riscv_vfadd_vv_f32m1(مجموع_v, __riscv_vget_v_f32m4_f32m1(acc_vec, 2), gvl);
            مجموع_v               = __riscv_vfadd_vv_f32m1(مجموع_v, __riscv_vget_v_f32m4_f32m1(acc_vec, 3), gvl);
            مجموع_v               = __riscv_vfredusum_vs_f32m1_f32m1(مجموع_v, zero_v, gvl);
            صف_مجموع             = __riscv_vfmv_f_s_f32m1_f32(مجموع_v);
        } ورنہ {
            GGML_ABORT("fatal error");
        }

        dst_صف[0] = صف_مجموع;
    }
}

سانچہ <قسم_نام T> خالی forward_repeat_nrows(ggml_compute_params * params, ggml_tensor * op) {
    ثابت ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    int64_t nrows            = ggml_nrows(src0);
    int64_t nrows_فی_thread = (nrows + nth - 1) / nth;
    int64_t ir_آغاز         = ith * nrows_فی_thread;
    int64_t ir_انتہا           = std::min(ir_آغاز + nrows_فی_thread, nrows);

    اگر (src0->ne[0] == 1) {
        برائے (int64_t ir = ir_آغاز; ir < ir_انتہا; ir++) {
            T * src_صف = (T *) ((حرف *) src0->ڈیٹا + ir * src0->nb[1]);
            T * dst_صف = (T *) ((حرف *) dst->ڈیٹا + ir * dst->nb[1]);

            T src_scalar = src_صف[0];

            int64_t لمبائی = dst->ne[0];
            int64_t idx    = 0;
            size_t  vl     = 0;

            جب (لمبائی > 0) {
                اگر constexpr (std::is_same_v<T, int32_t>) {
                    vl             = __riscv_vsetvl_e32m4(لمبائی);
                    vint32m4_t vec = __riscv_vmv_v_x_i32m4(src_scalar, vl);
                    __riscv_vse32_v_i32m4(dst_صف + idx, vec, vl);
                } ورنہ اگر constexpr (std::is_same_v<T, int16_t>) {
                    vl             = __riscv_vsetvl_e16m4(لمبائی);
                    vint16m4_t vec = __riscv_vmv_v_x_i16m4(src_scalar, vl);
                    __riscv_vse16_v_i16m4((dst_صف + idx), vec, vl);
                } ورنہ {
                    GGML_ABORT("fatal error");
                }
                idx += vl;
                لمبائی -= vl;
            }
        }
    } ورنہ اگر (src0->ne[0] == dst->ne[0]) {
        برائے (int64_t ir = ir_آغاز; ir < ir_انتہا; ir++) {
            T * src_صف = (T *) ((حرف *) src0->ڈیٹا + ir * src0->nb[1]);
            T * dst_صف = (T *) ((حرف *) dst->ڈیٹا + ir * dst->nb[1]);

            int64_t لمبائی = dst->ne[0];
            int64_t idx    = 0;
            size_t  vl     = 0;

            جب (لمبائی > 0) {
                اگر constexpr (std::is_same_v<T, int32_t>) {
                    vl             = __riscv_vsetvl_e32m4(لمبائی);
                    vint32m4_t vec = __riscv_vle32_v_i32m4(src_صف + idx, vl);
                    __riscv_vse32_v_i32m4(dst_صف + idx, vec, vl);
                } ورنہ اگر constexpr (std::is_same_v<T, int16_t>) {
                    vl             = __riscv_vsetvl_e16m4(لمبائی);
                    vint16m4_t vec = __riscv_vle16_v_i16m4((src_صف + idx), vl);
                    __riscv_vse16_v_i16m4((dst_صف + idx), vec, vl);
                } ورنہ {
                    GGML_ABORT("fatal error");
                }
                idx += vl;
                لمبائی -= vl;
            }
        }
    } ورنہ {
        GGML_ABORT("fatal error");
    }
}

سانچہ <قسم_نام T> خالی forward_repeat_dim1(ggml_compute_params * params, ggml_tensor * op) {
    ثابت ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت int64_t ne0 = dst->ne[0];
    ثابت int64_t ne1 = dst->ne[1];
    ثابت int64_t ne2 = dst->ne[2];
    ثابت int64_t ne3 = dst->ne[3];

    ثابت int64_t کل_batches      = ne2 * ne3;
    ثابت int64_t batches_فی_thread = (کل_batches + nth - 1) / nth;
    ثابت int64_t batch_آغاز        = ith * batches_فی_thread;
    ثابت int64_t batch_انتہا          = std::min(batch_آغاز + batches_فی_thread, کل_batches);

    برائے (int64_t b = batch_آغاز; b < batch_انتہا; b++) {
        ثابت int64_t i3 = b / ne2;
        ثابت int64_t i2 = b % ne2;

        T * src_بنیاد  = (T *) ((حرف *) src0->ڈیٹا + i2 * src0->nb[2] + i3 * src0->nb[3]);
        T * dst_batch = (T *) ((حرف *) dst->ڈیٹا + i2 * dst->nb[2] + i3 * dst->nb[3]);

        برائے (int64_t i1 = 0; i1 < ne1; i1++) {
            T *     dst_ptr = (T *) ((حرف *) dst_batch + i1 * dst->nb[1]);
            int64_t لمبائی  = ne0;
            int64_t idx     = 0;

            جب (لمبائی > 0) {
                اگر constexpr (std::is_same_v<T, int32_t>) {
                    size_t     vl  = __riscv_vsetvl_e32m4(لمبائی);
                    vint32m4_t vec = __riscv_vle32_v_i32m4(src_بنیاد + idx, vl);
                    __riscv_vse32_v_i32m4(dst_ptr + idx, vec, vl);
                    idx += vl;
                    لمبائی -= vl;
                } ورنہ اگر constexpr (std::is_same_v<T, int16_t>) {
                    size_t     vl  = __riscv_vsetvl_e16m4(لمبائی);
                    vint16m4_t vec = __riscv_vle16_v_i16m4((src_بنیاد + idx), vl);
                    __riscv_vse16_v_i16m4((dst_ptr + idx), vec, vl);
                    idx += vl;
                    لمبائی -= vl;
                } ورنہ {
                    GGML_ABORT("fatal error");
                }
            }
        }
    }
}

سانچہ <قسم_نام T> خالی forward_get_صفیں(ggml_compute_params * params, ggml_tensor * op) {
    ثابت ggml_tensor * src0 = op->src[0];
    ثابت ggml_tensor * src1 = op->src[1];
    ggml_tensor *       dst  = op;

    GGML_TENSOR_ثنائی_OP_LOCALS

    ثابت int64_t nc = ne00;
    ثابت int64_t nr = ggml_nelements(src1);

    أكد(ne0 == nc);
    أكد(ne02 == ne11);
    أكد(nb00 == حجم(اعشاری));
    أكد(ggml_nrows(op) == nr);

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    صحیح صفیں_nth = nth;
    صحیح ستون_تعداد_nth = 1;

    اگر (nr == 1) {
        صفیں_nth = 1;
        ستون_تعداد_nth = nth;
    }

    // rows per thread
    ثابت صحیح dr = (nr + صفیں_nth - 1) / صفیں_nth;
    ثابت صحیح dc = (nc + ستون_تعداد_nth - 1) / ستون_تعداد_nth;

    صحیح صفیں_ith = ith % صفیں_nth;
    صحیح ستون_تعداد_ith = ith % ستون_تعداد_nth;

    // row range for this thread
    ثابت صحیح ir0 = dr * صفیں_ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    ثابت صحیح cr0 = dc * ستون_تعداد_ith;
    ثابت صحیح cr1 = MIN(cr0 + dc, nc);

    برائے (int64_t i = ir0; i < ir1; ++i) {
        ثابت int64_t i12 = i / (ne11 * ne10);
        ثابت int64_t i11 = (i - i12 * ne11 * ne10) / ne10;
        ثابت int64_t i10 = (i - i12 * ne11 * ne10 - i11 * ne10);
        ثابت int64_t i01 = *(int32_t *) ((حرف *) src1->ڈیٹا + i10 * nb10 + i11 * nb11 + i12 * nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        memcpy1d(((حرف *) dst->ڈیٹا + i10 * nb1 + i11 * nb2 + i12 * nb3) + cr0 * حجم(T),
                 ((حرف *) src0->ڈیٹا + i01 * nb01 + i11 * nb02 + i12 * nb03) + cr0 * حجم(T),
                 (cr1 - cr0) * حجم(T));
    }
}

سانچہ <قسم_نام T> خالی forward_concat(ggml_compute_params * params, ggml_tensor * op) {
    ثابت ggml_tensor * src0 = op->src[0];
    ثابت ggml_tensor * src1 = op->src[1];
    ggml_tensor *       dst  = op;

    GGML_ASSERT(ggml_type_سائز(src0->type) == حجم(اعشاری));

    GGML_TENSOR_ثنائی_OP_LOCALS

    ثابت int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim == 0 && nb0 == حجم(اعشاری) && nb1 == حجم(اعشاری) * (ne00 + ne10));

    ثابت int64_t nr = ggml_nrows(dst);
    ثابت int64_t nc = ne0;

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    صحیح صفیں_nth = nth;
    صحیح ستون_تعداد_nth = 1;

    اگر (nr == 1) {
        صفیں_nth = 1;
        ستون_تعداد_nth = nth;
    }

    ثابت صحیح dr = (nr + صفیں_nth - 1) / صفیں_nth;
    ثابت صحیح dc = (nc + ستون_تعداد_nth - 1) / ستون_تعداد_nth;

    صحیح صفیں_ith = ith % صفیں_nth;
    صحیح ستون_تعداد_ith = ith % ستون_تعداد_nth;

    // row range for this thread
    ثابت صحیح ir0 = dr * صفیں_ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    ثابت صحیح cr0 = dc * ستون_تعداد_ith;
    ثابت صحیح cr1 = MIN(cr0 + dc, nc);

    int64_t o[4] = { 0, 0, 0, 0 };
    o[dim]       = src0->ne[dim];
    ثابت اعشاری * x;

    برائے (int64_t i = ir0; i < ir1; ++i) {
        ثابت int64_t i3 = i / (ne02 * ne01);
        ثابت int64_t i2 = (i - i3 * ne02 * ne01) / ne01;
        ثابت int64_t i1 = (i - i3 * ne02 * ne01 - i2 * ne01);

        برائے (صحیح i0 = cr0; i0 < cr1; i0++) {
            اگر (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                x = (ثابت اعشاری *) ((ثابت حرف *) src0->ڈیٹا + (i0) *nb00 + (i1) *nb01 + (i2) *nb02 + (i3) *nb03);
            } ورنہ {
                x = (ثابت اعشاری *) ((ثابت حرف *) src1->ڈیٹا + (i0 - o[0]) * nb10 + (i1 - o[1]) * nb11 +
                                     (i2 - o[2]) * nb12 + (i3 - o[3]) * nb13);
            }

            اعشاری * y = (اعشاری *) ((حرف *) dst->ڈیٹا + i0 * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3);

            *y = *x;
        }
    }
}

سانچہ خالی forward_ثنائی<GGML_OP_جوڑو, اعشاری>(ggml_compute_params * params, ggml_tensor * op);
سانچہ خالی forward_ثنائی<GGML_OP_SUB, اعشاری>(ggml_compute_params * params, ggml_tensor * op);
سانچہ خالی forward_ثنائی<GGML_OP_MUL, اعشاری>(ggml_compute_params * params, ggml_tensor * op);
سانچہ خالی forward_ثنائی<GGML_OP_DIV, اعشاری>(ggml_compute_params * params, ggml_tensor * op);
سانچہ خالی forward_ثنائی<GGML_OP_جوڑو, _Float16>(ggml_compute_params * params, ggml_tensor * op);
سانچہ خالی forward_ثنائی<GGML_OP_SUB, _Float16>(ggml_compute_params * params, ggml_tensor * op);
سانچہ خالی forward_ثنائی<GGML_OP_MUL, _Float16>(ggml_compute_params * params, ggml_tensor * op);
سانچہ خالی forward_ثنائی<GGML_OP_DIV, _Float16>(ggml_compute_params * params, ggml_tensor * op);
سانچہ خالی forward_مجموع_صفیں<اعشاری>(ثابت ggml_compute_params * params, ggml_tensor * op);
سانچہ خالی forward_مجموع_صفیں<_Float16>(ثابت ggml_compute_params * params, ggml_tensor * op);
سانچہ خالی forward_repeat_nrows<int32_t>(ggml_compute_params * params, ggml_tensor * op);
سانچہ خالی forward_repeat_nrows<int16_t>(ggml_compute_params * params, ggml_tensor * op);
سانچہ خالی forward_repeat_dim1<int32_t>(ggml_compute_params * params, ggml_tensor * op);
سانچہ خالی forward_repeat_dim1<int16_t>(ggml_compute_params * params, ggml_tensor * op);
سانچہ خالی forward_get_صفیں<int32_t>(ggml_compute_params * params, ggml_tensor * op);
سانچہ خالی forward_get_صفیں<int16_t>(ggml_compute_params * params, ggml_tensor * op);
سانچہ خالی forward_concat<int32_t>(ggml_compute_params * params, ggml_tensor * op);
سانچہ خالی forward_concat<int16_t>(ggml_compute_params * params, ggml_tensor * op);

}  // namespace spacemit_kernels::rvv
