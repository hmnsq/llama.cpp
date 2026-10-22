#అన్తర్భావ "rvv_kernels.h"

#అన్తర్భావ "common.h"
#అన్తర్భావ "ggml.h"
#అన్తర్భావ "ops.h"
#అన్తర్భావ "string.h"

#అన్తర్భావ <algorithm>
#అన్తర్భావ <cmath>
#అన్తర్భావ <cstdint>
#అన్తర్భావ <stdexcept>

#ఒకవేళ !defined(__riscv_v) || !defined(__riscv_v_intrinsic)
#    error "riscv v extension or v_intrinsic not enabled"
#లేకపోతే
#    include <riscv_vector.h>
#అయితే_ముగింపు

#ఒకవేళ !defined(__riscv_zfh)
#    error "riscv zfh extension not enabled"
#అయితే_ముగింపు

#ఒకవేళ defined(__GNUC__)
#    pragma GCC diagnostic ignored "-Woverlength-strings"
#    pragma GCC diagnostic ignored "-Wcast-qual"
#    pragma GCC diagnostic ignored "-Wunused-parameter"
#అయితే_ముగింపు

నామాకాశ spacemit_kernels::rvv {

నామాకాశ {

స్వయంచాలక align_పైకి(size_t value, size_t alignment) {
    తిరిగి (value + alignment - 1) / alignment * alignment;
}

స్థిర అంతఃవరుస తార్కిక flash_attn_ext_supported_d_vlen1024_vf16(int64_t d) {
    తిరిగి d > 0 && d <= 128;
}

స్థిర అంతఃవరుస తార్కిక flash_attn_ext_supported_shape_vlen1024_vf16(int64_t DK, int64_t DV) {
    తిరిగి flash_attn_ext_supported_d_vlen1024_vf16(DK) && flash_attn_ext_supported_d_vlen1024_vf16(DV);
}

స్థిర అంతఃవరుస దశాంశం reduce_కూడిక_f32m4_vlen1024(vfloat32m4_t v, size_t vl) {
    vfloat32m1_t s_v = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    s_v              = __riscv_vfredusum_vs_f32m4_f32m1(v, s_v, vl);
    తిరిగి __riscv_vfmv_f_s_f32m1_f32(s_v);
}

స్థిర అంతఃవరుస దశాంశం reduce_కూడిక_f32m2_vlen1024(vfloat32m2_t v, size_t vl) {
    vfloat32m1_t s_v = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    s_v              = __riscv_vfredusum_vs_f32m2_f32m1(v, s_v, vl);
    తిరిగి __riscv_vfmv_f_s_f32m1_f32(s_v);
}

// Adapted from ggml_v_expf_m2 in vec.h. This is accurate enough for softmax.
స్థిర అంతఃవరుస vfloat32m2_t rvv_expf_approx_f32m2(vfloat32m2_t x, size_t vl) {
    స్థిరరాశి vfloat32m2_t r = __riscv_vfmv_v_f_f32m2(0x1.8p23f, vl);
    స్థిరరాశి vfloat32m2_t z = __riscv_vfmacc_vf_f32m2(r, 0x1.715476p+0f, x, vl);
    స్థిరరాశి vfloat32m2_t n = __riscv_vfsub_vv_f32m2(z, r, vl);
    స్థిరరాశి vfloat32m2_t b =
        __riscv_vfnmsac_vf_f32m2(__riscv_vfnmsac_vf_f32m2(x, 0x1.62e4p-1f, n, vl), 0x1.7f7d1cp-20f, n, vl);
    స్థిరరాశి vuint32m2_t  e = __riscv_vsll_vx_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(z), 23, vl);
    స్థిరరాశి vfloat32m2_t k = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vadd_vx_u32m2(e, 0x3f800000, vl));
    స్థిరరాశి vbool16_t    c = __riscv_vmfgt_vf_f32m2_b16(__riscv_vfabs_v_f32m2(n, vl), 126.0f, vl);
    స్థిరరాశి vfloat32m2_t u = __riscv_vfmul_vv_f32m2(b, b, vl);
    స్థిరరాశి vfloat32m2_t j = __riscv_vfmacc_vv_f32m2(
        __riscv_vfmul_vf_f32m2(b, 0x1.ffffecp-1f, vl),
        __riscv_vfmacc_vv_f32m2(
            __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(0x1.fffdb6p-2f, vl), 0x1.555e66p-3f, b, vl),
            __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(0x1.573e2ep-5f, vl), 0x1.0e4020p-7f, b, vl), u, vl),
        u, vl);

    ఒకవేళ (!__riscv_vcpop_m_b16(c, vl)) {
        తిరిగి __riscv_vfmacc_vv_f32m2(k, j, k, vl);
    }

    స్థిరరాశి vbool16_t    dm = __riscv_vmfle_vf_f32m2_b16(n, 0.0f, vl);
    స్థిరరాశి vuint32m2_t  d  = __riscv_vmerge_vxm_u32m2(__riscv_vmv_v_x_u32m2(0, vl), 0x82000000, dm, vl);
    స్థిరరాశి vfloat32m2_t s1 = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vadd_vx_u32m2(d, 0x7f000000, vl));
    స్థిరరాశి vfloat32m2_t s2 = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vsub_vv_u32m2(e, d, vl));
    స్థిరరాశి vfloat32m2_t r1 =
        __riscv_vmerge_vvm_f32m2(__riscv_vfmacc_vv_f32m2(k, k, j, vl),
                                 __riscv_vfmul_vv_f32m2(__riscv_vfmacc_vv_f32m2(s2, s2, j, vl), s1, vl), c, vl);
    తిరిగి __riscv_vmerge_vvm_f32m2(r1, __riscv_vfmul_vv_f32m2(s1, s1, vl),
                                    __riscv_vmfgt_vf_f32m2_b16(__riscv_vfabs_v_f32m2(n, vl), 192.0f, vl), vl);
}

స్థిర అంతఃవరుస vfloat32m2_t rvv_tanh_approx_f32m2(vfloat32m2_t x, size_t vl) {
    స్థిరరాశి vfloat32m2_t abs_x       = __riscv_vfabs_v_f32m2(x, vl);
    స్థిరరాశి vfloat32m2_t neg_2_abs   = __riscv_vfmul_vf_f32m2(abs_x, -2.0f, vl);
    స్థిరరాశి vfloat32m2_t exp_పదం    = rvv_expf_approx_f32m2(neg_2_abs, vl);
    స్థిరరాశి vfloat32m2_t numerator   = __riscv_vfsub_vf_f32m2(exp_పదం, 1.0f, vl);
    స్థిరరాశి vfloat32m2_t denominator = __riscv_vfadd_vf_f32m2(exp_పదం, 1.0f, vl);
    స్థిరరాశి vfloat32m2_t tanh_abs    = __riscv_vfneg_v_f32m2(__riscv_vfdiv_vv_f32m2(numerator, denominator, vl), vl);
    స్థిరరాశి vbool16_t    neg_ముసుగు    = __riscv_vmflt_vf_f32m2_b16(x, 0.0f, vl);
    స్థిరరాశి vfloat32m2_t tanh_neg    = __riscv_vfneg_v_f32m2(tanh_abs, vl);
    తిరిగి __riscv_vmerge_vvm_f32m2(tanh_abs, tanh_neg, neg_ముసుగు, vl);
}

స్థిర శూన్య rvv_softcap_tanh_inplace_f32(దశాంశం * dst, int64_t dst_stride, int64_t tile_అడ్డువరుసలు, int64_t n, దశాంశం softcap) {
    కొరకు (పూర్ణాంకం tq = 0; tq < tile_అడ్డువరుసలు; ++tq, dst += dst_stride) {
        దశాంశం * dst_అడ్డువరుస   = dst;
        int64_t మిగిలిన = n;
        అయితే (మిగిలిన > 0) {
            స్థిరరాశి size_t vl = __riscv_vsetvl_e32m2(మిగిలిన);
            vfloat32m2_t v  = __riscv_vle32_v_f32m2(dst_అడ్డువరుస, vl);
            v               = rvv_tanh_approx_f32m2(v, vl);
            v               = __riscv_vfmul_vf_f32m2(v, softcap, vl);
            __riscv_vse32_v_f32m2(dst_అడ్డువరుస, v, vl);
            dst_అడ్డువరుస += vl;
            మిగిలిన -= vl;
        }
    }
}

స్థిర అంతఃవరుస దశాంశం rvv_softmax_exp_inplace_f32(దశాంశం * dst, int64_t n, దశాంశం max_విలువ) {
    దశాంశం అడ్డువరుస_కూడిక = 0.0f;
    అయితే (n > 0) {
        స్థిరరాశి size_t vl = __riscv_vsetvl_e32m2(n);
        vfloat32m2_t v  = __riscv_vle32_v_f32m2(dst, vl);
        v               = __riscv_vfsub_vf_f32m2(v, max_విలువ, vl);
        v               = rvv_expf_approx_f32m2(v, vl);
        __riscv_vse32_v_f32m2(dst, v, vl);
        అడ్డువరుస_కూడిక += reduce_కూడిక_f32m2_vlen1024(v, vl);
        dst += vl;
        n -= vl;
    }
    తిరిగి అడ్డువరుస_కూడిక;
}

స్థిర అంతఃవరుస దశాంశం rvv_కూడు_max_inplace_f32(దశాంశం * dst, స్థిరరాశి దశాంశం * src, int64_t n) {
    దశాంశం max_val = -INFINITY;
    అయితే (n > 0) {
        స్థిరరాశి size_t vl   = __riscv_vsetvl_e32m4(n);
        vfloat32m4_t vdst = __riscv_vle32_v_f32m4(dst, vl);
        vfloat32m4_t vsrc = __riscv_vle32_v_f32m4(src, vl);
        vdst              = __riscv_vfadd_vv_f32m4(vdst, vsrc, vl);
        __riscv_vse32_v_f32m4(dst, vdst, vl);

        vfloat32m1_t విత్తనం = __riscv_vfmv_v_f_f32m1(max_val, 1);
        విత్తనం              = __riscv_vfredmax_vs_f32m4_f32m1(vdst, విత్తనం, vl);
        max_val           = __riscv_vfmv_f_s_f32m1_f32(విత్తనం);

        dst += vl;
        src += vl;
        n -= vl;
    }
    తిరిగి max_val;
}

స్థిర అంతఃవరుస దశాంశం rvv_softcap_కూడు_max_inplace_f32(దశాంశం * dst, స్థిరరాశి దశాంశం * src, int64_t n, దశాంశం softcap) {
    ఒకవేళ (softcap == 0.0f) {
        తిరిగి rvv_కూడు_max_inplace_f32(dst, src, n);
    }

    దశాంశం max_val = -INFINITY;
    అయితే (n > 0) {
        స్థిరరాశి size_t vl   = __riscv_vsetvl_e32m2(n);
        vfloat32m2_t vdst = __riscv_vle32_v_f32m2(dst, vl);
        vfloat32m2_t vsrc = __riscv_vle32_v_f32m2(src, vl);
        vdst              = rvv_tanh_approx_f32m2(vdst, vl);
        vdst              = __riscv_vfmul_vf_f32m2(vdst, softcap, vl);
        vdst              = __riscv_vfadd_vv_f32m2(vdst, vsrc, vl);
        __riscv_vse32_v_f32m2(dst, vdst, vl);

        vfloat32m1_t విత్తనం = __riscv_vfmv_v_f_f32m1(max_val, 1);
        విత్తనం              = __riscv_vfredmax_vs_f32m2_f32m1(vdst, విత్తనం, vl);
        max_val           = __riscv_vfmv_f_s_f32m1_f32(విత్తనం);

        dst += vl;
        src += vl;
        n -= vl;
    }
    తిరిగి max_val;
}

స్థిర అంతఃవరుస శూన్య rvv_zero_f32(దశాంశం * dst, int64_t n) {
    అయితే (n > 0) {
        స్థిరరాశి size_t       vl = __riscv_vsetvl_e32m4(n);
        స్థిరరాశి vfloat32m4_t z  = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        __riscv_vse32_v_f32m4(dst, z, vl);
        dst += vl;
        n -= vl;
    }
}

స్థిర అంతఃవరుస శూన్య rvv_scale_f32(దశాంశం * dst, దశాంశం scale, int64_t n) {
    అయితే (n > 0) {
        స్థిరరాశి size_t vl = __riscv_vsetvl_e32m4(n);
        vfloat32m4_t v  = __riscv_vle32_v_f32m4(dst, vl);
        v               = __riscv_vfmul_vf_f32m4(v, scale, vl);
        __riscv_vse32_v_f32m4(dst, v, vl);
        dst += vl;
        n -= vl;
    }
}

స్థిర అంతఃవరుస శూన్య rvv_కూడు_inplace_f32(దశాంశం *       dst,
                                       int64_t       dst_stride,
                                       స్థిరరాశి దశాంశం * src,
                                       int64_t       src_stride,
                                       int64_t       tile_అడ్డువరుసలు,
                                       int64_t       n) {
    కొరకు (పూర్ణాంకం tq = 0; tq < tile_అడ్డువరుసలు; ++tq, dst += dst_stride, src += src_stride) {
        int64_t       మిగిలిన = n;
        దశాంశం *       dst_అడ్డువరుస   = dst;
        స్థిరరాశి దశాంశం * src_అడ్డువరుస   = src;
        అయితే (మిగిలిన > 0) {
            స్థిరరాశి size_t vl   = __riscv_vsetvl_e32m4(మిగిలిన);
            vfloat32m4_t vdst = __riscv_vle32_v_f32m4(dst_అడ్డువరుస, vl);
            vfloat32m4_t vsrc = __riscv_vle32_v_f32m4(src_అడ్డువరుస, vl);
            vdst              = __riscv_vfadd_vv_f32m4(vdst, vsrc, vl);
            __riscv_vse32_v_f32m4(dst_అడ్డువరుస, vdst, vl);
            dst_అడ్డువరుస += vl;
            src_అడ్డువరుస += vl;
            మిగిలిన -= vl;
        }
    }
}

స్థిర అంతఃవరుస దశాంశం rvv_max_f32(స్థిరరాశి దశాంశం * src, int64_t n) {
    దశాంశం max_val = -INFINITY;
    అయితే (n > 0) {
        స్థిరరాశి size_t       vl   = __riscv_vsetvl_e32m4(n);
        స్థిరరాశి vfloat32m4_t v    = __riscv_vle32_v_f32m4(src, vl);
        vfloat32m1_t       విత్తనం = __riscv_vfmv_v_f_f32m1(max_val, 1);
        విత్తనం                    = __riscv_vfredmax_vs_f32m4_f32m1(v, విత్తనం, vl);
        max_val                 = __riscv_vfmv_f_s_f32m1_f32(విత్తనం);
        src += vl;
        n -= vl;
    }
    తిరిగి max_val;
}

స్థిర శూన్య rvv_pack_f32_as_scaled_f16(శూన్య *       dst,
                                       int64_t      dst_అడ్డువరుస_stride,
                                       స్థిరరాశి శూన్య * src,
                                       int64_t      src_అడ్డువరుస_stride,
                                       int64_t      tile_అడ్డువరుసలు,
                                       int64_t      n,
                                       దశాంశం        scale) {
    కొరకు (పూర్ణాంకం tq = 0; tq < tile_అడ్డువరుసలు; ++tq) {
        స్థిరరాశి దశాంశం * అడ్డువరుస_ptr     = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src + tq * src_అడ్డువరుస_stride);
        _Float16 *    dst_అడ్డువరుస_ptr = (_Float16 *) ((అక్షరం *) dst + tq * dst_అడ్డువరుస_stride);
        int64_t       మిగిలిన   = n;
        అయితే (మిగిలిన > 0) {
            స్థిరరాశి size_t vl        = __riscv_vsetvl_e32m4(మిగిలిన);
            vfloat32m4_t v32       = __riscv_vle32_v_f32m4(అడ్డువరుస_ptr, vl);
            v32                    = __riscv_vfmul_vf_f32m4(v32, scale, vl);
            స్థిరరాశి vfloat16m2_t v16 = __riscv_vfncvt_f_f_w_f16m2(v32, vl);
            __riscv_vse16_v_f16m2(dst_అడ్డువరుస_ptr, v16, vl);
            dst_అడ్డువరుస_ptr += vl;
            అడ్డువరుస_ptr += vl;
            మిగిలిన -= vl;
        }
    }
}

స్థిర శూన్య rvv_pack_scaled_f16_as_f32(శూన్య *       dst,
                                       int64_t      dst_అడ్డువరుస_stride,
                                       స్థిరరాశి శూన్య * src,
                                       int64_t      src_అడ్డువరుస_stride,
                                       int64_t      tile_అడ్డువరుసలు,
                                       int64_t      n,
                                       దశాంశం        scale) {
    కొరకు (పూర్ణాంకం tq = 0; tq < tile_అడ్డువరుసలు; ++tq) {
        స్థిరరాశి _Float16 * అడ్డువరుస_ptr     = (స్థిరరాశి _Float16 *) ((స్థిరరాశి అక్షరం *) src + tq * src_అడ్డువరుస_stride);
        దశాంశం *          dst_అడ్డువరుస_ptr = (దశాంశం *) ((అక్షరం *) dst + tq * dst_అడ్డువరుస_stride);
        int64_t          మిగిలిన   = n;
        అయితే (మిగిలిన > 0) {
            స్థిరరాశి size_t       vl  = __riscv_vsetvl_e16m2(మిగిలిన);
            స్థిరరాశి vfloat16m2_t v16 = __riscv_vle16_v_f16m2(అడ్డువరుస_ptr, vl);
            vfloat32m4_t       v32 = __riscv_vfwcvt_f_f_v_f32m4(v16, vl);
            v32                    = __riscv_vfmul_vf_f32m4(v32, scale, vl);
            __riscv_vse32_v_f32m4(dst_అడ్డువరుస_ptr, v32, vl);
            dst_అడ్డువరుస_ptr += vl;
            అడ్డువరుస_ptr += vl;
            మిగిలిన -= vl;
        }
    }
}

స్థిర శూన్య rvv_pack_scaled_f32_as_f32(శూన్య *       dst,
                                       int64_t      dst_అడ్డువరుస_stride,
                                       స్థిరరాశి శూన్య * src,
                                       int64_t      src_అడ్డువరుస_stride,
                                       int64_t      tile_అడ్డువరుసలు,
                                       int64_t      n,
                                       దశాంశం *      scale) {
    కొరకు (పూర్ణాంకం tq = 0; tq < tile_అడ్డువరుసలు; ++tq) {
        స్థిరరాశి దశాంశం * అడ్డువరుస_ptr     = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src + tq * src_అడ్డువరుస_stride);
        దశాంశం *       dst_అడ్డువరుస_ptr = (దశాంశం *) ((అక్షరం *) dst + tq * dst_అడ్డువరుస_stride);
        int64_t       మిగిలిన   = n;
        అయితే (మిగిలిన > 0) {
            స్థిరరాశి size_t vl  = __riscv_vsetvl_e32m4(మిగిలిన);
            vfloat32m4_t v32 = __riscv_vle32_v_f32m4(అడ్డువరుస_ptr, vl);
            v32              = __riscv_vfmul_vf_f32m4(v32, scale[tq], vl);
            __riscv_vse32_v_f32m4(dst_అడ్డువరుస_ptr, v32, vl);
            dst_అడ్డువరుస_ptr += vl;
            అడ్డువరుస_ptr += vl;
            మిగిలిన -= vl;
        }
    }
}

స్థిర అంతఃవరుస శూన్య rvv_transposed_s32_mn_గమ్యం_nm(int8_t * dst,
                                               int64_t  n_dst_stride,
                                               int8_t * src,
                                               int64_t  m_src_stride,
                                               int64_t  m,
                                               int64_t  n) {
    int8_t * లోపల  = src;
    int8_t * బహిర్గతం = dst;

    __asm__ అస్థిర(
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

        : [src] "+r"(లోపల), [dst] "+r"(బహిర్గతం), [isz0] "+r"(m)
        : [isz1] "r"(n), [is0] "r"(m_src_stride), [os0] "r"(n_dst_stride)
        : "cc", "t0", "t1", "t2", "t3", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "a1");
}

స్థిర అంతఃవరుస శూన్య rvv_transposed_s16_mn_గమ్యం_nm(int8_t * dst,
                                               int64_t  n_dst_stride,
                                               int8_t * src,
                                               int64_t  m_src_stride,
                                               int64_t  m,
                                               int64_t  n) {
    int8_t * లోపల  = src;
    int8_t * బహిర్గతం = dst;

    __asm__ అస్థిర(
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

        : [src] "+r"(లోపల), [dst] "+r"(బహిర్గతం), [isz0] "+r"(m)
        : [isz1] "r"(n), [is0] "r"(m_src_stride), [os0] "r"(n_dst_stride)
        : "cc", "t0", "t1", "t2", "t3", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "a1");
}

స్థిర అంతఃవరుస శూన్య rvv_qk_dot_tile_f16_x1(దశాంశం *          dst,
                                          స్థిరరాశి _Float16 * q_అడ్డువరుస,
                                          స్థిరరాశి _Float16 * k_pack,
                                          int64_t          dk,
                                          int64_t          kv_tile) {
    స్థిరరాశి size_t vl  = __riscv_vsetvl_e16m1(kv_tile);
    vfloat32m2_t acc = __riscv_vfmv_v_f_f32m2(0.0f, vl);

    కొరకు (int64_t d = 0; d < dk; ++d) {
        స్థిరరాశి vfloat16m1_t k_vec = __riscv_vle16_v_f16m1(k_pack + d * ggml_fa_tile_config::KV, vl);
        acc                      = __riscv_vfwmacc_vf_f32m2(acc, q_అడ్డువరుస[d], k_vec, vl);
    }

    __riscv_vse32_v_f32m2(dst, acc, vl);
}

స్థిర అంతఃవరుస శూన్య rvv_qk_dot_tile_f16_x4(దశాంశం *          dst0,
                                          దశాంశం *          dst1,
                                          దశాంశం *          dst2,
                                          దశాంశం *          dst3,
                                          స్థిరరాశి _Float16 * q0,
                                          స్థిరరాశి _Float16 * q1,
                                          స్థిరరాశి _Float16 * q2,
                                          స్థిరరాశి _Float16 * q3,
                                          స్థిరరాశి _Float16 * k_pack,
                                          int64_t          dk,
                                          int64_t          kv_tile) {
    స్థిరరాశి size_t vl   = __riscv_vsetvl_e16m1(kv_tile);
    vfloat32m2_t acc0 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    vfloat32m2_t acc1 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    vfloat32m2_t acc2 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    vfloat32m2_t acc3 = __riscv_vfmv_v_f_f32m2(0.0f, vl);

    కొరకు (int64_t d = 0; d < dk; ++d) {
        స్థిరరాశి vfloat16m1_t k_vec = __riscv_vle16_v_f16m1(k_pack + d * ggml_fa_tile_config::KV, vl);
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

స్థిర అంతఃవరుస శూన్య rvv_pv_accumulate_f16_x1(దశాంశం *          dst,
                                            స్థిరరాశి దశాంశం *    prob,
                                            స్థిరరాశి _Float16 * v_pack,
                                            int64_t          kv_tile,
                                            int64_t          dv) {
    int64_t d_ఎడమ = dv;
    int64_t d_off  = 0;

    అయితే (d_ఎడమ > 0) {
        స్థిరరాశి size_t vl  = __riscv_vsetvl_e16m2(d_ఎడమ);
        vfloat32m4_t acc = __riscv_vle32_v_f32m4(dst + d_off, vl);

        కొరకు (int64_t tk = 0; tk < kv_tile; ++tk) {
            స్థిరరాశి vfloat16m2_t v16 = __riscv_vle16_v_f16m2(v_pack + tk * dv + d_off, vl);
            స్థిరరాశి vfloat32m4_t v32 = __riscv_vfwcvt_f_f_v_f32m4(v16, vl);
            acc                    = __riscv_vfmacc_vf_f32m4(acc, prob[tk], v32, vl);
        }

        __riscv_vse32_v_f32m4(dst + d_off, acc, vl);
        d_ఎడమ -= vl;
        d_off += vl;
    }
}

స్థిర అంతఃవరుస శూన్య rvv_pv_accumulate_f16_x4(దశాంశం *          dst0,
                                            దశాంశం *          dst1,
                                            దశాంశం *          dst2,
                                            దశాంశం *          dst3,
                                            స్థిరరాశి దశాంశం *    prob0,
                                            స్థిరరాశి దశాంశం *    prob1,
                                            స్థిరరాశి దశాంశం *    prob2,
                                            స్థిరరాశి దశాంశం *    prob3,
                                            స్థిరరాశి _Float16 * v_pack,
                                            int64_t          kv_tile,
                                            int64_t          dv) {
    int64_t d_ఎడమ = dv;
    int64_t d_off  = 0;

    అయితే (d_ఎడమ > 0) {
        స్థిరరాశి size_t vl   = __riscv_vsetvl_e16m2(d_ఎడమ);
        vfloat32m4_t acc0 = __riscv_vle32_v_f32m4(dst0 + d_off, vl);
        vfloat32m4_t acc1 = __riscv_vle32_v_f32m4(dst1 + d_off, vl);
        vfloat32m4_t acc2 = __riscv_vle32_v_f32m4(dst2 + d_off, vl);
        vfloat32m4_t acc3 = __riscv_vle32_v_f32m4(dst3 + d_off, vl);

        కొరకు (int64_t tk = 0; tk < kv_tile; ++tk) {
            స్థిరరాశి vfloat16m2_t v16 = __riscv_vle16_v_f16m2(v_pack + tk * dv + d_off, vl);
            స్థిరరాశి vfloat32m4_t v32 = __riscv_vfwcvt_f_f_v_f32m4(v16, vl);
            acc0                   = __riscv_vfmacc_vf_f32m4(acc0, prob0[tk], v32, vl);
            acc1                   = __riscv_vfmacc_vf_f32m4(acc1, prob1[tk], v32, vl);
            acc2                   = __riscv_vfmacc_vf_f32m4(acc2, prob2[tk], v32, vl);
            acc3                   = __riscv_vfmacc_vf_f32m4(acc3, prob3[tk], v32, vl);
        }

        __riscv_vse32_v_f32m4(dst0 + d_off, acc0, vl);
        __riscv_vse32_v_f32m4(dst1 + d_off, acc1, vl);
        __riscv_vse32_v_f32m4(dst2 + d_off, acc2, vl);
        __riscv_vse32_v_f32m4(dst3 + d_off, acc3, vl);
        d_ఎడమ -= vl;
        d_off += vl;
    }
}

స్థిర అంతఃవరుస శూన్య rvv_qk_dot_tile(దశాంశం *       dst,
                                   స్థిరరాశి దశాంశం * q_అడ్డువరుస,
                                   స్థిరరాశి దశాంశం * k_pack,
                                   int64_t       dk,
                                   int64_t       kv_tile,
                                   దశాంశం         scale) {
    స్థిరరాశి size_t vl  = __riscv_vsetvl_e32m4(kv_tile);
    vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);

    కొరకు (int64_t d = 0; d < dk; ++d) {
        స్థిరరాశి vfloat32m4_t k_vec = __riscv_vle32_v_f32m4(k_pack + d * kv_tile, vl);
        acc                      = __riscv_vfmacc_vf_f32m4(acc, q_అడ్డువరుస[d] * scale, k_vec, vl);
    }

    __riscv_vse32_v_f32m4(dst, acc, vl);
}

స్థిర అంతఃవరుస శూన్య rvv_pv_accumulate(దశాంశం *       dst,
                                     స్థిరరాశి దశాంశం * prob,
                                     స్థిరరాశి దశాంశం * v_pack,
                                     int64_t       kv_tile,
                                     int64_t       dv) {
    int64_t d_ఎడమ = dv;
    int64_t d_off  = 0;

    అయితే (d_ఎడమ > 0) {
        స్థిరరాశి size_t vl  = __riscv_vsetvl_e32m4(d_ఎడమ);
        vfloat32m4_t acc = __riscv_vle32_v_f32m4(dst + d_off, vl);

        కొరకు (int64_t tk = 0; tk < kv_tile; ++tk) {
            స్థిరరాశి vfloat32m4_t v_vec = __riscv_vle32_v_f32m4(v_pack + tk * dv + d_off, vl);
            acc                      = __riscv_vfmacc_vf_f32m4(acc, prob[tk], v_vec, vl);
        }

        __riscv_vse32_v_f32m4(dst + d_off, acc, vl);
        d_ఎడమ -= vl;
        d_off += vl;
    }
}

స్థిర శూన్య క్రమమార్పు_transpose_impl(స్థిరరాశి ggml_tensor * src0,
                                   ggml_tensor *       dst,
                                   int64_t             batch,
                                   int64_t             m,
                                   int64_t             n,
                                   int64_t             batch_stride,
                                   int64_t             m_src_stride,
                                   int64_t             n_src_stride,
                                   int64_t             n_dst_stride,
                                   పూర్ణాంకం                 ith,
                                   పూర్ణాంకం                 nth) {
    GGML_ASSERT(n_src_stride == పరిమాణం(int32_t) || n_src_stride == పరిమాణం(int16_t));

    ఒకవేళ (n_src_stride == పరిమాణం(int32_t)) {
        కొరకు (int64_t bi = ith; bi < batch; bi += nth) {
            rvv_transposed_s32_mn_గమ్యం_nm((int8_t *) ((అక్షరం *) dst->దత్తాంశం + bi * batch_stride), n_dst_stride,
                                        (int8_t *) ((అక్షరం *) src0->దత్తాంశం + bi * batch_stride), m_src_stride, m, n);
        }
    } లేకపోతే ఒకవేళ (n_src_stride == పరిమాణం(int16_t)) {
        కొరకు (int64_t bi = ith; bi < batch; bi += nth) {
            rvv_transposed_s16_mn_గమ్యం_nm((int8_t *) ((అక్షరం *) dst->దత్తాంశం + bi * batch_stride), n_dst_stride,
                                        (int8_t *) ((అక్షరం *) src0->దత్తాంశం + bi * batch_stride), m_src_stride, m, n);
        }
    } లేకపోతే {
        GGML_ABORT("not implemented");
    }
}

మూస <size_t QLEN>
స్థిర శూన్య flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_mrow(దశాంశం **            pq,
                                                                  స్థిరరాశి అక్షరం *        k_దత్తాంశం_అడ్డువరుస,
                                                                  స్థిరరాశి అక్షరం *        v_దత్తాంశం_అడ్డువరుస,
                                                                  స్థిరరాశి ggml_fp16_t * mp,
                                                                  దశాంశం **            sinks,
                                                                  దశాంశం **            dst,
                                                                  దశాంశం               scale,
                                                                  దశాంశం               logit_softcap,
                                                                  దశాంశం               వాలు,
                                                                  int64_t             nek1,
                                                                  int64_t             nbk1,
                                                                  int64_t             nbv1,
                                                                  int64_t             DV,
                                                                  int64_t             DK,
                                                                  శూన్య *              tcm_బఫరు,
                                                                  size_t              tcm_బఫరు_కొలత) {
    GGML_ASSERT(flash_attn_ext_supported_shape_vlen1024_vf16(DK, DV));
    దశాంశం S[QLEN] = { 0.0f };       //కూడిక
    దశాంశం M[QLEN] = { -INFINITY };  // maximum KQ value

    _Float16 *   kq16_బఫరు          = (_Float16 *) tcm_బఫరు;
    _Float16 *   qv_బఫరు            = kq16_బఫరు + QLEN * DV;
    స్థిరరాశి size_t qkv_తాత్కాలిక_బఫరు_కొలత = (QLEN * DV + QLEN * DK) * పరిమాణం(_Float16);
    అక్షరం *       kv_tile_బఫరు       = (అక్షరం *) (qv_బఫరు + QLEN * DK);

    {
        vfloat16m2_t VKQ16_v = __riscv_vfmv_v_f_f16m2(0.0f, DV);
        కొరకు (int64_t i = 0; i < QLEN; ++i) {
            __riscv_vse16_v_f16m2(kq16_బఫరు + i * DV, VKQ16_v, DV);
            vfloat16m2_t Q_q_v = __riscv_vfncvt_f_f_w_f16m2(__riscv_vle32_v_f32m4(pq[i], DK), DK);
            __riscv_vse16_v_f16m2(qv_బఫరు + i * DK, Q_q_v, DK);
        }
    }

    స్థిరరాశి uintptr_t scratch_addr = reinterpret_cast<uintptr_t>(kv_tile_బఫరు);
    స్థిరరాశి size_t    scratch_కొలత = tcm_బఫరు_కొలత > qkv_తాత్కాలిక_బఫరు_కొలత ? tcm_బఫరు_కొలత - qkv_తాత్కాలిక_బఫరు_కొలత : 0;
    స్థిరరాశి uintptr_t kq_tile_addr = align_పైకి(scratch_addr, alignof(దశాంశం));
    స్థిరరాశి size_t    scratch_ఉపసర్గ = kq_tile_addr - scratch_addr;
    స్థిరరాశి size_t    packed_tile_కొలత =
        QLEN * పరిమాణం(దశాంశం) + DK * పరిమాణం(_Float16) + DV * పరిమాణం(_Float16) + పరిమాణం(దశాంశం);
    స్థిరరాశి int64_t max_ic_tile_అడుగు = ((int64_t) __riscv_vsetvlmax_e16m1()) & ~((int64_t) 7);
    స్థిరరాశి int64_t max_fit_by_tcm =
        scratch_కొలత > scratch_ఉపసర్గ ? (int64_t) ((scratch_కొలత - scratch_ఉపసర్గ) / packed_tile_కొలత) : 0;
    స్థిరరాశి int64_t ic_tile_అడుగు = std::min(max_ic_tile_అడుగు, max_fit_by_tcm) & ~((int64_t) 7);

    స్థిరరాశి uintptr_t k_tile_addr  = kq_tile_addr + QLEN * ic_tile_అడుగు * పరిమాణం(దశాంశం);
    స్థిరరాశి uintptr_t v_tile_addr  = k_tile_addr + DK * ic_tile_అడుగు * పరిమాణం(_Float16);
    స్థిరరాశి uintptr_t mv_tile_addr = v_tile_addr + ic_tile_అడుగు * DV * పరిమాణం(_Float16);

    ఒకవేళ (ic_tile_అడుగు >= 8) {
        దశాంశం *    kq_tile_బఫరు = reinterpret_cast<దశాంశం *>(kq_tile_addr);
        _Float16 * k_tile_pack    = reinterpret_cast<_Float16 *>(k_tile_addr);
        _Float16 * v_tile_pack    = reinterpret_cast<_Float16 *>(v_tile_addr);
        దశాంశం *    mv_tile_pack   = reinterpret_cast<దశాంశం *>(mv_tile_addr);

        స్థిరరాశి int64_t k_tile_byte_stride = ic_tile_అడుగు * (int64_t) పరిమాణం(_Float16);

        int64_t ic_అడుగు = 0;
        కొరకు (int64_t ic = 0; ic < nek1; ++ic) {
            స్థిరరాశి దశాంశం mv = mp ? వాలు * ((_Float16 *) mp)[ic] : 0.0f;

            ఒకవేళ (mv != -INFINITY) {
                స్థిరరాశి _Float16 * k_దత్తాంశం = (స్థిరరాశి _Float16 *) (k_దత్తాంశం_అడ్డువరుస + ic * nbk1);
                స్థిరరాశి _Float16 * v_దత్తాంశం = (స్థిరరాశి _Float16 *) (v_దత్తాంశం_అడ్డువరుస + ic * nbv1);

                స్థిరరాశి vfloat16m2_t k_దత్తాంశం_v = __riscv_vle16_v_f16m2(k_దత్తాంశం, DK);
                స్థిరరాశి vfloat16m2_t v_దత్తాంశం_v = __riscv_vle16_v_f16m2(v_దత్తాంశం, DV);
                __riscv_vsse16_v_f16m2(k_tile_pack + ic_అడుగు, k_tile_byte_stride, k_దత్తాంశం_v, DK);
                __riscv_vse16_v_f16m2(v_tile_pack + ic_అడుగు * DV, v_దత్తాంశం_v, DV);
                mv_tile_pack[ic_అడుగు] = mv;
                ic_అడుగు++;
            }

            ఒకవేళ (ic_అడుగు > 0 && (ic_అడుగు == ic_tile_అడుగు || ic == (nek1 - 1))) {
                ఒకవేళ constexpr (QLEN == 4) {
                    స్థిరరాశి size_t qk_vl   = __riscv_vsetvl_e16m1(ic_అడుగు);
                    vfloat32m2_t qk_acc0 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);
                    vfloat32m2_t qk_acc1 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);
                    vfloat32m2_t qk_acc2 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);
                    vfloat32m2_t qk_acc3 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);

                    కొరకు (int64_t d = 0; d < DK; ++d) {
                        స్థిరరాశి vfloat16m1_t k_vec = __riscv_vle16_v_f16m1(k_tile_pack + d * ic_tile_అడుగు, qk_vl);
                        qk_acc0 = __riscv_vfwmacc_vf_f32m2(qk_acc0, qv_బఫరు[0 * DK + d], k_vec, qk_vl);
                        qk_acc1 = __riscv_vfwmacc_vf_f32m2(qk_acc1, qv_బఫరు[1 * DK + d], k_vec, qk_vl);
                        qk_acc2 = __riscv_vfwmacc_vf_f32m2(qk_acc2, qv_బఫరు[2 * DK + d], k_vec, qk_vl);
                        qk_acc3 = __riscv_vfwmacc_vf_f32m2(qk_acc3, qv_బఫరు[3 * DK + d], k_vec, qk_vl);
                    }

                    qk_acc0 = __riscv_vfmul_vf_f32m2(qk_acc0, scale, qk_vl);
                    qk_acc1 = __riscv_vfmul_vf_f32m2(qk_acc1, scale, qk_vl);
                    qk_acc2 = __riscv_vfmul_vf_f32m2(qk_acc2, scale, qk_vl);
                    qk_acc3 = __riscv_vfmul_vf_f32m2(qk_acc3, scale, qk_vl);

                    __riscv_vse32_v_f32m2(kq_tile_బఫరు + 0 * ic_tile_అడుగు, qk_acc0, qk_vl);
                    __riscv_vse32_v_f32m2(kq_tile_బఫరు + 1 * ic_tile_అడుగు, qk_acc1, qk_vl);
                    __riscv_vse32_v_f32m2(kq_tile_బఫరు + 2 * ic_tile_అడుగు, qk_acc2, qk_vl);
                    __riscv_vse32_v_f32m2(kq_tile_బఫరు + 3 * ic_tile_అడుగు, qk_acc3, qk_vl);
                } లేకపోతే {
                    static_assert(QLEN == 2, "unsupported QLEN");

                    స్థిరరాశి size_t qk_vl   = __riscv_vsetvl_e16m1(ic_అడుగు);
                    vfloat32m2_t qk_acc0 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);
                    vfloat32m2_t qk_acc1 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);

                    కొరకు (int64_t d = 0; d < DK; ++d) {
                        స్థిరరాశి vfloat16m1_t k_vec = __riscv_vle16_v_f16m1(k_tile_pack + d * ic_tile_అడుగు, qk_vl);
                        qk_acc0 = __riscv_vfwmacc_vf_f32m2(qk_acc0, qv_బఫరు[0 * DK + d], k_vec, qk_vl);
                        qk_acc1 = __riscv_vfwmacc_vf_f32m2(qk_acc1, qv_బఫరు[1 * DK + d], k_vec, qk_vl);
                    }

                    qk_acc0 = __riscv_vfmul_vf_f32m2(qk_acc0, scale, qk_vl);
                    qk_acc1 = __riscv_vfmul_vf_f32m2(qk_acc1, scale, qk_vl);

                    __riscv_vse32_v_f32m2(kq_tile_బఫరు + 0 * ic_tile_అడుగు, qk_acc0, qk_vl);
                    __riscv_vse32_v_f32m2(kq_tile_బఫరు + 1 * ic_tile_అడుగు, qk_acc1, qk_vl);
                }

                కొరకు (పూర్ణాంకం i = 0; i < QLEN; ++i) {
                    దశాంశం *     అడ్డువరుస_ptr = kq_tile_బఫరు + i * ic_tile_అడుగు;
                    స్థిరరాశి దశాంశం tile_max =
                        rvv_softcap_కూడు_max_inplace_f32(అడ్డువరుస_ptr, mv_tile_pack, ic_అడుగు, logit_softcap);

                    స్థిరరాశి దశాంశం Mold = M[i];

                    ఒకవేళ (tile_max > Mold) {
                        స్థిరరాశి దశాంశం ms = expf(Mold - tile_max);
                        M[i]           = tile_max;
                        S[i] *= ms;

                        vfloat16m2_t VKQ16_v = __riscv_vle16_v_f16m2(kq16_బఫరు + i * DV, DV);
                        VKQ16_v              = __riscv_vfmul_vf_f16m2(VKQ16_v, (_Float16) ms, DV);
                        __riscv_vse16_v_f16m2(kq16_బఫరు + i * DV, VKQ16_v, DV);
                    }

                    S[i] += rvv_softmax_exp_inplace_f32(అడ్డువరుస_ptr, ic_అడుగు, M[i]);
                }

                ఒకవేళ constexpr (QLEN == 4) {
                    vfloat16m2_t pv_acc0 = __riscv_vle16_v_f16m2(kq16_బఫరు + 0 * DV, DV);
                    vfloat16m2_t pv_acc1 = __riscv_vle16_v_f16m2(kq16_బఫరు + 1 * DV, DV);
                    vfloat16m2_t pv_acc2 = __riscv_vle16_v_f16m2(kq16_బఫరు + 2 * DV, DV);
                    vfloat16m2_t pv_acc3 = __riscv_vle16_v_f16m2(kq16_బఫరు + 3 * DV, DV);

                    కొరకు (int64_t tk = 0; tk < ic_అడుగు; ++tk) {
                        స్థిరరాశి vfloat16m2_t v16 = __riscv_vle16_v_f16m2(v_tile_pack + tk * DV, DV);
                        pv_acc0 =
                            __riscv_vfmacc_vf_f16m2(pv_acc0, (_Float16) kq_tile_బఫరు[0 * ic_tile_అడుగు + tk], v16, DV);
                        pv_acc1 =
                            __riscv_vfmacc_vf_f16m2(pv_acc1, (_Float16) kq_tile_బఫరు[1 * ic_tile_అడుగు + tk], v16, DV);
                        pv_acc2 =
                            __riscv_vfmacc_vf_f16m2(pv_acc2, (_Float16) kq_tile_బఫరు[2 * ic_tile_అడుగు + tk], v16, DV);
                        pv_acc3 =
                            __riscv_vfmacc_vf_f16m2(pv_acc3, (_Float16) kq_tile_బఫరు[3 * ic_tile_అడుగు + tk], v16, DV);
                    }

                    __riscv_vse16_v_f16m2(kq16_బఫరు + 0 * DV, pv_acc0, DV);
                    __riscv_vse16_v_f16m2(kq16_బఫరు + 1 * DV, pv_acc1, DV);
                    __riscv_vse16_v_f16m2(kq16_బఫరు + 2 * DV, pv_acc2, DV);
                    __riscv_vse16_v_f16m2(kq16_బఫరు + 3 * DV, pv_acc3, DV);
                } లేకపోతే {
                    static_assert(QLEN == 2, "unsupported QLEN");
                    vfloat16m2_t pv_acc0 = __riscv_vle16_v_f16m2(kq16_బఫరు + 0 * DV, DV);
                    vfloat16m2_t pv_acc1 = __riscv_vle16_v_f16m2(kq16_బఫరు + 1 * DV, DV);

                    కొరకు (int64_t tk = 0; tk < ic_అడుగు; ++tk) {
                        స్థిరరాశి vfloat16m2_t v16 = __riscv_vle16_v_f16m2(v_tile_pack + tk * DV, DV);
                        pv_acc0 =
                            __riscv_vfmacc_vf_f16m2(pv_acc0, (_Float16) kq_tile_బఫరు[0 * ic_tile_అడుగు + tk], v16, DV);
                        pv_acc1 =
                            __riscv_vfmacc_vf_f16m2(pv_acc1, (_Float16) kq_tile_బఫరు[1 * ic_tile_అడుగు + tk], v16, DV);
                    }

                    __riscv_vse16_v_f16m2(kq16_బఫరు + 0 * DV, pv_acc0, DV);
                    __riscv_vse16_v_f16m2(kq16_బఫరు + 1 * DV, pv_acc1, DV);
                }

                ic_అడుగు = 0;
            }
        }
    } లేకపోతే {
        కొరకు (int64_t ic = 0; ic < nek1; ++ic) {
            స్థిరరాశి దశాంశం mv = mp ? వాలు * ((_Float16 *) mp)[ic] : 0.0f;

            స్థిరరాశి అక్షరం * k_దత్తాంశం = k_దత్తాంశం_అడ్డువరుస + ic * nbk1;
            స్థిరరాశి అక్షరం * v_దత్తాంశం = v_దత్తాంశం_అడ్డువరుస + ic * nbv1;

            vfloat16m2_t k_దత్తాంశం_v;
            vfloat16m2_t v_దత్తాంశం_v;

            ఒకవేళ (mv != -INFINITY) {
                k_దత్తాంశం_v = __riscv_vle16_v_f16m2((_Float16 *) k_దత్తాంశం, DK);
                v_దత్తాంశం_v = __riscv_vle16_v_f16m2((_Float16 *) v_దత్తాంశం, DV);
            } లేకపోతే {
                కొనసాగు;
            }

            కొరకు (పూర్ణాంకం i = 0; i < QLEN; ++i) {
                vfloat16m2_t Q_q_v    = __riscv_vle16_v_f16m2(qv_బఫరు + i * DK, DK);
                vfloat32m4_t qk_acc_v = __riscv_vfwmul_vv_f32m4(k_దత్తాంశం_v, Q_q_v, DK);
                దశాంశం        s        = reduce_కూడిక_f32m4_vlen1024(qk_acc_v, DK);
                s                     = s * scale;
                ఒకవేళ (logit_softcap != 0.0f) {
                    s = logit_softcap * tanhf(s);
                }
                s += mv;

                స్థిరరాశి దశాంశం Mold = M[i];

                దశాంశం ms = 1.0f;  // upon new higher max val, scale VKQ and KQ sum with this value
                దశాంశం vs = 1.0f;  // post-softmax KQ value, expf(s - M)

                vfloat16m2_t VKQ16_v = __riscv_vle16_v_f16m2(kq16_బఫరు + i * DV, DV);
                ఒకవేళ (s > M[i]) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M[i] = s;
                    ms   = expf(Mold - M[i]);

                    // V = V*expf(Mold - M)
                    VKQ16_v = __riscv_vfmul_vf_f16m2(VKQ16_v, ms, DV);
                } లేకపోతే {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M[i]);
                }
                VKQ16_v = __riscv_vfmacc_vf_f16m2(VKQ16_v, vs, v_దత్తాంశం_v, DV);
                __riscv_vse16_v_f16m2(kq16_బఫరు + i * DV, VKQ16_v, DV);
                S[i] = S[i] * ms + vs;  // scale and increment sum with partial sum
            }
        }
    }

    కొరకు (పూర్ణాంకం i = 0; i < QLEN; ++i) {
        vfloat16m2_t VKQ16_v = __riscv_vle16_v_f16m2(kq16_బఫరు + i * DV, DV);
        vfloat32m4_t VKQ32_v = __riscv_vfwcvt_f_f_v_f32m4(VKQ16_v, DV);

        // sinks
        ఒకవేళ (sinks[i]) {
            స్థిరరాశి దశాంశం s = *(sinks[i]);

            దశాంశం ms = 1.0f;
            దశాంశం vs = 1.0f;

            ఒకవేళ (s > M[i]) {
                ms      = expf(M[i] - s);
                M[i]    = s;
                VKQ32_v = __riscv_vfmul_vf_f32m4(VKQ32_v, ms, DV);
            } లేకపోతే {
                vs = expf(s - M[i]);
            }

            S[i] = S[i] * ms + vs;
        }

        // V /= S
        స్థిరరాశి దశాంశం S_inv = S[i] == 0.0f ? 0.0f : 1.0f / S[i];

        VKQ32_v = __riscv_vfmul_vf_f32m4(VKQ32_v, S_inv, DV);

        __riscv_vse32_v_f32m4(dst[i], VKQ32_v, DV);
    }
}

స్థిర శూన్య flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_m1(స్థిరరాశి దశాంశం *       pq,
                                                                స్థిరరాశి అక్షరం *        k_దత్తాంశం_అడ్డువరుస,
                                                                స్థిరరాశి అక్షరం *        v_దత్తాంశం_అడ్డువరుస,
                                                                స్థిరరాశి ggml_fp16_t * mp,
                                                                స్థిరరాశి దశాంశం *       sinks,
                                                                దశాంశం *             dst,
                                                                దశాంశం               scale,
                                                                దశాంశం               logit_softcap,
                                                                దశాంశం               వాలు,
                                                                int64_t             nek1,
                                                                int64_t             nbk1,
                                                                int64_t             nbv1,
                                                                int64_t             DV,
                                                                int64_t             DK) {
    GGML_ASSERT(flash_attn_ext_supported_shape_vlen1024_vf16(DK, DV));

    దశాంశం S = 0.0f;       //కూడిక
    దశాంశం M = -INFINITY;  // maximum KQ value

    vfloat16m2_t VKQ16_v = __riscv_vfmv_v_f_f16m2(0.0f, DV);

    vfloat16m2_t Q_q_v = __riscv_vfncvt_f_f_w_f16m2(__riscv_vle32_v_f32m4(pq, DK), DK);

    కొరకు (int64_t ic = 0; ic < nek1; ++ic) {
        స్థిరరాశి దశాంశం mv = mp ? వాలు * ((_Float16 *) mp)[ic] : 0.0f;
        ఒకవేళ (mv == -INFINITY) {
            కొనసాగు;
        }

        స్థిరరాశి అక్షరం * k_దత్తాంశం = k_దత్తాంశం_అడ్డువరుస + ic * nbk1;

        vfloat16m2_t k_దత్తాంశం_v = __riscv_vle16_v_f16m2((_Float16 *) k_దత్తాంశం, DK);

        vfloat32m4_t qk_acc_v = __riscv_vfwmul_vv_f32m4(k_దత్తాంశం_v, Q_q_v, DK);
        దశాంశం        s        = reduce_కూడిక_f32m4_vlen1024(qk_acc_v, DK);

        s = s * scale;  // scale KQ value

        ఒకవేళ (logit_softcap != 0.0f) {
            s = logit_softcap * tanhf(s);
        }

        s += mv;  // apply mask

        స్థిరరాశి దశాంశం Mold = M;

        దశాంశం ms = 1.0f;  // upon new higher max val, scale VKQ and KQ sum with this value
        దశాంశం vs = 1.0f;  // post-softmax KQ value, expf(s - M)

        స్థిరరాశి అక్షరం * v_దత్తాంశం = v_దత్తాంశం_అడ్డువరుస + ic * nbv1;

        vfloat16m2_t v_దత్తాంశం_v = __riscv_vle16_v_f16m2((_Float16 *) v_దత్తాంశం, DV);

        ఒకవేళ (s > M) {
            // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
            M  = s;
            ms = expf(Mold - M);

            // V = V*expf(Mold - M)
            VKQ16_v = __riscv_vfmul_vf_f16m2(VKQ16_v, ms, DV);
        } లేకపోతే {
            // no new maximum, ms == 1.0f, vs != 1.0f
            vs = expf(s - M);
        }

        VKQ16_v = __riscv_vfmacc_vf_f16m2(VKQ16_v, vs, v_దత్తాంశం_v, DV);

        S = S * ms + vs;  // scale and increment sum with partial sum
    }

    vfloat32m4_t VKQ32_v = __riscv_vfwcvt_f_f_v_f32m4(VKQ16_v, DV);

    // sinks
    ఒకవేళ (sinks) {
        స్థిరరాశి దశాంశం s = *sinks;

        దశాంశం ms = 1.0f;
        దశాంశం vs = 1.0f;

        ఒకవేళ (s > M) {
            ms      = expf(M - s);
            M       = s;
            VKQ32_v = __riscv_vfmul_vf_f32m4(VKQ32_v, ms, DV);
        } లేకపోతే {
            vs = expf(s - M);
        }

        S = S * ms + vs;
    }

    // V /= S
    స్థిరరాశి దశాంశం S_inv = S == 0.0f ? 0.0f : 1.0f / S;

    VKQ32_v = __riscv_vfmul_vf_f32m4(VKQ32_v, S_inv, DV);

    __riscv_vse32_v_f32m4(dst, VKQ32_v, DV);
}

}  // namespace

శూన్య memcpy1d(శూన్య * dst, స్థిరరాశి శూన్య * src, int64_t size) {
    size_t byte_కొలత_all = size;
    size_t vlen          = __riscv_vlenb() * 8;
    ఒకవేళ (vlen == 256) {
        // 1024 bytes
        __asm__ అస్థిర(
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
            : [size] "r"(byte_కొలత_all)
            : "cc", "t0", "t1");
    } లేకపోతే ఒకవేళ (vlen == 1024) {
        // 2048 bytes
        __asm__ అస్థిర(
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
            : [size] "r"(byte_కొలత_all)
            : "cc", "t0", "t1", "t2", "t3", "t5");
    } లేకపోతే {
        __asm__ అస్థిర(
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
            : [size] "r"(byte_కొలత_all)
            : "cc", "t0", "t1", "t2", "t4", "t3");
    }
}

శూన్య memcpy2d(శూన్య * dst, int64_t dst_stride, స్థిరరాశి శూన్య * src, int64_t src_stride, int64_t tile_అడ్డువరుసలు, int64_t size) {
    కొరకు (int64_t i = 0; i < tile_అడ్డువరుసలు; ++i) {
        memcpy1d((అక్షరం *) dst + i * dst_stride, (స్థిరరాశి అక్షరం *) src + i * src_stride, size);
    }
}

శూన్య forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(స్థిరరాశి ggml_compute_params * params,
                                                        ggml_tensor *               dst,
                                                        పూర్ణాంకం                         ir0,
                                                        పూర్ణాంకం                         ir1,
                                                        శూన్య *                      tcm_బఫరు,
                                                        size_t                      tcm_బఫరు_కొలత) {
    స్థిరరాశి ggml_tensor * q     = dst->src[0];
    స్థిరరాశి ggml_tensor * k     = dst->src[1];
    స్థిరరాశి ggml_tensor * v     = dst->src[2];
    స్థిరరాశి ggml_tensor * ముసుగు  = dst->src[3];
    స్థిరరాశి ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q, ne)
    GGML_TENSOR_LOCALS(size_t, nbq, q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k, ne)
    GGML_TENSOR_LOCALS(size_t, nbk, k, nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v, ne)
    GGML_TENSOR_LOCALS(size_t, nbv, v, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t, nb, dst, nb)

    స్థిరరాశి int64_t DK = nek0;
    స్థిరరాశి int64_t DV = nev0;
    స్థిరరాశి int64_t N  = neq1;

    GGML_ASSERT(flash_attn_ext_supported_shape_vlen1024_vf16(DK, DV));

    // broadcast factors
    స్థిరరాశి int64_t rk2 = neq2 / nek2;
    స్థిరరాశి int64_t rk3 = neq3 / nek3;

    స్థిరరాశి int64_t rv2 = neq2 / nev2;
    స్థిరరాశి int64_t rv3 = neq3 / nev3;

    // parallelize by q rows using ggml_vec_dot_f32

    దశాంశం scale         = *((దశాంశం *) dst->op_params + 0);
    దశాంశం max_bias      = *((దశాంశం *) dst->op_params + 1);
    దశాంశం logit_softcap = *((దశాంశం *) dst->op_params + 2);

    ఒకవేళ (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    స్థిరరాశి uint32_t n_తల      = neq2;
    స్థిరరాశి uint32_t n_తల_log2 = 1u << (uint32_t) floor(log2(n_తల));

    స్థిరరాశి దశాంశం m0 = powf(2.0f, -(max_bias) / n_తల_log2);
    స్థిరరాశి దశాంశం m1 = powf(2.0f, -(max_bias / 2.0f) / n_తల_log2);

    స్థిరరాశి పూర్ణాంకం KV_అడ్డువరుస_కొలత = DK * పరిమాణం(_Float16) + DV * పరిమాణం(_Float16);

    పూర్ణాంకం ith     = params->ith;
    పూర్ణాంకం ir_అడుగు = 1;
    కొరకు (పూర్ణాంకం ir = ir0; ir < ir1; ir += ir_అడుగు) {
        // q indices
        స్థిరరాశి పూర్ణాంకం iq3 = ir / (neq2 * neq1);
        స్థిరరాశి పూర్ణాంకం iq2 = (ir - iq3 * neq2 * neq1) / neq1;
        స్థిరరాశి పూర్ణాంకం iq1 = (ir - iq3 * neq2 * neq1 - iq2 * neq1);

        స్థిరరాశి పూర్ణాంకం iq3_1 = (ir + 1) / (neq2 * neq1);
        స్థిరరాశి పూర్ణాంకం iq2_1 = (ir + 1 - iq3_1 * neq2 * neq1) / neq1;
        స్థిరరాశి పూర్ణాంకం iq1_1 = (ir + 1 - iq3_1 * neq2 * neq1 - iq2_1 * neq1);

        స్థిరరాశి పూర్ణాంకం iq3_2 = (ir + 2) / (neq2 * neq1);
        స్థిరరాశి పూర్ణాంకం iq2_2 = (ir + 2 - iq3_2 * neq2 * neq1) / neq1;
        స్థిరరాశి పూర్ణాంకం iq1_2 = (ir + 2 - iq3_2 * neq2 * neq1 - iq2_2 * neq1);

        స్థిరరాశి పూర్ణాంకం iq3_3 = (ir + 3) / (neq2 * neq1);
        స్థిరరాశి పూర్ణాంకం iq2_3 = (ir + 3 - iq3_3 * neq2 * neq1) / neq1;
        స్థిరరాశి పూర్ణాంకం iq1_3 = (ir + 3 - iq3_3 * neq2 * neq1 - iq2_3 * neq1);

        స్థిరరాశి uint32_t h = iq2;  // head index
        స్థిరరాశి దశాంశం    వాలు =
            (max_bias > 0.0f) ? h < n_తల_log2 ? powf(m0, h + 1) : powf(m1, 2 * (h - n_తల_log2) + 1) : 1.0f;

        స్థిరరాశి ggml_fp16_t * mp =
            ముసుగు ? (ggml_fp16_t *) ((అక్షరం *) ముసుగు->దత్తాంశం + iq1 * ముసుగు->nb[1] + (iq2 % ముసుగు->ne[2]) * ముసుగు->nb[2] +
                                    (iq3 % ముసుగు->ne[3]) * ముసుగు->nb[3]) :
                   ఖాళీ;

        స్థిరరాశి తార్కిక mp_equal_2 = iq1_1 == iq1 && (iq2 % ముసుగు->ne[2]) == (iq2_1 % ముసుగు->ne[2]) &&
                                (iq3 % ముసుగు->ne[3]) == (iq3_1 % ముసుగు->ne[3]);

        స్థిరరాశి తార్కిక mp_equal_4 = mp_equal_2 && iq1_2 == iq1 && (iq2 % ముసుగు->ne[2]) == (iq2_2 % ముసుగు->ne[2]) &&
                                (iq3 % ముసుగు->ne[3]) == (iq3_2 % ముసుగు->ne[3]) && iq1_3 == iq1 &&
                                (iq2 % ముసుగు->ne[2]) == (iq2_3 % ముసుగు->ne[2]) &&
                                (iq3 % ముసుగు->ne[3]) == (iq3_3 % ముసుగు->ne[3]);

        // k indices
        స్థిరరాశి పూర్ణాంకం ik3 = iq3 / rk3;
        స్థిరరాశి పూర్ణాంకం ik2 = iq2 / rk2;

        స్థిరరాశి పూర్ణాంకం ik3_1 = iq3_1 / rk3;
        స్థిరరాశి పూర్ణాంకం ik2_1 = iq2_1 / rk2;

        స్థిరరాశి పూర్ణాంకం ik3_2 = iq3_2 / rk3;
        స్థిరరాశి పూర్ణాంకం ik2_2 = iq2_2 / rk2;

        స్థిరరాశి పూర్ణాంకం ik3_3 = iq3_3 / rk3;
        స్థిరరాశి పూర్ణాంకం ik2_3 = iq2_3 / rk2;

        // v indices
        స్థిరరాశి పూర్ణాంకం iv3 = iq3 / rv3;
        స్థిరరాశి పూర్ణాంకం iv2 = iq2 / rv2;

        స్థిరరాశి పూర్ణాంకం iv3_1 = iq3_1 / rv3;
        స్థిరరాశి పూర్ణాంకం iv2_1 = iq2_1 / rv2;

        స్థిరరాశి పూర్ణాంకం iv3_2 = iq3_2 / rv3;
        స్థిరరాశి పూర్ణాంకం iv2_2 = iq2_2 / rv2;

        స్థిరరాశి పూర్ణాంకం iv3_3 = iq3_3 / rv3;
        స్థిరరాశి పూర్ణాంకం iv2_3 = iq2_3 / rv2;

        స్థిరరాశి దశాంశం * pq = (స్థిరరాశి దశాంశం *) ((అక్షరం *) q->దత్తాంశం + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3));

        std::array<దశాంశం *, 4> pq_బఫరు;
        std::array<దశాంశం *, 4> sinks_బఫరు;
        std::array<దశాంశం *, 4> dst_బఫరు;

        ఒకవేళ (tcm_బఫరు != శూన్యసూచిక && 4 * KV_అడ్డువరుస_కొలత < tcm_బఫరు_కొలత && ir < (ir1 - 3) && mp_equal_4 &&
            ik3_3 == ik3 && ik2_3 == ik2 && iv3_3 == iv3 && iv2_3 == iv2 && ik3_2 == ik3 && ik2_2 == ik2 &&
            iv3_2 == iv3 && iv2_2 == iv2 && ik3_1 == ik3 && ik2_1 == ik2 && iv3_1 == iv3 && iv2_1 == iv2) {
            ir_అడుగు = 4;

            pq_బఫరు[0] = (దశాంశం *) ((అక్షరం *) q->దత్తాంశం + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3));
            pq_బఫరు[1] = (దశాంశం *) ((అక్షరం *) q->దత్తాంశం + (iq1_1 * nbq1 + iq2_1 * nbq2 + iq3_1 * nbq3));
            pq_బఫరు[2] = (దశాంశం *) ((అక్షరం *) q->దత్తాంశం + (iq1_2 * nbq1 + iq2_2 * nbq2 + iq3_2 * nbq3));
            pq_బఫరు[3] = (దశాంశం *) ((అక్షరం *) q->దత్తాంశం + (iq1_3 * nbq1 + iq2_3 * nbq2 + iq3_3 * nbq3));

            sinks_బఫరు[0] = sinks ? ((దశాంశం *) ((అక్షరం *) sinks->దత్తాంశం)) + iq2 : శూన్యసూచిక;
            sinks_బఫరు[1] = sinks ? ((దశాంశం *) ((అక్షరం *) sinks->దత్తాంశం)) + iq2_1 : శూన్యసూచిక;
            sinks_బఫరు[2] = sinks ? ((దశాంశం *) ((అక్షరం *) sinks->దత్తాంశం)) + iq2_2 : శూన్యసూచిక;
            sinks_బఫరు[3] = sinks ? ((దశాంశం *) ((అక్షరం *) sinks->దత్తాంశం)) + iq2_3 : శూన్యసూచిక;

            dst_బఫరు[0] = (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + (iq3 * ne2 * ne1 + iq2 + iq1 * ne1) * nb1);
            dst_బఫరు[1] = (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + (iq3_1 * ne2 * ne1 + iq2_1 + iq1_1 * ne1) * nb1);
            dst_బఫరు[2] = (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + (iq3_2 * ne2 * ne1 + iq2_2 + iq1_2 * ne1) * nb1);
            dst_బఫరు[3] = (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + (iq3_3 * ne2 * ne1 + iq2_3 + iq1_3 * ne1) * nb1);

            flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_mrow<4>(  //
                pq_బఫరు.దత్తాంశం(),                                      //
                (స్థిరరాశి అక్షరం *) k->దత్తాంశం + (ik2 * nbk2 + ik3 * nbk3),    //
                (స్థిరరాశి అక్షరం *) v->దత్తాంశం + (iv2 * nbv2 + iv3 * nbv3),    //
                mp,                                                    //
                sinks_బఫరు.దత్తాంశం(),                                   //
                dst_బఫరు.దత్తాంశం(),                                     //
                scale, logit_softcap, వాలు, nek1, nbk1, nbv1, DV, DK, tcm_బఫరు, tcm_బఫరు_కొలత);
        } లేకపోతే ఒకవేళ (tcm_బఫరు != శూన్యసూచిక && 2 * KV_అడ్డువరుస_కొలత < tcm_బఫరు_కొలత && ir < (ir1 - 1) && mp_equal_2 &&
                   ik3_1 == ik3 && ik2_1 == ik2 && iv3_1 == iv3 && iv2_1 == iv2) {
            ir_అడుగు = 2;

            pq_బఫరు[0] = (దశాంశం *) ((అక్షరం *) q->దత్తాంశం + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3));
            pq_బఫరు[1] = (దశాంశం *) ((అక్షరం *) q->దత్తాంశం + (iq1_1 * nbq1 + iq2_1 * nbq2 + iq3_1 * nbq3));

            sinks_బఫరు[0] = sinks ? ((దశాంశం *) ((అక్షరం *) sinks->దత్తాంశం)) + iq2 : శూన్యసూచిక;
            sinks_బఫరు[1] = sinks ? ((దశాంశం *) ((అక్షరం *) sinks->దత్తాంశం)) + iq2_1 : శూన్యసూచిక;

            dst_బఫరు[0] = (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + (iq3 * ne2 * ne1 + iq2 + iq1 * ne1) * nb1);
            dst_బఫరు[1] = (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + (iq3_1 * ne2 * ne1 + iq2_1 + iq1_1 * ne1) * nb1);

            flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_mrow<2>(  //
                pq_బఫరు.దత్తాంశం(),                                      //
                (స్థిరరాశి అక్షరం *) k->దత్తాంశం + (ik2 * nbk2 + ik3 * nbk3),    //
                (స్థిరరాశి అక్షరం *) v->దత్తాంశం + (iv2 * nbv2 + iv3 * nbv3),    //
                mp,                                                    //
                sinks_బఫరు.దత్తాంశం(),                                   //
                dst_బఫరు.దత్తాంశం(),                                     //
                scale, logit_softcap, వాలు, nek1, nbk1, nbv1, DV, DK, tcm_బఫరు, tcm_బఫరు_కొలత);
        } లేకపోతే {
            ir_అడుగు = 1;
            flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_m1(                             //
                pq,                                                                          //
                (స్థిరరాశి అక్షరం *) k->దత్తాంశం + (ik2 * nbk2 + ik3 * nbk3),                          //
                (స్థిరరాశి అక్షరం *) v->దత్తాంశం + (iv2 * nbv2 + iv3 * nbv3),                          //
                mp,                                                                          //
                sinks ? ((దశాంశం *) ((అక్షరం *) sinks->దత్తాంశం)) + h : శూన్యసూచిక,                    //
                (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + (iq3 * ne2 * ne1 + iq2 + iq1 * ne1) * nb1),  //
                scale, logit_softcap, వాలు, nek1, nbk1, nbv1, DV, DK);
        }
    }
}

శూన్య forward_flash_attn_ext_f16_tiled_vlen1024_vf16(స్థిరరాశి ggml_compute_params * params,
                                                    ggml_tensor *               dst,
                                                    పూర్ణాంకం                         ir0,
                                                    పూర్ణాంకం                         ir1,
                                                    శూన్య *                      tcm_బఫరు,
                                                    size_t                      tcm_బఫరు_కొలత) {
    స్థిరరాశి ggml_tensor * q     = dst->src[0];
    స్థిరరాశి ggml_tensor * k     = dst->src[1];
    స్థిరరాశి ggml_tensor * v     = dst->src[2];
    స్థిరరాశి ggml_tensor * ముసుగు  = dst->src[3];
    స్థిరరాశి ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q, ne)
    GGML_TENSOR_LOCALS(size_t, nbq, q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k, ne)
    GGML_TENSOR_LOCALS(size_t, nbk, k, nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v, ne)
    GGML_TENSOR_LOCALS(size_t, nbv, v, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t, nb, dst, nb)

    స్థిరరాశి int64_t DK = nek0;
    స్థిరరాశి int64_t DV = nev0;
    స్థిరరాశి int64_t N  = neq1;

    GGML_ASSERT(flash_attn_ext_supported_shape_vlen1024_vf16(DK, DV));

    GGML_ASSERT(ne0 == DV);
    GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_కొలత(q->type));
    GGML_ASSERT(nbk0 == ggml_type_కొలత(k->type));
    GGML_ASSERT(nbv0 == ggml_type_కొలత(v->type));

    GGML_ASSERT(neq0 == DK);
    GGML_ASSERT(nek0 == DK);
    GGML_ASSERT(nev0 == DV);

    GGML_ASSERT(neq1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == పరిమాణం(దశాంశం));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(k->type == v->type);
    స్థిరరాశి ggml_type kv_type = k->type;

    // broadcast factors
    స్థిరరాశి int64_t rk2 = neq2 / nek2;
    స్థిరరాశి int64_t rk3 = neq3 / nek3;

    స్థిరరాశి int64_t rv2 = neq2 / nev2;
    స్థిరరాశి int64_t rv3 = neq3 / nev3;

    దశాంశం * param_list    = (దశాంశం *) dst->op_params;
    దశాంశం   scale         = param_list[0];
    దశాంశం   max_bias      = param_list[1];
    దశాంశం   logit_softcap = param_list[2];

    ఒకవేళ (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    స్థిరరాశి uint32_t n_తల      = neq2;
    స్థిరరాశి uint32_t n_తల_log2 = 1u << (uint32_t) floor(log2(n_తల));

    స్థిరరాశి దశాంశం m0 = powf(2.0f, -(max_bias) / n_తల_log2);
    స్థిరరాశి దశాంశం m1 = powf(2.0f, -(max_bias / 2.0f) / n_తల_log2);

    పూర్ణాంకం ith = params->ith;

    స్థిర constexpr పూర్ణాంకం Q_TILE_SZ  = ggml_fa_tile_config::Q;
    స్థిర constexpr పూర్ణాంకం KV_TILE_SZ = ggml_fa_tile_config::KV;

    // Per-thread scratch layout:
    // Q_f32:   Q_TILE_SZ * DK
    // KQ:      Q_TILE_SZ * KV_TILE_SZ
    // mask32:  Q_TILE_SZ * KV_TILE_SZ
    // VKQ32:   Q_TILE_SZ * DV
    // V32:     KV_TILE_SZ * DV
    // K_f32:   DK * KV_TILE_SZ (transposed K tile)
    దశాంశం *      పునాది = (దశాంశం *) params->wdata + ith * (Q_TILE_SZ * DK + 2 * Q_TILE_SZ * KV_TILE_SZ + Q_TILE_SZ * DV +
                                                    KV_TILE_SZ * DV + KV_TILE_SZ * DK + CACHE_పంక్తి_కొలత_F32);
    స్థిరరాశి size_t పునాది_కొలత =
        (Q_TILE_SZ * DK + 2 * Q_TILE_SZ * KV_TILE_SZ + Q_TILE_SZ * DV + KV_TILE_SZ * DV + KV_TILE_SZ * DK) *
            పరిమాణం(దశాంశం) +
        CACHE_పంక్తి_కొలత_F32;

    ఒకవేళ (పునాది_కొలత <= tcm_బఫరు_కొలత && tcm_బఫరు != శూన్యసూచిక) {
        పునాది = (దశాంశం *) tcm_బఫరు;
    }

    దశాంశం   S_M_Buf[Q_TILE_SZ * 2];  // buffer to hold S, M, bias for one tile to reduce register pressure in main loop
    దశాంశం * S = S_M_Buf;
    దశాంశం * M = S_M_Buf + Q_TILE_SZ;

    పూర్ణాంకం ir = ir0;
    అయితే (ir < ir1) {
        // q indices for the start of this tile
        స్థిరరాశి పూర్ణాంకం iq3 = ir / (neq2 * neq1);
        స్థిరరాశి పూర్ణాంకం iq2 = (ir - iq3 * neq2 * neq1) / neq1;
        స్థిరరాశి పూర్ణాంకం iq1 = (ir - iq3 * neq2 * neq1 - iq2 * neq1);

        // Number of valid rows in this tile:
        // - limited by tile size (Q_TILE_SZ)
        // - limited by chunk boundary (ir1 - ir)
        // - limited by head boundary (neq1 - iq1) to avoid crossing into next head
        స్థిరరాశి పూర్ణాంకం tile_అడ్డువరుసలు = MIN(Q_TILE_SZ, MIN((పూర్ణాంకం) (ir1 - ir), (పూర్ణాంకం) (neq1 - iq1)));
        GGML_ASSERT(tile_అడ్డువరుసలు > 0);

        స్థిరరాశి uint32_t h = iq2;  // head index
        స్థిరరాశి దశాంశం    వాలు =
            (max_bias > 0.0f) ? h < n_తల_log2 ? powf(m0, h + 1) : powf(m1, 2 * (h - n_తల_log2) + 1) : 1.0f;

        కొరకు (పూర్ణాంకం i = 0; i < Q_TILE_SZ; ++i) {
            S[i] = 0.;
            M[i] = -INFINITY;
        }

        దశాంశం *    Q_f32  = పునాది;
        దశాంశం *    KQ     = (దశాంశం *) ((అక్షరం *) పునాది + Q_TILE_SZ * DK * పరిమాణం(దశాంశం));
        దశాంశం *    mask32 = KQ + Q_TILE_SZ * KV_TILE_SZ;
        దశాంశం *    VKQ32  = mask32 + Q_TILE_SZ * KV_TILE_SZ;
        దశాంశం *    V32    = VKQ32 + Q_TILE_SZ * DV;
        దశాంశం *    K_f32  = V32 + KV_TILE_SZ * DV;
        _Float16 * Q_f16  = (_Float16 *) Q_f32;
        _Float16 * V_f16  = (_Float16 *) V32;
        _Float16 * K_f16  = (_Float16 *) K_f32;

        rvv_zero_f32(VKQ32, Q_TILE_SZ * DV);

        // k indices
        స్థిరరాశి పూర్ణాంకం ik3 = iq3 / rk3;
        స్థిరరాశి పూర్ణాంకం ik2 = iq2 / rk2;

        // v indices
        స్థిరరాశి పూర్ణాంకం iv3 = iq3 / rv3;
        స్థిరరాశి పూర్ణాంకం iv2 = iq2 / rv2;

        స్థిరరాశి దశాంశం * pq = (స్థిరరాశి దశాంశం *) ((అక్షరం *) q->దత్తాంశం + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3));
        ఒకవేళ (kv_type == GGML_TYPE_F16) {
            rvv_pack_f32_as_scaled_f16((uint8_t *) Q_f16, DK * పరిమాణం(_Float16), (uint8_t *) pq, nbq1, tile_అడ్డువరుసలు, DK,
                                       scale);
        } లేకపోతే {
            memcpy2d(Q_f32, DK * పరిమాణం(దశాంశం), pq, nbq1, tile_అడ్డువరుసలు, DK * పరిమాణం(దశాంశం));
        }

        కొరకు (int64_t ic = 0; ic < nek1; ic += KV_TILE_SZ) {
            స్థిరరాశి పూర్ణాంకం kv_tile = (పూర్ణాంకం) std::min((int64_t) KV_TILE_SZ, nek1 - ic);

            rvv_zero_f32(K_f32, DK * KV_TILE_SZ);
            rvv_zero_f32(V32, KV_TILE_SZ * DV);

            // skip the tile entirely if all the masks are -inf
            ఒకవేళ (ముసుగు) {
                తార్కిక                can_skip = నిజం;
                స్థిరరాశి ggml_fp16_t * mp_అడ్డువరుస =
                    (స్థిరరాశి ggml_fp16_t *) ((స్థిరరాశి అక్షరం *) ముసుగు->దత్తాంశం + iq1 * ముసుగు->nb[1] +
                                           (iq2 % ముసుగు->ne[2]) * ముసుగు->nb[2] + (iq3 % ముసుగు->ne[3]) * ముసుగు->nb[3]);
                rvv_pack_scaled_f16_as_f32(mask32, KV_TILE_SZ * పరిమాణం(దశాంశం), mp_అడ్డువరుస + ic, ముసుగు->nb[1], tile_అడ్డువరుసలు,
                                           kv_tile, వాలు);

                కొరకు (పూర్ణాంకం tq = 0; tq < tile_అడ్డువరుసలు; tq++) {
                    కొరకు (పూర్ణాంకం tk = 0; tk < kv_tile; tk++) {
                        ఒకవేళ (mask32[tq * KV_TILE_SZ + tk] != -INFINITY) {
                            can_skip = అబద్ధం;
                        }
                    }
                    // Pad remaining mask entries with -inf
                    కొరకు (పూర్ణాంకం tk = kv_tile; tk < KV_TILE_SZ; tk++) {
                        mask32[tq * KV_TILE_SZ + tk] = -INFINITY;
                    }
                }

                ఒకవేళ (can_skip) {
                    కొనసాగు;
                }
            }

            ఒకవేళ (kv_type == GGML_TYPE_F16) {
                rvv_transposed_s16_mn_గమ్యం_nm((int8_t *) K_f16, KV_TILE_SZ * పరిమాణం(_Float16),
                                            (int8_t *) k->దత్తాంశం + ic * nbk1 + ik2 * nbk2 + ik3 * nbk3, nbk1, kv_tile,
                                            DK);

                పూర్ణాంకం tq = 0;
                కొరకు (; tq + 3 < tile_అడ్డువరుసలు; tq += 4) {
                    rvv_qk_dot_tile_f16_x4(KQ + (tq + 0) * KV_TILE_SZ, KQ + (tq + 1) * KV_TILE_SZ,
                                           KQ + (tq + 2) * KV_TILE_SZ, KQ + (tq + 3) * KV_TILE_SZ,
                                           Q_f16 + (tq + 0) * DK, Q_f16 + (tq + 1) * DK, Q_f16 + (tq + 2) * DK,
                                           Q_f16 + (tq + 3) * DK, K_f16, DK, kv_tile);
                }
                కొరకు (; tq < tile_అడ్డువరుసలు; ++tq) {
                    rvv_qk_dot_tile_f16_x1(KQ + tq * KV_TILE_SZ, Q_f16 + tq * DK, K_f16, DK, kv_tile);
                }
            } లేకపోతే {
                కొరకు (పూర్ణాంకం tk = 0; tk < kv_tile; tk++) {
                    స్థిరరాశి అక్షరం *  k_దత్తాంశం = (స్థిరరాశి అక్షరం *) k->దత్తాంశం + (ic + tk) * nbk1 + ik2 * nbk2 + ik3 * nbk3;
                    దశాంశం *       k_నిలువు  = K_f32 + tk;
                    స్థిరరాశి దశాంశం * k_src  = (స్థిరరాశి దశాంశం *) k_దత్తాంశం;
                    కొరకు (int64_t dk = 0; dk < DK; ++dk) {
                        k_నిలువు[dk * KV_TILE_SZ] = k_src[dk];
                    }
                }

                కొరకు (పూర్ణాంకం tq = 0; tq < tile_అడ్డువరుసలు; ++tq) {
                    rvv_qk_dot_tile(KQ + tq * KV_TILE_SZ, Q_f32 + tq * DK, K_f32, DK, KV_TILE_SZ, scale);
                }
            }

            // Set padded KQ entries to -inf so softmax gives them zero weight
            ఒకవేళ (kv_tile < KV_TILE_SZ) {
                కొరకు (పూర్ణాంకం tq = 0; tq < tile_అడ్డువరుసలు; tq++) {
                    కొరకు (పూర్ణాంకం tk = kv_tile; tk < KV_TILE_SZ; tk++) {
                        KQ[tq * KV_TILE_SZ + tk] = -INFINITY;
                    }
                }
            }

            ఒకవేళ (logit_softcap != 0.0f) {
                rvv_softcap_tanh_inplace_f32(KQ, KV_TILE_SZ, tile_అడ్డువరుసలు, KV_TILE_SZ, logit_softcap);
            }

            ఒకవేళ (ముసుగు) {
                rvv_కూడు_inplace_f32(KQ, KV_TILE_SZ, mask32, KV_TILE_SZ, tile_అడ్డువరుసలు, KV_TILE_SZ);
            }

            తార్కిక skip[Q_TILE_SZ] = {};

            కొరకు (పూర్ణాంకం tq = 0; tq < tile_అడ్డువరుసలు; tq++) {
                దశాంశం * kq_అడ్డువరుస = KQ + tq * KV_TILE_SZ;

                స్థిరరాశి దశాంశం tile_max = rvv_max_f32(kq_అడ్డువరుస, KV_TILE_SZ);

                ఒకవేళ (tile_max == -INFINITY) {
                    skip[tq] = నిజం;
                    కొనసాగు;
                }

                స్థిరరాశి దశాంశం Mold = M[tq];
                స్థిరరాశి దశాంశం Mnew = fmaxf(Mold, tile_max);

                ఒకవేళ (Mnew > Mold) {
                    స్థిరరాశి దశాంశం ms = expf(Mold - Mnew);
                    rvv_scale_f32(VKQ32 + tq * DV, ms, DV);
                    S[tq] *= ms;
                }
                M[tq] = Mnew;

                S[tq] += rvv_softmax_exp_inplace_f32(kq_అడ్డువరుస, KV_TILE_SZ, Mnew);
            }

            // Pack V as contiguous [KV_TILE_SZ][DV].
            ఒకవేళ (kv_type == GGML_TYPE_F16) {
                స్థిరరాశి అక్షరం * v_దత్తాంశం = (స్థిరరాశి అక్షరం *) v->దత్తాంశం + ic * nbv1 + iv2 * nbv2 + iv3 * nbv3;
                memcpy2d(V_f16, DV * పరిమాణం(_Float16), v_దత్తాంశం, nbv1, kv_tile, DV * పరిమాణం(_Float16));

                పూర్ణాంకం tq = 0;
                కొరకు (; tq + 3 < tile_అడ్డువరుసలు; tq += 4) {
                    ఒకవేళ (skip[tq + 0] || skip[tq + 1] || skip[tq + 2] || skip[tq + 3]) {
                        కొరకు (పూర్ణాంకం i = 0; i < 4; ++i) {
                            ఒకవేళ (!skip[tq + i]) {
                                rvv_pv_accumulate_f16_x1(VKQ32 + (tq + i) * DV, KQ + (tq + i) * KV_TILE_SZ, V_f16,
                                                         KV_TILE_SZ, DV);
                            }
                        }
                        కొనసాగు;
                    }

                    rvv_pv_accumulate_f16_x4(VKQ32 + (tq + 0) * DV, VKQ32 + (tq + 1) * DV, VKQ32 + (tq + 2) * DV,
                                             VKQ32 + (tq + 3) * DV, KQ + (tq + 0) * KV_TILE_SZ,
                                             KQ + (tq + 1) * KV_TILE_SZ, KQ + (tq + 2) * KV_TILE_SZ,
                                             KQ + (tq + 3) * KV_TILE_SZ, V_f16, KV_TILE_SZ, DV);
                }
                కొరకు (; tq < tile_అడ్డువరుసలు; ++tq) {
                    ఒకవేళ (!skip[tq]) {
                        rvv_pv_accumulate_f16_x1(VKQ32 + tq * DV, KQ + tq * KV_TILE_SZ, V_f16, KV_TILE_SZ, DV);
                    }
                }
            } లేకపోతే {
                స్థిరరాశి అక్షరం * v_దత్తాంశం = (స్థిరరాశి అక్షరం *) v->దత్తాంశం + ic * nbv1 + iv2 * nbv2 + iv3 * nbv3;
                memcpy2d(V32, DV * పరిమాణం(దశాంశం), v_దత్తాంశం, nbv1, kv_tile, DV * పరిమాణం(దశాంశం));

                కొరకు (పూర్ణాంకం tq = 0; tq < tile_అడ్డువరుసలు; ++tq) {
                    ఒకవేళ (!skip[tq]) {
                        rvv_pv_accumulate(VKQ32 + tq * DV, KQ + tq * KV_TILE_SZ, V32, KV_TILE_SZ, DV);
                    }
                }
            }
        }

        // sinks (apply only to valid rows in the tile)
        ఒకవేళ (sinks) {
            స్థిరరాశి దశాంశం s = ((దశాంశం *) ((అక్షరం *) sinks->దత్తాంశం))[h];

            కొరకు (పూర్ణాంకం tq = 0; tq < tile_అడ్డువరుసలు; tq++) {
                దశాంశం ms = 1.0f;
                దశాంశం vs = 1.0f;

                ఒకవేళ (s > M[tq]) {
                    ms = expf(M[tq] - s);
                    rvv_scale_f32(VKQ32 + tq * DV, ms, DV);
                } లేకపోతే {
                    vs = expf(s - M[tq]);
                }

                దశాంశం S_తాత్కాలిక = S[tq] * ms + vs;
                S[tq]        = S_తాత్కాలిక == 0.0f ? 0.0f : 1.0f / S_తాత్కాలిక;
            }
        } లేకపోతే {
            కొరకు (పూర్ణాంకం tq = 0; tq < tile_అడ్డువరుసలు; tq++) {
                స్థిరరాశి దశాంశం S_inv = S[tq] == 0.0f ? 0.0f : 1.0f / S[tq];
                S[tq]             = S_inv;
            }
        }

        దశాంశం * dst_ptr = (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + (iq3 * ne2 * ne1 + iq2 + (iq1) *ne1) * nb1);
        rvv_pack_scaled_f32_as_f32(dst_ptr, nb1 * ne1, VKQ32, DV * పరిమాణం(దశాంశం), tile_అడ్డువరుసలు, DV, S);

        ir += tile_అడ్డువరుసలు;
    }
}

శూన్య forward_rms_norm_f32(ggml_compute_params * params, ggml_tensor * op) {
    స్థిరరాశి ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(src0->nb[0] == పరిమాణం(దశాంశం));

    పూర్ణాంకం ith = params->ith;
    పూర్ణాంకం nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    దశాంశం epsilon = *((దశాంశం *) dst->op_params);

    GGML_ASSERT(epsilon > 0.0f);

    స్వయంచాలక * ఆదానం  = (అక్షరం *) src0->దత్తాంశం;
    స్వయంచాలక * output = (అక్షరం *) dst->దత్తాంశం;

    స్థిరరాశి స్వయంచాలక hidden_కొలత     = ne00;
    స్థిరరాశి స్వయంచాలక task_లెక్క      = ne01 * ne02 * ne03;
    స్థిరరాశి స్వయంచాలక task_చొప్పున_thread = (task_లెక్క + nth - 1) / nth;

    స్థిరరాశి స్వయంచాలక task_begin = ith * task_చొప్పున_thread;
    స్థిరరాశి స్వయంచాలక task_చివర   = std::min((ith + 1) * task_చొప్పున_thread, task_లెక్క);

    కొరకు (స్వయంచాలక task_idx = task_begin; task_idx < task_చివర; task_idx++) {
        int64_t i03 = task_idx / (ne02 * ne01);
        int64_t i02 = (task_idx - i03 * ne02 * ne01) / ne01;
        int64_t i01 = (task_idx - i03 * ne02 * ne01 - i02 * ne01);

        స్వయంచాలక * p_ఆదానం       = (దశాంశం *) (ఆదానం + i01 * nb01 + i02 * nb02 + i03 * nb03);
        స్వయంచాలక * p_output      = (దశాంశం *) (output + i01 * nb1 + i02 * nb2 + i03 * nb3);
        స్వయంచాలక * p_తాత్కాలిక_output = p_output;

        size_t       gvl    = __riscv_vsetvlmax_e32m4();
        vfloat32m4_t కూడిక_sq = __riscv_vfmv_v_f_f32m4(0.f, gvl);
        int64_t      నిడివి = hidden_కొలత;
        అయితే (నిడివి > 0) {
            gvl                   = __riscv_vsetvl_e32m4(నిడివి);
            vfloat32m4_t src_దత్తాంశం = __riscv_vle32_v_f32m4(p_ఆదానం, gvl);
            కూడిక_sq                = __riscv_vfmacc_vv_f32m4(కూడిక_sq, src_దత్తాంశం, src_దత్తాంశం, gvl);
            __riscv_vse32_v_f32m4(p_తాత్కాలిక_output, src_దత్తాంశం, gvl);

            p_ఆదానం += gvl;
            p_తాత్కాలిక_output += gvl;
            నిడివి -= gvl;
        }

        gvl                 = __riscv_vsetvlmax_e32m1();
        vfloat32m1_t zero_v = __riscv_vfmv_v_f_f32m1(0.f, gvl);
        vfloat32m1_t సగటు_వర్గం_v =
            __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(కూడిక_sq, 0), __riscv_vget_v_f32m4_f32m1(కూడిక_sq, 1), gvl);

        సగటు_వర్గం_v = __riscv_vfadd_vv_f32m1(సగటు_వర్గం_v, __riscv_vget_v_f32m4_f32m1(కూడిక_sq, 2), gvl);
        సగటు_వర్గం_v = __riscv_vfadd_vv_f32m1(సగటు_వర్గం_v, __riscv_vget_v_f32m4_f32m1(కూడిక_sq, 3), gvl);
        సగటు_వర్గం_v = __riscv_vfredusum_vs_f32m1_f32m1(సగటు_వర్గం_v, zero_v, gvl);

        దశాంశం సగటు_వర్గం = __riscv_vfmv_f_s_f32m1_f32(సగటు_వర్గం_v);
        సగటు_వర్గం /= hidden_కొలత;

        సగటు_వర్గం = వర్గమూలం(సగటు_వర్గం + epsilon);

        సగటు_వర్గం   = 1.0f / సగటు_వర్గం;
        నిడివి        = hidden_కొలత;
        p_తాత్కాలిక_output = p_output;

        అయితే (నిడివి > 0) {
            gvl                   = __riscv_vsetvl_e32m4(నిడివి);
            vfloat32m4_t src_దత్తాంశం = __riscv_vle32_v_f32m4(p_తాత్కాలిక_output, gvl);
            src_దత్తాంశం              = __riscv_vfmul_vf_f32m4(src_దత్తాంశం, సగటు_వర్గం, gvl);
            __riscv_vse32_v_f32m4(p_output, src_దత్తాంశం, gvl);
            p_తాత్కాలిక_output += gvl;
            p_output += gvl;
            నిడివి -= gvl;
        }
    }
}

మూస <size_t MB_అడ్డువరుసలు>
శూన్య quantize_a_nrow_i8_ref(size_t blk_len, స్థిరరాశి దశాంశం * a_ptr, size_t లెక్క_k, uint8_t * quant_a_ptr) {
    int64_t a_blk_stride        = q8_blk_కొలత(blk_len, నిజం);
    int64_t a_nrow_ఖండం_stride = a_blk_stride * MB_అడ్డువరుసలు;
    కొరకు (size_t k = 0; k < లెక్క_k; k += blk_len, quant_a_ptr += a_nrow_ఖండం_stride) {
        దశాంశం *   scale_a_ptr = reinterpret_cast<దశాంశం *>(quant_a_ptr);
        int16_t * a_కూడిక_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + పరిమాణం(దశాంశం) * MB_అడ్డువరుసలు);
        int8_t *  quant_a_blk =
            reinterpret_cast<int8_t *>(quant_a_ptr + పరిమాణం(దశాంశం) * MB_అడ్డువరుసలు + పరిమాణం(int16_t) * MB_అడ్డువరుసలు);

        కొరకు (size_t అడ్డువరుస = 0; అడ్డువరుస < MB_అడ్డువరుసలు; అడ్డువరుస++) {
            దశాంశం max_abs_a = 0.0f;
            కొరకు (size_t bk = 0; bk < blk_len; bk++) {
                max_abs_a = std::max(max_abs_a, std::నిరపేక్ష(a_ptr[అడ్డువరుస * లెక్క_k + k + bk]));
            }

            దశాంశం rep_scale_a = ((1 << 7) - 1) / max_abs_a;
            scale_a_ptr[అడ్డువరుస]  = 1 / rep_scale_a;

            int16_t a_కూడిక = 0;
            కొరకు (size_t bk = 0; bk < blk_len; bk++) {
                స్థిరరాశి int8_t quantized = static_cast<int8_t>(
                    std::clamp(std::nearbyintf(a_ptr[అడ్డువరుస * లెక్క_k + k + bk] * rep_scale_a), -128.0f, 127.0f));
                quant_a_blk[అడ్డువరుస * blk_len + bk] = quantized;
                a_కూడిక += quantized;
            }
            a_కూడిక_ptr[అడ్డువరుస] = -a_కూడిక;
        }
    }
}

మూస <size_t MB_అడ్డువరుసలు>
శూన్య quantize_a_nrow_i8_hp_ref(size_t blk_len, స్థిరరాశి దశాంశం * a_ptr, size_t లెక్క_k, uint8_t * quant_a_ptr) {
    constexpr size_t k_subblk_len = 32;
    స్థిరరాశి size_t     subblk_లెక్క = blk_len / k_subblk_len;

    GGML_ASSERT(blk_len == 256);

    దశాంశం   scale_తాత్కాలిక[8]       = { 0.0f };
    int64_t a_blk_stride        = q8_hp_blk_కొలత(blk_len, నిజం, నిజం);
    int64_t a_nrow_ఖండం_stride = a_blk_stride * MB_అడ్డువరుసలు;
    int64_t a_subblk_stride     = q8_hp_blk_కొలత(k_subblk_len, అబద్ధం, అబద్ధం) * MB_అడ్డువరుసలు;

    కొరకు (size_t k = 0; k < లెక్క_k; k += blk_len, quant_a_ptr += a_nrow_ఖండం_stride) {
        _Float16 * a_కూడిక_ptr = reinterpret_cast<_Float16 *>(quant_a_ptr + a_subblk_stride * subblk_లెక్క);

        దశాంశం scale_avg = 0.0f;
        కొరకు (size_t kk = 0; kk < subblk_లెక్క; kk++) {
            దశాంశం max_abs_a = 0.0f;
            కొరకు (size_t అడ్డువరుస = 0; అడ్డువరుస < MB_అడ్డువరుసలు; అడ్డువరుస++) {
                కొరకు (size_t bk = 0; bk < k_subblk_len; bk++) {
                    max_abs_a = std::max(max_abs_a, std::నిరపేక్ష(a_ptr[అడ్డువరుస * లెక్క_k + k + bk + kk * k_subblk_len]));
                }
            }
            scale_తాత్కాలిక[kk] = max_abs_a / ((1 << 7) - 1);
            scale_avg += scale_తాత్కాలిక[kk];
        }

        scale_avg /= subblk_లెక్క;
        దశాంశం scale_గుణకం = 1.0f / scale_avg;

        _Float16 * scale_avg_ptr =
            reinterpret_cast<_Float16 *>(quant_a_ptr + a_nrow_ఖండం_stride - పరిమాణం(_Float16) * MB_అడ్డువరుసలు);
        scale_avg_ptr[0] = scale_avg;

        కొరకు (size_t kk = 0; kk < subblk_లెక్క; kk++) {
            uint8_t *  a_subblk_పునాది = quant_a_ptr + kk * a_subblk_stride;
            _Float16 * scale_a_ptr   = reinterpret_cast<_Float16 *>(a_subblk_పునాది);
            int8_t *   quant_a_blk   = reinterpret_cast<int8_t *>(a_subblk_పునాది + పరిమాణం(_Float16) * MB_అడ్డువరుసలు);

            scale_a_ptr[0] = static_cast<_Float16>(scale_తాత్కాలిక[kk] * scale_గుణకం);

            స్థిరరాశి దశాంశం rep_scale_a = 1.0f / scale_తాత్కాలిక[kk];

            కొరకు (size_t అడ్డువరుస = 0; అడ్డువరుస < MB_అడ్డువరుసలు; అడ్డువరుస++) {
                int16_t a_కూడిక = 0;
                కొరకు (size_t bk = 0; bk < k_subblk_len; bk++) {
                    స్థిరరాశి int8_t quantized = static_cast<int8_t>(
                        std::clamp(std::nearbyintf(a_ptr[అడ్డువరుస * లెక్క_k + k + bk + kk * k_subblk_len] * rep_scale_a),
                                   -128.0f, 127.0f));
                    quant_a_blk[అడ్డువరుస * k_subblk_len + bk] = quantized;
                    a_కూడిక += quantized;
                }
                a_కూడిక_ptr[అడ్డువరుస * subblk_లెక్క + kk] = static_cast<_Float16>(-a_కూడిక) * static_cast<_Float16>(8.0f);
            }
        }
    }
}

మూస <size_t MB_అడ్డువరుసలు>
శూన్య quantize_a_nrow_i8k_ref(size_t blk_len, స్థిరరాశి దశాంశం * a_ptr, size_t లెక్క_k, uint8_t * quant_a_ptr) {
    int64_t a_blk_stride        = q8k_blk_కొలత(256);
    int64_t a_nrow_ఖండం_stride = a_blk_stride * MB_అడ్డువరుసలు;
    int64_t a_కూడిక_కొలత          = 256 / 16;

    కొరకు (size_t k = 0; k < లెక్క_k; k += blk_len, quant_a_ptr += a_nrow_ఖండం_stride) {
        దశాంశం *   scale_a_ptr = reinterpret_cast<దశాంశం *>(quant_a_ptr);
        int16_t * a_కూడిక_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + పరిమాణం(దశాంశం) * MB_అడ్డువరుసలు);
        int8_t *  quant_a_blk =
            reinterpret_cast<int8_t *>(quant_a_ptr + పరిమాణం(దశాంశం) * MB_అడ్డువరుసలు + పరిమాణం(int16_t) * a_కూడిక_కొలత * MB_అడ్డువరుసలు);

        కొరకు (size_t అడ్డువరుస = 0; అడ్డువరుస < MB_అడ్డువరుసలు; అడ్డువరుస++) {
            దశాంశం max_a     = 0.0f;
            దశాంశం max_abs_a = 0.0f;
            కొరకు (size_t bk = 0; bk < blk_len; bk++) {
                దశాంశం ax = std::నిరపేక్ష(a_ptr[అడ్డువరుస * లెక్క_k + k + bk]);
                ఒకవేళ (ax > max_abs_a) {
                    max_abs_a = ax;
                    max_a     = a_ptr[అడ్డువరుస * లెక్క_k + k + bk];
                }
            }

            ఒకవేళ (!max_abs_a) {
                scale_a_ptr[అడ్డువరుస] = 0;
                కొరకు (size_t bki = 0; bki < a_కూడిక_కొలత; bki++) {
                    కొరకు (size_t bk = bki * 16; bk < (bki + 1) * 16; bk++) {
                        quant_a_blk[అడ్డువరుస * blk_len + bk] = 0;
                    }
                    a_కూడిక_ptr[అడ్డువరుస * a_కూడిక_కొలత + bki] = 0;
                }
                కొనసాగు;
            }

            దశాంశం rep_scale_a = ((1 << 7) - 1) / max_abs_a;
            scale_a_ptr[అడ్డువరుస]  = 1 / rep_scale_a;

            కొరకు (size_t bki = 0; bki < a_కూడిక_కొలత; bki++) {
                int16_t a_కూడిక = 0;
                కొరకు (size_t bk = bki * 16; bk < (bki + 1) * 16; bk++) {
                    స్థిరరాశి int8_t quantized = static_cast<int8_t>(
                        std::clamp(std::nearbyintf(a_ptr[అడ్డువరుస * లెక్క_k + k + bk] * rep_scale_a), -128.0f, 127.0f));
                    quant_a_blk[అడ్డువరుస * blk_len + bk] = quantized;
                    a_కూడిక += quantized;
                }
                a_కూడిక_ptr[అడ్డువరుస * a_కూడిక_కొలత + bki] = -a_కూడిక;
            }
        }
    }
}

శూన్య quantize_a_అడ్డువరుస_i8(size_t blk_len, స్థిరరాశి దశాంశం * a_ptr, size_t లెక్క_k, uint8_t * quant_a_ptr) {
    GGML_ASSERT(blk_len == 32);
    int64_t a_blk_stride = q8_blk_కొలత(blk_len, నిజం);
    size_t  vlenb        = __riscv_vlenb();

    ఒకవేళ (vlenb == 128) {
        కొరకు (size_t k = 0; k < లెక్క_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            దశాంశం *   scale_a_ptr = reinterpret_cast<దశాంశం *>(quant_a_ptr);
            int16_t * a_కూడిక_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + పరిమాణం(దశాంశం));
            int8_t *  quant_a_blk = reinterpret_cast<int8_t *>(quant_a_ptr + పరిమాణం(దశాంశం) + పరిమాణం(int16_t));

            size_t       vl      = __riscv_vsetvl_e32m1(blk_len);
            vfloat32m1_t v_a     = __riscv_vle32_v_f32m1(a_ptr + k, vl);
            vfloat32m1_t v_a_abs = __riscv_vfabs_v_f32m1(v_a, vl);

            vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m1_f32m1(v_a_abs, tmp, vl);
            దశాంశం        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

            దశాంశం scale_a     = max_abs_a / ((1 << 7) - 1);
            దశాంశం rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
            scale_a_ptr[0]    = scale_a;

            vfloat32m1_t v_a_scale    = __riscv_vfmul_vf_f32m1(v_a, rep_scale_a, vl);
            vint16mf2_t  v_a_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a_scale, vl);
            vint8mf4_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a_quant, vl);

            vint16m1_t tmp_కూడిక = __riscv_vmv_v_x_i16m1(0, vl);
            vint16m1_t v_a_కూడిక = __riscv_vwredsum_vs_i8mf4_i16m1(v_a_quant_i8, tmp_కూడిక, vl);
            int16_t    a_కూడిక   = __riscv_vmv_x_s_i16m1_i16(v_a_కూడిక);
            a_కూడిక_ptr[0]       = -a_కూడిక;

            __riscv_vse8_v_i8mf4(quant_a_blk, v_a_quant_i8, vl);
        }
    } లేకపోతే ఒకవేళ (vlenb == 32) {
        కొరకు (size_t k = 0; k < లెక్క_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            దశాంశం *   scale_a_ptr = reinterpret_cast<దశాంశం *>(quant_a_ptr);
            int16_t * a_కూడిక_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + పరిమాణం(దశాంశం));
            int8_t *  quant_a_blk = reinterpret_cast<int8_t *>(quant_a_ptr + పరిమాణం(దశాంశం) + పరిమాణం(int16_t));

            size_t       vl      = __riscv_vsetvl_e32m4(blk_len);
            vfloat32m4_t v_a     = __riscv_vle32_v_f32m4(a_ptr + k, vl);
            vfloat32m4_t v_a_abs = __riscv_vfabs_v_f32m4(v_a, vl);

            vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m4_f32m1(v_a_abs, tmp, vl);
            దశాంశం        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

            దశాంశం scale_a     = max_abs_a / ((1 << 7) - 1);
            దశాంశం rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
            scale_a_ptr[0]    = scale_a;

            vfloat32m4_t v_a_scale    = __riscv_vfmul_vf_f32m4(v_a, rep_scale_a, vl);
            vint16m2_t   v_a_quant    = __riscv_vfncvt_x_f_w_i16m2(v_a_scale, vl);
            vint8m1_t    v_a_quant_i8 = __riscv_vncvt_x_x_w_i8m1(v_a_quant, vl);

            vint16m1_t tmp_కూడిక = __riscv_vmv_v_x_i16m1(0, vl);
            vint16m1_t v_a_కూడిక = __riscv_vwredsum_vs_i8m1_i16m1(v_a_quant_i8, tmp_కూడిక, vl);
            int16_t    a_కూడిక   = __riscv_vmv_x_s_i16m1_i16(v_a_కూడిక);
            a_కూడిక_ptr[0]       = -a_కూడిక;

            __riscv_vse8_v_i8m1(quant_a_blk, v_a_quant_i8, vl);
        }
    } లేకపోతే {
        quantize_a_nrow_i8_ref<1>(blk_len, a_ptr, లెక్క_k, quant_a_ptr);
    }
}

శూన్య quantize_a_4row_i8(size_t blk_len, స్థిరరాశి దశాంశం * a_ptr, size_t లెక్క_k, uint8_t * quant_a_ptr) {
    GGML_ASSERT(blk_len == 32);
    int64_t a_blk_stride        = q8_blk_కొలత(blk_len, నిజం);
    int64_t a_nrow_ఖండం_stride = a_blk_stride * 4;
    size_t  vlenb               = __riscv_vlenb();

    ఒకవేళ (vlenb == 128) {
        కొరకు (size_t k = 0; k < లెక్క_k; k += blk_len, quant_a_ptr += a_nrow_ఖండం_stride) {
            దశాంశం *   scale_a_ptr = reinterpret_cast<దశాంశం *>(quant_a_ptr);
            int16_t * a_కూడిక_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + పరిమాణం(దశాంశం) * 4);
            int8_t *  quant_a_blk = reinterpret_cast<int8_t *>(quant_a_ptr + పరిమాణం(దశాంశం) * 4 + పరిమాణం(int16_t) * 4);

            కొరకు (size_t mi = 0; mi < 4; mi++) {
                size_t       vl      = __riscv_vsetvl_e32m1(blk_len);
                vfloat32m1_t v_a     = __riscv_vle32_v_f32m1(a_ptr + mi * లెక్క_k + k, vl);
                vfloat32m1_t v_a_abs = __riscv_vfabs_v_f32m1(v_a, vl);

                vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m1_f32m1(v_a_abs, tmp, vl);
                దశాంశం        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                దశాంశం scale_a     = max_abs_a / ((1 << 7) - 1);
                దశాంశం rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
                scale_a_ptr[mi]   = scale_a;

                vfloat32m1_t v_a_scale    = __riscv_vfmul_vf_f32m1(v_a, rep_scale_a, vl);
                vint16mf2_t  v_a_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a_scale, vl);
                vint8mf4_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a_quant, vl);

                vint16m1_t tmp_కూడిక = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a_కూడిక = __riscv_vwredsum_vs_i8mf4_i16m1(v_a_quant_i8, tmp_కూడిక, vl);
                int16_t    a_కూడిక   = __riscv_vmv_x_s_i16m1_i16(v_a_కూడిక);
                a_కూడిక_ptr[mi]      = -a_కూడిక;

                __riscv_vse8_v_i8mf4(quant_a_blk + mi * blk_len, v_a_quant_i8, vl);
            }
        }
    } లేకపోతే ఒకవేళ (vlenb == 32) {
        కొరకు (size_t k = 0; k < లెక్క_k; k += blk_len, quant_a_ptr += a_nrow_ఖండం_stride) {
            దశాంశం *   scale_a_ptr = reinterpret_cast<దశాంశం *>(quant_a_ptr);
            int16_t * a_కూడిక_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + పరిమాణం(దశాంశం) * 4);
            int8_t *  quant_a_blk = reinterpret_cast<int8_t *>(quant_a_ptr + పరిమాణం(దశాంశం) * 4 + పరిమాణం(int16_t) * 4);

            కొరకు (size_t mi = 0; mi < 4; mi++) {
                size_t       vl      = __riscv_vsetvl_e32m4(blk_len);
                vfloat32m4_t v_a     = __riscv_vle32_v_f32m4(a_ptr + mi * లెక్క_k + k, vl);
                vfloat32m4_t v_a_abs = __riscv_vfabs_v_f32m4(v_a, vl);

                vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m4_f32m1(v_a_abs, tmp, vl);
                దశాంశం        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                దశాంశం scale_a     = max_abs_a / ((1 << 7) - 1);
                దశాంశం rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
                scale_a_ptr[mi]   = scale_a;

                vfloat32m4_t v_a_scale    = __riscv_vfmul_vf_f32m4(v_a, rep_scale_a, vl);
                vint16m2_t   v_a_quant    = __riscv_vfncvt_x_f_w_i16m2(v_a_scale, vl);
                vint8m1_t    v_a_quant_i8 = __riscv_vncvt_x_x_w_i8m1(v_a_quant, vl);

                vint16m1_t tmp_కూడిక = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a_కూడిక = __riscv_vwredsum_vs_i8m1_i16m1(v_a_quant_i8, tmp_కూడిక, vl);
                int16_t    a_కూడిక   = __riscv_vmv_x_s_i16m1_i16(v_a_కూడిక);
                a_కూడిక_ptr[mi]      = -a_కూడిక;

                __riscv_vse8_v_i8m1(quant_a_blk + mi * blk_len, v_a_quant_i8, vl);
            }
        }
    } లేకపోతే {
        quantize_a_nrow_i8_ref<4>(blk_len, a_ptr, లెక్క_k, quant_a_ptr);
    }
}

శూన్య quantize_a_అడ్డువరుస_i8_hp(size_t blk_len, స్థిరరాశి దశాంశం * a_ptr, size_t లెక్క_k, uint8_t * quant_a_ptr) {
    constexpr size_t k_subblk_len = 32;
    GGML_ASSERT(blk_len == 256);

    constexpr size_t subblk_లెక్క             = 256 / k_subblk_len;
    int64_t          a_blk_stride             = q8_hp_blk_కొలత(blk_len, నిజం, నిజం);
    int64_t          a_subblk_stride          = q8_hp_blk_కొలత(k_subblk_len, అబద్ధం, అబద్ధం);
    size_t           vlenb                    = __riscv_vlenb();
    దశాంశం            scale_తాత్కాలిక[subblk_లెక్క] = { 0.0f };

    ఒకవేళ (vlenb == 128) {
        కొరకు (size_t k = 0; k < లెక్క_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            _Float16 * a_కూడిక_ptr     = reinterpret_cast<_Float16 *>(quant_a_ptr + a_subblk_stride * subblk_లెక్క);
            _Float16 * scale_avg_ptr = reinterpret_cast<_Float16 *>(quant_a_ptr + a_blk_stride - పరిమాణం(_Float16));
            దశాంశం      scale_avg     = 0.0f;

            కొరకు (size_t kk = 0; kk < subblk_లెక్క; ++kk) {
                స్థిరరాశి దశాంశం * a_src_ptr = a_ptr + k + kk * k_subblk_len;

                size_t       vl      = __riscv_vsetvl_e32m1(k_subblk_len);
                vfloat32m1_t v_a     = __riscv_vle32_v_f32m1(a_src_ptr, vl);
                vfloat32m1_t v_a_abs = __riscv_vfabs_v_f32m1(v_a, vl);

                vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m1_f32m1(v_a_abs, tmp, vl);
                దశాంశం        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                scale_తాత్కాలిక[kk] = max_abs_a / ((1 << 7) - 1);
                scale_avg += scale_తాత్కాలిక[kk];
            }

            scale_avg /= subblk_లెక్క;
            స్థిరరాశి దశాంశం scale_గుణకం = scale_avg ? 1.0f / scale_avg : 0.0f;
            scale_avg_ptr[0]         = static_cast<_Float16>(scale_avg);

            కొరకు (size_t kk = 0; kk < subblk_లెక్క; ++kk) {
                uint8_t *     a_subblk_పునాది = quant_a_ptr + kk * a_subblk_stride;
                _Float16 *    scale_a_ptr   = reinterpret_cast<_Float16 *>(a_subblk_పునాది);
                int8_t *      quant_a_blk   = reinterpret_cast<int8_t *>(a_subblk_పునాది + పరిమాణం(_Float16));
                స్థిరరాశి దశాంశం * a_src_ptr     = a_ptr + k + kk * k_subblk_len;

                size_t       vl          = __riscv_vsetvl_e32m1(k_subblk_len);
                vfloat32m1_t v_a         = __riscv_vle32_v_f32m1(a_src_ptr, vl);
                దశాంశం        rep_scale_a = scale_తాత్కాలిక[kk] ? 1.0f / scale_తాత్కాలిక[kk] : 0.0f;
                scale_a_ptr[0]           = static_cast<_Float16>(scale_తాత్కాలిక[kk] * scale_గుణకం);

                vfloat32m1_t v_a_scale    = __riscv_vfmul_vf_f32m1(v_a, rep_scale_a, vl);
                vint16mf2_t  v_a_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a_scale, vl);
                vint8mf4_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a_quant, vl);

                vint16m1_t tmp_కూడిక = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a_కూడిక = __riscv_vwredsum_vs_i8mf4_i16m1(v_a_quant_i8, tmp_కూడిక, vl);
                int16_t    a_కూడిక   = __riscv_vmv_x_s_i16m1_i16(v_a_కూడిక);
                a_కూడిక_ptr[kk]      = static_cast<_Float16>(-a_కూడిక) * static_cast<_Float16>(8.0f);

                __riscv_vse8_v_i8mf4(quant_a_blk, v_a_quant_i8, vl);
            }
        }
    } లేకపోతే ఒకవేళ (vlenb == 32) {
        కొరకు (size_t k = 0; k < లెక్క_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            _Float16 * a_కూడిక_ptr     = reinterpret_cast<_Float16 *>(quant_a_ptr + a_subblk_stride * subblk_లెక్క);
            _Float16 * scale_avg_ptr = reinterpret_cast<_Float16 *>(quant_a_ptr + a_blk_stride - పరిమాణం(_Float16));
            దశాంశం      scale_avg     = 0.0f;

            కొరకు (size_t kk = 0; kk < subblk_లెక్క; ++kk) {
                స్థిరరాశి దశాంశం * a_src_ptr = a_ptr + k + kk * k_subblk_len;

                size_t       vl      = __riscv_vsetvl_e32m4(k_subblk_len);
                vfloat32m4_t v_a     = __riscv_vle32_v_f32m4(a_src_ptr, vl);
                vfloat32m4_t v_a_abs = __riscv_vfabs_v_f32m4(v_a, vl);

                vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m4_f32m1(v_a_abs, tmp, vl);
                దశాంశం        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                scale_తాత్కాలిక[kk] = max_abs_a / ((1 << 7) - 1);
                scale_avg += scale_తాత్కాలిక[kk];
            }

            scale_avg /= subblk_లెక్క;
            స్థిరరాశి దశాంశం scale_గుణకం = scale_avg ? 1.0f / scale_avg : 0.0f;
            scale_avg_ptr[0]         = static_cast<_Float16>(scale_avg);

            కొరకు (size_t kk = 0; kk < subblk_లెక్క; ++kk) {
                uint8_t *     a_subblk_పునాది = quant_a_ptr + kk * a_subblk_stride;
                _Float16 *    scale_a_ptr   = reinterpret_cast<_Float16 *>(a_subblk_పునాది);
                int8_t *      quant_a_blk   = reinterpret_cast<int8_t *>(a_subblk_పునాది + పరిమాణం(_Float16));
                స్థిరరాశి దశాంశం * a_src_ptr     = a_ptr + k + kk * k_subblk_len;

                size_t       vl          = __riscv_vsetvl_e32m4(k_subblk_len);
                vfloat32m4_t v_a         = __riscv_vle32_v_f32m4(a_src_ptr, vl);
                దశాంశం        rep_scale_a = scale_తాత్కాలిక[kk] ? 1.0f / scale_తాత్కాలిక[kk] : 0.0f;
                scale_a_ptr[0]           = static_cast<_Float16>(scale_తాత్కాలిక[kk] * scale_గుణకం);

                vfloat32m4_t v_a_scale    = __riscv_vfmul_vf_f32m4(v_a, rep_scale_a, vl);
                vint16m2_t   v_a_quant    = __riscv_vfncvt_x_f_w_i16m2(v_a_scale, vl);
                vint8m1_t    v_a_quant_i8 = __riscv_vncvt_x_x_w_i8m1(v_a_quant, vl);

                vint16m1_t tmp_కూడిక = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a_కూడిక = __riscv_vwredsum_vs_i8m1_i16m1(v_a_quant_i8, tmp_కూడిక, vl);
                int16_t    a_కూడిక   = __riscv_vmv_x_s_i16m1_i16(v_a_కూడిక);
                a_కూడిక_ptr[kk]      = static_cast<_Float16>(-a_కూడిక) * static_cast<_Float16>(8.0f);

                __riscv_vse8_v_i8m1(quant_a_blk, v_a_quant_i8, vl);
            }
        }
    } లేకపోతే {
        quantize_a_nrow_i8_hp_ref<1>(blk_len, a_ptr, లెక్క_k, quant_a_ptr);
    }
}

శూన్య quantize_a_4row_i8_hp(size_t blk_len, స్థిరరాశి దశాంశం * a_ptr, size_t లెక్క_k, uint8_t * quant_a_ptr) {
    constexpr size_t k_subblk_len = 32;
    GGML_ASSERT(blk_len == 256);

    constexpr size_t subblk_లెక్క             = 256 / k_subblk_len;
    int64_t          a_blk_stride             = q8_hp_blk_కొలత(blk_len, నిజం, నిజం);
    int64_t          a_nrow_ఖండం_stride      = a_blk_stride * 4;
    int64_t          a_subblk_stride          = q8_hp_blk_కొలత(k_subblk_len, అబద్ధం, అబద్ధం) * 4;
    size_t           vlenb                    = __riscv_vlenb();
    దశాంశం            scale_తాత్కాలిక[subblk_లెక్క] = { 0.0f };

    ఒకవేళ (vlenb == 128) {
        కొరకు (size_t k = 0; k < లెక్క_k; k += blk_len, quant_a_ptr += a_nrow_ఖండం_stride) {
            _Float16 * a_కూడిక_ptr = reinterpret_cast<_Float16 *>(quant_a_ptr + a_subblk_stride * subblk_లెక్క);
            _Float16 * scale_avg_ptr =
                reinterpret_cast<_Float16 *>(quant_a_ptr + a_nrow_ఖండం_stride - పరిమాణం(_Float16) * 4);
            దశాంశం scale_avg = 0.0f;

            కొరకు (size_t kk = 0; kk < subblk_లెక్క; ++kk) {
                స్థిరరాశి దశాంశం * a_src_ptr0 = a_ptr + 0 * లెక్క_k + k + kk * k_subblk_len;
                స్థిరరాశి దశాంశం * a_src_ptr1 = a_ptr + 1 * లెక్క_k + k + kk * k_subblk_len;
                స్థిరరాశి దశాంశం * a_src_ptr2 = a_ptr + 2 * లెక్క_k + k + kk * k_subblk_len;
                స్థిరరాశి దశాంశం * a_src_ptr3 = a_ptr + 3 * లెక్క_k + k + kk * k_subblk_len;

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
                దశాంశం        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                scale_తాత్కాలిక[kk] = max_abs_a / ((1 << 7) - 1);
                scale_avg += scale_తాత్కాలిక[kk];
            }

            scale_avg /= subblk_లెక్క;
            స్థిరరాశి దశాంశం scale_గుణకం = scale_avg ? 1.0f / scale_avg : 0.0f;
            scale_avg_ptr[0]         = static_cast<_Float16>(scale_avg);

            కొరకు (size_t kk = 0; kk < subblk_లెక్క; ++kk) {
                uint8_t *     a_subblk_పునాది = quant_a_ptr + kk * a_subblk_stride;
                _Float16 *    scale_a_ptr   = reinterpret_cast<_Float16 *>(a_subblk_పునాది);
                int8_t *      quant_a_blk   = reinterpret_cast<int8_t *>(a_subblk_పునాది + పరిమాణం(_Float16) * 4);
                స్థిరరాశి దశాంశం * a_src_ptr0    = a_ptr + 0 * లెక్క_k + k + kk * k_subblk_len;
                స్థిరరాశి దశాంశం * a_src_ptr1    = a_ptr + 1 * లెక్క_k + k + kk * k_subblk_len;
                స్థిరరాశి దశాంశం * a_src_ptr2    = a_ptr + 2 * లెక్క_k + k + kk * k_subblk_len;
                స్థిరరాశి దశాంశం * a_src_ptr3    = a_ptr + 3 * లెక్క_k + k + kk * k_subblk_len;

                size_t       vl   = __riscv_vsetvl_e32m1(k_subblk_len);
                vfloat32m1_t v_a0 = __riscv_vle32_v_f32m1(a_src_ptr0, vl);
                vfloat32m1_t v_a1 = __riscv_vle32_v_f32m1(a_src_ptr1, vl);
                vfloat32m1_t v_a2 = __riscv_vle32_v_f32m1(a_src_ptr2, vl);
                vfloat32m1_t v_a3 = __riscv_vle32_v_f32m1(a_src_ptr3, vl);

                దశాంశం rep_scale_a = scale_తాత్కాలిక[kk] ? 1.0f / scale_తాత్కాలిక[kk] : 0.0f;
                scale_a_ptr[0]    = static_cast<_Float16>(scale_తాత్కాలిక[kk] * scale_గుణకం);

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
                vint16m1_t v_a0_కూడిక = __riscv_vwredsum_vs_i8mf4_i16m1(v_a0_quant_i8, tmp_sum0, vl);
                vint16m1_t v_a1_కూడిక = __riscv_vwredsum_vs_i8mf4_i16m1(v_a1_quant_i8, tmp_sum1, vl);
                vint16m1_t v_a2_కూడిక = __riscv_vwredsum_vs_i8mf4_i16m1(v_a2_quant_i8, tmp_sum2, vl);
                vint16m1_t v_a3_కూడిక = __riscv_vwredsum_vs_i8mf4_i16m1(v_a3_quant_i8, tmp_sum3, vl);

                a_కూడిక_ptr[0 * subblk_లెక్క + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a0_కూడిక)) * static_cast<_Float16>(8.0f);
                a_కూడిక_ptr[1 * subblk_లెక్క + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a1_కూడిక)) * static_cast<_Float16>(8.0f);
                a_కూడిక_ptr[2 * subblk_లెక్క + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a2_కూడిక)) * static_cast<_Float16>(8.0f);
                a_కూడిక_ptr[3 * subblk_లెక్క + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a3_కూడిక)) * static_cast<_Float16>(8.0f);

                __riscv_vse8_v_i8mf4(quant_a_blk + 0 * k_subblk_len, v_a0_quant_i8, vl);
                __riscv_vse8_v_i8mf4(quant_a_blk + 1 * k_subblk_len, v_a1_quant_i8, vl);
                __riscv_vse8_v_i8mf4(quant_a_blk + 2 * k_subblk_len, v_a2_quant_i8, vl);
                __riscv_vse8_v_i8mf4(quant_a_blk + 3 * k_subblk_len, v_a3_quant_i8, vl);
            }
        }
    } లేకపోతే ఒకవేళ (vlenb == 32) {
        కొరకు (size_t k = 0; k < లెక్క_k; k += blk_len, quant_a_ptr += a_nrow_ఖండం_stride) {
            _Float16 * a_కూడిక_ptr = reinterpret_cast<_Float16 *>(quant_a_ptr + a_subblk_stride * subblk_లెక్క);
            _Float16 * scale_avg_ptr =
                reinterpret_cast<_Float16 *>(quant_a_ptr + a_nrow_ఖండం_stride - పరిమాణం(_Float16) * 4);
            దశాంశం scale_avg = 0.0f;

            కొరకు (size_t kk = 0; kk < subblk_లెక్క; ++kk) {
                స్థిరరాశి దశాంశం * a_src_ptr0 = a_ptr + 0 * లెక్క_k + k + kk * k_subblk_len;
                స్థిరరాశి దశాంశం * a_src_ptr1 = a_ptr + 1 * లెక్క_k + k + kk * k_subblk_len;
                స్థిరరాశి దశాంశం * a_src_ptr2 = a_ptr + 2 * లెక్క_k + k + kk * k_subblk_len;
                స్థిరరాశి దశాంశం * a_src_ptr3 = a_ptr + 3 * లెక్క_k + k + kk * k_subblk_len;

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
                దశాంశం        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                scale_తాత్కాలిక[kk] = max_abs_a / ((1 << 7) - 1);
                scale_avg += scale_తాత్కాలిక[kk];
            }

            scale_avg /= subblk_లెక్క;
            స్థిరరాశి దశాంశం scale_గుణకం = scale_avg ? 1.0f / scale_avg : 0.0f;
            scale_avg_ptr[0]         = static_cast<_Float16>(scale_avg);

            కొరకు (size_t kk = 0; kk < subblk_లెక్క; ++kk) {
                uint8_t *     a_subblk_పునాది = quant_a_ptr + kk * a_subblk_stride;
                _Float16 *    scale_a_ptr   = reinterpret_cast<_Float16 *>(a_subblk_పునాది);
                int8_t *      quant_a_blk   = reinterpret_cast<int8_t *>(a_subblk_పునాది + పరిమాణం(_Float16) * 4);
                స్థిరరాశి దశాంశం * a_src_ptr0    = a_ptr + 0 * లెక్క_k + k + kk * k_subblk_len;
                స్థిరరాశి దశాంశం * a_src_ptr1    = a_ptr + 1 * లెక్క_k + k + kk * k_subblk_len;
                స్థిరరాశి దశాంశం * a_src_ptr2    = a_ptr + 2 * లెక్క_k + k + kk * k_subblk_len;
                స్థిరరాశి దశాంశం * a_src_ptr3    = a_ptr + 3 * లెక్క_k + k + kk * k_subblk_len;

                size_t       vl   = __riscv_vsetvl_e32m4(k_subblk_len);
                vfloat32m4_t v_a0 = __riscv_vle32_v_f32m4(a_src_ptr0, vl);
                vfloat32m4_t v_a1 = __riscv_vle32_v_f32m4(a_src_ptr1, vl);
                vfloat32m4_t v_a2 = __riscv_vle32_v_f32m4(a_src_ptr2, vl);
                vfloat32m4_t v_a3 = __riscv_vle32_v_f32m4(a_src_ptr3, vl);

                దశాంశం rep_scale_a = scale_తాత్కాలిక[kk] ? 1.0f / scale_తాత్కాలిక[kk] : 0.0f;
                scale_a_ptr[0]    = static_cast<_Float16>(scale_తాత్కాలిక[kk] * scale_గుణకం);

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
                vint16m1_t v_a0_కూడిక = __riscv_vwredsum_vs_i8m1_i16m1(v_a0_quant_i8, tmp_sum0, vl);
                vint16m1_t v_a1_కూడిక = __riscv_vwredsum_vs_i8m1_i16m1(v_a1_quant_i8, tmp_sum1, vl);
                vint16m1_t v_a2_కూడిక = __riscv_vwredsum_vs_i8m1_i16m1(v_a2_quant_i8, tmp_sum2, vl);
                vint16m1_t v_a3_కూడిక = __riscv_vwredsum_vs_i8m1_i16m1(v_a3_quant_i8, tmp_sum3, vl);

                a_కూడిక_ptr[0 * subblk_లెక్క + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a0_కూడిక)) * static_cast<_Float16>(8.0f);
                a_కూడిక_ptr[1 * subblk_లెక్క + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a1_కూడిక)) * static_cast<_Float16>(8.0f);
                a_కూడిక_ptr[2 * subblk_లెక్క + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a2_కూడిక)) * static_cast<_Float16>(8.0f);
                a_కూడిక_ptr[3 * subblk_లెక్క + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a3_కూడిక)) * static_cast<_Float16>(8.0f);

                __riscv_vse8_v_i8m1(quant_a_blk + 0 * k_subblk_len, v_a0_quant_i8, vl);
                __riscv_vse8_v_i8m1(quant_a_blk + 1 * k_subblk_len, v_a1_quant_i8, vl);
                __riscv_vse8_v_i8m1(quant_a_blk + 2 * k_subblk_len, v_a2_quant_i8, vl);
                __riscv_vse8_v_i8m1(quant_a_blk + 3 * k_subblk_len, v_a3_quant_i8, vl);
            }
        }
    } లేకపోతే {
        quantize_a_nrow_i8_hp_ref<4>(blk_len, a_ptr, లెక్క_k, quant_a_ptr);
    }
}

శూన్య quantize_a_అడ్డువరుస_i8k(size_t blk_len, స్థిరరాశి దశాంశం * a_ptr, size_t లెక్క_k, uint8_t * quant_a_ptr) {
    GGML_ASSERT(blk_len == 256);
    constexpr int64_t a_blk_stride = q8k_blk_కొలత(256);
    constexpr int64_t a_కూడిక_కొలత   = 256 / 16;
    size_t            vlenb        = __riscv_vlenb();

    ఒకవేళ (vlenb == 128) {
        // vlen = 1024 bits, can process 32 float32 elements with m1
        కొరకు (size_t k = 0; k < లెక్క_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            దశాంశం *   scale_a_ptr = reinterpret_cast<దశాంశం *>(quant_a_ptr);
            int16_t * a_కూడిక_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + పరిమాణం(దశాంశం));
            int8_t *  quant_a_blk =
                reinterpret_cast<int8_t *>(quant_a_ptr + పరిమాణం(దశాంశం) + పరిమాణం(int16_t) * a_కూడిక_కొలత);

            // Find max absolute value across all 256 elements
            size_t       vl        = __riscv_vsetvl_e32m1(16);
            vfloat32m1_t v_max_abs = __riscv_vfmv_v_f_f32m1(0.0f, vl);

            కొరకు (size_t bki = 0; bki < a_కూడిక_కొలత; bki++) {
                vfloat32m1_t v_a     = __riscv_vle32_v_f32m1(a_ptr + k + bki * 16, vl);
                vfloat32m1_t v_a_abs = __riscv_vfabs_v_f32m1(v_a, vl);
                v_max_abs            = __riscv_vfmax_vv_f32m1(v_a_abs, v_max_abs, vl);
            }
            vfloat32m1_t tmp         = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t v_local_max = __riscv_vfredmax_vs_f32m1_f32m1(v_max_abs, tmp, vl);
            దశాంశం        max_abs_a   = __riscv_vfmv_f_s_f32m1_f32(v_local_max);

            దశాంశం scale_a     = max_abs_a / ((1 << 7) - 1);
            దశాంశం rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
            scale_a_ptr[0]    = scale_a;

            // Quantize and compute sums for each 16-element group
            కొరకు (size_t bki = 0; bki < a_కూడిక_కొలత; bki++) {
                vfloat32m1_t v_a          = __riscv_vle32_v_f32m1(a_ptr + k + bki * 16, vl);
                vfloat32m1_t v_a_scale    = __riscv_vfmul_vf_f32m1(v_a, rep_scale_a, vl);
                vint16mf2_t  v_a_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a_scale, vl);
                vint8mf4_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a_quant, vl);

                vint16m1_t tmp_కూడిక = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a_కూడిక = __riscv_vwredsum_vs_i8mf4_i16m1(v_a_quant_i8, tmp_కూడిక, vl);
                int16_t    a_కూడిక   = __riscv_vmv_x_s_i16m1_i16(v_a_కూడిక);
                a_కూడిక_ptr[bki]     = -a_కూడిక;

                __riscv_vse8_v_i8mf4(quant_a_blk + bki * 16, v_a_quant_i8, vl);
            }
        }
    } లేకపోతే ఒకవేళ (vlenb == 32) {
        // vlen = 256 bits, can process 8 float32 elements with m1
        కొరకు (size_t k = 0; k < లెక్క_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            దశాంశం *   scale_a_ptr = reinterpret_cast<దశాంశం *>(quant_a_ptr);
            int16_t * a_కూడిక_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + పరిమాణం(దశాంశం));
            int8_t *  quant_a_blk =
                reinterpret_cast<int8_t *>(quant_a_ptr + పరిమాణం(దశాంశం) + పరిమాణం(int16_t) * a_కూడిక_కొలత);

            // Find max absolute value across all 256 elements
            size_t       vl        = __riscv_vsetvl_e32m2(16);
            vfloat32m2_t v_max_abs = __riscv_vfmv_v_f_f32m2(0.0f, vl);

            కొరకు (size_t bki = 0; bki < a_కూడిక_కొలత; bki++) {
                vfloat32m2_t v_a     = __riscv_vle32_v_f32m2(a_ptr + k + bki * 16, vl);
                vfloat32m2_t v_a_abs = __riscv_vfabs_v_f32m2(v_a, vl);
                v_max_abs            = __riscv_vfmax_vv_f32m2(v_a_abs, v_max_abs, vl);
            }
            vfloat32m1_t tmp         = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t v_local_max = __riscv_vfredmax_vs_f32m2_f32m1(v_max_abs, tmp, vl);
            దశాంశం        max_abs_a   = __riscv_vfmv_f_s_f32m1_f32(v_local_max);

            దశాంశం scale_a     = max_abs_a / ((1 << 7) - 1);
            దశాంశం rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
            scale_a_ptr[0]    = scale_a;

            // Quantize and compute sums for each 16-element group
            కొరకు (size_t bki = 0; bki < a_కూడిక_కొలత; bki++) {
                vfloat32m2_t v_a          = __riscv_vle32_v_f32m2(a_ptr + k + bki * 16, vl);
                vfloat32m2_t v_a_scale    = __riscv_vfmul_vf_f32m2(v_a, rep_scale_a, vl);
                vint16m1_t   v_a_quant    = __riscv_vfncvt_x_f_w_i16m1(v_a_scale, vl);
                vint8mf2_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf2(v_a_quant, vl);

                vint16m1_t tmp_కూడిక = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a_కూడిక = __riscv_vwredsum_vs_i8mf2_i16m1(v_a_quant_i8, tmp_కూడిక, vl);
                int16_t    a_కూడిక   = __riscv_vmv_x_s_i16m1_i16(v_a_కూడిక);
                a_కూడిక_ptr[bki]     = -a_కూడిక;

                __riscv_vse8_v_i8mf2(quant_a_blk + bki * 16, v_a_quant_i8, vl);
            }
        }
    } లేకపోతే {
        quantize_a_nrow_i8k_ref<1>(blk_len, a_ptr, లెక్క_k, quant_a_ptr);
    }
}

శూన్య quantize_a_4row_i8k(size_t blk_len, స్థిరరాశి దశాంశం * a_ptr, size_t లెక్క_k, uint8_t * quant_a_ptr) {
    GGML_ASSERT(blk_len == 256);
    constexpr int64_t a_blk_stride        = q8k_blk_కొలత(256);
    constexpr int64_t a_nrow_ఖండం_stride = a_blk_stride * 4;
    constexpr int64_t a_కూడిక_కొలత          = 256 / 16;
    size_t            vlenb               = __riscv_vlenb();

    ఒకవేళ (vlenb == 128) {
        // vlen = 1024 bits
        కొరకు (size_t k = 0; k < లెక్క_k; k += blk_len, quant_a_ptr += a_nrow_ఖండం_stride) {
            దశాంశం *   scale_a_ptr = reinterpret_cast<దశాంశం *>(quant_a_ptr);
            int16_t * a_కూడిక_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + పరిమాణం(దశాంశం) * 4);
            int8_t *  quant_a_blk =
                reinterpret_cast<int8_t *>(quant_a_ptr + పరిమాణం(దశాంశం) * 4 + పరిమాణం(int16_t) * a_కూడిక_కొలత * 4);

            కొరకు (size_t mi = 0; mi < 4; mi++) {
                // Find max absolute value across all 256 elements for this row
                size_t       vl        = __riscv_vsetvl_e32m1(16);
                vfloat32m1_t v_max_abs = __riscv_vfmv_v_f_f32m1(0.0f, vl);

                కొరకు (size_t bki = 0; bki < a_కూడిక_కొలత; bki++) {
                    vfloat32m1_t v_a     = __riscv_vle32_v_f32m1(a_ptr + mi * లెక్క_k + k + bki * 16, vl);
                    vfloat32m1_t v_a_abs = __riscv_vfabs_v_f32m1(v_a, vl);
                    v_max_abs            = __riscv_vfmax_vv_f32m1(v_a_abs, v_max_abs, vl);
                }
                vfloat32m1_t tmp         = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_local_max = __riscv_vfredmax_vs_f32m1_f32m1(v_max_abs, tmp, vl);
                దశాంశం        max_abs_a   = __riscv_vfmv_f_s_f32m1_f32(v_local_max);

                దశాంశం scale_a     = max_abs_a / ((1 << 7) - 1);
                దశాంశం rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
                scale_a_ptr[mi]   = scale_a;

                // Quantize and compute sums for each 16-element group
                కొరకు (size_t bki = 0; bki < a_కూడిక_కొలత; bki++) {
                    vfloat32m1_t v_a          = __riscv_vle32_v_f32m1(a_ptr + mi * లెక్క_k + k + bki * 16, vl);
                    vfloat32m1_t v_a_scale    = __riscv_vfmul_vf_f32m1(v_a, rep_scale_a, vl);
                    vint16mf2_t  v_a_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a_scale, vl);
                    vint8mf4_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a_quant, vl);

                    vint16m1_t tmp_కూడిక               = __riscv_vmv_v_x_i16m1(0, vl);
                    vint16m1_t v_a_కూడిక               = __riscv_vwredsum_vs_i8mf4_i16m1(v_a_quant_i8, tmp_కూడిక, vl);
                    int16_t    a_కూడిక                 = __riscv_vmv_x_s_i16m1_i16(v_a_కూడిక);
                    a_కూడిక_ptr[mi * a_కూడిక_కొలత + bki] = -a_కూడిక;

                    __riscv_vse8_v_i8mf4(quant_a_blk + mi * blk_len + bki * 16, v_a_quant_i8, vl);
                }
            }
        }
    } లేకపోతే ఒకవేళ (vlenb == 32) {
        // vlen = 256 bits
        కొరకు (size_t k = 0; k < లెక్క_k; k += blk_len, quant_a_ptr += a_nrow_ఖండం_stride) {
            దశాంశం *   scale_a_ptr = reinterpret_cast<దశాంశం *>(quant_a_ptr);
            int16_t * a_కూడిక_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + పరిమాణం(దశాంశం) * 4);
            int8_t *  quant_a_blk =
                reinterpret_cast<int8_t *>(quant_a_ptr + పరిమాణం(దశాంశం) * 4 + పరిమాణం(int16_t) * a_కూడిక_కొలత * 4);

            కొరకు (size_t mi = 0; mi < 4; mi++) {
                // Find max absolute value across all 256 elements for this row
                size_t       vl        = __riscv_vsetvl_e32m2(16);
                vfloat32m2_t v_max_abs = __riscv_vfmv_v_f_f32m2(0.0f, vl);

                కొరకు (size_t bki = 0; bki < a_కూడిక_కొలత; bki++) {
                    vfloat32m2_t v_a     = __riscv_vle32_v_f32m2(a_ptr + mi * లెక్క_k + k + bki * 16, vl);
                    vfloat32m2_t v_a_abs = __riscv_vfabs_v_f32m2(v_a, vl);
                    v_max_abs            = __riscv_vfmax_vv_f32m2(v_a_abs, v_max_abs, vl);
                }
                vfloat32m1_t tmp         = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_local_max = __riscv_vfredmax_vs_f32m2_f32m1(v_max_abs, tmp, vl);
                దశాంశం        max_abs_a   = __riscv_vfmv_f_s_f32m1_f32(v_local_max);

                దశాంశం scale_a     = max_abs_a / ((1 << 7) - 1);
                దశాంశం rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
                scale_a_ptr[mi]   = scale_a;

                // Quantize and compute sums for each 16-element group
                కొరకు (size_t bki = 0; bki < a_కూడిక_కొలత; bki++) {
                    vfloat32m2_t v_a          = __riscv_vle32_v_f32m2(a_ptr + mi * లెక్క_k + k + bki * 16, vl);
                    vfloat32m2_t v_a_scale    = __riscv_vfmul_vf_f32m2(v_a, rep_scale_a, vl);
                    vint16m1_t   v_a_quant    = __riscv_vfncvt_x_f_w_i16m1(v_a_scale, vl);
                    vint8mf2_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf2(v_a_quant, vl);

                    vint16m1_t tmp_కూడిక               = __riscv_vmv_v_x_i16m1(0, vl);
                    vint16m1_t v_a_కూడిక               = __riscv_vwredsum_vs_i8mf2_i16m1(v_a_quant_i8, tmp_కూడిక, vl);
                    int16_t    a_కూడిక                 = __riscv_vmv_x_s_i16m1_i16(v_a_కూడిక);
                    a_కూడిక_ptr[mi * a_కూడిక_కొలత + bki] = -a_కూడిక;

                    __riscv_vse8_v_i8mf2(quant_a_blk + mi * blk_len + bki * 16, v_a_quant_i8, vl);
                }
            }
        }
    } లేకపోతే {
        quantize_a_nrow_i8k_ref<4>(blk_len, a_ptr, లెక్క_k, quant_a_ptr);
    }
}

శూన్య forward_cpy_with_క్రమమార్పు(ggml_compute_params * params, ggml_tensor * op) {
    స్థిరరాశి ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;
    స్థిరరాశి పూర్ణాంకం           ith  = params->ith;
    స్థిరరాశి పూర్ణాంకం           nth  = params->nth;

    // [batch, m, n] -> [batch, n, m]
    int64_t batch = src0->ne[2] * src0->ne[3];
    int64_t m     = src0->ne[1];
    int64_t n     = src0->ne[0];

    int64_t batch_stride = src0->nb[2];
    int64_t m_src_stride = src0->nb[0];
    int64_t n_src_stride = src0->nb[1];
    int64_t n_dst_stride = n_src_stride * m;

    క్రమమార్పు_transpose_impl(src0, dst, batch, m, n, batch_stride, m_src_stride, n_src_stride, n_dst_stride, ith, nth);
}

శూన్య forward_cont_with_క్రమమార్పు(ggml_compute_params * params, ggml_tensor * op) {
    స్థిరరాశి ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;
    స్థిరరాశి పూర్ణాంకం           ith  = params->ith;
    స్థిరరాశి పూర్ణాంకం           nth  = params->nth;

    // [batch, m, n] -> [batch, n, m]
    int64_t batch = dst->ne[2] * dst->ne[3];
    int64_t n     = dst->ne[1];
    int64_t m     = dst->ne[0];

    int64_t batch_stride = dst->nb[2];
    int64_t m_src_stride = src0->nb[0];
    int64_t n_src_stride = src0->nb[1];
    int64_t n_dst_stride = dst->nb[1];

    క్రమమార్పు_transpose_impl(src0, dst, batch, m, n, batch_stride, m_src_stride, n_src_stride, n_dst_stride, ith, nth);
}

శూన్య forward_norm_f32(ggml_compute_params * params, ggml_tensor * op) {
    స్థిరరాశి ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(src0->nb[0] == పరిమాణం(దశాంశం));

    పూర్ణాంకం ith = params->ith;
    పూర్ణాంకం nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    దశాంశం epsilon = *((దశాంశం *) dst->op_params);

    GGML_ASSERT(epsilon > 0.0f);

    స్వయంచాలక * ఆదానం  = (అక్షరం *) src0->దత్తాంశం;
    స్వయంచాలక * output = (అక్షరం *) dst->దత్తాంశం;

    స్థిరరాశి స్వయంచాలక hidden_కొలత     = ne00;
    స్థిరరాశి స్వయంచాలక task_లెక్క      = ne01 * ne02 * ne03;
    స్థిరరాశి స్వయంచాలక task_చొప్పున_thread = (task_లెక్క + nth - 1) / nth;

    స్థిరరాశి స్వయంచాలక task_begin = ith * task_చొప్పున_thread;
    స్థిరరాశి స్వయంచాలక task_చివర   = std::min((ith + 1) * task_చొప్పున_thread, task_లెక్క);

    కొరకు (స్వయంచాలక task_idx = task_begin; task_idx < task_చివర; task_idx++) {
        int64_t i03 = task_idx / (ne02 * ne01);
        int64_t i02 = (task_idx - i03 * ne02 * ne01) / ne01;
        int64_t i01 = (task_idx - i03 * ne02 * ne01 - i02 * ne01);

        స్వయంచాలక * p_ఆదానం       = (దశాంశం *) (ఆదానం + i01 * nb01 + i02 * nb02 + i03 * nb03);
        స్వయంచాలక * p_output      = (దశాంశం *) (output + i01 * nb1 + i02 * nb2 + i03 * nb3);
        స్వయంచాలక * p_తాత్కాలిక_output = p_output;

        size_t       gvl    = __riscv_vsetvlmax_e32m4();
        vfloat32m4_t కూడిక    = __riscv_vfmv_v_f_f32m4(0.f, gvl);
        vfloat32m4_t కూడిక_sq = __riscv_vfmv_v_f_f32m4(0.f, gvl);
        int64_t      నిడివి = hidden_కొలత;
        అయితే (నిడివి > 0) {
            gvl                   = __riscv_vsetvl_e32m4(నిడివి);
            // load data
            vfloat32m4_t src_దత్తాంశం = __riscv_vle32_v_f32m4(p_ఆదానం, gvl);

            కూడిక    = __riscv_vfadd_vv_f32m4(కూడిక, src_దత్తాంశం, gvl);
            కూడిక_sq = __riscv_vfmacc_vv_f32m4(కూడిక_sq, src_దత్తాంశం, src_దత్తాంశం, gvl);

            __riscv_vse32_v_f32m4(p_తాత్కాలిక_output, src_దత్తాంశం, gvl);

            p_ఆదానం += gvl;
            p_తాత్కాలిక_output += gvl;
            నిడివి -= gvl;
        }

        gvl = __riscv_vsetvlmax_e32m1();

        దశాంశం        సగటు_2   = 0.f;
        vfloat32m1_t zero_v = __riscv_vfmv_v_f_f32m1(0.f, gvl);
        vfloat32m1_t సగటు_v =
            __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(కూడిక, 0), __riscv_vget_v_f32m4_f32m1(కూడిక, 1), gvl);
        సగటు_v = __riscv_vfadd_vv_f32m1(సగటు_v, __riscv_vget_v_f32m4_f32m1(కూడిక, 2), gvl);
        సగటు_v = __riscv_vfadd_vv_f32m1(సగటు_v, __riscv_vget_v_f32m4_f32m1(కూడిక, 3), gvl);
        సగటు_v = __riscv_vfredusum_vs_f32m1_f32m1(సగటు_v, zero_v, gvl);
        సగటు_2   = __riscv_vfmv_f_s_f32m1_f32(సగటు_v);
        సగటు_2 /= hidden_కొలత;

        vfloat32m1_t సగటు_వర్గం_v =
            __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(కూడిక_sq, 0), __riscv_vget_v_f32m4_f32m1(కూడిక_sq, 1), gvl);
        సగటు_వర్గం_v = __riscv_vfadd_vv_f32m1(సగటు_వర్గం_v, __riscv_vget_v_f32m4_f32m1(కూడిక_sq, 2), gvl);
        సగటు_వర్గం_v = __riscv_vfadd_vv_f32m1(సగటు_వర్గం_v, __riscv_vget_v_f32m4_f32m1(కూడిక_sq, 3), gvl);
        సగటు_వర్గం_v = __riscv_vfredusum_vs_f32m1_f32m1(సగటు_వర్గం_v, zero_v, gvl);

        దశాంశం సగటు_వర్గం = __riscv_vfmv_f_s_f32m1_f32(సగటు_వర్గం_v);
        సగటు_వర్గం /= hidden_కొలత;
        సగటు_వర్గం = వర్గమూలం(సగటు_వర్గం - సగటు_2 * సగటు_2 + epsilon);

        సగటు_వర్గం   = 1.0f / సగటు_వర్గం;
        నిడివి        = hidden_కొలత;
        p_తాత్కాలిక_output = p_output;

        అయితే (నిడివి > 0) {
            gvl                   = __riscv_vsetvl_e32m4(నిడివి);
            vfloat32m4_t src_దత్తాంశం = __riscv_vle32_v_f32m4(p_తాత్కాలిక_output, gvl);
            src_దత్తాంశం              = __riscv_vfsub_vf_f32m4(src_దత్తాంశం, సగటు_2, gvl);
            src_దత్తాంశం              = __riscv_vfmul_vf_f32m4(src_దత్తాంశం, సగటు_వర్గం, gvl);
            __riscv_vse32_v_f32m4(p_output, src_దత్తాంశం, gvl);
            p_తాత్కాలిక_output += gvl;
            p_output += gvl;
            నిడివి -= gvl;
        }
    }
}

మూస <ggml_op op_type, రకం_పేరు T> శూన్య forward_ద్విమాన(ggml_compute_params * params, ggml_tensor * op) {
    స్థిరరాశి ggml_tensor * src0 = op->src[0];
    స్థిరరాశి ggml_tensor * src1 = op->src[1];
    ggml_tensor *       dst  = op;
    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    స్వయంచాలక src0_అడ్డువరుసలు = ggml_nrows(src0);
    స్వయంచాలక src1_అడ్డువరుసలు = ggml_nrows(src1);

    పూర్ణాంకం ith = params->ith;
    పూర్ణాంకం nth = params->nth;

    GGML_TENSOR_ద్విమాన_OP_LOCALS

    GGML_ASSERT(nb0 == పరిమాణం(T));
    GGML_ASSERT(nb00 == పరిమాణం(T));

    స్థిరరాశి స్వయంచాలక [ir0, ir1] = get_thread_range(params, src0);

    స్వయంచాలక compute_func_vv = [&](int64_t blk_len, int64_t r, T * src0_ptr, T * src1_ptr, T * dst_ptr) {
        int64_t idx = 0;
        ఒకవేళ constexpr (op_type == GGML_OP_కూడు) {
            ఒకవేళ constexpr (std::is_same_v<T, దశాంశం>) {
                కొరకు (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e32m4(blk_len);
                    vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + idx + r, vl);
                    vfloat32m4_t rhs = __riscv_vle32_v_f32m4(src1_ptr + idx, vl);
                    vfloat32m4_t res = __riscv_vfadd_vv_f32m4(lhs, rhs, vl);
                    __riscv_vse32_v_f32m4(dst_ptr + idx + r, res, vl);
                }
            } లేకపోతే ఒకవేళ constexpr (std::is_same_v<T, _Float16>) {
                కొరకు (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e16m4(blk_len);
                    vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + idx + r), vl);
                    vfloat16m4_t rhs = __riscv_vle16_v_f16m4((src1_ptr + idx), vl);
                    vfloat16m4_t res = __riscv_vfadd_vv_f16m4(lhs, rhs, vl);
                    __riscv_vse16_v_f16m4((dst_ptr + idx + r), res, vl);
                }
            } లేకపోతే {
                GGML_ABORT("fatal error");
            }
        } లేకపోతే ఒకవేళ constexpr (op_type == GGML_OP_SUB) {
            ఒకవేళ constexpr (std::is_same_v<T, దశాంశం>) {
                కొరకు (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e32m4(blk_len);
                    vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + idx + r, vl);
                    vfloat32m4_t rhs = __riscv_vle32_v_f32m4(src1_ptr + idx, vl);
                    vfloat32m4_t res = __riscv_vfsub_vv_f32m4(lhs, rhs, vl);
                    __riscv_vse32_v_f32m4(dst_ptr + idx + r, res, vl);
                }
            } లేకపోతే ఒకవేళ constexpr (std::is_same_v<T, _Float16>) {
                కొరకు (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e16m4(blk_len);
                    vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + idx + r), vl);
                    vfloat16m4_t rhs = __riscv_vle16_v_f16m4((src1_ptr + idx), vl);
                    vfloat16m4_t res = __riscv_vfsub_vv_f16m4(lhs, rhs, vl);
                    __riscv_vse16_v_f16m4((dst_ptr + idx + r), res, vl);
                }
            } లేకపోతే {
                GGML_ABORT("fatal error");
            }
        } లేకపోతే ఒకవేళ constexpr (op_type == GGML_OP_MUL) {
            ఒకవేళ constexpr (std::is_same_v<T, దశాంశం>) {
                కొరకు (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e32m4(blk_len);
                    vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + idx + r, vl);
                    vfloat32m4_t rhs = __riscv_vle32_v_f32m4(src1_ptr + idx, vl);
                    vfloat32m4_t res = __riscv_vfmul_vv_f32m4(lhs, rhs, vl);
                    __riscv_vse32_v_f32m4(dst_ptr + idx + r, res, vl);
                }
            } లేకపోతే ఒకవేళ constexpr (std::is_same_v<T, _Float16>) {
                కొరకు (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e16m4(blk_len);
                    vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + idx + r), vl);
                    vfloat16m4_t rhs = __riscv_vle16_v_f16m4((src1_ptr + idx), vl);
                    vfloat16m4_t res = __riscv_vfmul_vv_f16m4(lhs, rhs, vl);
                    __riscv_vse16_v_f16m4((dst_ptr + idx + r), res, vl);
                }
            } లేకపోతే {
                GGML_ABORT("fatal error");
            }
        } లేకపోతే ఒకవేళ constexpr (op_type == GGML_OP_DIV) {
            ఒకవేళ constexpr (std::is_same_v<T, దశాంశం>) {
                కొరకు (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e32m4(blk_len);
                    vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + idx + r, vl);
                    vfloat32m4_t rhs = __riscv_vle32_v_f32m4(src1_ptr + idx, vl);
                    vfloat32m4_t res = __riscv_vfdiv_vv_f32m4(lhs, rhs, vl);
                    __riscv_vse32_v_f32m4(dst_ptr + idx + r, res, vl);
                }
            } లేకపోతే ఒకవేళ constexpr (std::is_same_v<T, _Float16>) {
                కొరకు (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e16m4(blk_len);
                    vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + idx + r), vl);
                    vfloat16m4_t rhs = __riscv_vle16_v_f16m4((src1_ptr + idx), vl);
                    vfloat16m4_t res = __riscv_vfdiv_vv_f16m4(lhs, rhs, vl);
                    __riscv_vse16_v_f16m4((dst_ptr + idx + r), res, vl);
                }
            } లేకపోతే {
                GGML_ABORT("fatal error");
            }
        } లేకపోతే {
            GGML_ABORT("fatal error");
        }
    };

    ఒకవేళ (src0_అడ్డువరుసలు == src1_అడ్డువరుసలు && src0_అడ్డువరుసలు == 1 && ne00 == ne10) {
        int64_t task_చొప్పున_thread = (ne00 + nth - 1) / nth;
        int64_t task_begin      = ith * task_చొప్పున_thread;
        int64_t task_చివర        = std::min((ith + 1) * task_చొప్పున_thread, ne00);

        T * dst_ptr  = ((T *) dst->దత్తాంశం) + task_begin;
        T * src0_ptr = ((T *) src0->దత్తాంశం) + task_begin;
        T * src1_ptr = ((T *) src1->దత్తాంశం) + task_begin;

        compute_func_vv(task_చివర - task_begin, 0, src0_ptr, src1_ptr, dst_ptr);
    } లేకపోతే ఒకవేళ (ne10 > 1) {
        కొరకు (int64_t ir = ir0; ir < ir1; ++ir) {
            స్థిరరాశి int64_t i03 = ir / (ne02 * ne01);
            స్థిరరాశి int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            స్థిరరాశి int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

            స్థిరరాశి int64_t i13 = i03 % ne13;
            స్థిరరాశి int64_t i12 = i02 % ne12;
            స్థిరరాశి int64_t i11 = i01 % ne11;

            T * dst_ptr  = (T *) ((అక్షరం *) dst->దత్తాంశం + i03 * nb3 + i02 * nb2 + i01 * nb1);
            T * src0_ptr = (T *) ((అక్షరం *) src0->దత్తాంశం + i03 * nb03 + i02 * nb02 + i01 * nb01);
            T * src1_ptr = (T *) ((అక్షరం *) src1->దత్తాంశం + i13 * nb13 + i12 * nb12 + i11 * nb11);

            // src1 is broadcastable across src0 and dst in i1, i2, i3
            కొరకు (int64_t r = 0; r < ne00; r += ne10) {
                compute_func_vv(ne10, r, src0_ptr, src1_ptr, dst_ptr);
            }
        }
    } లేకపోతే {
        కొరకు (int64_t ir = ir0; ir < ir1; ++ir) {
            స్థిరరాశి int64_t i03 = ir / (ne02 * ne01);
            స్థిరరాశి int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            స్థిరరాశి int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

            స్థిరరాశి int64_t i13 = i03 % ne13;
            స్థిరరాశి int64_t i12 = i02 % ne12;
            స్థిరరాశి int64_t i11 = i01 % ne11;

            T * dst_ptr  = (T *) ((అక్షరం *) dst->దత్తాంశం + i03 * nb3 + i02 * nb2 + i01 * nb1);
            T * src0_ptr = (T *) ((అక్షరం *) src0->దత్తాంశం + i03 * nb03 + i02 * nb02 + i01 * nb01);
            T * src1_ptr = (T *) ((అక్షరం *) src1->దత్తాంశం + i13 * nb13 + i12 * nb12 + i11 * nb11);

            T       rhs_scalar = src1_ptr[0];
            int64_t blk_len    = ne00;
            int64_t r          = 0;

            కొరకు (size_t vl; blk_len > 0; blk_len -= vl, r += vl) {
                ఒకవేళ constexpr (op_type == GGML_OP_కూడు) {
                    ఒకవేళ constexpr (std::is_same_v<T, దశాంశం>) {
                        vl               = __riscv_vsetvl_e32m4(blk_len);
                        vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + r, vl);
                        vfloat32m4_t res = __riscv_vfadd_vf_f32m4(lhs, rhs_scalar, vl);
                        __riscv_vse32_v_f32m4(dst_ptr + r, res, vl);
                    } లేకపోతే ఒకవేళ constexpr (std::is_same_v<T, _Float16>) {
                        vl               = __riscv_vsetvl_e16m4(blk_len);
                        vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + r), vl);
                        vfloat16m4_t res = __riscv_vfadd_vf_f16m4(lhs, rhs_scalar, vl);
                        __riscv_vse16_v_f16m4((dst_ptr + r), res, vl);
                    } లేకపోతే {
                        GGML_ABORT("fatal error");
                    }
                } లేకపోతే ఒకవేళ constexpr (op_type == GGML_OP_SUB) {
                    ఒకవేళ constexpr (std::is_same_v<T, దశాంశం>) {
                        vl               = __riscv_vsetvl_e32m4(blk_len);
                        vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + r, vl);
                        vfloat32m4_t res = __riscv_vfsub_vf_f32m4(lhs, rhs_scalar, vl);
                        __riscv_vse32_v_f32m4(dst_ptr + r, res, vl);
                    } లేకపోతే ఒకవేళ constexpr (std::is_same_v<T, _Float16>) {
                        vl               = __riscv_vsetvl_e16m4(blk_len);
                        vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + r), vl);
                        vfloat16m4_t res = __riscv_vfsub_vf_f16m4(lhs, rhs_scalar, vl);
                        __riscv_vse16_v_f16m4((dst_ptr + r), res, vl);
                    } లేకపోతే {
                        GGML_ABORT("fatal error");
                    }
                } లేకపోతే ఒకవేళ constexpr (op_type == GGML_OP_MUL) {
                    ఒకవేళ constexpr (std::is_same_v<T, దశాంశం>) {
                        vl               = __riscv_vsetvl_e32m4(blk_len);
                        vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + r, vl);
                        vfloat32m4_t res = __riscv_vfmul_vf_f32m4(lhs, rhs_scalar, vl);
                        __riscv_vse32_v_f32m4(dst_ptr + r, res, vl);
                    } లేకపోతే ఒకవేళ constexpr (std::is_same_v<T, _Float16>) {
                        vl               = __riscv_vsetvl_e16m4(blk_len);
                        vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + r), vl);
                        vfloat16m4_t res = __riscv_vfmul_vf_f16m4(lhs, rhs_scalar, vl);
                        __riscv_vse16_v_f16m4((dst_ptr + r), res, vl);
                    } లేకపోతే {
                        GGML_ABORT("fatal error");
                    }
                } లేకపోతే ఒకవేళ constexpr (op_type == GGML_OP_DIV) {
                    ఒకవేళ constexpr (std::is_same_v<T, దశాంశం>) {
                        vl               = __riscv_vsetvl_e32m4(blk_len);
                        vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + r, vl);
                        vfloat32m4_t res = __riscv_vfdiv_vf_f32m4(lhs, rhs_scalar, vl);
                        __riscv_vse32_v_f32m4(dst_ptr + r, res, vl);
                    } లేకపోతే ఒకవేళ constexpr (std::is_same_v<T, _Float16>) {
                        vl               = __riscv_vsetvl_e16m4(blk_len);
                        vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + r), vl);
                        vfloat16m4_t res = __riscv_vfdiv_vf_f16m4(lhs, rhs_scalar, vl);
                        __riscv_vse16_v_f16m4((dst_ptr + r), res, vl);
                    } లేకపోతే {
                        GGML_ABORT("fatal error");
                    }
                } లేకపోతే {
                    GGML_ABORT("fatal error");
                }
            }
        }
    }
}

మూస <రకం_పేరు T> శూన్య forward_కూడిక_అడ్డువరుసలు(స్థిరరాశి ggml_compute_params * params, ggml_tensor * op) {
    స్థిరరాశి ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne0 == 1);
    GGML_ASSERT(ne1 == ne01);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    int64_t n_task          = ne01 * ne02 * ne03;
    int64_t task_చొప్పున_thread = (n_task + nth - 1) / nth;
    int64_t ir_ప్రారంభం        = ith * task_చొప్పున_thread;
    int64_t ir_చివర          = std::min(ir_ప్రారంభం + task_చొప్పున_thread, n_task);

    కొరకు (int64_t ir = ir_ప్రారంభం; ir < ir_చివర; ir++) {
        స్థిరరాశి int64_t i3 = ir / (ne02 * ne01);
        స్థిరరాశి int64_t i2 = (ir - i3 * ne02 * ne01) / ne01;
        స్థిరరాశి int64_t i1 = (ir - i3 * ne02 * ne01 - i2 * ne01);

        T * src_అడ్డువరుస = (T *) ((అక్షరం *) src0->దత్తాంశం + i1 * nb01 + i2 * nb02 + i3 * nb03);
        T * dst_అడ్డువరుస = (T *) ((అక్షరం *) op->దత్తాంశం + i1 * nb1 + i2 * nb2 + i3 * nb3);

        దశాంశం అడ్డువరుస_కూడిక = 0;

        ఒకవేళ constexpr (std::is_same_v<T, దశాంశం>) {
            size_t        gvl     = __riscv_vsetvlmax_e32m4();
            vfloat32m4_t  acc_vec = __riscv_vfmv_v_f_f32m4(0.0f, gvl);
            int64_t       నిడివి  = ne00;
            స్థిరరాశి దశాంశం * p_దత్తాంశం  = src_అడ్డువరుస;

            అయితే (నిడివి > 0) {
                size_t       vl  = __riscv_vsetvl_e32m4(నిడివి);
                vfloat32m4_t vec = __riscv_vle32_v_f32m4(p_దత్తాంశం, vl);
                acc_vec          = __riscv_vfadd_vv_f32m4(acc_vec, vec, vl);
                p_దత్తాంశం += vl;
                నిడివి -= vl;
            }

            gvl                 = __riscv_vsetvlmax_e32m1();
            vfloat32m1_t zero_v = __riscv_vfmv_v_f_f32m1(0.0f, gvl);
            vfloat32m1_t కూడిక_v  = __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(acc_vec, 0),
                                                         __riscv_vget_v_f32m4_f32m1(acc_vec, 1), gvl);
            కూడిక_v               = __riscv_vfadd_vv_f32m1(కూడిక_v, __riscv_vget_v_f32m4_f32m1(acc_vec, 2), gvl);
            కూడిక_v               = __riscv_vfadd_vv_f32m1(కూడిక_v, __riscv_vget_v_f32m4_f32m1(acc_vec, 3), gvl);
            కూడిక_v               = __riscv_vfredusum_vs_f32m1_f32m1(కూడిక_v, zero_v, gvl);
            అడ్డువరుస_కూడిక             = __riscv_vfmv_f_s_f32m1_f32(కూడిక_v);
        } లేకపోతే ఒకవేళ constexpr (std::is_same_v<T, _Float16>) {
            size_t           gvl     = __riscv_vsetvlmax_e16m2();
            vfloat32m4_t     acc_vec = __riscv_vfmv_v_f_f32m4(0.0f, gvl);
            int64_t          నిడివి  = ne00;
            స్థిరరాశి _Float16 * p_దత్తాంశం  = src_అడ్డువరుస;

            అయితే (నిడివి > 0) {
                size_t       vl      = __riscv_vsetvl_e16m2(నిడివి);
                vfloat16m2_t vec_f16 = __riscv_vle16_v_f16m2(p_దత్తాంశం, vl);
                vfloat32m4_t vec_f32 = __riscv_vfwcvt_f_f_v_f32m4(vec_f16, vl);
                acc_vec              = __riscv_vfadd_vv_f32m4(acc_vec, vec_f32, vl);
                p_దత్తాంశం += vl;
                నిడివి -= vl;
            }

            gvl                 = __riscv_vsetvlmax_e32m1();
            vfloat32m1_t zero_v = __riscv_vfmv_v_f_f32m1(0.0f, gvl);
            vfloat32m1_t కూడిక_v  = __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(acc_vec, 0),
                                                         __riscv_vget_v_f32m4_f32m1(acc_vec, 1), gvl);
            కూడిక_v               = __riscv_vfadd_vv_f32m1(కూడిక_v, __riscv_vget_v_f32m4_f32m1(acc_vec, 2), gvl);
            కూడిక_v               = __riscv_vfadd_vv_f32m1(కూడిక_v, __riscv_vget_v_f32m4_f32m1(acc_vec, 3), gvl);
            కూడిక_v               = __riscv_vfredusum_vs_f32m1_f32m1(కూడిక_v, zero_v, gvl);
            అడ్డువరుస_కూడిక             = __riscv_vfmv_f_s_f32m1_f32(కూడిక_v);
        } లేకపోతే {
            GGML_ABORT("fatal error");
        }

        dst_అడ్డువరుస[0] = అడ్డువరుస_కూడిక;
    }
}

మూస <రకం_పేరు T> శూన్య forward_repeat_nrows(ggml_compute_params * params, ggml_tensor * op) {
    స్థిరరాశి ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    int64_t nrows            = ggml_nrows(src0);
    int64_t nrows_చొప్పున_thread = (nrows + nth - 1) / nth;
    int64_t ir_ప్రారంభం         = ith * nrows_చొప్పున_thread;
    int64_t ir_చివర           = std::min(ir_ప్రారంభం + nrows_చొప్పున_thread, nrows);

    ఒకవేళ (src0->ne[0] == 1) {
        కొరకు (int64_t ir = ir_ప్రారంభం; ir < ir_చివర; ir++) {
            T * src_అడ్డువరుస = (T *) ((అక్షరం *) src0->దత్తాంశం + ir * src0->nb[1]);
            T * dst_అడ్డువరుస = (T *) ((అక్షరం *) dst->దత్తాంశం + ir * dst->nb[1]);

            T src_scalar = src_అడ్డువరుస[0];

            int64_t నిడివి = dst->ne[0];
            int64_t idx    = 0;
            size_t  vl     = 0;

            అయితే (నిడివి > 0) {
                ఒకవేళ constexpr (std::is_same_v<T, int32_t>) {
                    vl             = __riscv_vsetvl_e32m4(నిడివి);
                    vint32m4_t vec = __riscv_vmv_v_x_i32m4(src_scalar, vl);
                    __riscv_vse32_v_i32m4(dst_అడ్డువరుస + idx, vec, vl);
                } లేకపోతే ఒకవేళ constexpr (std::is_same_v<T, int16_t>) {
                    vl             = __riscv_vsetvl_e16m4(నిడివి);
                    vint16m4_t vec = __riscv_vmv_v_x_i16m4(src_scalar, vl);
                    __riscv_vse16_v_i16m4((dst_అడ్డువరుస + idx), vec, vl);
                } లేకపోతే {
                    GGML_ABORT("fatal error");
                }
                idx += vl;
                నిడివి -= vl;
            }
        }
    } లేకపోతే ఒకవేళ (src0->ne[0] == dst->ne[0]) {
        కొరకు (int64_t ir = ir_ప్రారంభం; ir < ir_చివర; ir++) {
            T * src_అడ్డువరుస = (T *) ((అక్షరం *) src0->దత్తాంశం + ir * src0->nb[1]);
            T * dst_అడ్డువరుస = (T *) ((అక్షరం *) dst->దత్తాంశం + ir * dst->nb[1]);

            int64_t నిడివి = dst->ne[0];
            int64_t idx    = 0;
            size_t  vl     = 0;

            అయితే (నిడివి > 0) {
                ఒకవేళ constexpr (std::is_same_v<T, int32_t>) {
                    vl             = __riscv_vsetvl_e32m4(నిడివి);
                    vint32m4_t vec = __riscv_vle32_v_i32m4(src_అడ్డువరుస + idx, vl);
                    __riscv_vse32_v_i32m4(dst_అడ్డువరుస + idx, vec, vl);
                } లేకపోతే ఒకవేళ constexpr (std::is_same_v<T, int16_t>) {
                    vl             = __riscv_vsetvl_e16m4(నిడివి);
                    vint16m4_t vec = __riscv_vle16_v_i16m4((src_అడ్డువరుస + idx), vl);
                    __riscv_vse16_v_i16m4((dst_అడ్డువరుస + idx), vec, vl);
                } లేకపోతే {
                    GGML_ABORT("fatal error");
                }
                idx += vl;
                నిడివి -= vl;
            }
        }
    } లేకపోతే {
        GGML_ABORT("fatal error");
    }
}

మూస <రకం_పేరు T> శూన్య forward_repeat_dim1(ggml_compute_params * params, ggml_tensor * op) {
    స్థిరరాశి ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి int64_t ne0 = dst->ne[0];
    స్థిరరాశి int64_t ne1 = dst->ne[1];
    స్థిరరాశి int64_t ne2 = dst->ne[2];
    స్థిరరాశి int64_t ne3 = dst->ne[3];

    స్థిరరాశి int64_t మొత్తం_batches      = ne2 * ne3;
    స్థిరరాశి int64_t batches_చొప్పున_thread = (మొత్తం_batches + nth - 1) / nth;
    స్థిరరాశి int64_t batch_ప్రారంభం        = ith * batches_చొప్పున_thread;
    స్థిరరాశి int64_t batch_చివర          = std::min(batch_ప్రారంభం + batches_చొప్పున_thread, మొత్తం_batches);

    కొరకు (int64_t b = batch_ప్రారంభం; b < batch_చివర; b++) {
        స్థిరరాశి int64_t i3 = b / ne2;
        స్థిరరాశి int64_t i2 = b % ne2;

        T * src_పునాది  = (T *) ((అక్షరం *) src0->దత్తాంశం + i2 * src0->nb[2] + i3 * src0->nb[3]);
        T * dst_batch = (T *) ((అక్షరం *) dst->దత్తాంశం + i2 * dst->nb[2] + i3 * dst->nb[3]);

        కొరకు (int64_t i1 = 0; i1 < ne1; i1++) {
            T *     dst_ptr = (T *) ((అక్షరం *) dst_batch + i1 * dst->nb[1]);
            int64_t నిడివి  = ne0;
            int64_t idx     = 0;

            అయితే (నిడివి > 0) {
                ఒకవేళ constexpr (std::is_same_v<T, int32_t>) {
                    size_t     vl  = __riscv_vsetvl_e32m4(నిడివి);
                    vint32m4_t vec = __riscv_vle32_v_i32m4(src_పునాది + idx, vl);
                    __riscv_vse32_v_i32m4(dst_ptr + idx, vec, vl);
                    idx += vl;
                    నిడివి -= vl;
                } లేకపోతే ఒకవేళ constexpr (std::is_same_v<T, int16_t>) {
                    size_t     vl  = __riscv_vsetvl_e16m4(నిడివి);
                    vint16m4_t vec = __riscv_vle16_v_i16m4((src_పునాది + idx), vl);
                    __riscv_vse16_v_i16m4((dst_ptr + idx), vec, vl);
                    idx += vl;
                    నిడివి -= vl;
                } లేకపోతే {
                    GGML_ABORT("fatal error");
                }
            }
        }
    }
}

మూస <రకం_పేరు T> శూన్య forward_get_అడ్డువరుసలు(ggml_compute_params * params, ggml_tensor * op) {
    స్థిరరాశి ggml_tensor * src0 = op->src[0];
    స్థిరరాశి ggml_tensor * src1 = op->src[1];
    ggml_tensor *       dst  = op;

    GGML_TENSOR_ద్విమాన_OP_LOCALS

    స్థిరరాశి int64_t nc = ne00;
    స్థిరరాశి int64_t nr = ggml_nelements(src1);

    ప్రతిజ్ఞా(ne0 == nc);
    ప్రతిజ్ఞా(ne02 == ne11);
    ప్రతిజ్ఞా(nb00 == పరిమాణం(దశాంశం));
    ప్రతిజ్ఞా(ggml_nrows(op) == nr);

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    పూర్ణాంకం అడ్డువరుసలు_nth = nth;
    పూర్ణాంకం నిలువుల_సంఖ్య_nth = 1;

    ఒకవేళ (nr == 1) {
        అడ్డువరుసలు_nth = 1;
        నిలువుల_సంఖ్య_nth = nth;
    }

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + అడ్డువరుసలు_nth - 1) / అడ్డువరుసలు_nth;
    స్థిరరాశి పూర్ణాంకం dc = (nc + నిలువుల_సంఖ్య_nth - 1) / నిలువుల_సంఖ్య_nth;

    పూర్ణాంకం అడ్డువరుసలు_ith = ith % అడ్డువరుసలు_nth;
    పూర్ణాంకం నిలువుల_సంఖ్య_ith = ith % నిలువుల_సంఖ్య_nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr * అడ్డువరుసలు_ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    స్థిరరాశి పూర్ణాంకం cr0 = dc * నిలువుల_సంఖ్య_ith;
    స్థిరరాశి పూర్ణాంకం cr1 = MIN(cr0 + dc, nc);

    కొరకు (int64_t i = ir0; i < ir1; ++i) {
        స్థిరరాశి int64_t i12 = i / (ne11 * ne10);
        స్థిరరాశి int64_t i11 = (i - i12 * ne11 * ne10) / ne10;
        స్థిరరాశి int64_t i10 = (i - i12 * ne11 * ne10 - i11 * ne10);
        స్థిరరాశి int64_t i01 = *(int32_t *) ((అక్షరం *) src1->దత్తాంశం + i10 * nb10 + i11 * nb11 + i12 * nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        memcpy1d(((అక్షరం *) dst->దత్తాంశం + i10 * nb1 + i11 * nb2 + i12 * nb3) + cr0 * పరిమాణం(T),
                 ((అక్షరం *) src0->దత్తాంశం + i01 * nb01 + i11 * nb02 + i12 * nb03) + cr0 * పరిమాణం(T),
                 (cr1 - cr0) * పరిమాణం(T));
    }
}

మూస <రకం_పేరు T> శూన్య forward_concat(ggml_compute_params * params, ggml_tensor * op) {
    స్థిరరాశి ggml_tensor * src0 = op->src[0];
    స్థిరరాశి ggml_tensor * src1 = op->src[1];
    ggml_tensor *       dst  = op;

    GGML_ASSERT(ggml_type_కొలత(src0->type) == పరిమాణం(దశాంశం));

    GGML_TENSOR_ద్విమాన_OP_LOCALS

    స్థిరరాశి int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim == 0 && nb0 == పరిమాణం(దశాంశం) && nb1 == పరిమాణం(దశాంశం) * (ne00 + ne10));

    స్థిరరాశి int64_t nr = ggml_nrows(dst);
    స్థిరరాశి int64_t nc = ne0;

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    పూర్ణాంకం అడ్డువరుసలు_nth = nth;
    పూర్ణాంకం నిలువుల_సంఖ్య_nth = 1;

    ఒకవేళ (nr == 1) {
        అడ్డువరుసలు_nth = 1;
        నిలువుల_సంఖ్య_nth = nth;
    }

    స్థిరరాశి పూర్ణాంకం dr = (nr + అడ్డువరుసలు_nth - 1) / అడ్డువరుసలు_nth;
    స్థిరరాశి పూర్ణాంకం dc = (nc + నిలువుల_సంఖ్య_nth - 1) / నిలువుల_సంఖ్య_nth;

    పూర్ణాంకం అడ్డువరుసలు_ith = ith % అడ్డువరుసలు_nth;
    పూర్ణాంకం నిలువుల_సంఖ్య_ith = ith % నిలువుల_సంఖ్య_nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr * అడ్డువరుసలు_ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    స్థిరరాశి పూర్ణాంకం cr0 = dc * నిలువుల_సంఖ్య_ith;
    స్థిరరాశి పూర్ణాంకం cr1 = MIN(cr0 + dc, nc);

    int64_t o[4] = { 0, 0, 0, 0 };
    o[dim]       = src0->ne[dim];
    స్థిరరాశి దశాంశం * x;

    కొరకు (int64_t i = ir0; i < ir1; ++i) {
        స్థిరరాశి int64_t i3 = i / (ne02 * ne01);
        స్థిరరాశి int64_t i2 = (i - i3 * ne02 * ne01) / ne01;
        స్థిరరాశి int64_t i1 = (i - i3 * ne02 * ne01 - i2 * ne01);

        కొరకు (పూర్ణాంకం i0 = cr0; i0 < cr1; i0++) {
            ఒకవేళ (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                x = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src0->దత్తాంశం + (i0) *nb00 + (i1) *nb01 + (i2) *nb02 + (i3) *nb03);
            } లేకపోతే {
                x = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src1->దత్తాంశం + (i0 - o[0]) * nb10 + (i1 - o[1]) * nb11 +
                                     (i2 - o[2]) * nb12 + (i3 - o[3]) * nb13);
            }

            దశాంశం * y = (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + i0 * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3);

            *y = *x;
        }
    }
}

మూస శూన్య forward_ద్విమాన<GGML_OP_కూడు, దశాంశం>(ggml_compute_params * params, ggml_tensor * op);
మూస శూన్య forward_ద్విమాన<GGML_OP_SUB, దశాంశం>(ggml_compute_params * params, ggml_tensor * op);
మూస శూన్య forward_ద్విమాన<GGML_OP_MUL, దశాంశం>(ggml_compute_params * params, ggml_tensor * op);
మూస శూన్య forward_ద్విమాన<GGML_OP_DIV, దశాంశం>(ggml_compute_params * params, ggml_tensor * op);
మూస శూన్య forward_ద్విమాన<GGML_OP_కూడు, _Float16>(ggml_compute_params * params, ggml_tensor * op);
మూస శూన్య forward_ద్విమాన<GGML_OP_SUB, _Float16>(ggml_compute_params * params, ggml_tensor * op);
మూస శూన్య forward_ద్విమాన<GGML_OP_MUL, _Float16>(ggml_compute_params * params, ggml_tensor * op);
మూస శూన్య forward_ద్విమాన<GGML_OP_DIV, _Float16>(ggml_compute_params * params, ggml_tensor * op);
మూస శూన్య forward_కూడిక_అడ్డువరుసలు<దశాంశం>(స్థిరరాశి ggml_compute_params * params, ggml_tensor * op);
మూస శూన్య forward_కూడిక_అడ్డువరుసలు<_Float16>(స్థిరరాశి ggml_compute_params * params, ggml_tensor * op);
మూస శూన్య forward_repeat_nrows<int32_t>(ggml_compute_params * params, ggml_tensor * op);
మూస శూన్య forward_repeat_nrows<int16_t>(ggml_compute_params * params, ggml_tensor * op);
మూస శూన్య forward_repeat_dim1<int32_t>(ggml_compute_params * params, ggml_tensor * op);
మూస శూన్య forward_repeat_dim1<int16_t>(ggml_compute_params * params, ggml_tensor * op);
మూస శూన్య forward_get_అడ్డువరుసలు<int32_t>(ggml_compute_params * params, ggml_tensor * op);
మూస శూన్య forward_get_అడ్డువరుసలు<int16_t>(ggml_compute_params * params, ggml_tensor * op);
మూస శూన్య forward_concat<int32_t>(ggml_compute_params * params, ggml_tensor * op);
మూస శూన్య forward_concat<int16_t>(ggml_compute_params * params, ggml_tensor * op);

}  // namespace spacemit_kernels::rvv
