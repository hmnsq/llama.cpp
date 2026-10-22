#ಅನ್ತರ್ಭಾವ "rvv_kernels.h"

#ಅನ್ತರ್ಭಾವ "common.h"
#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "ops.h"
#ಅನ್ತರ್ಭಾವ "string.h"

#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <cmath>
#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <stdexcept>

#ಆದರೆ !defined(__riscv_v) || !defined(__riscv_v_intrinsic)
#    error "riscv v extension or v_intrinsic not enabled"
#ಇಲ್ಲದಿದ್ದರೆ
#    include <riscv_vector.h>
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ !defined(__riscv_zfh)
#    error "riscv zfh extension not enabled"
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(__GNUC__)
#    pragma GCC diagnostic ignored "-Woverlength-strings"
#    pragma GCC diagnostic ignored "-Wcast-qual"
#    pragma GCC diagnostic ignored "-Wunused-parameter"
#ಆದರೆ_ಅಂತ್ಯ

ನಾಮಾಕಾಶ spacemit_kernels::rvv {

ನಾಮಾಕಾಶ {

ಸ್ವಯಂಚಾಲಿತ align_ಮೇಲೆ(size_t value, size_t alignment) {
    ಹಿಂತಿರುಗಿ (value + alignment - 1) / alignment * alignment;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ತಾರ್ಕಿಕ flash_attn_ext_supported_d_vlen1024_vf16(int64_t d) {
    ಹಿಂತಿರುಗಿ d > 0 && d <= 128;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ತಾರ್ಕಿಕ flash_attn_ext_supported_shape_vlen1024_vf16(int64_t DK, int64_t DV) {
    ಹಿಂತಿರುಗಿ flash_attn_ext_supported_d_vlen1024_vf16(DK) && flash_attn_ext_supported_d_vlen1024_vf16(DV);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ reduce_ಮೊತ್ತ_f32m4_vlen1024(vfloat32m4_t v, size_t vl) {
    vfloat32m1_t s_v = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    s_v              = __riscv_vfredusum_vs_f32m4_f32m1(v, s_v, vl);
    ಹಿಂತಿರುಗಿ __riscv_vfmv_f_s_f32m1_f32(s_v);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ reduce_ಮೊತ್ತ_f32m2_vlen1024(vfloat32m2_t v, size_t vl) {
    vfloat32m1_t s_v = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    s_v              = __riscv_vfredusum_vs_f32m2_f32m1(v, s_v, vl);
    ಹಿಂತಿರುಗಿ __riscv_vfmv_f_s_f32m1_f32(s_v);
}

// Adapted from ggml_v_expf_m2 in vec.h. This is accurate enough for softmax.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ vfloat32m2_t rvv_expf_approx_f32m2(vfloat32m2_t x, size_t vl) {
    ಸ್ಥಿರಾಂಕ vfloat32m2_t r = __riscv_vfmv_v_f_f32m2(0x1.8p23f, vl);
    ಸ್ಥಿರಾಂಕ vfloat32m2_t z = __riscv_vfmacc_vf_f32m2(r, 0x1.715476p+0f, x, vl);
    ಸ್ಥಿರಾಂಕ vfloat32m2_t n = __riscv_vfsub_vv_f32m2(z, r, vl);
    ಸ್ಥಿರಾಂಕ vfloat32m2_t b =
        __riscv_vfnmsac_vf_f32m2(__riscv_vfnmsac_vf_f32m2(x, 0x1.62e4p-1f, n, vl), 0x1.7f7d1cp-20f, n, vl);
    ಸ್ಥಿರಾಂಕ vuint32m2_t  e = __riscv_vsll_vx_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(z), 23, vl);
    ಸ್ಥಿರಾಂಕ vfloat32m2_t k = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vadd_vx_u32m2(e, 0x3f800000, vl));
    ಸ್ಥಿರಾಂಕ vbool16_t    c = __riscv_vmfgt_vf_f32m2_b16(__riscv_vfabs_v_f32m2(n, vl), 126.0f, vl);
    ಸ್ಥಿರಾಂಕ vfloat32m2_t u = __riscv_vfmul_vv_f32m2(b, b, vl);
    ಸ್ಥಿರಾಂಕ vfloat32m2_t j = __riscv_vfmacc_vv_f32m2(
        __riscv_vfmul_vf_f32m2(b, 0x1.ffffecp-1f, vl),
        __riscv_vfmacc_vv_f32m2(
            __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(0x1.fffdb6p-2f, vl), 0x1.555e66p-3f, b, vl),
            __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(0x1.573e2ep-5f, vl), 0x1.0e4020p-7f, b, vl), u, vl),
        u, vl);

    ಆದರೆ (!__riscv_vcpop_m_b16(c, vl)) {
        ಹಿಂತಿರುಗಿ __riscv_vfmacc_vv_f32m2(k, j, k, vl);
    }

    ಸ್ಥಿರಾಂಕ vbool16_t    dm = __riscv_vmfle_vf_f32m2_b16(n, 0.0f, vl);
    ಸ್ಥಿರಾಂಕ vuint32m2_t  d  = __riscv_vmerge_vxm_u32m2(__riscv_vmv_v_x_u32m2(0, vl), 0x82000000, dm, vl);
    ಸ್ಥಿರಾಂಕ vfloat32m2_t s1 = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vadd_vx_u32m2(d, 0x7f000000, vl));
    ಸ್ಥಿರಾಂಕ vfloat32m2_t s2 = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vsub_vv_u32m2(e, d, vl));
    ಸ್ಥಿರಾಂಕ vfloat32m2_t r1 =
        __riscv_vmerge_vvm_f32m2(__riscv_vfmacc_vv_f32m2(k, k, j, vl),
                                 __riscv_vfmul_vv_f32m2(__riscv_vfmacc_vv_f32m2(s2, s2, j, vl), s1, vl), c, vl);
    ಹಿಂತಿರುಗಿ __riscv_vmerge_vvm_f32m2(r1, __riscv_vfmul_vv_f32m2(s1, s1, vl),
                                    __riscv_vmfgt_vf_f32m2_b16(__riscv_vfabs_v_f32m2(n, vl), 192.0f, vl), vl);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ vfloat32m2_t rvv_tanh_approx_f32m2(vfloat32m2_t x, size_t vl) {
    ಸ್ಥಿರಾಂಕ vfloat32m2_t abs_x       = __riscv_vfabs_v_f32m2(x, vl);
    ಸ್ಥಿರಾಂಕ vfloat32m2_t neg_2_abs   = __riscv_vfmul_vf_f32m2(abs_x, -2.0f, vl);
    ಸ್ಥಿರಾಂಕ vfloat32m2_t exp_ಪದ    = rvv_expf_approx_f32m2(neg_2_abs, vl);
    ಸ್ಥಿರಾಂಕ vfloat32m2_t numerator   = __riscv_vfsub_vf_f32m2(exp_ಪದ, 1.0f, vl);
    ಸ್ಥಿರಾಂಕ vfloat32m2_t denominator = __riscv_vfadd_vf_f32m2(exp_ಪದ, 1.0f, vl);
    ಸ್ಥಿರಾಂಕ vfloat32m2_t tanh_abs    = __riscv_vfneg_v_f32m2(__riscv_vfdiv_vv_f32m2(numerator, denominator, vl), vl);
    ಸ್ಥಿರಾಂಕ vbool16_t    neg_ಮುಖವಾಡ    = __riscv_vmflt_vf_f32m2_b16(x, 0.0f, vl);
    ಸ್ಥಿರಾಂಕ vfloat32m2_t tanh_neg    = __riscv_vfneg_v_f32m2(tanh_abs, vl);
    ಹಿಂತಿರುಗಿ __riscv_vmerge_vvm_f32m2(tanh_abs, tanh_neg, neg_ಮುಖವಾಡ, vl);
}

ಸ್ಥಿರ ಶೂನ್ಯ rvv_softcap_tanh_inplace_f32(ದಶಮಾಂಶ * dst, int64_t dst_stride, int64_t tile_ಅಡ್ಡಸಾಲುಗಳು, int64_t n, ದಶಮಾಂಶ softcap) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ tq = 0; tq < tile_ಅಡ್ಡಸಾಲುಗಳು; ++tq, dst += dst_stride) {
        ದಶಮಾಂಶ * dst_ಅಡ್ಡಸಾಲು   = dst;
        int64_t ಉಳಿದ = n;
        ಆಗ (ಉಳಿದ > 0) {
            ಸ್ಥಿರಾಂಕ size_t vl = __riscv_vsetvl_e32m2(ಉಳಿದ);
            vfloat32m2_t v  = __riscv_vle32_v_f32m2(dst_ಅಡ್ಡಸಾಲು, vl);
            v               = rvv_tanh_approx_f32m2(v, vl);
            v               = __riscv_vfmul_vf_f32m2(v, softcap, vl);
            __riscv_vse32_v_f32m2(dst_ಅಡ್ಡಸಾಲು, v, vl);
            dst_ಅಡ್ಡಸಾಲು += vl;
            ಉಳಿದ -= vl;
        }
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ rvv_softmax_exp_inplace_f32(ದಶಮಾಂಶ * dst, int64_t n, ದಶಮಾಂಶ max_ಮೌಲ್ಯ) {
    ದಶಮಾಂಶ ಅಡ್ಡಸಾಲು_ಮೊತ್ತ = 0.0f;
    ಆಗ (n > 0) {
        ಸ್ಥಿರಾಂಕ size_t vl = __riscv_vsetvl_e32m2(n);
        vfloat32m2_t v  = __riscv_vle32_v_f32m2(dst, vl);
        v               = __riscv_vfsub_vf_f32m2(v, max_ಮೌಲ್ಯ, vl);
        v               = rvv_expf_approx_f32m2(v, vl);
        __riscv_vse32_v_f32m2(dst, v, vl);
        ಅಡ್ಡಸಾಲು_ಮೊತ್ತ += reduce_ಮೊತ್ತ_f32m2_vlen1024(v, vl);
        dst += vl;
        n -= vl;
    }
    ಹಿಂತಿರುಗಿ ಅಡ್ಡಸಾಲು_ಮೊತ್ತ;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ rvv_ಕೂಡಿಸು_max_inplace_f32(ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src, int64_t n) {
    ದಶಮಾಂಶ max_val = -INFINITY;
    ಆಗ (n > 0) {
        ಸ್ಥಿರಾಂಕ size_t vl   = __riscv_vsetvl_e32m4(n);
        vfloat32m4_t vdst = __riscv_vle32_v_f32m4(dst, vl);
        vfloat32m4_t vsrc = __riscv_vle32_v_f32m4(src, vl);
        vdst              = __riscv_vfadd_vv_f32m4(vdst, vsrc, vl);
        __riscv_vse32_v_f32m4(dst, vdst, vl);

        vfloat32m1_t ಬೀಜ = __riscv_vfmv_v_f_f32m1(max_val, 1);
        ಬೀಜ              = __riscv_vfredmax_vs_f32m4_f32m1(vdst, ಬೀಜ, vl);
        max_val           = __riscv_vfmv_f_s_f32m1_f32(ಬೀಜ);

        dst += vl;
        src += vl;
        n -= vl;
    }
    ಹಿಂತಿರುಗಿ max_val;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ rvv_softcap_ಕೂಡಿಸು_max_inplace_f32(ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src, int64_t n, ದಶಮಾಂಶ softcap) {
    ಆದರೆ (softcap == 0.0f) {
        ಹಿಂತಿರುಗಿ rvv_ಕೂಡಿಸು_max_inplace_f32(dst, src, n);
    }

    ದಶಮಾಂಶ max_val = -INFINITY;
    ಆಗ (n > 0) {
        ಸ್ಥಿರಾಂಕ size_t vl   = __riscv_vsetvl_e32m2(n);
        vfloat32m2_t vdst = __riscv_vle32_v_f32m2(dst, vl);
        vfloat32m2_t vsrc = __riscv_vle32_v_f32m2(src, vl);
        vdst              = rvv_tanh_approx_f32m2(vdst, vl);
        vdst              = __riscv_vfmul_vf_f32m2(vdst, softcap, vl);
        vdst              = __riscv_vfadd_vv_f32m2(vdst, vsrc, vl);
        __riscv_vse32_v_f32m2(dst, vdst, vl);

        vfloat32m1_t ಬೀಜ = __riscv_vfmv_v_f_f32m1(max_val, 1);
        ಬೀಜ              = __riscv_vfredmax_vs_f32m2_f32m1(vdst, ಬೀಜ, vl);
        max_val           = __riscv_vfmv_f_s_f32m1_f32(ಬೀಜ);

        dst += vl;
        src += vl;
        n -= vl;
    }
    ಹಿಂತಿರುಗಿ max_val;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ rvv_zero_f32(ದಶಮಾಂಶ * dst, int64_t n) {
    ಆಗ (n > 0) {
        ಸ್ಥಿರಾಂಕ size_t       vl = __riscv_vsetvl_e32m4(n);
        ಸ್ಥಿರಾಂಕ vfloat32m4_t z  = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        __riscv_vse32_v_f32m4(dst, z, vl);
        dst += vl;
        n -= vl;
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ rvv_scale_f32(ದಶಮಾಂಶ * dst, ದಶಮಾಂಶ scale, int64_t n) {
    ಆಗ (n > 0) {
        ಸ್ಥಿರಾಂಕ size_t vl = __riscv_vsetvl_e32m4(n);
        vfloat32m4_t v  = __riscv_vle32_v_f32m4(dst, vl);
        v               = __riscv_vfmul_vf_f32m4(v, scale, vl);
        __riscv_vse32_v_f32m4(dst, v, vl);
        dst += vl;
        n -= vl;
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ rvv_ಕೂಡಿಸು_inplace_f32(ದಶಮಾಂಶ *       dst,
                                       int64_t       dst_stride,
                                       ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src,
                                       int64_t       src_stride,
                                       int64_t       tile_ಅಡ್ಡಸಾಲುಗಳು,
                                       int64_t       n) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ tq = 0; tq < tile_ಅಡ್ಡಸಾಲುಗಳು; ++tq, dst += dst_stride, src += src_stride) {
        int64_t       ಉಳಿದ = n;
        ದಶಮಾಂಶ *       dst_ಅಡ್ಡಸಾಲು   = dst;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ಅಡ್ಡಸಾಲು   = src;
        ಆಗ (ಉಳಿದ > 0) {
            ಸ್ಥಿರಾಂಕ size_t vl   = __riscv_vsetvl_e32m4(ಉಳಿದ);
            vfloat32m4_t vdst = __riscv_vle32_v_f32m4(dst_ಅಡ್ಡಸಾಲು, vl);
            vfloat32m4_t vsrc = __riscv_vle32_v_f32m4(src_ಅಡ್ಡಸಾಲು, vl);
            vdst              = __riscv_vfadd_vv_f32m4(vdst, vsrc, vl);
            __riscv_vse32_v_f32m4(dst_ಅಡ್ಡಸಾಲು, vdst, vl);
            dst_ಅಡ್ಡಸಾಲು += vl;
            src_ಅಡ್ಡಸಾಲು += vl;
            ಉಳಿದ -= vl;
        }
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ rvv_max_f32(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src, int64_t n) {
    ದಶಮಾಂಶ max_val = -INFINITY;
    ಆಗ (n > 0) {
        ಸ್ಥಿರಾಂಕ size_t       vl   = __riscv_vsetvl_e32m4(n);
        ಸ್ಥಿರಾಂಕ vfloat32m4_t v    = __riscv_vle32_v_f32m4(src, vl);
        vfloat32m1_t       ಬೀಜ = __riscv_vfmv_v_f_f32m1(max_val, 1);
        ಬೀಜ                    = __riscv_vfredmax_vs_f32m4_f32m1(v, ಬೀಜ, vl);
        max_val                 = __riscv_vfmv_f_s_f32m1_f32(ಬೀಜ);
        src += vl;
        n -= vl;
    }
    ಹಿಂತಿರುಗಿ max_val;
}

ಸ್ಥಿರ ಶೂನ್ಯ rvv_pack_f32_as_scaled_f16(ಶೂನ್ಯ *       dst,
                                       int64_t      dst_ಅಡ್ಡಸಾಲು_stride,
                                       ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src,
                                       int64_t      src_ಅಡ್ಡಸಾಲು_stride,
                                       int64_t      tile_ಅಡ್ಡಸಾಲುಗಳು,
                                       int64_t      n,
                                       ದಶಮಾಂಶ        scale) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ tq = 0; tq < tile_ಅಡ್ಡಸಾಲುಗಳು; ++tq) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಅಡ್ಡಸಾಲು_ptr     = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src + tq * src_ಅಡ್ಡಸಾಲು_stride);
        _Float16 *    dst_ಅಡ್ಡಸಾಲು_ptr = (_Float16 *) ((ಅಕ್ಷರ *) dst + tq * dst_ಅಡ್ಡಸಾಲು_stride);
        int64_t       ಉಳಿದ   = n;
        ಆಗ (ಉಳಿದ > 0) {
            ಸ್ಥಿರಾಂಕ size_t vl        = __riscv_vsetvl_e32m4(ಉಳಿದ);
            vfloat32m4_t v32       = __riscv_vle32_v_f32m4(ಅಡ್ಡಸಾಲು_ptr, vl);
            v32                    = __riscv_vfmul_vf_f32m4(v32, scale, vl);
            ಸ್ಥಿರಾಂಕ vfloat16m2_t v16 = __riscv_vfncvt_f_f_w_f16m2(v32, vl);
            __riscv_vse16_v_f16m2(dst_ಅಡ್ಡಸಾಲು_ptr, v16, vl);
            dst_ಅಡ್ಡಸಾಲು_ptr += vl;
            ಅಡ್ಡಸಾಲು_ptr += vl;
            ಉಳಿದ -= vl;
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ rvv_pack_scaled_f16_as_f32(ಶೂನ್ಯ *       dst,
                                       int64_t      dst_ಅಡ್ಡಸಾಲು_stride,
                                       ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src,
                                       int64_t      src_ಅಡ್ಡಸಾಲು_stride,
                                       int64_t      tile_ಅಡ್ಡಸಾಲುಗಳು,
                                       int64_t      n,
                                       ದಶಮಾಂಶ        scale) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ tq = 0; tq < tile_ಅಡ್ಡಸಾಲುಗಳು; ++tq) {
        ಸ್ಥಿರಾಂಕ _Float16 * ಅಡ್ಡಸಾಲು_ptr     = (ಸ್ಥಿರಾಂಕ _Float16 *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src + tq * src_ಅಡ್ಡಸಾಲು_stride);
        ದಶಮಾಂಶ *          dst_ಅಡ್ಡಸಾಲು_ptr = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst + tq * dst_ಅಡ್ಡಸಾಲು_stride);
        int64_t          ಉಳಿದ   = n;
        ಆಗ (ಉಳಿದ > 0) {
            ಸ್ಥಿರಾಂಕ size_t       vl  = __riscv_vsetvl_e16m2(ಉಳಿದ);
            ಸ್ಥಿರಾಂಕ vfloat16m2_t v16 = __riscv_vle16_v_f16m2(ಅಡ್ಡಸಾಲು_ptr, vl);
            vfloat32m4_t       v32 = __riscv_vfwcvt_f_f_v_f32m4(v16, vl);
            v32                    = __riscv_vfmul_vf_f32m4(v32, scale, vl);
            __riscv_vse32_v_f32m4(dst_ಅಡ್ಡಸಾಲು_ptr, v32, vl);
            dst_ಅಡ್ಡಸಾಲು_ptr += vl;
            ಅಡ್ಡಸಾಲು_ptr += vl;
            ಉಳಿದ -= vl;
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ rvv_pack_scaled_f32_as_f32(ಶೂನ್ಯ *       dst,
                                       int64_t      dst_ಅಡ್ಡಸಾಲು_stride,
                                       ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src,
                                       int64_t      src_ಅಡ್ಡಸಾಲು_stride,
                                       int64_t      tile_ಅಡ್ಡಸಾಲುಗಳು,
                                       int64_t      n,
                                       ದಶಮಾಂಶ *      scale) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ tq = 0; tq < tile_ಅಡ್ಡಸಾಲುಗಳು; ++tq) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಅಡ್ಡಸಾಲು_ptr     = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src + tq * src_ಅಡ್ಡಸಾಲು_stride);
        ದಶಮಾಂಶ *       dst_ಅಡ್ಡಸಾಲು_ptr = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst + tq * dst_ಅಡ್ಡಸಾಲು_stride);
        int64_t       ಉಳಿದ   = n;
        ಆಗ (ಉಳಿದ > 0) {
            ಸ್ಥಿರಾಂಕ size_t vl  = __riscv_vsetvl_e32m4(ಉಳಿದ);
            vfloat32m4_t v32 = __riscv_vle32_v_f32m4(ಅಡ್ಡಸಾಲು_ptr, vl);
            v32              = __riscv_vfmul_vf_f32m4(v32, scale[tq], vl);
            __riscv_vse32_v_f32m4(dst_ಅಡ್ಡಸಾಲು_ptr, v32, vl);
            dst_ಅಡ್ಡಸಾಲು_ptr += vl;
            ಅಡ್ಡಸಾಲು_ptr += vl;
            ಉಳಿದ -= vl;
        }
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ rvv_transposed_s32_mn_ಗಮ್ಯ_nm(int8_t * dst,
                                               int64_t  n_dst_stride,
                                               int8_t * src,
                                               int64_t  m_src_stride,
                                               int64_t  m,
                                               int64_t  n) {
    int8_t * ಒಳ  = src;
    int8_t * ಹೊರಹೋಗು = dst;

    __asm__ ಅಸ್ಥಿರ(
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

        : [src] "+r"(ಒಳ), [dst] "+r"(ಹೊರಹೋಗು), [isz0] "+r"(m)
        : [isz1] "r"(n), [is0] "r"(m_src_stride), [os0] "r"(n_dst_stride)
        : "cc", "t0", "t1", "t2", "t3", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "a1");
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ rvv_transposed_s16_mn_ಗಮ್ಯ_nm(int8_t * dst,
                                               int64_t  n_dst_stride,
                                               int8_t * src,
                                               int64_t  m_src_stride,
                                               int64_t  m,
                                               int64_t  n) {
    int8_t * ಒಳ  = src;
    int8_t * ಹೊರಹೋಗು = dst;

    __asm__ ಅಸ್ಥಿರ(
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

        : [src] "+r"(ಒಳ), [dst] "+r"(ಹೊರಹೋಗು), [isz0] "+r"(m)
        : [isz1] "r"(n), [is0] "r"(m_src_stride), [os0] "r"(n_dst_stride)
        : "cc", "t0", "t1", "t2", "t3", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "a1");
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ rvv_qk_dot_tile_f16_x1(ದಶಮಾಂಶ *          dst,
                                          ಸ್ಥಿರಾಂಕ _Float16 * q_ಅಡ್ಡಸಾಲು,
                                          ಸ್ಥಿರಾಂಕ _Float16 * k_pack,
                                          int64_t          dk,
                                          int64_t          kv_tile) {
    ಸ್ಥಿರಾಂಕ size_t vl  = __riscv_vsetvl_e16m1(kv_tile);
    vfloat32m2_t acc = __riscv_vfmv_v_f_f32m2(0.0f, vl);

    ಗಾಗಿ (int64_t d = 0; d < dk; ++d) {
        ಸ್ಥಿರಾಂಕ vfloat16m1_t k_vec = __riscv_vle16_v_f16m1(k_pack + d * ggml_fa_tile_config::KV, vl);
        acc                      = __riscv_vfwmacc_vf_f32m2(acc, q_ಅಡ್ಡಸಾಲು[d], k_vec, vl);
    }

    __riscv_vse32_v_f32m2(dst, acc, vl);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ rvv_qk_dot_tile_f16_x4(ದಶಮಾಂಶ *          dst0,
                                          ದಶಮಾಂಶ *          dst1,
                                          ದಶಮಾಂಶ *          dst2,
                                          ದಶಮಾಂಶ *          dst3,
                                          ಸ್ಥಿರಾಂಕ _Float16 * q0,
                                          ಸ್ಥಿರಾಂಕ _Float16 * q1,
                                          ಸ್ಥಿರಾಂಕ _Float16 * q2,
                                          ಸ್ಥಿರಾಂಕ _Float16 * q3,
                                          ಸ್ಥಿರಾಂಕ _Float16 * k_pack,
                                          int64_t          dk,
                                          int64_t          kv_tile) {
    ಸ್ಥಿರಾಂಕ size_t vl   = __riscv_vsetvl_e16m1(kv_tile);
    vfloat32m2_t acc0 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    vfloat32m2_t acc1 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    vfloat32m2_t acc2 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    vfloat32m2_t acc3 = __riscv_vfmv_v_f_f32m2(0.0f, vl);

    ಗಾಗಿ (int64_t d = 0; d < dk; ++d) {
        ಸ್ಥಿರಾಂಕ vfloat16m1_t k_vec = __riscv_vle16_v_f16m1(k_pack + d * ggml_fa_tile_config::KV, vl);
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

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ rvv_pv_accumulate_f16_x1(ದಶಮಾಂಶ *          dst,
                                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *    prob,
                                            ಸ್ಥಿರಾಂಕ _Float16 * v_pack,
                                            int64_t          kv_tile,
                                            int64_t          dv) {
    int64_t d_ಎಡ = dv;
    int64_t d_off  = 0;

    ಆಗ (d_ಎಡ > 0) {
        ಸ್ಥಿರಾಂಕ size_t vl  = __riscv_vsetvl_e16m2(d_ಎಡ);
        vfloat32m4_t acc = __riscv_vle32_v_f32m4(dst + d_off, vl);

        ಗಾಗಿ (int64_t tk = 0; tk < kv_tile; ++tk) {
            ಸ್ಥಿರಾಂಕ vfloat16m2_t v16 = __riscv_vle16_v_f16m2(v_pack + tk * dv + d_off, vl);
            ಸ್ಥಿರಾಂಕ vfloat32m4_t v32 = __riscv_vfwcvt_f_f_v_f32m4(v16, vl);
            acc                    = __riscv_vfmacc_vf_f32m4(acc, prob[tk], v32, vl);
        }

        __riscv_vse32_v_f32m4(dst + d_off, acc, vl);
        d_ಎಡ -= vl;
        d_off += vl;
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ rvv_pv_accumulate_f16_x4(ದಶಮಾಂಶ *          dst0,
                                            ದಶಮಾಂಶ *          dst1,
                                            ದಶಮಾಂಶ *          dst2,
                                            ದಶಮಾಂಶ *          dst3,
                                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *    prob0,
                                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *    prob1,
                                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *    prob2,
                                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *    prob3,
                                            ಸ್ಥಿರಾಂಕ _Float16 * v_pack,
                                            int64_t          kv_tile,
                                            int64_t          dv) {
    int64_t d_ಎಡ = dv;
    int64_t d_off  = 0;

    ಆಗ (d_ಎಡ > 0) {
        ಸ್ಥಿರಾಂಕ size_t vl   = __riscv_vsetvl_e16m2(d_ಎಡ);
        vfloat32m4_t acc0 = __riscv_vle32_v_f32m4(dst0 + d_off, vl);
        vfloat32m4_t acc1 = __riscv_vle32_v_f32m4(dst1 + d_off, vl);
        vfloat32m4_t acc2 = __riscv_vle32_v_f32m4(dst2 + d_off, vl);
        vfloat32m4_t acc3 = __riscv_vle32_v_f32m4(dst3 + d_off, vl);

        ಗಾಗಿ (int64_t tk = 0; tk < kv_tile; ++tk) {
            ಸ್ಥಿರಾಂಕ vfloat16m2_t v16 = __riscv_vle16_v_f16m2(v_pack + tk * dv + d_off, vl);
            ಸ್ಥಿರಾಂಕ vfloat32m4_t v32 = __riscv_vfwcvt_f_f_v_f32m4(v16, vl);
            acc0                   = __riscv_vfmacc_vf_f32m4(acc0, prob0[tk], v32, vl);
            acc1                   = __riscv_vfmacc_vf_f32m4(acc1, prob1[tk], v32, vl);
            acc2                   = __riscv_vfmacc_vf_f32m4(acc2, prob2[tk], v32, vl);
            acc3                   = __riscv_vfmacc_vf_f32m4(acc3, prob3[tk], v32, vl);
        }

        __riscv_vse32_v_f32m4(dst0 + d_off, acc0, vl);
        __riscv_vse32_v_f32m4(dst1 + d_off, acc1, vl);
        __riscv_vse32_v_f32m4(dst2 + d_off, acc2, vl);
        __riscv_vse32_v_f32m4(dst3 + d_off, acc3, vl);
        d_ಎಡ -= vl;
        d_off += vl;
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ rvv_qk_dot_tile(ದಶಮಾಂಶ *       dst,
                                   ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * q_ಅಡ್ಡಸಾಲು,
                                   ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * k_pack,
                                   int64_t       dk,
                                   int64_t       kv_tile,
                                   ದಶಮಾಂಶ         scale) {
    ಸ್ಥಿರಾಂಕ size_t vl  = __riscv_vsetvl_e32m4(kv_tile);
    vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);

    ಗಾಗಿ (int64_t d = 0; d < dk; ++d) {
        ಸ್ಥಿರಾಂಕ vfloat32m4_t k_vec = __riscv_vle32_v_f32m4(k_pack + d * kv_tile, vl);
        acc                      = __riscv_vfmacc_vf_f32m4(acc, q_ಅಡ್ಡಸಾಲು[d] * scale, k_vec, vl);
    }

    __riscv_vse32_v_f32m4(dst, acc, vl);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ rvv_pv_accumulate(ದಶಮಾಂಶ *       dst,
                                     ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * prob,
                                     ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * v_pack,
                                     int64_t       kv_tile,
                                     int64_t       dv) {
    int64_t d_ಎಡ = dv;
    int64_t d_off  = 0;

    ಆಗ (d_ಎಡ > 0) {
        ಸ್ಥಿರಾಂಕ size_t vl  = __riscv_vsetvl_e32m4(d_ಎಡ);
        vfloat32m4_t acc = __riscv_vle32_v_f32m4(dst + d_off, vl);

        ಗಾಗಿ (int64_t tk = 0; tk < kv_tile; ++tk) {
            ಸ್ಥಿರಾಂಕ vfloat32m4_t v_vec = __riscv_vle32_v_f32m4(v_pack + tk * dv + d_off, vl);
            acc                      = __riscv_vfmacc_vf_f32m4(acc, prob[tk], v_vec, vl);
        }

        __riscv_vse32_v_f32m4(dst + d_off, acc, vl);
        d_ಎಡ -= vl;
        d_off += vl;
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ಕ್ರಮಬದಲಾವಣೆ_transpose_impl(ಸ್ಥಿರಾಂಕ ggml_tensor * src0,
                                   ggml_tensor *       dst,
                                   int64_t             batch,
                                   int64_t             m,
                                   int64_t             n,
                                   int64_t             batch_stride,
                                   int64_t             m_src_stride,
                                   int64_t             n_src_stride,
                                   int64_t             n_dst_stride,
                                   ಪೂರ್ಣಾಂಕ                 ith,
                                   ಪೂರ್ಣಾಂಕ                 nth) {
    GGML_ASSERT(n_src_stride == ಗಾತ್ರ(int32_t) || n_src_stride == ಗಾತ್ರ(int16_t));

    ಆದರೆ (n_src_stride == ಗಾತ್ರ(int32_t)) {
        ಗಾಗಿ (int64_t bi = ith; bi < batch; bi += nth) {
            rvv_transposed_s32_mn_ಗಮ್ಯ_nm((int8_t *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + bi * batch_stride), n_dst_stride,
                                        (int8_t *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + bi * batch_stride), m_src_stride, m, n);
        }
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (n_src_stride == ಗಾತ್ರ(int16_t)) {
        ಗಾಗಿ (int64_t bi = ith; bi < batch; bi += nth) {
            rvv_transposed_s16_mn_ಗಮ್ಯ_nm((int8_t *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + bi * batch_stride), n_dst_stride,
                                        (int8_t *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + bi * batch_stride), m_src_stride, m, n);
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_ABORT("not implemented");
    }
}

ಮಾದರಿ <size_t QLEN>
ಸ್ಥಿರ ಶೂನ್ಯ flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_mrow(ದಶಮಾಂಶ **            pq,
                                                                  ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *        k_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು,
                                                                  ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *        v_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು,
                                                                  ಸ್ಥಿರಾಂಕ ggml_fp16_t * mp,
                                                                  ದಶಮಾಂಶ **            sinks,
                                                                  ದಶಮಾಂಶ **            dst,
                                                                  ದಶಮಾಂಶ               scale,
                                                                  ದಶಮಾಂಶ               logit_softcap,
                                                                  ದಶಮಾಂಶ               ಇಳಿಜಾರು,
                                                                  int64_t             nek1,
                                                                  int64_t             nbk1,
                                                                  int64_t             nbv1,
                                                                  int64_t             DV,
                                                                  int64_t             DK,
                                                                  ಶೂನ್ಯ *              tcm_ಬಫರ್,
                                                                  size_t              tcm_ಬಫರ್_ಗಾತ್ರ) {
    GGML_ASSERT(flash_attn_ext_supported_shape_vlen1024_vf16(DK, DV));
    ದಶಮಾಂಶ S[QLEN] = { 0.0f };       //ಮೊತ್ತ
    ದಶಮಾಂಶ M[QLEN] = { -INFINITY };  // maximum KQ value

    _Float16 *   kq16_ಬಫರ್          = (_Float16 *) tcm_ಬಫರ್;
    _Float16 *   qv_ಬಫರ್            = kq16_ಬಫರ್ + QLEN * DV;
    ಸ್ಥಿರಾಂಕ size_t qkv_ತಾತ್ಕಾಲಿಕ_ಬಫರ್_ಗಾತ್ರ = (QLEN * DV + QLEN * DK) * ಗಾತ್ರ(_Float16);
    ಅಕ್ಷರ *       kv_tile_ಬಫರ್       = (ಅಕ್ಷರ *) (qv_ಬಫರ್ + QLEN * DK);

    {
        vfloat16m2_t VKQ16_v = __riscv_vfmv_v_f_f16m2(0.0f, DV);
        ಗಾಗಿ (int64_t i = 0; i < QLEN; ++i) {
            __riscv_vse16_v_f16m2(kq16_ಬಫರ್ + i * DV, VKQ16_v, DV);
            vfloat16m2_t Q_q_v = __riscv_vfncvt_f_f_w_f16m2(__riscv_vle32_v_f32m4(pq[i], DK), DK);
            __riscv_vse16_v_f16m2(qv_ಬಫರ್ + i * DK, Q_q_v, DK);
        }
    }

    ಸ್ಥಿರಾಂಕ uintptr_t scratch_addr = reinterpret_cast<uintptr_t>(kv_tile_ಬಫರ್);
    ಸ್ಥಿರಾಂಕ size_t    scratch_ಗಾತ್ರ = tcm_ಬಫರ್_ಗಾತ್ರ > qkv_ತಾತ್ಕಾಲಿಕ_ಬಫರ್_ಗಾತ್ರ ? tcm_ಬಫರ್_ಗಾತ್ರ - qkv_ತಾತ್ಕಾಲಿಕ_ಬಫರ್_ಗಾತ್ರ : 0;
    ಸ್ಥಿರಾಂಕ uintptr_t kq_tile_addr = align_ಮೇಲೆ(scratch_addr, alignof(ದಶಮಾಂಶ));
    ಸ್ಥಿರಾಂಕ size_t    scratch_ಪೂರ್ವಪ್ರತ್ಯಯ = kq_tile_addr - scratch_addr;
    ಸ್ಥಿರಾಂಕ size_t    packed_tile_ಗಾತ್ರ =
        QLEN * ಗಾತ್ರ(ದಶಮಾಂಶ) + DK * ಗಾತ್ರ(_Float16) + DV * ಗಾತ್ರ(_Float16) + ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ int64_t max_ic_tile_ಹೆಜ್ಜೆ = ((int64_t) __riscv_vsetvlmax_e16m1()) & ~((int64_t) 7);
    ಸ್ಥಿರಾಂಕ int64_t max_fit_by_tcm =
        scratch_ಗಾತ್ರ > scratch_ಪೂರ್ವಪ್ರತ್ಯಯ ? (int64_t) ((scratch_ಗಾತ್ರ - scratch_ಪೂರ್ವಪ್ರತ್ಯಯ) / packed_tile_ಗಾತ್ರ) : 0;
    ಸ್ಥಿರಾಂಕ int64_t ic_tile_ಹೆಜ್ಜೆ = std::min(max_ic_tile_ಹೆಜ್ಜೆ, max_fit_by_tcm) & ~((int64_t) 7);

    ಸ್ಥಿರಾಂಕ uintptr_t k_tile_addr  = kq_tile_addr + QLEN * ic_tile_ಹೆಜ್ಜೆ * ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ uintptr_t v_tile_addr  = k_tile_addr + DK * ic_tile_ಹೆಜ್ಜೆ * ಗಾತ್ರ(_Float16);
    ಸ್ಥಿರಾಂಕ uintptr_t mv_tile_addr = v_tile_addr + ic_tile_ಹೆಜ್ಜೆ * DV * ಗಾತ್ರ(_Float16);

    ಆದರೆ (ic_tile_ಹೆಜ್ಜೆ >= 8) {
        ದಶಮಾಂಶ *    kq_tile_ಬಫರ್ = reinterpret_cast<ದಶಮಾಂಶ *>(kq_tile_addr);
        _Float16 * k_tile_pack    = reinterpret_cast<_Float16 *>(k_tile_addr);
        _Float16 * v_tile_pack    = reinterpret_cast<_Float16 *>(v_tile_addr);
        ದಶಮಾಂಶ *    mv_tile_pack   = reinterpret_cast<ದಶಮಾಂಶ *>(mv_tile_addr);

        ಸ್ಥಿರಾಂಕ int64_t k_tile_byte_stride = ic_tile_ಹೆಜ್ಜೆ * (int64_t) ಗಾತ್ರ(_Float16);

        int64_t ic_ಹೆಜ್ಜೆ = 0;
        ಗಾಗಿ (int64_t ic = 0; ic < nek1; ++ic) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ mv = mp ? ಇಳಿಜಾರು * ((_Float16 *) mp)[ic] : 0.0f;

            ಆದರೆ (mv != -INFINITY) {
                ಸ್ಥಿರಾಂಕ _Float16 * k_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ _Float16 *) (k_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು + ic * nbk1);
                ಸ್ಥಿರಾಂಕ _Float16 * v_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ _Float16 *) (v_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು + ic * nbv1);

                ಸ್ಥಿರಾಂಕ vfloat16m2_t k_ದತ್ತಾಂಶ_v = __riscv_vle16_v_f16m2(k_ದತ್ತಾಂಶ, DK);
                ಸ್ಥಿರಾಂಕ vfloat16m2_t v_ದತ್ತಾಂಶ_v = __riscv_vle16_v_f16m2(v_ದತ್ತಾಂಶ, DV);
                __riscv_vsse16_v_f16m2(k_tile_pack + ic_ಹೆಜ್ಜೆ, k_tile_byte_stride, k_ದತ್ತಾಂಶ_v, DK);
                __riscv_vse16_v_f16m2(v_tile_pack + ic_ಹೆಜ್ಜೆ * DV, v_ದತ್ತಾಂಶ_v, DV);
                mv_tile_pack[ic_ಹೆಜ್ಜೆ] = mv;
                ic_ಹೆಜ್ಜೆ++;
            }

            ಆದರೆ (ic_ಹೆಜ್ಜೆ > 0 && (ic_ಹೆಜ್ಜೆ == ic_tile_ಹೆಜ್ಜೆ || ic == (nek1 - 1))) {
                ಆದರೆ constexpr (QLEN == 4) {
                    ಸ್ಥಿರಾಂಕ size_t qk_vl   = __riscv_vsetvl_e16m1(ic_ಹೆಜ್ಜೆ);
                    vfloat32m2_t qk_acc0 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);
                    vfloat32m2_t qk_acc1 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);
                    vfloat32m2_t qk_acc2 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);
                    vfloat32m2_t qk_acc3 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);

                    ಗಾಗಿ (int64_t d = 0; d < DK; ++d) {
                        ಸ್ಥಿರಾಂಕ vfloat16m1_t k_vec = __riscv_vle16_v_f16m1(k_tile_pack + d * ic_tile_ಹೆಜ್ಜೆ, qk_vl);
                        qk_acc0 = __riscv_vfwmacc_vf_f32m2(qk_acc0, qv_ಬಫರ್[0 * DK + d], k_vec, qk_vl);
                        qk_acc1 = __riscv_vfwmacc_vf_f32m2(qk_acc1, qv_ಬಫರ್[1 * DK + d], k_vec, qk_vl);
                        qk_acc2 = __riscv_vfwmacc_vf_f32m2(qk_acc2, qv_ಬಫರ್[2 * DK + d], k_vec, qk_vl);
                        qk_acc3 = __riscv_vfwmacc_vf_f32m2(qk_acc3, qv_ಬಫರ್[3 * DK + d], k_vec, qk_vl);
                    }

                    qk_acc0 = __riscv_vfmul_vf_f32m2(qk_acc0, scale, qk_vl);
                    qk_acc1 = __riscv_vfmul_vf_f32m2(qk_acc1, scale, qk_vl);
                    qk_acc2 = __riscv_vfmul_vf_f32m2(qk_acc2, scale, qk_vl);
                    qk_acc3 = __riscv_vfmul_vf_f32m2(qk_acc3, scale, qk_vl);

                    __riscv_vse32_v_f32m2(kq_tile_ಬಫರ್ + 0 * ic_tile_ಹೆಜ್ಜೆ, qk_acc0, qk_vl);
                    __riscv_vse32_v_f32m2(kq_tile_ಬಫರ್ + 1 * ic_tile_ಹೆಜ್ಜೆ, qk_acc1, qk_vl);
                    __riscv_vse32_v_f32m2(kq_tile_ಬಫರ್ + 2 * ic_tile_ಹೆಜ್ಜೆ, qk_acc2, qk_vl);
                    __riscv_vse32_v_f32m2(kq_tile_ಬಫರ್ + 3 * ic_tile_ಹೆಜ್ಜೆ, qk_acc3, qk_vl);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    static_assert(QLEN == 2, "unsupported QLEN");

                    ಸ್ಥಿರಾಂಕ size_t qk_vl   = __riscv_vsetvl_e16m1(ic_ಹೆಜ್ಜೆ);
                    vfloat32m2_t qk_acc0 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);
                    vfloat32m2_t qk_acc1 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);

                    ಗಾಗಿ (int64_t d = 0; d < DK; ++d) {
                        ಸ್ಥಿರಾಂಕ vfloat16m1_t k_vec = __riscv_vle16_v_f16m1(k_tile_pack + d * ic_tile_ಹೆಜ್ಜೆ, qk_vl);
                        qk_acc0 = __riscv_vfwmacc_vf_f32m2(qk_acc0, qv_ಬಫರ್[0 * DK + d], k_vec, qk_vl);
                        qk_acc1 = __riscv_vfwmacc_vf_f32m2(qk_acc1, qv_ಬಫರ್[1 * DK + d], k_vec, qk_vl);
                    }

                    qk_acc0 = __riscv_vfmul_vf_f32m2(qk_acc0, scale, qk_vl);
                    qk_acc1 = __riscv_vfmul_vf_f32m2(qk_acc1, scale, qk_vl);

                    __riscv_vse32_v_f32m2(kq_tile_ಬಫರ್ + 0 * ic_tile_ಹೆಜ್ಜೆ, qk_acc0, qk_vl);
                    __riscv_vse32_v_f32m2(kq_tile_ಬಫರ್ + 1 * ic_tile_ಹೆಜ್ಜೆ, qk_acc1, qk_vl);
                }

                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < QLEN; ++i) {
                    ದಶಮಾಂಶ *     ಅಡ್ಡಸಾಲು_ptr = kq_tile_ಬಫರ್ + i * ic_tile_ಹೆಜ್ಜೆ;
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ tile_max =
                        rvv_softcap_ಕೂಡಿಸು_max_inplace_f32(ಅಡ್ಡಸಾಲು_ptr, mv_tile_pack, ic_ಹೆಜ್ಜೆ, logit_softcap);

                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ Mold = M[i];

                    ಆದರೆ (tile_max > Mold) {
                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ms = expf(Mold - tile_max);
                        M[i]           = tile_max;
                        S[i] *= ms;

                        vfloat16m2_t VKQ16_v = __riscv_vle16_v_f16m2(kq16_ಬಫರ್ + i * DV, DV);
                        VKQ16_v              = __riscv_vfmul_vf_f16m2(VKQ16_v, (_Float16) ms, DV);
                        __riscv_vse16_v_f16m2(kq16_ಬಫರ್ + i * DV, VKQ16_v, DV);
                    }

                    S[i] += rvv_softmax_exp_inplace_f32(ಅಡ್ಡಸಾಲು_ptr, ic_ಹೆಜ್ಜೆ, M[i]);
                }

                ಆದರೆ constexpr (QLEN == 4) {
                    vfloat16m2_t pv_acc0 = __riscv_vle16_v_f16m2(kq16_ಬಫರ್ + 0 * DV, DV);
                    vfloat16m2_t pv_acc1 = __riscv_vle16_v_f16m2(kq16_ಬಫರ್ + 1 * DV, DV);
                    vfloat16m2_t pv_acc2 = __riscv_vle16_v_f16m2(kq16_ಬಫರ್ + 2 * DV, DV);
                    vfloat16m2_t pv_acc3 = __riscv_vle16_v_f16m2(kq16_ಬಫರ್ + 3 * DV, DV);

                    ಗಾಗಿ (int64_t tk = 0; tk < ic_ಹೆಜ್ಜೆ; ++tk) {
                        ಸ್ಥಿರಾಂಕ vfloat16m2_t v16 = __riscv_vle16_v_f16m2(v_tile_pack + tk * DV, DV);
                        pv_acc0 =
                            __riscv_vfmacc_vf_f16m2(pv_acc0, (_Float16) kq_tile_ಬಫರ್[0 * ic_tile_ಹೆಜ್ಜೆ + tk], v16, DV);
                        pv_acc1 =
                            __riscv_vfmacc_vf_f16m2(pv_acc1, (_Float16) kq_tile_ಬಫರ್[1 * ic_tile_ಹೆಜ್ಜೆ + tk], v16, DV);
                        pv_acc2 =
                            __riscv_vfmacc_vf_f16m2(pv_acc2, (_Float16) kq_tile_ಬಫರ್[2 * ic_tile_ಹೆಜ್ಜೆ + tk], v16, DV);
                        pv_acc3 =
                            __riscv_vfmacc_vf_f16m2(pv_acc3, (_Float16) kq_tile_ಬಫರ್[3 * ic_tile_ಹೆಜ್ಜೆ + tk], v16, DV);
                    }

                    __riscv_vse16_v_f16m2(kq16_ಬಫರ್ + 0 * DV, pv_acc0, DV);
                    __riscv_vse16_v_f16m2(kq16_ಬಫರ್ + 1 * DV, pv_acc1, DV);
                    __riscv_vse16_v_f16m2(kq16_ಬಫರ್ + 2 * DV, pv_acc2, DV);
                    __riscv_vse16_v_f16m2(kq16_ಬಫರ್ + 3 * DV, pv_acc3, DV);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    static_assert(QLEN == 2, "unsupported QLEN");
                    vfloat16m2_t pv_acc0 = __riscv_vle16_v_f16m2(kq16_ಬಫರ್ + 0 * DV, DV);
                    vfloat16m2_t pv_acc1 = __riscv_vle16_v_f16m2(kq16_ಬಫರ್ + 1 * DV, DV);

                    ಗಾಗಿ (int64_t tk = 0; tk < ic_ಹೆಜ್ಜೆ; ++tk) {
                        ಸ್ಥಿರಾಂಕ vfloat16m2_t v16 = __riscv_vle16_v_f16m2(v_tile_pack + tk * DV, DV);
                        pv_acc0 =
                            __riscv_vfmacc_vf_f16m2(pv_acc0, (_Float16) kq_tile_ಬಫರ್[0 * ic_tile_ಹೆಜ್ಜೆ + tk], v16, DV);
                        pv_acc1 =
                            __riscv_vfmacc_vf_f16m2(pv_acc1, (_Float16) kq_tile_ಬಫರ್[1 * ic_tile_ಹೆಜ್ಜೆ + tk], v16, DV);
                    }

                    __riscv_vse16_v_f16m2(kq16_ಬಫರ್ + 0 * DV, pv_acc0, DV);
                    __riscv_vse16_v_f16m2(kq16_ಬಫರ್ + 1 * DV, pv_acc1, DV);
                }

                ic_ಹೆಜ್ಜೆ = 0;
            }
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಗಾಗಿ (int64_t ic = 0; ic < nek1; ++ic) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ mv = mp ? ಇಳಿಜಾರು * ((_Float16 *) mp)[ic] : 0.0f;

            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * k_ದತ್ತಾಂಶ = k_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು + ic * nbk1;
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * v_ದತ್ತಾಂಶ = v_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು + ic * nbv1;

            vfloat16m2_t k_ದತ್ತಾಂಶ_v;
            vfloat16m2_t v_ದತ್ತಾಂಶ_v;

            ಆದರೆ (mv != -INFINITY) {
                k_ದತ್ತಾಂಶ_v = __riscv_vle16_v_f16m2((_Float16 *) k_ದತ್ತಾಂಶ, DK);
                v_ದತ್ತಾಂಶ_v = __riscv_vle16_v_f16m2((_Float16 *) v_ದತ್ತಾಂಶ, DV);
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಮುಂದುವರಿ;
            }

            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < QLEN; ++i) {
                vfloat16m2_t Q_q_v    = __riscv_vle16_v_f16m2(qv_ಬಫರ್ + i * DK, DK);
                vfloat32m4_t qk_acc_v = __riscv_vfwmul_vv_f32m4(k_ದತ್ತಾಂಶ_v, Q_q_v, DK);
                ದಶಮಾಂಶ        s        = reduce_ಮೊತ್ತ_f32m4_vlen1024(qk_acc_v, DK);
                s                     = s * scale;
                ಆದರೆ (logit_softcap != 0.0f) {
                    s = logit_softcap * tanhf(s);
                }
                s += mv;

                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ Mold = M[i];

                ದಶಮಾಂಶ ms = 1.0f;  // upon new higher max val, scale VKQ and KQ sum with this value
                ದಶಮಾಂಶ vs = 1.0f;  // post-softmax KQ value, expf(s - M)

                vfloat16m2_t VKQ16_v = __riscv_vle16_v_f16m2(kq16_ಬಫರ್ + i * DV, DV);
                ಆದರೆ (s > M[i]) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M[i] = s;
                    ms   = expf(Mold - M[i]);

                    // V = V*expf(Mold - M)
                    VKQ16_v = __riscv_vfmul_vf_f16m2(VKQ16_v, ms, DV);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M[i]);
                }
                VKQ16_v = __riscv_vfmacc_vf_f16m2(VKQ16_v, vs, v_ದತ್ತಾಂಶ_v, DV);
                __riscv_vse16_v_f16m2(kq16_ಬಫರ್ + i * DV, VKQ16_v, DV);
                S[i] = S[i] * ms + vs;  // scale and increment sum with partial sum
            }
        }
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < QLEN; ++i) {
        vfloat16m2_t VKQ16_v = __riscv_vle16_v_f16m2(kq16_ಬಫರ್ + i * DV, DV);
        vfloat32m4_t VKQ32_v = __riscv_vfwcvt_f_f_v_f32m4(VKQ16_v, DV);

        // sinks
        ಆದರೆ (sinks[i]) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ s = *(sinks[i]);

            ದಶಮಾಂಶ ms = 1.0f;
            ದಶಮಾಂಶ vs = 1.0f;

            ಆದರೆ (s > M[i]) {
                ms      = expf(M[i] - s);
                M[i]    = s;
                VKQ32_v = __riscv_vfmul_vf_f32m4(VKQ32_v, ms, DV);
            } ಇಲ್ಲದಿದ್ದರೆ {
                vs = expf(s - M[i]);
            }

            S[i] = S[i] * ms + vs;
        }

        // V /= S
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ S_inv = S[i] == 0.0f ? 0.0f : 1.0f / S[i];

        VKQ32_v = __riscv_vfmul_vf_f32m4(VKQ32_v, S_inv, DV);

        __riscv_vse32_v_f32m4(dst[i], VKQ32_v, DV);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_m1(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *       pq,
                                                                ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *        k_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು,
                                                                ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *        v_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು,
                                                                ಸ್ಥಿರಾಂಕ ggml_fp16_t * mp,
                                                                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *       sinks,
                                                                ದಶಮಾಂಶ *             dst,
                                                                ದಶಮಾಂಶ               scale,
                                                                ದಶಮಾಂಶ               logit_softcap,
                                                                ದಶಮಾಂಶ               ಇಳಿಜಾರು,
                                                                int64_t             nek1,
                                                                int64_t             nbk1,
                                                                int64_t             nbv1,
                                                                int64_t             DV,
                                                                int64_t             DK) {
    GGML_ASSERT(flash_attn_ext_supported_shape_vlen1024_vf16(DK, DV));

    ದಶಮಾಂಶ S = 0.0f;       //ಮೊತ್ತ
    ದಶಮಾಂಶ M = -INFINITY;  // maximum KQ value

    vfloat16m2_t VKQ16_v = __riscv_vfmv_v_f_f16m2(0.0f, DV);

    vfloat16m2_t Q_q_v = __riscv_vfncvt_f_f_w_f16m2(__riscv_vle32_v_f32m4(pq, DK), DK);

    ಗಾಗಿ (int64_t ic = 0; ic < nek1; ++ic) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ mv = mp ? ಇಳಿಜಾರು * ((_Float16 *) mp)[ic] : 0.0f;
        ಆದರೆ (mv == -INFINITY) {
            ಮುಂದುವರಿ;
        }

        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * k_ದತ್ತಾಂಶ = k_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು + ic * nbk1;

        vfloat16m2_t k_ದತ್ತಾಂಶ_v = __riscv_vle16_v_f16m2((_Float16 *) k_ದತ್ತಾಂಶ, DK);

        vfloat32m4_t qk_acc_v = __riscv_vfwmul_vv_f32m4(k_ದತ್ತಾಂಶ_v, Q_q_v, DK);
        ದಶಮಾಂಶ        s        = reduce_ಮೊತ್ತ_f32m4_vlen1024(qk_acc_v, DK);

        s = s * scale;  // scale KQ value

        ಆದರೆ (logit_softcap != 0.0f) {
            s = logit_softcap * tanhf(s);
        }

        s += mv;  // apply mask

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ Mold = M;

        ದಶಮಾಂಶ ms = 1.0f;  // upon new higher max val, scale VKQ and KQ sum with this value
        ದಶಮಾಂಶ vs = 1.0f;  // post-softmax KQ value, expf(s - M)

        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * v_ದತ್ತಾಂಶ = v_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು + ic * nbv1;

        vfloat16m2_t v_ದತ್ತಾಂಶ_v = __riscv_vle16_v_f16m2((_Float16 *) v_ದತ್ತಾಂಶ, DV);

        ಆದರೆ (s > M) {
            // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
            M  = s;
            ms = expf(Mold - M);

            // V = V*expf(Mold - M)
            VKQ16_v = __riscv_vfmul_vf_f16m2(VKQ16_v, ms, DV);
        } ಇಲ್ಲದಿದ್ದರೆ {
            // no new maximum, ms == 1.0f, vs != 1.0f
            vs = expf(s - M);
        }

        VKQ16_v = __riscv_vfmacc_vf_f16m2(VKQ16_v, vs, v_ದತ್ತಾಂಶ_v, DV);

        S = S * ms + vs;  // scale and increment sum with partial sum
    }

    vfloat32m4_t VKQ32_v = __riscv_vfwcvt_f_f_v_f32m4(VKQ16_v, DV);

    // sinks
    ಆದರೆ (sinks) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ s = *sinks;

        ದಶಮಾಂಶ ms = 1.0f;
        ದಶಮಾಂಶ vs = 1.0f;

        ಆದರೆ (s > M) {
            ms      = expf(M - s);
            M       = s;
            VKQ32_v = __riscv_vfmul_vf_f32m4(VKQ32_v, ms, DV);
        } ಇಲ್ಲದಿದ್ದರೆ {
            vs = expf(s - M);
        }

        S = S * ms + vs;
    }

    // V /= S
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ S_inv = S == 0.0f ? 0.0f : 1.0f / S;

    VKQ32_v = __riscv_vfmul_vf_f32m4(VKQ32_v, S_inv, DV);

    __riscv_vse32_v_f32m4(dst, VKQ32_v, DV);
}

}  // namespace

ಶೂನ್ಯ memcpy1d(ಶೂನ್ಯ * dst, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src, int64_t size) {
    size_t byte_ಗಾತ್ರ_all = size;
    size_t vlen          = __riscv_vlenb() * 8;
    ಆದರೆ (vlen == 256) {
        // 1024 bytes
        __asm__ ಅಸ್ಥಿರ(
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
            : [size] "r"(byte_ಗಾತ್ರ_all)
            : "cc", "t0", "t1");
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (vlen == 1024) {
        // 2048 bytes
        __asm__ ಅಸ್ಥಿರ(
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
            : [size] "r"(byte_ಗಾತ್ರ_all)
            : "cc", "t0", "t1", "t2", "t3", "t5");
    } ಇಲ್ಲದಿದ್ದರೆ {
        __asm__ ಅಸ್ಥಿರ(
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
            : [size] "r"(byte_ಗಾತ್ರ_all)
            : "cc", "t0", "t1", "t2", "t4", "t3");
    }
}

ಶೂನ್ಯ memcpy2d(ಶೂನ್ಯ * dst, int64_t dst_stride, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src, int64_t src_stride, int64_t tile_ಅಡ್ಡಸಾಲುಗಳು, int64_t size) {
    ಗಾಗಿ (int64_t i = 0; i < tile_ಅಡ್ಡಸಾಲುಗಳು; ++i) {
        memcpy1d((ಅಕ್ಷರ *) dst + i * dst_stride, (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src + i * src_stride, size);
    }
}

ಶೂನ್ಯ forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
                                                        ggml_tensor *               dst,
                                                        ಪೂರ್ಣಾಂಕ                         ir0,
                                                        ಪೂರ್ಣಾಂಕ                         ir1,
                                                        ಶೂನ್ಯ *                      tcm_ಬಫರ್,
                                                        size_t                      tcm_ಬಫರ್_ಗಾತ್ರ) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * q     = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * k     = dst->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * v     = dst->src[2];
    ಸ್ಥಿರಾಂಕ ggml_tensor * ಮುಖವಾಡ  = dst->src[3];
    ಸ್ಥಿರಾಂಕ ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q, ne)
    GGML_TENSOR_LOCALS(size_t, nbq, q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k, ne)
    GGML_TENSOR_LOCALS(size_t, nbk, k, nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v, ne)
    GGML_TENSOR_LOCALS(size_t, nbv, v, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t, nb, dst, nb)

    ಸ್ಥಿರಾಂಕ int64_t DK = nek0;
    ಸ್ಥಿರಾಂಕ int64_t DV = nev0;
    ಸ್ಥಿರಾಂಕ int64_t N  = neq1;

    GGML_ASSERT(flash_attn_ext_supported_shape_vlen1024_vf16(DK, DV));

    // broadcast factors
    ಸ್ಥಿರಾಂಕ int64_t rk2 = neq2 / nek2;
    ಸ್ಥಿರಾಂಕ int64_t rk3 = neq3 / nek3;

    ಸ್ಥಿರಾಂಕ int64_t rv2 = neq2 / nev2;
    ಸ್ಥಿರಾಂಕ int64_t rv3 = neq3 / nev3;

    // parallelize by q rows using ggml_vec_dot_f32

    ದಶಮಾಂಶ scale         = *((ದಶಮಾಂಶ *) dst->op_params + 0);
    ದಶಮಾಂಶ max_bias      = *((ದಶಮಾಂಶ *) dst->op_params + 1);
    ದಶಮಾಂಶ logit_softcap = *((ದಶಮಾಂಶ *) dst->op_params + 2);

    ಆದರೆ (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    ಸ್ಥಿರಾಂಕ uint32_t n_ತಲೆ      = neq2;
    ಸ್ಥಿರಾಂಕ uint32_t n_ತಲೆ_log2 = 1u << (uint32_t) floor(log2(n_ತಲೆ));

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m0 = powf(2.0f, -(max_bias) / n_ತಲೆ_log2);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m1 = powf(2.0f, -(max_bias / 2.0f) / n_ತಲೆ_log2);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ KV_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = DK * ಗಾತ್ರ(_Float16) + DV * ಗಾತ್ರ(_Float16);

    ಪೂರ್ಣಾಂಕ ith     = params->ith;
    ಪೂರ್ಣಾಂಕ ir_ಹೆಜ್ಜೆ = 1;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ir = ir0; ir < ir1; ir += ir_ಹೆಜ್ಜೆ) {
        // q indices
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iq3 = ir / (neq2 * neq1);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iq2 = (ir - iq3 * neq2 * neq1) / neq1;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iq1 = (ir - iq3 * neq2 * neq1 - iq2 * neq1);

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iq3_1 = (ir + 1) / (neq2 * neq1);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iq2_1 = (ir + 1 - iq3_1 * neq2 * neq1) / neq1;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iq1_1 = (ir + 1 - iq3_1 * neq2 * neq1 - iq2_1 * neq1);

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iq3_2 = (ir + 2) / (neq2 * neq1);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iq2_2 = (ir + 2 - iq3_2 * neq2 * neq1) / neq1;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iq1_2 = (ir + 2 - iq3_2 * neq2 * neq1 - iq2_2 * neq1);

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iq3_3 = (ir + 3) / (neq2 * neq1);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iq2_3 = (ir + 3 - iq3_3 * neq2 * neq1) / neq1;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iq1_3 = (ir + 3 - iq3_3 * neq2 * neq1 - iq2_3 * neq1);

        ಸ್ಥಿರಾಂಕ uint32_t h = iq2;  // head index
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ    ಇಳಿಜಾರು =
            (max_bias > 0.0f) ? h < n_ತಲೆ_log2 ? powf(m0, h + 1) : powf(m1, 2 * (h - n_ತಲೆ_log2) + 1) : 1.0f;

        ಸ್ಥಿರಾಂಕ ggml_fp16_t * mp =
            ಮುಖವಾಡ ? (ggml_fp16_t *) ((ಅಕ್ಷರ *) ಮುಖವಾಡ->ದತ್ತಾಂಶ + iq1 * ಮುಖವಾಡ->nb[1] + (iq2 % ಮುಖವಾಡ->ne[2]) * ಮುಖವಾಡ->nb[2] +
                                    (iq3 % ಮುಖವಾಡ->ne[3]) * ಮುಖವಾಡ->nb[3]) :
                   ಖಾಲಿ;

        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ mp_equal_2 = iq1_1 == iq1 && (iq2 % ಮುಖವಾಡ->ne[2]) == (iq2_1 % ಮುಖವಾಡ->ne[2]) &&
                                (iq3 % ಮುಖವಾಡ->ne[3]) == (iq3_1 % ಮುಖವಾಡ->ne[3]);

        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ mp_equal_4 = mp_equal_2 && iq1_2 == iq1 && (iq2 % ಮುಖವಾಡ->ne[2]) == (iq2_2 % ಮುಖವಾಡ->ne[2]) &&
                                (iq3 % ಮುಖವಾಡ->ne[3]) == (iq3_2 % ಮುಖವಾಡ->ne[3]) && iq1_3 == iq1 &&
                                (iq2 % ಮುಖವಾಡ->ne[2]) == (iq2_3 % ಮುಖವಾಡ->ne[2]) &&
                                (iq3 % ಮುಖವಾಡ->ne[3]) == (iq3_3 % ಮುಖವಾಡ->ne[3]);

        // k indices
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ik3 = iq3 / rk3;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ik2 = iq2 / rk2;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ik3_1 = iq3_1 / rk3;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ik2_1 = iq2_1 / rk2;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ik3_2 = iq3_2 / rk3;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ik2_2 = iq2_2 / rk2;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ik3_3 = iq3_3 / rk3;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ik2_3 = iq2_3 / rk2;

        // v indices
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iv3 = iq3 / rv3;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iv2 = iq2 / rv2;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iv3_1 = iq3_1 / rv3;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iv2_1 = iq2_1 / rv2;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iv3_2 = iq3_2 / rv3;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iv2_2 = iq2_2 / rv2;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iv3_3 = iq3_3 / rv3;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iv2_3 = iq2_3 / rv2;

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * pq = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) q->ದತ್ತಾಂಶ + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3));

        std::array<ದಶಮಾಂಶ *, 4> pq_ಬಫರ್;
        std::array<ದಶಮಾಂಶ *, 4> sinks_ಬಫರ್;
        std::array<ದಶಮಾಂಶ *, 4> dst_ಬಫರ್;

        ಆದರೆ (tcm_ಬಫರ್ != ಶೂನ್ಯಸೂಚಕ && 4 * KV_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ < tcm_ಬಫರ್_ಗಾತ್ರ && ir < (ir1 - 3) && mp_equal_4 &&
            ik3_3 == ik3 && ik2_3 == ik2 && iv3_3 == iv3 && iv2_3 == iv2 && ik3_2 == ik3 && ik2_2 == ik2 &&
            iv3_2 == iv3 && iv2_2 == iv2 && ik3_1 == ik3 && ik2_1 == ik2 && iv3_1 == iv3 && iv2_1 == iv2) {
            ir_ಹೆಜ್ಜೆ = 4;

            pq_ಬಫರ್[0] = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) q->ದತ್ತಾಂಶ + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3));
            pq_ಬಫರ್[1] = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) q->ದತ್ತಾಂಶ + (iq1_1 * nbq1 + iq2_1 * nbq2 + iq3_1 * nbq3));
            pq_ಬಫರ್[2] = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) q->ದತ್ತಾಂಶ + (iq1_2 * nbq1 + iq2_2 * nbq2 + iq3_2 * nbq3));
            pq_ಬಫರ್[3] = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) q->ದತ್ತಾಂಶ + (iq1_3 * nbq1 + iq2_3 * nbq2 + iq3_3 * nbq3));

            sinks_ಬಫರ್[0] = sinks ? ((ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) sinks->ದತ್ತಾಂಶ)) + iq2 : ಶೂನ್ಯಸೂಚಕ;
            sinks_ಬಫರ್[1] = sinks ? ((ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) sinks->ದತ್ತಾಂಶ)) + iq2_1 : ಶೂನ್ಯಸೂಚಕ;
            sinks_ಬಫರ್[2] = sinks ? ((ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) sinks->ದತ್ತಾಂಶ)) + iq2_2 : ಶೂನ್ಯಸೂಚಕ;
            sinks_ಬಫರ್[3] = sinks ? ((ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) sinks->ದತ್ತಾಂಶ)) + iq2_3 : ಶೂನ್ಯಸೂಚಕ;

            dst_ಬಫರ್[0] = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + (iq3 * ne2 * ne1 + iq2 + iq1 * ne1) * nb1);
            dst_ಬಫರ್[1] = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + (iq3_1 * ne2 * ne1 + iq2_1 + iq1_1 * ne1) * nb1);
            dst_ಬಫರ್[2] = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + (iq3_2 * ne2 * ne1 + iq2_2 + iq1_2 * ne1) * nb1);
            dst_ಬಫರ್[3] = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + (iq3_3 * ne2 * ne1 + iq2_3 + iq1_3 * ne1) * nb1);

            flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_mrow<4>(  //
                pq_ಬಫರ್.ದತ್ತಾಂಶ(),                                      //
                (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) k->ದತ್ತಾಂಶ + (ik2 * nbk2 + ik3 * nbk3),    //
                (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) v->ದತ್ತಾಂಶ + (iv2 * nbv2 + iv3 * nbv3),    //
                mp,                                                    //
                sinks_ಬಫರ್.ದತ್ತಾಂಶ(),                                   //
                dst_ಬಫರ್.ದತ್ತಾಂಶ(),                                     //
                scale, logit_softcap, ಇಳಿಜಾರು, nek1, nbk1, nbv1, DV, DK, tcm_ಬಫರ್, tcm_ಬಫರ್_ಗಾತ್ರ);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tcm_ಬಫರ್ != ಶೂನ್ಯಸೂಚಕ && 2 * KV_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ < tcm_ಬಫರ್_ಗಾತ್ರ && ir < (ir1 - 1) && mp_equal_2 &&
                   ik3_1 == ik3 && ik2_1 == ik2 && iv3_1 == iv3 && iv2_1 == iv2) {
            ir_ಹೆಜ್ಜೆ = 2;

            pq_ಬಫರ್[0] = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) q->ದತ್ತಾಂಶ + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3));
            pq_ಬಫರ್[1] = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) q->ದತ್ತಾಂಶ + (iq1_1 * nbq1 + iq2_1 * nbq2 + iq3_1 * nbq3));

            sinks_ಬಫರ್[0] = sinks ? ((ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) sinks->ದತ್ತಾಂಶ)) + iq2 : ಶೂನ್ಯಸೂಚಕ;
            sinks_ಬಫರ್[1] = sinks ? ((ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) sinks->ದತ್ತಾಂಶ)) + iq2_1 : ಶೂನ್ಯಸೂಚಕ;

            dst_ಬಫರ್[0] = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + (iq3 * ne2 * ne1 + iq2 + iq1 * ne1) * nb1);
            dst_ಬಫರ್[1] = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + (iq3_1 * ne2 * ne1 + iq2_1 + iq1_1 * ne1) * nb1);

            flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_mrow<2>(  //
                pq_ಬಫರ್.ದತ್ತಾಂಶ(),                                      //
                (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) k->ದತ್ತಾಂಶ + (ik2 * nbk2 + ik3 * nbk3),    //
                (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) v->ದತ್ತಾಂಶ + (iv2 * nbv2 + iv3 * nbv3),    //
                mp,                                                    //
                sinks_ಬಫರ್.ದತ್ತಾಂಶ(),                                   //
                dst_ಬಫರ್.ದತ್ತಾಂಶ(),                                     //
                scale, logit_softcap, ಇಳಿಜಾರು, nek1, nbk1, nbv1, DV, DK, tcm_ಬಫರ್, tcm_ಬಫರ್_ಗಾತ್ರ);
        } ಇಲ್ಲದಿದ್ದರೆ {
            ir_ಹೆಜ್ಜೆ = 1;
            flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_m1(                             //
                pq,                                                                          //
                (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) k->ದತ್ತಾಂಶ + (ik2 * nbk2 + ik3 * nbk3),                          //
                (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) v->ದತ್ತಾಂಶ + (iv2 * nbv2 + iv3 * nbv3),                          //
                mp,                                                                          //
                sinks ? ((ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) sinks->ದತ್ತಾಂಶ)) + h : ಶೂನ್ಯಸೂಚಕ,                    //
                (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + (iq3 * ne2 * ne1 + iq2 + iq1 * ne1) * nb1),  //
                scale, logit_softcap, ಇಳಿಜಾರು, nek1, nbk1, nbv1, DV, DK);
        }
    }
}

ಶೂನ್ಯ forward_flash_attn_ext_f16_tiled_vlen1024_vf16(ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
                                                    ggml_tensor *               dst,
                                                    ಪೂರ್ಣಾಂಕ                         ir0,
                                                    ಪೂರ್ಣಾಂಕ                         ir1,
                                                    ಶೂನ್ಯ *                      tcm_ಬಫರ್,
                                                    size_t                      tcm_ಬಫರ್_ಗಾತ್ರ) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * q     = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * k     = dst->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * v     = dst->src[2];
    ಸ್ಥಿರಾಂಕ ggml_tensor * ಮುಖವಾಡ  = dst->src[3];
    ಸ್ಥಿರಾಂಕ ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q, ne)
    GGML_TENSOR_LOCALS(size_t, nbq, q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k, ne)
    GGML_TENSOR_LOCALS(size_t, nbk, k, nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v, ne)
    GGML_TENSOR_LOCALS(size_t, nbv, v, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t, nb, dst, nb)

    ಸ್ಥಿರಾಂಕ int64_t DK = nek0;
    ಸ್ಥಿರಾಂಕ int64_t DV = nev0;
    ಸ್ಥಿರಾಂಕ int64_t N  = neq1;

    GGML_ASSERT(flash_attn_ext_supported_shape_vlen1024_vf16(DK, DV));

    GGML_ASSERT(ne0 == DV);
    GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_ಗಾತ್ರ(q->type));
    GGML_ASSERT(nbk0 == ggml_type_ಗಾತ್ರ(k->type));
    GGML_ASSERT(nbv0 == ggml_type_ಗಾತ್ರ(v->type));

    GGML_ASSERT(neq0 == DK);
    GGML_ASSERT(nek0 == DK);
    GGML_ASSERT(nev0 == DV);

    GGML_ASSERT(neq1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == ಗಾತ್ರ(ದಶಮಾಂಶ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(k->type == v->type);
    ಸ್ಥಿರಾಂಕ ggml_type kv_type = k->type;

    // broadcast factors
    ಸ್ಥಿರಾಂಕ int64_t rk2 = neq2 / nek2;
    ಸ್ಥಿರಾಂಕ int64_t rk3 = neq3 / nek3;

    ಸ್ಥಿರಾಂಕ int64_t rv2 = neq2 / nev2;
    ಸ್ಥಿರಾಂಕ int64_t rv3 = neq3 / nev3;

    ದಶಮಾಂಶ * param_list    = (ದಶಮಾಂಶ *) dst->op_params;
    ದಶಮಾಂಶ   scale         = param_list[0];
    ದಶಮಾಂಶ   max_bias      = param_list[1];
    ದಶಮಾಂಶ   logit_softcap = param_list[2];

    ಆದರೆ (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    ಸ್ಥಿರಾಂಕ uint32_t n_ತಲೆ      = neq2;
    ಸ್ಥಿರಾಂಕ uint32_t n_ತಲೆ_log2 = 1u << (uint32_t) floor(log2(n_ತಲೆ));

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m0 = powf(2.0f, -(max_bias) / n_ತಲೆ_log2);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m1 = powf(2.0f, -(max_bias / 2.0f) / n_ತಲೆ_log2);

    ಪೂರ್ಣಾಂಕ ith = params->ith;

    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ Q_TILE_SZ  = ggml_fa_tile_config::Q;
    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ KV_TILE_SZ = ggml_fa_tile_config::KV;

    // Per-thread scratch layout:
    // Q_f32:   Q_TILE_SZ * DK
    // KQ:      Q_TILE_SZ * KV_TILE_SZ
    // mask32:  Q_TILE_SZ * KV_TILE_SZ
    // VKQ32:   Q_TILE_SZ * DV
    // V32:     KV_TILE_SZ * DV
    // K_f32:   DK * KV_TILE_SZ (transposed K tile)
    ದಶಮಾಂಶ *      ತಳ = (ದಶಮಾಂಶ *) params->wdata + ith * (Q_TILE_SZ * DK + 2 * Q_TILE_SZ * KV_TILE_SZ + Q_TILE_SZ * DV +
                                                    KV_TILE_SZ * DV + KV_TILE_SZ * DK + CACHE_ಸಾಲು_ಗಾತ್ರ_F32);
    ಸ್ಥಿರಾಂಕ size_t ತಳ_ಗಾತ್ರ =
        (Q_TILE_SZ * DK + 2 * Q_TILE_SZ * KV_TILE_SZ + Q_TILE_SZ * DV + KV_TILE_SZ * DV + KV_TILE_SZ * DK) *
            ಗಾತ್ರ(ದಶಮಾಂಶ) +
        CACHE_ಸಾಲು_ಗಾತ್ರ_F32;

    ಆದರೆ (ತಳ_ಗಾತ್ರ <= tcm_ಬಫರ್_ಗಾತ್ರ && tcm_ಬಫರ್ != ಶೂನ್ಯಸೂಚಕ) {
        ತಳ = (ದಶಮಾಂಶ *) tcm_ಬಫರ್;
    }

    ದಶಮಾಂಶ   S_M_Buf[Q_TILE_SZ * 2];  // buffer to hold S, M, bias for one tile to reduce register pressure in main loop
    ದಶಮಾಂಶ * S = S_M_Buf;
    ದಶಮಾಂಶ * M = S_M_Buf + Q_TILE_SZ;

    ಪೂರ್ಣಾಂಕ ir = ir0;
    ಆಗ (ir < ir1) {
        // q indices for the start of this tile
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iq3 = ir / (neq2 * neq1);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iq2 = (ir - iq3 * neq2 * neq1) / neq1;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iq1 = (ir - iq3 * neq2 * neq1 - iq2 * neq1);

        // Number of valid rows in this tile:
        // - limited by tile size (Q_TILE_SZ)
        // - limited by chunk boundary (ir1 - ir)
        // - limited by head boundary (neq1 - iq1) to avoid crossing into next head
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tile_ಅಡ್ಡಸಾಲುಗಳು = MIN(Q_TILE_SZ, MIN((ಪೂರ್ಣಾಂಕ) (ir1 - ir), (ಪೂರ್ಣಾಂಕ) (neq1 - iq1)));
        GGML_ASSERT(tile_ಅಡ್ಡಸಾಲುಗಳು > 0);

        ಸ್ಥಿರಾಂಕ uint32_t h = iq2;  // head index
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ    ಇಳಿಜಾರು =
            (max_bias > 0.0f) ? h < n_ತಲೆ_log2 ? powf(m0, h + 1) : powf(m1, 2 * (h - n_ತಲೆ_log2) + 1) : 1.0f;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < Q_TILE_SZ; ++i) {
            S[i] = 0.;
            M[i] = -INFINITY;
        }

        ದಶಮಾಂಶ *    Q_f32  = ತಳ;
        ದಶಮಾಂಶ *    KQ     = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) ತಳ + Q_TILE_SZ * DK * ಗಾತ್ರ(ದಶಮಾಂಶ));
        ದಶಮಾಂಶ *    mask32 = KQ + Q_TILE_SZ * KV_TILE_SZ;
        ದಶಮಾಂಶ *    VKQ32  = mask32 + Q_TILE_SZ * KV_TILE_SZ;
        ದಶಮಾಂಶ *    V32    = VKQ32 + Q_TILE_SZ * DV;
        ದಶಮಾಂಶ *    K_f32  = V32 + KV_TILE_SZ * DV;
        _Float16 * Q_f16  = (_Float16 *) Q_f32;
        _Float16 * V_f16  = (_Float16 *) V32;
        _Float16 * K_f16  = (_Float16 *) K_f32;

        rvv_zero_f32(VKQ32, Q_TILE_SZ * DV);

        // k indices
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ik3 = iq3 / rk3;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ik2 = iq2 / rk2;

        // v indices
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iv3 = iq3 / rv3;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iv2 = iq2 / rv2;

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * pq = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) q->ದತ್ತಾಂಶ + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3));
        ಆದರೆ (kv_type == GGML_TYPE_F16) {
            rvv_pack_f32_as_scaled_f16((uint8_t *) Q_f16, DK * ಗಾತ್ರ(_Float16), (uint8_t *) pq, nbq1, tile_ಅಡ್ಡಸಾಲುಗಳು, DK,
                                       scale);
        } ಇಲ್ಲದಿದ್ದರೆ {
            memcpy2d(Q_f32, DK * ಗಾತ್ರ(ದಶಮಾಂಶ), pq, nbq1, tile_ಅಡ್ಡಸಾಲುಗಳು, DK * ಗಾತ್ರ(ದಶಮಾಂಶ));
        }

        ಗಾಗಿ (int64_t ic = 0; ic < nek1; ic += KV_TILE_SZ) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kv_tile = (ಪೂರ್ಣಾಂಕ) std::min((int64_t) KV_TILE_SZ, nek1 - ic);

            rvv_zero_f32(K_f32, DK * KV_TILE_SZ);
            rvv_zero_f32(V32, KV_TILE_SZ * DV);

            // skip the tile entirely if all the masks are -inf
            ಆದರೆ (ಮುಖವಾಡ) {
                ತಾರ್ಕಿಕ                can_skip = ನಿಜ;
                ಸ್ಥಿರಾಂಕ ggml_fp16_t * mp_ಅಡ್ಡಸಾಲು =
                    (ಸ್ಥಿರಾಂಕ ggml_fp16_t *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) ಮುಖವಾಡ->ದತ್ತಾಂಶ + iq1 * ಮುಖವಾಡ->nb[1] +
                                           (iq2 % ಮುಖವಾಡ->ne[2]) * ಮುಖವಾಡ->nb[2] + (iq3 % ಮುಖವಾಡ->ne[3]) * ಮುಖವಾಡ->nb[3]);
                rvv_pack_scaled_f16_as_f32(mask32, KV_TILE_SZ * ಗಾತ್ರ(ದಶಮಾಂಶ), mp_ಅಡ್ಡಸಾಲು + ic, ಮುಖವಾಡ->nb[1], tile_ಅಡ್ಡಸಾಲುಗಳು,
                                           kv_tile, ಇಳಿಜಾರು);

                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ tq = 0; tq < tile_ಅಡ್ಡಸಾಲುಗಳು; tq++) {
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ tk = 0; tk < kv_tile; tk++) {
                        ಆದರೆ (mask32[tq * KV_TILE_SZ + tk] != -INFINITY) {
                            can_skip = ಸುಳ್ಳು;
                        }
                    }
                    // Pad remaining mask entries with -inf
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ tk = kv_tile; tk < KV_TILE_SZ; tk++) {
                        mask32[tq * KV_TILE_SZ + tk] = -INFINITY;
                    }
                }

                ಆದರೆ (can_skip) {
                    ಮುಂದುವರಿ;
                }
            }

            ಆದರೆ (kv_type == GGML_TYPE_F16) {
                rvv_transposed_s16_mn_ಗಮ್ಯ_nm((int8_t *) K_f16, KV_TILE_SZ * ಗಾತ್ರ(_Float16),
                                            (int8_t *) k->ದತ್ತಾಂಶ + ic * nbk1 + ik2 * nbk2 + ik3 * nbk3, nbk1, kv_tile,
                                            DK);

                ಪೂರ್ಣಾಂಕ tq = 0;
                ಗಾಗಿ (; tq + 3 < tile_ಅಡ್ಡಸಾಲುಗಳು; tq += 4) {
                    rvv_qk_dot_tile_f16_x4(KQ + (tq + 0) * KV_TILE_SZ, KQ + (tq + 1) * KV_TILE_SZ,
                                           KQ + (tq + 2) * KV_TILE_SZ, KQ + (tq + 3) * KV_TILE_SZ,
                                           Q_f16 + (tq + 0) * DK, Q_f16 + (tq + 1) * DK, Q_f16 + (tq + 2) * DK,
                                           Q_f16 + (tq + 3) * DK, K_f16, DK, kv_tile);
                }
                ಗಾಗಿ (; tq < tile_ಅಡ್ಡಸಾಲುಗಳು; ++tq) {
                    rvv_qk_dot_tile_f16_x1(KQ + tq * KV_TILE_SZ, Q_f16 + tq * DK, K_f16, DK, kv_tile);
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ tk = 0; tk < kv_tile; tk++) {
                    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *  k_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) k->ದತ್ತಾಂಶ + (ic + tk) * nbk1 + ik2 * nbk2 + ik3 * nbk3;
                    ದಶಮಾಂಶ *       k_ಲಂಬಸಾಲು  = K_f32 + tk;
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * k_src  = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) k_ದತ್ತಾಂಶ;
                    ಗಾಗಿ (int64_t dk = 0; dk < DK; ++dk) {
                        k_ಲಂಬಸಾಲು[dk * KV_TILE_SZ] = k_src[dk];
                    }
                }

                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ tq = 0; tq < tile_ಅಡ್ಡಸಾಲುಗಳು; ++tq) {
                    rvv_qk_dot_tile(KQ + tq * KV_TILE_SZ, Q_f32 + tq * DK, K_f32, DK, KV_TILE_SZ, scale);
                }
            }

            // Set padded KQ entries to -inf so softmax gives them zero weight
            ಆದರೆ (kv_tile < KV_TILE_SZ) {
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ tq = 0; tq < tile_ಅಡ್ಡಸಾಲುಗಳು; tq++) {
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ tk = kv_tile; tk < KV_TILE_SZ; tk++) {
                        KQ[tq * KV_TILE_SZ + tk] = -INFINITY;
                    }
                }
            }

            ಆದರೆ (logit_softcap != 0.0f) {
                rvv_softcap_tanh_inplace_f32(KQ, KV_TILE_SZ, tile_ಅಡ್ಡಸಾಲುಗಳು, KV_TILE_SZ, logit_softcap);
            }

            ಆದರೆ (ಮುಖವಾಡ) {
                rvv_ಕೂಡಿಸು_inplace_f32(KQ, KV_TILE_SZ, mask32, KV_TILE_SZ, tile_ಅಡ್ಡಸಾಲುಗಳು, KV_TILE_SZ);
            }

            ತಾರ್ಕಿಕ skip[Q_TILE_SZ] = {};

            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ tq = 0; tq < tile_ಅಡ್ಡಸಾಲುಗಳು; tq++) {
                ದಶಮಾಂಶ * kq_ಅಡ್ಡಸಾಲು = KQ + tq * KV_TILE_SZ;

                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ tile_max = rvv_max_f32(kq_ಅಡ್ಡಸಾಲು, KV_TILE_SZ);

                ಆದರೆ (tile_max == -INFINITY) {
                    skip[tq] = ನಿಜ;
                    ಮುಂದುವರಿ;
                }

                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ Mold = M[tq];
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ Mnew = fmaxf(Mold, tile_max);

                ಆದರೆ (Mnew > Mold) {
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ms = expf(Mold - Mnew);
                    rvv_scale_f32(VKQ32 + tq * DV, ms, DV);
                    S[tq] *= ms;
                }
                M[tq] = Mnew;

                S[tq] += rvv_softmax_exp_inplace_f32(kq_ಅಡ್ಡಸಾಲು, KV_TILE_SZ, Mnew);
            }

            // Pack V as contiguous [KV_TILE_SZ][DV].
            ಆದರೆ (kv_type == GGML_TYPE_F16) {
                ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * v_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) v->ದತ್ತಾಂಶ + ic * nbv1 + iv2 * nbv2 + iv3 * nbv3;
                memcpy2d(V_f16, DV * ಗಾತ್ರ(_Float16), v_ದತ್ತಾಂಶ, nbv1, kv_tile, DV * ಗಾತ್ರ(_Float16));

                ಪೂರ್ಣಾಂಕ tq = 0;
                ಗಾಗಿ (; tq + 3 < tile_ಅಡ್ಡಸಾಲುಗಳು; tq += 4) {
                    ಆದರೆ (skip[tq + 0] || skip[tq + 1] || skip[tq + 2] || skip[tq + 3]) {
                        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 4; ++i) {
                            ಆದರೆ (!skip[tq + i]) {
                                rvv_pv_accumulate_f16_x1(VKQ32 + (tq + i) * DV, KQ + (tq + i) * KV_TILE_SZ, V_f16,
                                                         KV_TILE_SZ, DV);
                            }
                        }
                        ಮುಂದುವರಿ;
                    }

                    rvv_pv_accumulate_f16_x4(VKQ32 + (tq + 0) * DV, VKQ32 + (tq + 1) * DV, VKQ32 + (tq + 2) * DV,
                                             VKQ32 + (tq + 3) * DV, KQ + (tq + 0) * KV_TILE_SZ,
                                             KQ + (tq + 1) * KV_TILE_SZ, KQ + (tq + 2) * KV_TILE_SZ,
                                             KQ + (tq + 3) * KV_TILE_SZ, V_f16, KV_TILE_SZ, DV);
                }
                ಗಾಗಿ (; tq < tile_ಅಡ್ಡಸಾಲುಗಳು; ++tq) {
                    ಆದರೆ (!skip[tq]) {
                        rvv_pv_accumulate_f16_x1(VKQ32 + tq * DV, KQ + tq * KV_TILE_SZ, V_f16, KV_TILE_SZ, DV);
                    }
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * v_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) v->ದತ್ತಾಂಶ + ic * nbv1 + iv2 * nbv2 + iv3 * nbv3;
                memcpy2d(V32, DV * ಗಾತ್ರ(ದಶಮಾಂಶ), v_ದತ್ತಾಂಶ, nbv1, kv_tile, DV * ಗಾತ್ರ(ದಶಮಾಂಶ));

                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ tq = 0; tq < tile_ಅಡ್ಡಸಾಲುಗಳು; ++tq) {
                    ಆದರೆ (!skip[tq]) {
                        rvv_pv_accumulate(VKQ32 + tq * DV, KQ + tq * KV_TILE_SZ, V32, KV_TILE_SZ, DV);
                    }
                }
            }
        }

        // sinks (apply only to valid rows in the tile)
        ಆದರೆ (sinks) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ s = ((ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) sinks->ದತ್ತಾಂಶ))[h];

            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ tq = 0; tq < tile_ಅಡ್ಡಸಾಲುಗಳು; tq++) {
                ದಶಮಾಂಶ ms = 1.0f;
                ದಶಮಾಂಶ vs = 1.0f;

                ಆದರೆ (s > M[tq]) {
                    ms = expf(M[tq] - s);
                    rvv_scale_f32(VKQ32 + tq * DV, ms, DV);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    vs = expf(s - M[tq]);
                }

                ದಶಮಾಂಶ S_ತಾತ್ಕಾಲಿಕ = S[tq] * ms + vs;
                S[tq]        = S_ತಾತ್ಕಾಲಿಕ == 0.0f ? 0.0f : 1.0f / S_ತಾತ್ಕಾಲಿಕ;
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ tq = 0; tq < tile_ಅಡ್ಡಸಾಲುಗಳು; tq++) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ S_inv = S[tq] == 0.0f ? 0.0f : 1.0f / S[tq];
                S[tq]             = S_inv;
            }
        }

        ದಶಮಾಂಶ * dst_ptr = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + (iq3 * ne2 * ne1 + iq2 + (iq1) *ne1) * nb1);
        rvv_pack_scaled_f32_as_f32(dst_ptr, nb1 * ne1, VKQ32, DV * ಗಾತ್ರ(ದಶಮಾಂಶ), tile_ಅಡ್ಡಸಾಲುಗಳು, DV, S);

        ir += tile_ಅಡ್ಡಸಾಲುಗಳು;
    }
}

ಶೂನ್ಯ forward_rms_norm_f32(ggml_compute_params * params, ggml_tensor * op) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(src0->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಪೂರ್ಣಾಂಕ nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    ದಶಮಾಂಶ epsilon = *((ದಶಮಾಂಶ *) dst->op_params);

    GGML_ASSERT(epsilon > 0.0f);

    ಸ್ವಯಂಚಾಲಿತ * ಒಳಹರಿವು  = (ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ;
    ಸ್ವಯಂಚಾಲಿತ * output = (ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ;

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ hidden_ಗಾತ್ರ     = ne00;
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ task_ಎಣಿಕೆ      = ne01 * ne02 * ne03;
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ task_ಪ್ರತಿ_thread = (task_ಎಣಿಕೆ + nth - 1) / nth;

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ task_begin = ith * task_ಪ್ರತಿ_thread;
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ task_ಕೊನೆ   = std::min((ith + 1) * task_ಪ್ರತಿ_thread, task_ಎಣಿಕೆ);

    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ task_idx = task_begin; task_idx < task_ಕೊನೆ; task_idx++) {
        int64_t i03 = task_idx / (ne02 * ne01);
        int64_t i02 = (task_idx - i03 * ne02 * ne01) / ne01;
        int64_t i01 = (task_idx - i03 * ne02 * ne01 - i02 * ne01);

        ಸ್ವಯಂಚಾಲಿತ * p_ಒಳಹರಿವು       = (ದಶಮಾಂಶ *) (ಒಳಹರಿವು + i01 * nb01 + i02 * nb02 + i03 * nb03);
        ಸ್ವಯಂಚಾಲಿತ * p_output      = (ದಶಮಾಂಶ *) (output + i01 * nb1 + i02 * nb2 + i03 * nb3);
        ಸ್ವಯಂಚಾಲಿತ * p_ತಾತ್ಕಾಲಿಕ_output = p_output;

        size_t       gvl    = __riscv_vsetvlmax_e32m4();
        vfloat32m4_t ಮೊತ್ತ_sq = __riscv_vfmv_v_f_f32m4(0.f, gvl);
        int64_t      ಉದ್ದಳತೆ = hidden_ಗಾತ್ರ;
        ಆಗ (ಉದ್ದಳತೆ > 0) {
            gvl                   = __riscv_vsetvl_e32m4(ಉದ್ದಳತೆ);
            vfloat32m4_t src_ದತ್ತಾಂಶ = __riscv_vle32_v_f32m4(p_ಒಳಹರಿವು, gvl);
            ಮೊತ್ತ_sq                = __riscv_vfmacc_vv_f32m4(ಮೊತ್ತ_sq, src_ದತ್ತಾಂಶ, src_ದತ್ತಾಂಶ, gvl);
            __riscv_vse32_v_f32m4(p_ತಾತ್ಕಾಲಿಕ_output, src_ದತ್ತಾಂಶ, gvl);

            p_ಒಳಹರಿವು += gvl;
            p_ತಾತ್ಕಾಲಿಕ_output += gvl;
            ಉದ್ದಳತೆ -= gvl;
        }

        gvl                 = __riscv_vsetvlmax_e32m1();
        vfloat32m1_t zero_v = __riscv_vfmv_v_f_f32m1(0.f, gvl);
        vfloat32m1_t ಸರಾಸರಿ_ವರ್ಗಫಲ_v =
            __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(ಮೊತ್ತ_sq, 0), __riscv_vget_v_f32m4_f32m1(ಮೊತ್ತ_sq, 1), gvl);

        ಸರಾಸರಿ_ವರ್ಗಫಲ_v = __riscv_vfadd_vv_f32m1(ಸರಾಸರಿ_ವರ್ಗಫಲ_v, __riscv_vget_v_f32m4_f32m1(ಮೊತ್ತ_sq, 2), gvl);
        ಸರಾಸರಿ_ವರ್ಗಫಲ_v = __riscv_vfadd_vv_f32m1(ಸರಾಸರಿ_ವರ್ಗಫಲ_v, __riscv_vget_v_f32m4_f32m1(ಮೊತ್ತ_sq, 3), gvl);
        ಸರಾಸರಿ_ವರ್ಗಫಲ_v = __riscv_vfredusum_vs_f32m1_f32m1(ಸರಾಸರಿ_ವರ್ಗಫಲ_v, zero_v, gvl);

        ದಶಮಾಂಶ ಸರಾಸರಿ_ವರ್ಗಫಲ = __riscv_vfmv_f_s_f32m1_f32(ಸರಾಸರಿ_ವರ್ಗಫಲ_v);
        ಸರಾಸರಿ_ವರ್ಗಫಲ /= hidden_ಗಾತ್ರ;

        ಸರಾಸರಿ_ವರ್ಗಫಲ = ವರ್ಗಮೂಲ(ಸರಾಸರಿ_ವರ್ಗಫಲ + epsilon);

        ಸರಾಸರಿ_ವರ್ಗಫಲ   = 1.0f / ಸರಾಸರಿ_ವರ್ಗಫಲ;
        ಉದ್ದಳತೆ        = hidden_ಗಾತ್ರ;
        p_ತಾತ್ಕಾಲಿಕ_output = p_output;

        ಆಗ (ಉದ್ದಳತೆ > 0) {
            gvl                   = __riscv_vsetvl_e32m4(ಉದ್ದಳತೆ);
            vfloat32m4_t src_ದತ್ತಾಂಶ = __riscv_vle32_v_f32m4(p_ತಾತ್ಕಾಲಿಕ_output, gvl);
            src_ದತ್ತಾಂಶ              = __riscv_vfmul_vf_f32m4(src_ದತ್ತಾಂಶ, ಸರಾಸರಿ_ವರ್ಗಫಲ, gvl);
            __riscv_vse32_v_f32m4(p_output, src_ದತ್ತಾಂಶ, gvl);
            p_ತಾತ್ಕಾಲಿಕ_output += gvl;
            p_output += gvl;
            ಉದ್ದಳತೆ -= gvl;
        }
    }
}

ಮಾದರಿ <size_t MB_ಅಡ್ಡಸಾಲುಗಳು>
ಶೂನ್ಯ quantize_a_nrow_i8_ref(size_t blk_len, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * a_ptr, size_t ಎಣಿಕೆ_k, uint8_t * quant_a_ptr) {
    int64_t a_blk_stride        = q8_blk_ಗಾತ್ರ(blk_len, ನಿಜ);
    int64_t a_nrow_ಖಂಡ_stride = a_blk_stride * MB_ಅಡ್ಡಸಾಲುಗಳು;
    ಗಾಗಿ (size_t k = 0; k < ಎಣಿಕೆ_k; k += blk_len, quant_a_ptr += a_nrow_ಖಂಡ_stride) {
        ದಶಮಾಂಶ *   scale_a_ptr = reinterpret_cast<ದಶಮಾಂಶ *>(quant_a_ptr);
        int16_t * a_ಮೊತ್ತ_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + ಗಾತ್ರ(ದಶಮಾಂಶ) * MB_ಅಡ್ಡಸಾಲುಗಳು);
        int8_t *  quant_a_blk =
            reinterpret_cast<int8_t *>(quant_a_ptr + ಗಾತ್ರ(ದಶಮಾಂಶ) * MB_ಅಡ್ಡಸಾಲುಗಳು + ಗಾತ್ರ(int16_t) * MB_ಅಡ್ಡಸಾಲುಗಳು);

        ಗಾಗಿ (size_t ಅಡ್ಡಸಾಲು = 0; ಅಡ್ಡಸಾಲು < MB_ಅಡ್ಡಸಾಲುಗಳು; ಅಡ್ಡಸಾಲು++) {
            ದಶಮಾಂಶ max_abs_a = 0.0f;
            ಗಾಗಿ (size_t bk = 0; bk < blk_len; bk++) {
                max_abs_a = std::max(max_abs_a, std::ನಿರಪೇಕ್ಷ(a_ptr[ಅಡ್ಡಸಾಲು * ಎಣಿಕೆ_k + k + bk]));
            }

            ದಶಮಾಂಶ rep_scale_a = ((1 << 7) - 1) / max_abs_a;
            scale_a_ptr[ಅಡ್ಡಸಾಲು]  = 1 / rep_scale_a;

            int16_t a_ಮೊತ್ತ = 0;
            ಗಾಗಿ (size_t bk = 0; bk < blk_len; bk++) {
                ಸ್ಥಿರಾಂಕ int8_t quantized = static_cast<int8_t>(
                    std::clamp(std::nearbyintf(a_ptr[ಅಡ್ಡಸಾಲು * ಎಣಿಕೆ_k + k + bk] * rep_scale_a), -128.0f, 127.0f));
                quant_a_blk[ಅಡ್ಡಸಾಲು * blk_len + bk] = quantized;
                a_ಮೊತ್ತ += quantized;
            }
            a_ಮೊತ್ತ_ptr[ಅಡ್ಡಸಾಲು] = -a_ಮೊತ್ತ;
        }
    }
}

ಮಾದರಿ <size_t MB_ಅಡ್ಡಸಾಲುಗಳು>
ಶೂನ್ಯ quantize_a_nrow_i8_hp_ref(size_t blk_len, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * a_ptr, size_t ಎಣಿಕೆ_k, uint8_t * quant_a_ptr) {
    constexpr size_t k_subblk_len = 32;
    ಸ್ಥಿರಾಂಕ size_t     subblk_ಎಣಿಕೆ = blk_len / k_subblk_len;

    GGML_ASSERT(blk_len == 256);

    ದಶಮಾಂಶ   scale_ತಾತ್ಕಾಲಿಕ[8]       = { 0.0f };
    int64_t a_blk_stride        = q8_hp_blk_ಗಾತ್ರ(blk_len, ನಿಜ, ನಿಜ);
    int64_t a_nrow_ಖಂಡ_stride = a_blk_stride * MB_ಅಡ್ಡಸಾಲುಗಳು;
    int64_t a_subblk_stride     = q8_hp_blk_ಗಾತ್ರ(k_subblk_len, ಸುಳ್ಳು, ಸುಳ್ಳು) * MB_ಅಡ್ಡಸಾಲುಗಳು;

    ಗಾಗಿ (size_t k = 0; k < ಎಣಿಕೆ_k; k += blk_len, quant_a_ptr += a_nrow_ಖಂಡ_stride) {
        _Float16 * a_ಮೊತ್ತ_ptr = reinterpret_cast<_Float16 *>(quant_a_ptr + a_subblk_stride * subblk_ಎಣಿಕೆ);

        ದಶಮಾಂಶ scale_avg = 0.0f;
        ಗಾಗಿ (size_t kk = 0; kk < subblk_ಎಣಿಕೆ; kk++) {
            ದಶಮಾಂಶ max_abs_a = 0.0f;
            ಗಾಗಿ (size_t ಅಡ್ಡಸಾಲು = 0; ಅಡ್ಡಸಾಲು < MB_ಅಡ್ಡಸಾಲುಗಳು; ಅಡ್ಡಸಾಲು++) {
                ಗಾಗಿ (size_t bk = 0; bk < k_subblk_len; bk++) {
                    max_abs_a = std::max(max_abs_a, std::ನಿರಪೇಕ್ಷ(a_ptr[ಅಡ್ಡಸಾಲು * ಎಣಿಕೆ_k + k + bk + kk * k_subblk_len]));
                }
            }
            scale_ತಾತ್ಕಾಲಿಕ[kk] = max_abs_a / ((1 << 7) - 1);
            scale_avg += scale_ತಾತ್ಕಾಲಿಕ[kk];
        }

        scale_avg /= subblk_ಎಣಿಕೆ;
        ದಶಮಾಂಶ scale_ಗುಣಕ = 1.0f / scale_avg;

        _Float16 * scale_avg_ptr =
            reinterpret_cast<_Float16 *>(quant_a_ptr + a_nrow_ಖಂಡ_stride - ಗಾತ್ರ(_Float16) * MB_ಅಡ್ಡಸಾಲುಗಳು);
        scale_avg_ptr[0] = scale_avg;

        ಗಾಗಿ (size_t kk = 0; kk < subblk_ಎಣಿಕೆ; kk++) {
            uint8_t *  a_subblk_ತಳ = quant_a_ptr + kk * a_subblk_stride;
            _Float16 * scale_a_ptr   = reinterpret_cast<_Float16 *>(a_subblk_ತಳ);
            int8_t *   quant_a_blk   = reinterpret_cast<int8_t *>(a_subblk_ತಳ + ಗಾತ್ರ(_Float16) * MB_ಅಡ್ಡಸಾಲುಗಳು);

            scale_a_ptr[0] = static_cast<_Float16>(scale_ತಾತ್ಕಾಲಿಕ[kk] * scale_ಗುಣಕ);

            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ rep_scale_a = 1.0f / scale_ತಾತ್ಕಾಲಿಕ[kk];

            ಗಾಗಿ (size_t ಅಡ್ಡಸಾಲು = 0; ಅಡ್ಡಸಾಲು < MB_ಅಡ್ಡಸಾಲುಗಳು; ಅಡ್ಡಸಾಲು++) {
                int16_t a_ಮೊತ್ತ = 0;
                ಗಾಗಿ (size_t bk = 0; bk < k_subblk_len; bk++) {
                    ಸ್ಥಿರಾಂಕ int8_t quantized = static_cast<int8_t>(
                        std::clamp(std::nearbyintf(a_ptr[ಅಡ್ಡಸಾಲು * ಎಣಿಕೆ_k + k + bk + kk * k_subblk_len] * rep_scale_a),
                                   -128.0f, 127.0f));
                    quant_a_blk[ಅಡ್ಡಸಾಲು * k_subblk_len + bk] = quantized;
                    a_ಮೊತ್ತ += quantized;
                }
                a_ಮೊತ್ತ_ptr[ಅಡ್ಡಸಾಲು * subblk_ಎಣಿಕೆ + kk] = static_cast<_Float16>(-a_ಮೊತ್ತ) * static_cast<_Float16>(8.0f);
            }
        }
    }
}

ಮಾದರಿ <size_t MB_ಅಡ್ಡಸಾಲುಗಳು>
ಶೂನ್ಯ quantize_a_nrow_i8k_ref(size_t blk_len, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * a_ptr, size_t ಎಣಿಕೆ_k, uint8_t * quant_a_ptr) {
    int64_t a_blk_stride        = q8k_blk_ಗಾತ್ರ(256);
    int64_t a_nrow_ಖಂಡ_stride = a_blk_stride * MB_ಅಡ್ಡಸಾಲುಗಳು;
    int64_t a_ಮೊತ್ತ_ಗಾತ್ರ          = 256 / 16;

    ಗಾಗಿ (size_t k = 0; k < ಎಣಿಕೆ_k; k += blk_len, quant_a_ptr += a_nrow_ಖಂಡ_stride) {
        ದಶಮಾಂಶ *   scale_a_ptr = reinterpret_cast<ದಶಮಾಂಶ *>(quant_a_ptr);
        int16_t * a_ಮೊತ್ತ_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + ಗಾತ್ರ(ದಶಮಾಂಶ) * MB_ಅಡ್ಡಸಾಲುಗಳು);
        int8_t *  quant_a_blk =
            reinterpret_cast<int8_t *>(quant_a_ptr + ಗಾತ್ರ(ದಶಮಾಂಶ) * MB_ಅಡ್ಡಸಾಲುಗಳು + ಗಾತ್ರ(int16_t) * a_ಮೊತ್ತ_ಗಾತ್ರ * MB_ಅಡ್ಡಸಾಲುಗಳು);

        ಗಾಗಿ (size_t ಅಡ್ಡಸಾಲು = 0; ಅಡ್ಡಸಾಲು < MB_ಅಡ್ಡಸಾಲುಗಳು; ಅಡ್ಡಸಾಲು++) {
            ದಶಮಾಂಶ max_a     = 0.0f;
            ದಶಮಾಂಶ max_abs_a = 0.0f;
            ಗಾಗಿ (size_t bk = 0; bk < blk_len; bk++) {
                ದಶಮಾಂಶ ax = std::ನಿರಪೇಕ್ಷ(a_ptr[ಅಡ್ಡಸಾಲು * ಎಣಿಕೆ_k + k + bk]);
                ಆದರೆ (ax > max_abs_a) {
                    max_abs_a = ax;
                    max_a     = a_ptr[ಅಡ್ಡಸಾಲು * ಎಣಿಕೆ_k + k + bk];
                }
            }

            ಆದರೆ (!max_abs_a) {
                scale_a_ptr[ಅಡ್ಡಸಾಲು] = 0;
                ಗಾಗಿ (size_t bki = 0; bki < a_ಮೊತ್ತ_ಗಾತ್ರ; bki++) {
                    ಗಾಗಿ (size_t bk = bki * 16; bk < (bki + 1) * 16; bk++) {
                        quant_a_blk[ಅಡ್ಡಸಾಲು * blk_len + bk] = 0;
                    }
                    a_ಮೊತ್ತ_ptr[ಅಡ್ಡಸಾಲು * a_ಮೊತ್ತ_ಗಾತ್ರ + bki] = 0;
                }
                ಮುಂದುವರಿ;
            }

            ದಶಮಾಂಶ rep_scale_a = ((1 << 7) - 1) / max_abs_a;
            scale_a_ptr[ಅಡ್ಡಸಾಲು]  = 1 / rep_scale_a;

            ಗಾಗಿ (size_t bki = 0; bki < a_ಮೊತ್ತ_ಗಾತ್ರ; bki++) {
                int16_t a_ಮೊತ್ತ = 0;
                ಗಾಗಿ (size_t bk = bki * 16; bk < (bki + 1) * 16; bk++) {
                    ಸ್ಥಿರಾಂಕ int8_t quantized = static_cast<int8_t>(
                        std::clamp(std::nearbyintf(a_ptr[ಅಡ್ಡಸಾಲು * ಎಣಿಕೆ_k + k + bk] * rep_scale_a), -128.0f, 127.0f));
                    quant_a_blk[ಅಡ್ಡಸಾಲು * blk_len + bk] = quantized;
                    a_ಮೊತ್ತ += quantized;
                }
                a_ಮೊತ್ತ_ptr[ಅಡ್ಡಸಾಲು * a_ಮೊತ್ತ_ಗಾತ್ರ + bki] = -a_ಮೊತ್ತ;
            }
        }
    }
}

ಶೂನ್ಯ quantize_a_ಅಡ್ಡಸಾಲು_i8(size_t blk_len, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * a_ptr, size_t ಎಣಿಕೆ_k, uint8_t * quant_a_ptr) {
    GGML_ASSERT(blk_len == 32);
    int64_t a_blk_stride = q8_blk_ಗಾತ್ರ(blk_len, ನಿಜ);
    size_t  vlenb        = __riscv_vlenb();

    ಆದರೆ (vlenb == 128) {
        ಗಾಗಿ (size_t k = 0; k < ಎಣಿಕೆ_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            ದಶಮಾಂಶ *   scale_a_ptr = reinterpret_cast<ದಶಮಾಂಶ *>(quant_a_ptr);
            int16_t * a_ಮೊತ್ತ_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + ಗಾತ್ರ(ದಶಮಾಂಶ));
            int8_t *  quant_a_blk = reinterpret_cast<int8_t *>(quant_a_ptr + ಗಾತ್ರ(ದಶಮಾಂಶ) + ಗಾತ್ರ(int16_t));

            size_t       vl      = __riscv_vsetvl_e32m1(blk_len);
            vfloat32m1_t v_a     = __riscv_vle32_v_f32m1(a_ptr + k, vl);
            vfloat32m1_t v_a_abs = __riscv_vfabs_v_f32m1(v_a, vl);

            vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m1_f32m1(v_a_abs, tmp, vl);
            ದಶಮಾಂಶ        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

            ದಶಮಾಂಶ scale_a     = max_abs_a / ((1 << 7) - 1);
            ದಶಮಾಂಶ rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
            scale_a_ptr[0]    = scale_a;

            vfloat32m1_t v_a_scale    = __riscv_vfmul_vf_f32m1(v_a, rep_scale_a, vl);
            vint16mf2_t  v_a_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a_scale, vl);
            vint8mf4_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a_quant, vl);

            vint16m1_t tmp_ಮೊತ್ತ = __riscv_vmv_v_x_i16m1(0, vl);
            vint16m1_t v_a_ಮೊತ್ತ = __riscv_vwredsum_vs_i8mf4_i16m1(v_a_quant_i8, tmp_ಮೊತ್ತ, vl);
            int16_t    a_ಮೊತ್ತ   = __riscv_vmv_x_s_i16m1_i16(v_a_ಮೊತ್ತ);
            a_ಮೊತ್ತ_ptr[0]       = -a_ಮೊತ್ತ;

            __riscv_vse8_v_i8mf4(quant_a_blk, v_a_quant_i8, vl);
        }
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (vlenb == 32) {
        ಗಾಗಿ (size_t k = 0; k < ಎಣಿಕೆ_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            ದಶಮಾಂಶ *   scale_a_ptr = reinterpret_cast<ದಶಮಾಂಶ *>(quant_a_ptr);
            int16_t * a_ಮೊತ್ತ_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + ಗಾತ್ರ(ದಶಮಾಂಶ));
            int8_t *  quant_a_blk = reinterpret_cast<int8_t *>(quant_a_ptr + ಗಾತ್ರ(ದಶಮಾಂಶ) + ಗಾತ್ರ(int16_t));

            size_t       vl      = __riscv_vsetvl_e32m4(blk_len);
            vfloat32m4_t v_a     = __riscv_vle32_v_f32m4(a_ptr + k, vl);
            vfloat32m4_t v_a_abs = __riscv_vfabs_v_f32m4(v_a, vl);

            vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m4_f32m1(v_a_abs, tmp, vl);
            ದಶಮಾಂಶ        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

            ದಶಮಾಂಶ scale_a     = max_abs_a / ((1 << 7) - 1);
            ದಶಮಾಂಶ rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
            scale_a_ptr[0]    = scale_a;

            vfloat32m4_t v_a_scale    = __riscv_vfmul_vf_f32m4(v_a, rep_scale_a, vl);
            vint16m2_t   v_a_quant    = __riscv_vfncvt_x_f_w_i16m2(v_a_scale, vl);
            vint8m1_t    v_a_quant_i8 = __riscv_vncvt_x_x_w_i8m1(v_a_quant, vl);

            vint16m1_t tmp_ಮೊತ್ತ = __riscv_vmv_v_x_i16m1(0, vl);
            vint16m1_t v_a_ಮೊತ್ತ = __riscv_vwredsum_vs_i8m1_i16m1(v_a_quant_i8, tmp_ಮೊತ್ತ, vl);
            int16_t    a_ಮೊತ್ತ   = __riscv_vmv_x_s_i16m1_i16(v_a_ಮೊತ್ತ);
            a_ಮೊತ್ತ_ptr[0]       = -a_ಮೊತ್ತ;

            __riscv_vse8_v_i8m1(quant_a_blk, v_a_quant_i8, vl);
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        quantize_a_nrow_i8_ref<1>(blk_len, a_ptr, ಎಣಿಕೆ_k, quant_a_ptr);
    }
}

ಶೂನ್ಯ quantize_a_4row_i8(size_t blk_len, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * a_ptr, size_t ಎಣಿಕೆ_k, uint8_t * quant_a_ptr) {
    GGML_ASSERT(blk_len == 32);
    int64_t a_blk_stride        = q8_blk_ಗಾತ್ರ(blk_len, ನಿಜ);
    int64_t a_nrow_ಖಂಡ_stride = a_blk_stride * 4;
    size_t  vlenb               = __riscv_vlenb();

    ಆದರೆ (vlenb == 128) {
        ಗಾಗಿ (size_t k = 0; k < ಎಣಿಕೆ_k; k += blk_len, quant_a_ptr += a_nrow_ಖಂಡ_stride) {
            ದಶಮಾಂಶ *   scale_a_ptr = reinterpret_cast<ದಶಮಾಂಶ *>(quant_a_ptr);
            int16_t * a_ಮೊತ್ತ_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + ಗಾತ್ರ(ದಶಮಾಂಶ) * 4);
            int8_t *  quant_a_blk = reinterpret_cast<int8_t *>(quant_a_ptr + ಗಾತ್ರ(ದಶಮಾಂಶ) * 4 + ಗಾತ್ರ(int16_t) * 4);

            ಗಾಗಿ (size_t mi = 0; mi < 4; mi++) {
                size_t       vl      = __riscv_vsetvl_e32m1(blk_len);
                vfloat32m1_t v_a     = __riscv_vle32_v_f32m1(a_ptr + mi * ಎಣಿಕೆ_k + k, vl);
                vfloat32m1_t v_a_abs = __riscv_vfabs_v_f32m1(v_a, vl);

                vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m1_f32m1(v_a_abs, tmp, vl);
                ದಶಮಾಂಶ        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                ದಶಮಾಂಶ scale_a     = max_abs_a / ((1 << 7) - 1);
                ದಶಮಾಂಶ rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
                scale_a_ptr[mi]   = scale_a;

                vfloat32m1_t v_a_scale    = __riscv_vfmul_vf_f32m1(v_a, rep_scale_a, vl);
                vint16mf2_t  v_a_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a_scale, vl);
                vint8mf4_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a_quant, vl);

                vint16m1_t tmp_ಮೊತ್ತ = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a_ಮೊತ್ತ = __riscv_vwredsum_vs_i8mf4_i16m1(v_a_quant_i8, tmp_ಮೊತ್ತ, vl);
                int16_t    a_ಮೊತ್ತ   = __riscv_vmv_x_s_i16m1_i16(v_a_ಮೊತ್ತ);
                a_ಮೊತ್ತ_ptr[mi]      = -a_ಮೊತ್ತ;

                __riscv_vse8_v_i8mf4(quant_a_blk + mi * blk_len, v_a_quant_i8, vl);
            }
        }
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (vlenb == 32) {
        ಗಾಗಿ (size_t k = 0; k < ಎಣಿಕೆ_k; k += blk_len, quant_a_ptr += a_nrow_ಖಂಡ_stride) {
            ದಶಮಾಂಶ *   scale_a_ptr = reinterpret_cast<ದಶಮಾಂಶ *>(quant_a_ptr);
            int16_t * a_ಮೊತ್ತ_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + ಗಾತ್ರ(ದಶಮಾಂಶ) * 4);
            int8_t *  quant_a_blk = reinterpret_cast<int8_t *>(quant_a_ptr + ಗಾತ್ರ(ದಶಮಾಂಶ) * 4 + ಗಾತ್ರ(int16_t) * 4);

            ಗಾಗಿ (size_t mi = 0; mi < 4; mi++) {
                size_t       vl      = __riscv_vsetvl_e32m4(blk_len);
                vfloat32m4_t v_a     = __riscv_vle32_v_f32m4(a_ptr + mi * ಎಣಿಕೆ_k + k, vl);
                vfloat32m4_t v_a_abs = __riscv_vfabs_v_f32m4(v_a, vl);

                vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m4_f32m1(v_a_abs, tmp, vl);
                ದಶಮಾಂಶ        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                ದಶಮಾಂಶ scale_a     = max_abs_a / ((1 << 7) - 1);
                ದಶಮಾಂಶ rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
                scale_a_ptr[mi]   = scale_a;

                vfloat32m4_t v_a_scale    = __riscv_vfmul_vf_f32m4(v_a, rep_scale_a, vl);
                vint16m2_t   v_a_quant    = __riscv_vfncvt_x_f_w_i16m2(v_a_scale, vl);
                vint8m1_t    v_a_quant_i8 = __riscv_vncvt_x_x_w_i8m1(v_a_quant, vl);

                vint16m1_t tmp_ಮೊತ್ತ = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a_ಮೊತ್ತ = __riscv_vwredsum_vs_i8m1_i16m1(v_a_quant_i8, tmp_ಮೊತ್ತ, vl);
                int16_t    a_ಮೊತ್ತ   = __riscv_vmv_x_s_i16m1_i16(v_a_ಮೊತ್ತ);
                a_ಮೊತ್ತ_ptr[mi]      = -a_ಮೊತ್ತ;

                __riscv_vse8_v_i8m1(quant_a_blk + mi * blk_len, v_a_quant_i8, vl);
            }
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        quantize_a_nrow_i8_ref<4>(blk_len, a_ptr, ಎಣಿಕೆ_k, quant_a_ptr);
    }
}

ಶೂನ್ಯ quantize_a_ಅಡ್ಡಸಾಲು_i8_hp(size_t blk_len, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * a_ptr, size_t ಎಣಿಕೆ_k, uint8_t * quant_a_ptr) {
    constexpr size_t k_subblk_len = 32;
    GGML_ASSERT(blk_len == 256);

    constexpr size_t subblk_ಎಣಿಕೆ             = 256 / k_subblk_len;
    int64_t          a_blk_stride             = q8_hp_blk_ಗಾತ್ರ(blk_len, ನಿಜ, ನಿಜ);
    int64_t          a_subblk_stride          = q8_hp_blk_ಗಾತ್ರ(k_subblk_len, ಸುಳ್ಳು, ಸುಳ್ಳು);
    size_t           vlenb                    = __riscv_vlenb();
    ದಶಮಾಂಶ            scale_ತಾತ್ಕಾಲಿಕ[subblk_ಎಣಿಕೆ] = { 0.0f };

    ಆದರೆ (vlenb == 128) {
        ಗಾಗಿ (size_t k = 0; k < ಎಣಿಕೆ_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            _Float16 * a_ಮೊತ್ತ_ptr     = reinterpret_cast<_Float16 *>(quant_a_ptr + a_subblk_stride * subblk_ಎಣಿಕೆ);
            _Float16 * scale_avg_ptr = reinterpret_cast<_Float16 *>(quant_a_ptr + a_blk_stride - ಗಾತ್ರ(_Float16));
            ದಶಮಾಂಶ      scale_avg     = 0.0f;

            ಗಾಗಿ (size_t kk = 0; kk < subblk_ಎಣಿಕೆ; ++kk) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * a_src_ptr = a_ptr + k + kk * k_subblk_len;

                size_t       vl      = __riscv_vsetvl_e32m1(k_subblk_len);
                vfloat32m1_t v_a     = __riscv_vle32_v_f32m1(a_src_ptr, vl);
                vfloat32m1_t v_a_abs = __riscv_vfabs_v_f32m1(v_a, vl);

                vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m1_f32m1(v_a_abs, tmp, vl);
                ದಶಮಾಂಶ        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                scale_ತಾತ್ಕಾಲಿಕ[kk] = max_abs_a / ((1 << 7) - 1);
                scale_avg += scale_ತಾತ್ಕಾಲಿಕ[kk];
            }

            scale_avg /= subblk_ಎಣಿಕೆ;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale_ಗುಣಕ = scale_avg ? 1.0f / scale_avg : 0.0f;
            scale_avg_ptr[0]         = static_cast<_Float16>(scale_avg);

            ಗಾಗಿ (size_t kk = 0; kk < subblk_ಎಣಿಕೆ; ++kk) {
                uint8_t *     a_subblk_ತಳ = quant_a_ptr + kk * a_subblk_stride;
                _Float16 *    scale_a_ptr   = reinterpret_cast<_Float16 *>(a_subblk_ತಳ);
                int8_t *      quant_a_blk   = reinterpret_cast<int8_t *>(a_subblk_ತಳ + ಗಾತ್ರ(_Float16));
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * a_src_ptr     = a_ptr + k + kk * k_subblk_len;

                size_t       vl          = __riscv_vsetvl_e32m1(k_subblk_len);
                vfloat32m1_t v_a         = __riscv_vle32_v_f32m1(a_src_ptr, vl);
                ದಶಮಾಂಶ        rep_scale_a = scale_ತಾತ್ಕಾಲಿಕ[kk] ? 1.0f / scale_ತಾತ್ಕಾಲಿಕ[kk] : 0.0f;
                scale_a_ptr[0]           = static_cast<_Float16>(scale_ತಾತ್ಕಾಲಿಕ[kk] * scale_ಗುಣಕ);

                vfloat32m1_t v_a_scale    = __riscv_vfmul_vf_f32m1(v_a, rep_scale_a, vl);
                vint16mf2_t  v_a_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a_scale, vl);
                vint8mf4_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a_quant, vl);

                vint16m1_t tmp_ಮೊತ್ತ = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a_ಮೊತ್ತ = __riscv_vwredsum_vs_i8mf4_i16m1(v_a_quant_i8, tmp_ಮೊತ್ತ, vl);
                int16_t    a_ಮೊತ್ತ   = __riscv_vmv_x_s_i16m1_i16(v_a_ಮೊತ್ತ);
                a_ಮೊತ್ತ_ptr[kk]      = static_cast<_Float16>(-a_ಮೊತ್ತ) * static_cast<_Float16>(8.0f);

                __riscv_vse8_v_i8mf4(quant_a_blk, v_a_quant_i8, vl);
            }
        }
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (vlenb == 32) {
        ಗಾಗಿ (size_t k = 0; k < ಎಣಿಕೆ_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            _Float16 * a_ಮೊತ್ತ_ptr     = reinterpret_cast<_Float16 *>(quant_a_ptr + a_subblk_stride * subblk_ಎಣಿಕೆ);
            _Float16 * scale_avg_ptr = reinterpret_cast<_Float16 *>(quant_a_ptr + a_blk_stride - ಗಾತ್ರ(_Float16));
            ದಶಮಾಂಶ      scale_avg     = 0.0f;

            ಗಾಗಿ (size_t kk = 0; kk < subblk_ಎಣಿಕೆ; ++kk) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * a_src_ptr = a_ptr + k + kk * k_subblk_len;

                size_t       vl      = __riscv_vsetvl_e32m4(k_subblk_len);
                vfloat32m4_t v_a     = __riscv_vle32_v_f32m4(a_src_ptr, vl);
                vfloat32m4_t v_a_abs = __riscv_vfabs_v_f32m4(v_a, vl);

                vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m4_f32m1(v_a_abs, tmp, vl);
                ದಶಮಾಂಶ        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                scale_ತಾತ್ಕಾಲಿಕ[kk] = max_abs_a / ((1 << 7) - 1);
                scale_avg += scale_ತಾತ್ಕಾಲಿಕ[kk];
            }

            scale_avg /= subblk_ಎಣಿಕೆ;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale_ಗುಣಕ = scale_avg ? 1.0f / scale_avg : 0.0f;
            scale_avg_ptr[0]         = static_cast<_Float16>(scale_avg);

            ಗಾಗಿ (size_t kk = 0; kk < subblk_ಎಣಿಕೆ; ++kk) {
                uint8_t *     a_subblk_ತಳ = quant_a_ptr + kk * a_subblk_stride;
                _Float16 *    scale_a_ptr   = reinterpret_cast<_Float16 *>(a_subblk_ತಳ);
                int8_t *      quant_a_blk   = reinterpret_cast<int8_t *>(a_subblk_ತಳ + ಗಾತ್ರ(_Float16));
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * a_src_ptr     = a_ptr + k + kk * k_subblk_len;

                size_t       vl          = __riscv_vsetvl_e32m4(k_subblk_len);
                vfloat32m4_t v_a         = __riscv_vle32_v_f32m4(a_src_ptr, vl);
                ದಶಮಾಂಶ        rep_scale_a = scale_ತಾತ್ಕಾಲಿಕ[kk] ? 1.0f / scale_ತಾತ್ಕಾಲಿಕ[kk] : 0.0f;
                scale_a_ptr[0]           = static_cast<_Float16>(scale_ತಾತ್ಕಾಲಿಕ[kk] * scale_ಗುಣಕ);

                vfloat32m4_t v_a_scale    = __riscv_vfmul_vf_f32m4(v_a, rep_scale_a, vl);
                vint16m2_t   v_a_quant    = __riscv_vfncvt_x_f_w_i16m2(v_a_scale, vl);
                vint8m1_t    v_a_quant_i8 = __riscv_vncvt_x_x_w_i8m1(v_a_quant, vl);

                vint16m1_t tmp_ಮೊತ್ತ = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a_ಮೊತ್ತ = __riscv_vwredsum_vs_i8m1_i16m1(v_a_quant_i8, tmp_ಮೊತ್ತ, vl);
                int16_t    a_ಮೊತ್ತ   = __riscv_vmv_x_s_i16m1_i16(v_a_ಮೊತ್ತ);
                a_ಮೊತ್ತ_ptr[kk]      = static_cast<_Float16>(-a_ಮೊತ್ತ) * static_cast<_Float16>(8.0f);

                __riscv_vse8_v_i8m1(quant_a_blk, v_a_quant_i8, vl);
            }
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        quantize_a_nrow_i8_hp_ref<1>(blk_len, a_ptr, ಎಣಿಕೆ_k, quant_a_ptr);
    }
}

ಶೂನ್ಯ quantize_a_4row_i8_hp(size_t blk_len, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * a_ptr, size_t ಎಣಿಕೆ_k, uint8_t * quant_a_ptr) {
    constexpr size_t k_subblk_len = 32;
    GGML_ASSERT(blk_len == 256);

    constexpr size_t subblk_ಎಣಿಕೆ             = 256 / k_subblk_len;
    int64_t          a_blk_stride             = q8_hp_blk_ಗಾತ್ರ(blk_len, ನಿಜ, ನಿಜ);
    int64_t          a_nrow_ಖಂಡ_stride      = a_blk_stride * 4;
    int64_t          a_subblk_stride          = q8_hp_blk_ಗಾತ್ರ(k_subblk_len, ಸುಳ್ಳು, ಸುಳ್ಳು) * 4;
    size_t           vlenb                    = __riscv_vlenb();
    ದಶಮಾಂಶ            scale_ತಾತ್ಕಾಲಿಕ[subblk_ಎಣಿಕೆ] = { 0.0f };

    ಆದರೆ (vlenb == 128) {
        ಗಾಗಿ (size_t k = 0; k < ಎಣಿಕೆ_k; k += blk_len, quant_a_ptr += a_nrow_ಖಂಡ_stride) {
            _Float16 * a_ಮೊತ್ತ_ptr = reinterpret_cast<_Float16 *>(quant_a_ptr + a_subblk_stride * subblk_ಎಣಿಕೆ);
            _Float16 * scale_avg_ptr =
                reinterpret_cast<_Float16 *>(quant_a_ptr + a_nrow_ಖಂಡ_stride - ಗಾತ್ರ(_Float16) * 4);
            ದಶಮಾಂಶ scale_avg = 0.0f;

            ಗಾಗಿ (size_t kk = 0; kk < subblk_ಎಣಿಕೆ; ++kk) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * a_src_ptr0 = a_ptr + 0 * ಎಣಿಕೆ_k + k + kk * k_subblk_len;
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * a_src_ptr1 = a_ptr + 1 * ಎಣಿಕೆ_k + k + kk * k_subblk_len;
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * a_src_ptr2 = a_ptr + 2 * ಎಣಿಕೆ_k + k + kk * k_subblk_len;
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * a_src_ptr3 = a_ptr + 3 * ಎಣಿಕೆ_k + k + kk * k_subblk_len;

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
                ದಶಮಾಂಶ        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                scale_ತಾತ್ಕಾಲಿಕ[kk] = max_abs_a / ((1 << 7) - 1);
                scale_avg += scale_ತಾತ್ಕಾಲಿಕ[kk];
            }

            scale_avg /= subblk_ಎಣಿಕೆ;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale_ಗುಣಕ = scale_avg ? 1.0f / scale_avg : 0.0f;
            scale_avg_ptr[0]         = static_cast<_Float16>(scale_avg);

            ಗಾಗಿ (size_t kk = 0; kk < subblk_ಎಣಿಕೆ; ++kk) {
                uint8_t *     a_subblk_ತಳ = quant_a_ptr + kk * a_subblk_stride;
                _Float16 *    scale_a_ptr   = reinterpret_cast<_Float16 *>(a_subblk_ತಳ);
                int8_t *      quant_a_blk   = reinterpret_cast<int8_t *>(a_subblk_ತಳ + ಗಾತ್ರ(_Float16) * 4);
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * a_src_ptr0    = a_ptr + 0 * ಎಣಿಕೆ_k + k + kk * k_subblk_len;
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * a_src_ptr1    = a_ptr + 1 * ಎಣಿಕೆ_k + k + kk * k_subblk_len;
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * a_src_ptr2    = a_ptr + 2 * ಎಣಿಕೆ_k + k + kk * k_subblk_len;
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * a_src_ptr3    = a_ptr + 3 * ಎಣಿಕೆ_k + k + kk * k_subblk_len;

                size_t       vl   = __riscv_vsetvl_e32m1(k_subblk_len);
                vfloat32m1_t v_a0 = __riscv_vle32_v_f32m1(a_src_ptr0, vl);
                vfloat32m1_t v_a1 = __riscv_vle32_v_f32m1(a_src_ptr1, vl);
                vfloat32m1_t v_a2 = __riscv_vle32_v_f32m1(a_src_ptr2, vl);
                vfloat32m1_t v_a3 = __riscv_vle32_v_f32m1(a_src_ptr3, vl);

                ದಶಮಾಂಶ rep_scale_a = scale_ತಾತ್ಕಾಲಿಕ[kk] ? 1.0f / scale_ತಾತ್ಕಾಲಿಕ[kk] : 0.0f;
                scale_a_ptr[0]    = static_cast<_Float16>(scale_ತಾತ್ಕಾಲಿಕ[kk] * scale_ಗುಣಕ);

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
                vint16m1_t v_a0_ಮೊತ್ತ = __riscv_vwredsum_vs_i8mf4_i16m1(v_a0_quant_i8, tmp_sum0, vl);
                vint16m1_t v_a1_ಮೊತ್ತ = __riscv_vwredsum_vs_i8mf4_i16m1(v_a1_quant_i8, tmp_sum1, vl);
                vint16m1_t v_a2_ಮೊತ್ತ = __riscv_vwredsum_vs_i8mf4_i16m1(v_a2_quant_i8, tmp_sum2, vl);
                vint16m1_t v_a3_ಮೊತ್ತ = __riscv_vwredsum_vs_i8mf4_i16m1(v_a3_quant_i8, tmp_sum3, vl);

                a_ಮೊತ್ತ_ptr[0 * subblk_ಎಣಿಕೆ + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a0_ಮೊತ್ತ)) * static_cast<_Float16>(8.0f);
                a_ಮೊತ್ತ_ptr[1 * subblk_ಎಣಿಕೆ + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a1_ಮೊತ್ತ)) * static_cast<_Float16>(8.0f);
                a_ಮೊತ್ತ_ptr[2 * subblk_ಎಣಿಕೆ + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a2_ಮೊತ್ತ)) * static_cast<_Float16>(8.0f);
                a_ಮೊತ್ತ_ptr[3 * subblk_ಎಣಿಕೆ + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a3_ಮೊತ್ತ)) * static_cast<_Float16>(8.0f);

                __riscv_vse8_v_i8mf4(quant_a_blk + 0 * k_subblk_len, v_a0_quant_i8, vl);
                __riscv_vse8_v_i8mf4(quant_a_blk + 1 * k_subblk_len, v_a1_quant_i8, vl);
                __riscv_vse8_v_i8mf4(quant_a_blk + 2 * k_subblk_len, v_a2_quant_i8, vl);
                __riscv_vse8_v_i8mf4(quant_a_blk + 3 * k_subblk_len, v_a3_quant_i8, vl);
            }
        }
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (vlenb == 32) {
        ಗಾಗಿ (size_t k = 0; k < ಎಣಿಕೆ_k; k += blk_len, quant_a_ptr += a_nrow_ಖಂಡ_stride) {
            _Float16 * a_ಮೊತ್ತ_ptr = reinterpret_cast<_Float16 *>(quant_a_ptr + a_subblk_stride * subblk_ಎಣಿಕೆ);
            _Float16 * scale_avg_ptr =
                reinterpret_cast<_Float16 *>(quant_a_ptr + a_nrow_ಖಂಡ_stride - ಗಾತ್ರ(_Float16) * 4);
            ದಶಮಾಂಶ scale_avg = 0.0f;

            ಗಾಗಿ (size_t kk = 0; kk < subblk_ಎಣಿಕೆ; ++kk) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * a_src_ptr0 = a_ptr + 0 * ಎಣಿಕೆ_k + k + kk * k_subblk_len;
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * a_src_ptr1 = a_ptr + 1 * ಎಣಿಕೆ_k + k + kk * k_subblk_len;
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * a_src_ptr2 = a_ptr + 2 * ಎಣಿಕೆ_k + k + kk * k_subblk_len;
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * a_src_ptr3 = a_ptr + 3 * ಎಣಿಕೆ_k + k + kk * k_subblk_len;

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
                ದಶಮಾಂಶ        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                scale_ತಾತ್ಕಾಲಿಕ[kk] = max_abs_a / ((1 << 7) - 1);
                scale_avg += scale_ತಾತ್ಕಾಲಿಕ[kk];
            }

            scale_avg /= subblk_ಎಣಿಕೆ;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale_ಗುಣಕ = scale_avg ? 1.0f / scale_avg : 0.0f;
            scale_avg_ptr[0]         = static_cast<_Float16>(scale_avg);

            ಗಾಗಿ (size_t kk = 0; kk < subblk_ಎಣಿಕೆ; ++kk) {
                uint8_t *     a_subblk_ತಳ = quant_a_ptr + kk * a_subblk_stride;
                _Float16 *    scale_a_ptr   = reinterpret_cast<_Float16 *>(a_subblk_ತಳ);
                int8_t *      quant_a_blk   = reinterpret_cast<int8_t *>(a_subblk_ತಳ + ಗಾತ್ರ(_Float16) * 4);
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * a_src_ptr0    = a_ptr + 0 * ಎಣಿಕೆ_k + k + kk * k_subblk_len;
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * a_src_ptr1    = a_ptr + 1 * ಎಣಿಕೆ_k + k + kk * k_subblk_len;
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * a_src_ptr2    = a_ptr + 2 * ಎಣಿಕೆ_k + k + kk * k_subblk_len;
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * a_src_ptr3    = a_ptr + 3 * ಎಣಿಕೆ_k + k + kk * k_subblk_len;

                size_t       vl   = __riscv_vsetvl_e32m4(k_subblk_len);
                vfloat32m4_t v_a0 = __riscv_vle32_v_f32m4(a_src_ptr0, vl);
                vfloat32m4_t v_a1 = __riscv_vle32_v_f32m4(a_src_ptr1, vl);
                vfloat32m4_t v_a2 = __riscv_vle32_v_f32m4(a_src_ptr2, vl);
                vfloat32m4_t v_a3 = __riscv_vle32_v_f32m4(a_src_ptr3, vl);

                ದಶಮಾಂಶ rep_scale_a = scale_ತಾತ್ಕಾಲಿಕ[kk] ? 1.0f / scale_ತಾತ್ಕಾಲಿಕ[kk] : 0.0f;
                scale_a_ptr[0]    = static_cast<_Float16>(scale_ತಾತ್ಕಾಲಿಕ[kk] * scale_ಗುಣಕ);

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
                vint16m1_t v_a0_ಮೊತ್ತ = __riscv_vwredsum_vs_i8m1_i16m1(v_a0_quant_i8, tmp_sum0, vl);
                vint16m1_t v_a1_ಮೊತ್ತ = __riscv_vwredsum_vs_i8m1_i16m1(v_a1_quant_i8, tmp_sum1, vl);
                vint16m1_t v_a2_ಮೊತ್ತ = __riscv_vwredsum_vs_i8m1_i16m1(v_a2_quant_i8, tmp_sum2, vl);
                vint16m1_t v_a3_ಮೊತ್ತ = __riscv_vwredsum_vs_i8m1_i16m1(v_a3_quant_i8, tmp_sum3, vl);

                a_ಮೊತ್ತ_ptr[0 * subblk_ಎಣಿಕೆ + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a0_ಮೊತ್ತ)) * static_cast<_Float16>(8.0f);
                a_ಮೊತ್ತ_ptr[1 * subblk_ಎಣಿಕೆ + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a1_ಮೊತ್ತ)) * static_cast<_Float16>(8.0f);
                a_ಮೊತ್ತ_ptr[2 * subblk_ಎಣಿಕೆ + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a2_ಮೊತ್ತ)) * static_cast<_Float16>(8.0f);
                a_ಮೊತ್ತ_ptr[3 * subblk_ಎಣಿಕೆ + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a3_ಮೊತ್ತ)) * static_cast<_Float16>(8.0f);

                __riscv_vse8_v_i8m1(quant_a_blk + 0 * k_subblk_len, v_a0_quant_i8, vl);
                __riscv_vse8_v_i8m1(quant_a_blk + 1 * k_subblk_len, v_a1_quant_i8, vl);
                __riscv_vse8_v_i8m1(quant_a_blk + 2 * k_subblk_len, v_a2_quant_i8, vl);
                __riscv_vse8_v_i8m1(quant_a_blk + 3 * k_subblk_len, v_a3_quant_i8, vl);
            }
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        quantize_a_nrow_i8_hp_ref<4>(blk_len, a_ptr, ಎಣಿಕೆ_k, quant_a_ptr);
    }
}

ಶೂನ್ಯ quantize_a_ಅಡ್ಡಸಾಲು_i8k(size_t blk_len, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * a_ptr, size_t ಎಣಿಕೆ_k, uint8_t * quant_a_ptr) {
    GGML_ASSERT(blk_len == 256);
    constexpr int64_t a_blk_stride = q8k_blk_ಗಾತ್ರ(256);
    constexpr int64_t a_ಮೊತ್ತ_ಗಾತ್ರ   = 256 / 16;
    size_t            vlenb        = __riscv_vlenb();

    ಆದರೆ (vlenb == 128) {
        // vlen = 1024 bits, can process 32 float32 elements with m1
        ಗಾಗಿ (size_t k = 0; k < ಎಣಿಕೆ_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            ದಶಮಾಂಶ *   scale_a_ptr = reinterpret_cast<ದಶಮಾಂಶ *>(quant_a_ptr);
            int16_t * a_ಮೊತ್ತ_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + ಗಾತ್ರ(ದಶಮಾಂಶ));
            int8_t *  quant_a_blk =
                reinterpret_cast<int8_t *>(quant_a_ptr + ಗಾತ್ರ(ದಶಮಾಂಶ) + ಗಾತ್ರ(int16_t) * a_ಮೊತ್ತ_ಗಾತ್ರ);

            // Find max absolute value across all 256 elements
            size_t       vl        = __riscv_vsetvl_e32m1(16);
            vfloat32m1_t v_max_abs = __riscv_vfmv_v_f_f32m1(0.0f, vl);

            ಗಾಗಿ (size_t bki = 0; bki < a_ಮೊತ್ತ_ಗಾತ್ರ; bki++) {
                vfloat32m1_t v_a     = __riscv_vle32_v_f32m1(a_ptr + k + bki * 16, vl);
                vfloat32m1_t v_a_abs = __riscv_vfabs_v_f32m1(v_a, vl);
                v_max_abs            = __riscv_vfmax_vv_f32m1(v_a_abs, v_max_abs, vl);
            }
            vfloat32m1_t tmp         = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t v_local_max = __riscv_vfredmax_vs_f32m1_f32m1(v_max_abs, tmp, vl);
            ದಶಮಾಂಶ        max_abs_a   = __riscv_vfmv_f_s_f32m1_f32(v_local_max);

            ದಶಮಾಂಶ scale_a     = max_abs_a / ((1 << 7) - 1);
            ದಶಮಾಂಶ rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
            scale_a_ptr[0]    = scale_a;

            // Quantize and compute sums for each 16-element group
            ಗಾಗಿ (size_t bki = 0; bki < a_ಮೊತ್ತ_ಗಾತ್ರ; bki++) {
                vfloat32m1_t v_a          = __riscv_vle32_v_f32m1(a_ptr + k + bki * 16, vl);
                vfloat32m1_t v_a_scale    = __riscv_vfmul_vf_f32m1(v_a, rep_scale_a, vl);
                vint16mf2_t  v_a_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a_scale, vl);
                vint8mf4_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a_quant, vl);

                vint16m1_t tmp_ಮೊತ್ತ = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a_ಮೊತ್ತ = __riscv_vwredsum_vs_i8mf4_i16m1(v_a_quant_i8, tmp_ಮೊತ್ತ, vl);
                int16_t    a_ಮೊತ್ತ   = __riscv_vmv_x_s_i16m1_i16(v_a_ಮೊತ್ತ);
                a_ಮೊತ್ತ_ptr[bki]     = -a_ಮೊತ್ತ;

                __riscv_vse8_v_i8mf4(quant_a_blk + bki * 16, v_a_quant_i8, vl);
            }
        }
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (vlenb == 32) {
        // vlen = 256 bits, can process 8 float32 elements with m1
        ಗಾಗಿ (size_t k = 0; k < ಎಣಿಕೆ_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            ದಶಮಾಂಶ *   scale_a_ptr = reinterpret_cast<ದಶಮಾಂಶ *>(quant_a_ptr);
            int16_t * a_ಮೊತ್ತ_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + ಗಾತ್ರ(ದಶಮಾಂಶ));
            int8_t *  quant_a_blk =
                reinterpret_cast<int8_t *>(quant_a_ptr + ಗಾತ್ರ(ದಶಮಾಂಶ) + ಗಾತ್ರ(int16_t) * a_ಮೊತ್ತ_ಗಾತ್ರ);

            // Find max absolute value across all 256 elements
            size_t       vl        = __riscv_vsetvl_e32m2(16);
            vfloat32m2_t v_max_abs = __riscv_vfmv_v_f_f32m2(0.0f, vl);

            ಗಾಗಿ (size_t bki = 0; bki < a_ಮೊತ್ತ_ಗಾತ್ರ; bki++) {
                vfloat32m2_t v_a     = __riscv_vle32_v_f32m2(a_ptr + k + bki * 16, vl);
                vfloat32m2_t v_a_abs = __riscv_vfabs_v_f32m2(v_a, vl);
                v_max_abs            = __riscv_vfmax_vv_f32m2(v_a_abs, v_max_abs, vl);
            }
            vfloat32m1_t tmp         = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t v_local_max = __riscv_vfredmax_vs_f32m2_f32m1(v_max_abs, tmp, vl);
            ದಶಮಾಂಶ        max_abs_a   = __riscv_vfmv_f_s_f32m1_f32(v_local_max);

            ದಶಮಾಂಶ scale_a     = max_abs_a / ((1 << 7) - 1);
            ದಶಮಾಂಶ rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
            scale_a_ptr[0]    = scale_a;

            // Quantize and compute sums for each 16-element group
            ಗಾಗಿ (size_t bki = 0; bki < a_ಮೊತ್ತ_ಗಾತ್ರ; bki++) {
                vfloat32m2_t v_a          = __riscv_vle32_v_f32m2(a_ptr + k + bki * 16, vl);
                vfloat32m2_t v_a_scale    = __riscv_vfmul_vf_f32m2(v_a, rep_scale_a, vl);
                vint16m1_t   v_a_quant    = __riscv_vfncvt_x_f_w_i16m1(v_a_scale, vl);
                vint8mf2_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf2(v_a_quant, vl);

                vint16m1_t tmp_ಮೊತ್ತ = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a_ಮೊತ್ತ = __riscv_vwredsum_vs_i8mf2_i16m1(v_a_quant_i8, tmp_ಮೊತ್ತ, vl);
                int16_t    a_ಮೊತ್ತ   = __riscv_vmv_x_s_i16m1_i16(v_a_ಮೊತ್ತ);
                a_ಮೊತ್ತ_ptr[bki]     = -a_ಮೊತ್ತ;

                __riscv_vse8_v_i8mf2(quant_a_blk + bki * 16, v_a_quant_i8, vl);
            }
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        quantize_a_nrow_i8k_ref<1>(blk_len, a_ptr, ಎಣಿಕೆ_k, quant_a_ptr);
    }
}

ಶೂನ್ಯ quantize_a_4row_i8k(size_t blk_len, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * a_ptr, size_t ಎಣಿಕೆ_k, uint8_t * quant_a_ptr) {
    GGML_ASSERT(blk_len == 256);
    constexpr int64_t a_blk_stride        = q8k_blk_ಗಾತ್ರ(256);
    constexpr int64_t a_nrow_ಖಂಡ_stride = a_blk_stride * 4;
    constexpr int64_t a_ಮೊತ್ತ_ಗಾತ್ರ          = 256 / 16;
    size_t            vlenb               = __riscv_vlenb();

    ಆದರೆ (vlenb == 128) {
        // vlen = 1024 bits
        ಗಾಗಿ (size_t k = 0; k < ಎಣಿಕೆ_k; k += blk_len, quant_a_ptr += a_nrow_ಖಂಡ_stride) {
            ದಶಮಾಂಶ *   scale_a_ptr = reinterpret_cast<ದಶಮಾಂಶ *>(quant_a_ptr);
            int16_t * a_ಮೊತ್ತ_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + ಗಾತ್ರ(ದಶಮಾಂಶ) * 4);
            int8_t *  quant_a_blk =
                reinterpret_cast<int8_t *>(quant_a_ptr + ಗಾತ್ರ(ದಶಮಾಂಶ) * 4 + ಗಾತ್ರ(int16_t) * a_ಮೊತ್ತ_ಗಾತ್ರ * 4);

            ಗಾಗಿ (size_t mi = 0; mi < 4; mi++) {
                // Find max absolute value across all 256 elements for this row
                size_t       vl        = __riscv_vsetvl_e32m1(16);
                vfloat32m1_t v_max_abs = __riscv_vfmv_v_f_f32m1(0.0f, vl);

                ಗಾಗಿ (size_t bki = 0; bki < a_ಮೊತ್ತ_ಗಾತ್ರ; bki++) {
                    vfloat32m1_t v_a     = __riscv_vle32_v_f32m1(a_ptr + mi * ಎಣಿಕೆ_k + k + bki * 16, vl);
                    vfloat32m1_t v_a_abs = __riscv_vfabs_v_f32m1(v_a, vl);
                    v_max_abs            = __riscv_vfmax_vv_f32m1(v_a_abs, v_max_abs, vl);
                }
                vfloat32m1_t tmp         = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_local_max = __riscv_vfredmax_vs_f32m1_f32m1(v_max_abs, tmp, vl);
                ದಶಮಾಂಶ        max_abs_a   = __riscv_vfmv_f_s_f32m1_f32(v_local_max);

                ದಶಮಾಂಶ scale_a     = max_abs_a / ((1 << 7) - 1);
                ದಶಮಾಂಶ rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
                scale_a_ptr[mi]   = scale_a;

                // Quantize and compute sums for each 16-element group
                ಗಾಗಿ (size_t bki = 0; bki < a_ಮೊತ್ತ_ಗಾತ್ರ; bki++) {
                    vfloat32m1_t v_a          = __riscv_vle32_v_f32m1(a_ptr + mi * ಎಣಿಕೆ_k + k + bki * 16, vl);
                    vfloat32m1_t v_a_scale    = __riscv_vfmul_vf_f32m1(v_a, rep_scale_a, vl);
                    vint16mf2_t  v_a_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a_scale, vl);
                    vint8mf4_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a_quant, vl);

                    vint16m1_t tmp_ಮೊತ್ತ               = __riscv_vmv_v_x_i16m1(0, vl);
                    vint16m1_t v_a_ಮೊತ್ತ               = __riscv_vwredsum_vs_i8mf4_i16m1(v_a_quant_i8, tmp_ಮೊತ್ತ, vl);
                    int16_t    a_ಮೊತ್ತ                 = __riscv_vmv_x_s_i16m1_i16(v_a_ಮೊತ್ತ);
                    a_ಮೊತ್ತ_ptr[mi * a_ಮೊತ್ತ_ಗಾತ್ರ + bki] = -a_ಮೊತ್ತ;

                    __riscv_vse8_v_i8mf4(quant_a_blk + mi * blk_len + bki * 16, v_a_quant_i8, vl);
                }
            }
        }
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (vlenb == 32) {
        // vlen = 256 bits
        ಗಾಗಿ (size_t k = 0; k < ಎಣಿಕೆ_k; k += blk_len, quant_a_ptr += a_nrow_ಖಂಡ_stride) {
            ದಶಮಾಂಶ *   scale_a_ptr = reinterpret_cast<ದಶಮಾಂಶ *>(quant_a_ptr);
            int16_t * a_ಮೊತ್ತ_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + ಗಾತ್ರ(ದಶಮಾಂಶ) * 4);
            int8_t *  quant_a_blk =
                reinterpret_cast<int8_t *>(quant_a_ptr + ಗಾತ್ರ(ದಶಮಾಂಶ) * 4 + ಗಾತ್ರ(int16_t) * a_ಮೊತ್ತ_ಗಾತ್ರ * 4);

            ಗಾಗಿ (size_t mi = 0; mi < 4; mi++) {
                // Find max absolute value across all 256 elements for this row
                size_t       vl        = __riscv_vsetvl_e32m2(16);
                vfloat32m2_t v_max_abs = __riscv_vfmv_v_f_f32m2(0.0f, vl);

                ಗಾಗಿ (size_t bki = 0; bki < a_ಮೊತ್ತ_ಗಾತ್ರ; bki++) {
                    vfloat32m2_t v_a     = __riscv_vle32_v_f32m2(a_ptr + mi * ಎಣಿಕೆ_k + k + bki * 16, vl);
                    vfloat32m2_t v_a_abs = __riscv_vfabs_v_f32m2(v_a, vl);
                    v_max_abs            = __riscv_vfmax_vv_f32m2(v_a_abs, v_max_abs, vl);
                }
                vfloat32m1_t tmp         = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_local_max = __riscv_vfredmax_vs_f32m2_f32m1(v_max_abs, tmp, vl);
                ದಶಮಾಂಶ        max_abs_a   = __riscv_vfmv_f_s_f32m1_f32(v_local_max);

                ದಶಮಾಂಶ scale_a     = max_abs_a / ((1 << 7) - 1);
                ದಶಮಾಂಶ rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
                scale_a_ptr[mi]   = scale_a;

                // Quantize and compute sums for each 16-element group
                ಗಾಗಿ (size_t bki = 0; bki < a_ಮೊತ್ತ_ಗಾತ್ರ; bki++) {
                    vfloat32m2_t v_a          = __riscv_vle32_v_f32m2(a_ptr + mi * ಎಣಿಕೆ_k + k + bki * 16, vl);
                    vfloat32m2_t v_a_scale    = __riscv_vfmul_vf_f32m2(v_a, rep_scale_a, vl);
                    vint16m1_t   v_a_quant    = __riscv_vfncvt_x_f_w_i16m1(v_a_scale, vl);
                    vint8mf2_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf2(v_a_quant, vl);

                    vint16m1_t tmp_ಮೊತ್ತ               = __riscv_vmv_v_x_i16m1(0, vl);
                    vint16m1_t v_a_ಮೊತ್ತ               = __riscv_vwredsum_vs_i8mf2_i16m1(v_a_quant_i8, tmp_ಮೊತ್ತ, vl);
                    int16_t    a_ಮೊತ್ತ                 = __riscv_vmv_x_s_i16m1_i16(v_a_ಮೊತ್ತ);
                    a_ಮೊತ್ತ_ptr[mi * a_ಮೊತ್ತ_ಗಾತ್ರ + bki] = -a_ಮೊತ್ತ;

                    __riscv_vse8_v_i8mf2(quant_a_blk + mi * blk_len + bki * 16, v_a_quant_i8, vl);
                }
            }
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        quantize_a_nrow_i8k_ref<4>(blk_len, a_ptr, ಎಣಿಕೆ_k, quant_a_ptr);
    }
}

ಶೂನ್ಯ forward_cpy_with_ಕ್ರಮಬದಲಾವಣೆ(ggml_compute_params * params, ggml_tensor * op) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ           ith  = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ           nth  = params->nth;

    // [batch, m, n] -> [batch, n, m]
    int64_t batch = src0->ne[2] * src0->ne[3];
    int64_t m     = src0->ne[1];
    int64_t n     = src0->ne[0];

    int64_t batch_stride = src0->nb[2];
    int64_t m_src_stride = src0->nb[0];
    int64_t n_src_stride = src0->nb[1];
    int64_t n_dst_stride = n_src_stride * m;

    ಕ್ರಮಬದಲಾವಣೆ_transpose_impl(src0, dst, batch, m, n, batch_stride, m_src_stride, n_src_stride, n_dst_stride, ith, nth);
}

ಶೂನ್ಯ forward_cont_with_ಕ್ರಮಬದಲಾವಣೆ(ggml_compute_params * params, ggml_tensor * op) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ           ith  = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ           nth  = params->nth;

    // [batch, m, n] -> [batch, n, m]
    int64_t batch = dst->ne[2] * dst->ne[3];
    int64_t n     = dst->ne[1];
    int64_t m     = dst->ne[0];

    int64_t batch_stride = dst->nb[2];
    int64_t m_src_stride = src0->nb[0];
    int64_t n_src_stride = src0->nb[1];
    int64_t n_dst_stride = dst->nb[1];

    ಕ್ರಮಬದಲಾವಣೆ_transpose_impl(src0, dst, batch, m, n, batch_stride, m_src_stride, n_src_stride, n_dst_stride, ith, nth);
}

ಶೂನ್ಯ forward_norm_f32(ggml_compute_params * params, ggml_tensor * op) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(src0->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಪೂರ್ಣಾಂಕ nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    ದಶಮಾಂಶ epsilon = *((ದಶಮಾಂಶ *) dst->op_params);

    GGML_ASSERT(epsilon > 0.0f);

    ಸ್ವಯಂಚಾಲಿತ * ಒಳಹರಿವು  = (ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ;
    ಸ್ವಯಂಚಾಲಿತ * output = (ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ;

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ hidden_ಗಾತ್ರ     = ne00;
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ task_ಎಣಿಕೆ      = ne01 * ne02 * ne03;
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ task_ಪ್ರತಿ_thread = (task_ಎಣಿಕೆ + nth - 1) / nth;

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ task_begin = ith * task_ಪ್ರತಿ_thread;
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ task_ಕೊನೆ   = std::min((ith + 1) * task_ಪ್ರತಿ_thread, task_ಎಣಿಕೆ);

    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ task_idx = task_begin; task_idx < task_ಕೊನೆ; task_idx++) {
        int64_t i03 = task_idx / (ne02 * ne01);
        int64_t i02 = (task_idx - i03 * ne02 * ne01) / ne01;
        int64_t i01 = (task_idx - i03 * ne02 * ne01 - i02 * ne01);

        ಸ್ವಯಂಚಾಲಿತ * p_ಒಳಹರಿವು       = (ದಶಮಾಂಶ *) (ಒಳಹರಿವು + i01 * nb01 + i02 * nb02 + i03 * nb03);
        ಸ್ವಯಂಚಾಲಿತ * p_output      = (ದಶಮಾಂಶ *) (output + i01 * nb1 + i02 * nb2 + i03 * nb3);
        ಸ್ವಯಂಚಾಲಿತ * p_ತಾತ್ಕಾಲಿಕ_output = p_output;

        size_t       gvl    = __riscv_vsetvlmax_e32m4();
        vfloat32m4_t ಮೊತ್ತ    = __riscv_vfmv_v_f_f32m4(0.f, gvl);
        vfloat32m4_t ಮೊತ್ತ_sq = __riscv_vfmv_v_f_f32m4(0.f, gvl);
        int64_t      ಉದ್ದಳತೆ = hidden_ಗಾತ್ರ;
        ಆಗ (ಉದ್ದಳತೆ > 0) {
            gvl                   = __riscv_vsetvl_e32m4(ಉದ್ದಳತೆ);
            // load data
            vfloat32m4_t src_ದತ್ತಾಂಶ = __riscv_vle32_v_f32m4(p_ಒಳಹರಿವು, gvl);

            ಮೊತ್ತ    = __riscv_vfadd_vv_f32m4(ಮೊತ್ತ, src_ದತ್ತಾಂಶ, gvl);
            ಮೊತ್ತ_sq = __riscv_vfmacc_vv_f32m4(ಮೊತ್ತ_sq, src_ದತ್ತಾಂಶ, src_ದತ್ತಾಂಶ, gvl);

            __riscv_vse32_v_f32m4(p_ತಾತ್ಕಾಲಿಕ_output, src_ದತ್ತಾಂಶ, gvl);

            p_ಒಳಹರಿವು += gvl;
            p_ತಾತ್ಕಾಲಿಕ_output += gvl;
            ಉದ್ದಳತೆ -= gvl;
        }

        gvl = __riscv_vsetvlmax_e32m1();

        ದಶಮಾಂಶ        ಸರಾಸರಿ_2   = 0.f;
        vfloat32m1_t zero_v = __riscv_vfmv_v_f_f32m1(0.f, gvl);
        vfloat32m1_t ಸರಾಸರಿ_v =
            __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(ಮೊತ್ತ, 0), __riscv_vget_v_f32m4_f32m1(ಮೊತ್ತ, 1), gvl);
        ಸರಾಸರಿ_v = __riscv_vfadd_vv_f32m1(ಸರಾಸರಿ_v, __riscv_vget_v_f32m4_f32m1(ಮೊತ್ತ, 2), gvl);
        ಸರಾಸರಿ_v = __riscv_vfadd_vv_f32m1(ಸರಾಸರಿ_v, __riscv_vget_v_f32m4_f32m1(ಮೊತ್ತ, 3), gvl);
        ಸರಾಸರಿ_v = __riscv_vfredusum_vs_f32m1_f32m1(ಸರಾಸರಿ_v, zero_v, gvl);
        ಸರಾಸರಿ_2   = __riscv_vfmv_f_s_f32m1_f32(ಸರಾಸರಿ_v);
        ಸರಾಸರಿ_2 /= hidden_ಗಾತ್ರ;

        vfloat32m1_t ಸರಾಸರಿ_ವರ್ಗಫಲ_v =
            __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(ಮೊತ್ತ_sq, 0), __riscv_vget_v_f32m4_f32m1(ಮೊತ್ತ_sq, 1), gvl);
        ಸರಾಸರಿ_ವರ್ಗಫಲ_v = __riscv_vfadd_vv_f32m1(ಸರಾಸರಿ_ವರ್ಗಫಲ_v, __riscv_vget_v_f32m4_f32m1(ಮೊತ್ತ_sq, 2), gvl);
        ಸರಾಸರಿ_ವರ್ಗಫಲ_v = __riscv_vfadd_vv_f32m1(ಸರಾಸರಿ_ವರ್ಗಫಲ_v, __riscv_vget_v_f32m4_f32m1(ಮೊತ್ತ_sq, 3), gvl);
        ಸರಾಸರಿ_ವರ್ಗಫಲ_v = __riscv_vfredusum_vs_f32m1_f32m1(ಸರಾಸರಿ_ವರ್ಗಫಲ_v, zero_v, gvl);

        ದಶಮಾಂಶ ಸರಾಸರಿ_ವರ್ಗಫಲ = __riscv_vfmv_f_s_f32m1_f32(ಸರಾಸರಿ_ವರ್ಗಫಲ_v);
        ಸರಾಸರಿ_ವರ್ಗಫಲ /= hidden_ಗಾತ್ರ;
        ಸರಾಸರಿ_ವರ್ಗಫಲ = ವರ್ಗಮೂಲ(ಸರಾಸರಿ_ವರ್ಗಫಲ - ಸರಾಸರಿ_2 * ಸರಾಸರಿ_2 + epsilon);

        ಸರಾಸರಿ_ವರ್ಗಫಲ   = 1.0f / ಸರಾಸರಿ_ವರ್ಗಫಲ;
        ಉದ್ದಳತೆ        = hidden_ಗಾತ್ರ;
        p_ತಾತ್ಕಾಲಿಕ_output = p_output;

        ಆಗ (ಉದ್ದಳತೆ > 0) {
            gvl                   = __riscv_vsetvl_e32m4(ಉದ್ದಳತೆ);
            vfloat32m4_t src_ದತ್ತಾಂಶ = __riscv_vle32_v_f32m4(p_ತಾತ್ಕಾಲಿಕ_output, gvl);
            src_ದತ್ತಾಂಶ              = __riscv_vfsub_vf_f32m4(src_ದತ್ತಾಂಶ, ಸರಾಸರಿ_2, gvl);
            src_ದತ್ತಾಂಶ              = __riscv_vfmul_vf_f32m4(src_ದತ್ತಾಂಶ, ಸರಾಸರಿ_ವರ್ಗಫಲ, gvl);
            __riscv_vse32_v_f32m4(p_output, src_ದತ್ತಾಂಶ, gvl);
            p_ತಾತ್ಕಾಲಿಕ_output += gvl;
            p_output += gvl;
            ಉದ್ದಳತೆ -= gvl;
        }
    }
}

ಮಾದರಿ <ggml_op op_type, ಬಗೆ_ಹೆಸರು T> ಶೂನ್ಯ forward_ದ್ವಿಮಾನ(ggml_compute_params * params, ggml_tensor * op) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = op->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = op->src[1];
    ggml_tensor *       dst  = op;
    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    ಸ್ವಯಂಚಾಲಿತ src0_ಅಡ್ಡಸಾಲುಗಳು = ggml_nrows(src0);
    ಸ್ವಯಂಚಾಲಿತ src1_ಅಡ್ಡಸಾಲುಗಳು = ggml_nrows(src1);

    ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಪೂರ್ಣಾಂಕ nth = params->nth;

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    GGML_ASSERT(nb0 == ಗಾತ್ರ(T));
    GGML_ASSERT(nb00 == ಗಾತ್ರ(T));

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ [ir0, ir1] = get_thread_range(params, src0);

    ಸ್ವಯಂಚಾಲಿತ compute_func_vv = [&](int64_t blk_len, int64_t r, T * src0_ptr, T * src1_ptr, T * dst_ptr) {
        int64_t idx = 0;
        ಆದರೆ constexpr (op_type == GGML_OP_ಕೂಡಿಸು) {
            ಆದರೆ constexpr (std::is_same_v<T, ದಶಮಾಂಶ>) {
                ಗಾಗಿ (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e32m4(blk_len);
                    vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + idx + r, vl);
                    vfloat32m4_t rhs = __riscv_vle32_v_f32m4(src1_ptr + idx, vl);
                    vfloat32m4_t res = __riscv_vfadd_vv_f32m4(lhs, rhs, vl);
                    __riscv_vse32_v_f32m4(dst_ptr + idx + r, res, vl);
                }
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<T, _Float16>) {
                ಗಾಗಿ (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e16m4(blk_len);
                    vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + idx + r), vl);
                    vfloat16m4_t rhs = __riscv_vle16_v_f16m4((src1_ptr + idx), vl);
                    vfloat16m4_t res = __riscv_vfadd_vv_f16m4(lhs, rhs, vl);
                    __riscv_vse16_v_f16m4((dst_ptr + idx + r), res, vl);
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                GGML_ABORT("fatal error");
            }
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (op_type == GGML_OP_SUB) {
            ಆದರೆ constexpr (std::is_same_v<T, ದಶಮಾಂಶ>) {
                ಗಾಗಿ (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e32m4(blk_len);
                    vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + idx + r, vl);
                    vfloat32m4_t rhs = __riscv_vle32_v_f32m4(src1_ptr + idx, vl);
                    vfloat32m4_t res = __riscv_vfsub_vv_f32m4(lhs, rhs, vl);
                    __riscv_vse32_v_f32m4(dst_ptr + idx + r, res, vl);
                }
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<T, _Float16>) {
                ಗಾಗಿ (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e16m4(blk_len);
                    vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + idx + r), vl);
                    vfloat16m4_t rhs = __riscv_vle16_v_f16m4((src1_ptr + idx), vl);
                    vfloat16m4_t res = __riscv_vfsub_vv_f16m4(lhs, rhs, vl);
                    __riscv_vse16_v_f16m4((dst_ptr + idx + r), res, vl);
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                GGML_ABORT("fatal error");
            }
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (op_type == GGML_OP_MUL) {
            ಆದರೆ constexpr (std::is_same_v<T, ದಶಮಾಂಶ>) {
                ಗಾಗಿ (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e32m4(blk_len);
                    vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + idx + r, vl);
                    vfloat32m4_t rhs = __riscv_vle32_v_f32m4(src1_ptr + idx, vl);
                    vfloat32m4_t res = __riscv_vfmul_vv_f32m4(lhs, rhs, vl);
                    __riscv_vse32_v_f32m4(dst_ptr + idx + r, res, vl);
                }
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<T, _Float16>) {
                ಗಾಗಿ (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e16m4(blk_len);
                    vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + idx + r), vl);
                    vfloat16m4_t rhs = __riscv_vle16_v_f16m4((src1_ptr + idx), vl);
                    vfloat16m4_t res = __riscv_vfmul_vv_f16m4(lhs, rhs, vl);
                    __riscv_vse16_v_f16m4((dst_ptr + idx + r), res, vl);
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                GGML_ABORT("fatal error");
            }
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (op_type == GGML_OP_DIV) {
            ಆದರೆ constexpr (std::is_same_v<T, ದಶಮಾಂಶ>) {
                ಗಾಗಿ (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e32m4(blk_len);
                    vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + idx + r, vl);
                    vfloat32m4_t rhs = __riscv_vle32_v_f32m4(src1_ptr + idx, vl);
                    vfloat32m4_t res = __riscv_vfdiv_vv_f32m4(lhs, rhs, vl);
                    __riscv_vse32_v_f32m4(dst_ptr + idx + r, res, vl);
                }
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<T, _Float16>) {
                ಗಾಗಿ (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e16m4(blk_len);
                    vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + idx + r), vl);
                    vfloat16m4_t rhs = __riscv_vle16_v_f16m4((src1_ptr + idx), vl);
                    vfloat16m4_t res = __riscv_vfdiv_vv_f16m4(lhs, rhs, vl);
                    __riscv_vse16_v_f16m4((dst_ptr + idx + r), res, vl);
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                GGML_ABORT("fatal error");
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_ABORT("fatal error");
        }
    };

    ಆದರೆ (src0_ಅಡ್ಡಸಾಲುಗಳು == src1_ಅಡ್ಡಸಾಲುಗಳು && src0_ಅಡ್ಡಸಾಲುಗಳು == 1 && ne00 == ne10) {
        int64_t task_ಪ್ರತಿ_thread = (ne00 + nth - 1) / nth;
        int64_t task_begin      = ith * task_ಪ್ರತಿ_thread;
        int64_t task_ಕೊನೆ        = std::min((ith + 1) * task_ಪ್ರತಿ_thread, ne00);

        T * dst_ptr  = ((T *) dst->ದತ್ತಾಂಶ) + task_begin;
        T * src0_ptr = ((T *) src0->ದತ್ತಾಂಶ) + task_begin;
        T * src1_ptr = ((T *) src1->ದತ್ತಾಂಶ) + task_begin;

        compute_func_vv(task_ಕೊನೆ - task_begin, 0, src0_ptr, src1_ptr, dst_ptr);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ne10 > 1) {
        ಗಾಗಿ (int64_t ir = ir0; ir < ir1; ++ir) {
            ಸ್ಥಿರಾಂಕ int64_t i03 = ir / (ne02 * ne01);
            ಸ್ಥಿರಾಂಕ int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            ಸ್ಥಿರಾಂಕ int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

            ಸ್ಥಿರಾಂಕ int64_t i13 = i03 % ne13;
            ಸ್ಥಿರಾಂಕ int64_t i12 = i02 % ne12;
            ಸ್ಥಿರಾಂಕ int64_t i11 = i01 % ne11;

            T * dst_ptr  = (T *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i03 * nb3 + i02 * nb2 + i01 * nb1);
            T * src0_ptr = (T *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i03 * nb03 + i02 * nb02 + i01 * nb01);
            T * src1_ptr = (T *) ((ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + i13 * nb13 + i12 * nb12 + i11 * nb11);

            // src1 is broadcastable across src0 and dst in i1, i2, i3
            ಗಾಗಿ (int64_t r = 0; r < ne00; r += ne10) {
                compute_func_vv(ne10, r, src0_ptr, src1_ptr, dst_ptr);
            }
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಗಾಗಿ (int64_t ir = ir0; ir < ir1; ++ir) {
            ಸ್ಥಿರಾಂಕ int64_t i03 = ir / (ne02 * ne01);
            ಸ್ಥಿರಾಂಕ int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            ಸ್ಥಿರಾಂಕ int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

            ಸ್ಥಿರಾಂಕ int64_t i13 = i03 % ne13;
            ಸ್ಥಿರಾಂಕ int64_t i12 = i02 % ne12;
            ಸ್ಥಿರಾಂಕ int64_t i11 = i01 % ne11;

            T * dst_ptr  = (T *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i03 * nb3 + i02 * nb2 + i01 * nb1);
            T * src0_ptr = (T *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i03 * nb03 + i02 * nb02 + i01 * nb01);
            T * src1_ptr = (T *) ((ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + i13 * nb13 + i12 * nb12 + i11 * nb11);

            T       rhs_scalar = src1_ptr[0];
            int64_t blk_len    = ne00;
            int64_t r          = 0;

            ಗಾಗಿ (size_t vl; blk_len > 0; blk_len -= vl, r += vl) {
                ಆದರೆ constexpr (op_type == GGML_OP_ಕೂಡಿಸು) {
                    ಆದರೆ constexpr (std::is_same_v<T, ದಶಮಾಂಶ>) {
                        vl               = __riscv_vsetvl_e32m4(blk_len);
                        vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + r, vl);
                        vfloat32m4_t res = __riscv_vfadd_vf_f32m4(lhs, rhs_scalar, vl);
                        __riscv_vse32_v_f32m4(dst_ptr + r, res, vl);
                    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<T, _Float16>) {
                        vl               = __riscv_vsetvl_e16m4(blk_len);
                        vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + r), vl);
                        vfloat16m4_t res = __riscv_vfadd_vf_f16m4(lhs, rhs_scalar, vl);
                        __riscv_vse16_v_f16m4((dst_ptr + r), res, vl);
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        GGML_ABORT("fatal error");
                    }
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (op_type == GGML_OP_SUB) {
                    ಆದರೆ constexpr (std::is_same_v<T, ದಶಮಾಂಶ>) {
                        vl               = __riscv_vsetvl_e32m4(blk_len);
                        vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + r, vl);
                        vfloat32m4_t res = __riscv_vfsub_vf_f32m4(lhs, rhs_scalar, vl);
                        __riscv_vse32_v_f32m4(dst_ptr + r, res, vl);
                    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<T, _Float16>) {
                        vl               = __riscv_vsetvl_e16m4(blk_len);
                        vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + r), vl);
                        vfloat16m4_t res = __riscv_vfsub_vf_f16m4(lhs, rhs_scalar, vl);
                        __riscv_vse16_v_f16m4((dst_ptr + r), res, vl);
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        GGML_ABORT("fatal error");
                    }
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (op_type == GGML_OP_MUL) {
                    ಆದರೆ constexpr (std::is_same_v<T, ದಶಮಾಂಶ>) {
                        vl               = __riscv_vsetvl_e32m4(blk_len);
                        vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + r, vl);
                        vfloat32m4_t res = __riscv_vfmul_vf_f32m4(lhs, rhs_scalar, vl);
                        __riscv_vse32_v_f32m4(dst_ptr + r, res, vl);
                    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<T, _Float16>) {
                        vl               = __riscv_vsetvl_e16m4(blk_len);
                        vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + r), vl);
                        vfloat16m4_t res = __riscv_vfmul_vf_f16m4(lhs, rhs_scalar, vl);
                        __riscv_vse16_v_f16m4((dst_ptr + r), res, vl);
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        GGML_ABORT("fatal error");
                    }
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (op_type == GGML_OP_DIV) {
                    ಆದರೆ constexpr (std::is_same_v<T, ದಶಮಾಂಶ>) {
                        vl               = __riscv_vsetvl_e32m4(blk_len);
                        vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + r, vl);
                        vfloat32m4_t res = __riscv_vfdiv_vf_f32m4(lhs, rhs_scalar, vl);
                        __riscv_vse32_v_f32m4(dst_ptr + r, res, vl);
                    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<T, _Float16>) {
                        vl               = __riscv_vsetvl_e16m4(blk_len);
                        vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + r), vl);
                        vfloat16m4_t res = __riscv_vfdiv_vf_f16m4(lhs, rhs_scalar, vl);
                        __riscv_vse16_v_f16m4((dst_ptr + r), res, vl);
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        GGML_ABORT("fatal error");
                    }
                } ಇಲ್ಲದಿದ್ದರೆ {
                    GGML_ABORT("fatal error");
                }
            }
        }
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T> ಶೂನ್ಯ forward_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * op) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne0 == 1);
    GGML_ASSERT(ne1 == ne01);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    int64_t n_task          = ne01 * ne02 * ne03;
    int64_t task_ಪ್ರತಿ_thread = (n_task + nth - 1) / nth;
    int64_t ir_ಆರಂಭ        = ith * task_ಪ್ರತಿ_thread;
    int64_t ir_ಕೊನೆ          = std::min(ir_ಆರಂಭ + task_ಪ್ರತಿ_thread, n_task);

    ಗಾಗಿ (int64_t ir = ir_ಆರಂಭ; ir < ir_ಕೊನೆ; ir++) {
        ಸ್ಥಿರಾಂಕ int64_t i3 = ir / (ne02 * ne01);
        ಸ್ಥಿರಾಂಕ int64_t i2 = (ir - i3 * ne02 * ne01) / ne01;
        ಸ್ಥಿರಾಂಕ int64_t i1 = (ir - i3 * ne02 * ne01 - i2 * ne01);

        T * src_ಅಡ್ಡಸಾಲು = (T *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i1 * nb01 + i2 * nb02 + i3 * nb03);
        T * dst_ಅಡ್ಡಸಾಲು = (T *) ((ಅಕ್ಷರ *) op->ದತ್ತಾಂಶ + i1 * nb1 + i2 * nb2 + i3 * nb3);

        ದಶಮಾಂಶ ಅಡ್ಡಸಾಲು_ಮೊತ್ತ = 0;

        ಆದರೆ constexpr (std::is_same_v<T, ದಶಮಾಂಶ>) {
            size_t        gvl     = __riscv_vsetvlmax_e32m4();
            vfloat32m4_t  acc_vec = __riscv_vfmv_v_f_f32m4(0.0f, gvl);
            int64_t       ಉದ್ದಳತೆ  = ne00;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * p_ದತ್ತಾಂಶ  = src_ಅಡ್ಡಸಾಲು;

            ಆಗ (ಉದ್ದಳತೆ > 0) {
                size_t       vl  = __riscv_vsetvl_e32m4(ಉದ್ದಳತೆ);
                vfloat32m4_t vec = __riscv_vle32_v_f32m4(p_ದತ್ತಾಂಶ, vl);
                acc_vec          = __riscv_vfadd_vv_f32m4(acc_vec, vec, vl);
                p_ದತ್ತಾಂಶ += vl;
                ಉದ್ದಳತೆ -= vl;
            }

            gvl                 = __riscv_vsetvlmax_e32m1();
            vfloat32m1_t zero_v = __riscv_vfmv_v_f_f32m1(0.0f, gvl);
            vfloat32m1_t ಮೊತ್ತ_v  = __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(acc_vec, 0),
                                                         __riscv_vget_v_f32m4_f32m1(acc_vec, 1), gvl);
            ಮೊತ್ತ_v               = __riscv_vfadd_vv_f32m1(ಮೊತ್ತ_v, __riscv_vget_v_f32m4_f32m1(acc_vec, 2), gvl);
            ಮೊತ್ತ_v               = __riscv_vfadd_vv_f32m1(ಮೊತ್ತ_v, __riscv_vget_v_f32m4_f32m1(acc_vec, 3), gvl);
            ಮೊತ್ತ_v               = __riscv_vfredusum_vs_f32m1_f32m1(ಮೊತ್ತ_v, zero_v, gvl);
            ಅಡ್ಡಸಾಲು_ಮೊತ್ತ             = __riscv_vfmv_f_s_f32m1_f32(ಮೊತ್ತ_v);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<T, _Float16>) {
            size_t           gvl     = __riscv_vsetvlmax_e16m2();
            vfloat32m4_t     acc_vec = __riscv_vfmv_v_f_f32m4(0.0f, gvl);
            int64_t          ಉದ್ದಳತೆ  = ne00;
            ಸ್ಥಿರಾಂಕ _Float16 * p_ದತ್ತಾಂಶ  = src_ಅಡ್ಡಸಾಲು;

            ಆಗ (ಉದ್ದಳತೆ > 0) {
                size_t       vl      = __riscv_vsetvl_e16m2(ಉದ್ದಳತೆ);
                vfloat16m2_t vec_f16 = __riscv_vle16_v_f16m2(p_ದತ್ತಾಂಶ, vl);
                vfloat32m4_t vec_f32 = __riscv_vfwcvt_f_f_v_f32m4(vec_f16, vl);
                acc_vec              = __riscv_vfadd_vv_f32m4(acc_vec, vec_f32, vl);
                p_ದತ್ತಾಂಶ += vl;
                ಉದ್ದಳತೆ -= vl;
            }

            gvl                 = __riscv_vsetvlmax_e32m1();
            vfloat32m1_t zero_v = __riscv_vfmv_v_f_f32m1(0.0f, gvl);
            vfloat32m1_t ಮೊತ್ತ_v  = __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(acc_vec, 0),
                                                         __riscv_vget_v_f32m4_f32m1(acc_vec, 1), gvl);
            ಮೊತ್ತ_v               = __riscv_vfadd_vv_f32m1(ಮೊತ್ತ_v, __riscv_vget_v_f32m4_f32m1(acc_vec, 2), gvl);
            ಮೊತ್ತ_v               = __riscv_vfadd_vv_f32m1(ಮೊತ್ತ_v, __riscv_vget_v_f32m4_f32m1(acc_vec, 3), gvl);
            ಮೊತ್ತ_v               = __riscv_vfredusum_vs_f32m1_f32m1(ಮೊತ್ತ_v, zero_v, gvl);
            ಅಡ್ಡಸಾಲು_ಮೊತ್ತ             = __riscv_vfmv_f_s_f32m1_f32(ಮೊತ್ತ_v);
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_ABORT("fatal error");
        }

        dst_ಅಡ್ಡಸಾಲು[0] = ಅಡ್ಡಸಾಲು_ಮೊತ್ತ;
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T> ಶೂನ್ಯ forward_repeat_nrows(ggml_compute_params * params, ggml_tensor * op) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    int64_t nrows            = ggml_nrows(src0);
    int64_t nrows_ಪ್ರತಿ_thread = (nrows + nth - 1) / nth;
    int64_t ir_ಆರಂಭ         = ith * nrows_ಪ್ರತಿ_thread;
    int64_t ir_ಕೊನೆ           = std::min(ir_ಆರಂಭ + nrows_ಪ್ರತಿ_thread, nrows);

    ಆದರೆ (src0->ne[0] == 1) {
        ಗಾಗಿ (int64_t ir = ir_ಆರಂಭ; ir < ir_ಕೊನೆ; ir++) {
            T * src_ಅಡ್ಡಸಾಲು = (T *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + ir * src0->nb[1]);
            T * dst_ಅಡ್ಡಸಾಲು = (T *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + ir * dst->nb[1]);

            T src_scalar = src_ಅಡ್ಡಸಾಲು[0];

            int64_t ಉದ್ದಳತೆ = dst->ne[0];
            int64_t idx    = 0;
            size_t  vl     = 0;

            ಆಗ (ಉದ್ದಳತೆ > 0) {
                ಆದರೆ constexpr (std::is_same_v<T, int32_t>) {
                    vl             = __riscv_vsetvl_e32m4(ಉದ್ದಳತೆ);
                    vint32m4_t vec = __riscv_vmv_v_x_i32m4(src_scalar, vl);
                    __riscv_vse32_v_i32m4(dst_ಅಡ್ಡಸಾಲು + idx, vec, vl);
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<T, int16_t>) {
                    vl             = __riscv_vsetvl_e16m4(ಉದ್ದಳತೆ);
                    vint16m4_t vec = __riscv_vmv_v_x_i16m4(src_scalar, vl);
                    __riscv_vse16_v_i16m4((dst_ಅಡ್ಡಸಾಲು + idx), vec, vl);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    GGML_ABORT("fatal error");
                }
                idx += vl;
                ಉದ್ದಳತೆ -= vl;
            }
        }
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->ne[0] == dst->ne[0]) {
        ಗಾಗಿ (int64_t ir = ir_ಆರಂಭ; ir < ir_ಕೊನೆ; ir++) {
            T * src_ಅಡ್ಡಸಾಲು = (T *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + ir * src0->nb[1]);
            T * dst_ಅಡ್ಡಸಾಲು = (T *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + ir * dst->nb[1]);

            int64_t ಉದ್ದಳತೆ = dst->ne[0];
            int64_t idx    = 0;
            size_t  vl     = 0;

            ಆಗ (ಉದ್ದಳತೆ > 0) {
                ಆದರೆ constexpr (std::is_same_v<T, int32_t>) {
                    vl             = __riscv_vsetvl_e32m4(ಉದ್ದಳತೆ);
                    vint32m4_t vec = __riscv_vle32_v_i32m4(src_ಅಡ್ಡಸಾಲು + idx, vl);
                    __riscv_vse32_v_i32m4(dst_ಅಡ್ಡಸಾಲು + idx, vec, vl);
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<T, int16_t>) {
                    vl             = __riscv_vsetvl_e16m4(ಉದ್ದಳತೆ);
                    vint16m4_t vec = __riscv_vle16_v_i16m4((src_ಅಡ್ಡಸಾಲು + idx), vl);
                    __riscv_vse16_v_i16m4((dst_ಅಡ್ಡಸಾಲು + idx), vec, vl);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    GGML_ABORT("fatal error");
                }
                idx += vl;
                ಉದ್ದಳತೆ -= vl;
            }
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_ABORT("fatal error");
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T> ಶೂನ್ಯ forward_repeat_dim1(ggml_compute_params * params, ggml_tensor * op) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ int64_t ne0 = dst->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ne1 = dst->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t ne2 = dst->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t ne3 = dst->ne[3];

    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_batches      = ne2 * ne3;
    ಸ್ಥಿರಾಂಕ int64_t batches_ಪ್ರತಿ_thread = (ಒಟ್ಟು_batches + nth - 1) / nth;
    ಸ್ಥಿರಾಂಕ int64_t batch_ಆರಂಭ        = ith * batches_ಪ್ರತಿ_thread;
    ಸ್ಥಿರಾಂಕ int64_t batch_ಕೊನೆ          = std::min(batch_ಆರಂಭ + batches_ಪ್ರತಿ_thread, ಒಟ್ಟು_batches);

    ಗಾಗಿ (int64_t b = batch_ಆರಂಭ; b < batch_ಕೊನೆ; b++) {
        ಸ್ಥಿರಾಂಕ int64_t i3 = b / ne2;
        ಸ್ಥಿರಾಂಕ int64_t i2 = b % ne2;

        T * src_ತಳ  = (T *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i2 * src0->nb[2] + i3 * src0->nb[3]);
        T * dst_batch = (T *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i2 * dst->nb[2] + i3 * dst->nb[3]);

        ಗಾಗಿ (int64_t i1 = 0; i1 < ne1; i1++) {
            T *     dst_ptr = (T *) ((ಅಕ್ಷರ *) dst_batch + i1 * dst->nb[1]);
            int64_t ಉದ್ದಳತೆ  = ne0;
            int64_t idx     = 0;

            ಆಗ (ಉದ್ದಳತೆ > 0) {
                ಆದರೆ constexpr (std::is_same_v<T, int32_t>) {
                    size_t     vl  = __riscv_vsetvl_e32m4(ಉದ್ದಳತೆ);
                    vint32m4_t vec = __riscv_vle32_v_i32m4(src_ತಳ + idx, vl);
                    __riscv_vse32_v_i32m4(dst_ptr + idx, vec, vl);
                    idx += vl;
                    ಉದ್ದಳತೆ -= vl;
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<T, int16_t>) {
                    size_t     vl  = __riscv_vsetvl_e16m4(ಉದ್ದಳತೆ);
                    vint16m4_t vec = __riscv_vle16_v_i16m4((src_ತಳ + idx), vl);
                    __riscv_vse16_v_i16m4((dst_ptr + idx), vec, vl);
                    idx += vl;
                    ಉದ್ದಳತೆ -= vl;
                } ಇಲ್ಲದಿದ್ದರೆ {
                    GGML_ABORT("fatal error");
                }
            }
        }
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T> ಶೂನ್ಯ forward_get_ಅಡ್ಡಸಾಲುಗಳು(ggml_compute_params * params, ggml_tensor * op) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = op->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = op->src[1];
    ggml_tensor *       dst  = op;

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    ಸ್ಥಿರಾಂಕ int64_t nc = ne00;
    ಸ್ಥಿರಾಂಕ int64_t nr = ggml_nelements(src1);

    ಪ್ರತಿಜ್ಞಾ(ne0 == nc);
    ಪ್ರತಿಜ್ಞಾ(ne02 == ne11);
    ಪ್ರತಿಜ್ಞಾ(nb00 == ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಪ್ರತಿಜ್ಞಾ(ggml_nrows(op) == nr);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲುಗಳು_nth = nth;
    ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_nth = 1;

    ಆದರೆ (nr == 1) {
        ಅಡ್ಡಸಾಲುಗಳು_nth = 1;
        ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_nth = nth;
    }

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + ಅಡ್ಡಸಾಲುಗಳು_nth - 1) / ಅಡ್ಡಸಾಲುಗಳು_nth;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dc = (nc + ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_nth - 1) / ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_nth;

    ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲುಗಳು_ith = ith % ಅಡ್ಡಸಾಲುಗಳು_nth;
    ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ith = ith % ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr * ಅಡ್ಡಸಾಲುಗಳು_ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ cr0 = dc * ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ cr1 = MIN(cr0 + dc, nc);

    ಗಾಗಿ (int64_t i = ir0; i < ir1; ++i) {
        ಸ್ಥಿರಾಂಕ int64_t i12 = i / (ne11 * ne10);
        ಸ್ಥಿರಾಂಕ int64_t i11 = (i - i12 * ne11 * ne10) / ne10;
        ಸ್ಥಿರಾಂಕ int64_t i10 = (i - i12 * ne11 * ne10 - i11 * ne10);
        ಸ್ಥಿರಾಂಕ int64_t i01 = *(int32_t *) ((ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + i10 * nb10 + i11 * nb11 + i12 * nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        memcpy1d(((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i10 * nb1 + i11 * nb2 + i12 * nb3) + cr0 * ಗಾತ್ರ(T),
                 ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i01 * nb01 + i11 * nb02 + i12 * nb03) + cr0 * ಗಾತ್ರ(T),
                 (cr1 - cr0) * ಗಾತ್ರ(T));
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T> ಶೂನ್ಯ forward_concat(ggml_compute_params * params, ggml_tensor * op) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = op->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = op->src[1];
    ggml_tensor *       dst  = op;

    GGML_ASSERT(ggml_type_ಗಾತ್ರ(src0->type) == ಗಾತ್ರ(ದಶಮಾಂಶ));

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    ಸ್ಥಿರಾಂಕ int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim == 0 && nb0 == ಗಾತ್ರ(ದಶಮಾಂಶ) && nb1 == ಗಾತ್ರ(ದಶಮಾಂಶ) * (ne00 + ne10));

    ಸ್ಥಿರಾಂಕ int64_t nr = ggml_nrows(dst);
    ಸ್ಥಿರಾಂಕ int64_t nc = ne0;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲುಗಳು_nth = nth;
    ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_nth = 1;

    ಆದರೆ (nr == 1) {
        ಅಡ್ಡಸಾಲುಗಳು_nth = 1;
        ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_nth = nth;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + ಅಡ್ಡಸಾಲುಗಳು_nth - 1) / ಅಡ್ಡಸಾಲುಗಳು_nth;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dc = (nc + ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_nth - 1) / ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_nth;

    ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲುಗಳು_ith = ith % ಅಡ್ಡಸಾಲುಗಳು_nth;
    ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ith = ith % ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr * ಅಡ್ಡಸಾಲುಗಳು_ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ cr0 = dc * ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ cr1 = MIN(cr0 + dc, nc);

    int64_t o[4] = { 0, 0, 0, 0 };
    o[dim]       = src0->ne[dim];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x;

    ಗಾಗಿ (int64_t i = ir0; i < ir1; ++i) {
        ಸ್ಥಿರಾಂಕ int64_t i3 = i / (ne02 * ne01);
        ಸ್ಥಿರಾಂಕ int64_t i2 = (i - i3 * ne02 * ne01) / ne01;
        ಸ್ಥಿರಾಂಕ int64_t i1 = (i - i3 * ne02 * ne01 - i2 * ne01);

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = cr0; i0 < cr1; i0++) {
            ಆದರೆ (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                x = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + (i0) *nb00 + (i1) *nb01 + (i2) *nb02 + (i3) *nb03);
            } ಇಲ್ಲದಿದ್ದರೆ {
                x = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + (i0 - o[0]) * nb10 + (i1 - o[1]) * nb11 +
                                     (i2 - o[2]) * nb12 + (i3 - o[3]) * nb13);
            }

            ದಶಮಾಂಶ * y = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i0 * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3);

            *y = *x;
        }
    }
}

ಮಾದರಿ ಶೂನ್ಯ forward_ದ್ವಿಮಾನ<GGML_OP_ಕೂಡಿಸು, ದಶಮಾಂಶ>(ggml_compute_params * params, ggml_tensor * op);
ಮಾದರಿ ಶೂನ್ಯ forward_ದ್ವಿಮಾನ<GGML_OP_SUB, ದಶಮಾಂಶ>(ggml_compute_params * params, ggml_tensor * op);
ಮಾದರಿ ಶೂನ್ಯ forward_ದ್ವಿಮಾನ<GGML_OP_MUL, ದಶಮಾಂಶ>(ggml_compute_params * params, ggml_tensor * op);
ಮಾದರಿ ಶೂನ್ಯ forward_ದ್ವಿಮಾನ<GGML_OP_DIV, ದಶಮಾಂಶ>(ggml_compute_params * params, ggml_tensor * op);
ಮಾದರಿ ಶೂನ್ಯ forward_ದ್ವಿಮಾನ<GGML_OP_ಕೂಡಿಸು, _Float16>(ggml_compute_params * params, ggml_tensor * op);
ಮಾದರಿ ಶೂನ್ಯ forward_ದ್ವಿಮಾನ<GGML_OP_SUB, _Float16>(ggml_compute_params * params, ggml_tensor * op);
ಮಾದರಿ ಶೂನ್ಯ forward_ದ್ವಿಮಾನ<GGML_OP_MUL, _Float16>(ggml_compute_params * params, ggml_tensor * op);
ಮಾದರಿ ಶೂನ್ಯ forward_ದ್ವಿಮಾನ<GGML_OP_DIV, _Float16>(ggml_compute_params * params, ggml_tensor * op);
ಮಾದರಿ ಶೂನ್ಯ forward_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು<ದಶಮಾಂಶ>(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * op);
ಮಾದರಿ ಶೂನ್ಯ forward_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು<_Float16>(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * op);
ಮಾದರಿ ಶೂನ್ಯ forward_repeat_nrows<int32_t>(ggml_compute_params * params, ggml_tensor * op);
ಮಾದರಿ ಶೂನ್ಯ forward_repeat_nrows<int16_t>(ggml_compute_params * params, ggml_tensor * op);
ಮಾದರಿ ಶೂನ್ಯ forward_repeat_dim1<int32_t>(ggml_compute_params * params, ggml_tensor * op);
ಮಾದರಿ ಶೂನ್ಯ forward_repeat_dim1<int16_t>(ggml_compute_params * params, ggml_tensor * op);
ಮಾದರಿ ಶೂನ್ಯ forward_get_ಅಡ್ಡಸಾಲುಗಳು<int32_t>(ggml_compute_params * params, ggml_tensor * op);
ಮಾದರಿ ಶೂನ್ಯ forward_get_ಅಡ್ಡಸಾಲುಗಳು<int16_t>(ggml_compute_params * params, ggml_tensor * op);
ಮಾದರಿ ಶೂನ್ಯ forward_concat<int32_t>(ggml_compute_params * params, ggml_tensor * op);
ಮಾದರಿ ಶೂನ್ಯ forward_concat<int16_t>(ggml_compute_params * params, ggml_tensor * op);

}  // namespace spacemit_kernels::rvv
